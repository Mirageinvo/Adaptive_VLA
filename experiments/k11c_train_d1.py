"""K-11c. Этап D1: обучение головы поправки на кэшированном позднем состоянии.

ЧТО ОБУЧАЕТСЯ. Только `hicora_head.proj.*` и `hicora_head.net.*` — 738400
параметров. Ствол, голова черновика, `bos`, `res_norm` и кодек заморожены;
базис и rho — буферы. Это проверено в K-11b и проверяется здесь заново
отказом, а не предупреждением.

ПРОХОД VLM НЕ НУЖЕН. `h24` и `q0` уже собраны в кэш K-11a, поэтому вперёд
идёт лишь голова и замороженный декодер. Эпоха занимает минуты, и перебор
двух сидов, двух скоростей и двух мишеней выполним целиком, а не выборочно.

ДВЕ ОПОРЫ, И ОНИ НЕ РАВНОЦЕННЫ.
  - `D(z*)` — полное трёхуровневое восстановление. Относительно него оракул
    ранга 32 (90.4% в K-11a) действительно ВЕРХНЯЯ ГРАНИЦА.
  - `A*` — истинные действия из датасета. Относительно них тот же оракул
    потолком НЕ является: поправка `dz` непрерывна и в решётку кодов
    попадать не обязана, поэтому `D(z0+dz)` в принципе может оказаться
    ближе к `A*`, чем `D(z*)`.
Обе доли печатаются рядом и помечаются, иначе «превысили оракул» звучало бы
как ошибка измерения, а не как свойство непрерывной поправки.

ЭПОХА 0 ВХОДИТ В ЗАМЕРЫ И ПРОВЕРЯЕТСЯ ОТКАЗОМ. При нулевой инициализации
последнего слоя доля улучшения обязана быть РОВНО нулём, а действия —
совпадать с черновиком. Если это не так, обучение сравнивается не с той
опорой, и любые дальнейшие числа бессмысленны.

ОТБОР ПРЕ-РЕГИСТРИРОВАН И ИДЁТ ПО СРЕДНЕМУ ПО СИДАМ, а не по лучшему сиду:
выбор лучшего сида — это выбор шума. Мишень и скорость обучения —
параметры, отбираемые на val; test не открывается.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k11a_build_hicora_cache as k11a  # noqa: E402
import k11b_hicora_identity as k11b  # noqa: E402
import k11p_residual_probe as k11p  # noqa: E402

N_POS = k11a.N_POS
H_EXEC = k11a.H_EXEC
N_STAT = k11p.N_STAT
# Арены оценки. `draft` и `oracle` — опоры, остальное обучается.
ARMS = ("draft", "oracle", "head")
ACT_TOL = k11b.ACT_TOL
TRAIN_PREFIXES = ("proj.", "net.", "lin.")
GRIP_TOL = 0.005
SEL_SEEDS_MIN = 2


def make_linear_head():
    """Линейная голова с тем же контрактом, что у настоящей.

    ЗАЧЕМ ОНА. Линейный зонд возвращает 10.5%, обученная MLP — 7.4%. Причин
    может быть четыре: архитектура, потеря, оптимизация или переобучение.
    Матрица «линейная или MLP» на «мишень-коэффициенты или мишень-действия»
    их разделяет: если линейная голова с мишенью-коэффициентами
    воспроизводит зонд, конвейер обучения исправен, и виновата не он.

    Класс определён ЗДЕСЬ, а не в `hicora_vla`: тот файл заморожен, его sha
    записана в кэш и сверяется при каждом запуске. Диагностическая ветвь не
    имеет права менять исполняемую модель.
    """
    import torch
    import torch.nn as nn

    class LinearHead(nn.Module):
        def __init__(self, d_hidden, d_latent, rank=32, squash="tanh"):
            super().__init__()
            # СПОСОБ ОГРАНИЧЕНИЯ — ПАРАМЕТР. Зонд ограничивает коэффициенты
            # ОБРЕЗАНИЕМ, голова — `tanh`. Это РАЗНЫЕ модели, и чтобы
            # проверить конвейер обучения на известном ответе, контрольная
            # голова обязана уметь ровно то же, что зонд.
            if squash not in ("tanh", "clip"):
                raise ValueError(f"неизвестное ограничение {squash}")
            self.squash = squash
            self.rank, self.d_latent = int(rank), int(d_latent)
            self.lin = nn.Linear(d_hidden + d_latent, self.rank)
            # НУЛЕВАЯ ИНИЦИАЛИЗАЦИЯ, как у настоящей головы: без неё эпоха 0
            # не была бы тождественна черновику и опора уехала бы.
            nn.init.zeros_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)
            self.register_buffer("basis", torch.zeros(d_latent, self.rank))
            self.register_buffer("rho", torch.ones(self.rank))

        def set_basis(self, B):
            b = torch.as_tensor(B).float()
            dev_ = float((b.T @ b - torch.eye(b.shape[1])).abs().max())
            if dev_ > 1e-4:
                raise ValueError(f"базис не ортонормален: {dev_:.2e}")
            self.basis.copy_(b.to(self.basis.device))
            return self

        def set_rho(self, r):
            r = torch.as_tensor(r).float()
            if not torch.isfinite(r).all() or bool((r <= 0).any()):
                raise ValueError("rho обязана быть конечной и положительной")
            self.rho.copy_(r.to(self.rho.device))
            return self

        def coeffs(self, h, z0):
            # `z0` под stop-gradient, как в настоящей голове: поправка
            # строится К черновику, а не меняет его.
            x = self.lin(torch.cat([h, z0.detach()], -1))
            return torch.tanh(x) if self.squash == "tanh" \
                else torch.clamp(x, -1.0, 1.0)

        def forward(self, h, z0):
            c = self.coeffs(h, z0)
            return (self.rho * c) @ self.basis.T, c

    return LinearHead


def loss_by_channel(pred, target, h_exec=H_EXEC):
    """Вклад каждой группы каналов в ту же потерю действий.

    ЗАЧЕМ. Потеря усредняется по СЕМИ каналам в АБСОЛЮТНОМ масштабе. Схват
    принимает значения около +-1, а положение и вращение имеют RMS порядка
    0.12: в квадратичном режиме это разница примерно в сто раз. Если потеря
    почти целиком состоит из схвата, то на позу градиента почти не остаётся —
    а отчётная метрика нормирует ошибку по каждому каналу отдельно и схват
    выносит в отдельный гейт. Тогда «обучаем на действиях» означает на деле
    «обучаем на схвате», и расхождение обучающей и отчётной метрик
    объясняется без всякой мистики.

    Возвращает сумму вкладов, равную общей потере: доли складываются в 1.
    """
    import torch
    p_ = pred[:, :h_exec]
    t_ = target[:, :h_exec]
    out, tot = {}, 0.0
    for nm, sl in (("pos", slice(0, 3)), ("rot", slice(3, 6)),
                   ("grip", slice(6, 7))):
        # ВКЛАД, А НЕ СРЕДНЕЕ ПО ГРУППЕ: домножаем на долю каналов, чтобы
        # сумма трёх величин равнялась общей потере.
        v = float(torch.nn.functional.smooth_l1_loss(
            p_[..., sl], t_[..., sl])) * (sl.stop - sl.start) / 7.0
        out[nm] = v
        tot += v
    out["total"] = tot
    for nm in ("pos", "rot", "grip"):
        out[nm + "_frac"] = out[nm] / tot if tot > 0 else None
    return out


def loss_terms(pred, target, h_exec=H_EXEC, weights=None):
    """Потеря в пространстве действий на ИСПОЛНЯЕМЫХ шагах 0..h_exec-1.

    Ось времени — шаги чанка, а не латентные позиции Perceiver. Хвост чанка
    не исполняется при частоте вызовов, измеренной в K-9.

    ВЕСА КАНАЛОВ. Равномерное усреднение по семи каналам ИЗМЕРЕНО как
    источник расхождения: схват занимает 41-52% потери, тогда как в отчётном
    числе он вынесен в отдельный гейт и в долю по позе не входит. Голова,
    обученная на такой потере, разменивает точность позы на точность схвата —
    в обеих парах её позиционная составляющая на val ВЫШЕ, чем у головы,
    обученной на коэффициентах.

    `weights` — три множителя (положение, вращение, схват). `None` означает
    прежнее равномерное поведение и оставлено, чтобы старые прогоны
    воспроизводились ровно.
    """
    import torch
    p = pred[:, :h_exec]
    t = target[:, :h_exec]
    if weights is None:
        return torch.nn.functional.smooth_l1_loss(p, t)
    wp, wr, wg = (float(x) for x in weights)
    f = torch.nn.functional.smooth_l1_loss
    # Доли каналов сохранены (3/7, 3/7, 1/7), чтобы при весах (1,1,1)
    # величина совпадала с равномерной потерей до последнего знака.
    return (wp * 3.0 / 7.0 * f(p[..., 0:3], t[..., 0:3])
            + wr * 3.0 / 7.0 * f(p[..., 3:6], t[..., 3:6])
            + wg * 1.0 / 7.0 * f(p[..., 6:7], t[..., 6:7]))


def epoch0_ok(gain_pos, gain_rot, act_max_abs, tol=ACT_TOL, eps=1e-9):
    """Нулевая эпоха обязана быть тождественна черновику.

    Это не формальность. Если при нулевой голове доля отлична от нуля,
    значит опора посчитана другим кодом, чем ветка головы, и весь прирост
    может оказаться разницей конвейеров, а не обучением.
    """
    return (abs(float(gain_pos)) <= eps and abs(float(gain_rot)) <= eps
            and float(act_max_abs) <= tol)


def trainable_report(head):
    """Обучаемые тензоры и их число параметров. Всё вне списка — отказ."""
    tr, extra, n_par = [], [], 0
    for n, p in head.named_parameters():
        if p.requires_grad:
            if not n.startswith(TRAIN_PREFIXES):
                extra.append(n)
            tr.append(n)
            n_par += int(p.numel())
    if extra:
        raise SystemExit(f"обучаемое вне белого списка: {extra[:5]}")
    if not tr:
        raise SystemExit("обучаемых тензоров нет: голова заморожена целиком")
    return tr, n_par


def select_arm(arms, tol=GRIP_TOL, min_seeds=SEL_SEEDS_MIN):
    """Пре-регистрированный отбор: по СРЕДНЕМУ ПО СИДАМ, с гейтом по схвату.

    `arms` — словарь ключ -> список записей по сидам, каждая с полями
    `pos`, `rot`, `grip_delta_hi` — ВЕРХНЯЯ ГРАНИЦА ИНТЕРВАЛА ПАРНОЙ
    РАЗНИЦЫ «голова минус черновик», а не интервал абсолютной ошибки.

    ТРИ ПРАВИЛА, КОТОРЫЕ ЛЕГКО НАРУШИТЬ СЛУЧАЙНО.
    Первое: берётся среднее по сидам, а не лучший сид — выбор лучшего сида
    есть выбор шума, и на двух сидах это даёт заметное смещение вверх.
    Второе: схват — ЖЁСТКИЙ гейт по ВЕРХНЕЙ границе, а не слагаемое: рука,
    роняющая предмет, не компенсируется точностью позы.
    Третье: разница ПАРНАЯ. Прежняя версия сравнивала верхнюю границу
    АБСОЛЮТНОЙ ошибки головы с ТОЧЕЧНЫМ схватом черновика и тем выбрасывала
    корреляцию: голова и черновик мерены на одних эпизодах, их разброс общий,
    и такой гейт мог и отвергнуть годную конфигурацию, и принять негодную.
    """
    rows, skipped = [], []
    for key, runs in sorted(arms.items()):
        if len(runs) < min_seeds:
            skipped.append((key, len(runs)))
            continue
        if any(r.get("grip_delta_hi") is None for r in runs):
            raise SystemExit(
                f"у {key} нет верхней границы ПАРНОЙ разницы по схвату: гейт "
                f"нечем считать. Отчётный прогон обязан идти с бутстрапом")
        pos = float(np.mean([r["pos"] for r in runs]))
        rot = float(np.mean([r["rot"] for r in runs]))
        # Гейт по ХУДШЕМУ сиду: конфигурация, проваливающая схват хотя бы на
        # одном сиде, не годится — при новом сиде провалит и она.
        grip_hi = float(max(r["grip_delta_hi"] for r in runs))
        ok = grip_hi <= tol + 1e-12
        rows.append(dict(key=key, pos=pos, rot=rot, grip_delta_hi=grip_hi,
                         pos_window=float(np.mean(
                             [r.get("pos_window", r["pos"]) for r in runs])),
                         window=int(max(r.get("window", 1) for r in runs)),
                         ok=ok, score=0.5 * (pos + rot), n_seeds=len(runs),
                         # ПО ХУДШЕМУ СИДУ, А НЕ ПО СРЕДНЕМУ: рост одного
                         # сида и падение другого взаимно уничтожились бы,
                         # и неустоявшаяся пара читалась бы как полка.
                         last_delta=max((r.get("last_delta", 0.0)
                                         for r in runs), key=abs)))
    good = [r for r in rows if r["ok"]]
    best = max(good, key=lambda r: r["score"]) if good else None
    return best, sorted(rows, key=lambda r: -r["score"]), skipped


CONV_TOL = 0.005   # прирост доли за последнюю эпоху, ниже которого считаем,
                   # что кривая вышла на полку


def read_train(best, rows, probe_pos, oracle_pos, skipped=(), last_delta=None,
               conv_tol=CONV_TOL):
    """Чтение результата обучения относительно двух ориентиров.

    ПРИЧИНА ОТСУТСТВИЯ ПОБЕДИТЕЛЯ НАЗЫВАЕТСЯ ТОЧНО. Прежняя версия при пустой
    таблице печатала «ни одна конфигурация не прошла гейт по схвату», хотя на
    деле ни одна конфигурация до гейта и не дошла: при одном сиде правило
    отбора их отбрасывает. Сообщение выдавало нехватку сидов за провал
    схвата — тот же класс ошибки, что мы ловим в измерениях.
    """
    if best is None and not rows:
        why = (f" (пропущены из-за нехватки сидов: "
               + ", ".join(f"{k} — {n}" for k, n in skipped) + ")"
               ) if skipped else ""
        return ("ОТБОР НЕ ПРОВОДИЛСЯ: ни одна конфигурация не набрала "
                f"минимума в {SEL_SEEDS_MIN} сида{why}. Это НЕ провал гейта "
                "по схвату и вообще не результат — прогон с одним сидом "
                "годится только для проверки механики")
    if best is None:
        return ("НИ ОДНА КОНФИГУРАЦИЯ НЕ ПРОШЛА ГЕЙТ ПО СХВАТУ. Поза и "
                "вращение здесь не важны: поправка, переворачивающая знак "
                "схвата, непригодна независимо от их качества")
    p = best["pos"]
    txt = (f"ЛУЧШАЯ КОНФИГУРАЦИЯ {best['key']}: доля возвращённого улучшения "
           f"{p:.1%} по положению и {best['rot']:.1%} по вращению "
           f"(среднее по {best['n_seeds']} сидам), схват прошёл гейт")
    # СНАЧАЛА ВОПРОС О СХОДИМОСТИ, ПОТОМ ОБ АРХИТЕКТУРЕ.
    #
    # Прежняя версия при `p <= probe_pos` объявляла, что три слоя с tanh не
    # дали ничего сверх линейной регрессии. На РАСТУЩЕЙ кривой это вывод о
    # причине, которого данные не позволяют: доля могла не дойти до зонда
    # просто потому, что эпох не хватило. Тот же класс ошибки, что выдавать
    # совпадение изменений за механизм.
    if last_delta is not None and abs(last_delta) > conv_tol:
        d_ = "выросла" if last_delta > 0 else "упала"
        return (txt + f".\n  СРАВНЕНИЕ С ЗОНДОМ НЕДЕЙСТВИТЕЛЬНО: разница "
                f"средних по двум окнам {last_delta:+.1%} (доля {d_}) при "
                f"допуске {conv_tol:.1%} — кривая НЕ вышла на полку хотя бы "
                f"у одного сида. Вывод об архитектуре на неустоявшейся "
                f"кривой сделать нельзя")
    if p <= probe_pos:
        txt += (f".\n  ОБУЧЕННАЯ ГОЛОВА НЕ ПРЕВЗОШЛА ЛИНЕЙНЫЙ ЗОНД "
                f"({probe_pos:.1%})"
                + (f" ПРИ ВЫШЕДШЕЙ НА ПОЛКУ КРИВОЙ (разница средних по двум "
                   f"окнам {last_delta:+.1%})" if last_delta is not None
                   else " (сходимость НЕ проверена: прирост за последнюю "
                        "эпоху неизвестен)")
                + ": три слоя с tanh не дали ничего сверх линейной регрессии "
                  "на тех же входах. Это не отменяет наличие сигнала, но "
                  "обесценивает выбор архитектуры головы")
    elif p < oracle_pos / 2:
        txt += (f".\n  Это выше зонда ({probe_pos:.1%}), но меньше половины "
                f"оракула ({oracle_pos:.1%}): ёмкость ранга 32 в основном "
                f"не используется")
    else:
        txt += (f".\n  Это выше зонда ({probe_pos:.1%}) и больше половины "
                f"оракула ({oracle_pos:.1%})")
    return txt


def selftest():
    # --- нулевая эпоха -----------------------------------------------------
    assert epoch0_ok(0.0, 0.0, 0.0)
    assert epoch0_ok(0.0, -0.0, ACT_TOL)
    # КОНТРОЛЬ: любое отклонение обязано ловиться, иначе проверка пустая
    assert not epoch0_ok(1e-6, 0.0, 0.0)
    assert not epoch0_ok(0.0, -1e-6, 0.0)
    assert not epoch0_ok(0.0, 0.0, ACT_TOL * 10)

    # --- отбор -------------------------------------------------------------
    # Записи содержат ВЕРХНЮЮ ГРАНИЦУ ПАРНОЙ РАЗНИЦЫ по схвату, а не
    # абсолютную ошибку: гейт задан как «голова не хуже черновика больше чем
    # на допуск», и считаться обязан так же.
    arms = {
        "star/1e-4": [dict(pos=0.30, rot=0.28, grip_delta_hi=-0.004),
                      dict(pos=0.34, rot=0.32, grip_delta_hi=0.001)],
        "star/1e-3": [dict(pos=0.50, rot=0.10, grip_delta_hi=0.000),
                      dict(pos=0.10, rot=0.10, grip_delta_hi=0.000)],
        "act/1e-4": [dict(pos=0.60, rot=0.60, grip_delta_hi=0.061),
                     dict(pos=0.60, rot=0.60, grip_delta_hi=0.001)],
    }
    best, rows, _sk = select_arm(arms)
    # среднее по сидам: star/1e-4 даёт 0.31, star/1e-3 — 0.30
    assert best["key"] == "star/1e-4", best
    # КОНТРОЛЬ: по ЛУЧШЕМУ сиду победил бы star/1e-3 (0.50) — правило обязано
    # этого не делать
    assert max(arms["star/1e-3"], key=lambda r: r["pos"])["pos"] > best["pos"]
    # схват провален на одном сиде -> вся конфигурация не годится
    assert not [r for r in rows if r["key"] == "act/1e-4"][0]["ok"]
    # РОВНО НА ДОПУСКЕ проходит, чуть выше — нет
    b_e, *_ = select_arm({"e": [dict(pos=0.9, rot=0.9,
                                    grip_delta_hi=GRIP_TOL)] * 2})
    assert b_e is not None
    b_e2, *_ = select_arm({"e": [dict(pos=0.9, rot=0.9,
                                     grip_delta_hi=GRIP_TOL * 1.01)] * 2})
    assert b_e2 is None
    # КОНТРОЛЬ ЕДИНИЦ: абсолютная доля ошибок схвата (около 2.5%) НЕ является
    # разницей и обязана проваливать гейт, если подставить её по ошибке
    b_abs, *_ = select_arm({"x": [dict(pos=0.9, rot=0.9,
                                      grip_delta_hi=0.025)] * 2})
    assert b_abs is None, "абсолютная величина принята вместо разницы"
    # конфигурация с одним сидом не участвует вовсе
    b2, r2, _ = select_arm(dict(arms, solo=[dict(pos=0.99, rot=0.99,
                                              grip_delta_hi=0.0)]))
    assert b2["key"] == "star/1e-4" and "solo" not in [r["key"] for r in r2]
    # ни одной прошедшей -> None, а не молчаливый выбор
    b3, *_ = select_arm({"x": [dict(pos=0.9, rot=0.9, grip_delta_hi=0.9)] * 2})
    assert b3 is None
    # отсутствие интервала — ОТКАЗ, а не переход на точечную оценку
    try:
        select_arm({"x": [dict(pos=0.9, rot=0.9, grip_delta_hi=None)] * 2})
    except SystemExit:
        pass
    else:
        raise AssertionError("конфигурация без интервала принята")
    # ПРИЧИНА НАЗЫВАЕТСЯ ТОЧНО. Пустая таблица — это нехватка сидов, а не
    # провал гейта; на смоук-прогоне с одним сидом прежняя версия печатала
    # именно «не прошла гейт по схвату», то есть выдавала одну причину за
    # другую.
    t_none = read_train(None, [], 0.105, 0.904, skipped=[("star/1e-4", 1)])
    assert "ОТБОР НЕ ПРОВОДИЛСЯ" in t_none and "нехватки сидов" in t_none
    assert "НЕ провал гейта" in t_none, t_none
    # а при непустой таблице без прошедших — действительно провал гейта
    t_gate = read_train(None, [dict(key="x", pos=0.1, rot=0.1,
                                    grip_delta_hi=0.9, ok=False, score=0.1,
                                    n_seeds=2)], 0.105, 0.904)
    assert "НИ ОДНА КОНФИГУРАЦИЯ НЕ ПРОШЛА ГЕЙТ" in t_gate, t_gate
    assert "ОТБОР НЕ ПРОВОДИЛСЯ" not in t_gate

    # --- чтение: сходимость проверяется ДО вывода об архитектуре ----------
    # ИМЕННО ЭТОТ СЛУЧАЙ ВСТРЕТИЛСЯ НА ДАННЫХ: 7.8% против зонда 10.5% при
    # приросте +1.5 п.п. за последнюю эпоху. Прежняя версия объявляла на этом
    # вывод об архитектуре, хотя кривая ещё росла.
    t_und = read_train(dict(key="k", pos=0.078, rot=0.072, n_seeds=2), [],
                       0.105, 0.904, last_delta=0.015)
    # ПАДЕНИЕ — ТОЖЕ НЕ ПОЛКА. Прежнее правило сравнивало величину со знаком,
    # поэтому −3% читались как сходимость.
    t_down = read_train(dict(key="k", pos=0.078, rot=0.072, n_seeds=2), [],
                        0.105, 0.904, last_delta=-0.03)
    assert "НЕДЕЙСТВИТЕЛЬНО" in t_down and "упала" in t_down, t_down
    assert "не дали ничего сверх" not in t_down, t_down
    assert "НЕДЕЙСТВИТЕЛЬНО" in t_und and "НЕ вышла на полку" in t_und, t_und
    assert "не дали ничего сверх" not in t_und, t_und
    # на вышедшей на полку кривой вывод об архитектуре делается
    t_conv = read_train(dict(key="k", pos=0.078, rot=0.072, n_seeds=2), [],
                        0.105, 0.904, last_delta=0.001)
    assert "НЕ ПРЕВЗОШЛА ЛИНЕЙНЫЙ ЗОНД" in t_conv, t_conv
    # и если голова зонд превзошла, недообучение вывод не блокирует... но
    # блокирует: растущая кривая делает недействительным ЛЮБОЕ сравнение
    t_up = read_train(dict(key="k", pos=0.30, rot=0.30, n_seeds=2), [],
                      0.105, 0.904, last_delta=0.05)
    assert "НЕДЕЙСТВИТЕЛЬНО" in t_up, t_up
    # без сведений о приросте поведение прежнее
    assert "НЕ ПРЕВЗОШЛА" in read_train(
        dict(key="k", pos=0.078, rot=0.072, n_seeds=2), [], 0.105, 0.904)

    # --- чтение ------------------------------------------------------------
    t = read_train(dict(key="k", pos=0.09, rot=0.09, n_seeds=2), [], 0.105,
                   0.904)
    assert "НЕ ПРЕВЗОШЛА ЛИНЕЙНЫЙ ЗОНД" in t, t
    t = read_train(dict(key="k", pos=0.30, rot=0.30, n_seeds=2), [], 0.105,
                   0.904)
    assert "меньше половины" in t, t
    t = read_train(dict(key="k", pos=0.70, rot=0.70, n_seeds=2), [], 0.105,
                   0.904)
    assert "больше половины" in t, t

    # --- белый список обучаемого -------------------------------------------
    class FakeP:
        def __init__(self, req, n):
            self.requires_grad, self._n = req, n

        def numel(self):
            return self._n

    class FakeHead:
        def __init__(self, items):
            self._i = items

        def named_parameters(self):
            return iter(self._i)

    ok_head = FakeHead([("proj.weight", FakeP(True, 10)),
                        ("net.0.weight", FakeP(True, 20)),
                        ("basis", FakeP(False, 99))])
    tr, n = trainable_report(ok_head)
    assert n == 30 and len(tr) == 2, (tr, n)
    # КОНТРОЛЬ: посторонний обучаемый параметр обязан ловиться отказом
    bad = FakeHead([("proj.weight", FakeP(True, 10)),
                    ("basis", FakeP(True, 99))])
    try:
        trainable_report(bad)
    except SystemExit:
        pass
    else:
        raise AssertionError("обучаемый буфер принят")
    # и полностью замороженная голова — тоже отказ, а не «ноль параметров»
    try:
        trainable_report(FakeHead([("proj.weight", FakeP(False, 10))]))
    except SystemExit:
        pass
    else:
        raise AssertionError("замороженная голова принята")

    # --- линейная диагностическая голова -----------------------------------
    try:
        import torch as _t
    except ImportError:
        raise SystemExit(
            "нет torch: сетевая часть самопроверки НЕ ВЫПОЛНЕНА. Ставьте "
            "CPU-сборку и повторите — молча пропущенные тесты уже прятали "
            "падения")
    LH = make_linear_head()
    lh = LH(8, 6, rank=3)
    Bq = _t.linalg.qr(_t.randn(6, 3))[0]
    lh.set_basis(Bq)
    lh.set_rho(_t.full((3,), 0.5))
    nm_, np_ = trainable_report(lh)
    assert np_ == (8 + 6) * 3 + 3 and len(nm_) == 2, (nm_, np_)
    hh, zz = _t.randn(2, 4, 8), _t.randn(2, 4, 6)
    dz_, c_ = lh(hh, zz)
    # ТОЖДЕСТВЕННОСТЬ ПРИ НУЛЕВОЙ ИНИЦИАЛИЗАЦИИ — иначе эпоха 0 не сойдётся
    # с черновиком и опора уедет.
    assert float(dz_.abs().max()) == 0.0, "линейная голова не нулевая в старте"
    # ...но живая: после шага поправка обязана появиться
    op_ = _t.optim.AdamW([p for p in lh.parameters() if p.requires_grad],
                         lr=1e-2)
    (lh(hh, zz)[1] - 1.0).pow(2).mean().backward()
    op_.step()
    assert float(lh(hh, zz)[0].abs().max()) > 0, "линейная голова мертва"
    # предел соблюдается: базис ортонормален, значит ||dz|| <= ||rho||
    assert bool((lh(hh, zz)[0].norm(dim=-1) <= lh.rho.norm() + 1e-5).all())
    # неортонормальный базис и неположительная rho — отказ
    for bad_call in (lambda: lh.set_basis(_t.randn(6, 3)),
                     lambda: lh.set_rho(_t.zeros(3))):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("линейная голова приняла негодные буферы")

    # --- потеря режет префикс ----------------------------------------------
    try:
        import torch
    except ImportError:
        raise SystemExit(
            "нет torch: сетевая часть самопроверки НЕ ВЫПОЛНЕНА. Ставьте "
            "CPU-сборку и повторите — молча пропущенные тесты уже прятали "
            "падения")
    a = torch.zeros(3, 20, 7)
    b = torch.zeros(3, 20, 7)
    b[:, H_EXEC:] = 100.0
    assert float(loss_terms(a, b)) == 0.0, "потеря захватила хвост чанка"
    # ВЕСА (1,1,1) ОБЯЗАНЫ ВОСПРОИЗВОДИТЬ РАВНОМЕРНУЮ ПОТЕРЮ до последнего
    # знака — иначе прежние прогоны перестают сравниваться с новыми.
    rg3 = torch.Generator().manual_seed(7)
    qa = torch.randn(3, 20, 7, generator=rg3) * 0.1
    qb = torch.randn(3, 20, 7, generator=rg3) * 0.1
    assert abs(float(loss_terms(qa, qb, weights=(1, 1, 1)))
               - float(loss_terms(qa, qb))) < 1e-9
    # нулевой вес схвата обязан убирать его вклад целиком
    qc = qb.clone()
    qc[..., 6] += 5.0
    assert abs(float(loss_terms(qc, qb, weights=(1, 1, 0)))
               - float(loss_terms(qb, qb, weights=(1, 1, 0)))) < 1e-9
    # и вес схвата обязан менять величину, если ошибка схвата есть
    assert float(loss_terms(qc, qb, weights=(1, 1, 1))) > \
        float(loss_terms(qc, qb, weights=(1, 1, 0.1)))
    # --- разложение по каналам складывается в общую потерю ------------------
    rg2 = torch.Generator().manual_seed(3)
    pa = torch.randn(4, 20, 7, generator=rg2) * 0.1
    pb = torch.randn(4, 20, 7, generator=rg2) * 0.1
    dec = loss_by_channel(pa, pb)
    assert abs(dec["total"] - float(loss_terms(pa, pb))) < 1e-6, dec
    assert abs(sum(dec[k] for k in ("pos", "rot", "grip"))
               - dec["total"]) < 1e-9
    assert abs(sum(dec[k + "_frac"] for k in ("pos", "rot", "grip"))
               - 1.0) < 1e-9
    # КОНТРОЛЬ: большая ошибка ТОЛЬКО в схвате обязана дать долю схвата
    # близкую к единице — иначе разложение не показывает, чем занята потеря.
    pc = pb.clone()
    pc[..., 6] += 2.0
    d2 = loss_by_channel(pc, pb)
    assert d2["grip_frac"] > 0.9, d2
    assert d2["pos_frac"] < 0.05 and d2["rot_frac"] < 0.05, d2
    b2_ = torch.zeros(3, 20, 7)
    b2_[:, 0] = 1.0
    assert float(loss_terms(a, b2_)) > 0.0, "потеря не видит исполняемых шагов"

    print("самопроверка k11c пройдена (версия «сходимость перед выводом об архитектуре»): "
          "нулевая эпоха ловит любое отклонение от черновика, отбор идёт по "
          "среднему по сидам и НЕ выбрал бы лучший сид, гейт схвата считает "
          "ПАРНУЮ разницу и отвергает подставленную вместо неё абсолютную "
          "величину, отсутствие интервала — отказ, а не точечная оценка, "
          "вывод об архитектуре блокируется, пока кривая не вышла на полку, "
          "веса каналов при (1,1,1) воспроизводят равномерную потерю и "
          "нулевой вес убирает канал целиком, разложение потери по каналам "
          "складывается в общую и показывает "
          "канал, которым она занята, линейная диагностическая голова нулевая "
          "в старте, живая после "
          "шага и соблюдает предел, конфигурация с одним сидом не участвует "
          "и её отсутствие "
          "называется нехваткой сидов, а не провалом гейта, посторонний "
          "обучаемый "
          "параметр и полностью замороженная голова — отказ, потеря режет "
          "хвост чанка и видит исполняемые шаги")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", help="префикс кэша K-11a")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512,
                    help="ширина головы; обязана совпадать с init_hicora")
    ap.add_argument("--proj", type=int, default=64)
    ap.add_argument("--chan-weights", default=None,
                    help="три множителя каналов потери действий через запятую "
                         "(положение, вращение, схват). Без флага — прежнее "
                         "равномерное усреднение по семи каналам")
    ap.add_argument("--preload", action="store_true",
                    help="держать отвод train в ОЗУ (около 3 ГиБ): за эпоху "
                         "иначе идут 121 тыс. случайных чтений с диска")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--lrs", default="1e-4,1e-3")
    ap.add_argument("--wds", default="0",
                    help="затухание весов; перебирается наравне со скоростью. "
                         "Потеря на обучении падает, а доля на val стоит — "
                         "это переобучение, и штраф на веса первый кандидат")
    ap.add_argument("--targets", default="star,action",
                    help="star — D(z*), action — истинные действия A*, "
                         "coef — коэффициенты, та же мишень, что у зонда")
    ap.add_argument("--archs", default="mlp",
                    help="mlp — рабочая голова; linear — диагностическая "
                         "линейная той же формы выхода. Матрица "
                         "«архитектура на мишень» разделяет причины отставания "
                         "от зонда")
    ap.add_argument("--eval-ckpts", default=None,
                    help="каталог или список .pt через запятую: только "
                         "перекрёстная оценка сохранённых весов, без обучения")
    ap.add_argument("--eval-probe", default=None,
                    help="npz с весами зонда: контроль воспроизведения. "
                         "Оценивается дважды — с обрезанием (как у зонда) и с "
                         "tanh (как у головы)")
    ap.add_argument("--allow-probe-mismatch", action="store_true",
                    help="разрешить ориентир зонда, собранный на других "
                         "артефактах")
    ap.add_argument("--val-n", type=int, default=0,
                    help="ограничить val для скорости; 0 — весь")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--res-norm-cache", default="data/k11c_res_norm.pt")
    ap.add_argument("--probe", default="data/k11p_probe.json",
                    help="отчёт зонда: его доля — ориентир при чтении")
    ap.add_argument("--allow-module-drift", action="store_true")
    ap.add_argument("--out", default="data/k11c_d1.json")
    ap.add_argument("--ckpt-dir", default="data/k11c")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()

    for f in ("cache", "out", "ckpt_dir", "res_norm_cache", "probe"):
        if getattr(args, f):
            setattr(args, f, os.path.abspath(getattr(args, f)))
    args.root = os.path.abspath(args.root)
    sys.path.insert(0, args.root)
    sha = k11a.file_sha1(__file__)
    print(f"k11c sha1 {sha}")
    for need, why in ((args.ckpt, "--ckpt"), (args.cache, "--cache")):
        if not need:
            raise SystemExit(f"нужен {why}")

    import torch
    import actioncodec  # noqa: F401
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if "action_codec" not in CONFIG_MAPPING:
        raise SystemExit("тип «action_codec» не зарегистрирован")
    from utils import seed_everything
    import hicora_vla as hv
    import joint12_vla as jv

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("контейнер потерял видеокарты")
        gi = int(args.device.split(":")[1]) if ":" in args.device else 0
        free_b, total_b = torch.cuda.mem_get_info(gi)
        print(f"  {args.device}: свободно {free_b / 2 ** 30:.1f} ГиБ из "
              f"{total_b / 2 ** 30:.1f}")
    dev, dt = torch.device(args.device), getattr(torch, args.dtype)

    prefix = args.cache
    meta = json.load(open(prefix + ".meta.json"))
    diag = json.load(open(prefix + ".diag.json"))
    basis_p, rho_p = prefix + ".basis.npy", prefix + ".rho.npy"
    B = np.load(basis_p).astype(np.float32)
    rho = np.load(rho_p).astype(np.float32)
    k11b.check_artifacts(diag, prefix, k11a.file_sha1(basis_p),
                         k11a.file_sha1(rho_p), B.shape[1],
                         k11a.file_sha1(prefix + ".meta.json"))
    mod_drift = k11b.check_cache_fields(
        meta, diag, dict(ckpt=args.ckpt),
        dict(hicora_vla_sha1=k11a.file_sha1(hv.__file__),
             joint12_vla_sha1=k11a.file_sha1(jv.__file__)),
        allow_drift=args.allow_module_drift)
    if mod_drift:
        print(f"  ВНИМАНИЕ: версии модулей разошлись с кэшем: {mod_drift}")
    rank = int(B.shape[1])
    dev_i = float(np.abs(B.T.astype(np.float64) @ B.astype(np.float64)
                         - np.eye(rank)).max())
    if dev_i > 1e-4:
        raise SystemExit(f"базис не ортонормален: {dev_i:.2e}")

    stamp_p = prefix + ".artifacts.json"
    if not os.path.exists(stamp_p):
        raise SystemExit(f"нет {stamp_p}: правильность позднего входа не "
                         f"подтверждена, запустите K-11b")
    stamp = json.load(open(stamp_p))
    tap = max(meta["saved_taps"])
    arr_sha = {f"h{tap}": k11a.file_sha1(f"{prefix}.h{tap}.npy")}
    for nm in ("q0hat", "ktrue", "split", "codebooks"):
        p_ = f"{prefix}.{nm}.npy"
        if os.path.exists(p_):
            arr_sha[nm] = k11a.file_sha1(p_)

    # --- res_norm: снимается один раз и кэшируется, sha сверяется -----------
    if os.path.exists(args.res_norm_cache):
        res_norm = torch.load(args.res_norm_cache, map_location=dev,
                              weights_only=False).to(dev).eval()
        print(f"  res_norm прочитана из {args.res_norm_cache}")
    else:
        import copy
        from smolvla.bar import SmolVLABlockwiseAR
        from utils import get_cfg
        from joint12_vla import make_joint12_class
        cfg0 = get_cfg(os.path.join(args.root, args.cfg_path))
        cfg0.TRAINING.ckpt_dir = args.ckpt
        cfg0.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt
        m0 = make_joint12_class(SmolVLABlockwiseAR).from_pretrained(
            **cfg0.MODEL.vlm.kwargs).to(dev, dt).eval()
        m0.init_joint_fast(depth=int(meta["depth"]), head_dtype=dt)
        res_norm = copy.deepcopy(m0.action_expert.norm).to(dev).eval()
        del m0
        torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(args.res_norm_cache) or ".",
                    exist_ok=True)
        torch.save(res_norm, args.res_norm_cache)
        print(f"  res_norm снята с чекпойнта и сохранена в "
              f"{args.res_norm_cache}; модель выгружена")
    for p_ in res_norm.parameters():
        p_.requires_grad_(False)
    rn_sha = k11a.state_sha1(res_norm)
    k11p.check_stamp(stamp, arr_sha, rn_sha, tap,
                     k11a.file_sha1(prefix + ".meta.json"),
                     basis_sha=k11a.file_sha1(basis_p),
                     rho_sha=k11a.file_sha1(rho_p),
                     k11b_sha=k11a.file_sha1(k11b.__file__))
    print(f"  вход заверён K-11b ({stamp['script_sha1']}), res_norm sha "
          f"{rn_sha}; отпечатки {len(arr_sha)} массивов совпали")

    _, codec, E, _ = k11a.load_codec(args)
    codec.requires_grad_(False)
    codec.eval()
    Esav = np.load(prefix + ".codebooks.npy")
    if float(np.abs(Esav - E.cpu().numpy()).max()) > 1e-4:
        raise SystemExit("кодовые книги разошлись с кэшем")
    unfrozen = [n for n, p_ in codec.named_parameters() if p_.requires_grad]
    if unfrozen:
        raise SystemExit(f"кодек не заморожен: {unfrozen[:3]}")

    # --- данные -------------------------------------------------------------
    q0 = np.load(prefix + ".q0hat.npy")
    Kt = np.load(prefix + ".ktrue.npy")
    split = np.load(prefix + ".split.npy", allow_pickle=True).astype(str)
    H = np.load(f"{prefix}.h{tap}.npy", mmap_mode="r")
    dc = np.load(meta["cache"], allow_pickle=True)
    epi = np.asarray(dc["episode"]).astype(np.int64)
    ACT = dc["action"]
    if ACT.shape[0] != len(q0):
        raise SystemExit(f"действий {ACT.shape[0]}, кодов {len(q0)}")
    tr = np.where(split == "train")[0]
    va = np.where(split == "val")[0]
    if len(np.intersect1d(np.unique(epi[tr]), np.unique(epi[va]))):
        raise SystemExit("эпизоды train и val пересекаются")
    if args.val_n and len(va) > args.val_n:
        eps_v = np.unique(epi[va])
        rg = np.random.default_rng(41)
        keep, took = set(), 0
        for e in rg.permutation(eps_v):
            keep.add(int(e))
            took += int((epi[va] == e).sum())
            if took >= args.val_n:
                break
        va = va[np.isin(epi[va], list(keep))]
    print(f"D1: train {len(tr)}, val {len(va)} ({len(np.unique(epi[va]))} "
          f"эпизодов), ранг {rank}, ||rho|| {float(np.linalg.norm(rho)):.4f}")

    Bt = torch.as_tensor(B, device=dev)
    rho_t = torch.as_tensor(rho, device=dev)
    D_H = int(H.shape[2])

    def z_of(codes0, all_levels=None):
        z = E[0][torch.as_tensor(np.asarray(codes0)).long().to(dev)]
        if all_levels is not None:
            k = torch.as_tensor(np.asarray(all_levels)).long().to(dev)
            for l in range(1, E.shape[0]):
                z = z + E[l][k[:, l, :]]
        return z

    # ПРЕДЗАГРУЗКА НЕ МЕНЯЕТ ЧИСЛА, только источник байтов: массив тот же,
    # его sha уже сверена с отпечатками K-11b выше.
    Hmem = None
    if args.preload:
        need = H.shape[0] * H.shape[1] * H.shape[2] * 2 / 2 ** 30
        print(f"  предзагрузка отвода в ОЗУ: {need:.1f} ГиБ")
        Hmem = np.asarray(H)

    def take_h(idx):
        return (Hmem[idx] if Hmem is not None else np.asarray(H[idx]))

    def inputs(idx):
        hb = torch.from_numpy(take_h(idx)).to(dev, dt)
        with torch.no_grad():
            hn = res_norm(hb).float()
            z0b = z_of(q0[idx])
            zsb = z_of(Kt[idx, 0, :], Kt[idx])
        return hn, z0b, zsb

    def decode(z):
        return codec._decode(z, embodiment_ids=0)[0][..., :7].float()

    # ФИКСИРОВАННЫЕ СЛУЧАЙНЫЕ ПОДВЫБОРКИ ДЛЯ РАЗРЫВА, одни на весь прогон.
    _rg_gap = np.random.default_rng(51)
    n_gap = min(4096, len(tr), len(va))
    gap_tr = np.sort(_rg_gap.choice(tr, n_gap, replace=False))
    gap_va = np.sort(_rg_gap.choice(va, n_gap, replace=False))
    print(f"  разрыв считается на фиксированных случайных подвыборках по "
          f"{n_gap} наблюдений (сид 51), одними и теми же финальными весами")

    ep_va = epi[va]
    ep_ids = np.unique(ep_va)
    ep_pos = {int(e): i for i, e in enumerate(ep_ids)}

    def train_loss_on(head, idx, tgt):
        """Потеря обучения на заданных индексах. ТА ЖЕ функция, что в цикле.

        БЕЗ ЭТОГО ПЕРЕОБУЧЕНИЕ НЕ ДИАГНОСТИРУЕТСЯ. Прежде на train падал
        `SmoothL1` действий, а на val стояла доля возвращённого RMS — это
        РАЗНЫЕ величины, и их расхождение совместимо и с переобучением, и с
        несовпадением функций качества, и с ограничением оптимизации. Чтобы
        различить, нужна одна и та же потеря по обе стороны.
        """
        import torch
        hn, z0b, zsb = inputs(idx)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=dt):
            dz, c = head(hn, z0b)
        with torch.no_grad():
            if tgt == "coef":
                a_ = (zsb - z0b).float() @ Bt
                t_ = torch.clamp(a_, -rho_t, rho_t) / rho_t
                return float(torch.nn.functional.smooth_l1_loss(c.float(), t_))
            A_ = decode(z0b + dz.float())
            T_ = decode(zsb) if tgt == "star" else torch.from_numpy(
                np.asarray(ACT[idx], np.float32)).to(dev)
            return float(loss_terms(A_, T_))

    def evaluate(head, n_boot=0, tgt=None):
        """Доли улучшения относительно ОБЕИХ опор плюс насыщение и val-потеря."""
        S = {"star": np.zeros((len(ep_ids), len(ARMS), N_STAT)),
             "action": np.zeros((len(ep_ids), len(ARMS), N_STAT))}
        act_gap, sat_tok, sat_n = 0.0, 0, 0
        head.eval()
        for i, j in k11a.plan_batches(len(va), args.batch):
            idx = va[i:j]
            hn, z0b, zsb = inputs(idx)
            with torch.no_grad(), torch.autocast(device_type=dev.type,
                                                 dtype=dt):
                dz, c = head(hn, z0b)
            with torch.no_grad():
                A_draft = decode(z0b).cpu().numpy()
                A_star = decode(zsb).cpu().numpy()
                a_coef = (zsb - z0b).float() @ Bt
                A_orc = decode(z0b + torch.clamp(
                    a_coef, -rho_t, rho_t) @ Bt.T).cpu().numpy()
                A_head = decode(z0b + dz.float()).cpu().numpy()
            # НАСЫЩЕНИЕ СЧИТАЕТСЯ В ЕДИНИЦАХ tanh. Голова возвращает `c` —
            # безразмерный выход tanh в [-1, 1]; предел применяется к нему
            # умножением. Сравнение `|c| > rho*0.99` смешивало две шкалы:
            # печаталась величина, зависящая от масштаба rho и насыщения не
            # означающая. Порог тот же, что в `hv.saturation_report`.
            over = (c.abs().float() > 0.99)
            sat_tok += int(over.any(-1).sum())
            sat_n += int(over.shape[0] * over.shape[1])
            act_gap = max(act_gap, float(np.abs(A_head - A_draft).max()))
            A_true = np.asarray(ACT[idx], np.float64)
            for ref_name, ref in (("star", A_star), ("action", A_true)):
                for ai, arr in enumerate((A_draft, A_orc, A_head)):
                    per = k11p.err_per_obs(arr, ref)
                    for o in range(len(idx)):
                        S[ref_name][ep_pos[int(ep_va[i + o])], ai] += per[o]
        out = {}
        for ref_name in ("star", "action"):
            tot = S[ref_name].sum(0)
            g = {}
            base = k11p.finish(tot[ARMS.index("draft")])
            for ai, nm in enumerate(ARMS):
                e = k11p.finish(tot[ai])
                g[nm] = dict(
                    pos=float(1 - e["pos"] / base["pos"]) if base["pos"] > 0
                    else None,
                    rot=float(1 - e["rot"] / base["rot"]) if base["rot"] > 0
                    else None,
                    grip=e["grip"], err_pos=e["pos"], err_rot=e["rot"])
            if n_boot:
                dr = k11p.boot_draws(S[ref_name], n_boot=n_boot,
                                     seed=k11p.BOOT_SEED)
                gg = []
                for x in dr:
                    b2 = k11p.finish(x[ARMS.index("draft")])
                    e2 = k11p.finish(x[ARMS.index("head")])
                    # СХВАТ — ПАРНАЯ РАЗНИЦА ВНУТРИ РОЗЫГРЫША, а не интервал
                    # абсолютной величины. Сравнивать верхнюю границу
                    # абсолютной ошибки головы с ТОЧЕЧНЫМ схватом черновика
                    # значит выбросить корреляцию: голова и черновик мерены
                    # на одних эпизодах, и их разброс общий. Правило,
                    # заявленное как «верхняя граница разницы», обязано и
                    # считаться как разница — так это сделано в K-11p.
                    gg.append((1 - e2["pos"] / b2["pos"],
                               1 - e2["rot"] / b2["rot"],
                               e2["grip"], e2["grip"] - b2["grip"]))
                g["head"]["ci_pos"] = list(k11p.ci([x[0] for x in gg]))
                g["head"]["ci_rot"] = list(k11p.ci([x[1] for x in gg]))
                g["head"]["ci_grip"] = list(k11p.ci([x[2] for x in gg]))
                g["head"]["ci_grip_delta"] = list(k11p.ci([x[3] for x in gg]))
                g["head"]["grip_delta"] = float(
                    g["head"]["grip"] - g["draft"]["grip"])
            out[ref_name] = g
        out["saturated_tokens"] = float(sat_tok / max(sat_n, 1))
        out["act_gap_vs_draft"] = act_gap
        if tgt is not None:
            # РАЗРЫВ СЧИТАЕТСЯ ОДНИМИ И ТЕМИ ЖЕ ФИНАЛЬНЫМИ ВЕСАМИ по обе
            # стороны. Прежде «train» была бегущим средним потерь ВО ВРЕМЯ
            # обновления весов внутри эпохи, а «val» — оценкой уже готовой
            # модели: это не две оценки одной модели, и разрыв между ними
            # ничего не диагностировал.
            #
            # Подвыборки СЛУЧАЙНЫЕ и фиксированные, а не префикс: кэш
            # упорядочен по эпизодам, и `va[:4096]` — это первые задачи.
            for nm_, sub in (("train_loss_fixed", gap_tr),
                             ("val_loss", gap_va)):
                ls = []
                for i2, j2 in k11a.plan_batches(len(sub), args.batch):
                    ls.append(train_loss_on(head, sub[i2:j2], tgt))
                out[nm_] = float(np.mean(ls)) if ls else None
        return out

    # --- ПРОИСХОЖДЕНИЕ ЗОНДА СВЕРЯЕТСЯ: это ЦЕНТРАЛЬНОЕ сравнение --------
    # Прежде доля зонда читалась из произвольного файла без единой проверки.
    # Зонд, посчитанный на другом кэше, другом базисе или другом ранге, дал бы
    # ориентир, к которому наши числа отношения не имеют, — и сравнение
    # «голова против зонда» стало бы сравнением двух разных задач.
    probe_pos, probe_meta = None, None
    if os.path.exists(args.probe):
        pr = json.load(open(args.probe))
        bad = []
        for k_, want in (("cache", prefix), ("rank", rank),
                         ("basis_sha1", k11a.file_sha1(basis_p)),
                         ("rho_sha1", k11a.file_sha1(rho_p))):
            got = pr.get(k_)
            if got is None:
                bad.append(f"{k_}: нет поля")
            elif str(got) != str(want):
                bad.append(f"{k_}: {got!r} против {want!r}")
        if not pr.get("routing_ok"):
            bad.append("routing_ok: вердикт зонда не положителен")
        if bad and not args.allow_probe_mismatch:
            raise SystemExit(
                "ориентир зонда не сверился:\n    " + "\n    ".join(bad)
                + "\n  Сравнение головы с чужим зондом бессмысленно. "
                  "Пересчитайте зонд или разрешите флагом "
                  "--allow-probe-mismatch")
        if bad:
            print(f"  ВНИМАНИЕ: ориентир зонда принят с расхождениями {bad}")
        probe_pos = (pr.get("gains", {}).get("both", {}) or {}).get("pos")
        probe_meta = dict(path=args.probe, script_sha1=pr.get("script_sha1"),
                          rank=pr.get("rank"), mismatch=bad,
                          json_sha1=k11a.file_sha1(args.probe))
        print(f"  ориентир зонда: {probe_pos:.1%} по положению, "
              f"зонд {pr.get('script_sha1')}, sha отчёта "
              f"{probe_meta['json_sha1']}; кэш, ранг, базис и rho совпали"
              if probe_pos is not None else "  зонд без поля gains.both.pos")
    else:
        print(f"  ориентира зонда нет ({args.probe} отсутствует): вывод о "
              f"сравнении с ним делаться не будет")

    def cross_eval():
        """ОДНИ И ТЕ ЖЕ ТРИ МЕТРИКИ ДЛЯ ВСЕХ СОХРАНЁННЫХ ВЕСОВ.

        ЗАЧЕМ. При обучении каждая ветвь считала потерю ПО СВОЕЙ мишени, и
        числа вроде 0.01380 и 0.00885 лежат в разных пространствах: они не
        говорят, какая голова лучше оптимизировала действия. Пока обе не
        измерены ОДНИМИ метриками, вывод «виновата функция потерь» не
        доказан — он одинаково совместим с трудностью оптимизации и с
        несовпадением обучающей и отчётной метрик.

        ЧИТАТЬ ТАК: если голова с мишенью-коэффициентами лучше и по потере
        ДЕЙСТВИЙ — виновата оптимизация action-loss. Если она хуже по
        действиям, но лучше по доле RMS — расходятся обучающая и отчётная
        метрики, и это другой вывод.
        """
        paths = []
        for tok in str(args.eval_ckpts).split(","):
            tok = os.path.abspath(tok.strip())
            if os.path.isdir(tok):
                paths += sorted(os.path.join(tok, f) for f in os.listdir(tok)
                                if f.endswith(".pt"))
            elif os.path.exists(tok):
                paths.append(tok)
            else:
                raise SystemExit(f"нет {tok}")
        if not paths:
            raise SystemExit("не нашлось ни одного .pt")
        rows_e = []

        def losses_on(head, idx, tgt):
            return float(np.mean([train_loss_on(head, idx[i2:j2], tgt)
                                  for i2, j2 in k11a.plan_batches(
                                      len(idx), args.batch)]))

        def decompose(head, idx):
            """Чем занята потеря действий: разложение по группам каналов."""
            acc = {}
            for i2, j2 in k11a.plan_batches(len(idx), args.batch):
                sel = idx[i2:j2]
                hn, z0b, zsb = inputs(sel)
                with torch.no_grad(), torch.autocast(device_type=dev.type,
                                                     dtype=dt):
                    dz, _ = head(hn, z0b)
                with torch.no_grad():
                    d_ = loss_by_channel(decode(z0b + dz.float()), decode(zsb))
                for k_, v_ in d_.items():
                    if not k_.endswith("_frac"):
                        acc[k_] = acc.get(k_, 0.0) + v_ * len(sel)
            n_ = max(len(idx), 1)
            out_ = {k_: v_ / n_ for k_, v_ in acc.items()}
            for k_ in ("pos", "rot", "grip"):
                out_[k_ + "_frac"] = (out_[k_] / out_["total"]
                                      if out_["total"] > 0 else None)
            return out_

        def measure(head, name, extra=None):
            ev = evaluate(head, n_boot=args.n_boot)
            g = ev["star"]["head"]
            lc = losses_on(head, gap_va, "coef")
            la = losses_on(head, gap_va, "star")
            # СТОРОНА TRAIN ТЕМИ ЖЕ ВЕСАМИ: отделяет «не смогла
            # минимизировать» от «минимизировала, но не обобщила».
            lc_tr = losses_on(head, gap_tr, "coef")
            la_tr = losses_on(head, gap_tr, "star")
            dec = decompose(head, gap_va)
            r = dict(name=name, pos=g["pos"], rot=g["rot"],
                     grip_delta=g.get("grip_delta"),
                     grip_delta_hi=(g.get("ci_grip_delta") or [None, None])[1],
                     loss_coef=lc, loss_action=la,
                     loss_coef_train=lc_tr, loss_action_train=la_tr,
                     action_by_channel=dec,
                     saturated=ev["saturated_tokens"])
            if extra:
                r.update(extra)
            rows_e.append(r)
            print(f"    {name}: поз {g['pos']:.1%}, вр {g['rot']:.1%}; "
                  f"коэф {lc_tr:.5f}/{lc:.5f}, действия "
                  f"{la_tr:.5f}/{la:.5f} (train/val); насыщено "
                  f"{ev['saturated_tokens']:.1%}", flush=True)
            print(f"      потеря действий по каналам: поз "
                  f"{dec['pos_frac']:.1%}, вр {dec['rot_frac']:.1%}, схват "
                  f"{dec['grip_frac']:.1%}", flush=True)

        print(f"\n  перекрёстная оценка {len(paths)} чекпойнтов, все метрики "
              f"на одном val ({len(va)} наблюдений):")
        for cp_ in paths:
            obj_ = torch.load(cp_, map_location="cpu", weights_only=False)
            for k_, want_ in (("basis_sha1", k11a.file_sha1(basis_p)),
                              ("rho_sha1", k11a.file_sha1(rho_p))):
                if obj_.get(k_) != want_:
                    raise SystemExit(
                        f"{os.path.basename(cp_)}: {k_} = {obj_.get(k_)}, а "
                        f"сейчас {want_} — веса от других артефактов")
            a_ = obj_.get("arch", "mlp")
            h_ = (hv.make_residual_head()(D_H, int(E.shape[-1]), rank=rank,
                                          hidden=args.hidden, proj=args.proj)
                  if a_ == "mlp"
                  else make_linear_head()(D_H, int(E.shape[-1]), rank=rank))
            h_ = h_.to(dev)
            h_.set_basis(torch.as_tensor(B))
            h_.set_rho(torch.as_tensor(rho))
            h_.float()
            st_ = {k_[len("hicora_head."):]: v
                   for k_, v in obj_["state"].items()
                   if k_.startswith("hicora_head.")}
            rep = h_.load_state_dict(st_, strict=False)
            if rep.unexpected_keys:
                raise SystemExit(f"лишние ключи в {cp_}: "
                                 f"{rep.unexpected_keys[:3]}")
            measure(h_, os.path.basename(cp_).replace(".pt", ""),
                    dict(arch=a_, target=obj_.get("target"),
                         lr=obj_.get("lr"), wd=obj_.get("wd"),
                         seed=obj_.get("seed"), ckpt=cp_))
            del h_
            torch.cuda.empty_cache()

        # КОНТРОЛЬ НА ИЗВЕСТНОМ ОТВЕТЕ. Веса зонда в голове С ОБРЕЗАНИЕМ
        # обязаны воспроизвести его долю: иначе расходятся пути оценки и всё
        # сравнение недействительно. Та же подстановка с `tanh` отделяет цену
        # способа ограничения от цены обучения.
        if args.eval_probe:
            zp = np.load(args.eval_probe, allow_pickle=True)
            wq = np.asarray(zp["w_clip"], np.float32)
            bq = np.asarray(zp["b_clip"], np.float32)
            need_sh = (D_H + int(E.shape[-1]), rank)
            if tuple(wq.shape) != need_sh:
                raise SystemExit(f"веса зонда формы {wq.shape}, ждали "
                                 f"{need_sh}")
            for sq in ("clip", "tanh"):
                hp = make_linear_head()(D_H, int(E.shape[-1]), rank=rank,
                                        squash=sq).to(dev)
                hp.set_basis(torch.as_tensor(B))
                hp.set_rho(torch.as_tensor(rho))
                hp.float()
                with torch.no_grad():
                    hp.lin.weight.copy_(torch.as_tensor(wq.T))
                    hp.lin.bias.copy_(torch.as_tensor(bq))
                measure(hp, f"зонд/{sq}",
                        dict(arch=f"probe-{sq}", target="coef-exact"))
                del hp
                torch.cuda.empty_cache()

        print(f"\n  {'веса':>34}{'поз':>8}{'вр':>8}{'коэф':>10}"
              f"{'действия':>11}{'дейст.tr':>10}{'схват в потере':>16}")
        for r in sorted(rows_e, key=lambda x: -x["pos"]):
            print(f"    {r['name']:>32}{r['pos']:>8.1%}{r['rot']:>8.1%}"
                  f"{r['loss_coef']:>10.5f}{r['loss_action']:>11.5f}"
                  f"{r['loss_action_train']:>10.5f}"
                  f"{r['action_by_channel']['grip_frac']:>15.1%}")
        if probe_pos is not None:
            print(f"    ориентир зонда по доле: {probe_pos:.1%}")
        print("    ЧИТАТЬ ТАК: обе потери посчитаны ОДНИМИ весами на ОДНОМ "
              "наборе.\n    Голова с мишенью-коэффициентами лучше и по потере "
              "ДЕЙСТВИЙ — виновата\n    оптимизация action-loss. Хуже по "
              "действиям, но лучше по доле —\n    расходятся обучающая и "
              "отчётная метрики.")
        out_e = dict(script_sha1=sha, cache=prefix, rank=rank, rows=rows_e,
                     probe_pos=probe_pos, probe_weights=args.eval_probe,
                     n_val=int(len(va)), n_gap=int(len(gap_va)))
        tmp_e = args.out + ".tmp"
        json.dump(out_e, open(tmp_e, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp_e, args.out)
        print(f"\n  сохранено: {args.out}")

    if args.eval_ckpts:
        cross_eval()
        return

    os.makedirs(args.ckpt_dir, exist_ok=True)
    if args.n_boot < 1:
        raise SystemExit(
            "--n-boot 0 обесценил бы гейт по схвату: он задан как ВЕРХНЯЯ "
            "граница интервала парной разницы, и без бутстрапа считался бы "
            "по точечной оценке. Это ослабление правила, а не ускорение")
    seeds = [int(x) for x in str(args.seeds).split(",") if x]
    lrs = [float(x) for x in str(args.lrs).split(",") if x]
    targets = [t for t in str(args.targets).split(",") if t]
    for t in targets:
        if t not in ("star", "action", "coef"):
            raise SystemExit(f"неизвестная мишень {t}")
    archs = [a for a in str(args.archs).split(",") if a]
    for a_ in archs:
        if a_ not in ("mlp", "linear"):
            raise SystemExit(f"неизвестная архитектура {a_}")
    runs, arms_acc = [], {}
    oracle_pos = None

    chan_w = None
    if args.chan_weights:
        chan_w = [float(x) for x in str(args.chan_weights).split(",") if x]
        if len(chan_w) != 3 or any(w < 0 for w in chan_w):
            raise SystemExit("--chan-weights: три неотрицательных числа")
        print(f"  веса каналов потери действий: положение {chan_w[0]:g}, "
              f"вращение {chan_w[1]:g}, схват {chan_w[2]:g}")
    wds = [float(x) for x in str(args.wds).split(",") if x]
    for tgt in targets:
      for arch in archs:
       for wd in wds:
        for lr in lrs:
            key = f"{arch}/{tgt}/{lr:g}/wd{wd:g}"
            for seed in seeds:
                seed_everything(seed)
                # ГОЛОВА СТРОИТСЯ ТОЙ ЖЕ ФАБРИКОЙ И ТЕМИ ЖЕ РАЗМЕРАМИ,
                # что и в `init_hicora`: `make_residual_head()` возвращает
                # КЛАСС. Иначе обученные веса не легли бы в модель на K-11e —
                # и заметили бы это только там.
                if arch == "mlp":
                    head = hv.make_residual_head()(
                        D_H, int(E.shape[-1]), rank=rank,
                        hidden=args.hidden, proj=args.proj).to(dev)
                else:
                    head = make_linear_head()(
                        D_H, int(E.shape[-1]), rank=rank).to(dev)
                head.set_basis(torch.as_tensor(B))
                head.set_rho(torch.as_tensor(rho))
                head.float()
                names, n_par = trainable_report(head)
                opt = torch.optim.AdamW(
                    [p for p in head.parameters() if p.requires_grad], lr=lr,
                    weight_decay=wd)
                scaler = torch.amp.GradScaler(dev.type)

                # ЭПОХА 0 — ЧАСТЬ ЗАМЕРА, А НЕ ФОРМАЛЬНОСТЬ.
                e0 = evaluate(head, tgt=tgt)
                g0 = e0["star"]["head"]
                if not epoch0_ok(g0["pos"], g0["rot"], e0["act_gap_vs_draft"]):
                    raise SystemExit(
                        f"эпоха 0 НЕ тождественна черновику: доля "
                        f"{g0['pos']:.3e}/{g0['rot']:.3e}, |ΔA| "
                        f"{e0['act_gap_vs_draft']:.3e}. Опора посчитана не "
                        f"тем кодом, что ветка головы")
                if oracle_pos is None:
                    oracle_pos = e0["star"]["oracle"]["pos"]
                    print(f"  опоры на val: оракул r={rank} даёт "
                          f"{oracle_pos:.1%} по положению и "
                          f"{e0['star']['oracle']['rot']:.1%} по вращению "
                          f"(опора D(z*)); относительно A* — "
                          f"{e0['action']['oracle']['pos']:.1%} и "
                          f"{e0['action']['oracle']['rot']:.1%}, и там это "
                          f"НЕ потолок")
                print(f"\n  {key} сид {seed}: обучаемых {len(names)} "
                      f"тензоров, {n_par} параметров; эпоха 0 тождественна "
                      f"черновику")

                hist = [dict(epoch=0, val=e0)]
                rng = np.random.default_rng(1000 + seed)
                t0 = time.time()
                for ep in range(1, args.epochs + 1):
                    head.train()
                    order = rng.permutation(len(tr))
                    run_loss, nb = 0.0, 0
                    for i, j in k11a.plan_batches(len(tr), args.batch):
                        idx = np.sort(tr[order[i:j]])
                        hn, z0b, zsb = inputs(idx)
                        with torch.autocast(device_type=dev.type, dtype=dt):
                            dz, c_out = head(hn, z0b)
                        if tgt == "coef":
                            # МИШЕНЬ ЗОНДА: ограниченные коэффициенты. Декодер
                            # в графе не участвует, и обучение идёт заметно
                            # быстрее — но и оценивается всё равно в
                            # пространстве действий, как у всех вариантов.
                            with torch.no_grad():
                                a_c = (zsb - z0b).float() @ Bt
                                T = torch.clamp(a_c, -rho_t, rho_t) / rho_t
                            loss = torch.nn.functional.smooth_l1_loss(
                                c_out.float(), T)
                        else:
                            A = decode(z0b + dz.float())
                            if tgt == "star":
                                with torch.no_grad():
                                    T = decode(zsb)
                            else:
                                T = torch.from_numpy(
                                    np.asarray(ACT[idx], np.float32)).to(dev)
                            loss = loss_terms(A, T, weights=chan_w)
                        opt.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                        run_loss += float(loss)
                        nb += 1
                        if nb % 100 == 0:
                            print(f"      эпоха {ep}, батч {nb}, потеря "
                                  f"{run_loss / nb:.5f}", flush=True)
                    ev = evaluate(head, n_boot=args.n_boot
                                  if ep == args.epochs else 0, tgt=tgt)
                    hist.append(dict(epoch=ep, loss=run_loss / max(nb, 1),
                                     val=ev))
                    gs = ev["star"]["head"]
                    ga = ev["action"]["head"]
                    vl, tlf = ev.get("val_loss"), ev.get("train_loss_fixed")
                    print(f"    эпоха {ep}: потеря бегущая "
                          f"{run_loss / max(nb, 1):.5f}; финальными весами "
                          "train "
                          + ("—" if tlf is None else f"{tlf:.5f}")
                          + " / val " + ("—" if vl is None else f"{vl:.5f}")
                          + f"; D(z*) поз {gs['pos']:.1%} вр {gs['rot']:.1%} "
                          f"знак {gs['grip']:.1%}; A* поз {ga['pos']:.1%} вр "
                          f"{ga['rot']:.1%}; насыщено токенов "
                          f"{ev['saturated_tokens']:.1%}", flush=True)
                last = hist[-1]["val"]
                cp = os.path.join(args.ckpt_dir,
                                  f"d1_{arch}_{tgt}_{lr:g}_wd{wd:g}_s{seed}.pt")
                # ЧЕКПОЙНТ СООТВЕТСТВУЕТ ОТЧЁТНОМУ ЧИСЛУ. Прежде доля была
                # средним по трём эпохам, а сохранялись веса ТОЛЬКО последней:
                # ни одной конкретной головы, про которую доказано 13.0%, не
                # существовало. Теперь окно остаётся диагностикой стабильности,
                # а отчётное число и гейт берутся у ОДНОЙ сохранённой эпохи.
                torch.save(dict(
                    state={f"hicora_head.{k}": v.detach().cpu()
                           for k, v in head.state_dict().items()
                           if k.startswith(TRAIN_PREFIXES)},
                    target=tgt, arch=arch, lr=lr, wd=wd, seed=seed,
                    chan_weights=chan_w,
                    epochs=args.epochs,
                    rank=rank, script_sha1=sha,
                    basis_sha1=k11a.file_sha1(basis_p),
                    rho_sha1=k11a.file_sha1(rho_p),
                    res_norm_sha1=rn_sha, cache=prefix), cp)
                rec = dict(key=key, target=tgt, arch=arch, lr=lr, wd=wd,
                           seed=seed,
                           ckpt=cp, minutes=(time.time() - t0) / 60.0,
                           hist=hist)
                runs.append(rec)
                gd = last["star"]["head"].get("ci_grip_delta")
                if not gd or gd[1] is None:
                    raise SystemExit(
                        "нет интервала ПАРНОЙ разницы по схвату: гейт "
                        "выродился бы в точечную оценку. Запускайте с "
                        "--n-boot больше нуля")
                # ПРИРОСТ ЗА ПОСЛЕДНЮЮ ЭПОХУ — признак того, вышла ли
                # кривая на полку. Без него вывод об архитектуре делался бы
                # на недообученной голове.
                # СХОДИМОСТЬ ПО ОКНУ, А НЕ ПО ОДНОЙ ЭПОХЕ. Доля на val
                # колеблется в пределах процентного пункта от эпохи к эпохе
                # (у одного из сидов наблюдалось 7.7 -> 6.6 -> 7.4 -> 6.8),
                # поэтому разность двух соседних эпох — шумная статистика:
                # она может и объявить сходимость на растущей кривой, и
                # наоборот. Сравниваются средние по двум окнам.
                gains_h = [h["val"]["star"]["head"]["pos"] for h in hist[1:]]
                rot_h = [h["val"]["star"]["head"]["rot"] for h in hist[1:]]
                w_ = min(3, len(gains_h) // 2)
                if w_ >= 1:
                    delta_ = float(np.mean(gains_h[-w_:])
                                   - np.mean(gains_h[-2 * w_:-w_]))
                else:
                    delta_ = float(gains_h[-1]) if gains_h else 0.0
                prev = last["star"]["head"]["pos"] - delta_
                # ОЦЕНКА ПО ОКНУ, А НЕ ПО ПОСЛЕДНЕЙ ЭПОХЕ. Доля колеблется
                # на ±1 п.п. от эпохи к эпохе при разрыве с зондом в 3 п.п.:
                # у одного из сидов последняя эпоха дала 5.7% против 7.4% на
                # предыдущей — чистый шум, который пошёл бы в отбор. Это НЕ
                # выбор лучшего (иначе брали бы максимум), а снижение
                # дисперсии; значение последней эпохи сохраняется рядом.
                w_s_ = min(3, len(gains_h))
                arms_acc.setdefault(key, []).append(dict(
                    # ОТЧЁТНОЕ ЧИСЛО — У СОХРАНЁННОЙ ЭПОХИ, а окно рядом как
                    # диагностика стабильности.
                    pos=last["star"]["head"]["pos"],
                    rot=last["star"]["head"]["rot"],
                    pos_window=float(np.mean(gains_h[-w_s_:])),
                    rot_window=float(np.mean(rot_h[-w_s_:])),
                    window=int(w_s_),
                    grip_delta=last["star"]["head"]["grip_delta"],
                    grip_delta_hi=gd[1],
                    last_delta=float(last["star"]["head"]["pos"] - prev)))
                del head, opt
                torch.cuda.empty_cache()

    draft_grip = runs[0]["hist"][0]["val"]["star"]["draft"]["grip"]
    best, rows, skipped = select_arm(arms_acc)
    print(f"\n  сводка по конфигурациям (опора D(z*), среднее по сидам, "
          f"схват черновика {draft_grip:.1%}):")
    print(f"    {'конфигурация':>22}{'сидов':>7}{'поз ckpt':>10}{'поз окно':>10}{'вр ckpt':>9}"
          f"{'Δсхват сверху':>15}{'гейт':>7}{'Δокон':>10}")
    for r in rows:
        print(f"    {r['key']:>22}{r['n_seeds']:>7}{r['pos']:>10.1%}"
              f"{r['pos_window']:>10.1%}{r['rot']:>9.1%}{r['grip_delta_hi']:>14.2%}"
              f"{('да' if r['ok'] else 'НЕТ'):>7}{r['last_delta']:>+10.1%}")
    print(f"    («поз ckpt» — у СОХРАНЁННОЙ эпохи, оно и идёт в отбор и в "
          f"K-11e;\n     «поз окно» — среднее по последним "
          f"{rows[0]['window'] if rows else 3} эпохам, только диагностика "
          f"стабильности)")
    print(f"    (Δокон — разница средних по двум последним окнам эпох, "
          f"худший сид;\n     модуль выше {CONV_TOL:.1%} означает, что кривая "
          f"не вышла на полку и вывод\n     об архитектуре недействителен)")
    print(f"    (Δсхват — ВЕРХНЯЯ граница интервала ПАРНОЙ разницы "
          f"«голова минус черновик»,\n     допуск {GRIP_TOL:.1%}; худший сид "
          f"конфигурации)")
    print(f"\n  {read_train(best, rows, probe_pos or 0.0, oracle_pos or 1.0, skipped, None if best is None else best.get('last_delta'))}")
    print("  ЧИТАТЬ ТАК: доли относительно D(z*) сопоставимы с зондом и "
          "оракулом.\n  Доли относительно A* сопоставимы между собой, но "
          "оракул там НЕ потолок:\n  непрерывная поправка в решётку кодов "
          "попадать не обязана. Успех в\n  симуляторе отсюда НЕ следует — "
          "это K-11e.")

    out = dict(script_sha1=sha, cache=prefix, rank=rank, tap=int(tap),
               res_norm_sha1=rn_sha, module_drift=mod_drift,
               input_verified_by=stamp.get("script_sha1"),
               n_train=int(len(tr)), n_val=int(len(va)),
               n_val_episodes=int(len(ep_ids)),
               epochs=args.epochs, batch=args.batch,
               seeds=seeds, lrs=lrs, wds=wds, targets=targets, archs=archs,
               probe_pos=probe_pos, probe=probe_meta, oracle_pos=oracle_pos,
               draft_grip=draft_grip, runs=runs,
               selection=rows, best=best, skipped=skipped,
               grip_tol=GRIP_TOL,
               array_sha1=arr_sha,
               basis_sha1=k11a.file_sha1(basis_p),
               rho_sha1=k11a.file_sha1(rho_p),
               hicora_vla_sha1=k11a.file_sha1(hv.__file__))
    tmp = args.out + ".tmp"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"\n  сохранено: {args.out}; чекпойнты в {args.ckpt_dir}")


if __name__ == "__main__":
    main()
