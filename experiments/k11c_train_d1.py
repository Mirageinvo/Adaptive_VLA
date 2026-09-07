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
TRAIN_PREFIXES = ("proj.", "net.")
GRIP_TOL = 0.005
SEL_SEEDS_MIN = 2


def loss_terms(pred, target, h_exec=H_EXEC):
    """Потеря в пространстве действий на ИСПОЛНЯЕМЫХ шагах 0..h_exec-1.

    Ось времени — шаги чанка, а не латентные позиции Perceiver. Хвост чанка
    не исполняется при частоте вызовов, измеренной в K-9, и включать его в
    потерю значило бы тратить ёмкость ранга 32 на то, что будет отброшено.

    Каналы взвешены РАВНОМЕРНО. Знак схвата НЕ оптимизируется отдельно: он
    участвует в решении как жёсткий гейт, и подгонять его прямо означало бы
    обучать модель под собственный критерий приёмки.
    """
    import torch
    p = pred[:, :h_exec]
    t = target[:, :h_exec]
    return torch.nn.functional.smooth_l1_loss(p, t)


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
                         ok=ok, score=0.5 * (pos + rot), n_seeds=len(runs),
                         last_delta=float(np.mean(
                             [r.get("last_delta", 0.0) for r in runs]))))
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
    if last_delta is not None and last_delta > conv_tol:
        return (txt + f".\n  СРАВНЕНИЕ С ЗОНДОМ НЕДЕЙСТВИТЕЛЬНО: за последнюю "
                f"эпоху доля выросла на {last_delta:+.1%} при допуске "
                f"{conv_tol:.1%}, то есть обучение НЕ вышло на полку. Вывод "
                f"об архитектуре головы на растущей кривой сделать нельзя — "
                f"нужно больше эпох")
    if p <= probe_pos:
        txt += (f".\n  ОБУЧЕННАЯ ГОЛОВА НЕ ПРЕВЗОШЛА ЛИНЕЙНЫЙ ЗОНД "
                f"({probe_pos:.1%})"
                + (f" ПРИ ВЫШЕДШЕЙ НА ПОЛКУ КРИВОЙ (прирост за последнюю "
                   f"эпоху {last_delta:+.1%})" if last_delta is not None
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
    assert "НЕДЕЙСТВИТЕЛЬНО" in t_und and "НЕ вышло на полку" in t_und, t_und
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
    b2_ = torch.zeros(3, 20, 7)
    b2_[:, 0] = 1.0
    assert float(loss_terms(a, b2_)) > 0.0, "потеря не видит исполняемых шагов"

    print("самопроверка k11c пройдена (версия «сходимость перед выводом об архитектуре»): "
          "нулевая эпоха ловит любое отклонение от черновика, отбор идёт по "
          "среднему по сидам и НЕ выбрал бы лучший сид, гейт схвата считает "
          "ПАРНУЮ разницу и отвергает подставленную вместо неё абсолютную "
          "величину, отсутствие интервала — отказ, а не точечная оценка, "
          "вывод об архитектуре блокируется, пока кривая не вышла на полку, "
          "конфигурация с одним сидом не участвует и её отсутствие "
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
                    help="star — D(z*), action — истинные действия A*")
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

    ep_va = epi[va]
    ep_ids = np.unique(ep_va)
    ep_pos = {int(e): i for i, e in enumerate(ep_ids)}

    def evaluate(head, n_boot=0):
        """Доли улучшения относительно ОБЕИХ опор плюс насыщение."""
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
        return out

    probe_pos = None
    if os.path.exists(args.probe):
        pr = json.load(open(args.probe))
        probe_pos = (pr.get("gains", {}).get("both", {}) or {}).get("pos")
        print(f"  ориентир зонда: {probe_pos:.1%} по положению "
              f"({os.path.basename(args.probe)})" if probe_pos is not None
              else "  зонд без поля gains.both.pos")

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
        if t not in ("star", "action"):
            raise SystemExit(f"неизвестная мишень {t}")
    runs, arms_acc = [], {}
    oracle_pos = None

    wds = [float(x) for x in str(args.wds).split(",") if x]
    for tgt in targets:
      for wd in wds:
        for lr in lrs:
            key = f"{tgt}/{lr:g}/wd{wd:g}"
            for seed in seeds:
                seed_everything(seed)
                # ГОЛОВА СТРОИТСЯ ТОЙ ЖЕ ФАБРИКОЙ И ТЕМИ ЖЕ РАЗМЕРАМИ,
                # что и в `init_hicora`: `make_residual_head()` возвращает
                # КЛАСС. Иначе обученные веса не легли бы в модель на K-11e —
                # и заметили бы это только там.
                head = hv.make_residual_head()(
                    D_H, int(E.shape[-1]), rank=rank,
                    hidden=args.hidden, proj=args.proj).to(dev)
                head.set_basis(torch.as_tensor(B))
                head.set_rho(torch.as_tensor(rho))
                head.float()
                names, n_par = trainable_report(head)
                opt = torch.optim.AdamW(
                    [p for p in head.parameters() if p.requires_grad], lr=lr,
                    weight_decay=wd)
                scaler = torch.amp.GradScaler(dev.type)

                # ЭПОХА 0 — ЧАСТЬ ЗАМЕРА, А НЕ ФОРМАЛЬНОСТЬ.
                e0 = evaluate(head)
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
                            dz, _ = head(hn, z0b)
                        A = decode(z0b + dz.float())
                        if tgt == "star":
                            with torch.no_grad():
                                T = decode(zsb)
                        else:
                            T = torch.from_numpy(
                                np.asarray(ACT[idx], np.float32)).to(dev)
                        loss = loss_terms(A, T)
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
                                  if ep == args.epochs else 0)
                    hist.append(dict(epoch=ep, loss=run_loss / max(nb, 1),
                                     val=ev))
                    gs = ev["star"]["head"]
                    ga = ev["action"]["head"]
                    print(f"    эпоха {ep}: потеря {run_loss / max(nb, 1):.5f}"
                          f"; D(z*) поз {gs['pos']:.1%} вр {gs['rot']:.1%} "
                          f"знак {gs['grip']:.1%}; A* поз {ga['pos']:.1%} вр "
                          f"{ga['rot']:.1%}; насыщено токенов "
                          f"{ev['saturated_tokens']:.1%}", flush=True)
                last = hist[-1]["val"]
                cp = os.path.join(args.ckpt_dir,
                                  f"d1_{tgt}_{lr:g}_wd{wd:g}_s{seed}.pt")
                torch.save(dict(
                    state={f"hicora_head.{k}": v.detach().cpu()
                           for k, v in head.state_dict().items()
                           if k.startswith(TRAIN_PREFIXES)},
                    target=tgt, lr=lr, wd=wd, seed=seed, epochs=args.epochs,
                    rank=rank, script_sha1=sha,
                    basis_sha1=k11a.file_sha1(basis_p),
                    rho_sha1=k11a.file_sha1(rho_p),
                    res_norm_sha1=rn_sha, cache=prefix), cp)
                rec = dict(key=key, target=tgt, lr=lr, wd=wd, seed=seed,
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
                w_ = min(3, len(gains_h) // 2)
                if w_ >= 1:
                    delta_ = float(np.mean(gains_h[-w_:])
                                   - np.mean(gains_h[-2 * w_:-w_]))
                else:
                    delta_ = float(gains_h[-1]) if gains_h else 0.0
                prev = last["star"]["head"]["pos"] - delta_
                arms_acc.setdefault(key, []).append(dict(
                    pos=last["star"]["head"]["pos"],
                    rot=last["star"]["head"]["rot"],
                    grip_delta=last["star"]["head"]["grip_delta"],
                    grip_delta_hi=gd[1],
                    last_delta=float(last["star"]["head"]["pos"] - prev)))
                del head, opt
                torch.cuda.empty_cache()

    draft_grip = runs[0]["hist"][0]["val"]["star"]["draft"]["grip"]
    best, rows, skipped = select_arm(arms_acc)
    print(f"\n  сводка по конфигурациям (опора D(z*), среднее по сидам, "
          f"схват черновика {draft_grip:.1%}):")
    print(f"    {'конфигурация':>22}{'сидов':>7}{'поз':>8}{'вр':>8}"
          f"{'Δсхват сверху':>15}{'гейт':>7}{'прирост':>10}")
    for r in rows:
        print(f"    {r['key']:>22}{r['n_seeds']:>7}{r['pos']:>8.1%}"
              f"{r['rot']:>8.1%}{r['grip_delta_hi']:>14.2%}"
              f"{('да' if r['ok'] else 'НЕТ'):>7}{r['last_delta']:>+10.1%}")
    print(f"    (прирост — за ПОСЛЕДНЮЮ эпоху; выше {CONV_TOL:.1%} означает, "
          f"что кривая не вышла\n     на полку и вывод об архитектуре "
          f"недействителен)")
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
               seeds=seeds, lrs=lrs, wds=wds, targets=targets,
               probe_pos=probe_pos, oracle_pos=oracle_pos,
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
