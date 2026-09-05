"""K-11a: кэш отводов h12/h18/h24 и выбор размерности поправки.

ЗАЧЕМ. HiCoRA-D исправляет НЕ произвольный черновик, а тот код q0, который
модель действительно предсказала на слое 12. Поэтому остаток считается от
предсказанного q0_hat, а не от истинного q0*, и кэш хранит именно
предсказанные коды.

ОДИН ПРОХОД, ТРИ ОТВОДА. Двадцать четыре слоя исполняются ровно один раз;
состояние потока действий снимается после слоёв 12, 18 и 24. `joint12_vla.py`
не правится: его sha записан в каждой ячейке симуляторного гейта K-9.

РАЗМЕРНОСТЬ ВЫБИРАЕТСЯ ПОСЛЕ ДЕКОДЕРА, А НЕ В ЛАТЕНТЕ. Шестнадцать позиций
ActionCodec — это ГЛОБАЛЬНЫЕ ЗАПРОСЫ Perceiver, а не двадцать шагов чанка:
позиция 0 не соответствует действию 0, и каждый запрос влияет на весь чанк.
Прежняя версия называла первые восемь ЛАТЕНТНЫХ позиций «исполняемым
префиксом» — величина не имела заявленного смысла. Здесь префикс определяется
только в пространстве действий, после единственного декодирования, как
A[:, :8], а ранг выбирается по доле возвращённого улучшения

    G_r = [E(D(z0), A*) - E(D(z0 + P_r r), A*)] / [E(D(z0), A*) - E(D(z*), A*)]

отдельно по положению и по вращению. Объяснённая дисперсия латентного остатка
остаётся описательной: крупная латентная компонента может почти не влиять на
декодер, а малая — сильно.

БАЗИС СТРОИТСЯ ТОЛЬКО НА TRAIN, РАНГ ВЫБИРАЕТСЯ НА VAL, TEST НЕ ТРОГАЕТСЯ.
Иначе архитектурное решение принимается по тем же данным, на которых потом
объявляется результат.

БАЗИС НЕЦЕНТРИРОВАННЫЙ. Голова задаёт подпространство через ноль (dz = B c),
собственного вектора среднего у неё нет. Центрированная PCA показала бы 90%
объяснённой дисперсии и при среднем остатке, лежащем вне базиса.

ПАМЯТЬ: НАКАПЛИВАЕТСЯ ГРАМИАН, А НЕ ОСТАТОК. Массив (150000, 16, 512) в fp64
весит 9 ГиБ, а одновременно их нужно несколько. Здесь потоком копится
G = sum r^T r размера (D, D), и его собственные векторы дают базис без
хранения самого остатка.

ПРО h18: ЕГО СЕЙЧАС НИКТО НЕ ЧИТАЕТ, И ЭТО СОЗНАТЕЛЬНО. HiCoRA-D устроена как
h12 -> q0 и h24 -> поправка; средний отвод нужен только будущей HiCoRA-V с
двумя ступенями. Он снимается проходом (это бесплатно), но по умолчанию НЕ
СОХРАНЯЕТСЯ — см. `--save-taps`. Хранить треть кэша под ветку, которой ещё
нет, значило бы платить за неё диском заранее.

ЧТО ЕЩЁ МЕРЯЕТСЯ. Наложение весов Joint12 меняет вход слоёв 13-24:
перезаписываются не только слои 1-12, но и `bos_embedding`, участвующий во
всех шагах внимания. Печатаются ЧЕТЫРЕ разные величины, которые легко
перепутать:
  1. q0 Joint12 против исходной головы на чистом h12 — что изменило
     дообучение;
  2. q0 Joint12 против НАСТОЯЩЕГО coarse24 (generate на 24 слоях) —
     расстояние до опоры, у которой измерены 90.0%;
  3. расхождение ДЕКОДИРОВАННЫХ действий этих двух черновиков, отдельно по
     положению, вращению и знаку схвата;
  4. относительный дрейф h24 против чистого прохода.

СХВАТ. Утверждение «схват берётся из q0» было бы неверным: декодер отображает
латент совместно во все семь каналов, и любое изменение z меняет схват. Он
измеряется отдельным столбцом и в одну величину с позой не сводится.

Запуск:
    python3 experiments/k11a_build_hicora_cache.py --selftest

    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k11a_build_hicora_cache.py --ckpt <base> \\
        --cache data/k9_teacher_150k.npz --q0-source joint12 \\
        --joint-ckpt data/k9c_joint12.pt --out data/k11a_joint12

    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k11a_build_hicora_cache.py --ckpt <base> \\
        --diagnose data/k11a_joint12
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

N_POS, N_LEVEL, T_CHUNK, H_EXEC = 16, 3, 20, 8
TAPS = (12, 18, 24)
Q0_SOURCES = ("joint12", "readout", "coarse24")
PCA_RANKS = (4, 8, 16, 32, 64)
# Порог зафиксирован ДО прогона: принимается наименьший ранг, возвращающий
# столько доступного улучшения ПОСЛЕ ДЕКОДЕРА одновременно по положению и по
# вращению на исполняемых шагах.
GAIN_TARGET = 0.90
# ДОПУСК ПО СХВАТУ, ЗАФИКСИРОВАННЫЙ ДО ЧИСЕЛ. «Не смешивать схват с позой»
# означает отдельный ЖЁСТКИЙ критерий, а не отсутствие критерия: без него ранг,
# возвращающий 95% позы и переворачивающий 20% схватов, был бы выбран.
GRIP_DELTA = 0.005


def plan_batches(n, batch):
    return [(i, min(i + batch, n)) for i in range(0, n, batch)]


def file_sha1(path):
    """SHA самого файла, а не имён ключей.

    Прежняя версия хешировала `str(sorted(state))`, то есть СПИСОК ИМЁН: все
    эпохи одной архитектуры получали один отпечаток, и происхождение
    переставало что-либо значить.
    """
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


DATASET_REV = "v2.0"


def check_manifest(man, cache_meta, epi, split, rev=DATASET_REV):
    """Сверка манифеста K-9a. Отдельной функцией — ради самопроверки.

    ФОРМАТ ИМЕННО ТОТ, ЧТО ПИШЕТ K-9a: параллельные списки `episodes` и
    `splits`, плюс `split_seed`, `dataset_revision`, `dataset_repo`. Поле
    ревизии называется `dataset_revision`, а НЕ `revision`: чтение по
    неверному имени давало None и молча пропускало сравнение — ошибка,
    которую видно только на настоящем файле, поэтому её и проверяет
    самопроверка.

    Возвращает справку о манифесте; при любом расхождении бросает SystemExit.
    """
    if not (isinstance(man, dict) and "episodes" in man and "splits" in man):
        raise SystemExit("манифест: ожидались поля «episodes» и «splits»; "
                         "формат не тот, что пишет K-9a, и сверка частей "
                         "была бы пропущена")
    eps = [int(e) for e in man["episodes"]]
    sp = [str(x) for x in man["splits"]]
    if len(eps) != len(sp):
        raise SystemExit("манифест: длины episodes и splits не совпадают")
    if len(eps) != len(set(eps)):
        raise SystemExit("манифест: эпизоды дублируются")
    bad = sorted(set(sp) - {"train", "val", "test"})
    if bad:
        raise SystemExit(f"манифест: недопустимые метки частей {bad}")
    if int(man.get("split_seed", -1)) != int(cache_meta.get("split_seed", -2)):
        raise SystemExit(f"манифест: сид разбиения {man.get('split_seed')} "
                         f"против {cache_meta.get('split_seed')} в meta кэша")
    got_rev = man.get("dataset_revision")
    if got_rev != rev:
        raise SystemExit(f"манифест собран на ревизии {got_rev!r}, а "
                         f"состояния читаются с {rev!r}: это разные данные")
    sp_of = dict(zip(eps, sp))
    uniq = [int(e) for e in np.unique(epi)]
    miss = sorted(set(uniq) - set(eps))
    if miss:
        raise SystemExit(f"эпизоды {miss[:5]} есть в кэше, но не в манифесте")
    # ЧАСТЬ КАЖДОГО ЭПИЗОДА СВЕРЯЕТСЯ ПОШТУЧНО, и заодно проверяется, что
    # внутри эпизода она одна: смешение означало бы утечку между train и val
    # на уровне наблюдений.
    mixed, wrong = [], []
    sp_arr = np.asarray(split).astype(str)
    for e in uniq:
        vals = set(sp_arr[np.where(epi == e)[0]].tolist())
        if len(vals) != 1:
            mixed.append(e)
        elif sp_of[e] != next(iter(vals)):
            wrong.append(e)
    if mixed:
        raise SystemExit(f"эпизоды {mixed[:5]} размечены разными частями "
                         f"внутри себя — утечка между train и val")
    if wrong:
        raise SystemExit(f"часть не совпала с манифестом у эпизодов "
                         f"{wrong[:5]}")
    return dict(n_episodes=len(eps), split_seed=int(man["split_seed"]),
                dataset_revision=got_rev, dataset_repo=man.get("dataset_repo"),
                n_checked=len(uniq))


def gram_basis(G, rank):
    """Ортонормированный базис из НЕЦЕНТРИРОВАННОГО грамиана."""
    w, V = np.linalg.eigh(np.asarray(G, np.float64))
    o = np.argsort(w)[::-1]
    return V[:, o[:int(rank)]], w[o]


def explained(eigvals, ranks, d_latent=None):
    e = np.clip(np.asarray(eigvals, np.float64), 0, None)
    tot = float(e.sum())
    if tot <= 0:
        return {int(r): 0.0 for r in ranks}
    return {int(r): float(e[:int(r)].sum() / tot) for r in ranks
            if d_latent is None or int(r) <= int(d_latent)}


def err_sums(a, ref):
    """Суммы квадратов и счётчики на ИСПОЛНЯЕМЫХ ШАГАХ 0-7.

    СУММЫ, А НЕ ГОТОВЫЕ RMS. Среднее корней по батчам не равно корню общего
    среднего и зависит от размера последнего батча — величина слегка менялась
    бы от `--batch`. Корень берётся один раз, в `finish`.

    Ось времени — шаги чанка (их двадцать), а не латентные позиции Perceiver
    (их шестнадцать). Смешение позы со схватом в одно число уже стоило нам
    гейта, неспособного подтверждать (K-10g).
    """
    A = np.asarray(a, np.float64)[:, :H_EXEC]
    R = np.asarray(ref, np.float64)[:, :H_EXEC]
    d = A - R
    return dict(pos_sq=float((d[..., :3] ** 2).sum()), pos_n=int(d[..., :3].size),
                rot_sq=float((d[..., 3:6] ** 2).sum()),
                rot_n=int(d[..., 3:6].size),
                grip_bad=int((np.sign(A[..., 6])
                              != np.sign(R[..., 6])).sum()),
                grip_n=int(A[..., 6].size))


def err_add(acc, part):
    for k, v in part.items():
        acc[k] = acc.get(k, 0) + v
    return acc


def err_finish(acc):
    return dict(pos=float(np.sqrt(acc["pos_sq"] / max(acc["pos_n"], 1))),
                rot=float(np.sqrt(acc["rot_sq"] / max(acc["rot_n"], 1))),
                grip=float(acc["grip_bad"] / max(acc["grip_n"], 1)))


def action_err(a, ref):
    """Удобная обёртка для одного батча — тот же код, что и в накоплении."""
    return err_finish(err_sums(a, ref))


def gain(err_z0, err_r, err_star, eps=1e-12):
    """Доля ДОСТУПНОГО улучшения, возвращённая поправкой ранга r.

    Знаменатель — расстояние от черновика до полного трёхуровневого
    восстановления: это всё, что поправка вообще может вернуть. Отношение
    сырых ошибок польстило бы рангу тем сильнее, чем хуже черновик.
    """
    denom = err_z0 - err_star
    if denom <= eps:
        return None
    return float((err_z0 - err_r) / denom)


def pick_rank(gains, grip=None, grip_draft=None, ranks=PCA_RANKS,
              target=GAIN_TARGET, delta=GRIP_DELTA, d_latent=None):
    """Наименьший ранг, проходящий ТРИ условия одновременно.

    Доля возвращённого улучшения не ниже target И по положению, И по
    вращению — среднее по ним спрятало бы ранг, вытягивающий положение и
    проваливающий вращение. И ошибка знака схвата не хуже черновика больше
    чем на delta: без этого условия схват просто не участвовал бы в решении.

    РАНГ, РАВНЫЙ РАЗМЕРНОСТИ ЛАТЕНТА, — не сжатие: d компонент описывают d
    измерений полностью.
    """
    for r in sorted(int(x) for x in ranks):
        if d_latent is not None and r >= int(d_latent):
            break
        g = gains.get(r)
        if not g or g.get("pos") is None or g.get("rot") is None:
            continue
        if g["pos"] < target or g["rot"] < target:
            continue
        if grip is not None and grip_draft is not None:
            if grip.get(r) is None or grip[r] > grip_draft + delta:
                continue
        return r
    return None


def read_rank(rank, gains, grip=None, grip_draft=None, ranks=PCA_RANKS,
              target=GAIN_TARGET, delta=GRIP_DELTA):
    top = max(int(x) for x in ranks)
    if rank is not None:
        g = gains[rank]
        gr = ""
        if grip is not None and grip_draft is not None:
            gr = (f", знак схвата {grip[rank]:.1%} против {grip_draft:.1%} у "
                  f"черновика при допуске {delta:.1%}")
        return (f"ранг {rank} возвращает {g['pos']:.1%} доступного улучшения "
                f"по положению и {g['rot']:.1%} по вращению на шагах 0-7 "
                f"(порог {target:.0%}, зафиксирован до прогона){gr} — брать "
                f"r={rank}")
    if grip is not None and grip_draft is not None:
        blocked = [r for r in sorted(gains)
                   if gains[r].get("pos") is not None
                   and gains[r]["pos"] >= target
                   and gains[r].get("rot") is not None
                   and gains[r]["rot"] >= target
                   and grip.get(r) is not None
                   and grip[r] > grip_draft + delta]
        if blocked:
            return (f"ранги {blocked} проходят по позе, но ПОРТЯТ ЗНАК СХВАТА "
                    f"сверх допуска {delta:.1%}: латентная поправка ломает "
                    f"дискретный канал. Это отдельное стоп-условие, а не "
                    f"повод усреднить его с позой")
    best = gains.get(top) or {}
    gp = best.get("pos") or 0.0
    gr = best.get("rot") or 0.0
    return (f"даже ранг {top} возвращает лишь {gp:.1%} по положению и "
            f"{gr:.1%} по вращению — остаток не описывается низкоранговым "
            f"латентным базисом. По плану это стоп-условие ветки: сравнить с "
            f"поправкой прямо в пространстве действий как КОНТРОЛЕМ, а не "
            f"переходить к ней молча")


def selftest():
    assert plan_batches(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert sum(b - a for a, b in plan_batches(1000, 64)) == 1000

    # --- базис из грамиана на матрице ИЗВЕСТНОГО ранга ---------------------
    rng = np.random.default_rng(0)
    Bt = np.linalg.qr(rng.normal(size=(40, 6)))[0]
    X = rng.normal(size=(500, 6)) @ Bt.T
    Bh, w = gram_basis(X.T @ X, 6)
    assert np.abs(Bh.T @ Bh - np.eye(6)).max() < 1e-9, "базис не ортонормирован"
    # Восстановленное подпространство совпадает с истинным: проекция
    # сохраняет длину каждого столбца истинного базиса.
    assert np.abs(np.linalg.norm(Bh @ Bh.T @ Bt, axis=0) - 1).max() < 1e-8
    e = explained(w, PCA_RANKS)
    assert e[8] > 0.999 and e[4] < 0.999

    # --- НЕЦЕНТРИРОВАННОСТЬ ЗНАЧИМА ----------------------------------------
    # Остаток со смещением: центрированная PCA нашла бы шум, но голова,
    # задающая подпространство через ноль, среднее не представит.
    mu = rng.normal(size=40) * 5.0
    Y = mu[None] + rng.normal(size=(500, 40)) * 0.01
    e_unc = explained(gram_basis(Y.T @ Y, 1)[1], (1,))[1]
    Yc = Y - Y.mean(0)
    e_cen = explained(gram_basis(Yc.T @ Yc, 1)[1], (1,))[1]
    assert e_unc > 0.99 > e_cen, (e_unc, e_cen)

    # --- доля возвращённого улучшения --------------------------------------
    assert abs(gain(1.0, 0.5, 0.0) - 0.5) < 1e-12
    assert abs(gain(1.0, 0.1, 0.0) - 0.9) < 1e-12
    # Черновик уже идеален — улучшать нечего, и правило обязано сказать «не
    # определено», а не «100%».
    assert gain(0.3, 0.3, 0.3) is None
    # Поправка ХУЖЕ черновика даёт отрицательную долю, и это должно быть видно.
    assert gain(1.0, 1.5, 0.0) < 0

    # --- выбор ранга --------------------------------------------------------
    gs = {4: dict(pos=0.5, rot=0.5), 8: dict(pos=0.95, rot=0.4),
          16: dict(pos=0.96, rot=0.93), 32: dict(pos=0.99, rot=0.99),
          64: dict(pos=0.99, rot=0.99)}
    # РАНГ 8 НЕ ГОДИТСЯ, хотя положение вытягивает: вращение обязано пройти
    # порог тоже, иначе среднее спрятало бы провал канала.
    assert pick_rank(gs) == 16, pick_rank(gs)
    low = {r: dict(pos=0.2, rot=0.2) for r in PCA_RANKS}
    assert pick_rank(low) is None
    assert "стоп-условие" in read_rank(None, low)
    assert "r=16" in read_rank(16, gs)
    assert pick_rank(gs, d_latent=16) is None, "ранг = размерность принят"

    # --- СХВАТ ОТВЕРГАЕТ РАНГ САМОСТОЯТЕЛЬНО --------------------------------
    # Ранг 16 идеален по позе, но переворачивает пятую часть схватов: он
    # обязан быть отвергнут, а не усреднён с позой.
    gr_bad = {4: 0.02, 8: 0.02, 16: 0.20, 32: 0.021, 64: 0.02}
    assert pick_rank(gs, grip=gr_bad, grip_draft=0.02) == 32, \
        pick_rank(gs, grip=gr_bad, grip_draft=0.02)
    # Если схват портят ВСЕ проходящие ранги, вердикт обязан назвать причину.
    gr_all = {r: 0.30 for r in PCA_RANKS}
    assert pick_rank(gs, grip=gr_all, grip_draft=0.02) is None
    txt = read_rank(None, gs, grip=gr_all, grip_draft=0.02)
    assert "ПОРТЯТ ЗНАК СХВАТА" in txt, txt
    # Допуск действует: чуть хуже черновика — ещё проходит.
    gr_ok = {r: 0.02 + GRIP_DELTA for r in PCA_RANKS}
    assert pick_rank(gs, grip=gr_ok, grip_draft=0.02) == 16

    # --- АГРЕГИРОВАНИЕ НЕ ЗАВИСИТ ОТ РАЗБИЕНИЯ НА БАТЧИ ---------------------
    rng2 = np.random.default_rng(7)
    A = rng2.normal(size=(37, T_CHUNK, 7))
    R = rng2.normal(size=(37, T_CHUNK, 7))
    whole = action_err(A, R)
    for bs in (1, 5, 16, 37):
        acc = {}
        for i, j in plan_batches(len(A), bs):
            err_add(acc, err_sums(A[i:j], R[i:j]))
        got = err_finish(acc)
        for k in ("pos", "rot", "grip"):
            assert abs(got[k] - whole[k]) < 1e-12, (bs, k, got[k], whole[k])
    # Контроль самого теста: СРЕДНЕЕ RMS ПО БАТЧАМ от разбиения ЗАВИСИТ, и
    # именно поэтому оно заменено на суммы.
    naive = float(np.mean([action_err(A[i:j], R[i:j])["pos"]
                           for i, j in plan_batches(len(A), 5)]))
    assert abs(naive - whole["pos"]) > 1e-9, "тест инвариантности вырожден"

    # --- МАНИФЕСТ В НАСТОЯЩЕМ ФОРМАТЕ K-9a ----------------------------------
    # Формат я один раз угадал неверно (ждал словарь episode->split и поле
    # `revision` вместо `dataset_revision`), и на реальном файле сверка
    # молча вырождалась. Поэтому здесь именно тот словарь, который пишет
    # K-9a, и каждое искажение обязано приводить к отказу.
    good_man = dict(episodes=[1, 2, 3], splits=["train", "val", "test"],
                    split_seed=17, created_by="abc",
                    dataset_revision="v2.0",
                    dataset_repo="physical-intelligence/libero")
    cm = dict(split_seed=17)
    epi_t = np.array([1, 1, 2, 3])
    sp_t = np.array(["train", "train", "val", "test"])
    info = check_manifest(good_man, cm, epi_t, sp_t)
    assert info["dataset_revision"] == "v2.0" and info["n_checked"] == 3, info

    def must_fail(man, cmeta, e_, s_, frag, rev="v2.0"):
        try:
            check_manifest(man, cmeta, e_, s_, rev=rev)
        except SystemExit as ex:
            assert frag in str(ex), (frag, str(ex))
            return
        raise AssertionError(f"пропущено: {frag}")

    must_fail(dict(good_man, episodes=[1, 1, 3]), cm, epi_t, sp_t, "дублируются")
    must_fail(dict(good_man, splits=["train", "val"]), cm, epi_t, sp_t, "длины")
    must_fail(dict(good_man, splits=["train", "val", "trian"]), cm, epi_t,
              sp_t, "недопустимые")
    must_fail(good_man, dict(split_seed=99), epi_t, sp_t, "сид разбиения")
    must_fail({"eps": [1]}, cm, epi_t, sp_t, "episodes")
    # ПЕРЕСТАВЛЕННЫЙ SPLIT — главный случай: списки той же длины, метки
    # допустимые, но эпизод 2 объявлен тестовым, а в кэше он валидационный.
    must_fail(dict(good_man, splits=["train", "test", "val"]), cm, epi_t,
              sp_t, "не совпала")
    # НЕВЕРНАЯ РЕВИЗИЯ обязана отвергаться, а не читаться как None.
    must_fail(dict(good_man, dataset_revision="v1.0"), cm, epi_t, sp_t,
              "ревизии")
    must_fail({k: v for k, v in good_man.items() if k != "dataset_revision"},
              cm, epi_t, sp_t, "ревизии")
    # Смешанная часть внутри эпизода — утечка между train и val.
    must_fail(good_man, cm, epi_t, np.array(["train", "val", "val", "test"]),
              "разными частями")
    # Эпизод кэша вне манифеста.
    must_fail(good_man, cm, np.array([1, 2, 3, 9]),
              np.array(["train", "val", "test", "train"]), "не в манифесте")

    # --- ошибки по каналам --------------------------------------------------
    a = np.zeros((4, T_CHUNK, 7)); ref = np.zeros((4, T_CHUNK, 7))
    a[:, :, 0] = 0.3
    a[:, :, 6] = 1.0; ref[:, :, 6] = -1.0
    er = action_err(a, ref)
    assert abs(er["pos"] - 0.3 / np.sqrt(3)) < 1e-9, er
    assert er["rot"] == 0.0 and er["grip"] == 1.0
    # ПРЕФИКС РЕЖЕТСЯ ПО ШАГАМ ЧАНКА. Ось времени длиной 20; ошибка, лежащая
    # целиком в хвосте, обязана дать ноль. Прежняя версия резала ось из 16
    # латентных позиций Perceiver, и величина не значила заявленного.
    a2 = np.zeros((4, T_CHUNK, 7)); a2[:, H_EXEC:, 0] = 100.0
    a2[:, :, 6] = -1.0
    assert action_err(a2, ref)["pos"] == 0.0, "срез взял хвост"
    assert a2.shape[1] != N_POS, "ось времени спутана с латентными позициями"

    print("самопроверка k11a пройдена (версия «манифест до дорогого, отпечаток декодера»): "
          "базис из нецентрированного грамиана восстанавливает подпространство, "
          "центрирование теряет смещение, доля улучшения не определена при "
          "идеальном черновике и отрицательна при ухудшении, ранг требует "
          "порога по обоим каналам, схват отвергает ранг отдельным условием, "
          "агрегирование не зависит от размера батча, префикс режется по "
          "шагам чанка")


def decoder_probe(codec, E, dev):
    """Отпечаток ПОВЕДЕНИЯ декодера, а не только кодовых книг.

    Совпадение книг не доказывает совпадение нейронного декодера: книги — это
    выход `out_project(decode_code(...))`, а `_decode` содержит ещё и всю
    остальную сеть. Здесь на фиксированном псевдослучайном латенте берётся
    выход декодера и хешируется — если сменится что угодно в декодере, отпечаток
    изменится.
    """
    import torch
    g = torch.Generator(device="cpu").manual_seed(20260905)
    z = torch.randn(4, N_POS, int(E.shape[-1]), generator=g).to(dev)
    with torch.no_grad():
        y = codec._decode(z, embodiment_ids=0)[0][..., :7].float().cpu()
    a = np.ascontiguousarray(np.round(y.numpy(), 5).astype(np.float32))
    return hashlib.sha1(a.tobytes()).hexdigest()[:12]


def state_sha1(module):
    """SHA всего state_dict модуля, в фиксированном порядке ключей."""
    h = hashlib.sha1()
    for k in sorted(module.state_dict()):
        v = module.state_dict()[k]
        h.update(k.encode())
        h.update(np.ascontiguousarray(
            v.detach().float().cpu().numpy()).tobytes())
    return h.hexdigest()[:12]


def load_codec(args):
    """Только процессор и кодек: для диагностики VLM не нужен."""
    import torch
    from utils import VisionLanguageActionProcessor
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = ac if hasattr(ac, "vq") else getattr(ac, "codec", None)
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь")
    dev = torch.device(args.device)
    codec = codec.to(dev).eval()
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float()
    return proc, codec, E, dev


def diagnose(args):
    import torch
    prefix = args.diagnose
    meta = json.load(open(prefix + ".meta.json"))
    _, codec, E, dev = load_codec(args)
    D = int(E.shape[-1])
    if D != int(meta["d_latent"]):
        raise SystemExit(f"размерность латента {D} против {meta['d_latent']} "
                         f"в кэше: другой чекпойнт")
    if meta.get("ckpt") != args.ckpt:
        raise SystemExit(f"кэш собран чекпойнтом {meta.get('ckpt')}, а "
                         f"декодер берётся из {args.ckpt}")
    # КНИГИ СВЕРЯЮТСЯ С СОХРАНЁННЫМИ, А НЕ ПРОСТО ЗАГРУЖАЮТСЯ ЗАНОВО.
    # Совпадения пути чекпойнта мало: локальный кэш HuggingFace или сам
    # процессор могли смениться под тем же именем, и остаток считался бы в
    # одних координатах, а декодировался в других.
    Esav = np.load(prefix + ".codebooks.npy")
    Ecur = E.cpu().numpy()
    if Esav.shape != Ecur.shape:
        raise SystemExit(f"книги формы {Ecur.shape} против {Esav.shape} в "
                         f"кэше: другой кодек")
    dmax = float(np.abs(Esav - Ecur).max())
    if dmax > args.codebook_tol:
        raise SystemExit(
            f"кодовые книги разошлись на {dmax:.3e} при допуске "
            f"{args.codebook_tol:.0e}: декодер не тот, которым собран кэш")
    cb_sha = hashlib.sha1(np.ascontiguousarray(
        Esav.astype(np.float32)).tobytes()).hexdigest()[:12]
    # SHA СРАВНИВАЕТСЯ С ЗАПИСАННЫМ, а не просто печатается: иначе он был бы
    # справкой, а не проверкой.
    if meta.get("codebooks_sha1") not in (None, cb_sha):
        raise SystemExit(f"sha книг {cb_sha} против {meta['codebooks_sha1']} "
                         f"в кэше")
    # И ОТДЕЛЬНО — ПОВЕДЕНИЕ ДЕКОДЕРА: книги могут совпасть, а сеть за ними
    # смениться, и остаток считался бы в одних координатах, а декодировался
    # в других.
    probe = decoder_probe(codec, E, dev)
    st_sha = state_sha1(codec)
    if meta.get("decoder_probe") not in (None, probe):
        raise SystemExit(
            f"отпечаток декодера {probe} против {meta['decoder_probe']} в "
            f"кэше: книги те же, но сеть декодера другая")
    if meta.get("codec_state_sha1") not in (None, st_sha):
        raise SystemExit(f"sha весов кодека {st_sha} против "
                         f"{meta['codec_state_sha1']}")
    print(f"  книги сверены: max|Δ| = {dmax:.3e}, sha {cb_sha}; декодер "
          f"сверен: проба {probe}, веса {st_sha}")

    q0 = np.load(prefix + ".q0hat.npy")
    Kt = np.load(prefix + ".ktrue.npy")
    split = np.load(prefix + ".split.npy", allow_pickle=True).astype(str)
    N = len(q0)
    tr = np.where(split == "train")[0]
    va = np.where(split == "val")[0]
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit("нет train или val: базис строится на train, ранг "
                         "выбирается на val, test не трогается")
    print(f"диагностика: {N} наблюдений, train {len(tr)}, val {len(va)}, "
          f"test {int((split == 'test').sum())} (НЕ используется)")

    def z_of(codes0, all_levels=None):
        z = E[0][torch.as_tensor(np.asarray(codes0)).long().to(dev)]
        if all_levels is not None:
            k = torch.as_tensor(np.asarray(all_levels)).long().to(dev)
            for l in range(1, E.shape[0]):
                z = z + E[l][k[:, l, :]]
        return z

    # --- ГРАМИАН ПОТОКОМ, ТОЛЬКО TRAIN -------------------------------------
    G = np.zeros((D, D), np.float64)
    n_rows = 0
    for i, j in plan_batches(len(tr), args.batch * 16):
        s = tr[i:j]
        with torch.no_grad():
            r = (z_of(Kt[s, 0, :], Kt[s]) - z_of(q0[s])).reshape(-1, D)
            G += (r.double().T @ r.double()).cpu().numpy()
        n_rows += int(r.shape[0])
    print(f"  грамиан по {n_rows} строкам train, размер {G.shape}: сам "
          f"остаток не хранится")
    Bfull, w = gram_basis(G, min(max(PCA_RANKS), D))
    exp = explained(w, PCA_RANKS, d_latent=D)
    print(f"\n  объяснённая дисперсия латентного остатка (ОПИСАТЕЛЬНО, "
          f"размерность {D}):")
    for rk in PCA_RANKS:
        if rk in exp:
            mark = "  (= размерность, не сжатие)" if rk >= D else ""
            print(f"    ранг {rk:>3}: {exp[rk]:.1%}{mark}")

    # --- rho СЧИТАЕТСЯ ДО ВЫБОРА РАНГА, НА TRAIN ---------------------------
    # ПОРЯДОК ПРИНЦИПИАЛЕН. Прежде ранг выбирался по НЕОГРАНИЧЕННОЙ проекции,
    # а rho считалась после. Но обученная голова способна выдать только
    # |c_i| <= rho_i, поэтому выбранный так ранг мог быть недостижим
    # архитектурой. Теперь rho определяется для координат полного базиса на
    # train, а на val оценивается уже ОГРАНИЧЕННАЯ поправка.
    Bt = torch.as_tensor(Bfull, dtype=torch.float32, device=dev)
    take = tr if not args.diag_n else tr[:args.diag_n * 4]
    coef = []
    for i, j in plan_batches(len(take), args.batch * 16):
        s_ = take[i:j]
        with torch.no_grad():
            r = (z_of(Kt[s_, 0, :], Kt[s_]) - z_of(q0[s_])).reshape(-1, D)
            coef.append((r @ Bt).abs().cpu().numpy())
    coef = np.concatenate(coef)
    rho_full = np.percentile(coef, args.rho_pct, axis=0)
    if not np.isfinite(rho_full).all() or (rho_full <= 0).any():
        raise SystemExit("процентиль дал ноль или бесконечность: остаток "
                         "вырожден хотя бы по одной координате базиса")
    print(f"\n  rho: процентиль {args.rho_pct} по {len(coef)} строкам TRAIN, "
          f"по каждой из {Bfull.shape[1]} координат полного базиса")

    ranks = [r for r in PCA_RANKS if r <= Bfull.shape[1]]
    print(f"\n  насыщение на train (процентиль по КАЖДОЙ координате не значит "
          f"{100 - args.rho_pct:.0f}%\n  векторов вне предела: при r "
          f"координатах хотя бы одна выходит гораздо чаще):")
    print(f"    {'ранг':>8}{'коэфф.вне':>12}{'токенов вне':>13}"
          f"{'наблюд.вне':>13}{'координат':>11}{'||rho||':>10}")
    sat_stats = {}
    for rk in ranks:
        c_ = coef[:, :rk]
        rr = rho_full[:rk]
        over = c_ > rr[None]
        per_tok = over.any(1)
        # Наблюдение = 16 латентных позиций подряд.
        per_obs = per_tok.reshape(-1, N_POS).any(1)
        sat_stats[rk] = dict(coef_frac=float(over.mean()),
                             token_frac=float(per_tok.mean()),
                             obs_frac=float(per_obs.mean()),
                             mean_clamped=float(over.sum(1).mean()),
                             rho_norm=float(np.linalg.norm(rr)))
        st_ = sat_stats[rk]
        print(f"    {rk:>8}{st_['coef_frac']:>11.1%}"
              f"{st_['token_frac']:>12.1%}{st_['obs_frac']:>12.1%}"
              f"{st_['mean_clamped']:>11.2f}{st_['rho_norm']:>10.4f}")
    print(f"    (столбец «координат» — сколько в среднем координат из r "
          f"обрезается)")

    # --- ДОЛЯ УЛУЧШЕНИЯ НА VAL, ПОСЛЕ ДЕКОДЕРА, С ОГРАНИЧЕНИЕМ -------------
    va_s = va if not args.diag_n else va[:args.diag_n]
    print(f"\n  доля возвращённого улучшения на val ({len(va_s)} "
          f"наблюдений), ПОСЛЕ единственного декодирования.")
    print(f"  «огр.» — поправка в пределах rho, то есть достижимая головой; "
          f"«неогр.» — верхняя\n  граница проекции, которую архитектура "
          f"воспроизвести не обязана.")
    acc_c = {r: {} for r in ranks}
    acc_u = {r: {} for r in ranks}
    acc0 = {}
    for i, j in plan_batches(len(va_s), args.batch):
        s_ = va_s[i:j]
        with torch.no_grad():
            z0b = z_of(q0[s_])
            zsb = z_of(Kt[s_, 0, :], Kt[s_])
            A0 = codec._decode(z0b, embodiment_ids=0)[0][..., :7].float()
            As = codec._decode(zsb, embodiment_ids=0)[0][..., :7].float()
            As_n = As.cpu().numpy()
            err_add(acc0, err_sums(A0.cpu().numpy(), As_n))
            rb = zsb - z0b
            for rk in ranks:
                Br = Bt[:, :rk].to(rb.dtype)
                cf = rb @ Br
                lim = torch.as_tensor(rho_full[:rk], dtype=cf.dtype,
                                      device=cf.device)
                for tag, cc, acc_ in (("u", cf, acc_u),
                                      ("c", torch.clamp(cf, -lim, lim),
                                       acc_c)):
                    Ar = codec._decode(z0b + cc @ Br.T,
                                       embodiment_ids=0)[0][..., :7].float()
                    err_add(acc_[rk], err_sums(Ar.cpu().numpy(), As_n))

    e_z0 = err_finish(acc0)
    e_st = dict(pos=0.0, rot=0.0, grip=0.0)
    gains, gains_u, grips, grips_u = {}, {}, {}, {}
    print(f"    {'ранг':>8}{'поз.огр':>10}{'доля':>8}{'доля неогр':>12}"
          f"{'вр.огр':>10}{'доля':>8}{'знак':>8}")
    for rk in ranks:
        ec = err_finish(acc_c[rk])
        eu = err_finish(acc_u[rk])
        gains[rk] = {k: gain(e_z0[k], ec[k], e_st[k]) for k in ("pos", "rot")}
        gains_u[rk] = {k: gain(e_z0[k], eu[k], e_st[k])
                       for k in ("pos", "rot")}
        grips[rk] = ec["grip"]
        # СХВАТ У НЕОГРАНИЧЕННОЙ ВЕТКИ СВОЙ. Подстановка сюда `grips`
        # означала бы, что «без ограничения выбрали бы ранг X» посчитано с
        # чужим гейтом по схвату, и вывод мог оказаться неверным.
        grips_u[rk] = eu["grip"]
        f = lambda x: "—" if x is None else f"{x:.1%}"
        print(f"    {rk:>8}{ec['pos']:>10.5f}{f(gains[rk]['pos']):>8}"
              f"{f(gains_u[rk]['pos']):>12}{ec['rot']:>10.5f}"
              f"{f(gains[rk]['rot']):>8}{ec['grip']:>7.1%}")
    print(f"    {'черновик':>8}{e_z0['pos']:>10.5f}{'0.0%':>8}{'0.0%':>12}"
          f"{e_z0['rot']:>10.5f}{'0.0%':>8}{e_z0['grip']:>7.1%}")

    rank = pick_rank(gains, grip=grips, grip_draft=e_z0["grip"], d_latent=D)
    rank_u = pick_rank(gains_u, grip=grips_u, grip_draft=e_z0["grip"],
                       d_latent=D)
    print(f"\n  {read_rank(rank, gains, grip=grips, grip_draft=e_z0['grip'])}")
    if rank_u != rank:
        print(f"  БЕЗ ограничения был бы выбран ранг {rank_u} — разница и "
              f"есть цена предела амплитуды.")
    print("  СТОЛБЕЦ «знак» участвует в решении ОТДЕЛЬНЫМ жёстким условием "
          f"(допуск {GRIP_DELTA:.1%}),\n  а не усредняется с позой: латентная "
          "поправка меняет схват, и ранг,\n  переворачивающий его, "
          "отвергается независимо от качества позы.")

    rho = rho_full[:rank] if rank is not None else None

    out = dict(n_obs=int(N), q0_source=meta["q0_source"], d_latent=D,
               n_train=int(len(tr)), n_val=int(len(va_s)),
               explained={str(k): v for k, v in exp.items()},
               gains={str(k): v for k, v in gains.items()},
               gains_unclamped={str(k): v for k, v in gains_u.items()},
               grip={str(k): v for k, v in grips.items()},
               grip_unclamped={str(k): v for k, v in grips_u.items()},
               saturation={str(k): v for k, v in sat_stats.items()},
               err_draft=e_z0, rank=rank, rank_unclamped=rank_u,
               gain_target=GAIN_TARGET, grip_delta=GRIP_DELTA,
               rho=None if rho is None else rho.tolist(),
               rho_pct=args.rho_pct, prefix=prefix,
               cache_script_sha1=meta.get("script_sha1"),
               cache_meta_sha1=file_sha1(prefix + ".meta.json"),
               script_sha1=file_sha1(__file__))
    json.dump(out, open(prefix + ".diag.json", "w"), ensure_ascii=False,
              indent=1)
    if rank is not None:
        np.save(prefix + ".basis.npy", Bfull[:, :rank].astype(np.float32))
        np.save(prefix + ".rho.npy", rho.astype(np.float32))
        print(f"\n  базис и rho сохранены: {prefix}.{{basis,rho}}.npy")
    print(f"  сохранено: {prefix}.diag.json")
    print("\n  ЧИТАТЬ ТАК: это ВЫБОР РАЗМЕРНОСТИ на val, а не свидетельство, "
          "что обученная\n  голова улучшит успех. Доля улучшения посчитана по "
          "ИСТИННОМУ остатку,\n  спроецированному на базис, то есть это "
          "ВЕРХНЯЯ граница для ранга r,\n  обученной головой недостижимая.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--diagnose", default=None)
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", default="data/k9_teacher_150k.npz")
    ap.add_argument("--q0-source", choices=Q0_SOURCES, default=None)
    ap.add_argument("--joint-ckpt", default=None)
    ap.add_argument("--readout", default=None)
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--diag-n", type=int, default=20000)
    ap.add_argument("--rho-pct", type=float, default=95.0)
    ap.add_argument("--dataset-revision", default=DATASET_REV,
                    help="ревизия датасета; обязана совпасть с той, на "
                         "которой собран манифест")
    ap.add_argument("--codebook-tol", type=float, default=1e-5,
                    help="допуск сверки книг кэша с текущим кодеком")
    ap.add_argument("--drift-n", type=int, default=2000)
    # ОТВОДЫ СНИМАЮТСЯ ВСЕ, СОХРАНЯЮТСЯ НЕ ВСЕ. Голове HiCoRA-D нужен только
    # h24: черновик берётся из уже сохранённых кодов q0hat, а h18 понадобится
    # лишь будущей HiCoRA-V. Каждый лишний отвод — это N*16*D*2 байт на диске,
    # и хранить их «на всякий случай» стоило бы втрое дороже.
    ap.add_argument("--save-taps", default="12,24",
                    help="какие отводы писать на диск через запятую; проход "
                         "всё равно полный. Для HiCoRA-D достаточно 24, "
                         "12 нужен для проверки шума fp16 и тождеств K-11b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    save_taps = tuple(int(x) for x in str(args.save_taps).split(",") if x)
    bad = [t for t in save_taps if t not in TAPS]
    if bad:
        raise SystemExit(f"отводы {bad} вне {TAPS}")
    if max(TAPS) not in save_taps:
        raise SystemExit(f"отвод {max(TAPS)} обязателен: именно его читает "
                         f"голова поправки")

    # ПУТИ АБСОЛЮТИЗИРУЮТСЯ СРАЗУ. Прежняя версия делала chdir в каталог
    # actioncodec, и все относительные пути начинали разрешаться от него:
    # `data/k9_teacher_150k.npz` искался в third_party/actioncodec/data.
    # chdir убран совсем — K-9e работает без него.
    for f in ("cache", "joint_ckpt", "readout", "out", "diagnose"):
        v = getattr(args, f)
        if v:
            setattr(args, f, os.path.abspath(v))
    args.root = os.path.abspath(args.root)
    sys.path.insert(0, args.root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sha = file_sha1(__file__)
    print(f"k11a sha1 {sha}")

    if args.diagnose:
        if not args.ckpt:
            raise SystemExit("диагностике нужен --ckpt: доля улучшения "
                             "считается ПОСЛЕ декодера")
        diagnose(args)
        return

    for need, why in ((args.ckpt, "--ckpt"), (args.out, "--out"),
                      (args.q0_source, "--q0-source")):
        if not need:
            raise SystemExit(f"нужен {why}")
    if args.q0_source == "joint12" and not args.joint_ckpt:
        raise SystemExit("источник joint12 требует --joint-ckpt")
    if args.q0_source == "readout" and not args.readout:
        raise SystemExit("источник readout требует --readout")
    if args.q0_source == "coarse24":
        raise SystemExit(
            "источник coarse24 несовместим с HiCoRA: q0 берётся с последнего "
            "слоя, и поправку строить не на чем. Он существует только как "
            "верхняя граница качества q0 в отдельных сравнениях")

    import copy
    import torch
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (STATE_Q01, STATE_Q99, VisionLanguageActionProcessor,
                       dict_apply, get_cfg, process_state, prompt_template,
                       seed_everything)
    from joint12_vla import make_joint12_class
    import joint12_vla as jv
    import hicora_vla as hv

    seed_everything(0)
    dev, dt = torch.device(args.device), getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(args.root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt

    d = np.load(args.cache, allow_pickle=True)
    N = len(d["episode"]) if args.limit is None else min(args.limit,
                                                         len(d["episode"]))
    epi, stp = d["episode"][:N], d["step"][:N]
    if len(set(zip(epi.tolist(), stp.tolist()))) != N:
        raise SystemExit("ключи (episode, step) в исходном кэше не уникальны")
    tsk, offs = d["task"][:N], d["pos_offset"][:N].astype(np.int64)
    split = np.asarray(d["split"][:N]).astype(str)
    # ДИАПАЗОН ПРОВЕРЯЕТСЯ ДО ПРИВЕДЕНИЯ ТИПА: код 40000 при int16 стал бы
    # отрицательным и прошёл бы проверку «>= 0» после каста.
    Ktrue_raw = d["K_true"][:N]
    if int(Ktrue_raw.min()) < 0:
        raise SystemExit(f"в K_true отрицательный код {int(Ktrue_raw.min())}")
    if int(Ktrue_raw.max()) >= 2 ** 15:
        raise SystemExit(f"код {int(Ktrue_raw.max())} не помещается в int16")
    Ktrue = Ktrue_raw.astype(np.int16)
    cache_meta = json.loads(str(d["meta"]))
    print(f"кэш K-9a: {len(d['episode'])} наблюдений, берём {N}")
    if cache_meta.get("ckpt") != args.ckpt:
        raise SystemExit(
            f"кэш собран чекпойнтом {cache_meta.get('ckpt')}, а сейчас "
            f"{args.ckpt}: коды K_true и черновик относились бы к разным "
            f"моделям")
    if tuple(Ktrue.shape[1:]) != (N_LEVEL, N_POS):
        raise SystemExit(f"K_true формы {Ktrue.shape}, ожидалось "
                         f"(N, {N_LEVEL}, {N_POS})")

    # ПРОВЕРКИ, А НЕ ПЕЧАТЬ. Прежняя версия просто выводила манифест, сид и
    # словарь: отчёт утверждал, что они проверены, а код их только показывал.
    # Производитель кэша обязан отказаться ДО часового прогона.
    def need_eq(got, want, what):
        if got != want:
            raise SystemExit(f"{what}: {got!r} против {want!r} в meta кэша")

    need_eq(int(cache_meta["n_obs"]), len(d["episode"]), "число наблюдений")
    vals = set(np.unique(split).tolist())
    if not vals <= {"train", "val", "test"} or not vals:
        raise SystemExit(f"недопустимые значения split: {sorted(vals)}")
    print(f"  происхождение кэша: манифест {cache_meta.get('manifest')}, "
          f"сид разбиения {cache_meta.get('split_seed')}, словарь "
          f"{cache_meta.get('vocab')}, части {sorted(vals)}")

    # --- МАНИФЕСТ СВЕРЯЕТСЯ ДО ВСЕГО ДОРОГОГО -----------------------------
    # Раньше эта проверка стояла ПОСЛЕ загрузки состояний из parquet и после
    # создания модели, то есть отказ наступал минут через сорок. Здесь она
    # первая после чтения npz.
    man_path = cache_meta.get("manifest")
    if not man_path or not os.path.exists(man_path):
        raise SystemExit(
            f"манифест {man_path!r} из meta кэша не найден. Сборка без сверки "
            f"списка эпизодов и разбиения запрещена: часовой прогон дал бы "
            f"кэш, происхождение которого нечем подтвердить")
    man_info = check_manifest(json.load(open(man_path)), cache_meta, epi,
                              split, rev=args.dataset_revision)
    man_info.update(path=man_path, sha1=file_sha1(man_path))
    print(f"  манифест сверен: {man_info['n_episodes']} эпизодов, сид "
          f"{man_info['split_seed']}, ревизия "
          f"{man_info['dataset_revision']}, репозиторий "
          f"{man_info['dataset_repo']}, sha {man_info['sha1']}; части "
          f"совпали у всех {man_info['n_checked']} эпизодов кэша")

    # --- кадры: ИМЯ ФАЙЛА ТОЧНО КАК В K-9a ---------------------------------
    # K-9a пишет `<cache>.images.npy`, где <cache> уже содержит `.npz`.
    # Замена расширения давала бы несуществующий путь.
    img_path = args.cache + ".images.npy"
    if not os.path.exists(img_path):
        raise SystemExit(f"нет {img_path}: сборка кадров из parquet в память "
                         f"на {N} наблюдений уже убивала прогон")
    IMG = np.load(img_path, mmap_mode="r")
    if IMG.shape[0] != len(d["episode"]):
        raise SystemExit(f"кадров {IMG.shape[0]}, наблюдений "
                         f"{len(d['episode'])}")
    if cache_meta.get("images_file") != os.path.basename(img_path):
        raise SystemExit(f"meta называет кадры "
                         f"{cache_meta.get('images_file')!r}, а открыт "
                         f"{os.path.basename(img_path)!r}")
    if IMG.dtype != np.uint8:
        raise SystemExit(f"кадры типа {IMG.dtype}, ожидался uint8: "
                         f"нормировка была бы применена дважды")
    want_shape = tuple(cache_meta.get("image_shape") or ())
    if want_shape and tuple(IMG.shape[1:]) != want_shape:
        raise SystemExit(f"кадры формы {tuple(IMG.shape[1:])} против "
                         f"{want_shape} в meta")
    print(f"  кадры: {IMG.shape}, {IMG.dtype}, {IMG.nbytes / 2 ** 30:.1f} "
          f"ГиБ, имя совпало с meta")

    # --- состояния: ИЗ PARQUET, в NPZ их нет -------------------------------
    st = None
    rid, rev = "physical-intelligence/libero", args.dataset_revision
    uniq = np.unique(epi)
    for j, e in enumerate(uniq):
        f = hf_hub_download(rid, f"data/chunk-{int(e) // 1000:03d}/"
                            f"episode_{int(e):06d}.parquet",
                            repo_type="dataset", revision=rev)
        t = pq.read_table(f)
        S_ = np.asarray(t.column("state").to_pylist(), np.float32)
        if st is None:
            st = np.zeros((N, S_.shape[1]), np.float64)
        elif st.shape[1] != S_.shape[1]:
            raise SystemExit(f"эпизод {e}: состояние {S_.shape[1]}-мерное, "
                             f"раньше было {st.shape[1]}-мерное")
        for r_ in np.where(epi == e)[0]:
            st[r_] = S_[int(stp[r_])]
        if j % 400 == 0:
            print(f"  эпизодов {j}/{len(uniq)}", flush=True)
    if st.shape[1] == len(STATE_Q01) + 1:
        st = process_state(st)
    if not np.isfinite(st).all():
        raise SystemExit("в состояниях есть nan или inf")
    st_n = (st - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0
    print("  состояния собраны")

    # --- модель -------------------------------------------------------------
    Cls = make_joint12_class(SmolVLABlockwiseAR)
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    model.init_joint_fast(depth=args.depth, head_dtype=dt)

    ac = proc.action_processor
    codec = ac if hasattr(ac, "vq") else getattr(ac, "codec", None)
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь")
    codec = codec.to(dev).eval()
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float()
    V = int(codec.vocab_size)
    if int(cache_meta.get("vocab", V)) != V:
        raise SystemExit(f"словарь кэша {cache_meta.get('vocab')} против "
                         f"{V} у кодека")
    if int(Ktrue_raw.max()) >= V:
        raise SystemExit(f"в K_true код {int(Ktrue_raw.max())} при словаре "
                         f"{V}")
    if int(E.shape[0]) != N_LEVEL:
        raise SystemExit(f"уровней в кодеке {int(E.shape[0])}, ожидалось "
                         f"{N_LEVEL}")

    # ИСХОДНАЯ ФИНАЛЬНАЯ НОРМА СНИМАЕТСЯ ДО НАЛОЖЕНИЯ ВЕСОВ: чекпойнт Joint12
    # перезапишет `action_expert.norm`, обученную читать h12, а поздняя ветвь
    # обязана читать h24 своей нормой.
    res_norm = copy.deepcopy(model.action_expert.norm)

    if args.q0_source == "joint12":
        wsha = file_sha1(args.joint_ckpt)
        obj = torch.load(args.joint_ckpt, map_location="cpu",
                         weights_only=False)
        if int(obj.get("depth", -1)) != args.depth:
            raise SystemExit(f"чекпойнт глубины {obj.get('depth')}, задано "
                             f"{args.depth}")
        state = obj["state"]
        # СТРОГО, КАК В K-9d/K-9e: частично применённый чекпойнт даёт
        # правдоподобные, но неверные числа, и таблица этого не покажет.
        stray = [k for k in state
                 if not any(k.startswith(p) or k == p.rstrip(".")
                            for p in model.trainable_prefixes)]
        if stray:
            raise SystemExit(f"{len(stray)} ключей вне белого списка: "
                             f"{stray[:5]}")
        own = dict(model.named_parameters())
        missing = [k for k in own if own[k].requires_grad and k not in state]
        if missing:
            raise SystemExit(f"нет {len(missing)} обучаемых весов: "
                             f"{missing[:5]}")
        with torch.no_grad():
            for k, v in state.items():
                if tuple(own[k].shape) != tuple(v.shape):
                    raise SystemExit(f"форма {k}")
                if not torch.isfinite(v).all():
                    raise SystemExit(f"в {k} есть nan или inf")
                # FP32, А НЕ dtype МОДЕЛИ. Прежняя версия писала
                # `own[k].dtype`, то есть fp16, и округляла обученные веса:
                # исполнялась бы ДРУГАЯ модель, не та, у которой измерены
                # 89.5%.
                own[k].data = v.to(dev, torch.float32)
        touched = sorted({k.split(".layers.")[0] if ".layers." in k
                          else k.rsplit(".", 1)[0] for k in state})
        src_meta = dict(path=args.joint_ckpt, depth=int(obj["depth"]),
                        tensors=len(state), touched=touched,
                        weights_sha1=wsha)
        print(f"  веса Joint12: {len(state)} тензоров, файл sha {wsha}")
        print(f"  ОБЩИЕ ДЛЯ ВСЕЙ ГЛУБИНЫ И ПЕРЕЗАПИСАННЫЕ: "
              f"{[t for t in touched if 'layers' not in t]}")
    else:
        wsha = file_sha1(args.readout)
        obj = torch.load(args.readout, map_location="cpu", weights_only=False)
        state = {k: v for k, v in obj["state"].items()
                 if k.startswith(("fast_head.", "action_expert.norm."))}
        if not state:
            raise SystemExit(f"{args.readout}: нет весов головы-читалки")
        own = dict(model.named_parameters())
        with torch.no_grad():
            for k, v in state.items():
                own[k].data = v.to(dev, torch.float32)
        src_meta = dict(path=args.readout, tensors=len(state),
                        weights_sha1=wsha, touched=sorted(state),
                        note="ствол ИСХОДНЫЙ, поздние слои в своём "
                             "распределении")
        print(f"  голова-читалка: {len(state)} тензоров, файл sha {wsha}")

    # РЕЖИМ ВЫЧИСЛЕНИЯ КАК В K-9c/K-9e: обучаемое в fp32, проход под autocast.
    n32 = model.to_fp32_trainable()
    print(f"  режим как в K-9c: {n32} тензоров в fp32, проход под autocast "
          f"{args.dtype}")
    model.eval()

    model.__class__ = hv.make_hicora_class(type(model))
    model.set_codebooks(E)
    model.set_res_norm(res_norm.to(dev))
    model.taps, model.q0_depth = TAPS, args.depth
    model.n_layers_total = len(model.action_expert.layers)
    D_H, D_Z = int(model.fast_head.in_features), int(E.shape[-1])
    if max(TAPS) != model.n_layers_total:
        raise SystemExit(f"последний отвод {max(TAPS)} против "
                         f"{model.n_layers_total} слоёв: проход неполный")
    print(f"  отводы {TAPS}, h={D_H}, латент={D_Z}, слоёв "
          f"{model.n_layers_total}")

    outdir = os.path.dirname(args.out) or "."
    os.makedirs(outdir, exist_ok=True)
    need = len(save_taps) * N * N_POS * D_H * 2 / 2 ** 30
    stfs = os.statvfs(outdir)
    free = stfs.f_bavail * stfs.f_frsize / 2 ** 30
    print(f"  сохраняются отводы {save_taps} (проход полный, снимаются все "
          f"{TAPS}): {len(save_taps)} x ({N}, {N_POS}, {D_H}) fp16 = "
          f"{need:.2f} ГиБ, свободно {free:.1f} ГиБ")
    if free < need * 1.15:
        raise SystemExit(
            f"места не хватит с запасом: нужно {need * 1.15:.1f} ГиБ, есть "
            f"{free:.1f}. Уменьшите --save-taps до 24, возьмите --limit или "
            f"освободите диск — падение на последнем батче стоило бы всего "
            f"прогона")

    taps_mm = {t: np.lib.format.open_memmap(
        f"{args.out}.h{t}.npy", mode="w+", dtype=np.float16,
        shape=(N, N_POS, D_H)) for t in save_taps}
    q0hat = np.zeros((N, N_POS), np.int16)

    groups = []
    for po in sorted({int(v) for v in offs}):
        ipo = np.where(offs == po)[0]
        for i, j in plan_batches(len(ipo), args.batch):
            groups.append((po, ipo[i:j]))
    print(f"  батчей {len(groups)} по {args.batch}; порядок по офсету — "
          f"position_offset задаётся на весь вызов")

    def build(sel):
        image = torch.from_numpy(np.asarray(IMG[sel]))
        msgs = []
        for gi in sel:
            m = prompt_template(
                st_n[gi], None, str(tsk[gi]),
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        b = proc(text=texts, images=[[image[k].numpy()]
                                     for k in range(len(sel))],
                 return_tensors="pt", padding=True, padding_side="left",
                 action_processor_kwargs={"embodiment_ids": 0})
        return dict_apply(lambda x: x.to(dev, dt), b)

    seen = set()
    for gi, (po, sel) in enumerate(groups):
        b = build(sel)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=dt):
            v, pp = model.build_inputs(position_offset=po, **b)
            tp = model.forward_taps(
                vlm_inputs_embeds=v, attention_mask=b.get("attention_mask"),
                position_ids=pp)
            _, q0 = model.q0_from(tp[args.depth])
        seen.add(int(tp["layers_run"]))
        for t in save_taps:
            taps_mm[t][sel] = tp[t].float().cpu().numpy().astype(np.float16)
        q0hat[sel] = q0.cpu().numpy().astype(np.int16)
        if gi % 50 == 0:
            print(f"    батч {gi}/{len(groups)}", flush=True)
    for t in save_taps:
        taps_mm[t].flush()
    if seen != {model.n_layers_total}:
        raise SystemExit(f"глубина прохода {sorted(seen)} вместо "
                         f"{model.n_layers_total}")
    print(f"  проход: ровно {model.n_layers_total} слоёв во всех батчах")

    # --- ШУМ ХРАНЕНИЯ FP16 ИЗМЕРЯЕТСЯ, А НЕ ПРЕДПОЛАГАЕТСЯ -----------------
    if args.depth not in save_taps:
        raise SystemExit(
            f"отвод {args.depth} не сохраняется, а без него нечем измерить "
            f"шум fp16 на черновике: добавьте его в --save-taps")
    chk = np.random.default_rng(0).choice(N, min(2048, N), replace=False)
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=dt):
        hb = torch.from_numpy(np.asarray(taps_mm[args.depth][chk])).to(dev, dt)
        _, qb = model.q0_from(hb)
    mism = float((qb.cpu().numpy().astype(np.int16) != q0hat[chk]).mean())
    print(f"  шум fp16: {mism:.3%} токенов q0 расходятся при перегоне из кэша")
    if mism > 0.005:
        raise SystemExit(f"расхождение {mism:.3%} выше 0.5%: кэш непригоден "
                         f"как источник черновика")

    # --- ЧЕТЫРЕ РАЗНЫЕ ВЕЛИЧИНЫ ДРЕЙФА -------------------------------------
    drift = None
    if args.q0_source == "joint12" and args.drift_n > 0:
        print(f"\n  дрейф на {args.drift_n} наблюдениях ЧИСТЫМ проходом")
        clean = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
        clean.init_joint_fast(depth=args.depth, head_dtype=dt)
        clean_norm = copy.deepcopy(clean.action_expert.norm).to(dev)
        clean.__class__ = hv.make_hicora_class(type(clean))
        clean.set_codebooks(E)
        clean.set_res_norm(clean_norm)
        clean.taps, clean.q0_depth = TAPS, args.depth
        clean.n_layers_total = len(clean.action_expert.layers)
        sub = np.random.default_rng(1).choice(N, min(args.drift_n, N),
                                              replace=False)
        rel, ag12, agc, ntok = [], 0, 0, 0
        # СУММЫ, А НЕ СРЕДНЕЕ RMS ПО БАТЧАМ — та же поправка, что в основной
        # диагностике; здесь она была пропущена.
        acc_d = {}
        for po in sorted({int(offs[i]) for i in sub}):
            ipo = np.asarray([i for i in sub if int(offs[i]) == po])
            for i, j in plan_batches(len(ipo), args.batch):
                s_ = ipo[i:j]
                b = build(s_)
                with torch.no_grad(), torch.autocast(device_type=dev.type,
                                                     dtype=dt):
                    v, pp = clean.build_inputs(position_offset=po, **b)
                    tp = clean.forward_taps(
                        vlm_inputs_embeds=v,
                        attention_mask=b.get("attention_mask"),
                        position_ids=pp)
                    _, q12 = clean.q0_from(tp[args.depth])
                    # НАСТОЯЩИЙ coarse24 — это generate на полной глубине, а
                    # не голова-читалка на отводе. Именно у него измерены
                    # 90.0%, и сравнивать надо с ним.
                    toks = clean.generate(**b, position_offset=po,
                                          do_sample=False,
                                          initial_position_shift=1)
                    qc = toks.cpu().numpy().reshape(
                        -1, N_LEVEL, N_POS)[:, 0, :].astype(np.int16)
                a = tp[max(TAPS)].float().cpu().numpy()
                bb = np.asarray(taps_mm[max(TAPS)][s_], np.float32)
                rel.append((np.linalg.norm(bb - a, axis=-1)
                            / np.maximum(np.linalg.norm(a, axis=-1),
                                         1e-6)).ravel())
                ag12 += int((q12.cpu().numpy().astype(np.int16)
                             == q0hat[s_]).sum())
                agc += int((qc == q0hat[s_]).sum())
                ntok += int(q0hat[s_].size)
                with torch.no_grad():
                    Aj = codec._decode(
                        E[0][torch.as_tensor(q0hat[s_]).long().to(dev)],
                        embodiment_ids=0)[0][..., :7].float().cpu().numpy()
                    Ac = codec._decode(
                        E[0][torch.as_tensor(qc).long().to(dev)],
                        embodiment_ids=0)[0][..., :7].float().cpu().numpy()
                err_add(acc_d, err_sums(Aj, Ac))
        rel = np.concatenate(rel)
        e_d = err_finish(acc_d)
        drift = dict(n=int(len(sub)), h24_rel_mean=float(rel.mean()),
                     h24_rel_p95=float(np.percentile(rel, 95)),
                     q0_vs_clean_h12=float(ag12 / max(ntok, 1)),
                     q0_vs_coarse24=float(agc / max(ntok, 1)),
                     act_pos=e_d["pos"], act_rot=e_d["rot"],
                     act_grip=e_d["grip"])
        print(f"    1. q0 Joint12 против исходной головы на h12: "
              f"{drift['q0_vs_clean_h12']:.1%} совпадений")
        print(f"    2. q0 Joint12 против НАСТОЯЩЕГО coarse24 (generate, "
              f"{model.n_layers_total} слоёв): {drift['q0_vs_coarse24']:.1%}")
        print(f"    3. декодированные действия, шаги 0-7: положение "
              f"{drift['act_pos']:.5f}, вращение {drift['act_rot']:.5f}, "
              f"знак схвата {drift['act_grip']:.1%}")
        print(f"    4. дрейф h24: относительный {drift['h24_rel_mean']:.3f}, "
              f"p95 {drift['h24_rel_p95']:.3f}")
        print("    ЧИТАТЬ ТАК: 1 и 2 — РАЗНЫЕ величины. Первая о том, что "
              "изменило\n    дообучение головы, вторая о расстоянии до опоры "
              "с известными 90.0%.\n    Большой дрейф h24 означает, что слои "
              "13-24 читают вход вне своего\n    распределения — это не "
              "запрет на HiCoRA, но объяснение на случай,\n    если поправка "
              "окажется бесполезной, и повод сравнить с источником readout.")
        del clean
        torch.cuda.empty_cache()

    np.save(f"{args.out}.q0hat.npy", q0hat)
    np.save(f"{args.out}.ktrue.npy", Ktrue)
    np.save(f"{args.out}.split.npy", split)
    np.save(f"{args.out}.codebooks.npy", E.cpu().numpy().astype(np.float32))
    meta = dict(n_obs=int(N), q0_source=args.q0_source, taps=list(TAPS),
                saved_taps=list(save_taps),
                depth=args.depth, d_hidden=D_H, d_latent=D_Z, ckpt=args.ckpt,
                cache=args.cache, source=src_meta, fp16_q0_mismatch=mism,
                drift=drift, script_sha1=sha, cache_meta=cache_meta,
                manifest=man_info, vocab=V,
                dataset_revision=args.dataset_revision,
                codebooks_sha1=hashlib.sha1(np.ascontiguousarray(
                    E.cpu().numpy().astype(np.float32)).tobytes()
                ).hexdigest()[:12],
                decoder_probe=decoder_probe(codec, E, dev),
                codec_state_sha1=state_sha1(codec),
                hicora_vla_sha1=file_sha1(hv.__file__),
                joint12_vla_sha1=file_sha1(jv.__file__),
                keys_sha1=hashlib.sha1(np.ascontiguousarray(
                    np.stack([epi, stp])).tobytes()).hexdigest()[:12])
    json.dump(meta, open(f"{args.out}.meta.json", "w"), ensure_ascii=False,
              indent=1)
    hs = ",".join(f"h{t}" for t in save_taps)
    print(f"\n  сохранено: {args.out}.{{{hs},q0hat,ktrue,split,codebooks}}"
          f".npy и .meta.json")
    print(f"  ключи sha {meta['keys_sha1']}")
    print("\n  ДАЛЬШЕ: --diagnose для выбора ранга и rho, затем K-11b с "
          "проверками\n  тождества. Обучение головы не начинать до "
          "тождества.")


if __name__ == "__main__":
    main()
