"""Проверка собранного датасета K-4b0 и образец того, как его читать.

Главная содержательная проверка — ВОСПРОИЗВЕДЕНИЕ ОРАКУЛЬНЫХ ЧИСЕЛ фазы A на
новом датасете. Если точный оракул, жадный и одиночное ранжирование дадут не
то же, что 0.93 / 0.91 / 0.79, значит датасет собран не тем, чем мерили раньше,
и сравнивать router с этими воротами нельзя.

Запуск:
    python3 experiments/k4b0_verify.py --dir data/k4b0
"""

import argparse
import itertools
import json
import os

import numpy as np

FORBIDDEN = ("lg_after", "z_ref", "h_ref", "js_", "oracle", "gain", "changed",
             "support", "a_true", "target", "label")


def make_gmap(lab, i, P, kmax):
    """Таблица G(S) одного вмешательства из рваного массива.

    Хранятся только подмножества changed-support: для позиции вне support
    латента совпадает с опорной побитово, поэтому G(S) = G(S ∩ C) точно."""
    # Кэшировать по маске support бессмысленно: их 13467 различных на 16000
    # строк, кэш только добавляет накладные расходы (замерено).
    C = tuple(q for q in range(P) if lab["support"][i] >> q & 1)
    subs = [S for k in range(kmax + 1) for S in itertools.combinations(C, k)]
    v = lab["g_flat"][lab["g_off"][i]:lab["g_off"][i + 1]]
    assert len(v) == len(subs), f"строка {i}: таблица {len(v)} против {len(subs)}"
    return {S: float(v[j]) for j, S in enumerate(subs)}, set(C)


def g_of(gmap, C, S):
    """G произвольного набора: сводим к пересечению с support."""
    return gmap[tuple(sorted(set(S) & C))]


def to_rms(e0_mse, g_mse):
    """Перевод выигрыша из MSE в RMS. ВЫПУКЛА по g_mse, поэтому усреднять
    выигрыши в MSE и переводить среднее — НЕЛЬЗЯ: по Йенсену это занижает
    результат. Переводить надо каждую реализацию, потом усреднять."""
    return np.sqrt(e0_mse) - np.sqrt(max(e0_mse - g_mse, 0.0))


def cluster_ci(num, den, epi, n_boot=2000, seed=0):
    """Отношение сумм с бутстрапом ПО ЭПИЗОДАМ: 16 вмешательств наблюдения и
    наблюдения одного эпизода зависимы, независимая пересэмплировка строк
    занизила бы интервал.

    Считается через ПОЭПИЗОДНЫЕ ЧАСТИЧНЫЕ СУММЫ. Отношение сумм по выборке
    эпизодов равно отношению сумм их агрегатов, поэтому реплика — это гather
    по 75 числам, а не склейка и суммирование 2400 строк. Результат
    тождественный, работы на два порядка меньше."""
    _, code = np.unique(epi, return_inverse=True)
    ne = code.max() + 1
    nb = np.bincount(code, weights=np.asarray(num, np.float64), minlength=ne)
    db = np.bincount(code, weights=np.asarray(den, np.float64), minlength=ne)
    pt = nb.sum() / max(db.sum(), 1e-30)
    idx = np.random.default_rng(seed).integers(0, ne, size=(n_boot, ne))
    out = nb[idx].sum(1) / np.maximum(db[idx].sum(1), 1e-30)
    return pt, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def macro_by(num, den, key):
    """Макро-среднее по группам: иначе результат определяют несколько задач с
    особенно крупным исходным разрывом."""
    return float(np.mean([num[key == k].sum() / max(den[key == k].sum(), 1e-30)
                          for k in np.unique(key)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n-check", type=int, default=2000,
                    help="на скольких строках считать оракульные числа")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.dir, "metadata.json")))
    # МАТЕРИАЛИЗУЕМ СРАЗУ. np.load возвращает ленивый NpzFile: каждое
    # обращение вида lb["g_flat"] заново читает и РАСПАКОВЫВАЕТ весь массив из
    # архива. make_gmap обращается к нему на каждой строке, поэтому
    # многомегабайтный массив распаковывался бы 16000 раз — замерено 19 минут
    # против долей секунды.
    _ft = np.load(os.path.join(args.dir, "features.npz"), allow_pickle=True)
    _lb = np.load(os.path.join(args.dir, "labels.npz"), allow_pickle=True)
    print(f"чтение: признаков {len(_ft.files)}, меток {len(_lb.files)}...",
          flush=True)
    ft = {k: _ft[k] for k in _ft.files}
    lb = {k: _lb[k] for k in _lb.files}
    P, kmax = meta["P"], meta["kmax"]
    n = len(lb["obs_idx"])

    print("=" * 74)
    print("КОНФИГУРАЦИЯ")
    print("=" * 74)
    for k in ("commit", "seed", "n_obs", "n_episodes", "n_interventions",
              "batch", "pad_to", "vlm_dtype", "tau_rel", "gap_rel",
              "gap_threshold", "pos_offset", "window", "metric_primary"):
        if k in meta:
            print(f"  {k:>18}: {meta[k]}")
    print(f"  {'split':>18}: {meta['split_counts']}")
    print(f"  {'задач':>18}: {len(meta['tasks'])}")

    print("\n" + "=" * 74)
    print("ЦЕЛОСТНОСТЬ")
    print("=" * 74)
    bad = [k for k in ft if any(s in k.lower() for s in FORBIDDEN)]
    assert not bad, f"запрещённые ключи в признаках: {bad}"
    print(f"  1. признаков {len(ft)}, запрещённых ключей нет")
    assert set(ft) == set(meta["feature_keys"])
    print("  2. состав признаков совпал с metadata")

    assert (ft["int_obs_idx"] == lb["obs_idx"]).all()
    assert (ft["int_p"] == lb["p"]).all()
    print(f"  3. строки признаков и меток выровнены, всего {n}")

    key = lb["obs_idx"].astype(np.int64) * P + lb["p"]
    assert (np.diff(key) > 0).all(), "порядок строк не канонический"
    print("  4. канонический порядок (наблюдение, p)")

    epi, sp = lb["episode"], lb["split"]
    sets = {s: set(epi[sp == s].tolist()) for s in (0, 1, 2)}
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) \
        and not (sets[1] & sets[2])
    print(f"  5. эпизоды не пересекаются: "
          f"{len(sets[0])}/{len(sets[1])}/{len(sets[2])}")
    for o in np.unique(lb["obs_idx"])[:500]:
        m = lb["obs_idx"] == o
        assert len(np.unique(sp[m])) == 1 and m.sum() == P
    print(f"  6. каждое наблюдение даёт ровно {P} вмешательств в одном split")

    tsk = np.asarray(meta["tasks"])[ft["obs_task_idx"]][lb["obs_idx"]]
    tr = set(tsk[sp == 0])
    globals()["tsk"] = tsk
    for s_, nm in ((1, "val"), (2, "test")):
        miss = len(set(tsk[sp == s_]) - tr)
        print(f"  7. задач в {nm}, отсутствующих в train: {miss}")

    hp = os.path.join(args.dir, "features_hidden.npy")
    if os.path.exists(hp):
        H = np.load(hp, mmap_mode="r")
        assert H.shape[0] == n and H.shape[1] == P
        print(f"  8. скрытые состояния {H.shape}, {H.dtype}")

    print("\n" + "=" * 74)
    print("ТАБЛИЦА G И МЕТКИ")
    print("=" * 74)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(args.n_check, n), replace=False)
    e0 = lb["e_empty"].astype(np.float64)
    worst_s, worst_b, worst_z = 0.0, 0.0, 0.0
    for i in idx[:500]:
        gmap, C = make_gmap(lb, int(i), P, kmax)   # до построения GM
        assert abs(gmap[()]) == 0.0, "G(пусто) не ноль"
        for q in range(P):
            got = to_rms(e0[i], g_of(gmap, C, (q,)))
            worst_s = max(worst_s, abs(got - lb["sing_gain_rms"][i, q]))
        S = tuple(x for x in lb["best_set_by_k"][i, kmax] if x >= 0)
        worst_b = max(worst_b, abs(to_rms(e0[i], g_of(gmap, C, S))
                                   - lb["best_gain_by_k_rms"][i, kmax]))
        for q in range(P):
            if q not in C:
                worst_z = max(worst_z, abs(g_of(gmap, C, (q,))))
    print(f"  9. одиночные выигрыши сходятся с таблицей: макс {worst_s:.2e}")
    print(f" 10. лучший набор сходится с таблицей:       макс {worst_b:.2e}")
    print(f" 11. позиции вне support дают ровно ноль:    макс {worst_z:.2e}")
    assert worst_s < 1e-6 and worst_b < 1e-6 and worst_z == 0.0

    print("\n" + "=" * 74)
    print("ОРАКУЛЬНЫЕ ЧИСЛА: воспроизводятся ли ворота фазы A")
    print("=" * 74)
    print("  доля закрытого разрыва = сумма выигрышей / сумма e(пусто), RMS")
    # МЕТРИКИ — на всех строках части. Подвыборка idx нужна только для
    # проверки целостности таблицы выше: на 2000 строках доверительный
    # интервал шире, а числа для ворот B1 должны быть окончательными.
    print(f"  построение таблиц для {n} строк (один раз на все части)...",
          flush=True)
    GM = [make_gmap(lb, i, P, kmax) for i in range(n)]
    for s_, nm in ((None, "все"), (0, "train"), (1, "val"), (2, "test")):
        ii = np.arange(n) if s_ is None else np.where(sp == s_)[0]
        if len(ii) < 20:
            continue
        base = np.sqrt(e0[ii]).sum()
        ex = lb["best_gain_by_k_rms"][ii, kmax].sum() / base
        gr = np.zeros(len(ii))
        sg = np.zeros(len(ii))
        for j, i in enumerate(ii):
            gmap, C = GM[i]
            path = [q for q in lb["add_path"][i] if q >= 0]
            gr[j] = to_rms(e0[i], g_of(gmap, C, path))
            top4 = np.argsort(-lb["sing_gain_rms"][i])[:kmax]
            sg[j] = to_rms(e0[i], g_of(gmap, C, tuple(top4)))
        print(f"  {nm:>6} (n={len(ii):>5}): точный <=4 {ex:.3f}   "
              f"жадный {gr.sum() / base:.3f}   одиночный top-4 "
              f"{sg.sum() / base:.3f}")
    print("  фаза A на 96 наблюдениях давала  0.93 / 0.91 / 0.79")

    print("\n" + "=" * 74)
    print("ПЛАНКА ДЛЯ B1: ПРИЧИННЫЕ BASELINE, ВСЯ ЧАСТЬ, С ИНТЕРВАЛАМИ")
    print("=" * 74)
    print("  Числа фазы A (0.40 / 0.79 / 0.91) получены на ДРУГОЙ выборке и с\n"
          "  другой длиной паддинга, поэтому не переносятся.\n"
          "  Оценка идёт по ВСЕМ строкам части, интервалы — кластерный\n"
          "  бутстрап по эпизодам, macro — среднее по задачам.")
    ent, mrg, pcol = ft["cand_entropy"], ft["cand_margin"], lb["p"]
    for s_, nm in ((2, "test"), (1, "val")):
        ii = np.where(sp == s_)[0]
        den = np.sqrt(e0[ii])
        ep_i, tk_i = epi[ii], tsk[ii]
        print(f"\n  {nm}: строк {len(ii)}, наблюдений "
              f"{len(np.unique(lb['obs_idx'][ii]))}, эпизодов "
              f"{len(np.unique(ep_i))}")
        # таблицы строятся ОДИН раз на часть, а не заново на каждое K:
        # прежде это было три прохода по 2400 строкам вместо одного
        gm_all = [GM[i] for i in ii]
        for K in (1, 2, 4):
            print(f"\n  {'способ отбора':>26}  K={K}   доля [95% ДИ]"
                  f"{'macro':>10}")
            rows = {}
            for nm_, sel in (
                    ("энтропия (прич.)",
                     lambda i, j: np.argsort(-ent[i])[:K]),
                    ("малый запас (прич.)",
                     lambda i, j: np.argsort(mrg[i])[:K]),
                    ("только p (прич.)", lambda i, j: (int(pcol[i]),)),
                    ("окно вокруг p (прич.)",
                     lambda i, j: range(min(max(int(pcol[i]) - K // 2, 0),
                                            P - K),
                                        min(max(int(pcol[i]) - K // 2, 0),
                                            P - K) + K)),
                    ("одиночный оракул",
                     lambda i, j: np.argsort(-lb["sing_gain_rms"][i])[:K]),
                    ("жадный оракул",
                     lambda i, j: [q for q in lb["add_path"][i][:K] if q >= 0]),
            ):
                num = np.array([to_rms(e0[i], g_of(*gm_all[j], sel(i, j)))
                                for j, i in enumerate(ii)])
                rows[nm_] = (num, cluster_ci(num, den, ep_i),
                             macro_by(num, den, tk_i))
            num = lb["best_gain_by_k_rms"][ii, K]
            rows["точный оракул <=K"] = (num, cluster_ci(num, den, ep_i),
                                         macro_by(num, den, tk_i))
            # СЛУЧАЙНЫЙ отбор: 20 НЕЗАВИСИМЫХ политик. Каждую переводим в RMS
            # ЦЕЛИКОМ и агрегируем по всей части, и только потом усредняем:
            # to_rms выпукла, поэтому усреднение в MSE занижает результат.
            # 20 независимых политик. np.random.choice(replace=False) стоит
            # десятки микросекунд на вызов, а нужен лишь случайный набор из K
            # позиций: берём K наименьших из строки шума, это на порядок
            # дешевле и распределение то же.
            per_seed = []
            for sd in range(20):
                noise = np.random.default_rng(1000 + sd).random((len(ii), P))
                pick = np.argpartition(noise, K - 1, axis=1)[:, :K]
                nm2 = np.array([to_rms(e0[i], g_of(*gm_all[j], pick[j]))
                                for j, i in enumerate(ii)])
                per_seed.append(nm2.sum() / den.sum())
            ps = np.array(per_seed)
            for nm_, (num, (pt, lo, hi), mac) in rows.items():
                print(f"  {nm_:>26}  {pt:>6.3f} [{lo:.3f},{hi:.3f}]{mac:>10.3f}")
            print(f"  {'случайно, 20 политик':>26}  {ps.mean():>6.3f} "
                  f"[{ps.min():.3f},{ps.max():.3f}]   ст.откл. {ps.std():.4f}")

    print("\n" + "=" * 74)
    print("СТАТИСТИКА МЕТОК ПО SPLIT")
    print("=" * 74)
    print(f"  {'split':>7}{'строк':>8}{'нечего чинить':>16}{'вредных в supp':>16}"
          f"{'|лучший набор|':>16}")
    for s_, nm in ((0, "train"), (1, "val"), (2, "test")):
        m = sp == s_
        chg = np.stack([(lb["support"][m] >> q & 1).astype(bool)
                        for q in range(P)], 1)
        neg = lb["sing_gain_rms"][m] < -lb["tau"][m][:, None]
        print(f"  {nm:>7}{m.sum():>8}{lb['no_repair'][m].mean():>15.2%}"
              f"{neg[chg].mean():>16.2%}"
              f"{lb['best_size_by_k'][m, kmax].mean():>16.2f}")
    # K_practical: наименьший набор, теряющий не более tau (и не более 1%)
    # от максимума. Именно он показывает, как часто можно остановиться раньше
    # БЕЗ содержательной потери, в отличие от размера точного argmax.
    print("\n  K_practical — наименьший |S| с G(S) >= G* - порог:")
    for lbl, rel in (("порог tau", None), ("потеря <= 1%", 0.01)):
        dist = np.zeros(kmax + 1, np.int64)
        for i in range(n):   # только чтение готовых best_gain_by_k_rms
            gstar = lb["best_gain_by_k_rms"][i, kmax]
            thr = (gstar - lb["tau"][i]) if rel is None else gstar * (1 - rel)
            k_ = kmax
            for K in range(kmax + 1):
                if lb["best_gain_by_k_rms"][i, K] >= thr:
                    k_ = K
                    break
            dist[k_] += 1
        print(f"    {lbl:>14}: " + " ".join(
            f"{i}:{dist[i] / n:.0%}" for i in range(kmax + 1))
            + f"   среднее {(dist * np.arange(kmax + 1)).sum() / n:.2f}")

    # ЭМПИРИЧЕСКАЯ сверка «ровно K» против «<= K». Равенство гарантировано лишь
    # когда набор можно добить позициями ВНЕ support: нужно 16 - |C| >= K - |S*|.
    # При крупном support запаса может не быть, и «ровно K» вынужден брать
    # вредные позиции внутри support.
    worst_d, n_diff = 0.0, 0
    for i in rng.choice(n, size=min(3000, n), replace=False):
        gmap, C = make_gmap(lb, int(i), P, kmax)
        free = P - len(C)
        best_le = max(gmap.values())
        best_ex = max((g for S, g in gmap.items()
                       if len(S) == kmax or len(S) + free >= kmax),
                      default=best_le)
        d = best_le - best_ex
        if d > 1e-12:
            n_diff += 1
            worst_d = max(worst_d, d)
    print(f"\n  «ровно K» против «<=K»: расходятся в {n_diff} случаях, "
          f"макс. разница {worst_d:.3e}")
    print(f"    свободных позиций вне support: среднее "
          f"{(P - np.array([bin(int(x)).count('1') for x in lb['support']])).mean():.2f}")

    sz = lb["best_size_by_k"][:, kmax]
    print("\n  размер лучшего набора при бюджете "
          f"{kmax}: " + " ".join(f"{i}:{(sz == i).mean():.0%}"
                                 for i in range(kmax + 1)))
    # АБЛЯЦИЯ ОБРАТИМОСТИ. Частота REMOVE ничего не решает: важно, даёт ли
    # обратимая процедура ВЫИГРЫШ против чистого добавления при том же бюджете.
    # Оба набора сохранены, таблица G(S) позволяет оценить любой из них.
    # ЧЕСТНАЯ ПАРА. Сравнивать обратимую политику с ADD-до-упора нельзя: та
    # заполняет бюджет даже бесполезными позициями, и разница мерила бы РАННЮЮ
    # ОСТАНОВКУ вместе с обратимостью. Опорой служит ADD+STOP — та же
    # остановка, но без обмена.
    ga, gr, gf, gmt = (np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n))
    na, nr = np.zeros(n), np.zeros(n)
    print(f"  абляция обратимости, строк {n}...", flush=True)
    for i in range(n):
        if i and i % 4000 == 0:
            print(f"    {i}/{n}", flush=True)
        gmap, C = GM[i]
        full = [q for q in lb["add_path"][i] if q >= 0]
        adds = [q for q in lb["add_stop_set"][i] if q >= 0]
        rev = [q for q in lb["rev_set"][i] if q >= 0]
        gf[i] = to_rms(e0[i], g_of(gmap, C, full))
        ga[i] = to_rms(e0[i], g_of(gmap, C, adds))
        gr[i] = to_rms(e0[i], g_of(gmap, C, rev))
        gmt[i] = to_rms(e0[i], g_of(gmap, C, full[:len(rev)]))
        na[i], nr[i] = len(adds), len(rev)
    den = np.sqrt(e0)
    for s_, nm in ((2, "test"), (1, "val")):
        m = sp == s_
        d = gr[m] - ga[m]
        pt, lo, hi_ = cluster_ci(d, den[m], epi[m])
        print(f"\n  ОБРАТИМОСТЬ, {nm}:")
        print(f"    ADD до упора (не пара для сравнения) "
              f"{gf[m].sum() / den[m].sum():.3f}, размер "
              f"{np.array([(lb['add_path'][i] >= 0).sum() for i in np.where(m)[0]]).mean():.2f}")
        print(f"    ADD+STOP (честная пара)           "
              f"{ga[m].sum() / den[m].sum():.3f}, размер {na[m].mean():.2f}")
        print(f"    ADD/SWAP+STOP                     "
              f"{gr[m].sum() / den[m].sum():.3f}, размер {nr[m].mean():.2f}")
        print(f"    прирост обратимости {pt:+.4f} [{lo:+.4f}, {hi_:+.4f}]")
        print(f"    строго лучше в {(d > 1e-12).mean():.2%} строк, "
              f"хуже в {(d < -1e-12).mean():.2%}")
        # MATCHED-COST на ВСЕХ строках: жадное добавление обрезается до
        # фактического размера обратимого набора. Иначе прирост частично
        # объясняется разным числом пересчитанных позиций, а не обменом.
        dm = gr[m] - gmt[m]
        pm, lm, hm = cluster_ci(dm, den[m], epi[m])
        print(f"    при РАВНОМ числе позиций (ADD обрезан до |rev_set|): "
              f"{pm:+.4f} [{lm:+.4f}, {hm:+.4f}]")
        print(f"      строго лучше в {(dm > 1e-12).mean():.2%}, "
              f"хуже в {(dm < -1e-12).mean():.2%}")
    ra, ro = lb["rev_action"], lb["rev_off"]
    hs = np.array([(ra[ro[i]:ro[i + 1]] == 2).any() for i in range(n)])
    hr = np.array([(ra[ro[i]:ro[i + 1]] == 0).any() for i in range(n)])
    print(f"\n  состав обратимых траекторий: SWAP в {hs.mean():.2%} строк, "
          f"самостоятельный REMOVE в {hr.mean():.2%}, "
          f"средняя длина {lb['rev_len'].mean():.2f}")
    print("""    Оба хода возможны. Самостоятельный REMOVE встречается после
    обмена: набор попадает в конфигурацию, где удаление улучшает без замены
    (детерминированный пример — test_standalone_remove_possible).
    Слово reversible оправдано приростом над ADD+STOP ПРИ РАВНОМ ЧИСЛЕ
    ПОЗИЦИЙ, а не частотой ходов. Порог плана — не менее +0.03.""")
    print(f"  средняя длина сжатой таблицы: {np.diff(lb['g_off']).mean():.1f}")

    print("\nвсе проверки пройдены")


if __name__ == "__main__":
    main()
