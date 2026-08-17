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
    C = tuple(q for q in range(P) if lab["support"][i] >> q & 1)
    subs = [S for k in range(kmax + 1) for S in itertools.combinations(C, k)]
    v = lab["g_flat"][lab["g_off"][i]:lab["g_off"][i + 1]]
    assert len(v) == len(subs), f"строка {i}: таблица {len(v)} против {len(subs)}"
    return {S: float(v[j]) for j, S in enumerate(subs)}, set(C)


def g_of(gmap, C, S):
    """G произвольного набора: сводим к пересечению с support."""
    return gmap[tuple(sorted(set(S) & C))]


def to_rms(e0_mse, g_mse):
    return np.sqrt(e0_mse) - np.sqrt(max(e0_mse - g_mse, 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n-check", type=int, default=2000,
                    help="на скольких строках считать оракульные числа")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.dir, "metadata.json")))
    ft = np.load(os.path.join(args.dir, "features.npz"), allow_pickle=True)
    lb = np.load(os.path.join(args.dir, "labels.npz"), allow_pickle=True)
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
    bad = [k for k in ft.files if any(s in k.lower() for s in FORBIDDEN)]
    assert not bad, f"запрещённые ключи в признаках: {bad}"
    print(f"  1. признаков {len(ft.files)}, запрещённых ключей нет")
    assert set(ft.files) == set(meta["feature_keys"])
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
        gmap, C = make_gmap(lb, int(i), P, kmax)
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
    for s_, nm in ((None, "все"), (0, "train"), (1, "val"), (2, "test")):
        m = np.ones(n, bool) if s_ is None else (sp == s_)
        ii = idx[m[idx]]
        if len(ii) < 20:
            continue
        base = np.sqrt(e0[ii]).sum()
        ex = lb["best_gain_by_k_rms"][ii, kmax].sum() / base
        gr = np.zeros(len(ii))
        sg = np.zeros(len(ii))
        for j, i in enumerate(ii):
            gmap, C = make_gmap(lb, int(i), P, kmax)
            path = [q for q in lb["add_path"][i] if q >= 0]
            gr[j] = to_rms(e0[i], g_of(gmap, C, path))
            top4 = np.argsort(-lb["sing_gain_rms"][i])[:kmax]
            sg[j] = to_rms(e0[i], g_of(gmap, C, tuple(top4)))
        print(f"  {nm:>6} (n={len(ii):>5}): точный <=4 {ex:.3f}   "
              f"жадный {gr.sum() / base:.3f}   одиночный top-4 "
              f"{sg.sum() / base:.3f}")
    print("  фаза A на 96 наблюдениях давала  0.93 / 0.91 / 0.79")

    print("\n" + "=" * 74)
    print("ПЛАНКА ДЛЯ B1: ПРИЧИННЫЕ BASELINE НА ЭТОМ ЖЕ ДАТАСЕТЕ")
    print("=" * 74)
    print("  Числа фазы A (0.40 / 0.79 / 0.91) получены на ДРУГОЙ выборке и с\n"
          "  другой длиной паддинга, поэтому переносить их нельзя — router\n"
          "  сравнивается с тем, что измерено здесь.")
    ent, mrg = ft["cand_entropy"], ft["cand_margin"]
    pcol = lb["p"]
    rs = np.random.default_rng(1)
    for s_, nm in ((2, "test"), (1, "val")):
        ii = idx[(sp == s_)[idx]]
        if len(ii) < 20:
            continue
        base = np.sqrt(e0[ii]).sum()
        print(f"\n  {nm} (n={len(ii)}), доля закрытого разрыва:")
        print(f"  {'способ отбора':>28}" + "".join(f"{f'K={K}':>10}"
                                                   for K in (1, 2, 4)))
        rows = {}
        for K in (1, 2, 4):
            acc = {k: 0.0 for k in
                   ("энтропия (прич.)", "малый запас (прич.)",
                    "только p (прич.)", "окно вокруг p (прич.)",
                    "случайно, 20 сид. (прич.)", "одиночный оракул",
                    "жадный оракул", "точный оракул <=K")}
            for i in ii:
                gmap, C = make_gmap(lb, int(i), P, kmax)
                pp = int(pcol[i])
                acc["энтропия (прич.)"] += to_rms(
                    e0[i], g_of(gmap, C, np.argsort(-ent[i])[:K]))
                acc["малый запас (прич.)"] += to_rms(
                    e0[i], g_of(gmap, C, np.argsort(mrg[i])[:K]))
                acc["только p (прич.)"] += to_rms(e0[i], g_of(gmap, C, (pp,)))
                w0 = min(max(pp - K // 2, 0), P - K)
                acc["окно вокруг p (прич.)"] += to_rms(
                    e0[i], g_of(gmap, C, range(w0, w0 + K)))
                r = np.mean([g_of(gmap, C, rs.choice(P, K, replace=False))
                             for _ in range(20)])
                acc["случайно, 20 сид. (прич.)"] += to_rms(e0[i], r)
                acc["одиночный оракул"] += to_rms(
                    e0[i], g_of(gmap, C, np.argsort(-lb["sing_gain_rms"][i])[:K]))
                acc["жадный оракул"] += to_rms(
                    e0[i], g_of(gmap, C,
                                [q for q in lb["add_path"][i][:K] if q >= 0]))
                acc["точный оракул <=K"] += lb["best_gain_by_k_rms"][i, K]
            for k, v in acc.items():
                rows.setdefault(k, {})[K] = v / base
        for k, r in rows.items():
            print(f"  {k:>28}" + "".join(f"{r[K]:>10.3f}" for K in (1, 2, 4)))

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
    sz = lb["best_size_by_k"][:, kmax]
    print("\n  размер лучшего набора при бюджете "
          f"{kmax}: " + " ".join(f"{i}:{(sz == i).mean():.0%}"
                                 for i in range(kmax + 1)))
    print(f"  обратимые траектории с REMOVE: "
          f"{(lb['rev_action'] == 0).any(1).mean():.2%}, "
          f"средняя длина {lb['rev_len'].mean():.2f}")
    print(f"  средняя длина сжатой таблицы: {np.diff(lb['g_off']).mean():.1f}")

    print("\nвсе проверки пройдены")


if __name__ == "__main__":
    main()
