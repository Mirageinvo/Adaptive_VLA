#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Совпадает ли состав частей из `split.npy` с тем, что лежит в кэше целей.

ЗАЧЕМ. K-14d раньше брал части из `q1_cache`, а тот по протоколу D' сам
перестраивается от нового q0 — зависимость замыкалась в круг. Теперь части
строятся из `split.npy` прежней канонической цепочкой. Утверждение «состав от
этого не изменился» проверяемо БЕЗ модели и без GPU: обе стороны — просто
номера строк. Если состав разошёлся, это находка более крупная, чем сам
цикл: значит прежние числа K-14b считались не на том наборе, что заявлен.

    python experiments/k14d_check_parts_source.py \
        --q1-cache data/k14b/q1_cache.npz
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--q1-cache", default="data/k14b/q1_cache.npz")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import json
    import k13a_build_trajectory_basis as k13a
    import k14a_oracle_cache as k14a
    import k14_common as kc
    from k11c_train_d1 import split_episodes

    meta = json.load(open(f"{a.cache}.meta.json"))
    N = int(meta["n_obs"])
    d = np.load(meta["cache"], allow_pickle=True)
    epi = np.asarray(d["episode"]).astype(np.int64)[:N]
    idx, _ = k13a.load_split(f"{a.cache}.split.npy", N)

    # ТА ЖЕ ЦЕПОЧКА И ТЕ ЖЕ КОНСТАНТЫ, что во всех прежних работах.
    parts, sm = k14a.build_parts(idx, epi, kc.SEL_FRAC, kc.SPLIT_SEED, 0,
                                 np.random.default_rng(0), split_episodes)
    print(f"  из split.npy: {sm}")

    with np.load(a.q1_cache, allow_pickle=True) as z:
        rows_c, part_c = np.asarray(z["rows"], np.int64), z["part"].astype(str)

    bad = []
    for nm in kc.CANONICAL_PARTS:
        fr = np.sort(np.asarray(parts[nm], np.int64))
        fc = np.sort(rows_c[part_c == nm])
        s_r, s_c = kc.arr_sha(fr), kc.arr_sha(fc)
        same = s_r == s_c
        extra = len(np.setdiff1d(fr, fc))
        miss = len(np.setdiff1d(fc, fr))
        print(f"    {nm:12s} split {len(fr):6d} строк {s_r} | кэш "
              f"{len(fc):6d} строк {s_c} | "
              f"{'СОВПАЛ' if same else f'РАЗОШЁЛСЯ (+{extra}/-{miss})'}")
        if not same:
            bad.append(nm)

    over = [(x, y) for i, x in enumerate(kc.CANONICAL_PARTS)
            for y in kc.CANONICAL_PARTS[i + 1:]
            if len(np.intersect1d(parts[x], parts[y]))]
    if over:
        bad.append(f"части пересекаются: {over}")
    print(f"\n  ИТОГ: {'составы совпали' if not bad else f'расхождения {bad}'}")
    return 0 if not bad else 4


if __name__ == "__main__":
    sys.exit(main())
