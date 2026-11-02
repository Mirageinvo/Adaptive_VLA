#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin: парное сравнение двух пилотов (по умолчанию parents24 − flat24,
seed 0) на одном банке состояний.

Берутся только задачи с действительной отметкой завершения
(rt2_eval.task_done_problems: модель, n, SHA журнала, полная проверка).
Пара — (задача, seed среды); наборы seed обоих пилотов в каждой задаче
обязаны совпадать (иначе отказ: банки различаются). Выводится SR с равными
весами задач, по задачам, выигрыши/потери по парам, точный Мак-Немар и 95%
бутстрэп ΔSR с пересэмплированием пар внутри задачи. Пилот разведочный
(8 задач × 10 состояний), вывод — описательный.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BOOT = dict(n=10000, seed=20261102)


def load(log_dir, tasks, n):
    import rt2_eval as re_
    out, prob = {}, []
    for t in tasks:
        p = re_.task_done_problems(log_dir, t, n)
        if p:
            prob += p
            continue
        for e in re_.read_task(log_dir, t):
            out[(t, int(e["seed"]))] = bool(e["success"])
    return out, prob


def compare(a, b, boot=BOOT):
    import k15h_analyze as ha
    if set(a) != set(b):
        raise SystemExit("наборы (задача, seed) пилотов различаются: "
                         f"{sorted(set(a) ^ set(b))[:4]}")
    st = sorted(a)
    by = {}
    for k in st:
        by.setdefault(k[0], []).append(k)
    gain = sum(1 for k in st if a[k] and not b[k])
    loss = sum(1 for k in st if b[k] and not a[k])
    D = {t: np.array([float(a[k]) - float(b[k]) for k in v])
         for t, v in by.items()}
    rng = np.random.default_rng(boot["seed"])
    bs = np.array([np.mean([d[rng.integers(0, len(d), len(d))].mean()
                            for d in D.values()]) for _ in range(boot["n"])])
    sa, sb = ha.equal_weight(a, st), ha.equal_weight(b, st)
    return dict(sr_a=sa, sr_b=sb, d=sa - sb, gain=gain, loss=loss,
                mcnemar_p=ha.mcnemar_p(gain, loss), n=len(st),
                ci95=[float(np.percentile(bs, 2.5)),
                      float(np.percentile(bs, 97.5))],
                per_task={t: [int(sum(a[k] for k in v)),
                              int(sum(b[k] for k in v)), len(v)]
                          for t, v in by.items()})


def selftest():
    a = {("x", 100000 + i): i < 3 for i in range(10)}
    a.update({("y", 100000 + i): i < 5 for i in range(10)})
    b = {k: (v if k[0] == "y" else k[1] < 100001) for k, v in a.items()}
    r = compare(a, b, dict(n=500, seed=1))
    assert r["gain"] == 2 and r["loss"] == 0 and abs(r["d"] - 0.1) < 1e-12
    try:
        compare(a, {k: v for k, v in list(b.items())[:-1]})
        raise AssertionError("разные наборы приняты")
    except SystemExit:
        pass
    print("самопроверка rt2_compare пройдена: пары, равные веса задач, "
          "отказ при разных наборах seed")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    import rt1_common as rc
    ap = argparse.ArgumentParser(description="RoboTwin: парный пилот")
    ap.add_argument("--a", default="reports/rt2/eval/pilot_s0_parents24")
    ap.add_argument("--b", default="reports/rt2/eval/pilot_s0_flat24")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="reports/rt2/compare_s0_parents24_flat24.json")
    a = ap.parse_args()
    tasks = list(rc.TASKS)
    A, pa = load(a.a, tasks, a.n)
    B, pb = load(a.b, tasks, a.n)
    if pa or pb:
        print("  НЕПОЛНО/ОТКАЗ: " + "; ".join((pa + pb)[:6]))
        return 3
    r = compare(A, B)
    na, nb = os.path.basename(a.a), os.path.basename(a.b)
    print(f"  {na} − {nb} ({r['n']} пар, 8 задач): SR {100 * r['sr_a']:.1f}% "
          f"/ {100 * r['sr_b']:.1f}%, Δ {100 * r['d']:+.1f} п.п. "
          f"[{100 * r['ci95'][0]:+.1f}; {100 * r['ci95'][1]:+.1f}], "
          f"+{r['gain']}/−{r['loss']}, Мак-Немар p {r['mcnemar_p']:.4f}")
    for t, v in r["per_task"].items():
        print(f"    {t:20} {v[0]}/{v[2]}  против  {v[1]}/{v[2]}")
    json.dump(dict(r, a=a.a, b=a.b, note="разведочный пилот; описательно"),
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
