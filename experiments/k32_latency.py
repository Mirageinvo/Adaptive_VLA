#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-32: задержка seed 0, B = 1, одна очередь val K-21: R и итоговые T/P/F
(детерминированно: actor/router, выбранный оператор с пересчётом потомков,
ограничитель — внутри таймера; critic не загружается). Полный путь до
готового чанка [8, 7] на CPU, синхронизация CUDA, счётчики (один vision,
каждый слой один раз) проверяются рукой. 50 прогревов, 300 измерений,
чередование блоками по 10. Доля нулевых правок — отдельно. Ориентир
+20% p50 к R — инженерная характеристика. Отчёт reports/k32/latency.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT = "reports/k32/latency.json"
NAMES = ("R", "T", "P", "F")
WARM, N_MEAS, BLOCK, TARGET = 50, 300, 10, 20.0


def main():
    ap = argparse.ArgumentParser(description="K-32: задержка")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=N_MEAS)
    a = ap.parse_args()
    import torch
    import k20_data as kd20
    import k21_data as d21
    import k21_latency as kl
    import k29_latency as kl29
    import k32_protocol as pc
    dev = torch.device(a.device)
    plans = {k: pc.load_eval_plan_kind(k, a.seed) for k in NAMES}
    D = d21.load_prepared(require_teacher=False)
    mod, bar, M, dec = pc.load_source_dec(torch, plans["R"], a.seed, a.device)
    arms = {k: pc.make_eval_arm(torch, plans[k], k, a.seed, mod, dec, M)[0]
            for k in NAMES}
    queue = kl29.make_queue(D, np.asarray(D["val_rows"]), 1,
                            max(WARM, a.n), seed=32005)
    cache = {int(r): np.array(D["IMG"][int(r)], copy=True)
             for rows in queue for r in rows}

    def run(k, rows, j):
        arm = arms[k]
        arm.begin(0, j, j)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        batch = kd20.prompt_batch(bar, [cache[int(i)] for i in rows],
                                  kd20.norm_states(np.asarray(
                                      D["state8"][rows])),
                                  D["task"][rows].tolist())
        a_c, rec = arm.decide(batch, int(D["pos_offset"][rows][0]),
                              np.asarray(D["state8"][rows], np.float64), 0)
        torch.cuda.synchronize(dev)
        if a_c.shape != (8, 7) or not np.isfinite(a_c).all():
            raise SystemExit(f"{k}: выход {a_c.shape}")
        return (time.perf_counter() - t0) * 1e3, bool(rec.get("zero", True))
    for k in NAMES:
        for j in range(WARM):
            run(k, queue[j % len(queue)], j)
    meas = {k: [] for k in NAMES}
    zero = {k: 0 for k in NAMES}
    pos = {k: 0 for k in NAMES}
    torch.cuda.reset_peak_memory_stats(dev)
    while min(len(v) for v in meas.values()) < a.n:
        for k in NAMES:
            for _ in range(BLOCK):
                if len(meas[k]) >= a.n:
                    break
                ms, z = run(k, queue[pos[k] % len(queue)], pos[k])
                pos[k] += 1
                meas[k].append(ms)
                zero[k] += int(z)
    blk = {k: kl.q(v) for k, v in meas.items()}
    res = dict(seed=a.seed, batch=1, warmup=WARM, n=a.n, ms=blk,
               zero_share={k: zero[k] / a.n for k in NAMES},
               peak_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
               device=torch.cuda.get_device_name(dev))
    for k in NAMES[1:]:
        res[f"{k}_vs_R_pct"] = 100 * (blk[k]["median"] / blk["R"]["median"]
                                      - 1)
    res["within_target"] = all(res[f"{k}_vs_R_pct"] <= TARGET
                               for k in NAMES[1:])
    for k in NAMES:
        print(f"  {k}: p50 {blk[k]['median']:.1f} мс, p95 {blk[k]['p95']:.1f}"
              + (f" ({res[f'{k}_vs_R_pct']:+.1f}% к R); нулевых правок "
                 f"{res['zero_share'][k]:.2f}" if k != "R" else ""),
              flush=True)
    os.makedirs("reports/k32", exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=float)
    print(f"  отчёт: {OUT}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        assert NAMES == ("R", "T", "P", "F") and WARM == 50
        print("самопроверка k32_latency: конфигурация (замер — на GPU)")
        sys.exit(0)
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            sys.exit(3)
        raise
