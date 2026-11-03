#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-31: задержка seed 0 на одинаковых входах (очередь val K-21, B = 1):
reference, repair eligible, flatrepair eligible (весь пул, нормировка,
CPU-признаки и селектор внутри таймера), repair spent (ремонт уже
использован — только reference; флаг ставится вне таймера). Полный путь до
готового чанка [8, 7] на CPU, синхронизация CUDA, счётчики (один vision,
каждый слой один раз) проверяются рукой на каждом вызове. 50 прогревов,
300 измерений, чередование блоками по 10. Ориентир: eligible ≤ +20% p50 к
reference — превышение сообщается, не отказ. B = 5 не измеряется: рука
K-31 по построению B = 1. Отчёт reports/k31/latency.json.
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

OUT = "reports/k31/latency.json"
NAMES = ("reference", "repair_eligible", "flatrepair_eligible",
         "repair_spent")
WARM, N_MEAS, BLOCK = 50, 300, 10
TARGET = 20.0


def main():
    ap = argparse.ArgumentParser(description="K-31: задержка")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=N_MEAS)
    a = ap.parse_args()
    import torch
    import k20_data as kd20
    import k21_data as d21
    import k21_latency as kl
    import k29_latency as kl29
    import k31_policy as kpol
    import k31_protocol as pc
    plan = pc.load_eval_plan(a.seed)
    p = pc.load_prepare()
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    mod, bar, M, dec = pc.load_source_dec(torch, p, a.seed, a.device)
    sel = pc.load_selector(torch, plan, a.seed)
    arms = dict(reference=kpol.Arm(torch, mod, dec, M, mode="reference",
                                   source_seed=a.seed),
                repair_eligible=kpol.Arm(torch, mod, dec, M, mode="repair",
                                         source_seed=a.seed, space=3,
                                         selector=sel),
                flatrepair_eligible=kpol.Arm(torch, mod, dec, M,
                                             mode="flatrepair",
                                             source_seed=a.seed, space=3,
                                             selector=sel),
                repair_spent=kpol.Arm(torch, mod, dec, M, mode="repair",
                                      source_seed=a.seed, space=3,
                                      selector=sel))
    cache = {}
    queue = kl29.make_queue(D, np.asarray(D["val_rows"]), 1,
                            max(WARM, a.n), seed=20261231)
    for rows in queue:
        for r in rows:
            cache.setdefault(int(r), np.array(D["IMG"][int(r)], copy=True))

    def inputs(rows):
        batch = kd20.prompt_batch(bar, [cache[int(i)] for i in rows],
                                  kd20.norm_states(np.asarray(
                                      D["state8"][rows])),
                                  D["task"][rows].tolist())
        return batch, int(D["pos_offset"][rows][0]), np.asarray(
            D["state8"][rows], np.float64)

    def run(nm, rows, j):
        arm = arms[nm]
        arm.begin(0, j)                             # ключ RNG пула
        if nm == "repair_spent":
            arm.used = True                         # вне таймера
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        batch, po, s8 = inputs(rows)
        a_c, rec = arm.decide(batch, po, s8, 0)
        torch.cuda.synchronize(dev)
        ms = (time.perf_counter() - t0) * 1e3
        if a_c.shape != (8, 7) or not np.isfinite(a_c).all():
            raise SystemExit(f"{nm}: выход {a_c.shape}")
        if nm == "repair_spent" and rec["chosen"]:
            raise SystemExit("spent: выбран ремонт")
        return ms, rec["chosen"]
    for nm in NAMES:
        for j in range(WARM):
            run(nm, queue[j % len(queue)], j)
    meas = {nm: [] for nm in NAMES}
    chosen = {nm: 0 for nm in NAMES}
    pos = {nm: 0 for nm in NAMES}
    torch.cuda.reset_peak_memory_stats(dev)
    while min(len(v) for v in meas.values()) < a.n:
        for nm in NAMES:
            for _ in range(BLOCK):
                if len(meas[nm]) >= a.n:
                    break
                ms, ch = run(nm, queue[pos[nm] % len(queue)], pos[nm])
                pos[nm] += 1
                meas[nm].append(ms)
                chosen[nm] += int(ch > 0)
    blk = {nm: kl.q(v) for nm, v in meas.items()}
    res = dict(seed=a.seed, batch=1, warmup=WARM, n=a.n, block=BLOCK,
               ms=blk, chosen=chosen,
               peak_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
               device=torch.cuda.get_device_name(dev),
               selector_sha1=plan["selector"]["sha1"], status="выполнено")
    for nm in NAMES[1:]:
        d = 100 * (blk[nm]["median"] / blk["reference"]["median"] - 1)
        res[f"{nm}_vs_reference_pct"] = d
    res["within_target"] = all(res[f"{nm}_vs_reference_pct"] <= TARGET
                               for nm in ("repair_eligible",
                                          "flatrepair_eligible"))
    for nm in NAMES:
        print(f"  {nm:20}: p50 {blk[nm]['median']:.1f} мс, p95 "
              f"{blk[nm]['p95']:.1f}" + (
                  f" ({res[f'{nm}_vs_reference_pct']:+.1f}% к reference)"
                  if nm != "reference" else ""), flush=True)
    print("  ориентир +20% eligible: " + ("в пределах" if res["within_target"]
                                          else "ПРЕВЫШЕН (ограничение)"))
    os.makedirs("reports/k31", exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=float)
    print(f"  отчёт: {OUT}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        assert NAMES[0] == "reference" and WARM == 50 and N_MEAS == 300
        print("самопроверка k31_latency: конфигурация (замер — на GPU)")
        sys.exit(0)
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            sys.exit(3)
        raise
