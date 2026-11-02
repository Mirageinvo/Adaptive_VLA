#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-30: задержка seed 0 — H (parents_hybrid24) и C (parents_softplan24) в
одном процессе на свободной карте, одна очередь входов.

Полный путь: наблюдение -> готовый чанк [B, 20, 7] на CPU (классификатор
корней, softmax/argmax, lookup/смесь, инъекция и readout включены; метки,
кодировщик экспертных действий и debug — нет). Пул — 40 наблюдений val_sel:
восемь задач с ≥ 5 строк, первые восемь описаний в лексикографическом
порядке, по пять детерминированно выбранных строк (seed 20261130); batch 5 —
из одной задачи (единый pos_offset). B = 1 и 5; 50 прогревов, ≥ 300
измерений на режим, чередование блоками по 10, своя позиция очереди у
каждого режима. Отчёт reports/k30/latency.json: median, p95, throughput,
память, отношение H/C (ориентир ≤ 1.10, не критерий).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT = "reports/k30/latency.json"
WARM, N_MEAS, BLOCK = 50, 300, 10
POOL_SEED = 20261130


def pool(D):
    val = np.asarray(D["val_rows"])
    task = np.asarray(D["task"])[val]
    ok = sorted(t for t in set(task.tolist()) if (task == t).sum() >= 5)[:8]
    if len(ok) < 8:
        raise SystemExit(f"задач с ≥ 5 строк val_sel: {len(ok)}")
    rng = np.random.default_rng(POOL_SEED)
    return {t: np.sort(rng.choice(val[task == t], 5, replace=False))
            for t in ok}


def q(x):
    x = np.asarray(x)
    return dict(median=float(np.median(x)), p95=float(np.percentile(x, 95)),
                mean=float(x.mean()), n=int(len(x)))


def main():
    ap = argparse.ArgumentParser(description="K-30: задержка")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    import k20_data as kd20
    import k21_data as d21
    import k21_model as km
    import k30_model as k30m
    import k30_policy as kp
    import k30_protocol as pc
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    models, bars = {}, {}
    for mode in ("hard", "soft"):
        path = pc.plan_path(a.seed, mode)
        plan = json.load(open(path, encoding="utf-8"))
        prob = pc.check_plan_inputs(plan)
        if prob:
            raise SystemExit(f"{path}: {prob[:2]}")
        lab = k30m.label_of(mode, a.seed)
        models[mode], bars[mode], _c, _p = kp.load_policy(
            torch, lab, plan["specs"][lab], a.device)
    P = pool(D)
    cache = {int(r): np.array(D["IMG"][int(r)], copy=True)
             for rows in P.values() for r in rows}
    ac = torch.autocast("cuda", dtype=torch.float16)

    def run(mode, rows):
        bar, mod = bars[mode], models[mode]
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        batch = kd20.prompt_batch(bar, [cache[int(i)] for i in rows],
                                  kd20.norm_states(np.asarray(
                                      D["state8"][rows])),
                                  D["task"][rows].tolist())
        pos = np.unique(D["pos_offset"][rows])
        if len(pos) != 1:
            raise SystemExit("разные pos_offset в батче")
        s11 = torch.as_tensor(km.state11(np.asarray(D["state8"][rows])),
                              device=dev)
        with torch.no_grad(), ac:
            o = mod(batch, int(pos[0]), s11)
        g = torch.where(o["grip_logit"].float() > 0, 1.0, -1.0)
        chunk = torch.cat([o["arm"].float(), g[..., None]], -1).cpu().numpy()
        ms = (time.perf_counter() - t0) * 1e3
        c = mod.counters
        if chunk.shape != (len(rows), 20, 7) or not np.isfinite(chunk).all() \
                or c["vision"] != 1 or c["classifier"] != 1:
            raise SystemExit(f"{mode}: выход {chunk.shape} или счётчики {c}")
        return ms
    tasks = sorted(P)
    res = dict(seed=a.seed, pool={t: P[t].tolist() for t in tasks},
               pool_sha=hashlib.sha1(np.concatenate(
                   [P[t] for t in tasks]).tobytes()).hexdigest())
    for bs in (1, 5):
        queue = [P[tasks[j % 8]][:bs] if bs == 5 else
                 P[tasks[(j // 5) % 8]][j % 5:j % 5 + 1]
                 for j in range(max(WARM, N_MEAS))]
        for mode in ("hard", "soft"):
            for j in range(WARM):
                run(mode, queue[j % len(queue)])
        meas = {m: [] for m in ("hard", "soft")}
        pos = {m: 0 for m in meas}
        torch.cuda.reset_peak_memory_stats(dev)
        while min(len(v) for v in meas.values()) < N_MEAS:
            for m in ("hard", "soft"):
                for _ in range(BLOCK):
                    if len(meas[m]) >= N_MEAS:
                        break
                    meas[m].append(run(m, queue[pos[m] % len(queue)]))
                    pos[m] += 1
        blk = {m: dict(q(v), throughput=1e3 * bs / float(np.median(v)))
               for m, v in meas.items()}
        blk["ratio_h_c"] = blk["hard"]["median"] / blk["soft"]["median"]
        blk["peak_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        res[f"batch{bs}"] = blk
        print(f"  batch {bs}: H p50 {blk['hard']['median']:.1f} мс (p95 "
              f"{blk['hard']['p95']:.1f}), C p50 {blk['soft']['median']:.1f} "
              f"мс (p95 {blk['soft']['p95']:.1f}); H/C "
              f"{blk['ratio_h_c']:.3f} (ориентир ≤ 1.10)", flush=True)
    res["device"] = torch.cuda.get_device_name(dev)
    os.makedirs("reports/k30", exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(f"  отчёт: {OUT}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        D = dict(val_rows=np.arange(60),
                 task=np.array([f"task {i % 10}" for i in range(60)]))
        P = pool(D)
        assert len(P) == 8 and list(P) == sorted(P)[:8] and \
            all(len(v) == 5 for v in P.values())
        assert all(np.array_equal(P[t], pool(D)[t]) for t in P)
        print("самопроверка k30_latency: пул 8 задач × 5 строк, "
              "детерминирован (замер — на GPU)")
        sys.exit(0)
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            sys.exit(3)
        raise
