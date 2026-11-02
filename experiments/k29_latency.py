#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-29: задержка seed 0 — исторический K-25 (FP16-путь), init-control
(экспорт + reference FP32) и final-RL (экспорт + actor FP32 + схват
reference), одна очередь входов (val K-21, батч — строки одной задачи),
B = 1 и 5; 50 прогревов, ≥ 300 измерений на политику, чередование блоками
по 10; полный путь до готового чанка [B, 8, 7] на CPU. Критик и шум не
входят. Отчёт reports/k29/latency.json.
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

import k21_latency as kl  # noqa: E402

OUT = "reports/k29/latency.json"
NAMES = ("k25_hist", "init_control", "final_rl")


def make_queue(D, rows, bs, n, seed=20261102):
    rng = np.random.default_rng([seed, bs])
    by = {}
    for r in rows:
        by.setdefault(str(D["task"][r]), []).append(int(r))
    tasks = sorted(t for t, v in by.items() if len(v) >= bs)
    return [np.sort(rng.choice(by[tasks[j % len(tasks)]], bs, replace=False))
            for j in range(n)]


def main():
    ap = argparse.ArgumentParser(description="K-29: задержка")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--n", type=int, default=kl.N_MEAS)
    a = ap.parse_args()
    import torch
    import k20_data as kd20
    import k21_data as d21
    import k21_model as km
    import k29_harness as kh
    import k29_model as k29m
    import k29_policy as kp
    import k29_protocol as pc
    p = pc.load_prepare()
    want = kh.expected_model(p, a.seed, "final")
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    mod, bar, M, actor, ref, _c = pc.load_all(torch, p, a.seed, a.device)
    f = torch.load(pc.final_path(a.seed), map_location="cpu",
                   weights_only=False)
    actor.load_state_dict(f["actor"])
    if k29m.tensor_hash(actor.state_dict()) != want["actor_hash"]:
        raise SystemExit("final actor ≠ отчёту обучения")
    actor.eval()
    pols = dict(init_control=kp.Policy(torch, mod, actor, ref, M, "reference"),
                final_rl=kp.Policy(torch, mod, actor, ref, M, "actor"))
    ac = torch.autocast("cuda", dtype=torch.float16)
    cache = {}
    val = np.asarray(D["val_rows"])

    def inputs(rows):
        batch = kd20.prompt_batch(bar, [cache[int(i)] for i in rows],
                                  kd20.norm_states(np.asarray(
                                      D["state8"][rows])),
                                  D["task"][rows].tolist())
        pos = np.unique(D["pos_offset"][rows])
        if len(pos) != 1:
            raise SystemExit("в батче разные pos_offset")
        return batch, int(pos[0]), np.asarray(D["state8"][rows], np.float64)

    def run(name, rows):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        batch, po, st8 = inputs(rows)
        with torch.no_grad():
            if name == "k25_hist":
                s11 = torch.as_tensor(km.state11(st8), device=dev)
                with ac:
                    o = mod(batch, po, s11)
                g = torch.where(o["grip_logit"][:, :8].float() > 0, 1.0, -1.0)
                chunk = torch.cat([o["arm"][:, :8].float(), g[..., None]],
                                  -1).cpu().numpy()
                cnt = mod.counters
            else:
                out = pols[name].chunk(batch, po, st8)
                chunk, cnt = out["a_codec"], out["counters"]
        ms = (time.perf_counter() - t0) * 1e3
        return dict(ms=ms, shape=list(np.shape(chunk)),
                    finite=bool(np.isfinite(chunk).all()), cnt=cnt)
    res = dict(seed=a.seed, model=want, queues={},
               note="полный путь до готового чанка [B, 8, 7] на CPU; fp16 "
                    "trunk; декодеры K-29 — FP32; критик и шум не входят")
    for bs in a.batches:
        queue = make_queue(D, val, bs, max(kl.WARM, a.n))
        res["queues"][f"batch{bs}"] = hashlib.sha1(
            np.stack(queue).astype(np.int64).tobytes()).hexdigest()
        for rows in queue:
            for r in rows:
                cache.setdefault(int(r), np.array(D["IMG"][int(r)], copy=True))
        for nm in NAMES:
            for j in range(kl.WARM):
                run(nm, queue[j % len(queue)])
        meas = {nm: [] for nm in NAMES}
        pos = {nm: 0 for nm in NAMES}
        torch.cuda.reset_peak_memory_stats(dev)
        while min(len(v) for v in meas.values()) < a.n:
            for nm in NAMES:
                for _ in range(kl.BLOCK):
                    if len(meas[nm]) >= a.n:
                        break
                    r = run(nm, queue[pos[nm] % len(queue)])
                    pos[nm] += 1
                    c = r["cnt"]
                    if not r["finite"] or r["shape"] != [bs, 8, 7] or \
                            c["vision"] != 1 or any(x != 1 for x in c["text"]):
                        raise SystemExit(f"{nm}: выход {r['shape']} или "
                                         f"счётчики {c}")
                    meas[nm].append(r["ms"])
        blk = {nm: kl.q(v) for nm, v in meas.items()}
        res[f"batch{bs}"] = dict(blk, peak_gb=torch.cuda.max_memory_allocated(
            dev) / 2 ** 30)
        for nm in NAMES:
            print(f"  batch {bs} {nm:13}: p50 {blk[nm]['median']:.1f} мс, p95 "
                  f"{blk[nm]['p95']:.1f}", flush=True)
        for x, y in (("init_control", "k25_hist"), ("final_rl", "k25_hist"),
                     ("final_rl", "init_control")):
            d = 100 * (blk[x]["median"] / blk[y]["median"] - 1)
            res[f"batch{bs}"][f"{x}_vs_{y}_pct"] = d
            print(f"    {x} к {y}: {d:+.1f}%", flush=True)
    res["device"] = torch.cuda.get_device_name(dev)
    os.makedirs("reports/k29", exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=float)
    print(f"  отчёт: {OUT}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        q = make_queue(dict(task=np.array(["a"] * 6 + ["b"] * 6)),
                       np.arange(12), 5, 4)
        assert all(len(set(np.array(["a"] * 6 + ["b"] * 6)[x])) == 1
                   for x in q)
        print("самопроверка k29_latency: очередь строк одной задачи (замер — "
              "на GPU)")
        sys.exit(0)
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            sys.exit(3)
        raise
