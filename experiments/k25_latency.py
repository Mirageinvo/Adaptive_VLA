#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-25: задержка parents_cond24 против parents_mr24, parents24 и flat24
(seed 0).

Протокол K-21 (k21_latency): общая очередь наблюдений val_sel, батчи 1 и 5
(батч — из одной задачи, общий pos_offset), 50 прогревов, ≥ 300 измерений на
политику, чередование блоками по 10; полный путь — от подготовки входа до
готового чанка [B, 20, 7] на CPU; vision — внутри единственного вызова.
Модели загружаются один раз, eval/no_grad, checkpointing выключен, LoRA не
слита. У cond в таймер входят адаптеры, гейты и сборка путей; у MR — QR и
гейт ранга. Веса и SHA — из плана K-25 seed 0 и закреплённой в нём записи
сида. Отчёт reports/k25/latency.json
(прежние отчёты не перезаписываются).
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

import k21_latency as kl  # noqa: E402

OUT = "reports/k25/latency.json"


def main():
    ap = argparse.ArgumentParser(description="K-25: латентность")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--n", type=int, default=kl.N_MEAS)
    a = ap.parse_args()
    import torch
    import k20_data as kd
    import k21_data as d21
    import k21_model as km
    import k21_policy as kp21
    import k22_policy as kp22
    import k24_policy as kp24
    import k25_policy as kp25
    import k25_protocol as pc
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    plan = json.load(open(pc.plan_path(0), encoding="utf-8"))
    prob = pc.check_plan_inputs(plan)
    if prob:
        print("  ОТКАЗ: " + "; ".join(prob[:3]))
        return 3
    # контроли — из записи сида, закреплённой в плане (её отпечаток сверен
    # check_plan_inputs с prepare.json), а не из изменяемого prepare.json
    pr = plan["seed_entry"]
    if pc.entry_fp(pr) != plan["seed_entry_fp"]:
        print("  ОТКАЗ: запись сида в плане не совпадает с отпечатком")
        return 3
    lab = "parents_cond24_s0"
    models, src = {}, {}
    mod, _b, _c, _p = kp25.load_policy(torch, lab, plan["specs"][lab],
                                       a.device)
    models["parents_cond24"] = mod
    src["parents_cond24"] = plan["specs"][lab]["ckpt_sha1"]
    cm = pr["control_mr"]
    mod, _b, _c, _p = kp24.load_policy(
        torch, "parents_mr24_s0", dict(arch="parents_mr24", seed=0,
                                       ckpt=cm["ckpt"],
                                       weights_hash=cm["weights_hash"]),
        a.device)
    models["parents_mr24"] = mod
    src["parents_mr24"] = cm["ckpt_sha1"]
    mod, _b, _c, _p = kp22.load_policy(torch, "parents24",
                                       pr["control_parents"]["ckpt"], a.device)
    models["parents24"] = mod
    src["parents24"] = pr["control_parents"]["ckpt_sha1"]
    mod, _b, _c, _p = kp21.load_policy(torch, "flat24",
                                       pr["control_flat"]["ckpt"], a.device)
    models["flat24"] = mod
    src["flat24"] = pr["control_flat"]["ckpt_sha1"]
    for k, m in models.items():
        m.checkpoint = False
        m.eval()
    bar = d21.load_bar(dev, D["prep_report"]["bar_pin"])
    timers = {k: kl.VisionTimer(torch, m.bar.vlm) for k, m in models.items()}
    ac = torch.autocast("cuda", dtype=torch.float16)
    va = np.asarray(D["val_rows"])
    cache = {}

    def run(lab, rows):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        batch = kd.prompt_batch(bar, [cache[int(i)] for i in rows],
                                kd.norm_states(np.asarray(D["state8"][rows])),
                                D["task"][rows].tolist())
        pos = np.unique(D["pos_offset"][rows])
        if len(pos) != 1:
            raise SystemExit("в батче разные pos_offset")
        s11 = torch.as_tensor(km.state11(np.asarray(D["state8"][rows])),
                              device=dev)
        torch.cuda.synchronize(dev)
        t_prep = time.perf_counter() - t0
        g0 = torch.cuda.Event(enable_timing=True)
        g1 = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            g0.record()
            with ac:
                o = models[lab](batch, int(pos[0]), s11)
            g = torch.where(o["grip_logit"].float() > 0, 1.0, -1.0)
            chunk = torch.cat([o["arm"].float(), g[..., None]], -1)
            g1.record()
        a_np = chunk.cpu().numpy()
        t_total = time.perf_counter() - t0
        v_ms, v_calls = timers[lab].take()
        return dict(prep=t_prep * 1e3, vision=v_ms, vision_calls=v_calls,
                    policy_gpu=g0.elapsed_time(g1), total=t_total * 1e3,
                    shape=list(a_np.shape),
                    finite=bool(np.isfinite(a_np).all()))
    labs = list(models)
    import hashlib
    res = dict(sources=src, seed_entry_fp=plan["seed_entry_fp"],
               plan_sha1=pc.sha(pc.plan_path(0)), queues={}, note=("полный путь: подготовка входа -> чанк "
               "[B, 20, 7] на CPU; у parents_cond24 адаптеры, гейты и "
               "сборка путей, у parents_mr24 QR и гейт ранга внутри forward; "
               "fp16 autocast; checkpointing off; LoRA не слита"))
    for bs in a.batches:
        queue = kl.make_queue(D["task"], va, bs, max(kl.WARM, a.n))
        qa = np.stack(queue).astype(np.int64)
        res["queues"][f"batch{bs}"] = dict(
            n=len(queue), sha1=hashlib.sha1(qa.tobytes()).hexdigest(),
            first=qa[:3].tolist())
        for r in np.unique(np.concatenate(queue)):
            if int(r) not in cache:
                cache[int(r)] = np.array(D["IMG"][int(r)], copy=True)
        for lab in labs:
            for j in range(kl.WARM):
                run(lab, queue[j % len(queue)])
        meas = {lab: [] for lab in labs}
        torch.cuda.reset_peak_memory_stats(dev)
        pos = {lab: 0 for lab in labs}
        while min(len(v) for v in meas.values()) < a.n:
            for lab in labs:
                for _ in range(kl.BLOCK):
                    if len(meas[lab]) >= a.n:
                        break
                    r = run(lab, queue[pos[lab] % len(queue)])
                    pos[lab] += 1
                    if not r["finite"] or r["vision_calls"] != 1 or \
                            r["shape"][1:] != [20, 7]:
                        raise SystemExit(f"{lab}: выход {r['shape']}, vision "
                                         f"{r['vision_calls']}")
                    meas[lab].append(r)
        res[f"batch{bs}"] = {lab: dict(
            {k: kl.q([m[k] for m in v]) for k in ("prep", "vision",
                                                   "policy_gpu", "total")},
            throughput_obs_s=float(bs * 1e3 / np.median(
                [m["total"] for m in v])), n=len(v))
            for lab, v in meas.items()}
        res[f"batch{bs}"]["peak_memory_gb_all_loaded"] = \
            torch.cuda.max_memory_allocated(dev) / 2 ** 30
        for lab in labs:
            r = res[f"batch{bs}"][lab]
            print(f"  batch {bs} {lab:13}: p50 {r['total']['median']:.1f} мс, "
                  f"p95 {r['total']['p95']:.1f}, GPU {r['policy_gpu']['median']:.1f}"
                  f" мс; {r['throughput_obs_s']:.1f} набл./с", flush=True)
        base = res[f"batch{bs}"]["parents24"]["total"]["median"]
        for k in ("parents_cond24", "parents_mr24"):
            x = res[f"batch{bs}"][k]["total"]["median"]
            res[f"batch{bs}"][f"{k}_vs_parents_pct"] = 100 * (x / base - 1)
            print(f"    {k} к parents24: {100 * (x / base - 1):+.1f}% "
                  f"(ориентир ≤ +10%)", flush=True)
    res["device"] = torch.cuda.get_device_name(dev)
    os.makedirs("reports/k25", exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(f"  отчёт: {OUT}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print("самопроверка k25_latency: только импорт (замер — на GPU)")
        sys.exit(0)
    sys.exit(main())
