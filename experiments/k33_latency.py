#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-33: задержка T0 (parents_attn24_s0), F0 (flat_attn24_s0), B (fullbar,
штатный generate) и BC (fullbar_cached, допуск обязателен) в одном замере.

Протокол K-21/K-26 (k21_latency): общая очередь наблюдений val_sel, батчи 1
и 5 (батч — из одной задачи, общий pos_offset), 50 прогревов, ≥ 300
измерений на политику, чередование блоками по 10; полный путь — от
подготовки входа до готового float32-чанка [B, 20, 7] на CPU, CUDA
synchronize до и после; vision — внутри единственного вызова. Веса T0/F0 —
из плана K-33 машины A (источники K-26), снимок BAR — закреплённый.
Замер только на свободной карте (проверяет k33_run.sh); при занятой карте
анализ завершается с latency_pending. Не делает sim.step.
Отчёт reports/k33/latency.json.
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

OUT = "reports/k33/latency.json"
PAIRS = (("parents_attn24", "fullbar"), ("flat_attn24", "fullbar"),
         ("parents_attn24", "fullbar_cached"), ("flat_attn24",
                                                "fullbar_cached"),
         ("parents_attn24", "flat_attn24"))


def main():
    ap = argparse.ArgumentParser(description="K-33: латентность")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--plan", default="reports/k33/plan_A.json")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--n", type=int, default=kl.N_MEAS)
    a = ap.parse_args()
    import torch
    import k20_data as kd
    import k21_data as d21
    import k21_model as km
    import k26_policy as kp26
    import k33_eval as ke
    dev = torch.device(a.device)
    plan = json.load(open(a.plan, encoding="utf-8"))
    prob = ke.plan_problems(plan, arms=["T0", "F0", "BC"])
    if prob:
        print("  ОТКАЗ: план K-33: " + "; ".join(prob[:3]))
        return 3
    D = d21.load_prepared(require_teacher=False)
    models, src = {}, {}
    for arm, mode in (("T0", "parents_attn24"), ("F0", "flat_attn24")):
        s_ = plan["arms"][arm]["source"]
        spec = json.load(open(s_["k26_plan"], encoding="utf-8"))["specs"][
            plan["arms"][arm]["label"]]
        mod, _b, _c, _p = kp26.load_policy(torch, plan["arms"][arm]["label"],
                                           spec, a.device)
        models[mode] = mod
        src[mode] = spec["ckpt_sha1"]
    for k, m in models.items():
        m.checkpoint = False
        m.eval()
        if hasattr(m, "collect"):
            m.collect = False
            m.keep_maps = False
    bar = d21.load_bar(dev, D["prep_report"]["bar_pin"])
    models["fullbar"] = bar["model"]
    import k21_barcache as bc
    adm = bc.require_admitted(D)
    models["fullbar_cached"] = bar["model"]
    cached = dict(status="измерен", equivalence=adm.get("equivalence"))
    vt, timers = {}, {}
    for k, m in models.items():                 # общий vlm — один таймер
        vlm = m.vlm if k.startswith("fullbar") else m.bar.vlm
        if id(vlm) not in vt:
            vt[id(vlm)] = kl.VisionTimer(torch, vlm)
        timers[k] = vt[id(vlm)]
    E, codec = bar["E"], bar["codec"]
    ac = torch.autocast("cuda", dtype=torch.float16)
    va = np.asarray(D["val_rows"])
    cache, bank = {}, {}

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
        o = {}
        with torch.no_grad():
            g0.record()
            if lab.startswith("fullbar"):
                po = int(pos[0])
                if lab == "fullbar_cached":
                    import k21_barcache as bc
                    tk = bc.generate_cached(torch, models[lab], batch, po)
                else:
                    tk = models[lab].generate(**batch, position_offset=po,
                                              do_sample=False,
                                              initial_position_shift=1)
                K = tk.reshape(len(rows), 3, 16)
                z = sum(E[j][K[:, j, :].long()] for j in range(3))
                x, _ = codec._decode(z, embodiment_ids=0)
                chunk = x[..., :7].float()
            else:
                with ac:
                    o = models[lab](batch, int(pos[0]), s11)
                g = torch.where(o["grip_logit"].float() > 0, 1.0, -1.0)
                chunk = torch.cat([o["arm"].float(), g[..., None]], -1)
            g1.record()
        a_np = chunk.cpu().numpy()
        t_total = time.perf_counter() - t0
        v_ms, v_calls = timers[lab].take()
        if "bank" in o:
            bank[lab] = dict(n_max=o["bank"]["n_max"], dv=o["bank"]["dv"])
        return dict(prep=t_prep * 1e3, vision=v_ms, vision_calls=v_calls,
                    policy_gpu=g0.elapsed_time(g1), total=t_total * 1e3,
                    shape=list(a_np.shape), dtype=str(a_np.dtype),
                    finite=bool(np.isfinite(a_np).all()))
    labs = list(models)
    res = dict(sources=src, queues={}, note=("полный путь: подготовка входа -> "
                                "float32-чанк [B, 20, 7] на CPU; у новых "
                                "моделей банк, K/V, шесть router и readout "
                                "внутри forward; fp16 autocast; "
                                "checkpointing off; LoRA не слита; fullbar "
                                "— штатный generate + декодер кодека в том "
                                "же замере и очереди"))
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
                            r["shape"][1:] != [20, 7] or \
                            r["dtype"] != "float32":
                        raise SystemExit(f"{lab}: выход {r['shape']} "
                                         f"{r['dtype']}, vision "
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
            print(f"  batch {bs} {lab:15}: p50 {r['total']['median']:.1f} мс, "
                  f"p95 {r['total']['p95']:.1f}, GPU "
                  f"{r['policy_gpu']['median']:.1f} мс; "
                  f"{r['throughput_obs_s']:.1f} набл./с", flush=True)
        for new, old in PAIRS:
            if old not in res[f"batch{bs}"]:
                continue
            x = res[f"batch{bs}"][new]["total"]
            b = res[f"batch{bs}"][old]["total"]
            d50 = 100 * (x["median"] / b["median"] - 1)
            d95 = 100 * (x["p95"] / b["p95"] - 1)
            res[f"batch{bs}"][f"{new}_vs_{old}_pct"] = dict(
                median=d50, p95=d95)
            print(f"    {new} к {old}: медиана {d50:+.1f}%, p95 {d95:+.1f}%",
                  flush=True)
    res["bank"] = bank
    res["fullbar_cached"] = cached
    res["device"] = torch.cuda.get_device_name(dev)
    res["threads"] = dict(torch_intra=torch.get_num_threads(),
                          torch_inter=torch.get_num_interop_threads(),
                          omp=os.environ.get("OMP_NUM_THREADS"))
    res["machine"] = plan["machine"]
    res["plan_contract_fp"] = plan["contract_fp"]
    os.makedirs("reports/k33", exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(f"  отчёт: {OUT}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print("самопроверка k33_latency: только импорт (замер — на GPU)")
        sys.exit(0)
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            sys.exit(3)
        raise
