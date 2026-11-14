#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-35: задержка A / C / N / C_fast (один загруженный K-35) и S (K-34 T0).

Протокол K-21/K-26 (k21_latency): общий фиксированный пул val_sel, батчи
1 и 5, 50 прогревов и 300 измерений на руку; A/C/N/C_fast одного
экземпляра чередуются блоками по 10; полный путь — от CPU-наблюдения
(кадр в памяти) через препроцессинг до CPU float32-чанка [B, 20, 7], CUDA
synchronize до и после; vision не кешируется. Отдельно: препроцессинг
(до входа в модель), vision (таймер), policy = total − препроцессинг.
S — в отдельной области жизни модели. CPU: OMP/MKL = 4 (k35_run.sh),
torch intra = 4, inter = 1. Только свободная карта (k35_run.sh, ≤ 2048
МиБ); иначе — latency_pending. У A/C/N одинаковы Q = 192, router-Q = 48,
24 слоя, три головы, две сети выбора; C_fast — без сетей выбора.
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
import k35_model as k35m  # noqa: E402
import k35_policy as kp  # noqa: E402

OUT = "reports/k35/latency.json"
K35_ARMS = ("A", "C", "N", "C_fast")


def ops(c):
    return dict(text_layers=int(sum(c["text"])),
                expert_layers=int(sum(c["expert"])),
                action_query_layer_ops=int(sum(c["active"])),
                router_query_ops=int(sum(c["router"])),
                expand=int(c["expand"]), heads=len(c["heads"]),
                decisions=int(c.get("decisions", 0)))


def stats(v):
    return kl.q(v)


def measure(torch, a, D, cache, queues, arms, make, fwd, check, kd, km):
    """arms: метки; make() -> (mod, bar); fwd(mod, arm, batch, po, s11, B)
    -> out; check(mod, arm) -> проблемы. Одна модель в памяти."""
    dev = torch.device(a.device)
    out = dict(mem_before_load_gb=torch.cuda.memory_allocated(dev) / 2 ** 30)
    mod, bar = make()
    mod.checkpoint = False
    vt = kl.VisionTimer(torch, mod.bar.vlm)
    ac = torch.autocast("cuda", dtype=torch.float16)

    def run(arm, rows):
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
        t1 = time.perf_counter()
        with torch.no_grad(), ac:
            o = fwd(mod, arm, batch, int(pos[0]), s11, len(rows))
        g = torch.where(o["grip_logit"].float() > 0, 1.0, -1.0)
        arr = torch.cat([o["arm"].float(), g[..., None]], -1).cpu().numpy()
        torch.cuda.synchronize(dev)
        t2 = time.perf_counter()
        v_ms, v_calls = vt.take()
        prob = check(mod, arm)
        if prob or v_calls != 1 or arr.shape[1:] != (20, 7) or \
                arr.dtype != np.float32 or not np.isfinite(arr).all():
            raise SystemExit(f"{arm}: {prob} vision {v_calls} {arr.shape}")
        return dict(total=(t2 - t0) * 1e3, pre=(t1 - t0) * 1e3,
                    policy=(t2 - t1) * 1e3, vision=v_ms), ops(mod.counters)
    try:
        for bs, q in queues.items():
            for arm in arms:
                for j in range(kl.WARM):
                    run(arm, q[j % len(q)])
            torch.cuda.reset_peak_memory_stats(dev)
            meas = {x: {k: [] for k in ("total", "pre", "policy", "vision")}
                    for x in arms}
            op, pos = {}, {x: 0 for x in arms}
            while min(len(v["total"]) for v in meas.values()) < a.n:
                for arm in arms:
                    for _ in range(kl.BLOCK):
                        if len(meas[arm]["total"]) >= a.n:
                            break
                        t, op[arm] = run(arm, q[pos[arm] % len(q)])
                        pos[arm] += 1
                        for k, v in t.items():
                            meas[arm][k].append(v)
            r = {x: dict({k: stats(v) for k, v in meas[x].items()},
                         ops=op[x], n=len(meas[x]["total"])) for x in arms}
            r["peak_memory_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
            out[f"batch{bs}"] = r
            print(f"  batch {bs}: " + ", ".join(
                f"{x} p50 {r[x]['total']['median']:.1f} мс (p95 "
                f"{r[x]['total']['p95']:.1f})" for x in arms), flush=True)
    finally:
        vt.vlm.__dict__.pop("get_image_features", None)
        vt.vlm = vt.orig = None
        del vt, mod, bar
    return out


def main():
    ap = argparse.ArgumentParser(description="K-35: латентность")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--n", type=int, default=kl.N_MEAS)
    a = ap.parse_args()
    import gc
    import torch
    import k20_data as kd
    import k21_data as d21
    import k21_model as km
    import k34_model as k34m
    import k34_policy as kp34
    import k35_protocol as pc
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    plan_file = pc.plan_path("pilot")
    plan = json.load(open(plan_file, encoding="utf-8"))
    pr = pc.plan_problems(plan)
    if pr:
        print("  ОТКАЗ: план K-35: " + "; ".join(pr[:3]))
        return 3
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    va = np.asarray(D["val_rows"])
    res = dict(plan_sha1=pc.sha(plan_file), queues={},
               device=torch.cuda.get_device_name(dev),
               mem_used_before_mb=int((torch.cuda.mem_get_info(dev)[1] -
                                       torch.cuda.mem_get_info(dev)[0]) /
                                      2 ** 20),
               threads=dict(torch_intra=torch.get_num_threads(),
                            torch_inter=torch.get_num_interop_threads(),
                            omp=os.environ.get("OMP_NUM_THREADS"),
                            mkl=os.environ.get("MKL_NUM_THREADS")),
               cpu=os.cpu_count())
    cache = {}
    queues = {bs: kl.make_queue(D["task"], va, bs, max(kl.WARM, a.n))
              for bs in a.batches}
    for bs, q in queues.items():
        qa = np.stack(q).astype(np.int64)
        res["queues"][f"batch{bs}"] = dict(
            n=len(q), sha1=hashlib.sha1(qa.tobytes()).hexdigest())
        for r in np.unique(np.concatenate(q)):
            if int(r) not in cache:
                cache[int(r)] = np.array(D["IMG"][int(r)], copy=True)
    spA = plan["arms"]["k35_A"]["spec"]
    c_const = plan["arms"]["k35_C"]["spec"]["c_const"]
    ctrls = {}

    def make35():
        mod, bar, _ck = kp.load_decoder(torch, spA, a.device)
        B = kp.load_selectors(torch, spA, int(mod.q0.shape[-1]))
        for x in K35_ARMS:
            ctrls[x] = kp.Controller(
                torch, x, B, a.device, c_const=c_const if x in (
                    "C", "C_fast") else None,
                ctx=dict(task=0, inits=list(range(5)), env_seed=101))
        return mod, bar
    calls = {"n": 0}

    def fwd35(mod, arm, batch, po, s11, B):
        calls["n"] += 1
        ct = ctrls[arm]
        ct.ctx["inits"] = list(range(B))
        ct.begin(calls["n"])
        return mod(batch, po, s11, controller=ct)

    def check35(mod, arm):
        return k35m.counters_problems(mod.counters, decisions=2)
    res["k35"] = measure(torch, a, D, cache, queues, K35_ARMS, make35, fwd35,
                         check35, kd, km)
    ctrls.clear()
    gc.collect()
    torch.cuda.empty_cache()
    spS = plan["arms"]["k35_S"]["spec"]["k34"]

    def makeS():
        mod, bar, _ck, _p = kp34.load_trained(torch, "parents_attn24", 0, 24,
                                              spS, a.device)
        return mod, bar

    def fwdS(mod, arm, batch, po, s11, B):
        return mod(batch, po, s11, exit_depth=24)

    def checkS(mod, arm):
        return k34m.counters_problems("parents_attn24", mod.counters, 24)
    res["S"] = measure(torch, a, D, cache, queues, ("S",), makeS, fwdS,
                       checkS, kd, km)
    gc.collect()
    torch.cuda.empty_cache()
    for bs in a.batches:
        r = res["k35"][f"batch{bs}"]
        m = {x: r[x]["total"]["median"] for x in K35_ARMS}
        res[f"summary_batch{bs}"] = dict(
            p50=dict(m, S=res["S"][f"batch{bs}"]["S"]["total"]["median"]),
            A_over_C=m["A"] / m["C"], A_over_C_fast=m["A"] / m["C_fast"],
            selection_overhead_A_vs_C_fast=m["A"] / m["C_fast"] - 1,
            note="ориентир доп. задержки выбора A к C_fast ≤ 10% — без "
                 "обещания; сокращение Q к S не объявлять ускорением")
    res["ckpt"] = dict(decoder_sha1=spA["decoder_sha1"],
                       selectors_sha1=spA["selectors_sha1"],
                       s_ckpt_sha1=plan["arms"]["k35_S"]["spec"]["ckpt_sha1"])
    os.makedirs("reports/k35", exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=float)
    for bs in a.batches:
        s = res[f"summary_batch{bs}"]
        print(f"  B{bs}: p50 {s['p50']}; A/C {s['A_over_C']:.3f}, "
              f"A/C_fast {s['A_over_C_fast']:.3f}")
    print(f"  отчёт: {OUT}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print("самопроверка k35_latency: только импорт (замер — на GPU)")
        sys.exit(0)
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            sys.exit(3)
        raise
