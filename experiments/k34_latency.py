#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-34: задержка выходов 8/16/24 одного checkpoint.

Протокол K-21/K-26 (k21_latency): общий фиксированный пул val_sel, батчи 1
и 5 (батч из одной задачи, общий pos_offset), 50 прогревов и 300 измерений
на глубину; глубины одной модели чередуются блоками по 10 без перезагрузки
модели; полный путь — от CPU-наблюдения (кадр уже в памяти, без чтения
диска) через препроцессинг до CPU float32-чанка [B, 20, 7], CUDA
synchronize до и после; vision не кешируется между вызовами. Модели разных
checkpoint — в отдельных областях жизни (одна модель в памяти).
Выводятся p50/p95, пиковая память, выполненные текстовые/экспертные слои,
action-query-layer и router-query операции; число слоёв в коэффициент
ускорения не переводится. Только свободная карта (проверяет k34_run.sh).
--kind pilot: T0; --kind eval: T0 и F0 (seed 0).
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
import k34_model as k34m  # noqa: E402

OUT = "reports/k34/latency_{kind}.json"


def ops(counters):
    return dict(text_layers=int(sum(counters["text"])),
                expert_layers=int(sum(counters["expert"])),
                action_query_layer_ops=int(sum(counters["active"])),
                router_query_ops=int(sum(counters["router"])),
                expand=int(counters["expand"]))


def main():
    ap = argparse.ArgumentParser(description="K-34: латентность")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--kind", choices=("pilot", "eval"), default="pilot")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--n", type=int, default=kl.N_MEAS)
    a = ap.parse_args()
    import torch
    import k20_data as kd
    import k21_data as d21
    import k21_model as km
    import k34_policy as kp
    import k34_protocol as pc
    plan_file = pc.plan_path("pilot" if a.kind == "pilot" else "eval")
    plan = json.load(open(plan_file, encoding="utf-8"))
    pr = pc.plan_problems(plan)
    if pr:
        print("  ОТКАЗ: план K-34: " + "; ".join(pr[:3]))
        return 3
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    va = np.asarray(D["val_rows"])
    modes = ["parents_attn24"] + (["flat_attn24"] if a.kind == "eval" else [])
    res = dict(kind=a.kind, plan_sha1=pc.sha(plan_file), queues={},
               device=torch.cuda.get_device_name(dev),
               threads=dict(torch_intra=torch.get_num_threads(),
                            torch_inter=torch.get_num_interop_threads(),
                            omp=os.environ.get("OMP_NUM_THREADS")))
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
    for mode in modes:
        lab = kp.label(mode, 0, 24)
        spec = plan["arms"][lab]["spec"]
        mod, bar, _ck, _p = kp.load_trained(torch, mode, 0, 24, spec,
                                            a.device)
        mod.checkpoint = False
        vt = kl.VisionTimer(torch, mod.bar.vlm)
        ac = torch.autocast("cuda", dtype=torch.float16)

        def run(d, rows):
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            batch = kd.prompt_batch(bar, [cache[int(i)] for i in rows],
                                    kd.norm_states(np.asarray(
                                        D["state8"][rows])),
                                    D["task"][rows].tolist())
            pos = np.unique(D["pos_offset"][rows])
            if len(pos) != 1:
                raise SystemExit("в батче разные pos_offset")
            s11 = torch.as_tensor(km.state11(np.asarray(D["state8"][rows])),
                                  device=dev)
            with torch.no_grad(), ac:
                o = mod(batch, int(pos[0]), s11, exit_depth=d)
            g = torch.where(o["grip_logit"].float() > 0, 1.0, -1.0)
            chunk = torch.cat([o["arm"].float(), g[..., None]], -1)
            out = chunk.cpu().numpy()
            torch.cuda.synchronize(dev)
            tot = (time.perf_counter() - t0) * 1e3
            v_ms, v_calls = vt.take()
            prob = k34m.counters_problems(mode, mod.counters, d)
            if prob or v_calls != 1 or out.shape[1:] != (20, 7) or \
                    out.dtype != np.float32 or not np.isfinite(out).all():
                raise SystemExit(f"{lab} d{d}: {prob} vision {v_calls} "
                                 f"{out.shape}")
            return tot, ops(mod.counters)
        res[mode] = {}
        for bs, q in queues.items():
            for d in k34m.EXITS:
                for j in range(kl.WARM):
                    run(d, q[j % len(q)])
            torch.cuda.reset_peak_memory_stats(dev)
            meas = {d: [] for d in k34m.EXITS}
            op = {}
            pos = {d: 0 for d in k34m.EXITS}
            while min(len(v) for v in meas.values()) < a.n:
                for d in k34m.EXITS:
                    for _ in range(kl.BLOCK):
                        if len(meas[d]) >= a.n:
                            break
                        t, op[d] = run(d, q[pos[d] % len(q)])
                        pos[d] += 1
                        meas[d].append(t)
            r = {f"d{d}": dict(total=kl.q(meas[d]), ops=op[d], n=len(meas[d]))
                 for d in k34m.EXITS}
            for d in (8, 16):
                r[f"d{d}"]["p50_ratio_to_d24"] = \
                    r[f"d{d}"]["total"]["median"] / r["d24"]["total"]["median"]
            r["peak_memory_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
            res[mode][f"batch{bs}"] = r
            print(f"  {mode} batch {bs}: " + ", ".join(
                f"d{d} p50 {r[f'd{d}']['total']['median']:.1f} мс "
                f"(p95 {r[f'd{d}']['total']['p95']:.1f})"
                for d in k34m.EXITS), flush=True)
        del mod, bar
        torch.cuda.empty_cache()
    if a.kind == "pilot":
        res["batch1"] = res["parents_attn24"]["batch1"]
    os.makedirs("reports/k34", exist_ok=True)
    json.dump(res, open(OUT.format(kind=a.kind), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"  отчёт: {OUT.format(kind=a.kind)}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print("самопроверка k34_latency: только импорт (замер — на GPU)")
        sys.exit(0)
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            sys.exit(3)
        raise
