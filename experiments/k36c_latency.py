#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-36 (§15): задержка B = 1 — Tplus, T32, R (сид 0) и original BAR.

После фиксации кандидата (план C). Общая очередь val_sel K-21
(k21_latency.make_queue, B = 1), 50 прогревов и 300 измерений на руку;
R/T32/Tplus одного загруженного источника чередуются блоками по 10; BAR —
в отдельной области жизни модели. Полный путь: кадр в памяти -> подготовка
(k20_data.prompt_batch) -> тяжёлый проход/экспорт, reference, CPU actor,
оператор, ограничитель (R/T32/Tplus — k32_policy.Arm.decide) либо
generate + сборка кодов (BAR) -> CPU float32 чанк; CUDA synchronize до и
после. Critic не загружается. Сим не используется.
Отчёт: p50/p95, пиковая память, доля вызовов с нулевой правкой (zero) у
T32/Tplus. Ориентир +20% к R — описательно. Только свободная карта
(k36bc_run.sh: занято ≤ 2048 МиБ, иначе latency_pending).
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

import k36_harness as kh  # noqa: E402
import k36_protocol as pc  # noqa: E402

OUT = "reports/k36/latency.json"
ARMS = ("R_s0", "T32_s0", "Tplus_s0")


def latency_problems(plan_c_sha1, queue_sha1=None, n=300, warm=50):
    """Отчёт задержки принадлежит этому плану C, полной очереди и объёму:
    plan_c_sha1, B = 1, n/warm, все четыре руки с n измерениями."""
    if not os.path.exists(OUT):
        return None
    r = json.load(open(OUT, encoding="utf-8"))
    p = []
    if r.get("plan_c_sha1") != plan_c_sha1:
        p.append("задержка другого плана C")
    if r.get("batch") != 1 or r.get("n") != n or r.get("warm") != warm:
        p.append("задержка: батч/объём ≠ 1 / 300 / 50")
    if queue_sha1 is not None and r.get("queue_sha1") != queue_sha1:
        p.append("задержка: другая очередь")
    pol = r.get("policies") or {}
    bar = r.get("bar") or {}
    for k in ARMS:
        if (pol.get(k) or {}).get("n") != n:
            p.append(f"задержка: нет полного замера {k}")
    if (bar.get("BAR") or {}).get("n") != n:
        p.append("задержка: нет полного замера BAR")
    return p


def queue_sha(D, n=300, warm=50):
    import hashlib
    import k21_latency as kl
    q = kl.make_queue(D["task"], np.asarray(D["val_rows"]), 1, max(warm, n))
    return hashlib.sha1(np.stack(q).astype(np.int64).tobytes()).hexdigest()


def measure(torch, dev, queue, cache, D, arms_fn, n, warm, block):
    """arms_fn() -> {метка: callable(rows) -> (чанк, rec)}; одна модель."""
    import k21_latency as kl
    arms = arms_fn()
    names = list(arms)
    for nm in names:
        for j in range(warm):
            arms[nm](queue[j % len(queue)])
    torch.cuda.reset_peak_memory_stats(dev)
    meas = {nm: [] for nm in names}
    zero = {nm: [] for nm in names}
    pos = {nm: 0 for nm in names}
    while min(len(v) for v in meas.values()) < n:
        for nm in names:
            for _ in range(block):
                if len(meas[nm]) >= n:
                    break
                rows = queue[pos[nm] % len(queue)]
                pos[nm] += 1
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                a, rec = arms[nm](rows)
                torch.cuda.synchronize(dev)
                meas[nm].append((time.perf_counter() - t0) * 1e3)
                if a.dtype != np.float32 or not np.isfinite(a).all():
                    raise SystemExit(f"{nm}: чанк не float32/конечен")
                if "zero" in rec:
                    zero[nm].append(bool(rec["zero"]))
    out = {nm: dict(total=kl.q(meas[nm]), n=len(meas[nm]),
                    zero_share=float(np.mean(zero[nm])) if zero[nm] else None)
           for nm in names}
    out["peak_memory_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    return out


def _policy_lifetime(torch, prep, plan, dev, queue, cache, D, a, device):
    import k20_data as kd20
    import k36b_screen as kbs
    v = kh.load_source(torch, 0, device)
    arms = dict(R_s0=kh.make_arm(torch, prep, "R_s0", v),
                T32_s0=kh.make_arm(torch, prep, "T32_s0", v),
                Tplus_s0=kh.make_arm(torch, prep, "T32_s0", v,
                                     actor=kbs.load_tplus(torch, prep, 0,
                                                          v["mod"].H)))
    bar = v["bar"]

    def call(arm):
        def f(rows):
            r = int(rows[0])
            batch = kd20.prompt_batch(bar, [cache[r]], kd20.norm_states(
                np.asarray(D["state8"][[r]])), [D["task"][r]])
            arm.begin(0, 0, 0)
            return arm.decide(batch, int(D["pos_offset"][r]),
                              np.asarray(D["state8"][[r]], np.float64), 0)
        return f
    return measure(torch, dev, queue, cache, D,
                   lambda: {k: call(x) for k, x in arms.items()}, a.n,
                   a.warm, 10)


def _bar_lifetime(torch, dev, queue, cache, D, a, device):
    import k20_data as kd20
    import k36c_bar as kbar
    b = kbar.load_bar(torch, device)
    arm = kbar.BarArm(torch, b["bar"])

    def f(rows):
        r = int(rows[0])
        batch = kd20.prompt_batch(b["bar"], [cache[r]], kd20.norm_states(
            np.asarray(D["state8"][[r]])), [D["task"][r]])
        arm.begin(0, 0, 0)
        return arm.decide(batch, int(D["pos_offset"][r]), None, 0)
    return measure(torch, dev, queue, cache, D, lambda: dict(BAR=f), a.n,
                   a.warm, 10)


def main():
    import hashlib
    ap = argparse.ArgumentParser(description="K-36: задержка B1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warm", type=int, default=50)
    a = ap.parse_args()
    import torch
    import k21_data as d21
    import k21_latency as kl
    import k36c_confirm as kc
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    prep = pc.load_prepare()
    p = kc.plan_c_problems(prep)
    if p:
        print("  ОТКАЗ: план C: " + "; ".join(p[:3]))
        return 3
    plan = json.load(open(kc.PLAN_C, encoding="utf-8"))
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    va = np.asarray(D["val_rows"])
    queue = kl.make_queue(D["task"], va, 1, max(a.warm, a.n))
    cache = {int(r): np.array(D["IMG"][int(r)], copy=True)
             for r in np.unique(np.concatenate(queue))}
    free0 = torch.cuda.mem_get_info(dev)
    res = dict(plan_c_sha1=pc.sha(kc.PLAN_C), batch=1, n=a.n, warm=a.warm,
               device=torch.cuda.get_device_name(dev),
               mem_used_before_mb=int((free0[1] - free0[0]) / 2 ** 20),
               queue_sha1=hashlib.sha1(np.stack(queue).astype(np.int64)
                                       .tobytes()).hexdigest(),
               threads=dict(intra=torch.get_num_threads(),
                            inter=torch.get_num_interop_threads(),
                            omp=os.environ.get("OMP_NUM_THREADS")))
    res["policies"] = _policy_lifetime(torch, prep, plan, dev, queue, cache,
                                       D, a, a.device)
    kh.free(torch)
    res["mem_after_policies_gb"] = torch.cuda.memory_allocated(dev) / 2 ** 30
    res["bar"] = _bar_lifetime(torch, dev, queue, cache, D, a, a.device)
    kh.free(torch)
    m = {k: res["policies"][k]["total"]["median"] for k in ARMS}
    m["BAR"] = res["bar"]["BAR"]["total"]["median"]
    res["summary"] = dict(p50=m, Tplus_over_R=m["Tplus_s0"] / m["R_s0"],
                          T32_over_R=m["T32_s0"] / m["R_s0"],
                          Tplus_over_BAR=m["Tplus_s0"] / m["BAR"],
                          note="ориентир +20% к R — описательно")
    pc.save_json(OUT, res)
    print("  задержка B1 p50: " + ", ".join(f"{k} {v:.1f} мс"
                                            for k, v in m.items()) +
          f"; Tplus/R {res['summary']['Tplus_over_R']:.3f}, Tplus/BAR "
          f"{res['summary']['Tplus_over_BAR']:.3f}; {OUT}")
    return 0


def selftest():
    import tempfile
    g = globals()
    keep = g["OUT"]
    with tempfile.TemporaryDirectory() as td:
        try:
            g["OUT"] = os.path.join(td, "lat.json")
            assert latency_problems("P", "Q") is None       # нет -> pending
            good = dict(plan_c_sha1="P", batch=1, n=300, warm=50,
                        queue_sha1="Q",
                        policies={k: dict(n=300) for k in ARMS},
                        bar=dict(BAR=dict(n=300)))
            pc.save_json(g["OUT"], good)
            assert latency_problems("P", "Q") == []
            for bad in (dict(good, plan_c_sha1="X"), dict(good, n=1),
                        dict(good, warm=0), dict(good, queue_sha1="Z"),
                        dict(good, policies={"R_s0": dict(n=300)}),
                        dict(good, bar={"BAR": dict(n=1)})):
                pc.save_json(g["OUT"], bad)
                assert latency_problems("P", "Q"), bad
        finally:
            g["OUT"] = keep
    print("самопроверка k36c_latency пройдена: задержка чужого плана C, "
          "неполного объёма, другой очереди или без руки — отказ; нет "
          "файла — latency_pending (замер — на GPU)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            sys.exit(3)
        raise
