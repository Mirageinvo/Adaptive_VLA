#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-27: задержка parents_mem24 (seed 0) против исторического parents24 и
исходного BAR в одном замере.

Очередь — цепочки val (t−8 -> t) с шагом 8 из индекса K-27; батч — цепочки
одной задачи (общий pos_offset); батчи 1 и 5; 50 прогревов, ≥ 300 измерений
на политику, чередование блоками по 10. Полный путь до float32-чанка
[B, 20, 7] на CPU. Режимы parents_mem24:
  cold   — первый вызов эпизода (памяти нет, путь parents24 + снимок);
  steady — память построена НЕ измеряемым вызовом той же модели на t−8 этих
           же цепочек; в таймер входят выбор памяти из MemoryBank (k = 8),
           сборка, forward с двумя чтениями, снимок в банк.
parents24 и fullbar — на тех же текущих наблюдениях. Счётчики: один vision,
каждый слой один раз, чтения root/mid (steady). Ориентир — ≤ +10% медианы
к parents24; p50/p95 сообщаются при любом результате. Отчёт
reports/k27/latency.json.
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

OUT = "reports/k27/latency.json"


def make_queue(D, ix, bs, n, seed=20261028):
    """Батчи по bs цепочек одной задачи (номера цепочек индекса val)."""
    rng = np.random.default_rng([seed, bs])
    by = {}
    for i, c in enumerate(ix["cur"]):
        by.setdefault(str(D["task"][c]), []).append(i)
    tasks = sorted(t for t, v in by.items() if len(v) >= bs)
    return [np.sort(rng.choice(by[tasks[j % len(tasks)]], bs, replace=False))
            for j in range(n)]


def main():
    ap = argparse.ArgumentParser(description="K-27: латентность")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--n", type=int, default=kl.N_MEAS)
    a = ap.parse_args()
    import torch
    import k20_data as kd
    import k21_data as d21
    import k21_model as km
    import k22_policy as kp22
    import k27_data as kd27
    import k27_memory as kmem
    import k27_model as k27m
    import k27_policy as kp27
    import k27_protocol as pc
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    S = kd27.load()
    ix = S["val"]
    plan = json.load(open(pc.plan_path(0), encoding="utf-8"))
    prob = pc.check_plan_inputs(plan)
    if prob:
        print("  ОТКАЗ: " + "; ".join(prob[:3]))
        return 3
    lab = "parents_mem24_s0"
    spec = plan["specs"][lab]
    mem, _b, _c, _p = kp27.load_policy(torch, lab, spec, a.device)
    par, _b2, _c2, _p2 = kp22.load_policy(torch, "parents24",
                                          spec["source"]["ckpt"], a.device)
    bar = d21.load_bar(dev, D["prep_report"]["bar_pin"])
    models = dict(parents_mem24=mem, parents24=par, fullbar=bar["model"])
    for k, m in models.items():
        if k != "fullbar":
            m.checkpoint = False
            m.eval()
    vt, timers = {}, {}
    for k, m in models.items():
        vlm = m.vlm if k == "fullbar" else m.bar.vlm
        if id(vlm) not in vt:
            vt[id(vlm)] = kl.VisionTimer(torch, vlm)
        timers[k] = vt[id(vlm)]
    E, codec = bar["E"], bar["codec"]
    ac = torch.autocast("cuda", dtype=torch.float16)
    cache = {}

    def inputs(rows):
        batch = kd.prompt_batch(bar, [cache[int(i)] for i in rows],
                                kd.norm_states(np.asarray(D["state8"][rows])),
                                D["task"][rows].tolist())
        pos = np.unique(D["pos_offset"][rows])
        if len(pos) != 1:
            raise SystemExit("в батче разные pos_offset")
        s11 = torch.as_tensor(km.state11(np.asarray(D["state8"][rows])),
                              device=dev)
        return batch, int(pos[0]), s11

    def warm_memory(bank, chains, B):
        rows = ix["prev1"][chains]
        batch, po, s11 = inputs(rows)
        with torch.no_grad(), ac:
            o = mem(batch, po, s11, None)
        bank.update(list(range(B)), [1] * B, [0] * B,
                    kmem.split_snapshot(o["snapshot"]))

    def run(name, chains):
        B = len(chains)
        bank = None
        if name == "steady":
            bank = kmem.MemoryBank()
            warm_memory(bank, chains, B)
            torch.cuda.synchronize(dev)
        rows = ix["cur"][chains]
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        batch, po, s11 = inputs(rows)
        g0 = torch.cuda.Event(enable_timing=True)
        g1 = torch.cuda.Event(enable_timing=True)
        lab_ = "parents_mem24" if name in ("cold", "steady") else name
        with torch.no_grad():
            g0.record()
            if name == "fullbar":
                tk = models[lab_].generate(**batch, position_offset=po,
                                           do_sample=False,
                                           initial_position_shift=1)
                K = tk.reshape(B, 3, 16)
                z = sum(E[j][K[:, j, :].long()] for j in range(3))
                x, _ = codec._decode(z, embodiment_ids=0)
                chunk = x[..., :7].float()
            else:
                mi = None
                if name == "steady":
                    mi = k27m.mem_input(torch, bank, list(range(B)), [1] * B,
                                        [kmem.K_EXEC] * B)
                with ac:
                    o = models[lab_](batch, po, s11, mi) \
                        if lab_ == "parents_mem24" else \
                        models[lab_](batch, po, s11)
                g = torch.where(o["grip_logit"].float() > 0, 1.0, -1.0)
                chunk = torch.cat([o["arm"].float(), g[..., None]], -1)
                if lab_ == "parents_mem24":
                    rows_ = kmem.split_snapshot(o["snapshot"])
            g1.record()
        a_np = chunk.cpu().numpy()
        if lab_ == "parents_mem24":
            (bank or kmem.MemoryBank()).update(
                list(range(B)), [1] * B, [kmem.K_EXEC] * B, rows_)
        t_total = time.perf_counter() - t0
        v_ms, v_calls = timers[lab_].take()
        reads = models[lab_].counters["mem_reads"] \
            if lab_ == "parents_mem24" else None
        return dict(vision=v_ms, vision_calls=v_calls,
                    policy_gpu=g0.elapsed_time(g1), total=t_total * 1e3,
                    shape=list(a_np.shape), dtype=str(a_np.dtype),
                    finite=bool(np.isfinite(a_np).all()), reads=reads)
    names = ["steady", "cold", "parents24", "fullbar"]
    res = dict(source=dict(memory=spec["ckpt_sha1"],
                           parents24=spec["source"]["ckpt_sha1"]),
               plan_sha1=pc.sha(pc.plan_path(0)), queues={},
               note="полный путь: подготовка входа -> float32-чанк [B, 20, 7]"
                    " на CPU; steady — память из не измеряемого вызова на "
                    "t−8 тех же цепочек, в таймере выбор/сборка памяти, "
                    "forward, снимок в банк; fp16 autocast; checkpointing "
                    "off; LoRA не слита")
    for bs in a.batches:
        queue = make_queue(D, ix, bs, max(kl.WARM, a.n))
        qa = np.stack(queue).astype(np.int64)
        res["queues"][f"batch{bs}"] = dict(n=len(queue), sha1=hashlib.sha1(
            qa.tobytes()).hexdigest())
        for ch in queue:
            for r in np.r_[ix["cur"][ch], ix["prev1"][ch]]:
                if int(r) not in cache:
                    cache[int(r)] = np.array(D["IMG"][int(r)], copy=True)
        for nm in names:
            for j in range(kl.WARM):
                run(nm, queue[j % len(queue)])
        meas = {nm: [] for nm in names}
        torch.cuda.reset_peak_memory_stats(dev)
        pos = {nm: 0 for nm in names}
        while min(len(v) for v in meas.values()) < a.n:
            for nm in names:
                for _ in range(kl.BLOCK):
                    if len(meas[nm]) >= a.n:
                        break
                    r = run(nm, queue[pos[nm] % len(queue)])
                    pos[nm] += 1
                    want = ["root", "mid"] if nm == "steady" else \
                        ([] if nm == "cold" else None)
                    if not r["finite"] or r["vision_calls"] != 1 or \
                            r["shape"][1:] != [20, 7] or \
                            r["dtype"] != "float32" or r["reads"] != want:
                        raise SystemExit(f"{nm}: выход {r['shape']} "
                                         f"{r['dtype']}, vision "
                                         f"{r['vision_calls']}, чтения "
                                         f"{r['reads']}")
                    meas[nm].append(r)
        blk = {nm: dict({k: kl.q([m[k] for m in v]) for k in
                         ("vision", "policy_gpu", "total")}, n=len(v))
               for nm, v in meas.items()}
        res[f"batch{bs}"] = dict(
            steady=dict(parents_mem24=blk["steady"],
                        parents24=blk["parents24"], fullbar=blk["fullbar"]),
            cold=dict(parents_mem24=blk["cold"], parents24=blk["parents24"],
                      fullbar=blk["fullbar"]),
            peak_memory_gb_all_loaded=torch.cuda.max_memory_allocated(dev) /
            2 ** 30)
        for nm in names:
            r = blk[nm]
            print(f"  batch {bs} {nm:10}: p50 {r['total']['median']:.1f} мс, "
                  f"p95 {r['total']['p95']:.1f}, GPU "
                  f"{r['policy_gpu']['median']:.1f} мс", flush=True)
        for nm in ("steady", "cold"):
            x, y = blk[nm]["total"], blk["parents24"]["total"]
            d50 = 100 * (x["median"] / y["median"] - 1)
            res[f"batch{bs}"][f"{nm}_vs_parents24_pct"] = dict(
                median=d50, p95=100 * (x["p95"] / y["p95"] - 1),
                within_10pct=d50 <= 10)
            print(f"    parents_mem24 {nm} к parents24: медиана {d50:+.1f}% "
                  f"(ориентир ≤ +10%)", flush=True)
    res["device"] = torch.cuda.get_device_name(dev)
    os.makedirs("reports/k27", exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(f"  отчёт: {OUT}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        q = make_queue(dict(task=np.array(["a"] * 6 + ["b"] * 6)),
                       dict(cur=np.arange(12)), 5, 4)
        assert all(len(set(np.array(["a"] * 6 + ["b"] * 6)[x])) == 1
                   for x in q)
        print("самопроверка k27_latency: очередь цепочек одной задачи "
              "(замер — на GPU)")
        sys.exit(0)
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            sys.exit(3)
        raise
