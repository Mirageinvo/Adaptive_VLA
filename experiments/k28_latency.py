#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-28: цена цикла «один тяжёлый план + восемь лёгких шагов» для
parents_goal_fb24 (seed 0) против исходного parents24 и штатного BAR.

T_cycle = подготовка входа + VLA + дерево поз/контексты + перенос на CPU +
Σ T_control_step[0:8] (CPU-контроллер по свежим состояниям). Физическая
симуляция не входит. parents24 и BAR: построение чанка, перенос на CPU и
выдача его восьми строк (без вычислений обратной связи).
Очередь — строки val_sel одной задачи (общий pos_offset), батчи 1 и 5;
свежие состояния j = 0…7 — закреплённые реальные последовательности
k28_data (одна очередь для всех политик); 50 прогревов, ≥ 300 циклов на
политику, чередование блоками по 10. CUDA синхронизируется на границе
GPU-части; полное время — wall-clock до готовых CPU-команд. Отдельно:
p50/p95 одного шага контроллера и одной обработки позы (state8 -> p, R).
Ориентир надстройки — ≤ +10% T_cycle к parents24 (инженерный, не основание
отбрасывать SR). Отчёт reports/k28/latency.json.
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

OUT = "reports/k28/latency.json"
STEPS = 8


def make_queue(D, rows, bs, n, seed=20261030):
    """Батчи по bs строк одной задачи."""
    rng = np.random.default_rng([seed, bs])
    by = {}
    for r in rows:
        by.setdefault(str(D["task"][r]), []).append(int(r))
    tasks = sorted(t for t, v in by.items() if len(v) >= bs)
    return [np.sort(rng.choice(by[tasks[j % len(tasks)]], bs, replace=False))
            for j in range(n)]


def main():
    ap = argparse.ArgumentParser(description="K-28: латентность цикла")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--n", type=int, default=kl.N_MEAS)
    a = ap.parse_args()
    import torch
    import k20_data as kd
    import k21_data as d21
    import k21_model as km
    import k22_policy as kp22
    import k28_control as kc
    import k28_data as kd28
    import k28_geometry as kg
    import k28_policy as kp
    import k28_protocol as pc
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    S = kd28.load()
    G = kg.build(torch)
    pth = pc.plan_path(0)
    plan = json.load(open(pth, encoding="utf-8"))
    prob = pc.check_plan_inputs(plan)
    if prob:
        raise SystemExit("; ".join(prob[:3]))
    lab = "parents_goal_fb24_s0"
    spec = plan["specs"][lab]
    fb, _b, _p, _m, _s = kp.load_model(torch, lab, spec, a.device)
    par, _b2, _c2, _p2 = kp22.load_policy(torch, "parents24",
                                          spec["source"]["ckpt"], a.device)
    bar = d21.load_bar(dev, D["prep_report"]["bar_pin"])
    for m in (fb, par):
        m.checkpoint = False
        m.eval()
    rt = kc.build(torch)["Runtime"](fb.ctrl.state_dict())
    E, codec = bar["E"], bar["codec"]
    ac = torch.autocast("cuda", dtype=torch.float16)
    cache = {}
    val = np.asarray(D["val_rows"])
    ok_rows = val[S["state_valid"][S["pos"][val]][:, :STEPS].all(1)]
    nplan = [0]

    def inputs(rows):
        batch = kd.prompt_batch(bar, [cache[int(i)] for i in rows],
                                kd.norm_states(np.asarray(D["state8"][rows])),
                                D["task"][rows].tolist())
        pos = np.unique(D["pos_offset"][rows])
        if len(pos) != 1:
            raise SystemExit("в батче разные pos_offset")
        return batch, int(pos[0])

    def cycle(name, rows):
        B = len(rows)
        seq = S["state_seq8"][S["pos"][rows]][:, :STEPS]
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        batch, po = inputs(rows)
        st8 = np.asarray(D["state8"][rows], np.float64)
        steps_ms, cmds = [], []
        with torch.no_grad():
            if name == "fullbar":
                tk = bar["model"].generate(**batch, position_offset=po,
                                           do_sample=False,
                                           initial_position_shift=1)
                K = tk.reshape(B, 3, 16)
                z = sum(E[j][K[:, j, :].long()] for j in range(3))
                x, _ = codec._decode(z, embodiment_ids=0)
                chunk = x[..., :7].float().cpu().numpy()
            elif name == "parents24":
                s11 = torch.as_tensor(km.state11(st8), device=dev)
                with ac:
                    o = par(batch, po, s11)
                g = torch.where(o["grip_logit"].float() > 0, 1.0, -1.0)
                chunk = torch.cat([o["arm"].float(), g[..., None]], -1) \
                    .cpu().numpy()
            else:
                s11 = torch.as_tensor(km.state11(st8), device=dev)
                pl = fb.plan(batch, po, s11, torch.as_tensor(
                    st8, dtype=torch.float32, device=dev),
                    plan_id=f"lat{nplan[0]}", autocast=ac)
                cpu = fb.to_cpu_plan(pl)
                nplan[0] += 1
            torch.cuda.synchronize(dev)
            t_plan = time.perf_counter() - t0
            if name == "parents_goal_fb24":
                rt.begin(cpu)
                act = np.ones(B, bool)
                for j in range(STEPS):
                    t1 = time.perf_counter()
                    a_, _d = rt.step(cpu, seq[:, j], j, act)
                    steps_ms.append((time.perf_counter() - t1) * 1e3)
                    cmds.append(a_)
            else:
                for j in range(STEPS):
                    cmds.append(np.asarray(chunk[:, j], np.float32))
        t_cycle = time.perf_counter() - t0
        A = np.stack(cmds)
        return dict(plan=t_plan * 1e3, cycle=t_cycle * 1e3,
                    ctrl_steps=steps_ms, shape=list(A.shape),
                    finite=bool(np.isfinite(A).all()),
                    counters=getattr(fb if name == "parents_goal_fb24"
                                     else par, "counters", None)
                    if name != "fullbar" else None)

    def pose_ms(rows):
        s = torch.as_tensor(np.asarray(S["state_seq8"][S["pos"][rows]][:, 1],
                                       np.float32))
        t1 = time.perf_counter()
        with torch.no_grad():
            G["pose_of_state8"](s)
        return (time.perf_counter() - t1) * 1e3
    names = ["parents_goal_fb24", "parents24", "fullbar"]
    res = dict(source=dict(k28=spec["ckpt_sha1"],
                           parents24=spec["source"]["ckpt_sha1"]),
               plan_sha1=pc.full_sha(pth), queues={},
               cpu_threads=torch.get_num_threads(),
               note="T_cycle = подготовка входа + тяжёлый проход + дерево поз/"
                    "контексты + перенос на CPU + 8 шагов CPU-контроллера по "
                    "закреплённым реальным состояниям (без симуляции); "
                    "parents24/BAR — чанк + перенос + выдача 8 строк; fp16 "
                    "autocast, checkpointing off, LoRA не слита")
    for bs in a.batches:
        queue = make_queue(D, ok_rows, bs, max(kl.WARM, a.n))
        qa = np.stack(queue).astype(np.int64)
        res["queues"][f"batch{bs}"] = dict(n=len(queue), sha1=hashlib.sha1(
            qa.tobytes()).hexdigest())
        for rows in queue:
            for r in rows:
                if int(r) not in cache:
                    cache[int(r)] = np.array(D["IMG"][int(r)], copy=True)
        for nm in names:
            for j in range(kl.WARM):
                cycle(nm, queue[j % len(queue)])
        meas = {nm: [] for nm in names}
        poses = []
        torch.cuda.reset_peak_memory_stats(dev)
        pos = {nm: 0 for nm in names}
        while min(len(v) for v in meas.values()) < a.n:
            for nm in names:
                for _ in range(kl.BLOCK):
                    if len(meas[nm]) >= a.n:
                        break
                    rows = queue[pos[nm] % len(queue)]
                    r = cycle(nm, rows)
                    pos[nm] += 1
                    if not r["finite"] or r["shape"] != [STEPS, bs, 7]:
                        raise SystemExit(f"{nm}: команды {r['shape']}, "
                                         f"конечны {r['finite']}")
                    c = r["counters"]
                    if c is not None and (c["vision"] != 1 or any(
                            x != 1 for x in c["text"])):
                        raise SystemExit(f"{nm}: счётчики {c}")
                    if nm == "parents_goal_fb24":
                        poses.append(pose_ms(rows))
                    meas[nm].append(r)
        blk = {}
        for nm, v in meas.items():
            blk[nm] = dict(plan=kl.q([m["plan"] for m in v]),
                           cycle=kl.q([m["cycle"] for m in v]),
                           cycle_per_step=kl.q([m["cycle"] / STEPS
                                                for m in v]),
                           heavy_calls=1, light_calls=STEPS if
                           nm == "parents_goal_fb24" else 0, n=len(v))
            if nm == "parents_goal_fb24":
                blk[nm]["ctrl_step"] = kl.q([x for m in v
                                             for x in m["ctrl_steps"]])
                blk[nm]["ctrl_sum"] = kl.q([sum(m["ctrl_steps"]) for m in v])
                blk[nm]["pose"] = kl.q(poses)
        res[f"batch{bs}"] = dict(
            blk, peak_memory_gb_all_loaded=torch.cuda.max_memory_allocated(
                dev) / 2 ** 30)
        for nm in names:
            r = blk[nm]
            print(f"  batch {bs} {nm:18}: T_cycle p50 "
                  f"{r['cycle']['median']:.1f} мс, p95 {r['cycle']['p95']:.1f}"
                  f"; план p50 {r['plan']['median']:.1f} мс", flush=True)
        f = blk["parents_goal_fb24"]
        print(f"    шаг контроллера p50 {f['ctrl_step']['median']:.3f} мс, "
              f"p95 {f['ctrl_step']['p95']:.3f}; поза p50 "
              f"{f['pose']['median']:.3f} мс", flush=True)
        for ref in ("parents24", "fullbar"):
            x, y = f["cycle"], blk[ref]["cycle"]
            d50 = 100 * (x["median"] / y["median"] - 1)
            res[f"batch{bs}"][f"vs_{ref}_pct"] = dict(
                median=d50, p95=100 * (x["p95"] / y["p95"] - 1),
                within_10pct=d50 <= 10 if ref == "parents24" else None)
            print(f"    K-28 к {ref}: T_cycle медиана {d50:+.1f}%"
                  + (" (ориентир ≤ +10%)" if ref == "parents24" else ""),
                  flush=True)
    res["device"] = torch.cuda.get_device_name(dev)
    os.makedirs("reports/k28", exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(f"  отчёт: {OUT}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        q = make_queue(dict(task=np.array(["a"] * 6 + ["b"] * 6)),
                       np.arange(12), 5, 4)
        assert all(len(set(np.array(["a"] * 6 + ["b"] * 6)[x])) == 1
                   for x in q)
        print("самопроверка k28_latency: очередь строк одной задачи "
              "(замер — на GPU)")
        sys.exit(0)
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            sys.exit(3)
        raise
