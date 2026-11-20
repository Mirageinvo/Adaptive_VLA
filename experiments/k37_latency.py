#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-37, подтверждение: сквозная задержка B1 для T1, T2, T2P.

  python3 experiments/k37_latency.py --run-root R --device cuda:N
Очередь: 15 сырых наблюдений после ожидания (dev-сцены space 0 банка
K-33, задачи 0/6/8 по 5, хеши сверяются с банком). На сид: одна область
жизни источника и трёх политик; разогрев 30 вызовов на политику; затем
политики чередуются блоками по 5 до 150 замеров на политику (15 × 10
повторов); два сида — 300 замеров на политику. Таймер: io.prep (сырое
CPU-наблюдение -> входы) + decide + CPU float32 чанк + io.scale; begin и
описание задачи — до таймера. Карта: блокировка по UUID (как collect),
занятость до загрузки ≤ 2048 МиБ, иначе код 4 (latency_pending).
Записываются: исходная занятость, нагрузка CPU узла (loadavg), занятость
и загрузка остальных карт узла, версии среды. Отчёт: p50/p95 на политику
(по сидам и вместе) и отношения T2/T1, T2P/T1.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k36_bank as kb36  # noqa: E402
import k36_harness as kh  # noqa: E402
import k36_protocol as pc  # noqa: E402
import k37_confirm as kcf  # noqa: E402

GPU_MAX_MB = 2048
N_PER_TASK = 5
REPEATS = 10
WARM = 30
BLOCK = 5
CAPS = dict(checks=2 * 3 * N_PER_TASK * 10)


def queue_ids():
    return [(t, cid) for t in kcf.TASKS
            for cid, _p, _h, _hf in kb36.dev_cases(t, N_PER_TASK)]


def queue_sha(ids=None):
    ids = ids if ids is not None else queue_ids()
    return hashlib.sha1(json.dumps([[int(t), c] for t, c in ids]).encode()
                        ).hexdigest()


def collect_obs(sim, io, L, stamp):
    import k18_collector as k18c
    import k31_collect as kcol
    out = []
    for t, cid in queue_ids():
        pos, ih, ihf = next((p_, h, hf) for c, p_, h, hf in
                            kb36.dev_cases(t, N_PER_TASK) if c == cid)
        eid = f"checks/latency_obs/{stamp}/t{t}/{cid}"
        L.reserve("checks", eid, amount=kcol.ENV["waiting_steps"])
        io.seed(kcol.ENV["seed"])
        obs = sim.reset(t, [pos])
        dummy = np.array([kcol.DUMMY])
        for _ in range(kcol.ENV["waiting_steps"]):
            obs, _r, d, _s = sim.step(dummy)
            if bool(d[0]):
                raise SystemExit(f"{cid}: done во время ожидания")
        L.commit(eid, kcol.ENV["waiting_steps"])
        if ih is not None:
            h, hf = k18c.init_hashes(obs, 1)
            if (h[0], hf[0]) != (ih, ihf):
                raise SystemExit(f"{cid}: начальный хеш ≠ банку")
        out.append((t, {k: np.array(v, copy=True) for k, v in obs.items()}))
    return out


def q(x):
    x = np.asarray(x, np.float64)
    return dict(p50=float(np.quantile(x, 0.5)),
                p95=float(np.quantile(x, 0.95)), n=int(len(x)),
                mean=float(x.mean()))


def measure(torch, dev, queue, arms, n, warm=WARM, block=BLOCK, sync=None):
    """arms: {имя: (io, arm)} -> {имя: [мс]}; чередование блоками."""
    sync = sync or (lambda: torch.cuda.synchronize(dev))
    names = list(arms)
    descs = {t: arms[names[0]][0].task_desc(t) for t, _o in queue}

    def call(nm, item):
        io, arm = arms[nm]
        t, obs = item
        batch, po, s8 = io.prep(obs, descs[t], t)
        a, _rec = arm.decide(batch, po, s8, 0)
        cmd = io.scale(np.asarray(a, np.float32)[None])[0]
        return a, cmd
    for nm in names:
        for j in range(warm):
            arms[nm][1].begin(queue[j % len(queue)][0], 0, 0)
            call(nm, queue[j % len(queue)])
    meas = {nm: [] for nm in names}
    pos = {nm: 0 for nm in names}
    while min(len(v) for v in meas.values()) < n:
        for nm in names:
            for _ in range(block):
                if len(meas[nm]) >= n:
                    break
                item = queue[pos[nm] % len(queue)]
                pos[nm] += 1
                arms[nm][1].begin(item[0], 0, 0)
                sync()
                t0 = time.perf_counter()
                a, cmd = call(nm, item)
                sync()
                meas[nm].append((time.perf_counter() - t0) * 1e3)
                if np.asarray(a).dtype != np.float32 or \
                        not np.isfinite(a).all() or \
                        not np.isfinite(cmd).all():
                    raise SystemExit(f"{nm}: чанк/команда не конечны")
    return meas


def summarize(raw):
    """raw: {сид: {политика: [мс]}} -> сводка и отношения к T1."""
    pol = sorted({p_ for v in raw.values() for p_ in v})
    allp = {p_: [x for s in raw for x in raw[s][p_]] for p_ in pol}
    out = dict(by_seed={str(s): {p_: q(v) for p_, v in raw[s].items()}
                        for s in raw},
               all={p_: q(v) for p_, v in allp.items()})
    t1 = out["all"].get("T1")
    if t1:
        out["ratio_to_T1"] = {p_: dict(
            p50=out["all"][p_]["p50"] / t1["p50"],
            p95=out["all"][p_]["p95"] / t1["p95"]) for p_ in pol}
    for p_, v in out["all"].items():
        if not (np.isfinite(v["p50"]) and np.isfinite(v["p95"]) and
                v["p50"] > 0 and v["p95"] >= v["p50"]):
            raise SystemExit(f"{p_}: нечисловая или неупорядоченная сводка")
    return out


def node_gpus():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,"
             "utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=60)
        return [x.strip() for x in out.strip().splitlines()]
    except Exception as e:                                   # noqa: BLE001
        return [f"нет: {type(e).__name__}"]


def main():
    ap = argparse.ArgumentParser(description="K-37: задержка B1")
    ap.add_argument("--run-root", default=os.environ.get("K37_RUN_ROOT"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if not a.run_root:
        print("  ОТКАЗ: нужен --run-root")
        return 3
    try:
        return run(a)
    except kcf.Busy as e:
        print(f"  latency_pending: {e}")
        return 4
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


def run(a):
    plan, k36 = kcf.load_plan(a.run_root)
    idx = kcf.dev_index(a.device)
    g = kcf.gpu_query(idx)
    lk = kcf.Lock(os.path.join(kcf.lock_dir(), g["uuid"] + ".lock"),
                  f"карта {g['uuid']}", kcf.Busy)
    try:
        if g["used"] > GPU_MAX_MB:
            raise kcf.Busy(f"{a.device}: занято {g['used']} МиБ > "
                           f"{GPU_MAX_MB}")
        import k31_collect as kcol
        import torch
        import k29_policy as kp
        import k31_harness as kh31
        out_dir = os.path.join(a.run_root, "latency")
        L = kcol.Ledger(os.path.join(out_dir, "ledger.json"), CAPS)
        sim = kb36.BankSim("bank", kb36.SPACE_DEV)
        try:
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            io0 = kh31.IO(kp.Inputs(torch, None, "cpu"))
            queue = collect_obs(sim, io0, L, time.strftime("%Y%m%dT%H%M%S"))
        finally:
            kcf.close_sim(sim)
        g2 = kcf.gpu_query(idx)
        if g2["used"] > GPU_MAX_MB:
            raise kcf.Busy(f"{a.device}: занято {g2['used']} МиБ перед "
                           f"загрузкой")
        bind = kcf.bind_device(torch, idx, g2)      # до загрузки моделей
        H = kcf.load_heads(torch, plan)
        dev = torch.device(a.device)
        res = dict(plan_sha256=plan["sha256"], batch=1, warm=WARM,
                   block=BLOCK, repeats=REPEATS, queue=queue_ids(),
                   queue_sha1=queue_sha(), gpu=g2,
                   node_gpus_before=node_gpus(), loadavg=os.getloadavg(),
                   versions=kcf.env_versions(), device_binding=bind,
                   boundary="сырое CPU-наблюдение -> io.prep -> T32 (+ "
                            "головка) -> CPU float32 чанк -> io.scale",
                   time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        raw = {}
        for s in kcf.SEEDS:
            v = kh.load_source(torch, s, a.device)
            arms = kcf.make_arms(torch, k36, s, v, H)
            raw[s] = measure(torch, dev, queue,
                             {p_: (v["io"], arms[p_])
                              for p_ in kcf.POLICIES},
                             len(queue) * REPEATS)
            arms = v = None
            kh.free(torch)
        res["node_gpus_after"] = node_gpus()
        res["loadavg_after"] = os.getloadavg()
        res["raw_ms"] = {str(s): r for s, r in raw.items()}
        res["summary"] = summarize(raw)
        pc.save_json(os.path.join(out_dir, "latency.json"), res)
        sm = res["summary"]
        print("  задержка B1: " + "; ".join(
            f"{p_} p50 {sm['all'][p_]['p50']:.1f} / p95 "
            f"{sm['all'][p_]['p95']:.1f} мс (n {sm['all'][p_]['n']})"
            for p_ in kcf.POLICIES) + "; к T1 p50: " + ", ".join(
            f"{p_} {sm['ratio_to_T1'][p_]['p50']:.3f}"
            for p_ in ("T2", "T2P")))
        return 0
    finally:
        lk.release()


def selftest():
    class IO:
        def task_desc(self, t):
            return "x"

        def prep(self, obs, d, t):
            return None, 0, None

        def scale(self, a):
            return a[:, :, :7]

    class Arm:
        def __init__(self, k):
            self.k = k

        def begin(self, *a):
            pass

        def decide(self, *a):
            time.sleep(0.0005 * self.k)
            return np.zeros((8, 7), np.float32), {}
    queue = [(0, {}), (6, {}), (8, {})]
    raw = {s: measure(None, None, queue,
                      {"T1": (IO(), Arm(1)), "T2": (IO(), Arm(2)),
                       "T2P": (IO(), Arm(2))}, 20, warm=2,
                      sync=lambda: None) for s in (0, 1)}
    assert all(len(raw[s][p_]) == 20 for s in raw for p_ in raw[s])
    sm = summarize(raw)
    assert sm["all"]["T1"]["n"] == 40 and sm["ratio_to_T1"]["T2"]["p50"] > 1
    try:
        summarize({0: {"T1": [float("nan")] * 3}})
        raise AssertionError("NaN прошёл")
    except SystemExit:
        pass
    assert CAPS["checks"] == 300
    print("самопроверка k37_latency пройдена: чередование блоками, равный "
          "объём на политику, сводка по сидам и вместе, отношения к T1, "
          "NaN — отказ")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main())
