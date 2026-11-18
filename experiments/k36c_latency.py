#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-36 (§15): задержка B = 1 — Tplus, T32, R (сид 0) и original BAR.

После фиксации кандидата (план C). Очередь — СЫРЫЕ CPU-наблюдения (state +
кадры двух камер, как их отдаёт среда) после 10 waiting-шагов на 50
dev-сценах банка K-33 (5 на задачу, задачи 0…9, порядок idx); начальные
хеши сверяются с банком; физические шаги ожидания резервируются в checks
при каждом сборе. Сырых кадров в кеше K-21 нет (там уже обработанные),
поэтому очередь val_sel K-21 заменена этой — отклонение от формулировки
§15, записано явно.
Таймер (CUDA synchronize до и после) охватывает ПОЛНЫЙ путь: подготовку
наблюдения k29_policy.Inputs (переворот, обрезка/масштаб, склейка двух
камер, подсказка, процессор — как в раскатке) -> R/T32/Tplus
(k32_policy.Arm.decide: экспорт, reference, CPU actor, оператор,
ограничитель) либо BAR (generate + сборка кодов) -> CPU float32 чанк.
50 прогревов и 300 измерений на руку; R/T32/Tplus одной загрузки источника
чередуются блоками по 10; BAR — в отдельной области жизни модели. Critic
не загружается. Ревизия BAR = prepare K-36.
Отчёт: p50/p95 (конечные, > 0, p95 ≥ p50), пиковая память, доля вызовов с
нулевой правкой у T32/Tplus. Ориентир +20% к R — описательно. Только
свободная карта (k36bc_run.sh: занято ≤ 2048 МиБ, иначе latency_pending).
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

import k36_bank as kb36  # noqa: E402
import k36_harness as kh  # noqa: E402
import k36_protocol as pc  # noqa: E402

OUT = "reports/k36/latency.json"
ARMS = ("R_s0", "T32_s0", "Tplus_s0")
GPU_MAX_MB = 2048
PENDING = 4                      # код: карта занята -> latency_pending
N_PER_TASK = 5


def queue_ids():
    """Детерминированная очередь dev-сцен: задачи 0…9, по 5 (порядок idx)."""
    return [(t, cid) for t in range(10)
            for cid, _p, _h, _hf in kb36.dev_cases(t, N_PER_TASK)]


def queue_sha(ids=None):
    ids = ids if ids is not None else queue_ids()
    return hashlib.sha1(json.dumps([[int(t), c] for t, c in ids]).encode()
                        ).hexdigest()


def stats_ok(q):
    try:
        m, p95 = float(q["median"]), float(q["p95"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.isfinite(m) and np.isfinite(p95) and m > 0 and p95 >= m)


def latency_problems(plan_c_sha1, n=300, warm=50, qsha=None):
    """Отчёт задержки принадлежит этому плану C и очереди, полный объём, у
    всех четырёх рук конечные p50/p95 (> 0, p95 ≥ p50). Нет файла — None
    (latency_pending), без чтения данных."""
    if not os.path.exists(OUT):
        return None
    try:
        r = json.load(open(OUT, encoding="utf-8"))
        if not isinstance(r, dict):
            raise ValueError("не объект")
    except (OSError, ValueError) as e:
        return [f"отчёт задержки не читается: {e}"]
    p = []
    if r.get("plan_c_sha1") != plan_c_sha1:
        p.append("задержка другого плана C")
    if r.get("batch") != 1 or r.get("n") != n or r.get("warm") != warm:
        p.append("задержка: батч/объём ≠ 1 / 300 / 50")
    want_q = qsha if qsha is not None else queue_sha()
    try:
        q_list = queue_sha([(int(t), c) for t, c in r.get("queue") or []])
    except (TypeError, ValueError):
        q_list = None
    if r.get("queue_sha1") != want_q or q_list != want_q:
        p.append("задержка: другая очередь (SHA или фактический список)")
    used = r.get("mem_used_before_mb")
    if not isinstance(used, int) or not 0 <= used <= GPU_MAX_MB:
        p.append(f"задержка: занятость карты перед загрузкой {used} МиБ "
                 f"> {GPU_MAX_MB}")
    pol = r.get("policies") or {}
    bar = r.get("bar") or {}
    for k, v in [(k, pol.get(k)) for k in ARMS] + [("BAR", bar.get("BAR"))]:
        if not isinstance(v, dict) or v.get("n") != n or \
                not stats_ok(v.get("total")):
            p.append(f"задержка {k}: нет полного замера или p50/p95")
    if not p:
        want_s = summary_of(r)
        got = r.get("summary") or {}

        def same(a, b):
            return isinstance(a, (int, float)) and not isinstance(a, bool) \
                and np.isfinite(a) and np.isfinite(b) and \
                abs(a - b) <= 1e-9 * max(1.0, abs(b))
        if any(not same(got.get(k), v) for k, v in want_s.items()
               if k != "p50") or not isinstance(got.get("p50"), dict) or \
                set(got["p50"]) != set(want_s["p50"]) or \
                any(not same(got["p50"][k], v)
                    for k, v in want_s["p50"].items()):
            p.append("задержка: записанная сводка (медианы/отношения) ≠ "
                     "пересчёту из замеров или не конечна")
    return p


def summary_of(r):
    """Отношения — только из принятых медиан."""
    m = {k: float(r["policies"][k]["total"]["median"]) for k in ARMS}
    m["BAR"] = float(r["bar"]["BAR"]["total"]["median"])
    return dict(p50=m, Tplus_over_R=m["Tplus_s0"] / m["R_s0"],
                T32_over_R=m["T32_s0"] / m["R_s0"],
                Tplus_over_BAR=m["Tplus_s0"] / m["BAR"])


def gpu_used_mb(device):
    """Занятость карты по nvidia-smi (до CUDA-контекста процесса); ответ
    не число (N/A) или сбой — технический отказ, а не «карта занята»."""
    import subprocess
    idx = str(device).split(":")[-1]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits", "-i", idx],
                             capture_output=True, text=True, check=True)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError,
            IndexError) as e:
        raise SystemExit(f"nvidia-smi: занятость {device} не прочитана "
                         f"({type(e).__name__})")


def collect_obs(sim, io, L, stamp):
    """Сырые наблюдения после ожидания на dev-сценах; каждый сбор —
    свои резервы checks (повторный замер честно расходует шаги)."""
    import k18_collector as k18c
    import k31_collect as kcol
    import k36b_gate as kg
    out = []
    for t, cid in queue_ids():
        pos, ih, ihf = next((p_, h, hf) for c, p_, h, hf in
                            kb36.dev_cases(t, N_PER_TASK) if c == cid)
        eid = f"checks/latency_obs/{stamp}/t{t}/{cid}"
        pr = kg.pending_problems(L, [("checks", "checks/latency_obs/")])
        if pr:
            raise SystemExit("; ".join(pr))
        L.reserve("checks", eid, amount=kcol.ENV["waiting_steps"])
        io.seed(kcol.ENV["seed"])
        obs = sim.reset(t, [pos])
        dummy = np.array([kcol.DUMMY])
        for _ in range(kcol.ENV["waiting_steps"]):
            obs, _r, d, _s = sim.step(dummy)
            if bool(d[0]):
                raise SystemExit(f"{cid}: done во время ожидания")
        L.commit(eid, kcol.ENV["waiting_steps"])
        h, hf = k18c.init_hashes(obs, 1)
        if (h[0], hf[0]) != (ih, ihf):
            raise SystemExit(f"{cid}: начальный хеш ≠ банку")
        out.append((t, {k: np.array(v, copy=True) for k, v in obs.items()}))
    return out


def measure(torch, dev, queue, arms, n, warm, block=10):
    """arms: {метка: (io, arm)}; одна область жизни модели; таймер —
    от сырого наблюдения до CPU float32 чанка."""
    import k21_latency as kl
    names = list(arms)

    descs = {t: arms[names[0]][0].task_desc(t) for t, _o in queue}

    def prepare(nm, item):              # вне таймера: раз на эпизод
        arms[nm][1].begin(item[0], 0, 0)

    def call(nm, item):                 # таймер: наблюдение -> команда
        io, arm = arms[nm]
        t, obs = item
        batch, po, s8 = io.prep(obs, descs[t], t)
        a, rec = arm.decide(batch, po, s8, 0)
        cmd = io.scale(np.asarray(a, np.float32)[None])[0]
        return a, cmd, rec
    for nm in names:
        for j in range(warm):
            prepare(nm, queue[j % len(queue)])
            call(nm, queue[j % len(queue)])
    torch.cuda.reset_peak_memory_stats(dev)
    meas = {nm: [] for nm in names}
    zero = {nm: [] for nm in names}
    pos = {nm: 0 for nm in names}
    while min(len(v) for v in meas.values()) < n:
        for nm in names:
            for _ in range(block):
                if len(meas[nm]) >= n:
                    break
                item = queue[pos[nm] % len(queue)]
                pos[nm] += 1
                prepare(nm, item)
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                a, cmd, rec = call(nm, item)
                torch.cuda.synchronize(dev)
                meas[nm].append((time.perf_counter() - t0) * 1e3)
                if np.asarray(a).dtype != np.float32 or \
                        not np.isfinite(a).all() or \
                        not np.isfinite(cmd).all():
                    raise SystemExit(f"{nm}: чанк/команда не конечны")
                if "zero" in rec:
                    zero[nm].append(bool(rec["zero"]))
    out = {nm: dict(total=kl.q(meas[nm]), n=len(meas[nm]),
                    zero_share=float(np.mean(zero[nm])) if zero[nm] else None)
           for nm in names}
    out["peak_memory_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    return out


def _policy_lifetime(torch, prep, dev, queue, a):
    import k36b_screen as kbs
    v = kh.load_source(torch, 0, a.device)
    arms = dict(R_s0=(v["io"], kh.make_arm(torch, prep, "R_s0", v)),
                T32_s0=(v["io"], kh.make_arm(torch, prep, "T32_s0", v)),
                Tplus_s0=(v["io"], kh.make_arm(
                    torch, prep, "T32_s0", v,
                    actor=kbs.load_tplus(torch, prep, 0, v["mod"].H))))
    return measure(torch, dev, queue, arms, a.n, a.warm)


def _bar_lifetime(torch, prep, dev, queue, a):
    import k36c_bar as kbar
    b = kbar.load_bar(torch, a.device, prep)
    return measure(torch, dev, queue,
                   dict(BAR=(b["io"], kbar.BarArm(torch, b["bar"]))),
                   a.n, a.warm)


def main():
    ap = argparse.ArgumentParser(description="K-36: задержка B1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warm", type=int, default=50)
    a = ap.parse_args()
    sim = kb36.BankSim("bank", kb36.SPACE_DEV)       # до CUDA
    import torch
    import k29_policy as kp
    import k31_harness as kh31
    import k36c_confirm as kc
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        prep = pc.load_prepare()
        p = kc.plan_c_problems(prep)
        if p:
            raise SystemExit("план C: " + "; ".join(p[:3]))
        L = pc.ledger()
        io0 = kh31.IO(kp.Inputs(torch, None, "cpu"))
        queue = collect_obs(sim, io0, L, time.strftime("%Y%m%dT%H%M%S"))
    finally:
        sim.close()
    used = gpu_used_mb(a.device)       # непосредственно перед загрузкой
    if used > GPU_MAX_MB:
        print(f"  latency_pending: на {a.device} занято {used} МиБ > "
              f"{GPU_MAX_MB} после сбора очереди")
        return PENDING
    dev = torch.device(a.device)
    res = dict(plan_c_sha1=pc.sha(kc.PLAN_C), batch=1, n=a.n, warm=a.warm,
               queue=[[t, c] for t, c in queue_ids()], queue_sha1=queue_sha(),
               device=torch.cuda.get_device_name(dev),
               mem_used_before_mb=int(used),
               threads=dict(intra=torch.get_num_threads(),
                            inter=torch.get_num_interop_threads(),
                            omp=os.environ.get("OMP_NUM_THREADS")),
               boundary="сырое CPU-наблюдение -> Inputs (кадры, подсказка, "
                        "процессор) -> политика -> CPU float32 чанк",
               queue_note="50 dev-сцен банка K-33 (сырые кадры); очередь "
                          "val_sel K-21 содержит только обработанные кадры")
    res["policies"] = _policy_lifetime(torch, prep, dev, queue, a)
    kh.free(torch)
    res["mem_after_policies_gb"] = torch.cuda.memory_allocated(dev) / 2 ** 30
    res["bar"] = _bar_lifetime(torch, prep, dev, queue, a)
    kh.free(torch)
    res["summary"] = summary_of(res)
    m = res["summary"]["p50"]
    res["summary_note"] = ("ориентир +20% к R — описательно; задержка — "
                           "модели сида 0; p50/p95 ПЕРВОГО вызова после "
                           "ожидания на 50 dev-сценах (по 6 измерений на "
                           "наблюдение)")
    pc.save_json(OUT, res)
    pr = latency_problems(res["plan_c_sha1"], a.n, a.warm)
    if pr:
        raise SystemExit("отчёт задержки неполон: " + "; ".join(pr))
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
            assert latency_problems("P", qsha="Q") is None   # нет -> pending
            Q = None
            ok = dict(total=dict(median=120.0, p95=130.0), n=300)
            ql = [[0, "a"], [1, "b"]]
            Q = queue_sha([(t, c) for t, c in ql])
            good = dict(plan_c_sha1="P", batch=1, n=300, warm=50,
                        queue=ql, queue_sha1=Q, mem_used_before_mb=512,
                        policies={k: dict(ok) for k in ARMS},
                        bar=dict(BAR=dict(ok)))
            good["summary"] = summary_of(good)
            pc.save_json(g["OUT"], good)
            assert latency_problems("P", qsha=Q) == []
            nop = {k: dict(n=300) for k in ARMS}
            for bad in (dict(good, plan_c_sha1="X"), dict(good, n=1),
                        dict(good, warm=0), dict(good, queue_sha1="Z"),
                        dict(good, policies={"R_s0": dict(ok)}),
                        dict(good, bar={"BAR": dict(n=1)}),
                        dict(good, policies=nop),          # без p50/p95
                        dict(good, bar=dict(BAR=dict(
                            total=dict(median=120.0, p95=100.0), n=300))),
                        dict(good, bar=dict(BAR=dict(
                            total=dict(median=float("nan"), p95=1.0),
                            n=300)))):
                pc.save_json(g["OUT"], bad)
                assert latency_problems("P", qsha=Q), bad
            # записанное отношение 0.5 при медианах 15/10 мс — отказ
            tw = dict(good, policies=dict(good["policies"], Tplus_s0=dict(
                total=dict(median=15.0, p95=16.0), n=300)),
                bar=dict(BAR=dict(total=dict(median=10.0, p95=11.0), n=300)))
            tw["summary"] = dict(summary_of(tw), Tplus_over_BAR=0.5)
            pc.save_json(g["OUT"], tw)
            assert latency_problems("P", qsha=Q)
            assert abs(summary_of(tw)["Tplus_over_BAR"] - 1.5) < 1e-12
            for bad in (dict(good, queue=[[0, "a"]]),
                        dict(good, mem_used_before_mb=8192),
                        dict(good, summary=dict(good["summary"],
                                                Tplus_over_BAR=float("nan"))),
                        dict(good, summary=dict(good["summary"], p50=dict(
                            good["summary"]["p50"], BAR=1.0)))):
                pc.save_json(g["OUT"], bad)
                assert latency_problems("P", qsha=Q), bad
            # медиана 5e−324 мс: отношение Inf уже при пересчёте — отказ
            tiny = dict(good, bar=dict(BAR=dict(total=dict(
                median=5e-324, p95=1e-300), n=300)))
            tiny["summary"] = dict(good["summary"])
            for bad in (tiny,):
                pc.save_json(g["OUT"], bad)
                assert latency_problems("P", qsha=Q), bad
            open(g["OUT"], "w").write("{испорчен")
            assert latency_problems("P", qsha=Q)
        finally:
            g["OUT"] = keep
    assert stats_ok(dict(median=1.0, p95=1.0)) and \
        not stats_ok(dict(median=0.0, p95=1.0))
    assert queue_sha([(0, "a")]) == queue_sha([(0, "a")]) != \
        queue_sha([(0, "b")])
    print("самопроверка k36c_latency пройдена: задержка чужого плана C, "
          "неполного объёма, другой очереди, без руки, без p50/p95, с p95 < "
          "p50 или NaN, испорченный файл, неверное/NaN отношение, ложная "
          "медиана сводки, "
          "чужой список очереди, занятая карта — отказ; нет файла — "
          "latency_pending без чтения данных (замер — на GPU)")
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
