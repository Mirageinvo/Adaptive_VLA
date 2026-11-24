#!/usr/bin/env python3
"""K-39: перенос ветви 2×2 на обычные старты задачи 8 (архив c2).

Ветвь — одна из GripOnly / ArmOnly / RefreshOnce с наибольшим взвешенным
приростом на оставшихся стартах (weighted.rest) среди прошедших +3 п.п. в
summary.json прогона k39_grip_probe. Нет такой — no_gain, сбора нет.
Сцены: 64 физические сцены c2 задачи 8 по хешу, без отбора по исходу T1 и
триггеру, не из natural_probe/v1 и не из 2×2; оба сида — 128 пар. T1 — архив
(повтор не нужен там, где нет вмешательства). Где сработал онлайн-триггер —
повтор T1 до o4 (сверка с архивом) и одна раскатка ветви. Без триггера
исход ветви = исход T1. Анализ: SR T1 и ветви, W/L/D, Δ, 95% ДИ — bootstrap
по физическим сценам (оба сида вместе), 50 000, seed 39003; доля
срабатываний триггера.

Команды: --prepare-only | --part K --device D | --summarize.
"""

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import socket

import numpy as np

from k31_collect import ENV, EXEC_MAX, Ledger

TASK = 8
N_SCENES = 64
PROTOCOL = "k39-grip-transfer-v1"
SCOPE = "grip_transfer"
CASE_RESERVE = 2 * EXEC_MAX
N_PARTS = 2
BOOT = dict(n=50_000, seed=39003)
CODE = ("k39_grip_transfer.py", "k39_grip_probe.py", "k39_failure_probe.py",
        "k39_cache.py", "k39_pulse.py")


def rank(key):
    return hashlib.sha256(f"{PROTOCOL}/{key}".encode()).hexdigest()


def pick_arm(gplan, gsum):
    import k39_grip_probe as gp
    if gsum.get("plan") != gplan["sha256"]:
        raise SystemExit("summary 2×2 не от этого плана")
    w = gsum["weighted"]["rest"]
    ok = [a for a in gp.ARMS if w[a]["est"] >= gp.THRESHOLD_PP]
    if sorted(ok) != sorted(gsum.get("signals", [])):
        raise SystemExit("signals в summary ≠ пересчёту")
    return max(ok, key=lambda a: (w[a]["est"], a)) if ok else None


def tail_for(arm, B4, R4):
    import k39_grip_probe as gp
    return dict(GripOnly=gp.grip_tail, ArmOnly=gp.arm_tail,
                RefreshOnce=lambda b, r: np.array(r, np.float32, copy=True)
                )[arm](B4, R4)


def prepare(a):
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    import k39_failure_probe as kfp
    import k39_grip_probe as gp
    gplan = pc.load_json(str(a.grip / "plan.json"))
    gsp = a.grip / "summary.json"
    if not gsp.exists():
        raise SystemExit("нет summary.json прогона 2×2")
    gsum = pc.load_json(str(gsp))
    arm = pick_arm(gplan, gsum)
    v1p = pc.load_json(str(a.v1 / "plan.json"))
    if v1p.get("sha256") != gplan["v1_plan"]:
        raise SystemExit("v1 ≠ плану 2×2")
    used = {j["want"]["case_id"] for j in v1p["jobs"]} | \
        {j["want"]["case_id"] for j in gplan["jobs"]}
    plan, k36 = kcf.load_plan(str(a.archive))
    if plan["sha256"] != gplan["archive_plan"]:
        raise SystemExit("архив c2 ≠ плану 2×2")
    C = kh.consts(k36)
    merged = pc.load_json(a.archive / "merged.json")
    if merged.get("ok") is not True or \
            merged.get("plan_sha256") != plan["sha256"]:
        raise SystemExit("merged.json не подтверждает архивный план")
    rows = {r["eid"]: r for r in merged["rows"]
            if r["policy"] == "T1" and r["task"] == TASK}
    eps = []
    for part in range(kcf.N_PARTS):
        led = pc.load_json(Path(kcf.part_dir(str(a.archive), part)) /
                           "ledger.json")
        if led["pending"] is not None or led["caps"] != kcf.CAP_PART:
            raise SystemExit(f"часть {part}: открытый резерв или лимиты")
        for w in kcf.part_wants(plan, k36, part):
            if w["task"] != TASK or w["kind"] != "T":
                continue
            r = rows.pop(w["eid"], None)
            if r is None or r["part"] != part or r["seed"] != w["seed"] or \
                    r["case_id"] != w["case_id"]:
                raise SystemExit(f"{w['eid']}: merged.json ≠ плану")
            if w["case_id"] in used:
                continue
            rec, z = kfp.reference(a.archive, part, w)
            succ = rec["summary"]["success"]
            pr = kh.episode_problems(rec, z, w, C)
            if pr or succ is not r["success"] or led["done"].get(
                    w["eid"]) != {"scope": "confirm", "actual": rec["phys"]}:
                raise SystemExit(f"{w['eid']}: архив не прошёл сверку ({pr})")
            J = {k[2:]: z[k] for k in z if k.startswith("j_")}
            pt = gp.online_point(z["actions"], J)
            eps.append(dict(want=w, arch_part=part, success=succ,
                            g=None if pt is None else pt[0],
                            t=None if pt is None else pt[1],
                            physical=rec["phys"], npz_sha1=rec["npz_sha1"]))
    if rows:
        raise SystemExit("в merged.json остались записи T1 вне плана")
    cases = sorted({e["want"]["case_id"] for e in eps}, key=rank)
    if len(cases) < N_SCENES:
        raise SystemExit(f"свободных сцен {len(cases)} < {N_SCENES}")
    take = set(cases[:N_SCENES])
    eps = sorted((e for e in eps if e["want"]["case_id"] in take),
                 key=lambda e: (e["want"]["seed"], rank(e["want"]["eid"])))
    if Counter(e["want"]["case_id"] for e in eps) != {c: 2 for c in take}:
        raise SystemExit("не у всех сцен оба сида")
    jobs = [e for e in eps if e["t"] is not None]
    for i, j in enumerate(jobs):
        j["part"] = i % N_PARTS
    here = Path(__file__).resolve().parent
    out = dict(protocol=PROTOCOL, arm=arm, grip_plan=gplan["sha256"],
               grip_summary_sha256=kcf.sha256(gsp),
               archive_plan=plan["sha256"], v1_plan=v1p["sha256"],
               code={n: kcf.sha256(here / n) for n in CODE},
               episodes=eps, n_trigger=len(jobs),
               caps={SCOPE: (len(jobs) // N_PARTS + 1) * CASE_RESERVE},
               boot=BOOT)
    return out, k36


# --- сбор --------------------------------------------------------------------------------
def part_dir(out, k):
    return out / f"part-{int(k)}"


def scene_files(out, job, plan):
    w = job["want"]
    name = f"s{w['seed']}_{w['case_id']}"
    d = part_dir(out, job["part"])
    ident = dict(eid=w["eid"], plan=plan["sha256"], seed=w["seed"],
                 t1=job["success"], t=job["t"], g=job["g"], arm=plan["arm"],
                 archive_npz_sha1=job["npz_sha1"], physical=job["physical"])
    return name, d / f"{name}.json", d / f"{name}.png", ident


def run_job(sim, io, armobj, job, rec, z, arm):
    from k31_collect import run_episode
    from k36_harness import SUMMARY, episode_number
    import k39_failure_probe as kfp
    import k39_grip_probe as gp
    w, t = job["want"], job["t"]
    eid = w["eid"]
    B, R4, o0, o4, hashes, frames = kfp.replay(sim, io, armobj, job, rec, z)
    suffix = tail_for(arm, B[4:], R4)
    wrapped = kfp.CheckedMidpoint(sim, ENV["waiting_steps"] + t + 4, o4)
    armobj.episode = episode_number(eid)
    A = z["actions"]
    ep = run_episode(wrapped, io, armobj, w["task"], w["reset_id"],
                     prefix=A[:t], fork=dict(
                         tau=t, hash=o0,
                         a_codec=np.concatenate((B[:4], suffix))))
    if wrapped.count < wrapped.steps or \
            (ep["init_hash"], ep["init_hash_full"]) != hashes or \
            not np.array_equal(ep["actions"][:t + 4], A[:t + 4]):
        raise SystemExit(f"{eid}: общий префикс ≠ архиву")
    same = bool(np.array_equal(suffix, B[4:]))
    if same and any(ep[k] != rec["summary"][k] for k in
                    ("success", "success_step", "done_step", "steps",
                     "actions_sha1")):
        raise SystemExit(f"{eid}: хвост = B, эпизод ≠ T1")
    return dict(branch={k: ep[k] for k in (*SUMMARY, "seconds")}, o0=o0,
                o4=o4, tail_same=same, grip=gp.grip_info(B[4:], R4)), frames


def checked_result(path, fp, ident):
    import k36_harness as kh
    import k36_protocol as pc
    row = pc.load_json(path)
    if not isinstance(row, dict) or any(
            type(row.get(k)) is not type(v) or row[k] != v
            for k, v in ident.items()) or not fp.is_file():
        raise SystemExit(f"{path}: запись не соответствует плану")
    if any(not isinstance(row.get(k), str) or not row[k]
           for k in ("o0", "o4")) or not isinstance(row.get("tail_same"),
                                                     bool):
        raise SystemExit(f"{path}: нет хешей или tail_same")
    ep = row.get("branch")
    if not isinstance(ep, dict):
        raise SystemExit(f"{path}: нет ветви")
    pr = kh.summary_problems(dict(phys=ep.get("physical")), ep,
                             ep.get("steps"))
    if pr or ep["steps"] < row["t"] + 4:
        raise SystemExit(f"{path}: неверная сводка ({pr})")
    if row["tail_same"] and ep["success"] != row["t1"]:
        raise SystemExit(f"{path}: хвост = B, исход ≠ T1")
    return row


def case_spend(row):
    return row["physical"] + row["branch"]["physical"]


def restore_rows(out, plan, part, ledger):
    jobs = {f"{plan['sha256']}/{j['want']['eid']}": j
            for j in plan["episodes"] if j["t"] is not None and
            j["part"] == part}
    done, pending = ledger.d["done"], ledger.d["pending"]
    pid = ledger.pending_id()
    if not set(done).issubset(jobs):
        raise SystemExit("ledger: учтённые эпизоды вне части")
    if pending is not None and (pid not in jobs or pid in done or pending !=
                                dict(id=pid, scope=SCOPE,
                                     amount=CASE_RESERVE)):
        raise SystemExit("ledger: открытый резерв не из части")
    for c in ledger.d["crashes"]:
        if c.get("id") not in jobs or c.get("scope") != SCOPE or \
                c.get("amount") != CASE_RESERVE:
            raise SystemExit("ledger: авария не из части")
    rows = {}
    for key, j in jobs.items():
        _n, path, fp, ident = scene_files(out, j, plan)
        if path.exists():
            row = checked_result(path, fp, ident)
            if key != pid and done.get(key) != dict(scope=SCOPE,
                                                    actual=case_spend(row)):
                raise SystemExit(f"{path}: расход ≠ записи журнала")
            rows[key] = row
        elif key in done:
            raise SystemExit(f"{path}: учтён, результата нет")
    spent = sum(case_spend(rows[k]) for k in done) + \
        CASE_RESERVE * len(ledger.d["crashes"])
    if ledger.d["spent"] != {SCOPE: spent}:
        raise SystemExit("ledger: расход не сходится")
    if pid in rows:
        ledger.settle(pid, case_spend(rows[pid]))
    return rows


def collect(a, plan, k36):
    import torch
    import k36_bank as kb36
    import k36_harness as kh
    import k36_protocol as pc
    import k37_bank as kbk
    import k37_confirm as kcf
    import k39_failure_probe as kfp
    from k38_grip_run import threads_problems
    if a.part not in range(N_PARTS):
        raise SystemExit(f"часть {a.part}")
    pd_ = part_dir(a.out, a.part)
    pd_.mkdir(parents=True, exist_ok=True)
    plock = kcf.Lock(str(pd_ / ".part.lock"), f"part {a.part}", kcf.Busy)
    try:
        ledger = Ledger(str(pd_ / "ledger.json"), plan["caps"])
        rows = restore_rows(a.out, plan, a.part, ledger)
        mine = [j for j in plan["episodes"] if j["t"] is not None and
                j["part"] == a.part]
        todo = [j for j in mine
                if f"{plan['sha256']}/{j['want']['eid']}" not in rows]
        if not todo:
            print(f"  part {a.part}: готова ({len(rows)})", flush=True)
            return
        if pr := threads_problems(torch):
            raise SystemExit(f"потоки: {pr}")
        idx = kcf.dev_index(a.device)
        gpu = kcf.gpu_query(idx)
        lock = kcf.Lock(str(Path(kcf.lock_dir()) / (gpu["uuid"] + ".lock")),
                        gpu["uuid"], kcf.Busy)
        sim = None
        try:
            kcf.check_gpu(idx, kcf.RES["min_free_gb"])
            kcf.check_disk(a.out)
            binding = kcf.bind_device(torch, idx, gpu)
            adm = dict(host=socket.gethostname(), uuid=gpu["uuid"],
                       pci=gpu["pci"], binding=binding,
                       threads=torch.get_num_threads())
            sim = kb36.BankSim("bank", kbk.SPACE, bank_dir=kbk.BANK_DIR)
            for seed in (0, 1):
                src = armobj = None
                for j in todo:
                    w = j["want"]
                    if w["seed"] != seed:
                        continue
                    key = f"{plan['sha256']}/{w['eid']}"
                    name, path, fp, ident = scene_files(a.out, j, plan)
                    if src is None:
                        kcf.check_gpu(idx, kcf.RES["min_free_gb"])
                        src = kh.load_source(torch, seed, a.device)
                        armobj = kh.make_arm(torch, k36, f"T32_s{seed}", src)
                    rec, z = kfp.reference(a.archive, j["arch_part"], w)
                    if rec["npz_sha1"] != j["npz_sha1"] or \
                            rec["phys"] != j["physical"] or \
                            rec["summary"]["success"] is not j["success"]:
                        raise SystemExit(f"{w['eid']}: архив изменился")
                    kcf.check_disk(a.out)
                    ledger.reserve(SCOPE, key, amount=CASE_RESERVE)
                    res, frames = run_job(sim, src["io"], armobj, j, rec, z,
                                          plan["arm"])
                    row = dict(ident, **res, gpu=adm)
                    kfp.save_frames(fp, frames)
                    pc.save_json(str(path), row)
                    if checked_result(path, fp, ident) != row:
                        raise SystemExit(f"{path}: запись изменилась")
                    ledger.commit(key, case_spend(row))
                    rows[key] = row
                    print(f"  {name}: T1 {int(row['t1'])}, {plan['arm']} "
                          f"{int(row['branch']['success'])}; "
                          f"{len(rows)}/{len(mine)}", flush=True)
                if src is not None:
                    src = armobj = None
                    kh.free(torch)
        finally:
            if sim is not None:
                kcf.close_sim(sim)
            lock.release()
        print(f"  part {a.part}: готова", flush=True)
    finally:
        plock.release()


# --- анализ -------------------------------------------------------------------------------
def outcomes(plan, rows_by_eid):
    """-> [(case_id, seed, y_T1, y_arm, trigger)] по всем 128 эпизодам."""
    out = []
    for e in plan["episodes"]:
        w = e["want"]
        y1 = int(e["success"])
        if e["t"] is None:
            out.append((w["case_id"], w["seed"], y1, y1, False))
        else:
            r = rows_by_eid[w["eid"]]
            out.append((w["case_id"], w["seed"], y1,
                        int(r["branch"]["success"]), True))
    return out


def stats(oc, n_boot=BOOT["n"], seed=BOOT["seed"]):
    d = np.array([y2 - y1 for _c, _s, y1, y2, _t in oc])
    cases = sorted({c for c, *_ in oc})
    per = {c: [] for c in cases}
    for (c, _s, y1, y2, _t) in oc:
        per[c].append(y2 - y1)
    m = np.array([np.mean(per[c]) for c in cases])
    rng = np.random.default_rng(seed)
    b = m[rng.integers(0, len(m), (n_boot, len(m)))].mean(1)
    lo, hi = np.quantile(100 * b, [0.025, 0.975])
    trig = [x for x in oc if x[4]]
    return dict(
        N=len(oc), scenes=len(cases),
        sr_T1=float(np.mean([y1 for _c, _s, y1, _y, _t in oc])),
        sr_arm=float(np.mean([y2 for _c, _s, _y, y2, _t in oc])),
        W=int((d == 1).sum()), L=int((d == -1).sum()), D=int(d.sum()),
        delta_pp=float(100 * m.mean()), lo=float(lo), hi=float(hi),
        trigger_share=len(trig) / len(oc),
        trigger_by_t1={"fail": sum(1 for x in trig if not x[2]),
                       "success": sum(1 for x in trig if x[2])},
        per_seed={str(s): int(sum(y2 - y1 for _c, ss, y1, y2, _t in oc
                                  if ss == s)) for s in (0, 1)})


def summarize_cmd(a, plan):
    import k36_protocol as pc
    import k37_confirm as kcf
    rows = {}
    for k in range(N_PARTS):
        n_k = sum(1 for j in plan["episodes"]
                  if j["t"] is not None and j["part"] == k)
        p = part_dir(a.out, k) / "ledger.json"
        if n_k == 0:
            continue
        if not p.exists():
            raise SystemExit(f"part {k}: нет журнала")
        lk = kcf.Lock(str(part_dir(a.out, k) / ".part.lock"), f"part {k}",
                      kcf.Busy)
        try:
            led = Ledger(str(p), plan["caps"])
            if led.pending_id() is not None:
                raise SystemExit(f"part {k}: открытый резерв — сначала "
                                 f"перезапустите сбор части")
            rr = restore_rows(a.out, plan, k, led)
        finally:
            lk.release()
        if len(rr) != n_k:
            raise SystemExit(f"part {k}: неполная")
        rows.update({r["eid"]: r for r in rr.values()})
    st = stats(outcomes(plan, rows))
    pc.save_json(str(a.out / "summary.json"), dict(
        st, plan=plan["sha256"], arm=plan["arm"], rows=list(rows.values())))
    print(f"  перенос {plan['arm']} на {st['scenes']} сцен × 2 сида "
          f"(N {st['N']}): SR T1 {100 * st['sr_T1']:.1f}, ветвь "
          f"{100 * st['sr_arm']:.1f}; W {st['W']} L {st['L']} D "
          f"{st['D']:+d}; Δ {st['delta_pp']:+.1f} п.п. 95% "
          f"[{st['lo']:+.1f}; {st['hi']:+.1f}]; по сидам {st['per_seed']}; "
          f"триггер {100 * st['trigger_share']:.0f}% "
          f"{st['trigger_by_t1']}", flush=True)


def selftest():
    import json as _json
    import tempfile
    # выбор ветви
    gplan = dict(sha256="G")
    w = {a: dict(est=e) for a, e in (("GripOnly", 3.5), ("ArmOnly", 1.0),
                                     ("RefreshOnce", 4.2))}
    gs = dict(plan="G", weighted=dict(rest=w),
              signals=["GripOnly", "RefreshOnce"])
    assert pick_arm(gplan, gs) == "RefreshOnce"
    w2 = {a: dict(est=1.0) for a in w}
    assert pick_arm(gplan, dict(gs, weighted=dict(rest=w2),
                                signals=[])) is None
    try:
        pick_arm(gplan, dict(gs, plan="X"))
        raise AssertionError
    except SystemExit:
        pass
    # хвосты
    B4 = np.zeros((4, 7), np.float32)
    B4[:, 6] = 1
    R4 = np.ones((4, 7), np.float32) * 0.1
    R4[:, 6] = -1
    assert np.array_equal(tail_for("GripOnly", B4, R4)[:, 6], R4[:, 6]) and \
        np.array_equal(tail_for("GripOnly", B4, R4)[:, :6], B4[:, :6])
    assert np.array_equal(tail_for("ArmOnly", B4, R4)[:, :6], R4[:, :6])
    assert np.array_equal(tail_for("RefreshOnce", B4, R4), R4)
    # анализ: эпизоды без триггера = T1, ДИ по сценам
    plan = dict(episodes=[])
    rows = {}
    for c in range(64):
        for s in (0, 1):
            trig = c % 4 != 0
            e = dict(want=dict(case_id=f"c{c}", seed=s, eid=f"e{c}_{s}"),
                     success=c % 3 != 0, t=8 if trig else None)
            plan["episodes"].append(e)
            if trig:
                rows[f"e{c}_{s}"] = dict(branch=dict(
                    success=e["success"] or (c % 6 == 3 and s == 0)))
    st = stats(outcomes(plan, rows), n_boot=500)
    want_w = sum(1 for c in range(64) if c % 4 != 0 and c % 3 == 0 and
                 c % 6 == 3)
    assert st["W"] == want_w and st["L"] == 0 and st["N"] == 128
    assert abs(st["delta_pp"] - 100 * want_w / 128) < 1e-9
    assert st["trigger_share"] == 96 / 128
    # запись: проверка tail_same
    ep = dict(success=True, success_step=5, done_step=9, own_steps=10,
              steps=10, physical=20, init_hash="a", init_hash_full="b",
              actions_sha1="c", seconds=1.0)
    ident = dict(eid="e", plan="P", seed=0, t1=True, t=0, g=5,
                 arm="GripOnly", archive_npz_sha1="z", physical=20)
    row = dict(ident, o0="h0", o4="h4", tail_same=True, grip={},
               branch=dict(ep))
    with tempfile.TemporaryDirectory() as td:
        pth, fp = Path(td) / "r.json", Path(td) / "r.png"
        fp.write_bytes(b"x")
        pth.write_text(_json.dumps(row))
        assert checked_result(pth, fp, ident)["t1"] is True
        pth.write_text(_json.dumps(dict(row, branch=dict(
            ep, success=False, success_step=-1))))
        try:
            checked_result(pth, fp, ident)
            raise AssertionError("tail_same с другим исходом принят")
        except SystemExit:
            pass
    print("самопроверка k39_grip_transfer пройдена: выбор ветви по "
          "weighted.rest и signals; хвосты трёх ветвей; эпизоды без "
          "триггера = T1; W/L/D, Δ, доля триггера; tail_same")
    return 0


def main():
    import sys
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path,
                    default=Path("reports/k37c/runs/c2"))
    ap.add_argument("--v1", type=Path,
                    default=Path("reports/k39/natural_probe/v1"))
    ap.add_argument("--grip", type=Path,
                    default=Path("reports/k39/grip_probe/v1"))
    ap.add_argument("--out", type=Path,
                    default=Path("reports/k39/grip_transfer/v1"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--part", type=int)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    import k36_protocol as pc
    import k37_confirm as kcf
    if a.out.resolve().is_relative_to(a.archive.resolve()):
        raise SystemExit("каталог результатов должен быть вне архива c2")
    plan, k36 = prepare(a)
    if plan["arm"] is None:
        print("  2×2: ни одна ветвь не прошла порог — перенос не нужен "
              "(no_gain)", flush=True)
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    plan["sha256"] = kcf.canon_sha(plan)
    pp = a.out / "plan.json"
    if a.prepare_only:
        lk = kcf.Lock(str(a.out / ".plan.lock"), str(a.out), kcf.Busy)
        try:
            if pp.exists() and pc.load_json(str(pp)) != plan:
                raise SystemExit("план изменился, укажите новый --out")
            if not pp.exists():
                pc.save_json(str(pp), plan)
        finally:
            lk.release()
        print(f"  перенос {plan['arm']}: {N_SCENES} сцен, "
              f"{len(plan['episodes'])} эпизодов, с триггером "
              f"{plan['n_trigger']}; SHA {plan['sha256'][:16]}", flush=True)
        return 0
    if not pp.exists():
        raise SystemExit("плана нет — сначала --prepare-only")
    if pc.load_json(str(pp)) != plan:
        raise SystemExit("план на диске ≠ пересчитанному")
    if a.summarize:
        summarize_cmd(a, plan)
        return 0
    if a.part is None:
        raise SystemExit("нужен --part 0|1")
    collect(a, plan, k36)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
