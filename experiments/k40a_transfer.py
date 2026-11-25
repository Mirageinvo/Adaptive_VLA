#!/usr/bin/env python3
"""K-40 A: перенос ArmOnly на все свободные сцены t8 c2 + обучающие данные.

Ветвь задана явно: ArmOnly (рука R[:4], схват B[4:8]); R — полный T32 в o4
(elapsed t + 4). Сцены: все 88 физических сцен c2 задачи 8 вне
natural_probe/v1 и 2×2, оба сида — 176 пар; отбор не зависит от исхода T1 и
триггера. Онлайн-триггер — k39_grip_probe.online_point. Без триггера исход
ArmOnly = исход T1 (архив).
Для каждого старта с триггером (повтор T1 до конца со сверкой + раскатка
ArmOnly) пишется npz обучающих данных:
  o0/o4 — сырое state и обе камеры; S0, S4 — вектор признаков дерева T32
  (корни, средние узлы, листья, путь; rec["S"]) в o0 и o4; B, R; g, t;
  состояние (state 9) после первой команды закрытия (шаг g) и в конце
  чанка (t + 8) у T1 и у ArmOnly; действия ArmOnly; исходы.
Решение (ресурсное): D = W − L на 176 парах;
  signal       D ≥ 6 и по обоим сидам ≥ 0
  weak_signal  D ∈ {4, 5} и по обоим сидам ≥ 0
  offline_only W > 0 (иначе) и D ≥ 0
  negative     D < 0
  no_rescues   W = 0 и D ≥ 0
Команды: --prepare-only | --part K --device D | --summarize | --selftest.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import socket

import numpy as np

from k31_collect import ENV, EXEC_MAX, Ledger

TASK = 8
N_SCENES = 88
ARM = "ArmOnly"
PROTOCOL = "k40a-armonly-transfer-v1"
SCOPE = "k40a_transfer"
CASE_RESERVE = 2 * EXEC_MAX
N_PARTS = 2
BOOT = dict(n=50_000, seed=40001)
CODE = ("k40a_transfer.py", "k39_grip_transfer.py", "k39_grip_probe.py",
        "k39_failure_probe.py", "k39_cache.py", "k39_pulse.py")


def rank(key):
    return hashlib.sha256(f"{PROTOCOL}/{key}".encode()).hexdigest()


def decision(st):
    D, W = st["D"], st["W"]
    seeds_ok = all(v >= 0 for v in st["per_seed"].values())
    if D >= 6 and seeds_ok:
        return "signal"
    if D in (4, 5) and seeds_ok:
        return "weak_signal"
    if D < 0:
        return "negative"
    return "offline_only" if W > 0 else "no_rescues"


# --- план ---------------------------------------------------------------------------------
def prepare(a):
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    import k39_failure_probe as kfp
    import k39_grip_probe as gp
    gplan = pc.load_json(str(a.grip / "plan.json"))
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
    if len(cases) != N_SCENES:
        raise SystemExit(f"свободных сцен {len(cases)} ≠ {N_SCENES}")
    eps.sort(key=lambda e: (e["want"]["seed"], rank(e["want"]["eid"])))
    if Counter(e["want"]["case_id"] for e in eps) != {c: 2 for c in cases}:
        raise SystemExit("не у всех сцен оба сида")
    jobs = [e for e in eps if e["t"] is not None]
    for i, j in enumerate(jobs):
        j["part"] = i % N_PARTS
    cov = Counter()
    for e in eps:
        grp = f"s{e['want']['seed']}_{'success' if e['success'] else 'fail'}"
        cov[grp] += 1
        if e["t"] is not None:
            cov[f"{grp}_trigger"] += 1
    here = Path(__file__).resolve().parent
    return dict(protocol=PROTOCOL, arm=ARM, grip_plan=gplan["sha256"],
                archive_plan=plan["sha256"], v1_plan=v1p["sha256"],
                code={n: kcf.sha256(here / n) for n in CODE},
                episodes=eps, coverage=dict(cov), n_trigger=len(jobs),
                caps={SCOPE: (len(jobs) // N_PARTS + 1) * CASE_RESERVE},
                boot=BOOT), k36


# --- повтор с захватом ---------------------------------------------------------------------
def decide_rec(arm, io, desc, task, init, ep_no, chunk, obs, elapsed):
    arm.begin(task, init, ep_no)
    arm.call = int(chunk)
    batch, po, s8 = io.prep(obs, desc, task)
    a, rec = arm.decide(batch, po, s8, int(elapsed))
    return np.asarray(a, np.float32), rec


def raw_obs(obs):
    from k39_cache import CAMERAS
    return {k: np.array(obs[k][0], copy=True) for k in ("state", *CAMERAS)}


def state9(obs):
    st_ = np.asarray(obs["state"], np.float64)
    if st_.shape != (1, 9) or not np.isfinite(st_).all():
        raise SystemExit(f"state: форма {st_.shape} или нечисловые значения")
    return st_[0].copy()


def replay_capture(sim, io, arm, job, rec, z):
    """Повтор архивного T1 со сверкой; B и R с признаками дерева; кадры и
    состояния в o0, o4, после шага g и в конце чанка."""
    from k31_collect import obs_hash
    from k36_harness import episode_number
    from k39_cache import Replay
    w, t, g = job["want"], job["t"], job["g"]
    task, init, eid = w["task"], w["reset_id"], w["eid"]
    rp = Replay(sim, io, task, init)
    q = rec["summary"]
    hashes = q["init_hash"], q["init_hash_full"]
    if (rp.ih[0][0], rp.ih[1][0]) != hashes:
        raise SystemExit(f"{eid}: начальные хеши ≠ архиву")
    A = z["actions"]
    ii = np.flatnonzero(z["j_elapsed"] == t)
    if len(ii) != 1:
        raise SystemExit(f"{eid}: нет единственной архивной команды в t={t}")
    Barch = z["j_a_codec"][ii[0]]
    cap = {}
    desc = io.task_desc(task)
    for step in range(len(A) + 1):
        if step == t:
            cap["o0_hash"] = obs_hash(rp.obs)
            cap["o0"] = raw_obs(rp.obs)
            B, r0 = decide_rec(arm, io, desc, task, init, episode_number(eid),
                               t // 8, rp.obs, t)
            if not np.array_equal(B, Barch):
                raise SystemExit(f"{eid}: прогноз T32 в o0 ≠ архивному")
            cap["S0"] = np.asarray(r0["S"], np.float32)
        if step == t + 4:
            cap["o4_hash"] = obs_hash(rp.obs)
            cap["o4"] = raw_obs(rp.obs)
            R, r4 = decide_rec(arm, io, desc, task, init, episode_number(eid),
                               t // 8 + 1, rp.obs, step)
            cap["S4"] = np.asarray(r4["S"], np.float32)
        if step == g + 1:
            cap["st_g_T1"] = state9(rp.obs)
        if step == t + 8:
            cap["st_end_T1"] = state9(rp.obs)
        if step < len(A):
            rp.step(A[step])
    if (rp.reward >= 1, rp.succ_step, rp.done_step, rp.steps) != (
            q["success"], q["success_step"], q["done_step"], q["steps"]):
        raise SystemExit(f"{eid}: повтор T1 ≠ архивному исходу или terminal")
    for k in ("st_g_T1", "st_end_T1"):
        cap.setdefault(k, np.full(9, np.nan))       # эпизод кончился раньше
    return B, R, hashes, cap


class Probe:
    """Обёртка BankSim: сверка o4 и захват state после заданных шагов."""

    def __init__(self, sim, o4_step, o4_hash, marks):
        self.sim, self.o4_step, self.o4_hash = sim, o4_step, o4_hash
        self.marks = marks                         # имя -> число шагов среды
        self.got = {}

    def reset(self, task, starts):
        self.count = 0
        return self.sim.reset(task, starts)

    def step(self, actions):
        from k31_collect import obs_hash
        res = self.sim.step(actions)
        self.count += 1
        if self.count == self.o4_step and obs_hash(res[0]) != self.o4_hash:
            raise SystemExit("наблюдение o4 ветви ≠ повтору T1")
        for k, n in self.marks.items():
            if self.count == n:
                self.got[k] = state9(res[0])
        return res


def run_job(sim, io, armobj, job, rec, z):
    from k31_collect import run_episode
    from k36_harness import SUMMARY, episode_number
    import k39_grip_probe as gp
    w, t, g = job["want"], job["t"], job["g"]
    eid = w["eid"]
    B, R, hashes, cap = replay_capture(sim, io, armobj, job, rec, z)
    suffix = gp.arm_tail(B[4:], R[:4])
    W0 = ENV["waiting_steps"]
    pr = Probe(sim, W0 + t + 4, cap["o4_hash"],
               dict(st_g_A=W0 + g + 1, st_end_A=W0 + t + 8))
    armobj.episode = episode_number(eid)
    A = z["actions"]
    ep = run_episode(pr, io, armobj, w["task"], w["reset_id"], prefix=A[:t],
                     fork=dict(tau=t, hash=cap["o0_hash"],
                               a_codec=np.concatenate((B[:4], suffix))))
    if pr.count < pr.o4_step or \
            (ep["init_hash"], ep["init_hash_full"]) != hashes or \
            not np.array_equal(ep["actions"][:t + 4], A[:t + 4]):
        raise SystemExit(f"{eid}: общий префикс ≠ архиву")
    same = bool(np.array_equal(suffix, B[4:]))
    if same and any(ep[k] != rec["summary"][k] for k in
                    ("success", "success_step", "done_step", "steps",
                     "actions_sha1")):
        raise SystemExit(f"{eid}: хвост = B, эпизод ≠ T1")
    for k in ("st_g_A", "st_end_A"):
        pr.got.setdefault(k, np.full(9, np.nan))
    data = dict(B=B, R=R, tail=suffix, S0=cap["S0"], S4=cap["S4"],
                g=np.int64(g), t=np.int64(t),
                actions_A=np.asarray(ep["actions"], np.float32),
                st_g_T1=cap["st_g_T1"], st_end_T1=cap["st_end_T1"],
                st_g_A=pr.got["st_g_A"], st_end_A=pr.got["st_end_A"],
                y_T1=np.bool_(rec["summary"]["success"]),
                y_A=np.bool_(ep["success"]),
                **{f"o0_{k}": v for k, v in cap["o0"].items()},
                **{f"o4_{k}": v for k, v in cap["o4"].items()})
    res = dict(branch={k: ep[k] for k in (*SUMMARY, "seconds")},
               o0=cap["o0_hash"], o4=cap["o4_hash"], tail_same=same,
               grip=gp.grip_info(B[4:], R[:4]))
    return res, data


# --- запись и восстановление ------------------------------------------------------------------
def part_dir(out, k):
    return out / f"part-{int(k)}"


def scene_files(out, job, plan):
    w = job["want"]
    name = f"s{w['seed']}_{w['case_id']}"
    d = part_dir(out, job["part"])
    ident = dict(eid=w["eid"], plan=plan["sha256"], seed=w["seed"],
                 t1=job["success"], t=job["t"], g=job["g"], arm=plan["arm"],
                 archive_npz_sha1=job["npz_sha1"], physical=job["physical"])
    return name, d / f"{name}.json", d / f"{name}.npz", ident


def file_sha(p):
    import k37_confirm as kcf
    return kcf.sha256(p)


def save_data(path, data):
    tmp = path.with_suffix(".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **data)
    tmp.replace(path)
    return file_sha(path)


DATA_KEYS = ("B", "R", "tail", "S0", "S4", "g", "t", "actions_A", "st_g_T1",
             "st_end_T1", "st_g_A", "st_end_A", "y_T1", "y_A")


def checked_result(path, dp, ident):
    import k36_harness as kh
    import k36_protocol as pc
    row = pc.load_json(path)
    if not isinstance(row, dict) or any(
            type(row.get(k)) is not type(v) or row[k] != v
            for k, v in ident.items()):
        raise SystemExit(f"{path}: запись не соответствует плану")
    if any(not isinstance(row.get(k), str) or not row[k]
           for k in ("o0", "o4", "data_sha256")) or \
            not isinstance(row.get("tail_same"), bool):
        raise SystemExit(f"{path}: нет хешей, данных или tail_same")
    if not dp.is_file() or file_sha(dp) != row["data_sha256"]:
        raise SystemExit(f"{dp}: обучающие данные отсутствуют или изменены")
    ep = row.get("branch")
    if not isinstance(ep, dict):
        raise SystemExit(f"{path}: нет ветви")
    pr = kh.summary_problems(dict(phys=ep.get("physical")), ep,
                             ep.get("steps"))
    if pr or ep["steps"] < row["t"] + 4:
        raise SystemExit(f"{path}: неверная сводка ({pr})")
    if row["tail_same"] and ep["success"] != row["t1"]:
        raise SystemExit(f"{path}: хвост = B, исход ≠ T1")
    with np.load(dp, allow_pickle=False) as z:
        miss = [k for k in DATA_KEYS if k not in z.files]
        if miss or bool(z["y_A"]) != ep["success"] or \
                bool(z["y_T1"]) != row["t1"] or int(z["t"]) != row["t"] or \
                int(z["g"]) != row["g"]:
            raise SystemExit(f"{dp}: данные ≠ записи ({miss})")
    return row


def case_spend(row):
    return row["physical"] + row["branch"]["physical"]


def part_jobs(plan, part):
    return {f"{plan['sha256']}/{j['want']['eid']}": j
            for j in plan["episodes"] if j["t"] is not None and
            j["part"] == part}


def restore_rows(out, plan, part, ledger):
    jobs = part_jobs(plan, part)
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
        _n, path, dp, ident = scene_files(out, j, plan)
        if path.exists():
            row = checked_result(path, dp, ident)
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
        mine = part_jobs(plan, a.part)
        todo = [j for k, j in mine.items() if k not in rows]
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
                    name, path, dp, ident = scene_files(a.out, j, plan)
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
                    res, data = run_job(sim, src["io"], armobj, j, rec, z)
                    dsha = save_data(dp, data)
                    row = dict(ident, **res, data_sha256=dsha, gpu=adm)
                    pc.save_json(str(path), row)
                    if checked_result(path, dp, ident) != row:
                        raise SystemExit(f"{path}: запись изменилась")
                    ledger.commit(key, case_spend(row))
                    rows[key] = row
                    print(f"  {name}: T1 {int(row['t1'])}, {ARM} "
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


def summarize_cmd(a, plan):
    import k36_protocol as pc
    import k37_confirm as kcf
    import k39_grip_transfer as gt
    rows = {}
    for k in range(N_PARTS):
        n_k = len(part_jobs(plan, k))
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
    st = gt.stats(gt.outcomes(plan, rows), n_boot=BOOT["n"],
                  seed=BOOT["seed"])
    st["decision"] = decision(st)
    st["grip_shift"] = dict(Counter(r["grip"]["shift"] for r in rows.values()))
    pc.save_json(str(a.out / "summary.json"), dict(
        st, plan=plan["sha256"], arm=plan["arm"], rows=list(rows.values())))
    print(f"  {ARM} на {st['scenes']} сцен × 2 сида (N {st['N']}): SR T1 "
          f"{100 * st['sr_T1']:.1f}, {ARM} {100 * st['sr_arm']:.1f}; W "
          f"{st['W']} L {st['L']} D {st['D']:+d}; Δ {st['delta_pp']:+.1f} "
          f"п.п. 95% [{st['lo']:+.1f}; {st['hi']:+.1f}]; по сидам "
          f"{st['per_seed']}; триггер {100 * st['trigger_share']:.0f}% "
          f"{st['trigger_by_t1']}", flush=True)
    print(f"  K-40 A перенос: {st['decision']}", flush=True)


def selftest():
    import tempfile
    # решение
    mk = lambda D, W, s0, s1: dict(D=D, W=W, per_seed={"0": s0, "1": s1})  # noqa
    assert decision(mk(6, 8, 3, 3)) == "signal"
    assert decision(mk(7, 9, 8, -1)) == "offline_only"
    assert decision(mk(5, 6, 2, 3)) == "weak_signal"
    assert decision(mk(4, 4, 4, 0)) == "weak_signal"
    assert decision(mk(3, 5, 1, 2)) == "offline_only"
    assert decision(mk(0, 0, 0, 0)) == "no_rescues"
    assert decision(mk(-1, 3, 0, -1)) == "negative"
    # хвост ArmOnly
    import k39_grip_probe as gp
    B4 = np.zeros((4, 7), np.float32)
    B4[:, 6] = 1
    R = np.ones((8, 7), np.float32) * 0.2
    R[:, 6] = -1
    t_ = gp.arm_tail(B4, R[:4])
    assert np.array_equal(t_[:, :6], R[:4, :6]) and \
        np.array_equal(t_[:, 6], B4[:, 6])
    # Probe: сверка o4 и захват
    import k31_collect as kcol

    class S:
        def reset(self, task, starts):
            self.n = 0
            return self._o()

        def _o(self):
            st_ = np.zeros((1, 9))
            st_[0, 0] = self.n
            return dict(state=st_, agentview_image=np.zeros((1, 2, 2, 3),
                                                             np.uint8),
                        robot0_eye_in_hand_image=np.zeros((1, 2, 2, 3),
                                                          np.uint8))

        def step(self, a):
            self.n += 1
            return self._o(), np.zeros(1), np.zeros(1, bool), np.zeros(1,
                                                                       bool)
    s = S()
    s.reset(0, [0])
    want_h = kcol.obs_hash(dict(s._o(), state=np.array([[4.0] + [0] * 8])))
    p = Probe(S(), 4, want_h, dict(a=2, b=5))
    p.reset(0, [0])
    for _ in range(6):
        p.step(None)
    assert p.got["a"][0] == 2 and p.got["b"][0] == 5
    p2 = Probe(S(), 4, "чужой", {})
    p2.reset(0, [0])
    try:
        for _ in range(4):
            p2.step(None)
        raise AssertionError("чужой o4 принят")
    except SystemExit:
        pass
    # запись данных и проверка
    ident = dict(eid="e", plan="P", seed=0, t1=False, t=8, g=13,
                 arm=ARM, archive_npz_sha1="z", physical=300)
    ep = dict(success=True, success_step=50, done_step=60, own_steps=61,
              steps=61, physical=71, init_hash="a", init_hash_full="b",
              actions_sha1="c", seconds=1.0)
    data = {k: np.zeros(3) for k in DATA_KEYS}
    data.update(g=np.int64(13), t=np.int64(8), y_T1=np.bool_(False),
                y_A=np.bool_(True))
    with tempfile.TemporaryDirectory() as td:
        dp, pth = Path(td) / "r.npz", Path(td) / "r.json"
        dsha = save_data(dp, data)
        row = dict(ident, o0="h0", o4="h4", tail_same=False, grip={},
                   branch=ep, data_sha256=dsha)
        pth.write_text(json.dumps(row))
        assert checked_result(pth, dp, ident)["t1"] is False
        save_data(dp, dict(data, y_A=np.bool_(False)))
        try:
            checked_result(pth, dp, ident)
            raise AssertionError("подменённые данные приняты")
        except SystemExit:
            pass
    print("самопроверка k40a_transfer пройдена: правило решения (signal / "
          "weak / offline / negative / no_rescues); хвост ArmOnly; сверка "
          "o4 и захват состояний; данные с SHA и согласованием исходов")
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
                    default=Path("reports/k40a/transfer/v1"))
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
        print(f"  K-40 A: {N_SCENES} сцен, {len(plan['episodes'])} "
              f"эпизодов, с триггером {plan['n_trigger']}; покрытие "
              f"{dict(sorted(plan['coverage'].items()))}; SHA "
              f"{plan['sha256'][:16]}", flush=True)
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
