#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-39 B: импульс внутри чанка и один пересчёт T32 (RefreshOnce) против T1.

Команды (run-id, по умолчанию b1; план и отчёты — reports/k39/pulse/<run>,
банк и шарды — data/k39/pulse/<run>):
  bank [--workers N]          банк space 4: задачи 0/2/3/5/6/8, по 5 сцен
                              (позиция 0 — калибровка, 1–4 — пилот)
  calibrate --device D        36 коротких повторов (+12 при 0.225)
  prepare                     plan.json + plan.sha256 (неизменяем)
  gpuinfo / assign / preflight [--reference] / collect --part N / status
  merge / analyze
Окружение: OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 (проверяется).

Эпизод: до закреплённого t (24 + 8·ранг) — обычный T1. В чанке t:
B = decide(o0); P = B[:4], строки 1 и 2 по оси a: clip(B + s·amp·half[a],
L[a], U[a]); остальное (в т. ч. схват) — B; итоговый чанк масштабируется
io.scale один раз. Если 4 шага исполнены и среда жива — o4 (хеш до выбора
хвоста). T1: хвост B[4:8]; RefreshOnce: k39_run.chunk_mid(Refresh4) —
decide в o4 с elapsed t + 4, хвост R[:4]. Дальше — снова T1.
Пары (сцена, сид, уровень): одинаковые действия до хвоста и хеши o0/o4;
между уровнями — одинаковая история до первой изменяемой команды.
Анализ: по уровню W/L/D, Δ; главный Δ_strong и I = Δ_strong − Δ_zero;
bootstrap по сценам в задаче (оба сида, обе политики, все уровни вместе),
50 000, seed 39001, задачи с равным весом. Сигнал: D_strong > 0 и
D_strong > D_zero.
"""
import argparse
import hashlib
import json
import os
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PROTOCOL = "k39-pulse-v1"
TASKS = (0, 2, 3, 5, 6, 8)
SEEDS = (0, 1)
POLS = ("T1", "RefreshOnce")
LEVELS = ("zero", "mild", "strong")
AMP = dict(zero=0.0, mild=0.15, strong=0.30)
AMP_ALT = 0.225
SPACE = 4
N_PILOT, MAX_CAND = 4, 48
EP = 610
N_PARTS = 2
BOOT = dict(n=50_000, seed=39001)
TOL = dict(arm=1e-5)
OK, TECH, BUSY = 0, 3, 4
CODE = ("k39_pulse.py", "k39_run.py", "k39_cache.py", "k37_confirm.py",
        "k37_bank.py", "k38_grip_run.py", "k36_harness.py", "k36_bank.py",
        "k32_policy.py", "k32_harness.py", "k31_collect.py", "k29_policy.py",
        "k33_bank.py", "k33_protocol.py", "k18_collector.py")
EEF = slice(2, 5)                      # raw state: gripper 2, eef_pos 3, quat 4


def h256(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sha256(p):
    import k37_confirm as kcf
    return kcf.sha256(p)


def rep_dir(run):
    return os.path.join("reports/k39/pulse", run)


def data_dir(run):
    return os.path.join("data/k39/pulse", run)


def bank_dir(run):
    return os.path.join(data_dir(run), "bank")


# --- банк -------------------------------------------------------------------------------------
def bank(a):
    import k31_collect as kcol
    import k33_bank as kb
    import k33_protocol as p33
    import k37_bank as k37b
    import k39_run as kr
    mp = os.path.join(rep_dir(a.run_id), "bank_manifest.json")
    if os.path.exists(mp):
        m = json.load(open(mp, encoding="utf-8"))
        bad = [p for p, s in m["files"].items()
               if not os.path.exists(p) or sha256(p) != s]
        if bad:
            raise SystemExit(f"банк закреплён, файлы изменены: {bad}")
        print(f"  {mp}: закреплён (bank_id {m['bank_id'][:12]})")
        return OK
    for t in TASKS:
        if os.path.exists(kb.task_path(SPACE, t, bank_dir(a.run_id))):
            raise SystemExit(f"space {SPACE} t{t} уже занят — нужен разбор")
    L = kcol.Ledger(os.path.join(data_dir(a.run_id), "bank_ledger.json"),
                    dict(bank=10 * len(TASKS) * (MAX_CAND + N_PILOT + 1)))
    seen, seeds0 = k37b.k33_seen()
    prev = [(k37b.SPACE, k37b.TASKS, k37b.BANK_DIR),
            (kr.SPACE, kr.TASKS, kr.BANK_DIR)]
    for sp, tasks, d in prev:
        for t in tasks:
            z = np.load(kb.task_path(sp, t, d))
            seen.update(str(x) for x in z["sig"])
            seeds0.update(int(x) for x in z["seed"])
    files, per = {}, {}
    work = os.path.join(data_dir(a.run_id), "bank_work")
    for t in TASKS:
        official = set(kb.official_joint_sigs(t))
        res = k37b.run_chunk(L, f"bank/gen/t{t}",
                             [(SPACE, t, i) for i in range(MAX_CAND)],
                             kb._gen_job, a.workers,
                             os.path.join(work, f"gen_t{t}.pkl"))
        cases, rej = kb.select(res, N_PILOT + 1, seen, official)
        if len(cases) < N_PILOT + 1:
            raise SystemExit(f"t{t}: допустимых {len(cases)}")
        p = kb.save_task(SPACE, t, cases, cases[0]["meta"], bank_dir(a.run_id))
        files[p] = sha256(p)
        per[str(t)] = dict(cases=len(cases), rejects=rej)
    seeds = [int(x) for p_ in files for x in np.load(p_)["seed"]]
    if len(set(seeds)) != len(seeds) or set(seeds) & seeds0:
        raise SystemExit("коллизия seed")
    jobs = []
    for t in TASKS:
        cs, meta = kb.load_task(SPACE, t, bank_dir(a.run_id))
        jobs += [(SPACE, t, c, meta) for c in cs]
    res = k37b.run_chunk(L, "bank/replay", jobs, kb._replay_job, a.workers,
                         os.path.join(work, "replay.pkl"))
    by = {r["case_id"]: r for r in res}
    bad = [c["case_id"] for _s, _t, c, _m in jobs
           if not by[c["case_id"]].get("ok") or
           (by[c["case_id"]]["init_hash"], by[c["case_id"]]["init_hash_full"])
           != (c["init_hash"], c["init_hash_full"]) or
           by[c["case_id"]]["done"] or by[c["case_id"]]["success"]]
    if bad:
        raise SystemExit(f"холодный повтор: расхождения {bad[:3]}")
    man = dict(kind="k39_pulse_bank", space=SPACE, tasks=list(TASKS),
               files=files, per=per, replay=dict(n=len(jobs), bad=0),
               versions=p33.versions(),
               bank_id=h256(json.dumps(dict(files=files), sort_keys=True)))
    p33.save_json(mp, man)
    print(f"  банк space {SPACE}: {len(jobs)} сцен, повтор совпал; bank_id "
          f"{man['bank_id'][:12]}")
    return OK


def scenes(run):
    """-> (манифест, калибровочные, пилотные) с рангом, t, осью, знаком."""
    import k33_bank as kb
    mp = os.path.join(rep_dir(run), "bank_manifest.json")
    m = json.load(open(mp, encoding="utf-8"))
    if any(sha256(p) != s for p, s in m["files"].items()):
        raise SystemExit("банк изменён")
    calib, pilot = [], []
    for j, t in enumerate(TASKS):
        cs, _m = kb.load_task(SPACE, t, bank_dir(run))
        combos = sorted([(0, 1), (0, -1), (1, 1), (1, -1)],
                        key=lambda c: h256(f"k39-pulse-direction-v1|{t}|"
                                           f"{c[0]}|{c[1]}"))
        mk = lambda c, pos, r: dict(  # noqa: E731
            task=t, case_id=c["case_id"], pos=pos, rank=r,
            t=24 + 8 * r, axis=combos[r][0], sign=combos[r][1],
            hash=[c["init_hash"], c["init_hash_full"]])
        calib.append(mk(cs[0], 0, j % 4))
        pil = sorted(range(1, N_PILOT + 1), key=lambda i: h256(
            f"k39-pulse-rank-v1|{t}|{cs[i]['case_id']}"))
        for r, i in enumerate(pil):
            pilot.append(mk(cs[i], i, r))
    return m, calib, pilot


# --- импульс и эпизод ---------------------------------------------------------------------------
def perturb(B, axis, sign, amp, C):
    """-> (P [4,7], дельта по строкам 1–2, клип по строкам 1–2)."""
    P = np.array(B[:4], np.float32, copy=True)
    if amp == 0.0:
        return P, np.zeros(2, np.float32), np.zeros(2, bool)
    d = np.float32(sign * amp * float(C["half"][axis]))
    want = np.zeros(2, np.float32)
    clip = np.zeros(2, bool)
    for i, j in enumerate((1, 2)):
        raw = np.float32(B[j, axis] + d)
        v = np.float32(np.clip(raw, C["L"][axis], C["U"][axis]))
        P[j, axis] = v
        want[i] = v - B[j, axis]
        clip[i] = v != raw
    return P, want, clip


def ep_number(run, task, cid, seed):
    import k36_harness as kh
    return kh.episode_number(f"k39-pulse|{run}|{task}|{cid}|{seed}")


def eef_of(obs):
    st_ = np.asarray(obs["state"], np.float64)
    if st_.shape != (1, 9) or not np.isfinite(st_).all():
        raise SystemExit(f"state: форма {st_.shape} или нечисловые значения")
    return st_[0, EEF].copy()


def run_pulse(sim, io, arm, sc, seed, amp, policy, C, run, stop_after=False,
              keep_o4=False, trace=False):
    """Эпизод с импульсом в чанке sc["t"]; stop_after — оборвать сразу после
    префикса целевого чанка (калибровка)."""
    import k18_collector as k18c
    import k31_collect as kcol
    import k39_run as kr
    t0 = time.time()
    task = sc["task"]
    io.seed(kcol.ENV["seed"])
    obs = sim.reset(task, [sc["pos"]])
    desc = io.task_desc(task)
    arm.begin(task, sc["pos"], ep_number(run, task, sc["case_id"], seed))
    reward, succ_step, done_step, steps = 0.0, -1, -1, 0
    dummy = np.array([kcol.DUMMY])
    wd = False
    for _ in range(kcol.ENV["waiting_steps"]):
        obs, r_, d_, _s = sim.step(dummy)
        reward = min(max(reward + float(r_[0]), 0.0), 1.0)
        wd |= bool(d_[0])
    if wd or reward >= 1.0:
        raise SystemExit("среда завершилась во время ожидания")
    ih, ihf = k18c.init_hashes(obs, 1)
    acts, calls, heavy = [], [], dict(n=0)
    eefs = [] if trace else None          # кисть после каждого шага
    tgt = dict(reached=False, k=0, mid=False, refresh=False, o0=None,
               o4=None, eef4=None, P=None, delta=None, clip=None, R=None,
               B=None, o4_obs=None)
    fwd0 = arm.forward

    def counted(*x, **kw):
        heavy["n"] += 1
        return fwd0(*x, **kw)
    arm.forward = counted

    def step(a):
        nonlocal obs, reward, succ_step, done_step, steps
        acts.append(np.asarray(a, np.float32).copy())
        obs, r_, d_, _s = sim.step(np.asarray(a, np.float32)[None])
        prev = reward
        reward = min(max(reward + float(r_[0]), 0.0), 1.0)
        if reward >= 1.0 and prev < 1.0 and succ_step < 0:
            succ_step = steps
        if bool(d_[0]) and done_step < 0:
            done_step = steps
        steps += 1
        if eefs is not None:
            eefs.append(eef_of(obs))
    live = lambda: done_step < 0 and steps < kcol.ENV["max_steps"]  # noqa
    try:
        while live():
            t = steps
            call = t // 8 + 1
            arm.call = call - 1
            batch, po, s8 = io.prep(obs, desc, task)
            B, rec = arm.decide(batch, po, s8, t)
            B = np.asarray(B, np.float32)
            final = B.copy()
            target = t == sc["t"]
            if target:
                P, dl, cl = perturb(B, sc["axis"], sc["sign"], amp, C)
                final[:4] = P
                tgt.update(reached=True, o0=kcol.obs_hash(obs), P=P,
                           delta=dl, clip=cl, B=B.copy())
            ex = io.scale(final[None])[0]
            k = 0
            for j in range(4):
                if not live():
                    break
                step(ex[j])
                k += 1
            mid = k == 4 and live()
            R = None
            if target:
                tgt.update(k=k, mid=bool(mid))
                if mid:
                    st_ = np.asarray(obs["state"], np.float64)
                    if st_.shape != (1, 9) or not np.isfinite(st_).all():
                        raise SystemExit(f"state o4: форма {st_.shape} или "
                                         f"нечисловые значения")
                    tgt.update(o4=kcol.obs_hash(obs), eef4=st_[0, EEF].copy())
                    if keep_o4:
                        tgt["o4_obs"] = {kk: np.array(v, copy=True)
                                         for kk, v in obs.items()}
                if stop_after:
                    arm.executed(k)
                    calls.append(dict(k=k, a_codec=B, rec=rec, final=final,
                                      target=True, mid=bool(mid), R=None))
                    break
                if mid and policy == "RefreshOnce":
                    tail, _c, _s, R = kr.chunk_mid(None, "Refresh4", None, B,
                                                   obs, io, arm, desc, task,
                                                   t)
                    final[4:8] = tail
                    tgt.update(refresh=True, R=R)
                ex = io.scale(final[None])[0]
            if mid:
                for j in range(4, 8):
                    if not live():
                        break
                    step(ex[j])
                    k += 1
            arm.executed(k)
            calls.append(dict(k=k, a_codec=B, rec=rec, final=final,
                              target=bool(target), mid=bool(mid), R=R))
    finally:
        arm.forward = fwd0
    A = np.stack(acts) if acts else np.zeros((0, 7), np.float32)
    own = done_step + 1 if done_step >= 0 else steps
    return dict(success=bool(reward >= 1.0), success_step=int(succ_step),
                done_step=int(done_step), own_steps=int(own),
                steps=int(steps), physical=int(10 + steps), init_hash=ih[0],
                init_hash_full=ihf[0], actions=A,
                actions_sha1=kcol.actions_sha(A), calls=calls,
                heavy=int(heavy["n"]), target=tgt,
                eef_trace=None if eefs is None else
                (np.stack(eefs) if eefs else np.zeros((0, 3))),
                seconds=time.time() - t0)


SUMMARY = ("success", "success_step", "done_step", "own_steps", "steps",
           "init_hash", "init_hash_full", "actions_sha1")


def save_ep(path, job, ep, plan_sha, extra):
    import k32_harness as kh32
    n = len(ep["calls"])
    calls = [dict(c, rec=dict(c["rec"], a_codec=c["a_codec"]))
             for c in ep["calls"]]
    J = kh32.pack_journal([(0, calls)], 6)
    tg = ep["target"]
    K = dict(final=np.stack([c["final"] for c in ep["calls"]]) if n else
             np.zeros((0, 8, 7), np.float32),
             target=np.array([c["target"] for c in ep["calls"]], bool),
             mid=np.array([c["mid"] for c in ep["calls"]], bool),
             R=tg["R"] if tg["R"] is not None else np.zeros((8, 7),
                                                            np.float32),
             P=tg["P"] if tg["P"] is not None else np.zeros((4, 7),
                                                            np.float32),
             delta=tg["delta"] if tg["delta"] is not None else
             np.zeros(2, np.float32),
             clip=tg["clip"] if tg["clip"] is not None else np.zeros(2, bool),
             eef4=tg["eef4"] if tg["eef4"] is not None else np.zeros(3))
    meta = dict(job=job, plan=plan_sha, phys=int(ep["physical"]),
                heavy=int(ep["heavy"]), seconds=float(ep["seconds"]),
                summary={k: ep[k] for k in SUMMARY},
                target={k: (tg[k] if k in ("reached", "k", "mid", "refresh",
                                           "o0", "o4") else None)
                        for k in ("reached", "k", "mid", "refresh", "o0",
                                  "o4")}, **extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as fh:
        np.savez_compressed(fh, meta=np.array(json.dumps(meta)),
                            actions=ep["actions"].astype(np.float32),
                            **{f"j_{k}": v for k, v in J.items()},
                            **{f"k_{k}": v for k, v in K.items()})
    os.replace(path + ".tmp", path)
    return meta


def ep_problems(z, meta, job, C):
    """Один эпизод: номинальный журнал K-32, подстановка и хвост."""
    import k29_policy as kp
    import k31_collect as kcol
    import k32_harness as kh32
    p = []
    A = z["actions"]
    s = meta["summary"]
    if kcol.actions_sha(A) != s["actions_sha1"]:
        return ["действия ≠ SHA"]
    if job["scene"].get("hash") is not None and \
            (s["init_hash"], s["init_hash_full"]) != tuple(job["scene"]["hash"]):
        p.append("начальный хеш ≠ банку")
    J = {k[2:]: v for k, v in z.items() if k.startswith("j_")}
    K = {k[2:]: v for k, v in z.items() if k.startswith("k_")}
    n = len(J.get("ep", []))
    if n == 0:
        return p + ["журнал пуст"]
    q = C["q"]
    A_nom = np.concatenate([kp.scale_exec(J["a_codec"][i][None], q)[0]
                            [:int(J["k"][i])] for i in range(n)])
    p += [f"база: {x}" for x in kh32.journal_problems(
        J, [A_nom], [s["done_step"]], C["half"], C["mid"], C["L"], C["U"],
        q, "T")[:3]]
    tstar, amp = job["scene"]["t"], job["amp"]
    tg_idx = [i for i in range(n) if int(J["elapsed"][i]) == tstar]
    if [i for i in range(n) if K["target"][i]] != tg_idx:
        p.append("целевой чанк ≠ t")
    tg = meta["target"]
    ti = [i for i in range(n) if K["target"][i]]
    if bool(tg["reached"]) != bool(ti) or len(ti) > 1:
        p.append("target.reached ≠ журналу")
    elif ti:
        kk = int(J["k"][ti[0]])
        if tg["k"] != min(kk, 4) or bool(tg["mid"]) != (kk > 4) or \
                not tg["o0"]:
            p.append("target k/mid/o0 ≠ журналу")
        if bool(tg["o4"]) != bool(tg["mid"]):
            p.append("хеш o4 не согласован с mid")
    elif tg["o0"] or tg["o4"] or tg["mid"] or tg["k"]:
        p.append("target без вмешательства не пуст")
    refresh = False
    for i in range(n):
        B, F = J["a_codec"][i], K["final"][i]
        if not K["target"][i]:
            if not np.array_equal(F, B):
                p.append(f"вызов {i}: нецелевой чанк ≠ B")
            continue
        P, dl, cl = perturb(B, job["scene"]["axis"], job["scene"]["sign"],
                            amp, C)
        if not (np.array_equal(F[:4], P) and np.array_equal(K["P"], P) and
                np.array_equal(K["delta"], dl) and
                np.array_equal(K["clip"], cl)):
            p.append("префикс ≠ пересборке импульса")
        mid = int(J["k"][i]) > 4
        if bool(K["mid"][i]) != mid:
            p.append("mid ≠ (k > 4)")
        if job["policy"] == "RefreshOnce" and mid:
            refresh = True
            if not np.array_equal(F[4:8], K["R"][:4]):
                p.append("хвост ≠ R[:4]")
        elif not np.array_equal(F[4:8], B[4:8]):
            p.append("хвост T1 ≠ B[4:8]")
    ex = np.concatenate([kp.scale_exec(K["final"][i][None], q)[0]
                         [:int(J["k"][i])] for i in range(n)])
    if not np.array_equal(ex.astype(np.float32), np.asarray(A, np.float32)):
        p.append("исполненное ≠ io.scale(итоговых чанков)")
    if meta["heavy"] != n + int(refresh) or meta["target"]["refresh"] != \
            refresh:
        p.append(f"тяжёлых вызовов {meta['heavy']} ≠ {n + int(refresh)}")
    ss, ds, st_ = s["success_step"], s["done_step"], s["steps"]
    if meta["phys"] != 10 + len(A) or st_ != len(A) or meta["phys"] > EP:
        p.append("расход ≠ шагам")
    if s["success"]:
        if not (0 <= ss < st_ and (ds < 0 or ss <= ds)):
            p.append("success_step несогласован")
    elif ss != -1:
        p.append("success_step при неуспехе")
    if ds >= 0:
        if ds + 1 != st_ or s["own_steps"] != st_:
            p.append("done_step/steps несогласованы")
    elif st_ != 600:
        p.append("без terminal эпизод не дошёл до лимита")
    return p


def pair_problems(a, b):
    """a, b — (meta, z) двух политик одного (сцена, сид, уровень)."""
    p = []
    ma, za = a
    mb, zb = b
    sa, sb = ma["summary"], mb["summary"]
    if (sa["init_hash"], sa["init_hash_full"]) != (sb["init_hash"],
                                                   sb["init_hash_full"]):
        p.append("post-wait хеши пары различны")
    ta, tb = ma["target"], mb["target"]
    if not ta["reached"]:
        if not (np.array_equal(za["actions"], zb["actions"]) and sa == sb):
            p.append("пара без вмешательства: эпизоды различны")
        return p
    if (ta["o0"], ta["o4"], ta["mid"]) != (tb["o0"], tb["o4"], tb["mid"]):
        p.append("хеши o0/o4 пары различны")
    if not ta["o0"] or not tb["o0"] or (ta["mid"] and not (ta["o4"] and
                                                            tb["o4"])):
        p.append("нет хеша o0/o4 в достигнутой точке")
    t = int(np.flatnonzero(za["k_target"])[0])
    cut = int(za["j_elapsed"][t]) + (4 if ta["mid"] else int(za["j_k"][t]))
    if not np.array_equal(za["actions"][:cut], zb["actions"][:cut]):
        p.append("физический префикс пары различен")
    if not np.array_equal(za["j_a_codec"][t], zb["j_a_codec"][t]):
        p.append("номинальный B пары различен")
    if not ta["mid"] and not (np.array_equal(za["actions"], zb["actions"])
                              and sa == sb):
        p.append("без o4 эпизоды пары различны")
    return p


# --- план -----------------------------------------------------------------------------------------
def queue(plan, part):
    out = []
    for s in SEEDS:
        for sc in sorted((x for x in plan["pilot"] if x["rank"] % N_PARTS ==
                          part), key=lambda x: (x["rank"], x["task"])):
            k = (sc["rank"] + s) % 3
            lv = LEVELS[k:] + LEVELS[:k]
            for lev in lv:
                li = LEVELS.index(lev)
                pol = POLS if (sc["rank"] + s + li) % 2 == 0 else POLS[::-1]
                for pl in pol:
                    out.append(dict(
                        job=f"ep/{lev}/{pl}/s{s}/t{sc['task']}/{sc['case_id']}",
                        level=lev, amp=plan["amp"][lev], policy=pl, seed=s,
                        scene=sc))
    return out


def plan_path(run):
    return os.path.join(rep_dir(run), "plan.json")


def prepare(a):
    import k36_protocol as pc
    import k37_confirm as kcf
    import k39_cache as kc
    run = a.run_id
    calp = os.path.join(rep_dir(run), "calibration.json")
    if not os.path.exists(calp):
        raise SystemExit("нет калибровки")
    cal = pc.load_json(calp)
    if cal.get("status") != "ok":
        raise SystemExit(f"калибровка: {cal.get('status')}")
    m, calib, pilot = scenes(run)
    k36 = pc.load_prepare()
    plan = dict(kind="k39_pulse_plan", run_id=run, protocol=PROTOCOL,
                code={f: sha256(os.path.join(HERE, f)) for f in CODE},
                k36_prepare_sha1=k36["sha1"],
                k32={str(s): k36["k32"]["seeds"][str(s)]["T"]["actor_hash"]
                     for s in SEEDS},
                bank=dict(bank_id=m["bank_id"], files=m["files"],
                          space=SPACE, dir=bank_dir(run)),
                calibration_sha256=sha256(calp),
                ref_data_plan_sha256=kc.load_data_plan("f2")["sha256"],
                amp=dict(AMP, strong=cal["strong"]),
                pilot=pilot, calib=calib, levels=list(LEVELS),
                policies=list(POLS), seeds=list(SEEDS),
                env=dict(B=1, exec_seed=101, waiting_steps=10, max_steps=600,
                         prefix=4, horizon=8),
                parts={str(k): dict(n=0, cap=0) for k in range(N_PARTS)},
                analysis=dict(BOOT, rule="D_strong > 0 and D_strong > D_zero"))
    for k in range(N_PARTS):
        n = len(queue(plan, k))
        plan["parts"][str(k)] = dict(n=n, cap=(n + 1) * EP)
    sh = kcf.canon_sha(plan)
    p = plan_path(run)
    shp = p.replace(".json", ".sha256")
    if os.path.exists(p) or os.path.exists(shp):
        old = json.load(open(p, encoding="utf-8"))
        if not (kcf.canon_sha(old) == open(shp).read().strip() == sh):
            raise SystemExit(f"{p} закреплён иначе — нужен новый run-id")
        print(f"  план закреплён: {sh[:16]}")
        return OK
    pc.save_json(p, plan)
    with open(shp, "w") as fh:
        fh.write(sh + "\n")
    print(f"  plan {p}: sha256 {sh[:16]}; сцен {len(pilot)}, эпизодов "
          f"{sum(v['n'] for v in plan['parts'].values())}; strong "
          f"{plan['amp']['strong']}")
    return OK


def load_plan(run, verify=True):
    import k36_protocol as pc
    import k37_confirm as kcf
    p = json.load(open(plan_path(run), encoding="utf-8"))
    sh = kcf.canon_sha(p)
    if sh != open(plan_path(run).replace(".json", ".sha256")).read().strip():
        raise SystemExit("plan ≠ sha256")
    if verify:
        bad = [f for f, s in p["code"].items()
               if sha256(os.path.join(HERE, f)) != s]
        if bad:
            raise SystemExit(f"код ≠ плану: {bad}")
        if pc.load_prepare()["sha1"] != p["k36_prepare_sha1"]:
            raise SystemExit("prepare K-36 ≠ плану")
        for f, s in p["bank"]["files"].items():
            if sha256(f) != s:
                raise SystemExit(f"банк {f} ≠ плану")
        if sha256(os.path.join(rep_dir(run), "calibration.json")) != \
                p["calibration_sha256"]:
            raise SystemExit("калибровка ≠ плану")
    p["sha256"] = sh
    return p


# --- общая обвязка карты -----------------------------------------------------------------------------
class Card:
    """Потоки, ресурсы, блокировка UUID, привязка устройства, среда."""

    def __init__(self, device, run):
        import torch
        import k38_grip_run as kgr
        import k37_confirm as kcf
        tp = kgr.threads_problems(torch)
        if tp:
            raise SystemExit("потоки: " + "; ".join(tp))
        self.torch, self.kcf = torch, kcf
        self.idx = kcf.dev_index(device)
        self.g = kcf.gpu_query(self.idx)
        self.lock = kcf.Lock(os.path.join(kcf.lock_dir(), self.g["uuid"] +
                                          ".lock"), f"карта {self.g['uuid']}",
                             kcf.Busy)
        self.device = device

    def start(self, run):
        import k36_bank as kb36
        self.kcf.check_gpu(self.idx, self.kcf.RES["min_free_gb"])
        self.kcf.check_disk(rep_dir(run))
        self.sim = kb36.BankSim("bank", SPACE, bank_dir=bank_dir(run))
        self.kcf.bind_device(self.torch, self.idx, self.g)
        self.torch.cuda.reset_peak_memory_stats(self.idx)

    def close(self):
        if getattr(self, "sim", None) is not None:
            self.kcf.close_sim(self.sim)
        self.lock.release()


def source(card, k36, s):
    import k36_harness as kh
    card.kcf.check_gpu(card.idx, card.kcf.RES["min_free_gb"])
    v = kh.load_source(card.torch, s, card.device)
    return v, kh.make_arm(card.torch, k36, f"T32_s{s}", v)


# --- калибровка ------------------------------------------------------------------------------------------
def cal_row_path(run, job):
    return os.path.join(rep_dir(run), "calib", job.replace("/", "__") +
                        ".json")


def calibrate(a):
    """0 / 0.15 / 0.30 на обоих сидах; при клипе > 0.5 суммарно на 0.30 —
    0.225 тоже на обоих. Строка повтора пишется атомарно до commit;
    повторный запуск читает готовые строки."""
    import k36_harness as kh
    import k36_protocol as pc
    import k31_collect as kcol
    run = a.run_id
    _m, calib, _p = scenes(run)
    k36 = pc.load_prepare()
    C = kh.consts(k36)
    L = kcol.Ledger(os.path.join(rep_dir(run), "calib_ledger.json"),
                    dict(calib=48 * EP))
    pd = L.d["pending"]
    if pd and os.path.exists(cal_row_path(run, pd["id"])):
        r = pc.load_json(cal_row_path(run, pd["id"]))
        pr = cal_row_problems(r, pd["id"])
        if pd["scope"] != "calib" or pd["amount"] != EP or \
                not pd["id"].startswith("cal/"):
            pr.append(f"резерв {pd} не калибровочный")
        if pr:
            raise SystemExit(f"резерв {pd['id']}: {pr}")
        L.commit(pd["id"], r["physical"])
    card = None

    def stage(levels):
        nonlocal card
        rows = []
        for s in SEEDS:
            todo = [(sc, lev) for sc in calib for lev in levels]
            jobs = {(sc["task"], lev): f"cal/{lev}/s{s}/t{sc['task']}/"
                    f"{sc['case_id']}" for sc, lev in todo}
            need = [x for x in todo if not os.path.exists(
                cal_row_path(run, jobs[(x[0]["task"], x[1])]))]
            v = arm = None
            if need:
                if card is None:
                    card = Card(a.device, run)
                    card.start(run)
                v, arm = source(card, k36, s)
            for sc, lev in todo:
                job = jobs[(sc["task"], lev)]
                rp = cal_row_path(run, job)
                if os.path.exists(rp):
                    r = pc.load_json(rp)
                    pr = cal_row_problems(r, job)
                    if L.d["done"].get(job) != dict(scope="calib",
                                                    actual=r.get("physical")):
                        pr.append(f"журнал {L.d['done'].get(job)} ≠ строке")
                    if pr:
                        raise SystemExit(f"{job}: {pr}")
                    rows.append(r)
                    continue
                if L.is_done(job):
                    raise SystemExit(f"{job}: учтён, строки нет")
                amp = AMP_ALT if lev == "alt" else AMP[lev]
                L.reserve("calib", job, amount=EP)
                ep = run_pulse(card.sim, v["io"], arm, sc, s, amp, "T1", C,
                               run, stop_after=True)
                tg = ep["target"]
                r = dict(job=job, task=sc["task"], seed=s, level=lev, amp=amp,
                         reached=tg["reached"], k=tg["k"], mid=tg["mid"],
                         delta=None if tg["delta"] is None else
                         (tg["delta"] / C["half"][sc["axis"]]).tolist(),
                         clip=None if tg["clip"] is None else
                         tg["clip"].tolist(),
                         eef4=None if tg["eef4"] is None else
                         tg["eef4"].tolist(), axis=sc["axis"],
                         physical=int(ep["physical"]))
                pc.save_json(rp, r)               # атомарно, до commit
                L.commit(job, r["physical"])
                rows.append(r)
            if arm is not None:
                arm = v = None
                kh.free(card.torch)
        return rows
    try:
        rows = stage(LEVELS)
        if frac_clip([r for r in rows if r["level"] == "strong"]) > 0.5:
            rows += stage(("alt",))
    finally:
        if card is not None:
            card.close()
    rep = summarize_cal(rows)
    pc.save_json(os.path.join(rep_dir(run), "calibration.json"), rep)
    for k_, x in rep["levels"].items():
        if k_ == "zero":
            print(f"  zero: o4 {x['o4_share']:.2f} (из {x['n']})")
            continue
        print(f"  {k_}: o4 {x['o4_share']:.2f}, клип {x['clip_share']:.2f}, "
              f"|Δ|/half {x['delta_mean']:.3f} (ключей {x['n_cmp']}), "
              f"смещение кисти медиана {x['eef_med']:.4f} м, p90 "
              f"{x['eef_p90']:.4f} м")
    print(f"  калибровка: {rep['status']}; strong = {rep['strong']} "
          f"({rep['reason']})")
    return OK if rep["status"] == "ok" else TECH


def cal_row_problems(r, job):
    p = []
    if r.get("job") != job:
        p.append("строка чужого задания")
    ph = r.get("physical")
    if type(ph) is not int or not 10 < ph <= EP:
        p.append(f"расход {ph!r} вне (10, {EP}]")
    return p


PROBE_OFFSETS = (4, 8, 16, 24)


def probe_job(amp, s, sc):
    return f"probe/a{amp:.3f}/s{s}/t{sc['task']}/{sc['case_id']}"


def probe(a):
    """T1 до конца на 12 калибровочных стартах при амплитудах --amps (0
    добавляется всегда): исходы пар и смещение кисти к нулю на t+4/8/16/24.
    Не входит в план; строки — reports/k39/pulse/<run>/probe/."""
    import k36_harness as kh
    import k36_protocol as pc
    import k31_collect as kcol
    run = a.run_id
    amps = sorted({0.0} | {float(x) for x in a.amps.split(",")})
    if len(amps) > 6 or max(amps) > 1.0 or min(amps) < 0:
        raise SystemExit(f"амплитуды {amps}")
    _m, calib, _p = scenes(run)
    k36 = pc.load_prepare()
    C = kh.consts(k36)
    L = kcol.Ledger(os.path.join(rep_dir(run), "probe_ledger.json"),
                    dict(probe=12 * 6 * EP))
    pd = L.d["pending"]
    if pd and os.path.exists(cal_row_path(run, pd["id"])):
        r = pc.load_json(cal_row_path(run, pd["id"]))
        pr = cal_row_problems(r, pd["id"])
        if pd["scope"] != "probe" or pd["amount"] != EP or \
                not pd["id"].startswith("probe/"):
            pr.append(f"резерв {pd} не пробный")
        if pr:
            raise SystemExit(f"резерв {pd['id']}: {pr}")
        L.commit(pd["id"], r["physical"])
    card = None
    rows = []
    try:
        for s in SEEDS:
            todo = [(sc, amp) for sc in calib for amp in amps]
            need = [x for x in todo if not os.path.exists(
                cal_row_path(run, probe_job(x[1], s, x[0])))]
            v = arm = None
            if need:
                if card is None:
                    card = Card(a.device, run)
                    card.start(run)
                v, arm = source(card, k36, s)
            for sc, amp in todo:
                job = probe_job(amp, s, sc)
                rp = cal_row_path(run, job)
                if os.path.exists(rp):
                    r = pc.load_json(rp)
                    pr = cal_row_problems(r, job)
                    want = dict(task=sc["task"], case_id=sc["case_id"],
                                seed=s, amp=amp, t=sc["t"], axis=sc["axis"],
                                init=list(sc["hash"]))
                    if {k_: r.get(k_) for k_ in want} != want:
                        pr.append("параметры строки ≠ заданию")
                    if L.d["done"].get(job) != dict(scope="probe",
                                                    actual=r.get("physical")):
                        pr.append(f"журнал {L.d['done'].get(job)} ≠ строке")
                    if pr:
                        raise SystemExit(f"{job}: {pr}")
                    rows.append(r)
                    continue
                if L.is_done(job):
                    raise SystemExit(f"{job}: учтён, строки нет")
                L.reserve("probe", job, amount=EP)
                ep = run_pulse(card.sim, v["io"], arm, sc, s, amp, "T1", C,
                               run, trace=True)
                tg, tr = ep["target"], ep["eef_trace"]
                if [ep["init_hash"], ep["init_hash_full"]] != list(sc["hash"]):
                    print(f"  {job}: начальный хеш ≠ банку — резерв открыт",
                          flush=True)
                    return TECH
                A = ep["actions"]
                r = dict(job=job, task=sc["task"], case_id=sc["case_id"],
                         seed=s, amp=amp, t=sc["t"], axis=sc["axis"],
                         init=[ep["init_hash"], ep["init_hash_full"]],
                         o0=tg["o0"], actions_sha1=ep["actions_sha1"],
                         # история до первой изменяемой команды (строка 1)
                         prefix_sha1=kcol.actions_sha(A[:sc["t"] + 1])
                         if tg["reached"] else ep["actions_sha1"],
                         success=ep["success"], steps=ep["steps"],
                         done_step=ep["done_step"], reached=tg["reached"],
                         k=tg["k"], mid=tg["mid"],
                         clip=None if tg["clip"] is None else
                         tg["clip"].tolist(),
                         eef={str(d): (tr[sc["t"] + d - 1].tolist()
                                       if sc["t"] + d - 1 < len(tr) else None)
                              for d in PROBE_OFFSETS},
                         physical=int(ep["physical"]))
                pc.save_json(rp, r)
                L.commit(job, r["physical"])
                rows.append(r)
                print(f"  {job}: успех {r['success']}, шагов {r['steps']}",
                      flush=True)
            if arm is not None:
                arm = v = None
                kh.free(card.torch)
    finally:
        if card is not None:
            card.close()
    rep = summarize_probe(rows)
    if rep["problems"]:
        raise SystemExit("probe: " + "; ".join(rep["problems"][:4]))
    pc.save_json(os.path.join(rep_dir(run), "probe.json"), rep)
    print(f"  T1 без импульса: успех {rep['sr_zero']}/{rep['n']}")
    for k_, x in rep["levels"].items():
        dv = ", ".join(f"+{d}: {x['eef'][d]['med']:.4f}/"
                       f"{x['eef'][d]['p90']:.4f} (n {x['eef'][d]['n']})"
                       if x["eef"][d]["n"] else f"+{d}: —"
                       for d in x["eef"])
        print(f"  амплитуда {k_}: успех {x['sr']}/{x['n']}; сломал "
              f"{x['broke']} {x['broke_keys']}; починил {x['fixed']} "
              f"{x['fixed_keys']}; o4 {x['o4']}/{x['n']}, клип "
              f"{x['clip_share']:.2f}; смещение кисти к нулю, м (медиана/p90)"
              f": {dv}")
    return OK


def summarize_probe(rows):
    by = {(r["task"], r["seed"], r["amp"]): r for r in rows}
    keys = sorted({(r["task"], r["seed"]) for r in rows})
    zero = {k: by[k + (0.0,)] for k in keys if k + (0.0,) in by}
    prob = []
    if len(by) != len(rows):
        prob.append("дубли строк")
    for amp in sorted({r["amp"] for r in rows}):
        n_ = sum(k + (amp,) in by and k in zero for k in keys)
        if n_ != len(TASKS) * len(SEEDS):
            prob.append(f"амплитуда {amp:g}: пар {n_} ≠ "
                        f"{len(TASKS) * len(SEEDS)}")
    for (t, s_, amp), r in by.items():
        z = zero.get((t, s_))
        if z is None or amp == 0.0:
            continue
        if (r.get("init"), r.get("case_id"), r.get("t"), r.get("axis")) != \
                (z.get("init"), z.get("case_id"), z.get("t"), z.get("axis")):
            prob.append(f"t{t} s{s_} {amp:g}: старт ≠ нулевому")
        elif r["reached"] != z["reached"]:
            prob.append(f"t{t} s{s_} {amp:g}: достижение t ≠ нулевому")
        elif r["reached"] and (not r.get("o0") or r["o0"] != z.get("o0") or
                               r.get("prefix_sha1") != z.get("prefix_sha1")):
            prob.append(f"t{t} s{s_} {amp:g}: o0/префикс ≠ нулевому")
        elif not r["reached"] and (r.get("actions_sha1") !=
                                   z.get("actions_sha1") or
                                   r["success"] != z["success"]):
            prob.append(f"t{t} s{s_} {amp:g}: без импульса эпизод ≠ нулевому")
    lv = {}
    for amp in sorted({r["amp"] for r in rows} - {0.0}):
        rs = [by[k + (amp,)] for k in keys if k + (amp,) in by and k in zero]
        brk = [[r["task"], r["seed"]] for r in rs
               if zero[(r["task"], r["seed"])]["success"] and not r["success"]]
        fix = [[r["task"], r["seed"]] for r in rs
               if not zero[(r["task"], r["seed"])]["success"] and r["success"]]
        ev = {}
        for d in PROBE_OFFSETS:
            dd = [float(np.linalg.norm(np.subtract(
                r["eef"][str(d)], zero[(r["task"], r["seed"])]["eef"][str(d)])))
                for r in rs if r["eef"][str(d)] is not None and
                zero[(r["task"], r["seed"])]["eef"][str(d)] is not None]
            ev[str(d)] = dict(n=len(dd),
                              med=float(np.median(dd)) if dd else None,
                              p90=float(np.quantile(dd, 0.9)) if dd else None)
        lv[f"{amp:g}"] = dict(n=len(rs), sr=sum(r["success"] for r in rs),
                              broke=len(brk), broke_keys=brk, fixed=len(fix),
                              fixed_keys=fix,
                              o4=sum(bool(r["mid"]) for r in rs),
                              clip_share=frac_clip(rs), eef=ev)
    return dict(n=len(zero), sr_zero=sum(r["success"] for r in zero.values()),
                levels=lv, problems=prob, rows=rows)


def executed_deltas(r):
    """|Δ|/half только по исполненным строкам 1 и 2."""
    if r["delta"] is None:
        return []
    return [abs(x) for e, x in zip((r["k"] > 1, r["k"] > 2), r["delta"])
            if e]


def frac_clip(rows):
    n = c = 0
    for r in rows:
        if r["clip"] is None:
            continue
        ex = [r["k"] > 1, r["k"] > 2]           # исполнены ли строки 1 и 2
        for e, cl in zip(ex, r["clip"]):
            if e:
                n += 1
                c += int(cl)
    return c / n if n else 0.0


def summarize_cal(rows):
    by = {(r["task"], r["seed"], r["level"]): r for r in rows}
    keys = sorted({(r["task"], r["seed"]) for r in rows})
    have = sorted({r["level"] for r in rows} - {"zero"})
    st_raw = [r for r in rows if r["level"] == "strong"]
    chosen = "alt" if frac_clip(st_raw) > 0.5 else "strong"
    # сопоставимые ключи: o4 достигнут на нуле, мягком и выбранном уровнях
    cmp_ = [k for k in keys if all(by.get(k + (lv,), {}).get("mid")
                                   for lv in ("zero", "mild", chosen))]
    lv = {}
    for lev in ["zero"] + have:
        rs = [r for r in rows if r["level"] == lev]
        # метрики уровня — по стартам, где o4 есть и на нём, и на нуле
        own = [by[k + (lev,)] for k in keys
               if by.get(k + (lev,), {}).get("mid") and
               by.get(k + ("zero",), {}).get("mid")]
        disp = [float(np.linalg.norm(np.subtract(r["eef4"], by[(
            r["task"], r["seed"], "zero")]["eef4"]))) for r in own]
        # |Δ| для сравнения 1.1× — на общих ключах (нуль, мягкий, выбранный)
        rc = [by[k + (lev,)] for k in cmp_ if k + (lev,) in by]
        dl = [x for r in rc for x in executed_deltas(r)]
        lv[lev] = dict(n=len(rs), n_cmp=len(rc), n_o4=len(own),
                       o4_share=float(np.mean([r["mid"] for r in rs])),
                       clip_share=frac_clip(rs),
                       delta_mean=float(np.mean(dl)) if dl else 0.0,
                       eef_med=float(np.median(disp)) if disp else 0.0,
                       eef_p90=float(np.quantile(disp, 0.9)) if disp else 0.0)
    status = "ok"
    strong = AMP_ALT if chosen == "alt" else AMP["strong"]
    reason = ("клип 0.30 > 0.5 по обоим сидам, взят 0.225"
              if chosen == "alt" else "клип 0.30 ≤ 0.5")
    use = lv.get(chosen)
    if use is None:
        status, reason = "needs_review", "нет повторов выбранного уровня"
    elif use["o4_share"] < 0.5:
        status, reason = "needs_review", f"{chosen}: большинство без o4"
    elif use["clip_share"] > 0.5:
        status, reason = "needs_review", f"{chosen}: клип > 0.5"
    elif not cmp_:
        status, reason = "needs_review", "нет сопоставимых ключей"
    elif use["delta_mean"] <= 1.1 * lv["mild"]["delta_mean"]:
        status, reason = "needs_review", "сильный ≈ мягкий по факту"
    return dict(status=status, strong=strong, chosen=chosen, reason=reason,
                levels=lv, comparable=[list(k) for k in cmp_], rows=rows)


# --- допуск и сбор -------------------------------------------------------------------------------------
def shard_path(run, job):
    return os.path.join(data_dir(run), "shards",
                        job.replace("/", "__") + ".npz")


def shard_meta(path):
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta"]))


def check_shard(run, j, plan, C, tag, done="_no"):
    path = shard_path(run, j["job"])
    m = shard_meta(path)
    if m is None:
        return None, None, ["нет шарда"]
    p = []
    if m.get("job") != j["job"] or m.get("plan") != plan["sha256"]:
        p.append("шард чужого задания или плана")
    if type(m.get("phys")) is not int or not 0 < m["phys"] <= EP:
        p.append("расход вне пределов")
    if p:
        return m, None, p
    with np.load(path, allow_pickle=False) as z:
        zz = dict(z)
    p += ep_problems(zz, m, j, C)
    if done != "_no" and done != dict(scope=tag, actual=m["phys"]):
        p.append(f"журнал {done} ≠ записи")
    return m, zz, p


def run_jobs(card, run, plan, jobs, ledger_path, caps, tag, k36, C):
    import k31_collect as kcol
    import k36_harness as kh
    L = kcol.Ledger(ledger_path, caps)
    by = {j["job"]: j for j in jobs}
    pd = L.d["pending"]
    if pd:
        if pd["id"] not in by or pd["scope"] != tag or pd["amount"] != EP:
            raise SystemExit(f"открытый резерв {pd['id']} не из очереди")
        m, _z, pr = check_shard(run, by[pd["id"]], plan, C, tag)
        if m is not None:
            if pr:
                raise SystemExit(f"резерв {pd['id']}: {pr[:3]}")
            L.settle(pd["id"], m["phys"])
    for s in SEEDS:
        js = [j for j in jobs if j["seed"] == s]
        if not js:
            continue
        v, arm = source(card, k36, s)
        for j in js:
            if shard_meta(shard_path(run, j["job"])) is not None:
                _m, _z, pr = check_shard(run, j, plan, C, tag,
                                         L.d["done"].get(j["job"]))
                if pr:
                    raise SystemExit(f"{j['job']}: {pr[:3]}")
                continue
            if L.is_done(j["job"]):
                raise SystemExit(f"{j['job']}: учтён, шарда нет")
            card.kcf.check_disk(rep_dir(run))
            L.reserve(tag, j["job"], amount=EP)
            ep = run_pulse(card.sim, v["io"], arm, j["scene"], s, j["amp"],
                           j["policy"], C, run)
            meta = save_ep(shard_path(run, j["job"]), j["job"], ep,
                           plan["sha256"], dict(level=j["level"],
                                                policy=j["policy"]))
            _m, _z, pr = check_shard(run, j, plan, C, tag)
            if pr:
                print(f"  {j['job']}: ПРОБЛЕМЫ {pr[:3]} — резерв открыт",
                      flush=True)
                return TECH
            L.commit(j["job"], meta["phys"])
            n = sum(L.is_done(x) for x in by)
            print(f"  {j['job']}: успех {meta['summary']['success']}, шагов "
                  f"{meta['summary']['steps']}, o4 {meta['target']['mid']}, "
                  f"пересчёт {meta['target']['refresh']}; {n}/{len(jobs)}",
                  flush=True)
        arm = v = None
        kh.free(card.torch)
    return OK


def preflight(a):
    import k36_harness as kh
    import k36_protocol as pc
    import k38_grip_run as kgr
    import k39_cache as kc
    run = a.run_id
    plan = load_plan(run)
    k36 = pc.load_prepare()
    C = kh.consts(k36)
    card = Card(a.device, run)
    tagc = f"{socket.gethostname()}__{card.g['uuid']}"
    d = os.path.join(rep_dir(run), "preflight", tagc)
    prob, first = [], {}
    try:
        card.start(run)
        # 1) T1 с нулевым импульсом = архив c2 (micro K-39: s0 t0, s1 t8)
        dp = kc.load_data_plan("f2")
        if dp["sha256"] != plan["ref_data_plan_sha256"]:
            raise SystemExit("data_plan f2 ≠ плану")
        eps = {(e["seed"], e["task"]): e for e in dp["episodes"]
               if e["split"] == "micro"}
        import k36_bank as kb36
        sim2 = kb36.BankSim("bank", 2, bank_dir=dp["bank"]["dir"])
        try:
            for (s, t) in ((0, 0), (1, 8)):
                e = eps[(s, t)]
                rec, rz = kc.load_input("f2", e)
                v, arm = source(card, k36, s)
                arm.begin(t, e["pos"])
                sc = dict(task=t, case_id=e["case_id"], pos=e["pos"], t=-1,
                          axis=0, sign=1)
                ep = run_pulse(sim2, v["io"], arm, sc, s, 0.0, "T1", C, run)
                sa = rec["summary"]
                if not np.array_equal(np.asarray(ep["actions"], np.float32),
                                      np.asarray(rz["actions"], np.float32)) \
                        or (ep["success"], ep["done_step"], ep["steps"]) != \
                        (sa["success"], sa["done_step"], sa["steps"]) or \
                        [ep["init_hash"], ep["init_hash_full"]] != \
                        list(e["hash"]):
                    prob.append(f"T1 нулевой s{s} t{t} ≠ архиву c2")
                arm = v = None
                kh.free(card.torch)
        finally:
            card.kcf.close_sim(sim2)
        # 2) калибровочная сцена каждого сида: обе политики на сильном уровне
        amp = plan["amp"]["strong"]
        for s in SEEDS:
            sc = plan["calib"][s % len(plan["calib"])]
            v, arm = source(card, k36, s)
            res = {}
            for pol in POLS:
                ep = run_pulse(card.sim, v["io"], arm, sc, s, amp, pol, C,
                               run, keep_o4=True)
                res[pol] = ep
            a1, a2 = res["T1"], res["RefreshOnce"]
            t1, t2 = a1["target"], a2["target"]
            if not t1["mid"]:
                prob.append(f"s{s}: калибровочная сцена без o4")
            else:
                cut = sc["t"] + 4
                if not np.array_equal(a1["actions"][:cut],
                                      a2["actions"][:cut]) or \
                        (t1["o0"], t1["o4"]) != (t2["o0"], t2["o4"]):
                    prob.append(f"s{s}: префикс или o4 пары различны")
                if not t2["refresh"] or t1["refresh"] or \
                        a1["heavy"] != len(a1["calls"]) or \
                        a2["heavy"] != len(a2["calls"]) + 1 or \
                        t2["P"] is None or np.array_equal(t2["P"],
                                                          t2["B"][:4]):
                    prob.append(f"s{s}: импульс или один пересчёт не "
                                f"подтверждены")
                if a.reference:      # прямой вызов T32 в o4 = записанный R
                    import k36_harness as kh2
                    arm2 = kh2.make_arm(card.torch, k36, f"T32_s{s}", v)
                    arm2.begin(sc["task"], sc["pos"],
                               ep_number(run, sc["task"], sc["case_id"], s))
                    arm2.call = sc["t"] // 8
                    io = v["io"]
                    b2, p2, s82 = io.prep(t2["o4_obs"],
                                          io.task_desc(sc["task"]),
                                          sc["task"])
                    R2, _ = arm2.decide(b2, p2, s82, sc["t"] + 4)
                    if not np.array_equal(np.asarray(R2, np.float32),
                                          t2["R"]):
                        prob.append(f"s{s}: прямой T32 в o4 ≠ R")
                    arm2 = None
                first[str(s)] = dict(
                    o0=t1["o0"], o4=t1["o4"], succ=[a1["success"],
                                                    a2["success"]],
                    R4=t2["R"][:4].tolist())
            arm = v = None
            kh.free(card.torch)
    finally:
        card.close()
    refp = os.path.join(rep_dir(run), "preflight", "reference.json")
    if a.reference:
        if not prob:
            pc.save_json(refp, dict(first=first, plan=plan["sha256"],
                                    uuid=card.g["uuid"]))
    else:
        ref = pc.load_json(refp) if os.path.exists(refp) else None
        if not ref or ref.get("plan") != plan["sha256"]:
            raise SystemExit("нет эталона допуска этого плана")
        for s, v_ in first.items():
            r = ref["first"].get(s)
            if r is None or (r["o0"], r["o4"], r["succ"]) != (
                    v_["o0"], v_["o4"], v_["succ"]):
                prob.append(f"s{s}: хеши или исходы ≠ эталону")
            elif np.abs(np.subtract(r["R4"], v_["R4"]))[:, :6].max() > \
                    TOL["arm"] or np.subtract(r["R4"], v_["R4"])[:, 6].any():
                prob.append(f"s{s}: новый хвост ≠ эталону")
    rep = dict(admitted=not prob, problems=prob, first=first,
               plan=plan["sha256"], env=kgr.env_record(card.torch),
               uuid=card.g["uuid"],
               peak_gb=float(card.torch.cuda.max_memory_reserved(card.idx)
                             / 2 ** 30))
    pc.save_json(d + ".json", rep)
    print(f"  preflight K-39 B {a.device}: " + ("ДОПУЩЕНА" if not prob else
                                                "НЕ допущена: " +
                                                "; ".join(prob[:3])))
    return OK if not prob else TECH


def assign(a):
    import fcntl
    import k36_protocol as pc
    plan = load_plan(a.run_id, verify=False)
    path = os.path.join(rep_dir(a.run_id), "assignments.json")
    with open(os.path.join(rep_dir(a.run_id), ".assign.lock"), "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        d = json.load(open(path)) if os.path.exists(path) else \
            dict(history=[])
        k = int(a.part)
        if str(k) not in plan["parts"]:
            raise SystemExit(f"часть {k}")
        last = next((h for h in reversed(d["history"]) if h["part"] == k),
                    None)
        if a.state == "assigned" and last and last["state"] == "assigned" \
                and (last["host"], last["uuid"]) != (a.host, a.uuid):
            raise SystemExit(f"part {k} уже назначен")
        rec = dict(part=k, host=a.host or last["host"], uuid=a.uuid or
                   last["uuid"], state=a.state, plan_sha256=plan["sha256"],
                   n=1 + sum(h["part"] == k for h in d["history"]),
                   time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        d["history"].append(rec)
        pc.save_json(path, d)
        pc.save_json(os.path.join(rep_dir(a.run_id), f"part-{k}",
                                  "assignment.json"), rec)
        fcntl.flock(fh, fcntl.LOCK_UN)
    print(f"  part {k}: {a.state} -> {rec['host']} {rec['uuid']}")
    return OK


def collect(a):
    import k36_harness as kh
    import k36_protocol as pc
    import k38_grip_run as kgr
    run, k = a.run_id, int(a.part)
    plan = load_plan(run)
    pd = os.path.join(rep_dir(run), f"part-{k}")
    asg = pc.load_json(os.path.join(pd, "assignment.json"))
    card = Card(a.device, run)
    try:
        if asg.get("plan_sha256") != plan["sha256"] or asg.get("uuid") != \
                card.g["uuid"] or asg.get("host") != socket.gethostname() or \
                asg.get("state") != "assigned":
            raise SystemExit(f"part {k}: назначение не этой карте/плану")
        adm = pc.load_json(os.path.join(rep_dir(run), "preflight",
                                        f"{socket.gethostname()}__"
                                        f"{card.g['uuid']}.json"))
        if not adm.get("admitted") or adm.get("plan") != plan["sha256"] or \
                adm.get("env") != kgr.env_record(card.torch):
            raise SystemExit("карта не допущена для этого плана/среды")
        k36 = pc.load_prepare()
        C = kh.consts(k36)
        card.start(run)
        partlock = card.kcf.Lock(os.path.join(pd, ".lock"), f"part {k}",
                                 card.kcf.Busy)
        try:
            rc = run_jobs(card, run, plan, queue(plan, k),
                          os.path.join(pd, "ledger.json"),
                          dict(screen=plan["parts"][str(k)]["cap"]), "screen",
                          k36, C)
        finally:
            partlock.release()
    finally:
        card.close()
    print(f"  part {k}: " + ("готова" if rc == OK else "остановлена"))
    return rc


def status(a):
    plan = load_plan(a.run_id, verify=False)
    for k, v in plan["parts"].items():
        lp = os.path.join(rep_dir(a.run_id), f"part-{k}", "ledger.json")
        if not os.path.exists(lp):
            print(f"  part {k}: не начата (0/{v['n']})")
            continue
        d = json.load(open(lp))
        print(f"  part {k}: {len(d['done'])}/{v['n']}; расход {d['spent']}; "
              f"резерв {(d['pending'] or {}).get('id', '—')}; аварий "
              f"{len(d['crashes'])}")
    return OK


# --- слияние и анализ ---------------------------------------------------------------------------------
def build_merge(run, plan):
    import k36_harness as kh
    import k36_protocol as pc
    C = kh.consts(pc.load_prepare())
    prob, rows, got = [], [], {}
    for k in range(N_PARTS):
        lp = os.path.join(rep_dir(run), f"part-{k}", "ledger.json")
        if not os.path.exists(lp):
            prob.append(f"part {k}: нет журнала")
            continue
        led = pc.load_json(lp)
        jobs = queue(plan, k)
        if led["pending"] or set(led["done"]) != {j["job"] for j in jobs}:
            prob.append(f"part {k}: журнал ≠ плану")
        tot = 0
        for j in jobs:
            m, zz, pr = check_shard(run, j, plan, C, "screen",
                                    led["done"].get(j["job"]))
            prob += [f"{j['job']}: {x}" for x in pr]
            if m is None or pr:
                continue
            tot += m["phys"]
            sc = j["scene"]
            got[(sc["task"], sc["case_id"], j["seed"], j["level"],
                 j["policy"])] = (m, zz)
            K = {kk[2:]: zz[kk] for kk in zz if kk.startswith("k_")}
            ti = np.flatnonzero(K["target"])
            replaced = bool(m["target"]["refresh"] and len(ti) and not
                            np.array_equal(K["final"][ti[0]][4:8],
                                           zz["j_a_codec"][ti[0]][4:8]))
            rows.append(dict(task=sc["task"], case_id=sc["case_id"],
                             seed=j["seed"], level=j["level"],
                             policy=j["policy"],
                             success=m["summary"]["success"],
                             reached=m["target"]["reached"],
                             o4=m["target"]["mid"],
                             clip=K["clip"].tolist(),
                             eef4=K["eef4"].tolist(), replaced=replaced,
                             phys=m["phys"], seconds=m["seconds"]))
        crash = sum(int(c["amount"]) for c in led["crashes"])
        if led["spent"].get("screen", 0) != tot + crash:
            prob.append(f"part {k}: расход ≠ сумме")
    # пары и уровни
    keys = {(r["task"], r["case_id"], r["seed"]) for r in rows}
    for key in keys:
        for lev in LEVELS:
            a_, b_ = got.get(key + (lev, "T1")), got.get(key + (lev,
                                                                "RefreshOnce"))
            if a_ is None or b_ is None:
                prob.append(f"{key} {lev}: неполная пара")
                continue
            prob += [f"{key} {lev}: {x}" for x in pair_problems(a_, b_)]
        z0 = got.get(key + ("zero", "T1"))
        for lev in ("mild", "strong"):
            zl = got.get(key + (lev, "T1"))
            if z0 is None or zl is None:
                continue
            m0, a0 = z0
            ml, al = zl
            if m0["target"]["reached"] != ml["target"]["reached"] or \
                    m0["target"]["o0"] != ml["target"]["o0"]:
                prob.append(f"{key} {lev}: o0 ≠ нулевому уровню")
            elif m0["target"]["reached"]:
                t = int(np.flatnonzero(a0["k_target"])[0])
                cut = int(a0["j_elapsed"][t]) + 1
                if not np.array_equal(a0["actions"][:cut],
                                      al["actions"][:cut]):
                    prob.append(f"{key} {lev}: история до импульса ≠")
    want = sum(v["n"] for v in plan["parts"].values())
    if len(rows) != want:
        prob.append(f"эпизодов {len(rows)} ≠ {want}")
    return prob, rows


def merge(a):
    import k36_protocol as pc
    plan = load_plan(a.run_id)
    prob, rows = build_merge(a.run_id, plan)
    pc.save_json(os.path.join(rep_dir(a.run_id), "merged.json"),
                 dict(plan=plan["sha256"], ok=not prob, problems=prob[:100],
                      rows=rows if not prob else []))
    print("  merge: " + (f"{len(rows)} эпизодов, проверка пройдена"
                         if not prob else "ОТКАЗ: " + "; ".join(prob[:4])))
    return OK if not prob else TECH


def stats(rows, n_boot=BOOT["n"], seed=BOOT["seed"]):
    tab = {}
    for r in rows:
        tab[(r["task"], r["case_id"], r["seed"], r["level"], r["policy"])] = \
            int(r["success"])
    scenes_ = sorted({(r["task"], r["case_id"]) for r in rows})
    out = {}
    dsc = {}
    for lev in LEVELS:
        dif = {}
        for (t, c) in scenes_:
            for s in SEEDS:
                y2 = tab.get((t, c, s, lev, "RefreshOnce"))
                y1 = tab.get((t, c, s, lev, "T1"))
                if y1 is not None and y2 is not None:
                    dif[(t, c, s)] = y2 - y1
        W = sum(v == 1 for v in dif.values())
        L_ = sum(v == -1 for v in dif.values())
        n = len(dif)
        out[lev] = dict(N=n, W=W, L=L_, D=W - L_, pp=100 * (W - L_) / max(n, 1),
                        sr_T1=float(np.mean([tab[(t, c, s, lev, "T1")]
                                             for (t, c, s) in dif])),
                        sr_R=float(np.mean([tab[(t, c, s, lev, "RefreshOnce")]
                                            for (t, c, s) in dif])),
                        per_seed={str(s): int(sum(v for k, v in dif.items()
                                                  if k[2] == s))
                                  for s in SEEDS},
                        by_task={str(t): int(sum(v for k, v in dif.items()
                                                 if k[0] == t))
                                 for t in TASKS},
                        rescue_scenes=len({(k[0], k[1]) for k, v in
                                           dif.items() if v == 1}))
        dsc[lev] = {sc: np.mean([dif[sc + (s,)] for s in SEEDS
                                 if sc + (s,) in dif]) for sc in scenes_}
    rng = np.random.default_rng(seed)
    ts = sorted({t for t, _c in scenes_})
    idx = {t: rng.integers(0, sum(1 for x in scenes_ if x[0] == t),
                           (n_boot, sum(1 for x in scenes_ if x[0] == t)))
           for t in ts}

    def boot(vals):
        per = {t: np.array([vals[sc] for sc in scenes_ if sc[0] == t])
               for t in ts}
        est = float(np.mean([per[t].mean() for t in ts]))
        bs = np.mean([per[t][idx[t]].mean(1) for t in ts], 0)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        return dict(est=est, lo=float(lo), hi=float(hi))
    for lev in LEVELS:
        out[lev]["delta"] = boot(dsc[lev])
    out["I_strong"] = boot({sc: dsc["strong"][sc] - dsc["zero"][sc]
                            for sc in scenes_})
    out["signal"] = bool(out["strong"]["D"] > 0 and
                         out["strong"]["D"] > out["zero"]["D"])
    return out


def analyze(a):
    import k36_protocol as pc
    plan = load_plan(a.run_id)
    m = pc.load_json(os.path.join(rep_dir(a.run_id), "merged.json"))
    if not m.get("ok") or m.get("plan") != plan["sha256"]:
        raise SystemExit("merge не пройден")
    prob, rows = build_merge(a.run_id, plan)
    if prob:
        raise SystemExit("проверка при анализе: " + "; ".join(prob[:3]))
    st = stats(rows)
    desc = {}
    for lev in LEVELS:
        rs = [r for r in rows if r["level"] == lev]
        desc[lev] = dict(reached=float(np.mean([r["reached"] for r in rs])),
                         o4=float(np.mean([r["o4"] for r in rs])),
                         replaced=float(np.mean([r["replaced"] for r in rs
                                                 if r["policy"] ==
                                                 "RefreshOnce"])),
                         clip=float(np.mean([x for r in rs for x in r["clip"]]))
                         if lev != "zero" else 0.0)
    t1 = {(r["task"], r["case_id"], r["seed"], r["level"]): r for r in rows
          if r["policy"] == "T1"}
    for lev in ("mild", "strong"):
        d_ = [float(np.linalg.norm(np.subtract(r["eef4"], t1[k[:3] + (
            "zero",)]["eef4"]))) for k, r in t1.items() if k[3] == lev and
            r["o4"] and t1.get(k[:3] + ("zero",), {}).get("o4")]
        desc[lev]["eef_disp"] = dict(
            n=len(d_), med=float(np.median(d_)) if d_ else None,
            p90=float(np.quantile(d_, 0.9)) if d_ else None)
    pc.save_json(os.path.join(rep_dir(a.run_id), "analysis.json"),
                 dict(plan=plan["sha256"], stats=st, desc=desc, boot=BOOT))
    for lev in ("mild", "strong"):
        e_ = desc[lev]["eef_disp"]
        if e_["n"]:
            print(f"  смещение кисти {lev} к zero (T1, o4): медиана "
                  f"{e_['med']:.4f} м, p90 {e_['p90']:.4f} м, ключей "
                  f"{e_['n']}")
    for lev in LEVELS:
        v = st[lev]
        print(f"  {lev}: SR T1 {100 * v['sr_T1']:.1f} / Refresh "
              f"{100 * v['sr_R']:.1f}; W {v['W']} L {v['L']} D {v['D']:+d} "
              f"(N {v['N']}); Δ {100 * v['delta']['est']:+.1f} п.п. 95% "
              f"[{100 * v['delta']['lo']:+.1f}; {100 * v['delta']['hi']:+.1f}]"
              f"; по сидам {v['per_seed']}; сцен со спасением "
              f"{v['rescue_scenes']}; o4 {desc[lev]['o4']:.2f}, заменён хвост "
              f"{desc[lev]['replaced']:.2f}")
    i = st["I_strong"]
    print(f"  I_strong = Δ_strong − Δ_zero: {100 * i['est']:+.1f} п.п. 95% "
          f"[{100 * i['lo']:+.1f}; {100 * i['hi']:+.1f}]")
    print(f"  K-39 B: {'сигнал' if st['signal'] else 'нет сигнала'} "
          f"(D_strong > 0 и > D_zero)")
    return OK


# --- самопроверка -------------------------------------------------------------------------------------
def selftest():
    import tempfile
    from collections import Counter
    import torch
    import k21_model as km21
    import k25_model as k25m
    import k29_model as k29m
    import k32_model as km32
    import k32_ppo as kppo
    import k31_collect as kcol
    import k36_harness as kh
    k29m.no_tf32(torch)
    torch.manual_seed(0)
    C = dict(half=np.ones(6, np.float32), mid=np.zeros(6), L=-np.ones(6),
             U=np.ones(6))
    # импульс: строки 1–2 по оси, клип, ноль — побитово
    B = np.zeros((8, 7), np.float32)
    B[:, 0] = 0.9
    B[:, 6] = 1.0
    P, dl, cl = perturb(B, 0, 1, 0.3, C)
    assert P[1, 0] == 1.0 and P[2, 0] == 1.0 and cl.all()
    assert np.array_equal(P[[0, 3]], B[[0, 3]]) and \
        np.array_equal(P[:, 1:], B[:4, 1:])
    P0, _d, _c = perturb(B, 0, 1, 0.0, C)
    assert np.array_equal(P0, B[:4])
    P2, dl2, cl2 = perturb(B, 1, -1, 0.3, C)
    assert not cl2.any() and np.allclose(dl2, -0.3)
    # очередь: 2 части × 144, все 6 вариантов сцены в одной части
    pil = [dict(task=t, case_id=f"c{t}_{r}", pos=r + 1, rank=r, t=24 + 8 * r,
                axis=r // 2, sign=1 if r % 2 == 0 else -1, hash=None)
           for t in TASKS for r in range(4)]
    plan = dict(pilot=pil, amp=dict(AMP))
    qs = [queue(plan, k) for k in range(N_PARTS)]
    assert [len(q) for q in qs] == [144, 144]
    assert len({j["job"] for q in qs for j in q}) == 288
    for q in qs:
        c = Counter((j["scene"]["case_id"], j["seed"]) for j in q)
        assert set(c.values()) == {6}
        assert Counter(j["scene"]["task"] for j in q) == {t: 24 for t in TASKS}
    allq = qs[0] + qs[1]
    for lev in LEVELS:
        js = [j for j in allq if j["level"] == lev]
        firsts = [js[i]["policy"] for i in range(0, len(js), 2)]
        assert Counter(firsts) == {"T1": 24, "RefreshOnce": 24}, lev
    # игрушечный полный цикл: 6 комбинаций одного старта
    M = k29m.build(torch)
    Lb = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    Ub = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    mod = k25m.build(torch)["K25"](km21.fake_bar(torch), Lb, Ub,
                                   seed=0).eval()
    for p_ in mod.parameters():
        p_.requires_grad_(False)
    dec = M["Decoder"](mod).freeze()
    H = mod.H
    norm = km32.fit_normalizer(np.random.default_rng(0).normal(
        size=(64, 6 * H + 72)), H)
    actor, _c = km32.build(torch, H, 6, 0, norm)
    with torch.no_grad():
        g2 = torch.Generator().manual_seed(7)
        for p_ in actor.out.parameters():
            p_.copy_(torch.randn(p_.shape, generator=g2))

    class Sim(kppo.ToySim):
        END = 22                    # done на этом шаге среды (10 — ожидание)

        def step(self, a):
            self.t += 1
            self.acc += float(np.asarray(a)[0, 0])
            if self.t <= 10:
                return self._obs(), np.array([0.0]), np.array([False]), \
                    np.array([False])
            d = self.t >= self.END
            s_ = bool(d and (self.init + int(abs(self.acc) * 1e3)) % 2 == 0)
            return self._obs(), np.array([float(s_)]), np.array([d]), \
                np.array([s_])

        def _obs(self):
            o = super()._obs()
            for k in ("agentview_image", "robot0_eye_in_hand_image"):
                o[k] = np.broadcast_to(o[k][:, :1, :1], (1, 224, 224, 3)
                                       ).copy()
            return o

    class FIO(kcol.FakeIO):
        @property
        def inp(self):
            return self

        def state8(self, obs):
            return np.asarray(obs["state"][:, :8], np.float64)

        def prep(self, obs, desc, task):
            x = float(obs["state"][0, 0])
            bt = km21.fake_batch(torch, B=1, seed=int(abs(x) * 1e3) % 13)
            s8 = np.zeros((1, 8))
            s8[0, :3] = x
            return bt, 1, s8
    io = FIO()
    k36 = dict(k32=dict(k32_contract="C", seeds={"0": dict(
        source=dict(weights_hash="W"), T=dict(actor_hash="A"))}),
        k32_plan=dict(bounds=dict(L=list(Lb) + [-1.0], U=list(Ub) + [1.0]),
                      scale_q=np.asarray(io.max_act_q).tolist()))
    Ck = kh.consts(k36)
    src = dict(mod=mod, dec=dec, M=M, io=io, actors=dict(T=actor))
    sc = dict(task=0, case_id="toy", pos=3, rank=0, t=8, axis=0, sign=1,
              hash=None)
    g_ = globals()
    keep = g_["shard_path"]
    with tempfile.TemporaryDirectory() as td:
        g_["shard_path"] = lambda run, job: os.path.join(
            td, job.replace("/", "__") + ".npz")
        try:
            for END in (22, 40):    # terminal ровно после 4-го / позже
                Sim.END = END
                got = {}
                for lev in LEVELS:
                    for pol in POLS:
                        arm = kh.make_arm(torch, k36, "T32_s0", src)
                        ep = run_pulse(Sim(), io, arm, sc, 0, AMP[lev], pol, Ck,
                                       "toy")
                        tgi = [i for i, c in enumerate(ep["calls"])
                               if c["target"]]
                        assert len(tgi) == 1
                        if END == 40:
                            assert ep["target"]["mid"] and \
                                len(ep["calls"]) > tgi[0] + 1
                            assert ep["target"]["refresh"] == \
                                (pol == "RefreshOnce")
                            assert ep["heavy"] == len(ep["calls"]) + int(
                                pol == "RefreshOnce")
                        else:
                            assert not ep["target"]["mid"] and \
                                ep["heavy"] == len(ep["calls"])
                        j = dict(job=f"ep/{lev}/{pol}/s0/t0/toy", level=lev,
                                 amp=AMP[lev], policy=pol, seed=0, scene=sc)
                        m = save_ep(shard_path("toy", j["job"]), j["job"], ep,
                                    "P", dict(level=lev, policy=pol))
                        pl = dict(sha256="P")
                        mm, zz, pr = check_shard("toy", j, pl, Ck, "screen")
                        assert pr == [], (lev, pol, pr)
                        got[(lev, pol)] = (mm, zz)
                        assert m["target"]["reached"]
                        if pol == "RefreshOnce" and m["target"]["mid"]:
                            assert m["target"]["refresh"] and m["heavy"] == \
                                len(ep["calls"]) + 1
                    assert pair_problems(got[(lev, "T1")],
                                         got[(lev, "RefreshOnce")]) == []
            # порча: хвост, хеш o4, чужой префикс — отказ
            mm, zz = got[("strong", "RefreshOnce")]
            bad = dict(zz)
            ti = int(np.flatnonzero(zz["k_target"])[0])
            bad["k_final"] = zz["k_final"].copy()
            bad["k_final"][ti, 5, 0] += 0.01
            assert ep_problems(bad, mm, dict(
                job="x", level="strong", amp=AMP["strong"],
                policy="RefreshOnce", seed=0, scene=sc), Ck)
            m2 = json.loads(json.dumps(mm))
            m2["target"]["o4"] = "чужой"
            assert pair_problems((m2, zz), got[("strong", "T1")])
            za = dict(got[("strong", "T1")][1])
            za["actions"] = za["actions"].copy()
            za["actions"][0, 0] += 1
            assert pair_problems((got[("strong", "T1")][0], za),
                                 got[("strong", "RefreshOnce")])
            # target без хешей / с чужим k — отказ эпизода и пары
            m3 = json.loads(json.dumps(mm))
            m3["target"].update(o0=None, o4=None, k=0)
            jb = dict(job="x", level="strong", amp=AMP["strong"],
                      policy="RefreshOnce", seed=0, scene=sc)
            assert ep_problems(zz, m3, jb, Ck)
            m4 = json.loads(json.dumps(got[("strong", "T1")][0]))
            m4["target"].update(o0=None, o4=None)
            m5 = json.loads(json.dumps(mm))
            m5["target"].update(o0=None, o4=None)
            assert pair_problems((m4, got[("strong", "T1")][1]), (m5, zz))
            # трасса кисти: по точке на шаг, действия не меняются
            arm = kh.make_arm(torch, k36, "T32_s0", src)
            et = run_pulse(Sim(), io, arm, sc, 0, AMP["strong"], "T1", Ck,
                           "toy", trace=True)
            assert et["eef_trace"].shape == (et["steps"], 3) and \
                np.array_equal(et["actions"],
                               got[("strong", "T1")][1]["actions"])
            # нулевой T1 (END 40) = обычный эпизод K-39 T1
            import k39_run as kr
            arm = kh.make_arm(torch, k36, "T32_s0", src)
            e1 = kr.run_episode(Sim(), io, arm, None, "T1", 0, 3)
            assert np.array_equal(e1["actions"],
                                  got[("zero", "T1")][1]["actions"])
            # без записи журнала — отказ
            j = dict(job="ep/zero/T1/s0/t0/toy", level="zero", amp=0.0,
                     policy="T1", seed=0, scene=sc)
            assert check_shard("toy", j, dict(sha256="P"), Ck, "screen",
                               None)[2]
        finally:
            g_["shard_path"] = keep
    # калибровка: решение о 0.225 по обоим сидам; выбранный уровень без o4
    def crow(t, s_, lev, mid, clip, d=0.3):
        return dict(task=t, seed=s_, level=lev, k=4 if mid else 1, mid=mid,
                    reached=True, clip=[clip, clip], delta=[d, d],
                    eef4=[0.0, 0.0, d] if mid else None)
    rr = []
    for t in TASKS:
        for s_ in SEEDS:
            rr += [crow(t, s_, "zero", True, False, 0.0),
                   crow(t, s_, "mild", True, False, 0.15),
                   crow(t, s_, "strong", True, s_ == 0 or t < 5),
                   crow(t, s_, "alt", False, False, 0.225)]
    cs = summarize_cal(rr)
    assert cs["chosen"] == "alt" and cs["status"] == "needs_review"
    cs = summarize_cal([r for r in rr if r["level"] != "alt"])
    assert cs["status"] == "needs_review"          # выбран alt, его нет
    rr2 = [dict(r, clip=[False, False]) if r["level"] == "strong" else r
           for r in rr if r["level"] != "alt"]
    cs = summarize_cal(rr2)
    assert cs["chosen"] == "strong" and cs["status"] == "ok", cs["reason"]
    assert executed_deltas(dict(delta=[0.3, 0.3], k=2)) == [0.3]
    rr3 = [dict(r, mid=False, k=2, eef4=None)
           if (r["level"], r["task"], r["seed"]) == ("strong", 0, 0) else
           (dict(r, mid=True, k=4, eef4=[0.0, 0.0, 0.225])
            if r["level"] == "alt" else r) for r in rr]
    cs = summarize_cal(rr3)
    assert cs["chosen"] == "alt" and cs["status"] == "ok", cs["reason"]
    assert cs["levels"]["strong"]["n_o4"] == 11 and \
        cs["levels"]["alt"]["n_cmp"] == 12 and "zero" in cs["levels"]
    assert cal_row_problems(dict(job="cal/x", physical=-9), "cal/x")
    assert cal_row_problems(dict(job="cal/y", physical=40), "cal/x")
    assert cal_row_problems(dict(job="cal/x", physical=40), "cal/x") == []
    pz = lambda t, s_, a_, ok, e: dict(  # noqa: E731
        task=t, seed=s_, amp=a_, success=ok, mid=True, k=4, reached=True,
        case_id=f"c{t}", t=24, axis=0, init=[f"h{t}", "f"], o0=f"o{t}{s_}",
        prefix_sha1=f"p{t}{s_}", actions_sha1=f"a{t}{s_}{a_}",
        clip=[False, False], eef={str(d): [0.0, 0.0, e if d < 16 else 0.0]
                                  for d in PROBE_OFFSETS})
    pr_ = [pz(t, s_, 0.0, True, 0.0) for t in TASKS for s_ in SEEDS] + \
        [pz(t, s_, 0.3, not (t == 0 and s_ == 1), 0.006)
         for t in TASKS for s_ in SEEDS]
    sp = summarize_probe(pr_)
    assert sp["problems"] == [], sp["problems"]
    assert sp["sr_zero"] == 12 and sp["levels"]["0.3"]["broke"] == 1
    bad_ = [dict(r, o0="чужой") if (r["task"], r["seed"], r["amp"]) ==
            (0, 1, 0.3) else r for r in pr_]
    assert summarize_probe(bad_)["problems"]
    bad_ = [dict(r, init=["x", "y"]) if (r["task"], r["seed"], r["amp"]) ==
            (2, 0, 0.3) else r for r in pr_]
    assert summarize_probe(bad_)["problems"]
    six = pr_ + [pz(t, s_, 0.6, True, 0.012) for t in TASKS for s_ in SEEDS]
    six[-1] = dict(six[-1], amp=0.3)          # подмена amp → дубль/нехватка
    assert summarize_probe(six)["problems"]
    assert abs(sp["levels"]["0.3"]["eef"]["8"]["med"] - 0.006) < 1e-9 and \
        sp["levels"]["0.3"]["eef"]["16"]["med"] == 0.0
    # анализ: известные W/L и взаимодействие
    rows = []
    for t in TASKS:
        for c in range(4):
            for s in SEEDS:
                for lev in LEVELS:
                    y1 = c > 0
                    y2 = {"zero": y1, "mild": y1,
                          "strong": True if c == 0 else not (c == 1 and s == 0)
                          }[lev]
                    rows += [dict(task=t, case_id=f"c{t}_{c}", seed=s,
                                  level=lev, policy="T1", success=y1),
                             dict(task=t, case_id=f"c{t}_{c}", seed=s,
                                  level=lev, policy="RefreshOnce",
                                  success=y2)]
    st = stats(rows, n_boot=500)
    assert st["strong"]["W"] == 12 and st["strong"]["L"] == 6
    assert st["zero"]["D"] == 0 and st["signal"]
    assert st["I_strong"]["est"] > 0
    print("самопроверка k39_pulse пройдена: импульс (строки 1–2, ось, клип, "
          "ноль побитово); очередь 2 × 144, все 6 вариантов сцены в части; "
          "порядок T1 первым 24/48 на уровне; игрушечный цикл при terminal "
          "на 4-м шаге и позже: 6 комбинаций приняты, один пересчёт и "
          "возврат к T1; порча хвоста/o4/префикса/target и шард без журнала "
          "— отказ; нулевой T1 = T1 K-39; калибровка по обоим сидам; трасса и сводка "
          "probe; W/L/D и I")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-39 B: импульс и пересчёт")
    ap.add_argument("cmd", choices=("bank", "calibrate", "probe", "prepare",
                                    "gpuinfo",
                                    "assign", "preflight", "collect",
                                    "status", "merge", "analyze"))
    ap.add_argument("--run-id", default="b1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--host")
    ap.add_argument("--uuid")
    ap.add_argument("--state", default="assigned",
                    choices=("assigned", "released", "done"))
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amps", default="0.3")
    a = ap.parse_args()
    import k37_confirm as kcf
    fn = dict(bank=bank, calibrate=calibrate, probe=probe, prepare=prepare,
              gpuinfo=kcf.gpuinfo, assign=assign, preflight=preflight,
              collect=collect, status=status, merge=merge,
              analyze=analyze)[a.cmd]
    try:
        return fn(a)
    except kcf.Busy as e:
        print(f"  ВРЕМЕННО НЕТ РЕСУРСА: {e}", flush=True)
        return BUSY
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return TECH
        raise


if __name__ == "__main__":
    sys.exit(main())
