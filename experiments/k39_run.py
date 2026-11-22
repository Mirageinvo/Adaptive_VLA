#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-39 f2: развилка 4+4 в замкнутом цикле, SR и задержка.

Команды (run = reports/k39/<run_id>):
  bank                                   банк space 3 (задачи 0/2/3/5/6/8)
  prepare-screen --run-id f2 --profile one|two|three
  gpuinfo --device D
  assign --run-id f2 --part N --host H --uuid U [--state …]
  preflight --run-id f2 --device D [--reference]
  collect --run-id f2 --part N --device D
  status | merge | analyze --run-id f2
  latency --run-id f2 --device D
Окружение: OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 (проверяется).

Политики (у всех один тяжёлый вызов T32 в начале чанка; исполняются
B[0:4] своего вызова; в o4 до первого действия хвоста — решение):
  T1         хвост B[4:8];
  TreeOld    селектор по признакам o0 из банка (B, C1, C2);
  TreeFresh  тот же банк и веса, селектор по o4;
  FlatFresh  flat по o4 выдаёт хвост;
  Refresh4   второй тяжёлый вызов T32 в o4 (elapsed t + 4), хвост —
             первые 4 команды нового прогноза (подмножество сцен).
Банк C1/C2 и признаки o0 — в начале чанка; в середине Tree/Flat не
вызывают T32 (счётчик тяжёлых вызовов на границе Arm.forward).
Журнал вызова: номинальный B и запись T32 (проверка K-32 по реконструкции
номинальных команд), C1/C2, оценки селектора, выбор, итоговый чанк,
исполненное k, достигнута ли середина, второй прогноз Refresh4.
Проверка: префикс = B[:4]; хвост = выбранному кандидату (выбор = argmax
оценок, равенство -> B, C1), flat/refresh — записанному; рука в [L, U];
схват ±1; исполненное = io.scale(итоговых чанков)[:k]; счётчик тяжёлых
вызовов = числу чанков (+ числу середин у Refresh4).
Анализ: пары сцена × сид; W/L/D для TreeFresh − TreeOld (главный),
TreeFresh − FlatFresh, TreeFresh − T1, Refresh4 − T1 (подмножество);
bootstrap по сценам в задаче (50 000, seed 39001), задачи с равным весом;
тройки 0/6/8 и 2/3/5 отдельно; fresh_signal: 100·D ≥ 3·N и D_s ≥ 0;
tree_dev_candidate: ещё D(Fresh − Flat) > 0 и D(Fresh − T1) > 0.
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

PROTOCOL = "k39-run-v1"
TASKS = (0, 2, 3, 5, 6, 8)
TRAIN_TASKS, TRANSFER_TASKS = (0, 6, 8), (2, 3, 5)
SEEDS = (0, 1)
MAIN = ("T1", "TreeOld", "TreeFresh", "FlatFresh")
SPACE = 3
BANK_DIR = "data/k39/bank"
BANK_MANIFEST = "reports/k39/bank_manifest.json"
BANK_LEDGER = "data/k39/bank_ledger.json"
N_BANK, MAX_CAND = 12, 48
PROFILES = dict(one=4, two=8, three=12)
EP = 610
BOOT = dict(n=50_000, seed=39001)
TIE_TOL = 0.0
OK, TECH, BUSY = 0, 3, 4
CODE = ("k39_run.py", "k39_model.py", "k39_cache.py", "k37_confirm.py",
        "k37_bank.py", "k38_grip_run.py", "k36_harness.py", "k36_bank.py",
        "k32_policy.py", "k32_harness.py", "k31_collect.py", "k29_policy.py",
        "k33_bank.py")


def h256(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sha256(p):
    import k37_confirm as kcf
    return kcf.sha256(p)


def rep_dir(run):
    return os.path.join("reports/k39", run)


def screen_dir(run):
    return os.path.join(rep_dir(run), "screen")


# --- банк space 3 -------------------------------------------------------------------------------
def bank(a):
    import k31_collect as kcol
    import k33_bank as kb
    import k33_protocol as p33
    import k37_bank as k37b
    if os.path.exists(BANK_MANIFEST):
        m = json.load(open(BANK_MANIFEST, encoding="utf-8"))
        bad = [p for p, s in m["files"].items()
               if not os.path.exists(p) or sha256(p) != s]
        if bad:
            raise SystemExit(f"банк закреплён, файлы изменены: {bad}")
        print(f"  {BANK_MANIFEST}: закреплён (bank_id {m['bank_id'][:12]})")
        return OK
    L = kcol.Ledger(BANK_LEDGER, dict(bank=10 * (len(TASKS) * MAX_CAND +
                                                 len(TASKS) * N_BANK)))
    seen, seeds0 = k37b.k33_seen()
    for t in k37b.TASKS:                       # банк K-37 (space 2)
        z = np.load(kb.task_path(k37b.SPACE, t, k37b.BANK_DIR))
        seen.update(str(x) for x in z["sig"])
        seeds0.update(int(x) for x in z["seed"])
    files, per = {}, {}
    for t in TASKS:
        official = set(kb.official_joint_sigs(t))
        jobs = [(SPACE, t, i) for i in range(MAX_CAND)]
        res = k37b.run_chunk(L, f"bank/gen/t{t}", jobs, kb._gen_job,
                             a.workers, os.path.join("data/k39/bank_work",
                                                     f"gen_t{t}.pkl"))
        cases, rej = kb.select(res, N_BANK, seen, official)
        if len(cases) < N_BANK:
            raise SystemExit(f"t{t}: допустимых {len(cases)} < {N_BANK}")
        p = kb.save_task(SPACE, t, cases, cases[0]["meta"], BANK_DIR)
        files[p] = sha256(p)
        per[str(t)] = dict(cases=len(cases), rejects=rej)
    seeds = [int(x) for p_ in files for x in np.load(p_)["seed"]]
    if len(set(seeds)) != len(seeds) or set(seeds) & seeds0:
        raise SystemExit("коллизия seed")
    jobs = []
    for t in TASKS:
        cs, meta = kb.load_task(SPACE, t, BANK_DIR)
        jobs += [(SPACE, t, c, meta) for c in cs]
    res = k37b.run_chunk(L, "bank/replay", jobs, kb._replay_job, a.workers,
                         "data/k39/bank_work/replay.pkl")
    by = {r["case_id"]: r for r in res}
    bad = [c["case_id"] for _s, _t, c, _m in jobs
           if not by[c["case_id"]].get("ok") or
           (by[c["case_id"]]["init_hash"], by[c["case_id"]]["init_hash_full"])
           != (c["init_hash"], c["init_hash_full"]) or
           by[c["case_id"]]["done"] or by[c["case_id"]]["success"]]
    man = dict(kind="k39_bank", space=SPACE, tasks=list(TASKS),
               n_per_task=N_BANK, files=files, per=per,
               replay=dict(n=len(jobs), bad=len(bad), examples=bad[:10]),
               versions=p33.versions())
    man["bank_id"] = h256(json.dumps(dict(files=files), sort_keys=True))
    if bad:
        raise SystemExit(f"холодный повтор: расхождения {bad[:3]}")
    p33.save_json(BANK_MANIFEST, man)
    print(f"  банк space {SPACE}: {len(jobs)} сцен, повтор совпал; "
          f"bank_id {man['bank_id'][:12]}")
    return OK


def bank_scenes(n_per_task):
    import k33_bank as kb
    m = json.load(open(BANK_MANIFEST, encoding="utf-8"))
    bad = [p for p, s in m["files"].items() if sha256(p) != s]
    if bad or m["replay"]["bad"]:
        raise SystemExit("банк K-39 изменён или повтор не пройден")
    out = []
    for t in TASKS:
        cs, _m = kb.load_task(SPACE, t, BANK_DIR)
        for pos, c in enumerate(cs[:n_per_task]):
            out.append(dict(task=t, case_id=c["case_id"], pos=pos,
                            hash=[c["init_hash"], c["init_hash_full"]]))
    return m, out


# --- план скрининга -------------------------------------------------------------------------------
def policy_order(rank, seed, pols):
    k = (rank + seed) % len(pols)
    return pols[k:] + pols[:k]


def make_plan(run, profile, n_parts):
    import k36_protocol as pc
    import k39_cache as kc
    dp = kc.load_data_plan(run)
    fits = {}
    for s in SEEDS:
        f = pc.load_json(os.path.join(rep_dir(run), f"fit_s{s}.json"))
        if f.get("plan") != dp["sha256"] or f.get("seed") != s:
            raise SystemExit(f"fit s{s} ≠ data_plan")
        fits[str(s)] = dict(sha256=f["sha256"], model_code=f["model_code"])
    n = PROFILES[profile]
    bm, scenes = bank_scenes(n)
    for sc in scenes:
        sc["rank"] = sc["pos"]
    refresh = {sc["case_id"] for sc in scenes if sc["pos"] < max(1, n // 4)}
    parts = {str(k): [] for k in range(n_parts)}
    for i, sc in enumerate(sorted(scenes, key=lambda x: (x["pos"],
                                                         x["task"]))):
        parts[str(i % n_parts)].append(sc["case_id"])
    k36 = pc.load_prepare()
    plan = dict(kind="k39_screen_plan", run_id=run, protocol=PROTOCOL,
                profile=profile, data_plan_sha256=dp["sha256"],
                k36_prepare_sha1=k36["sha1"],
                code={f: sha256(os.path.join(HERE, f)) for f in CODE},
                fits=fits, bank=dict(bank_id=bm["bank_id"], files=bm["files"],
                                     space=SPACE, dir=BANK_DIR),
                scenes=scenes, refresh=sorted(refresh), parts=parts,
                policies=list(MAIN) + ["Refresh4"], seeds=list(SEEDS),
                env=dict(B=1, exec_seed=101, waiting_steps=10, max_steps=600,
                         prefix=4, horizon=8),
                caps=dict(part=None, preflight=8 * EP), threads=8,
                analysis=dict(BOOT, unit="scene", task_weights="equal",
                              rule=dict(fresh_signal="100*D(Fresh-Old) >= 3N "
                                        "and D_s >= 0",
                                        tree_dev="and D(Fresh-Flat) > 0 and "
                                        "D(Fresh-T1) > 0")))
    for k, ids in parts.items():
        n_ep = len(ids) * len(SEEDS) * len(MAIN) + sum(
            len(SEEDS) for c in ids if c in refresh)
        plan["parts"][k] = dict(scenes=ids, n_episodes=n_ep,
                                cap=(n_ep + 1) * EP)
    return plan


def plan_path(run):
    return os.path.join(screen_dir(run), "screen_plan.json")


def prepare_screen(a):
    import k36_protocol as pc
    import k37_confirm as kcf
    p = plan_path(a.run_id)
    new = make_plan(a.run_id, a.profile, a.parts)
    sh = kcf.canon_sha(new)
    shp = p.replace(".json", ".sha256")
    if os.path.exists(p) or os.path.exists(shp):
        old = json.load(open(p, encoding="utf-8"))
        if not (kcf.canon_sha(old) == open(shp).read().strip() == sh):
            raise SystemExit(f"{p} закреплён иначе")
        print(f"  план скрининга закреплён: {sh[:16]}")
        return OK
    pc.save_json(p, new)
    with open(shp, "w") as fh:
        fh.write(sh + "\n")
    n = sum(v["n_episodes"] for v in new["parts"].values())
    print(f"  screen_plan {p}: sha256 {sh[:16]}; профиль {a.profile}; сцен "
          f"{len(new['scenes'])}, эпизодов {n} (из них Refresh4 "
          f"{len(new['refresh']) * 2}), частей {a.parts}")
    return OK


def load_plan(run, verify=True):
    import k36_protocol as pc
    import k37_confirm as kcf
    p = json.load(open(plan_path(run), encoding="utf-8"))
    sh = kcf.canon_sha(p)
    if sh != open(plan_path(run).replace(".json", ".sha256")).read().strip():
        raise SystemExit("screen_plan ≠ sha256")
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
    p["sha256"] = sh
    return p


def part_jobs(plan, k):
    by = {sc["case_id"]: sc for sc in plan["scenes"]}
    out = []
    for s in SEEDS:
        for cid in plan["parts"][str(k)]["scenes"]:
            sc = by[cid]
            pols = policy_order(sc["rank"], s, list(MAIN))
            if cid in plan["refresh"]:
                pols = pols + ["Refresh4"]
            for pol in pols:
                out.append(dict(job=f"ep/{pol}/s{s}/t{sc['task']}/{cid}",
                                policy=pol, seed=s, scene=sc))
    return out


# --- головы во время исполнения ---------------------------------------------------------------
class Heads:
    """Признаки и решения лёгких голов на устройстве признаков."""

    def __init__(self, torch, models, C, dev):
        import k39_model as km
        self.torch, self.km, self.C, self.dev = torch, km, C, dev
        self.m = models
        vm, vs = models["v_stats"]
        self.vm = torch.as_tensor(vm, dtype=torch.float32, device=dev)
        self.vs = torch.as_tensor(vs, dtype=torch.float32, device=dev)
        self.sm, self.ss = models["s11_stats"]

    def sense(self, obs):
        """Сырое наблюдение -> (v [64], s11 [11]) тензоры на устройстве."""
        import k21_model as k21
        torch, km = self.torch, self.km
        rgb = np.stack([obs[k][0] for k in ("agentview_image",
                                            "robot0_eye_in_hand_image")])
        x = torch.as_tensor(km.prep_rgb(torch, rgb)[None], device=self.dev
                            ).float() / 255
        with torch.no_grad():
            v = (self.m["enc"](x)[0] - self.vm) / self.vs
        s8 = np.asarray(obs["state"][:, :8], np.float64) if \
            "s8" not in obs else obs["s8"]
        s11 = (k21.state11(s8) - self.sm) / self.ss
        return v, torch.as_tensor(s11[0], dtype=torch.float32,
                                  device=self.dev)

    def context(self, B, t, sense0):
        """x0 и нормированный хвост B — без генерации кандидатов."""
        torch, km = self.torch, self.km
        v0, s0 = sense0
        bf = torch.as_tensor(km.chunk_feat(B, self.C), device=self.dev)
        x0 = torch.cat([v0, s0, bf, torch.tensor([t / 600.0],
                                                 device=self.dev)])[None]
        b_arm = torch.as_tensor(km.norm_arm(B[4:8], self.C)[None],
                                dtype=torch.float32, device=self.dev)
        return dict(x0=x0, b_arm=b_arm, cands=None)

    def start(self, B, t, sense0):
        """Контекст и банк C1/C2 (кодек) из сохранённого o0 (дерево)."""
        torch, km = self.torch, self.km
        st = self.context(B, t, sense0)
        x0, b_arm = st["x0"], st["b_arm"]
        with torch.no_grad():
            alts = km.gen_alts(torch, self.m["gen2"], x0, b_arm, 2)
        cands = [B[4:8].astype(np.float32)]
        for arm, gl in alts:
            c = np.zeros((4, 7), np.float32)
            c[:, :6] = (arm[0].cpu().numpy() * self.C["half"] +
                        self.C["mid"]).astype(np.float32)
            c[:, 6] = np.where(gl[0].cpu().numpy() > 0, 1.0, -1.0)
            cands.append(c)
        return dict(x0=x0, b_arm=b_arm, cands=np.stack(cands))

    def select(self, st, sense):
        torch = self.torch
        v, s = sense
        xm = torch.cat([st["x0"], v[None], s[None]], 1)
        with torch.no_grad():
            sc = self.m["sel2"](xm)[0].cpu().numpy()
        return int(np.argmax(sc)), sc

    def flat(self, st, sense):
        torch, km = self.torch, self.km
        v, s = sense
        xm = torch.cat([st["x0"], v[None], s[None]], 1)
        with torch.no_grad():
            o = self.m["flat"](xm)
            arm, gl = km.tail_from(torch, o[:, :24].reshape(-1, 4, 6),
                                   o[:, 24:], st["b_arm"])
        c = np.zeros((4, 7), np.float32)
        c[:, :6] = (arm[0].cpu().numpy() * self.C["half"] +
                    self.C["mid"]).astype(np.float32)
        c[:, 6] = np.where(gl[0].cpu().numpy() > 0, 1.0, -1.0)
        return c


# --- эпизод 4+4 ------------------------------------------------------------------------------------
def chunk_start(heads, policy, B, t, obs, s8):
    """Начало чанка: дерево — признаки o0, контекст и банк; flat — признаки
    o0 и контекст; T1 / Refresh4 — ничего."""
    if policy in ("TreeOld", "TreeFresh"):
        sense0 = heads.sense(dict(obs, s8=s8))
        return dict(heads.start(B, t, sense0), sense0=sense0)
    if policy == "FlatFresh":
        sense0 = heads.sense(dict(obs, s8=s8))
        return dict(heads.context(B, t, sense0), sense0=sense0)
    return None


def chunk_mid(heads, policy, st, B, obs_mid, io, arm, desc, task, t):
    """Решение в o4 -> (хвост [4,7], выбор, оценки, новый прогноз)."""
    if policy == "TreeOld":
        ch, sc = heads.select(st, st["sense0"])
        return st["cands"][ch], ch, sc, None
    if policy == "TreeFresh":
        ch, sc = heads.select(st, heads.sense(dict(
            obs_mid, s8=io.inp.state8(obs_mid))))
        return st["cands"][ch], ch, sc, None
    if policy == "FlatFresh":
        return heads.flat(st, heads.sense(dict(
            obs_mid, s8=io.inp.state8(obs_mid)))), 0, None, None
    if policy == "Refresh4":
        saved = arm.last
        call = arm.call
        b2, p2, s82 = io.prep(obs_mid, desc, task)
        r2, _r = arm.decide(b2, p2, s82, t + 4)
        arm.last, arm.call = saved, call
        r2 = np.asarray(r2, np.float32)
        return r2[:4], 0, None, r2
    return B[4:8], 0, None, None


def run_episode(sim, io, arm, heads, policy, task, reset_id):
    """Цикл с решением после четырёх реальных шагов. arm — k32_policy.Arm
    (детерминированный T); heads — Heads или None для T1/Refresh4."""
    import k18_collector as k18c
    import k31_collect as kcol
    t0 = time.time()
    io.seed(kcol.ENV["seed"])
    obs = sim.reset(task, [reset_id])
    desc = io.task_desc(task)
    arm.begin(task, reset_id)
    reward, succ_step, done_step, steps = 0.0, -1, -1, 0
    dummy = np.array([kcol.DUMMY])
    wait_done = False
    for _ in range(kcol.ENV["waiting_steps"]):
        obs, r_, d_, _s = sim.step(dummy)
        reward = min(max(reward + float(r_[0]), 0.0), 1.0)
        wait_done |= bool(d_[0])
    if wait_done or reward >= 1.0:
        raise SystemExit("среда завершилась во время ожидания")
    ih, ihf = k18c.init_hashes(obs, 1)
    acts, calls = [], []
    heavy = dict(n=0)
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
    try:
        while done_step < 0 and steps < kcol.ENV["max_steps"]:
            t = steps
            call = t // 8 + 1
            arm.call = call - 1
            batch, po, s8 = io.prep(obs, desc, task)
            B, rec = arm.decide(batch, po, s8, t)
            B = np.asarray(B, np.float32)
            st = chunk_start(heads, policy, B, t, obs, s8)
            prefix = io.scale(B[None])[0][:4]
            k = 0
            for j in range(4):
                if done_step >= 0 or steps >= kcol.ENV["max_steps"]:
                    break
                step(prefix[j])
                k += 1
            mid = k == 4 and done_step < 0 and steps < kcol.ENV["max_steps"]
            final = B.copy()
            choice, scores, refresh_a = 0, None, None
            if mid:
                final[4:8], choice, scores, refresh_a = chunk_mid(
                    heads, policy, st, B, obs, io, arm, desc, task, t)
                tail = io.scale(final[None])[0][4:8]
                for j in range(4):
                    if done_step >= 0 or steps >= kcol.ENV["max_steps"]:
                        break
                    step(tail[j])
                    k += 1
            arm.executed(k)
            calls.append(dict(call=call, k=k, a_codec=B, rec=rec,
                              final=final, mid=bool(mid), choice=int(choice),
                              scores=None if scores is None else
                              np.asarray(scores, np.float32),
                              cands=None if st is None else st.get("cands"),
                              refresh=refresh_a))
    finally:
        arm.forward = fwd0
    own = done_step + 1 if done_step >= 0 else steps
    A = np.stack(acts) if acts else np.zeros((0, 7), np.float32)
    return dict(task=int(task), init=int(reset_id),
                success=bool(reward >= 1.0), success_step=int(succ_step),
                done_step=int(done_step), own_steps=int(own),
                steps=int(steps), physical=int(10 + steps), init_hash=ih[0],
                init_hash_full=ihf[0], actions=A,
                actions_sha1=kcol.actions_sha(A), calls=calls,
                heavy=int(heavy["n"]), seconds=time.time() - t0)


def save_ep(path, job, ep, plan_sha):
    import k32_harness as kh32
    n = len(ep["calls"])
    calls = [dict(c, rec=dict(c["rec"], a_codec=c["a_codec"]))
             for c in ep["calls"]]
    J = kh32.pack_journal([(0, calls)], 6)
    K = dict(final=np.stack([c["final"] for c in ep["calls"]]) if n else
             np.zeros((0, 8, 7), np.float32),
             mid=np.array([c["mid"] for c in ep["calls"]], bool),
             choice=np.array([c["choice"] for c in ep["calls"]], np.int64),
             scores=np.stack([c["scores"] if c["scores"] is not None else
                              np.full(3, np.nan, np.float32)
                              for c in ep["calls"]]) if n else
             np.zeros((0, 3), np.float32),
             cands=np.stack([c["cands"] if c["cands"] is not None else
                             np.zeros((3, 4, 7), np.float32)
                             for c in ep["calls"]]) if n else
             np.zeros((0, 3, 4, 7), np.float32),
             refresh=np.stack([c["refresh"] if c["refresh"] is not None else
                               np.zeros((8, 7), np.float32)
                               for c in ep["calls"]]) if n else
             np.zeros((0, 8, 7), np.float32))
    meta = dict(job=job, plan=plan_sha, phys=int(ep["physical"]),
                heavy=int(ep["heavy"]), seconds=float(ep["seconds"]),
                summary={k: ep[k] for k in ("success", "success_step",
                                            "done_step", "own_steps",
                                            "steps", "init_hash",
                                            "init_hash_full",
                                            "actions_sha1")})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as fh:
        np.savez_compressed(fh, meta=np.array(json.dumps(meta)),
                            actions=ep["actions"].astype(np.float32),
                            **{f"j_{k}": v for k, v in J.items()},
                            **{f"k_{k}": v for k, v in K.items()})
    os.replace(path + ".tmp", path)
    return meta


def ep_problems(z, meta, policy, C, want_hash=None):
    """Проверка записанного эпизода 4+4."""
    import k29_policy as kp
    import k31_collect as kcol
    import k32_harness as kh32
    p = []
    A = z["actions"]
    s = meta["summary"]
    if kcol.actions_sha(A) != s["actions_sha1"]:
        return ["действия ≠ SHA"]
    if want_hash is not None and (s["init_hash"], s["init_hash_full"]) != \
            tuple(want_hash):
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
        q, "T")[:3] if "исполненное" not in x]
    for i in range(n):
        B, F = J["a_codec"][i], K["final"][i]
        if not np.array_equal(F[:4], B[:4]):
            p.append(f"вызов {i}: префикс ≠ B[:4]")
        if not bool(K["mid"][i]):
            if not np.array_equal(F, B):
                p.append(f"вызов {i}: без середины чанк ≠ B")
            continue
        tail = F[4:8]
        if policy == "T1" and not np.array_equal(tail, B[4:8]):
            p.append(f"вызов {i}: T1 хвост ≠ B")
        if policy in ("TreeOld", "TreeFresh"):
            sc = K["scores"][i]
            ch = int(K["choice"][i])
            if not np.isfinite(sc).all() or ch != int(np.argmax(sc)):
                p.append(f"вызов {i}: выбор ≠ argmax оценок")
            if not np.array_equal(K["cands"][i][0], B[4:8]):
                p.append(f"вызов {i}: кандидат B ≠ B[4:8]")
            if not np.array_equal(tail, K["cands"][i][ch]):
                p.append(f"вызов {i}: хвост ≠ выбранному кандидату")
        if policy == "Refresh4" and not np.array_equal(tail,
                                                       K["refresh"][i][:4]):
            p.append(f"вызов {i}: хвост ≠ новому прогнозу")
        if ((tail[:, :6] < C["L"] - 1e-4) | (tail[:, :6] > C["U"] + 1e-4)
                ).any() or not np.isin(tail[:, 6], (-1.0, 1.0)).all():
            p.append(f"вызов {i}: хвост вне пределов/схват не ±1")
    ex = np.concatenate([kp.scale_exec(K["final"][i][None], q)[0]
                         [:int(J["k"][i])] for i in range(n)])
    if not np.array_equal(ex.astype(np.float32), np.asarray(A, np.float32)):
        p.append("исполненное ≠ io.scale(итоговых чанков)")
    n_mid = int(K["mid"].sum())
    want_heavy = n + (n_mid if policy == "Refresh4" else 0)
    if meta["heavy"] != want_heavy:
        p.append(f"тяжёлых вызовов {meta['heavy']} ≠ {want_heavy}")
    if meta["phys"] != 10 + len(A) or s["steps"] != len(A):
        p.append("расход ≠ шагам")
    if not np.array_equal(K["mid"], np.asarray(J["k"]) > 4):
        p.append("mid ≠ (k > 4)")
    ss, ds, st_ = s["success_step"], s["done_step"], s["steps"]
    if s["success"]:
        if not (0 <= ss < st_ and (ds < 0 or ss <= ds)):
            p.append("success_step несогласован с успехом")
    elif ss != -1:
        p.append("success_step при неуспехе")
    if ds >= 0:
        if ds + 1 != st_ or s["own_steps"] != st_:
            p.append("done_step/steps несогласованы")
    elif st_ != 600 or s["own_steps"] != st_:
        p.append("без terminal эпизод не дошёл до лимита")
    return p


# --- исполнение очереди ----------------------------------------------------------------------------
def shard_path(run, job):
    return os.path.join("data/k39", run, "screen",
                        job.replace("/", "__") + ".npz")


def shard_meta(path):
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta"]))


def check_shard(run, j, plan, C, tag, after=None, done=None):
    """-> (meta, problems): job, план, расход, полная проверка эпизода,
    дополнительная сверка; done — запись журнала (если есть)."""
    path = shard_path(run, j["job"])
    m = shard_meta(path)
    if m is None:
        return None, ["нет шарда"]
    p = []
    if m.get("job") != j["job"] or m.get("plan") != plan["sha256"]:
        p.append("шард чужого задания или плана")
    if type(m.get("phys")) is not int or not 0 < m["phys"] <= EP:
        p.append(f"расход {m.get('phys')} вне 1…{EP}")
    if m.get("problems"):
        p.append(f"записанные проблемы {m['problems'][:2]}")
    if p:
        return m, p
    with np.load(path, allow_pickle=False) as z:
        zz = dict(z)
    p += ep_problems(zz, m, j["policy"], C, j["scene"].get("hash"))
    if after:
        p += after(j, zz, m)
    if done is not None and done != dict(scope=tag, actual=m["phys"]):
        p.append(f"журнал {done} ≠ записи")
    return m, p


def load_heads(torch, run, plan, s, C, dev):
    import k39_model as km
    m = km.load_models(torch, run, s, dev, plan["data_plan_sha256"],
                       plan["fits"][str(s)]["sha256"])
    return Heads(torch, m, C, dev)


def admitted(run, uuid, torch, plan_sha):
    import k36_protocol as pc
    import k38_grip_run as kgr
    p = os.path.join(screen_dir(run), "preflight",
                     f"{socket.gethostname()}__{uuid}.json")
    if not os.path.exists(p):
        raise SystemExit(f"карта {uuid} не прошла preflight K-39")
    r = pc.load_json(p)
    if not r.get("admitted") or r.get("env") != kgr.env_record(torch) or \
            r.get("plan") != plan_sha:
        raise SystemExit("preflight: не допущена, другой план или среда")
    return r


def run_queue(a, jobs, scope_dir, caps, tag, validate_plan, sim_kind,
              plan, after=None):
    """Общий цикл: блокировки, ресурсы, журнал, транзакции по шардам."""
    import torch
    import k31_collect as kcol
    import k36_bank as kb36
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    import k38_grip_run as kgr
    tp = kgr.threads_problems(torch)
    if tp:
        raise SystemExit("потоки: " + "; ".join(tp))
    k36 = pc.load_prepare()
    C = kh.consts(k36)
    idx = kcf.dev_index(a.device)
    g = kcf.gpu_query(idx)
    locks = [kcf.Lock(os.path.join(scope_dir, ".lock"), tag, kcf.Busy)]
    try:
        locks.append(kcf.Lock(os.path.join(kcf.lock_dir(), g["uuid"] +
                                           ".lock"), f"карта {g['uuid']}",
                              kcf.Busy))
        kcf.check_gpu(idx, kcf.RES["min_free_gb"])
        kcf.check_disk(scope_dir)
        L = kcol.Ledger(os.path.join(scope_dir, "ledger.json"), caps)
        pd = L.d["pending"]
        if pd:
            byjob = {j["job"]: j for j in jobs}
            if pd["id"] not in byjob or pd["scope"] != tag or \
                    pd["amount"] != EP:
                raise SystemExit(f"открытый резерв {pd['id']} не из этой "
                                 f"очереди")
            m, pr = check_shard(a.run_id, byjob[pd["id"]], plan, C, tag,
                                after)
            if m is not None:
                if pr:
                    raise SystemExit(f"записанный эпизод резерва {pd['id']} "
                                     f"не прошёл проверку: {pr[:3]}")
                L.settle(pd["id"], m["phys"])
        sim = kb36.BankSim("bank", sim_kind[0], bank_dir=sim_kind[1])
        dev = torch.device(a.device)
        res = []
        try:
            kcf.bind_device(torch, idx, g)
            torch.cuda.reset_peak_memory_stats(idx)
            for s in SEEDS:
                js = [j for j in jobs if j["seed"] == s]
                if not js:
                    continue
                kcf.check_gpu(idx, kcf.RES["min_free_gb"])
                v = kh.load_source(torch, s, a.device)
                arm = kh.make_arm(torch, k36, f"T32_s{s}", v)
                heads = load_heads(torch, a.run_id, plan, s, C, dev) if \
                    any(j["policy"] in MAIN[1:] for j in js) else None
                for j in js:
                    path = shard_path(a.run_id, j["job"])
                    if shard_meta(path) is not None:
                        m, pr = check_shard(a.run_id, j, plan, C, tag, after,
                                            L.d["done"].get(j["job"]))
                        if pr:
                            raise SystemExit(f"{path}: {pr[:3]}")
                        res.append((j, m))
                        continue
                    if L.is_done(j["job"]):
                        raise SystemExit(f"{j['job']}: учтён, шарда нет")
                    kcf.check_disk(scope_dir)
                    L.reserve(tag, j["job"], amount=EP)
                    arm.episode = kh.episode_number(j["job"])
                    ep = run_episode(sim, v["io"], arm, heads, j["policy"],
                                     j["scene"]["task"], j["scene"]["pos"])
                    meta = save_ep(path, j["job"], ep, plan["sha256"])
                    _m, prob = check_shard(a.run_id, j, plan, C, tag, after)
                    if prob:
                        meta["problems"] = prob
                        with np.load(path, allow_pickle=False) as z:
                            zz = dict(z)
                        zz["meta"] = np.array(json.dumps(meta))
                        with open(path + ".tmp", "wb") as fh:
                            np.savez_compressed(fh, **zz)
                        os.replace(path + ".tmp", path)
                        print(f"  {j['job']}: ПРОБЛЕМЫ {prob[:3]} — резерв "
                              f"открыт, остановка", flush=True)
                        return TECH, res
                    L.commit(j["job"], meta["phys"])
                    res.append((j, meta))
                    n = sum(L.is_done(x["job"]) for x in jobs)
                    print(f"  {j['job']}: успех {meta['summary']['success']}, "
                          f"шагов {meta['summary']['steps']}, тяжёлых "
                          f"{meta['heavy']}; {n}/{len(jobs)}", flush=True)
                arm = v = heads = None
                kh.free(torch)
        except torch.cuda.OutOfMemoryError as e:
            raise kcf.Busy(f"CUDA OOM: {e}")
        finally:
            kcf.close_sim(sim)
        return OK, res
    finally:
        for lk in reversed(locks):
            lk.release()


def preflight(a):
    """Принудительный T1 исполнителя 4+4 = архивному T1 c2 на двух
    эпизодах micro (s0 t0, s1 t8); затем TreeFresh и FlatFresh на тех же
    сценах — решения первой развилки (эталон или сверка с эталоном)."""
    import torch
    import k36_protocol as pc
    import k39_cache as kc
    import k38_grip_run as kgr
    import k37_confirm as kcf
    plan = load_plan(a.run_id)
    dp = kc.load_data_plan(a.run_id)
    eps = {(e["seed"], e["task"]): e for e in dp["episodes"]
           if e["split"] == "micro"}
    picks = [eps[(0, 0)], eps[(1, 8)]]
    jobs = []
    for e in picks:  # имена дополняются хостом/UUID ниже
        sc = dict(task=e["task"], case_id=e["case_id"], pos=e["pos"],
                  hash=e["hash"])
        for pol in ("T1", "TreeFresh", "FlatFresh"):
            jobs.append(dict(job=f"pf/{pol}/s{e['seed']}/t{e['task']}/"
                             f"{e['case_id']}", policy=pol, seed=e["seed"],
                             scene=sc, eid=e["eid"]))
    idx = kcf.dev_index(a.device)
    g = kcf.gpu_query(idx)
    tagc = f"{socket.gethostname()}__{g['uuid']}"
    for j in jobs:                          # шарды допуска — свои у карты
        j["job"] = j["job"].replace("pf/", f"pf/{tagc}/", 1)
    d = os.path.join(screen_dir(a.run_id), "preflight", tagc)
    os.makedirs(d, exist_ok=True)
    refs = {e["eid"]: kc.load_input(a.run_id, e) for e in picks}

    def after(j, z, meta):
        if j["policy"] != "T1":
            return []
        rec, rz = refs[j["eid"]]
        p = []
        if not np.array_equal(np.asarray(z["actions"], np.float32),
                              np.asarray(rz["actions"], np.float32)):
            p.append("T1 4+4: действия ≠ архиву c2")
        sa = rec["summary"]
        if (meta["summary"]["success"], meta["summary"]["done_step"],
                meta["summary"]["steps"]) != (sa["success"], sa["done_step"],
                                              sa["steps"]):
            p.append("T1 4+4: исход ≠ архиву c2")
        return p
    bank = (2, dp["bank"]["dir"])
    rc, res = run_queue(a, jobs, d, dict(preflight=8 * EP), "preflight",
                        None, bank, plan, after)
    first = {}
    for j, m in res:
        with np.load(shard_path(a.run_id, j["job"]), allow_pickle=False) as z:
            K = {k[2:]: z[k] for k in z.files if k.startswith("k_")}
            i = int(np.flatnonzero(K["mid"])[0]) if K["mid"].any() else -1
            key = j["job"].replace(f"pf/{tagc}/", "pf/", 1)
            first[key] = dict(mid=bool(i >= 0),
                success=m["summary"]["success"], call=i,
                choice=int(K["choice"][i]) if i >= 0 else -1,
                tail=K["final"][i][4:8].tolist() if i >= 0 else None)
    prob = [] if rc == OK else ["очередь допуска не завершена"]
    refp = os.path.join(screen_dir(a.run_id), "preflight", "reference.json")
    if a.reference:
        if rc == OK:
            pc.save_json(refp, dict(first=first, uuid=g["uuid"],
                                    host=socket.gethostname(),
                                    plan=plan["sha256"]))
    else:
        if not os.path.exists(refp):
            raise SystemExit("нет эталона: preflight --reference")
        refd = pc.load_json(refp)
        if refd.get("plan") != plan["sha256"]:
            raise SystemExit("эталон допуска от другого плана")
        ref = refd["first"]
        for k_, v in first.items():
            r = ref.get(k_)
            if r is None or r["mid"] != v["mid"] or r["call"] != v["call"] \
                    or r["choice"] != v["choice"] or \
                    r["success"] != v["success"] or (
                        v["tail"] is not None and np.abs(
                            np.subtract(r["tail"], v["tail"]))[:, :6].max()
                        > 1e-5) or (v["tail"] is not None and np.subtract(
                            r["tail"], v["tail"])[:, 6].any()):
                prob.append(f"{k_}: решение первой развилки ≠ эталону")
    rep = dict(admitted=not prob, problems=prob, first=first,
               env=kgr.env_record(torch), uuid=g["uuid"],
               peak_gb=float(torch.cuda.max_memory_reserved(idx) / 2 ** 30),
               plan=plan["sha256"])
    pc.save_json(d + ".json", rep)
    print(f"  preflight K-39 {a.device}: " + ("ДОПУЩЕНА" if not prob else
                                              "НЕ допущена: " +
                                              "; ".join(prob[:3])))
    return OK if not prob else TECH


def collect(a):
    import k36_protocol as pc
    import k37_confirm as kcf
    import torch
    plan = load_plan(a.run_id)
    k = int(a.part)
    pd = os.path.join(screen_dir(a.run_id), f"part-{k:02d}")
    asg_p = os.path.join(pd, "assignment.json")
    if not os.path.exists(asg_p):
        raise SystemExit(f"part {k}: нет назначения")
    asg = pc.load_json(asg_p)
    g = kcf.gpu_query(kcf.dev_index(a.device))
    if asg.get("plan_sha256") != plan["sha256"] or asg.get("uuid") != \
            g["uuid"] or asg.get("host") != socket.gethostname() or \
            asg.get("state") != "assigned":
        raise SystemExit(f"part {k}: назначение не этой карте/плану")
    admitted(a.run_id, g["uuid"], torch, plan["sha256"])
    jobs = part_jobs(plan, k)
    rc, res = run_queue(a, jobs, pd, dict(screen=plan["parts"][str(k)]["cap"]),
                        "screen", None, (SPACE, BANK_DIR), plan)
    print(f"  part {k}: " + ("готова" if rc == OK else "остановлена"))
    return rc


def assign(a):
    import fcntl
    import k36_protocol as pc
    plan = load_plan(a.run_id, verify=False)
    path = os.path.join(screen_dir(a.run_id), "assignments.json")
    lk = os.path.join(screen_dir(a.run_id), ".assign.lock")
    with open(lk, "a+") as fh:
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
        rec = dict(part=k, host=a.host if a.state == "assigned" else
                   last["host"], uuid=a.uuid if a.state == "assigned" else
                   last["uuid"], state=a.state, plan_sha256=plan["sha256"],
                   n=1 + sum(h["part"] == k for h in d["history"]),
                   time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        d["history"].append(rec)
        pc.save_json(path, d)
        pc.save_json(os.path.join(screen_dir(a.run_id), f"part-{k:02d}",
                                  "assignment.json"), rec)
        fcntl.flock(fh, fcntl.LOCK_UN)
    print(f"  part {k}: {a.state} -> {rec['host']} {rec['uuid']}")
    return OK


def status(a):
    plan = load_plan(a.run_id, verify=False)
    tot = 0
    for k, v in plan["parts"].items():
        lp = os.path.join(screen_dir(a.run_id), f"part-{int(k):02d}",
                          "ledger.json")
        if not os.path.exists(lp):
            print(f"  part {k}: не начата (0/{v['n_episodes']})")
            continue
        d = json.load(open(lp))
        tot += len(d["done"])
        print(f"  part {k}: {len(d['done'])}/{v['n_episodes']}; расход "
              f"{d['spent']}; резерв {(d['pending'] or {}).get('id', '—')}; "
              f"аварий {len(d['crashes'])}")
    print(f"  всего {tot}/{sum(v['n_episodes'] for v in plan['parts'].values())}")
    return OK


def build_merge(run, plan):
    import k36_harness as kh
    import k36_protocol as pc
    C = kh.consts(pc.load_prepare())
    prob, rows = [], []
    for k, v in plan["parts"].items():
        pd = os.path.join(screen_dir(run), f"part-{int(k):02d}")
        lp = os.path.join(pd, "ledger.json")
        if not os.path.exists(lp):
            prob.append(f"part {k}: нет журнала")
            continue
        led = pc.load_json(lp)
        jobs = part_jobs(plan, int(k))
        if led["pending"] or set(led["done"]) != {j["job"] for j in jobs}:
            prob.append(f"part {k}: журнал ≠ плану")
        tot = 0
        for j in jobs:
            m, pr = check_shard(run, j, plan, C, "screen", None,
                                led["done"].get(j["job"]))
            prob += [f"{j['job']}: {x}" for x in pr]
            if m is None or pr:
                continue
            with np.load(shard_path(run, j["job"]), allow_pickle=False) as z:
                zz = dict(z)
            tot += m["phys"]
            K = {kk[2:]: zz[kk] for kk in zz if kk.startswith("k_")}
            rows.append(dict(job=j["job"], policy=j["policy"], seed=j["seed"],
                             task=j["scene"]["task"],
                             case_id=j["scene"]["case_id"],
                             success=m["summary"]["success"],
                             steps=m["summary"]["steps"], heavy=m["heavy"],
                             mids=int(K["mid"].sum()),
                             alt=int((K["choice"][K["mid"]] != 0).sum())
                             if K["mid"].any() else 0,
                             seconds=m["seconds"]))
        crash = sum(int(c["amount"]) for c in led["crashes"])
        if led["spent"].get("screen", 0) != tot + crash:
            prob.append(f"part {k}: расход ≠ сумме")
    if len({r["job"] for r in rows}) != len(rows):
        prob.append("дубли")
    return prob, rows


def merge(a):
    import k36_protocol as pc
    plan = load_plan(a.run_id)
    prob, rows = build_merge(a.run_id, plan)
    pc.save_json(os.path.join(screen_dir(a.run_id), "merged.json"),
                 dict(plan=plan["sha256"], ok=not prob, problems=prob[:100],
                      rows=rows if not prob else []))
    print("  merge: " + (f"{len(rows)} эпизодов, проверка пройдена"
                         if not prob else "ОТКАЗ: " + "; ".join(prob[:4])))
    return OK if not prob else TECH


def contrast(rows, a_, b_, tasks, n_boot=BOOT["n"], seed=BOOT["seed"]):
    tab = {}
    for r in rows:
        tab.setdefault((r["task"], r["case_id"], r["seed"]), {})[
            r["policy"]] = int(r["success"])
    keys = [k for k, v in tab.items() if a_ in v and b_ in v and
            k[0] in tasks]
    if not keys:
        return None
    dif = {k: tab[k][a_] - tab[k][b_] for k in keys}
    W = sum(v == 1 for v in dif.values())
    L_ = sum(v == -1 for v in dif.values())
    scenes = sorted({(k[0], k[1]) for k in keys})
    dsc = {sc: np.mean([dif[k] for k in keys if (k[0], k[1]) == sc])
           for sc in scenes}
    ts = sorted({t for t, _c in scenes})
    rng = np.random.default_rng(seed)
    per = {t: np.array([dsc[sc] for sc in scenes if sc[0] == t]) for t in ts}
    est = float(np.mean([per[t].mean() for t in ts]))
    bs = np.mean([per[t][rng.integers(0, len(per[t]),
                                      (n_boot, len(per[t])))].mean(1)
                  for t in ts], 0)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return dict(N=len(keys), W=int(W), L=int(L_), D=int(W - L_),
                delta=est, lo=float(lo), hi=float(hi),
                per_seed={str(s): int(sum(v for k, v in dif.items()
                                          if k[2] == s)) for s in SEEDS},
                sr_a=float(np.mean([tab[k][a_] for k in keys])),
                sr_b=float(np.mean([tab[k][b_] for k in keys])))


def analyze(a):
    import k36_protocol as pc
    plan = load_plan(a.run_id)
    m = pc.load_json(os.path.join(screen_dir(a.run_id), "merged.json"))
    if not m.get("ok") or m.get("plan") != plan["sha256"]:
        raise SystemExit("merge не пройден")
    prob, rows = build_merge(a.run_id, plan)
    if prob:
        raise SystemExit("проверка при анализе: " + "; ".join(prob[:3]))
    out = {}
    for nm, tasks in (("all", TASKS), ("train_tasks", TRAIN_TASKS),
                      ("transfer_tasks", TRANSFER_TASKS)):
        out[nm] = {f"{x}-{y}": contrast(rows, x, y, tasks)
                   for x, y in (("TreeFresh", "TreeOld"),
                                ("TreeFresh", "FlatFresh"),
                                ("TreeFresh", "T1"), ("Refresh4", "T1"),
                                ("FlatFresh", "T1"), ("TreeOld", "T1"))}
    fo = out["all"]["TreeFresh-TreeOld"]
    fresh = bool(fo and 100 * fo["D"] >= 3 * fo["N"] and
                 all(v >= 0 for v in fo["per_seed"].values()))
    tree = fresh and out["all"]["TreeFresh-FlatFresh"]["D"] > 0 and \
        out["all"]["TreeFresh-T1"]["D"] > 0
    sr = {p_: float(np.mean([r["success"] for r in rows
                             if r["policy"] == p_])) for p_ in
          plan["policies"]}
    alt = {p_: (sum(r["alt"] for r in rows if r["policy"] == p_) /
                max(1, sum(r["mids"] for r in rows if r["policy"] == p_)))
           for p_ in ("TreeOld", "TreeFresh")}
    status_ = "tree_dev_candidate" if tree else ("fresh_signal" if fresh
                                                 else "no_signal")
    pc.save_json(os.path.join(screen_dir(a.run_id), "analysis.json"),
                 dict(plan=plan["sha256"], sr=sr, contrasts=out,
                      alt_share=alt, status=status_, boot=BOOT))
    for k_, v in out["all"].items():
        if v:
            print(f"  {k_}: W {v['W']} L {v['L']} D {v['D']:+d} (N "
                  f"{v['N']}); Δ {100 * v['delta']:+.1f} п.п. 95% "
                  f"[{100 * v['lo']:+.1f}; {100 * v['hi']:+.1f}]; по сидам "
                  f"{v['per_seed']}")
    print("  SR: " + ", ".join(f"{k} {100 * v:.1f}" for k, v in sr.items()) +
          f"; доля альтернатив {alt}")
    print(f"  K-39 screen: {status_}")
    return OK


# --- задержка ---------------------------------------------------------------------------------------
def latency(a):
    """L0: сырое o0 -> физический префикс (io.prep + T32 + признаки o0 и
    банк для Tree/Flat + масштаб); L4: сырое o4 -> физический хвост. 15 пар
    на сид из кеша (raw), 10 прогревов, 10 повторов каждой пары."""
    import torch
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    import k38_grip_run as kgr
    import k39_cache as kc
    plan = load_plan(a.run_id)
    dp = kc.load_data_plan(a.run_id)
    idx = kcf.dev_index(a.device)
    g = kcf.gpu_query(idx)
    lk = kcf.Lock(os.path.join(kcf.lock_dir(), g["uuid"] + ".lock"),
                  f"карта {g['uuid']}", kcf.Busy)
    try:
        if g["used"] > 2048:
            raise kcf.Busy(f"занято {g['used']} МиБ > 2048")
        k36 = pc.load_prepare()
        C = kh.consts(k36)
        dev = torch.device(a.device)
        kcf.bind_device(torch, idx, g)
        by = {e["eid"]: e for e in dp["episodes"]}
        res, qrec = {}, []
        for s in SEEDS:
            pairs = [r for r in dp["raw_pairs"] if by[r["eid"]]["seed"] == s]
            if len(pairs) != 15:
                raise SystemExit(f"s{s}: сырых пар {len(pairs)} ≠ 15")
            raws = []
            for r in pairs:
                with np.load(kc.shard_path(a.run_id, f"raw/{r['eid']}/"
                                           f"{r['t']}"), allow_pickle=False) \
                        as z:
                    o0 = {k[3:]: z[k] for k in z.files if k.startswith("o0_")}
                    o4 = {k[3:]: z[k] for k in z.files if k.startswith("o4_")}
                need = {"state", "agentview_image", "robot0_eye_in_hand_image"}
                if set(o0) != need or set(o4) != need:
                    raise SystemExit(f"сырая пара {r['eid']}/{r['t']} неполна")
                qrec.append([r["eid"], r["t"], sha256(kc.shard_path(
                    a.run_id, f"raw/{r['eid']}/{r['t']}"))])
                raws.append((by[r["eid"]]["task"], r["t"], o0, o4))
            v = kh.load_source(torch, s, a.device)
            arm = kh.make_arm(torch, k36, f"T32_s{s}", v)
            heads = load_heads(torch, a.run_id, plan, s, C, dev)
            io = v["io"]

            descs = {tk: io.task_desc(tk) for tk in {r[0] for r in raws}}

            def one(pol, task, t, o0, o4):
                arm.begin(task, 0)                 # вне таймера, как в эпизоде
                arm.call = t // 8
                desc = descs[task]
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                batch, po, s8 = io.prep(o0, desc, task)
                B, _r = arm.decide(batch, po, s8, t)
                B = np.asarray(B, np.float32)
                st = chunk_start(heads, pol, B, t, o0, s8)
                io.scale(B[None])
                torch.cuda.synchronize(dev)
                t1 = time.perf_counter()
                final = B.copy()
                final[4:8] = chunk_mid(heads, pol, st, B, o4, io, arm, desc,
                                       task, t)[0]
                io.scale(final[None])
                torch.cuda.synchronize(dev)
                t2 = time.perf_counter()
                return (t1 - t0) * 1e3, (t2 - t1) * 1e3
            pols = list(MAIN) + ["Refresh4"]
            for j in range(10):
                for pol in pols:
                    one(pol, *raws[j % len(raws)])
            meas = {p_: [] for p_ in pols}
            order = np.random.default_rng([39002, s]).permutation(
                len(pols) * len(raws) * 10)
            for o in order:
                pol = pols[o % len(pols)]
                rr = raws[(o // len(pols)) % len(raws)]
                meas[pol].append(one(pol, *rr))
            res[str(s)] = {p_: np.asarray(v_).tolist() for p_, v_ in
                           meas.items()}
            arm = v = heads = None
            kh.free(torch)
        summ = {}
        for p_ in list(MAIN) + ["Refresh4"]:
            x = np.concatenate([np.asarray(res[str(s)][p_]) for s in SEEDS])
            tot = x.sum(1)
            if not np.isfinite(x).all() or len(x) != 300:
                raise SystemExit(f"{p_}: неполный или нечисловой замер")
            summ[p_] = dict(n=int(len(x)),
                            L0=dict(p50=float(np.quantile(x[:, 0], 0.5)),
                                    p95=float(np.quantile(x[:, 0], 0.95))),
                            L4=dict(p50=float(np.quantile(x[:, 1], 0.5)),
                                    p95=float(np.quantile(x[:, 1], 0.95))),
                            sum=dict(p50=float(np.quantile(tot, 0.5)),
                                     p95=float(np.quantile(tot, 0.95))))
        pc.save_json(os.path.join(screen_dir(a.run_id), "latency.json"),
                     dict(plan=plan["sha256"], gpu=g, raw=res, summary=summ,
                          queue=qrec, queue_sha256=h256(json.dumps(qrec)),
                          env=kgr.env_record(torch),
                          loadavg=os.getloadavg()))
        for p_, v_ in summ.items():
            print(f"  {p_}: L0 p50 {v_['L0']['p50']:.1f} / p95 "
                  f"{v_['L0']['p95']:.1f}; L4 p50 {v_['L4']['p50']:.2f} / p95 "
                  f"{v_['L4']['p95']:.2f}; сумма p50 {v_['sum']['p50']:.1f} мс")
        return OK
    finally:
        lk.release()


# --- самопроверка ------------------------------------------------------------------------------------
def selftest():
    import torch
    import k21_model as km21
    import k25_model as k25m
    import k29_model as k29m
    import k32_model as km32
    import k32_ppo as kppo
    import k31_collect as kcol
    import k36_harness as kh
    import k39_model as km
    k29m.no_tf32(torch)
    torch.manual_seed(0)
    # порядок политик и очередь
    assert policy_order(1, 0, list(MAIN)) == list(MAIN[1:]) + [MAIN[0]]
    # игрушечный источник, среда с кадрами 224
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
    C = kh.consts(k36)
    src = dict(mod=mod, dec=dec, M=M, io=io, actors=dict(T=actor))
    # T1 4+4 = штатному коллектору побитово
    arm = kh.make_arm(torch, k36, "T32_s0", src)
    ref = kcol.run_episode(Sim(), io, arm, 0, 3)
    arm = kh.make_arm(torch, k36, "T32_s0", src)
    ep = run_episode(Sim(), io, arm, None, "T1", 0, 3)
    assert np.array_equal(ep["actions"], ref["actions"]) and \
        ep["success"] == ref["success"] and ep["heavy"] == len(ep["calls"])
    # головы (случайные веса) и все политики
    B_ = km.build(torch)
    models = dict(enc=B_["Enc"]().eval(), s11_stats=(np.zeros(11),
                                                     np.ones(11)),
                  v_stats=(np.zeros(64), np.ones(64)),
                  gen2=B_["mlp"](132, 96, 56).eval(),
                  sel2=B_["mlp"](207, 32, 3).eval(),
                  flat=B_["mlp"](207, 106, 28).eval())
    heads = Heads(torch, models, C, torch.device("cpu"))
    ncall = dict(gen=0)
    g0 = models["gen2"].forward

    def gcount(x):
        ncall["gen"] += 1
        return g0(x)
    models["gen2"].forward = gcount
    import copy
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for pol in list(MAIN) + ["Refresh4"]:
            arm = kh.make_arm(torch, k36, "T32_s0", src)
            ncall["gen"] = 0
            e = run_episode(Sim(), io, arm, heads, pol, 0, 3)
            # банк кандидатов строят только деревья
            assert (ncall["gen"] > 0) == (pol in ("TreeOld", "TreeFresh")), \
                (pol, ncall)
            path = os.path.join(td, f"{pol}.npz")
            meta = save_ep(path, pol, e, "P")
            with np.load(path, allow_pickle=False) as z:
                zz = dict(z)
            pr = ep_problems(zz, meta, pol, C)
            assert pr == [], (pol, pr)
            n_mid = int(zz["k_mid"].sum())
            assert meta["heavy"] == len(e["calls"]) + (n_mid if pol ==
                                                       "Refresh4" else 0)
            if pol in ("TreeOld", "TreeFresh"):
                bad = dict(zz)
                bad["k_final"] = zz["k_final"].copy()
                i = int(np.flatnonzero(zz["k_mid"])[0])
                bad["k_final"][i, 4, 0] += 0.01
                assert ep_problems(bad, meta, pol, C)
            if pol == "T1":
                assert np.array_equal(e["actions"], ref["actions"])
            # перевёрнутый успех и сброшенный mid — отказ
            m2 = copy.deepcopy(meta)
            m2["summary"]["success"] = not m2["summary"]["success"]
            assert ep_problems(zz, m2, pol, C)
            if zz["k_mid"].any():
                bad = dict(zz)
                bad["k_mid"] = zz["k_mid"].copy()
                bad["k_mid"][int(np.flatnonzero(zz["k_mid"])[0])] = False
                assert ep_problems(bad, meta, pol, C)
    # проверка шарда до settle: чужой план, расход, запись журнала
    g_ = globals()
    keep_sp = g_["shard_path"]
    with tempfile.TemporaryDirectory() as td:
        g_["shard_path"] = lambda run, job: os.path.join(
            td, job.replace("/", "__") + ".npz")
        try:
            arm = kh.make_arm(torch, k36, "T32_s0", src)
            e = run_episode(Sim(), io, arm, None, "T1", 0, 3)
            j = dict(job="ep/T1/s0/t0/x", policy="T1", seed=0,
                     scene=dict(task=0, case_id="x", pos=3, hash=None))
            meta = save_ep(shard_path("r", j["job"]), j["job"], e, "P")
            plan_ = dict(sha256="P")
            assert check_shard("r", j, plan_, C, "screen")[1] == []
            assert check_shard("r", j, dict(sha256="Q"), C, "screen")[1]
            assert check_shard("r", dict(j, job="ep/T1/s0/t0/y"), plan_, C,
                               "screen")[1]
            assert check_shard("r", j, plan_, C, "screen", None,
                               dict(scope="screen", actual=meta["phys"] + 1)
                               )[1]
            assert check_shard("r", j, plan_, C, "screen",
                               lambda *x: ["архив ≠"])[1] == ["архив ≠"]
        finally:
            g_["shard_path"] = keep_sp
    # TreeOld и TreeFresh при o4 = o0 дают один выбор
    st = heads.start(ref["calls"][0]["a_codec"], 0, heads.sense(dict(
        Sim().reset(0, [3]), s8=np.zeros((1, 8)))))
    obs = Sim().reset(0, [3])
    s_ = heads.sense(dict(obs, s8=np.zeros((1, 8))))
    assert heads.select(st, s_)[0] == heads.select(st, s_)[0]
    # контраст и правило
    rows = []
    for t in TASKS:
        for c in range(4):
            for s in SEEDS:
                for pol, y in (("TreeFresh", 1), ("TreeOld", c > 0),
                               ("FlatFresh", c > 1), ("T1", c > 1)):
                    rows.append(dict(task=t, case_id=f"c{t}_{c}", seed=s,
                                     policy=pol, success=bool(y)))
    cr = contrast(rows, "TreeFresh", "TreeOld", TASKS, n_boot=500)
    assert cr["D"] == 12 and cr["L"] == 0 and cr["N"] == 48
    print("самопроверка k39_run пройдена: T1 4+4 = коллектору побитово; "
          "все пять политик проходят проверку записи; один тяжёлый вызов на "
          "чанк (Refresh4 — плюс один на середину); банк C1/C2 строят только "
          "деревья; порча хвоста, перевёрнутый успех, сброшенный mid — отказ; "
          "одинаковый вход — один выбор; W/L/D")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-39 раскатки")
    ap.add_argument("cmd", choices=("bank", "prepare-screen", "gpuinfo",
                                    "assign", "preflight", "collect",
                                    "status", "merge", "analyze", "latency"))
    ap.add_argument("--run-id", default="f2")
    ap.add_argument("--profile", choices=tuple(PROFILES), default="one")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host")
    ap.add_argument("--uuid")
    ap.add_argument("--state", default="assigned",
                    choices=("assigned", "released", "done"))
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import k37_confirm as kcf
    fn = {"bank": bank, "prepare-screen": prepare_screen,
          "gpuinfo": kcf.gpuinfo, "assign": assign, "preflight": preflight,
          "collect": collect, "status": status, "merge": merge,
          "analyze": analyze, "latency": latency}[a.cmd]
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
