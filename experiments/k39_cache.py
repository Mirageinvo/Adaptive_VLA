#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-39, этап 1: автономный кеш пар наблюдений o0 / o4 и меток учителя.

Команды:
  prepare-data --run-id f2 [--profile f1|f2] [--archive reports/k37c/runs/c2]
  micro --run-id f2 --device D      6 номинальных повторов и проверки развилок
  collect --run-id f2 --device D    360 номинальных эпизодов + сёстры
  status --run-id f2
Окружение: OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 (проверяется).

Выборка (prepare-data; без чтения исходов для ранжирования): задачи 0/6/8;
исключить 20 сцен скрининга g1; остаток по (sha256("k39-f1-scene|t|cid"),
cid): 48 train (первые 32 — gen, последние 16 — sel), 12 diag, 1 micro;
оба сида. Архивные T1 c2 проверяются (k38_grip_run.archive_ep) и
копируются в data/k39/<run>/inputs/<sha256>.{json,npz}.
Окна эпизода: j_elapsed = t, j_k ≥ 5, t + 8 ≤ 600; хронологический список
делится на min(8, n) близких групп (np.array_split), из группы — минимум
sha256("k39-f1-window|eid|t").
Строка (collect): BankSim space 2, io.seed(101), reset, 10 DUMMY, сверка
post-wait хешей; архивные физические действия по одному; o0 — до
actions[t], o4 — после actions[t+3]; actions[t:t+4] = io.scale(B[:4]);
учитель — отдельный k32_policy.Arm(kind=T) своего сида на o4, elapsed t+4,
episode = номер архивного эпизода, call = номер чанка; Y = новый прогноз
[8,7] (метка — первые 4). Полный повтор сверяется с архивом (исход,
steps, success_step, done_step). На первом окне эпизода (в micro — на
каждом) прямой T32 на o0 = j_a_codec побитово.
Сёстры: на каждой задаче фиксированные префиксы списков gen / sel / diag
(2 / 2 / 2 у f1, 16 / 16 / 6 у f2). Окно по хешу среди отобранных
с j_k = 8; ось x/y по хешу;
два повтора из reset с ±0.15·half на индексах 1, 2 первых четырёх
команд (клип по L, U; io.scale один раз); Y на своём o4.
Хранение: исходные кадры [2, H, W, 3] и проприоцепция в каждой точке,
хеши наблюдений, миниатюры 2×3×48×48 uint8, state8, B, Y, исполненный
физический префикс и маски. Сжатие без потерь, шард на эпизод, атомарная
публикация и журнал reports/k39/<run>/ledger.json.
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

PROTOCOL = "k39-cache-v2"
TASKS = (0, 6, 8)
SEEDS = (0, 1)
N_TRAIN, N_GEN, N_DIAG = 48, 32, 12
WIN_MAX = 8
MIN_K = 5
EP = 610
SIS_DELTA = 0.15
THREADS = 8
THUMB = 48
FORK_SCENES = {"f1": (2, 2, 2), "f2": (16, 16, 6)}
TRAIN_SPLITS = ("gen", "sel", "diag")
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
CODE = ("k39_cache.py", "k38_grip_run.py", "k37_confirm.py",
        "k36_harness.py", "k36_bank.py", "k36_protocol.py", "k32_policy.py",
        "k31_collect.py", "k29_policy.py", "k33_bank.py")


def rep_dir(run):
    return os.path.join("reports/k39", run)


def data_dir(run):
    return os.path.join("data/k39", run)


def sha256(p):
    import k37_confirm as kcf
    return kcf.sha256(p)


def h256(s):
    return hashlib.sha256(s.encode()).hexdigest()


# --- выборка -------------------------------------------------------------------------------
def split_scenes(c2_scenes, g1_ids):
    """-> {задача: dict(gen, sel, diag, micro)} списков сцен."""
    out = {}
    for t in TASKS:
        rest = sorted((s for s in c2_scenes if s["task"] == t and
                       s["case_id"] not in g1_ids),
                      key=lambda s: (h256(f"k39-f1-scene|{t}|{s['case_id']}"),
                                     s["case_id"]))
        need = N_TRAIN + N_DIAG + 1
        if len(rest) < need:
            raise SystemExit(f"t{t}: сцен {len(rest)} < {need}")
        tr = rest[:N_TRAIN]
        out[t] = dict(gen=tr[:N_GEN], sel=tr[N_GEN:], diag=rest[N_TRAIN:
                                                                N_TRAIN +
                                                                N_DIAG],
                      micro=[rest[N_TRAIN + N_DIAG]])
    return out


def pick_windows(eid, elapsed, k, max_steps=600):
    """-> (отобранные t, число исключённых коротких окон)."""
    ok = [int(t) for t, kk in zip(elapsed, k)
          if int(kk) >= MIN_K and int(t) + 8 <= max_steps]
    excl = len(elapsed) - len(ok)
    if not ok:
        return [], excl
    groups = np.array_split(np.asarray(ok), min(WIN_MAX, len(ok)))
    sel = [int(min(g, key=lambda t: h256(f"k39-f1-window|{eid}|{int(t)}")))
           for g in groups if len(g)]
    return sorted(sel), excl


def profile_caps(profile):
    nominal = len(TASKS) * len(SEEDS) * (N_TRAIN + N_DIAG)
    sisters = len(TASKS) * len(SEEDS) * 2 * sum(FORK_SCENES[profile])
    return dict(cache=(nominal + sisters) * EP, checks=48 * EP)


def sister_specs(e, signs):
    eligible = [t for t in e["windows"] if e["k_at"][str(t)] == 8]
    if not eligible:
        return [dict(eid=e["eid"], missing=True)]
    t = min(eligible, key=lambda tt: h256(f"k39-f1-sister|{e['eid']}|{tt}"))
    axis = int(h256(f"k39-f1-axis|{e['case_id']}|{t}")[0], 16) % 2
    return [dict(eid=e["eid"], t=t, axis=axis, sign=sign, missing=False,
                 job=f"sis/{e['eid']}/{t}/{sign:+d}") for sign in signs]


def prepare_data(a):
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    import k38_grip_run as kgr
    run = a.run_id
    profile = a.profile or ("f2" if run == "f2" else "f1")
    rd, dd = rep_dir(run), data_dir(run)
    planp = os.path.join(rd, "data_plan.json")
    c2, k36 = kcf.load_plan(a.archive)
    g1p = os.path.join(kgr.RUNS, "g1")
    g1, _k = kgr.load_plan(g1p, verify=False)
    if g1["c2"]["plan_sha256"] != c2["sha256"]:
        raise SystemExit("скрининг g1 закреплён на другом архиве c2")
    g1_ids = {s["case_id"] for s in g1["scenes"]}
    C = kh.consts(k36)
    case_part = {sc["case_id"]: int(k) for k, v in c2["parts"].items()
                 for sc in v["scenes"]}
    sp = split_scenes(c2["scenes"], g1_ids)
    eps, n_excl = [], 0
    os.makedirs(os.path.join(dd, "inputs"), exist_ok=True)
    for t in TASKS:
        for split in ("gen", "sel", "diag", "micro"):
            for sc in sp[t][split]:
                for s in SEEDS:
                    (rec, z), jp, np_ = kgr.archive_ep(a.archive, c2, k36,
                                                       case_part, s, sc, C)
                    J = {kk[2:]: v for kk, v in z.items()
                         if kk.startswith("j_")}
                    wins, ex = pick_windows(rec["eid"], J["elapsed"], J["k"])
                    n_excl += ex
                    cp = {}
                    for src, ext in ((jp, "json"), (np_, "npz")):
                        sh = sha256(src)
                        dst = os.path.join(dd, "inputs", f"{sh}.{ext}")
                        if not os.path.exists(dst):
                            import shutil
                            shutil.copyfile(src, dst + ".tmp")
                            os.replace(dst + ".tmp", dst)
                        if sha256(dst) != sh:
                            raise SystemExit(f"копия {dst} ≠ источнику")
                        cp[ext] = sh
                    eps.append(dict(eid=rec["eid"], seed=s, task=t,
                                    case_id=sc["case_id"], pos=sc["pos"],
                                    hash=sc["hash"], split=split,
                                    windows=wins, input=cp,
                                    k_at={str(int(tt)): int(kk) for tt, kk in
                                          zip(J["elapsed"], J["k"])}))
    # Сёстры получают те же идентификаторы окон в f1 и f2.
    # Очередь из сырых пар нужна отдельно для фиксированного замера задержки.
    by_case = {(e["case_id"], e["seed"]): e for e in eps}
    sisters, micro_sisters, raw = [], [], []
    for t in TASKS:
        for split, n in zip(TRAIN_SPLITS, FORK_SCENES[profile]):
            for sc in sp[t][split][:n]:
                for s in SEEDS:
                    sisters.extend(sister_specs(by_case[sc["case_id"], s],
                                                (1, -1)))
        for sc in sp[t]["diag"][:5]:
            for s in SEEDS:
                e = by_case[sc["case_id"], s]
                if e["windows"]:
                    w = min(e["windows"], key=lambda tt: h256(
                        f"k39-f1-raw|{e['eid']}|{tt}"))
                    raw.append(dict(eid=e["eid"], t=int(w)))
    if profile == "f2":
        for s in SEEDS:
            for t in TASKS:
                sc = sp[t]["micro"][0]
                specs = sister_specs(by_case[sc["case_id"], s], (0, 1, -1))
                if not specs[0]["missing"]:
                    micro_sisters.extend(specs)
                    break
            else:
                raise SystemExit(f"micro s{s}: нет полного окна для развилки")
    plan = dict(kind="k39_data_plan", run_id=run, protocol=PROTOCOL,
                profile=profile,
                code={f: sha256(os.path.join(HERE, f)) for f in CODE},
                c2_plan_sha256=c2["sha256"], g1_plan_sha256=g1["sha256"],
                k36_prepare_sha1=k36["sha1"], bank=c2["bank"],
                splits={str(t): {k: [s["case_id"] for s in v]
                                 for k, v in sp[t].items()} for t in TASKS},
                episodes=eps, sisters=sisters, micro_sisters=micro_sisters,
                raw_pairs=raw, n_excluded_short=n_excl,
                caps=profile_caps(profile), threads=THREADS,
                thumb=THUMB, sister_delta=SIS_DELTA)
    sh = kcf.canon_sha(plan)
    if os.path.exists(planp):
        old = json.load(open(planp, encoding="utf-8"))
        rec_sh = open(os.path.join(rd, "data_plan.sha256")).read().strip()
        if not (kcf.canon_sha(old) == rec_sh == sh):
            raise SystemExit(f"{planp} закреплён иначе")
        print(f"  data_plan закреплён, совпадает: {sh[:16]}")
        return 0
    pc.save_json(planp, plan)
    with open(os.path.join(rd, "data_plan.sha256"), "w") as fh:
        fh.write(sh + "\n")
    nw = {sp_: sum(len(e["windows"]) for e in eps if e["split"] in sp_)
          for sp_ in (("gen", "sel"), ("diag",), ("micro",))}
    print(f"  data_plan {planp} ({profile}): sha256 {sh[:16]}; "
          f"эпизодов {len(eps)} "
          f"(train {sum(e['split'] in ('gen', 'sel') for e in eps)}, diag "
          f"{sum(e['split'] == 'diag' for e in eps)}, micro "
          f"{sum(e['split'] == 'micro' for e in eps)}); окон train "
          f"{nw[('gen', 'sel')]}, diag {nw[('diag',)]}, micro "
          f"{nw[('micro',)]}; исключено коротких {n_excl}; сестёр "
          f"{sum(not x['missing'] for x in sisters)} (без окна k=8: "
          f"{sum(x['missing'] for x in sisters)}); сырых пар для замера "
          f"{len(raw)}; развилок micro {len(micro_sisters)}; "
          f"лимит cache {plan['caps']['cache']}")
    return 0


def load_data_plan(run, verify=True):
    import k37_confirm as kcf
    rd = rep_dir(run)
    p = json.load(open(os.path.join(rd, "data_plan.json"), encoding="utf-8"))
    sh = kcf.canon_sha(p)
    if sh != open(os.path.join(rd, "data_plan.sha256")).read().strip():
        raise SystemExit("data_plan.json ≠ data_plan.sha256")
    if verify:
        if p["protocol"] != PROTOCOL:
            raise SystemExit("для этого плана нужна исходная версия кеша")
        bad = [f for f, s in p["code"].items()
               if sha256(os.path.join(HERE, f)) != s]
        if bad:
            raise SystemExit(f"код ≠ data_plan: {bad}")
        for f, s in p["bank"]["files"].items():
            if not os.path.exists(f) or sha256(f) != s:
                raise SystemExit(f"банк {f} ≠ плану")
    p["sha256"] = sh
    return p


def load_input(run, e):
    import k36_protocol as pc
    dd = data_dir(run)
    jp = os.path.join(dd, "inputs", f"{e['input']['json']}.json")
    np_ = os.path.join(dd, "inputs", f"{e['input']['npz']}.npz")
    if sha256(jp) != e["input"]["json"] or sha256(np_) != e["input"]["npz"]:
        raise SystemExit(f"вход {e['eid']} ≠ плану")
    return pc.load_json(jp), dict(np.load(np_))


# --- наблюдения ----------------------------------------------------------------------------
def thumbs(torch, obs):
    """Две камеры как в k29_policy.Inputs: [:, :, ::-1], центральный crop
    196 (смещение (224 − 196) / 2 = 14), bilinear antialias до 48 ->
    [2, 3, 48, 48] uint8. Чистый torch (без torchvision)."""
    out = []
    for key in CAMERAS:
        x = torch.tensor(np.asarray(obs[key])[:, :, ::-1].copy()).permute(
            0, 3, 1, 2).float()
        H, W = x.shape[-2:]
        top, left = int(round((H - 196) / 2)), int(round((W - 196) / 2))
        x = x[..., top:top + 196, left:left + 196]
        x = torch.nn.functional.interpolate(x, size=(THUMB, THUMB),
                                            mode="bilinear",
                                            align_corners=False,
                                            antialias=True)
        out.append(torch.clamp(torch.round(x), 0, 255).to(torch.uint8)[0])
    return torch.stack(out).numpy()


def raw_obs(obs):
    return {k: np.array(obs[k], copy=True) for k in ("state", *CAMERAS)}


def observation_fields(torch, io, obs, elapsed):
    from k31_collect import obs_hash
    return {f"o{elapsed}": thumbs(torch, obs),
            f"s{elapsed}": np.array(io.inp.state8(obs)[0], copy=True),
            f"rgb{elapsed}": np.stack([obs[key][0] for key in CAMERAS]),
            f"state{elapsed}_raw": np.array(obs["state"][0], copy=True),
            f"obs{elapsed}_hash": obs_hash(obs)}


class Replay:
    """Повтор архивного эпизода в BankSim с точками захвата наблюдений."""

    def __init__(self, sim, io, task, reset_id):
        import k31_collect as kcol
        self.kcol, self.sim, self.io = kcol, sim, io
        io.seed(kcol.ENV["seed"])
        self.obs = sim.reset(task, [reset_id])
        dummy = np.array([kcol.DUMMY])
        done = False
        self.reward = 0.0
        for _ in range(kcol.ENV["waiting_steps"]):
            self.obs, r_, d_, _s = sim.step(dummy)
            done |= bool(d_[0])
            self.reward = min(max(self.reward + float(r_[0]), 0.0), 1.0)
        if done or self.reward >= 1.0:
            raise SystemExit("среда завершилась во время ожидания")
        import k18_collector as k18c
        self.ih = k18c.init_hashes(self.obs, 1)
        self.steps, self.succ_step, self.done_step = 0, -1, -1

    def step(self, a):
        if self.done_step >= 0:
            raise SystemExit("повтор продолжился после terminal")
        obs, r_, d_, _s = self.sim.step(np.asarray(a, np.float32)[None])
        self.obs = obs
        prev = self.reward
        self.reward = min(max(self.reward + float(r_[0]), 0.0), 1.0)
        if self.reward >= 1.0 and prev < 1.0 and self.succ_step < 0:
            self.succ_step = self.steps
        if bool(d_[0]) and self.done_step < 0:
            self.done_step = self.steps
        self.steps += 1


def teacher(torch, arm, io, desc, task, reset_id, ep_no, chunk, obs,
            elapsed):
    """Изолированный запрос T32 -> (a [8,7], секунды)."""
    t0 = time.monotonic()
    arm.begin(task, reset_id, ep_no)
    arm.call = int(chunk)
    batch, po, s8 = io.prep(obs, desc, task)
    a, _rec = arm.decide(batch, po, s8, int(elapsed))
    return np.asarray(a, np.float32), time.monotonic() - t0


def run_episode_rows(torch, sim, io, arm, e, ref, check_all, raw_ts):
    """Полный повтор + окна -> (данные шарда, проверки, время)."""
    import k21_model as km
    import k36_harness as kh
    rec, z = ref
    A = np.asarray(z["actions"], np.float32)
    J = {k[2:]: v for k, v in z.items() if k.startswith("j_")}
    desc = io.task_desc(e["task"])
    ep_no = kh.episode_number(e["eid"])
    q = io.inp.max_act_q
    import k29_policy as kp
    t0 = time.monotonic()
    rp = Replay(sim, io, e["task"], e["pos"])
    prob, tsec = [], []
    if (rp.ih[0][0], rp.ih[1][0]) != (rec["summary"]["init_hash"],
                                      rec["summary"]["init_hash_full"]):
        prob.append("post-wait хеши ≠ архиву")
    wins = set(e["windows"])
    idx = {int(t): i for i, t in enumerate(J["elapsed"])}
    rows = dict(t=[], o0=[], o4=[], s0=[], s4=[], B=[], Y=[], k=[],
                direct_ok=[], prefix_phys=[], rgb0=[], rgb4=[],
                state0_raw=[], state4_raw=[], obs0_hash=[], obs4_hash=[])
    raws = {}
    first = True
    pend = {}
    for st in range(len(A)):
        if st in wins:
            i = idx[st]
            B = np.asarray(J["a_codec"][i], np.float32)
            if not np.array_equal(kp.scale_exec(B[None], q)[0][:4],
                                  A[st:st + 4]):
                prob.append(f"t={st}: actions ≠ io.scale(B[:4])")
            ok = None
            if check_all or first:
                a0, ts = teacher(torch, arm, io, desc, e["task"], e["pos"],
                                 ep_no, st // 8, rp.obs, st)
                tsec.append(ts)
                ok = bool(np.array_equal(a0, B))
                if not ok:
                    prob.append(f"t={st}: прямой T32 ≠ j_a_codec")
                first = False
            pend[st + 4] = dict(t=st, B=B, prefix_phys=A[st:st + 4],
                                k=int(J["k"][i]), ok=ok,
                                raw0=raw_obs(rp.obs) if st in raw_ts
                                else None,
                                **observation_fields(torch, io, rp.obs, 0))
        if st in pend:
            p_ = pend.pop(st)
            Y, ts = teacher(torch, arm, io, desc, e["task"], e["pos"],
                            ep_no, p_["t"] // 8, rp.obs, st)
            tsec.append(ts)
            if check_all:
                Y2, ts2 = teacher(torch, arm, io, desc, e["task"], e["pos"],
                                  ep_no, p_["t"] // 8, rp.obs, st)
                tsec.append(ts2)
                if not np.array_equal(Y, Y2):
                    prob.append(f"t={st}: повтор off-grid ≠")
            p_.update(observation_fields(torch, io, rp.obs, 4))
            p_.update(Y=Y, direct_ok=-1 if p_["ok"] is None else int(p_["ok"]))
            for kk in rows:
                v = p_[kk]
                rows[kk].append(v)
            if p_["raw0"] is not None:
                raws[p_["t"]] = dict(o0=p_["raw0"], o4=raw_obs(rp.obs))
        rp.step(A[st])
    s = rec["summary"]
    succ = rp.reward >= 1.0
    if (succ, rp.succ_step, rp.done_step, rp.steps) != (
            s["success"], s["success_step"], s["done_step"], s["steps"]):
        prob.append(f"повтор ≠ архиву: {(succ, rp.succ_step, rp.done_step, rp.steps)}"
                    f" vs {(s['success'], s['success_step'], s['done_step'], s['steps'])}")
    if pend:
        prob.append(f"окна без o4: {sorted(pend)}")
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["state11_0"] = km.state11(out["s0"]) if len(out["s0"]) else \
        np.zeros((0, 11))
    out["state11_4"] = km.state11(out["s4"]) if len(out["s4"]) else \
        np.zeros((0, 11))
    return out, raws, prob, dict(replay_s=time.monotonic() - t0 - sum(tsec),
                                 teacher_s=sum(tsec), n_teacher=len(tsec),
                                 phys=10 + len(A))


def run_sister(torch, sim, io, arm, e, ref, sis, C, nominal):
    """Префикс из reset до t, затем 4 команды с ±δ на индексах 1, 2."""
    import k29_policy as kp
    import k36_harness as kh
    rec, z = ref
    A = np.asarray(z["actions"], np.float32)
    J = {k[2:]: v for k, v in z.items() if k.startswith("j_")}
    t = sis["t"]
    i = {int(tt): j for j, tt in enumerate(J["elapsed"])}[t]
    B = np.asarray(J["a_codec"][i], np.float32)
    P = B[:4].copy()
    if sis["sign"]:
        axis = sis["axis"]
        delta = SIS_DELTA * float(C["half"][axis]) * sis["sign"]
        for j in (1, 2):
            P[j, axis] = np.clip(P[j, axis] + delta, C["L"][axis], C["U"][axis])
    phys = kp.scale_exec(P[None], io.inp.max_act_q)[0]
    start = time.monotonic()
    rp = Replay(sim, io, e["task"], e["pos"])
    prob = []
    if (rp.ih[0][0], rp.ih[1][0]) != (rec["summary"]["init_hash"],
                                      rec["summary"]["init_hash_full"]):
        prob.append("сестра: post-wait хеши ≠ архиву")
    for st in range(t):
        rp.step(A[st])
    o0 = observation_fields(torch, io, rp.obs, 0)
    if o0["obs0_hash"] != nominal["obs0_hash"]:
        prob.append("сестра: o0 ≠ номинальному наблюдению")
    if not np.array_equal(phys[0], A[t]):
        prob.append("действие до возмущения ≠ архиву")
    alive = 0
    for j in range(4):
        if rp.done_step >= 0:
            break
        rp.step(phys[j])
        alive += 1
    o_mid = observation_fields(torch, io, rp.obs, 4)
    Y, ts = None, 0.0
    if rp.done_step < 0 and alive == 4:
        Y, ts = teacher(torch, arm, io, io.task_desc(e["task"]), e["task"],
                        e["pos"], kh.episode_number(e["eid"]), t // 8,
                        rp.obs, t + 4)
    if sis["sign"] == 0 and (Y is None or
                            not np.array_equal(phys, A[t:t + 4]) or
                            o_mid["obs4_hash"] != nominal["obs4_hash"] or
                            not np.array_equal(Y, nominal["Y"])):
        prob.append("нулевой повтор: действия, o4 или Y ≠ номиналу")
    # Фотография из обрыва имеет собственную отметку и не служит входом o4.
    out = dict(t=np.array(t), axis=np.array(sis["axis"]),
               sign=np.array(sis["sign"]), B=B, issued=B[:4],
               executed=P, amp=np.array(float(P[1, sis["axis"]] -
                                              B[1, sis["axis"]])),
               prefix_phys=phys[:alive], alive=np.array(alive),
               terminal_step=np.array(rp.done_step), has_o4=np.array(alive == 4),
               Y=Y if Y is not None else np.zeros((8, 7), np.float32),
               has_y=np.array(Y is not None), **o0, **o_mid)
    return out, prob, dict(phys=10 + t + alive, teacher_s=ts,
                           n_teacher=int(Y is not None),
                           replay_s=time.monotonic() - start - ts)


# --- исполнение с журналом -----------------------------------------------------------------
def shard_path(run, job):
    return os.path.join(data_dir(run), "shards",
                        job.replace("/", "__") + ".npz")


def write_shard(path, data, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as fh:
        np.savez_compressed(fh, meta=np.array(json.dumps(meta)), **data)
    os.replace(path + ".tmp", path)


def shard_ok(path, job, plan_sha):
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        m = json.loads(str(z["meta"]))
        phys = m.get("phys")
        if (m.get("job") != job or m.get("plan") != plan_sha or
                m.get("problems") or type(phys) is not int or
                not 0 < phys <= EP):
            return False
        for k in z.files:
            v = z[k]
            if v.dtype.kind == "f" and not np.isfinite(v).all():
                return False
    return m


def jobs_of(plan, splits, with_sisters):
    js = [("ep", e) for e in plan["episodes"] if e["split"] in splits]
    if with_sisters:
        by = {e["eid"]: e for e in plan["episodes"]}
        which = "micro_sisters" if splits == ("micro",) else "sisters"
        js += [("sis", dict(by[s["eid"]], sis=s)) for s in plan.get(which, [])
               if not s["missing"]]
    return js


def job_name(tag, kind, e):
    return f"{tag}/{e['eid'] if kind == 'ep' else e['sis']['job']}"


def nominal_window(run, tag, e):
    """Один проверенный выше в той же очереди номинальный результат."""
    path = shard_path(run, job_name(tag, "ep", e))
    with np.load(path, allow_pickle=False) as z:
        ix = np.flatnonzero(z["t"] == e["sis"]["t"])
        if len(ix) != 1:
            raise SystemExit(f"{path}: нет единственного номинального окна")
        return {k: z[k][ix[0]] for k in ("obs0_hash", "obs4_hash", "Y")}


def restore_pending(run, plan, ledger):
    pending = ledger.d["pending"]
    if pending is None:
        return
    scope = pending["scope"]
    if scope == "checks":
        tag, splits = "micro", ("micro",)
    elif scope == "cache":
        tag, splits = "cache", TRAIN_SPLITS
    else:
        raise SystemExit(f"резерв с неизвестной областью {scope}")
    job = pending["id"]
    expected = {job_name(tag, k, e) for k, e in jobs_of(plan, splits, True)}
    if job not in expected or pending["amount"] != EP:
        raise SystemExit(f"резерв {job} не соответствует data_plan")
    m = shard_ok(shard_path(run, job), job, plan["sha256"])
    if m is False:
        raise SystemExit(f"{job}: повреждён записанный шард резерва")
    if m:
        ledger.settle(job, m["phys"])


def sister_report(run, tag, plan):
    which = "micro_sisters" if tag == "micro" else "sisters"
    count = dict(planned=0, both_live=0, different_obs=0, different_target=0)
    for spec in plan[which]:
        if spec["missing"] or spec["sign"] != 1:
            continue
        count["planned"] += 1
        plus = f"{tag}/{spec['job']}"
        minus = plus.rsplit("/", 1)[0] + "/-1"
        with np.load(shard_path(run, plus), allow_pickle=False) as pos, \
                np.load(shard_path(run, minus), allow_pickle=False) as neg:
            if bool(pos["has_y"]) and bool(neg["has_y"]):
                count["both_live"] += 1
                count["different_obs"] += int(pos["obs4_hash"] != neg["obs4_hash"])
                yp, yn = pos["Y"][:4], neg["Y"][:4]
                count["different_target"] += int(
                    np.max(np.abs(yp[:, :6] - yn[:, :6])) > 1e-5 or
                    not np.array_equal(yp[:, 6], yn[:, 6]))
    return count


def execute(a, splits, with_sisters, check_all, scope, tag):
    import torch
    import k31_collect as kcol
    import k36_bank as kb36
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    import k38_grip_run as kgr
    run = a.run_id
    plan = load_data_plan(run)
    k36 = pc.load_prepare()
    if k36["sha1"] != plan["k36_prepare_sha1"]:
        raise SystemExit("prepare K-36 ≠ data_plan")
    tp = kgr.threads_problems(torch)
    if tp:
        raise SystemExit("потоки: " + "; ".join(tp))
    C = kh.consts(k36)
    idx = kcf.dev_index(a.device)
    g = kcf.gpu_query(idx)
    rd = rep_dir(run)
    locks = [kcf.Lock(os.path.join(rd, ".ledger.lock"),
                     f"журнал {run}", kcf.Busy)]
    try:
        locks.append(kcf.Lock(os.path.join(kcf.lock_dir(),
                                           g["uuid"] + ".lock"),
                              f"карта {g['uuid']}", kcf.Busy))
        kcf.check_gpu(idx, kcf.RES["min_free_gb"])
        kcf.check_disk(rd)
        if scope == "cache":
            mp = os.path.join(rd, "micro.json")
            if not os.path.exists(mp):
                raise SystemExit("сначала выполните micro для этого data_plan")
            m = pc.load_json(mp)
            if m.get("rc") != 0 or m.get("plan_sha256") != plan["sha256"]:
                raise SystemExit("micro не пройден для этого data_plan")
        L = kcol.Ledger(os.path.join(rd, "ledger.json"), plan["caps"])
        restore_pending(run, plan, L)
        raw_ts = {}
        for r in plan["raw_pairs"]:
            raw_ts.setdefault(r["eid"], set()).add(r["t"])
        jobs = jobs_of(plan, splits, with_sisters)
        sim = kb36.BankSim("bank", 2, bank_dir=plan["bank"]["dir"])
        stats = []
        try:
            kcf.bind_device(torch, idx, g)
            torch.cuda.reset_peak_memory_stats(idx)
            for s in SEEDS:
                js = [j for j in jobs if j[1]["seed"] == s]
                if not js:
                    continue
                kcf.check_gpu(idx, kcf.RES["min_free_gb"])
                v = kh.load_source(torch, s, a.device)
                arm = kh.make_arm(torch, k36, f"T32_s{s}", v)
                for kind, e in js:
                    job = job_name(tag, kind, e)
                    path = shard_path(run, job)
                    m = shard_ok(path, job, plan["sha256"])
                    if m:
                        if L.d["done"].get(job) != dict(scope=scope,
                                                        actual=m["phys"]):
                            raise SystemExit(f"{job}: шард ≠ журналу")
                        stats.append(m)
                        continue
                    if m is False:
                        raise SystemExit(f"{path}: шард с проблемами или "
                                         f"чужой — разбор вручную")
                    if L.is_done(job):
                        raise SystemExit(f"{job}: учтён, шарда нет")
                    ref = load_input(run, e)
                    nominal = nominal_window(run, tag, e) if kind == "sis" else None
                    kcf.check_disk(rd)
                    L.reserve(scope, job, amount=EP)
                    if kind == "ep":
                        data, raws, prob, st = run_episode_rows(
                            torch, sim, v["io"], arm, e, ref, check_all,
                            raw_ts.get(e["eid"], set()))
                        for tt, rw in raws.items():
                            write_shard(shard_path(run, f"raw/{e['eid']}/{tt}"),
                                        {f"{w}_{k}": x for w in ("o0", "o4")
                                         for k, x in rw[w].items()},
                                        dict(job=f"raw/{e['eid']}/{tt}",
                                             plan=plan["sha256"], phys=0,
                                             problems=[]))
                    else:
                        data, prob, st = run_sister(torch, sim, v["io"], arm,
                                                    e, ref, e["sis"], C, nominal)
                    meta = dict(job=job, plan=plan["sha256"], eid=e["eid"],
                                seed=s, task=e["task"], case_id=e["case_id"],
                                split=e["split"], kind=kind, problems=prob,
                                **st)
                    write_shard(path, data, meta)
                    stats.append(meta)
                    if prob or not shard_ok(path, job, plan["sha256"]):
                        print(f"  {job}: ПРОБЛЕМЫ {prob[:3]} — резерв "
                              f"открыт, остановка", flush=True)
                        return 3, stats
                    L.commit(job, st["phys"])
                    n = sum(L.is_done(job_name(tag, k, x)) for k, x in js)
                    print(f"  s{s} {kind} {e['case_id']} {e['split']}: "
                          f"окон {len(data['t']) if kind == 'ep' else 1}, "
                          f"replay {st.get('replay_s', 0):.1f} с, учитель "
                          f"{st.get('teacher_s', 0):.1f} с; сид {n}/"
                          f"{len(js)}", flush=True)
                arm = v = None
                kh.free(torch)
        except torch.cuda.OutOfMemoryError as e:
            raise kcf.Busy(f"CUDA OOM: {e}")
        finally:
            kcf.close_sim(sim)
        peak = torch.cuda.max_memory_reserved(idx) / 2 ** 30
        return 0, stats + [dict(peak_gb=peak, env=kgr.env_record(torch),
                                gpu=g)]
    finally:
        for lk in reversed(locks):
            lk.release()


def micro(a):
    import k36_protocol as pc
    rc, st = execute(a, ("micro",), True, True, "checks", "micro")
    plan = load_data_plan(a.run_id, verify=False)
    eps = [x for x in st if x.get("kind") == "ep"]
    sibs = [x for x in st if x.get("kind") == "sis"]
    rep = dict(rc=rc, plan_sha256=plan["sha256"], episodes=eps, sisters=sibs,
               tail=st[-1] if st and "peak_gb" in st[-1] else None,
               time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if rc == 0 and eps:
        rs = np.mean([x["replay_s"] for x in eps])
        tq = np.sum([x["teacher_s"] for x in eps]) / max(
            1, np.sum([x["n_teacher"] for x in eps]))
        nw = sum(len(e["windows"]) for e in plan["episodes"]
                 if e["split"] in TRAIN_SPLITS)
        ne = sum(e["split"] in TRAIN_SPLITS for e in plan["episodes"])
        nd = sum(e["split"] in TRAIN_SPLITS and bool(e["windows"])
                 for e in plan["episodes"])
        ns = sum(not s["missing"] for s in plan["sisters"])
        short_s = (np.mean([x["replay_s"] + x["teacher_s"] for x in sibs])
                   if sibs else rs * 0.6 + tq)
        eta = (ne * rs + (nw + nd) * tq + ns * short_s) / 3600
        rep["eta_h"] = float(eta)
        print(f"  micro: эпизодов {len(eps)}, replay {rs:.1f} с/эп, учитель "
              f"{tq:.2f} с/вызов; collect: {ne} эп., {nw} окон, {ns} сестёр "
              f"-> ориентировочно {eta:.1f} ч", flush=True)
        rep["forks"] = sister_report(a.run_id, "micro", plan)
        if sibs:
            print(f"  micro: сёстры {rep['forks']}; "
                  f"короткий повтор {short_s:.1f} с в среднем", flush=True)
    pc.save_json(os.path.join(rep_dir(a.run_id), "micro.json"), rep)
    print("  micro: " + ("пройден" if rc == 0 else "НЕ пройден"))
    return rc


def collect(a):
    rc, st = execute(a, TRAIN_SPLITS, True, False, "cache",
                     "cache")
    import k36_protocol as pc
    rep = dict(rc=rc, n=len([x for x in st if "job" in x]),
               tail=st[-1] if st and "peak_gb" in st[-1] else None,
               time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if rc == 0:
        plan = load_data_plan(a.run_id, verify=False)
        rep["forks"] = sister_report(a.run_id, "cache", plan)
        print(f"  collect: сёстры {rep['forks']}", flush=True)
    pc.save_json(os.path.join(rep_dir(a.run_id), "collect_last.json"), rep)
    print("  collect: " + ("завершён" if rc == 0 else "остановлен"))
    return rc


def status(a):
    plan = load_data_plan(a.run_id, verify=False)
    lp = os.path.join(rep_dir(a.run_id), "ledger.json")
    d = json.load(open(lp)) if os.path.exists(lp) else dict(done={},
                                                            spent={},
                                                            pending=None,
                                                            crashes=[])
    for tag, splits in (("micro", ("micro",)), ("cache", TRAIN_SPLITS)):
        js = jobs_of(plan, splits, True)
        names = [job_name(tag, k, e) for k, e in js]
        print(f"  {tag}: {sum(n in d['done'] for n in names)}/{len(names)}")
    print(f"  расход {d['spent']}; резерв {(d['pending'] or {}).get('id', '—')}"
          f"; аварий {len(d['crashes'])}")
    return 0


def selftest_replay(torch):
    import tempfile
    import k31_collect as kcol
    from k32_ppo import ToySim

    class Sim(ToySim):
        def _obs(self):
            obs = super()._obs()
            for key in CAMERAS:
                obs[key] = np.broadcast_to(obs[key][:, :1, :1],
                                           (1, 224, 224, 3)).copy()
            return obs

    class IO(kcol.FakeIO):
        @property
        def inp(self):
            return self

        def state8(self, obs):
            return np.asarray(obs["state"][:, :8], np.float64)

    class Arm(kcol.FakeAgent):
        def begin(self, task, init, episode=None):
            super().begin(task, init)

    io, arm = IO(), Arm()
    bas = kcol.run_episode(Sim(), io, arm, 0, 2)
    B = bas["calls"][0]["a_codec"]
    e = dict(eid="toy", task=0, pos=2, windows=[0])
    ref = (dict(summary=bas), dict(actions=bas["actions"],
                                   j_elapsed=np.array([0]),
                                   j_a_codec=B[None], j_k=np.array([8])))
    out, _, prob, _ = run_episode_rows(torch, Sim(), io, arm, e, ref,
                                       True, set())
    assert not prob, prob
    assert out["rgb4"].shape == (1, 2, 224, 224, 3)
    assert out["rgb4"].dtype == np.uint8
    assert np.array_equal(out["prefix_phys"][0], bas["actions"][:4])
    nominal = {k: out[k][0] for k in ("obs0_hash", "obs4_hash", "Y")}
    C = dict(half=np.ones(6), L=-np.ones(6), U=np.ones(6))
    sibs = []
    for sign in (0, 1, -1):
        spec = dict(t=0, axis=0, sign=sign)
        data, prob, _ = run_sister(torch, Sim(), io, arm, e, ref, spec, C,
                                     nominal)
        assert not prob and bool(data["has_y"]), prob
        assert data["obs0_hash"] == nominal["obs0_hash"]
        sibs.append(data)
    assert np.array_equal(sibs[0]["rgb4"], out["rgb4"][0])
    assert np.array_equal(sibs[0]["state4_raw"], out["state4_raw"][0])
    assert sibs[1]["obs4_hash"] != sibs[2]["obs4_hash"]
    assert not np.array_equal(sibs[1]["Y"], sibs[2]["Y"])
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "row.npz")
        write_shard(path, out, dict(job="toy", plan="test", phys=22,
                                    problems=[]))
        assert shard_ok(path, "toy", "test")
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z["rgb4"], out["rgb4"])
            assert np.array_equal(z["state4_raw"], out["state4_raw"])


def selftest():
    sc = [dict(task=t, case_id=f"k33s2_t{t}_c{i:05d}", pos=i, hash=None)
          for t in TASKS for i in range(160)]
    g1 = {s["case_id"] for s in sc[:20]} | {s["case_id"] for s in
                                            sc[160:180]} | \
        {s["case_id"] for s in sc[320:340]}
    sp = split_scenes(sc, g1)
    for t in TASKS:
        ids = [s["case_id"] for k in ("gen", "sel", "diag", "micro")
               for s in sp[t][k]]
        assert len(ids) == len(set(ids)) == 61 and not set(ids) & g1
        assert len(sp[t]["gen"]) == 32 and len(sp[t]["sel"]) == 16
    assert split_scenes(sc, g1) == sp
    el = np.arange(0, 600, 8)
    k = np.full(len(el), 8)
    k[-1] = 3
    w, ex = pick_windows("e", el, k)
    assert len(w) == 8 and ex == 1 and all(t + 8 <= 600 for t in w)
    assert w == sorted(w) and pick_windows("e", el, k) == (w, ex)
    w2, ex2 = pick_windows("e", np.array([0, 8, 16]), np.array([8, 4, 8]))
    assert w2 == [0, 16] and ex2 == 1
    # миниатюры: форма, тип; crop по центру; [:, :, ::-1] (ось ширины у
    # [n, H, W, 3]) как в Inputs — левое уходит вправо
    import torch
    img = np.zeros((1, 224, 224, 3), np.uint8)
    img[0, :, :, 0] = 200                          # канал 0
    img[0, :14, :, :] = 255                        # полоса вне crop
    th = thumbs(torch, dict(agentview_image=img,
                            robot0_eye_in_hand_image=img))
    assert th.shape == (2, 3, 48, 48) and th.dtype == np.uint8
    assert (th[0, 0] == 200).all() and (th[0, 2] == 0).all()
    img2 = np.zeros((1, 224, 224, 3), np.uint8)
    img2[0, :, 14:112, 1] = 255                    # левая половина crop
    t2 = thumbs(torch, dict(agentview_image=img2,
                            robot0_eye_in_hand_image=img2))
    assert t2[0, 1, :, -1].min() == 255 and t2[0, 1, :, 0].max() == 0
    assert profile_caps("f1") == dict(cache=263520, checks=29280)
    assert profile_caps("f2") == dict(cache=497760, checks=29280)
    selftest_replay(torch)
    print("самопроверка k39_cache пройдена: разбиение 32/16/12/1 без сцен "
          "g1, детерминизм; окна (j_k ≥ 5, t + 8 ≤ 600, ≤ 8 групп); "
          "исходные кадры и миниатюры; нулевой и два сестринских повтора; "
          "лимиты f1/f2 263 520 / 497 760")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-39 кеш")
    ap.add_argument("cmd", choices=("prepare-data", "micro", "collect",
                                    "status"))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--profile", choices=tuple(FORK_SCENES))
    ap.add_argument("--archive", default="reports/k37c/runs/c2")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    import k37_confirm as kcf
    fn = {"prepare-data": prepare_data, "micro": micro, "collect": collect,
          "status": status}[a.cmd]
    try:
        return fn(a)
    except kcf.Busy as e:
        print(f"  ВРЕМЕННО НЕТ РЕСУРСА: {e}", flush=True)
        return 4
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
