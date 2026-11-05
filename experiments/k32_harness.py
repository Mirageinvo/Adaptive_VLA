#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-32: SR-оценка B = 1 (детерминированно) — блоки по 5 последовательных
эпизодов, руки parents_ref32 / parents_subtree_rl32 / parents_profile_rl32 /
parents_direct_rl32 сида.

Эпизод — k31_collect.run_episode с k32_policy.Arm (stochastic=False). Блок
(reports/k32/eval/<label>/t<T>_i<B>.json + npz) пишется только целиком;
обвязка: n_envs = 1, последовательно, B = 1.
Проверка блока (CPU): метка/сид/план/контракт/актор; init b…b+4 по
порядку; описание; начальный хеш = холодному эталону K-25; действия npz =
action_sha1 и = scale_exec(записанных команд)[:k] побитово; журнал вызова:
формы и конечность всех используемых групп; схват = знак логита
reference; рука reference = mid + half·tanh(Z0/half) (≤ 1e−6); для actor:
r = argmax logits (ничья — меньший), η = μ_r побитово; η ≡ 0 -> рука
reference побитово; иначе D пересчитывается по записанным величинам (T —
из Zhat, P — из профиля w_r и c_r, F — из η), затем ограничитель и рука —
побитово; последовательность вызовов (elapsed + k).
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k31_collect as kcol  # noqa: E402
import k32_operator as ko  # noqa: E402

REQUIRED = ("ep", "elapsed", "k", "a_codec", "Z0", "ref_arm", "grip_logit",
            "has_actor", "logits", "mu", "r", "eta", "c", "w", "Zhat",
            "has_zhat", "zero", "D", "Du", "path")
TASKS = tuple(range(10))
BLOCKS = (30, 35, 40)
PRIMARY = (0, 6, 8)
KINDS = dict(R="ref32", T="subtree_rl32", P="profile_rl32", F="direct_rl32")
ARM_TOL = 1e-6
N8 = ko.N8


def label(kind, seed):
    return f"parents_{KINDS[kind]}_s{seed}"


def parse_label(lab):
    for k, v in KINDS.items():
        pre = f"parents_{v}_s"
        if lab.startswith(pre):
            return k, int(lab[len(pre):])
    raise ValueError(lab)


def eval_dir(lab):
    return f"reports/k32/eval/{lab}"


def block_path(lab, task, block):
    return os.path.join(eval_dir(lab), f"t{task}_i{block}.json")


def pack_journal(ec, d):
    rows = [(ep, c) for ep, calls in ec for c in calls]
    n = len(rows)
    J = dict(ep=np.zeros(n, np.int64), elapsed=np.zeros(n, np.int64),
             k=np.zeros(n, np.int64), a_codec=np.zeros((n, N8, 7), np.float32),
             Z0=np.zeros((n, 20, 6), np.float32),
             ref_arm=np.zeros((n, N8, 6), np.float32),
             grip_logit=np.zeros((n, N8), np.float32),
             has_actor=np.zeros(n, bool), logits=np.zeros((n, 3), np.float32),
             mu=np.zeros((n, 3, d), np.float32), r=np.zeros(n, np.int64),
             eta=np.zeros((n, d), np.float32), c=np.zeros((n, 3), np.float32),
             w=np.zeros((n, N8), np.float32),
             Zhat=np.zeros((n, 20, 6), np.float32),
             has_zhat=np.zeros(n, bool), zero=np.zeros(n, bool),
             D=np.zeros((n, N8, 6), np.float32),
             Du=np.zeros((n, N8, 6), np.float32), path=np.zeros((n, 3),
                                                                 np.int64))
    for i, (ep, cl) in enumerate(rows):
        rec = cl["rec"]
        J["ep"][i], J["elapsed"][i], J["k"][i] = ep, rec["elapsed"], cl["k"]
        J["a_codec"][i], J["Z0"][i] = rec["a_codec"], rec["Z0"]
        J["ref_arm"][i], J["grip_logit"][i] = rec["ref_arm"], rec["grip_logit"]
        if "logits" in rec:
            J["has_actor"][i] = True
            for k_ in ("logits", "mu", "r", "eta", "c", "zero", "D", "Du",
                       "path"):
                J[k_][i] = rec[k_]
            J["w"][i] = rec.get("w_r", np.zeros(N8))
            if rec.get("Zhat") is not None:
                J["Zhat"][i], J["has_zhat"][i] = rec["Zhat"], True
    return J


def journal_problems(J, A_eps, done_steps, half, mid, L, U, q, kind):
    import torch
    import k29_policy as kp
    p = []
    miss = [k_ for k_ in REQUIRED if k_ not in J]
    if miss:
        return [f"журнал: нет полей {miss}"]
    n = len(J["ep"])
    d = J["eta"].shape[-1] if J["eta"].ndim == 2 else -1
    shapes = dict(ep=(n,), elapsed=(n,), k=(n,), a_codec=(n, N8, 7),
                  Z0=(n, 20, 6), ref_arm=(n, N8, 6), grip_logit=(n, N8),
                  has_actor=(n,), logits=(n, 3), mu=(n, 3, d), r=(n,),
                  eta=(n, d), c=(n, 3), w=(n, N8), Zhat=(n, 20, 6),
                  has_zhat=(n,), zero=(n,), D=(n, N8, 6), Du=(n, N8, 6),
                  path=(n, 3))
    bad = [k_ for k_, sh in shapes.items() if J[k_].shape != sh]
    if bad or (kind != "R" and d != ko.DIM[kind]):
        return [f"журнал: неверные формы {bad or ['eta']}"]
    if sorted(set(J["ep"].tolist())) != list(range(len(A_eps))):
        return [f"журнал: эпизоды {sorted(set(J['ep'].tolist()))} ≠ "
                f"0…{len(A_eps) - 1}"]
    if not np.array_equal(J["ep"], np.sort(J["ep"], kind="stable")):
        return ["журнал: записи эпизодов не по порядку"]
    if ((J["k"] < 1) | (J["k"] > N8)).any():
        return ["журнал: k вне 1…8"]
    for k_, v in J.items():
        if v.dtype.kind == "f" and not np.isfinite(v).all():
            return [f"журнал {k_}: нечисловые значения"]
    h = torch.as_tensor(np.asarray(half, np.float32))
    m_ = torch.as_tensor(np.asarray(mid, np.float32))
    rc = dict(half=h, mid=m_)
    for i in range(n):
        Z0 = torch.as_tensor(J["Z0"][i])
        with torch.autocast("cpu", enabled=False):
            ra = (m_ + h * torch.tanh(Z0 / h))[:N8].numpy()
        if not np.abs(ra - J["ref_arm"][i]).max() <= ARM_TOL:
            p.append(f"вызов {i}: рука reference ≠ tanh(Z0)")
        g = np.where(J["grip_logit"][i] > 0, 1.0, -1.0)
        a = J["a_codec"][i]
        if not np.array_equal(a[:, 6], g):
            p.append(f"вызов {i}: схват ≠ знаку логита reference")
        if ((a[:, :6] < L - 1e-4) | (a[:, :6] > U + 1e-4)).any():
            p.append(f"вызов {i}: рука вне пределов")
        if kind == "R":
            if J["has_actor"][i] or not np.array_equal(a[:, :6],
                                                       J["ref_arm"][i]):
                p.append(f"вызов {i}: reference исполнил не свою руку")
            continue
        if not J["has_actor"][i]:
            p.append(f"вызов {i}: нет журнала actor")
            continue
        r = int(np.argmax(J["logits"][i]))
        if r != int(J["r"][i]) or not np.array_equal(J["eta"][i],
                                                     J["mu"][i][r]):
            p.append(f"вызов {i}: r/η ≠ детерминированному правилу")
            continue
        eta = J["eta"][i]
        if not eta.any():
            if not (J["zero"][i] and np.array_equal(a[:, :6],
                                                    J["ref_arm"][i])):
                p.append(f"вызов {i}: η = 0, но рука ≠ reference")
            continue
        rc["Z0"], rc["arm"] = Z0, torch.as_tensor(J["ref_arm"][i])
        if kind == "T":
            if not J["has_zhat"][i]:
                p.append(f"вызов {i}: нет Zhat")
                continue
            with torch.autocast("cpu", enabled=False):
                D = ((torch.as_tensor(J["Zhat"][i]) - Z0) / h)[:N8]
        elif kind == "P":
            av = ko.ALPHA * np.tanh(eta.astype(np.float64))
            D = torch.as_tensor((J["w"][i].astype(np.float64)[:, None] *
                                 (av / float(J["c"][i][r]))[None]).astype(
                np.float32))
        else:
            D = torch.as_tensor((ko.ALPHA * np.tanh(ko.ar_transform(eta)))
                                .astype(np.float32))
        if not np.array_equal(D.numpy(), J["D"][i]):
            p.append(f"вызов {i}: D ≠ пересборке")
        Du, _lim = ko.limit(torch, D)
        with torch.autocast("cpu", enabled=False):
            arm = (m_ + h * torch.tanh(Z0[:N8] / h + Du)).numpy()
        if not (np.array_equal(Du.numpy(), J["Du"][i]) and
                np.array_equal(arm, a[:, :6])):
            p.append(f"вызов {i}: ограничитель/рука ≠ пересборке")
    for e in sorted(set(J["ep"].tolist())):
        idx = np.nonzero(J["ep"] == e)[0]
        ex = np.concatenate([kp.scale_exec(J["a_codec"][i][None], q)[0]
                             [:int(J["k"][i])] for i in idx])
        if not np.array_equal(ex.astype(np.float32),
                              np.asarray(A_eps[e], np.float32)):
            p.append(f"эпизод {e}: исполненное ≠ масштабу записанных команд")
        if int(J["elapsed"][idx[0]]) != 0 or any(
                int(J["elapsed"][x]) + int(J["k"][x]) != int(J["elapsed"][y])
                for x, y in zip(idx[:-1], idx[1:])):
            p.append(f"эпизод {e}: нарушена последовательность вызовов")
        ds = int(done_steps[e])
        own = ds + 1 if ds >= 0 else len(A_eps[e])
        if len(A_eps[e]) != own:
            p.append(f"эпизод {e}: шагов {len(A_eps[e])} ≠ {own}")
    return p


def block_problems(path, plan, lab, task, block):
    if not os.path.exists(path):
        return ["нет блока"]
    try:
        d = json.load(open(path, encoding="utf-8"))
        npath = os.path.join(os.path.dirname(path), d["npz"])
        z = dict(np.load(npath))
    except Exception as e:                                   # noqa: BLE001
        return [f"не читается: {type(e).__name__}"]
    p = []
    if kcol.sha(npath) != d.get("npz_sha1"):
        p.append("npz изменён")
    kind, seed = parse_label(lab)
    want = plan["arms"][lab]
    for k_ in ("label", "seed", "plan_sha1", "contract", "actor_hash"):
        if d.get(k_) != want.get(k_):
            p.append(f"{k_} ≠ плану")
    if d.get("harness") != dict(kcol.ENV, sequential=True, B=1):
        p.append("обвязка не B = 1 последовательно")
    if int(d.get("task", -1)) != task or \
            d.get("task_description") != plan["tasks"][str(task)]:
        p.append("задача/описание")
    starts = list(range(block, block + 5))
    got = [int(e.get("init_state_id", -1)) for e in d.get("episodes") or []]
    if got != starts or d.get("starts") != starts:
        p.append(f"init {got} ≠ {starts}")
    if p:
        return p
    eps = d["episodes"]
    offs = np.asarray(z.get("ep_offsets", []))
    if offs.shape != (6,) or offs[0] != 0 or (np.diff(offs) < 0).any() or \
            offs[-1] != len(z["actions"]):
        return p + ["смещения эпизодов не покрывают действия"]
    A_eps = [z["actions"][offs[i]:offs[i + 1]] for i in range(5)]
    for i, e in enumerate(eps):
        if e["init_hash"] != plan["init_ref"].get(
                f"{task}:{e['init_state_id']}"):
            p.append(f"эпизод {i}: начальный хеш ≠ холодному эталону")
        if kcol.actions_sha(A_eps[i]) != e["actions_sha1"]:
            p.append(f"эпизод {i}: действия ≠ npz")
    J = {k_[2:]: v for k_, v in z.items() if k_.startswith("j_")}
    return p + journal_problems(J, A_eps, [e["done_step"] for e in eps],
                                plan["half"], plan["mid"],
                                np.asarray(plan["L"]), np.asarray(plan["U"]),
                                np.asarray(plan["scale_q"]), kind)


def write_block(path, d, J, A_eps):
    npz = os.path.basename(path)[:-5] + ".npz"
    npath = os.path.join(os.path.dirname(path), npz)
    offs = np.cumsum([0] + [len(a) for a in A_eps])
    A = np.concatenate(A_eps) if A_eps else np.zeros((0, 7), np.float32)
    with open(npath + ".tmp", "wb") as fh:
        np.savez_compressed(fh, actions=A.astype(np.float32), ep_offsets=offs,
                            **{f"j_{k}": v for k, v in J.items()})
    os.replace(npath + ".tmp", npath)
    d = dict(d, npz=npz, npz_sha1=kcol.sha(npath))
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(path + ".tmp", path)


def ep_paths(lab, task, init):
    d = os.path.join(eval_dir(lab), "ep")
    return os.path.join(d, f"t{task}_i{init}.json"), \
        os.path.join(d, f"t{task}_i{init}.npz")


def save_episode(lab, task, init, eid, want, ep, dim):
    jp, np_ = ep_paths(lab, task, init)
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    J = pack_journal([(0, ep["calls"])], dim)
    with open(np_ + ".tmp", "wb") as fh:
        np.savez_compressed(fh, actions=ep["actions"].astype(np.float32),
                            **{f"j_{k}": v for k, v in J.items()})
    os.replace(np_ + ".tmp", np_)
    rec = dict(eid=eid, label=want["label"], plan_sha1=want["plan_sha1"],
               actor_hash=want["actor_hash"], contract=want["contract"],
               phys=int(ep["physical"]), npz_sha1=kcol.sha(np_),
               summary=dict(init_state_id=int(init), **{
                   k: ep[k] for k in ("success", "success_step", "done_step",
                                      "own_steps", "steps", "physical",
                                      "init_hash", "init_hash_full",
                                      "actions_sha1")}))
    with open(jp + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, default=float)
    os.replace(jp + ".tmp", jp)
    return rec


def load_episode(lab, task, init, eid, want):
    jp, np_ = ep_paths(lab, task, init)
    if not (os.path.exists(jp) and os.path.exists(np_)):
        return None
    rec = json.load(open(jp, encoding="utf-8"))
    bad = [k for k, v in (("eid", eid), ("label", want["label"]),
                          ("plan_sha1", want["plan_sha1"]),
                          ("actor_hash", want["actor_hash"]),
                          ("contract", want["contract"]))
           if rec.get(k) != v]
    if bad or kcol.sha(np_) != rec.get("npz_sha1"):
        raise SystemExit(f"{jp}: сохранённый эпизод чужой/изменён ({bad})")
    z = dict(np.load(np_))
    return rec, z


def ledger_owner_problems(ledger, eid, phys, scope="sr"):
    dn = ledger.d["done"].get(eid)
    pd = ledger.d["pending"] or {}
    if dn is not None:
        return [] if dn["scope"] == scope and int(dn["actual"]) == int(
            phys) else [f"{eid}: учтён иначе ({dn})"]
    if pd.get("id") == eid and pd.get("scope") == scope:
        return []
    return [f"{eid}: результат не учтён в бюджете"]


def run_block(sim, io, arm, plan, lab, task, block, ledger, d, crash=None):
    """Эпизод: reserve -> исполнение -> атомарный файл эпизода -> commit;
    сохранённый эпизод переживает перезапуск (settle, без повтора)."""
    want = plan["arms"][lab]
    eps, Js, A_eps = [], [], []
    for i, init in enumerate(range(block, block + 5)):
        eid = f"sr/{lab}/t{task}_i{init}"
        got = load_episode(lab, task, init, eid, want)
        if got is None:
            if ledger.is_done(eid):
                raise SystemExit(f"{eid}: учтён, но результата нет")
            ledger.reserve("sr", eid)
            arm.episode = 10 ** 7 + init
            ep = kcol.run_episode(sim, io, arm, task, init)
            save_episode(lab, task, init, eid, want, ep, d)
            if crash == (task, init):
                raise KeyboardInterrupt("после записи эпизода")
            ledger.commit(eid, ep["physical"])
            got = load_episode(lab, task, init, eid, want)
        rec, z = got
        pr = ledger_owner_problems(ledger, eid, rec["phys"])
        if pr:
            raise SystemExit(pr[0])
        ledger.settle(eid, rec["phys"])
        eps.append(rec["summary"])
        J = {k[2:]: v for k, v in z.items() if k.startswith("j_")}
        J["ep"] = np.full(len(J["ep"]), i, np.int64)
        Js.append(J)
        A_eps.append(z["actions"])
    J = {k: np.concatenate([x[k] for x in Js]) for k in Js[0]}
    write_block(block_path(lab, task, block), dict(
        label=lab, seed=want["seed"], plan_sha1=want["plan_sha1"],
        contract=want["contract"], actor_hash=want["actor_hash"],
        task=int(task), task_description=plan["tasks"][str(task)],
        starts=list(range(block, block + 5)), episodes=eps,
        harness=dict(kcol.ENV, sequential=True, B=1)), J, A_eps)
    return eps


def outcomes(plan, lab, tasks=TASKS):
    out = {}
    for t in tasks:
        for b in BLOCKS:
            path = block_path(lab, t, b)
            pr = block_problems(path, plan, lab, t, b)
            if pr:
                raise SystemExit(f"{lab} t{t} i{b}: {pr[:2]}")
            for e in json.load(open(path, encoding="utf-8"))["episodes"]:
                key = (t, int(e["init_state_id"]))
                if key in out or not 30 <= key[1] < 45:
                    raise SystemExit(f"{lab}: повтор/старт вне 30…44 {key}")
                out[key] = bool(e["success"])
    if set(out) != {(t, i) for t in tasks for b in BLOCKS
                    for i in range(b, b + 5)}:
        raise SystemExit(f"{lab}: набор состояний неполон")
    return out


def run_eval(a):
    import k18_collector as k18c
    import k32_protocol as pc
    sim = k18c.Sim(1, "persistent")
    import torch
    import k29_policy as kp
    import k31_harness as kh31
    plan = pc.load_eval_plan_kind(a.kind, a.seed)
    lab = label(a.kind, a.seed)
    odir = eval_dir(lab)
    os.makedirs(os.path.join(odir, "quarantine"), exist_ok=True)
    todo = []
    for t in TASKS:
        for b in BLOCKS:
            path = block_path(lab, t, b)
            if os.path.exists(path):
                pr = block_problems(path, plan, lab, t, b)
                if not pr:
                    continue
                q = os.path.join(odir, "quarantine",
                                 time.strftime("%Y%m%dT%H%M%S"))
                os.makedirs(q, exist_ok=True)
                for f in (path, path[:-5] + ".npz"):
                    if os.path.exists(f):
                        shutil.move(f, q)
                raise SystemExit(f"{lab} t{t} i{b}: готовый блок не прошёл "
                                 f"проверку ({pr[:2]}) -> {q}")
            todo.append((t, b))
    if not todo:
        print(f"  {lab}: все 30 блоков готовы и проверены")
        sim.close()
        return 0
    try:
        mod, bar, M, dec = pc.load_source_dec(torch, plan, a.seed, a.device)
        arm, d = pc.make_eval_arm(torch, plan, a.kind, a.seed, mod, dec, M)
        io = kh31.IO(kp.Inputs(torch, bar["proc"], a.device))
        ledger = pc.ledger()
        for t, b in todo:
            t0 = time.time()
            eps = run_block(sim, io, arm, plan, lab, t, b, ledger, d)
            pr = block_problems(block_path(lab, t, b), plan, lab, t, b)
            if pr:
                raise SystemExit(f"{lab} t{t} i{b}: новый блок не прошёл "
                                 f"проверку: {pr[:3]}")
            print(f"  {lab} t{t} i{b}: успех "
                  f"{sum(e['success'] for e in eps)}/5, "
                  f"{(time.time() - t0) / 60:.1f} мин", flush=True)
        return 0
    finally:
        sim.close()


def selftest():
    import tempfile
    import torch
    import k21_model as km
    import k25_model as k25m
    import k29_model as k29m
    import k32_model as km32
    import k32_policy as kpol
    k29m.no_tf32(torch)
    M = k29m.build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    mod = k25m.build(torch)["K25"](km.fake_bar(torch), L, U, seed=0).eval()
    with torch.no_grad():
        g = torch.Generator().manual_seed(3)
        for p_ in list(mod.cond.parameters()) + list(
                mod.readout.parameters()):
            p_.add_(torch.randn(p_.shape, generator=g) * 0.05)
    for p_ in mod.parameters():
        p_.requires_grad_(False)
    dec = M["Decoder"](mod).freeze()
    H = mod.H
    norm = km32.fit_normalizer(np.random.default_rng(0).normal(
        size=(64, 6 * H + 72)), H)
    import k32_ppo as kppo

    class FIO(kcol.FakeIO):
        def prep(self, obs, desc, task):
            x = float(obs["state"][0, 0])
            bt = km.fake_batch(torch, B=1, seed=int(abs(x) * 1e3) % 13)
            s8 = np.zeros((1, 8))
            s8[0, :3] = x
            return bt, 1, s8
    io = FIO()
    import k31_readout as kr
    h, m = [v.numpy() for v in kr.build(torch)["half_mid"](dec._L, dec._U,
                                                         "cpu")]
    with tempfile.TemporaryDirectory() as td:
        cwd = os.getcwd()
        os.chdir(td)
        try:
            plan = dict(tasks={"0": "toy"}, half=h.tolist(), mid=m.tolist(),
                        L=L.tolist(), U=U.tolist(),
                        scale_q=io.max_act_q.tolist(), init_ref={}, arms={})
            for kind in ("R", "T", "P", "F"):
                d = ko.DIM.get(kind, 6)
                actor = None
                if kind != "R":
                    actor, _c = km32.build(torch, H, d, 0, norm)
                    with torch.no_grad():                # ненулевой actor
                        g2 = torch.Generator().manual_seed(7)
                        for p_ in actor.out.parameters():
                            p_.copy_(torch.randn(p_.shape, generator=g2))
                arm = kpol.Arm(torch, mod, dec, M, kind=kind, seed=0,
                               actor=actor)
                lab = label(kind, 0)
                plan["arms"][lab] = dict(
                    label=lab, seed=0, plan_sha1="P", contract="C",
                    actor_hash=None if actor is None else
                    kppo.tensor_hash(actor.state_dict()))
                os.makedirs(eval_dir(lab), exist_ok=True)
                L_ = kcol.Ledger("ledger.json", dict(sr=732000))
                eps = run_block(kppo.ToySim(), io, arm, plan, lab, 0, 30, L_,
                                d)
                for e in eps:
                    plan["init_ref"][f"0:{e['init_state_id']}"] = \
                        e["init_hash"]
                pr = block_problems(block_path(lab, 0, 30), plan, lab, 0, 30)
                assert pr == [], (kind, pr)
                if kind == "R":
                    continue
                path = block_path(lab, 0, 30)
                dj = json.load(open(path))
                z = dict(np.load(os.path.join(eval_dir(lab), dj["npz"])))
                assert z["j_has_actor"].all() and not z["j_zero"].all()

                def mutate(f):
                    zz = {k_: v.copy() for k_, v in z.items()}
                    f(zz)
                    with open(os.path.join(eval_dir(lab), dj["npz"]),
                              "wb") as fh:
                        np.savez_compressed(fh, **zz)
                    json.dump(dict(dj, npz_sha1=kcol.sha(os.path.join(
                        eval_dir(lab), dj["npz"]))), open(path, "w"))
                    return block_problems(path, plan, lab, 0, 30)
                for nm, f in (
                        ("NaN μ", lambda zz: zz["j_mu"].__setitem__(
                            (0, 0, 0), np.nan)),
                        ("+Inf D", lambda zz: zz["j_D"].__setitem__(
                            (0, 0, 0), np.inf)),
                        ("η", lambda zz: zz["j_eta"].__setitem__(
                            (0, 0), zz["j_eta"][0, 0] + 1e-3)),
                        ("Du", lambda zz: zz["j_Du"].__setitem__(
                            (0, 0, 0), zz["j_Du"][0, 0, 0] * 2 + 1e-3)),
                        ("рука", lambda zz: zz["j_a_codec"].__setitem__(
                            (0, 0, 0), zz["j_a_codec"][0, 0, 0] + 1e-5)),
                        ("схват", lambda zz: zz["j_a_codec"].__setitem__(
                            (0, 0, 6), -zz["j_a_codec"][0, 0, 6])),
                        ("действие", lambda zz: zz["actions"].__setitem__(
                            (2, 0), zz["actions"][2, 0] + 1e-6))):
                    assert mutate(f), (kind, nm)
                ne = int((z["j_ep"] == 4).sum())
                for nm, f in (
                        ("пустой журнал", lambda zz: [zz.__setitem__(k_, v[:0])
                                                      for k_, v in list(
                                                          zz.items())
                                                      if k_.startswith("j_")]),
                        ("нет эпизода 4", lambda zz: [zz.__setitem__(
                            k_, v[z["j_ep"] != 4]) for k_, v in list(
                                zz.items()) if k_.startswith("j_")]),
                        ("нет последнего вызова", lambda zz: [zz.__setitem__(
                            k_, v[:-1]) for k_, v in list(zz.items())
                            if k_.startswith("j_")]),
                        ("нет поля", lambda zz: zz.pop("j_Du")),
                        ("смещения", lambda zz: zz.__setitem__(
                            "ep_offsets", zz["ep_offsets"] - 1))):
                    assert mutate(f), (kind, nm)
                assert ne >= 1
                mutate(lambda zz: None)
                assert block_problems(path, plan, lab, 0, 30) == []
            json.dump(dict(dj, harness=dict(dj["harness"], n_envs=5)),
                      open(path, "w"))
            assert any("обвязка" in x for x in block_problems(
                path, plan, lab, 0, 30))
            # авария SR между записью эпизода и commit: повтор без исполнения
            labc = label("T", 1)
            plan["arms"][labc] = dict(plan["arms"][lab], label=labc, seed=1)
            os.makedirs(eval_dir(labc), exist_ok=True)
            Lc = kcol.Ledger("ledger_c.json", dict(sr=732000))
            try:
                run_block(kppo.ToySim(), io, arm, plan, labc, 0, 30, Lc, d,
                          crash=(0, 32))
                raise AssertionError("авария SR не сработала")
            except KeyboardInterrupt:
                pass
            eps_c = run_block(kppo.ToySim(), io, arm, plan, labc, 0, 30,
                              kcol.Ledger("ledger_c.json", dict(sr=732000)),
                              d)
            Lc2 = kcol.Ledger("ledger_c.json", dict(sr=732000))
            assert Lc2.spent("sr") == sum(e["physical"] for e in eps_c) and \
                not Lc2.d["crashes"], "SR задвоен"
        finally:
            os.chdir(cwd)
    print("самопроверка k32_harness пройдена: блоки R/T/P/F (B = 1, "
          "ненулевой actor) проходят независимую пересборку; NaN/±Inf, "
          "подмена η, Du, руки, схвата, физического действия, обвязка "
          "n_envs = 5, пустой журнал, нет эпизода, нет последнего вызова, нет "
          "поля, смещения — отказ; авария SR между записью эпизода и commit "
          "— эпизод сохранён, расход один раз")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-32: SR B = 1")
    ap.add_argument("cmd", choices=("eval",))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--kind", choices=tuple(KINDS), required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    try:
        return run_eval(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
