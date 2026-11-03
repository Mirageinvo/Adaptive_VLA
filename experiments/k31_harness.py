#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-31: SR-оценка B = 1 — блоки по 5 последовательных эпизодов.

Руки: parents_ref31_s<S>, parents_repair31_s<S>, parents_flatrepair31_s<S>
(один источник K-25 сида; repair/flatrepair — один общий селектор сида).
Эпизод — k31_collect.run_episode (свежая среда, ожидание, защёлка, 600
шагов) с k31_policy.Arm. Блок (reports/k31/eval/<label>/t<T>_i<B>.json +
npz) пишется только целиком; обвязка записывается явно: n_envs = 1,
последовательно, B = 1.
Проверка блока (CPU): метка/сид/план/источник/селектор/контракт; init
b…b+4 по порядку; описание задачи; начальный хеш каждого эпизода =
холодному эталону; действия npz = action_sha1 и = масштабу записанных
команд побитово; журнал вызовов: формы и конечность всех групп (NaN
допустим только как маркер недопустимой строки пула — ровно там);
схват = знак логита reference; reference-рука = mid + half·tanh(Z0/half)
(≤ 1e−6); строки пула = mid + half·tanh(Z0/half + Du) побитово, RMS Du
префикса = 0.02 (1e−6), λ ≤ 8, max|Du| ≤ 0.08, m одинаков; селектор
переигрывается на CPU (batch 7) и правило выбора пересчитывается целиком;
не более одного исполненного ремонта на эпизод, после него — reference.
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

import k31_candidates as kc  # noqa: E402
import k31_collect as kcol  # noqa: E402

TASKS = tuple(range(10))
BLOCKS = (30, 35, 40)
PRIMARY = (0, 6, 8)
KINDS = ("ref", "repair", "flatrepair")
MODE = dict(ref="reference", repair="repair", flatrepair="flatrepair")
ARM_TOL = 1e-6
CALL_KEYS = ("ep", "call", "elapsed", "k", "used_before", "used_after",
             "chosen", "a_codec", "Z0", "ref_arm", "grip_logit", "has_pool")
POOL_KEYS = ("valid", "allowed", "m", "lam", "Du", "pool_arm", "X",
             "sel_logits", "scores")


def label(kind, seed):
    return f"parents_{kind}31_s{seed}"


def parse_label(lab):
    for k in KINDS:
        pre = f"parents_{k}31_s"
        if lab.startswith(pre):
            return k, int(lab[len(pre):])
    raise ValueError(lab)


def eval_dir(lab):
    return f"reports/k31/eval/{lab}"


def block_path(lab, task, block):
    return os.path.join(eval_dir(lab), f"t{task}_i{block}.json")


class IO:
    """Адаптер k29_policy.Inputs к интерфейсу k31_collect."""

    def __init__(self, inputs):
        self.inp = inputs

    def seed(self, s):
        self.inp.seed(s)

    def task_desc(self, t):
        return self.inp.task_desc(t)

    def prep(self, obs, desc, task):
        return self.inp(obs, desc, task)

    def scale(self, a):
        return self.inp.scale(a)


def pack_journal(episodes_calls):
    """[(ep, calls)] -> dict массивов журнала (фиксированные формы)."""
    rows = [(ep, c) for ep, calls in episodes_calls for c in calls]
    n = len(rows)
    J = dict(ep=np.zeros(n, np.int64), call=np.zeros(n, np.int64),
             elapsed=np.zeros(n, np.int64), k=np.zeros(n, np.int64),
             used_before=np.zeros(n, bool), used_after=np.zeros(n, bool),
             chosen=np.zeros(n, np.int64),
             a_codec=np.zeros((n, kc.N8, 7), np.float32),
             Z0=np.zeros((n, kc.T, 6), np.float32),
             ref_arm=np.zeros((n, kc.N8, 6), np.float32),
             grip_logit=np.zeros((n, kc.N8), np.float32),
             has_pool=np.zeros(n, bool),
             valid=np.zeros((n, 7), bool), allowed=np.zeros((n, 7), bool),
             m=np.zeros(n, np.int64), lam=np.full((n, 7), np.nan),
             Du=np.full((n, 7, kc.T, 6), np.nan, np.float32),
             pool_arm=np.full((n, 7, kc.N8, 6), np.nan, np.float32),
             X=np.zeros((n, 7, kc.N_FEAT), np.float32),
             sel_logits=np.zeros((n, 7, 3), np.float32),
             scores=np.zeros((n, 7), np.float64))
    for i, (ep, c) in enumerate(rows):
        r = c["rec"]
        J["ep"][i], J["call"][i], J["elapsed"][i] = ep, r["call"], \
            r["elapsed"]
        J["k"][i], J["used_before"][i] = r["k"], r["used_before"]
        J["used_after"][i], J["chosen"][i] = r["used_after"], r["chosen"]
        J["a_codec"][i], J["Z0"][i] = r["a_codec"], r["Z0"]
        J["ref_arm"][i], J["grip_logit"][i] = r["ref_arm"], r["grip_logit"]
        if "X" in r:
            J["has_pool"][i] = True
            for k_ in POOL_KEYS:
                J[k_][i] = r[k_]
    return J


def journal_problems(J, A_eps, done_steps, half, mid, L, U, scale_q,
                     selector, family):
    """J — журнал блока; A_eps — исполненные действия эпизодов; selector —
    объект scores(X) или None (reference)."""
    import torch
    p = []
    n = len(J["ep"])
    shapes = dict(a_codec=(n, kc.N8, 7), Z0=(n, kc.T, 6),
                  ref_arm=(n, kc.N8, 6), grip_logit=(n, kc.N8),
                  Du=(n, 7, kc.T, 6), pool_arm=(n, 7, kc.N8, 6),
                  X=(n, 7, kc.N_FEAT), lam=(n, 7), scores=(n, 7))
    for k_, s in shapes.items():
        if J[k_].shape != s:
            return [f"журнал {k_} формы {J[k_].shape}"]
    for k_ in ("a_codec", "Z0", "ref_arm", "grip_logit", "X", "sel_logits"):
        if not np.isfinite(np.asarray(J[k_], np.float64)).all():
            return [f"журнал {k_}: нечисловые значения"]
    hp, al = J["has_pool"], J["allowed"]
    for k_, cell in (("Du", (kc.T, 6)), ("pool_arm", (kc.N8, 6))):
        fin = np.isfinite(J[k_]).reshape(n, 7, -1)
        want = np.zeros((n, 7), bool)
        want[hp] = al[hp]
        if k_ == "pool_arm":
            want[:, 0] = False
        else:
            want[hp, 0] = True
        if not (np.array_equal(fin.all(-1), want) and
                np.array_equal(fin.any(-1), want)):
            return [f"журнал {k_}: NaN/±Inf вне недопустимых строк"]
    lam_f = np.isfinite(J["lam"])
    if (lam_f & ~(al & hp[:, None])).any() or \
            (~lam_f[:, 1:] & (al & hp[:, None])[:, 1:]).any():
        return ["журнал λ: нечисловые значения у разрешённых строк"]
    sc = np.where(hp[:, None] & al, J["scores"], 0.0)
    if not np.isfinite(sc).all():
        return ["журнал score: нечисловые значения"]
    h = torch.as_tensor(np.asarray(half, np.float32))
    m_ = torch.as_tensor(np.asarray(mid, np.float32))
    for i in range(n):
        Z0 = torch.as_tensor(J["Z0"][i])
        with torch.autocast("cpu", enabled=False):
            ra = (m_ + h * torch.tanh(Z0 / h))[:kc.N8].numpy()
        if not np.abs(ra - J["ref_arm"][i]).max() <= ARM_TOL:
            p.append(f"вызов {i}: рука reference ≠ tanh(Z0)")
        g = np.where(J["grip_logit"][i] > 0, 1.0, -1.0)
        if not np.array_equal(J["a_codec"][i][:, 6], g):
            p.append(f"вызов {i}: схват ≠ знаку логита reference")
        a = J["a_codec"][i][:, :6]
        if ((a < L - 1e-4) | (a > U + 1e-4)).any():
            p.append(f"вызов {i}: рука вне пределов")
        ch = int(J["chosen"][i])
        if not hp[i]:
            if ch or not np.array_equal(a, J["ref_arm"][i]):
                p.append(f"вызов {i}: без пула исполнено не reference")
            continue
        if selector is None:
            p.append(f"вызов {i}: пул у reference")
            continue
        vt = [j for j in kc.TREE_ROWS if J["valid"][i, j]]
        vf = [j for j in kc.FLAT_ROWS if J["valid"][i, j]]
        m = min(len(vt), len(vf))
        want = np.zeros(7, bool)
        want[0] = True
        want[vt[:m]] = True
        want[vf[:m]] = True
        if not np.array_equal(want, al[i]) or m != int(J["m"][i]):
            p.append(f"вызов {i}: m/разрешённые строки ≠ правилу")
        for j in range(1, 7):
            if not al[i, j]:
                continue
            Du = torch.as_tensor(J["Du"][i, j])
            rms = float(torch.sqrt(torch.mean(Du[:kc.N8] ** 2)))
            if not abs(rms - kc.TARGET_RMS) <= kc.RMS_TOL or \
                    not float(J["lam"][i, j]) <= kc.LAMBDA_MAX or \
                    not float(Du[:kc.N8].abs().max()) <= kc.DU_MAX:
                p.append(f"вызов {i} строка {j}: нормировка")
            with torch.autocast("cpu", enabled=False):
                arm = (m_ + h * torch.tanh(Z0 / h + Du))[:kc.N8].numpy()
            if not np.array_equal(arm, J["pool_arm"][i, j]):
                p.append(f"вызов {i} строка {j}: рука пула ≠ пересборке")
        lg, s = selector.scores(J["X"][i])
        s = np.where(al[i], s, -np.inf)
        s[0] = 0.0
        if not np.array_equal(np.asarray(lg, np.float32), J["sel_logits"][i]):
            p.append(f"вызов {i}: логиты селектора ≠ CPU-повтору")
        if kc.choose(s, al[i], family) != ch:
            p.append(f"вызов {i}: выбор ≠ правилу")
        want_arm = J["ref_arm"][i] if ch == 0 else J["pool_arm"][i, ch]
        if not np.array_equal(a, want_arm):
            p.append(f"вызов {i}: исполнена не выбранная рука")
    for e in sorted(set(J["ep"].tolist())):
        idx = np.nonzero(J["ep"] == e)[0]
        rep = [i for i in idx if J["chosen"][i] and J["k"][i] >= 1]
        if len(rep) > 1:
            p.append(f"эпизод {e}: больше одного ремонта")
        if rep:
            after = [i for i in idx if i > rep[0]]
            if any(J["has_pool"][i] or J["chosen"][i] for i in after):
                p.append(f"эпизод {e}: после ремонта не только reference")
            if not all(J["used_before"][i] for i in after):
                p.append(f"эпизод {e}: флаг ремонта сброшен внутри эпизода")
        if J["used_before"][idx[0]]:
            p.append(f"эпизод {e}: флаг ремонта унаследован")
        A = np.asarray(A_eps[e], np.float32)
        ex = []
        for i in idx:
            import k29_policy as kp
            ex.append(kp.scale_exec(J["a_codec"][i][None], scale_q)[0]
                      [:int(J["k"][i])])
        ex = np.concatenate(ex) if ex else np.zeros((0, 7), np.float32)
        if not np.array_equal(ex.astype(np.float32), A):
            p.append(f"эпизод {e}: исполненное ≠ масштабу записанных команд")
        if int(J["elapsed"][idx[0]]) != 0 or any(
                int(J["elapsed"][a_]) + int(J["k"][a_]) !=
                int(J["elapsed"][b_]) for a_, b_ in zip(idx[:-1], idx[1:])):
            p.append(f"эпизод {e}: нарушена последовательность вызовов")
        ds = int(done_steps[e])
        own = ds + 1 if ds >= 0 else len(A)
        if len(A) != own:
            p.append(f"эпизод {e}: шагов {len(A)} ≠ собственным {own}")
    return p


def block_problems(path, plan, lab, task, block, selector=None):
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
    for k_ in ("label", "seed", "plan_sha1", "contract"):
        if d.get(k_) != want[k_]:
            p.append(f"{k_} ≠ плану")
    if d.get("harness") != dict(kcol.ENV, sequential=True, B=1):
        p.append("обвязка не B = 1 последовательно")
    if int(d.get("task", -1)) != task or \
            d.get("task_description") != plan["tasks"][str(task)]:
        p.append("задача/описание")
    starts = list(range(block, block + 5))
    got = [int(e.get("init_state_id", -1)) for e in d.get("episodes") or []]
    if got != starts or d.get("starts") != starts:
        p.append(f"init эпизодов {got} ≠ {starts}")
    if p:
        return p
    eps = d["episodes"]
    offs = np.asarray(z["ep_offsets"])
    A = z["actions"]
    A_eps = [A[offs[i]:offs[i + 1]] for i in range(5)]
    for i, e in enumerate(eps):
        if e["init_hash"] != plan["init_ref"].get(f"{task}:{e['init_state_id']}"):
            p.append(f"эпизод {i}: начальный хеш ≠ холодному эталону")
        if kcol.actions_sha(A_eps[i]) != e["actions_sha1"]:
            p.append(f"эпизод {i}: действия ≠ npz")
    J = {k_: z[f"j_{k_}"] for k_ in CALL_KEYS + POOL_KEYS}
    if selector is None and kind != "ref":
        return p + ["нет селектора для проверки"]
    p += journal_problems(J, A_eps, [e["done_step"] for e in eps],
                          plan["half"], plan["mid"],
                          np.asarray(plan["L"]), np.asarray(plan["U"]),
                          np.asarray(plan["scale_q"]),
                          selector if kind != "ref" else None,
                          "tree" if kind == "repair" else "flat")
    return p


def write_block(path, d, J, A_eps):
    npz = os.path.basename(path)[:-5] + ".npz"
    npath = os.path.join(os.path.dirname(path), npz)
    offs = np.cumsum([0] + [len(a) for a in A_eps])
    A = np.concatenate(A_eps) if A_eps else np.zeros((0, 7), np.float32)
    with open(npath + ".tmp", "wb") as fh:
        np.savez_compressed(fh, actions=A.astype(np.float32),
                            ep_offsets=offs,
                            **{f"j_{k}": v for k, v in J.items()})
    os.replace(npath + ".tmp", npath)
    d = dict(d, npz=npz, npz_sha1=kcol.sha(npath))
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(path + ".tmp", path)


def run_block(sim, io, arm, plan, lab, task, block, ledger):
    eps, ec, A_eps = [], [], []
    for i, init in enumerate(range(block, block + 5)):
        eid = f"sr/{lab}/t{task}_i{init}"
        if ledger.is_done(eid):
            eid = f"{eid}@{time.strftime('%Y%m%dT%H%M%S')}"
        ledger.reserve("sr", eid)
        ep = kcol.run_episode(sim, io, arm, task, init)
        ledger.commit(eid, ep["physical"])
        eps.append(dict(init_state_id=int(init), **{
            k: ep[k] for k in ("success", "success_step", "done_step",
                               "own_steps", "steps", "physical", "init_hash",
                               "init_hash_full", "actions_sha1")},
            repairs=int(sum(1 for c in ep["calls"]
                            if c["rec"]["chosen"] and c["k"] >= 1)),
            repair_rows=[c["rec"]["chosen_name"] for c in ep["calls"]
                         if c["rec"]["chosen"] and c["k"] >= 1]))
        ec.append((i, ep["calls"]))
        A_eps.append(ep["actions"])
    J = pack_journal(ec)
    want = plan["arms"][lab]
    d = dict(label=lab, seed=want["seed"], plan_sha1=want["plan_sha1"],
             contract=want["contract"], task=int(task),
             task_description=plan["tasks"][str(task)],
             starts=list(range(block, block + 5)), episodes=eps,
             harness=dict(kcol.ENV, sequential=True, B=1))
    write_block(block_path(lab, task, block), d, J, A_eps)
    return eps


def outcomes(plan, lab, selector=None, tasks=TASKS):
    out = {}
    for t in tasks:
        for b in BLOCKS:
            path = block_path(lab, t, b)
            pr = block_problems(path, plan, lab, t, b, selector)
            if pr:
                raise SystemExit(f"{lab} t{t} i{b}: {pr[:2]}")
            for e in json.load(open(path, encoding="utf-8"))["episodes"]:
                key = (t, int(e["init_state_id"]))
                if key in out:
                    raise SystemExit(f"{lab}: повтор состояния {key}")
                if not 30 <= key[1] < 45:
                    raise SystemExit(f"{lab}: старт {key} вне 30…44")
                out[key] = bool(e["success"])
    want = {(t, i) for t in tasks for b in BLOCKS for i in range(b, b + 5)}
    if set(out) != want:
        raise SystemExit(f"{lab}: набор состояний ≠ {len(want)}")
    return out


def run_eval(a):
    import k18_collector as k18c
    import k31_protocol as pc
    sim = k18c.Sim(1, "persistent")
    import torch
    import k29_policy as kp
    plan = pc.load_eval_plan(a.seed)
    lab = label(a.kind, a.seed)
    odir = eval_dir(lab)
    os.makedirs(os.path.join(odir, "quarantine"), exist_ok=True)
    sel = pc.load_selector(torch, plan, a.seed) if a.kind != "ref" else None
    todo = []
    for t in TASKS:
        for b in BLOCKS:
            path = block_path(lab, t, b)
            if os.path.exists(path):
                pr = block_problems(path, plan, lab, t, b, sel)
                if not pr:
                    continue
                q = os.path.join(odir, "quarantine",
                                 time.strftime("%Y%m%dT%H%M%S"))
                os.makedirs(q, exist_ok=True)
                for f in (path, path[:-5] + ".npz"):
                    if os.path.exists(f):
                        shutil.move(f, q)
                print(f"  {lab} t{t} i{b}: в карантин ({pr[:2]})",
                      flush=True)
                raise SystemExit(f"{lab} t{t} i{b}: готовый блок не прошёл "
                                 f"проверку -> {q}")
            todo.append((t, b))
    if not todo:
        print(f"  {lab}: все 30 блоков готовы и проверены")
        sim.close()
        return 0
    try:
        mod, bar, M, dec = pc.load_source_dec(torch, plan, a.seed, a.device)
        import k31_policy as kpol
        arm = kpol.Arm(torch, mod, dec, M, mode=MODE[a.kind],
                       source_seed=a.seed, space=0, selector=sel)
        io = IO(kp.Inputs(torch, bar["proc"], a.device))
        ledger = pc.ledger()
        for t, b in todo:
            t0 = time.time()
            eps = run_block(sim, io, arm, plan, lab, t, b, ledger)
            pr = block_problems(block_path(lab, t, b), plan, lab, t, b, sel)
            if pr:
                raise SystemExit(f"{lab} t{t} i{b}: новый блок не прошёл "
                                 f"проверку: {pr[:3]}")
            print(f"  {lab} t{t} i{b}: успех "
                  f"{sum(e['success'] for e in eps)}/5, ремонтов "
                  f"{sum(e['repairs'] for e in eps)}, "
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
    import k29_policy as kp
    import k31_policy as kpol
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

    class Sel:
        def ctx(self, mean, pth):
            return np.zeros(32, np.float32)

        def scores(self, X):
            lg = np.zeros((7, 3), np.float32)
            lg[:, 2] = X[:, 92] * 50.0
            s = (np.exp(lg[:, 2]) - 1) / (np.exp(lg[:, 2]) + 2)
            s = s.astype(np.float64)
            s[0] = 0.0
            return lg, s

    class FIO(kcol.FakeIO):
        def prep(self, obs, desc, task):
            x = float(obs["state"][0, 0])
            bt = km.fake_batch(torch, B=1, seed=int(abs(x) * 1e4) % 97)
            s8 = np.zeros((1, 8))
            s8[0, :3] = x
            return bt, 1, s8
    sel = Sel()
    io = FIO()
    q = io.max_act_q
    E = __import__("k31_readout").build(torch)
    half, mid = [v.numpy() for v in E["half_mid"](dec._L, dec._U, "cpu")]
    caps = dict(sr=549000)
    with tempfile.TemporaryDirectory() as td:
        cwd = os.getcwd()
        os.chdir(td)
        try:
            plan = dict(tasks={"0": "toy"}, half=half.tolist(),
                        mid=mid.tolist(), L=L.tolist(), U=U.tolist(),
                        scale_q=q.tolist(),
                        init_ref={f"0:{i}": None for i in range(30, 35)},
                        arms={})
            res = {}
            for kind in ("ref", "repair"):
                lab = label(kind, 0)
                plan["arms"][lab] = dict(label=lab, seed=0, plan_sha1="P",
                                         contract="C")
                os.makedirs(eval_dir(lab), exist_ok=True)
                arm = kpol.Arm(torch, mod, dec, M, mode=MODE[kind],
                               source_seed=0,
                               selector=sel if kind != "ref" else None)
                ledger = kcol.Ledger("ledger.json", caps)
                eps = run_block(kcol.FakeSim(), io, arm, plan, lab, 0, 30,
                                ledger)
                for e in eps:
                    plan["init_ref"][f"0:{e['init_state_id']}"] = \
                        e["init_hash"]
                res[kind] = eps
            for kind in ("ref", "repair"):
                lab = label(kind, 0)
                pr = block_problems(block_path(lab, 0, 30), plan, lab, 0, 30,
                                    sel if kind != "ref" else None)
                assert pr == [], (kind, pr)
            assert all(e["repairs"] <= 1 for e in res["repair"])
            assert any(e["repairs"] == 1 for e in res["repair"])
            lab = label("repair", 0)
            path = block_path(lab, 0, 30)
            d = json.load(open(path))
            z = dict(np.load(os.path.join(eval_dir(lab), d["npz"])))

            def mutate(f):
                zz = {k: v.copy() for k, v in z.items()}
                f(zz)
                with open(os.path.join(eval_dir(lab), d["npz"]), "wb") as fh:
                    np.savez_compressed(fh, **zz)
                d2 = dict(d, npz_sha1=kcol.sha(os.path.join(eval_dir(lab),
                                                            d["npz"])))
                json.dump(d2, open(path, "w"))
                return block_problems(path, plan, lab, 0, 30, sel)
            i_rep = int(np.nonzero(z["j_chosen"])[0][0])
            for nm, f in (
                    ("NaN в X", lambda zz: zz["j_X"].__setitem__(
                        (0, 1, 3), np.nan)),
                    ("+Inf в Z0", lambda zz: zz["j_Z0"].__setitem__(
                        (0, 0, 0), np.inf)),
                    ("−Inf в рука пула", lambda zz: zz["j_pool_arm"]
                     .__setitem__((i_rep, int(z["j_chosen"][i_rep]), 0, 0),
                                  -np.inf)),
                    ("схват", lambda zz: zz["j_a_codec"].__setitem__(
                        (0, 0, 6), -zz["j_a_codec"][0, 0, 6])),
                    ("выбор", lambda zz: zz["j_chosen"].__setitem__(
                        i_rep, 0)),
                    ("второй ремонт", lambda zz: (
                        zz["j_chosen"].__setitem__(i_rep + 1, 1),
                        zz["j_k"].__setitem__(i_rep + 1, 8))),
                    ("действия", lambda zz: zz["actions"].__setitem__(
                        (3, 0), zz["actions"][3, 0] + 1e-6))):
                assert mutate(f), nm
            mutate(lambda zz: None)
            assert block_problems(path, plan, lab, 0, 30, sel) == []
            json.dump(dict(d, harness=dict(d["harness"], n_envs=5)),
                      open(path, "w"))
            assert any("обвязка" in x for x in block_problems(
                path, plan, lab, 0, 30, sel))
            json.dump(dict(d, starts=list(range(31, 36))), open(path, "w"))
            assert block_problems(path, plan, lab, 0, 30, sel)
            pl2 = dict(plan, init_ref=dict(plan["init_ref"], **{"0:32": "x"}))
            json.dump(d, open(path, "w"))
            assert any("хеш" in x for x in block_problems(
                path, pl2, lab, 0, 30, sel))
        finally:
            os.chdir(cwd)
    assert kp.scale_exec is not None
    print("самопроверка k31_harness пройдена: блоки reference и repair (B = "
          "1, 5 последовательных эпизодов) проходят независимую проверку; "
          "не больше одного ремонта; NaN/±Inf в журнале, схват, выбор, "
          "второй ремонт, действия, обвязка n_envs = 5, старты, начальный "
          "хеш — отказ")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-31: SR B = 1")
    ap.add_argument("cmd", choices=("eval",))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--kind", choices=KINDS, required=True)
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
