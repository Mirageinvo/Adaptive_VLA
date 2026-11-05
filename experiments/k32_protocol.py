#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-32: base_prepare -> normalizer -> контракт; real-check, collector-smoke,
обучение T/P/F × seed, планы SR.

prepare — reports/k32/base_prepare.json (неизменяем; сверка в JSON-форме,
  конфликт — отказ до записи): источники K-25 s0/s1 (шаг best из отчёта),
  BAR, пределы, банки, описания 10 задач (план A = LIBERO), холодный эталон
  30…44 (K-25), порядок 1024 строк K-21 для нормировки (train, баланс
  задача -> эпизод -> строка; метки задач — текст), конфигурации dist /
  operator / model / PPO, бюджеты, статистический протокол, SHA кода.
normalizer DEV — data/k32/normalizer_s<S>.npz по base_prepare (elapsed = 0
  на демонстрационных строках) + reports/k32/normalizer_manifest.json.
  base_prepare не переписывается.
Контракт = отпечаток base_prepare + SHA обоих normalizer и манифеста.
real-check DEV, collector-smoke DEV, train DEV, plan-eval (по руке) — см.
функции. Модели источников — последовательно.
"""
import argparse
import datetime
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k31_collect as kcol  # noqa: E402
import k32_collect as kcol32  # noqa: E402
import k32_dist as kd  # noqa: E402
import k32_operator as ko  # noqa: E402
import k32_ppo as kppo  # noqa: E402

BASE = "reports/k32/base_prepare.json"
NORM_MAN = "reports/k32/normalizer_manifest.json"
REAL_CHECK = "reports/k32/real_check.json"
SMOKE = "reports/k32/smoke.json"
LEDGER = "data/k32/ledger.json"
SEEDS = (0, 1)
OPS = ("T", "P", "F")
ORDER = [(op, s) for s in SEEDS for op in OPS]
N_NORM = 1024
CAPS = dict({f"train_{o}_s{s}": kppo.CAP for o in OPS for s in SEEDS},
            **{f"val_{o}_s{s}": kppo.VAL_CAP for o in OPS for s in SEEDS},
            sr=4 * 2 * 150 * 610, checks=50_000)
SMOKE_EPS = ((0, 0), (0, 1), (8, 0), (8, 1))
STATS = dict(holm=6, boot=dict(n=10000, seed=32005), ni_margin=0.03,
             primary=[0, 6, 8], comparisons=["T-R", "T-P", "T-F"])
Z_TOL, ARM_TOL, DU_TOL = 1e-5, 1e-6, 1e-5
MODEL_CODE = ("k32_dist.py", "k32_operator.py", "k32_model.py",
              "k32_policy.py", "k32_collect.py", "k32_ppo.py",
              "k32_harness.py", "k32_protocol.py", "k31_readout.py",
              "k31_collect.py", "k31_harness.py", "k31_protocol.py",
              "k31_analyze.py", "k32_analyze.py",
              "k29_model.py", "k29_policy.py", "k29_dist.py",
              "k25_readout.py", "k25_model.py", "k25_policy.py",
              "k22_model.py", "k21_model.py", "k21_tree.py", "k21_data.py",
              "k21_train.py", "k20_data.py", "k18_collector.py",
              "k16_models.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def fp(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()
                        ).hexdigest()


def jnorm(d):
    return json.loads(json.dumps(d, default=str))


def save_json(path, d):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(path + ".tmp", path)


def free():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- base_prepare -----------------------------------------------------------------
def norm_rows(D, n=N_NORM, seed=32007):
    """Детерминированно n уникальных train-строк, баланс задача -> эпизод ->
    строка (метки задач — как есть, без приведения к int)."""
    rows = np.asarray(D["train_rows"], np.int64)
    task = np.asarray(D["task"])[rows]
    epi = np.asarray(D["episode"])[rows]
    rng = np.random.default_rng(seed)
    by = {}
    for r, t, e in zip(rows, task, epi):
        by.setdefault(str(t), {}).setdefault(int(e), []).append(int(r))
    tasks = sorted(by)
    eps = {t: [int(x) for x in rng.permutation(sorted(by[t]))] for t in tasks}
    pools = {t: {e: [int(x) for x in rng.permutation(by[t][e])]
                 for e in by[t]} for t in tasks}
    out, ei = [], {t: 0 for t in tasks}
    while len(out) < n:
        progress = False
        for t in tasks:
            for _ in range(len(eps[t])):
                e = eps[t][ei[t] % len(eps[t])]
                ei[t] += 1
                if pools[t][e]:
                    out.append(pools[t][e].pop())
                    progress = True
                    break
            if len(out) >= n:
                break
        if not progress:
            raise SystemExit("строк K-21 меньше 1024")
    if len(set(out)) != len(out):
        raise SystemExit("строки нормировки не уникальны")
    return out


def make_base():
    import k21_data as d21
    import k29_model as km
    import k29_policy as kp
    import k31_protocol as p31
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    srcs = {}
    for s in SEEDS:
        e = km.source_entry(s)
        tr = json.load(open(e["report"], encoding="utf-8"))
        if e["prep_id"] != prep["prep_id"] or tr.get("best_update") is None \
                or int(tr["best_update"]) != int(e["update"]):
            raise SystemExit(f"seed {s}: источник/шаг best ≠ отчёту K-25")
        srcs[str(s)] = e
    import k23_protocol as p23
    rA = p23.ref_k21_main()
    desc = {}
    for (l_, _s, t_, _b), f in rA["blocks"].items():
        if l_ == "fullbar":
            desc.setdefault(int(t_), json.load(open(f, encoding="utf-8"))
                            .get("task_description"))
    from libero.libero import benchmark
    bm = benchmark.get_benchmark_dict()["libero_10"]()
    tasks = {}
    for t in range(10):
        if desc.get(t) != bm.get_task(t).language or \
                len(bm.get_task_init_states(t)) < 45:
            raise SystemExit(f"задача {t}: описание/число стартов")
        tasks[str(t)] = desc[t]
    _pins, init_ref = p31.k25_init_ref()
    D = d21.load_prepared(require_teacher=False)
    U, r = kp.harness_utils()
    q = np.maximum(np.abs(U.ACTION_Q99), np.abs(U.ACTION_Q01))
    import k32_model as km32
    return jnorm(dict(
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        prep_id=prep["prep_id"], prepare_sha1=sha(d21.REPORT),
        bar_revision=prep["bar_pin"]["revision"], snapshot=snap,
        bounds=prep["bounds_codec"], state_norm=prep["state_norm"],
        sources=srcs, tasks=tasks, init_ref=init_ref,
        banks=dict(train=list(kcol32.TRAIN_INIT), val=list(kcol32.VAL_INIT),
                   eval=list(range(30, 45)), unused=list(range(25, 30)),
                   closed=list(range(45, 50))),
        norm_rows=norm_rows(D), norm_cfg=dict(n=N_NORM, seed=32007,
                                              std_floor=km32.STD_FLOOR,
                                              clip=km32.CLIP),
        dist=kd.CONFIG, operator=ko.CONFIG, ppo=kppo.CFG, env=kcol.ENV,
        caps=CAPS, smoke=[list(x) for x in SMOKE_EPS], stats=STATS,
        precision=dict(trunk="fp16 autocast, no_grad, TF32 off", B=1,
                       reference="FP32", actor_critic="FP32 CPU"),
        scale_q=q.tolist(), offset_table_sha1=sha("data/pos_offset_table.json"),
        harness_cfg_sha1=sha(os.path.join(r, "config/eval/bar.yaml")),
        code={f: sha(os.path.join(HERE, f)) for f in MODEL_CODE}))


BASE_KEYS = ("prep_id", "prepare_sha1", "bar_revision", "bounds", "state_norm",
             "sources", "tasks", "init_ref", "banks", "norm_rows", "norm_cfg",
             "dist", "operator", "ppo", "env", "caps", "smoke", "stats",
             "precision", "scale_q", "offset_table_sha1", "harness_cfg_sha1",
             "code")


def prepare():
    rep = make_base()
    if os.path.exists(BASE):
        old = json.load(open(BASE, encoding="utf-8"))
        diff = [k for k in BASE_KEYS if old.get(k) != rep.get(k)]
        if diff:
            raise SystemExit(f"{BASE} закреплён; изменилось {diff}")
        print(f"  {BASE}: закреплён, сверка пройдена")
        return 0
    save_json(BASE, rep)
    print(f"  {BASE}: источники " + ", ".join(
        f"s{s} {e['weights_hash'][:12]} (обн. {e['update']})"
        for s, e in rep["sources"].items()) + f"; строк нормировки "
        f"{len(rep['norm_rows'])}")
    return 0


def load_base():
    if not os.path.exists(BASE):
        raise SystemExit("нет base_prepare: k32_protocol.py prepare")
    b = json.load(open(BASE, encoding="utf-8"))
    bad = [f for f, s in b["code"].items() if sha(os.path.join(HERE, f)) != s]
    if bad:
        raise SystemExit(f"код изменён после prepare: {bad}")
    return b


def norm_path(seed):
    return f"data/k32/normalizer_s{seed}.npz"


def contract(b=None):
    b = b or load_base()
    if not os.path.exists(NORM_MAN):
        raise SystemExit("нет нормировки: k32_protocol.py normalizer")
    man = json.load(open(NORM_MAN, encoding="utf-8"))
    if man.get("base") != fp({k: b[k] for k in BASE_KEYS}):
        raise SystemExit("нормировка построена по другому base_prepare")
    files = {}
    for s in SEEDS:
        p = norm_path(s)
        if not os.path.exists(p) or sha(p) != man["files"][str(s)]:
            raise SystemExit(f"нормировка s{s} отсутствует/изменена")
        files[str(s)] = man["files"][str(s)]
    return fp(dict(base=man["base"], norm=files, man=sha(NORM_MAN)))


def load_norm(seed):
    z = dict(np.load(norm_path(seed)))
    return {k: z[k] for k in ("actor_mean", "actor_std", "critic_mean",
                              "critic_std")}


def source_entry_of(doc, seed):
    try:
        return doc["sources"][str(seed)]
    except (KeyError, TypeError):
        raise SystemExit(f"в документе нет источника seed {seed}")


def load_source_dec(torch, doc, seed, device):
    import k29_model as km
    km.no_tf32(torch)
    e = source_entry_of(doc, seed)
    mod, bar, _prep = km.load_source(torch, e, device)
    if km.frozen_hash(mod) != e["weights_hash"]:
        raise SystemExit("источник после загрузки ≠ отпечатку")
    M = km.build(torch)
    return mod, bar, M, M["Decoder"](mod).freeze()


def ledger():
    return kcol.Ledger(LEDGER, CAPS)


# --- нормировка --------------------------------------------------------------------
def normalizer(a):
    import torch
    import k20_data as kd20
    import k21_data as d21
    import k21_model as km
    import k31_readout as kr
    import k32_model as km32
    b = load_base()
    D = d21.load_prepared(require_teacher=False)
    rows = b["norm_rows"]
    man = dict(base=fp({k: b[k] for k in BASE_KEYS}), files={}, stats={})
    for s in SEEDS:
        mod, bar, M, dec = load_source_dec(torch, b, s, a.device)
        h, _m = kr.build(torch)["half_mid"](dec._L, dec._U, "cpu")
        S_rows, floor = [], 0
        ac = torch.autocast("cuda", dtype=torch.float16,
                            enabled=torch.device(a.device).type == "cuda")
        for r in rows:
            batch = kd20.prompt_batch(bar, [np.array(D["IMG"][r], copy=True)],
                                      kd20.norm_states(np.asarray(
                                          D["state8"][[r]])), [D["task"][r]])
            s11 = torch.as_tensor(km.state11(np.asarray(D["state8"][[r]],
                                                        np.float64)),
                                  device=a.device)
            with torch.no_grad(), ac:
                F, s11n, _c = M["export"](mod, batch, int(D["pos_offset"][r]),
                                          s11)
            with torch.no_grad():
                ref = dec(F)
            pi = ko.path_info(F)
            floor += int(pi["floor"].any())
            S_rows.append(km32.state_vector(
                F, s11n[0].float().cpu().numpy(), 0,
                ref["Z"][0].float().cpu().numpy(), h.numpy(), pi))
        N = km32.fit_normalizer(np.stack(S_rows), mod.H)
        p = norm_path(s)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + ".tmp", "wb") as fh:
            np.savez(fh, **{k: N[k] for k in ("actor_mean", "actor_std",
                                              "critic_mean", "critic_std")},
                     H=np.array(N["H"]), rows=np.asarray(rows))
        os.replace(p + ".tmp", p)
        man["files"][str(s)] = sha(p)
        man["stats"][str(s)] = dict(H=N["H"], n_rows=N["n_rows"],
                                    sha=km32.norm_sha(N), floor_rows=floor,
                                    actor_std_min=float(N["actor_std"].min()),
                                    critic_std_min=float(N["critic_std"]
                                                         .min()))
        print(f"  нормировка s{s}: строк {N['n_rows']}, H {N['H']}, c-floor "
              f"на {floor} строках; {p}", flush=True)
        del mod, bar, M, dec
        free()
    save_json(NORM_MAN, man)
    return 0


# --- общие сборки ------------------------------------------------------------------
def build_actor(torch, b, seed, op, H):
    import k32_model as km32
    return km32.build(torch, H, ko.DIM[op], seed, load_norm(seed))


def run_dir(op, seed):
    return f"data/k32/runs/{op}_s{seed}"


def final_path(op, seed):
    return os.path.join(run_dir(op, seed), "final.pt")


def train_report(op, seed):
    return f"reports/k32/train_{op}_s{seed}.json"


def fresh(kind):
    try:
        c = contract()
    except SystemExit as e:
        return [str(e.code)]
    path = dict(real_check=REAL_CHECK, smoke=SMOKE)[kind]
    if not os.path.exists(path):
        return [f"нет {path}"]
    d = json.load(open(path, encoding="utf-8"))
    pr = []
    if d.get("contract") != c:
        pr.append(f"{path}: другой контракт")
    if not d.get("ok"):
        pr.append(f"{path}: не пройден")
    return pr


# --- real-check ----------------------------------------------------------------------
def real_check(a):
    import k18_collector as k18c
    sim = k18c.Sim(5, "persistent")
    import torch
    import k25_policy as kp25
    import k29_policy as kp
    import k21_model as km
    import k32_model as km32
    import k32_policy as kpol
    b = load_base()
    c = contract(b)
    L = ledger()
    try:
        obs5 = sim.reset(0, [0, 1, 2, 3, 4])
        dummy = np.array([kcol.DUMMY] * 5)
        for _ in range(kcol.ENV["waiting_steps"]):
            obs5, _r, _d, _s = sim.step(dummy)
    finally:
        sim.close()
    eid = f"checks/real_check@{time.strftime('%Y%m%dT%H%M%S')}"
    L.reserve("checks", eid, amount=5 * kcol.ENV["waiting_steps"])
    L.commit(eid, 5 * kcol.ENV["waiting_steps"])
    rep = dict(contract=c, seeds={})
    prob = []
    for s in SEEDS:
        mod, bar, M, dec = load_source_dec(torch, b, s, a.device)
        dev = torch.device(a.device)
        inp = kp.Inputs(torch, bar["proc"], a.device)
        desc = inp.task_desc(0)
        ac = torch.autocast(dev.type, dtype=torch.float16)
        r = {}
        batch5, po, s8_5 = inp(obs5, desc, 0)
        s11 = torch.as_tensor(km.state11(s8_5), device=dev)
        with torch.no_grad(), ac:
            F5, _sn, cnt = M["export"](mod, batch5, po, s11)
            legacy = dec(F5, autocast=ac)
            real = mod(batch5, po, s11)
        r["legacy_B5"] = bool(torch.equal(legacy["arm"], real["arm"]) and
                              torch.equal(legacy["Z"], real["y"]))
        r["counters"] = kp25.counters_problems(cnt)
        frozen_v = {n: p._version for n, p in mod.named_parameters()}
        H = mod.H
        arms, recs = {}, {}
        for kind in ("R",) + OPS:
            actor = None
            if kind != "R":
                actor, critic = build_actor(torch, b, s, kind, H)
                r[f"params_{kind}"] = dict(
                    actor=km32.n_params(actor), critic=km32.n_params(critic))
            arm = kpol.Arm(torch, mod, dec, M, kind=kind, seed=s, actor=actor)
            arm.begin(0, 0, 0)
            b1 = {k: v[:1] for k, v in obs5.items()}
            bt1, po1, s81 = inp(b1, desc, 0)
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            a_, rec = arm.decide(bt1, po1, s81, 0)
            torch.cuda.synchronize(dev)
            r[f"ms_{kind}"] = (time.perf_counter() - t0) * 1e3
            arms[kind], recs[kind] = a_, rec
        r["mean_init_eq_reference"] = {k: bool(np.array_equal(arms[k],
                                                               arms["R"]))
                                       for k in OPS}
        aT, _c = build_actor(torch, b, s, "T", H)
        aP, _c = build_actor(torch, b, s, "P", H)
        r["T_P_init_equal"] = all(torch.equal(x, y) for x, y in zip(
            aT.state_dict().values(), aP.state_dict().values()))
        # ненулевые фиксированные η: тождества и границы на настоящих весах
        import k31_readout as kr
        E = kr.build(torch)
        bt1, po1, s81 = inp({k: v[:1] for k, v in obs5.items()}, desc, 0)
        with torch.no_grad(), ac:
            F1, s11n1, _c = M["export"](mod, bt1, po1, torch.as_tensor(
                km.state11(s81), device=dev))
        with torch.no_grad():
            ref1 = dec(F1)
        h, m = E["half_mid"](dec._L, dec._U, "cpu")
        rc = dict(Z0=ref1["Z"][0].float().cpu(), arm=ref1["arm"][0, :8]
                  .float().cpu(), half=h, mid=m)
        pi = ko.path_info(F1)

        def edited(rr, e, dec=dec, F1=F1, ref1=ref1, pi=pi):
            o = E["edited"](dec, F1, ref1, [kr.LEVELS[rr]], [pi["path"][rr]],
                            np.asarray(e)[None])
            return o["Zhat"][0].float().cpu()
        rng = np.random.default_rng(32000 + s)
        ident, bounds = {}, []
        for t in range(4):
            eta = rng.normal(size=6).astype(np.float32)
            for rr in range(3):
                dT = ko.raw_effect(torch, "T", rr, eta, rc, pi, edited)[0]
                dP = ko.raw_effect(torch, "P", rr, eta, rc, pi, edited)[0]
                ident.setdefault(kr.LEVELS[rr], []).append(
                    float((dT - dP).abs().max()))
            for op in OPS:
                e = (3 * rng.normal(size=ko.DIM[op])).astype(np.float32)
                _arm, j = ko.command(torch, op, int(rng.integers(3)), e, rc,
                                     pi, edited)
                bounds.append(dict(op=op, max=j["lim"]["max"],
                                   rms=j["lim"]["rms"],
                                   clipped=j["lim"]["clipped"],
                                   capped=j["lim"]["capped"]))
        r["T_minus_P_maxabs"] = {k: max(v) for k, v in ident.items()}
        r["bounds_ok"] = all(x["max"] <= ko.CLIP + 1e-7 and
                             x["rms"] <= ko.RMS_CAP + 1e-7 for x in bounds)
        r["c"] = pi["c"].tolist()
        r["c_floor"] = pi["floor"].tolist()
        # кеш ↔ online для ненулевого actor и механический шаг
        S = np.stack([recs[k]["S"] for k in OPS] * 6)
        Snp = S + np.random.default_rng(1).normal(0, 0.01, S.shape).astype(
            np.float32)
        aN, cN = build_actor(torch, b, s, "T", H)
        with torch.no_grad():
            g_ = torch.Generator().manual_seed(5)
            for p_ in aN.out.parameters():
                p_.copy_(0.1 * torch.randn(p_.shape, generator=g_))
        X = km32.actor_inputs(torch, torch.as_tensor(Snp), H)
        mus = {mb: kppo.policy_all(torch, aN, X, mb)[1] for mb in (8, 32, 64)}
        r["cache_mu_max_mb"] = max(float((mus[k] - mus[32]).abs().max())
                                   for k in (8, 64))
        Xn = torch.as_tensor(km32.actor_inputs_np(Snp, H), dtype=torch.float32)
        r["np_vs_torch_inputs_max"] = float((Xn - X).abs().max())
        oa = torch.optim.AdamW(aN.parameters(), lr=1e-4)
        oc = torch.optim.Adam(cN.parameters(), lr=3e-4)
        h0 = kppo.tensor_hash(aN.state_dict())
        lg, raw = aN(X)
        loss = -kd.mu_of(torch, raw).sum() + lg.sum() + \
            ((cN(torch.as_tensor(Snp)) - 0.5) ** 2).sum()
        loss.backward()
        oa.step()
        oc.step()
        r["mechanical_step_changes_actor"] = kppo.tensor_hash(
            aN.state_dict()) != h0
        r["source_no_grad"] = all(p.grad is None for p in mod.parameters())
        r["source_versions_same"] = all(
            p._version == frozen_v[n] for n, p in mod.named_parameters())
        r["peak_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30 \
            if dev.type == "cuda" else None
        rep["seeds"][str(s)] = r
        for k, v in (("legacy_B5", r["legacy_B5"]),
                     ("mean_init", all(r["mean_init_eq_reference"].values())),
                     ("T_P_init", r["T_P_init_equal"]),
                     ("leaf T=P", r["T_minus_P_maxabs"]["leaf"] <= 1e-6),
                     ("bounds", r["bounds_ok"]),
                     ("cache", r["cache_mu_max_mb"] <= 1e-6),
                     ("inputs", r["np_vs_torch_inputs_max"] <= 1e-6),
                     ("step", r["mechanical_step_changes_actor"]),
                     ("frozen", r["source_no_grad"] and
                      r["source_versions_same"])):
            if not v:
                prob.append(f"s{s}: {k}")
        if r["counters"]:
            prob.append(f"s{s}: счётчики")
        print(f"  s{s}: legacy B5 {r['legacy_B5']}; mean-init = R "
              f"{r['mean_init_eq_reference']}; T=P init {r['T_P_init_equal']}"
              f"; |T−P| root/mid/leaf {r['T_minus_P_maxabs']}; c {r['c']}; "
              f"границы {r['bounds_ok']}; кеш {r['cache_mu_max_mb']:.1e}; "
              f"мс R/T/P/F {r['ms_R']:.0f}/{r['ms_T']:.0f}/{r['ms_P']:.0f}/"
              f"{r['ms_F']:.0f}; параметры {r['params_T']} {r['params_F']}",
              flush=True)
        del mod, bar, M, dec
        free()
    rep["ok"] = not prob
    rep["problems"] = prob
    save_json(REAL_CHECK, rep)
    print("  " + ("real-check K-32 пройден" if not prob else
                  "real-check НЕ пройден: " + "; ".join(prob)))
    return 0 if not prob else 3


# --- collector-smoke ----------------------------------------------------------------
def smoke(a):
    pr = fresh("real_check")
    if pr:
        raise SystemExit("smoke без свежего real-check: " + "; ".join(pr))
    import k18_collector as k18c
    sim = k18c.Sim(1, "persistent")
    import torch
    import k29_policy as kp
    import k31_harness as kh31
    import k32_harness as kh
    import k32_policy as kpol
    b = load_base()
    c = contract(b)
    L = ledger()
    rep = dict(contract=c, runs={})
    prob = []
    t_all = time.time()
    phys_all = 0
    try:
        for s in SEEDS:
            mod, bar, M, dec = load_source_dec(torch, b, s, a.device)
            io = kh31.IO(kp.Inputs(torch, bar["proc"], a.device))
            for op in OPS:
                actor, _c = build_actor(torch, b, s, op, mod.H)
                arm = kpol.Arm(torch, mod, dec, M, kind=op, seed=s,
                               actor=actor, stochastic=True, space=1)
                arm.keep_first = True
                ec, A_eps, ds, trs, n_ph, firsts = [], [], [], [], 0, []
                for j, (task, init) in enumerate(SMOKE_EPS):
                    eid = f"checks/smoke/{op}/s{s}/t{task}_i{init}"
                    if L.is_done(eid):
                        eid += f"@{time.strftime('%Y%m%dT%H%M%S')}"
                    L.reserve("checks", eid)
                    arm.episode = 9 * 10 ** 6 + j
                    ep = kcol.run_episode(sim, io, arm, task, init)
                    L.commit(eid, ep["physical"])
                    n_ph += ep["physical"]
                    ec.append((j, ep["calls"]))
                    A_eps.append(ep["actions"])
                    ds.append(ep["done_step"])
                    trs += [cl["rec"] for cl in ep["calls"]]
                    firsts.append((arm.first_F, ep["calls"][0]["rec"]))
                phys_all += n_ph
                J = kh.pack_journal(ec, ko.DIM[op])
                pj = smoke_journal_problems(torch, J, trs, A_eps, ds, b, op,
                                            s, actor)
                rb_p, rb_st = rebuild_problems(torch, firsts, b, op, s)
                pj += rb_p
                rep["runs"][f"{op}_s{s}"] = dict(
                    physical=n_ph, problems=pj,
                    clip_share=float(np.mean([t_["clipped"] for t_ in trs])),
                    cap_share=float(np.mean([t_["capped"] for t_ in trs])),
                    rms_mean=float(np.mean([t_["rms"] for t_ in trs])),
                    route=np.bincount([t_["r"] for t_ in trs],
                                      minlength=3).tolist(), rebuild=rb_st)
                prob += [f"{op} s{s}: {x}" for x in pj]
                print(f"  smoke {op} s{s}: шагов {n_ph}; RMS Du "
                      f"{rep['runs'][f'{op}_s{s}']['rms_mean']:.4f}; clip/cap "
                      f"{rep['runs'][f'{op}_s{s}']['clip_share']:.2f}/"
                      f"{rep['runs'][f'{op}_s{s}']['cap_share']:.2f}; маршруты "
                      f"{rep['runs'][f'{op}_s{s}']['route']}; "
                      + ("ок" if not pj else "; ".join(pj[:3])), flush=True)
            del mod, bar, M, dec
            free()
    finally:
        sim.close()
    sec = time.time() - t_all
    rep.update(ok=not prob, problems=prob, physical=phys_all, seconds=sec,
               sec_per_step=sec / max(phys_all, 1))
    rep["forecast_hours"] = dict(
        train_upper=rep["sec_per_step"] * 6 * kppo.CAP / 3600,
        sr_upper=rep["sec_per_step"] * CAPS["sr"] / 3600)
    save_json(SMOKE, rep)
    print(f"  collector-smoke: шагов {phys_all}, {sec / 60:.1f} мин, "
          f"{rep['sec_per_step'] * 1e3:.0f} мс/шаг; верхняя оценка обучения "
          f"{rep['forecast_hours']['train_upper']:.0f} ч, SR "
          f"{rep['forecast_hours']['sr_upper']:.0f} ч; "
          + ("пройден" if not prob else "НЕ пройден"))
    return 0 if not prob else 3


def smoke_journal_problems(torch, J, trs, A_eps, ds, b, op, seed, actor):
    """Стохастический журнал: (r, η) = повтору сэмплирования по ключу,
    log π = пересчёту по кешу S текущим actor, D/Du/рука и исполненное —
    как в проверке блока SR."""
    import k31_readout as kr
    import k32_harness as kh
    import k32_model as km32
    import k32_policy as kpol
    p = []
    S = torch.as_tensor(np.stack([t["S"] for t in trs]))
    X = km32.actor_inputs(torch, S, (S.shape[1] - 72) // 6)
    lg, mu = kppo.policy_all(torch, actor, X, 32)
    if float((mu - torch.as_tensor(np.stack([t["mu"] for t in trs])))
             .abs().max()) > 1e-6:
        p.append("μ кеша ≠ записанному")
    lp = kd.log_prob(torch, lg.double(), mu.double(),
                     torch.as_tensor([t["r"] for t in trs]),
                     torch.as_tensor(np.stack([t["eta"] for t in trs]))
                     .double())
    if float((lp - torch.as_tensor([t["logpi"] for t in trs])).abs().max()) \
            > 1e-4:
        p.append("log π кеша ≠ записанному")
    for t in trs:
        gr, gg = kpol.gens(1, seed, t["episode"], t["call"])
        rr, ee, _p = kd.sample(t["logits"], t["mu"], gr, gg)
        if rr != t["r"] or not np.array_equal(ee, t["eta"]):
            p.append(f"вызов {t['episode']}/{t['call']}: (r, η) ≠ повтору")
            break
    E = kr.build(torch)
    h, m = [v.numpy() for v in E["half_mid"](
        np.asarray(b["bounds"]["L"])[:6], np.asarray(b["bounds"]["U"])[:6],
        "cpu")]
    Jd = dict(J)
    Jd["logits"] = np.stack([np.eye(3)[t["r"]] for t in trs]).astype(
        np.float32)
    Jd["mu"] = np.stack([np.repeat(t["eta"][None], 3, 0) for t in trs])
    p += kh.journal_problems(Jd, A_eps, ds, h, m,
                             np.asarray(b["bounds"]["L"])[:6],
                             np.asarray(b["bounds"]["U"])[:6],
                             np.asarray(b["scale_q"]), op)
    return p


def rebuild_problems(torch, firsts, b, op, seed):
    """Независимая CPU-пересборка первого вызова каждого smoke-эпизода из
    закреплённых весов: reference (Z0), путь/c, оператор, ограничитель,
    команда. Допуски: Z0/Zhat 1e−5 отн., Du 1e−5, команды 1e−6, c 1e−6."""
    import k31_protocol as p31
    import k31_readout as kr
    dcpu = p31.cpu_decoder(torch, b["sources"][str(seed)])
    E = kr.build(torch)
    h, m = E["half_mid"](dcpu._L, dcpu._U, "cpu")
    p, st = [], dict(Z0=0.0, Zhat=0.0, Du=0.0, arm=0.0, c=0.0)
    for F, rec in firsts:
        with torch.no_grad():
            ref = dcpu(F)
        st["Z0"] = max(st["Z0"], p31._rel(ref["Z"][0].numpy(), rec["Z0"]))
        pi = ko.path_info(F)
        if list(pi["path"]) != list(rec["path"]):
            p.append("путь пересборки ≠ записанному")
            continue
        st["c"] = max(st["c"], float(np.abs(pi["c"] - rec["c"]).max()))
        rc = dict(Z0=ref["Z"][0].float(), arm=ref["arm"][0, :8].float(),
                  half=h, mid=m)

        def edited(rr, e, F=F, ref=ref, pi=pi):
            o = E["edited"](dcpu, F, ref, [kr.LEVELS[rr]], [pi["path"][rr]],
                            np.asarray(e)[None])
            return o["Zhat"][0].float()
        arm, j = ko.command(torch, op, rec["r"], rec["eta"], rc, pi, edited)
        if op == "T" and not j["zero"]:
            st["Zhat"] = max(st["Zhat"], p31._rel(j["Zhat"], rec["Zhat"]))
        st["Du"] = max(st["Du"], float(np.abs(j["Du"] - rec["Du"]).max()))
        st["arm"] = max(st["arm"], float(np.abs(arm.numpy() -
                                                rec["a_codec"][:, :6]).max()))
    for k_, tol in (("Z0", Z_TOL), ("Zhat", Z_TOL), ("Du", DU_TOL),
                    ("arm", ARM_TOL), ("c", 1e-6)):
        if not st[k_] <= tol:
            p.append(f"пересборка {k_}: {st[k_]:.2e} > {tol}")
    return p, st


# --- обучение ------------------------------------------------------------------------
def train(a):
    for k in ("real_check", "smoke"):
        pr = fresh(k)
        if pr:
            raise SystemExit(f"обучение без свежего {k}: " + "; ".join(pr))
    import k18_collector as k18c
    sim = k18c.Sim(1, "persistent")
    import torch
    import k29_policy as kp
    import k31_harness as kh31
    import k32_policy as kpol
    b = load_base()
    c = contract(b)
    L = ledger()
    todo = [(o, s) for o, s in ORDER if a.only is None or f"{o}_s{s}" ==
            a.only]
    try:
        cur = None
        for op, s in todo:
            if os.path.exists(train_report(op, s)):
                rj = json.load(open(train_report(op, s), encoding="utf-8"))
                if rj.get("contract") == c and rj.get("finished") and \
                        sha(final_path(op, s)) == rj.get("final_sha1"):
                    print(f"  {op} s{s}: обучение завершено и закреплено")
                    continue
            if cur != s:
                if cur is not None:
                    del mod, bar, M, dec                  # noqa: F821
                    free()
                mod, bar, M, dec = load_source_dec(torch, b, s, a.device)
                io = kh31.IO(kp.Inputs(torch, bar["proc"], a.device))
                cur = s
            actor, critic = build_actor(torch, b, s, op, mod.H)

            def make_arm(act, stochastic, op=op, s=s):
                return kpol.Arm(torch, mod, dec, M, kind=op, seed=s,
                                actor=act, stochastic=stochastic)
            tr = kppo.Trainer(torch, seed=s, op=op, H=mod.H, actor=actor,
                              critic=critic, make_arm=make_arm, sim=sim,
                              io=io, ledger=L, odir=run_dir(op, s),
                              contract=dict(contract=c, op=op, seed=s),
                              log=lambda x: print(x, flush=True))
            st = tr.run()
            fpth = final_path(op, s)
            torch.save(dict(actor=actor.state_dict(),
                            actor_hash=tr.actor_hash(), op=op, seed=s,
                            contract=c), fpth + ".tmp")
            os.replace(fpth + ".tmp", fpth)
            save_json(train_report(op, s), dict(
                contract=c, op=op, seed=s, finished=True,
                final_sha1=sha(fpth), actor_hash=tr.actor_hash(),
                updates=st["update"], accepted=st["accepted"],
                no_actor_update=st["no_actor_update"], val=st["val"],
                train_steps=tr.spent(), discarded=st["discarded"],
                history=[{k: v for k, v in u.items() if k != "history"}
                         for u in st["updates"]]))
            print(f"  {op} s{s}: итог — обновлений {st['update']} (принято "
                  f"{st['accepted']}), шагов {tr.spent()}; валидация "
                  f"{ {k: round(v['sr'], 3) for k, v in st['val'].items()} }"
                  + ("; stopped_no_actor_update" if st["no_actor_update"]
                     else ""), flush=True)
    finally:
        sim.close()
    return 0


# --- планы SR ------------------------------------------------------------------------
def eval_plan_path(kind, seed):
    return f"reports/k32/plan_eval_{kind}_s{seed}.json"


def make_eval_plan(b, c, kind, seed, half, mid, actor=None):
    import k32_harness as kh
    lab = kh.label(kind, seed)
    arm = dict(label=lab, seed=int(seed), contract=c,
               actor_hash=None if actor is None else actor["actor_hash"],
               actor_path=None if actor is None else actor["path"],
               actor_sha1=None if actor is None else actor["sha1"])
    return jnorm(dict(kind=kind, seed=int(seed),
                      sources={str(seed): b["sources"][str(seed)]},
                      tasks=b["tasks"], init_ref=b["init_ref"],
                      half=[float(x) for x in half],
                      mid=[float(x) for x in mid],
                      L=list(b["bounds"]["L"])[:6],
                      U=list(b["bounds"]["U"])[:6], scale_q=b["scale_q"],
                      arms={lab: arm}, contract=c, code=b["code"]))


def plan_eval(a):
    import torch
    import k31_readout as kr
    b = load_base()
    c = contract(b)
    h, m = kr.build(torch)["half_mid"](np.asarray(b["bounds"]["L"])[:6],
                                       np.asarray(b["bounds"]["U"])[:6],
                                       "cpu")
    kinds = ("R",) if a.which == "ref" else OPS
    for s in SEEDS:
        for kind in kinds:
            actor = None
            if kind != "R":
                rj = json.load(open(train_report(kind, s), encoding="utf-8"))
                if rj.get("contract") != c or not rj.get("finished") or \
                        sha(final_path(kind, s)) != rj["final_sha1"]:
                    raise SystemExit(f"{kind} s{s}: обучение не закреплено")
                actor = dict(actor_hash=rj["actor_hash"],
                             path=final_path(kind, s),
                             sha1=rj["final_sha1"])
            plan = make_eval_plan(b, c, kind, s, h, m, actor)
            source_entry_of(plan, s)
            path = eval_plan_path(kind, s)
            if os.path.exists(path):
                if json.load(open(path, encoding="utf-8")) != plan:
                    raise SystemExit(f"{path} закреплён и отличается")
                continue
            save_json(path, plan)
            print(f"  {path}: записан")
    return 0


def load_eval_plan_kind(kind, seed):
    c = contract()
    path = eval_plan_path(kind, seed)
    if not os.path.exists(path):
        raise SystemExit(f"нет {path}")
    plan = json.load(open(path, encoding="utf-8"))
    if plan["contract"] != c:
        raise SystemExit(f"{path}: другой контракт")
    ps = sha(path)
    for v in plan["arms"].values():
        v["plan_sha1"] = ps
    return plan


def make_eval_arm(torch, plan, kind, seed, mod, dec, M):
    import k32_model as km32
    import k32_policy as kpol
    if kind == "R":
        return kpol.Arm(torch, mod, dec, M, kind="R", seed=seed), 6
    arm_d = next(iter(plan["arms"].values()))
    if sha(arm_d["actor_path"]) != arm_d["actor_sha1"]:
        raise SystemExit("final actor ≠ плану SR")
    ck = torch.load(arm_d["actor_path"], map_location="cpu",
                    weights_only=False)
    actor, _c = km32.build(torch, mod.H, ko.DIM[kind], seed, load_norm(seed))
    actor.load_state_dict(ck["actor"])
    if kppo.tensor_hash(actor.state_dict()) != arm_d["actor_hash"]:
        raise SystemExit("веса final actor ≠ плану")
    actor.eval()
    return kpol.Arm(torch, mod, dec, M, kind=kind, seed=seed, actor=actor), \
        ko.DIM[kind]


def rebuild_selftest():
    import tempfile
    import torch
    import k21_model as km
    import k21_train as kt
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
        for q in list(mod.cond.parameters()) + list(mod.readout.parameters()):
            q.add_(torch.randn(q.shape, generator=g) * 0.05)
    for q in mod.parameters():
        q.requires_grad_(False)
    dec = M["Decoder"](mod).freeze()
    H = mod.H
    norm = km32.fit_normalizer(np.random.default_rng(0).normal(
        size=(64, 6 * H + 72)), H)
    with tempfile.TemporaryDirectory() as td:
        ck = os.path.join(td, "best.pt")
        torch.save(dict(weights=kt.trainable_state(mod)), ck)
        b = dict(sources={"0": dict(ckpt=ck, ckpt_sha1=sha(ck))})
        for op in OPS:
            actor, _c = km32.build(torch, H, ko.DIM[op], 0, norm)
            arm = kpol.Arm(torch, mod, dec, M, kind=op, seed=0, actor=actor,
                           stochastic=True, space=1)
            arm.keep_first = True
            arm.begin(0, 1, 5)
            _a, rec = arm.decide(km.fake_batch(torch, B=1, seed=2), 1,
                                 np.zeros((1, 8)), 0)
            pr, st = rebuild_problems(torch, [(arm.first_F, rec)], b, op, 0)
            assert pr == [], (op, pr, st)
            # журнал стохастического smoke: повтор (r, η), log π, D/Du/рука
            import k32_harness as kh

            class FIO(kcol.FakeIO):
                def prep(self, obs, desc, task):
                    x = float(obs["state"][0, 0])
                    return km.fake_batch(torch, B=1, seed=int(abs(x) * 1e3)
                                         % 13), 1, np.zeros((1, 8))
            io = FIO()
            ec, A_eps, ds, trs = [], [], [], []
            for j in range(2):
                arm.episode = 9 * 10 ** 6 + j
                ep = kcol.run_episode(kppo.ToySim(), io, arm, 0, j)
                ec.append((j, ep["calls"]))
                A_eps.append(ep["actions"])
                ds.append(ep["done_step"])
                trs += [cl["rec"] for cl in ep["calls"]]
            J = kh.pack_journal(ec, ko.DIM[op])
            bb = dict(b, bounds=dict(L=L.tolist(), U=U.tolist()),
                      scale_q=io.max_act_q.tolist())
            assert smoke_journal_problems(torch, J, trs, A_eps, ds, bb, op,
                                          0, actor) == [], op
            trs[1] = dict(trs[1], eta=trs[1]["eta"] + 1e-3)
            assert smoke_journal_problems(torch, J, trs, A_eps, ds, bb, op,
                                          0, actor), op
            for k_, f in (("Du", lambda r_: r_["Du"].__iadd__(1e-3)),
                          ("a_codec", lambda r_: r_["a_codec"][:, :6]
                           .__iadd__(1e-5)),
                          ("path", lambda r_: r_.__setitem__(
                              "path", [9, 9, 9]))):
                r2 = {kk: (v.copy() if hasattr(v, "copy") else v)
                      for kk, v in rec.items()}
                f(r2)
                assert rebuild_problems(torch, [(arm.first_F, r2)], b, op,
                                        0)[0], (op, k_)


def selftest():
    # строки нормировки: детерминизм, уникальность, баланс задач
    D = dict(train_rows=np.arange(400), task=np.array(
        ["a"] * 200 + ["b"] * 150 + ["c"] * 50), episode=np.arange(400) // 10)
    r1, r2 = norm_rows(D, 120), norm_rows(D, 120)
    assert r1 == r2 and len(set(r1)) == 120
    cnt = {t: sum(1 for r in r1 if D["task"][r] == t) for t in "abc"}
    assert cnt == {"a": 40, "b": 40, "c": 40}, cnt
    # план SR -> загрузчик источника (тот же формат, что base_prepare)
    b = dict(sources={str(s): dict(seed=s, ckpt=f"c{s}") for s in SEEDS},
             tasks={"0": "x"}, init_ref={}, bounds=dict(L=[-1.0] * 7,
                                                        U=[1.0] * 7),
             scale_q=[1.0] * 7, code={})
    for s in SEEDS:
        for kind in ("R", "T"):
            act = None if kind == "R" else dict(actor_hash="h", path="p",
                                                sha1="x")
            pl = jnorm(make_eval_plan(b, "C", kind, s, [1.0] * 6, [0.0] * 6,
                                      act))
            assert source_entry_of(pl, s) == b["sources"][str(s)]
    try:
        source_entry_of(dict(source={}), 0)
        raise AssertionError("старый формат принят")
    except SystemExit:
        pass
    assert len(CAPS) == 14 and sum(v for k, v in CAPS.items()
                                   if k.startswith("train")) == 1_200_000
    assert CAPS["sr"] == 732_000
    total = sum(CAPS.values()) - 6 * kppo.VAL_CAP + 164_700
    assert total == 2_146_700, total
    rebuild_selftest()
    print("самопроверка k32_protocol пройдена: CPU-пересборка вызова T/P/F "
          "из весов чекпойнта (подмена Du/руки/пути — отказ), журнал стохастического "
          "smoke (повтор (r, η) по ключу, log π, D/Du/рука; подмена η — "
          "отказ); строки нормировки "
          "(детерминизм, уникальность, баланс задач, метки — текст), план SR "
          "-> загрузчик источника, бюджеты (обучение 1.2 млн, SR 732 тыс., "
          "всего с валидацией и проверками 2 146 700)")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-32: протокол")
    ap.add_argument("cmd", choices=("prepare", "normalizer", "real-check",
                                    "smoke", "train", "plan-eval",
                                    "check-fresh"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--kind", default=None)
    ap.add_argument("--which", choices=("ref", "final"), default="final")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    try:
        if a.cmd == "prepare":
            return prepare()
        if a.cmd == "normalizer":
            return normalizer(a)
        if a.cmd == "real-check":
            return real_check(a)
        if a.cmd == "smoke":
            return smoke(a)
        if a.cmd == "train":
            return train(a)
        if a.cmd == "plan-eval":
            return plan_eval(a)
        if a.cmd == "check-fresh":
            pr = fresh(a.kind)
            print("  " + ("свежо" if not pr else "; ".join(pr)))
            return 0 if not pr else 1
    except kcol.Mismatch as e:
        print(f"  ОСТАНОВ (воспроизведение): {e}", flush=True)
        return 3
    except ko.Technical as e:
        print(f"  ОСТАНОВ (технически): {e}", flush=True)
        return 3
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
