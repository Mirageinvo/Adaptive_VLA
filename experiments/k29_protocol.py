#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-29: подготовка, real-check, collector-smoke и обучение по сидам.

prepare — reports/k29/prepare.json (неизменяем после создания; повтор только
  сверяет): источники K-25 обоих сидов (SHA, отпечаток, отчёт, код), снимок
  BAR, пределы и нормировка состояния подготовки K-21, банки задач/начальных
  состояний (описания задач — из плана A K-21, сверка с LIBERO, число
  начальных состояний, непересечение банков), шум, точность, обучаемые
  ключи, конфигурация PPO, SHA кода K-29/K-25/K-22/K-21 и сборщика;
  закреплённые блоки K-25 (описательно) и их начальные хеши 30…44 — эталон
  холодного старта для оценки.
real-check — reports/k29/real_check.json: для обоих сидов на строках val
  K-21 (B = 1 и 5): экспорт + декодер в точности источника = K-25 побитово
  (arm, логит схвата, Z); actor FP32 = reference FP32 побитово; описательно
  |FP32 − K-25|; счётчики; кеш: online B = 5 против minibatch 8/32/64 —
  max|Δμ| ≤ 1e−5, max|Δ log π| ≤ 1e−3; полный шаг PPO (actor и critic
  меняются, конечны, замороженное неизменно); пик памяти экспорта B = 5 и
  PPO minibatch 8/32/64; время.
smoke — reports/k29/smoke.json: две волны seed 0 (задачи 0 и 8, init из
  0…19) с шумом, без сохранения весов: переданные = записанные действия,
  рука = mid + half·tanh(Y/half), k и маски, защёлка награды, бюджет,
  пересчёт μ/log π по кешу.
train --seed S — обучение (k29_ppo.Trainer), итог data/k29/runs/sS/final.pt
  и reports/k29/train_sS.json.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k29_collect as kc  # noqa: E402
import k29_dist as kd  # noqa: E402
import k29_ppo as kpo  # noqa: E402

PREPARE = "reports/k29/prepare.json"
REAL_CHECK = "reports/k29/real_check.json"
SMOKE = "reports/k29/smoke.json"
RESOURCES = "reports/k29/resources.json"
SEEDS = kpo.SEEDS
EVAL_TASKS = (0, 1, 6, 7, 8)
PRIMARY_TASKS = kc.CYCLE_TASKS
EVAL_BLOCKS = (30, 35, 40)
CLOSED = tuple(range(45, 50))
SMOKE_MAX = 2 * kc.WAVE_MAX
MODEL_CODE = ("k29_dist.py", "k29_model.py", "k29_policy.py",
              "k29_collect.py", "k29_ppo.py", "k29_protocol.py",
              "k25_readout.py", "k25_model.py", "k25_policy.py",
              "k25_protocol.py", "k22_model.py", "k21_model.py",
              "k21_tree.py", "k21_data.py", "k21_train.py", "k20_data.py",
              "k18_collector.py", "k16_models.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def fp(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()
                        ).hexdigest()


def code_sha():
    return {f: sha(os.path.join(HERE, f)) for f in MODEL_CODE}


def add_resource(kind, amount, path=RESOURCES):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = json.load(open(path, encoding="utf-8")) if os.path.exists(path) \
        else {}
    d[kind] = int(d.get(kind, 0)) + int(amount)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)
    os.replace(path + ".tmp", path)


# --- подготовка -------------------------------------------------------------------
def banks():
    b = dict(train=dict(tasks=list(PRIMARY_TASKS), init=list(kpo.TRAIN_INIT)),
             val=dict(tasks=list(PRIMARY_TASKS), init=list(kpo.VAL_INIT)),
             eval=dict(tasks=list(EVAL_TASKS), init=list(range(30, 45)),
                       blocks=list(EVAL_BLOCKS)),
             closed=list(CLOSED), unused=list(range(25, 30)))
    sets = [set(b["train"]["init"]), set(b["val"]["init"]),
            set(b["eval"]["init"])]
    for i in range(3):
        for j in range(i + 1, 3):
            if sets[i] & sets[j]:
                raise SystemExit("банки начальных состояний пересекаются")
    if any(x in CLOSED for s_ in sets for x in s_):
        raise SystemExit("банки задевают 45…49")
    return b


def k25_refs():
    """Закреплённые блоки K-25 обоих сидов: pin + эталон начальных хешей и
    исходы (описательно)."""
    import k23_protocol as p23
    import k26_protocol as p26
    pins, init_ref, outc = {}, {}, {}
    for s in SEEDS:
        r = p26.ref_k25(s)
        pins[f"k25_s{s}"] = p23.pin_ref(r)
        for (lab, su, t, b), f in r["blocks"].items():
            if int(t) not in EVAL_TASKS:
                continue
            d = json.load(open(f, encoding="utf-8"))
            for e in d["episodes"]:
                key = f"{int(t)}:{int(e['init_state_id'])}"
                ih = e["init_hash"]
                if init_ref.setdefault(key, ih) != ih:
                    raise SystemExit(f"эталон старта {key} различается между "
                                     f"сидами K-25")
                outc[f"{s}|{key}"] = bool(e["success"])
    want = {f"{t}:{i}" for t in EVAL_TASKS for i in range(30, 45)}
    if set(init_ref) != want:
        raise SystemExit("эталон стартов K-25 неполон")
    return pins, init_ref, outc


def make_prepare():
    import k21_data as d21
    import k23_protocol as p23
    import k29_model as km
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    srcs = {}
    for s in SEEDS:
        e = km.source_entry(s)
        if e["prep_id"] != prep["prep_id"]:
            raise SystemExit(f"seed {s}: источник другой подготовки K-21")
        srcs[str(s)] = e
    rA = p23.ref_k21_main()
    desc = {}
    for (l_, s_, t_, b_), f in rA["blocks"].items():
        if l_ == "fullbar":
            desc.setdefault(int(t_), json.load(open(f, encoding="utf-8"))
                            .get("task_description"))
    from libero.libero import benchmark
    bm = benchmark.get_benchmark_dict()["libero_10"]()
    tasks = {}
    for t in EVAL_TASKS:
        lang = bm.get_task(t).language
        n_init = len(bm.get_task_init_states(t))
        if desc.get(t) != lang:
            raise SystemExit(f"задача {t}: описание плана A {desc.get(t)!r} ≠ "
                             f"LIBERO {lang!r}")
        if n_init < 45:
            raise SystemExit(f"задача {t}: начальных состояний {n_init} < 45 "
                             f"— банки 0…44 не помещаются")
        tasks[str(t)] = dict(description=lang, n_init_states=int(n_init))
    pins, init_ref, outc = k25_refs()
    import k29_policy as kp
    U, r = kp.harness_utils()
    rep = dict(
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        prep_id=prep["prep_id"], prepare_sha1=sha(d21.REPORT),
        bar_revision=prep["bar_pin"]["revision"], snapshot=snap,
        bounds=prep["bounds_codec"], state_norm=prep["state_norm"],
        sources=srcs, tasks=tasks, banks=banks(), env=kc.ENV,
        cycle_max=kc.CYCLE_MAX, wave_max=kc.WAVE_MAX, dist=kd.CONFIG,
        ppo=kpo.CFG, precision=dict(trunk="fp16 autocast, no_grad",
                                    decoders="float32 вне autocast, TF32 off",
                                    critic="float32"),
        trainable=list(__import__("k29_model").TRAINABLE),
        k25_refs=pins, init_ref=init_ref, k25_outcomes=outc,
        offset_table_sha1=sha("data/pos_offset_table.json"),
        harness_cfg_sha1=sha(os.path.join(r, "config/eval/bar.yaml")),
        code=code_sha())
    return rep


CONTRACT_KEYS = ("prep_id", "prepare_sha1", "bar_revision", "bounds",
                 "state_norm", "sources", "tasks", "banks", "env", "cycle_max",
                 "dist", "ppo", "precision", "trainable", "init_ref",
                 "offset_table_sha1", "harness_cfg_sha1", "code")


def prepare():
    rep = make_prepare()
    if os.path.exists(PREPARE):
        old = json.load(open(PREPARE, encoding="utf-8"))
        diff = [k for k in CONTRACT_KEYS if old.get(k) != rep.get(k)]
        if diff:
            raise SystemExit(f"{PREPARE} уже закреплён; изменилось: {diff} — "
                             f"новая постановка требует нового каталога")
        print(f"  {PREPARE}: закреплён, сверка пройдена")
        return 0
    os.makedirs("reports/k29", exist_ok=True)
    with open(PREPARE + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    os.replace(PREPARE + ".tmp", PREPARE)
    print(f"  {PREPARE}: источники " + ", ".join(
        f"s{s} {e['weights_hash'][:12]} (обн. {e['update']})"
        for s, e in rep["sources"].items()))
    for t, v in rep["tasks"].items():
        print(f"    задача {t}: «{v['description']}», начальных состояний "
              f"{v['n_init_states']}")
    return 0


def load_prepare():
    if not os.path.exists(PREPARE):
        raise SystemExit("нет подготовки: k29_protocol.py prepare")
    p = json.load(open(PREPARE, encoding="utf-8"))
    bad = [f for f, s in p["code"].items() if sha(os.path.join(HERE, f)) != s]
    if bad:
        raise SystemExit(f"код изменён после prepare: {bad}")
    return p


def contract(p, seed=None):
    c = {k: p[k] for k in CONTRACT_KEYS}
    c = dict(fingerprint=fp(c))
    if seed is not None:
        c["seed"] = int(seed)
    return c


# --- общие загрузки ---------------------------------------------------------------
def load_all(torch, p, seed, device):
    import k29_model as km
    km.no_tf32(torch)
    e = p["sources"][str(seed)]
    mod, bar, _prep = km.load_source(torch, e, device)
    if km.frozen_hash(mod) != e["weights_hash"]:
        raise SystemExit("источник после загрузки ≠ отпечатку")
    M = km.build(torch)
    actor = M["Decoder"](mod)
    ref = M["Decoder"](mod).freeze()
    critic = M["Critic"](mod.H, torch.Generator().manual_seed(
        29000 + int(seed))).to(next(mod.parameters()).device)
    return mod, bar, M, actor, ref, critic


# --- real-check -------------------------------------------------------------------
def real_check(a):
    import torch
    import k21_data as d21
    import k25_policy as kp25
    import k29_model as km
    p = load_prepare()
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    val = np.asarray(D["val_rows"])
    task = np.asarray(D["task"])[val]
    t0 = sorted(set(task.tolist()))[0]
    one = val[task == t0][:5]
    rows40 = np.concatenate([val[task == t][:5] for t in
                             sorted(set(task.tolist()))[:8]])
    rep, prob = dict(seeds={}), []
    ac = torch.autocast("cuda", dtype=torch.float16)
    for s in SEEDS:
        r = {}
        mod, bar, M, actor, ref, critic = load_all(torch, p, s, a.device)
        for B in (1, 5):
            batch, po, st, _t, _v = d21.make_batch(D, one[:B], bar, dev)
            with torch.no_grad():
                with ac:
                    o = mod(batch, po, st)
                    F, s11n, cnt = M["export"](mod, batch, po, st)
                leg = ref(F, autocast=ac)
                fa, fr = actor(F), ref(F)
            same_leg = all(torch.equal(x, y) for x, y in (
                (leg["arm"], o["arm"]), (leg["grip_logit"], o["grip_logit"]),
                (leg["Z"], o["y"])))
            same_ar = all(torch.equal(fa[k], fr[k]) for k in
                          ("Z", "arm", "grip_logit"))
            r[f"B{B}"] = dict(
                legacy_equal_k25=same_leg, actor_equal_reference=same_ar,
                fp32_vs_k25=dict(
                    Z=float((fr["Z"] - o["y"].float()).abs().max()),
                    arm=float((fr["arm"] - o["arm"].float()).abs().max()),
                    grip=float((fr["grip_logit"] - o["grip_logit"].float())
                               .abs().max()),
                    grip_sign_diff=int(((fr["grip_logit"] > 0) !=
                                        (o["grip_logit"] > 0)).sum())),
                counters=kp25.counters_problems(cnt))
            if not same_leg:
                prob.append(f"s{s} B{B}: экспорт + декодер источника ≠ K-25")
            if not same_ar:
                prob.append(f"s{s} B{B}: actor FP32 ≠ reference FP32")
            if r[f"B{B}"]["counters"]:
                prob.append(f"s{s} B{B}: счётчики {r[f'B{B}']['counters']}")
        # кеш: online B = 5 против рабочих minibatch
        g = np.random.default_rng(29)
        mus, Fs, ss = [], [], []
        torch.cuda.reset_peak_memory_stats(dev)
        t_exp = []
        for i in range(0, len(rows40), 5):
            batch, po, st, _t, _v = d21.make_batch(D, rows40[i:i + 5], bar,
                                                   dev)
            torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            with torch.no_grad():
                with ac:
                    F, s11n, _c = M["export"](mod, batch, po, st)
                mus.append(M["mu_of"](actor(F)).float().cpu())
                ref(F)
            torch.cuda.synchronize(dev)
            t_exp.append((time.perf_counter() - t1) * 1e3)
            Fs.append({k: v.cpu() for k, v in F.items()})
            ss.append(s11n.cpu())
        r["peak_gb_export_b5"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        r["ms_export_decode_b5"] = float(np.median(t_exp))
        N = len(rows40)
        buf = dict(F={k: torch.cat([f[k] for f in Fs]) for k in Fs[0]},
                   s11n=torch.cat(ss), mu=torch.cat(mus))
        buf["mu_ref"] = buf["mu"].clone()
        buf["Y"] = buf["mu"] + torch.as_tensor(
            kd.sample_noise(g, N), dtype=torch.float32)
        buf["k"] = g.integers(1, 9, N)
        buf["R"] = (g.random(N) < 0.5).astype(np.float32)
        buf["elapsed"] = g.integers(0, 600, N)
        buf["episode"] = np.arange(N) // 3
        for mbs in (8, 32, 64):
            mu = kpo.mu_all(torch, actor, M, buf, dev, mbs)
            lp_a = kd.log_prob(torch, buf["Y"], mu, torch.as_tensor(buf["k"]))
            lp_b = kd.log_prob(torch, buf["Y"], buf["mu"],
                               torch.as_tensor(buf["k"]))
            dm = float((mu - buf["mu"]).abs().max())
            dl = float((lp_a - lp_b).abs().max())
            r[f"cache_mb{mbs}"] = dict(mu_max=dm, logp_max=dl)
            if not (dm <= kpo.CFG["mu_tol"] and dl <= kpo.CFG["logp_tol"]):
                prob.append(f"s{s}: кеш minibatch {mbs}: Δμ {dm:.2e}, "
                            f"Δlog π {dl:.2e}")
        # память и время PPO-пересчёта
        for mbs in (8, 32, 64):
            torch.cuda.reset_peak_memory_stats(dev)
            idx = np.arange(min(mbs, N))
            torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            mu = M["mu_of"](actor(kpo.to_dev(buf["F"], idx, dev)))
            mu.float().sum().backward()
            torch.cuda.synchronize(dev)
            actor.zero_grad(set_to_none=True)
            r[f"ppo_mb{mbs}"] = dict(
                peak_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                ms_fwd_bwd=(time.perf_counter() - t1) * 1e3)
        # полный шаг PPO
        h_a, h_c = km.tensor_hash(actor.state_dict()), km.tensor_hash(
            critic.state_dict())
        h_r = km.tensor_hash(ref.state_dict())
        oa = torch.optim.AdamW(actor.parameters(), lr=kpo.CFG["lr_actor"],
                               weight_decay=0.0)
        oc = torch.optim.Adam(critic.parameters(), lr=kpo.CFG["lr_critic"])
        cfg1 = dict(kpo.CFG, epochs=1)
        stp = kpo.ppo_update(torch, M, actor, critic, oa, oc, buf, dev,
                             (s, 99, 0), cfg1)
        r["ppo_step"] = dict(
            epochs=stp["epochs"], rollback=stp["rollback"],
            mu_recompute=stp["mu_recompute_max"],
            actor_changed=km.tensor_hash(actor.state_dict()) != h_a,
            critic_changed=km.tensor_hash(critic.state_dict()) != h_c,
            reference_same=km.tensor_hash(ref.state_dict()) == h_r,
            source_same=km.frozen_hash(mod) == p["sources"][str(s)][
                "weights_hash"],
            finite=all(bool(torch.isfinite(q).all()) for q in
                       list(actor.parameters()) + list(critic.parameters())))
        bad = [k for k in ("actor_changed", "critic_changed", "reference_same",
                           "source_same", "finite") if not r["ppo_step"][k]]
        if bad:
            prob.append(f"s{s}: шаг PPO: {bad}")
        r["params"] = dict(
            actor=sum(q.numel() for q in actor.parameters()),
            critic=sum(q.numel() for q in critic.parameters()),
            trainable_keys=sorted(actor.trainable()))
        rep["seeds"][str(s)] = r
        print(f"  seed {s}: " + json.dumps({k: v for k, v in r.items()
                                            if k != "params"}, default=float),
              flush=True)
        print(f"    обучаемых параметров actor {r['params']['actor']}, critic "
              f"{r['params']['critic']}", flush=True)
        del mod, bar, actor, ref, critic, buf
        torch.cuda.empty_cache()
    rep["problems"] = prob
    rep["contract"] = check_contract(a)
    with open(REAL_CHECK + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(REAL_CHECK + ".tmp", REAL_CHECK)
    print("  " + ("ОТКАЗ: " + "; ".join(prob) if prob else
                  "real-check K-29 пройден") + f"; {REAL_CHECK}", flush=True)
    return 3 if prob else 0


def check_contract(a):
    import torch
    p = load_prepare()
    return dict(prepare=contract(p)["fingerprint"],
                device=torch.cuda.get_device_name(torch.device(a.device)),
                torch=torch.__version__)


def check_fresh(a):
    path = REAL_CHECK if a.kind == "real-check" else SMOKE
    if not os.path.exists(path):
        print(f"  {path}: нет")
        return 1
    r = json.load(open(path, encoding="utf-8"))
    ok = r.get("problems") == [] and r.get("contract") == check_contract(a)
    print(f"  {path}: " + ("пройден тем же контрактом" if ok else
                           "пересчитать"))
    return 0 if ok else 1


# --- collector-smoke -----------------------------------------------------------------
def smoke(a):
    import k18_collector as k18c
    sim = k18c.Sim(5, "persistent")         # среды до CUDA
    import torch
    import k29_policy as kp
    p = load_prepare()
    try:
        mod, bar, M, actor, ref, critic = load_all(torch, p, 0, a.device)
        dev = torch.device(a.device)
        inputs = kp.Inputs(torch, bar["proc"], a.device)
        pol = kp.Policy(torch, mod, actor, ref, M, "actor")
        sent = []
        step0 = sim.step

        def step(acts):
            sent.append(np.asarray(acts, np.float32).copy())
            return step0(acts)
        sim.step = step
        prob, waves, bufs, phys = [], [], [], 0
        for w, task in enumerate((0, 8)):
            starts = kpo.pick_init(0, 10_000 + w)
            sent.clear()
            eps, buf, log, c = kc.run_wave(
                sim, inputs, pol, task, starts,
                noise_gen=np.random.default_rng([0, 8, w]))
            phys += c["physical"]
            A = log["actions"]
            exec_sent = np.stack(sent[kc.ENV["waiting_steps"]:]) if \
                len(sent) > kc.ENV["waiting_steps"] else np.zeros((0, 5, 7))
            if not np.array_equal(exec_sent, A):
                prob.append(f"волна {w}: переданные ≠ записанным")
            if c["physical"] != 5 * (10 + c["steps"]) or \
                    len(sent) != 10 + c["steps"]:
                prob.append(f"волна {w}: учёт шагов")
            # рука = mid + half·tanh(Y/half) по записанным Y активных сред
            half, mid = pol.half.cpu(), pol.mid.cpu()
            arm = (mid + half * torch.tanh(buf["Y"] / half)).numpy()
            rec = np.concatenate([ch["a_codec"][ch["active"]] for ch in
                                  log["chunks"] if ch["active"].any()])
            if not np.array_equal(arm, rec[..., :6]):
                prob.append(f"волна {w}: рука ≠ tanh(Y)")
            for i, e in enumerate(eps):
                kk = buf["k"][buf["env"] == i]
                if kk.sum() != e["own_steps"]:
                    prob.append(f"волна {w} среда {i}: Σk ≠ own_steps")
                if set(buf["R"][buf["env"] == i].tolist()) != {
                        float(e["success"])}:
                    prob.append(f"волна {w} среда {i}: награда")
            waves.append(dict(task=task, starts=starts, counts=c,
                              success=[e["success"] for e in eps],
                              done_step=[e["done_step"] for e in eps],
                              init_hash=[e["init_hash"] for e in eps]))
            bufs.append(buf)
        add_resource("smoke", phys)
        if phys > SMOKE_MAX:
            prob.append(f"smoke {phys} > {SMOKE_MAX}")
        buf = kc.concat(bufs)
        mu = kpo.mu_all(torch, actor, M, buf, dev, kpo.CFG["minibatch"])
        kk = torch.as_tensor(buf["k"])
        dm = float((mu - buf["mu"]).abs().max())
        dl = float((kd.log_prob(torch, buf["Y"], mu, kk) -
                    kd.log_prob(torch, buf["Y"], buf["mu"], kk)).abs().max())
        if not (dm <= kpo.CFG["mu_tol"] and dl <= kpo.CFG["logp_tol"]):
            prob.append(f"пересчёт по кешу: Δμ {dm:.2e}, Δlog π {dl:.2e}")
        rep = dict(waves=waves, physical=phys, transitions=int(len(buf["k"])),
                   recompute=dict(mu_max=dm, logp_max=dl),
                   k_hist=np.bincount(buf["k"], minlength=9).tolist(),
                   problems=prob, contract=check_contract(a))
        with open(SMOKE + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=1, default=float)
        os.replace(SMOKE + ".tmp", SMOKE)
        for wv in waves:
            print(f"  smoke задача {wv['task']}, init {wv['starts']}: успех "
                  f"{sum(wv['success'])}/5, шагов {wv['counts']['physical']}, "
                  f"{wv['counts']['seconds']:.0f} с", flush=True)
        print(f"  пересчёт по кешу: Δμ {dm:.1e}, Δlog π {dl:.1e}; k "
              f"{rep['k_hist']}", flush=True)
        print("  " + ("ОТКАЗ: " + "; ".join(prob) if prob else
                      "collector-smoke пройден") + f"; {SMOKE}", flush=True)
        return 3 if prob else 0
    finally:
        sim.close()


# --- обучение --------------------------------------------------------------------
def final_path(seed):
    return os.path.join(kpo.run_dir(seed), "final.pt")


def train(a):
    import k18_collector as k18c
    if a.seed not in SEEDS:
        raise SystemExit(f"seed {a.seed}: только {SEEDS}")
    sim = k18c.Sim(5, "persistent")
    import torch
    import k29_model as km
    import k29_policy as kp
    p = load_prepare()
    try:
        mod, bar, M, actor, ref, critic = load_all(torch, p, a.seed, a.device)
        inputs = kp.Inputs(torch, bar["proc"], a.device)
        tr = kpo.Trainer(torch, a.seed, M, mod, actor, ref, critic, sim,
                         inputs, kp.Policy, torch.device(a.device),
                         contract(p, a.seed),
                         log=lambda *x: print(*x, flush=True))
        tr.src_hash = p["sources"][str(a.seed)]["weights_hash"]
        st = tr.run()
        tr.check_frozen()
        h = tr.actor_hash()
        fpath = final_path(a.seed)
        if os.path.exists(fpath):
            old = torch.load(fpath, map_location="cpu", weights_only=False)
            if old["actor_hash"] != h:
                raise SystemExit(f"{fpath} уже закреплён с другим actor")
        else:
            torch.save(dict(actor=actor.state_dict(), actor_hash=h,
                            seed=a.seed, contract=contract(p, a.seed),
                            accepted=st["accepted"], updates=st["update"]),
                       fpath + ".tmp")
            os.replace(fpath + ".tmp", fpath)
        rep = dict(seed=a.seed, final=os.path.abspath(fpath),
                   final_sha1=sha(fpath), actor_hash=h,
                   reference_hash=km.tensor_hash(ref.state_dict()),
                   changed=h != km.tensor_hash(ref.state_dict()),
                   accepted_updates=st["accepted"], updates=st["update"],
                   ledger=tr.ledger.d, validation=st["val"],
                   update_stats=st["updates"], episodes=st["episodes"],
                   transitions=st["transitions"],
                   policy_steps=st["policy_steps"], contract=contract(
                       p, a.seed),
                   note="итог — последний принятый actor на исчерпании "
                        "заданного бюджета; валидация только описательна")
        os.makedirs("reports/k29", exist_ok=True)
        with open(kpo.report_path(a.seed), "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=1, default=float)
        print(f"  K-29 seed {a.seed}: обновлений {st['update']} (принято "
              f"{st['accepted']}), бюджет {tr.ledger.d['train_spent']}/"
              f"{tr.ledger.cap}; actor " + ("изменён" if rep["changed"] else
                                            "НЕ изменён") +
              f"; {kpo.report_path(a.seed)}", flush=True)
        return 0
    finally:
        sim.close()


def selftest():
    b = banks()
    assert set(b["train"]["init"]).isdisjoint(b["eval"]["init"])
    assert max(b["eval"]["init"]) < 45 and b["closed"] == list(range(45, 50))
    assert fp(dict(a=1)) == fp(dict(a=1)) != fp(dict(a=2))
    assert len(set(MODEL_CODE)) == len(MODEL_CODE)
    for f in MODEL_CODE:
        assert os.path.exists(os.path.join(HERE, f)), f
    print("самопроверка k29_protocol пройдена: банки не пересекаются, 45…49 "
          "закрыты, отпечатки, список кода существует")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-29")
    ap.add_argument("cmd", choices=("prepare", "real-check", "smoke", "train",
                                    "check-fresh", "selftest"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", choices=("real-check", "smoke"),
                    default="real-check")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    try:
        return dict(prepare=lambda: prepare(), smoke=lambda: smoke(a),
                    train=lambda: train(a),
                    **{"real-check": lambda: real_check(a),
                       "check-fresh": lambda: check_fresh(a)})[a.cmd]()
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
