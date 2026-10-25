#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-24: обучение parents_mr24 (seed 0 и 1), real-check и fitcheck.

Тот же k21_train (файлы K-21/K-22/K-23 не меняются). Подмены в процессе:
  - построитель -> K24 (readout mr1) с seed=S; очередь — k23_train.
    seeded_schedule(S) ДО fastload (её же получает предзагрузчик); панель —
    прежняя K-21 (k21_train.SEED + 5);
  - каталоги data/k24/runs/s<S>/parents_mr24, отчёты reports/k24/;
  - контракт: seed, архитектура, readout (версия, формула, FP32 QR, порог
    ранга), бюджет, отпечаток очереди, SHA кода K-24 и переиспользованного;
  - бюджет main ЗАКРЕПЛЁН по parents-контролю того же сида (из
    reports/k24/prepare.json), без индивидуального добора: Trainer.run(B);
  - на каждой проверке — диагностика уровней на 64 фиксированных строках
    val_sel: RMS и доли энергии G0/G1/G2, ошибки тождеств, мин. σ W0/W1,
    насыщение; счётчик шагов и пропусков GradScaler (порог K-23 — 50 подряд).
Режимы:
  real-check --seed 0 [--micros 8 32] — настоящий BAR (V100): forward, loss,
    backward, unscale/clip, AdamW lr 0; память, конечность, счётчики,
    пределы, ранг, тождества (≤ 5e-5), градиенты (k23_smoke); legacy-путь на
    весах K-22 best = K-22 (|Δarm| ≤ 1e-6, отн. raw ≤ 1e-5); начальные веса
    K24(S) = K-22(S) и отпечаток = историческому init_hash;
  fitcheck — k21_train.fitcheck на K24 (128 строк, 500 итераций, lr×3,
    прежние критерии), отдельный каталог, после — пересоздание и init_hash;
  train --seed S — основное обучение до закреплённого бюджета.
"""
import argparse
import builtins
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k21_train as kt  # noqa: E402
import k23_train as k23t  # noqa: E402
import k24_model as k24m  # noqa: E402
import k24_readout as kr  # noqa: E402

SEEDS = (0, 1)
MODE = k24m.MODE
PREPARE = "reports/k24/prepare.json"
CODE = ("k24_readout.py", "k24_model.py", "k24_train.py", "k22_model.py",
        "k21_fastload.py", "k23_train.py", "k23_smoke.py")
DEPS = CODE + ("k21_train.py", "k21_model.py", "k21_tree.py",
               "k21_data.py", "k20_data.py", "k16_models.py",
               "k24_protocol.py", "k23_protocol.py")
TRAIN_MICRO = 32
DIAG_N, DIAG_SEED = 64, 20261025
HIST_INIT = {0: "data/k22/runs/parents24/resume.pt",
             1: "data/k23/runs/s1/parents24/resume.pt"}


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def outdir(seed):
    return f"data/k24/runs/s{seed}"


def report_path(seed):
    return f"reports/k24/train_s{seed}.json"


def seed_entry(seed):
    """Запись сида из prepare.json с проверкой отпечатка (k24_protocol)."""
    import k24_protocol as pc
    if not os.path.exists(PREPARE):
        raise SystemExit(f"нет {PREPARE}: сначала k24_protocol.py prepare")
    return pc.seed_entry(seed)


def check_contract(kind, a):
    """Контракт проверки: при любом расхождении готовый отчёт не
    переиспользуется (зависимости, данные, BAR, точность, устройство,
    microbatch; для fitcheck — конфигурация подгонки)."""
    import torch
    import k21_data as d21
    import k24_protocol as pc
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    tj = json.load(open(d21.TEACHER_REPORT, encoding="utf-8"))
    c = dict(kind=kind,
             code={f: sha(os.path.join(HERE, f)) for f in DEPS},
             prep_id=prep["prep_id"], prepare_sha1=pc.full_sha(d21.REPORT),
             teacher_sha1=tj["out"]["sha1"],
             teacher_report_sha1=pc.full_sha(d21.TEACHER_REPORT),
             bar_revision=prep["bar_pin"]["revision"],
             bar_files=hashlib.sha1(json.dumps(
                 prep["bar_pin"]["files"], sort_keys=True).encode())
             .hexdigest(),
             precision="float16 autocast + GradScaler",
             device=torch.cuda.get_device_name(torch.device(a.device)),
             seed_entry_fp_s0=seed_entry(0)["fingerprint"])
    if kind == "real-check":
        c.update(micros=sorted(a.micros), seed=a.seed,
                 train_micro=TRAIN_MICRO)
    else:
        c.update(fit=dict(n=kt.FIT_N, updates=kt.FIT_UPDATES, lr_mult=3,
                          ratio=kt.FIT_RATIO, grip=kt.FIT_GRIP,
                          rows_seed=kt.SEED + 9, micro=a.micro,
                          eval_micro=a.eval_micro, seed=0))
    return c


def check_fresh(a):
    p = dict(real_check="reports/k24/real_check.json",
             fitcheck="reports/k24/fitcheck.json")[a.kind.replace("-", "_")]
    if not os.path.exists(p):
        print(f"  {p}: нет")
        return 1
    r = json.load(open(p, encoding="utf-8"))
    passed = (r.get("problems") == []) if a.kind == "real-check" else \
        (bool(r.get("ok")) and r.get("exit") == 0)
    want = check_contract(a.kind, a)
    got = r.get("contract") or {}
    diff = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
    ok = passed and not diff
    print(f"  {p}: " + ("пройден тем же контрактом" if ok else
                        f"пересчитать (пройден: {passed}; расхождения "
                        f"контракта: {diff})"))
    return 0 if ok else 1


def make_model_fn(seed, readout_mode="mr"):
    def make_model(torch, mode, D, dev, pin, checkpoint=True):
        import k21_data as d21
        bar = d21.load_bar(dev, pin)
        m = bar["model"]
        m.action_expert.float()
        M = k24m.build(torch)
        mod = M["K24"](m, D["L"], D["U"], D["state_mean"], D["state_std"],
                       checkpoint=checkpoint, seed=seed,
                       readout_mode=readout_mode).to(dev)
        return mod, bar
    return make_model


def diag_rows(D):
    vr = np.asarray(D["val_rows"])
    return np.sort(np.random.default_rng(DIAG_SEED).choice(
        vr, min(DIAG_N, len(vr)), replace=False))


def mr_diag(torch, mod, bar, D, rows, dev, dt, micro):
    import k21_model as km
    R = kr.build(torch)
    agg = dict(e=[], idv=[], r0=[], r1=[], sat=[])
    L = torch.as_tensor(D["L"], device=dev)
    U = torch.as_tensor(D["U"], device=dev)
    was = mod.training
    mod.eval()
    with torch.no_grad():
        for s0 in range(0, len(rows), micro):
            out = kt.run_micro(torch, km, mod, bar, D, rows[s0:s0 + micro],
                               "expert", dev, dt)[0]
            mr = out["mr"]
            agg["e"].append(R["energy"](mr))
            agg["idv"].append(R["identity_errors"](mr))
            agg["r0"].append(float(mr["rank0"].min()))
            agg["r1"].append(float(mr["rank1"].min()))
            a = out["arm"].float()
            agg["sat"].append(float((((a - L).abs() < 0.01 * (U - L)) |
                                     ((U - a).abs() < 0.01 * (U - L)))
                                    .float().mean()))
    if was:
        mod.train()
    e = {k: float(np.mean([x[k] for x in agg["e"]])) for k in agg["e"][0]}
    return dict(e, identity_P0Z=max(x["P0Z"] for x in agg["idv"]),
                identity_P1Z=max(x["P1Z"] for x in agg["idv"]),
                orth=max(x["orth"] for x in agg["idv"]),
                min_rank_ratio_W0=min(agg["r0"]),
                min_rank_ratio_W1=min(agg["r1"]),
                saturation=float(np.mean(agg["sat"])), rows=len(rows))


def install(seed, budget, torch=None, fastload=True, fit_dir=None,
            entry_fp=None):
    if seed not in SEEDS:
        raise SystemExit(f"training seed {seed}: в K-24 только {SEEDS}")
    sched = k23t.seeded_schedule(kt.schedule, seed)
    kt.schedule = sched
    kt.make_model = make_model_fn(seed)
    kt.OUTDIR = fit_dir or outdir(seed)
    redirect = {"reports/k21/fitcheck.json": "reports/k24/fitcheck.json",
                "reports/k21/train.json": report_path(seed) + ".unused"}

    def _open(path, *a, **k):
        n = os.path.normpath(str(path))
        for src, dst in redirect.items():
            if n == os.path.normpath(src) and a and "w" in a[0]:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                path = dst
        return builtins.open(path, *a, **k)
    kt.open = _open
    orig_contract = kt.run_contract
    fp_cache = {}

    def run_contract(mode, a, D):
        c = orig_contract(mode, a, D)
        if a.micro not in fp_cache:
            fp_cache[a.micro] = k23t.schedule_fp(sched, D, a.micro)
        c.update(seed=seed, architecture=k24m.ARCHITECTURE, readout=dict(
            version=kr.VERSION, formula=kr.FORMULA, qr="float32, autocast "
            "off, reduced", rank_tol=kr.RANK_TOL),
            budget=dict(warmup=kt.WARM_UPDATES, main=budget,
                        main_max_scheduler=kt.MAIN_MAX, extension=False),
            k24=dict(train_seed=seed, schedule_fp=fp_cache[a.micro],
                     seed_entry_fp=entry_fp,
                     panel="K-21 (k21_train.SEED + 5)"))
        c["code"] = dict(c["code"], **{f: sha(os.path.join(HERE, f))
                                       for f in CODE})
        return c
    kt.run_contract = run_contract
    orig_check = kt.Trainer.check

    def check(self, opt, u):
        orig_check(self, opt, u)
        sc = self.scaler
        c = dict(getattr(sc, "_k23", {}))
        c["scale"] = float(sc.get_scale()) if sc.is_enabled() else None
        rec = self.st["curve"][-1]
        rec["scaler"] = c
        t0 = time.time()
        rec["mr_diag"] = mr_diag(torch, self.mod, self.bar, self.D,
                                 diag_rows(self.D), self.dev, self.dt,
                                 self.a.eval_micro)
        d = rec["mr_diag"]
        print(f"    {self.tag}: шагов {c.get('steps')}, пропусков GradScaler "
              f"{c.get('skips')}; энергия G0/G1/G2 {d['share_G0']:.2f}/"
              f"{d['share_G1']:.2f}/{d['share_G2']:.2f}, тождества "
              f"{max(d['identity_P0Z'], d['identity_P1Z']):.1e}, мин σ "
              f"{d['min_rank_ratio_W0']:.2e}/{d['min_rank_ratio_W1']:.2e}, "
              f"насыщение {d['saturation']:.3f} ({time.time() - t0:.0f} с)",
              flush=True)
    kt.Trainer.check = check
    if torch is not None:
        torch.cuda.amp.GradScaler = k23t.counting_scaler(
            torch.cuda.amp.GradScaler)
    st = None
    if fastload:
        import k21_fastload as fl
        st = fl.install()
    return dict(fastload=st, schedule=sched)


def hist_init_hash(torch, seed):
    p = HIST_INIT[seed]
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location="cpu", weights_only=False)["init_hash"]


def train(a):
    import torch
    import k21_data as d21
    e = seed_entry(a.seed)
    budget = int(e["budget_main"])
    st = install(a.seed, budget, torch, entry_fp=e["fingerprint"])
    D = d21.load_prepared(require_teacher=True)
    have = os.path.exists(os.path.join(outdir(a.seed), MODE, "resume.pt"))
    print(f"  K-24 seed {a.seed}: бюджет main {budget} (закреплён по "
          f"контролю {e['control_parents']['source']}); "
          + ("продолжение" if have else "новый запуск"), flush=True)
    tr = kt.Trainer(torch, a, D, MODE, MODE)
    want = e["control_parents"]["init_hash"]
    if tr.init_hash != want:
        raise SystemExit(f"начальные веса {tr.init_hash} ≠ контролю {want}")
    tr.run(budget)
    rs = torch.load(os.path.join(outdir(a.seed), MODE, "resume.pt"),
                    map_location="cpu", weights_only=False)
    b = torch.load(os.path.join(outdir(a.seed), MODE, "best.pt"),
                   map_location="cpu", weights_only=False)
    if rs["u"] != budget or rs["phase"] != "main":
        raise SystemExit(f"обучение не дошло до бюджета: {rs['phase']} "
                         f"{rs['u']}/{budget}")
    rep = dict(seed=a.seed, mode=MODE, budget_main=budget,
               updates=rs["u"], best=rs["best"],
               best_sha1=kt.sha(os.path.join(outdir(a.seed), MODE, "best.pt")),
               best_weights_hash=b["weights_hash"], best_update=b["update"],
               init_hash=rs["init_hash"],
               scaler=(rs.get("scaler") or {}).get("_k23"),
               curve=rs["curve"], config=rs["config"])
    os.makedirs("reports/k24", exist_ok=True)
    json.dump(rep, open(report_path(a.seed), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    pf = st["fastload"]["pf"] if st["fastload"] else None
    if pf is not None:
        print(f"  fastload: из предзагрузки {pf.hits}, синхронно {pf.misses}")
    print(f"  K-24 seed {a.seed}: лучшая {rs['best']}, отчёт "
          f"{report_path(a.seed)}", flush=True)
    return 0


def fitcheck(a):
    import torch
    import k21_data as d21
    e0 = seed_entry(0)
    install(0, e0["budget_main"], torch, fastload=False,
            fit_dir="data/k24/fitcheck", entry_fp=e0["fingerprint"])
    D = d21.load_prepared(require_teacher=True)
    code = kt.fitcheck(torch, a, D)
    p = "reports/k24/fitcheck.json"
    if os.path.exists(p):
        r = json.load(open(p, encoding="utf-8"))
        r["contract"] = check_contract("fitcheck", a)
        r["exit"] = code
        json.dump(r, open(p, "w", encoding="utf-8"), indent=1)
    return code


def real_check(a):
    import torch
    import k21_data as d21
    import k21_model as km
    import k22_model as k22
    import k23_smoke as sm
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=True)
    pin = D["prep_report"]["bar_pin"]
    M = k24m.build(torch)
    R = M["R"]
    prob, rep = [], dict(seed=a.seed, micros=a.micros)
    if TRAIN_MICRO not in a.micros:
        prob.append(f"microbatch обучения {TRAIN_MICRO} не проверяется")
    rows = D["train_rows"][:: max(1, len(D["train_rows"]) // 64)][:64]

    def fresh(cls, seed, **kw):
        bar = d21.load_bar(dev, pin)
        m = bar["model"]
        m.action_expert.float()
        return cls(m, D["L"], D["U"], D["state_mean"], D["state_std"],
                   checkpoint=True, seed=seed, **kw).to(dev), bar
    # 1. инициализация K24(S) = K-22(S) и = историческому отпечатку
    for s in SEEDS:
        a24, _b = fresh(M["K24"], s)
        h24 = kt.state_hash(kt.trainable_state(a24))
        del a24, _b
        torch.cuda.empty_cache()
        hh = hist_init_hash(torch, s)
        rep[f"init_s{s}"] = dict(k24=h24, historical=hh,
                                 equal=(hh == h24) if hh else None)
        if hh is not None and hh != h24:
            prob.append(f"seed {s}: начальные веса ≠ историческим")
    # 2. шаги на micro 8 и 32
    mod, bar = fresh(M["K24"], a.seed)
    named = [(n, p) for n, p in mod.named_parameters() if p.requires_grad]
    gn = sm.groups_named(mod)
    opt = torch.optim.AdamW([p for _n, p in named], lr=0.0)
    scaler = torch.cuda.amp.GradScaler()
    last = len(mod.bar.vlm.text_model.layers) - 1
    Lt = torch.as_tensor(D["L"], device=dev)
    Ut = torch.as_tensor(D["U"], device=dev)
    for mb in a.micros:
        r = {}
        try:
            batch, po, st, tgt, val = d21.make_batch(D, rows[:mb], bar, dev)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.time()
            mod.train()
            with torch.autocast("cuda", dtype=torch.float16):
                out = mod(batch, po, st)
                loss, _l1, _b = km.loss_fn(torch, out, tgt, val)
            scaler.scale(loss.mean()).backward()
            gp, none, norms = sm.grad_problems(named, gn, last)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for _n, p in named], 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(dev)
            ie = R["identity_errors"](out["mr"])
            arm = out["arm"].float()
            c = mod.counters
            checks = dict(
                counters=c["active"] == [4] * 8 + [12] * 8 + [28] * 8
                and c["vision"] == 1 and all(x == 1 for x in c["text"]),
                finite=bool(torch.isfinite(arm).all()
                            and torch.isfinite(loss).all()),
                bounds=bool(((arm >= Lt - 1e-4) & (arm <= Ut + 1e-4)).all()),
                shapes=list(out["mr"]["Y2"].shape) == [mb, 20, 7],
                identities=max(ie.values()) <= kr.ID_TOL_MIXED,
                grads=not gp)
            r = dict(checks=checks, seconds=time.time() - t0,
                     peak_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                     identity=ie, rank0=float(out["mr"]["rank0"].min()),
                     rank1=float(out["mr"]["rank1"].min()),
                     group_grad_norms=norms, no_grad_params=none,
                     energy=R["energy"](out["mr"]))
            bad = [k for k, v in checks.items() if not v]
            if bad:
                prob.append(f"micro {mb}: {bad}")
        except torch.cuda.OutOfMemoryError:
            r = dict(oom=True)
            prob.append(f"micro {mb}: OOM")
        rep[f"micro_{mb}"] = r
        print(f"  micro {mb}: {json.dumps({k: r.get(k) for k in ('checks', 'peak_gb', 'seconds', 'identity', 'rank0', 'rank1')}, default=float)}",
              flush=True)
    del mod, bar, opt
    torch.cuda.empty_cache()
    # 3. legacy-путь = K-22 на весах K-22 best
    ck = torch.load("data/k22/runs/parents24/best.pt", map_location="cpu",
                    weights_only=False)
    outs = {}
    for name, cls, kw in (("k22", k22.build(torch)["K22"], {}),
                          ("legacy", M["K24"], dict(readout_mode="legacy"))):
        m_, b_ = fresh(cls, 0, **kw)
        rep.setdefault("params", {})[name] = dict(
            total=sum(p.numel() for p in m_.parameters()),
            trainable=sum(p.numel() for g in m_.groups().values() for p in g),
            tensors=len(kt.trainable_state(m_)))
        own = dict(m_.named_parameters())
        own.update(dict(m_.named_buffers()))
        with torch.no_grad():
            for k, v in ck["weights"].items():
                own[k].copy_(v.to(own[k].device, own[k].dtype))
        m_.eval()
        batch, po, st, _t, _v = d21.make_batch(D, rows[:8], b_, dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            o = m_(batch, po, st)
        outs[name] = {k: o[k].float().cpu() for k in ("arm", "y")}
        del m_, b_
        torch.cuda.empty_cache()
    da = float((outs["k22"]["arm"] - outs["legacy"]["arm"]).abs().max())
    yr = float((outs["k22"]["y"] - outs["legacy"]["y"]).abs().max() /
               max(1.0, float(outs["k22"]["y"].abs().max())))
    rep["legacy_vs_k22"] = dict(arm_abs=da, raw_rel=yr)
    if rep["params"]["k22"] != rep["params"]["legacy"]:
        prob.append(f"число параметров K24 ≠ K22: {rep['params']}")
    if not (da <= 1e-6 and yr <= 1e-5):
        prob.append(f"legacy ≠ K-22: |Δarm| {da:.2e}, отн. raw {yr:.2e}")
    rep["problems"] = prob
    rep["contract"] = check_contract("real-check", a)
    os.makedirs("reports/k24", exist_ok=True)
    json.dump(rep, open("reports/k24/real_check.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"  параметры: {rep['params']}")
    print(f"  legacy vs K-22: |Δarm| {da:.2e}, отн. raw {yr:.2e}; инициализация "
          f"{ {s: rep[f'init_s{s}']['equal'] for s in SEEDS} }")
    print("  " + ("ОТКАЗ: " + "; ".join(prob) if prob else
                  "real-check K-24 пройден") + "; reports/k24/real_check.json")
    return 3 if prob else 0


def selftest():
    import copy
    import tempfile
    import types
    import torch
    import k21_data as d21
    import k21_model as km
    # непрерывное обучение = остановка + возобновление (настоящий Trainer)
    bar0 = km.fake_bar(torch)
    saved = {k: getattr(kt, k) for k in (
        "WARM_UPDATES", "MAIN_UPDATES", "EVAL_EVERY", "SAVE_EVERY",
        "PANEL_N", "EFF_BATCH", "OUTDIR", "make_model", "run_contract",
        "schedule")}
    saved_mb = d21.make_batch
    nr = 64
    L, U = -np.ones(6) * 1.1, np.ones(6) * 4.0
    DT = dict(train_rows=np.arange(48), val_rows=np.arange(48, 56),
              teacher_rows=np.arange(48, 56), task=np.arange(nr) % 4,
              episode=np.arange(nr) % 8, L=L, U=U, state_mean=None,
              state_std=None, prep_report=dict(bar_pin=None))

    def fake_mb(D_, rows, bar, dev, teacher_idx=None):
        rows = np.asarray(rows)
        b = km.fake_batch(torch, B=len(rows), seed=int(rows.sum()) % 1000)
        g = torch.Generator().manual_seed(int(rows.sum()) + 7)
        st_ = torch.randn(len(rows), 11, generator=g)
        tgt = torch.rand(len(rows), 20, 7, generator=g) * 2
        tgt[..., 6] = torch.where(tgt[..., 6] > 1, 1.0, -1.0)
        return b, 1, st_, tgt, torch.ones(len(rows), 20, dtype=torch.bool)
    M = k24m.build(torch)

    def fake_make_model(torch_, mode, D_, dev, pin, checkpoint=True):
        m = copy.deepcopy(bar0)
        return M["K24"](m, L, U, seed=0), dict(model=m)
    try:
        kt.WARM_UPDATES, kt.MAIN_UPDATES = 2, 4
        kt.EVAL_EVERY, kt.SAVE_EVERY, kt.PANEL_N, kt.EFF_BATCH = 2, 1, 8, 8
        kt.make_model = fake_make_model
        kt.run_contract = lambda mode, a_, D_: dict(mode=mode, seed=0)
        kt.schedule = k23t.seeded_schedule(saved["schedule"], 0)
        d21.make_batch = fake_mb
        a_ = types.SimpleNamespace(device="cpu", dtype="float32", micro=4,
                                   eval_micro=4)
        res = {}
        with tempfile.TemporaryDirectory() as td:
            for name, stops in (("cont", (4,)), ("resume", (2, 4))):
                kt.OUTDIR = os.path.join(td, name)
                for until in stops:
                    tr = kt.Trainer(torch, a_, DT, MODE, MODE)
                    tr.run(until)
                res[name] = (kt.state_hash(kt.trainable_state(tr.mod)),
                             [(c["update"], round(c["val"]["L"][0], 10))
                              for c in tr.st["curve"]])
        assert res["cont"] == res["resume"], res
        # диагностика уровней на игрушечной модели
        m = M["K24"](copy.deepcopy(bar0), L, U, seed=0)
        DT2 = dict(DT)
        dg = mr_diag(torch, m, dict(model=m), DT2, np.arange(48, 56),
                     torch.device("cpu"), torch.float32, 4)
        assert abs(dg["share_G0"] + dg["share_G1"] + dg["share_G2"] - 1) \
            < 1e-3, dg
        assert dg["identity_P0Z"] < kr.ID_TOL_CPU32
    finally:
        for k, v in saved.items():
            setattr(kt, k, v)
        d21.make_batch = saved_mb
    want = dict(kind="real-check", micros=[8, 32], prep_id="p",
                code={"a.py": "1"}, device="V100")
    for bad in (dict(want, micros=[8]), dict(want, prep_id="q"),
                dict(want, code={"a.py": "2"}), dict(want, device="A100")):
        assert sorted(k for k in set(want) | set(bad)
                      if want.get(k) != bad.get(k)), bad
    try:
        install(2, 8000, fastload=False)
        raise AssertionError("seed 2 принят")
    except SystemExit:
        pass
    finally:
        for k, v in saved.items():
            setattr(kt, k, v)
    print("самопроверка k24_train пройдена: непрерывное = остановка + "
          "возобновление (Trainer, K24), диагностика уровней (доли энергии "
          "в сумме 1, тождества), отказ для seed вне {0, 1}")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-24: обучение и проверки")
    ap.add_argument("cmd", choices=("real-check", "fitcheck", "train",
                                    "check-fresh"))
    ap.add_argument("--kind", choices=("real-check", "fitcheck"),
                    default="real-check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--eval-micro", type=int, default=8)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--micros", type=int, nargs="+", default=[8, 32])
    a = ap.parse_args()
    try:
        if a.cmd == "real-check":
            return real_check(a)
        if a.cmd == "fitcheck":
            return fitcheck(a)
        if a.cmd == "check-fresh":
            return check_fresh(a)
        return train(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
