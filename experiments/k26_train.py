#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-26: обучение parents_attn24 и flat_attn24 (seed 0 и 1), real-check,
fitcheck, решение о microbatch.

Тот же k21_train (файлы K-21–K-25 не меняются). Подмены в процессе:
  - построитель -> k26_model (режим по имени, seed=S); очередь —
    k23_train.seeded_schedule(S) ДО fastload; train-панель — K-21;
  - каталоги data/k26/runs/s<S>/<режим>, отчёты reports/k26/;
  - контракт запуска: seed, архитектура, конфигурация router, число новых
    параметров, бюджет 1000 + 8000 без добора, отпечаток очереди, запись
    сида, SHA реальных зависимостей (не всех файлов экспериментов);
  - на каждой проверке — 64 фиксированные строки val_sel: по шести точкам
    энтропия (по уровням, нормирована на log N_b), KL ребёнка к prior,
    масса на padding (обязана быть 0), RMS поправки к RMS ha; N токенов,
    норма Wo; счётчики GradScaler (шаги, пропуски; попытки — u).
Порядок обучений задаёт k26_run.sh: flat_s0, parents_s0, flat_s1, parents_s1.
Инициализация: тензоры, общие с базовой моделью, сверяются с историческим
init того же сида (контроль из записи подготовки K-26); router — отдельно.
Режимы: real-check --mode M (отдельный процесс на режим), micro-decision,
fitcheck (parents_attn24), train --seed S --mode M, check-fresh.
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
import k26_attention as ka  # noqa: E402
import k26_model as k26m  # noqa: E402

SEEDS = (0, 1)
MODES = k26m.MODES
ORDER = ((0, "flat_attn24"), (0, "parents_attn24"), (1, "flat_attn24"),
         (1, "parents_attn24"))
PREPARE = "reports/k26/prepare.json"
MICRO_FILE = "reports/k26/micro.json"
BUDGET = 8000
CODE = ("k26_attention.py", "k26_model.py", "k26_train.py", "k22_model.py",
        "k21_fastload.py", "k23_train.py", "k23_smoke.py")
DEPS = CODE + kt.CODE_DEPS
DIAG_N, DIAG_SEED = 64, 20261027
HIST_INIT = {("parents_attn24", 0): "data/k22/runs/parents24/resume.pt",
             ("parents_attn24", 1): "data/k23/runs/s1/parents24/resume.pt",
             ("flat_attn24", 0): "data/k21/runs/flat24/resume.pt",
             ("flat_attn24", 1): "data/k23/runs/s1/flat24/resume.pt"}
LEGACY_CKPT = {"parents_attn24": "data/k22/runs/parents24/best.pt",
               "flat_attn24": "data/k21/runs/flat24/best.pt"}
CONTROL = dict(parents_attn24="control_parents", flat_attn24="control_flat")
FIT_MODE = "parents_attn24"


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def outdir(seed):
    return f"data/k26/runs/s{seed}"


def report_path(seed, mode):
    return f"reports/k26/train_s{seed}_{mode}.json"


def realcheck_path(mode):
    return f"reports/k26/real_check_{mode}.json"


def seed_entry(seed):
    import k26_protocol as pc
    if not os.path.exists(PREPARE):
        raise SystemExit(f"нет {PREPARE}: сначала k26_protocol.py prepare")
    return pc.seed_entry(seed)


def shared_hash(st):
    return kt.state_hash(k26m.shared_keys(st))


def router_hash(st):
    return kt.state_hash({k: v for k, v in st.items()
                          if k.startswith("router.")})


def check_contract(kind, a):
    """Контракт проверки: при любом расхождении готовый отчёт не
    переиспользуется (реальные зависимости, данные, BAR, точность,
    устройство, microbatch, режим; для fitcheck — конфигурация подгонки)."""
    import torch
    import k21_data as d21
    import k21_protocol as pp
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    tj = json.load(open(d21.TEACHER_REPORT, encoding="utf-8"))
    c = dict(kind=kind,
             code={f: sha(os.path.join(HERE, f)) for f in DEPS},
             prep_id=prep["prep_id"], prepare_sha1=pp.full_sha(d21.REPORT),
             teacher_sha1=tj["out"]["sha1"],
             teacher_report_sha1=pp.full_sha(d21.TEACHER_REPORT),
             bar_revision=prep["bar_pin"]["revision"],
             bar_files=hashlib.sha1(json.dumps(
                 prep["bar_pin"]["files"], sort_keys=True).encode())
             .hexdigest(),
             precision=f"{a.dtype} autocast" + (
                 " + GradScaler" if a.dtype == "float16" else ""),
             device=torch.cuda.get_device_name(torch.device(a.device)),
             router=ka.CONFIG)
    if kind == "real-check":
        c.update(mode=a.mode, micros=sorted(a.micros), seed=a.seed)
    else:
        c.update(fit=dict(mode=FIT_MODE, n=kt.FIT_N, updates=kt.FIT_UPDATES,
                          lr_mult=3, ratio=kt.FIT_RATIO, grip=kt.FIT_GRIP,
                          rows_seed=kt.SEED + 9, micro=a.micro,
                          eval_micro=a.eval_micro, seed=0))
    return c


def check_fresh(a):
    p = realcheck_path(a.mode) if a.kind == "real-check" else \
        "reports/k26/fitcheck.json"
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


def make_model_fn(seed, fit=False):
    def make_model(torch, mode, D, dev, pin, checkpoint=True):
        import k21_data as d21
        if fit and mode == "tree_adapt24":       # kt.fitcheck строит так
            mode = FIT_MODE
        if mode not in MODES:
            raise SystemExit(f"K-26 не обучает режим {mode!r}")
        bar = d21.load_bar(dev, pin)
        m = bar["model"]
        m.action_expert.float()
        M = k26m.build(torch)
        mod = M["make"](mode, m, D["L"], D["U"], D["state_mean"],
                        D["state_std"], checkpoint=checkpoint,
                        seed=seed).to(dev)
        return mod, bar
    return make_model


def diag_rows(D):
    vr = np.asarray(D["val_rows"])
    return np.sort(np.random.default_rng(DIAG_SEED).choice(
        vr, min(DIAG_N, len(vr)), replace=False))


def attn_diag(torch, mod, bar, D, rows, dev, dt, micro):
    """Агрегаты router на фиксированных строках (без градиента)."""
    import k21_model as km
    acc, ntok, fin = {}, [], True
    was = mod.training
    mod.eval()
    mod.collect = True
    try:
        with torch.no_grad():
            for s0 in range(0, len(rows), micro):
                out = kt.run_micro(torch, km, mod, bar, D,
                                   rows[s0:s0 + micro], "expert", dev, dt)[0]
                for i, inf in enumerate(out["router"]):
                    f = ka.info_floats(inf)
                    fin = fin and f["finite"]
                    for lv, x in enumerate(f["levels"]):
                        acc.setdefault((i, f"H{lv}"), []).append(x["entropy"])
                        if x["kl_to_prior"] is not None:
                            acc.setdefault((i, f"KL{lv}"), []).append(
                                x["kl_to_prior"])
                    for k in ("pad_mass", "sum_err", "rms_ratio"):
                        acc.setdefault((i, k), []).append(f[k])
                ntok += out["bank"]["counts"].tolist()
    finally:
        mod.collect = False
        if was:
            mod.train()
    pts = []
    for i in range(len(ka.AFTER)):
        d = {k: (max(v) if k in ("pad_mass", "sum_err") else
                 float(np.mean(v))) for (j, k), v in acc.items() if j == i}
        pts.append(dict(after_layer=ka.AFTER[i] + 1, **d))
    return dict(points=pts, pad_mass_max=max(p["pad_mass"] for p in pts),
                sum_err_max=max(p["sum_err"] for p in pts), finite=fin,
                n_tokens=[int(min(ntok)), int(max(ntok))],
                wo_norm=float(mod.router.Wo.weight.detach().float().norm()),
                rows=len(rows))


def diag_line(d):
    last = d["points"][-1]
    hs = " ".join(f"H{lv} {last[f'H{lv}']:.3f}" for lv in range(3)
                  if f"H{lv}" in last)
    kl = " ".join(f"KL{lv} {last[f'KL{lv}']:.3f}" for lv in (1, 2)
                  if f"KL{lv}" in last)
    rms = "/".join(f"{p['rms_ratio']:.3f}" for p in d["points"])
    return (f"после слоя 24: {hs} {kl}; RMS поправки {rms}; padding "
            f"{d['pad_mass_max']:.1e}; |Wo| {d['wo_norm']:.3f}; N "
            f"{d['n_tokens']}")


def install(seed, budget, torch=None, fastload=True, fit=False,
            entry_fp=None):
    if seed not in SEEDS:
        raise SystemExit(f"training seed {seed}: в K-26 только {SEEDS}")
    sched = k23t.seeded_schedule(kt.schedule, seed)
    kt.schedule = sched
    kt.make_model = make_model_fn(seed, fit=fit)
    kt.OUTDIR = "data/k26/fitcheck" if fit else outdir(seed)
    redirect = {"reports/k21/fitcheck.json": "reports/k26/fitcheck.json",
                "reports/k21/train.json": "reports/k26/unused_train.json"}

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
        if fit and mode == "tree_adapt24":
            mode = FIT_MODE
        if mode not in MODES:
            raise SystemExit(f"K-26 не обучает режим {mode!r}")
        c = orig_contract(mode, a, D)
        if a.micro not in fp_cache:
            fp_cache[a.micro] = k23t.schedule_fp(sched, D, a.micro)
        c.update(seed=seed, architecture=k26m.ARCH[mode], router=ka.CONFIG,
                 budget=dict(warmup=kt.WARM_UPDATES, main=budget,
                             main_max_scheduler=kt.MAIN_MAX, extension=False),
                 k26=dict(train_seed=seed, schedule_fp=fp_cache[a.micro],
                          seed_entry_fp=entry_fp, base=k26m.BASE[mode],
                          panel="K-21 (k21_train.SEED + 5)",
                          eff_batch=kt.EFF_BATCH,
                          accumulation=kt.EFF_BATCH // a.micro))
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
        c["attempted"] = int(u)
        rec = self.st["curve"][-1]
        rec["scaler"] = c
        t0 = time.time()
        rec["attn_diag"] = attn_diag(torch, self.mod, self.bar, self.D,
                                     diag_rows(self.D), self.dev, self.dt,
                                     self.a.eval_micro)
        d = rec["attn_diag"]
        if d["pad_mass_max"] != 0.0 or not d["finite"] or \
                d["sum_err_max"] > ka.SUM_TOL:
            raise SystemExit(f"{self.tag}: router — padding "
                             f"{d['pad_mass_max']}, суммы {d['sum_err_max']}"
                             f", конечность {d['finite']}")
        print(f"    {self.tag}: шагов {c.get('steps')}, пропусков GradScaler "
              f"{c.get('skips')}; {diag_line(d)} ({time.time() - t0:.0f} с)",
              flush=True)
    kt.Trainer.check = check
    if torch is not None:
        torch.cuda.amp.GradScaler = k23t.counting_scaler(
            torch.cuda.amp.GradScaler)
    st = None
    import k21_data as d21
    orig_make = d21.make_batch
    if fastload:
        import k21_fastload as fl
        st = fl.install()
    return dict(fastload=st, schedule=sched, orig_make=orig_make)


def fastload_check(torch, D, bar, dev, sched, micro, orig_make, n_scan=8):
    """Явная сверка ускоренной сборки (k21_fastload: cpu_part + to_device)
    с прежней k21_data.make_batch на выбранном micro: первый экспертный и
    первый учительский microbatch очереди этого сида (батч, pos_offset,
    состояние, цели, маска; у учителя — teacher_idx и его строки).
    k21_fastload не меняется."""
    import copy
    import k21_fastload as fl
    reader = fl.RowReader(D["prep_report"]["inputs"]["images"]["path"]) \
        if "prep_report" in D and "inputs" in D["prep_report"] else None
    pbar = dict(cfg=bar.get("cfg"), proc=copy.deepcopy(bar.get("proc")))
    res = {}
    for u in range(n_scan):
        src, idx = sched(D, u + 1, micro, start=u)[0]
        if src in res:
            continue
        r = np.asarray(idx[0])
        ti = r if src == "teacher" else None
        rows = np.asarray(D["teacher_rows"])[r] if ti is not None else r
        fast = fl.to_device(D, fl.cpu_part(D, rows, pbar, reader, ti), rows,
                            bar, dev, ti)
        ref = orig_make(D, rows, bar, dev, teacher_idx=ti)
        ok = bool(fl.same_batch(fast, ref))
        if ti is not None:
            want = torch.as_tensor(D["teacher_target"][ti]).to(fast[3].device)
            ok = ok and bool(torch.equal(fast[3], want)) and bool(
                (fast[4] == 1).all())
        res[src] = dict(update=u, rows=len(rows), equal=ok,
                        teacher_idx=None if ti is None else
                        [int(x) for x in ti[:4]])
        if len(res) == 2:
            break
    if set(res) != {"expert", "teacher"} or not all(v["equal"] for v in
                                                    res.values()):
        raise SystemExit(f"fastload: сверка экспертного и учительского "
                         f"батча не пройдена: {res}")
    print(f"  fastload (micro {micro}): экспертный (u {res['expert']['update']})"
          f" и учительский (u {res['teacher']['update']}) батчи совпали с "
          f"прежним путём побитово", flush=True)
    return res


def hist_init_hash(torch, mode, seed):
    p = HIST_INIT[(mode, seed)]
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location="cpu", weights_only=False)["init_hash"]


def decided_micro():
    if not os.path.exists(MICRO_FILE):
        return None
    return int(json.load(open(MICRO_FILE, encoding="utf-8"))["micro"])


def micro_decision(a):
    """Общий microbatch обучения для обеих моделей — до первого обучения:
    32, если real-check обоих режимов прошёл на 32; 8 × 4, если на 32 OOM,
    а на 8 оба прошли. Однажды записанное решение не меняется."""
    res = {}
    for m in MODES:
        p = realcheck_path(m)
        if not os.path.exists(p):
            raise SystemExit(f"нет {p}")
        r = json.load(open(p, encoding="utf-8"))
        res[m] = r
    oom32 = any(r.get("oom32") for r in res.values())
    bad = {m: r["problems"] for m, r in res.items() if r["problems"]}
    if bad:
        raise SystemExit(f"real-check с проблемами: {bad}")
    ok8 = all((r.get("micro_8") or {}).get("checks") and all(
        r["micro_8"]["checks"].values()) for r in res.values())
    micro = 8 if oom32 else 32
    if oom32 and not ok8:
        raise SystemExit("OOM на 32 и нет пригодного 8")
    old = decided_micro()
    if old is not None and old != micro:
        raise SystemExit(f"решение о microbatch уже записано: {old}, сейчас "
                         f"вышло бы {micro} — без молчаливой смены")
    rec = dict(micro=micro, accumulation=kt.EFF_BATCH // micro,
               eff_batch=kt.EFF_BATCH, reason="micro 32 OOM" if oom32
               else "micro 32 пройден обоими режимами",
               real_check_sha1={m: sha(realcheck_path(m)) for m in MODES},
               peak_gb={m: {k: (r.get(k) or {}).get("peak_gb") for k in
                            ("micro_8", "micro_32")} for m, r in res.items()})
    if old is None:
        os.makedirs("reports/k26", exist_ok=True)
        json.dump(rec, open(MICRO_FILE, "w", encoding="utf-8"), indent=1)
    print(f"  microbatch обучения: {micro} × {kt.EFF_BATCH // micro} "
          f"({rec['reason']}); {MICRO_FILE}")
    return 0


def train(a):
    import torch
    import k21_data as d21
    if a.mode not in MODES:
        raise SystemExit(f"режим {a.mode}")
    mic = decided_micro()
    if mic is None or mic != a.micro:
        raise SystemExit(f"microbatch {a.micro} ≠ решению {MICRO_FILE} "
                         f"({mic})")
    e = seed_entry(a.seed)
    budget = int(e["budget_main"])
    if budget != BUDGET:
        raise SystemExit(f"бюджет записи сида {budget} ≠ {BUDGET}")
    st = install(a.seed, budget, torch, entry_fp=e["fingerprint"])
    D = d21.load_prepared(require_teacher=True)
    have = os.path.exists(os.path.join(outdir(a.seed), a.mode, "resume.pt"))
    print(f"  K-26 {a.mode} seed {a.seed}: бюджет main {budget} (закреплён, "
          f"без добора), micro {a.micro} × {kt.EFF_BATCH // a.micro}; "
          + ("продолжение" if have else "новый запуск"), flush=True)
    tr = kt.Trainer(torch, a, D, a.mode, a.mode)
    flc = fastload_check(torch, D, tr.bar, tr.dev, st["schedule"], a.micro,
                         st["orig_make"])
    s0 = kt.trainable_state(tr.mod)
    want = e[CONTROL[a.mode]]["init_hash"]
    sh, rh = shared_hash(s0), router_hash(s0)
    del s0
    if sh != want:
        raise SystemExit(f"общие с {k26m.BASE[a.mode]} начальные веса {sh} ≠ "
                         f"контролю {want}")
    other = report_path(a.seed, [m for m in MODES if m != a.mode][0])
    if os.path.exists(other):
        oh = json.load(open(other, encoding="utf-8")).get("router_init_hash")
        if oh != rh:
            raise SystemExit(f"начальный router ≠ router пары ({other})")
    tr.run(budget)
    rs = torch.load(os.path.join(outdir(a.seed), a.mode, "resume.pt"),
                    map_location="cpu", weights_only=False)
    b = torch.load(os.path.join(outdir(a.seed), a.mode, "best.pt"),
                   map_location="cpu", weights_only=False)
    if rs["u"] != budget or rs["phase"] != "main":
        raise SystemExit(f"обучение не дошло до бюджета: {rs['phase']} "
                         f"{rs['u']}/{budget}")
    rep = dict(seed=a.seed, mode=a.mode, architecture=k26m.ARCH[a.mode],
               budget_main=budget, updates=rs["u"], best=rs["best"],
               best_sha1=kt.sha(os.path.join(outdir(a.seed), a.mode,
                                             "best.pt")),
               best_weights_hash=b["weights_hash"], best_update=b["update"],
               init_hash=rs["init_hash"], init_hash_shared=sh,
               router_init_hash=rh, control_init_hash=want,
               schedule_fp=rs["config"]["k26"]["schedule_fp"],
               fastload_check=flc,
               scaler=(rs.get("scaler") or {}).get("_k23"),
               curve=rs["curve"], config=rs["config"])
    os.makedirs("reports/k26", exist_ok=True)
    json.dump(rep, open(report_path(a.seed, a.mode), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    if os.path.exists(other):
        o = json.load(open(other, encoding="utf-8"))
        if o.get("schedule_fp") != rep["schedule_fp"]:
            raise SystemExit("очередь пары различается")
    pf = st["fastload"]["pf"] if st["fastload"] else None
    if pf is not None:
        print(f"  fastload: из предзагрузки {pf.hits}, синхронно {pf.misses}")
    print(f"  K-26 {a.mode} seed {a.seed}: лучшая {rs['best']}, отчёт "
          f"{report_path(a.seed, a.mode)}", flush=True)
    return 0


MAPS_N = 4


def maps(a):
    """До четырёх заранее выбранных наблюдений (первые строки фиксированной
    диагностической панели val_sel, не по удачному поведению): карты P после
    слоя 24 по позициям image-токенов в порядке inputs_merger. Разметка
    tiles на изображение не проверялась — это индексные карты, не наложение
    на кадр."""
    import torch
    import k21_data as d21
    import k21_model as km
    rp = report_path(a.seed, a.mode)
    tr = json.load(open(rp, encoding="utf-8"))
    ck = os.path.join(outdir(a.seed), a.mode, "best.pt")
    if kt.sha(ck) != tr["best_sha1"]:
        raise SystemExit(f"{ck} ≠ отчёту {rp}")
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    mod, bar = make_model_fn(a.seed)(torch, a.mode, D, dev,
                                     D["prep_report"]["bar_pin"],
                                     checkpoint=False)
    c = torch.load(ck, map_location="cpu", weights_only=False)
    own = dict(mod.named_parameters())
    own.update(dict(mod.named_buffers()))
    with torch.no_grad():
        for k, v in c["weights"].items():
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    if kt.state_hash(kt.trainable_state(mod)) != tr["best_weights_hash"]:
        raise SystemExit("веса после загрузки ≠ отчёту обучения")
    rows = diag_rows(D)[:MAPS_N]
    mod.eval()
    mod.keep_maps = True
    t0 = time.time()
    with torch.no_grad():
        out = kt.run_micro(torch, km, mod, bar, D, rows, "expert", dev,
                           torch.float16)[0]
    mp = out["router"][-1]["maps"]
    P = mp["P"].numpy().astype(np.float16)
    path = f"reports/k26/maps_s{a.seed}_{a.mode}.npz"
    np.savez_compressed(path, P=P, valid=mp["valid"].numpy(),
                        rows=np.asarray(rows),
                        counts=out["bank"]["counts"].cpu().numpy())
    meta = dict(seed=a.seed, mode=a.mode, rows=[int(r) for r in rows],
                tasks=[str(D["task"][r]) for r in rows],
                shape_P=list(P.shape), after_layer=24,
                nodes="parents: 0:4 корни, 4:12 средние, 12:28 листья; flat: "
                      "16 query" if a.mode == "parents_attn24" else "16 query",
                note="индексные карты по позициям image-токенов (порядок "
                     "inputs_merger); разметка tiles на изображение не "
                     "проверена — не рисовать как прямоугольную сетку; карта "
                     "не доказывает причинную пользу для SR",
                best_sha1=tr["best_sha1"], seconds=time.time() - t0)
    json.dump(meta, open(path.replace(".npz", ".json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  карты: {path} ({os.path.getsize(path) / 2 ** 20:.2f} МБ, P "
          f"{list(P.shape)})", flush=True)
    return 0


def fitcheck(a):
    import torch
    import k21_data as d21
    e0 = seed_entry(0)
    install(0, e0["budget_main"], torch, fastload=False, fit=True,
            entry_fp=e0["fingerprint"])
    rec = []
    orig_eval = kt.evaluate

    def evaluate(torch_, km_, mod, *args, **kw):
        if hasattr(mod, "router"):
            r = mod.router
            rec.append(dict(
                wo_norm=float(r.Wo.weight.detach().float().norm()),
                grad={n: (float(getattr(r, n).weight.grad.float().norm())
                          if getattr(r, n).weight.grad is not None else None)
                      for n in ("Wq", "Wk", "Wv", "Wo")}))
        return orig_eval(torch_, km_, mod, *args, **kw)
    kt.evaluate = evaluate
    D = d21.load_prepared(require_teacher=True)
    try:
        code = kt.fitcheck(torch, a, D)
    finally:
        kt.evaluate = orig_eval
    fin = rec[-1] if rec else {}
    gq = fin.get("grad") or {}
    # критерии подгонки — прежние (L1, схват); нормы router только
    # выводятся; нечисловые значения — техническая ошибка
    finite = all(np.isfinite(r["wo_norm"]) and all(
        v is None or np.isfinite(v) for v in r["grad"].values())
        for r in rec)
    print(f"  fitcheck router (описательно): |Wo| {fin.get('wo_norm')}, "
          f"градиенты "
          f"{ {k: (None if v is None else round(v, 6)) for k, v in gq.items()} }"
          + ("" if finite else " — НЕЧИСЛОВЫЕ значения"), flush=True)
    p = "reports/k26/fitcheck.json"
    if os.path.exists(p):
        r = json.load(open(p, encoding="utf-8"))
        r["router_trace"] = rec
        r["router_finite"] = finite
        r["mode"] = FIT_MODE
        r["ok"] = bool(r.get("ok")) and finite
        code = code if finite else 3
        r["contract"] = check_contract("fitcheck", a)
        r["exit"] = code
        json.dump(r, open(p, "w", encoding="utf-8"), indent=1)
    return code


def real_check(a):
    import torch
    import k21_data as d21
    import k21_model as km
    import k23_smoke as sm
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=True)
    pin = D["prep_report"]["bar_pin"]
    M = k26m.build(torch)
    mode = a.mode
    prob, rep = [], dict(mode=mode, seed=a.seed, micros=a.micros)
    if 32 not in a.micros or 8 not in a.micros:
        prob.append("проверяются обязательные micro 8 и 32")
    if a.dtype != "float16":
        prob.append(f"real-check для float16 + GradScaler, запрошен {a.dtype}")
    rows = D["train_rows"][:: max(1, len(D["train_rows"]) // 64)][:64]

    def fresh(seed, base=False, **kw):
        bar = d21.load_bar(dev, pin)
        m = bar["model"]
        m.action_expert.float()
        f = M["make_base"] if base else M["make"]
        return f(mode, m, D["L"], D["U"], D["state_mean"], D["state_std"],
                 checkpoint=True, seed=seed, **kw).to(dev), bar
    # 1. начальные веса: общие = историческому init, router по сидам
    for s in SEEDS:
        m_, _b = fresh(s)
        st = kt.trainable_state(m_)
        h_sh, h_r = shared_hash(st), router_hash(st)
        del m_, _b, st
        torch.cuda.empty_cache()
        hh = hist_init_hash(torch, mode, s)
        rep[f"init_s{s}"] = dict(shared=h_sh, router=h_r, historical=hh,
                                 equal=(hh == h_sh) if hh else None)
        if hh is None or hh != h_sh:
            prob.append(f"seed {s}: общие начальные веса ≠ историческим "
                        f"({hh})")
    # 2. визуальный банк на настоящем входе: позиции = выход inputs_merger
    mod, bar = fresh(a.seed)
    model = mod.bar
    batch, po, st, tgt, val = d21.make_batch(D, rows[:8], bar, dev)
    with torch.no_grad():
        img = model.vlm.get_image_features(batch["pixel_values"],
                                           batch.get("pixel_attention_mask"))
        _b, _l, hv0, _i = model._build_vlm_inputs_embeds(
            input_ids=batch["input_ids"], inputs_embeds=None,
            pixel_values=None, pixel_attention_mask=None,
            image_hidden_states=img)
        E, valid, counts = M["R"]["pack_image_bank"](
            hv0, batch["input_ids"], batch["attention_mask"],
            model.image_token_id)
        flat = torch.cat([E[b, :int(counts[b])] for b in range(E.shape[0])])
        ref = img.reshape(-1, img.shape[-1]).to(hv0.dtype)
        img_ids = int((batch["input_ids"] == model.image_token_id).sum())
    vlp = getattr(bar["proc"], "vl_processor", bar["proc"])
    ip = getattr(vlp, "image_processor", None)
    pcfg = dict(image_seq_len=getattr(vlp, "image_seq_len", None),
                image_processor=type(ip).__name__ if ip is not None else None)
    if ip is not None and hasattr(ip, "to_dict"):
        d_ = ip.to_dict()
        pcfg.update({k: d_.get(k) for k in (
            "size", "max_image_size", "do_image_splitting", "do_resize")
            if k in d_})
    rep["bank"] = dict(
        pixel_values=list(batch["pixel_values"].shape),
        image_features=list(img.shape), hv0=list(hv0.shape),
        E=list(E.shape), dv=int(E.shape[-1]), counts=counts.tolist(),
        image_token_id=int(model.image_token_id),
        image_ids_total=img_ids, equal_to_merger=bool(
            flat.shape == ref.shape and torch.equal(flat, ref)),
        processor=pcfg, dtype=str(hv0.dtype))
    if not rep["bank"]["equal_to_merger"]:
        prob.append("банк ≠ признакам inputs_merger (позиции/порядок)")
    if img_ids != int(counts.sum()):
        prob.append(f"image-токенов {img_ids}, в банке {int(counts.sum())} "
                    f"(часть в padding?)")
    if E.shape[-1] != mod.router.Dv:
        prob.append(f"Dv банка {E.shape[-1]} ≠ router {mod.router.Dv}")
    print(f"  банк: {json.dumps(rep['bank'], ensure_ascii=False)}",
          flush=True)
    del img, hv0, E, valid, flat, ref
    # 3. шаги на micro 8 и 32: полный forward/backward/AdamW (LR обучения)
    named = [(n, p) for n, p in mod.named_parameters() if p.requires_grad]
    gn = sm.groups_named(mod)
    gr = mod.groups()
    opt = torch.optim.AdamW([dict(params=gr[k], lr=kt.LR[k]) for k in
                             ("expert", "lora", "new")],
                            weight_decay=kt.WD)
    scaler = torch.cuda.amp.GradScaler()
    last = len(mod.bar.vlm.text_model.layers) - 1
    Lt = torch.as_tensor(D["L"], device=dev)
    Ut = torch.as_tensor(D["U"], device=dev)
    optc = dict(applied=0, skipped=0)

    def step(mb):
        batch, po, st, tgt, val = d21.make_batch(D, rows[:mb], bar, dev)
        mod.train()
        mod.collect = True
        with torch.autocast("cuda", dtype=torch.float16):
            out = mod(batch, po, st)
            loss, _l1, _b = km.loss_fn(torch, out, tgt, val)
        mod.collect = False
        scaler.scale(loss.mean()).backward()
        gp, none, norms = sm.grad_problems(named, gn, last)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for _n, p in named], kt.CLIP)
        before = float(scaler.get_scale())
        scaler.step(opt)
        scaler.update()
        applied = float(scaler.get_scale()) >= before   # пропуск = scale ↓
        optc["applied" if applied else "skipped"] += 1
        opt.zero_grad(set_to_none=True)
        return out, loss, gp, none, norms, applied
    for mb in sorted(a.micros):
        r = {}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.time()
            out, loss, gp, none, norms, applied = step(mb)
            torch.cuda.synchronize(dev)
            arm = out["arm"].float()
            infos = [ka.info_floats(x) for x in out["router"]]
            pf = all(bool(torch.isfinite(p).all()) for _n, p in named)
            checks = dict(
                counters=not k26m.counters_problems(mode, mod.counters),
                finite=bool(torch.isfinite(arm).all()
                            and torch.isfinite(loss).all()),
                bounds=bool(((arm >= Lt - 1e-4) & (arm <= Ut + 1e-4)).all()),
                shapes=tuple(arm.shape) == (mb, 20, 6),
                grads=not gp, params_finite=pf,
                router=all(x["finite"] and x["pad_mass"] == 0.0 and
                           x["sum_err"] <= ka.SUM_TOL for x in infos))
            r = dict(checks=checks, seconds=time.time() - t0,
                     peak_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                     loss=float(loss.mean()), group_grad_norms=norms,
                     no_grad_params=none, grad_problems=gp,
                     bank=dict(n_max=out["bank"]["n_max"],
                               dv=out["bank"]["dv"]),
                     router=[dict(rms_ratio=x["rms_ratio"],
                                  levels=x["levels"]) for x in infos],
                     scaler_scale=float(scaler.get_scale()),
                     optimizer_step_applied=applied)
            bad = [k for k, v in checks.items() if not v]
            if bad:
                prob.append(f"micro {mb}: {bad}")
            del out, loss
        except torch.cuda.OutOfMemoryError:
            r = dict(oom=True)
            c8 = (rep.get("micro_8") or {}).get("checks") or {}
            if mb == 32 and c8 and all(c8.values()):
                rep["oom32"] = True       # решение 8 × 4 — micro-decision
            else:
                prob.append(f"micro {mb}: OOM")
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        rep[f"micro_{mb}"] = r
        print(f"  micro {mb}: {json.dumps({k: r.get(k) for k in ('checks', 'peak_gb', 'seconds', 'oom')}, default=float)}",
              flush=True)
    # 4. путь градиента от настоящей инициализации: после шага, изменившего
    # Wo (пропуски GradScaler повторяются), Q/K/V получают градиент
    tries = 0
    while float(mod.router.Wo.weight.detach().abs().sum()) == 0 and tries < 8:
        step(8)
        tries += 1
    batch, po, st, tgt, val = d21.make_batch(D, rows[:8], bar, dev)
    mod.train()
    with torch.autocast("cuda", dtype=torch.float16):
        out = mod(batch, po, st)
        loss, _l1, _b = km.loss_fn(torch, out, tgt, val)
    # как в обучении: backward через масштаб GradScaler (без него малые
    # fp16-градиенты через Wo ≈ 1e-4 и γ уходят в ноль), норма / масштаб
    sc = float(scaler.get_scale())
    scaler.scale(loss.mean()).backward()
    rgr = {n: float(getattr(mod.router, n).weight.grad.float().norm()) / sc
           for n in ("Wq", "Wk", "Wv", "Wo")}
    rep["optimizer_steps"] = dict(optc, note="пропущенный GradScaler шаг "
                                  "не считается выполненным AdamW")
    rep["router_grad_after_step"] = dict(
        grads=rgr, extra_steps=tries, grad_scale=sc,
        wo_norm=float(mod.router.Wo.weight.detach().float().norm()))
    if float(mod.router.Wo.weight.detach().abs().sum()) == 0:
        prob.append(f"Wo не изменился за {tries} доп. шагов (пропуски "
                    f"GradScaler: {optc['skipped']})")
    if not all(np.isfinite(v) and v > 0 for v in rgr.values()):
        prob.append(f"градиент router после шага: {rgr}")
    print(f"  router после шага: {rep['router_grad_after_step']}", flush=True)
    H = int(mod.H)
    mod = bar = opt = out = loss = None      # замыкание step() держит ячейки
    torch.cuda.empty_cache()
    # 5. legacy: router выключен (и нулевой Wo) на обученных общих весах
    # = исторический класс того же режима
    ck = torch.load(LEGACY_CKPT[mode], map_location="cpu", weights_only=False)
    outs = {}
    for name, base, kw in (("base", True, {}), ("off", False,
                                                dict(router="off")),
                           ("on_zero_wo", False, {})):
        m_, b_ = fresh(0, base=base, **kw)
        rep.setdefault("params", {})[name] = dict(
            total=sum(p.numel() for p in m_.parameters()),
            trainable=sum(p.numel() for g in m_.groups().values() for p in g))
        own = dict(m_.named_parameters())
        own.update(dict(m_.named_buffers()))
        with torch.no_grad():
            for k, v in ck["weights"].items():
                own[k].copy_(v.to(own[k].device, own[k].dtype))
        m_.eval()
        batch, po, st, _t, _v = d21.make_batch(D, rows[:8], b_, dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            o = m_(batch, po, st)
        outs[name] = {k: o[k].float().cpu() for k in ("arm", "grip_logit")}
        del m_, b_
        torch.cuda.empty_cache()
    leg = {}
    for name in ("off", "on_zero_wo"):
        leg[name] = max(float((outs["base"][k] - outs[name][k]).abs().max())
                        for k in ("arm", "grip_logit"))
        if not leg[name] <= 1e-6:
            prob.append(f"{name} ≠ {k26m.BASE[mode]}: {leg[name]:.2e}")
    rep["legacy_vs_base"] = leg
    pb, pn = rep["params"]["base"], rep["params"]["on_zero_wo"]
    n_new = pn["total"] - pb["total"]
    want_new = ka.n_new_params(H, rep["bank"]["dv"])
    rep["params"]["new"] = dict(total=n_new, expected=want_new,
                                share_of_trainable=n_new / pb["trainable"])
    if n_new != pn["trainable"] - pb["trainable"] or n_new != want_new or \
            n_new / pb["trainable"] >= 0.01:
        prob.append(f"новые параметры: {rep['params']['new']}")
    rep["problems"] = prob
    rep["contract"] = check_contract("real-check", a)
    os.makedirs("reports/k26", exist_ok=True)
    json.dump(rep, open(realcheck_path(mode), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"  параметры: {rep['params']}")
    print(f"  legacy (router off / нулевой Wo) vs {k26m.BASE[mode]}: "
          f"{ {k: f'{v:.2e}' for k, v in leg.items()} }; инициализация "
          f"{ {s: rep[f'init_s{s}']['equal'] for s in SEEDS} }; router "
          f"{ {s: rep[f'init_s{s}']['router'][:12] for s in SEEDS} }")
    print("  " + ("ОТКАЗ: " + "; ".join(prob) if prob else
                  f"real-check K-26 {mode} пройден") +
          f"; {realcheck_path(mode)}")
    return 3 if prob else 0


def selftest():
    import copy
    import tempfile
    import types
    import torch
    import k21_data as d21
    bar0 = k26m.fake_bar(torch)
    saved = {k: getattr(kt, k) for k in (
        "WARM_UPDATES", "MAIN_UPDATES", "EVAL_EVERY", "SAVE_EVERY",
        "PANEL_N", "EFF_BATCH", "OUTDIR", "make_model", "run_contract",
        "schedule")}
    saved_mb = d21.make_batch
    saved_check = kt.Trainer.check
    nr = 64
    L, U = -np.ones(6) * 1.1, np.ones(6) * 4.0
    DT = dict(train_rows=np.arange(48), val_rows=np.arange(48, 56),
              teacher_rows=np.arange(48, 56), task=np.arange(nr) % 4,
              episode=np.arange(nr) % 8, L=L, U=U, state_mean=None,
              state_std=None, prep_report=dict(bar_pin=None))

    def fake_mb(D_, rows, bar, dev, teacher_idx=None):
        rows = np.asarray(rows)
        b = k26m.fake_batch(torch, B=len(rows), seed=int(rows.sum()) % 1000)
        g = torch.Generator().manual_seed(int(rows.sum()) + 7)
        st_ = torch.randn(len(rows), 11, generator=g)
        tgt = torch.rand(len(rows), 20, 7, generator=g) * 2
        tgt[..., 6] = torch.where(tgt[..., 6] > 1, 1.0, -1.0)
        return b, 1, st_, tgt, torch.ones(len(rows), 20, dtype=torch.bool)
    M = k26m.build(torch)
    try:
        d21.make_batch = fake_mb
        a_ = types.SimpleNamespace(device="cpu", dtype="float32", micro=4,
                                   eval_micro=4)
        for mode in MODES:
            def fake_make_model(torch_, mode_, D_, dev, pin, checkpoint=True,
                                _m=mode):
                assert mode_ == _m, mode_
                m = copy.deepcopy(bar0)
                return M["make"](mode_, m, L, U, seed=1), dict(model=m)
            kt.WARM_UPDATES, kt.MAIN_UPDATES = 2, 4
            kt.EVAL_EVERY, kt.SAVE_EVERY, kt.PANEL_N, kt.EFF_BATCH = 2, 1, 8, 8
            kt.make_model = fake_make_model
            kt.run_contract = lambda mode_, a__, D_: dict(mode=mode_, seed=1)
            kt.schedule = k23t.seeded_schedule(saved["schedule"], 1)
            res = {}
            with tempfile.TemporaryDirectory() as td:
                for name, stops in (("cont", (4,)), ("resume", (2, 4))):
                    kt.OUTDIR = os.path.join(td, name)
                    for until in stops:
                        tr = kt.Trainer(torch, a_, DT, mode, mode)
                        tr.run(until)
                    res[name] = (kt.state_hash(kt.trainable_state(tr.mod)),
                                 [(c["update"], round(c["val"]["L"][0], 10))
                                  for c in tr.st["curve"]])
            assert res["cont"] == res["resume"], (mode, res)
            m = M["make"](mode, copy.deepcopy(bar0), L, U, seed=0)
            dg = attn_diag(torch, m, dict(model=m), DT, np.arange(48, 56),
                           torch.device("cpu"), torch.float32, 4)
            assert dg["pad_mass_max"] == 0.0 and dg["finite"] and \
                len(dg["points"]) == 6 and dg["wo_norm"] == 0.0
            assert all(p["rms_ratio"] == 0.0 for p in dg["points"])
            if mode == "parents_attn24":
                assert "KL2" in dg["points"][-1] and "H2" in dg["points"][-1]
            assert m.collect is False
            _ = diag_line(dg)
    finally:
        for k, v in saved.items():
            setattr(kt, k, v)
        d21.make_batch = saved_mb
        kt.Trainer.check = saved_check
    # настоящая фабрика тренера на игрушечном BAR: нужный класс и режим для
    # обоих режимов и сидов; fitcheck строит parents_attn24
    saved_lb = d21.load_bar
    try:
        d21.load_bar = lambda dev, pin: dict(model=copy.deepcopy(bar0))
        DF = dict(L=L, U=U, state_mean=None, state_std=None)
        for s_ in SEEDS:
            hashes = {}
            for mode in MODES:
                mf, _bf = make_model_fn(s_)(torch, mode, DF,
                                            torch.device("cpu"), None,
                                            checkpoint=False)
                want = dict(parents_attn24="K26Parents",
                            flat_attn24="K26Flat")[mode]
                assert type(mf).__name__ == want and mf.mode == mode and \
                    mf.router_mode == "on", (type(mf), mf.mode)
                stf = kt.trainable_state(mf)
                base = M["make_base"](mode, copy.deepcopy(bar0), L, U,
                                      seed=s_)
                assert shared_hash(stf) == kt.state_hash(
                    kt.trainable_state(base))
                hashes[mode] = router_hash(stf)
                with torch.no_grad():
                    of = mf.eval()(k26m.fake_batch(torch, B=2, pad=1), 1,
                                   torch.randn(2, 11))
                assert "bank" in of and of["arm"].shape == (2, 20, 6)
            assert len(set(hashes.values())) == 1
            mfit, _b = make_model_fn(s_, fit=True)(
                torch, "tree_adapt24", DF, torch.device("cpu"), None,
                checkpoint=False)
            assert mfit.mode == FIT_MODE
            try:
                make_model_fn(s_)(torch, "tree_adapt24", DF,
                                  torch.device("cpu"), None)
                raise AssertionError("tree_adapt24 принят вне fitcheck")
            except SystemExit:
                pass
    finally:
        d21.load_bar = saved_lb
    # явная сверка fastload: экспертный и учительский батч; порча цели
    # учителя или батча — отказ
    import k21_fastload as fl
    saved_fl = (fl.cpu_part, fl.to_device, fl.same_batch)
    DFL = dict(DT, teacher_rows=np.arange(48, 56),
               teacher_target=np.random.default_rng(0).random((8, 20, 7))
               .astype(np.float32))

    def orig_mb(D_, rows, bar, dev, teacher_idx=None):
        b, po, st_, tgt, val = fake_mb(D_, rows, bar, dev)
        po = torch.full((len(rows),), po, dtype=torch.long)
        if teacher_idx is not None:
            tgt = torch.as_tensor(D_["teacher_target"][np.asarray(
                teacher_idx)])
            val = torch.ones(tgt.shape[:2])
        return b, po, st_, tgt, val
    try:
        for corrupt, ok_ in ((None, True), ("teacher", False),
                             ("expert", False)):
            fl.cpu_part = lambda D_, rows, pb, rd, ti=None: (rows, ti)

            def to_dev(D_, b, rows, bar, dev, ti=None, _c=corrupt):
                out = list(orig_mb(D_, rows, bar, dev, teacher_idx=ti))
                if (_c == "teacher" and ti is not None) or \
                        (_c == "expert" and ti is None):
                    out[3] = out[3] + 1
                return tuple(out)
            fl.to_device = to_dev
            sch = k23t.seeded_schedule(saved["schedule"], 0)
            try:
                r_ = fastload_check(torch, DFL, dict(cfg=None, proc=None),
                                    torch.device("cpu"), sch, 4, orig_mb)
                assert ok_ and r_["teacher"]["equal"] and \
                    r_["teacher"]["update"] == 3, r_
            except SystemExit:
                assert not ok_, corrupt
    finally:
        fl.cpu_part, fl.to_device, fl.same_batch = saved_fl
    try:
        install(2, 8000, fastload=False)
        raise AssertionError("seed 2 принят")
    except SystemExit:
        pass
    finally:
        for k, v in saved.items():
            setattr(kt, k, v)
        kt.Trainer.check = saved_check
    assert len(set(DEPS)) == len(DEPS) and "k26_protocol.py" not in DEPS
    assert [m for _s, m in ORDER] == ["flat_attn24", "parents_attn24"] * 2
    print("самопроверка k26_train пройдена: непрерывное = остановка + "
          "возобновление (Trainer, оба режима), явная сверка fastload "
          "(экспертный и учительский батч; порча — отказ), диагностика "
          "router (6 "
          "точек, padding 0, нулевой Wo -> поправка 0, KL листьев); "
          "настоящая фабрика строит K26Parents/K26Flat для seed 0 и 1, "
          "общие тензоры = базовой модели, router пары одинаков, fitcheck — "
          "parents_attn24; отказ для seed вне {0, 1}; протокол не входит в "
          "зависимости обучения")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-26: обучение и проверки")
    ap.add_argument("cmd", choices=("real-check", "fitcheck", "train",
                                    "check-fresh", "micro-decision",
                                    "micro", "maps"))
    ap.add_argument("--kind", choices=("real-check", "fitcheck"),
                    default="real-check")
    ap.add_argument("--mode", choices=MODES, default="parents_attn24")
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
            if decided_micro() not in (None, a.micro):
                raise SystemExit(f"fitcheck micro {a.micro} ≠ решению "
                                 f"{decided_micro()}")
            return fitcheck(a)
        if a.cmd == "check-fresh":
            return check_fresh(a)
        if a.cmd == "micro-decision":
            return micro_decision(a)
        if a.cmd == "maps":
            return maps(a)
        if a.cmd == "micro":
            m = decided_micro()
            if m is None:
                raise SystemExit("решения о microbatch нет")
            print(m)
            return 0
        return train(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
