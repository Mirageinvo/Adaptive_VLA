#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-26: подготовка, планы SR и проверка блоков parents_attn24/flat_attn24.

prepare — reports/k26/prepare.json: подготовка K-21 (общие источники),
  конфигурация router, по сидам 0 и 1 — исторические контроли parents24
  (K-22/K-23) и flat24 (K-21/K-23): чекпойнт, SHA, отпечаток весов, отчёт,
  бюджет, начальный отпечаток (общие тензоры K-26 сверяются с ним); бюджет
  main K-26 = 8000; запись сида с отпечатком (k24_protocol.entry_fp);
  готовые записи при повторном prepare не меняются.
plan --seed S --mode M — reports/k26/plan_s<S>_<M>.json: метка M_s<S>,
  LIBERO-10, задачи 0–9, блоки 30/35/40 (план A K-21: setup, старты, smoke
  задачи 0 и 8 блока 30); источники с SHA каждого блока: план A K-21
  (fullbar, flat24_s0), fullbar_cached, K-22 (s0) или повтор K-23 (s1), K-24
  и K-25 того же сида (MR и cond — описательно).
check-block — штатные проверки + журнал router каждого вызова: формы,
  конечность, масса на padding = 0, ошибка сумм P ≤ 5e-6, флаги
  конечности, число токенов банка ≥ 1 и постоянно, уровни энтропии по
  точкам; у parents — геометрия корней; в smoke-блоках — debug-фрагмент
  первого вызова (S, P, log-prior, маска одной среды после слоя 24):
  P и prior пересчитываются на CPU (numpy float64) по формулам k26.
"""
import argparse
import datetime
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k21_protocol as pp  # noqa: E402
import k23_protocol as p23  # noqa: E402
import k24_protocol as p24  # noqa: E402  (регистрирует k23_repeat_s1)
import k25_protocol as p25  # noqa: E402  (регистрирует k24_s0/k24_s1)
import k26_attention as ka  # noqa: E402

KIND = "k26_plan"
PREPARE = "reports/k26/prepare.json"
BUDGET = 8000
MODES = ("parents_attn24", "flat_attn24")
BOOT = dict(n=10000, seed=20261027)
COMMON_KEYS = p24.COMMON_KEYS
entry_fp = p24.entry_fp
DBG_TOL = 1e-5
LEVELS = dict(parents_attn24=[1, 1, 2, 2, 3, 3], flat_attn24=[1] * 6)
CODE = ("k26_protocol.py", "k26_policy.py", "k26_harness.py",
        "k26_analyze.py", "k26_run.sh", "k26_train.py", "k26_model.py",
        "k26_attention.py", "k26_latency.py") + tuple(
            f for f in p25.CODE if not f.startswith("k26"))
sha, full_sha = pp.sha, pp.full_sha


def plan_path(seed, mode):
    return f"reports/k26/plan_s{seed}_{mode}.json"


def run_dir(plan_path_):
    return os.path.join("reports/k26/run", f"p{sha(plan_path_)}")


def block_name(lab, suite, t, b):
    return f"{lab}_{suite}_t{t}_i{b}.json"


def parse_label(label):
    m = re.fullmatch(r"(parents_attn24|flat_attn24)_s(\d+)", str(label))
    if not m:
        raise SystemExit(f"метка {label!r}: нужна parents_attn24_s<seed> "
                         f"или flat_attn24_s<seed>")
    return m.group(1), int(m.group(2))


# --- источники -------------------------------------------------------------
def ref_k25(seed):
    path = p25.plan_path(seed)
    plan = json.load(open(path, encoding="utf-8"))
    prob = p25.check_plan_inputs(plan)
    if prob:
        raise SystemExit(f"план K-25 s{seed}: " + "; ".join(prob[:3]))
    run = p25.run_dir(path)
    ok, gaps = p25.check_run(plan, path, run)
    want = len(plan["cells"]) * len(plan["labels"])
    if len(ok) != want or gaps:
        raise SystemExit(f"план K-25 s{seed}: проверено {len(ok)}/{want}, "
                         f"проблем {len(gaps)}")
    return dict(name=f"k25_s{seed}", path=os.path.abspath(path),
                sha12=sha(path), run_dir=os.path.abspath(run), plan=plan,
                blocks={(lab, s, int(t), int(b)): f
                        for (lab, s, t, b), f in ok.items()})


p23.REFS["k25_s0"] = lambda: ref_k25(0)
p23.REFS["k25_s1"] = lambda: ref_k25(1)


def entry_problems(e, common=None):
    p = []
    if entry_fp(e) != e.get("fingerprint"):
        p.append("отпечаток записи сида не совпадает с содержимым")
    for arch in ("parents", "flat"):
        c = e[f"control_{arch}"]
        if not os.path.exists(c["ckpt"]) or full_sha(c["ckpt"]) != \
                c["ckpt_sha1"]:
            p.append(f"контроль {arch}: чекпойнт изменён или отсутствует")
        if not os.path.exists(c["train_report"]) or \
                full_sha(c["train_report"]) != c["train_report_sha1"]:
            p.append(f"контроль {arch}: отчёт обучения изменён")
        if int(c["updates"]) != int(e["budget_main"]):
            p.append(f"контроль {arch}: бюджет {c['updates']} ≠ "
                     f"{e['budget_main']}")
    if e.get("router") != ka.CONFIG:
        p.append("конфигурация router изменилась")
    if common is not None and e.get("common") != common:
        p.append("общие источники (подготовка, учитель, BAR) изменились")
    return p


def seed_entry(seed, path=None):
    pr = json.load(open(path or PREPARE, encoding="utf-8"))
    e = (pr.get("seeds") or {}).get(str(seed))
    if not e or e.get("status") != "ready":
        raise SystemExit(f"seed {seed}: подготовка K-26 не готова")
    prob = entry_problems(e, {k: pr[k] for k in COMMON_KEYS})
    if prob:
        raise SystemExit(f"seed {seed}: " + "; ".join(prob))
    return e


def prepare():
    import k21_data as d21
    old = json.load(open(PREPARE, encoding="utf-8")) \
        if os.path.exists(PREPARE) else None
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    tj = json.load(open(d21.TEACHER_REPORT, encoding="utf-8"))
    D = d21.load_prepared(require_teacher=False)
    vt, missing, alltasks = p24.val_coverage(D["task"], D["val_rows"],
                                             D["train_rows"])
    rep = dict(created=(old or {}).get("created") or datetime.datetime.now()
               .isoformat(timespec="seconds"),
               prep_id=prep["prep_id"], prepare_sha1=full_sha(d21.REPORT),
               teacher_sha1=tj["out"]["sha1"],
               teacher_report_sha1=full_sha(d21.TEACHER_REPORT),
               bar_revision=prep["bar_pin"]["revision"],
               bounds=prep["bounds_codec"], train_tasks=len(alltasks),
               val_tasks=len(vt), val_tasks_missing=missing,
               router=ka.CONFIG, seeds=dict((old or {}).get("seeds") or {}))
    common = {k: rep[k] for k in COMMON_KEYS}
    if old:
        for k in COMMON_KEYS:
            if old[k] != rep[k]:
                raise SystemExit(f"подготовка K-21 изменилась: {k}")
    for s in (0, 1):
        e = rep["seeds"].get(str(s))
        if e and e.get("status") == "ready":
            prob = entry_problems(e, common)
            if prob:
                raise SystemExit(f"seed {s}: " + "; ".join(prob))
            continue
        cp, cf = p24.control("parents24", s), p24.control("flat24", s)
        if cp is None or cf is None:
            rep["seeds"][str(s)] = dict(status="pending",
                                        why="контроль ещё не обучен")
            continue
        for nm, c in (("parents", cp), ("flat", cf)):
            if c["seed"] != s:
                raise SystemExit(f"seed {s}: seed контроля {nm} {c['seed']}")
        e = dict(status="ready", seed=s, budget_main=BUDGET,
                 control_parents=cp, control_flat=cf, router=ka.CONFIG,
                 common=common, fixed=datetime.datetime.now().isoformat(
                     timespec="seconds"))
        e["fingerprint"] = entry_fp(e)
        prob = entry_problems(e, common)
        if prob:
            raise SystemExit(f"seed {s}: " + "; ".join(prob))
        rep["seeds"][str(s)] = e
    rep["code"] = {f: sha(os.path.join(HERE, f)) for f in CODE
                   if os.path.exists(os.path.join(HERE, f))}
    os.makedirs("reports/k26", exist_ok=True)
    json.dump(rep, open(PREPARE + ".tmp", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    os.replace(PREPARE + ".tmp", PREPARE)
    print(f"  подготовка {PREPARE}: prep_id {rep['prep_id']}; val_sel задач "
          f"{rep['val_tasks']} из {rep['train_tasks']}; router "
          f"{ka.VERSION} (γ {ka.GAMMA}, ε {ka.EPS}, после слоёв "
          f"{ka.CONFIG['after_layers_1based']})")
    for s, e in sorted(rep["seeds"].items()):
        print(f"    seed {s}: {e['status']}" + (
            f", бюджет {e['budget_main']}; контроли parents/flat "
            f"{e['control_parents']['source']}/{e['control_flat']['source']}"
            f"; init parents {e['control_parents']['init_hash'][:12]}, flat "
            f"{e['control_flat']['init_hash'][:12]}; отпечаток "
            f"{e['fingerprint'][:12]}" if e["status"] == "ready"
            else f" ({e.get('why')})"))
    return 0


# --- план -------------------------------------------------------------------
def make_plan(seed, mode, device):
    import torch
    import k21_data as d21
    import k26_train as k26t
    if mode not in MODES:
        raise SystemExit(f"режим {mode}")
    e = seed_entry(seed)
    trp = k26t.report_path(seed, mode)
    tr = json.load(open(trp, encoding="utf-8"))
    if int(tr["updates"]) != int(e["budget_main"]):
        raise SystemExit("обучение K-26 не совпадает с бюджетом")
    ck = os.path.join(k26t.outdir(seed), mode, "best.pt")
    if full_sha(ck) != tr["best_sha1"]:
        raise SystemExit(f"{ck} ≠ отчёту обучения")
    c = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = c["config"]
    if c["weights_hash"] != tr["best_weights_hash"] or \
            int(cfg["seed"]) != seed or cfg.get("mode") != mode or \
            cfg.get("architecture") != f"k26_{mode}" or \
            cfg.get("router") != ka.CONFIG or \
            cfg["k26"].get("seed_entry_fp") != e["fingerprint"]:
        raise SystemExit("чекпойнт K-26: отпечаток, seed, режим, "
                         "архитектура, router или запись подготовки")
    lab = f"{mode}_s{seed}"
    spec = dict(arch=mode, seed=seed, ckpt=os.path.abspath(ck),
                ckpt_sha1=full_sha(ck), weights_hash=c["weights_hash"],
                best_update=int(c["update"]), val_L=float(c["val_L"]),
                train_report=os.path.abspath(trp),
                train_report_sha1=full_sha(trp),
                prep_id=cfg["prep_id"], budget_main=BUDGET,
                init_hash=tr["init_hash"],
                router_init_hash=tr["router_init_hash"],
                micro=int(cfg["micro"]), router=ka.CONFIG)
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    rA = p23.ref_k21_main()
    A = rA["plan"]
    refs = {"k21_main_A": p23.pin_ref(rA),
            "fullbar_cached": p23.pin_ref(p23.ref_cached()),
            f"k24_s{seed}": p23.pin_ref(p25.ref_k24(seed)),
            f"k25_s{seed}": p23.pin_ref(ref_k25(seed))}
    if seed == 0:
        refs["k22"] = p23.pin_ref(p23.ref_k22())
    else:
        refs[f"k23_repeat_s{seed}"] = p23.pin_ref(p24.ref_k23_repeat(seed))
    desc = {}
    for (l_, s_, t_, b_), f in rA["blocks"].items():
        if l_ == "fullbar":
            desc.setdefault(t_, json.load(open(f, encoding="utf-8")).get(
                "task_description"))
    if any(b + 4 >= 45 for b in A["blocks"]):
        raise SystemExit("блоки задевают резерв 45–49")
    inp = dict(prepare=d21.REPORT, teacher=d21.TEACHER_REPORT,
               train_report=trp)
    return dict(
        kind=KIND, name=f"k26_s{seed}_{mode}", train_seed=seed, mode=mode,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        labels=[lab], policy={lab: "k15f"}, specs={lab: spec},
        seed_entry=e, seed_entry_fp=e["fingerprint"],
        controls=dict(pair=f"{[m for m in MODES if m != mode][0]}_s{seed}",
                      old_same_arch=("parents24" if mode == "parents_attn24"
                                     else "flat24") + f"_s{seed}",
                      descriptive=[f"parents_mr24_s{seed}",
                                   f"parents_cond24_s{seed}", "fullbar",
                                   "fullbar_cached"]),
        suites=["10"], tasks=list(A["tasks"]), blocks=list(A["blocks"]),
        cells=[["10", t, b] for t in A["tasks"] for b in A["blocks"]],
        smoke=[["10", t, b] for t, b in A["smoke"]], setups={"10": A["setup"]},
        suite_tasks={"10": [desc[t] for t in sorted(desc)]},
        refs=A["refs"], refs_sha1=A["refs_sha1"], device=str(device),
        snapshot_path=snap, prep_id=prep["prep_id"],
        bar_revision=prep["bar_pin"]["revision"], bounds=prep["bounds_codec"],
        n_envs=5, env_seed=101, bootstrap=dict(BOOT), router=ka.CONFIG,
        references=refs,
        inputs={k: os.path.abspath(v) for k, v in inp.items()},
        inputs_sha1={k: sha(v) for k, v in inp.items()},
        code={f: sha(os.path.join(HERE, f)) for f in CODE
              if os.path.exists(os.path.join(HERE, f))},
        primary=dict(comparison=["parents_attn24_s{S}", "flat_attn24_s{S}"],
                     holm_over_seeds=True),
        note="разработочный банк 30–44 (разведочно); основное — "
             "parents_attn24 − flat_attn24 того же сида; обязательные "
             "вторичные — новый против старого той же архитектуры")


def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план K-26"]
    for grp in ("inputs", "refs"):
        for k, f in (plan.get(grp) or {}).items():
            if not os.path.exists(f) or sha(f) != plan[f"{grp}_sha1"][k]:
                p.append(f"{grp} {k} изменён или отсутствует ({f})")
    try:
        e = seed_entry(plan["train_seed"])
        if e["fingerprint"] != plan["seed_entry_fp"] or \
                entry_fp(plan["seed_entry"]) != plan["seed_entry_fp"]:
            p.append("запись подготовки сида ≠ закреплённой в плане")
    except SystemExit as ex:
        p.append(f"запись подготовки сида: {ex.code}")
    if plan.get("router") != ka.CONFIG:
        p.append("конфигурация router ≠ плану")
    for lab, s in plan["specs"].items():
        if not os.path.exists(s["ckpt"]) or full_sha(s["ckpt"]) != \
                s["ckpt_sha1"]:
            p.append(f"чекпойнт {lab} изменён или отсутствует")
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    return p


# --- журнал router -------------------------------------------------------------
def np_tree_probs(S, valid, sizes, eps, hierarchical):
    """Независимая numpy-реализация (float64): S [h, A, N], valid [N]."""
    S = np.asarray(S, np.float64)
    v = np.asarray(valid, bool)
    n = v.sum()
    lu = np.where(v, -np.log(n), -np.inf)

    def lsm(x):
        x = np.where(v, x, -np.inf)
        m = x.max(-1, keepdims=True)
        return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))
    if not hierarchical:
        lp = lsm(S + lu)
        return np.exp(lp), np.broadcast_to(lu, S.shape)
    outs, pri, off, prev = [], [], 0, None
    for lev, k in enumerate(sizes):
        s = S[:, off:off + k]
        if lev == 0:
            lpr = np.broadcast_to(lu, s.shape)
        else:
            par = np.arange(k) // 2
            with np.errstate(divide="ignore"):
                lpr = np.log((1 - eps) * np.exp(prev[:, par]) + eps *
                             np.exp(lu))
        lp = lsm(s + lpr)
        outs.append(lp)
        pri.append(lpr)
        prev = lp
        off += k
    return np.exp(np.concatenate(outs, 1)), np.concatenate(pri, 1)


def debug_problems(z, label, mode):
    keys = ("S", "P", "valid", "log_prior")
    need = [f"k21_attn_dbg_{k}" for k in keys]
    miss = [k for k in need if k not in z]
    if miss:
        return [f"{label}: в smoke-журнале нет debug-фрагмента {miss}"]
    S, P, v, lpr = (np.asarray(z[f"k21_attn_dbg_{k}"]) for k in keys)
    if S.ndim != 4 or S.shape[0] != 1 or P.shape != S.shape or \
            v.shape != (1, S.shape[-1]) or lpr.shape != S.shape:
        return [f"{label}: debug-фрагмент формы {S.shape}/{P.shape}/"
                f"{v.shape}/{lpr.shape}"]
    A = S.shape[2]
    hier = mode == "parents_attn24"
    want_A = 28 if hier else 16
    if A != want_A:
        return [f"{label}: debug-фрагмент на {A} узлах, ожидалось {want_A}"]
    vv = v[0].astype(bool)
    Pn, lprn = np_tree_probs(S[0], vv, (4, 8, 16) if hier else (16,),
                             ka.EPS, hier)
    p = []
    d = float(np.abs(Pn - P[0]).max())
    if not d <= DBG_TOL:
        p.append(f"{label}: P debug-фрагмента ≠ пересчёту на CPU ({d:.2e})")
    dl = float(np.abs(np.where(vv, lprn - lpr[0], 0.0)).max())
    if not dl <= DBG_TOL * 10:
        p.append(f"{label}: log-prior ≠ пересчёту ({dl:.2e})")
    if float(np.abs(P[0][..., ~vv]).max(initial=0.0)) != 0.0:
        p.append(f"{label}: debug P на padding ≠ 0")
    return p


def attn_problems(z, label, mode, debug=False):
    keys = ("pad", "sumerr", "finite", "rms", "H", "KL", "ntok")
    need = [f"k21_attn_{k}" for k in keys] + [f"k15d_level_{label}"]
    if mode == "parents_attn24":
        need += ["k21_root", "k21_attn_split1", "k21_attn_split2"]
    miss = [k for k in need if k not in z]
    if miss:
        return [f"{label}: в журнале нет {miss}"]
    g = {k: np.asarray(z[f"k21_attn_{k}"], np.float64) for k in keys}
    lv = np.asarray(z[f"k15d_level_{label}"])
    C, B = lv.shape[:2]
    want = dict(pad=(C, 6), sumerr=(C, 6), finite=(C, 6), rms=(C, 6),
                H=(C, 6, 3), KL=(C, 6, 3), ntok=(C, B))
    p = [f"{label}: attn_{k} формы {g[k].shape}, ожидалось {s}"
         for k, s in want.items() if g[k].shape != s]
    if mode == "parents_attn24":
        for k, s in (("k21_root", (C, B, 4, 2)),
                     ("k21_attn_split1", (C, B, 4)),
                     ("k21_attn_split2", (C, B, 8))):
            if np.asarray(z[k]).shape != s:
                p.append(f"{label}: {k} формы {np.asarray(z[k]).shape}")
    if p:
        return p
    bad = [k for k, v in g.items() if not np.isfinite(v).all()]
    if bad:
        return [f"{label}: нечисловые значения в attn_{bad}"]
    if (g["pad"] != 0).any():
        p.append(f"{label}: внимание на padding (max {g['pad'].max():.2e})")
    if (g["sumerr"] > ka.SUM_TOL).any():
        p.append(f"{label}: ошибка сумм P {g['sumerr'].max():.2e}")
    if (g["finite"] != 1).any():
        p.append(f"{label}: router вернул нечисловые P")
    if (g["ntok"] < 1).any() or len(np.unique(g["ntok"])) != 1:
        p.append(f"{label}: токены банка {np.unique(g['ntok']).tolist()}")
    nl = np.asarray(LEVELS[mode])
    have = (g["H"] >= 0).sum(-1)
    if (have != nl[None]).any():
        p.append(f"{label}: уровни по точкам {have[0].tolist()} ≠ "
                 f"{nl.tolist()}")
    if (g["H"] > 1 + 1e-6).any() or (g["rms"] < 0).any():
        p.append(f"{label}: энтропия > 1 или RMS < 0")
    if mode == "parents_attn24":
        root = np.asarray(z["k21_root"], np.float64)
        if not np.isfinite(root).all() or (root[..., 1] < root[..., 0]).any():
            p.append(f"{label}: интервалы корней")
    if debug and not p:
        p += debug_problems(z, label, mode)
    return p


def check_block(plan, plan_path_, path, label, suite, task, block):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import k16_protocol as p16
    if label not in plan["labels"]:
        return [f"метка {label!r} не из плана"]
    if [suite, int(task), int(block)] not in [[c[0], int(c[1]), int(c[2])]
                                              for c in plan["cells"]]:
        return [f"ячейка {suite}/{task}/{block} не из плана"]
    p = pr.validate_block(path, label, int(task), int(block),
                          require_single=True, setup=plan["setups"][suite])
    if p:
        return p
    d, z = p16.load(path)
    if os.path.basename(os.path.normpath(str(d.get("ckpt")))) != \
            plan["bar_revision"]:
        p.append(f"{label}: снимок BAR блока не закреплённой ревизии")
    if d.get("task_description") != plan["suite_tasks"][suite][int(task)]:
        p.append(f"{label}: задача блока ≠ задаче плана")
    spec = plan["specs"][label]
    arch, seed = parse_label(label)
    j = d.get("joint") or {}
    smoke = [suite, int(task), int(block)] in [
        [c[0], int(c[1]), int(c[2])] for c in plan["smoke"]]
    want = dict(label=label, arch=arch, train_seed=seed,
                plan_sha1=sha(plan_path_), prep_id=plan["prep_id"],
                bar_revision=plan["bar_revision"], ckpt_sha1=spec["ckpt_sha1"],
                weights_hash=spec["weights_hash"],
                router_version=ka.VERSION, router_debug=smoke,
                k21_policy_sha1=full_sha(os.path.join(HERE,
                                                      "k21_main_policy.py")),
                k26_policy_sha1=full_sha(os.path.join(HERE, "k26_policy.py")),
                harness_adapter_sha1=sha(os.path.join(HERE, "k26_harness.py")))
    for k, v in want.items():
        if j.get(k) != v:
            p.append(f"{label}: {k} = {j.get(k)!r}, ожидалось {v!r}")
    p += pp.journal_problems(label, z, plan["bounds"])
    if not p:
        p += attn_problems(z, label, arch, debug=smoke)
    if not p:
        p += hpc.actions_problems(label, z, hpc.harness_scale())
    return p


def check_run(plan, plan_path_, rdir, cells=None):
    ok, gaps = {}, []
    for s, t, b in cells or plan["cells"]:
        for lab in plan["labels"]:
            pth = os.path.join(rdir, block_name(lab, s, t, b))
            if not os.path.exists(pth):
                gaps.append(dict(label=lab, suite=s, task=t, block=b,
                                 why="не готов"))
                continue
            prob = check_block(plan, plan_path_, pth, lab, s, t, b)
            if prob:
                gaps.append(dict(label=lab, suite=s, task=t, block=b,
                                 why="; ".join(prob[:2])))
            else:
                ok[(lab, s, int(t), int(b))] = pth
    return ok, gaps


def fake_journal(torch, mode, C=2, B=3, seed=0, debug=True):
    """Журнал как у k26_policy на игрушечной модели (для самопроверок)."""
    import copy
    import k26_model as k26m
    import k26_policy as kp
    M = k26m.build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    m = M["make"](mode, copy.deepcopy(k26m.fake_bar(torch)), L, U, seed=0)
    with torch.no_grad():
        m.router.Wo.weight.normal_(0, 0.5, generator=torch.Generator()
                                   .manual_seed(seed))
    m.eval()
    lab = f"{mode}_s0"
    rows = {}
    for c in range(C):
        m.collect = True
        m.keep_maps = debug and c == 0
        with torch.no_grad():
            out = m(k26m.fake_batch(torch, B=B, seed=seed + c, pad=2), 1,
                    torch.randn(B, 11))
        rec = kp.journal_record(out, mode, keep_debug=debug and c == 0)
        g = torch.where(out["grip_logit"] > 0, 1.0, -1.0)
        rec["level"] = torch.cat([out["arm"], g[..., None]], -1)[:, :8] \
            .numpy().astype(np.float32)
        if "root" in out:
            rec["root"] = out["root"].numpy()
        for k, v in rec.items():
            rows.setdefault(k, []).append(np.asarray(v))
    z = {}
    for k, v in rows.items():
        arr = np.stack(v)
        if k == "level":
            z[f"k15d_level_{lab}"] = arr
        else:
            z[f"k21_{k}"] = arr
    return z, lab, dict(L=L.tolist(), U=U.tolist())


def selftest():
    import torch
    R = ka.build(torch)
    g = torch.Generator().manual_seed(4)
    # numpy-пересчёт = torch-реализации
    S = torch.randn(2, 4, 28, 9, generator=g, dtype=torch.float64) * 3
    v = torch.ones(2, 9, dtype=torch.bool)
    v[1, 5:] = False
    for hier, sz in ((True, (4, 8, 16)), (False, (28,))):
        o = R["tree_probs"](S, v, sz, ka.EPS, hier)
        for b in range(2):
            Pn, lpn = np_tree_probs(S[b].numpy(), v[b].numpy(), sz, ka.EPS,
                                    hier)
            assert np.abs(Pn - o["P"][b].numpy()).max() < 1e-12
            vb = v[b].numpy()
            assert np.abs(np.where(vb, lpn - o["log_prior"][b].numpy(),
                                   0)).max() < 1e-12
    for mode in MODES:
        z, lab, bounds = fake_journal(torch, mode)
        assert attn_problems(z, lab, mode, debug=True) == [], \
            attn_problems(z, lab, mode, debug=True)
        for key, mut, word in (
                ("k21_attn_pad", lambda a: a + 1e-3, "padding"),
                ("k21_attn_sumerr", lambda a: a + 1e-3, "сумм"),
                ("k21_attn_finite", lambda a: a * 0, "нечисловые P"),
                ("k21_attn_ntok", lambda a: a * 0, "токены"),
                ("k21_attn_dbg_P", lambda a: a * 1.01, "пересчёту"),
                ("k21_attn_dbg_S", lambda a: a + np.random.default_rng(0)
                 .normal(0, 1, a.shape), "пересчёту")):
            zb = dict(z)
            zb[key] = mut(np.array(z[key], copy=True))
            pb = attn_problems(zb, lab, mode, debug=True)
            assert pb and any(word in x for x in pb), (mode, key, pb)
        zb = dict(z)
        h = np.array(z["k21_attn_H"], copy=True)
        h[:, 0, 1] = 0.5
        zb["k21_attn_H"] = h
        assert any("уровни" in x for x in attn_problems(zb, lab, mode))
        for val in (np.nan, np.inf):
            zb = dict(z)
            a = np.array(z["k21_attn_rms"], copy=True)
            a[0, 0] = val
            zb["k21_attn_rms"] = a
            assert any("нечисловые" in x for x in attn_problems(zb, lab,
                                                                mode))
        zb = {k: v for k, v in z.items() if not k.startswith("k21_attn_dbg")}
        assert attn_problems(zb, lab, mode, debug=False) == []
        assert any("debug" in x for x in attn_problems(zb, lab, mode,
                                                       debug=True))
    zp, labp, _b = fake_journal(torch, "parents_attn24")
    assert any("формы" in x or "нет" in x for x in attn_problems(
        zp, "flat_attn24_s0", "flat_attn24"))
    assert parse_label("flat_attn24_s1") == ("flat_attn24", 1)
    try:
        parse_label("parents24_s0")
        raise AssertionError("чужая метка принята")
    except SystemExit:
        pass
    assert len(set(CODE)) == len(CODE)
    print("самопроверка k26_protocol пройдена: numpy-пересчёт P/prior = "
          "torch (оба режима), журнал router игрушечной руки проходит; "
          "padding, суммы, флаг конечности, токены, уровни, NaN/Inf, порча "
          "debug-фрагмента (P, S) и его отсутствие в smoke — отказ")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-26: подготовка и планы")
    ap.add_argument("cmd", choices=("prepare", "plan", "check-plan",
                                    "check-block", "smoke-check", "cells",
                                    "info", "selftest"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=MODES, default="parents_attn24")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--block", nargs=5, default=None)
    ap.add_argument("--which", choices=("smoke", "all"), default="all")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    try:
        if a.cmd == "prepare":
            return prepare()
        if a.cmd == "plan":
            pth = plan_path(a.seed, a.mode)
            if os.path.exists(pth):
                print(f"  план {pth} уже зафиксирован")
                return 0
            plan = make_plan(a.seed, a.mode, a.device)
            with open(pth + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=1)
            os.replace(pth + ".tmp", pth)
            print(f"  план {pth}: {plan['labels']}, ячеек {len(plan['cells'])}"
                  f"; источников {sum(len(r['blocks']) for r in plan['references'].values())} блоков")
            return 0
        plan = json.load(open(a.plan, encoding="utf-8"))
        if a.cmd == "check-plan":
            prob = check_plan_inputs(plan)
            if not prob:
                p23.verify_refs(plan)
            print("\n".join(prob) if prob else "  входы и источники сверены")
            return 3 if prob else 0
        if a.cmd == "info":
            print(plan["snapshot_path"])
            return 0
        if a.cmd == "cells":
            for s, t, b in (plan["smoke"] if a.which == "smoke"
                            else plan["cells"]):
                for lab in plan["labels"]:
                    print(f"{lab}:{s}:{t}:{b}")
            return 0
        if a.cmd == "check-block":
            path, lab, s, t, b = a.block
            prob = check_block(plan, a.plan, path, lab, s, t, b)
            if prob:
                print("; ".join(prob[:6]))
            return 2 if prob else 0
        ok, gaps = check_run(plan, a.plan, run_dir(a.plan),
                             [tuple(x) for x in plan["smoke"]])
        print(f"  smoke: проверено {len(ok)} из {len(plan['smoke'])}; проблем "
              f"{len(gaps)}" + (f": {gaps[:4]}" if gaps else ""))
        if ok:
            sz = [os.path.getsize(os.path.join(os.path.dirname(f), json.load(
                open(f, encoding="utf-8"))["actions_npz"])) for f in ok.values()]
            print(f"  размер журнала smoke (с debug-фрагментом): "
                  f"{np.mean(sz) / 2 ** 20:.1f} МБ на блок (NPZ)")
        return 3 if gaps else 0
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
