#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-25: подготовка, планы SR и проверка блоков parents_cond24.

prepare — reports/k25/prepare.json: подготовка K-21 (общие источники), по
  сидам 0 и 1 — контроли parents_mr24 (K-24, основной), parents24 и flat24
  (K-22/K-23/K-21): чекпойнт, SHA, отпечаток весов, отчёт, бюджет, начальный
  отпечаток; бюджет main K-25 = 8000 (= фактический у всех трёх контролей);
  полная запись сида с отпечатком (k24_protocol.entry_fp); готовые записи
  при повторном prepare не меняются (сверка отпечатка и файлов).
plan --seed S — метка parents_cond24_s<S>, LIBERO-10, задачи 0–9, блоки
  30/35/40; источники с SHA каждого блока: план A K-21, fullbar_cached,
  K-22 (s0) или повтор K-23 (s1), план K-24 того же сида.
check-block — штатные проверки + журнал сборки: формы и конечность всех
  массивов; W0/W1/W2 из геометрии (≤ 1e-5); по записанным W, C0, g1, g2,
  δ1, y2 пересборка путей z1 -> z2 -> Z и формы G0+G1+G2 (отн. 5e-5);
  tanh-формула и схват = знак Σ W2·y2[6] против исполненных команд.
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
import k25_readout as kr  # noqa: E402

KIND = "k25_plan"
PREPARE = "reports/k25/prepare.json"
BUDGET = 8000
BOOT = dict(n=10000, seed=20261026)
TOL, GEOM_TOL = p24.MR_TOL, p24.GEOM_TOL
T = 20
COMMON_KEYS = p24.COMMON_KEYS
entry_fp = p24.entry_fp
CODE = ("k25_protocol.py", "k25_policy.py", "k25_harness.py",
        "k25_analyze.py", "k25_run.sh", "k25_train.py", "k25_model.py",
        "k25_readout.py", "k25_latency.py") + tuple(
            f for f in p24.CODE if not f.startswith("k25"))
sha, full_sha = pp.sha, pp.full_sha


def plan_path(seed):
    return f"reports/k25/plan_s{seed}.json"


def run_dir(plan_path_):
    return os.path.join("reports/k25/run", f"p{sha(plan_path_)}")


def block_name(lab, suite, t, b):
    return f"{lab}_{suite}_t{t}_i{b}.json"


def parse_label(label):
    m = re.fullmatch(r"parents_cond24_s(\d+)", str(label))
    if not m:
        raise SystemExit(f"метка {label!r}: нужна parents_cond24_s<seed>")
    return "parents_cond24", int(m.group(1))


# --- источники -------------------------------------------------------------
def ref_k24(seed):
    path = p24.plan_path(seed)
    plan = json.load(open(path, encoding="utf-8"))
    prob = p24.check_plan_inputs(plan)
    if prob:
        raise SystemExit(f"план K-24 s{seed}: " + "; ".join(prob[:3]))
    run = p24.run_dir(path)
    ok, gaps = p24.check_run(plan, path, run)
    want = len(plan["cells"]) * len(plan["labels"])
    if len(ok) != want or gaps:
        raise SystemExit(f"план K-24 s{seed}: проверено {len(ok)}/{want}, "
                         f"проблем {len(gaps)}")
    return dict(name=f"k24_s{seed}", path=os.path.abspath(path),
                sha12=sha(path), run_dir=os.path.abspath(run), plan=plan,
                blocks={(lab, s, int(t), int(b)): f
                        for (lab, s, t, b), f in ok.items()})


p23.REFS["k24_s0"] = lambda: ref_k24(0)
p23.REFS["k24_s1"] = lambda: ref_k24(1)


def control_mr(seed):
    import torch
    ck = f"data/k24/runs/s{seed}/parents_mr24"
    tr = f"reports/k24/train_s{seed}.json"
    if not os.path.exists(tr) or not os.path.exists(f"{ck}/best.pt"):
        return None
    r = json.load(open(tr, encoding="utf-8"))
    if full_sha(f"{ck}/best.pt") != r["best_sha1"]:
        raise SystemExit(f"{ck}/best.pt ≠ отчёту {tr}")
    rs = torch.load(f"{ck}/resume.pt", map_location="cpu",
                    weights_only=False)
    return dict(source="K-24", ckpt=os.path.abspath(f"{ck}/best.pt"),
                ckpt_sha1=r["best_sha1"], weights_hash=r["best_weights_hash"],
                train_report=os.path.abspath(tr), train_report_sha1=full_sha(tr),
                updates=int(r["updates"]), best=r["best"],
                init_hash=rs["init_hash"], seed=int(rs["config"]["seed"]))


def entry_problems(e, common=None):
    p = []
    if entry_fp(e) != e.get("fingerprint"):
        p.append("отпечаток записи сида не совпадает с содержимым")
    for arch in ("mr", "parents", "flat"):
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
    if common is not None and e.get("common") != common:
        p.append("общие источники (подготовка, учитель, BAR) изменились")
    return p


def seed_entry(seed, path=None):
    pr = json.load(open(path or PREPARE, encoding="utf-8"))
    e = (pr.get("seeds") or {}).get(str(seed))
    if not e or e.get("status") != "ready":
        raise SystemExit(f"seed {seed}: подготовка не готова")
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
               readout=dict(version=kr.VERSION, formula=kr.FORMULA,
                            gate_init=kr.GATE_INIT),
               seeds=dict((old or {}).get("seeds") or {}))
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
        cm = control_mr(s)
        cp, cf = p24.control("parents24", s), p24.control("flat24", s)
        if cm is None or cp is None or cf is None:
            rep["seeds"][str(s)] = dict(status="pending",
                                        why="контроль ещё не обучен")
            continue
        for nm, c in (("mr", cm), ("parents", cp), ("flat", cf)):
            if c["seed"] != s:
                raise SystemExit(f"seed {s}: seed контроля {nm} {c['seed']}")
        e = dict(status="ready", seed=s, budget_main=BUDGET,
                 control_mr=cm, control_parents=cp, control_flat=cf,
                 common=common, fixed=datetime.datetime.now().isoformat(
                     timespec="seconds"))
        prob = entry_problems(dict(e, fingerprint=entry_fp(e)), common)
        if prob:
            raise SystemExit(f"seed {s}: " + "; ".join(prob))
        e["fingerprint"] = entry_fp(e)
        rep["seeds"][str(s)] = e
    rep["code"] = {f: sha(os.path.join(HERE, f)) for f in CODE
                   if os.path.exists(os.path.join(HERE, f))}
    os.makedirs("reports/k25", exist_ok=True)
    json.dump(rep, open(PREPARE + ".tmp", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    os.replace(PREPARE + ".tmp", PREPARE)
    print(f"  подготовка {PREPARE}: prep_id {rep['prep_id']}; val_sel задач "
          f"{rep['val_tasks']} из {rep['train_tasks']}")
    for s, e in sorted(rep["seeds"].items()):
        print(f"    seed {s}: {e['status']}" + (
            f", бюджет {e['budget_main']}; контроли MR/parents/flat "
            f"{e['control_mr']['source']}/{e['control_parents']['source']}/"
            f"{e['control_flat']['source']}; init parents "
            f"{e['control_parents']['init_hash'][:12]}; отпечаток "
            f"{e['fingerprint'][:12]}" if e["status"] == "ready"
            else f" ({e.get('why')})"))
    return 0


# --- план -------------------------------------------------------------------
def make_plan(seed, device):
    import torch
    import k21_data as d21
    import k25_train as k25t
    e = seed_entry(seed)
    trp = k25t.report_path(seed)
    tr = json.load(open(trp, encoding="utf-8"))
    if int(tr["updates"]) != int(e["budget_main"]):
        raise SystemExit("обучение K-25 не совпадает с бюджетом")
    ck = os.path.join(k25t.outdir(seed), k25t.MODE, "best.pt")
    if full_sha(ck) != tr["best_sha1"]:
        raise SystemExit(f"{ck} ≠ отчёту обучения")
    c = torch.load(ck, map_location="cpu", weights_only=False)
    if c["weights_hash"] != tr["best_weights_hash"] or \
            int(c["config"]["seed"]) != seed or \
            c["config"].get("architecture") != "k25_parents_cond24" or \
            c["config"]["k25"].get("seed_entry_fp") != e["fingerprint"]:
        raise SystemExit("чекпойнт K-25: отпечаток, seed, архитектура или "
                         "запись подготовки")
    lab = f"parents_cond24_s{seed}"
    spec = dict(arch="parents_cond24", seed=seed, ckpt=os.path.abspath(ck),
                ckpt_sha1=full_sha(ck), weights_hash=c["weights_hash"],
                best_update=int(c["update"]), val_L=float(c["val_L"]),
                train_report=os.path.abspath(trp),
                train_report_sha1=full_sha(trp),
                prep_id=c["config"]["prep_id"], budget_main=BUDGET)
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    rA = p23.ref_k21_main()
    A = rA["plan"]
    refs = {"k21_main_A": p23.pin_ref(rA),
            "fullbar_cached": p23.pin_ref(p23.ref_cached()),
            f"k24_s{seed}": p23.pin_ref(ref_k24(seed))}
    if seed == 0:
        refs["k22"] = p23.pin_ref(p23.ref_k22())
    else:
        refs[f"k23_repeat_s{seed}"] = p23.pin_ref(p24.ref_k23_repeat(seed))
    desc = {}
    for (l_, s_, t_, b_), f in rA["blocks"].items():
        if l_ == "fullbar":
            desc.setdefault(t_, json.load(open(f, encoding="utf-8")).get(
                "task_description"))
    inp = dict(prepare=d21.REPORT, teacher=d21.TEACHER_REPORT,
               train_report=trp)
    return dict(
        kind=KIND, name=f"k25_s{seed}", train_seed=seed,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        labels=[lab], policy={lab: "k15f"}, specs={lab: spec},
        seed_entry=e, seed_entry_fp=e["fingerprint"],
        controls=dict(main=f"parents_mr24_s{seed}",
                      parents=f"parents24_s{seed}", flat=f"flat24_s{seed}"),
        suites=["10"], tasks=list(A["tasks"]), blocks=list(A["blocks"]),
        cells=[["10", t, b] for t in A["tasks"] for b in A["blocks"]],
        smoke=[["10", t, b] for t, b in A["smoke"]], setups={"10": A["setup"]},
        suite_tasks={"10": [desc[t] for t in sorted(desc)]},
        refs=A["refs"], refs_sha1=A["refs_sha1"], device=str(device),
        snapshot_path=snap, prep_id=prep["prep_id"],
        bar_revision=prep["bar_pin"]["revision"], bounds=prep["bounds_codec"],
        n_envs=5, env_seed=101, bootstrap=dict(BOOT),
        readout=dict(version=kr.VERSION, tol=TOL, geom_tol=GEOM_TOL),
        references=refs,
        inputs={k: os.path.abspath(v) for k, v in inp.items()},
        inputs_sha1={k: sha(v) for k, v in inp.items()},
        code={f: sha(os.path.join(HERE, f)) for f in CODE
              if os.path.exists(os.path.join(HERE, f))},
        note="разработочный банк 30–44; основное — parents_cond24 − "
             "parents_mr24 того же сида; parents и flat — обязательные "
             "вторичные")


def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план K-25"]
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
    for lab, s in plan["specs"].items():
        if not os.path.exists(s["ckpt"]) or full_sha(s["ckpt"]) != \
                s["ckpt_sha1"]:
            p.append(f"чекпойнт {lab} изменён или отсутствует")
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    return p


# --- журнал сборки -------------------------------------------------------------
COND_SHAPES = dict(C0=(4, 6), g1=(8,), g2=(16,), delta1=(8, T, 6),
                   y2=(16, T, 7), Z=(T, 6), G0=(T, 6), G1=(T, 6), G2=(T, 6),
                   W0=(T, 4), W1=(T, 8), W2=(T, 16), split1=(4,),
                   split2=(8,))


def cond_problems(z, label, bounds, tol=TOL):
    need = [f"k21_cond_{k}" for k in COND_SHAPES] + \
        ["k21_root", f"k15d_level_{label}"]
    miss = [k for k in need if k not in z]
    if miss:
        return [f"{label}: в журнале нет {miss}"]
    g = {k: np.asarray(z[f"k21_cond_{k}"], np.float64) for k in COND_SHAPES}
    root = np.asarray(z["k21_root"], np.float64)
    lv0 = np.asarray(z[f"k15d_level_{label}"], np.float64)
    if g["Z"].ndim != 4:
        return [f"{label}: Z формы {g['Z'].shape}"]
    C, B = g["Z"].shape[:2]
    p = [f"{label}: {k} формы {g[k].shape}, ожидалось {(C, B) + tail}"
         for k, tail in COND_SHAPES.items() if g[k].shape != (C, B) + tail]
    if root.shape != (C, B, 4, 2):
        p.append(f"{label}: root формы {root.shape}")
    if lv0.shape[:2] != (C, B) or lv0.shape[-1] != 7:
        p.append(f"{label}: журнал команд формы {lv0.shape}")
    if p:
        return p
    bad = [k for k, v in list(g.items()) + [("root", root), ("level", lv0)]
           if not np.isfinite(v).all()]
    if bad:
        return [f"{label}: нечисловые значения в {bad}"]
    f = {k: v.reshape((C * B,) + v.shape[2:]) for k, v in g.items()}
    root = root.reshape((C * B,) + root.shape[2:])
    W0, W1, W2 = p24.weights_from_geometry(root, f["split1"], f["split2"])
    for nm, a, b in (("W0", W0, f["W0"]), ("W1", W1, f["W1"]),
                     ("W2", W2, f["W2"])):
        d = float(np.abs(a - b).max())
        if not d <= GEOM_TOL:
            p.append(f"{label}: {nm} из геометрии ≠ журналу ({d:.2e})")
    mp, lpar = np.arange(8) // 2, np.arange(16) // 2
    z1 = f["C0"][:, mp][:, :, None, :] + f["g1"][..., None, None] * \
        f["delta1"]
    z2 = z1[:, lpar] + f["g2"][..., None, None] * f["y2"][..., :6]
    Z = np.einsum("ntl,nltc->ntc", f["W2"], z2)
    G0 = f["W0"] @ f["C0"]
    G1 = np.einsum("ntj,njtc->ntc", f["W1"], f["g1"][..., None, None] *
                   f["delta1"])
    G2 = np.einsum("ntl,nltc->ntc", f["W2"], f["g2"][..., None, None] *
                   f["y2"][..., :6])
    sc = np.maximum(1.0, np.max(np.stack([
        np.abs(x).reshape(C * B, -1).max(1) for x in (Z, G0, G1, G2)]),
        0))[:, None, None]
    for nm, a, b in (("Z", Z, f["Z"]), ("G0", G0, f["G0"]),
                     ("G1", G1, f["G1"]), ("G2", G2, f["G2"]),
                     ("G0+G1+G2", G0 + G1 + G2, f["Z"])):
        d = float((np.abs(a - b) / sc).max())
        if not d <= tol:
            p.append(f"{label}: {nm} пересборки ≠ журналу ({d:.2e})")
    L, U = np.asarray(bounds["L"]), np.asarray(bounds["U"])
    mid, half = (U + L) / 2, (U - L) / 2
    arm = mid + half * np.tanh(f["Z"] / half)
    lv = lv0.reshape((C * B,) + lv0.shape[2:])
    d = float((np.abs(arm[:, :8] - lv[..., :6]) /
               np.maximum(1.0, np.abs(lv[..., :6]))).max())
    if not d <= tol:
        p.append(f"{label}: tanh-формула ≠ исполненным командам ({d:.2e})")
    gl = np.einsum("ntl,nlt->nt", f["W2"], f["y2"][..., 6])
    if not np.array_equal(np.where(gl[:, :8] > 0, 1.0, -1.0), lv[..., 6]):
        p.append(f"{label}: схват ≠ знаку Σ W2·y2[6]")
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
    want = dict(label=label, arch=arch, train_seed=seed,
                plan_sha1=sha(plan_path_), prep_id=plan["prep_id"],
                bar_revision=plan["bar_revision"], ckpt_sha1=spec["ckpt_sha1"],
                weights_hash=spec["weights_hash"], readout_version=kr.VERSION,
                k21_policy_sha1=full_sha(os.path.join(HERE,
                                                      "k21_main_policy.py")),
                k25_policy_sha1=full_sha(os.path.join(HERE, "k25_policy.py")),
                harness_adapter_sha1=sha(os.path.join(HERE, "k25_harness.py")))
    for k, v in want.items():
        if j.get(k) != v:
            p.append(f"{label}: {k} = {j.get(k)!r}, ожидалось {v!r}")
    p += pp.journal_problems(label, z, plan["bounds"])
    if not p:
        p += cond_problems(z, label, plan["bounds"])
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


def selftest():
    import torch
    import k21_tree as ktm
    TT = ktm.build(torch)
    R = kr.build(torch)
    nn = torch.nn
    g = torch.Generator().manual_seed(3)
    C, B, H = 2, 3, 16
    N = C * B
    iv = TT["root"](torch.randn(N, 4, generator=g) * 2)
    W0 = TT["root_weights"](iv)
    iv1, s1 = TT["split"](iv, torch.randn(N, 4, generator=g))
    W1 = TT["split_weights"](W0, iv, s1)
    iv2, s2 = TT["split"](iv1, torch.randn(N, 8, generator=g))
    W2 = TT["split_weights"](W1, iv1, s2)
    Rd = ktm.build(torch)["Readout"](H, np.zeros(6), np.ones(6))
    for prm in Rd.parameters():
        nn.init.normal_(prm, std=0.3, generator=g)
    cm = R["CondModules"](H)
    nn.init.normal_(cm.A_a.weight, std=0.3, generator=g)
    h = torch.randn(N, 28, H, generator=g)
    bounds = dict(L=[-1.0, -2.0, -0.5, -1.0, -3.0, -1.0],
                  U=[2.0, 1.0, 0.5, 4.0, 3.0, 1.5])
    with torch.no_grad():
        o = R["cond_readout"](Rd, cm, nn.LayerNorm(H), h[:, :4], h[:, 4:12],
                              h[:, 12:28], iv, iv1, iv2, W0, W1, W2,
                              bounds["L"], bounds["U"])
    lab = "parents_cond24_s0"

    def r(x):
        x = x.detach().numpy().astype(np.float32)
        return x.reshape((C, B) + x.shape[1:])
    src = dict(o, W0=W0, W1=W1, W2=W2, split1=s1, split2=s2)
    z = {f"k21_cond_{k}": r(src[k]) for k in COND_SHAPES}
    z["k21_root"] = r(iv)
    z[f"k15d_level_{lab}"] = np.concatenate([
        r(o["arm"])[:, :, :8], np.where(r(o["grip_logit"])[:, :, :8, None]
                                        > 0, 1.0, -1.0)], -1).astype(
                                            np.float32)
    assert cond_problems(z, lab, bounds) == [], cond_problems(z, lab, bounds)
    for key, mut in (("k21_cond_Z", lambda a: a + 0.01),
                     ("k21_cond_g2", lambda a: a * 1.5),
                     ("k21_cond_delta1", lambda a: a + 0.05),
                     ("k21_root", lambda a: a + 0.3)):
        zb = dict(z)
        zb[key] = mut(z[key])
        assert cond_problems(zb, lab, bounds), key
    for key in ("k21_cond_y2", "k21_cond_C0", "k21_cond_W1", "k21_root",
                f"k15d_level_{lab}"):
        for val in (np.nan, np.inf, -np.inf):
            zb = dict(z)
            arr = np.array(z[key], copy=True)
            arr.reshape(-1)[0] = val
            zb[key] = arr
            pb = cond_problems(zb, lab, bounds)
            assert pb and "нечисловые" in pb[0], (key, val, pb)
    zs = dict(z)
    zs["k21_cond_delta1"] = z["k21_cond_delta1"][..., :5]
    assert any("delta1 формы" in x for x in cond_problems(zs, lab, bounds))
    zg = dict(z)
    zg[f"k15d_level_{lab}"] = z[f"k15d_level_{lab}"].copy()
    zg[f"k15d_level_{lab}"][0, 0, 0, 6] *= -1
    assert any("схват" in x for x in cond_problems(zg, lab, bounds))
    assert parse_label("parents_cond24_s1") == ("parents_cond24", 1)
    assert len(set(CODE)) == len(CODE)
    print("самопроверка k25_protocol пройдена: пересборка путей z1 -> z2 -> Z "
          "и G0+G1+G2 по журналу, W из геометрии, tanh и схват; порча Z, "
          "гейта, δ1, геометрии, схвата, NaN/±Inf и неверная форма — отказ")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-25: подготовка и планы")
    ap.add_argument("mode", choices=("prepare", "plan", "check-plan",
                                     "check-block", "smoke-check", "cells",
                                     "info", "selftest"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--block", nargs=5, default=None)
    ap.add_argument("--which", choices=("smoke", "all"), default="all")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    try:
        if a.mode == "prepare":
            return prepare()
        if a.mode == "plan":
            pth = plan_path(a.seed)
            if os.path.exists(pth):
                print(f"  план {pth} уже зафиксирован")
                return 0
            plan = make_plan(a.seed, a.device)
            with open(pth + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=1)
            os.replace(pth + ".tmp", pth)
            print(f"  план {pth}: {plan['labels']}, ячеек {len(plan['cells'])}"
                  f"; источников {sum(len(r['blocks']) for r in plan['references'].values())} блоков")
            return 0
        plan = json.load(open(a.plan, encoding="utf-8"))
        if a.mode == "check-plan":
            prob = check_plan_inputs(plan)
            if not prob:
                p23.verify_refs(plan)
            print("\n".join(prob) if prob else "  входы и источники сверены")
            return 3 if prob else 0
        if a.mode == "info":
            print(plan["snapshot_path"])
            return 0
        if a.mode == "cells":
            for s, t, b in (plan["smoke"] if a.which == "smoke"
                            else plan["cells"]):
                for lab in plan["labels"]:
                    print(f"{lab}:{s}:{t}:{b}")
            return 0
        if a.mode == "check-block":
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
            print(f"  размер журнала smoke: {np.mean(sz) / 2 ** 20:.1f} МБ на "
                  f"блок (NPZ)")
        return 3 if gaps else 0
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
