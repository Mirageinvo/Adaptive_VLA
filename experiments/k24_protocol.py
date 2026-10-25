#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-24: подготовка, планы SR и проверка блоков parents_mr24.

prepare — reports/k24/prepare.json: подготовка K-21 (prep_id, SHA отчётов,
  учитель, ревизия BAR, пределы), покрытие val_sel по задачам, по сидам —
  контроли parents/flat (чекпойнт, SHA, отпечаток весов, отчёт обучения),
  ЗАКРЕПЛЁННЫЙ бюджет main = фактический бюджет parents-контроля того же
  сида и его начальный отпечаток. Seed 1 — после завершения K-23 (иначе
  status=pending); готовые записи при повторном prepare не меняются.
plan --seed S — reports/k24/plan_s<S>.json: метка parents_mr24_s<S>
  (чекпойнт K-24 этого сида), LIBERO-10, задачи 0–9, блоки 30/35/40,
  setup и стартовые файлы плана A; закреплённые источники (SHA каждого
  блока): seed 0 — план A K-21 (fullbar, flat24_s0, tree), K-22
  (parents24_s0), fullbar_cached; seed 1 — план повтора K-23 (flat24_s1,
  parents24_s1), план A (fullbar), fullbar_cached.
check-block — штатные проверки блока + журнал сборки: W0/W1/W2 заново из
  записанной геометрии (|Δ| ≤ 1e-5); по ЗАПИСАННЫМ W — проекторы, G0/G1/G2/Z
  из C0/C1/Y2, тождества и tanh-формула против журнала (отн. 5e-5); схват =
  знак Y2[6].
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
import k24_readout as kr  # noqa: E402

KIND = "k24_plan"
PREPARE = "reports/k24/prepare.json"
BOOT = dict(n=10000, seed=20261025)
MR_TOL = 5e-5
GEOM_TOL = 1e-5
T = 20
CODE = ("k24_protocol.py", "k24_policy.py", "k24_harness.py",
        "k24_analyze.py", "k24_run.sh", "k24_train.py", "k24_model.py",
        "k24_readout.py", "k24_latency.py") + tuple(
            f for f in p23.CODE if f not in ("k23_protocol.py",)) + (
                "k23_protocol.py",)
sha, full_sha = pp.sha, pp.full_sha


def plan_path(seed):
    return f"reports/k24/plan_s{seed}.json"


def run_dir(plan_path_):
    return os.path.join("reports/k24/run", f"p{sha(plan_path_)}")


def block_name(lab, suite, t, b):
    return f"{lab}_{suite}_t{t}_i{b}.json"


def parse_label(label):
    m = re.fullmatch(r"parents_mr24_s(\d+)", str(label))
    if not m:
        raise SystemExit(f"метка {label!r}: нужна parents_mr24_s<seed>")
    return "parents_mr24", int(m.group(1))


# --- источники -------------------------------------------------------------
def ref_k23_repeat(seed=1):
    path = p23.plan_path("repeat", seed)
    plan = json.load(open(path, encoding="utf-8"))
    prob = p23.check_plan_inputs(plan)
    if prob:
        raise SystemExit(f"план повтора K-23 s{seed}: " + "; ".join(prob[:3]))
    run = p23.run_dir(path)
    ok, gaps = p23.check_run(plan, path, run)
    want = len(plan["cells"]) * len(plan["labels"])
    if len(ok) != want or gaps:
        raise SystemExit(f"план повтора K-23 s{seed}: проверено {len(ok)}/"
                         f"{want}, проблем {len(gaps)}")
    return dict(name=f"k23_repeat_s{seed}", path=os.path.abspath(path),
                sha12=sha(path), run_dir=os.path.abspath(run), plan=plan,
                blocks={(lab, s, int(t), int(b)): f
                        for (lab, s, t, b), f in ok.items()})


# локально и явно: verify_refs K-23 находит источник по имени в REFS
p23.REFS["k23_repeat_s1"] = lambda: ref_k23_repeat(1)


def control(arch, seed):
    """Чекпойнт, отчёт обучения и начальный отпечаток контроля."""
    import torch
    if seed == 0:
        ck = dict(parents24="data/k22/runs/parents24",
                  flat24="data/k21/runs/flat24")[arch]
        tr = dict(parents24="reports/k22/train.json",
                  flat24="reports/k21/train.json")[arch]
        src = dict(parents24="K-22", flat24="K-21")[arch]
    else:
        ck = f"data/k23/runs/s{seed}/{arch}"
        tr = f"reports/k23/train_s{seed}.json"
        src = "K-23"
    if not os.path.exists(tr) or not os.path.exists(f"{ck}/best.pt"):
        return None
    rep = json.load(open(tr, encoding="utf-8"))
    if arch not in rep:
        return None
    r = rep[arch]
    if full_sha(f"{ck}/best.pt") != r["best_sha1"]:
        raise SystemExit(f"{ck}/best.pt ≠ отчёту {tr}")
    rs = torch.load(f"{ck}/resume.pt", map_location="cpu",
                    weights_only=False)
    return dict(source=src, ckpt=os.path.abspath(f"{ck}/best.pt"),
                ckpt_sha1=r["best_sha1"], weights_hash=r["best_weights_hash"],
                train_report=os.path.abspath(tr), train_report_sha1=full_sha(tr),
                updates=int(r["updates"]), best=r["best"],
                init_hash=rs["init_hash"], seed=int(rs["config"]["seed"]))


def entry_fp(e):
    """Отпечаток полной записи сида (всё, кроме самого отпечатка)."""
    import hashlib
    body = {k: v for k, v in e.items() if k != "fingerprint"}
    return hashlib.sha1(json.dumps(body, sort_keys=True, default=str)
                        .encode()).hexdigest()


COMMON_KEYS = ("prep_id", "prepare_sha1", "teacher_sha1",
               "teacher_report_sha1", "bar_revision")


def entry_problems(e, common=None):
    """Запись сида не изменена: отпечаток, файлы контролей, общие
    источники."""
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
    if int(e["budget_main"]) != int(e["control_parents"]["updates"]):
        p.append("бюджет ≠ фактическому бюджету parents-контроля")
    if common is not None and e.get("common") != common:
        p.append("общие источники (подготовка, учитель, BAR) изменились")
    return p


def seed_entry(seed, path=None):
    """Готовая запись сида из prepare.json после проверки отпечатка."""
    pr = json.load(open(path or PREPARE, encoding="utf-8"))
    e = (pr.get("seeds") or {}).get(str(seed))
    if not e or e.get("status") != "ready":
        raise SystemExit(f"seed {seed}: подготовка не готова")
    prob = entry_problems(e, {k: pr[k] for k in COMMON_KEYS})
    if prob:
        raise SystemExit(f"seed {seed}: " + "; ".join(prob))
    return e


def val_coverage(task, val_rows, train_rows):
    """Задачи val_sel и задачи обучения без строк в val_sel. Метки задач —
    как в подготовке K-21 (текстовые описания), без приведения типов."""
    task = np.asarray(task)
    vt = sorted({str(x) for x in task[np.asarray(val_rows)]})
    tr = sorted({str(x) for x in task[np.asarray(train_rows)]})
    return vt, [t for t in tr if t not in vt], tr


def prepare():
    import k21_data as d21
    old = json.load(open(PREPARE, encoding="utf-8")) \
        if os.path.exists(PREPARE) else None
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    tj = json.load(open(d21.TEACHER_REPORT, encoding="utf-8"))
    D = d21.load_prepared(require_teacher=False)
    vt, missing, alltasks = val_coverage(D["task"], D["val_rows"],
                                         D["train_rows"])
    rep = dict(created=(old or {}).get("created") or datetime.datetime.now()
               .isoformat(timespec="seconds"),
               prep_id=prep["prep_id"], prepare_sha1=full_sha(d21.REPORT),
               teacher_sha1=tj["out"]["sha1"],
               teacher_report_sha1=full_sha(d21.TEACHER_REPORT),
               bar_revision=prep["bar_pin"]["revision"],
               bounds=prep["bounds_codec"],
               train_rows=int(len(D["train_rows"])),
               val_rows=int(len(D["val_rows"])),
               train_tasks=len(alltasks), val_tasks=len(vt),
               val_tasks_missing=missing,
               readout=dict(version=kr.VERSION, formula=kr.FORMULA,
                            rank_tol=kr.RANK_TOL),
               seeds=dict((old or {}).get("seeds") or {}))
    common = {k: rep[k] for k in COMMON_KEYS}
    if old:
        for k in COMMON_KEYS:
            if old[k] != rep[k]:
                raise SystemExit(f"подготовка K-21 изменилась: {k}")
    for s in (0, 1):
        e = rep["seeds"].get(str(s))
        if e and e.get("status") == "ready":
            # готовая запись не меняется: отпечаток, файлы, источники
            prob = entry_problems(e, common)
            if prob:
                raise SystemExit(f"seed {s}: " + "; ".join(prob))
            continue
        cp, cf = control("parents24", s), control("flat24", s)
        if cp is None or cf is None:
            rep["seeds"][str(s)] = dict(status="pending",
                                        why="контроль ещё не обучен")
            continue
        if cp["seed"] != s or cf["seed"] != s:
            raise SystemExit(f"seed {s}: seed контроля {cp['seed']}/"
                             f"{cf['seed']}")
        e = dict(status="ready", seed=s, budget_main=cp["updates"],
                 control_parents=cp, control_flat=cf, common=common,
                 fixed=datetime.datetime.now().isoformat(timespec="seconds"))
        e["fingerprint"] = entry_fp(e)
        rep["seeds"][str(s)] = e
    rep["code"] = {f: sha(os.path.join(HERE, f)) for f in CODE
                   if os.path.exists(os.path.join(HERE, f))}
    os.makedirs("reports/k24", exist_ok=True)
    json.dump(rep, open(PREPARE + ".tmp", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    os.replace(PREPARE + ".tmp", PREPARE)
    print(f"  подготовка {PREPARE}: prep_id {rep['prep_id']}; val_sel задач "
          f"{rep['val_tasks']}, без проверки {rep['val_tasks_missing']}")
    for s, e in sorted(rep["seeds"].items()):
        print(f"    seed {s}: {e['status']}" + (
            f", бюджет main {e['budget_main']} (контроль "
            f"{e['control_parents']['source']}), init "
            f"{e['control_parents']['init_hash'][:12]}, отпечаток записи "
            f"{e['fingerprint'][:12]}"
            if e["status"] == "ready" else f" ({e.get('why')})"))
    return 0


# --- план -------------------------------------------------------------------
def make_plan(seed, device):
    import torch
    import k21_data as d21
    import k24_train as k24t
    e = seed_entry(seed)
    trp = k24t.report_path(seed)
    tr = json.load(open(trp, encoding="utf-8"))
    if int(tr["updates"]) != int(e["budget_main"]):
        raise SystemExit("обучение K-24 не совпадает с закреплённым бюджетом")
    ck = os.path.join(k24t.outdir(seed), k24t.MODE, "best.pt")
    if full_sha(ck) != tr["best_sha1"]:
        raise SystemExit(f"{ck} ≠ отчёту обучения")
    c = torch.load(ck, map_location="cpu", weights_only=False)
    if c["weights_hash"] != tr["best_weights_hash"] or \
            int(c["config"]["seed"]) != seed or \
            c["config"].get("architecture") != "k24_parents_mr24":
        raise SystemExit("чекпойнт K-24: отпечаток, seed или архитектура")
    if c["config"]["k24"].get("seed_entry_fp") != e["fingerprint"]:
        raise SystemExit("чекпойнт K-24 обучен по другой записи подготовки")
    lab = f"parents_mr24_s{seed}"
    spec = dict(arch="parents_mr24", seed=seed, ckpt=os.path.abspath(ck),
                ckpt_sha1=full_sha(ck), weights_hash=c["weights_hash"],
                best_update=int(c["update"]), val_L=float(c["val_L"]),
                train_report=os.path.abspath(trp),
                train_report_sha1=full_sha(trp), prep_id=c["config"]["prep_id"],
                budget_main=int(e["budget_main"]))
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    rA = p23.ref_k21_main()
    A = rA["plan"]
    refs = {"k21_main_A": p23.pin_ref(rA),
            "fullbar_cached": p23.pin_ref(p23.ref_cached())}
    if seed == 0:
        refs["k22"] = p23.pin_ref(p23.ref_k22())
    else:
        refs[f"k23_repeat_s{seed}"] = p23.pin_ref(ref_k23_repeat(seed))
    desc = {}
    for (l_, s_, t_, b_), f in rA["blocks"].items():
        if l_ == "fullbar":
            desc.setdefault(t_, json.load(open(f, encoding="utf-8")).get(
                "task_description"))
    return dict(
        kind=KIND, name=f"k24_s{seed}", train_seed=seed,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        labels=[lab], policy={lab: "k15f"}, specs={lab: spec},
        seed_entry=e, seed_entry_fp=e["fingerprint"],
        controls=dict(parents=f"parents24_s{seed}", flat=f"flat24_s{seed}"),
        suites=["10"], tasks=list(A["tasks"]), blocks=list(A["blocks"]),
        cells=[["10", t, b] for t in A["tasks"] for b in A["blocks"]],
        smoke=[["10", t, b] for t, b in A["smoke"]], setups={"10": A["setup"]},
        suite_tasks={"10": [desc[t] for t in sorted(desc)]},
        refs=A["refs"], refs_sha1=A["refs_sha1"], device=str(device),
        snapshot_path=snap, prep_id=prep["prep_id"],
        bar_revision=prep["bar_pin"]["revision"], bounds=prep["bounds_codec"],
        n_envs=5, env_seed=101, bootstrap=dict(BOOT),
        readout=dict(version=kr.VERSION, rank_tol=kr.RANK_TOL, tol=MR_TOL),
        references=refs,
        inputs={k: os.path.abspath(v) for k, v in dict(
            prepare=d21.REPORT, teacher=d21.TEACHER_REPORT,
            k24_prepare=PREPARE, train_report=trp).items()},
        inputs_sha1={k: sha(v) for k, v in dict(
            prepare=d21.REPORT, teacher=d21.TEACHER_REPORT,
            k24_prepare=PREPARE, train_report=trp).items()},
        code={f: sha(os.path.join(HERE, f)) for f in CODE
              if os.path.exists(os.path.join(HERE, f))},
        note="разработочный банк 30–44; основное — parents_mr24 − parents24 "
             "того же сида; flat — обязательное вторичное")


def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план K-24"]
    for grp in ("inputs", "refs"):
        for k, f in (plan.get(grp) or {}).items():
            if k == "k24_prepare":
                continue          # дополняется для seed 1; запись сида ниже
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
def weights_from_geometry(root, s1, s2):
    """root [N,4,2], s1 [N,4], s2 [N,8] -> W0, W1, W2 (float64, те же
    формулы k21_tree)."""
    import k21_tree as ktm
    x = np.arange(T, dtype=np.float64) + 0.5

    def sig(u):
        return 1.0 / (1.0 + np.exp(-u))
    b = root[:, 1:, 0]
    S = sig((x[None, :, None] - b[:, None, :]) / ktm.TAU_ROOT)
    Sx = np.concatenate([np.ones_like(S[..., :1]), S,
                         np.zeros_like(S[..., :1])], -1)
    W0 = Sx[..., :-1] - Sx[..., 1:]

    def split_w(w, iv, s):
        width = iv[..., 1] - iv[..., 0]
        g = sig((x[None, :, None] - s[:, None, :]) /
                (ktm.TAU_REL * width)[:, None, :])
        return np.stack([w * (1 - g), w * g], -1).reshape(w.shape[0], T, -1)

    def children(iv, s):
        l_, r_ = iv[..., 0], iv[..., 1]
        ch = np.stack([np.stack([l_, s], -1), np.stack([s, r_], -1)], 2)
        return ch.reshape(iv.shape[0], -1, 2)
    W1 = split_w(W0, root, s1)
    iv2 = children(root, s1)
    W2 = split_w(W1, iv2, s2)
    return W0, W1, W2


def mr_problems(z, label, bounds, tol=MR_TOL):
    need = [f"k21_mr_{k}" for k in ("C0", "C1", "Y2", "Z", "G0", "G1", "G2",
                                    "W0", "W1", "W2", "split1", "split2")] + \
        ["k21_root", f"k15d_level_{label}"]
    miss = [k for k in need if k not in z]
    if miss:
        return [f"{label}: в журнале нет {miss}"]
    g = {k: np.asarray(z[f"k21_mr_{k}"], np.float64) for k in (
        "C0", "C1", "Y2", "Z", "G0", "G1", "G2", "W0", "W1", "W2", "split1",
        "split2")}
    root = np.asarray(z["k21_root"], np.float64)
    lv0 = np.asarray(z[f"k15d_level_{label}"], np.float64)
    if g["Z"].ndim != 4:
        return [f"{label}: Z формы {g['Z'].shape}"]
    C, B = g["Z"].shape[:2]
    want = dict(C0=(4, 6), C1=(8, 6), Y2=(T, 7), Z=(T, 6), G0=(T, 6),
                G1=(T, 6), G2=(T, 6), W0=(T, 4), W1=(T, 8), W2=(T, 16),
                split1=(4,), split2=(8,))
    p = []
    for k, tail in want.items():
        if g[k].shape != (C, B) + tail:
            p.append(f"{label}: {k} формы {g[k].shape}, ожидалось "
                     f"{(C, B) + tail}")
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
    flat = {k: v.reshape((C * B,) + v.shape[2:]) for k, v in g.items()}
    root = root.reshape((C * B,) + root.shape[2:])
    W0, W1, W2 = weights_from_geometry(root, flat["split1"], flat["split2"])
    for nm, a, b in (("W0", W0, flat["W0"]), ("W1", W1, flat["W1"]),
                     ("W2", W2, flat["W2"])):
        d = float(np.abs(a - b).max())
        if not d <= GEOM_TOL:
            p.append(f"{label}: {nm} из геометрии ≠ журналу ({d:.2e})")
    Q0 = np.linalg.qr(flat["W0"], mode="reduced")[0]
    Q1 = np.linalg.qr(flat["W1"], mode="reduced")[0]

    def proj(Q, X):
        return Q @ (np.swapaxes(Q, -1, -2) @ X)
    G0 = flat["W0"] @ flat["C0"]
    G1r = flat["W1"] @ flat["C1"]
    G1 = G1r - proj(Q0, G1r)
    G2 = flat["Y2"][..., :6] - proj(Q1, flat["Y2"][..., :6])
    Z = G0 + G1 + G2
    sc = np.maximum(1.0, np.max(np.abs(np.stack([
        np.abs(x).reshape(C * B, -1).max(1) for x in (Z, G0, G1, G2)])), 0))
    sc = sc[:, None, None]
    for nm, a, b in (("Z", Z, flat["Z"]), ("G0", G0, flat["G0"]),
                     ("G1", G1, flat["G1"]), ("G2", G2, flat["G2"])):
        d = float((np.abs(a - b) / sc).max())
        if not d <= tol:
            p.append(f"{label}: {nm} пересборки ≠ журналу ({d:.2e})")
    for nm, a, b in (("P0Z", proj(Q0, flat["Z"]), flat["G0"]),
                     ("P1Z", proj(Q1, flat["Z"]), flat["G0"] + flat["G1"])):
        d = float((np.abs(a - b) / sc).max())
        if not d <= tol:
            p.append(f"{label}: тождество {nm} ({d:.2e})")
    L, U = np.asarray(bounds["L"]), np.asarray(bounds["U"])
    mid, half = (U + L) / 2, (U - L) / 2
    arm = mid + half * np.tanh(flat["Z"] / half)
    lv = lv0.reshape((C * B,) + lv0.shape[2:])
    d = float((np.abs(arm[:, :8] - lv[..., :6]) /
               np.maximum(1.0, np.abs(lv[..., :6]))).max())
    if not d <= tol:
        p.append(f"{label}: tanh-формула ≠ исполненным командам ({d:.2e})")
    gr = np.where(flat["Y2"][:, :8, 6] > 0, 1.0, -1.0)
    if not np.array_equal(gr, lv[..., 6]):
        p.append(f"{label}: схват ≠ знаку Y2[6]")
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
                weights_hash=spec["weights_hash"],
                readout_version=kr.VERSION, rank_tol=kr.RANK_TOL,
                k21_policy_sha1=full_sha(os.path.join(HERE,
                                                      "k21_main_policy.py")),
                k24_policy_sha1=full_sha(os.path.join(HERE, "k24_policy.py")),
                harness_adapter_sha1=sha(os.path.join(HERE, "k24_harness.py")))
    for k, v in want.items():
        if j.get(k) != v:
            p.append(f"{label}: {k} = {j.get(k)!r}, ожидалось {v!r}")
    p += pp.journal_problems(label, z, plan["bounds"])
    if not p:
        p += mr_problems(z, label, plan["bounds"])
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
    g = torch.Generator().manual_seed(3)
    C, B = 2, 3
    lg = torch.randn(C * B, 4, generator=g) * 2
    v1 = torch.randn(C * B, 4, generator=g)
    v2 = torch.randn(C * B, 8, generator=g)
    iv = TT["root"](lg)
    W0 = TT["root_weights"](iv)
    iv2, s1 = TT["split"](iv, v1)
    W1 = TT["split_weights"](W0, iv, s1)
    _iv3, s2 = TT["split"](iv2, v2)
    W2 = TT["split_weights"](W1, iv2, s2)
    C0 = torch.randn(C * B, 4, 6, generator=g)
    C1 = torch.randn(C * B, 8, 6, generator=g)
    Y2 = torch.randn(C * B, T, 7, generator=g)
    bounds = dict(L=[-1.0, -2.0, -0.5, -1.0, -3.0, -1.0],
                  U=[2.0, 1.0, 0.5, 4.0, 3.0, 1.5])
    o = R["assemble"](W0, W1, C0, C1, Y2, bounds["L"], bounds["U"])
    lab = "parents_mr24_s0"

    def r(x):
        x = x.detach().numpy().astype(np.float32)
        return x.reshape((C, B) + x.shape[1:])
    z = {f"k21_mr_{k}": r(v) for k, v in dict(
        C0=C0, C1=C1, Y2=Y2, Z=o["Z"], G0=o["G0"], G1=o["G1"], G2=o["G2"],
        W0=W0, W1=W1, W2=W2, split1=s1, split2=s2).items()}
    z["k21_root"] = r(iv)
    lv = np.concatenate([r(o["arm"])[:, :, :8], np.where(
        r(Y2)[:, :, :8, 6:7] > 0, 1.0, -1.0)], -1).astype(np.float32)
    z[f"k15d_level_{lab}"] = lv
    assert mr_problems(z, lab, bounds) == [], mr_problems(z, lab, bounds)
    z2 = dict(z)
    z2["k21_mr_Z"] = z["k21_mr_Z"] + 0.01
    assert any("Z" in x for x in mr_problems(z2, lab, bounds))
    z3 = dict(z)
    z3["k21_root"] = z["k21_root"] + 0.3
    assert any("W0" in x for x in mr_problems(z3, lab, bounds))
    z4 = dict(z)
    z4[f"k15d_level_{lab}"] = lv.copy()
    z4[f"k15d_level_{lab}"][0, 0, 0, 6] *= -1
    assert any("схват" in x for x in mr_problems(z4, lab, bounds))
    for key in ("k21_mr_Z", "k21_mr_G0", "k21_mr_C0", "k21_mr_Y2",
                "k21_mr_W1", "k21_mr_split2", "k21_root",
                f"k15d_level_{lab}"):
        for val in (np.nan, np.inf, -np.inf):
            zb = dict(z)
            arr = np.array(z[key], copy=True)
            arr.reshape(-1)[0] = val
            zb[key] = arr
            pb = mr_problems(zb, lab, bounds)
            assert pb and "нечисловые" in pb[0], (key, val, pb)
    zs = dict(z)
    zs["k21_mr_C1"] = z["k21_mr_C1"][..., :5]
    assert any("C1 формы" in x for x in mr_problems(zs, lab, bounds))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ck, trr = os.path.join(td, "c.pt"), os.path.join(td, "t.json")
        open(ck, "wb").write(b"w")
        open(trr, "w").write("{}")
        ctl = dict(ckpt=ck, ckpt_sha1=full_sha(ck), train_report=trr,
                   train_report_sha1=full_sha(trr), updates=8000,
                   init_hash="h0")
        common = {k: "x" for k in COMMON_KEYS}
        e = dict(status="ready", seed=0, budget_main=8000,
                 control_parents=dict(ctl), control_flat=dict(ctl),
                 common=common)
        e["fingerprint"] = entry_fp(e)
        assert entry_problems(e, common) == []
        for mut in (lambda x: x["control_parents"].update(ckpt="/другой.pt"),
                    lambda x: x["control_parents"].update(init_hash="h1"),
                    lambda x: x["control_flat"].update(ckpt_sha1="0" * 40),
                    lambda x: x.update(budget_main=16000),
                    lambda x: x["common"].update(prep_id="y")):
            e2 = json.loads(json.dumps(e))
            mut(e2)
            assert entry_problems(e2, common), e2
        open(ck, "wb").write(b"v")
        assert any("чекпойнт" in x for x in entry_problems(e, common))
    tk = np.array(["put the mug", "open the drawer", "put the mug", "x"])
    vt, miss, tr = val_coverage(tk, [0, 1], [0, 1, 2, 3])
    assert vt == ["open the drawer", "put the mug"] and miss == ["x"] and \
        len(tr) == 3
    assert parse_label("parents_mr24_s1") == ("parents_mr24", 1)
    assert len(set(CODE)) == len(CODE)
    print("самопроверка k24_protocol пройдена: W0/W1/W2 из геометрии, "
          "пересборка Z/G по записанным W, тождества, tanh и схват против "
          "журнала; порча Z, геометрии и схвата, NaN/±Inf в любом массиве "
          "и неверная форма — отказ; отпечаток записи сида ловит смену "
          "чекпойнта, init_hash, SHA, бюджета, источников и файла; "
          "покрытие val_sel по текстовым меткам задач")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-24: подготовка и планы")
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
            print(f"  план {pth}: {plan['labels']}, ячеек {len(plan['cells'])};"
                  f" источников {sum(len(r['blocks']) for r in plan['references'].values())} блоков")
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
        return 3 if gaps else 0
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
