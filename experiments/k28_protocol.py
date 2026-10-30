#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-28: источники, планы SR и проверка блоков parents_goal_fb24.

prepare — reports/k28/sources.json: общие источники (подготовка K-21,
  подготовка K-28 и её k28_prep_id, ревизия BAR, конфигурация контроллера);
  по сидам 0 и 1 — замороженная база: исторический parents24 (K-22 / K-23),
  чекпойнт, SHA, отпечаток весов, отчёт обучения; отпечаток записи сида.
  Готовые записи не меняются.
plan --seed S --kind legacy — plan_s<S>_legacy.json: метка
  parents_goal_fb24legacy_s<S> (новый пошаговый цикл, только номинальный
  чанк), smoke: задачи 0 и 8 блока 30. legacycheck сверяет с закреплёнными
  блоками исторического parents24 сида: setup, сид, начальные хеши, исходы,
  шаги, action_sha1 каждого эпизода.
plan --seed S --kind control — plan_s<S>_control.json: та же legacy-метка на
  всех 30 блоках — свежий контроль parents24 ТОЛЬКО при неподтверждённом
  переиспользовании (начальные хеши совпали, действия — нет) и до основного
  SR; закрепляется в основном плане.
plan --seed S --kind main — plan_s<S>.json: метка parents_goal_fb24_s<S>,
  LIBERO-10, задачи 0–9, блоки 30/35/40, smoke 0/8 блока 30; чекпойнт K-28
  (best по val_sel), источник, контроль (исторический или свежий),
  обязательные источники: план A K-21 (fullbar, flat24_s0), fullbar cached,
  K-22 (s0) или повтор K-23 (s1); описательные — K-24/K-25/K-26/K-27.
check-block — validate_block (setup = SETUP K-9h + SHA k28_rollout_core,
  снимок BAR) + журнал K-28:
  все режимы: формы, конечность, plan_id подряд, номинал в [L, U], схват
  ±1; legacy — исполненные действия = масштаб номинального чанка;
  fb — шаги (plan, j) = (n // 8, n % 8), маска активности = эпизоды до
  done_step, a0 и схват шага = чанк плана, |Δa| ≤ κ, a = clip(a0 + Δa),
  неактивные = номинал без поправки, X_ref(0) = поза состояния в момент
  плана, исполненные действия = масштаб журнала шага;
  в smoke fb — пересчёт на CPU: ошибки (numpy float64, независимая
  геометрия) и весь контроллер (Runtime из чекпойнта) по сохранённым
  планам, свежим состояниям и маскам.
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k21_protocol as pp  # noqa: E402
import k23_protocol as p23  # noqa: E402
import k24_protocol as p24  # noqa: E402
import k25_protocol as p25  # noqa: E402  (k24_s*)
import k26_protocol as p26  # noqa: E402  (k25_s*, verify_optional)
import k28_control as kc  # noqa: E402
import k28_policy as kp  # noqa: E402

KIND = "k28_plan"
PREPARE = "reports/k28/sources.json"
BOOT = dict(n=10000, seed=20261030)
COMMON_KEYS = ("prep_id", "prepare_sha1", "bar_revision", "k28_prep_id",
               "k28_prepare_sha1", "control")
entry_fp = p24.entry_fp
H = 8
DBG_TOL = 1e-5
E_TOL = 2e-3
CODE = ("k28_protocol.py", "k28_policy.py", "k28_harness.py",
        "k28_rollout_core.py", "k28_rollout.sh", "k28_run.sh",
        "k28_train.py", "k28_model.py", "k28_control.py", "k28_geometry.py",
        "k28_data.py", "k22_model.py", "k21_model.py", "k21_tree.py",
        "k21_data.py", "k9h_multiarm_gate.py", "k15g_local_probe.py",
        "k15h_protocol.py", "k21_protocol.py")
sha, full_sha = pp.sha, pp.full_sha


def plan_path(seed, kind="main"):
    return f"reports/k28/plan_s{seed}.json" if kind == "main" else \
        f"reports/k28/plan_s{seed}_{kind}.json"


def legacy_report(seed):
    return f"reports/k28/legacycheck_s{seed}.json"


def run_dir(plan_path_):
    return os.path.join("reports/k28/run", f"p{sha(plan_path_)}")


def block_name(lab, suite, t, b):
    return f"{lab}_{suite}_t{t}_i{b}.json"


# --- необязательные описательные источники ----------------------------------------
def _ref(name, mod, path):
    plan = json.load(open(path, encoding="utf-8"))
    prob = mod.check_plan_inputs(plan)
    if prob:
        raise SystemExit(f"план {name}: " + "; ".join(prob[:3]))
    run = mod.run_dir(path)
    ok, gaps = mod.check_run(plan, path, run)
    want = len(plan["cells"]) * len(plan["labels"])
    if len(ok) != want or gaps:
        raise SystemExit(f"план {name}: проверено {len(ok)}/{want}, проблем "
                         f"{len(gaps)}")
    return dict(name=name, path=os.path.abspath(path), sha12=sha(path),
                run_dir=os.path.abspath(run), plan=plan,
                blocks={(lab, s, int(t), int(b)): f
                        for (lab, s, t, b), f in ok.items()})


def ref_k26(seed, mode):
    return _ref(f"k26_{mode}_s{seed}", p26, p26.plan_path(seed, mode))


def ref_k27(seed):
    import k27_protocol as p27
    return _ref(f"k27_s{seed}", p27, p27.plan_path(seed))


for _s in (0, 1):
    for _m in p26.MODES:
        p23.REFS[f"k26_{_m}_s{_s}"] = (lambda s=_s, m=_m: ref_k26(s, m))
    p23.REFS[f"k27_s{_s}"] = (lambda s=_s: ref_k27(s))


def optional_refs(seed):
    out = {}
    fns = [(f"k24_s{seed}", lambda: p25.ref_k24(seed)),
           (f"k25_s{seed}", lambda: p26.ref_k25(seed)),
           (f"k27_s{seed}", lambda: ref_k27(seed))]
    fns += [(f"k26_{m}_s{seed}", (lambda m=m: ref_k26(seed, m)))
            for m in p26.MODES]
    for name, fn in fns:
        try:
            out[name] = dict(status="pinned", ref=p23.pin_ref(fn()))
        except (SystemExit, OSError, KeyError, ValueError,
                ImportError) as ex:
            out[name] = dict(status="не выполнено", why="нет полного "
                             "проверенного набора: " + (
                                 str(ex.code) if isinstance(ex, SystemExit)
                                 else repr(ex)))
    return out


# --- подготовка ------------------------------------------------------------------
def common_now():
    import k21_data as d21
    import k28_data as kd
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    if not os.path.exists(kd.REPORT):
        raise SystemExit("нет подготовки K-28: k28_data.py prepare")
    k = json.load(open(kd.REPORT, encoding="utf-8"))
    if k["k21_prep_id"] != prep["prep_id"]:
        raise SystemExit("подготовка K-28 от другой подготовки K-21")
    return dict(prep_id=prep["prep_id"], prepare_sha1=full_sha(d21.REPORT),
                bar_revision=prep["bar_pin"]["revision"],
                k28_prep_id=k["k28_prep_id"],
                k28_prepare_sha1=full_sha(kd.REPORT), control=kc.CONFIG)


def entry_problems(e, common=None):
    p = []
    if entry_fp(e) != e.get("fingerprint"):
        p.append("отпечаток записи сида не совпадает с содержимым")
    s = e["source"]
    if not os.path.exists(s["ckpt"]) or full_sha(s["ckpt"]) != s["ckpt_sha1"]:
        p.append("источник: чекпойнт изменён или отсутствует")
    if not os.path.exists(s["train_report"]) or \
            full_sha(s["train_report"]) != s["train_report_sha1"]:
        p.append("источник: отчёт обучения изменён")
    if common is not None and e.get("common") != common:
        p.append("общие источники изменились")
    return p


def seed_entry(seed, path=None):
    pr = json.load(open(path or PREPARE, encoding="utf-8"))
    e = (pr.get("seeds") or {}).get(str(seed))
    if not e or e.get("status") != "ready":
        raise SystemExit(f"seed {seed}: подготовка K-28 не готова")
    prob = entry_problems(e, {k: pr[k] for k in COMMON_KEYS})
    if prob:
        raise SystemExit(f"seed {seed}: " + "; ".join(prob))
    return e


def prepare():
    old = json.load(open(PREPARE, encoding="utf-8")) \
        if os.path.exists(PREPARE) else None
    common = common_now()
    if old:
        for k in COMMON_KEYS:
            if old[k] != common[k]:
                raise SystemExit(f"общий источник изменился после "
                                 f"фиксации: {k}")
    rep = dict(created=(old or {}).get("created") or datetime.datetime.now()
               .isoformat(timespec="seconds"), **common,
               seeds=dict((old or {}).get("seeds") or {}))
    for s in (0, 1):
        e = rep["seeds"].get(str(s))
        if e and e.get("status") == "ready":
            prob = entry_problems(e, common)
            if prob:
                raise SystemExit(f"seed {s}: " + "; ".join(prob))
            continue
        c = p24.control("parents24", s)
        if c is None or c["seed"] != s:
            raise SystemExit(f"seed {s}: нет исторического parents24")
        e = dict(status="ready", seed=s,
                 source={k: c[k] for k in ("source", "ckpt", "ckpt_sha1",
                                           "weights_hash", "train_report",
                                           "train_report_sha1", "updates",
                                           "seed")},
                 common=common,
                 fixed=datetime.datetime.now().isoformat(timespec="seconds"))
        e["fingerprint"] = entry_fp(e)
        rep["seeds"][str(s)] = e
    os.makedirs("reports/k28", exist_ok=True)
    json.dump(rep, open(PREPARE + ".tmp", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    os.replace(PREPARE + ".tmp", PREPARE)
    print(f"  источники {PREPARE}: prep_id {rep['prep_id']}, k28_prep_id "
          f"{rep['k28_prep_id']}, контроллер {kc.VERSION}")
    for s, e in sorted(rep["seeds"].items()):
        print(f"    seed {s}: база {e['source']['source']} "
              f"{e['source']['ckpt']} (веса {e['source']['weights_hash'][:12]})"
              f"; отпечаток {e['fingerprint'][:12]}")
    return 0


# --- контроль --------------------------------------------------------------------
def control_status(seed):
    """-> ("historical" | "fresh", отчёт legacycheck) или отказ."""
    p = legacy_report(seed)
    if not os.path.exists(p):
        raise SystemExit(f"seed {seed}: legacycheck не выполнен")
    r = json.load(open(p, encoding="utf-8"))
    lp = plan_path(seed, "legacy")
    if r.get("plan_sha1") != full_sha(lp):
        raise SystemExit(f"{p}: от другого legacy-плана")
    if r["status"] == "identical":
        return "historical", r
    if r["status"] == "control_needed":
        return "fresh", r
    raise SystemExit(f"{p}: статус {r['status']}")


# --- планы ------------------------------------------------------------------------
def main_spec(seed, e):
    import torch
    import k28_train as k28t
    trp = k28t.report_path(seed)
    tr = json.load(open(trp, encoding="utf-8"))
    if int(tr["warm"]) != k28t.WARM or int(tr["main"]) != k28t.MAIN:
        raise SystemExit(f"обучение K-28 seed {seed}: бюджет {tr['warm']}+"
                         f"{tr['main']}")
    ck = os.path.join(k28t.outdir(seed), "best.pt")
    if full_sha(ck) != tr["best_sha1"]:
        raise SystemExit(f"{ck} ≠ отчёту обучения")
    c = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = c["config"]
    if c["new_hash"] != tr["best_new_hash"] or int(cfg["seed"]) != seed or \
            cfg["mode"] != "parents_goal_fb24" or \
            cfg["seed_entry_fp"] != e["fingerprint"] or \
            c["base_hash"] != e["source"]["weights_hash"] or \
            cfg["control"] != kc.CONFIG:
        raise SystemExit("чекпойнт K-28: отпечаток, seed, режим, запись "
                         "подготовки, база или контроллер")
    u = int(c["update"])
    if not (250 <= u <= k28t.MAIN and u % k28t.EVAL_EVERY == 0):
        raise SystemExit(f"best не из основных проверок 250…2000: {u}")
    return dict(mode="fb", seed=seed, source=e["source"],
                ckpt=os.path.abspath(ck), ckpt_sha1=full_sha(ck),
                new_hash=c["new_hash"], best_update=u, crit=float(c["crit"]),
                sigma_p=cfg["sigma_p"], sigma_R=cfg["sigma_R"],
                train_report=os.path.abspath(trp),
                train_report_sha1=full_sha(trp)), dict(train_report=trp)


def make_plan(seed, kind, device):
    import k21_data as d21
    import k28_rollout_core as core
    e = seed_entry(seed)
    src = e["source"]
    inp = dict(prepare=d21.REPORT, k28_sources=PREPARE)
    control = None
    if kind in ("legacy", "control"):
        lab = f"parents_goal_fb24legacy_s{seed}"
        spec = dict(mode="legacy", seed=seed, source=src,
                    ckpt_sha1=src["ckpt_sha1"])
        if kind == "control":
            st, r = control_status(seed)
            if st != "fresh":
                raise SystemExit("свежий контроль не нужен: legacy совпал с "
                                 "историческим parents24")
    else:
        lab = f"parents_goal_fb24_s{seed}"
        spec, extra = main_spec(seed, e)
        inp.update(extra)
        st, r = control_status(seed)
        control = dict(status=st, legacycheck=os.path.abspath(
            legacy_report(seed)), legacycheck_sha1=full_sha(
                legacy_report(seed)))
        if st == "fresh":
            cp = plan_path(seed, "control")
            cpl = json.load(open(cp, encoding="utf-8"))
            ok, gaps = check_run(cpl, cp, run_dir(cp))
            if gaps or len(ok) != len(cpl["cells"]):
                raise SystemExit(f"свежий контроль seed {seed} неполон: "
                                 f"{len(ok)}/{len(cpl['cells'])}")
            control.update(plan=os.path.abspath(cp), plan_sha1=full_sha(cp),
                           label=cpl["labels"][0],
                           blocks={f"{k[0]}|{k[1]}|{k[2]}|{k[3]}":
                                   full_sha(f) for k, f in ok.items()})
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    rA = p23.ref_k21_main()
    A = rA["plan"]
    refs = {"k21_main_A": p23.pin_ref(rA),
            "fullbar_cached": p23.pin_ref(p23.ref_cached())}
    if seed == 0:
        refs["k22"] = p23.pin_ref(p23.ref_k22())
    else:
        refs[f"k23_repeat_s{seed}"] = p23.pin_ref(p24.ref_k23_repeat(seed))
    desc = {}
    for (l_, s_, t_, b_), f in rA["blocks"].items():
        if l_ == "fullbar":
            desc.setdefault(t_, json.load(open(f, encoding="utf-8")).get(
                "task_description"))
    smoke = [["10", t, b] for t, b in A["smoke"]]
    cells = smoke if kind == "legacy" else \
        [["10", t, b] for t in A["tasks"] for b in A["blocks"]]
    if any(int(c[2]) + 4 >= 45 for c in cells):
        raise SystemExit("блоки задевают резерв 45–49")
    return dict(
        kind=KIND, plan_kind=kind, name=f"k28_s{seed}_{kind}",
        train_seed=seed,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        labels=[lab], specs={lab: spec}, seed_entry=e,
        seed_entry_fp=e["fingerprint"], control=control,
        controls=dict(primary=f"parents24_s{seed}",
                      engineering=["fullbar", "fullbar_cached"],
                      descriptive=[f"flat24_s{seed}"],
                      optional_descriptive=[
                          f"parents_mr24_s{seed}", f"parents_cond24_s{seed}",
                          f"parents_attn24_s{seed}", f"flat_attn24_s{seed}",
                          f"parents_mem24_s{seed}"]),
        suites=["10"], tasks=list(A["tasks"]), blocks=list(A["blocks"]),
        cells=cells, smoke=smoke,
        setups={"10": dict(core.expected_setup(), ckpt=snap)},
        suite_tasks={"10": [desc[t] for t in sorted(desc)]},
        refs=A["refs"], refs_sha1=A["refs_sha1"], device=str(device),
        snapshot_path=snap, prep_id=prep["prep_id"],
        bar_revision=prep["bar_pin"]["revision"], bounds=prep["bounds_codec"],
        n_envs=5, env_seed=101, bootstrap=dict(BOOT),
        control_config=kc.CONFIG, heavy_stride=H, control_stride=1,
        horizon=20, references=refs,
        optional_references=optional_refs(seed) if kind == "main" else {},
        inputs={k: os.path.abspath(v) for k, v in inp.items()},
        inputs_sha1={k: sha(v) for k, v in inp.items()},
        code={f: sha(os.path.join(HERE, f)) for f in CODE
              if os.path.exists(os.path.join(HERE, f))},
        decision_policy="два вывода: project_result (SR к BAR, задержка "
                        "T_cycle, изменение к parents24) и mechanism "
                        "(parents_goal_fb24 − parents24 того же сида, "
                        "McNemar, Holm по двум сидам, совместный бутстрэп); "
                        "SR и p не влияют на код завершения",
        note="разработочный банк 30–44 (разведочно); надстройка над "
             "замороженным parents24")


def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план K-28"]
    for grp in ("inputs", "refs"):
        for k, f in (plan.get(grp) or {}).items():
            if k == "k28_sources":
                continue          # запись сида проверяется ниже
            if not os.path.exists(f) or sha(f) != plan[f"{grp}_sha1"][k]:
                p.append(f"{grp} {k} изменён или отсутствует ({f})")
    try:
        e = seed_entry(plan["train_seed"])
        if e["fingerprint"] != plan["seed_entry_fp"] or \
                entry_fp(plan["seed_entry"]) != plan["seed_entry_fp"]:
            p.append("запись подготовки сида ≠ закреплённой в плане")
    except SystemExit as ex:
        p.append(f"запись подготовки сида: {ex.code}")
    if plan.get("control_config") != kc.CONFIG:
        p.append("конфигурация контроллера ≠ плану")
    for lab, s in plan["specs"].items():
        fs = [(s["source"]["ckpt"], s["source"]["ckpt_sha1"])]
        if s["mode"] == "fb":
            fs.append((s["ckpt"], s["ckpt_sha1"]))
        for f, h in fs:
            if not os.path.exists(f) or full_sha(f) != h:
                p.append(f"чекпойнт {lab} изменён или отсутствует ({f})")
    c = plan.get("control")
    if c:
        for f, h in ((c["legacycheck"], c["legacycheck_sha1"]),
                     (c.get("plan"), c.get("plan_sha1"))):
            if f and (not os.path.exists(f) or full_sha(f) != h):
                p.append(f"контроль: {f} изменён или отсутствует")
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    return p


# --- журнал K-28 -------------------------------------------------------------------
def np_hat(w):
    z = np.zeros(w.shape[:-1] + (3, 3))
    z[..., 0, 1], z[..., 0, 2] = -w[..., 2], w[..., 1]
    z[..., 1, 0], z[..., 1, 2] = w[..., 2], -w[..., 0]
    z[..., 2, 0], z[..., 2, 1] = -w[..., 1], w[..., 0]
    return z


def np_exp(w):
    th = np.linalg.norm(w, axis=-1)[..., None, None]
    K = np_hat(w)
    small = th < 1e-8
    ths = np.where(small, 1.0, th)
    A = np.where(small, 1.0, np.sin(ths) / ths)
    B = np.where(small, 0.5, (1 - np.cos(ths)) / ths ** 2)
    return np.eye(3) + A * K + B * (K @ K)


def np_log(R):
    c = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(th)
    f = np.where(th < 1e-6, 0.5, th / (2 * np.where(th < 1e-6, 1.0, s)))
    return v * f[..., None]


def np_errors(ref_p, ref_R, s8, sigma_p, sigma_R):
    """Независимая (numpy float64) ошибка шага: e_p, e_R до обрезки."""
    p = s8[..., :3]
    R = np_exp(s8[..., 3:6])
    ep = (ref_p - p) / np.asarray(sigma_p, np.float64)
    eR = np_log(ref_R @ np.swapaxes(R, -1, -2)) / float(sigma_R)
    return np.concatenate([ep, eR], -1)


LEG_KEYS = ("plan_a", "plan_grip_logit", "plan_id")
FB_KEYS = ("plan_nodes_t", "plan_nodes_p", "plan_nodes_R", "plan_ref_p",
           "plan_ref_R", "plan_c", "step_plan", "step_j", "step_state8",
           "step_e", "step_e_norm", "step_r", "step_da", "step_a0", "step_a",
           "step_active")


def journal_problems(z, label, mode, bounds, scale, spec=None,
                     recompute=False):
    """Журнал K-28 блока и связь с исполненными действиями."""
    need = [f"k28_{k}" for k in LEG_KEYS + (FB_KEYS if mode == "fb" else ())]
    miss = [k for k in need + ["actions", "done_step"] if k not in z]
    if miss:
        return [f"{label}: в журнале нет {miss}"]
    g = {k[4:]: np.asarray(z[k]) for k in need}
    A = np.asarray(z["actions"])
    ds = np.asarray(z["done_step"])
    N, B = A.shape[:2]
    C = len(g["plan_a"])
    p = []
    if mode == "legacy" and any(f"k28_{k}" in z for k in FB_KEYS):
        p.append(f"{label}: legacy с журналом контроллера")
    if g["plan_a"].shape != (C, B, 20, 7) or C != -(-N // H):
        return [f"{label}: plan_a формы {g['plan_a'].shape}, шагов {N}"]
    if g["plan_id"].tolist() != list(range(C)):
        p.append(f"{label}: plan_id не подряд")
    bad = [k for k, v in g.items() if v.dtype.kind == "f" and
           not np.isfinite(v).all()]
    if bad:
        return p + [f"{label}: нечисловые значения в {bad}"]
    L, U = np.asarray(bounds["L"])[:6], np.asarray(bounds["U"])[:6]
    a0 = g["plan_a"]
    if (a0[..., :6] < L - 1e-4).any() or (a0[..., :6] > U + 1e-4).any():
        p.append(f"{label}: номинал вне пределов кодека")
    if not np.isin(a0[..., 6], (-1.0, 1.0)).all():
        p.append(f"{label}: схват не ±1")
    if not np.array_equal(a0[..., 6], np.where(g["plan_grip_logit"] > 0,
                                               1.0, -1.0)):
        p.append(f"{label}: схват ≠ знаку логита")
    n = np.arange(N)
    alive = (ds[None, :] < 0) | (n[:, None] <= ds[None, :])
    if mode == "legacy":
        cmd = a0[n // H, :, n % H]                         # [N, B, 7]
    else:
        want = dict(plan_nodes_t=(C, B, 17), plan_nodes_p=(C, B, 17, 3),
                    plan_nodes_R=(C, B, 17, 3, 3), plan_ref_p=(C, B, 21, 3),
                    plan_ref_R=(C, B, 21, 3, 3), plan_c=(C, B, 20, kc.CTX),
                    step_plan=(N,), step_j=(N,), step_state8=(N, B, 8),
                    step_e=(N, B, 6), step_e_norm=(N, B), step_r=(N, B, 6),
                    step_da=(N, B, 6), step_a0=(N, B, 6), step_a=(N, B, 7),
                    step_active=(N, B))
        bad = [f"{k} {g[k].shape}≠{s}" for k, s in want.items()
               if g[k].shape != s]
        if bad:
            return p + [f"{label}: формы {bad[:3]}"]
        if g["step_plan"].tolist() != (n // H).tolist() or \
                g["step_j"].tolist() != (n % H).tolist():
            p.append(f"{label}: шаги (plan, j) не по расписанию 8")
        if not np.array_equal(g["step_active"].astype(bool), alive):
            p.append(f"{label}: маска активности ≠ эпизодам до done_step")
        nom = a0[n // H, :, n % H]
        if not np.array_equal(g["step_a0"], nom[..., :6]):
            p.append(f"{label}: a0 шага ≠ номинальному чанку плана")
        a = g["step_a"]
        if not np.array_equal(a[..., 6], nom[..., 6]):
            p.append(f"{label}: схват изменён контроллером")
        k_ = kc.KAPPA_FRAC * (U - L) / 2
        da = g["step_da"].astype(np.float64)
        if (np.abs(da) > k_ * (1 + 1e-5) + 1e-7).any():
            p.append(f"{label}: |Δa| > κ")
        if (a[..., :6] < L - 1e-5).any() or (a[..., :6] > U + 1e-5).any():
            p.append(f"{label}: команда вне [L, U]")
        clip = np.clip(g["step_a0"].astype(np.float64) + da, L, U)
        if np.abs(clip - a[..., :6]).max() > 1e-5:
            p.append(f"{label}: a ≠ clip(a0 + Δa)")
        off = ~alive
        if (g["step_da"][off] != 0).any() or \
                not np.array_equal(a[off][:, :6], nom[off][:, :6]):
            p.append(f"{label}: неактивная среда получила поправку")
        first = g["step_state8"][n[n % H == 0]].astype(np.float64)
        if np.abs(g["plan_ref_p"][:, :, 0] - first[..., :3]).max() > 1e-5:
            p.append(f"{label}: X_ref(0) ≠ позе состояния в момент плана")
        cmd = a
    exp = np.array(cmd, np.float32, copy=True)
    exp[..., :-1] = exp[..., :-1] * scale[:-1]
    exp[..., -1] = -exp[..., -1]
    exp = exp.astype(np.float32)
    for b in range(B):
        end = int(ds[b]) + 1 if int(ds[b]) >= 0 else N
        if not np.allclose(A[:end, b], exp[:end, b], rtol=0, atol=1e-6):
            p.append(f"{label}: исполненные действия среды {b} ≠ журналу "
                     f"({'номинал' if mode == 'legacy' else 'шаг'})")
    if recompute and mode == "fb" and not p:
        p += recompute_problems(g, label, spec, bounds)
    return p


def recompute_problems(g, label, spec, bounds):
    """CPU-пересчёт: ошибки — независимой numpy-геометрией (float64), весь
    шаг — Runtime с весами контроллера из чекпойнта, по сохранённым планам,
    свежим состояниям и маскам."""
    import torch
    C_ = kc.build(torch)
    ck = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    ctrl = {k[len("ctrl."):]: v for k, v in ck["weights"].items()
            if k.startswith("ctrl.")}
    rt = C_["Runtime"](ctrl)
    sp, sR = np.asarray(spec["sigma_p"]), float(spec["sigma_R"])
    N = len(g["step_plan"])
    n = np.arange(N)
    e_np = np_errors(g["plan_ref_p"][g["step_plan"], :, g["step_j"]]
                     .astype(np.float64),
                     g["plan_ref_R"][g["step_plan"], :, g["step_j"]]
                     .astype(np.float64),
                     g["step_state8"].astype(np.float64), sp, sR)
    p = []
    de = np.abs(np.clip(e_np, -kc.E_CLIP, kc.E_CLIP) - g["step_e"]).max()
    dn = np.abs(np.linalg.norm(e_np, axis=-1) - g["step_e_norm"]).max() / \
        max(1.0, float(np.abs(g["step_e_norm"]).max()))
    if not (de <= E_TOL and dn <= E_TOL):
        p.append(f"{label}: ошибка шага ≠ независимому пересчёту "
                 f"({de:.2e}, {dn:.2e})")
    da_max = 0.0
    for c in range(len(g["plan_a"])):
        steps = n[g["step_plan"] == c]
        plan = dict(plan_id=f"re{c}",
                    a0=torch.as_tensor(g["plan_a"][c, :, :, :6]),
                    grip=torch.as_tensor(g["plan_a"][c, :, :, 6]),
                    ref_p=torch.as_tensor(g["plan_ref_p"][c]),
                    ref_R=torch.as_tensor(g["plan_ref_R"][c]),
                    c=torch.as_tensor(g["plan_c"][c]),
                    sigma_p=np.asarray(sp, np.float32), sigma_R=sR,
                    L=np.asarray(bounds["L"]), U=np.asarray(bounds["U"]))
        rt.begin(plan)
        for i in steps:
            a, _d = rt.step(plan, g["step_state8"][i], int(g["step_j"][i]),
                            g["step_active"][i].astype(bool))
            da_max = max(da_max, float(np.abs(a - g["step_a"][i]).max()))
    if not da_max <= DBG_TOL:
        p.append(f"{label}: команды ≠ CPU-пересчёту контроллера "
                 f"({da_max:.2e})")
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
    mode, seed = kp.parse_label(label)
    smoke = [suite, int(task), int(block)] in [
        [c[0], int(c[1]), int(c[2])] for c in plan["smoke"]]
    j = d.get("joint") or {}
    want = dict(label=label, arch="parents_goal_fb24", mode=mode,
                train_seed=seed, plan_sha1=sha(plan_path_),
                prep_id=plan["prep_id"], bar_revision=plan["bar_revision"],
                ckpt_sha1=spec["ckpt_sha1"],
                base_weights_hash=spec["source"]["weights_hash"],
                new_hash=spec.get("new_hash"), control=kc.CONFIG,
                heavy_stride=H, control_stride=1,
                k28_policy_sha1=full_sha(os.path.join(HERE, "k28_policy.py")),
                harness_adapter_sha1=sha(os.path.join(HERE, "k28_harness.py")))
    for k, v in want.items():
        if j.get(k) != v:
            p.append(f"{label}: {k} = {j.get(k)!r}, ожидалось {v!r}")
    if d.get("policy") != "k28":
        p.append(f"{label}: policy {d.get('policy')!r}")
    if not p:
        p += journal_problems(z, label, mode, plan["bounds"],
                              hpc.harness_scale(), spec,
                              recompute=smoke and mode == "fb")
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


def control_blocks(plan):
    """Контроль основного плана: {(parents24_sS, набор, задача, блок):
    путь} — исторический (закреплённые источники) или свежий (план
    control, SHA блоков закреплены)."""
    seed = plan["train_seed"]
    c = plan["control"]
    key = f"parents24_s{seed}"
    if c["status"] == "historical":
        refs = p23.verify_refs(plan)
        return {k: f for k, f in refs.items() if k[0] == key}
    cp = c["plan"]
    cpl = json.load(open(cp, encoding="utf-8"))
    prob = check_plan_inputs(cpl)
    if prob:
        raise SystemExit("план свежего контроля: " + "; ".join(prob[:3]))
    ok, gaps = check_run(cpl, cp, run_dir(cp))
    if gaps:
        raise SystemExit(f"свежий контроль: {gaps[:2]}")
    got = {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": full_sha(f) for k, f in ok.items()}
    if got != c["blocks"]:
        raise SystemExit("свежий контроль ≠ закреплённым блокам")
    return {(key, s, t, b): f for (lab, s, t, b), f in ok.items()}


# --- legacy-сверка ---------------------------------------------------------------
SETUP_CMP = ("suite", "horizon", "max_steps", "waiting_steps", "ensemble",
             "rollout_seed_mode", "seed", "n_envs", "task_id", "init_start")
EP_CMP = ("init_state_id", "init_hash", "init_hash_full", "rollout_seed")
OUT_CMP = ("success", "success_step", "done_step", "own_steps",
           "grip_positive_step", "action_sha1")


def compare_legacy(new, old):
    """(новый блок, исторический) -> (начальные расхождения, исходные
    расхождения, детали)."""
    import k16_protocol as p16
    dn, _ = p16.load(new)
    do, _ = p16.load(old)
    start, outc, det = [], [], []
    for k in SETUP_CMP:
        if dn.get(k) != do.get(k):
            start.append(f"{k}: {dn.get(k)!r} ≠ {do.get(k)!r}")
    en = {e["init_state_id"]: e for e in dn["episodes"]}
    eo = {e["init_state_id"]: e for e in do["episodes"]}
    if sorted(en) != sorted(eo):
        start.append("разные начальные состояния")
        return start, outc, det
    for i in sorted(eo):
        s_ = [k for k in EP_CMP if en[i].get(k) != eo[i].get(k)]
        o_ = [k for k in OUT_CMP if en[i].get(k) != eo[i].get(k)]
        start += [f"эпизод {i}: {k}" for k in s_]
        outc += [f"эпизод {i}: {k}" for k in o_]
        det.append(dict(init=i, same_start=not s_, same=not o_, diff=o_,
                        success_new=en[i]["success"],
                        success_old=eo[i]["success"]))
    return start, outc, det


def legacycheck(plan, path):
    if plan.get("plan_kind") != "legacy":
        raise SystemExit("legacycheck — только для legacy-плана")
    ok, gaps = check_run(plan, path, run_dir(path))
    if gaps:
        raise SystemExit(f"legacy-блоки не прошли проверку: {gaps[:2]}")
    refs = p23.verify_refs(plan)
    seed = plan["train_seed"]
    start, outc, det = [], [], []
    for (lab, s, t, b), f in sorted(ok.items()):
        ref = refs.get((f"parents24_s{seed}", s, t, b))
        if ref is None:
            raise SystemExit(f"нет закреплённого блока parents24_s{seed} "
                             f"{s}/{t}/{b}")
        st_, oc_, d_ = compare_legacy(f, ref)
        start += [f"t{t} i{b} {x}" for x in st_]
        outc += [f"t{t} i{b} {x}" for x in oc_]
        det += [dict(task=t, block=b, **x) for x in d_]
    status = "start_mismatch" if start else (
        "control_needed" if outc else "identical")
    rep = dict(seed=seed, status=status, plan=os.path.abspath(path),
               plan_sha1=full_sha(path), episodes=len(det),
               start_mismatches=start, outcome_mismatches=outc, details=det)
    os.makedirs("reports/k28", exist_ok=True)
    json.dump(rep, open(legacy_report(seed), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    same = sum(x["same"] for x in det)
    print(f"  legacy seed {seed}: эпизодов {len(det)}, совпало побитово "
          f"{same}; статус {status}")
    if start:
        raise SystemExit("начальные состояния/setup ≠ историческим: "
                         + "; ".join(start[:4]) + ". Цепочка остановлена: "
                         "нужна проверка начальных состояний")
    if outc:
        print("  ВНИМАНИЕ: действия/исходы не совпали с историческим "
              "parents24 при совпавших начальных состояниях: "
              + "; ".join(outc[:4]) + ". По правилу плана (прямая сверка "
              "номинала на одинаковых входах — real-check) снимается свежий "
              "контроль parents24 на тех же 150 состояниях до основного SR")
        return 4
    return 0


# --- самопроверка ------------------------------------------------------------------
def toy_block(mode, seed=0, C=3, B=5, ds=(5, -1, 23, 23, 10)):
    """Журнал игрушечного блока через настоящую руку k28_policy (fake BAR) и
    арифметику k28_rollout_core."""
    import copy
    import contextlib
    import torch
    import k21_model as km
    import k28_model as k28m
    import k28_rollout_core as core
    M = k28m.build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0, 1.0])
    mod = M["K28"](copy.deepcopy(km.fake_bar(torch)), L[:6], U[:6], seed=0,
                   mode=mode).eval()
    with torch.no_grad():
        mod.ctrl.l3.weight.normal_(0, 0.5, generator=torch.Generator()
                                   .manual_seed(seed))
    for p_ in mod.parameters():
        p_.requires_grad_(False)
    arm = kp.Arm(torch, mod, None, None, mode,
                 f"parents_goal_fb24{'legacy' if mode == 'legacy' else ''}"
                 f"_s0", {})
    rng = np.random.default_rng(seed)
    s8 = np.concatenate([rng.normal(0, 0.1, (B, 3)), rng.normal(0, 0.5, (B, 3)),
                         np.zeros((B, 2))], 1)
    scale = np.array([0.9, 0.9, 0.9, 0.5, 0.5, 0.5, 1.0])
    ds = np.asarray(ds, np.int64)
    acts = []
    for c in range(C):
        pl = arm.plan(km.fake_batch(torch, B=B, seed=c), 1, s8,
                      contextlib.nullcontext())
        for j in range(H):
            n = c * H + j
            active = (ds < 0) | (n <= ds)
            a = arm.step(pl, s8, j, active)
            acts.append(core.scale_exec(np.asarray(a, np.float32), scale)
                        .astype(np.float32))
            s8 = s8 + np.concatenate([rng.normal(0, 0.003, (B, 3)),
                                      rng.normal(0, 0.01, (B, 3)),
                                      np.zeros((B, 2))], 1) * active[:, None]
    z = dict(arm.take(), actions=np.stack(acts), done_step=ds)
    sd = {k: v for k, v in mod.state_dict().items() if k.startswith("ctrl.")}
    return z, dict(L=L.tolist(), U=U.tolist()), scale, sd, mod


def selftest():
    import tempfile
    import torch
    lab = "parents_goal_fb24_s0"
    z, bounds, scale, sd, mod = toy_block("fb")
    with tempfile.TemporaryDirectory() as td:
        ck = os.path.join(td, "best.pt")
        torch.save(dict(weights={k: v.clone() for k, v in sd.items()}), ck)
        spec = dict(ckpt=ck, sigma_p=mod.sigma_p.tolist(),
                    sigma_R=mod.sigma_R)
        ok = journal_problems(z, lab, "fb", bounds, scale, spec,
                              recompute=True)
        assert ok == [], ok
        assert np.abs(z["k28_step_da"]).max() > 0

        def bad(key, fn, word, rec=False):
            zb = dict(z)
            zb[key] = fn(np.array(z[key], copy=True))
            pb = journal_problems(zb, lab, "fb", bounds, scale, spec,
                                  recompute=rec)
            assert pb and any(word in x for x in pb), (key, pb)
        # исполненные действия заменены номиналом -> отказ
        nom = np.array(z["k28_step_a"], copy=True)
        nom[..., :6] = z["k28_step_a0"]

        def to_exec(a):
            e = np.array(a, np.float32, copy=True)
            e[..., :-1] = e[..., :-1] * scale[:-1]
            e[..., -1] = -e[..., -1]
            return e.astype(np.float32)
        bad("actions", lambda a: to_exec(nom), "исполненные")
        bad("k28_step_da", lambda a: a * 3, "κ")
        bad("k28_step_j", lambda a: (a + 1) % H, "расписанию")
        bad("k28_step_active", lambda a: np.ones_like(a), "активности")
        bad("k28_step_a", lambda a: np.concatenate(
            [a[..., :6], -a[..., 6:]], -1), "схват")
        bad("k28_plan_ref_p", lambda a: a + 0.01, "X_ref(0)")
        bad("k28_plan_id", lambda a: a[::-1].copy(), "plan_id")
        # устаревшее состояние (момент плана вместо свежего) -> пересчёт
        n = np.arange(len(z["k28_step_plan"]))
        bad("k28_step_state8", lambda a: a[(n // H) * H], "пересчёт",
            rec=True)
        bad("k28_step_e", lambda a: a + 0.05, "пересчёт", rec=True)
        zn = dict(z)
        arr = np.array(z["k28_step_r"], copy=True)
        arr[0, 0, 0] = np.nan
        zn["k28_step_r"] = arr
        assert any("нечисловые" in x for x in journal_problems(
            zn, lab, "fb", bounds, scale, spec))
    zl, bounds, scale, _sd, _m = toy_block("legacy")
    labl = "parents_goal_fb24legacy_s0"
    assert journal_problems(zl, labl, "legacy", bounds, scale) == []
    zb = dict(zl)
    zb["actions"] = np.array(zl["actions"], copy=True)
    zb["actions"][3, 1, 0] += 0.01
    assert any("исполненные" in x for x in journal_problems(
        zb, labl, "legacy", bounds, scale))
    assert any("legacy с журналом" in x for x in journal_problems(
        dict(zl, k28_step_a=z["k28_step_a"]), labl, "legacy", bounds, scale))
    # numpy-геометрия = геометрии k28 (независимый пересчёт)
    import k28_geometry as kg
    G = kg.build(torch)
    w = np.random.default_rng(3).normal(0, 1.0, (50, 3))
    R = G["exp_so3"](torch.as_tensor(w, dtype=torch.float64)).numpy()
    assert np.abs(np_exp(w) - R).max() < 1e-9
    assert np.abs(np_log(np_exp(w * 0.5)) - w * 0.5).max() < 1e-7
    assert len(set(CODE)) == len(CODE)
    print("самопроверка k28_protocol пройдена: журнал игрушечного блока "
          "(fb и legacy через настоящую руку) проходит, CPU-пересчёт ошибок "
          "(numpy) и контроллера совпадает; номинал вместо исполненного, "
          "|Δa| > κ, расписание, маска, схват, X_ref(0), plan_id, "
          "устаревшее состояние, порча ошибки, NaN и журнал контроллера у "
          "legacy — отказ")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-28: источники и планы")
    ap.add_argument("cmd", choices=("prepare", "plan", "check-plan",
                                    "check-block", "smoke-check", "cells",
                                    "info", "legacycheck", "control-status",
                                    "selftest"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", choices=("main", "legacy", "control"),
                    default="main")
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
        if a.cmd == "control-status":
            st, _r = control_status(a.seed)
            print(st)
            return 0
        if a.cmd == "plan":
            pth = plan_path(a.seed, a.kind)
            if os.path.exists(pth):
                print(f"  план {pth} уже зафиксирован")
                return 0
            plan = make_plan(a.seed, a.kind, a.device)
            with open(pth + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=1)
            os.replace(pth + ".tmp", pth)
            print(f"  план {pth}: {plan['labels']}, ячеек {len(plan['cells'])}"
                  + (f"; контроль {plan['control']['status']}"
                     if plan.get("control") else ""))
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
        if a.cmd == "legacycheck":
            return legacycheck(plan, a.plan)
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
