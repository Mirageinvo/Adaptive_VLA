#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-27: подготовка, планы SR и проверка блоков parents_mem24.

prepare — reports/k27/prepare.json: подготовка K-21 (общие источники),
  индекс последовательностей (SHA), конфигурация памяти; по сидам 0 и 1 —
  источник: исторический parents24 (K-22 / K-23): чекпойнт, SHA, отпечаток
  весов, отчёт; бюджет 2000; отпечаток записи сида. Готовые записи не
  меняются.
plan --seed S --kind off  — reports/k27/plan_s<S>_off.json: метка
  parents_mem24off_s<S> (K27, memory off, веса parents24), только smoke
  (задачи 0 и 8, блок 30): техническая сверка с закреплёнными блоками
  parents24 (offcheck: исход и action_sha1 каждого эпизода).
plan --seed S --kind main — reports/k27/plan_s<S>.json: метка
  parents_mem24_s<S>, LIBERO-10, задачи 0–9, блоки 30/35/40, smoke 0/8
  блока 30; обязательные источники: план A K-21 (fullbar, flat24_s0), fullbar
  cached, K-22 (s0) или повтор K-23 (s1); необязательные описательные —
  K-24 (MR) и K-25 (cond).
check-block — штатные проверки + журнал памяти: формы, конечность,
  эпизод постоянен в блоке, шаги растут на 8, первый вызов без памяти,
  далее k = 8 у всех сред (main) или память не используется (off), гейты в
  (0, 1), суммы строк A0/A1 ∈ {0, 1}, мёртвые узлы без веса; в smoke-блоках
  main — debug-фрагмент: сдвиг, живые узлы, перекрытия, A0, prior средних и
  A1 пересчитываются на CPU (numpy float64).
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
import k27_memory as kmem  # noqa: E402
import k27_policy as kp  # noqa: E402

KIND = "k27_plan"
PREPARE = "reports/k27/prepare.json"
SEQ_REPORT = "reports/k27/sequences.json"
BUDGET = 2000
BOOT = dict(n=10000, seed=20261028)
COMMON_KEYS = p24.COMMON_KEYS
entry_fp = p24.entry_fp
DBG_TOL = 1e-4
CODE = ("k27_protocol.py", "k27_policy.py", "k27_harness.py",
        "k27_analyze.py", "k27_run.sh", "k27_train.py", "k27_model.py",
        "k27_memory.py", "k27_data.py", "k27_latency.py", "k22_model.py",
        "k21_model.py", "k21_tree.py", "k21_data.py", "k21_main_policy.py",
        "k21_policy.py", "k16_harness.py", "k16_policy.py",
        "k9h_multiarm_gate.py", "k15g_local_probe.py", "k15h_protocol.py",
        "k15f_policy.py", "k21_protocol.py")
sha, full_sha = pp.sha, pp.full_sha


def plan_path(seed, kind="main"):
    return f"reports/k27/plan_s{seed}.json" if kind == "main" else \
        f"reports/k27/plan_s{seed}_off.json"


def run_dir(plan_path_):
    return os.path.join("reports/k27/run", f"p{sha(plan_path_)}")


def block_name(lab, suite, t, b):
    return f"{lab}_{suite}_t{t}_i{b}.json"


# --- подготовка ------------------------------------------------------------------
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
    if e.get("memory") != kmem.CONFIG:
        p.append("конфигурация памяти изменилась")
    if common is not None and e.get("common") != common:
        p.append("общие источники изменились")
    return p


def seed_entry(seed, path=None):
    pr = json.load(open(path or PREPARE, encoding="utf-8"))
    e = (pr.get("seeds") or {}).get(str(seed))
    if not e or e.get("status") != "ready":
        raise SystemExit(f"seed {seed}: подготовка K-27 не готова")
    prob = entry_problems(e, {k: pr[k] for k in COMMON_KEYS})
    if prob:
        raise SystemExit(f"seed {seed}: " + "; ".join(prob))
    return e


def prepare():
    import k21_data as d21
    if not os.path.exists(SEQ_REPORT):
        raise SystemExit("нет индекса последовательностей: k27_data.py "
                         "prepare")
    old = json.load(open(PREPARE, encoding="utf-8")) \
        if os.path.exists(PREPARE) else None
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    tj = json.load(open(d21.TEACHER_REPORT, encoding="utf-8"))
    sq = json.load(open(SEQ_REPORT, encoding="utf-8"))
    if sq["prep_id"] != prep["prep_id"]:
        raise SystemExit("индекс последовательностей другой подготовки")
    rep = dict(created=(old or {}).get("created") or datetime.datetime.now()
               .isoformat(timespec="seconds"),
               prep_id=prep["prep_id"], prepare_sha1=full_sha(d21.REPORT),
               teacher_sha1=tj["out"]["sha1"],
               teacher_report_sha1=full_sha(d21.TEACHER_REPORT),
               bar_revision=prep["bar_pin"]["revision"],
               bounds=prep["bounds_codec"], memory=kmem.CONFIG,
               sequences=dict(report_sha1=full_sha(SEQ_REPORT),
                              npz_sha1=sq["npz_sha1"],
                              train_chains=sq["splits"]["train"]["chains"],
                              val_chains=sq["splits"]["val"]["chains"],
                              train_tasks_without_pair=sq["splits"]["train"]
                              ["tasks_without_pair"]),
               seeds=dict((old or {}).get("seeds") or {}))
    common = {k: rep[k] for k in COMMON_KEYS}
    if old:
        for k in COMMON_KEYS:
            if old[k] != rep[k]:
                raise SystemExit(f"подготовка K-21 изменилась: {k}")
        if old.get("sequences", {}).get("npz_sha1") not in (None,
                                                            sq["npz_sha1"]):
            raise SystemExit("индекс последовательностей изменился после "
                             "фиксации")
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
        e = dict(status="ready", seed=s, budget=BUDGET, memory=kmem.CONFIG,
                 source={k: c[k] for k in ("source", "ckpt", "ckpt_sha1",
                                           "weights_hash", "train_report",
                                           "train_report_sha1", "updates",
                                           "seed")},
                 sequences_npz_sha1=sq["npz_sha1"], common=common,
                 fixed=datetime.datetime.now().isoformat(timespec="seconds"))
        e["fingerprint"] = entry_fp(e)
        rep["seeds"][str(s)] = e
    os.makedirs("reports/k27", exist_ok=True)
    json.dump(rep, open(PREPARE + ".tmp", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    os.replace(PREPARE + ".tmp", PREPARE)
    print(f"  подготовка {PREPARE}: prep_id {rep['prep_id']}; цепочек train "
          f"{rep['sequences']['train_chains']}, val "
          f"{rep['sequences']['val_chains']}; память {kmem.VERSION}")
    for s, e in sorted(rep["seeds"].items()):
        print(f"    seed {s}: {e['status']}; источник {e['source']['source']} "
              f"{e['source']['ckpt']} (веса {e['source']['weights_hash'][:12]})"
              f"; отпечаток {e['fingerprint'][:12]}")
    if rep["sequences"]["train_tasks_without_pair"]:
        print(f"  ВНИМАНИЕ: train-задачи без пары: "
              f"{rep['sequences']['train_tasks_without_pair']}")
    return 0


# --- планы ------------------------------------------------------------------------
def optional_refs(seed):
    out = {}
    for name, fn in ((f"k24_s{seed}", lambda: p25.ref_k24(seed)),
                     (f"k25_s{seed}", lambda: p26.ref_k25(seed))):
        try:
            out[name] = dict(status="pinned", ref=p23.pin_ref(fn()))
        except (SystemExit, OSError, KeyError, ValueError) as ex:
            out[name] = dict(status="не выполнено", why="нет полного "
                             "проверенного набора: " + (
                                 str(ex.code) if isinstance(ex, SystemExit)
                                 else repr(ex)))
    return out


def make_plan(seed, kind, device):
    import torch
    import k21_data as d21
    import k27_train as k27t
    e = seed_entry(seed)
    src = e["source"]
    if kind == "off":
        lab = f"parents_mem24off_s{seed}"
        spec = dict(memory="off", seed=seed, source=src,
                    weights_hash=src["weights_hash"], ckpt=src["ckpt"],
                    ckpt_sha1=src["ckpt_sha1"])
        inp = dict(prepare=d21.REPORT, k27_prepare=PREPARE)
    else:
        trp = k27t.report_path(seed)
        tr = json.load(open(trp, encoding="utf-8"))
        if int(tr["updates"]) != BUDGET:
            raise SystemExit(f"обучение K-27 seed {seed}: {tr['updates']} ≠ "
                             f"{BUDGET}")
        ck = os.path.join(k27t.outdir(seed), "best.pt")
        if full_sha(ck) != tr["best_sha1"]:
            raise SystemExit(f"{ck} ≠ отчёту обучения")
        c = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = c["config"]
        if c["mem_hash"] != tr["best_mem_hash"] or \
                int(cfg["seed"]) != seed or cfg["mode"] != "parents_mem24" or \
                cfg["memory"] != kmem.CONFIG or \
                cfg["seed_entry_fp"] != e["fingerprint"] or \
                c["base_hash"] != src["weights_hash"]:
            raise SystemExit("чекпойнт K-27: отпечаток, seed, режим, память, "
                             "запись подготовки или база")
        lab = f"parents_mem24_s{seed}"
        spec = dict(memory="on", seed=seed, source=src, ckpt=os.path.abspath(
            ck), ckpt_sha1=full_sha(ck), mem_hash=c["mem_hash"],
            weights_hash=c["mem_hash"], best_update=int(c["update"]),
            val_L=float(c["val_L"]), train_report=os.path.abspath(trp),
            train_report_sha1=full_sha(trp))
        inp = dict(prepare=d21.REPORT, k27_prepare=PREPARE, train_report=trp)
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
    cells = smoke if kind == "off" else \
        [["10", t, b] for t in A["tasks"] for b in A["blocks"]]
    if any(int(c[2]) + 4 >= 45 for c in cells):
        raise SystemExit("блоки задевают резерв 45–49")
    return dict(
        kind=KIND, plan_kind=kind, name=f"k27_s{seed}_{kind}",
        train_seed=seed,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        labels=[lab], policy={lab: "k15f"}, specs={lab: spec},
        seed_entry=e, seed_entry_fp=e["fingerprint"],
        controls=dict(primary=f"parents24_s{seed}",
                      engineering=["fullbar", "fullbar_cached"],
                      descriptive=[f"flat24_s{seed}"],
                      optional_descriptive=[f"parents_mr24_s{seed}",
                                            f"parents_cond24_s{seed}"]),
        suites=["10"], tasks=list(A["tasks"]), blocks=list(A["blocks"]),
        cells=cells, smoke=smoke, setups={"10": A["setup"]},
        suite_tasks={"10": [desc[t] for t in sorted(desc)]},
        refs=A["refs"], refs_sha1=A["refs_sha1"], device=str(device),
        snapshot_path=snap, prep_id=prep["prep_id"],
        bar_revision=prep["bar_pin"]["revision"], bounds=prep["bounds_codec"],
        n_envs=5, env_seed=101, bootstrap=dict(BOOT), memory=kmem.CONFIG,
        references=refs,
        optional_references=optional_refs(seed) if kind == "main" else {},
        inputs={k: os.path.abspath(v) for k, v in inp.items()},
        inputs_sha1={k: sha(v) for k, v in inp.items()},
        code={f: sha(os.path.join(HERE, f)) for f in CODE
              if os.path.exists(os.path.join(HERE, f))},
        decision_policy="два вывода, как в K-26: project_result (SR к BAR, "
                        "задержка, изменение к parents24) и mechanism "
                        "(parents_mem24 − parents24, McNemar, Holm по двум "
                        "сидам); SR и p не влияют на код завершения",
        note="разработочный банк 30–44 (разведочно); адаптация памяти "
             "поверх обученного parents24")


def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план K-27"]
    for grp in ("inputs", "refs"):
        for k, f in (plan.get(grp) or {}).items():
            if k == "k27_prepare":
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
    if plan.get("memory") != kmem.CONFIG:
        p.append("конфигурация памяти ≠ плану")
    for lab, s in plan["specs"].items():
        for f, h in ((s["ckpt"], s["ckpt_sha1"]),
                     (s["source"]["ckpt"], s["source"]["ckpt_sha1"])):
            if not os.path.exists(f) or full_sha(f) != h:
                p.append(f"чекпойнт {lab} изменён или отсутствует ({f})")
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    return p


# --- журнал памяти --------------------------------------------------------------
def np_softmax_masked(x, allowed):
    any_ = allowed.any(-1, keepdims=True)
    z = np.where(allowed, x, -np.inf)
    z = np.where(any_, z, 0.0)
    m = z.max(-1, keepdims=True)
    e = np.exp(z - m)
    A = e / e.sum(-1, keepdims=True)
    return np.where(any_ & allowed, A, 0.0)


def np_recheck(dbg):
    """Независимый пересчёт A0 и A1 по debug-фрагменту (float64)."""
    f = {k: np.asarray(v, np.float64) for k, v in dbg.items()}
    k = f["k"][:, None]
    use = f["use"] > 0

    def shift(iv):
        l_ = np.maximum(iv[..., 0] - k, 0.0)
        r_ = iv[..., 1] - k
        return np.stack([l_, np.maximum(r_, l_)], -1), r_ > l_

    def wsum(W):
        keep = (np.arange(kmem.T)[None] >= k).astype(np.float64)
        return (W * keep[..., None]).sum(1)

    def overlap(a, b):
        lo = np.maximum(a[:, :, None, 0], b[:, None, :, 0])
        hi = np.minimum(a[:, :, None, 1], b[:, None, :, 1])
        return np.maximum(hi - lo, 0.0)
    ivr, alr = shift(f["old_iv_root"])
    ivm, alm = shift(f["old_iv_mid"])
    alr &= (wsum(f["old_W0"]) > 0) & use[:, None]
    alm &= (wsum(f["old_W1"]) > 0) & use[:, None]
    O0 = overlap(f["root_now"], ivr)
    al0 = (alr[:, None, :] & (O0 > 0))[:, None]
    with np.errstate(divide="ignore"):
        A0 = np_softmax_masked(f["S0"] + np.where(al0, np.log(np.where(
            al0, O0[:, None], 1.0)), 0.0), al0)
    am = alm.astype(np.float64)
    d = am.reshape(-1, 4, 2).sum(-1)[:, np.arange(8) // 2]
    j2 = np.arange(8) // 2
    par = A0[:, :, j2][:, :, :, j2]
    pp_ = np.where(d[:, None, None] > 0, par / np.maximum(d, 1)[:, None,
                                                               None], 0.0)
    pr = (1 - kmem.MIX) * pp_ + kmem.MIX * am[:, None, None] / \
        np.maximum(am.sum(-1), 1)[:, None, None, None]
    pr = np.where(alm[:, None, None], pr, 0.0)
    O1 = overlap(f["mid_now"], ivm)
    base = alm[:, None, :] & (O1 > 0)
    al1 = base[:, None] & (pr > 0)
    with np.errstate(divide="ignore"):
        lg = f["S1"] + np.where(base, np.log(np.where(base, O1, 1.0)),
                                0.0)[:, None] + np.where(
            al1, np.log(np.where(al1, pr, 1.0)), 0.0)
    A1 = np_softmax_masked(lg, al1)
    return A0, A1, alr, alm


def mem_problems(z, label, mode, debug=False):
    keys = ("env", "episode", "step", "k", "use", "iv_mid", "alive_root",
            "alive_mid", "A0", "A1", "g_root", "g_mid", "rms")
    need = [f"k21_mem_{k}" for k in keys] + [f"k15d_level_{label}",
                                             "k21_root"]
    miss = [k for k in need if k not in z]
    if miss:
        return [f"{label}: в журнале нет {miss}"]
    g = {k: np.asarray(z[f"k21_mem_{k}"], np.float64) for k in keys}
    C, B = np.asarray(z[f"k15d_level_{label}"]).shape[:2]
    want = dict(env=(C, B), episode=(C, B), step=(C, B), k=(C, B),
                use=(C, B), iv_mid=(C, B, 8, 2), alive_root=(C, B, 4),
                alive_mid=(C, B, 8), A0=(C, B, 4, 4), A1=(C, B, 8, 8),
                g_root=(C, B, 4), g_mid=(C, B, 8), rms=(C, 2))
    p = [f"{label}: mem_{k} формы {g[k].shape}, ожидалось {s}"
         for k, s in want.items() if g[k].shape != s]
    if p:
        return p
    bad = [k for k, v in g.items() if not np.isfinite(v).all()]
    if bad:
        return [f"{label}: нечисловые значения в mem_{bad}"]
    if (g["episode"] != g["episode"][:1]).any():
        p.append(f"{label}: эпизод меняется внутри блока")
    if C > 1 and (np.diff(g["step"], axis=0) != kmem.K_EXEC).any():
        p.append(f"{label}: между вызовами исполнено не {kmem.K_EXEC} шагов")
    if (g["use"][0] != 0).any():
        p.append(f"{label}: память на первом вызове эпизода")
    if mode == "on":
        if C > 1 and ((g["use"][1:] != 1).any() or
                      (g["k"][1:] != kmem.K_EXEC).any()):
            p.append(f"{label}: память/k на вызовах после первого")
    elif (g["use"] != 0).any() or (g["A0"] != 0).any() or \
            (g["g_root"] != -1).any():
        p.append(f"{label}: memory off использовал память")
    for nm, n in (("A0", 4), ("A1", 8)):
        rs = g[nm].sum(-1)
        if not (np.isclose(rs, 0, atol=1e-5) | np.isclose(rs, 1, atol=1e-5)
                ).all():
            p.append(f"{label}: суммы строк {nm} не 0/1")
        alive = g["alive_root" if nm == "A0" else "alive_mid"]
        if (g[nm] * (1 - alive[:, :, None, :]) != 0).any():
            p.append(f"{label}: вес у мёртвого узла в {nm}")
        _ = n
    for nm in ("g_root", "g_mid"):
        v = g[nm][g[nm] != -1]
        if ((v <= 0) | (v >= 1)).any():
            p.append(f"{label}: гейт вне (0, 1)")
    if (g["alive_root"] * (1 - g["use"][..., None]) != 0).any():
        p.append(f"{label}: живые узлы без памяти")
    if debug and not p:
        dk = ("S0", "A0", "S1", "A1", "root_now", "mid_now", "old_iv_root",
              "old_iv_mid", "old_W0", "old_W1", "old_chunk", "k", "use")
        miss = [k for k in dk if f"k21_mem_dbg_{k}" not in z]
        if miss:
            return [f"{label}: в smoke-журнале нет debug-фрагмента {miss}"]
        dbg = {k: np.asarray(z[f"k21_mem_dbg_{k}"])[0] for k in dk}
        A0, A1, alr, alm = np_recheck(dbg)
        d0 = float(np.abs(A0 - dbg["A0"]).max())
        d1 = float(np.abs(A1 - dbg["A1"]).max())
        if not (d0 <= DBG_TOL and d1 <= DBG_TOL):
            p.append(f"{label}: A0/A1 ≠ пересчёту на CPU ({d0:.2e}/"
                     f"{d1:.2e})")
        if not alr.any():
            p.append(f"{label}: в debug-фрагменте нет живых корней")
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
    want = dict(label=label, arch="parents_mem24", memory=mode,
                train_seed=seed, plan_sha1=sha(plan_path_),
                prep_id=plan["prep_id"], bar_revision=plan["bar_revision"],
                ckpt_sha1=spec["ckpt_sha1"], weights_hash=spec["weights_hash"],
                memory_version=kmem.VERSION, memory_debug=smoke,
                k21_policy_sha1=full_sha(os.path.join(HERE,
                                                      "k21_main_policy.py")),
                k27_policy_sha1=full_sha(os.path.join(HERE, "k27_policy.py")),
                harness_adapter_sha1=sha(os.path.join(HERE, "k27_harness.py")))
    for k, v in want.items():
        if j.get(k) != v:
            p.append(f"{label}: {k} = {j.get(k)!r}, ожидалось {v!r}")
    p += pp.journal_problems(label, z, plan["bounds"])
    if not p:
        p += mem_problems(z, label, mode, debug=smoke and mode == "on")
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


def offcheck(plan, path):
    """memory off = закреплённые блоки parents24 того же сида: исход и
    action_sha1 каждого эпизода."""
    import k16_protocol as p16
    if plan.get("plan_kind") != "off":
        raise SystemExit("offcheck — только для плана off")
    ok, gaps = check_run(plan, path, run_dir(path))
    if gaps:
        raise SystemExit(f"блоки memory off не прошли проверку: {gaps[:2]}")
    refs = p23.verify_refs(plan)
    seed = plan["train_seed"]
    res, bad = [], []
    for (lab, s, t, b), f in sorted(ok.items()):
        ref = refs.get((f"parents24_s{seed}", s, t, b))
        if ref is None:
            raise SystemExit(f"нет закреплённого блока parents24_s{seed} "
                             f"{s}/{t}/{b}")
        da, _za = p16.load(f)
        dr, _zr = p16.load(ref)
        ea = {e["init_state_id"]: e for e in da["episodes"]}
        er = {e["init_state_id"]: e for e in dr["episodes"]}
        for i in sorted(er):
            same = ea.get(i, {}).get("action_sha1") == er[i]["action_sha1"] \
                and ea.get(i, {}).get("success") == er[i]["success"]
            res.append(dict(task=t, block=b, init=i, same=same,
                            success_off=ea.get(i, {}).get("success"),
                            success_ref=er[i]["success"]))
            if not same:
                bad.append((t, i))
    rep = dict(seed=seed, episodes=len(res), mismatches=bad, details=res)
    os.makedirs("reports/k27", exist_ok=True)
    json.dump(rep, open(f"reports/k27/offcheck_s{seed}.json", "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  memory off seed {seed}: эпизодов {len(res)}, совпало "
          f"{len(res) - len(bad)}" + (f"; РАСХОЖДЕНИЯ {bad}" if bad else
                                      " (действия и исходы побитово)"))
    if bad:
        raise SystemExit("memory off ≠ закреплённому parents24 — ошибка "
                         "адаптера, а не повод сменить контроль")
    return 0


def selftest():
    import copy
    import torch
    import k21_model as km
    import k27_model as k27m
    M = k27m.build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    m = M["K27"](copy.deepcopy(km.fake_bar(torch)), L, U, seed=0).eval()
    with torch.no_grad():
        m.mem.Wo.weight.normal_(0, 0.5, generator=torch.Generator()
                                .manual_seed(1))
    lab = "parents_mem24_s0"
    bank = kmem.MemoryBank()
    rows = {}
    for i, step_ in enumerate((10, 18, 26)):
        env, eps = [0, 1, 2], [4, 4, 4]
        use, ks_, _r = bank.select(env, eps, [step_] * 3)
        mi = k27m.mem_input(torch, bank, env, eps, [step_] * 3)
        with torch.no_grad():
            o = m(km.fake_batch(torch, B=3, seed=i), 1,
                  torch.randn(3, 11, generator=torch.Generator()
                              .manual_seed(i)), mi)
        rec = kp.journal_record(o, 3, env, dict(episode=eps,
                                                steps=[step_] * 3, k=ks_),
                                use, keep_debug=i == 1, mem_in=mi)
        g = torch.where(o["grip_logit"] > 0, 1.0, -1.0)
        rec["level"] = torch.cat([o["arm"], g[..., None]], -1)[:, :8].numpy()
        rec["root"] = o["root"].numpy()
        for k, v in rec.items():
            rows.setdefault(k, []).append(np.asarray(v))
        bank.update(env, eps, [step_] * 3, kmem.split_snapshot(o["snapshot"]))
    z = {}
    for k, v in rows.items():
        arr = np.stack(v)
        z[f"k15d_level_{lab}" if k == "level" else f"k21_{k}"] = arr
    assert mem_problems(z, lab, "on", debug=True) == [], \
        mem_problems(z, lab, "on", debug=True)
    for key, mut, word in (
            ("k21_mem_A0", lambda a: a * 1.1, "суммы"),
            ("k21_mem_step", lambda a: a + np.arange(len(a))[:, None],
             "шагов"),
            ("k21_mem_use", lambda a: np.ones_like(a), "первом"),
            ("k21_mem_episode", lambda a: a + np.arange(len(a))[:, None],
             "эпизод"),
            ("k21_mem_dbg_S0", lambda a: a + 3 * np.random.default_rng(0)
             .normal(size=a.shape), "пересчёту"),
            ("k21_mem_dbg_k", lambda a: a + 4, "пересчёту"),
            ("k21_mem_g_root", lambda a: np.where(a > 0, 1.5, a), "гейт")):
        zb = dict(z)
        zb[key] = mut(np.array(z[key], copy=True, dtype=np.float64))
        pb = mem_problems(zb, lab, "on", debug=True)
        assert pb and any(word in x for x in pb), (key, pb)
    zb = {k: v for k, v in z.items() if "dbg" not in k}
    assert any("debug" in x for x in mem_problems(zb, lab, "on", debug=True))
    assert any("memory off" in x for x in mem_problems(z, lab, "off"))
    zn = dict(z)
    a = np.array(z["k21_mem_rms"], copy=True)
    a[0, 0] = np.nan
    zn["k21_mem_rms"] = a
    assert any("нечисловые" in x for x in mem_problems(zn, lab, "on"))
    assert len(set(CODE)) == len(CODE)
    print("самопроверка k27_protocol пройдена: журнал памяти игрушечной руки "
          "(3 вызова) проходит, CPU-пересчёт A0/A1 по debug-фрагменту "
          "совпадает; порча сумм, шагов, первого вызова, эпизода, scores, k, "
          "гейта, NaN, отсутствие debug в smoke и память у off — отказ")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-27: подготовка и планы")
    ap.add_argument("cmd", choices=("prepare", "plan", "check-plan",
                                    "check-block", "smoke-check", "cells",
                                    "info", "offcheck", "selftest"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", choices=("main", "off"), default="main")
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
            pth = plan_path(a.seed, a.kind)
            if os.path.exists(pth):
                print(f"  план {pth} уже зафиксирован")
                return 0
            plan = make_plan(a.seed, a.kind, a.device)
            with open(pth + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=1)
            os.replace(pth + ".tmp", pth)
            print(f"  план {pth}: {plan['labels']}, ячеек {len(plan['cells'])}")
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
        if a.cmd == "offcheck":
            return offcheck(plan, a.plan)
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
