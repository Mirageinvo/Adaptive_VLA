#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-23: планы оценки (пакеты A и C) и проверка блоков.

Ячейка плана — (набор, задача, блок); набор — значение --task-suite харнесса
("10" — LIBERO-10, "spatial", "object", "goal"). Метка новой руки —
<архитектура>_s<training seed>; её спецификация (архитектура, seed, путь
и SHA чекпойнта, отпечаток весов) — только из плана.

Планы:
  suites_s0  (A) — parents24_s0 (чекпойнт K-22, update 8000) на spatial,
     object, goal; задачи 0–9, блок 0 (старты 0–4), 30 блоков; setup,
     описания задач и блоки — из проверенного плана наборов K-21, его 90
     блоков (fullbar, flat24 -> flat24_s0, tree_adapt24 -> tree_adapt24_s0)
     заново проверяются и закрепляются по SHA JSON/NPZ; LIBERO-10 для
     сводки четырёх наборов — из плана A K-21 и плана K-22 (тоже
     закрепляются). Основное сравнение — parents24_s0 − flat24_s0 по
     объединению трёх наборов.
  repeat_s<S> (C) — flat24_s<S> и parents24_s<S> (обучение K-23) на
     LIBERO-10, задачи 0–9, блоки 30/35/40 (старты 30–44), 60 блоков; setup
     и стартовые файлы — из плана A; переиспользуются fullbar, flat24_s0,
     tree_adapt24_s0 (план A), parents24_s0 (план K-22) и fullbar_cached
     (его план; статус самостоятельного эталона сохраняется). Основное
     сравнение — parents24_s<S> − flat24_s<S>.
Резерв 45–49 и val_confirm не используются; старты 30–44 — разработочный
банк (повтор на нём — повтор обучения, не новая проверка стартов).
"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k21_protocol as pp  # noqa: E402
import k23_policy as kp  # noqa: E402

KIND = "k23_plan"
BOOT = dict(n=10000, seed=20261023)
LABEL_MAP = {"fullbar": "fullbar", "fullbar_cached": "fullbar_cached",
             "flat24": "flat24_s0", "tree_adapt24": "tree_adapt24_s0",
             "parents24": "parents24_s0"}
REPEAT_SEEDS = (1,)
CODE = ("k23_protocol.py", "k23_policy.py", "k23_harness.py",
        "k23_analyze.py", "k23_run.sh", "k23_train.py", "k21_main_policy.py",
        "k21_policy.py", "k22_model.py", "k21_model.py", "k21_tree.py",
        "k21_data.py", "k21_train.py", "k21_fastload.py", "k20_data.py",
        "k16_models.py", "k16_harness.py", "k16_policy.py",
        "k9h_multiarm_gate.py", "k15g_local_probe.py", "k15h_protocol.py",
        "k15f_policy.py", "k21_protocol.py", "k21_main_protocol.py",
        "k21_suite_protocol.py", "k22_protocol.py", "k21_cached_protocol.py",
        "k21_barcache.py")
sha, full_sha = pp.sha, pp.full_sha


def plan_path(kind, seed=None):
    return ("reports/k23/plan_suites_s0.json" if kind == "suites" else
            f"reports/k23/plan_repeat_s{seed}.json")


def run_dir(plan_path_):
    return os.path.join("reports/k23/run", f"p{sha(plan_path_)}")


def block_name(lab, suite, t, b):
    return f"{lab}_{suite}_t{t}_i{b}.json"


# --- прежние результаты --------------------------------------------------------
def _ref(name, path, run, plan, ok, gaps, want, keyfn):
    if len(ok) != want or gaps:
        raise SystemExit(f"{name}: проверено {len(ok)}/{want}, проблем "
                         f"{len(gaps)}")
    blocks = {}
    for k, f in ok.items():
        lab, s, t, b = keyfn(k)
        kk = (LABEL_MAP[lab], s, int(t), int(b))
        if kk in blocks:
            raise SystemExit(f"{name}: повтор {kk}")
        blocks[kk] = f
    return dict(name=name, path=os.path.abspath(path), sha12=sha(path),
                run_dir=os.path.abspath(run), plan=plan, blocks=blocks)


def ref_k21_suites():
    import k21_suite_protocol as sp
    plan = json.load(open(sp.PLAN))
    prob = sp.check_plan_inputs(plan)
    if prob:
        raise SystemExit("план наборов K-21: " + "; ".join(prob[:3]))
    run = os.path.join("reports/k21/run", f"p{sp.sha(sp.PLAN)}")
    ok, gaps = sp.check_run(plan, sp.PLAN, run)
    want = len(plan["labels"]) * len(plan["suites"]) * len(plan["tasks"]) \
        * len(plan["blocks"])
    return _ref("k21_suites", sp.PLAN, run, plan, ok, gaps, want,
                lambda k: k)


def ref_k21_main():
    import k21_main_protocol as mp
    path = mp.plan_path("A")
    plan = json.load(open(path))
    prob = mp.check_plan_inputs(plan)
    if prob:
        raise SystemExit("план A K-21: " + "; ".join(prob[:3]))
    run = os.path.join("reports/k21/run", f"p{mp.sha(path)}")
    ok, gaps = mp.check_run(plan, path, run)
    want = len(plan["labels"]) * len(plan["tasks"]) * len(plan["blocks"])
    return _ref("k21_main_A", path, run, plan, ok, gaps, want,
                lambda k: (k[0], "10", k[1], k[2]))


def ref_k22():
    import k22_protocol as p22
    plan = json.load(open(p22.PLAN))
    prob = p22.check_plan_inputs(plan)
    if prob:
        raise SystemExit("план K-22: " + "; ".join(prob[:3]))
    run = os.path.join("reports/k22/run", f"p{p22.sha(p22.PLAN)}")
    ok, gaps = p22.check_run(plan, p22.PLAN, run)
    want = len(plan["labels"]) * len(plan["tasks"]) * len(plan["blocks"])
    return _ref("k22", p22.PLAN, run, plan, ok, gaps, want,
                lambda k: (k[0], "10", k[1], k[2]))


def ref_cached():
    import k21_cached_protocol as pc
    plan = json.load(open(pc.PLAN))
    prob = pc.check_plan_inputs(plan)
    if prob:
        raise SystemExit("план fullbar_cached: " + "; ".join(prob[:3]))
    run = os.path.join("reports/k21/run_cached", f"p{pc.sha(pc.PLAN)}")
    ok, gaps = pc.check_run(plan, pc.PLAN, run)
    want = len(plan["labels"]) * len(plan["tasks"]) * len(plan["blocks"])
    return _ref("fullbar_cached", pc.PLAN, run, plan, ok, gaps, want,
                lambda k: (k[0], "10", k[1], k[2]))


REFS = dict(k21_suites=ref_k21_suites, k21_main_A=ref_k21_main,
            k22=ref_k22, fullbar_cached=ref_cached)


def npz_of(path):
    d = json.load(open(path))
    return os.path.join(os.path.dirname(path), d["actions_npz"])


def pin_ref(r):
    """Запись для плана: путь и SHA плана-источника, SHA каждого блока."""
    pins = {}
    for (lab, s, t, b), f in sorted(r["blocks"].items()):
        pins[f"{lab}|{s}|{t}|{b}"] = dict(
            json=os.path.abspath(f), json_sha1=full_sha(f),
            npz=os.path.abspath(npz_of(f)), npz_sha1=full_sha(npz_of(f)))
    return dict(name=r["name"], path=r["path"], sha12=r["sha12"],
                run_dir=r["run_dir"], labels=sorted({k[0] for k in
                                                     r["blocks"]}),
                blocks=pins)


def verify_refs(plan):
    """Заново: проверки каждого плана-источника + совпадение SHA его
    блоков с закреплёнными. -> {(метка, набор, задача, блок): путь}."""
    out = {}
    for name, pin in plan["references"].items():
        r = REFS[name]()
        if r["sha12"] != pin["sha12"]:
            raise SystemExit(f"{name}: план-источник изменён")
        keys = {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": (k, f)
                for k, f in r["blocks"].items()}
        if set(keys) != set(pin["blocks"]):
            raise SystemExit(f"{name}: состав блоков изменился")
        for kk, (k, f) in keys.items():
            pb = pin["blocks"][kk]
            if full_sha(f) != pb["json_sha1"] or \
                    full_sha(npz_of(f)) != pb["npz_sha1"]:
                raise SystemExit(f"{name}: блок {kk} изменён")
            if k in out:
                raise SystemExit(f"повтор {k} в планах-источниках")
            out[k] = f
    return out


# --- спецификации новых рук ------------------------------------------------------
def make_spec(label, ckpt, train_report, train_key):
    import torch
    arch, seed = kp.parse_label(label)
    tr = json.load(open(train_report))[train_key]
    if full_sha(ckpt) != tr["best_sha1"]:
        raise SystemExit(f"{ckpt} не совпадает с отчётом обучения")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if ck["weights_hash"] != tr["best_weights_hash"]:
        raise SystemExit(f"{label}: отпечаток весов ≠ отчёту обучения")
    spec = dict(arch=arch, seed=seed, ckpt=os.path.abspath(ckpt),
                ckpt_sha1=full_sha(ckpt), weights_hash=ck["weights_hash"],
                best_update=int(ck["update"]), val_L=float(ck["val_L"]),
                train_report=os.path.abspath(train_report),
                train_report_sha1=full_sha(train_report),
                prep_id=ck["config"]["prep_id"])
    prob = kp.spec_problems(label, spec, ck["config"], ck["weights_hash"])
    if prob:
        raise SystemExit("; ".join(prob))
    return spec


def _base(prep, snap):
    return dict(kind=KIND, created=datetime.datetime.now().isoformat(
        timespec="seconds"), snapshot_path=snap, prep_id=prep["prep_id"],
        bar_revision=prep["bar_pin"]["revision"],
        bounds=prep["bounds_codec"], n_envs=5, env_seed=101,
        bootstrap=dict(BOOT),
        code={f: sha(os.path.join(HERE, f)) for f in CODE
              if os.path.exists(os.path.join(HERE, f))})


def _inputs(d):
    return dict(inputs={k: os.path.abspath(v) for k, v in d.items()},
                inputs_sha1={k: sha(v) for k, v in d.items()})


def make_suites_plan(device):
    import k21_data as d21
    import k21_suite_protocol as sp
    prep = json.load(open(d21.REPORT))
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    r = ref_k21_suites()
    a = r["plan"]
    if a["prep_id"] != prep["prep_id"] or os.path.normpath(
            a["snapshot_path"]) != os.path.normpath(snap):
        raise SystemExit("план наборов K-21: другая подготовка или снимок")
    st = sp.suite_tasks()
    if st != a["suite_tasks"]:
        raise SystemExit("описания задач наборов ≠ плану наборов K-21")
    prob = sp.offsets_problems(json.load(open(sp.OFFSETS)), st)
    if prob:
        raise SystemExit("; ".join(prob[:4]))
    lab = "parents24_s0"
    spec = make_spec(lab, "data/k22/runs/parents24/best.pt",
                     "reports/k22/train.json", "parents24")
    if spec["best_update"] != 8000:
        raise SystemExit(f"parents24: лучшая точка {spec['best_update']}, "
                         f"ожидалась закреплённая 8000")
    cells = [(s, t, b) for s in a["suites"] for t in a["tasks"]
             for b in a["blocks"]]
    refs = {"k21_suites": pin_ref(r), "k21_main_A": pin_ref(ref_k21_main()),
            "k22": pin_ref(ref_k22())}
    return dict(
        _base(prep, snap), name="suites_s0", labels=[lab],
        policy={lab: "k15f"}, specs={lab: spec}, suites=list(a["suites"]),
        tasks=list(a["tasks"]), blocks=list(a["blocks"]),
        cells=[list(c) for c in cells],
        smoke=[list(x) for x in a["smoke"]], setups=a["setups"],
        suite_tasks=a["suite_tasks"], device=str(device), references=refs,
        **_inputs(dict(prepare=d21.REPORT, teacher=d21.TEACHER_REPORT,
                       offsets=sp.OFFSETS)),
        primary=dict(comparison=[lab, "flat24_s0"], scope="pooled3",
                     note="основное — parents24_s0 − flat24_s0 по объединению "
                          "spatial/object/goal; разбивка по наборам, "
                          "сравнения с BAR и сводка четырёх наборов — "
                          "описательные"),
        note="расширение оценки на обучавшиеся задачи; старое основное "
             "сравнение K-21 не переписывается")


def make_repeat_plan(seed, device):
    import k21_data as d21
    import k23_train as k23t
    if seed not in REPEAT_SEEDS:
        raise SystemExit(f"training seed {seed} не из плана K-23")
    prep = json.load(open(d21.REPORT))
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    rA, r22, rc = ref_k21_main(), ref_k22(), ref_cached()
    A = rA["plan"]
    if A["prep_id"] != prep["prep_id"] or os.path.normpath(
            A["setup"]["ckpt"]) != os.path.normpath(snap):
        raise SystemExit("план A: другая подготовка или снимок")
    trp = k23t.report_path(seed)
    tr = json.load(open(trp))
    if not (tr.get("pair") or {}).get("same_schedule"):
        raise SystemExit("пара обучения без общей очереди")
    specs = {}
    for arch in ("flat24", "parents24"):
        lab = f"{arch}_s{seed}"
        specs[lab] = make_spec(lab, os.path.join(k23t.outdir(seed), arch,
                                                 "best.pt"), trp, arch)
    # описания задач LIBERO-10 — из блоков fullbar плана A
    desc = {}
    for (lab, s, t, b), f in rA["blocks"].items():
        if lab == "fullbar":
            d_ = json.load(open(f)).get("task_description")
            if desc.setdefault(t, d_) != d_:
                raise SystemExit(f"план A: разные описания задачи {t}")
    tasks, blocks = list(A["tasks"]), list(A["blocks"])
    if any(b + 4 >= 45 for b in blocks):
        raise SystemExit("блоки задевают резерв 45–49")
    return dict(
        _base(prep, snap), name=f"repeat_s{seed}", train_seed=seed,
        labels=list(specs), policy={k: "k15f" for k in specs}, specs=specs,
        suites=["10"], tasks=tasks, blocks=blocks,
        cells=[["10", t, b] for t in tasks for b in blocks],
        smoke=[["10", t, b] for t, b in A["smoke"]],
        setups={"10": A["setup"]},
        suite_tasks={"10": [desc[t] for t in sorted(desc)]},
        refs=A["refs"], refs_sha1=A["refs_sha1"], device=str(device),
        references={"k21_main_A": pin_ref(rA), "k22": pin_ref(r22),
                    "fullbar_cached": pin_ref(rc)},
        **_inputs(dict(prepare=d21.REPORT, teacher=d21.TEACHER_REPORT,
                       train_report=trp)),
        primary=dict(comparison=[f"parents24_s{seed}", f"flat24_s{seed}"],
                     scope="libero10"),
        note="повтор обучения на разработочном банке 30–44 (не новая "
             "проверка стартов); fullbar_cached — самостоятельный эталон")


# --- проверки --------------------------------------------------------------------
def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план K-23"]
    for grp in ("inputs", "refs"):
        for k, f in (plan.get(grp) or {}).items():
            if not os.path.exists(f) or sha(f) != plan[f"{grp}_sha1"][k]:
                p.append(f"{grp} {k} изменён или отсутствует ({f})")
    for lab, s in plan["specs"].items():
        if not os.path.exists(s["ckpt"]) or full_sha(s["ckpt"]) != \
                s["ckpt_sha1"]:
            p.append(f"чекпойнт {lab} изменён или отсутствует")
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
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
    arch, seed = kp.parse_label(label)
    j = d.get("joint") or {}
    want = dict(label=label, arch=arch, train_seed=seed,
                plan_sha1=sha(plan_path_), prep_id=plan["prep_id"],
                bar_revision=plan["bar_revision"],
                ckpt_sha1=spec["ckpt_sha1"], weights_hash=spec["weights_hash"],
                k21_policy_sha1=full_sha(os.path.join(HERE,
                                                      "k21_main_policy.py")),
                k23_policy_sha1=full_sha(os.path.join(HERE, "k23_policy.py")),
                harness_adapter_sha1=sha(os.path.join(HERE,
                                                      "k23_harness.py")))
    for k, v in want.items():
        if j.get(k) != v:
            p.append(f"{label}: {k} = {j.get(k)!r}, ожидалось {v!r}")
    p += pp.journal_problems(label, z, plan["bounds"])
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
    assert block_name("parents24_s0", "goal", 3, 0) == \
        "parents24_s0_goal_t3_i0.json"
    assert set(LABEL_MAP.values()) >= {"flat24_s0", "parents24_s0"}
    r = _ref("x", __file__, ".", {}, {("flat24", "goal", 1, 0): "f"}, [], 1,
             lambda k: k)
    assert list(r["blocks"]) == [("flat24_s0", "goal", 1, 0)]
    for bad in (dict(ok={}, gaps=[], want=1),
                dict(ok={("flat24", "goal", 1, 0): "f"}, gaps=[{"x": 1}],
                     want=1)):
        try:
            _ref("x", __file__, ".", {}, bad["ok"], bad["gaps"], bad["want"],
                 lambda k: k)
            raise AssertionError(bad)
        except SystemExit:
            pass
    try:
        _ref("x", __file__, ".", {}, {("flat24", "g", 1, 0): "a",
                                      ("flat24", "g", "1", "0"): "b"}, [], 2,
             lambda k: k)
        raise AssertionError("повтор принят")
    except SystemExit:
        pass
    assert len(set(CODE)) == len(CODE) and REPEAT_SEEDS == (1,)
    print("самопроверка k23_protocol пройдена: имена блоков, отображение "
          "меток, неполный/с проблемами/повтор источника — отказ")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-23: планы и проверки")
    ap.add_argument("mode", choices=("plan-suites", "plan-repeat",
                                     "check-plan", "check-block",
                                     "smoke-check", "cells", "info",
                                     "selftest"))
    ap.add_argument("--plan", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--block", nargs=5, default=None)
    ap.add_argument("--which", choices=("smoke", "all"), default="all")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    try:
        if a.mode in ("plan-suites", "plan-repeat"):
            pth = plan_path("suites" if a.mode == "plan-suites" else
                            "repeat", a.seed)
            if os.path.exists(pth):
                print(f"  план {pth} уже зафиксирован")
                return 0
            plan = make_suites_plan(a.device) if a.mode == "plan-suites" \
                else make_repeat_plan(a.seed, a.device)
            os.makedirs(os.path.dirname(pth), exist_ok=True)
            with open(pth + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=1)
            os.replace(pth + ".tmp", pth)
            nref = sum(len(r["blocks"]) for r in plan["references"].values())
            print(f"  план {pth}: метки {plan['labels']}, ячеек "
                  f"{len(plan['cells'])}, новых блоков "
                  f"{len(plan['cells']) * len(plan['labels'])}; "
                  f"переиспользуется проверенных блоков {nref}")
            return 0
        plan = json.load(open(a.plan))
        if a.mode == "check-plan":
            prob = check_plan_inputs(plan)
            if not prob:
                verify_refs(plan)
            print("\n".join(prob) if prob else
                  "  входы и планы-источники сверены")
            return 3 if prob else 0
        if a.mode == "info":
            print(plan["snapshot_path"])
            return 0
        if a.mode == "cells":
            cells = plan["smoke"] if a.which == "smoke" else plan["cells"]
            for s, t, b in cells:
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
        n = len(plan["smoke"]) * len(plan["labels"])
        print(f"  smoke: проверено блоков {len(ok)} из {n}; проблем "
              f"{len(gaps)}" + (f": {gaps[:4]}" if gaps else ""))
        return 3 if gaps else 0
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
