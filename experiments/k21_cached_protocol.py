#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fullbar_cached: SR самостоятельного эталона (BAR с кэшем префикса) на
тех же 150 состояниях, что план A K-21 (старты 30–44: блоки 30, 35, 40 всех
десяти задач). 30 новых процессов — только fullbar_cached; fullbar, flat24,
tree_adapt24 берутся из закреплённых и проверяемых заново результатов плана
A K-21 (тот же setup, снимок BAR, сид, правила завершения). Резерв 45–49
закрыт. Эквивалентность штатному пути не подтверждена: план требует
пройденного допуска k21_barcache --admit для той же ревизии, подготовки и
кода и записывает его статус.

Smoke — задачи 0 и 8 блока 30. Проверка блока — validate_block, контракт
руки, журнал (нумерация, конечность), actions = журнал.
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

import k21_main_protocol as mp  # noqa: E402
import k21_protocol as pp  # noqa: E402

KIND = "k21_cached_plan"
PLAN = "reports/k21/plan_cached.json"
LABELS = ("fullbar_cached",)
POLICY = dict(fullbar_cached="k15f")
BLOCKS, SMOKE = mp.BLOCKS, mp.SMOKE
K21_PLAN = mp.plan_path("A")
K21_LABELS = ("fullbar", "flat24", "tree_adapt24")
CODE = ("k21_cached_policy.py", "k21_cached_harness.py",
        "k21_cached_protocol.py", "k21_cached_analyze.py",
        "k21_cached_rollout.sh", "k21_barcache.py", "k20_data.py") + tuple(f for f in mp.CODE if f != "k20_data.py")
sha, full_sha = pp.sha, pp.full_sha


def k21_run_dir(k21_plan_path=K21_PLAN):
    return os.path.join("reports/k21/run", f"p{mp.sha(k21_plan_path)}")


def k21_results(k21_plan_path=K21_PLAN):
    """Проверенные блоки плана A K-21 (заново, тем же check_run)."""
    a = json.load(open(k21_plan_path))
    prob = mp.check_plan_inputs(a)
    if prob:
        raise SystemExit("план A K-21: " + "; ".join(prob[:3]))
    if tuple(a["labels"]) != K21_LABELS:
        raise SystemExit(f"план A K-21: политики {a['labels']}")
    ok, gaps = mp.check_run(a, k21_plan_path, k21_run_dir(k21_plan_path))
    want = len(a["labels"]) * len(a["tasks"]) * len(a["blocks"])
    if len(ok) != want or gaps:
        raise SystemExit(f"план A K-21 неполон: проверено {len(ok)}/{want}, "
                         f"проблем {len(gaps)}")
    return a, ok


def make_plan(device):
    import k21_data as d21
    a, _ok = k21_results()
    prep = json.load(open(d21.REPORT))
    for k, v in (("prep_id", prep["prep_id"]),
                 ("bar_revision", prep["bar_pin"]["revision"])):
        if a[k] != v:
            raise SystemExit(f"план A K-21: {k} ≠ текущей подготовке")
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    if os.path.normpath(snap) != os.path.normpath(a["setup"]["ckpt"]):
        raise SystemExit("снимок BAR не совпадает с планом A K-21")
    import k21_barcache as bc
    adm = bc.require_admitted(dict(prep_report=prep))
    inputs = dict(prepare=d21.REPORT, teacher=d21.TEACHER_REPORT,
                  k21_plan_A=K21_PLAN, barcache_admit=bc.ADMIT_REPORT)
    inputs.update({k: v for k, v in a["inputs"].items()
                   if k in ("harness_utils", "inventory")})
    inv = json.load(open(a["inputs"]["inventory"]))
    used = {t: sorted(int(x) for x in v if 45 <= int(x) <= 49)
            for t, v in inv["used"].items()}
    if any(used.values()):
        raise SystemExit(f"резерв 45–49 использован: {used}")
    return dict(
        kind=KIND, created=datetime.datetime.now().isoformat(
            timespec="seconds"),
        labels=list(LABELS), policy=dict(POLICY), tasks=a["tasks"],
        blocks=a["blocks"], n_envs=a["n_envs"], smoke=a["smoke"],
        device=str(device), setup=a["setup"], snapshot_path=snap,
        prep_id=a["prep_id"], bar_revision=a["bar_revision"],
        bounds=a["bounds"],
        reference=dict(kind="standalone", equivalence=adm["equivalence"],
                       equivalence_confirmed=adm["equivalence_confirmed"],
                       barcache_sha1=adm["barcache_sha1"]),
        k21_plan=dict(path=os.path.abspath(K21_PLAN), sha12=mp.sha(K21_PLAN),
                      run_dir=os.path.abspath(k21_run_dir()),
                      labels=list(K21_LABELS)),
        inputs={k: os.path.abspath(v) for k, v in inputs.items()},
        inputs_sha1={k: sha(v) for k, v in inputs.items()},
        refs=a["refs"], refs_sha1=a["refs_sha1"],
        code={f: sha(os.path.join(HERE, f)) for f in CODE
              if os.path.exists(os.path.join(HERE, f))},
        model_note=("fullbar_cached: исходный BAR, префикс один раз, greedy "
                    "3×16 кодов, сборка трёх уровней; самостоятельный "
                    "эталон — " + adm["equivalence"]),
        note=("те же 150 состояний 30–44, что план A K-21; fullbar, flat24 "
              "и tree_adapt24 переиспользуются из плана A; резерв 45–49 "
              "закрыт"))


def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план оценки fullbar_cached"]
    for grp in ("inputs", "refs"):
        for k, f in plan[grp].items():
            if not os.path.exists(f) or sha(f) != plan[f"{grp}_sha1"][k]:
                p.append(f"{grp} {k} изменён или отсутствует ({f})")
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    return p


def block_name(lab, t, b):
    return f"{lab}_t{t}_i{b}.json"


def journal_problems(label, z):
    """Журнал декодированного BAR: нумерация вызовов и конечность (пределы
    кодека и схват ±1 — свойства однопроходных рук, у BAR их нет, как и у
    штатной руки fullbar)."""
    need = (f"k15d_level_{label}", "k15g_chunk", "actions", "done_step",
            "k21_tokens")
    miss = [k for k in need if k not in z]
    if miss:
        return [f"{label}: в журнале нет {miss}"]
    p = []
    C = len(z["k15g_chunk"])
    if z["k15g_chunk"].tolist() != list(range(1, C + 1)):
        p.append(f"{label}: нумерация вызовов нарушена")
    if not np.isfinite(np.asarray(z[f"k15d_level_{label}"], np.float64)).all():
        p.append(f"{label}: журнал не конечен")
    return p


def check_block(plan, plan_path_, path, label, task, block):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import k16_protocol as p16
    if label not in plan["labels"]:
        return [f"метка {label!r} не из плана fullbar_cached"]
    p = pr.validate_block(path, label, int(task), int(block),
                          require_single=True, setup=plan["setup"])
    if p:
        return p
    d, z = p16.load(path)
    if os.path.basename(os.path.normpath(str(d.get("ckpt")))) != \
            plan["bar_revision"]:
        p.append(f"{label}: снимок BAR блока не закреплённой ревизии")
    j = d.get("joint") or {}
    ref = plan["reference"]
    want = dict(label=label, arm="fullbar_cached", plan_sha1=sha(plan_path_),
                prep_id=plan["prep_id"], bar_revision=plan["bar_revision"],
                reference_kind="standalone",
                barcache_sha1=ref["barcache_sha1"],
                equivalence=ref["equivalence"],
                k21_cached_policy_sha1=full_sha(os.path.join(
                    HERE, "k21_cached_policy.py")),
                harness_adapter_sha1=sha(os.path.join(
                    HERE, "k21_cached_harness.py")))
    for k, v in want.items():
        if j.get(k) != v:
            p.append(f"{label}: {k} = {j.get(k)!r}, ожидалось {v!r}")
    p += journal_problems(label, z)
    if not p:
        p += hpc.actions_problems(label, z, hpc.harness_scale())
    return p


def check_run(plan, plan_path_, run_dir, pairs=None):
    ok, gaps = {}, []
    pairs = pairs or [(t, b) for t in plan["tasks"] for b in plan["blocks"]]
    for t, b in pairs:
        for lab in plan["labels"]:
            pth = os.path.join(run_dir, block_name(lab, t, b))
            if not os.path.exists(pth):
                gaps.append(dict(label=lab, task=t, block=b, why="не готов"))
                continue
            prob = check_block(plan, plan_path_, pth, lab, t, b)
            if prob:
                gaps.append(dict(label=lab, task=t, block=b,
                                 why="; ".join(prob[:2])))
            else:
                ok[(lab, t, b)] = pth
    return ok, gaps


def selftest():
    assert set(LABELS) == set(POLICY) and not set(LABELS) & set(K21_LABELS)
    assert list(BLOCKS) == [30, 35, 40] and BLOCKS[-1] + 4 < 45
    assert len(LABELS) * 10 * len(BLOCKS) == 30
    assert block_name(LABELS[0], 3, 35) == "fullbar_cached_t3_i35.json"
    z = dict(k15d_level_fullbar_cached=np.zeros((3, 5, 8, 7)),
             k15g_chunk=np.arange(1, 4), actions=0, done_step=0, k21_tokens=0)
    assert journal_problems(LABELS[0], z) == []
    z["k15g_chunk"] = np.array([1, 3, 2])
    assert journal_problems(LABELS[0], z)
    assert len(set(CODE)) == len(CODE)
    print("самопроверка k21_cached_protocol пройдена: одна политика, блоки "
          "30–44 без резерва, 30 процессов, журнал")
    return 0


def main():
    ap = argparse.ArgumentParser(description="fullbar_cached: SR эталона")
    ap.add_argument("mode", choices=("plan", "check-plan", "check-block",
                                     "smoke-check", "selftest"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--block", nargs=4, default=None)
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    try:
        if a.mode == "plan":
            if os.path.exists(PLAN):
                raise SystemExit(f"{PLAN} уже зафиксирован")
            plan = make_plan(a.device)
            os.makedirs(os.path.dirname(PLAN), exist_ok=True)
            with open(PLAN + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=1)
            os.replace(PLAN + ".tmp", PLAN)
            print(f"  план {PLAN}: политики {plan['labels']}, блоки "
                  f"{plan['blocks']}, процессов "
                  f"{len(plan['labels']) * 10 * len(plan['blocks'])}; "
                  f"{plan['reference']['equivalence']}; "
                  f"план A K-21 {plan['k21_plan']['sha12']} проверен (90 "
                  f"блоков)")
            return 0
        plan = json.load(open(PLAN))
        if a.mode == "check-plan":
            prob = check_plan_inputs(plan)
            print("\n".join(prob) if prob else "  входы плана сверены")
            return 3 if prob else 0
        if a.mode == "check-block":
            path, lab, t, b = a.block
            prob = check_block(plan, PLAN, path, lab, t, b)
            if prob:
                print("; ".join(prob[:6]))
            return 2 if prob else 0
        ok, gaps = check_run(plan, PLAN, a.run_dir,
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
