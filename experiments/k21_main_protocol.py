#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: расширенная оценка на стартах 30–44 (блоки 30, 35, 40 всех десяти
задач — 150 парных состояний; использовались в проекте раньше, отложены от
выбора по пилоту). Резерв 45–49 закрыт.

Планы:
  A — fullbar, flat24, tree_adapt24 (90 процессов);
  B — tree_fixed24 (30 процессов), после его обучения.
Smoke каждого плана — задачи 0 и 8 блока 30. Тот же закреплённый снимок BAR,
те же setup, сид и правила завершения, что у пилота; проверка блока — та же
логика (validate_block, контракт руки, журнал, actions = журнал), но для
руки k21_main_policy и адаптера k21_main_harness.
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

KIND = "k21_main_plan"
sha, full_sha = pp.sha, pp.full_sha
BLOCKS = (30, 35, 40)
SMOKE = [(0, 30), (8, 30)]
POLICY = dict(fullbar="fullbar", flat24="k15f", tree_adapt24="k15f",
              tree_fixed24="k15f")
SETS = dict(A=("fullbar", "flat24", "tree_adapt24"), B=("tree_fixed24",))
TRAIN_REPORT = dict(flat24="reports/k21/train.json",
                    tree_adapt24="reports/k21/train.json",
                    tree_fixed24="reports/k21/train_fixed.json")
CODE = ("k21_main_policy.py", "k21_main_harness.py", "k21_main_protocol.py",
        "k21_main_analyze.py", "k21_main_rollout.sh", "k21_policy.py",
        "k21_protocol.py", "k21_model.py", "k21_tree.py", "k21_data.py",
        "k21_train.py", "k20_data.py", "k16_models.py", "k16_harness.py",
        "k16_policy.py", "k9h_multiarm_gate.py", "k15g_local_probe.py",
        "k15h_protocol.py", "k15f_policy.py", "k21_fastload.py",
        "k21_train_fast.py", "k21_train_fixed.py")


def plan_path(name):
    return f"reports/k21/plan_main_{name}.json"


def make_plan(name, device):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import k21_data as d21
    import torch
    labels = SETS[name]
    prep = json.load(open(d21.REPORT))
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    if os.path.basename(os.path.normpath(snap)) != \
            prep["bar_pin"]["revision"]:
        raise SystemExit("путь снимка не соответствует ревизии")
    inputs = dict(prepare=d21.REPORT, teacher=d21.TEACHER_REPORT,
                  harness_utils=hpc.HARNESS_UTILS,
                  inventory="reports/k15h/inventory.json")
    best = {}
    for lab in labels:
        if lab == "fullbar":
            continue
        tr = json.load(open(TRAIN_REPORT[lab]))
        p = f"data/k21/runs/{lab}/best.pt"
        if pp.full_sha(p) != tr[lab]["best_sha1"]:
            raise SystemExit(f"{p} не совпадает с отчётом обучения")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if ck["config"]["prep_id"] != prep["prep_id"]:
            raise SystemExit(f"{lab}: чекпойнт от другой подготовки")
        inputs[f"ckpt_{lab}"] = p
        inputs[f"train_report_{lab}"] = TRAIN_REPORT[lab]
        best[lab] = tr[lab]["best"]
    inv = json.load(open("reports/k15h/inventory.json"))
    refs = {}
    for t, v in inv["libero_bank"].items():
        f = v["path"]
        if not os.path.exists(f) or pp.full_sha(f) != v["sha1"]:
            raise SystemExit(f"файл стартов задачи {t} изменён: {f}")
        refs[f"init_t{t}"] = f
    used = {t: sorted(int(x) for x in v if 45 <= int(x) <= 49)
            for t, v in inv["used"].items()}
    if any(used.values()):
        raise SystemExit(f"резерв 45–49 использован: {used}")
    return dict(
        kind=KIND, name=name, created=datetime.datetime.now().isoformat(
            timespec="seconds"),
        labels=list(labels), policy={k: POLICY[k] for k in labels},
        tasks=list(range(10)), blocks=list(BLOCKS), n_envs=5, smoke=SMOKE,
        device=str(device), setup=dict(pr.expected_setup(), ckpt=snap),
        snapshot_path=snap, prep_id=prep["prep_id"],
        bar_revision=prep["bar_pin"]["revision"],
        bounds=prep["bounds_codec"], best=best,
        inputs={k: os.path.abspath(v) for k, v in inputs.items()},
        inputs_sha1={k: pp.sha(v) for k, v in inputs.items()},
        refs={k: os.path.abspath(v) for k, v in refs.items()},
        refs_sha1={k: pp.sha(v) for k, v in refs.items()},
        code={f: pp.sha(os.path.join(HERE, f)) for f in CODE
              if os.path.exists(os.path.join(HERE, f))},
        note=("старты 30–44: отложены от выбора по пилоту, но использовались "
              "в проекте; не полностью независимая проверка; резерв 45–49 "
              "закрыт"))


def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план расширенной оценки K-21"]
    for grp in ("inputs", "refs"):
        for k, f in plan[grp].items():
            if not os.path.exists(f) or pp.sha(f) != plan[f"{grp}_sha1"][k]:
                p.append(f"{grp} {k} изменён или отсутствует ({f})")
    for f, s_ in plan["code"].items():
        if pp.sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    return p


def check_block(plan, plan_path_, path, label, task, block):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import k16_protocol as p16
    p = pr.validate_block(path, label, int(task), int(block),
                          require_single=True, setup=plan["setup"])
    if p:
        return p
    d, z = p16.load(path)
    if os.path.basename(os.path.normpath(str(d.get("ckpt")))) != \
            plan["bar_revision"]:
        p.append(f"{label}: снимок BAR блока не закреплённой ревизии")
    if label == "fullbar":
        return p
    j = d.get("joint") or {}
    want = dict(label=label, plan_sha1=pp.sha(plan_path_),
                prep_id=plan["prep_id"], bar_revision=plan["bar_revision"],
                ckpt_sha1=pp.full_sha(plan["inputs"][f"ckpt_{label}"]),
                k21_policy_sha1=pp.full_sha(os.path.join(
                    HERE, "k21_main_policy.py")),
                harness_adapter_sha1=pp.sha(os.path.join(
                    HERE, "k21_main_harness.py")))
    for k, v in want.items():
        if j.get(k) != v:
            p.append(f"{label}: {k} = {j.get(k)!r}, ожидалось {v!r}")
    p += pp.journal_problems(label, z, plan["bounds"])
    if not p:
        p += hpc.actions_problems(label, z, hpc.harness_scale())
    return p


def check_run(plan, plan_path_, run_dir, pairs=None):
    ok, gaps = {}, []
    pairs = pairs or [(t, b) for t in plan["tasks"] for b in plan["blocks"]]
    for t, b in pairs:
        for lab in plan["labels"]:
            pth = os.path.join(run_dir, f"{lab}_t{t}_i{b}.json")
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
    assert set(SETS["A"]) | set(SETS["B"]) == set(POLICY)
    assert all(45 > b + 4 for b in BLOCKS[:-1]) and BLOCKS[-1] + 4 == 44
    assert set(TRAIN_REPORT) == {"flat24", "tree_adapt24", "tree_fixed24"}
    print("самопроверка k21_main_protocol пройдена: наборы политик, блоки "
          "30–44 без резерва")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-21: расширенная оценка")
    ap.add_argument("mode", choices=("plan", "check-plan", "check-block",
                                     "smoke-check", "selftest"))
    ap.add_argument("--name", default="A", choices=tuple(SETS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--block", nargs=4, default=None)
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    pth = plan_path(a.name)
    try:
        if a.mode == "plan":
            if os.path.exists(pth):
                raise SystemExit(f"{pth} уже зафиксирован")
            plan = make_plan(a.name, a.device)
            with open(pth + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=1)
            os.replace(pth + ".tmp", pth)
            print(f"  план {pth}: политики {plan['labels']}, блоки "
                  f"{plan['blocks']}, процессов "
                  f"{len(plan['labels']) * 10 * len(plan['blocks'])}")
            return 0
        plan = json.load(open(pth))
        if a.mode == "check-plan":
            prob = check_plan_inputs(plan)
            print("\n".join(prob) if prob else "  входы плана сверены")
            return 3 if prob else 0
        if a.mode == "check-block":
            path, lab, t, b = a.block
            prob = check_block(plan, pth, path, lab, t, b)
            if prob:
                print("; ".join(prob[:6]))
            return 2 if prob else 0
        ok, gaps = check_run(plan, pth, a.run_dir,
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
