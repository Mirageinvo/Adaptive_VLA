#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: расширение оценки на LIBERO spatial, object и goal (без
переобучения; задачи этих наборов входят в 40 обучающих — это расширение
оценки, не перенос на новые задачи).

План: fullbar, flat24, tree_adapt24; наборы spatial, object, goal; задачи
0–9; блоки (по умолчанию 0 — старты 0–4; на этих наборах старты в проекте
не использовались); 5 сред, сид 101; тот же закреплённый снимок BAR и те же
чекпойнты, что в плане A. Предел эпизода задаёт среда (libero2gym: 300
шагов для этих наборов); harness --max-steps 600 — только верхняя граница
цикла, как в остальных планах. Смещения позиций (offsets_by_suite) должны
совпадать со смещениями задач по описанию, на которых шло обучение.
Smoke — задачи 0 и 5 первого блока каждого набора.
"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k21_main_protocol as mp  # noqa: E402
import k21_protocol as pp  # noqa: E402

KIND = "k21_suite_plan"
PLAN = "reports/k21/plan_suites.json"
SUITES = ("spatial", "object", "goal")
LABELS = ("fullbar", "flat24", "tree_adapt24")
OFFSETS = "data/pos_offset_table.json"
CODE = mp.CODE + ("k21_suite_protocol.py", "k21_suite_harness.py",
                  "k21_suite_rollout.sh", "k21_suite_analyze.py")
sha, full_sha = pp.sha, pp.full_sha


def offsets_problems(tb, suite_tasks):
    """suite_tasks: {набор: [описание задачи 0..9]}."""
    p = []
    by = tb.get("offsets_by_suite") or {}
    tasks = tb.get("tasks") or {}
    for s, descs in suite_tasks.items():
        row = by.get(s)
        if row is None or len(row) < len(descs):
            p.append(f"нет смещений набора {s} в offsets_by_suite")
            continue
        for i, dsc in enumerate(descs):
            want = (tasks.get(dsc) or {}).get("pos_offset")
            if want is None:
                p.append(f"{s}/{i}: задача «{dsc[:40]}» не из обучающих")
            elif int(row[i]) != int(want):
                p.append(f"{s}/{i}: смещение {row[i]} ≠ обучающему {want}")
    return p


def suite_tasks():
    from libero.libero import benchmark
    out = {}
    for s in SUITES:
        b = benchmark.get_benchmark_dict()[f"libero_{s}"]()
        out[s] = [b.get_task(i).language for i in range(b.n_tasks)]
    return out


def make_plan(device, blocks):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import k21_data as d21
    import torch
    prep = json.load(open(d21.REPORT))
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    tb = json.load(open(OFFSETS))
    st = suite_tasks()
    prob = offsets_problems(tb, st)
    if prob:
        raise SystemExit("; ".join(prob[:4]))
    inputs = dict(prepare=d21.REPORT, teacher=d21.TEACHER_REPORT,
                  harness_utils=hpc.HARNESS_UTILS, offsets=OFFSETS)
    best = {}
    for lab in ("flat24", "tree_adapt24"):
        tr = json.load(open(mp.TRAIN_REPORT[lab]))
        p = f"data/k21/runs/{lab}/best.pt"
        if full_sha(p) != tr[lab]["best_sha1"]:
            raise SystemExit(f"{p} не совпадает с отчётом обучения")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if ck["config"]["prep_id"] != prep["prep_id"]:
            raise SystemExit(f"{lab}: чекпойнт от другой подготовки")
        inputs[f"ckpt_{lab}"] = p
        best[lab] = tr[lab]["best"]
    setups = {s: dict(pr.expected_setup(), suite=s, ckpt=snap)
              for s in SUITES}
    return dict(
        kind=KIND, created=datetime.datetime.now().isoformat(
            timespec="seconds"),
        labels=list(LABELS), policy={k: mp.POLICY[k] for k in LABELS},
        suites=list(SUITES), suite_tasks=st, tasks=list(range(10)),
        blocks=list(blocks), n_envs=5,
        smoke=[(s, t, blocks[0]) for s in SUITES for t in (0, 5)],
        device=str(device), setups=setups, snapshot_path=snap,
        prep_id=prep["prep_id"], bar_revision=prep["bar_pin"]["revision"],
        bounds=prep["bounds_codec"], best=best,
        inputs={k: os.path.abspath(v) for k, v in inputs.items()},
        inputs_sha1={k: sha(v) for k, v in inputs.items()},
        code={f: sha(os.path.join(HERE, f)) for f in CODE
              if os.path.exists(os.path.join(HERE, f))},
        note=("расширение оценки на наборы, чьи задачи входят в обучение; "
              "старты этих наборов в проекте не использовались"))


def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план оценки наборов K-21"]
    for k, f in plan["inputs"].items():
        if not os.path.exists(f) or sha(f) != plan["inputs_sha1"][k]:
            p.append(f"inputs {k} изменён или отсутствует ({f})")
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    return p


def block_name(lab, suite, t, b):
    return f"{lab}_{suite}_t{t}_i{b}.json"


def check_block(plan, plan_path, path, label, suite, task, block):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import k16_protocol as p16
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
    if label == "fullbar":
        return p
    j = d.get("joint") or {}
    want = dict(label=label, plan_sha1=sha(plan_path),
                prep_id=plan["prep_id"], bar_revision=plan["bar_revision"],
                ckpt_sha1=full_sha(plan["inputs"][f"ckpt_{label}"]),
                k21_policy_sha1=full_sha(os.path.join(HERE,
                                                      "k21_main_policy.py")),
                harness_adapter_sha1=sha(os.path.join(
                    HERE, "k21_suite_harness.py")))
    for k, v in want.items():
        if j.get(k) != v:
            p.append(f"{label}: {k} = {j.get(k)!r}, ожидалось {v!r}")
    p += pp.journal_problems(label, z, plan["bounds"])
    if not p:
        p += hpc.actions_problems(label, z, hpc.harness_scale())
    return p


def check_run(plan, plan_path, run_dir, triples=None):
    ok, gaps = {}, []
    triples = triples or [(s, t, b) for s in plan["suites"]
                          for t in plan["tasks"] for b in plan["blocks"]]
    for s, t, b in triples:
        for lab in plan["labels"]:
            pth = os.path.join(run_dir, block_name(lab, s, t, b))
            if not os.path.exists(pth):
                gaps.append(dict(label=lab, suite=s, task=t, block=b,
                                 why="не готов"))
                continue
            prob = check_block(plan, plan_path, pth, lab, s, t, b)
            if prob:
                gaps.append(dict(label=lab, suite=s, task=t, block=b,
                                 why="; ".join(prob[:2])))
            else:
                ok[(lab, s, t, b)] = pth
    return ok, gaps


def selftest():
    st = dict(spatial=["a", "b"], object=["c"])
    tb = dict(offsets_by_suite=dict(spatial=[1, 2], object=[3]),
              tasks=dict(a=dict(pos_offset=1), b=dict(pos_offset=2),
                         c=dict(pos_offset=3)))
    assert offsets_problems(tb, st) == []
    tb2 = dict(tb, offsets_by_suite=dict(spatial=[1, 5], object=[3]))
    assert offsets_problems(tb2, st)
    assert offsets_problems(dict(tb, offsets_by_suite=dict(spatial=[1, 2])),
                            st)
    assert offsets_problems(dict(tb, tasks=dict(a=dict(pos_offset=1))), st)
    assert block_name("flat24", "goal", 3, 0) == "flat24_goal_t3_i0.json"
    print("самопроверка k21_suite_protocol пройдена: сверка смещений с "
          "обучающими, имена блоков")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-21: оценка на наборах LIBERO")
    ap.add_argument("mode", choices=("plan", "check-plan", "check-block",
                                     "smoke-check", "selftest"))
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--blocks", type=int, nargs="+", default=[0])
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--block", nargs=5, default=None,
                    metavar=("JSON", "МЕТКА", "НАБОР", "ЗАДАЧА", "БЛОК"))
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    try:
        if a.mode == "plan":
            if os.path.exists(a.plan):
                raise SystemExit(f"{a.plan} уже зафиксирован")
            if any(b + 4 >= 45 for b in a.blocks):
                raise SystemExit("старты 45–49 — резерв")
            plan = make_plan(a.device, a.blocks)
            with open(a.plan + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=1)
            os.replace(a.plan + ".tmp", a.plan)
            n = len(plan["labels"]) * len(SUITES) * 10 * len(plan["blocks"])
            print(f"  план {a.plan}: наборы {plan['suites']}, блоки "
                  f"{plan['blocks']}, процессов {n}")
            return 0
        plan = json.load(open(a.plan))
        if a.mode == "check-plan":
            prob = check_plan_inputs(plan)
            print("\n".join(prob) if prob else "  входы плана сверены")
            return 3 if prob else 0
        if a.mode == "check-block":
            path, lab, s, t, b = a.block
            prob = check_block(plan, a.plan, path, lab, s, t, b)
            if prob:
                print("; ".join(prob[:6]))
            return 2 if prob else 0
        ok, gaps = check_run(plan, a.plan, a.run_dir,
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
