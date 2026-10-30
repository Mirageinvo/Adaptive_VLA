#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-28: запуск пошагового цикла (k28_rollout_core) по плану.

Сверяет входы, код и источники плана, метку и ячейку (набор, задача,
блок), контракт раскатки (H = 8, max_steps 600, waiting 10, ensemble off,
сид 101 fixed, 5 сред); порядок как в K-9h: seed_everything(seed) ->
get_envs -> модель; руку строит k28_policy по спецификации метки плана.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def parse(argv=None):
    ap = argparse.ArgumentParser(description="K-28: пошаговая раскатка")
    ap.add_argument("--k28-plan", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--task-suite", default="10")
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--init-starts", required=True)
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--rollout-seed-mode", default="fixed")
    ap.add_argument("--ensemble", default="off")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--waiting-steps", type=int, default=10)
    ap.add_argument("--run-tag", default="k28_s101")
    ap.add_argument("--arm-label", required=True)
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def argv_problems(plan, a):
    p = []
    if a.arm_label not in plan["labels"]:
        p.append(f"метка {a.arm_label!r} не из плана")
    starts = str(a.init_starts).replace(",", " ").split()
    if len(starts) != 1:
        p.append("ровно один блок")
    cell = [str(a.task_suite), int(a.task_id),
            int(starts[0]) if len(starts) == 1 else None]
    if cell not in [[str(c[0]), int(c[1]), int(c[2])] for c in plan["cells"]]:
        p.append(f"ячейка {cell} не из плана")
    want = dict(n_envs=plan["n_envs"], seed=plan["env_seed"], horizon=8,
                max_steps=600, waiting_steps=10, ensemble="off",
                rollout_seed_mode="fixed")
    for k, v in want.items():
        if getattr(a, k) != v:
            p.append(f"--{k.replace('_', '-')} {getattr(a, k)} ≠ {v}")
    if "{i0}" not in a.out:
        p.append("--out без {i0}")
    return p


def main(argv=None):
    a = parse(argv)
    import k28_policy as kp
    import k28_protocol as pc
    import k28_rollout_core as core
    plan = json.load(open(a.k28_plan, encoding="utf-8"))
    prob = pc.check_plan_inputs(plan) + argv_problems(plan, a)
    if prob:
        raise SystemExit("адаптер K-28: " + "; ".join(prob[:6]))
    root = os.path.abspath(a.root)
    sys.path.insert(0, root)
    import torch
    import actioncodec  # noqa: F401
    from utils import get_envs, seed_everything
    tb = json.load(open(a.offset_table))
    a._pos_off = int(tb["offsets_by_suite"][a.task_suite][a.task_id])
    seed_everything(a.seed)
    envs, task_desc = get_envs(a.task_suite, {"task_id": a.task_id,
                                              "image_size": 224}, a.n_envs)
    a._envs, a._task_desc = envs, task_desc
    try:
        arm = kp.build_arm(torch, a.arm_label, plan["specs"][a.arm_label],
                           a.device, pc.sha(a.k28_plan))
        arm.meta["harness_adapter_sha1"] = pc.sha(os.path.abspath(__file__))
        arm.meta["k28_plan"] = os.path.abspath(a.k28_plan)
        print(f"=== suite {a.task_suite}, задача {a.task_id}, офсет "
              f"{a._pos_off}; «{task_desc}»; метка {a.arm_label} "
              f"({arm.mode})", flush=True)
        core.run(a, arm)
    finally:
        try:
            envs.close()
        except Exception:
            pass
    return 0


def selftest():
    plan = dict(labels=["parents_goal_fb24_s0"], n_envs=5, env_seed=101,
                cells=[["10", 0, 30], ["10", 8, 30]])
    good = ["--k28-plan", "p", "--ckpt", "c", "--task-id", "8",
            "--init-starts", "30", "--arm-label", "parents_goal_fb24_s0",
            "--out", "x_{i0}.json"]
    assert argv_problems(plan, parse(good)) == []
    for extra in (["--horizon", "4"], ["--ensemble", "on"],
                  ["--seed", "102"], ["--init-starts", "35"],
                  ["--arm-label", "parents24_s0"], ["--n-envs", "10"],
                  ["--out", "x.json"]):
        assert argv_problems(plan, parse(good + extra)), extra
    print("самопроверка k28_harness пройдена: метка, ячейка, контракт "
          "раскатки (H, ensemble, сид, число сред, шаги), шаблон --out")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv and len(sys.argv) == 2:
        sys.exit(selftest())
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            sys.exit(3)
        raise
