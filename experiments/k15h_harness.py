#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15h: адаптер харнесса k9h_multiarm_gate.py (харнесс не меняется).

Как k15g_harness: подменяет в загруженном модуле k15f_policy функцию
build_arm сборщиком K-15h и запускает main() харнесса. Собственный аргумент
`--k15h-plan PLAN` вынимается из argv до харнесса. Направления сред блока
берутся из плана по --task-id и --init-starts (ровно один блок на
процесс); перед сборкой руки план проверяется (код и входы не изменились).

  python3 experiments/k15h_harness.py --k15h-plan reports/k15h/plan.json \\
      <аргументы k9h> --policy k15f --k15f-basis data/k15f/basis_s0.pt \\
      --arm-label learned_uniform --task-id 3 --init-starts 25 ...
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def adapter_sha():
    return hashlib.sha1(open(os.path.abspath(__file__), "rb").read()
                        ).hexdigest()[:12]


def take_plan_arg(argv):
    """Вынуть --k15h-plan из argv. (путь, остальные аргументы)."""
    if "--k15h-plan" not in argv:
        raise SystemExit("адаптер K-15h: нужен --k15h-plan")
    i = argv.index("--k15h-plan")
    return argv[i + 1], argv[:i] + argv[i + 2:]


def arg(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def block_directions(plan, task, block):
    """Направления пяти сред блока по порядку env — из плана."""
    tab = plan["directions"].get(str(task)) or {}
    states = list(range(int(block), int(block) + int(plan["n_envs"])))
    miss = [s for s in states if str(s) not in tab]
    if miss:
        raise SystemExit(f"адаптер K-15h: состояний {miss} задачи {task} "
                         f"нет в плане")
    return [tab[str(s)] for s in states]


def check_argv(argv, plan):
    import k15h_policy as hp
    p = []
    if arg(argv, "--policy") != "k15f":
        p.append("адаптер работает только с --policy k15f")
    lab = arg(argv, "--arm-label")
    if lab not in hp.LABELS or lab not in plan["labels"]:
        p.append(f"метка {lab!r} не из плана K-15h")
    starts = str(arg(argv, "--init-starts") or "").replace(",", " ").split()
    if len(starts) != 1:
        p.append("ровно один блок на процесс")
    else:
        t = arg(argv, "--task-id")
        if int(starts[0]) not in [int(b) for b in
                                  plan["blocks"].get(str(t), [])]:
            p.append(f"блок {starts[0]} задачи {t} не из плана")
    if arg(argv, "--n-envs") != str(plan["n_envs"]):
        p.append("--n-envs не равен плану")
    return p


def install(plan_path, argv):
    import k15f_policy
    import k15h_policy as hp
    import k15h_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan, plan_path)
    prob += check_argv(argv, plan)
    if prob:
        raise SystemExit("адаптер K-15h: " + "; ".join(prob))
    task = arg(argv, "--task-id")
    block = str(arg(argv, "--init-starts")).replace(",", " ").split()[0]
    dirs = block_directions(plan, task, block)
    plan_sha = pc.sha(plan_path)

    def build(device, basis, label, torch, **kw):
        arm = hp.build_arm(device, basis, label, torch, directions=dirs,
                           plan_sha1=plan_sha, **kw)
        arm.meta["harness_adapter_sha1"] = adapter_sha()
        arm.meta["k15h_plan"] = os.path.abspath(plan_path)
        return arm
    k15f_policy.build_arm = build
    return k15f_policy


def selftest():
    plan = dict(n_envs=5, labels=["q0ref", "learned_uniform"],
                blocks={"3": [25]},
                directions={"3": {str(s): "1m" for s in range(25, 30)}})
    good = ["--policy", "k15f", "--arm-label", "learned_uniform",
            "--task-id", "3", "--init-starts", "25", "--n-envs", "5"]
    assert check_argv(good, plan) == []
    for i, v in ((3, "p2l0p"), (7, "25,30"), (7, "30"), (1, "k15d"),
                 (9, "10")):
        bad = list(good)
        bad[i] = v
        assert check_argv(bad, plan), (i, v)
    assert block_directions(plan, 3, 25) == ["1m"] * 5
    try:
        block_directions(plan, 3, 30)
        raise AssertionError("принят блок вне плана")
    except SystemExit:
        pass
    p, rest = take_plan_arg(["--a", "1", "--k15h-plan", "P", "--b"])
    assert p == "P" and rest == ["--a", "1", "--b"]
    try:
        take_plan_arg(["--a"])
        raise AssertionError("принят запуск без плана")
    except SystemExit:
        pass
    assert len(adapter_sha()) == 12
    print("самопроверка k15h_harness пройдена")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv and len(sys.argv) == 2:
        sys.exit(selftest())
    plan_path, rest = take_plan_arg(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    install(plan_path, rest)
    import k9h_multiarm_gate as k9h
    k9h.main()
