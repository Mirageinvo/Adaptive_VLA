#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-24: адаптер харнесса k9h (харнесс не меняется). Рука плана —
k24_policy через подмену k15f_policy.build_arm (--policy k15f); спецификация
руки берётся из плана по метке. Перед запуском сверяются входы, код и
источники плана и то, что (набор, задача, блок) — ячейка плана."""
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


def argv_problems(plan, argv, arg):
    prob = []
    lab = arg(argv, "--arm-label")
    if lab not in plan["labels"]:
        prob.append(f"метка {lab!r} не из плана K-24")
    if arg(argv, "--policy") != plan["policy"].get(lab):
        prob.append(f"политика должна быть {plan['policy'].get(lab)}")
    starts = str(arg(argv, "--init-starts") or "").replace(",", " ").split()
    try:
        cell = [str(arg(argv, "--task-suite")), int(arg(argv, "--task-id",
                                                         -1)),
                int(starts[0]) if len(starts) == 1 else None]
    except ValueError:
        cell = None
    if len(starts) != 1:
        prob.append("ровно один блок")
    if cell not in [[str(c[0]), int(c[1]), int(c[2])] for c in plan["cells"]]:
        prob.append(f"ячейка {cell} не из плана")
    if arg(argv, "--n-envs") != str(plan["n_envs"]):
        prob.append("--n-envs не равен плану")
    if str(arg(argv, "--seed")) != str(plan["env_seed"]):
        prob.append("сид среды не равен плану")
    return prob


def install(plan_path, argv):
    import k15f_policy
    import k16_harness as h16
    import k24_policy as kp
    import k24_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan) + argv_problems(plan, argv, h16.arg)
    if prob:
        raise SystemExit("адаптер K-24: " + "; ".join(prob[:6]))
    h16.wrap_process_state(h16.arg(argv, "--root", "third_party/actioncodec"))
    plan_sha = pc.sha(plan_path)

    def build(device, basis, label, torch, **kw):
        arm = kp.build_arm(device, basis, label, torch,
                           spec=plan["specs"][label], plan_sha1=plan_sha)
        arm.meta["harness_adapter_sha1"] = adapter_sha()
        arm.meta["k24_plan"] = os.path.abspath(plan_path)
        return arm
    k15f_policy.build_arm = build


def selftest():
    import k16_harness as h16
    assert len(adapter_sha()) == 12
    plan = dict(labels=["parents_mr24_s1"], policy={"parents_mr24_s1": "k15f"},
                cells=[["10", 0, 30], ["10", 8, 30]], n_envs=5, env_seed=101)
    good = ["--policy", "k15f", "--arm-label", "parents_mr24_s1", "--task-suite",
            "10", "--task-id", "8", "--init-starts", "30", "--n-envs", "5",
            "--seed", "101"]
    assert argv_problems(plan, good, h16.arg) == []
    for i, v in ((5, "spatial"), (7, "1"), (9, "35"), (3, "parents24_s1"),
                 (13, "102")):
        bad = list(good)
        bad[i] = v
        assert argv_problems(plan, bad, h16.arg), bad
    print("самопроверка k24_harness пройдена: ячейка, метка, сид среды")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv and len(sys.argv) == 2:
        sys.exit(selftest())
    if "--k24-plan" not in sys.argv:
        raise SystemExit("адаптер K-24: нужен --k24-plan")
    i = sys.argv.index("--k24-plan")
    plan_path = sys.argv[i + 1]
    rest = sys.argv[1:i] + sys.argv[i + 2:]
    sys.argv = [sys.argv[0]] + rest
    install(plan_path, rest)
    import k9h_multiarm_gate as k9h
    k9h.main()
