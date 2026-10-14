#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16r: адаптер харнесса (харнесс и файлы K-16 не меняются).

`--k16r-plan PLAN` вынимается из argv; план проверяется; обёртка
process_state — та же, что в K-16 (k16_harness.wrap_process_state);
рука — k16r_policy (поворот поправки goal_band_hist_event)."""
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


def install(plan_path, argv):
    import k15f_policy
    import k16_harness as h16
    import k16r_policy as rp
    import k16r_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan)
    lab = h16.arg(argv, "--arm-label")
    if lab not in plan["labels"]:
        prob.append(f"метка {lab!r} не из плана K-16r")
    if h16.arg(argv, "--policy") != "k15f":
        prob.append("только --policy k15f")
    if str(h16.arg(argv, "--init-starts")) != str(plan["block"]):
        prob.append(f"ровно один блок {plan['block']}")
    task = int(h16.arg(argv, "--task-id", -1))
    if task not in plan["tasks"]:
        prob.append("задача не из плана")
    if h16.arg(argv, "--n-envs") != str(plan["n_envs"]):
        prob.append("--n-envs не равен плану")
    if prob:
        raise SystemExit("адаптер K-16r: " + "; ".join(prob[:6]))
    h16.wrap_process_state(h16.arg(argv, "--root", "third_party/actioncodec"))
    plan_sha = pc.sha(plan_path)
    states = [plan["block"] + i for i in range(plan["n_envs"])]

    def build(device, basis, label, torch, **kw):
        arm = rp.build_arm(device, basis, label, torch,
                           models_path=plan["inputs"]["models"],
                           plan_sha1=plan_sha, task=task, states=states, **kw)
        arm.meta["harness_adapter_sha1"] = adapter_sha()
        arm.meta["k16r_plan"] = os.path.abspath(plan_path)
        return arm
    k15f_policy.build_arm = build


if __name__ == "__main__":
    if "--selftest" in sys.argv and len(sys.argv) == 2:
        assert len(adapter_sha()) == 12
        print("самопроверка k16r_harness пройдена")
        sys.exit(0)
    if "--k16r-plan" not in sys.argv:
        raise SystemExit("адаптер K-16r: нужен --k16r-plan")
    i = sys.argv.index("--k16r-plan")
    plan_path = sys.argv[i + 1]
    rest = sys.argv[1:i] + sys.argv[i + 2:]
    sys.argv = [sys.argv[0]] + rest
    install(plan_path, rest)
    import k9h_multiarm_gate as k9h
    k9h.main()
