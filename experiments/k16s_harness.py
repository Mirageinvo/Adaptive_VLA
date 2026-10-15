#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16s: адаптер харнесса (харнесс и файлы K-16/K-16r не меняются)."""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def adapter_sha():
    return hashlib.sha1(open(os.path.abspath(__file__), "rb").read()
                        ).hexdigest()[:12]


def install(plan_path, argv):
    import k15f_policy
    import k16_harness as h16
    import k16s_policy as sp
    import k16s_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan)
    lab = h16.arg(argv, "--arm-label")
    if lab not in plan["labels"]:
        prob.append(f"метка {lab!r} не из плана K-16s")
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
        raise SystemExit("адаптер K-16s: " + "; ".join(prob[:6]))
    h16.wrap_process_state(h16.arg(argv, "--root", "third_party/actioncodec"))
    R = np.load(plan["inputs"]["reference"])
    ti = plan["tasks"].index(task)
    ref = dict(call=R["call"][ti], a0=R["a0"][ti], level=R["level"][ti])
    states = [plan["block"] + i for i in range(plan["n_envs"])]
    plan_sha = pc.sha(plan_path)

    def build(device, basis, label, torch, **kw):
        arm = sp.build_arm(device, basis, label, torch,
                           models_path=plan["inputs"]["models"],
                           plan_sha1=plan_sha, task=task, states=states,
                           ref=ref, ref_sha1=plan["inputs_sha1"]["reference"],
                           **kw)
        arm.meta["harness_adapter_sha1"] = adapter_sha()
        arm.meta["k16s_plan"] = os.path.abspath(plan_path)
        return arm
    k15f_policy.build_arm = build


if __name__ == "__main__":
    if "--selftest" in sys.argv and len(sys.argv) == 2:
        assert len(adapter_sha()) == 12
        print("самопроверка k16s_harness пройдена")
        sys.exit(0)
    if "--k16s-plan" not in sys.argv:
        raise SystemExit("адаптер K-16s: нужен --k16s-plan")
    i = sys.argv.index("--k16s-plan")
    plan_path = sys.argv[i + 1]
    rest = sys.argv[1:i] + sys.argv[i + 2:]
    sys.argv = [sys.argv[0]] + rest
    install(plan_path, rest)
    import k9h_multiarm_gate as k9h
    k9h.main()
