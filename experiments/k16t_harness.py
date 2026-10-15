#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16t: адаптер — learned (рука K-16 goal_band_hist_event без изменений)
или поворот записанного вмешательства (k16s_policy) на задаче 8, блоках
30 и 35. Харнесс и файлы K-16/K-16s не меняются."""
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
    import k16_policy as kp
    import k16s_policy as sp
    import k16t_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan)
    lab = h16.arg(argv, "--arm-label")
    if lab not in [plan["learned"]] + plan["rot"]:
        prob.append(f"метка {lab!r} не из плана K-16t")
    if h16.arg(argv, "--policy") != "k15f":
        prob.append("только --policy k15f")
    block = int(str(h16.arg(argv, "--init-starts")))
    if block not in plan["blocks"]:
        prob.append("блок не из плана")
    if int(h16.arg(argv, "--task-id", -1)) != plan["task"]:
        prob.append("задача не из плана")
    if h16.arg(argv, "--n-envs") != str(plan["n_envs"]):
        prob.append("--n-envs не равен плану")
    ref = None
    if lab in plan["rot"]:
        run_dir = os.path.dirname(os.path.abspath(h16.arg(argv, "--out")))
        prob += pc.reference_ok(plan, plan_path, run_dir)
        if not prob:
            ref = pc.ref_for_block(np.load(pc.REF), block)
    if prob:
        raise SystemExit("адаптер K-16t: " + "; ".join(prob[:6]))
    h16.wrap_process_state(h16.arg(argv, "--root", "third_party/actioncodec"))
    plan_sha = pc.sha(plan_path)
    states = [block + i for i in range(plan["n_envs"])]

    def build(device, basis, label, torch, **kw):
        if label == plan["learned"]:
            arm = kp.build_arm(device, basis, label, torch,
                               models_path=plan["inputs"]["models"],
                               plan_sha1=plan_sha, **kw)
        else:
            arm = sp.build_arm(device, basis, label, torch,
                               models_path=plan["inputs"]["models"],
                               plan_sha1=plan_sha, task=plan["task"],
                               states=states, ref=ref,
                               ref_sha1=pc.sha(pc.REF), **kw)
        arm.meta["harness_adapter_sha1"] = adapter_sha()
        arm.meta["k16t_plan"] = os.path.abspath(plan_path)
        return arm
    k15f_policy.build_arm = build


if __name__ == "__main__":
    if "--selftest" in sys.argv and len(sys.argv) == 2:
        assert len(adapter_sha()) == 12
        print("самопроверка k16t_harness пройдена")
        sys.exit(0)
    if "--k16t-plan" not in sys.argv:
        raise SystemExit("адаптер K-16t: нужен --k16t-plan")
    i = sys.argv.index("--k16t-plan")
    plan_path = sys.argv[i + 1]
    rest = sys.argv[1:i] + sys.argv[i + 2:]
    sys.argv = [sys.argv[0]] + rest
    install(plan_path, rest)
    import k9h_multiarm_gate as k9h
    k9h.main()
