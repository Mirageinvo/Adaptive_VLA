#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-17: адаптер харнесса (харнесс и файлы K-16 не меняются)."""
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
    import k17_policy as bp
    import k17_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan)
    lab = h16.arg(argv, "--arm-label")
    if lab not in plan["labels"]:
        prob.append(f"метка {lab!r} не из плана K-17")
    if h16.arg(argv, "--policy") != "k15f":
        prob.append("только --policy k15f")
    starts = str(h16.arg(argv, "--init-starts") or "").replace(",", " ")
    if len(starts.split()) != 1 or int(starts.split()[0]) not in \
            plan["blocks"]:
        prob.append("ровно один блок из плана")
    if int(h16.arg(argv, "--task-id", -1)) not in plan["tasks"]:
        prob.append("задача не из плана")
    if h16.arg(argv, "--n-envs") != str(plan["n_envs"]):
        prob.append("--n-envs не равен плану")
    if prob:
        raise SystemExit("адаптер K-17: " + "; ".join(prob[:6]))
    h16.wrap_process_state(h16.arg(argv, "--root", "third_party/actioncodec"))
    plan_sha = pc.sha(plan_path)

    def build(device, basis, label, torch, **kw):
        arm = bp.build_arm(device, basis, label, torch,
                           models_path=plan["inputs"]["models"],
                           plan_sha1=plan_sha, **kw)
        arm.meta["harness_adapter_sha1"] = adapter_sha()
        arm.meta["k17_plan"] = os.path.abspath(plan_path)
        return arm
    k15f_policy.build_arm = build


if __name__ == "__main__":
    if "--selftest" in sys.argv and len(sys.argv) == 2:
        assert len(adapter_sha()) == 12
        print("самопроверка k17_harness пройдена")
        sys.exit(0)
    if "--k17-plan" not in sys.argv:
        raise SystemExit("адаптер K-17: нужен --k17-plan")
    i = sys.argv.index("--k17-plan")
    plan_path = sys.argv[i + 1]
    rest = sys.argv[1:i] + sys.argv[i + 2:]
    sys.argv = [sys.argv[0]] + rest
    install(plan_path, rest)
    import k9h_multiarm_gate as k9h
    k9h.main()
