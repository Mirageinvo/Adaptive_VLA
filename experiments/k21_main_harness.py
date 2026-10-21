#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21 (расширенная оценка): адаптер харнесса k9h (харнесс не меняется).

fullbar — штатная политика харнесса (--policy fullbar); flat24 и
tree_adapt24, tree_fixed24 — рука k21_main_policy через подмену k15f_policy.build_arm
(--policy k15f). Перед запуском сверяются входы и код плана."""
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
    import k21_main_policy as kp21
    import k21_main_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan)
    lab = h16.arg(argv, "--arm-label")
    if lab not in plan["labels"]:
        prob.append(f"метка {lab!r} не из плана расширенной оценки K-21")
    want_pol = plan["policy"].get(lab)
    if h16.arg(argv, "--policy") != want_pol:
        prob.append(f"политика должна быть {want_pol}")
    starts = str(h16.arg(argv, "--init-starts") or "").replace(",", " ")
    if len(starts.split()) != 1 or int(starts.split()[0]) not in \
            plan["blocks"]:
        prob.append("ровно один блок из плана")
    if int(h16.arg(argv, "--task-id", -1)) not in plan["tasks"]:
        prob.append("задача не из плана")
    if h16.arg(argv, "--n-envs") != str(plan["n_envs"]):
        prob.append("--n-envs не равен плану")
    if prob:
        raise SystemExit("адаптер K-21 (расширенная оценка): " + "; ".join(prob[:6]))
    h16.wrap_process_state(h16.arg(argv, "--root", "third_party/actioncodec"))
    plan_sha = pc.sha(plan_path)
    if want_pol == "k15f":
        def build(device, basis, label, torch, **kw):
            arm = kp21.build_arm(device, basis, label, torch,
                                 ckpt=plan["inputs"][f"ckpt_{label}"],
                                 plan_sha1=plan_sha)
            arm.meta["harness_adapter_sha1"] = adapter_sha()
            arm.meta["k21_plan"] = os.path.abspath(plan_path)
            return arm
        k15f_policy.build_arm = build


if __name__ == "__main__":
    if "--selftest" in sys.argv and len(sys.argv) == 2:
        assert len(adapter_sha()) == 12
        print("самопроверка k21_main_harness пройдена")
        sys.exit(0)
    if "--k21-main-plan" not in sys.argv:
        raise SystemExit("адаптер K-21 (расширенная оценка): нужен --k21-main-plan")
    i = sys.argv.index("--k21-main-plan")
    plan_path = sys.argv[i + 1]
    rest = sys.argv[1:i] + sys.argv[i + 2:]
    sys.argv = [sys.argv[0]] + rest
    install(plan_path, rest)
    import k9h_multiarm_gate as k9h
    k9h.main()
