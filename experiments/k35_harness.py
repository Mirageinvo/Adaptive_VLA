#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-35: адаптер холодного харнесса K-9h (харнесс не меняется).

Рука — k35_policy через подмену k15f_policy.build_arm (--policy k15f):
A/C/N одного загруженного K-35 или S (K-34 T0, выход 24). До запуска
сверяются план (код, подготовка, checkpoint, селекторы, гейты) и то, что
(метка, задача, блок) — ячейка плана; ровно один блок, --n-envs 5,
--seed 101, LIBERO-10. Контекст N (задача, init элементов, сид среды) —
из той же ячейки. Отладочный снимок — только в плане smoke.
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


def argv_problems(plan, argv, arg):
    p = []
    lab = arg(argv, "--arm-label")
    if lab not in plan["arms"]:
        p.append(f"метка {lab!r} не из плана K-35")
    if arg(argv, "--policy") != "k15f":
        p.append("--policy должна быть k15f")
    starts = str(arg(argv, "--init-starts") or "").replace(",", " ").split()
    if len(starts) != 1:
        p.append("ровно один блок")
    else:
        cell = [lab, int(arg(argv, "--task-id", -1)), int(starts[0])]
        if cell not in [list(c) for c in plan["cells"]]:
            p.append(f"ячейка {cell} не из плана")
    for k, v in (("--task-suite", "10"), ("--n-envs", str(plan["n_envs"])),
                 ("--seed", str(plan["env_seed"])),
                 ("--rollout-seed-mode", "fixed")):
        if str(arg(argv, k)) != v:
            p.append(f"{k} должно быть {v}")
    if str(arg(argv, "--ckpt")) != plan["setup"]["ckpt"]:
        p.append("--ckpt ≠ снимку BAR плана")
    return p


def ctx_of(plan, argv, arg):
    start = int(str(arg(argv, "--init-starts")).replace(",", " ").split()[0])
    return dict(task=int(arg(argv, "--task-id")),
                inits=[start + i for i in range(int(plan["n_envs"]))],
                env_seed=int(plan["env_seed"]))


def install(plan_path, argv):
    import k15f_policy
    import k16_harness as h16
    import k35_policy as kp
    import k35_protocol as pc
    plan = json.load(open(plan_path, encoding="utf-8"))
    prob = pc.plan_problems(plan) + argv_problems(plan, argv, h16.arg)
    if prob:
        raise SystemExit("адаптер K-35: " + "; ".join(prob[:6]))
    h16.wrap_process_state(h16.arg(argv, "--root", "third_party/actioncodec"))
    plan_sha = pc.sha(plan_path)
    lab = h16.arg(argv, "--arm-label")
    spec = plan["arms"][lab]["spec"]
    ctx = ctx_of(plan, argv, h16.arg)

    def build(device, basis, label, torch, **kw):
        if label != lab:
            raise SystemExit(f"K-35: метка {label}")
        arm = kp.build_arm(device, basis, label, torch, spec=spec,
                           plan_sha1=plan_sha, ctx=ctx,
                           debug=bool(plan["debug"]))
        arm.meta.update(harness_adapter_sha1=adapter_sha(),
                        k35_plan_sha1=plan_sha)
        return arm
    k15f_policy.build_arm = build


def selftest():
    import k16_harness as h16
    plan = dict(arms={"k35_A": {}, "k35_S": {}},
                cells=[["k35_A", 3, 20], ["k35_S", 3, 20]],
                n_envs=5, env_seed=101, setup=dict(ckpt="/snap"))
    good = ["--policy", "k15f", "--arm-label", "k35_A",
            "--task-suite", "10", "--task-id", "3", "--init-starts", "20",
            "--n-envs", "5", "--seed", "101", "--rollout-seed-mode", "fixed",
            "--ckpt", "/snap"]
    assert argv_problems(plan, good, h16.arg) == []
    assert ctx_of(plan, good, h16.arg) == dict(
        task=3, inits=[20, 21, 22, 23, 24], env_seed=101)
    for i, v in ((1, "fullbar"), (3, "k35_N"), (7, "4"), (9, "25"),
                 (11, "4"), (13, "102"), (15, "block"), (17, "/other")):
        bad = list(good)
        bad[i] = v
        assert argv_problems(plan, bad, h16.arg), bad
    two = list(good)
    two[9] = "20,25"
    assert argv_problems(plan, two, h16.arg)
    print("самопроверка k35_harness пройдена: метка и ячейка плана, один "
          "блок, сид среды, число сред, режим сида, снимок BAR, контекст N "
          "из ячейки")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv and len(sys.argv) == 2:
        sys.exit(selftest())
    if "--k35-plan" not in sys.argv:
        raise SystemExit("адаптер K-35: нужен --k35-plan")
    i = sys.argv.index("--k35-plan")
    plan_path = sys.argv[i + 1]
    rest = sys.argv[1:i] + sys.argv[i + 2:]
    sys.argv = [sys.argv[0]] + rest
    install(plan_path, rest)
    import k9h_multiarm_gate as k9h
    k9h.main()
