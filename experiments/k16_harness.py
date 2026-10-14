#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16: адаптер харнесса k9h_multiarm_gate.py (харнесс не меняется).

Как k15h_harness: подменяет k15f_policy.build_arm сборщиком K-16 и
запускает main() харнесса. `--k16-plan PLAN` вынимается из argv. Перед
сборкой руки план проверяется (входы, эталоны, код), метка — из плана,
блок — ровно один и равен блоку плана; доноры сред блока — из плана.

НАБЛЮДАЕМОЕ СОСТОЯНИЕ РУКИ: обёртка utils.process_state (пакет utils
корня ActionCodec, из которого харнесс импортирует process_state)
возвращает ТОТ ЖЕ объект результата и копирует его в
k16_policy.STATE_BUF. Харнесс вызывает её ровно один раз перед каждым
вызовом политики, так что путь вычисления q0 не меняется.
"""
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


def arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def take_plan_arg(argv):
    if "--k16-plan" not in argv:
        raise SystemExit("адаптер K-16: нужен --k16-plan")
    i = argv.index("--k16-plan")
    return argv[i + 1], argv[:i] + argv[i + 2:]


def check_argv(argv, plan):
    import k16_policy as kp
    p = []
    if arg(argv, "--policy") != "k15f":
        p.append("адаптер работает только с --policy k15f")
    lab = arg(argv, "--arm-label")
    if lab not in kp.LABELS or lab not in plan["labels"]:
        p.append(f"метка {lab!r} не из плана K-16")
    starts = str(arg(argv, "--init-starts") or "").replace(",", " ").split()
    if starts != [str(plan["block"])]:
        p.append(f"ровно один блок {plan['block']} на процесс")
    if int(arg(argv, "--task-id", -1)) not in plan["tasks"]:
        p.append("задача не из плана")
    if arg(argv, "--n-envs") != str(plan["n_envs"]):
        p.append("--n-envs не равен плану")
    return p


def wrap_process_state(root):
    """Обёртка utils.process_state: тот же результат + копия в буфер."""
    import k16_policy as kp
    r = os.path.abspath(root)
    if r not in sys.path:
        sys.path.insert(0, r)
    import utils
    if getattr(utils.process_state, "_k16", False):
        return utils
    orig = utils.process_state

    def process_state(state):
        out = orig(state)
        kp.STATE_BUF.append(np.array(out, dtype=np.float64, copy=True))
        return out
    process_state._k16 = True
    process_state._orig = orig
    utils.process_state = process_state
    return utils


def install(plan_path, argv):
    import k15f_policy
    import k16_policy as kp
    import k16_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan) + check_argv(argv, plan)
    if prob:
        raise SystemExit("адаптер K-16: " + "; ".join(prob[:6]))
    task = str(arg(argv, "--task-id"))
    lab = arg(argv, "--arm-label")
    donors = None
    if kp.POLICIES[lab][3] == "time":
        donors = [int(plan["donors"][task][str(plan["block"] + i)][1])
                  for i in range(plan["n_envs"])]
    plan_sha = pc.sha(plan_path)
    wrap_process_state(arg(argv, "--root", "third_party/actioncodec"))

    def build(device, basis, label, torch, **kw):
        arm = kp.build_arm(device, basis, label, torch,
                           models_path=plan["inputs"]["models"],
                           plan_sha1=plan_sha, donor_calls=donors, **kw)
        arm.meta["harness_adapter_sha1"] = adapter_sha()
        arm.meta["k16_plan"] = os.path.abspath(plan_path)
        return arm
    k15f_policy.build_arm = build
    return k15f_policy


def selftest():
    import tempfile
    plan = dict(labels=["q0", "goal_band_hist_time"], block=25, tasks=[0, 8],
                n_envs=5)
    good = ["--policy", "k15f", "--arm-label", "q0", "--task-id", "8",
            "--init-starts", "25", "--n-envs", "5"]
    assert check_argv(good, plan) == []
    for i, v in ((3, "direct_hist_event"), (7, "25,30"), (7, "30"),
                 (5, "3"), (1, "k15d"), (9, "10")):
        bad = list(good)
        bad[i] = v
        assert check_argv(bad, plan), (i, v)
    p, rest = take_plan_arg(["--a", "--k16-plan", "P", "--b"])
    assert p == "P" and rest == ["--a", "--b"]
    # обёртка: тот же объект результата, копия в буфер
    import k16_policy as kp
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "utils"))
        open(os.path.join(td, "utils", "__init__.py"), "w").write(
            "def process_state(s):\n    return s * 2\n")
        saved = sys.modules.pop("utils", None)
        try:
            u = wrap_process_state(td)
            x = np.ones((2, 8))
            n0 = len(kp.STATE_BUF)
            y = u.process_state(x)
            assert np.array_equal(y, x * 2) and len(kp.STATE_BUF) == n0 + 1
            assert np.array_equal(kp.STATE_BUF[-1], x * 2)
            y[0, 0] = 99                       # буфер — копия
            assert kp.STATE_BUF[-1][0, 0] == 2
            assert wrap_process_state(td) is u   # повторно не оборачивает
            u.process_state(x)
            assert len(kp.STATE_BUF) == n0 + 2
        finally:
            sys.modules.pop("utils", None)
            if saved is not None:
                sys.modules["utils"] = saved
            sys.path.remove(os.path.abspath(td))
    print("самопроверка k16_harness пройдена: аргументы, план, обёртка "
          "process_state (тот же результат, копия, без двойной обёртки)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv and len(sys.argv) == 2:
        sys.exit(selftest())
    plan_path, rest = take_plan_arg(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    install(plan_path, rest)
    import k9h_multiarm_gate as k9h
    k9h.main()
