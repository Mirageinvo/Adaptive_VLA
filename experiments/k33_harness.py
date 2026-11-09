#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-33: адаптер харнесса K-9h (харнесс не меняется) — новый источник сцен
и шесть рук.

До запуска k9h_multiarm_gate.main подменяется utils.get_envs: векторная
среда libero2gym (воркеры создаются один раз, до модели) из k33_bank.K33Env,
которая на каждом reset создаёт новый объект среды:
  --k33-mode legacy — штатный init_state_id (create seed 0 -> native reset
                      -> init file), для legacy-smoke против действующего
                      холодного пути;
  --k33-mode bank   — init_state_id = индекс случая задачи в банке
                      (--k33-space, --k33-bank-dir).
Обязательно: --task-suite 10, --n-envs 5, --seed 101,
--rollout-seed-mode fixed (по умолчанию K-9h — block), ровно один блок,
кратный 5. Ожидание выполняет только K-9h. process_state оборачивается так
же, как в действующих адаптерах (k16_harness.wrap_process_state).

С --k33-plan P --k33-arm ID --k33-debug 0|1 рука берётся из плана K-33
(k33_policy: T0/F0/T1/F1 — k26_policy, BC — k21_cached_policy, B — штатный
fullbar K-9h); до запуска сверяются план (код, банк, источники руки) и то,
что (фаза, рука, задача, блок, debug) — ячейка плана этой машины. Без
--k33-plan — прежний минимальный режим (только --policy fullbar), которым
пройдены legacy-smoke и dev-replay первой очереди.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def take(argv, name, default=None):
    if name not in argv:
        return default, argv
    i = argv.index(name)
    return argv[i + 1], argv[:i] + argv[i + 2:]


def argv_problems(argv, mode, n_cases=None):
    import k16_harness as h16
    p = []
    for k, v in (("--task-suite", "10"), ("--n-envs", "5"), ("--seed", "101"),
                 ("--rollout-seed-mode", "fixed")):
        if str(h16.arg(argv, k)) != v:
            p.append(f"{k} должно быть {v}")
    if mode not in ("legacy", "bank"):
        p.append(f"режим {mode}")
    starts = str(h16.arg(argv, "--init-starts") or "").replace(",", " ")
    starts = starts.split()
    if not starts:
        p.append("нужен --init-starts")
    for s in starts:
        if not s.isdigit() or int(s) % 5:
            p.append(f"блок {s} не кратен 5")
        elif mode == "bank" and n_cases is not None and \
                int(s) + 5 > n_cases:
            p.append(f"блок {s} вне банка задачи ({n_cases})")
    return p


def arm_argv_problems(argv, arm):
    """Аргументы руки = k33_policy.k9h_args(arm); ровно один блок."""
    import k16_harness as h16
    import k33_policy as kp
    want = kp.k9h_args(arm)
    p = []
    for i in range(0, len(want), 2):
        if str(h16.arg(argv, want[i])) != want[i + 1]:
            p.append(f"{want[i]} должно быть {want[i + 1]}")
    if kp.ARMS[arm]["policy"] != "k15f" and "--k15f-basis" in argv:
        p.append("--k15f-basis у штатной руки")
    starts = str(h16.arg(argv, "--init-starts") or "").replace(",", " ")
    if len(starts.split()) != 1:
        p.append("ровно один блок на процесс")
    return p


def install(argv, mode, space, bank_dir):
    import k16_harness as h16
    import k33_bank as kb
    n_cases = None
    if mode == "bank":
        cases, _m = kb.load_task(int(space), int(h16.arg(argv, "--task-id")),
                                 bank_dir)
        n_cases = len(cases)
    pr = argv_problems(argv, mode, n_cases)
    if pr:
        raise SystemExit("адаптер K-33: " + "; ".join(pr))
    utils = h16.wrap_process_state(h16.arg(argv, "--root",
                                           "third_party/actioncodec"))
    utils.get_envs = kb.make_get_envs(mode, int(space), bank_dir)


def install_planned(argv, plan_path, arm, debug, mode, space, bank_dir):
    """Рука из плана K-33: сверка плана и ячейки, затем среды и рука."""
    import json
    import k16_harness as h16
    import k33_eval as ke
    import k33_policy as kp
    plan = json.load(open(plan_path, encoding="utf-8"))
    cell = dict(arm=arm, task=int(h16.arg(argv, "--task-id", -1)),
                start=int(str(h16.arg(argv, "--init-starts") or "-1")
                          .replace(",", " ").split()[0]),
                mode=mode, space=None if mode == "legacy" else int(space),
                debug=bool(int(debug)))
    pr = ke.plan_problems(plan, arms=[arm]) + arm_argv_problems(argv, arm) + \
        ke.cell_problems(plan, cell)
    if mode == "bank" and os.path.abspath(bank_dir) != \
            os.path.abspath(plan["bank"]["dir"]):
        pr.append("каталог банка ≠ плану")
    if str(h16.arg(argv, "--ckpt")) != plan["setup"]["ckpt"]:
        pr.append("--ckpt ≠ снимку BAR плана")
    if pr:
        raise SystemExit("адаптер K-33: " + "; ".join(pr[:6]))
    install(argv, mode, space, bank_dir)
    kp.install_arm(arm, plan["arms"][arm]["source"], ke.sha(plan_path),
                   cell["debug"])


def selftest():
    good = ["--task-suite", "10", "--n-envs", "5", "--seed", "101",
            "--rollout-seed-mode", "fixed", "--init-starts", "30",
            "--task-id", "0"]
    assert argv_problems(good, "legacy") == []
    for i, v in ((1, "spatial"), (3, "4"), (5, "102"), (7, "block"),
                 (9, "31")):
        bad = list(good)
        bad[i] = v
        assert argv_problems(bad, "legacy"), bad
    b = list(good)
    b[9] = "295"
    assert argv_problems(b, "bank", 300) == []
    b[9] = "300"
    assert argv_problems(b, "bank", 300)
    assert take(["a", "--k33-mode", "bank", "b"], "--k33-mode") == \
        ("bank", ["a", "b"])
    import k33_policy as kp
    for arm in kp.ARMS:
        a = good + kp.k9h_args(arm)
        assert arm_argv_problems(a, arm) == [], arm
        assert arm_argv_problems(good + kp.k9h_args(
            "B" if arm != "B" else "T0"), arm), arm
    two = list(good) + kp.k9h_args("T0")
    two[9] = "30,35"
    assert arm_argv_problems(two, "T0")
    print("самопроверка k33_harness пройдена: обязательные --seed 101, "
          "--rollout-seed-mode fixed, --n-envs 5, LIBERO-10, блоки кратны 5 и "
          "в пределах банка; аргументы каждой руки = плану, один блок на "
          "процесс")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv and len(sys.argv) == 2:
        sys.exit(selftest())
    argv = sys.argv[1:]
    mode, argv = take(argv, "--k33-mode")
    space, argv = take(argv, "--k33-space", "1")
    bank_dir, argv = take(argv, "--k33-bank-dir", "data/k33/bank")
    plan_path, argv = take(argv, "--k33-plan")
    arm, argv = take(argv, "--k33-arm")
    debug, argv = take(argv, "--k33-debug", "0")
    if plan_path:
        if not arm:
            raise SystemExit("адаптер K-33: с --k33-plan нужен --k33-arm")
        install_planned(argv, plan_path, arm, debug, mode, space, bank_dir)
    else:
        if arm:
            raise SystemExit("адаптер K-33: --k33-arm только с --k33-plan")
        install(argv, mode, space, bank_dir)
    sys.argv = [sys.argv[0]] + argv
    import k9h_multiarm_gate as k9h
    k9h.main()
