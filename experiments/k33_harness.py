#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-33: адаптер харнесса K-9h (харнесс не меняется) — новый источник сцен.

До запуска k9h_multiarm_gate.main подменяется utils.get_envs: векторная
среда libero2gym (воркеры создаются один раз, до модели) из k33_bank.K33Env,
которая на каждом reset создаёт новый объект среды:
  --k33-mode legacy — штатный init_state_id (create seed 0 -> native reset
                      -> init file), для legacy-smoke против действующего
                      холодного пути;
  --k33-mode bank   — init_state_id = индекс случая задачи в банке
                      (--k33-space, --k33-bank-dir).
Обязательно: --task-suite 10, --n-envs 5, --seed 101,
--rollout-seed-mode fixed (по умолчанию K-9h — block). Ожидание выполняет
только K-9h. process_state оборачивается так же, как в действующих
адаптерах (k16_harness.wrap_process_state). Рука в этом минимальном
адаптере — собственный путь K-9h (--policy fullbar); остальные руки K-33
подключаются отдельно.
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
    print("самопроверка k33_harness пройдена: обязательные --seed 101, "
          "--rollout-seed-mode fixed, --n-envs 5, LIBERO-10, блоки кратны 5 и "
          "в пределах банка")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv and len(sys.argv) == 2:
        sys.exit(selftest())
    argv = sys.argv[1:]
    mode, argv = take(argv, "--k33-mode")
    space, argv = take(argv, "--k33-space", "1")
    bank_dir, argv = take(argv, "--k33-bank-dir", "data/k33/bank")
    install(argv, mode, space, bank_dir)
    sys.argv = [sys.argv[0]] + argv
    import k9h_multiarm_gate as k9h
    k9h.main()
