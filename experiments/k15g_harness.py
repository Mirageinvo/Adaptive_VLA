#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15g: адаптер харнесса для руки локальной поправки.

Харнесс k9h_multiarm_gate.py НЕ МЕНЯЕТСЯ: от него зависит идущий K-15f M1.
Адаптер подменяет в уже загруженном модуле k15f_policy функцию build_arm
на сборщик руки K-15g и запускает main() харнесса с теми же аргументами.
Харнесс вызывает `import k15f_policy` внутри main и получает этот же
(подменённый) модуль; всё остальное — среды, сиды, декод, запись JSON и
npz — исполняется его неизменённым кодом. В артефакте script_sha1 — по-
прежнему отпечаток k9h_multiarm_gate.py, а отпечаток адаптера пишется в
метаданные руки (`harness_adapter_sha1`).

Принимаются только метки K-15g (q0ref, p<k><l|r><j><p|m>) и только с
--policy k15f; любая другая метка — отказ, чтобы адаптер нельзя было по
ошибке использовать вместо харнесса для M1.

  python3 experiments/k15g_harness.py <аргументы k9h> --policy k15f \\
      --k15f-basis data/k15f/basis_s0.pt --arm-label p2l0p ...
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def adapter_sha():
    return hashlib.sha1(open(os.path.abspath(__file__), "rb").read()
                        ).hexdigest()[:12]


def install():
    """Подменить k15f_policy.build_arm сборщиком K-15g. Возвращает модуль."""
    import k15f_policy
    import k15g_local_policy as lp

    def build(device, basis, label, torch, **kw):
        if not lp.is_label(label):
            raise SystemExit(f"адаптер K-15g принимает только метки K-15g, "
                             f"дано {label!r}")
        arm = lp.build_arm(device, basis, label, torch, **kw)
        arm.meta["harness_adapter_sha1"] = adapter_sha()
        return arm
    k15f_policy.build_arm = build
    return k15f_policy


def check_argv(argv):
    p = []
    if "--policy" not in argv or argv[argv.index("--policy") + 1] != "k15f":
        p.append("адаптер работает только с --policy k15f")
    if "--arm-label" not in argv:
        p.append("нет --arm-label")
    else:
        import k15g_local_policy as lp
        lab = argv[argv.index("--arm-label") + 1]
        if not lp.is_label(lab):
            p.append(f"метка {lab!r} не из K-15g")
    return p


def selftest():
    assert check_argv(["--policy", "k15f", "--arm-label", "p2l0p"]) == []
    assert check_argv(["--policy", "k15f", "--arm-label", "l0p"])
    assert check_argv(["--policy", "k15d", "--arm-label", "p2l0p"])
    assert check_argv(["--policy", "k15f"])
    mod = install()
    try:
        mod.build_arm("cpu", "x", "l0p", None)
        raise AssertionError("адаптер принял метку M1")
    except SystemExit as e:
        assert "только метки K-15g" in str(e)
    assert len(adapter_sha()) == 12
    print("самопроверка k15g_harness пройдена")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv and len(sys.argv) == 2:
        sys.exit(selftest())
    prob = check_argv(sys.argv[1:])
    if prob:
        raise SystemExit("адаптер K-15g: " + "; ".join(prob))
    install()
    import k9h_multiarm_gate as k9h
    k9h.main()
