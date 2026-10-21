#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: обучение tree_fixed24 тем же k21_train (тот же контракт, то же
расписание индексов, исходная инициализация, правило добора — по кривой
этой модели) с ускоренной загрузкой k21_fastload.

Отчёт обучения пишется в reports/k21/train_fixed.json, а не в train.json:
последний закреплён в плане пилота. k21_train.py не меняется (его SHA
входит в контракт уже обученных моделей)."""
import builtins
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k21_fastload as fl  # noqa: E402
import k21_train as kt  # noqa: E402

TARGET = "reports/k21/train.json"
REDIRECT = "reports/k21/train_fixed.json"


def _open(path, *a, **k):
    if os.path.normpath(str(path)) == os.path.normpath(TARGET) and a and \
            "w" in a[0]:
        path = REDIRECT
    return builtins.open(path, *a, **k)


if __name__ == "__main__":
    if "--modes" not in sys.argv:
        sys.argv += ["--modes", "tree_fixed24"]
    i = sys.argv.index("--modes")
    if sys.argv[i + 1:i + 2] != ["tree_fixed24"] or (
            len(sys.argv) > i + 2 and not sys.argv[i + 2].startswith("--")):
        raise SystemExit("этот запускатель обучает только tree_fixed24")
    kt.open = _open                     # перенаправление отчёта
    st = fl.install()
    code = kt.main()
    pf = st["pf"]
    if pf is not None:
        print(f"  fastload: из предзагрузки {pf.hits}, синхронно "
              f"{pf.misses}", flush=True)
    sys.exit(code)
