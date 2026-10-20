#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: тот же k21_train (fitcheck/train), но с ускоренной загрузкой
батчей (k21_fastload). Аргументы — как у k21_train.py."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k21_fastload as fl  # noqa: E402
import k21_train as kt  # noqa: E402

if __name__ == "__main__":
    st = fl.install(None)
    code = kt.main()
    pf = st["pf"]
    if pf is not None:
        print(f"  fastload: из предзагрузки {pf.hits}, синхронно "
              f"{pf.misses}", flush=True)
    sys.exit(code)
