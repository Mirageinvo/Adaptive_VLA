#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: замер этапов одного обучающего обновления (диагностика скорости).

Тот же путь, что в k21_train (make_model, schedule, run_micro, AdamW,
GradScaler), на отдельной карте; для каждого обновления — время сборки
батча (кадры + процессор), forward, backward, unscale/clip, шага AdamW,
с синхронизацией CUDA на границах. Обучение не трогает и в его контракт
не входит.
"""
import argparse
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--mode", default="flat24")
    ap.add_argument("--updates", type=int, default=6)
    ap.add_argument("--micro", type=int, default=32)
    a = ap.parse_args()
    import torch
    import k21_data as d21
    import k21_model as km
    import k21_train as kt
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=True)
    mod, bar = kt.make_model(torch, a.mode, D, dev,
                             D["prep_report"]["bar_pin"])
    mod.groups_all = mod.groups()
    kt.set_phase(mod, "main")
    opt = kt.make_opt(torch, mod, "main")
    scaler = torch.cuda.amp.GradScaler()
    dt = torch.float16
    sync = lambda: torch.cuda.synchronize(dev)            # noqa: E731
    rows_all = []
    for u in range(a.updates):
        src, idx = kt.schedule(D, 1000 + u + 1, a.micro, start=1000 + u)[0]
        rows = idx[0]
        t = {}
        sync()
        t0 = time.time()
        if src == "teacher":
            b = d21.make_batch(D, D["teacher_rows"][rows], bar, dev,
                               teacher_idx=rows)
        else:
            b = d21.make_batch(D, rows, bar, dev)
        sync()
        t["batch"] = time.time() - t0
        batch, po, st, tgt, val = b
        t0 = time.time()
        with torch.autocast("cuda", dtype=dt):
            out = mod(batch, po, st)
            loss, _l1, _b = km.loss_fn(torch, out, tgt, val)
        sync()
        t["forward"] = time.time() - t0
        t0 = time.time()
        scaler.scale(loss.mean()).backward()
        sync()
        t["backward"] = time.time() - t0
        t0 = time.time()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], 1.0)
        sync()
        t["clip"] = time.time() - t0
        t0 = time.time()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sync()
        t["adamw"] = time.time() - t0
        t["total"] = sum(t.values())
        t["src"] = src
        rows_all.append(t)
        print(f"  обновление {u}: " + ", ".join(
            f"{k} {v:.2f} с" if isinstance(v, float) else f"{k} {v}"
            for k, v in t.items()), flush=True)
    keys = ("batch", "forward", "backward", "clip", "adamw", "total")
    med = {k: float(np.median([r[k] for r in rows_all[1:]])) for k in keys}
    print("  медиана без первого: " + ", ".join(f"{k} {v:.2f} с"
                                                for k, v in med.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
