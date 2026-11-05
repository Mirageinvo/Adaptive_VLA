#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-32: волна сбора (5 отдельных эпизодов B = 1 одной задачи) и переходы PPO.

Эпизод — k31_collect.run_episode (свежая среда, seed 101, 10 шагов
ожидания, защёлка первого done/успеха, остановка на первом из них, лимит
600) с k32_policy.Arm (стохастически при сборе). Переход — каждый вызов
политики (только активные границы): S, logits_old, μ_old, r, η (весь), log π,
k, elapsed, R = итоговый успех эпизода. Старты волны — 5 разных init из
train-банка (локальный RNG [32, seed, оператор-независимо, волна]).
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k31_collect as kcol  # noqa: E402

WAVE = 5
WAVE_MAX = WAVE * kcol.EXEC_MAX                     # 3050
TASKS = (0, 6, 8)
CYCLE_MAX = WAVE_MAX * len(TASKS)                   # 9150
TRAIN_INIT = tuple(range(0, 20))
VAL_INIT = tuple(range(20, 25))


def pick_init(seed, wave_id, bank=TRAIN_INIT, n=WAVE):
    g = np.random.default_rng([32, int(seed), 1, int(wave_id)])
    return [int(x) for x in g.choice(np.asarray(bank), n, replace=False)]


def run_wave(sim, io, arm, task, starts, *, episode_base, record=True):
    """-> (эпизоды, переходы | None, физические шаги)."""
    eps, tr = [], []
    phys = 0
    t0 = time.time()
    for j, init in enumerate(starts):
        arm.episode = int(episode_base + j)
        ep = kcol.run_episode(sim, io, arm, task, init)
        phys += ep["physical"]
        eps.append(dict(task=int(task), init=int(init),
                        episode=int(episode_base + j),
                        success=bool(ep["success"]),
                        done_step=ep["done_step"], own_steps=ep["own_steps"],
                        steps=ep["steps"], physical=ep["physical"],
                        init_hash=ep["init_hash"],
                        actions_sha1=ep["actions_sha1"],
                        calls=len(ep["calls"])))
        if record:
            for c in ep["calls"]:
                r = c["rec"]
                tr.append(dict(S=r["S"], logits=r["logits"], mu=r["mu"],
                               r=r["r"], eta=r["eta"], logpi=r["logpi"],
                               k=int(c["k"]), elapsed=int(r["elapsed"]),
                               R=float(ep["success"]),
                               episode=int(episode_base + j),
                               clipped=bool(r["clipped"]),
                               capped=bool(r["capped"]), rms=float(r["rms"])))
    buf = pack(tr) if record and tr else None
    return eps, buf, phys, time.time() - t0


def pack(tr):
    return dict(S=np.stack([t["S"] for t in tr]).astype(np.float32),
                logits=np.stack([t["logits"] for t in tr]).astype(np.float32),
                mu=np.stack([t["mu"] for t in tr]).astype(np.float32),
                r=np.array([t["r"] for t in tr], np.int64),
                eta=np.stack([t["eta"] for t in tr]).astype(np.float32),
                logpi=np.array([t["logpi"] for t in tr], np.float64),
                k=np.array([t["k"] for t in tr], np.int64),
                elapsed=np.array([t["elapsed"] for t in tr], np.int64),
                R=np.array([t["R"] for t in tr], np.float32),
                episode=np.array([t["episode"] for t in tr], np.int64),
                clipped=np.array([t["clipped"] for t in tr]),
                capped=np.array([t["capped"] for t in tr]),
                rms=np.array([t["rms"] for t in tr], np.float32))


def concat(bufs):
    bufs = [b for b in bufs if b is not None]
    return {k: np.concatenate([b[k] for b in bufs]) for k in bufs[0]}


def buf_problems(b, d, H):
    """Формы и конечность всех числовых полей буфера."""
    n = len(b["r"])
    want = dict(S=(n, 6 * H + 72), logits=(n, 3), mu=(n, 3, d), eta=(n, d))
    p = [f"{k} формы {b[k].shape}" for k, s in want.items()
         if b[k].shape != s]
    if p:
        return p
    for k in ("S", "logits", "mu", "eta", "logpi", "R", "rms"):
        if not np.isfinite(np.asarray(b[k], np.float64)).all():
            p.append(f"{k}: нечисловые значения")
    if ((b["r"] < 0) | (b["r"] > 2)).any() or ((b["k"] < 1) |
                                               (b["k"] > 8)).any():
        p.append("r/k вне диапазона")
    return p


def selftest():
    assert pick_init(0, 3) == pick_init(0, 3) and \
        len(set(pick_init(0, 3))) == WAVE and \
        set(pick_init(1, 3)) <= set(TRAIN_INIT)
    b = dict(S=np.zeros((2, 6 * 4 + 72), np.float32),
             logits=np.zeros((2, 3), np.float32),
             mu=np.zeros((2, 3, 6), np.float32),
             eta=np.zeros((2, 6), np.float32), logpi=np.zeros(2),
             R=np.zeros(2, np.float32), rms=np.zeros(2, np.float32),
             r=np.array([0, 2]), k=np.array([8, 3]))
    assert buf_problems(b, 6, 4) == []
    for k, v in (("eta", np.inf), ("S", np.nan), ("logpi", -np.inf)):
        bb = {kk: np.array(vv, copy=True) for kk, vv in b.items()}
        bb[k].flat[0] = v
        assert buf_problems(bb, 6, 4), k
    bb = dict(b, eta=np.zeros((2, 48), np.float32))
    assert buf_problems(bb, 6, 4)
    print("самопроверка k32_collect пройдена: старты волны (5 разных из "
          "0…19, воспроизводимо), формы/конечность буфера (NaN/±Inf, "
          "неверная размерность η — отказ)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
