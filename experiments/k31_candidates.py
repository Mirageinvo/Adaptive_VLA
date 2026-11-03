#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-31: фиксированный генератор предложений, нормировка эффекта, общий пул,
признаки селектора (149) и правило выбора. Всё — FP32 на CPU.

Строки пула (порядок фиксирован): noop, root, mid, leaf, flat0, flat1, flat2.
RNG: np.random.Generator(SeedSequence([31, space, source_seed, task, init,
call, stream])) — локальный, глобальные генераторы не трогаются; space 0 —
основной набор, 1 — smoke.
Дерево: g ~ N(0, I6), d = g/‖g‖ (нулевая норма — технический отказ),
  e_v = 0.1·d — один d для root*, mid*, leaf* (знак не подбирается).
Плоский контроль: три массива [8,6] AR(1), ρ = 0.8, единичная стационарная
  дисперсия (k29_dist.sample_noise, σ = 1), × 0.1, дополнение нулями до [20,6].
Нормировка (одинаковая амплитуда эффекта):
  u0 = Z0/half; D_tree = (Zhat − Z0)/half; r = sqrt(mean(D[:8]²));
  λ = 0.02/r; Du = λ·D; u = u0 + Du; arm = mid + half·tanh(u).
  Валиден: r > 0, λ ≤ 8, max|Du[:8]| ≤ 0.08; иначе invalid с причиной (без
  нового направления/амплитуды). Нечисловые значения и RMS(Du[:8]) ≠ 0.02
  (допуск 1e−6) — технический отказ.
Пул: m = min(#valid tree, #valid flat); первые m tree (root→mid→leaf) и
  первые m flat (0→1→2); отбор до исходов; m = 0 — только noop.
Признаки строки (149): PCA средних 16 + PCA пути 16, s11n 11, elapsed/600,
  u0[:8] 48, Du[:8] 48, one-hot noop/root/mid/leaf/flat 5, центр/20 и
  ширина/20 интервала, средняя масса узла на префиксе, log2(λ)/4. Для flat
  интервал [0,20], масса 1; для noop и недопустимых строк поля поправки — 0.
Выбор: score = p(rescue) − p(harm), noop = 0 напрямую; в разрешённой семье
  максимум, только если score > 0.10 + 1e−6; при разности ≤ 1e−6 tree —
  leaf, mid, root; flat — меньший ID; иначе noop.
"""
import math
import sys

import numpy as np

ROWS = ("noop", "root", "mid", "leaf", "flat0", "flat1", "flat2")
TREE_ROWS = (1, 2, 3)
FLAT_ROWS = (4, 5, 6)
TREE_PREF = (3, 2, 1)           # при равенстве: leaf, mid, root
FLAT_PREF = (4, 5, 6)
N8 = 8
T = 20
E_NORM = 0.1
FLAT_SCALE = 0.1
RHO = 0.8
TARGET_RMS = 0.02
LAMBDA_MAX = 8.0
DU_MAX = 0.08
RMS_TOL = 1e-6
THRESH = 0.10
TIE = 1e-6
N_FEAT = 149
STREAMS = dict(tree=1, flat=2, fork=3, perm=4, init=5)
CONFIG = dict(rows=list(ROWS), e_norm=E_NORM, flat_scale=FLAT_SCALE, rho=RHO,
              target_rms=TARGET_RMS, lambda_max=LAMBDA_MAX, du_max=DU_MAX,
              rms_tol=RMS_TOL, threshold=THRESH, tie=TIE, n_feat=N_FEAT,
              streams=STREAMS, horizon=T, n_exec=N8)


class Technical(Exception):
    """Нечисловые значения/сбой RNG — технический отказ, не invalid."""


def rng(space, source_seed, task, init, call, stream):
    if stream not in STREAMS:
        raise ValueError(stream)
    ss = np.random.SeedSequence([31, int(space), int(source_seed), int(task),
                                 int(init), int(call), STREAMS[stream]])
    return np.random.Generator(np.random.PCG64(ss))


def tree_direction(gen):
    g = gen.standard_normal(6)
    n = float(np.linalg.norm(g))
    if not n > 0 or not math.isfinite(n):
        raise Technical("нулевая/нечисловая норма направления дерева")
    return (E_NORM * g / n).astype(np.float32), g


def flat_arrays(gen):
    """[3, 20, 6] float32: 0.1·AR(1) (σ = 1, ρ = 0.8) на шагах 0…7, нули."""
    import k29_dist as kd
    a = kd.sample_noise(gen, 3, sigma=1.0, rho=RHO)          # [3, 8, 6]
    out = np.zeros((3, T, 6), np.float32)
    out[:, :N8] = (FLAT_SCALE * a).astype(np.float32)
    return out


def normalize(torch, D):
    """D [T,6] (FP32) -> dict(valid, why, r, lam, Du). Технический отказ —
    исключение Technical."""
    D = torch.as_tensor(D, dtype=torch.float32)
    if not bool(torch.isfinite(D).all()):
        raise Technical("эффект предложения не конечен")
    r = torch.sqrt(torch.mean(D[:N8] ** 2))
    rv = float(r)
    if not math.isfinite(rv):
        raise Technical("r не конечно")
    if not rv > 0:
        return dict(valid=False, why="нулевой эффект", r=rv, lam=None,
                    Du=None)
    lam = torch.tensor(TARGET_RMS, dtype=torch.float32) / r
    lv = float(lam)
    if lv > LAMBDA_MAX:
        return dict(valid=False, why="λ > 8", r=rv, lam=lv, Du=None)
    Du = lam * D
    if not bool(torch.isfinite(Du).all()):
        raise Technical("Du не конечно")
    if float(Du[:N8].abs().max()) > DU_MAX:
        return dict(valid=False, why="max|Du| > 0.08", r=rv, lam=lv, Du=None)
    rms = float(torch.sqrt(torch.mean(Du[:N8] ** 2)))
    if not abs(rms - TARGET_RMS) <= RMS_TOL:
        raise Technical(f"RMS Du {rms} ≠ 0.02")
    return dict(valid=True, why=None, r=rv, lam=lv, Du=Du)


def commands(torch, Z0, Du, half, mid):
    """FP32 CPU: arm [8,6] = mid + half·tanh(Z0/half + Du)."""
    with torch.autocast("cpu", enabled=False):
        u = Z0 / half + Du
        return (mid + half * torch.tanh(u))[:N8]


def pool(torch, Z0, Zhat, flat, half, mid):
    """Z0 [T,6]; Zhat [3,T,6] (root, mid, leaf) или None; flat [3,T,6].
    -> dict(rows: 7 записей, valid [7], allowed [7], m, arm [7,8,6] | None).
    Строки tree — по сырому эффекту (Zhat − Z0)/half; flat — сам массив."""
    Z0 = torch.as_tensor(Z0, dtype=torch.float32)
    half = torch.as_tensor(half, dtype=torch.float32)
    mid = torch.as_tensor(mid, dtype=torch.float32)
    rows = [dict(name="noop", valid=True, why=None, r=None, lam=None,
                 Du=torch.zeros(T, 6))]
    for i, nm in enumerate(("root", "mid", "leaf")):
        if Zhat is None:
            rows.append(dict(name=nm, valid=False, why="нет предложения",
                             r=None, lam=None, Du=None))
            continue
        with torch.autocast("cpu", enabled=False):
            D = (torch.as_tensor(Zhat[i], dtype=torch.float32) - Z0) / half
        rows.append(dict(name=nm, **normalize(torch, D)))
    for i in range(3):
        rows.append(dict(name=f"flat{i}", **normalize(torch, flat[i])))
    valid = np.array([r["valid"] for r in rows])
    vt = [i for i in TREE_ROWS if valid[i]]
    vf = [i for i in FLAT_ROWS if valid[i]]
    m = min(len(vt), len(vf))
    allowed = np.zeros(7, bool)
    allowed[0] = True
    allowed[vt[:m]] = True
    allowed[vf[:m]] = True
    arm = []
    for i, r in enumerate(rows):
        arm.append(commands(torch, Z0, r["Du"], half, mid)
                   if (allowed[i] and i > 0) else torch.full((N8, 6),
                                                             float("nan")))
    return dict(rows=rows, valid=valid, allowed=allowed, m=m,
                arm=torch.stack(arm))


def features(ctx32, s11n, elapsed, u0, P, info):
    """-> [7,149] float32 сырых признаков (стандартизация — у селектора).
    ctx32 [32], s11n [11], elapsed (шаги), u0 [T,6], P — pool(), info —
    k31_readout.node_info (центр/ширина/масса узлов пути)."""
    X = np.zeros((7, N_FEAT), np.float32)
    base = np.concatenate([np.asarray(ctx32, np.float64).ravel(),
                           np.asarray(s11n, np.float64).ravel(),
                           [float(elapsed) / 600.0],
                           np.asarray(u0, np.float64)[:N8].ravel()])
    assert base.shape == (32 + 11 + 1 + 48,)
    for i in range(7):
        x = np.zeros(N_FEAT)
        x[:92] = base
        hot = np.zeros(5)
        hot[min(i, 4)] = 1.0
        x[140:145] = hot
        if i > 0 and P["allowed"][i]:
            r = P["rows"][i]
            x[92:140] = np.asarray(r["Du"][:N8], np.float64).ravel()
            if i in TREE_ROWS:
                nd = info[ROWS[i]]
                x[145:147] = [nd["center"] / T, nd["width"] / T]
                x[147] = nd["mass"]
            else:
                x[145:147] = [10.0 / T, 1.0]
                x[147] = 1.0
            x[148] = math.log2(r["lam"]) / 4.0
        X[i] = x.astype(np.float32)
    if not np.isfinite(X).all():
        raise Technical("признаки не конечны")
    return X


def choose(scores, allowed, family):
    """scores [7] (noop уже 0), allowed [7], family tree|flat -> индекс."""
    s = np.asarray(scores, np.float64)
    fam = TREE_ROWS if family == "tree" else FLAT_ROWS
    pref = TREE_PREF if family == "tree" else FLAT_PREF
    ok = [i for i in fam if allowed[i] and s[i] > THRESH + TIE]
    if not ok:
        return 0
    best = max(s[i] for i in ok)
    near = [i for i in ok if s[i] >= best - TIE]
    return next(i for i in pref if i in near)


def selftest():
    import torch
    # RNG: повтор точен, потоки независимы, глобальный RNG не сдвигается
    st = np.random.get_state()[1][:5].copy()
    ts = torch.random.get_rng_state().clone()
    a = tree_direction(rng(0, 0, 8, 3, 7, "tree"))[0]
    b = tree_direction(rng(0, 0, 8, 3, 7, "tree"))[0]
    c = tree_direction(rng(0, 0, 8, 3, 8, "tree"))[0]
    s1 = tree_direction(rng(1, 0, 8, 3, 7, "tree"))[0]
    assert np.array_equal(a, b) and not np.array_equal(a, c) and \
        not np.array_equal(a, s1)
    assert abs(float(np.linalg.norm(a)) - E_NORM) < 1e-6
    assert np.array_equal(np.random.get_state()[1][:5], st) and \
        torch.equal(torch.random.get_rng_state(), ts)
    fl = flat_arrays(rng(0, 0, 8, 3, 7, "flat"))
    assert fl.shape == (3, T, 6) and (fl[:, N8:] == 0).all()
    big = flat_arrays(rng(0, 0, 0, 0, 0, "flat"))
    assert np.abs(big[:, :N8]).max() < 1.0
    # нормировка
    half = torch.tensor([1.5, 2.0, 1.0, 0.6, 3.0, 1.5])
    mid = torch.tensor([0.1, -0.2, 0.0, 0.3, 0.0, 0.5])
    Z0 = torch.randn(T, 6, generator=torch.Generator().manual_seed(0)) * 0.5
    D = torch.zeros(T, 6)
    D[:N8, 0] = 0.05
    n = normalize(torch, D)
    assert n["valid"] and abs(float(torch.sqrt((n["Du"][:N8] ** 2).mean()))
                              - TARGET_RMS) < 1e-6
    assert not normalize(torch, torch.zeros(T, 6))["valid"]
    tiny = torch.zeros(T, 6)
    tiny[0, 0] = 1e-4
    assert normalize(torch, tiny)["why"] == "λ > 8"
    spike = torch.zeros(T, 6)
    spike[0, 0] = 1.0                             # вся энергия в одном
    assert normalize(torch, spike)["why"] == "max|Du| > 0.08"
    for bad in (float("nan"), float("inf"), -float("inf")):
        Db = D.clone()
        Db[3, 2] = bad
        try:
            normalize(torch, Db)
            raise AssertionError("нечисловое принято")
        except Technical:
            pass
    # хвост D (шаги 8…19) масштабируется, но в r не входит
    D2 = D.clone()
    D2[N8:] = 7.0
    n2 = normalize(torch, D2)
    assert n2["valid"] and n2["lam"] == n["lam"]
    # пул: m одинаков, отбор до исходов, tanh-рука, noop без поправки
    Zhat = torch.stack([Z0 + half * 0.05, Z0 + half * 1e-6,
                        Z0 + half * 0.04])         # mid: λ > 8 -> invalid
    P = pool(torch, Z0, Zhat, fl, half, mid)
    assert P["valid"].tolist() == [True, True, False, True, True, True, True]
    assert P["m"] == 2 and P["allowed"].tolist() == [True, True, False, True,
                                                     True, True, False]
    want = mid + half * torch.tanh(Z0 / half + P["rows"][1]["Du"])
    assert torch.equal(P["arm"][1], want[:N8])
    ch = P["arm"][1] - (mid + half * torch.tanh(Z0 / half))[:N8]
    assert float(ch.abs().max()) <= float((half * P["rows"][1]["Du"][:N8])
                                          .abs().max()) + 1e-6
    assert torch.isnan(P["arm"][0]).all() and torch.isnan(P["arm"][2]).all()
    P0 = pool(torch, Z0, None, fl, half, mid)
    assert P0["m"] == 0 and P0["allowed"].tolist() == [True] + [False] * 6
    # признаки
    info = dict(root=dict(center=10.0, width=20.0, mass=1.0),
                mid=dict(center=5.0, width=10.0, mass=0.6),
                leaf=dict(center=2.5, width=5.0, mass=0.4))
    X = features(np.ones(32), np.zeros(11), 16, Z0 / half, P, info)
    assert X.shape == (7, N_FEAT) and X.dtype == np.float32
    assert (X[0, 92:140] == 0).all() and X[0, 140] == 1 and X[0, 148] == 0
    assert X[2, 92:] .tolist() == [0.0] * 48 + [0, 0, 1, 0, 0] + [0.0] * 4
    assert X[4, 144] == 1 and X[4, 145] == 0.5 and X[4, 146] == 1.0 and \
        X[4, 147] == 1.0
    assert abs(X[3, 148] - math.log2(P["rows"][3]["lam"]) / 4) < 1e-6
    assert X[1, 145] == 0.5 and X[3, 147] == np.float32(0.4)
    assert (X[:, :92] == X[0, :92]).all()           # контекст общий
    # выбор
    al = np.array([True] * 7)
    assert choose([0, .5, .2, .3, .9, .1, .1], al, "tree") == 1
    assert choose([0, .5, .2, .3, .9, .1, .1], al, "flat") == 4
    assert choose([0, .1, .1, .1000005, 0, 0, 0], al, "tree") == 0
    assert choose([0, .3, .3 + 5e-7, .3, 0, 0, 0], al, "tree") == 3
    assert choose([0, .3, .3 + 5e-7, .2, 0, 0, 0], al, "tree") == 2
    assert choose([0, .9, .25, .2, 0, 0, 0],
                  np.array([True, False] + [True] * 5), "tree") == 2
    assert choose([0, .9, .2, .2, 0, 0, 0],
                  np.array([True, False] + [True] * 5), "tree") == 3
    assert choose([0, 0, 0, 0, .4, .4, .4], al, "flat") == 4
    print("самопроверка k31_candidates пройдена: локальный RNG (повтор точен, "
          "потоки/пространства независимы, глобальные не сдвигаются), "
          "плоский AR(1) с нулями после 8, нормировка (RMS 0.02, λ ≤ 8, "
          "max ≤ 0.08, хвост вне r, NaN/±Inf — технический отказ), пул с "
          "одинаковым m до исходов, tanh-рука, 149 признаков, правило выбора "
          "(порог 0.10, ничьи leaf→mid→root и меньший flat)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
