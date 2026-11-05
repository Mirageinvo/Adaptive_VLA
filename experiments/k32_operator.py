#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-32: операторы поправки T (поддерево), P (профиль), F (чанк) и общий
ограничитель. Всё — FP32 вне autocast.

Путь (как K-31): leaf* = argmax_l mean W2[:8, l], mid* = leaf*//2,
root* = mid*//2; r ∈ {0, 1, 2} = {root*, mid*, leaf*}; w_r — столбец
W0/W1/W2 узла; c_r = max(0.05, sqrt(mean(w_r[:8]²))).
T: a = 0.05·tanh(η) (6), e = a/c_r, правка узла k31_readout.edited
   (half·e к сырой команде узла, пересчёт потомков, вне поддерева —
   reference побитово); D_T = ((Zhat − Z0)/half)[:8].
P: D_P[t, j] = w_r(t)·a_j/c_r, t < 8 (тот же профиль, без пересчёта).
F: v_0 = η_0, v_t = 0.8·v_{t−1} + 0.6·η_t по η.reshape(8, 6);
   D_F = 0.05·tanh(v).
Ограничитель (общий): D1 = clip(D, ±0.08); rms = sqrt(mean(D1²)) по 8×6;
s = 1 при rms = 0, иначе min(1, 0.04/rms); Du = s·D1;
arm = mid + half·tanh(Z0[:8]/half + Du). η ≡ 0 -> готовая рука reference
без изменения арифметического пути. Схват — всегда reference.
"""
import math
import sys

import numpy as np

ALPHA = 0.05
C_FLOOR = 0.05
CLIP = 0.08
RMS_CAP = 0.04
RHO = 0.8
N8 = 8
OPS = ("T", "P", "F")
DIM = dict(T=6, P=6, F=48)
CONFIG = dict(alpha=ALPHA, c_floor=C_FLOOR, clip=CLIP, rms_cap=RMS_CAP,
              rho=RHO, n_exec=N8, dims=DIM)


class Technical(Exception):
    pass


def path_info(F1):
    """F1 — признаки одного наблюдения -> path (root, mid, leaf), w [3,20]
    (FP32 numpy), c [3], центр/ширина [3,2], масса [3], флаг floor [3]."""
    import k31_readout as kr
    W = [np.asarray(F1[k][0].detach().float().cpu().numpy(), np.float64)
         for k in ("W0", "W1", "W2")]
    for k_, Wk in zip(("W0", "W1", "W2"), W):
        if not np.isfinite(Wk).all() or (Wk < -1e-6).any() or \
                np.abs(Wk.sum(1) - 1).max() > 1e-3:
            raise Technical(f"геометрия {k_}: не неотрицательна/сумма ≠ 1")
    path = kr.path_of(W[2])
    w = np.stack([W[0][:, path[0]], W[1][:, path[1]], W[2][:, path[2]]])
    rms = np.sqrt((w[:, :N8] ** 2).mean(1))
    c = np.maximum(C_FLOOR, rms)
    info = kr.node_info(F1, path)
    cw = np.array([[info[k]["center"], info[k]["width"]]
                   for k in ("root", "mid", "leaf")])
    mass = np.array([info[k]["mass"] for k in ("root", "mid", "leaf")])
    return dict(path=path, w=w.astype(np.float32), c=c.astype(np.float32),
                cw=cw, mass=mass, floor=rms < C_FLOOR)


def ar_transform(eta48):
    e = np.asarray(eta48, np.float64).reshape(N8, 6)
    v = np.zeros_like(e)
    v[0] = e[0]
    s = math.sqrt(1 - RHO ** 2)
    for t in range(1, N8):
        v[t] = RHO * v[t - 1] + s * e[t]
    return v


def limit(torch, D):
    """D [8,6] -> (Du, info). Нечисловое — технический отказ."""
    D = torch.as_tensor(D, dtype=torch.float32)
    if D.shape != (N8, 6):
        raise Technical(f"D формы {tuple(D.shape)}")
    if not bool(torch.isfinite(D).all()):
        raise Technical("D не конечен")
    with torch.autocast("cpu", enabled=False):
        D1 = torch.clamp(D, -CLIP, CLIP)
        rms = torch.sqrt(torch.mean(D1 ** 2))
        rv = float(rms)
        s = 1.0 if rv == 0.0 else min(1.0, RMS_CAP / rv)
        Du = D1 * s
    return Du, dict(clipped=bool((D.abs() > CLIP).any()), rms_raw=rv,
                    scale=float(s), capped=s < 1.0,
                    rms=float(torch.sqrt(torch.mean(Du ** 2))),
                    max=float(Du.abs().max()))


def raw_effect(torch, op, r, eta, ref_cpu, pinfo, edited=None):
    """-> D [8,6] FP32 (CPU) и сырьё оператора для журнала."""
    eta = np.asarray(eta, np.float32)
    half = ref_cpu["half"]
    if op in ("T", "P"):
        a = ALPHA * np.tanh(eta.astype(np.float64))
        e = (a / float(pinfo["c"][r])).astype(np.float32)
        if op == "P":
            w = pinfo["w"][r][:N8].astype(np.float64)
            D = (w[:, None] * (a / float(pinfo["c"][r]))[None]).astype(
                np.float32)
            return torch.as_tensor(D), dict(e=e, profile=pinfo["w"][r][:N8])
        Zhat = edited(r, e)                          # [T,6] CPU FP32
        with torch.autocast("cpu", enabled=False):
            D = ((Zhat - ref_cpu["Z0"]) / half)[:N8]
        return D, dict(e=e, Zhat=Zhat.numpy())
    if op == "F":
        v = ar_transform(eta)
        D = (ALPHA * np.tanh(v)).astype(np.float32)
        return torch.as_tensor(D), dict(v=v.astype(np.float32))
    raise ValueError(op)


def command(torch, op, r, eta, ref_cpu, pinfo, edited=None):
    """-> (arm [8,6] FP32 CPU, журнал). η ≡ 0 — рука reference."""
    eta = np.asarray(eta, np.float32)
    if not np.isfinite(eta).all():
        raise Technical("η не конечен")
    if not eta.any():
        return ref_cpu["arm"].clone(), dict(zero=True, D=np.zeros((N8, 6),
                                                                  np.float32),
                                            Du=np.zeros((N8, 6), np.float32),
                                            lim=dict(clipped=False,
                                                     capped=False, scale=1.0,
                                                     rms=0.0, rms_raw=0.0,
                                                     max=0.0))
    D, raw = raw_effect(torch, op, r, eta, ref_cpu, pinfo, edited)
    Du, lim = limit(torch, D)
    with torch.autocast("cpu", enabled=False):
        arm = ref_cpu["mid"] + ref_cpu["half"] * torch.tanh(
            ref_cpu["Z0"][:N8] / ref_cpu["half"] + Du)
    if not bool(torch.isfinite(arm).all()):
        raise Technical("команда не конечна")
    return arm, dict(zero=False, D=np.asarray(D, np.float32),
                     Du=Du.numpy(), lim=lim, **raw)


def selftest():
    import torch
    import k21_model as km
    import k25_model as k25m
    import k29_model as k29m
    import k31_readout as kr
    k29m.no_tf32(torch)
    M = k29m.build(torch)
    E = kr.build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    mod = k25m.build(torch)["K25"](km.fake_bar(torch), L, U, seed=0).eval()
    with torch.no_grad():
        g = torch.Generator().manual_seed(3)
        for p in list(mod.cond.parameters()) + list(mod.readout.parameters()):
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    for p in mod.parameters():
        p.requires_grad_(False)

    def setup(dec):
        bt = km.fake_batch(torch, B=1, seed=4)
        with torch.no_grad():
            F, _s, _c = M["export"](mod, bt, 1, torch.as_tensor(
                km.state11(np.zeros((1, 8)))))
            ref = dec(F)
        h, m = E["half_mid"](dec._L, dec._U, "cpu")
        rc = dict(Z0=ref["Z"][0].float(), arm=ref["arm"][0, :N8].float(),
                  half=h, mid=m)
        pi = path_info(F)

        def edited(r, e):
            o = E["edited"](dec, F, ref, [kr.LEVELS[r]], [pi["path"][r]],
                            np.asarray(e)[None])
            return o["Zhat"][0].float()
        return F, ref, rc, pi, edited
    dec = M["Decoder"](mod).freeze()
    F, ref, rc, pi, edited = setup(dec)
    # 1. геометрия и c_r
    assert pi["path"][1] == pi["path"][2] // 2 and \
        pi["path"][0] == pi["path"][1] // 2
    assert (pi["c"] >= C_FLOOR).all() and pi["w"].shape == (3, 20)
    # 2. η = 0 -> reference побитово для всех операторов
    for op in OPS:
        arm, j = command(torch, op, 1, np.zeros(DIM[op]), rc, pi, edited)
        assert torch.equal(arm, rc["arm"]) and j["zero"], op
    rng = np.random.default_rng(0)
    # 3. leaf: T = P; A_a = 0: T = P на всех уровнях; A_a ≠ 0: root/mid ≠
    eta = rng.normal(size=6).astype(np.float32)
    DT = {r: raw_effect(torch, "T", r, eta, rc, pi, edited)[0]
          for r in range(3)}
    DP = {r: raw_effect(torch, "P", r, eta, rc, pi, edited)[0]
          for r in range(3)}
    assert float((DT[2] - DP[2]).abs().max()) <= 1e-6
    assert max(float((DT[r] - DP[r]).abs().max()) for r in (0, 1)) > 1e-6
    dec0 = M["Decoder"](mod).freeze()
    with torch.no_grad():
        dec0.cond.A_a.weight.zero_()
    F0, ref0, rc0, pi0, ed0 = setup(dec0)
    for r in range(3):
        a = raw_effect(torch, "T", r, eta, rc0, pi0, ed0)[0]
        b = raw_effect(torch, "P", r, eta, rc0, pi0, ed0)[0]
        assert float((a - b).abs().max()) <= 1e-6, r
    # 4. направление и величина e зависят от η, без нормировки длины
    e1 = raw_effect(torch, "T", 0, eta, rc, pi, edited)[1]["e"]
    e2 = raw_effect(torch, "T", 0, 0.5 * eta, rc, pi, edited)[1]["e"]
    assert np.linalg.norm(e2) < np.linalg.norm(e1)
    assert np.abs(e1).max() <= 1.0 + 1e-6
    # 5. ограничитель: границы для любых конечных η, большие значения
    for scale in (1e-3, 1.0, 10.0, 1e3):
        for op in OPS:
            et = (scale * rng.normal(size=DIM[op])).astype(np.float32)
            arm, j = command(torch, op, int(rng.integers(3)), et, rc, pi,
                             edited)
            assert j["lim"]["max"] <= CLIP + 1e-7 and \
                j["lim"]["rms"] <= RMS_CAP + 1e-7, (op, scale)
            d = ((arm - rc["arm"]) / rc["half"]).abs()
            assert float(d.max()) <= float(np.abs(j["Du"]).max()) + 1e-6
    Dbig = torch.full((N8, 6), 0.5)
    Du, info = limit(torch, Dbig)
    assert info["clipped"] and info["capped"] and \
        abs(info["rms"] - RMS_CAP) < 1e-6
    for bad in (np.nan, np.inf, -np.inf):
        try:
            command(torch, "F", 0, np.full(48, bad, np.float32), rc, pi,
                    edited)
            raise AssertionError("нечисловое η принято")
        except Technical:
            pass
        try:
            limit(torch, torch.full((N8, 6), bad))
            raise AssertionError("нечисловое D принято")
        except Technical:
            pass
    # 6. F: изменение хвоста η меняет ранние команды через общий RMS-cap
    eF = (3.0 * rng.normal(size=48)).astype(np.float32)
    a1 = command(torch, "F", 0, eF, rc, pi, edited)[0]
    eF2 = eF.copy()
    eF2[42:] *= 5.0
    a2 = command(torch, "F", 0, eF2, rc, pi, edited)[0]
    assert float((a1[0] - a2[0]).abs().max()) > 0, \
        "хвост F не влияет на ранние команды при общем cap"
    # 7. T меняет только поддерево; схват вне оператора
    o = E["edited"](dec, F, ref, ["mid"], [pi["path"][1]], np.ones((1, 6)) *
                    0.1)
    _r, mids, leaves = kr.descendants("mid", pi["path"][1])
    ol = [x for x in range(16) if x not in leaves]
    assert torch.equal(o["z2"][0, ol], ref["z2"][0, ol])
    print("самопроверка k32_operator пройдена: путь и c_r, η = 0 -> "
          "reference побитово (T/P/F), leaf T = P, при A_a = 0 T = P на всех "
          "уровнях, при A_a ≠ 0 root/mid различаются, e без нормировки "
          "длины (|e| ≤ 1), границы max/RMS при любых η, clip и cap, "
          "NaN/±Inf — отказ, хвост F меняет ранние команды через общий cap, "
          "вне поддерева — побитово")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
