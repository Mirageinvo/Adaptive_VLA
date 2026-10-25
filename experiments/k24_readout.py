#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-24: сборка команды из выходов всех уровней дерева (multiresolution
readout, версия mr1).

Входы (одно наблюдение = строка батча):
  W0 [B,20,4], W1 [B,20,8] — мягкие веса корней и средних узлов (прежняя
     маршрутизация K-21/K-22; сумма детей W1 = W0, суммы строк 1);
  C0 [B,4,6], C1 [B,8,6] — коэффициенты руки общего readout MLP на
     окончательных hidden корней и средних (local_time 0, global_time —
     центр интервала / 20);
  Y2 [B,20,7] — прежний листовой raw readout до tanh (смешан через W2).
Сборка (float32, autocast выключен; QR reduced):
  Q0 = qr(W0), Q1 = qr(W1); proj(Q, X) = Q (Qᵀ X);
  G0 = W0 C0; G1 = W1 C1 − proj(Q0, W1 C1); G2 = Y2[:6] − proj(Q1, Y2[:6]);
  Z = G0 + G1 + G2; arm = mid + half·tanh(Z/half); grip_logit = Y2[6].
При полном ранге: P0 Z = G0, P1 Z = G0 + G1, компоненты попарно
ортогональны по времени (raw; после tanh тождества не утверждаются).
Гейт ранга в каждом forward: конечность и σ_min/σ_max ≥ 1e-5 для W0, W1
(SVD под no_grad); нарушение — технический отказ с диагностикой.
"""
import sys

import numpy as np

VERSION = "mr1"
RANK_TOL = 1e-5
FORMULA = ("Z = W0 C0 + (I-P0) W1 C1 + (I-P1) Y2[:6]; P_k = Q_k Q_k^T, "
           "Q_k = qr(W_k) reduced, float32; arm = mid + half*tanh(Z/half); "
           "grip_logit = Y2[6]")
T = 20
ID_TOL_CPU32, ID_TOL_MIXED = 1e-5, 5e-5


def build(torch):
    import k21_tree as kt
    TT = kt.build(torch)
    phi = TT["phi"]

    def proj(Q, X):
        return Q @ (Q.transpose(-1, -2) @ X)

    def rank_ratio(W):
        """σ_min/σ_max по строкам; строка с нечисловыми весами -> nan (SVD
        для неё не вызывается с нечисловыми значениями)."""
        with torch.no_grad():
            fin = torch.isfinite(W).flatten(1).all(1)
            s = torch.linalg.svdvals(torch.where(
                fin[:, None, None], W, torch.zeros_like(W)))
            r = s[..., -1] / s[..., 0].clamp_min(1e-30)
            return torch.where(fin, r, torch.full_like(r, float("nan")))

    def rank_gate(W, name, geometry=None):
        r = rank_ratio(W)
        bad = ~torch.isfinite(r) | (r < RANK_TOL)
        if bool(bad.any()):
            i = int(torch.nonzero(bad)[0])
            g = "" if geometry is None else \
                f"; геометрия строки {i}: {geometry(i)}"
            raise SystemExit(f"K-24: гейт ранга {name}: σmin/σmax = "
                             f"{float(r[i]):.3g} < {RANK_TOL} или нечисловые "
                             f"веса (строка {i}){g}")
        return r

    def node_coeffs(readout, h, iv):
        """h [B,n,H] — окончательные hidden узлов после action_expert.norm;
        iv [B,n,2] -> C [B,n,6] (тот же readout.ln и readout.mlp)."""
        B, n, H = h.shape
        c = iv.float().mean(-1)
        hf = readout.ln(h.float())
        inp = torch.cat([hf, phi(torch.zeros_like(c)), phi(c / T)], -1)
        return readout.mlp(inp)[..., :6]

    def assemble(W0, W1, C0, C1, Y2, L, U, geometry=None):
        dev = W0.device.type
        with torch.autocast(device_type=dev, enabled=False):
            W0f, W1f = W0.float(), W1.float()
            C0f, C1f, Y2f = C0.float(), C1.float(), Y2.float()
            r0 = rank_gate(W0f, "W0", geometry)
            r1 = rank_gate(W1f, "W1", geometry)
            Q0, _ = torch.linalg.qr(W0f, mode="reduced")
            Q1, _ = torch.linalg.qr(W1f, mode="reduced")
            G0 = W0f @ C0f
            G1r = W1f @ C1f
            G1 = G1r - proj(Q0, G1r)
            Y2a = Y2f[..., :6]
            G2 = Y2a - proj(Q1, Y2a)
            Z = G0 + G1 + G2
            Lt = torch.as_tensor(L, device=Z.device, dtype=Z.dtype)
            Ut = torch.as_tensor(U, device=Z.device, dtype=Z.dtype)
            mid, half = (Ut + Lt) / 2, (Ut - Lt) / 2
            arm = mid + half * torch.tanh(Z / half)
        return dict(arm=arm, grip_logit=Y2f[..., 6], Z=Z, G0=G0, G1=G1,
                    G2=G2, rank0=r0, rank1=r1, Q0=Q0, Q1=Q1)

    def identity_errors(o):
        """max относительные ошибки P0Z = G0, P1Z = G0+G1 и ортогональности,
        нормированные на max(1, max|компонент|) по строке."""
        with torch.no_grad():
            Z, G0, G1, G2 = (o[k].float() for k in ("Z", "G0", "G1", "G2"))
            sc = torch.stack([x.abs().flatten(1).max(1).values
                              for x in (Z, G0, G1, G2)], 1).max(1).values
            sc = sc.clamp_min(1.0)[:, None, None]
            e0 = (proj(o["Q0"], Z) - G0).abs() / sc
            e1 = (proj(o["Q1"], Z) - G0 - G1).abs() / sc
            orth = max(float(((a * b).sum(1).abs() / (sc[:, 0] ** 2)).max())
                       for a, b in ((G0, G1), (G0, G2), (G1, G2)))
            return dict(P0Z=float(e0.max()), P1Z=float(e1.max()),
                        orth=orth)

    def energy(o):
        with torch.no_grad():
            z = float((o["Z"] ** 2).sum())
            out = {}
            for k in ("G0", "G1", "G2"):
                g = o[k]
                out[f"rms_{k}"] = float(g.pow(2).mean().sqrt())
                out[f"share_{k}"] = float((g ** 2).sum()) / max(z, 1e-30)
            return out

    return dict(proj=proj, rank_ratio=rank_ratio, rank_gate=rank_gate,
                node_coeffs=node_coeffs, assemble=assemble,
                identity_errors=identity_errors, energy=energy, TT=TT)


# --- самопроверка (математика сборки) -----------------------------------------
def _geom(torch, TT, B, scale, gen, dtype):
    lg = torch.randn(B, 4, generator=gen, dtype=dtype) * scale
    v1 = torch.randn(B, 4, generator=gen, dtype=dtype) * scale
    v2 = torch.randn(B, 8, generator=gen, dtype=dtype) * scale
    iv = TT["root"](lg)
    W0 = TT["root_weights"](iv)
    iv2, s1 = TT["split"](iv, v1)
    W1 = TT["split_weights"](W0, iv, s1)
    iv3, s2 = TT["split"](iv2, v2)
    W2 = TT["split_weights"](W1, iv2, s2)
    return W0, W1, W2


def selftest():
    import torch
    R = build(torch)
    TT = R["TT"]
    gen = torch.Generator().manual_seed(20261025)
    L = np.array([-1.0, -0.5, -2.0, 0.0, -3.0, -1.0])
    U = np.array([1.0, 2.0, 0.5, 4.0, 3.0, 0.2])          # асимметричные
    worst = dict(cond0=0.0, cond1=0.0, nest=0.0, id32=0.0)
    for dtype in (torch.float64, torch.float32):
        for scale in (0.0, 1.0, 5.0, 20.0, 100.0):
            B = 64
            W0, W1, W2 = _geom(torch, TT, B, scale, gen, dtype)
            # 1. неотрицательность, суммы строк, вложенность по детям
            for W in (W0, W1, W2):
                assert (W >= -1e-12).all()
                assert torch.allclose(W.sum(-1), torch.ones_like(
                    W.sum(-1)), atol=1e-5)
            n1 = (W1.reshape(B, T, 4, 2).sum(-1) - W0).abs().max()
            n2 = (W2.reshape(B, T, 8, 2).sum(-1) - W1).abs().max()
            worst["nest"] = max(worst["nest"], float(max(n1, n2)))
            assert float(max(n1, n2)) < 1e-5
            # 2. полный ранг
            r0, r1 = R["rank_ratio"](W0.float()), R["rank_ratio"](W1.float())
            assert (r0 >= RANK_TOL).all() and (r1 >= RANK_TOL).all()
            worst["cond0"] = max(worst["cond0"], float((1 / r0).max()))
            worst["cond1"] = max(worst["cond1"], float((1 / r1).max()))
            C0 = torch.randn(B, 4, 6, generator=gen, dtype=dtype)
            C1 = torch.randn(B, 8, 6, generator=gen, dtype=dtype)
            Y2 = torch.randn(B, T, 7, generator=gen, dtype=dtype) * 3
            o = R["assemble"](W0, W1, C0, C1, Y2, L, U)
            # 3. проекторы идемпотентны, P0 P1 = P0
            P0 = o["Q0"] @ o["Q0"].transpose(-1, -2)
            P1 = o["Q1"] @ o["Q1"].transpose(-1, -2)
            assert torch.allclose(P0 @ P0, P0, atol=1e-5)
            assert torch.allclose(P0 @ P1, P0, atol=1e-5)
            # 4. тождества и ортогональность
            e = R["identity_errors"](o)
            worst["id32"] = max(worst["id32"], e["P0Z"], e["P1Z"], e["orth"])
            assert max(e.values()) < ID_TOL_CPU32, (scale, dtype, e)
            # 5. C1 не меняет P0Z; Y2 не меняет P0Z/P1Z
            o2 = R["assemble"](W0, W1, C0, C1 * 3 + 1, Y2, L, U)
            o3 = R["assemble"](W0, W1, C0, C1, Y2 * -2 + 0.5, L, U)
            sc = max(1.0, float(o["Z"].abs().max()), float(o2["Z"].abs().max()),
                     float(o3["Z"].abs().max()))
            assert float((R["proj"](o["Q0"], o["Z"]) - R["proj"](
                o2["Q0"], o2["Z"])).abs().max()) / sc < 2 * ID_TOL_CPU32
            for k in ("Q0", "Q1"):
                assert float((R["proj"](o[k], o["Z"]) - R["proj"](
                    o3[k], o3["Z"])).abs().max()) / sc < 2 * ID_TOL_CPU32
            # 6. асимметричные пределы, конечность
            lo = torch.as_tensor(L, dtype=o["arm"].dtype)
            hi = torch.as_tensor(U, dtype=o["arm"].dtype)
            assert torch.isfinite(o["arm"]).all() and \
                (o["arm"] >= lo - 1e-5).all() and (o["arm"] <= hi + 1e-5).all()
            # 7. схват — из листового raw, сборка руки его не меняет
            assert torch.equal(o["grip_logit"], o2["grip_logit"])
            assert torch.equal(o["grip_logit"], Y2[..., 6].float())
    # 2'. гейт на вырожденной матрице
    W0, W1, _ = _geom(torch, TT, 2, 1.0, gen, torch.float32)
    Wd = W0.clone()
    Wd[1, :, 3] = Wd[1, :, 2]
    try:
        R["assemble"](Wd, W1, torch.zeros(2, 4, 6), torch.zeros(2, 8, 6),
                      torch.zeros(2, T, 7), L, U)
        raise AssertionError("вырожденная W0 принята")
    except SystemExit as ex:
        assert "строка 1" in str(ex)
    Wn = W1.clone()
    Wn[0, 0, 0] = float("nan")
    try:
        R["assemble"](W0, Wn, torch.zeros(2, 4, 6), torch.zeros(2, 8, 6),
                      torch.zeros(2, T, 7), L, U)
        raise AssertionError("нечисловая W1 принята")
    except SystemExit:
        pass
    # gradcheck в double: та же геометрия формулами k21_tree (DMIN, RMIN,
    # τ), но без приведения к float32 — функции K-21 приводят логиты к
    # float32; сверка double-реализации с ними на тех же входах
    import k21_tree as ktm

    def geom64(lg, v1):
        d = ktm.DMIN + (T - 4 * ktm.DMIN) * torch.softmax(lg, -1)
        b = torch.cat([torch.zeros_like(d[:, :1]), torch.cumsum(d, -1)], -1)
        iv = torch.stack([b[:, :-1], b[:, 1:]], -1)
        x = torch.arange(T, dtype=lg.dtype) + 0.5
        Sg = torch.sigmoid((x[None, :, None] - iv[:, 1:, 0][:, None, :])
                           / ktm.TAU_ROOT)
        Sx = torch.cat([torch.ones_like(Sg[..., :1]), Sg,
                        torch.zeros_like(Sg[..., :1])], -1)
        W0 = Sx[..., :-1] - Sx[..., 1:]
        left, right = iv[..., 0], iv[..., 1]
        r = ktm.RMIN + (1 - 2 * ktm.RMIN) * torch.sigmoid(v1)
        sp = left + r * (right - left)
        tau = ktm.TAU_REL * (right - left)
        g = torch.sigmoid((x[None, :, None] - sp[:, None, :]) / tau[:, None])
        W1 = torch.stack([W0 * (1 - g), W0 * g], -1).reshape(
            lg.shape[0], T, -1)
        return W0, W1

    g2 = torch.Generator().manual_seed(7)
    lg = torch.randn(3, 4, generator=g2, dtype=torch.float64)
    v1 = torch.randn(3, 4, generator=g2, dtype=torch.float64)
    A0, A1 = geom64(lg, v1)
    iv = TT["root"](lg)
    B0 = TT["root_weights"](iv)
    _iv2, s1 = TT["split"](iv, v1)
    B1 = TT["split_weights"](B0, iv, s1)
    assert float((A0 - B0.double()).abs().max()) < 1e-5 and \
        float((A1 - B1.double()).abs().max()) < 1e-5

    def f(lg, v1, C0, C1, Y2):
        W0, W1 = geom64(lg, v1)
        Q0, _ = torch.linalg.qr(W0, mode="reduced")
        Q1, _ = torch.linalg.qr(W1, mode="reduced")
        G1 = W1 @ C1
        Z = W0 @ C0 + G1 - R["proj"](Q0, G1) + Y2[..., :6] - \
            R["proj"](Q1, Y2[..., :6])
        return torch.tanh(Z / 4.0)
    args = [torch.randn(1, 4, generator=g2, dtype=torch.float64) * 0.5,
            torch.randn(1, 4, generator=g2, dtype=torch.float64) * 0.5,
            torch.randn(1, 4, 6, generator=g2, dtype=torch.float64) * 0.3,
            torch.randn(1, 8, 6, generator=g2, dtype=torch.float64) * 0.3,
            torch.randn(1, T, 7, generator=g2, dtype=torch.float64) * 0.3]
    for x in args:
        x.requires_grad_(True)
    assert torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-5,
                                    rtol=1e-4)
    # тождества в float64 (та же формула, double)
    W0, W1 = geom64(lg, v1)
    C0 = torch.randn(3, 4, 6, generator=g2, dtype=torch.float64)
    C1 = torch.randn(3, 8, 6, generator=g2, dtype=torch.float64)
    Y2 = torch.randn(3, T, 7, generator=g2, dtype=torch.float64)
    Q0, _ = torch.linalg.qr(W0, mode="reduced")
    Q1, _ = torch.linalg.qr(W1, mode="reduced")
    G0 = W0 @ C0
    G1 = W1 @ C1 - R["proj"](Q0, W1 @ C1)
    G2 = Y2[..., :6] - R["proj"](Q1, Y2[..., :6])
    Z = G0 + G1 + G2
    assert float((R["proj"](Q0, Z) - G0).abs().max()) < 1e-12
    assert float((R["proj"](Q1, Z) - G0 - G1).abs().max()) < 1e-12
    print("самопроверка k24_readout пройдена: веса (неотрицательность, суммы, "
          "вложенность ≤ {nest:.1e}), ранг (cond W0 ≤ {cond0:.2f}, W1 ≤ "
          "{cond1:.1f}), гейт на вырожденной и нечисловой матрице, проекторы, "
          "тождества и ортогональность (≤ {id32:.1e}), независимость P0Z/P1Z "
          "от уточнений, пределы, схват из листьев (сборка float32, геометрия "
          "K-21; логиты до ±100); gradcheck и тождества в double на той же "
          "геометрии, сверенной с k21_tree".format(**worst))
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
