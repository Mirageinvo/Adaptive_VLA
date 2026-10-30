#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-28: геометрия поз руки (SO(3) и интерполяция) — одна реализация для
подготовки, модели и CPU-контроллера.

Соглашения (зафиксированы для всего K-28):
  поза X = (p, R): p — позиция EEF в метрах (мировые оси наблюдения), R —
  матрица вращения из axis-angle состояния (state8[3:6]);
  вращательная ошибка контроллера  Log(R_ref @ R_now.T)  (мировые оси);
  наложение вращательной поправки   Exp(δω) @ R_base;
  интерполяция  R(α) = R_L @ Exp(α · Log(R_L.T @ R_R)),  p(α) линейно.
Exp/Log устойчивы около 0 (ряды) и около π (симметричная часть); угол —
atan2(|v|/2, (tr − 1)/2) без acos. Всё — во float32/float64 входа, вне
autocast; явно поданный float64 сохраняется.
"""
import math
import sys

EPS_SMALL = 1e-4
NEAR_PI = 1e-3


def build(torch):
    def _dt(x):
        return torch.promote_types(x.dtype, torch.float32)

    def hat(w):
        z = torch.zeros_like(w[..., 0])
        return torch.stack([torch.stack([z, -w[..., 2], w[..., 1]], -1),
                            torch.stack([w[..., 2], z, -w[..., 0]], -1),
                            torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)

    def exp_so3(w):
        """[..., 3] -> [..., 3, 3]."""
        with torch.autocast(w.device.type, enabled=False):
            w = w.to(_dt(w))
            th2 = (w * w).sum(-1)
            th = torch.sqrt(th2.clamp_min(1e-30))
            small = th < EPS_SMALL
            ts = torch.where(small, torch.ones_like(th), th)
            A = torch.where(small, 1 - th2 / 6 + th2 * th2 / 120,
                            torch.sin(ts) / ts)
            B = torch.where(small, 0.5 - th2 / 24 + th2 * th2 / 720,
                            (1 - torch.cos(ts)) / (ts * ts))
            K = hat(w)
            I = torch.eye(3, dtype=w.dtype, device=w.device).expand(K.shape)
            return I + A[..., None, None] * K + B[..., None, None] * (K @ K)

    def log_so3(R):
        """[..., 3, 3] -> [..., 3] (угол в [0, π])."""
        with torch.autocast(R.device.type, enabled=False):
            R = R.to(_dt(R))
            v = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                             R[..., 0, 2] - R[..., 2, 0],
                             R[..., 1, 0] - R[..., 0, 1]], -1)
            tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
            c = ((tr - 1) / 2).clamp(-1.0, 1.0)
            s = torch.sqrt((v * v).sum(-1).clamp_min(1e-30)) / 2
            th = torch.atan2(s, c)
            small = th < EPS_SMALL
            ss = torch.where(small, torch.ones_like(s), s)
            f = torch.where(small, 0.5 + th * th / 12, th / (2 * ss))
            w = f[..., None] * v
            # около π: ось из симметричной части, знак — по v
            near = th > math.pi - NEAR_PI
            if bool(near.any()):
                Bm = (R + R.transpose(-1, -2)) / 2 - c[..., None, None] * \
                    torch.eye(3, dtype=R.dtype, device=R.device)
                d = torch.diagonal(Bm, dim1=-2, dim2=-1)
                k = d.argmax(-1)
                col = torch.gather(Bm, -1, k[..., None, None].expand(
                    *Bm.shape[:-1], 1))[..., 0]
                n = col / torch.sqrt((col * col).sum(-1, keepdim=True)
                                     .clamp_min(1e-30))
                sgn = torch.where((n * v).sum(-1) < 0, -1.0, 1.0).to(R.dtype)
                w_pi = n * (th * sgn)[..., None]
                w = torch.where(near[..., None], w_pi, w)
            return w

    def interp(pL, RL, pR, RR, alpha):
        """alpha [...] в [0, 1] -> (p, R)."""
        a = alpha.to(_dt(pL))
        p = pL + a[..., None] * (pR - pL)
        R = RL @ exp_so3(a[..., None] * log_so3(RL.transpose(-1, -2) @ RR))
        return p, R

    def rot_err(R_ref, R_now):
        return log_so3(R_ref @ R_now.transpose(-1, -2))

    def pose_of_state8(s8):
        """state8 [..., 8] (позиция, axis-angle, пальцы) -> (p, R)."""
        s8 = s8.to(_dt(s8))
        return s8[..., :3], exp_so3(s8[..., 3:6])

    def frob2(Ra, Rb):
        return ((Ra - Rb) ** 2).sum((-1, -2))

    return dict(hat=hat, exp_so3=exp_so3, log_so3=log_so3, interp=interp,
                rot_err=rot_err, pose_of_state8=pose_of_state8, frob2=frob2)


def selftest():
    import torch
    G = build(torch)
    g = torch.Generator().manual_seed(0)
    dt = torch.float64
    # Exp/Log взаимно обратны: обычные, около 0, около π
    w = torch.randn(64, 3, generator=g, dtype=dt)
    w = w / w.norm(dim=-1, keepdim=True) * torch.rand(64, 1, generator=g,
                                                      dtype=dt) * 3.0
    for ws in (w, w * 1e-6, w / w.norm(dim=-1, keepdim=True) *
               (math.pi - 1e-5)):
        R = G["exp_so3"](ws)
        assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3, dtype=dt)
                              .expand_as(R), atol=1e-10)
        assert torch.allclose(torch.det(R), torch.ones(64, dtype=dt))
        w2 = G["log_so3"](R)
        assert torch.isfinite(w2).all()
        assert torch.allclose(G["exp_so3"](w2), R, atol=1e-8), ws.norm(dim=-1)[:3]
    # ровно π
    R = G["exp_so3"](torch.tensor([[0.0, 0.0, math.pi]], dtype=dt))
    w2 = G["log_so3"](R)
    assert torch.isfinite(w2).all() and abs(float(w2.norm()) - math.pi) < 1e-6
    # интерполяция: концы, поступательное и известный поворот
    pL, pR = torch.zeros(1, 3, dtype=dt), torch.tensor([[1.0, 2.0, 3.0]], dtype=dt)
    RL = G["exp_so3"](torch.tensor([[0.1, 0.2, 0.3]], dtype=dt))
    Rz = G["exp_so3"](torch.tensor([[0.0, 0.0, 1.0]], dtype=dt))
    RR = Rz @ RL
    for a, want_p in ((0.0, pL), (1.0, pR)):
        p, Rm = G["interp"](pL, RL, pR, RR, torch.tensor([a], dtype=dt))
        assert torch.allclose(p, want_p) and torch.allclose(
            Rm, RL if a == 0 else RR, atol=1e-10)
    p, Rm = G["interp"](pL, RL, pR, RR, torch.tensor([0.25], dtype=dt))
    assert torch.allclose(p, pR * 0.25)
    # геодезическая: угол до RL — четверть полного
    e = G["rot_err"](Rm, RL).norm(dim=-1)
    assert abs(float(e) - 0.25) < 1e-8, float(e)
    # ошибка контроллера: R_ref = Exp(δ) R_now -> Log = δ
    d = torch.tensor([[0.05, -0.02, 0.1]], dtype=dt)
    assert torch.allclose(G["rot_err"](G["exp_so3"](d) @ RL, RL), d,
                          atol=1e-10)
    # gradcheck (невырожденные)
    wg = (torch.randn(4, 3, generator=g, dtype=dt) * 0.7).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: G["log_so3"](G["exp_so3"](x)),
                                    (wg,), eps=1e-6, atol=1e-5)
    a_ = torch.rand(4, generator=g, dtype=dt).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: G["interp"](
        pL.expand(4, 3), RL.expand(4, 3, 3), pR.expand(4, 3),
        RR.expand(4, 3, 3), x)[1], (a_,), eps=1e-6, atol=1e-5)
    # float32 вне autocast; bf16-autocast не меняет результат
    w32 = w.float()
    R32 = G["exp_so3"](w32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        R32b = G["exp_so3"](w32)
        L32b = G["log_so3"](R32b)
    assert R32b.dtype == torch.float32 and torch.equal(R32, R32b)
    assert L32b.dtype == torch.float32
    print("самопроверка k28_geometry пройдена: Exp/Log ортонормированы и "
          "взаимно обратны (обычные, около 0, около π и ровно π), "
          "интерполяция (концы, поступательная, геодезическая четверть), "
          "ошибка Log(R_ref R_nowᵀ), gradcheck float64, float32 вне autocast")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
