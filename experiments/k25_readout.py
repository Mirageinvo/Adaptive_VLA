#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-25: условный readout вдоль пути дерева (parents_cond24, версия cond1).

Признаки узлов после слоя 24: v_n = readout.ln(action_expert.norm(h_n)
.float()) — корни h[:, :4], средние h[:, 4:12], листья h[:, 12:28]. Родитель
среднего j — корень j//2, родитель листа l — средний l//2.
Корни (как K-24): C0_r = MLP([v_r, φ(0), φ(center_r/20)])[:6].
Новые общие модули (группа new):
  A_h: Linear(H, H, bias=False), A_a: Linear(6, H, bias=False) — нули;
  gate: Linear(H, 1), weight 0, bias log(0.1/0.9) — g = 0.1 в начале.
Ребёнок n с родителем p и прогнозом родителя z_p(t):
  parent_command = tanh(z_p(t)/half), half = (U − L)/2;
  u_n(t) = v_n + A_h(v_p) + A_a(parent_command);
  g_n = sigmoid(gate(v_n)) — скаляр на узел.
Средние: δ1_j(t) = MLP([u_j(t), φ(local_j(t)), φ(t/20)])[:6];
  z1_j(t) = C0_{j//2} + g1_j·δ1_j(t).
Листья: y2_l(t) = MLP([u_l(t), φ(local_l(t)), φ(t/20)]) (7 выходов, один
  вызов); z2_l(t) = z1_{l//2}(t) + g2_l·y2_l(t)[:6].
Сборка (float32): Z(t) = Σ_l W2(t,l)·z2_l(t); grip_logit(t) =
  Σ_l W2(t,l)·y2_l(t)[6]; arm = mid + half·tanh(Z/half).
Эквивалентная форма (вложенность W): Z = G0 + G1 + G2, G0 = W0 C0,
  G1 = Σ_j W1 g1 δ1, G2 = Σ_l W2 g2 y2[:6] (компоненты НЕ ортогональны).
"""
import math
import sys

import numpy as np

VERSION = "cond1"
FORMULA = ("z1_j = C0_{j//2} + g1_j*MLP([v_j + A_h v_p + A_a tanh(z_p/half), "
           "phi(local), phi(t/T)])[:6]; z2_l = z1_{l//2} + g2_l*y2_l[:6]; "
           "Z = sum_l W2 z2_l; grip = sum_l W2 y2_l[6]; "
           "arm = mid + half*tanh(Z/half)")
GATE_INIT = 0.1
T = 20
REL_TOL = 5e-5


def build(torch):
    import k21_tree as kt
    nn = torch.nn
    TT = kt.build(torch)
    phi, x_points = TT["phi"], TT["x_points"]
    MID_PARENT = torch.arange(8) // 2
    LEAF_PARENT = torch.arange(16) // 2

    class CondModules(nn.Module):
        def __init__(self, H):
            super().__init__()
            self.A_h = nn.Linear(H, H, bias=False)
            self.A_a = nn.Linear(6, H, bias=False)
            self.gate = nn.Linear(H, 1)
            with torch.no_grad():
                self.A_h.weight.zero_()
                self.A_a.weight.zero_()
                self.gate.weight.zero_()
                self.gate.bias.fill_(math.log(GATE_INIT / (1 - GATE_INIT)))

    def n_new_params(H):
        return H * H + 6 * H + H + 1

    def node_feats(readout, norm, h):
        return readout.ln(norm(h).float())

    def time_feats(iv):
        """iv [B,n,2] -> φ(local) [B,n,T,PHI] (clamp ±2), φ(t/T) [T,PHI]."""
        x = x_points(iv.device)
        c = iv.float().mean(-1)
        wd = (iv[..., 1] - iv[..., 0]).float().clamp_min(1e-6)
        lt = torch.clamp(2 * (x[None, None] - c[..., None]) / wd[..., None],
                         -2, 2)
        return phi(lt), phi(x / T)

    def mlp_t(readout, u, iv):
        """u [B,n,T,H] -> MLP([u, φ(local), φ(t/T)]) [B,n,T,7]."""
        pl, pg = time_feats(iv)
        B, n = u.shape[:2]
        inp = torch.cat([u, pl.to(u.dtype),
                         pg[None, None].expand(B, n, T, -1).to(u.dtype)], -1)
        return readout.mlp(inp)

    def half_mid(L, U, device):
        Lt = torch.as_tensor(L, device=device, dtype=torch.float32)
        Ut = torch.as_tensor(U, device=device, dtype=torch.float32)
        return (Ut - Lt) / 2, (Ut + Lt) / 2

    def cond_readout(readout, cm, norm, h0, h1, h2, iv0, iv1, iv2, W0, W1,
                     W2, L, U, gates_on=True):
        """-> dict(arm, grip_logit, Z, C0, g1, g2, delta1, y2, z1, z2,
        G0, G1, G2). Сырые суммы, tanh-контекст, смешивание и ограничение —
        float32."""
        dev = h0.device
        dt = dev.type

        def fp32():
            return torch.autocast(device_type=dt, enabled=False)
        half, mid = half_mid(L, U, dev)
        v0 = node_feats(readout, norm, h0)
        v1 = node_feats(readout, norm, h1)
        v2 = node_feats(readout, norm, h2)
        c0 = iv0.float().mean(-1)
        C0 = readout.mlp(torch.cat([v0, phi(torch.zeros_like(c0)),
                                    phi(c0 / T)], -1))[..., :6].float()
        mp, lp = MID_PARENT.to(dev), LEAF_PARENT.to(dev)
        # средние: контекст предка во float32, MLP — под внешним autocast
        with fp32():
            z1p = C0[:, mp][:, :, None, :].expand(-1, -1, T, -1).float()
            pc1 = torch.tanh(z1p / half)
        u1 = v1[:, :, None, :] + cm.A_h(v0[:, mp])[:, :, None, :] + \
            cm.A_a(pc1)
        delta1 = mlp_t(readout, u1, iv1)[..., :6]
        g1 = torch.sigmoid(cm.gate(v1))[..., 0]
        with fp32():
            delta1, g1 = delta1.float(), g1.float()
            if not gates_on:
                g1 = torch.zeros_like(g1)
            z1 = z1p + g1[..., None, None] * delta1
            z2p = z1[:, lp]
            pc2 = torch.tanh(z2p / half)
        # листья
        u2 = v2[:, :, None, :] + cm.A_h(v1[:, lp])[:, :, None, :] + \
            cm.A_a(pc2)
        y2 = mlp_t(readout, u2, iv2)
        g2 = torch.sigmoid(cm.gate(v2))[..., 0]
        with fp32():
            y2, g2 = y2.float(), g2.float()
            if not gates_on:
                g2 = torch.zeros_like(g2)
            z2 = z2p + g2[..., None, None] * y2[..., :6]
            W2f = W2.float()
            Z = torch.einsum("btl,bltc->btc", W2f, z2)
            grip = torch.einsum("btl,blt->bt", W2f, y2[..., 6])
            arm = mid + half * torch.tanh(Z / half)
            G0 = W0.float() @ C0
            G1 = torch.einsum("btj,bjtc->btc", W1.float(),
                              g1[..., None, None] * delta1)
            G2 = torch.einsum("btl,bltc->btc", W2f, g2[..., None, None]
                              * y2[..., :6])
        return dict(arm=arm, grip_logit=grip, Z=Z, C0=C0, g1=g1, g2=g2,
                    delta1=delta1, y2=y2, z1=z1, z2=z2, G0=G0, G1=G1, G2=G2)

    def recon_error(o):
        with torch.no_grad():
            s = torch.stack([x.abs().flatten(1).max(1).values for x in
                             (o["Z"], o["G0"], o["G1"], o["G2"])], 1)
            sc = s.max(1).values.clamp_min(1.0)[:, None, None]
            return float(((o["G0"] + o["G1"] + o["G2"] - o["Z"]).abs()
                          / sc).max())

    def stats(o):
        with torch.no_grad():
            def q(x):
                x = x.flatten().float()
                return [float(torch.quantile(x, p)) for p in (0.05, 0.5,
                                                              0.95)]
            return dict(g1_q=q(o["g1"]), g2_q=q(o["g2"]),
                        rms_G0=float(o["G0"].pow(2).mean().sqrt()),
                        rms_G1=float(o["G1"].pow(2).mean().sqrt()),
                        rms_G2=float(o["G2"].pow(2).mean().sqrt()),
                        rms_Z=float(o["Z"].pow(2).mean().sqrt()),
                        recon=recon_error(o))

    return dict(CondModules=CondModules, n_new_params=n_new_params,
                cond_readout=cond_readout, recon_error=recon_error,
                stats=stats, time_feats=time_feats, TT=TT,
                MID_PARENT=MID_PARENT, LEAF_PARENT=LEAF_PARENT)


def selftest():
    import torch
    import k21_tree as kt
    R = build(torch)
    TT = R["TT"]
    nn = torch.nn
    torch.manual_seed(0)
    H, B = 32, 3
    Rd = kt.build(torch)["Readout"](H, np.zeros(6), np.ones(6))
    for p in Rd.parameters():
        nn.init.normal_(p, std=0.3)
    norm = nn.LayerNorm(H)
    cm = R["CondModules"](H)
    L = np.array([-1.0, -2.0, -0.5, -1.0, -3.0, -1.0])
    U = np.array([2.0, 1.0, 0.5, 4.0, 3.0, 1.5])
    g = torch.Generator().manual_seed(3)
    lg = torch.randn(B, 4, generator=g) * 2
    iv = TT["root"](lg)
    W0 = TT["root_weights"](iv)
    iv1, s1 = TT["split"](iv, torch.randn(B, 4, generator=g))
    W1 = TT["split_weights"](W0, iv, s1)
    iv2, s2 = TT["split"](iv1, torch.randn(B, 8, generator=g))
    W2 = TT["split_weights"](W1, iv1, s2)
    h = torch.randn(B, 28, H, generator=g)

    def run(**kw):
        return R["cond_readout"](Rd, cm, norm, h[:, :4], h[:, 4:12],
                                 h[:, 12:28], iv, iv1, iv2, W0, W1, W2, L, U,
                                 **kw)
    # 1. вложенность W, индексы родителей
    assert torch.allclose(W1.reshape(B, T, 4, 2).sum(-1), W0, atol=1e-6)
    assert torch.allclose(W2.reshape(B, T, 8, 2).sum(-1), W1, atol=1e-6)
    assert R["MID_PARENT"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert R["LEAF_PARENT"].tolist() == [j for j in range(8) for _ in (0, 1)]
    o = run()
    assert o["delta1"].shape == (B, 8, T, 6) and o["y2"].shape == (B, 16, T, 7)
    assert all(torch.isfinite(v).all() for v in o.values())
    # 2. две формулы Z совпадают; нулевые гейты -> W0 C0
    assert R["recon_error"](o) < 1e-5, R["recon_error"](o)
    oz = run(gates_on=False)
    assert torch.allclose(oz["Z"], W0 @ oz["C0"], atol=1e-5)
    half = (torch.as_tensor(U) - torch.as_tensor(L)).float() / 2
    mid = (torch.as_tensor(U) + torch.as_tensor(L)).float() / 2
    assert torch.allclose(o["arm"], mid + half * torch.tanh(o["Z"] / half))
    assert (o["arm"] >= torch.as_tensor(L).float() - 1e-5).all() and \
        (o["arm"] <= torch.as_tensor(U).float() + 1e-5).all()
    gl = torch.einsum("btl,blt->bt", W2, o["y2"][..., 6])
    assert torch.allclose(o["grip_logit"], gl, atol=1e-6)
    # гейт в начале = 0.1
    assert torch.allclose(o["g1"], torch.full_like(o["g1"], GATE_INIT))
    # 3. при ненулевом A_a residual зависит от прогноза родителя
    with torch.no_grad():
        nn.init.normal_(cm.A_a.weight, std=0.5)
    a1 = run()
    with torch.no_grad():
        Rd.mlp[-1].bias[:6] += 0.7          # меняет C0 (прогноз корня)
    a2 = run()
    assert float((a1["delta1"] - a2["delta1"]).detach().abs().max()) > 1e-4, \
        "δ1 не зависит от прогноза родителя"
    with torch.no_grad():
        Rd.mlp[-1].bias[:6] -= 0.7
        cm.A_a.weight.zero_()
    # 4. градиенты новых модулей и readout конечны
    with torch.no_grad():
        nn.init.normal_(cm.A_h.weight, std=0.05)
    o = run()
    (o["arm"].sum() + o["grip_logit"].sum()).backward()
    for nm, p in list(cm.named_parameters()) + list(Rd.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(), nm
    assert float(cm.A_a.weight.grad.abs().sum()) > 0
    assert float(cm.gate.weight.grad.abs().sum()) > 0
    # 5. детерминированная инициализация новых модулей
    c1, c2 = R["CondModules"](H), R["CondModules"](H)
    assert all(torch.equal(a, b) for a, b in zip(c1.parameters(),
                                                c2.parameters()))
    assert sum(p.numel() for p in c1.parameters()) == R["n_new_params"](H)
    assert R["n_new_params"](768) == 595201
    # bf16 autocast: итоговые массивы float32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ob = run()
    assert all(ob[k].dtype == torch.float32 for k in (
        "Z", "arm", "grip_logit", "G0", "G1", "G2", "z1", "z2")), \
        {k: ob[k].dtype for k in ("Z", "arm", "grip_logit")}
    assert float((ob["Z"] - run()["Z"]).detach().abs().max()) < 0.1
    print("самопроверка k25_readout пройдена: вложенность и индексы "
          "родителей, две формулы Z, нулевые гейты -> W0C0, tanh и схват, "
          "g = 0.1 в начале, δ зависит от прогноза родителя при A_a ≠ 0, "
          "градиенты, детерминированная инициализация, 595201 новых "
          "параметров при H=768, float32 под bf16 autocast")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
