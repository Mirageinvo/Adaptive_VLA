#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-30: фиксированная геометрия дерева, план корней (hard/soft, STE) и
непрерывные уточнения детей. Всё — float32 с явно выключенным autocast.

Геометрия (одинакова у H и C): корни [0,5],[5,10],[10,15],[15,20]; каждое
деление пополам (середины 2.5, листья 1.25); точки t + 0.5; веса W0/W1/W2 —
формулы K-21 (fixed: τ_root = 0.5, τ = 0.1·ширина родителя); буферы.
План (после слоя 8, по h8 до изменения):
  v0 = root_ln(h8.float()); logits = root_logits(v0) [B,4,128];
  p = softmax(logits); ids = argmax (ничья — меньший ID);
  b_soft = p @ P; b_hard = P[ids];
  C (soft): b_used = b_soft; H (hard): b_used = b_hard, при обучении
  градиент — по b_soft (STE: прямой проход — точная копия b_hard).
Уточнения (K-25 cond, корень из плана):
  C0 = half·atanh(clip(b_used, −1+1e−3, 1−1e−3));
  u1_j = v1_j + A_h(v0_{j//2}) + A_a(tanh(C0_{j//2}/half));
  z1_j = C0_{j//2} + σ(gate(v1_j))·MLP([u1_j, φ(local), φ(t/20)])[:6];
  u2_l = v2_l + A_h(v1_{l//2}) + A_a(tanh(z1_{l//2}/half));
  y2_l = MLP([u2_l, φ(local), φ(t/20)]); z2_l = z1_{l//2} + σ(gate(v2_l))·y2_l[:6];
  Z = Σ_l W2 z2_l; arm = mid + half·tanh(Z/half); grip = Σ_l W2 y2_l[6].
  G0 = W0 C0, G1 = Σ W1 g1 δ1, G2 = Σ W2 g2 y2[:6]; Z = G0 + G1 + G2.
"""
import sys

import numpy as np

EPS_PLAN = 1e-3
T = 20
VERSION = "k30_hybrid1"


def build(torch):
    import k21_tree as kt
    TT = kt.build(torch)
    phi, x_points = TT["phi"], TT["x_points"]
    MID_PARENT = torch.arange(8) // 2
    LEAF_PARENT = torch.arange(16) // 2

    def fp32(dev):
        return torch.autocast(device_type=dev.type, enabled=False)

    def geometry():
        """-> dict IV0 [4,2], IV1 [8,2], IV2 [16,2], W0 [20,4], W1 [20,8],
        W2 [20,16] (float32, CPU)."""
        with torch.no_grad():
            iv0 = TT["root"](torch.zeros(1, 4), True)
            w0 = TT["root_weights"](iv0)
            iv1, s1 = TT["split"](iv0, torch.zeros(1, 4), True)
            w1 = TT["split_weights"](w0, iv0, s1)
            iv2, s2 = TT["split"](iv1, torch.zeros(1, 8), True)
            w2 = TT["split_weights"](w1, iv1, s2)
        return dict(IV0=iv0[0].float(), IV1=iv1[0].float(),
                    IV2=iv2[0].float(), W0=w0[0].float(), W1=w1[0].float(),
                    W2=w2[0].float())

    class STE(torch.autograd.Function):
        """Прямой проход — копия hard; обратный — градиент в soft."""
        @staticmethod
        def forward(ctx, soft, hard):
            return hard.clone()

        @staticmethod
        def backward(ctx, g):
            return g, None

    def plan(root_ln, root_logits, h8, P, mode, training):
        """-> dict(v0, logits, p, ids, b_soft, b_hard, b_used)."""
        dev = h8.device
        dt = root_ln.weight.dtype                     # float32 в модели
        with fp32(dev):
            v0 = root_ln(h8.to(dt))
            logits = root_logits(v0)
            p = torch.softmax(logits, -1)
            ids = torch.argmax(logits, -1)            # первый максимум
            Pf = P.to(dt)
            b_soft = p @ Pf
            b_hard = Pf[ids]
            if mode == "soft":
                b_used = b_soft
            elif mode == "hard":
                b_used = STE.apply(b_soft, b_hard) if training and \
                    torch.is_grad_enabled() else b_hard
            else:
                raise ValueError(mode)
        return dict(v0=v0, logits=logits, p=p, ids=ids, b_soft=b_soft,
                    b_hard=b_hard, b_used=b_used)

    def half_mid(L, U, device):
        Lt = torch.as_tensor(np.asarray(L)[:6], device=device,
                             dtype=torch.float32)
        Ut = torch.as_tensor(np.asarray(U)[:6], device=device,
                             dtype=torch.float32)
        return (Ut - Lt) / 2, (Ut + Lt) / 2

    def mlp_t(readout, u, iv):
        pl, pg = time_feats(iv)
        B, n = u.shape[:2]
        inp = torch.cat([u, pl.to(u.dtype),
                         pg[None, None].expand(B, n, T, -1).to(u.dtype)], -1)
        return readout.mlp(inp)

    def time_feats(iv):
        x = x_points(iv.device)
        c = iv.float().mean(-1)
        wd = (iv[..., 1] - iv[..., 0]).float().clamp_min(1e-6)
        lt = torch.clamp(2 * (x[None, None] - c[..., None]) / wd[..., None],
                         -2, 2)
        return phi(lt), phi(x / T)

    def readout(readout_mod, cm, v0, b_used, x1, x2, G, L, U):
        """x1/x2 — action_expert.norm(h).float() средних/листьев (норма
        применена ОДИН раз вызывающим); G — геометрия на устройстве."""
        dev = x1.device
        B = x1.shape[0]
        with fp32(dev):
            dt = readout_mod.ln.weight.dtype          # float32 в модели
            half, mid = half_mid(L, U, dev)
            half, mid = half.to(dt), mid.to(dt)
            v1 = readout_mod.ln(x1.to(dt))
            v2 = readout_mod.ln(x2.to(dt))
            bu = b_used.to(dt).clamp(-1 + EPS_PLAN, 1 - EPS_PLAN)
            C0 = half * torch.atanh(bu)                       # [B,4,6]
            iv1 = G["IV1"][None].expand(B, -1, -1)
            iv2 = G["IV2"][None].expand(B, -1, -1)
            W0 = G["W0"].to(dt)[None].expand(B, -1, -1)
            W1 = G["W1"].to(dt)[None].expand(B, -1, -1)
            W2 = G["W2"].to(dt)[None].expand(B, -1, -1)
            mp, lp = MID_PARENT.to(dev), LEAF_PARENT.to(dev)
            z1p = C0[:, mp][:, :, None, :].expand(-1, -1, T, -1)
            pc1 = torch.tanh(z1p / half)
            u1 = v1[:, :, None, :] + cm.A_h(v0[:, mp].to(dt))[:, :, None, :] \
                + cm.A_a(pc1)
            delta1 = mlp_t(readout_mod, u1, iv1)[..., :6]
            g1 = torch.sigmoid(cm.gate(v1))[..., 0]
            z1 = z1p + g1[..., None, None] * delta1
            z2p = z1[:, lp]
            pc2 = torch.tanh(z2p / half)
            u2 = v2[:, :, None, :] + cm.A_h(v1[:, lp])[:, :, None, :] + \
                cm.A_a(pc2)
            y2 = mlp_t(readout_mod, u2, iv2)
            g2 = torch.sigmoid(cm.gate(v2))[..., 0]
            z2 = z2p + g2[..., None, None] * y2[..., :6]
            Z = torch.einsum("btl,bltc->btc", W2, z2)
            grip = torch.einsum("btl,blt->bt", W2, y2[..., 6])
            arm = mid + half * torch.tanh(Z / half)
            G0 = W0 @ C0
            G1 = torch.einsum("btj,bjtc->btc", W1, g1[..., None, None]
                              * delta1)
            G2 = torch.einsum("btl,bltc->btc", W2, g2[..., None, None]
                              * y2[..., :6])
        return dict(arm=arm, grip_logit=grip, Z=Z, C0=C0, g1=g1, g2=g2,
                    delta1=delta1, y2=y2, z1=z1, z2=z2, G0=G0, G1=G1, G2=G2)

    return dict(geometry=geometry, plan=plan, readout=readout, STE=STE,
                half_mid=half_mid, TT=TT, MID_PARENT=MID_PARENT,
                LEAF_PARENT=LEAF_PARENT)


def np_plan(logits, P, mode):
    """CPU-пересборка плана: ids = argmax, hard = P[ids], soft = softmax@P."""
    lg = np.asarray(logits, np.float64)
    ids = np.argmax(lg, -1)
    P = np.asarray(P, np.float64)
    e = np.exp(lg - lg.max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)
    return ids, (P[ids] if mode == "hard" else p @ P)


def np_readout(f, L, U):
    """CPU-пересборка C0, z1, z2, Z, G0..G2, руки и логита схвата из
    журнала (float64). f: b_used [N,4,6], g1 [N,8], g2 [N,16], delta1
    [N,8,T,6], y2 [N,16,T,7], W0/W1/W2."""
    L = np.asarray(L, np.float64)[:6]
    U = np.asarray(U, np.float64)[:6]
    half, mid = (U - L) / 2, (U + L) / 2
    bu = np.clip(np.asarray(f["b_used"], np.float64), -1 + EPS_PLAN,
                 1 - EPS_PLAN)
    C0 = half * np.arctanh(bu)
    mp, lp = np.arange(8) // 2, np.arange(16) // 2
    z1 = C0[:, mp][:, :, None, :] + f["g1"][..., None, None] * f["delta1"]
    z2 = z1[:, lp] + f["g2"][..., None, None] * f["y2"][..., :6]
    Z = np.einsum("ntl,nltc->ntc", f["W2"], z2)
    G0 = f["W0"] @ C0
    G1 = np.einsum("ntj,njtc->ntc", f["W1"], f["g1"][..., None, None] *
                   f["delta1"])
    G2 = np.einsum("ntl,nltc->ntc", f["W2"], f["g2"][..., None, None] *
                   f["y2"][..., :6])
    arm = mid + half * np.tanh(Z / half)
    grip = np.einsum("ntl,nlt->nt", f["W2"], f["y2"][..., 6])
    return dict(C0=C0, Z=Z, G0=G0, G1=G1, G2=G2, arm=arm, grip=grip,
                half=half, mid=mid)


def selftest():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    R = build(torch)
    G = R["geometry"]()
    # геометрия: интервалы, вложенность, сумма листьев 1, неотрицательность
    assert G["IV0"].tolist() == [[0, 5], [5, 10], [10, 15], [15, 20]]
    assert torch.allclose(G["IV1"][:, 1] - G["IV1"][:, 0], torch.full((8,),
                                                                      2.5))
    assert torch.allclose(G["IV2"][:, 1] - G["IV2"][:, 0], torch.full((16,),
                                                                      1.25))
    for W in (G["W0"], G["W1"], G["W2"]):
        assert (W >= 0).all() and torch.allclose(W.sum(-1), torch.ones(20),
                                                 atol=1e-6)
    assert torch.allclose(G["W2"].reshape(20, 8, 2).sum(-1), G["W1"],
                          atol=1e-6)
    assert torch.allclose(G["W1"].reshape(20, 4, 2).sum(-1), G["W0"],
                          atol=1e-6)
    # план: hard = P[ids], soft = p@P, one-hot -> совпадают, ничья — меньший
    nn = torch.nn
    H, K = 16, 128
    g = torch.Generator().manual_seed(0)
    P = (torch.rand(K, 6, generator=g) * 2 - 1).float()
    ln, lg = nn.LayerNorm(H), nn.Linear(H, K)
    h8 = torch.randn(3, 4, H, generator=g)
    for mode in ("hard", "soft"):
        o = R["plan"](ln, lg, h8, P, mode, False)
        assert torch.equal(o["b_hard"], P[o["ids"]])
        assert torch.allclose(o["b_soft"], o["p"] @ P)
        assert o["b_used"].dtype == torch.float32
    with torch.no_grad():
        lg.weight.zero_()
        lg.bias.zero_()
        lg.bias[7] = 50.0
        lg.bias[3] = 50.0                                   # точная ничья
    o = R["plan"](ln, lg, h8, P, "hard", False)
    assert (o["ids"] == 3).all()
    with torch.no_grad():
        lg.bias.zero_()
        lg.bias[5] = 1e4
    o = R["plan"](ln, lg, h8, P, "soft", False)
    assert torch.allclose(o["b_soft"], o["b_hard"], atol=1e-6)
    # под внешним bf16 autocast внутренний расчёт остаётся FP32
    lg2 = nn.Linear(H, K)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        o = R["plan"](ln, lg2, h8, P, "soft", False)
    assert o["logits"].dtype == torch.float32 and \
        o["b_used"].dtype == torch.float32
    # STE: прямой — hard побитово, обратный — как у soft
    lg3 = nn.Linear(H, K).double()
    ln3 = nn.LayerNorm(H).double()
    w = torch.randn(3, 4, 6, generator=g).double()
    o = R["plan"](ln3, lg3, h8.double(), P.double(), "hard", True)
    assert torch.equal(o["b_used"], o["b_hard"])
    (o["b_used"] * w).sum().backward()
    gh = lg3.weight.grad.clone()
    lg3.weight.grad = None
    o2 = R["plan"](ln3, lg3, h8.double(), P.double(), "soft", True)
    (o2["b_used"] * w).sum().backward()
    assert torch.allclose(gh, lg3.weight.grad)
    Pg = P.double().clone().requires_grad_(True)
    o = R["plan"](ln3, lg3, h8.double(), Pg, "hard", True)
    (o["b_used"] * w).sum().backward()
    assert Pg.grad is not None   # P — буфер в модели; здесь проверка пути
    # readout: Z = G0+G1+G2, g = 0 -> mid + half·tanh(W0 C0 / half),
    # асимметричные L/U, независимость от соседей, gradcheck float64
    import k25_readout as kr
    cm = kr.build(torch)["CondModules"](H).double()
    import k21_tree as kt
    rd = kt.build(torch)["Readout"](H, np.zeros(6), np.ones(6)).double()
    with torch.no_grad():
        for p_ in list(cm.parameters()) + list(rd.parameters()):
            p_.add_(torch.randn(p_.shape, generator=g, dtype=p_.dtype) * 0.1)
    Gd = {k: v.double() for k, v in G.items()}
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    v0 = torch.randn(2, 4, H, generator=g).double()
    bu = (torch.rand(2, 4, 6, generator=g).double() * 1.8 - 0.9)
    x1 = torch.randn(2, 8, H, generator=g).double()
    x2 = torch.randn(2, 16, H, generator=g).double()

    def run(bu_, x1_=x1):
        out = R["readout"](rd, cm, v0, bu_, x1_, x2, Gd, L, U)
        return out
    o = run(bu)
    assert torch.allclose(o["G0"] + o["G1"] + o["G2"], o["Z"], atol=1e-10)
    hf, md = R["half_mid"](L, U, "cpu")
    with torch.no_grad():
        cm.gate.weight.zero_()
        cm.gate.bias.fill_(-1e4)
    o0 = run(bu)
    C0 = hf.double() * torch.atanh(bu.clamp(-1 + EPS_PLAN, 1 - EPS_PLAN))
    want = md.double() + hf.double() * torch.tanh((Gd["W0"] @ C0) / hf.double())
    assert torch.allclose(o0["arm"], want, atol=1e-9)
    assert (o["arm"] >= torch.as_tensor(L) - 1e-9).all() and \
        (o["arm"] <= torch.as_tensor(U) + 1e-9).all()
    bu2 = bu.clone()
    bu2[1] += 0.05
    oa, ob = run(bu), run(bu2)               # меняется только среда 1
    assert torch.allclose(oa["Z"][0], ob["Z"][0], atol=0, rtol=0) and \
        not torch.allclose(oa["Z"][1], ob["Z"][1])
    with torch.no_grad():
        cm.gate.bias.fill_(0.3)
    bg = bu[:1].clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda b_: R["readout"](rd, cm, v0[:1], b_, x1[:1], x2[:1], Gd, L,
                                U)["arm"].sum(), (bg,), eps=1e-6, atol=1e-5)
    # CPU-пересборка журнала = torch
    f = {k: o[k].detach().numpy() for k in ("g1", "g2", "delta1", "y2")}
    f.update(b_used=bu.numpy(), W0=Gd["W0"].numpy()[None].repeat(2, 0),
             W1=Gd["W1"].numpy()[None].repeat(2, 0),
             W2=Gd["W2"].numpy()[None].repeat(2, 0))
    rec = np_readout(f, L, U)
    assert np.abs(rec["Z"] - o["Z"].detach().numpy()).max() < 1e-6
    assert np.abs(rec["arm"] - o["arm"].detach().numpy()).max() < 1e-6
    ids, bh = np_plan(np.array([[[0.0, 2.0, 2.0]]]), np.eye(3), "hard")
    assert ids[0, 0] == 1
    print("самопроверка k30_readout пройдена: фиксированная геометрия "
          "(интервалы, вложенность, суммы, неотрицательность), hard = P[ids], "
          "soft = p@P, one-hot -> совпадают, ничья — меньший ID, FP32 под "
          "bf16 autocast, STE (прямой — hard, обратный — как soft), "
          "Z = G0+G1+G2, g = 0 -> mid + half·tanh(W0·C0/half), пределы L/U, "
          "независимость сред, gradcheck float64, CPU-пересборка")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
