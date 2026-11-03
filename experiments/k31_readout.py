#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-31: точки вмешательства в условный readout parents_cond24 (K-25).

Номинальный путь — FP32 reference K-29 (k29_model.Decoder, autocast выкл.):
C0, z1, z2, delta1, y2, g1, g2, Z0, логит схвата. Здесь — та же сборка с
одним ненулевым узлом e (на кандидата ровно один):
  C0*_r       = C0_r + half·e0_r
  z1*_j(t)    = C0*_{j//2} + g1_j·delta1*_j(t) + half·e1_j
  z2*_l(t)    = z1*_{l//2}(t) + g2_l·y2*_l(t)[:6] + half·e2_l
  Zhat(t)     = Σ_l W2(t,l)·z2*_l(t)
delta1*, y2* — прежние MLP/A_h/A_a с ИЗМЕНЁННОЙ численной командой родителя
(tanh(z_p/half)); hidden, гейты g и геометрия W/iv не пересчитываются.
Вне поддерева (сам узел и потомки) значения подставляются из reference
побитово (torch.where). Схват — всегда логит reference: изменённый
y2*[..., 6] не используется. Отсутствие вмешательства — reference без
изменения арифметического пути (вызывающий не зовёт edited для noop).

Ветвь: leaf* = argmax_l mean_{t<8} W2[t,l] (ничья — меньший индекс),
mid* = leaf*//2, root* = mid*//2; для узла — центр/ширина интервала и
средняя масса W на первых 8 шагах.
"""
import sys

import numpy as np

N8 = 8
T = 20
LEVELS = ("root", "mid", "leaf")


def path_of(W2):
    """W2 [T,16] (одно наблюдение) -> (root, mid, leaf); ничья — меньший."""
    m = np.asarray(W2, np.float64)[:N8].mean(0)
    leaf = int(np.argmax(m))                    # np.argmax: первый максимум
    mid = leaf // 2
    return mid // 2, mid, leaf


def descendants(level, idx):
    """-> (roots, mids, leaves): узел и все его потомки."""
    if level == "root":
        return [idx], [2 * idx, 2 * idx + 1], list(range(4 * idx, 4 * idx + 4))
    if level == "mid":
        return [], [idx], [2 * idx, 2 * idx + 1]
    if level == "leaf":
        return [], [], [idx]
    raise ValueError(level)


def node_info(F1, path):
    """F1 — признаки одного наблюдения (тензоры [1,…]) -> по уровням:
    центр, ширина интервала и средняя масса W на префиксе 8."""
    out = {}
    for lev, idx, iv, W in (("root", path[0], "iv0", "W0"),
                            ("mid", path[1], "iv1", "W1"),
                            ("leaf", path[2], "iv2", "W2")):
        a = np.asarray(F1[iv][0, idx].detach().float().cpu().numpy(),
                       np.float64)
        w = np.asarray(F1[W][0, :N8, idx].detach().float().cpu().numpy(),
                       np.float64)
        out[lev] = dict(index=int(idx), center=float(a.mean()),
                        width=float(a[1] - a[0]), mass=float(w.mean()))
    return out


def build(torch):
    import k25_readout as kr
    R = kr.build(torch)
    phi = R["TT"]["phi"]
    MP = R["MID_PARENT"]
    LP = R["LEAF_PARENT"]

    def half_mid(L, U, device):
        Lt = torch.as_tensor(L, device=device, dtype=torch.float32)
        Ut = torch.as_tensor(U, device=device, dtype=torch.float32)
        return (Ut - Lt) / 2, (Ut + Lt) / 2

    def mlp_t(readout, u, iv):
        pl, pg = R["time_feats"](iv)
        B, n = u.shape[:2]
        inp = torch.cat([u, pl.to(u.dtype),
                         pg[None, None].expand(B, n, T, -1).to(u.dtype)], -1)
        return readout.mlp(inp)

    def masks(levels, idxs, device):
        """[N] уровней/индексов -> маски поддерева r[N,4], m[N,8], l[N,16]."""
        N = len(levels)
        r = torch.zeros(N, 4, dtype=torch.bool, device=device)
        m = torch.zeros(N, 8, dtype=torch.bool, device=device)
        lf = torch.zeros(N, 16, dtype=torch.bool, device=device)
        for i, (lv, ix) in enumerate(zip(levels, idxs)):
            a, b, c = descendants(lv, int(ix))
            r[i, a] = True
            m[i, b] = True
            lf[i, c] = True
        return r, m, lf

    def edited(dec, F1, ref, levels, idxs, e):
        """Одно наблюдение F1 ([1,…]), reference ref = dec(F1); N кандидатов:
        уровень, индекс узла, e [N,6] (сырой вектор поправки узла, FP32).
        -> dict(Zhat [N,T,6], z1, z2, delta1, y2, C0) — FP32, без схвата."""
        dev = F1["x0"].device
        N = len(levels)
        rd, cm = dec.readout, dec.cond

        def rep(x):
            return x.expand(N, *x.shape[1:])
        with torch.no_grad(), torch.autocast(device_type=dev.type,
                                             enabled=False):
            half, _mid = half_mid(dec._L, dec._U, dev)
            e = torch.as_tensor(e, dtype=torch.float32, device=dev)
            mr, mm, ml = masks(levels, idxs, dev)
            lv = list(levels)
            sel_r = torch.tensor([x == "root" for x in lv], device=dev)
            sel_m = torch.tensor([x == "mid" for x in lv], device=dev)
            sel_l = torch.tensor([x == "leaf" for x in lv], device=dev)
            ix = torch.as_tensor([int(i) for i in idxs], device=dev)
            e0 = torch.zeros(N, 4, 6, device=dev)
            e1 = torch.zeros(N, 8, 6, device=dev)
            e2 = torch.zeros(N, 16, 6, device=dev)
            ar = torch.arange(N, device=dev)
            e0[ar[sel_r], ix[sel_r]] = e[sel_r]
            e1[ar[sel_m], ix[sel_m]] = e[sel_m]
            e2[ar[sel_l], ix[sel_l]] = e[sel_l]
            v0 = rd.ln(rep(F1["x0"]).float())
            v1 = rd.ln(rep(F1["x1"]).float())
            v2 = rd.ln(rep(F1["x2"]).float())
            C0r, g1, g2 = rep(ref["C0"]), rep(ref["g1"]), rep(ref["g2"])
            C0 = torch.where(mr[..., None], C0r + half * e0, C0r)
            mp, lp = MP.to(dev), LP.to(dev)
            z1p = C0[:, mp][:, :, None, :].expand(-1, -1, T, -1)
            pc1 = torch.tanh(z1p / half)
            u1 = v1[:, :, None, :] + cm.A_h(v0[:, mp])[:, :, None, :] + \
                cm.A_a(pc1)
            d1 = mlp_t(rd, u1, rep(F1["iv1"]))[..., :6].float()
            mm4 = mm[:, :, None, None]
            delta1 = torch.where(mm4, d1, rep(ref["delta1"]))
            z1n = z1p + g1[..., None, None] * delta1 + half * e1[:, :, None, :]
            z1 = torch.where(mm4, z1n, rep(ref["z1"]))
            z2p = z1[:, lp]
            pc2 = torch.tanh(z2p / half)
            u2 = v2[:, :, None, :] + cm.A_h(v1[:, lp])[:, :, None, :] + \
                cm.A_a(pc2)
            y = mlp_t(rd, u2, rep(F1["iv2"])).float()
            ml4 = ml[:, :, None, None]
            y2 = torch.where(ml4, y, rep(ref["y2"]))
            z2n = z2p + g2[..., None, None] * y2[..., :6] + \
                half * e2[:, :, None, :]
            z2 = torch.where(ml4, z2n, rep(ref["z2"]))
            Zhat = torch.einsum("btl,bltc->btc", rep(F1["W2"]).float(), z2)
        return dict(Zhat=Zhat, z1=z1, z2=z2, delta1=delta1, y2=y2, C0=C0,
                    masks=(mr, mm, ml))

    return dict(edited=edited, masks=masks, half_mid=half_mid, R=R, phi=phi)


def selftest():
    import torch
    import k21_model as km
    import k25_model as k25m
    import k29_model as k29m
    torch.manual_seed(0)
    k29m.no_tf32(torch)
    M = k29m.build(torch)
    E = build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    mod = k25m.build(torch)["K25"](km.fake_bar(torch), L, U, seed=0).eval()
    with torch.no_grad():                    # «обученный» источник, A_a ≠ 0
        g = torch.Generator().manual_seed(3)
        for p in list(mod.cond.parameters()) + list(mod.readout.parameters()):
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    for p in mod.parameters():
        p.requires_grad_(False)
    dec = M["Decoder"](mod).freeze()
    bt = km.fake_batch(torch, B=1, seed=4)
    s11 = torch.randn(1, 11, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        F, _s, _c = M["export"](mod, bt, 1, s11)
        ref = dec(F)
    W2 = F["W2"][0].float().numpy()
    path = path_of(W2)
    # 1. индексы и вложенность пути; ничья — меньший индекс
    assert path[1] == path[2] // 2 and path[0] == path[1] // 2
    Wt = np.zeros((T, 16))
    Wt[:, 5] = Wt[:, 3] = 1.0
    assert path_of(Wt) == (0, 1, 3)
    assert descendants("root", 1) == ([1], [2, 3], [4, 5, 6, 7])
    assert descendants("mid", 5) == ([], [5], [10, 11])
    info = node_info(F, path)
    assert 0 <= info["leaf"]["mass"] <= info["mid"]["mass"] + 1e-6 <= \
        info["root"]["mass"] + 2e-6
    for a, b in (("root", "mid"), ("mid", "leaf")):
        assert info[a]["width"] >= info[b]["width"] - 1e-5
    levels = list(LEVELS)
    idxs = list(path)
    # 2. нулевая правка любого узла = reference побитово (формулы повторены)
    for lv, ix in zip(levels, idxs):
        z = E["edited"](dec, F, ref, [lv], [ix], np.zeros((1, 6)))
        assert torch.equal(z["Zhat"], ref["Z"]), lv
        assert torch.equal(z["z2"], ref["z2"]) and \
            torch.equal(z["y2"], ref["y2"]), lv
    # 3. ненулевая правка: потомки меняются (через A_a), вне поддерева —
    # побитово прежние; схват не трогается (в выходе его нет)
    d = np.random.default_rng(0).normal(size=6)
    d = 0.1 * d / np.linalg.norm(d)
    o = E["edited"](dec, F, ref, levels, idxs, np.stack([d] * 3))
    r, m, lf = path[0], path[1], path[2]
    for i, lv in enumerate(levels):
        roots, mids, leaves = descendants(lv, idxs[i])
        other_l = [x for x in range(16) if x not in leaves]
        other_m = [x for x in range(8) if x not in mids]
        assert torch.equal(o["z2"][i, other_l], ref["z2"][0, other_l]), lv
        assert torch.equal(o["y2"][i, other_l], ref["y2"][0, other_l]), lv
        assert torch.equal(o["z1"][i, other_m], ref["z1"][0, other_m]), lv
        assert float((o["z2"][i, leaves] - ref["z2"][0, leaves]).abs()
                     .max()) > 1e-4, lv
    # корень: дети пересчитаны с новой командой родителя (не только сдвиг)
    sh = float((o["z1"][0, 2 * r] - ref["z1"][0, 2 * r]
                - (E["half_mid"](L, U, "cpu")[0] * torch.as_tensor(
                    d, dtype=torch.float32))).abs().max())
    assert sh > 1e-6, "при A_a ≠ 0 delta1 детей не изменилась"
    with torch.no_grad():
        dec0 = M["Decoder"](mod).freeze()
        dec0.cond.A_a.weight.zero_()
        ref0 = dec0(F)
        o0 = E["edited"](dec0, F, ref0, ["root"], [r], d[None])
        h = E["half_mid"](L, U, "cpu")[0]
        want = ref0["z1"][0, 2 * r] + h * torch.as_tensor(d,
                                                          dtype=torch.float32)
        assert torch.allclose(o0["z1"][0, 2 * r], want, atol=1e-6), \
            "при A_a = 0 должно остаться только наследование"
    # 4. FP32 под внешним autocast
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ob = E["edited"](dec, F, ref, levels, idxs, np.stack([d] * 3))
    assert ob["Zhat"].dtype == torch.float32 and \
        torch.equal(ob["Zhat"], o["Zhat"])
    # 5. независимость кандидатов друг от друга в батче
    o1 = E["edited"](dec, F, ref, ["mid"], [m], d[None])
    assert torch.equal(o1["Zhat"][0], o["Zhat"][1])
    assert lf == idxs[2]
    print("самопроверка k31_readout пройдена: путь по W2 (ничья — меньший), "
          "потомки, центр/ширина/масса узлов; нулевая правка каждого уровня = "
          "reference побитово; ненулевая меняет поддерево, вне него побитово "
          "прежнее; при A_a ≠ 0 дети пересчитаны, при A_a = 0 — только "
          "наследование; FP32 под autocast; кандидаты в батче независимы")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
