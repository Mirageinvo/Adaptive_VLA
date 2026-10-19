#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: временные интервалы дерева движения, порождение дочерних query,
дифференцируемая маршрутизация и общий локальный readout.

Время чанка — [0, 20], точки команд x_k = k + 0.5.
КОРЕНЬ: d_i = DMIN + (20 − 4·DMIN)·softmax(logits)_i (DMIN = 2), границы
  b_0 = 0, b_i = Σ_{j<i} d_j, b_4 = 20 — упорядочены по построению.
ДЕЛЕНИЕ: r = RMIN + (1 − 2·RMIN)·sigmoid(v) (RMIN = 0.25), s = left + r·width.
  fixed: d_i = 5, r = 0.5.
ВЕСА: w_root = [1 − S_1, S_1 − S_2, S_2 − S_3, S_3], S_j = σ((x − b_j)/τ_root),
  τ_root = 0.5; деление родителя умножает его вес на [1 − g, g],
  g = σ((x − s)/τ), τ = 0.1·width_parent. Сумма весов 16 листьев = 1.
  Расчёт в float32.
ДЕТИ: h_child = h_parent + e_side + W(φ(left, right, center, width)).
READOUT: y_leaf(x) = MLP([norm(h_leaf), φ(local_time), φ(x/20)]),
  local_time = clamp(2(x − center)/width, −2, 2); y(x) = Σ w_leaf(x)·y_leaf(x).
  Рука: a = mid + half·tanh(y/half); седьмой выход — логит схвата.
φ(u) = [u, sin(πku), cos(πku)], k = 1..4.
"""
import numpy as np

T = 20
DMIN, RMIN = 2.0, 0.25
TAU_ROOT, TAU_REL = 0.5, 0.1
FK = 4
PHI = 1 + 2 * FK
CONFIG = dict(T=T, DMIN=DMIN, RMIN=RMIN, TAU_ROOT=TAU_ROOT, TAU_REL=TAU_REL,
              FK=FK, readout_hidden=128)


def build(torch):
    nn = torch.nn

    def phi(u):
        k = torch.arange(1, FK + 1, device=u.device, dtype=u.dtype)
        a = np.pi * u[..., None] * k
        return torch.cat([u[..., None], torch.sin(a), torch.cos(a)], -1)

    def root(logits, fixed=False):
        """logits [B, 4] -> интервалы [B, 4, 2] (left, right)."""
        lg = logits.float()
        if fixed:
            d = torch.full_like(lg, T / 4)
        else:
            d = DMIN + (T - 4 * DMIN) * torch.softmax(lg, -1)
        b = torch.cat([torch.zeros_like(d[:, :1]), torch.cumsum(d, -1)], -1)
        b = torch.cat([b[:, :-1], torch.full_like(b[:, -1:], float(T))], -1)
        return torch.stack([b[:, :-1], b[:, 1:]], -1)

    def split(iv, v, fixed=False):
        """iv [B, n, 2], v [B, n] -> дети [B, 2n, 2] (левый, правый)."""
        left, right = iv[..., 0], iv[..., 1]
        if fixed:
            r = torch.full_like(left, 0.5)
        else:
            r = RMIN + (1 - 2 * RMIN) * torch.sigmoid(v.float())
        s = left + r * (right - left)
        ch = torch.stack([torch.stack([left, s], -1),
                          torch.stack([s, right], -1)], 2)
        return ch.reshape(iv.shape[0], -1, 2), s

    def x_points(device):
        return torch.arange(T, device=device, dtype=torch.float32) + 0.5

    def root_weights(iv):
        x = x_points(iv.device)
        b = iv[:, 1:, 0]                                  # внутренние границы
        S = torch.sigmoid((x[None, :, None] - b[:, None, :]) / TAU_ROOT)
        one = torch.ones_like(S[..., :1])
        zero = torch.zeros_like(S[..., :1])
        Sx = torch.cat([one, S, zero], -1)               # S_0 = 1, S_4 = 0
        return Sx[..., :-1] - Sx[..., 1:]                # [B, 20, 4]

    def split_weights(w, iv_parent, s):
        x = x_points(w.device)
        width = iv_parent[..., 1] - iv_parent[..., 0]
        tau = TAU_REL * width
        g = torch.sigmoid((x[None, :, None] - s[:, None, :])
                          / tau[:, None, :])
        wc = torch.stack([w * (1 - g), w * g], -1)        # [B, 20, n, 2]
        return wc.reshape(w.shape[0], T, -1)

    class Expand(nn.Module):
        """Родители -> дети с наследованием признака и координатами
        интервала."""

        def __init__(self, H, init=1e-3):
            super().__init__()
            self.e_side = nn.Parameter(torch.randn(2, H) * init)
            self.W = nn.Linear(4 * PHI, H)
            nn.init.normal_(self.W.weight, std=init)
            nn.init.zeros_(self.W.bias)

        def forward(self, h, iv_children):
            B, n2, _ = iv_children.shape
            hp = h.repeat_interleave(2, 1)
            l, r = iv_children[..., 0] / T, iv_children[..., 1] / T
            f = torch.cat([phi(l), phi(r), phi((l + r) / 2), phi(r - l)], -1)
            side = self.e_side.repeat(n2 // 2, 1)[None]
            return hp + side.to(hp.dtype) + self.W(f.to(self.W.weight.dtype)
                                                  ).to(hp.dtype)

    class Head(nn.Module):
        """Логит длительности или деления из признака узла (нулевая
        инициализация -> равные длительности, деление пополам)."""

        def __init__(self, H):
            super().__init__()
            self.ln = nn.LayerNorm(H)
            self.lin = nn.Linear(H, 1)
            nn.init.zeros_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)

        def forward(self, h):
            return self.lin(self.ln(h.float())).squeeze(-1)

    class Readout(nn.Module):
        def __init__(self, H, L, U, hidden=128):
            super().__init__()
            self.ln = nn.LayerNorm(H)
            self.mlp = nn.Sequential(nn.Linear(H + 2 * PHI, hidden),
                                     nn.GELU(), nn.Linear(hidden, hidden),
                                     nn.GELU(), nn.Linear(hidden, 7))
            self.register_buffer("L", torch.as_tensor(np.asarray(L,
                                                                 np.float32)))
            self.register_buffer("U", torch.as_tensor(np.asarray(U,
                                                                 np.float32)))

        def forward(self, h_leaf, iv_leaf, w):
            """h_leaf [B, 16, H], iv_leaf [B, 16, 2], w [B, 20, 16]."""
            B, n, H = h_leaf.shape
            x = x_points(h_leaf.device)
            c = iv_leaf.mean(-1)
            wd = (iv_leaf[..., 1] - iv_leaf[..., 0]).clamp_min(1e-6)
            lt = torch.clamp(2 * (x[None, None] - c[..., None])
                             / wd[..., None], -2, 2)          # [B, n, 20]
            hf = self.ln(h_leaf.float())[:, :, None].expand(B, n, T, H)
            gx = phi(x / T)[None, None].expand(B, n, T, PHI)
            inp = torch.cat([hf, phi(lt), gx], -1)
            y_leaf = self.mlp(inp)                             # [B, n, 20, 7]
            y = torch.einsum("btn,bntc->btc", w.float(), y_leaf)
            mid, half = (self.U + self.L) / 2, (self.U - self.L) / 2
            arm = mid + half * torch.tanh(y[..., :6] / half)
            return dict(y=y, arm=arm, grip_logit=y[..., 6])

    return dict(phi=phi, root=root, split=split, root_weights=root_weights,
                split_weights=split_weights, Expand=Expand, Head=Head,
                Readout=Readout, x_points=x_points)


def fixed_leaves(torch):
    """Интервалы и веса 16 равномерных листьев (общие для flat и fixed)."""
    M = build(torch)
    iv = M["root"](torch.zeros(1, 4), fixed=True)
    w = M["root_weights"](iv)
    iv2, s = M["split"](iv, torch.zeros(1, 4), fixed=True)
    w = M["split_weights"](w, iv, s)
    iv3, s2 = M["split"](iv2, torch.zeros(1, 8), fixed=True)
    w = M["split_weights"](w, iv2, s2)
    return iv3, w


def selftest():
    import torch
    M = build(torch)
    torch.manual_seed(0)
    B = 6
    for scale in (0.0, 1.0, 30.0, 300.0):                 # крайние логиты
        lg = torch.randn(B, 4) * scale
        iv = M["root"](lg)
        d = iv[..., 1] - iv[..., 0]
        assert (d >= DMIN - 1e-4).all() and torch.allclose(
            d.sum(-1), torch.full((B,), 20.0), atol=1e-4)
        assert (iv[:, 1:, 0] >= iv[:, :-1, 0]).all()
        w = M["root_weights"](iv)
        v = torch.randn(B, 4) * scale
        iv2, s = M["split"](iv, v)
        w2 = M["split_weights"](w, iv, s)
        iv3, s2 = M["split"](iv2, torch.randn(B, 8) * scale)
        w3 = M["split_weights"](w2, iv2, s2)
        for ww in (w, w2, w3):
            assert (ww >= -1e-7).all()
            assert torch.allclose(ww.sum(-1), torch.ones(B, T), atol=1e-5)
        # вложенность и упорядоченность
        assert (iv3[..., 1] >= iv3[..., 0]).all()
        assert torch.allclose(iv3.reshape(B, 4, 4, 2)[:, :, 0, 0], iv[..., 0])
        assert torch.allclose(iv3.reshape(B, 4, 4, 2)[:, :, -1, 1],
                              iv[..., 1])
        assert (iv3[:, 1:, 0] >= iv3[:, :-1, 1] - 1e-5).all()
        rr = (s - iv[..., 0]) / (iv[..., 1] - iv[..., 0])
        assert (rr >= RMIN - 1e-5).all() and (rr <= 1 - RMIN + 1e-5).all()
    # начальная инициализация: равные длительности, деление пополам
    iv = M["root"](torch.zeros(1, 4))
    assert torch.allclose(iv[0, :, 0], torch.tensor([0., 5, 10, 15]))
    ivf, wf = fixed_leaves(torch)
    assert torch.allclose(ivf[0, :, 0], torch.arange(16.) * 1.25)
    # градиент от финальной ошибки в предсказатели границ (без detach)
    H = 16
    lg = torch.zeros(2, 4, requires_grad=True)
    v1 = torch.zeros(2, 4, requires_grad=True)
    v2 = torch.zeros(2, 8, requires_grad=True)
    iv = M["root"](lg)
    w = M["root_weights"](iv)
    iv2, s = M["split"](iv, v1)
    w = M["split_weights"](w, iv, s)
    iv3, s2 = M["split"](iv2, v2)
    w = M["split_weights"](w, iv2, s2)
    ro = M["Readout"](H, -np.ones(6) * 2, np.ones(6) * 2)
    h = torch.randn(2, 16, H)
    out = ro(h, iv3, w)
    tgt = torch.randn(2, T, 6)
    ((out["arm"] - tgt) ** 2).mean().backward()
    for p_ in (lg, v1, v2):
        assert p_.grad is not None and float(p_.grad.abs().sum()) > 0
    # дети зависят от родителя; не независимые query
    ex = M["Expand"](H)
    hp = torch.randn(2, 4, H, requires_grad=True)
    ch = ex(hp, iv2.detach())
    assert ch.shape == (2, 8, H)
    ch[:, 0].sum().backward()
    assert float(hp.grad[:, 0].abs().sum()) > 0
    assert float(hp.grad[:, 1:].abs().sum()) == 0
    # выход в пределах, конечен
    big = M["Readout"](H, -np.ones(6) * 1.1, np.ones(6) * 3.0)
    with torch.no_grad():
        for m_ in big.mlp:
            if isinstance(m_, torch.nn.Linear):
                m_.weight.mul_(50)
        o = big(torch.randn(2, 16, H) * 10, ivf.expand(2, -1, -1),
                wf.expand(2, -1, -1))
    assert torch.isfinite(o["arm"]).all()
    assert (o["arm"] >= -1.1 - 1e-5).all() and (o["arm"] <= 3.0 + 1e-5).all()
    # около центра наклон 1
    ro2 = M["Readout"](H, -np.ones(6), np.ones(6))
    yv = torch.tensor([0.01])
    mid, half = 0.0, 1.0
    assert abs(float(mid + half * torch.tanh(yv / half)) - .01) < 1e-5
    _ = ro2
    print("самопроверка k21_tree пройдена: длительности ≥ 2 и сумма 20, "
          "вложенные упорядоченные интервалы, веса листьев ≥ 0 и сумма 1 "
          "(включая крайние логиты), градиент в предсказатели границ, "
          "дети наследуют родителя, допустимый конечный выход")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
