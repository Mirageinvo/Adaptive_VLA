#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-18: признаки наблюдения, актёр с базисом поправок, критик, ограничения.

НАБЛЮДЕНИЕ (одинаково для всех вариантов), вектор OBS_DIM:
  enc(h18) [512] — замороженный G_pre.enc K-17;
  a0 [8 × 7] — текущий чанк q0 (единицы кодека);
  признаки состояния с историей и возрастом [48] (k16_models.state_features);
  позиция окна one-hot [3].
Цель, номер задачи и старта, будущее и исход во вход не подаются.

АКТЁР: ствол Linear(OBS, 256) -> ReLU -> Linear(256, 256) -> ReLU = z;
V(x) = Linear(256, 48·4) (нулевая инициализация); голова коэффициентов
Linear(256 + 192, k) (нулевая инициализация), k = 4 (pca4, basis4) или 48
(full48). Базис:
  full48_rl  B = I_48;
  pca4_rl    B = U (V(x) участвует только во входе головы коэффициентов);
  basis4_rl  B = orth(U + V(x)) — QR, знаки столбцов: diag(R) > 0.
c = v / sqrt(1 + ||v||²),  δ_exec = ρ · B c  (||δ_exec|| < ρ).
ОГРАНИЧЕНИЕ КОМАНД (общая функция): если a0_exec + δ_exec выходит за
пределы входа контроллера, весь δ умножается на один множитель f ∈ [0, 1]
(направление сохраняется). Критик и replay получают δ после ограничения.
δ_codec = δ_exec / scale; a[:8, :6] = a0 + δ_codec; схват и хвост — a0;
при δ = 0 действие побитово равно a0. α и r_sigma K-16 не применяются.

КРИТИК: Linear(OBS + 48, 256) -> ReLU -> Linear(256, 256) -> ReLU ->
Linear(256, 1); вход — исполняемая 48-мерная поправка.
"""
import numpy as np

H, N_ARM = 8, 6
D_EXEC = H * N_ARM
ENC_DIM = 512
A0_DIM = H * 7
SF_DIM = 48
POS_DIM = 3
OBS_DIM = ENC_DIM + A0_DIM + SF_DIM + POS_DIM
RANK = 4
HID = 256
VARIANTS = ("full48_rl", "pca4_rl", "basis4_rl")
SL_A0 = slice(ENC_DIM, ENC_DIM + A0_DIM)


def make_obs(enc, a0, sf, pos):
    """enc [N, 512], a0 [N, 8, 7], sf [N, 48], pos [N] -> [N, OBS_DIM]."""
    n = len(pos)
    oh = np.zeros((n, POS_DIM), np.float32)
    oh[np.arange(n), np.asarray(pos, int)] = 1
    return np.concatenate([np.asarray(enc, np.float32).reshape(n, -1),
                           np.asarray(a0, np.float32)[:, :H].reshape(n, -1),
                           np.asarray(sf, np.float32), oh], 1)


def a0_exec_np(obs, scale):
    a0 = np.asarray(obs)[:, SL_A0].reshape(-1, H, 7)[:, :, :N_ARM]
    return (a0.astype(np.float64) * np.asarray(scale)[:N_ARM]).reshape(
        len(a0), -1)


def shrink_np(a0e, d, lo, hi):
    """Общий множитель f: a0e + f·d в [lo, hi] (по элементам)."""
    a0e = np.asarray(a0e, np.float64)
    d = np.asarray(d, np.float64)
    lo = np.tile(np.asarray(lo, np.float64), H)
    hi = np.tile(np.asarray(hi, np.float64), H)
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(d > 0, np.maximum(hi - a0e, 0) / d, np.inf)
        dn = np.where(d < 0, np.minimum(lo - a0e, 0) / d, np.inf)
    f = np.clip(np.minimum(up, dn).min(-1), 0.0, 1.0)
    return d * f[:, None], f


def apply_exec(a0, d_exec, scale):
    """a0 [N, T, 7] (кодек), d_exec [N, 48] -> действие (кодек). Схват и
    хвост — a0; при нулевой поправке строки побитово равны a0."""
    a = np.array(a0, np.float32, copy=True)
    d = np.asarray(d_exec, np.float64).reshape(-1, H, N_ARM)
    nz = np.any(d != 0, axis=(1, 2))
    if nz.any():
        dc = d[nz] / np.asarray(scale, np.float64)[:N_ARM]
        a[nz, :H, :N_ARM] = (a[nz, :H, :N_ARM].astype(np.float64)
                             + dc).astype(np.float32)
    return a


def build(torch):
    nn = torch.nn

    def orth(M):
        """M [N, 48, 4] -> ортонормированный базис (QR, diag(R) > 0) и
        min |diag(R)| для контроля ранга."""
        Q, R = torch.linalg.qr(M)
        dg = torch.diagonal(R, dim1=-2, dim2=-1)
        s = torch.where(dg < 0, -torch.ones_like(dg), torch.ones_like(dg))
        return Q * s[:, None, :], dg.abs().min(-1).values

    def shrink(a0e, d, lo, hi):
        lo = lo.repeat(H)
        hi = hi.repeat(H)
        big = torch.full_like(d, float("inf"))
        safe = torch.where(d == 0, torch.ones_like(d), d)
        up = torch.where(d > 0, torch.clamp(hi - a0e, min=0) / safe, big)
        dn = torch.where(d < 0, torch.clamp(lo - a0e, max=0) / safe, big)
        f = torch.clamp(torch.minimum(up, dn).min(-1).values, 0.0, 1.0)
        return d * f[:, None], f

    class Actor(nn.Module):
        def __init__(self, variant, U, rho, scale, lo, hi, seed=0):
            super().__init__()
            if variant not in VARIANTS:
                raise ValueError(variant)
            self.variant = variant
            self.k = D_EXEC if variant == "full48_rl" else RANK
            torch.manual_seed(seed)
            self.trunk = nn.Sequential(nn.Linear(OBS_DIM, HID), nn.ReLU(),
                                       nn.Linear(HID, HID), nn.ReLU())
            self.V = nn.Linear(HID, D_EXEC * RANK)
            self.coef = nn.Linear(HID + D_EXEC * RANK, self.k)
            for m in (self.V, self.coef):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
            self.register_buffer("U", torch.as_tensor(np.asarray(
                U, np.float32)))
            self.register_buffer("rho", torch.tensor(float(rho)))
            self.register_buffer("scale", torch.as_tensor(np.asarray(
                scale, np.float32)[:N_ARM]))
            self.register_buffer("lo", torch.as_tensor(np.asarray(
                lo, np.float32)))
            self.register_buffer("hi", torch.as_tensor(np.asarray(
                hi, np.float32)))

        def basis(self, Vx):
            n = Vx.shape[0]
            if self.variant == "full48_rl":
                return None, None
            Ub = self.U.unsqueeze(0).expand(n, -1, -1)
            if self.variant == "pca4_rl":
                return Ub, None
            B, mind = orth(Ub + Vx.view(n, D_EXEC, RANK))
            return B, mind

        def forward(self, obs, noise=None):
            """obs [N, OBS] -> dict: v, c, B, delta_pre, delta, f, mind.
            noise [N, k] прибавляется к v до ограничения нормы."""
            z = self.trunk(obs)
            Vx = self.V(z)
            v = self.coef(torch.cat([z, Vx], 1))
            if noise is not None:
                v = v + noise
            c = v / torch.sqrt(1 + (v ** 2).sum(-1, keepdim=True))
            B, mind = self.basis(Vx)
            if B is None:
                d = self.rho * c
            else:
                d = self.rho * torch.einsum("nij,nj->ni", B, c)
            a0 = obs[:, SL_A0].reshape(-1, H, 7)[:, :, :N_ARM]
            a0e = (a0 * self.scale).reshape(len(obs), -1)
            ds, f = shrink(a0e, d, self.lo, self.hi)
            return dict(v=v, c=c, B=B, delta_pre=d, delta=ds, f=f,
                        mind=mind, Vx=Vx)

    class Critic(nn.Module):
        def __init__(self, seed):
            super().__init__()
            torch.manual_seed(seed)
            self.net = nn.Sequential(nn.Linear(OBS_DIM + D_EXEC, HID),
                                     nn.ReLU(), nn.Linear(HID, HID),
                                     nn.ReLU(), nn.Linear(HID, 1))

        def forward(self, obs, d):
            return self.net(torch.cat([obs, d], 1)).squeeze(-1)

    return dict(Actor=Actor, Critic=Critic, orth=orth, shrink=shrink)


def principal_angles(B, U):
    """Главные углы (градусы) между подпространствами span(B) и span(U)."""
    s = np.linalg.svd(np.asarray(U).T @ np.asarray(B), compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def selftest():
    import torch
    rng = np.random.default_rng(0)
    M = build(torch)
    U, _ = np.linalg.qr(rng.standard_normal((48, 4)))
    scale = np.array([.9375, .91, .9375, .2036, .2636, .375, 1.0])
    lo, hi = -np.ones(6), np.ones(6)
    acts = {v: M["Actor"](v, U, 0.7068, scale, lo, hi, seed=0)
            for v in VARIANTS}
    # одинаковые ствол и V; число параметров в пределах 10%
    for v in VARIANTS[1:]:
        for n_ in ("trunk", "V"):
            assert all(torch.equal(p, q) for p, q in zip(
                getattr(acts[VARIANTS[0]], n_).parameters(),
                getattr(acts[v], n_).parameters()))
    assert n_params(acts["pca4_rl"]) == n_params(acts["basis4_rl"])
    nf, n4 = n_params(acts["full48_rl"]), n_params(acts["pca4_rl"])
    assert abs(nf - n4) / n4 < 0.10, (nf, n4)
    obs = torch.as_tensor(rng.standard_normal((16, OBS_DIM)).astype(
        np.float32) * .3)
    # нулевая инициализация: δ = 0 побитово -> действие = a0
    for v, A in acts.items():
        o = A(obs)
        assert not o["delta"].any(), v
        a0 = rng.uniform(-.5, .5, (16, 20, 7)).astype(np.float32)
        a0[3, 0, 0] = -0.0
        assert apply_exec(a0, o["delta"].detach().numpy(), scale).tobytes() \
            == a0.tobytes()
    # базис basis4: ортонормален, при V = 0 равен U (с точностью до знака
    # столбцов, фиксированного QR), главные углы 0
    B, mind = M["orth"](torch.as_tensor(U, dtype=torch.float32)[None]
                        + 0.3 * torch.randn(1, 48, 4))
    Bn = B[0].numpy()
    assert np.allclose(Bn.T @ Bn, np.eye(4), atol=1e-5) and float(mind) > 0
    o = acts["basis4_rl"](obs, noise=torch.ones(16, 4))
    assert np.allclose(principal_angles(o["B"][0].detach().numpy(), U), 0,
                       atol=0.1)
    # норма < ρ, ограничение команд — общий множитель, направление то же
    for v, A in acts.items():
        k = A.k
        o = A(obs, noise=torch.as_tensor(rng.standard_normal((16, k))
                                         .astype(np.float32) * 5))
        n = o["delta_pre"].norm(dim=-1)
        assert (n < 0.7068 + 1e-6).all() and (n > 0).all()
    a0e = np.full((2, 48), 0.95)
    d = np.full((2, 48), 0.1)
    d[1] = -0.1
    ds, f = shrink_np(a0e, d, lo, hi)
    assert np.isclose(f[0], 0.5) and f[1] == 1.0
    assert np.all(a0e + ds <= 1 + 1e-12)
    dt, ft = M["shrink"](torch.as_tensor(a0e), torch.as_tensor(d),
                         torch.as_tensor(lo), torch.as_tensor(hi))
    assert np.allclose(dt.numpy(), ds) and np.allclose(ft.numpy(), f)
    # градиент от критика достигает базиса basis4 при ненулевых коэфф.
    A = acts["basis4_rl"]
    with torch.no_grad():
        A.coef.weight.normal_(0, .1)
    Cr = M["Critic"](1)
    q = Cr(obs, A(obs)["delta"]).mean()
    q.backward()
    assert float(A.V.weight.grad.abs().sum()) > 0
    assert float(A.trunk[0].weight.grad.abs().sum()) > 0
    # pca4: V получает градиент только через голову коэффициентов
    A2 = acts["pca4_rl"]
    with torch.no_grad():
        A2.coef.weight.normal_(0, .1)
    Cr(obs, A2(obs)["delta"]).mean().backward()
    assert float(A2.V.weight.grad.abs().sum()) > 0
    # признаки наблюдения
    ob = make_obs(np.ones((2, 16, 32)), np.ones((2, 20, 7)), np.ones((2, 48)),
                  [0, 2])
    assert ob.shape == (2, OBS_DIM) and ob[1, -1] == 1 and ob[0, -3] == 1
    assert np.allclose(a0_exec_np(ob, scale)[0, :6], scale[:6])
    print("самопроверка k18_models пройдена: равенство ствола и V, "
          "параметры (±10%), нулевой старт = a0 побитово, ортонормальность и "
          "знаки QR, главные углы, предел ρ, общий множитель команд, "
          "градиент критика в базис")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
