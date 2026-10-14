#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16: признаки и модели (общие для обучения и руки).

ВХОДЫ (явный список): h18 [16, 768] текущего вызова, a0 [8, 7] (единицы
кодека), наблюдаемое состояние руки [8] = (pos 3, axis-angle 3, пальцы 2),
история состояний через 8, 16, 24 команд назад [3, 8] с маской [3],
возраст эпизода в командах. Будущие состояния, номер старта и исход во
входы не попадают.

ПРИЗНАКИ СОСТОЯНИЯ: текущее — pos, вращение в 6D, пальцы (11 чисел);
история — относительно текущего: Δpos (3), относительное вращение в 6D
(6), пальцы (2), маска (1) — 12 чисел на позицию. Вариант без истории
(«now») получает те же маски и возраст, но значения прошлых состояний
обнулены.

ПРЕДСКАЗАТЕЛИ ЦЕЛИ (goal_now, goal_hist): h18 -> проекция по позициям ->
MLP вместе с a0, состоянием, возрастом и историей -> Δpos (3), вращение 6D
(6), логит команды схвата после события (1), логиты типа (3).

КОРРЕКТИРУЮЩИЕ ГОЛОВЫ (поправка руки r [8, 6], единицы кодека):
  direct    те же входы, что у goal_hist (h18, a0, состояние, история),
            -> r; вспомогательный выход цели (та же разметка, вес AUX);
  goal_flat ПРЕДСКАЗАННАЯ цель + текущее состояние + a0 -> r;
  goal_band те же входы -> плавная часть (первые 2 коэффициента
            ортонормированного DCT-II по 8 шагам на канал) -> оставшиеся 6
            коэффициентов с учётом ПРЕДСКАЗАНИЯ первой ветки -> обратное
            DCT, сумма частей.
Число параметров голов выравнивается шириной скрытого слоя (±10%).
"""
import math

import numpy as np

H, N_ARM = 8, 6
HIST_OFFSETS = (8, 16, 24)
N_LOW = 2
GOAL_DIM = 3 + 6 + 1 + 3
AUX = 0.5


# --- геометрия (numpy и torch) ---------------------------------------------
def aa_to_mat_np(v):
    v = np.asarray(v, np.float64)
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    k = np.where(th > 1e-12, v / np.maximum(th, 1e-300), 0.0)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def mat_to_6d_np(R):
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def state_features(state, hist, hist_mask, age, use_hist=True):
    """state [N, 8], hist [N, 3, 8], hist_mask [N, 3], age [N] (команд) ->
    признаки [N, 11 + 36 + 1] (numpy, float32)."""
    state = np.asarray(state, np.float64)
    R = aa_to_mat_np(state[:, 3:6])
    cur = np.concatenate([state[:, :3], mat_to_6d_np(R), state[:, 6:8]], 1)
    hist = np.asarray(hist, np.float64)
    m = np.asarray(hist_mask, np.float64)
    Rh = aa_to_mat_np(hist[..., 3:6])
    rel = np.swapaxes(R, -1, -2)[:, None] @ Rh
    hv = np.concatenate([hist[..., :3] - state[:, None, :3],
                         mat_to_6d_np(rel), hist[..., 6:8]], -1)
    hv = hv * m[..., None]
    if not use_hist:
        hv = np.zeros_like(hv)
    hf = np.concatenate([hv, m[..., None]], -1).reshape(len(state), -1)
    ag = (np.asarray(age, np.float64) / (8.0 * 50.0))[:, None]
    return np.concatenate([cur, hf, ag], 1).astype(np.float32)


STATE_FEAT = 11 + 12 * len(HIST_OFFSETS) + 1


def dct_matrix(n=H):
    k = np.arange(n)[:, None]
    t = np.arange(n)[None, :]
    m = np.cos(np.pi * (t + 0.5) * k / n)
    m[0] *= 1.0 / math.sqrt(n)
    m[1:] *= math.sqrt(2.0 / n)
    return m                                     # [частота, время]


# --- модели ------------------------------------------------------------------
def build(torch):
    nn = torch.nn
    D = torch.tensor(dct_matrix(), dtype=torch.float32)

    class H18Enc(nn.Module):
        def __init__(self, d_model=768, proj=32, n_pos=16):
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.proj = nn.Linear(d_model, proj)
            self.out_dim = proj * n_pos

        def forward(self, h18):
            return self.proj(self.norm(h18.float())).flatten(1)

    def mlp(i, h, o, depth=2):
        layers, d = [], i
        for _ in range(depth):
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers.append(nn.Linear(d, o))
        return nn.Sequential(*layers)

    class GoalNet(nn.Module):
        """h18, a0, признаки состояния -> цель (GOAL_DIM)."""

        def __init__(self, hidden=256):
            super().__init__()
            self.enc = H18Enc()
            self.body = mlp(self.enc.out_dim + H * 7 + STATE_FEAT, hidden,
                            GOAL_DIM)

        def forward(self, h18, a0, sf):
            x = torch.cat([self.enc(h18), a0.flatten(1), sf], 1)
            return self.body(x)

    class Direct(nn.Module):
        """Прямая голова: богатые входы -> r [8, 6]; вспомогательная цель."""

        def __init__(self, hidden=256):
            super().__init__()
            self.enc = H18Enc()
            d = self.enc.out_dim + H * 7 + STATE_FEAT
            self.trunk = mlp(d, hidden, hidden, depth=1)
            self.r = nn.Sequential(nn.GELU(), nn.Linear(hidden, H * N_ARM))
            self.aux = nn.Sequential(nn.GELU(), nn.Linear(hidden, GOAL_DIM))

        def forward(self, h18, a0, sf, goal=None):
            z = self.trunk(torch.cat([self.enc(h18), a0.flatten(1), sf], 1))
            return self.r(z).view(-1, H, N_ARM), self.aux(z)

    def goal_in(goal, sf, a0):
        g = torch.cat([goal[:, :9], torch.tanh(goal[:, 9:10]),
                       torch.softmax(goal[:, 10:13], -1)], 1)
        return torch.cat([g, sf[:, :11], a0.flatten(1)], 1)

    GIN = GOAL_DIM + 11 + H * 7

    class GoalFlat(nn.Module):
        def __init__(self, hidden=256):
            super().__init__()
            self.body = mlp(GIN, hidden, H * N_ARM, depth=3)

        def forward(self, h18, a0, sf, goal):
            return self.body(goal_in(goal, sf, a0)).view(-1, H, N_ARM), None

    class GoalBand(nn.Module):
        """Плавная часть (N_LOW коэффициентов DCT на канал), затем остаток
        с учётом ПРЕДСКАЗАНИЯ плавной части."""

        def __init__(self, hidden=256):
            super().__init__()
            self.low = mlp(GIN, hidden, N_LOW * N_ARM, depth=2)
            self.high = mlp(GIN + N_LOW * N_ARM, hidden,
                            (H - N_LOW) * N_ARM, depth=2)
            self.register_buffer("D", D.clone())

        def parts(self, h18, a0, sf, goal):
            x = goal_in(goal, sf, a0)
            lo = self.low(x)
            hi = self.high(torch.cat([x, lo], 1))
            coef = torch.cat([lo.view(-1, N_LOW, N_ARM),
                              hi.view(-1, H - N_LOW, N_ARM)], 1)
            # обратное DCT: время = D^T · коэффициенты
            r_lo = torch.einsum("ft,bfc->btc", self.D[:N_LOW],
                                coef[:, :N_LOW])
            r_hi = torch.einsum("ft,bfc->btc", self.D[N_LOW:],
                                coef[:, N_LOW:])
            return r_lo, r_hi, coef

        def forward(self, h18, a0, sf, goal):
            r_lo, r_hi, _c = self.parts(h18, a0, sf, goal)
            return r_lo + r_hi, None

    return dict(GoalNet=GoalNet, Direct=Direct, GoalFlat=GoalFlat,
                GoalBand=GoalBand)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def matched_widths(torch, cls, direct_hidden=256, tol=0.10):
    """Ширины goal_flat / goal_band, при которых число параметров в
    пределах tol от прямой головы."""
    target = n_params(cls["Direct"](direct_hidden))
    out = {}
    for name in ("GoalFlat", "GoalBand"):
        best = None
        for h in range(64, 2049, 8):
            n = n_params(cls[name](h))
            if best is None or abs(n - target) < abs(best[1] - target):
                best = (h, n)
        if abs(best[1] - target) > tol * target:
            raise SystemExit(f"{name}: не удаётся выровнять параметры")
        out[name] = best[0]
    return target, out
