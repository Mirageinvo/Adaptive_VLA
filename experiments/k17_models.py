#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-17: сохраняемая геометрическая цель и три последовательные коррекции.

ПРЕДСКАЗАТЕЛЬ ЦЕЛИ G (один раз на окно, по наблюдению НАЧАЛА окна):
h18 [16, 768], a0 [8, 7], признаки состояния с историей и возрастом
(k16_models.state_features) -> 9 чисел: Δp / dpos_sigma (3) и
относительное вращение R_t^T R_{t+24} в 6D (6). Схват и тип события не
предсказываются. hidden() — вектор перед выходным слоем (GOAL_HIDDEN).

МИРОВАЯ ЦЕЛЬ (неизменна три вызова):
    p* = p_t + Δp̂,  R* = R_t · R̂_rel.
ГЕОМЕТРИЧЕСКАЯ ОШИБКА на каждом вызове (по ТЕКУЩЕМУ состоянию):
    e_p = p* − p_cur (/ dpos_sigma),  E_R = R_cur^T R* (6D)  -> 9 чисел.

КОРРЕКТОР C (структура CtxHead K-16b: Linear -> GELU -> Linear -> GELU,
линейные выходы поправки 8 × 6 и вспомогательной цели 9). Входы, общие
для обеих моделей: текущие h18, a0, признаки состояния; СОХРАНЁННЫЙ
скрытый вектор G начала окна; номер позиции (one-hot 0/1/2). Девять
геометрических каналов:
    ctx_memory  нули;
    held_goal   текущая ошибка (e_p, E_R) до сохранённой цели.
Размер входа одинаков; при одном сиде начальные веса C совпадают
побитово. Вспомогательная цель C — конечная поза окна относительно
текущего состояния позиции (только в потерях).
"""
import numpy as np

import k16_models as km

KINDS = ("ctx_memory", "held_goal")
G_OUT = 9
GEO = 9
N_POS = 3
HIDDEN = 208                          # ширина CtxHead K-16b
GOAL_HIDDEN = km.GOAL_HIDDEN


# --- геометрия: numpy ---------------------------------------------------------
def sixd_to_mat_np(x):
    x = np.asarray(x, np.float64)
    a, b = x[..., :3], x[..., 3:]
    c0 = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (c0 * b).sum(-1, keepdims=True) * c0
    c1 = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([c0, c1, np.cross(c0, c1)], -1)


def world_goal_np(g, p_t, R_t, dpos_sigma):
    """g [N, 9] (выход G) -> (p* [N, 3], R* [N, 3, 3])."""
    g = np.asarray(g, np.float64)
    p = np.asarray(p_t, np.float64) + g[:, :3] * np.asarray(dpos_sigma)
    return p, np.asarray(R_t, np.float64) @ sixd_to_mat_np(g[:, 3:9])


def geo_error_np(p_star, R_star, p_cur, R_cur, dpos_sigma):
    """Ошибка до сохранённой мировой цели по текущему состоянию [N, 9]."""
    e = (np.asarray(p_star) - np.asarray(p_cur)) / np.asarray(dpos_sigma)
    E = np.swapaxes(np.asarray(R_cur), -1, -2) @ np.asarray(R_star)
    return np.concatenate([e, km.mat_to_6d_np(E)], -1).astype(np.float32)


# --- модели -----------------------------------------------------------------
def build(torch, d_model=768, n_pos=16):
    nn = torch.nn

    def sixd_to_mat(x):
        a, b = x[..., :3], x[..., 3:]
        c0 = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        b = b - (c0 * b).sum(-1, keepdim=True) * c0
        c1 = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.stack([c0, c1, torch.cross(c0, c1, dim=-1)], -1)

    def mat_to_6d(R):
        return torch.cat([R[..., :, 0], R[..., :, 1]], -1)

    class H18Enc(nn.Module):
        def __init__(self, proj=32):
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.proj = nn.Linear(d_model, proj)
            self.out_dim = proj * n_pos

        def forward(self, h18):
            return self.proj(self.norm(h18.float())).flatten(1)

    class G(nn.Module):
        def __init__(self, hidden=GOAL_HIDDEN):
            super().__init__()
            self.enc = H18Enc()
            d = self.enc.out_dim + km.H * 7 + km.STATE_FEAT
            self.body = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                      nn.Linear(hidden, hidden), nn.GELU(),
                                      nn.Linear(hidden, G_OUT))

        def hidden(self, h18, a0, sf):
            x = torch.cat([self.enc(h18), a0.flatten(1), sf], 1)
            return self.body[:-1](x)

        def forward(self, h18, a0, sf):
            hid = self.hidden(h18, a0, sf)
            return self.body[-1](hid), hid

    class C(nn.Module):
        def __init__(self, hidden=HIDDEN):
            super().__init__()
            self.enc = H18Enc()
            d = (self.enc.out_dim + km.H * 7 + km.STATE_FEAT + GOAL_HIDDEN
                 + N_POS + GEO)
            self.trunk = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                       nn.Linear(hidden, hidden), nn.GELU())
            self.r = nn.Linear(hidden, km.H * km.N_ARM)
            self.aux = nn.Linear(hidden, G_OUT)

        def forward(self, h18, a0, sf, hid, pos_idx, geo):
            oh = torch.nn.functional.one_hot(pos_idx.long(), N_POS).float()
            z = self.trunk(torch.cat([self.enc(h18), a0.flatten(1), sf, hid,
                                      oh, geo], 1))
            return self.r(z).view(-1, km.H, km.N_ARM), self.aux(z)

    class Pair(nn.Module):
        """G + C одной модели. start() — один раз на окно; correct() —
        на каждой из трёх позиций. Обучение и рука используют одни и те
        же методы."""

        def __init__(self, kind, dpos_sigma, g_state=None, c_seed=0):
            super().__init__()
            if kind not in KINDS:
                raise ValueError(f"модель {kind!r}")
            self.kind = kind
            self.G = G()
            if g_state is not None:
                self.G.load_state_dict({k: v.clone()
                                        for k, v in g_state.items()})
            torch.manual_seed(c_seed)            # одинаковая инициализация C
            self.C = C()
            self.register_buffer("dpos_sigma", torch.as_tensor(
                np.asarray(dpos_sigma, np.float32)))

        def start(self, h18, a0, sf, p_t, R_t):
            g, hid = self.G(h18, a0, sf)
            p_star = p_t + g[:, :3] * self.dpos_sigma
            R_star = R_t @ sixd_to_mat(g[:, 3:9])
            return dict(goal=g, hid=hid, p_star=p_star, R_star=R_star)

        def geo(self, p_star, R_star, p_cur, R_cur):
            e = (p_star - p_cur) / self.dpos_sigma
            E = R_cur.transpose(-1, -2) @ R_star
            g = torch.cat([e, mat_to_6d(E)], -1)
            if self.kind == "ctx_memory":
                g = torch.zeros_like(g)
            return g

        def correct(self, h18, a0, sf, hid, pos_idx, p_star, R_star, p_cur,
                    R_cur):
            geo = self.geo(p_star, R_star, p_cur, R_cur)
            r, aux = self.C(h18, a0, sf, hid, pos_idx, geo)
            return r, aux, geo

    return dict(G=G, C=C, Pair=Pair, sixd_to_mat=sixd_to_mat,
                mat_to_6d=mat_to_6d)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def selftest():
    import torch
    rng = np.random.default_rng(0)
    M = build(torch)
    # 6D -> матрица: torch = numpy, ортонормальность
    x = rng.standard_normal((20, 6))
    Rt = M["sixd_to_mat"](torch.as_tensor(x)).numpy()
    Rn = sixd_to_mat_np(x)
    assert np.allclose(Rt, Rn, atol=1e-10)
    assert np.allclose(Rn @ np.swapaxes(Rn, -1, -2), np.eye(3), atol=1e-10)
    assert np.allclose(np.linalg.det(Rn), 1)
    ds = np.array([0.1, 0.2, 0.05])
    p_t = rng.standard_normal((20, 3))
    R_t = km.aa_to_mat_np(rng.standard_normal((20, 3)) * 0.5)
    g = rng.standard_normal((20, 9))
    p_s, R_s = world_goal_np(g, p_t, R_t, ds)
    # без движения руки ошибка до цели = исходная относительная цель
    e0 = geo_error_np(p_s, R_s, p_t, R_t, ds)
    assert np.allclose(e0[:, :3], g[:, :3])
    assert np.allclose(sixd_to_mat_np(e0[:, 3:]), sixd_to_mat_np(g[:, 3:]),
                       atol=1e-10)
    # движение руки: ошибка пересчитывается от текущей позы
    p_c = p_t + 0.03
    R_c = R_t @ km.aa_to_mat_np(np.tile([0.0, 0.2, 0.0], (20, 1)))
    e1 = geo_error_np(p_s, R_s, p_c, R_c, ds)
    assert np.allclose(e1[:, :3] * ds, p_s - p_c)
    assert np.allclose(R_c @ sixd_to_mat_np(e1[:, 3:]), R_s, atol=1e-10)
    assert not np.allclose(e1, e0)
    # torch-версия Pair совпадает с numpy
    pr = M["Pair"]("held_goal", ds).double()
    t = lambda a: torch.as_tensor(np.asarray(a, np.float64))  # noqa: E731
    gt = t(g)
    ps = t(p_t) + gt[:, :3] * pr.dpos_sigma
    Rs = t(R_t) @ M["sixd_to_mat"](gt[:, 3:9])
    assert np.allclose(ps.numpy(), p_s) and np.allclose(Rs.numpy(), R_s)
    ge = pr.geo(ps, Rs, t(p_c), t(R_c)).numpy()
    assert np.allclose(ge, e1, atol=1e-6)
    # ctx_memory: геометрические каналы — нули; C одинаковы побитово
    a = M["Pair"]("ctx_memory", ds, c_seed=3)
    b = M["Pair"]("held_goal", ds, c_seed=3)
    assert all(torch.equal(p, q) for p, q in zip(a.C.parameters(),
                                                 b.C.parameters()))
    assert n_params(a) == n_params(b)
    assert not a.geo(ps.float(), Rs.float(), t(p_c).float(),
                     t(R_c).float()).any()
    print("самопроверка k17_models пройдена: 6D torch = numpy, мировая цель "
          "и пересчёт ошибки при движении, без движения смысл цели "
          "сохранён, нули у ctx_memory, одинаковая инициализация C")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
