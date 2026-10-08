#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15f: непрерывное многомерное уточнение — математика и модель.

КАСАТЕЛЬНОЕ ПРОСТРАНСТВО. Поправка исполняемой части чанка (H_EXEC = 8
шагов) — вектор t ∈ R^{8×7} в нормированных координатах:

  рука   t[k, c] = (a[k, c] − a0[k, c]) / σ_c,   c = 0..5,
  схват  t[k, 6] = (ℓ[k] − ℓ0[k]) / σ_g,         ℓ = 2·atanh(clip(g)),

σ_c — RMS остатка демонстрации к q0 по каналу руки на train, σ_g — RMS
остатка ЛОГИТА схвата. Обратное отображение (`combine`) — ровно
параметризация K-15d: рука аддитивно, схват через приращение логита,
g = g0 + tanh(ℓ/2) − tanh(ℓ0/2). Хвост чанка за H_EXEC равен a0. При t = 0
действие побитово равно a0.

БАЗИС. K = 4 направления δ_j(s) ∈ R^{8×7}:

  δ_j(s) = ρ_j · (U_j + w_j(s)),   w_j ⊥ U_j,   ‖w_j‖ <= κ,

U_j — j-я главная компонента (без центрирования) остатка демонстрации на
train, ρ_j — p90 |<r*, U_j>|, w_j(s) — поправка от h18 (последний слой
нулевой: в нуле базис — чистая PCA).

ЗНАК И ИНДЕКС ПРИВЯЗАНЫ К U_j СТРУКТУРНО, а не регуляризатором: w_j
проецируется на ортогональное дополнение U_j и гладко ограничивается по
норме (κ·tanh(‖w‖)/‖w‖). Тогда cos(δ_j, U_j) = 1/sqrt(1 + ‖w_j‖²) >=
1/sqrt(1 + κ²) при ЛЮБЫХ весах — при κ = 1 не ниже 0.707. Первая версия
держала якорь только штрафом, и на игрушечном обучении косинус одного
направления ушёл в −0.15: c_j = +1 означало бы в разных состояниях
противоположные действия. Распределение косинуса всё равно печатается.

ДЕЙСТВИЕ ПРИ КОЭФФИЦИЕНТАХ c ∈ [−1, 1]^4:  t(s) = Σ_j c_j δ_j(s),
a(s) = combine(a0, t(s)).

КОНТРОЛЬ (M1). B_ctrl(s) = B(s)·R, R — фиксированная ортогональная
матрица 56×56, блочная: отдельно для 48 координат руки и 8 координат
схвата (вращение не смешивает руку со схватом). Сохраняет нормы
направлений, их попарную геометрию и величину поправки при данном c.

ПРЕДОБУЧЕНИЕ БАЗИСА — НЕ ПО УСПЕХУ. Для строки коэффициенты проекции
остатка на текущий базис — ridge 4×4: c* = (ΔΔᵀ + λI)^{-1} Δ r*, потеря —
Smooth L1 остатка проекции плюс ортогональность, нормы и якорь к PCA.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

K_BASIS = 4
H_EXEC = 8
N_CH = 7
D_T = H_EXEC * N_CH                 # 56 координат касательного пространства
GRIP_EPS = 1e-3                     # как в K-15d
HP = dict(proj=64, hidden=512, ridge=1e-3, w_ortho=0.1, w_norm=0.1,
          w_anchor=0.1, smooth_l1_beta=1.0, amp_quantile=90.0, kappa=1.0)
# ПРЕДЕЛЫ ГЕОМЕТРИИ БАЗИСА — зарегистрированы до данных (08.10.2026).
# Якорь не мешает двум направлениям схлопнуться к общему вектору (оба
# могут быть пропорциональны U_1 + U_2 при косинусе 0.707 со своими
# якорями). Поэтому на val_sel для выбранной точки обязательны:
GEOMETRY = dict(cond_max=10.0, pair_cos_max=0.9)
# ЗАВИСИМОСТЬ ОТ h18 (гейт): относительный RMS изменения базиса при
# перестановке h18 между строками батча не ниже этого порога. Ниже —
# базис фактически глобальная PCA: отдельный baseline, а не иерархия.
STATE_DEP_MIN_REL = 0.01


def geometry_ok(diag, limits=None):
    lim = dict(GEOMETRY if limits is None else limits)
    return bool(diag["cond_max"] <= lim["cond_max"]
                and diag["pair_cos_max"] <= lim["pair_cos_max"])


def grip_logit(g):
    c = 1.0 - GRIP_EPS
    return 2.0 * torch.atanh(g.float().clamp(-c, c))


# --- касательное пространство ------------------------------------------------
def to_tangent(a, a0, sigma_arm, sigma_g, h_exec=H_EXEC):
    """Действие a относительно a0 -> t [B, 8, 7] в нормированных единицах."""
    h = int(h_exec)
    t_arm = (a[:, :h, :6].float() - a0[:, :h, :6].float()) / sigma_arm
    t_g = (grip_logit(a[:, :h, 6]) - grip_logit(a0[:, :h, 6])) / sigma_g
    return torch.cat([t_arm, t_g[..., None]], dim=-1)


def combine(a0, t, sigma_arm, sigma_g, h_exec=H_EXEC):
    """a0 [B, T, 7] и t [B, 8, 7] -> действие [B, T, 7]. t = 0 -> a0 точно."""
    h = int(h_exec)
    arm = a0[:, :h, :6] + sigma_arm * t[..., :6]
    lg0 = grip_logit(a0[:, :h, 6])
    lg = lg0 + sigma_g * t[..., 6]
    g = a0[:, :h, 6] + (torch.tanh(lg * 0.5) - torch.tanh(lg0 * 0.5))
    exec_part = torch.cat([arm, g[..., None]], dim=-1)
    return torch.cat([exec_part.to(a0.dtype), a0[:, h:]], dim=1)


def compose(basis, c):
    """basis [B, K, 56], c [B, K] или [K] -> t [B, 8, 7]."""
    if c.ndim == 1:
        c = c[None].expand(basis.shape[0], -1)
    return torch.einsum("bk,bkd->bd", c.to(basis.dtype), basis).view(
        basis.shape[0], H_EXEC, N_CH)


# --- статистика train и PCA --------------------------------------------------
def tangent_stats(a0_exec, act_exec, k=K_BASIS, quantile=None):
    """σ руки/схвата, якоря PCA U [K, 56], амплитуды ρ [K].

    PCA — без центрирования: поправка отсчитывается от нуля, и средний
    остаток — законное направление.
    """
    q = HP["amp_quantile"] if quantile is None else quantile
    a0e, ae = a0_exec.double(), act_exec.double()
    r_arm = ae[..., :6] - a0e[..., :6]
    sigma_arm = r_arm.pow(2).mean(dim=(0, 1)).sqrt().clamp_min(1e-4)
    r_g = grip_logit(ae[..., 6]).double() - grip_logit(a0e[..., 6]).double()
    sigma_g = r_g.pow(2).mean().sqrt().clamp_min(1e-4)
    t = torch.cat([r_arm / sigma_arm, (r_g / sigma_g)[..., None]],
                  dim=-1).reshape(len(a0e), -1)
    # SVD второго момента: собственные векторы t^T t
    gram = t.T @ t / len(t)
    evals, evecs = torch.linalg.eigh(gram)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    U = evecs[:, :k].T.contiguous()                      # [K, 56]
    # знак: среднее проекций положительно (детерминированная привязка)
    proj = t @ U.T                                        # [N, K]
    sign = torch.where(proj.mean(0) < 0, -1.0, 1.0).to(U.dtype)
    U = U * sign[:, None]
    proj = proj * sign
    rho = torch.quantile(proj.abs().float(), q / 100.0, dim=0).double()
    energy = float(evals[:k].sum() / evals.sum())
    return dict(sigma_arm=sigma_arm.float(), sigma_g=float(sigma_g),
                U=U.float(), rho=rho.float().clamp_min(1e-6),
                eigvals=evals[:k].float(), energy_k=energy,
                quantile=float(q), rows=int(len(a0e)),
                proj_p50=proj.abs().median(0).values.float())


def stats_sha(st):
    h = hashlib.sha1()
    for k in ("sigma_arm", "U", "rho"):
        h.update(st[k].detach().cpu().contiguous().numpy().tobytes())
    h.update(repr((float(st["sigma_g"]), st["quantile"], st["rows"])).encode())
    return h.hexdigest()[:12]


def rotation(seed, dtype=torch.float32):
    """Блочная ортогональная R [56, 56]: рука (48) и схват (8) отдельно.

    Координаты t упорядочены как [шаг][канал]; индексы руки — каналы 0..5
    каждого шага, схвата — канал 6.
    """
    g = torch.Generator().manual_seed(int(seed))
    idx = torch.arange(D_T).view(H_EXEC, N_CH)
    arm_idx = idx[:, :6].reshape(-1)
    grip_idx = idx[:, 6].reshape(-1)
    R = torch.zeros(D_T, D_T, dtype=torch.float64)
    for ids in (arm_idx, grip_idx):
        n = len(ids)
        q, r = torch.linalg.qr(torch.randn(n, n, generator=g,
                                           dtype=torch.float64))
        q = q * torch.sign(torch.diagonal(r))[None]
        R[ids[:, None], ids[None, :]] = q
    return R.to(dtype)


# --- чистый проход до 18-го слоя ----------------------------------------------
H18_DEPTH = 18


def plain_h18(model, *, vlm_inputs_embeds, attention_mask, position_ids,
              decode):
    """q0, a0 и h18 ИСХОДНОГО замороженного backbone.

    Префикс 0..11 — дословно как в forward_depth_aligned_rvq и
    DepthRefiner.run (тот же cat, та же маска): q0 обязан совпасть с
    каноническим побитово. Слои 12..17 — исходные веса БЕЗ feedback K-14,
    без LoRA и φ K-15d, без replace_prev.
    """
    with torch.no_grad():
        batch = int(vlm_inputs_embeds.shape[0])
        device, dtype = vlm_inputs_embeds.device, vlm_inputs_embeds.dtype
        n_pos = int(model.block_size)
        bos = model.bos_embedding.expand(batch, n_pos, -1).to(device, dtype)
        empty = torch.empty((batch, 0, bos.shape[-1]), device=device,
                            dtype=dtype)
        act = torch.cat([bos, empty], dim=1)
        mask4d = model._build_joint_attention_mask_blockwise_ar(
            attention_mask=attention_mask,
            vlm_seq_len=vlm_inputs_embeds.shape[1], action_seq_len=n_pos,
            device=device, action_key_mask=torch.ones(
                (batch, n_pos), device=device, dtype=torch.long))
        vlm = vlm_inputs_embeds
        q0 = None
        for i in range(H18_DEPTH):
            vlm, act = model._shared_attention_forward(
                vlm_hidden_states=vlm, action_hidden_states=act,
                layer_idx=i, attention_mask=mask4d,
                position_ids=position_ids, past_key_values=None,
                use_cache=False, cache_position=None)
            if i == 11:
                q0 = model._joint_depth_logits(act, 0).argmax(dim=-1)
        a0 = decode(model.depth_aligned_book(0)[q0]).float()
    return q0, a0, act


class RMSNorm(nn.Module):
    """RMS-норма с весом и eps финальной нормы action expert.

    Своя, а не копия модуля модели: предобучение базиса идёт по кэшу без
    загрузки модели, а рука — на модели; класс и параметры обязаны
    совпадать в обоих местах.
    """

    def __init__(self, weight, eps):
        super().__init__()
        # БУФЕР, А НЕ ПАРАМЕТР: норма заморожена. Как параметр она попадала
        # в оптимизатор, менялась при обучении, и гейт (требующий побитового
        # равенства с нормой модели) отказывал бы на любой обученной эпохе.
        self.register_buffer("weight", torch.as_tensor(weight).float()
                             .clone())
        self.eps = float(eps)

    def forward(self, x):
        x = x.float()
        v = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(v + self.eps))


def norm_of(model):
    """(вес, eps) финальной нормы action expert."""
    n = model.action_expert.norm
    eps = getattr(n, "variance_epsilon", getattr(n, "eps", None))
    if eps is None:
        raise RuntimeError("у финальной нормы нет eps")
    return n.weight.detach().float().cpu(), float(eps)


# --- голова базиса -------------------------------------------------------------
class BasisHead(nn.Module):
    """h18 [B, 16, D] и a0 -> базис [B, K, 56] в касательном пространстве."""

    def __init__(self, norm_weight, norm_eps, d_model, n_pos, stats,
                 hp=None):
        super().__init__()
        hp = dict(HP if hp is None else hp)
        self.hp = hp
        self.norm = RMSNorm(norm_weight, norm_eps)
        self.proj = nn.Linear(d_model, hp["proj"])
        self.mlp_in = nn.Linear(n_pos * hp["proj"] + D_T, hp["hidden"])
        self.out = nn.Linear(hp["hidden"], K_BASIS * D_T)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("U", stats["U"].float().clone())
        self.register_buffer("rho", stats["rho"].float().clone())
        self.register_buffer("sigma_arm", stats["sigma_arm"].float().clone())
        self.register_buffer("sigma_g", torch.tensor(float(stats["sigma_g"])))
        self.d_model, self.n_pos = int(d_model), int(n_pos)

    def delta_u(self, h18, a0):
        """w(s) [B, K, 56]: ⊥ U_j, ‖w_j‖ <= κ; нулевой в начальной точке."""
        z = self.proj(self.norm(h18.float())).flatten(1)
        f = a0[:, :H_EXEC].float().reshape(a0.shape[0], -1)
        z = F.gelu(self.mlp_in(torch.cat([z, f], dim=1)))
        raw = self.out(z).view(-1, K_BASIS, D_T)
        U = self.U[None]
        perp = raw - (raw * U).sum(-1, keepdim=True) * U
        n = perp.norm(dim=-1, keepdim=True)
        kappa = float(self.hp["kappa"])
        # κ·tanh(n)/n, в нуле предел κ (а не 0/0)
        scale = torch.where(n > 1e-8, kappa * torch.tanh(n) / n.clamp_min(1e-8),
                            torch.full_like(n, kappa))
        return perp * scale

    def forward(self, h18, a0):
        """Базис δ [B, K, 56] и его нормированная форма u = δ/ρ."""
        u = self.U[None] + self.delta_u(h18, a0)
        return u * self.rho[None, :, None], u

    def act(self, h18, a0, c, R=None):
        """Действие при коэффициентах c; R — контрольное вращение."""
        basis, _u = self(h18, a0)
        if R is not None:
            basis = basis @ R.to(basis.dtype)
        t = compose(basis, c)
        return combine(a0.float(), t, self.sigma_arm, self.sigma_g), basis


def ridge_coeffs(basis, r, lam):
    """c* [B, K] = (ΔΔᵀ + λI)^{-1} Δ r, Δ = basis [B, K, 56], r [B, 56]."""
    G = basis @ basis.transpose(1, 2)
    eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
    rhs = (basis @ r[..., None])
    return torch.linalg.solve(G + lam * eye, rhs)[..., 0]


def basis_loss(head, basis, u, r, hp=None):
    """Реконструкция остатка проекцией + ортогональность + нормы + якорь."""
    hp = dict(HP if hp is None else hp)
    c = ridge_coeffs(basis, r, hp["ridge"])
    rec = torch.einsum("bk,bkd->bd", c, basis)
    l_rec = F.smooth_l1_loss(rec, r, beta=hp["smooth_l1_beta"])
    un = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    gram = un @ un.transpose(1, 2)
    off = gram - torch.diag_embed(torch.diagonal(gram, dim1=1, dim2=2))
    l_ortho = off.pow(2).mean()
    l_norm = (u.norm(dim=-1) - 1.0).pow(2).mean()
    l_anchor = (u - head.U[None]).pow(2).mean()
    total = (l_rec + hp["w_ortho"] * l_ortho + hp["w_norm"] * l_norm
             + hp["w_anchor"] * l_anchor)
    return total, dict(rec=l_rec.detach(), ortho=l_ortho.detach(),
                       norm=l_norm.detach(), anchor=l_anchor.detach(),
                       coeffs=c.detach())


def basis_diagnostics(head, basis, u, r):
    """Объяснённая энергия, обусловленность, нормы, косинусы, якоря."""
    with torch.no_grad():
        c = ridge_coeffs(basis, r, head.hp["ridge"])
        rec = torch.einsum("bk,bkd->bd", c, basis)
        energy = 1.0 - float((r - rec).pow(2).sum() / r.pow(2).sum()
                             .clamp_min(1e-12))
        sv = torch.linalg.svdvals(basis)
        cond = (sv[:, 0] / sv[:, -1].clamp_min(1e-12))
        un = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        gram = un @ un.transpose(1, 2)
        iu = torch.triu_indices(K_BASIS, K_BASIS, 1)
        cos_pairs = gram[:, iu[0], iu[1]].abs()
        anchor_cos = (un * head.U[None]).sum(-1)          # [B, K]
        norms = basis.norm(dim=-1)
        return dict(
            explained_energy=energy,
            cond_median=float(cond.median()), cond_max=float(cond.max()),
            norm_min=float(norms.min()), norm_max=float(norms.max()),
            pair_cos_mean=float(cos_pairs.mean()),
            pair_cos_max=float(cos_pairs.max()),
            anchor_cos_min=[float(x) for x in anchor_cos.min(0).values],
            anchor_cos_p05=[float(x) for x in
                            torch.quantile(anchor_cos, 0.05, dim=0)],
            anchor_cos_nonpos=int((anchor_cos <= 0).sum()),
            delta_u_rms=float((u - head.U[None]).pow(2).mean().sqrt()),
            coeff_abs_p90=[float(x) for x in
                           torch.quantile(c.abs(), 0.9, dim=0)])


def state_sha(module):
    h = hashlib.sha1()
    for k, v in sorted(module.state_dict().items()):
        h.update(f"{k}|{tuple(v.shape)}|{v.dtype}".encode())
        h.update(v.detach().cpu().contiguous().reshape(-1)
                 .view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:12]


# --- самопроверка --------------------------------------------------------------
def _toy(seed=0, n=400, B=6):
    g = torch.Generator().manual_seed(seed)
    a0 = torch.rand(n, 20, 7, generator=g) * 1.6 - 0.8
    act = a0.clone()
    act[:, :H_EXEC, :6] += 0.05 * torch.randn(n, H_EXEC, 6, generator=g)
    flip = torch.rand(n, H_EXEC, generator=g) < 0.1
    act[:, :H_EXEC, 6] = torch.where(flip, -torch.sign(a0[:, :H_EXEC, 6]),
                                     torch.sign(a0[:, :H_EXEC, 6]))
    st = tangent_stats(a0[:, :H_EXEC], act[:, :H_EXEC])
    head = BasisHead(torch.ones(16), 1e-6, 16, 4, st,
                     hp=dict(HP, proj=4, hidden=16))
    h18 = torch.randn(B, 4, 16, generator=g)
    return a0, act, st, head, h18


def selftest():
    a0, act, st, head, h18 = _toy()
    B = h18.shape[0]
    x0 = a0[:B]
    sa, sg = st["sigma_arm"], st["sigma_g"]
    # 1. t = 0 -> a0 побитово; туда-обратно восстанавливает руку и схват
    z = torch.zeros(B, H_EXEC, N_CH)
    assert torch.equal(combine(x0, z, sa, sg), x0)
    t = to_tangent(act[:B], x0, sa, sg)
    back = combine(x0, t, sa, sg)
    assert torch.allclose(back[:, :H_EXEC, :6], act[:B, :H_EXEC, :6],
                          atol=1e-5)
    assert torch.equal(back[:, H_EXEC:], x0[:, H_EXEC:])
    # 2. PCA: ортонормированность якорей, знак, амплитуда положительна
    U = st["U"]
    assert torch.allclose(U @ U.T, torch.eye(K_BASIS), atol=1e-5)
    assert bool((st["rho"] > 0).all()) and 0 < st["energy_k"] <= 1
    # 3. нулевая голова: базис = ρ·U, c = 0 -> a0 побитово
    basis, u = head(h18, x0)
    assert torch.allclose(u, U[None].expand(B, -1, -1))
    a_zero, _ = head.act(h18, x0, torch.zeros(K_BASIS))
    assert torch.equal(a_zero, x0)
    # 4. c = e_j даёт ровно направление j; -e_j — противоположное;
    #    сумма коэффициентов — сумма направлений (в касательном)
    for j in range(K_BASIS):
        e = torch.zeros(K_BASIS)
        e[j] = 1.0
        tp = compose(basis, e)
        assert torch.allclose(tp.reshape(B, -1), basis[:, j])
        assert torch.allclose(compose(basis, -e), -tp)
    e01 = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert torch.allclose(compose(basis, e01).reshape(B, -1),
                          basis[:, 0] + basis[:, 1], atol=1e-6)
    # 5. перестановка базиса вместе с коэффициентами не меняет поправку
    perm = torch.tensor([2, 0, 3, 1])
    c = torch.tensor([0.3, -0.5, 0.7, 0.1])
    assert torch.allclose(compose(basis[:, perm], c[perm]),
                          compose(basis, c), atol=1e-6)
    # 6. порядок строк батча не влияет
    rp = torch.randperm(B)
    a_c, _ = head.act(h18, x0, c)
    a_p, _ = head.act(h18[rp], x0[rp], c)
    assert torch.allclose(a_p, a_c[rp], atol=1e-6)
    # 7. схват ограничен при любых c в [-1, 1]; всё конечно
    for cc in (torch.ones(K_BASIS), -torch.ones(K_BASIS)):
        a_x, _ = head.act(h18, x0, cc)
        off = x0[:, :H_EXEC, 6] - torch.tanh(grip_logit(x0[:, :H_EXEC, 6])
                                             * 0.5)
        assert bool(((a_x[:, :H_EXEC, 6] - off).abs() <= 1 + 1e-5).all())
        assert bool(torch.isfinite(a_x).all())
    # 8. контроль: R ортогональна, блочная (рука/схват не смешиваются),
    #    сохраняет нормы и попарную геометрию
    R = rotation(7)
    assert torch.allclose(R @ R.T, torch.eye(D_T), atol=1e-5)
    idx = torch.arange(D_T).view(H_EXEC, N_CH)
    arm_i, grip_i = idx[:, :6].reshape(-1), idx[:, 6].reshape(-1)
    assert float(R[arm_i][:, grip_i].abs().max()) == 0.0
    bR = basis @ R
    assert torch.allclose(bR.norm(dim=-1), basis.norm(dim=-1), atol=1e-5)
    assert torch.allclose(bR @ bR.transpose(1, 2),
                          basis @ basis.transpose(1, 2), atol=1e-4)
    assert not torch.allclose(bR, basis)
    assert torch.equal(rotation(7), R)               # детерминированность
    # 9. ridge и потеря: обучение уменьшает потерю, градиенты доходят
    r = to_tangent(act[:64], a0[:64], sa, sg).reshape(64, -1)
    h64 = torch.randn(64, 4, 16, generator=torch.Generator().manual_seed(3))
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    first = None
    for step in range(60):
        b_, u_ = head(h64, a0[:64])
        loss, parts = basis_loss(head, b_, u_, r)
        if first is None:
            first = float(parts["rec"])
            loss.backward()
            g_out = float(head.out.weight.grad.abs().sum())
            assert g_out > 0, "градиент не дошёл до выхода головы"
            # КОНТРОЛЬ вырожденности: при нулевом выходе вход головы
            # градиента не получает — проверка выше не тривиальна
            assert float(head.mlp_in.weight.grad.abs().sum()) == 0.0
        else:
            loss.backward()
        opt.step()
        opt.zero_grad()
    b_, u_ = head(h64, a0[:64])
    _l, parts = basis_loss(head, b_, u_, r)
    assert float(parts["rec"]) < first, (float(parts["rec"]), first)
    d = basis_diagnostics(head, b_, u_, r)
    assert 0 <= d["explained_energy"] <= 1 and d["anchor_cos_nonpos"] == 0
    # СТРУКТУРНАЯ ГАРАНТИЯ ЯКОРЯ при любых весах: даже огромный выход
    # головы не опускает косинус ниже 1/sqrt(1 + κ²)
    bound = 1.0 / math.sqrt(1.0 + HP["kappa"] ** 2)
    assert min(d["anchor_cos_min"]) >= bound - 1e-5, d["anchor_cos_min"]
    with torch.no_grad():
        saved = head.out.weight.clone()
        head.out.weight.normal_(0, 50.0)
        _b, u_big = head(h64, a0[:64])
        un = u_big / u_big.norm(dim=-1, keepdim=True)
        assert float((un * head.U[None]).sum(-1).min()) >= bound - 1e-5
        head.out.weight.copy_(saved)
    # 10. ненулевая голова: h18 меняет базис; при c = 0 действие — a0
    b1, _ = head(h64, a0[:64])
    b2, _ = head(h64.roll(1, 0), a0[:64])
    assert not torch.allclose(b1, b2)
    a_z, _ = head.act(h64, a0[:64], torch.zeros(K_BASIS))
    assert torch.equal(a_z, a0[:64].float())
    # 11. норма заморожена: после обучения побитово прежняя, в оптимизатор
    #     не попадает
    a0b, actb, stb, headb, _h = _toy(seed=5)
    w0 = headb.norm.weight.clone()
    names = [n for n, _p in headb.named_parameters()]
    assert "norm.weight" not in names, names
    optb = torch.optim.Adam(headb.parameters(), lr=1e-2)
    rb = to_tangent(actb[:32], a0b[:32], stb["sigma_arm"], stb["sigma_g"]
                    ).reshape(32, -1)
    hb = torch.randn(32, 4, 16)
    for _ in range(5):
        bb, ub = headb(hb, a0b[:32])
        lb, _ = basis_loss(headb, bb, ub, rb)
        lb.backward()
        optb.step()
        optb.zero_grad()
    assert torch.equal(headb.norm.weight, w0), "норма изменилась"
    assert "norm.weight" in headb.state_dict()
    # 12. пределы геометрии ловят схлопывание
    assert geometry_ok(dict(cond_max=2.0, pair_cos_max=0.3))
    assert not geometry_ok(dict(cond_max=50.0, pair_cos_max=0.3))
    assert not geometry_ok(dict(cond_max=2.0, pair_cos_max=0.95))
    print("самопроверка k15f_continuous_refine пройдена: касательное "
          "пространство, PCA-якоря, состав, контроль, ridge, обучаемость")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    print("это модуль")
