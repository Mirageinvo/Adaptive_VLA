#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-29: гладкий шум AR(1) исполненного окна, точная плотность префикса и KL.

Для каждого из 6 каналов руки, j = 0…7 (шаги исполненного окна):
  d_0 = σ ε_0,  d_j = ρ d_{j−1} + σ √(1−ρ²) ε_j,  Y_j = μ_j + d_j,
σ = 0.03 (сырые координаты кодека, до tanh), ρ = 0.8; ε — N(0, 1).
Белизна: e_0 = d_0/σ, e_j = (d_j − ρ d_{j−1}) / (σ√(1−ρ²)).
log π(Y_{0:k} | s) = −½ Σ_{j<k,c} e² − 3k log(2π) − 6k log σ
                     − 3(k−1) log(1−ρ²)       (6 каналов; k = 1…8).
KL(a‖b) при общей ковариации = ½ ‖whiten(μ_a − μ_b)‖² по тому же префиксу.
Всё — float32/float64 вне autocast; маска — префикс длины k, хвост окна и
шаги 8…19 в плотность не входят. Ratio — exp разности суммарных log_prob.
"""
import math
import sys

import numpy as np

SIGMA = 0.03
RHO = 0.8
N_EXEC = 8
CH = 6
CONFIG = dict(sigma=SIGMA, rho=RHO, n_exec=N_EXEC, channels=CH,
              space="raw Z (до tanh)", noise="AR(1) по шагам, каналы "
              "независимы, заново на каждый чанк")


def sample_noise(gen, n, sigma=SIGMA, rho=RHO):
    """np.random.Generator -> d [n, 8, 6] (float64)."""
    eps = gen.standard_normal((n, N_EXEC, CH))
    d = np.zeros_like(eps)
    d[:, 0] = sigma * eps[:, 0]
    s = sigma * math.sqrt(1 - rho * rho)
    for j in range(1, N_EXEC):
        d[:, j] = rho * d[:, j - 1] + s * eps[:, j]
    return d


def prefix_mask(torch, k, n=N_EXEC, device=None):
    """k [B] (1…8) -> маска [B, 8] (float) первых k шагов."""
    k = torch.as_tensor(k, device=device)
    return (torch.arange(n, device=k.device)[None] < k[:, None]).to(
        torch.float32)


def whiten(torch, d, sigma=SIGMA, rho=RHO):
    """d [B, 8, 6] -> e [B, 8, 6] (белые компоненты)."""
    e0 = d[:, :1] / sigma
    e1 = (d[:, 1:] - rho * d[:, :-1]) / (sigma * math.sqrt(1 - rho * rho))
    return torch.cat([e0, e1], 1)


def log_prob(torch, Y, mu, k, sigma=SIGMA, rho=RHO):
    """Суммарная log-плотность префикса Y[:, :k] при среднем mu -> [B]."""
    e = whiten(torch, Y - mu, sigma, rho)
    m = prefix_mask(torch, k, device=Y.device).to(e.dtype)
    kf = m.sum(1)
    quad = -0.5 * ((e * e).sum(-1) * m).sum(1)
    return quad - kf * (CH / 2) * math.log(2 * math.pi) - \
        kf * CH * math.log(sigma) - \
        (kf - 1).clamp_min(0) * (CH / 2) * math.log(1 - rho * rho)


def kl(torch, mu_a, mu_b, k, sigma=SIGMA, rho=RHO):
    """KL(a‖b) префикса длины k при общей ковариации AR(1) -> [B]."""
    e = whiten(torch, mu_a - mu_b, sigma, rho)
    m = prefix_mask(torch, k, device=mu_a.device).to(e.dtype)
    return 0.5 * ((e * e).sum(-1) * m).sum(1)


def cov_1d(k, sigma=SIGMA, rho=RHO):
    """Ковариация одного канала на k шагах (стационарный AR(1))."""
    i = np.arange(k)
    return sigma ** 2 * rho ** np.abs(i[:, None] - i[None, :])


def selftest():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    g = np.random.default_rng(0)
    # ковариация положительно определена; эмпирическая ≈ теоретической
    for k in range(1, N_EXEC + 1):
        assert np.linalg.eigvalsh(cov_1d(k)).min() > 0
    d = sample_noise(g, 200000)
    emp = np.cov(d[:, :, 0].T)
    assert np.abs(emp - cov_1d(N_EXEC)).max() < 3e-5, \
        np.abs(emp - cov_1d(N_EXEC)).max()
    # log_prob префикса = полной многомерной нормали (6 каналов независимы)
    B = 7
    mu = torch.as_tensor(g.normal(0, 0.5, (B, N_EXEC, CH)), dtype=torch.float64)
    Y = mu + torch.as_tensor(sample_noise(g, B))
    for k in range(1, N_EXEC + 1):
        kk = torch.full((B,), k)
        lp = log_prob(torch, Y, mu, kk)
        C = torch.as_tensor(cov_1d(k))
        ref = torch.zeros(B, dtype=torch.float64)
        for c in range(CH):
            mvn = torch.distributions.MultivariateNormal(mu[:, :k, c], C)
            ref = ref + mvn.log_prob(Y[:, :k, c])
        assert float((lp - ref).abs().max()) < 1e-9, (k, lp, ref)
        # KL = аналитическому KL двух гауссиан
        mb = mu + 0.05
        klv = kl(torch, mu, mb, kk)
        ref_kl = torch.zeros(B, dtype=torch.float64)
        for c in range(CH):
            a = torch.distributions.MultivariateNormal(mu[:, :k, c], C)
            b = torch.distributions.MultivariateNormal(mb[:, :k, c], C)
            ref_kl = ref_kl + torch.distributions.kl_divergence(a, b)
        assert float((klv - ref_kl).abs().max()) < 1e-9
    # изменение Y и mu после k не меняет log_prob и KL
    k3 = torch.full((B,), 3)
    Y2 = Y.clone()
    Y2[:, 3:] += 5.0
    mu2 = mu.clone()
    mu2[:, 3:] -= 2.0
    assert torch.equal(log_prob(torch, Y, mu, k3), log_prob(torch, Y2, mu2, k3))
    assert torch.allclose(kl(torch, mu, mu2, k3), torch.zeros(B, dtype=mu.dtype))
    # ratio при одинаковых параметрах = 1; ratio — exp разности сумм
    r = torch.exp(log_prob(torch, Y, mu, k3) - log_prob(torch, Y, mu, k3))
    assert torch.equal(r, torch.ones_like(r))
    # не равно произведению независимых нормалей (корреляция по времени)
    ind = torch.distributions.Normal(mu[:, :3], SIGMA).log_prob(Y[:, :3]).sum(
        (1, 2))
    assert float((ind - log_prob(torch, Y, mu, k3)).abs().max()) > 1e-3
    # градиент идёт через mu, Y константно
    m = mu.clone().requires_grad_(True)
    log_prob(torch, Y, m, k3).sum().backward()
    assert m.grad[:, 3:].abs().max() == 0 and m.grad[:, :3].abs().max() > 0
    print("самопроверка k29_dist пройдена: ковариация AR(1) PD и совпадает с "
          "эмпирической, log_prob префикса k = 1…8 = полной многомерной "
          "нормали, KL = аналитическому, хвост после k не влияет, ratio = 1 "
          "при равных параметрах, ≠ независимой модели, градиент только "
          "через μ префикса")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
