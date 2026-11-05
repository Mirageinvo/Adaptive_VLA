#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-32: латентная политика (r, η) — выбор узла и гауссова поправка.

  p = softmax(logits) [3];  μ_r = 2·tanh(raw_μ_r/2) [3, d];
  r ~ Categorical(p);  η ~ N(μ_r, σ²I_d), σ = 0.5.
log π(r, η) = log p_r − d/2·log(2πσ²) − Σ(η − μ_r)²/(2σ²) — по ВСЕМУ η
(d = 6 для T/P, 48 для F; хвост F не обрезается по k).
KL_joint(θ‖b) = Σ_r p_θ,r [log(p_θ,r/p_b,r) + Σ_j (μ_θ,r,j − μ_b,r,j)²/(2σ²)]
— по всем трём маршрутам. Референс для штрафа — начальная стохастическая
политика: равномерный маршрут и μ = 0.
Сэмплирование — локальные np.random.Generator: маршрут и гауссов шум —
разные потоки (d = 48 не сдвигает поток маршрута). Оценка: r = argmax p
(ничья — root→mid→leaf, т. е. меньший индекс), η = μ_r.
"""
import math
import sys

import numpy as np

SIGMA = 0.5
N_ROUTES = 3
CONFIG = dict(sigma=SIGMA, routes=N_ROUTES, mu_squash="2*tanh(raw/2)",
              density="categorical × N(η; μ_r, σ²I_d), весь η")


def mu_of(torch, raw):
    return 2.0 * torch.tanh(raw / 2.0)


def log_prob(torch, logits, mu, r, eta, sigma=SIGMA):
    """logits [N,3], mu [N,3,d], r [N] (long), eta [N,d] -> [N]."""
    d = mu.shape[-1]
    lp_r = torch.log_softmax(logits, -1).gather(1, r[:, None])[:, 0]
    m = mu.gather(1, r[:, None, None].expand(-1, 1, d))[:, 0]
    q = ((eta - m) ** 2).sum(-1)
    return lp_r - 0.5 * d * math.log(2 * math.pi * sigma ** 2) - \
        q / (2 * sigma ** 2)


def kl_joint(torch, logits_a, mu_a, logits_b, mu_b, sigma=SIGMA):
    """KL(a‖b) по всем маршрутам -> [N]."""
    la = torch.log_softmax(logits_a, -1)
    lb = torch.log_softmax(logits_b, -1)
    pa = la.exp()
    g = ((mu_a - mu_b) ** 2).sum(-1) / (2 * sigma ** 2)       # [N,3]
    return (pa * (la - lb + g)).sum(-1)


def init_reference(torch, n, d, device=None):
    """Начальная стохастическая политика: logits 0, μ 0."""
    return (torch.zeros(n, N_ROUTES, device=device),
            torch.zeros(n, N_ROUTES, d, device=device))


def sample(logits, mu, g_route, g_gauss, sigma=SIGMA):
    """Одно наблюдение: logits [3], mu [3,d] (numpy float64) -> (r, η)."""
    lg = np.asarray(logits, np.float64)
    p = np.exp(lg - lg.max())
    p /= p.sum()
    u = g_route.random()
    r = int(min(np.searchsorted(np.cumsum(p), u, side="right"), 2))
    m = np.asarray(mu, np.float64)[r]
    eta = m + sigma * g_gauss.standard_normal(m.shape[0])
    return r, eta.astype(np.float32), p


def deterministic(logits, mu):
    lg = np.asarray(logits, np.float64)
    r = int(np.argmax(lg))                  # первый максимум: root→mid→leaf
    return r, np.asarray(mu, np.float32)[r].copy()


def selftest():
    import torch
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    for d in (6, 48):
        N = 5
        lg = torch.randn(N, 3, generator=g, dtype=torch.float64)
        mu = mu_of(torch, torch.randn(N, 3, d, generator=g,
                                      dtype=torch.float64))
        r = torch.randint(0, 3, (N,), generator=g)
        eta = torch.randn(N, d, generator=g, dtype=torch.float64)
        lp = log_prob(torch, lg, mu, r, eta)
        # аналитика: категориальная × многомерная нормаль
        for i in range(N):
            ref = float(torch.log_softmax(lg[i], -1)[r[i]]) + float(
                torch.distributions.MultivariateNormal(
                    mu[i, r[i]], SIGMA ** 2 * torch.eye(d,
                                                        dtype=torch.float64))
                .log_prob(eta[i]))
            assert abs(float(lp[i]) - ref) < 1e-9, (d, i)
        # ratio = 1 при равных параметрах; замена r/η/μ/logits меняет log π
        assert torch.equal(torch.exp(lp - log_prob(torch, lg, mu, r, eta)),
                           torch.ones(N, dtype=torch.float64))
        assert not torch.equal(lp, log_prob(torch, lg, mu, (r + 1) % 3, eta))
        assert not torch.equal(lp, log_prob(torch, lg, mu, r, eta + 0.1))
        assert not torch.equal(lp, log_prob(torch, lg + torch.tensor(
            [1.0, 0, 0], dtype=torch.float64), mu, r, eta))
        # KL: аналитика = Монте-Карло, все маршруты, ≥ 0, 0 при равенстве
        lg2 = lg + 0.3 * torch.randn(N, 3, generator=g, dtype=torch.float64)
        mu2 = mu + 0.2 * torch.randn(N, 3, d, generator=g,
                                     dtype=torch.float64)
        kl = kl_joint(torch, lg, mu, lg2, mu2)
        assert (kl >= 0).all()
        assert torch.allclose(kl_joint(torch, lg, mu, lg, mu),
                              torch.zeros(N, dtype=torch.float64))
        rng = np.random.default_rng(3)
        M = 40000
        for i in range(2):
            rs, es = [], []
            for _ in range(M):
                rr, ee, _p = sample(lg[i].numpy(), mu[i].numpy(), rng, rng)
                rs.append(rr)
                es.append(ee)
            rt = torch.as_tensor(rs)
            et = torch.as_tensor(np.stack(es), dtype=torch.float64)
            la = log_prob(torch, lg[i].expand(M, 3), mu[i].expand(M, 3, d),
                          rt, et)
            lb = log_prob(torch, lg2[i].expand(M, 3), mu2[i].expand(M, 3, d),
                          rt, et)
            mc = (la - lb)
            se = float(mc.std()) / math.sqrt(M)
            assert abs(float(mc.mean()) - float(kl[i])) < 5 * se + 1e-3, \
                (d, float(mc.mean()), float(kl[i]), se)
        # KL учитывает и невыбранные маршруты
        mu3 = mu.clone()
        mu3[:, 2] += 1.0
        lg3 = lg.clone()
        lg3[:, 2] = -50.0
        assert float(kl_joint(torch, lg3, mu3, lg3, mu).min()) < 1e-6
        assert float(kl_joint(torch, lg, mu3, lg, mu).min()) > 0
    # потоки: d = 48 не сдвигает маршрут
    r6 = [sample(np.zeros(3), np.zeros((3, 6)), np.random.default_rng(5),
                 np.random.default_rng(9))[0] for _ in range(3)]
    r48 = [sample(np.zeros(3), np.zeros((3, 48)), np.random.default_rng(5),
                  np.random.default_rng(9))[0] for _ in range(3)]
    assert r6 == r48
    assert deterministic(np.zeros(3), np.zeros((3, 6)))[0] == 0
    assert deterministic(np.array([0, 2, 2.0]), np.zeros((3, 6)))[0] == 1
    print("самопроверка k32_dist пройдена: log π = категориальная × "
          "многомерная нормаль (d = 6 и 48), ratio = 1 при равных "
          "параметрах, замена r/η/μ/logits меняет log π, joint KL = "
          "Монте-Карло (5 SE), учитывает все маршруты, потоки маршрута и "
          "шума независимы, детерминированный выбор с ничьей root→mid→leaf")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
