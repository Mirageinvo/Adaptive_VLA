#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-18: replay, шум исследования и TD3 для актёра с базисом поправок.

REPLAY хранит наблюдение, ИСПОЛНЯЕМУЮ поправку δ (48, после всех
ограничений), награду, следующее наблюдение, признак завершения, эпизод и
позицию окна; для аудита — коэффициенты, версию поведения и маску
исполненных шагов. Старые переходы никогда не пересчитываются через
текущий базис.

TD3: два критика с целевыми копиями; y = r + (1 − d)·min_i Q_i'(x',
δ'(x')), δ' — целевая копия ВСЕЙ политики (включая базис) с шумом
сглаживания на v; актёр максимизирует Q_1(x, δ_θ(x)) раз в policy_delay
обновлений критиков; τ-обновление целевых сетей вместе с актёром.

ШУМ ИССЛЕДОВАНИЯ: направление равномерно на сфере размерности k, длина
|N(0, σ)| ∧ cap — одно распределение длины для всех вариантов. Ключ —
(сид, задача, старт, номер посещения, позиция окна): длина одинакова у
всех вариантов, направление — у вариантов с одинаковым k.
"""
import copy

import numpy as np

import k18_models as m18


def key_noise(seed, task, start, visit, pos, k, sigma, cap):
    ln = np.random.default_rng([seed, 1, task, start, visit, pos])
    length = min(abs(float(ln.normal(0.0, sigma))), cap)
    dr = np.random.default_rng([seed, 2, k, task, start, visit, pos])
    u = dr.normal(size=k)
    u /= max(np.linalg.norm(u), 1e-12)
    return (u * length).astype(np.float32), length


def sphere_noise_t(torch, n, k, sigma, cap, gen, device):
    u = torch.randn(n, k, generator=gen, device="cpu")
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ln = torch.clamp(torch.randn(n, 1, generator=gen).abs() * sigma,
                     max=cap)
    return (u * ln).to(device)


class Replay:
    KEYS = dict(obs=(m18.OBS_DIM,), delta=(m18.D_EXEC,), rew=(), next=(
        m18.OBS_DIM,), done=(), episode=(), pos=(), version=(), task=(),
        coef=(m18.D_EXEC,), k=(), exec_mask=(m18.H,))
    DT = dict(episode=np.int64, pos=np.int8, version=np.int64, task=np.int8,
              k=np.int8, exec_mask=np.bool_)

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.n = 0
        self.ptr = 0
        self.d = {k: np.zeros((0,) + s, self.DT.get(k, np.float32))
                  for k, s in self.KEYS.items()}

    def _grow(self, need):
        cur = len(self.d["rew"])
        if need <= cur:
            return
        new = min(self.capacity, max(need, 2 * cur, 256))
        for k, s in self.KEYS.items():
            z = np.zeros((new,) + s, self.DT.get(k, np.float32))
            z[:cur] = self.d[k]
            self.d[k] = z

    def add(self, **tr):
        n = len(tr["rew"])
        for i in range(n):
            if self.n < self.capacity:
                self._grow(self.n + 1)
                j = self.n
                self.n += 1
            else:
                j = self.ptr
                self.ptr = (self.ptr + 1) % self.capacity
            for k in self.KEYS:
                v = tr.get(k)
                if v is None:
                    continue
                v = np.asarray(v[i])
                if k == "coef":
                    z = np.zeros(m18.D_EXEC, np.float32)
                    z[:v.size] = v.reshape(-1)
                    v = z
                self.d[k][j] = v

    def sample(self, rng, b):
        idx = rng.integers(0, self.n, b)
        return {k: v[idx] for k, v in self.d.items()}

    def state(self):
        return dict(n=self.n, ptr=self.ptr, capacity=self.capacity,
                    d={k: v[:self.n].copy() for k, v in self.d.items()})

    def load(self, st):
        self.capacity, self.n, self.ptr = st["capacity"], st["n"], st["ptr"]
        self.d = {k: np.array(v) for k, v in st["d"].items()}


class Agent:
    def __init__(self, torch, variant, U, rho, scale, lo, hi, cfg, device,
                 seed=0, batch_seed=None):
        self.torch, self.cfg, self.dev = torch, cfg, device
        M = m18.build(torch)
        self.actor = M["Actor"](variant, U, rho, scale, lo, hi,
                                seed=seed).to(device)
        self.c1 = M["Critic"](seed + 101).to(device)
        self.c2 = M["Critic"](seed + 202).to(device)
        self.actor_t = copy.deepcopy(self.actor)
        self.c1_t, self.c2_t = copy.deepcopy(self.c1), copy.deepcopy(self.c2)
        for m in (self.actor_t, self.c1_t, self.c2_t):
            for p in m.parameters():
                p.requires_grad_(False)
        lr = cfg["lr"]
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(list(self.c1.parameters())
                                      + list(self.c2.parameters()), lr=lr)
        bs = seed + 404 if batch_seed is None else int(batch_seed)
        self.gen = torch.Generator().manual_seed(bs + 1)   # шум сглаживания
        self.rng = np.random.default_rng(bs)               # батчи replay
        self.n_critic = 0
        self.n_actor = 0
        self.version = 0
        self.stats = []

    @property
    def k(self):
        return self.actor.k

    def act(self, obs_np, noise_np=None):
        """Детерминированный выход (+ шум на v) -> numpy-словарь."""
        torch = self.torch
        self.actor.eval()
        with torch.no_grad():
            o = self.actor(torch.as_tensor(obs_np, device=self.dev),
                           None if noise_np is None else
                           torch.as_tensor(noise_np, device=self.dev))
        out = {k: (v.float().cpu().numpy() if v is not None else None)
               for k, v in o.items()}
        if out["mind"] is not None and not (out["mind"] > 1e-6).all():
            raise SystemExit(f"потеря ранга базиса: min|diag R| "
                             f"{float(out['mind'].min()):.2e}")
        for k in ("delta", "c"):
            if not np.isfinite(out[k]).all():
                raise SystemExit(f"нечисловой выход актёра ({k})")
        return out

    def update(self, replay, n_updates, actor_on):
        torch, cfg = self.torch, self.cfg
        F = torch.nn.functional
        rec = []
        if replay.n < cfg["critic_start"]:
            return rec
        for _ in range(int(n_updates)):
            b = replay.sample(self.rng, cfg["batch"])
            t = {k: torch.as_tensor(np.asarray(v, np.float32), device=self.dev)
                 for k, v in b.items() if k in ("obs", "delta", "rew", "next",
                                                "done")}
            with torch.no_grad():
                nz = sphere_noise_t(torch, len(t["rew"]), self.k,
                                    self.cfg["target_sigma"],
                                    self.cfg["target_cap"], self.gen,
                                    self.dev)
                ot = self.actor_t(t["next"], nz)
                if ot["mind"] is not None and not bool(
                        (ot["mind"] > 1e-6).all()):
                    raise SystemExit("потеря ранга целевого базиса")
                q_t = torch.minimum(self.c1_t(t["next"], ot["delta"]),
                                    self.c2_t(t["next"], ot["delta"]))
                y = t["rew"] + (1 - t["done"]) * cfg["gamma"] * q_t
            q1, q2 = self.c1(t["obs"], t["delta"]), self.c2(t["obs"],
                                                             t["delta"])
            lc = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            if not bool(torch.isfinite(lc)):
                raise SystemExit("нечисловая ошибка критиков")
            self.opt_c.zero_grad()
            lc.backward()
            self.opt_c.step()
            self.n_critic += 1
            r = dict(td=float((q1 - y).abs().mean().detach()),
                     q=float(q1.mean().detach()),
                     y=float(y.mean()))
            if actor_on and self.n_critic % cfg["policy_delay"] == 0:
                self.actor.train()
                o = self.actor(t["obs"])
                la = -self.c1(t["obs"], o["delta"]).mean()
                if not bool(torch.isfinite(la)):
                    raise SystemExit("нечисловая ошибка актёра")
                self.opt_a.zero_grad()
                la.backward()
                self.opt_a.step()
                self.n_actor += 1
                r["actor_q"] = float(-la.detach())
                with torch.no_grad():
                    for net, tg in ((self.actor, self.actor_t),
                                    (self.c1, self.c1_t),
                                    (self.c2, self.c2_t)):
                        for p, pt in zip(net.parameters(), tg.parameters()):
                            pt.mul_(1 - cfg["tau"]).add_(cfg["tau"] * p)
            rec.append(r)
        return rec

    def state(self):
        return dict(actor=self.actor.state_dict(),
                    actor_t=self.actor_t.state_dict(),
                    c1=self.c1.state_dict(), c2=self.c2.state_dict(),
                    c1_t=self.c1_t.state_dict(), c2_t=self.c2_t.state_dict(),
                    opt_a=self.opt_a.state_dict(),
                    opt_c=self.opt_c.state_dict(),
                    gen=self.gen.get_state(),
                    rng=self.rng.bit_generator.state,
                    n_critic=self.n_critic, n_actor=self.n_actor,
                    version=self.version)

    def load(self, st):
        for k in ("actor", "actor_t", "c1", "c2", "c1_t", "c2_t"):
            getattr(self, k).load_state_dict(st[k])
        self.opt_a.load_state_dict(st["opt_a"])
        self.opt_c.load_state_dict(st["opt_c"])
        self.gen.set_state(st["gen"])
        self.rng.bit_generator.state = st["rng"]
        self.n_critic, self.n_actor = st["n_critic"], st["n_actor"]
        self.version = st["version"]


# --- самопроверка ------------------------------------------------------------
def _toy_cfg():
    return dict(lr=3e-4, batch=64, tau=0.005, policy_delay=2, utd=4,
                gamma=1.0, capacity=100000, critic_start=128,
                target_sigma=0.10, target_cap=0.20)


def toy_episode(rng, agent, w_star, noise, U):
    """Три решения; наблюдение случайно; итог — успех с вероятностью
    sigmoid(30·⟨Σ δ_k, w*⟩ − 1)."""
    obs = rng.standard_normal((3, m18.OBS_DIM)).astype(np.float32) * .3
    obs[:, m18.SL_A0] = 0
    obs[:, -3:] = np.eye(3, dtype=np.float32)
    ds, cs = [], []
    for k in range(3):
        nz = None
        if noise:
            nz, _l = key_noise(7, 0, int(rng.integers(1e9)), 0, k, agent.k,
                               .15, .3)
            nz = nz[None]
        o = agent.act(obs[k:k + 1], nz)
        ds.append(o["delta"][0])
        cs.append(o["c"][0])
    s = sum(float(d @ w_star) for d in ds)
    p = 1 / (1 + np.exp(-(30 * s - 1)))
    succ = float(rng.random() < p)
    tr = dict(obs=obs, delta=np.stack(ds), rew=np.array([0, 0, succ]),
              next=np.concatenate([obs[1:], obs[2:]]),
              done=np.array([0, 0, 1.0]), episode=np.zeros(3),
              pos=np.arange(3), coef=np.stack(cs), version=np.zeros(3),
              task=np.zeros(3), k=np.full(3, agent.k),
              exec_mask=np.ones((3, 8), bool))
    return tr, p


def selftest():
    import tempfile
    import torch
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.standard_normal((48, 4)))
    scale = np.array([.9375, .91, .9375, .2036, .2636, .375, 1.0])
    lo, hi = -np.ones(6), np.ones(6)
    cfg = _toy_cfg()
    # шум: одинаковая длина у всех k, одинаковое направление при равном k
    a, la = key_noise(1, 2, 3, 0, 1, 4, .15, .3)
    b, lb = key_noise(1, 2, 3, 0, 1, 48, .15, .3)
    c, _lc = key_noise(1, 2, 3, 0, 1, 4, .15, .3)
    assert la == lb and np.allclose(np.linalg.norm(a), np.linalg.norm(b))
    assert np.array_equal(a, c) and la <= .3
    d, _ld = key_noise(1, 2, 3, 1, 1, 4, .15, .3)
    assert not np.array_equal(a, d)
    # replay: старое действие не меняется после изменения базиса
    ag = Agent(torch, "basis4_rl", U, .7068, scale, lo, hi, cfg, "cpu")
    with torch.no_grad():
        ag.actor.coef.weight.normal_(0, .2)
        ag.actor.coef.bias.normal_(0, .2)
    rp = Replay(1000)
    tr, _p = toy_episode(rng, ag, np.zeros(48), False, U)
    rp.add(**tr)
    before = rp.d["delta"][:3].copy()
    with torch.no_grad():
        ag.actor.V.weight.normal_(0, .5)
    assert np.array_equal(rp.d["delta"][:3], before)
    o_new = ag.act(tr["obs"])
    assert not np.allclose(o_new["delta"], before)
    # целевой переход использует ЦЕЛЕВОЙ базис: изменение основной V не
    # меняет y; изменение целевой — меняет
    for _ in range(60):
        rp.add(**toy_episode(rng, ag, np.zeros(48), True, U)[0])
    tt = {k: torch.as_tensor(np.asarray(v[:32], np.float32))
          for k, v in rp.d.items() if k in ("next",)}
    g0 = torch.Generator().manual_seed(1)
    nz = sphere_noise_t(torch, 32, 4, .1, .2, g0, "cpu")
    with torch.no_grad():
        y1 = ag.actor_t(tt["next"], nz)["delta"].clone()
        ag.actor.V.weight.add_(1.0)
        y2 = ag.actor_t(tt["next"], nz)["delta"].clone()
        ag.actor_t.V.weight.normal_(0, .5)
        y3 = ag.actor_t(tt["next"], nz)["delta"].clone()
    assert torch.equal(y1, y2) and not torch.equal(y1, y3)
    # сохранение и загрузка восстанавливают обучение
    ag2 = Agent(torch, "basis4_rl", U, .7068, scale, lo, hi, cfg, "cpu")
    ag2.update(rp, 5, True)
    st = copy.deepcopy(ag2.state())
    rs = copy.deepcopy(rp.state())
    r_a = ag2.update(rp, 6, True)
    ag3 = Agent(torch, "basis4_rl", U, .7068, scale, lo, hi, cfg, "cpu",
                seed=9)
    ag3.load(st)
    rp3 = Replay(10)
    rp3.load(rs)
    r_b = ag3.update(rp3, 6, True)
    assert r_a == r_b, (r_a, r_b)
    with tempfile.TemporaryDirectory() as td:
        torch.save(dict(a=ag3.state(), r=rp3.state()), td + "/s.pt")
        z = torch.load(td + "/s.pt", weights_only=False)
        assert z["r"]["n"] == rp3.n
    # игрушечная задача: полный цикл сбора и обновлений повышает успех
    w_star = U @ np.array([1.0, -1.0, .5, 0]) / 1.5
    for variant in ("pca4_rl", "basis4_rl", "full48_rl"):
        ag = Agent(torch, variant, U, .7068, scale, lo, hi, cfg, "cpu")
        rp = Replay(100000)
        p0 = np.mean([toy_episode(np.random.default_rng(i), ag, w_star,
                                  False, U)[1] for i in range(20)])
        rng = np.random.default_rng(1)
        for ep in range(220):
            tr, _p = toy_episode(rng, ag, w_star, True, U)
            rp.add(**tr)
            ag.update(rp, 4 * 3, ep >= 20)
        p1 = np.mean([toy_episode(np.random.default_rng(i), ag, w_star,
                                  False, U)[1] for i in range(20)])
        assert p1 > p0 + 0.2, (variant, p0, p1)
        print(f"    игрушка {variant}: вероятность успеха {p0:.2f} -> "
              f"{p1:.2f}")
    print("самопроверка k18_td3 пройдена: шум по ключу, replay хранит "
          "исполненную δ, целевой базис, сохранение и загрузка, игрушечная "
          "задача улучшается у всех вариантов")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
