#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-32: рука B = 1 — reference (R) и обучаемые T / P / F.

Вызов: тяжёлый проход k29_model.export (fp16 autocast, no_grad, TF32 выкл.),
FP32 reference (Z0, рука, логит схвата); путь и c_r (k32_operator.path_info);
сырой вектор S (k32_model.state_vector); actor на CPU в FP32 по трём узлам ->
logits [3], μ [3, d]. Стохастически (сбор): r ~ Categorical(p), η ~ N(μ_r,
σ²I) из локальных потоков (seed, оператор, эпизод, вызов; маршрут и шум —
разные потоки). Детерминированно (оценка): r = argmax p, η = μ_r. Оператор
-> ограничитель -> рука; схват — всегда знак логита reference. η ≡ 0 ->
рука reference побитово (детерминированная начальная политика = R).
Журнал вызова: S, logits, μ, r, η, log π, путь/c/центры, D, Du, флаги
ограничителя, рука, команды.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k32_dist as kd  # noqa: E402
import k32_model as km32  # noqa: E402
import k32_operator as ko  # noqa: E402

N8 = ko.N8
KINDS = ("R", "T", "P", "F")


def gens(space, seed, episode, call):
    """Потоки маршрута и шума — без оператора в ключе: T/P/F одного сида
    получают одинаковые потоки (F берёт больше нормалей из своего потока
    шума, поток маршрута при этом не сдвигается)."""
    base = [32, int(space), int(seed), int(episode), int(call)]
    return (np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                base + [1]))),
            np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                base + [2]))))


class Arm:
    def __init__(self, torch, mod, dec, M, *, kind, seed, actor=None,
                 stochastic=False, space=0):
        if kind not in KINDS:
            raise ValueError(kind)
        if kind != "R" and actor is None:
            raise SystemExit(f"{kind}: нужен actor")
        import k31_readout as kr
        self.torch, self.mod, self.dec, self.M = torch, mod, dec, M
        self.E = kr.build(torch)
        self.kind, self.seed, self.actor = kind, int(seed), actor
        self.stochastic, self.space = bool(stochastic), int(space)
        self.dev = next(mod.parameters()).device
        self.H = int(mod.H)
        self.L = np.asarray(mod._L, np.float64)[:6]
        self.U = np.asarray(mod._U, np.float64)[:6]
        self.half, self.mid = self.E["half_mid"](dec._L, dec._U, "cpu")
        self.autocast = torch.autocast(self.dev.type, dtype=torch.float16,
                                       enabled=self.dev.type == "cuda")
        self.episode = 0
        self.call = 0
        self.task = self.init = None
        self.last = None

    def begin(self, task, init, episode=None):
        self.task, self.init = int(task), int(init)
        if episode is not None:
            self.episode = int(episode)
        self.call = 0
        self.last = None

    def forward(self, batch, po, state8):
        import k21_model as km
        import k25_policy as kp25
        torch = self.torch
        s11 = torch.as_tensor(km.state11(np.asarray(state8, np.float64)),
                              device=self.dev)
        with torch.no_grad(), self.autocast:
            F, s11n, cnt = self.M["export"](self.mod, batch, po, s11)
        pr = kp25.counters_problems(cnt)
        if pr:
            raise SystemExit("тяжёлый проход: " + "; ".join(pr))
        with torch.no_grad():
            ref = self.dec(F)
        return F, s11n, ref

    def decide(self, batch, po, state8, elapsed):
        torch = self.torch
        self.call += 1
        F, s11n, ref = self.forward(batch, po, state8)
        if getattr(self, "keep_first", False) and self.call == 1:
            self.first_F = {k: v.detach().cpu() for k, v in F.items()}
        rc = dict(Z0=ref["Z"][0].detach().float().cpu(),
                  arm=ref["arm"][0, :N8].detach().float().cpu(),
                  half=self.half, mid=self.mid)
        gl = ref["grip_logit"][0, :N8].detach().float().cpu()
        grip = torch.where(gl > 0, 1.0, -1.0)
        rec = dict(task=self.task, init=self.init, episode=self.episode,
                   call=self.call, elapsed=int(elapsed), Z0=rc["Z0"].numpy(),
                   ref_arm=rc["arm"].numpy(), grip_logit=gl.numpy())
        arm = rc["arm"]
        if self.kind != "R":
            pi = ko.path_info(F)
            S = km32.state_vector(F, s11n[0].float().cpu().numpy(), elapsed,
                                  rc["Z0"].numpy(), self.half.numpy(), pi)
            X = torch.as_tensor(km32.actor_inputs_np(S[None], self.H),
                                dtype=torch.float32)
            with torch.no_grad(), torch.autocast("cpu", enabled=False):
                lg, raw = self.actor(X)
                mu = kd.mu_of(torch, raw)
            lg_n, mu_n = lg[0].numpy(), mu[0].numpy()
            if self.stochastic:
                gr, gg = gens(self.space, self.seed, self.episode, self.call)
                r, eta, _p = kd.sample(lg_n, mu_n, gr, gg)
            else:
                r, eta = kd.deterministic(lg_n, mu_n)
            with torch.no_grad():
                lp = float(kd.log_prob(torch, lg, mu,
                                       torch.as_tensor([r]),
                                       torch.as_tensor(eta[None]))[0])

            def edited(rr, e):
                import k31_readout as kr
                o = self.E["edited"](self.dec, F, ref, [kr.LEVELS[rr]],
                                     [pi["path"][rr]], np.asarray(e)[None])
                return o["Zhat"][0].detach().float().cpu()
            arm, oj = ko.command(torch, self.kind, r, eta, rc, pi, edited)
            rec.update(w_r=pi["w"][r][:N8].copy(), Zhat=oj.get("Zhat"),
                       S=S, logits=lg_n, mu=mu_n, r=int(r), eta=eta,
                       logpi=lp, path=list(pi["path"]), c=pi["c"],
                       floor=pi["floor"], zero=oj["zero"], D=oj["D"],
                       Du=oj["Du"], clipped=oj["lim"]["clipped"],
                       capped=oj["lim"]["capped"], scale=oj["lim"]["scale"],
                       rms=oj["lim"]["rms"])
        a = torch.cat([arm, grip[:, None]], -1).numpy().astype(np.float32)
        if not np.isfinite(a).all():
            raise SystemExit("команды не конечны")
        if ((a[:, :6] < self.L - 1e-4) | (a[:, :6] > self.U + 1e-4)).any():
            raise SystemExit("рука вне пределов кодека")
        rec["a_codec"] = a
        self.last = dict(rec=rec)
        return a, rec

    def executed(self, k):
        self.last["rec"]["k"] = int(k)


def selftest():
    import torch
    import k21_model as km
    import k25_model as k25m
    import k29_model as k29m
    k29m.no_tf32(torch)
    M = k29m.build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    mod = k25m.build(torch)["K25"](km.fake_bar(torch), L, U, seed=0).eval()
    with torch.no_grad():
        g = torch.Generator().manual_seed(3)
        for p in list(mod.cond.parameters()) + list(mod.readout.parameters()):
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    for p in mod.parameters():
        p.requires_grad_(False)
    dec = M["Decoder"](mod).freeze()
    H = mod.H
    norm = dict(actor_mean=np.zeros(4 * H + 67), actor_std=np.ones(4 * H + 67),
                critic_mean=np.zeros(6 * H + 72),
                critic_std=np.ones(6 * H + 72))
    bt = km.fake_batch(torch, B=1, seed=2)
    st8 = np.zeros((1, 8))
    R = Arm(torch, mod, dec, M, kind="R", seed=0)
    R.begin(8, 3)
    aR, _ = R.decide(bt, 1, st8, 0)
    for op in ("T", "P", "F"):
        act, _c = km32.build(torch, H, ko.DIM[op], 0, norm)
        det = Arm(torch, mod, dec, M, kind=op, seed=0, actor=act)
        det.begin(8, 3)
        aD, rD = det.decide(bt, 1, st8, 0)
        assert np.array_equal(aD, aR) and rD["zero"], op   # init = R
        sto = Arm(torch, mod, dec, M, kind=op, seed=0, actor=act,
                  stochastic=True)
        sto.begin(8, 3, episode=7)
        aS, rS = sto.decide(bt, 1, st8, 0)
        assert not np.array_equal(aS, aR) and not rS["zero"]
        assert np.array_equal(aS[:, 6], aR[:, 6])           # схват reference
        assert rS["rms"] <= ko.RMS_CAP + 1e-7
        sto2 = Arm(torch, mod, dec, M, kind=op, seed=0, actor=act,
                   stochastic=True)
        sto2.begin(8, 3, episode=7)
        aS2, rS2 = sto2.decide(bt, 1, st8, 0)
        assert np.array_equal(aS, aS2) and rS["r"] == rS2["r"]
        # log π записан по полному η и равен пересчёту
        lp = kd.log_prob(torch, torch.as_tensor(rS["logits"][None]),
                         torch.as_tensor(rS["mu"][None]),
                         torch.as_tensor([rS["r"]]),
                         torch.as_tensor(rS["eta"][None]))
        assert abs(float(lp[0]) - rS["logpi"]) < 1e-6
        assert rS["eta"].shape == (ko.DIM[op],)
        # другой эпизод — другой шум
        sto3 = Arm(torch, mod, dec, M, kind=op, seed=0, actor=act,
                   stochastic=True)
        sto3.begin(8, 3, episode=8)
        assert not np.array_equal(sto3.decide(bt, 1, st8, 0)[1]["eta"],
                                  rS["eta"])
    # T и P при одном seed — одинаковые маршрут и η (общие потоки и веса)
    aT, _ = km32.build(torch, H, 6, 0, norm)
    aP, _ = km32.build(torch, H, 6, 0, norm)
    t = Arm(torch, mod, dec, M, kind="T", seed=0, actor=aT, stochastic=True)
    p_ = Arm(torch, mod, dec, M, kind="P", seed=0, actor=aP, stochastic=True)
    t.begin(0, 1, 2)
    p_.begin(0, 1, 2)
    rt, rp = t.decide(bt, 1, st8, 0)[1], p_.decide(bt, 1, st8, 0)[1]
    assert rt["r"] == rp["r"] and np.array_equal(rt["eta"], rp["eta"])
    aF, _ = km32.build(torch, H, 48, 0, norm)
    f_ = Arm(torch, mod, dec, M, kind="F", seed=0, actor=aF, stochastic=True)
    f_.begin(0, 1, 2)
    rf = f_.decide(bt, 1, st8, 0)[1]
    assert rf["r"] == rt["r"] and np.array_equal(rf["eta"][:6], rt["eta"])
    print("самопроверка k32_policy пройдена: детерминированная начальная "
          "политика T/P/F = reference побитово, стохастическая — ограниченная "
          "поправка, схват reference, воспроизводимость по (seed, эпизод, "
          "вызов), log π по полному η = пересчёту, другой эпизод — другой "
          "шум; T/P/F одного сида — общие потоки (тот же маршрут, у F "
          "первые 6 нормалей совпадают с T)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
