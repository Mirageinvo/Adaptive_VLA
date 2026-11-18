#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-37: оператор «родитель плюс лист» поверх замороженного T32 (K-32).

Шаг родителя — штатный k32_policy.Arm.decide (T, детерминированно); из того
же вызова перехватываются F, s11n, reference. Лист:
  - маршрут T32 = leaf или поправка родителя нулевая -> команда T32;
  - иначе потомки-листья выбранного узла (k31_readout.descendants); для
    каждого early = mean W2[0:4, l], late = mean W2[4:8, l] (float64);
    подходят листья с late > early; выбирается наибольший late, ничья —
    меньший индекс; нет подходящих -> команда T32.
Поправка листа: w(t) = W2[t, l], t < 8; c_l = max(0.05, √(mean w²))
  (float64, затем округление w и c_l до float32 — эти же значения пишутся в
  журнал); D_l(t) = w(t)/c_l · 0.025·tanh(u) (float64 -> float32).
Сборка (координаты K-32): Du = limit32(D_parent + D_l) — D_parent = rec["D"]
  ДО ограничителя; arm = mid + half·tanh(Z0[:8]/half + Du); схват — T32.
  u ≡ 0 -> штатный a_codec T32 напрямую (побитово).
Эталон (тест): правка родителя k31_readout.edited -> z2; к выбранному листу
  + half·e_l, e_l = 0.025·tanh(u)/c_l; Zhat = Σ_l W2·z2; D = (Zhat − Z0)/half.
Признаки листа: вход actor K-32 для узла «лист l» (средние, x2[l], s11n,
  elapsed/600, Z0/half, one-hot leaf, центр/ширина/20, масса, c_l) -> его
  нормировка, l1-GELU-l2-GELU -> 32 числа перед actor.out.
Вход головки (96): 32 признака, η_parent/2 (6), Du_parent/0.04 (48),
  w/c_l (8), индикаторы уровня родителя root/mid (2). Головка — Linear(96, 6)
  с нулевыми весами и смещением (582 числа).
"""
import sys

import numpy as np

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k32_operator as ko  # noqa: E402

LEAF_ALPHA = 0.025
N8 = 8
N_IN, N_OUT = 96, 6
REASON = dict(active=0, leaf_route=1, zero_parent=2, no_late=3)


def choose_leaf(W2, r, path):
    """W2 [T, 16] одного наблюдения, r — уровень маршрута T32 (0 root,
    1 mid, 2 leaf), path (root, mid, leaf) -> (лист | None, причина)."""
    import k31_readout as kr
    if int(r) == 2:
        return None, REASON["leaf_route"]
    lev = kr.LEVELS[int(r)]
    _rr, _mm, leaves = kr.descendants(lev, int(path[int(r)]))
    W = np.asarray(W2, np.float64)
    best, best_late = None, None
    for l in sorted(leaves):
        early, late = W[0:4, l].mean(), W[4:8, l].mean()
        if late > early and (best is None or late > best_late):
            best, best_late = int(l), late
    if best is None:
        return None, REASON["no_late"]
    return best, REASON["active"]


def leaf_profile(W2, l):
    w = np.asarray(W2, np.float64)[:N8, int(l)]
    c = max(ko.C_FLOOR, float(np.sqrt((w ** 2).mean())))
    return w, c


def leaf_delta(w, c, u):
    u = np.asarray(u, np.float64)
    return (np.asarray(w, np.float64)[:, None] / float(c) * LEAF_ALPHA *
            np.tanh(u)[None]).astype(np.float32)


def compose(torch, Z0, half, mid, D_parent, D_leaf):
    """-> (arm [8,6] float32 CPU, Du, info ограничителя)."""
    D = torch.as_tensor(np.asarray(D_parent, np.float32) +
                        np.asarray(D_leaf, np.float32))
    Du, lim = ko.limit(torch, D)
    with torch.autocast("cpu", enabled=False):
        Z0t = torch.as_tensor(np.asarray(Z0, np.float32))
        h = torch.as_tensor(np.asarray(half, np.float32))
        m = torch.as_tensor(np.asarray(mid, np.float32))
        arm = m + h * torch.tanh(Z0t[:N8] / h + Du)
    return arm.numpy(), Du.numpy(), lim


def full_reference(torch, E, dec, F1, ref, r, path, eta, c_parent, leaf,
                   u):
    """Независимый эталон: z2 после правки родителя + half·e_leaf к листу,
    повторное смешивание по W2 -> D [8,6] (до ограничителя)."""
    import k31_readout as kr
    a = ko.ALPHA * np.tanh(np.asarray(eta, np.float64))
    e = (a / float(c_parent)).astype(np.float32)
    o = E["edited"](dec, F1, ref, [kr.LEVELS[int(r)]], [path[int(r)]],
                    np.asarray(e)[None])
    with torch.no_grad(), torch.autocast(F1["x0"].device.type,
                                         enabled=False):
        half, _m = E["half_mid"](dec._L, dec._U, F1["x0"].device)
        W2 = F1["W2"][0].float()
        w, cl = leaf_profile(W2.cpu().numpy(), leaf)
        el = torch.as_tensor((LEAF_ALPHA * np.tanh(np.asarray(u, np.float64))
                              / cl).astype(np.float32),
                             device=W2.device)
        z2 = o["z2"][0].clone()
        z2[leaf] = z2[leaf] + half * el
        Zhat = torch.einsum("tl,ltc->tc", W2, z2)
        D = ((Zhat - ref["Z"][0].float()) / half)[:N8]
    return D.cpu().numpy()


def leaf_actor_input(F1, s11n, elapsed, Z0, half, leaf, c_leaf):
    """Сырой вход actor K-32 [4H+67] для узла «лист leaf»."""
    x0 = F1["x0"][0].float().cpu().numpy()
    x1 = F1["x1"][0].float().cpu().numpy()
    x2 = F1["x2"][0].float().cpu().numpy()
    iv = np.asarray(F1["iv2"][0, leaf].float().cpu().numpy(), np.float64)
    W2 = np.asarray(F1["W2"][0].float().cpu().numpy(), np.float64)
    u0 = np.asarray(Z0, np.float32)[:N8] / np.asarray(half, np.float32)
    return np.concatenate([
        x0.mean(0), x1.mean(0), x2.mean(0), x2[leaf],
        np.asarray(s11n, np.float32).ravel(), [float(elapsed) / 600.0],
        u0.ravel(), [0.0, 0.0, 1.0],
        [iv.mean() / 20.0, (iv[1] - iv[0]) / 20.0],
        [W2[:N8, leaf].mean()], [float(c_leaf)]]).astype(np.float32)


def actor_hidden(torch, actor, x):
    """32 признака узлового MLP actor перед actor.out (без градиента)."""
    import k32_model as km32
    with torch.no_grad(), torch.autocast("cpu", enabled=False):
        X = torch.as_tensor(np.asarray(x, np.float32))[None]
        z = torch.clamp((X - actor.mean) / actor.std, -km32.CLIP, km32.CLIP)
        g = torch.nn.functional.gelu
        h = g(actor.l2(g(actor.l1(z))))
    return h[0].numpy().astype(np.float32)


def head_input(h32, eta_parent, Du_parent, w, c, r):
    lev = np.zeros(2, np.float32)
    lev[int(r)] = 1.0                        # r ∈ {0 root, 1 mid}
    return np.concatenate([
        np.asarray(h32, np.float32),
        np.asarray(eta_parent, np.float32) / 2.0,
        (np.asarray(Du_parent, np.float32) / ko.RMS_CAP).ravel(),
        (np.asarray(w, np.float64) / float(c)).astype(np.float32),
        lev]).astype(np.float32)


def make_head(torch):
    lin = torch.nn.Linear(N_IN, N_OUT)
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.zero_()
    return lin


def head_u(torch, head, x96):
    with torch.no_grad(), torch.autocast("cpu", enabled=False):
        return head(torch.as_tensor(np.asarray(x96, np.float32))[None])[0] \
            .numpy().astype(np.float32)


class T2Arm:
    """T32 + головка листа. API k32_policy.Arm (begin/decide/executed)."""

    def __init__(self, torch, base, head):
        self.torch, self.base, self.head = torch, base, head
        self.cap = {}
        fwd0 = base.forward

        def forward(batch, po, state8):
            F, s11n, ref = fwd0(batch, po, state8)
            self.cap = dict(F=F, s11n=s11n, ref=ref)
            return F, s11n, ref
        base.forward = forward
        self.half = base.half.numpy()
        self.mid = base.mid.numpy()

    @property
    def episode(self):
        return self.base.episode

    @episode.setter
    def episode(self, v):
        self.base.episode = v

    @property
    def call(self):
        return self.base.call

    @call.setter
    def call(self, v):
        self.base.call = v

    @property
    def keep_first(self):
        return getattr(self.base, "keep_first", False)

    @keep_first.setter
    def keep_first(self, v):
        self.base.keep_first = v

    def begin(self, task, init, episode=None):
        self.base.begin(task, init, episode)

    def executed(self, k):
        self.base.executed(k)

    def decide(self, batch, po, state8, elapsed):
        torch = self.torch
        a0, rec = self.base.decide(batch, po, state8, elapsed)
        F, s11n = self.cap["F"], self.cap["s11n"]
        r = int(rec.get("r", 2))
        ext = dict(base_a_codec=a0.copy(), k37_reason=REASON["leaf_route"],
                   k37_leaf=-1, k37_w=np.zeros(N8, np.float32),
                   k37_c=np.float32(0), k37_u=np.zeros(6, np.float32),
                   k37_Dl=np.zeros((N8, 6), np.float32),
                   k37_Du2=np.zeros((N8, 6), np.float32))
        if "logits" not in rec or rec.get("zero", True):
            ext["k37_reason"] = REASON["zero_parent"] if "logits" in rec \
                else REASON["leaf_route"]
            rec.update(ext)
            return a0, rec
        W2 = np.asarray(F["W2"][0].float().cpu().numpy(), np.float64)
        leaf, why = choose_leaf(W2, r, rec["path"])
        ext["k37_reason"] = why
        if leaf is None:
            rec.update(ext)
            return a0, rec
        w64, c64 = leaf_profile(W2, leaf)
        w, c = w64.astype(np.float32), np.float32(c64)   # как в журнале
        x = leaf_actor_input(F, s11n[0].float().cpu().numpy(), elapsed,
                             rec["Z0"], self.half, leaf, c)
        h = actor_hidden(torch, self.base.actor, x)
        x96 = head_input(h, rec["eta"], rec["Du"], w, float(c), r)
        u = head_u(torch, self.head, x96)
        ext.update(k37_leaf=int(leaf), k37_w=w, k37_c=c, k37_u=u,
                   k37_x=x96)
        if not u.any():                     # нулевая головка -> T32 побитово
            rec.update(ext)
            return a0, rec
        Dl = leaf_delta(w, float(c), u)
        arm, Du2, _lim = compose(torch, rec["Z0"], self.half, self.mid,
                                 rec["D"], Dl)
        a = np.concatenate([arm, a0[:, 6:7]], 1).astype(np.float32)
        if not np.isfinite(a).all():
            raise SystemExit("K-37: команда не конечна")
        ext.update(k37_Dl=Dl, k37_Du2=Du2)
        rec.update(ext)
        rec["a_codec"] = a
        self.base.last = dict(rec=rec)
        return a, rec


def selftest():
    import torch
    import k21_model as km
    import k25_model as k25m
    import k29_model as k29m
    import k31_readout as kr
    import k32_model as km32
    import k32_policy as kpol
    k29m.no_tf32(torch)
    # выбор листа: ничья — меньший индекс, только потомки, late > early
    W = np.zeros((20, 16))
    W[:4, 5], W[4:8, 5] = 0.1, 0.3
    W[:4, 4], W[4:8, 4] = 0.0, 0.3
    W[:4, 9], W[4:8, 9] = 0.0, 0.9                    # не потомок root 1
    assert choose_leaf(W, 0, (1, 2, 5)) == (4, 0)     # ничья 4/5 -> 4
    assert choose_leaf(W, 1, (1, 2, 5)) == (4, 0)
    assert choose_leaf(W, 2, (1, 2, 5)) == (None, REASON["leaf_route"])
    W2 = np.zeros((20, 16))
    W2[:4, 4] = 0.5
    assert choose_leaf(W2, 1, (1, 2, 4)) == (None, REASON["no_late"])
    # реальная сборка на игрушечном K-25: быстрая формула = эталону
    M = k29m.build(torch)
    E = kr.build(torch)
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
    norm = km32.fit_normalizer(np.random.default_rng(0).normal(
        size=(64, 6 * H + 72)), H)
    actor, _c = km32.build(torch, H, 6, 0, norm)
    with torch.no_grad():
        g2 = torch.Generator().manual_seed(7)
        for p in actor.out.parameters():
            p.copy_(torch.randn(p.shape, generator=g2))
    worst, n_act = 0.0, 0
    for seed in range(12):
        base = kpol.Arm(torch, mod, dec, M, kind="T", seed=0, actor=actor)
        head = make_head(torch)
        arm = T2Arm(torch, base, head)
        bt = km.fake_batch(torch, B=1, seed=seed)
        s8 = np.random.default_rng(seed).normal(size=(1, 8)) * 0.1
        arm.begin(0, 0, 0)
        a_zero, rec0 = arm.decide(bt, 1, s8, 8)
        base2 = kpol.Arm(torch, mod, dec, M, kind="T", seed=0, actor=actor)
        base2.begin(0, 0, 0)
        a_t32, _r = base2.decide(bt, 1, s8, 8)
        assert np.array_equal(a_zero, a_t32), "нулевая головка ≠ T32"
        if rec0["k37_reason"] != 0:
            continue
        n_act += 1
        with torch.no_grad():
            head.bias.copy_(torch.as_tensor(
                np.random.default_rng(seed).normal(size=6) * 2.0))
        arm.begin(0, 0, 0)
        a2, rec = arm.decide(bt, 1, s8, 8)
        F, ref = arm.cap["F"], arm.cap["ref"]
        Dref = full_reference(torch, E, dec, F, ref, rec["r"], rec["path"],
                              rec["eta"], rec["c"][rec["r"]],
                              rec["k37_leaf"], rec["k37_u"])
        Dfast = np.asarray(rec["D"]) + rec["k37_Dl"]
        worst = max(worst, float(np.abs(Dref - Dfast).max()))
        assert not np.array_equal(a2[:, :6], a_t32[:, :6])
        assert np.array_equal(a2[:, 6], a_t32[:, 6])          # схват T32
    assert n_act >= 1, "на игрушке нет активных вызовов"
    assert worst <= 1e-5, worst
    print(f"самопроверка k37_operator пройдена: выбор позднего потомка "
          f"(ничья — меньший, только потомки, leaf/нет позднего — пропуск); "
          f"нулевая головка = T32 побитово; быстрая сумма = полной "
          f"пересборке листьев (|ΔD| ≤ {worst:.1e}, активных {n_act}); схват "
          f"T32")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; запуск — через k37_parent_leaf.py")
