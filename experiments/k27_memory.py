#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-27: память дерева о собственном плане — снимок, сдвиг времени,
перекрытия, сводка собственных команд, сопоставление двух деревьев и
gated read; учёт состояния сред между вызовами.

СНИМОК одного наблюдения (detach): final hidden 4 корней и 8 средних после
слоя 24 (до финальной нормы), интервалы корней [4,2] и средних [8,2], веса
W0 [20,4], W1 [20,8], собственный предсказанный чанк [20,7] (рука после
ограничителя в единицах кодека, схват ±1), state11 в момент предсказания;
идентификаторы среды/эпизода и шаг среды — у учёта сред (MemoryBank).
СДВИГ на фактически исполненные k шагов: l' = max(l − k, 0), r' = r − k,
alive = r' > l'. СВОДКА узла: взвешенное его W среднее СОБСТВЕННЫХ команд
старого чанка по строкам t ≥ k; рука — (a − mid)/half, схват ±1; нулевой
вес — нулевая сводка и запрещённый узел.
ПРИЗНАК старого узла e = [LN(h), сводка 7, state_old 11, state_now 11
(нормировка K-21), (l', r', центр, ширина)/20, уровень one-hot 2]:
Dm = H + 35.
СОПОСТАВЛЕНИЕ (4 головы по 32, d = 128):
  корни   A0 = softmax(S0 + log O0), разрешены alive ∧ O0 > 0;
  средние A1 = softmax(S1 + log O1 + log prior1),
          prior1_jm = 0.9·A0[j//2, m//2]/d_m + 0.1/N_alive_mid;
  пустая строка -> A = 0 и поправка ровно 0 (безопасные logits, без NaN·0).
READ: C = concat_heads(A·V); g = σ(Wg[LN(h), C] + bg);
  h ← h + has_overlap·g·Wo(C). Wo = 0, Wg = 0, bg = log(0.1/0.9).
Маски, интервалы, нормировки, scores, prior, softmax и контекст — float32
вне autocast (явно поданный float64 сохраняется); проекции — под autocast.
"""
import math
import sys

import numpy as np

T = 20
K_EXEC = 8
D_MEM, N_HEADS = 128, 4
HEAD_DIM = D_MEM // N_HEADS
GATE_INIT = 0.1
MIX = 0.1
LN_EPS = 1e-5
EXTRA = 7 + 11 + 11 + 4 + 2           # 35
VERSION = "mem1"
CONFIG = dict(version=VERSION, d=D_MEM, heads=N_HEADS, head_dim=HEAD_DIM,
              gate_init=GATE_INIT, prior_mix=MIX, ln_eps=LN_EPS, extra=EXTRA,
              k_exec=K_EXEC, read_points=["after layer 8: roots",
                                          "after layer 16: mids"],
              shift="l'=max(l-k,0), r'=r-k, alive=r'>l'",
              root="softmax(S0 + log O0)",
              mid="softmax(S1 + log O1 + log(0.9 A0[j//2,m//2]/d_m + "
                  "0.1/N_alive))",
              gate="sigmoid(Wg[LN(h),C]+bg), Wg=0, bg=log(1/9); Wo=0")
SNAP_KEYS = ("h_root", "h_mid", "iv_root", "iv_mid", "W0", "W1", "chunk",
             "state11")


def n_new_params(H, d=D_MEM):
    Dm = H + EXTRA
    return 2 * d * (H + Dm) + (H + d + 1)


def build(torch):
    nn = torch.nn
    F = nn.functional

    def wdt(*xs):
        dt = torch.float32
        for x in xs:
            dt = torch.promote_types(dt, x.dtype)
        return dt

    # --- время и интервалы ------------------------------------------------
    def shift(iv, k):
        """iv [B, n, 2] (время старого вызова), k [B] исполненных шагов ->
        (iv_rem [B, n, 2], alive [B, n])."""
        dt = wdt(iv)
        iv = iv.to(dt)
        k = torch.as_tensor(k, dtype=dt, device=iv.device).reshape(-1, 1)
        l_ = torch.clamp(iv[..., 0] - k, min=0.0)
        r_ = iv[..., 1] - k
        alive = r_ > l_
        return torch.stack([l_, torch.maximum(r_, l_)], -1), alive

    def overlap(iv_now, iv_old):
        """[B, N, 2], [B, M, 2] -> длина пересечения [B, N, M]."""
        dt = wdt(iv_now, iv_old)
        a, b = iv_now.to(dt), iv_old.to(dt)
        lo = torch.maximum(a[:, :, None, 0], b[:, None, :, 0])
        hi = torch.minimum(a[:, :, None, 1], b[:, None, :, 1])
        return torch.clamp(hi - lo, min=0.0)

    def command_summary(chunk, W, k, L, U):
        """chunk [B, 20, 7] собственных команд, W [B, 20, n] -> сводка
        [B, n, 7] по строкам t ≥ k; вес [B, n]. Весь расчёт вне autocast
        (float32, явно поданный float64 сохраняется)."""
        with torch.autocast(chunk.device.type, enabled=False):
            return _summary(chunk, W, k, L, U)

    def _summary(chunk, W, k, L, U):
        dt = wdt(chunk, W)
        x = chunk.to(dt)
        L = torch.as_tensor(np.asarray(L), dtype=dt, device=x.device)
        U = torch.as_tensor(np.asarray(U), dtype=dt, device=x.device)
        mid, half = (U + L) / 2, (U - L) / 2
        xn = torch.cat([(x[..., :6] - mid) / half, x[..., 6:7]], -1)
        k = torch.as_tensor(k, device=x.device).reshape(-1, 1)
        keep = (torch.arange(T, device=x.device)[None] >= k).to(dt)
        w = W.to(dt) * keep[..., None]                     # [B, 20, n]
        sw = w.sum(1)                                      # [B, n]
        num = torch.einsum("btn,btc->bnc", w, xn)
        safe = torch.where(sw > 0, sw, torch.ones_like(sw))
        a = torch.where((sw > 0)[..., None], num / safe[..., None],
                        torch.zeros_like(num))
        return a, sw

    def interval_feat(iv):
        l_, r_ = iv[..., 0], iv[..., 1]
        return torch.stack([l_, r_, (l_ + r_) / 2, r_ - l_], -1) / T

    # --- сопоставление -----------------------------------------------------
    def masked_softmax(logits, allowed):
        """logits [B, h, N, M], allowed [B, 1|h, N, M] -> A; пустая строка
        -> строго 0 (без softmax запрещённой строки)."""
        any_ = allowed.any(-1, keepdim=True)
        x = torch.where(allowed, logits, torch.full((), float("-inf"),
                                                    dtype=logits.dtype,
                                                    device=logits.device))
        x = torch.where(any_, x, torch.zeros((), dtype=logits.dtype,
                                             device=logits.device))
        A = torch.softmax(x, -1)
        return torch.where(any_ & allowed, A, torch.zeros_like(A))

    def safe_log(x, allowed):
        return torch.where(allowed, torch.log(torch.where(
            allowed, x, torch.ones_like(x))), torch.zeros_like(x))

    def root_match(S0, O0, alive_root):
        """S0 [B, h, 4, 4], O0 [B, 4, 4], alive [B, 4] -> A0, allowed."""
        with torch.autocast(S0.device.type, enabled=False):
            return _root_match(S0, O0, alive_root)

    def _root_match(S0, O0, alive_root):
        dt = wdt(S0, O0)
        allowed = (alive_root[:, None, :] & (O0 > 0))[:, None]
        A0 = masked_softmax(S0.to(dt) + safe_log(O0.to(dt), allowed[:, 0])
                            [:, None], allowed)
        return A0, allowed

    def mid_prior(A0, alive_mid):
        """A0 [B, h, 4, 4] -> prior1 [B, h, 8, 8] (нули у мёртвых)."""
        dt = wdt(A0)
        am = alive_mid.to(dt)                              # [B, 8]
        d = am.view(-1, 4, 2).sum(-1)                      # живых детей корня
        d_m = d[:, torch.arange(8, device=A0.device) // 2]  # [B, 8]
        j2 = torch.arange(8, device=A0.device) // 2
        par = A0.to(dt)[:, :, j2][:, :, :, j2]             # [B, h, 8, 8]
        pp = torch.where(d_m[:, None, None] > 0, par / d_m.clamp_min(1)
                         [:, None, None], torch.zeros_like(par))
        n_alive = am.sum(-1).clamp_min(1)[:, None, None, None]
        u = am[:, None, None] / n_alive
        pr = (1 - MIX) * pp + MIX * u
        return torch.where(alive_mid[:, None, None].bool(), pr,
                           torch.zeros_like(pr))

    def mid_match(S1, O1, alive_mid, A0):
        with torch.autocast(S1.device.type, enabled=False):
            return _mid_match(S1, O1, alive_mid, A0)

    def _mid_match(S1, O1, alive_mid, A0):
        dt = wdt(S1, O1, A0)
        prior = mid_prior(A0, alive_mid)
        base = alive_mid[:, None, :] & (O1 > 0)            # [B, 8, 8]
        allowed = base[:, None] & (prior > 0)
        logits = S1.to(dt) + safe_log(O1.to(dt), base)[:, None] + \
            safe_log(prior, allowed)
        return masked_softmax(logits, allowed), allowed, prior

    def context(A, V):
        dt = wdt(A, V)
        C = torch.matmul(A.to(dt), V.to(dt))
        B, h, N, dh = C.shape
        return C.transpose(1, 2).reshape(B, N, h * dh)

    def uniform_(t, bound, gen):
        with torch.no_grad():
            t.copy_((torch.rand(t.shape, generator=gen) * 2 - 1) * bound)

    class MemoryModule(nn.Module):
        def __init__(self, H, gen):
            super().__init__()
            self.H, self.Dm = H, H + EXTRA
            self.h, self.dh = N_HEADS, HEAD_DIM
            self.Wq = nn.Linear(H, D_MEM, bias=False)
            self.Wk = nn.Linear(self.Dm, D_MEM, bias=False)
            self.Wv = nn.Linear(self.Dm, D_MEM, bias=False)
            self.Wo = nn.Linear(D_MEM, H, bias=False)
            self.gate = nn.Linear(H + D_MEM, 1)
            for m in (self.Wq, self.Wk, self.Wv):
                uniform_(m.weight, 1.0 / math.sqrt(m.in_features), gen)
            with torch.no_grad():
                self.Wo.weight.zero_()
                self.gate.weight.zero_()
                self.gate.bias.fill_(math.log(GATE_INIT / (1 - GATE_INIT)))

        def _heads(self, x):
            B, n, _ = x.shape
            return x.view(B, n, self.h, self.dh).transpose(1, 2)

        def project(self, e):
            return self._heads(self.Wk(e)), self._heads(self.Wv(e))

        def scores(self, h_now, K):
            dev = h_now.device.type
            with torch.autocast(dev, enabled=False):
                hn = F.layer_norm(h_now.float(), (self.H,), eps=LN_EPS)
            Q = self._heads(self.Wq(hn))
            with torch.autocast(dev, enabled=False):
                S = torch.matmul(Q.float(), K.float().transpose(-1, -2)) / \
                    math.sqrt(self.dh)
            return S, hn

        def update(self, h_now, hn, A, allowed, V):
            dev = h_now.device.type
            with torch.autocast(dev, enabled=False):
                C = context(A, V)
            has = allowed.any(-1).any(1)                    # [B, N]
            g = torch.sigmoid(self.gate(torch.cat([hn, C], -1)))  # [B, N, 1]
            upd = g * self.Wo(C)
            upd = torch.where(has[..., None], upd.to(h_now.dtype),
                              torch.zeros((), dtype=h_now.dtype,
                                          device=h_now.device))
            return h_now + upd, g[..., 0], has

    return dict(shift=shift, overlap=overlap, command_summary=command_summary,
                interval_feat=interval_feat, masked_softmax=masked_softmax,
                root_match=root_match, mid_prior=mid_prior,
                mid_match=mid_match, context=context,
                MemoryModule=MemoryModule)


def node_features(torch, M, h, a, st_old, st_now, iv_rem, level):
    """[LN(h), сводка, state_old, state_now, интервал, уровень] -> [B, n, Dm]
    (float32, вне autocast)."""
    F = torch.nn.functional
    B, n, H = h.shape
    with torch.autocast(h.device.type, enabled=False):
        hn = F.layer_norm(h.float(), (H,), eps=LN_EPS)
        lv = torch.zeros(B, n, 2, device=h.device)
        lv[..., level] = 1.0
        return torch.cat([hn, a.float(), st_old.float()[:, None].expand(
            B, n, -1), st_now.float()[:, None].expand(B, n, -1),
            M["interval_feat"](iv_rem.float()), lv], -1)


# --- состояние сред между вызовами --------------------------------------------
class MemoryBank:
    """Снимки по env_id. Вызов: env_ids, episode_ids, steps (шаг среды в
    момент наблюдения) -> какие среды получают память и k. Пустая память:
    первый вызов, reset, новый эпизод (в т. ч. с тем же env_id), k ≥ 20.
    Отрицательный k (часы назад) — ошибка; повтор env_id в батче — ошибка."""

    def __init__(self):
        self.mem = {}

    def reset(self, env_id=None):
        if env_id is None:
            self.mem.clear()
        else:
            self.mem.pop(int(env_id), None)

    @staticmethod
    def _check(env_ids, *cols):
        env_ids = [int(x) for x in env_ids]
        n = [len(env_ids)] + [len(c) for c in cols]
        if len(set(n)) != 1:
            raise SystemExit(f"память: разная длина входов {n}")
        if len(set(env_ids)) != len(env_ids):
            raise SystemExit(f"память: повтор env_id в батче {env_ids}")
        return env_ids

    def select(self, env_ids, episode_ids, steps):
        env_ids = self._check(env_ids, list(episode_ids), list(steps))
        use, ks, snaps = [], [], []
        for e, ep, s in zip(env_ids, episode_ids, steps):
            m = self.mem.get(e)
            if m is None or m["episode"] != int(ep):
                use.append(False)
                ks.append(0)
                snaps.append(None)
                continue
            k = int(s) - m["step"]
            if k < 0:
                raise SystemExit(f"память: среда {e} — шаг {s} раньше "
                                 f"снимка {m['step']}")
            if m["env"] != e:
                raise SystemExit("память: снимок другой среды")
            ok = k < T
            use.append(ok)
            ks.append(k if ok else 0)
            snaps.append(m["snap"] if ok else None)
        return use, ks, snaps

    def update(self, env_ids, episode_ids, steps, snap_rows):
        env_ids = self._check(env_ids, list(episode_ids), list(steps),
                              list(snap_rows))       # до изменения банка
        for e, ep, s, sn in zip(env_ids, episode_ids, steps, snap_rows):
            self.mem[int(e)] = dict(env=int(e), episode=int(ep), step=int(s),
                                    snap=sn)

    def nbytes(self):
        return sum(sum(v.element_size() * v.numel() for v in m["snap"].values())
                   for m in self.mem.values())


def split_snapshot(snap):
    """Батч-снимок {k: [B, ...]} -> список по средам (detach, clone)."""
    B = snap["h_root"].shape[0]
    return [{k: v[b].detach().clone() for k, v in snap.items()}
            for b in range(B)]


def stack_snapshot(torch, rows, like):
    """Список снимков (None — пусто) -> батч {k: [B, ...]} и маска; пустые
    строки — нули (маска запрещает их узлы)."""
    out, mask = {}, []
    for k in SNAP_KEYS:
        ref = like[k][0]                    # like — батч-снимок [B, ...]
        out[k] = torch.stack([(r[k].to(ref.device, ref.dtype) if r is not None
                               else torch.zeros_like(ref)) for r in rows])
    mask = torch.tensor([r is not None for r in rows], device=like["h_root"]
                        .device)
    return out, mask


def selftest():
    import torch
    M = build(torch)
    g = torch.Generator().manual_seed(0)
    # 1. сдвиг на 0 / 8 / 20; частично истёкший узел; остаток ≤ 12 при k = 8
    iv = torch.tensor([[[0., 5.], [5., 9.], [9., 14.], [14., 20.]]])
    for k, alive in ((0, [1, 1, 1, 1]), (8, [0, 1, 1, 1]),
                     (20, [0, 0, 0, 0])):
        r, a = M["shift"](iv, torch.tensor([k]))
        assert a[0].tolist() == [bool(x) for x in alive], (k, a)
        if k == 8:
            assert r[0, 1].tolist() == [0.0, 1.0] and \
                r[0, 3].tolist() == [6.0, 12.0]
            w = (r[..., 1] - r[..., 0]) * a
            assert abs(float(w.sum()) - 12.0) < 1e-6
    O = M["overlap"](torch.tensor([[[0., 4.], [4., 20.]]]),
                     torch.tensor([[[0., 1.], [6., 12.], [12., 12.]]]))
    assert O[0].tolist() == [[1., 0., 0.], [0., 6., 0.]]
    # 4. сводка — только собственный старый чанк, строки t ≥ k, веса узла
    L, U = np.full(6, -2.0), np.full(6, 2.0)
    ch = torch.zeros(1, 20, 7)
    ch[0, :8, :6] = 2.0                      # исполненная часть — не читается
    ch[0, 8:, :6] = 1.0
    ch[0, :, 6] = torch.where(torch.arange(20) < 12, -1.0, 1.0)
    W = torch.zeros(1, 20, 2)
    W[0, :10, 0] = 1.0
    W[0, 10:, 1] = 1.0
    a, sw = M["command_summary"](ch, W, torch.tensor([8]), L, U)
    assert torch.allclose(a[0, 0, :6], torch.full((6,), 0.5))
    assert float(sw[0, 0]) == 2.0 and float(sw[0, 1]) == 10.0
    assert abs(float(a[0, 1, 6]) - 0.6) < 1e-6
    # под bf16-autocast сводка совпадает с float32 по значениям (дробные
    # веса и команды)
    chf = torch.rand(2, 20, 7, generator=g) * 1.7 - 0.85
    Wf = torch.rand(2, 20, 4, generator=g)
    kf = torch.tensor([8, 3])
    ref, swr = M["command_summary"](chf, Wf, kf, L, U)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got, swg = M["command_summary"](chf, Wf, kf, L, U)
    assert got.dtype == torch.float32 and torch.equal(got, ref) and \
        torch.equal(swg, swr)
    a2, sw2 = M["command_summary"](ch, W, torch.tensor([12]), L, U)
    assert float(sw2[0, 0]) == 0 and float(a2[0, 0].abs().sum()) == 0
    # 3. A ≥ 0, сумма строки 1 при допустимых ключах и 0 при пустой; нули
    B, h = 3, 4
    S0 = torch.randn(B, h, 4, 4, generator=g) * 3
    ivn = M["shift"](torch.tensor([[[0., 5.], [5., 10.], [10., 15.],
                                    [15., 20.]]]).expand(B, -1, -1),
                     torch.zeros(B))[0]
    ivo = torch.tensor([[[0., 3.], [3., 8.], [8., 14.], [14., 20.]]]) \
        .expand(B, -1, -1)
    ivr, alv = M["shift"](ivo, torch.tensor([8, 8, 20]))
    O0 = M["overlap"](ivn, ivr)
    A0, al0 = M["root_match"](S0, O0, alv)
    assert torch.isfinite(A0).all() and (A0 >= 0).all()
    rs = A0.sum(-1)
    row_ok = al0.any(-1).expand_as(rs)
    assert torch.allclose(rs[row_ok], torch.ones_like(rs[row_ok]))
    assert float(rs[~row_ok].abs().max()) == 0.0 and not bool(row_ok[2].any())
    assert bool(row_ok[0].any())
    assert float(A0[~al0.expand_as(A0)].abs().max()) == 0.0
    # 2. prior средних: индексы родителей, разное число живых детей
    A0x = torch.zeros(1, 1, 4, 4)
    A0x[0, 0, 0, 1] = 1.0                    # текущий корень 0 <- старый 1
    A0x[0, 0, 1, 3] = 1.0
    alive_mid = torch.tensor([[1, 1, 1, 0, 1, 1, 1, 1]], dtype=torch.bool)
    pr = M["mid_prior"](A0x, alive_mid)[0, 0]
    # текущие средние 0,1 (родитель 0) -> старые 2 (жив; 3 мёртв: d = 1)
    assert abs(float(pr[0, 2]) - (0.9 + 0.1 / 7)) < 1e-6 and \
        float(pr[0, 3]) == 0.0
    assert abs(float(pr[2, 6]) - (0.45 + 0.1 / 7)) < 1e-6   # родитель 1 -> 3
    assert torch.allclose(pr.sum(-1)[:4], torch.ones(4))
    # 5. функциональная зависимость средних от соответствия корней при
    # фиксированных собственных scores; gradcheck float64
    S1 = torch.randn(1, 1, 8, 8, generator=g)
    O1 = torch.ones(1, 8, 8)
    A0y = A0x.clone()
    A0y[0, 0, 0] = torch.tensor([0., 0., 0., 1.])
    m1 = M["mid_match"](S1, O1, alive_mid, A0x)[0]
    m2 = M["mid_match"](S1, O1, alive_mid, A0y)[0]
    assert float((m1[0, 0, :2] - m2[0, 0, :2]).abs().max()) > 1e-3
    assert torch.allclose(m1[0, 0, 2:], m2[0, 0, 2:])
    S0d = torch.randn(2, 2, 4, 4, dtype=torch.float64, generator=g,
                      requires_grad=True)
    S1d = torch.randn(2, 2, 8, 8, dtype=torch.float64, generator=g,
                      requires_grad=True)
    Vd = torch.randn(2, 2, 8, 3, dtype=torch.float64, generator=g,
                     requires_grad=True)
    ivn2 = torch.tensor([[[0., 5.], [5., 10.], [10., 15.], [15., 20.]]],
                        dtype=torch.float64).expand(2, -1, -1)
    ivo2, alv2 = M["shift"](torch.tensor([[[0., 9.], [9., 12.], [12., 16.],
                                           [16., 20.]]], dtype=torch.float64)
                            .expand(2, -1, -1), torch.tensor([8, 8]))
    O0d = M["overlap"](ivn2, ivo2)
    mids_n = torch.tensor([[[0., 2.], [2., 5.], [5., 7.], [7., 10.], [10., 12.],
                            [12., 15.], [15., 17.], [17., 20.]]],
                          dtype=torch.float64).expand(2, -1, -1)
    mids_o, alm = M["shift"](torch.tensor(
        [[[0., 4.], [4., 9.], [9., 10.], [10., 12.], [12., 14.], [14., 16.],
          [16., 18.], [18., 20.]]], dtype=torch.float64).expand(2, -1, -1),
        torch.tensor([8, 8]))
    O1d = M["overlap"](mids_n, mids_o)

    def f(s0, s1, v):
        A0_, _a = M["root_match"](s0, O0d, alv2)
        A1_, _b, _p = M["mid_match"](s1, O1d, alm, A0_)
        assert A1_.dtype == torch.float64
        return M["context"](A1_, v)
    assert torch.autograd.gradcheck(f, (S0d, S1d, Vd), eps=1e-6, atol=1e-5)
    # нет detach внутри forward: контекст средних даёт градиент scores корней
    s0 = S0d.detach().clone().requires_grad_(True)
    f(s0, S1d.detach(), Vd.detach()).pow(2).sum().backward()
    assert float(s0.grad.abs().sum()) > 0
    # модуль: нулевые Wo/Wg -> поправка 0; пустая строка -> 0 при любом Wo
    H = 16
    mm = M["MemoryModule"](H, torch.Generator().manual_seed(1))
    assert sum(p.numel() for p in mm.parameters()) == n_new_params(H)
    assert n_new_params(768) == 403073
    hn = torch.randn(B, 4, H, generator=g)
    e = torch.randn(B, 4, H + EXTRA, generator=g)
    K, V = mm.project(e)
    S, hl = mm.scores(hn, K)
    A, al = M["root_match"](S, O0, alv)
    out, gt, has = mm.update(hn, hl, A, al, V)
    assert torch.equal(out, hn) and torch.allclose(gt, torch.full_like(gt, .1))
    with torch.no_grad():
        mm.Wo.weight.normal_(0, 1, generator=g)
    out, gt, has = mm.update(hn, hl, A, al, V)
    assert torch.equal(out[2], hn[2]) and not has[2].any()
    assert not torch.allclose(out[0], hn[0])
    # MemoryBank: перестановка, независимые reset, новый эпизод с тем же
    # env_id, k ≥ 20, часы назад
    bank = MemoryBank()
    snap = {k: torch.zeros(1) + i for i, k in enumerate(SNAP_KEYS)}
    bank.update([0, 1, 2], [10, 11, 12], [8, 8, 8], [snap] * 3)
    use, ks, _s = bank.select([2, 0, 1], [12, 10, 99], [16, 16, 16])
    assert use == [True, True, False] and ks == [8, 8, 0]
    bank.reset(0)
    use, ks, _s = bank.select([0, 2], [10, 12], [16, 40])
    assert use == [False, False]
    try:
        bank.select([1], [11], [4])
        raise AssertionError("часы назад приняты")
    except SystemExit:
        pass
    try:
        bank.select([1, 1], [11, 11], [16, 16])
        raise AssertionError("повтор env_id принят")
    except SystemExit:
        pass
    # разная длина входов и повтор env_id в update — отказ, банк не меняется
    before = {e: dict(m) for e, m in bank.mem.items()}
    for args in (([5, 6], [1], [0, 0], [snap, snap]),
                 ([5, 6], [1, 2], [0, 0], [snap]),
                 ([5, 5], [1, 2], [0, 0], [snap, snap])):
        try:
            bank.update(*args)
            raise AssertionError(f"update принял {args[:3]}")
        except SystemExit:
            pass
        assert bank.mem == before
    for args in (([1, 2], [11], [16, 16]), ([1, 2], [11, 12], [16])):
        try:
            bank.select(*args)
            raise AssertionError(f"select принял {args}")
        except SystemExit:
            pass
    print("самопроверка k27_memory пройдена: сдвиг 0/8/20, частичное "
          "истечение, остаток 12 при k = 8, перекрытия, сводка только по "
          "собственным строкам t ≥ k, A ≥ 0 / суммы 1 / пустая строка 0 / "
          "запрещённые 0, prior средних (индексы, разное число живых "
          "детей), зависимость средних от соответствия корней, gradcheck "
          "float64, градиент к scores корней, нулевые Wo/Wg -> 0, пустая "
          "строка -> 0 при любом Wo, 403073 параметра при H = 768, "
          "MemoryBank (перестановка, reset, новый эпизод, k ≥ 20, часы "
          "назад, повтор env_id, разная длина входов — банк не меняется); "
          "сводка под bf16-autocast = float32 по значениям")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
