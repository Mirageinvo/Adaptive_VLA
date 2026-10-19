#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-20: иерархический (coarse + fine) и плоский токенизаторы чанка 20 × 7.

ВРЕМЕННАЯ ИЕРАРХИЯ (рука, 6 каналов, координаты кодека): P — средние по
пяти блокам из 4 команд, повторённые внутри блока; C* = P(A), R* = (I − P)A,
A = C* + R* точно. Схват не усредняется и принадлежит нижнему уровню.

  coarse: энкодер(C*, схват = 0) -> 8 латентов -> VQ(2048 × 128) ->
          декодер -> Y_c; C = mid(L, U) + half(L, U) · P(tanh(Y_c));
  fine:   энкодер(R*, схват) -> 8 латентов -> VQ -> декодер, который видит
          и fine-, и coarse-коды -> Y_f [20 × 6] и логиты схвата [20];
          R = (I − P) Y_f.
  СБОРКА: в каждом блоке один β ∈ [0, 1] (максимальный, при котором все
  C + βR блока в [L, U]; при R = 0 β = 1) -> A = C + βR. Средние блоков
  сохраняются точно (R имеет нулевые средние, β общий на блок).
  Схват исполнения — знак логита (±1 в координатах кодека).
ПЛОСКИЙ: энкодер(A) -> 16 латентов -> VQ -> декодер -> рука
  mid + half · tanh(Y), логиты схвата; энкодер и декодер по 4 блока.
Perceiver-подобные блоки ширины 256 (cross-attention, self-attention, MLP).

ОБУЧЕНИЕ: batch 256, AdamW 3e−4, warmup 500, косинус до 3e−5, clip 1.0,
до 20 000 обновлений (один заранее заданный добор до 30 000, если лучший шаг
последний и ошибка ещё падает); каждый 4-й батч — цели учителя (полный BAR,
train_fit). Потеря: нормированная MSE руки (20 шагов) + MSE coarse-части
P(Â) против P(A*) + BCE схвата + VQ (codebook + 0.25 commitment). Словари:
инициализация латентами первого батча, перезапуск неиспользуемых кодов
каждые 500 обновлений до 75% бюджета. Выбор шага — по val_sel: MSE руки
(норм.) + BCE схвата на истинных кодах (hard). Нормировки — только train.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

T, NB, BL = 20, 5, 4
N_ARM = 6
VOCAB, CODE_DIM, WIDTH = 2048, 128, 256
STEPS, STEPS_MAX, BATCH = 20000, 30000, 256
LR0, LR1, WARM = 3e-4, 3e-5, 500
EVAL_EVERY = 500


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


# --- оператор P и сборка (numpy и torch) ----------------------------------------
def P_np(A):
    A = np.asarray(A, np.float64)
    m = A.reshape(A.shape[:-2] + (NB, BL, A.shape[-1])).mean(-2,
                                                            keepdims=True)
    return np.broadcast_to(m, A.shape[:-2] + (NB, BL, A.shape[-1])).reshape(
        A.shape)


def beta_np(C, R, L, U):
    C, R = np.asarray(C, np.float64), np.asarray(R, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(R > 0, (np.asarray(U) - C) / R, np.inf)
        dn = np.where(R < 0, (np.asarray(L) - C) / R, np.inf)
    b = np.minimum(up, dn)
    b = b.reshape(b.shape[:-2] + (NB, BL * N_ARM)).min(-1)
    return np.clip(b, 0.0, 1.0)


def build(torch):
    nn = torch.nn
    F = nn.functional

    def P(A):
        s = A.shape
        m = A.reshape(s[:-2] + (NB, BL, s[-1])).mean(-2, keepdim=True)
        return m.expand(s[:-2] + (NB, BL, s[-1])).reshape(s)

    def beta(C, R, L, U):
        big = torch.full_like(R, float("inf"))
        safe = torch.where(R == 0, torch.ones_like(R), R)
        up = torch.where(R > 0, (U - C) / safe, big)
        dn = torch.where(R < 0, (L - C) / safe, big)
        b = torch.minimum(up, dn)
        b = b.reshape(b.shape[:-2] + (NB, BL * N_ARM)).min(-1).values
        return torch.clamp(b, 0.0, 1.0)

    class Block(nn.Module):
        def __init__(self, w=WIDTH, heads=4):
            super().__init__()
            self.ln_q, self.ln_kv = nn.LayerNorm(w), nn.LayerNorm(w)
            self.ca = nn.MultiheadAttention(w, heads, batch_first=True)
            self.ln_s = nn.LayerNorm(w)
            self.sa = nn.MultiheadAttention(w, heads, batch_first=True)
            self.ln_m = nn.LayerNorm(w)
            self.mlp = nn.Sequential(nn.Linear(w, 4 * w), nn.GELU(),
                                     nn.Linear(4 * w, w))

        def forward(self, q, kv):
            kvn = self.ln_kv(kv)
            q = q + self.ca(self.ln_q(q), kvn, kvn, need_weights=False)[0]
            s = self.ln_s(q)
            q = q + self.sa(s, s, s, need_weights=False)[0]
            return q + self.mlp(self.ln_m(q))

    class Enc(nn.Module):
        def __init__(self, ch, n_lat, blocks):
            super().__init__()
            self.inp = nn.Linear(ch, WIDTH)
            self.pos = nn.Parameter(torch.randn(1, T, WIDTH) * .02)
            self.lat = nn.Parameter(torch.randn(1, n_lat, WIDTH) * .02)
            self.blocks = nn.ModuleList(Block() for _ in range(blocks))
            self.out = nn.Linear(WIDTH, CODE_DIM)

        def forward(self, x):
            h = self.inp(x) + self.pos
            q = self.lat.expand(len(x), -1, -1)
            for b in self.blocks:
                q = b(q, h)
            return self.out(q)

    class Dec(nn.Module):
        def __init__(self, n_mem, out_ch, blocks):
            super().__init__()
            self.inp = nn.Linear(CODE_DIM, WIDTH)
            self.mpos = nn.Parameter(torch.randn(1, n_mem, WIDTH) * .02)
            self.q = nn.Parameter(torch.randn(1, T, WIDTH) * .02)
            self.blocks = nn.ModuleList(Block() for _ in range(blocks))
            self.out = nn.Linear(WIDTH, out_ch)

        def forward(self, mem):
            m = self.inp(mem) + self.mpos
            q = self.q.expand(len(mem), -1, -1)
            for b in self.blocks:
                q = b(q, m)
            return self.out(q)

    class VQ(nn.Module):
        def __init__(self, n=VOCAB, d=CODE_DIM):
            super().__init__()
            self.code = nn.Parameter(torch.randn(n, d) * .1)
            self.register_buffer("used", torch.zeros(n))

        def nearest(self, z):
            d = (z.pow(2).sum(-1, keepdim=True) - 2 * z @ self.code.t()
                 + self.code.pow(2).sum(-1))
            return d.argmin(-1)

        def embed(self, idx):
            return F.embedding(idx, self.code)

        def forward(self, z):
            idx = self.nearest(z)
            zq = self.embed(idx)
            loss = F.mse_loss(zq, z.detach()) + .25 * F.mse_loss(
                z, zq.detach())
            if self.training:
                with torch.no_grad():
                    self.used.index_add_(0, idx.reshape(-1),
                                         torch.ones(idx.numel(),
                                                    device=idx.device))
            return z + (zq - z).detach(), idx, loss

        @torch.no_grad()
        def restart(self, z_pool, init=False):
            """Неиспользуемые с прошлого раза коды <- случайные латенты."""
            z = z_pool.reshape(-1, z_pool.shape[-1])
            dead = torch.ones_like(self.used, dtype=torch.bool) if init \
                else self.used == 0
            k = int(dead.sum())
            if k:
                pick = torch.randint(0, len(z), (k,), device=z.device)
                self.code.data[dead] = z[pick] + 1e-3 * torch.randn_like(
                    z[pick])
            self.used.zero_()
            return k

    class HierCodec(nn.Module):
        kind = "hier"

        def __init__(self, L, U):
            super().__init__()
            self.register_buffer("L", torch.as_tensor(np.asarray(L,
                                                                 np.float32)))
            self.register_buffer("U", torch.as_tensor(np.asarray(U,
                                                                 np.float32)))
            self.enc_c, self.enc_f = Enc(7, 8, 2), Enc(7, 8, 2)
            self.vq_c, self.vq_f = VQ(), VQ()
            self.dec_c = Dec(8, N_ARM, 2)
            self.dec_f = Dec(16, N_ARM + 1, 2)
            self.lvl = nn.Parameter(torch.randn(1, 2, 1, CODE_DIM) * .02)

        def split(self, A):
            arm = A[..., :N_ARM]
            C = P(arm)
            xc = torch.cat([C, torch.zeros_like(A[..., 6:7])], -1)
            xf = torch.cat([arm - C, A[..., 6:7]], -1)
            return xc, xf

        def encode_latent(self, A):
            xc, xf = self.split(A)
            return self.enc_c(xc), self.enc_f(xf)

        def encode(self, A):
            zc, zf = self.encode_latent(A)
            return self.vq_c.nearest(zc), self.vq_f.nearest(zf)

        def decode_emb(self, ec, ef):
            mid, half = (self.U + self.L) / 2, (self.U - self.L) / 2
            Yc = self.dec_c(ec)
            C = mid + half * P(torch.tanh(Yc))
            o = self.dec_f(torch.cat([ef + self.lvl[:, 1], ec + self.lvl[:,
                                                                   0]], 1))
            Yf, gl = o[..., :N_ARM], o[..., N_ARM]
            R = Yf - P(Yf)
            b = beta(C, R, self.L, self.U)
            bb = b.repeat_interleave(BL, -1)[..., None]
            arm = C + bb * R
            return dict(C=C, R=R, beta=b, arm=arm, grip_logit=gl)

        def decode(self, kc, kf):
            return self.decode_emb(self.vq_c.embed(kc), self.vq_f.embed(kf))

        def forward(self, A):
            zc, zf = self.encode_latent(A)
            qc, kc, lc = self.vq_c(zc)
            qf, kf, lf = self.vq_f(zf)
            o = self.decode_emb(qc, qf)
            o.update(kc=kc, kf=kf, vq=lc + lf, vq_c=lc, vq_f=lf, zc=zc, zf=zf)
            return o

        def vqs(self):
            return dict(c=self.vq_c, f=self.vq_f)

    class FlatCodec(nn.Module):
        kind = "flat"

        def __init__(self, L, U):
            super().__init__()
            self.register_buffer("L", torch.as_tensor(np.asarray(L,
                                                                 np.float32)))
            self.register_buffer("U", torch.as_tensor(np.asarray(U,
                                                                 np.float32)))
            self.enc = Enc(7, 16, 4)
            self.vq = VQ()
            self.dec = Dec(16, N_ARM + 1, 4)

        def encode(self, A):
            return self.vq.nearest(self.enc(A))

        def decode_emb(self, e):
            mid, half = (self.U + self.L) / 2, (self.U - self.L) / 2
            o = self.dec(e)
            arm = mid + half * torch.tanh(o[..., :N_ARM])
            return dict(arm=arm, grip_logit=o[..., N_ARM])

        def decode(self, k):
            return self.decode_emb(self.vq.embed(k))

        def forward(self, A):
            z = self.enc(A)
            q, k, lq = self.vq(z)
            o = self.decode_emb(q)
            o.update(k=k, vq=lq, z=z)
            return o

        def vqs(self):
            return dict(flat=self.vq)

    return dict(P=P, beta=beta, HierCodec=HierCodec, FlatCodec=FlatCodec)


def action_out(o):
    """Исполняемый чанк в координатах кодека: рука + схват ±1."""
    import torch
    g = torch.where(o["grip_logit"] > 0, 1.0, -1.0)
    return torch.cat([o["arm"], g[..., None]], -1)


def losses(torch, o, A, std, Pf):
    F = torch.nn.functional
    arm, tgt = o["arm"], A[..., :N_ARM]
    l_arm = (((arm - tgt) / std) ** 2).mean()
    l_c = (((Pf(arm) - Pf(tgt)) / std) ** 2).mean()
    l_g = F.binary_cross_entropy_with_logits(o["grip_logit"],
                                             (A[..., 6] > 0).float())
    return dict(arm=l_arm, coarse=l_c, grip=l_g, vq=o["vq"],
                total=l_arm + l_c + l_g + o["vq"])


# --- веса строк ---------------------------------------------------------------------
def row_weights(task, episode):
    """Равные задачи -> равные эпизоды внутри задачи -> равные строки
    внутри эпизода; нормировано к среднему 1."""
    task, episode = np.asarray(task), np.asarray(episode)
    w = np.zeros(len(task))
    for t in np.unique(task):
        mt = task == t
        eps = np.unique(episode[mt])
        for e in eps:
            me = mt & (episode == e)
            w[me] = 1.0 / (len(eps) * me.sum())
    return w * len(w) / max(w.sum(), 1e-300)


# --- оценка ------------------------------------------------------------------------
def first_switch(g):
    s = np.sign(g)
    d = np.nonzero(s[1:] != s[:-1])[0]
    return int(d[0]) + 1 if len(d) else -1


def chunk_metrics(act, gl, A, std, w):
    """Метрики чанков: взвешенные (основные) и обычные средние.
    gl — логиты схвата или None (тогда BCE не считается)."""
    s_ = np.asarray(std)
    w = np.asarray(w, np.float64)

    def wm(x):
        return float((x * w).sum() / w.sum())
    e = ((act[..., :6] - A[..., :6]) / s_) ** 2
    Pa, Pt = P_np(act[..., :6]), P_np(A[..., :6])
    gt = A[..., 6] > 0
    gp = act[..., 6] > 0
    res = dict(
        mse_arm_20=wm(e.mean((1, 2))), mse_arm_8=wm(e[:, :8].mean((1, 2))),
        mse_coarse=wm((((Pa - Pt) / s_) ** 2).mean((1, 2))),
        mse_fine=wm(((((act[..., :6] - Pa) - (A[..., :6] - Pt)) / s_) ** 2)
                    .mean((1, 2))),
        grip_acc_20=wm((gp == gt).mean(1)),
        grip_acc_8=wm((gp[:, :8] == gt[:, :8]).mean(1)),
        rows=dict(mse_arm_20=float(e.mean()), mse_arm_8=float(
            e[:, :8].mean()), grip_acc_20=float((gp == gt).mean())))
    if gl is not None:
        z = np.clip(gl, -50, 50)
        bce = np.logaddexp(0, z) - z * gt
        res["grip_bce"] = wm(bce.mean(1))
        res["rows"]["grip_bce"] = float(bce.mean())
    sw_t = np.array([first_switch(np.where(g, 1, -1)) for g in gt])
    sw_p = np.array([first_switch(np.where(g, 1, -1)) for g in gp])
    has = sw_t >= 0
    ok = has & (sw_p >= 0)
    res["switch"] = dict(
        chunks_with_switch=int(has.sum()),
        detected=float((w[has] * (sw_p[has] >= 0)).sum() / w[has].sum())
        if has.any() else None,
        exact=float((w[has] * (sw_p[has] == sw_t[has])).sum()
                    / w[has].sum()) if has.any() else None,
        mean_abs_err=float((w[ok] * np.abs(sw_p[ok] - sw_t[ok])).sum()
                           / w[ok].sum()) if ok.any() else None,
        false_switch=float((w[~has] * (sw_p[~has] >= 0)).sum()
                           / w[~has].sum()) if (~has).any() else None)
    return res


def evaluate(torch, model, A, std, dev, weights=None, bs=1024):
    """Hard-реконструкция на истинных кодах: взвешенные метрики (+ обычные),
    использование словарей, β."""
    model.eval()
    acts, gls, betas, codes = [], [], [], {}
    with torch.no_grad():
        for s in range(0, len(A), bs):
            a = torch.as_tensor(A[s:s + bs], dtype=torch.float32, device=dev)
            if model.kind == "hier":
                kc, kf = model.encode(a)
                o = model.decode(kc, kf)
                codes.setdefault("c", []).append(kc.cpu().numpy())
                codes.setdefault("f", []).append(kf.cpu().numpy())
                betas.append(o["beta"].cpu().numpy())
            else:
                k = model.encode(a)
                o = model.decode(k)
                codes.setdefault("flat", []).append(k.cpu().numpy())
            acts.append(action_out(o).cpu().numpy())
            gls.append(o["grip_logit"].cpu().numpy())
    act, gl = np.concatenate(acts), np.concatenate(gls)
    if not (np.isfinite(act).all() and np.isfinite(gl).all()):
        raise SystemExit(f"кодек {model.kind}: нечисловая реконструкция")
    w = np.ones(len(A)) if weights is None else weights
    res = chunk_metrics(act, gl, A, std, w)
    if betas:
        b = np.concatenate(betas)
        res["beta_lt1_rate"] = float((b < 1).mean())
        res["beta_min"] = float(b.min())
    for k, v in codes.items():
        c = np.concatenate(v).reshape(-1)
        cnt = np.bincount(c, minlength=VOCAB)
        p = cnt / cnt.sum()
        res[f"usage_{k}"] = dict(used=int((cnt > 0).sum()),
                                 frac=float((cnt > 0).mean()),
                                 perplexity=float(np.exp(-(p[p > 0] * np.log(
                                     p[p > 0])).sum())))
    return res, act, codes


def select_metric(r):
    """Выбор шага: взвешенная MSE руки (20) + взвешенная BCE схвата."""
    return r["mse_arm_20"] + r["grip_bce"]


# --- обучение -------------------------------------------------------------------
RESTART_END = int(0.75 * STEPS)       # граница перезапусков не зависит от добора
LR_EXT_END = 3e-6


def cos_lr(step):
    """Основное расписание до STEPS; в доборе — продолжение снижения от LR1
    к LR_EXT_END без скачка."""
    if step < WARM:
        return LR0 * (step + 1) / WARM
    if step < STEPS:
        t = (step - WARM) / max(STEPS - WARM, 1)
        return LR1 + 0.5 * (LR0 - LR1) * (1 + math.cos(math.pi * t))
    t = (step - STEPS) / max(STEPS_MAX - STEPS, 1)
    return LR_EXT_END + 0.5 * (LR1 - LR_EXT_END) * (1 + math.cos(
        math.pi * min(t, 1.0)))


def train(torch, kind, data, dev, steps=STEPS, steps_max=STEPS_MAX,
          eval_every=EVAL_EVERY, batch=BATCH, log=print, seed=0,
          resume=None, restart_end=RESTART_END, save_every=2000):
    M = build(torch)
    torch.manual_seed(seed)
    cls = M["HierCodec"] if kind == "hier" else M["FlatCodec"]
    model = cls(data["L"], data["U"]).to(dev)
    std = torch.as_tensor(data["std"], dtype=torch.float32, device=dev)
    Xe = torch.as_tensor(data["train"], dtype=torch.float32, device=dev)
    we = torch.as_tensor(data["train_w"], dtype=torch.float64)
    Xt = torch.as_tensor(data["teacher"], dtype=torch.float32, device=dev) \
        if data.get("teacher") is not None and len(data["teacher"]) else None
    wt = torch.as_tensor(data["teacher_w"], dtype=torch.float64) \
        if Xt is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=LR0, weight_decay=0.0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    vw = data.get("val_w")
    # фиксированная обучающая панель (взвешенная выборка 2048 строк)
    pr = np.random.default_rng(seed + 7)
    pw = np.asarray(data["train_w"], np.float64)
    panel = np.sort(pr.choice(len(data["train"]), min(2048, len(
        data["train"])), replace=False, p=pw / pw.sum()))
    curve, best, total, step = [], None, steps, 0
    acc = {"expert": {}, "teacher": {}}

    def pool():
        i = torch.multinomial(we, 4096, replacement=True, generator=g).to(dev)
        with torch.no_grad():
            if kind == "hier":
                zc, zf = model.encode_latent(Xe[i])
                return dict(c=zc, f=zf)
            return dict(flat=model.enc(Xe[i]))
    if resume and os.path.exists(resume):
        st = torch.load(resume, map_location="cpu", weights_only=False)
        if st["kind"] != kind or st["config"] != dict(steps=steps,
                                                       steps_max=steps_max,
                                                       batch=batch,
                                                       seed=seed):
            raise SystemExit(f"состояние {resume} от другой конфигурации")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        g.set_state(st["gen"])
        curve, total, step = st["curve"], st["total"], st["step"]
        best = st["best"]
        log(f"  {kind}: продолжение с шага {step}", flush=True)
    else:
        for nm, z in pool().items():
            model.vqs()[nm].restart(z, init=True)
        init, _a, _c = evaluate(torch, model, data["val"], data["std"], dev,
                                vw)
        curve.append(dict(step=0, train=None, val=init, dead=None,
                          seconds=0.0))
    t0 = time.time()
    while step < total:
        model.train()
        for gr in opt.param_groups:
            gr["lr"] = cos_lr(step)
        is_t = Xt is not None and step % 4 == 3
        src, sw = (Xt, wt) if is_t else (Xe, we)
        i = torch.multinomial(sw, batch, replacement=True, generator=g).to(dev)
        A = src[i]
        o = model(A)
        L_ = losses(torch, o, A, std, M["P"])
        if not bool(torch.isfinite(L_["total"])):
            raise SystemExit(f"кодек {kind}: нечисловая потеря на шаге {step}")
        opt.zero_grad()
        L_["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        a_ = acc["teacher" if is_t else "expert"]
        for k, v in L_.items():
            a_[k] = a_.get(k, 0.0) + float(v.detach())
        a_["n"] = a_.get("n", 0) + 1
        dead = None
        if step % 500 == 0 and step <= restart_end:
            dead = {nm: model.vqs()[nm].restart(z) for nm, z in pool().items()}
        if step % eval_every == 0 or step == total:
            r, _a, _c = evaluate(torch, model, data["val"], data["std"], dev,
                                 vw)
            rp, _a, _c = evaluate(torch, model, data["train"][panel],
                                  data["std"], dev,
                                  np.asarray(data["train_w"])[panel])
            tr = {src_: {k: v / a_["n"] for k, v in a_.items() if k != "n"}
                  for src_, a_ in acc.items() if a_.get("n")}
            acc = {"expert": {}, "teacher": {}}
            curve.append(dict(step=step, train=tr, train_panel=rp, val=r,
                              dead=dead, lr=cos_lr(step - 1),
                              seconds=time.time() - t0))
            m = select_metric(r)
            if not np.isfinite(m):
                raise SystemExit(f"кодек {kind}: нечисловая проверка")
            if best is None or m < best[1]:
                best = (step, m, {k: v.detach().cpu().clone() for k, v in
                                  model.state_dict().items()})
            te = tr.get("expert", {})
            log(f"  {kind} шаг {step}: эксперт {te.get('total', float('nan')):.4f}"
                f", учитель {tr.get('teacher', {}).get('total', float('nan')):.4f}"
                f"; панель train рука-20 {rp['mse_arm_20']:.4f}; val рука-20 "
                f"{r['mse_arm_20']:.4f}, -8 {r['mse_arm_8']:.4f}, BCE схвата "
                f"{r['grip_bce']:.4f}, {time.time() - t0:.0f} с", flush=True)
            if step == total and total < steps_max and best[0] == step and \
                    len(curve) >= 3 and curve[-1]["val"]["mse_arm_20"] < \
                    curve[-3]["val"]["mse_arm_20"]:
                total = steps_max
                log(f"  {kind}: лучший шаг последний и ошибка падает — "
                    f"заранее заданный добор до {steps_max}", flush=True)
        if resume and (step % save_every == 0 or step == total):
            torch.save(dict(kind=kind, model=model.state_dict(),
                            opt=opt.state_dict(), gen=g.get_state(),
                            curve=curve, total=total, step=step, best=best,
                            config=dict(steps=steps, steps_max=steps_max,
                                        batch=batch, seed=seed)),
                       resume + ".tmp")
            os.replace(resume + ".tmp", resume)
    model.load_state_dict(best[2])
    return model, dict(best_step=best[0], best_metric=best[1],
                       best_is_last=best[0] == total, steps=total,
                       curve=curve, params=sum(p.numel() for p in
                                               model.parameters()))


def load_data(index, cache_meta, teacher, orig, prepare_report):
    """Строгая загрузка: завершённый prepare, совпадение отпечатков,
    учитель и исходные реконструкции обязательны."""
    for f in (index, teacher, orig, prepare_report):
        if not os.path.exists(f):
            raise SystemExit(f"нет {f}: prepare не завершён")
    rep = json.load(open(prepare_report))
    if rep.get("kind") != "k20_prepare" or not rep.get("complete"):
        raise SystemExit("prepare не завершён")
    for f, k in ((index, "index_sha1"), (teacher, "teacher_sha1"),
                 (orig, "orig_codec_val_sha1")):
        if sha(f) != rep.get(k):
            raise SystemExit(f"{f} не совпадает с отпечатком prepare")
    meta = json.load(open(f"{cache_meta}.meta.json"))
    d = np.load(meta["cache"], allow_pickle=True)
    act = np.asarray(d["action"], np.float32)
    ix = dict(np.load(index, allow_pickle=False))
    t = dict(np.load(teacher))
    o = dict(np.load(orig))
    n = len(ix["teacher"])
    if not (np.array_equal(t["rows"], ix["teacher"])
            and len(np.unique(t["rows"])) == n
            and t["raw"].shape == (n, T, 7) and t["target"].shape == (n, T, 7)
            and np.isfinite(t["raw"]).all()
            and np.isfinite(t["target"]).all()):
        raise SystemExit("учитель: строки, уникальность, формы или "
                         "конечность не совпадают")
    if not np.isin(t["rows"], ix["train_fit"]).all():
        raise SystemExit("строки учителя вне train_fit")
    want = t["raw"].copy()
    want[..., :6] = np.clip(t["raw"][..., :6], ix["L"], ix["U"])
    if not np.array_equal(want, t["target"]):
        raise SystemExit("цель учителя ≠ clip(raw, L, U) по руке")
    nv = len(ix["val_sel"])
    if not (np.array_equal(o["rows"], ix["val_sel"])
            and o["recon3"].shape == (nv, T, 7)
            and o["recon1"].shape == (nv, T, 7)
            and np.isfinite(o["recon3"]).all()
            and np.isfinite(o["recon1"]).all()):
        raise SystemExit("реконструкции исходного кодека не совпадают с "
                         "val_sel или не конечны")
    tr, va = ix["train"], ix["val_sel"]
    return dict(L=ix["L"], U=ix["U"], std=ix["std_arm"],
                train=act[tr], val=act[va],
                train_w=row_weights(ix["task"][tr], ix["episode"][tr]),
                val_w=row_weights(ix["task"][va], ix["episode"][va]),
                teacher=np.asarray(t["target"], np.float32),
                teacher_w=row_weights(ix["task"][t["rows"]],
                                      ix["episode"][t["rows"]]),
                orig=o, ix=ix, prepare=rep)


FIT_RATIO, FIT_GRIP = 0.25, 0.95


def fit_ok(r, X, std):
    """Подгонка 512: ошибка руки < FIT_RATIO · min(нулевое, среднее
    предсказание) и точность схвата ≥ FIT_GRIP; всё конечно."""
    s_ = np.asarray(std)
    zero = float(((X[..., :6] / s_) ** 2).mean())
    mean = float((((X[..., :6] - X[..., :6].mean(0)) / s_) ** 2).mean())
    fin = r["rows"]["mse_arm_20"]
    ok = (np.isfinite(fin) and fin < FIT_RATIO * min(zero, mean)
          and r["rows"]["grip_acc_20"] >= FIT_GRIP)
    return bool(ok), dict(zero=zero, mean=mean, final=fin,
                          grip=r["rows"]["grip_acc_20"],
                          threshold=FIT_RATIO * min(zero, mean))


def main_fitcheck(a):
    """Подгонка 512 фиксированных train-чанков (проверка оптимизации);
    провал критерия останавливает основной запуск."""
    import torch
    dev = torch.device(a.device)
    data = load_data(a.index, a.cache, a.teacher, a.orig, a.prepare)
    rng = np.random.default_rng(20261019)
    sel = np.sort(rng.choice(len(data["train"]), 512, replace=False))
    X = data["train"][sel]
    d = dict(data, train=X, val=X, teacher=None, train_w=np.ones(512),
             val_w=np.ones(512))
    rep, bad = {}, []
    for kind in a.kinds:
        _m, r = train(torch, kind, d, dev, steps=a.fit_steps,
                      steps_max=a.fit_steps, eval_every=500, batch=128,
                      log=lambda *x, **k: None, restart_end=a.fit_steps // 2)
        ok, info = fit_ok(r["curve"][-1]["val"], X, data["std"])
        rep[kind] = dict(steps=a.fit_steps, ok=ok, **info)
        print(f"  подгонка 512 чанков {kind}: рука-20 {info['final']:.4f} "
              f"(нулевое {info['zero']:.4f}, среднее {info['mean']:.4f}, "
              f"порог {info['threshold']:.4f}), схват "
              f"{100 * info['grip']:.1f}% — "
              + ("ПРОЙДЕНА" if ok else "НЕ ПРОЙДЕНА"), flush=True)
        if not ok:
            bad.append(kind)
    os.makedirs("reports/k20", exist_ok=True)
    json.dump(rep, open("reports/k20/codec_fitcheck.json", "w"), indent=1)
    if bad:
        print(f"  ОТКАЗ: подгонка не пройдена: {bad}")
        return 3
    return 0


def main_train(a):
    import torch
    dev = torch.device(a.device)
    data = load_data(a.index, a.cache, a.teacher, a.orig, a.prepare)
    rep = {}
    for kind in a.kinds:
        out = os.path.join("data/k20", f"codec_{kind}.pt")
        if os.path.exists(out):
            raise SystemExit(f"{out} уже есть")
        model, r = train(torch, kind, data, dev, steps=a.steps,
                         steps_max=a.steps_max,
                         resume=os.path.join("data/k20",
                                             f"codec_{kind}.resume.pt"))
        ev, _act, _codes = evaluate(torch, model, data["val"], data["std"],
                                    dev, data["val_w"])
        r["val_sel"] = ev
        r["teacher_fit"], _x, _y = evaluate(torch, model, data["teacher"],
                                            data["std"], dev,
                                            data["teacher_w"])
        o = data["orig"]
        for lv in ("recon3", "recon1"):
            r[f"orig_{lv}"] = chunk_metrics(o[lv], None, data["val"],
                                            data["std"], data["val_w"])
        state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(dict(kind=kind, state=state, L=data["L"], U=data["U"],
                        std=data["std"], best_step=r["best_step"],
                        index_sha1=sha(a.index),
                        prepare_sha1=sha(a.prepare),
                        code_sha1=sha(os.path.abspath(__file__))),
                   out + ".tmp")
        os.replace(out + ".tmp", out)
        rs = os.path.join("data/k20", f"codec_{kind}.resume.pt")
        if os.path.exists(rs):
            os.remove(rs)
        r["path"], r["sha1"] = os.path.abspath(out), sha(out)
        rep[kind] = r
        print(f"  {kind}: шаг {r['best_step']}"
              + (" (последний)" if r["best_is_last"] else "")
              + f", параметров {r['params']}; val_sel (истинные коды, "
              f"равные задачи/эпизоды): рука-20 {ev['mse_arm_20']:.4f}, "
              f"рука-8 {ev['mse_arm_8']:.4f}, coarse {ev['mse_coarse']:.4f},"
              f" fine {ev['mse_fine']:.4f}, схват {100 * ev['grip_acc_20']:.1f}"
              f"%, BCE {ev['grip_bce']:.4f}; переключения {ev['switch']}",
              flush=True)
        for lv in ("recon3", "recon1"):
            x = r[f"orig_{lv}"]
            print(f"    исходный ActionCodec {lv}: рука-20 "
                  f"{x['mse_arm_20']:.4f}, рука-8 {x['mse_arm_8']:.4f}, "
                  f"схват {100 * x['grip_acc_20']:.1f}%", flush=True)
    os.makedirs("reports/k20", exist_ok=True)
    p = "reports/k20/codec.json"
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(rep)
    json.dump(old, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"  отчёт: {p}")
    return 0


# --- самопроверка ------------------------------------------------------------------
def selftest():
    import tempfile
    import torch
    rng = np.random.default_rng(0)
    M = build(torch)
    A = rng.uniform(-1, 1, (8, T, 7))
    # точная сборка
    C = P_np(A[..., :6])
    assert np.allclose(C + (A[..., :6] - C), A[..., :6])
    assert np.allclose(P_np(A[..., :6] - C), 0, atol=1e-12)
    At = torch.as_tensor(A)
    assert np.allclose(M["P"](At[..., :6]).numpy(), C)
    # β: границы, средние, нули, насыщение
    L, U = -np.ones(6) * 1.2, np.ones(6) * 1.5
    Ct = np.clip(C * 1.3, L + 1e-3, U - 1e-3)
    R = (A[..., :6] - C) * 3
    b = beta_np(Ct, R, L, U)
    out = Ct + np.repeat(b, BL, -1)[..., None] * R
    assert (out >= L - 1e-9).all() and (out <= U + 1e-9).all()
    assert np.allclose(P_np(out), Ct)
    assert (beta_np(Ct, np.zeros_like(R), L, U) == 1).all()
    bt = M["beta"](torch.as_tensor(Ct), torch.as_tensor(R),
                   torch.as_tensor(L), torch.as_tensor(U))
    assert np.allclose(bt.numpy(), b)
    sat = np.tile(U, (1, T, 1))
    assert (beta_np(sat, np.ones((1, T, 6)), L, U) == 0).all()
    # модели: схват не проходит через усреднение, средние сохраняются
    for cls in ("HierCodec", "FlatCodec"):
        m = M[cls](L, U)
        o = m(torch.as_tensor(A, dtype=torch.float32))
        assert o["arm"].shape == (8, T, 6) and o["grip_logit"].shape == (8, T)
        assert ((o["arm"] >= torch.as_tensor(L).float() - 1e-5).all()
                and (o["arm"] <= torch.as_tensor(U).float() + 1e-5).all())
        if cls == "HierCodec":
            assert torch.allclose(M["P"](o["arm"]), o["C"], atol=1e-5)
            xc, xf = m.split(torch.as_tensor(A, dtype=torch.float32))
            assert not xc[..., 6].any()
            assert torch.equal(xf[..., 6], torch.as_tensor(
                A[..., 6], dtype=torch.float32))
    # подгонка 512 чанков маленьким кодеком (оба вида)
    X = rng.uniform(-.8, .8, (512, T, 7)).astype(np.float32)
    X[..., :6] = np.cumsum(X[..., :6] * .1, axis=1).clip(-1, 1)
    X[..., 6] = np.where(np.arange(T) < rng.integers(0, T, (512, 1)), -1., 1.)
    data = dict(L=L, U=U, std=np.full(6, .3), train=X, val=X[:128],
                teacher=X[:64], train_w=np.ones(512), val_w=np.ones(128),
                teacher_w=np.ones(64))
    for kind in ("hier", "flat"):
        m, r = train(torch, kind, data, "cpu", steps=240, steps_max=240,
                     eval_every=120, batch=32, log=lambda *x, **k: None)
        c = r["curve"]
        assert c[-1]["val"]["mse_arm_20"] < 0.6 * c[0]["val"]["mse_arm_20"], \
            (kind, c[0]["val"]["mse_arm_20"], c[-1]["val"]["mse_arm_20"])
        assert all(c_["train"] is None or set(c_["train"]) <= {"expert",
                                                                "teacher"}
                   for c_ in c)
        assert "expert" in c[-1]["train"] and "teacher" in c[-1]["train"]
        ev, act, codes = evaluate(torch, m, X[:128], np.full(6, .3), "cpu")
        ev2, act2, _ = evaluate(torch, m, X[:128], np.full(6, .3), "cpu")
        assert np.array_equal(act, act2)                    # детерминизм
        assert set(np.unique(act[..., 6])) <= {-1.0, 1.0}
        with tempfile.TemporaryDirectory() as td:
            torch.save(m.state_dict(), td + "/m.pt")
            m2 = M["HierCodec" if kind == "hier" else "FlatCodec"](L, U)
            m2.load_state_dict(torch.load(td + "/m.pt"))
            ev3, act3, _ = evaluate(torch, m2, X[:128], np.full(6, .3),
                                    "cpu")
            assert np.array_equal(act, act3)
    # строгая загрузка: отказы на повторах, формах, NaN, нарушении clip,
    # неполном prepare и изменённых файлах
    with tempfile.TemporaryDirectory() as td:
        N = 60
        act = rng.uniform(-1, 1, (N, T, 7)).astype(np.float32)
        np.savez(td + "/cache.npz", action=act)
        json.dump(dict(cache=td + "/cache.npz"), open(td + "/c.meta.json",
                                                      "w"))
        L6, U6 = -np.ones(6) * 1.1, np.ones(6) * 1.1
        ix = dict(train=np.arange(40), val_sel=np.arange(40, 60),
                  train_fit=np.arange(30), teacher=np.arange(0, 30, 3),
                  task=np.array(["a", "b"] * 30), episode=np.arange(N) // 5,
                  L=L6, U=U6, std_arm=np.ones(6))
        raw = rng.uniform(-1.5, 1.5, (10, T, 7)).astype(np.float32)
        tgt = raw.copy()
        tgt[..., :6] = np.clip(raw[..., :6], L6, U6)

        def files(ixd=ix, tr=None, oo=None, complete=True):
            np.savez(td + "/index.npz", **ixd)
            np.savez(td + "/teacher.npz", **(tr or dict(
                rows=ixd["teacher"], raw=raw, target=tgt)))
            np.savez(td + "/orig.npz", **(oo or dict(
                rows=ixd["val_sel"], recon3=act[40:], recon1=act[40:])))
            json.dump(dict(kind="k20_prepare", complete=complete,
                           index_sha1=sha(td + "/index.npz"),
                           teacher_sha1=sha(td + "/teacher.npz"),
                           orig_codec_val_sha1=sha(td + "/orig.npz")),
                      open(td + "/prep.json", "w"))

        def ld():
            return load_data(td + "/index.npz", td + "/c", td + "/teacher.npz",
                             td + "/orig.npz", td + "/prep.json")
        files()
        dd = ld()
        assert dd["train"].shape == (40, T, 7) and len(dd["teacher"]) == 10
        bad = [lambda: files(tr=dict(rows=np.r_[ix["teacher"][:-1], 0],
                                     raw=raw, target=tgt)),
               lambda: files(tr=dict(rows=ix["teacher"], raw=raw,
                                     target=tgt[:9])),
               lambda: files(tr=dict(rows=ix["teacher"], raw=raw * np.nan,
                                     target=tgt)),
               lambda: files(tr=dict(rows=ix["teacher"], raw=raw,
                                     target=raw)),
               lambda: files(oo=dict(rows=ix["val_sel"], recon3=act[40:] *
                                     np.nan, recon1=act[40:])),
               lambda: files(complete=False)]
        for i_, mk in enumerate(bad):
            mk()
            try:
                ld()
                raise AssertionError(f"принят повреждённый вход {i_}")
            except SystemExit:
                pass
        files()
        np.savez(td + "/teacher.npz", rows=ix["teacher"], raw=raw,
                 target=tgt * 1.0)               # другой файл -> другой sha
        open(td + "/teacher.npz", "ab").write(b"x")
        try:
            ld()
            raise AssertionError("принят изменённый файл учителя")
        except SystemExit:
            pass
    # веса: равные задачи, эпизоды, строки
    w = row_weights(["a", "a", "a", "b"], [1, 1, 2, 3])
    assert np.allclose(w / w.sum(), [.125, .125, .25, .5])
    # расписание: без скачка LR в доборе, перезапуски не возобновляются
    assert abs(cos_lr(STEPS) - LR1) < 1e-9 and cos_lr(STEPS + 1) <= LR1
    assert cos_lr(STEPS_MAX) <= LR_EXT_END * 1.01
    lrs = [cos_lr(k) for k in range(WARM, STEPS_MAX)]
    assert all(b <= a + 1e-15 for a, b in zip(lrs, lrs[1:]))
    assert RESTART_END == 15000
    # критерий подгонки: среднее предсказание не проходит
    Xf = rng.uniform(-1, 1, (64, T, 7))
    rr = dict(rows=dict(mse_arm_20=float((((Xf[..., :6] - Xf[..., :6].mean(0))
                                           / .3) ** 2).mean()),
                        grip_acc_20=1.0))
    assert not fit_ok(rr, Xf, np.full(6, .3))[0]
    rr["rows"]["mse_arm_20"] = 1e-4
    assert fit_ok(rr, Xf, np.full(6, .3))[0]
    print("самопроверка k20_codec пройдена: точная сборка P, β (границы, "
          "средние, нули, насыщение), схват вне усреднения, допустимые "
          "команды, подгонка 512 чанков (иерархия и плоский), "
          "детерминизм, сохранение, раздельные ошибки эксперта и учителя, "
          "веса задач/эпизодов, расписание добора, критерий подгонки, "
          "строгая загрузка входов")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-20: токенизаторы")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--index", default="data/k20/index.npz")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--teacher", default="data/k20/teacher.npz")
    ap.add_argument("--orig", default="data/k20/orig_codec_val.npz")
    ap.add_argument("--prepare", default="reports/k20/prepare.json")
    ap.add_argument("--kinds", nargs="+", default=["hier", "flat"])
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--steps-max", type=int, default=STEPS_MAX)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fitcheck", action="store_true")
    ap.add_argument("--fit-steps", type=int, default=2000)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        if a.fitcheck:
            return main_fitcheck(a)
        return main_train(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
