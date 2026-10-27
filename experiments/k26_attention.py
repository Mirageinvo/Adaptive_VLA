#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-26: визуальное чтение action-query с наследованием внимания по дереву.

Визуальный банк — признаки изображения из НАЧАЛЬНОГО hv0 (после
vision/connector, вставленные inputs_merger в префикс, до слоя 1 VLM) в
позициях (input_ids == image_token_id) & attention_mask; порядок —
порядок появления позиций. E [B, Nmax, Dv] (отсоединён от графа vision),
valid [B, Nmax]. Наблюдение без единого image-токена — отказ.

Общий VisualRouter (один на модель, шесть точек подключения):
  En = LN(E) (float32, без affine, eps 1e-5); K = Wk(En), V = Wv(En) —
  один раз на forward; Q = Wq(LN(ha)) всех активных query сразу;
  S = Q·Kᵀ/√32 (float32, вне autocast), 4 головы по 32, d = 128;
  корни:   P0 = softmax(S0 + log U);
  средние: P1_j = softmax(S1_j + log((1−ε)·P0_{j//2} + ε·U));
  листья:  P2_l = softmax(S2_l + log((1−ε)·P1_{l//2} + ε·U));
  flat:    P = softmax(S + log U);
  C = concat_heads(P·V); ha ← ha + γ·Wo(C), γ = 0.1, ε = 0.1 фиксированы.
U — равномерное по допустимым токенам; на padding P строго 0. log-prior —
через logaddexp (ε = 1 даёт ровно log U, т. е. независимое внимание).
Линейные Wq/Wk/Wv/Wo без bias, Wo = 0 при инициализации (выход модели =
базовой), Wq/Wk/Wv — U(±1/√fan_in) из локального генератора.
Состояния между вызовами нет; карты в каждой точке считаются заново.
"""
import math
import sys

D_ROUTER, N_HEADS = 128, 4
HEAD_DIM = D_ROUTER // N_HEADS
GAMMA, EPS = 0.1, 0.1
LN_EPS = 1e-5
SUM_TOL = 5e-6
AFTER = (3, 7, 11, 15, 19, 23)          # индексы слоёв (с нуля): после 4, 8, …, 24
VERSION = "attn1"
CONFIG = dict(version=VERSION, d=D_ROUTER, heads=N_HEADS, head_dim=HEAD_DIM,
              gamma=GAMMA, eps=EPS, ln_eps=LN_EPS, temperature=1.0,
              bias=False, after_layers_1based=[a + 1 for a in AFTER],
              wo_init="zeros", qkv_init="U(±1/sqrt(fan_in)), local generator",
              bank="hv0 image positions (input_ids == image_token_id) & "
                   "attention_mask, before VLM layer 1")


def n_new_params(H, Dv, d=D_ROUTER):
    return 2 * d * (H + Dv)


def build(torch):
    nn = torch.nn
    F = nn.functional
    NEG = float("-inf")

    def wdt(*xs):
        dt = torch.float32
        for x in xs:
            dt = torch.promote_types(dt, x.dtype)
        return dt

    def check_rows(valid):
        if valid.dtype != torch.bool or valid.dim() != 2:
            raise SystemExit(f"маска банка: {valid.dtype}, {tuple(valid.shape)}")
        if not bool(valid.any(-1).all()):
            raise SystemExit("визуальный банк: строка без допустимых "
                             "image-токенов (до softmax)")

    def pack_image_bank(hv0, input_ids, attention_mask, image_token_id):
        """hv0 [B, V, Dv] -> E [B, Nmax, Dv] (detach), valid [B, Nmax],
        counts [B]. Позиции — (ids == image_token_id) & attention_mask."""
        B, V, Dv = hv0.shape
        if tuple(input_ids.shape) != (B, V) or \
                tuple(attention_mask.shape) != (B, V):
            raise SystemExit(f"банк: input_ids {tuple(input_ids.shape)}, "
                             f"mask {tuple(attention_mask.shape)}, hv0 "
                             f"{(B, V)}")
        m = (input_ids == int(image_token_id)) & attention_mask.bool()
        counts = m.sum(1)
        if bool((counts == 0).any()):
            raise SystemExit("визуальный банк: у наблюдения нет ни одного "
                             "действительного image-токена")
        nmax = int(counts.max())
        E = hv0.new_zeros((B, nmax, Dv))
        valid = torch.zeros((B, nmax), dtype=torch.bool, device=hv0.device)
        src = hv0.detach()
        for b in range(B):
            n = int(counts[b])
            E[b, :n] = src[b, m[b]]
            valid[b, :n] = True
        return E, valid, counts

    def log_uniform(valid, dt):
        """[B, N] -> [B, 1, 1, N]: −log N_b на допустимых, 0 на padding
        (безопасное значение; маска накладывается на логиты)."""
        n = valid.sum(-1, keepdim=True).to(dt)
        lu = torch.where(valid, -torch.log(n).expand(valid.shape),
                         torch.zeros((), dtype=dt, device=valid.device))
        return lu[:, None, None, :]

    def masked_log_softmax(x, v4):
        return torch.log_softmax(torch.where(
            v4, x, torch.full((), NEG, dtype=x.dtype, device=x.device)), -1)

    def tree_probs(S, valid, sizes, eps, hierarchical=True):
        """S [B, h, A, N]; sizes — числа узлов уровней ((16,) у flat).
        -> dict(P, logP, log_prior) [B, h, A, N]. Явно переданный float64
        сохраняется, half/bf16 повышаются до float32."""
        check_rows(valid)
        dt = wdt(S)
        S = S.to(dt)
        B, h, A, N = S.shape
        if valid.shape != (B, N):
            raise SystemExit(f"маска {tuple(valid.shape)} ≠ {(B, N)}")
        if sum(sizes) != A:
            raise SystemExit(f"узлов {A}, уровни {sizes}")
        if not 0.0 < eps <= 1.0:
            raise SystemExit(f"epsilon {eps} вне (0, 1]")
        v4 = valid[:, None, None, :]
        lu = log_uniform(valid, dt)
        if not hierarchical:
            if len(sizes) != 1:
                raise SystemExit(f"flat: уровни {sizes}")
            lp = masked_log_softmax(S + lu, v4)
            return dict(P=lp.exp(), logP=lp, log_prior=lu.expand(B, h, A, N))
        outs, priors, off, prev = [], [], 0, None
        for lev, n in enumerate(sizes):
            s = S[:, :, off:off + n]
            if lev == 0:
                lpr = lu.expand(B, h, n, N)
            else:
                if n != 2 * sizes[lev - 1]:
                    raise SystemExit(f"уровень {lev}: {n} ≠ 2·{sizes[lev - 1]}")
                par = torch.arange(n, device=S.device) // 2
                lpp = torch.where(v4, prev, torch.zeros((), dtype=dt,
                                                        device=S.device))
                lpp = lpp[:, :, par]
                if eps >= 1.0:
                    lpr = lu.expand(B, h, n, N)
                else:
                    lpr = torch.logaddexp(lpp + math.log1p(-eps),
                                          lu + math.log(eps))
            lp = masked_log_softmax(s + lpr, v4)
            outs.append(lp)
            priors.append(lpr)
            prev = lp
            off += n
        lp = torch.cat(outs, 2)
        return dict(P=lp.exp(), logP=lp, log_prior=torch.cat(priors, 2))

    def weighted_context(P, Vimg):
        """P [B, h, A, N], Vimg [B, h, N, dh] -> [B, A, h·dh]."""
        dt = wdt(P, Vimg)
        C = torch.matmul(P.to(dt), Vimg.to(dt))
        B, h, A, dh = C.shape
        return C.transpose(1, 2).reshape(B, A, h * dh)

    def summaries(tp, valid, sizes, ha, upd):
        """Агрегаты одной точки (без градиента): по уровням нормированная
        энтропия и KL(P ‖ prior); масса на padding, ошибка суммы, RMS
        поправки к RMS ha, конечность."""
        with torch.no_grad():
            P, lp, lpr = (tp[k].float() for k in ("P", "logP", "log_prior"))
            v4 = valid[:, None, None, :]
            n = valid.sum(-1).float()
            on = v4 & (P > 0)
            zero = torch.zeros((), device=P.device)
            ent = -torch.where(on, P * lp, zero).sum(-1)            # [B,h,A]
            ln = torch.log(n)[:, None, None]
            ent_n = torch.where(ln > 0, ent / ln.clamp_min(1e-30), zero)
            kl = torch.where(on, P * (lp - lpr), zero).sum(-1)
            lev, off = [], 0
            for i, k in enumerate(sizes):
                sl = slice(off, off + k)
                lev.append(dict(nodes=k, entropy=ent_n[:, :, sl].mean(),
                                kl_to_prior=kl[:, :, sl].mean() if i else
                                None))
                off += k
            finite = torch.isfinite(P).all() & torch.isfinite(
                torch.where(v4.expand_as(lp), lp, zero)).all()
            return dict(
                levels=lev, pad_mass=(P * (~v4)).sum(-1).max(),
                sum_err=(P.sum(-1) - 1).abs().max(), finite=finite,
                rms_ratio=upd.float().pow(2).mean().sqrt() /
                ha.float().pow(2).mean().sqrt().clamp_min(1e-30),
                n_tokens=n)

    def uniform_(t, bound, gen):
        with torch.no_grad():
            t.copy_((torch.rand(t.shape, generator=gen) * 2 - 1) * bound)

    class VisualRouter(nn.Module):
        def __init__(self, H, Dv, gen, d=D_ROUTER, heads=N_HEADS,
                     gamma=GAMMA, eps=EPS):
            super().__init__()
            if d % heads:
                raise SystemExit(f"d {d} не делится на {heads} голов")
            self.H, self.Dv, self.d, self.h, self.dh = H, Dv, d, heads, \
                d // heads
            self.gamma, self.eps = float(gamma), float(eps)
            self.Wq = nn.Linear(H, d, bias=False)
            self.Wk = nn.Linear(Dv, d, bias=False)
            self.Wv = nn.Linear(Dv, d, bias=False)
            self.Wo = nn.Linear(d, H, bias=False)
            for m in (self.Wq, self.Wk, self.Wv):
                uniform_(m.weight, 1.0 / math.sqrt(m.in_features), gen)
            with torch.no_grad():
                self.Wo.weight.zero_()

        def _heads(self, x):
            B, T_, _ = x.shape
            return x.view(B, T_, self.h, self.dh).transpose(1, 2)

        def project_bank(self, E, valid):
            check_rows(valid)
            if E.shape[-1] != self.Dv or valid.shape != E.shape[:2]:
                raise SystemExit(f"банк {tuple(E.shape)}, маска "
                                 f"{tuple(valid.shape)}, Dv {self.Dv}")
            with torch.autocast(E.device.type, enabled=False):
                En = F.layer_norm(E.detach().float(), (self.Dv,), eps=LN_EPS)
                En = torch.where(valid[..., None], En, torch.zeros(
                    (), device=En.device))
            return self._heads(self.Wk(En)), self._heads(self.Wv(En))

        def forward(self, ha, K, Vimg, valid, sizes, hierarchical,
                    collect=False, keep_maps=False):
            dev = ha.device.type
            with torch.autocast(dev, enabled=False):
                hn = F.layer_norm(ha.float(), (self.H,), eps=LN_EPS)
            Q = self._heads(self.Wq(hn))
            with torch.autocast(dev, enabled=False):
                S = torch.matmul(Q.float(), K.float().transpose(-1, -2)) / \
                    math.sqrt(self.dh)
                tp = tree_probs(S, valid, sizes,
                                self.eps if hierarchical else 1.0,
                                hierarchical)
                C = weighted_context(tp["P"], Vimg)
            upd = self.gamma * self.Wo(C).to(ha.dtype)
            info = None
            if collect or keep_maps:
                info = summaries(tp, valid, sizes, ha, upd)
                if keep_maps:
                    info["maps"] = dict(
                        S=S.detach().float().cpu(),
                        P=tp["P"].detach().float().cpu(),
                        log_prior=tp["log_prior"].detach().float().cpu(),
                        valid=valid.detach().cpu())
            return ha + upd, info

    return dict(pack_image_bank=pack_image_bank, tree_probs=tree_probs,
                weighted_context=weighted_context, VisualRouter=VisualRouter,
                log_uniform=log_uniform, summaries=summaries)


def info_floats(info):
    """Агрегаты точки -> обычные числа (одна выгрузка на CPU)."""
    def f(x):
        return None if x is None else float(x)
    return dict(levels=[dict(nodes=v["nodes"], entropy=f(v["entropy"]),
                             kl_to_prior=f(v["kl_to_prior"]))
                        for v in info["levels"]],
                pad_mass=f(info["pad_mass"]), sum_err=f(info["sum_err"]),
                finite=bool(info["finite"]), rms_ratio=f(info["rms_ratio"]),
                n_tokens=[int(x) for x in info["n_tokens"].tolist()])


def selftest():
    import torch
    R = build(torch)
    tp, wc = R["tree_probs"], R["weighted_context"]
    g = torch.Generator().manual_seed(0)
    sizes = (4, 8, 16)
    B, h, N = 3, 4, 9
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[1, 6:] = False
    valid[2, 1:] = False                          # один допустимый токен
    S = torch.randn(B, h, 28, N, generator=g) * 3
    # 1. формы, конечность, сумма 1, нули на padding
    for hier, sz in ((True, sizes), (False, (28,))):
        o = tp(S, valid, sz, EPS, hier)
        P = o["P"]
        assert P.shape == S.shape and torch.isfinite(P).all() and \
            (P >= 0).all()
        assert float((P.sum(-1) - 1).abs().max()) <= SUM_TOL
        assert float(P[1, :, :, 6:].abs().max()) == 0.0
        assert float(P[2, :, :, 1:].abs().max()) == 0.0
        assert torch.allclose(P[2, :, :, 0], torch.ones(h, 28))
    # 3. нулевые scores -> равномерные карты; epsilon = 1 -> независимые
    o = tp(torch.zeros(B, h, 28, N), valid, sizes, EPS)
    nb = valid.sum(-1).float()
    assert torch.allclose(o["P"].sum(-1), torch.ones(B, h, 28))
    assert torch.allclose(o["P"][0], torch.full((h, 28, N), 1 / 9), atol=1e-7)
    assert torch.allclose(o["P"][1, :, :, :6], torch.full((h, 28, 6),
                                                          1 / 6), atol=1e-7)
    _ = nb
    ind = tp(S, valid, (28,), 1.0, False)["P"]
    e1 = tp(S, valid, sizes, 1.0, True)["P"]
    assert torch.equal(e1, ind)
    # 4. меняются только scores родителя -> меняется карта ребёнка при
    # epsilon 0.1 и не меняется при epsilon 1 (корень -> средний, средний ->
    # лист); проверяется сама дочерняя карта
    for par_sl, ch_sl in ((slice(0, 4), slice(4, 12)),
                          (slice(4, 12), slice(12, 28))):
        S2 = S.clone()
        S2[:, :, par_sl] += torch.randn(S2[:, :, par_sl].shape,
                                        generator=g) * 2
        for eps, changes in ((EPS, True), (1.0, False)):
            a = tp(S, valid, sizes, eps)["P"][:, :, ch_sl]
            b = tp(S2, valid, sizes, eps)["P"][:, :, ch_sl]
            d = float((a - b).abs().max())
            assert (d > 1e-4) if changes else (d == 0.0), (par_sl, eps, d)
    # 5. индексы родителей: корень r -> средние 2r, 2r+1; средний j (полный
    # индекс 4 + j) -> листья 2j, 2j+1; разные пики родителей
    Np = 40
    vv = torch.ones(1, Np, dtype=torch.bool)
    Sx = torch.zeros(1, 1, 28, Np)
    for r in range(4):
        Sx[0, 0, r, r] = 30.0                     # пик корня r на токене r
    o = tp(Sx, vv, sizes, EPS)["P"][0, 0]
    for j in range(8):
        assert int(o[4 + j].argmax()) == j // 2, (j, o[4 + j].argmax())
    for l_ in range(16):
        assert int(o[12 + l_].argmax()) == (l_ // 2) // 2
    Sx2 = Sx.clone()
    for j in range(8):
        Sx2[0, 0, 4 + j, 10 + j] = 40.0           # собственный пик среднего
    o2 = tp(Sx2, vv, sizes, EPS)["P"][0, 0]
    for l_ in range(16):
        assert int(o2[12 + l_].argmax()) == 10 + l_ // 2, l_
    # 6. большие конечные scores, один токен, разное число токенов,
    # полностью запрещённая строка -> понятный отказ до softmax
    big = torch.randn(B, h, 28, N, generator=g) * 1e6
    ob = tp(big, valid, sizes, EPS)["P"]
    assert torch.isfinite(ob).all() and float((ob.sum(-1) - 1).abs().max()) \
        <= SUM_TOL
    bad = valid.clone()
    bad[0] = False
    try:
        tp(S, bad, sizes, EPS)
        raise AssertionError("строка без токенов принята")
    except SystemExit as e:
        assert "без допустимых" in str(e.code)
    # 7. gradcheck float64: наследование и взвешенный контекст (та же функция)
    Sd = torch.randn(2, 2, 28, 6, dtype=torch.float64, generator=g,
                     requires_grad=True)
    vd = torch.ones(2, 6, dtype=torch.bool)
    vd[1, 4:] = False
    Vd = torch.randn(2, 2, 6, 3, dtype=torch.float64, generator=g,
                     requires_grad=True)
    assert tp(Sd, vd, sizes, EPS)["P"].dtype == torch.float64
    assert torch.autograd.gradcheck(
        lambda s, v: wc(tp(s, vd, sizes, EPS)["P"], v), (Sd, Vd),
        eps=1e-6, atol=1e-5)
    assert torch.autograd.gradcheck(
        lambda s: tp(s, vd, (28,), 1.0, False)["P"], (Sd,), eps=1e-6,
        atol=1e-5)
    # нет detach: потеря контекста листьев даёт градиент scores корней
    # (только через наследование: при epsilon 1 — ровно ноль)
    for eps, nz in ((EPS, True), (1.0, False)):
        s = S.clone().requires_grad_(True)
        Vx = torch.randn(B, h, N, 5, generator=g)
        c = wc(tp(s, valid, sizes, eps)["P"], Vx)[:, 12:28]
        c.pow(2).sum().backward()
        gr = float(s.grad[:, :, :4].abs().sum())
        assert (gr > 0) if nz else (gr == 0.0), (eps, gr)
    # half -> float32
    assert tp(S.half(), valid, sizes, EPS)["P"].dtype == torch.float32
    # банк: позиции image-токенов, порядок, padding и текст не попадают
    IMG = 7
    ids = torch.tensor([[1, 7, 7, 2, 7, 3], [0, 0, 7, 7, 4, 5],
                        [7, 2, 2, 2, 2, 2]])
    am = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1]])
    hv = torch.arange(3 * 6 * 2, dtype=torch.float32).view(3, 6, 2)
    E, vb, cnt = R["pack_image_bank"](hv, ids, am, IMG)
    assert cnt.tolist() == [3, 2, 1] and E.shape == (3, 3, 2)
    assert torch.equal(E[0], hv[0, [1, 2, 4]]) and torch.equal(
        E[1, :2], hv[1, [2, 3]]) and torch.equal(E[2, :1], hv[2, [0]])
    assert vb.tolist() == [[True] * 3, [True, True, False],
                           [True, False, False]]
    assert float(E[1, 2].abs().sum()) == 0 and float(E[2, 1:].abs().sum()) == 0
    ids_pad = ids.clone()
    ids_pad[1, :2] = IMG                          # image-id в padding — не банк
    E2, vb2, cnt2 = R["pack_image_bank"](hv, ids_pad, am, IMG)
    assert cnt2.tolist() == [3, 2, 1] and torch.equal(E2, E)
    try:
        R["pack_image_bank"](hv, torch.zeros_like(ids), am, IMG)
        raise AssertionError("наблюдение без image-токенов принято")
    except SystemExit:
        pass
    # 2. router: нет смешения элементов батча (разный N_b, перестановка
    # соседей, лишний padding) в float32
    Hh, Dv = 24, 20
    rt = R["VisualRouter"](Hh, Dv, torch.Generator().manual_seed(1))
    with torch.no_grad():
        rt.Wo.weight.normal_(0, 0.3, generator=g)
    Eb = torch.randn(3, 11, Dv, generator=g)
    vb = torch.zeros(3, 11, dtype=torch.bool)
    for b, n in enumerate((5, 3, 11)):
        vb[b, :n] = True
    ha = torch.randn(3, 28, Hh, generator=g)
    K, Vv = rt.project_bank(Eb, vb)
    out, info = rt(ha, K, Vv, vb, sizes, True, collect=True)
    K1, V1 = rt.project_bank(Eb[:1, :5], vb[:1, :5])
    o1, _i = rt(ha[:1], K1, V1, vb[:1, :5], sizes, True)
    assert torch.allclose(out[0], o1[0], atol=1e-5)
    perm = [2, 0, 1]
    K3, V3 = rt.project_bank(Eb[perm], vb[perm])
    o3, _i = rt(ha[perm], K3, V3, vb[perm], sizes, True)
    assert torch.allclose(o3[1], out[0], atol=1e-5)
    Ep = torch.cat([Eb, torch.randn(3, 4, Dv, generator=g)], 1)
    vp = torch.cat([vb, torch.zeros(3, 4, dtype=torch.bool)], 1)
    Kp, Vp = rt.project_bank(Ep, vp)
    op, _i = rt(ha, Kp, Vp, vp, sizes, True)
    assert torch.allclose(op, out, atol=1e-5)
    fi = info_floats(info)
    assert fi["pad_mass"] == 0.0 and fi["sum_err"] <= SUM_TOL and \
        fi["finite"] and fi["n_tokens"] == [5, 3, 11]
    assert fi["levels"][0]["kl_to_prior"] is None and \
        fi["levels"][2]["kl_to_prior"] >= 0
    # нулевой Wo -> выход равен входу побитово; градиенты: Wo ≠ 0,
    # Wq/Wk/Wv = 0 на первом шаге, после изменения Wo — ненулевые
    rz = R["VisualRouter"](Hh, Dv, torch.Generator().manual_seed(1))
    assert torch.equal(rz.Wq.weight, rt.Wq.weight)
    Kz, Vz = rz.project_bank(Eb, vb)
    oz, _i = rz(ha, Kz, Vz, vb, sizes, True)
    assert torch.equal(oz, ha)
    oz.pow(2).sum().backward()
    assert float(rz.Wo.weight.grad.abs().sum()) > 0
    assert all(float(m.weight.grad.abs().sum()) == 0 for m in
               (rz.Wq, rz.Wk, rz.Wv))
    with torch.no_grad():
        rz.Wo.weight.add_(rz.Wo.weight.grad.sign() * -1e-2)
    rz.zero_grad()
    Kz, Vz = rz.project_bank(Eb, vb)
    oz, _i = rz(ha, Kz, Vz, vb, sizes, True)
    oz.pow(2).sum().backward()
    assert all(float(m.weight.grad.abs().sum()) > 0 for m in
               (rz.Wq, rz.Wk, rz.Wv, rz.Wo))
    assert n_new_params(768, 2048) == 720896
    assert sum(p.numel() for p in rt.parameters()) == n_new_params(Hh, Dv)
    print("самопроверка k26_attention пройдена: формы, сумма 1 (5e-6), нули "
          "на padding, нулевые scores -> равномерно, epsilon 1 = независимое "
          "внимание, влияние родителя на карту ребёнка (корень -> средний -> "
          "лист) только при epsilon 0.1, индексы родителей, большие scores и "
          "один токен, отказ на пустой строке, gradcheck float64, градиент "
          "предков без detach, банк по image-позициям, нет смешения батча, "
          "нулевой Wo = вход побитово, градиенты Q/K/V после изменения Wo")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
