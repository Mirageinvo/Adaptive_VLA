#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-35: компактный проход 4 -> 8 -> 12 с выбором ветвей и смешанной сборкой.

Источник — K-34 T0 (parents_attn24 с головами readout8/16/24). Класс
K35 наследует K34Parents: веса, router, Expand, геометрия и головы — те же.
source_forward(...) — базовый K-34 forward (глубина 24) на этих же весах.

forward(batch, po, state, cuts=None, controller=None):
  слои 1–8:  4 корня (слоты 1, 5, 9, 13), router после 4 и 8;
  решение a (cuts // 6 или controller.first): два корня PAIRS[a]
             разделяются; создаются только их 4 средних (exp1), остальные
             корни — конечные (снимок после слоя 8, readout8);
  слои 9–16: 4 корня + 4 средних (слоты 2m КАЖДОЙ строки), router с явными
             родителями (средний -> его корень), уровни (4, 4);
  решение b (cuts % 6 или controller.second): две локальные позиции
             PAIRS[b] среди 4 средних разделяются; создаются только 4 листа
             (exp2), остальные средние — конечные (снимок после слоя 16,
             readout16);
  слои 17–24: 4 корня + 4 средних + 4 листа (слоты l), router (4, 4, 4),
             лист -> его средний; листья — readout24 (readout).
Веса конечных узлов: корневые W0, деление родителя W·(1−g), W·g
(сохранение массы) — сумма по 8 конечным узлам = 1 в каждой точке.
Сборка: сырые y трёх уровней (каждая голова — с весами своих конечных
узлов, без перенормировки) приводятся к float32 и суммируются вне
autocast; затем один раз arm = mid + half·tanh(y/half), grip = y[..., 6].
Невыбранные потомки не создаются (нет hidden, Linear, attention).
Счётчики: active [4]*8 + [8]*8 + [12]*8, router [4,4,8,8,12,12], vision 1,
K/V 1, expand 2, heads [8, 16, 24].
"""
import math
import sys

import numpy as np

import k21_model as km
import k21_tree as kt
import k26_attention as ka
import k34_model as k34m
import k35_topology as tp

HEADS = {"root": "readout8", "mid": "readout16", "leaf": "readout"}


def build(torch):
    import torch.utils.checkpoint as tuc
    M34 = k34m.build(torch)
    R = ka.build(torch)
    F = torch.nn.functional
    NEG = float("-inf")
    TAB = {k: torch.as_tensor(v) for k, v in tp.tables().items()}
    PAIRS_T = torch.as_tensor(tp.PAIRS, dtype=torch.long)

    def mapped_probs(S, valid, sizes, parents, eps):
        """S [B, h, A, N]; sizes — числа узлов уровней; parents — список
        [B, n_l] (для уровней 1…) индексов узла-родителя в ПРЕДЫДУЩЕМ
        компактном уровне. Корни — равномерный prior; ребёнок — смесь карты
        своего фактического родителя из того же вызова и равномерного."""
        R_check = R["log_uniform"]
        if valid.dtype != torch.bool or not bool(valid.any(-1).all()):
            raise SystemExit("визуальный банк: строка без допустимых токенов")
        dt = torch.promote_types(torch.float32, S.dtype)
        S = S.to(dt)
        B, h, A, N = S.shape
        if sum(sizes) != A:
            raise SystemExit(f"узлов {A}, уровни {sizes}")
        if len(parents) != len(sizes) - 1:
            raise SystemExit("родители заданы не для всех уровней")
        v4 = valid[:, None, None, :]
        lu = R_check(valid, dt)
        outs, priors, off, prev = [], [], 0, None
        for lev, n in enumerate(sizes):
            s = S[:, :, off:off + n]
            if lev == 0:
                lpr = lu.expand(B, h, n, N)
            else:
                par = parents[lev - 1]
                if tuple(par.shape) != (B, n):
                    raise SystemExit(f"родители уровня {lev}: "
                                     f"{tuple(par.shape)} ≠ {(B, n)}")
                if int(par.min()) < 0 or int(par.max()) >= sizes[lev - 1]:
                    raise SystemExit("индекс родителя вне предыдущего уровня")
                lpp = torch.where(v4, prev, torch.zeros((), dtype=dt,
                                                        device=S.device))
                idx = par.to(S.device)[:, None, :, None].expand(B, h, n, N)
                lpp = torch.gather(lpp, 2, idx)
                lpr = torch.logaddexp(lpp + math.log1p(-eps),
                                      lu + math.log(eps))
            x = torch.where(v4, s + lpr, torch.full((), NEG, dtype=dt,
                                                    device=S.device))
            lp = torch.log_softmax(x, -1)
            outs.append(lp)
            priors.append(lpr)
            prev = lp
            off += n
        lp = torch.cat(outs, 2)
        return dict(P=lp.exp(), logP=lp, log_prior=torch.cat(priors, 2))

    def router_mapped(router, ha, K, Vimg, valid, sizes, parents,
                      collect=False, keep_maps=False, capture=None):
        """Тот же VisualRouter (Wq/Wk/Wv/Wo источника), карты — с явными
        родителями. capture (список) получает сохранённые входы карт для
        сверки с медленным эталоном."""
        dev = ha.device.type
        with torch.autocast(dev, enabled=False):
            hn = F.layer_norm(ha.float(), (router.H,), eps=ka.LN_EPS)
        Q = router._heads(router.Wq(hn))
        with torch.autocast(dev, enabled=False):
            S = torch.matmul(Q.float(), K.float().transpose(-1, -2)) / \
                math.sqrt(router.dh)
            tpr = mapped_probs(S, valid, sizes, parents, router.eps)
            if capture is not None:
                capture.append(dict(
                    S=S.detach().float().cpu(), valid=valid.detach().cpu(),
                    sizes=tuple(sizes), parents=[x.detach().cpu()
                                                 for x in parents],
                    P=tpr["P"].detach().float().cpu()))
            C = R["weighted_context"](tpr["P"], Vimg)
        upd = router.gamma * router.Wo(C).to(ha.dtype)
        info = None
        if collect or keep_maps:
            info = R["summaries"](tpr, valid, sizes, ha, upd)
        return ha + upd, info

    def gather_nodes(x, idx):
        """x [B, n, ...], idx [B, k] -> [B, k, ...]."""
        shape = list(idx.shape) + [1] * (x.dim() - 2)
        ix = idx.to(x.device).view(shape).expand(*idx.shape, *x.shape[2:])
        return torch.gather(x, 1, ix)

    def gather_w(w, idx):
        """w [B, 20, n], idx [B, k] -> [B, 20, k]."""
        ix = idx.to(w.device)[:, None, :].expand(w.shape[0], w.shape[1],
                                                 idx.shape[1])
        return torch.gather(w, 2, ix)

    class K35(M34["K34Parents"]):
        """Компактное дерево фиксированного бюджета поверх K-34 T0."""

        def source_forward(self, batch, po, state):
            return M34["K34Parents"].forward(self, batch, po, state,
                                             exit_depth=24)

        def _pos_rows(self, B, V, slots, po, dev):
            po = torch.as_tensor(po, device=dev, dtype=torch.long).reshape(-1)
            if po.numel() == 1:
                po = po.expand(B)
            sl = slots.to(dev).long()
            ap = V + po[:, None] + (po[:, None] + 1) * sl
            vp = torch.arange(V, device=dev, dtype=torch.long)[None].expand(
                B, -1)
            return torch.cat([vp, ap], 1)

        def _partial(self, level, h, iv, w, cnt):
            """Сырой y уровня: голова уровня с весами ЕГО конечных узлов."""
            cnt["heads"].append(level)
            norm = self.bar.action_expert.norm
            out = getattr(self, HEADS[level])(norm(h), iv, w)
            return out["y"]

        def forward(self, batch, po, state, cuts=None, controller=None):
            if not getattr(self, "exits_attached", False):
                raise SystemExit("K-35 требует головы K-34 (attach_exits)")
            if cuts is None and controller is None:
                controller = getattr(self, "controller", None)  # рука K-9h
            if (cuts is None) == (controller is None):
                raise SystemExit("нужен ровно один из cuts / controller")
            bar = self.bar
            dev = batch["input_ids"].device
            on = self.router_mode == "on"
            if not on:
                raise SystemExit("K-35 — только с router (parents_attn24)")
            cnt = dict(vision=0, text=[0] * 24, expert=[0] * 24,
                       active=[0] * 24, router=[], kv=0, expand=0, heads=[],
                       decisions=0)
            with torch.no_grad():
                img = bar.vlm.get_image_features(
                    batch["pixel_values"], batch.get("pixel_attention_mask"))
            cnt["vision"] += 1
            _b, _l, hv, _i = bar._build_vlm_inputs_embeds(
                input_ids=batch["input_ids"], inputs_embeds=None,
                pixel_values=None, pixel_attention_mask=None,
                image_hidden_states=img)
            B, V = hv.shape[0], hv.shape[1]
            bank = self._bank(batch, hv)
            cnt["kv"] += 1
            st = torch.as_tensor(state, device=dev, dtype=torch.float32)
            st = st.clone()
            ix = [0, 1, 2, 9, 10]
            st[:, ix] = (st[:, ix] - self.state_mean) / self.state_std
            ha = (self.q0 + self.slot).expand(B, -1, -1) + \
                self.state_proj(st)[:, None]
            if cuts is not None:
                cuts_t = torch.as_tensor(cuts, dtype=torch.long).reshape(-1)
                if cuts_t.numel() == 1:
                    cuts_t = cuts_t.expand(B)
                if cuts_t.numel() != B or int(cuts_t.min()) < 0 or \
                        int(cuts_t.max()) >= tp.N_CUTS:
                    raise SystemExit("срезы: число или диапазон")
            infos, snap = [], {}

            def run_stage(l0, l1, slots_rows, sizes, parents):
                nonlocal hv, ha
                A = ha.shape[1]
                mask = self._mask(batch["attention_mask"], V, A, dev)
                pos = self._pos_rows(B, V, slots_rows, po, dev)
                cos, sin = bar.vlm.text_model.rotary_emb(
                    hv[:, :1].expand(B, V + A, hv.shape[-1]),
                    position_ids=pos)
                for l in range(l0, l1):
                    if self.checkpoint and self.training:
                        hv, ha = tuc.checkpoint(self._layer, l, hv, ha, mask,
                                                cos, sin, use_reentrant=False)
                    else:
                        hv, ha = self._layer(l, hv, ha, mask, cos, sin)
                    cnt["text"][l] += 1
                    cnt["expert"][l] += 1
                    cnt["active"][l] = A
                    if l in ka.AFTER:
                        ha, inf = router_mapped(
                            self.router, ha, bank["K"], bank["V"],
                            bank["valid"], sizes, parents,
                            collect=self.collect, keep_maps=False,
                            capture=getattr(self, "router_capture", None))
                        cnt["router"].append(int(ha.shape[1]))
                        infos.append(inf)

            # --- стадия 1: корни -------------------------------------------------
            root_slots = torch.as_tensor(tp.ROOT_SLOTS)[None].expand(B, -1)
            run_stage(0, 8, root_slots, (4,), [])
            roots = ha
            norm = bar.action_expert.norm
            iv = self.T["root"](self.dur(roots), False)            # [B, 4, 2]
            W0 = self.T["root_weights"](iv)                        # [B, 20, 4]
            v1 = self.split1(roots)                                # [B, 4]
            roots_n = norm(roots)
            if cuts is not None:
                a = cuts_t // 6
            else:
                a = controller.first(dict(roots_n=roots_n, iv=iv, B=B))
                cnt["decisions"] += 1
            a = torch.as_tensor(a, dtype=torch.long).reshape(-1).cpu()
            if a.numel() != B or int(a.min()) < 0 or int(a.max()) > 5:
                raise SystemExit("решение a вне 0…5")
            sel_r = TAB["split_roots"][a]                          # [B, 2]
            term_r = TAB["term_roots"][a]
            iv_sel = gather_nodes(iv, sel_r)
            iv_mid, s1 = self.T["split"](iv_sel, gather_nodes(
                v1[..., None], sel_r)[..., 0], False)              # [B, 4, 2]
            W_mid = self.T["split_weights"](gather_w(W0, sel_r), iv_sel, s1)
            y_root = self._partial("root", gather_nodes(roots, term_r),
                                   gather_nodes(iv, term_r),
                                   gather_w(W0, term_r), cnt)
            snap.update(roots_n=roots_n, iv_root=iv, W0=W0)
            h_mid = self.exp1(gather_nodes(roots, sel_r), iv_mid)
            cnt["expand"] += 1
            ha = torch.cat([roots, h_mid], 1)
            # --- стадия 2: корни + 4 средних -----------------------------------
            par_mid = TAB["mid_parent_local"][a]                   # [B, 4]
            run_stage(8, 16, TAB["slots2"][a], (4, 4), [par_mid])
            mids = ha[:, 4:8]
            v2 = self.split2(mids)                                 # [B, 4]
            mids_n = norm(mids)
            if cuts is not None:
                b = cuts_t % 6
            else:
                b = controller.second(dict(roots_n=roots_n, iv=iv,
                                           mids_n=mids_n, iv_mid=iv_mid,
                                           mids=TAB["mids"][a], a=a, B=B))
                cnt["decisions"] += 1
            b = torch.as_tensor(b, dtype=torch.long).reshape(-1).cpu()
            if b.numel() != B or int(b.min()) < 0 or int(b.max()) > 5:
                raise SystemExit("решение b вне 0…5")
            c = 6 * a + b
            sel_m = PAIRS_T[b]                                     # [B, 2]
            term_m = TAB["term_mid_local"][c]
            iv_msel = gather_nodes(iv_mid, sel_m)
            iv_leaf, s2 = self.T["split"](iv_msel, gather_nodes(
                v2[..., None], sel_m)[..., 0], False)
            W_leaf = self.T["split_weights"](gather_w(W_mid, sel_m), iv_msel,
                                             s2)
            y_mid = self._partial("mid", gather_nodes(mids, term_m),
                                  gather_nodes(iv_mid, term_m),
                                  gather_w(W_mid, term_m), cnt)
            snap.update(mids_n=mids_n, iv_mid=iv_mid, W_mid=W_mid)
            h_leaf = self.exp2(gather_nodes(mids, sel_m), iv_leaf)
            cnt["expand"] += 1
            ha = torch.cat([ha, h_leaf], 1)
            # --- стадия 3: корни + средние + 4 листа ---------------------------
            par_leaf = TAB["leaf_parent_local"][c]
            run_stage(16, 24, TAB["slots3"][c], (4, 4, 4),
                      [par_mid, par_leaf])
            leaves = ha[:, 8:12]
            y_leaf = self._partial("leaf", leaves, iv_leaf, W_leaf, cnt)
            # --- смешанная сборка (float32, вне autocast) ----------------------
            with torch.autocast(dev.type, enabled=False):
                yr, ym, yl = y_root.float(), y_mid.float(), y_leaf.float()
                y = yr + ym + yl
                Lb, Ub = self.readout.L.float(), self.readout.U.float()
                mid, half = (Ub + Lb) / 2, (Ub - Lb) / 2
                arm = mid + half * torch.tanh(y[..., :6] / half)
                w_sum = gather_w(W0, term_r).sum(-1) + gather_w(
                    W_mid, term_m).sum(-1) + W_leaf.sum(-1)
            out = dict(arm=arm, grip_logit=y[..., 6], y=y,
                       partial=dict(root=yr, mid=ym, leaf=yl), c=c, a=a, b=b,
                       w_sum=w_sum, iv_leaf=iv_leaf, W_leaf=W_leaf,
                       term_roots=term_r, term_mid_local=term_m,
                       snap=snap)
            out["bank"] = dict(counts=bank["counts"],
                               n_max=int(bank["E"].shape[1]),
                               dv=int(bank["E"].shape[2]))
            if self.collect:
                out["router"] = infos
            self.counters = cnt
            return out

    def make(bar, L, U, state_mean=None, state_std=None, **kw):
        m = K35(bar, L, U, state_mean, state_std, **kw)
        return m

    return dict(K35=K35, make=make, mapped_probs=mapped_probs,
                router_mapped=router_mapped, M34=M34, R=R,
                gather_nodes=gather_nodes, gather_w=gather_w)


def groups(mod):
    """Непересекающиеся группы: expert, lora, new (без голов), heads (все
    три readout)."""
    hid = {id(p) for nm in ("readout8", "readout16", "readout")
           for p in getattr(mod, nm).parameters()}
    g = mod.groups()
    return dict(expert=g["expert"], lora=g["lora"],
                new=[p for p in g["new"] if id(p) not in hid],
                heads=[p for p in g["new"] if id(p) in hid])


def counters_problems(c, decisions=None):
    if c is None:
        return ["счётчики не записаны"]
    want = dict(vision=1, kv=1, text=[1] * 24, expert=[1] * 24,
                active=tp.ACTIVE, router=tp.ROUTER_ACTIVE, expand=2,
                heads=["root", "mid", "leaf"])
    if decisions is not None:
        want["decisions"] = decisions
    return [f"счётчик {k}: {c.get(k)} ≠ {v}" for k, v in want.items()
            if c.get(k) != v]


def assemble_np(partials, L, U):
    """Независимая сборка из сохранённых partial y (float32)."""
    y = sum(np.asarray(partials[k], np.float32) for k in ("root", "mid",
                                                          "leaf"))
    L = np.asarray(L, np.float32)
    U = np.asarray(U, np.float32)
    mid, half = (U + L) / 2, (U - L) / 2
    return y, mid + half * np.tanh(y[..., :6] / half), y[..., 6]


# --- самопроверка ------------------------------------------------------------------
def selftest():
    import copy
    import torch
    import k21_train as ktr
    import k26_model as k26m
    M = build(torch)
    R = M["R"]
    g = torch.Generator().manual_seed(0)
    # 8. mapped-router на регулярном полном дереве = исходная формула
    B, h, N = 3, 4, 9
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[1, 6:] = False
    S = torch.randn(B, h, 28, N, generator=g) * 2
    par1 = (torch.arange(8) // 2)[None].expand(B, -1)
    par2 = (torch.arange(16) // 2)[None].expand(B, -1)
    ref = R["tree_probs"](S, valid, (4, 8, 16), ka.EPS, True)
    mp = M["mapped_probs"](S, valid, (4, 8, 16), [par1, par2], ka.EPS)
    assert torch.allclose(ref["P"], mp["P"], atol=1e-7), \
        float((ref["P"] - mp["P"]).abs().max())
    # сокращённое дерево против медленного расчёта по явным родителям
    S12 = torch.randn(B, h, 12, N, generator=g) * 2
    pm = torch.tensor([[2, 2, 3, 3], [0, 0, 1, 1], [1, 1, 3, 3]])
    pl = torch.tensor([[2, 2, 3, 3], [0, 0, 3, 3], [1, 1, 2, 2]])
    mp = M["mapped_probs"](S12, valid, (4, 4, 4), [pm, pl], ka.EPS)

    def slow(Sb, vb, pmb, plb):
        n = int(vb.sum())
        out = torch.zeros(h, 12, N, dtype=torch.float64)
        lu = -math.log(n)
        for head in range(h):
            maps = []
            for j in range(12):
                lev = 0 if j < 4 else (1 if j < 8 else 2)
                s = Sb[head, j].double()
                if lev == 0:
                    pr = torch.full((N,), lu, dtype=torch.float64)
                else:
                    pidx = (pmb[j - 4] if lev == 1 else 4 + plb[j - 8])
                    pp = maps[pidx]
                    pr = torch.log((1 - ka.EPS) * pp + ka.EPS * math.exp(lu))
                x = torch.where(vb, s + pr, torch.tensor(-math.inf,
                                                         dtype=torch.float64))
                p = torch.softmax(x, -1)
                maps.append(p)
                out[head, j] = p
        return out
    for bb in range(B):
        sref = slow(S12[bb], valid[bb], pm[bb], pl[bb])
        assert torch.allclose(mp["P"][bb].double(), sref, atol=5e-6), bb
    assert float(mp["P"][1, :, :, 6:].abs().max()) == 0.0
    # 9. изменение scores другого (не родительского) корня не меняет prior
    bump = torch.randn(h, N, generator=g) * 3   # разное по токенам
    S2 = S12.clone()
    S2[0, :, 0] += bump                    # корень 0 — не родитель строки 0
    mp2 = M["mapped_probs"](S2, valid, (4, 4, 4), [pm, pl], ka.EPS)
    assert torch.allclose(mp2["log_prior"][0, :, 4:], mp["log_prior"][0, :,
                                                                      4:])
    S3 = S12.clone()
    S3[0, :, 2] += bump                    # корень 2 — родитель средних 0, 1
    mp3 = M["mapped_probs"](S3, valid, (4, 4, 4), [pm, pl], ka.EPS)
    assert not torch.allclose(mp3["log_prior"][0, :, 4:6],
                              mp["log_prior"][0, :, 4:6])
    # --- модель на игрушечном BAR ---------------------------------------------------
    bar0 = k26m.fake_bar(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    bt = k26m.fake_batch(torch, B=3, pad=2)
    st = torch.randn(3, 11)
    k34 = M["M34"]["make"]("parents_attn24", copy.deepcopy(bar0), L, U,
                           seed=0)
    with torch.no_grad():
        k34.router.Wo.weight.normal_(0, 0.3, generator=torch.Generator()
                                     .manual_seed(3))
        for n, p in k34.named_parameters():
            if p.requires_grad and not n.startswith("router."):
                p.add_(torch.randn(p.shape, generator=torch.Generator()
                                   .manual_seed(p.numel())) * 0.02)
    k34.attach_exits()
    with torch.no_grad():                   # головы 8/16/24 различаются
        for p in k34.readout8.parameters():
            p.add_(0.05)
        for p in k34.readout16.parameters():
            p.sub_(0.05)
    m = M["make"](copy.deepcopy(bar0), L, U, seed=0)
    m.attach_exits()
    own = dict(m.named_parameters())
    own.update(dict(m.named_buffers()))
    with torch.no_grad():
        for k, v in ktr.trainable_state(k34).items():
            own[k].copy_(v)
    k34.eval()
    m.eval()
    # source_forward на общих весах = K-34 побитово; порча общего тензора
    # ломает сверку
    with torch.no_grad():
        a34 = k34(bt, 1, st, exit_depth=24)["arm"]
        a35 = m.source_forward(bt, 1, st)["arm"]
    assert torch.equal(a34, a35)
    bad = copy.deepcopy(m)
    with torch.no_grad():
        bad.readout.mlp[0].weight.add_(0.01)
        assert not torch.equal(bad.source_forward(bt, 1, st)["arm"], a34)
    # все 36 срезов: формы, конечность, веса, счётчики, детерминизм
    outs = {}
    with torch.no_grad():
        for c in range(36):
            o = m(bt, 1, st, cuts=[c] * 3)
            assert counters_problems(m.counters) == [], (c, m.counters)
            assert o["arm"].shape == (3, 20, 6) and torch.isfinite(
                o["arm"]).all() and torch.isfinite(o["y"]).all()
            # 4. сумма весов 8 конечных узлов = 1 (5e-6)
            assert float((o["w_sum"] - 1).abs().max()) <= 5e-6, c
            outs[c] = o
        # разные c внутри батча = те же строки при однородных c
        mix = [0, 17, 35]
        om = m(bt, 1, st, cuts=mix)
        for i, c in enumerate(mix):
            assert torch.allclose(om["arm"][i], outs[c]["arm"][i],
                                  atol=1e-5), (i, c)
    assert len({tuple(outs[c]["arm"][0].flatten()[:5].tolist())
                for c in range(36)}) > 30
    # 10. сборка = независимой; сумма уже ограниченных arm — провал
    o = outs[14]
    y, arm, grip = assemble_np({k: v.numpy() for k, v in o["partial"].items()},
                               L, U)
    assert np.allclose(y, o["y"].numpy(), atol=1e-6) and np.allclose(
        arm, o["arm"].numpy(), atol=1e-6) and np.array_equal(
        grip, o["grip_logit"].numpy())
    Lf, Uf = np.float32(L), np.float32(U)
    midh, halfh = (Uf + Lf) / 2, (Uf - Lf) / 2
    wrong = sum(midh + halfh * np.tanh(o["partial"][k].numpy()[..., :6] /
                                       halfh) for k in ("root", "mid", "leaf"))
    assert not np.allclose(wrong, o["arm"].numpy(), atol=1e-3)
    # 11. поздние слои не меняют вклад конечного корня; слой 20 — вклад
    # конечного среднего; слой 12 меняет средний, но не корень
    late = copy.deepcopy(m)
    with torch.no_grad():
        for p in late.bar.action_expert.layers[20].parameters():
            p.add_(0.3)
        for p in late.exp2.parameters():
            p.add_(0.3)
        ol = late(bt, 1, st, cuts=[14] * 3)
    assert torch.equal(ol["partial"]["root"], o["partial"]["root"])
    assert torch.equal(ol["partial"]["mid"], o["partial"]["mid"])
    assert not torch.equal(ol["partial"]["leaf"], o["partial"]["leaf"])
    midl = copy.deepcopy(m)
    with torch.no_grad():
        for p in midl.bar.action_expert.layers[12].parameters():
            p.add_(0.3)
        om2 = midl(bt, 1, st, cuts=[14] * 3)
    assert torch.equal(om2["partial"]["root"], o["partial"]["root"])
    assert not torch.equal(om2["partial"]["mid"], o["partial"]["mid"])
    # 7. невыбранные потомки не создаются: exp1/exp2 получают по 2 родителя
    calls = []
    for nm in ("exp1", "exp2"):
        f0 = getattr(m, nm).forward

        def rec(hh, ivc, _f=f0, _nm=nm):
            calls.append((_nm, tuple(hh.shape), tuple(ivc.shape)))
            return _f(hh, ivc)
        getattr(m, nm).forward = rec
    with torch.no_grad():
        m(bt, 1, st, cuts=[5, 20, 33])
    assert calls == [("exp1", (3, 2, m.H), (3, 4, 2)),
                     ("exp2", (3, 2, m.H), (3, 4, 2))], calls
    for nm in ("exp1", "exp2"):
        del getattr(m, nm).forward
    # группы: головы отдельно и без пересечений
    gr = groups(m)
    ids = [id(p) for k in gr for p in gr[k]]
    assert len(ids) == len(set(ids))
    assert any(p is m.readout.mlp[0].weight for p in gr["heads"])
    # градиент смешанного выхода проходит в эксперт и все три головы
    m.train()
    m.checkpoint = True
    o = m(bt, 1, st, cuts=[14, 20, 7])
    km.loss_fn(torch, o, torch.rand(3, 20, 7))[0].mean().backward()
    for nm in ("readout8", "readout16", "readout"):
        assert float(getattr(m, nm).mlp[0].weight.grad.abs().sum()) > 0, nm
    assert float(m.bar.action_expert.layers[0].self_attn.q_proj.weight.grad
                 .abs().sum()) > 0
    print("самопроверка k35_model пройдена: mapped-router = исходной формуле "
          "на полном дереве и медленному расчёту на сокращённом (padding), "
          "prior не переадресуется чужим корнем; source_forward = K-34 "
          "побитово (порча ломает); 36 срезов — формы, конечность, сумма "
          "весов 8 конечных узлов = 1, счётчики 4/8/12 и router 4/4/8/8/12/12; "
          "смешанный батч = однородным; сборка из partial y = независимой, "
          "сумма ограниченных arm — нет; поздние слои не меняют вклад "
          "конечных корня/среднего; создаются только выбранные потомки; "
          "группы без пересечений; градиенты в эксперт и три головы")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
