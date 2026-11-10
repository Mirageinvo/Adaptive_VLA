#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-34: исполнимые выходы дерева после слоёв 8, 16 и 24 (один checkpoint).

Поверх k26_model (parents_attn24 / flat_attn24; router, маски, RoPE, LoRA,
состояние, нормировки — без изменений) добавлены две головы readout8 и
readout16 — независимые глубокие копии исходного readout (readout24),
создаваемые ПОСЛЕ загрузки весов источника (attach_exits): deepcopy не
трогает RNG и не делит storage.

Выход глубины d снимается после router соответствующего слоя (индексы 7 и
15, router вызывается после слоёв 4, 8, 12, 16, 20, 24):
  T, d=8:  корни ha[:, :4]; I0 = root(dur(roots)), W0 = root_weights(I0);
           split1, Expand1, слои 9–24, split2, Expand2 не вызываются;
  T, d=16: средние ha[:, 4:12] с I1/W1 (после split1); слои 17–24,
           split2, Expand2 не вызываются;
  T, d=24: прежнее полное дерево, 16 листьев (readout24 = исходный);
  F:       все 16 query, прежние фиксированные интервалы/веса flat; backbone
           реально завершается на глубине d.
x_d = action_expert.norm(h_d) ровно один раз; голова — k21_tree.Readout
(20×7: рука mid + half·tanh(y/half), логит схвата).
forward(..., exit_depth=d) — один выбранный выход с настоящим усечением и
одной головой; forward_all(...) — один полный проход, три выхода (обучение,
диагностика); exit_depth=24 без attach — точный путь K-26 (legacy/source).
Счётчики: exit_depth, vision, text/expert по слоям, active, router, kv,
expand, heads.
"""
import copy
import sys

import numpy as np

import k21_model as km
import k21_tree as kt
import k22_model as k22
import k26_attention as ka
import k26_model as k26m

MODES = k26m.MODES
EXITS = (8, 16, 24)
EXIT_LAYER = {8: 7, 16: 15}                # индекс слоя, после router которого
HEADS = {8: "readout8", 16: "readout16", 24: "readout"}
ARCH = dict(parents_attn24="k34_parents_attn24_exits",
            flat_attn24="k34_flat_attn24_exits")
PHI, HIDDEN = 9, 128


def head_params(H, hidden=HIDDEN, phi=PHI):
    """Параметры одной k21_tree.Readout: LayerNorm(H) + MLP
    (H+2φ -> hidden -> hidden -> 7)."""
    return 2 * H + (H + 2 * phi) * hidden + hidden + hidden * hidden + \
        hidden + hidden * 7 + 7


def n_new_params(H):
    return 2 * head_params(H)


def expected_counters(mode, d, all_exits=False):
    """Ожидаемые счётчики выбранной глубины (или полного прохода)."""
    stop = 24 if all_exits else d
    act = k26m.ACTIVE[mode]
    n_router = sum(1 for x in ka.AFTER if x < stop)
    return dict(
        vision=1, text=[1] * stop + [0] * (24 - stop),
        expert=[1] * stop + [0] * (24 - stop),
        active=act[:stop] + [0] * (24 - stop),
        router=k26m.ROUTER_ACTIVE[mode][:n_router], kv=1,
        expand=(0 if mode == "flat_attn24" else
                {8: 0, 16: 1, 24: 2}[stop]),
        heads=list(EXITS) if all_exits else [d],
        exit_depth="all" if all_exits else d)


def counters_problems(mode, c, d, all_exits=False):
    if c is None:
        return ["счётчики не записаны"]
    want = expected_counters(mode, d, all_exits)
    return [f"счётчик {k}: {c.get(k)} ≠ {v}" for k, v in want.items()
            if c.get(k) != v]


def build(torch):
    import torch.utils.checkpoint as tuc
    M = k26m.build(torch)

    class Exits:
        """Выходы 8/16/24 поверх Attn.forward K-26 (тот же порядок операций
        для глубины 24)."""

        def attach_exits(self):
            """Головы 8/16 — глубокие копии readout ПОСЛЕ загрузки весов
            источника; без общего storage; RNG не используется."""
            if getattr(self, "exits_attached", False):
                raise SystemExit("головы K-34 уже созданы")
            self.readout8 = copy.deepcopy(self.readout)
            self.readout16 = copy.deepcopy(self.readout)
            self.exits_attached = True
            self.exit_depth = 24

        def heads_params(self):
            return [p for nm in ("readout8", "readout16")
                    for p in getattr(self, nm).parameters()]

        def forward_all(self, batch, po, state):
            return self.forward(batch, po, state, all_exits=True)

        def _head(self, d, h, iv, w, cnt):
            cnt["heads"].append(d)
            norm = self.bar.action_expert.norm
            return getattr(self, HEADS[d])(norm(h), iv, w)

        def forward(self, batch, po, state, exit_depth=None, all_exits=False):
            d = int(getattr(self, "exit_depth", 24) if exit_depth is None
                    else exit_depth)
            if d not in EXITS:
                raise SystemExit(f"глубина выхода {d}")
            if (all_exits or d != 24) and \
                    not getattr(self, "exits_attached", False):
                raise SystemExit("ранние выходы без attach_exits")
            need = set(EXITS) if all_exits else {d}
            stop = 24 if all_exits else d
            bar = self.bar
            dev = batch["input_ids"].device
            hier = self.hierarchical
            on = self.router_mode == "on"
            cnt = dict(vision=0, text=[0] * 24, expert=[0] * 24,
                       active=[0] * 24, router=[], kv=0, expand=0, heads=[],
                       exit_depth="all" if all_exits else d)
            with torch.no_grad():
                img = bar.vlm.get_image_features(
                    batch["pixel_values"], batch.get("pixel_attention_mask"))
            cnt["vision"] += 1
            _b, _l, hv, _i = bar._build_vlm_inputs_embeds(
                input_ids=batch["input_ids"], inputs_embeds=None,
                pixel_values=None, pixel_attention_mask=None,
                image_hidden_states=img)
            B, V = hv.shape[0], hv.shape[1]
            bank = None
            if on:
                bank = self._bank(batch, hv)
                cnt["kv"] += 1
            st = torch.as_tensor(state, device=dev, dtype=torch.float32)
            st = st.clone()
            idx = [0, 1, 2, 9, 10]
            st[:, idx] = (st[:, idx] - self.state_mean) / self.state_std
            ha = (self.q0 + self.slot).expand(B, -1, -1) + \
                self.state_proj(st)[:, None]
            stages = k22.STAGES if hier else km.STAGES_FLAT
            aux, infos, exits = {}, [], {}
            flat_iw = None
            if not hier and need & {8, 16}:
                iv_f, w_f = kt.fixed_leaves(torch)
                flat_iw = (iv_f.to(dev).expand(B, -1, -1),
                           w_f.to(dev).expand(B, -1, -1))

            def finish(o):
                if on:
                    o["bank"] = dict(counts=bank["counts"],
                                     n_max=int(bank["E"].shape[1]),
                                     dv=int(bank["E"].shape[2]))
                    if self.collect or self.keep_maps:
                        o["router"] = infos
                o["exit_depth"] = cnt["exit_depth"]
                self.counters = cnt
                return o

            for si, (l0, l1, slots) in enumerate(stages):
                A = ha.shape[1]
                if A != len(slots):
                    raise SystemExit(f"активных query {A}, слотов "
                                     f"{len(slots)}")
                mask = self._mask(batch["attention_mask"], V, A, dev)
                pos = self._pos(B, V, slots, po, dev)
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
                    if on and l in ka.AFTER:
                        sz = SIZES[ha.shape[1]] if hier else (ha.shape[1],)
                        ha, inf = self.router(ha, bank["K"], bank["V"],
                                              bank["valid"], sz, hier,
                                              collect=self.collect,
                                              keep_maps=self.keep_maps)
                        cnt["router"].append(int(ha.shape[1]))
                        infos.append(inf)
                    if not hier and l + 1 in (8, 16):
                        dd = l + 1
                        if dd in need:
                            exits[dd] = dict(self._head(dd, ha, *flat_iw,
                                                        cnt),
                                             nodes=flat_iw[0],
                                             weights=flat_iw[1])
                        if dd == stop:
                            return finish(exits[dd])
                if not hier:
                    continue
                if si == 0:
                    roots = ha
                    iv = self.T["root"](self.dur(roots), False)
                    w = self.T["root_weights"](iv)
                    if 8 in need:
                        exits[8] = dict(self._head(8, roots, iv, w, cnt),
                                        nodes=iv, weights=w)
                    if stop == 8:
                        return finish(exits[8])
                    iv2, s1 = self.T["split"](iv, self.split1(roots), False)
                    w = self.T["split_weights"](w, iv, s1)
                    aux.update(root=iv, split1=s1)
                    ha = torch.cat([roots, self.exp1(roots, iv2)], 1)
                    cnt["expand"] += 1
                elif si == 1:
                    mids = ha[:, 4:12]
                    if 16 in need:
                        exits[16] = dict(self._head(16, mids, iv2, w, cnt),
                                         nodes=iv2, weights=w, **aux)
                    if stop == 16:
                        return finish(exits[16])
                    iv3, s2 = self.T["split"](iv2, self.split2(mids), False)
                    w = self.T["split_weights"](w, iv2, s2)
                    aux.update(mid=iv2, split2=s2)
                    ha = torch.cat([ha, self.exp2(mids, iv3)], 1)
                    cnt["expand"] += 1
            if hier:
                out = self._head(24, ha[:, 12:28], iv3, w, cnt)
                out.update(leaves=iv3, weights=w, hv_last=hv, **aux)
            else:
                iv_cur, w = kt.fixed_leaves(torch)
                iv_cur = iv_cur.to(dev).expand(B, -1, -1)
                w = w.to(dev).expand(B, -1, -1)
                out = self._head(24, ha, iv_cur, w, cnt)
                out.update(leaves=iv_cur, weights=w, hv_last=hv)
            if all_exits:
                exits[24] = out
                out = dict(out, exits=exits)
            return finish(out)

    SIZES = k26m.SIZES

    class K34Parents(Exits, M["K26Parents"]):
        forward = Exits.forward

    class K34Flat(Exits, M["K26Flat"]):
        forward = Exits.forward

    def make(mode, bar, L, U, state_mean=None, state_std=None, **kw):
        if mode == "parents_attn24":
            return K34Parents(bar, L, U, state_mean, state_std, **kw)
        if mode == "flat_attn24":
            return K34Flat(bar, L, U, state_mean, state_std, **kw)
        raise SystemExit(f"K-34 не строит режим {mode!r}")

    return dict(make=make, make_k26=M["make"], K34Parents=K34Parents,
                K34Flat=K34Flat, M26=M)


def groups(mod):
    """Группы обучения K-34: expert, lora, new (группа new K-26 — router,
    readout24, query, геометрия), heads (readout8/readout16)."""
    hid = {id(p) for p in mod.heads_params()}
    g = mod.groups()
    return dict(expert=g["expert"], lora=g["lora"],
                new=[p for p in g["new"] if id(p) not in hid],
                heads=[p for p in g["new"] if id(p) in hid])


# --- самопроверка (CPU, игрушечный BAR) ---------------------------------------------
def selftest():
    import torch
    import k21_train as ktr
    M = build(torch)
    bar0 = k26m.fake_bar(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    bt = k26m.fake_batch(torch, B=3, pad=2)
    st = torch.randn(3, 11)

    def src(mode, s=0):
        """Источник K-26 с «обученными» весами (Wo ≠ 0, шум)."""
        m = M["make_k26"](mode, copy.deepcopy(bar0), L, U, seed=s)
        with torch.no_grad():
            m.router.Wo.weight.normal_(0, 0.3,
                                       generator=torch.Generator()
                                       .manual_seed(3))
            for n, p in m.named_parameters():
                if p.requires_grad and not n.startswith("router."):
                    p.add_(torch.randn(p.shape, generator=torch.Generator()
                                       .manual_seed(p.numel())) * 0.01)
        return m

    def k34_from(m26, mode, s=0):
        m = M["make"](mode, copy.deepcopy(bar0), L, U, seed=s)
        own = dict(m.named_parameters())
        own.update(dict(m.named_buffers()))
        with torch.no_grad():
            for k, v in ktr.trainable_state(m26).items():
                own[k].copy_(v)
        return m

    for mode in MODES:
        m26 = src(mode).eval()
        m = k34_from(m26, mode)
        # до attach: путь источника = K-26 побитово
        m.eval()
        with torch.no_grad():
            a26 = m26(bt, 1, st)
            a24 = m(bt, 1, st)
        assert torch.equal(a26["arm"], a24["arm"]) and torch.equal(
            a26["grip_logit"], a24["grip_logit"]), mode
        try:
            m(bt, 1, st, exit_depth=8)
            raise AssertionError("ранний выход без attach")
        except SystemExit:
            pass
        rng0 = torch.random.get_rng_state()
        m.attach_exits()
        assert torch.equal(rng0, torch.random.get_rng_state())   # 9: RNG
        # 9: без общего storage, до обучения головы = readout
        p24 = dict(m.readout.named_parameters())
        for nm in ("readout8", "readout16"):
            for k, p in getattr(m, nm).named_parameters():
                assert torch.equal(p, p24[k]) and \
                    p.data_ptr() != p24[k].data_ptr(), (nm, k)
            for k, b in getattr(m, nm).named_buffers():
                assert b.data_ptr() != dict(m.readout.named_buffers())[k] \
                    .data_ptr()
        # число новых параметров = формуле
        newp = sum(p.numel() for p in m.heads_params())
        assert newp == n_new_params(m.H), (newp, n_new_params(m.H))
        assert n_new_params(768) == 239374
        m.eval()
        o, cnts = {}, {}
        with torch.no_grad():
            for d in EXITS:
                o[d] = m(bt, 1, st, exit_depth=d)
                cnts[d] = copy.deepcopy(m.counters)
            oa = m.forward_all(bt, 1, st)
            ca = copy.deepcopy(m.counters)
        # глубина 24 после attach = K-26 побитово
        assert torch.equal(o[24]["arm"], a26["arm"])
        # 1, 13: формы и конечность
        for d in EXITS:
            assert o[d]["arm"].shape == (3, 20, 6) and \
                o[d]["grip_logit"].shape == (3, 20)
            assert torch.isfinite(o[d]["arm"]).all() and \
                torch.isfinite(o[d]["grip_logit"]).all()
            assert o[d]["exit_depth"] == d
            # 2, 3, 8: усечение, слои по одному разу, одна голова
            assert counters_problems(mode, cnts[d], d) == [], (mode, d,
                                                               cnts[d])
        assert counters_problems(mode, ca, 24, all_exits=True) == [], ca
        # 5: усечённые выходы = выходам полного прохода (снимок, не финал)
        for d in EXITS:
            assert torch.equal(o[d]["arm"], oa["exits"][d]["arm"]), (mode, d)
            assert torch.equal(o[d]["grip_logit"],
                               oa["exits"][d]["grip_logit"])
        assert not torch.allclose(o[8]["arm"], o[24]["arm"])
        # 6: изменение только позднего слоя/Expand не меняет ранний выход
        late = copy.deepcopy(m)
        with torch.no_grad():
            for p in late.bar.action_expert.layers[20].parameters():
                p.add_(0.3)
            if mode == "parents_attn24":
                for p in late.exp2.parameters():
                    p.add_(0.3)
            o_late = {d: late(bt, 1, st, exit_depth=d) for d in EXITS}
        assert torch.equal(o_late[8]["arm"], o[8]["arm"])
        assert torch.equal(o_late[16]["arm"], o[16]["arm"])
        assert not torch.equal(o_late[24]["arm"], o[24]["arm"])
        mid = copy.deepcopy(m)
        with torch.no_grad():
            for p in mid.bar.action_expert.layers[12].parameters():
                p.add_(0.3)
            o_mid = {d: mid(bt, 1, st, exit_depth=d) for d in EXITS}
        assert torch.equal(o_mid[8]["arm"], o[8]["arm"])
        assert not torch.equal(o_mid[16]["arm"], o[16]["arm"])
        # 7: изменение одной головы не меняет две другие
        h = copy.deepcopy(m)
        with torch.no_grad():
            for p in h.readout16.parameters():
                p.add_(0.2)
            oh = h.forward_all(bt, 1, st)
        assert torch.equal(oh["exits"][8]["arm"], oa["exits"][8]["arm"])
        assert torch.equal(oh["exits"][24]["arm"], oa["exits"][24]["arm"])
        assert not torch.equal(oh["exits"][16]["arm"], oa["exits"][16]["arm"])
        # группы: головы отдельно, router и readout24 — в new
        g = groups(m)
        ids = {id(p) for p in m.heads_params()}
        assert {id(p) for p in g["heads"]} == ids and not any(
            id(p) in ids for k in ("expert", "lora", "new") for p in g[k])
        assert any(p is m.readout.mlp[0].weight for p in g["new"])
        # 11: потеря каждого выхода даёт градиент в разрешённые ранние слои
        m.train()
        m.checkpoint = True
        for d in EXITS:
            m.zero_grad(set_to_none=True)
            out = m.forward_all(bt, 1, st)["exits"][d]
            loss = km.loss_fn(torch, out, torch.rand(3, 20, 7))[0].mean()
            loss.backward()
            ae = m.bar.action_expert.layers
            assert float(ae[0].self_attn.q_proj.weight.grad.abs().sum()) > 0
            g23 = ae[23].self_attn.q_proj.weight.grad
            if d < 24:
                assert g23 is None or float(g23.abs().sum()) == 0, (mode, d)
            hd = getattr(m, HEADS[d])
            assert float(hd.mlp[0].weight.grad.abs().sum()) > 0
            for dd in EXITS:
                if dd != d:
                    gg = getattr(m, HEADS[dd]).mlp[0].weight.grad
                    assert gg is None or float(gg.abs().sum()) == 0
            assert all(torch.isfinite(p.grad).all() for p in m.parameters()
                       if p.grad is not None)
        m.zero_grad(set_to_none=True)
        m.eval()
        # повтор без изменений — побитово
        with torch.no_grad():
            assert torch.equal(m(bt, 1, st, exit_depth=16)["arm"],
                               o[16]["arm"])
    print("самопроверка k34_model пройдена: до attach и на глубине 24 — путь "
          "K-26 побитово; головы 8/16 — глубокие копии без общего storage, "
          "RNG не сдвигается, параметров 2·Readout (239374 при H=768); "
          "формы 20×7 и конечность; d=8 без слоёв 9–24 и Expand, d=16 без "
          "17–24 и Expand2, слои по одному разу, одна голова; forward_all — "
          "один проход и три головы; усечённые выходы = снимкам полного "
          "прохода; поздний слой/Expand не меняет ранние выходы, голова — "
          "только свой выход; градиенты каждого выхода — в ранние слои и "
          "свою голову")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
