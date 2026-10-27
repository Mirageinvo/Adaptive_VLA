#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-26: parents_attn24 и flat_attn24 — визуальное чтение action-query
(k26_attention) поверх parents24 K-22 и flat24 K-21.

Backbone, query (4/12/28 у parents, 16 у flat), RoPE-адреса, маска,
порождение детей, геометрия, температуры и readout листьев — без изменений;
при одном seed все тензоры базовой модели совпадают побитово (router
создаётся последним из собственного генератора seed·100003 + 61, глобальный
RNG torch восстанавливается). Добавлен один общий модуль self.router (группа
new, 2·128·(H + Dv) параметров), шесть вызовов — после слоёв 4, 8, 12, 16,
20, 24 (после полного слоя, включая MLP); после 8/16 геометрические головы и
Expand видят hidden уже после router. Визуальный банк — image-позиции
начального hv0; K/V банка — один раз на forward.
router="off" — точный путь базовой модели (для сверки с K-22/K-21 на общих
весах); при нулевом Wo включённый router даёт тот же выход побитово.
"""
import sys
import types

import numpy as np

import k21_model as km
import k21_tree as kt
import k22_model as k22
import k26_attention as ka

MODES = ("parents_attn24", "flat_attn24")
ARCH = dict(parents_attn24="k26_parents_attn24",
            flat_attn24="k26_flat_attn24")
BASE = dict(parents_attn24="parents24", flat_attn24="flat24")
ROUTER_SEED_OFFSET = 61
ACTIVE = dict(parents_attn24=[4] * 8 + [12] * 8 + [28] * 8,
              flat_attn24=[16] * 24)
ROUTER_ACTIVE = dict(parents_attn24=[4, 4, 12, 12, 28, 28],
                     flat_attn24=[16] * 6)
SIZES = {4: (4,), 12: (4, 8), 28: (4, 8, 16)}
IMG_ID_FAKE = 49


def router_query_ops(mode):
    return sum(ROUTER_ACTIVE[mode])


def build(torch):
    import torch.utils.checkpoint as tuc
    K21 = km.build(torch)["K21"]
    K22 = k22.build(torch)["K22"]
    R = ka.build(torch)
    T = kt.build(torch)

    class Attn:
        """Общая часть: router, банк, forward с шестью точками."""

        def _init_router(self, seed, router):
            if router not in ("on", "off"):
                raise ValueError(router)
            Dv = self.bar.vlm.text_model.config.hidden_size
            rng = torch.random.get_rng_state()
            try:
                self.router = R["VisualRouter"](
                    self.H, Dv, torch.Generator().manual_seed(
                        seed * 100003 + ROUTER_SEED_OFFSET))
            finally:
                torch.random.set_rng_state(rng)
            self.router_mode = router
            self.collect = False
            self.keep_maps = False

        def _bank(self, batch, hv):
            iid = getattr(self.bar, "image_token_id", None)
            if iid is None:
                raise SystemExit("у BAR нет image_token_id")
            E, valid, counts = R["pack_image_bank"](
                hv, batch["input_ids"], batch["attention_mask"], iid)
            K, V = self.router.project_bank(E, valid)
            return dict(E=E, valid=valid, counts=counts, K=K, V=V)

        def forward(self, batch, po, state):
            bar = self.bar
            dev = batch["input_ids"].device
            hier = self.hierarchical
            on = self.router_mode == "on"
            cnt = dict(vision=0, text=[0] * 24, expert=[0] * 24,
                       active=[0] * 24, router=[], kv=0)
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
            aux, infos = {}, []
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
                if not hier:
                    continue
                if si == 0:
                    roots = ha
                    iv = self.T["root"](self.dur(roots), False)
                    w = self.T["root_weights"](iv)
                    iv2, s1 = self.T["split"](iv, self.split1(roots), False)
                    w = self.T["split_weights"](w, iv, s1)
                    aux.update(root=iv, split1=s1)
                    ha = torch.cat([roots, self.exp1(roots, iv2)], 1)
                elif si == 1:
                    mids = ha[:, 4:12]
                    iv3, s2 = self.T["split"](iv2, self.split2(mids), False)
                    w = self.T["split_weights"](w, iv2, s2)
                    aux.update(mid=iv2, split2=s2)
                    ha = torch.cat([ha, self.exp2(mids, iv3)], 1)
            norm = bar.action_expert.norm
            if hier:
                out = self.readout(norm(ha[:, 12:28]), iv3, w)
                out.update(leaves=iv3, weights=w, hv_last=hv, **aux)
            else:
                iv_cur, w = kt.fixed_leaves(torch)
                iv_cur = iv_cur.to(dev).expand(B, -1, -1)
                w = w.to(dev).expand(B, -1, -1)
                out = self.readout(norm(ha), iv_cur, w)
                out.update(leaves=iv_cur, weights=w, hv_last=hv)
            if on:
                out["bank"] = dict(counts=bank["counts"],
                                   n_max=int(bank["E"].shape[1]),
                                   dv=int(bank["E"].shape[2]))
                if self.collect or self.keep_maps:
                    out["router"] = infos
            self.counters = cnt
            return out

    class K26Parents(Attn, K22):
        hierarchical = True

        def __init__(self, bar, L, U, state_mean=None, state_std=None,
                     backend="math", checkpoint=False, seed=0, lora=True,
                     router="on"):
            K22.__init__(self, bar, L, U, state_mean, state_std,
                         backend=backend, checkpoint=checkpoint, seed=seed,
                         lora=lora)
            self._init_router(seed, router)
            self.mode = "parents_attn24" if router == "on" else k22.MODE

        forward = Attn.forward

    class K26Flat(Attn, K21):
        hierarchical = False

        def __init__(self, bar, L, U, state_mean=None, state_std=None,
                     backend="math", checkpoint=False, seed=0, lora=True,
                     router="on"):
            K21.__init__(self, bar, "flat24", L, U, state_mean, state_std,
                         backend=backend, checkpoint=checkpoint, seed=seed,
                         lora=lora)
            self._init_router(seed, router)
            self.mode = "flat_attn24" if router == "on" else "flat24"

        forward = Attn.forward

    def make(mode, bar, L, U, state_mean=None, state_std=None, **kw):
        if mode == "parents_attn24":
            return K26Parents(bar, L, U, state_mean, state_std, **kw)
        if mode == "flat_attn24":
            return K26Flat(bar, L, U, state_mean, state_std, **kw)
        raise SystemExit(f"K-26 не строит режим {mode!r}")

    def make_base(mode, bar, L, U, state_mean=None, state_std=None, **kw):
        """Исторический базовый класс того же режима (для сверки)."""
        if mode == "parents_attn24":
            return K22(bar, L, U, state_mean, state_std, **kw)
        return K21(bar, "flat24", L, U, state_mean, state_std, **kw)

    return dict(K26Parents=K26Parents, K26Flat=K26Flat, make=make,
                make_base=make_base, K22=K22, K21=K21, R=R, T=T)


def shared_keys(st):
    return {k: v for k, v in st.items() if not k.startswith("router.")}


def counters_problems(mode, c, router=True):
    if c is None:
        return ["счётчики не записаны"]
    p = []
    if c.get("vision") != 1:
        p.append(f"vision {c.get('vision')}")
    for k in ("text", "expert"):
        v = c.get(k, [])
        if len(v) != 24 or any(x != 1 for x in v):
            p.append(f"слои {k} не по одному разу")
    if c.get("active") != ACTIVE[mode]:
        p.append(f"активные query {c.get('active')}")
    if router:
        if c.get("router") != ROUTER_ACTIVE[mode]:
            p.append(f"вызовы router {c.get('router')}")
        if c.get("kv") != 1:
            p.append(f"K/V банка {c.get('kv')} раз")
    return p


# --- игрушечный BAR с image-токенами ----------------------------------------
def fake_bar(torch):
    """km.fake_bar + image_token_id и merger как у SmolVLM: признаки
    изображения в позиции input_ids == image_token_id, в порядке."""
    bar = km.fake_bar(torch)
    bar.image_token_id = IMG_ID_FAKE

    def build_embeds(self, *, input_ids, inputs_embeds, pixel_values,
                     pixel_attention_mask, image_hidden_states):
        e = self.vlm.text_model.emb(input_ids)
        m = input_ids == self.image_token_id
        if int(m.sum()) != image_hidden_states.shape[0] * \
                image_hidden_states.shape[1]:
            raise SystemExit("игрушечный merger: число image-токенов")
        e = e.clone()
        e[m] = image_hidden_states.reshape(-1, e.shape[-1]).to(e.dtype)
        return e.shape[0], e.shape[1], e, image_hidden_states
    bar._build_vlm_inputs_embeds = types.MethodType(build_embeds, bar)
    return bar


def fake_batch(torch, B=3, V=14, n_img=5, seed=0, pad=None, p0=4):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, IMG_ID_FAKE - 1, (B, V), generator=g)
    ids[:, p0:p0 + n_img] = IMG_ID_FAKE
    am = torch.ones(B, V, dtype=torch.long)
    if pad:
        if pad > p0:
            raise ValueError("padding задевает image-токены")
        am[1, :pad] = 0
        ids[1, :pad] = 0
    return dict(input_ids=ids, attention_mask=am,
                pixel_values=torch.randn(B, n_img, 12, generator=g))


def selftest():
    import copy
    import os
    import tempfile
    import torch
    import k21_train as ktr
    M = build(torch)
    bar0 = fake_bar(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    bt = fake_batch(torch, B=3, pad=2)
    st = torch.randn(3, 11)

    def mk(mode, s, base=False, **kw):
        f = M["make_base"] if base else M["make"]
        return f(mode, copy.deepcopy(bar0), L, U, seed=s, **kw)
    for mode in MODES:
        for s in (0, 1):
            a, b = mk(mode, s), mk(mode, s, base=True)
            sa, sb = ktr.trainable_state(a), ktr.trainable_state(b)
            new = sorted(set(sa) - set(sb))
            assert new == ["router.Wk.weight", "router.Wo.weight",
                           "router.Wq.weight", "router.Wv.weight"], new
            assert set(sb) <= set(sa) and all(torch.equal(sa[k], sb[k])
                                              for k in sb), (mode, s)
            assert ktr.state_hash(shared_keys(sa)) == ktr.state_hash(sb)
            assert sum(sa[k].numel() for k in new) == ka.n_new_params(
                a.H, a.router.Dv)
            gr = a.groups()
            ids = [id(p) for p in a.router.parameters()]
            assert sum(id(p) in ids for p in gr["new"]) == 4 and not any(
                id(p) in ids for k in ("expert", "lora") for p in gr[k])
            assert float(a.router.Wo.weight.detach().abs().sum()) == 0.0
            # нулевой Wo: выход = базовой модели побитово; off — тоже
            for x in (a, b):
                x.eval()
            off = mk(mode, s, router="off").eval()
            with torch.no_grad():
                oa, ob, of = a(bt, 1, st), b(bt, 1, st), off(bt, 1, st)
            for k in ("arm", "grip_logit"):
                assert torch.equal(oa[k], ob[k]) and torch.equal(of[k], ob[k])
            assert ka.AFTER == (3, 7, 11, 15, 19, 23)
            assert counters_problems(mode, a.counters) == [], a.counters
            assert counters_problems(mode, off.counters, router=False) == []
            assert off.counters["router"] == [] and off.counters["kv"] == 0
            assert oa["bank"]["counts"].tolist() == [5, 5, 5]
    # router обеих моделей одного seed побитово одинаков, от seed зависит
    for s in (0, 1):
        rp, rf = mk("parents_attn24", s).router, mk("flat_attn24", s).router
        assert all(torch.equal(x, y) for x, y in zip(rp.parameters(),
                                                     rf.parameters()))
    assert not torch.equal(mk("flat_attn24", 0).router.Wq.weight,
                           mk("flat_attn24", 1).router.Wq.weight)
    # создание router не сдвигает глобальный RNG
    torch.manual_seed(5)
    mk("flat_attn24", 0)
    r1 = torch.rand(3)
    torch.manual_seed(5)
    mk("flat_attn24", 0, base=True)
    assert torch.equal(torch.rand(3), r1)
    # router off на общих "обученных" весах = базовая модель
    for mode in MODES:
        b = mk(mode, 0, base=True)
        with torch.no_grad():
            for p in b.parameters():
                if p.requires_grad:
                    p.add_(torch.randn(p.shape, generator=torch.Generator()
                                       .manual_seed(p.numel())) * 0.01)
        off = mk(mode, 0, router="off")
        on = mk(mode, 0)
        sb = ktr.trainable_state(b)
        for x in (off, on):
            own = dict(x.named_parameters())
            own.update(dict(x.named_buffers()))
            with torch.no_grad():
                for k, v in sb.items():
                    own[k].copy_(v)
            x.eval()
        b.eval()
        with torch.no_grad():
            ob, of, on_ = b(bt, 1, st), off(bt, 1, st), on(bt, 1, st)
        assert torch.equal(ob["arm"], of["arm"]) and torch.equal(
            ob["arm"], on_["arm"])
        with torch.no_grad():
            on.router.Wo.weight.normal_(0, 0.5)
            on2 = on(bt, 1, st)
        assert not torch.allclose(on2["arm"], ob["arm"], atol=1e-6)
    # градиенты от настоящей инициализации: первый шаг — Wo ≠ 0 (Q/K/V
    # могут быть 0), после шага, изменившего Wo, — Q/K/V тоже; эксперт
    # ранний/поздний, LoRA; замороженное не меняется
    for mode in MODES:
        m = mk(mode, 0)
        m.train()
        m.checkpoint = True
        frozen = {n: p.detach().clone() for n, p in m.named_parameters()
                  if not p.requires_grad}
        opt = torch.optim.AdamW([p for g in m.groups().values() for p in g],
                                lr=1e-2)
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            out = m(bt, 1, st)
            loss, _a, _b = km.loss_fn(torch, out, torch.rand(3, 20, 7))
            loss.mean().backward()
            r = m.router
            assert float(r.Wo.weight.grad.abs().sum()) > 0, (mode, step)
            if step == 1:
                for w in (r.Wq, r.Wk, r.Wv):
                    assert float(w.weight.grad.abs().sum()) > 0, (mode, w)
            ae = m.bar.action_expert.layers
            for li in (0, 23):
                assert float(ae[li].self_attn.q_proj.weight.grad.abs()
                             .sum()) > 0
            assert all(torch.isfinite(p.grad).all() for g in
                       m.groups().values() for p in g if p.grad is not None)
            opt.step()
        assert float(m.router.Wo.weight.detach().abs().sum()) > 0
        for n, p in m.named_parameters():
            if n in frozen:
                assert torch.equal(p, frozen[n]), n
        if mode == "parents_attn24":
            for nm in ("dur", "split1", "split2"):
                assert float(getattr(m, nm).lin.weight.grad.abs().sum()) > 0
    # агрегаты, независимость от соседей батча, разметка не влияет,
    # bf16-autocast не ломает сборку
    for mode in MODES:
        m = mk(mode, 0)
        with torch.no_grad():
            m.router.Wo.weight.normal_(0, 0.5)
        m.eval()
        m.collect = True
        with torch.no_grad():
            o = m(bt, 1, st)
        assert len(o["router"]) == 6
        for inf in o["router"]:
            fi = ka.info_floats(inf)
            assert fi["pad_mass"] == 0.0 and fi["sum_err"] <= ka.SUM_TOL \
                and fi["finite"] and fi["rms_ratio"] > 0
        lv = [len(inf["levels"]) for inf in o["router"]]
        assert lv == ([1, 1, 2, 2, 3, 3] if mode == "parents_attn24"
                      else [1] * 6), lv
        m.collect = False
        bt2 = fake_batch(torch, B=3, pad=2, seed=7)
        mixed = {k: v.clone() for k, v in bt.items()}
        for k in mixed:
            mixed[k][1:] = bt2[k][1:]
        with torch.no_grad():
            a1 = m(bt, 1, st)["arm"]
            om = m(mixed, 1, torch.cat([st[:1], torch.randn(2, 11)]))
            km.loss_fn(torch, dict(arm=a1, grip_logit=a1[..., 0]),
                       torch.rand(3, 20, 7) * 9)
            a2 = m(bt, 1, st)["arm"]
        assert torch.allclose(om["arm"][0], a1[0], atol=1e-5)
        assert torch.equal(a1, a2)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            ob16 = m(bt, 1, st)
        lo, hi = torch.as_tensor(L).float(), torch.as_tensor(U).float()
        arm = ob16["arm"].float()
        assert torch.isfinite(arm).all() and (arm >= lo - 1e-3).all() and \
            (arm <= hi + 1e-3).all()
        # сохранение/загрузка
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "w.pt")
            torch.save(ktr.trainable_state(m), p)
            m2 = mk(mode, 1)
            own = dict(m2.named_parameters())
            own.update(dict(m2.named_buffers()))
            with torch.no_grad():
                for k, v in torch.load(p, weights_only=False).items():
                    own[k].copy_(v)
            m2.eval()
            with torch.no_grad():
                assert torch.equal(m2(bt, 1, st)["arm"], a1)
    # банк без image-токенов -> отказ
    m = mk("flat_attn24", 0).eval()
    bad = {k: v.clone() for k, v in bt.items()}
    bad["attention_mask"][0] = 0
    bad["attention_mask"][0, :4] = 1
    try:
        with torch.no_grad():
            m(bad, 1, st)
        raise AssertionError("наблюдение без image-токенов принято")
    except SystemExit as e:
        assert "image" in str(e.code)
    assert router_query_ops("parents_attn24") == 88 and \
        router_query_ops("flat_attn24") == 96
    print("самопроверка k26_model пройдена: общие тензоры = K-22/K-21 "
          "(seed 0 и 1), новые — ровно router.{Wq, Wk, Wv, Wo} один раз в "
          "new, router пары одинаков, глобальный RNG не сдвигается; нулевой "
          "Wo и router off = базовой модели побитово (и на общих "
          "изменённых весах); счётчики: vision 1, слои по 1, router 6 "
          "(4/4/12/12/28/28 и 16×6), K/V 1; градиенты Wo, затем Q/K/V, "
          "эксперта, LoRA, границ; замороженное не меняется; агрегаты "
          "(padding 0, суммы), независимость от соседей, разметка не "
          "влияет, bf16, save/load, отказ без image-токенов")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
