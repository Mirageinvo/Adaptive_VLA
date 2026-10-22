#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-22: однопроходное дерево движения с сохранением активных родителей.

Отличие от tree_adapt24 K-21 — только состав активных query по слоям:
  слои 1–8    4 корня;
  слои 9–16   4 корня + 8 детей (дети создаются из корней, как в K-21;
              корни продолжают обновляться);
  слои 17–24  4 корня + 8 средних + 16 листьев (листья — из средних).
Все активные query двунаправленно видят друг друга и префикс, так что
дети и листья могут повторно обращаться к своим обновляющимся предкам.
Команды выдаются только из листьев (тот же readout, интервалы и
маршрутизация, что у tree_adapt24). RoPE-адреса узлов — как в K-21
(корни [1, 5, 9, 13], средние [0, 2, …, 14], листья [0..15]); уровень
узла различим по содержанию. query-layer операций: 4·8 + 12·8 + 28·8 = 352
(flat24 — 384, tree_adapt24 — 224).
Модуль не меняет k21_model.py (его SHA закреплён в контракте обученных
моделей K-21) и строится поверх его класса.
"""
import sys

import numpy as np

import k21_model as km
import k21_tree as kt

MODE = "parents24"
ROOT_SLOTS = (1, 5, 9, 13)
MID_SLOTS = tuple(range(0, 16, 2))
LEAF_SLOTS = tuple(range(16))
STAGES = ((0, 8, ROOT_SLOTS), (8, 16, ROOT_SLOTS + MID_SLOTS),
          (16, 24, ROOT_SLOTS + MID_SLOTS + LEAF_SLOTS))


def query_layer_ops():
    return sum((l1 - l0) * len(s) for l0, l1, s in STAGES)


def build(torch):
    import torch.utils.checkpoint as tuc
    M = km.build(torch)
    K21 = M["K21"]

    class K22(K21):
        def __init__(self, bar, L, U, state_mean=None, state_std=None,
                     backend="math", checkpoint=False, seed=0, lora=True):
            super().__init__(bar, "tree_adapt24", L, U, state_mean, state_std,
                             backend=backend, checkpoint=checkpoint,
                             seed=seed, lora=lora)
            self.mode = MODE

        def forward(self, batch, po, state):
            bar = self.bar
            dev = batch["input_ids"].device
            cnt = dict(vision=0, text=[0] * 24, expert=[0] * 24,
                       active=[0] * 24)
            with torch.no_grad():
                img = bar.vlm.get_image_features(
                    batch["pixel_values"], batch.get("pixel_attention_mask"))
            cnt["vision"] += 1
            _b, _l, hv, _i = bar._build_vlm_inputs_embeds(
                input_ids=batch["input_ids"], inputs_embeds=None,
                pixel_values=None, pixel_attention_mask=None,
                image_hidden_states=img)
            B, V = hv.shape[0], hv.shape[1]
            st = torch.as_tensor(state, device=dev, dtype=torch.float32)
            st = st.clone()
            idx = [0, 1, 2, 9, 10]
            st[:, idx] = (st[:, idx] - self.state_mean) / self.state_std
            ha = (self.q0 + self.slot).expand(B, -1, -1) + \
                self.state_proj(st)[:, None]
            aux = {}
            for si, (l0, l1, slots) in enumerate(STAGES):
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
            leaves = ha[:, 12:28]
            h = bar.action_expert.norm(leaves)
            out = self.readout(h, iv3, w)
            out.update(leaves=iv3, weights=w, hv_last=hv, **aux)
            self.counters = cnt
            return out

    return dict(K22=K22, K21=K21)


def selftest():
    import copy
    import torch
    M = build(torch)
    torch.manual_seed(0)
    L, U = -np.ones(6) * 1.1, np.ones(6) * 4.0
    bar0 = km.fake_bar(torch)
    m = M["K22"](copy.deepcopy(bar0), L, U, seed=0)
    t21 = km.build(torch)["K21"](copy.deepcopy(bar0), "tree_adapt24", L, U,
                                  seed=0)
    b = km.fake_batch(torch, pad=3)
    st = torch.randn(3, 11)
    m.eval()
    with torch.no_grad():
        o = m(b, 1, st)
        o21 = t21.eval()(b, 1, st)
    c = m.counters
    assert c["vision"] == 1 and all(x == 1 for x in c["text"]) and \
        all(x == 1 for x in c["expert"])
    assert c["active"] == [4] * 8 + [12] * 8 + [28] * 8, c["active"]
    assert query_layer_ops() == 352
    assert o["arm"].shape == (3, 20, 6) and torch.isfinite(o["arm"]).all()
    assert torch.allclose(o["weights"].sum(-1), torch.ones(3, 20), atol=1e-5)
    # те же начальные веса общих модулей, что у tree_adapt24 K-21
    for nm in ("state_proj", "readout", "dur", "split1", "split2", "exp1",
               "exp2"):
        assert all(torch.equal(p, q) for p, q in zip(
            getattr(m, nm).parameters(), getattr(t21, nm).parameters())), nm
    # префикс не видит query; на слоях 1–8 K-22 = K-21 (те же 4 корня)
    keep = b["attention_mask"].bool()
    assert torch.allclose(o["hv_last"][keep], o21["hv_last"][keep],
                          atol=1e-5)
    assert torch.allclose(o["root"], o21["root"], atol=1e-6)
    # родители активны: изменение корня после слоя 8 влияет на листья через
    # внимание (а не только через порождение) — градиент по exp1 и корням
    m.train()
    m.checkpoint = True
    o = m(b, 1, st)
    loss, _a, _b = km.loss_fn(torch, o, torch.randn(3, 20, 7))
    loss.mean().backward()
    ae = m.bar.action_expert.layers
    for li in (0, 23):
        assert float(ae[li].self_attn.q_proj.weight.grad.abs().sum()) > 0
    for nm in ("dur", "split1", "split2"):
        assert float(getattr(m, nm).lin.weight.grad.abs().sum()) > 0, nm
    assert float(m.exp1.W.weight.grad.abs().sum()) > 0
    lb = m.bar.vlm.text_model.layers[0].self_attn.q_proj.B.grad
    assert lb is not None and float(lb.abs().sum()) > 0
    # независимость от соседей батча
    m.eval()
    b2 = dict(b, input_ids=b["input_ids"].clone(),
              pixel_values=b["pixel_values"].clone())
    b2["input_ids"][2] = (b2["input_ids"][2] + 3) % 50
    with torch.no_grad():
        oa, ob = m(b, 1, st), m(b2, 1, st)
    assert torch.allclose(oa["arm"][0], ob["arm"][0], atol=1e-5)
    _ = kt
    print("самопроверка k22_model пройдена: один обход 24 слоёв, активные "
          "4/12/28, query-layer 352, листья -> readout, общие модули как у "
          "tree_adapt24, слои 1–8 = K-21, градиенты эксперта, LoRA, границ, "
          "независимость от соседей батча")
    return 0


if __name__ == "__main__":
    sys.exit(selftest())
