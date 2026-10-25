#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-24: parents_mr24 — parents24 K-22 с выходом со всех уровней дерева.

Тот же backbone, query 4/12/28, RoPE-адреса, маска, порождение детей,
геометрия и температуры; те же обучаемые тензоры (класс — подкласс K22 без
новых параметров; при одном seed все тензоры совпадают побитово). Отличие —
только сборка руки (k24_readout, mr1) из окончательных hidden корней
(ha[:, :4]), средних (ha[:, 4:12]) и листьев (ha[:, 12:28]) после слоя 24:
  C0, C1 — общий readout MLP на узлах (action_expert.norm -> readout.ln ->
  readout.mlp, local_time 0, global_time — центр интервала / 20);
  Y2 — прежний листовой raw readout (тот же вызов self.readout).
readout_mode="legacy" — диагностический режим: прежняя сборка из листьев
(должен воспроизводить K-22 на тех же весах и входах); не обучается.
forward — копия K22.forward с сохранением W0/W1/W2 (K22 перезаписывает W0).
"""
import sys

import numpy as np

import k21_model as km
import k22_model as k22
import k24_readout as kr

MODE = "parents_mr24"
ARCHITECTURE = "k24_parents_mr24"


def query_layer_ops():
    return k22.query_layer_ops()


def build(torch):
    import torch.utils.checkpoint as tuc
    K22 = k22.build(torch)["K22"]
    R = kr.build(torch)

    class K24(K22):
        def __init__(self, bar, L, U, state_mean=None, state_std=None,
                     backend="math", checkpoint=False, seed=0, lora=True,
                     readout_mode="mr"):
            super().__init__(bar, L, U, state_mean, state_std,
                             backend=backend, checkpoint=checkpoint,
                             seed=seed, lora=lora)
            if readout_mode not in ("mr", "legacy"):
                raise ValueError(readout_mode)
            self.readout_mode = readout_mode
            self.mode = MODE if readout_mode == "mr" else k22.MODE
            self._L = np.asarray(L, np.float64)
            self._U = np.asarray(U, np.float64)

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
            for si, (l0, l1, slots) in enumerate(k22.STAGES):
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
                    W0 = self.T["root_weights"](iv)
                    iv2, s1 = self.T["split"](iv, self.split1(roots), False)
                    W1 = self.T["split_weights"](W0, iv, s1)
                    aux.update(root=iv, split1=s1)
                    ha = torch.cat([roots, self.exp1(roots, iv2)], 1)
                elif si == 1:
                    mids = ha[:, 4:12]
                    iv3, s2 = self.T["split"](iv2, self.split2(mids), False)
                    W2 = self.T["split_weights"](W1, iv2, s2)
                    aux.update(mid=iv2, split2=s2)
                    ha = torch.cat([ha, self.exp2(mids, iv3)], 1)
            norm = bar.action_expert.norm
            leaf = self.readout(norm(ha[:, 12:28]), iv3, W2)
            out = dict(leaves=iv3, weights=W2, hv_last=hv, **aux)
            if self.readout_mode == "legacy":
                out.update(leaf)
            else:
                C0 = R["node_coeffs"](self.readout, norm(ha[:, :4]), iv)
                C1 = R["node_coeffs"](self.readout, norm(ha[:, 4:12]), iv2)
                Y2 = leaf["y"]

                def geometry(i):
                    return dict(root=iv[i].detach().float().cpu().tolist(),
                                split1=s1[i].detach().float().cpu().tolist(),
                                split2=s2[i].detach().float().cpu().tolist())
                o = R["assemble"](W0, W1, C0, C1, Y2, self._L, self._U,
                                  geometry)
                out.update(y=o["Z"], arm=o["arm"], grip_logit=o["grip_logit"],
                           mr=dict(C0=C0, C1=C1, Y2=Y2, Z=o["Z"], G0=o["G0"],
                                   G1=o["G1"], G2=o["G2"], W0=W0, W1=W1,
                                   W2=W2, rank0=o["rank0"], rank1=o["rank1"],
                                   Q0=o["Q0"], Q1=o["Q1"]))
            self.counters = cnt
            return out

    return dict(K24=K24, K22=K22, R=R)


def selftest():
    import copy
    import os
    import tempfile
    import torch
    import k21_train as ktr
    M = build(torch)
    bar0 = km.fake_bar(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])

    def mk(cls, s, **kw):
        return cls(copy.deepcopy(bar0), L, U, seed=s, **kw)
    for s in (0, 1):
        a, b = mk(M["K24"], s), mk(M["K22"], s)
        sa, sb = ktr.trainable_state(a), ktr.trainable_state(b)
        assert sorted(sa) == sorted(sb)
        assert all(torch.equal(sa[k], sb[k]) for k in sa), s
        assert sum(p.numel() for p in a.parameters()) == \
            sum(p.numel() for p in b.parameters())
    m = mk(M["K24"], 0)
    lg = mk(M["K24"], 0, readout_mode="legacy")
    ref = mk(M["K22"], 0)
    bt = km.fake_batch(torch, B=3, pad=2)
    st = torch.randn(3, 11)
    for x in (m, lg, ref):
        x.eval()
    with torch.no_grad():
        o, ol, orf = m(bt, 1, st), lg(bt, 1, st), ref(bt, 1, st)
    # legacy = K-22 побитово на тех же весах и входах
    assert torch.equal(ol["arm"], orf["arm"]) and \
        torch.equal(ol["grip_logit"], orf["grip_logit"])
    assert m.counters["active"] == [4] * 8 + [12] * 8 + [28] * 8
    assert m.counters["vision"] == 1 and all(c == 1 for c in m.counters["text"])
    assert o["arm"].shape == (3, 20, 6) and o["mr"]["Y2"].shape == (3, 20, 7)
    assert torch.equal(o["grip_logit"], o["mr"]["Y2"][..., 6].float())
    e = M["R"]["identity_errors"](o["mr"])
    assert max(e.values()) < kr.ID_TOL_CPU32, e
    lo, hi = torch.as_tensor(L).float(), torch.as_tensor(U).float()
    assert (o["arm"] >= lo - 1e-5).all() and (o["arm"] <= hi + 1e-5).all()
    # соседи батча не влияют на выход другого наблюдения
    bt2 = km.fake_batch(torch, B=3, pad=2, seed=5)
    mixed = {k: v.clone() for k, v in bt.items()}
    for k in mixed:
        mixed[k][1:] = bt2[k][1:]
    with torch.no_grad():
        om = m(mixed, 1, torch.cat([st[:1], torch.randn(2, 11)]))
    assert torch.allclose(om["arm"][0], o["arm"][0], atol=1e-6)
    # градиенты: каждый уровень участвует; общие группы
    m.train()
    out = m(bt, 1, st)
    for k in ("C0", "C1", "Y2"):
        out["mr"][k].retain_grad()
    tgt = torch.rand(3, 20, 7)
    loss, _a, _b = km.loss_fn(torch, out, tgt)
    loss.mean().backward()
    for k in ("C0", "C1", "Y2"):
        g = out["mr"][k].grad
        assert g is not None and torch.isfinite(g).all() and \
            float(g[..., :6].abs().sum()) > 0, k
    gn = {k: sum(float((p.grad ** 2).sum()) for p in v if p.grad is not None)
          for k, v in m.groups().items()}
    assert all(v > 0 for v in gn.values()), gn
    for nm in ("dur", "split1", "split2", "readout"):
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
                   for p in getattr(m, nm).parameters()), nm
    # разметка (цели) на forward не влияет
    with torch.no_grad():
        m.eval()
        a1 = m(bt, 1, st)["arm"]
        km.loss_fn(torch, dict(arm=a1, grip_logit=a1[..., 0]),
                   torch.rand(3, 20, 7) * 9)
        a2 = m(bt, 1, st)["arm"]
    assert torch.equal(a1, a2)
    # save/load восстанавливает выход и геометрию
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "w.pt")
        torch.save(ktr.trainable_state(m), p)
        m2 = mk(M["K24"], 1)
        own = dict(m2.named_parameters())
        own.update(dict(m2.named_buffers()))
        with torch.no_grad():
            for k, v in torch.load(p, weights_only=False).items():
                own[k].copy_(v)
        m2.eval()
        with torch.no_grad():
            o2 = m2(bt, 1, st)
        assert torch.equal(o2["arm"], a1) and torch.equal(
            o2["root"], m(bt, 1, st)["root"])
    assert query_layer_ops() == 352
    print("самопроверка k24_model пройдена: параметры и инициализация = K-22 "
          "(seed 0 и 1, побитово), legacy = K-22 побитово, счётчики 4/12/28, "
          "тождества сборки, пределы, схват из листьев, независимость от "
          "соседей, градиенты C0/C1/Y2 и всех групп, разметка не влияет, "
          "save/load")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
