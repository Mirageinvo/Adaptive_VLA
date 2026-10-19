#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: однопроходная VLA на исходном полном BAR — tree_adapt24,
tree_fixed24, flat24.

ГРАФ (один inference-forward): vision encoder один раз (без градиента —
заморожен); для каждого из 24 слоёв — совместное внимание префикса
(изображение + текст) и активных action-query: префикс causal, с padding,
не видит action-ключей; action-query видят весь допустимый префикс и друг
друга двунаправленно. Слой реализован отдельным циклом на модулях исходных
слоёв VLM и эксперта (как _shared_attention_forward BAR), длина action-части
меняется между слоями:
  tree:  слои 1–8 — 4 query; после 8 — длительности и деления (4 -> 8);
         слои 9–16 — 8; после 16 — деления (8 -> 16); слои 17–24 — 16.
  flat:  16 query на всех 24 слоях, фиксированные равномерные листья.
Родители после расширения дальше не идут; дети создаются из родителей.
RoPE-позиции query: база префикса + po + (po + 1)·слот (как
_build_action_pos_ids_strided BAR, po — смещение задачи); слоты: корень
[1, 5, 9, 13], середина [0, 2, …, 14], листья [0..15]. Это внутренние адреса
дерева, не время исполнения.

Query: копия исходного bos_embedding + малое различие слотов + проекция
состояния (11 = позиция 3, вращение 6D, пальцы 2; нормировка — по train).
Обучаются: все 24 слоя эксперта (+ финальная норма), новые модули,
LoRA r16/α32 на q/k/v/o всех текстовых слоёв VLM. Vision encoder, базовые
веса текста и эмбеддинги заморожены.

Внимание: backend "math" (как BAR: fp32-маска −1e9, softmax в fp32) по
умолчанию; "sdpa" — сверяемая альтернатива.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k21_tree as kt  # noqa: E402

MODES = ("flat24", "tree_adapt24", "tree_fixed24")
N_LAYERS = 24
STAGES_TREE = ((0, 8, (1, 5, 9, 13)), (8, 16, tuple(range(0, 16, 2))),
               (16, 24, tuple(range(16))))
STAGES_FLAT = ((0, 24, tuple(range(16))),)
LORA_R, LORA_ALPHA = 16, 32
STATE_DIM = 11


# --- RoPE / GQA (формулы Llama) -------------------------------------------------
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return __import__("torch").cat((-x2, x1), -1)


def apply_rope(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x, n):
    if n == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n, t, d).reshape(b, h * n, t, d)


def state11(state8, mean=None, std=None):
    """[N, 8] (pos 3, axis-angle 3, пальцы 2) -> [N, 11]; нормировка
    позиции и пальцев по train."""
    import k16_models as km
    s = np.asarray(state8, np.float64)
    R = km.aa_to_mat_np(s[:, 3:6])
    x = np.concatenate([s[:, :3], km.mat_to_6d_np(R), s[:, 6:8]], 1)
    if mean is not None:
        idx = [0, 1, 2, 9, 10]
        x[:, idx] = (x[:, idx] - np.asarray(mean)) / np.asarray(std)
    return x.astype(np.float32)


def build(torch):
    nn = torch.nn
    F = nn.functional
    T = kt.build(torch)

    class LoRALinear(nn.Module):
        def __init__(self, base, r=LORA_R, alpha=LORA_ALPHA):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.A = nn.Parameter(torch.empty(r, base.in_features))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.B = nn.Parameter(torch.zeros(base.out_features, r))
            self.scale = alpha / r

        def forward(self, x):
            return self.base(x) + (x @ self.A.t().to(x.dtype)) @ \
                self.B.t().to(x.dtype) * self.scale

    class K21(nn.Module):
        def __init__(self, bar, mode, L, U, state_mean=None, state_std=None,
                     backend="math", checkpoint=False, seed=0, lora=True):
            super().__init__()
            if mode not in MODES:
                raise ValueError(mode)
            self.bar, self.mode = bar, mode
            self.backend, self.checkpoint = backend, checkpoint
            tm = bar.vlm.text_model
            cfg = tm.config
            self.nh = cfg.num_attention_heads
            self.nkv = getattr(cfg, "num_key_value_heads", self.nh)
            self.hd = cfg.hidden_size // self.nh
            self.H = bar.action_expert.config.hidden_size
            nl_v, nl_e = len(tm.layers), len(bar.action_expert.layers)
            if nl_v != N_LAYERS or nl_e != N_LAYERS:
                raise SystemExit(f"слоёв VLM {nl_v}, эксперта {nl_e}; "
                                 f"ожидалось {N_LAYERS}")
            if bar.action_expert.config.num_attention_heads != self.nh:
                raise SystemExit("головы эксперта ≠ головам VLM")
            # заморозка: всё, кроме эксперта
            for p in bar.parameters():
                p.requires_grad_(False)
            for p in bar.action_expert.parameters():
                p.requires_grad_(True)
            self.lora_names = []
            if lora:
                for i, layer in enumerate(tm.layers):
                    for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                        base = getattr(layer.self_attn, nm)
                        setattr(layer.self_attn, nm, LoRALinear(base))
                        self.lora_names.append(f"text.{i}.{nm}")
            g = torch.Generator().manual_seed(seed + 11)
            n0 = 16 if mode == "flat24" else 4
            self.q0 = nn.Parameter(bar.bos_embedding.detach().float()
                                   .reshape(1, 1, -1).repeat(1, n0, 1)
                                   .clone())
            self.slot = nn.Parameter(torch.randn(1, n0, self.H, generator=g)
                                     * 0.02)
            self.state_proj = nn.Linear(STATE_DIM, self.H)
            with torch.no_grad():
                self.state_proj.weight.copy_(torch.randn(
                    self.H, STATE_DIM, generator=g) * 0.02)
                self.state_proj.bias.zero_()
            torch.manual_seed(seed + 13)
            if mode != "flat24":
                self.dur = T["Head"](self.H)
                self.split1 = T["Head"](self.H)
                self.split2 = T["Head"](self.H)
                self.exp1 = T["Expand"](self.H)
                self.exp2 = T["Expand"](self.H)
            self.readout = T["Readout"](self.H, L, U)
            self.T = T
            sm = np.zeros(5) if state_mean is None else state_mean
            ss = np.ones(5) if state_std is None else state_std
            self.register_buffer("state_mean", torch.as_tensor(
                np.asarray(sm, np.float32)))
            self.register_buffer("state_std", torch.as_tensor(
                np.asarray(ss, np.float32)))
            self.counters = None

        # --- группы параметров -------------------------------------------------
        def groups(self):
            ex, lo, new = [], [], []
            for n, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if n.startswith("bar.action_expert."):
                    ex.append(p)
                elif ".A" in n[-2:] or ".B" in n[-2:]:
                    lo.append(p)
                elif n.startswith("bar."):
                    raise SystemExit(f"неожиданно обучаемый {n}")
                else:
                    new.append(p)
            return dict(expert=ex, lora=lo, new=new)

        def report(self):
            gr = self.groups()
            tot = sum(p.numel() for p in self.parameters())
            sub = dict(
                vision=sum(p.numel() for p in self.bar.vlm.vision_model
                           .parameters()) if hasattr(self.bar.vlm,
                                                     "vision_model") else None,
                text=sum(p.numel() for n, p in self.bar.vlm.named_parameters()
                         if "text_model" in n and not n.endswith((".A", ".B"))),
                expert=sum(p.numel() for p in self.bar.action_expert
                           .parameters()))
            return dict(total=tot, trainable={k: sum(p.numel() for p in v)
                                              for k, v in gr.items()},
                        subsystems=sub, lora_modules=len(self.lora_names),
                        hidden_expert=self.H, heads=self.nh, kv_heads=self.nkv,
                        head_dim=self.hd, mode=self.mode,
                        backend=self.backend)

        # --- один слой ---------------------------------------------------------
        def _layer(self, l, hv, ha, mask, cos, sin):
            vl = self.bar.vlm.text_model.layers[l]
            al = self.bar.action_expert.layers[l]
            vn, an = vl.input_layernorm(hv), al.input_layernorm(ha)
            B, V, A = hv.shape[0], hv.shape[1], ha.shape[1]

            def heads(x, n):
                return x.view(B, x.shape[1], n, self.hd).transpose(1, 2)
            q = torch.cat([heads(vl.self_attn.q_proj(vn), self.nh),
                           heads(al.self_attn.q_proj(an), self.nh)], 2)
            k = torch.cat([heads(vl.self_attn.k_proj(vn), self.nkv),
                           heads(al.self_attn.k_proj(an), self.nkv)], 2)
            v = torch.cat([heads(vl.self_attn.v_proj(vn), self.nkv),
                           heads(al.self_attn.v_proj(an), self.nkv)], 2)
            dt = q.dtype
            k, v = k.to(dt), v.to(dt)
            q, k = apply_rope(q, k, cos.to(dt), sin.to(dt))
            k = repeat_kv(k, self.nh // self.nkv)
            v = repeat_kv(v, self.nh // self.nkv)
            if self.backend == "sdpa":
                o = F.scaled_dot_product_attention(q, k, v,
                                                   attn_mask=mask.to(dt),
                                                   dropout_p=0.0)
            else:
                w = torch.matmul(q, k.transpose(-1, -2)) * (self.hd ** -0.5)
                w = torch.softmax(w + mask, -1, dtype=torch.float32).to(dt)
                o = torch.matmul(w, v)
            o = o.transpose(1, 2).reshape(B, V + A, -1)
            hv = hv + vl.self_attn.o_proj(o[:, :V])
            ha = ha + al.self_attn.o_proj(o[:, V:])
            hv = hv + vl.mlp(vl.post_attention_layernorm(hv))
            ha = ha + al.mlp(al.post_attention_layernorm(ha))
            return hv, ha

        def _mask(self, attn_mask, V, A, dev):
            B = attn_mask.shape[0]
            Tt = V + A
            m = torch.zeros((B, 1, Tt, Tt), device=dev, dtype=torch.float32)
            m[:, :, :V, V:] = -1e9
            causal = torch.triu(torch.ones((V, V), device=dev,
                                           dtype=torch.bool), 1)
            m[:, :, :V, :V].masked_fill_(causal, -1e9)
            keys = torch.cat([attn_mask.to(dev).long(),
                              torch.ones((B, A), device=dev,
                                         dtype=torch.long)], 1)
            return m + (keys == 0).float()[:, None, None, :] * -1e9

        def _pos(self, B, V, slots, po, dev):
            po = torch.as_tensor(po, device=dev, dtype=torch.long).reshape(-1)
            if po.numel() == 1:
                po = po.expand(B)
            sl = torch.as_tensor(slots, device=dev, dtype=torch.long)[None]
            ap = V + po[:, None] + (po[:, None] + 1) * sl
            vp = torch.arange(V, device=dev, dtype=torch.long)[None].expand(
                B, -1)
            return torch.cat([vp, ap], 1)

        # --- проход ------------------------------------------------------------
        def forward(self, batch, po, state):
            bar = self.bar
            dev = batch["input_ids"].device
            cnt = dict(vision=0, text=[0] * N_LAYERS, expert=[0] * N_LAYERS,
                       active=[0] * N_LAYERS)
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
            idx = [0, 1, 2, 9, 10]
            st = st.clone()
            st[:, idx] = (st[:, idx] - self.state_mean) / self.state_std
            ha = (self.q0 + self.slot).expand(B, -1, -1) + \
                self.state_proj(st)[:, None]
            stages = STAGES_FLAT if self.mode == "flat24" else STAGES_TREE
            fixed = self.mode == "tree_fixed24"
            aux = {}
            for si, (l0, l1, slots) in enumerate(stages):
                A = ha.shape[1]
                if A != len(slots):
                    raise SystemExit(f"активных query {A}, слотов "
                                     f"{len(slots)}")
                mask = self._mask(batch["attention_mask"], V, A, dev)
                pos = self._pos(B, V, slots, po, dev)
                cos, sin = bar.vlm.text_model.rotary_emb(
                    hv[:, :1].expand(B, V + A, hv.shape[-1]), position_ids=pos)
                for l in range(l0, l1):
                    if self.checkpoint and self.training:
                        hv, ha = torch.utils.checkpoint.checkpoint(
                            self._layer, l, hv, ha, mask, cos, sin,
                            use_reentrant=False)
                    else:
                        hv, ha = self._layer(l, hv, ha, mask, cos, sin)
                    cnt["text"][l] += 1
                    cnt["expert"][l] += 1
                    cnt["active"][l] = A
                if self.mode == "flat24":
                    continue
                if si == 0:
                    iv = self.T["root"](self.dur(ha), fixed)
                    w = self.T["root_weights"](iv)
                    iv2, s1 = self.T["split"](iv, self.split1(ha), fixed)
                    w = self.T["split_weights"](w, iv, s1)
                    aux.update(root=iv, split1=s1)
                    ha = self.exp1(ha, iv2)
                    iv_cur = iv2
                elif si == 1:
                    iv3, s2 = self.T["split"](iv_cur, self.split2(ha), fixed)
                    w = self.T["split_weights"](w, iv_cur, s2)
                    aux.update(mid=iv_cur, split2=s2)
                    ha = self.exp2(ha, iv3)
                    iv_cur = iv3
            if self.mode == "flat24":
                iv_cur, w = kt.fixed_leaves(torch)
                iv_cur = iv_cur.to(dev).expand(B, -1, -1)
                w = w.to(dev).expand(B, -1, -1)
            h = bar.action_expert.norm(ha)
            out = self.readout(h, iv_cur, w)
            out.update(leaves=iv_cur, weights=w, hv_last=hv, **aux)
            self.counters = cnt
            return out

    return dict(K21=K21, LoRALinear=LoRALinear)


def query_layer_ops(mode):
    st = STAGES_FLAT if mode == "flat24" else STAGES_TREE
    return sum((l1 - l0) * len(s) for l0, l1, s in st)


def loss_fn(torch, out, target, valid=None, grip_w=0.1):
    """L = L1 руки (единицы кодека, по действительным позициям) +
    0.1·BCE схвата; по строке — среднее по её позициям."""
    F = torch.nn.functional
    if valid is None:
        valid = torch.ones(target.shape[:2], device=target.device)
    v = valid.float()
    nv = v.sum(1)
    if (nv == 0).any():
        raise SystemExit("строка без действительных позиций")
    l1 = ((out["arm"] - target[..., :6]).abs().mean(-1) * v).sum(1) / nv
    gt = (target[..., 6] > 0).float()
    keep = v * (target[..., 6] != 0).float()
    bce = F.binary_cross_entropy_with_logits(out["grip_logit"], gt,
                                             reduction="none")
    bce = (bce * keep).sum(1) / keep.sum(1).clamp_min(1)
    return l1 + grip_w * bce, l1, bce


# --- игрушечный BAR для CPU-самопроверки ------------------------------------------
def fake_bar(torch, Hv=64, He=48, nh=4, nkv=2, n_layers=N_LAYERS, vocab=50):
    nn = torch.nn

    class Cfg:
        pass

    class Attn(nn.Module):
        def __init__(self, H):
            super().__init__()
            hd = Hv // nh
            self.q_proj = nn.Linear(H, nh * hd, bias=False)
            self.k_proj = nn.Linear(H, nkv * hd, bias=False)
            self.v_proj = nn.Linear(H, nkv * hd, bias=False)
            self.o_proj = nn.Linear(nh * hd, H, bias=False)

    class Layer(nn.Module):
        def __init__(self, H):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(H)
            self.self_attn = Attn(H)
            self.post_attention_layernorm = nn.LayerNorm(H)
            self.mlp = nn.Sequential(nn.Linear(H, 2 * H), nn.GELU(),
                                     nn.Linear(2 * H, H))

    class Rot(nn.Module):
        def forward(self, x, position_ids):
            hd = Hv // nh
            inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
            f = position_ids[..., None].float() * inv
            e = torch.cat([f, f], -1)
            return e.cos().to(x.dtype), e.sin().to(x.dtype)

    class Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()
            self.config.num_attention_heads = nh
            self.config.num_key_value_heads = nkv
            self.config.hidden_size = Hv
            self.layers = nn.ModuleList(Layer(Hv) for _ in range(n_layers))
            self.rotary_emb = Rot()
            self.emb = nn.Embedding(vocab, Hv)

    class VLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_model = Text()
            self.vision_model = nn.Linear(12, Hv)

        def get_image_features(self, pv, pm=None):
            return self.vision_model(pv)

    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()
            self.config.hidden_size = He
            self.config.num_attention_heads = nh
            self.layers = nn.ModuleList(Layer(He) for _ in range(n_layers))
            self.norm = nn.LayerNorm(He)

    class Bar(nn.Module):
        def __init__(self):
            super().__init__()
            self.vlm = VLM()
            self.action_expert = Expert()
            self.bos_embedding = nn.Parameter(torch.randn(1, 1, He) * .02)

        def _build_vlm_inputs_embeds(self, *, input_ids, inputs_embeds,
                                     pixel_values, pixel_attention_mask,
                                     image_hidden_states):
            e = self.vlm.text_model.emb(input_ids)
            e = torch.cat([image_hidden_states, e], 1)
            return e.shape[0], e.shape[1], e, image_hidden_states
    return Bar()


def fake_batch(torch, B=3, V_txt=7, n_img=5, seed=0, pad=None):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 50, (B, V_txt), generator=g)
    am = torch.ones(B, n_img + V_txt, dtype=torch.long)
    if pad:
        am[1, :pad] = 0
    return dict(input_ids=ids, attention_mask=am,
                pixel_values=torch.randn(B, n_img, 12, generator=g))


def selftest():
    import copy
    import tempfile
    import torch
    M = build(torch)
    torch.manual_seed(0)
    L, U = -np.ones(6) * 1.1, np.ones(6) * 4.0
    bar0 = fake_bar(torch)
    models = {m: M["K21"](copy.deepcopy(bar0), m, L, U, seed=0)
              for m in MODES}
    b = fake_batch(torch, pad=3)
    st = torch.randn(3, 11)
    outs = {}
    for m, mod in models.items():
        mod.eval()
        with torch.no_grad():
            outs[m] = mod(b, 1, st)
        c = mod.counters
        assert c["vision"] == 1 and all(x == 1 for x in c["text"]) and \
            all(x == 1 for x in c["expert"]), (m, c)
        want = [16] * 24 if m == "flat24" else [4] * 8 + [8] * 8 + [16] * 8
        assert c["active"] == want, (m, c["active"])
        o = outs[m]
        assert o["arm"].shape == (3, 20, 6) and o["grip_logit"].shape == (3,
                                                                          20)
        assert torch.isfinite(o["arm"]).all()
        assert (o["arm"] >= -1.1 - 1e-5).all() and (o["arm"] <= 4 + 1e-5).all()
        assert torch.allclose(o["weights"].sum(-1), torch.ones(3, 20),
                              atol=1e-5)
    assert query_layer_ops("tree_adapt24") == 224
    assert query_layer_ops("flat24") == 384
    # префикс не зависит от action-query: одинаков у всех вариантов на
    # действительных позициях (строки padding, у которых все ключи закрыты,
    # усредняют по всем ключам, как в BAR, и ключами для других не служат)
    keep = b["attention_mask"].bool()
    assert torch.allclose(outs["flat24"]["hv_last"][keep],
                          outs["tree_adapt24"]["hv_last"][keep], atol=1e-5)
    # начальное дерево = равномерная сетка (нулевые головы)
    ivf, _w = kt.fixed_leaves(torch)
    assert torch.allclose(outs["tree_adapt24"]["leaves"][0], ivf[0],
                          atol=1e-4)
    # соседи по батчу не меняют выход (одинаковые длины)
    m = models["tree_adapt24"]
    b2 = dict(b, input_ids=b["input_ids"].clone(),
              pixel_values=b["pixel_values"].clone())
    b2["input_ids"][2] = (b2["input_ids"][2] + 3) % 50
    b2["pixel_values"][2] += 1.0
    with torch.no_grad():
        o2 = m(b2, 1, st)
    assert torch.allclose(o2["arm"][0], outs["tree_adapt24"]["arm"][0],
                          atol=1e-5)
    # math и sdpa совпадают
    ms = copy.deepcopy(m)
    ms.backend = "sdpa"
    with torch.no_grad():
        o3 = ms(b, 1, st)
    assert torch.allclose(o3["arm"], outs["tree_adapt24"]["arm"], atol=1e-4)
    # backward: градиенты в ранних и поздних слоях эксперта, LoRA (B),
    # головы границ, расширения, readout; vision и базовый текст заморожены
    for mname in ("tree_adapt24", "flat24"):
        mod = models[mname]
        mod.train()
        mod.checkpoint = True
        o = mod(b, 1, st)
        tgt = torch.randn(3, 20, 7)
        loss, _l1, _b = loss_fn(torch, o, tgt)
        loss.mean().backward()
        ae = mod.bar.action_expert.layers
        for li in (0, 23):
            gq = ae[li].self_attn.q_proj.weight.grad
            assert gq is not None and float(gq.abs().sum()) > 0, (mname, li)
        lb = mod.bar.vlm.text_model.layers[0].self_attn.q_proj.B.grad
        assert lb is not None and float(lb.abs().sum()) > 0
        assert mod.bar.vlm.vision_model.weight.grad is None
        assert mod.bar.vlm.text_model.layers[0].self_attn.q_proj.base.weight \
            .grad is None
        if mname == "tree_adapt24":
            for hd in (mod.dur, mod.split1, mod.split2):
                assert float(hd.lin.weight.grad.abs().sum()) > 0, hd
            assert float(mod.exp1.W.weight.grad.abs().sum()) > 0
        gr = mod.groups()
        assert len(gr["lora"]) == 24 * 4 * 2 and len(gr["expert"]) > 0
    # сохранение/загрузка восстанавливает выход и границы
    m = models["tree_adapt24"]
    m.eval()
    with torch.no_grad():
        for hd in (m.dur, m.split1, m.split2):
            hd.lin.weight.normal_(0, 1)
        oa = m(b, 1, st)
    with tempfile.TemporaryDirectory() as td:
        torch.save(m.state_dict(), td + "/m.pt")
        m2 = M["K21"](copy.deepcopy(bar0), "tree_adapt24", L, U, seed=5)
        m2.load_state_dict(torch.load(td + "/m.pt"))
        m2.eval()
        with torch.no_grad():
            ob = m2(b, 1, st)
    assert torch.equal(oa["arm"], ob["arm"]) and torch.equal(oa["leaves"],
                                                             ob["leaves"])
    assert not torch.allclose(oa["leaves"][0], ivf[0], atol=1e-3)
    # потеря: маска позиций и нулевой схват
    o = dict(arm=torch.zeros(1, 20, 6), grip_logit=torch.zeros(1, 20))
    tgt = torch.zeros(1, 20, 7)
    tgt[0, :10, :6] = 1.0
    val = torch.zeros(1, 20)
    val[0, :10] = 1
    l, l1, bce = loss_fn(torch, o, tgt, val)
    assert abs(float(l1) - 1.0) < 1e-6 and float(bce) == 0.0
    tgt[0, :, 6] = 1.0
    _l, _l1, bce = loss_fn(torch, o, tgt, val)
    assert abs(float(bce) - math.log(2)) < 1e-5
    print("самопроверка k21_model пройдена: один обход 24 слоёв, длины "
          "4/8/16 и 16, query-layer 224/384, префикс не видит query, "
          "начальная равномерная сетка, независимость от соседей батча, "
          "math = sdpa, градиенты эксперта (ранний/поздний), LoRA, границ "
          "и расширений, заморозка vision и базового текста, сохранение, "
          "маска позиций и нулевой схват в потере")
    return 0


# --- проверка на настоящем BAR -------------------------------------------------------
def real_check(a):
    """Настоящие изображения -> forward/backward flat и tree; формы,
    счётчики, градиенты, допустимость, память и скорость по microbatch."""
    import torch
    import k20_data as kd
    import k21_data as d21
    dev = torch.device(a.device)
    D = d21.load_prepared(a.prep)
    M = build(torch)
    rep = {}
    rows = D["train_rows"][:: max(1, len(D["train_rows"]) // 64)][:64]
    for mode in ("tree_adapt24", "flat24"):
        bar = kd.load_bar(dev)                 # свежие исходные веса
        model = bar["model"]
        extra = [n for n, _m in model.named_modules()
                 if "lora" in n.lower() or "adapter" in n.lower()]
        if extra:
            raise SystemExit(f"в исходном BAR есть адаптеры: {extra[:3]}")
        print(f"  {mode}: класс {type(model).__name__}, ревизия "
              f"{bar['revision']}; блоков BAR {model.num_blocks}, токенов на "
              f"блок {model.block_size}", flush=True)
        model.action_expert.float()
        mod = M["K21"](model, mode, D["L"], D["U"], D["state_mean"],
                       D["state_std"], checkpoint=True).to(dev)
        r = mod.report()
        print(f"  {mode}: {json.dumps(r, ensure_ascii=False)}", flush=True)
        sizes = {}
        for mb in a.microbatches:
            sel = rows[:mb]
            batch, po, st, tgt, val = d21.make_batch(D, sel, bar, dev)
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)
            t0 = time.time()
            try:
                mod.train()
                with torch.autocast("cuda", dtype=getattr(torch, a.dtype)):
                    out = mod(batch, po, st)
                    loss, l1, bce = loss_fn(torch, out, tgt, val)
                loss.mean().backward()
                torch.cuda.synchronize(dev)
                ok = True
            except torch.cuda.OutOfMemoryError:
                ok = False
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated(dev) / 2 ** 30
            sizes[mb] = dict(ok=ok, seconds=el, peak_gb=mem)
            if ok:
                c = mod.counters
                gr = mod.groups()
                gnorm = {k: float(torch.sqrt(sum((p.grad.float() ** 2).sum()
                                                 for p in v
                                                 if p.grad is not None)))
                         for k, v in gr.items()}
                ae = mod.bar.action_expert.layers
                ge = [float(ae[i].self_attn.q_proj.weight.grad.norm())
                      for i in (0, 23)]
                fin = bool(torch.isfinite(out["arm"]).all())
                inb = bool(((out["arm"] >= torch.as_tensor(D["L"], device=dev)
                             - 1e-4) & (out["arm"] <= torch.as_tensor(
                                 D["U"], device=dev) + 1e-4)).all())
                sizes[mb].update(loss=float(loss.mean()), grad=gnorm,
                                 grad_expert_first_last=ge, finite=fin,
                                 in_bounds=inb, active=c["active"][::8],
                                 vision_calls=c["vision"],
                                 layers_once=all(x == 1 for x in c["text"])
                                 and all(x == 1 for x in c["expert"]))
            mod.zero_grad(set_to_none=True)
            print(f"    microbatch {mb}: {sizes[mb]}", flush=True)
            if not ok:
                break
        rep[mode] = dict(report=r, microbatch=sizes,
                         query_layer_ops=query_layer_ops(mode))
        del mod, model, bar
        torch.cuda.empty_cache()
    os.makedirs("reports/k21", exist_ok=True)
    json.dump(rep, open("reports/k21/real_check.json", "w"),
              ensure_ascii=False, indent=1, default=float)
    print("  отчёт: reports/k21/real_check.json")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-21: модель")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--real-check", action="store_true")
    ap.add_argument("--prep", default="data/k21/prep.npz")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--microbatches", type=int, nargs="+",
                    default=[1, 2, 4, 8])
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.real_check:
        return real_check(a)
    raise SystemExit("режим: --selftest или --real-check")


if __name__ == "__main__":
    sys.exit(main())
