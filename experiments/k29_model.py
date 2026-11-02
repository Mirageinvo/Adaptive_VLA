#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-29: экспорт признаков parents_cond24 (K-25) после слоя 24, FP32-декодеры
(actor и замороженный reference) и критик.

export(mod, batch, po, state11) — копия пути K25.forward до readout (один
vision, каждый слой один раз, query 4/12/28), под внешним autocast
вызывающего; возвращает
  x0 = action_expert.norm(h_root).float() [B, 4, H], x1 [B, 8, H],
  x2 [B, 16, H] — норма эксперта применена РОВНО один раз;
  iv0/iv1/iv2, W0/W1/W2 — в исходном dtype (без преобразований: от него
  зависит округление в time_feats);
  s11n — state11 после нормировки базы (индексы 0,1,2,9,10).
Decoder — глубокая копия обученных readout.{ln, mlp} и cond.{A_h, A_a,
gate}; вызов k25_readout.cond_readout с identity вместо нормы эксперта.
  fp32: autocast выключен, TF32 выключен — путь actor/reference K-29;
  legacy(autocast): тот же вызов под autocast источника — сверка экспортёра
  с настоящим K-25 (побитово).
mu = Z[:, :8, :6] (сырой выход до tanh); схват — знак логита reference.
Critic: [mean x0, mean x1, mean x2, s11n, elapsed/600] (3H + 12) -> 256 ->
128 -> 1, GELU, последний слой нулевой (V = sigmoid(0) = 0.5).
"""
import copy
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k29_dist as kd  # noqa: E402

N8 = kd.N_EXEC
MAX_STEPS = 600
ACTIVE = [4] * 8 + [12] * 8 + [28] * 8
FEAT_KEYS = ("x0", "x1", "x2", "iv0", "iv1", "iv2", "W0", "W1", "W2")
TRAINABLE = ("readout.ln.", "readout.mlp.", "cond.A_h.", "cond.A_a.",
             "cond.gate.")
CRITIC_HID = (256, 128)


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def tensor_hash(st):
    h = hashlib.sha1()
    for k in sorted(st):
        h.update(k.encode())
        h.update(np.ascontiguousarray(st[k].detach().float().cpu().numpy())
                 .tobytes())
    return h.hexdigest()


def identity(h):
    return h


def no_tf32(torch):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# --- источник --------------------------------------------------------------------
def source_paths(seed):
    return (f"data/k25/runs/s{seed}/parents_cond24/best.pt",
            f"reports/k25/train_s{seed}.json")


def source_entry(seed):
    """Описание источника seed (для prepare): SHA файла = отчёту обучения,
    отпечаток весов, режим, seed, архитектура, prep_id, код контракта."""
    import torch
    ck, rep = source_paths(seed)
    for f in (ck, rep):
        if not os.path.exists(f):
            raise SystemExit(f"seed {seed}: нет {f}")
    tr = json.load(open(rep, encoding="utf-8"))
    if sha(ck) != tr["best_sha1"]:
        raise SystemExit(f"{ck} ≠ отчёту обучения K-25")
    c = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = c["config"]
    if c["weights_hash"] != tr["best_weights_hash"] or \
            int(cfg["seed"]) != seed or cfg.get("mode") != "parents_cond24" \
            or cfg.get("architecture") != "k25_parents_cond24":
        raise SystemExit(f"seed {seed}: источник K-25 не parents_cond24 "
                         f"этого сида или отпечаток ≠ отчёту")
    bad = [f for f, s in cfg["code"].items()
           if sha(os.path.join(HERE, f)) != s]
    if bad:
        raise SystemExit(f"seed {seed}: код K-25 изменён после обучения: {bad}")
    return dict(seed=seed, ckpt=os.path.abspath(ck), ckpt_sha1=sha(ck),
                weights_hash=c["weights_hash"], report=os.path.abspath(rep),
                report_sha1=sha(rep), prep_id=cfg["prep_id"],
                update=int(c["update"]), code=cfg["code"],
                teacher=(cfg.get("teacher") or {}).get("sha1")
                if isinstance(cfg.get("teacher"), dict) else None)


def load_source(torch, entry, device):
    """Настоящий K-25 источника (k25_policy.load_policy: режим, seed,
    архитектура, версия readout, prep_id, код, отпечаток) -> mod, bar, prep."""
    import k25_policy as kp25
    if sha(entry["ckpt"]) != entry["ckpt_sha1"]:
        raise SystemExit(f"{entry['ckpt']} ≠ SHA подготовки K-29")
    spec = dict(arch="parents_cond24", seed=int(entry["seed"]),
                ckpt=entry["ckpt"], weights_hash=entry["weights_hash"])
    mod, bar, ck, prep = kp25.load_policy(
        torch, f"parents_cond24_s{entry['seed']}", spec, device)
    if mod.readout_mode != "cond":
        raise SystemExit("источник не в режиме cond")
    return mod, bar, prep


def frozen_hash(mod):
    import k21_train as kt
    return kt.state_hash(kt.trainable_state(mod))


def build(torch):
    import k22_model as k22
    import k25_readout as kr
    nn = torch.nn
    R = kr.build(torch)

    def export(mod, batch, po, state11):
        """Признаки после слоя 24 (под autocast вызывающего, no_grad)."""
        bar = mod.bar
        dev = batch["input_ids"].device
        cnt = dict(vision=0, text=[0] * 24, expert=[0] * 24, active=[0] * 24)
        with torch.no_grad():
            img = bar.vlm.get_image_features(
                batch["pixel_values"], batch.get("pixel_attention_mask"))
            cnt["vision"] += 1
            _b, _l, hv, _i = bar._build_vlm_inputs_embeds(
                input_ids=batch["input_ids"], inputs_embeds=None,
                pixel_values=None, pixel_attention_mask=None,
                image_hidden_states=img)
            B, V = hv.shape[0], hv.shape[1]
            st = torch.as_tensor(state11, device=dev, dtype=torch.float32)
            st = st.clone()
            idx = [0, 1, 2, 9, 10]
            st[:, idx] = (st[:, idx] - mod.state_mean) / mod.state_std
            ha = (mod.q0 + mod.slot).expand(B, -1, -1) + \
                mod.state_proj(st)[:, None]
            for si, (l0, l1, slots) in enumerate(k22.STAGES):
                A = ha.shape[1]
                if A != len(slots):
                    raise SystemExit(f"активных query {A}, слотов "
                                     f"{len(slots)}")
                mask = mod._mask(batch["attention_mask"], V, A, dev)
                pos = mod._pos(B, V, slots, po, dev)
                cos, sin = bar.vlm.text_model.rotary_emb(
                    hv[:, :1].expand(B, V + A, hv.shape[-1]),
                    position_ids=pos)
                for l in range(l0, l1):
                    hv, ha = mod._layer(l, hv, ha, mask, cos, sin)
                    cnt["text"][l] += 1
                    cnt["expert"][l] += 1
                    cnt["active"][l] = A
                if si == 0:
                    roots = ha
                    iv = mod.T["root"](mod.dur(roots), False)
                    W0 = mod.T["root_weights"](iv)
                    iv2, s1 = mod.T["split"](iv, mod.split1(roots), False)
                    W1 = mod.T["split_weights"](W0, iv, s1)
                    ha = torch.cat([roots, mod.exp1(roots, iv2)], 1)
                elif si == 1:
                    mids = ha[:, 4:12]
                    iv3, s2 = mod.T["split"](iv2, mod.split2(mids), False)
                    W2 = mod.T["split_weights"](W1, iv2, s2)
                    ha = torch.cat([ha, mod.exp2(mids, iv3)], 1)
            norm = bar.action_expert.norm
            F = dict(x0=norm(ha[:, :4]).float(), x1=norm(ha[:, 4:12]).float(),
                     x2=norm(ha[:, 12:28]).float(), iv0=iv, iv1=iv2, iv2=iv3,
                     W0=W0, W1=W1, W2=W2)
        mod.counters = cnt
        return F, st, cnt

    class Decoder(nn.Module):
        """Копия readout.{ln, mlp} и cond.{A_h, A_a, gate} источника."""

        def __init__(self, src):
            super().__init__()
            self.readout = copy.deepcopy(src.readout).float()
            self.cond = copy.deepcopy(src.cond).float()
            self._L = np.asarray(src._L, np.float64)
            self._U = np.asarray(src._U, np.float64)
            for p in self.parameters():           # reference замораживает
                p.requires_grad_(True)            # вызывающий (freeze)

        def freeze(self):
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            return self

        def forward(self, F, autocast=None):
            dt = F["x0"].device.type
            ctx = autocast if autocast is not None else \
                torch.autocast(device_type=dt, enabled=False)
            with ctx:
                o = R["cond_readout"](self.readout, self.cond, identity,
                                      F["x0"], F["x1"], F["x2"], F["iv0"],
                                      F["iv1"], F["iv2"], F["W0"], F["W1"],
                                      F["W2"], self._L, self._U)
            return o

        def trainable(self):
            return {n: p for n, p in self.named_parameters()
                    if n.startswith(TRAINABLE)}

    def mu_of(o):
        return o["Z"][:, :N8, :]

    def uniform_(t, bound, gen):
        with torch.no_grad():
            t.copy_((torch.rand(t.shape, generator=gen) * 2 - 1) * bound)

    class Critic(nn.Module):
        def __init__(self, H, gen):
            super().__init__()
            d = 3 * H + 12
            self.l1 = nn.Linear(d, CRITIC_HID[0])
            self.l2 = nn.Linear(*CRITIC_HID)
            self.l3 = nn.Linear(CRITIC_HID[1], 1)
            for m in (self.l1, self.l2):
                b = 1.0 / math.sqrt(m.in_features)
                uniform_(m.weight, b, gen)
                uniform_(m.bias, b, gen)
            with torch.no_grad():
                self.l3.weight.zero_()
                self.l3.bias.zero_()

        def forward(self, F, s11n, elapsed):
            g = nn.functional.gelu
            x = torch.cat([F["x0"].float().mean(1), F["x1"].float().mean(1),
                           F["x2"].float().mean(1), s11n.float(),
                           (elapsed.float() / MAX_STEPS)[:, None]], -1)
            return self.l3(g(self.l2(g(self.l1(x)))))[:, 0]

    return dict(export=export, Decoder=Decoder, Critic=Critic, mu_of=mu_of,
                R=R)


def selftest():
    import torch
    import k21_model as km
    import k25_model as k25m
    no_tf32(torch)
    M = build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    mod = k25m.build(torch)["K25"](km.fake_bar(torch), L, U, seed=0).eval()
    with torch.no_grad():                     # «обученные» новые модули
        g = torch.Generator().manual_seed(3)
        for p in list(mod.cond.parameters()) + list(mod.readout.parameters()):
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    for p in mod.parameters():
        p.requires_grad_(False)
    for B in (1, 5):
        bt = km.fake_batch(torch, B=B, seed=B)
        s11 = torch.randn(B, 11, generator=torch.Generator().manual_seed(B))
        with torch.no_grad():
            ref = mod(bt, 1, s11)
            F, s11n, cnt = M["export"](mod, bt, 1, s11)
            dec = M["Decoder"](mod)
            o = dec(F, autocast=torch.autocast("cpu", enabled=False))
        assert cnt["vision"] == 1 and all(x == 1 for x in cnt["text"]) and \
            cnt["active"] == ACTIVE
        # экспортёр + копия декодера = настоящий K-25 побитово
        assert torch.equal(o["arm"], ref["arm"]) and \
            torch.equal(o["grip_logit"], ref["grip_logit"]) and \
            torch.equal(o["Z"], ref["y"])
        # норма эксперта ровно один раз: x0 = norm(h_root)
        assert F["x0"].dtype == torch.float32 and F["x0"].shape == (B, 4,
                                                                    mod.H)
        # actor и reference FP32 — побитово равны
        a, r = M["Decoder"](mod), M["Decoder"](mod)
        with torch.no_grad():
            oa, orr = a(F), r(F)
        assert torch.equal(oa["Z"], orr["Z"]) and \
            torch.equal(oa["grip_logit"], orr["grip_logit"])
        assert M["mu_of"](oa).shape == (B, N8, 6)
    # обучаемые ключи и критик
    tk = sorted(a.trainable())
    assert tk and all(k.startswith(TRAINABLE) for k in tk)
    n_tr = sum(p.numel() for p in a.trainable().values())
    assert n_tr == sum(p.numel() for p in a.parameters())
    c = M["Critic"](mod.H, torch.Generator().manual_seed(0))
    v = c(F, s11n, torch.zeros(F["x0"].shape[0]))
    assert torch.equal(v, torch.zeros_like(v))
    # градиент actor не трогает исходный модуль; без утечки исхода во вход
    a.train()
    lo = M["mu_of"](a(F)).sum()
    lo.backward()
    assert all(p.grad is not None for p in a.trainable().values()
               if p.requires_grad)
    assert all(p.grad is None for p in mod.parameters())
    print(f"самопроверка k29_model пройдена: экспорт (один проход, 4/12/28) + "
          f"копия декодера = K-25 побитово при B = 1 и 5, норма эксперта один "
          f"раз, actor = reference FP32 побитово, обучаемых параметров "
          f"{n_tr} (только readout.ln/mlp и cond), критик V = 0.5 в начале, "
          f"градиенты не доходят до источника")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
