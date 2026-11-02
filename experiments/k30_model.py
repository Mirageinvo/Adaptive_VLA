#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-30: parents_hybrid24 (hard) и parents_softplan24 (soft) — K-25 с
выбором плана корней после слоя 8 и непрерывными уточнениями детей.

Один тяжёлый проход: vision один раз, каждый слой VLM и эксперта один раз,
query 4/12/28. После слоёв 1–8 — план по h8 (k30_readout.plan, FP32, один
вызов классификатора); h_root_plus = h8 + plan_proj(b_used) (ОДИН раз, до
Expand1); средние = Expand1(h_root_plus, IV1); слои 9–16; листья =
Expand2(средние, IV2); слои 17–24; readout k30_readout с v0 = root_ln(h8)
и x1/x2 = action_expert.norm(h).float() (норма один раз).
Геометрия фиксирована (буферы); dur/split1/split2 источника не вызываются,
не обучаются. Новые модули (одинаковые у H и C): root_ln LayerNorm(H),
root_logits Linear(H, 128) (Normal(0, 1e−3), bias 0), plan_proj Linear(6, H,
bias=False) (Normal(0, 1e−3)) — локальный генератор seed·100003 + 3071,
глобальный RNG не сдвигается. Буферы: P [128, 6] float32, plan_s, геометрия.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MODES = dict(hard="parents_hybrid24", soft="parents_softplan24")
ARCH = "k30_hybrid_root_plans"
NEW_PREFIX = ("root_ln.", "root_logits.", "plan_proj.")
UNUSED_PREFIX = ("dur.", "split1.", "split2.")
SEED_SPACE = 3071
K = 128
ACTIVE = [4] * 8 + [12] * 8 + [28] * 8


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def label_of(mode, seed):
    return f"{MODES[mode]}_s{seed}"


def parse_label(label):
    for mode, name in MODES.items():
        if label.startswith(name + "_s"):
            return mode, int(label[len(name) + 2:])
    raise SystemExit(f"метка {label!r} не K-30")


def new_param_count(H):
    return H * (K + 8) + K


def build(torch):
    import k22_model as k22
    import k25_model as k25m
    import k30_readout as kr
    import torch.utils.checkpoint as tuc
    nn = torch.nn
    K25 = k25m.build(torch)["K25"]
    R = kr.build(torch)

    class K30(K25):
        def __init__(self, bar, L, U, P, plan_s, state_mean=None,
                     state_std=None, checkpoint=False, seed=0,
                     plan_mode="hard"):
            super().__init__(bar, L, U, state_mean, state_std,
                             checkpoint=checkpoint, seed=seed,
                             readout_mode="cond")
            if plan_mode not in MODES:
                raise ValueError(plan_mode)
            self.plan_mode = plan_mode
            self.mode = MODES[plan_mode]
            H = self.H
            with torch.random.fork_rng(devices=[]):   # глобальный RNG цел
                self.root_ln = nn.LayerNorm(H, eps=1e-5)
                self.root_logits = nn.Linear(H, K)
                self.plan_proj = nn.Linear(6, H, bias=False)
            g = torch.Generator().manual_seed(seed * 100003 + SEED_SPACE)
            with torch.no_grad():
                self.root_ln.weight.fill_(1.0)
                self.root_ln.bias.zero_()
                self.root_logits.weight.copy_(torch.randn(
                    K, H, generator=g) * 1e-3)
                self.root_logits.bias.zero_()
                self.plan_proj.weight.copy_(torch.randn(H, 6, generator=g)
                                            * 1e-3)
            P = torch.as_tensor(np.asarray(P, np.float32))
            if tuple(P.shape) != (K, 6):
                raise SystemExit(f"словарь формы {tuple(P.shape)}")
            self.register_buffer("codec_P", P.clone())
            self.register_buffer("codec_s", torch.as_tensor(
                np.asarray(plan_s, np.float32)))
            for k_, v in R["geometry"]().items():
                self.register_buffer(f"geo_{k_}", v)
            self.classifier_calls = 0

        def geo(self):
            return {k: getattr(self, f"geo_{k}") for k in
                    ("IV0", "IV1", "IV2", "W0", "W1", "W2")}

        def set_trainable(self, phase):
            """warmup — только новые модули; main — эксперт, LoRA,
            используемые старые (q0, slot, state_proj, exp1, exp2, readout,
            cond) и новые; dur/split, vision, базовый текст — никогда."""
            for n, p in self.named_parameters():
                if n.startswith(NEW_PREFIX):
                    on = True
                elif phase == "warmup":
                    on = False
                elif n.startswith("bar.action_expert."):
                    on = True
                elif n.endswith((".A", ".B")) and n.startswith("bar."):
                    on = True
                elif n.startswith("bar.") or n.startswith(UNUSED_PREFIX):
                    on = False
                else:
                    on = True
                p.requires_grad_(on)

        def groups30(self):
            ex, lo, other, new = [], [], [], []
            for n, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if n.startswith(NEW_PREFIX):
                    new.append(p)
                elif n.startswith("bar.action_expert."):
                    ex.append(p)
                elif n.endswith((".A", ".B")):
                    lo.append(p)
                elif n.startswith("bar.") or n.startswith(UNUSED_PREFIX):
                    raise SystemExit(f"неожиданно обучаемый {n}")
                else:
                    other.append(p)
            return dict(expert=ex, lora=lo, other=other, new=new)

        def forward(self, batch, po, state, debug=False):
            bar = self.bar
            dev = batch["input_ids"].device
            cnt = dict(vision=0, text=[0] * 24, expert=[0] * 24,
                       active=[0] * 24, classifier=0)
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
            G = self.geo()
            plan = dbg = None
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
                    if self.checkpoint and self.training and \
                            torch.is_grad_enabled():
                        hv, ha = tuc.checkpoint(self._layer, l, hv, ha, mask,
                                                cos, sin, use_reentrant=False)
                    else:
                        hv, ha = self._layer(l, hv, ha, mask, cos, sin)
                    cnt["text"][l] += 1
                    cnt["expert"][l] += 1
                    cnt["active"][l] = A
                if si == 0:
                    h8 = ha
                    plan = R["plan"](self.root_ln, self.root_logits, h8,
                                     self.codec_P, self.plan_mode,
                                     self.training)
                    cnt["classifier"] += 1
                    with torch.autocast(dev.type, enabled=False):
                        inj = self.plan_proj(plan["b_used"].float())
                    hrp = h8 + inj.to(h8.dtype)
                    iv1 = G["IV1"][None].expand(B, -1, -1)
                    mids0 = self.exp1(hrp, iv1)
                    if debug:
                        dbg = dict(h8=h8.detach().float(),
                                   hrp=hrp.detach().float(),
                                   mid_init=mids0.detach().float())
                    ha = torch.cat([hrp, mids0], 1)
                elif si == 1:
                    mids = ha[:, 4:12]
                    iv2 = G["IV2"][None].expand(B, -1, -1)
                    ha = torch.cat([ha, self.exp2(mids, iv2)], 1)
            norm = bar.action_expert.norm
            x1 = norm(ha[:, 4:12]).float()
            x2 = norm(ha[:, 12:28]).float()
            o = R["readout"](self.readout, self.cond, plan["v0"],
                             plan["b_used"], x1, x2, G, self._L, self._U)
            self.counters = cnt
            out = dict(arm=o["arm"], grip_logit=o["grip_logit"], y=o["Z"],
                       plan=plan, cond=o, hv_last=hv)
            if debug:
                out["debug"] = dbg
            return out

    return dict(K30=K30, K25=K25, R=R)


# --- источник и фабрика ------------------------------------------------------------
def source_paths(seed):
    return (f"data/k25/runs/s{seed}/parents_cond24/best.pt",
            f"reports/k25/train_s{seed}.json")


def source_entry(seed):
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
    import k21_train as kt
    if c["weights_hash"] != tr["best_weights_hash"] or \
            kt.state_hash(c["weights"]) != c["weights_hash"] or \
            int(cfg["seed"]) != seed or \
            cfg.get("mode") != "parents_cond24" or \
            cfg.get("architecture") != "k25_parents_cond24":
        raise SystemExit(f"seed {seed}: источник K-25 не parents_cond24 "
                         f"этого сида или отпечаток ≠ отчёту/содержимому")
    bad = [f for f, s in cfg["code"].items()
           if sha(os.path.join(HERE, f)) != s]
    if bad:
        raise SystemExit(f"seed {seed}: код K-25 изменён: {bad}")
    return dict(seed=seed, ckpt=os.path.abspath(ck), ckpt_sha1=sha(ck),
                weights_hash=c["weights_hash"], report=os.path.abspath(rep),
                report_sha1=sha(rep), prep_id=cfg["prep_id"],
                update=int(c["update"]), code=cfg["code"])


def shared_hash(torch, mod, keys):
    import k21_train as kt
    st = kt.trainable_state(mod)
    miss = [k for k in keys if k not in st]
    if miss:
        raise SystemExit(f"в модели нет ключей источника: {miss[:3]}")
    return kt.state_hash({k: st[k] for k in keys})


def new_state(mod):
    import k21_train as kt
    st = kt.trainable_state(mod)
    return {k: v for k, v in st.items() if k.startswith(NEW_PREFIX)}


def full_hash(mod):
    """Полный отпечаток модели: обучаемые и новые тензоры + буферы словаря и
    геометрии (k21_train.trainable_state их включает)."""
    import k21_train as kt
    return kt.state_hash(kt.trainable_state(mod))


def make_model(torch, seed, mode, entry, codec, D, dev, checkpoint=True,
               bar=None):
    """K30 с весами источника K-25 и новыми модулями сида."""
    import k21_data as d21
    M = build(torch)
    if mode not in MODES:
        raise SystemExit(f"режим {mode}")
    if sha(entry["ckpt"]) != entry["ckpt_sha1"]:
        raise SystemExit(f"{entry['ckpt']} ≠ SHA подготовки")
    ck = torch.load(entry["ckpt"], map_location="cpu", weights_only=False)
    if ck["weights_hash"] != entry["weights_hash"]:
        raise SystemExit("источник ≠ отпечатку подготовки")
    prep = D["prep_report"]
    if prep["prep_id"] != entry["prep_id"]:
        raise SystemExit("источник другой подготовки K-21")
    if bar is None:
        bar = d21.load_bar(dev, prep["bar_pin"])
    m = bar["model"]
    m.action_expert.float()
    mod = M["K30"](m, D["L"], D["U"], codec["P32"], codec["s"],
                   D["state_mean"], D["state_std"], checkpoint=checkpoint,
                   seed=seed, plan_mode=mode).to(dev)
    own = dict(mod.named_parameters())
    own.update(dict(mod.named_buffers()))
    with torch.no_grad():
        for k, v in ck["weights"].items():
            if k not in own:
                raise SystemExit(f"в K30 нет {k}")
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    if shared_hash(torch, mod, list(ck["weights"])) != ck["weights_hash"]:
        raise SystemExit("веса источника после загрузки ≠ отпечатку")
    mod.set_trainable("main")
    return mod, bar


def selftest():
    import copy
    import torch
    import k21_model as km
    import k25_model as k25m
    torch.backends.cuda.matmul.allow_tf32 = False
    M = build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    g = torch.Generator().manual_seed(0)
    P = (torch.rand(K, 6, generator=g) * 1.6 - 0.8).numpy()
    s = np.full(6, 0.3)
    bar0 = km.fake_bar(torch)
    H = bar0.action_expert.config.hidden_size

    def mk(mode, seed=0):
        return M["K30"](copy.deepcopy(bar0), L, U, P, s, seed=seed,
                        plan_mode=mode)
    # глобальный RNG не сдвигается созданием новых модулей
    torch.manual_seed(5)
    k25m.build(torch)["K25"](copy.deepcopy(bar0), L, U, seed=0)
    b25 = torch.rand(3)
    torch.manual_seed(5)
    mk("hard")
    b30 = torch.rand(3)
    assert torch.equal(b25, b30)          # новые модули RNG не сдвигают
    # H и C: одинаковые все тензоры; разные сиды — разные новые
    h, c = mk("hard"), mk("soft")
    sh, sc = h.state_dict(), c.state_dict()
    assert set(sh) == set(sc) and all(torch.equal(sh[k], sc[k]) for k in sh)
    h1 = mk("hard", 1)
    assert not torch.equal(h1.root_logits.weight, h.root_logits.weight)
    assert sum(p.numel() for n, p in h.named_parameters()
               if n.startswith(NEW_PREFIX)) == new_param_count(H)
    # счётчики и единственный вызов классификатора после слоя 8
    h.eval()
    c.eval()
    bt = km.fake_batch(torch, B=3)
    st = torch.randn(3, 11, generator=g)
    with torch.no_grad():
        oh, oc = h(bt, 1, st, debug=True), c(bt, 1, st)
    cn = h.counters
    assert cn["vision"] == 1 and all(x == 1 for x in cn["text"]) and \
        cn["active"] == ACTIVE and cn["classifier"] == 1
    assert torch.equal(oh["plan"]["b_used"], torch.as_tensor(P)[
        oh["plan"]["ids"]])
    assert torch.allclose(oc["plan"]["b_used"], oc["plan"]["p"] @
                          torch.as_tensor(P))
    for o in (oh, oc):
        assert torch.isfinite(o["arm"]).all() and o["arm"].shape == (3, 20, 6)
        cd = o["cond"]
        assert torch.allclose(cd["G0"] + cd["G1"] + cd["G2"], cd["Z"],
                              atol=1e-5)
    # причинность: изменение весов после слоя 8 не меняет план
    h2 = copy.deepcopy(h)
    with torch.no_grad():
        for l in range(8, 24):
            for p in h2.bar.action_expert.layers[l].parameters():
                p.add_(0.3)
        h2.readout.mlp[0].weight.add_(0.5)
    with torch.no_grad():
        o2 = h2(bt, 1, st)
    for k_ in ("logits", "ids", "b_used"):
        assert torch.equal(o2["plan"][k_], oh["plan"][k_])
    assert not torch.equal(o2["arm"], oh["arm"])
    # передача плана: при ненулевом plan_proj замена кода меняет детей
    # (одно добавление), при plan_proj = 0 код всё ещё меняет C0
    d = oh["debug"]
    inj = h.plan_proj(oh["plan"]["b_used"]).detach()
    assert torch.allclose(d["hrp"], d["h8"] + inj, atol=1e-6)
    want_mid = h.exp1(d["hrp"], h.geo()["IV1"][None].expand(3, -1, -1))
    assert torch.allclose(d["mid_init"], want_mid.detach(), atol=1e-6)
    h3 = copy.deepcopy(h)
    with torch.no_grad():
        h3.plan_proj.weight.zero_()
        h3.root_logits.bias[:] = 0
        h3.root_logits.weight.zero_()
        h3.root_logits.bias[1] = 10.0
        oa = h3(bt, 1, st)
        h3.root_logits.bias[1] = 0.0
        h3.root_logits.bias[100] = 10.0
        ob = h3(bt, 1, st)
    assert not torch.allclose(oa["cond"]["C0"], ob["cond"]["C0"])
    # группы: warmup — только новые; main — без dur/split и базы
    h.set_trainable("warmup")
    gw = h.groups30()
    assert not gw["expert"] and not gw["lora"] and not gw["other"] and \
        len(gw["new"]) == 5
    h.set_trainable("main")
    gm = h.groups30()
    names = {id(p): n for n, p in h.named_parameters()}
    allp = [p for v in gm.values() for p in v]
    assert len(allp) == len({id(p) for p in allp})
    assert not any(names[id(p)].startswith(UNUSED_PREFIX) for p in allp)
    assert gm["expert"] and gm["lora"] and gm["other"]
    # истинные метки в forward не участвуют (сигнатура без целей); STE
    h.train()
    oh2 = h(bt, 1, st)
    assert torch.equal(oh2["plan"]["b_used"], oh2["plan"]["b_hard"])
    oh2["arm"].sum().backward()
    assert h.root_logits.weight.grad is not None and \
        h.plan_proj.weight.grad is not None and \
        all(p.grad is None for p in h.dur.parameters())
    print(f"самопроверка k30_model пройдена: H/C одинаковые тензоры, сиды "
          f"различаются, новых параметров {new_param_count(H)} при H = {H}, "
          f"один проход (4/12/28) и один вызов классификатора после слоя 8, "
          f"hard = P[ids], soft = p@P, Z = G0+G1+G2, причинность плана, "
          f"передача плана (одно добавление до Expand1), C0 зависит от кода "
          f"при plan_proj = 0, группы warmup/main без dur/split, STE-градиент")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
