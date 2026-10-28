#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-27: parents_mem24 — parents24 K-22 с памятью о собственном плане
(k27_memory) поверх ЗАМОРОЖЕННЫХ весов parents24.

Backbone, query 4/12/28, RoPE-адреса, маска, порождение детей, геометрия,
температуры и readout листьев — K-22 без изменений; все тензоры K-22
заморожены (requires_grad = False; base остаётся в eval и в режиме train).
Обучается только self.mem (группа new, 403 073 параметра при H = 768),
создаётся последним из собственного генератора seed·100003 + 71 (глобальный
RNG восстанавливается).
Чтения памяти (одно на уровень):
  после слоя 8 — интервалы текущих корней прежней головой dur (до памяти),
    сопоставление с оставшимися старыми корнями, gated residual; split1 и
    Expand видят обновлённые корни;
  после слоя 16 — средние с оставшимися старыми средними, prior от A0;
    split2 и Expand видят обновлённые средние.
После слоя 24 — прежний readout и новый снимок (detach): h корней/средних
до нормы, интервалы, W0/W1, собственный чанк [20, 7] (рука после
ограничителя, схват ±1), state11.
memory="off", mem=None или ни одной среды с памятью — точный путь K-22
(модуль не вызывается).
"""
import sys

import numpy as np

import k21_model as km
import k22_model as k22
import k27_memory as kmem

MODE = "parents_mem24"
ARCHITECTURE = "k27_parents_mem24"
MEM_SEED_OFFSET = 71


def build(torch):
    import torch.utils.checkpoint as tuc
    K22 = k22.build(torch)["K22"]
    MM = kmem.build(torch)

    def _rms(new, old):
        with torch.no_grad():
            return (new.float() - old.float()).pow(2).mean().sqrt() / \
                old.float().pow(2).mean().sqrt().clamp_min(1e-30)

    class K27(K22):
        def __init__(self, bar, L, U, state_mean=None, state_std=None,
                     backend="math", checkpoint=False, seed=0, lora=True,
                     memory="on"):
            super().__init__(bar, L, U, state_mean, state_std,
                             backend=backend, checkpoint=checkpoint,
                             seed=seed, lora=lora)
            if memory not in ("on", "off"):
                raise ValueError(memory)
            for p in self.parameters():
                p.requires_grad_(False)
            rng = torch.random.get_rng_state()
            try:
                self.mem = MM["MemoryModule"](self.H, torch.Generator()
                                              .manual_seed(seed * 100003 +
                                                           MEM_SEED_OFFSET))
            finally:
                torch.random.set_rng_state(rng)
            self.memory_mode = memory
            self.mode = MODE if memory == "on" else k22.MODE
            self._L = np.asarray(L, np.float64)
            self._U = np.asarray(U, np.float64)

        def groups(self):
            return dict(expert=[], lora=[], new=list(self.mem.parameters()))

        def train(self, mode=True):
            super().train(mode)
            self.bar.eval()                       # база без dropout
            return self

        def norm_state(self, st):
            st = torch.as_tensor(st, device=self.state_mean.device,
                                 dtype=torch.float32).clone()
            idx = [0, 1, 2, 9, 10]
            st[:, idx] = (st[:, idx] - self.state_mean) / self.state_std
            return st

        def _prepare(self, mem, state, B, dev):
            """Признаки старых узлов, K/V (один раз на вызов)."""
            sn = {k: v.to(dev) for k, v in mem["snap"].items()}
            use = torch.as_tensor(mem["use"], device=dev, dtype=torch.bool)
            k = torch.as_tensor(mem["k"], device=dev, dtype=torch.float32)
            ivr, alr = MM["shift"](sn["iv_root"].float(), k)
            ivm, alm = MM["shift"](sn["iv_mid"].float(), k)
            ar, swr = MM["command_summary"](sn["chunk"], sn["W0"], k,
                                            self._L, self._U)
            am, swm = MM["command_summary"](sn["chunk"], sn["W1"], k,
                                            self._L, self._U)
            alr = alr & (swr > 0) & use[:, None]
            alm = alm & (swm > 0) & use[:, None]
            so, sw = self.norm_state(sn["state11"]), self.norm_state(state)
            e = torch.cat([
                kmem.node_features(torch, MM, sn["h_root"], ar, so, sw, ivr,
                                   0),
                kmem.node_features(torch, MM, sn["h_mid"], am, so, sw, ivm,
                                   1)], 1)
            K, V = self.mem.project(e)
            return dict(ivr=ivr, alr=alr, ivm=ivm, alm=alm, K=K, V=V, k=k,
                        use=use)

        def forward(self, batch, po, state, mem=None):
            bar = self.bar
            dev = batch["input_ids"].device
            cnt = dict(vision=0, text=[0] * 24, expert=[0] * 24,
                       active=[0] * 24, mem_reads=[])
            with torch.no_grad():
                img = bar.vlm.get_image_features(
                    batch["pixel_values"], batch.get("pixel_attention_mask"))
            cnt["vision"] += 1
            _b, _l, hv, _i = bar._build_vlm_inputs_embeds(
                input_ids=batch["input_ids"], inputs_embeds=None,
                pixel_values=None, pixel_attention_mask=None,
                image_hidden_states=img)
            B, V = hv.shape[0], hv.shape[1]
            on = self.memory_mode == "on" and mem is not None and \
                bool(torch.as_tensor(mem["use"]).any())
            P = self._prepare(mem, state, B, dev) if on else None
            st = self.norm_state(state).to(dev)
            ha = (self.q0 + self.slot).expand(B, -1, -1) + \
                self.state_proj(st)[:, None]
            aux, info = {}, {}
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
                    if on:
                        S0, hn = self.mem.scores(roots, P["K"][:, :, :4])
                        O0 = MM["overlap"](iv, P["ivr"])
                        A0, al0 = MM["root_match"](S0, O0, P["alr"])
                        r0 = roots
                        roots, g0, has0 = self.mem.update(
                            roots, hn, A0, al0, P["V"][:, :, :4])
                        cnt["mem_reads"].append("root")
                        info.update(A0=A0, g_root=g0, has_root=has0,
                                    alive_root=P["alr"], S0=S0,
                                    iv_rem_root=P["ivr"], k=P["k"],
                                    use=P["use"], rms_root=_rms(roots, r0))
                    W0 = self.T["root_weights"](iv)
                    iv2, s1 = self.T["split"](iv, self.split1(roots), False)
                    W1 = self.T["split_weights"](W0, iv, s1)
                    aux.update(root=iv, split1=s1)
                    ha = torch.cat([roots, self.exp1(roots, iv2)], 1)
                elif si == 1:
                    mids = ha[:, 4:12]
                    if on and bool(P["alm"].any()):
                        S1, hn = self.mem.scores(mids, P["K"][:, :, 4:])
                        O1 = MM["overlap"](iv2, P["ivm"])
                        A1, al1, _pr = MM["mid_match"](S1, O1, P["alm"],
                                                       info["A0"])
                        m0 = mids
                        mids, g1, has1 = self.mem.update(
                            mids, hn, A1, al1, P["V"][:, :, 4:])
                        ha = torch.cat([ha[:, :4], mids], 1)
                        cnt["mem_reads"].append("mid")
                        info.update(A1=A1, g_mid=g1, has_mid=has1,
                                    alive_mid=P["alm"], S1=S1,
                                    iv_rem_mid=P["ivm"],
                                    rms_mid=_rms(mids, m0))
                    iv3, s2 = self.T["split"](iv2, self.split2(mids), False)
                    W2 = self.T["split_weights"](W1, iv2, s2)
                    aux.update(mid=iv2, split2=s2)
                    ha = torch.cat([ha, self.exp2(mids, iv3)], 1)
            out = self.readout(bar.action_expert.norm(ha[:, 12:28]), iv3, W2)
            out.update(leaves=iv3, weights=W2, hv_last=hv, **aux)
            g = torch.where(out["grip_logit"].detach().float() > 0, 1.0, -1.0)
            out["snapshot"] = dict(
                h_root=ha[:, :4].detach().float(),
                h_mid=ha[:, 4:12].detach().float(),
                iv_root=iv.detach().float(), iv_mid=iv2.detach().float(),
                W0=W0.detach().float(), W1=W1.detach().float(),
                chunk=torch.cat([out["arm"].detach().float(), g[..., None]],
                                -1),
                state11=torch.as_tensor(state, device=dev,
                                        dtype=torch.float32).detach().clone())
            if on:
                out["mem"] = info
            self.counters = cnt
            return out

    return dict(K27=K27, K22=K22, MM=MM)


def shared_keys(st):
    return {k: v for k, v in st.items() if not k.startswith("mem.")}


def mem_input(torch, bank, env_ids, episode_ids, steps, like=None):
    """MemoryBank -> аргумент mem forward (None, если памяти нет ни у кого).
    like — батч-снимок для форм пустых строк (по умолчанию — первая
    непустая строка)."""
    use, ks, rows = bank.select(env_ids, episode_ids, steps)
    if not any(use):
        return None
    if like is None:
        r0 = next(r for r in rows if r is not None)
        like = {k: v[None] for k, v in r0.items()}
    snap, _m = kmem.stack_snapshot(torch, rows, like)
    return dict(snap=snap, use=use, k=ks)


def selftest():
    import copy
    import torch
    import k21_train as ktr
    M = build(torch)
    bar0 = km.fake_bar(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    g = torch.Generator().manual_seed(0)

    def mk(cls, s=0, **kw):
        return cls(copy.deepcopy(bar0), L, U, seed=s, **kw)

    # «обученная» база: общие тензоры K-22 возмущены и скопированы
    base = mk(M["K22"])
    with torch.no_grad():
        for p in base.parameters():
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=g) * 0.02)
    sb = ktr.trainable_state(base)
    # база сравнения заморожена, как в руке SR (k22_policy): на CPU сам
    # requires_grad меняет ядро Linear на 1 ULP
    for p in base.parameters():
        p.requires_grad_(False)

    def load(m):
        own = dict(m.named_parameters())
        own.update(dict(m.named_buffers()))
        with torch.no_grad():
            for k, v in sb.items():
                own[k].copy_(v)
        return m.eval()
    base.eval()
    m = load(mk(M["K27"]))
    off = load(mk(M["K27"], memory="off"))
    new = sorted(set(ktr.trainable_state(m)) - set(sb))
    assert all(k.startswith("mem.") for k in new) and len(new) == 6, new
    assert ktr.state_hash(shared_keys(ktr.trainable_state(m))) == \
        ktr.state_hash(sb)
    assert sum(p.numel() for p in m.groups()["new"]) == kmem.n_new_params(
        m.H)
    assert all(not p.requires_grad for n, p in m.named_parameters()
               if not n.startswith("mem."))
    bt = km.fake_batch(torch, B=3, pad=2)
    st = torch.randn(3, 11, generator=g)
    with torch.no_grad():
        ob = base(bt, 1, st)
        o1 = m(bt, 1, st)                       # первый вызов: памяти нет
        oo = off(bt, 1, st)
    for o in (o1, oo):
        assert torch.equal(o["arm"], ob["arm"]) and torch.equal(
            o["grip_logit"], ob["grip_logit"])
    c = m.counters
    assert c["vision"] == 1 and all(x == 1 for x in c["text"]) and \
        c["active"] == [4] * 8 + [12] * 8 + [28] * 8 and c["mem_reads"] == []
    snap = o1["snapshot"]
    assert all(v.grad_fn is None and not v.requires_grad
               for v in snap.values())
    # 6. нулевой Wo при наличии памяти -> выход K-22 побитово
    bank = kmem.MemoryBank()
    bank.update([0, 1, 2], [5, 6, 7], [0, 0, 0], kmem.split_snapshot(snap))
    bt2 = km.fake_batch(torch, B=3, pad=2, seed=3)
    st2 = torch.randn(3, 11, generator=g)
    mi = mem_input(torch, bank, [0, 1, 2], [5, 6, 7], [8, 8, 8], snap)
    with torch.no_grad():
        ob2 = base(bt2, 1, st2)
        om = m(bt2, 1, st2, mi)
    assert m.counters["mem_reads"] == ["root", "mid"]
    assert torch.equal(om["arm"], ob2["arm"])
    A0 = om["mem"]["A0"]
    assert torch.isfinite(A0).all() and (A0 >= 0).all()
    assert torch.allclose(om["mem"]["g_root"], torch.full_like(
        om["mem"]["g_root"], 0.1))
    # 7. ненулевой Wo: память влияет; перестановка сред, reset одной,
    # новый эпизод с тем же env_id — чужая память не используется
    with torch.no_grad():
        m.mem.Wo.weight.normal_(0, 0.5, generator=g)
        om = m(bt2, 1, st2, mi)
    assert not torch.allclose(om["arm"], ob2["arm"], atol=1e-6)
    perm = [2, 0, 1]
    btp = {k: v[perm] for k, v in bt2.items()}
    mip = mem_input(torch, bank, [2, 0, 1], [7, 5, 6], [8, 8, 8])
    with torch.no_grad():
        op = m(btp, 1, st2[perm], mip)
    assert torch.allclose(op["arm"], om["arm"][perm], atol=1e-5)
    mir = mem_input(torch, bank, [0, 1, 2], [5, 99, 7], [8, 8, 8], snap)
    with torch.no_grad():
        orr = m(bt2, 1, st2, mir)
    assert torch.allclose(orr["arm"][1], ob2["arm"][1], atol=1e-5)
    assert torch.allclose(orr["arm"][0], om["arm"][0], atol=1e-5)
    bank.reset(2)
    mi3 = mem_input(torch, bank, [0, 1, 2], [5, 6, 7], [8, 8, 8], snap)
    with torch.no_grad():
        o3 = m(bt2, 1, st2, mi3)
    assert torch.allclose(o3["arm"][2], ob2["arm"][2], atol=1e-5)
    # k ≥ 20 — пустая память у всех -> путь K-22
    assert mem_input(torch, bank, [0, 1], [5, 6], [40, 40], snap) is None
    # 8. длина истории и память не растут с числом вызовов
    bank = kmem.MemoryBank()
    sizes = []
    stp = 0
    with torch.no_grad():
        for _ in range(4):
            mi_ = mem_input(torch, bank, [0, 1, 2], [5, 6, 7], [stp] * 3,
                            snap)
            o_ = m(bt2, 1, st2, mi_)
            bank.update([0, 1, 2], [5, 6, 7], [stp] * 3,
                        kmem.split_snapshot(o_["snapshot"]))
            sizes.append(bank.nbytes())
            stp += 8
    assert len(set(sizes)) == 1 and len(bank.mem) == 3
    per_env = sizes[0] / 3
    # 9 / обучение: цепочка t−16, t−8 (no_grad, собственные снимки), t с
    # градиентом; база не меняется; Wo получает градиент, Q/K/V/gate — 0 при
    # нулевом Wo; после шага — все; через прошлые вызовы градиента нет
    mt = load(mk(M["K27"], s=1))
    mt.train()
    mt.checkpoint = True
    assert not mt.bar.training
    frozen = {n: p.detach().clone() for n, p in mt.named_parameters()
              if not n.startswith("mem.")}
    opt = torch.optim.AdamW(mt.groups()["new"], lr=1e-2)
    for step_ in range(2):
        opt.zero_grad(set_to_none=True)
        bk = kmem.MemoryBank()
        with torch.no_grad():
            for t_ in (0, 8):
                mi_ = mem_input(torch, bk, [0, 1, 2], [5, 6, 7], [t_] * 3,
                                snap)
                o_ = mt(km.fake_batch(torch, B=3, seed=t_), 1, st2, mi_)
                bk.update([0, 1, 2], [5, 6, 7], [t_] * 3,
                          kmem.split_snapshot(o_["snapshot"]))
        mi_ = mem_input(torch, bk, [0, 1, 2], [5, 6, 7], [16] * 3, snap)
        out = mt(bt2, 1, st2, mi_)
        tgt = torch.rand(3, 20, 7, generator=g)
        loss, _a, _b = km.loss_fn(torch, out, tgt)
        loss.mean().backward()
        mm = mt.mem
        assert float(mm.Wo.weight.grad.abs().sum()) > 0
        if step_ == 0:
            assert all(float(w.weight.grad.abs().sum()) == 0 for w in
                       (mm.Wq, mm.Wk, mm.Wv, mm.gate))
        else:
            assert all(float(w.weight.grad.abs().sum()) > 0 for w in
                       (mm.Wq, mm.Wk, mm.Wv, mm.gate)), step_
        assert all(p.grad is None for n, p in mt.named_parameters()
                   if not n.startswith("mem."))
        opt.step()
    for n, p in mt.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), n
    # разметка не влияет на выход
    mt.eval()
    with torch.no_grad():
        a1 = mt(bt2, 1, st2, mi_)["arm"]
        km.loss_fn(torch, dict(arm=a1, grip_logit=a1[..., 0]),
                   torch.rand(3, 20, 7) * 9)
        a2 = mt(bt2, 1, st2, mi_)["arm"]
    assert torch.equal(a1, a2)
    # генератор модуля не сдвигает глобальный RNG; от seed зависит
    torch.manual_seed(3)
    mk(M["K27"])
    r1 = torch.rand(2)
    torch.manual_seed(3)
    mk(M["K22"])
    assert torch.equal(torch.rand(2), r1)
    assert not torch.equal(mk(M["K27"], 0).mem.Wq.weight,
                           mk(M["K27"], 1).mem.Wq.weight)
    print(f"самопроверка k27_model пройдена: общие тензоры = K-22, обучается "
          f"только mem.* ({kmem.n_new_params(m.H)} при H = {m.H}); первый "
          f"вызов, memory off, нулевой Wo = K-22 побитово; счётчики (vision "
          f"1, слои по 1, чтения root/mid); перестановка сред, новый эпизод "
          f"с тем же env_id, reset одной среды, k ≥ 20 — чужая память не "
          f"используется; снимок detach, размер не растёт "
          f"({per_env / 1024:.1f} КБ на среду при H = {m.H}); цепочка из "
          f"трёх вызовов: градиент только у mem.*, Wo сначала, Q/K/V/gate "
          f"после шага, база неизменна, база в eval; разметка не влияет; "
          f"глобальный RNG не сдвигается")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
