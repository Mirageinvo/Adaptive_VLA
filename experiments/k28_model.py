#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-28: parents_goal_fb24 — дерево физических подцелей и контексты малого
контроллера поверх ЗАМОРОЖЕННОГО parents24 (K-22).

Тяжёлый путь — проход K-22 без изменений (vision 1, 24 слоя VLM и
эксперта по разу, активные 4/12/28, readout листьев) -> номинальный чанк
(рука после ограничителя в единицах кодека, схват — знак логита). База в
eval и без градиента (в обучении — под no_grad: все новые модули стоят
после слоя 24). Новые модули (группа new):
  pose_root  [LN(h_корня), l/20, r/20] -> (d, ω): G_i = (p0 + σ_p d_i,
             Exp(σ_R ω_i) R0) в момент b_{i+1} (b_0 = 0, b_4 = 20);
  pose_mid   [LN(среднее h пары средних), позы родителя относительно X0
             (σ-нормировано), (l, r, split1)/20] -> поправка к интерполяции
             в split1: M_i (левый ребёнок X_L -> M_i, правый M_i -> X_R;
             X_R наследуется точно);
  pose_leaf  то же для 8 средних интервалов в split2 -> K_j;
  ctx_proj   LN(h листа) -> 64, смешивание прежними весами W2 в точках
             команд j + 0.5;
  ctrl       k28_control.Controller.
Узлы: 17 поз (X0 + 4 + 4 + 8), упорядоченных по времени; X_ref(u),
u = 0…20 — линейная позиция и геодезическая ориентация между соседними
узлами (времена — u состояния, не x = j + 0.5 команды; адреса RoPE — не
времена). Контекст c_j (94): 64 + φ(j/20) 9 + смешанные φ(локальное время
листьев) 9 + (a0_j − mid)/half 6 + приращение опорной позы j -> j+1
(σ-нормировано) 6.
mode="legacy": только номинальный путь (дерево и контроллер не строятся).
"""
import math
import sys

import numpy as np

import k21_model as km
import k22_model as k22
import k28_control as kc
import k28_geometry as kg

MODE = "parents_goal_fb24"
ARCHITECTURE = "k28_parents_goal_fb24"
SEED_OFFSET = 81
HEAD_HID, CTX_H = 128, 64
N_NODES = 17


def build(torch):
    import torch.utils.checkpoint as tuc
    nn = torch.nn
    F = nn.functional
    K22 = k22.build(torch)["K22"]
    T = km.kt.build(torch)
    G = kg.build(torch)
    C = kc.build(torch)

    def uniform_(t, bound, gen):
        with torch.no_grad():
            t.copy_((torch.rand(t.shape, generator=gen) * 2 - 1) * bound)

    class PoseHead(nn.Module):
        def __init__(self, din, gen):
            super().__init__()
            self.l1 = nn.Linear(din, HEAD_HID)
            self.l2 = nn.Linear(HEAD_HID, 6)
            b = 1.0 / math.sqrt(din)
            uniform_(self.l1.weight, b, gen)
            uniform_(self.l1.bias, b, gen)
            with torch.no_grad():
                self.l2.weight.copy_(torch.randn(self.l2.weight.shape,
                                                 generator=gen) * 1e-3)
                self.l2.bias.zero_()

        def forward(self, x):
            return self.l2(F.gelu(self.l1(x)))

    class K28(K22):
        def __init__(self, bar, L, U, state_mean=None, state_std=None,
                     backend="math", checkpoint=False, seed=0, lora=True,
                     sigma_p=(0.01, 0.01, 0.01), sigma_R=0.02, mode="fb"):
            super().__init__(bar, L, U, state_mean, state_std,
                             backend=backend, checkpoint=checkpoint,
                             seed=seed, lora=lora)
            if mode not in ("fb", "legacy"):
                raise ValueError(mode)
            for p in self.parameters():
                p.requires_grad_(False)
            H = self.H
            rng = torch.random.get_rng_state()
            try:
                g = torch.Generator().manual_seed(seed * 100003 + SEED_OFFSET)
                self.pose_root = PoseHead(H + 2, g)
                self.pose_mid = PoseHead(H + 15, g)
                self.pose_leaf = PoseHead(H + 15, g)
                self.ctx_proj = nn.Linear(H, CTX_H)
                uniform_(self.ctx_proj.weight, 1.0 / math.sqrt(H), g)
                uniform_(self.ctx_proj.bias, 1.0 / math.sqrt(H), g)
                self.ctrl = C["Controller"](g)
            finally:
                torch.random.set_rng_state(rng)
            self.register_buffer("sigma_p", torch.as_tensor(
                np.asarray(sigma_p, np.float32)))
            self.sigma_R = float(sigma_R)
            self.fb_mode = mode
            self.mode = MODE if mode == "fb" else k22.MODE
            self._L = np.asarray(L, np.float64)
            self._U = np.asarray(U, np.float64)

        # --- группы --------------------------------------------------------
        def new_modules(self):
            return dict(pose=[self.pose_root, self.pose_mid, self.pose_leaf],
                        ctx=[self.ctx_proj], ctrl=[self.ctrl])

        def groups(self):
            m = self.new_modules()
            ps = [p for k in ("pose", "ctx", "ctrl") for mm in m[k]
                  for p in mm.parameters()]
            return dict(expert=[], lora=[], new=ps)

        def pose_params(self):
            return [p for mm in self.new_modules()["pose"]
                    for p in mm.parameters()]

        def train(self, mode=True):
            super().train(mode)
            self.bar.eval()
            return self

        # --- тяжёлый путь (копия K22.forward + hidden узлов) ---------------
        def base(self, batch, po, state):
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
            out = self.readout(bar.action_expert.norm(leaves), iv3, w)
            out.update(leaves=iv3, weights=w, hv_last=hv, h_root=ha[:, :4],
                       h_mid=ha[:, 4:12], h_leaf=leaves, **aux)
            self.counters = cnt
            return out

        def forward(self, batch, po, state):
            """Номинальный путь (как K22.forward) — для руки legacy."""
            return self.base(batch, po, state)

        # --- дерево поз ----------------------------------------------------
        def _rel(self, p, R, p0, R0):
            return torch.cat([(p - p0[:, None]) / self.sigma_p,
                              G["rot_err"](R, R0[:, None].expand_as(R)) /
                              self.sigma_R], -1)

        def _apply_delta(self, p_base, R_base, d):
            return p_base + self.sigma_p * d[..., :3], \
                G["exp_so3"](self.sigma_R * d[..., 3:]) @ R_base

        def pose_tree(self, base, state8):
            """-> узлы (t [B, 17], p [B, 17, 3], R [B, 17, 3, 3]) и
            предсказания по уровням (для потерь)."""
            with torch.autocast(state8.device.type, enabled=False):
                p0, R0 = G["pose_of_state8"](torch.as_tensor(
                    state8, dtype=torch.float32, device=self.sigma_p.device))
                iv, iv2 = base["root"].float(), base["mid"].float()
                s1, s2 = base["split1"].float(), base["split2"].float()
                B = p0.shape[0]
                ln = lambda h: F.layer_norm(h.float(), (h.shape[-1],))  # noqa
                xr = torch.cat([ln(base["h_root"]), iv / 20.0], -1)
                dr = self.pose_root(xr)                           # [B, 4, 6]
                pG = p0[:, None] + self.sigma_p * dr[..., :3]
                RG = G["exp_so3"](self.sigma_R * dr[..., 3:]) @ \
                    R0[:, None]
                # левые концы корней: X0, G0, G1, G2
                pL = torch.cat([p0[:, None], pG[:, :3]], 1)
                RL = torch.cat([R0[:, None], RG[:, :3]], 1)
                l_, r_ = iv[..., 0], iv[..., 1]
                a1 = ((s1 - l_) / (r_ - l_).clamp_min(1e-6)).clamp(0, 1)
                pb, Rb = G["interp"](pL, RL, pG, RG, a1)
                hm = ln(base["h_mid"]).view(B, 4, 2, -1).mean(2)
                xm = torch.cat([hm, self._rel(pL, RL, p0, R0),
                                self._rel(pG, RG, p0, R0),
                                torch.stack([l_, r_, s1], -1) / 20.0], -1)
                pM, RM = self._apply_delta(pb, Rb, self.pose_mid(xm))
                # 8 средних интервалов: [L_i, M_i], [M_i, G_i]
                pl2 = torch.stack([pL, pM], 2).reshape(B, 8, 3)
                Rl2 = torch.stack([RL, RM], 2).reshape(B, 8, 3, 3)
                pr2 = torch.stack([pM, pG], 2).reshape(B, 8, 3)
                Rr2 = torch.stack([RM, RG], 2).reshape(B, 8, 3, 3)
                l2, r2 = iv2[..., 0], iv2[..., 1]
                a2 = ((s2 - l2) / (r2 - l2).clamp_min(1e-6)).clamp(0, 1)
                pb2, Rb2 = G["interp"](pl2, Rl2, pr2, Rr2, a2)
                hl = ln(base["h_leaf"]).view(B, 8, 2, -1).mean(2)
                xl = torch.cat([hl, self._rel(pl2, Rl2, p0, R0),
                                self._rel(pr2, Rr2, p0, R0),
                                torch.stack([l2, r2, s2], -1) / 20.0], -1)
                pK, RK = self._apply_delta(pb2, Rb2, self.pose_leaf(xl))
                # порядок по времени: X0, (K, M, K, G) × 4
                tp = [torch.zeros(B, 1, device=p0.device)]
                pp, RR = [p0[:, None]], [R0[:, None]]
                for i in range(4):
                    tp += [s2[:, 2 * i:2 * i + 1], s1[:, i:i + 1],
                           s2[:, 2 * i + 1:2 * i + 2], r_[:, i:i + 1]]
                    pp += [pK[:, 2 * i:2 * i + 1], pM[:, i:i + 1],
                           pK[:, 2 * i + 1:2 * i + 2], pG[:, i:i + 1]]
                    RR += [RK[:, 2 * i:2 * i + 1], RM[:, i:i + 1],
                           RK[:, 2 * i + 1:2 * i + 2], RG[:, i:i + 1]]
                nodes = dict(t=torch.cat(tp, 1), p=torch.cat(pp, 1),
                             R=torch.cat(RR, 1))
                levels = dict(root=(r_, pG, RG), mid=(s1, pM, RM),
                              leaf=(s2, pK, RK))
                return nodes, levels, (p0, R0)

        def reference(self, nodes):
            """X_ref(u), u = 0…20 -> (p [B, 21, 3], R [B, 21, 3, 3])."""
            with torch.autocast(nodes["p"].device.type, enabled=False):
                t = nodes["t"]
                B = t.shape[0]
                u = torch.arange(21, device=t.device, dtype=t.dtype)
                k = (torch.searchsorted(t.contiguous(),
                                        u.expand(B, 21).contiguous(),
                                        right=True) - 1).clamp(0, N_NODES - 2)
                bi = torch.arange(B, device=t.device)[:, None].expand(B, 21)
                tl, tr = t[bi, k], t[bi, k + 1]
                a = ((u[None] - tl) / (tr - tl).clamp_min(1e-6)).clamp(0, 1)
                p, R = G["interp"](nodes["p"][bi, k], nodes["R"][bi, k],
                                   nodes["p"][bi, k + 1],
                                   nodes["R"][bi, k + 1], a)
                # u = 0 — ровно X0
                p = torch.cat([nodes["p"][:, :1], p[:, 1:]], 1)
                R = torch.cat([nodes["R"][:, :1], R[:, 1:]], 1)
                return p, R

        def context(self, base, a0, ref_p, ref_R):
            """c_j [B, 20, 94]."""
            with torch.autocast(a0.device.type, enabled=False):
                W = base["weights"].float()                       # [B, 20, 16]
                hl = F.layer_norm(base["h_leaf"].float(),
                                  (base["h_leaf"].shape[-1],))
                proj = torch.einsum("btn,bnc->btc", W, self.ctx_proj(hl))
                B = W.shape[0]
                x = T["x_points"](W.device)
                iv = base["leaves"].float()
                cc = iv.mean(-1)
                wd = (iv[..., 1] - iv[..., 0]).clamp_min(1e-6)
                lt = torch.clamp(2 * (x[None, None] - cc[..., None])
                                 / wd[..., None], -2, 2)          # [B, 16, 20]
                lfe = torch.einsum("btn,bntc->btc", W, T["phi"](lt))
                jj = torch.arange(20, device=W.device, dtype=torch.float32)
                phj = T["phi"](jj / 20.0)[None].expand(B, 20, 9)
                Lt = torch.as_tensor(self._L[:6], dtype=torch.float32,
                                     device=W.device)
                Ut = torch.as_tensor(self._U[:6], dtype=torch.float32,
                                     device=W.device)
                an = (a0.float() - (Ut + Lt) / 2) / ((Ut - Lt) / 2)
                dp = (ref_p[:, 1:] - ref_p[:, :-1]) / self.sigma_p
                dR = G["rot_err"](ref_R[:, 1:], ref_R[:, :-1]) / self.sigma_R
                return torch.cat([proj, phj, lfe, an, dp, dR], -1)

        def plan(self, batch, po, state11, state8, plan_id=None,
                 autocast=None):
            """Один тяжёлый проход (под autocast — как у исходной политики) +
            дерево поз + контексты (float32 вне autocast). Без будущих
            состояний и целей."""
            import contextlib
            with torch.no_grad(), (autocast or contextlib.nullcontext()):
                base = self.base(batch, po, state11)
            return self._plan_from_base(base, state8, plan_id)

        def _plan_from_base(self, base, state8, plan_id):
            with torch.autocast(base["arm"].device.type, enabled=False):
                return self._plan_fp32(base, state8, plan_id)

        def _plan_fp32(self, base, state8, plan_id):
            a0 = base["arm"].float()
            grip = torch.where(base["grip_logit"].float() > 0, 1.0, -1.0)
            out = dict(plan_id=plan_id, a0=a0, grip=grip,
                       grip_logit=base["grip_logit"].float(), base=base,
                       sigma_p=self.sigma_p, sigma_R=self.sigma_R,
                       L=self._L, U=self._U)
            if self.fb_mode == "legacy":
                return out
            nodes, levels, x0 = self.pose_tree(base, state8)
            ref_p, ref_R = self.reference(nodes)
            out.update(nodes=nodes, levels=levels, x0=x0, ref_p=ref_p,
                       ref_R=ref_R, c=self.context(base, a0, ref_p, ref_R))
            return out

        def control_all(self, plan, state_seq8):
            """Обучение: контроллер на шагах j = 0…19 с наблюдаемыми
            состояниями state_seq8 [B, ≥20, 8] -> a [B, 20, 6], Δa, e."""
            B = plan["a0"].shape[0]
            s = torch.as_tensor(state_seq8[:, :20], dtype=torch.float32,
                                device=plan["a0"].device)
            p_now, R_now = G["pose_of_state8"](s)
            e, en = C["errors"](plan["ref_p"][:, :20], plan["ref_R"][:, :20],
                                p_now, R_now, self.sigma_p, self.sigma_R)
            r = self.ctrl.residual(plan["c"].reshape(B * 20, -1),
                                   e.reshape(B * 20, -1)).reshape(B, 20, 6)
            a, da, fl = C["correct"](plan["a0"], r, self._L, self._U)
            return dict(a=a, da=da, e=e, e_norm=en, r=r, **fl)

        def to_cpu_plan(self, plan):
            keys = ("a0", "grip", "ref_p", "ref_R", "c")
            out = {k: plan[k].detach().float().cpu() for k in keys}
            out.update(plan_id=plan["plan_id"],
                       sigma_p=self.sigma_p.detach().cpu().numpy(),
                       sigma_R=self.sigma_R, L=self._L, U=self._U)
            return out

    return dict(K28=K28, K22=K22, C=C, G=G, PoseHead=PoseHead)


NEW_PREFIXES = ("pose_root.", "pose_mid.", "pose_leaf.", "ctx_proj.",
                "ctrl.", "sigma_p")


def shared_keys(st):
    """Тензоры базы parents24 (без новых модулей и масштабов K-28)."""
    return {k: v for k, v in st.items() if not k.startswith(NEW_PREFIXES)}


def new_keys(st):
    return {k: v for k, v in st.items() if k.startswith(NEW_PREFIXES)}


def new_weights(st):
    """Обучаемые тензоры надстройки (без буфера sigma_p — он из контракта);
    по ним считается отпечаток new_hash в обучении и в политике."""
    return {k: v for k, v in new_keys(st).items() if k != "sigma_p"}


def new_param_count(mod):
    return sum(p.numel() for p in mod.groups()["new"])


def selftest():
    import copy
    import torch
    import k21_train as ktr
    M = build(torch)
    G = M["G"]
    bar0 = km.fake_bar(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    g = torch.Generator().manual_seed(0)
    base = M["K22"](copy.deepcopy(bar0), L, U, seed=0)
    with torch.no_grad():
        for p in base.parameters():
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=g) * 0.02)
        for hd in (base.dur, base.split1, base.split2):     # неравномерное
            hd.lin.weight.normal_(0, 0.5, generator=g)       # дерево
    sb = ktr.trainable_state(base)
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()

    def mk(s=0, **kw):
        m = M["K28"](copy.deepcopy(bar0), L, U, seed=s, sigma_p=(0.02, 0.02,
                                                              0.02),
                     sigma_R=0.05, **kw)
        own = dict(m.named_parameters())
        own.update(dict(m.named_buffers()))
        with torch.no_grad():
            for k, v in sb.items():
                own[k].copy_(v)
        return m.eval()
    m = mk()
    assert ktr.state_hash(shared_keys(ktr.trainable_state(m))) == \
        ktr.state_hash(sb)
    assert all(not p.requires_grad for n, p in m.named_parameters()
               if n.split(".")[0] not in ("pose_root", "pose_mid",
                                          "pose_leaf", "ctx_proj", "ctrl"))
    bt = km.fake_batch(torch, B=3, pad=2)
    st11 = torch.randn(3, 11, generator=g)
    st8 = torch.cat([torch.randn(3, 3, generator=g) * 0.1,
                     torch.randn(3, 3, generator=g) * 0.5,
                     torch.zeros(3, 2)], 1)
    with torch.no_grad():
        ob = base(bt, 1, st11)
    pl = m.plan(bt, 1, st11, st8, plan_id="x")
    lg = mk(mode="legacy").plan(bt, 1, st11, st8)
    for o in (pl, lg):
        assert torch.equal(o["a0"], ob["arm"]) and torch.equal(
            o["grip_logit"], ob["grip_logit"])
    assert "ref_p" not in lg
    c = m.counters
    assert c["vision"] == 1 and all(x == 1 for x in c["text"]) and \
        c["active"] == [4] * 8 + [12] * 8 + [28] * 8
    # дерево: X0 точно, времена упорядочены, конечные позы наследуются,
    # опорная траектория проходит через узлы
    nd = pl["nodes"]
    p0, R0 = G["pose_of_state8"](st8)
    assert torch.equal(nd["p"][:, 0], p0) and torch.equal(nd["R"][:, 0], R0)
    assert (nd["t"][:, 1:] >= nd["t"][:, :-1]).all()
    assert torch.allclose(nd["t"][:, -1], torch.full((3,), 20.0))
    assert torch.equal(pl["ref_p"][:, 0], p0) and torch.equal(
        pl["ref_R"][:, 0], R0)
    assert torch.allclose(pl["ref_p"][:, 20], nd["p"][:, -1], atol=1e-5)
    # уточнение середины не двигает конечную позу родителя
    rG = pl["levels"]["root"][1].clone()
    with torch.no_grad():
        m.pose_mid.l2.weight.normal_(0, 1, generator=g)
        m.pose_leaf.l2.weight.normal_(0, 1, generator=g)
    pl2 = m.plan(bt, 1, st11, st8, plan_id="y")
    assert torch.equal(pl2["levels"]["root"][1], rG)
    assert not torch.allclose(pl2["levels"]["mid"][1], pl["levels"]["mid"][1])
    # непрерывность опорной траектории в узлах: X_ref(t_k) = узел
    pr = pl2["ref_p"]
    for k_ in range(1, N_NODES):
        t_k = pl2["nodes"]["t"][:, k_]
        hit = (t_k - t_k.round()).abs() < 1e-6
        for b in torch.nonzero(hit).flatten().tolist():
            u = int(round(float(t_k[b])))
            assert torch.allclose(pr[b, u], pl2["nodes"]["p"][b, k_],
                                  atol=1e-5)
    assert pl["c"].shape == (3, 20, kc.CTX)
    # причинность: plan не зависит от будущего — сигнатура без будущих
    # состояний; контроллер на нулевой ошибке и нулевом последнем слое
    seq = torch.cat([st8[:, None]] * 21, 1)
    ca = m.control_all(pl, seq)
    assert torch.equal(ca["a"], pl["a0"])            # Wo-аналог = 0
    with torch.no_grad():
        m.ctrl.l3.weight.normal_(0, 1, generator=g)
    pl3 = m.plan(bt, 1, st11, st8, plan_id="z")
    seq_ref = torch.cat([G["log_so3"](pl3["ref_R"][:, :20]).new_zeros(3, 20,
                                                                      8)], 0)
    seq_ref[..., :3] = pl3["ref_p"][:, :20].detach()
    seq_ref[..., 3:6] = G["log_so3"](pl3["ref_R"][:, :20]).detach()
    seq_ref = seq_ref.detach()
    ca0 = m.control_all(pl3, seq_ref)                # ошибка ≈ 0
    assert float((ca0["a"] - pl3["a0"]).abs().max()) < 1e-4
    seq_b = seq_ref.clone()
    seq_b[:, 3:, :3] += 0.05
    cab = m.control_all(pl3, seq_b)
    assert torch.equal(cab["a"][:, :3], ca0["a"][:, :3])   # шаги до изменения
    assert float((cab["a"][:, 3:] - ca0["a"][:, 3:]).abs().max()) > 0
    # обучение: градиенты только у новых модулей; база неизменна
    mt = mk(s=1)
    mt.train()
    assert not mt.bar.training
    frozen = {n: p.detach().clone() for n, p in mt.named_parameters()
              if not p.requires_grad}
    opt = torch.optim.AdamW(mt.groups()["new"], lr=1e-2)
    for step_ in range(2):
        opt.zero_grad(set_to_none=True)
        plt = mt.plan(bt, 1, st11, st8)
        cat = mt.control_all(plt, seq_b)
        loss = (cat["a"] - torch.rand(3, 20, 6, generator=g)).abs().mean() \
            + (plt["ref_p"] - seq_b[..., :3][:, :1]).pow(2).mean()
        loss.backward()
        assert all(p.grad is None for n, p in mt.named_parameters()
                   if n in frozen)
        assert mt.ctrl.l3.weight.grad is not None and \
            float(mt.ctrl.l3.weight.grad.abs().sum()) > 0
        assert float(mt.pose_root.l2.weight.grad.abs().sum()) > 0
        if step_ == 1:
            assert float(mt.ctrl.l1.weight.grad.abs().sum()) > 0
        opt.step()
    for n, p in mt.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), n
    # независимость сред батча
    with torch.no_grad():
        q = m.plan(bt, 1, st11, st8, plan_id="a")
        bt2 = km.fake_batch(torch, B=3, pad=2, seed=9)
        mixed = {k: v.clone() for k, v in bt.items()}
        for k in mixed:
            mixed[k][1:] = bt2[k][1:]
        st8m = st8.clone()
        st8m[1:] += 0.3
        qm = m.plan(mixed, 1, st11, st8m, plan_id="b")
    assert torch.allclose(q["ref_p"][0], qm["ref_p"][0], atol=1e-5)
    assert torch.allclose(q["c"][0], qm["c"][0], atol=1e-5)
    # глобальный RNG не сдвигается; размер надстройки
    torch.manual_seed(1)
    mk()
    r1 = torch.rand(2)
    torch.manual_seed(1)
    M["K22"](copy.deepcopy(bar0), L, U, seed=0)
    assert torch.equal(torch.rand(2), r1)
    n_new = new_param_count(m)
    print(f"самопроверка k28_model пройдена: номинал (fb и legacy) = K-22 "
          f"побитово, счётчики одного прохода, X0 точно, времена узлов "
          f"упорядочены, X_ref(0) = X0 и через узлы, уточнение середины не "
          f"двигает конечную позу корня, контекст 94, нулевой последний "
          f"слой -> номинал, свежее состояние меняет только свои и "
          f"последующие шаги, градиенты только у новых модулей, база "
          f"неизменна и в eval, независимость сред, RNG; новых параметров "
          f"{n_new} при H = {m.H}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
