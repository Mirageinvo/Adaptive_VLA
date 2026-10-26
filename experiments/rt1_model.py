#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, этап 1: flat24 и parents24 для 14 каналов двуручного робота.

Те же однопроходные модели K-21 (flat24) и K-22 (parents24) — тот же обход
24 слоёв, тот же состав query, те же LoRA r16/α32, обучаемый эксперт; файлы
K-21/K-22 не меняются: подкласс после super().__init__ заменяет только
  - state_proj: Linear(14 -> H) (было 11), веса N(0, 0.02²) из генератора
    seed·100003 + 21, смещение 0 — как у K-21;
  - readout: тот же MLP по листьям и временным признакам, выход 14:
    12 суставов a = mid + half·tanh(y/half) в технических пределах робота
    (URDF), 2 схвата sigmoid(y) ∈ [0, 1]; инициализация Linear
    U(±1/√fan_in) из генератора seed·100003 + 31, LayerNorm 1/0 — как у
    K-21.
Внутренняя нормировка состояния K-21 (индексы 0,1,2,9,10) отключена
нулевым средним и единичным разбросом: адаптер подаёт уже нормированное
14-канальное состояние. Потеря: L1 суставов (рад) + GRIP_W·L1 схватов по
действительным позициям чанка.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rt1_common as rc  # noqa: E402

ARCHS = ("flat24", "parents24")


def build(torch):
    import k21_model as km
    import k21_tree as kt
    import k22_model as k22
    nn = torch.nn
    T = kt.build(torch)
    phi, x_points = T["phi"], T["x_points"]
    arm_idx = list(rc.ARM_IDX)
    grip_idx = list(rc.GRIP_IDX)

    def reinit(mod, gen):
        for m in mod.modules():
            if isinstance(m, nn.Linear):
                b = 1.0 / math.sqrt(m.in_features)
                with torch.no_grad():
                    m.weight.copy_((torch.rand(m.weight.shape, generator=gen)
                                    * 2 - 1) * b)
                    if m.bias is not None:
                        m.bias.copy_((torch.rand(m.bias.shape, generator=gen)
                                      * 2 - 1) * b)
            elif isinstance(m, nn.LayerNorm):
                with torch.no_grad():
                    m.weight.fill_(1.0)
                    m.bias.zero_()

    class RTReadout(nn.Module):
        def __init__(self, H, L, U, hidden=128):
            super().__init__()
            self.ln = nn.LayerNorm(H)
            self.mlp = nn.Sequential(nn.Linear(H + 2 * kt.PHI, hidden),
                                     nn.GELU(), nn.Linear(hidden, hidden),
                                     nn.GELU(), nn.Linear(hidden, rc.N_CH))
            L, U = np.asarray(L, np.float32), np.asarray(U, np.float32)
            self.register_buffer("L", torch.as_tensor(L[arm_idx]))
            self.register_buffer("U", torch.as_tensor(U[arm_idx]))

        def forward(self, h_leaf, iv_leaf, w):
            B, n, H = h_leaf.shape
            x = x_points(h_leaf.device)
            c = iv_leaf.mean(-1)
            wd = (iv_leaf[..., 1] - iv_leaf[..., 0]).clamp_min(1e-6)
            lt = torch.clamp(2 * (x[None, None] - c[..., None])
                             / wd[..., None], -2, 2)
            hf = self.ln(h_leaf.float())[:, :, None].expand(B, n, kt.T, H)
            gx = phi(x / kt.T)[None, None].expand(B, n, kt.T, kt.PHI)
            y_leaf = self.mlp(torch.cat([hf, phi(lt), gx], -1))
            # смешивание, ограничение и сборка команды — float32 с
            # выключенным autocast (под bf16 einsum иначе даёт bf16, а запись
            # float32 через индексирование в bf16-тензор теряет точность)
            with torch.autocast(device_type=y_leaf.device.type,
                                enabled=False):
                y = torch.einsum("btn,bntc->btc", w.float(),
                                 y_leaf.float()).float()
                mid, half = (self.U + self.L) / 2, (self.U - self.L) / 2
                arm = mid + half * torch.tanh(y[..., arm_idx] / half)
                grip = torch.sigmoid(y[..., grip_idx])
                act = torch.zeros(y.shape, dtype=torch.float32,
                                  device=y.device)
                act[..., arm_idx] = arm
                act[..., grip_idx] = grip
            return dict(y=y, arm=arm, grip=grip, action=act)

    def rt_init(self, L, U, seed):
        H = self.H
        self.state_proj = nn.Linear(rc.N_CH, H)
        gp = torch.Generator().manual_seed(seed * 100003 + 21)
        with torch.no_grad():
            self.state_proj.weight.copy_(torch.randn(H, rc.N_CH, generator=gp)
                                         * 0.02)
            self.state_proj.bias.zero_()
        self.readout = RTReadout(H, L, U)
        reinit(self.readout, torch.Generator().manual_seed(seed * 100003 + 31))
        self.rt_bounds = (np.asarray(L, np.float32), np.asarray(U, np.float32))

    dummy = dict(L=np.zeros(6), U=np.ones(6), state_mean=np.zeros(5),
                 state_std=np.ones(5))
    K21 = km.build(torch)["K21"]
    K22 = k22.build(torch)["K22"]

    class RTFlat(K21):
        def __init__(self, bar, L, U, checkpoint=False, seed=0, lora=True):
            super().__init__(bar, "flat24", dummy["L"], dummy["U"],
                             dummy["state_mean"], dummy["state_std"],
                             checkpoint=checkpoint, seed=seed, lora=lora)
            rt_init(self, L, U, seed)
            self.arch = "flat24"

    class RTParents(K22):
        def __init__(self, bar, L, U, checkpoint=False, seed=0, lora=True):
            super().__init__(bar, dummy["L"], dummy["U"],
                             dummy["state_mean"], dummy["state_std"],
                             checkpoint=checkpoint, seed=seed, lora=lora)
            rt_init(self, L, U, seed)
            self.arch = "parents24"

    return dict(flat24=RTFlat, parents24=RTParents, RTReadout=RTReadout)


def loss_fn(torch, out, target, valid, grip_w=rc.GRIP_W):
    """target [B, 20, 14] (рад, схваты [0, 1]); valid [B, 20]."""
    v = valid.float()
    nv = v.sum(1)
    if (nv == 0).any():
        raise SystemExit("строка без действительных позиций")
    tgt = target.float()
    la = (out["arm"] - tgt[..., list(rc.ARM_IDX)]).abs().mean(-1)
    lg = (out["grip"] - tgt[..., list(rc.GRIP_IDX)]).abs().mean(-1)
    la = (la * v).sum(1) / nv
    lg = (lg * v).sum(1) / nv
    return la + grip_w * lg, la, lg


def selftest():
    import copy
    import torch
    import k21_model as km
    import k21_train as ktr
    import k23_train as k23t
    M = build(torch)
    bar0 = km.fake_bar(torch)
    L = np.full(14, -2.0)
    U = np.full(14, 2.0)
    L[list(rc.GRIP_IDX)], U[list(rc.GRIP_IDX)] = 0, 1
    mods = {a: M[a](copy.deepcopy(bar0), L, U, seed=1) for a in ARCHS}
    st = {a: ktr.trainable_state(m) for a, m in mods.items()}
    cp = k23t.init_compare(st["flat24"], st["parents24"])
    assert k23t.pair_problems(cp) == [], k23t.pair_problems(cp)
    assert "state_proj.weight" in cp["same"] and \
        "readout.mlp.4.weight" in cp["same"]
    m0 = M["flat24"](copy.deepcopy(bar0), L, U, seed=0)
    c0 = k23t.init_compare(st["flat24"], ktr.trainable_state(m0))
    assert "state_proj.weight" in c0["diff"] and \
        "readout.mlp.0.weight" in c0["diff"]
    b = km.fake_batch(torch, B=2, pad=1)
    s = torch.randn(2, 14)
    want = dict(flat24=[16] * 24, parents24=[4] * 8 + [12] * 8 + [28] * 8)
    for a, m in mods.items():
        m.train()
        out = m(b, rc.POS_OFFSET, s)
        assert out["action"].shape == (2, 20, 14)
        assert m.counters["active"] == want[a], (a, m.counters["active"])
        g = out["grip"]
        assert (g >= 0).all() and (g <= 1).all()
        arm = out["arm"]
        assert (arm >= -2 - 1e-5).all() and (arm <= 2 + 1e-5).all()
        tgt = torch.rand(2, 20, 14)
        val = torch.ones(2, 20)
        val[1, 15:] = 0
        loss, la, lg = loss_fn(torch, out, tgt, val)
        loss.mean().backward()
        gn = {k: sum(float((p.grad ** 2).sum()) for p in v
                     if p.grad is not None) for k, v in m.groups().items()}
        assert all(x > 0 for x in gn.values()), gn
        assert m.state_proj.weight.grad is not None and \
            m.readout.mlp[4].weight.grad is not None
    # bf16 autocast: команды float32 и совпадают с float32-путём
    rd = mods["flat24"].readout
    hh = torch.randn(2, 16, rd.ln.normalized_shape[0])
    iv = torch.stack([torch.arange(16.0), torch.arange(16.0) + 1.25], -1)[None]
    iv = iv.expand(2, -1, -1) * (20 / 17.25)
    ww = torch.softmax(torch.randn(2, 20, 16), -1)
    with torch.no_grad():
        o32 = rd(hh, iv, ww)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            o16 = rd(hh, iv, ww)
    assert all(o16[k].dtype == torch.float32 for k in ("y", "arm", "grip",
                                                       "action"))
    assert torch.equal(o16["action"][..., list(rc.ARM_IDX)], o16["arm"])
    assert torch.equal(o16["action"][..., list(rc.GRIP_IDX)], o16["grip"])
    assert float((o16["action"] - o32["action"]).abs().max()) < 0.1
    print("самопроверка rt1_model пройдена: 14 каналов (суставы в "
          "пределах, схваты в [0, 1]), активные query 16 и 4/12/28, общие "
          "тензоры пары равны, seed меняет новые модули, градиенты всех "
          "групп, float32-команды под bf16 autocast")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
