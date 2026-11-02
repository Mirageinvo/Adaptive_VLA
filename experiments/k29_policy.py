#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-29: вход как у харнесса K-9h и шаг политики на чанк.

Inputs — подготовка батча из наблюдения дословно как в k9h / k28_rollout_core
(нормировка состояния STATE_Q01/Q99 для промпта, кадры agentview |
eye_in_hand с CenterCrop 196 -> Resize 224, промпт, процессор, перенос
dict_apply(.to(dev, float16)), pos_offset задачи из таблицы); state11 для
базы — k21_model.state11(process_state(obs)).
Policy.chunk(batch, po, state8, noise) — один тяжёлый проход (экспорт
признаков под fp16 autocast, no_grad) и два дешёвых декодера:
  mode "actor"     — μ = Z_actor[:8] (FP32), Y = μ + d (d — шум AR(1) или
                     нули), рука = mid + half·tanh(Y/half);
  mode "reference" — то же на замороженной копии (исходный контроль K-29);
  mode "legacy"    — декодер источника под fp16 autocast (исторический путь
                     K-25; только сверки/описание).
Схват во всех режимах — знак логита reference на тех же признаках (±1);
actor схват не меняет. Масштаб исполнения и инверсия знака схвата — одно
место (scale_exec, как в K-9h).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k29_dist as kd  # noqa: E402

N8 = kd.N_EXEC
MODES = ("actor", "reference", "legacy")


def scale_exec(a_codec, max_act_q):
    action = np.array(a_codec, copy=True)
    action[..., :-1] = action[..., :-1] * max_act_q[..., :-1]
    action[..., -1] = -action[..., -1]
    return action


def harness_utils(root="third_party/actioncodec"):
    r = os.path.abspath(root)
    if r not in sys.path:
        sys.path.insert(0, r)
    import utils as U
    return U, r


def _half_mid(torch, L, U, device):
    Lt = torch.as_tensor(L, device=device, dtype=torch.float32)
    Ut = torch.as_tensor(U, device=device, dtype=torch.float32)
    return (Ut - Lt) / 2, (Ut + Lt) / 2


class Inputs:
    """Батч и pos_offset из наблюдения — как в харнессе K-9h."""

    def __init__(self, torch, proc, device, root="third_party/actioncodec",
                 cfg_path="config/eval/bar.yaml",
                 offset_table="data/pos_offset_table.json"):
        import json
        from torchvision.transforms.v2 import CenterCrop, Compose, Resize
        self.torch, self.proc = torch, proc
        self.dev = torch.device(device)
        U, r = harness_utils(root)
        self.U = U
        self.cfg = U.get_cfg(os.path.join(r, cfg_path))
        self.tf = Compose([CenterCrop(int(224 * 0.875)), Resize(224)])
        self.offsets = json.load(open(offset_table))["offsets_by_suite"]["10"]
        self.max_act_q = np.maximum(np.abs(U.ACTION_Q99), np.abs(U.ACTION_Q01))

    def seed(self, s):
        self.U.seed_everything(int(s))

    def scale(self, a_codec):
        return scale_exec(a_codec, self.max_act_q)

    def task_desc(self, task):
        from libero.libero import benchmark
        return benchmark.get_benchmark_dict()["libero_10"]().get_task(
            int(task)).language

    def state8(self, obs):
        return np.asarray(self.U.process_state(obs["state"]), np.float64)

    def __call__(self, obs, desc, task):
        torch, U, cfg = self.torch, self.U, self.cfg
        n = len(obs["state"])
        s8 = self.state8(obs)
        state = ((s8 - U.STATE_Q01) / (U.STATE_Q99 - U.STATE_Q01) * 2.0 - 1.0)
        i1 = self.tf(torch.tensor(
            obs["agentview_image"][:, :, ::-1].copy()).permute(0, 3, 1, 2))
        i2 = self.tf(torch.tensor(
            obs["robot0_eye_in_hand_image"][:, :, ::-1].copy()
        ).permute(0, 3, 1, 2))
        image = torch.cat([i1, i2], dim=-1)
        msgs = []
        for i in range(n):
            m = U.prompt_template(
                state[i], None, desc,
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        batch = self.proc(text=texts,
                          images=[[image[i].numpy()] for i in range(n)],
                          return_tensors="pt", padding=True,
                          padding_side="left",
                          action_processor_kwargs={"embodiment_ids": 0})
        batch = U.dict_apply(lambda x: x.to(self.dev, torch.float16), batch)
        return batch, int(self.offsets[int(task)]), s8


class Policy:
    """Шаг на чанк: признаки, μ, Y, команды руки, схват reference."""

    def __init__(self, torch, mod, actor, reference, M, mode="actor"):
        if mode not in MODES:
            raise ValueError(mode)
        self.torch, self.mod, self.M = torch, mod, M
        self.actor, self.reference, self.mode = actor, reference, mode
        self.dev = next(mod.parameters()).device
        L = np.asarray(mod._L, np.float64)[:6]
        U_ = np.asarray(mod._U, np.float64)[:6]
        # та же арифметика, что k25_readout.half_mid (float32 из L, U)
        self.half, self.mid = _half_mid(torch, mod._L, mod._U, self.dev)
        self.L, self.U = L, U_
        self.autocast = torch.autocast(self.dev.type, dtype=torch.float16,
                                       enabled=self.dev.type == "cuda")

    def features(self, batch, po, state8):
        import k21_model as km
        torch = self.torch
        s11 = torch.as_tensor(km.state11(np.asarray(state8, np.float64)),
                              device=self.dev)
        with torch.no_grad(), self.autocast:
            F, s11n, cnt = self.M["export"](self.mod, batch, po, s11)
        return F, s11n, cnt

    def decode(self, F, noise=None):
        """F -> dict(mu, mu_ref, Y, arm, grip_logit_ref, grip) на устройстве.
        noise [B, 8, 6] (float32) или None (детерминированно)."""
        torch = self.torch
        with torch.no_grad():
            o_ref = self.reference(F)
            if self.mode == "legacy":
                o = self.reference(F, autocast=self.autocast)
            elif self.mode == "reference":
                o = o_ref
            else:
                o = self.actor(F)
            mu = self.M["mu_of"](o).float()
            mu_ref = self.M["mu_of"](o_ref).float()
            if self.mode == "legacy" and noise is not None:
                raise SystemExit("legacy — только без шума")
            Y = mu if noise is None else mu + torch.as_tensor(
                noise, dtype=torch.float32, device=self.dev)
            with torch.autocast(self.dev.type, enabled=False):
                arm = self.mid + self.half * torch.tanh(Y / self.half)
            if self.mode == "legacy":
                arm = o["arm"][:, :N8].float()        # исторический путь
            gl = (o if self.mode == "legacy" else o_ref)["grip_logit"][
                :, :N8].float()
            grip = torch.where(gl > 0, 1.0, -1.0)
        return dict(mu=mu, mu_ref=mu_ref, Y=Y, arm=arm, grip_logit_ref=gl,
                    grip=grip, legacy_arm=o["arm"][:, :N8].float()
                    if self.mode == "legacy" else None)

    def chunk(self, batch, po, state8, noise=None):
        F, s11n, cnt = self.features(batch, po, state8)
        d = self.decode(F, noise)
        a = self.torch.cat([d["arm"], d["grip"][..., None]], -1)
        a_np = a.cpu().numpy().astype(np.float32)
        if not np.isfinite(a_np).all():
            raise SystemExit("команды не конечны")
        if ((a_np[..., :6] < self.L - 1e-4) | (a_np[..., :6] > self.U + 1e-4)
            ).any():
            raise SystemExit("рука вне пределов кодека")
        return dict(F=F, s11n=s11n, counters=cnt, a_codec=a_np, **d)


def selftest():
    import torch
    import k21_model as km
    import k25_model as k25m
    import k29_model as k29m
    k29m.no_tf32(torch)
    M = k29m.build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    mod = k25m.build(torch)["K25"](km.fake_bar(torch), L, U, seed=0).eval()
    for p in mod.parameters():
        p.requires_grad_(False)
    actor, ref = M["Decoder"](mod), M["Decoder"](mod).freeze()
    with torch.no_grad():
        actor.readout.mlp[-1].weight.add_(0.3)            # actor ≠ reference
    st8 = np.concatenate([np.random.default_rng(0).normal(0, .1, (5, 3)),
                          np.random.default_rng(1).normal(0, .5, (5, 3)),
                          np.zeros((5, 2))], 1)
    bt = km.fake_batch(torch, B=5)
    outs = {}
    for mode in ("actor", "reference"):
        P = Policy(torch, mod, actor, ref, M, mode)
        outs[mode] = P.chunk(bt, 1, st8)
    a, r = outs["actor"], outs["reference"]
    # без шума Y = μ, рука = рука декодера K-25 побитово (та же арифметика)
    assert torch.equal(a["Y"], a["mu"])
    with torch.no_grad():
        assert torch.equal(r["arm"], ref(r["F"])["arm"][:, :N8])
    # схват всегда от reference, не от actor
    assert torch.equal(a["grip"], r["grip"])
    assert not torch.equal(a["mu"], r["mu"])
    assert torch.equal(a["mu_ref"], r["mu"])
    # шум добавляется только к руке
    P = Policy(torch, mod, actor, ref, M, "actor")
    d = kd.sample_noise(np.random.default_rng(0), 5).astype(np.float32)
    n = P.chunk(bt, 1, st8, d)
    assert torch.allclose(n["Y"] - n["mu"], torch.as_tensor(d), atol=1e-6)
    assert torch.equal(n["grip"], a["grip"])
    assert a["a_codec"].shape == (5, N8, 7)
    q = np.array([0.9] * 6 + [1.0])
    e = scale_exec(a["a_codec"], q)
    assert np.array_equal(e[..., 6], -a["a_codec"][..., 6])
    print("самопроверка k29_policy пройдена: без шума Y = μ, схват — знак "
          "логита reference в любом режиме, шум только на руке, "
          "масштаб/инверсия схвата как в K-9h")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
