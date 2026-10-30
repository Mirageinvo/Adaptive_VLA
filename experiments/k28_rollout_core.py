#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-28: пошаговый цикл раскатки LIBERO (холодный блок, как K-9h).

Повторяет k9h_multiarm_gate для одного пути руки (без прочих политик):
тот же порядок «сиды -> среды -> модель», reset блока с init_state_id,
ожидание waiting_steps командой [0, …, 0, −1], обработка кадров (CenterCrop
196 -> Resize 224, склейка agentview | wrist), нормировка состояния
STATE_Q01/Q99, промпт, масштаб действия max(|Q99|, |Q01|) и инверсия знака
схвата в одном месте, условия завершения, хеши начальных состояний и
исполненных действий, формат JSON + npz блока.
Отличие одно: команда шага t берётся у руки ДО записи в журнал —
  a_codec = arm.step(plan, process_state(свежий obs), t, active = ~done);
  a_exec  = масштаб(a_codec);  acts_log.append(a_exec);  envs.step(a_exec)
— в среду уходит ровно записанный массив. В legacy-режиме руки шаг — строка
номинального чанка (численно путь K-9h). Тяжёлый план — один на 8 шагов.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def script_sha12():
    import k9h_multiarm_gate as k9h
    return k9h.file_sha12(os.path.abspath(__file__))


def expected_setup():
    import k15f_measure_subspace as kms
    return dict(kms.SETUP_EXPECTED, script_sha1=script_sha12())


def scale_exec(a_codec, max_act_q):
    """Та же арифметика, что в K-9h для чанка: рука × max|Q|, знак схвата
    инвертирован."""
    action = np.copy(a_codec)
    action[..., :-1] = action[..., :-1] * max_act_q[..., :-1]
    action[..., -1] = -action[..., -1]
    return action


def run(args, arm):
    """Блоки args.init_starts; запись JSON/npz (как K-9h) + журнал K-28."""
    import torch
    from torchvision.transforms.v2 import CenterCrop, Compose, Resize
    import k9h_multiarm_gate as k9h
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       dict_apply, get_cfg, process_state, prompt_template,
                       seed_everything)
    root = os.path.abspath(args.root)
    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt
    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))
    tf = Compose([CenterCrop(int(224 * 0.875)), Resize(224)])
    proc = arm.bar["proc"]
    autocast = torch.autocast("cuda", dtype=torch.float16)
    envs, task_desc, pos_off = args._envs, args._task_desc, args._pos_off
    blocks = [int(x) for x in str(args.init_starts).replace(",", " ")
              .split()]
    for blk in blocks:
        args.init_start = blk
        out_path = args.out.format(i0=blk)
        roll_seed = k9h.rollout_seed(args.seed, blk, args.rollout_seed_mode)
        print(f"\n  --- блок {blk}-{blk + args.n_envs - 1}, сид раскатки "
              f"{roll_seed} (режим {args.rollout_seed_mode})", flush=True)
        t0 = time.time()
        seed_everything(roll_seed)
        n = args.n_envs
        obs = envs.reset(options=[{"init_state_id": blk + i}
                                  for i in range(n)])
        reward = np.zeros(n)
        done = np.zeros(n, bool)
        dummy = np.array([[0, 0, 0, 0, 0, 0, -1]] * n)
        for _ in range(args.waiting_steps):
            obs, r_, done, _ = envs.step(dummy)
            reward = np.clip(reward + r_, 0, 1)

        def _h(parts):
            return hashlib.sha1(np.ascontiguousarray(
                np.concatenate(parts).astype(np.float32)).tobytes()
            ).hexdigest()[:16]
        init_hash = [_h([obs["state"][i].ravel(),
                         obs["agentview_image"][i].ravel() / 255.0])
                     for i in range(n)]
        init_hash_full = [_h([obs["state"][i].ravel(),
                              obs["agentview_image"][i].ravel() / 255.0,
                              obs["robot0_eye_in_hand_image"][i].ravel()
                              / 255.0]) for i in range(n)]
        calls = steps = 0
        succ_step = np.full(n, -1, np.int64)
        done_step = np.full(n, -1, np.int64)
        grip_pos_step = np.full(n, -1, np.int64)
        acts_log = []
        while not np.all(done) and steps < args.max_steps:
            s8 = np.asarray(process_state(obs["state"]), np.float64)
            state = ((s8 - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0)
            i1 = tf(torch.tensor(
                obs["agentview_image"][:, :, ::-1].copy()).permute(0, 3, 1, 2))
            i2 = tf(torch.tensor(
                obs["robot0_eye_in_hand_image"][:, :, ::-1].copy()
            ).permute(0, 3, 1, 2))
            image = torch.cat([i1, i2], dim=-1)
            msgs = []
            for i in range(n):
                m = prompt_template(
                    state[i], None, task_desc,
                    mode=cfg.MODEL.vla_processor.kwargs.mode,
                    action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                    action_token_len=cfg.MODEL.action_processor.token_len)
                m[1]["content"] = m[1]["content"][1:]
                msgs.append(m)
            texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
            batch = proc(text=texts,
                         images=[[image[i].numpy()] for i in range(n)],
                         return_tensors="pt", padding=True,
                         padding_side="left",
                         action_processor_kwargs={"embodiment_ids": 0})
            batch = dict_apply(lambda x: x.to(dev, dtype), batch)
            plan = arm.plan(batch, pos_off, s8, autocast)
            calls += 1
            for t in range(args.horizon):
                if np.all(done) or steps >= args.max_steps:
                    break
                s_now = np.asarray(process_state(obs["state"]), np.float64)
                a_codec = arm.step(plan, s_now, t, ~np.asarray(done, bool))
                a_t = scale_exec(np.asarray(a_codec, np.float32), max_act_q)
                acts_log.append(np.asarray(a_t, np.float32).copy())
                newly_grip = (a_t[:, -1] > 0) & (grip_pos_step < 0) & (~done)
                grip_pos_step[newly_grip] = steps
                obs, r_, done, _ = envs.step(a_t)
                prev_rw = reward
                reward = np.clip(reward + r_, 0, 1)
                newly_succ = (reward >= 1.0) & (prev_rw < 1.0)
                succ_step[newly_succ & (succ_step < 0)] = steps
                newly_done = np.asarray(done, bool) & (done_step < 0)
                done_step[newly_done] = steps
                steps += 1
        A = (np.stack(acts_log) if acts_log
             else np.zeros((0, n, 7), np.float32))

        def _ahash(i):
            end = int(done_step[i]) + 1 if done_step[i] >= 0 else len(A)
            return hashlib.sha1(np.ascontiguousarray(A[:end, i]).tobytes()
                                ).hexdigest()[:16]
        eps = [dict(success=bool(reward[i] >= 1.0), env_steps=steps,
                    policy_calls=calls, init_state_id=blk + i, env_index=i,
                    init_hash=init_hash[i], init_hash_full=init_hash_full[i],
                    rollout_seed=roll_seed, success_step=int(succ_step[i]),
                    done_step=int(done_step[i]),
                    grip_positive_step=int(grip_pos_step[i]),
                    own_steps=int(done_step[i] + 1 if done_step[i] >= 0
                                  else len(A)),
                    action_sha1=_ahash(i)) for i in range(n)]
        s = k9h.summarize(eps)
        print(f"  успех {sum(e['success'] for e in eps)}/{n}, шагов {steps}, "
              f"тяжёлых вызовов {calls}, {(time.time() - t0) / 60:.1f} мин",
              flush=True)
        write_block(args, arm, out_path, eps, s, A, done_step, roll_seed,
                    task_desc, pos_off)


def write_block(args, arm, out_path, eps, s, A, done_step, roll_seed,
                task_desc, pos_off):
    import k9h_multiarm_gate as k9h
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    sha = script_sha12()
    ap_ = os.path.splitext(out_path)[0] + ".actions.npz"
    tmp_a = ap_ + f".tmp.{os.getpid()}"
    with open(tmp_a, "wb") as fh_:
        np.savez_compressed(
            fh_, actions=np.asarray(A, np.float32),
            done_step=np.asarray(done_step, np.int64),
            init_state_id=np.asarray([e["init_state_id"] for e in eps],
                                     np.int64),
            action_sha1=np.asarray([e["action_sha1"] for e in eps],
                                   dtype="U16"),
            meta=json.dumps(dict(
                arm_label=args.arm_label, policy="k28", task_id=args.task_id,
                suite=args.task_suite, init_start=args.init_start,
                seed=args.seed, rollout_seed=roll_seed,
                horizon=args.horizon, script_sha1=sha,
                model_fingerprint=arm.meta.get("model_fingerprint")),
                ensure_ascii=False),
            **arm.take())
    with np.load(tmp_a, allow_pickle=True) as z_:
        if "actions" not in z_.files:
            raise SystemExit(f"{ap_}: массив actions не перечитался")
    os.replace(tmp_a, ap_)
    tmp_out = out_path + f".tmp.{os.getpid()}"
    argv = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    json.dump(dict(summary=s, episodes=eps, arm_label=args.arm_label,
                   run_tag=args.run_tag, policy="k28", levels=None, depth=24,
                   horizon=args.horizon, task_id=args.task_id,
                   suite=args.task_suite, pos_offset=pos_off,
                   ensemble=args.ensemble, init_start=args.init_start,
                   n_envs=args.n_envs, task_description=task_desc,
                   seed=args.seed, rollout_seed=roll_seed,
                   rollout_seed_mode=args.rollout_seed_mode,
                   max_steps=args.max_steps, waiting_steps=args.waiting_steps,
                   device=args.device, ckpt=args.ckpt, joint=arm.meta,
                   actions_npz=os.path.basename(ap_),
                   actions_npz_sha1=k9h.file_sha12(ap_),
                   actions_shape=list(np.asarray(A).shape), script_sha1=sha,
                   argv=argv),
              open(tmp_out, "w"), ensure_ascii=False, indent=1)
    json.load(open(tmp_out))
    os.replace(tmp_out, out_path)
    print(f"  сохранено: {out_path}", flush=True)


def selftest():
    q = np.array([0.9, 0.9, 0.9, 0.5, 0.5, 0.5, 1.0])
    a = np.array([[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 1.0]], np.float32)
    e = scale_exec(a, q)
    assert e.dtype == np.float32 and float(e[0, -1]) == -1.0
    # построчно = как для всего чанка в K-9h
    chunk = np.random.default_rng(0).uniform(-1, 1, (5, 20, 7)).astype(
        np.float32)
    whole = np.copy(chunk)
    whole[..., :-1] = whole[..., :-1] * q[..., :-1]
    whole[..., -1] = -whole[..., -1]
    rows = np.stack([scale_exec(chunk[:, t], q) for t in range(20)], 1)
    assert np.array_equal(rows, whole)
    toy_rollout_check()
    print("самопроверка k28_rollout_core пройдена: масштаб и знак схвата "
          "построчно = арифметике чанка K-9h побитово; игрушечная раскатка: "
          "каждый шаг получает process_state свежего obs (не состояние "
          "момента плана), в среду уходит ровно записанный массив, план — "
          "раз в 8 шагов, завершённые среды неактивны")
    return 0


def toy_rollout_check():
    """Цикл run() на фальшивых средах/утилитах: состояние шага t =
    process_state(obs после шага t − 1); envs.step получает записанный
    массив; план раз в 8 шагов; маска активности = ~done."""
    import tempfile
    import types
    n, T_done = 3, (13, 30, 21)
    fake_utils = types.ModuleType("utils")
    fake_utils.ACTION_Q01 = -np.ones(7) * 0.5
    fake_utils.ACTION_Q99 = np.ones(7) * 0.8
    fake_utils.STATE_Q01 = -np.ones(8)
    fake_utils.STATE_Q99 = np.ones(8)
    fake_utils.dict_apply = lambda f, d: {k: f(v) for k, v in d.items()}
    ns = types.SimpleNamespace
    fake_utils.get_cfg = lambda path: ns(
        TRAINING=ns(ckpt_dir=None), MODEL=ns(
            vlm=ns(kwargs=ns(pretrained_model_name_or_path=None)),
            vla_processor=ns(kwargs=ns(mode="m")),
            action_processor=ns(vocab_size=8, token_len=4)))
    fake_utils.process_state = lambda st: np.asarray(st, np.float64)[:, :8] \
        * 1.0
    fake_utils.prompt_template = lambda *a_, **k_: [dict(content="s"),
                                                    dict(content="xy")]
    fake_utils.seed_everything = lambda s: None
    tv = types.ModuleType("torchvision.transforms.v2")
    tv.CenterCrop = tv.Resize = lambda *a_, **k_: (lambda x: x)
    tv.Compose = lambda fs: (lambda x: x)

    class Envs:
        def __init__(self):
            self.t = -10
            self.st = np.zeros((n, 9))
            self.sent = []

        def obs(self):
            img = np.zeros((n, 4, 4, 3), np.uint8)
            return dict(state=self.st.copy(), agentview_image=img,
                        robot0_eye_in_hand_image=img)

        def reset(self, options):
            self.t = -10
            return self.obs()

        def step(self, a):
            a = np.asarray(a)
            if self.t >= 0:
                self.sent.append(a.copy())
            self.st = self.st + 0.01 * (1 + np.arange(9))[None] + \
                a[:, :1].sum() * 0 + (self.t + 11)
            self.t += 1
            done = np.array([self.t > d for d in T_done]) \
                if self.t >= 0 else np.zeros(n, bool)
            return self.obs(), done.astype(float) * 0, done, {}

    class Proc:
        def apply_chat_template(self, m, add_generation_prompt):
            return ["t"] * len(m)

        def __call__(self, **k):
            import torch
            return dict(input_ids=torch.zeros(n, 2))

    class FakeArm:
        def __init__(self):
            self.bar = dict(proc=Proc())
            self.meta = {}
            self.got, self.plans, self.envs = [], [], None
            self.active = []

        def plan(self, batch, po, s8, ac):
            self.plans.append((len(self.got), np.array(s8, copy=True)))
            return dict(i=len(self.plans))

        def step(self, plan, s_now, t, active):
            self.got.append((t, np.array(s_now, copy=True),
                             fake_utils.process_state(self.envs.st)))
            self.active.append(np.array(active, copy=True))
            return np.tile(np.r_[np.full(6, 0.1 * len(self.got)), 1.0],
                           (n, 1)).astype(np.float32)

        def take(self):
            return {}
    saved = {k: sys.modules.get(k) for k in ("utils",
                                             "torchvision.transforms.v2")}
    try:
        sys.modules["utils"] = fake_utils
        sys.modules["torchvision.transforms.v2"] = tv
        envs, arm = Envs(), FakeArm()
        arm.envs = envs
        with tempfile.TemporaryDirectory() as td:
            args = types.SimpleNamespace(
                root=td, device="cpu", dtype="float32", cfg_path="c",
                ckpt="ck", _envs=envs, _task_desc="d", _pos_off=0,
                init_starts="30", out=os.path.join(td, "b_{i0}.json"),
                seed=101, rollout_seed_mode="fixed", n_envs=n,
                waiting_steps=10, max_steps=600, horizon=8,
                arm_label="x", run_tag="r", task_suite="10", task_id=0,
                ensemble="off")
            run(args, arm)
            z = np.load(os.path.join(td, "b_30.actions.npz"))
            A = z["actions"]
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    N = len(arm.got)
    assert N == max(T_done) + 1 == len(A) == len(envs.sent)
    for t_, s_now, s_env in arm.got:
        assert np.array_equal(s_now, s_env), "шаг получил не свежее состояние"
    assert not np.array_equal(arm.got[1][1], arm.got[0][1])
    assert [i for i, _ in arm.plans] == list(range(0, N, 8))
    assert [t for t, _a, _b in arm.got] == [i % 8 for i in range(N)]
    for i, a_ in enumerate(envs.sent):
        assert np.array_equal(a_, A[i]), "в среду ушёл не записанный массив"
    act = np.stack(arm.active)
    want = np.stack([np.arange(N) <= d for d in T_done], 1)
    assert np.array_equal(act, want), "маска активности ≠ ~done"


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; запуск — через k28_harness.py")
