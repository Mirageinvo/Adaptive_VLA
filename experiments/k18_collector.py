#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-18: сборщик переходов из симулятора (волна = 5 сред одной задачи).

СРЕДЫ: собственные долго живущие CPU-воркеры (multiprocessing, forkserver,
как в libero2gym), создаются ДО инициализации CUDA. Каждый эпизод — свежий
объект среды (gym.make("libero-10-v0", task_id, image_size=224), как в
get_envs харнесса) и reset(init_state_id): повторный reset одного объекта
давал нехолодный старт (K-15f). Режим "fresh" дополнительно перезапускает
процессы воркеров перед каждой волной. Наблюдения склеиваются
gym.vector.utils.concatenate по пространству среды (state — float32), как
в AsyncVectorEnv без общей памяти. Успех — официальный check_success среды
после каждого шага; награда среды сверяется с ним.

ЦИКЛ ВОЛНЫ — дословно цикл rollout() харнесса k9h (seed_everything(101),
10 шагов ожидания [0,…,0,−1], нормировка состояния, кадры, подсказка,
процессор, float16, 8 команд на вызов, масштаб max|Q| и знак схвата,
переписывание done на каждом шаге, предел 600 шагов). q0 — a0 пути h18
(k15f_continuous_refine.plain_h18) того же замороженного backbone.

ОКНО: первое причинное событие схвата (Tracker K-16) и два следующих
вызова (Window K-17). На этих вызовах политика получает наблюдение
(enc(h18), a0, состояние с историей, возраст, позиция) и даёт исполняемую
поправку; вне окна — a0 побитово.

ПЕРЕХОДЫ: решения окна, начавшиеся до завершения среды. После позиции 0 и 1
при продолжении — награда 0 и следующее наблюдение; последнее принятое
решение терминально, награда — итоговый успех эпизода (после продолжения
q0). Действие в replay — весь выбранный вектор; маска исполненных шагов
чанка — отдельно.
"""
import hashlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

H = 8
DUMMY = [0, 0, 0, 0, 0, 0, -1]


# --- воркеры сред ---------------------------------------------------------------
def _worker(conn):
    import gym
    import libero2gym  # noqa: F401  регистрация libero-10-v0
    env = None
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "make":
                if env is not None:
                    env.close()
                    env = None
                env = gym.make("libero-10-v0", task_id=int(data),
                               image_size=224)
                conn.send(("ok", env.observation_space))
            elif cmd == "reset":
                conn.send(("ok", env.reset(init_state_id=int(data))))
            elif cmd == "step":
                obs, r, d, info = env.step(data)
                s = bool(env.unwrapped.check_success())
                conn.send(("ok", (obs, float(r), bool(d), s)))
            elif cmd == "close":
                if env is not None:
                    env.close()
                conn.send(("ok", None))
                break
            else:
                raise RuntimeError(f"команда {cmd!r}")
    except Exception as e:                                   # noqa: BLE001
        import traceback
        conn.send(("err", f"{e}\n{traceback.format_exc()}"))


class Sim:
    """n воркеров; reset(task, starts) — свежая среда на каждый эпизод."""

    def __init__(self, n, mode="persistent"):
        import multiprocessing as mp
        if mode not in ("persistent", "fresh"):
            raise ValueError(mode)
        self.n, self.mode = n, mode
        self.ctx = mp.get_context("forkserver")
        self.procs, self.conns = [], []
        self.space = None
        self._start()

    def _start(self):
        for _ in range(self.n):
            a, b = self.ctx.Pipe()
            p = self.ctx.Process(target=_worker, args=(b,), daemon=True)
            p.start()
            b.close()
            self.procs.append(p)
            self.conns.append(a)

    def _all(self, cmds):
        for c, x in zip(self.conns, cmds):
            c.send(x)
        out = []
        for i, c in enumerate(self.conns):
            st, v = c.recv()
            if st != "ok":
                raise SystemExit(f"воркер среды {i}: {v}")
            out.append(v)
        return out

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
                c.recv()
            except Exception:                                # noqa: BLE001
                pass
        for p in self.procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        self.procs, self.conns = [], []

    def reset(self, task, starts):
        from gym.vector.utils import concatenate, create_empty_array
        if self.mode == "fresh" and self.procs:
            self.close()
            self._start()
        spaces = self._all([("make", task)] * self.n)
        self.space = spaces[0]
        self.buf = create_empty_array(self.space, n=self.n, fn=np.zeros)
        obs = self._all([("reset", s) for s in starts])
        self.buf = concatenate(self.space, obs, self.buf)
        import copy
        return copy.deepcopy(self.buf)

    def step(self, actions):
        from gym.vector.utils import concatenate
        import copy
        res = self._all([("step", a) for a in actions])
        obs, r, d, s = zip(*res)
        self.buf = concatenate(self.space, obs, self.buf)
        return (copy.deepcopy(self.buf), np.array(r), np.array(d, np.bool_),
                np.array(s, np.bool_))


# --- хеши (формулы харнесса) -----------------------------------------------------
def init_hashes(obs, n):
    def _h(parts):
        return hashlib.sha1(np.ascontiguousarray(
            np.concatenate(parts).astype(np.float32)).tobytes()
        ).hexdigest()[:16]
    ih = [_h([obs["state"][i].ravel(),
              obs["agentview_image"][i].ravel() / 255.0]) for i in range(n)]
    ihf = [_h([obs["state"][i].ravel(),
               obs["agentview_image"][i].ravel() / 255.0,
               obs["robot0_eye_in_hand_image"][i].ravel() / 255.0])
           for i in range(n)]
    return ih, ihf


def action_sha(A, i, done_step):
    end = int(done_step) + 1 if int(done_step) >= 0 else len(A)
    return hashlib.sha1(np.ascontiguousarray(A[:end, i]).tobytes()
                        ).hexdigest()[:16]


def build_id(meta):
    """Идентификатор сборки q0 по её отпечаткам."""
    import json
    return hashlib.sha1(json.dumps(meta, sort_keys=True).encode()
                        ).hexdigest()[:12]


def reached(done_step, call):
    return bool(done_step < 0 or done_step >= H * (call - 1))


# --- базовая модель ----------------------------------------------------------------
class VLA:
    """Замороженная q0: подготовка входа как в харнессе, a0 и h18 пути
    plain_h18. Проверки гейта тождества — как у руки K-17."""

    def __init__(self, device, basis="data/k15f/basis_s0.pt",
                 identity_report=None, root="third_party/actioncodec",
                 cfg_path="config/eval/bar.yaml",
                 offset_table="data/pos_offset_table.json"):
        import json
        import torch
        import k15_context
        import k15d_policy as kdp
        import k15f_build_cache as kb
        import k15f_check_identity as ki
        import k15f_continuous_refine as kf
        import k15f_policy as kfp
        from torchvision.transforms.v2 import CenterCrop, Compose, Resize
        self.torch = torch
        tag = str(device).replace(":", "")
        identity_report = identity_report or \
            f"reports/k15f/identity_{tag}.json"
        g15a, _g = kdp.gate_paths(device)
        for f_ in (basis, identity_report, g15a, offset_table):
            if not os.path.exists(f_):
                raise SystemExit(f"нет {f_}")
        basis_sha = kb.sha_file(basis)
        rep = json.load(open(identity_report))
        prob = kfp.check_identity_report(
            rep, basis_sha=basis_sha, device=device,
            code=kb.sha_file(kf.__file__), gate_sha=kb.sha_file(ki.__file__))
        ck_b = torch.load(basis, map_location="cpu", weights_only=False)
        prob += ki.check_basis_ckpt(ck_b)
        if prob:
            raise SystemExit("q0 K-18 не собрана: " + "; ".join(prob))
        ctx = k15_context.build(kdp.context_namespace(device, g15a))
        self.ctx, self.model, self.dev, self.kf = ctx, ctx.model, ctx.dev, kf
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        frozen, _n, _e = k15_context.k15t.frozen_content_sha(
            self.model, torch, set())
        if frozen != rep.get("frozen_sha1"):
            raise SystemExit("замороженное не то, что в гейте тождества")
        self.frozen_sha = frozen
        r = os.path.abspath(root)
        if r not in sys.path:
            sys.path.insert(0, r)
        import utils as U
        self.U = U
        self.cfg = U.get_cfg(os.path.join(r, cfg_path))
        self.proc = ctx.proc
        self.tf = Compose([CenterCrop(int(224 * 0.875)), Resize(224)])
        self.offsets = json.load(open(offset_table))["offsets_by_suite"]["10"]
        self.autocast = torch.autocast("cuda", dtype=torch.float16)
        self.max_act_q = np.maximum(np.abs(U.ACTION_Q99),
                                    np.abs(U.ACTION_Q01))
        self.meta = dict(frozen_content_sha=frozen,
                         joint_sha1=ctx.joint_sha,
                         code_version=ctx.code_version, codec=ctx.codec_fp,
                         basis_sha1=basis_sha,
                         identity_report_sha1=kb.sha_file(identity_report),
                         k15a_gate_sha1=kb.sha_file(g15a),
                         offset_table_sha1=kb.sha_file(offset_table),
                         cfg_sha1=kb.sha_file(os.path.join(r, cfg_path)),
                         refine_module_sha1=kb.sha_file(kf.__file__))
        self.meta = json.loads(json.dumps(self.meta, default=str))
        self.build_id = build_id(self.meta)

    def task_desc(self, task):
        from libero.libero import benchmark
        return benchmark.get_benchmark_dict()["libero_10"]().get_task(
            int(task)).language

    def __call__(self, obs, task_desc, task):
        torch, U, cfg = self.torch, self.U, self.cfg
        n = len(obs["state"])
        state = ((U.process_state(obs["state"]) - U.STATE_Q01)
                 / (U.STATE_Q99 - U.STATE_Q01) * 2.0 - 1.0)
        i1 = self.tf(torch.tensor(
            obs["agentview_image"][:, :, ::-1].copy()).permute(0, 3, 1, 2))
        i2 = self.tf(torch.tensor(
            obs["robot0_eye_in_hand_image"][:, :, ::-1].copy()
        ).permute(0, 3, 1, 2))
        image = torch.cat([i1, i2], dim=-1)
        msgs = []
        for i in range(n):
            m = U.prompt_template(
                state[i], None, task_desc,
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
        pos_off = int(self.offsets[int(task)])
        with torch.no_grad(), self.autocast:
            v, p_ = self.model.build_inputs(position_offset=pos_off, **batch)
            _q0, a0, h18 = self.kf.plain_h18(
                self.model, vlm_inputs_embeds=v,
                attention_mask=batch.get("attention_mask"), position_ids=p_,
                decode=self.ctx.decode_fp32)
        return a0.float().cpu().numpy(), h18

    def seed(self, s):
        self.U.seed_everything(s)

    def process_state(self, st):
        return self.U.process_state(st)


class Encoder:
    """Замороженный G_pre.enc K-17 на устройстве малых сетей."""

    def __init__(self, path, device, sha_expected=None):
        import torch
        import k17_models as m17
        import k18_prepare as kp
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if sha_expected and kp.state_sha(ck["state"]) != sha_expected:
            raise SystemExit("кодировщик h18 не тот, что в конфигурации")
        M = m17.build(torch, int(ck["d_model"]), int(ck["n_pos"]))
        self.enc = M["G"]().enc
        self.enc.load_state_dict(ck["state"])
        self.enc = self.enc.to(device).eval()
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.torch, self.dev = torch, device

    def __call__(self, h18):
        with self.torch.no_grad():
            return self.enc(h18.to(self.dev).float()).cpu().numpy()


# --- волна -------------------------------------------------------------------
def run_wave(sim, vla, enc, task, starts, decide, *, scale, rho, env,
             visits=None, check=True):
    """decide(obs [m, OBS], keys [(start, visit, pos)]) -> dict(delta [m, 48],
    coef [m, k], noise_len [m], f [m]) или None (чистый q0 без окна).

    Возвращает (эпизоды, решения по средам, журнал, счётчики)."""
    import k16_models as km
    import k16_policy as kpol
    import k17_policy as k17p
    import k18_models as m18
    n = len(starts)
    visits = visits or [0] * n
    vla.seed(int(env["seed"]))
    t0 = time.time()
    obs = sim.reset(task, starts)
    desc = vla.task_desc(task)
    reward = np.zeros(n)
    done = np.zeros(n, bool)
    succ_any = np.zeros(n, bool)
    dummy = np.array([DUMMY] * n)
    for _ in range(int(env["waiting_steps"])):
        obs, r_, done, s_ = sim.step(dummy)
        reward = np.clip(reward + r_, 0, 1)
        succ_any |= s_
    ih, ihf = init_hashes(obs, n)
    calls = steps = 0
    succ_step = np.full(n, -1, np.int64)
    done_step = np.full(n, -1, np.int64)
    acts_log, a0_log, lv_log = [], [], []
    tr = kpol.Tracker(n)
    win = k17p.Window(n)
    dec = [[] for _ in range(n)]
    mism = 0
    while not np.all(done) and steps < int(env["max_steps"]):
        st = np.asarray(vla.process_state(obs["state"]), np.float64)
        a0, h18 = vla(obs, desc, task)
        ev, trig, hist, mask = tr.step(st, -a0[:, :H, 6],
                                       None if decide is None else "event",
                                       [-1] * n)
        _new, active, wpos = win.step(trig, tr.call)
        d_exec = np.zeros((n, m18.D_EXEC))
        if decide is not None and active.any():
            idx = np.nonzero(active)[0]
            sf = km.state_features(st, hist, mask,
                                   np.full(n, 8 * (tr.call - 1)), True)
            ob = m18.make_obs(enc(h18[idx]), a0[idx], sf[idx], wpos[idx])
            keys = [(int(starts[i]), int(visits[i]), int(wpos[i]))
                    for i in idx]
            out = decide(ob, keys)
            d_exec[idx] = out["delta"]
            for j, i in enumerate(idx):
                dec[i].append(dict(call=tr.call, pos=int(wpos[i]), obs=ob[j],
                                   delta=out["delta"][j],
                                   coef=out["coef"][j],
                                   noise_len=float(out["noise_len"][j]),
                                   f=float(out["f"][j])))
        act = m18.apply_exec(a0, d_exec, scale)
        if check:
            nact = ~active if decide is not None else np.ones(n, bool)
            if not np.array_equal(act[nact], a0[nact]):
                raise SystemExit("вне окна действие не равно a0")
            if not (np.array_equal(act[..., 6], a0[..., 6])
                    and np.array_equal(act[:, H:], a0[:, H:])):
                raise SystemExit("схват или хвост изменены")
            if (np.linalg.norm(d_exec, axis=1) > rho * (1 + 1e-6)).any():
                raise SystemExit("норма поправки выше ρ")
            got = ((act[:, :H, :6].astype(np.float64)
                    - a0[:, :H, :6]) * np.asarray(scale)[:6]).reshape(n, -1)
            if not np.allclose(got, d_exec, atol=1e-5):
                raise SystemExit("исполненная поправка ≠ выбранной")
            if not np.isfinite(act).all():
                raise SystemExit("действие не конечно")
        tr.after(st, -act[:, :H, 6])
        a0_log.append(a0[:, :H].astype(np.float32))
        lv_log.append(act[:, :H].astype(np.float32))
        calls += 1
        action = np.copy(act)
        action[..., :-1] = action[..., :-1] * vla.max_act_q[..., :-1]
        action[..., -1] = -action[..., -1]
        for t in range(int(env["horizon"])):
            if np.all(done) or steps >= int(env["max_steps"]):
                break
            a_t = np.asarray(action[:, t])
            acts_log.append(np.asarray(a_t, np.float32).copy())
            obs, r_, done, s_ = sim.step(a_t)
            mism += int(((r_ >= 1.0) != s_).sum())
            prev = reward
            reward = np.clip(reward + r_, 0, 1)
            succ_any |= s_
            newly = (reward >= 1.0) & (prev < 1.0)
            succ_step[newly & (succ_step < 0)] = steps
            nd = np.asarray(done, bool) & (done_step < 0)
            done_step[nd] = steps
            steps += 1
    if mism:
        raise SystemExit(f"награда среды расходится с check_success "
                         f"({mism} шагов)")
    if not np.array_equal(succ_any, reward >= 1.0):
        raise SystemExit("успех по check_success ≠ успеху по награде")
    A = np.stack(acts_log) if acts_log else np.zeros((0, n, 7), np.float32)
    wait = int(env["waiting_steps"])
    own = np.where(done_step >= 0, done_step + 1, steps)
    live = int((own + wait).sum())
    total = int(n * (steps + wait))
    eps = [dict(task=int(task), init_state_id=int(starts[i]),
                visit=int(visits[i]), success=bool(reward[i] >= 1.0),
                success_step=int(succ_step[i]), done_step=int(done_step[i]),
                own_steps=int(own[i]), init_hash=ih[i], init_hash_full=ihf[i],
                action_sha1=action_sha(A, i, done_step[i]),
                window_start=int(win.start_call[i]),
                first_event=int(tr.first_event[i]))
           for i in range(n)]
    log = dict(a0=np.stack(a0_log), act=np.stack(lv_log), actions=A,
               done_step=done_step)
    counts = dict(commands_live=live, commands_after_done=total - live,
                  commands_total=total, calls=calls, steps=steps,
                  seconds=time.time() - t0)
    return eps, dec, log, counts


def transitions(eps, dec, task):
    """Решения окна -> RL-переходы (см. шапку модуля)."""
    out = []
    for i, e in enumerate(eps):
        ds = int(e["done_step"])
        valid = [d for d in dec[i] if reached(ds, d["call"])]
        for j, d in enumerate(valid):
            last = j == len(valid) - 1
            c0 = H * (d["call"] - 1)
            mask = np.array([ds < 0 or c0 + t <= ds for t in range(H)])
            out.append(dict(
                obs=d["obs"], delta=d["delta"], coef=d["coef"],
                rew=float(e["success"]) if last else 0.0,
                next=d["obs"] if last else valid[j + 1]["obs"],
                done=1.0 if last else 0.0, pos=d["pos"], task=int(task),
                start=e["init_state_id"], visit=e["visit"],
                noise_len=d["noise_len"], f=d["f"], exec_mask=mask))
    return out


# --- самопроверка: игрушечные среда и модель ----------------------------------------
class FakeSim:
    """Детерминированная игрушка: success, когда сумма x-команд превысит
    порог; done при успехе или на шаге 600 (включая ожидание)."""

    def __init__(self, n, thr=3.0):
        self.n, self.thr = n, thr

    def reset(self, task, starts):
        self.t = np.zeros(self.n, int)
        self.x = np.asarray(starts, float) * 0.01
        self.task = task
        return self._obs()

    def _obs(self):
        st = np.zeros((self.n, 9), np.float32)
        st[:, 0:2] = 0.02
        st[:, 2] = self.x
        st[:, 5] = 1.0
        img = np.zeros((self.n, 4, 4, 3), np.uint8)
        return dict(state=st, agentview_image=img,
                    robot0_eye_in_hand_image=img)

    def step(self, a):
        a = np.asarray(a)
        self.t += 1
        self.x = self.x + a[:, 0] * 0.01
        s = self.x > self.thr
        d = s | (self.t >= 600)
        return self._obs(), s.astype(float), d, s


class FakeVLA:
    def __init__(self, tag="toy"):
        self.max_act_q = np.array([.9375, .91, .9375, .2036, .2636, .375, 1.])
        self.meta = dict(toy=tag)
        self.build_id = build_id(self.meta)

    def seed(self, s):
        pass

    def task_desc(self, t):
        return "toy"

    def process_state(self, st):
        st = np.asarray(st, np.float64)
        return np.concatenate([st[:, 2:5], np.zeros((len(st), 3)),
                               st[:, :2]], 1)

    def __call__(self, obs, desc, task):
        import torch
        n = len(obs["state"])
        x = obs["state"][:, 2].astype(np.float64)
        a0 = np.zeros((n, 10, 7), np.float32)
        a0[..., 0] = 0.5
        call = int(round(x[0] * 1000)) % 7
        a0[:, :, 6] = 1.0 if call < 3 else -1.0     # смены схвата
        return a0, torch.zeros(n, 16, 768)


def selftest():
    import k18_models as m18
    scale = FakeVLA().max_act_q
    env = dict(seed=101, waiting_steps=10, max_steps=600, horizon=8)
    rho = 0.7068
    enc = lambda h: np.zeros((len(h), 512), np.float32)   # noqa: E731
    sim, vla = FakeSim(3), FakeVLA()
    e_q0, d_q0, l_q0, c_q0 = run_wave(sim, vla, enc, 0, [0, 1, 2], None,
                                      scale=scale, rho=rho, env=env)
    assert not any(d_q0) and np.array_equal(l_q0["act"], l_q0["a0"])

    def zero(ob, keys):
        m = len(ob)
        return dict(delta=np.zeros((m, 48)), coef=np.zeros((m, 4)),
                    noise_len=np.zeros(m), f=np.ones(m))
    e_z, d_z, l_z, c_z = run_wave(sim, vla, enc, 0, [0, 1, 2], zero,
                                  scale=scale, rho=rho, env=env)
    # нулевая политика побитово = q0
    assert [e["action_sha1"] for e in e_z] == [e["action_sha1"] for e in e_q0]
    assert c_z == {**c_z, "commands_live": c_q0["commands_live"]}
    assert all(len(d) == 3 for d in d_z), [len(d) for d in d_z]
    trs = transitions(e_z, d_z, 0)
    assert len(trs) == 9
    for i in range(3):
        t3 = trs[3 * i:3 * i + 3]
        assert [t["done"] for t in t3] == [0, 0, 1]
        assert t3[2]["rew"] == float(e_z[i]["success"])
        assert np.array_equal(t3[0]["next"], t3[1]["obs"])
    # ненулевая политика: поправка исполнена и записана
    def push(ob, keys):
        m = len(ob)
        d = np.zeros((m, 48))
        d[:, 0::6] = 0.05
        return dict(delta=d, coef=np.ones((m, 4)), noise_len=np.zeros(m),
                    f=np.ones(m))
    e_p, d_p, l_p, c_p = run_wave(sim, vla, enc, 0, [0, 1, 2], push,
                                  scale=scale, rho=rho, env=env)
    assert not np.array_equal(l_p["act"], l_p["a0"])
    # раннее завершение: решение после done не принимается, терминальная
    # награда — последнему принятому
    e2 = [dict(done_step=20, success=True, init_state_id=0, visit=0)]
    d2 = [[dict(call=c, pos=k, obs=np.full(3, c), delta=np.zeros(48),
                coef=np.zeros(4), noise_len=0.0, f=1.0)
           for k, c in enumerate((2, 3, 4))]]
    t2 = transitions(e2, d2, 0)
    assert len(t2) == 2 and t2[1]["done"] == 1 and t2[1]["rew"] == 1.0
    assert t2[1]["exec_mask"].tolist() == [True] * 5 + [False] * 3
    # без события — переходов нет; таймаут — награда 0
    assert transitions([dict(done_step=-1, success=False, init_state_id=0,
                             visit=0)], [[]], 0) == []
    e3 = [dict(done_step=-1, success=False, init_state_id=0, visit=0)]
    t3 = transitions(e3, d2, 0)
    assert len(t3) == 3 and t3[2]["rew"] == 0.0 and t3[2]["exec_mask"].all()
    assert m18.OBS_DIM == len(d_z[0][0]["obs"])
    print("самопроверка k18_collector пройдена: цикл волны (q0 и нулевая "
          "политика побитово равны), окно из трёх решений, переходы "
          "(терминальная награда после продолжения q0, раннее завершение, "
          "маска чанка, без события, таймаут)")
    return 0


if __name__ == "__main__":
    sys.exit(selftest())
