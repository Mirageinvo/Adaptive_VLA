#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-29: волна сбора (5 сред одной задачи), переходы PPO, учёт шагов и
журнал бюджета.

Среды — k18_collector.Sim (постоянные воркеры, свежая среда gym.make на
каждый эпизод, холодный старт). Цикл волны — дословно k9h: seed_everything
(101), reset(init_state_id), 10 шагов ожидания [0,…,0,−1], на каждой границе
чанка — текущие наблюдения, один тяжёлый проход B = 5, оба декодера, шум
(при сборе) и до 8 команд; масштаб и инверсия схвата — k29_policy.scale_exec;
цикл идёт, пока не все текущие raw done и шагов < 600 (как в харнессе).
Защёлка: первый done (done_step) и первый успех (награда ≥ 1) фиксируются;
последующий False не «оживляет» среду. Переход PPO — чанк среды, активной на
его начале (done_step < 0); k — число исполненных активных шагов этого чанка
(до шага первого done включительно); R — успех эпизода (одна единица).
Завершённые среды продолжают получать команды до конца волны; все
физические шаги (ожидание и хвосты) идут в бюджет: n·(10 + шагов волны).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k29_dist as kd  # noqa: E402

N8 = kd.N_EXEC
ENV = dict(seed=101, waiting_steps=10, max_steps=600, horizon=8, n_envs=5)
WAVE_MAX = ENV["n_envs"] * (ENV["max_steps"] + ENV["waiting_steps"])   # 3050
CYCLE_TASKS = (0, 6, 8)
CYCLE_MAX = WAVE_MAX * len(CYCLE_TASKS)                                   # 9150
DUMMY = [0, 0, 0, 0, 0, 0, -1]


def run_wave(sim, inputs, policy, task, starts, *, noise_gen=None,
             record=True, check=True):
    """-> (эпизоды, переходы | None, журнал, счётчики)."""
    import k18_collector as kc
    import k25_policy as kp25
    n = len(starts)
    t0 = time.time()
    inputs.seed(ENV["seed"])
    obs = sim.reset(task, list(starts))
    desc = inputs.task_desc(task)
    reward = np.zeros(n)
    done = np.zeros(n, bool)
    succ_any = np.zeros(n, bool)
    dummy = np.array([DUMMY] * n)
    for _ in range(ENV["waiting_steps"]):
        obs, r_, done, s_ = sim.step(dummy)
        reward = np.clip(reward + r_, 0, 1)
        succ_any |= s_
    ih, ihf = kc.init_hashes(obs, n)
    steps = calls = mism = 0
    succ_step = np.full(n, -1, np.int64)
    done_step = np.full(n, -1, np.int64)
    acts_log, chunks = [], []
    tr = []
    while not np.all(done) and steps < ENV["max_steps"]:
        batch, po, s8 = inputs(obs, desc, task)
        active0 = done_step < 0
        noise = None if noise_gen is None else \
            kd.sample_noise(noise_gen, n).astype(np.float32)
        out = policy.chunk(batch, po, s8, noise)
        if check:
            pr = kp25.counters_problems(out["counters"])
            if pr:
                raise SystemExit("тяжёлый проход: " + "; ".join(pr))
        calls += 1
        a_exec = inputs.scale(out["a_codec"])
        start = steps
        kcnt = np.zeros(n, np.int64)
        for t in range(ENV["horizon"]):
            if np.all(done) or steps >= ENV["max_steps"]:
                break
            a_t = np.asarray(a_exec[:, t], np.float32).copy()
            alive = done_step < 0
            acts_log.append(a_t.copy())
            obs, r_, done, s_ = sim.step(a_t)
            mism += int(((r_ >= 1.0) != s_).sum())
            prev = reward
            reward = np.clip(reward + r_, 0, 1)
            succ_any |= s_
            newly = (reward >= 1.0) & (prev < 1.0)
            succ_step[newly & (succ_step < 0)] = steps
            nd = np.asarray(done, bool) & (done_step < 0)
            done_step[nd] = steps
            kcnt[alive] += 1
            steps += 1
        chunks.append(dict(a_codec=out["a_codec"], active=active0, k=kcnt))
        if record and active0.any():
            idx = np.nonzero(active0)[0]
            if check and ((kcnt[idx] < 1) | (kcnt[idx] > N8)).any():
                raise SystemExit(f"k вне 1…8: {kcnt[idx]}")
            F = {k_: v[idx].detach().cpu() for k_, v in out["F"].items()}
            tr.append(dict(F=F, s11n=out["s11n"][idx].detach().cpu(),
                           Y=out["Y"][idx].detach().cpu(),
                           mu=out["mu"][idx].detach().cpu(),
                           mu_ref=out["mu_ref"][idx].detach().cpu(),
                           k=kcnt[idx].copy(),
                           elapsed=np.full(len(idx), start),
                           env=idx.copy(), chunk=np.full(len(idx), calls - 1)))
    if mism:
        raise SystemExit(f"награда среды расходится с check_success ({mism})")
    if not np.array_equal(succ_any, reward >= 1.0):
        raise SystemExit("успех по check_success ≠ успеху по награде")
    A = np.stack(acts_log) if acts_log else np.zeros((0, n, 7), np.float32)
    wait = ENV["waiting_steps"]
    own = np.where(done_step >= 0, done_step + 1, steps)
    eps = [dict(task=int(task), init_state_id=int(starts[i]), env_index=i,
                success=bool(reward[i] >= 1.0), success_step=int(succ_step[i]),
                done_step=int(done_step[i]), own_steps=int(own[i]),
                env_steps=steps, policy_calls=calls,
                init_hash=ih[i], init_hash_full=ihf[i],
                action_sha1=kc.action_sha(A, i, done_step[i]))
           for i in range(n)]
    R = np.array([float(e["success"]) for e in eps])
    buf = pack(tr, R, task, starts) if record else None
    counts = dict(physical=int(n * (steps + wait)), waiting=int(n * wait),
                  policy_steps=int(own.sum()), tail=int(n * steps - own.sum()),
                  steps=int(steps), calls=int(calls), episodes=n,
                  transitions=0 if buf is None else int(len(buf["k"])),
                  seconds=time.time() - t0)
    log = dict(actions=A, done_step=done_step, chunks=chunks)
    return eps, buf, log, counts


def pack(tr, R, task, starts):
    """Переходы волны -> dict тензоров/массивов (CPU)."""
    import torch
    if not tr:
        return None
    F = {k: torch.cat([t["F"][k] for t in tr]) for k in tr[0]["F"]}
    env = np.concatenate([t["env"] for t in tr])
    out = dict(F=F, s11n=torch.cat([t["s11n"] for t in tr]),
               Y=torch.cat([t["Y"] for t in tr]),
               mu=torch.cat([t["mu"] for t in tr]),
               mu_ref=torch.cat([t["mu_ref"] for t in tr]),
               k=np.concatenate([t["k"] for t in tr]),
               elapsed=np.concatenate([t["elapsed"] for t in tr]),
               env=env, chunk=np.concatenate([t["chunk"] for t in tr]),
               init=np.asarray(starts)[env], task=np.full(len(env), task),
               R=R[env].astype(np.float32))
    return out


def concat(bufs):
    """Несколько волн/циклов -> один batch (эпизоды получают сквозной id)."""
    import torch
    bufs = [b for b in bufs if b is not None]
    out = dict(F={k: torch.cat([b["F"][k] for b in bufs])
                  for k in bufs[0]["F"]})
    for k in ("s11n", "Y", "mu", "mu_ref"):
        out[k] = torch.cat([b[k] for b in bufs])
    ep, off = [], 0
    for b in bufs:
        ep.append(b["env"] + off)
        off += int(b["env"].max()) + 1
    for k in ("k", "elapsed", "env", "chunk", "init", "task", "R"):
        out[k] = np.concatenate([b[k] for b in bufs])
    out["episode"] = np.concatenate(ep)
    return out


# --- журнал бюджета -------------------------------------------------------------
class Ledger:
    """Атомарный учёт физических шагов. Резерв цикла записывается ДО его
    начала; после цикла заменяется фактическим расходом. При загрузке с
    неосвобождённым резервом он считается потраченным (консервативно)."""

    def __init__(self, path, cap):
        self.path, self.cap = path, int(cap)
        if os.path.exists(path):
            self.d = json.load(open(path, encoding="utf-8"))
            if int(self.d["cap"]) != self.cap:
                raise SystemExit(f"{path}: другой кап {self.d['cap']}")
            if self.d["reserved"]:
                self.d["crashes"].append(dict(
                    reserved=self.d["reserved"], what=self.d["reserved_for"],
                    time=time.strftime("%Y-%m-%dT%H:%M:%S")))
                self.d["train_spent"] += self.d["reserved"]
                self.d["reserved"], self.d["reserved_for"] = 0, None
                self._save()
        else:
            self.d = dict(cap=self.cap, train_spent=0, reserved=0,
                          reserved_for=None, other={}, crashes=[], cycles=[])
            self._save()

    def _save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.d, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def remaining(self):
        return self.cap - self.d["train_spent"] - self.d["reserved"]

    def can_cycle(self):
        return self.remaining() >= CYCLE_MAX

    def reserve(self, what, amount=CYCLE_MAX):
        if self.d["reserved"]:
            raise SystemExit("резерв уже открыт")
        if self.remaining() < amount:
            raise SystemExit("резерв не помещается в бюджет")
        self.d["reserved"], self.d["reserved_for"] = int(amount), what
        self._save()

    def commit(self, actual, info=None):
        if not self.d["reserved"]:
            raise SystemExit("нет открытого резерва")
        if actual > self.d["reserved"]:
            raise SystemExit(f"расход {actual} > резерва {self.d['reserved']}")
        self.d["train_spent"] += int(actual)
        self.d["cycles"].append(dict(what=self.d["reserved_for"],
                                     actual=int(actual), **(info or {})))
        self.d["reserved"], self.d["reserved_for"] = 0, None
        self._save()

    def add_other(self, kind, amount):
        """Проверки, оценки, smoke — вне training-капа, но в отчёте."""
        self.d["other"][kind] = int(self.d["other"].get(kind, 0)) + int(amount)
        self._save()


# --- игрушки для самопроверок -----------------------------------------------------
class FakeSim:
    """x растёт с командой; успех при x > thr; done среды 0 «мигает»
    (True на шаге, затем False — как бывает в LIBERO)."""

    def __init__(self, n=5, thr=1.0, flicker=True):
        self.n, self.thr, self.flicker = n, thr, flicker

    def reset(self, task, starts):
        self.t = 0
        self.x = np.asarray(starts, float) * 0.01
        return self._obs()

    def _obs(self):
        st = np.zeros((self.n, 9), np.float32)
        st[:, 0] = self.x
        img = np.zeros((self.n, 4, 4, 3), np.uint8)
        return dict(state=st, agentview_image=img,
                    robot0_eye_in_hand_image=img)

    def step(self, a):
        a = np.asarray(a)
        self.t += 1
        self.x = self.x + np.abs(a[:, 0]) * 0.02
        s = self.x > self.thr
        if self.flicker and self.t == 25:
            s[0] = True       # успех на одном шаге (done = успех), затем False
        d = s.copy()
        return self._obs(), s.astype(float), d, s


class FakeInputs:
    max_act_q = np.array([0.9] * 6 + [1.0])

    def seed(self, s):
        pass

    def task_desc(self, t):
        return "toy"

    def __call__(self, obs, desc, task):
        return dict(x=obs["state"][:, 0].copy()), 0, \
            np.asarray(obs["state"][:, :8], np.float64)

    def scale(self, a):
        import k29_policy as kp
        return kp.scale_exec(a, self.max_act_q)


class FakePolicy:
    """μ среды i зависит только от её собственного состояния."""

    def __init__(self, gain=0.5):
        self.gain = gain

    def chunk(self, batch, po, state8, noise=None):
        import torch
        x = torch.as_tensor(batch["x"], dtype=torch.float32)
        n = len(x)
        mu = (self.gain + 0.1 * x)[:, None, None].expand(n, N8, 6).clone()
        Y = mu if noise is None else mu + torch.as_tensor(noise)
        arm = torch.tanh(Y)
        a = torch.cat([arm, torch.ones(n, N8, 1)], -1).numpy()
        F = dict(x0=x[:, None, None].expand(n, 4, 3).clone(),
                 x1=torch.zeros(n, 8, 3), x2=torch.zeros(n, 16, 3))
        cnt = dict(vision=1, text=[1] * 24, expert=[1] * 24,
                   active=[4] * 8 + [12] * 8 + [28] * 8)
        return dict(F=F, s11n=torch.zeros(n, 11), counters=cnt, a_codec=a,
                    mu=mu, mu_ref=mu.clone(), Y=Y)


def selftest():
    import tempfile
    sim = FakeSim()
    eps, buf, log, c = run_wave(sim, FakeInputs(), FakePolicy(), 0,
                                [0, 3, 7, 11, 19],
                                noise_gen=np.random.default_rng(0))
    ds = log["done_step"]
    # защёлка: «мигнувший» успех среды 0 (t = 25 с учётом 10 шагов
    # ожидания — шаг политики 14) фиксирует успех и done; последующий False
    # её не «оживляет»: переходов после шага 14 нет
    assert ds[0] == 14 and eps[0]["done_step"] == 14 and \
        eps[0]["success"] and eps[0]["success_step"] == 14 and \
        eps[0]["own_steps"] == 15
    # переходы только от сред, активных в начале чанка; k = 1…8
    k, envs, ch = buf["k"], buf["env"], buf["chunk"]
    assert ((k >= 1) & (k <= N8)).all()
    for i in range(5):
        kk = k[envs == i]
        own = eps[i]["own_steps"]
        assert kk.sum() == own, (i, kk, own)          # k суммируется в own
        assert (ch[envs == i] * N8 < own).all()
        assert (buf["elapsed"][envs == i] == ch[envs == i] * N8).all()
    # награда: одна единица на эпизод, R перехода = успех эпизода
    for i in range(5):
        assert set(buf["R"][envs == i].tolist()) == {float(eps[i]["success"])}
    # бюджет: все физические шаги, включая ожидание и хвосты
    assert c["physical"] == 5 * (10 + c["steps"])
    assert c["physical"] == c["waiting"] + c["policy_steps"] + c["tail"]
    # исполненное = записанному (масштаб K-9h)
    A = log["actions"]
    q = FakeInputs.max_act_q
    a0 = log["chunks"][0]["a_codec"][:, 0]
    assert np.allclose(A[0][:, :6], a0[:, :6] * q[:6]) and \
        np.array_equal(A[0][:, 6], -a0[:, 6])
    # независимость от соседей: иная судьба среды 4 не меняет переходы среды 1
    sim2 = FakeSim()
    e2, b2, _l, _c = run_wave(sim2, FakeInputs(), FakePolicy(), 0,
                              [0, 3, 7, 11, 2],
                              noise_gen=np.random.default_rng(0))
    m1, m2 = buf["env"] == 1, b2["env"] == 1
    import torch
    n_ = min(m1.sum(), m2.sum())
    assert torch.equal(buf["Y"][m1][:n_], b2["Y"][m2][:n_])
    assert torch.equal(buf["mu"][m1][:n_], b2["mu"][m2][:n_])
    # журнал бюджета: резерв, фактический расход, авария = резерв потрачен
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ledger.json")
        L = Ledger(p, 30000)
        L.reserve("c1")
        L.commit(5000)
        assert L.remaining() == 25000
        L.reserve("c2")                      # «авария» внутри цикла
        L2 = Ledger(p, 30000)
        assert L2.d["train_spent"] == 5000 + CYCLE_MAX and L2.d["crashes"]
        assert L2.remaining() == 25000 - CYCLE_MAX
        assert L2.can_cycle() and not Ledger(p, 30000).d["reserved"]
        L2.reserve("c3")
        try:
            L2.commit(CYCLE_MAX + 1)
            raise AssertionError("расход больше резерва принят")
        except SystemExit:
            pass
    cc = concat([buf, b2])
    assert len(set(cc["episode"][:len(buf["k"])])) == 5 and \
        cc["episode"].max() == 9
    print(f"самопроверка k29_collect пройдена: защёлка успеха/done (мигнувший "
          f"done не оживляет среду), переходы только активных сред, Σk = own_steps, "
          f"одна награда на эпизод, бюджет = n·(10 + шаги) с хвостами, "
          f"исполненное = записанному, μ среды не зависит от соседей, журнал "
          f"бюджета (резерв, авария, отказ перерасхода); волна {c}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
