#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-31: эпизод B = 1, выбор развилки, ветви «свежая среда + повтор
исполненного префикса», проверка noop и журнал бюджета.

Эпизод (run_episode): k18_collector.Sim(1) — свежая среда gym.make на
эпизод; seed_everything(101), reset(init), 10 шагов ожидания [0,…,0,−1];
затем (если задан) побитовый повтор записанных исполненных команд
префикса (tau шагов, tau кратно 8); затем вызовы политики на границах
чанков (до 8 команд, масштаб/инверсия схвата — как K-9h). Первый done и
первый успех защёлкиваются; после первого done исполнение не продолжается
(мигающий raw done среду не «оживляет»); лимит — 600 управляющих шагов
всего (остаток горизонта 600 − tau, не новые 600). Награда сверяется с
check_success. На каждой границе — хеш наблюдения (state, обе камеры:
dtype/форма/байты).
Развилка: на границе tau вместо решения политики исполняется заданный
чанк (кандидат или записанная команда reference для noop); наблюдение
развилки обязано совпасть с закреплённым хешем, иначе технический отказ.
Reservoir (поток fork): k-я активная граница становится развилкой с
вероятностью 1/k — ровно одна, равномерно, без исходов.
Случай (run_case): reference -> развилка и пул -> noop (обязан совпасть с
reference после tau: действия, хеши границ, done/success, шаги) -> все
разрешённые кандидаты в фиксированной перестановке (поток perm). Каждая
ветвь — новая среда; резерв 610 шагов до начала, после — фактический
расход; запись ветви атомарна, повторно не начисляется.
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

import k31_candidates as kc  # noqa: E402

ENV = dict(seed=101, waiting_steps=10, max_steps=600, horizon=8, n_envs=1)
EXEC_MAX = ENV["max_steps"] + ENV["waiting_steps"]          # 610
DUMMY = [0, 0, 0, 0, 0, 0, -1]
N8 = kc.N8


class Mismatch(Exception):
    """Технический отказ воспроизведения (развилка/noop/префикс)."""


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def obs_hash(obs):
    h = hashlib.sha1()
    for k in ("state", "agentview_image", "robot0_eye_in_hand_image"):
        a = np.ascontiguousarray(np.asarray(obs[k]))
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def actions_sha(A):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(A, np.float32))
                        .tobytes()).hexdigest()


# --- эпизод -----------------------------------------------------------------------
def run_episode(sim, io, agent, task, init, *, prefix=None, fork=None,
                reservoir=None, capture=None):
    """io: seed(s), task_desc(t), prep(obs, desc, task) -> (batch, po, s8),
    scale(a_codec [n,8,7]) -> исполняемые; agent: begin, decide, executed.
    prefix [tau,7] — записанные исполненные команды; fork — dict(tau, hash,
    a_codec [8,7]); reservoir — np.random.Generator (поток fork);
    capture(agent) -> полезная нагрузка развилки."""
    import k18_collector as k18c
    t0 = time.time()
    io.seed(ENV["seed"])
    obs = sim.reset(task, [init])
    desc = io.task_desc(task)
    agent.begin(task, init)
    reward, mism = 0.0, 0
    succ_step = done_step = -1
    wait_done = False
    dummy = np.array([DUMMY])
    for _ in range(ENV["waiting_steps"]):
        obs, r_, d_, s_ = sim.step(dummy)
        mism += int((float(r_[0]) >= 1.0) != bool(s_[0]))
        reward = min(max(reward + float(r_[0]), 0.0), 1.0)
        wait_done |= bool(d_[0])
    if wait_done or reward >= 1.0:
        raise Mismatch(f"t{task} i{init}: среда завершилась во время ожидания")
    ih, ihf = k18c.init_hashes(obs, 1)
    acts, calls = [], []
    steps = 0

    def step(a):
        nonlocal obs, reward, succ_step, done_step, steps, mism
        acts.append(np.asarray(a, np.float32).copy())
        obs, r_, d_, s_ = sim.step(np.asarray(a, np.float32)[None])
        mism += int((float(r_[0]) >= 1.0) != bool(s_[0]))
        prev = reward
        reward = min(max(reward + float(r_[0]), 0.0), 1.0)
        if reward >= 1.0 and prev < 1.0 and succ_step < 0:
            succ_step = steps
        if bool(d_[0]) and done_step < 0:
            done_step = steps
        steps += 1
    if prefix is not None:
        for a in np.asarray(prefix, np.float32):
            if done_step >= 0:
                raise Mismatch("префикс: среда завершилась до развилки")
            step(a)
    forked = False
    k_res = 0
    chosen = None
    while done_step < 0 and steps < ENV["max_steps"]:
        if steps % N8:
            raise Mismatch(f"граница вызова не кратна 8 ({steps})")
        call = steps // N8 + 1
        h = obs_hash(obs)
        if fork is not None and steps == int(fork["tau"]) and not forked:
            if h != fork["hash"]:
                raise Mismatch(f"t{task} i{init}: наблюдение развилки tau "
                               f"{steps} ≠ закреплённому")
            a_codec = np.asarray(fork["a_codec"], np.float32)
            rec, forced = dict(forced=True, call=call), True
            forked = True
        else:
            agent.call = call - 1
            batch, po, s8 = io.prep(obs, desc, task)
            a_codec, rec = agent.decide(batch, po, s8, steps)
            forced = False
        if reservoir is not None and not forced:
            k_res += 1
            if reservoir.random() < 1.0 / k_res:
                chosen = dict(call=call, tau=steps, hash=h,
                              a_codec=np.asarray(a_codec, np.float32).copy(),
                              payload=capture(agent) if capture else None)
        a_exec = io.scale(np.asarray(a_codec, np.float32)[None])[0]
        k = 0
        for t in range(ENV["horizon"]):
            if done_step >= 0 or steps >= ENV["max_steps"]:
                break
            step(a_exec[t])
            k += 1
        if not forced:
            agent.executed(k)
        calls.append(dict(call=call, tau=steps - k, hash=h, k=k,
                          forced=forced, a_codec=np.asarray(a_codec,
                                                            np.float32),
                          rec=rec))
    if fork is not None and not forked:
        raise Mismatch("развилка не достигнута")
    if mism:
        raise SystemExit(f"награда среды расходится с check_success ({mism})")
    own = done_step + 1 if done_step >= 0 else steps
    A = np.stack(acts) if acts else np.zeros((0, 7), np.float32)
    return dict(task=int(task), init=int(init), success=bool(reward >= 1.0),
                success_step=int(succ_step), done_step=int(done_step),
                own_steps=int(own), steps=int(steps),
                physical=int(ENV["waiting_steps"] + steps),
                init_hash=ih[0], init_hash_full=ihf[0], actions=A,
                actions_sha1=actions_sha(A), calls=calls, reservoir=chosen,
                n_active_calls=k_res, seconds=time.time() - t0)


def continuation_problems(ref, br, tau):
    """noop против reference после tau."""
    p = []
    if br["success"] != ref["success"] or \
            br["success_step"] != ref["success_step"]:
        p.append("успех/шаг успеха")
    if br["done_step"] != ref["done_step"] or br["own_steps"] != \
            ref["own_steps"] or br["steps"] != ref["steps"]:
        p.append("done/собственные шаги")
    if not np.array_equal(br["actions"], ref["actions"]):
        p.append("исполненные действия")
    hr = [c["hash"] for c in ref["calls"] if c["tau"] >= tau]
    hb = [c["hash"] for c in br["calls"] if c["tau"] >= tau]
    if hr != hb:
        p.append("хеши наблюдений на границах")
    return p


# --- журнал бюджета ---------------------------------------------------------------
class Ledger:
    """Физические шаги по областям (scope -> cap). Резерв пишется до
    исполнения; commit заменяет его фактическим расходом и помечает id
    исполненным (повтор id не начисляется); незакрытый резерв при загрузке
    списывается консервативно как авария."""

    def __init__(self, path, caps):
        self.path = path
        if os.path.exists(path):
            self.d = json.load(open(path, encoding="utf-8"))
            if self.d["caps"] != {k: int(v) for k, v in caps.items()}:
                raise SystemExit(f"{path}: другие лимиты {self.d['caps']}")
            if self.d["pending"]:
                pd = self.d["pending"]
                self.d["spent"][pd["scope"]] = self.d["spent"].get(
                    pd["scope"], 0) + int(pd["amount"])
                self.d["crashes"].append(dict(pd, time=time.strftime(
                    "%Y-%m-%dT%H:%M:%S")))
                self.d["pending"] = None
                self._save()
        else:
            self.d = dict(caps={k: int(v) for k, v in caps.items()},
                          spent={k: 0 for k in caps}, pending=None,
                          done={}, crashes=[])
            self._save()

    def _save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(self.d, fh, ensure_ascii=False, indent=1)
        os.replace(self.path + ".tmp", self.path)

    def spent(self, scope):
        return int(self.d["spent"].get(scope, 0))

    def is_done(self, eid):
        return eid in self.d["done"]

    def reserve(self, scope, eid, amount=EXEC_MAX, limit=None):
        if self.d["pending"]:
            raise SystemExit("резерв уже открыт")
        if eid in self.d["done"]:
            raise SystemExit(f"{eid} уже исполнен и учтён")
        cap = self.d["caps"][scope] if limit is None else min(
            int(limit), self.d["caps"][scope])
        if self.spent(scope) + int(amount) > cap:
            raise SystemExit(f"бюджет {scope}: {self.spent(scope)} + "
                             f"{amount} > {cap}")
        self.d["pending"] = dict(scope=scope, id=eid, amount=int(amount))
        self._save()

    def commit(self, eid, actual):
        pd = self.d["pending"]
        if not pd or pd["id"] != eid:
            raise SystemExit("нет открытого резерва этого исполнения")
        if int(actual) > pd["amount"]:
            raise SystemExit(f"расход {actual} > резерва {pd['amount']}")
        self.d["spent"][pd["scope"]] = self.spent(pd["scope"]) + int(actual)
        self.d["done"][eid] = dict(scope=pd["scope"], actual=int(actual))
        self.d["pending"] = None
        self._save()


# --- случай ------------------------------------------------------------------------
def branch_order(space, seed, task, init, allowed):
    names = [kc.ROWS[i] for i in range(1, 7) if allowed[i]]
    perm = kc.rng(space, seed, task, init, 0, "perm").permutation(len(names))
    return [names[i] for i in perm]


def _save_json(path, d):
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(path + ".tmp", path)


def _save_npz(path, **arr):
    with open(path + ".tmp", "wb") as fh:
        np.savez_compressed(fh, **arr)
    os.replace(path + ".tmp", path)


def _ep_summary(ep):
    return {k: ep[k] for k in ("success", "success_step", "done_step",
                               "own_steps", "steps", "physical", "init_hash",
                               "init_hash_full", "actions_sha1",
                               "n_active_calls")}


def run_case(sim, io, agent, task, init, *, seed, space, cdir, ledger, scope,
             limit, pool_fn, capture, tag, max_branches=None, repeat=None):
    """-> dict случая (case.json). pool_fn(payload, fork) -> dict(arms [7,8,6]
    float32 (NaN — нет), allowed [7], прочие поля пула для npz/json).
    max_branches — smoke: один tree и один flat (первые в перестановке);
    repeat=True — smoke: выбранный tree исполняется повторно."""
    os.makedirs(cdir, exist_ok=True)
    rj = os.path.join(cdir, "ref.json")
    rid = f"{tag}/ref"
    if os.path.exists(rj):
        ref_d = json.load(open(rj, encoding="utf-8"))
        z = dict(np.load(os.path.join(cdir, "ref.npz")))
        if sha(os.path.join(cdir, "ref.npz")) != ref_d["npz_sha1"]:
            raise Mismatch("ref.npz изменён")
        ref = dict(ref_d["episode"], actions=z["actions"],
                   calls=[dict(tau=int(t), hash=h) for t, h in
                          zip(z["call_tau"], ref_d["call_hashes"])])
        fork = dict(tau=ref_d["fork"]["tau"], hash=ref_d["fork"]["hash"],
                    call=ref_d["fork"]["call"])
        pool = dict(z_pool=z)
        allowed = np.asarray(z["allowed"], bool)
        arms = z["arms"]
        grip = z["fork_grip"]
        ref_codec = z["fork_ref_codec"]
    else:
        ledger.reserve(scope, rid, limit=limit)
        ref = run_episode(sim, io, agent, task, init,
                          reservoir=kc.rng(space, seed, task, init, 0, "fork"),
                          capture=capture)
        ledger.commit(rid, ref["physical"])
        ch = ref["reservoir"]
        if ch is None:
            raise Mismatch(f"t{task} i{init}: нет активной границы для развилки")
        fork = dict(tau=ch["tau"], hash=ch["hash"], call=ch["call"])
        pool = pool_fn(ch["payload"], fork)
        allowed = np.asarray(pool["allowed"], bool)
        arms = np.asarray(pool["arms"], np.float32)
        ref_codec = ch["a_codec"]
        grip = ref_codec[:, 6].copy()
        extra = {k: v for k, v in pool.get("npz", {}).items()}
        _save_npz(os.path.join(cdir, "ref.npz"), actions=ref["actions"],
                  call_tau=np.array([c["tau"] for c in ref["calls"]]),
                  allowed=allowed, arms=arms, fork_grip=grip,
                  fork_ref_codec=ref_codec, **extra)
        _save_json(rj, dict(
            task=int(task), init=int(init), seed=int(seed), space=int(space),
            episode=_ep_summary(ref), fork=fork,
            call_hashes=[c["hash"] for c in ref["calls"]],
            pool=pool.get("json", {}),
            npz_sha1=sha(os.path.join(cdir, "ref.npz"))))
    names = branch_order(space, seed, task, init, allowed)
    if max_branches is not None:
        tree = [n for n in names if n in ("root", "mid", "leaf")][:1]
        flat = [n for n in names if n.startswith("flat")][:1]
        names = [n for n in names if n in tree + flat]
    rep_name = next((n for n in names if n in ("root", "mid", "leaf")),
                    None) if repeat else None
    order = ["noop"] + names + ([f"{rep_name}#2"] if rep_name else [])
    out = {}
    for name in order:
        base = name.split("#")[0]
        bj = os.path.join(cdir, f"b_{name}.json")
        bid = f"{tag}/b_{name}"
        if os.path.exists(bj):
            b = json.load(open(bj, encoding="utf-8"))
            if sha(os.path.join(cdir, f"b_{name}.npz")) != b["npz_sha1"]:
                raise Mismatch(f"{bid}: npz изменён")
            out[name] = b
            continue
        i = kc.ROWS.index(base)
        if base == "noop":
            a_codec = np.asarray(ref_codec, np.float32)
        else:
            a_codec = np.concatenate([arms[i], grip[:, None]], 1).astype(
                np.float32)
        if not np.isfinite(a_codec).all():
            raise Mismatch(f"{bid}: команды кандидата не конечны")
        ledger.reserve(scope, bid, limit=limit)
        br = run_episode(sim, io, agent, task, init,
                         prefix=np.asarray(ref["actions"])[:fork["tau"]],
                         fork=dict(fork, a_codec=a_codec))
        ledger.commit(bid, br["physical"])
        if base == "noop":
            pr = continuation_problems(ref, br, fork["tau"])
            if pr:
                _save_json(os.path.join(cdir, "noop_mismatch.json"),
                           dict(problems=pr, ref=_ep_summary(ref),
                                noop=_ep_summary(br)))
                raise Mismatch(f"t{task} i{init}: noop не воспроизвёл "
                               f"reference: {pr}")
        _save_npz(os.path.join(cdir, f"b_{name}.npz"), actions=br["actions"],
                  fork_codec=a_codec)
        b = dict(name=name, episode=_ep_summary(br),
                 npz_sha1=sha(os.path.join(cdir, f"b_{name}.npz")))
        _save_json(bj, b)
        out[name] = b
    R0 = bool(ref["success"])
    labels = {n: int(out[n]["episode"]["success"]) - int(R0) for n in out
              if n != "noop" and "#" not in n}
    case = dict(task=int(task), init=int(init), seed=int(seed),
                space=int(space), fork=dict(tau=fork["tau"],
                                            call=fork["call"]),
                R0=R0, branches={n: out[n]["episode"]["success"]
                                 for n in out}, labels=labels,
                allowed=[bool(x) for x in allowed], order=order,
                physical=dict(ref=json.load(open(rj))["episode"]["physical"],
                              **{n: out[n]["episode"]["physical"]
                                 for n in out}))
    if rep_name:
        a_, b_ = out[rep_name]["episode"], out[f"{rep_name}#2"]["episode"]
        case["repeat_equal"] = all(a_[k] == b_[k] for k in (
            "success", "done_step", "own_steps", "actions_sha1"))
    _save_json(os.path.join(cdir, "case.json"), case)
    return case


# --- игрушки ---------------------------------------------------------------------
class FakeSim:
    """Игрушка с историей контроллера: v = 0.8·v + a; x += 0.05·v; успех при
    x > thr; raw done мигает (True на одном шаге без успеха)."""

    def __init__(self, thr=1.5, flicker_at=None):
        self.thr, self.flicker_at = thr, flicker_at

    def reset(self, task, starts):
        self.t = 0
        self.x = float(starts[0]) * 0.01 + 0.1 * int(task)
        self.v = 0.0
        return self._obs()

    def _obs(self):
        st = np.zeros((1, 9), np.float32)
        st[0, 0], st[0, 1] = self.x, self.v
        img = np.full((1, 4, 4, 3), int(abs(self.x) * 10) % 255, np.uint8)
        return dict(state=st, agentview_image=img,
                    robot0_eye_in_hand_image=img[..., ::-1].copy())

    def step(self, a):
        a = np.asarray(a, np.float64)
        self.t += 1
        self.v = 0.8 * self.v + float(a[0, 0])
        self.x += 0.05 * self.v
        s = self.x > self.thr
        d = s or (self.flicker_at is not None and self.t == self.flicker_at)
        return self._obs(), np.array([float(s)]), np.array([d]), \
            np.array([s])


class FakeIO:
    max_act_q = np.array([0.9] * 6 + [1.0])

    def seed(self, s):
        pass

    def task_desc(self, t):
        return "toy"

    def prep(self, obs, desc, task):
        return dict(x=float(obs["state"][0, 0]),
                    v=float(obs["state"][0, 1])), 0, \
            np.asarray(obs["state"][:, :8], np.float64)

    def scale(self, a):
        import k29_policy as kp
        return kp.scale_exec(a, self.max_act_q)


class FakeAgent:
    def __init__(self, gain=0.02):
        self.gain = gain
        self.call = 0

    def begin(self, task, init):
        self.call = 0
        self.last = None

    def decide(self, batch, po, s8, elapsed):
        self.call += 1
        a = np.zeros((N8, 7), np.float32)
        a[:, 0] = self.gain * (1.0 + 0.5 * np.sin(3 * batch["x"])) - \
            0.01 * batch["v"]
        a[:, 6] = 1.0
        self.last = dict(x=batch["x"], call=self.call)
        return a, dict(call=self.call)

    def executed(self, k):
        pass


def selftest():
    import tempfile
    io, ag = FakeIO(), FakeAgent()
    caps = dict(dev_s0=400000, checks=20000)
    # reference: защёлка и раннее прекращение при мигающем done
    r1 = run_episode(FakeSim(flicker_at=10 + 21), io, ag, 0, 3)
    assert r1["done_step"] == 20 and not r1["success"] and \
        r1["steps"] == 21 and r1["own_steps"] == 21
    assert r1["physical"] == 10 + 21
    r = run_episode(FakeSim(), io, ag, 0, 3,
                    reservoir=kc.rng(0, 0, 0, 3, 0, "fork"),
                    capture=lambda a: dict(x=a.last["x"]))
    assert r["success"] and r["reservoir"] is not None and \
        r["reservoir"]["tau"] % N8 == 0
    # reservoir: равномерен по активным границам (статистика по потокам)
    taus = []
    for s in range(400):
        rr = run_episode(FakeSim(), io, ag, 0, 3,
                         reservoir=kc.rng(0, s, 0, 3, 0, "fork"))
        taus.append(rr["reservoir"]["call"])
    cnt = np.bincount(taus)[1:]
    assert len(cnt) == r["n_active_calls"] and cnt.min() > 0
    # повтор префикса + noop воспроизводят продолжение; подмена команды и
    # другой старт — отказ; один qpos без истории контроллера — не то же
    tau = r["reservoir"]["tau"]
    noop = run_episode(FakeSim(), io, ag, 0, 3, prefix=r["actions"][:tau],
                       fork=dict(tau=tau, hash=r["reservoir"]["hash"],
                                 a_codec=r["reservoir"]["a_codec"]))
    assert continuation_problems(r, noop, tau) == []
    bad = r["actions"][:tau].copy()
    if tau:
        bad[tau // 2, 0] += 1e-3
        try:
            run_episode(FakeSim(), io, ag, 0, 3, prefix=bad,
                        fork=dict(tau=tau, hash=r["reservoir"]["hash"],
                                  a_codec=r["reservoir"]["a_codec"]))
            raise AssertionError("подмена команды префикса принята")
        except Mismatch:
            pass
    try:
        run_episode(FakeSim(), io, ag, 0, 4, prefix=r["actions"][:tau],
                    fork=dict(tau=tau, hash=r["reservoir"]["hash"],
                              a_codec=r["reservoir"]["a_codec"]))
        raise AssertionError("другой старт принят")
    except Mismatch:
        pass
    sim_q = FakeSim()
    sim_q.reset(0, [3])
    for _ in range(10):
        sim_q.step(np.array([DUMMY]))
    for a in r["actions"][:tau]:
        sim_q.step(a[None])
    v_true = sim_q.v
    if tau:
        assert v_true != 0.0      # история контроллера: одного x мало
    # кандидат меняет продолжение; noop ≠ кандидат
    cand = r["reservoir"]["a_codec"].copy()
    cand[:, 0] += 0.3
    cb = run_episode(FakeSim(), io, ag, 0, 3, prefix=r["actions"][:tau],
                     fork=dict(tau=tau, hash=r["reservoir"]["hash"],
                               a_codec=cand))
    assert not np.array_equal(cb["actions"], r["actions"])
    assert cb["physical"] <= EXEC_MAX and cb["steps"] <= 600
    # случай целиком, возобновление без двойного расхода, перестановка
    with tempfile.TemporaryDirectory() as td:
        lp = os.path.join(td, "ledger.json")
        L = Ledger(lp, caps)

        def pool_fn(payload, fork):
            arms = np.full((7, N8, 6), np.nan, np.float32)
            al = np.array([True, True, False, True, True, True, False])
            for i in np.nonzero(al)[0][1:]:
                arms[i] = 0.05 * i
            return dict(arms=arms, allowed=al, json=dict(m=2),
                        npz=dict(m=np.array(2)))
        cd = os.path.join(td, "case")
        c1 = run_case(FakeSim(), io, ag, 0, 3, seed=0, space=0, cdir=cd,
                      ledger=L, scope="dev_s0", limit=None, pool_fn=pool_fn,
                      capture=lambda a: None, tag="s0/t0_i3")
        spent = L.spent("dev_s0")
        assert set(c1["labels"]) == {"root", "leaf", "flat0", "flat1"}
        assert c1["order"][0] == "noop" and c1["order"][1:] == branch_order(
            0, 0, 0, 3, pool_fn(None, None)["allowed"])
        assert spent == sum(c1["physical"].values())
        c2 = run_case(FakeSim(), io, ag, 0, 3, seed=0, space=0, cdir=cd,
                      ledger=Ledger(lp, caps), scope="dev_s0", limit=None,
                      pool_fn=pool_fn, capture=lambda a: None,
                      tag="s0/t0_i3")
        assert c2["labels"] == c1["labels"] and \
            Ledger(lp, caps).spent("dev_s0") == spent
        # авария посреди ветви: резерв списан, данные не задвоены
        L3 = Ledger(lp, caps)
        L3.reserve("dev_s0", "x/crash")
        L4 = Ledger(lp, caps)
        assert L4.spent("dev_s0") == spent + EXEC_MAX and L4.d["crashes"]
        try:
            L4.reserve("dev_s0", "s0/t0_i3/ref")
            raise AssertionError("повторное начисление исполненного id")
        except SystemExit:
            pass
        try:
            L4.reserve("checks", "y", amount=20001)
            raise AssertionError("превышение лимита принято")
        except SystemExit:
            pass
        try:
            L4.reserve("dev_s0", "z", limit=spent)
            raise AssertionError("лимит этапа не применён")
        except SystemExit:
            pass
        # smoke: повтор кандидата воспроизводим
        cs = run_case(FakeSim(), io, ag, 0, 5, seed=0, space=1,
                      cdir=os.path.join(td, "smoke"), ledger=L4,
                      scope="checks", limit=None, pool_fn=pool_fn,
                      capture=lambda a: None, tag="smoke/t0_i5",
                      max_branches=2, repeat=True)
        assert cs["repeat_equal"] and len(cs["order"]) == 4
        # noop, расходящийся с reference, — технический отказ
        ag_bad = FakeAgent()
        orig = ag_bad.decide
        cnt_ = dict(n=0)

        def flaky(batch, po, s8, el):
            cnt_["n"] += 1
            a, r_ = orig(batch, po, s8, el)
            if cnt_["n"] > 40:
                a = a.copy()
                a[:, 0] += 1e-3
            return a, r_
        ag_bad.decide = flaky
        try:
            run_case(FakeSim(), io, ag_bad, 0, 7, seed=0, space=0,
                     cdir=os.path.join(td, "bad"), ledger=L4, scope="dev_s0",
                     limit=None, pool_fn=pool_fn, capture=lambda a: None,
                     tag="s0/t0_i7")
            raise AssertionError("невоспроизводимый noop принят")
        except Mismatch:
            pass
    print("самопроверка k31_collect пройдена: защёлка done (мигающий done "
          "прекращает эпизод), reservoir — одна равномерная граница, "
          "префикс + noop воспроизводят reference, подмена команды/старта — "
          "отказ, история контроллера; случай: noop первым, кандидаты в "
          "фиксированной перестановке, метки Rc − R0, возобновление без "
          "двойного расхода, авария списывает резерв, лимиты области/этапа; "
          "smoke-повтор кандидата; невоспроизводимый noop — отказ")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
