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
    """Физические шаги по областям (scope -> cap). Порядок исполнения:
    reserve(id) -> исполнение -> атомарная запись результата (с id и
    фактическим расходом) -> commit(id). Незакрытый резерв переживает
    перезапуск: если результат id уже записан, settle(id, расход) закрывает
    его фактическим расходом (результат не теряется, расход не задваивается);
    иначе при следующем reserve резерв списывается целиком как авария
    (аварийный расход в бюджет не возвращается). Повтор исполненного id не
    начисляется и не резервируется."""

    def __init__(self, path, caps):
        self.path = path
        if os.path.exists(path):
            self.d = json.load(open(path, encoding="utf-8"))
            if self.d["caps"] != {k: int(v) for k, v in caps.items()}:
                raise SystemExit(f"{path}: другие лимиты {self.d['caps']}")
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

    def pending_id(self):
        return (self.d["pending"] or {}).get("id")

    def _crash_pending(self):
        pd = self.d["pending"]
        self.d["spent"][pd["scope"]] = self.spent(pd["scope"]) + \
            int(pd["amount"])
        self.d["crashes"].append(dict(pd, time=time.strftime(
            "%Y-%m-%dT%H:%M:%S")))
        self.d["pending"] = None
        self._save()

    def reserve(self, scope, eid, amount=EXEC_MAX, limit=None):
        if eid in self.d["done"]:
            raise SystemExit(f"{eid} уже исполнен и учтён")
        if self.d["pending"]:
            self._crash_pending()             # результата нет — авария
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

    def settle(self, eid, actual):
        """Результат eid записан: учесть его, если commit не успел."""
        if eid in self.d["done"]:
            if int(self.d["done"][eid]["actual"]) != int(actual):
                raise SystemExit(f"{eid}: расход результата ≠ учтённому")
            return False
        if self.pending_id() == eid:
            self.commit(eid, actual)
            return True
        raise SystemExit(f"{eid}: результат записан, но не учтён в бюджете")


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


EP_EQ = ("success", "success_step", "done_step", "own_steps", "steps",
         "actions_sha1")


def case_problems(cdir, *, seed, space, task, init, smoke=False):
    """Общая проверка завершённого случая (гейт, набор, возобновление):
    идентификаторы, SHA файлов, обязательные ветви, исходы, R0, метки
    Rc − R0, подтверждение noop, повтор (smoke). Пустой список — случай
    цел."""
    p = []
    paths = {k: os.path.join(cdir, k) for k in ("case.json", "ref.json",
                                                 "ref.npz")}
    miss = [k for k, v in paths.items() if not os.path.exists(v)]
    if miss:
        return [f"нет {miss}"]
    try:
        case = json.load(open(paths["case.json"], encoding="utf-8"))
        rj = json.load(open(paths["ref.json"], encoding="utf-8"))
        z = np.load(paths["ref.npz"])
        allowed = np.asarray(z["allowed"], bool)
    except Exception as e:                                   # noqa: BLE001
        return [f"не читается: {type(e).__name__}"]
    want = dict(seed=int(seed), space=int(space), task=int(task),
                init=int(init))
    for doc, nm in ((case, "case"), (rj, "ref")):
        bad = [k for k, v in want.items() if doc.get(k) != v]
        if bad:
            p.append(f"{nm}: идентификаторы {bad}")
    if sha(paths["ref.npz"]) != rj.get("npz_sha1"):
        p.append("ref.npz изменён")
    if case.get("R0") != bool(rj["episode"]["success"]):
        p.append("R0 ≠ исходу reference")
    if case.get("allowed") != [bool(x) for x in allowed]:
        p.append("allowed ≠ ref.npz")
    if case.get("fork", {}).get("tau") != rj["fork"]["tau"] or \
            case.get("fork", {}).get("call") != rj["fork"]["call"]:
        p.append("развилка ≠ ref.json")
    order = case.get("order") or []
    names = [n for n in order if n != "noop" and "#" not in n]
    allowed_names = [kc.ROWS[i] for i in range(1, 7) if allowed[i]]
    if not order or order[0] != "noop":
        p.append("noop не первый")
    if smoke:
        if not set(names) <= set(allowed_names):
            p.append("ветви вне разрешённого пула")
    elif names != branch_order(space, seed, task, init, allowed):
        p.append("набор/порядок ветвей ≠ разрешённым кандидатам в "
                 "перестановке")
    eps = {}
    for n in order:
        bj = os.path.join(cdir, f"b_{n}.json")
        bn = os.path.join(cdir, f"b_{n}.npz")
        if not (os.path.exists(bj) and os.path.exists(bn)):
            p.append(f"нет ветви {n}")
            continue
        b = json.load(open(bj, encoding="utf-8"))
        if sha(bn) != b.get("npz_sha1") or b.get("name") != n:
            p.append(f"ветвь {n}: файл изменён")
        eps[n] = b["episode"]
        if bool(b["episode"]["success"]) != bool(
                (case.get("branches") or {}).get(n)):
            p.append(f"ветвь {n}: исход ≠ case.json")
        if (case.get("physical") or {}).get(n) != b["episode"]["physical"]:
            p.append(f"ветвь {n}: расход ≠ case.json")
    if p:
        return p
    R0 = int(case["R0"])
    want_lab = {n: int(eps[n]["success"]) - R0 for n in names}
    if case.get("labels") != want_lab:
        p.append(f"метки ≠ Rc − R0 ({case.get('labels')} vs {want_lab})")
    ne = eps["noop"]
    if any(ne[k] != rj["episode"][k] for k in EP_EQ):
        p.append("noop не подтверждает reference")
    reps = [n for n in order if "#" in n]
    if smoke:
        if len(reps) != 1:
            p.append("smoke: нет повтора кандидата")
        else:
            a_, b_ = eps[reps[0].split("#")[0]], eps[reps[0]]
            eq = all(a_[k] == b_[k] for k in EP_EQ)
            if case.get("repeat_equal") != eq or not eq:
                p.append("smoke: повтор кандидата не воспроизвёлся")
    elif reps:
        p.append("повторы вне smoke")
    return p


def run_case(sim, io, agent, task, init, *, seed, space, cdir, ledger, scope,
             limit, pool_fn, capture, tag, max_branches=None, repeat=None):
    """-> dict случая (case.json). pool_fn(payload, fork) -> dict(arms [7,8,6]
    float32 (NaN — нет), allowed [7], npz/json — поля пула).
    max_branches — smoke: один tree и один flat (первые в перестановке);
    repeat=True — smoke: выбранный tree исполняется повторно.
    Результат каждого исполнения записывается ДО commit бюджета; при
    возобновлении записанный результат закрывается ledger.settle."""
    smoke = max_branches is not None
    os.makedirs(cdir, exist_ok=True)
    cj = os.path.join(cdir, "case.json")
    if os.path.exists(cj):
        pr = case_problems(cdir, seed=seed, space=space, task=task,
                           init=init, smoke=smoke)
        if pr:
            raise Mismatch(f"{tag}: готовый случай повреждён: {pr[:3]}")
        return json.load(open(cj, encoding="utf-8"))
    if os.path.exists(os.path.join(cdir, "noop_mismatch.json")):
        raise Mismatch(f"{tag}: noop ранее не воспроизвёл reference "
                       f"(noop_mismatch.json)")
    rj = os.path.join(cdir, "ref.json")
    rid = f"{tag}/ref"
    if os.path.exists(rj):
        ref_d = json.load(open(rj, encoding="utf-8"))
        if sha(os.path.join(cdir, "ref.npz")) != ref_d["npz_sha1"] or \
                ref_d.get("eid") != rid:
            raise Mismatch(f"{rid}: ref.npz/ref.json изменены")
        ledger.settle(rid, ref_d["episode"]["physical"])
        z = dict(np.load(os.path.join(cdir, "ref.npz")))
        ref = dict(ref_d["episode"], actions=z["actions"],
                   calls=[dict(tau=int(t), hash=h) for t, h in
                          zip(z["call_tau"], ref_d["call_hashes"])])
        fork = dict(tau=ref_d["fork"]["tau"], hash=ref_d["fork"]["hash"],
                    call=ref_d["fork"]["call"])
        allowed = np.asarray(z["allowed"], bool)
        arms, grip = z["arms"], z["fork_grip"]
        ref_codec = z["fork_ref_codec"]
    else:
        ledger.reserve(scope, rid, limit=limit)
        ref = run_episode(sim, io, agent, task, init,
                          reservoir=kc.rng(space, seed, task, init, 0, "fork"),
                          capture=capture)
        ch = ref["reservoir"]
        if ch is None:
            raise Mismatch(f"t{task} i{init}: нет активной границы для развилки")
        fork = dict(tau=ch["tau"], hash=ch["hash"], call=ch["call"])
        pool = pool_fn(ch["payload"], fork)
        allowed = np.asarray(pool["allowed"], bool)
        arms = np.asarray(pool["arms"], np.float32)
        ref_codec = ch["a_codec"]
        grip = ref_codec[:, 6].copy()
        _save_npz(os.path.join(cdir, "ref.npz"), actions=ref["actions"],
                  call_tau=np.array([c["tau"] for c in ref["calls"]]),
                  allowed=allowed, arms=arms, fork_grip=grip,
                  fork_ref_codec=ref_codec, **pool.get("npz", {}))
        _save_json(rj, dict(
            task=int(task), init=int(init), seed=int(seed), space=int(space),
            eid=rid, episode=_ep_summary(ref), fork=fork,
            call_hashes=[c["hash"] for c in ref["calls"]],
            pool=pool.get("json", {}),
            npz_sha1=sha(os.path.join(cdir, "ref.npz"))))
        ledger.commit(rid, ref["physical"])
    names = branch_order(space, seed, task, init, allowed)
    if smoke:
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
            if sha(os.path.join(cdir, f"b_{name}.npz")) != b["npz_sha1"] or \
                    b.get("eid") != bid:
                raise Mismatch(f"{bid}: файлы ветви изменены")
            ledger.settle(bid, b["episode"]["physical"])
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
        if base == "noop":
            pr = continuation_problems(ref, br, fork["tau"])
            if pr:
                _save_json(os.path.join(cdir, "noop_mismatch.json"),
                           dict(problems=pr, ref=_ep_summary(ref),
                                noop=_ep_summary(br), eid=bid))
                ledger.commit(bid, br["physical"])
                raise Mismatch(f"t{task} i{init}: noop не воспроизвёл "
                               f"reference: {pr}")
        _save_npz(os.path.join(cdir, f"b_{name}.npz"), actions=br["actions"],
                  fork_codec=a_codec)
        b = dict(name=name, eid=bid, episode=_ep_summary(br),
                 npz_sha1=sha(os.path.join(cdir, f"b_{name}.npz")))
        _save_json(bj, b)
        ledger.commit(bid, br["physical"])
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
        case["repeat_equal"] = all(a_[k] == b_[k] for k in EP_EQ)
    _save_json(cj, case)
    pr = case_problems(cdir, seed=seed, space=space, task=task, init=init,
                       smoke=smoke)
    if pr:
        raise Mismatch(f"{tag}: новый случай не прошёл проверку: {pr[:3]}")
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
        # общая проверка случая: подмена метки / R0 / исхода — отказ
        assert case_problems(cd, seed=0, space=0, task=0, init=3) == []
        cj = os.path.join(cd, "case.json")
        good = open(cj).read()
        for mut in (lambda c: c["labels"].__setitem__(
                        "root", 1 - c["labels"]["root"]),
                    lambda c: c.__setitem__("R0", not c["R0"]),
                    lambda c: c["branches"].__setitem__(
                        "leaf", not c["branches"]["leaf"]),
                    lambda c: c.__setitem__("order", c["order"][:-1]),
                    lambda c: c.__setitem__("init", 4)):
            c_ = json.loads(good)
            mut(c_)
            json.dump(c_, open(cj, "w"))
            assert case_problems(cd, seed=0, space=0, task=0, init=3)
            try:
                run_case(FakeSim(), io, ag, 0, 3, seed=0, space=0, cdir=cd,
                         ledger=Ledger(lp, caps), scope="dev_s0", limit=None,
                         pool_fn=pool_fn, capture=lambda a: None,
                         tag="s0/t0_i3")
                raise AssertionError("повреждённый случай принят")
            except Mismatch:
                pass
        open(cj, "w").write(good)
        # авария между записью результата и commit — для reference и для
        # кандидата: результат не теряется, расход не задваивается
        for crash_at in ("ref", "b_leaf"):
            lp2 = os.path.join(td, f"l_{crash_at}.json")
            cd2 = os.path.join(td, f"c_{crash_at}")

            class Dying(Ledger):
                def commit(self, eid, actual):
                    if eid.endswith("/" + crash_at):
                        raise RuntimeError("процесс убит до commit")
                    return super().commit(eid, actual)
            try:
                run_case(FakeSim(), io, ag, 0, 3, seed=0, space=0, cdir=cd2,
                         ledger=Dying(lp2, caps), scope="dev_s0", limit=None,
                         pool_fn=pool_fn, capture=lambda a: None,
                         tag="s0/t0_i3")
                raise AssertionError("авария не сработала")
            except RuntimeError:
                pass
            Lr = Ledger(lp2, caps)
            assert Lr.pending_id() == f"s0/t0_i3/{crash_at}"
            cr = run_case(FakeSim(), io, ag, 0, 3, seed=0, space=0, cdir=cd2,
                          ledger=Lr, scope="dev_s0", limit=None,
                          pool_fn=pool_fn, capture=lambda a: None,
                          tag="s0/t0_i3")
            Lr = Ledger(lp2, caps)
            assert cr["labels"] == c1["labels"] and not Lr.d["crashes"]
            assert Lr.spent("dev_s0") == sum(cr["physical"].values()), \
                crash_at
        # авария до записи результата: резерв списан целиком, не возвращён
        lp3, cd3 = os.path.join(td, "l3.json"), os.path.join(td, "c3")
        ag_die = FakeAgent()
        od, n_ = ag_die.decide, dict(n=0)

        def die(batch, po, s8, el):
            n_["n"] += 1
            if n_["n"] == 60:
                raise RuntimeError("процесс убит посреди ветви")
            return od(batch, po, s8, el)
        ag_die.decide = die
        try:
            run_case(FakeSim(), io, ag_die, 0, 3, seed=0, space=0, cdir=cd3,
                     ledger=Ledger(lp3, caps), scope="dev_s0", limit=None,
                     pool_fn=pool_fn, capture=lambda a: None, tag="s0/t0_i3")
            raise AssertionError("авария не сработала")
        except RuntimeError:
            pass
        c3 = run_case(FakeSim(), io, ag, 0, 3, seed=0, space=0, cdir=cd3,
                      ledger=Ledger(lp3, caps), scope="dev_s0", limit=None,
                      pool_fn=pool_fn, capture=lambda a: None, tag="s0/t0_i3")
        L3 = Ledger(lp3, caps)
        assert len(L3.d["crashes"]) == 1 and L3.spent("dev_s0") == \
            sum(c3["physical"].values()) + EXEC_MAX
        L4 = Ledger(lp, caps)
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
