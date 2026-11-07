#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-33: свежий банк начальных сцен LIBERO-10 (native reset) и адаптер среды.

Рецепт сцены (явно закреплён):
  np.random.seed(scene_seed); random.seed(scene_seed);
  create_libero_env(libero_10, task, 224, cameras, seed=scene_seed)  # env.seed
  np.random.seed(scene_seed) -> native reset (OffScreenRenderEnv.reset) ->
  снимок S0 (до ожидания): flattened sim state, имена тел/суставов,
  размерности, отпечаток модели, runtime body_pos/body_quat (в т. ч.
  фикстуры), каноническая физическая подпись (позы тел и qpos суставов,
  кватернионы со знаком w ≥ 0, округление 1e−6);
  затем execution seed 101 (np/random) и 10 шагов ожидания [0,…,0,−1] —
  post-wait init_hash / init_hash_full по формуле K-9h.
Воспроизведение: новый объект среды по тому же рецепту, native reset,
  сверка имён/размерностей/отпечатка модели и runtime-массивов поз с
  снимком (несовпадение — отказ: ничего не подменяется), set_init_state(S0),
  execution seed 101. Ожидание делает вызывающий (K-9h) — адаптер reset
  возвращает наблюдение S0.
Seed кандидата: SeedSequence([33, 20261106, space, task, idx]).generate_state(
  1, uint32)[0]; space 0 — development (5/задача), 1 — confirmation (300).
Допустимость (технически, без политик): корректный reset, конечные
  состояние/наблюдение, нет успеха/done до и после ожидания, нет дубликата
  физической подписи (внутри и между пространствами; с официальными init —
  по qpos суставов). Брать первые допустимые в порядке idx.
K33Env — среда для libero2gym.vector_env.AsyncVectorEnv: на каждом reset
  закрывает прежний объект среды и создаёт новый в том же CPU-воркере
  (воркеры создаются один раз в get_envs, до модели). Режимы: legacy
  (штатный init_state_id: create seed 0 -> native reset -> init file) и
  bank (init_state_id = индекс случая задачи в банке).
"""
import hashlib
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SUITE = "libero_10"
N_TASKS = 10
SPACES = {0: 5, 1: 300}
EXEC_SEED = 101
WAIT = 10
DUMMY = [0, 0, 0, 0, 0, 0, -1]
ROUND = 1e-6
CAMERAS = ["agentview", "robot0_eye_in_hand"]
BANK_DIR = "data/k33/bank"
MANIFEST = "reports/k33/bank_manifest.json"
RECIPE = dict(seed_seq=[33, 20261106], scene_seed="np+random+env.seed before "
              "create, np.random.seed before native reset", exec_seed=EXEC_SEED,
              wait=WAIT, dummy=DUMMY, round=ROUND, cameras=CAMERAS,
              image_size=224, suite=SUITE)


class Technical(Exception):
    pass


def scene_seed(space, task, idx):
    return int(np.random.SeedSequence([33, 20261106, int(space), int(task),
                                       int(idx)]).generate_state(
        1, np.uint32)[0])


def case_id(space, task, idx):
    return f"k33s{space}_t{task}_c{idx:05d}"


# --- описание сцены --------------------------------------------------------------
def _names(model, n, fn):
    return [str(getattr(model, fn)(i)) for i in range(n)]


def model_meta(sim):
    m = sim.model
    bodies = _names(m, m.nbody, "body_id2name")
    joints = _names(m, m.njnt, "joint_id2name")
    h = hashlib.sha1()
    for k in ("body_mass", "geom_size", "jnt_type", "body_parentid",
              "geom_type", "jnt_qposadr"):
        a = getattr(m, k, None)
        if a is not None:
            a = np.ascontiguousarray(np.asarray(a))
            h.update(k.encode())
            h.update(str(a.dtype).encode())
            h.update(a.tobytes())
    h.update(json.dumps([bodies, joints, int(m.nq), int(m.nv)]).encode())
    return dict(bodies=bodies, joints=joints, nq=int(m.nq), nv=int(m.nv),
                nbody=int(m.nbody), njnt=int(m.njnt), fingerprint=h.hexdigest())


def runtime_arrays(sim):
    return (np.array(sim.model.body_pos, np.float64, copy=True),
            np.array(sim.model.body_quat, np.float64, copy=True))


def canon_quat(q):
    """Знак кватерниона (w, x, y, z): w ≥ 0; форма входа сохраняется.
    (Для подписи используется quat_key — однозначный и при w = 0.)"""
    q = np.asarray(q, np.float64)
    flat = q.reshape(-1, 4).copy()
    flat[flat[:, 0] < 0] *= -1
    return flat.reshape(q.shape)


def quat_key(q):
    """Кватернион -> целые на сетке 1e−6 с однозначным знаком: первая
    ненулевая компонента ОКРУГЛЁННОГО кватерниона положительна (q и −q,
    в т. ч. при w = 0 около π, дают одно и то же)."""
    k = np.round(np.asarray(q, np.float64) / ROUND).astype(np.int64)
    nz = np.nonzero(k)[0]
    if len(nz) and k[nz[0]] < 0:
        k = -k
    return k


def signature(sim, meta=None, joints_only=False):
    """Каноническая физическая подпись: qpos суставов по именам (кватернионы
    свободных суставов со знаком w ≥ 0) и позы тел модели (фикстуры)."""
    meta = meta or model_meta(sim)
    m, d = sim.model, sim.data
    parts = []
    for j, name in enumerate(meta["joints"]):
        adr = int(m.jnt_qposadr[j])
        t = int(m.jnt_type[j])
        n = {0: 7, 1: 4, 2: 1, 3: 1}[t]
        q = np.asarray(d.qpos[adr:adr + n], np.float64).copy()
        if t == 0:
            k = np.concatenate([np.round(q[:3] / ROUND).astype(np.int64),
                                quat_key(q[3:])])
        elif t == 1:
            k = quat_key(q)
        else:
            k = np.round(q / ROUND).astype(np.int64)
        parts.append((name, k))
    if not joints_only:
        bp, bq = runtime_arrays(sim)
        for i, name in enumerate(meta["bodies"]):
            parts.append(("body:" + name, np.concatenate(
                [np.round(bp[i] / ROUND).astype(np.int64), quat_key(bq[i])])))
    h = hashlib.sha1()
    for name, a in parts:
        h.update(name.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def post_wait_hashes(obs1):
    """obs одной среды (без batch) -> (init_hash, init_hash_full) K-9h."""
    import k18_collector as k18c
    o = {k: np.asarray(v)[None] for k, v in obs1.items()}
    o["state"] = o["state"].astype(np.float32)
    ih, ihf = k18c.init_hashes(o, 1)
    return ih[0], ihf[0]


def finite_obs(obs):
    return all(np.isfinite(np.asarray(v, np.float64)).all()
               for v in obs.values())


# --- создание / восстановление ------------------------------------------------------
def seed_all(s):
    np.random.seed(int(s) % (2 ** 32))
    random.seed(int(s))


def _create(task, seed, image_size=224, cameras=None):
    from libero2gym.libero_env import create_libero_env
    seed_all(seed)
    return create_libero_env(SUITE, task, image_size,
                             CAMERAS if cameras is None else cameras,
                             seed=seed)


def _obs_now(env):
    """Наблюдение в форме LiberoEnv.get_observations (state float64, кадры
    перевёрнуты), по текущему состоянию OffScreenRenderEnv."""
    from collections import OrderedDict
    raw = env.env._get_observations()
    obs = dict(state=np.concatenate([raw["robot0_gripper_qpos"],
                                     raw["robot0_eef_pos"],
                                     raw["robot0_eef_quat"]]))
    for c in CAMERAS:
        obs[f"{c}_image"] = raw[f"{c}_image"][::-1]
    return OrderedDict(obs)


def generate(task, seed):
    """Новая сцена: -> dict снимка (до ожидания) + post-wait хеши."""
    env, _init = _create(task, seed)
    try:
        seed_all(seed)
        env.reset()                                  # native reset
        sim = env.sim
        meta = model_meta(sim)
        bp, bq = runtime_arrays(sim)
        S0 = np.array(env.get_sim_state(), np.float64, copy=True)
        sig = signature(sim, meta)
        sig_j = signature(sim, meta, joints_only=True)
        succ0 = bool(env.check_success())
        seed_all(EXEC_SEED)
        obs, done, succ_w = None, False, False
        for _ in range(WAIT):
            _raw, _r, d, _i = env.step(np.array(DUMMY))
            done |= bool(d)
            succ_w |= bool(env.check_success())
            obs = _obs_now(env)
        ih, ihf = post_wait_hashes(obs)
        ok = np.isfinite(S0).all() and finite_obs(obs)
        return dict(S0=S0, body_pos=bp, body_quat=bq, meta=meta, sig=sig,
                    sig_joints=sig_j, success_at_reset=succ0,
                    done_wait=done, success_wait=succ_w, finite=bool(ok),
                    init_hash=ih, init_hash_full=ihf, steps=WAIT)
    finally:
        env.close()


def restore_into(env, case, meta_want):
    """Native reset того же рецепта в НОВОМ объекте env и S0; сверки до
    подстановки состояния. -> наблюдение S0 (сырое)."""
    seed_all(case["seed"])
    env.reset()
    sim = env.sim
    meta = model_meta(sim)
    for k in ("bodies", "joints", "nq", "nv", "fingerprint"):
        if meta[k] != meta_want[k]:
            raise Technical(f"{case['case_id']}: модель/имена ≠ банку ({k})")
    bp, bq = runtime_arrays(sim)
    if not (np.array_equal(bp, case["body_pos"]) and
            np.array_equal(bq, case["body_quat"])):
        raise Technical(f"{case['case_id']}: runtime-позы тел/фикстур после "
                        f"native reset ≠ снимку")
    raw = env.set_init_state(np.asarray(case["S0"], np.float64))
    if signature(sim, meta) != case["sig"]:
        raise Technical(f"{case['case_id']}: подпись сцены после "
                        f"восстановления ≠ снимку")
    seed_all(EXEC_SEED)
    return raw


def replay(task, case, meta_want):
    """Холодное воспроизведение: новый env, восстановление, ожидание ->
    post-wait хеши и флаги."""
    env, _i = _create(task, case["seed"])
    try:
        restore_into(env, case, meta_want)
        obs, done, succ = None, False, bool(env.check_success())
        for _ in range(WAIT):
            _raw, _r, d, _i2 = env.step(np.array(DUMMY))
            done |= bool(d)
            succ |= bool(env.check_success())
            obs = _obs_now(env)
        ih, ihf = post_wait_hashes(obs)
        return dict(init_hash=ih, init_hash_full=ihf, done=done, success=succ,
                    finite=finite_obs(obs), steps=WAIT)
    finally:
        env.close()


# --- хранение банка ------------------------------------------------------------------
def task_path(space, task, bank_dir=BANK_DIR):
    return os.path.join(bank_dir, f"s{space}", f"t{task}.npz")


def save_task(space, task, cases, meta, bank_dir=BANK_DIR):
    p = task_path(space, task, bank_dir)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "wb") as fh:
        np.savez(fh, seed=np.array([c["seed"] for c in cases], np.int64),
                 idx=np.array([c["idx"] for c in cases], np.int64),
                 S0=np.stack([c["S0"] for c in cases]),
                 body_pos=np.stack([c["body_pos"] for c in cases]),
                 body_quat=np.stack([c["body_quat"] for c in cases]),
                 sig=np.array([c["sig"] for c in cases]),
                 init_hash=np.array([c["init_hash"] for c in cases]),
                 init_hash_full=np.array([c["init_hash_full"] for c in cases]),
                 meta=np.array(json.dumps(meta)))
    os.replace(p + ".tmp", p)
    return p


def load_task(space, task, bank_dir=BANK_DIR):
    z = dict(np.load(task_path(space, task, bank_dir), allow_pickle=False))
    meta = json.loads(str(z["meta"]))
    cases = [dict(case_id=case_id(space, task, int(z["idx"][i])),
                  idx=int(z["idx"][i]), seed=int(z["seed"][i]), S0=z["S0"][i],
                  body_pos=z["body_pos"][i], body_quat=z["body_quat"][i],
                  sig=str(z["sig"][i]), init_hash=str(z["init_hash"][i]),
                  init_hash_full=str(z["init_hash_full"][i]))
             for i in range(len(z["seed"]))]
    return cases, meta


# --- среда для векторного окружения K-9h -----------------------------------------------
def make_env_class():
    from libero2gym.libero_env import Libero10Env, create_libero_env

    class K33Env(Libero10Env):
        def __init__(self, dummy=False, task_id=0, image_size=224,
                     k33_mode="legacy", k33_space=1, k33_bank_dir=BANK_DIR,
                     **kw):
            self.k33_mode, self.k33_space = k33_mode, int(k33_space)
            self.k33_bank_dir, self.k33_task = k33_bank_dir, int(task_id)
            self._cases = self._meta = None
            super().__init__(dummy=dummy, task_id=task_id,
                             image_size=image_size, **kw)

        def _recreate(self, seed):
            if self.env is not None:
                try:
                    self.env.close()
                except Exception:                            # noqa: BLE001
                    pass
            if self.k33_mode == "bank":
                seed_all(seed)
            self.env, self.init_states = create_libero_env(
                SUITE, self.k33_task, self.image_size, self.camera_names,
                seed=seed)

        def reset(self, init_state_id=None):
            if self.dummy:
                return super().reset(init_state_id)
            i = int(init_state_id)
            if self.k33_mode == "legacy":
                self._recreate(0)                  # как свежий LiberoEnv
                self.env.reset()
                obs = self.set_init_state(self.init_states[i])
            elif self.k33_mode == "bank":
                if self._cases is None:
                    self._cases, self._meta = load_task(
                        self.k33_space, self.k33_task, self.k33_bank_dir)
                case = self._cases[i]
                self._recreate(case["seed"])
                restore_into(self.env, case, self._meta)
                obs = self.get_observations()
            else:
                raise Technical(f"режим {self.k33_mode}")
            self.episode_steps = 0
            return obs
    return K33Env


def make_get_envs(mode, space=1, bank_dir=BANK_DIR):
    """Подмена utils.get_envs для K-9h: воркеры создаются здесь, один раз."""
    def get_envs(task_suite, env_kwargs, n_envs):
        from functools import partial
        from libero.libero import benchmark
        from libero2gym.vector_env import AsyncVectorEnv
        if str(task_suite) != "10":
            raise Technical("K-33: только LIBERO-10")
        K33Env = make_env_class()
        kw = dict(env_kwargs, k33_mode=mode, k33_space=space,
                  k33_bank_dir=os.path.abspath(bank_dir))
        envs = AsyncVectorEnv(
            env_fns=[partial(K33Env, **kw) for _ in range(n_envs)],
            dummy_env_fn=partial(K33Env, dummy=True, **kw))
        task = benchmark.get_benchmark_dict()["libero_10"]().get_task(
            env_kwargs["task_id"])
        return envs, task.language
    return get_envs


# --- построение и проверка банка ----------------------------------------------------
def _gen_job(args):
    space, task, idx = args
    s = scene_seed(space, task, idx)
    t0 = time.time()
    try:
        g = generate(task, s)
        g.update(seed=s, idx=idx, space=space, task=task, ok=True,
                 seconds=time.time() - t0)
    except Exception as e:                                   # noqa: BLE001
        g = dict(seed=s, idx=idx, space=space, task=task, ok=False,
                 why=f"{type(e).__name__}: {e}", steps=WAIT)
    return g


def _replay_job(args):
    space, task, case, meta = args
    try:
        r = replay(task, case, meta)
        r.update(ok=True)
    except Exception as e:                                   # noqa: BLE001
        r = dict(ok=False, why=f"{type(e).__name__}: {e}", steps=WAIT)
    r.update(case_id=case["case_id"])
    return r


def official_joint_sigs(task):
    """Подписи qpos суставов официальных init (без шагов среды)."""
    env, init = _create(task, 0)
    try:
        seed_all(0)
        env.reset()
        out = []
        for st in init:
            env.set_init_state(np.asarray(st, np.float64))
            out.append(signature(env.sim, joints_only=True))
        return out
    finally:
        env.close()


def select(results, n, seen_sigs, official):
    """Первые n допустимых по idx; причины отбраковки."""
    cases, rejects = [], []
    for g in sorted(results, key=lambda r: r["idx"]):
        if len(cases) >= n:
            break
        why = None
        if not g.get("ok"):
            why = g.get("why", "ошибка reset")
        elif not g["finite"]:
            why = "нечисловое состояние/наблюдение"
        elif g["success_at_reset"] or g["success_wait"] or g["done_wait"]:
            why = "успех/done до управления"
        elif g["sig"] in seen_sigs:
            why = "дубликат физической сцены"
        elif g["sig_joints"] in official:
            why = "совпадает с официальным init"
        if why:
            rejects.append(dict(idx=g["idx"], seed=g["seed"], why=why))
            continue
        seen_sigs.add(g["sig"])
        cases.append(dict(g, case_id=case_id(g["space"], g["task"], g["idx"])))
    return cases, rejects


def selftest():
    # seed: детерминизм, различие пространств/задач/индексов
    a = scene_seed(1, 3, 7)
    assert a == scene_seed(1, 3, 7) and a != scene_seed(0, 3, 7) and \
        a != scene_seed(1, 4, 7) and a != scene_seed(1, 3, 8)
    assert case_id(0, 3, 7) != case_id(1, 3, 7)
    seeds = {scene_seed(s, t, i) for s in (0, 1) for t in range(10)
             for i in range(300 if s else 5)}
    assert len(seeds) == 3050, "коллизии seed в нужном объёме"
    # канонизация кватернионов
    q = np.array([[-0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, 0.5]])
    assert (canon_quat(q)[:, 0] >= 0).all()
    assert np.array_equal(canon_quat(np.array([-1.0, 0, 0, 0])),
                          np.array([1.0, 0, 0, 0]))

    # игрушечная модель: подпись ловит смену позы фикстуры при том же qpos
    class M:
        nbody, njnt, nq, nv = 3, 2, 8, 7
        body_pos = np.zeros((3, 3))
        body_quat = np.tile([1.0, 0, 0, 0], (3, 1))
        body_mass = np.ones(3)
        jnt_type = np.array([0, 3])
        jnt_qposadr = np.array([0, 7])

        def body_id2name(self, i):
            return ["world", "table", "obj"][i]

        def joint_id2name(self, i):
            return ["obj_free", "hinge"][i]

    class D:
        qpos = np.array([0.1, 0.2, 0.3, -1.0, 0, 0, 0, 0.5])

    class S:
        model, data = M(), D()
    sim = S()
    meta = model_meta(sim)
    s1 = signature(sim, meta)
    sim.data.qpos = np.array([0.1, 0.2, 0.3, 1.0, 0, 0, 0, 0.5])  # q ≡ −q
    assert signature(sim, meta) == s1, "знак кватерниона меняет подпись"
    for q0 in ([0.0, 1.0, 0, 0], [1e-9, 0.6, 0.8, 0], [2e-6, 0, 0.6, 0.8]):
        q0 = np.array(q0)
        assert np.array_equal(quat_key(q0), quat_key(-q0)), q0
        sim.data.qpos = np.concatenate([[0.1, 0.2, 0.3], q0, [0.5]])
        s_a = signature(sim, meta)
        sim.data.qpos = np.concatenate([[0.1, 0.2, 0.3], -q0, [0.5]])
        assert signature(sim, meta) == s_a, f"q и −q около π: {q0}"
    sim.data.qpos = np.array([0.1, 0.2, 0.3, 1.0, 0, 0, 0, 0.5])
    sj = signature(sim, meta, joints_only=True)
    sim.model.body_pos = sim.model.body_pos.copy()
    sim.model.body_pos[1, 0] += 1e-3                     # сдвиг фикстуры
    assert signature(sim, meta) != s1, "сдвиг фикстуры не замечен"
    assert signature(sim, meta, joints_only=True) == sj
    sim.data.qpos = sim.data.qpos + 2e-7                  # ниже шага 1e−6
    # отбор: первые допустимые по idx, причины, дубликаты
    base = dict(ok=True, finite=True, success_at_reset=False,
                success_wait=False, done_wait=False, space=1, task=0)
    res = [dict(base, idx=i, seed=i, sig=f"s{i % 4}", sig_joints=f"j{i}")
           for i in range(8)]
    res[1] = dict(res[1], success_wait=True)
    res[2] = dict(res[2], ok=False, why="RandomizationError")
    res[3] = dict(res[3], sig_joints="off")
    seen = set()
    cs, rj = select(res, 3, seen, {"off"})
    assert [c["idx"] for c in cs] == [0, 5, 6], [c["idx"] for c in cs]
    assert [r["why"] for r in rj] == ["успех/done до управления",
                                      "RandomizationError",
                                      "совпадает с официальным init",
                                      "дубликат физической сцены"], rj
    print("самопроверка k33_bank пройдена: seed сцен (детерминизм, без "
          "коллизий в 3050), case_id по пространствам, канонизация "
          "кватернионов (q и −q, в т. ч. w = 0 около π), подпись ловит сдвиг фикстуры при том же qpos и не "
          "зависит от знака кватерниона, отбор первых допустимых с причинами "
          "(успех/done, ошибка reset, официальный init, дубликат)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; построение банка — через k33_protocol.py")
