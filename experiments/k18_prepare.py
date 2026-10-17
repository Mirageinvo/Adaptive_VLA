#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-18 (подготовка, CPU): PCA-базис поправок, расписание, конфигурация.

PCA: r* трёх позиций окон половины B K-17 -> исполняемые единицы (× scale
руки харнесса) -> 48 чисел; взвешенная центрированная ковариация (задачи
равны, эпизоды равны, окна делят вес эпизода, позиции равны); четыре
ведущих собственных вектора; знак: наибольшая по модулю компонента
положительна. Средняя поправка к действию не добавляется.

КОДИРОВЩИК h18: G_pre.enc из data/k17/models.pt (до совместного обучения),
закреплён SHA содержимого.

ПРЕДЕЛ ρ: clip_env_norm data/k16/models.pt (без округления).
ПРЕДЕЛЫ КОМАНД: входные пределы контроллера установленной среды (OSC) —
измеряются на реальной среде (--check-env); поправка уменьшается общим
множителем, чтобы a0 + δ не выходило за них (без скрытой обрезки).

СТАРТЫ: обучение 0–24, промежуточная оценка 25–29, итоговая 30–44,
резерв 45–49 закрыт. Расписание обучения — волны по 5 сред одной задачи;
в каждом круге задачи в случайном порядке, внутри задачи старты по
перестановке (каждые 5 волн задачи покрывают 25 стартов); одинаково для
всех вариантов. Общий набор q0 — первые 5 стартов перестановки каждой
задачи.

ВЫХОД: data/k18/config.json, data/k18/pca.npz, data/k18/enc.pt.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUTDIR = "data/k18"
TASKS = tuple(range(10))
N_ENVS = 5
TRAIN_STARTS = tuple(range(0, 25))
MID_STARTS = tuple(range(25, 30))
FINAL_STARTS = tuple(range(30, 45))
RESERVED = tuple(range(45, 50))
VARIANTS = ("full48_rl", "pca4_rl", "basis4_rl")
RANK = 4
SEEDS = dict(train=0, schedule=20261017, explore=20261018, batch=20261019)
N_SCHEDULE = 3000
BUDGET = dict(first=100_000, final=300_000)
TD3 = dict(lr=3e-4, batch=128, tau=0.005, policy_delay=2, utd=4, gamma=1.0,
           capacity=100_000, hidden=256, critic_start=128,
           init_noise_episodes=64)
NOISE = dict(explore_sigma=0.15, explore_cap=0.30, target_sigma=0.10,
             target_cap=0.20)
ENV = dict(suite="10", horizon=8, max_steps=600, waiting_steps=10,
           ensemble="off", seed=101, rollout_seed_mode="fixed", n_envs=5,
           ckpt="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
CODE = ("k18_prepare.py", "k18_models.py", "k18_td3.py", "k18_collector.py",
        "k18_train.py", "k18_analyze.py", "k18_run.sh", "k16_models.py",
        "k16_policy.py", "k17_models.py", "k17_policy.py", "k17_train.py",
        "k15f_continuous_refine.py", "k15f_policy.py", "k15h_protocol.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def state_sha(state):
    h = hashlib.sha1()
    for k in sorted(state):
        h.update(k.encode())
        h.update(np.ascontiguousarray(state[k].detach().cpu().numpy()
                                      .astype(np.float32)).tobytes())
    return h.hexdigest()


def weighted_pca(X, w, k=RANK):
    """X [N, 48], w [N] -> (U [48, k], собств. значения по убыванию,
    доли объяснённой дисперсии, среднее)."""
    X = np.asarray(X, np.float64)
    w = np.asarray(w, np.float64)
    w = w / w.sum()
    mu = (w[:, None] * X).sum(0)
    Xc = X - mu
    C = (w[:, None] * Xc).T @ Xc
    ev, V = np.linalg.eigh(C)
    order = np.argsort(ev)[::-1]
    ev, V = ev[order], V[:, order]
    U = V[:, :k].copy()
    for j in range(k):                       # знак: наибольшая |компонента| > 0
        i = int(np.argmax(np.abs(U[:, j])))
        if U[i, j] < 0:
            U[:, j] = -U[:, j]
    return U, ev, ev / ev.sum(), mu


def pca_from_windows(windows, scale):
    import k17_train as kt
    wj = json.load(open(windows[:-4] + ".json"))
    if wj.get("kind") != "k17_windows" or wj["out"]["sha1"] != sha(windows):
        raise SystemExit("окна K-17 не те или изменены")
    Z = np.load(windows, allow_pickle=False)
    rs = np.asarray(Z["B__rstar"], np.float64)            # [N, 3, 8, 6]
    w = kt.window_weights(Z["B__task"], Z["B__episode"])
    X = (rs * np.asarray(scale[:6])).reshape(len(rs) * 3, 48)
    U, ev, ratio, mu = weighted_pca(X, np.repeat(w, 3))
    return dict(U=U, eigvals=ev, explained=ratio, mean_exec=mu,
                n_windows=len(rs), n_samples=len(X))


def make_schedule(seed=SEEDS["schedule"], n=N_SCHEDULE):
    """q0-набор и расписание волн обучения (одинаковое для вариантов)."""
    rng = np.random.default_rng(seed)
    perm = {t: [int(x) for x in rng.permutation(TRAIN_STARTS)]
            for t in TASKS}
    pos = {t: 0 for t in TASKS}

    def take(t):
        out = []
        for _ in range(N_ENVS):
            k = pos[t] % len(TRAIN_STARTS)
            if k == 0 and pos[t] > 0:                   # новая перестановка
                perm[t] = [int(x) for x in rng.permutation(TRAIN_STARTS)]
            out.append(perm[t][k])
            pos[t] += 1
        return out
    q0 = [dict(task=t, starts=take(t)) for t in TASKS]
    waves = []
    while len(waves) < n:
        for t in rng.permutation(TASKS):
            waves.append(dict(task=int(t), starts=take(int(t))))
    visits = {}
    for wv in q0 + waves:
        for s in wv["starts"]:
            key = f"{wv['task']}:{s}"
            wv.setdefault("visit", []).append(visits.get(key, 0))
            visits[key] = visits.get(key, 0) + 1
    return q0, waves[:n]


def env_limits():
    """Пределы входа контроллера руки и пространства действий среды."""
    import gym
    import libero2gym  # noqa: F401
    env = gym.make("libero-10-v0", task_id=0, image_size=224)
    try:
        u = env.unwrapped
        rob = u.env.env.robots[0]
        c = rob.controller
        lo = np.asarray(c.input_min, np.float64).reshape(-1)
        hi = np.asarray(c.input_max, np.float64).reshape(-1)
        if lo.size == 1:
            lo, hi = np.full(6, lo[0]), np.full(6, hi[0])
        return dict(controller=type(c).__name__, input_min=lo[:6].tolist(),
                    input_max=hi[:6].tolist(),
                    action_low=np.asarray(env.action_space.low).tolist(),
                    action_high=np.asarray(env.action_space.high).tolist(),
                    control_freq=float(getattr(u.env.env, "control_freq",
                                               -1)))
    finally:
        env.close()


def cold_refs():
    """Холодные эталоны q0: K-15h q0ref (блоки 25, 30, 35) и q0 K-17
    (блок 40) всех задач."""
    import glob
    out = {}
    c = sorted(glob.glob("reports/k15h/run/p*/"))
    if len(c) != 1:
        raise SystemExit(f"каталог K-15h неоднозначен: {c}")
    p17 = json.load(open("reports/k17/plan.json"))
    import k17_protocol as pc17
    r17 = os.path.join("reports/k17/run", f"p{pc17.sha('reports/k17/plan.json')}")
    assert p17["kind"] == "k17_plan"
    for t in TASKS:
        for b in (25, 30, 35):
            out[f"{t}:{b}"] = os.path.join(c[0], f"q0ref_t{t}_i{b}.json")
        out[f"{t}:40"] = os.path.join(r17, f"q0_t{t}_i40.json")
    miss = [f for f in out.values() if not os.path.exists(f)]
    if miss:
        raise SystemExit(f"нет холодных эталонов q0: {miss[:3]}")
    return out


def main_prepare(a):
    if os.path.exists(os.path.join(OUTDIR, "run", "state.pt")):
        raise SystemExit("обучение K-18 уже начато (data/k18/run/state.pt): "
                         "конфигурацию не перезаписываю")
    import torch
    import k15h_protocol as hpc
    import k17_models as m17
    os.makedirs(OUTDIR, exist_ok=True)
    scale = hpc.harness_scale()
    ck16 = torch.load(a.k16_models, map_location="cpu", weights_only=False)
    rho = float(ck16["clip_env_norm"])
    if not np.allclose(np.asarray(ck16["scale_arm"]), scale[:6]):
        raise SystemExit("scale_arm K-16 не равен масштабу харнесса")
    P = pca_from_windows(a.windows, scale)
    U = P["U"]
    dev = float(np.abs(U.T @ U - np.eye(RANK)).max())
    if dev > 1e-10:
        raise SystemExit(f"PCA-базис не ортонормален: {dev:.2e}")
    np.savez(os.path.join(OUTDIR, "pca.npz"), **P)
    ck17 = torch.load(a.k17_models, map_location="cpu", weights_only=False)
    M = m17.build(torch, int(ck17["d_model"]), int(ck17["n_pos"]))
    G = M["G"]()
    G.load_state_dict(ck17["G_pre"])
    enc_state = {k: v.clone() for k, v in G.enc.state_dict().items()}
    torch.save(dict(state=enc_state, d_model=int(ck17["d_model"]),
                    n_pos=int(ck17["n_pos"]), source=os.path.abspath(
                        a.k17_models), source_sha1=sha(a.k17_models)),
               os.path.join(OUTDIR, "enc.pt"))
    lim = env_limits() if a.check_env else None
    if lim is not None:
        lo, hi = np.asarray(lim["input_min"]), np.asarray(lim["input_max"])
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
                and np.all(hi > lo)):
            raise SystemExit(f"пределы контроллера неверны: {lim}")
    q0_set, waves = make_schedule()
    inv = json.load(open("reports/k15h/inventory.json"))
    bank = {t: dict(path=v["path"], sha1=v["sha1"])
            for t, v in inv["libero_bank"].items()}
    for t, v in bank.items():
        if not os.path.exists(v["path"]) or sha(v["path"]) != v["sha1"]:
            raise SystemExit(f"файл стартов задачи {t} изменён")
    refs = cold_refs()
    pca_p = os.path.join(OUTDIR, "pca.npz")
    enc_p = os.path.join(OUTDIR, "enc.pt")
    cfg = dict(
        kind="k18_config", variants=list(VARIANTS), rank=RANK,
        tasks=list(TASKS), n_envs=N_ENVS, starts=dict(
            train=list(TRAIN_STARTS), mid=list(MID_STARTS),
            final=list(FINAL_STARTS), reserved=list(RESERVED)),
        rho=rho, scale=scale.tolist(),
        bounds=(dict(lo=lim["input_min"], hi=lim["input_max"])
                if lim else None), env_limits=lim,
        seeds=SEEDS, budget=BUDGET, td3=TD3, noise=NOISE, env=ENV,
        window=dict(start="первое причинное событие схвата (Tracker K-16)",
                    calls=3),
        q0_set=q0_set, schedule=waves,
        pca=dict(path=os.path.abspath(pca_p), sha1=sha(pca_p),
                 explained=P["explained"][:8].tolist(),
                 explained4=float(P["explained"][:RANK].sum()),
                 sign="наибольшая по модулю компонента положительна"),
        enc=dict(path=os.path.abspath(enc_p), sha1=sha(enc_p),
                 state_sha1=state_sha(enc_state)),
        inputs=dict(windows=os.path.abspath(a.windows),
                    windows_sha1=sha(a.windows),
                    k16_models=os.path.abspath(a.k16_models),
                    k16_models_sha1=sha(a.k16_models),
                    k17_models=os.path.abspath(a.k17_models),
                    k17_models_sha1=sha(a.k17_models),
                    harness=os.path.join(HERE, "k9h_multiarm_gate.py"),
                    harness_sha1=sha(os.path.join(HERE,
                                                  "k9h_multiarm_gate.py")),
                    harness_utils=hpc.HARNESS_UTILS,
                    harness_utils_sha1=sha(hpc.HARNESS_UTILS)),
        libero_bank=bank,
        cold_refs={k: dict(path=os.path.abspath(v), sha1=sha(v))
                   for k, v in refs.items()},
        code={f: sha(os.path.join(HERE, f)) for f in CODE
              if os.path.exists(os.path.join(HERE, f))},
        note=("старты 25–44 отложены от RL-обучения K-18, но использовались "
              "в проекте раньше; резерв 45–49 и val_confirm закрыты"))
    out = os.path.join(OUTDIR, "config.json")
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=1)
    os.replace(out + ".tmp", out)
    print(f"  PCA: доли первых 8 компонент "
          f"{np.round(P['explained'][:8], 4).tolist()}, четыре — "
          f"{P['explained'][:RANK].sum():.3f}; окон {P['n_windows']}")
    print(f"  ρ = {rho!r}; масштаб руки {np.round(scale[:6], 4).tolist()}")
    print(f"  пределы контроллера: {lim}")
    print(f"  расписание: q0-набор {len(q0_set)} волн, обучение "
          f"{len(waves)} волн; конфигурация {out} (sha {sha(out)[:12]})")
    return 0


def selftest():
    rng = np.random.default_rng(0)
    # PCA: известные направления восстанавливаются, веса учитываются
    d = rng.standard_normal((48, 4))
    Q, _ = np.linalg.qr(d)
    X = (rng.standard_normal((4000, 4)) * [5, 3, 2, 1]) @ Q.T + \
        rng.standard_normal((4000, 48)) * 0.05
    U, ev, ratio, mu = weighted_pca(X, np.ones(len(X)))
    assert np.allclose(np.abs(U.T @ Q), np.eye(4), atol=0.05)
    assert np.allclose(U.T @ U, np.eye(4), atol=1e-12)
    assert all(U[np.argmax(np.abs(U[:, j])), j] > 0 for j in range(4))
    assert ev[0] >= ev[1] >= ev[2] and abs(ratio.sum() - 1) < 1e-9
    w = np.ones(len(X))
    w[:2000] = 0
    _u, _e, _r, mu2 = weighted_pca(X + np.r_[np.zeros((2000, 48)),
                                             np.ones((2000, 48))], w)
    assert np.allclose(mu2, X[2000:].mean(0) + 1, atol=1e-9)
    # расписание: одинаковость, равномерность, только обучающие старты
    q0a, wa = make_schedule()
    q0b, wb = make_schedule()
    assert q0a == q0b and wa == wb
    starts = [s for wv in q0a + wa for s in wv["starts"]]
    assert set(starts) <= set(TRAIN_STARTS)
    from collections import Counter
    tc = Counter(wv["task"] for wv in wa[:1000])
    assert set(tc.values()) == {100}
    per = Counter((wv["task"], s) for wv in q0a + wa[:500] for s in
                  wv["starts"])
    assert max(per.values()) - min(per.values()) <= 1
    assert all(len(set(wv["starts"])) == N_ENVS for wv in q0a + wa[:500])
    assert q0a[0]["visit"] == [0] * 5
    print("самопроверка k18_prepare пройдена: взвешенный PCA (направления, "
          "знак, среднее), расписание одинаково и равномерно, только "
          "обучающие старты")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-18: подготовка")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--windows", default="reports/k17/windows.npz")
    ap.add_argument("--k16-models", default="data/k16/models.pt")
    ap.add_argument("--k17-models", default="data/k17/models.pt")
    ap.add_argument("--check-env", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        return main_prepare(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
