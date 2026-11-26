#!/usr/bin/env python3
"""K-41: раннер DP-C (KDPE) с ключом шума на среду; манифест, старты, допуск.

Запуск — в окружении kdpe, из корня KDPE (относительные пути cfg):
  cd /home/malinin_aa/KDPE && source /opt/conda/bin/activate /home/malinin_aa/envs/kdpe
  MUJOCO_GL=egl python /home/malinin_aa/Adaptive_VLA/experiments/k41_runner.py <cmd>
Команды:
  manifest          версии, коммит KDPE, SHA весов и датасета, ключевые поля
                    cfg -> reports/k41/manifest.json (в Adaptive_VLA)
  starts            сиды 4610000…4610099: reset, хеши наблюдений после
                    reset -> reports/k41/starts.json
  admit             один старт (n_envs = 1): штатный predict_action с
                    генератором чанка против обёртки с нулевой головой при
                    том же ключе — действия побитово, исход, шаги; время
  bench --n N       N стартов одним батчем, нулевая голова: время, SR
Политика: веса step_79999.ckpt, EMA, фильтр passall, population 1,
n_action_steps 8; FP32, TF32 выключен, cudnn детерминированный.
Ключ шума: sha256("k41|task|start|replica|chunk") -> torch.Generator на
устройстве; начальный шум и шум каждого шага DDPM при t > 0 — из него
(шаг scheduler.step выполняется по каждой среде со своим генератором,
U-Net — одним батчем). Голова получает признаки энкодера и B и
возвращает поправку; нулевая голова -> действие B побитово.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK = "assembly_mg"
KDPE = Path("/home/malinin_aa/KDPE")
CKPT = KDPE / "data/weights/diffusion_cnn_assembly_mg/checkpoints/step_79999.ckpt"
CKPT_SHA = "eaa7fa94d22cb883329252dcb2e077ae99a6128b20d86d79101898e10af47e19"
DATA = KDPE / "data/datasets/robomimic/datasets/assembly"
START0, N_STARTS = 4610000, 100            # Screen
BENCH0 = 4600000                            # замер: старые сиды штатной оценки
DATASET = DATA / "image_abs.hdf5"           # единый путь: манифест и раннер
OUT = Path("/home/malinin_aa/Adaptive_VLA/reports/k41")
N_ACT = 8


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def noise_seed(start, replica, chunk, task=TASK):
    return int(hashlib.sha256(f"k41|{task}|{start}|{replica}|{chunk}"
                              .encode()).hexdigest()[:15], 16)


def obs_hash(obs, i):
    h = hashlib.sha1()
    for k in sorted(obs):
        a = np.ascontiguousarray(np.asarray(obs[k][i]))
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# --- окружение KDPE ------------------------------------------------------------------------------------
def kdpe_setup():
    """Импорты KDPE: локальный пакет, регистрация сред MimicGen, config KDPE
    (clearconf разбирает sys.argv при импорте — подставляем свои ключи)."""
    if Path.cwd().resolve() != KDPE.resolve():
        raise SystemExit(f"запускать из {KDPE}")
    sys.path.insert(0, str(KDPE))
    keep = sys.argv
    sys.argv = [keep[0], "--model", "cnn", "--task", TASK, "--population",
                "1", "--Filter.name", "passall", "--output_dir",
                str(OUT / "kdpe_config")]      # не трогать штатный каталог
    try:
        import mimicgen                 # регистрирует ThreePieceAssembly_D0
        import robosuite.environments.base as B
        assert mimicgen
        if "ThreePieceAssembly_D0" not in B.REGISTERED_ENVS:
            raise SystemExit("среда MimicGen не зарегистрирована")
        from config import Config
        import diffusion_policy.env_runner.robomimic_image_runner as rr
        assert rr
        if Config.shift != "none":
            raise SystemExit(f"Config.shift = {Config.shift}")
    finally:
        sys.argv = keep


def torch_setup(torch):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_policy(torch, device):
    import dill
    import hydra
    from omegaconf import open_dict, DictConfig
    if sha256(CKPT) != CKPT_SHA:
        raise SystemExit("SHA весов ≠ релизу")
    payload = torch.load(open(CKPT, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    with open_dict(cfg):
        cfg.policy.filter = DictConfig(dict(
            target="diffusion_policy.model.filters.passall", bandwidth=None,
            threshold=None, seed=None))
        cfg.policy.population = 1
        cfg.policy.n_action_steps = N_ACT
    cls = hydra.utils.get_class(cfg._target_)
    ws = cls(cfg, output_dir=str(OUT / "ws"))
    ws.load_payload(payload, exclude_keys=None, include_keys=None)
    if not cfg.training.use_ema:
        raise SystemExit("ожидалась EMA")
    policy = ws.ema_model
    policy.to(torch.device(device))
    policy.eval()
    return policy, cfg


def make_runner(cfg, start0, n, n_envs):
    import hydra
    from omegaconf import open_dict
    with open_dict(cfg):
        r = cfg.task.env_runner
        r.n_envs, r.n_test, r.n_train = n_envs, n, 0
        r.n_test_vis, r.n_train_vis = 0, 0
        r.n_action_steps = N_ACT
        r.test_start_seed = start0
        r.dataset_path = str(DATASET)
    return hydra.utils.instantiate(cfg.task.env_runner,
                                   output_dir=str(OUT / "runner"))


# --- сэмплирование с ключом на среду ------------------------------------------------------------------
def encode(policy, obs_dict):
    """Как predict_action: нормализация и признаки энкодера -> global_cond."""
    from diffusion_policy.common.pytorch_util import dict_apply
    nobs = policy.normalizer.normalize(obs_dict)
    B = next(iter(nobs.values())).shape[0]
    To = policy.n_obs_steps
    this = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
    return policy.obs_encoder(this).reshape(B, -1)


def sample(torch, policy, obs_dict, gens):
    """-> (action [B, 8, Da], action_pred [B, T, Da], global_cond)."""
    if not policy.obs_as_global_cond or policy.population != 1:
        raise SystemExit("ожидались obs_as_global_cond и population 1")
    gc = encode(policy, obs_dict)
    B, T, Da = gc.shape[0], policy.horizon, policy.action_dim
    dev, dt = policy.device, policy.dtype
    traj = torch.cat([torch.randn((1, T, Da), dtype=dt, device=dev,
                                  generator=g) for g in gens])
    sch = policy.noise_scheduler
    sch.set_timesteps(policy.num_inference_steps)
    for t in sch.timesteps:
        out = policy.model(traj, t, local_cond=None, global_cond=gc)
        traj = torch.cat([sch.step(out[i:i + 1], t, traj[i:i + 1],
                                   generator=gens[i], **policy.kwargs
                                   ).prev_sample for i in range(B)])
    pred = policy.normalizer["action"].unnormalize(traj[..., :Da])
    start = policy.n_obs_steps - 1
    return pred[:, start:start + N_ACT], pred, gc


def compose(action, delta):
    """Поза кисти: B + Δ; схват из B. Δ = None -> B."""
    if delta is None:
        return action
    out = action.clone()
    out[..., :-1] = action[..., :-1] + delta
    return out


# --- эпизоды ------------------------------------------------------------------------------------------------
def run_batch(torch, runner, policy, idx, replica, head=None, mode="ours",
              log=None):
    """Эпизоды стартов runner.env_seeds[idx] одним батчем (len(idx) ==
    n_envs). mode: ours — ключ на среду (+ голова); std — штатный
    predict_action с генератором чанка (только n_envs = 1)."""
    from diffusion_policy.common.pytorch_util import dict_apply
    env = runner.env
    n_envs = len(runner.env_fns)
    if len(idx) != n_envs:
        raise SystemExit("батч ≠ числу сред")
    seeds = [runner.env_seeds[i] for i in idx]
    env.call_each("run_dill_function",
                  args_list=[(runner.env_init_fn_dills[i],) for i in idx])
    obs = env.reset()
    hashes = [obs_hash(obs, i) for i in range(n_envs)]
    policy.reset()
    acts, done, chunk, t0 = [], False, 0, time.time()
    while not done:
        od = dict_apply(dict(obs), lambda x: torch.from_numpy(x).to(
            device=policy.device))
        gens = [torch.Generator(device=policy.device).manual_seed(
            noise_seed(s, replica, chunk)) for s in seeds]
        with torch.no_grad():
            if mode == "std":
                if n_envs != 1:
                    raise SystemExit("штатный режим — только одна среда")
                policy.kwargs["generator"] = gens[0]
                try:
                    a = policy.predict_action(od)["action"]
                finally:
                    policy.kwargs.pop("generator", None)
            else:
                a, _pred, gc = sample(torch, policy, od, gens)
                d = None if head is None else head(gc, a)
                a = compose(a, d)
        a = a.detach().cpu().numpy()
        if not np.isfinite(a).all():
            raise SystemExit("нечисловое действие")
        acts.append(a)
        obs, _r, done, _i = env.step(runner.undo_transform_action(a))
        done = bool(np.all(done))
        chunk += 1
    rew = env.call("get_attr", "reward")
    res = []
    for i in range(n_envs):
        r = np.asarray(rew[i], float)
        hit = np.flatnonzero(r >= 1)
        res.append(dict(seed=int(seeds[i]), reset_hash=hashes[i],
                        success=bool(r.max() >= 1) if len(r) else False,
                        first_success=int(hit[0]) if len(hit) else -1,
                        steps=int(len(r)), chunks=chunk))
    return res, np.stack(acts, 1), time.time() - t0


# --- команды ------------------------------------------------------------------------------------------------
def cmd_manifest(a):
    import torch
    import mujoco
    import robosuite
    import robomimic
    import diffusers
    import h5py
    kdpe_setup()
    _p, cfg = load_policy(torch, a.device)
    f = h5py.File(DATASET, "r")
    ea = json.loads(f["data"].attrs["env_args"])
    m = dict(
        kind="k41_manifest",
        kdpe_commit=subprocess.check_output(
            ["git", "-C", str(KDPE), "rev-parse", "HEAD"]).decode().strip(),
        ckpt=dict(path=str(CKPT), sha256=CKPT_SHA),
        dataset=dict(d0_sha256=sha256(DATA / "three_piece_assembly_d0.hdf5"),
                     image_abs_path=str(DATASET),
                     image_abs_sha256=sha256(DATASET),
                     n_demos=len(f["data"].keys()), env_name=ea["env_name"],
                     env_version=ea.get("env_version"),
                     env_kwargs=ea["env_kwargs"]),
        versions=dict(torch=torch.__version__, mujoco=mujoco.__version__,
                      robosuite=robosuite.__version__,
                      robomimic=getattr(robomimic, "__version__", None),
                      diffusers=diffusers.__version__,
                      python=sys.version.split()[0]),
        cfg=dict(horizon=cfg.horizon, n_obs_steps=cfg.n_obs_steps,
                 n_action_steps=N_ACT, max_steps=cfg.task.env_runner.max_steps,
                 abs_action=cfg.task.abs_action,
                 num_inference_steps=cfg.policy.num_inference_steps,
                 scheduler=dict(cfg.policy.noise_scheduler)),
        protocol=dict(starts=[START0, START0 + N_STARTS - 1], filter="passall",
                      population=1, fp32=True, tf32=False,
                      noise_key="sha256(k41|task|start|replica|chunk)[:15]"),
        gpu=torch.cuda.get_device_name(0))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(m, indent=1, default=str))
    print(json.dumps({k: m[k] for k in ("kdpe_commit", "versions")}, indent=1))
    print(f"  manifest: {OUT / 'manifest.json'}")
    return 0


def cmd_starts(a):
    import torch
    kdpe_setup()
    torch_setup(torch)
    _p, cfg = load_policy(torch, a.device)
    runner = make_runner(cfg, START0, N_STARTS, a.n_envs)
    env = runner.env
    hs = {}
    for b in range(0, N_STARTS, a.n_envs):
        idx = list(range(b, min(b + a.n_envs, N_STARTS)))
        pad = idx + [idx[0]] * (a.n_envs - len(idx))   # хвост — как в раннере
        env.call_each("run_dill_function",
                      args_list=[(runner.env_init_fn_dills[i],) for i in pad])
        obs = env.reset()
        for j, i in enumerate(idx):
            hs[str(runner.env_seeds[i])] = obs_hash(obs, j)
    if len(set(hs.values())) != N_STARTS:
        raise SystemExit("совпадающие состояния после reset")
    (OUT / "starts.json").write_text(json.dumps(dict(
        kind="k41_starts", task=TASK, start0=START0, n=N_STARTS,
        reset_hash=hs), indent=1))
    print(f"  стартов {len(hs)}, все различны; {OUT / 'starts.json'}")
    return 0


def first_diff(A, B):
    """Номер первого чанка с отличающимися действиями (-1 — совпадают)."""
    if A.shape != B.shape:
        n = min(A.shape[1], B.shape[1])
        d = [i for i in range(n) if not np.array_equal(A[:, i], B[:, i])]
        return d[0] if d else n
    d = [i for i in range(A.shape[1]) if not np.array_equal(A[:, i], B[:, i])]
    return d[0] if d else -1


def warm_reset(runner, idx):
    """Один сброс стартов idx до прогонов: кеш состояний раннера
    заполнен, и все прогоны идут одинаковым путём reset_to."""
    env = runner.env
    env.call_each("run_dill_function",
                  args_list=[(runner.env_init_fn_dills[i],) for i in idx])
    obs = env.reset()
    return [obs_hash(obs, j) for j in range(len(idx))]   # холодный сброс


def cmd_admit(a):
    """Один старт, кеш сброса заполнен заранее (одинаковый путь reset_to у
    всех прогонов). Прогоны: штатный (std1), штатный (std2), обёртка с
    нулевой головой (ours) при одном ключе. Допуск: ours = std2 побитово
    (действия, исход). std1 против std2 — воспроизводимость среды."""
    import torch
    sp = OUT / "starts.json"
    if not sp.exists():
        raise SystemExit("нет starts.json — сначала команда starts")
    st = json.loads(sp.read_text())
    kdpe_setup()
    torch_setup(torch)
    policy, cfg = load_policy(torch, a.device)
    runner = make_runner(cfg, START0, 1, 1)
    cold = warm_reset(runner, [0])[0]
    zero = lambda gc, act: torch.zeros_like(act[..., :-1])  # noqa: E731
    r1, A1, s1 = run_batch(torch, runner, policy, [0], 0, mode="std")
    r2, A2, s2 = run_batch(torch, runner, policy, [0], 0, mode="std")
    r3, A3, s3 = run_batch(torch, runner, policy, [0], 0, head=zero)
    same = A2.shape == A3.shape and np.array_equal(A2, A3) and r2 == r3
    env_rep = A1.shape == A2.shape and np.array_equal(A1, A2) and r1 == r2
    cold_ok = st["reset_hash"].get(str(START0)) == cold
    cached_ok = r1[0]["reset_hash"] == r2[0]["reset_hash"] == \
        r3[0]["reset_hash"]
    rep = dict(kind="k41_admit", start=START0, same=bool(same),
               env_repeatable=bool(env_rep), cold_reset_ok=bool(cold_ok),
               cached_reset_ok=bool(cached_ok),
               cold_vs_cached_obs_equal=bool(cold == r1[0]["reset_hash"]),
               std1=r1[0], std2=r2[0], ours=r3[0],
               first_diff=dict(std2_ours=first_diff(A2, A3),
                               std1_std2=first_diff(A1, A2)),
               chunks=[int(A1.shape[1]), int(A2.shape[1]), int(A3.shape[1])],
               seconds=dict(std1=s1, std2=s2, ours=s3))
    (OUT / "admit.json").write_text(json.dumps(rep, indent=1))
    ok = same and env_rep and cold_ok and cached_ok
    print(f"  допуск K-41: {'СОВПАДАЕТ' if ok else 'РАСХОЖДЕНИЕ'}; обёртка = "
          f"штатной: {'да' if same else 'НЕТ'}; среда воспроизводима: "
          f"{'да' if env_rep else 'НЕТ'}; холодный сброс = starts.json: "
          f"{'да' if cold_ok else 'НЕТ'}; кешированные сбросы равны: "
          f"{'да' if cached_ok else 'НЕТ'}; первый отличающийся чанк "
          f"{rep['first_diff']}; исходы std1/std2/ours "
          f"{[r1[0]['success'], r2[0]['success'], r3[0]['success']]}; время "
          f"{s1:.0f}/{s2:.0f}/{s3:.0f} с")
    return 0 if ok else 3


# --- диагностика: сэмплер на одинаковом входе, среда на одинаковых командах ----------------------
def _pop_cache(e, seed):
    e.env.env.seed_state_map.pop(seed, None)


def _sim_state(e):
    rm = e.env.env.env                     # MultiStep -> Video -> Image -> robomimic
    return (np.asarray(rm.get_state()["states"], np.float64).copy(),
            np.asarray(rm.env.sim.data.qacc_warmstart, np.float64).copy())


def _call(env, fn, **kwargs):
    """KDPE вызывает callback как fn(env): параметры — только именованные."""
    import dill
    import functools
    return env.call_each("run_dill_function", args_list=[(dill.dumps(
        functools.partial(fn, **kwargs) if kwargs else fn),)])[0]


def _act(torch, policy, od, seed, mode):
    g = torch.Generator(device=policy.device).manual_seed(seed)
    with torch.no_grad():
        if mode == "std":
            policy.kwargs["generator"] = g
            try:
                return policy.predict_action(od)["action"].cpu().numpy()
            finally:
                policy.kwargs.pop("generator", None)
        return sample(torch, policy, od, [g])[0].cpu().numpy()


def cmd_diag(a):
    """(1) На записанных o_0 и o_8 штатного прогона: std, std, ours с
    генератором одного ключа — побитово. (2) Первый чанк -> команды среды;
    дважды: сброс + эти 8 команд; сравнение наблюдения и состояния
    симулятора (и qacc_warmstart) после сброса и после 8 шагов —
    (A) кешированный reset_to, (B) штатный сброс по сиду (кеш очищается
    перед каждым сбросом)."""
    import torch
    kdpe_setup()                       # путь к локальному diffusion_policy
    from diffusion_policy.common.pytorch_util import dict_apply
    torch_setup(torch)
    policy, cfg = load_policy(torch, a.device)
    runner = make_runner(cfg, START0, a.n, 1)
    env = runner.env
    tod = lambda o: dict_apply(dict(o), lambda x: torch.from_numpy(x).to(  # noqa
        device=policy.device))
    rows = []
    for i in range(a.n):
        seed = int(runner.env_seeds[i])
        init = [(runner.env_init_fn_dills[i],)]
        env.call_each("run_dill_function", args_list=init)
        env.reset()                                    # холодный, кеш заполнен
        # (1) сэмплер
        env.call_each("run_dill_function", args_list=init)
        o0 = env.reset()
        k0 = noise_seed(seed, 0, 0)
        a0 = _act(torch, policy, tod(o0), k0, "std")
        cmd = runner.undo_transform_action(a0)
        o8, *_ = env.step(cmd)
        samp = []
        for c, o in enumerate((o0, o8)):
            k = noise_seed(seed, 0, c)
            x = [_act(torch, policy, tod(o), k, m) for m in ("std", "std",
                                                            "ours")]
            samp.append(dict(chunk=c, std_std=bool(np.array_equal(x[0], x[1])),
                             std_ours=bool(np.array_equal(x[1], x[2])),
                             max_abs=float(np.abs(x[1] - x[2]).max())))
        # (2) среда на одинаковых командах
        envr = {}
        for name, cold in (("cached", False), ("cold", True)):
            rec = []
            for _rep in range(2):
                if cold:
                    _call(env, _pop_cache, seed=seed)
                env.call_each("run_dill_function", args_list=init)
                obs = env.reset()
                h0, (s0, w0) = obs_hash(obs, 0), _call(env, _sim_state)
                obs, *_ = env.step(cmd)
                h8, (s8, w8) = obs_hash(obs, 0), _call(env, _sim_state)
                rec.append((h0, s0, w0, h8, s8, w8))
            r1, r2 = rec
            envr[name] = dict(
                obs0_equal=r1[0] == r2[0], state0_equal=bool(np.array_equal(
                    r1[1], r2[1])), warm0_equal=bool(np.array_equal(r1[2],
                                                                   r2[2])),
                obs8_equal=r1[3] == r2[3], state8_equal=bool(np.array_equal(
                    r1[4], r2[4])),
                state8_max_abs=float(np.abs(r1[4] - r2[4]).max()),
                warm0_max_abs=float(np.abs(r1[2] - r2[2]).max()))
        rows.append(dict(seed=seed, sampler=samp, env=envr))
        print(f"  старт {seed}: сэмплер "
              f"{[(x['chunk'], x['std_std'], x['std_ours']) for x in samp]}; "
              + "; ".join(f"{n}: o0 {v['obs0_equal']}, s0 {v['state0_equal']}"
                          f", warm0 {v['warm0_equal']}, o8 {v['obs8_equal']}, "
                          f"s8 {v['state8_equal']} (max {v['state8_max_abs']:.2g})"
                          for n, v in envr.items()), flush=True)
    (OUT / "diag.json").write_text(json.dumps(dict(kind="k41_diag", rows=rows),
                                              indent=1))
    return 0


def cmd_bench(a):
    import torch
    kdpe_setup()
    torch_setup(torch)
    policy, cfg = load_policy(torch, a.device)
    runner = make_runner(cfg, BENCH0, a.n, a.n)
    warm_reset(runner, list(range(a.n)))       # как в Screen: путь reset_to
    res, _A, sec = run_batch(torch, runner, policy, list(range(a.n)), 0)
    sr = np.mean([r["success"] for r in res])
    rep = dict(kind="k41_bench", n=a.n, seeds=[BENCH0, BENCH0 + a.n - 1],
               seconds=sec, sr=float(sr),
               first_success=[r["first_success"] for r in res],
               results=res)
    (OUT / f"bench_{a.n}.json").write_text(json.dumps(rep, indent=1))
    print(f"  bench: {a.n} сред одним батчем, {sec:.0f} с; SR {100 * sr:.0f}%;"
          f" медиана шага успеха "
          f"{np.median([x for x in rep['first_success'] if x >= 0] or [-1]):.0f}")
    return 0


def selftest():
    s = {noise_seed(START0 + i, r, c) for i in range(3) for r in (0, 1)
         for c in range(3)}
    assert len(s) == 18 and all(0 <= x < 2 ** 60 for x in s)
    obs = dict(a=np.arange(12).reshape(2, 6), b=np.ones((2, 3)))
    assert obs_hash(obs, 0) != obs_hash(obs, 1)
    A = np.zeros((1, 5, 8, 10))
    B = A.copy()
    B[0, 3, 0, 0] = 1
    assert first_diff(A, A) == -1 and first_diff(A, B) == 3
    assert first_diff(A, A[:, :4]) == 4
    try:
        import dill
    except ImportError:                   # локально без dill — пропуск
        dill = None

    class FE:
        def __init__(self):
            self.env = type("V", (), {})()
            self.env.env = type("I", (), {})()
            self.env.env.seed_state_map = {1: "a", 2: "b"}

        def call_each(self, name, args_list):
            return [dill.loads(args_list[0][0])(self)]
    if dill is not None:
        fe = FE()
        _call(fe, _pop_cache, seed=2)
        assert fe.env.env.seed_state_map == {1: "a"}
    print("самопроверка k41_runner пройдена: ключи шума различны по старту, "
          "реплике и чанку; хеш наблюдения по среде")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("manifest", "starts", "admit", "bench",
                                    "diag"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-envs", type=int, default=50)
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return dict(manifest=cmd_manifest, starts=cmd_starts, admit=cmd_admit,
                bench=cmd_bench, diag=cmd_diag)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
