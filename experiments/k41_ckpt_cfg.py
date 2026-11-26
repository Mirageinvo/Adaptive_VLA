#!/usr/bin/env python3
"""K-41: прочитать конфигурацию релизного чекпойнта KDPE (DP-C) без запуска.

Нужны только torch, dill, omegaconf (код diffusion_policy не нужен:
сохранённые объекты в payload["pickles"] остаются байтами).
Печатает то, что закрепляет протокол среды и политики: задачу и путь
датасета (вариант D), shape_meta (камеры, низкоразмерные наблюдения,
действие), abs_action / rotation_rep, horizon / n_obs_steps /
n_action_steps, scheduler и число шагов диффузии, параметры раннера
(лимит эпизода, сиды), use_ema; число параметров model / ema_model; SHA256
файла. Полный cfg сохраняется в YAML.

  python3 k41_ckpt_cfg.py --ckpt .../step_79999.ckpt --out reports/k41/cfg_assembly.yaml
"""
import argparse
import hashlib
from pathlib import Path


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def pick(cfg, path):
    from omegaconf import OmegaConf
    try:
        v = OmegaConf.select(cfg, path)
    except Exception as e:                      # noqa: BLE001
        return f"<ошибка: {e}>"
    if v is None:
        return None
    return OmegaConf.to_container(v, resolve=False) if \
        OmegaConf.is_config(v) else v


KEYS = ("name", "task_name", "task.name", "task.dataset_path",
        "task.abs_action", "task.dataset.rotation_rep",
        "task.dataset._target_", "task.shape_meta",
        "task.env_runner._target_", "task.env_runner.dataset_path",
        "task.env_runner.max_steps", "task.env_runner.n_obs_steps",
        "task.env_runner.n_action_steps", "task.env_runner.render_obs_key",
        "task.env_runner.abs_action", "task.env_runner.n_test",
        "task.env_runner.test_start_seed", "task.env_runner.n_envs",
        "horizon", "n_obs_steps", "n_action_steps", "n_latency_steps",
        "past_action_visible", "policy._target_",
        "policy.num_inference_steps", "policy.obs_as_global_cond",
        "policy.noise_scheduler", "policy.crop_shape",
        "policy.obs_encoder_group_norm", "policy.eval_fixed_crop",
        "training.use_ema", "training.seed", "training.num_epochs")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import dill
    import torch
    from omegaconf import OmegaConf
    print(f"  SHA256 {a.ckpt.name}: {sha256(a.ckpt)}", flush=True)
    payload = torch.load(open(a.ckpt, "rb"), map_location="cpu",
                         pickle_module=dill)
    print(f"  ключи payload: {sorted(payload)}")
    cfg = payload["cfg"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(OmegaConf.to_yaml(cfg, resolve=False))
    print(f"  полный cfg: {a.out}")
    for k in KEYS:
        print(f"  {k}: {pick(cfg, k)}")
    sd = payload.get("state_dicts", {})
    for name, st in sd.items():
        if hasattr(st, "items"):
            n = sum(v.numel() for v in st.values() if hasattr(v, "numel"))
            print(f"  state_dicts[{name}]: тензоров {len(st)}, параметров "
                  f"{n:,}")
        else:
            print(f"  state_dicts[{name}]: {type(st).__name__}")
    print(f"  pickles: {sorted(payload.get('pickles', {}))}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
