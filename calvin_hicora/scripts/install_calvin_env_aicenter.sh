#!/usr/bin/env bash
# Install official mees/calvin_env (PyBullet) on aicenter and verify physics dt=1/30.
# No mocks. Gym id calvin_pred-v0 is NOT in upstream — verify via official PlayTableSimEnv.
set -euo pipefail

REPO="${REPO:-/home/askhabaliev_gs/Adaptive_VLA-1}"
SRC="${REPO}/third_party/calvin_env_source"
VENV="${VENV:-/home/askhabaliev_gs/venvs/calvin_env}"
LOG_DIR="${REPO}/logs"
LOG="${LOG_DIR}/calvin_env_compile_$(date +%Y%m%d_%H%M%S).log"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel}"
VAL_DATASET="${VAL_DATASET:-/datasets/askhabaliev_gs/calvin_full/task_D_D/task_D_D/validation}"

mkdir -p "${LOG_DIR}" "$(dirname "${VENV}")"
exec > >(tee -a "${LOG}") 2>&1

echo "=== calvin_env compile $(date -Is) ==="
echo "log=${LOG} venv=${VENV}"
echo "NOTE: upstream calvin_env physics backend is PyBullet (not MuJoCo)."

if [ ! -d "${SRC}/.git" ]; then
  git clone --recurse-submodules https://github.com/mees/calvin_env.git "${SRC}"
fi
git -C "${SRC}" submodule update --init --recursive

docker run --rm --network host \
  -v /home/askhabaliev_gs:/home/askhabaliev_gs \
  -v /datasets/askhabaliev_gs:/datasets/askhabaliev_gs \
  -e HOME=/home/askhabaliev_gs \
  -e PIP_CACHE_DIR=/tmp/pip-cache \
  -w "${SRC}" \
  "${IMAGE}" \
  bash -lc '
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
mkdir -p /tmp/pip-cache
export PIP_CACHE_DIR=/tmp/pip-cache
apt-get update
apt-get install -y --no-install-recommends \
  libgl1-mesa-glx libgl1-mesa-dev libosmesa6-dev \
  libglew-dev libglfw3 libglfw3-dev \
  patchelf freeglut3-dev \
  build-essential git curl ca-certificates \
  libglib2.0-0 libsm6 libxext6 libxrender1 \
  python3-venv

VENV=/home/askhabaliev_gs/venvs/calvin_env
SRC=/home/askhabaliev_gs/Adaptive_VLA-1/third_party/calvin_env_source
python -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -U pip setuptools wheel
# Pin hydra/antlr stack; prefer binary wheels (no PIP_TARGET).
python -m pip install --no-cache-dir \
  "numpy<2" \
  "antlr4-python3-runtime==4.9.3" \
  cloudpickle GitPython "gym==0.26.2" \
  "hydra-core==1.3.2" hydra-colorlog \
  matplotlib numba numpy-quaternion omegaconf \
  opencv-python pandas pybullet scipy pillow pyyaml

cd "${SRC}/tacto"
python -m pip install --no-cache-dir --no-deps -e .
cd "${SRC}"
python -m pip install --no-cache-dir --no-deps -e .

python - <<'PY'
print("=== import gate ===")
import calvin_env
print("calvin_env", calvin_env.__file__)
import gym
print("gym", gym.__version__, gym.__file__)
import tacto
print("tacto", tacto.__file__, "has_Sensor", hasattr(tacto, "Sensor"))
try:
    env = gym.make("calvin_env:calvin_pred-v0")
    print("UNEXPECTED: gym.make calvin_pred-v0 worked", env)
except Exception as exc:
    print("EXPECTED_NO_GYM_ID:", type(exc).__name__, str(exc)[:240])

from pathlib import Path
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate

val = Path("/datasets/askhabaliev_gs/calvin_full/task_D_D/task_D_D/validation")
cfg_path = val / ".hydra" / "merged_config.yaml"
assert cfg_path.is_file(), cfg_path
cfg = OmegaConf.load(cfg_path)
OmegaConf.resolve(cfg)
with open_dict(cfg):
    # Headless DIRECT: real PyBullet physics, no A100 EGL fight.
    cfg.env.use_egl = False
    cfg.env.show_gui = False
    cfg.env.use_vr = False
    # RGB VLA eval uses static+gripper; tactile needs DIGIT assets.
    if "tactile" in cfg.cameras:
        del cfg.cameras["tactile"]
    cfg.env.cameras = cfg.cameras
env = instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)
dt = 1.0 / float(env.control_freq)
print("SUCCESS_PHYSICS:", f"{dt:.5f}")
print(
    "control_freq=",
    env.control_freq,
    "action_repeat=",
    getattr(env, "action_repeat", None),
    "backend=pybullet",
    "cameras=",
    list(cfg.cameras.keys()),
)
obs = env.reset()
print(
    "reset_ok",
    type(obs).__name__ if not isinstance(obs, dict) else sorted(obs.keys())[:12],
)
env.close()
print("SUCCESS_IMPORT_AND_PHYSICS")
PY
'

echo "=== DONE $(date -Is) ==="
