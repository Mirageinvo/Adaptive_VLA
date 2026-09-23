#!/usr/bin/env bash
# Bootstrap aicenter4 host env for single-GPU0 VLA train (no Docker GPU).
set -euo pipefail

ROOT="${ROOT:-/datasets/askhabaliev_g}"
REPO="${REPO:-${ROOT}/repo/Adaptive_VLA}"
VENV="${VENV:-${ROOT}/venvs/vla}"
PY="${PY:-python3}"

mkdir -p "${ROOT}"/{checkpoints,logs,hf_cache,repo} "$(dirname "${VENV}")"

if [ ! -d "${REPO}/.git" ]; then
  git clone --branch HiCoRA-VLA --single-branch \
    https://github.com/Mirageinvo/Adaptive_VLA.git "${REPO}"
else
  git -C "${REPO}" fetch origin HiCoRA-VLA
  git -C "${REPO}" checkout HiCoRA-VLA
  git -C "${REPO}" pull --ff-only origin HiCoRA-VLA || true
fi

if [ ! -x "${VENV}/bin/python" ]; then
  "${PY}" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip setuptools wheel

# CUDA 12.x wheels run on driver 595 / CUDA 13.2 hosts.
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision torchaudio

pip install \
  "lightning>=2.4,<2.6" \
  omegaconf peft accelerate einops \
  "transformers>=4.45" safetensors huggingface_hub \
  pillow numpy

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "n", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu0", torch.cuda.get_device_name(0))
PY

echo "BOOTSTRAP_OK repo=${REPO} venv=${VENV}"
