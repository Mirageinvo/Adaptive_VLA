#!/usr/bin/env bash
# Host-native (no Docker GPU) VLA production train on aicenter4 GPU0.
set -euo pipefail

ROOT="${ROOT:-/datasets/askhabaliev_g}"
REPO="${REPO:-${ROOT}/repo/Adaptive_VLA}"
VENV="${VENV:-${ROOT}/venvs/vla}"
CONFIG="${CONFIG:-config/train/bar_calvin_production_s1_h100_1gpu.yaml}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
LOG="${LOG:-${LOG_DIR}/vla_production_s1_$(date +%Y%m%d_%H%M%S).log}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-${ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/third_party/actioncodec:${PYTHONPATH:-}"

mkdir -p "${LOG_DIR}" "${HF_HOME}" \
  "${ROOT}/checkpoints/vla/production_s1"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

cd "${REPO}/third_party/actioncodec"
echo "=== aicenter4 production GPU0 $(date -Is) ===" | tee -a "${LOG}"
echo "cuda_visible=${CUDA_VISIBLE_DEVICES} config=${CONFIG}" | tee -a "${LOG}"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "${LOG}"

python scripts/train_vla.py --config "${CONFIG}" 2>&1 | tee -a "${LOG}"
