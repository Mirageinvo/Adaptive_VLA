#!/usr/bin/env bash
set -euo pipefail

# AATM oracle compression launcher.
#
# Usage:
#   bash scripts/run_aatm_oracle.sh smoke
#   bash scripts/run_aatm_oracle.sh medium
#   HF_HOME=~/huggingface_cache CUDA_VISIBLE_DEVICES=0 bash scripts/run_aatm_oracle.sh full

MODE="${1:-smoke}"
DEVICE="${AATM_DEVICE:-cuda}"
OUT_ROOT="${AATM_OUT_ROOT:-artifacts/merge}"
HF_CACHE_DIR="${HF_HOME:-$HOME/huggingface_cache}"

case "${MODE}" in
  smoke)
    N_CHUNKS=32
    N_EPISODES=16
    RANDOM_SEEDS=4
    ;;
  medium)
    N_CHUNKS=256
    N_EPISODES=128
    RANDOM_SEEDS=8
    ;;
  full)
    N_CHUNKS=2048
    N_EPISODES=512
    RANDOM_SEEDS=20
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected one of: smoke | medium | full" >&2
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export HF_HOME="${HF_CACHE_DIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

OUT_DIR="${OUT_ROOT}/oracle_${MODE}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${HF_HOME}"

echo "[aatm] mode=${MODE} device=${DEVICE} out=${OUT_DIR}"

python experiments/merge0_integration.py \
  --device "${DEVICE}" \
  --n-chunks "${N_CHUNKS}" \
  --n-episodes "${N_EPISODES}" \
  --output "${OUT_ROOT}/stage0_integration_${MODE}.json"

python experiments/merge0_smoke.py \
  --device "${DEVICE}" \
  --n-chunks "${N_CHUNKS}" \
  --n-episodes "${N_EPISODES}" \
  --output "${OUT_ROOT}/merge0_smoke_${MODE}.json"

python experiments/merge1_oracle_compression.py \
  --device "${DEVICE}" \
  --n-chunks "${N_CHUNKS}" \
  --n-episodes "${N_EPISODES}" \
  --random-seeds "${RANDOM_SEEDS}" \
  --output "${OUT_DIR}"

echo "[aatm] done -> ${OUT_DIR}/summary.json"
