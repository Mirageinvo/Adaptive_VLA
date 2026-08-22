#!/usr/bin/env bash
set -euo pipefail

# AATM oracle compression launcher.
#
# Usage:
#   bash scripts/run_aatm_oracle.sh smoke
#   bash scripts/run_aatm_oracle.sh medium
#   AATM_NUM_GPUS=2 bash scripts/run_aatm_oracle.sh medium

MODE="${1:-smoke}"
NUM_GPUS="${AATM_NUM_GPUS:-2}"
OUT_ROOT="${AATM_OUT_ROOT:-artifacts/merge}"
HF_CACHE_DIR="${HF_HOME:-$HOME/huggingface_cache}"

case "${MODE}" in
  smoke)
    N_CHUNKS=32
    N_EPISODES=16
    RANDOM_SEEDS=4
    NUM_GPUS=1
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

run_stage0() {
  local gpu="$1"
  CUDA_VISIBLE_DEVICES="${gpu}" python experiments/merge0_integration.py \
    --device cuda \
    --n-chunks "${N_CHUNKS}" \
    --n-episodes "${N_EPISODES}" \
    --output "${OUT_ROOT}/stage0_integration_${MODE}.json"
  CUDA_VISIBLE_DEVICES="${gpu}" python experiments/merge0_smoke.py \
    --device cuda \
    --n-chunks "${N_CHUNKS}" \
    --n-episodes "${N_EPISODES}" \
    --output "${OUT_ROOT}/merge0_smoke_${MODE}.json"
}

run_merge1_shard() {
  local gpu="$1"
  local shard="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" python experiments/merge1_oracle_compression.py \
    --device cuda \
    --n-chunks "${N_CHUNKS}" \
    --n-episodes "${N_EPISODES}" \
    --random-seeds "${RANDOM_SEEDS}" \
    --shard-id "${shard}" \
    --num-shards "${NUM_GPUS}" \
    --output "${OUT_DIR}" \
    > "${LOG_DIR}/merge1_${MODE}_gpu${gpu}_shard${shard}.log" 2>&1
}

echo "[aatm] mode=${MODE} num_gpus=${NUM_GPUS} out=${OUT_DIR}"

if [[ "${NUM_GPUS}" -le 1 ]]; then
  run_stage0 0
  run_merge1_shard 0 0
  if [[ "${NUM_GPUS}" -eq 1 ]]; then
    # summary written by merge1 when num_shards=1
    echo "[aatm] done -> ${OUT_DIR}/summary.json"
    exit 0
  fi
else
  run_stage0 0
  pids=()
  for ((shard=0; shard<NUM_GPUS; shard++)); do
    run_merge1_shard "${shard}" "${shard}" &
    pids+=("$!")
  done
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      fail=1
    fi
  done
  if [[ "${fail}" -ne 0 ]]; then
    echo "[aatm] one or more shards failed" >&2
    exit 1
  fi
  python experiments/merge1_merge_shards.py \
    --input-dir "${OUT_DIR}" \
    --random-seeds "${RANDOM_SEEDS}" \
    --output "${OUT_DIR}/summary.json"
fi

echo "[aatm] done -> ${OUT_DIR}/summary.json"
