#!/usr/bin/env bash
set -euo pipefail

# Dual-GPU Phase A rate3 launcher with retry + merge.
#
# Usage:
#   bash scripts/run_apb_rvq_rate3_phase_a.sh
#   APB_RVQ_ORACLE_DIR=artifacts/apb_rvq/oracle_full bash scripts/run_apb_rvq_rate3_phase_a.sh

ORACLE_DIR="${APB_RVQ_ORACLE_DIR:-artifacts/apb_rvq/oracle_full}"
OUT_DIR="${APB_RVQ_LABELS_DIR:-artifacts/apb_rvq/labels_phase_a}"
BUDGETS="${APB_RVQ_LABEL_BUDGETS:-20,24,28,32,36,40}"
DEVICE="${APB_RVQ_DEVICE:-cuda}"
MAX_RETRIES="${APB_RVQ_MAX_RETRIES:-3}"
RETRY_SLEEP_SEC="${APB_RVQ_RETRY_SLEEP_SEC:-20}"
ENCODE_BS="${APB_RVQ_ENCODE_BATCH:-64}"
WINDOW="${APB_RVQ_WINDOW:-128}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate apb-rvq
export HF_HOME="${HF_HOME:-${HOME}/huggingface_cache}"

N_CHUNKS="$(python3 - <<PY
import numpy as np
print(int(np.load("${ORACLE_DIR}/depth_maps.npz")["depth_exact"].shape[0]))
PY
)"
MID=$((N_CHUNKS / 2))

run_with_retry() {
  local name="$1"
  shift
  local attempt=1
  while (( attempt <= MAX_RETRIES )); do
    echo "[rate3] ${name} attempt ${attempt}/${MAX_RETRIES}"
    if "$@"; then
      return 0
    fi
    if (( attempt == MAX_RETRIES )); then
      echo "[rate3] ${name} failed after ${MAX_RETRIES}" >&2
      return 1
    fi
    echo "[rate3] ${name} retry in ${RETRY_SLEEP_SEC}s" >&2
    attempt=$((attempt + 1))
    sleep "${RETRY_SLEEP_SEC}"
  done
}

echo "[rate3] n_chunks=${N_CHUNKS} mid=${MID}"
echo "[rate3] oracle=${ORACLE_DIR}"
echo "[rate3] out=${OUT_DIR}"

run_with_retry shard0 env CUDA_VISIBLE_DEVICES=0 python3 experiments/rate3_build_labels.py \
  --oracle-dir "${ORACLE_DIR}" \
  --output "${OUT_DIR}" \
  --budgets "${BUDGETS}" \
  --shard-start 0 \
  --shard-end "${MID}" \
  --device "${DEVICE}" \
  --encode-batch-size "${ENCODE_BS}" \
  --window-size "${WINDOW}" \
  --resume \
  > >(tee -a "${OUT_DIR}/logs/shard0.log") 2>&1 &
PID0=$!

run_with_retry shard1 env CUDA_VISIBLE_DEVICES=1 python3 experiments/rate3_build_labels.py \
  --oracle-dir "${ORACLE_DIR}" \
  --output "${OUT_DIR}" \
  --budgets "${BUDGETS}" \
  --shard-start "${MID}" \
  --shard-end "${N_CHUNKS}" \
  --device "${DEVICE}" \
  --encode-batch-size "${ENCODE_BS}" \
  --window-size "${WINDOW}" \
  --resume \
  > >(tee -a "${OUT_DIR}/logs/shard1.log") 2>&1 &
PID1=$!

EC0=0
EC1=0
wait ${PID0} || EC0=$?
wait ${PID1} || EC1=$?
if (( EC0 != 0 || EC1 != 0 )); then
  echo "[rate3] shard failure: shard0=${EC0} shard1=${EC1}" >&2
  exit 1
fi

python3 experiments/rate3_merge_labels.py \
  --input-dir "${OUT_DIR}" \
  --output "${OUT_DIR}/labels.parquet" | tee -a "${OUT_DIR}/logs/merge.log"

# Auto-kick rate4 on budget 28 once labels exist (single GPU).
if [[ "${APB_RVQ_AUTO_RATE4:-1}" == "1" ]]; then
  echo "[rate3] launching rate4 train on budget 28"
  CUDA_VISIBLE_DEVICES=0 python3 experiments/rate4_train_router.py \
    --labels "${OUT_DIR}/labels.parquet" \
    --output artifacts/apb_rvq/router_phase_a_b28 \
    --budgets 28 \
    --epochs "${APB_RVQ_RATE4_EPOCHS:-30}" \
    --device cuda \
    2>&1 | tee -a "${OUT_DIR}/logs/rate4_b28.log"
fi

echo "[rate3] Phase A pipeline completed"
