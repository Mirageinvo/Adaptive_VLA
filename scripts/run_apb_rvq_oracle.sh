#!/usr/bin/env bash
set -euo pipefail

# Reproducible APB-RVQ oracle launcher.
#
# Usage examples:
#   bash scripts/run_apb_rvq_oracle.sh smoke
#   bash scripts/run_apb_rvq_oracle.sh medium
#   HF_HOME=/data/hf CUDA_VISIBLE_DEVICES=0 bash scripts/run_apb_rvq_oracle.sh full
#   bash scripts/watch_apb_rvq_oracle.sh full

MODE="${1:-medium}"
DEVICE="${APB_RVQ_DEVICE:-cuda}"
OUT_ROOT="${APB_RVQ_OUT_ROOT:-artifacts/apb_rvq}"
HF_CACHE_DIR="${HF_HOME:-$HOME/huggingface_cache}"
MIN_FREE_GB="${APB_RVQ_MIN_FREE_GB:-50}"
MAX_RETRIES="${APB_RVQ_MAX_RETRIES:-3}"
RETRY_SLEEP_SEC="${APB_RVQ_RETRY_SLEEP_SEC:-30}"
PREFLIGHT_STRICT="${APB_RVQ_PREFLIGHT_STRICT:-1}"
FORCE_PREFLIGHT="${APB_RVQ_FORCE_PREFLIGHT:-0}"
CHECKPOINT_EVERY="${APB_RVQ_CHECKPOINT_EVERY:-1}"

case "${MODE}" in
  smoke)
    N_CHUNKS=64
    N_EPISODES=32
    RANDOM_SEEDS=4
    STATIC_PRIOR_CHUNKS=32
    BEAM_WIDTH=32
    N_BEAM_CHUNKS=16
    EXACT_POSITIONS=6
    N_EXACT_CHUNKS=8
    ;;
  medium)
    N_CHUNKS=256
    N_EPISODES=128
    RANDOM_SEEDS=8
    STATIC_PRIOR_CHUNKS=96
    BEAM_WIDTH=64
    N_BEAM_CHUNKS=32
    EXACT_POSITIONS=8
    N_EXACT_CHUNKS=16
    ;;
  full)
    N_CHUNKS=2048
    N_EPISODES=512
    RANDOM_SEEDS=20
    STATIC_PRIOR_CHUNKS=256
    BEAM_WIDTH=128
    N_BEAM_CHUNKS=128
    EXACT_POSITIONS=8
    N_EXACT_CHUNKS=64
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

run_with_retry() {
  local attempt=1
  local max_attempts="${MAX_RETRIES}"
  while (( attempt <= max_attempts )); do
    if "$@"; then
      return 0
    fi
    if (( attempt == max_attempts )); then
      echo "[retry] all ${max_attempts} attempts failed for: $*" >&2
      return 1
    fi
    echo "[retry] attempt ${attempt}/${max_attempts} failed; retrying in ${RETRY_SLEEP_SEC}s: $*" >&2
    attempt=$((attempt + 1))
    sleep "${RETRY_SLEEP_SEC}"
  done
}

echo "[run] repo root:      ${REPO_ROOT}"
echo "[run] mode:           ${MODE}"
echo "[run] device:         ${DEVICE}"
echo "[run] hf cache:       ${HF_HOME}"
echo "[run] output dir:     ${OUT_DIR}"
echo "[run] chunks:         ${N_CHUNKS}"
echo "[run] episodes:       ${N_EPISODES}"
echo "[run] random seeds:   ${RANDOM_SEEDS}"
echo "[run] max retries:    ${MAX_RETRIES}"
echo "[run] preflight strict: ${PREFLIGHT_STRICT}"

if [[ "${FORCE_PREFLIGHT}" == "1" || ! -f "${OUT_ROOT}/preflight_${MODE}.json" ]]; then
  PREFLIGHT_ARGS=(--output "${OUT_ROOT}/preflight_${MODE}.json" --min-free-gb "${MIN_FREE_GB}")
  if [[ "${PREFLIGHT_STRICT}" == "1" ]]; then
    PREFLIGHT_ARGS+=(--strict)
  fi
  python3 experiments/rate0_preflight.py "${PREFLIGHT_ARGS[@]}" | tee "${LOG_DIR}/preflight_${MODE}.log"
else
  echo "[run] skip preflight: ${OUT_ROOT}/preflight_${MODE}.json exists"
fi

if [[ "${FORCE_PREFLIGHT}" == "1" || ! -f "${OUT_ROOT}/rate0_smoke_${MODE}.json" ]]; then
  python3 experiments/rate0_smoke.py \
    --n-chunks 32 \
    --n-episodes 16 \
    --device "${DEVICE}" \
    --output "${OUT_ROOT}/rate0_smoke_${MODE}.json" | tee "${LOG_DIR}/rate0_smoke_${MODE}.log"
else
  echo "[run] skip smoke: ${OUT_ROOT}/rate0_smoke_${MODE}.json exists"
fi

RATE1_ARGS=(
  --n-chunks "${N_CHUNKS}"
  --n-episodes "${N_EPISODES}"
  --budgets 16,20,24,28,32,36,40,44,48
  --random-seeds "${RANDOM_SEEDS}"
  --static-prior-chunks "${STATIC_PRIOR_CHUNKS}"
  --device "${DEVICE}"
  --output "${OUT_DIR}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
)

if [[ -f "${OUT_DIR}/checkpoint.json" ]]; then
  RATE1_ARGS+=(--resume)
  echo "[run] resume rate1 from checkpoint: ${OUT_DIR}/checkpoint.json"
fi

if [[ ! -f "${OUT_DIR}/metrics.json" ]]; then
  run_with_retry bash -c "
    python3 experiments/rate1_oracle_greedy.py $(printf '%q ' "${RATE1_ARGS[@]}") 2>&1 | tee -a '${LOG_DIR}/rate1_${MODE}.log'
  "
else
  echo "[run] skip rate1: ${OUT_DIR}/metrics.json exists"
fi

if [[ -f "${OUT_DIR}/metrics.json" && ! -f "${OUT_DIR}/validate_search.json" ]]; then
  RATE2_RESUME=""
  if [[ -f "${OUT_DIR}/validate_checkpoint.json" ]]; then
    RATE2_RESUME="--resume"
    echo "[run] resume rate2 from checkpoint: ${OUT_DIR}/validate_checkpoint.json"
  fi
  run_with_retry bash -c "
    python3 experiments/rate2_validate_search.py \
      --oracle-dir '${OUT_DIR}' \
      --beam-width '${BEAM_WIDTH}' \
      --n-beam-chunks '${N_BEAM_CHUNKS}' \
      --exact-positions '${EXACT_POSITIONS}' \
      --n-exact-chunks '${N_EXACT_CHUNKS}' \
      --budget 32 \
      --device '${DEVICE}' \
      --checkpoint-every '${CHECKPOINT_EVERY}' \
      ${RATE2_RESUME} 2>&1 | tee -a '${LOG_DIR}/rate2_${MODE}.log'
  "
else
  echo "[run] skip rate2: validate_search.json exists or metrics.json missing"
fi

echo "[run] completed: ${OUT_DIR}"
