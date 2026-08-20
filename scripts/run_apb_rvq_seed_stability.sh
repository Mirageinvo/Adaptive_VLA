#!/usr/bin/env bash
set -euo pipefail

# Episode-selection seed stability runs for Gate-1.
# Correct CLI: --output (not --output-dir).
#
# Usage:
#   bash scripts/run_apb_rvq_seed_stability.sh
#   APB_RVQ_SEED_N_CHUNKS=512 bash scripts/run_apb_rvq_seed_stability.sh

N_CHUNKS="${APB_RVQ_SEED_N_CHUNKS:-512}"
N_EPISODES="${APB_RVQ_SEED_N_EPISODES:-256}"
RANDOM_SEEDS="${APB_RVQ_SEED_RANDOM_SEEDS:-8}"
DEVICE="${APB_RVQ_DEVICE:-cuda}"
OUT_ROOT="${APB_RVQ_OUT_ROOT:-artifacts/apb_rvq}"
SEEDS="${APB_RVQ_EPISODE_SEEDS:-1 2}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

for seed in ${SEEDS}; do
  out="${OUT_ROOT}/oracle_seed${seed}"
  echo "[seed] starting seed=${seed} n_chunks=${N_CHUNKS} -> ${out}"
  python3 experiments/rate1_oracle_greedy.py \
    --n-chunks "${N_CHUNKS}" \
    --n-episodes "${N_EPISODES}" \
    --seed "${seed}" \
    --budgets 16,20,24,28,32,36,40,44,48 \
    --random-seeds "${RANDOM_SEEDS}" \
    --static-prior-chunks 96 \
    --device "${DEVICE}" \
    --output "${out}" \
    --checkpoint-every 1
  if ! python3 experiments/rate1_recompute_summary.py --oracle-dir "${out}"; then
    echo "[seed] WARNING: recompute failed for seed=${seed}; continuing" >&2
  fi
  echo "[seed] done seed=${seed}"
done

echo "[seed] all requested seeds completed"
