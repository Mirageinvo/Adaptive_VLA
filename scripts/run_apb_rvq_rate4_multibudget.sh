#!/usr/bin/env bash
set -euo pipefail

# After Phase A labels: dual-GPU multi-budget rate4 + rate5 eval + Phase B visual cache.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate apb-rvq
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${HOME}/huggingface_cache}"

LABELS="${APB_RVQ_LABELS:-artifacts/apb_rvq/labels_phase_a/labels.parquet}"
EPOCHS="${APB_RVQ_RATE4_EPOCHS:-40}"
mkdir -p artifacts/apb_rvq/logs_rate4 artifacts/apb_rvq/visual_cache_phase_b/logs

train_one() {
  local gpu="$1"
  local budget="$2"
  local out="artifacts/apb_rvq/router_phase_a_b${budget}"
  echo "[rate4] GPU${gpu} budget=${budget}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 experiments/rate4_train_router.py \
    --labels "${LABELS}" \
    --output "${out}" \
    --budgets "${budget}" \
    --epochs "${EPOCHS}" \
    --device cuda \
    2>&1 | tee "artifacts/apb_rvq/logs_rate4/train_b${budget}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 experiments/rate5_eval_router.py \
    --labels "${LABELS}" \
    --checkpoint "${out}/best.pt" \
    --split "${out}/split.json" \
    --budget "${budget}" \
    --n-eval -1 \
    --random-seeds 20 \
    --device cuda \
    --output "${out}/eval_retained.json" \
    2>&1 | tee "artifacts/apb_rvq/logs_rate4/eval_b${budget}.log"
}

# Wave 1: B24 on GPU0, B28 on GPU1
train_one 0 24 &
PID0=$!
train_one 1 28 &
PID1=$!
wait ${PID0}
wait ${PID1}

# Wave 2: B32 GPU0, B36 GPU1
train_one 0 32 &
PID0=$!
train_one 1 36 &
PID1=$!
wait ${PID0}
wait ${PID1}

# Wave 3: B20 GPU0, B40 GPU1
train_one 0 20 &
PID0=$!
train_one 1 40 &
PID1=$!
wait ${PID0}
wait ${PID1}

# Phase B visual cache on both GPUs (I/O heavy; GPUs unused but shards parallel)
python3 experiments/rate3_cache_visuals.py \
  --oracle-dir artifacts/apb_rvq/oracle_full \
  --output artifacts/apb_rvq/visual_cache_phase_b \
  --shard-start 0 --shard-end 1024 --resume \
  2>&1 | tee artifacts/apb_rvq/visual_cache_phase_b/logs/shard0.log &
PIDV0=$!
python3 experiments/rate3_cache_visuals.py \
  --oracle-dir artifacts/apb_rvq/oracle_full \
  --output artifacts/apb_rvq/visual_cache_phase_b \
  --shard-start 1024 --shard-end 2048 --resume \
  2>&1 | tee artifacts/apb_rvq/visual_cache_phase_b/logs/shard1.log &
PIDV1=$!
wait ${PIDV0}
wait ${PIDV1}

python3 - <<'PY'
import json
from pathlib import Path
rows=[]
for p in sorted(Path("artifacts/apb_rvq").glob("router_phase_a_b*/eval_retained.json")):
    rows.append(json.loads(p.read_text()))
Path("artifacts/apb_rvq/router_phase_a_summary.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
PY

echo "[done] multi-budget rate4/5 + phase B cache"
