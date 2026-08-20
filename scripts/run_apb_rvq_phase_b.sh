#!/usr/bin/env bash
set -euo pipefail

# Phase B full pipeline: wait for BAR download → dual-GPU feature extract →
# multi-budget vision router train+eval → A/B compare.
#
# Usage: bash scripts/run_apb_rvq_phase_b.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate apb-rvq
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/actioncodec:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${HOME}/huggingface_cache}"
export PYTHONUNBUFFERED=1

CKPT="${APB_RVQ_BAR_CKPT:-ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO}"
VIS="${APB_RVQ_VIS_CACHE:-artifacts/apb_rvq/visual_cache_phase_b}"
LABELS="${APB_RVQ_LABELS:-artifacts/apb_rvq/labels_phase_a/labels.parquet}"
FEAT_DIR="${APB_RVQ_BAR_FEATS:-artifacts/apb_rvq/bar_features_phase_b}"
EPOCHS="${APB_RVQ_RATE4_EPOCHS:-40}"
BS="${APB_RVQ_BAR_BATCH:-8}"

mkdir -p "${FEAT_DIR}/logs" artifacts/apb_rvq/logs_phase_b

echo "[phaseb] ensuring BAR checkpoint is local..."
python3 - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("${CKPT}")
print("[phaseb] BAR at", p)
PY

N_CHUNKS="$(python3 - <<'PY'
import numpy as np
from pathlib import Path
ids=set()
for p in Path("artifacts/apb_rvq/visual_cache_phase_b").glob("vis_*.npz"):
    ids.update(np.load(p, allow_pickle=True)["chunk_idx"].tolist())
print(max(ids)+1 if ids else 0)
PY
)"
MID=$((N_CHUNKS / 2))
echo "[phaseb] n_chunks=${N_CHUNKS} mid=${MID}"

# Dual-GPU feature extract
CUDA_VISIBLE_DEVICES=0 python3 experiments/rate3_extract_bar_features.py \
  --ckpt "${CKPT}" \
  --visual-cache "${VIS}" \
  --labels "${LABELS}" \
  --output "${FEAT_DIR}" \
  --shard-start 0 --shard-end "${MID}" \
  --batch-size "${BS}" --device cuda --resume \
  2>&1 | tee "${FEAT_DIR}/logs/feat0.log" &
PID0=$!
CUDA_VISIBLE_DEVICES=1 python3 experiments/rate3_extract_bar_features.py \
  --ckpt "${CKPT}" \
  --visual-cache "${VIS}" \
  --labels "${LABELS}" \
  --output "${FEAT_DIR}" \
  --shard-start "${MID}" --shard-end "${N_CHUNKS}" \
  --batch-size "${BS}" --device cuda --resume \
  2>&1 | tee "${FEAT_DIR}/logs/feat1.log" &
PID1=$!
wait ${PID0}
wait ${PID1}

python3 experiments/rate3_merge_bar_features.py \
  --merge-dir "${FEAT_DIR}" \
  --merge-out "${FEAT_DIR}/obs_pooled_ctx.npz" \
  --output artifacts/apb_rvq/phase_ab_compare.json \
  2>&1 | tee artifacts/apb_rvq/logs_phase_b/merge_feats.log

train_one() {
  local gpu="$1"
  local budget="$2"
  local out="artifacts/apb_rvq/router_phase_b_b${budget}"
  echo "[phaseb] GPU${gpu} train budget=${budget}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 experiments/rate4_train_router.py \
    --labels "${LABELS}" \
    --bar-features "${FEAT_DIR}" \
    --output "${out}" \
    --budgets "${budget}" \
    --epochs "${EPOCHS}" \
    --device cuda \
    2>&1 | tee "artifacts/apb_rvq/logs_phase_b/train_b${budget}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 experiments/rate5_eval_router.py \
    --labels "${LABELS}" \
    --checkpoint "${out}/best.pt" \
    --bar-features "${FEAT_DIR}" \
    --split "${out}/split.json" \
    --budget "${budget}" \
    --n-eval -1 \
    --random-seeds 20 \
    --device cuda \
    --output "${out}/eval_retained.json" \
    2>&1 | tee "artifacts/apb_rvq/logs_phase_b/eval_b${budget}.log"
}

train_one 0 24 &
PID0=$!
train_one 1 28 &
PID1=$!
wait ${PID0}; wait ${PID1}

train_one 0 32 &
PID0=$!
train_one 1 36 &
PID1=$!
wait ${PID0}; wait ${PID1}

train_one 0 20 &
PID0=$!
train_one 1 40 &
PID1=$!
wait ${PID0}; wait ${PID1}

python3 experiments/rate3_merge_bar_features.py \
  --phase-a-dir artifacts/apb_rvq \
  --phase-b-dir artifacts/apb_rvq \
  --output artifacts/apb_rvq/phase_ab_compare.json \
  | tee artifacts/apb_rvq/logs_phase_b/compare.log

python3 - <<'PY'
import json
from pathlib import Path
rows=[]
for p in sorted(Path("artifacts/apb_rvq").glob("router_phase_b_b*/eval_retained.json")):
    rows.append(json.loads(p.read_text()))
Path("artifacts/apb_rvq/router_phase_b_summary.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
PY

echo "[phaseb] DONE"
