#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${APB_RVQ_LABELS_DIR:-artifacts/apb_rvq/labels_phase_a}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

while true; do
  echo "===== $(date -Is) ====="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader || true
  ls -lh "${OUT_DIR}"/shard_*.parquet 2>/dev/null || echo "(no shards yet)"
  ls -lh "${OUT_DIR}"/*.checkpoint.json 2>/dev/null || true
  if [[ -f "${OUT_DIR}/labels.parquet" ]]; then
    echo "labels.parquet ready"
    ls -lh "${OUT_DIR}/labels.parquet" "${OUT_DIR}/label_stats.json" 2>/dev/null || true
  fi
  tail -n 3 "${OUT_DIR}/logs/shard0.log" 2>/dev/null || true
  tail -n 3 "${OUT_DIR}/logs/shard1.log" 2>/dev/null || true
  sleep "${APB_RVQ_WATCH_SLEEP:-30}"
done
