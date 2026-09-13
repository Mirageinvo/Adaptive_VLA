#!/usr/bin/env bash
# GPU smoke / 200-step codec benchmark on one V100.
# Run from the Adaptive_VLA repo root inside the dedicated container.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROTOCOL="${PROTOCOL:-$ROOT/calvin_hicora/protocols/codec_protocol.json}"
DATA_ROOT="${CALVIN_DATA_ROOT:-$ROOT/data/calvin_converted/debug}"
OUT_ROOT="${CALVIN_OUTPUT_ROOT:-$ROOT/outputs/calvin_codec_gpu_smoke}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"

mkdir -p "$OUT_ROOT"

echo "data_root=$DATA_ROOT"
echo "output_root=$OUT_ROOT"
echo "device=$DEVICE"

python3 "$ROOT/calvin_hicora/scripts/check_dataloader_workers.py" "$DATA_ROOT"

python3 "$ROOT/calvin_hicora/scripts/train_codec.py" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_ROOT/base_vq_smoke" \
  --stage base_vq \
  --seed "$SEED" \
  --protocol "$PROTOCOL" \
  --device "$DEVICE" \
  --smoke

python3 "$ROOT/calvin_hicora/scripts/train_codec.py" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_ROOT/base_vq_memory" \
  --stage base_vq \
  --seed "$SEED" \
  --protocol "$PROTOCOL" \
  --device "$DEVICE" \
  --memory-smoke \
  --batch-size "${MICRO_BATCH:-256}"

python3 "$ROOT/calvin_hicora/scripts/train_codec.py" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_ROOT/base_vq_benchmark" \
  --stage base_vq \
  --seed "$SEED" \
  --protocol "$PROTOCOL" \
  --device "$DEVICE" \
  --benchmark

echo "GPU codec smoke/benchmark finished under $OUT_ROOT"
echo "Record peak_cuda_memory_mib and steps/s into FINDINGS_CALVIN.md"
