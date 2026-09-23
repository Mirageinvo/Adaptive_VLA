#!/usr/bin/env bash
# Host-native exclusive 2.2B memory probe on aicenter4 GPU0 (no Docker).
set -euo pipefail

ROOT="${ROOT:-/datasets/askhabaliev_g}"
REPO="${REPO:-${ROOT}/repo/Adaptive_VLA}"
VENV="${VENV:-${ROOT}/venvs/vla}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
STEPS="${STEPS:-10}"
OUT_JSON="${OUT_JSON:-${ROOT}/logs/vla_2b_memory_probe_gpu0.json}"
LOG="${LOG:-${LOG_DIR}/vla_2b_probe_gpu0_$(date +%Y%m%d_%H%M%S).log}"
VRAM_LOG="${VRAM_LOG:-${LOG_DIR}/vla_2b_probe_gpu0_vram.log}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-${ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/third_party/actioncodec:${PYTHONPATH:-}"

mkdir -p "${LOG_DIR}" "${HF_HOME}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

: > "${VRAM_LOG}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -l 1 > "${VRAM_LOG}" &
SMI_PID=$!
cleanup() { kill "${SMI_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "=== aicenter4 GPU0 probe $(date -Is) steps=${STEPS} ===" | tee -a "${LOG}"
cd "${REPO}"
set +e
# Single process — probe normally expects torchrun nproc=2; force nproc=1.
torchrun --standalone --nproc_per_node=1 \
  calvin_hicora/scripts/vla_memory_probe.py \
  --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --steps "${STEPS}" \
  --micro-batch-size 1 \
  --require-exclusive-gpus \
  --activation-checkpointing \
  --nvidia-smi-every 1 \
  --output-json "${OUT_JSON}" 2>&1 | tee -a "${LOG}"
RC=${PIPESTATUS[0]}
set -e
cleanup
trap - EXIT

python3 - <<PY | tee -a "${LOG}"
from pathlib import Path
rows={}
for line in Path("${VRAM_LOG}").read_text().splitlines():
    if not line.strip() or line.lower().startswith("index"): continue
    p=[x.strip() for x in line.split(",")]
    if len(p)<2: continue
    try:
        i=int(p[0]); m=float(p[1].replace("MiB","").strip())
    except ValueError: continue
    rows.setdefault(i,[]).append(m)
for i,v in sorted(rows.items()):
    print(f"GPU{i}: n={len(v)} peak={max(v):.1f} min={min(v):.1f} delta={max(v)-min(v):.1f} MiB")
PY
echo "=== DONE rc=${RC} $(date -Is) ===" | tee -a "${LOG}"
exit "${RC}"
