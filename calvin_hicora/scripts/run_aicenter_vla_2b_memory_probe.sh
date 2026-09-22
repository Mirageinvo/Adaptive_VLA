#!/usr/bin/env bash
# Honest dual-GPU FSDP memory probe for SmolVLM2-2.2B-Instruct on aicenter.
# Fail-closed: --require-exclusive-gpus aborts if foreign CUDA apps hold VRAM.
# Do NOT treat 256M overfit VRAM deltas as a 2.2B forecast.
set -euo pipefail

HOST_REPO="${HOST_REPO:-/home/askhabaliev_gs/Adaptive_VLA-1}"
HOST_PYDEPS="${HOST_PYDEPS:-/home/askhabaliev_gs/pydeps}"
HOST_HF="${HOST_HF:-/home/askhabaliev_gs/.cache/huggingface}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel}"
LOG_DIR="${LOG_DIR:-${HOST_REPO}/logs}"
STEPS="${STEPS:-10}"
MODEL_ID="${MODEL_ID:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"
OUT_JSON="${OUT_JSON:-${HOST_REPO}/results/baselines/vla_2b_memory_probe_exclusive.json}"
LOG="${LOG:-${LOG_DIR}/vla_2b_memory_probe_$(date +%Y%m%d_%H%M%S).log}"
VRAM_LOG="${VRAM_LOG:-${LOG_DIR}/vla_2b_probe_vram.log}"

mkdir -p "${LOG_DIR}" "$(dirname "${OUT_JSON}")" "${HOST_PYDEPS}" "${HOST_HF}"

echo "=== aicenter 2.2B exclusive memory probe $(date -Is) ===" | tee -a "${LOG}"
echo "model=${MODEL_ID} steps=${STEPS}" | tee -a "${LOG}"
echo "out_json=${OUT_JSON}" | tee -a "${LOG}"
echo "log=${LOG} vram_log=${VRAM_LOG}" | tee -a "${LOG}"

# Host-side nvidia-smi sampler (container may not see truthful OS VRAM).
: > "${VRAM_LOG}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -l 1 > "${VRAM_LOG}" &
SMI_PID=$!
cleanup_smi() {
  if kill -0 "${SMI_PID}" 2>/dev/null; then
    kill "${SMI_PID}" 2>/dev/null || true
    wait "${SMI_PID}" 2>/dev/null || true
  fi
}
trap cleanup_smi EXIT

set +e
docker run --rm --gpus '"device=0,1"' \
  --shm-size=16g \
  --network host \
  -v "${HOST_REPO}:${HOST_REPO}" \
  -v "${HOST_PYDEPS}:${HOST_PYDEPS}" \
  -v "${HOST_HF}:/root/.cache/huggingface" \
  -e HOME=/home/askhabaliev_gs \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/hub \
  -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
  -e PYTHONUNBUFFERED=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -w "${HOST_REPO}" \
  "${IMAGE}" \
  bash -lc "
set -euo pipefail
export PYTHONPATH='${HOST_REPO}/third_party/actioncodec:/opt/conda/lib/python3.11/site-packages:${HOST_PYDEPS}'
export PATH='${HOST_PYDEPS}/bin:'\"\$PATH\"

python - <<'PY'
import importlib, torch
for m in ('lightning','transformers','accelerate','einops'):
    importlib.import_module(m)
    print('OK', m)
print('torch', torch.__version__, 'cuda', torch.cuda.device_count())
PY

# Host nvidia-smi is bind-visible; probe uses it for exclusive gate.
torchrun --standalone --nproc_per_node=2 \
  calvin_hicora/scripts/vla_memory_probe.py \
  --model-id '${MODEL_ID}' \
  --steps ${STEPS} \
  --micro-batch-size 1 \
  --require-exclusive-gpus \
  --activation-checkpointing \
  --nvidia-smi-every 1 \
  --output-json '${OUT_JSON}'
" 2>&1 | tee -a "${LOG}"
DOCKER_RC=${PIPESTATUS[0]}
set -e

cleanup_smi
trap - EXIT

echo "=== VRAM PEAKS from ${VRAM_LOG} ===" | tee -a "${LOG}"
python3 - <<PY | tee -a "${LOG}"
from pathlib import Path
path = Path("${VRAM_LOG}")
rows = {}
for line in path.read_text().splitlines():
    line = line.strip()
    if not line or line.lower().startswith("index"):
        continue
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        continue
    try:
        idx = int(parts[0])
        mem = float(parts[1].replace("MiB", "").strip())
    except ValueError:
        continue
    rows.setdefault(idx, []).append(mem)
for idx in sorted(rows):
    vals = rows[idx]
    print(f"GPU{idx}: samples={len(vals)} peak_used_MiB={max(vals):.1f} min_used_MiB={min(vals):.1f} delta={max(vals)-min(vals):.1f}")
if not rows:
    raise SystemExit("ERROR: no nvidia-smi samples")
PY

echo "=== DONE $(date -Is) docker_rc=${DOCKER_RC} ===" | tee -a "${LOG}"
exit "${DOCKER_RC}"
