#!/usr/bin/env bash
# Launch BAR --debug-overfit on aicenter (2x A100, FSDP + fp16).
set -euo pipefail

HOST_REPO="${HOST_REPO:-/home/askhabaliev_gs/Adaptive_VLA-1}"
HOST_DATA="${HOST_DATA:-/datasets/askhabaliev_gs}"
HOST_PYDEPS="${HOST_PYDEPS:-/home/askhabaliev_gs/pydeps}"
HOST_HF="${HOST_HF:-/home/askhabaliev_gs/.cache/huggingface}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel}"
CONFIG="${CONFIG:-config/train/bar_calvin_debug_overfit.yaml}"
LOG_DIR="${LOG_DIR:-/home/askhabaliev_gs/Adaptive_VLA-1/logs}"
LOG="${LOG:-${LOG_DIR}/vla_debug_overfit_$(date +%Y%m%d_%H%M%S).log}"
VRAM_LOG="${VRAM_LOG:-${LOG_DIR}/vla_vram_profile.log}"
# Short hardware-gate profile: DEBUG_OVERFIT_STEPS=10 DEBUG_OVERFIT_SKIP_PLATEAU=1
DEBUG_OVERFIT_STEPS="${DEBUG_OVERFIT_STEPS:-80}"
DEBUG_OVERFIT_SKIP_PLATEAU="${DEBUG_OVERFIT_SKIP_PLATEAU:-}"

mkdir -p "${LOG_DIR}" "${HOST_PYDEPS}" "${HOST_HF}"

echo "=== aicenter debug-overfit $(date -Is) ===" | tee -a "${LOG}"
echo "log=${LOG}" | tee -a "${LOG}"
echo "vram_log=${VRAM_LOG}" | tee -a "${LOG}"
echo "DEBUG_OVERFIT_STEPS=${DEBUG_OVERFIT_STEPS}" | tee -a "${LOG}"

# OS-level VRAM sampler on the HOST (sees real A100 nvidia-smi).
: > "${VRAM_LOG}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -l 1 > "${VRAM_LOG}" &
SMI_PID=$!
echo "nvidia-smi_pid=${SMI_PID}" | tee -a "${LOG}"

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
  -v "${HOST_REPO}:${HOST_REPO}" \
  -v "${HOST_DATA}:/datasets/askhabaliev_gs" \
  -v "${HOST_PYDEPS}:${HOST_PYDEPS}" \
  -v "${HOST_HF}:/root/.cache/huggingface" \
  -e HOME=/home/askhabaliev_gs \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/hub \
  -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
  -e PYTHONUNBUFFERED=1 \
  -e CALVIN_DATA_ROOT=/datasets/askhabaliev_gs/calvin_full/task_D_D \
  -e DEBUG_OVERFIT_STEPS="${DEBUG_OVERFIT_STEPS}" \
  -e DEBUG_OVERFIT_SKIP_PLATEAU="${DEBUG_OVERFIT_SKIP_PLATEAU}" \
  -w "${HOST_REPO}/third_party/actioncodec" \
  "${IMAGE}" \
  bash -lc "
set -euo pipefail
# Container torch/torchvision MUST win over anything left in pydeps.
export PYTHONPATH='${HOST_REPO}/third_party/actioncodec:/opt/conda/lib/python3.11/site-packages:${HOST_PYDEPS}'
export PATH='${HOST_PYDEPS}/bin:'\"\$PATH\"
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import importlib
need = [
    'lightning', 'omegaconf', 'peft', 'accelerate', 'PIL', 'einops',
    'transformers', 'safetensors', 'huggingface_hub',
]
missing = []
for m in need:
    try:
        importlib.import_module(m if m != 'PIL' else 'PIL')
        print('OK', m)
    except Exception as e:
        print('MISS', m, type(e).__name__, e)
        missing.append(m)
if missing:
    raise SystemExit('missing: ' + ','.join(missing))
import torch
print('torch', torch.__version__, torch.__file__)
print('cuda', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY

python scripts/train_vla.py --config '${CONFIG}' --debug-overfit
" 2>&1 | tee -a "${LOG}"
DOCKER_RC=${PIPESTATUS[0]}
set -e

cleanup_smi
trap - EXIT

echo "=== VRAM PEAKS from ${VRAM_LOG} ===" | tee -a "${LOG}"
python3 - <<PY | tee -a "${LOG}"
from pathlib import Path
path = Path("${VRAM_LOG}")
rows = []
for line in path.read_text().splitlines():
    line = line.strip()
    if not line or line.lower().startswith("index"):
        continue
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        continue
    try:
        idx = int(parts[0])
        # memory.used may be "12345 MiB"
        mem_tok = parts[1].replace("MiB", "").strip()
        mem = float(mem_tok)
    except ValueError:
        continue
    rows.append((idx, mem))
by_gpu = {}
for idx, mem in rows:
    by_gpu.setdefault(idx, []).append(mem)
for idx in sorted(by_gpu):
    vals = by_gpu[idx]
    print(f"GPU{idx}: samples={len(vals)} peak_used_MiB={max(vals):.1f} min_used_MiB={min(vals):.1f}")
if not by_gpu:
    print("ERROR: no nvidia-smi samples parsed from", path)
    raise SystemExit(2)
PY

echo "=== DONE $(date -Is) docker_rc=${DOCKER_RC} ===" | tee -a "${LOG}"
exit "${DOCKER_RC}"
