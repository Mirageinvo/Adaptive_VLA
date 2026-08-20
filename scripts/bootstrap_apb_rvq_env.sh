#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a fresh Linux machine for the APB-RVQ oracle stage.
#
# Usage:
#   bash scripts/bootstrap_apb_rvq_env.sh /path/to/install/root
#
# This script:
#   1. clones or updates Adaptive_VLA
#   2. creates a conda env named apb-rvq
#   3. installs pinned dependencies from requirements.txt
#   4. creates a Hugging Face cache directory

INSTALL_ROOT="${1:-$HOME/work}"
REPO_URL="https://github.com/Mirageinvo/Adaptive_VLA.git"
REPO_DIR="${INSTALL_ROOT}/Adaptive_VLA"
ENV_NAME="${APB_RVQ_ENV_NAME:-apb-rvq}"
PYTHON_VERSION="${APB_RVQ_PYTHON:-3.10}"
CUDA_CHANNEL="${APB_RVQ_CUDA_CHANNEL:-cu121}"
HF_CACHE_DIR="${HF_HOME:-$HOME/huggingface_cache}"

echo "[bootstrap] install root: ${INSTALL_ROOT}"
echo "[bootstrap] repo dir:     ${REPO_DIR}"
echo "[bootstrap] env name:     ${ENV_NAME}"
echo "[bootstrap] python:       ${PYTHON_VERSION}"
echo "[bootstrap] hf cache:     ${HF_CACHE_DIR}"

if ! command -v git >/dev/null 2>&1; then
  echo "[bootstrap] git is required" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[bootstrap] conda is required but was not found in PATH" >&2
  echo "[bootstrap] install Miniconda/Mambaforge first, then rerun" >&2
  exit 1
fi

mkdir -p "${INSTALL_ROOT}"
mkdir -p "${HF_CACHE_DIR}"

if [ -d "${REPO_DIR}/.git" ]; then
  echo "[bootstrap] updating existing repo"
  git -C "${REPO_DIR}" fetch --all --tags
else
  echo "[bootstrap] cloning repo"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
  echo "[bootstrap] creating conda env ${ENV_NAME}"
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1" \
  --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}"
python -m pip install -r requirements.txt

cat <<EOF
[bootstrap] done
[bootstrap] next steps:
  export HF_HOME="${HF_CACHE_DIR}"
  cd "${REPO_DIR}"
  conda activate "${ENV_NAME}"
  python3 experiments/rate0_preflight.py --output artifacts/apb_rvq/preflight.json --min-free-gb 50
EOF
