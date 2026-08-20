#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/huggingface_cache}"
mkdir -p "$HOME/work" "$HF_HOME" "$HOME/tmp_install"

LOG="${HOME}/apb_rvq_setup.log"
exec > >(tee -a "$LOG") 2>&1

printf '[start] %s\n' "$(date)"

cd "$HOME/tmp_install"
if [ ! -x "$HOME/miniconda3/bin/conda" ]; then
  echo '[step] install_miniconda'
  curl -L -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash miniconda.sh -b -p "$HOME/miniconda3"
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
export PATH="$HOME/miniconda3/bin:$PATH"

if [ ! -d "$HOME/work/Adaptive_VLA/.git" ]; then
  echo '[step] clone_repo'
  git clone https://github.com/Mirageinvo/Adaptive_VLA.git "$HOME/work/Adaptive_VLA"
fi

cd "$HOME/work/Adaptive_VLA"
git fetch --all --tags || true
git checkout feature/apb-rvq-vla || true

echo '[step] bootstrap_env'
bash scripts/bootstrap_apb_rvq_env.sh "$HOME/work"

echo '[step] run_medium'
bash scripts/run_apb_rvq_oracle.sh medium

printf '[done] %s\n' "$(date)"
