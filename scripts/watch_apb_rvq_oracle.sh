#!/usr/bin/env bash
set -euo pipefail

# Watchdog wrapper: restart the oracle pipeline if the tmux session dies or the run fails.
#
# Usage:
#   bash scripts/watch_apb_rvq_oracle.sh full
#   APB_RVQ_WATCHDOG_RESTARTS=5 bash scripts/watch_apb_rvq_oracle.sh medium

MODE="${1:-full}"
SESSION="${APB_RVQ_TMUX_SESSION:-apb_rvq_${MODE}}"
LOG_FILE="${APB_RVQ_WATCHDOG_LOG:-${HOME}/apb_rvq_${MODE}_watchdog.log}"
MAX_RESTARTS="${APB_RVQ_WATCHDOG_RESTARTS:-5}"
RESTART_SLEEP_SEC="${APB_RVQ_WATCHDOG_SLEEP_SEC:-60}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${APB_RVQ_OUT_ROOT:-artifacts/apb_rvq}/oracle_${MODE}"

log() {
  echo "[watchdog $(date -Is)] $*" | tee -a "${LOG_FILE}"
}

is_complete() {
  [[ -f "${REPO_ROOT}/${OUT_DIR}/metrics.json" && -f "${REPO_ROOT}/${OUT_DIR}/validate_search.json" ]]
}

run_once() {
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  tmux new-session -d -s "${SESSION}" "
    source \"${HOME}/miniconda3/etc/profile.d/conda.sh\"
    conda activate apb-rvq
    export HF_HOME=\"${HF_HOME:-${HOME}/huggingface_cache}\"
    export PYTHONPATH=\"${REPO_ROOT}:\${PYTHONPATH:-}\"
    cd \"${REPO_ROOT}\"
    bash scripts/run_apb_rvq_oracle.sh \"${MODE}\" >> \"${HOME}/apb_rvq_${MODE}.log\" 2>&1
  "
}

log "starting watchdog for mode=${MODE}, session=${SESSION}, max_restarts=${MAX_RESTARTS}"

for ((restart=1; restart <= MAX_RESTARTS; restart++)); do
  if is_complete; then
    log "pipeline already complete: ${OUT_DIR}"
    exit 0
  fi

  log "launch attempt ${restart}/${MAX_RESTARTS}"
  run_once

  while tmux has-session -t "${SESSION}" 2>/dev/null; do
    if is_complete; then
      log "pipeline completed successfully"
      exit 0
    fi
    sleep 30
  done

  if is_complete; then
    log "pipeline completed successfully after session exit"
    exit 0
  fi

  log "session ${SESSION} exited before completion"
  if (( restart < MAX_RESTARTS )); then
    log "sleeping ${RESTART_SLEEP_SEC}s before restart"
    sleep "${RESTART_SLEEP_SEC}"
  fi
done

log "watchdog exhausted ${MAX_RESTARTS} restarts"
exit 1
