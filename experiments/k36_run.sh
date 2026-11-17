#!/bin/bash
# K-36: подтверждение и развитие SR через RL-коррекцию дерева (план v1.0).
# Первая часть реализации — до решения по этапу A:
#   bash experiments/k36_run.sh selftest [cpu]
#   bash experiments/k36_run.sh prepare            # дважды: идемпотентен
#   bash experiments/k36_run.sh cpu-audit
#   bash experiments/k36_run.sh real-check DEV     # сохранённые входы K-21, без раскаток
#   bash experiments/k36_run.sh collector-smoke DEV  # legacy-check -> bank-smoke -> cold-repeat
#   bash experiments/k36_run.sh confirm-a DEV      # 360 блоков / 1800 эпизодов (продолжение с места)
#   bash experiments/k36_run.sh analyze-a
#   bash experiments/k36_run.sh status
#   bash experiments/k36_run.sh run DEV            # всё выше по порядку
# continue-b / screen-b / confirm-c / latency / analyze — вторая часть
# реализации (пока технический отказ, код 3).
# Свободная память карты до загрузки: K36_FREE_MB (10240); диск K36_DISK_MB
# (12288), ждать до K36_WAIT_MIN = 720 мин, чужое не удаляется.
set -uo pipefail
MODE="${1:?нужен режим}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs/k36 reports/k36 data/k36
exec 9>logs/k36/.k36.lock
flock -n 9 || { echo "ОТКАЗ: другой k36_run.sh уже выполняется"; exit 2; }
LOGF="logs/k36/${MODE}_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $*, коммит $(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
PY=python3
PROT="$PY experiments/k36_protocol.py"
HAR="$PY experiments/k36_harness.py"
ANA="$PY experiments/k36_analyze.py"

gpu_free_ge () {   # DEV нужно_МиБ
  local idx="${1#cuda:}" free
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  [ -n "$free" ] || { echo "ОТКАЗ: nvidia-smi не отвечает для $1"; return 3; }
  echo "    $1: свободно $free МиБ (нужно $2)"
  [ "$free" -ge "$2" ] || { echo "ОТКАЗ: на $1 свободно $free МиБ < $2"; return 3; }
}
disk_wait () {
  local need="${K36_DISK_MB:-12288}" waited=0 free
  while :; do
    free=$(df -Pm reports | awk 'NR==2 {print $4}')
    [ "$free" -ge "$need" ] && return 0
    [ "$waited" -ge "${K36_WAIT_MIN:-720}" ] && { echo "ОТКАЗ: места $free МиБ < $need"; return 3; }
    echo "    места $free МиБ < $need — жду 10 мин ($(date '+%H:%M'))"; sleep 600; waited=$((waited + 10))
  done
}
selftests () {
  local m
  for m in k36_bank k36_inputs k36_protocol k36_harness k36_analyze; do
    $PY "experiments/${m}.py" --selftest || { echo "ОТКАЗ: самопроверка $m"; return 1; }
  done
}
gpu_step () {   # команда_харнесса DEV
  gpu_free_ge "$2" "${K36_FREE_MB:-10240}" || return 3
  disk_wait || return 3
  $HAR "$1" --device "$2"
}
smoke () {   # DEV
  echo "=== legacy-check (24 эпизода)"; gpu_step legacy-check "$1" || return 3
  echo "=== bank-smoke (36 эпизодов)"; gpu_step bank-smoke "$1" || return 3
  echo "=== cold-repeat (12 эпизодов, отдельный холодный процесс)"; gpu_step cold-repeat "$1" || return 3
}
later () { echo "ОТКАЗ: $1 — вторая часть реализации K-36 (ещё не написана)"; return 3; }

case "$MODE" in
  selftest) selftests; finish $? ;;
  prepare) $PROT prepare; finish $? ;;
  cpu-audit) $ANA cpu-audit; finish $? ;;
  real-check) gpu_step real-check "${2:-cuda:0}"; finish $? ;;
  collector-smoke) smoke "${2:-cuda:0}"; finish $? ;;
  confirm-a) gpu_step confirm-a "${2:-cuda:0}"; finish $? ;;
  analyze-a) $ANA analyze-a; finish $? ;;
  status) $HAR status; finish $? ;;
  continue-b|screen-b|confirm-c|latency|analyze) later "$MODE"; finish $? ;;
  run)
    DEV="${2:-cuda:0}"
    echo "=== ЭТАП самопроверки"; selftests || finish 1
    echo "=== ЭТАП prepare"; $PROT prepare || finish 3
    echo "=== ЭТАП cpu-audit"; $ANA cpu-audit || finish 3
    echo "=== ЭТАП real-check"; gpu_step real-check "$DEV" || { echo "ОСТАНОВ: real-check"; finish 3; }
    echo "=== ЭТАП collector-smoke"; smoke "$DEV" || { echo "ОСТАНОВ: smoke"; finish 3; }
    echo "=== ЭТАП confirm-a"; gpu_step confirm-a "$DEV" || { echo "ОСТАНОВ: этап A"; finish 3; }
    echo "=== ЭТАП анализ A"; $ANA analyze-a || finish 3
    echo "=== K-36 этап A завершён $(date); решение — reports/k36/gate_a.json"
    finish 0 ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
