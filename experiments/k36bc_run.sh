#!/bin/bash
# K-36, вторая часть: этапы B и C (файлы этапа A и k36_run.sh не меняются).
#   bash experiments/k36bc_run.sh selftest
#   bash experiments/k36bc_run.sh gate-a             # точное решение A: 0 passed, 4 stopped, 3 сбой
#   bash experiments/k36bc_run.sh continue-b DEV     # только при passed_A_development
#   bash experiments/k36bc_run.sh screen-b DEV       # 180 эпизодов Tplus -> гейт B
#   bash experiments/k36bc_run.sh bar-preflight DEV  # legacy 4 -> bank-smoke 6 -> cold-repeat 3
#   bash experiments/k36bc_run.sh confirm-c DEV      # 616 блоков (только при passed_B_candidate)
#   bash experiments/k36bc_run.sh latency DEV        # свободная карта, иначе latency_pending
#   bash experiments/k36bc_run.sh analyze
#   bash experiments/k36bc_run.sh run DEV            # B -> (passed) C -> задержка -> анализ
# Проверенный отрицательный статус A/B — код 0 в run (4 у gate-a / gate-b);
# отсутствующие/повреждённые данные и технические ошибки — 3.
# Общая блокировка с k36_run.sh (logs/k36/.k36.lock).
set -uo pipefail
MODE="${1:?нужен режим}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs/k36 reports/k36 data/k36
exec 9>logs/k36/.k36.lock
flock -n 9 || { echo "ОТКАЗ: другой раннер K-36 уже выполняется"; exit 2; }
LOGF="logs/k36/bc_${MODE}_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $*, коммит $(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
PY=python3

need_mb () {
  $PY -c "import sys; sys.path.insert(0, 'experiments'); import k36_harness as h; print(int(h.need_free_gb() * 1024))"
}
gpu_free_ge () {   # DEV нужно_МиБ
  local idx="${1#cuda:}" free
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  [[ "$free" =~ ^[0-9]+$ ]] || { echo "ОТКАЗ: nvidia-smi для $1 вернул '${free}' (не число)"; return 3; }
  echo "    $1: свободно $free МиБ (нужно $2)"
  [ "$free" -ge "$2" ] || { echo "ОТКАЗ: на $1 свободно $free МиБ < $2"; return 3; }
}
gpu_used_le () {   # DEV порог: 0 — да, 4 — занята, 3 — сбой
  local idx="${1#cuda:}" used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  [[ "$used" =~ ^[0-9]+$ ]] || { echo "ОТКАЗ: nvidia-smi для $1 вернул '${used}' (не число)"; return 3; }
  echo "    $1: занято $used МиБ (порог $2)"
  [ "$used" -le "$2" ] || return 4
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
gpu () {   # DEV
  local need; need="$(need_mb)" || { echo "ОТКАЗ: порог памяти не вычислен"; return 3; }
  gpu_free_ge "$1" "$need" || return 3
  disk_wait || return 3
}
selftests () {
  local m
  for m in k36b_gate k36b_train k36b_screen k36c_bar k36c_confirm k36c_analyze k36c_latency; do
    $PY "experiments/${m}.py" --selftest || { echo "ОТКАЗ: самопроверка $m"; return 1; }
  done
}
continue_b () { gpu "$1" || return 3; $PY experiments/k36b_train.py continue-b --device "$1"; }
screen_b () { gpu "$1" || return 3; $PY experiments/k36b_screen.py screen-b --device "$1"; }
bar_preflight () {
  local k
  for k in bar-legacy bar-smoke bar-repeat; do
    echo "=== $k"; gpu "$1" || return 3
    $PY experiments/k36c_bar.py "$k" --device "$1" || return 3
  done
}
confirm_c () { gpu "$1" || return 3; $PY experiments/k36c_confirm.py confirm-c --device "$1"; }
latency () {
  local rc=0; gpu_used_le "$1" 2048 || rc=$?
  [ "$rc" -eq 4 ] && { echo "    latency_pending: карта занята"; return 0; }
  [ "$rc" -eq 0 ] || return 3
  rc=0; $PY experiments/k36c_latency.py --device "$1" || rc=$?
  [ "$rc" -eq 4 ] && { echo "    latency_pending: карта занята перед загрузкой модели"; return 0; }
  [ "$rc" -eq 0 ] || { echo "ОТКАЗ: замер задержки (код $rc)"; return 3; }
}
analyze () { $PY experiments/k36c_analyze.py analyze || { echo "ОТКАЗ: анализ C"; return 3; }; }

case "$MODE" in
  selftest) selftests; finish $? ;;
  gate-a) $PY experiments/k36b_gate.py gate-a; finish $? ;;
  continue-b) continue_b "${2:-cuda:0}"; finish $? ;;
  screen-b) screen_b "${2:-cuda:0}"; finish $? ;;
  bar-preflight) bar_preflight "${2:-cuda:0}"; finish $? ;;
  confirm-c) confirm_c "${2:-cuda:0}"; finish $? ;;
  latency) latency "${2:-cuda:0}"; finish $? ;;
  analyze) analyze; finish $? ;;
  run)
    DEV="${2:-cuda:0}"
    echo "=== ЭТАП самопроверки"; selftests || finish 1
    rc=0; $PY experiments/k36b_gate.py gate-a || rc=$?
    [ "$rc" -eq 4 ] && { echo "=== K-36: проверенное решение A — stopped_A_no_signal; B не открывается"; finish 0; }
    [ "$rc" -eq 0 ] || { echo "ОТКАЗ: решение A отсутствует или не сходится (код $rc)"; finish 3; }
    echo "=== ЭТАП продолжение T"; continue_b "$DEV" || finish 3
    echo "=== ЭТАП screen B"; screen_b "$DEV" || finish 3
    rc=0; $PY experiments/k36b_screen.py gate-b || rc=$?
    [ "$rc" -eq 4 ] && { echo "=== K-36: проверенное решение B — stopped_B_no_gain; C не открывается"; finish 0; }
    [ "$rc" -eq 0 ] || { echo "ОТКАЗ: решение B отсутствует или не сходится (код $rc)"; finish 3; }
    echo "=== ЭТАП предварительные проверки BAR"; bar_preflight "$DEV" || finish 3
    echo "=== ЭТАП C"; confirm_c "$DEV" || finish 3
    echo "=== ЭТАП задержка"; [ -f reports/k36/latency.json ] || latency "$DEV" || finish 3
    echo "=== ЭТАП итоговый анализ"; analyze || finish 3
    echo "=== K-36 завершён $(date)"; finish 0 ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
