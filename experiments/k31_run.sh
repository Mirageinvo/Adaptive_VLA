#!/bin/bash
# K-31: выбор поддерева для коррекции по исходам развилок (parents_cond24 K-25,
# seeds 0 и 1, LIBERO-10).
#   bash experiments/k31_run.sh selftest [cpu]
#   bash experiments/k31_run.sh prepare
#   bash experiments/k31_run.sh real-check DEV
#   bash experiments/k31_run.sh fork-smoke DEV
#   bash experiments/k31_run.sh pilot DEV        # A обоих сидов + гейт
#   bash experiments/k31_run.sh collect DEV      # B, только после допуска A
#   bash experiments/k31_run.sh fitcheck | train | plan-eval   # CPU
#   bash experiments/k31_run.sh eval DEV         # SR: ref s0/s1, repair/flat s0, s1
#   bash experiments/k31_run.sh analyze | latency DEV
#   bash experiments/k31_run.sh run DEV          # вся разрешённая цепочка
# run: ждёт конца K-30 -> самопроверки -> prepare -> real-check -> fork-smoke ->
# пилот A (гейт; информационный стоп — код 0) -> добор B -> набор, fitcheck,
# обучение обоих селекторов (неактивность в обоих сидах — информационный стоп)
# -> план SR -> SR (свежий reference обоих сидов, затем repair/flatrepair) ->
# анализ -> latency (почти свободная карта) -> итоговый анализ. Готовые
# проверенные этапы переиспользуются. Технический отказ — код 3; повторный
# run продолжает с проверенного места. Перед GPU-этапами ждём свободной памяти
# (K31_NEED_MB, 8000) и места (K31_NEED_DISK_MB, 3000) по 10 мин до 12 ч.
set -uo pipefail
MODE="${1:?нужен режим}"
DEV="${2:-cuda:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p logs/k31 reports/k31 data/k31
if [ -z "${K31_LOCKED:-}" ]; then
  exec 9>logs/k31/.k31.lock
  flock -n 9 || { echo "ОТКАЗ: другой k31_run.sh уже выполняется"; exit 2; }
  export K31_LOCKED=1
fi
if [ "$MODE" = run ]; then LOGF="logs/k31/run.log"; else LOGF="logs/k31/${MODE}_$(date +%Y%m%dT%H%M%S).log"; fi
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит $(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
PY=python3
PROT="$PY experiments/k31_protocol.py"

k30_busy () { pgrep -f "k30_run.sh|k30_train.py|k30_harness.py|k30_latency.py|k9h_multiarm_gate.py" >/dev/null; }
wait_k30 () { while k30_busy; do echo "    K-30 ещё выполняется — жду 10 мин ($(date '+%H:%M'))"; sleep 600; done; }
wait_until () {   # описание, команда-проверка
  local what="$1" waited=0 cap="${K31_WAIT_MIN:-720}"; shift
  while :; do
    if "$@"; then return 0; fi
    [ "$waited" -ge "$cap" ] && { echo "ОТКАЗ: $what — не дождались за $waited мин"; return 3; }
    echo "    $what: жду 10 мин (прошло $waited из $cap, $(date '+%H:%M'))"
    sleep 600; waited=$((waited + 10))
  done
}
mem_ok () {
  local idx="${DEV#cuda:}" free need="${K31_NEED_MB:-8000}"
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  [ -n "$free" ] || { echo "    nvidia-smi не отвечает для $DEV"; return 1; }
  echo "    $DEV: свободно $free МиБ (нужно ≥ $need)"
  [ "$free" -ge "$need" ]
}
disk_ok () {
  local free need="${K31_NEED_DISK_MB:-3000}"
  free=$(df -Pm data | awk 'NR==2 {print $4}')
  echo "    место: свободно ${free} МиБ (нужно ≥ $need)"
  [ "$free" -ge "$need" ]
}
gpu_strict () {
  local idx="${DEV#cuda:}" used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  [ -n "$used" ] && [ "$used" -le 2048 ]
}
gpu_stage () { wait_until "память $DEV" mem_ok || return 3; wait_until "место на диске" disk_ok || return 3; }
selftests () {
  local m
  for m in k31_readout k31_candidates k31_policy k31_collect k31_data k31_train \
           k31_harness k31_analyze k31_latency k29_model k29_policy k29_dist \
           k25_readout k18_collector; do
    $PY "experiments/${m}.py" --selftest >/dev/null 2>&1 \
      || { echo "ОТКАЗ: самопроверка ${m}"; return 1; }
  done
  $PROT --selftest >/dev/null 2>&1 || { echo "ОТКАЗ: самопроверка k31_protocol"; return 1; }
  echo "    самопроверки K-31 пройдены"
}
fresh () { $PROT check-fresh --kind "$1" >/dev/null 2>&1; }
real_check () { fresh real_check && { echo "    real-check: свежий"; return 0; }; gpu_stage || return 3; $PROT real-check --device "$DEV"; }
fork_smoke () { fresh smoke && { echo "    fork-smoke: свежий"; return 0; }; gpu_stage || return 3; $PROT fork-smoke --device "$DEV"; }
pilot () { gpu_stage || return 3; $PROT pilot --device "$DEV"; }
collect () { gpu_stage || return 3; $PROT collect --device "$DEV"; }
datasets () {
  if [ -f data/k31/dataset_s0.npz ] && [ -f data/k31/dataset_s1.npz ]; then echo "    наборы: уже построены"; return 0; fi
  $PROT dataset
}
fitcheck () { fresh fitcheck && { echo "    fitcheck: свежий"; return 0; }; $PROT fitcheck; }
train () { $PROT train; }
eval_all () {
  local SK S K
  for SK in "0 ref" "1 ref" "0 repair" "0 flatrepair" "1 repair" "1 flatrepair"; do
    read -r S K <<< "$SK"
    echo "=== ЭТАП SR $K seed $S"
    gpu_stage || return 3
    $PY experiments/k31_harness.py eval --seed "$S" --kind "$K" --device "$DEV" || return 3
  done
}

case "$MODE" in
  selftest) selftests; finish $? ;;
  prepare) $PROT prepare; finish $? ;;
  real-check) wait_k30; selftests || finish 1; $PROT prepare || finish 3; real_check; finish $? ;;
  fork-smoke) wait_k30; $PROT prepare || finish 3; fork_smoke; finish $? ;;
  pilot) wait_k30; pilot; finish $? ;;
  collect) wait_k30; collect; finish $? ;;
  fitcheck) datasets || finish 3; fitcheck; finish $? ;;
  train) train; finish $? ;;
  plan-eval) $PROT plan-eval; finish $? ;;
  eval) wait_k30; eval_all; finish $? ;;
  analyze) $PY experiments/k31_analyze.py; finish $? ;;
  latency) gpu_strict || { echo "ОТКАЗ: на $DEV занято > 2048 МиБ"; finish 3; }; $PY experiments/k31_latency.py --device "$DEV"; finish $? ;;
  run)
    echo "=== ЭТАП ожидание конца K-30"; wait_k30
    echo "=== ЭТАП самопроверки"; selftests || finish 1
    echo "=== ЭТАП подготовка"; $PROT prepare || finish 3
    echo "=== ЭТАП real-check"; real_check || { echo "ОСТАНОВ: real-check"; finish 3; }
    echo "=== ЭТАП fork-smoke"; fork_smoke || { echo "ОСТАНОВ: fork-smoke"; finish 3; }
    echo "=== ЭТАП пилот A и гейт"; pilot; RC=$?
    if [ $RC -eq 4 ]; then echo "=== ИНФОРМАЦИОННЫЙ СТОП после пилота (см. reports/k31/pilot.json)"; finish 0; fi
    [ $RC -eq 0 ] || { echo "ОСТАНОВ: пилот"; finish 3; }
    echo "=== ЭТАП добор B"; collect || { echo "ОСТАНОВ: добор"; finish 3; }
    echo "=== ЭТАП наборы и fitcheck"; datasets || finish 3; fitcheck || { echo "ОСТАНОВ: fitcheck"; finish 3; }
    echo "=== ЭТАП обучение селекторов"; train; RC=$?
    if [ $RC -eq 4 ]; then echo "=== ИНФОРМАЦИОННЫЙ СТОП: селектор неактивен в обоих сидах"; finish 0; fi
    [ $RC -eq 0 ] || { echo "ОСТАНОВ: обучение"; finish 3; }
    echo "=== ЭТАП план SR"; $PROT plan-eval || finish 3
    eval_all || { echo "ОСТАНОВ: SR"; finish 3; }
    echo "=== ЭТАП анализ"; $PY experiments/k31_analyze.py || finish 3
    echo "=== ЭТАП latency"
    if gpu_strict; then
      $PY experiments/k31_latency.py --device "$DEV" || finish 3
      echo "=== ЭТАП итоговый анализ"; $PY experiments/k31_analyze.py || finish 3
    else
      echo "    латентность ожидается: bash experiments/k31_run.sh latency $DEV"
    fi
    echo "=== ЦЕПОЧКА K-31 ЗАВЕРШЕНА $(date)"; finish 0 ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
