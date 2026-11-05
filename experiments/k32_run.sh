#!/bin/bash
# K-32: обучаемая коррекция поддерева (T), контроли P (профиль) и F (чанк),
# свежий reference R; seeds 0 и 1; LIBERO-10.
#   bash experiments/k32_run.sh selftest [cpu]
#   bash experiments/k32_run.sh prepare
#   bash experiments/k32_run.sh normalizer DEV
#   bash experiments/k32_run.sh real-check DEV
#   bash experiments/k32_run.sh collector-smoke DEV
#   bash experiments/k32_run.sh init-control DEV   # SR R обоих сидов
#   bash experiments/k32_run.sh train DEV          # T/P/F × оба сида
#   bash experiments/k32_run.sh final-eval DEV     # SR T/P/F обоих сидов
#   bash experiments/k32_run.sh analyze | latency DEV
#   bash experiments/k32_run.sh run DEV            # вся разрешённая цепочка
# run: ждёт конца K-31 -> самопроверки -> prepare -> normalizer -> real-check ->
# collector-smoke -> init-control -> обучение T_s0 P_s0 F_s0 T_s1 P_s1 F_s1 ->
# final-eval -> анализ -> latency (почти свободная карта) -> итоговый анализ.
# Готовые проверенные этапы переиспользуются; технический отказ — код 3.
# Запланированная большая проверка LIBERO имеет приоритет: run не запускать,
# пока она стоит в очереди на эту карту. Перед GPU-этапами ждём свободной
# памяти (K32_NEED_MB, 8000) и места (K32_NEED_DISK_MB, 5000) до 12 ч.
set -uo pipefail
MODE="${1:?нужен режим}"
DEV="${2:-cuda:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p logs/k32 reports/k32 data/k32
if [ -z "${K32_LOCKED:-}" ]; then
  exec 9>logs/k32/.k32.lock
  flock -n 9 || { echo "ОТКАЗ: другой k32_run.sh уже выполняется"; exit 2; }
  export K32_LOCKED=1
fi
if [ "$MODE" = run ]; then LOGF="logs/k32/run.log"; else LOGF="logs/k32/${MODE}_$(date +%Y%m%dT%H%M%S).log"; fi
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит $(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
PY=python3
PROT="$PY experiments/k32_protocol.py"

k31_busy () { pgrep -f "k31_run.sh|k31_protocol.py|k31_harness.py|k31_latency.py" >/dev/null; }
wait_k31 () { while k31_busy; do echo "    K-31 ещё выполняется — жду 10 мин ($(date '+%H:%M'))"; sleep 600; done; }
wait_until () {
  local what="$1" waited=0 cap="${K32_WAIT_MIN:-720}"; shift
  while :; do
    if "$@"; then return 0; fi
    [ "$waited" -ge "$cap" ] && { echo "ОТКАЗ: $what — не дождались за $waited мин"; return 3; }
    echo "    $what: жду 10 мин (прошло $waited из $cap, $(date '+%H:%M'))"
    sleep 600; waited=$((waited + 10))
  done
}
mem_ok () {
  local idx="${DEV#cuda:}" free need="${K32_NEED_MB:-8000}"
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  [ -n "$free" ] || { echo "    nvidia-smi не отвечает для $DEV"; return 1; }
  echo "    $DEV: свободно $free МиБ (нужно ≥ $need)"; [ "$free" -ge "$need" ]
}
disk_ok () {
  local free need="${K32_NEED_DISK_MB:-5000}"
  free=$(df -Pm data | awk 'NR==2 {print $4}')
  echo "    место: свободно ${free} МиБ (нужно ≥ $need)"; [ "$free" -ge "$need" ]
}
gpu_strict () {
  local idx="${DEV#cuda:}" used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  [ -n "$used" ] && [ "$used" -le 2048 ]
}
gpu_stage () { wait_until "память $DEV" mem_ok || return 3; wait_until "место на диске" disk_ok || return 3; }
selftests () {
  local m
  for m in k32_dist k32_operator k32_model k32_policy k32_collect k32_ppo \
           k32_harness k32_analyze k32_latency k31_readout k31_collect \
           k29_model k29_dist; do
    $PY "experiments/${m}.py" --selftest >/dev/null 2>&1 || { echo "ОТКАЗ: самопроверка ${m}"; return 1; }
  done
  $PROT --selftest >/dev/null 2>&1 || { echo "ОТКАЗ: самопроверка k32_protocol"; return 1; }
  echo "    самопроверки K-32 пройдены"
}
fresh () { $PROT check-fresh --kind "$1" >/dev/null 2>&1; }
norm_ready () { [ -f reports/k32/normalizer_manifest.json ] && $PY -c "import sys; sys.path.insert(0,'experiments'); import k32_protocol as p; p.contract()" >/dev/null 2>&1; }
normalizer () { norm_ready && { echo "    нормировка: готова"; return 0; }; gpu_stage || return 3; $PROT normalizer --device "$DEV"; }
real_check () { fresh real_check && { echo "    real-check: свежий"; return 0; }; gpu_stage || return 3; $PROT real-check --device "$DEV"; }
smoke () { fresh smoke && { echo "    collector-smoke: свежий"; return 0; }; gpu_stage || return 3; $PROT smoke --device "$DEV"; }
sr () {   # KIND SEED
  gpu_stage || return 3
  $PY experiments/k32_harness.py eval --seed "$2" --kind "$1" --device "$DEV"
}
init_control () {
  $PROT plan-eval --which ref || return 3
  local S; for S in 0 1; do echo "=== ЭТАП SR R seed $S"; sr R "$S" || return 3; done
}
train () { gpu_stage || return 3; $PROT train --device "$DEV"; }
final_eval () {
  $PROT plan-eval --which final || return 3
  local SK S K
  for SK in "0 T" "0 P" "0 F" "1 T" "1 P" "1 F"; do
    read -r S K <<< "$SK"; echo "=== ЭТАП SR $K seed $S"; sr "$K" "$S" || return 3
  done
}

case "$MODE" in
  selftest) selftests; finish $? ;;
  prepare) $PROT prepare; finish $? ;;
  normalizer) wait_k31; $PROT prepare || finish 3; normalizer; finish $? ;;
  real-check) wait_k31; selftests || finish 1; $PROT prepare || finish 3; real_check; finish $? ;;
  collector-smoke) wait_k31; smoke; finish $? ;;
  init-control) wait_k31; init_control; finish $? ;;
  train) wait_k31; train; finish $? ;;
  final-eval) wait_k31; final_eval; finish $? ;;
  analyze) $PY experiments/k32_analyze.py; finish $? ;;
  latency) gpu_strict || { echo "ОТКАЗ: на $DEV занято > 2048 МиБ"; finish 3; }; $PY experiments/k32_latency.py --device "$DEV"; finish $? ;;
  run)
    echo "=== ЭТАП ожидание конца K-31"; wait_k31
    echo "=== ЭТАП самопроверки"; selftests || finish 1
    echo "=== ЭТАП подготовка"; $PROT prepare || finish 3
    echo "=== ЭТАП нормировка"; normalizer || { echo "ОСТАНОВ: нормировка"; finish 3; }
    echo "=== ЭТАП real-check"; real_check || { echo "ОСТАНОВ: real-check"; finish 3; }
    echo "=== ЭТАП collector-smoke"; smoke || { echo "ОСТАНОВ: smoke"; finish 3; }
    echo "=== ЭТАП init-control"; init_control || { echo "ОСТАНОВ: init-control"; finish 3; }
    echo "=== ЭТАП обучение"; train || { echo "ОСТАНОВ: обучение"; finish 3; }
    echo "=== ЭТАП final-eval"; final_eval || { echo "ОСТАНОВ: final-eval"; finish 3; }
    echo "=== ЭТАП анализ"; $PY experiments/k32_analyze.py || finish 3
    echo "=== ЭТАП latency"
    if gpu_strict; then
      $PY experiments/k32_latency.py --device "$DEV" || finish 3
      echo "=== ЭТАП итоговый анализ"; $PY experiments/k32_analyze.py || finish 3
    else
      echo "    латентность ожидается: bash experiments/k32_run.sh latency $DEV"
    fi
    echo "=== ЦЕПОЧКА K-32 ЗАВЕРШЕНА $(date)"; finish 0 ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
