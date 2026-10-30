#!/bin/bash
# K-28: parents_goal_fb24 (дерево поз + CPU-контроллер со свежим состоянием
# поверх замороженного parents24), seeds 0 и 1.
#   bash experiments/k28_run.sh selftest [cpu]
#   bash experiments/k28_run.sh prepare            # будущие состояния + источники
#   bash experiments/k28_run.sh real-check DEV     # оба сида + решение micro
#   bash experiments/k28_run.sh fitcheck   DEV
#   bash experiments/k28_run.sh run        DEV     # вся цепочка
#   bash experiments/k28_run.sh analyze | latency DEV
# run: selftest -> prepare -> real-check -> fitcheck -> legacy-smoke обоих
# сидов (задачи 0 и 8 блока 30) и сверка с историческим parents24; при
# совпавших начальных состояниях, но иных действиях — свежий контроль
# parents24 на 30 блоках этого сида -> для seed 0 и 1: обучение 500 + 2000 ->
# план -> smoke -> 30 блоков; после seed 0 — промежуточный анализ; затем
# анализ двух сидов -> latency (свободная карта) -> итоговый анализ. Любой
# технический отказ — останов; повторный run продолжает с проверенного
# места. SR и p-value на код не влияют. flock; проверка карты и места;
# MuJoCo и обучение не одновременно.
#   K28_MAX_USED_MB=N — допустимая занятость карты чужими процессами для
#   обучения и SR (по умолчанию 2048 МиБ); latency — всегда ≤ 2048 МиБ.
set -uo pipefail
MODE="${1:?нужен режим}"
DEV="${2:-cuda:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
ENVSEED=101
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p logs/k28 reports/k28 reports/k28/run
if [ -z "${K28_LOCKED:-}" ]; then
  exec 9>logs/k28/.k28.lock
  flock -n 9 || { echo "ОТКАЗ: другой k28_run.sh уже выполняется"; exit 2; }
  export K28_LOCKED=1
fi
if [ "$MODE" = run ]; then LOGF="logs/k28/run.log"; else LOGF="logs/k28/${MODE}_$(date +%Y%m%dT%H%M%S).log"; fi
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит $(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
PY=python3
# shellcheck source=k28_rollout.sh
source "$ROOT/experiments/k28_rollout.sh"

gpu_free () {   # [strict]: для latency порог всегда 2048 МиБ
  local idx="${DEV#cuda:}" used lim="${K28_MAX_USED_MB:-2048}"
  [ "${1:-}" = strict ] && lim=2048
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  [ -n "$used" ] || { echo "ОТКАЗ: nvidia-smi не отвечает для $DEV"; return 3; }
  echo "    $DEV: занято $used МиБ (допустимо до $lim)"
  [ "$used" -le "$lim" ] || { echo "ОТКАЗ: на $DEV занято $used МиБ"; return 3; }
}
disk_ok () {
  local free
  free=$(df -Pm data | awk 'NR==2 {print $4}')
  echo "    место: свободно ${free} МиБ (нужно ~500)"
  [ "$free" -ge 500 ] || { echo "ОТКАЗ: мало места"; return 3; }
}
selftests () {
  local m
  for m in k28_geometry k28_control k28_model k28_policy k28_harness \
           k28_rollout_core k28_data k28_failures k28_analyze k28_latency \
           k9h_multiarm_gate; do
    $PY "experiments/${m}.py" --selftest >/dev/null 2>&1 \
      || { echo "ОТКАЗ: самопроверка ${m}"; return 1; }
  done
  $PY experiments/k28_protocol.py selftest >/dev/null 2>&1 || { echo "ОТКАЗ: самопроверка k28_protocol"; return 1; }
  $PY experiments/k28_train.py --selftest >/dev/null 2>&1 || { echo "ОТКАЗ: самопроверка k28_train"; return 1; }
  echo "    самопроверки K-28 пройдены"
}
prepare () {
  if [ ! -f reports/k28/prepare.json ]; then
    $PY experiments/k28_data.py || return 3
  else
    echo "    будущие состояния уже подготовлены: reports/k28/prepare.json"
  fi
  $PY experiments/k28_protocol.py prepare || return 3
}
micro () { $PY experiments/k28_train.py micro; }
real_check () {
  if $PY experiments/k28_train.py check-fresh --kind real-check --device "$DEV" \
       --micros 8 32; then return 0; fi
  gpu_free || return 3
  $PY experiments/k28_train.py real-check --device "$DEV" --micros 8 32
}
fitcheck () {
  local M; M="$(micro)" || { echo "ОТКАЗ: нет решения о microbatch"; return 3; }
  if $PY experiments/k28_train.py check-fresh --kind fitcheck --device "$DEV" \
       --micro "$M"; then return 0; fi
  gpu_free || return 3
  $PY experiments/k28_train.py fitcheck --device "$DEV" --micro "$M"
}
legacy_one () {   # seed: legacy-smoke, сверка, при необходимости свежий контроль
  local SEED="$1" rc=0
  $PY experiments/k28_protocol.py plan --seed "$SEED" --kind legacy --device "$DEV" || return 3
  run_plan "reports/k28/plan_s${SEED}_legacy.json" smoke || return $?
  $PY experiments/k28_protocol.py legacycheck --plan "reports/k28/plan_s${SEED}_legacy.json" || rc=$?
  if [ $rc -eq 4 ]; then
    echo "=== ЭТАП свежий контроль parents24 seed $SEED (переиспользование не подтверждено)"
    $PY experiments/k28_protocol.py plan --seed "$SEED" --kind control --device "$DEV" || return 3
    run_plan "reports/k28/plan_s${SEED}_control.json" all || return $?
    return 0
  fi
  return $rc
}
train_one () {   # seed
  local SEED="$1" M
  M="$(micro)" || { echo "ОТКАЗ: нет решения о microbatch"; return 3; }
  gpu_free || return 3; disk_ok || return 3
  $PY experiments/k28_train.py train --seed "$SEED" --device "$DEV" --micro "$M" --dtype float16 || return $?
}
sr_one () {   # seed
  local SEED="$1"
  $PY experiments/k28_protocol.py plan --seed "$SEED" --kind main --device "$DEV" || return 3
  run_plan "reports/k28/plan_s${SEED}.json" all || return $?
}

case "$MODE" in
  selftest) selftests; finish $? ;;
  prepare) prepare; finish $? ;;
  real-check) selftests || finish 1; prepare || finish 3; real_check; finish $? ;;
  fitcheck) selftests || finish 1; prepare || finish 3; fitcheck; finish $? ;;
  analyze) $PY experiments/k28_analyze.py; finish $? ;;
  latency) gpu_free strict || finish 3; $PY experiments/k28_latency.py --device "$DEV"; finish $? ;;
  run)
    echo "=== ЭТАП самопроверки"; selftests || finish 1
    echo "=== ЭТАП подготовка"; prepare || finish 3
    echo "=== ЭТАП real-check и решение microbatch"; real_check || { echo "ОСТАНОВ: real-check"; finish 3; }
    echo "=== ЭТАП fitcheck"; fitcheck || { echo "ОСТАНОВ: fitcheck не пройден"; finish 3; }
    for SEED in 0 1; do
      echo "=== ЭТАП legacy-smoke seed $SEED (сверка с историческим parents24)"; legacy_one "$SEED" || finish $?
    done
    for SEED in 0 1; do
      if [ -f "reports/k28/plan_s${SEED}.json" ]; then
        echo "=== ЭТАП обучение seed $SEED: уже обучено и закреплено в плане — пропуск"
      else
        echo "=== ЭТАП обучение seed $SEED"; train_one "$SEED" || finish $?
      fi
      echo "=== ЭТАП SR seed $SEED"; sr_one "$SEED" || finish $?
      if [ "$SEED" = 0 ]; then
        echo "=== ЭТАП анализ seed 0 (промежуточно)"; $PY experiments/k28_analyze.py --seeds 0 || finish 3
      fi
    done
    echo "=== ЭТАП анализ двух сидов"; $PY experiments/k28_analyze.py || finish 3
    echo "=== ЭТАП latency"
    if gpu_free strict; then
      $PY experiments/k28_latency.py --device "$DEV" || finish 3
      echo "=== ЭТАП итоговый анализ (с T_cycle)"; $PY experiments/k28_analyze.py || finish 3
    else
      echo "    свободного окна нет — анализ SR готов, латентность ожидается: bash experiments/k28_run.sh latency $DEV"
    fi
    echo "=== ЦЕПОЧКА K-28 ЗАВЕРШЕНА $(date)"; finish 0 ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
