#!/bin/bash
# K-29: ограниченное online PPO условного декодера parents_cond24 (arm-only),
# training seeds 0 и 1, V100.
#   bash experiments/k29_run.sh selftest [cpu]
#   bash experiments/k29_run.sh prepare
#   bash experiments/k29_run.sh real-check      DEV
#   bash experiments/k29_run.sh collector-smoke DEV
#   bash experiments/k29_run.sh run             DEV   # вся цепочка
#   bash experiments/k29_run.sh analyze | latency DEV
# run: самопроверки -> prepare -> real-check и smoke (готовые того же
# контракта переиспользуются) -> init-control s0 и s1 (30…44, до RL) ->
# s0: обучение (кап 200 000 физических шагов) -> final-оценка -> s1: то же ->
# анализ двух сидов -> latency (свободная карта, иначе «ожидается").
# Технический отказ — останов с причиной; повторный run продолжает с
# проверенного места (готовые блоки, сохранённое обучение, журнал бюджета).
# SR и p на код не влияют. Сбор, PPO и оценки — последовательно на одной
# карте. K29_MAX_USED_MB — допустимая чужая занятость карты (по умолчанию
# 2048 МиБ); latency — всегда ≤ 2048.
set -uo pipefail
MODE="${1:?нужен режим}"
DEV="${2:-cuda:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p logs/k29 reports/k29 data/k29
if [ -z "${K29_LOCKED:-}" ]; then
  exec 9>logs/k29/.k29.lock
  flock -n 9 || { echo "ОТКАЗ: другой k29_run.sh уже выполняется"; exit 2; }
  export K29_LOCKED=1
fi
if [ "$MODE" = run ]; then LOGF="logs/k29/run.log"; else LOGF="logs/k29/${MODE}_$(date +%Y%m%dT%H%M%S).log"; fi
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит $(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
PY=python3

gpu_free () {   # [strict]
  local idx="${DEV#cuda:}" used lim="${K29_MAX_USED_MB:-2048}"
  [ "${1:-}" = strict ] && lim=2048
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  [ -n "$used" ] || { echo "ОТКАЗ: nvidia-smi не отвечает для $DEV"; return 3; }
  echo "    $DEV: занято $used МиБ (допустимо до $lim)"
  [ "$used" -le "$lim" ] || { echo "ОТКАЗ: на $DEV занято $used МиБ"; return 3; }
}
disk_ok () {
  local free; free=$(df -Pm data | awk 'NR==2 {print $4}')
  echo "    место: свободно ${free} МиБ (нужно ~3000)"
  [ "$free" -ge 3000 ] || { echo "ОТКАЗ: мало места"; return 3; }
}
selftests () {
  local m
  for m in k29_dist k29_model k29_policy k29_collect k29_ppo k29_harness \
           k29_latency; do
    $PY "experiments/${m}.py" --selftest >/dev/null 2>&1 \
      || { echo "ОТКАЗ: самопроверка ${m}"; return 1; }
  done
  $PY experiments/k29_analyze.py --selftest >/dev/null 2>&1 || { echo "ОТКАЗ: самопроверка k29_analyze"; return 1; }
  $PY experiments/k29_protocol.py selftest >/dev/null 2>&1 || { echo "ОТКАЗ: самопроверка k29_protocol"; return 1; }
  echo "    самопроверки K-29 пройдены"
}
prepare () { $PY experiments/k29_protocol.py prepare || return 3; }
real_check () {
  if $PY experiments/k29_protocol.py check-fresh --kind real-check --device "$DEV"; then return 0; fi
  gpu_free || return 3
  $PY experiments/k29_protocol.py real-check --device "$DEV"
}
smoke () {
  if $PY experiments/k29_protocol.py check-fresh --kind smoke --device "$DEV"; then return 0; fi
  gpu_free || return 3
  $PY experiments/k29_protocol.py smoke --device "$DEV"
}
evaluate () {   # seed which
  local S="$1" W="$2"
  if $PY experiments/k29_harness.py check --seed "$S" --which "$W" >/dev/null 2>&1; then
    echo "    s$S $W: уже готово и проверено"; return 0
  fi
  gpu_free || return 3
  $PY experiments/k29_harness.py eval --seed "$S" --which "$W" --device "$DEV" || return $?
  $PY experiments/k29_harness.py check --seed "$S" --which "$W" || return 3
}
train_one () {   # seed
  local S="$1"
  if [ -f "reports/k29/train_s${S}.json" ]; then echo "    s$S: обучение завершено и закреплено"; return 0; fi
  gpu_free || return 3; disk_ok || return 3
  $PY experiments/k29_protocol.py train --seed "$S" --device "$DEV" || return $?
}

case "$MODE" in
  selftest) selftests; finish $? ;;
  prepare) prepare; finish $? ;;
  real-check) selftests || finish 1; prepare || finish 3; real_check; finish $? ;;
  collector-smoke) selftests || finish 1; prepare || finish 3; smoke; finish $? ;;
  analyze) $PY experiments/k29_analyze.py; finish $? ;;
  latency) gpu_free strict || finish 3; $PY experiments/k29_latency.py --device "$DEV"; finish $? ;;
  run)
    echo "=== ЭТАП самопроверки"; selftests || finish 1
    echo "=== ЭТАП подготовка"; prepare || finish 3
    echo "=== ЭТАП real-check"; real_check || { echo "ОСТАНОВ: real-check"; finish 3; }
    echo "=== ЭТАП collector-smoke"; smoke || { echo "ОСТАНОВ: collector-smoke"; finish 3; }
    for S in 0 1; do
      echo "=== ЭТАП init-control seed $S (30…44, до RL)"; evaluate "$S" init || finish $?
    done
    for S in 0 1; do
      echo "=== ЭТАП обучение seed $S"; train_one "$S" || finish $?
      echo "=== ЭТАП final-оценка seed $S"; evaluate "$S" final || finish $?
      if [ "$S" = 0 ]; then
        echo "=== ЭТАП анализ seed 0 (промежуточно)"; $PY experiments/k29_analyze.py --seeds 0 || true
      fi
    done
    echo "=== ЭТАП анализ двух сидов"; $PY experiments/k29_analyze.py || finish 3
    echo "=== ЭТАП latency"
    if gpu_free strict; then
      $PY experiments/k29_latency.py --device "$DEV" || finish 3
      echo "=== ЭТАП итоговый анализ"; $PY experiments/k29_analyze.py || finish 3
    else
      echo "    латентность ожидается: bash experiments/k29_run.sh latency $DEV"
    fi
    echo "=== ЦЕПОЧКА K-29 ЗАВЕРШЕНА $(date)"; finish 0 ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
