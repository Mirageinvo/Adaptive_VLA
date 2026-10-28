#!/bin/bash
# K-27: parents_mem24 (память дерева о собственном плане поверх
# замороженного parents24), seeds 0 и 1.
#   bash experiments/k27_run.sh selftest [cpu]
#   bash experiments/k27_run.sh prepare  [cpu]     # индекс цепочек + запись сидов
#   bash experiments/k27_run.sh real-check DEV     # оба сида + решение micro
#   bash experiments/k27_run.sh fitcheck   DEV
#   bash experiments/k27_run.sh run        DEV     # вся цепочка
#   bash experiments/k27_run.sh analyze | latency DEV
# run: ждёт окончания K-26; selftest -> prepare -> real-check -> fitcheck ->
# для seed 0 и 1: memory-off smoke (задачи 0 и 8 блока 30) и сверка с
# закреплёнными блоками parents24 -> обучение 2000 -> план -> smoke -> 30
# блоков; после seed 0 — промежуточный анализ; затем анализ двух сидов ->
# latency (свободная карта) -> итоговый анализ. Любой ненулевой код —
# останов; повторный run продолжает с проверенного места. SR и p-value на
# код не влияют. flock; проверка карты и места; MuJoCo и обучение не
# одновременно.
#   K27_MAX_USED_MB=N — допустимая занятость карты чужими процессами для
#   обучения и SR (по умолчанию 2048 МиБ); latency — всегда ≤ 2048 МиБ.
set -uo pipefail
MODE="${1:?нужен режим}"
DEV="${2:-cuda:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
ENVSEED=101
BASIS="data/k15f/basis_s0.pt"
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p logs/k27 reports/k27 reports/k27/run
if [ -z "${K27_LOCKED:-}" ]; then
  exec 9>logs/k27/.k27.lock
  flock -n 9 || { echo "ОТКАЗ: другой k27_run.sh уже выполняется"; exit 2; }
  export K27_LOCKED=1
fi
if [ "$MODE" = run ]; then LOGF="logs/k27/run.log"; else LOGF="logs/k27/${MODE}_$(date +%Y%m%dT%H%M%S).log"; fi
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит $(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
PY=python3

k26_busy () { pgrep -f "k26_run.sh|k26_continue.sh|k26_train.py|k26_harness.py|k26_latency.py" >/dev/null; }
wait_k26 () {
  while k26_busy; do echo "    K-26 ещё выполняется — жду 10 мин ($(date '+%H:%M'))"; sleep 600; done
}
gpu_free () {   # [strict]: для latency порог всегда 2048 МиБ
  local idx="${DEV#cuda:}" used lim="${K27_MAX_USED_MB:-2048}"
  [ "${1:-}" = strict ] && lim=2048
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  [ -n "$used" ] || { echo "ОТКАЗ: nvidia-smi не отвечает для $DEV"; return 3; }
  echo "    $DEV: занято $used МиБ (допустимо до $lim)"
  [ "$used" -le "$lim" ] || { echo "ОТКАЗ: на $DEV занято $used МиБ"; return 3; }
}
disk_ok () {
  local free
  free=$(df -Pm data | awk 'NR==2 {print $4}')
  echo "    место: свободно ${free} МиБ (нужно ~200)"
  [ "$free" -ge 200 ] || { echo "ОТКАЗ: мало места"; return 3; }
}
selftests () {
  local m
  for m in k27_memory k27_model k27_policy k27_harness k27_analyze k27_latency \
           k16_policy k16_harness k9h_multiarm_gate k21_main_policy; do
    $PY "experiments/${m}.py" --selftest >/dev/null 2>&1 \
      || { echo "ОТКАЗ: самопроверка ${m}"; return 1; }
  done
  $PY experiments/k27_data.py selftest >/dev/null 2>&1 || { echo "ОТКАЗ: самопроверка k27_data"; return 1; }
  $PY experiments/k27_protocol.py selftest >/dev/null 2>&1 || { echo "ОТКАЗ: самопроверка k27_protocol"; return 1; }
  $PY experiments/k27_train.py --selftest >/dev/null 2>&1 || { echo "ОТКАЗ: самопроверка k27_train"; return 1; }
  echo "    самопроверки K-27 пройдены"
}
prepare () {
  if [ ! -f reports/k27/sequences.json ]; then
    $PY experiments/k27_data.py prepare || return 3
  else
    echo "    индекс последовательностей уже есть: reports/k27/sequences.json"
  fi
  $PY experiments/k27_protocol.py prepare || return 3
}
micro () { $PY experiments/k27_train.py micro; }
real_check () {
  if $PY experiments/k27_train.py check-fresh --kind real-check --device "$DEV" \
       --micros 8 16 32; then return 0; fi
  gpu_free || return 3
  $PY experiments/k27_train.py real-check --device "$DEV" --micros 8 16 32
}
fitcheck () {
  local M; M="$(micro)" || { echo "ОТКАЗ: нет решения о microbatch"; return 3; }
  if $PY experiments/k27_train.py check-fresh --kind fitcheck --device "$DEV" \
       --micro "$M"; then return 0; fi
  gpu_free || return 3
  $PY experiments/k27_train.py fitcheck --device "$DEV" --micro "$M"
}
run_one () {   # план каталог метка набор задача блок
  local P="$1" OUTD="$2" L="$3" S="$4" T="$5" B="$6" try rc TMP CKPT F Q
  F="$OUTD/${L}_${S}_t${T}_i${B}.json"
  chk () { $PY experiments/k27_protocol.py check-block --plan "$P" --block "$F" "$L" "$S" "$T" "$B"; }
  if [ -f "$F" ] && chk >/dev/null 2>&1; then echo "    $L $S t$T i$B: уже есть"; return 0; fi
  if [ -f "$F" ]; then
    chk || true
    Q="$OUTD/quarantine/$(date +%Y%m%dT%H%M%S)"; mkdir -p "$Q"
    mv "$F" "${F%.json}.actions.npz" "$Q/" 2>/dev/null || true
    echo "ОСТАНОВ: $L $S t$T i$B — существующий блок не прошёл проверку -> $Q"; return 3
  fi
  CKPT="$($PY experiments/k27_protocol.py info --plan "$P")"
  for try in 1 2 3; do
    echo "    $L $S t$T i$B — попытка $try, $(date '+%H:%M:%S')"
    TMP="$(mktemp)"; rc=0
    $PY experiments/k27_harness.py --k27-plan "$P" \
      --ckpt "$CKPT" --task-suite "$S" --n-envs 5 --seed $ENVSEED \
      --rollout-seed-mode fixed --ensemble off --horizon 8 --max-steps 600 \
      --waiting-steps 10 --run-tag "k27_s${ENVSEED}" --device "$DEV" \
      --save-actions --policy k15f --k15f-basis "$BASIS" --arm-label "$L" \
      --task-id "$T" --init-starts "$B" \
      --out "$OUTD/${L}_${S}_t${T}_i{i0}.json" > "$TMP" 2>&1 || rc=$?
    cat "$TMP"
    if [ $rc -eq 0 ]; then
      rm -f "$TMP"
      if ! chk >/dev/null 2>&1; then chk || true; echo "ОСТАНОВ: $L $S t$T i$B — новый результат не прошёл проверку"; return 3; fi
      sleep 3; return 0
    fi
    if grep -q "No CUDA GPUs are available" "$TMP" && [ "$try" -lt 3 ]; then
      echo "    нет GPU — пауза 120 с и повтор"; rm -f "$TMP"; sleep 120; continue
    fi
    rm -f "$TMP"; echo "ОСТАНОВ: $L $S t$T i$B — код $rc"; return $rc
  done
}
run_plan () {   # план which(smoke|all)
  local P="$1" WHICH="$2" OUTD cell CODE L S T B
  [ -f "$P" ] || { echo "ОТКАЗ: нет плана $P"; return 1; }
  $PY experiments/k27_protocol.py check-plan --plan "$P" || { echo "ОТКАЗ: входы/код/источники плана"; return 1; }
  OUTD="reports/k27/run/p$(sha1sum "$P" | cut -c1-12)"; mkdir -p "$OUTD/quarantine"
  echo "    план $P, каталог $OUTD"
  gpu_free || return 3
  for cell in $($PY experiments/k27_protocol.py cells --plan "$P" --which smoke); do
    IFS=: read -r L S T B <<< "$cell"; run_one "$P" "$OUTD" "$L" "$S" "$T" "$B" || return $?
  done
  CODE=0; $PY experiments/k27_protocol.py smoke-check --plan "$P" || CODE=$?
  [ $CODE -ne 0 ] && { echo "ОСТАНОВ: smoke не прошёл технические проверки"; return $CODE; }
  [ "$WHICH" = smoke ] && return 0
  for cell in $($PY experiments/k27_protocol.py cells --plan "$P" --which all); do
    IFS=: read -r L S T B <<< "$cell"; run_one "$P" "$OUTD" "$L" "$S" "$T" "$B" || return $?
  done
  echo "=== раскатки плана $P закончены $(date)"
}
off_check () {   # seed
  local SEED="$1"
  $PY experiments/k27_protocol.py plan --seed "$SEED" --kind off --device "$DEV" || return 3
  run_plan "reports/k27/plan_s${SEED}_off.json" smoke || return $?
  $PY experiments/k27_protocol.py offcheck --plan "reports/k27/plan_s${SEED}_off.json" || return 3
}
train_one () {   # seed
  local SEED="$1" M
  M="$(micro)" || { echo "ОТКАЗ: нет решения о microbatch"; return 3; }
  gpu_free || return 3; disk_ok || return 3
  $PY experiments/k27_train.py train --seed "$SEED" --device "$DEV" --micro "$M" --dtype float16 || return $?
}
sr_one () {   # seed
  local SEED="$1"
  $PY experiments/k27_protocol.py plan --seed "$SEED" --kind main --device "$DEV" || return 3
  run_plan "reports/k27/plan_s${SEED}.json" all || return $?
}

case "$MODE" in
  selftest) selftests; finish $? ;;
  prepare) prepare; finish $? ;;
  real-check) wait_k26; selftests || finish 1; prepare || finish 3; real_check; finish $? ;;
  fitcheck) wait_k26; selftests || finish 1; prepare || finish 3; fitcheck; finish $? ;;
  analyze) $PY experiments/k27_analyze.py; finish $? ;;
  latency) gpu_free strict || finish 3; $PY experiments/k27_latency.py --device "$DEV"; finish $? ;;
  run)
    echo "=== ЭТАП ожидание конца K-26"; wait_k26
    echo "=== ЭТАП самопроверки"; selftests || finish 1
    echo "=== ЭТАП подготовка"; prepare || finish 3
    echo "=== ЭТАП real-check и решение microbatch"; real_check || { echo "ОСТАНОВ: real-check"; finish 3; }
    echo "=== ЭТАП fitcheck"; fitcheck || { echo "ОСТАНОВ: fitcheck не пройден"; finish 3; }
    for SEED in 0 1; do
      echo "=== ЭТАП memory off seed $SEED (сверка с parents24)"; off_check "$SEED" || finish $?
      if [ -f "reports/k27/plan_s${SEED}.json" ]; then
        echo "=== ЭТАП обучение seed $SEED: уже обучено и закреплено в плане — пропуск"
      else
        echo "=== ЭТАП обучение seed $SEED"; train_one "$SEED" || finish $?
      fi
      echo "=== ЭТАП SR seed $SEED"; sr_one "$SEED" || finish $?
      if [ "$SEED" = 0 ]; then
        echo "=== ЭТАП анализ seed 0 (промежуточно)"; $PY experiments/k27_analyze.py --seeds 0 || finish 3
      fi
    done
    echo "=== ЭТАП анализ двух сидов"; $PY experiments/k27_analyze.py || finish 3
    echo "=== ЭТАП latency"; gpu_free strict || finish 3; $PY experiments/k27_latency.py --device "$DEV" || finish 3
    echo "=== ЭТАП итоговый анализ (с задержкой)"; $PY experiments/k27_analyze.py || finish 3
    echo "=== ЦЕПОЧКА K-27 ЗАВЕРШЕНА $(date)"; finish 0 ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
