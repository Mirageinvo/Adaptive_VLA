#!/bin/bash
# K-30: дискретные планы корней (H, parents_hybrid24) против непрерывной
# смеси тех же прототипов (C, parents_softplan24), seeds 0 и 1, LIBERO-10.
#   bash experiments/k30_run.sh selftest [cpu]
#   bash experiments/k30_run.sh prepare            # кодек (CPU) + base prepare
#   bash experiments/k30_run.sh real-check DEV     # + решение micro
#   bash experiments/k30_run.sh fitcheck   DEV
#   bash experiments/k30_run.sh run        DEV     # вся цепочка
#   bash experiments/k30_run.sh analyze | latency DEV
# run: ждёт окончания K-29 -> самопроверки -> prepare (сверка) -> real-check и
# fitcheck (готовые того же контракта переиспользуются) -> H_s0 обучение/SR ->
# C_s0 -> H_s1 -> C_s1 -> анализ двух сидов -> latency (свободная карта) ->
# итоговый анализ. Все четыре обучения — независимо от промежуточного SR.
# Технический отказ — останов; повторный run продолжает с проверенного места.
# flock; MuJoCo и обучение не одновременно. Перед обучением/раскатками —
# ожидание свободной памяти (см. gpu_free); latency — всегда занято ≤ 2048.
set -uo pipefail
MODE="${1:?нужен режим}"
DEV="${2:-cuda:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
ENVSEED=101
BASIS="data/k15f/basis_s0.pt"
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p logs/k30 reports/k30 reports/k30/run data/k30
if [ -z "${K30_LOCKED:-}" ]; then
  exec 9>logs/k30/.k30.lock
  flock -n 9 || { echo "ОТКАЗ: другой k30_run.sh уже выполняется"; exit 2; }
  export K30_LOCKED=1
fi
if [ "$MODE" = run ]; then LOGF="logs/k30/run.log"; else LOGF="logs/k30/${MODE}_$(date +%Y%m%dT%H%M%S).log"; fi
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит $(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
PY=python3

k29_busy () { pgrep -f "k29_run.sh|k29_protocol.py|k29_harness.py|k29_latency.py" >/dev/null; }
wait_k29 () { while k29_busy; do echo "    K-29 ещё выполняется — жду 10 мин ($(date '+%H:%M'))"; sleep 600; done; }
# gpu_free strict — для latency: занято ≤ 2048 МиБ, без ожидания (чужая
# нагрузка искажает замер). gpu_free train|roll — по свободной памяти:
# нужно K30_NEED_TRAIN_MB (14000; пик micro 32 ≈ 9.2 ГиБ) или
# K30_NEED_ROLL_MB (8000); если меньше — ждём по 10 мин до K30_GPU_WAIT_MIN
# (720) и только потом отказ.
gpu_free () {
  local idx="${DEV#cuda:}" mode="${1:-train}" used free need waited=0
  local cap="${K30_GPU_WAIT_MIN:-720}"
  while :; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
    [ -n "$used" ] && [ -n "$free" ] || { echo "ОТКАЗ: nvidia-smi не отвечает для $DEV"; return 3; }
    if [ "$mode" = strict ]; then
      echo "    $DEV: занято $used МиБ (допустимо до 2048)"
      [ "$used" -le 2048 ] || { echo "ОТКАЗ: на $DEV занято $used МиБ"; return 3; }
      return 0
    fi
    if [ "$mode" = roll ]; then need="${K30_NEED_ROLL_MB:-8000}"; else need="${K30_NEED_TRAIN_MB:-14000}"; fi
    if [ "$free" -ge "$need" ]; then
      echo "    $DEV: свободно $free МиБ, занято $used (нужно ≥ $need)"; return 0
    fi
    if [ "$waited" -ge "$cap" ]; then
      echo "ОТКАЗ: на $DEV свободно $free МиБ < $need после ожидания $waited мин"; return 3
    fi
    echo "    $DEV: свободно $free МиБ < $need — жду 10 мин (прошло $waited из $cap, $(date '+%H:%M'))"
    sleep 600; waited=$((waited + 10))
  done
}
# Место на вариант: обучение seed 0 — 4.1 ГиБ (файл продолжения + best.pt),
# раскатки — ~0.15 ГиБ; нужно K30_NEED_DISK_MB (8000) свободно. Если
# меньше — ждём по 10 мин до K30_DISK_WAIT_MIN (360) и только потом отказ.
disk_ok () {
  local free need="${K30_NEED_DISK_MB:-8000}" waited=0 cap="${K30_DISK_WAIT_MIN:-360}"
  while :; do
    free=$(df -Pm data | awk 'NR==2 {print $4}')
    [ -n "$free" ] || { echo "ОТКАЗ: df не отвечает"; return 3; }
    if [ "$free" -ge "$need" ]; then
      echo "    место: свободно ${free} МиБ (нужно ≥ $need)"; return 0
    fi
    if [ "$waited" -ge "$cap" ]; then
      echo "ОТКАЗ: мало места — свободно ${free} МиБ < $need после ожидания $waited мин"; return 3
    fi
    echo "    место: свободно ${free} МиБ < $need — жду 10 мин (прошло $waited из $cap, $(date '+%H:%M'))"
    sleep 600; waited=$((waited + 10))
  done
}
selftests () {
  local m
  for m in k30_codec k30_data k30_readout k30_model k30_train k30_policy \
           k30_harness k30_latency k16_policy k16_harness k9h_multiarm_gate \
           k21_main_policy; do
    $PY "experiments/${m}.py" --selftest >/dev/null 2>&1 \
      || { echo "ОТКАЗ: самопроверка ${m}"; return 1; }
  done
  $PY experiments/k30_protocol.py selftest >/dev/null 2>&1 || { echo "ОТКАЗ: самопроверка k30_protocol"; return 1; }
  $PY experiments/k30_analyze.py --selftest >/dev/null 2>&1 || { echo "ОТКАЗ: самопроверка k30_analyze"; return 1; }
  echo "    самопроверки K-30 пройдены"
}
prepare () {
  $PY experiments/k30_data.py codec || return 3
  $PY experiments/k30_train.py prepare || return 3
}
micro () { $PY experiments/k30_train.py micro; }
real_check () {
  if $PY experiments/k30_train.py check-fresh --kind real-check --device "$DEV"; then return 0; fi
  gpu_free || return 3
  $PY experiments/k30_train.py real-check --device "$DEV"
}
fitcheck () {
  local M; M="$(micro)" || { echo "ОТКАЗ: нет решения о micro"; return 3; }
  if $PY experiments/k30_train.py check-fresh --kind fitcheck --device "$DEV" --micro "$M"; then return 0; fi
  gpu_free || return 3
  $PY experiments/k30_train.py fitcheck --device "$DEV" --micro "$M"
}
run_one () {   # план каталог метка набор задача блок
  local P="$1" OUTD="$2" L="$3" S="$4" T="$5" B="$6" try rc TMP CKPT
  local F="$OUTD/${L}_${S}_t${T}_i${B}.json"
  chk () { $PY experiments/k30_protocol.py check-block --plan "$P" --block "$F" "$L" "$S" "$T" "$B"; }
  if [ -f "$F" ] && chk >/dev/null 2>&1; then echo "    $L $S t$T i$B: уже есть"; return 0; fi
  if [ -f "$F" ]; then
    chk || true
    local Q="$OUTD/quarantine/$(date +%Y%m%dT%H%M%S)"; mkdir -p "$Q"
    mv "$F" "${F%.json}.actions.npz" "$Q/" 2>/dev/null || true
    echo "ОСТАНОВ: $L $S t$T i$B — существующий блок не прошёл проверку -> $Q"; return 3
  fi
  CKPT="$($PY experiments/k30_protocol.py info --plan "$P")"
  for try in 1 2 3; do
    echo "    $L $S t$T i$B — попытка $try, $(date '+%H:%M:%S')"
    TMP="$(mktemp)"; rc=0
    $PY experiments/k30_harness.py --k30-plan "$P" \
      --ckpt "$CKPT" --task-suite "$S" --n-envs 5 --seed $ENVSEED \
      --rollout-seed-mode fixed --ensemble off --horizon 8 --max-steps 600 \
      --waiting-steps 10 --run-tag "k30_s${ENVSEED}" --device "$DEV" \
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
run_plan () {
  local P="$1" OUTD cell CODE L S T B
  [ -f "$P" ] || { echo "ОТКАЗ: нет плана $P"; return 1; }
  $PY experiments/k30_protocol.py check-plan --plan "$P" || { echo "ОТКАЗ: входы/код/источники плана"; return 1; }
  OUTD="reports/k30/run/p$(sha1sum "$P" | cut -c1-12)"; mkdir -p "$OUTD/quarantine"
  echo "    план $P, каталог $OUTD"
  gpu_free roll || return 3
  for cell in $($PY experiments/k30_protocol.py cells --plan "$P" --which smoke); do
    IFS=: read -r L S T B <<< "$cell"; run_one "$P" "$OUTD" "$L" "$S" "$T" "$B" || return $?
  done
  CODE=0; $PY experiments/k30_protocol.py smoke-check --plan "$P" || CODE=$?
  [ $CODE -ne 0 ] && { echo "ОСТАНОВ: smoke не прошёл технические проверки"; return $CODE; }
  for cell in $($PY experiments/k30_protocol.py cells --plan "$P" --which all); do
    IFS=: read -r L S T B <<< "$cell"; run_one "$P" "$OUTD" "$L" "$S" "$T" "$B" || return $?
  done
  echo "=== раскатки плана $P закончены $(date)"
}
variant () {   # seed mode
  local S="$1" Mo="$2" M
  if [ -f "reports/k30/train_s${S}_${Mo}.json" ]; then
    echo "    s$S $Mo: обучение завершено и закреплено"
  else
    M="$(micro)" || { echo "ОТКАЗ: нет решения о micro"; return 3; }
    gpu_free || return 3; disk_ok || return 3
    $PY experiments/k30_train.py train --seed "$S" --mode "$Mo" --device "$DEV" --micro "$M" || return $?
  fi
  $PY experiments/k30_protocol.py plan --seed "$S" --mode "$Mo" --device "$DEV" || return 3
  run_plan "reports/k30/plan_s${S}_${Mo}.json" || return $?
}

case "$MODE" in
  selftest) selftests; finish $? ;;
  prepare) prepare; finish $? ;;
  real-check) wait_k29; selftests || finish 1; prepare || finish 3; real_check; finish $? ;;
  fitcheck) wait_k29; selftests || finish 1; prepare || finish 3; fitcheck; finish $? ;;
  analyze) $PY experiments/k30_analyze.py; finish $? ;;
  latency) gpu_free strict || finish 3; $PY experiments/k30_latency.py --device "$DEV"; finish $? ;;
  run)
    echo "=== ЭТАП ожидание конца K-29"; wait_k29
    echo "=== ЭТАП самопроверки"; selftests || finish 1
    echo "=== ЭТАП подготовка (кодек и base prepare)"; prepare || finish 3
    echo "=== ЭТАП real-check и решение micro"; real_check || { echo "ОСТАНОВ: real-check"; finish 3; }
    echo "=== ЭТАП fitcheck"; fitcheck || { echo "ОСТАНОВ: fitcheck не пройден"; finish 3; }
    for SM in "0 hard" "0 soft" "1 hard" "1 soft"; do
      read -r S Mo <<< "$SM"
      echo "=== ЭТАП $Mo seed $S: обучение и SR"; variant "$S" "$Mo" || finish $?
      if [ "$SM" = "0 soft" ]; then
        echo "=== ЭТАП анализ seed 0 (промежуточно)"; $PY experiments/k30_analyze.py --seeds 0 || true
      fi
    done
    echo "=== ЭТАП анализ двух сидов"; $PY experiments/k30_analyze.py || finish 3
    echo "=== ЭТАП latency"
    if gpu_free strict; then
      $PY experiments/k30_latency.py --device "$DEV" || finish 3
      echo "=== ЭТАП итоговый анализ"; $PY experiments/k30_analyze.py || finish 3
    else
      echo "    латентность ожидается: bash experiments/k30_run.sh latency $DEV"
    fi
    echo "=== ЦЕПОЧКА K-30 ЗАВЕРШЕНА $(date)"; finish 0 ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
