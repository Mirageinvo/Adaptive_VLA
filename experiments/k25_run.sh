#!/bin/bash
# K-25: parents_cond24 (условный readout вдоль пути дерева), seeds 0 и 1.
#   bash experiments/k25_run.sh selftest [cpu]
#   bash experiments/k25_run.sh prepare  [cpu]
#   bash experiments/k25_run.sh real-check DEV
#   bash experiments/k25_run.sh fitcheck   DEV
#   bash experiments/k25_run.sh run        DEV     # вся цепочка
#   bash experiments/k25_run.sh analyze | latency DEV
# run: ждёт, пока K-23/K-24 освободят карту; selftest -> prepare -> real-check и
# fitcheck (готовые отчёты той же версии кода переиспользуются) -> обучение
# cond seed 0 -> план -> smoke -> 30 блоков -> анализ seed 0 -> prepare seed 1
# (сверка записи) -> обучение cond seed 1 ->
# план -> smoke -> 30 блоков -> анализ двух сидов -> latency. Любой
# ненулевой код — останов; повторный run продолжает с проверенного места
# (обучение — с resume.pt, готовые блоки переиспользуются). flock; проверка
# занятости карты и места; MuJoCo и обучение не одновременно.
set -uo pipefail
MODE="${1:?нужен режим}"
DEV="${2:-cuda:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
ENVSEED=101
BASIS="data/k15f/basis_s0.pt"
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p logs/k25 reports/k25 reports/k25/run
if [ -z "${K25_LOCKED:-}" ]; then
  exec 9>logs/k25/.k25.lock
  flock -n 9 || { echo "ОТКАЗ: другой k25_run.sh уже выполняется"; exit 2; }
  export K25_LOCKED=1
fi
if [ "$MODE" = run ]; then LOGF="logs/k25/run.log"; else LOGF="logs/k25/${MODE}_$(date +%Y%m%dT%H%M%S).log"; fi
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит $(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
PY=python3

k23_busy () { pgrep -f "k23_train.py|k23_run.sh|k23_continue.sh|k23_harness.py|k24_train.py|k24_run.sh|k24_harness.py|k24_latency.py" >/dev/null; }
wait_k23_gpu () {
  while k23_busy; do echo "    K-23 ещё выполняется — жду 10 мин ($(date '+%H:%M'))"; sleep 600; done
}
gpu_free () {
  local idx="${DEV#cuda:}" used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  [ -n "$used" ] || { echo "ОТКАЗ: nvidia-smi не отвечает для $DEV"; return 3; }
  echo "    $DEV: занято $used МиБ"
  [ "$used" -le 2048 ] || { echo "ОТКАЗ: на $DEV занято $used МиБ"; return 3; }
}
disk_ok () {
  local need=0 f sz free
  for f in data/k22/runs/parents24/best.pt data/k22/runs/parents24/resume.pt; do
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0); need=$((need + sz))
  done
  need=$((need * 2 * 13 / 10 / 1024 / 1024))
  free=$(df -Pm data | awk 'NR==2 {print $4}')
  echo "    место: нужно ~${need} МиБ, свободно ${free} МиБ"
  [ "$free" -ge "$need" ] || { echo "ОТКАЗ: мало места"; return 3; }
}
selftests () {
  local m
  for m in k25_readout k25_model k25_policy k25_harness k25_analyze \
           k23_smoke k16_policy k16_harness k9h_multiarm_gate k21_main_policy; do
    $PY "experiments/${m}.py" --selftest >/dev/null 2>&1 \
      || { echo "ОТКАЗ: самопроверка ${m}"; return 1; }
  done
  $PY experiments/k25_protocol.py selftest >/dev/null 2>&1 \
    || { echo "ОТКАЗ: самопроверка k25_protocol"; return 1; }
  $PY experiments/k25_train.py --selftest >/dev/null 2>&1 \
    || { echo "ОТКАЗ: самопроверка k25_train"; return 1; }
  echo "    самопроверки K-25 пройдены"
}
real_check () {   # готовый отчёт — только при совпадении полного контракта
  if $PY experiments/k25_train.py check-fresh --kind real-check --seed 0 \
       --device "$DEV" --micros 8 32; then return 0; fi
  gpu_free || return 3
  $PY experiments/k25_train.py real-check --seed 0 --device "$DEV" --micros 8 32
}
fitcheck () {
  if $PY experiments/k25_train.py check-fresh --kind fitcheck --device "$DEV" \
       --micro 32; then return 0; fi
  gpu_free || return 3
  $PY experiments/k25_train.py fitcheck --device "$DEV" --micro 32
}

run_one () {   # план каталог метка набор задача блок
  local P="$1" OUTD="$2" L="$3" S="$4" T="$5" B="$6" try rc TMP CKPT
  local F="$OUTD/${L}_${S}_t${T}_i${B}.json"
  chk () { $PY experiments/k25_protocol.py check-block --plan "$P" --block "$F" "$L" "$S" "$T" "$B"; }
  if [ -f "$F" ] && chk >/dev/null 2>&1; then echo "    $L $S t$T i$B: уже есть"; return 0; fi
  if [ -f "$F" ]; then
    chk || true
    local Q="$OUTD/quarantine/$(date +%Y%m%dT%H%M%S)"; mkdir -p "$Q"
    mv "$F" "${F%.json}.actions.npz" "$Q/" 2>/dev/null || true
    echo "ОСТАНОВ: $L $S t$T i$B — существующий блок не прошёл проверку -> $Q"; return 3
  fi
  CKPT="$($PY experiments/k25_protocol.py info --plan "$P")"
  for try in 1 2 3; do
    echo "    $L $S t$T i$B — попытка $try, $(date '+%H:%M:%S')"
    TMP="$(mktemp)"; rc=0
    $PY experiments/k25_harness.py --k25-plan "$P" \
      --ckpt "$CKPT" --task-suite "$S" --n-envs 5 --seed $ENVSEED \
      --rollout-seed-mode fixed --ensemble off --horizon 8 --max-steps 600 \
      --waiting-steps 10 --run-tag "k25_s${ENVSEED}" --device "$DEV" \
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
  local P="$1" OUTD cell CODE
  [ -f "$P" ] || { echo "ОТКАЗ: нет плана $P"; return 1; }
  $PY experiments/k25_protocol.py check-plan --plan "$P" || { echo "ОТКАЗ: входы/код/источники плана"; return 1; }
  OUTD="reports/k25/run/p$(sha1sum "$P" | cut -c1-12)"; mkdir -p "$OUTD/quarantine"
  echo "    план $P, каталог $OUTD"
  gpu_free || return 3
  for cell in $($PY experiments/k25_protocol.py cells --plan "$P" --which smoke); do
    IFS=: read -r L S T B <<< "$cell"; run_one "$P" "$OUTD" "$L" "$S" "$T" "$B" || return $?
  done
  CODE=0; $PY experiments/k25_protocol.py smoke-check --plan "$P" || CODE=$?
  [ $CODE -ne 0 ] && { echo "ОСТАНОВ: smoke не прошёл технические проверки"; return $CODE; }
  for cell in $($PY experiments/k25_protocol.py cells --plan "$P" --which all); do
    IFS=: read -r L S T B <<< "$cell"; run_one "$P" "$OUTD" "$L" "$S" "$T" "$B" || return $?
  done
  echo "=== раскатки плана $P закончены $(date)"
}
seed_ready () { $PY -c "import json,sys; e=json.load(open('reports/k25/prepare.json'))['seeds'].get('$1') or {}; sys.exit(0 if e.get('status')=='ready' else 1)"; }
train_seed () {   # seed
  gpu_free || return 3; disk_ok || return 3
  $PY experiments/k25_train.py train --seed "$1" --device "$DEV" --micro 32 --dtype float16 || return $?
}
sr_seed () {   # seed
  $PY experiments/k25_protocol.py plan --seed "$1" --device "$DEV" || return 3
  run_plan "reports/k25/plan_s$1.json" || return $?
}

case "$MODE" in
  selftest) selftests; finish $? ;;
  prepare) $PY experiments/k25_protocol.py prepare; finish $? ;;
  real-check) wait_k23_gpu; selftests || finish 1; $PY experiments/k25_protocol.py prepare || finish 3; real_check; finish $? ;;
  fitcheck) wait_k23_gpu; selftests || finish 1; $PY experiments/k25_protocol.py prepare || finish 3; fitcheck; finish $? ;;
  analyze) $PY experiments/k25_analyze.py; finish $? ;;
  latency) gpu_free || finish 3; $PY experiments/k25_latency.py --device "$DEV"; finish $? ;;
  run)
    echo "=== ЭТАП ожидание карты (K-23/K-24)"; wait_k23_gpu
    echo "=== ЭТАП самопроверки"; selftests || finish 1
    echo "=== ЭТАП подготовка"; $PY experiments/k25_protocol.py prepare || finish 3
    seed_ready 0 || { echo "ОТКАЗ: seed 0 не готов в подготовке"; finish 3; }
    echo "=== ЭТАП real-check"; real_check || { echo "ОСТАНОВ: real-check"; finish 3; }
    echo "=== ЭТАП fitcheck"; fitcheck || { echo "ОСТАНОВ: fitcheck не пройден"; finish 3; }
    echo "=== ЭТАП обучение cond seed 0"; train_seed 0 || finish $?
    echo "=== ЭТАП SR seed 0"; sr_seed 0 || finish $?
    echo "=== ЭТАП анализ seed 0"; $PY experiments/k25_analyze.py --seeds 0 || finish 3
    echo "=== ЭТАП подготовка seed 1"; $PY experiments/k25_protocol.py prepare || finish 3
    seed_ready 1 || { echo "ОТКАЗ: seed 1 не готов в подготовке"; finish 3; }
    echo "=== ЭТАП обучение cond seed 1"; train_seed 1 || finish $?
    echo "=== ЭТАП SR seed 1"; sr_seed 1 || finish $?
    echo "=== ЭТАП анализ двух сидов"; $PY experiments/k25_analyze.py || finish 3
    echo "=== ЭТАП latency"; gpu_free || finish 3; $PY experiments/k25_latency.py --device "$DEV" || finish 3
    echo "=== ЦЕПОЧКА K-25 ЗАВЕРШЕНА $(date)"; finish 0 ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
