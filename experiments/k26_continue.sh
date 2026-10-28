#!/bin/bash
# K-26: продолжение цепочки run после ошибки k26_run.sh (переменные цикла
# ячеек SR не были локальными: read -r L S T B перезаписывал номер сида, и
# после SR flat_attn24 seed 0 следующий этап получал «seed 10»).
# k26_run.sh закреплён в планах SR (код плана) и не меняется; этот файл —
# та же цепочка с исправлением:
#   - L S T B локальны в run_plan, внешний цикл — SEED/ARCH;
#   - обучение пропускается, если план SR этой модели уже зафиксирован
#     (модель обучена и закреплена; отчёт обучения не перезаписывается).
# Режимы и порядок — как в k26_run.sh; общий lock-файл logs/k26/.k26.lock,
# общий лог logs/k26/run.log.
#   nohup bash experiments/k26_continue.sh run cuda:0 > /dev/null 2>&1 &
set -uo pipefail
MODE="${1:?нужен режим}"
DEV="${2:-cuda:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
ENVSEED=101
BASIS="data/k15f/basis_s0.pt"
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p logs/k26 reports/k26 reports/k26/run
if [ -z "${K26_LOCKED:-}" ]; then
  exec 9>logs/k26/.k26.lock
  flock -n 9 || { echo "ОТКАЗ: другой k26_run.sh уже выполняется"; exit 2; }
  export K26_LOCKED=1
fi
if [ "$MODE" = run ]; then LOGF="logs/k26/run.log"; else LOGF="logs/k26/${MODE}_$(date +%Y%m%dT%H%M%S).log"; fi
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит $(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
PY=python3
MODES="flat_attn24 parents_attn24"

k25_busy () { pgrep -f "k25_run.sh|k25_train.py|k25_harness.py|k25_latency.py|k25_analyze.py" >/dev/null; }
wait_k25 () {
  while k25_busy; do echo "    K-25 ещё выполняется — жду 10 мин ($(date '+%H:%M'))"; sleep 600; done
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
  [ "$free" -ge "$need" ] || { echo "ОТКАЗ: мало места (старые данные автоматически не удаляются)"; return 3; }
}
selftests () {
  local m
  for m in k26_attention k26_model k26_policy k26_harness k26_analyze \
           k23_smoke k16_policy k16_harness k9h_multiarm_gate k21_main_policy; do
    $PY "experiments/${m}.py" --selftest >/dev/null 2>&1 \
      || { echo "ОТКАЗ: самопроверка ${m}"; return 1; }
  done
  $PY experiments/k26_protocol.py selftest >/dev/null 2>&1 \
    || { echo "ОТКАЗ: самопроверка k26_protocol"; return 1; }
  $PY experiments/k26_train.py --selftest >/dev/null 2>&1 \
    || { echo "ОТКАЗ: самопроверка k26_train"; return 1; }
  echo "    самопроверки K-26 пройдены"
}
real_check () {   # оба режима, каждый в своём процессе; затем общее решение micro
  local m
  for m in $MODES; do
    if $PY experiments/k26_train.py check-fresh --kind real-check --mode "$m" \
         --seed 0 --device "$DEV" --micros 8 32; then continue; fi
    gpu_free || return 3
    $PY experiments/k26_train.py real-check --mode "$m" --seed 0 --device "$DEV" --micros 8 32 || return $?
  done
  $PY experiments/k26_train.py micro-decision || return 3
}
micro () { $PY experiments/k26_train.py micro; }
fitcheck () {
  local M; M="$(micro)" || { echo "ОТКАЗ: нет решения о microbatch"; return 3; }
  if $PY experiments/k26_train.py check-fresh --kind fitcheck --device "$DEV" \
       --micro "$M"; then return 0; fi
  gpu_free || return 3
  $PY experiments/k26_train.py fitcheck --device "$DEV" --micro "$M"
}

run_one () {   # план каталог метка набор задача блок
  local P="$1" OUTD="$2" L="$3" S="$4" T="$5" B="$6" try rc TMP CKPT
  local F="$OUTD/${L}_${S}_t${T}_i${B}.json"
  chk () { $PY experiments/k26_protocol.py check-block --plan "$P" --block "$F" "$L" "$S" "$T" "$B"; }
  if [ -f "$F" ] && chk >/dev/null 2>&1; then echo "    $L $S t$T i$B: уже есть"; return 0; fi
  if [ -f "$F" ]; then
    chk || true
    local Q="$OUTD/quarantine/$(date +%Y%m%dT%H%M%S)"; mkdir -p "$Q"
    mv "$F" "${F%.json}.actions.npz" "$Q/" 2>/dev/null || true
    echo "ОСТАНОВ: $L $S t$T i$B — существующий блок не прошёл проверку -> $Q"; return 3
  fi
  CKPT="$($PY experiments/k26_protocol.py info --plan "$P")"
  for try in 1 2 3; do
    echo "    $L $S t$T i$B — попытка $try, $(date '+%H:%M:%S')"
    TMP="$(mktemp)"; rc=0
    $PY experiments/k26_harness.py --k26-plan "$P" \
      --ckpt "$CKPT" --task-suite "$S" --n-envs 5 --seed $ENVSEED \
      --rollout-seed-mode fixed --ensemble off --horizon 8 --max-steps 600 \
      --waiting-steps 10 --run-tag "k26_s${ENVSEED}" --device "$DEV" \
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
  $PY experiments/k26_protocol.py check-plan --plan "$P" || { echo "ОТКАЗ: входы/код/источники плана"; return 1; }
  OUTD="reports/k26/run/p$(sha1sum "$P" | cut -c1-12)"; mkdir -p "$OUTD/quarantine"
  echo "    план $P, каталог $OUTD"
  gpu_free || return 3
  for cell in $($PY experiments/k26_protocol.py cells --plan "$P" --which smoke); do
    IFS=: read -r L S T B <<< "$cell"; run_one "$P" "$OUTD" "$L" "$S" "$T" "$B" || return $?
  done
  CODE=0; $PY experiments/k26_protocol.py smoke-check --plan "$P" || CODE=$?
  [ $CODE -ne 0 ] && { echo "ОСТАНОВ: smoke не прошёл технические проверки"; return $CODE; }
  for cell in $($PY experiments/k26_protocol.py cells --plan "$P" --which all); do
    IFS=: read -r L S T B <<< "$cell"; run_one "$P" "$OUTD" "$L" "$S" "$T" "$B" || return $?
  done
  echo "=== раскатки плана $P закончены $(date)"
}
seed_ready () { $PY -c "import json,sys; e=json.load(open('reports/k26/prepare.json'))['seeds'].get('$1') or {}; sys.exit(0 if e.get('status')=='ready' else 1)"; }
train_one () {   # seed режим
  local M; M="$(micro)" || { echo "ОТКАЗ: нет решения о microbatch"; return 3; }
  gpu_free || return 3; disk_ok || return 3
  $PY experiments/k26_train.py train --seed "$1" --mode "$2" --device "$DEV" --micro "$M" --dtype float16 || return $?
}
sr_one () {   # seed режим
  $PY experiments/k26_protocol.py plan --seed "$1" --mode "$2" --device "$DEV" || return 3
  run_plan "reports/k26/plan_s$1_$2.json" || return $?
}

case "$MODE" in
  selftest) selftests; finish $? ;;
  prepare) $PY experiments/k26_protocol.py prepare; finish $? ;;
  real-check) wait_k25; selftests || finish 1; $PY experiments/k26_protocol.py prepare || finish 3; real_check; finish $? ;;
  fitcheck) wait_k25; selftests || finish 1; $PY experiments/k26_protocol.py prepare || finish 3; fitcheck; finish $? ;;
  analyze) $PY experiments/k26_analyze.py; finish $? ;;
  latency) gpu_free || finish 3; $PY experiments/k26_latency.py --device "$DEV"; finish $? ;;
  run)
    echo "=== ЭТАП ожидание конца K-25"; wait_k25
    echo "=== ЭТАП самопроверки"; selftests || finish 1
    echo "=== ЭТАП подготовка"; $PY experiments/k26_protocol.py prepare || finish 3
    seed_ready 0 && seed_ready 1 || { echo "ОТКАЗ: сиды не готовы в подготовке"; finish 3; }
    echo "=== ЭТАП real-check (оба режима) и решение microbatch"; real_check || { echo "ОСТАНОВ: real-check"; finish 3; }
    echo "=== ЭТАП fitcheck parents_attn24"; fitcheck || { echo "ОСТАНОВ: fitcheck не пройден"; finish 3; }
    for SEED in 0 1; do
      for ARCH in $MODES; do
        if [ -f "reports/k26/plan_s${SEED}_${ARCH}.json" ]; then
          echo "=== ЭТАП обучение $ARCH seed $SEED: уже обучено и закреплено в плане — пропуск"
        else
          echo "=== ЭТАП обучение $ARCH seed $SEED"; train_one "$SEED" "$ARCH" || finish $?
          $PY experiments/k26_train.py maps --seed "$SEED" --mode "$ARCH" --device "$DEV" \
            || echo "    ВНИМАНИЕ: иллюстративные карты $ARCH seed $SEED не сохранены (описательно, цепочка продолжается)"
        fi
        echo "=== ЭТАП SR $ARCH seed $SEED"; sr_one "$SEED" "$ARCH" || finish $?
      done
      if [ "$SEED" = 0 ]; then
        echo "=== ЭТАП анализ seed 0 (промежуточно)"; $PY experiments/k26_analyze.py --seeds 0 || finish 3
      fi
    done
    echo "=== ЭТАП анализ двух сидов"; $PY experiments/k26_analyze.py || finish 3
    echo "=== ЭТАП latency (с исходным BAR в том же замере)"; gpu_free || finish 3; $PY experiments/k26_latency.py --device "$DEV" || finish 3
    echo "=== ЭТАП итоговый анализ (с задержкой)"; $PY experiments/k26_analyze.py || finish 3
    echo "=== ЦЕПОЧКА K-26 ЗАВЕРШЕНА $(date)"; finish 0 ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
