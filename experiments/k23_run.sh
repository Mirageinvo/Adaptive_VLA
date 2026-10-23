#!/bin/bash
# K-23: parents24 s0 на spatial/object/goal (пакет A), повтор обучения
# flat24/parents24 с training seed 1 (пакет B) и его SR на LIBERO-10 (пакет C).
#   bash experiments/k23_run.sh selftest
#   bash experiments/k23_run.sh plan-suites  [DEV]   | run-suites  [DEV]
#   bash experiments/k23_run.sh train SEED   [DEV]   (smoke тренера + обучение)
#   bash experiments/k23_run.sh plan-repeat SEED [DEV] | run-repeat SEED [DEV]
#   bash experiments/k23_run.sh analyze-suites | analyze-repeat SEED
#   bash experiments/k23_run.sh chain DEV SEED        (всё по порядку)
# Каждый блок — отдельный холодный процесс; проверка сразу после; smoke
# первым, при его техническом успехе — остальные блоки и анализ (SR smoke на
# продолжение не влияет). Любой ненулевой код — останов; повтор только при
# потере GPU. Сам с собой параллельно не запускается (flock). Перед длинными
# этапами — проверка занятости карты; перед обучением — свободного места.
set -uo pipefail
MODE="${1:?нужен режим}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
case "$MODE" in
  train|plan-repeat|run-repeat|analyze-repeat) SEEDT="${2:?нужен training seed}"; DEV="${3:-cuda:0}";;
  chain) DEV="${2:-cuda:0}"; SEEDT="${3:?нужен training seed}";;
  *) DEV="${2:-cuda:0}"; SEEDT="";;
esac
ENVSEED=101
BASIS="data/k15f/basis_s0.pt"
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
mkdir -p logs/k23 reports/k23 reports/k23/run
if [ -z "${K23_LOCKED:-}" ]; then
  exec 9>logs/k23/.k23.lock
  flock -n 9 || { echo "ОТКАЗ: другой k23_run.sh уже выполняется"; exit 2; }
  export K23_LOCKED=1
fi
LOGF="logs/k23/${MODE}${SEEDT:+_s$SEEDT}_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV${SEEDT:+, training seed $SEEDT}, коммит" \
     "$(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
PY=python3

gpu_free () {   # занятость карты: >2 ГБ уже занято — отказ
  local idx="${DEV#cuda:}" used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  if [ -z "$used" ]; then echo "ОТКАЗ: nvidia-smi не отвечает для $DEV"; return 3; fi
  echo "    $DEV: занято $used МиБ"
  if [ "$used" -gt 2048 ]; then
    echo "ОТКАЗ: на $DEV уже занято $used МиБ — карта не свободна"; return 3
  fi
}

disk_ok () {   # место под пару: (best + resume) × 2 модели × 2 (атомарная запись) × 1.3
  local need=0 f sz
  for f in data/k22/runs/parents24/best.pt data/k22/runs/parents24/resume.pt \
           data/k21/runs/flat24/best.pt data/k21/runs/flat24/resume.pt; do
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0); need=$((need + sz))
  done
  need=$((need * 2 * 13 / 10 / 1024 / 1024))
  local free
  free=$(df -Pm data | awk 'NR==2 {print $4}')
  echo "    место: нужно ~${need} МиБ (пара, с атомарной записью и запасом), свободно ${free} МиБ"
  du -sh data/k21/runs data/k22/runs data/k23/runs 2>/dev/null || true
  if [ "$free" -lt "$need" ]; then echo "ОТКАЗ: мало места"; return 3; fi
}

selftests () {
  local m
  for m in k23_policy k23_harness k16_policy k16_harness k9h_multiarm_gate \
           k21_main_policy k22_policy; do
    $PY "experiments/${m}.py" --selftest >/dev/null \
      || { echo "ОТКАЗ: самопроверка ${m}"; return 1; }
  done
  $PY experiments/k23_protocol.py selftest >/dev/null \
    || { echo "ОТКАЗ: самопроверка k23_protocol"; return 1; }
  $PY experiments/k23_analyze.py selftest >/dev/null \
    || { echo "ОТКАЗ: самопроверка k23_analyze"; return 1; }
  $PY experiments/k23_train.py --selftest >/dev/null 2>&1 \
    || { echo "ОТКАЗ: самопроверка k23_train"; return 1; }
  echo "    самопроверки K-23 пройдены"
}

run_one () {   # план каталог метка набор задача блок
  local P="$1" OUTD="$2" L="$3" S="$4" T="$5" B="$6" try rc TMP CKPT
  local F="$OUTD/${L}_${S}_t${T}_i${B}.json"
  chk () { $PY experiments/k23_protocol.py check-block --plan "$P" \
             --block "$F" "$L" "$S" "$T" "$B"; }
  if [ -f "$F" ] && chk >/dev/null 2>&1; then echo "    $L $S t$T i$B: уже есть"; return 0; fi
  if [ -f "$F" ]; then
    chk || true
    local Q="$OUTD/quarantine/$(date +%Y%m%dT%H%M%S)"; mkdir -p "$Q"
    mv "$F" "${F%.json}.actions.npz" "$Q/" 2>/dev/null || true
    echo "ОСТАНОВ: $L $S t$T i$B — существующий блок не прошёл проверку -> $Q"
    return 3
  fi
  CKPT="$($PY experiments/k23_protocol.py info --plan "$P")"
  for try in 1 2 3; do
    echo "    $L $S t$T i$B — попытка $try, $(date '+%H:%M:%S')"
    TMP="$(mktemp)"; rc=0
    $PY experiments/k23_harness.py --k23-plan "$P" \
      --ckpt "$CKPT" --task-suite "$S" --n-envs 5 --seed $ENVSEED \
      --rollout-seed-mode fixed --ensemble off --horizon 8 --max-steps 600 \
      --waiting-steps 10 --run-tag "k23_s${ENVSEED}" --device "$DEV" \
      --save-actions --policy k15f --k15f-basis "$BASIS" --arm-label "$L" \
      --task-id "$T" --init-starts "$B" \
      --out "$OUTD/${L}_${S}_t${T}_i{i0}.json" > "$TMP" 2>&1 || rc=$?
    cat "$TMP"
    if [ $rc -eq 0 ]; then
      rm -f "$TMP"
      if ! chk >/dev/null 2>&1; then
        chk || true
        echo "ОСТАНОВ: $L $S t$T i$B — новый результат не прошёл проверку"
        return 3
      fi
      sleep 3; return 0
    fi
    if grep -q "No CUDA GPUs are available" "$TMP" && [ "$try" -lt 3 ]; then
      echo "    нет GPU — пауза 120 с и повтор"; rm -f "$TMP"; sleep 120
      continue
    fi
    rm -f "$TMP"; echo "ОСТАНОВ: $L $S t$T i$B — код $rc"; return $rc
  done
}

run_plan () {   # план; анализ — вызывающий
  local P="$1" OUTD cell CODE
  [ -f "$P" ] || { echo "ОТКАЗ: нет плана $P"; return 1; }
  $PY experiments/k23_protocol.py check-plan --plan "$P" \
    || { echo "ОТКАЗ: входы, код или планы-источники не совпадают с планом"; return 1; }
  OUTD="reports/k23/run/p$(sha1sum "$P" | cut -c1-12)"
  mkdir -p "$OUTD/quarantine"
  echo "    план $P, каталог $OUTD"
  gpu_free || return 3
  for cell in $($PY experiments/k23_protocol.py cells --plan "$P" --which smoke); do
    IFS=: read -r L S T B <<< "$cell"
    run_one "$P" "$OUTD" "$L" "$S" "$T" "$B" || return $?
  done
  CODE=0
  $PY experiments/k23_protocol.py smoke-check --plan "$P" || CODE=$?
  [ $CODE -ne 0 ] && { echo "ОСТАНОВ: smoke не прошёл технические проверки"; return $CODE; }
  for cell in $($PY experiments/k23_protocol.py cells --plan "$P" --which all); do
    IFS=: read -r L S T B <<< "$cell"
    run_one "$P" "$OUTD" "$L" "$S" "$T" "$B" || return $?
  done
  echo "=== раскатки плана $P закончены $(date)"
}

do_train () {
  gpu_free || return 3
  disk_ok || return 3
  $PY experiments/k23_train.py smoke --seed "$SEEDT" --device "$DEV" --micro 32 \
    || { echo "ОСТАНОВ: smoke тренера"; return 3; }
  $PY experiments/k23_train.py train --seed "$SEEDT" --device "$DEV" \
    --micro 32 --dtype float16 || return $?
}

PS="reports/k23/plan_suites_s0.json"
PR () { echo "reports/k23/plan_repeat_s$1.json"; }
case "$MODE" in
  selftest) selftests; finish $? ;;
  plan-suites) selftests || finish 1
    $PY experiments/k23_protocol.py plan-suites --device "$DEV"; finish $? ;;
  run-suites) selftests || finish 1
    run_plan "$PS" || finish $?
    $PY experiments/k23_analyze.py suites; finish $? ;;
  analyze-suites) $PY experiments/k23_analyze.py suites; finish $? ;;
  train) selftests || finish 1; do_train; finish $? ;;
  plan-repeat) selftests || finish 1
    $PY experiments/k23_protocol.py plan-repeat --seed "$SEEDT" --device "$DEV"; finish $? ;;
  run-repeat) selftests || finish 1
    run_plan "$(PR "$SEEDT")" || finish $?
    $PY experiments/k23_analyze.py repeat --seed "$SEEDT"; finish $? ;;
  analyze-repeat) $PY experiments/k23_analyze.py repeat --seed "$SEEDT"; finish $? ;;
  chain)
    echo "=== ЭТАП самопроверки"; selftests || finish 1
    echo "=== ЭТАП план наборов s0"; $PY experiments/k23_protocol.py plan-suites --device "$DEV" || finish 3
    echo "=== ЭТАП раскатки наборов s0"; run_plan "$PS" || finish $?
    echo "=== ЭТАП анализ наборов"; $PY experiments/k23_analyze.py suites || finish $?
    echo "=== ЭТАП обучение пары seed $SEEDT"; do_train || finish $?
    echo "=== ЭТАП план повтора seed $SEEDT"; $PY experiments/k23_protocol.py plan-repeat --seed "$SEEDT" --device "$DEV" || finish 3
    echo "=== ЭТАП раскатки повтора seed $SEEDT"; run_plan "$(PR "$SEEDT")" || finish $?
    echo "=== ЭТАП анализ повтора и сводка сидов"; $PY experiments/k23_analyze.py repeat --seed "$SEEDT" || finish $?
    echo "=== ЦЕПОЧКА K-23 ЗАВЕРШЕНА $(date); лог $LOGF"; finish 0 ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
