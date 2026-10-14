#!/bin/bash
# K-16: пилот — plan, smoke (задачи 0 и 8), run (все 10 задач), analyze.
#
#   plan     план (чекпойнт голов, доноры, эталоны K-15h, код);
#   smoke    задачи 0 и 8, блок 25, шесть политик (12 процессов), затем
#            проверка цепочки; результаты входят в пилот;
#   run      все задачи плана; готовые проверенные блоки пропускаются;
#            затем анализ;
#   analyze  только анализ.
# Лог — с первой строки. Каждый блок каждой политики — отдельный процесс;
# проверка сразу после; любой ненулевой код останавливает раннер. Повтор —
# только при «No CUDA GPUs are available».
set -euo pipefail
MODE="${1:?нужен режим: plan, smoke, run или analyze}"
DEV="${2:-cuda:1}"
SEED=101
BASIS="data/k15f/basis_s0.pt"
DTAG="$(echo "$DEV" | tr -d ':')"
IDENT="reports/k15f/identity_${DTAG}.json"
PLAN="reports/k16/plan.json"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p logs/k16 reports/k16
LOGF="logs/k16/${MODE}_s${SEED}.log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит" \
     "$(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
for m in k16_policy k16_harness k16_analyze k16_train k15f_policy \
         k9h_multiarm_gate; do
  python3 "experiments/${m}.py" --selftest >/dev/null \
    || { echo "ОТКАЗ: самопроверка ${m}"; finish 1; }
done
python3 experiments/k16_protocol.py selftest >/dev/null \
  || { echo "ОТКАЗ: самопроверка k16_protocol"; finish 1; }

if [ "$MODE" = plan ]; then
  CODE=0
  python3 experiments/k16_protocol.py plan --plan "$PLAN" --basis "$BASIS" \
    --identity "$IDENT" --device "$DEV" || CODE=$?
  finish $CODE
fi
[ -f "$PLAN" ] || { echo "ОТКАЗ: нет плана $PLAN"; finish 1; }
python3 experiments/k16_protocol.py check-plan --plan "$PLAN" \
  || { echo "ОТКАЗ: входы или код не совпадают с планом"; finish 1; }
sha12 () { sha1sum "$1" | cut -c1-12; }
OUTD="reports/k16/run/p$(sha12 "$PLAN")"
mkdir -p "$OUTD/quarantine"
echo "    план $PLAN, каталог $OUTD"

if [ "$MODE" = analyze ]; then
  CODE=0
  python3 experiments/k16_analyze.py --plan "$PLAN" --run-dir "$OUTD" \
    || CODE=$?
  finish $CODE
fi
BLK="$(python3 -c "import json; print(json.load(open('$PLAN'))['block'])")"
COMMON="--ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
  --task-suite 10 --n-envs 5 --seed $SEED --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 --waiting-steps 10 \
  --run-tag k16_s${SEED} --device $DEV --save-actions \
  --policy k15f --k15f-basis $BASIS"

valid () {
  [ -f "$1" ] && python3 experiments/k16_protocol.py check-block \
    --plan "$PLAN" --block "$1" "$2" "$3" >/dev/null 2>&1
}

run_one () {   # метка задача
  local L="$1" T="$2" try rc TMP
  local F="$OUTD/${L}_t${T}_i${BLK}.json"
  if valid "$F" "$L" "$T"; then echo "    $L t$T: уже есть"; return 0; fi
  if [ -f "$F" ]; then
    local Q="$OUTD/quarantine/$(date +%Y%m%dT%H%M%S)"; mkdir -p "$Q"
    mv "$F" "${F%.json}.actions.npz" "$Q/" 2>/dev/null || true
    echo "    $L t$T: повреждённый результат -> $Q"
  fi
  for try in 1 2 3; do
    echo "    $L t$T — попытка $try, $(date '+%H:%M:%S')"
    TMP="$(mktemp)"; rc=0
    # shellcheck disable=SC2086
    python3 experiments/k16_harness.py --k16-plan "$PLAN" $COMMON \
      --arm-label "$L" --task-id "$T" --init-starts "$BLK" \
      --out "$OUTD/${L}_t${T}_i{i0}.json" > "$TMP" 2>&1 || rc=$?
    cat "$TMP"
    if [ $rc -eq 0 ]; then
      rm -f "$TMP"
      if ! valid "$F" "$L" "$T"; then
        python3 experiments/k16_protocol.py check-block --plan "$PLAN" \
          --block "$F" "$L" "$T" || true
        echo "ОСТАНОВ: $L t$T — новый результат не прошёл проверку"
        return 3
      fi
      sleep 3; return 0
    fi
    if grep -q "No CUDA GPUs are available" "$TMP" && [ "$try" -lt 3 ]; then
      echo "    нет GPU — пауза 120 с и повтор"; rm -f "$TMP"; sleep 120
      continue
    fi
    rm -f "$TMP"; echo "ОСТАНОВ: $L t$T — код $rc"; return $rc
  done
}

LABELS="$(python3 -c "import json; print(' '.join(json.load(open('$PLAN'))['labels']))")"
if [ "$MODE" = smoke ]; then
  TASKS="$(python3 -c "import json; print(' '.join(map(str, json.load(open('$PLAN'))['smoke_tasks'])))")"
elif [ "$MODE" = run ]; then
  TASKS="$(python3 -c "import json; print(' '.join(map(str, json.load(open('$PLAN'))['tasks'])))")"
else
  echo "режим $MODE неизвестен"; finish 2
fi
for T in $TASKS; do
  echo "--- задача $T, $(date)"
  for L in $LABELS; do                    # q0 первым (порядок плана)
    run_one "$L" "$T" || finish $?
  done
done
echo "=== раскатки закончены $(date)"
CODE=0
if [ "$MODE" = smoke ]; then
  python3 experiments/k16_protocol.py smoke-check --plan "$PLAN" \
    --run-dir "$OUTD" || CODE=$?
else
  python3 experiments/k16_analyze.py --plan "$PLAN" --run-dir "$OUTD" \
    || CODE=$?
fi
finish $CODE
