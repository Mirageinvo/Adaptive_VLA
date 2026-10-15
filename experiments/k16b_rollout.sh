#!/bin/bash
# K-16b: ctx против ctx_goal — plan, smoke (задачи 0 и 8, блок 30),
# run (блоки 30 и 35 всех задач, 60 процессов), analyze.
# Обучение — отдельно: python3 experiments/k16b_train.py --device cuda:1.
# Лог с первой строки; каждый блок — отдельный процесс; проверка сразу
# после; любой ненулевой код — останов; повтор только при потере GPU.
set -euo pipefail
MODE="${1:?нужен режим: plan, smoke, run или analyze}"
DEV="${2:-cuda:1}"
SEED=101
BASIS="data/k15f/basis_s0.pt"
DTAG="$(echo "$DEV" | tr -d ':')"
IDENT="reports/k15f/identity_${DTAG}.json"
PLAN="reports/k16b/plan.json"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p logs/k16b reports/k16b
LOGF="logs/k16b/${MODE}_s${SEED}.log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит" \
     "$(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
for m in k16b_policy k16b_harness k16b_analyze k16b_train k16_policy \
         k16_harness k15f_policy k9h_multiarm_gate; do
  python3 "experiments/${m}.py" --selftest >/dev/null \
    || { echo "ОТКАЗ: самопроверка ${m}"; finish 1; }
done
python3 experiments/k16b_protocol.py selftest >/dev/null \
  || { echo "ОТКАЗ: самопроверка k16b_protocol"; finish 1; }

if [ "$MODE" = plan ]; then
  CODE=0
  python3 experiments/k16b_protocol.py plan --plan "$PLAN" --basis "$BASIS" \
    --identity "$IDENT" --device "$DEV" || CODE=$?
  finish $CODE
fi
[ -f "$PLAN" ] || { echo "ОТКАЗ: нет плана $PLAN"; finish 1; }
python3 experiments/k16b_protocol.py check-plan --plan "$PLAN" \
  || { echo "ОТКАЗ: входы или код не совпадают с планом"; finish 1; }
sha12 () { sha1sum "$1" | cut -c1-12; }
OUTD="reports/k16b/run/p$(sha12 "$PLAN")"
mkdir -p "$OUTD/quarantine"
echo "    план $PLAN, каталог $OUTD"
if [ "$MODE" = analyze ]; then
  CODE=0
  python3 experiments/k16b_analyze.py --plan "$PLAN" --run-dir "$OUTD" \
    || CODE=$?
  finish $CODE
fi
COMMON="--ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
  --task-suite 10 --n-envs 5 --seed $SEED --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 --waiting-steps 10 \
  --run-tag k16b_s${SEED} --device $DEV --save-actions \
  --policy k15f --k15f-basis $BASIS"

valid () {
  [ -f "$1" ] && python3 experiments/k16b_protocol.py check-block \
    --plan "$PLAN" --block "$1" "$2" "$3" "$4" >/dev/null 2>&1
}

run_one () {   # метка задача блок
  local L="$1" T="$2" B="$3" try rc TMP
  local F="$OUTD/${L}_t${T}_i${B}.json"
  if valid "$F" "$L" "$T" "$B"; then echo "    $L t$T i$B: уже есть"; return 0; fi
  if [ -f "$F" ]; then
    local Q="$OUTD/quarantine/$(date +%Y%m%dT%H%M%S)"; mkdir -p "$Q"
    mv "$F" "${F%.json}.actions.npz" "$Q/" 2>/dev/null || true
    echo "    $L t$T i$B: повреждённый результат -> $Q"
  fi
  for try in 1 2 3; do
    echo "    $L t$T i$B — попытка $try, $(date '+%H:%M:%S')"
    TMP="$(mktemp)"; rc=0
    # shellcheck disable=SC2086
    python3 experiments/k16b_harness.py --k16b-plan "$PLAN" $COMMON \
      --arm-label "$L" --task-id "$T" --init-starts "$B" \
      --out "$OUTD/${L}_t${T}_i{i0}.json" > "$TMP" 2>&1 || rc=$?
    cat "$TMP"
    if [ $rc -eq 0 ]; then
      rm -f "$TMP"
      if ! valid "$F" "$L" "$T" "$B"; then
        python3 experiments/k16b_protocol.py check-block --plan "$PLAN" \
          --block "$F" "$L" "$T" "$B" || true
        echo "ОСТАНОВ: $L t$T i$B — новый результат не прошёл проверку"
        return 3
      fi
      sleep 3; return 0
    fi
    if grep -q "No CUDA GPUs are available" "$TMP" && [ "$try" -lt 3 ]; then
      echo "    нет GPU — пауза 120 с и повтор"; rm -f "$TMP"; sleep 120
      continue
    fi
    rm -f "$TMP"; echo "ОСТАНОВ: $L t$T i$B — код $rc"; return $rc
  done
}

LABELS="$(python3 -c "import json; print(' '.join(json.load(open('$PLAN'))['labels']))")"
if [ "$MODE" = smoke ]; then
  PAIRS="$(python3 -c "import json; print(' '.join(f'{t}:{b}' for t, b in json.load(open('$PLAN'))['smoke']))")"
elif [ "$MODE" = run ]; then
  PAIRS="$(python3 -c "import json; p=json.load(open('$PLAN')); print(' '.join(f'{t}:{b}' for t in p['tasks'] for b in p['blocks']))")"
else
  echo "режим $MODE неизвестен"; finish 2
fi
for PT in $PAIRS; do
  T="${PT%%:*}"; B="${PT##*:}"
  echo "--- задача $T, блок $B, $(date)"
  for L in $LABELS; do                 # q0 первым
    run_one "$L" "$T" "$B" || finish $?
  done
done
echo "=== раскатки закончены $(date)"
CODE=0
if [ "$MODE" = smoke ]; then
  python3 experiments/k16b_protocol.py smoke-check --plan "$PLAN" \
    --run-dir "$OUTD" || CODE=$?
else
  python3 experiments/k16b_analyze.py --plan "$PLAN" --run-dir "$OUTD" \
    || CODE=$?
fi
finish $CODE
