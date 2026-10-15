#!/bin/bash
# K-16s: контроль направления ЗАПИСАННОГО вмешательства learned K-16.
#   plan     план (метки и сиды, эталоны K-16, код);
#   smoke    stored_replay и stored_rot_s3 на задаче 2 + проверка (replay
#            обязан совпасть с learned K-16; s3 входит в контроль);
#   run      stored_rot_s1..s3 на 10 задачах (30; готовые пропускаются),
#            затем анализ;
#   analyze  только анализ.
# Лог с первой строки; каждый блок — отдельный процесс; проверка сразу
# после; любой ненулевой код — останов; повтор только при потере GPU.
set -euo pipefail
MODE="${1:?нужен режим: plan, smoke, run или analyze}"
DEV="${2:-cuda:1}"
SEED=101
BASIS="data/k15f/basis_s0.pt"
PLAN="reports/k16s/plan.json"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p logs/k16s reports/k16s
LOGF="logs/k16s/${MODE}_s${SEED}.log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит" \
     "$(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
for m in k16s_policy k16s_harness k16s_analyze k16r_policy k16_policy k16_harness \
         k15f_policy k9h_multiarm_gate; do
  python3 "experiments/${m}.py" --selftest >/dev/null \
    || { echo "ОТКАЗ: самопроверка ${m}"; finish 1; }
done
python3 experiments/k16s_protocol.py selftest >/dev/null \
  || { echo "ОТКАЗ: самопроверка k16s_protocol"; finish 1; }

if [ "$MODE" = plan ]; then
  CODE=0
  python3 experiments/k16s_protocol.py plan --plan "$PLAN" --device "$DEV" \
    || CODE=$?
  finish $CODE
fi
[ -f "$PLAN" ] || { echo "ОТКАЗ: нет плана $PLAN"; finish 1; }
python3 experiments/k16s_protocol.py check-plan --plan "$PLAN" \
  || { echo "ОТКАЗ: входы или код не совпадают с планом"; finish 1; }
sha12 () { sha1sum "$1" | cut -c1-12; }
OUTD="reports/k16s/run/p$(sha12 "$PLAN")"
mkdir -p "$OUTD/quarantine"
echo "    план $PLAN, каталог $OUTD"
if [ "$MODE" = analyze ]; then
  CODE=0
  python3 experiments/k16s_analyze.py --plan "$PLAN" --run-dir "$OUTD" \
    || CODE=$?
  finish $CODE
fi
BLK="$(python3 -c "import json; print(json.load(open('$PLAN'))['block'])")"
COMMON="--ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
  --task-suite 10 --n-envs 5 --seed $SEED --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 --waiting-steps 10 \
  --run-tag k16s_s${SEED} --device $DEV --save-actions \
  --policy k15f --k15f-basis $BASIS"

valid () {
  [ -f "$1" ] && python3 experiments/k16s_protocol.py check-block \
    --plan "$PLAN" --block "$1" "$2" "$3" >/dev/null 2>&1
}

run_one () {
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
    python3 experiments/k16s_harness.py --k16s-plan "$PLAN" $COMMON \
      --arm-label "$L" --task-id "$T" --init-starts "$BLK" \
      --out "$OUTD/${L}_t${T}_i{i0}.json" > "$TMP" 2>&1 || rc=$?
    cat "$TMP"
    if [ $rc -eq 0 ]; then
      rm -f "$TMP"
      if ! valid "$F" "$L" "$T"; then
        python3 experiments/k16s_protocol.py check-block --plan "$PLAN" \
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

if [ "$MODE" = smoke ]; then
  PAIRS="$(python3 -c "import json; print(' '.join(f'{l}:{t}' for l, t in json.load(open('$PLAN'))['smoke']))")"
elif [ "$MODE" = run ]; then
  PAIRS="$(python3 -c "import json; p=json.load(open('$PLAN')); print(' '.join(f'{l}:{t}' for l in p['labels'] if l != 'stored_replay' for t in p['tasks']))")"
else
  echo "режим $MODE неизвестен"; finish 2
fi
for PT in $PAIRS; do
  run_one "${PT%%:*}" "${PT##*:}" || finish $?
done
echo "=== раскатки закончены $(date)"
CODE=0
if [ "$MODE" = smoke ]; then
  python3 experiments/k16s_protocol.py smoke-check --plan "$PLAN" \
    --run-dir "$OUTD" || CODE=$?
else
  python3 experiments/k16s_analyze.py --plan "$PLAN" --run-dir "$OUTD" \
    || CODE=$?
fi
finish $CODE
