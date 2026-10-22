#!/bin/bash
# K-21: оценка fullbar, flat24, tree_adapt24 на LIBERO spatial, object, goal
# (без переобучения). Режимы: selftest, plan [блоки...], smoke, run
# (smoke или его перепроверка, затем при техническом успехе остальное и
# анализ), analyze. Каждый блок — отдельный холодный процесс; проверка
# сразу после; любой ненулевой код — останов; повтор только при потере GPU.
#   bash experiments/k21_suite_rollout.sh plan cuda:0        # блок 0
#   bash experiments/k21_suite_rollout.sh plan cuda:0 0 5 10 # три блока
set -euo pipefail
MODE="${1:?нужен режим: selftest, plan, smoke, run или analyze}"
DEV="${2:-cuda:0}"
shift 2 || true
SEED=101
BASIS="data/k15f/basis_s0.pt"
PLAN="reports/k21/plan_suites.json"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
mkdir -p logs/k21 reports/k21/run
LOGF="logs/k21/suites_${MODE}_s${SEED}.log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит" \
     "$(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
for m in k21_main_policy k16_policy k16_harness k9h_multiarm_gate; do
  python3 "experiments/${m}.py" --selftest >/dev/null \
    || { echo "ОТКАЗ: самопроверка ${m}"; finish 1; }
done
python3 experiments/k21_suite_harness.py --selftest >/dev/null \
  || { echo "ОТКАЗ: самопроверка k21_suite_harness"; finish 1; }
python3 experiments/k21_suite_protocol.py selftest >/dev/null \
  || { echo "ОТКАЗ: самопроверка k21_suite_protocol"; finish 1; }
python3 experiments/k21_suite_analyze.py --selftest >/dev/null \
  || { echo "ОТКАЗ: самопроверка k21_suite_analyze"; finish 1; }
if [ "$MODE" = selftest ]; then echo "    все самопроверки пройдены"; finish 0; fi
if [ "$MODE" = plan ]; then
  CODE=0
  BL="${*:-0}"
  # shellcheck disable=SC2086
  python3 experiments/k21_suite_protocol.py plan --plan "$PLAN" \
    --device "$DEV" --blocks $BL || CODE=$?
  finish $CODE
fi
[ -f "$PLAN" ] || { echo "ОТКАЗ: нет плана $PLAN"; finish 1; }
python3 experiments/k21_suite_protocol.py check-plan --plan "$PLAN" \
  || { echo "ОТКАЗ: входы или код не совпадают с планом"; finish 1; }
sha12 () { sha1sum "$1" | cut -c1-12; }
OUTD="reports/k21/run/p$(sha12 "$PLAN")"
mkdir -p "$OUTD/quarantine"
echo "    план $PLAN, каталог $OUTD"
if [ "$MODE" = analyze ]; then
  CODE=0
  python3 experiments/k21_suite_analyze.py --plan "$PLAN" --run-dir "$OUTD" \
    || CODE=$?
  finish $CODE
fi
CKPT="$(python3 -c "import json; print(json.load(open('$PLAN'))['snapshot_path'])")"
[ -d "$CKPT" ] || { echo "ОТКАЗ: нет закреплённого снимка BAR $CKPT"; finish 1; }
echo "    снимок BAR: $CKPT"
COMMON="--ckpt $CKPT --n-envs 5 --seed $SEED --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 --waiting-steps 10 \
  --run-tag k21suites_s${SEED} --device $DEV --save-actions"
polargs () {
  if [ "$1" = fullbar ]; then echo "--policy fullbar"
  else echo "--policy k15f --k15f-basis $BASIS"; fi
}
valid () {   # json метка набор задача блок
  [ -f "$1" ] && python3 experiments/k21_suite_protocol.py check-block \
    --plan "$PLAN" --block "$1" "$2" "$3" "$4" "$5" >/dev/null 2>&1
}
run_one () {   # метка набор задача блок
  local L="$1" S="$2" T="$3" B="$4" try rc TMP
  local F="$OUTD/${L}_${S}_t${T}_i${B}.json"
  if valid "$F" "$L" "$S" "$T" "$B"; then echo "    $L $S t$T i$B: уже есть"; return 0; fi
  if [ -f "$F" ]; then
    python3 experiments/k21_suite_protocol.py check-block --plan "$PLAN" \
      --block "$F" "$L" "$S" "$T" "$B" || true
    local Q="$OUTD/quarantine/$(date +%Y%m%dT%H%M%S)"; mkdir -p "$Q"
    mv "$F" "${F%.json}.actions.npz" "$Q/" 2>/dev/null || true
    echo "ОСТАНОВ: $L $S t$T i$B — существующий блок не прошёл проверку -> $Q"
    return 3
  fi
  for try in 1 2 3; do
    echo "    $L $S t$T i$B — попытка $try, $(date '+%H:%M:%S')"
    TMP="$(mktemp)"; rc=0
    # shellcheck disable=SC2086
    python3 experiments/k21_suite_harness.py --k21-suite-plan "$PLAN" \
      $COMMON $(polargs "$L") --task-suite "$S" --arm-label "$L" \
      --task-id "$T" --init-starts "$B" \
      --out "$OUTD/${L}_${S}_t${T}_i{i0}.json" > "$TMP" 2>&1 || rc=$?
    cat "$TMP"
    if [ $rc -eq 0 ]; then
      rm -f "$TMP"
      if ! valid "$F" "$L" "$S" "$T" "$B"; then
        python3 experiments/k21_suite_protocol.py check-block --plan "$PLAN" \
          --block "$F" "$L" "$S" "$T" "$B" || true
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
LABELS="$(python3 -c "import json; print(' '.join(json.load(open('$PLAN'))['labels']))")"
SMOKE="$(python3 -c "import json; print(' '.join(f'{s}:{t}:{b}' for s, t, b in json.load(open('$PLAN'))['smoke']))")"
ALL="$(python3 -c "import json; p=json.load(open('$PLAN')); print(' '.join(f'{s}:{t}:{b}' for s in p['suites'] for b in p['blocks'] for t in p['tasks']))")"
run_triples () {
  for X in $1; do
    S="${X%%:*}"; R="${X#*:}"; T="${R%%:*}"; B="${R##*:}"
    echo "--- набор $S, задача $T, блок $B, $(date)"
    for L in $LABELS; do run_one "$L" "$S" "$T" "$B" || finish $?; done
  done
}
case "$MODE" in smoke|run) ;; *) echo "режим $MODE неизвестен"; finish 2 ;; esac
run_triples "$SMOKE"
CODE=0
python3 experiments/k21_suite_protocol.py smoke-check --plan "$PLAN" \
  --run-dir "$OUTD" || CODE=$?
if [ "$MODE" = smoke ] || [ $CODE -ne 0 ]; then
  [ $CODE -ne 0 ] && echo "ОСТАНОВ: smoke не прошёл технические проверки"
  finish $CODE
fi
run_triples "$ALL"
echo "=== раскатки закончены $(date)"
python3 experiments/k21_suite_analyze.py --plan "$PLAN" --run-dir "$OUTD" \
  || CODE=$?
finish $CODE
