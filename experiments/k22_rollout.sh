#!/bin/bash
# K-22: оценка parents24 на стартах 30–44 (блоки 30, 35, 40) — 30 процессов;
# fullbar, flat24, tree_adapt24 — из закреплённого плана A K-21.
# Режимы: selftest, plan, run (smoke — задачи 0 и 8 блока 30 — или его
# перепроверка, затем при техническом успехе остальные процессы и анализ),
# smoke, analyze.
# Лог с первой строки; каждый блок — отдельный холодный процесс; проверка
# сразу после; любой ненулевой код — останов; повтор только при потере GPU.
# Карта по умолчанию cuda:0 (cuda:1 занята другим пользователем).
set -euo pipefail
MODE="${1:?нужен режим: selftest, plan, smoke, run или analyze}"
DEV="${2:-cuda:0}"
SEED=101
BASIS="data/k15f/basis_s0.pt"
DTAG="$(echo "$DEV" | tr -d ':')"
PLAN="reports/k22/plan_main.json"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1        # снимок BAR — только из локального кэша (закреплённая ревизия)
mkdir -p logs/k22 reports/k22 reports/k22/run
LOGF="logs/k22/main_${MODE}_s${SEED}.log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит" \
     "$(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
for m in k22_policy k22_model k22_train k22_analyze k21_main_policy k21_main_analyze k16_policy k16_harness k9h_multiarm_gate; do
  python3 "experiments/${m}.py" --selftest >/dev/null \
    || { echo "ОТКАЗ: самопроверка ${m}"; finish 1; }
done
python3 experiments/k22_harness.py --selftest >/dev/null \
  || { echo "ОТКАЗ: самопроверка k22_harness"; finish 1; }
python3 experiments/k22_protocol.py selftest >/dev/null \
  || { echo "ОТКАЗ: самопроверка k22_protocol"; finish 1; }
if [ "$MODE" = selftest ]; then
  echo "    все самопроверки пройдены"; finish 0
fi

if [ "$MODE" = plan ]; then
  CODE=0
  python3 experiments/k22_protocol.py plan \
    --device "$DEV" || CODE=$?
  finish $CODE
fi
[ -f "$PLAN" ] || { echo "ОТКАЗ: нет плана $PLAN"; finish 1; }
python3 experiments/k22_protocol.py check-plan \
  || { echo "ОТКАЗ: входы или код не совпадают с планом"; finish 1; }
sha12 () { sha1sum "$1" | cut -c1-12; }
OUTD="reports/k22/run/p$(sha12 "$PLAN")"
mkdir -p "$OUTD/quarantine"
echo "    план $PLAN, каталог $OUTD"
if [ "$MODE" = analyze ]; then
  CODE=0
  python3 experiments/k22_analyze.py || CODE=$?
  finish $CODE
fi
CKPT="$(python3 -c "import json; print(json.load(open('$PLAN'))['setup']['ckpt'])")"
[ -d "$CKPT" ] || { echo "ОТКАЗ: нет закреплённого снимка BAR $CKPT"; finish 1; }
echo "    снимок BAR: $CKPT"
COMMON="--ckpt $CKPT \
  --task-suite 10 --n-envs 5 --seed $SEED --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 --waiting-steps 10 \
  --run-tag k22_s${SEED} --device $DEV --save-actions"
polargs () {   # метка -> аргументы политики (только parents24)
  echo "--policy k15f --k15f-basis $BASIS"
}

valid () {
  [ -f "$1" ] && python3 experiments/k22_protocol.py check-block --block "$1" "$2" "$3" "$4" >/dev/null 2>&1
}

run_one () {   # метка задача блок
  local L="$1" T="$2" B="$3" try rc TMP
  local F="$OUTD/${L}_t${T}_i${B}.json"
  if valid "$F" "$L" "$T" "$B"; then echo "    $L t$T i$B: уже есть"; return 0; fi
  if [ -f "$F" ]; then
    # существующий блок не прошёл проверку: останов с причиной (без
    # автоматического пересчёта); блок переносится в карантин для разбора
    python3 experiments/k22_protocol.py check-block \
      --block "$F" "$L" "$T" "$B" || true
    local Q="$OUTD/quarantine/$(date +%Y%m%dT%H%M%S)"; mkdir -p "$Q"
    mv "$F" "${F%.json}.actions.npz" "$Q/" 2>/dev/null || true
    echo "ОСТАНОВ: $L t$T i$B — существующий блок не прошёл проверку -> $Q"
    return 3
  fi
  for try in 1 2 3; do
    echo "    $L t$T i$B — попытка $try, $(date '+%H:%M:%S')"
    TMP="$(mktemp)"; rc=0
    # shellcheck disable=SC2086
    python3 experiments/k22_harness.py --k22-plan "$PLAN" $COMMON \
      $(polargs "$L") \
      --arm-label "$L" --task-id "$T" --init-starts "$B" \
      --out "$OUTD/${L}_t${T}_i{i0}.json" > "$TMP" 2>&1 || rc=$?
    cat "$TMP"
    if [ $rc -eq 0 ]; then
      rm -f "$TMP"
      if ! valid "$F" "$L" "$T" "$B"; then
        python3 experiments/k22_protocol.py check-block \
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
SMOKE_PAIRS="$(python3 -c "import json; print(' '.join(f'{t}:{b}' for t, b in json.load(open('$PLAN'))['smoke']))")"
ALL_PAIRS="$(python3 -c "import json; p=json.load(open('$PLAN')); print(' '.join(f'{t}:{b}' for t in p['tasks'] for b in p['blocks']))")"
run_pairs () {
  for PT in $1; do
    T="${PT%%:*}"; B="${PT##*:}"
    echo "--- задача $T, блок $B, $(date)"
    for L in $LABELS; do
      run_one "$L" "$T" "$B" || finish $?
    done
  done
}
case "$MODE" in
  smoke|run) ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
# smoke (или перепроверка готовых блоков smoke) — всегда первым
run_pairs "$SMOKE_PAIRS"
CODE=0
python3 experiments/k22_protocol.py smoke-check \
  --run-dir "$OUTD" || CODE=$?
if [ "$MODE" = smoke ] || [ $CODE -ne 0 ]; then
  [ $CODE -ne 0 ] && echo "ОСТАНОВ: smoke не прошёл технические проверки"
  finish $CODE
fi
# технический успех smoke -> остальные блоки (продолжение не зависит от SR)
run_pairs "$ALL_PAIRS"
echo "=== раскатки закончены $(date)"
python3 experiments/k22_analyze.py || CODE=$?
finish $CODE
