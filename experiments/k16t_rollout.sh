#!/bin/bash
# K-16t: подтверждение K-16s на задаче 8, блоки 30 и 35 (8 процессов).
#   plan     план (q0 K-16b, модели K-16, сиды, код);
#   run      learned (2) -> таблица эталонов -> повороты (6) -> анализ;
#            готовые проверенные блоки пропускаются;
#   analyze  только анализ.
# Лог с первой строки; каждый блок — отдельный процесс; проверка сразу
# после; любой ненулевой код — останов; повтор только при потере GPU.
set -euo pipefail
MODE="${1:?нужен режим: plan, run или analyze}"
DEV="${2:-cuda:1}"
SEED=101
BASIS="data/k15f/basis_s0.pt"
PLAN="reports/k16t/plan.json"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p logs/k16t reports/k16t
LOGF="logs/k16t/${MODE}_s${SEED}.log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит" \
     "$(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
for m in k16s_policy k16r_policy k16_policy k16_harness k16t_harness \
         k15f_policy k9h_multiarm_gate; do
  python3 "experiments/${m}.py" --selftest >/dev/null \
    || { echo "ОТКАЗ: самопроверка ${m}"; finish 1; }
done
python3 experiments/k16t_protocol.py selftest >/dev/null \
  || { echo "ОТКАЗ: самопроверка k16t_protocol"; finish 1; }
if [ "$MODE" = plan ]; then
  CODE=0
  python3 experiments/k16t_protocol.py plan --plan "$PLAN" --device "$DEV" \
    || CODE=$?
  finish $CODE
fi
[ -f "$PLAN" ] || { echo "ОТКАЗ: нет плана $PLAN"; finish 1; }
python3 experiments/k16t_protocol.py check-plan --plan "$PLAN" \
  || { echo "ОТКАЗ: входы или код не совпадают с планом"; finish 1; }
sha12 () { sha1sum "$1" | cut -c1-12; }
OUTD="reports/k16t/run/p$(sha12 "$PLAN")"
mkdir -p "$OUTD/quarantine"
echo "    план $PLAN, каталог $OUTD"
analyze () {
  python3 experiments/k16t_protocol.py analyze --plan "$PLAN" \
    --run-dir "$OUTD"
}
if [ "$MODE" = analyze ]; then CODE=0; analyze || CODE=$?; finish $CODE; fi
[ "$MODE" = run ] || { echo "режим $MODE неизвестен"; finish 2; }
COMMON="--ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
  --task-suite 10 --n-envs 5 --seed $SEED --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 --waiting-steps 10 \
  --run-tag k16t_s${SEED} --device $DEV --save-actions \
  --policy k15f --k15f-basis $BASIS"
T="$(python3 -c "import json; print(json.load(open('$PLAN'))['task'])")"
LEARNED="$(python3 -c "import json; print(json.load(open('$PLAN'))['learned'])")"
ROTS="$(python3 -c "import json; print(' '.join(json.load(open('$PLAN'))['rot']))")"
BLKS="$(python3 -c "import json; print(' '.join(map(str, json.load(open('$PLAN'))['blocks'])))")"

valid () {   # путь метка блок
  [ -f "$1" ] || return 1
  if [ "$2" = "$LEARNED" ]; then
    python3 experiments/k16t_protocol.py check-learned --plan "$PLAN" \
      --block "$1" "$2" "$3" >/dev/null 2>&1
  else
    python3 experiments/k16t_protocol.py check-rot --plan "$PLAN" \
      --run-dir "$OUTD" --block "$1" "$2" "$3" >/dev/null 2>&1
  fi
}

run_one () {   # метка блок
  local L="$1" B="$2" try rc TMP
  local F="$OUTD/${L}_t${T}_i${B}.json"
  if valid "$F" "$L" "$B"; then echo "    $L i$B: уже есть"; return 0; fi
  if [ -f "$F" ]; then
    local Q="$OUTD/quarantine/$(date +%Y%m%dT%H%M%S)"; mkdir -p "$Q"
    mv "$F" "${F%.json}.actions.npz" "$Q/" 2>/dev/null || true
    echo "    $L i$B: повреждённый результат -> $Q"
  fi
  for try in 1 2 3; do
    echo "    $L i$B — попытка $try, $(date '+%H:%M:%S')"
    TMP="$(mktemp)"; rc=0
    # shellcheck disable=SC2086
    python3 experiments/k16t_harness.py --k16t-plan "$PLAN" $COMMON \
      --arm-label "$L" --task-id "$T" --init-starts "$B" \
      --out "$OUTD/${L}_t${T}_i{i0}.json" > "$TMP" 2>&1 || rc=$?
    cat "$TMP"
    if [ $rc -eq 0 ]; then
      rm -f "$TMP"
      if ! valid "$F" "$L" "$B"; then
        echo "ОСТАНОВ: $L i$B — новый результат не прошёл проверку"
        return 3
      fi
      sleep 3; return 0
    fi
    if grep -q "No CUDA GPUs are available" "$TMP" && [ "$try" -lt 3 ]; then
      echo "    нет GPU — пауза 120 с и повтор"; rm -f "$TMP"; sleep 120
      continue
    fi
    rm -f "$TMP"; echo "ОСТАНОВ: $L i$B — код $rc"; return $rc
  done
}

for B in $BLKS; do run_one "$LEARNED" "$B" || finish $?; done
if [ ! -f reports/k16t/reference.npz ]; then
  python3 experiments/k16t_protocol.py reference --plan "$PLAN" \
    --run-dir "$OUTD" || finish 3
fi
for L in $ROTS; do
  for B in $BLKS; do run_one "$L" "$B" || finish $?; done
done
echo "=== раскатки закончены $(date)"
CODE=0; analyze || CODE=$?
finish $CODE
