#!/bin/bash
# K-15h: независимая проверка на свежем банке стартов.
#
#   inventory  использованные (задача, состояние) и выбор свежего банка
#              (до любых раскаток на нём);
#   smoke      старые блоки задач 0 и 8 (блок 0), пять политик, 10
#              процессов; проверка цепочки и эталонов (q0 M1-cold, p4r3p
#              K-15g); результаты не используются для подбора;
#   plan       окончательный план (после чистого smoke);
#   run        раскатки по плану: в каждом блоке q0ref, затем четыре
#              вмешательства; КАЖДЫЙ блок каждой политики — отдельный
#              процесс; проверка сразу после прогона; затем анализ;
#   analyze    только анализ.
#
# ЛОГ — с первой строки (tee), включая отказы самопроверок. Возобновление —
# только по полностью проверенному блоку; повреждённый переносится в
# quarantine/. Повтор — только при «No CUDA GPUs are available».
set -euo pipefail
MODE="${1:?нужен режим: inventory, smoke, plan, run или analyze}"
DEV="${2:-cuda:1}"
SEED=101
BASIS="data/k15f/basis_s0.pt"
DTAG="$(echo "$DEV" | tr -d ':')"
IDENT="reports/k15f/identity_${DTAG}.json"
PLAN="reports/k15h/plan.json"
SPLAN="reports/k15h/smoke_plan.json"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p logs/k15h reports/k15h
LOGF="logs/k15h/${MODE}_s${SEED}.log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит" \
     "$(git rev-parse --short HEAD 2>/dev/null)"
TRACKED="$(git ls-files reports/k15h logs/k15h 2>/dev/null | head -3)"
[ -z "$TRACKED" ] || { echo "ОТКАЗ: reports/k15h отслеживается git"; exit 1; }
for m in k15h_policy k15h_harness k15h_analyze k15h_static_basis \
         k15f_policy k9h_multiarm_gate; do
  python3 "experiments/${m}.py" --selftest >/dev/null \
    || { echo "ОТКАЗ: самопроверка ${m}"; exit 1; }
done
python3 experiments/k15h_protocol.py selftest >/dev/null \
  || { echo "ОТКАЗ: самопроверка k15h_protocol"; exit 1; }

finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }

case "$MODE" in
  inventory)
    CODE=0; python3 experiments/k15h_protocol.py inventory || CODE=$?
    finish $CODE ;;
  plan)
    CODE=0
    python3 experiments/k15h_protocol.py plan --plan "$PLAN" \
      --basis "$BASIS" --identity "$IDENT" --device "$DEV" || CODE=$?
    finish $CODE ;;
  smoke)
    if [ ! -f "$SPLAN" ]; then
      python3 experiments/k15h_protocol.py smoke-plan --plan "$SPLAN" \
        --basis "$BASIS" --identity "$IDENT" --device "$DEV" \
        || finish 1
    fi
    P="$SPLAN"; SUB=smoke ;;
  run|analyze) P="$PLAN"; SUB=run ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
[ -f "$P" ] || { echo "ОТКАЗ: нет плана $P"; finish 1; }
python3 experiments/k15h_protocol.py check-plan --plan "$P" \
  || { echo "ОТКАЗ: входы или код не совпадают с планом"; finish 1; }
sha12 () { sha1sum "$1" | cut -c1-12; }
TAG="p$(sha12 "$P")_h$(sha12 experiments/k15h_policy.py)_a$(sha12 \
experiments/k15h_harness.py)_k9h$(sha12 experiments/k9h_multiarm_gate.py)"
OUTD="reports/k15h/${SUB}/${TAG}"
mkdir -p "$OUTD/quarantine"
echo "    план $P, каталог $OUTD"

if [ "$MODE" = analyze ]; then
  CODE=0
  python3 experiments/k15h_analyze.py --plan "$P" --run-dir "$OUTD" || CODE=$?
  finish $CODE
fi

COMMON="--ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
  --task-suite 10 --n-envs 5 --seed $SEED --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 --waiting-steps 10 \
  --run-tag k15h_${SUB}_s${SEED} --device $DEV --save-actions \
  --policy k15f --k15f-basis $BASIS"

valid () {   # путь метка задача блок -> 0 цел
  [ -f "$1" ] && python3 experiments/k15h_protocol.py check-block \
    --plan "$P" --block "$1" "$2" "$3" "$4" >/dev/null 2>&1
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
    python3 experiments/k15h_harness.py --k15h-plan "$P" $COMMON \
      --arm-label "$L" --task-id "$T" --init-starts "$B" \
      --out "$OUTD/${L}_t${T}_i{i0}.json" > "$TMP" 2>&1 || rc=$?
    cat "$TMP"
    if [ $rc -eq 0 ]; then
      rm -f "$TMP"
      if ! valid "$F" "$L" "$T" "$B"; then
        python3 experiments/k15h_protocol.py check-block --plan "$P" \
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

LABELS="$(python3 -c "import json; print(' '.join(json.load(open('$P'))['labels']))")"
TASKS="$(python3 -c "import json; print(' '.join(json.load(open('$P'))['blocks']))")"
for T in $TASKS; do
  BLK="$(python3 -c "import json; print(' '.join(map(str, json.load(open('$P'))['blocks']['$T'])))")"
  echo "--- задача $T, блоки $BLK, $(date)"
  for B in $BLK; do
    for L in $LABELS; do                 # q0ref первым (порядок плана)
      run_one "$L" "$T" "$B" || finish $?
    done
  done
done
echo "=== раскатки закончены $(date)"
CODE=0
if [ "$MODE" = smoke ]; then
  python3 experiments/k15h_protocol.py smoke-check --plan "$P" \
    --run-dir "$OUTD" || CODE=$?
else
  python3 experiments/k15h_analyze.py --plan "$P" --run-dir "$OUTD" || CODE=$?
fi
finish $CODE
