#!/bin/bash
# K-15g: локальная поправка — smoke, план, раскатки, проверка и анализ.
#
#   analyze-m1  анализ готовых раскаток K-15f M1 (без новых раскаток);
#   smoke       задачи 0 и 8, блок 0: q0ref, p2l0p, p2r0p, p4l0p, p4r0p
#               через адаптер k15g_harness.py; затем проверка протокола
#               (smoke-план, без вердикта M1);
#   plan        фиксация неизменяемого плана (нужен итоговый отчёт M1):
#               задачи 2, 8, 9, блоки из переписи M1, q0ref + 32 руки;
#   run         раскатки по плану: q0ref, затем 32 руки (2 момента × 16);
#   analyze     проверки протокола и агрегация.
#
# ВОЗОБНОВЛЕНИЕ — только для полностью проверенных пар JSON+npz.
# Повреждённый результат переносится в quarantine/ для разбора, не
# перезаписывается молча. ПОВТОР — только распознанного инфраструктурного
# сбоя (No CUDA GPUs are available), не более двух раз с паузой; любой
# другой отказ останавливает раннер.
#
# M1 K-15f этим скриптом не трогается: харнесс k9h_multiarm_gate.py не
# меняется, рука подключается адаптером.
set -euo pipefail
MODE="${1:?нужен режим: analyze-m1, smoke, plan, run или analyze}"
DEV="${2:-cuda:1}"
SEED=101
BASIS="data/k15f/basis_s0.pt"
DTAG="$(echo "$DEV" | tr -d ':')"
IDENT="reports/k15f/identity_${DTAG}.json"
CENSUS="reports/k15f/m1_census.json"
PLAN="reports/k15g/local_plan.json"
SPLAN="reports/k15g/smoke_plan.json"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p logs/k15g reports/k15g
TRACKED="$(git ls-files reports/k15g logs/k15g 2>/dev/null | head -3)"
[ -z "$TRACKED" ] || { echo "ОТКАЗ: reports/k15g отслеживается git"; exit 1; }
for m in k15g_analyze_m1 k15g_local_policy k15g_local_probe k15g_harness \
         k15f_policy k9h_multiarm_gate; do
  python3 "experiments/${m}.py" --selftest >/dev/null \
    || { echo "ОТКАЗ: самопроверка ${m}"; exit 1; }
done
M1DIR="$(ls -td reports/k15f/m1/s101/*/ 2>/dev/null | head -1)"

if [ "$MODE" = "analyze-m1" ]; then
  python3 experiments/k15g_analyze_m1.py --m1-dir "$M1DIR" \
    --census "$CENSUS" --basis "$BASIS" --out reports/k15g/m1_analysis.json
  exit $?
fi
if [ "$MODE" = "plan" ]; then
  M1REP="$(ls -t reports/k15f/m1_k9h*.json 2>/dev/null | head -1)"
  [ -n "$M1REP" ] || { echo "ОТКАЗ: нет итогового отчёта M1"; exit 1; }
  python3 experiments/k15g_local_probe.py --mode plan --plan "$PLAN" \
    --basis "$BASIS" --identity "$IDENT" --census "$CENSUS" \
    --m1-report "$M1REP" --census-q0-dir "$M1DIR" --device "$DEV"
  exit $?
fi

case "$MODE" in
  smoke)
    if [ ! -f "$SPLAN" ]; then
      python3 experiments/k15g_local_probe.py --mode plan --smoke \
        --plan "$SPLAN" --basis "$BASIS" --identity "$IDENT" \
        --census "$CENSUS" --census-q0-dir "$M1DIR" --device "$DEV"
    fi
    P="$SPLAN" ;;
  run|analyze) P="$PLAN" ;;
  *) echo "режим $MODE неизвестен"; exit 2 ;;
esac
[ -f "$P" ] || { echo "ОТКАЗ: нет плана $P (сначала plan)"; exit 1; }
sha12 () { sha1sum "$1" | cut -c1-12; }
TAG="p$(sha12 "$P")_k9h$(sha12 experiments/k9h_multiarm_gate.py)_a$(sha12 \
experiments/k15g_harness.py)_g$(sha12 experiments/k15g_local_policy.py)"
OUTD="reports/k15g/local/${MODE/analyze/run}/${TAG}"
LOGF="logs/k15g/local_${MODE}_s${SEED}.log"
mkdir -p "$OUTD" "$OUTD/quarantine"
exec >> "$LOGF" 2>&1
echo "=== СТАРТ $(date) === режим $MODE, план $P, каталог $OUTD"
echo "    коммит $(git rev-parse --short HEAD 2>/dev/null)"

if [ "$MODE" = "analyze" ]; then
  python3 experiments/k15g_local_probe.py --mode analyze --plan "$P" \
    --run-dir "$OUTD" --out reports/k15g/local_analysis.json
  exit $?
fi

# входы плана совпадают с файлами на диске — до первой раскатки
python3 - "$P" <<'PY' || { echo "ОТКАЗ: входы не совпадают с планом"; exit 1; }
import json, sys
sys.path.insert(0, "experiments")
import k15g_local_probe as pr
prob = pr.check_plan_inputs(json.load(open(sys.argv[1])))
print("\n".join(prob) if prob else "  входы плана сверены")
sys.exit(1 if prob else 0)
PY

COMMON="--ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
  --task-suite 10 --n-envs 5 --seed $SEED --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 \
  --run-tag k15g_local_s${SEED} --device $DEV --save-actions \
  --policy k15f --k15f-basis $BASIS"

block_state () {   # путь метка задача блок -> 0 цел, 1 нет, 2 повреждён
  # ГОТОВ — только если пара JSON+npz полностью согласована: отпечаток npz,
  # метка, задача, состояния блока, action_sha1/init_state_id/done_step
  # npz = JSON, сид 101 (k15g_local_probe.validate_block)
  [ -f "$1" ] || return 1
  python3 experiments/k15g_local_probe.py --validate-block "$1" "$2" "$3" \
    "$4" >/dev/null 2>&1 && return 0
  return 2
}

run_arm () {   # метка, задача, блоки
  local L="$1" T="$2" NEED="" st
  for I0 in $3; do
    local F="$OUTD/${L}_t${T}_i${I0}.json"
    st=0; block_state "$F" "$L" "$T" "$I0" || st=$?
    if [ "$st" -eq 0 ]; then continue; fi
    if [ "$st" -eq 2 ]; then
      local Q="$OUTD/quarantine/$(date +%Y%m%dT%H%M%S)"
      mkdir -p "$Q"
      mv "$F" "${F%.json}.actions.npz" "$Q/" 2>/dev/null || true
      echo "    $L t$T i$I0: повреждённый результат -> $Q"
    fi
    NEED="$NEED${NEED:+,}$I0"
  done
  [ -z "$NEED" ] && { echo "    $L t$T: уже есть"; return 0; }
  local try rc OUT
  for try in 1 2 3; do
    echo "    $L t$T блоки $NEED — попытка $try, $(date '+%H:%M:%S')"
    OUT="$(mktemp)"
    rc=0
    # shellcheck disable=SC2086
    python3 experiments/k15g_harness.py $COMMON --task-id "$T" \
      --init-starts "$NEED" --arm-label "$L" \
      --out "$OUTD/${L}_t${T}_i{i0}.json" > "$OUT" 2>&1 || rc=$?
    cat "$OUT"
    if [ $rc -eq 0 ]; then rm -f "$OUT"; sleep 5; return 0; fi
    if grep -q "No CUDA GPUs are available" "$OUT" && [ "$try" -lt 3 ]; then
      echo "    инфраструктурный сбой (нет GPU) — пауза 120 с и повтор"
      rm -f "$OUT"; sleep 120; continue
    fi
    rm -f "$OUT"
    echo "ОСТАНОВ: $L, задача $T — код $rc"
    return $rc
  done
}

LABELS="$(python3 -c "import json; print(' '.join(json.load(open('$P'))['candidates']))")"
TASKS="$(python3 -c "import json; print(' '.join(map(str, json.load(open('$P'))['tasks'])))")"
# КАЖДЫЙ БЛОК — В ОТДЕЛЬНОМ ПРОЦЕССЕ. В K-15f M1 на задаче 8 reset между
# блоками одного процесса восстанавливал среду не полностью: старт блока 5
# зависел от того, что рука сделала в блоке 0 (14 из 16 рук разошлись с
# переписью; одиночный прогон совпал). Поэтому блоки не объединяются.
for T in $TASKS; do
  BLK="$(python3 -c "import json; print(' '.join(map(str, json.load(open('$P'))['blocks']['$T'])))")"
  echo "--- задача $T, блоки $BLK, $(date)"
  for B in $BLK; do
    run_arm q0ref "$T" "$B"                  # эталон — первым
    for L in $LABELS; do
      [ "$L" = q0ref ] && continue
      run_arm "$L" "$T" "$B"
    done
  done
done
echo "=== раскатки закончены $(date)"
CODE=0
python3 experiments/k15g_local_probe.py --mode analyze --plan "$P" \
  --run-dir "$OUTD" \
  --out "reports/k15g/local_${MODE}_analysis.json" || CODE=$?
echo "=== КОНЕЦ $(date), анализ: код $CODE ==="
exit $CODE
