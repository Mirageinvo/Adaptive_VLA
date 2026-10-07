#!/bin/bash
# K-15e M0: раскатки семейства a0 + α·δ на всех задачах LIBERO-10.
#
#   smoke  задачи 0 и 8, состояния 0-4: все руки, тождества a000 = q0 и
#          a100 = h18 K-15d (по хешам действий), диапазон, конечность;
#   probe  задачи 0-9, состояния 0-4 (50 кластеров), все руки — замер
#          оракула и зеркального контроля направления.
#
# Анализ (k15e_measure_scale.py) — fail-closed: тождества a000 = q0 и
# a100 = h18 обязаны выполняться на КАЖДОМ эпизоде, сид 101, один контракт
# h18 у всех α. Любое нарушение — код 3; в цепочке smoke && probe он
# останавливает probe.
#
# РУКИ: q0; a000 a025 a050 a075 a100 (масштабы поправки h18); am050 am100
# (контроль: та же поправка против направления). Каждая рука на каждой
# задаче — отдельный процесс харнесса (загрузка модели ~2-3 мин).
#
# Состояния 0-4 уже использовались — результат диагностический, решение о
# дорогом датасете, а не оценка метода. Финальный банк 25-49 не трогается.
# Один execution seed 101, 5 сред на блок — как в K-15d, чтобы хеши
# действий a100 можно было сверить с прежней рукой h18. Порядок рук
# циклически сдвигается по задаче. Git только читается.
#
#   setsid nohup bash experiments/k15e_scale_rollout.sh probe cuda:1 \
#     > logs/k15e/probe_$(date +%Y%m%dT%H%M%S).log 2>&1 &
set -euo pipefail
MODE="${1:?нужен режим: smoke или probe}"
DEV="${2:-cuda:1}"
SEED=101
case "$MODE" in
  smoke) TASKS="0 8" ;;
  probe) TASKS="0 1 2 3 4 5 6 7 8 9" ;;
  *) echo "режим $MODE неизвестен"; exit 2 ;;
esac
STATES="0"
ARMS="q0 a000 a025 a050 a075 a100 am050 am100"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p logs/k15e reports/k15e
DTAG="$(echo "$DEV" | tr -d ':')"
REP="reports/k15d/inference_h1p1_s0_${DTAG}.json"
for f in data/k15d/h1p1_s0.pt "$REP"; do
  [ -f "$f" ] || { echo "ОТКАЗ: нет $f"; exit 1; }
done
TRACKED="$(git ls-files reports/k15e logs/k15e 2>/dev/null | head -3)"
[ -z "$TRACKED" ] || { echo "ОТКАЗ: reports/k15e отслеживается git"; exit 1; }
for m in k15e_candidates k15e_policy k15e_measure_scale k15d_policy \
         k9h_multiarm_gate; do
  python3 "experiments/${m}.py" --selftest >/dev/null \
    || { echo "ОТКАЗ: самопроверка ${m}"; exit 1; }
done
sha12 () { sha1sum "$1" | cut -c1-12; }
TAG="k9h$(sha12 experiments/k9h_multiarm_gate.py)_c$(sha12 \
experiments/k15e_candidates.py)_p$(sha12 experiments/k15e_policy.py)_h18$(sha12 \
data/k15d/h1p1_s0.pt)$(sha12 "$REP")"
OUTD="reports/k15e/m0/${MODE}/s${SEED}/${TAG}"
LOGF="logs/k15e/m0_${MODE}_s${SEED}.log"
mkdir -p "$OUTD"
exec >> "$LOGF" 2>&1

echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, задачи $TASKS, руки $ARMS"
echo "    коммит $(git rev-parse --short HEAD 2>/dev/null); каталог $OUTD"

COMMON="--ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
  --task-suite 10 --n-envs 5 --seed $SEED --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 \
  --run-tag k15e_m0_${MODE}_s${SEED} --device $DEV --save-actions"
DRVQ="--policy depthrvq --policy-ckpt data/k9d_ep3.pt \
  --q1-ckpt data/k14c/q1_main_s0.pt --expect-q1-variant main \
  --expect-q1-seed 0 --expect-q0-manifest data/k14d/q0_b8_e0.manifest.json \
  --depth-rvq-mode fast"

alpha_of () {
  python3 -c "import sys; sys.path.insert(0,'experiments'); \
import k15e_candidates as k; print(k.label_alpha('$1'))"
}

run_arm () {   # метка, задача
  local L="$1" T="$2" rc=0 NEED=""
  for I0 in $STATES; do
    local F="$OUTD/${L}_t${T}_i${I0}.json"
    # ГОТОВ — только если JSON читается И его npz действий на месте с тем
    # же отпечатком (харнесс публикует npz раньше JSON).
    if [ -f "$F" ] && python3 - "$F" >/dev/null 2>&1 <<'PY'
import hashlib, json, os, sys
d = json.load(open(sys.argv[1]))
f = os.path.join(os.path.dirname(sys.argv[1]), d["actions_npz"])
h = hashlib.sha1(open(f, "rb").read()).hexdigest()[:12]
sys.exit(0 if h == d["actions_npz_sha1"] else 1)
PY
    then continue; fi
    rm -f "$F" "${F%.json}.actions.npz"
    NEED="$NEED${NEED:+,}$I0"
  done
  [ -z "$NEED" ] && { echo "    $L t$T: уже есть"; return 0; }
  echo "    $L t$T — $(date '+%H:%M:%S')"
  local -a EXTRA
  if [ "$L" = q0 ]; then
    # shellcheck disable=SC2206
    EXTRA=($DRVQ)
  else
    EXTRA=(--policy k15e --k15e-alpha "$(alpha_of "$L")")
  fi
  # shellcheck disable=SC2086
  python3 experiments/k9h_multiarm_gate.py $COMMON "${EXTRA[@]}" \
    --task-id "$T" --init-starts "$NEED" --arm-label "$L" \
    --out "$OUTD/${L}_t${T}_i{i0}.json" || rc=$?
  if [ $rc -ne 0 ]; then
    echo "ОСТАНОВ: $L, задача $T — код $rc"
    return $rc
  fi
  echo "    $L t$T готово: $(grep -h 'успех ' "$LOGF" | tail -1)"
  sleep 5
}

for T in $TASKS; do
  echo "--- задача $T, $(date), свободно $(free -g | awk 'NR==2{print $7}') ГБ"
  set -- $ARMS
  N=$#; SEQ=""
  for k in $(seq 0 $((N - 1))); do
    i=$(( (k + T + SEED) % N + 1 ))
    SEQ="$SEQ ${!i}"
  done
  echo "    порядок рук:$SEQ"
  for A in $SEQ; do run_arm "$A" "$T"; done
done

echo "=== раскатки закончены $(date) ==="
ARTS="$(ls "$OUTD"/*_t*_i0.json | tr '\n' ' ')"
# Прежняя рука h18 K-15d на тех же стартах (confirm: задачи 0-7, safety:
# 8-9) — для сверки a100 по хешам действий.
H18="$(ls reports/k15d/rollout/confirm_10/s101/*/h18_t*_i0.json \
          reports/k15d/rollout/safety_10/s101/*/h18_t*_i0.json 2>/dev/null \
       | tr '\n' ' ' || true)"
TASKS_CSV="$(echo "$TASKS" | tr ' ' ',')"
CODE=0
# shellcheck disable=SC2086
python3 experiments/k15e_measure_scale.py --arts $ARTS --h18-arts $H18 \
  --tasks "$TASKS_CSV" --states 0,1,2,3,4 \
  --out "reports/k15e/m0_${MODE}_s${SEED}_${TAG}.json" || CODE=$?
echo "=== КОНЕЦ $(date), анализ: код $CODE ==="
exit $CODE
