#!/bin/bash
# K-15f этап A: подготовка базиса и поведенческий замер M1.
#
# ПОДГОТОВКА (без раскаток):
#   prep      smoke-кэш h18 -> overfit и smoke базиса -> полный кэш h18
#             (~1.5-2 ч на карте гейта K-15a) -> полное предобучение базиса
#             (по кэшу, без модели) -> гейт тождества на настоящей модели.
#             Отчёт об амплитуде печатается предобучением.
# РАСКАТКИ (LIBERO-10, сид 101, 5 сред на блок):
#   smoke     задачи 0 и 8, состояния 0-4: q0, z, l0p, r0p; тождество z = q0
#             обязательно;
#   census    перепись q0 на design-состояниях 0-9 всех задач; ФИКСИРУЕТ
#             провалы, диагностические успехи и блоки (reports/k15f/
#             m1_census.json) ДО раскаток кандидатов;
#   m1        16 рук (l/r × 4 × ±) на блоках из переписи и анализ по
#             правилу M1 (k15f_measure_subspace.RULE);
#   rollouts  smoke && census && m1.
#
# Состояния 10-24 (cross-fit) и 25-49 (final) не используются.
# CUDA_VISIBLE_DEVICES не выставляется. Git только читается.
#
#   setsid nohup bash experiments/k15f_rollout.sh prep cuda:1 \
#     > logs/k15f/prep_$(date +%Y%m%dT%H%M%S).log 2>&1 &
set -euo pipefail
MODE="${1:?нужен режим: prep, smoke, census, m1 или rollouts}"
DEV="${2:-cuda:1}"
SEED=101
BASIS="data/k15f/basis_s0.pt"
CACHE="data/k15f/h18_cache"
CENSUS="reports/k15f/m1_census.json"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p logs/k15f reports/k15f data/k15f
DTAG="$(echo "$DEV" | tr -d ':')"
IDENT="reports/k15f/identity_${DTAG}.json"
TRACKED="$(git ls-files reports/k15f data/k15f logs/k15f 2>/dev/null | head -3)"
[ -z "$TRACKED" ] || { echo "ОТКАЗ: reports/k15f отслеживается git"; exit 1; }
for m in k15f_continuous_refine k15f_pretrain_basis k15f_policy \
         k15f_measure_subspace k9h_multiarm_gate; do
  python3 "experiments/${m}.py" --selftest >/dev/null \
    || { echo "ОТКАЗ: самопроверка ${m}"; exit 1; }
done

if [ "$MODE" = "prep" ]; then
  echo "=== ПОДГОТОВКА $(date), карта $DEV, коммит" \
       "$(git rev-parse --short HEAD 2>/dev/null)"
  step () { echo; echo "=== $1 — $(date '+%H:%M:%S')"; }
  step "интеграция на игрушечной среде (CPU)"
  python3 experiments/k15f_check_identity.py --integration
  step "smoke-кэш h18"
  python3 experiments/k15f_build_cache.py --device "$DEV" \
    --smoke-batches 40 --out "$CACHE" --overwrite
  step "overfit базиса (smoke-кэш)"
  python3 experiments/k15f_pretrain_basis.py --mode overfit --device "$DEV" \
    --cache "$CACHE"
  step "smoke базиса"
  python3 experiments/k15f_pretrain_basis.py --mode smoke --device "$DEV" \
    --cache "$CACHE"
  step "полный кэш h18"
  if [ -f "$CACHE/COMPLETE" ]; then
    echo "  кэш уже готов: $CACHE"
  else
    python3 experiments/k15f_build_cache.py --device "$DEV" --out "$CACHE"
  fi
  step "полное предобучение базиса"
  python3 experiments/k15f_pretrain_basis.py --mode full --device "$DEV" \
    --cache "$CACHE" --out "$BASIS"
  step "гейт тождества"
  python3 experiments/k15f_check_identity.py --device "$DEV" \
    --basis "$BASIS" --h18-cache "$CACHE" --out "$IDENT"
  echo; echo "=== ПОДГОТОВКА ЗАКОНЧЕНА $(date)"
  exit 0
fi

if [ "$MODE" = "rollouts" ]; then
  bash "$0" smoke "$DEV" && bash "$0" census "$DEV" && bash "$0" m1 "$DEV"
  exit $?
fi

for f in "$BASIS" "$IDENT"; do
  [ -f "$f" ] || { echo "ОТКАЗ: нет $f (сначала prep)"; exit 1; }
done
sha12 () { sha1sum "$1" | cut -c1-12; }
TAG="k9h$(sha12 experiments/k9h_multiarm_gate.py)_b$(sha12 \
"$BASIS")_i$(sha12 "$IDENT")_p$(sha12 experiments/k15f_policy.py)"
OUTD="reports/k15f/m1/s${SEED}/${TAG}"
LOGF="logs/k15f/m1_${MODE}_s${SEED}.log"
mkdir -p "$OUTD"
exec >> "$LOGF" 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV; каталог $OUTD"
echo "    коммит $(git rev-parse --short HEAD 2>/dev/null)"

COMMON="--ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
  --task-suite 10 --n-envs 5 --seed $SEED --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 \
  --run-tag k15f_m1_s${SEED} --device $DEV --save-actions"
DRVQ="--policy depthrvq --policy-ckpt data/k9d_ep3.pt \
  --q1-ckpt data/k14c/q1_main_s0.pt --expect-q1-variant main \
  --expect-q1-seed 0 --expect-q0-manifest data/k14d/q0_b8_e0.manifest.json \
  --depth-rvq-mode fast"

block_ok () {   # JSON читается и его npz с тем же отпечатком на месте
  [ -f "$1" ] && python3 - "$1" >/dev/null 2>&1 <<'PY'
import hashlib, json, os, sys
d = json.load(open(sys.argv[1]))
f = os.path.join(os.path.dirname(sys.argv[1]), d["actions_npz"])
sys.exit(0 if hashlib.sha1(open(f, "rb").read()).hexdigest()[:12]
         == d["actions_npz_sha1"] else 1)
PY
}

run_arm () {   # метка, задача, блоки (через пробел)
  local L="$1" T="$2" rc=0 NEED=""
  for I0 in $3; do
    local F="$OUTD/${L}_t${T}_i${I0}.json"
    block_ok "$F" && continue
    rm -f "$F" "${F%.json}.actions.npz"
    NEED="$NEED${NEED:+,}$I0"
  done
  [ -z "$NEED" ] && { echo "    $L t$T: уже есть"; return 0; }
  echo "    $L t$T блоки $NEED — $(date '+%H:%M:%S')"
  local -a EXTRA
  if [ "$L" = q0 ]; then
    # shellcheck disable=SC2206
    EXTRA=($DRVQ)
  else
    EXTRA=(--policy k15f --k15f-basis "$BASIS")
  fi
  # shellcheck disable=SC2086
  python3 experiments/k9h_multiarm_gate.py $COMMON "${EXTRA[@]}" \
    --task-id "$T" --init-starts "$NEED" --arm-label "$L" \
    --out "$OUTD/${L}_t${T}_i{i0}.json" || rc=$?
  if [ $rc -ne 0 ]; then
    echo "ОСТАНОВ: $L, задача $T — код $rc"
    return $rc
  fi
  sleep 5
}

arts_of () {   # метки... -> файлы артефактов
  local out=""
  for L in "$@"; do
    out="$out $(ls "$OUTD"/${L}_t*_i*.json 2>/dev/null | tr '\n' ' ' || true)"
  done
  echo "$out"
}

CODE=0
case "$MODE" in
  smoke)
    for T in 0 8; do
      for L in q0 z l0p r0p; do run_arm "$L" "$T" "0"; done
    done
    python3 experiments/k15f_measure_subspace.py --mode smoke \
      --q0-arts $(ls "$OUTD"/q0_t0_i0.json "$OUTD"/q0_t8_i0.json) \
      --arts $(arts_of z l0p r0p) \
      --out "reports/k15f/m1_smoke_${TAG}.json" || CODE=$?
    ;;
  census)
    for T in 0 1 2 3 4 5 6 7 8 9; do run_arm q0 "$T" "0 5"; done
    python3 experiments/k15f_measure_subspace.py --mode census \
      --q0-arts $(arts_of q0) --census "$CENSUS" || CODE=$?
    ;;
  m1)
    [ -f "$CENSUS" ] || { echo "ОТКАЗ: нет переписи $CENSUS"; exit 1; }
    LABELS="$(python3 -c "import sys; sys.path.insert(0,'experiments'); \
import k15f_policy as k; print(' '.join(k.all_labels()))")"
    for T in 0 1 2 3 4 5 6 7 8 9; do
      BLK="$(python3 -c "import json,sys; \
print(' '.join(map(str, json.load(open('$CENSUS'))['blocks'].get('$T', []))))")"
      [ -z "$BLK" ] && continue
      set -- $LABELS
      N=$#; SEQ=""
      for k in $(seq 0 $((N - 1))); do
        i=$(( (k + T + SEED) % N + 1 ))
        SEQ="$SEQ ${!i}"
      done
      echo "--- задача $T, блоки $BLK, $(date)"
      for L in $SEQ; do run_arm "$L" "$T" "$BLK"; done
    done
    # shellcheck disable=SC2046
    python3 experiments/k15f_measure_subspace.py --mode m1 \
      --census "$CENSUS" --q0-arts $(arts_of q0) --arts $(arts_of $LABELS) \
      --out "reports/k15f/m1_${TAG}.json" || CODE=$?
    ;;
  *) echo "режим $MODE неизвестен"; exit 2 ;;
esac
echo "=== КОНЕЦ $(date), анализ: код $CODE ==="
exit $CODE
