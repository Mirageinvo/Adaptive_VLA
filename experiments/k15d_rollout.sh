#!/bin/bash
# K-15d: проверка вывода и парные роллауты q0, D0, H1.
#
#   infer   проверка вывода чекпойнтов D0 и H1p2 на карте роллаута
#           (k15d_check_inference: весь val_sel через функцию руки);
#   smoke   задача 8, состояния 0-4, три руки и ПОВТОР D0 — проверка
#           запуска, конечности, диапазона, записи a0/a1/a2, правильного
#           чекпойнта и детерминизма (хеши действий повтора совпадают);
#   safety  задачи 8-9, состояния 0-24, 50 кластеров — технический пилот;
#   dev     задачи 0-9, состояния 0-24, 250 кластеров — парный dev.
#
# РУКИ: q0 — точный K-14 черновик (как в K-14q/K-15c); d0 и h1 — рука
# k15d с чекпойнтами data/k15d/d0_s0.pt и data/k15d/h1p2_s0.pt. Рука, чей
# чекпойнт не допущен фильтром, пропускается переменной SKIP="d0" или
# SKIP="h1"; анализ тогда сравнивает с q0 только оставшуюся.
#
# ОДИН execution seed (101), порядок рук переставляется по задаче. Каталог
# артефактов несёт отпечатки чекпойнтов, отчётов проверки вывода и
# харнесса: готовые блоки пропускаются только внутри него.
#
# КАРТА: обе руки k15d берут гейты K-15a и K-15d ЭТОЙ карты (для cuda:1 —
# канонические, для cuda:0 — *_cuda0.json). Роллауты идут по одному
# процессу: две MuJoCo-раскатки параллельно не помещаются в память хоста.
#
#   bash experiments/k15d_rollout.sh infer cuda:1
#   bash experiments/k15d_rollout.sh smoke cuda:1
#   bash experiments/k15d_rollout.sh safety cuda:1
set -euo pipefail
MODE="${1:?нужен режим: infer, smoke, safety или dev}"
DEV="${2:-cuda:1}"
SEED=101
D0_CK="${D0_CK:-data/k15d/d0_s0.pt}"
H1_CK="${H1_CK:-data/k15d/h1p2_s0.pt}"
SKIP="${SKIP:-}"
DTAG="$(echo "$DEV" | tr -d ':')"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p logs/k15d reports/k15d/rollout

report_of () {   # чекпойнт -> отчёт проверки вывода на этой карте
  echo "reports/k15d/inference_$(basename "${1%.pt}")_${DTAG}.json"
}
want () { case " $SKIP " in *" $1 "*) return 1;; esac; return 0; }

if [ "$MODE" = "infer" ]; then
  RC=0
  for A in d0 h1; do
    want "$A" || { echo "  $A пропущена (SKIP)"; continue; }
    CK="$D0_CK"; [ "$A" = h1 ] && CK="$H1_CK"
    echo "=== проверка вывода $A: $CK на $DEV — $(date '+%H:%M:%S')"
    python3 experiments/k15d_check_inference.py --checkpoint "$CK" \
      --device "$DEV" --out "$(report_of "$CK")" || RC=$?
  done
  exit $RC
fi

case "$MODE" in
  smoke)  TASKS="8"; STATES="0" ;;
  safety) TASKS="8 9"; STATES="0 5 10 15 20" ;;
  dev)    TASKS="0 1 2 3 4 5 6 7 8 9"; STATES="0 5 10 15 20" ;;
  *) echo "режим $MODE неизвестен"; exit 2 ;;
esac
D0_REP="$(report_of "$D0_CK")"
H1_REP="$(report_of "$H1_CK")"
ARMS="q0"
for A in d0 h1; do
  want "$A" || continue
  CK="$D0_CK"; REP="$D0_REP"
  [ "$A" = h1 ] && { CK="$H1_CK"; REP="$H1_REP"; }
  for f in "$CK" "$REP"; do
    [ -f "$f" ] || { echo "ОТКАЗ: нет $f (сначала режим infer)"; exit 1; }
  done
  ARMS="$ARMS $A"
done
[ "$ARMS" = "q0" ] && { echo "ОТКАЗ: нет ни одной руки K-15d"; exit 1; }
for m in k15d_depth_refine k15d_policy k15d_check_inference k15d_behavior \
         k9h_multiarm_gate; do
  python3 "experiments/${m}.py" --selftest >/dev/null \
    || { echo "ОТКАЗ: самопроверка ${m}"; exit 1; }
done
sha12 () { sha1sum "$1" | cut -c1-12; }
TAG="k9h$(sha12 experiments/k9h_multiarm_gate.py)"
want d0 && TAG="${TAG}_d0$(sha12 "$D0_CK")$(sha12 "$D0_REP")"
want h1 && TAG="${TAG}_h1$(sha12 "$H1_CK")$(sha12 "$H1_REP")"
OUTD="reports/k15d/rollout/${MODE}/s${SEED}/${TAG}"
LOGF="logs/k15d/rollout_${MODE}_s${SEED}.log"
mkdir -p "$OUTD"
exec >> "$LOGF" 2>&1

echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, сид $SEED, руки $ARMS"
echo "    коммит $(git rev-parse --short HEAD 2>/dev/null); каталог $OUTD"

COMMON="--ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
  --task-suite 10 --n-envs 5 --seed $SEED --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 \
  --run-tag k15d_${MODE}_s${SEED} --device $DEV --save-actions"
DRVQ="--policy depthrvq --policy-ckpt data/k9d_ep3.pt \
  --q1-ckpt data/k14c/q1_main_s0.pt --expect-q1-variant main \
  --expect-q1-seed 0 --expect-q0-manifest data/k14d/q0_b8_e0.manifest.json \
  --depth-rvq-mode fast"

arm_args () {   # рука -> аргументы харнесса
  case "$1" in
    q0)       echo "$DRVQ" ;;
    d0|d0r)   echo "--policy k15d --refiner $D0_CK --k15d-report $D0_REP" ;;
    h1)       echo "--policy k15d --refiner $H1_CK --k15d-report $H1_REP" ;;
  esac
}

run_arm () {   # метка, задача
  local L="$1" T="$2" rc=0 NEED=""
  for I0 in $STATES; do
    local F="$OUTD/${L}_t${T}_i${I0}.json"
    if [ -f "$F" ] && python3 -c "import json;json.load(open('$F'))" \
         >/dev/null 2>&1; then continue; fi
    rm -f "$F"
    NEED="$NEED${NEED:+,}$I0"
  done
  [ -z "$NEED" ] && { echo "    $L t$T: все блоки уже есть"; return 0; }
  echo "    $L t$T: блоки $NEED — $(date '+%H:%M:%S')"
  # shellcheck disable=SC2046
  python3 experiments/k9h_multiarm_gate.py $COMMON $(arm_args "$L") \
    --task-id "$T" --init-starts "$NEED" --arm-label "$L" \
    --out "$OUTD/${L}_t${T}_i{i0}.json" || rc=$?
  if [ $rc -ne 0 ]; then
    echo "ОСТАНОВ: $L, задача $T, блоки $NEED — код $rc"
    return $rc
  fi
  echo "    $L t$T готово: $(grep -h 'успех ' "$LOGF" | tail -1)"
  sleep 10
}

for T in $TASKS; do
  echo "--- задача $T, $(date), свободно $(free -g | awk 'NR==2{print $7}') ГБ"
  # порядок рук циклически сдвигается по задаче
  set -- $ARMS
  N=$#; SEQ=""
  for k in $(seq 0 $((N - 1))); do
    i=$(( (k + T + SEED) % N + 1 ))
    SEQ="$SEQ ${!i}"
  done
  echo "    порядок рук:$SEQ"
  for A in $SEQ; do run_arm "$A" "$T"; done
done

if [ "$MODE" = "smoke" ] && want d0; then
  # ДЕТЕРМИНИЗМ: повтор руки d0 обязан дать побитово те же действия.
  for T in $TASKS; do run_arm d0r "$T"; done
  # set -e оборвал бы скрипт до понятного сообщения: исход берётся явно.
  if ! python3 - "$OUTD" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
bad = 0
for f in sorted(glob.glob(os.path.join(d, "d0r_t*_i*.json"))):
    g = f.replace("/d0r_", "/d0_")
    a = [e["action_sha1"] for e in json.load(open(f))["episodes"]]
    b = [e["action_sha1"] for e in json.load(open(g))["episodes"]]
    same = sum(x == y for x, y in zip(a, b))
    print(f"    детерминизм {os.path.basename(g)}: {same}/{len(a)} эпизодов "
          f"совпали")
    bad += len(a) - same
sys.exit(1 if bad else 0)
PY
  then
    echo "ОСТАНОВ: повтор d0 разошёлся"
    exit 1
  fi
fi

echo "=== раскатки закончены $(date) ==="
ARTS=""
for A in $ARMS; do
  ARTS="$ARTS $(ls "$OUTD"/${A}_t*_i*.json 2>/dev/null | tr '\n' ' ')"
done
CODE=0
if [ "$MODE" = "smoke" ]; then
  python3 experiments/k15d_behavior.py --mode safety --allow-partial \
    --arts $ARTS --out "$OUTD/behavior_smoke.json" --overwrite || CODE=$?
  # на пяти эпизодах код 4 (разность успеха) ничего не значит
  [ "$CODE" = 4 ] && CODE=0
  echo "=== СМОУК ЗАКОНЧЕН $(date), код $CODE. Это НЕ результат ==="
  exit $CODE
fi
python3 experiments/k15d_behavior.py --mode "$MODE" --arts $ARTS \
  --out "reports/k15d/rollout/behavior_${MODE}_s${SEED}_${TAG}.json" \
  --overwrite || CODE=$?
echo "=== КОНЕЦ $(date), анализ: код $CODE ==="
exit $CODE
