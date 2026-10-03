#!/bin/bash
# K-15c: парный поведенческий прогон q0 против селектора ранга.
#
# ДВА РЕЖИМА:
#   safety  задачи 8 и 9, состояния 0-24, 50 кластеров. ТЕХНИЧЕСКИЙ пилот:
#           нет NaN, раскатка доходит до конца, действия в диапазоне, нет
#           катастрофического падения, провенанс и парность верны. Как
#           go/no-go по эффективности НЕ используется — 50 кластеров мало.
#   dev     задачи 0-9, состояния 0-24, 250 кластеров — парный dev.
#
# ОДИН execution seed (101): раскатка измеренно детерминирована, согласие
# повторов было 250 из 250 кластеров (§53.11). Обе руки идут с одним сидом,
# порядок рук переставляется по задаче.
#
# РУКИ:
#   q0    точный K-14 черновик — та же рука и те же аргументы, что в K-14q;
#   k15c  один проход на 24 слоя, голова выбирает один из восьми путей,
#         один декод. Рука собирается только с отчётом проверки вывода,
#         пройденным ЭТОЙ головой и ЭТИМ модулем голов.
#
# РЕЖИМ ПРИВЯЗАН К СТАТУСУ ГОЛОВЫ из отчёта проверки вывода: dev — только
# при основном пороге (primary), safety — при основном или разведочном.
#
# КАТАЛОГ АРТЕФАКТОВ НЕСЁТ ОТПЕЧАТКИ головы, отчёта проверки вывода и
# харнесса. Готовые блоки пропускаются только внутри него: роллауты другой
# головы или другой версии харнесса не будут приняты за готовые.
#
# КАРТА — та же, что у гейта K-15a: рука k15c сверяет device. Git здесь не
# вызывается, кроме чтения хеша коммита для лога.
#
#   bash experiments/k15c_rollout.sh safety cuda:1 \
#       data/k15c/selectors/h24_candidate_s0.pt \
#       reports/k15c/inference_h24_candidate.json
set -euo pipefail
MODE="${1:?нужен режим: safety или dev}"
DEV="${2:-cuda:1}"
SELECTOR="${3:?нужен чекпойнт h24-головы}"
REPORT="${4:?нужен отчёт проверки вывода}"
SEED=101

case "$MODE" in
  safety) TASKS="8 9" ;;
  dev)    TASKS="0 1 2 3 4 5 6 7 8 9" ;;
  *) echo "режим $MODE неизвестен: safety или dev"; exit 2 ;;
esac
STATES="0 5 10 15 20"              # блоки по 5: состояния 0-24

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
for f in "$SELECTOR" "$REPORT" data/k15c/rank_cache/COMPLETE; do
  if [ ! -f "$f" ]; then echo "ОТКАЗ: нет $f"; exit 1; fi
done
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

STATUS="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))\
.get('selector_status') or '')" "$REPORT")"
case "$MODE:$STATUS" in
  dev:primary|safety:primary|safety:pilot) ;;
  *) echo "ОТКАЗ: режим $MODE при статусе головы «$STATUS»: dev положен только"
     echo "  при основном пороге, safety — при основном или разведочном"
     exit 3 ;;
esac
sha12 () { sha1sum "$1" | cut -c1-12; }
TAG="sel$(sha12 "$SELECTOR")_rep$(sha12 "$REPORT")_k9h$(sha12 \
experiments/k9h_multiarm_gate.py)"
# САМОПРОВЕРКИ ДО СРЕД И МОДЕЛИ: секунды, без GPU.
for m in k15c_rank_selector k15c_policy k15c_behavior k9h_multiarm_gate; do
  if ! python3 "experiments/${m}.py" --selftest >/dev/null; then
    echo "ОТКАЗ: самопроверка ${m} не прошла"; exit 1
  fi
done

TASKS_RUN="${SMOKE_TASKS:-$TASKS}"
if [ -n "${SMOKE_TASKS:-}" ] || [ -n "${SMOKE_BLOCKS:-}" ]; then
  IS_SMOKE=1
  [ -n "${SMOKE_BLOCKS:-}" ] && STATES="$SMOKE_BLOCKS"
  OUTD="reports/k15c/rollout/smoke_${MODE}_s${SEED}/${TAG}"
  LOGF="logs/k15c/rollout_smoke_${MODE}_s${SEED}.log"
else
  IS_SMOKE=0
  OUTD="reports/k15c/rollout/${MODE}/s${SEED}/${TAG}"
  LOGF="logs/k15c/rollout_${MODE}_s${SEED}.log"
fi
mkdir -p logs/k15c "$OUTD"
exec >> "$LOGF" 2>&1

echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, сид $SEED"
echo "    коммит $(git rev-parse --short HEAD 2>/dev/null)"
echo "    голова $SELECTOR (статус $STATUS), проверка вывода $REPORT"
echo "    каталог $OUTD"
echo "    задачи: $TASKS_RUN; блоки состояний: $STATES"
[ "$IS_SMOKE" = "1" ] && echo "    СМОУК: набор урезан, артефакты в $OUTD"

COMMON="--ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
  --task-suite 10 --n-envs 5 --seed $SEED --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 \
  --run-tag k15c_${MODE}_s${SEED} --device $DEV --save-actions"
# РУКА q0 — ДОСЛОВНО КАК В K-14q: те же веса, голова и режим fast.
DRVQ="--policy depthrvq --policy-ckpt data/k9d_ep3.pt \
  --q1-ckpt data/k14c/q1_main_s0.pt --expect-q1-variant main \
  --expect-q1-seed 0 --expect-q0-manifest data/k14d/q0_b8_e0.manifest.json"
K15C="--policy k15c --selector $SELECTOR --inference-report $REPORT"

run_arm () {   # $1 метка, $2 задача, $3... аргументы руки
  # ВСЕ БЛОКИ ЗАДАЧИ — ОДНИМ ПРОЦЕССОМ: загрузка модели и сред дороже
  # самой раскатки. Готовые блоки пропускаются, возобновление безопасно.
  local L="$1" T="$2"; shift 2
  local rc=0 NEED=""
  for I0 in $STATES; do
    local F="$OUTD/${L}_t${T}_i${I0}.json"
    if [ -f "$F" ] && python3 -c "import json;json.load(open('$F'))" \
         >/dev/null 2>&1; then
      continue
    fi
    rm -f "$F"
    NEED="$NEED${NEED:+,}$I0"
  done
  if [ -z "$NEED" ]; then
    echo "    $L t$T: все блоки уже есть"
    return 0
  fi
  echo "    $L t$T: блоки $NEED"
  python3 experiments/k9h_multiarm_gate.py $COMMON "$@" \
    --task-id "$T" --init-starts "$NEED" \
    --arm-label "${L}" --out "$OUTD/${L}_t${T}_i{i0}.json" || rc=$?
  if [ $rc -ne 0 ]; then
    echo "ОСТАНОВ: $L, задача $T, блоки $NEED — код $rc"
    return $rc
  fi
  echo "    $L t$T готово: $(grep -h 'успех ' "$LOGF" | tail -1)"
  sleep 10
}

for T in $TASKS_RUN; do
  echo "--- задача $T, $(date), свободно $(free -g | awk 'NR==2{print $7}') ГБ"
  if [ $(( (T + SEED) % 2 )) -eq 0 ]; then SEQ="q0 k15c"
  else SEQ="k15c q0"; fi
  echo "    порядок рук: $SEQ"
  for A in $SEQ; do
    case $A in
      q0)   run_arm q0   "$T" $DRVQ --depth-rvq-mode fast ;;
      k15c) run_arm k15c "$T" $K15C ;;
    esac
  done
done

echo "=== раскатки закончены $(date) ==="
NT=$(echo $TASKS_RUN | wc -w); NB=$(echo $STATES | wc -w)
# ТОЛЬКО ФАЙЛЫ РУК: сводка анализа лежит рядом и в подсчёт входить не должна.
ARTS=$(ls "$OUTD"/q0_t*_i*.json "$OUTD"/k15c_t*_i*.json 2>/dev/null || true)
N=$(echo $ARTS | wc -w)
EXP=$(( 2 * NT * NB ))
echo "    артефактов $N (ожидается $EXP: 2 руки x $NT задач x $NB блоков)"
if [ "$N" -ne "$EXP" ]; then
  echo "ОСТАНОВ: набор неполный, анализ не запускаю"
  exit 4
fi
if [ "$IS_SMOKE" = "1" ]; then
  # АНАЛИЗ СМОУКА ПРОВЕРЯЕТ ПАРНОСТЬ, ПРОВЕНАНС И КОНЕЧНОСТЬ ДЕЙСТВИЙ, и его
  # отказ — отказ смоука. Код 4 (катастрофа по разности успеха) на пяти
  # эпизодах ничего не значит и смоук не валит.
  SC=0
  python3 experiments/k15c_behavior.py --mode "$MODE" --allow-partial \
    --arts $ARTS --out "$OUTD/behavior_smoke.json" --overwrite || SC=$?
  if [ "$SC" != 0 ] && [ "$SC" != 4 ]; then
    echo "ОСТАНОВ: анализ смоука отказал (код $SC)"
    exit "$SC"
  fi
  echo "=== СМОУК ЗАКОНЧЕН $(date). Это НЕ результат ==="
  exit 0
fi
# ПРИ set -e ОТКАЗ АНАЛИЗА ОБОРВАЛ БЫ СКРИПТ ДО СТРОКИ ИТОГА: код
# забирается явно.
CODE=0
python3 experiments/k15c_behavior.py --mode "$MODE" --arts $ARTS \
  --out "reports/k15c/rollout/behavior_${MODE}_s${SEED}_${TAG}.json" \
  --overwrite || CODE=$?
echo "=== КОНЕЦ $(date), анализ: код $CODE ==="
exit $CODE
