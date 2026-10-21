#!/bin/bash
# K-21: однопроходная VLA с деревом движения — пакет 1.
#   bash experiments/k21_run.sh selftest cpu
#   bash experiments/k21_run.sh prepare cuda:0      # индексы, маски, закрепление
#   bash experiments/k21_run.sh real-check cuda:0   # forward/backward, 3 режима
#   bash experiments/k21_run.sh teacher cuda:0      # учитель (сбор заново)
#   MICRO=4 DTYPE=float16 bash experiments/k21_run.sh fitcheck cuda:0
#   MICRO=4 DTYPE=float16 bash experiments/k21_run.sh train cuda:0
#   MICRO=32 DTYPE=float16 bash experiments/k21_run.sh train-fast cuda:0
#   bash experiments/k21_run.sh latency cuda:0
# Пилот SR — отдельным раннером: experiments/k21_rollout.sh plan|run|analyze
# Обучение (fitcheck, train), latency, plan и pilot добавляются в пакетах 2–3.
set -euo pipefail
MODE="${1:?нужен режим: selftest, prepare, real-check, teacher, fitcheck или train}"
MICRO="${MICRO:-4}"
DTYPE="${DTYPE:-float16}"
DEV="${2:-cuda:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p logs/k21 reports/k21 data/k21
LOGF="logs/k21/${MODE}.log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит" \
     "$(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
case "$MODE" in
  selftest)
    python3 experiments/k21_train.py selftest \
      || { echo "ОТКАЗ: самопроверка k21_train"; finish 1; }
    for m in k21_data k21_model; do
      python3 "experiments/${m}.py" --selftest \
        || { echo "ОТКАЗ: самопроверка ${m}"; finish 1; }
    done
    python3 experiments/k21_tree.py || { echo "ОТКАЗ: k21_tree"; finish 1; }
    finish 0 ;;
  prepare)
    CODE=0
    python3 experiments/k21_data.py --device "$DEV" || CODE=$?
    finish $CODE ;;
  real-check)
    # каждый режим — отдельный процесс (чистая память, свежие веса)
    for M in tree_adapt24 flat24 tree_fixed24; do
      CODE=0
      python3 experiments/k21_model.py --real-check --mode "$M" \
        --device "$DEV" || CODE=$?
      [ $CODE -ne 0 ] && { echo "ОТКАЗ: real-check $M, код $CODE"; finish $CODE; }
    done
    finish 0 ;;
  teacher)
    CODE=0
    python3 experiments/k21_data.py --teacher --device "$DEV" || CODE=$?
    finish $CODE ;;
  fitcheck|train)
    CODE=0
    python3 experiments/k21_train.py "$MODE" --device "$DEV" \
      --micro "$MICRO" --dtype "$DTYPE" || CODE=$?
    finish $CODE ;;
  latency)
    CODE=0
    python3 experiments/k21_latency.py --device "$DEV" || CODE=$?
    finish $CODE ;;
  train-fast)
    # тот же k21_train, загрузка батчей — k21_fastload (pread + предзагрузка);
    # продолжает тот же resume.pt; первые батчи сверяются побитово
    python3 experiments/k21_fastload.py >/dev/null \
      || { echo "ОТКАЗ: самопроверка k21_fastload"; finish 1; }
    CODE=0
    python3 experiments/k21_train_fast.py train --device "$DEV" \
      --micro "$MICRO" --dtype "$DTYPE" || CODE=$?
    finish $CODE ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
