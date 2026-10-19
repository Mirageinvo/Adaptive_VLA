#!/bin/bash
# K-21: однопроходная VLA с деревом движения — пакет 1.
#   bash experiments/k21_run.sh selftest cpu
#   bash experiments/k21_run.sh prepare cuda:0      # индексы, маски, учитель
#   bash experiments/k21_run.sh real-check cuda:0   # настоящий forward/backward
# Обучение (fitcheck, train), latency, plan и pilot добавляются в пакетах 2–3.
set -euo pipefail
MODE="${1:?нужен режим: selftest, prepare или real-check}"
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
    CODE=0
    python3 experiments/k21_model.py --real-check --device "$DEV" || CODE=$?
    finish $CODE ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
