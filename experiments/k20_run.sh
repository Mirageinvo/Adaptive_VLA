#!/bin/bash
# K-20: однопроходная временная иерархия действий — этап 1.
#   bash experiments/k20_run.sh selftest cpu
#   bash experiments/k20_run.sh prepare cuda:0   # индексы, учитель BAR, исходный кодек
#   bash experiments/k20_run.sh codec cuda:0     # подгонка 512 чанков, затем hier и flat
# Следующие этапы (warmup, first-sr, ...) добавляются вместе с моделью.
set -euo pipefail
MODE="${1:?нужен режим: selftest, prepare или codec}"
DEV="${2:-cuda:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p logs/k20 reports/k20 data/k20
LOGF="logs/k20/${MODE}.log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит" \
     "$(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
case "$MODE" in
  selftest)
    for m in k20_data k20_codec; do
      python3 "experiments/${m}.py" --selftest \
        || { echo "ОТКАЗ: самопроверка ${m}"; finish 1; }
    done
    finish 0 ;;
  prepare)
    CODE=0
    python3 experiments/k20_data.py --device "$DEV" || CODE=$?
    finish $CODE ;;
  codec)
    CODE=0
    python3 experiments/k20_codec.py --fitcheck --device "$DEV" || CODE=$?
    [ $CODE -ne 0 ] && finish $CODE
    python3 experiments/k20_codec.py --device "$DEV" || CODE=$?
    finish $CODE ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
