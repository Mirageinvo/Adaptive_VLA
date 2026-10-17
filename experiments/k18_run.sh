#!/bin/bash
# K-18: обучение адаптивного базиса поправок по успеху задачи.
#   bash experiments/k18_run.sh selftest cpu
#   bash experiments/k18_run.sh prepare cpu
#   bash experiments/k18_run.sh run cuda:1 cuda:0     # q0 на первой карте,
#                                                     # малые сети — на второй
#   bash experiments/k18_run.sh analyze
# run сам проходит предпроверки (холодный q0 0->8->0, нулевая политика),
# общий набор q0, smoke, обучение до 100 тыс. команд с отчётом SR, обучение
# до 300 тыс. и итоговую оценку; при перезапуске продолжает с последней
# завершённой волны. Любая техническая ошибка — останов с причиной в логе.
set -euo pipefail
MODE="${1:?нужен режим: selftest, prepare, run или analyze}"
DEV1="${2:-cuda:1}"
DEV2="${3:-cuda:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p logs/k18 reports/k18 data/k18
LOGF="logs/k18/${MODE}.log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карты $DEV1 / $DEV2, коммит" \
     "$(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
selftests () {
  for m in k18_prepare k18_analyze k16_policy k17_policy; do
    python3 "experiments/${m}.py" --selftest >/dev/null \
      || { echo "ОТКАЗ: самопроверка ${m}"; finish 1; }
  done
  for m in k18_models k18_collector; do
    python3 "experiments/${m}.py" >/dev/null \
      || { echo "ОТКАЗ: самопроверка ${m}"; finish 1; }
  done
  python3 experiments/k18_td3.py \
    || { echo "ОТКАЗ: самопроверка k18_td3"; finish 1; }
  python3 experiments/k18_train.py selftest >/dev/null \
    || { echo "ОТКАЗ: самопроверка k18_train"; finish 1; }
  echo "    самопроверки K-18 пройдены"
}
case "$MODE" in
  selftest)
    selftests; finish 0 ;;
  prepare)
    CODE=0
    python3 experiments/k18_prepare.py --check-env || CODE=$?
    finish $CODE ;;
  run)
    selftests
    python3 experiments/k18_train.py check-config \
      || { echo "ОТКАЗ: конфигурация или код не совпадают с подготовкой"; \
           finish 1; }
    CODE=0
    python3 experiments/k18_train.py run --vla-device "$DEV1" \
      --rl-device "$DEV2" || CODE=$?
    if [ $CODE -ne 0 ]; then echo "ОСТАНОВ: обучение, код $CODE"; finish $CODE; fi
    python3 experiments/k18_analyze.py || CODE=$?
    finish $CODE ;;
  analyze)
    CODE=0
    python3 experiments/k18_analyze.py || CODE=$?
    finish $CODE ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
