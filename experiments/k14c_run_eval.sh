#!/bin/bash
# Диагностика обученной головы q1: без обучения, без подтверждающей половины.
#
# ЖЁСТКИЙ РЕЖИМ. Любая незамеченная ошибка на этом шаге дороже, чем отказ:
# измерение идёт час, а его числа идут в FINDINGS. Поэтому set -e,
# самопроверка перед запуском и отказ при существующих выходных файлах.
#
# GIT СКРИПТ НЕ ТРОГАЕТ. Обновление рабочей копии — дело оператора: скрипт,
# который сам делает pull, меняет то, что собирается исполнить, и печатает
# номер коммита, которого мог не иметь на старте.
#
#   bash experiments/k14c_run_eval.sh              # cuda:1, сид 0, метка v2
#   bash experiments/k14c_run_eval.sh cuda:1 1 v3  # другой сид или метка
#
# КАРТА ТОЛЬКО ТА, НА КОТОРОЙ ПОСТРОЕН КАНОНИЧЕСКИЙ q0. На другой тренер
# откажется: он сверяет gpu_uuid и режим вычислений с манифестом черновика.
# Диагностика на чужой карте возможна, но требует явного
# --allow-device-drift и является измерением переносимости, а не оценкой.
set -euo pipefail
DEV="${1:-cuda:1}"
SEED="${2:-0}"
TAG="${3:-v2}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl

mkdir -p logs reports/k14c
echo "коммит рабочей копии: $(git rev-parse --short HEAD 2>/dev/null)"

CK="data/k14c/q1_main_s${SEED}.pt"
SUM="reports/k14c/eval_main_s${SEED}_${TAG}.json"
LOG="logs/k14c_eval_s${SEED}_${TAG}.log"

python experiments/k14c_train_q1.py --selftest
test -f "$CK"          || { echo "нет чекпойнта $CK"; exit 1; }
test ! -e "$SUM"       || { echo "$SUM уже есть"; exit 1; }
test ! -e "$LOG"       || { echo "$LOG уже есть"; exit 1; }

nohup setsid env PYTHONUNBUFFERED=1 PYTHONPATH="$PYTHONPATH" MUJOCO_GL=egl \
python experiments/k14c_train_q1.py \
  --variant main --seed "$SEED" \
  --eval-checkpoint "$CK" \
  --device "$DEV" \
  --q1-cache data/k14b/q1_canonical \
  --q0 data/k14d/q0_b8_e0.npz \
  --gate-r reports/k14d/gate_r.json \
  --oracle reports/k14a/oracle_canonical_cuda1.json \
  --summary "$SUM" \
  > "$LOG" 2>&1 < /dev/null &
echo "запущено, pid $!; лог $LOG, сводка $SUM"
