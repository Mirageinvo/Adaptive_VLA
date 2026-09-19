#!/bin/bash
# Канонические прогоны K-14c: один вариант, несколько сидов, последовательно.
#
# ПОСЛЕДОВАТЕЛЬНО И НА ОДНОЙ КАРТЕ. Тренер сверяет свой q0 с каноническим
# ПОБИТОВО, без допуска, а переносимость q0 между картами не измерена. Пока
# она не измерена, развести сиды по двум картам значило бы рискнуть тем, что
# один из них просто не стартует — или, хуже, стартует и решает другую задачу.
#
# СИДЫ РАЗЛИЧАЮТСЯ ТОЛЬКО ПОРЯДКОМ ДАННЫХ. Поздние головы инициализируются
# детерминированно, поэтому называть два прогона независимыми инициализациями
# неверно; это два порядка предъявления одних и тех же примеров.
#
#   bash experiments/k14_run_train.sh                  # main, сиды 0 и 1
#   bash experiments/k14_run_train.sh static cuda:1 0 1
set -u
VARIANT="${1:-main}"
DEV="${2:-cuda:1}"
shift 2 2>/dev/null || shift $# 
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
mkdir -p logs data/k14c
exec >> logs/k14_train.log 2>&1

Q0="data/k14d/q0_b8_e0.npz"
GR="reports/k14d/gate_r.json"
ORC="reports/k14a/oracle_canonical_$(echo "$DEV" | tr -d ':').json"
CACHE="data/k14b/q1_canonical"

echo "=== СТАРТ $(date) === вариант $VARIANT, карта $DEV, сиды ${SEEDS[*]}"
echo "    коммит $(git rev-parse --short HEAD 2>/dev/null)"

for S in "${SEEDS[@]}"; do
  LOG="logs/k14c_${VARIANT}_s${S}.log"
  echo "--- $VARIANT сид $S $(date) ---"
  python experiments/k14c_train_q1.py --variant "$VARIANT" --seed "$S" \
    --device "$DEV" --q1-cache "$CACHE" --q0 "$Q0" --gate-r "$GR" \
    --oracle "$ORC" > "$LOG" 2>&1
  rc=$?
  echo "    код $rc $(date)"
  if [ $rc -ne 0 ]; then
    echo "ОСТАНОВ: сид $S не дошёл, остальные не запускаю"
    tail -15 "$LOG"
    exit 1
  fi
  grep -E "ПОДТВЕРЖДЕНИЕ|Gate 4|выбрана эпоха|q0 совпал" "$LOG"
done
echo "=== КОНЕЦ $(date) ==="
