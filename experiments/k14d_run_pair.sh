#!/bin/bash
# Пара прогонов K-14d и Gate R, последовательно на одной карте.
#
# ПОЧЕМУ ПОСЛЕДОВАТЕЛЬНО И НА ОДНОЙ КАРТЕ. Gate R меняет ровно одну вещь —
# порядок исполнения батчей. Разведи прогоны по разным картам, и расхождение
# уже нельзя приписать причине; заверение сравнивает и устройство, и его
# физический UUID, поэтому такую пару оно всё равно отвергнет.
#
# ОСТАНОВ НА ПЕРВОМ ОТКАЗЕ. Если первый прогон упал, второй не запускается:
# полтора часа карты не тратятся на половину пары.
#
# Путь к LIBERO берётся из $LIBERO_PATH, иначе $HOME/LIBERO.
#
#   bash experiments/k14d_run_pair.sh            # порядки 0 и 7
#   bash experiments/k14d_run_pair.sh cuda:0 1 9 # карта и оба порядка
set -u
DEV="${1:-cuda:1}"
S1="${2:-0}"
S2="${3:-7}"
BATCH="${4:-8}"

# КОРЕНЬ БЕРЁТСЯ ИЗ ПУТИ САМОГО СКРИПТА, а не из $HOME: в контейнере
# домашний каталог пользователя и каталог репозитория разные.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
mkdir -p logs data/k14d
exec >> logs/k14d_pair.log 2>&1

A="data/k14d/q0_b${BATCH}_e${S1}"
B="data/k14d/q0_b${BATCH}_e${S2}"

echo "=== СТАРТ $(date) === карта $DEV, порядки $S1 и $S2, батч $BATCH"
echo "    коммит $(git rev-parse --short HEAD 2>/dev/null)"

run() {   # $1 — сид порядка, $2 — префикс вывода, $3 — файл лога
  echo "--- порядок $1 $(date) ---"
  python experiments/k14d_build_q0.py --device "$DEV" --batch "$BATCH" \
    --exec-order-seed "$1" --out "$2" > "$3" 2>&1
  local rc=$?
  echo "    код $rc $(date); последняя строка: $(tail -1 "$3")"
  return $rc
}

run "$S1" "$A" logs/k14d_e${S1}.log || {
  echo "ОСТАНОВ: первый прогон не дошёл, второй не запускаю"; exit 1; }
run "$S2" "$B" logs/k14d_e${S2}.log || {
  echo "ОСТАНОВ: второй прогон не дошёл, Gate R не считаю"; exit 1; }

echo "--- GATE R $(date) ---"
python experiments/k14d_build_q0.py \
  --gate-r "$A.manifest.json" "$B.manifest.json" > logs/k14d_gate_r.log 2>&1
grc=$?
echo "    код $grc"
cat logs/k14d_gate_r.log
echo "=== КОНЕЦ $(date) ==="
# КОД ОТКАЗА ПРОБРАСЫВАЕТСЯ НАРУЖУ. Прежде после echo и cat скрипт завершался
# нулём, и непройденный Gate R выглядел снаружи как успешная пара.
exit $grc
