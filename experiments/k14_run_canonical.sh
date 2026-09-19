#!/bin/bash
# Каноническая цепочка D': Gate 2 на заверенном черновике, затем кэш целей.
#
# ПОРЯДОК ИМЕННО ТАКОЙ. K-14b требует пройденного Gate 2 и сверяет с ним метки
# побитово, поэтому гейт обязан быть посчитан раньше. Обратный порядок означал
# бы, что кэш ссылается на артефакт, которого ещё нет, или на старый.
#
# ОДНА КАРТА НА ОБА ШАГА: K-14b отказывается брать гейт, пройденный на другом
# устройстве, и это правильно — метки и предел оракула должны быть посчитаны
# одной арифметикой.
#
#   bash experiments/k14_run_canonical.sh                     # cuda:1
#   bash experiments/k14_run_canonical.sh cuda:0 data/k14d/q0_b8_e7.npz
set -u
DEV="${1:-cuda:1}"
Q0="${2:-data/k14d/q0_b8_e0.npz}"
GR="${3:-reports/k14d/gate_r.json}"
[ "$GR" = "--only-b" ] && GR="reports/k14d/gate_r.json"

ONLY_B=0
for arg in "$@"; do [ "$arg" = "--only-b" ] && ONLY_B=1; done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
mkdir -p logs reports/k14a data/k14b
exec >> logs/k14_canonical.log 2>&1

ORC="reports/k14a/oracle_canonical_$(echo "$DEV" | tr -d ':').json"
OUT="data/k14b/q1_canonical"

echo "=== СТАРТ $(date) === карта $DEV, черновик $Q0"
echo "    коммит $(git rev-parse --short HEAD 2>/dev/null)"

if [ "$ONLY_B" = "0" ]; then
  echo "--- K-14a, Gate 2 $(date) ---"
  # БЕЗ --overwrite. Артефакт гейта неизменяем: на него уже ссылаются кэш
  # целей и чекпойнты. Прежняя версия перезаписывала его при каждом запуске,
  # а K-14b следом отказывался из-за существующего кэша — и старый кэш
  # оставался привязан к уничтоженной версии гейта. Для пересчёта — новое имя
  # через третий аргумент.
  python experiments/k14a_oracle_cache.py --device "$DEV" --q0 "$Q0" \
    --gate-r "$GR" --run-id "canon-$(date +%Y%m%dT%H%M%S)" \
    --out "$ORC" > logs/k14a_canonical.log 2>&1
  rc=$?
  echo "    код $rc $(date)"
  [ $rc -ne 0 ] && { echo "ОСТАНОВ: гейт не пройден, кэш целей не строю"; \
    exit 1; }
  grep -E "ГЕЙТ 2|РЕШЕНИЕ|ОБУЧАТЬ|черновик:" logs/k14a_canonical.log
else
  echo "--- K-14a пропущен, гейт берётся готовый: $ORC ---"
fi

echo "--- K-14b, кэш целей $(date) ---"
python experiments/k14b_build_q1_cache.py --device "$DEV" --q0 "$Q0" \
  --gate-r "$GR" --oracle "$ORC" --out "$OUT" \
  > logs/k14b_canonical.log 2>&1
rc=$?
echo "    код $rc $(date)"
[ $rc -ne 0 ] && { echo "ОСТАНОВ: кэш целей не построен"; exit 1; }
tail -15 logs/k14b_canonical.log
echo "=== КОНЕЦ $(date) ==="
