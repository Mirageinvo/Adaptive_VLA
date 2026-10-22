#!/bin/bash
# Согласованная цепочка (cuda:0, строго по очереди; останов на первом отказе):
#   1) SR spatial/object/goal (plan -> run);
#   2) K-22: real-check, fitcheck и обучение — micro 32 только если 32 прошёл
#      real-check;
#   3) допуск fullbar_cached, затем SR K-22 и fullbar_cached (по 30 процессов);
#   4) общий замер латентности всех политик.
# Этапы с уже готовыми результатами пропускаются по их собственным проверкам
# (готовые планы не пересоздаются, готовые блоки и чекпойнты переиспользуются).
set -uo pipefail
DEV="${1:-cuda:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
mkdir -p logs/k22
LOGF="logs/k22/chain_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOGF") 2>&1
step () {   # имя, команда...
  local name="$1"; shift
  echo "=== ЭТАП $name: $(date)"
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== ОСТАНОВ на этапе $name, код $rc: $(date)"; exit $rc
  fi
}
plan_once () {   # файл плана, команда создания...
  local f="$1"; shift
  if [ -f "$f" ]; then echo "    план $f уже зафиксирован"; return 0; fi
  "$@"
}
echo "=== СТАРТ цепочки $(date), карта $DEV, коммит" \
     "$(git rev-parse --short HEAD 2>/dev/null)"

step suites-plan plan_once reports/k21/plan_suites.json \
  bash experiments/k21_suite_rollout.sh plan "$DEV"
step suites-run bash experiments/k21_suite_rollout.sh run "$DEV"

step k22-real-check python3 experiments/k22_train.py real-check \
  --device "$DEV" --microbatches 8 32
MICRO="$(python3 -c "import json; r=json.load(open('reports/k22/real_check.json')); m=r['microbatch']; print(32 if (m.get('32') or {}).get('ok') and not r['problems'] else '')")"
if [ "$MICRO" != 32 ]; then
  echo "=== ОСТАНОВ: microbatch 32 не прошёл real-check (см."\
       "reports/k22/real_check.json); размер обучения — решение агента"
  exit 3
fi
step k22-fitcheck python3 experiments/k22_train.py fitcheck --device "$DEV" \
  --micro 32 --dtype float16
step k22-train python3 experiments/k22_train.py train --device "$DEV" \
  --micro 32 --dtype float16

step cached-admit python3 experiments/k21_barcache.py --admit --device "$DEV"
step k22-plan plan_once reports/k22/plan_main.json \
  bash experiments/k22_rollout.sh plan "$DEV"
step k22-run bash experiments/k22_rollout.sh run "$DEV"
step cached-plan plan_once reports/k21/plan_cached.json \
  bash experiments/k21_cached_rollout.sh plan "$DEV"
step cached-run bash experiments/k21_cached_rollout.sh run "$DEV"

step latency python3 experiments/k21_latency.py --device "$DEV" \
  --with-cached --with-parents24
echo "=== ЦЕПОЧКА ЗАВЕРШЕНА $(date); лог $LOGF"
