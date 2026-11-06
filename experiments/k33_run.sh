#!/bin/bash
# K-33: свежий банк LIBERO-10 (native reset) и подтверждающая оценка.
# Первая очередь (инфраструктурный риск — до остального пакета):
#   bash experiments/k33_run.sh selftest
#   bash experiments/k33_run.sh bank [WORKERS]        # без политик, CPU+рендер
#   bash experiments/k33_run.sh bank-check [WORKERS]  # два холодных воспроизведения
#   bash experiments/k33_run.sh legacy-smoke DEV      # fullbar t0/t8 i30: старый путь против адаптера
#   bash experiments/k33_run.sh dev-replay DEV        # fullbar на банке dev: хеши = банку
# Все физические шаги проверок — в журнале data/k33/ledger.json (checks).
set -uo pipefail
MODE="${1:?нужен режим}"
ARG2="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p logs/k33 reports/k33 data/k33
exec 9>logs/k33/.k33.lock
flock -n 9 || { echo "ОТКАЗ: другой k33_run.sh уже выполняется"; exit 2; }
LOGF="logs/k33/${MODE}_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE $ARG2, коммит $(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
PY=python3
PROT="$PY experiments/k33_protocol.py"
PLAN_A="$($PY -c "import sys; sys.path.insert(0,'experiments'); import k21_main_protocol as p; print(p.plan_path('A'))")"
CKPT="$($PY -c "import json; print(json.load(open('$PLAN_A'))['setup']['ckpt'])")"
COMMON () {   # DEV
  echo "--ckpt $CKPT --task-suite 10 --n-envs 5 --seed 101 --rollout-seed-mode fixed \
--ensemble off --horizon 8 --max-steps 600 --waiting-steps 10 --device $1 --save-actions \
--policy fullbar --arm-label fullbar"
}
block () {   # путь(cur|new) DEV задача блок каталог [режим] [space]
  local W="$1" DEV="$2" T="$3" B="$4" D="$5" M="${6:-legacy}" S="${7:-0}"
  local EID="checks/$(basename "$D")/$W/t${T}_i${B}@$(date +%s)" OUT="$D/fullbar_t${T}_i{i0}.json"
  mkdir -p "$D"
  $PROT ledger --op reserve --eid "$EID" || return 3
  if [ "$W" = cur ]; then
    # shellcheck disable=SC2046
    $PY experiments/k21_main_harness.py --k21-main-plan "$PLAN_A" $(COMMON "$DEV") \
      --run-tag k33_legacy --task-id "$T" --init-starts "$B" --out "$OUT" || return 3
  else
    # shellcheck disable=SC2046
    $PY experiments/k33_harness.py --k33-mode "$M" --k33-space "$S" $(COMMON "$DEV") \
      --run-tag k33_legacy --task-id "$T" --init-starts "$B" --out "$OUT" || return 3
  fi
  $PROT ledger --op commit --eid "$EID" --json-path "$D/fullbar_t${T}_i${B}.json"
}

case "$MODE" in
  selftest)
    for m in k33_bank k33_harness k33_protocol; do
      $PY "experiments/${m}.py" --selftest || finish 1
    done; finish 0 ;;
  bank) $PROT bank --workers "${ARG2:-6}"; finish $? ;;
  bank-check) $PROT bank-check --workers "${ARG2:-6}"; finish $? ;;
  legacy-smoke)
    DEV="${ARG2:-cuda:0}"
    for T in 0 8; do
      block cur "$DEV" "$T" 30 reports/k33/legacy/cur || finish 3
      block new "$DEV" "$T" 30 reports/k33/legacy/new legacy || finish 3
    done
    $PROT legacy-compare --cur reports/k33/legacy/cur --new reports/k33/legacy/new; finish $? ;;
  dev-replay)
    DEV="${ARG2:-cuda:0}"
    for T in 0 8; do
      block new "$DEV" "$T" 0 reports/k33/devreplay bank 0 || finish 3
    done
    $PROT dev-compare --new reports/k33/devreplay --space 0; finish $? ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
