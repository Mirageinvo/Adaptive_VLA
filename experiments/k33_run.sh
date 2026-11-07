#!/bin/bash
# K-33: свежий банк LIBERO-10 (native reset) и подтверждающая оценка.
# Первая очередь (инфраструктурный риск — до остального пакета):
#   bash experiments/k33_run.sh selftest
#   bash experiments/k33_run.sh bank [WORKERS]        # без политик, CPU+рендер
#   bash experiments/k33_run.sh bank-check [WORKERS]  # два холодных воспроизведения
#   bash experiments/k33_run.sh legacy-smoke DEV      # fullbar t0/t8 i30: старый путь против адаптера
#   bash experiments/k33_run.sh dev-replay DEV        # fullbar на банке dev: хеши = банку
# Поправка §14 (2026-11-08): банк v1 строился без set_init_state(S0) после
# снимка — пересборка в scope bank_v2 (лимит 100 000; scope bank закрыт):
#   bash experiments/k33_run.sh migrate             # журнал: + bank_v2 (однократно)
#   bash experiments/k33_run.sh postfix             # generate = replay, S0/подпись = банку v1
#   bash experiments/k33_run.sh archive-v1          # перенос банка v1 (не удаление)
#   bash experiments/k33_run.sh compare-v1          # ID/seed/S0/подписи v1 = v2
#   bash experiments/k33_run.sh rebuild-v2 W DEV    # всё по порядку с остановками
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
POSTFIX_CASES="0:9:0,0:0:0,0:8:0,1:9:0,1:2:0"
PLAN_A="" CKPT=""
plan_a () {   # путь плана A K-21 и снимок BAR (только для режимов с раскатками)
  PLAN_A="$($PY -c "import sys; sys.path.insert(0,'experiments'); import k21_main_protocol as p; print(p.plan_path('A'))")" || return 3
  CKPT="$($PY -c "import json; print(json.load(open('$PLAN_A'))['setup']['ckpt'])")" || return 3
  [ -d "$CKPT" ] || { echo "ОТКАЗ: нет снимка BAR $CKPT"; return 3; }
}
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

selftests () {
  for m in k33_bank k33_harness k33_protocol k33_diag; do
    $PY "experiments/${m}.py" --selftest || return 1
  done
}
dev_replay () {   # DEV
  plan_a || return 3
  for T in 0 8; do
    block new "$1" "$T" 0 reports/k33/devreplay bank 0 || return 3
  done
  $PROT dev-compare --new reports/k33/devreplay --space 0
}

case "$MODE" in
  selftest) selftests; finish $? ;;
  migrate) $PROT migrate-ledger; finish $? ;;
  postfix) $PY experiments/k33_diag.py --post-fix --cases "$POSTFIX_CASES"; finish $? ;;
  archive-v1) $PROT archive-v1; finish $? ;;
  compare-v1) $PROT compare-v1; finish $? ;;
  rebuild-v2)
    W="${ARG2:-6}"; DEV="${3:-cuda:0}"
    echo "=== ЭТАП миграция журнала"; $PROT migrate-ledger || finish 3
    echo "=== ЭТАП самопроверки"; selftests || finish 1
    echo "=== ЭТАП post-fix"
    $PY experiments/k33_diag.py --post-fix --cases "$POSTFIX_CASES" || { echo "ОСТАНОВ: post-fix"; finish 3; }
    echo "=== ЭТАП архив v1"; $PROT archive-v1 || finish 3
    echo "=== ЭТАП генерация v2"; $PROT bank --workers "$W" || finish 3
    echo "=== ЭТАП сверка v1/v2"; $PROT compare-v1 || { echo "ОСТАНОВ: сцены v2 ≠ v1"; finish 3; }
    echo "=== ЭТАП bank-check"; $PROT bank-check --workers "$W" || finish 3
    echo "=== ЭТАП dev-replay"; dev_replay "$DEV" || finish 3
    echo "=== ПЕРЕСБОРКА БАНКА v2 ЗАВЕРШЕНА"; finish 0 ;;
  bank) $PROT bank --workers "${ARG2:-6}"; finish $? ;;
  bank-check) $PROT bank-check --workers "${ARG2:-6}"; finish $? ;;
  legacy-smoke)
    DEV="${ARG2:-cuda:0}"; plan_a || finish 3
    for T in 0 8; do
      block cur "$DEV" "$T" 30 reports/k33/legacy/cur || finish 3
      block new "$DEV" "$T" 30 reports/k33/legacy/new legacy || finish 3
    done
    $PROT legacy-compare --cur reports/k33/legacy/cur --new reports/k33/legacy/new; finish $? ;;
  dev-replay) dev_replay "${ARG2:-cuda:0}"; finish $? ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
