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
# Вторая очередь (M = A ccmplanner | B cds2; один процесс K-9h = один блок):
#   bash experiments/k33_run.sh plan M              # план машины (закрепляется)
#   bash experiments/k33_run.sh legacy6 DEV         # A: 6 рук, t0/t8 i30, старый путь против адаптера
#   bash experiments/k33_run.sh dev DEV             # A: 50 dev-сцен × 6 рук + проверка
#   bash experiments/k33_run.sh repeat DEV          # A: холодные повторы dev t0/6/8 × 6 рук
#   bash experiments/k33_run.sh port DEV            # B: dev t0/6/8 × 6 рук (перенос) + проверка
#   bash experiments/k33_run.sh forecast M          # время/диск SR по фактическим dev-процессам
#   bash experiments/k33_run.sh sr M DEV            # SR машины после гейтов (продолжается с места)
#   bash experiments/k33_run.sh progress M | analyze | latency DEV
#   bash experiments/k33_run.sh eval M DEV          # selftest -> plan -> гейты машины -> forecast -> sr
# Все физические шаги проверок — в журнале data/k33/ledger.json (checks), SR —
# scope sr с лимитом машины (1800 блоков × 3050).
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
  for m in k33_bank k33_harness k33_protocol k33_diag k33_policy k33_eval \
           k33_analyze k33_latency; do
    $PY "experiments/${m}.py" --selftest || return 1
  done
}

# --- вторая очередь ---------------------------------------------------------------
EV="$PY experiments/k33_eval.py"
BASIS="data/k15f/basis_s0.pt"
MACH="" PLAN=""
use_plan () {   # M
  MACH="$1"; PLAN="reports/k33/plan_${MACH}.json"
  [ -f "$PLAN" ] || { echo "ОТКАЗ: нет плана $PLAN (bash experiments/k33_run.sh plan $MACH)"; return 3; }
  CKPT="$($PY -c "import json; print(json.load(open('$PLAN'))['setup']['ckpt'])")" || return 3
  [ -d "$CKPT" ] || { echo "ОТКАЗ: нет снимка BAR $CKPT"; return 3; }
}
COMMON0 () {   # DEV
  echo "--ckpt $CKPT --task-suite 10 --n-envs 5 --seed 101 --rollout-seed-mode fixed \
--ensemble off --horizon 8 --max-steps 600 --waiting-steps 10 --device $1 --save-actions"
}
arm_args () { $PY -c "import sys; sys.path.insert(0,'experiments'); import k33_policy as p; print(' '.join(p.k9h_args('$1')))"; }
k26_plan () { $PY -c "import sys; sys.path.insert(0,'experiments'); import k33_policy as p; print(p.k26_plan_path('$1'))"; }
gpu_free () {   # DEV
  local idx="${1#cuda:}" used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  [ -n "$used" ] || { echo "ОТКАЗ: nvidia-smi не отвечает для $1"; return 3; }
  [ "$used" -le 2048 ] || { echo "ОТКАЗ: на $1 занято $used МиБ"; return 3; }
}
disk_wait () {   # ждать места до K33_WAIT_MIN (720) минут; чужое не удаляется
  local need="${K33_DISK_MB:-3000}" waited=0 free
  while :; do
    free=$(df -Pm reports | awk 'NR==2 {print $4}')
    [ "$free" -ge "$need" ] && return 0
    [ "$waited" -ge "${K33_WAIT_MIN:-720}" ] && { echo "ОТКАЗ: места $free МиБ < $need за $waited мин"; return 3; }
    echo "    места $free МиБ < $need — жду 10 мин ($(date '+%H:%M'))"; sleep 600; waited=$((waited + 10))
  done
}
run_cell () {   # фаза рука задача блок режим space debug DEV
  local PH="$1" A="$2" T="$3" S="$4" MD="$5" SP="$6" DB="$7" DEV="$8" F D0 t0 rc=0
  local ARGS="--plan $PLAN --phase $PH --arm $A --task $T --start $S"
  D0="reports/k33/$PH/$MACH/$A"; F="$D0/t${T}_i${S}.json"
  if [ -f "$F" ]; then $EV block --op finish $ARGS >/dev/null || { $EV block --op finish $ARGS; return 3; }; return 0; fi
  disk_wait || return 3
  mkdir -p "$D0"
  t0=$(date +%s.%N)
  $EV block --op reserve $ARGS || return 3
  echo "    $PH $A t$T i$S — $(date '+%H:%M:%S')"
  if [ "$PH" = legacy_cur ]; then
    case "$A" in
      B) # shellcheck disable=SC2046
         $PY experiments/k21_main_harness.py --k21-main-plan "$($PY -c "import sys; sys.path.insert(0,'experiments'); import k21_main_protocol as p; print(p.plan_path('A'))")" \
           $(COMMON0 "$DEV") $(arm_args B) --run-tag k33_legacy --task-id "$T" --init-starts "$S" \
           --out "$D0/t${T}_i{i0}.json" > "$D0/t${T}_i${S}.log" 2>&1 || rc=$? ;;
      BC) # shellcheck disable=SC2046
         $PY experiments/k21_cached_harness.py --cached-plan reports/k21/plan_cached.json \
           $(COMMON0 "$DEV") $(arm_args BC) --run-tag k33_legacy --task-id "$T" --init-starts "$S" \
           --out "$D0/t${T}_i{i0}.json" > "$D0/t${T}_i${S}.log" 2>&1 || rc=$? ;;
      *) # shellcheck disable=SC2046
         $PY experiments/k26_harness.py --k26-plan "$(k26_plan "$A")" \
           $(COMMON0 "$DEV") $(arm_args "$A") --run-tag k33_legacy --task-id "$T" --init-starts "$S" \
           --out "$D0/t${T}_i{i0}.json" > "$D0/t${T}_i${S}.log" 2>&1 || rc=$? ;;
    esac
  else
    local SPA=""; [ "$SP" != "-" ] && SPA="--k33-space $SP"
    # shellcheck disable=SC2046,SC2086
    $PY experiments/k33_harness.py --k33-mode "$MD" $SPA --k33-plan "$PLAN" --k33-arm "$A" \
      --k33-debug "$DB" $(COMMON0 "$DEV") $(arm_args "$A") --run-tag "k33_$PH" \
      --task-id "$T" --init-starts "$S" --out "$D0/t${T}_i{i0}.json" > "$D0/t${T}_i${S}.log" 2>&1 || rc=$?
  fi
  if [ "$rc" -ne 0 ]; then
    grep -E "ОТКАЗ|Error|error" "$D0/t${T}_i${S}.log" | tail -n 5
    echo "ОСТАНОВ: $PH $A t$T i$S — код $rc (лог $D0/t${T}_i${S}.log; резерв останется открытым и при повторе будет списан как авария)"
    return 3
  fi
  grep -E "успех [0-9]/5" "$D0/t${T}_i${S}.log" | tail -n 1
  $EV block --op finish $ARGS || return 3
  # время всей ячейки: резерв, сверка источников, загрузка модели, раскатка,
  # проверка блока
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$PH" "$A" "$T" "$S" "$t0" "$(date +%s.%N)" >> "reports/k33/timing_${MACH}.tsv"
}
run_phase () {   # фаза DEV
  local PH="$1" DEV="$2" out n want c A T S MD SP DB
  # очередь — отдельной командой с проверкой кода и полноты (пустая или
  # усечённая очередь — отказ, а не «успех на нуле блоков»)
  out="$($EV cells --plan "$PLAN" --phase "$PH")" || { echo "$out"; echo "ОТКАЗ: очередь фазы $PH"; return 3; }
  want="$(printf '%s\n' "$out" | sed -n '1s/^#//p')"
  n="$(printf '%s\n' "$out" | tail -n +2 | grep -c .)"
  [ -n "$want" ] && [ "$want" -gt 0 ] && [ "$n" = "$want" ] || { echo "ОТКАЗ: очередь фазы $PH: $n из ${want:-?}"; return 3; }
  echo "    фаза $PH: ячеек $n"
  while IFS=: read -r A T S MD SP DB; do
    run_cell "$PH" "$A" "$T" "$S" "$MD" "$SP" "$DB" "$DEV" < /dev/null || return 3
  done < <(printf '%s\n' "$out" | tail -n +2)
}
gates_a () {   # DEV
  echo "=== ЭТАП legacy-smoke 6 рук (t0/t8 i30: старый путь и адаптер)"
  run_phase legacy_cur "$1" || return 3; run_phase legacy_new "$1" || return 3
  $EV legacy-compare --plan "$PLAN" || return 3
  echo "=== ЭТАП development-smoke (50 сцен × 6 рук)"
  run_phase dev "$1" || return 3; $EV dev-check --plan "$PLAN" --phase dev || return 3
  echo "=== ЭТАП холодные повторы dev t0/6/8 × 6 рук"
  run_phase repeat "$1" || return 3; $EV repeat-compare --plan "$PLAN" || return 3
}
gates_b () {   # DEV
  echo "=== ЭТАП перенос на B: dev t0/6/8 × 6 рук"
  run_phase port "$1" || return 3; $EV dev-check --plan "$PLAN" --phase port || return 3
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
  plan) $EV plan --machine "${ARG2:?нужна машина A|B}"; finish $? ;;
  legacy6) use_plan A || finish 3; gpu_free "${ARG2:-cuda:0}" || finish 3
    run_phase legacy_cur "${ARG2:-cuda:0}" || finish 3; run_phase legacy_new "${ARG2:-cuda:0}" || finish 3
    $EV legacy-compare --plan "$PLAN"; finish $? ;;
  dev) use_plan A || finish 3; gpu_free "${ARG2:-cuda:0}" || finish 3
    run_phase dev "${ARG2:-cuda:0}" || finish 3; $EV dev-check --plan "$PLAN" --phase dev; finish $? ;;
  repeat) use_plan A || finish 3; gpu_free "${ARG2:-cuda:0}" || finish 3
    run_phase repeat "${ARG2:-cuda:0}" || finish 3; $EV repeat-compare --plan "$PLAN"; finish $? ;;
  port) use_plan B || finish 3; gpu_free "${ARG2:-cuda:0}" || finish 3
    run_phase port "${ARG2:-cuda:0}" || finish 3; $EV dev-check --plan "$PLAN" --phase port; finish $? ;;
  forecast) use_plan "${ARG2:?нужна машина A|B}" || finish 3; $EV forecast --plan "$PLAN"; finish $? ;;
  progress) use_plan "${ARG2:?нужна машина A|B}" || finish 3; $EV progress --plan "$PLAN"; finish $? ;;
  sr) use_plan "${ARG2:?нужна машина A|B}" || finish 3; DEV="${3:-cuda:0}"
    $EV gates --plan "$PLAN" || finish 3; gpu_free "$DEV" || finish 3
    run_phase sr "$DEV" || finish 3; $EV progress --plan "$PLAN"; finish 0 ;;
  analyze) $PY experiments/k33_analyze.py; finish $? ;;
  latency) DEV="${ARG2:-cuda:0}"; gpu_free "$DEV" || { echo "    latency_pending: карта занята"; finish 3; }
    $PY experiments/k33_latency.py --device "$DEV"; finish $? ;;
  eval)
    M="${ARG2:?нужна машина A|B}"; DEV="${3:-cuda:0}"
    echo "=== ЭТАП самопроверки"; selftests || finish 1
    echo "=== ЭТАП план машины $M"; $EV plan --machine "$M" || finish 3
    use_plan "$M" || finish 3; gpu_free "$DEV" || finish 3
    if [ "$M" = A ]; then gates_a "$DEV" || finish 3; else gates_b "$DEV" || finish 3; fi
    echo "=== ЭТАП прогноз"; $EV forecast --plan "$PLAN" || finish 3
    echo "=== ЭТАП гейты SR"; $EV gates --plan "$PLAN" || finish 3
    echo "=== ЭТАП SR машины $M (1800 блоков)"; run_phase sr "$DEV" || finish 3
    $EV progress --plan "$PLAN"
    echo "=== SR МАШИНЫ $M ЗАВЕРШЁН $(date)"; finish 0 ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
