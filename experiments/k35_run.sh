#!/bin/bash
# K-35 (K35-A): адаптивное ветвление временного дерева при фиксированном бюджете.
#   bash experiments/k35_run.sh selftest [cpu]
#   bash experiments/k35_run.sh prepare
#   bash experiments/k35_run.sh real-check DEV
#   bash experiments/k35_run.sh fitcheck DEV
#   bash experiments/k35_run.sh collector-smoke DEV  # c=0/17/35, N, A(init), S; t0 i20
#   bash experiments/k35_run.sh train DEV            # warmup 500 + main 4000 (продолжение с места)
#   bash experiments/k35_run.sh oracle DEV           # 36 срезов на строку панелей (шарды)
#   bash experiments/k35_run.sh oracle-gate
#   bash experiments/k35_run.sh selectors [cpu]
#   bash experiments/k35_run.sh selector-gate
#   bash experiments/k35_run.sh pilot DEV            # A/C/N/S × 10 задач, 40 блоков
#   bash experiments/k35_run.sh latency DEV          # только свободная карта, иначе latency_pending
#   bash experiments/k35_run.sh analyze
#   bash experiments/k35_run.sh run-a DEV            # вся цепочка с техническими стопами и гейтами
# Блок: reserve -> K-9h (один процесс на блок) -> finish (проверка, commit);
# отвергнутый блок — карантин и маркер, без автоматического повтора.
# Обучение и oracle требуют ≥ K35_TRAIN_FREE_MB (14336) свободной памяти
# карты, раскатки — ≥ K35_ROLL_FREE_MB (8192); задержка — занято ≤ 2048 МиБ.
# Место: K35_DISK_MB (3000), ждать до K35_WAIT_MIN = 720 мин, чужое не удаляется.
set -uo pipefail
MODE="${1:?нужен режим}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs/k35 reports/k35 data/k35
exec 9>logs/k35/.k35.lock
flock -n 9 || { echo "ОТКАЗ: другой k35_run.sh уже выполняется"; exit 2; }
LOGF="logs/k35/${MODE}_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $*, коммит $(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
PY=python3
PROT="$PY experiments/k35_protocol.py"
TRN="$PY experiments/k35_train.py"
BASIS="data/k15f/basis_s0.pt"

gpu_used_le () {   # DEV порог_занято_МиБ
  local idx="${1#cuda:}" used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  [ -n "$used" ] || { echo "ОТКАЗ: nvidia-smi не отвечает для $1"; return 3; }
  echo "    $1: занято $used МиБ (порог $2)"
  [ "$used" -le "$2" ] || { echo "    на $1 занято $used МиБ > $2"; return 3; }
}
gpu_free_ge () {   # DEV нужно_свободно_МиБ
  local idx="${1#cuda:}" free
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  [ -n "$free" ] || { echo "ОТКАЗ: nvidia-smi не отвечает для $1"; return 3; }
  echo "    $1: свободно $free МиБ (нужно $2)"
  [ "$free" -ge "$2" ] || { echo "ОТКАЗ: на $1 свободно $free МиБ < $2"; return 3; }
}
disk_wait () {
  local need="${K35_DISK_MB:-3000}" waited=0 free
  while :; do
    free=$(df -Pm reports | awk 'NR==2 {print $4}')
    [ "$free" -ge "$need" ] && return 0
    [ "$waited" -ge "${K35_WAIT_MIN:-720}" ] && { echo "ОТКАЗ: места $free МиБ < $need"; return 3; }
    echo "    места $free МиБ < $need — жду 10 мин ($(date '+%H:%M'))"; sleep 600; waited=$((waited + 10))
  done
}
status_of () {   # файл -> статус
  $PY -c "import json,sys; print(json.load(open(sys.argv[1])).get('status'))" "$1" 2>/dev/null
}
selftests () {
  local m
  for m in k35_topology k35_model k35_oracle k35_selector k35_train k35_policy \
           k35_harness k35_protocol k35_analyze k35_latency; do
    $PY "experiments/${m}.py" --selftest || { echo "ОТКАЗ: самопроверка $m"; return 1; }
  done
}
run_cell () {   # план вид метка задача блок debug DEV
  local P="$1" K="$2" L="$3" T="$4" B="$5" DEV="$7" D0 F CK rc=0
  local ARGS="--plan $P --label $L --task $T --start $B"
  D0="reports/k35/$K/$L"; F="$D0/t${T}_i${B}.json"
  if [ -f "$F" ]; then $PROT block --op finish $ARGS >/dev/null || { $PROT block --op finish $ARGS; return 3; }; return 0; fi
  disk_wait || return 3
  mkdir -p "$D0"
  $PROT block --op reserve $ARGS || return 3
  CK="$($PY -c "import json; print(json.load(open('$P'))['setup']['ckpt'])")" || return 3
  echo "    $K $L t$T i$B — $(date '+%H:%M:%S')"
  $PY experiments/k35_harness.py --k35-plan "$P" --ckpt "$CK" --task-suite 10 --n-envs 5 \
    --seed 101 --rollout-seed-mode fixed --ensemble off --horizon 8 --max-steps 600 \
    --waiting-steps 10 --device "$DEV" --save-actions --policy k15f --k15f-basis "$BASIS" \
    --arm-label "$L" --task-id "$T" --init-starts "$B" --run-tag "k35_$K" \
    --out "$D0/t${T}_i{i0}.json" > "$D0/t${T}_i${B}.log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    grep -E "ОТКАЗ|Error|error" "$D0/t${T}_i${B}.log" | tail -n 5
    echo "ОСТАНОВ: $K $L t$T i$B — код $rc (лог $D0/t${T}_i${B}.log; резерв при повторе спишется как авария)"
    return 3
  fi
  grep -E "успех [0-9]/5" "$D0/t${T}_i${B}.log" | tail -n 1
  $PROT block --op finish $ARGS || return 3
}
run_plan () {   # вид DEV
  local K="$1" DEV="$2" P out n want L T B DB
  P="reports/k35/plan_$K.json"
  $PROT plan --kind "$K" --device "$DEV" || return 3
  out="$($PROT cells --plan "$P")" || { echo "$out"; echo "ОТКАЗ: очередь $K"; return 3; }
  want="$(printf '%s\n' "$out" | sed -n '1s/^#//p')"
  n="$(printf '%s\n' "$out" | tail -n +2 | grep -c .)"
  [ -n "$want" ] && [ "$want" -gt 0 ] && [ "$n" = "$want" ] || { echo "ОТКАЗ: очередь $K: $n из ${want:-?}"; return 3; }
  echo "    план $K: блоков $n"
  gpu_free_ge "$DEV" "${K35_ROLL_FREE_MB:-8192}" || return 3
  while IFS=: read -r L T B DB; do
    run_cell "$P" "$K" "$L" "$T" "$B" "$DB" "$DEV" < /dev/null || return 3
  done < <(printf '%s\n' "$out" | tail -n +2)
}
real_check () {   # DEV
  $TRN check-fresh --kind real_check && return 0
  gpu_free_ge "$1" "${K35_TRAIN_FREE_MB:-14336}" || return 3
  $TRN real-check --device "$1"
}
fitcheck () {   # DEV
  $TRN check-fresh --kind fitcheck && return 0
  $TRN check-fresh --kind real_check || { echo "ОТКАЗ: real-check не свеж"; return 3; }
  gpu_free_ge "$1" "${K35_TRAIN_FREE_MB:-14336}" || return 3
  $TRN fitcheck --device "$1"
}
smoke () {   # DEV — готовые блоки переиспользуются, проверка всегда заново
  run_plan smoke "$1" || return 3
  $PROT smoke-check --plan reports/k35/plan_smoke.json --device "$1"
}
train () {   # DEV
  local rp="reports/k35/train_s0.json"
  if [ -f "$rp" ]; then
    $TRN train-valid && return 0
    echo "ОТКАЗ: $rp не соответствует текущему контракту"; return 3
  fi
  $PROT smoke-verify || { echo "ОТКАЗ: collector-smoke не пройден или устарел"; return 3; }
  gpu_free_ge "$1" "${K35_TRAIN_FREE_MB:-14336}" || return 3; disk_wait || return 3
  $TRN train --device "$1"
}
oracle () {   # DEV
  $TRN train-valid || { echo "ОТКАЗ: обучение K-35 не готово"; return 3; }
  gpu_free_ge "$1" "${K35_TRAIN_FREE_MB:-14336}" || return 3; disk_wait || return 3
  $PROT oracle --device "$1"
}
gate_or_stop () {   # файл статус_успеха
  local st; st="$(status_of "$1")"
  [ "$st" = "passed" ] && return 0
  echo "=== K-35: $1 — статус ${st:-нет}; цепочка остановлена (информационный или технический стоп)"
  return 4
}
pilot () {   # DEV
  $PROT offline-verify || { echo "ОТКАЗ: офлайн-гейты не пройдены или устарели — пилот запрещён"; return 3; }
  run_plan pilot "$1" || return 3
  $PROT pilot-gate --plan reports/k35/plan_pilot.json
}
latency () {   # DEV
  gpu_used_le "$1" 2048 || { echo "    latency_pending: карта занята"; return 0; }
  $PY experiments/k35_latency.py --device "$1"
}

case "$MODE" in
  selftest) selftests; finish $? ;;
  prepare) $PROT prepare; finish $? ;;
  real-check) real_check "${2:-cuda:0}"; finish $? ;;
  fitcheck) fitcheck "${2:-cuda:0}"; finish $? ;;
  collector-smoke) smoke "${2:-cuda:0}"; finish $? ;;
  train) train "${2:-cuda:0}"; finish $? ;;
  oracle) oracle "${2:-cuda:0}"; finish $? ;;
  oracle-gate) $PROT oracle-gate; finish $? ;;
  selectors) $PROT selectors; finish $? ;;
  selector-gate) $PROT selector-gate; finish $? ;;
  pilot) pilot "${2:-cuda:0}"; finish $? ;;
  latency) latency "${2:-cuda:0}"; finish $? ;;
  analyze) $PY experiments/k35_analyze.py; finish $? ;;
  run-a)
    DEV="${2:-cuda:0}"
    echo "=== ЭТАП самопроверки"; selftests || finish 1
    echo "=== ЭТАП подготовка"; $PROT prepare || finish 3
    echo "=== ЭТАП real-check"; real_check "$DEV" || { echo "ОСТАНОВ: real-check"; finish 3; }
    echo "=== ЭТАП fitcheck"; fitcheck "$DEV" || { echo "ОСТАНОВ: fitcheck"; finish 3; }
    echo "=== ЭТАП collector-smoke"; smoke "$DEV" || { echo "ОСТАНОВ: smoke"; finish 3; }
    echo "=== ЭТАП обучение общего декодера"; train "$DEV" || { echo "ОСТАНОВ: обучение"; finish 3; }
    echo "=== ЭТАП оракул"; oracle "$DEV" || { echo "ОСТАНОВ: оракул"; finish 3; }
    echo "=== ЭТАП гейт оракула"; $PROT oracle-gate || finish 3
    gate_or_stop reports/k35/oracle_gate.json || { $PY experiments/k35_analyze.py; finish 0; }
    echo "=== ЭТАП селекторы"; [ -f data/k35/selectors.pt ] || $PROT selectors || finish 3
    $PROT selector-gate || finish 3
    gate_or_stop reports/k35/selector_gate.json || { $PY experiments/k35_analyze.py; finish 0; }
    echo "=== ЭТАП SR-пилот"; pilot "$DEV" || { echo "ОСТАНОВ: пилот"; finish 3; }
    echo "=== ЭТАП задержка"; [ -f reports/k35/latency.json ] || latency "$DEV"
    echo "=== ЭТАП анализ"; $PY experiments/k35_analyze.py
    echo "=== K35-A ЗАВЕРШЁН $(date); статус пилота: $(status_of reports/k35/pilot_gate.json)"
    finish 0 ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
