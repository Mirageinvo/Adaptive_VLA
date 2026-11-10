#!/bin/bash
# K-34: исполнимые выходы дерева после слоёв 8/16/24 (один checkpoint).
#   bash experiments/k34_run.sh selftest [cpu]
#   bash experiments/k34_run.sh prepare
#   bash experiments/k34_run.sh real-check DEV
#   bash experiments/k34_run.sh fitcheck DEV
#   bash experiments/k34_run.sh collector-smoke DEV   # T0 init, d8/16/24, t0 i20
#   bash experiments/k34_run.sh train SEED MODE DEV   # одно дообучение (продолжение с места)
#   bash experiments/k34_run.sh pilot DEV             # обучение T0 -> 40 блоков -> задержка -> гейт
#   bash experiments/k34_run.sh eval DEV              # после passed: T1, F0, F1 -> 480 блоков -> анализ
#   bash experiments/k34_run.sh analyze | latency KIND DEV
#   bash experiments/k34_run.sh run DEV               # всё по порядку с остановками
# Блок: reserve -> K-9h (один процесс на блок) -> finish (проверка, commit);
# отвергнутый блок — карантин и маркер, без автоматического повтора.
# Перед GPU-этапами — проверка карты (K34_GPU_MAX_MB, 2048) и места
# (K34_DISK_MB, 3000; ждать до K34_WAIT_MIN = 720 мин, чужое не удаляется).
set -uo pipefail
MODE="${1:?нужен режим}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p logs/k34 reports/k34 data/k34
exec 9>logs/k34/.k34.lock
flock -n 9 || { echo "ОТКАЗ: другой k34_run.sh уже выполняется"; exit 2; }
LOGF="logs/k34/${MODE}_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $*, коммит $(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
PY=python3
PROT="$PY experiments/k34_protocol.py"
TRN="$PY experiments/k34_train.py"
BASIS="data/k15f/basis_s0.pt"

gpu_free () {   # DEV [порог]
  local idx="${1#cuda:}" used max="${2:-${K34_GPU_MAX_MB:-2048}}"
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  [ -n "$used" ] || { echo "ОТКАЗ: nvidia-smi не отвечает для $1"; return 3; }
  echo "    $1: занято $used МиБ (порог $max)"
  [ "$used" -le "$max" ] || { echo "ОТКАЗ: на $1 занято $used МиБ > $max"; return 3; }
}
disk_wait () {
  local need="${K34_DISK_MB:-3000}" waited=0 free
  while :; do
    free=$(df -Pm reports | awk 'NR==2 {print $4}')
    [ "$free" -ge "$need" ] && return 0
    [ "$waited" -ge "${K34_WAIT_MIN:-720}" ] && { echo "ОТКАЗ: места $free МиБ < $need"; return 3; }
    echo "    места $free МиБ < $need — жду 10 мин ($(date '+%H:%M'))"; sleep 600; waited=$((waited + 10))
  done
}
selftests () {
  local m
  for m in k34_model k34_train k34_policy k34_protocol k34_harness k34_analyze k34_latency; do
    $PY "experiments/${m}.py" --selftest || { echo "ОТКАЗ: самопроверка $m"; return 1; }
  done
}
run_cell () {   # план вид метка задача блок debug DEV
  local P="$1" K="$2" L="$3" T="$4" B="$5" DEV="$7" D0 F CK rc=0
  local ARGS="--plan $P --label $L --task $T --start $B"
  D0="reports/k34/$K/$L"; F="$D0/t${T}_i${B}.json"
  if [ -f "$F" ]; then $PROT block --op finish $ARGS >/dev/null || { $PROT block --op finish $ARGS; return 3; }; return 0; fi
  disk_wait || return 3
  mkdir -p "$D0"
  $PROT block --op reserve $ARGS || return 3
  CK="$($PY -c "import json; print(json.load(open('$P'))['setup']['ckpt'])")" || return 3
  echo "    $K $L t$T i$B — $(date '+%H:%M:%S')"
  $PY experiments/k34_harness.py --k34-plan "$P" --ckpt "$CK" --task-suite 10 --n-envs 5 \
    --seed 101 --rollout-seed-mode fixed --ensemble off --horizon 8 --max-steps 600 \
    --waiting-steps 10 --device "$DEV" --save-actions --policy k15f --k15f-basis "$BASIS" \
    --arm-label "$L" --task-id "$T" --init-starts "$B" --run-tag "k34_$K" \
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
  P="reports/k34/plan_$K.json"
  $PROT plan --kind "$K" --device "$DEV" || return 3
  out="$($PROT cells --plan "$P")" || { echo "$out"; echo "ОТКАЗ: очередь $K"; return 3; }
  want="$(printf '%s\n' "$out" | sed -n '1s/^#//p')"
  n="$(printf '%s\n' "$out" | tail -n +2 | grep -c .)"
  [ -n "$want" ] && [ "$want" -gt 0 ] && [ "$n" = "$want" ] || { echo "ОТКАЗ: очередь $K: $n из ${want:-?}"; return 3; }
  echo "    план $K: блоков $n"
  gpu_free "$DEV" || return 3
  while IFS=: read -r L T B DB; do
    run_cell "$P" "$K" "$L" "$T" "$B" "$DB" "$DEV" < /dev/null || return 3
  done < <(printf '%s\n' "$out" | tail -n +2)
}
real_check () {   # DEV
  $TRN check-fresh real_check && return 0
  gpu_free "$1" || return 3
  $TRN real-check --device "$1"
}
fitcheck () {   # DEV
  $TRN check-fresh fitcheck && return 0
  gpu_free "$1" || return 3
  $TRN fitcheck --device "$1"
}
train_one () {   # SEED MODE DEV
  local rp="reports/k34/train_s$1_$2.json"
  [ -f "$rp" ] && { echo "    обучение $2 s$1: готово ($rp)"; return 0; }
  gpu_free "$3" || return 3; disk_wait || return 3
  $TRN train --seed "$1" --mode "$2" --device "$3"
}
smoke () {   # DEV
  [ -f reports/k34/smoke_check.json ] && $PY -c "import json,sys; sys.exit(0 if json.load(open('reports/k34/smoke_check.json')).get('ok') else 1)" \
    && { echo "    collector-smoke: пройден"; return 0; }
  run_plan smoke "$1" || return 3
  $PROT smoke-check --plan reports/k34/plan_smoke.json --device "$1"
}
latency () {   # KIND DEV
  gpu_free "$2" 2048 || { echo "    latency_pending: карта занята"; return 0; }
  $PY experiments/k34_latency.py --kind "$1" --device "$2"
}
pilot () {   # DEV
  echo "=== ЭТАП обучение T0 (parents_attn24 s0)"; train_one 0 parents_attn24 "$1" || return 3
  echo "=== ЭТАП пилот: 40 блоков"; run_plan pilot "$1" || return 3
  echo "=== ЭТАП задержка пилота"; [ -f reports/k34/latency_pilot.json ] || latency pilot "$1"
  echo "=== ЭТАП гейт пилота"; $PROT pilot-gate --plan reports/k34/plan_pilot.json
}
gate_passed () { $PY -c "import json,sys; sys.exit(0 if json.load(open('reports/k34/pilot_gate.json')).get('status')=='passed' else 1)" 2>/dev/null; }
evaluation () {   # DEV
  local DEV="$1" sm s m
  gate_passed || { echo "ОТКАЗ: гейт пилота не пройден (reports/k34/pilot_gate.json)"; return 3; }
  for sm in "1:parents_attn24" "0:flat_attn24" "1:flat_attn24"; do
    s="${sm%%:*}"; m="${sm#*:}"
    echo "=== ЭТАП обучение $m s$s"; train_one "$s" "$m" "$DEV" || return 3
  done
  echo "=== ЭТАП оценка: 480 блоков"; run_plan eval "$DEV" || return 3
  echo "=== ЭТАП анализ"; $PY experiments/k34_analyze.py || return 3
  echo "=== ЭТАП задержка"; [ -f reports/k34/latency_eval.json ] || latency eval "$DEV"
  echo "=== ЭТАП итоговый анализ"; $PY experiments/k34_analyze.py
}

case "$MODE" in
  selftest) selftests; finish $? ;;
  prepare) $PROT prepare; finish $? ;;
  real-check) real_check "${2:-cuda:0}"; finish $? ;;
  fitcheck) fitcheck "${2:-cuda:0}"; finish $? ;;
  collector-smoke) smoke "${2:-cuda:0}"; finish $? ;;
  train) train_one "${2:?seed}" "${3:?mode}" "${4:-cuda:0}"; finish $? ;;
  pilot) pilot "${2:-cuda:0}"; finish $? ;;
  eval) evaluation "${2:-cuda:0}"; finish $? ;;
  analyze) $PY experiments/k34_analyze.py; finish $? ;;
  latency) latency "${2:?pilot|eval}" "${3:-cuda:0}"; finish $? ;;
  run)
    DEV="${2:-cuda:0}"
    echo "=== ЭТАП самопроверки"; selftests || finish 1
    echo "=== ЭТАП подготовка"; $PROT prepare || finish 3
    echo "=== ЭТАП real-check"; real_check "$DEV" || { echo "ОСТАНОВ: real-check"; finish 3; }
    echo "=== ЭТАП fitcheck"; fitcheck "$DEV" || { echo "ОСТАНОВ: fitcheck"; finish 3; }
    echo "=== ЭТАП collector-smoke"; smoke "$DEV" || { echo "ОСТАНОВ: smoke"; finish 3; }
    pilot "$DEV" || finish 3
    if gate_passed; then evaluation "$DEV" || finish 3; echo "=== K-34 ЗАВЕРШЁН $(date)"; finish 0; fi
    echo "=== K-34: пилот не дал статуса passed — см. reports/k34/pilot_gate.json (информационный стоп или awaiting_latency)"
    finish 0 ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
