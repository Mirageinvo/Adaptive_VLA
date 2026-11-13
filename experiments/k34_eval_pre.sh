#!/bin/bash
# K-34: основная оценка ДО замера задержки — решение пользователя
# (руководителя исследования) от 2026-11-14. k34_run.sh и k34_protocol.py не
# меняются (закреплены по SHA в плане пилота); здесь те же функции блоков,
# та же блокировка logs/k34/.k34.lock и тот же журнал.
#   bash experiments/k34_eval_pre.sh DEV
# Порядок: гейт пилота (passed либо чистый awaiting_latency при условиях 1–2,
# k34_prelatency.py; решение записывается) -> проверка четырёх обучений
# (train-valid) -> план eval и 480 блоков -> анализ (latency_pending).
# Задержку затем снимать на свободной карте: k34_run.sh pilot DEV, затем
# k34_run.sh latency eval DEV и k34_run.sh analyze.
set -uo pipefail
DEV="${1:-cuda:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p logs/k34 reports/k34 data/k34
exec 9>logs/k34/.k34.lock
flock -n 9 || { echo "ОТКАЗ: другой k34_run.sh уже выполняется"; exit 2; }
LOGF="logs/k34/eval_pre_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === основная оценка до замера задержки, $DEV, коммит $(git rev-parse --short HEAD 2>/dev/null)"
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
$PY experiments/k34_prelatency.py --selftest >/dev/null || { echo "ОТКАЗ: самопроверка k34_prelatency"; finish 1; }
echo "=== ЭТАП гейт пилота (допускается awaiting_latency при условиях 1–2)"
$PY experiments/k34_prelatency.py verify reports/k34/plan_pilot.json || finish 3
echo "=== ЭТАП проверка четырёх обучений"
for sm in "0:parents_attn24" "1:parents_attn24" "0:flat_attn24" "1:flat_attn24"; do
  $TRN train-valid --seed "${sm%%:*}" --mode "${sm#*:}" || { echo "ОТКАЗ: обучение ${sm#*:} s${sm%%:*}"; finish 3; }
done
echo "=== ЭТАП оценка: 480 блоков"; run_plan eval "$DEV" || finish 3
echo "=== ЭТАП анализ (задержка — позже, на свободной карте)"; $PY experiments/k34_analyze.py || finish 3
echo "=== ОСНОВНАЯ ОЦЕНКА K-34 ЗАВЕРШЕНА $(date); задержка: k34_run.sh pilot/latency на свободной карте"
finish 0
