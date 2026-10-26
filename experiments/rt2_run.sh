#!/bin/bash
# RoboTwin, baseline flat24 seed 0 (модельное окружение, A100).
#   bash experiments/rt2_run.sh selftest
#   bash experiments/rt2_run.sh run-baseline DEV
# run-baseline: самопроверки -> подготовка rt1 сверена (prep.json без
# проблем, контракт) -> micro из ЗАКРЕПЛЁННОГО reports/rt1/realcheck.json
# (сам не подбирается) -> fitcheck (готовый отчёт того же контракта
# переиспользуется; при провале — останов) -> обучение flat24 seed 0 ->
# SR — только если готов проверенный адаптер оценки (rt2_eval.py),
# иначе явная отметка «SR ещё не выполнен».
# Карта общая с чужой задачей: перед этапами — проверка, что свободной
# памяти хватает на пик real-check выбранного micro с запасом.
set -uo pipefail
MODE="${1:?нужен режим}"
DEV="${2:-cuda:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONUTF8=1
mkdir -p logs/rt2 reports/rt2
exec 9>logs/rt2/.rt2.lock
flock -n 9 || { echo "ОТКАЗ: другой rt2_run.sh уже выполняется"; exit 2; }
LOGF="logs/rt2/baseline_s0.log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит $(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
PY=python3
FACTORY="rt1_model:flat24"

selftests () {
  local m
  for m in rt1_common rt1_prepare rt1_model rt2_train; do
    $PY "experiments/${m}.py" --selftest >/dev/null 2>&1 \
      || { echo "ОТКАЗ: самопроверка ${m}"; return 1; }
  done
  echo "    самопроверки RT пройдены"
}
micro () {
  $PY - <<'PY'
import json, sys
r = json.load(open("reports/rt1/realcheck.json", encoding="utf-8"))
if r.get("problems") or not r.get("common_micro"):
    sys.exit(1)
print(r["common_micro"])
PY
}
mem_ok () {   # свободная память >= пик real-check (flat24, выбранный micro) / запас
  local need free idx="${DEV#cuda:}"
  need=$($PY -c "import json; r=json.load(open('reports/rt1/realcheck.json')); m=str(r['common_micro']); print(int(r['archs']['flat24']['micros'][m]['peak_gb']*1024/0.85))")
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx" | tr -d ' ')
  echo "    $DEV: свободно ${free} МиБ, нужно ~${need} МиБ"
  [ "$free" -ge "$need" ] || { echo "ОТКАЗ: на $DEV мало свободной памяти"; return 3; }
}
prep_ok () {
  $PY - <<'PY'
import sys
sys.path.insert(0, "experiments")
import json, rt1_data as rd, rt1_prepare as rp
p = json.load(open(rp.PREP, encoding="utf-8"))
rd.verify_prep(p, rp.INDEX, files=False)
print(f"    подготовка {p['prep_id']} сверена")
PY
}

case "$MODE" in
  selftest) selftests; finish $? ;;
  run-baseline)
    echo "=== ЭТАП самопроверки"; selftests || finish 1
    echo "=== ЭТАП подготовка"; prep_ok || finish 3
    echo "=== ЭТАП micro из real-check"
    MB="$(micro)" || { echo "ОТКАЗ: real-check не пройден или нет общего micro"; finish 3; }
    echo "    общий micro $MB (накопление $((32 / MB)))"
    echo "=== ЭТАП fitcheck"
    if ! $PY experiments/rt2_train.py check-fresh --factory "$FACTORY" --seed 0 --device "$DEV" --micro "$MB"; then
      mem_ok || finish 3
      $PY experiments/rt2_train.py fitcheck --factory "$FACTORY" --seed 0 --device "$DEV" --micro "$MB" \
        || { echo "ОСТАНОВ: fitcheck не пройден"; finish 3; }
    fi
    echo "=== ЭТАП обучение flat24 seed 0"
    mem_ok || finish 3
    $PY experiments/rt2_train.py train --factory "$FACTORY" --seed 0 --device "$DEV" --micro "$MB" || finish $?
    if [ -f experiments/rt2_eval.py ] && $PY experiments/rt2_eval.py --selftest >/dev/null 2>&1; then
      echo "=== ЭТАП пилот SR"; bash experiments/rt2_eval_run.sh pilot "$DEV" || finish $?
    else
      echo "=== ОБУЧЕНИЕ ЗАВЕРШЕНО; SR ещё не выполнен (адаптер оценки не готов)"
    fi
    finish 0 ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac
