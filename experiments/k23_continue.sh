#!/bin/bash
# K-23: продолжение пакетов B -> C с исправленным smoke (k23_smoke.py).
# k23_run.sh chain/train вызывает прежний k23_train.smoke, который ложно
# отказывает на структурно неиспользуемых параметрах; закреплённые файлы
# (k23_train.py, k23_run.sh, k23_protocol.py) не меняются.
#   bash experiments/k23_continue.sh SEED [DEV]
# Порядок: smoke (k23_smoke.py; пропуск, если уже пройден тем же кодом) ->
# обучение пары (k23_train.py train, новый запуск или продолжение с resume)
# -> план SR (k23_run.sh plan-repeat) -> раскатки и анализ (run-repeat).
# Манифест с SHA всех участвующих файлов: reports/k23/continue_s<SEED>.json.
set -uo pipefail
SEEDT="${1:?нужен training seed}"
DEV="${2:-cuda:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p logs/k23 reports/k23
exec 8>logs/k23/.k23_continue.lock
flock -n 8 || { echo "ОТКАЗ: k23_continue.sh уже выполняется"; exit 2; }
LOGF="logs/k23/continue_s${SEEDT}_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === seed $SEEDT, карта $DEV, коммит $(git rev-parse --short HEAD 2>/dev/null)"
finish () { echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
MAN="reports/k23/continue_s${SEEDT}.json"
python3 - "$MAN" "$SEEDT" <<'PY' || finish 1
import hashlib, json, os, sys, datetime
man, seed = sys.argv[1], int(sys.argv[2])
files = ["experiments/k23_continue.sh", "experiments/k23_smoke.py",
         "experiments/k23_train.py", "experiments/k23_run.sh",
         "experiments/k23_protocol.py", "experiments/k23_analyze.py",
         "experiments/k23_policy.py", "experiments/k23_harness.py",
         "experiments/k21_train.py", "experiments/k21_fastload.py"]
sha = {f: hashlib.sha1(open(f, "rb").read()).hexdigest() for f in files}
old = json.load(open(man)) if os.path.exists(man) else None
if old and old["code"] != sha:
    print("  ВНИМАНИЕ: код изменился с прошлого запуска продолжения:",
          [f for f in files if old["code"].get(f) != sha[f]])
json.dump(dict(seed=seed, updated=datetime.datetime.now().isoformat(
    timespec="seconds"), code=sha, history=(old or {}).get("history", [])
    + [sha]), open(man, "w"), indent=1)
print(f"  манифест {man}")
PY
SMOKE="reports/k23/smoke_s${SEEDT}_grads.json"
if python3 -c "import json,sys; r=json.load(open('$SMOKE')); sys.exit(0 if r['problems']==[] else 1)" 2>/dev/null; then
  echo "=== ЭТАП smoke: уже пройден ($SMOKE)"
else
  echo "=== ЭТАП smoke"
  python3 experiments/k23_smoke.py --seed "$SEEDT" --device "$DEV" --micro 32 \
    || { echo "ОСТАНОВ: smoke тренера"; finish 3; }
fi
echo "=== ЭТАП обучение пары seed $SEEDT"
python3 experiments/k23_train.py train --seed "$SEEDT" --device "$DEV" \
  --micro 32 --dtype float16 --expect auto || finish $?
echo "=== ЭТАП план повтора seed $SEEDT"
bash experiments/k23_run.sh plan-repeat "$SEEDT" "$DEV" || finish 3
echo "=== ЭТАП раскатки, анализ и сводка сидов seed $SEEDT"
bash experiments/k23_run.sh run-repeat "$SEEDT" "$DEV" || finish $?
echo "=== ПРОДОЛЖЕНИЕ K-23 ЗАВЕРШЕНО $(date); лог $LOGF"; finish 0
