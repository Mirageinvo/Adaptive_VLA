#!/bin/bash
# RoboTwin, этап 2: официальная оценка обученной модели.
#   bash experiments/rt2_eval_run.sh smoke DEV [SEED] [ARCH]   # lift_pot и open_laptop, по 1 эпизоду
#   bash experiments/rt2_eval_run.sh pilot DEV [SEED] [ARCH]   # 8 задач × 10 состояний = 80 эпизодов
# Сервер политики (окружение vla, rt2_eval.py serve на DEV) поднимается один
# раз; клиент — официальный RoboTwin scripts/eval_policy_xpolicylab.py в
# окружении RoboTwin, по одной задаче подряд (seed 0: состояния от 100000,
# экспертная проверка, task_config demo_clean, частота 15). Журнал эпизодов —
# reports/rt2/eval/<режим>_s<SEED>_<ARCH>/<задача>/episodes.jsonl. Задача
# получает отметку завершения (done.json) только после кода клиента 0, чистого
# лога клиента и полной проверки журнала против манифеста модели; при
# повторном запуске пропускается только задача с такой отметкой, остальные —
# в карантин и заново.
# Не одновременно с обучением; flock. Окружения:
#   VLA_PY    — python окружения vla (по умолчанию /home/malinin_aa/envs/vla/bin/python)
#   RT_ENV_SH — скрипт активации окружения RoboTwin (по умолчанию /home/malinin_aa/robotwin_env.sh)
set -uo pipefail
MODE="${1:?нужен режим smoke|pilot}"
DEV="${2:-cuda:0}"
SEED="${3:-0}"
ARCH="${4:-flat24}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
VLA_PY="${VLA_PY:-/home/malinin_aa/envs/vla/bin/python}"
RT_ENV_SH="${RT_ENV_SH:-/home/malinin_aa/robotwin_env.sh}"
: "${ROBOTWIN_ROOT:?нужен ROBOTWIN_ROOT}"
export HF_HUB_OFFLINE=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
mkdir -p logs/rt2 reports/rt2/eval
exec 8>logs/rt2/.eval.lock
flock -n 8 || { echo "ОТКАЗ: другая оценка RoboTwin уже выполняется"; exit 2; }
case "$MODE" in
  smoke) TASKS="lift_pot open_laptop"; N=1 ;;
  pilot) TASKS="$($VLA_PY experiments/rt2_eval.py tasks)"; N=10 ;;
  *) echo "режим $MODE неизвестен"; exit 2 ;;
esac
LOGD="reports/rt2/eval/${MODE}_s${SEED}_${ARCH}"
LOGF="logs/rt2/eval_${MODE}_s${SEED}_${ARCH}.log"
mkdir -p "$LOGD/quarantine"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === $MODE, карта $DEV, seed $SEED, $ARCH, коммит $(git rev-parse --short HEAD 2>/dev/null)"
finish () { [ -n "${SPID:-}" ] && kill "$SPID" 2>/dev/null; echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
trap 'finish 3' INT TERM
if pgrep -f "rt2_train.py (train|fitcheck)" >/dev/null; then
  echo "ОТКАЗ: идёт обучение RoboTwin — оценка после него"; finish 3
fi
[ -f "$RT_ENV_SH" ] || { echo "ОТКАЗ: нет $RT_ENV_SH (активация окружения RoboTwin)"; finish 3; }
$VLA_PY experiments/rt2_policy.py --selftest >/dev/null 2>&1 || { echo "ОТКАЗ: самопроверка rt2_policy"; finish 1; }
$VLA_PY experiments/rt2_eval.py --selftest >/dev/null 2>&1 || { echo "ОТКАЗ: самопроверка rt2_eval"; finish 1; }

echo "=== ЭТАП ожидаемая модель (сверка журнала до запуска)"
$VLA_PY experiments/rt2_eval.py expect --seed "$SEED" --arch "$ARCH" --log-dir "$LOGD" \
  || { echo "ОТКАЗ: модель не совпадает с журналом $LOGD или чекпойнт не прошёл сверку"; finish 3; }

PORT="$($VLA_PY -c 'import socket; s=socket.socket(); s.bind(("localhost",0)); print(s.getsockname()[1])')"
echo "=== ЭТАП сервер политики (порт $PORT)"
$VLA_PY experiments/rt2_eval.py serve --seed "$SEED" --arch "$ARCH" --device "$DEV" \
  --port "$PORT" --log-dir "$LOGD" > "logs/rt2/server_${MODE}_s${SEED}_${ARCH}.log" 2>&1 &
SPID=$!
for i in $(seq 1 240); do
  kill -0 "$SPID" 2>/dev/null || { echo "ОТКАЗ: сервер политики завершился"; tail -n 30 "logs/rt2/server_${MODE}_s${SEED}_${ARCH}.log"; finish 3; }
  $VLA_PY -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('localhost',$PORT)))" && break
  sleep 5
done
$VLA_PY -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('localhost',$PORT)))" \
  || { echo "ОТКАЗ: сервер не открыл порт за 20 мин"; finish 3; }
grep -E "загружено" "logs/rt2/server_${MODE}_s${SEED}_${ARCH}.log" | tail -n 1

for T in $TASKS; do
  if $VLA_PY experiments/rt2_eval.py is-done --log-dir "$LOGD" --tasks "$T" --n "$N" >/dev/null 2>&1; then
    echo "    $T: уже завершена и проверена ($N эпизодов)"; continue
  fi
  if [ -d "$LOGD/$T" ]; then
    Q="$LOGD/quarantine/$(date +%Y%m%dT%H%M%S)_$T"; mv "$LOGD/$T" "$Q"
    echo "    $T: неполный журнал -> $Q, задача заново"
  fi
  echo "=== ЭТАП $T: $N эпизодов, $(date '+%H:%M:%S')"
  rc=0
  bash -c "source '$RT_ENV_SH' && cd '$ROBOTWIN_ROOT' && PYTHONUTF8=1 python scripts/eval_policy_xpolicylab.py \
    --task_name '$T' --policy_name adaptive_vla_rt2 --port '$PORT' --host localhost --protocol ws \
    --seed 0 --test_num '$N' --task_config demo_clean --frequency 15 --expert_check true \
    --additional_info 'ckpt_name=rt2_s${SEED}_${ARCH},action_type=joint'" \
    > "logs/rt2/client_${MODE}_${T}.log" 2>&1 || rc=$?
  grep -E "Success rate|Final success" "logs/rt2/client_${MODE}_${T}.log" | tail -n 2
  [ $rc -eq 0 ] || { echo "ОСТАНОВ: клиент $T код $rc (logs/rt2/client_${MODE}_${T}.log)"; finish 3; }
  if ! $VLA_PY experiments/rt2_eval.py client-check --log "logs/rt2/client_${MODE}_${T}.log"; then
    Q="$LOGD/quarantine/$(date +%Y%m%dT%H%M%S)_${T}_rollout_error"; mv "$LOGD/$T" "$Q" 2>/dev/null
    echo "ОСТАНОВ: $T технически непригодна (ошибка раскатки), журнал -> $Q"; finish 3
  fi
  kill -0 "$SPID" 2>/dev/null || { echo "ОСТАНОВ: сервер политики упал"; tail -n 30 "logs/rt2/server_${MODE}_s${SEED}_${ARCH}.log"; finish 3; }
  $VLA_PY experiments/rt2_eval.py mark-done --log-dir "$LOGD" --tasks "$T" --n "$N" \
    --log "logs/rt2/client_${MODE}_${T}.log" \
    || { echo "ОСТАНОВ: журнал $T не прошёл проверку — отметка завершения не поставлена"; finish 3; }
done
echo "=== ЭТАП сводка"
$VLA_PY experiments/rt2_eval.py summarize --log-dir "$LOGD" --n "$N" \
  $( [ "$MODE" = smoke ] && echo "--tasks $TASKS" ) \
  --out "reports/rt2/${MODE}_s${SEED}_${ARCH}.json" || finish 3
finish 0
