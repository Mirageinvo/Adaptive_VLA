#!/bin/bash
# RoboTwin, этап 3: явный банк случаев и тест n_exec = 4 против 8 на flat24
# seed 0 (тот же чекпойнт, чанк 20, частота 15, предобработка).
#   bash experiments/rt3_run.sh bank                 # CPU/рендер: банк 8×5
#   bash experiments/rt3_run.sh eval DEV NEXEC       # 40 эпизодов по банку
#   bash experiments/rt3_run.sh compare              # 4 против 8, парно
# Банк: первые 5 seed пилота flat24 RT2 каждой задачи (по возрастанию),
# прошедшие ОДНУ экспертную проверку сейчас и воспроизводимые (два свежих
# setup дают одинаковые отпечатки); при оценке случай не заменяется: если
# начальное состояние не совпало с банком — технический отказ. Сервер —
# rt2_eval.py serve (окружение vla) с --n-exec; клиент — rt3_cases.py eval
# (окружение RoboTwin, функции официального клиента). Не одновременно с
# обучением; flock.
set -uo pipefail
MODE="${1:?нужен режим bank|eval|compare}"
DEV="${2:-cuda:0}"
NEXEC="${3:-8}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
VLA_PY="${VLA_PY:-/home/malinin_aa/envs/vla/bin/python}"
RT_ENV_SH="${RT_ENV_SH:-/home/malinin_aa/robotwin_env.sh}"
: "${ROBOTWIN_ROOT:?нужен ROBOTWIN_ROOT}"
export HF_HUB_OFFLINE=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
BANK="$ROOT/reports/rt3/bank.json"
SRC="$ROOT/reports/rt2/eval/pilot_s0_flat24"
mkdir -p logs/rt3 reports/rt3/eval
exec 8>logs/rt3/.rt3.lock
flock -n 8 || { echo "ОТКАЗ: другой rt3_run.sh уже выполняется"; exit 2; }
LOGF="logs/rt3/${MODE}${NEXEC:+_$NEXEC}_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOGF") 2>&1
echo "=== СТАРТ $(date) === $MODE, карта $DEV, n_exec $NEXEC, коммит $(git rev-parse --short HEAD 2>/dev/null)"
SPID=""
finish () { [ -n "$SPID" ] && kill "$SPID" 2>/dev/null; echo "=== КОНЕЦ $(date), код $1 ==="; exit "$1"; }
trap 'finish 3' INT TERM
[ -f "$RT_ENV_SH" ] || { echo "ОТКАЗ: нет $RT_ENV_SH"; finish 3; }
if pgrep -f "rt2_train.py (train|fitcheck)" >/dev/null; then echo "ОТКАЗ: идёт обучение RoboTwin"; finish 3; fi
TASKS="$($VLA_PY experiments/rt2_eval.py tasks)"

case "$MODE" in
  bank)
    bash -c "source '$RT_ENV_SH' && cd '$ROBOTWIN_ROOT' && PYTHONUTF8=1 python '$ROOT/experiments/rt3_cases.py' bank \
      --root '$ROBOTWIN_ROOT' --source '$SRC' --tasks $TASKS --n 5 --out '$BANK'"
    finish $? ;;
  compare)
    $VLA_PY experiments/rt3_compare.py; finish $? ;;
  eval) ;;
  *) echo "режим $MODE неизвестен"; finish 2 ;;
esac

[ -f "$BANK" ] || { echo "ОТКАЗ: нет банка $BANK (rt3_run.sh bank)"; finish 3; }
N=5
LOGD="reports/rt3/eval/nexec${NEXEC}_s0_flat24"
mkdir -p "$LOGD/quarantine"
echo "=== ЭТАП ожидаемая модель (flat24 s0, n_exec $NEXEC)"
$VLA_PY experiments/rt2_eval.py expect --seed 0 --arch flat24 --log-dir "$LOGD" --n-exec "$NEXEC" \
  || { echo "ОТКАЗ: модель не совпадает с журналом $LOGD"; finish 3; }
PORT="$($VLA_PY -c 'import socket; s=socket.socket(); s.bind(("localhost",0)); print(s.getsockname()[1])')"
echo "=== ЭТАП сервер политики (порт $PORT)"
$VLA_PY experiments/rt2_eval.py serve --seed 0 --arch flat24 --device "$DEV" --n-exec "$NEXEC" \
  --port "$PORT" --log-dir "$LOGD" > "logs/rt3/server_nexec${NEXEC}.log" 2>&1 &
SPID=$!
for i in $(seq 1 240); do
  kill -0 "$SPID" 2>/dev/null || { echo "ОТКАЗ: сервер завершился"; tail -n 30 "logs/rt3/server_nexec${NEXEC}.log"; finish 3; }
  $VLA_PY -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('localhost',$PORT)))" && break
  sleep 5
done
grep -E "загружено" "logs/rt3/server_nexec${NEXEC}.log" | tail -n 1
for T in $TASKS; do
  if $VLA_PY experiments/rt2_eval.py is-done --log-dir "$LOGD" --tasks "$T" --n "$N" >/dev/null 2>&1; then
    echo "    $T: уже завершена и проверена"; continue
  fi
  if [ -d "$LOGD/$T" ]; then
    Q="$LOGD/quarantine/$(date +%Y%m%dT%H%M%S)_$T"; mv "$LOGD/$T" "$Q"; echo "    $T: неполный журнал -> $Q"
  fi
  echo "=== ЭТАП $T: $N случаев банка, $(date '+%H:%M:%S')"
  CL="logs/rt3/client_nexec${NEXEC}_${T}.log"; rc=0
  bash -c "source '$RT_ENV_SH' && cd '$ROBOTWIN_ROOT' && PYTHONUTF8=1 python '$ROOT/experiments/rt3_cases.py' eval \
    --root '$ROBOTWIN_ROOT' --bank '$BANK' --task '$T' --host localhost --port '$PORT' --frequency 15" \
    > "$CL" 2>&1 || rc=$?
  grep -E "Success rate|Final success|ОТКАЗ" "$CL" | tail -n 2
  [ $rc -eq 0 ] || { echo "ОСТАНОВ: клиент $T код $rc ($CL)"; finish 3; }
  $VLA_PY experiments/rt2_eval.py client-check --log "$CL" || { echo "ОСТАНОВ: ошибка раскатки $T"; finish 3; }
  kill -0 "$SPID" 2>/dev/null || { echo "ОСТАНОВ: сервер упал"; finish 3; }
  $VLA_PY experiments/rt2_eval.py mark-done --log-dir "$LOGD" --tasks "$T" --n "$N" --log "$CL" \
    || { echo "ОСТАНОВ: журнал $T не прошёл проверку"; finish 3; }
done
echo "=== ЭТАП сводка"
$VLA_PY experiments/rt2_eval.py summarize --log-dir "$LOGD" --n "$N" --out "reports/rt3/nexec${NEXEC}_s0_flat24.json" || finish 3
finish 0
