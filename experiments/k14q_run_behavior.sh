#!/bin/bash
# §53: парный поведенческий прогон трёх рук на заданном банке.
#
# ТРИ РУКИ НА ОДНИХ НАЧАЛЬНЫХ СОСТОЯНИЯХ:
#   q0    точный K-14 черновик, forward_joint_fast через политику depthrvq
#         в режиме fast (тождественность с рукой fast доказана §53.3);
#   q1    q0 + q1, depthrvq medium;
#   bar   полная BAR, три уровня, как практический ориентир.
#
# RE-ATTENTION НЕ УЧАСТВУЕТ: он не прошёл правило отбора §49, и включать
# проигравшего на другой метрике нельзя (§53.1).
#
# ОДИН execution seed НА ПОВТОР, ОБЩИЙ ДЛЯ ТРЁХ РУК (§53.2). Внутри повтора
# руки обязаны идти с одним --seed, иначе различие рук смешается с различием
# реализаций. Сиды повторов заданы заранее: 101 102 103 104.
#
# --rollout-seed-mode fixed ОБЯЗАТЕЛЕН: при block сид зависит от init_start,
# и один init_state_id получил бы разные сиды в разных блоках.
#
# БЛОК 5 СОСТОЯНИЙ — ограничение памяти хоста (§53.4), не выбор.
#
# ПОРЯДОК РУК ВНУТРИ БЛОКА РАНДОМИЗИРУЕТСЯ (§53.1): порядок не должен
# коррелировать с состоянием хоста.
#
#   bash experiments/k14q_run_behavior.sh dev   cuda:1 101
#   bash experiments/k14q_run_behavior.sh final cuda:1 101
set -euo pipefail
BANK="${1:?нужен банк: dev или final}"
DEV="${2:-cuda:1}"
SEED="${3:?нужен execution seed повтора, из 101..104}"

# БАНКИ ПО §53.7: состояния 0-9 возвращены, отдельный отбор отменён, потому
# что он не запускался и на эти состояния никто не смотрел.
case "$BANK" in
  dev)   STATES="0 5 10 15 20" ;;     # 0-24,  250 кластеров
  final) STATES="25 30 35 40 45" ;;   # 25-49, 250 кластеров
  *) echo "банк $BANK неизвестен: dev или final"; exit 2 ;;
esac
case "$SEED" in 101|102|103|104) ;; *)
  echo "сид $SEED не из зарегистрированных 101..104 (§53.2)"; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

# СМОУК ПИШЕТСЯ В ДРУГОЙ КАТАЛОГ И ПОД ДРУГИМ ИМЕНЕМ ЛОГА. Урезанный набор
# не должен иметь ни одного шанса быть принятым за банк: артефакты банка
# складываются только в reports/k14q/<банк>/s<сид>, и смоук туда не попадает.
TASKS_RUN="${SMOKE_TASKS:-0 1 2 3 4 5 6 7 8 9}"
if [ -n "${SMOKE_TASKS:-}" ] || [ -n "${SMOKE_BLOCKS:-}" ]; then
  IS_SMOKE=1
  [ -n "${SMOKE_BLOCKS:-}" ] && STATES="$SMOKE_BLOCKS"
  OUTD="reports/k14q/smoke_${BANK}_s${SEED}"
  LOGF="logs/k14q_smoke_${BANK}_s${SEED}.log"
else
  IS_SMOKE=0
  OUTD="reports/k14q/$BANK/s$SEED"
  LOGF="logs/k14q_${BANK}_s${SEED}.log"
fi
mkdir -p logs "$OUTD"
exec >> "$LOGF" 2>&1

echo "=== СТАРТ $(date) === банк $BANK, карта $DEV, сид повтора $SEED"
echo "    коммит $(git rev-parse --short HEAD 2>/dev/null)"
echo "    состояния блоками: $STATES"
echo "    задачи: $TASKS_RUN"
if [ "$IS_SMOKE" = "1" ]; then
  echo "    РЕЖИМ СМОУКА: набор урезан, артефакты в $OUTD и БАНКОМ НЕ ЯВЛЯЮТСЯ"
fi

# НА final ЕДУТ ТОЛЬКО НУЖНЫЕ РУКИ, И ВЫБОР БЕРЁТСЯ ИЗ dev. Запускать все
# три и выбирать по данным final значило бы выбирать и подтверждать на одной
# выборке. Имя выбранной системы лежит в файле-разрешении.
ARMS_RUN="q0 q1 bar"
if [ "$BANK" = "final" ] && [ "$IS_SMOKE" = "0" ]; then
  AUTH=reports/k14q/final_authorized.txt
  if [ ! -f "$AUTH" ]; then
    echo "ОСТАНОВ: банк final открывается ОДИН раз и только по решению,"
    echo "  принятому на dev (§53.5). Создайте $AUTH с одной строкой:"
    echo "    q0        если на dev выбрана она"
    echo "    q0+q1     если на dev выбрано уточнение"
    exit 3
  fi
  CHOSEN="$(tr -d ' \t\r\n' < "$AUTH")"
  case "$CHOSEN" in
    q0)    ARMS_RUN="q0 bar" ;;
    q0+q1) ARMS_RUN="q0 q1 bar" ;;
    *) echo "ОСТАНОВ: в $AUTH написано «$CHOSEN», ожидалось q0 или q0+q1"
       exit 3 ;;
  esac
  echo "    выбор из dev: $CHOSEN -> руки: $ARMS_RUN"
fi

COMMON="--ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
  --task-suite 10 --n-envs 5 --seed $SEED --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 \
  --run-tag k14q_${BANK}_s${SEED} --device $DEV --save-actions"
DRVQ="--policy depthrvq --policy-ckpt data/k9d_ep3.pt \
  --q1-ckpt data/k14c/q1_main_s0.pt --expect-q1-variant main \
  --expect-q1-seed 0 --expect-q0-manifest data/k14d/q0_b8_e0.manifest.json"

run_arm () {   # $1 метка, $2 задача, $3... аргументы руки
  # ВСЕ БЛОКИ ЗАДАЧИ — ОДНИМ ПРОЦЕССОМ. Загрузка модели и создание сред
  # занимают около 4 минут против 2 минут самой раскатки; на полном банке это
  # разница между 16 и 8 часами на повтор.
  local L="$1" T="$2"; shift 2
  local rc=0 NEED=""
  for I0 in $STATES; do
    local F="$OUTD/${L}_t${T}_i${I0}.json"
    if [ -f "$F" ] && python -c "import json;json.load(open('$F'))" \
         >/dev/null 2>&1; then
      continue
    fi
    rm -f "$F"
    NEED="$NEED${NEED:+,}$I0"
  done
  if [ -z "$NEED" ]; then
    echo "    $L t$T: все блоки уже есть"
    return 0
  fi
  echo "    $L t$T: блоки $NEED"
  python experiments/k9h_multiarm_gate.py $COMMON "$@" \
    --task-id "$T" --init-starts "$NEED" \
    --arm-label "${L}" --out "$OUTD/${L}_t${T}_i{i0}.json" || rc=$?
  if [ $rc -ne 0 ]; then
    echo "ОСТАНОВ: $L, задача $T, блоки $NEED — код $rc"
    return $rc
  fi
  # ЛОГ БЕРЁТСЯ ИЗ $LOGF, А НЕ СОБИРАЕТСЯ ИЗ ИМЁН ЗАНОВО. В режиме смоука
  # имя другое, и собранный путь указывал на несуществующий файл: строка
  # прогресса выходила пустой, а grep ругался в лог.
  echo "    $L t$T готово: $(grep -h 'успех ' "$LOGF" | tail -1)"
  sleep 10
}

for T in $TASKS_RUN; do
    echo "--- задача $T, блоки $STATES, $(date), свободно $(free -g | awk 'NR==2{print $7}') ГБ"
    # ПОРЯДОК РУК ТЕПЕРЬ ПЕРЕСТАВЛЯЕТСЯ ПО ЗАДАЧЕ И СИДУ. Блоки идут внутри
    # одного процесса руки, поэтому переставлять их порядок нечем — и это
    # цена экономии восьми часов на повтор.
    ORDER=$(( (T * 7 + SEED) % 3 ))
    # ПОРЯДОК — ПЕРЕСТАНОВКА ИМЕННО ЗАПУСКАЕМЫХ РУК, а не фиксированной
    # тройки: на final их может быть две.
    SEQ=""
    NA=$(echo $ARMS_RUN | wc -w)
    for k in $(seq 0 $((NA - 1))); do
      IDX=$(( (ORDER + k) % NA + 1 ))
      SEQ="$SEQ $(echo $ARMS_RUN | cut -d' ' -f$IDX)"
    done
    SEQ="$(echo $SEQ)"
    echo "    порядок рук: $SEQ"
    for A in $SEQ; do
      case $A in
        q0)  run_arm q0  "$T" $DRVQ --depth-rvq-mode fast ;;
        q1)  run_arm q1  "$T" $DRVQ --depth-rvq-mode medium ;;
        bar) run_arm bar "$T" --policy fullbar ;;
      esac
    done
done

echo "=== раскатки закончены $(date) ==="
N=$(ls "$OUTD"/*.json 2>/dev/null | wc -l)
if [ "$IS_SMOKE" = "1" ]; then
  NT=$(echo $TASKS_RUN | wc -w); NB=$(echo $STATES | wc -w)
  echo "    артефактов $N (смоук: 3 руки x $NT задач x $NB блоков)"
  echo "=== СМОУК ЗАКОНЧЕН $(date). Это НЕ банк ==="
  exit 0
fi
NA=$(echo $ARMS_RUN | wc -w)
EXP=$(( NA * 10 * 5 ))
echo "    артефактов $N (ожидается $EXP: $NA рук x 10 задач x 5 блоков)"
if [ "$N" -ne "$EXP" ]; then
  echo "ОСТАНОВ: набор неполный, анализ не запускаю"
  exit 4
fi
echo "=== КОНЕЦ $(date). Анализ — k14q_behavior.py по ВСЕМ повторам сразу ==="
