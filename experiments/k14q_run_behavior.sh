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
OUTD="reports/k14q/$BANK/s$SEED"
mkdir -p logs "$OUTD"
exec >> "logs/k14q_${BANK}_s${SEED}.log" 2>&1

echo "=== СТАРТ $(date) === банк $BANK, карта $DEV, сид повтора $SEED"
echo "    коммит $(git rev-parse --short HEAD 2>/dev/null)"
echo "    состояния блоками: $STATES"

if [ "$BANK" = "final" ] && [ ! -f reports/k14q/final_authorized.txt ]; then
  echo "ОСТАНОВ: банк final открывается ОДИН раз и только по решению,"
  echo "  принятому на dev (§53.5). Создайте reports/k14q/final_authorized.txt"
  echo "  с именем выбранной системы, если решение принято."
  exit 3
fi

COMMON="--ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
  --task-suite 10 --n-envs 5 --seed $SEED --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 \
  --run-tag k14q_${BANK}_s${SEED} --device $DEV --save-actions"
DRVQ="--policy depthrvq --policy-ckpt data/k9d_ep3.pt \
  --q1-ckpt data/k14c/q1_main_s0.pt --expect-q1-variant main \
  --expect-q1-seed 0 --expect-q0-manifest data/k14d/q0_b8_e0.manifest.json"

run_arm () {   # $1 метка, $2 задача, $3 начало блока, $4... аргументы руки
  local L="$1" T="$2" I0="$3"; shift 3
  local OUT="$OUTD/${L}_t${T}_i${I0}.json" rc=0
  if [ -f "$OUT" ] && python -c "import json,sys;json.load(open('$OUT'))" \
       >/dev/null 2>&1; then
    echo "    уже есть $OUT"
    return 0
  fi
  rm -f "$OUT"
  python experiments/k9h_multiarm_gate.py $COMMON "$@" \
    --task-id "$T" --init-start "$I0" \
    --arm-label "${L}" --out "$OUT" || rc=$?
  if [ $rc -ne 0 ]; then
    echo "ОСТАНОВ: $L, задача $T, блок $I0 — код $rc"
    return $rc
  fi
  echo "    $L t$T i$I0: $(grep -h 'успех ' "logs/k14q_${BANK}_s${SEED}.log" | tail -1)"
  sleep 10
}

for T in 0 1 2 3 4 5 6 7 8 9; do
  for I0 in $STATES; do
    echo "--- задача $T, состояния $I0-$((I0+4)) $(date), свободно $(free -g | awk 'NR==2{print $7}') ГБ"
    # ПОРЯДОК РУК ЗАВИСИТ ОТ (задача, блок, сид) ДЕТЕРМИНИРОВАННО: он
    # перемешан, но воспроизводим, и записан в имени прогона.
    ORDER=$(( (T * 7 + I0 / 5 * 3 + SEED) % 3 ))
    case $ORDER in
      0) SEQ="q0 q1 bar" ;;
      1) SEQ="q1 bar q0" ;;
      2) SEQ="bar q0 q1" ;;
    esac
    echo "    порядок рук: $SEQ"
    for A in $SEQ; do
      case $A in
        q0)  run_arm q0  "$T" "$I0" $DRVQ --depth-rvq-mode fast ;;
        q1)  run_arm q1  "$T" "$I0" $DRVQ --depth-rvq-mode medium ;;
        bar) run_arm bar "$T" "$I0" --policy fullbar ;;
      esac
    done
  done
done

echo "=== раскатки закончены $(date) ==="
N=$(ls "$OUTD"/*.json 2>/dev/null | wc -l)
echo "    артефактов $N (ожидается 150: 3 руки x 10 задач x 5 блоков)"
if [ "$N" -ne 150 ]; then
  echo "ОСТАНОВ: набор неполный, анализ не запускаю"
  exit 4
fi
echo "=== КОНЕЦ $(date). Анализ — k14q_behavior.py по ВСЕМ повторам сразу ==="
