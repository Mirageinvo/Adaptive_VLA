#!/usr/bin/env bash
# K-15: полный прогон тренера с записью кода возврата.
#
# ЗАЧЕМ СКРИПТ. `nohup ... &` возвращает код запуска фонового процесса, а не
# код Python, и различие 0/3/4/5 терялось: оно оставалось только текстом в
# логе. Здесь код пишется в файл рядом с артефактами.
#
# Коды: 0 кандидат пригоден; 3 заблокированы оба уровня; 4 кандидата нет;
#       5 схема сводки (чекпойнт цел); 1 провенанс, инвариант или
#       нечисловая величина.
#
# ЗАПУСКАТЬ ИЗ КОРНЯ РЕПОЗИТОРИЯ. Git здесь не вызывается намеренно.
set -uo pipefail

DEVICE="${1:-cuda:1}"
SEED="${2:-0}"
EPOCHS="${3:-3}"

STAMP="$(date +%Y%m%dT%H%M%S)"
SUMMARY="reports/k15/depth_rvq_s${SEED}.json"
LOG="logs/k15_train_s${SEED}_${STAMP}.log"
EXITFILE="reports/k15/depth_rvq_s${SEED}.exit"

mkdir -p logs reports/k15 data/k15

if [ -e "$EXITFILE" ]; then
    mv "$EXITFILE" "${EXITFILE}.${STAMP}.bak"
fi

echo "запуск: device=$DEVICE seed=$SEED epochs=$EPOCHS"
echo "  лог:    $LOG"
echo "  сводка: $SUMMARY"
echo "  код:    $EXITFILE"

PYTHONPATH="${HOME}/LIBERO" MUJOCO_GL=egl \
python3 experiments/k15_train_depth_rvq.py \
    --epochs "$EPOCHS" --seed "$SEED" --device "$DEVICE" \
    --summary "$SUMMARY" \
    >"$LOG" 2>&1
CODE=$?

printf '%s\n' "$CODE" > "$EXITFILE"
echo "код возврата $CODE записан в $EXITFILE"
case "$CODE" in
    0) echo "  кандидат пригоден" ;;
    3) echo "  роллаут заблокирован на обоих уровнях" ;;
    4) echo "  кандидата нет: отрицательный результат" ;;
    5) echo "  схема сводки нарушена, чекпойнт цел" ;;
    *) echo "  прогон упал, смотреть хвост лога" ;;
esac
tail -n 25 "$LOG"
exit "$CODE"
