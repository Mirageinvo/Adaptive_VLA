#!/usr/bin/env bash
# K-15b: все три замера одним фоновым процессом, с записью кодов возврата.
#
# ЗАЧЕМ СКРИПТ. `nohup ... &` возвращает код запуска фонового процесса, а не
# код Python, и различие 0/3/4 терялось бы. Здесь каждый замер пишет свой
# код в файл рядом с артефактами, а итоговый код процесса — это код M2,
# решающего замера.
#
# ЧТО ЗАПУСКАЕТСЯ
#   1. геометрия книги   — CPU, секунды, НИЧЕГО НЕ РЕШАЕТ, описание;
#   2. M2, мягкий путь   — решающий, 30-40 мин;
#   3. M1, грубый путь   — пересъёмка под текущий код, 18 мин.
#
# ПОЧЕМУ ПОСЛЕДОВАТЕЛЬНО, А НЕ НА ДВУХ КАРТАХ. Гейт K-15a снят на
# конкретной карте, и `k15_context` сверяет `device` как часть обстановки:
# на другой карте любой замер отказывает словами «гейт снят в другой
# обстановке». Переснимать гейт под вторую карту нельзя дёшево — на нём
# висит вся цепочка провенанса. Поэтому оба GPU-замера идут друг за
# другом на ОДНОЙ карте, той же, что у гейта.
#
# CUDA_VISIBLE_DEVICES не выставляется намеренно: robosuite выводит из неё
# MUJOCO_EGL_DEVICE_ID, и EGL перестаёт инициализироваться. Карта задаётся
# только --device.
#
# КОДЫ ЗАМЕРОВ: 0 порог пройден; 3 технический блокер или неполная часть;
#               4 отрицательный результат по объявленному порогу;
#               иное — падение, смотреть хвост лога.
#
# ЗАПУСКАТЬ ИЗ КОРНЯ РЕПОЗИТОРИЯ. Git здесь не вызывается намеренно.
set -uo pipefail

# ОДНА КАРТА НА ОБА ЗАМЕРА: та, на которой снят гейт K-15a.
DEVICE="${1:-cuda:1}"
STAMP="$(date +%Y%m%dT%H%M%S)"
LOGDIR="logs/k15b"
REP="reports/k15b"
mkdir -p "$LOGDIR" "$REP" data/k15b

if [ ! -f experiments/k15b_measure_soft.py ]; then
    echo "ОТКАЗ: запускать из корня репозитория" >&2
    exit 1
fi

# ВХОДНЫЕ АРТЕФАКТЫ ПРОВЕРЯЮТСЯ ДО ЗАГРУЗКИ МОДЕЛИ: иначе отказ придёт
# через две минуты после старта, уже заняв карту.
for f in data/k15b/c1_selected.pt data/k15b/rankpath_target_train.npz \
         data/k15b/q1_reader_s0.pt; do
    if [ ! -f "$f" ]; then
        echo "ОТКАЗ: нет $f" >&2
        exit 1
    fi
done

# САМОПРОВЕРКИ ПЕРЕД ВСЕМ. Они без GPU и занимают секунды, а ловят ровно
# то, что иначе выяснилось бы после загрузки модели.
echo "=== самопроверки ==="
for m in k15_context k15b_probe_and_extract k15b_build_rankpath_cache \
         k15b_train_stagewise k15b_measure_interface k15b_book_geometry \
         k15b_measure_soft; do
    if ! python3 "experiments/${m}.py" --selftest; then
        echo "ОТКАЗ: самопроверка ${m} не прошла" >&2
        exit 1
    fi
done

run_one() {
    local name="$1" exitfile="$2" log="$3"
    shift 3
    if [ -e "$exitfile" ]; then
        mv "$exitfile" "${exitfile}.${STAMP}.bak"
    fi
    "$@" >"$log" 2>&1
    local code=$?
    printf '%s\n' "$code" > "$exitfile"
    echo "[$name] код возврата $code -> $exitfile"
    return "$code"
}

echo
echo "=== 1/3 геометрия книги (CPU, ничего не решает) ==="
GEO_LOG="$LOGDIR/book_geometry_${STAMP}.log"
run_one geometry "$REP/book_geometry.exit" "$GEO_LOG" \
    python3 experiments/k15b_book_geometry.py --overwrite
GEO=$?
tail -n 30 "$GEO_LOG"

# ОКРУЖЕНИЕ ЭКСПОРТИРУЕТСЯ ОДИН РАЗ. LIBERO нужен обоим GPU-замерам,
# геометрии он был не нужен и до сюда не доходил.
export PYTHONPATH="${HOME}/LIBERO"
export MUJOCO_GL=egl

SOFT_LOG="$LOGDIR/measure_soft_${STAMP}.log"
IFACE_LOG="$LOGDIR/measure_interface_${STAMP}.log"

echo
echo "=== 2/3 M2, мягкий путь, РЕШАЮЩИЙ, на $DEVICE ==="
echo "  лог: $SOFT_LOG"
run_one soft "$REP/measure_soft.exit" "$SOFT_LOG" \
    python3 experiments/k15b_measure_soft.py \
        --device "$DEVICE" --overwrite
SOFT=$?

echo
echo "=== 3/3 M1, грубый путь, пересъёмка, на $DEVICE ==="
echo "  лог: $IFACE_LOG"
run_one iface "$REP/interface_measure.exit" "$IFACE_LOG" \
    python3 experiments/k15b_measure_interface.py \
        --device "$DEVICE" --overwrite
IFACE=$?

echo
echo "=== хвост M2 ==="
tail -n 45 "$SOFT_LOG"
echo
echo "=== хвост M1 ==="
tail -n 20 "$IFACE_LOG"

echo
echo "=== ИТОГ ==="
describe_code() {
    case "$1" in
        0) echo "порог пройден" ;;
        3) echo "технический блокер или неполная часть" ;;
        4) echo "отрицательный результат по объявленному порогу" ;;
        *) echo "ПАДЕНИЕ, смотреть хвост лога" ;;
    esac
}
if [ "$GEO" = 0 ]; then
    echo "  геометрия: 0 — описание записано, решений не принимает"
else
    echo "  геометрия: $GEO — ПАДЕНИЕ, смотреть $GEO_LOG"
fi
echo "  M2 мягкий путь: $SOFT ($(describe_code "$SOFT"))  <- решающий"
echo "  M1 грубый путь: $IFACE ($(describe_code "$IFACE"))"
echo "  сводки:"
echo "    $REP/book_geometry.json"
echo "    $REP/measure_soft.json"
echo "    $REP/interface_measure.json"

if [ "$SOFT" != 0 ] && [ "$SOFT" != 3 ] && [ "$SOFT" != 4 ]; then
    exit 1
fi
exit "$SOFT"
