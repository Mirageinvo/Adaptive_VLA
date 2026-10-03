#!/usr/bin/env bash
# K-15c: кэш, головы и при успехе проверка вывода — одним фоновым процессом.
#
# ПОРЯДОК
#   0. самопроверки всех файлов K-15b и K-15c (секунды, без GPU);
#   1. smoke-кэш на первых N батчах каждой части — техническая проверка
#      хуков, форм, порядка кодов, записи и чтения; canonical=false;
#   2. тренер на smoke-кэше — проверка конвейера, а не науки;
#   3. полный канонический кэш train + val_sel (порядка двух часов);
#   4. три головы на кэше (минуты);
#   5. только если хотя бы одна голова взяла критерий — проверка вывода
#      настоящим проходом для лучшей из прошедших.
# Остановка после любого технического отказа. Научный исход шага 4 (код 4)
# — не отказ: дальше по плану одна попытка с пулингом вниманием, а её
# запускают отдельно, увидев числа.
#
# КАРТА ОДНА — та, на которой снят гейт K-15a: построитель и проверка вывода
# загружают модель и сверяют `device`. CUDA_VISIBLE_DEVICES не выставляется:
# robosuite выводит из неё MUJOCO_EGL_DEVICE_ID, и EGL ломается.
#
# Код скрипта — код тренера (0/3/4), либо 3, если проверка вывода не
# сошлась, либо 1 при падении. Каждый шаг пишет свой код в reports/k15c.
# ЗАПУСКАТЬ ИЗ КОРНЯ РЕПОЗИТОРИЯ. Git здесь не вызывается намеренно.
set -uo pipefail

DEVICE="${1:-cuda:1}"
SMOKE_BATCHES="${2:-12}"
STAMP="$(date +%Y%m%dT%H%M%S)"
LOGDIR="logs/k15c"
REP="reports/k15c"
mkdir -p "$LOGDIR" "$REP" data/k15c

if [ ! -f experiments/k15c_build_rank_cache.py ]; then
    echo "ОТКАЗ: запускать из корня репозитория" >&2
    exit 1
fi
for f in data/k15b/c1_selected.pt data/k15b/q1_reader_s0.pt \
         data/k15b/rankpath_target_train.npz reports/k15b/measure_soft.json; do
    if [ ! -f "$f" ]; then
        echo "ОТКАЗ: нет $f" >&2
        exit 1
    fi
done

export PYTHONPATH="${HOME}/LIBERO"
export MUJOCO_GL=egl

echo "=== 0. самопроверки ==="
for m in k15_context k15b_probe_and_extract k15b_build_rankpath_cache \
         k15b_train_stagewise k15b_measure_interface k15b_measure_soft \
         k15c_rank_selector k15c_build_rank_cache k15c_train_rank_selector \
         k15c_check_inference; do
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
    echo "  [$name] лог: $log"
    "$@" >"$log" 2>&1
    local code=$?
    printf '%s\n' "$code" > "$exitfile"
    echo "  [$name] код возврата $code -> $exitfile"
    return "$code"
}

echo
echo "=== 1. smoke-кэш, $SMOKE_BATCHES батчей на часть, $DEVICE ==="
L1="$LOGDIR/cache_smoke_${STAMP}.log"
run_one cache_smoke "$REP/cache_smoke.exit" "$L1" \
    python3 experiments/k15c_build_rank_cache.py --device "$DEVICE" \
        --smoke-batches "$SMOKE_BATCHES" --overwrite
C1=$?
tail -n 12 "$L1"
if [ "$C1" != 0 ]; then
    echo "ОТКАЗ: smoke-кэш не собрался, полный не запускаю" >&2
    exit 1
fi

echo
echo "=== 2. тренер на smoke-кэше (конвейер, не наука) ==="
L2="$LOGDIR/train_smoke_${STAMP}.log"
run_one train_smoke "$REP/train_smoke.exit" "$L2" \
    python3 experiments/k15c_train_rank_selector.py \
        --cache data/k15c/rank_cache_smoke --allow-smoke \
        --device "$DEVICE" --epochs 2 --overwrite
C2=$?
grep -E "проба|ВЫБРАНА|ИСХОД|ОПОРНЫЕ" "$L2" | head -n 20
# 0, 3 и 4 на нескольких десятках строк ничего не значат научно — важно
# только, что конвейер дошёл до конца. Падение (иной код) — остановка.
if [ "$C2" != 0 ] && [ "$C2" != 3 ] && [ "$C2" != 4 ]; then
    echo "ОТКАЗ: тренер упал на smoke-кэше, полный кэш не запускаю" >&2
    tail -n 30 "$L2"
    exit 1
fi

echo
echo "=== 3. полный канонический кэш train + val_sel, $DEVICE ==="
L3="$LOGDIR/cache_${STAMP}.log"
run_one cache "$REP/cache.exit" "$L3" \
    python3 experiments/k15c_build_rank_cache.py --device "$DEVICE" \
        --overwrite
C3=$?
tail -n 15 "$L3"
if [ "$C3" != 0 ]; then
    echo "ОТКАЗ: канонический кэш не опубликован" >&2
    exit 1
fi

echo
echo "=== 4. головы h18_linear, h24_linear, h24_candidate ==="
L4="$LOGDIR/train_${STAMP}.log"
run_one train "$REP/train.exit" "$L4" \
    python3 experiments/k15c_train_rank_selector.py \
        --cache data/k15c/rank_cache --device "$DEVICE" --overwrite
C4=$?
grep -vE "^      эпоха" "$L4" | tail -n 45
if [ "$C4" != 0 ] && [ "$C4" != 3 ] && [ "$C4" != 4 ]; then
    echo "ПАДЕНИЕ тренера, смотреть $L4" >&2
    exit 1
fi
if [ "$C4" != 0 ]; then
    echo
    echo "=== ИТОГ: тренер дал код $C4 — проверку вывода не запускаю ==="
    exit "$C4"
fi

echo
echo "=== 5. проверка вывода настоящим проходом для лучшей прошедшей ==="
BEST="$(python3 - <<'PY'
import json
d = json.load(open("reports/k15c/selector_s0.json", encoding="utf-8"))
ok = [(v["val"]["rms"], v["checkpoint"]) for v in d["results"].values()
      if v.get("trained") and v["verdict"]["passed"]]
print(min(ok)[1] if ok else "")
PY
)"
if [ -z "$BEST" ]; then
    echo "ОТКАЗ: код 0, но прошедшей головы в сводке нет" >&2
    exit 1
fi
echo "  голова: $BEST"
L5="$LOGDIR/inference_${STAMP}.log"
run_one inference "$REP/inference.exit" "$L5" \
    python3 experiments/k15c_check_inference.py --device "$DEVICE" \
        --selector "$BEST" --overwrite
C5=$?
tail -n 12 "$L5"

echo
echo "=== ИТОГ ==="
echo "  тренер: $C4; проверка вывода: $C5"
if [ "$C5" != 0 ]; then
    exit 3
fi
exit 0
