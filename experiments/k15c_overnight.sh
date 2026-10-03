#!/usr/bin/env bash
# K-15c: вся ночная цепочка одним фоновым процессом, по дереву решений плана.
#
#   0. самопроверки всех файлов K-15b/K-15c и харнесса (секунды);
#   1. кэш: если канонический кэш уже есть и цел — пропуск; иначе smoke-кэш,
#      тренер на нём (только конвейер) и полный кэш (~2 ч);
#   2. головы h18_linear, h24_linear, h24_candidate (минуты);
#   3. только при коде 4 — следующая дешёвая голова h24_positional;
#   4. при коде 0 или 6 — проверка вывода лучшей прошедшей h24-головы;
#   5. smoke роллаута: задача 8, один блок, обе руки — первый живой
#      запуск руки k15c;
#   6. safety-роллаут: задачи 8-9, 50 кластеров (~1 ч);
#   7. ТОЛЬКО при основном критерии (код 0) — парный dev, 250 кластеров
#      (~4-5 ч). При разведочном (код 6) план разрешает лишь короткий
#      роллаут, и цепочка останавливается после safety.
#
# ОСТАНОВКИ — там, где план требует решения человека: код 4 после обеих
# очередей голов (дальше LoRA — отдельное решение), любой технический отказ,
# провал проверки вывода, провал safety.
#
# КАРТА ОДНА — та, на которой снят гейт K-15a. CUDA_VISIBLE_DEVICES не
# выставляется: robosuite выводит из неё MUJOCO_EGL_DEVICE_ID. Git не
# вызывается, кроме чтения хеша коммита для лога.
#
# ПОВТОРНЫЙ ЗАПУСК БЕЗОПАСЕН: готовый канонический кэш не пересобирается,
# готовые блоки роллаутов пропускаются.
#
#   cd /home/malinin_aa/Adaptive_VLA && mkdir -p logs/k15c && \
#   setsid nohup bash experiments/k15c_overnight.sh cuda:1 \
#     > logs/k15c/overnight_$(date +%Y%m%dT%H%M%S).log 2>&1 &
set -uo pipefail

DEVICE="${1:-cuda:1}"
STAMP="$(date +%Y%m%dT%H%M%S)"
LOGDIR="logs/k15c"
REP="reports/k15c"
mkdir -p "$LOGDIR" "$REP" data/k15c

if [ ! -f experiments/k15c_overnight.sh ]; then
    echo "ОТКАЗ: запускать из корня репозитория" >&2
    exit 1
fi
for f in data/k15b/c1_selected.pt data/k15b/q1_reader_s0.pt \
         data/k15b/rankpath_target_train.npz reports/k15b/measure_soft.json; do
    [ -f "$f" ] || { echo "ОТКАЗ: нет $f" >&2; exit 1; }
done
export PYTHONPATH="${HOME}/LIBERO"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
# ФАЙЛЫ, КОТОРЫЕ ЦЕПОЧКА ПЕРЕЗАПИСЫВАЕТ, НЕ ДОЛЖНЫ ОТСЛЕЖИВАТЬСЯ GIT.
# Проверка чистоты кода пропускает только НОВЫЕ файлы в reports/, data/,
# logs/; изменённый или переименованный в .bak отслеживаемый отчёт она
# считает грязным кодом, и следующий этап с моделью отказал бы посреди
# ночи. Git здесь только читается.
TRACKED="$(git ls-files reports/k15c data/k15c logs/k15c 2>/dev/null | head -3)"
if [ -n "$TRACKED" ]; then
    echo "ОТКАЗ: в git отслеживаются файлы, которые цепочка перезапишет:" >&2
    echo "$TRACKED" | sed 's/^/    /' >&2
    echo "  уберите их из индекса (git rm --cached) и закоммитьте" >&2
    exit 1
fi

stage() { echo; echo "=== $* — $(date '+%H:%M:%S') ==="; }

run_one() {   # имя, файл кода, лог, команда...
    local name="$1" exitfile="$2" log="$3"
    shift 3
    [ -e "$exitfile" ] && mv "$exitfile" "${exitfile}.${STAMP}.bak"
    echo "  [$name] лог: $log"
    "$@" >"$log" 2>&1
    local code=$?
    printf '%s\n' "$code" > "$exitfile"
    echo "  [$name] код $code"
    return "$code"
}

best_head() {   # сводка, ключ (primary_passed|pilot_passed) -> путь чекпойнта
    python3 - "$1" "$2" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
ok = [(v["val"]["rms"], v["checkpoint"]) for k, v in d["results"].items()
      if k.startswith("h24_") and v.get("trained")
      and not v.get("technical") and v["verdict"].get(sys.argv[2])]
print(min(ok)[1] if ok else "")
PY
}

echo "=== K-15c, ночная цепочка: старт $(date), карта $DEVICE, коммит" \
     "$(git rev-parse --short HEAD 2>/dev/null) ==="

stage "0. самопроверки"
for m in k15_context k15b_probe_and_extract k15b_build_rankpath_cache \
         k15b_train_stagewise k15b_measure_interface k15b_measure_soft \
         k15c_rank_selector k15c_build_rank_cache k15c_train_rank_selector \
         k15c_check_inference k15c_policy k15c_behavior k9h_multiarm_gate; do
    if ! python3 "experiments/${m}.py" --selftest >/dev/null 2>&1; then
        echo "ОТКАЗ: самопроверка ${m} не прошла" >&2
        exit 1
    fi
done
echo "  все самопроверки пройдены"

stage "1. кэш"
# ЕСЛИ ПОСТРОИТЕЛЬ УЖЕ ИДЁТ (например, оставлен от прежнего раннера), его
# дожидаются, а не запускают второй рядом: два построителя писали бы в один
# канонический каталог. Шаблон ПРИВЯЗАН К НАЧАЛУ командной строки, то есть к
# самому запуску python: просто подстрока ловила бы и чужие процессы, где
# имя файла лишь упомянуто (grep, редактор, оболочка), и цепочка ждала бы
# до утра.
BUILDER_RE='^python[0-9.]* +([^ ]*/)?k15c_build_rank_cache\.py'
if pgrep -f "$BUILDER_RE" >/dev/null; then
    echo "  построитель кэша уже работает — жду его окончания:"
    pgrep -af "$BUILDER_RE" | sed 's/^/    /'
    while pgrep -f "$BUILDER_RE" >/dev/null; do
        sleep "${WAIT_POLL:-120}"
        echo "    $(date '+%H:%M') ещё работает"
    done
    echo "  построитель завершился"
fi
if python3 - <<'PY'
import sys
sys.path.insert(0, "experiments")
import k15c_build_rank_cache as cb
man, probs = cb.validate_cache("data/k15c/rank_cache", verify_sha=False)
sys.exit(0 if (man is not None and not probs) else 1)
PY
then
    echo "  канонический кэш уже есть и цел — не пересобираю"
else
    run_one cache_smoke "$REP/cache_smoke.exit" \
        "$LOGDIR/cache_smoke_${STAMP}.log" \
        python3 experiments/k15c_build_rank_cache.py --device "$DEVICE" \
            --smoke-batches 12 --overwrite \
        || { echo "ОТКАЗ: smoke-кэш" >&2; exit 1; }
    run_one train_smoke "$REP/train_smoke.exit" \
        "$LOGDIR/train_smoke_${STAMP}.log" \
        python3 experiments/k15c_train_rank_selector.py \
            --cache data/k15c/rank_cache_smoke --allow-smoke \
            --device "$DEVICE" --epochs 2 --overwrite
    C=$?
    grep -hE "проба обучаемости" "$LOGDIR/train_smoke_${STAMP}.log" | head -4
    # 0/4/6 на сотне строк ничего не значат научно. Но 3 — технический
    # отказ ВСЕХ h24-голов, то есть конвейер не работает, и строить полный
    # кэш под него незачем.
    case "$C" in 0|4|6) ;; *)
        echo "ОТКАЗ: тренер на smoke-кэше дал код $C" >&2; exit 1 ;; esac
    run_one cache "$REP/cache.exit" "$LOGDIR/cache_${STAMP}.log" \
        python3 experiments/k15c_build_rank_cache.py --device "$DEVICE" \
            --overwrite \
        || { echo "ОТКАЗ: канонический кэш не опубликован" >&2; exit 1; }
    grep -hE "агрегаты|воспроизвёл M2|опубликовано" \
        "$LOGDIR/cache_${STAMP}.log" | tail -3
fi

stage "2. головы h18_linear, h24_linear, h24_candidate"
SUMMARY="$REP/selector_s0.json"
run_one train "$REP/train.exit" "$LOGDIR/train_${STAMP}.log" \
    python3 experiments/k15c_train_rank_selector.py \
        --cache data/k15c/rank_cache --device "$DEVICE" --overwrite
CT=$?
grep -hE "проба обучаемости|ВЫБРАНА|основной|ГЛУБИНА|ИСХОД" \
    "$LOGDIR/train_${STAMP}.log"
case "$CT" in 0|3|4|6) ;; *) echo "ОТКАЗ: тренер упал" >&2; exit 1 ;; esac

if [ "$CT" = 4 ]; then
    stage "3. ни одна голова не взяла порогов — h24_positional"
    SUMMARY="$REP/selector_positional_s0.json"
    run_one train_positional "$REP/train_positional.exit" \
        "$LOGDIR/train_positional_${STAMP}.log" \
        python3 experiments/k15c_train_rank_selector.py \
            --cache data/k15c/rank_cache --heads h24_positional \
            --device "$DEVICE" --summary "$SUMMARY" --overwrite
    CT=$?
    grep -hE "проба обучаемости|ВЫБРАНА|основной|ИСХОД" \
        "$LOGDIR/train_positional_${STAMP}.log"
    case "$CT" in 0|3|4|6) ;; *) echo "ОТКАЗ: тренер упал" >&2; exit 1 ;; esac
fi
if [ "$CT" != 0 ] && [ "$CT" != 6 ]; then
    echo
    echo "=== ИТОГ $(date): головы дали код $CT — роллаутов нет. Код 4:" \
         "следующий шаг по плану — LoRA слоёв 19-24, это отдельное решение;" \
         "код 3: технический отказ, выводов не делать ==="
    exit "$CT"
fi

stage "4. проверка вывода настоящим проходом"
KEY=$([ "$CT" = 0 ] && echo primary_passed || echo pilot_passed)
BEST="$(best_head "$SUMMARY" "$KEY")"
if [ -z "$BEST" ]; then
    echo "ОТКАЗ: код $CT, но h24-головы с $KEY нет" >&2
    exit 1
fi
HEAD_NAME="$(basename "$BEST" | sed 's/_s0\.pt$//')"
REPORT="$REP/inference_${HEAD_NAME}.json"
echo "  голова $BEST ($KEY)"
run_one inference "$REP/inference.exit" "$LOGDIR/inference_${STAMP}.log" \
    python3 experiments/k15c_check_inference.py --device "$DEVICE" \
        --selector "$BEST" --selector-summary "$SUMMARY" \
        --summary "$REPORT" --overwrite
CI=$?
tail -n 8 "$LOGDIR/inference_${STAMP}.log"
[ "$CI" = 0 ] || { echo "=== ИТОГ: проверка вывода не сошлась (код $CI)" \
                        "— роллаутов нет ==="; exit 3; }

stage "5. smoke роллаута: задача 8, блок 0, обе руки"
SMOKE_TASKS=8 SMOKE_BLOCKS=0 bash experiments/k15c_rollout.sh safety \
    "$DEVICE" "$BEST" "$REPORT"
CS=$?
tail -n 15 "$LOGDIR/rollout_smoke_safety_s101.log"
if [ "$CS" != 0 ]; then
    echo "=== ИТОГ: smoke роллаута упал (код $CS) ==="
    exit 1
fi

stage "6. safety-роллаут: задачи 8-9, 50 кластеров"
bash experiments/k15c_rollout.sh safety "$DEVICE" "$BEST" "$REPORT"
CSF=$?
grep -hE "успех:|rescue|ИСХОД" "$LOGDIR/rollout_safety_s101.log" | tail -4
if [ "$CSF" != 0 ]; then
    echo "=== ИТОГ: safety не пройден (код $CSF) — dev не запускаю ==="
    exit "$CSF"
fi
if [ "$CT" != 0 ]; then
    echo "=== ИТОГ $(date): голова взяла только РАЗВЕДОЧНЫЙ порог — по плану" \
         "лишь короткий роллаут, он сделан; dev не запускаю ==="
    exit 6
fi

stage "7. парный dev: 250 кластеров"
bash experiments/k15c_rollout.sh dev "$DEVICE" "$BEST" "$REPORT"
CD=$?
grep -hE "успех:|rescue|по задачам|выбранные|ИСХОД" \
    "$LOGDIR/rollout_dev_s101.log" | tail -6
echo
echo "=== ИТОГ $(date): головы $CT, вывод $CI, safety $CSF, dev $CD ==="
exit "$CD"
