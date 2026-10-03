#!/usr/bin/env bash
# K-15c: предполётная проверка ВСЕХ типов запусков ночной цепочки, ~25 мин.
#
# Каждый шаг — настоящий запуск на настоящей модели и данных, но на
# smoke-масштабе. Проверяется МЕХАНИКА, а не наука:
#
#   0. самопроверки всех файлов и синтаксис раннеров;
#   1. построитель кэша, smoke (12 батчей на часть, равномерно);
#   2. тренер голов на smoke-кэше — нужна хотя бы одна обученная h24-голова;
#   3. проверка вывода настоящим проходом (--allow-smoke);
#   4. роллаут обеих рук на задаче 8, две среды: q0 и k15c
#      (--k15c-preflight) — первый живой запуск руки k15c в харнессе. Рука
#      получает smoke-отчёт шага 3 и проверяет его ТЕМИ ЖЕ условиями, что
#      ночью настоящий: та же голова, тот же модуль голов, тот же кэш,
#      проверка пройдена;
#   5. парный анализ этих двух артефактов (--allow-preflight).
#
# Всё пишется в отдельные места (rank_cache_smoke, *_smoke, preflight/), и
# ни один артефакт отсюда не принимается настоящими шагами: smoke-голову
# отвергает рука роллаута, smoke-отчёт — тоже, preflight-артефакты —
# анализ.
#
# МОЖНО ЗАПУСКАТЬ, ПОКА ИДЁТ КАНОНИЧЕСКИЙ ПОСТРОИТЕЛЬ: модель здесь
# поднимается на той же карте (гейт K-15a), памяти V100 на два процесса
# хватает; тренер голов модели не грузит и идёт на второй карте.
#
#   bash experiments/k15c_preflight.sh cuda:1 cuda:0
set -uo pipefail
DEV_MODEL="${1:-cuda:1}"
DEV_TRAIN="${2:-cuda:0}"
STAMP="$(date +%Y%m%dT%H%M%S)"
LOGD="logs/k15c/preflight_${STAMP}"
OUTD="reports/k15c/rollout/preflight/${STAMP}"
mkdir -p "$LOGD" "$OUTD"

if [ ! -f experiments/k15c_preflight.sh ]; then
    echo "ОТКАЗ: запускать из корня репозитория"
    exit 1
fi
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

RESULTS=()
FAILED=0
step() {   # имя, допустимые коды через |, лог, команда...
    local name="$1" ok="$2" log="$3"
    shift 3
    echo
    echo "=== $name — $(date '+%H:%M:%S'), лог $log"
    "$@" >"$log" 2>&1
    local code=$?
    if [[ "|$ok|" == *"|$code|"* ]]; then
        RESULTS+=("ok    $name (код $code)")
        echo "  ok (код $code)"
        return 0
    fi
    RESULTS+=("ОТКАЗ $name (код $code, лог $log)")
    FAILED=1
    echo "  ОТКАЗ (код $code), хвост лога:"
    tail -n 15 "$log" | sed 's/^/    /'
    return 1
}
finish() {
    echo
    echo "=== ИТОГ ПРЕДПОЛЁТНОЙ ПРОВЕРКИ $(date '+%H:%M:%S') ==="
    printf '  %s\n' "${RESULTS[@]}"
    if [ "$FAILED" = 0 ]; then
        echo "  ВСЁ ПРОЙДЕНО: ночную цепочку можно запускать"
    else
        echo "  ЕСТЬ ОТКАЗЫ: ночную цепочку НЕ запускать"
    fi
    exit "$FAILED"
}

selftests() {
    local m
    for m in k15_context k15b_probe_and_extract k15b_build_rankpath_cache \
             k15b_train_stagewise k15b_measure_interface k15b_measure_soft \
             k15c_rank_selector k15c_build_rank_cache \
             k15c_train_rank_selector k15c_check_inference k15c_policy \
             k15c_behavior k15c_cache_diagnostics k9h_multiarm_gate; do
        if ! python3 "experiments/${m}.py" --selftest; then
            echo "сломан $m"
            return 1
        fi
    done
    for s in k15c_overnight.sh k15c_rollout.sh k15c_run.sh; do
        bash -n "experiments/$s" || { echo "синтаксис $s"; return 1; }
    done
}
step "0. самопроверки и синтаксис" "0" "$LOGD/0_selftests.log" selftests \
    || finish

step "1. построитель кэша, smoke" "0" "$LOGD/1_cache_smoke.log" \
    python3 experiments/k15c_build_rank_cache.py --device "$DEV_MODEL" \
        --smoke-batches 12 --overwrite || finish

step "2. тренер голов на smoke-кэше" "0|4|6" "$LOGD/2_train_smoke.log" \
    python3 experiments/k15c_train_rank_selector.py \
        --cache data/k15c/rank_cache_smoke --allow-smoke \
        --device "$DEV_TRAIN" --epochs 3 --overwrite || finish
grep -hE "проба обучаемости|ВЫБРАНА" "$LOGD/2_train_smoke.log" | sed 's/^/  /'
HEAD="$(python3 - <<'PY'
import json
d = json.load(open("reports/k15c/selector_smoke_s0.json", encoding="utf-8"))
ok = [(v["val"]["rms"], v["checkpoint"]) for k, v in d["results"].items()
      if k.startswith("h24_") and v.get("trained") and not v.get("technical")]
print(min(ok)[1] if ok else "")
PY
)"
if [ -z "$HEAD" ]; then
    RESULTS+=("ОТКАЗ нет обученной h24-головы на smoke-кэше")
    FAILED=1
    finish
fi
echo "  голова для дальнейших шагов: $HEAD"

REPORT="$OUTD/inference_smoke.json"
step "3. проверка вывода, smoke" "0" "$LOGD/3_inference_smoke.log" \
    python3 experiments/k15c_check_inference.py --device "$DEV_MODEL" \
        --rank-cache data/k15c/rank_cache_smoke --selector "$HEAD" \
        --selector-summary reports/k15c/selector_smoke_s0.json \
        --summary "$REPORT" --allow-smoke --overwrite || finish
grep -hE "настоящий проход|выбор расходится|ИСХОД" \
    "$LOGD/3_inference_smoke.log" | sed 's/^/  /'

COMMON="--ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
  --task-suite 10 --n-envs 2 --seed 101 --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 --task-id 8 \
  --init-starts 0 --run-tag k15c_preflight --device $DEV_MODEL \
  --save-actions"
step "4a. роллаут руки q0" "0" "$LOGD/4a_rollout_q0.log" \
    python3 experiments/k9h_multiarm_gate.py $COMMON \
        --policy depthrvq --policy-ckpt data/k9d_ep3.pt \
        --q1-ckpt data/k14c/q1_main_s0.pt --expect-q1-variant main \
        --expect-q1-seed 0 \
        --expect-q0-manifest data/k14d/q0_b8_e0.manifest.json \
        --depth-rvq-mode fast --arm-label q0 \
        --out "$OUTD/q0_t8_i{i0}.json" || finish
step "4b. роллаут руки k15c" "0" "$LOGD/4b_rollout_k15c.log" \
    python3 experiments/k9h_multiarm_gate.py $COMMON \
        --policy k15c --selector "$HEAD" --inference-report "$REPORT" \
        --rank-cache data/k15c/rank_cache_smoke --k15c-preflight \
        --arm-label k15c --out "$OUTD/k15c_t8_i{i0}.json" || finish
grep -hE "проверка k15c|тождество сборки|успех " \
    "$LOGD/4b_rollout_k15c.log" | sed 's/^/  /'

step "5. парный анализ" "0|4" "$LOGD/5_behavior.log" \
    python3 experiments/k15c_behavior.py --mode safety --allow-partial \
        --allow-preflight --arts "$OUTD"/q0_t8_i0.json \
        "$OUTD"/k15c_t8_i0.json --out "$OUTD/behavior.json" --overwrite \
    || finish
grep -hE "успех:|выбранные ранги|действия|ИСХОД" "$LOGD/5_behavior.log" \
    | sed 's/^/  /'
finish
