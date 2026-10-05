#!/usr/bin/env bash
# K-15d: проверки и обучение уточнения D0/H1 одним фоновым процессом.
#
#   check — самопроверки, интеграция на игрушечной среде (CPU), гейт нулевой
#           точки, overfit 32 строк для d0/h1p1/h1p2, smoke 100 батчей для
#           d0/h1p1/h1p2 (h1p2 — от smoke-h1p1). ~40 минут.
#   full  — полная эпоха d0, затем h1p1, затем h1p2 от лучшей точки h1p1.
#           ~10 часов. Требует, чтобы check в этом же коммите прошёл.
#   all   — check, и при полном успехе сразу full.
#
# КАРТА ОДНА — та, на которой снят гейт K-15a (cuda:1): параллельный запуск
# на cuda:0 отказал бы на сверке гейта. Фазы идут последовательно.
# CUDA_VISIBLE_DEVICES не выставляется. Git только читается.
#
#   cd /home/malinin_aa/Adaptive_VLA && mkdir -p logs/k15d && \
#   setsid nohup bash experiments/k15d_run.sh all cuda:1 \
#     > logs/k15d/run_$(date +%Y%m%dT%H%M%S).log 2>&1 &
set -uo pipefail

WHAT="${1:-check}"
DEVICE="${2:-cuda:1}"
SEED="${SEED:-0}"
STAMP="$(date +%Y%m%dT%H%M%S)"
LOGDIR="logs/k15d/${STAMP}"
REP="reports/k15d"
CODES="${REP}/codes"
case "$WHAT" in check|full|all) ;; *)
    echo "использование: $0 check|full|all [cuda:1]" >&2; exit 1;;
esac
if [ ! -f experiments/k15d_run.sh ]; then
    echo "ОТКАЗ: запускать из корня репозитория" >&2
    exit 1
fi
mkdir -p "$LOGDIR" "$CODES" data/k15d
export PYTHONPATH="${HOME}/LIBERO"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
TRACKED="$(git ls-files reports/k15d data/k15d logs/k15d 2>/dev/null | head -3)"
if [ -n "$TRACKED" ]; then
    echo "ОТКАЗ: в git отслеживаются файлы, которые цепочка перезапишет:" >&2
    echo "$TRACKED" | sed 's/^/    /' >&2
    exit 1
fi
HEAD_SHA="$(git rev-parse --short HEAD 2>/dev/null)"

stage() { echo; echo "=== $* — $(date '+%H:%M:%S') ==="; }

run_one() {   # имя, команда... -> код; лог и файл кода по имени
    local name="$1"
    shift
    local log="${LOGDIR}/${name}.log" codefile="${CODES}/${name}.code"
    [ -e "$codefile" ] && mv "$codefile" "${codefile}.${STAMP}.bak"
    echo "  [$name] лог: $log"
    "$@" >"$log" 2>&1
    local code=$?
    printf '%s %s\n' "$code" "$HEAD_SHA" > "$codefile"
    echo "  [$name] код $code; $(grep -E '^ИТОГ|^ГЕЙТ' "$log" | tail -1)"
    return "$code"
}

T="python3 experiments/k15d_train.py --device $DEVICE --seed $SEED"

do_check() {
    stage "0. самопроверки"
    for m in k15_context k15d_depth_refine k15d_check_init_identity \
             k15d_train; do
        if ! python3 "experiments/${m}.py" --selftest >/dev/null 2>&1; then
            echo "ОТКАЗ: самопроверка ${m} не прошла" >&2
            return 1
        fi
    done
    echo "  самопроверки пройдены"
    run_one integration python3 experiments/k15d_train.py --integration \
        || return 1

    stage "1. гейт нулевой точки"
    run_one gate python3 experiments/k15d_check_init_identity.py \
        --device "$DEVICE" --seed "$SEED" || return 1

    stage "2. overfit 32 строк"
    run_one overfit_d0 $T --phase d0 --mode overfit || return 1
    run_one overfit_h1p1 $T --phase h1p1 --mode overfit || return 1

    stage "3. smoke 100 батчей"
    run_one smoke_d0 $T --phase d0 --mode smoke || return 1
    run_one smoke_h1p1 $T --phase h1p1 --mode smoke || return 1
    run_one overfit_h1p2 $T --phase h1p2 --mode overfit \
        --phase1 "data/k15d/h1p1_s${SEED}_smoke.pt" --allow-smoke-phase1 \
        || return 1
    run_one smoke_h1p2 $T --phase h1p2 --mode smoke \
        --phase1 "data/k15d/h1p1_s${SEED}_smoke.pt" --allow-smoke-phase1 \
        || return 1
    printf '%s\n' "$HEAD_SHA" > "${CODES}/check_passed"
    echo "  CHECK ПРОЙДЕН в коммите $HEAD_SHA"
}

do_full() {
    if [ "$(cat "${CODES}/check_passed" 2>/dev/null)" != "$HEAD_SHA" ]; then
        echo "ОТКАЗ: check не пройден в коммите $HEAD_SHA" >&2
        return 1
    fi
    stage "4. полная эпоха d0"
    run_one full_d0 $T --phase d0 --mode full
    local c_d0=$?
    stage "5. полная эпоха h1p1"
    run_one full_h1p1 $T --phase h1p1 --mode full
    local c_p1=$?
    if [ "$c_p1" -ne 0 ] && [ "$c_p1" -ne 4 ]; then
        echo "  h1p1: технический отказ, h1p2 не запускается"
        return 1
    fi
    stage "6. полная эпоха h1p2"
    run_one full_h1p2 $T --phase h1p2 --mode full \
        --phase1 "data/k15d/h1p1_s${SEED}.pt"
    local c_p2=$?
    echo
    echo "  коды: d0 $c_d0, h1p1 $c_p1, h1p2 $c_p2 (0 — допущен к роллауту," \
         "4 — исправен, но не допущен, 3 — технический отказ)"
}

echo "=== K-15d $WHAT: старт $(date), карта $DEVICE, сид $SEED, коммит" \
     "$HEAD_SHA, логи $LOGDIR ==="
rc=0
case "$WHAT" in
    check) do_check; rc=$? ;;
    full)  do_full; rc=$? ;;
    all)   do_check && do_full; rc=$? ;;
esac
echo "=== K-15d $WHAT: конец $(date), код $rc ==="
exit "$rc"
