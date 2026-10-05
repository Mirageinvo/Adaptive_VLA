#!/usr/bin/env bash
# K-15d: проверки и обучение уточнения D0/H1 одним фоновым процессом.
#
#   check — самопроверки, интеграция на игрушечной среде (CPU), гейт нулевой
#           точки, overfit 32 строк для d0/h1p1/h1p2, smoke 100 батчей для
#           d0/h1p1/h1p2 (h1p2 — от smoke-h1p1). ~40 минут.
#   full  — полная эпоха d0, затем h1p1, затем h1p2 от лучшей точки h1p1.
#           ~10 часов. Требует, чтобы check в этом же коммите прошёл.
#   all   — check, и при полном успехе сразу full.
#   prep2 — подготовка ВТОРОЙ карты (аргумент — она, обычно cuda:0): гейт
#           K-15a и гейт K-15d на ней в ОТДЕЛЬНЫЕ файлы *_cuda0.json и
#           smoke d0 на ней. Гейт K-15a сверяет q0 побитово, так что карта
#           допускается, только если канонический q0 на ней воспроизводится.
#   fullpar — d0 на второй карте (D0_DEVICE, по умолчанию cuda:0) ПАРАЛЛЕЛЬНО
#           с h1p1 -> h1p2 на основной. Требует check на основной и prep2 на
#           второй в этом же коммите. ~6–7 часов вместо ~11–13.
#
# Гейты K-15a/K-15d основной карты (cuda:1) лежат в канонических файлах; у
# второй карты — свои файлы, канонические не трогаются.
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
D0_DEVICE="${D0_DEVICE:-cuda:0}"
case "$WHAT" in check|full|all|prep2|fullpar) ;; *)
    echo "использование: $0 check|full|all|prep2|fullpar [карта]" >&2
    exit 1;;
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
HEAD_FULL="$(git rev-parse HEAD 2>/dev/null)"
GATE_JSON="${REP}/init_identity.json"

gate_run_id() {   # [файл гейта K-15d]
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' \
        "${1:-$GATE_JSON}" 2>/dev/null
}
dev_tag() { echo "$1" | tr -d ':'; }   # cuda:0 -> cuda0
gate15a_for() { echo "reports/k15a/init_identity_$(dev_tag "$1").json"; }
gate15d_for() { echo "${REP}/init_identity_$(dev_tag "$1").json"; }
prep_marker() {   # карта
    printf 'commit=%s seed=%s device=%s gate_run_id=%s\n' \
        "$HEAD_FULL" "$SEED" "$1" "$(gate_run_id "$(gate15d_for "$1")")"
}

# Маркер check: коммит, сид, карта и запуск гейта. full по чужому маркеру
# (другой сид, другая карта, переснятый гейт) не стартует.
check_marker() {
    printf 'commit=%s seed=%s device=%s gate_run_id=%s\n' \
        "$HEAD_FULL" "$SEED" "$DEVICE" "$(gate_run_id)"
}

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
    check_marker > "${CODES}/check_passed"
    echo "  CHECK ПРОЙДЕН: $(cat "${CODES}/check_passed")"
}

do_full() {
    local want got
    want="$(check_marker)"
    got="$(cat "${CODES}/check_passed" 2>/dev/null)"
    if [ "$got" != "$want" ]; then
        echo "ОТКАЗ: check не пройден в этой обстановке" >&2
        echo "  маркер: ${got:-нет}" >&2
        echo "  нужно:  $want" >&2
        return 1
    fi
    stage "4. полная эпоха d0"
    run_one full_d0 $T --phase d0 --mode full
    local c_d0=$?
    stage "5. полная эпоха h1p1"
    run_one full_h1p1 $T --phase h1p1 --mode full
    local c_p1=$?
    local c_p2=3
    if [ "$c_p1" -ne 0 ] && [ "$c_p1" -ne 4 ]; then
        echo "  h1p1: технический отказ, h1p2 не запускается"
    else
        stage "6. полная эпоха h1p2"
        run_one full_h1p2 $T --phase h1p2 --mode full \
            --phase1 "data/k15d/h1p1_s${SEED}.pt"
        c_p2=$?
    fi
    final_code "$c_d0" "$c_p1" "$c_p2"
}

do_prep2() {   # подготовка второй карты $DEVICE
    local g15a g15d tag
    g15a="$(gate15a_for "$DEVICE")"
    g15d="$(gate15d_for "$DEVICE")"
    tag="$(dev_tag "$DEVICE")"
    stage "гейт K-15a на $DEVICE -> $g15a"
    run_one "k15a_gate_${tag}" python3 experiments/k15a_check_init_identity.py \
        --device "$DEVICE" --out "$g15a" --overwrite || return 1
    stage "гейт K-15d на $DEVICE -> $g15d"
    run_one "gate_${tag}" python3 experiments/k15d_check_init_identity.py \
        --device "$DEVICE" --seed "$SEED" --init-gate "$g15a" \
        --out "$g15d" || return 1
    stage "smoke d0 на $DEVICE"
    run_one "smoke_d0_${tag}" $T --phase d0 --mode smoke \
        --init-gate "$g15a" --k15d-gate "$g15d" \
        --out "data/k15d/d0_s${SEED}_smoke_${tag}.pt" \
        --report "${REP}/d0_s${SEED}_smoke_${tag}.json" || return 1
    prep_marker "$DEVICE" > "${CODES}/prep2_${tag}"
    echo "  КАРТА $DEVICE ГОТОВА: $(cat "${CODES}/prep2_${tag}")"
}

do_fullpar() {   # d0 на D0_DEVICE параллельно с h1p1 -> h1p2 на $DEVICE
    local want got tag g15a g15d
    want="$(check_marker)"
    got="$(cat "${CODES}/check_passed" 2>/dev/null)"
    if [ "$got" != "$want" ]; then
        echo "ОТКАЗ: check на $DEVICE не пройден в этой обстановке" >&2
        return 1
    fi
    tag="$(dev_tag "$D0_DEVICE")"
    want="$(prep_marker "$D0_DEVICE")"
    got="$(cat "${CODES}/prep2_${tag}" 2>/dev/null)"
    if [ "$got" != "$want" ]; then
        echo "ОТКАЗ: prep2 на $D0_DEVICE не пройден в этой обстановке" >&2
        echo "  маркер: ${got:-нет}" >&2
        echo "  нужно:  $want" >&2
        return 1
    fi
    g15a="$(gate15a_for "$D0_DEVICE")"
    g15d="$(gate15d_for "$D0_DEVICE")"
    stage "4. d0 на $D0_DEVICE (фоном) и h1p1 на $DEVICE"
    run_one full_d0 python3 experiments/k15d_train.py --device "$D0_DEVICE" \
        --seed "$SEED" --phase d0 --mode full \
        --init-gate "$g15a" --k15d-gate "$g15d" &
    local pid_d0=$!
    run_one full_h1p1 $T --phase h1p1 --mode full
    local c_p1=$?
    local c_p2=3
    if [ "$c_p1" -ne 0 ] && [ "$c_p1" -ne 4 ]; then
        echo "  h1p1: технический отказ, h1p2 не запускается"
    else
        stage "5. h1p2 на $DEVICE"
        run_one full_h1p2 $T --phase h1p2 --mode full \
            --phase1 "data/k15d/h1p1_s${SEED}.pt"
        c_p2=$?
    fi
    stage "6. жду d0 на $D0_DEVICE"
    wait "$pid_d0"
    local c_d0=$?
    final_code "$c_d0" "$c_p1" "$c_p2"
}

final_code() {   # c_d0 c_p1 c_p2
    echo
    echo "  коды: d0 $1, h1p1 $2, h1p2 $3 (0 — допущен к роллауту," \
         "4 — исправен, но не допущен, иное — технический отказ)"
    if [ "$1" -eq 0 ] || [ "$3" -eq 0 ]; then
        echo "  ИТОГ: есть допущенный финальный вариант"
        return 0
    elif [ "$1" -eq 4 ] && [ "$3" -eq 4 ]; then
        echo "  ИТОГ: оба финальных варианта исправны, ни один не допущен"
        return 4
    fi
    echo "  ИТОГ: технический отказ, допущенного финального варианта нет"
    return 3
}

echo "=== K-15d $WHAT: старт $(date), карта $DEVICE, сид $SEED, коммит" \
     "$HEAD_SHA, логи $LOGDIR ==="
rc=0
case "$WHAT" in
    check) do_check; rc=$? ;;
    full)  do_full; rc=$? ;;
    all)   do_check && do_full; rc=$? ;;
    prep2) do_prep2; rc=$? ;;
    fullpar) do_fullpar; rc=$? ;;
esac
echo "=== K-15d $WHAT: конец $(date), код $rc ==="
exit "$rc"
