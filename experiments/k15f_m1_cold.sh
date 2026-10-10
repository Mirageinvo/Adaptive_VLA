#!/bin/bash
# K-15f M1-cold: восстановление M1 на холодных стартах в отдельном каталоге.
#
#   cold-q0  холодные q0 блока 5 (одиночный процесс) для задач, где их ещё
#            нет в reports/k15f/diag; уже снятые (2, 8, 9) не пересчитываются;
#   census   новая перепись штатной census() по 100 эпизодам (блок 0 из M1
#            — первый в процессе; блок 5 — холодный) + отличия от исходной;
#   plan     фиксация плана восстановления (задания по НОВОЙ переписи);
#   run      сборка переиспользуемых блоков 0 и холодные вторые блоки всех
#            16 рук — КАЖДЫЙ отдельным процессом; затем официальный анализ;
#   analyze  только официальный анализ M1 по каталогу M1-cold.
#
# Код M1, k15f_measure_subspace.py и харнесс не меняются. Исходный M1 и его
# перепись не трогаются. Повтор — только при «No CUDA GPUs are available».
set -euo pipefail
MODE="${1:?нужен режим: cold-q0, census, plan, run или analyze}"
DEV="${2:-cuda:1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p logs/k15f reports/k15f/diag reports/k15f/m1cold
COLD=reports/k15f/m1cold
LOGF="logs/k15f/m1cold_${MODE}.log"
for m in k15f_m1_cold k15f_measure_subspace k15f_policy k9h_multiarm_gate; do
  python3 "experiments/${m}.py" $( [ "$m" = k15f_m1_cold ] && echo selftest \
    || echo --selftest ) >/dev/null || { echo "ОТКАЗ: самопроверка $m"; exit 1; }
done

COMMON="--ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
  --task-suite 10 --n-envs 5 --seed 101 --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 --device $DEV --save-actions"
DRVQ="--policy depthrvq --policy-ckpt data/k9d_ep3.pt \
  --q1-ckpt data/k14c/q1_main_s0.pt --expect-q1-variant main \
  --expect-q1-seed 0 --expect-q0-manifest data/k14d/q0_b8_e0.manifest.json \
  --depth-rvq-mode fast"

valid () {   # путь метка задача блок -> 0 цел
  [ -f "$1" ] && python3 experiments/k15g_local_probe.py --validate-block \
    "$1" "$2" "$3" "$4" >/dev/null 2>&1
}

run_one () {   # метка задача блок выходной-каталог [аргументы руки...]
  local L="$1" T="$2" B="$3" OUT="$4"; shift 4
  local F="$OUT/${L}_t${T}_i${B}.json" try rc TMP
  if valid "$F" "$L" "$T" "$B"; then echo "    $L t$T i$B: уже есть"; return 0; fi
  if [ -f "$F" ]; then
    local Q="$OUT/quarantine/$(date +%Y%m%dT%H%M%S)"; mkdir -p "$Q"
    mv "$F" "${F%.json}.actions.npz" "$Q/" 2>/dev/null || true
    echo "    $L t$T i$B: повреждённый результат -> $Q"
  fi
  for try in 1 2 3; do
    echo "    $L t$T i$B — попытка $try, $(date '+%H:%M:%S')"
    TMP="$(mktemp)"; rc=0
    # shellcheck disable=SC2086
    python3 experiments/k9h_multiarm_gate.py $COMMON "$@" --task-id "$T" \
      --init-starts "$B" --arm-label "$L" --out "$OUT/${L}_t${T}_i{i0}.json" \
      > "$TMP" 2>&1 || rc=$?
    cat "$TMP"
    if [ $rc -eq 0 ]; then rm -f "$TMP"; sleep 3; return 0; fi
    if grep -q "No CUDA GPUs are available" "$TMP" && [ "$try" -lt 3 ]; then
      echo "    нет GPU — пауза 120 с и повтор"; rm -f "$TMP"; sleep 120
      continue
    fi
    rm -f "$TMP"; echo "ОСТАНОВ: $L t$T i$B — код $rc"; return $rc
  done
}

exec >> "$LOGF" 2>&1
echo "=== СТАРТ $(date) === режим $MODE, карта $DEV, коммит" \
     "$(git rev-parse --short HEAD 2>/dev/null)"
CODE=0
case "$MODE" in
  cold-q0)
    for T in 0 1 2 3 4 5 6 7 8 9; do
      # shellcheck disable=SC2086
      run_one q0 "$T" 5 reports/k15f/diag $DRVQ --run-tag k15f_diag
    done ;;
  census) python3 experiments/k15f_m1_cold.py census || CODE=$? ;;
  plan)   python3 experiments/k15f_m1_cold.py plan || CODE=$? ;;
  run)
    python3 experiments/k15f_m1_cold.py assemble
    python3 experiments/k15f_m1_cold.py jobs | while read -r T L B; do
      run_one "$L" "$T" "$B" "$COLD" --policy k15f \
        --k15f-basis data/k15f/basis_s0.pt --run-tag k15f_m1cold_s101 \
        || exit $?
    done
    MODE=analyze ;&
  analyze)
    LABS="$(python3 -c "import sys; sys.path.insert(0,'experiments'); \
import k15f_policy as k; print(' '.join(k.all_labels()))")"
    ARTS=""
    for L in $LABS; do ARTS="$ARTS $(ls $COLD/${L}_t*_i*.json | tr '\n' ' ')"; done
    # shellcheck disable=SC2086
    python3 experiments/k15f_measure_subspace.py --mode m1 \
      --census reports/k15f/m1cold_census.json \
      --q0-arts $(ls $COLD/q0_t*_i*.json) --arts $ARTS \
      --basis-report reports/k15f/basis_s0.json \
      --out reports/k15f/m1cold_result.json || CODE=$? ;;
  *) echo "режим $MODE неизвестен"; exit 2 ;;
esac
echo "=== КОНЕЦ $(date), код $CODE ==="
exit $CODE
