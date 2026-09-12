#!/usr/bin/env bash
# K-12h: ПРОБНАЯ раскатка политикой HiCoRA-G на LIBERO-10 (без протокола).
#
# Отвечает на вопрос, который нельзя получить из K-11i: меняет ли один
# полнобатчевый шаг градиента УСПЕХ, а не только численно выживает. Состояния
# 0..29 для финального заявления всё равно сожжены прежними прогонами, а для
# обучения годны.
#
#   bash experiments/run_k12h_trial.sh cuda:1 0.10 0              # шаг 0
#   bash experiments/run_k12h_trial.sh cuda:1 0.10 1 data/k12h/head_step0.pt
set -u -o pipefail

DEV="${1:?нужно устройство}"
SIGMA="${2:?нужна sigma, например 0.10}"
STEP="${3:-0}"
RESUME="${4:-}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
STARTS="${STARTS:-0 5 10 15 20 25}"   # состояния 0..29 блоками по пять
NENV="${NENV:-5}"
CKPT="${CKPT:-ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO}"
HEAD="${HEAD:-data/k11d/d1_mlp_coef_0.001_wd0_s0.pt}"
# сид объявляется ЗДЕСЬ и сверяется с чекпойнтом: если HEAD
# поменяли на _s1, а D1SEED оставили нулём — прогон откажет
D1SEED="${D1SEED:-0}"
OUTDIR="${OUTDIR:-data/k12h/step$STEP}"
LOG="${LOG:-logs/k12h/roll_step$STEP.log}"
PY="${PY:-python}"

mkdir -p "$OUTDIR" "$(dirname "$LOG")"
[ -f "$HEAD" ] || { echo "нет головы D1: $HEAD"; exit 1; }
if [ -n "$RESUME" ] && [ ! -f "$RESUME" ]; then
  echo "нет головы для продолжения: $RESUME"; exit 1
fi
if [ "$STEP" -ne 0 ] && [ -z "$RESUME" ]; then
  echo "шаг $STEP без --resume-head: раскатка шла бы исходной D1, а метилась"
  echo "номером шага, которого не было"; exit 1
fi

done_n=0; skip_n=0; fail_n=0
for T in $TASKS; do
  for S in $STARTS; do
    OUT="$OUTDIR/roll_t${T}_s${S}.pt"
    if [ -s "$OUT" ]; then skip_n=$((skip_n+1)); continue; fi
    echo "[$(date +%H:%M:%S)] задача $T состояния $S..$((S+NENV-1))" | tee -a "$LOG"
    set -- --stage diag --arm policy --sigma "$SIGMA" --task-suite 10 \
      --task-id "$T" --init-start "$S" --n-envs "$NENV" --device "$DEV" \
      --rl-seed 0 --step-index "$STEP" --ckpt "$CKPT" --head-ckpt "$HEAD" \
      --expect-d1-seed "$D1SEED" --out "$OUT"
    [ -n "$RESUME" ] && set -- "$@" --resume-head "$RESUME"
    env PYTHONPATH="$HOME/LIBERO" MUJOCO_GL=egl "$PY" \
      experiments/k12d_rollout.py "$@" >> "$LOG" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      fail_n=$((fail_n+1)); echo "  ОТКАЗ rc=$rc, см. $LOG" | tee -a "$LOG"
    else
      done_n=$((done_n+1))
    fi
  done
done
echo "итог: сделано $done_n, пропущено готовых $skip_n, отказов $fail_n"
echo "успех по ячейкам:"; grep -h "успех" "$LOG" | tail -20
if [ "$STEP" -eq 0 ]; then
  echo
  echo "шаг градиента:"
  echo "  $PY experiments/k12e_pg_step.py --stage diag \\"
  echo "    --rollouts \"\$(ls $OUTDIR/roll_*.pt | tr '\\n' ',')\" \\"
  echo "    --replica diag_d10_rl0 --head-ckpt $HEAD --step-index 0 \\"
  echo "    --cb0 data/k12d/cb0.pt --lr 3e-6 --device $DEV \\"
  echo "    --out-head data/k12h/head_step0.pt --out data/k12h/step0.json"
fi
[ "$fail_n" -eq 0 ] || exit 1
