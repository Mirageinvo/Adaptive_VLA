#!/usr/bin/env bash
# K-12g: диагностика исходной D1 на нетронутых сюитах.
#
# ЗАЧЕМ ОТДЕЛЬНЫМ ФАЙЛОМ. Функция оболочки, определённая в интерактивной сессии,
# не видна из `nohup bash -c` — запуск падает с кодом 127, ничего не измерив.
# Раннер файлом этого класса ошибок не допускает и вдобавок пропускает уже
# готовые ячейки, так что прерванный прогон продолжается, а не начинается
# заново.
#
# Использование:
#   bash experiments/run_k12g_diag.sh cuda:0 object,goal
#   bash experiments/run_k12g_diag.sh cuda:1 spatial
set -u -o pipefail

DEV="${1:?нужно устройство, например cuda:0}"
SUITES="${2:?нужен список сюит через запятую, например object,goal}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
STARTS="${STARTS:-0 5}"           # состояния 0..9 двумя блоками по пять
NENV="${NENV:-5}"
CKPT="${CKPT:-ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO}"
HEAD="${HEAD:-data/k11d/d1_mlp_coef_0.001_wd0_s0.pt}"
# сид объявляется ЗДЕСЬ и сверяется с чекпойнтом: если HEAD
# поменяли на _s1, а D1SEED оставили нулём — прогон откажет
D1SEED="${D1SEED:-0}"
OUTDIR="${OUTDIR:-data/k12g/diag}"
LOGDIR="${LOGDIR:-logs/k12g}"
PY="${PY:-python}"

mkdir -p "$OUTDIR" "$LOGDIR"
[ -f "$HEAD" ] || { echo "нет головы D1: $HEAD"; exit 1; }

done_n=0; skip_n=0; fail_n=0
for SU in ${SUITES//,/ }; do
  LOG="$LOGDIR/diag_$SU.log"
  for T in $TASKS; do
    for S in $STARTS; do
      # ИМЯ ВКЛЮЧАЕТ СИД D1: иначе прогон для s1 пропустил бы файл,
      # созданный для s0, и вторая голова осталась бы неизмеренной
      OUT="$OUTDIR/${SU}_d1${D1SEED}_t${T}_s${S}.json"
      if [ -s "$OUT" ]; then skip_n=$((skip_n+1)); continue; fi
      echo "[$(date +%H:%M:%S)] $SU задача $T состояния $S..$((S+NENV-1))" \
        | tee -a "$LOG"
      env PYTHONPATH="$HOME/LIBERO" MUJOCO_GL=egl "$PY" \
        experiments/k12d_rollout.py \
        --stage diag --arm baseline --task-suite "$SU" --task-id "$T" \
        --init-start "$S" --n-envs "$NENV" --device "$DEV" \
        --ckpt "$CKPT" --head-ckpt "$HEAD" \
        --expect-d1-seed "$D1SEED" --out "$OUT" >> "$LOG" 2>&1
      rc=$?                      # код берём СРАЗУ: любая следующая команда
      if [ "$rc" -ne 0 ]; then   # (включая tee в конвейере) его затрёт
        fail_n=$((fail_n+1))
        echo "  ОТКАЗ rc=$rc, см. $LOG" | tee -a "$LOG"
      else
        done_n=$((done_n+1))
      fi
    done
  done
done
echo "итог: сделано $done_n, пропущено готовых $skip_n, отказов $fail_n"
echo "сводка: $PY experiments/k12f_final_gate.py --diag --cells '$OUTDIR/*.json'"
[ "$fail_n" -eq 0 ] || exit 1
