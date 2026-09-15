#!/usr/bin/env bash
# K-13c: детерминированное сравнение четырёх политик.
#
# fast12 и coarse24 не зависят от головы D1 и считаются ОДИН раз на обе.
# hicora_d1_det переиспользуется из K-12j (там рука называлась `baseline` при
# sigma=0 — та же политика при тех же условиях), поэтому здесь не считается.
#
#   bash experiments/run_k13c.sh cuda:0 "fast12 coarse24"
#   bash experiments/run_k13c.sh cuda:1 "t_s0 t_s1"
set -u -o pipefail

DEV="${1:?нужно устройство}"
WHAT="${2:?что считать: fast12 coarse24 t_s0 t_s1}"
TASKS="${TASKS:-3 6 8}"
STARTS="${STARTS:-30 35 40}"
NENV="${NENV:-5}"
CKPT="${CKPT:-ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO}"
OUTDIR="${OUTDIR:-data/k13c/cells}"
LOGDIR="${LOGDIR:-logs/k13c}"
RETRIES="${RETRIES:-3}"
RETRY_WAIT="${RETRY_WAIT:-300}"
PY="${PY:-python}"
ENVP=(env PYTHONPATH="$HOME/LIBERO" MUJOCO_GL=egl)

mkdir -p "$OUTDIR" "$LOGDIR"
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

done_n=0; skip_n=0
for W in $WHAT; do
  case "$W" in
    fast12)   ARM=fast12;          HEAD=""; EXTRA=() ;;
    coarse24) ARM=coarse24;        HEAD=""; EXTRA=() ;;
    t_s0)     ARM=hicora_t_d1_det; HEAD=s0
              EXTRA=(--hicora-t-ckpt data/k13b_hicora_t_s0.pt) ;;
    t_s1)     ARM=hicora_t_d1_det; HEAD=s1
              EXTRA=(--hicora-t-ckpt data/k13b_hicora_t_s1.pt) ;;
    # D1 ПЕРЕСЧИТЫВАЕТСЯ, А НЕ ПЕРЕИСПОЛЬЗУЕТСЯ. Ячейки K-12j сняты в том же
    # режиме точности, но поля precision_mode не несут — оно появилось позже.
    # Подставить его «по коду» значило бы заявить, а не измерить; пересчёт
    # стоит получаса и снимает допущение целиком.
    d1_s0)    ARM=hicora_d1_det;   HEAD=s0
              EXTRA=(--hicora-ckpt data/k11d/d1_mlp_coef_0.001_wd0_s0.pt) ;;
    d1_s1)    ARM=hicora_d1_det;   HEAD=s1
              EXTRA=(--hicora-ckpt data/k11d/d1_mlp_coef_0.001_wd0_s1.pt) ;;
    *) echo "неизвестно: $W"; exit 1 ;;
  esac
  LOG="$LOGDIR/$W.log"
  for T in $TASKS; do
    for S in $STARTS; do
      OUT="$OUTDIR/${W}_t${T}_s${S}.json"
      # ПРОПУСК ТОЛЬКО ПОСЛЕ СВЕРКИ СОСТАВА. «Файл непустой» пропустило бы
      # ячейку от другой задачи, другого предела шагов или, что случилось
      # сейчас, от другого режима точности исполнения.
      HS=""
      case "$W" in
        t_s0)  CK=data/k13b_hicora_t_s0.pt ;;
        t_s1)  CK=data/k13b_hicora_t_s1.pt ;;
        d1_s0) CK=data/k11d/d1_mlp_coef_0.001_wd0_s0.pt ;;
        d1_s1) CK=data/k11d/d1_mlp_coef_0.001_wd0_s1.pt ;;
        *)     CK="" ;;
      esac
      case "$W" in
        fast12|coarse24) HS="None" ;;
        *) HS=$($PY -c "import hashlib;print(hashlib.sha1(open('$CK','rb').read()).hexdigest()[:12])") ;;
      esac
      $PY experiments/k12j_cell_ok.py "$OUT" arm="$ARM" \
        head="${HEAD:-None}" task_id="$T" init_start="$S" n_envs="$NENV" \
        max_steps=600 head_sha1="$HS" \
        precision_mode=trunk_autocast_head_fp32 >/dev/null 2>>"$LOG"
      case $? in
        0) skip_n=$((skip_n+1)); continue ;;
        2) say "ЯЧЕЙКА $OUT ОТ ДРУГОЙ КОНФИГУРАЦИИ — остановка"; exit 2 ;;
      esac
      say "$W: задача $T, состояния $S..$((S+NENV-1))"
      A=(--arm "$ARM" --ckpt "$CKPT" --task-id "$T" --init-start "$S"
         --n-envs "$NENV" --device "$DEV" --out "$OUT" "${EXTRA[@]}")
      [ -n "$HEAD" ] && A+=(--head "$HEAD")
      try=0
      while : ; do
        "${ENVP[@]}" "$PY" experiments/k13c_cell.py "${A[@]}" >> "$LOG" 2>&1
        rc=$?
        [ "$rc" -eq 0 ] && { done_n=$((done_n+1)); break; }
        # повтор ТОЛЬКО при нехватке памяти: чужой процесс на общей машине
        if { tail -40 "$LOG" | grep -q "OutOfMemoryError" || \
             [ "$rc" -eq 137 ]; } && [ "$try" -lt "$RETRIES" ]; then
          try=$((try+1))
          say "нехватка памяти ($W t$T s$S), попытка $try через $RETRY_WAIT с"
          sleep "$RETRY_WAIT"
          continue
        fi
        say "ОТКАЗ $W t$T s$S rc=$rc — см. $LOG"
        exit 1
      done
    done
  done
done
echo "итог: посчитано $done_n, пропущено готовых $skip_n"
