#!/usr/bin/env bash
# K-12i: RL-smoke. Лестница 0 -> 1 -> 2 -> 4 принятых шагов, оценка на
# ОТЛОЖЕННЫХ состояниях, три руки и общий поток шума.
#
# ЧТО ОН ДОКАЗЫВАЕТ (и чего не доказывает). Главное сравнение — g_rl минус g0
# на состояниях, которых не было в градиенте. Сравнение с детерминированной D1
# вторично: выигрыш над ней может объясняться одним шумом, а не обучением.
# Это не доказательство превосходства метода, а проверка, что success-сигнал
# вообще проходит через голову и улучшает НОВОЕ поведение.
#
# ПОЧЕМУ ШУМ ОДИН И ТОТ ЖЕ. У g0 и g_rl режим eps eval с одним --eval-eps-seed:
# иначе вместе с весами менялись бы и случайные числа, и разницу нельзя было бы
# отнести к обучению.
#
#   bash experiments/run_k12i_smoke.sh cuda:0 s0 "0 2 6"
set -u -o pipefail

DEV="${1:?нужно устройство}"
HEADTAG="${2:?нужна голова: s0 или s1}"
TASKS="${3:?нужны задачи в кавычках, например \"0 2 6\"}"
SIGMA="${SIGMA:-0.10}"
# СТАРТ СВЕРХУ, А НЕ СНИЗУ. Дробление шага задумано как поиск наибольшего
# допустимого шага: при lr 3e-6 область доверия расходуется на 0.1%, сдвиг mu
# оказывается на четыре порядка ниже шума sigma, и лестница измеряла бы не
# «работает ли RL», а «различимо ли изменение, которого нет». K-11i это не
# нарушает: там lr 3e-6 был безопасен БЕЗ проверки приемлемости, а здесь она
# считается прямо на буфере.
LR="${LR:-1e-2}"
HALVINGS="${HALVINGS:-12}"
TRAIN_STARTS="${TRAIN_STARTS:-0 5 10 15 20 25}"   # состояния 0..29
EVAL_STARTS="${EVAL_STARTS:-30 35 40}"            # состояния 30..44, отложенные
LADDER="${LADDER:-1 2 4}"                         # после каких шагов оценивать
NENV="${NENV:-5}"
EVAL_SEED="${EVAL_SEED:-777}"
CKPT="${CKPT:-ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO}"
case "$HEADTAG" in
  s0) HEAD="${HEAD:-data/k11d/d1_mlp_coef_0.001_wd0_s0.pt}"; D1SEED=0 ;;
  s1) HEAD="${HEAD:-data/k11d/d1_mlp_coef_0.001_wd0_s1.pt}"; D1SEED=1 ;;
  *) echo "голова должна быть s0 или s1"; exit 1 ;;
esac
ROOT="${ROOT:-data/k12i/$HEADTAG}"
LOG="${LOG:-logs/k12i/$HEADTAG.log}"
PY="${PY:-python}"
ENVP=(env PYTHONPATH="$HOME/LIBERO" MUJOCO_GL=egl)

mkdir -p "$ROOT" "$(dirname "$LOG")"
[ -f "$HEAD" ] || { echo "нет головы D1: $HEAD"; exit 1; }
[ -s data/k12d/cb0.pt ] || { echo "нет data/k12d/cb0.pt: создайте её один раз"
                             echo "  --cb0-only"; exit 1; }

say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# --- раскатка: $1 рука, $2 каталог, $3 starts, $4 resume (или пусто), $5 шаг
roll () {
  local arm="$1" dir="$2" starts="$3" resume="$4" step="$5" rc
  mkdir -p "$dir"
  for T in $TASKS; do
    for S in $starts; do
      # расширение определяется рукой, а не цепочкой && ||: обучающая рука
      # пишет буфер (.pt), оценочные — только исходы (.json)
      local out
      case "$arm" in
        policy) out="$dir/t${T}_s${S}.pt" ;;
        *)      out="$dir/t${T}_s${S}.json" ;;
      esac
      if [ -s "$out" ]; then continue; fi
      local a=(--stage diag --arm "$arm" --task-suite 10 --task-id "$T"
               --init-start "$S" --n-envs "$NENV" --device "$DEV"
               --rl-seed 0 --step-index "$step" --ckpt "$CKPT"
               --head-ckpt "$HEAD" --expect-d1-seed "$D1SEED"
               --replica "smoke_$HEADTAG" --out "$out")
      case "$arm" in
        baseline) ;;
        policy)   a+=(--sigma "$SIGMA" --eps-mode train) ;;
        g0|g_rl)  a+=(--sigma "$SIGMA" --eps-mode eval
                      --eval-eps-seed "$EVAL_SEED" --no-buffer) ;;
      esac
      [ -n "$resume" ] && a+=(--resume-head "$resume")
      "${ENVP[@]}" "$PY" experiments/k12d_rollout.py "${a[@]}" >> "$LOG" 2>&1
      rc=$?
      [ "$rc" -eq 0 ] || { say "ОТКАЗ $arm t$T s$S rc=$rc"; return 1; }
    done
  done
  return 0
}

# --- шаг градиента: $1 каталог буфера, $2 resume, $3 индекс, $4 выход
step () {
  local dir="$1" resume="$2" idx="$3" outhead="$4" a=() rc
  a=(--stage diag --rollouts "$(ls "$dir"/*.pt | tr '\n' ',')"
     --replica "smoke_$HEADTAG" --head-ckpt "$HEAD" --step-index "$idx"
     --cb0 data/k12d/cb0.pt --lr "$LR" --halvings "$HALVINGS" --device "$DEV"
     --out-head "$outhead" --out "${outhead%.pt}.json")
  [ -n "$resume" ] && a+=(--resume-head "$resume")
  "$PY" experiments/k12e_pg_step.py "${a[@]}" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { say "ШАГ $idx ОТКАЗАЛ rc=$rc"; return 1; }
  [ -s "$outhead" ] || { say "ШАГ $idx не принят: головы нет"; return 2; }
  return 0
}

MAXSTEP=$(for n in $LADDER; do echo "$n"; done | sort -n | tail -1)

say "=== опорные руки на отложенных состояниях ($EVAL_STARTS) ==="
roll baseline "$ROOT/eval_d1_det" "$EVAL_STARTS" "" 0 || exit 1
roll g0       "$ROOT/eval_g0"     "$EVAL_STARTS" "" 0 || exit 1

PREV=""
for K in $(seq 1 "$MAXSTEP"); do
  say "=== шаг $K: раскатка обучения ==="
  roll policy "$ROOT/train_step$((K-1))" "$TRAIN_STARTS" "$PREV" "$((K-1))" \
    || exit 1
  say "=== шаг $K: градиент ==="
  HEADK="$ROOT/head_step$K.pt"
  if ! step "$ROOT/train_step$((K-1))" "$PREV" "$((K-1))" "$HEADK"; then
    say "лестница остановлена на шаге $K"; break
  fi
  PREV="$HEADK"
  for L in $LADDER; do
    if [ "$L" -eq "$K" ]; then
      say "=== оценка g_rl после $K принятых шагов ==="
      roll g_rl "$ROOT/eval_g_rl_step$K" "$EVAL_STARTS" "$HEADK" "$K" || exit 1
    fi
  done
done

say "=== готово. отчёт: ==="
echo "$PY experiments/k12i_smoke_report.py --root $ROOT" | tee -a "$LOG"
