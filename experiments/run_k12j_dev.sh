#!/usr/bin/env bash
# K-12j: development-обучение HiCoRA-G. Настоящий цикл, а не предварительный
# замер.
#
# ОДИН ПРИНЯТЫЙ ШАГ = один свежий on-policy буфер. Буфер, по которому уже
# сделано обновление, для второго обновления не используется: политика после
# шага другая, и сэмплы к ней не относятся.
#
# ПРОДОЛЖЕНИЕ С ЛЮБОГО ПРИНЯТОГО ШАГА. Раннер находит последнюю голову
# head_after_NNNN.pt, проверяет цепочку и продолжает с N+1; шаги 1..N не
# пересчитываются. step_index всюду означает ЧИСЛО УЖЕ ПРИНЯТЫХ ОБНОВЛЕНИЙ.
#
# ИЗОЛЯЦИЯ. Каталог включает сид D1, сид RL, sigma и список задач; состав
# прогона записан в config.json и сверяется. Готовая ячейка пропускается только
# после сверки ЕЁ конфигурации, а не по признаку «файл непустой».
#
#   bash experiments/run_k12j_dev.sh cuda:0 s0 4          # довести до 4 шагов
#   TARGET=10 bash experiments/run_k12j_dev.sh cuda:0 s0  # продолжить до 10
set -u -o pipefail

DEV="${1:?нужно устройство}"
HEADTAG="${2:?нужна голова: s0 или s1}"
TARGET="${3:-${TARGET:-4}}"
SIGMA="${SIGMA:-0.10}"
RL_SEED="${RL_SEED:-0}"
# Задачи задаются ПАРАМИ сюита:номер — object/0 и goal/0 это разные задачи.
# Голое число понимается как задача сюиты 10 (совместимость с этапом A).
TASKS="${TASKS:-3 6 8}"
TRAIN_STARTS="${TRAIN_STARTS:-0 5 10 15 20 25}"
EVAL_STARTS="${EVAL_STARTS:-30 35 40}"
LADDER="${LADDER:-1 2 4 6 8 10 12 16 20}"
NENV="${NENV:-5}"
# Предел шагов в эпизоде. Для настоящего прогона — 600, как в K-9h/K-11g; для
# быстрой проверки связности его снижают, и тогда ячейка считается за полминуты
# вместо трёх. Значение входит в состав ячейки, поэтому проверочные результаты
# нельзя случайно принять за настоящие.
MAXSTEPS="${MAXSTEPS:-600}"
# Машина общая: чужой процесс может занять карту целиком, и один такой отказ
# роняет всю лестницу. Повтор делается ТОЛЬКО при нехватке памяти — по тексту
# ошибки, а не по коду возврата: слепой повтор прятал бы настоящие ошибки.
RETRIES="${RETRIES:-3}"
RETRY_WAIT="${RETRY_WAIT:-300}"
EVAL_SEED="${EVAL_SEED:-777}"
LR="${LR:-1e-2}"
HALVINGS="${HALVINGS:-12}"
CKPT="${CKPT:-ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO}"
# ВИД ГОЛОВЫ — ПЕРЕМЕННАЯ СО СТАРЫМ ЗНАЧЕНИЕМ ПО УМОЛЧАНИЮ. Позиционные
# прогоны K-12j должны запускаться ровно так же, как запускались, вплоть до
# строки config.json: иначе уже посчитанные каталоги стали бы «чужими».
PY="${PY:-python}"
HEADKIND="${HEADKIND:-positional}"
case "$HEADKIND" in
  positional)
    case "$HEADTAG" in
      s0) HEAD="${HEAD:-data/k11d/d1_mlp_coef_0.001_wd0_s0.pt}"; D1SEED=0 ;;
      s1) HEAD="${HEAD:-data/k11d/d1_mlp_coef_0.001_wd0_s1.pt}"; D1SEED=1 ;;
      *) echo "голова должна быть s0 или s1"; exit 1 ;;
    esac
    KINDARGS=()
    OKARGS=()
    ;;
  trajectory)
    case "$HEADTAG" in
      s0) HEAD="${HEAD:-data/k13b_hicora_t_s0.pt}"; D1SEED=0 ;;
      s1) HEAD="${HEAD:-data/k13b_hicora_t_s1.pt}"; D1SEED=1 ;;
      *) echo "голова должна быть s0 или s1"; exit 1 ;;
    esac
    TRAJ_BASIS="${TRAJ_BASIS:-data/k13a_traj_basis}"
    SIGMA_JSON="${SIGMA_JSON:-data/k13d_sigma_t_${HEADTAG}.json}"
    [ -s "$SIGMA_JSON" ] || { echo "нет артефакта калибровки $SIGMA_JSON: "\
"рабочая точка sigma_T берётся из него, а не из командной строки"; exit 1; }
    # SIGMA БЕРЁТСЯ ИЗ АРТЕФАКТА, А НЕ ИЗ ОКРУЖЕНИЯ. Набранное руками число
    # разошлось бы с калиброванным в последнем знаке, и k12d отверг бы запуск —
    # правильно, но бессмысленно.
    SIGMA=$($PY -c "import json;print(round(json.load(open('$SIGMA_JSON'))['sigma_t'],6))")
    KINDARGS=(--head-kind trajectory --traj-basis "$TRAJ_BASIS"
              --sigma-json "$SIGMA_JSON")
    # ПРИ ПРОПУСКЕ ГОТОВОЙ ЯЧЕЙКИ СВЕРЯЕТСЯ И ВЕТВЬ. Без этого ячейка, лежащая
    # по тому же пути, но снятая другой головой или другой калибровкой, была бы
    # принята: имя файла о них ничего не говорит.
    HSHA=$($PY -c "import hashlib;print(hashlib.sha1(open('$HEAD','rb').read()).hexdigest()[:12])")
    SJSHA=$($PY -c "import hashlib;print(hashlib.sha1(open('$SIGMA_JSON','rb').read()).hexdigest()[:12])")
    OKARGS=(head_kind=trajectory head_sha1="$HSHA"
            sigma_json_sha1="$SJSHA")
    ;;
  *) echo "HEADKIND должен быть positional или trajectory"; exit 1 ;;
esac
KINDTAG=""; [ "$HEADKIND" = "trajectory" ] && KINDTAG="t_"
TAG="${TAG:-${KINDTAG}${HEADTAG}_sig${SIGMA}_rl${RL_SEED}_t$(echo $TASKS | tr -d ' :')}"
ROOT="${ROOT:-data/k12j/$TAG}"
LOG="${LOG:-logs/k12j/$TAG.log}"
ENVP=(env PYTHONPATH="$HOME/LIBERO" MUJOCO_GL=egl)

mkdir -p "$ROOT" "$(dirname "$LOG")"
[ -f "$HEAD" ] || { echo "нет головы D1: $HEAD"; exit 1; }
[ -s data/k12d/cb0.pt ] || { echo "нет data/k12d/cb0.pt: создайте её однажды "\
"через --cb0-only"; exit 1; }

say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# --- состав прогона фиксируется один раз и дальше только сверяется ---------
# БЛОКИРУЕТ ТОЛЬКО НАУЧНЫЙ СОСТАВ: сиды, sigma, задачи, состояния. Версия кода
# в него НЕ входит и записывается отдельно, историей: иначе исправление любой
# ошибки делало бы каталог «чужим», и продолжить начатый прогон было бы нельзя
# — а несовместимость кода и данных всё равно ловится построчной сверкой
# происхождения каждой ячейки.
CFG="$ROOT/config.json"
# СТРОКА СОСТАВА ДЛЯ ПОЗИЦИОННЫХ ПРОГОНОВ ОСТАЁТСЯ ПРЕЖНЕЙ ДО СИМВОЛА:
# добавление поля сделало бы уже посчитанные каталоги K-12j несовпадающими, и
# продолжить начатый прогон стало бы нельзя. У траекторной ветви каталоги
# новые, поэтому там состав пишется расширенным.
if [ "$HEADKIND" = "trajectory" ]; then
  NEW=$(printf '{"d1_seed":%d,"rl_seed":%d,"sigma":%s,"tasks":"%s","train_starts":"%s","eval_starts":"%s","n_envs":%d,"eval_eps_seed":%d,"head":"%s","head_kind":"%s","sigma_json":"%s","traj_basis":"%s"}' \
    "$D1SEED" "$RL_SEED" "$SIGMA" "$TASKS" "$TRAIN_STARTS" "$EVAL_STARTS" \
    "$NENV" "$EVAL_SEED" "$HEAD" "$HEADKIND" "$SIGMA_JSON" "$TRAJ_BASIS")
else
  NEW=$(printf '{"d1_seed":%d,"rl_seed":%d,"sigma":%s,"tasks":"%s","train_starts":"%s","eval_starts":"%s","n_envs":%d,"eval_eps_seed":%d,"head":"%s"}' \
    "$D1SEED" "$RL_SEED" "$SIGMA" "$TASKS" "$TRAIN_STARTS" "$EVAL_STARTS" \
    "$NENV" "$EVAL_SEED" "$HEAD")
fi
if [ -f "$CFG" ]; then
  if [ "$(cat "$CFG")" != "$NEW" ]; then
    echo "научный состав каталога $ROOT не совпадает с запрошенным:"
    echo "  было:  $(cat "$CFG")"
    echo "  стало: $NEW"
    echo "старые файлы не трогаю — задайте другой TAG"; exit 1
  fi
else
  printf '%s' "$NEW" > "$CFG"
fi
printf '{"time":"%s","k12d":"%s","k12e":"%s","target":%s}\n' \
  "$(date -Is)" \
  "$($PY -c "import hashlib;print(hashlib.sha1(open('experiments/k12d_rollout.py','rb').read()).hexdigest()[:12])")" \
  "$($PY -c "import hashlib;print(hashlib.sha1(open('experiments/k12e_pg_step.py','rb').read()).hexdigest()[:12])")" \
  "$TARGET" >> "$ROOT/script_versions.jsonl"

# --- раскатка: $1 рука, $2 каталог, $3 starts, $4 resume, $5 принято шагов --
roll () {
  local arm="$1" dir="$2" starts="$3" resume="$4" step="$5" rc out ext sg SU T legacy
  mkdir -p "$dir"
  # расширение и ожидаемая sigma — по руке, без цепочек && ||, где первый же
  # ложный шаг молча меняет смысл выражения
  case "$arm" in
    policy)              ext="pt";   sg="$SIGMA" ;;
    g0|g_rl)             ext="json"; sg="$SIGMA" ;;
    baseline|g_rl_mean)  ext="json"; sg="0.0" ;;
    *) say "неизвестная рука $arm"; return 1 ;;
  esac
  for TT in $TASKS; do
    # ЗАДАЧА — ПАРА сюита:номер. object/0 и goal/0 это разные задачи, и
    # различать их одним числом нельзя. Голое число понимается как сюита 10.
    case "$TT" in
      *:*) SU="${TT%%:*}"; T="${TT##*:}" ;;
      *)   SU="10";        T="$TT"       ;;
    esac
    for S in $starts; do
      # ИМЯ С СЮИТОЙ — новое; ячейки этапа A лежат под старым, без неё.
      # Искать только по новому значило бы пересчитать всё уже посчитанное,
      # поэтому старое имя принимается, если сама ячейка проходит сверку.
      out="$dir/${SU}_t${T}_s${S}.$ext"
      legacy="$dir/t${T}_s${S}.$ext"
      if [ ! -e "$out" ] && [ -e "$legacy" ]; then out="$legacy"; fi
      # ПРЕДЕЛ ШАГОВ И ЧИСЛО СРЕД — ЧАСТЬ СОСТАВА ЯЧЕЙКИ. Без их сверки
      # ячейка, посчитанная с укороченным пределом (проверочный прогон), молча
      # подошла бы настоящему: там, где эпизод обрывается раньше, успех
      # означает другое.
      $PY experiments/k12j_cell_ok.py "$out" arm="$arm" sigma="$sg" \
        step_index="$step" task_id="$T" init_start="$S" d1_seed="$D1SEED" \
        rl_seed="$RL_SEED" max_steps="$MAXSTEPS" n_envs="$NENV" suite="$SU" \
        ${OKARGS[@]+"${OKARGS[@]}"} >/dev/null 2>>"$LOG"
      case $? in
        0) continue ;;                      # годная ячейка уже есть
        2) say "ЯЧЕЙКА $out ЕСТЬ, НО ОТ ДРУГОЙ КОНФИГУРАЦИИ — остановка"
           return 2 ;;
      esac
      local a=(--stage diag --arm "$arm" --task-suite "$SU" --task-id "$T"
               --init-start "$S" --n-envs "$NENV" --device "$DEV"
               --rl-seed "$RL_SEED" --step-index "$step" --ckpt "$CKPT"
               --head-ckpt "$HEAD" --expect-d1-seed "$D1SEED"
               --replica "dev_${KINDTAG}${HEADTAG}_rl${RL_SEED}"
               --max-steps "$MAXSTEPS" --out "$out"
               ${KINDARGS[@]+"${KINDARGS[@]}"})
      case "$arm" in
        baseline) ;;
        # sigma ВЕТВИ: исполняется среднее, но голова должна быть из этой ветви
        g_rl_mean) a+=(--sigma "$SIGMA") ;;
        policy)   a+=(--sigma "$SIGMA" --eps-mode train) ;;
        g0|g_rl)  a+=(--sigma "$SIGMA" --eps-mode eval
                      --eval-eps-seed "$EVAL_SEED" --no-buffer) ;;
      esac
      [ -n "$resume" ] && a+=(--resume-head "$resume")
      local try=0
      while : ; do
        "${ENVP[@]}" "$PY" experiments/k12d_rollout.py "${a[@]}" >> "$LOG" 2>&1
        rc=$?
        [ "$rc" -eq 0 ] && break
        if tail -40 "$LOG" | grep -q "OutOfMemoryError" \
           && [ "$try" -lt "$RETRIES" ]; then
          try=$((try + 1))
          say "нехватка памяти на $DEV ($arm $SU/$T s$S), попытка $try из "\
"$RETRIES через $RETRY_WAIT с"
          sleep "$RETRY_WAIT"
          continue
        fi
        # rc=137 — SIGKILL, на этой машине это системный OOM-killer: питон не
        # успевает ничего написать, поэтому распознаётся по коду, а не по тексту
        if [ "$rc" -eq 137 ] && [ "$try" -lt "$RETRIES" ]; then
          try=$((try + 1))
          say "процесс убит по памяти ($arm $SU/$T s$S), попытка $try из "\
"$RETRIES через $RETRY_WAIT с"
          sleep "$RETRY_WAIT"
          continue
        fi
        say "ОТКАЗ $arm $SU/t$T s$S rc=$rc"
        return 1
      done
    done
  done
  return 0
}

head_path () { printf "%s/head_after_%04d.pt" "$ROOT" "$1"; }

# --- сколько обновлений уже принято ---------------------------------------
DONE=0
while [ -s "$(head_path $((DONE + 1)))" ]; do DONE=$((DONE + 1)); done
say "принято обновлений: $DONE, цель $TARGET (каталог $ROOT)"

# --- опорные руки считаются один раз --------------------------------------
say "=== опорные руки на отложенных состояниях ($EVAL_STARTS) ==="
roll baseline "$ROOT/eval_d1_det" "$EVAL_STARTS" "" 0 || exit 1
roll g0       "$ROOT/eval_g0"     "$EVAL_STARTS" "" 0 || exit 1

in_ladder () { for L in $LADDER; do [ "$L" -eq "$1" ] && return 0; done; return 1; }

evaluate () {
  local n="$1" hp; hp="$(head_path "$n")"
  say "=== оценка после $n обновлений ==="
  roll g_rl      "$ROOT/eval_g_rl_step$n"      "$EVAL_STARTS" "$hp" "$n" || return 1
  roll g_rl_mean "$ROOT/eval_g_rl_mean_step$n" "$EVAL_STARTS" "$hp" "$n" || return 1
}

# уже принятые шаги, попавшие в лестницу: evaluate идемпотентна — годные
# ячейки она пропустит после сверки конфигурации, а недостающие досчитает.
# Угадывать имя первой ячейки, чтобы решить, считать ли, значило бы снова
# полагаться на имя файла вместо его содержимого.
for n in $LADDER; do
  [ "$n" -le "$DONE" ] || continue
  evaluate "$n" || exit 1
done

while [ "$DONE" -lt "$TARGET" ]; do
  N=$DONE; K=$((DONE + 1))
  PREV=""; [ "$N" -gt 0 ] && PREV="$(head_path "$N")"
  say "=== обновление $K: свежий буфер политикой после $N обновлений ==="
  roll policy "$ROOT/train_after_$N" "$TRAIN_STARTS" "$PREV" "$N" || exit 1
  say "=== обновление $K: градиент ==="
  ARGS=(--stage diag --rollouts "$(ls "$ROOT/train_after_$N"/*.pt | tr '\n' ',')"
        --replica "dev_${KINDTAG}${HEADTAG}_rl${RL_SEED}" --head-ckpt "$HEAD"
        --step-index "$N" --rl-seed "$RL_SEED" --cb0 data/k12d/cb0.pt
        --lr "$LR" --halvings "$HALVINGS" --device "$DEV"
        --out-head "$(head_path "$K")" --out "$ROOT/step_$K.json")
  # ВИД ГОЛОВЫ ШАГ ОПРЕДЕЛЯЕТ ПО РАСКАТКАМ САМ; ему нужен только путь базиса.
  [ "$HEADKIND" = "trajectory" ] && ARGS+=(--traj-basis "$TRAJ_BASIS")
  [ -n "$PREV" ] && ARGS+=(--resume-head "$PREV")
  "$PY" experiments/k12e_pg_step.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { say "ШАГ $K ОТКАЗАЛ rc=$rc"; exit 1; }
  [ -s "$(head_path "$K")" ] || { say "ШАГ $K НЕ ПРИНЯТ (no_step) — остановка"
                                  exit 3; }
  DONE=$K
  in_ladder "$K" && { evaluate "$K" || exit 1; }
done

say "=== готово: принято $DONE обновлений ==="
echo "$PY experiments/k12i_smoke_report.py --root $ROOT" | tee -a "$LOG"
