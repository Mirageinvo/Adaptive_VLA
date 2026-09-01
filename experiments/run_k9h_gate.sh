#!/usr/bin/env bash
# K-9h: развёртка многорукого гейта.
#
# РУКИ ЗАДАЮТСЯ СПИСКОМ «метка:политика:чекпойнт:сред». Пустой чекпойнт — для
# опорных рук, они обязаны идти на исходных весах, и скрипт откажется принять
# у них --policy-ckpt.
#
# ЧИСЛО БЛОКОВ ВЫВОДИТСЯ ИЗ ЧИСЛА СРЕД, а не задаётся отдельно: покрываются
# всегда одни и те же init_state_id 0..STATES-1. Для контроля численного шума
# это принципиально — рука с батчем 5 обязана пройти РОВНО те же состояния,
# что рука с батчем 10, иначе сравнивались бы разные эпизоды.
#
# RUN_TAG ОБЩИЙ У ВСЕХ РУК ОДНОГО СРАВНЕНИЯ. Он входит в ключ ячейки
# агрегатора; если различать им руки, они попадут в разные эксперименты и не
# сопоставятся вовсе. Различает руки метка.
#
# НЕ ЗАДАВАЙТЕ CUDA_VISIBLE_DEVICES: robosuite выводит из него
# MUJOCO_EGL_DEVICE_ID, и EGL падает на каждой ячейке.
#
# Примеры:
#   # прямая тройка (закрывает перенос не-худшести по цепочке)
#   RUN_TAG=k9h_direct \
#   ARMS="fullbar:fullbar::10 coarse24_b10:coarse24::10 \
#         fast12_rstar:fast:data/k9g_frozen12_rstar.pt:10" \
#   nohup bash experiments/run_k9h_gate.sh > logs/k9h_direct.out 2>&1 &
#
#   # контроль шума: точный повтор и другая форма батча
#   RUN_TAG=k9h_noise \
#   ARMS="coarse24_b10:coarse24::10 coarse24_b10r:coarse24::10 \
#         coarse24_b5:coarse24::5" \
#   nohup bash experiments/run_k9h_gate.sh > logs/k9h_noise.out 2>&1 &
set -u

SUITE="${SUITE:-10}"
CKPT="${CKPT:-ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO}"
RUN_TAG="${RUN_TAG:?нужен RUN_TAG, общий для всех рук сравнения}"
ARMS="${ARMS:?нужен ARMS=\"метка:политика:чекпойнт:сред ...\"}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
STATES="${STATES:-40}"           # начальных состояний на задачу
ENS="${ENS:-on}"
HORIZON="${HORIZON:-8}"
# РЕЖИМ СИДА РАСКАТКИ. `block` повторяет K-6h/K-9d и годится, пока у всех рук
# одно число сред. Как только в сетке появляется рука с другим n_envs, режим
# ОБЯЗАН быть `fixed`: иначе один и тот же init_state_id получит разные сиды
# (при n_envs=10 состояние 5 лежит в блоке 0 и получает сид 0, при n_envs=5 —
# в блоке 5 и сид 5000), и сравнение измерит размер батча вместе с сидом.
# Ниже это проверяется и навязывается автоматически.
SEED_MODE="${SEED_MODE:-auto}"
EXPECT_DEPTH="${EXPECT_DEPTH:-12}"
EXPECT_SOURCE="${EXPECT_SOURCE:-frozen12_rstar}"
# УСТРОЙСТВО ЗАДАЁТСЯ ЯВНО, А НЕ МАСКИРОВКОЙ. CUDA_VISIBLE_DEVICES=1 ломает
# рендеринг: robosuite выводит из него MUJOCO_EGL_DEVICE_ID, а маскировка
# оставляет процессу одно устройство с индексом 0. Здесь модель кладётся на
# DEVICE, рендеринг — на MUJOCO_EGL_DEVICE_ID, и обе переменные независимы,
# потому что CUDA_VISIBLE_DEVICES остаётся незаданным.
DEVICE="${DEVICE:-cuda}"
TIMEOUT="${TIMEOUT:-2400}"
OUT="${OUT:-data/$RUN_TAG}"
LOGS="${LOGS:-logs/$RUN_TAG}"

mkdir -p "$OUT" "$LOGS"
export PYTHONPATH="${PYTHONPATH:-$HOME/LIBERO}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# --- разбор рук и подсчёт ячеек -------------------------------------------
total=0
for spec in $ARMS; do
  IFS=: read -r LAB POL CK NE <<< "$spec"
  if [ -z "${LAB:-}" ] || [ -z "${POL:-}" ] || [ -z "${NE:-}" ]; then
    echo "плохая запись руки: «$spec», нужно метка:политика:чекпойнт:сред"
    exit 1
  fi
  if [ $((STATES % NE)) -ne 0 ]; then
    echo "рука $LAB: $STATES состояний не делится на $NE сред — блоки не"
    echo "покроют тот же набор init_state_id, что у других рук"
    exit 1
  fi
  if [ "$POL" = "fast" ] && [ -z "${CK:-}" ]; then
    echo "рука $LAB: политика fast требует чекпойнт"; exit 1
  fi
  if [ "$POL" != "fast" ] && [ -n "${CK:-}" ]; then
    echo "рука $LAB: опорная политика $POL обязана идти на исходных весах"
    exit 1
  fi
  if [ -n "${CK:-}" ] && [ ! -s "$CK" ]; then
    echo "рука $LAB: нет файла $CK"; exit 1
  fi
  total=$((total + $(echo $TASKS | wc -w) * (STATES / NE)))
  nes="${nes:-} $NE"
done

# РАЗНОЕ ЧИСЛО СРЕД -> РЕЖИМ fixed ОБЯЗАТЕЛЕН. Решается здесь, а не оставляется
# на внимательность запускающего: именно эта деталь превращает контроль шума в
# смесь «размер батча плюс другой сид».
uniq_ne=$(printf '%s\n' $nes | sort -u | wc -l)
if [ "$SEED_MODE" = "auto" ]; then
  if [ "$uniq_ne" -gt 1 ]; then SEED_MODE=fixed; else SEED_MODE=block; fi
fi
if [ "$uniq_ne" -gt 1 ] && [ "$SEED_MODE" != "fixed" ]; then
  echo "ОТКАЗ: у рук разное число сред ($(printf '%s' "$nes")), а SEED_MODE="
  echo "  $SEED_MODE. В режиме block один init_state_id получит разные сиды,"
  echo "  и сравнение измерит размер батча вместе с сидом. Нужен fixed."
  exit 1
fi

echo "эксперимент $RUN_TAG, ячеек $total, ens=$ENS, состояний на задачу $STATES"
echo "режим сида раскатки: $SEED_MODE (число сред: $(printf '%s' "$nes"))"
echo "устройство модели: $DEVICE, рендеринг: MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-по умолчанию}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "ОТКАЗ: задан CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES. robosuite"
  echo "  выводит из него MUJOCO_EGL_DEVICE_ID, и EGL падает на каждой ячейке"
  echo "  (80 падений за 11 минут). Используйте DEVICE=cuda:N и"
  echo "  MUJOCO_EGL_DEVICE_ID=N вместо маскировки."
  exit 1
fi
for spec in $ARMS; do
  IFS=: read -r LAB POL CK NE <<< "$spec"
  s=""
  [ -n "${CK:-}" ] && s=" (sha $(sha1sum "$CK" | cut -c1-12))"
  echo "  рука $LAB: политика $POL, сред $NE, блоков $((STATES / NE))$s"
done
echo "каталоги: $OUT и $LOGS"
echo "начало: $(date '+%F %T')"

done_n=0; skip_n=0; fail_n=0
declare -A ok_arm fail_arm
t0=$(date +%s)

for T in $TASKS; do
for spec in $ARMS; do
  IFS=: read -r LAB POL CK NE <<< "$spec"
  ok_arm[$LAB]="${ok_arm[$LAB]:-0}"; fail_arm[$LAB]="${fail_arm[$LAB]:-0}"
  I=0
  while [ $I -lt $STATES ]; do
    name="${SUITE}_t${T}_i${I}_${LAB}_ens${ENS}"
    json="$OUT/$name.json"
    log="$LOGS/$name.log"
    done_n=$((done_n + 1))

    if [ -s "$json" ]; then
      skip_n=$((skip_n + 1))
      echo "[$done_n/$total] пропуск (готово): $name"
      I=$((I + NE)); continue
    fi

    # ГЛУБИНА И SOURCE ПРОВЕРЯЮТСЯ У КАЖДОЙ ЯЧЕЙКИ. Чекпойнт глубины 18 под
    # меткой fast12_rstar загрузился бы без единой жалобы.
    extra=""
    [ -n "${CK:-}" ] && extra="--policy-ckpt $CK --expect-depth $EXPECT_DEPTH \
--expect-source $EXPECT_SOURCE"

    el=$(( $(date +%s) - t0 ))
    echo "[$done_n/$total] $(date '+%T') прошло $((el / 60)) мин :: $name"
    timeout "$TIMEOUT" python3 experiments/k9h_multiarm_gate.py \
        --ckpt "$CKPT" --policy "$POL" $extra \
        --arm-label "$LAB" --run-tag "$RUN_TAG" \
        --rollout-seed-mode "$SEED_MODE" --device "$DEVICE" \
        --task-suite "$SUITE" --task-id "$T" --init-start "$I" \
        --n-envs "$NE" --horizon "$HORIZON" --ensemble "$ENS" \
        --out "$json" > "$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      fail_n=$((fail_n + 1))
      fail_arm[$LAB]=$(( ${fail_arm[$LAB]} + 1 ))
      echo "    ОШИБКА rc=$rc, см. $log"
      # УДАЛЯЕМ БЕЗУСЛОВНО: JSON пишется одним вызовом в самом конце, поэтому
      # при ненулевом коде он либо не нужен, либо оборван. Прежняя строка
      # `[ -s "$json" ] || rm -f` удаляла файл ровно тогда, когда он и так пуст.
      rm -f "$json"
      # БЫСТРЫЙ ОТКАЗ ПО РУКЕ: две упавшие ячейки одной руки при нуле её
      # успешных — общая причина, а не невезение. Считать по руке
      # обязательно: при нескольких руках общий счётчик молчал бы, пока
      # падает ровно та, ради которой всё затеяно.
      if [ ${ok_arm[$LAB]} -eq 0 ] && [ ${fail_arm[$LAB]} -ge 2 ]; then
        echo; echo "ОСТАНОВКА: у руки $LAB упало ${fail_arm[$LAB]} ячеек, "
        echo "успешных нет. Причина общая, см. $log:"
        tail -5 "$log" | sed 's/^/    /'
        exit 1
      fi
    else
      ok_arm[$LAB]=$(( ${ok_arm[$LAB]} + 1 ))
      grep -E "метка " "$log" | sed 's/^/  /'
    fi
    I=$((I + NE))
  done
done; done

echo
echo "конец: $(date '+%F %T'), всего $(( ($(date +%s) - t0) / 60 )) мин"
echo "посчитано $((done_n - skip_n - fail_n)), пропущено $skip_n, ошибок $fail_n"
echo
echo "ВЕРДИКТ НЕ ЗДЕСЬ. Разбор с обязательными проверками размера:"
echo "  python3 experiments/k6h_summarize.py --glob '$OUT/*.json' \\"
echo "      --field arm_label --test <метка> --ref <метка> --margin 5 \\"
echo "      --expect-pairs $(( $(echo $TASKS | wc -w) * STATES )) \\"
echo "      --expect-tasks $(echo $TASKS | wc -w) --require-full-hash"
echo "Руки сравниваются ПОПАРНО: при трёх руках агрегатор зовётся трижды и"
echo "каждый раз с --allow-extra-arms, либо по подмножеству файлов в --glob."

if [ $fail_n -ne 0 ]; then
  echo "РАЗВЁРТКА НЕПОЛНАЯ: $fail_n ячеек не посчитано. Перезапустите тот же"
  echo "  раннер — готовые ячейки он пропустит."
  exit 1
fi
