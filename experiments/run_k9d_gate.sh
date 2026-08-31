#!/usr/bin/env bash
# K-9d: парный симуляторный гейт Joint-12 против грубого выхода на 24 слоях.
#
# ЗАЧЕМ RUNNER. Ограничение то же, что в K-5b/K-6h: одна ячейка на процесс,
# потому что среды нельзя переиспользовать между руками (reset восстанавливает
# состояние не полностью), а пересоздать их после инициализации CUDA нельзя.
# Значит развёртка живёт в оболочке, и ей нужны пропуск готового, ограничение
# на зависание и отдельный лог на ячейку.
#
# СЕТКА: 10 задач x 4 блока начальных состояний x 2 руки = 80 ячеек, то есть
# 400 парных эпизодов на ОДНОМ протоколе ens=on. У K-6h было по 200 пар на
# протокол (два блока), а 400 — это сумма ens=on и ens=off. Здесь основной
# протокол вдвое мощнее опорного; ens=off, если понадобится, считается
# отдельной развёрткой с ENS=off и своим каталогом.
#
# КАТАЛОГ ПРИВЯЗАН К SHA ВЕСОВ. Раннер пропускает любой готовый непустой JSON,
# поэтому общий каталог молча смешал бы ep3 и ep4 или две версии кода. Имя по
# умолчанию содержит sha чекпойнта, и такое смешение становится невозможным.
#
# ЧЕКПОЙНТ КОПИРУЕТСЯ ДО ЗАПУСКА. Если k9c ещё идёт, он перезапишет
# best_imitation.pt в момент, когда очередная эпоха окажется лучше, и часть
# ячеек посчитается другими весами, чем остальные. Скрипт отказывается
# работать по файлу внутри каталога обучения.
#
# НЕ ЗАДАВАЙТЕ CUDA_VISIBLE_DEVICES. robosuite выводит MUJOCO_EGL_DEVICE_ID из
# первого элемента этого списка, а маскировка оставляет процессу ровно одну
# карту с индексом 0 — и EGL падает с «must be an integer between 0 and 0,
# got 1» на КАЖДОЙ ячейке. Стоило 80 падений за 11 минут. Опорные 89.0% в K-6h
# мерились без этого флага; чтобы освободить карту под другое, останавливайте
# обучение, а не маскируйте устройство.
#
# Запуск (каталоги по умолчанию уже содержат sha весов и режим усреднения):
#   cp data/k9c_150k/best_imitation.pt data/k9d_ep3.pt
#   JOINT=data/k9d_ep3.pt \
#       nohup bash experiments/run_k9d_gate.sh > logs/k9d_gate.out 2>&1 &
#   tail -f logs/k9d_gate.out
#
# Разбор (правило чтения записано в докстроке k9d_joint12_gate.py ДО запуска;
# путь каталога печатается раннером в начале и в конце):
#   python3 experiments/k6h_summarize.py --glob 'data/k9d_<sha>_enson/*.json' \
#       --field arm --test fast12 --ref coarse24 --margin 5
set -u

SUITE="${SUITE:-10}"
CKPT="${CKPT:-ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO}"
JOINT="${JOINT:?нужен JOINT=<копия best_imitation.pt вне каталога обучения>}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
INITS="${INITS:-0 10 20 30}"
ARMS="${ARMS:-coarse24 fast12}"
ENS="${ENS:-on}"
HORIZON="${HORIZON:-8}"
N_ENVS="${N_ENVS:-10}"
TIMEOUT="${TIMEOUT:-2400}"

if [ ! -s "$JOINT" ]; then
  echo "нет файла $JOINT"; exit 1
fi
case "$JOINT" in
  *k9c*) echo "ОТКАЗ: $JOINT лежит в каталоге обучения и может быть перезаписан"
         echo "посреди развёртки. Скопируйте его в отдельный файл."; exit 1;;
esac
JSHA=$(sha1sum "$JOINT" | cut -c1-12)

OUT="${OUT:-data/k9d_${JSHA}_ens${ENS}}"
LOGS="${LOGS:-logs/k9d_${JSHA}_ens${ENS}}"

mkdir -p "$OUT" "$LOGS"
export PYTHONPATH="${PYTHONPATH:-$HOME/LIBERO}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

total=0
for T in $TASKS; do for I in $INITS; do for A in $ARMS; do
  total=$((total + 1))
done; done; done

echo "ячеек: $total (10 задач x 4 блока x 2 руки = 400 парных эпизодов при"
echo "  ens=$ENS; у K-6h было по 200 пар на протокол)"
echo "опора K-6h: 89.0% при ens=on, 89.5% при ens=off"
echo "чекпойнт Joint-12: $JOINT (sha весов $JSHA)"
echo "каталоги: $OUT и $LOGS"
echo "начало: $(date '+%F %T')"

done_n=0; skip_n=0; fail_n=0
# СЧЁТЧИКИ ПО РУКАМ. Общего мало: если опора считается, а испытуемая падает,
# успехов уже не ноль, и общий быстрый отказ молчит — а из выборки уходит
# ровно та рука, ради которой всё затеяно. Именно такая потеря не случайна.
declare -A ok_arm fail_arm
for A in $ARMS; do ok_arm[$A]=0; fail_arm[$A]=0; done
t0=$(date +%s)
for T in $TASKS; do
for I in $INITS; do
for A in $ARMS; do
  name="${SUITE}_t${T}_i${I}_${A}_ens${ENS}"
  json="$OUT/$name.json"
  log="$LOGS/$name.log"
  done_n=$((done_n + 1))

  if [ -s "$json" ]; then
    skip_n=$((skip_n + 1))
    echo "[$done_n/$total] пропуск (готово): $name"
    continue
  fi

  # РУКА ОПОРЫ ИДЁТ БЕЗ --joint-ckpt: скрипт откажется его принять, и это
  # намеренно — опора обязана остаться на исходных весах.
  extra=""
  [ "$A" = "fast12" ] && extra="--joint-ckpt $JOINT"

  el=$(( $(date +%s) - t0 ))
  echo "[$done_n/$total] $(date '+%T') прошло $((el / 60)) мин :: $name"
  timeout "$TIMEOUT" python3 experiments/k9d_joint12_gate.py \
      --ckpt "$CKPT" --arm "$A" $extra \
      --task-suite "$SUITE" --task-id "$T" --init-start "$I" \
      --n-envs "$N_ENVS" --horizon "$HORIZON" --ensemble "$ENS" \
      --out "$json" > "$log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    fail_n=$((fail_n + 1))
    echo "    ОШИБКА rc=$rc, см. $log"
    # УДАЛЯЕМ БЕЗУСЛОВНО. Прежняя строка была `[ -s "$json" ] || rm -f`, то
    # есть удаляла файл ровно тогда, когда он и так пуст, а частично
    # записанный НЕПУСТОЙ JSON оставляла — и следующий запуск принимал его за
    # готовый. Ошибка унаследована из run_k5b_sweep.sh. Скрипт пишет JSON
    # одним json.dump в самом конце, поэтому при rc != 0 файл либо не нужен,
    # либо оборван.
    rm -f "$json"
    fail_arm[$A]=$(( ${fail_arm[$A]} + 1 ))
    # БЫСТРЫЙ ОТКАЗ ПО РУКЕ. Две упавшие ячейки одной руки при нуле её
    # успешных — это не невезение, а общая причина: не та переменная
    # окружения, нет чекпойнта, не та глубина. Проходить остальные 38 с той же
    # ошибкой бессмысленно; ровно так 80 падений заняли 11 минут и оставили
    # пустой каталог, а раннер отчитался кодом 0.
    if [ ${ok_arm[$A]} -eq 0 ] && [ ${fail_arm[$A]} -ge 2 ]; then
      echo
      echo "ОСТАНОВКА: у руки $A упало ${fail_arm[$A]} ячеек, успешных нет."
      echo "Причина общая, см. $log — последние строки:"
      tail -5 "$log" | sed 's/^/    /'
      exit 1
    fi
  else
    ok_arm[$A]=$(( ${ok_arm[$A]} + 1 ))
    grep -E "^  рука " "$log" | sed 's/^/  /'
  fi
done; done; done

ok_n=$((done_n - skip_n - fail_n))
echo
echo "конец: $(date '+%F %T'), всего $(( ($(date +%s) - t0) / 60 )) мин"
echo "посчитано $ok_n, пропущено $skip_n, ошибок $fail_n"
echo
echo "ВЕРДИКТ НЕ ЗДЕСЬ. Отдельная ячейка из десяти эпизодов ничего не решает:"
echo "  python3 experiments/k6h_summarize.py --glob '$OUT/*.json' \\"
echo "      --field arm --test fast12 --ref coarse24 --margin 5"
echo
echo "Вердикт принимать только при: $total файлов в $OUT, 400 полных пар,"
echo "10 задач, НИ ОДНОЙ строки «НЕПАРНЫХ», нуле ошибок здесь и одном sha"
echo "весов в отчёте агрегатора."

# КОД ВОЗВРАТА НЕНУЛЕВОЙ ПРИ ЛЮБОЙ УПАВШЕЙ ЯЧЕЙКЕ. Иначе развёртка с
# пропавшими ячейками выглядит успешной, а пропадать чаще может именно
# испытуемая рука — и тогда из выборки уходят её худшие эпизоды.
if [ $fail_n -ne 0 ]; then
  echo "РАЗВЁРТКА НЕПОЛНАЯ: $fail_n ячеек не посчитано. Перезапустите тот же"
  echo "  раннер — готовые ячейки он пропустит."
  exit 1
fi
