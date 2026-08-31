#!/usr/bin/env bash
# K-9d: парный симуляторный гейт Joint-12 против грубого выхода на 24 слоях.
#
# ЗАЧЕМ RUNNER. Ограничение то же, что в K-5b/K-6h: одна ячейка на процесс,
# потому что среды нельзя переиспользовать между руками (reset восстанавливает
# состояние не полностью), а пересоздать их после инициализации CUDA нельзя.
# Значит развёртка живёт в оболочке, и ей нужны пропуск готового, ограничение
# на зависание и отдельный лог на ячейку.
#
# СЕТКА: 10 задач x 4 блока начальных состояний x 2 руки = 80 ячеек,
# 400 парных эпизодов — ровно как в K-6h, чтобы опора 89.0% была сравнима.
#
# ЧЕКПОЙНТ КОПИРУЕТСЯ ДО ЗАПУСКА. Если k9c ещё идёт, он перезапишет
# best_imitation.pt в момент, когда очередная эпоха окажется лучше, и часть
# ячеек посчитается другими весами, чем остальные. Скрипт отказывается
# работать по файлу внутри каталога обучения.
#
# Запуск:
#   cp data/k9c_150k/best_imitation.pt data/k9d_ep2.pt
#   JOINT=data/k9d_ep2.pt nohup bash experiments/run_k9d_gate.sh \
#       > logs/k9d_gate.out 2>&1 &
#   tail -f logs/k9d_gate.out
#
# Разбор (правило чтения записано в докстроке k9d_joint12_gate.py ДО запуска):
#   python3 experiments/k6h_summarize.py --glob 'data/k9d/*.json' \
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
OUT="${OUT:-data/k9d}"
LOGS="${LOGS:-logs/k9d}"

if [ ! -s "$JOINT" ]; then
  echo "нет файла $JOINT"; exit 1
fi
case "$JOINT" in
  *k9c*) echo "ОТКАЗ: $JOINT лежит в каталоге обучения и может быть перезаписан"
         echo "посреди развёртки. Скопируйте его в отдельный файл."; exit 1;;
esac
JSHA=$(sha1sum "$JOINT" | cut -c1-12)

mkdir -p "$OUT" "$LOGS"
export PYTHONPATH="${PYTHONPATH:-$HOME/LIBERO}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

total=0
for T in $TASKS; do for I in $INITS; do for A in $ARMS; do
  total=$((total + 1))
done; done; done

echo "ячеек: $total (10 задач x 4 блока x 2 руки = 400 парных эпизодов)"
echo "опора K-6h при ens=$ENS: 89.0% (on) / 89.5% (off)"
echo "чекпойнт Joint-12: $JOINT (sha $JSHA)"
echo "начало: $(date '+%F %T')"

done_n=0; skip_n=0; fail_n=0
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
    # частично записанный JSON удаляем, иначе он будет принят за готовый
    [ -s "$json" ] || rm -f "$json"
  else
    grep -E "^  рука " "$log" | sed 's/^/  /'
  fi
done; done; done

echo
echo "конец: $(date '+%F %T'), всего $(( ($(date +%s) - t0) / 60 )) мин"
echo "посчитано $((done_n - skip_n - fail_n)), пропущено $skip_n, ошибок $fail_n"
echo
echo "ВЕРДИКТ НЕ ЗДЕСЬ. Отдельная ячейка из десяти эпизодов ничего не решает:"
echo "  python3 experiments/k6h_summarize.py --glob '$OUT/*.json' \\"
echo "      --field arm --test fast12 --ref coarse24 --margin 5"
