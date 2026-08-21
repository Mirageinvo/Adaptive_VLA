#!/usr/bin/env bash
# K-5b: полная развёртка по горизонту исполнения на одном suite.
#
# ЗАЧЕМ ОТДЕЛЬНЫЙ RUNNER. Одна конфигурация на процесс — вынужденное
# ограничение k5b_fixed_horizon_eval.py: среды нельзя переиспользовать между
# конфигурациями (reset восстанавливает состояние не полностью), а пересоздать
# их нельзя после загрузки модели (fork после инициализации CUDA вешает
# процесс). Значит развёртка живёт в оболочке, и ей нужны три вещи, которых
# нет у голого цикла: пропуск уже посчитанного, ограничение на зависание и
# отдельный лог на конфигурацию.
#
# ПРОПУСК УЖЕ ПОСЧИТАННОГО обязателен: восьмидесятичасовой прогон почти
# наверняка будет прерван, и повторять с нуля нельзя. Готовым считается только
# НЕПУСТОЙ JSON — оборванный на записи файл будет пересчитан.
#
# Запуск:
#   nohup bash experiments/run_k5b_sweep.sh 10 > logs/k5b_sweep.out 2>&1 &
#   tail -f logs/k5b_sweep.out
set -u

SUITE="${1:-10}"
CKPT="${CKPT:-ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
HORIZONS="${HORIZONS:-4 8 12 20}"
MODES="${MODES:-off on}"
N_ENVS="${N_ENVS:-10}"
K_SET="${K_SET:-5}"          # 10 x 5 = 50 эпизодов, как в официальном протоколе
TIMEOUT="${TIMEOUT:-2400}"   # 40 мин на конфигурацию: страховка от зависания
OUT="${OUT:-data/k5b_sweep}"
LOGS="${LOGS:-logs/k5b_sweep}"

mkdir -p "$OUT" "$LOGS"
export PYTHONPATH="${PYTHONPATH:-$HOME/LIBERO}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

total=0
for T in $TASKS; do for E in $MODES; do for H in $HORIZONS; do
  total=$((total + 1))
done; done; done

echo "суммарно конфигураций: $total"
echo "suite=$SUITE, эпизодов на конфигурацию: $((N_ENVS * K_SET))"
echo "начало: $(date '+%F %T')"

done_n=0; skip_n=0; fail_n=0
t0=$(date +%s)
for T in $TASKS; do
for E in $MODES; do
for H in $HORIZONS; do
  name="${SUITE}_t${T}_${E}_H${H}"
  json="$OUT/$name.json"
  log="$LOGS/$name.log"
  done_n=$((done_n + 1))

  if [ -s "$json" ]; then
    skip_n=$((skip_n + 1))
    echo "[$done_n/$total] пропуск (готово): $name"
    continue
  fi

  el=$(( $(date +%s) - t0 ))
  echo "[$done_n/$total] $(date '+%T') прошло $((el / 60)) мин :: $name"
  timeout "$TIMEOUT" python3 experiments/k5b_fixed_horizon_eval.py \
      --ckpt "$CKPT" --task-suite "$SUITE" --task-id "$T" \
      --n-envs "$N_ENVS" --k-set "$K_SET" \
      --horizons "$H" --ensemble "$E" --out "$json" > "$log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    fail_n=$((fail_n + 1))
    echo "    ОШИБКА rc=$rc, см. $log"
    # частично записанный JSON удаляем, иначе он будет принят за готовый
    [ -s "$json" ] || rm -f "$json"
  else
    grep -E "^    ens=" "$log" | sed 's/^/    /'
  fi
done; done; done

echo
echo "конец: $(date '+%F %T'), всего $(( ($(date +%s) - t0) / 60 )) мин"
echo "посчитано $((done_n - skip_n - fail_n)), пропущено $skip_n, ошибок $fail_n"
echo
python3 experiments/k5b_summarize.py --dir "$OUT" || true
