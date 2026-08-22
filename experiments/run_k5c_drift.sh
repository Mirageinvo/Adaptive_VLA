#!/usr/bin/env bash
# K-5c: сбор кривых устаревания чанка на одном suite.
#
# ДВА ГОРИЗОНТА ИСПОЛНЕНИЯ ОБЯЗАТЕЛЬНЫ. Внутри чанка исполнение открытоцикловое
# при любом H_exec, поэтому смещения меряются на правильных состояниях. Смещено
# другое — РАСПРЕДЕЛЕНИЕ ТОЧЕК ГЕНЕРАЦИИ: при H_exec=20 они лежат на траектории
# H=20, а разворачивать метод предполагается при H≈8. Сравнение кривых между
# двумя прогонами показывает, важен ли этот сдвиг. Совпали — дальше можно брать
# только дешёвый.
#
# СТОИМОСТЬ. Политика вызывается КАЖДЫЙ шаг, то есть примерно вчетверо чаще,
# чем в ячейке H=4 развёртки K-5b. Отсюда k-set по умолчанию 2 (20 эпизодов),
# а не 5: кривая плотная — на эпизод приходится ~400 моментов × 20 смещений,
# так что двадцати эпизодов хватает с запасом, а восьмидесяти часов у нас нет.
#
# Запуск:
#   nohup bash experiments/run_k5c_drift.sh 10 > logs/k5c_drift.out 2>&1 &
#   tail -f logs/k5c_drift.out
set -u

SUITE="${1:-10}"
CKPT="${CKPT:-ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
EXEC_H="${EXEC_H:-20 8}"
N_ENVS="${N_ENVS:-10}"
K_SET="${K_SET:-2}"
TIMEOUT="${TIMEOUT:-5400}"
OUT="${OUT:-data/k5c_drift}"
LOGS="${LOGS:-logs/k5c_drift}"

mkdir -p "$OUT" "$LOGS"
export PYTHONPATH="${PYTHONPATH:-$HOME/LIBERO}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# самопроверка ОДИН раз до всего: если инструмент сломан, прогон бессмыслен
python3 experiments/k5c_drift_probe.py --selftest || exit 1

total=0
for T in $TASKS; do for H in $EXEC_H; do total=$((total + 1)); done; done
echo "конфигураций: $total, эпизодов на конфигурацию: $((N_ENVS * K_SET))"
echo "начало: $(date '+%F %T')"

n=0; skip=0; fail=0
t0=$(date +%s)
for T in $TASKS; do
for H in $EXEC_H; do
  name="${SUITE}_t${T}_H${H}"
  npz="$OUT/$name.npz"
  n=$((n + 1))

  if [ -s "$npz" ]; then
    skip=$((skip + 1)); echo "[$n/$total] пропуск (готово): $name"; continue
  fi
  echo "[$n/$total] $(date '+%T') прошло $((($(date +%s) - t0) / 60)) мин :: $name"
  timeout "$TIMEOUT" python3 experiments/k5c_drift_probe.py \
      --ckpt "$CKPT" --task-suite "$SUITE" --task-id "$T" \
      --n-envs "$N_ENVS" --k-set "$K_SET" --exec-horizon "$H" \
      --out "$npz" > "$LOGS/$name.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    fail=$((fail + 1))
    echo "    ОШИБКА rc=$rc, см. $LOGS/$name.log"
    [ -s "$npz" ] || rm -f "$npz"
  else
    grep -E "раунд |тёплиц" "$LOGS/$name.log" | sed 's/^/    /'
  fi
done; done

echo
echo "конец: $(date '+%F %T'), всего $((($(date +%s) - t0) / 60)) мин"
echo "посчитано $((n - skip - fail)), пропущено $skip, ошибок $fail"
