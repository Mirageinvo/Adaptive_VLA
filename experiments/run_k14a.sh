#!/usr/bin/env bash
# K-14a: условный оракул в двух режимах вычислений, последовательно.
#
# ЗАЧЕМ ДВА РЕЖИМА. После §39 (смена точности переворачивает четверть исходов)
# считать оракул на CPU, а тренер обучать на GPU нельзя без измерения. Здесь
# расхождение измеряется, а не предполагается.
#
# КОД 4 — ЭТО РЕЗУЛЬТАТ, А НЕ ОШИБКА. Гейт не пройден означает «не обучать
# головы», и бегунок обязан довести обе части до конца и сравнить их, а не
# упасть на первой. Поэтому `set -e` здесь нет, а коды собираются явно.
#
# ЗАПУСК: nohup bash experiments/run_k14a.sh > logs/k14a/run.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
mkdir -p data/k14a logs/k14a

N_ROWS="${N_ROWS:-4096}"
PY="${PY:-python}"
ENVP=(env PYTHONPATH="${PYTHONPATH:-$HOME/LIBERO}" MUJOCO_GL=egl)
ts() { date +%H:%M:%S; }

echo "[$(ts)] самопроверки"
"${ENVP[@]}" "$PY" experiments/depth_rvq_joint12.py || exit 1
"${ENVP[@]}" "$PY" experiments/k14a_oracle_cache.py --selftest || exit 1

declare -A RC
for DEV in cuda:0 cpu; do
    TAG="${DEV/:/}"
    OUT="data/k14a/oracle_cache_${TAG}.json"
    LOG="logs/k14a/oracle_${TAG}.log"
    echo "[$(ts)] оракул на $DEV -> $OUT"
    "${ENVP[@]}" "$PY" experiments/k14a_oracle_cache.py \
        --device "$DEV" --n-rows "$N_ROWS" --out "$OUT" > "$LOG" 2>&1
    RC[$DEV]=$?
    echo "[$(ts)] $DEV: код возврата ${RC[$DEV]}"
    # 0 — обучать головы, 4 — гейт не пройден. Остальное это сбой.
    if [ "${RC[$DEV]}" -ne 0 ] && [ "${RC[$DEV]}" -ne 4 ]; then
        echo "[$(ts)] $DEV: СБОЙ, последние строки $LOG:"
        tail -n 20 "$LOG"
    else
        grep -E "РЕШЕНИЕ|ОБУЧАТЬ ГОЛОВЫ|ОТКАЗ|проба кодирования" "$LOG" \
            | sed 's/^/    /'
    fi
done

echo "[$(ts)] сравнение режимов"
"$PY" - <<'PY'
import json, os, sys
pa, pb = 'data/k14a/oracle_cache_cuda0.json', 'data/k14a/oracle_cache_cpu.json'
if not (os.path.exists(pa) and os.path.exists(pb)):
    print("  один из артефактов отсутствует — сравнивать нечего")
    sys.exit(0)
a, b = json.load(open(pa)), json.load(open(pb))
bad = []
for k in ("latent_capacity_ok", "action_oracle_ok",
          "dynamic_q1_relabeling_supported", "train_heads"):
    same = a[k] == b[k]
    print(f"  {k:32s} cuda {a[k]}   cpu {b[k]}"
          + ("" if same else "   <-- РЕШЕНИЯ РАСХОДЯТСЯ"))
    if not same:
        bad.append(k)
for part in ("val_confirm", "val_sel"):
    for m in ("A0", "A01_ze", "A012_ze", "Acodec"):
        ra = a["parts"][part]["vs_action." + m]["rms"]
        rb = b["parts"][part]["vs_action." + m]["rms"]
        print(f"  {part}.{m:9s} RMS-8  cuda {ra:.6f}  cpu {rb:.6f}  "
              f"|Δ| {abs(ra - rb):.2e}")
    for nm in ("dynamic_ze_vs_static_q1_disagree",
               "recovery_012_ze_vs_action"):
        va, vb = a["parts"][part].get(nm), b["parts"][part].get(nm)
        print(f"  {part}.{nm}: cuda {va}  cpu {vb}")
print("\n  РЕШЕНИЯ СОВПАЛИ" if not bad else
      f"\n  РЕШЕНИЯ РАСХОДЯТСЯ ПО {bad}: оракул и тренер нельзя считать в "
      f"разных режимах, это надо внести в план ДО тренера")
PY
echo "[$(ts)] готово: cuda rc=${RC[cuda:0]}, cpu rc=${RC[cpu]}"
