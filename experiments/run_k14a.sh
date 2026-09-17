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

# УНИКАЛЬНЫЙ НОМЕР ЗАПУСКА. Без него сравнение могло взять артефакт от
# ПРЕДЫДУЩЕГО прогона, оставшийся после сбоя текущего, и объявить режимы
# совпавшими, не посчитав ни одного из них.
RUN_ID="$(date +%Y%m%dT%H%M%S)-$$"
echo "[$(ts)] номер запуска: $RUN_ID"
FAILED=0
declare -A RC
for DEV in cuda:0 cpu; do
    TAG="${DEV/:/}"
    OUT="data/k14a/oracle_cache_${TAG}.json"
    LOG="logs/k14a/oracle_${TAG}.log"
    # СТАРЫЙ АРТЕФАКТ УБИРАЕТСЯ ДО ЗАПУСКА, а не перезаписывается по факту:
    # при сбое он остался бы на месте и выглядел бы свежим.
    rm -f "$OUT"
    echo "[$(ts)] оракул на $DEV -> $OUT"
    # НА CPU КЭШ СОБРАН НЕ БЫЛ: отпечаток ПОВЕДЕНИЯ декодера зависит от
    # устройства, и без флага прогон остановится. Флаг попадает в артефакт,
    # поэтому CPU-результат нельзя будет выдать за полученный в том же режиме.
    EXTRA=()
    [ "$DEV" = "cpu" ] && EXTRA=(--allow-probe-device-drift)
    "${ENVP[@]}" "$PY" experiments/k14a_oracle_cache.py \
        --device "$DEV" --n-rows "$N_ROWS" --out "$OUT" \
        ${EXTRA[@]+"${EXTRA[@]}"} > "$LOG" 2>&1
    RC[$DEV]=$?
    echo "[$(ts)] $DEV: код возврата ${RC[$DEV]}"
    # 0 — обучать головы, 4 — гейт не пройден. Остальное это сбой.
    if [ "${RC[$DEV]}" -ne 0 ] && [ "${RC[$DEV]}" -ne 4 ]; then
        echo "[$(ts)] $DEV: СБОЙ, последние строки $LOG:"
        tail -n 20 "$LOG"
        FAILED=1
    else
        grep -E "РЕШЕНИЕ|ОБУЧАТЬ ГОЛОВЫ|ОТКАЗ|проба кодирования" "$LOG" \
            | sed 's/^/    /'
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "[$(ts)] один из прогонов не состоялся — сравнение режимов НЕ "\
"выполняется: сравнивать было бы нечего или не то"
    exit 6
fi

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
# МЕТКИ СРАВНИВАЮТСЯ ПО ОТПЕЧАТКУ, А НЕ ПО ДОЛЕ РАСХОЖДЕНИЯ. Одинаковая доля
# несовпадений со статической целью не доказывает, что метки те же: одна
# позиция могла перейти из совпадения в расхождение, а другая обратно.
for part in ("train", "val_sel", "val_confirm"):
    for nm in ("q1_ze", "q2_ze", "q1_zq", "q2_zq"):
        ka, kb = a["parts"][part], b["parts"][part]
        sa, sb = ka.get(nm + "_sha1"), kb.get(nm + "_sha1")
        same = (sa == sb and sa is not None
                and ka.get(nm + "_dtype") == kb.get(nm + "_dtype")
                and ka.get(nm + "_shape") == kb.get(nm + "_shape"))
        print(f"  {part}.{nm}: cuda {sa} {ka.get(nm + '_dtype')}  cpu {sb}"
              + ("" if same else "   <-- МЕТКИ РАЗЛИЧАЮТСЯ"))
        if not same:
            bad.append(f"{part}.{nm}")
print("\n  РЕШЕНИЯ И МЕТКИ СОВПАЛИ" if not bad else
      f"\n  РАСХОЖДЕНИЕ ПО {bad}: оракул и тренер нельзя считать в "
      f"разных режимах, это надо внести в план ДО тренера")
sys.exit(0 if not bad else 7)
PY
CMP_RC=$?
echo "[$(ts)] готово: cuda rc=${RC[cuda:0]}, cpu rc=${RC[cpu]}, "\
"сравнение rc=$CMP_RC"
# ФИНАЛЬНЫЙ КОД НЕНУЛЕВОЙ ПРИ ЛЮБОМ РАСХОЖДЕНИИ. Бегунок, всегда
# завершающийся нулём, не является машинной проверкой: его вывод пришлось бы
# читать глазами, и тогда он ничем не лучше ручного запуска.
exit "$CMP_RC"
