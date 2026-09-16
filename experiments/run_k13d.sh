#!/bin/bash
# K-13d: калибровка sigma_T для обеих голов, последовательно.
#
# ПОСЛЕДОВАТЕЛЬНО, А НЕ ПАРАЛЛЕЛЬНО. Выигрыш от двух карт здесь минуты, а цена
# — потерянный код возврата одной из калибровок: в лестницу ушла бы sigma,
# которую никто не проверил. Вторая голова считается только если первая
# прошла.
#
# ЗАПУСК:  nohup bash experiments/run_k13d.sh cuda:0 > logs/k13d/run.log 2>&1 &
set -uo pipefail

DEV="${1:-cuda:0}"
HORIZON="${2:-8}"
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs/k13d data

ts() { date +%H:%M:%S; }

# КАРТЫ ПРОВЕРЯЮТСЯ ДО ЗАГРУЗКИ МОДЕЛЕЙ. Контейнер периодически теряет GPU, и
# без этой проверки отказ пришёл бы через минуту из середины скрипта, где его
# легко принять за ошибку эксперимента.
python - "$DEV" <<'PY' || exit 1
import sys
import torch
dev = sys.argv[1]
if not torch.cuda.is_available():
    raise SystemExit("torch не видит GPU: контейнер потерял карты, "
                     "нужен перезапуск докера с хоста")
i = int(dev.split(":")[1]) if ":" in dev else 0
if i >= torch.cuda.device_count():
    raise SystemExit(f"запрошено {dev}, а карт всего "
                     f"{torch.cuda.device_count()}")
print(f"  карты на месте: {torch.cuda.device_count()}, считаем на {dev}")
PY

rc_all=0
for S in s0 s1; do
    OUT="data/k13d_sigma_t_${S}.json"
    LOG="logs/k13d/${S}.log"
    if [ -s "$OUT" ]; then
        echo "[$(ts)] $S: $OUT уже есть, пропускаю"
        continue
    fi
    echo "[$(ts)] $S: старт"
    python experiments/k13d_calibrate_sigma.py \
        --device "$DEV" --horizon "$HORIZON" \
        --head-t "data/k13b_hicora_t_${S}.pt" \
        --head-d1 "data/k11d/d1_mlp_coef_0.001_wd0_${S}.pt" \
        --out "$OUT" > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(ts)] $S: ОТКАЗ, код $rc. Последние строки $LOG:"
        tail -n 15 "$LOG"
        # ВТОРАЯ ГОЛОВА НЕ СЧИТАЕТСЯ. Отказ на первой — это почти всегда общая
        # причина (карты, происхождение, кэш), и вторая упала бы так же,
        # только на двадцать минут позже.
        exit $rc
    fi
    echo "[$(ts)] $S: готово"
    grep -E "sigma_T|среднее гауссовой|происхождение сверено" "$LOG" | sed 's/^/    /'
done

echo "[$(ts)] обе калибровки готовы:"
for S in s0 s1; do
    python - "data/k13d_sigma_t_${S}.json" <<'PY'
import json, sys
o = json.load(open(sys.argv[1]))
off = abs(o["rms_t"] - o["target_rms"]) / o["target_rms"]
print(f"    {sys.argv[1]}: sigma_T = {o['sigma_t']:.5f}, RMS {o['rms_t']:.5f} "
      f"против цели {o['target_rms']:.5f} (отклонение {100 * off:.2f}%), "
      f"сид {o['head_seed']}, горизонт {o['horizon']}, "
      f"наивный перенос дал бы {o['rms_at_sigma_ref']:.5f}")
PY
done
exit $rc_all
