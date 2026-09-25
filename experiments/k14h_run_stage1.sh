#!/bin/bash
# §49, ЭТАП 1: три архитектуры под правилом отбора, последовательно.
#
# ЧТО СРАВНИВАЕТСЯ. baseline — нынешняя модель; reattn_state — тот же блок
# внимания, но во второй слот запроса подан ФИКСИРОВАННЫЙ нулевой черновик;
# reattn_draft — тот же блок с настоящим q0. Первая пара отвечает на вопрос
# «помогает ли лишний проход по префиксу сам по себе», вторая — «помогает ли
# ИМЕННО знание черновика». Без среднего звена превосходство reattn_draft
# объяснялось бы лишней ёмкостью, а не информацией q0.
#
# ОДИН ОБЪЕКТ В ДВУХ РЕЖИМАХ. reattn_state и reattn_draft — не две сборки, а
# один блок с переключённым слотом; число параметров у них совпадает
# побитово. Это проверено архитектурным гейтом до запуска.
#
# РЕЖИМ ОТБОРА, А НЕ ПОЛНЫЙ ПРОТОКОЛ. Подтверждающая половина не
# формируется вовсе: ни одно её наблюдение не проходит через модель. Она
# расходуется один раз и только на ту архитектуру, которую выберет K-14g.
#
# ПОСЛЕДОВАТЕЛЬНО И НА ОДНОЙ КАРТЕ: тренер сверяет q0 с каноническим
# побитово, переносимость q0 между картами не измерена.
#
#   bash experiments/k14h_run_stage1.sh              # cuda:1, сид 0
#   bash experiments/k14h_run_stage1.sh cuda:0 1
#   SKIP_SMOKE=1 bash experiments/k14h_run_stage1.sh # только этап 1
set -u
DEV="${1:-cuda:1}"
SEED="${2:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p logs data/k14c reports/k14c
exec >> logs/k14h_stage1.log 2>&1

Q0="data/k14d/q0_b8_e0.npz"
GR="reports/k14d/gate_r.json"
GATE="reports/k14h/identity.json"
ORC="reports/k14a/oracle_canonical_$(echo "$DEV" | tr -d ':').json"
CACHE="data/k14b/q1_canonical"
ARCHS=(baseline reattn_state reattn_draft)
STAMP="$(date +%Y%m%dT%H%M%S)"

echo "=== СТАРТ $(date) === §49 этап 1, карта $DEV, сид $SEED"
echo "    коммит $(git rev-parse --short HEAD 2>/dev/null)"
[ -f "$GATE" ] || { echo "ОСТАНОВ: нет $GATE, архитектурный гейт не снят"; exit 2; }

# КАРТА ОБЯЗАНА СОВПАСТЬ С ТОЙ, НА КОТОРОЙ СНЯТ ГЕЙТ. Тождественность
# проверена побитово на конкретном ускорителе, и переносимость между картами
# не измерена. Тренер откажется и сам, но лучше сказать это здесь, чем
# выяснять после загрузки модели.
GDEV="$(python -c "import json,sys;print(json.load(open('$GATE')).get('device',''))")"
if [ "$GDEV" != "$DEV" ]; then
  echo "ОСТАНОВ: гейт снят на $GDEV, запуск просится на $DEV"
  exit 2
fi

run_one() {   # $1 архитектура, $2 режим (smoke|sel)
  local ARCH="$1" KIND="$2" rc
  local LOG="logs/k14c_${KIND}_${ARCH}_s${SEED}.log"
  local EXTRA=() SUM="reports/k14c/${KIND}_${ARCH}_s${SEED}.json"
  if [ "$KIND" = "smoke" ]; then
    # СМОУК ПИШЕТСЯ ПОД ШТАМПОМ ВРЕМЕНИ. Голова из него непригодна, но путь
    # без штампа занял бы имя и сделал повторный запуск невозможным.
    # ДАМП СНИМАЕТСЯ И В СМОУКЕ. Иначе его код первый раз исполнился бы на
    # исходе седьмого часа первого длинного прогона — то есть проверялся бы
    # тем самым прогоном, который должен защищать.
    EXTRA=(--smoke --limit 3
           --out "data/k14c/smoke_${ARCH}_${STAMP}.pt"
           --dump-rows "data/k14c/rows_smoke_${ARCH}_${STAMP}.npz")
    SUM="reports/k14c/smoke_${ARCH}_${STAMP}.json"
  else
    # ПОСТРОЧНЫЙ ДАМП ОБЯЗАТЕЛЕН. Сводка хранит только агрегат val_sel, а
    # парный кластерный бутстрап по эпизодам складывает слагаемые. Без
    # дампа три головы были бы несравнимы правилом §49, и это выяснилось бы
    # через двадцать один час.
    EXTRA=(--selection-only
           --dump-rows "data/k14c/rows_sel_${ARCH}_s${SEED}.npz")
  fi
  echo "--- $KIND $ARCH сид $SEED $(date) ---"
  python experiments/k14c_train_q1.py \
    --variant main --architecture "$ARCH" --additive-feedback on \
    --identity-gate "$GATE" --seed "$SEED" --device "$DEV" \
    --epochs 4 --batch 8 \
    --q1-cache "$CACHE" --q0 "$Q0" --gate-r "$GR" --oracle "$ORC" \
    --summary "$SUM" "${EXTRA[@]}" > "$LOG" 2>&1
  rc=$?
  echo "    код $rc $(date)"
  if [ $rc -ne 0 ]; then
    echo "ОСТАНОВ: $KIND $ARCH завершился кодом $rc"
    tail -20 "$LOG"
    return $rc
  fi
  grep -E "архитектура|гейт тождественности|q0 совпал|выбрана эпоха|РЕЖИМ|построчная|сохранено" "$LOG"
  return 0
}

# СМОУК ПЕРЕД СЕМЬЮ ЧАСАМИ. Три коротких прогона по три батча на часть ловят
# несвязность — белый список, оптимизатор, формы — за минуты вместо того,
# чтобы обнаружить её на исходе первого длинного прогона.
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  for ARCH in "${ARCHS[@]}"; do
    run_one "$ARCH" smoke || exit $?
  done
  echo "=== смоук пройден на всех трёх архитектурах $(date) ==="
fi

# ОТРИЦАТЕЛЬНЫЙ РЕЗУЛЬТАТ ОДНОЙ АРХИТЕКТУРЫ НЕ ОСТАНАВЛИВАЕТ ОСТАЛЬНЫЕ:
# в режиме отбора Gate 4 не вычисляется вовсе, и ненулевой код здесь всегда
# означает технический отказ, а не научный исход.
for ARCH in "${ARCHS[@]}"; do
  run_one "$ARCH" sel || exit $?
done

echo "=== КОНЕЦ $(date) ==="
echo "Дальше: сравнение K-14g по правилу §49 (нижняя граница 90% парного"
echo "интервала против baseline строго выше нуля, иначе никого не выбираем)."
