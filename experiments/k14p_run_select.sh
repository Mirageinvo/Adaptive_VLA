#!/bin/bash
# §53.4: отборочный проход по десяти задачам LIBERO-10, рука q0.
#
# ЗАЧЕМ. Смоук на задаче 3 дал 100% успеха у всех рук. По пределу §37
# задачи без провалов разбавляют p_fail и делают невыполнимым весь набор,
# поэтому вторичный набор §53 отбирается ЗАРАНЕЕ и механически: задача
# включается, если провалов не менее 2 из 10 (правило §53.4, применяет k14p).
#
# БЛОК 5 СРЕД — ОГРАНИЧЕНИЕ ПАМЯТИ, А НЕ ВЫБОР. Десять сред вместе с
# depth-RVQ убиты OOM-killer'ом: на хосте 62 ГБ, 37 заняты вне контейнера,
# swap выбран целиком. Поэтому каждая задача покрывается двумя вызовами,
# состояния 0-4 и 5-9.
#
# --rollout-seed-mode fixed ОБЯЗАТЕЛЕН: при block сид раскатки зависит от
# init_start, и один и тот же init_state_id получил бы разные сиды в двух
# блоках одной задачи.
#
# СОСТОЯНИЯ 0-9 РАСХОДУЮТСЯ. Банки (§53.5): dev 10-29, final 30-49,
# по 20 состояний на задачу.
#
#   bash experiments/k14p_run_select.sh            # cuda:1, сид отбора 100
#   bash experiments/k14p_run_select.sh cuda:0 100
# set -euo pipefail, А НЕ set -u. При одном -u падение финального
# k14p_task_select.py не влияло на код возврата: следующий echo возвращал
# ноль, и раннер сообщал об успехе после ошибки.
set -euo pipefail
DEV="${1:-cuda:1}"
SEL_SEED="${2:-100}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p logs reports/k14p
exec >> logs/k14p_select.log 2>&1

echo "=== СТАРТ $(date) === отбор задач, карта $DEV, сид $SEL_SEED"
echo "    коммит $(git rev-parse --short HEAD 2>/dev/null)"

# ПРЕД-ПРОХОД: СМЕСЬ ВЕРСИЙ ОБНАРУЖИВАЕТСЯ ДО ПЕРВОГО БЛОКА, А НЕ ПОСЛЕ
# ПОСЛЕДНЕГО. Сборка банков требует единой script_sha1 на весь отбор. Если
# часть артефактов снята прежней версией k9h, а один битый пришлось бы
# пересчитать текущей, набор оказался бы смесью двух версий — и был бы
# справедливо отвергнут через час работы.
CUR_SHA="$(sha1sum experiments/k9h_multiarm_gate.py | cut -c1-12)"
OLD_SHAS="$(grep -ho '"script_sha1": "[^"]*"' reports/k14p/sel_t*_i*.json \
            2>/dev/null | sed 's/.*: "//; s/"//' | sort -u | tr '\n' ' ' \
            | sed 's/ *$//')"
NEED_RECOMPUTE=0
for T in 0 1 2 3 4 5 6 7 8 9; do
  for I0 in 0 5; do
    F="reports/k14p/sel_t${T}_i${I0}.json"
    if [ ! -f "$F" ] || ! python experiments/k14p_task_select.py \
          --validate-one "$F" --expect-policy fast >/dev/null 2>&1; then
      NEED_RECOMPUTE=1
    fi
  done
done
echo "    текущая версия k9h $CUR_SHA; в готовых артефактах: ${OLD_SHAS:-нет}"
if [ "$NEED_RECOMPUTE" = "1" ] && [ -n "$OLD_SHAS" ] \
   && [ "$OLD_SHAS" != "$CUR_SHA" ]; then
  echo "ОСТАНОВ: часть блоков надо пересчитать текущей версией k9h ($CUR_SHA),"
  echo "  а готовые сняты версией(ями) $OLD_SHAS. Набор получился бы смесью"
  echo "  версий и был бы отвергнут при сборке банков. Выберите одно:"
  echo "    1) пересчитать ВСЕ 20 блоков текущей версией:"
  echo "       rm -f reports/k14p/sel_t*_i*.json  и повторить запуск"
  echo "    2) пересчитать битый блок на том коммите, которым сняты остальные"
  exit 4
fi

ARTS=()
for T in 0 1 2 3 4 5 6 7 8 9; do
  for I0 in 0 5; do
    OUT="reports/k14p/sel_t${T}_i${I0}.json"
    # ВОЗОБНОВЛЕНИЕ ПРОВЕРЯЕТ АРТЕФАКТ, А НЕ ФАКТ ЕГО СУЩЕСТВОВАНИЯ. Прежде
    # любой существующий файл пропускался: обрезанный от убитого процесса или
    # снятый с другой конфигурацией считался готовым. Проверка — тем же
    # кодом, который потом собирает банки.
    if [ -f "$OUT" ]; then
      if python experiments/k14p_task_select.py --validate-one "$OUT"            --expect-policy fast >/dev/null 2>&1; then
        echo "    уже есть и проверен $OUT, пропускаю"
        ARTS+=("$OUT"); continue
      fi
      echo "    $OUT существует, но проверку не прошёл — пересчитываю"
      python experiments/k14p_task_select.py --validate-one "$OUT"         --expect-policy fast 2>&1 | tail -3 || true
      rm -f "$OUT"
    fi
    echo "--- задача $T, состояния $I0-$((I0+4)) $(date), свободно $(free -g | awk 'NR==2{print $7}') ГБ"
    # rc снимается явно: при set -e непосредственный выход по ошибке не дал
    # бы напечатать, на каком блоке мы встали.
    rc=0
    python experiments/k9h_multiarm_gate.py \
      --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
      --task-suite 10 --task-id "$T" --init-start "$I0" --n-envs 5 \
      --seed "$SEL_SEED" --rollout-seed-mode fixed \
      --ensemble off --horizon 8 --max-steps 600 \
      --run-tag k14p_select --device "$DEV" \
      --policy fast --policy-ckpt data/k9d_ep3.pt \
      --arm-label fast_s0 --out "$OUT" || rc=$?
    if [ $rc -ne 0 ]; then
      echo "ОСТАНОВ: задача $T, блок $I0 завершился кодом $rc"
      exit $rc
    fi
    ARTS+=("$OUT")
    sleep 15
  done
done

echo "=== раскатки закончены $(date), артефактов ${#ARTS[@]} ==="
if [ "${#ARTS[@]}" -ne 20 ]; then
  echo "ОСТАНОВ: артефактов ${#ARTS[@]}, а отбор определён на 10 задач x 2 блока"
  exit 3
fi
python experiments/k14p_task_select.py --runs "${ARTS[@]}" \
  --expect-policy fast --delta 0.05 --out reports/k14p/banks.json
echo "=== КОНЕЦ $(date), банки собраны ==="
