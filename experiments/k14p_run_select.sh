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
# СОСТОЯНИЯ 0-9 РАСХОДУЮТСЯ. Банки: dev 10-39, final 40-49.
#
#   bash experiments/k14p_run_select.sh            # cuda:1, сид отбора 100
#   bash experiments/k14p_run_select.sh cuda:0 100
set -u
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

ARTS=()
for T in 0 1 2 3 4 5 6 7 8 9; do
  for I0 in 0 5; do
    OUT="reports/k14p/sel_t${T}_i${I0}.json"
    if [ -f "$OUT" ]; then
      echo "    уже есть $OUT, пропускаю"
      ARTS+=("$OUT"); continue
    fi
    echo "--- задача $T, состояния $I0-$((I0+4)) $(date), свободно $(free -g | awk 'NR==2{print $7}') ГБ"
    python experiments/k9h_multiarm_gate.py \
      --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
      --task-suite 10 --task-id "$T" --init-start "$I0" --n-envs 5 \
      --seed "$SEL_SEED" --rollout-seed-mode fixed \
      --ensemble off --horizon 8 --max-steps 600 \
      --run-tag k14p_select --device "$DEV" \
      --policy fast --policy-ckpt data/k9d_ep3.pt \
      --arm-label fast_s0 --out "$OUT"
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "ОСТАНОВ: задача $T, блок $I0 завершился кодом $rc"
      exit $rc
    fi
    ARTS+=("$OUT")
    sleep 15
  done
done

echo "=== раскатки закончены $(date), артефактов ${#ARTS[@]} ==="
python experiments/k14p_task_select.py --runs "${ARTS[@]}" \
  --expect-policy fast --delta 0.05 --out reports/k14p/banks.json
echo "=== КОНЕЦ $(date) ==="
