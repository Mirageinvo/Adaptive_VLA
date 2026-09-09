#!/usr/bin/env bash
# K-11e: симуляторный гейт HiCoRA. Пять физических политик, 400 парных
# эпизодов, десять задач, общие начальные состояния.
#
# ЗАЧЕМ РАННЕР, А НЕ РУКИ. Правило «обязано пройти на обоих сидах» само по
# себе исполняемым не является: `k6h_summarize.py` возвращает код 0 и при
# непрохождении критерия. Здесь четыре анализа читаются из JSON, и отказ
# наступает, если хотя бы одна нижняя односторонняя граница не выше нуля.
#
# ПРЕ-РЕГИСТРАЦИЯ, ЗАПИСАННАЯ ДО ЗАПУСКА.
#   Критерий 1: hicora_s0 против joint12  — ПРЕВОСХОДСТВО, граница > 0.
#   Критерий 2: hicora_s0 против coarse24 — ПРЕВОСХОДСТВО, граница > 0.
#   Подтверждение: оба обязаны выполниться и на hicora_s1.
#   Ровно 400 пар и 10 задач; добора после просмотра интервала не будет.
#
# ЧЕГО ЭТОТ СТЕНД НЕ МЕРЯЕТ. Равную стоимость. Рука `coarse24` идёт через
# `generate()` по всем трём блокам RVQ, уровни 1-2 затем выбрасываются:
# успех сравнивается корректно, вычисления — нет. Это предмет K-11h.
#
# НЕ ЗАДАВАТЬ CUDA_VISIBLE_DEVICES: robosuite выводит из неё
# MUJOCO_EGL_DEVICE_ID, и EGL падает на каждой ячейке.
set -u -o pipefail

CKPT="${CKPT:-ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO}"
JOINT="${JOINT:-data/k9d_ep3.pt}"
H_S0="${H_S0:-data/k11d/d1_mlp_coef_0.001_wd0_s0.pt}"
H_S1="${H_S1:-data/k11d/d1_mlp_coef_0.001_wd0_s1.pt}"
OUT="${OUT:-data/k11e}"
TAG="${TAG:-k11e}"
DEV="${DEV:-cuda:0}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
BLOCKS="${BLOCKS:-0 10 20 30}"   # по 10 сред на блок => 40 эпизодов на задачу
NENV="${NENV:-10}"
ENS="${ENS:-on}"
PAIRS="${PAIRS:-400}"
NTASKS="${NTASKS:-10}"

mkdir -p "$OUT" logs

# --- ЯЧЕЙКИ -----------------------------------------------------------------
# Готовые пропускаются: прогон длинный, и падение одной ячейки не должно
# стоить всего остального. Так же устроен run_k9d_gate.sh.
cell() {  # policy arm_label [extra...]
  local pol="$1" arm="$2"; shift 2
  local f="$OUT/t${T}_i${I}_${arm}.json"
  if [ -s "$f" ]; then echo "  пропуск (готово): $f"; return 0; fi
  echo "  ячейка: задача $T, блок $I, рука $arm"
  PYTHONPATH="$HOME/LIBERO" MUJOCO_GL=egl \
  python experiments/k9h_multiarm_gate.py \
    --ckpt "$CKPT" --policy "$pol" --arm-label "$arm" --run-tag "$TAG" \
    --task-id "$T" --init-start "$I" --n-envs "$NENV" --ensemble "$ENS" \
    --device "$DEV" --out "$f" "$@" || return 1
}

for T in $TASKS; do
  for I in $BLOCKS; do
    cell fullbar  fullbar                                          || exit 1
    cell coarse24 coarse24                                         || exit 1
    cell fast     joint12   --policy-ckpt "$JOINT" --expect-depth 12 || exit 1
    cell hicora   hicora_s0 --policy-ckpt "$JOINT" --expect-depth 12 \
         --hicora-ckpt "$H_S0"                                     || exit 1
    cell hicora   hicora_s1 --policy-ckpt "$JOINT" --expect-depth 12 \
         --hicora-ckpt "$H_S1"                                     || exit 1
  done
done

# --- АНАЛИЗ -----------------------------------------------------------------
summ() {  # test ref out
  python experiments/k6h_summarize.py \
    --glob "$OUT/*.json" --field arm_label --test "$1" --ref "$2" \
    --hypothesis superiority --margin 0 \
    --expect-pairs "$PAIRS" --expect-tasks "$NTASKS" \
    --require-full-hash --allow-extra-arms \
    --out "$3" || return 1
}

summ hicora_s0 joint12  "$OUT/an_s0_vs_joint12.json"  || exit 1
summ hicora_s0 coarse24 "$OUT/an_s0_vs_coarse24.json" || exit 1
summ hicora_s1 joint12  "$OUT/an_s1_vs_joint12.json"  || exit 1
summ hicora_s1 coarse24 "$OUT/an_s1_vs_coarse24.json" || exit 1

# --- ОБЩИЙ ВЕРДИКТ, ИСПОЛНЯЕМЫЙ --------------------------------------------
python - "$OUT" <<'PY'
import json, os, sys
d = sys.argv[1]
files = {"s0 против joint12": "an_s0_vs_joint12.json",
         "s0 против coarse24": "an_s0_vs_coarse24.json",
         "s1 против joint12": "an_s1_vs_joint12.json",
         "s1 против coarse24": "an_s1_vs_coarse24.json"}
bad, rows = [], []
for name, f in files.items():
    p = os.path.join(d, f)
    if not os.path.exists(p):
        bad.append(f"{name}: нет {f}")
        continue
    cells = json.load(open(p))["cells"]
    if not cells:
        bad.append(f"{name}: пустой отчёт")
        continue
    for tag, c in sorted(cells.items()):
        lo = c.get("lower_1s")
        ok = c.get("superior") is True and lo is not None and lo > 0
        rows.append((name, tag, c.get("micro"), lo, ok))
        if not ok:
            bad.append(f"{name} [{tag}]: нижняя граница {lo}, превосходство "
                       f"НЕ доказано")
print("\n  ИТОГ K-11e")
print(f"    {'сравнение':>22}{'ячейка':>10}{'разница':>10}{'граница':>10}"
      f"{'вердикт':>12}")
for name, tag, mic, lo, ok in rows:
    print(f"    {name:>22}{tag:>10}"
          f"{('—' if mic is None else f'{mic:+.1f}'):>10}"
          f"{('—' if lo is None else f'{lo:+.1f}'):>10}"
          f"{('превзошла' if ok else 'НЕТ'):>12}")
if bad:
    print("\n  K-11e НЕ ПРОЙДЕН:")
    for b in bad:
        print(f"    {b}")
    print("    Правило зарегистрировано до запуска: превосходство обязано "
          "выполниться\n    по ОБОИМ сравнениям и на ОБОИХ сидах.")
    sys.exit(1)
print("\n  K-11e ПРОЙДЕН: превосходство доказано по обоим сравнениям на "
      "обоих сидах.")
print("    ОГОВОРКА: равная стоимость здесь НЕ измерена — coarse24 идёт "
      "через generate()\n    по всем трём блокам. Это K-11h.")
PY
