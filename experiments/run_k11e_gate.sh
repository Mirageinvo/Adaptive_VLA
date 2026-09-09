#!/usr/bin/env bash
# K-11e: симуляторный гейт HiCoRA. Пять физических политик, 400 парных
# эпизодов, десять задач, общие начальные состояния.
#
# ЗАЧЕМ РАННЕР, А НЕ РУКИ. Правило «обязано пройти на обоих сидах» само по
# себе исполняемым не является: `k6h_summarize.py` возвращает код 0 и при
# непрохождении критерия. Здесь четыре анализа читаются из JSON, и отказ
# наступает, если хотя бы одна нижняя односторонняя граница не выше нуля.
#
# КАТАЛОГИ РАЗДЕЛЕНЫ. Ячейки лежат в cells/, отчёты анализа в analysis/.
# Прежде и то и другое писалось в один каталог, а анализ читал `*.json`:
# второй анализ увидел бы отчёт первого как ячейку и упал на отсутствии
# `arm_label`, а при повторном запуске упал бы уже первый.
#
# ПРОТОКОЛ ФИКСИРУЕТСЯ ДО ЗАПУСКА в protocol.json и сверяется при
# возобновлении. Иначе «ровно 400 и 10 задач» означало бы лишь значения по
# умолчанию, а продолжение прогона другим чекпойнтом прошло бы незаметно.
#
# ПРЕ-РЕГИСТРАЦИЯ.
#   Критерий 1: hicora_s0 против joint12  — ПРЕВОСХОДСТВО, граница > 0.
#   Критерий 2: hicora_s0 против coarse24 — ПРЕВОСХОДСТВО, граница > 0.
#   Подтверждение: оба обязаны выполниться и на hicora_s1.
#   Ровно 400 пар и 10 задач; добора после просмотра интервала не будет.
#
# ЧЕГО ЭТОТ СТЕНД НЕ МЕРЯЕТ. Равную стоимость. Рука `coarse24` идёт через
# `generate()` по всем трём блокам RVQ, уровни 1-2 затем выбрасываются:
# успех сравнивается корректно, вычисления — нет. Это предмет K-11h.
# Поэтому непрохождение критерия 2 НЕ доказывает бесполезность HiCoRA: оно
# говорит о достаточности текущего D1, а решение по Парето — за K-11h.
#
# НЕ ЗАДАВАТЬ CUDA_VISIBLE_DEVICES: robosuite выводит из неё
# MUJOCO_EGL_DEVICE_ID, и EGL падает на каждой ячейке.
set -u -o pipefail

CKPT="${CKPT:-ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO}"
JOINT="${JOINT:-data/k9d_ep3.pt}"
H_S0="${H_S0:-data/k11d/d1_mlp_coef_0.001_wd0_s0.pt}"
H_S1="${H_S1:-data/k11d/d1_mlp_coef_0.001_wd0_s1.pt}"
ROOT="${ROOT:-data/k11e}"
TAG="${TAG:-k11e}"
DEV="${DEV:-cuda:0}"

# --- ПАРАМЕТРЫ ПРОТОКОЛА ЗАКРЕПЛЕНЫ, А НЕ НАСТРАИВАЮТСЯ ---------------------
# Пути и карта настраиваются, потому что от них не зависит зарегистрированное
# правило. Число пар, задач, сред и режим ансамбля — зависят.
TASKS="0 1 2 3 4 5 6 7 8 9"
BLOCKS="0 10 20 30"      # по 10 сред на блок => 40 эпизодов на задачу
NENV=10
ENS="on"
PAIRS=400
NTASKS=10

CELLS="$ROOT/cells"
ANA="$ROOT/analysis"
PROTO="$ROOT/protocol.json"
mkdir -p "$CELLS" "$ANA" logs

for f in "$JOINT" "$H_S0" "$H_S1"; do
  [ -s "$f" ] || { echo "нет файла $f"; exit 1; }
done

# --- ПАРАМЕТРЫ ИСПОЛНЕНИЯ ЗАДАЮТСЯ ЯВНО, а не наследуются из умолчаний
# K-9h: иначе смена умолчания молча изменила бы условия посреди прогона.
# Значения те же, что в K-6h/K-9d/K-9h, чтобы опорные числа оставались
# сопоставимыми. Менять их без причины нельзя: 520 вместо 600 сделало бы
# руки несравнимыми с прежними прогонами.
HORIZON=8
MAXSTEPS=600
WAITSTEPS=10
SEEDMODE="block"
SUITE="10"        # то же значение, что в K-6h/K-9d/K-9h

CFG=$(printf '{"run_tag":"%s","pairs":%d,"tasks":%d,"n_envs":%d,"ensemble":"%s","horizon":%d,"max_steps":%d,"waiting_steps":%d,"rollout_seed_mode":"%s","device":"%s","suite":"%s","task_ids":"%s","blocks":"%s"}' \
  "$TAG" "$PAIRS" "$NTASKS" "$NENV" "$ENS" "$HORIZON" "$MAXSTEPS" \
  "$WAITSTEPS" "$SEEDMODE" "$DEV" "$SUITE" "$TASKS" "$BLOCKS")

# --- ПРОТОКОЛ: ЗАПИСЬ ЛИБО СВЕРКА ------------------------------------------
python experiments/k11e_protocol.py init \
  --proto "$PROTO" --ckpt "$CKPT" --joint "$JOINT" \
  --h0 "$H_S0" --h1 "$H_S1" --cells "$CELLS" --cfg "$CFG" || exit 1

K9H_SHA=$(python -c "
import hashlib,sys
h=hashlib.sha1(open('experiments/k9h_multiarm_gate.py','rb').read())
print(h.hexdigest()[:12])")
echo "  версия стенда k9h $K9H_SHA"

# --- ЯЧЕЙКИ -----------------------------------------------------------------
# Готовые пропускаются: прогон длинный, и падение одной ячейки не должно
# стоить всего остального. Так же устроен run_k9d_gate.sh.
verify() {  # file arm
  python experiments/k11e_protocol.py check --proto "$PROTO" \
    --cell "$1" --arm "$2" --task "$T" --block "$I" --script-sha "$K9H_SHA"
}

cell() {  # policy arm_label [extra...]
  local pol="$1" arm="$2"; shift 2
  local f="$CELLS/t${T}_i${I}_${arm}.json"
  # ПРОПУСК ТОЛЬКО ПОСЛЕ СВЕРКИ С ПРОТОКОЛОМ. Прежде условием было «файл
  # непустой», и ячейки от ДРУГОЙ головы или другого Joint12 молча
  # переиспользовались бы, а вердикт оказался бы не для зарегистрированной
  # конфигурации.
  if [ -s "$f" ]; then
    verify "$f" "$arm" || { echo "  ГОТОВАЯ ЯЧЕЙКА НЕ ПРОШЛА СВЕРКУ: $f"; \
                            return 1; }
    echo "  пропуск (готово и сверено): $f"
    return 0
  fi
  echo "  ячейка: задача $T, блок $I, рука $arm"
  PYTHONPATH="$HOME/LIBERO" MUJOCO_GL=egl \
  python experiments/k9h_multiarm_gate.py \
    --ckpt "$CKPT" --policy "$pol" --arm-label "$arm" --run-tag "$TAG" \
    --task-id "$T" --init-start "$I" --n-envs "$NENV" --ensemble "$ENS" \
    --horizon "$HORIZON" --max-steps "$MAXSTEPS" \
    --waiting-steps "$WAITSTEPS" --rollout-seed-mode "$SEEDMODE" \
    --task-suite "$SUITE" --device "$DEV" --out "$f" "$@" || return 1
  # И ТОЛЬКО ЧТО ПОСЧИТАННАЯ СВЕРЯЕТСЯ ТОЖЕ: флаги могли разойтись с
  # протоколом, и заметить это лучше сразу, а не через сутки на анализе.
  verify "$f" "$arm" || return 1
}

for T in $TASKS; do
  for I in $BLOCKS; do
    cell fullbar  fullbar                                            || exit 1
    cell coarse24 coarse24                                           || exit 1
    cell fast     joint12   --policy-ckpt "$JOINT" --expect-depth 12 || exit 1
    cell hicora   hicora_s0 --policy-ckpt "$JOINT" --expect-depth 12 \
         --hicora-ckpt "$H_S0"                                       || exit 1
    cell hicora   hicora_s1 --policy-ckpt "$JOINT" --expect-depth 12 \
         --hicora-ckpt "$H_S1"                                       || exit 1
  done
done

# --- АНАЛИЗ -----------------------------------------------------------------
# Читаются ТОЛЬКО ячейки: отчёты анализа лежат в другом каталоге и под glob
# не попадают ни при первом запуске, ни при повторном.
summ() {  # test ref out
  python experiments/k6h_summarize.py \
    --glob "$CELLS/t*_i*_*.json" --field arm_label --test "$1" --ref "$2" \
    --hypothesis superiority --margin 0 \
    --expect-pairs "$PAIRS" --expect-tasks "$NTASKS" \
    --require-full-hash --allow-extra-arms \
    --out "$3" || return 1
}

summ hicora_s0 joint12  "$ANA/an_s0_vs_joint12.json"  || exit 1
summ hicora_s0 coarse24 "$ANA/an_s0_vs_coarse24.json" || exit 1
summ hicora_s1 joint12  "$ANA/an_s1_vs_joint12.json"  || exit 1
summ hicora_s1 coarse24 "$ANA/an_s1_vs_coarse24.json" || exit 1

# --- ОБЩИЙ ВЕРДИКТ, ИСПОЛНЯЕМЫЙ --------------------------------------------
python - "$ANA" <<'PY'
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
    print("    ОГОВОРКА: непрохождение против coarse24 НЕ доказывает "
          "бесполезность HiCoRA —\n    стоимость здесь не измерена, это K-11h.")
    sys.exit(1)
print("\n  K-11e ПРОЙДЕН: превосходство доказано по обоим сравнениям на "
      "обоих сидах.")
print("    ОГОВОРКА: равная стоимость здесь НЕ измерена — coarse24 идёт "
      "через generate()\n    по всем трём блокам. Это K-11h.")
PY
