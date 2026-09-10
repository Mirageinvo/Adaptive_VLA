#!/usr/bin/env bash
# K-11g: окно исследования HiCoRA-G. Сто сорок ячеек, 700 эпизодов.
#
# ЗАЧЕМ РАННЕР. Правило «минимальная годная sigma на обеих головах» само по
# себе исполняемым не является. Здесь оно читается агрегатором, который
# возвращает ненулевой код, если окна нет.
#
# ОДИН ПРОЦЕСС НА ЯЧЕЙКУ. Сто сорок загрузок модели дороже одного прогона,
# зато между руками не переносится ничего: ни состояние ГСЧ, ни log_std, ни
# состояние среды, ни контекст CUDA. Для одноразового гейта это верный обмен,
# и ячейки становятся атомарными и возобновляемыми.
#
# ПИЛОТ И ЗАРЕГИСТРИРОВАННЫЙ ПРОГОН РАЗДЕЛЕНЫ. Smoke-эпизоды из
# data/k11g_smoke* в этот гейт НЕ входят: по ним настраивался критерий, и
# переиспользовать их значило бы проверять правило на данных, его
# породивших. Поэтому здесь другой run_tag, ДРУГИЕ init_state_id, другая
# соль шума и чистый каталог ячеек.
#
# ПРЕ-РЕГИСТРАЦИЯ. Годной считается одна и та же МИНИМАЛЬНАЯ sigma, у которой
# на ОБЕИХ головах: rms_median >= 0.01, насыщение < 10%, падение успеха от
# СВОЕЙ детерминированной руки <= 10 пп, инварианты и паритет пройдены.
# Дискордантность исходов считается обязательно, но в критерий НЕ входит.
#
# ЭТО ИНЖЕНЕРНЫЙ ФИЛЬТР ДЛЯ ВЫБОРА НАЧАЛЬНОЙ sigma PPO, а не доказательство
# статистической не-худшести: 50 эпизодов на ячейку интервалов не дают.
#
# НЕ ЗАДАВАТЬ CUDA_VISIBLE_DEVICES: robosuite выводит из неё
# MUJOCO_EGL_DEVICE_ID, и EGL падает на каждой ячейке.
set -u -o pipefail

CKPT="${CKPT:-ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO}"
JOINT="${JOINT:-data/k9d_ep3.pt}"
H_S0="${H_S0:-data/k11d/d1_mlp_coef_0.001_wd0_s0.pt}"
H_S1="${H_S1:-data/k11d/d1_mlp_coef_0.001_wd0_s1.pt}"
ROOT="${ROOT:-data/k11g}"
DEV="${DEV:-cuda:0}"

# --- ПАРАМЕТРЫ ПРОТОКОЛА ЗАКРЕПЛЕНЫ, А НЕ НАСТРАИВАЮТСЯ ---------------------
TAG="k11g"
SUITE="10"
TASKS="0 1 2 3 4 5 6 7 8 9"
SIGMAS="0 0.03 0.05 0.07 0.10 0.30 0.50"
NENV=5
INIT=40          # СВЕЖИЕ состояния: пилот шёл на 0..4
EPS_SALT=1       # ДРУГОЕ пространство шума, чем у пилота
HORIZON=8
MAXSTEPS=600
WAITSTEPS=10
SEED=0
SEEDMODE="block"

CELLS="$ROOT/cells"
ANA="$ROOT/analysis"
PROTO="$ROOT/protocol.json"
mkdir -p "$CELLS" "$ANA" logs

for f in "$JOINT" "$H_S0" "$H_S1" data/pos_offset_table.json; do
  [ -s "$f" ] || { echo "нет файла $f"; exit 1; }
done

sha12() { python3 -c "
import hashlib,sys
h=hashlib.sha1()
with open(sys.argv[1],'rb') as f:
    for c in iter(lambda: f.read(1<<22), b''): h.update(c)
print(h.hexdigest()[:12])" "$1"; }

# Отпечатки, которые войдут в протокол и сверяются в каждой ячейке.
J_SHA=$(sha12 "$JOINT"); S0_SHA=$(sha12 "$H_S0"); S1_SHA=$(sha12 "$H_S1")
CELL_SHA=$(sha12 experiments/k11g_cell.py)
HG_SHA=$(sha12 experiments/hicora_g.py)
HV_SHA=$(sha12 experiments/hicora_vla.py)
JV_SHA=$(sha12 experiments/joint12_vla.py)
OT_SHA=$(sha12 data/pos_offset_table.json)
# res_norm, basis и rho берутся из чекпойнта головы: воркер их и так сверяет,
# но в протоколе они нужны, чтобы подмена не прошла через пропуск ячейки.
read -r RN_SHA BS_SHA RH_SHA <<<"$(python3 - "$H_S0" <<'PY'
import torch, sys
o = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(o["res_norm_sha1"], o["basis_sha1"], o["rho_sha1"])
PY
)"
echo "  воркер $CELL_SHA, hicora_g $HG_SHA, hicora_vla $HV_SHA"
echo "  черновик $J_SHA, головы $S0_SHA / $S1_SHA, res_norm $RN_SHA"

SIG_JSON=$(python3 -c "
import sys; print('[' + ','.join(str(float(x)) for x in sys.argv[1].split()) + ']')" "$SIGMAS")
TASK_JSON=$(python3 -c "
import sys; print('[' + ','.join(sys.argv[1].split()) + ']')" "$TASKS")
CFG=$(printf '{"run_tag":"%s","ckpt":"%s","suite":"%s","tasks":%s,"sigmas":%s,"n_envs":%d,"init_start":%d,"eps_salt":%d,"horizon":%d,"max_steps":%d,"waiting_steps":%d,"ensemble":"off","seed":%d,"rollout_seed_mode":"%s","preprocess":"CenterCrop(196)->Resize(224)","image_size":224,"dtype":"float16","joint_sha1":"%s","head_s0_sha1":"%s","head_s1_sha1":"%s","res_norm_sha1":"%s","basis_sha1":"%s","rho_sha1":"%s","offset_table_sha1":"%s","cell_script_sha1":"%s","hicora_g_sha1":"%s","hicora_vla_sha1":"%s","joint12_vla_sha1":"%s","min_rms":0.01}' \
  "$TAG" "$CKPT" "$SUITE" "$TASK_JSON" "$SIG_JSON" "$NENV" "$INIT" \
  "$EPS_SALT" "$HORIZON" "$MAXSTEPS" "$WAITSTEPS" "$SEED" "$SEEDMODE" \
  "$J_SHA" "$S0_SHA" "$S1_SHA" "$RN_SHA" "$BS_SHA" "$RH_SHA" "$OT_SHA" \
  "$CELL_SHA" "$HG_SHA" "$HV_SHA" "$JV_SHA")

python3 experiments/k11g_protocol.py init --proto "$PROTO" --cells "$CELLS" \
  --cfg "$CFG" || exit 1

verify() {  # file head sigma task
  python3 experiments/k11g_protocol.py check --proto "$PROTO" \
    --cell "$1" --head "$2" --sigma "$3" --task "$4"
}

cell() {  # head sigma task
  local hd="$1" sg="$2" t="$3"
  local f
  f="$CELLS/$(python3 -c "
import sys; sys.path.insert(0,'experiments')
import k11g_protocol as kp
print(kp.cell_name(sys.argv[1], sys.argv[2], sys.argv[3]))" "$t" "$hd" "$sg")"
  # ПРОПУСК ТОЛЬКО ПОСЛЕ СВЕРКИ С ПРОТОКОЛОМ. Условие «файл непустой»
  # переиспользовало бы ячейку от другой головы, другой соли или другой
  # версии воркера молча.
  if [ -s "$f" ]; then
    verify "$f" "$hd" "$sg" "$t" >/dev/null 2>&1 \
      && { echo "  пропуск (готово и сверено): $(basename "$f")"; return 0; }
    echo "  ГОТОВАЯ ЯЧЕЙКА НЕ ПРОШЛА СВЕРКУ: $f"
    verify "$f" "$hd" "$sg" "$t"
    return 1
  fi
  echo "  ячейка: голова $hd, sigma $sg, задача $t"
  PYTHONUNBUFFERED=1 PYTHONPATH="$HOME/LIBERO" MUJOCO_GL=egl \
  python -u experiments/k11g_cell.py \
    --ckpt "$CKPT" --policy-ckpt "$JOINT" \
    --hicora-s0 "$H_S0" --hicora-s1 "$H_S1" \
    --head "$hd" --sigma "$sg" --task-id "$t" \
    --task-suite "$SUITE" --n-envs "$NENV" --init-start "$INIT" \
    --horizon "$HORIZON" --max-steps "$MAXSTEPS" \
    --waiting-steps "$WAITSTEPS" --seed "$SEED" \
    --rollout-seed-mode "$SEEDMODE" --eps-salt "$EPS_SALT" \
    --run-tag "$TAG" --device "$DEV" --out "$f" || return 1
  # И ТОЛЬКО ЧТО ПОСЧИТАННАЯ СВЕРЯЕТСЯ ТОЖЕ: флаги могли разойтись с
  # протоколом, и заметить это лучше сразу, а не через пять часов.
  verify "$f" "$hd" "$sg" "$t" || return 1
}

N=0
for t in $TASKS; do
  for hd in s0 s1; do
    for sg in $SIGMAS; do
      cell "$hd" "$sg" "$t" || exit 1
      N=$((N + 1))
    done
  done
done
echo "  ячеек обработано: $N"

python3 experiments/k11g_protocol.py analyze --proto "$PROTO" \
  --cells "$CELLS" --out "$ANA/window.json"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo
  echo "  K-11g НЕ ПРОЙДЕН: годной sigma нет."
  echo "  Это означает, что в сетке нет режима, который меняет поведение"
  echo "  существенно и при этом безопасен на обеих головах. Следующий шаг —"
  echo "  не подгонять порог, а пересмотреть конструкцию исследования:"
  echo "  state-dependent sigma, иной ранг или иной базис поправки."
  exit "$rc"
fi
echo
echo "  K-11g ПРОЙДЕН. Дальше: PPO smoke-test на сохранённом батче (KL,"
echo "  clip fraction, переполнения), и только затем LIBERO PPO. Итоговая"
echo "  оценка RL — на НОВЫХ начальных состояниях, не на этих."
