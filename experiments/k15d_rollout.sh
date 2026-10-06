#!/bin/bash
# K-15d: проверка вывода и парные роллауты q0 и рук уточнения.
#
# РУКИ (метка -> чекпойнт, исполняемый уровень, слоёв на вызов):
#   q0    точный K-14 черновик (как в K-14q/K-15c);
#   d0    data/k15d/d0_s0.pt    a_d, 24 слоя;
#   h18   data/k15d/h1p1_s0.pt  a1,  проход ОСТАНАВЛИВАЕТСЯ на 18-м слое;
#   h1    data/k15d/h1p2_s0.pt  a2,  24 слоя.
# Руку можно исключить: SKIP="d0 h18". Состав передаётся анализу явно
# (--expected-arms) и проверяется точно.
#
# ИСКЛЮЧЕНИЕ ИЗ ФИЛЬТРА ДОПУСКА — только с текстовой причиной в переменной
# <РУКА>_OVERRIDE (D0_OVERRIDE, H18_OVERRIDE, H1_OVERRIDE). Причина уходит в
# отчёт проверки вывода, в метаданные руки и в каждый артефакт. Рука с
# исключением НЕ допускается в dev, пока не пройден safety с тем же
# чекпойнтом и той же причиной.
#
# РЕЖИМЫ:
#   infer   проверка вывода всех активных рук на карте роллаута;
#   smoke   задача 8, состояния 0-4, все руки и ПОВТОР каждой руки K-15d:
#           запуск, конечность, диапазон, запись уровней, детерминизм;
#   target  прицельный пилот: SUITE=<набор> TASKS="<номера>", состояния
#           0-24 — для задачи, где фильтр допуска нашёл риск;
#   safety  задачи 8-9 набора 10, 50 кластеров — технический пилот;
#   confirm ПОДТВЕРЖДЕНИЕ гипотезы «h18 лучше q0»: только руки q0 и h18,
#           задачи 0-7 набора 10, 200 НЕ виденных кластеров. Правило
#           зафиксировано в k15d_behavior.CONFIRM до роллаутов;
#   dev     задачи 0-9 набора 10, 250 кластеров (абляции и описание).
#
# Один execution seed (101), порядок рук циклически сдвигается по задаче.
# Каталог артефактов несёт отпечатки чекпойнтов, отчётов и харнесса.
# Роллауты идут одним процессом за раз: две MuJoCo-раскатки параллельно не
# помещаются в память хоста.
#
#   bash experiments/k15d_rollout.sh infer cuda:1
#   D0_OVERRIDE="..." SUITE=spatial TASKS="0" \
#       bash experiments/k15d_rollout.sh target cuda:1
set -euo pipefail
MODE="${1:?нужен режим: infer, smoke, target, safety, confirm или dev}"
DEV="${2:-cuda:1}"
SEED=101
SKIP="${SKIP:-}"
SUITE="${SUITE:-10}"
DTAG="$(echo "$DEV" | tr -d ':')"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${LIBERO_PATH:-$HOME/LIBERO}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p logs/k15d reports/k15d/rollout

ck_of () {
  case "$1" in
    d0)  echo "${D0_CK:-data/k15d/d0_s0.pt}" ;;
    h18) echo "${H18_CK:-data/k15d/h1p1_s0.pt}" ;;
    h1)  echo "${H1_CK:-data/k15d/h1p2_s0.pt}" ;;
  esac
}
override_of () {
  case "$1" in
    d0)  echo "${D0_OVERRIDE:-}" ;;
    h18) echo "${H18_OVERRIDE:-}" ;;
    h1)  echo "${H1_OVERRIDE:-}" ;;
  esac
}
report_of () {   # рука -> отчёт проверки вывода на этой карте
  local ck; ck="$(ck_of "$1")"
  echo "reports/k15d/inference_$(basename "${ck%.pt}")_${DTAG}.json"
}
want () { case " $SKIP " in *" $1 "*) return 1;; esac; return 0; }
K15D_ARMS=""
for A in d0 h18 h1; do want "$A" && K15D_ARMS="$K15D_ARMS $A"; done
K15D_ARMS="${K15D_ARMS# }"
# В confirm состав фиксирован правилом: только первичная пара.
[ "$MODE" = "confirm" ] && K15D_ARMS="h18"
[ -n "$K15D_ARMS" ] || { echo "ОТКАЗ: все руки K-15d исключены"; exit 1; }

if [ "$MODE" = "infer" ]; then
  RC=0
  for A in $K15D_ARMS; do
    OV="$(override_of "$A")"
    echo "=== проверка вывода $A: $(ck_of "$A") на $DEV" \
         "${OV:+(исключение: $OV)} — $(date '+%H:%M:%S')"
    python3 experiments/k15d_check_inference.py --checkpoint "$(ck_of "$A")" \
      --device "$DEV" --out "$(report_of "$A")" \
      ${OV:+--allow-failed-admission "$OV"} || RC=$?
  done
  exit $RC
fi

case "$MODE" in
  smoke)  SUITE=10; TASKS="8"; STATES="0" ;;
  safety) SUITE=10; TASKS="8 9"; STATES="0 5 10 15 20" ;;
  confirm) SUITE=10; TASKS="0 1 2 3 4 5 6 7"; STATES="0 5 10 15 20" ;;
  dev)    SUITE=10; TASKS="0 1 2 3 4 5 6 7 8 9"; STATES="0 5 10 15 20" ;;
  target) TASKS="${TASKS:?для target нужны TASKS и SUITE}"
          STATES="0 5 10 15 20" ;;
  *) echo "режим $MODE неизвестен"; exit 2 ;;
esac
for A in $K15D_ARMS; do
  for f in "$(ck_of "$A")" "$(report_of "$A")"; do
    [ -f "$f" ] || { echo "ОТКАЗ: нет $f (сначала режим infer)"; exit 1; }
  done
done
ARMS="q0 $K15D_ARMS"
EXPECTED="$(echo "$ARMS" | tr ' ' ',')"
for m in k15d_depth_refine k15d_policy k15d_check_inference k15d_behavior \
         k9h_multiarm_gate; do
  python3 "experiments/${m}.py" --selftest >/dev/null \
    || { echo "ОТКАЗ: самопроверка ${m}"; exit 1; }
done
sha12 () { sha1sum "$1" | cut -c1-12; }
TAG="k9h$(sha12 experiments/k9h_multiarm_gate.py)"
for A in $K15D_ARMS; do
  TAG="${TAG}_${A}$(sha12 "$(ck_of "$A")")$(sha12 "$(report_of "$A")")"
done

if [ "$MODE" = "dev" ] || [ "$MODE" = "confirm" ]; then
  # РУКА С ИСКЛЮЧЕНИЕМ ИДЁТ ДАЛЬШЕ ТОЛЬКО ПОСЛЕ ПРОЙДЕННОГО safety С ТЕМ ЖЕ
  # ИСПОЛНЕНИЕМ: чекпойнт, фаза и число слоёв (из спецификации метки),
  # версия харнесса, модули уточнения и руки — текущие, и та же причина.
  # Safety, снятый другим кодом, dev не открывает.
  for A in $K15D_ARMS; do
    OV="$(override_of "$A")"
    [ -n "$OV" ] || continue
    if ! python3 - "$A" "$(sha12 "$(ck_of "$A")")" "$OV" \
         "$(sha12 experiments/k9h_multiarm_gate.py)" \
         "$(sha12 experiments/k15d_depth_refine.py)" \
         "$(sha12 experiments/k15d_policy.py)" <<'PY'
import glob, json, os, sys
sys.path.insert(0, "experiments")
from k15d_behavior import ARM_SPEC
arm, cks, reason, harness, refine, policy = sys.argv[1:7]
spec = ARM_SPEC[arm]
want = dict(checkpoint_sha1=cks, phase=spec["phase"],
            layers_per_call=spec["layers_per_call"], harness_sha1=harness,
            refine_module_sha1=refine, policy_module_sha1=policy)
ok = False
for f in glob.glob("reports/k15d/rollout/behavior_safety_s101_*.json"):
    d = json.load(open(f))
    if d.get("verdict", {}).get("code") != 0 or d.get("partial"):
        continue
    for pv in (d.get("provenance") or {}).get(arm, []):
        same = all(str(pv.get(k)) == str(v) for k, v in want.items())
        ov = pv.get("admission_override") or {}
        if same and ov.get("reason") == reason:
            ok = True
            print(f"    допуск {arm}: safety {os.path.basename(f)}")
sys.exit(0 if ok else 1)
PY
    then
      echo "ОТКАЗ: рука $A с исключением из фильтра не прошла safety с тем"
      echo "  же исполнением (чекпойнт, фаза, слои, харнесс, модули) и той же"
      echo "  причиной — режим $MODE для неё закрыт"
      exit 1
    fi
  done
fi

OUTD="reports/k15d/rollout/${MODE}_${SUITE}/s${SEED}/${TAG}"
LOGF="logs/k15d/rollout_${MODE}_${SUITE}_s${SEED}.log"
mkdir -p "$OUTD"
exec >> "$LOGF" 2>&1

echo "=== СТАРТ $(date) === режим $MODE, набор $SUITE, задачи $TASKS, карта" \
     "$DEV, сид $SEED, руки $ARMS"
echo "    коммит $(git rev-parse --short HEAD 2>/dev/null); каталог $OUTD"
for A in $K15D_ARMS; do
  OV="$(override_of "$A")"
  [ -n "$OV" ] && echo "    ВНИМАНИЕ: $A НЕ допущена фильтром, исключение: $OV"
done

COMMON="--ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
  --task-suite $SUITE --n-envs 5 --seed $SEED --rollout-seed-mode fixed \
  --ensemble off --horizon 8 --max-steps 600 \
  --run-tag k15d_${MODE}_${SUITE}_s${SEED} --device $DEV --save-actions"
DRVQ="--policy depthrvq --policy-ckpt data/k9d_ep3.pt \
  --q1-ckpt data/k14c/q1_main_s0.pt --expect-q1-variant main \
  --expect-q1-seed 0 --expect-q0-manifest data/k14d/q0_b8_e0.manifest.json \
  --depth-rvq-mode fast"

run_arm () {   # метка (возможно с суффиксом r — повтор), задача
  local L="$1" T="$2" rc=0 NEED="" BASE="${1%r}"
  for I0 in $STATES; do
    local F="$OUTD/${L}_t${T}_i${I0}.json"
    if [ -f "$F" ] && python3 -c "import json;json.load(open('$F'))" \
         >/dev/null 2>&1; then continue; fi
    rm -f "$F"
    NEED="$NEED${NEED:+,}$I0"
  done
  [ -z "$NEED" ] && { echo "    $L t$T: все блоки уже есть"; return 0; }
  echo "    $L t$T: блоки $NEED — $(date '+%H:%M:%S')"
  local -a EXTRA
  if [ "$BASE" = q0 ]; then
    # shellcheck disable=SC2206
    EXTRA=($DRVQ)
  else
    EXTRA=(--policy k15d --refiner "$(ck_of "$BASE")"
           --k15d-report "$(report_of "$BASE")")
    local OV; OV="$(override_of "$BASE")"
    [ -n "$OV" ] && EXTRA+=(--k15d-admission-override "$OV")
  fi
  # shellcheck disable=SC2086
  python3 experiments/k9h_multiarm_gate.py $COMMON "${EXTRA[@]}" \
    --task-id "$T" --init-starts "$NEED" --arm-label "$L" \
    --out "$OUTD/${L}_t${T}_i{i0}.json" || rc=$?
  if [ $rc -ne 0 ]; then
    echo "ОСТАНОВ: $L, задача $T, блоки $NEED — код $rc"
    return $rc
  fi
  echo "    $L t$T готово: $(grep -h 'успех ' "$LOGF" | tail -1)"
  sleep 10
}

for T in $TASKS; do
  echo "--- задача $T, $(date), свободно $(free -g | awk 'NR==2{print $7}') ГБ"
  set -- $ARMS
  N=$#; SEQ=""
  for k in $(seq 0 $((N - 1))); do
    i=$(( (k + T + SEED) % N + 1 ))
    SEQ="$SEQ ${!i}"
  done
  echo "    порядок рук:$SEQ"
  for A in $SEQ; do run_arm "$A" "$T"; done
done

if [ "$MODE" = "smoke" ]; then
  # ДЕТЕРМИНИЗМ: повтор КАЖДОЙ руки K-15d обязан дать побитово те же
  # действия. Повторы (метка с суффиксом r) в анализ не входят.
  for A in $K15D_ARMS; do
    for T in $TASKS; do run_arm "${A}r" "$T"; done
  done
  if ! python3 - "$OUTD" $K15D_ARMS <<'PY'
import glob, json, os, sys
d, arms = sys.argv[1], sys.argv[2:]
bad = 0
for arm in arms:
    for f in sorted(glob.glob(os.path.join(d, f"{arm}r_t*_i*.json"))):
        g = os.path.join(d, os.path.basename(f).replace(f"{arm}r_", f"{arm}_",
                                                         1))
        ea = json.load(open(f))["episodes"]
        eb = json.load(open(g))["episodes"]
        # ДЛИНА И СТАРТЫ СВЕРЯЮТСЯ ЯВНО: zip молча обрезал бы лишнее.
        if [e["init_state_id"] for e in ea] != \
                [e["init_state_id"] for e in eb]:
            print(f"    {os.path.basename(g)}: состав эпизодов повтора иной")
            bad += max(len(ea), len(eb))
            continue
        same = sum(x["action_sha1"] == y["action_sha1"]
                   for x, y in zip(ea, eb))
        print(f"    детерминизм {os.path.basename(g)}: {same}/{len(ea)} "
              f"эпизодов совпали")
        bad += len(ea) - same
sys.exit(1 if bad else 0)
PY
  then
    echo "ОСТАНОВ: повтор руки K-15d разошёлся"
    exit 1
  fi
fi

echo "=== раскатки закончены $(date) ==="
ARTS=""
for A in $ARMS; do
  ARTS="$ARTS $(ls "$OUTD"/${A}_t*_i*.json 2>/dev/null | tr '\n' ' ')"
done
CODE=0
if [ "$MODE" = "smoke" ]; then
  python3 experiments/k15d_behavior.py --mode safety --allow-partial \
    --expected-arms "$EXPECTED" --arts $ARTS \
    --out "$OUTD/behavior_smoke.json" --overwrite || CODE=$?
  [ "$CODE" = 4 ] && CODE=0     # на пяти эпизодах разность ничего не значит
  echo "=== СМОУК ЗАКОНЧЕН $(date), код $CODE. Это НЕ результат ==="
  exit $CODE
fi
TARG=""
[ "$MODE" = "target" ] && TARG="--tasks $(echo "$TASKS" | tr ' ' ',')"
# shellcheck disable=SC2086
python3 experiments/k15d_behavior.py --mode "$MODE" $TARG \
  --expected-arms "$EXPECTED" --arts $ARTS \
  --out "reports/k15d/rollout/behavior_${MODE}_s${SEED}_${SUITE}_${TAG}.json" \
  --overwrite || CODE=$?
echo "=== КОНЕЦ $(date), анализ: код $CODE ==="
exit $CODE
