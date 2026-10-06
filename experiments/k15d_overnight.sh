#!/bin/bash
# K-15d: ночная цепочка роллаутов с явной машиной состояний.
#
#   этап     код 0               код 4                    иначе (технический)
#   confirm  -> final            final пропустить,        СТОП всей цепочки
#                                продолжить
#   final    продолжить          отрицательное            отметить отказ,
#                                подтверждение, далее     независимые этапы идут
#   safety   -> ablate           ablate запрещён          ablate запрещён
#   ablate   готово              (не ожидается)           отметить отказ
#
# safety здесь — переанализ: если каталог прежнего safety той же версии на
# месте, раскатки пропускаются и пересчитывается только сводка (её строгие
# поля нужны допуску D0 в ablate). Иначе safety раскатывается заново.
#
# ДО ЗАПУСКА проверяется, что D0_OVERRIDE побуквенно совпадает с причиной
# исключения в отчёте проверки вывода D0: иначе safety и ablate отказали бы
# через несколько часов.
#
# В конце — таблица всех кодов; код возврата ненулевой при любом
# техническом отказе. Git здесь только читается.
#
#   export D0_OVERRIDE="..."
#   setsid nohup bash experiments/k15d_overnight.sh cuda:1 \
#     > logs/k15d/night_$(date +%Y%m%dT%H%M%S).log 2>&1 &
set -uo pipefail
DEV="${1:-cuda:1}"
R=experiments/k15d_rollout.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p reports/k15d/codes logs/k15d
DTAG="$(echo "$DEV" | tr -d ':')"
D0_REP="reports/k15d/inference_d0_s0_${DTAG}.json"

echo "=== K-15d ночь: старт $(date), карта $DEV, коммит" \
     "$(git rev-parse --short HEAD 2>/dev/null)"
[ -f "$D0_REP" ] || { echo "ОТКАЗ: нет $D0_REP"; exit 1; }
if ! python3 - "$D0_REP" "${D0_OVERRIDE:-}" <<'PY'
import json, sys
rep, ov = sys.argv[1], sys.argv[2]
got = (json.load(open(rep)).get("admission_override") or {}).get("reason")
if got != ov:
    print(f"ОТКАЗ: D0_OVERRIDE не совпадает с причиной в {rep}:\n"
          f"  в отчёте: {got!r}\n  задано:   {ov!r}")
    sys.exit(1)
print("  причина исключения D0 совпадает с отчётом проверки вывода")
PY
then
  exit 1
fi

declare -A CODE NOTE
TECH=0
run () {   # этап -> код в CODE[этап]
  local st="$1" c
  echo; echo "=== $st — $(date '+%Y-%m-%d %H:%M:%S')"
  bash "$R" "$st" "$DEV"
  c=$?
  CODE[$st]=$c
  printf '%s %s\n' "$c" "$(git rev-parse --short HEAD 2>/dev/null)" \
    > "reports/k15d/codes/night_${st}.code"
  echo "=== $st: код $c"
}
skip () { CODE[$1]="—"; NOTE[$1]="$2"; echo; echo "=== $1 пропущен: $2"; }

finish () {
  echo
  echo "=== ИТОГ $(date '+%Y-%m-%d %H:%M:%S')"
  printf '  %-8s %-4s %s\n' "этап" "код" "примечание"
  for st in confirm final safety ablate; do
    printf '  %-8s %-4s %s\n' "$st" "${CODE[$st]:-—}" "${NOTE[$st]:-}"
  done
  echo "  технических отказов: $TECH"
  exit $(( TECH > 0 ? 1 : 0 ))
}

# --- confirm --------------------------------------------------------------
run confirm
case "${CODE[confirm]}" in
  0) NOTE[confirm]="h18 ПОДТВЕРЖДЁН" ;;
  4) NOTE[confirm]="h18 НЕ подтверждён" ;;
  *) NOTE[confirm]="технический отказ — цепочка остановлена"; TECH=$((TECH+1))
     skip final "confirm технически отказал"
     skip safety "confirm технически отказал"
     skip ablate "confirm технически отказал"
     finish ;;
esac

# --- final ----------------------------------------------------------------
if [ "${CODE[confirm]}" = 0 ]; then
  run final
  case "${CODE[final]}" in
    0) NOTE[final]="h18 ПОДТВЕРЖДЁН на финальном банке" ;;
    4) NOTE[final]="отрицательное финальное подтверждение" ;;
    *) NOTE[final]="технический отказ"; TECH=$((TECH+1)) ;;
  esac
else
  skip final "confirm не подтвердил h18 — финальный банк не открывается"
fi

# --- safety (переанализ) --------------------------------------------------
run safety
case "${CODE[safety]}" in
  0) NOTE[safety]="технический пилот пройден" ;;
  4) NOTE[safety]="катастрофа — ablate запрещён" ;;
  *) NOTE[safety]="технический отказ — ablate запрещён"; TECH=$((TECH+1)) ;;
esac

# --- ablate ---------------------------------------------------------------
if [ "${CODE[safety]}" = 0 ]; then
  run ablate
  case "${CODE[ablate]}" in
    0) NOTE[ablate]="сводный dev готов" ;;
    *) NOTE[ablate]="технический отказ"; TECH=$((TECH+1)) ;;
  esac
else
  skip ablate "safety не пройден (код ${CODE[safety]})"
fi

finish
