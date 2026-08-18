"""Таблица pos_offset ПО ЗАДАЧАМ из официального протокола BAR.

ЗАЧЕМ. `scripts/eval_libero.py` имеет умолчание pos_offset=4, но официальный
запуск `scripts/eval_libero_bar.sh` задаёт офсет ОТДЕЛЬНО ДЛЯ КАЖДОЙ ЗАДАЧИ:

    goal    (4 4 4 4 4 4 4 4 3 3)
    spatial (4 4 4 4 3 3 3 3 4 3)
    object  (4 4 4 4 4 4 4 4 4 3)
    10      (4 3 4 4 3 4 3 3 4 3)   переменная POS_OFFSET_LONG, suite "10"

Замер k4b0_padding_probe: между офсетами 3 и 4 меняется сам план BAR, а с ним
stale, z_ref и полезность позиций. Оракул при офсете 3 даёт 0.941, при 4 —
0.872. Значит ЕДИНЫЙ офсет не воспроизводит опубликованный режим модели.

ЧТО ДЕЛАЕМ. Массивы РАЗБИРАЮТСЯ ИЗ САМОГО .sh, а не дублируются константами:
иначе при обновлении вендоренного кода таблицы молча разойдутся. Описания задач
берутся из пакета libero тем же вызовом, что и в `scripts/utils.py:404`.

Результат — JSON вида {описание задачи: {suite, task_id, pos_offset}} плюс
sha256 исходного .sh. И файл, и хеш кладутся в metadata датасета.

Запуск (нужен установленный libero, как для симулятора):
    python3 experiments/k4b0_offset_table.py --out data/pos_offset_table.json
"""

import argparse
import hashlib
import json
import os
import re
import sys

# Имя переменной в .sh и имя suite для реестра — РАЗНЫЕ: последняя строка
# скрипта это `run_suite "10" "Long" POS_OFFSET_LONG`, то есть длинный набор
# передаётся как "10", а в реестре LIBERO зовётся libero_10. Пара
# (суффикс переменной, имя suite) берётся из скрипта, а не придумывается.
SUITES = (("goal", "goal"), ("spatial", "spatial"),
          ("object", "object"), ("long", "10"))


def parse_offsets(sh_path: str):
    """Массивы POS_OFFSET_<SUITE> из вендоренного скрипта."""
    src = open(sh_path).read()
    out = {}
    for var, suite in SUITES:
        m = re.search(rf"POS_OFFSET_{var.upper()}=\(([^)]*)\)", src)
        if not m:
            raise SystemExit(f"в {sh_path} не найден POS_OFFSET_{var.upper()}")
        vals = [int(x) for x in m.group(1).split()]
        if len(vals) != 10:
            raise SystemExit(f"{var}: ожидалось 10 офсетов, найдено {len(vals)}")
        out[suite] = vals
        # сверяем, что скрипт действительно вызывает run_suite с этим именем
        if not re.search(rf'run_suite\s+"{suite}"\s+\S+\s+POS_OFFSET_{var.upper()}',
                         src):
            raise SystemExit(
                f"в {sh_path} нет вызова run_suite \"{suite}\" с "
                f"POS_OFFSET_{var.upper()} — имена suite разошлись")
    return out, hashlib.sha256(src.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--script", default="scripts/eval_libero_bar.sh")
    ap.add_argument("--out", default="data/pos_offset_table.json")
    ap.add_argument("--libero-path", default=None,
                    help="путь к склонированному репозиторию LIBERO. Нужен, "
                         "если пакет не установлен: реестр задач читается по "
                         "пути, БЕЗ установки в рабочее окружение")
    args = ap.parse_args()

    sh = os.path.join(args.root, args.script)
    offs, sh_hash = parse_offsets(sh)
    print(f"офсеты разобраны из {sh}")
    for _, suite in SUITES:
        print(f"  {suite:>8}: {offs[suite]}")
    tot4 = sum(v.count(4) for v in offs.values())
    tot3 = sum(v.count(3) for v in offs.values())
    print(f"  всего задач {tot3 + tot4}: офсет 4 у {tot4}, офсет 3 у {tot3}\n")

    if args.libero_path:
        sys.path.insert(0, os.path.abspath(args.libero_path))
    try:
        from libero.libero import benchmark
    except ImportError as e:
        raise SystemExit(
            f"реестр задач LIBERO недоступен: {e}\n"
            "Он же используется в scripts/utils.py:404 и НЕ заменяется "
            "сопоставлением строк по смыслу: suite узнаётся легко, а офсет "
            "задан ПОРЯДКОМ внутри suite, и перестановка внутри десятки меняет "
            "офсет у большинства задач.\n"
            "Без установки в рабочее окружение:\n"
            "  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/"
            "LIBERO.git /tmp/LIBERO\n"
            "  python3 experiments/k4b0_offset_table.py "
            "--libero-path /tmp/LIBERO --out data/pos_offset_table.json")

    src_mod = getattr(benchmark, "__file__", "?")
    print(f"реестр задач: {src_mod}")

    bd = benchmark.get_benchmark_dict()
    print(f"ключи реестра: {sorted(bd)}")
    table, dup = {}, []
    for _, suite in SUITES:
        key = f"libero_{suite}"
        if key not in bd:
            raise SystemExit(f"в реестре нет {key}; доступны {sorted(bd)}")
        bench = bd[key]()
        for tid in range(10):
            lang = bench.get_task(tid).language
            if lang in table:
                dup.append(lang)
            table[lang] = dict(suite=suite, task_id=tid,
                               pos_offset=offs[suite][tid])
    if dup:
        raise SystemExit(f"описания задач повторяются между suite: {dup}")
    assert len(table) == 40, f"задач {len(table)}, ожидалось 40"

    print(f"описаний задач: {len(table)}, повторов нет")
    by_off = {}
    for v in table.values():
        by_off.setdefault(v["pos_offset"], []).append(v["suite"])
    for k in sorted(by_off):
        print(f"  офсет {k}: {len(by_off[k])} задач")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    blob = dict(source=args.script, source_sha256=sh_hash,
                offsets_by_suite=offs, tasks=table)
    with open(args.out, "w") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)
    h = hashlib.sha256(open(args.out, "rb").read()).hexdigest()[:16]
    print(f"\nсохранено: {args.out}  sha256:{h}")
    print(f"хеш исходного .sh: {sh_hash[:16]}")


if __name__ == "__main__":
    main()
