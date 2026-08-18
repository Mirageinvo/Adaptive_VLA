"""Таблица pos_offset ПО ЗАДАЧАМ из официального протокола BAR.

ЗАЧЕМ. `scripts/eval_libero.py` имеет умолчание pos_offset=4, но официальный
запуск `scripts/eval_libero_bar.sh` задаёт офсет ОТДЕЛЬНО ДЛЯ КАЖДОЙ ЗАДАЧИ:

    goal    (4 4 4 4 4 4 4 4 3 3)
    spatial (4 4 4 4 3 3 3 3 4 3)
    object  (4 4 4 4 4 4 4 4 4 3)
    long    (4 3 4 4 3 4 3 3 4 3)

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

SUITES = ("goal", "spatial", "object", "long")


def parse_offsets(sh_path: str):
    """Массивы POS_OFFSET_<SUITE> из вендоренного скрипта."""
    src = open(sh_path).read()
    out = {}
    for s in SUITES:
        m = re.search(rf"POS_OFFSET_{s.upper()}=\(([^)]*)\)", src)
        if not m:
            raise SystemExit(f"в {sh_path} не найден POS_OFFSET_{s.upper()}")
        vals = [int(x) for x in m.group(1).split()]
        if len(vals) != 10:
            raise SystemExit(f"{s}: ожидалось 10 офсетов, найдено {len(vals)}")
        out[s] = vals
    return out, hashlib.sha256(src.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--script", default="scripts/eval_libero_bar.sh")
    ap.add_argument("--out", default="data/pos_offset_table.json")
    args = ap.parse_args()

    sh = os.path.join(args.root, args.script)
    offs, sh_hash = parse_offsets(sh)
    print(f"офсеты разобраны из {sh}")
    for s in SUITES:
        print(f"  {s:>8}: {offs[s]}")
    tot4 = sum(v.count(4) for v in offs.values())
    tot3 = sum(v.count(3) for v in offs.values())
    print(f"  всего задач {tot3 + tot4}: офсет 4 у {tot4}, офсет 3 у {tot3}\n")

    try:
        from libero.libero import benchmark
    except ImportError as e:
        raise SystemExit(
            f"нужен пакет libero (тот же, что для симулятора): {e}\n"
            "он же используется в scripts/utils.py:404")

    bd = benchmark.get_benchmark_dict()
    table, dup = {}, []
    for s in SUITES:
        suite = bd[f"libero_{s}"]()
        for tid in range(10):
            lang = suite.get_task(tid).language
            if lang in table:
                dup.append(lang)
            table[lang] = dict(suite=s, task_id=tid, pos_offset=offs[s][tid])
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
