#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Годна ли готовая ячейка для пропуска: сверка ЕЁ конфигурации с ожидаемой.

«Файл непустой» — недостаточное условие: другая sigma, другой сид, другой
номер шага или другой список состояний дают файл с тем же именем и другим
смыслом. Здесь сверяются поля, а расхождение — отказ, а не молчаливый пропуск и
не перезапись.

Коды возврата: 0 — годна, 1 — нет файла, 2 — конфигурация не совпала.
"""
import json
import os
import sys


def read_meta(path):
    """Мета ячейки: сам JSON для оценочных, спутник .meta.json для буферов."""
    if path.endswith(".pt"):
        path = path + ".meta.json"
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        return json.load(open(path))
    except (ValueError, OSError):
        return None


def compare(meta, want):
    """(конфликты, отсутствующие поля) — ЭТО РАЗНЫЕ СЛУЧАИ.

    Конфликт означает, что ячейка получена при других настройках: её нельзя ни
    использовать, ни переписать, надо остановиться. Отсутствие поля означает
    лишь, что ячейка старше самой записи происхождения: сравнивать не с чем, и
    безопасный выход — пересчитать её, а не останавливать всю лестницу.
    """
    cfg = dict(meta.get("run_config") or {})
    bad, missing = [], []
    for k, v in sorted(want.items()):
        got = cfg.get(k, meta.get(k, None))
        if got is None:
            missing.append(f"{k}: в ячейке нет поля, ожидалось {v}")
            continue
        try:
            same = abs(float(got) - float(v)) < 1e-9
        except (TypeError, ValueError):
            same = str(got) == str(v)
        if not same:
            bad.append(f"{k}: в ячейке {got!r}, ожидалось {v!r}")
    return bad, missing


def selftest():
    import tempfile
    d = tempfile.mkdtemp(prefix="k12j_")
    p = os.path.join(d, "cell.json")
    json.dump(dict(run_config=dict(sigma=0.1, arm="g_rl", step_index=2,
                                   task_id=3, init_start=30, d1_seed=0,
                                   rl_seed=0)), open(p, "w"))
    m = read_meta(p)
    assert compare(m, dict(sigma=0.1, arm="g_rl", step_index=2)) == ([], [])
    assert compare(m, dict(sigma=0.03))[0][0].startswith("sigma:")
    assert compare(m, dict(step_index=3))[0][0].startswith("step_index:")
    # отсутствующее поле — во ВТОРОЙ список, а не в конфликты
    assert compare(m, dict(eval_eps_seed=777)) == (
        [], ["eval_eps_seed: в ячейке нет поля, ожидалось 777"])
    assert read_meta(os.path.join(d, "нет.json")) is None
    # спутник для буфера
    q = os.path.join(d, "roll.pt")
    json.dump(dict(run_config=dict(sigma=0.1)), open(q + ".meta.json", "w"))
    assert compare(read_meta(q), dict(sigma=0.1)) == ([], [])
    # старая ячейка без происхождения: не конфликт, но и не годна — пересчёт
    r = os.path.join(d, "old.json")
    json.dump(dict(episodes=[]), open(r, "w"))
    conf, miss = compare(read_meta(r), dict(sigma=0.1))
    assert conf == [] and miss, (conf, miss)
    print("самопроверка k12j_cell_ok пройдена")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        return selftest()
    if len(sys.argv) < 2:
        print("нужен путь к ячейке и пары ключ=значение", file=sys.stderr)
        return 2
    meta = read_meta(sys.argv[1])
    if meta is None:
        return 1
    want = {}
    for a in sys.argv[2:]:
        if "=" not in a:
            print(f"аргумент {a!r} не вида ключ=значение", file=sys.stderr)
            return 2
        k, v = a.split("=", 1)
        want[k] = v
    bad, missing = compare(meta, want)
    if bad:
        print(f"{sys.argv[1]}: КОНФЛИКТ " + "; ".join(bad), file=sys.stderr)
        return 2
    if missing:
        # не годна для пропуска, но и не конфликт: пусть пересчитают
        print(f"{sys.argv[1]}: без происхождения — " + "; ".join(missing),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
