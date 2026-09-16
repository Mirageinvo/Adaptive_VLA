#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-13e: попарная сверка двух наборов ячеек по каждому (задача, состояние).

ЗАЧЕМ ОТДЕЛЬНЫМ ИСПОЛНИТЕЛЕМ. Совпадение СУММАРНОГО успеха не доказывает, что
два прогона прошли одинаково: в K-13c рука hicora_t s0 сохранила ровно тот же
успех при восьми переворотах 4 против 4. Утверждение «новый скрипт
воспроизводит старый путь» требует сверки ИСХОДОВ, а не средних, и раз это
утверждение попадает в текст, оно считается кодом с самопроверкой, а не
разовой командой в терминале.

ЧТО СЧИТАЕТСЯ. Для каждой пары (рука, голова, задача, состояние), встреченной
в обоих наборах:
  - совпал ли исход,
  - совпал ли ХЭШ НАЧАЛЬНОГО СОСТОЯНИЯ. Если нет — сверка недействительна
    целиком: сравнивались бы разные эпизоды, а не два вычисления одного.
Разность оценивается точным McNemar по переворотам.

ЧЕГО ЭТОТ СЧЁТ НЕ ДАЁТ. Он НЕ измеряет «шум повторного запуска»: два набора
различаются известным образом (режим точности, версия скрипта), а не как две
выборки одного распределения. Это чувствительность детерминированной политики
к названному изменению, и только так её и следует называть.
"""
import argparse
import glob
import json
import os
import sys
from math import comb


def mcnemar_exact(b, c):
    """Двусторонний точный тест на симметрию переворотов."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / float(2 ** n)
    return min(1.0, 2.0 * tail)


def index_cells(paths, arm_from=None, head_from=None):
    """{(рука, голова, задача, состояние): (успех, хэш)} с отказом на дублях.

    Дубль молча перезаписался бы, и ячейка, посчитанная дважды разными
    версиями, осталась бы незамеченной — а именно её мы и ищем.
    """
    out, seen = {}, {}
    for p in paths:
        c = json.load(open(p))
        arm = arm_from or str(c.get("arm"))
        head = head_from or str(c.get("head"))
        for e in c["episodes"]:
            k = (arm, head, int(e["task_id"]), int(e["state_id"]))
            if k in out:
                raise SystemExit(
                    f"пара {k} встречена дважды: {seen[k]} и {p}")
            out[k] = (bool(e["success"]), e.get("init_hash_full"))
            seen[k] = p
    return out


def compare(a, b):
    """Разбивка по рукам: перевороты в обе стороны и точный p."""
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("у наборов нет общих пар — сверять нечего")
    bad_hash = [k for k in common if a[k][1] != b[k][1]]
    rows = {}
    for k in common:
        arm = f"{k[0]}:{k[1]}"
        r = rows.setdefault(arm, dict(n=0, up=0, down=0))
        r["n"] += 1
        if a[k][0] != b[k][0]:
            r["up" if b[k][0] else "down"] += 1
    for arm, r in rows.items():
        r["flips"] = r["up"] + r["down"]
        r["p"] = mcnemar_exact(r["up"], r["down"])
        r["delta_pp"] = 100.0 * (r["up"] - r["down"]) / r["n"]
    return rows, common, bad_hash


def selftest():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(5, 0) - 2 ** -4) < 1e-12      # 2 * (1/32)
    assert mcnemar_exact(3, 3) == 1.0

    mk = lambda succ, h: (succ, h)
    A = {("f", "None", 3, s): mk(s % 2 == 0, f"h{s}") for s in range(10)}
    B = dict(A)
    B[("f", "None", 3, 1)] = mk(True, "h1")       # провал -> успех
    B[("f", "None", 3, 2)] = mk(False, "h2")      # успех -> провал
    rows, common, bad = compare(A, B)
    assert not bad and len(common) == 10
    r = rows["f:None"]
    assert r["flips"] == 2 and r["up"] == 1 and r["down"] == 1
    assert abs(r["delta_pp"]) < 1e-9 and r["p"] == 1.0

    # расхождение хэша обязано быть замечено, а не утонуть в статистике
    C = dict(A)
    C[("f", "None", 3, 4)] = mk(True, "ДРУГОЙ")
    _r, _c, bad = compare(A, C)
    assert bad == [("f", "None", 3, 4)], bad

    # ноль переворотов — тоже результат, и он должен проходить
    rows, _, _ = compare(A, dict(A))
    assert rows["f:None"]["flips"] == 0
    print("самопроверка k13e_flip_audit пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--a", action="append", default=[],
                    help="маска первого набора; 'маска=рука=голова', чтобы "
                         "переименовать руку старого формата")
    ap.add_argument("--b", action="append", default=[])
    ap.add_argument("--label-a", default="старый")
    ap.add_argument("--label-b", default="новый")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not a.a or not a.b:
        ap.error("нужны --a и --b")

    def load(specs):
        acc = {}
        for sp in specs:
            parts = sp.split("=")
            pat, arm, head = (parts + [None, None])[:3]
            found = sorted(glob.glob(pat))
            if not found:
                raise SystemExit(f"по маске {pat} нет ячеек")
            got = index_cells(found, arm, head)
            dup = set(got) & set(acc)
            if dup:
                raise SystemExit(f"маски пересекаются по {sorted(dup)[:3]}")
            acc.update(got)
        return acc

    A, B = load(a.a), load(a.b)
    rows, common, bad = compare(A, B)
    print(f"\n  СВЕРКА «{a.label_a}» против «{a.label_b}»")
    print(f"    общих пар: {len(common)} "
          f"(в первом {len(A)}, во втором {len(B)})")
    if bad:
        raise SystemExit(
            f"у {len(bad)} пар разошёлся хэш начального состояния, например "
            f"{bad[:3]}: это разные эпизоды, а не два вычисления одного. "
            f"Сверка недействительна")
    print(f"    хэши начальных состояний совпали у всех {len(common)} пар")
    print(f"\n    {'рука':22s} {'пар':>4s} {'перев':>6s} {'->усп':>6s} "
          f"{'->пров':>7s} {'сдвиг':>8s} {'p':>7s}")
    tot_n = tot_f = 0
    for arm, r in sorted(rows.items()):
        print(f"    {arm:22s} {r['n']:4d} {r['flips']:6d} {r['up']:6d} "
              f"{r['down']:7d} {r['delta_pp']:+7.2f} {r['p']:7.3f}")
        tot_n += r["n"]
        tot_f += r["flips"]
    print(f"    {'ВСЕГО':22s} {tot_n:4d} {tot_f:6d}"
          f"   ({100.0 * tot_f / tot_n:.1f}% пар поменяли исход)")
    if tot_f == 0:
        print("\n    исходы совпали ПОБИТОВО: второй набор воспроизводит "
              "первый на каждой паре")
    else:
        print("\n    исходы совпадают НЕ на всех парах — совпадение средних, "
              "если оно есть, воспроизведением пути не является")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        tmp = f"{a.out}.tmp.{os.getpid()}"
        json.dump(dict(label_a=a.label_a, label_b=a.label_b,
                       masks_a=a.a, masks_b=a.b, n_common=len(common),
                       hash_mismatch=len(bad),
                       rows={k: v for k, v in rows.items()},
                       script_sha1=__import__("hashlib").sha1(
                           open(os.path.abspath(__file__), "rb").read()
                       ).hexdigest()[:12]),
                  open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, a.out)
        print(f"    сохранено: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
