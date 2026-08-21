"""K-5b: сводка развёртки по горизонту из сохранённых JSON.

ЗАЧЕМ ОТДЕЛЬНО. Каждая конфигурация считается в своём процессе, поэтому свести
их может только внешний разбор. Он же позволяет пересчитать сводку, не запуская
ничего заново, и проверить то, что внутри одного процесса проверить нельзя.

ПРОВЕРКА ТОЖДЕСТВА. При H, равном длине чанка, каждый момент времени покрыт
ровно одним планом, поэтому усреднение — пустая операция, и `ens=on` обязан
дать ПОТОЧЕЧНО те же исходы, что `ens=off`. Расхождение означает утечку
состояния или недетерминированность стенда; именно так был пойман прежний
дефект с переиспользованием сред.

ИНТЕРВАЛЫ. Кластерный бутстрап по ЗАДАЧАМ, а не по эпизодам: эпизоды одной
задачи зависимы (общая сцена, общий объект, общая политика), и независимая
пересэмплировка эпизодов занизила бы интервал.

Запуск:
    python3 experiments/k5b_summarize.py --selftest
    python3 experiments/k5b_summarize.py --dir data/k5b_sweep
"""

import argparse
import glob
import json
import os

import numpy as np


def cluster_ci(succ_by_task, n_boot=2000, seed=0):
    """Доля успеха с бутстрапом ПО ЗАДАЧАМ.

    succ_by_task: список пар (успехов, эпизодов) по задачам.
    """
    if not succ_by_task:
        return 0.0, 0.0, 0.0
    s = np.array([a for a, _ in succ_by_task], float)
    n = np.array([b for _, b in succ_by_task], float)
    pt = s.sum() / max(n.sum(), 1)
    idx = np.random.default_rng(seed).integers(0, len(s), (n_boot, len(s)))
    b = s[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(pt), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def selftest():
    """Известный ответ для агрегации и интервала."""
    # три задачи по 10 эпизодов: 10, 9, 8 успехов -> 27/30
    pt, lo, hi = cluster_ci([(10, 10), (9, 10), (8, 10)])
    assert abs(pt - 0.9) < 1e-12, f"доля посчитана неверно: {pt}"
    assert lo < pt < hi, "интервал не накрывает точку"
    # вырожденный случай: все успешны -> интервал схлопывается в единицу
    pt, lo, hi = cluster_ci([(10, 10)] * 5)
    assert pt == 1.0 and lo == 1.0 and hi == 1.0, "вырожденный случай неверен"
    # интервал по ЗАДАЧАМ шире, чем наивный по эпизодам: одна плохая задача
    # из трёх даёт больший разброс, чем те же исходы, размазанные ровно
    wide = cluster_ci([(10, 10), (10, 10), (0, 10)])
    even = cluster_ci([(7, 10), (7, 10), (6, 10)])
    assert (wide[2] - wide[1]) > (even[2] - even[1]), \
        "кластеризация по задачам не расширяет интервал"
    print("самопроверка пройдена: агрегация верна, интервал кластеризован "
          "по задачам")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/k5b_sweep")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()

    rows = {}
    for f in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        d = json.load(open(f))
        for k, v in d.items():
            if k == "meta":
                continue
            key = (bool(v["ensemble"]), int(v["horizon"]))
            rec = rows.setdefault(key, {})
            eps = v["episodes"]
            rec[int(v["task_id"])] = dict(
                succ=sum(e["success"] for e in eps), n=len(eps),
                calls=sum(e["policy_calls"] for e in eps),
                steps=sum(e["env_steps"] for e in eps),
                ms=sum(e["policy_ms"] for e in eps),
                per_episode=[bool(e["success"]) for e in eps])
    if not rows:
        raise SystemExit(f"в {args.dir} нет результатов")

    print("=" * 74)
    print(f"РАЗВЁРТКА ПО ГОРИЗОНТУ, {args.dir}")
    print("=" * 74)
    print(f"  {'усредн.':>8}{'H':>4}{'задач':>7}{'эпиз.':>7}{'успех':>9}"
          f"{'95% ДИ по задачам':>22}{'выз/дейст':>11}{'мс/дейст':>10}")
    tab = {}
    for (ens, H) in sorted(rows):
        rec = rows[(ens, H)]
        by_task = [(r["succ"], r["n"]) for r in rec.values()]
        pt, lo, hi = cluster_ci(by_task)
        st = sum(r["steps"] for r in rec.values())
        cl = sum(r["calls"] for r in rec.values())
        ms = sum(r["ms"] for r in rec.values())
        tab[(ens, H)] = dict(sr=pt, lo=lo, hi=hi, tasks=len(rec),
                             episodes=sum(r["n"] for r in rec.values()),
                             calls_per_action=cl / max(st, 1),
                             ms_per_action=ms / max(st, 1))
        print(f"  {str(ens):>8}{H:>4}{len(rec):>7}"
              f"{sum(r['n'] for r in rec.values()):>7}{pt:>8.1%}"
              f"      [{lo:>6.1%}, {hi:>6.1%}]{cl / max(st, 1):>11.3f}"
              f"{ms / max(st, 1):>10.1f}")

    # ---- ПРОВЕРКА ТОЖДЕСТВА ПРИ H = ДЛИНЕ ЧАНКА -------------------------
    print("\n  проверка тождества при H=20 (усреднение обязано быть пустым):")
    a, b = rows.get((False, 20)), rows.get((True, 20))
    if not a or not b:
        print("    нет обеих половин — пропущено")
    else:
        bad = [t for t in sorted(set(a) & set(b))
               if a[t]["per_episode"] != b[t]["per_episode"]]
        if bad:
            print(f"    РАСХОЖДЕНИЕ на задачах {bad} — стенд недетерминирован "
                  f"или состояние протекает между процессами")
        else:
            print(f"    поточечно совпадает на {len(set(a) & set(b))} задачах")

    # ---- ГЕТЕРОГЕННОСТЬ ПО ЗАДАЧАМ --------------------------------------
    # Предикат адаптивности: у разных задач ЛУЧШИЙ горизонт разный. Если он
    # везде одинаков, хватает одного фиксированного числа, и адаптивность не
    # нужна независимо от формы средней кривой.
    print("\n  лучший горизонт по задачам (ens=off), доля успеха:")
    off = {H: rows[(False, H)] for _, H in rows if (False, H) in rows}
    if off:
        hs = sorted(off)
        tasks = sorted({t for H in hs for t in off[H]})
        print(f"    {'задача':>7}" + "".join(f"{'H=' + str(H):>9}" for H in hs)
              + f"{'лучший':>9}")
        best = {}
        for t in tasks:
            vals = [off[H][t]["succ"] / off[H][t]["n"] if t in off[H] else None
                    for H in hs]
            bh = max((v, H) for v, H in zip(vals, hs) if v is not None)[1]
            best[t] = bh
            print(f"    {t:>7}" + "".join(
                f"{v:>9.0%}" if v is not None else f"{'—':>9}" for v in vals)
                + f"{bh:>9}")
        uniq = sorted(set(best.values()))
        print(f"    различных лучших горизонтов: {len(uniq)} из {len(hs)} "
              f"возможных — {uniq}")
        print("    ЧИТАТЬ ТАК: один и тот же лучший горизонт у всех задач\n"
              "    означает, что хватает фиксированного числа. Разные лучшие\n"
              "    горизонты — предикат для адаптивности, но по ЗАДАЧАМ он\n"
              "    лечится таблицей; метод требует различия ВНУТРИ задачи.")

    if args.out:
        json.dump({f"ens{int(e)}_H{h}": v for (e, h), v in tab.items()},
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
