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
    ap.add_argument("--partial", action="store_true",
                    help="не отбрасывать задачи, посчитанные не во всех "
                         "ячейках; ячейки станут НЕсравнимы")
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
                per_episode=[bool(e["success"]) for e in eps],
                by_init={int(e["init_state_id"]): bool(e["success"])
                         for e in eps})
    if not rows:
        raise SystemExit(f"в {args.dir} нет результатов")

    # ТОЛЬКО ЗАДАЧИ, ПОСЧИТАННЫЕ ВО ВСЕХ ЯЧЕЙКАХ. При частично добежавшем
    # прогоне ячейки опираются на разное число задач, и сравнивать их между
    # собой нельзя: разница между горизонтами смешается с разницей в составе
    # задач. Отбрасываем незавершённые и говорим об этом вслух.
    seen = {}
    for key, rec in rows.items():
        for t in rec:
            seen.setdefault(t, set()).add(key)
    full = {t for t, ks in seen.items() if ks == set(rows)}
    dropped = sorted(set(seen) - full)
    if dropped and not args.partial:
        print(f"  НЕПОЛНЫЕ ЗАДАЧИ ОТБРОШЕНЫ: {dropped}")
        print(f"  (посчитаны не во всех {len(rows)} ячейках; --partial "
              f"отключает отбор,\n   но тогда ячейки НЕсравнимы между собой)")
        rows = {k: {t: v for t, v in rec.items() if t in full}
                for k, rec in rows.items()}
        rows = {k: v for k, v in rows.items() if v}
        if not rows:
            raise SystemExit("нет ни одной задачи, посчитанной во всех ячейках")
    print(f"  задач в разборе: {len(full if not args.partial else seen)}, "
          f"ячеек: {len(rows)}")

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

    # ---- ПОЭПИЗОДНЫЙ ОРАКУЛ ---------------------------------------------
    # ГЛАВНЫЙ ВОПРОС МЕТОДА, и средняя кривая на него не отвечает. Плоская
    # кривая означает лишь, что В СРЕДНЕМ горизонт не важен. Но если при H=4
    # валятся одни эпизоды, а при H=20 другие, оракул, выбирающий горизонт под
    # каждое начальное состояние, взял бы и те и другие — запас есть, а в
    # агрегате его не видно. Если же валятся ОДНИ И ТЕ ЖЕ эпизоды, запаса нет.
    #
    # Различить это можно только по ПАРНЫМ исходам: развёртка гоняет все
    # горизонты с одинаковыми init_state_id, поэтому эпизод с данным
    # идентификатором — буквально одна и та же начальная сцена.
    #
    # НУЛЬ НЕЗАВИСИМОСТИ ОБЯЗАТЕЛЕН. Максимум по четырём плечам завышен сам по
    # себе: при независимых исходах с p=0.87 он даёт 1-0.13^4 = 0.9997, то есть
    # почти сто процентов из чистого шума. Поэтому исходы переставляются внутри
    # каждой пары (задача, H) независимо, и оракул считается заново: это то,
    # что даёт «четыре попытки» при отсутствии общей поэпизодной структуры.
    #
    # ОГОВОРКА: оракул выбирает ОДИН горизонт на весь эпизод по его начальному
    # состоянию. Настоящий адаптивный метод решает заново на каждом
    # перепланировании, то есть свободы у него больше. Поэтому вывод
    # односторонний: если даже этот оракул ничего не даёт над лучшим
    # фиксированным — довод против сильный; если даёт — это ещё не
    # доказательство, а повод идти в фазу C.
    print("\n" + "=" * 74)
    print("ПОЭПИЗОДНЫЙ ОРАКУЛ: есть ли запас у выбора горизонта по состоянию")
    print("=" * 74)
    rng = np.random.default_rng(0)
    for ens in sorted({e for e, _ in rows}):
        hs = sorted(h for e, h in rows if e == ens)
        if len(hs) < 2:
            continue
        tasks = sorted(set.intersection(*[set(rows[(ens, h)]) for h in hs]))
        # (задача, init_state_id) -> вектор исходов по горизонтам
        cols, per_task = {h: [] for h in hs}, []
        for t in tasks:
            ids = sorted(set.intersection(
                *[set(rows[(ens, h)][t]["by_init"]) for h in hs]))
            per_task.append(len(ids))
            for h in hs:
                cols[h] += [rows[(ens, h)][t]["by_init"][i] for i in ids]
        M = np.array([cols[h] for h in hs], bool)          # (H, эпизодов)
        if M.size == 0:
            continue
        marg = M.mean(1)
        best_i = int(np.argmax(marg))
        oracle = float(M.any(0).mean())

        # нуль: переставляем внутри каждой (задача, H) — так сохраняются и
        # маргинальные доли, и различия между задачами
        nulls = []
        bounds = np.cumsum([0] + per_task)
        for _ in range(2000):
            P = M.copy()
            for a, b in zip(bounds[:-1], bounds[1:]):
                for r in range(len(hs)):
                    P[r, a:b] = rng.permutation(P[r, a:b])
            nulls.append(P.any(0).mean())
        null = float(np.mean(nulls))

        print(f"\n  усреднение={ens}, эпизодов {M.shape[1]}, задач {len(tasks)}")
        print("    " + "  ".join(f"H={h}:{p:.1%}" for h, p in zip(hs, marg)))
        print(f"    лучший фиксированный H={hs[best_i]}: {marg[best_i]:.1%}")
        print(f"    поэпизодный оракул:            {oracle:.1%}"
              f"   (+{oracle - marg[best_i]:.1%})")
        print(f"    нуль независимости:            {null:.1%}"
              f"   (+{null - marg[best_i]:.1%})")
        print("    НУЛЬ НАСЫЩАЕТСЯ и потому мало о чём говорит: при успехе\n"
              "    около 87% и четырёх плечах он почти упирается в единицу.\n"
              "    Читать надо таблицы ниже. И помнить, что среда\n"
              "    ДЕТЕРМИНИРОВАНА: исход (эпизод, H) — не случайная величина,\n"
              "    измерительного шума в нём нет, поэтому разрыв оракула\n"
              "    настоящий. Вопрос не в том, реален ли он, а в том,\n"
              "    ПРЕДСКАЗУЕМ ли он из причинных признаков.")

        # ---- ПАРНЫЕ ТАБЛИЦЫ ПРОТИВ ЛУЧШЕГО ФИКСИРОВАННОГО ----------------
        # Не насыщаются и читаются напрямую: «спасено» — эпизоды, которые
        # лучший фиксированный горизонт проваливает, а альтернативный берёт.
        # Это и есть материал, на котором мог бы работать предсказатель.
        bm = M[best_i]
        print(f"\n    против лучшего H={hs[best_i]}, эпизодов {M.shape[1]}:")
        print(f"      {'H':>4}{'оба':>7}{'только лучший':>15}"
              f"{'СПАСЕНО альт.':>15}{'оба провал':>12}")
        for j, h in enumerate(hs):
            if j == best_i:
                continue
            alt = M[j]
            print(f"      {h:>4}{int((bm & alt).sum()):>7}"
                  f"{int((bm & ~alt).sum()):>15}{int((~bm & alt).sum()):>15}"
                  f"{int((~bm & ~alt).sum()):>12}")
        n_fail = int((~bm).sum())
        n_resc = int((~bm & M.any(0)).sum())
        print(f"    провалов у лучшего фиксированного: {n_fail}; из них хотя бы"
              f" один другой горизонт спасает: {n_resc}")
        if n_fail == 0:
            print("    лучший горизонт не проваливает ничего — запаса нет по"
                  " построению")
        elif n_resc == 0:
            print("    НИ ОДИН провал не спасается сменой горизонта: валятся\n"
                  "    одни и те же эпизоды при любом расписании, запаса у\n"
                  "    выбора горизонта по начальному состоянию НЕТ")
        else:
            print(f"    доля спасаемых провалов: {n_resc / n_fail:.0%} — вот"
                  " верхняя граница того,\n    что мог бы отыграть"
                  " предсказатель горизонта")

        # согласие исходов между горизонтами: доля эпизодов с одинаковым исходом
        print("\n    согласие исходов, доля эпизодов:")
        print("      " + "".join(f"{'H=' + str(h):>8}" for h in hs))
        for i, h in enumerate(hs):
            print(f"      H={h:<3}" + "".join(
                f"{(M[i] == M[j]).mean():>8.2f}" for j in range(len(hs))))

    if args.out:
        json.dump({f"ens{int(e)}_H{h}": v for (e, h), v in tab.items()},
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
