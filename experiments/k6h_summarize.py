"""K-6h, агрегатор: парное сравнение levels=1 против levels=3.

ПОЧЕМУ ОТДЕЛЬНЫЙ СКРИПТ. Каждая ячейка — свой процесс со своим файлом, потому
что среды нельзя переиспользовать. Средний успех одной ячейки из десяти
эпизодов не значит ничего; решение принимается только здесь.

ЧТО СЧИТАЕТСЯ. Эпизоды сопоставляются по ключу (suite, task_id, ensemble,
horizon, init_state_id) — то есть один и тот же эпизод при двух комплектациях
уровней. Дальше:
  * сверка парности по хешу начального наблюдения; расхождение — отказ, а не
    предупреждение, потому что тогда сравниваются разные эпизоды;
  * micro (по всем эпизодам) и macro (среднее по задачам) разности;
  * дискордантные пары и точный тест Макнемара;
  * кластерный бутстрап ПО ЗАДАЧАМ: эпизоды внутри задачи скоррелированы, и
    бутстрап по эпизодам дал бы интервал уже истинного;
  * ОДНОСТОРОННЯЯ нижняя граница — именно она отвечает на вопрос «не хуже ли
    чем на δ», тогда как пересечение двустороннего интервала с нулём означает
    лишь отсутствие доказательства разницы.

ПРАВИЛО ЧТЕНИЯ записано в k6h_coarse_gate.py до запуска: граница выше -10
пунктов -> тонкие уровни не нужны; ниже -> нужны.

Запуск:
    python3 experiments/k6h_summarize.py --selftest
    python3 experiments/k6h_summarize.py --glob 'data/k6h/*.json' --margin 10
"""

import argparse
import glob as globmod
import json
import math
import os
from collections import defaultdict

import numpy as np


def mcnemar_exact(b, c):
    """Двусторонний точный тест Макнемара. b и c — дискордантные пары.

    Хи-квадрат приближение здесь плохо: при 200 эпизодах дискордантных пар
    бывает десяток, и приближение завышает значимость.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cluster_bootstrap(by_task, n_boot=20000, seed=0):
    """Бутстрап ПО ЗАДАЧАМ: ресэмплируются задачи целиком, вместе со всеми
    своими эпизодами. Так учитывается, что эпизоды внутри задачи зависимы."""
    tasks = sorted(by_task)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(tasks), len(tasks))
        d = np.concatenate([by_task[tasks[j]] for j in pick])
        out[b] = d.mean()
    return out


def selftest():
    # 1. Макнемар: симметричные дискорданты не значимы, односторонние значимы.
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) > 0.9
    assert mcnemar_exact(0, 10) < 0.01, mcnemar_exact(0, 10)
    assert abs(mcnemar_exact(0, 1) - 1.0) < 1e-12, "одна пара не улика"

    # 2. Кластерный бутстрап ОБЯЗАН давать более широкий интервал, чем наивный,
    #    когда разность различается по задачам. Если нет — кластеризация не
    #    работает, и все выводы будут переуверенными.
    by_task = {f"t{i}": np.full(20, 0.5 if i < 5 else -0.5) for i in range(10)}
    flat = np.concatenate(list(by_task.values()))
    cl = cluster_bootstrap(by_task, 4000, seed=0)
    rng = np.random.default_rng(0)
    naive = np.array([rng.choice(flat, len(flat), replace=True).mean()
                      for _ in range(4000)])
    w_cl = np.percentile(cl, 97.5) - np.percentile(cl, 2.5)
    w_nv = np.percentile(naive, 97.5) - np.percentile(naive, 2.5)
    assert w_cl > 1.5 * w_nv, f"кластерный {w_cl:.3f} не шире наивного {w_nv:.3f}"

    # 3. Односторонняя граница строго выше нижней двусторонней (5% против 2.5%).
    x = cluster_bootstrap({f"t{i}": np.zeros(10) + i * 0.01 for i in range(10)},
                          4000, seed=1)
    assert np.percentile(x, 5) >= np.percentile(x, 2.5) - 1e-12

    # 4. Парность: разность считается по СОВПАДАЮЩИМ ключам, а не по средним
    #    двух наборов. Подмена среднего разностью средних — типичная ошибка.
    a = {1: 1, 2: 0, 3: 1}
    b = {1: 0, 2: 0, 3: 1}
    paired = np.mean([a[k] - b[k] for k in a])
    assert paired == 1 / 3 and paired == np.mean(list(a.values())) - np.mean(list(b.values()))
    a2, b2 = {1: 1, 2: 0}, {2: 0, 3: 1}      # ключи пересекаются частично
    common = sorted(set(a2) & set(b2))
    assert common == [2], "непарные эпизоды должны выпадать, а не усредняться"

    print("самопроверка пройдена: Макнемар точный, кластерный бутстрап шире "
          f"наивного ({w_cl:.3f} против {w_nv:.3f}), разность парная")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--glob", default="data/k6h/*.json")
    ap.add_argument("--margin", type=float, default=10.0,
                    help="граница не-хуже-чем, в пунктах успеха")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-hash-mismatch", action="store_true",
                    help="НЕ используйте: расхождение хешей означает, что "
                         "эпизоды стартовали из разных состояний")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return

    files = sorted(globmod.glob(args.glob))
    if not files:
        raise SystemExit(f"нет файлов по {args.glob}")
    cells = defaultdict(dict)          # (suite,task,ens,H,init_id) -> {levels: ep}
    shas, ckpts = set(), set()
    for f in files:
        d = json.load(open(f))
        shas.add(d.get("script_sha1", "?")); ckpts.add(d.get("ckpt", "?"))
        for e in d["episodes"]:
            key = (d["suite"], d["task_id"], d.get("ensemble", "?"),
                   d["horizon"], e["init_state_id"])
            lv = d["levels"]
            if lv in cells[key]:
                raise SystemExit(
                    f"дубль ячейки {key} уровень {lv}: два файла описывают один "
                    f"эпизод. Проверьте, не запущен ли один блок дважды.")
            cells[key][lv] = e
    if len(shas) > 1:
        print(f"  ВНИМАНИЕ: файлы получены РАЗНЫМИ версиями скрипта: {shas}")
    if len(ckpts) > 1:
        raise SystemExit(f"разные чекпойнты в одном сравнении: {ckpts}")

    print(f"  файлов {len(files)}, ячеек {len(cells)}")
    res = {}
    for ens in sorted({k[2] for k in cells}):
      for H in sorted({k[3] for k in cells if k[2] == ens}):
        keys = [k for k in cells if k[2] == ens and k[3] == H
                and 1 in cells[k] and 3 in cells[k]]
        unpaired = [k for k in cells if k[2] == ens and k[3] == H
                    and len(cells[k]) < 2]
        if not keys:
            continue
        bad = [k for k in keys
               if cells[k][1].get("init_hash") != cells[k][3].get("init_hash")]
        if bad and not args.allow_hash_mismatch:
            raise SystemExit(
                f"ens={ens} H={H}: у {len(bad)} из {len(keys)} пар РАЗНЫЕ хеши "
                f"начального наблюдения, например {bad[0]}.\nЭто значит, что "
                f"эпизоды с одним init_state_id стартовали из разных состояний "
                f"и парное сравнение недействительно.")

        by_task = defaultdict(list)
        b = c = 0
        for k in keys:
            s1 = int(cells[k][1]["success"]); s3 = int(cells[k][3]["success"])
            by_task[k[1]].append(s1 - s3)
            b += (s1 == 1 and s3 == 0); c += (s1 == 0 and s3 == 1)
        by_task = {t: np.asarray(v, float) for t, v in by_task.items()}
        flat = np.concatenate(list(by_task.values()))
        micro = flat.mean() * 100
        macro = float(np.mean([v.mean() for v in by_task.values()])) * 100
        boot = cluster_bootstrap(by_task, args.n_boot, args.seed) * 100
        lo1 = float(np.percentile(boot, 5))
        lo2, hi2 = (float(np.percentile(boot, 2.5)),
                    float(np.percentile(boot, 97.5)))
        p = mcnemar_exact(b, c)
        r1 = np.mean([int(cells[k][1]["success"]) for k in keys]) * 100
        r3 = np.mean([int(cells[k][3]["success"]) for k in keys]) * 100

        tag = f"ens={ens}, H={H}"
        print(f"\n{'=' * 68}\n  {tag}: {len(keys)} пар, "
              f"{len(by_task)} задач" + (f", НЕПАРНЫХ {len(unpaired)}"
                                         if unpaired else ""))
        print(f"    успех levels=1: {r1:5.1f}%     levels=3: {r3:5.1f}%")
        print(f"    парная разность (1 минус 3): micro {micro:+.1f} пп, "
              f"macro {macro:+.1f} пп")
        print(f"    дискордантных пар: 1 лучше {b}, 3 лучше {c}; "
              f"Макнемар p = {p:.3f}")
        print(f"    кластерный бутстрап по задачам: "
              f"95% ДИ [{lo2:+.1f}, {hi2:+.1f}] пп")
        print(f"    односторонняя нижняя граница (5%): {lo1:+.1f} пп")
        verdict = ("НЕ ХУЖЕ" if lo1 > -args.margin else "ХУЖЕ или неясно")
        print(f"    при границе {args.margin:.0f} пп: {verdict}")
        if lo1 <= -args.margin and micro > -args.margin:
            print(f"    (точечная оценка выше границы, но выборки не хватает — "
                  f"это НЕ подтверждение)")
        res[tag] = dict(n_pairs=len(keys), n_tasks=len(by_task),
                        n_unpaired=len(unpaired), rate_l1=r1, rate_l3=r3,
                        micro=micro, macro=macro, disc_1=b, disc_3=c,
                        mcnemar_p=p, ci95=[lo2, hi2], lower_1s=lo1,
                        margin=args.margin, non_inferior=bool(lo1 > -args.margin))

    if not res:
        raise SystemExit("не нашлось ни одной пары levels=1 / levels=3")
    print(f"\n  ЧИТАТЬ по односторонней границе, правило записано в докстроке")
    print(f"  k6h_coarse_gate.py ДО запуска. «ДИ пересекает ноль» — не вывод.")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        json.dump(dict(cells=res, files=files, script_shas=sorted(shas)),
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"  сохранено: {args.out}")


if __name__ == "__main__":
    main()
