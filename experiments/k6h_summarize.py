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

ПРАВИЛО ЧТЕНИЯ записано в докстроке соответствующего гейта ДО запуска, и оно
ТРЁХСТОРОННЕЕ: нижняя односторонняя граница выше -margin доказывает
не-худшесть; ВЕРХНЯЯ ниже -margin доказывает ухудшение более чем на margin;
между ними не доказано ничего. Прежняя версия печатала «ХУЖЕ или неясно» одной
строкой и тем склеивала второй исход с третьим.

ПОЛЕ РАЗДЕЛЕНИЯ РУК ОБОБЩЕНО. Изначально руки различались по `levels` (1 против
3). K-9d сравнивает Joint-12 с грубым выходом полной глубины и кладёт в файлы
поле `arm`. Статистика для обоих случаев одна и та же, поэтому обобщён только
ключ: --field/--test/--ref. Умолчания воспроизводят прежнее поведение
дословно, а `run_tag` вошёл в ключ ячейки, чтобы файлы K-6h и K-9d,
совпадающие по (suite, task, ens, H, init_id), не склеились в одну пару.

Запуск:
    python3 experiments/k6h_summarize.py --selftest
    python3 experiments/k6h_summarize.py --glob 'data/k6h/*.json' --margin 10
    python3 experiments/k6h_summarize.py --glob 'data/k9d/*.json' \\
        --field arm --test fast12 --ref coarse24 --margin 5
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
    # ПОЛЕ РАЗДЕЛЕНИЯ РУК. По умолчанию — levels 1 против 3, то есть K-6h без
    # изменений. K-9d кладёт в файлы поле arm, и та же статистика применяется к
    # паре fast12/coarse24. Обобщается ТОЛЬКО ключ; тесты, бутстрап и правило
    # чтения не трогаются, иначе пришлось бы заново подтверждать оценщик.
    ap.add_argument("--field", default="levels",
                    help="поле файла, различающее руки (levels или arm)")
    ap.add_argument("--test", default=None,
                    help="значение --field у ИСПЫТУЕМОЙ руки (по умолчанию 1)")
    ap.add_argument("--ref", default=None,
                    help="значение --field у ОПОРНОЙ руки (по умолчанию 3)")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-hash-mismatch", action="store_true",
                    help="НЕ используйте: расхождение хешей означает, что "
                         "эпизоды стартовали из разных состояний")
    # НЕПАРНЫЕ ЯЧЕЙКИ — ОТКАЗ, А НЕ СТРОЧКА В ОТЧЁТЕ. Ячейка пропадает, когда
    # процесс упал или не дошёл; падать чаще может именно испытуемая рука, и
    # тогда из выборки систематически исчезают её худшие эпизоды. Прежняя
    # версия печатала «НЕПАРНЫХ n» и всё равно выдавала вердикт.
    ap.add_argument("--allow-unpaired", action="store_true",
                    help="считать по неполной развёртке. Вердикт при этом "
                         "недействителен: потеря ячеек не случайна.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return

    if args.field == "levels":
        A = int(args.test) if args.test is not None else 1
        B = int(args.ref) if args.ref is not None else 3
    else:
        if args.test is None or args.ref is None:
            raise SystemExit(f"при --field {args.field} нужны --test и --ref")
        A, B = args.test, args.ref
    if A == B:
        raise SystemExit("--test и --ref обязаны различаться")

    files = sorted(globmod.glob(args.glob))
    if not files:
        raise SystemExit(f"нет файлов по {args.glob}")
    cells = defaultdict(dict)     # (tag,suite,task,ens,H,init_id) -> {рука: ep}
    shas, ckpts = set(), set()
    wsha = set()
    for f in files:
        d = json.load(open(f))
        shas.add(d.get("script_sha1", "?")); ckpts.add(d.get("ckpt", "?"))
        # SHA ВЕСОВ И SHA МОДУЛЯ ИНФЕРЕНСА. Путь к чекпойнту не удостоверяет
        # ничего: best_imitation.pt перезаписывается каждой лучшей эпохой.
        j = d.get("joint")
        if isinstance(j, dict):
            wsha.add((j.get("weights_sha1", "?"),
                      j.get("joint12_vla_sha1", "?")))
        if args.field not in d:
            raise SystemExit(
                f"в {f} нет поля «{args.field}» — файл получен другим "
                f"скриптом. Проверьте --glob и --field.")
        # run_tag В КЛЮЧЕ: ячейки K-6h и K-9d могут лежать рядом и совпадать по
        # (suite, task, ens, H, init_id). Без тега они молча склеились бы в
        # одну пару, и сравнивались бы эпизоды из разных экспериментов.
        for e in d["episodes"]:
            key = (d.get("run_tag", "k6h"), d["suite"], d["task_id"],
                   d.get("ensemble", "?"), d["horizon"], e["init_state_id"])
            lv = d[args.field]
            if lv in cells[key]:
                raise SystemExit(
                    f"дубль ячейки {key}, {args.field}={lv}: два файла "
                    f"описывают один эпизод. Проверьте, не запущен ли один "
                    f"блок дважды.")
            cells[key][lv] = e
    if len(shas) > 1:
        print(f"  ВНИМАНИЕ: файлы получены РАЗНЫМИ версиями скрипта: {shas}")
    if len(ckpts) > 1:
        raise SystemExit(f"разные чекпойнты в одном сравнении: {ckpts}")
    if len(wsha) > 1:
        raise SystemExit(
            f"в одном сравнении смешаны разные веса или разные версии "
            f"joint12_vla.py: {sorted(wsha)}.\nЭто значит, что часть ячеек "
            f"посчитана другой сетью. Развёртку надо вести в каталоге, "
            f"привязанном к sha весов.")
    if wsha:
        print(f"  Joint: веса sha {sorted(wsha)[0][0]}, "
              f"joint12_vla sha {sorted(wsha)[0][1]}")

    print(f"  файлов {len(files)}, ячеек {len(cells)}; "
          f"{args.field}: испытуемая {A}, опора {B}")
    res = {}
    for run in sorted({k[0] for k in cells}):
     for ens in sorted({k[3] for k in cells if k[0] == run}):
      for H in sorted({k[4] for k in cells
                       if k[0] == run and k[3] == ens}):
        sub = [k for k in cells if k[0] == run and k[3] == ens and k[4] == H]
        keys = [k for k in sub if A in cells[k] and B in cells[k]]
        unpaired = [k for k in sub if len(cells[k]) < 2]
        if not keys:
            continue
        # ПОЛНЫЙ ХЕШ ПРОВЕРЯЕТСЯ ТАМ, ГДЕ ОН ЕСТЬ У ОБЕИХ РУК. Он включает
        # камеру на запястье, в которую политика тоже смотрит; совпадение
        # только по agentview — более слабое условие, чем требуется.
        bad = [k for k in keys
               if cells[k][A].get("init_hash") != cells[k][B].get("init_hash")]
        badf = [k for k in keys
                if cells[k][A].get("init_hash_full") is not None
                and cells[k][B].get("init_hash_full") is not None
                and (cells[k][A]["init_hash_full"]
                     != cells[k][B]["init_hash_full"])]
        if (bad or badf) and not args.allow_hash_mismatch:
            raise SystemExit(
                f"{run} ens={ens} H={H}: у {len(bad)} из {len(keys)} пар "
                f"РАЗНЫЕ хеши начального наблюдения ({len(badf)} по полному "
                f"хешу с камерой запястья), например {(bad or badf)[0]}.\n"
                f"Это значит, что эпизоды с одним init_state_id стартовали из "
                f"разных состояний и парное сравнение недействительно.")
        n_full = sum(1 for k in keys
                     if cells[k][A].get("init_hash_full") is not None
                     and cells[k][B].get("init_hash_full") is not None)
        if unpaired and not args.allow_unpaired:
            raise SystemExit(
                f"{run} ens={ens} H={H}: {len(unpaired)} ячеек без пары при "
                f"{len(keys)} полных. Развёртка неполная, и вердикт по ней\n"
                f"недействителен: падать чаще может именно испытуемая рука, и "
                f"тогда из выборки систематически исчезают её худшие эпизоды.\n"
                f"Досчитайте недостающие ячейки или, понимая последствия, "
                f"--allow-unpaired.")

        by_task = defaultdict(list)
        b = c = 0
        for k in keys:
            s1 = int(cells[k][A]["success"]); s3 = int(cells[k][B]["success"])
            by_task[k[2]].append(s1 - s3)
            b += (s1 == 1 and s3 == 0); c += (s1 == 0 and s3 == 1)
        by_task = {t: np.asarray(v, float) for t, v in by_task.items()}
        flat = np.concatenate(list(by_task.values()))
        micro = flat.mean() * 100
        macro = float(np.mean([v.mean() for v in by_task.values()])) * 100
        boot = cluster_bootstrap(by_task, args.n_boot, args.seed) * 100
        lo1 = float(np.percentile(boot, 5))
        hi1 = float(np.percentile(boot, 95))
        lo2, hi2 = (float(np.percentile(boot, 2.5)),
                    float(np.percentile(boot, 97.5)))
        p = mcnemar_exact(b, c)
        r1 = np.mean([int(cells[k][A]["success"]) for k in keys]) * 100
        r3 = np.mean([int(cells[k][B]["success"]) for k in keys]) * 100

        tag = f"{run}, ens={ens}, H={H}"
        print(f"\n{'=' * 68}\n  {tag}: {len(keys)} пар, "
              f"{len(by_task)} задач" + (f", НЕПАРНЫХ {len(unpaired)}"
                                         if unpaired else ""))
        print(f"    успех {args.field}={A}: {r1:5.1f}%     "
              f"{args.field}={B}: {r3:5.1f}%")
        print(f"    парная разность ({A} минус {B}): micro {micro:+.1f} пп, "
              f"macro {macro:+.1f} пп")
        print(f"    дискордантных пар: {A} лучше {b}, {B} лучше {c}; "
              f"Макнемар p = {p:.3f}")
        print(f"    кластерный бутстрап по задачам: "
              f"95% ДИ [{lo2:+.1f}, {hi2:+.1f}] пп"
              + (f", полный хеш сверен у {n_full}" if n_full else ""))
        print(f"    односторонние границы (5%/95%): "
              f"нижняя {lo1:+.1f}, верхняя {hi1:+.1f} пп")
        # ТРИ ИСХОДА, А НЕ ДВА. Не-худшесть доказывает нижняя граница выше
        # -margin; ухудшение более чем на margin доказывает ВЕРХНЯЯ граница
        # ниже -margin. Между ними не доказано ничего, и читать «нижняя ниже
        # порога» как доказанное ухудшение — та же ошибка, что читать
        # неотвергнутую нулевую гипотезу как доказанное равенство.
        non_inf = bool(lo1 > -args.margin)
        inferior = bool(hi1 < -args.margin)
        verdict = ("НЕ ХУЖЕ (доказано)" if non_inf else
                   "ХУЖЕ более чем на границу (доказано)" if inferior else
                   "НЕ ДОКАЗАНО НИЧЕГО")
        print(f"    при границе {args.margin:.0f} пп: {verdict}")
        if not non_inf and not inferior:
            print(f"    ни одна из двух односторонних гипотез не подтверждена; "
                  f"точечная оценка {micro:+.1f} пп сама по себе НЕ вывод")
        res[tag] = dict(n_pairs=len(keys), n_tasks=len(by_task),
                        n_unpaired=len(unpaired), n_full_hash=n_full,
                        field=args.field, arm_test=str(A), arm_ref=str(B),
                        rate_l1=r1, rate_l3=r3,
                        micro=micro, macro=macro, disc_1=b, disc_3=c,
                        mcnemar_p=p, ci95=[lo2, hi2],
                        lower_1s=lo1, upper_1s=hi1,
                        margin=args.margin, non_inferior=non_inf,
                        inferior=inferior,
                        undetermined=bool(not non_inf and not inferior))

    if not res:
        raise SystemExit(
            f"не нашлось ни одной пары {args.field}={A} / {args.field}={B}")
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
