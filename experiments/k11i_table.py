"""K-11i: сводная таблица перебора численного режима по сохранённым JSON.

ЗАЧЕМ ОТДЕЛЬНЫЙ СКРИПТ. Прежняя таблица считалась одноразовой командой, и
аудируемого артефакта для неё не было. Здесь агрегация по сидам — код, который
можно перечитать и прогнать заново на тех же файлах.

ЧТО АГРЕГИРУЕТСЯ. Из каждого прогона берётся ПОСЛЕДНИЙ замер ряда `post`, то
есть состояние политики ПОСЛЕ всех шагов, измеренное на ВСЁМ буфере. По сидам
печатается медиана и размах: выбирать предохранители по одному сиду нельзя.

Запуск:
    python3 experiments/k11i_table.py --selftest
    python3 experiments/k11i_table.py --glob 'data/k11i/m_s*.json'
"""

import argparse
import glob
import json
import os
import re

import numpy as np

def run_key(d, path):
    """Параметры режима из САМОГО JSON, а не из имени файла.

    Разбор имени был хрупким: стоило назвать файлы иначе, и скрипт отказывал,
    хотя все параметры лежат внутри. Имя теперь ни на что не влияет, а
    отсутствие поля — отказ, а не догадка.
    """
    miss = [k for k in ("epochs", "lr", "minibatch", "batch", "seed")
            if d.get(k) is None]
    if miss:
        raise SystemExit(f"{path}: в JSON нет полей {miss}")
    return (int(d["epochs"]), int(d["minibatch"]), float(d["lr"]),
            int(d["seed"]))


def agg(vals):
    """Медиана и размах по сидам. Одно число по одному сиду — не вывод."""
    a = np.asarray(vals, float)
    return dict(median=float(np.median(a)), lo=float(a.min()),
                hi=float(a.max()), n=int(a.size))


def collect(files):
    """Последний замер ряда post по каждому (эпох, lr, голова, сид)."""
    rows = {}
    for f in sorted(files):
        d = json.load(open(f))
        ep, mb, lr, sd = run_key(d, f)
        if d.get("train_log_std"):
            raise SystemExit(f"{f}: sigma обучалась, это другая абляция")
        for hd, v in d["heads"].items():
            post = v.get("post")
            if not post:
                raise SystemExit(f"{f}, {hd}: нет ряда post — файл от старой "
                                 f"версии, где замер шёл ДО обновления")
            rows.setdefault((ep, mb, lr, hd), {})[sd] = dict(
                fin=post[-1], traj=[m["clip_frac"] for m in post],
                identity=v["identity"], steps=len(post))
    return rows


def table(rows, out=print):
    out(f"  {'эпох':>5}{'мб':>5}{'lr':>8}{'гол':>4}{'сид':>4}"
        f"{'обрезано, медиана [размах]':>27}{'q99|log_r|':>11}"
        f"{'KL':>10}{'выравн.':>9}{'хвост':>8}")
    best = []
    for key in sorted(rows, key=lambda k: (k[0], k[1], k[2], k[3])):
        ep, mb, lr, hd = key
        per = rows[key]
        cf = agg([p["fin"]["clip_frac"] for p in per.values()])
        q99 = agg([p["fin"]["log_ratio_q99"] for p in per.values()])
        kl = agg([p["fin"]["kl_exact"] for p in per.values()])
        al = agg([p["fin"]["align_mean"] for p in per.values()])
        # ПОЛЕ МОЖЕТ ОТСУТСТВОВАТЬ у прогонов, сделанных до его введения.
        # nan в таблице читался бы как измеренная величина; печатаем прочерк.
        tps = [p["fin"]["align_top1pct"] for p in per.values()
               if p["fin"].get("align_top1pct") is not None]
        tp = agg(tps) if tps else None
        tp_s = "—" if tp is None else f"{tp['median']:.2f}"
        out(f"  {ep:>5}{mb:>5}{lr:>8.0e}{hd:>4}{cf['n']:>4}"
            f"{100 * cf['median']:>18.1f}% "
            f"[{100 * cf['lo']:.0f}-{100 * cf['hi']:.0f}%]"
            f"{q99['median']:>11.3f}{kl['median']:>10.4g}"
            f"{al['median']:>9.3f}{tp_s:>8}")
        best.append((key, cf, q99))
    return best


def read_guard(best, max_clip=0.10, max_q99=0.5):
    """Какие режимы уложились в КАНДИДАТНЫЕ предохранители на ВСЕХ сидах.

    Пороги здесь НЕ зарегистрированы: они выбираются по этому же перебору,
    поэтому называются кандидатными. Регистрировать их можно только отдельным
    объявлением до следующего измерения.
    """
    ok = {}
    for key, cf, q99 in best:
        ok[key] = bool(cf["hi"] <= max_clip and q99["hi"] <= max_q99)
    regimes = sorted({k[:3] for k in ok})
    passing = [r for r in regimes
               if all(ok.get(r + (hd,), False) for hd in ("s0", "s1"))]
    return dict(per_arm=ok, passing=passing,
                chosen=(passing[0] if passing else None),
                thresholds=dict(max_clip=max_clip, max_q99=max_q99))


def selftest():
    # ПАРАМЕТРЫ ИЗ JSON, имя файла ни на что не влияет.
    d0 = dict(epochs=2, lr=1e-5, minibatch=64, batch=256, seed=3)
    assert run_key(d0, "любое_имя.json") == (2, 64, 1e-5, 3)
    for gone in ("epochs", "lr", "minibatch", "batch", "seed"):
        try:
            run_key({k: v for k, v in d0.items() if k != gone}, "f")
        except SystemExit:
            pass
        else:
            raise AssertionError(f"отсутствие {gone} принято")
    a = agg([0.1, 0.2, 0.3])
    assert a == dict(median=0.2, lo=0.1, hi=0.3, n=3), a

    def mk(cf, q99):
        return dict(fin=dict(clip_frac=cf, log_ratio_q99=q99, kl_exact=0.01,
                             align_mean=0.05), traj=[cf], identity={},
                    steps=1)

    # ОБА СИДА И ОБЕ ГОЛОВЫ обязаны уложиться
    rows = {(1, 64, 1e-5, hd): {s: mk(0.04, 0.2) for s in (0, 1)}
            for hd in ("s0", "s1")}
    g = read_guard([(k, agg([0.04, 0.04]), agg([0.2, 0.2])) for k in rows])
    assert g["chosen"] == (1, 64, 1e-5), g["chosen"]
    # ОДИН ПЛОХОЙ СИД ЛОМАЕТ РЕЖИМ: берётся размах, не медиана
    g2 = read_guard([((1, 64, 1e-5, "s0"), agg([0.04, 0.40]),
                      agg([0.2, 0.2])),
                     ((1, 64, 1e-5, "s1"), agg([0.04, 0.04]),
                      agg([0.2, 0.2]))])
    assert g2["chosen"] is None, g2
    # ХВОСТ ТОЖЕ ЛОМАЕТ, даже при малой доле обрезанных
    g3 = read_guard([((1, 64, 1e-5, "s0"), agg([0.02]), agg([3.0])),
                     ((1, 64, 1e-5, "s1"), agg([0.02]), agg([0.1]))])
    assert g3["chosen"] is None
    # ОТСУТСТВИЕ ГОЛОВЫ — не проходит
    g4 = read_guard([((1, 64, 1e-5, "s0"), agg([0.02]), agg([0.1]))])
    assert g4["chosen"] is None
    # ФАЙЛ ОТ СТАРОЙ ВЕРСИИ ОТВЕРГАЕТСЯ
    import tempfile
    d = tempfile.mkdtemp()
    f = os.path.join(d, "любое.json")
    base = dict(epochs=1, lr=1e-5, minibatch=64, batch=256, seed=0)
    json.dump(dict(base, heads=dict(s0=dict(identity={}, steps=[]))),
              open(f, "w"))
    try:
        collect([f])
    except SystemExit as e:
        assert "нет ряда post" in str(e), str(e)
    else:
        raise AssertionError("файл без post принят")
    json.dump(dict(base, train_log_std=True,
                   heads=dict(s0=dict(identity={},
                                      post=[mk(0.0, 0.0)["fin"]]))),
              open(f, "w"))
    try:
        collect([f])
    except SystemExit as e:
        assert "другая абляция" in str(e), str(e)
    else:
        raise AssertionError("прогон с обученной sigma принят")
    print("самопроверка k11i_table пройдена: агрегация по сидам берёт РАЗМАХ, "
          "один плохой\n  сид ломает режим, хвост ломает наравне с долей "
          "обрезанных, файлы от старой\n  версии и с обученной sigma "
          "отвергаются")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--glob", default="data/k11i/m_s*.json")
    ap.add_argument("--max-clip", type=float, default=0.10)
    ap.add_argument("--max-q99", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit(f"нет файлов по {a.glob}")
    rows = collect(files)
    print(f"  файлов {len(files)}, ячеек (эпох, lr, голова) {len(rows)}")
    print("\n  ПОСЛЕ всех шагов, замер на ВСЁМ буфере, агрегация по сидам")
    best = table(rows)
    g = read_guard(best, a.max_clip, a.max_q99)
    print(f"\n  КАНДИДАТНЫЕ предохранители: обрезано <= {100*a.max_clip:.0f}% "
          f"и q99|log_ratio| <= {a.max_q99} НА ВСЕХ сидах и обеих головах")
    for r in sorted({k[:3] for k in g["per_arm"]}):
        mark = "годен" if r in g["passing"] else "-"
        print(f"    эпох {r[0]}, минибатч {r[1]}, lr {r[2]:.0e}  {mark}")
    print("\n  ТРАЕКТОРИЯ доли обрезанных по шагам (сид 0, голова s0):")
    for key, per in sorted(rows.items()):
        ep, mb, lr, hd = key
        if hd != "s0" or 0 not in per:
            continue
        print(f"    эпох {ep}, мб {mb}, lr {lr:.0e}: "
              + " ".join(f"{100*c:.0f}%" for c in per[0]["traj"]))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        # КЛЮЧИ-КОРТЕЖИ JSON НЕ ПРИНИМАЕТ: приводим к строкам. Таблица до
        # этого уже напечатана, поэтому падение теряло только файл.
        g_out = dict(g)
        g_out["per_arm"] = {f"e{k[0]}|mb{k[1]}|lr{k[2]:.0e}|{k[3]}": v
                            for k, v in g["per_arm"].items()}
        g_out["passing"] = [f"e{r[0]}|mb{r[1]}|lr{r[2]:.0e}"
                            for r in g["passing"]]
        c_ = g["chosen"]
        g_out["chosen"] = (None if c_ is None
                           else f"e{c_[0]}|mb{c_[1]}|lr{c_[2]:.0e}")
        json.dump(dict(guard=g_out, n_files=len(files),
                       cells={f"e{k[0]}|mb{k[1]}|lr{k[2]:.0e}|{k[3]}":
                              {str(s): v["fin"] for s, v in per.items()}
                              for k, per in rows.items()}),
                  open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {a.out}")
    if g["chosen"] is None:
        print("\n  НИ ОДИН РЕЖИМ не уложился в кандидатные предохранители на "
              "всех сидах.\n  Подбирать порог под измеренное нельзя: нужен "
              "либо меньший lr, либо другая\n  конструкция обновления "
              "(один полный шаг на свежем батче вместо PPO).")
        raise SystemExit(1)
    print(f"\n  КАНДИДАТ: эпох {g['chosen'][0]}, минибатч "
          f"{g['chosen'][1]}, lr {g['chosen'][2]:.0e}. Пороги "
          f"выбраны ПО ЭТОМУ ЖЕ перебору,\n  поэтому кандидатные: "
          f"регистрировать их можно только отдельным объявлением до "
          f"следующего\n  измерения.")


if __name__ == "__main__":
    main()
