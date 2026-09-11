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

KEY = re.compile(r"m_s(\d+)_e(\d+)_lr([0-9.e-]+)\.json$")


def parse_name(path):
    m = KEY.search(os.path.basename(path))
    if not m:
        raise SystemExit(f"имя не разбирается: {path}. Ожидалось "
                         f"m_s<сид>_e<эпох>_lr<lr>.json")
    return int(m.group(1)), int(m.group(2)), m.group(3)


def agg(vals):
    """Медиана и размах по сидам. Одно число по одному сиду — не вывод."""
    a = np.asarray(vals, float)
    return dict(median=float(np.median(a)), lo=float(a.min()),
                hi=float(a.max()), n=int(a.size))


def collect(files):
    """Последний замер ряда post по каждому (эпох, lr, голова, сид)."""
    rows = {}
    for f in sorted(files):
        sd, ep, lr = parse_name(f)
        d = json.load(open(f))
        if d.get("train_log_std"):
            raise SystemExit(f"{f}: sigma обучалась, это другая абляция")
        for hd, v in d["heads"].items():
            post = v.get("post")
            if not post:
                raise SystemExit(f"{f}, {hd}: нет ряда post — файл от старой "
                                 f"версии, где замер шёл ДО обновления")
            rows.setdefault((ep, lr, hd), {})[sd] = dict(
                fin=post[-1], traj=[m["clip_frac"] for m in post],
                identity=v["identity"], steps=len(post))
    return rows


def table(rows, out=print):
    out(f"  {'эпох':>5}{'lr':>7}{'гол':>4}{'сидов':>6}"
        f"{'обрезано, медиана [размах]':>28}{'q99|log_r|':>12}"
        f"{'точный KL':>11}{'выравн.':>9}")
    best = []
    for (ep, lr, hd) in sorted(rows, key=lambda k: (k[0], float(k[1]), k[2])):
        per = rows[(ep, lr, hd)]
        cf = agg([p["fin"]["clip_frac"] for p in per.values()])
        q99 = agg([p["fin"]["log_ratio_q99"] for p in per.values()])
        kl = agg([p["fin"]["kl_exact"] for p in per.values()])
        al = agg([p["fin"]["align_mean"] for p in per.values()])
        out(f"  {ep:>5}{lr:>7}{hd:>4}{cf['n']:>6}"
            f"{100 * cf['median']:>19.1f}% "
            f"[{100 * cf['lo']:.0f}-{100 * cf['hi']:.0f}%]"
            f"{q99['median']:>12.3f}{kl['median']:>11.4g}"
            f"{al['median']:>9.3f}")
        best.append(((ep, lr, hd), cf, q99))
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
    regimes = sorted({(ep, lr) for (ep, lr, _) in ok},
                     key=lambda k: (k[0], float(k[1])))
    passing = [r for r in regimes
               if all(ok.get((r[0], r[1], hd), False) for hd in ("s0", "s1"))]
    return dict(per_arm=ok, passing=passing,
                chosen=(passing[0] if passing else None),
                thresholds=dict(max_clip=max_clip, max_q99=max_q99))


def selftest():
    assert parse_name("x/m_s3_e2_lr1e-5.json") == (3, 2, "1e-5")
    try:
        parse_name("bad.json")
    except SystemExit:
        pass
    else:
        raise AssertionError("чужое имя принято")
    a = agg([0.1, 0.2, 0.3])
    assert a == dict(median=0.2, lo=0.1, hi=0.3, n=3), a

    def mk(cf, q99):
        return dict(fin=dict(clip_frac=cf, log_ratio_q99=q99, kl_exact=0.01,
                             align_mean=0.05), traj=[cf], identity={},
                    steps=1)

    # ОБА СИДА И ОБЕ ГОЛОВЫ обязаны уложиться
    rows = {(1, "1e-5", hd): {s: mk(0.04, 0.2) for s in (0, 1)}
            for hd in ("s0", "s1")}
    g = read_guard([(k, agg([0.04, 0.04]), agg([0.2, 0.2])) for k in rows])
    assert g["chosen"] == (1, "1e-5"), g["chosen"]
    # ОДИН ПЛОХОЙ СИД ЛОМАЕТ РЕЖИМ: берётся размах, не медиана
    g2 = read_guard([((1, "1e-5", "s0"), agg([0.04, 0.40]), agg([0.2, 0.2])),
                     ((1, "1e-5", "s1"), agg([0.04, 0.04]), agg([0.2, 0.2]))])
    assert g2["chosen"] is None, g2
    # ХВОСТ ТОЖЕ ЛОМАЕТ, даже при малой доле обрезанных
    g3 = read_guard([((1, "1e-5", "s0"), agg([0.02]), agg([3.0])),
                     ((1, "1e-5", "s1"), agg([0.02]), agg([0.1]))])
    assert g3["chosen"] is None
    # ОТСУТСТВИЕ ГОЛОВЫ — не проходит
    g4 = read_guard([((1, "1e-5", "s0"), agg([0.02]), agg([0.1]))])
    assert g4["chosen"] is None
    # ФАЙЛ ОТ СТАРОЙ ВЕРСИИ ОТВЕРГАЕТСЯ
    import tempfile
    d = tempfile.mkdtemp()
    f = os.path.join(d, "m_s0_e1_lr1e-5.json")
    json.dump(dict(heads=dict(s0=dict(identity={}, steps=[]))), open(f, "w"))
    try:
        collect([f])
    except SystemExit as e:
        assert "нет ряда post" in str(e), str(e)
    else:
        raise AssertionError("файл без post принят")
    json.dump(dict(train_log_std=True,
                   heads=dict(s0=dict(identity={}, post=[mk(0.0, 0.0)["fin"]]))),
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
    for r in sorted({(ep, lr) for (ep, lr, _) in g["per_arm"]},
                    key=lambda k: (k[0], float(k[1]))):
        mark = "годен" if r in g["passing"] else "-"
        print(f"    эпох {r[0]}, lr {r[1]:<6} {mark}")
    print("\n  ТРАЕКТОРИЯ доли обрезанных по шагам (сид 0, голова s0):")
    for (ep, lr, hd), per in sorted(rows.items(),
                                    key=lambda kv: (kv[0][0],
                                                    float(kv[0][1]), kv[0][2])):
        if hd != "s0" or 0 not in per:
            continue
        print(f"    эпох {ep}, lr {lr:<6}: "
              + " ".join(f"{100*c:.0f}%" for c in per[0]["traj"]))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        json.dump(dict(guard=g, n_files=len(files),
                       cells={f"{k[0]}|{k[1]}|{k[2]}":
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
    print(f"\n  КАНДИДАТ: эпох {g['chosen'][0]}, lr {g['chosen'][1]}. Пороги "
          f"выбраны ПО ЭТОМУ ЖЕ перебору,\n  поэтому кандидатные: "
          f"регистрировать их можно только отдельным объявлением до "
          f"следующего\n  измерения.")


if __name__ == "__main__":
    main()
