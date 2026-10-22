#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: анализ оценки на LIBERO spatial, object, goal.

Только проверенные блоки; неполный набор — код 3 без вывода. По наборам и
в сумме по трём наборам (единица — состояние; равные веса задач внутри
объединения): SR политик, сравнения tree − flat, tree − fullbar,
flat − fullbar (выигрыши/потери, точный Мак-Немар, бутстрэп внутри задач;
с fullbar — консервативная нижняя граница не-хужести, маржа 3 п.п.).
Это расширение оценки на наборы, чьи задачи входят в обучение, а не
перенос на новые задачи.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BOOT_N, BOOT_SEED, MARGIN = 10000, 20261022, 0.03
PAIRS = (("tree_adapt24", "flat24"), ("tree_adapt24", "fullbar"),
         ("flat24", "fullbar"))


def summarize(outc, states):
    import k15h_analyze as ha
    import k21_analyze as pa
    rows = {}
    for lab, o in outc.items():
        pt = {}
        for k in states:
            v = pt.setdefault(str(k[0]), [0, 0])
            v[0] += bool(o[k])
            v[1] += 1
        rows[lab] = dict(sr_equal_weight=ha.equal_weight(o, states),
                         successes=int(sum(o[k] for k in states)),
                         n=len(states), per_task=pt)
    comp = []
    for a, b in PAIRS:
        c = dict(a=a, b=b, **ha.compare(outc[a], outc[b], states, BOOT_N,
                                        BOOT_SEED))
        if b == "fullbar":
            c["ni_lower_conservative"] = pa.ni_lower(c["gain"], c["loss"],
                                                     len(states))
            c["non_inferior_margin_3pp"] = c["ni_lower_conservative"] > \
                -MARGIN
        comp.append(c)
    return dict(rows=rows, comparisons=comp)


def run(plan_path, run_dir):
    import k16_protocol as p16
    import k21_suite_protocol as sp
    plan = json.load(open(plan_path))
    prob = sp.check_plan_inputs(plan)
    if prob:
        raise SystemExit("входы плана не совпадают: " + "; ".join(prob[:3]))
    ok, gaps = sp.check_run(plan, plan_path, run_dir)
    want = len(plan["labels"]) * len(plan["suites"]) * len(plan["tasks"]) \
        * len(plan["blocks"])
    res = dict(verified=len(ok), planned=want, gaps=gaps,
               complete=len(ok) == want and not gaps)
    if not res["complete"]:
        return res
    outc = {lab: {} for lab in plan["labels"]}
    for (lab, s, t, b), f in ok.items():
        d, _z = p16.load(f)
        for e in d["episodes"]:
            k = (f"{s}/{t}", int(e["init_state_id"]))
            if k in outc[lab]:
                raise SystemExit(f"повтор состояния {k} у {lab}")
            outc[lab][k] = bool(e["success"])
    states = sorted(outc["fullbar"])
    if any(sorted(o) != states for o in outc.values()):
        raise SystemExit("наборы состояний политик различаются")
    res["pooled"] = summarize(outc, states)
    res["by_suite"] = {}
    for s in plan["suites"]:
        st = [k for k in states if k[0].startswith(s + "/")]
        res["by_suite"][s] = summarize({lab: {k: o[k] for k in st}
                                        for lab, o in outc.items()}, st)
    res["note"] = plan.get("note")
    return res


def report(res):
    print(f"  проверенных блоков {res['verified']}/{res['planned']}; "
          f"непригодных или отсутствующих {len(res['gaps'])}")
    for g in [g for g in res["gaps"] if g["why"] != "не готов"][:10]:
        print(f"    ВНИМАНИЕ {g}")
    if not res["complete"]:
        print("  НАБОР НЕПОЛНЫЙ: вывода нет")
        return
    for name, blk in [("ВСЕ ТРИ НАБОРА", res["pooled"])] + \
            list(res["by_suite"].items()):
        print(f"  {name}:")
        for lab, r in blk["rows"].items():
            print(f"    {lab:13} SR {100 * r['sr_equal_weight']:5.1f}% "
                  f"({r['successes']}/{r['n']})")
        for c in blk["comparisons"]:
            s = (f"    {c['a']} − {c['b']}: ΔSR "
                 f"{100 * c['d_sr_equal_weight']:+.1f} п.п. "
                 f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
                 f"+{c['gain']}/−{c['loss']}, p {c['mcnemar_p']:.4f}")
            if "ni_lower_conservative" in c:
                s += (f"; нижняя граница {100 * c['ni_lower_conservative']:+.1f}"
                      f" п.п. ("
                      + ("не-хужесть установлена" if c[
                          "non_inferior_margin_3pp"] else "не установлена")
                      + ")")
            print(s)


def selftest():
    rng = np.random.default_rng(0)
    states = [(f"{s}/{t}", i) for s in ("spatial", "goal")
              for t in range(10) for i in range(5)]
    base = {k: bool(rng.random() < .8) for k in states}
    outc = {lab: dict(base) for lab in ("fullbar", "flat24", "tree_adapt24")}
    for k in [k for k in states if not base[k]][:5]:
        outc["tree_adapt24"][k] = True
    r = summarize(outc, states)
    assert r["comparisons"][0]["gain"] == 5 and r["rows"]["fullbar"]["n"] == 100
    print("самопроверка k21_suite_analyze пройдена")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-21: анализ наборов LIBERO")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", default="reports/k21/plan_suites.json")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default="reports/k21/suites_analysis.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        res = run(a.plan, a.run_dir)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    report(res)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1,
              default=str)
    print(f"  сводка: {a.out}")
    return 0 if res["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
