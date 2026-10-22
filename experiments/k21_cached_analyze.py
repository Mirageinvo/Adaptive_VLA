#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fullbar_cached: анализ SR самостоятельного эталона на 150 парных
состояниях 30–44 и дополнительный инженерный контроль.

fullbar_cached — из своего плана (30 блоков); fullbar, flat24, tree_adapt24
— из плана A K-21 (90 блоков, проверяются заново); parents24 — из плана K-22,
если он полностью проверен (иначе не входит, с отметкой). Неполный набор
плеча или плана A — код 3 без вывода.

Вопросы (процедуры K-21):
  описательно — fullbar_cached − fullbar: SR эталона с кэшем относительно
    штатного (эквивалентность штатному пути не подтверждена);
  дополнительный инженерный контроль — не хуже ли tree_adapt24, flat24 и
    parents24 эталона fullbar_cached с маржой 3 п.п. (консервативная нижняя
    граница Clopper–Pearson, Bonferroni).
Основные сравнения K-21 и K-22 этим анализом не меняются.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BOOT_N, BOOT_SEED = 10000, 20261024
MARGIN = 0.03
REF = "fullbar_cached"
CONTROLS = ("tree_adapt24", "flat24", "parents24")


def collect():
    import k21_cached_protocol as pc
    import k22_analyze as k22a
    plan = json.load(open(pc.PLAN))
    prob = pc.check_plan_inputs(plan)
    if prob:
        raise SystemExit("план fullbar_cached: " + "; ".join(prob[:3]))
    run = os.path.join("reports/k21/run_cached", f"p{pc.sha(pc.PLAN)}")
    ok, gaps = pc.check_run(plan, pc.PLAN, run)
    want = len(plan["labels"]) * len(plan["tasks"]) * len(plan["blocks"])
    res = dict(run_dir=run, verified=len(ok), planned=want, gaps=gaps,
               complete=len(ok) == want and not gaps,
               reference=plan["reference"])
    if not res["complete"]:
        return res, None, None
    kp = plan["k21_plan"]
    a, ok21 = pc.k21_results(kp["path"])
    if pc.sha(kp["path"]) != kp["sha12"]:
        raise SystemExit("план A K-21 изменён после плана fullbar_cached")
    for k in ("blocks", "tasks", "setup", "prep_id", "bar_revision",
              "n_envs"):
        if a[k] != plan[k]:
            raise SystemExit(f"план fullbar_cached: {k} ≠ плану A K-21")
    outc, extra = k22a.outcomes(ok)
    o21, e21 = k22a.outcomes(ok21)
    outc.update(o21)
    extra.update(e21)
    res["k21_plan_A"] = dict(sha12=kp["sha12"], verified=len(ok21))
    res["k22"] = "нет плана K-22"
    import k22_protocol as p22
    if os.path.exists(p22.PLAN):
        try:
            r22, o22, e22 = k22a.collect()
        except SystemExit as e:
            r22, o22 = dict(complete=False, why=str(e.code)), None
        if o22 is not None and r22["complete"]:
            pl22 = json.load(open(p22.PLAN))
            for k in ("blocks", "tasks", "setup", "prep_id", "bar_revision",
                      "n_envs"):
                if pl22[k] != plan[k]:
                    raise SystemExit(f"план K-22: {k} ≠ плану fullbar_cached")
            outc["parents24"] = o22["parents24"]
            extra["parents24"] = e22.get("parents24", {})
            res["k22"] = "parents24 включён (план K-22 полностью проверен)"
        else:
            res["k22"] = (f"план K-22 не полон ({r22.get('verified')}/"
                          f"{r22.get('planned')}): parents24 не включён")
    return res, outc, extra


def analyze(outc, extra):
    import k15h_analyze as ha
    import k21_analyze as pa
    states = sorted(outc[REF])
    if len(states) != 150 or any(sorted(o) != states for o in outc.values()):
        raise SystemExit("наборы состояний различаются или их не 150")
    rows = {lab: dict(sr_equal_weight=ha.equal_weight(o, states),
                      successes=int(sum(o[k] for k in states)),
                      **extra.get(lab, {})) for lab, o in outc.items()}

    def cmp(a, b):
        c = dict(a=a, b=b, **ha.compare(outc[a], outc[b], states, BOOT_N,
                                        BOOT_SEED))
        c["ni_lower_conservative"] = pa.ni_lower(c["gain"], c["loss"],
                                                 len(states))
        c["non_inferior_margin_3pp"] = c["ni_lower_conservative"] > -MARGIN
        return c
    out = dict(states=len(states), rows=rows,
               reference_vs_standard=cmp(REF, "fullbar"),
               controls=[cmp(lab, REF) for lab in CONTROLS if lab in outc])
    return out


def line(c):
    return (f"ΔSR {100 * c['d_sr_equal_weight']:+.1f} п.п. "
            f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
            f"+{c['gain']}/−{c['loss']}, Мак-Немар p {c['mcnemar_p']:.4f}")


def report(res, out):
    print(f"  план fullbar_cached: проверено {res['verified']}/"
          f"{res['planned']}; проблем "
          f"{len([g for g in res['gaps'] if g['why'] != 'не готов'])}"
          + ("" if res["complete"] else " — НЕПОЛНЫЙ"))
    print(f"  эталон: самостоятельный; {res['reference']['equivalence']}")
    if out is None:
        print("  НАБОР НЕПОЛНЫЙ: вывода нет")
        return
    print(f"  план A K-21 {res['k21_plan_A']['sha12']}: проверено "
          f"{res['k21_plan_A']['verified']} блоков; K-22: {res['k22']}")
    for lab, r in out["rows"].items():
        print(f"    {lab:15} SR {100 * r['sr_equal_weight']:5.1f}% "
              f"({r['successes']}/150)")
    print(f"  описательно: fullbar_cached − fullbar: "
          f"{line(out['reference_vs_standard'])}")
    for c in out["controls"]:
        print(f"  контроль: {c['a']} − fullbar_cached: {line(c)}; "
              f"нижняя граница {100 * c['ni_lower_conservative']:+.1f} п.п. "
              "-> " + ("не-хужесть (маржа 3 п.п.) УСТАНОВЛЕНА"
                       if c["non_inferior_margin_3pp"]
                       else "не-хужесть НЕ установлена"))


def selftest():
    rng = np.random.default_rng(0)
    states = [(t, s) for t in range(10) for s in range(30, 45)]
    base = {k: bool(rng.random() < .85) for k in states}
    outc = {lab: dict(base) for lab in ("fullbar", "flat24", "tree_adapt24",
                                        REF)}
    for k in [k for k in states if base[k]][:6]:
        outc[REF][k] = False
    r = analyze(outc, {})
    assert r["reference_vs_standard"]["loss"] == 6
    assert [c["a"] for c in r["controls"]] == ["tree_adapt24", "flat24"]
    assert all(c["gain"] == 6 for c in r["controls"])
    outc["parents24"] = dict(base)
    assert len(analyze(outc, {})["controls"]) == 3
    print("самопроверка k21_cached_analyze пройдена: эталон против штатного, "
          "контроли не-хужести, parents24 по наличию")
    return 0


def main():
    ap = argparse.ArgumentParser(description="fullbar_cached: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="reports/k21/cached_analysis.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        res, outc, extra = collect()
        if outc is None:
            report(res, None)
            return 3
        out = analyze(outc, extra)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    report(res, out)
    json.dump(dict(plan=res, **out), open(a.out, "w"), ensure_ascii=False,
              indent=1, default=str)
    print(f"  сводка: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
