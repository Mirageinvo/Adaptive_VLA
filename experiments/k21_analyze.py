#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: анализ пилота (fullbar, flat24, tree_adapt24; старты 25–29,
50 парных состояний — разработочный банк).

Только проверенные блоки; неполный набор — код 3 без вывода. Для каждой
политики: SR с равными весами задач, число успехов, SR по задачам, вызовы.
Сравнения (парные: выигрыши/потери, точный Мак-Немар, 95% бутстрэп внутри
задач): tree − flat (архитектура), tree − fullbar, flat − fullbar
(инженерный вопрос). Для разности с fullbar дополнительно консервативная
нижняя граница (Clopper–Pearson 97.5% для вероятностей выигрыша и потери,
Bonferroni) — описательно; 50 состояний не позволяют доказать маржу −3 п.п.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BOOT_N, BOOT_SEED = 10000, 20261021
COMPARE = (("tree_adapt24", "flat24"), ("tree_adapt24", "fullbar"),
           ("flat24", "fullbar"))


def cp_bounds(k, n, a=0.025):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def ni_lower(gain, loss, n):
    """Консервативная нижняя граница P(выигрыш) − P(потеря)."""
    lo_g, _ = cp_bounds(gain, n)
    _, hi_l = cp_bounds(loss, n)
    return lo_g - hi_l


def summarize(outc, extra, states):
    import k15h_analyze as ha
    rows = {}
    for lab, o in outc.items():
        pt = {}
        for k in states:
            v = pt.setdefault(str(k[0]), [0, 0])
            v[0] += bool(o[k])
            v[1] += 1
        rows[lab] = dict(sr_equal_weight=ha.equal_weight(o, states),
                         successes=int(sum(o[k] for k in states)),
                         per_task=pt, **extra.get(lab, {}))
    comp = []
    for a, b in COMPARE:
        c = dict(a=a, b=b, **ha.compare(outc[a], outc[b], states, BOOT_N,
                                        BOOT_SEED))
        if b == "fullbar":
            c["ni_lower_conservative"] = ni_lower(c["gain"], c["loss"],
                                                  len(states))
        comp.append(c)
    return dict(states=len(states), rows=rows, comparisons=comp)


def run(plan_path, run_dir):
    import k16_protocol as p16
    import k21_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan)
    if prob:
        raise SystemExit("входы плана K-21 не совпадают: " + "; ".join(
            prob[:4]))
    ok, gaps = pc.check_run(plan, plan_path, run_dir)
    want = len(plan["labels"]) * len(plan["tasks"]) * len(plan["blocks"])
    res = dict(plan=plan_path, run_dir=run_dir, verified=len(ok),
               planned=want, gaps=gaps,
               complete=len(ok) == want and not gaps)
    if not res["complete"]:
        return res
    outc = {lab: {} for lab in plan["labels"]}
    extra = {lab: dict(calls=0, env_steps=0) for lab in plan["labels"]}
    for (lab, t, b), pth in ok.items():
        d, z = p16.load(pth)
        for e in d["episodes"]:
            k = (t, int(e["init_state_id"]))
            if k in outc[lab]:
                raise SystemExit(f"повтор состояния {k} у {lab}")
            outc[lab][k] = bool(e["success"])
            extra[lab]["env_steps"] += int(e.get("own_steps", 0))
        extra[lab]["calls"] += int(d["episodes"][0].get("policy_calls", 0))
    states = sorted(outc["fullbar"])
    if any(sorted(o) != states for o in outc.values()) or \
            len(states) != 50:
        raise SystemExit("наборы состояний политик различаются или их не 50")
    res.update(summarize(outc, extra, states))
    res["note"] = plan.get("note")
    return res


def report(res):
    print(f"  проверенных блоков {res['verified']}/{res['planned']}; "
          f"непригодных или отсутствующих {len(res['gaps'])}")
    for g in [g for g in res["gaps"] if g["why"] != "не готов"][:10]:
        print(f"    ВНИМАНИЕ {g['label']} t{g['task']} i{g['block']}: "
              f"{g['why']}")
    if not res["complete"]:
        print("  НАБОР НЕПОЛНЫЙ: вывода нет")
        return
    for lab, r in res["rows"].items():
        pt = " ".join(f"t{t} {v[0]}/{v[1]}" for t, v in r["per_task"].items())
        print(f"    {lab:13} SR {100 * r['sr_equal_weight']:5.1f}% "
              f"({r['successes']}/50), вызовов {r['calls']}; {pt}")
    for c in res["comparisons"]:
        s = (f"  {c['a']} − {c['b']}: ΔSR {100 * c['d_sr_equal_weight']:+.1f}"
             f" п.п. [{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
             f"+{c['gain']}/−{c['loss']}, Мак-Немар p {c['mcnemar_p']:.4f}")
        if "ni_lower_conservative" in c:
            s += (f"; консервативная нижняя граница Δ "
                  f"{100 * c['ni_lower_conservative']:+.1f} п.п.")
        print(s)


def selftest():
    states = [(t, s) for t in range(10) for s in range(25, 30)]
    rng = np.random.default_rng(0)
    base = {k: bool(rng.random() < 0.85) for k in states}
    outc = {lab: dict(base) for lab in ("fullbar", "flat24", "tree_adapt24")}
    for k in [k for k in states if not base[k]][:3]:
        outc["tree_adapt24"][k] = True
    r = summarize(outc, {}, states)
    c = r["comparisons"][0]
    assert c["gain"] == 3 and c["loss"] == 0
    assert -1 <= r["comparisons"][1]["ni_lower_conservative"] <= 0.06
    assert ni_lower(0, 0, 50) < -0.05          # ноль расхождений ≠ равенство
    print("самопроверка k21_analyze пройдена: сравнения, консервативная "
          "нижняя граница (ноль расхождений не доказывает равенства)")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-21: анализ пилота")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", default="reports/k21/plan_pilot.json")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default="reports/k21/pilot_analysis.json")
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
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    print(f"  сводка: {a.out}")
    return 0 if res["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
