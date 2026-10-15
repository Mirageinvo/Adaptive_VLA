#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16b: ctx_goal против ctx (и каждая против q0) на 100 парных
состояниях (блоки 30 и 35 всех задач). Только проверенные блоки;
неполный набор — код 3. Заранее заданное сравнение: ctx_goal_event −
ctx_event (Мак-Немар, парный бутстрэп внутри задач). Явная цель
вычисляется из того же скрытого вектора — проверяется представление."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BOOT_N, BOOT_SEED = 10000, 20261015


def summarize(outc, extra, states):
    import k15h_analyze as ha
    q0 = outc["q0"]
    rows = {lab: dict(sr_equal_weight=ha.equal_weight(o, states),
                      d_sr=ha.equal_weight(o, states)
                      - ha.equal_weight(q0, states),
                      gain=sum(1 for k in states if o[k] and not q0[k]),
                      loss=sum(1 for k in states if q0[k] and not o[k]),
                      per_task={str(t): [sum(o[k] for k in states
                                             if k[0] == t),
                                         sum(1 for k in states
                                             if k[0] == t)]
                                for t in sorted({k[0] for k in states})},
                      **extra.get(lab, {}))
            for lab, o in outc.items()}
    primary = dict(a="ctx_goal_event", b="ctx_event",
                   **ha.compare(outc["ctx_goal_event"], outc["ctx_event"],
                                states, BOOT_N, BOOT_SEED))
    vs_q0 = {lab: ha.compare(o, q0, states, BOOT_N, BOOT_SEED)
             for lab, o in outc.items() if lab != "q0"}
    return dict(states=len(states), rows=rows, primary=primary, vs_q0=vs_q0)


def run(plan_path, run_dir):
    import k16_protocol as p16
    import k16b_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan)
    if prob:
        raise SystemExit("входы плана K-16b не совпадают: " + "; ".join(
            prob[:4]))
    ok, gaps = pc.check_run(plan, plan_path, run_dir)
    want = len(plan["labels"]) * len(plan["tasks"]) * len(plan["blocks"])
    res = dict(plan=plan_path, run_dir=run_dir, verified=len(ok),
               planned=want, gaps=gaps,
               complete=len(ok) == want and not gaps)
    if not res["complete"]:
        return res
    outc = {lab: {} for lab in plan["labels"]}
    extra = {lab: dict(impulses=0, impulses_executed=0) for lab in
             plan["labels"]}
    for (lab, t, b), pth in ok.items():
        d, z = p16.load(pth)
        imp = np.asarray(z["k16_impulse"], bool)
        for i, e in enumerate(d["episodes"]):
            outc[lab][(t, int(e["init_state_id"]))] = bool(e["success"])
            c = np.nonzero(imp[:, i])[0]
            if len(c):
                extra[lab]["impulses"] += 1
                extra[lab]["impulses_executed"] += p16.reached(
                    int(z["done_step"][i]), int(c[0]) + 1)
    res.update(summarize(outc, extra, sorted(outc["q0"])))
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
        print(f"    {lab:15} SR {100 * r['sr_equal_weight']:5.1f}%, ΔSR "
              f"{100 * r['d_sr']:+5.1f} п.п., +{r['gain']}/−{r['loss']}, "
              f"импульсов {r['impulses']} (исполнено "
              f"{r['impulses_executed']}); {pt}")
    c = res["primary"]
    print(f"  ЗАРАНЕЕ ЗАДАННОЕ: ctx_goal_event − ctx_event: ΔSR "
          f"{100 * c['d_sr_equal_weight']:+.1f} п.п. "
          f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
          f"+{c['gain']}/−{c['loss']}, p {c['mcnemar_p']:.4f}")
    for k, c in res["vs_q0"].items():
        print(f"    {k} − q0: ΔSR {100 * c['d_sr_equal_weight']:+.1f} п.п. "
              f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
              f"+{c['gain']}/−{c['loss']}, p {c['mcnemar_p']:.4f}")


def selftest():
    states = [(t, s) for t in range(10) for s in list(range(30, 40))]
    rng = np.random.default_rng(0)
    q0 = {k: bool(rng.random() < 0.8) for k in states}
    outc = {"q0": q0, "ctx_event": dict(q0), "ctx_goal_event": dict(q0)}
    for k in [k for k in states if not q0[k]][:4]:
        outc["ctx_goal_event"][k] = True
    r = summarize(outc, {}, states)
    assert r["primary"]["gain"] == 4 and r["states"] == 100
    assert r["rows"]["ctx_goal_event"]["gain"] == 4
    print("самопроверка k16b_analyze пройдена")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-16b: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", default="reports/k16b/plan.json")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default="reports/k16b/analysis.json")
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
