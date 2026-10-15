#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16u: анализ пилота (блок 40 всех задач, 50 парных состояний).

Только проверенные блоки; неполный набор — код 3 без вывода.
ЗАРАНЕЕ ЗАДАННЫЕ (Мак-Немар, Holm на 2, парный бутстрэп внутри задач):
  joint_goal − separate_goal  польза обратной связи от нижней головы;
  joint_goal − joint_ctx      польза явного представления цели при
                              совместном обучении.
ВТОРИЧНЫЕ: joint_goal − frozen_goal, separate_goal − frozen_goal, все
против q0. По каждой политике — SR по задачам и с равными весами,
вмешательства и недостигнутые события; задача 8 — описательно.
СЦЕПЛЕНИЕ БАТЧА: h18 импульсного вызова варианта против h18 q0 в том же
вызове, отдельно до первого вмешательства в батче и после соседа.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BOOT_N, BOOT_SEED = 10000, 20261016
PRIMARY = (("joint_goal_event", "separate_goal_event"),
           ("joint_goal_event", "joint_ctx_event"))
SECONDARY = (("joint_goal_event", "frozen_goal_event"),
             ("separate_goal_event", "frozen_goal_event"))


def summarize(outc, extra, states):
    import k15h_analyze as ha
    q0 = outc["q0"]
    rows = {}
    for lab, o in outc.items():
        pt = {}
        for k in states:
            v = pt.setdefault(str(k[0]), [0, 0])
            v[0] += bool(o[k])
            v[1] += 1
        rows[lab] = dict(sr_equal_weight=ha.equal_weight(o, states),
                         d_sr=ha.equal_weight(o, states)
                         - ha.equal_weight(q0, states),
                         gain=sum(1 for k in states if o[k] and not q0[k]),
                         loss=sum(1 for k in states if q0[k] and not o[k]),
                         per_task=pt, **extra.get(lab, {}))
    prim = [dict(a=a_, b=b_, **ha.compare(outc[a_], outc[b_], states, BOOT_N,
                                          BOOT_SEED)) for a_, b_ in PRIMARY]
    for c, p_ in zip(prim, ha.holm([c["mcnemar_p"] for c in prim])):
        c["holm_p"] = p_
    sec = [dict(a=a_, b=b_, **ha.compare(outc[a_], outc[b_], states, BOOT_N,
                                         BOOT_SEED)) for a_, b_ in SECONDARY]
    vs_q0 = {lab: ha.compare(o, q0, states, BOOT_N, BOOT_SEED)
             for lab, o in outc.items() if lab != "q0"}
    t8 = {lab: [sum(o[k] for k in states if k[0] == 8),
                sum(1 for k in states if k[0] == 8)]
          for lab, o in outc.items()}
    return dict(states=len(states), rows=rows, primary=prim, secondary=sec,
                vs_q0=vs_q0, task8_descriptive=t8)


def coupling(z, zq):
    """h18 импульсного вызова против q0: (до вмешательства в батче,
    после соседа) — [число импульсов, из них h18 отличается, max|Δ|]."""
    imp = np.asarray(z["k16_impulse"], bool)
    first_any = min(np.nonzero(imp.any(1))[0].tolist() or [10 ** 9])
    out = {"before": [0, 0, 0.0], "after": [0, 0, 0.0]}
    for k, b in zip(*np.nonzero(imp)):
        if k >= len(zq["k16_h18"]):
            continue
        d = float(np.abs(z["k16_h18"][k, b].astype(np.float32)
                         - zq["k16_h18"][k, b].astype(np.float32)).max())
        key = "after" if k > first_any else "before"
        out[key][0] += 1
        out[key][1] += int(d > 0)
        out[key][2] = max(out[key][2], d)
    return out


def run(plan_path, run_dir):
    import k16_protocol as p16
    import k16u_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan)
    if prob:
        raise SystemExit("входы плана K-16u не совпадают: " + "; ".join(
            prob[:4]))
    ok, gaps = pc.check_run(plan, plan_path, run_dir)
    want = len(plan["labels"]) * len(plan["tasks"]) * len(plan["blocks"])
    res = dict(plan=plan_path, run_dir=run_dir, verified=len(ok),
               planned=want, gaps=gaps,
               complete=len(ok) == want and not gaps)
    if not res["complete"]:
        return res
    outc = {lab: {} for lab in plan["labels"]}
    extra = {lab: dict(impulses=0, impulses_executed=0, no_event=0)
             for lab in plan["labels"]}
    coup = {lab: {"before": [0, 0, 0.0], "after": [0, 0, 0.0]}
            for lab in plan["labels"] if lab != "q0"}
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
            elif lab != "q0":
                extra[lab]["no_event"] += 1
        if lab != "q0":
            _dq, zq = p16.load(ok[("q0", t, b)])
            cc = coupling(z, zq)
            for key in ("before", "after"):
                coup[lab][key][0] += cc[key][0]
                coup[lab][key][1] += cc[key][1]
                coup[lab][key][2] = max(coup[lab][key][2], cc[key][2])
    res.update(summarize(outc, extra, sorted(outc["q0"])))
    res["batch_coupling_h18"] = coup
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
        print(f"    {lab:21} SR {100 * r['sr_equal_weight']:5.1f}%, ΔSR "
              f"{100 * r['d_sr']:+5.1f} п.п., +{r['gain']}/−{r['loss']}, "
              f"импульсов {r['impulses']} (исполнено "
              f"{r['impulses_executed']}, без события {r['no_event']}); {pt}")
    print("  ЗАРАНЕЕ ЗАДАННЫЕ (Мак-Немар, Holm на 2; 95% бутстрэп):")
    for c in res["primary"]:
        print(f"    {c['a']} − {c['b']}: ΔSR "
              f"{100 * c['d_sr_equal_weight']:+.1f} п.п. "
              f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
              f"+{c['gain']}/−{c['loss']}, p {c['mcnemar_p']:.4f}, Holm "
              f"{c['holm_p']:.4f}")
    print("  вторичные:")
    for c in res["secondary"]:
        print(f"    {c['a']} − {c['b']}: ΔSR "
              f"{100 * c['d_sr_equal_weight']:+.1f} п.п. "
              f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
              f"+{c['gain']}/−{c['loss']}, p {c['mcnemar_p']:.4f}")
    for k, c in res["vs_q0"].items():
        print(f"    {k} − q0: ΔSR {100 * c['d_sr_equal_weight']:+.1f} п.п. "
              f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
              f"+{c['gain']}/−{c['loss']}, p {c['mcnemar_p']:.4f}")
    print(f"  задача 8 (описательно): {res['task8_descriptive']}")
    print(f"  сцепление батча (h18 импульса против q0; [импульсов, "
          f"отличается, max|Δ|]): {res['batch_coupling_h18']}")


def selftest():
    states = [(t, s) for t in range(10) for s in range(40, 45)]
    rng = np.random.default_rng(0)
    q0 = {k: bool(rng.random() < 0.8) for k in states}
    labs = ("q0", "frozen_goal_event", "separate_goal_event",
            "joint_goal_event", "joint_ctx_event")
    outc = {lab: dict(q0) for lab in labs}
    for k in [k for k in states if not q0[k]][:4]:
        outc["joint_goal_event"][k] = True
    r = summarize(outc, {}, states)
    assert r["primary"][0]["gain"] == 4 and r["primary"][1]["gain"] == 4
    assert all(c["holm_p"] >= c["mcnemar_p"] for c in r["primary"])
    z = {"k16_impulse": np.zeros((5, 2), bool),
         "k16_h18": np.zeros((5, 2, 2, 2), np.float16)}
    zq = {"k16_h18": np.zeros((5, 2, 2, 2), np.float16)}
    z["k16_impulse"][1, 0] = True
    z["k16_impulse"][3, 1] = True
    z["k16_h18"][3, 1] = 0.5
    c = coupling(z, zq)
    assert c["before"] == [1, 0, 0.0] and c["after"][:2] == [1, 1]
    print("самопроверка k16u_analyze пройдена: два основных сравнения и "
          "Holm, сцепление батча до/после соседа")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-16u: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", default="reports/k16u/plan.json")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default="reports/k16u/analysis.json")
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
