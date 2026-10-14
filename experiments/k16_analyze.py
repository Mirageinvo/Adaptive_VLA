#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16: анализ пилота (разработочный, 50 общих состояний).

Только проверенные блоки (k16_protocol.check_run); неполный набор — код 3
и без вывода. Единица — начальное состояние; SR по задачам и с равными
весами задач; парная ΔSR против q0, спасения и повреждения; исполненные
импульсы (вызов импульса достигнут до завершения среды), время головы,
доля срабатываний предела.

ЗАРАНЕЕ ЗАДАННЫЕ СРАВНЕНИЯ (Мак-Немар, Holm на 4, парный бутстрэп внутри
задач):
  goal_flat_hist_event − direct_hist_event      явная ближайшая цель;
  goal_band_hist_event − goal_flat_hist_event   разделение поправки;
  goal_band_hist_event − goal_band_now_event    история;
  goal_band_hist_event − goal_band_hist_time    событийный момент.
ОРИЕНТИР без новых раскаток: static_uniform K-15h на тех же состояниях
(q0 K-16 воспроизводит q0ref K-15h — проверено протоколом).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PRIMARY = (("goal_flat_hist_event", "direct_hist_event"),
           ("goal_band_hist_event", "goal_flat_hist_event"),
           ("goal_band_hist_event", "goal_band_now_event"),
           ("goal_band_hist_event", "goal_band_hist_time"))
BOOT_N, BOOT_SEED = 10000, 20261014


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
    prim = []
    for a_, b_ in PRIMARY:
        prim.append(dict(a=a_, b=b_, **ha.compare(outc[a_], outc[b_], states,
                                                  BOOT_N, BOOT_SEED)))
    for c, p_ in zip(prim, ha.holm([c["mcnemar_p"] for c in prim])):
        c["holm_p"] = p_
    vs_q0 = {lab: ha.compare(o, q0, states, BOOT_N, BOOT_SEED)
             for lab, o in outc.items() if lab != "q0"}
    return dict(states=len(states), rows=rows, primary=prim, vs_q0=vs_q0)


def run(plan_path, run_dir):
    import k16_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan)
    if prob:
        raise SystemExit("входы плана K-16 не совпадают: " + "; ".join(
            prob[:4]))
    ok, gaps = pc.check_run(plan, plan_path, run_dir)
    want = len(plan["tasks"]) * len(plan["labels"])
    res = dict(plan=plan_path, run_dir=run_dir, verified=len(ok),
               planned=want, gaps=gaps,
               complete=len(ok) == want and not gaps)
    if not res["complete"]:
        return res
    outc = {lab: {} for lab in plan["labels"]}
    extra = {lab: dict(impulses=0, impulses_executed=0, clipped=0,
                       head_ms=[], norm_applied=[]) for lab in plan["labels"]}
    for (lab, t), pth in ok.items():
        d, z = pc.load(pth)
        imp = np.asarray(z["k16_impulse"], bool)
        for b, e in enumerate(d["episodes"]):
            outc[lab][(t, int(e["init_state_id"]))] = bool(e["success"])
            c = np.nonzero(imp[:, b])[0]
            if len(c):
                extra[lab]["impulses"] += 1
                extra[lab]["impulses_executed"] += pc.reached(
                    int(z["done_step"][b]), int(c[0]) + 1)
                extra[lab]["clipped"] += bool(z["k16_clipped"][c[0], b])
                extra[lab]["norm_applied"].append(
                    float(z["k16_norm_applied"][c[0], b]))
        hm = np.asarray(z["k16_head_ms"])[imp.any(1)]
        extra[lab]["head_ms"] += hm.tolist()
    for lab, e in extra.items():
        e["head_ms_mean"] = float(np.mean(e.pop("head_ms"))) if \
            e.get("head_ms") else None
        na = e.pop("norm_applied")
        e["norm_applied_median"] = float(np.median(na)) if na else None
    states = sorted(outc["q0"])
    res.update(summarize(outc, extra, states))
    # ориентир: static_uniform K-15h на тех же состояниях
    st = {}
    for t in plan["tasks"]:
        d = json.load(open(plan["refs"][f"static_t{t}"]))
        for e in d["episodes"]:
            st[(t, int(e["init_state_id"]))] = bool(e["success"])
    if set(st) == set(states):
        import k15h_analyze as ha
        res["reference_k15h_static_uniform"] = dict(
            sr_equal_weight=ha.equal_weight(st, states),
            vs_q0=ha.compare(st, outc["q0"], states, BOOT_N, BOOT_SEED),
            note="исторический ориентир без новых раскаток")
    return res


def report(res):
    print(f"  проверенных блоков {res['verified']}/{res['planned']}; "
          f"непригодных или отсутствующих {len(res['gaps'])}")
    for g in [g for g in res["gaps"] if g["why"] != "не готов"][:10]:
        print(f"    ВНИМАНИЕ {g['label']} t{g['task']}: {g['why']}")
    if not res["complete"]:
        print("  НАБОР НЕПОЛНЫЙ: вывода нет")
        return
    for lab, r in res["rows"].items():
        pt = " ".join(f"t{t} {v[0]}/{v[1]}" for t, v in r["per_task"].items())
        print(f"    {lab:22} SR {100 * r['sr_equal_weight']:5.1f}%, ΔSR "
              f"{100 * r['d_sr']:+5.1f} п.п., +{r['gain']}/−{r['loss']}, "
              f"импульсов {r['impulses']} (исполнено "
              f"{r['impulses_executed']}), предел {r['clipped']}, голова "
              f"{r['head_ms_mean'] or 0:.1f} мс; {pt}")
    print("  ЗАРАНЕЕ ЗАДАННЫЕ СРАВНЕНИЯ (Мак-Немар, Holm на 4; 95% "
          "бутстрэп):")
    for c in res["primary"]:
        print(f"    {c['a']} − {c['b']}: ΔSR "
              f"{100 * c['d_sr_equal_weight']:+.1f} п.п. "
              f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
              f"+{c['gain']}/−{c['loss']}, p {c['mcnemar_p']:.4f}, Holm "
              f"{c['holm_p']:.4f}")
    print("  против q0:")
    for k, c in res["vs_q0"].items():
        print(f"    {k}: ΔSR {100 * c['d_sr_equal_weight']:+.1f} п.п. "
              f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
              f"+{c['gain']}/−{c['loss']}, p {c['mcnemar_p']:.4f}")
    r = res.get("reference_k15h_static_uniform")
    if r:
        print(f"  ориентир static_uniform K-15h: SR "
              f"{100 * r['sr_equal_weight']:.1f}%, против q0 "
              f"+{r['vs_q0']['gain']}/−{r['vs_q0']['loss']}")


def selftest():
    states = [(t, s) for t in range(10) for s in range(25, 30)]
    rng = np.random.default_rng(0)
    q0 = {k: bool(rng.random() < 0.8) for k in states}
    outc = {lab: dict(q0) for lab in ("q0", "direct_hist_event",
                                      "goal_flat_hist_event",
                                      "goal_band_hist_event",
                                      "goal_band_now_event",
                                      "goal_band_hist_time")}
    fails = [k for k in states if not q0[k]]
    for k in fails[:3]:
        outc["goal_band_hist_event"][k] = True
    r = summarize(outc, {}, states)
    c = {(x["a"], x["b"]): x for x in r["primary"]}
    assert c[("goal_band_hist_event", "goal_flat_hist_event")]["gain"] == 3
    assert c[("goal_flat_hist_event", "direct_hist_event")]["gain"] == 0
    assert all(x["holm_p"] >= x["mcnemar_p"] for x in r["primary"])
    assert r["rows"]["goal_band_hist_event"]["gain"] == 3
    assert len(r["primary"]) == 4
    print("самопроверка k16_analyze пройдена: четыре сравнения, Holm, "
          "выигрыши против q0")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-16: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", default="reports/k16/plan.json")
    ap.add_argument("--run-dir", required=False)
    ap.add_argument("--out", default="reports/k16/analysis.json")
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
        json.dump(res, fh, ensure_ascii=False, indent=1,
                  default=lambda x: x.tolist() if hasattr(x, "tolist")
                  else str(x))
    print(f"  сводка: {a.out}")
    return 0 if res["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
