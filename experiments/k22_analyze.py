#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-22: анализ parents24 на 150 парных состояниях 30–44.

parents24 — из плана K-22 (30 блоков); fullbar, flat24, tree_adapt24 — из
плана A K-21 (90 блоков, проверяются заново тем же check_run). Только
проверенные блоки; неполный набор — код 3 без вывода. Планы должны
совпадать по блокам, задачам, setup, prep_id и ревизии BAR.

ЗАРАНЕЕ ЗАДАННЫЕ ВОПРОСЫ (те же процедуры, что в K-21):
  архитектурное семейство (точный Мак-Немар, Holm по двум):
    parents24 − tree_adapt24 — даёт ли сохранение активных родителей
      что-то сверх дерева K-21;
    parents24 − flat24 — иерархия с родителями против плоской модели;
  инженерный — не хуже ли parents24 полного BAR с маржой 3 п.п.
    (консервативная нижняя граница Clopper–Pearson, Bonferroni).
Парная ΔSR и 95% бутстрэп внутри задач — для всех. Незначимость не
доказывает равенства.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BOOT_N, BOOT_SEED = 10000, 20261023
MARGIN = 0.03
FAMILY = (("parents24", "tree_adapt24"), ("parents24", "flat24"))


def outcomes(ok):
    import k16_protocol as p16
    outc, extra = {}, {}
    for (lab, t, b), f in ok.items():
        d, _z = p16.load(f)
        o = outc.setdefault(lab, {})
        x = extra.setdefault(lab, dict(calls=0, steps=0))
        for e in d["episodes"]:
            k = (int(t), int(e["init_state_id"]))
            if k in o:
                raise SystemExit(f"повтор состояния {k} у {lab}")
            o[k] = bool(e["success"])
            x["steps"] += int(e.get("own_steps", 0))
        x["calls"] += int(d["episodes"][0].get("policy_calls", 0))
    return outc, extra


def collect():
    import k22_protocol as pc
    plan = json.load(open(pc.PLAN))
    prob = pc.check_plan_inputs(plan)
    if prob:
        raise SystemExit("план K-22: " + "; ".join(prob[:3]))
    run = os.path.join("reports/k22/run", f"p{pc.sha(pc.PLAN)}")
    ok, gaps = pc.check_run(plan, pc.PLAN, run)
    want = len(plan["labels"]) * len(plan["tasks"]) * len(plan["blocks"])
    res = dict(run_dir=run, verified=len(ok), planned=want, gaps=gaps,
               complete=len(ok) == want and not gaps)
    if not res["complete"]:
        return res, None, None
    kp = plan["k21_plan"]
    a, ok21 = pc.k21_results(kp["path"])
    if pc.sha(kp["path"]) != kp["sha12"]:
        raise SystemExit("план A K-21 изменён после плана K-22")
    for k in ("blocks", "tasks", "setup", "prep_id", "bar_revision",
              "n_envs"):
        if a[k] != plan[k]:
            raise SystemExit(f"план K-22: {k} ≠ плану A K-21")
    outc, extra = outcomes(ok)
    o21, e21 = outcomes(ok21)
    outc.update(o21)
    extra.update(e21)
    res["k21_plan_A"] = dict(sha12=kp["sha12"], verified=len(ok21))
    return res, outc, extra


def analyze(outc, extra):
    import k15h_analyze as ha
    import k21_analyze as pa
    states = sorted(outc["fullbar"])
    if len(states) != 150 or any(sorted(o) != states for o in outc.values()):
        raise SystemExit("наборы состояний различаются или их не 150")
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

    def cmp(a, b):
        c = dict(a=a, b=b, **ha.compare(outc[a], outc[b], states, BOOT_N,
                                        BOOT_SEED))
        c["ni_lower_conservative"] = pa.ni_lower(c["gain"], c["loss"],
                                                 len(states))
        return c
    out = dict(states=len(states), rows=rows)
    eng = cmp("parents24", "fullbar")
    eng["non_inferior_margin_3pp"] = eng["ni_lower_conservative"] > -MARGIN
    out["engineering"] = eng
    fam = [cmp(a, b) for a, b in FAMILY]
    for c, h in zip(fam, ha.holm([c["mcnemar_p"] for c in fam])):
        c["holm_p"] = h
    out["superiority_family"] = fam
    return out


def report(res, out):
    print(f"  план K-22: проверено {res['verified']}/{res['planned']}; "
          f"проблем {len([g for g in res['gaps'] if g['why'] != 'не готов'])}"
          + ("" if res["complete"] else " — НЕПОЛНЫЙ"))
    for g in [g for g in res["gaps"] if g["why"] != "не готов"][:10]:
        print(f"    ВНИМАНИЕ {g}")
    if out is None:
        print("  НАБОР НЕПОЛНЫЙ: вывода нет")
        return
    print(f"  план A K-21 {res['k21_plan_A']['sha12']}: проверено "
          f"{res['k21_plan_A']['verified']} блоков (переиспользуются)")
    for lab, r in out["rows"].items():
        pt = " ".join(f"t{t} {v[0]}/{v[1]}" for t, v in r["per_task"].items())
        print(f"    {lab:13} SR {100 * r['sr_equal_weight']:5.1f}% "
              f"({r['successes']}/150); {pt}")
    c = out["engineering"]
    print(f"  инженерный: parents24 − fullbar: ΔSR "
          f"{100 * c['d_sr_equal_weight']:+.1f} п.п. "
          f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
          f"+{c['gain']}/−{c['loss']}, p {c['mcnemar_p']:.4f}; консервативная "
          f"нижняя граница {100 * c['ni_lower_conservative']:+.1f} п.п. -> "
          + ("не-хужесть (маржа 3 п.п.) УСТАНОВЛЕНА"
             if c["non_inferior_margin_3pp"] else "не-хужесть НЕ установлена"))
    for c in out["superiority_family"]:
        print(f"  {c['a']} − {c['b']}: ΔSR {100 * c['d_sr_equal_weight']:+.1f}"
              f" п.п. [{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}],"
              f" +{c['gain']}/−{c['loss']}, Мак-Немар p {c['mcnemar_p']:.4f},"
              f" Holm {c['holm_p']:.4f}")


def selftest():
    rng = np.random.default_rng(0)
    states = [(t, s) for t in range(10) for s in range(30, 45)]
    base = {k: bool(rng.random() < .85) for k in states}
    outc = {lab: dict(base) for lab in ("fullbar", "flat24", "tree_adapt24",
                                        "parents24")}
    for k in [k for k in states if not base[k]][:8]:
        outc["parents24"][k] = True
    r = analyze(outc, {})
    fam = r["superiority_family"]
    assert [(c["a"], c["b"]) for c in fam] == list(FAMILY)
    assert fam[0]["gain"] == 8 and fam[0]["loss"] == 0
    assert all(c["holm_p"] >= c["mcnemar_p"] for c in fam)
    assert abs(min(fam[0]["holm_p"], fam[1]["holm_p"]) -
               min(1.0, 2 * min(fam[0]["mcnemar_p"], fam[1]["mcnemar_p"]))) \
        < 1e-12
    assert r["engineering"]["non_inferior_margin_3pp"]
    bad = {k: dict(v) for k, v in outc.items()}
    bad["parents24"].pop(states[0])
    try:
        analyze(bad, {})
        raise AssertionError("неполный набор принят")
    except SystemExit:
        pass
    print("самопроверка k22_analyze пройдена: семейство из двух и Holm, "
          "граница не-хужести, отказ при неполном наборе")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-22: анализ parents24")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="reports/k22/main_analysis.json")
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
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(plan=res, **out), open(a.out, "w"), ensure_ascii=False,
              indent=1, default=str)
    print(f"  сводка: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
