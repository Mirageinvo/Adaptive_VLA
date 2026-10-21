#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: анализ расширенной оценки (старты 30–44, 150 парных состояний).

План A обязателен (fullbar, flat24, tree_adapt24), план B — если готов
(tree_fixed24). Только проверенные блоки; неполный набор — код 3 без
вывода. Планы должны совпадать по блокам, setup, prep_id и ревизии BAR.

ЗАРАНЕЕ ЗАДАННЫЕ ВОПРОСЫ:
  инженерный — не хуже ли tree_adapt24 полного BAR с маржой 3 п.п.:
    консервативная нижняя граница Δ (Clopper–Pearson 97.5% для вероятностей
    выигрыша и потери, Bonferroni) > −0.03;
  архитектурный — tree_adapt24 − flat24: парная ΔSR, точный Мак-Немар,
    95% бутстрэп внутри задач;
  с планом B — tree_fixed24 − flat24 (раскрытие иерархии) и
    tree_adapt24 − tree_fixed24 (адаптивное время).
Семейство превосходства (Мак-Немар) поправляется по Holm. Незначимость не
доказывает равенства; «не-хужесть не установлена», если граница ниже −0.03.
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
MARGIN = 0.03


def load_plans():
    import k21_main_protocol as mp
    plans = {}
    for name in ("A", "B"):
        p = mp.plan_path(name)
        if os.path.exists(p):
            plans[name] = (p, json.load(open(p)))
    if "A" not in plans:
        raise SystemExit("нет плана A")
    a = plans["A"][1]
    for name, (_p, pl) in plans.items():
        for k in ("blocks", "tasks", "setup", "prep_id", "bar_revision",
                  "n_envs"):
            if pl[k] != a[k]:
                raise SystemExit(f"план {name}: {k} ≠ плану A")
    return plans


def collect(plans):
    import k16_protocol as p16
    import k21_main_protocol as mp
    res, outc, extra = {}, {}, {}
    for name, (pth, plan) in plans.items():
        prob = mp.check_plan_inputs(plan)
        if prob:
            raise SystemExit(f"план {name}: " + "; ".join(prob[:3]))
        run = os.path.join("reports/k21/run", f"p{mp.sha(pth)}")
        ok, gaps = mp.check_run(plan, pth, run)
        want = len(plan["labels"]) * len(plan["tasks"]) * len(plan["blocks"])
        res[name] = dict(run_dir=run, verified=len(ok), planned=want,
                         gaps=gaps, complete=len(ok) == want and not gaps)
        if not res[name]["complete"]:
            continue
        for (lab, t, b), f in ok.items():
            d, _z = p16.load(f)
            o = outc.setdefault(lab, {})
            x = extra.setdefault(lab, dict(calls=0, steps=0))
            for e in d["episodes"]:
                k = (t, int(e["init_state_id"]))
                if k in o:
                    raise SystemExit(f"повтор состояния {k} у {lab}")
                o[k] = bool(e["success"])
                x["steps"] += int(e.get("own_steps", 0))
            x["calls"] += int(d["episodes"][0].get("policy_calls", 0))
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
    eng = cmp("tree_adapt24", "fullbar")
    eng["non_inferior_margin_3pp"] = eng["ni_lower_conservative"] > -MARGIN
    out["engineering"] = eng
    out["engineering_flat"] = cmp("flat24", "fullbar")
    out["engineering_flat"]["non_inferior_margin_3pp"] = \
        out["engineering_flat"]["ni_lower_conservative"] > -MARGIN
    # семейство Holm — ВСЕГДА три запланированных сравнения; недоступные
    # (без плана B) входят с p = 1, поэтому поправка не зависит от B
    fam = [cmp("tree_adapt24", "flat24")]
    for a_, b_ in (("tree_fixed24", "flat24"),
                   ("tree_adapt24", "tree_fixed24")):
        if "tree_fixed24" in outc:
            fam.append(cmp(a_, b_))
        else:
            fam.append(dict(a=a_, b=b_, mcnemar_p=1.0, available=False))
    if "tree_fixed24" in outc:
        out["fixed_vs_fullbar"] = cmp("tree_fixed24", "fullbar")
    for c, h in zip(fam, ha.holm([c["mcnemar_p"] for c in fam])):
        c["holm_p"] = h
    out["superiority_family"] = fam
    return out


def report(res, out):
    for name, r in res.items():
        print(f"  план {name}: проверено {r['verified']}/{r['planned']}; "
              f"проблем {len([g for g in r['gaps'] if g['why'] != 'не готов'])}"
              + ("" if r["complete"] else " — НЕПОЛНЫЙ"))
    if out is None:
        print("  НАБОР НЕПОЛНЫЙ: вывода нет")
        return
    for lab, r in out["rows"].items():
        pt = " ".join(f"t{t} {v[0]}/{v[1]}" for t, v in r["per_task"].items())
        print(f"    {lab:13} SR {100 * r['sr_equal_weight']:5.1f}% "
              f"({r['successes']}/150); {pt}")
    for key, title in (("engineering", "инженерный: tree_adapt24 − fullbar"),
                       ("engineering_flat", "flat24 − fullbar")):
        c = out[key]
        print(f"  {title}: ΔSR {100 * c['d_sr_equal_weight']:+.1f} п.п. "
              f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
              f"+{c['gain']}/−{c['loss']}, p {c['mcnemar_p']:.4f}; "
              f"консервативная нижняя граница "
              f"{100 * c['ni_lower_conservative']:+.1f} п.п. -> "
              + ("не-хужесть (маржа 3 п.п.) УСТАНОВЛЕНА"
                 if c["non_inferior_margin_3pp"] else
                 "не-хужесть НЕ установлена"))
    for c in out["superiority_family"]:
        if c.get("available") is False:
            print(f"  {c['a']} − {c['b']}: нет плана B (в Holm с p = 1)")
            continue
        print(f"  {c['a']} − {c['b']}: ΔSR {100 * c['d_sr_equal_weight']:+.1f}"
              f" п.п. [{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}],"
              f" +{c['gain']}/−{c['loss']}, Мак-Немар p {c['mcnemar_p']:.4f},"
              f" Holm {c['holm_p']:.4f}")


def selftest():
    import k21_analyze as pa
    # покрытие консервативной нижней границы (модель пар, n = 150)
    rng = np.random.default_rng(0)
    for pg, pl in ((0.08, 0.06), (0.03, 0.03), (0.0, 0.02), (0.12, 0.02)):
        cov = 0
        for _ in range(2000):
            u = rng.random(150)
            g = int((u < pg).sum())
            lo = int(((u >= pg) & (u < pg + pl)).sum())
            cov += pa.ni_lower(g, lo, 150) <= pg - pl + 1e-12
        assert cov / 2000 >= 0.95, (pg, pl, cov / 2000)
    states = [(t, s) for t in range(10) for s in range(30, 45)]
    base = {k: bool(rng.random() < .85) for k in states}
    outc = {lab: dict(base) for lab in ("fullbar", "flat24", "tree_adapt24",
                                        "tree_fixed24")}
    r = analyze(outc, {})
    # ноль расхождений на 150 парах: граница −(1 − 0.025^(1/150)) ≈ −0.024
    # (выше −0.03 — не-хужесть с этой маржой установлена); на 50 — нет
    e = r["engineering"]
    assert e["gain"] == 0 and abs(e["ni_lower_conservative"] + (
        1 - 0.025 ** (1 / 150))) < 1e-9 and e["non_inferior_margin_3pp"]
    assert pa.ni_lower(0, 0, 50) < -MARGIN
    assert len(r["superiority_family"]) == 3
    assert all(c["holm_p"] >= c["mcnemar_p"] for c in r["superiority_family"])
    # без плана B семейство всё равно из трёх: tree − flat получает Holm
    # с множителем 3 (два недоступных сравнения с p = 1)
    o2 = {k: dict(v) for k, v in outc.items() if k != "tree_fixed24"}
    for k in [k for k in states if not base[k]][:8]:
        o2["tree_adapt24"][k] = True
    f2 = analyze(o2, {})["superiority_family"]
    assert len(f2) == 3 and f2[1]["available"] is False
    assert abs(f2[0]["holm_p"] - min(1.0, 3 * f2[0]["mcnemar_p"])) < 1e-12
    print("самопроверка k21_main_analyze пройдена: покрытие консервативной "
          "границы ≥ 95% (симуляция), семейство и Holm, граница при нуле "
          "расхождений (150 и 50 пар)")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-21: анализ расширенной оценки")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="reports/k21/main_analysis.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        plans = load_plans()
        res, outc, extra = collect(plans)
        if not res["A"]["complete"]:
            report(res, None)
            return 3
        if "B" in res and not res["B"]["complete"]:
            outc.pop("tree_fixed24", None)
        out = analyze(outc, extra)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    report(res, out)
    json.dump(dict(plans=res, **out), open(a.out, "w"), ensure_ascii=False,
              indent=1, default=str)
    print(f"  сводка: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
