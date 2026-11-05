#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-32: анализ SR — три вопроса на задачах 0/6/8 (45 пар на сид):
  T − R  project_result;  T − P  conditional_tree_evidence;
  T − F  structured_space_evidence.
Holm-6 (3 сравнения × 2 сида), точный Мак-Немар, rescue/harm, ΔSR и
интервалы; совместный парный бутстрэп (одни init внутри задачи для обоих
сидов и рук, 10 000, seed 32005). Значимый отрицательный результат —
«ухудшение», а не «установлено». Вторичное: P − R, F − R, остальные 7 задач
(консервативная NI k21_analyze, маржа 3 п.п., CP 95%), все 10, по задачам,
обучение (обновления, KL, валидация). Код — только полнота/проверки.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k31_analyze as a31  # noqa: E402

OUT = "reports/k32/analysis.json"
LATENCY = "reports/k32/latency.json"
SEEDS = (0, 1)
KINDS = ("R", "T", "P", "F")
MAIN = (("T", "R", "project_result"), ("T", "P", "conditional_tree_evidence"),
        ("T", "F", "structured_space_evidence"))
SECOND = (("P", "R"), ("F", "R"))
BOOT = dict(n=10000, seed=32005)
PRIMARY, REST = a31.PRIMARY, a31.REST


def verdict(ds, sig):
    if all(sig):
        return ("улучшение" if all(d > 0 for d in ds) else
                "ухудшение" if all(d < 0 for d in ds) else
                "разнонаправлено") + " в обоих сидах (Holm-6)"
    if any(sig):
        i = sig.index(True)
        return ("улучшение" if ds[i] > 0 else "ухудшение") + \
            f" только в s{SEEDS[i]} (Holm-6)"
    return "не установлено"


def analyze(outc):
    st = lambda o, ts: a31.states_of(o, ts)              # noqa: E731
    res = dict(per_seed={}, main={}, joint={}, secondary={})
    ps = {}
    for s in SEEDS:
        prim = st(outc[("R", s)], PRIMARY)
        r = {}
        for a, b, *_ in MAIN + SECOND:
            r[f"{a}-{b}"] = dict(
                primary=a31.compare(outc[(a, s)], outc[(b, s)], prim, BOOT),
                rest7=a31.compare(outc[(a, s)], outc[(b, s)],
                                  st(outc[("R", s)], REST), BOOT),
                all10=a31.compare(outc[(a, s)], outc[(b, s)],
                                  sorted(outc[("R", s)]), BOOT))
        for a, b, _n in MAIN:
            ps[f"{a}-{b} s{s}"] = r[f"{a}-{b}"]["primary"]["mcnemar_p"]
        r["per_task"] = {k: {str(t): int(sum(v for x, v in outc[(k, s)]
                                             .items() if x[0] == t))
                             for t in range(10)} for k in KINDS}
        res["per_seed"][str(s)] = r
    res["holm6"] = a31.holm(ps)
    for a, b, name in MAIN:
        key = f"{a}-{b}"
        pairs = {s: (outc[(a, s)], outc[(b, s)]) for s in SEEDS}
        jb = a31.joint_boot(pairs, PRIMARY, BOOT)
        ds = [res["per_seed"][str(s)][key]["primary"]["d_sr_equal_weight"]
              for s in SEEDS]
        sig = [res["holm6"][f"{key} s{s}"] < 0.05 for s in SEEDS]
        res["main"][name] = dict(comparison=key, d_by_seed=ds, joint=jb,
                                 holm_significant=sig, verdict=verdict(ds,
                                                                       sig))
        res["joint"][key] = dict(primary=jb, rest7=a31.joint_boot(
            pairs, REST, BOOT))
    for a, b in SECOND:
        pairs = {s: (outc[(a, s)], outc[(b, s)]) for s in SEEDS}
        res["secondary"][f"{a}-{b}"] = a31.joint_boot(pairs, PRIMARY, BOOT)
    return res


def training_summary():
    out = {}
    for op in ("T", "P", "F"):
        for s in SEEDS:
            p = f"reports/k32/train_{op}_s{s}.json"
            if not os.path.exists(p):
                continue
            d = json.load(open(p, encoding="utf-8"))
            h = d.get("history") or []
            out[f"{op}_s{s}"] = dict(
                updates=d["updates"], accepted=d["accepted"],
                no_actor_update=d["no_actor_update"],
                train_steps=d["train_steps"],
                val={k: v["sr"] for k, v in d["val"].items()},
                kl_init_final=h[-1]["kl_init_final"] if h else None,
                route_share_last=h[-1].get("route_share") if h else None,
                cap_share_last=h[-1].get("cap_share") if h else None)
    return out


def line(c):
    return a31.line(c)


def report(res):
    for s in SEEDS:
        r = res["per_seed"][str(s)]
        print(f"\n  SEED {s}")
        for a, b, *_ in MAIN + SECOND:
            c = r[f"{a}-{b}"]
            tag = "" if (a, b) not in SECOND else " (вторично)"
            print(f"    {a}−{b}{tag}: 0/6/8 {line(c['primary'])}; SR "
                  f"{100 * c['primary']['sr_a']:.1f}% / "
                  f"{100 * c['primary']['sr_b']:.1f}%")
            print(f"      остальные 7: {line(c['rest7'])}; NI-нижняя "
                  f"{100 * c['rest7']['ni_lower_conservative']:+.1f} п.п.")
        print("    по задачам R/T/P/F: " + ", ".join(
            f"t{t} " + "/".join(str(r["per_task"][k][str(t)]) for k in KINDS)
            for t in range(10)))
    print("\n  Holm-6: " + ", ".join(f"{k} {v:.4f}" for k, v in
                                     res["holm6"].items()))
    for name, v in res["main"].items():
        j = v["joint"]
        print(f"  {name} ({v['comparison']}): Δ " + ", ".join(
            f"s{s} {100 * d:+.1f}" for s, d in zip(SEEDS, v["d_by_seed"])) +
            f"; среднее {100 * j['d']:+.1f} п.п. [{100 * j['ci95'][0]:+.1f}; "
            f"{100 * j['ci95'][1]:+.1f}]; {v['verdict']}")
    for k, v in res["secondary"].items():
        print(f"  {k} (вторично): среднее {100 * v['d']:+.1f} п.п. "
              f"[{100 * v['ci95'][0]:+.1f}; {100 * v['ci95'][1]:+.1f}]")
    for k, v in res.get("training", {}).items():
        print(f"  обучение {k}: обновлений {v['updates']} (принято "
              f"{v['accepted']}), шагов {v['train_steps']}, валидация "
              f"{v['val']}, KL к init {v['kl_init_final']}"
              + ("; stopped_no_actor_update" if v["no_actor_update"] else ""))
    print(f"  задержка: {res.get('latency', {}).get('status', 'ожидается')}")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    import k32_harness as kh
    import k32_protocol as pc
    try:
        outc = {}
        for s in SEEDS:
            for k in KINDS:
                plan = pc.load_eval_plan_kind(k, s)
                outc[(k, s)] = kh.outcomes(plan, kh.label(k, s))
    except SystemExit as e:
        print(f"  ОТКАЗ/НЕПОЛНО: {e.code}")
        return 3
    res = analyze(outc)
    res["training"] = training_summary()
    res["latency"] = dict(json.load(open(LATENCY, encoding="utf-8")),
                          status="выполнено") if os.path.exists(LATENCY) \
        else dict(status="ожидается")
    report(res)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=float)
    print(f"  сводка: {OUT}")
    return 0


def selftest():
    import numpy as np
    rng = np.random.default_rng(0)
    states = [(t, i) for t in range(10) for i in range(30, 45)]
    outc = {}
    for s in SEEDS:
        base = {k: bool(rng.random() < 0.85) for k in states}
        for k in KINDS:
            outc[(k, s)] = dict(base)
    r0 = analyze(outc)
    assert all(v["verdict"] == "не установлено" for v in r0["main"].values())
    for s in SEEDS:                      # T сильно хуже R на 0/6/8
        outc[("T", s)] = {k: (k[0] not in PRIMARY) for k in states}
        outc[("R", s)] = {k: True for k in states}
    r1 = analyze(outc)
    assert r1["main"]["project_result"]["verdict"].startswith("ухудшение"), \
        r1["main"]["project_result"]
    assert len(r1["holm6"]) == 6
    print("самопроверка k32_analyze пройдена: нулевой эффект — не "
          "установлено, значимое падение — «ухудшение» (не «установлено»), "
          "Holm-6 по шести p")
    return 0


if __name__ == "__main__":
    sys.exit(main())
