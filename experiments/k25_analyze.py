#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-25: анализ parents_cond24 по сидам 0 и 1.

Только проверенные блоки (новые — check_run плана K-25, источники — заново
проверенные планы с закреплёнными SHA). Состояние — (набор/задача,
init_state_id), env seed 101; бутстрэп N = 10000, seed из плана.
По каждому сиду S:
  ОСНОВНОЕ — parents_cond24_sS − parents_mr24_sS: ΔSR (равные веса
  задач), выигрыши/потери, точный Мак-Немар, 95% бутстрэп внутри задач;
  Holm по двум основным p;
  обязательные вторичные — cond − flat24_sS и cond − parents24_sS;
  инженерные — cond − fullbar и cond − fullbar_cached (граница
  не-хужести, маржа 3 п.п.); задача 8 — описательно.
Сводка двух сидов (без отбора): SR, среднее и ско каждой архитектуры; Δ по
сидам и средняя для cond − MR, cond − flat, cond − parents; совместный
парный бутстрэп средней (одни и те же состояния внутри задачи для всех
моделей и сидов — условно на двух обученных реализациях); seed×state —
ограниченно описательно. 2×150 не объединяются в 300.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k23_analyze as a23  # noqa: E402
import k24_analyze as a24  # noqa: E402

OUT = "reports/k25/analysis.json"
ARCHS = (("parents_cond24", "parents_cond24_s{}"),
         ("parents_mr24", "parents_mr24_s{}"),
         ("parents24", "parents24_s{}"), ("flat24", "flat24_s{}"))


def load_seed(seed):
    import k23_protocol as p23
    import k25_protocol as pc
    path = pc.plan_path(seed)
    plan = json.load(open(path, encoding="utf-8"))
    prob = pc.check_plan_inputs(plan)
    if prob:
        raise SystemExit(f"{path}: " + "; ".join(prob[:3]))
    ok, gaps = pc.check_run(plan, path, pc.run_dir(path))
    want = len(plan["cells"]) * len(plan["labels"])
    res = dict(plan=path, verified=len(ok), planned=want, gaps=gaps,
               complete=len(ok) == want and not gaps)
    if not res["complete"]:
        return plan, res, None
    refs = p23.verify_refs(plan)
    return plan, res, {**a23.outcomes(refs), **a23.outcomes(ok)}


def per_seed(seed, plan, outc):
    import k15h_analyze as ha
    c, mr, par, fl = (f.format(seed) for _a, f in ARCHS)
    labs = [c, mr, par, fl, "fullbar", "fullbar_cached"]
    states = a23.same_states(outc, labs, 150)
    boot = plan["bootstrap"]
    return dict(
        seed=seed, states=len(states),
        sr={k: ha.equal_weight(outc[k], states) for k in labs},
        per_task={k: a23.per_task(outc[k], states) for k in labs},
        primary=a23.cmp(outc, c, mr, states, boot),
        vs_flat=a23.cmp(outc, c, fl, states, boot),
        vs_parents=a23.cmp(outc, c, par, states, boot),
        vs_fullbar=a23.cmp(outc, c, "fullbar", states, boot, True),
        vs_cached=a23.cmp(outc, c, "fullbar_cached", states, boot, True),
        task8={k: a23.per_task(outc[k], states).get("10/8") for k in labs}), \
        states


def summary(per, outs, states, boot):
    import k15h_analyze as ha
    seeds = sorted(per)
    rows = {}
    for arch, lab in ARCHS:
        v = [per[s]["sr"][lab.format(s)] for s in seeds]
        rows[arch] = dict(per_seed=dict(zip(seeds, v)), n_seeds=len(v),
                          mean=float(np.mean(v)),
                          sd=float(np.std(v, ddof=1)) if len(v) > 1 else None)
    out = dict(seeds=seeds, rows=rows, holm_primary=dict(zip(
        seeds, ha.holm([per[s]["primary"]["mcnemar_p"] for s in seeds]))))
    for key, other in (("mr", "parents_mr24_s{}"), ("flat", "flat24_s{}"),
                       ("parents", "parents24_s{}")):
        cmpk = dict(mr="primary", flat="vs_flat", parents="vs_parents")[key]
        d = {s: per[s][cmpk]["d_sr_equal_weight"] for s in seeds}
        est, ci = a24.joint_boot(outs, lambda s: f"parents_cond24_s{s}",
                                 lambda s, o=other: o.format(s), states,
                                 boot["n"], boot["seed"])
        out[f"cond_minus_{key}"] = dict(per_seed=d, mean=float(np.mean(list(
            d.values()))), joint_paired_bootstrap=dict(d=est, ci95=ci))
    ss = a23.seeds_summary({s: dict(flat24=outs[s][f"parents_mr24_s{s}"],
                                    parents24=outs[s][f"parents_cond24_s{s}"])
                            for s in seeds}, states, boot,
                           [s for s in seeds if s != 0] or seeds)
    out["seed_state_bootstrap_limited_cond_minus_mr"] = dict(
        interval=ss["interval_seed_state_bootstrap"],
        note="ограниченная описательная оценка (2 сида)")
    return out


def report(res, per, summ):
    line = a23.line
    for s, r in res.items():
        print(f"  seed {s}: {r['plan']}: проверено {r['verified']}/"
              f"{r['planned']}" + ("" if r["complete"] else " — НЕПОЛНЫЙ"))
    for s, o in sorted(per.items()):
        print(f"\n  SEED {s} (150 состояний 30–44): " + ", ".join(
            f"{k} {100 * v:.1f}%" for k, v in o["sr"].items()))
        print(f"    ОСНОВНОЕ parents_cond24 − parents_mr24: "
              f"{line(o['primary'])}" + (f"; Holm {summ['holm_primary'][s]:.4f}"
                                         if summ else ""))
        print(f"    cond − flat24: {line(o['vs_flat'])}")
        print(f"    cond − parents24: {line(o['vs_parents'])}")
        print(f"    cond − fullbar: {line(o['vs_fullbar'])}")
        print(f"    cond − fullbar_cached (доп.): {line(o['vs_cached'])}")
        print("    задача 8 (описательно): " + ", ".join(
            f"{k} {v[0]}/{v[1]}" for k, v in o["task8"].items()))
    if summ:
        print("\n  СВОДКА ДВУХ СИДОВ (без отбора):")
        for a, r in summ["rows"].items():
            print(f"    {a:15} " + " ".join(f"s{s} {100 * v:.1f}%" for s, v in
                                           r["per_seed"].items())
                  + f" | среднее {100 * r['mean']:.1f}%, ско "
                  + (f"{100 * r['sd']:.1f}" if r["sd"] is not None else "—"))
        for key in ("mr", "flat", "parents"):
            c = summ[f"cond_minus_{key}"]
            jb = c["joint_paired_bootstrap"]
            print(f"    cond − {key}: " + ", ".join(
                f"s{s} {100 * v:+.1f}" for s, v in c["per_seed"].items())
                  + f"; среднее {100 * c['mean']:+.1f} п.п.; совместный "
                  f"бутстрэп [{100 * jb['ci95'][0]:+.1f}; "
                  f"{100 * jb['ci95'][1]:+.1f}]")


def selftest():
    rng = np.random.default_rng(0)
    states = [(f"10/{t}", i) for t in range(10) for i in range(30, 45)]
    base = {k: bool(rng.random() < .85) for k in states}
    outs, per = {}, {}
    for s in (0, 1):
        o = {f.format(s): dict(base) for _a, f in ARCHS}
        o["fullbar"], o["fullbar_cached"] = dict(base), dict(base)
        for k in [k for k in states if not base[k]][:3 + s]:
            o[f"parents_cond24_s{s}"][k] = True
        outs[s] = o
        plan = dict(bootstrap=dict(n=300, seed=1))
        per[s], st = per_seed(s, plan, o)
    sm = summary(per, outs, st, dict(n=300, seed=1))
    assert sm["cond_minus_mr"]["per_seed"][0] > 0 and \
        sm["cond_minus_mr"]["per_seed"][1] > sm["cond_minus_mr"]["per_seed"][0]
    jb = sm["cond_minus_mr"]["joint_paired_bootstrap"]
    assert abs(jb["d"] - sm["cond_minus_mr"]["mean"]) < 1e-12
    assert len(sm["holm_primary"]) == 2
    print("самопроверка k25_analyze пройдена: основное cond − MR, вторичные "
          "flat/parents, совместный бутстрэп = средняя Δ сидов, Holm на 2")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-25: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    res, per, outs, st, boot = {}, {}, {}, None, None
    try:
        for s in a.seeds:
            plan, r, outc = load_seed(s)
            res[s] = r
            if outc is None:
                continue
            per[s], st = per_seed(s, plan, outc)
            outs[s] = outc
            boot = plan["bootstrap"]
        summ = summary(per, outs, st, boot) if len(per) == len(a.seeds) \
            and len(per) > 1 else None
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    report(res, per, summ)
    os.makedirs("reports/k25", exist_ok=True)
    out = OUT if summ else \
        f"reports/k25/analysis_s{'_'.join(map(str, a.seeds))}.json"
    json.dump(dict(plans=res, per_seed=per, summary=summ), open(
        out, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
        default=str)
    print(f"  сводка: {out}")
    return 0 if all(r["complete"] for r in res.values()) else 3


if __name__ == "__main__":
    sys.exit(main())
