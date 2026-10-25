#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-24: анализ parents_mr24 против parents24 и flat24 по сидам 0 и 1.

Только проверенные блоки (новые — check_run плана K-24, источники —
заново проверенные планы с закреплёнными SHA). Состояние — (набор/задача,
init_state_id), env seed 101; бутстрэп N = 10000, seed = 20261025 (из плана).
По каждому сиду S:
  ОСНОВНОЕ — parents_mr24_sS − parents24_sS: ΔSR (равные веса задач),
  выигрыши/потери, точный Мак-Немар, 95% парный бутстрэп внутри задач;
  Holm по двум основным p (только они в семействе);
  вторичное — MR − flat24_sS; MR − fullbar и MR − fullbar_cached
  (консервативная граница не-хужести, маржа 3 п.п.); задача 8 — описательно.
Сводка сидов (все, без отбора): SR каждой архитектуры, среднее и выборочное
ско; Δ_s и средняя Δ; совместный парный бутстрэп средней Δ (одни и те же
состояния внутри задачи для всех моделей и сидов — интервал условен на этих
двух обученных реализациях); seed×state бутстрэп — ограниченно описательно.
2×150 не объединяются в 300; Мак-Немар к средним не применяется.
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

OUT = "reports/k24/analysis.json"


def load_seed(seed):
    import k23_protocol as p23
    import k24_protocol as pc
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
    outc = {**a23.outcomes(refs), **a23.outcomes(ok)}
    return plan, res, outc


def per_seed(seed, plan, outc):
    import k15h_analyze as ha
    mr, par, fl = f"parents_mr24_s{seed}", f"parents24_s{seed}", \
        f"flat24_s{seed}"
    labs = [mr, par, fl, "fullbar", "fullbar_cached"]
    states = a23.same_states(outc, labs, 150)
    boot = plan["bootstrap"]
    out = dict(seed=seed, states=len(states),
               sr={k: ha.equal_weight(outc[k], states) for k in labs},
               per_task={k: a23.per_task(outc[k], states) for k in labs},
               primary=a23.cmp(outc, mr, par, states, boot),
               vs_flat=a23.cmp(outc, mr, fl, states, boot),
               vs_fullbar=a23.cmp(outc, mr, "fullbar", states, boot, True),
               vs_cached=a23.cmp(outc, mr, "fullbar_cached", states, boot,
                                 True),
               task8={k: a23.per_task(outc[k], states).get("10/8")
                      for k in labs})
    return out, states


def joint_boot(outc_by_seed, a_of, b_of, states, n, seed):
    """Средняя по сидам ΔSR(a − b); одни и те же состояния внутри задачи
    пересэмплируются для всех моделей и сидов."""
    rng = np.random.default_rng(seed)
    by = {}
    for k in states:
        by.setdefault(k[0], []).append(k)
    tasks = sorted(by)
    seeds = sorted(outc_by_seed)
    D = {s: {t: np.asarray([float(outc_by_seed[s][a_of(s)][k]) -
                            float(outc_by_seed[s][b_of(s)][k]) for k in by[t]])
             for t in tasks} for s in seeds}
    est = float(np.mean([np.mean([D[s][t].mean() for t in tasks])
                         for s in seeds]))
    bs = np.empty(n)
    for i in range(n):
        idx = {t: rng.integers(0, len(by[t]), len(by[t])) for t in tasks}
        bs[i] = np.mean([np.mean([D[s][t][idx[t]].mean() for t in tasks])
                         for s in seeds])
    return est, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def summary(per, outs, states, boot):
    import k15h_analyze as ha
    seeds = sorted(per)
    rows = {}
    for arch, lab in (("parents_mr24", "parents_mr24_s{}"),
                      ("parents24", "parents24_s{}"), ("flat24", "flat24_s{}")):
        v = [per[s]["sr"][lab.format(s)] for s in seeds]
        rows[arch] = dict(per_seed=dict(zip(seeds, v)), n_seeds=len(v),
                          mean=float(np.mean(v)),
                          sd=float(np.std(v, ddof=1)) if len(v) > 1 else None)
    d = {s: per[s]["primary"]["d_sr_equal_weight"] for s in seeds}
    df = {s: per[s]["vs_flat"]["d_sr_equal_weight"] for s in seeds}
    holm = ha.holm([per[s]["primary"]["mcnemar_p"] for s in seeds])
    jb = joint_boot(outs, lambda s: f"parents_mr24_s{s}",
                    lambda s: f"parents24_s{s}", states, boot["n"],
                    boot["seed"])
    jf = joint_boot(outs, lambda s: f"parents_mr24_s{s}",
                    lambda s: f"flat24_s{s}", states, boot["n"], boot["seed"])
    ss = a23.seeds_summary({s: dict(flat24=outs[s][f"parents24_s{s}"],
                                    parents24=outs[s][f"parents_mr24_s{s}"])
                            for s in seeds}, states, boot,
                           [s for s in seeds if s != 0] or seeds)
    return dict(seeds=seeds, rows=rows, delta_mr_parents=d,
                delta_mr_parents_mean=float(np.mean(list(d.values()))),
                holm_primary=dict(zip(seeds, holm)),
                joint_paired_bootstrap_mr_parents=dict(
                    d=jb[0], ci95=jb[1], note="условно на двух обученных "
                    "реализациях; не популяционная вариативность сидов"),
                delta_mr_flat=df,
                delta_mr_flat_mean=float(np.mean(list(df.values()))),
                joint_paired_bootstrap_mr_flat=dict(d=jf[0], ci95=jf[1]),
                seed_state_bootstrap_limited=dict(
                    interval=ss["interval_seed_state_bootstrap"],
                    note="ограниченная описательная оценка (2 сида)"))


def line(c):
    return a23.line(c)


def report(res, per, summ):
    for s, r in res.items():
        print(f"  seed {s}: {r['plan']}: проверено {r['verified']}/"
              f"{r['planned']}" + ("" if r["complete"] else " — НЕПОЛНЫЙ"))
    for s, o in sorted(per.items()):
        print(f"\n  SEED {s} (150 состояний 30–44): " + ", ".join(
            f"{k} {100 * v:.1f}%" for k, v in o["sr"].items()))
        print(f"    ОСНОВНОЕ parents_mr24 − parents24: {line(o['primary'])}"
              + (f"; Holm {summ['holm_primary'][s]:.4f}" if summ else ""))
        print(f"    MR − flat24: {line(o['vs_flat'])}")
        print(f"    MR − fullbar: {line(o['vs_fullbar'])}")
        print(f"    MR − fullbar_cached (доп.): {line(o['vs_cached'])}")
        print("    задача 8 (описательно): " + ", ".join(
            f"{k} {v[0]}/{v[1]}" for k, v in o["task8"].items()))
    if summ:
        print("\n  СВОДКА ДВУХ СИДОВ (без отбора):")
        for a, r in summ["rows"].items():
            print(f"    {a:13} " + " ".join(f"s{s} {100 * v:.1f}%" for s, v in
                                         r["per_seed"].items())
                  + f" | среднее {100 * r['mean']:.1f}%, ско "
                  + (f"{100 * r['sd']:.1f}" if r["sd"] is not None else "—"))
        jb = summ["joint_paired_bootstrap_mr_parents"]
        print("    MR − parents по сидам: " + ", ".join(
            f"s{s} {100 * v:+.1f}" for s, v in summ["delta_mr_parents"].items())
              + f"; среднее {100 * summ['delta_mr_parents_mean']:+.1f} п.п.; "
              f"совместный бутстрэп [{100 * jb['ci95'][0]:+.1f}; "
              f"{100 * jb['ci95'][1]:+.1f}]")
        jf = summ["joint_paired_bootstrap_mr_flat"]
        print("    MR − flat по сидам: " + ", ".join(
            f"s{s} {100 * v:+.1f}" for s, v in summ["delta_mr_flat"].items())
              + f"; среднее {100 * summ['delta_mr_flat_mean']:+.1f} п.п.; "
              f"[{100 * jf['ci95'][0]:+.1f}; {100 * jf['ci95'][1]:+.1f}]")


def selftest():
    rng = np.random.default_rng(0)
    states = [(f"10/{t}", i) for t in range(10) for i in range(30, 45)]
    base = {k: bool(rng.random() < .85) for k in states}
    outs = {}
    for s in (0, 1):
        o = {f"parents24_s{s}": dict(base), f"flat24_s{s}": dict(base),
             f"parents_mr24_s{s}": dict(base)}
        for k in [k for k in states if not base[k]][:4 + 2 * s]:
            o[f"parents_mr24_s{s}"][k] = True
        outs[s] = o
    est, ci = joint_boot(outs, lambda s: f"parents_mr24_s{s}",
                         lambda s: f"parents24_s{s}", states, 500, 1)
    d0 = np.mean([np.mean([float(outs[0]["parents_mr24_s0"][k]) - float(
        base[k]) for k in states if k[0] == f"10/{t}"]) for t in range(10)])
    d1 = np.mean([np.mean([float(outs[1]["parents_mr24_s1"][k]) - float(
        base[k]) for k in states if k[0] == f"10/{t}"]) for t in range(10)])
    assert abs(est - (d0 + d1) / 2) < 1e-12 and ci[0] <= est <= ci[1]
    print("самопроверка k24_analyze пройдена: совместный парный бутстрэп "
          "средней разности двух сидов (оценка = среднее Δ сидов)")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-24: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    res, per, outs, st = {}, {}, {}, None
    try:
        for s in a.seeds:
            plan, r, outc = load_seed(s)
            res[s] = r
            if outc is None:
                continue
            per[s], st = per_seed(s, plan, outc)
            outs[s] = outc
            boot = plan["bootstrap"]
        summ = summary(per, outs, st, boot) if len(per) == len(a.seeds) and \
            len(per) > 1 else None
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    report(res, per, summ)
    os.makedirs("reports/k24", exist_ok=True)
    out = OUT if summ else f"reports/k24/analysis_s{'_'.join(map(str, a.seeds))}.json"
    json.dump(dict(plans=res, per_seed=per, summary=summ), open(
        out, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
        default=str)
    print(f"  сводка: {out}")
    return 0 if all(r["complete"] for r in res.values()) else 3


if __name__ == "__main__":
    sys.exit(main())
