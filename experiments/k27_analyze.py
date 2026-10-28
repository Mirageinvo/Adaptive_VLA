#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-27: анализ parents_mem24 против исторического parents24 по сидам 0 и 1.

Только проверенные блоки (новые — check_run основного плана K-27,
обязательные источники — заново проверенные планы с закреплёнными SHA;
необязательные MR/cond — только при полном проверенном наборе, иначе
«сравнение не выполнено» с причиной). Состояние — (набор/задача,
init_state_id), env seed 101; бутстрэп N = 10000, seed из плана.
По каждому сиду S:
  механизм — parents_mem24_sS − parents24_sS: ΔSR (равные веса задач),
  выигрыши/потери, точный Мак-Немар, 95% бутстрэп внутри задач; Holm по
  двум p;
  инженерные — против fullbar и fullbar_cached (консервативная граница
  не-хужести, маржа 3 п.п.); flat24 того же сида, MR и cond — описательно;
  задача 8 — описательно.
Сводка двух сидов (всегда оба): SR, средние парные разности, совместный
парный бутстрэп (одни индексы состояний для всех политик и сидов; 2×150 не
объединяются в 300). Два вывода, как в K-26: project_result (SR к BAR —
точечно / не-хужесть / превосходство раздельно, задержка из сопоставимого
замера, изменение к parents24) и mechanism_evidence (memory − parents24).
SR и p-value не влияют на код завершения (3 — только неполный набор).
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
import k26_analyze as a26  # noqa: E402

OUT = "reports/k27/analysis.json"
LATENCY = "reports/k27/latency.json"
ALPHA = 0.05
LABS = dict(new="parents_mem24_s{}", old="parents24_s{}", flat="flat24_s{}",
            mr="parents_mr24_s{}", cond="parents_cond24_s{}")
REQUIRED = ("new", "old", "flat")
OPTIONAL = dict(mr="k24_s{}", cond="k25_s{}")


def lab(key, seed):
    return LABS[key].format(seed) if key in LABS else key


def load_seed(seed):
    import k23_protocol as p23
    import k26_protocol as p26
    import k27_protocol as pc
    path = pc.plan_path(seed)
    if not os.path.exists(path):
        return None, dict(plan=path, verified=0, planned=30, gaps=[],
                          complete=False), None, {}
    plan = json.load(open(path, encoding="utf-8"))
    prob = pc.check_plan_inputs(plan)
    if prob:
        raise SystemExit(f"{path}: " + "; ".join(prob[:3]))
    ok, gaps = pc.check_run(plan, path, pc.run_dir(path))
    want = len(plan["cells"]) * len(plan["labels"])
    res = dict(plan=path, verified=len(ok), planned=want, gaps=gaps,
               complete=len(ok) == want and not gaps)
    if not res["complete"]:
        return plan, res, None, {}
    outc = {**a23.outcomes(p23.verify_refs(plan)), **a23.outcomes(ok)}
    opt = {}
    for key, name in OPTIONAL.items():
        nm = name.format(seed)
        blocks, why = p26.verify_optional(dict(optional_references={
            nm: (plan.get("optional_references") or {}).get(
                nm, dict(status="не выполнено", why="не закреплён"))}))[nm]
        if blocks is None:
            opt[key] = dict(status="не выполнено", why=why)
        else:
            outc.update(a23.outcomes(blocks))
            opt[key] = dict(status="выполнено")
    return plan, res, outc, opt


def per_seed(seed, plan, outc, opt=None):
    import k15h_analyze as ha
    labs = [lab(k, seed) for k in REQUIRED] + ["fullbar", "fullbar_cached"]
    states = a23.same_states(outc, labs, 150)
    boot = plan["bootstrap"]
    nw = lab("new", seed)
    o = dict(seed=seed, states=len(states),
             sr={k: ha.equal_weight(outc[k], states) for k in labs},
             primary=a23.cmp(outc, nw, lab("old", seed), states, boot),
             vs_flat=a23.cmp(outc, nw, lab("flat", seed), states, boot),
             vs_fullbar=a23.cmp(outc, nw, "fullbar", states, boot, True),
             vs_cached=a23.cmp(outc, nw, "fullbar_cached", states, boot,
                               True),
             task8={k: a23.per_task(outc[k], states).get("10/8")
                    for k in labs})
    for key in OPTIONAL:
        st = (opt or {}).get(key) or dict(status="не выполнено",
                                          why="не закреплён")
        L_ = lab(key, seed)
        if st["status"] == "выполнено" and L_ in outc and \
                sorted(outc[L_]) == states:
            o["sr"][L_] = ha.equal_weight(outc[L_], states)
            o[f"vs_{key}"] = dict(a23.cmp(outc, nw, L_, states, boot),
                                  status="выполнено")
        else:
            o[f"vs_{key}"] = dict(status="не выполнено", why=st.get(
                "why", "состояния ≠ состояниям K-27"))
    return o, states


TERMS = dict(primary=[(1, "new"), (-1, "old")],
             vs_flat=[(1, "new"), (-1, "flat")],
             vs_fullbar=[(1, "new"), (-1, "fullbar")],
             vs_cached=[(1, "new"), (-1, "fullbar_cached")])


def joint_boot(outs, terms, states, n, seed):
    rng = np.random.default_rng(seed)
    by = {}
    for k in states:
        by.setdefault(k[0], []).append(k)
    tasks = sorted(by)
    seeds = sorted(outs)
    D = {s: {t: sum(c * np.asarray([float(outs[s][lab(key, s)][k])
                                    for k in by[t]]) for c, key in terms)
             for t in tasks} for s in seeds}
    est = float(np.mean([np.mean([D[s][t].mean() for t in tasks])
                         for s in seeds]))
    bs = np.empty(n)
    for i in range(n):
        idx = {t: rng.integers(0, len(by[t]), len(by[t])) for t in tasks}
        bs[i] = np.mean([np.mean([D[s][t][idx[t]].mean() for t in tasks])
                         for s in seeds])
    return est, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def read_latency(path=LATENCY):
    if not os.path.exists(path):
        return dict(status="не выполнено", why=f"нет {path}")
    r = json.load(open(path, encoding="utf-8"))
    out = dict(status="выполнено", device=r.get("device"), rows={})
    for bk, b in r.items():
        if not (bk.startswith("batch") and isinstance(b, dict)):
            continue
        for regime in ("steady", "cold"):
            x = (b.get(regime) or {}).get("parents_mem24")
            if x is None:
                continue
            row = dict(ms=x["total"]["median"])
            for ref in ("parents24", "fullbar"):
                y = (b.get(regime) or {}).get(ref)
                if y:
                    row[f"vs_{ref}_pct"] = dict(
                        median=100 * (x["total"]["median"] /
                                      y["total"]["median"] - 1),
                        p95=100 * (x["total"]["p95"] / y["total"]["p95"] - 1))
            out["rows"][f"{bk}_{regime}"] = row
    return out


def summary(per, outs, states, boot, latency=None):
    import k15h_analyze as ha
    seeds = sorted(per)
    rows = {}
    for key in REQUIRED + ("fullbar", "fullbar_cached"):
        v = [per[s]["sr"][lab(key, s)] for s in seeds]
        rows[LABS.get(key, key).format("S")] = dict(
            per_seed=dict(zip(seeds, v)), mean=float(np.mean(v)),
            sd=float(np.std(v, ddof=1)) if len(v) > 1 else None)
    for key in OPTIONAL:
        if all(per[s][f"vs_{key}"]["status"] == "выполнено" for s in seeds):
            v = [per[s]["sr"][lab(key, s)] for s in seeds]
            rows[LABS[key].format("S")] = dict(per_seed=dict(zip(seeds, v)),
                                               mean=float(np.mean(v)),
                                               sd=None)
    out = dict(seeds=seeds, rows=rows, holm_primary=dict(zip(
        seeds, ha.holm([per[s]["primary"]["mcnemar_p"] for s in seeds]))))
    for key, terms in TERMS.items():
        d = {s: per[s][key]["d_sr_equal_weight"] for s in seeds}
        est, ci = joint_boot(outs, terms, states, boot["n"], boot["seed"])
        out[key] = dict(per_seed=d, mean=float(np.mean(list(d.values()))),
                        joint_paired_bootstrap=dict(d=est, ci95=ci))
    ps = {s: dict(
        d=per[s]["vs_fullbar"]["d_sr_equal_weight"],
        **a26.stat_flags(per[s]["vs_fullbar"])) for s in seeds}
    out["project_result"] = dict(
        vs_fullbar=ps, vs_fullbar_mean=out["vs_fullbar"]["mean"],
        vs_old_parents_mean=out["primary"]["mean"],
        latency=latency or dict(status="не выполнено"),
        text="дерево с памятью к BAR: " + ", ".join(
            f"s{s} {100 * v['d']:+.1f} п.п. ({v['point']}; не-хужесть "
            f"{'установлена' if v['non_inferior_3pp'] else 'не установлена'}"
            f"; превосходство "
            f"{'установлено' if v['superior'] else 'не установлено'})"
            for s, v in ps.items()))
    d = out["primary"]["per_seed"]
    ci = out["primary"]["joint_paired_bootstrap"]["ci95"]
    holm = out["holm_primary"]
    if all(v > 0 for v in d.values()):
        sign = "точечно память выше parents24 в обоих сидах"
    elif all(v < 0 for v in d.values()):
        sign = "точечно память ниже parents24 в обоих сидах"
    elif all(v == 0 for v in d.values()):
        sign = "точечно равны в обоих сидах"
    else:
        sign = "знак разности различается между сидами"
    est = all(h < ALPHA for h in holm.values()) and (ci[0] > 0 or ci[1] < 0) \
        and len({np.sign(v) for v in d.values()}) == 1
    out["mechanism_evidence"] = dict(
        sign=sign, established=est, mean=out["primary"]["mean"],
        joint_ci95=ci, holm=holm,
        text=f"{sign}; " + ("различие статистически установлено"
                            if est else "статистически различие не "
                            "установлено"),
        note="польза древовидной памяти относительно обычной рекуррентной "
             "памяти этим этапом не выделяется")
    out["note"] = ("разработочный банк 30–44 (разведочно); интервалы условны "
                   "на двух обученных реализациях")
    return out


def exit_code(res):
    return 0 if all(r["complete"] for r in res.values()) else 3


def report(res, per, summ):
    line = a23.line
    for s, r in res.items():
        print(f"  seed {s}: {r['plan']}: проверено {r['verified']}/"
              f"{r['planned']}" + ("" if r["complete"] else " — НЕПОЛНЫЙ"))
    for s, o in sorted(per.items()):
        print(f"\n  SEED {s} (150 состояний 30–44): " + ", ".join(
            f"{k} {100 * v:.1f}%" for k, v in o["sr"].items()))
        print(f"    механизм parents_mem24 − parents24: {line(o['primary'])}"
              + (f"; Holm {summ['holm_primary'][s]:.4f}" if summ else ""))
        print(f"    − flat24 (описательно): {line(o['vs_flat'])}")
        print(f"    − fullbar: {line(o['vs_fullbar'])}")
        print(f"    − fullbar_cached: {line(o['vs_cached'])}")
        for key in OPTIONAL:
            c = o[f"vs_{key}"]
            print(f"    − {key} (необязательно): " + (
                line(c) if c["status"] == "выполнено" else
                f"сравнение не выполнено: {c['why']}"))
        print("    задача 8 (описательно): " + ", ".join(
            f"{k} {v[0]}/{v[1]}" for k, v in o["task8"].items()))
    if summ:
        print("\n  СВОДКА ДВУХ СИДОВ (без отбора):")
        for a, r in summ["rows"].items():
            print(f"    {a:18} " + " ".join(f"s{s} {100 * v:.1f}%" for s, v in
                                           r["per_seed"].items())
                  + f" | среднее {100 * r['mean']:.1f}%")
        for key, nm in (("primary", "механизм память − parents24"),
                        ("vs_flat", "память − flat24"),
                        ("vs_fullbar", "память − fullbar"),
                        ("vs_cached", "память − fullbar_cached")):
            c = summ[key]
            jb = c["joint_paired_bootstrap"]
            print(f"    {nm}: " + ", ".join(
                f"s{s} {100 * v:+.1f}" for s, v in c["per_seed"].items())
                  + f"; среднее {100 * c['mean']:+.1f} п.п. "
                  f"[{100 * jb['ci95'][0]:+.1f}; {100 * jb['ci95'][1]:+.1f}]")
        pr = summ["project_result"]
        print(f"\n  РЕЗУЛЬТАТ МОДЕЛИ: {pr['text']}")
        lt = pr["latency"]
        if lt.get("status") == "выполнено":
            for k, row in lt["rows"].items():
                print(f"    задержка {k}: {row['ms']:.1f} мс; " + ", ".join(
                    f"к {kk[3:-4]} {v['median']:+.1f}% (p95 {v['p95']:+.1f}%)"
                    for kk, v in row.items() if kk.startswith("vs_")))
        else:
            print(f"    задержка: не выполнено ({lt.get('why')})")
        me = summ["mechanism_evidence"]
        print(f"  СВИДЕТЕЛЬСТВО МЕХАНИЗМА: {me['text']}")


def selftest():
    def with_sr(n):
        per_t = [n // 10 + (1 if t < n % 10 else 0) for t in range(10)]
        return {(f"10/{t}", i): j < per_t[t] for t in range(10)
                for j, i in enumerate(range(30, 45))}

    def make(new, old=134, opt=True):
        outs, per = {}, {}
        for s in (0, 1):
            o = {lab("new", s): with_sr(new), lab("old", s): with_sr(old),
                 lab("flat", s): with_sr(133), "fullbar": with_sr(122),
                 "fullbar_cached": with_sr(129)}
            if opt:
                o[lab("mr", s)] = with_sr(135)
            outs[s] = o
            op = dict(mr=dict(status="выполнено") if opt else dict(
                status="не выполнено", why="тест"),
                cond=dict(status="не выполнено", why="нет полного набора"))
            per[s], st = per_seed(s, dict(bootstrap=dict(n=300, seed=1)), o,
                                  op)
        return summary(per, outs, st, dict(n=300, seed=1)), per
    sm, per = make(140)
    assert "сигнал" not in json.dumps(sm, ensure_ascii=False)
    assert sm["primary"]["mean"] > 0 and "выше" in \
        sm["mechanism_evidence"]["sign"]
    assert per[0]["vs_cond"]["status"] == "не выполнено" and \
        per[0]["vs_mr"]["status"] == "выполнено"
    assert "parents_mr24_sS" in sm["rows"] and "parents_cond24_sS" not in \
        sm["rows"]
    sm2, _p = make(130)
    assert "ниже" in sm2["mechanism_evidence"]["sign"]
    jb = sm["primary"]["joint_paired_bootstrap"]
    assert abs(jb["d"] - sm["primary"]["mean"]) < 1e-12
    assert exit_code({0: dict(complete=True), 1: dict(complete=False)}) == 3
    print("самопроверка k27_analyze пройдена: механизм память − parents24, "
          "Holm на 2, совместный бутстрэп = средняя Δ, MR/cond "
          "необязательны («не выполнено» с причиной), два вывода без общего "
          "сигнала, код — только полнота")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-27: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    res, per, outs, st, boot = {}, {}, {}, None, None
    try:
        for s in a.seeds:
            plan, r, outc, opt = load_seed(s)
            res[s] = r
            if outc is None:
                continue
            per[s], st = per_seed(s, plan, outc, opt)
            outs[s] = outc
            boot = plan["bootstrap"]
        summ = summary(per, outs, st, boot, read_latency()) \
            if len(per) == len(a.seeds) and len(per) > 1 else None
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    report(res, per, summ)
    os.makedirs("reports/k27", exist_ok=True)
    out = OUT if summ else \
        f"reports/k27/analysis_s{'_'.join(map(str, a.seeds))}.json"
    json.dump(dict(plans=res, per_seed=per, summary=summ), open(
        out, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
        default=str)
    print(f"  сводка: {out}")
    return exit_code(res)


if __name__ == "__main__":
    sys.exit(main())
