#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-31: анализ SR — два независимых вопроса.

Только проверенные блоки (k31_harness.outcomes: ровно 150 уникальных
состояний на руку/сид). Основное — задачи 0/6/8 (45 пар на сид):
  project_result:                parents_repair31 − parents_ref31;
  hierarchy_specific_evidence:   parents_repair31 − parents_flatrepair31;
по сидам: ΔSR (равные веса задач), rescue/harm, точный Мак-Немар, 95%
бутстрэп; Holm-4 по четырём основным p (заранее); совместный парный
бутстрэп двух сидов (одни и те же индексы init внутри задачи для обоих
сидов, 10 000, seed 20261231) — интервал условен на двух обученных
источниках/селекторах. Описательно: flatrepair − ref, остальные 7 задач
(перенос/сохранение; консервативная нижняя граница k21_analyze с маржой
3 п.п., Clopper–Pearson 95% на выигрыши и потери), все 10, по задачам,
применение ремонта (доля эпизодов, уровни, фактическая амплитуда).
Уровни ремонта не назначались случайно — их SR не причинный вклад уровня.
Код завершения зависит только от полноты/проверок (3), не от SR/p.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT = "reports/k31/analysis.json"
LATENCY = "reports/k31/latency.json"
PRIMARY = (0, 6, 8)
REST = (1, 2, 3, 4, 5, 7, 9)
BOOT = dict(n=10000, seed=20261231)
MARGIN = 0.03
SEEDS = (0, 1)
MAIN = (("repair", "ref"), ("repair", "flatrepair"))


def states_of(o, tasks):
    return sorted(k for k in o if k[0] in tasks)


def compare(a, b, states, boot=BOOT):
    import k15h_analyze as ha
    import k21_analyze as pa
    c = ha.compare(a, b, states, boot["n"], boot["seed"])
    c.update(sr_a=ha.equal_weight(a, states), sr_b=ha.equal_weight(b, states),
             n=len(states),
             ni_lower_conservative=pa.ni_lower(c["gain"], c["loss"],
                                               len(states)))
    c["non_inferior_margin_3pp"] = c["ni_lower_conservative"] > -MARGIN
    return c


def holm(ps):
    """{имя: p} -> {имя: скорректированное p} (Holm)."""
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def joint_boot(pairs, tasks, boot=BOOT):
    """pairs {seed: (a, b)} -> средняя по сидам ΔSR и 95% интервал; одни и
    те же пересэмплированные индексы init внутри задачи для всех сидов."""
    seeds = sorted(pairs)
    st = states_of(pairs[seeds[0]][0], tasks)
    for s in seeds:
        if states_of(pairs[s][0], tasks) != st or \
                states_of(pairs[s][1], tasks) != st:
            raise SystemExit("наборы состояний сидов различаются")
    by = {}
    for k in st:
        by.setdefault(k[0], []).append(k)
    D = {s: {t: np.array([float(pairs[s][0][k]) - float(pairs[s][1][k])
                          for k in v]) for t, v in by.items()} for s in seeds}
    est = float(np.mean([np.mean([D[s][t].mean() for t in by])
                         for s in seeds]))
    rng = np.random.default_rng(boot["seed"])
    bs = np.empty(boot["n"])
    for i in range(boot["n"]):
        idx = {t: rng.integers(0, len(v), len(v)) for t, v in by.items()}
        bs[i] = np.mean([np.mean([D[s][t][idx[t]].mean() for t in by])
                         for s in seeds])
    return dict(d=est, ci95=[float(np.percentile(bs, 2.5)),
                             float(np.percentile(bs, 97.5))])


def repair_stats(lab):
    """Применение ремонта по проверенным блокам руки."""
    import k31_harness as kh
    n_ep = n_rep = 0
    rows = {}
    amp = []
    for t in kh.TASKS:
        for b in kh.BLOCKS:
            path = kh.block_path(lab, t, b)
            d = json.load(open(path, encoding="utf-8"))
            z = np.load(os.path.join(os.path.dirname(path), d["npz"]))
            for e in d["episodes"]:
                n_ep += 1
                n_rep += int(e["repairs"] > 0)
                for r in e["repair_rows"]:
                    rows[r] = rows.get(r, 0) + 1
            ch = np.asarray(z["j_chosen"])
            k = np.asarray(z["j_k"])
            for i in np.nonzero((ch > 0) & (k >= 1))[0]:
                diff = z["j_a_codec"][i][:int(k[i]), :6] - \
                    z["j_ref_arm"][i][:int(k[i])]
                amp.append(float(np.sqrt(np.mean(diff ** 2))))
    return dict(episodes=n_ep, with_repair=n_rep,
                share=n_rep / max(n_ep, 1), rows=rows,
                exec_rms_codec=dict(
                    median=float(np.median(amp)) if amp else None,
                    max=float(np.max(amp)) if amp else None, n=len(amp)))


def load_all():
    import torch
    import k31_harness as kh
    import k31_protocol as pc
    outc, plans = {}, {}
    for s in SEEDS:
        plan = pc.load_eval_plan(s)
        sel = pc.load_selector(torch, plan, s)
        plans[s] = plan
        for k in kh.KINDS:
            lab = kh.label(k, s)
            outc[(k, s)] = kh.outcomes(plan, lab, sel if k != "ref" else None)
    return outc, plans


def analyze(outc):
    res = dict(per_seed={}, main={}, joint={}, descriptive={})
    ps = {}
    for s in SEEDS:
        r = {}
        prim = states_of(outc[("ref", s)], PRIMARY)
        for a, b in MAIN + (("flatrepair", "ref"),):
            r[f"{a}-{b}"] = dict(
                primary=compare(outc[(a, s)], outc[(b, s)], prim),
                rest7=compare(outc[(a, s)], outc[(b, s)],
                              states_of(outc[("ref", s)], REST)),
                all10=compare(outc[(a, s)], outc[(b, s)],
                              sorted(outc[("ref", s)])))
        for a, b in MAIN:
            ps[f"{a}-{b} s{s}"] = r[f"{a}-{b}"]["primary"]["mcnemar_p"]
        r["per_task"] = {k: {str(t): int(sum(outc[(k, s)][x] for x in
                                             outc[(k, s)] if x[0] == t))
                             for t in range(10)} for k in ("ref", "repair",
                                                           "flatrepair")}
        res["per_seed"][str(s)] = r
    res["holm4"] = holm(ps)
    for a, b in MAIN + (("flatrepair", "ref"),):
        pairs = {s: (outc[(a, s)], outc[(b, s)]) for s in SEEDS}
        res["joint"][f"{a}-{b}"] = dict(primary=joint_boot(pairs, PRIMARY),
                                        rest7=joint_boot(pairs, REST))
    for a, b in MAIN:
        key = f"{a}-{b}"
        sig = [res["holm4"][f"{key} s{s}"] < 0.05 for s in SEEDS]
        ds = [res["per_seed"][str(s)][key]["primary"]["d_sr_equal_weight"]
              for s in SEEDS]
        jb = res["joint"][key]["primary"]
        verdict = ("установлено в обоих сидах (Holm-4)" if all(sig) else
                   "в одном сиде (Holm-4)" if any(sig) else "не установлено")
        res["main"][key] = dict(d_by_seed=ds, joint=jb, holm_significant=sig,
                                verdict=verdict,
                                sign_consistent=bool(all(d > 0 for d in ds)
                                                     or all(d < 0
                                                            for d in ds)))
    res["project_result"] = res["main"]["repair-ref"]["verdict"]
    res["hierarchy_specific_evidence"] = res["main"][
        "repair-flatrepair"]["verdict"]
    return res


def line(c):
    return (f"ΔSR {100 * c['d_sr_equal_weight']:+.1f} п.п. "
            f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
            f"+{c['gain']}/−{c['loss']}, Мак-Немар p {c['mcnemar_p']:.4f}")


def report(res):
    for s in SEEDS:
        r = res["per_seed"][str(s)]
        print(f"\n  SEED {s}")
        for key in ("repair-ref", "repair-flatrepair", "flatrepair-ref"):
            c = r[key]
            tag = "" if key != "flatrepair-ref" else " (описательно)"
            print(f"    {key}{tag}: основное 0/6/8 {line(c['primary'])}; "
                  f"SR {100 * c['primary']['sr_a']:.1f}% / "
                  f"{100 * c['primary']['sr_b']:.1f}%")
            print(f"      остальные 7: {line(c['rest7'])}; NI-нижняя "
                  f"{100 * c['rest7']['ni_lower_conservative']:+.1f} п.п.; "
                  f"все 10: {line(c['all10'])}")
        print("    по задачам ref/repair/flatrepair: " + ", ".join(
            f"t{t} {r['per_task']['ref'][str(t)]}/"
            f"{r['per_task']['repair'][str(t)]}/"
            f"{r['per_task']['flatrepair'][str(t)]}" for t in range(10)))
        for k in ("repair", "flatrepair"):
            st = res["descriptive"].get(f"{k}_s{s}")
            if st:
                print(f"    применение {k}: {st['with_repair']}/"
                      f"{st['episodes']} эпизодов, строки {st['rows']}, "
                      f"RMS исполненной поправки (кодек) "
                      f"{st['exec_rms_codec']['median']}")
    print("\n  Holm-4: " + ", ".join(f"{k} {v:.4f}" for k, v in
                                       res["holm4"].items()))
    for key, v in res["main"].items():
        j = v["joint"]
        print(f"  ДВА СИДА {key}: Δ " + ", ".join(
            f"s{s} {100 * d:+.1f}" for s, d in zip(SEEDS, v["d_by_seed"])) +
            f"; среднее {100 * j['d']:+.1f} п.п. [{100 * j['ci95'][0]:+.1f}; "
            f"{100 * j['ci95'][1]:+.1f}]; {v['verdict']}")
    print(f"  project_result: {res['project_result']}")
    print(f"  hierarchy_specific_evidence: {res['hierarchy_specific_evidence']}")
    lt = res.get("latency", {})
    print(f"  задержка: {lt.get('status', 'не выполнено')}")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    try:
        outc, _plans = load_all()
    except SystemExit as e:
        print(f"  ОТКАЗ/НЕПОЛНО: {e.code}")
        return 3
    import k31_harness as kh
    res = analyze(outc)
    for s in SEEDS:
        for k in ("repair", "flatrepair"):
            res["descriptive"][f"{k}_s{s}"] = repair_stats(kh.label(k, s))
    res["latency"] = json.load(open(LATENCY, encoding="utf-8")) \
        if os.path.exists(LATENCY) else dict(status="ожидается")
    if "status" not in res["latency"]:
        res["latency"]["status"] = "выполнено"
    report(res)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=float)
    print(f"  сводка: {OUT}")
    return 0


def selftest():
    rng = np.random.default_rng(0)
    states = [(t, i) for t in range(10) for i in range(30, 45)]

    def arm(p):
        return {k: bool(rng.random() < p) for k in states}
    outc = {}
    for s in SEEDS:
        outc[("ref", s)] = arm(0.8)
        outc[("repair", s)] = dict(outc[("ref", s)])
        outc[("flatrepair", s)] = dict(outc[("ref", s)])
    r0 = analyze(outc)
    assert r0["project_result"] == "не установлено"
    assert r0["main"]["repair-ref"]["d_by_seed"] == [0.0, 0.0]
    # сильный выигрыш repair на 0/6/8 в обоих сидах -> установлено
    for s in SEEDS:
        outc[("ref", s)] = {k: (k[0] not in PRIMARY or k[1] < 34)
                            for k in states}
        outc[("repair", s)] = {k: True for k in states}
    r1 = analyze(outc)
    assert r1["project_result"] == "установлено в обоих сидах (Holm-4)"
    assert r1["main"]["repair-ref"]["joint"]["d"] > 0.5
    # Holm монотонен и ≥ исходных p
    h = holm(dict(a=0.01, b=0.04, c=0.03, d=0.5))
    assert h["a"] == 0.04 and h["c"] == 0.09 and h["b"] == 0.09 and \
        h["d"] == 0.5
    # совместный бутстрэп требует одинаковых состояний сидов
    bad = {0: (outc[("repair", 0)], outc[("ref", 0)]),
           1: ({k: v for k, v in outc[("repair", 1)].items()
                if k != (0, 30)}, outc[("ref", 1)])}
    try:
        joint_boot(bad, PRIMARY)
        raise AssertionError("разные наборы сидов приняты")
    except SystemExit:
        pass
    print("самопроверка k31_analyze пройдена: нулевой эффект — не "
          "установлено, сильный в обоих сидах — установлено (Holm-4), Holm "
          "монотонен, совместный бутстрэп требует одинаковых состояний")
    return 0


if __name__ == "__main__":
    sys.exit(main())
