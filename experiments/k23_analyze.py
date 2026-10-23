#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-23: анализ пакетов A и C и сводка двух training seeds.

Состояние — (набор/задача, init_state_id) при env seed 101; страта
бутстрэпа — набор/задача. Только проверенные блоки (новые — check_run плана,
прежние — заново проверенные планы-источники с закреплёнными SHA блоков);
неполный набор — код 3 без вывода. Бутстрэп: 10 000 повторов, seed
20261023 (из плана).

suites (A): parents24_s0 на spatial/object/goal; ОСНОВНОЕ — parents24_s0 −
  flat24_s0 по объединению трёх наборов (150 состояний, 30 задач: ΔSR с
  равными весами задач, выигрыши/потери, точный Мак-Немар, бутстрэп внутри
  задач). Описательно: по наборам, к fullbar (граница не-хужести), к tree;
  сводка ЧЕТЫРЁХ наборов (LIBERO-10 из плана A и K-22): каждому набору
  вес 1/4 (не 300 состояний вместе), разность и интервал — с теми же весами
  и стратификацией по набору и задаче. У fullbar_cached трёх наборов нет.
repeat --seed S (C): flat24_s<S>, parents24_s<S> на 150 состояниях 30–44;
  ОСНОВНОЕ — parents24_s<S> − flat24_s<S>; к fullbar — граница
  не-хужести (маржа 3 п.п.); к fullbar_cached — дополнительно; к tree и
  задача 8 — описательно. Затем сводка сидов 0 и S: все сиды (без отбора),
  среднее и выборочное ско по архитектуре, Δ_s = SR(parents) − SR(flat) по
  каждому сиду, средняя Δ, отдельно новый повтор; ограниченный описательный
  интервал — пересэмплирование пар сидов и состояний внутри задачи (одни и
  те же состояния для всех сидов и архитектур). 2 × 150 НЕ объединяются в
  300 независимых стартов, Мак-Немар к объединению не применяется.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MARGIN = 0.03
ALLOWED = {"10": set(range(30, 45)), "spatial": set(range(0, 5)),
           "object": set(range(0, 5)), "goal": set(range(0, 5))}


def outcomes(blocks, allowed=ALLOWED):
    """blocks {(метка, набор, задача, блок): путь} -> {метка: {состояние:
    успех}}. Отказ: повтор (метка, набор, задача, старт), старт вне
    разрешённых для набора, env seed не 101."""
    import k16_protocol as p16
    out = {}
    for (lab, s, t, b), f in sorted(blocks.items()):
        d, _z = p16.load(f) if isinstance(f, str) else (f, None)
        for e in d["episodes"]:
            i = int(e["init_state_id"])
            if i not in allowed[s] or i >= 45:
                raise SystemExit(f"{lab} {s}/{t}: старт {i} вне оценочного "
                                 f"набора")
            if int(e.get("rollout_seed", 101)) != 101:
                raise SystemExit(f"{lab} {s}/{t}/{i}: env seed не 101")
            k = (f"{s}/{int(t)}", i)
            o = out.setdefault(lab, {})
            if k in o:
                raise SystemExit(f"повтор состояния {k} у {lab}")
            o[k] = bool(e["success"])
    return out


def same_states(outc, labels, n=None):
    states = sorted(outc[labels[0]])
    for lab in labels:
        if sorted(outc[lab]) != states:
            raise SystemExit(f"состояния {lab} ≠ {labels[0]}")
    if n is not None and len(states) != n:
        raise SystemExit(f"состояний {len(states)}, ожидалось {n}")
    return states


def cmp(outc, a, b, states, boot, ni=False):
    import k15h_analyze as ha
    import k21_analyze as pa
    c = dict(a=a, b=b, **ha.compare(outc[a], outc[b], states, boot["n"],
                                    boot["seed"]))
    if ni:
        c["ni_lower_conservative"] = pa.ni_lower(c["gain"], c["loss"],
                                                 len(states))
        c["non_inferior_margin_3pp"] = c["ni_lower_conservative"] > -MARGIN
    return c


def per_task(o, states):
    pt = {}
    for k in states:
        v = pt.setdefault(k[0], [0, 0])
        v[0] += bool(o[k])
        v[1] += 1
    return pt


def suite_of(stratum):
    return stratum.split("/")[0]


def four_suite(outc, labels, states_by_suite):
    """SR каждого набора (равные веса задач), среднее четырёх с весом 1/4."""
    import k15h_analyze as ha
    out = {}
    for lab in labels:
        per = {s: ha.equal_weight(outc[lab], st)
               for s, st in states_by_suite.items()
               if all(k in outc[lab] for k in st)}
        out[lab] = dict(per_suite=per, mean4=(float(np.mean(list(
            per.values()))) if len(per) == len(states_by_suite) else None))
    return out


def boot_suite_weighted(a, b, states_by_suite, n, seed):
    """Δ(a − b): среднее по наборам (1/4) средних по задачам; состояния
    пересэмплируются внутри (набор, задача). -> (Δ, [2.5%, 97.5%])."""
    rng = np.random.default_rng(seed)
    strata = {}
    for s, st in states_by_suite.items():
        for k in st:
            strata.setdefault((s, k[0]), []).append(float(a[k]) -
                                                    float(b[k]))
    suites = list(states_by_suite)
    keys = sorted(strata)
    arr = [np.asarray(strata[k]) for k in keys]

    def stat(vals):
        by = {s: [] for s in suites}
        for (s, _t), v in zip(keys, vals):
            by[s].append(v)
        return float(np.mean([np.mean(by[s]) for s in suites]))
    est = stat([x.mean() for x in arr])
    bs = np.empty(n)
    for i in range(n):
        bs[i] = stat([x[rng.integers(0, len(x), len(x))].mean()
                      for x in arr])
    return est, [float(np.percentile(bs, 2.5)),
                 float(np.percentile(bs, 97.5))]


def seeds_summary(outc_by_seed, states, boot, new_seeds):
    """outc_by_seed {seed: {arch: {состояние: успех}}}."""
    import k15h_analyze as ha
    seeds = sorted(outc_by_seed)
    archs = ("flat24", "parents24")
    sr = {a: {s: ha.equal_weight(outc_by_seed[s][a], states) for s in seeds}
          for a in archs}
    rows = {a: dict(per_seed=sr[a], n_seeds=len(seeds),
                    mean=float(np.mean(list(sr[a].values()))),
                    sd=(float(np.std(list(sr[a].values()), ddof=1))
                        if len(seeds) > 1 else None)) for a in archs}
    delta = {s: sr["parents24"][s] - sr["flat24"][s] for s in seeds}
    # пересэмплирование сидов и состояний внутри задачи (общие для всех)
    rng = np.random.default_rng(boot["seed"])
    by = {}
    for k in states:
        by.setdefault(k[0], []).append(k)
    tasks = sorted(by)
    D = {s: {t: np.asarray([float(outc_by_seed[s]["parents24"][k]) -
                            float(outc_by_seed[s]["flat24"][k])
                            for k in by[t]]) for t in tasks} for s in seeds}
    bs = np.empty(boot["n"])
    for i in range(boot["n"]):
        ss = rng.choice(seeds, len(seeds), replace=True)
        idx = {t: rng.integers(0, len(by[t]), len(by[t])) for t in tasks}
        bs[i] = float(np.mean([np.mean([D[s][t][idx[t]].mean()
                                        for t in tasks]) for s in ss]))
    return dict(seeds=seeds, rows=rows, delta_per_seed=delta,
                delta_mean=float(np.mean(list(delta.values()))),
                delta_new_seeds=({s: delta[s] for s in new_seeds}),
                delta_new_mean=float(np.mean([delta[s] for s in new_seeds])),
                interval_seed_state_bootstrap=[
                    float(np.percentile(bs, 2.5)),
                    float(np.percentile(bs, 97.5))],
                note=f"{len(seeds)} training seeds: интервал — ограниченная "
                     f"описательная оценка, не доказательство превосходства; "
                     f"уникальных стартов {len(states)}")


# --- сбор --------------------------------------------------------------------
def load_plan(path):
    import k23_protocol as pc
    plan = json.load(open(path))
    prob = pc.check_plan_inputs(plan)
    if prob:
        raise SystemExit(f"{path}: " + "; ".join(prob[:3]))
    ok, gaps = pc.check_run(plan, path, pc.run_dir(path))
    want = len(plan["cells"]) * len(plan["labels"])
    res = dict(plan=path, verified=len(ok), planned=want, gaps=gaps,
               complete=len(ok) == want and not gaps)
    refs = pc.verify_refs(plan) if res["complete"] else None
    return plan, res, ok, refs


def analyze_suites(path):
    plan, res, ok, refs = load_plan(path)
    if not res["complete"]:
        return res, None
    boot = plan["bootstrap"]
    new = outcomes(ok)
    old = outcomes(refs)
    outc = {**old, **{k: {**old.get(k, {}), **v} for k, v in new.items()}}
    p, f = plan["primary"]["comparison"]
    three = [k for k in sorted(outc[p]) if suite_of(k[0]) in
             plan["suites"]]
    for lab in ("fullbar", "flat24_s0", "tree_adapt24_s0"):
        if not all(k in outc[lab] for k in three):
            raise SystemExit(f"{lab}: нет части состояний трёх наборов")
    if len(three) != 150:
        raise SystemExit(f"состояний трёх наборов {len(three)}")
    out = dict(states=len(three))
    out["primary"] = cmp(outc, p, f, three, boot)
    out["descriptive_pooled3"] = [cmp(outc, p, "fullbar", three, boot, True),
                                  cmp(outc, p, "tree_adapt24_s0", three,
                                      boot)]
    out["by_suite"] = {}
    for s in plan["suites"]:
        st = [k for k in three if suite_of(k[0]) == s]
        import k15h_analyze as ha
        out["by_suite"][s] = dict(
            sr={lab: ha.equal_weight(outc[lab], st) for lab in
                ("fullbar", "flat24_s0", "tree_adapt24_s0", p)},
            vs_flat=cmp(outc, p, f, st, boot))
    out["per_task"] = {lab: per_task(outc[lab], three) for lab in
                       ("fullbar", "flat24_s0", "tree_adapt24_s0", p)}
    # четыре набора: LIBERO-10 — 150 состояний 30–44
    ten = [k for k in sorted(outc["fullbar"]) if suite_of(k[0]) == "10"]
    sbs = {"10": ten, **{s: [k for k in three if suite_of(k[0]) == s]
                         for s in plan["suites"]}}
    labs4 = ("fullbar", "flat24_s0", "tree_adapt24_s0", p)
    for lab in labs4:
        if not all(k in outc[lab] for k in ten) or len(ten) != 150:
            raise SystemExit(f"{lab}: LIBERO-10 неполон")
    out["four_suites"] = four_suite(outc, labs4, sbs)
    d4, ci4 = boot_suite_weighted(outc[p], outc[f], sbs, boot["n"],
                                  boot["seed"])
    out["four_suites_diff"] = dict(a=p, b=f, d=d4, ci95=ci4,
                                   note="описательно; веса наборов 1/4")
    return res, out


def analyze_repeat(seed):
    import k23_protocol as pc
    path = pc.plan_path("repeat", seed)
    plan, res, ok, refs = load_plan(path)
    if not res["complete"]:
        return res, None, None
    if int(plan["train_seed"]) != seed:
        raise SystemExit("seed плана ≠ запрошенному")
    import k23_policy as kp
    for lab in plan["labels"]:
        if kp.parse_label(lab)[1] != seed:
            raise SystemExit(f"метка {lab} не seed {seed}")
    boot = plan["bootstrap"]
    outc = {**outcomes(refs), **outcomes(ok)}
    labs = ["fullbar", "fullbar_cached", "tree_adapt24_s0", "flat24_s0",
            "parents24_s0", f"flat24_s{seed}", f"parents24_s{seed}"]
    states = same_states(outc, labs, 150)
    p, f = plan["primary"]["comparison"]
    import k15h_analyze as ha
    out = dict(seed=seed, states=len(states),
               sr={lab: ha.equal_weight(outc[lab], states) for lab in labs},
               per_task={lab: per_task(outc[lab], states) for lab in labs})
    out["primary"] = cmp(outc, p, f, states, boot)
    out["vs_fullbar"] = [cmp(outc, x, "fullbar", states, boot, True)
                         for x in (p, f)]
    out["vs_fullbar_cached"] = [cmp(outc, x, "fullbar_cached", states, boot,
                                    True) for x in (p, f)]
    out["vs_tree_descriptive"] = cmp(outc, p, "tree_adapt24_s0", states,
                                     boot)
    out["task8_descriptive"] = {lab: per_task(outc[lab], states).get("10/8")
                                for lab in labs}
    by_seed = {0: dict(flat24=outc["flat24_s0"],
                       parents24=outc["parents24_s0"]),
               seed: dict(flat24=outc[f"flat24_s{seed}"],
                          parents24=outc[f"parents24_s{seed}"])}
    summ = seeds_summary(by_seed, states, boot, [seed])
    return res, out, summ


# --- печать -------------------------------------------------------------------
def line(c):
    s = (f"ΔSR {100 * c['d_sr_equal_weight']:+.1f} п.п. "
         f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
         f"+{c['gain']}/−{c['loss']}, Мак-Немар p {c['mcnemar_p']:.4f}")
    if "ni_lower_conservative" in c:
        s += (f"; нижняя граница {100 * c['ni_lower_conservative']:+.1f} "
              f"п.п. -> " + ("не-хужесть УСТАНОВЛЕНА" if
                             c["non_inferior_margin_3pp"] else
                             "не-хужесть НЕ установлена"))
    return s


def report_head(res):
    print(f"  {res['plan']}: проверено {res['verified']}/{res['planned']}; "
          f"проблем {len([g for g in res['gaps'] if g['why'] != 'не готов'])}"
          + ("" if res["complete"] else " — НЕПОЛНЫЙ: вывода нет"))
    for g in [g for g in res["gaps"] if g["why"] != "не готов"][:10]:
        print(f"    ВНИМАНИЕ {g}")


def report_suites(res, out):
    report_head(res)
    if out is None:
        return
    c = out["primary"]
    print(f"  ОСНОВНОЕ: {c['a']} − {c['b']} по объединению трёх наборов "
          f"(150): {line(c)}")
    print("  описательно:")
    for c in out["descriptive_pooled3"]:
        print(f"    {c['a']} − {c['b']}: {line(c)}")
    for s, v in out["by_suite"].items():
        print(f"    {s}: " + ", ".join(f"{k} {100 * x:.1f}%"
                                         for k, x in v["sr"].items())
              + f"; parents − flat {line(v['vs_flat'])}")
    print("  четыре набора (вес 1/4):")
    for lab, v in out["four_suites"].items():
        print(f"    {lab:16} " + " ".join(f"{s} {100 * x:5.1f}%" for s, x in
                                         v["per_suite"].items())
              + f" | среднее {100 * v['mean4']:.2f}%")
    d = out["four_suites_diff"]
    print(f"    {d['a']} − {d['b']}: {100 * d['d']:+.2f} п.п. "
          f"[{100 * d['ci95'][0]:+.2f}; {100 * d['ci95'][1]:+.2f}] "
          f"(описательно)")


def report_repeat(res, out, summ):
    report_head(res)
    if out is None:
        return
    print("  SR (150 состояний 30–44): " + ", ".join(
        f"{k} {100 * v:.1f}%" for k, v in out["sr"].items()))
    print(f"  ОСНОВНОЕ: {out['primary']['a']} − {out['primary']['b']}: "
          f"{line(out['primary'])}")
    for c in out["vs_fullbar"]:
        print(f"  к fullbar: {c['a']}: {line(c)}")
    for c in out["vs_fullbar_cached"]:
        print(f"  к fullbar_cached (доп.): {c['a']}: {line(c)}")
    print(f"  описательно к tree: {line(out['vs_tree_descriptive'])}")
    print("  задача 8 (описательно): " + ", ".join(
        f"{k} {v[0]}/{v[1]}" for k, v in out["task8_descriptive"].items()))
    print("  СВОДКА СИДОВ (все, без отбора):")
    for a, r in summ["rows"].items():
        print(f"    {a:10} " + " ".join(f"s{s} {100 * v:.1f}%" for s, v in
                                         r["per_seed"].items())
              + f" | среднее {100 * r['mean']:.1f}%, ско "
              + (f"{100 * r['sd']:.1f}" if r["sd"] is not None else "—")
              + f" (сидов {r['n_seeds']})")
    print("    Δ parents − flat по сидам: " + ", ".join(
        f"s{s} {100 * v:+.1f}" for s, v in summ["delta_per_seed"].items())
          + f"; среднее {100 * summ['delta_mean']:+.1f} п.п.; новый повтор "
          f"{100 * summ['delta_new_mean']:+.1f}; интервал (сиды × "
          f"состояния, описательно) "
          f"[{100 * summ['interval_seed_state_bootstrap'][0]:+.1f}; "
          f"{100 * summ['interval_seed_state_bootstrap'][1]:+.1f}]")


# --- самопроверка -------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(0)

    def blk(lab, s, t, b, succ):
        return {(lab, s, t, b): dict(episodes=[
            dict(init_state_id=b + i, success=bool(x), rollout_seed=101)
            for i, x in enumerate(succ)])}
    # повтор (метка, набор, задача, старт) — отказ
    bl = dict(blk("flat24_s1", "10", 0, 30, [1] * 5))
    bl[("flat24_s1", "10", 0, 31)] = bl[("flat24_s1", "10", 0, 30)]
    try:
        outcomes(bl)
        raise AssertionError("повтор принят")
    except SystemExit:
        pass
    # старт вне оценочного набора (резерв / обучающие старты) — отказ
    for s, b in (("10", 45), ("10", 0), ("goal", 30)):
        try:
            outcomes(blk("x_s1", s, 0, b, [1] * 5))
            raise AssertionError((s, b))
        except SystemExit:
            pass
    # разные наборы с одинаковым task_id — разные страты
    o = outcomes({**blk("a", "goal", 1, 0, [1] * 5),
                  **blk("a", "object", 1, 0, [0] * 5)})
    assert len({k[0] for k in o["a"]}) == 2
    # веса наборов 1/4, а не 300 состояний вместе
    st = {"10": [("10/0", i) for i in range(30, 45)] * 1,
          "goal": [("goal/0", i) for i in range(5)]}
    a = {k: True for v in st.values() for k in v}
    b = {**a, **{k: False for k in st["10"]}}
    fs = four_suite(dict(a=a, b=b), ["a", "b"], st)
    assert abs(fs["b"]["mean4"] - 0.5) < 1e-12       # (0 + 1) / 2
    pooled = np.mean([b[k] for v in st.values() for k in v])
    assert abs(pooled - 0.25) < 1e-12 and fs["b"]["mean4"] != pooled
    d, ci = boot_suite_weighted(a, b, st, 200, 1)
    assert abs(d - 0.5) < 1e-12 and ci[0] <= d <= ci[1]
    # сводка сидов: средние, ско, Δ по сидам, интервал
    states = [(f"10/{t}", i) for t in range(10) for i in range(30, 45)]
    base = {k: bool(rng.random() < .85) for k in states}
    by = {0: dict(flat24=dict(base), parents24=dict(base)),
          1: dict(flat24=dict(base), parents24=dict(base))}
    for k in [k for k in states if not base[k]][:6]:
        by[1]["parents24"][k] = True
    sm = seeds_summary(by, states, dict(n=300, seed=20261023), [1])
    assert sm["delta_per_seed"][0] == 0 and sm["delta_per_seed"][1] > 0
    assert abs(sm["delta_mean"] - sm["delta_per_seed"][1] / 2) < 1e-12
    assert sm["rows"]["parents24"]["sd"] > 0 and \
        sm["rows"]["flat24"]["sd"] == 0
    lo, hi = sm["interval_seed_state_bootstrap"]
    assert lo <= sm["delta_mean"] <= hi
    # неполные или разные наборы состояний — отказ
    try:
        same_states(dict(x=base, y={k: v for k, v in list(base.items())[1:]}),
                    ["x", "y"])
        raise AssertionError("разные наборы приняты")
    except SystemExit:
        pass
    try:
        same_states(dict(x=base), ["x"], 151)
        raise AssertionError("неверное число принято")
    except SystemExit:
        pass
    print("самопроверка k23_analyze пройдена: повтор и посторонние старты — "
          "отказ, наборы с одинаковым task_id раздельно, веса наборов 1/4, "
          "сводка сидов (среднее, ско, Δ, интервал), неполный набор — отказ")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-23: анализ")
    ap.add_argument("mode", choices=("suites", "repeat", "selftest"))
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    import k23_protocol as pc
    try:
        if a.mode == "suites":
            res, out = analyze_suites(pc.plan_path("suites"))
            report_suites(res, out)
            fn, body = "reports/k23/suites_analysis.json", dict(plan=res,
                                                                 **(out or {}))
        else:
            res, out, summ = analyze_repeat(a.seed)
            report_repeat(res, out, summ)
            fn = f"reports/k23/repeat_s{a.seed}_analysis.json"
            body = dict(plan=res, **(out or {}), seeds_summary=summ)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    os.makedirs("reports/k23", exist_ok=True)
    json.dump(body, open(fn, "w"), ensure_ascii=False, indent=1, default=str)
    print(f"  сводка: {fn}")
    return 0 if res["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
