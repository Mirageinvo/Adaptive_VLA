#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-36, этап C (§9.2–9.3, §10): итоговый анализ.

Таблица: случай × сид × рука (R, T32, Tplus) + общий вектор BAR (один бит
на случай, без «сидов BAR»). Эстиманды:
  adapted (0/6/8, по 100): Δ = ⅓ Σ_t mean_i ½ Σ_s (Y_X,s − Y_Y,s);
  rest7 (по 20): то же с ⅐ Σ_t;
  all10: равные веса десяти задач, затем двух сидов (не пул 440 случаев);
  X − BAR: d_t,i = ½ Σ_s Y_X,s,t,i − Y_BAR,t,i.
Бутстрэп: внутри каждой задачи случаи с возвращением, одни индексы для
всех рук, обоих сидов и BAR; 100 000 повторов, seed 36012.
Семья C (alpha 0.025, 97.5% интервал, нижняя граница > 0), фиксированная
последовательность: 1) Tplus−T32 adapted; 2) если 1 — Tplus−R adapted;
3) если 1 и 2 — Tplus−BAR all10. Неоткрытые сравнения — описательно;
отдельно обычные 95%. По сидам и задачам — SR, wins/losses, McNemar
(описательно). Код 0 при полных корректных данных любого знака, 3 —
неполные/повреждённые.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k36_analyze as ka  # noqa: E402
import k36_protocol as pc  # noqa: E402

OUT = "reports/k36/analysis_c.json"
ADAPTED = (0, 6, 8)
REST = tuple(t for t in range(10) if t not in ADAPTED)
ALL = tuple(range(10))
NAMES = ("R", "T32", "Tplus")


def dmap(Y, YB, x, y, tasks, seeds=(0, 1)):
    """-> {t: d_i}; y = 'BAR' — общий вектор BAR."""
    out = {}
    for t in tasks:
        a = np.mean([Y[x][s][t].astype(float) for s in seeds], 0)
        b = YB[t].astype(float) if y == "BAR" else \
            np.mean([Y[y][s][t].astype(float) for s in seeds], 0)
        out[t] = a - b
    return out


def boot(dmaps, tasks, n, seed, chunk=10_000):
    rng = np.random.default_rng(seed)
    out = {k: np.zeros(n) for k in dmaps}
    done = 0
    while done < n:
        m = min(chunk, n - done)
        acc = {k: np.zeros(m) for k in dmaps}
        for t in tasks:
            nt = len(next(iter(dmaps.values()))[t])
            idx = rng.integers(0, nt, (m, nt))
            for k, dm in dmaps.items():
                acc[k] += dm[t][idx].mean(1)
        for k in dmaps:
            out[k][done:done + m] = acc[k] / len(tasks)
        done += m
    return out


def macro(dm, tasks):
    return float(np.mean([dm[t].mean() for t in tasks]))


def analyze(Y, YB, n_boot=None, seed=None):
    st = pc.STATS
    n_boot = n_boot or st["boot_n"]
    seed = st["boot_seed_C"] if seed is None else seed
    comps = {"Tplus-T32@adapted": ("Tplus", "T32", ADAPTED),
             "Tplus-R@adapted": ("Tplus", "R", ADAPTED),
             "Tplus-BAR@all10": ("Tplus", "BAR", ALL),
             "T32-R@adapted": ("T32", "R", ADAPTED),
             "Tplus-T32@rest7": ("Tplus", "T32", REST),
             "Tplus-R@rest7": ("Tplus", "R", REST),
             "Tplus-T32@all10": ("Tplus", "T32", ALL),
             "Tplus-R@all10": ("Tplus", "R", ALL),
             "T32-BAR@all10": ("T32", "BAR", ALL),
             "R-BAR@all10": ("R", "BAR", ALL)}
    res = {}
    groups = {}
    for k, (x, y, tasks) in comps.items():
        groups.setdefault(tasks, {})[k] = dmap(Y, YB, x, y, tasks)
    for tasks, dms in groups.items():
        bs = boot(dms, tasks, n_boot, [seed, len(tasks)])
        for k, dm in dms.items():
            res[k] = dict(delta=macro(dm, tasks),
                          ci97_5=ka.ci(bs[k], st["level_decision"]),
                          ci95=ka.ci(bs[k], st["level_report"]))
    t1 = res["Tplus-T32@adapted"]["ci97_5"][0] > 0
    t2 = t1 and res["Tplus-R@adapted"]["ci97_5"][0] > 0
    t3 = t2 and res["Tplus-BAR@all10"]["ci97_5"][0] > 0
    family = dict(test1=dict(comparison="Tplus-T32@adapted",
                             confirmed=bool(t1)),
                  test2=dict(comparison="Tplus-R@adapted",
                             formally_tested=bool(t1), confirmed=bool(t2)),
                  test3=dict(comparison="Tplus-BAR@all10",
                             formally_tested=bool(t2), confirmed=bool(t3)))
    sr = {}
    for nm in NAMES:
        for s in (0, 1):
            sr[f"{nm}_s{s}"] = {str(t): float(Y[nm][s][t].mean()) for t in ALL}
        sr[f"{nm}_all10_macro"] = float(np.mean(
            [np.mean([Y[nm][s][t].mean() for t in ALL]) for s in (0, 1)]))
    sr["BAR"] = {str(t): float(YB[t].mean()) for t in ALL}
    sr["BAR_all10_macro"] = float(np.mean([YB[t].mean() for t in ALL]))
    per_seed = {}
    for s in (0, 1):
        per_seed[str(s)] = {}
        for x, y in (("Tplus", "T32"), ("Tplus", "R"), ("T32", "R")):
            X = np.concatenate([Y[x][s][t] for t in ADAPTED])
            Z = np.concatenate([Y[y][s][t] for t in ADAPTED])
            b, c = int((X & ~Z).sum()), int((~X & Z).sum())
            per_seed[str(s)][f"{x}-{y}@adapted"] = dict(
                delta=macro(dmap(Y, YB, x, y, ADAPTED, (s,)), ADAPTED),
                wins=b, losses=c, p_mcnemar=ka.mcnemar_exact(b, c))
        X = np.concatenate([Y["Tplus"][s][t] for t in ALL])
        Z = np.concatenate([YB[t] for t in ALL])
        b, c = int((X & ~Z).sum()), int((~X & Z).sum())
        per_seed[str(s)]["Tplus-BAR@all10"] = dict(
            delta=macro(dmap(Y, YB, "Tplus", "BAR", ALL, (s,)), ALL),
            wins=b, losses=c, p_mcnemar=ka.mcnemar_exact(b, c))
    return dict(comparisons=res, family=family, sr=sr, per_seed=per_seed,
                boot=dict(n=n_boot, seed=seed, shared="все руки, оба сида и "
                          "BAR на случае задачи"),
                note="рост на адаптированных задачах и превосходство над BAR "
                     "на LIBERO-10 — разные утверждения; NI не вводится")


def collect_c():
    import k36_harness as kh
    import k36c_bar as kbar
    import k36c_confirm as kc
    prep = pc.load_prepare()
    p = kc.plan_c_problems(prep)
    if p:
        raise SystemExit("; ".join(p))
    plan = json.load(open(kc.PLAN_C, encoding="utf-8"))
    C = kh.consts(prep)
    q = kbar.consts_bar(prep)
    gb = json.load(open(kbar.GATES["bar_smoke"], encoding="utf-8"))
    rev, vocab = gb["revision"], int(gb["vocab"])
    Y = {nm: {s: {} for s in (0, 1)} for nm in NAMES}
    YB = {}
    files, prob = {}, []
    for lab, t, g, cases in kc.cells_c(prep["splits"]):
        if not os.path.exists(kc.block_path(lab, t, g)):
            prob.append(f"нет блока {lab} t{t} b{g}")
            continue
        pr = kc.block_files_problems(prep, plan, (lab, t, g, cases), rev)
        if pr:
            prob.append(f"{lab} t{t} b{g}: {pr[:1]}")
    for t in ALL:
        ids = prep["splits"][str(t)]["c_used"]
        for lab in kc.ARMS_C:
            y = np.zeros(len(ids), bool)
            for i, cid in enumerate(ids):
                w = kc.c_want(prep, plan, lab, t, cid, rev)
                got = kh.load_ep(w)
                if got is None:
                    prob.append(f"{w['eid']}: нет эпизода")
                    continue
                pr = kbar.bar_episode_problems(*got, w, q, vocab) \
                    if lab == "BAR" else kh.episode_problems(*got, w, C)
                if pr:
                    prob.append(f"{w['eid']}: {pr[:2]}")
                y[i] = bool(got[0]["summary"]["success"])
                files.update(kh.ep_files(w))
            if lab == "BAR":
                YB[t] = y
            else:
                nm, _k, s = pc.parse_arm(lab)
                Y[nm][s][t] = y
    return Y, YB, files, prob, prep


def analyze_c(a):
    try:
        Y, YB, files, prob, prep = collect_c()
    except SystemExit as e:
        prob, prep = [str(e.code)], None
    if prob:
        pc.save_json(OUT, dict(status="technical_stop", problems=prob[:50]))
        print("  анализ C: ТЕХНИЧЕСКИЙ СТОП — " + "; ".join(prob[:4]))
        return 3
    import k36c_confirm as kc
    import k36c_latency as kl36
    lp = kl36.latency_problems(pc.sha(kc.PLAN_C))
    if lp:
        pc.save_json(OUT, dict(status="technical_stop", problems=lp))
        print("  анализ C: ТЕХНИЧЕСКИЙ СТОП — задержка: " + "; ".join(lp))
        return 3
    res = analyze(Y, YB)
    res.update(status="complete", prepare_sha1=prep["sha1"],
               n_files=len(files),
               latency=json.load(open(kl36.OUT)) if lp is not None else
               "latency_pending",
               table={f"{nm}_s{s}": {str(t): Y[nm][s][t].astype(int).tolist()
                                     for t in ALL}
                      for nm in NAMES for s in (0, 1)},
               table_BAR={str(t): YB[t].astype(int).tolist() for t in ALL})
    pc.save_json(OUT, res)
    for k, v in res["comparisons"].items():
        print(f"  {k}: Δ {100 * v['delta']:+.1f} п.п.; 97.5% "
              f"[{100 * v['ci97_5'][0]:+.1f}; {100 * v['ci97_5'][1]:+.1f}], "
              f"95% [{100 * v['ci95'][0]:+.1f}; {100 * v['ci95'][1]:+.1f}]")
    print(f"  семья C: {res['family']}")
    return 0


def selftest():
    rng = np.random.default_rng(0)

    def table(pp, pt, pr, pb):
        Y = {nm: {s: {} for s in (0, 1)} for nm in NAMES}
        YB = {}
        for t in ALL:
            n = 100 if t in ADAPTED else 20
            u = rng.random(n)
            for s in (0, 1):
                Y["Tplus"][s][t] = u < pp
                Y["T32"][s][t] = u < pt
                Y["R"][s][t] = u < pr
            YB[t] = u < pb
        return Y, YB
    r = analyze(*table(0.95, 0.75, 0.70, 0.70), n_boot=3000, seed=1)
    assert all(r["family"][k]["confirmed"] for k in ("test1", "test2",
                                                     "test3")), r["family"]
    r = analyze(*table(0.8, 0.8, 0.8, 0.8), n_boot=1000, seed=1)
    assert not r["family"]["test1"]["confirmed"] and \
        not r["family"]["test2"]["formally_tested"] and \
        not r["family"]["test3"]["formally_tested"]
    # all10 — равные веса задач (не пул случаев): задача с 20 случаями
    # весит как задача со 100
    Y, YB = table(0.8, 0.8, 0.8, 0.8)
    for s in (0, 1):
        Y["Tplus"][s][1] = np.ones(20, bool)
        Y["T32"][s][1] = np.zeros(20, bool)
        for t in ALL:
            if t != 1:
                Y["Tplus"][s][t] = Y["T32"][s][t].copy()
    r = analyze(Y, YB, n_boot=500, seed=1)
    assert abs(r["comparisons"]["Tplus-T32@all10"]["delta"] - 0.1) < 1e-12
    # BAR — один вектор: d = среднее по сидам Tplus − BAR
    d = dmap(Y, YB, "Tplus", "BAR", (0,))[0]
    assert np.allclose(d, (Y["Tplus"][0][0].astype(float) +
                           Y["Tplus"][1][0]) / 2 - YB[0])
    # общие индексы: детерминизм и = прямому расчёту при неравных задачах
    dm = {"a": {t: rng.random(100 if t in ADAPTED else 20) for t in ALL}}
    b1 = boot(dm, ALL, 300, 9)
    rr = np.random.default_rng(9)
    man = np.zeros(300)
    for t in ALL:
        nt = len(dm["a"][t])
        man += dm["a"][t][rr.integers(0, nt, (300, nt))].mean(1)
    assert np.allclose(man / 10, b1["a"])
    print("самопроверка k36c_analyze пройдена: семья C (последовательность "
          "1 -> 2 -> 3, неоткрытые — описательно), all10 с равными весами "
          "задач, общий вектор BAR, общие индексы бутстрэпа при неравных "
          "задачах")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-36 C: анализ")
    ap.add_argument("cmd", choices=("analyze",))
    a = ap.parse_args()
    try:
        return analyze_c(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
