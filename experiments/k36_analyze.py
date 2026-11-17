#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-36: анализ этапа A (§7) и CPU-аудит ожидаемой точности (§7.4).

Эстиманд: d_{t,i}^{X−Y} = ½ Σ_s (Y_{X,s,t,i} − Y_{Y,s,t,i});
Δ = ⅓ Σ_{t∈{0,6,8}} mean_i d_{t,i}. Бутстрэп: внутри каждой задачи
случаи с возвращением; ОДНИ индексы для всех рук и обоих сидов (и для обоих
сравнений), затем средние задач и macro-среднее; 100 000 повторов, seed
36011. Решение — двусторонний 97.5% интервал (квантили 0.0125 / 0.9875),
подтверждение — нижняя граница > 0; отдельно обычный 95%.
Семья A — фиксированная последовательность: 1) T32−P32; 2) T32−R
(формально — только если 1 подтверждён; иначе описательно).
Ресурсное правило перехода к B (не тест): Δ(T32−P32) ≥ 0.03, Δ(T32−R) ≥
0.03, на каждом сиде обе точечные разницы ≥ −0.02, полнота и проверки —
по неокруглённым величинам. Статус: passed_A_development /
stopped_A_no_signal; код 0 при корректных данных любого знака, 3 —
неполные/повреждённые данные.
По сидам и задачам — SR, wins/losses, q, точный McNemar, 95% парный
интервал (описательно).
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k36_protocol as pc  # noqa: E402

OUT_A = "reports/k36/analysis_a.json"
GATE_A = "reports/k36/gate_a.json"
AUDIT = "reports/k36/cpu_audit.json"
TASKS = (0, 6, 8)
NAMES = ("R", "T32", "P32")
COMPS = (("T32", "P32"), ("T32", "R"), ("P32", "R"))


def mcnemar_exact(b, c):
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1))
                     / (1 << n)))


def joint_d(Y, x, y, seeds=(0, 1)):
    """Y[name][s][t] — массив [n_t] bool -> {t: d_i [n_t]} (среднее по
    сидам)."""
    return {t: np.mean([Y[x][s][t].astype(float) - Y[y][s][t].astype(float)
                        for s in seeds], 0) for t in TASKS}


def macro(dmap):
    return float(np.mean([dmap[t].mean() for t in TASKS]))


def boot(dmaps, n, seed, chunk=10_000):
    """dmaps: {имя: {t: d [n_t]}} — общие индексы для всех имён.
    -> {имя: [n] macro-средних}."""
    rng = np.random.default_rng(seed)
    out = {k: np.zeros(n) for k in dmaps}
    done = 0
    while done < n:
        m = min(chunk, n - done)
        acc = {k: np.zeros(m) for k in dmaps}
        for t in TASKS:
            nt = len(next(iter(dmaps.values()))[t])
            idx = rng.integers(0, nt, (m, nt))
            for k, dm in dmaps.items():
                acc[k] += dm[t][idx].mean(1)
        for k in dmaps:
            out[k][done:done + m] = acc[k] / len(TASKS)
        done += m
    return out


def ci(v, level):
    a = (1 - level) / 2
    return [float(np.quantile(v, a)), float(np.quantile(v, 1 - a))]


def resource_rule(Y):
    """Ресурсное правило перехода к B — только по точечным величинам
    (неокруглённым), без бутстрэпа. -> (статус, флаги)."""
    R = pc.RULE_A
    j = {k: macro(joint_d(Y, *k.split("-"))) for k in ("T32-P32", "T32-R")}
    ps = {s: {k: macro(joint_d(Y, *k.split("-"), seeds=(s,)))
              for k in ("T32-P32", "T32-R")} for s in (0, 1)}
    rule = dict(joint_T32_P32=bool(j["T32-P32"] >= R["joint_T32_P32"]),
                joint_T32_R=bool(j["T32-R"] >= R["joint_T32_R"]),
                per_seed=bool(all(v >= R["per_seed_floor"] for s in (0, 1)
                                  for v in ps[s].values())))
    st = "passed_A_development" if all(rule.values()) else \
        "stopped_A_no_signal"
    return st, rule


def analyze(Y, n_boot=None, seed=None):
    st = pc.STATS
    n_boot = n_boot or st["boot_n"]
    seed = st["boot_seed_A"] if seed is None else seed
    dm = {f"{x}-{y}": joint_d(Y, x, y) for x, y in COMPS}
    bs = boot(dm, n_boot, seed)
    joint = {k: dict(delta=macro(v), ci97_5=ci(bs[k], st["level_decision"]),
                     ci95=ci(bs[k], st["level_report"]))
             for k, v in dm.items()}
    per_seed = {}
    for s in (0, 1):
        dms = {f"{x}-{y}": joint_d(Y, x, y, seeds=(s,)) for x, y in COMPS}
        bss = boot(dms, n_boot, [seed, s])
        per_seed[str(s)] = {}
        for x, y in COMPS:
            k = f"{x}-{y}"
            X = np.concatenate([Y[x][s][t] for t in TASKS])
            Z = np.concatenate([Y[y][s][t] for t in TASKS])
            b, c = int((X & ~Z).sum()), int((~X & Z).sum())
            per_seed[str(s)][k] = dict(
                delta=macro(dms[k]), ci95=ci(bss[k], st["level_report"]),
                wins=b, losses=c, q=(b + c) / len(X),
                p_mcnemar=mcnemar_exact(b, c),
                per_task={str(t): dict(
                    sr_x=float(Y[x][s][t].mean()),
                    sr_y=float(Y[y][s][t].mean()),
                    wins=int((Y[x][s][t] & ~Y[y][s][t]).sum()),
                    losses=int((~Y[x][s][t] & Y[y][s][t]).sum()))
                    for t in TASKS})
    sr = {f"{nm}_s{s}": {str(t): float(Y[nm][s][t].mean()) for t in TASKS}
          for nm in NAMES for s in (0, 1)}
    # фиксированная последовательность семьи A
    t1 = joint["T32-P32"]["ci97_5"][0] > 0
    t2 = t1 and joint["T32-R"]["ci97_5"][0] > 0
    family = dict(
        test1=dict(comparison="T32-P32", confirmed=bool(t1)),
        test2=dict(comparison="T32-R", formally_tested=bool(t1),
                   confirmed=bool(t2),
                   note=None if t1 else "описательно: тест 1 не подтверждён"))
    status, rule = resource_rule(Y)
    return dict(status=status, rule=rule, rule_thresholds=pc.RULE_A,
                family=family,
                joint=joint, per_seed=per_seed, sr=sr,
                boot=dict(n=n_boot, seed=seed, shared_indices="все руки и "
                          "оба сида на случае задачи"),
                estimand="эффект двух фиксированных обученных моделей на "
                         "распределении сцен этих задач (не среднее по "
                         "обучающим сидам)")


def collect_a():
    """-> (Y, проблемы, SHA файлов). Полный ожидаемый набор."""
    import k36_harness as kh
    prep = pc.load_prepare()
    C = kh.consts(prep)
    p = kh.checks_problems(prep)
    Y = {nm: {s: {} for s in (0, 1)} for nm in NAMES}
    files = {}
    cells = pc.cells_a(prep["splits"])
    blocks = {(lab, t, g): cases for lab, t, g, cases in cells}
    for (lab, t, g), cases in blocks.items():
        pr = kh.block_problems(prep, lab, t, g, cases, C)
        if pr:
            p.append(f"{lab} t{t} b{g}: {pr[:2]}")
    for lab in pc.ARMS_A:
        nm, _k, s = pc.parse_arm(lab)
        for t in TASKS:
            A = prep["splits"][str(t)]["A"]
            y = np.zeros(len(A), bool)
            for i, cid in enumerate(A):
                w = kh.a_want(prep, lab, t, cid)
                got = kh.load_ep(w)
                if got is None:
                    p.append(f"{lab} t{t} {cid}: нет эпизода")
                    continue
                y[i] = bool(got[0]["summary"]["success"])
                files.update(kh.ep_files(w))
            Y[nm][s][t] = y
    return Y, p, files, prep


def analyze_a(a):
    try:
        Y, p, files, prep = collect_a()
    except SystemExit as e:
        p, files, prep = [str(e.code)], {}, None
    if p:
        pc.save_json(OUT_A, dict(status="technical_stop", problems=p[:50]))
        print(f"  анализ A: ТЕХНИЧЕСКИЙ СТОП — неполные/непроверенные "
              f"данные ({len(p)}): " + "; ".join(p[:4]))
        return 3
    res = analyze(Y)
    fh = hashlib.sha1(json.dumps(sorted(files.items())).encode()).hexdigest()
    res.update(prepare_sha1=prep["sha1"], files_sha1=fh, n_files=len(files),
               table={f"{nm}_s{s}": {str(t): Y[nm][s][t].astype(int).tolist()
                                     for t in TASKS}
                      for nm in NAMES for s in (0, 1)},
               case_ids={str(t): prep["splits"][str(t)]["A"] for t in TASKS})
    pc.save_json(OUT_A, res)
    pc.save_json(GATE_A, dict(status=res["status"], rule=res["rule"],
                              family=res["family"],
                              joint={k: v["delta"] for k, v in
                                     res["joint"].items()},
                              prepare_sha1=prep["sha1"], files=files,
                              analysis_sha1=pc.sha(OUT_A)))
    for k in ("T32-P32", "T32-R", "P32-R"):
        j = res["joint"][k]
        print(f"  A {k}: Δ {100 * j['delta']:+.1f} п.п.; 97.5% "
              f"[{100 * j['ci97_5'][0]:+.1f}; {100 * j['ci97_5'][1]:+.1f}], "
              f"95% [{100 * j['ci95'][0]:+.1f}; {100 * j['ci95'][1]:+.1f}]; "
              f"по сидам " + " / ".join(
                  f"{100 * res['per_seed'][str(s)][k]['delta']:+.1f} "
                  f"({res['per_seed'][str(s)][k]['wins']}:"
                  f"{res['per_seed'][str(s)][k]['losses']}, p "
                  f"{res['per_seed'][str(s)][k]['p_mcnemar']:.3f})"
                  for s in (0, 1)))
    print(f"  семья A: тест 1 (T32−P32) "
          f"{'подтверждён' if res['family']['test1']['confirmed'] else 'не подтверждён'}"
          f"; тест 2 (T32−R) "
          f"{'подтверждён' if res['family']['test2']['confirmed'] else ('не подтверждён' if res['family']['test2']['formally_tested'] else 'описательно')}"
          f"; ресурсный статус {res['status']} {res['rule']}")
    return 0


def gate_a_problems():
    """Решение A свежее и полное (для continue-b): точный набор 1800
    эпизодов и их SHA, prepare, анализ не изменён, статус анализа и решения
    совпадают с ресурсным правилом, пересчитанным по исходам эпизодов."""
    import k36_harness as kh
    if not os.path.exists(GATE_A):
        return ["нет решения A"]
    g = json.load(open(GATE_A, encoding="utf-8"))
    try:
        prep = pc.load_prepare()
    except SystemExit as e:
        return [str(e.code)]
    p = []
    if g.get("prepare_sha1") != prep["sha1"]:
        p.append("решение A другого prepare")
    files = g.get("files") or {}
    if set(files) != kh.a_files(prep):
        p.append(f"решение A: набор файлов ≠ ожидаемым 1800 эпизодам "
                 f"({len(files)} файлов)")
    for f, sh in files.items():
        if not os.path.exists(f) or pc.sha(f) != sh:
            p.append("файлы A изменены после решения")
            break
    if not os.path.exists(OUT_A) or pc.sha(OUT_A) != g.get("analysis_sha1"):
        p.append("анализ A изменён после решения")
    if p:
        return p
    try:
        Y, pr, _f, _prep = collect_a()
    except SystemExit as e:
        return [f"сбор A: {e.code}"]
    if pr:
        return [f"данные A: {pr[:3]}"]
    st, rule = resource_rule(Y)
    an = json.load(open(OUT_A, encoding="utf-8"))
    if an.get("status") != st or g.get("status") != st or \
            g.get("rule") != rule or an.get("rule") != rule:
        return [f"статус решения/анализа ≠ пересчёту по исходам ({st})"]
    return [] if st == "passed_A_development" else [f"статус A: {st}"]


# --- CPU-аудит точности (§7.4) ---------------------------------------------------------
def k32_table():
    """Старая joint-таблица K-32: R/T/P × сиды × задачи 0/6/8 × init
    30…44 (одни сцены для обоих сидов)."""
    import k32_harness as kh32
    import k32_protocol as pc32
    Y = {nm: {s: {} for s in (0, 1)} for nm in NAMES}
    for kind, nm in (("R", "R"), ("T", "T32"), ("P", "P32")):
        for s in (0, 1):
            plan = pc32.load_eval_plan_kind(kind, s)
            o = kh32.outcomes(plan, kh32.label(kind, s), tasks=TASKS)
            for t in TASKS:
                Y[nm][s][t] = np.array([o[(t, i)] for i in range(30, 45)])
    return Y


def precision(Y, n_new=100):
    """Ожидаемая SE разницы при n_new случаев на задачу (§7.4). Совместно —
    внутри каждой задачи: v_t = Var_i((d0 + d1)/2), SE = √(Σ_t v_t/n)/3
    (наблюдаемая зависимость сидов); консервативно — r = 1 внутри задачи:
    v_t = ((sd0_t + sd1_t)/2)². По сиду: √(Σ_t Var_t(d_s)/n)/3."""
    out = {}
    T = len(TASKS)
    for x, y in (("T32", "P32"), ("T32", "R")):
        k = f"{x}-{y}"
        ds = {s: {t: Y[x][s][t].astype(float) - Y[y][s][t].astype(float)
                  for t in TASKS} for s in (0, 1)}
        se_seed = {str(s): float(np.sqrt(sum(ds[s][t].var(ddof=1)
                                             for t in TASKS) / n_new) / T)
                   for s in (0, 1)}
        v_obs = sum(((ds[0][t] + ds[1][t]) / 2).var(ddof=1) for t in TASKS)
        v_con = sum(((ds[0][t].std(ddof=1) + ds[1][t].std(ddof=1)) / 2) ** 2
                    for t in TASKS)
        vj = dict(observed=float(np.sqrt(v_obs / n_new) / T),
                  conservative_r1=float(np.sqrt(v_con / n_new) / T))
        corr_t = {}
        for t in TASKS:
            a, b = ds[0][t], ds[1][t]
            corr_t[str(t)] = float(np.corrcoef(a, b)[0, 1]) \
                if a.std() > 0 and b.std() > 0 else None
        q = {str(s): float(np.mean(np.concatenate(
            [np.abs(ds[s][t]) for t in TASKS]))) for s in (0, 1)}
        out[k] = dict(q_old=q, corr_seeds_within_task=corr_t,
                      se_seed=se_seed, se_joint=vj,
                      halfwidth97_5_joint={kk: 2.241 * v for kk, v in
                                           vj.items()},
                      old_delta={str(s): float(np.mean(
                          [ds[s][t].mean() for t in TASKS]))
                          for s in (0, 1)})
    return out


def cpu_audit(a):
    Y = k32_table()
    res = dict(n_new_per_task=100, source="K-32 SR, init 30…44, задачи 0/6/8 "
               "(q из K-33 не используется)", precision=precision(Y))
    pc.save_json(AUDIT, res)
    for k, v in res["precision"].items():
        print(f"  {k}: q K-32 {v['q_old']}, корреляция сидов по задачам "
              f"{v['corr_seeds_within_task']}; ожидаемая SE на сид "
              f"{v['se_seed']}, совместно {v['se_joint']}; полуширина 97.5% "
              f"{ {kk: round(100 * x, 1) for kk, x in v['halfwidth97_5_joint'].items()} } п.п.")
    return 0


def selftest():
    rng = np.random.default_rng(0)
    n = 100

    def table(pt=0.85, pp=0.75, pr=0.72, corr=True):
        Y = {nm: {s: {} for s in (0, 1)} for nm in NAMES}
        for t in TASKS:
            base = rng.random(n)
            for s in (0, 1):
                u = base if corr else rng.random(n)
                Y["T32"][s][t] = u < pt
                Y["P32"][s][t] = u < pp
                Y["R"][s][t] = u < pr
        return Y
    # явный сигнал: тест 1 и 2 подтверждены, ресурсное правило пройдено
    r = analyze(table(), n_boot=4000, seed=1)
    assert r["family"]["test1"]["confirmed"] and \
        r["family"]["test2"]["confirmed"] and \
        r["status"] == "passed_A_development", r["rule"]
    assert r["joint"]["T32-P32"]["ci97_5"][0] <= r["joint"]["T32-P32"][
        "ci95"][0]
    # одинаковые руки: Δ = 0, интервал вырожден, статус — стоп
    Z = table(0.8, 0.8, 0.8)
    r = analyze(Z, n_boot=2000, seed=1)
    assert all(abs(v["delta"]) == 0 and v["ci97_5"] == [0.0, 0.0]
               for v in r["joint"].values())
    assert r["status"] == "stopped_A_no_signal" and \
        not r["family"]["test1"]["confirmed"] and \
        not r["family"]["test2"]["formally_tested"]
    # порог по сидам: на одном сиде T32 хуже P32 больше чем на 2 п.п.
    Y = table()
    for t in TASKS:
        Y["T32"][1][t] = Y["P32"][1][t].copy()
        Y["T32"][1][t][:5] = False
        Y["P32"][1][t][:5] = True
    r = analyze(Y, n_boot=2000, seed=1)
    assert not r["rule"]["per_seed"] and r["status"] == "stopped_A_no_signal"
    # общие индексы: прямой resample = макро по задачам; детерминизм seed
    dm = dict(a={t: rng.random(n) for t in TASKS})
    b1, b2 = boot(dm, 500, 7), boot(dm, 500, 7)
    assert np.array_equal(b1["a"], b2["a"])
    rr = np.random.default_rng(7)
    man = np.zeros(500)
    for t in TASKS:
        man += dm["a"][t][rr.integers(0, n, (500, n))].mean(1)
    assert np.allclose(man / 3, b1["a"])
    # неравное число случаев по задачам и коррелированные сиды — та же
    # формула (прямой расчёт против векторного)
    dm2 = dict(a={0: rng.random(100), 6: rng.random(20), 8: rng.random(20)})
    bb = boot(dm2, 300, 5)
    rr = np.random.default_rng(5)
    man = np.zeros(300)
    for t in TASKS:
        nt = len(dm2["a"][t])
        man += dm2["a"][t][rr.integers(0, nt, (300, nt))].mean(1)
    assert np.allclose(man / 3, bb["a"])
    assert mcnemar_exact(0, 6) < 0.05 and mcnemar_exact(3, 3) == 1.0
    # точность: консервативная r = 1 не меньше наблюдаемой
    Yo = {nm: {s: {t: rng.random(15) < 0.7 for t in TASKS} for s in (0, 1)}
          for nm in NAMES}
    pr = precision(Yo)
    for v in pr.values():
        assert v["se_joint"]["conservative_r1"] >= v["se_joint"][
            "observed"] - 1e-12
    # совместная SE: внутри задач, а не через общую корреляцию
    Yp = {nm: {s: {} for s in (0, 1)} for nm in NAMES}
    rg = np.random.default_rng(3)
    for t, off in ((0, 0.0), (6, 0.4), (8, -0.3)):
        base = rg.random(15)
        for s in (0, 1):
            u = base if s == 0 else rg.random(15)
            Yp["T32"][s][t] = u < 0.6 + off * (s == 0)
            Yp["P32"][s][t] = u < 0.5
            Yp["R"][s][t] = u < 0.45
    pv = precision(Yp, n_new=100)["T32-P32"]
    d = {s: {t: Yp["T32"][s][t].astype(float) - Yp["P32"][s][t]
             .astype(float) for t in TASKS} for s in (0, 1)}
    want = np.sqrt(sum(((d[0][t] + d[1][t]) / 2).var(ddof=1)
                       for t in TASKS) / 100) / 3
    assert abs(pv["se_joint"]["observed"] - want) < 1e-12
    # гейт A: статус решения пересчитывается по исходам; подделанный
    # passed при стоп-исходах, чужой набор файлов, правка анализа — отказ
    import tempfile
    import k36_harness as kh
    G = globals()
    keep = dict(OUT_A=G["OUT_A"], GATE_A=G["GATE_A"],
                collect_a=G["collect_a"])
    keep_lp, keep_af = pc.load_prepare, kh.a_files
    with tempfile.TemporaryDirectory() as td:
        try:
            fs = [os.path.join(td, f"e{i}.json") for i in range(4)]
            for f in fs:
                open(f, "w").write(f)
            G.update(OUT_A=os.path.join(td, "an.json"),
                     GATE_A=os.path.join(td, "gate.json"))
            pc.load_prepare = lambda: dict(sha1="PREP")
            kh.a_files = lambda prep: set(fs)
            for Ytab, want_st in ((table(), "passed_A_development"),
                                  (table(0.8, 0.8, 0.8),
                                   "stopped_A_no_signal")):
                G["collect_a"] = lambda Ytab=Ytab: (Ytab, [], {}, None)
                st, ru = resource_rule(Ytab)
                assert st == want_st
                pc.save_json(G["OUT_A"], dict(status=st, rule=ru))
                gate = dict(status=st, rule=ru, prepare_sha1="PREP",
                            files={f: pc.sha(f) for f in fs},
                            analysis_sha1=pc.sha(G["OUT_A"]))
                pc.save_json(G["GATE_A"], gate)
                pr = gate_a_problems()
                assert (pr == []) == (st == "passed_A_development"), pr
                if st == "stopped_A_no_signal":     # подделка passed
                    pc.save_json(G["OUT_A"], dict(
                        status="passed_A_development", rule=dict(
                            joint_T32_P32=True, joint_T32_R=True,
                            per_seed=True)))
                    pc.save_json(G["GATE_A"], dict(
                        gate, status="passed_A_development",
                        rule=dict(joint_T32_P32=True, joint_T32_R=True,
                                  per_seed=True),
                        analysis_sha1=pc.sha(G["OUT_A"])))
                    assert gate_a_problems()
            pc.save_json(G["GATE_A"], dict(gate, files={fs[0]: pc.sha(
                fs[0])}))
            assert gate_a_problems()                  # неполный набор
        finally:
            G.update(keep)
            pc.load_prepare, kh.a_files = keep_lp, keep_af
    print("самопроверка k36_analyze пройдена: явный сигнал — оба теста "
          "семьи A и ресурсное правило; одинаковые руки — Δ 0, стоп, тест 2 "
          "не открывается; порог по сидам; общие индексы = прямому resample "
          "(в т. ч. неравные задачи); McNemar; консервативная SE ≥ "
          "наблюдаемой; совместная SE — внутри задач (не через общую "
          "корреляцию); гейт A — статус пересчитывается по исходам, "
          "подделанный passed и неполный набор файлов — отказ")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-36: анализ")
    ap.add_argument("cmd", choices=("analyze-a", "cpu-audit", "gate-a"))
    a = ap.parse_args()
    try:
        if a.cmd == "gate-a":
            p = gate_a_problems()
            print("  решение A: " + ("passed_A_development, привязки "
                                     "сходятся" if not p else "; ".join(p)))
            return 0 if not p else 4
        return dict(**{"analyze-a": analyze_a, "cpu-audit": cpu_audit})[
            a.cmd](a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
