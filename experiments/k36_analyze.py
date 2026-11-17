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
    # ресурсное правило перехода к B
    R = pc.RULE_A
    seeds_ok = all(per_seed[str(s)][k]["delta"] >= R["per_seed_floor"]
                   for s in (0, 1) for k in ("T32-P32", "T32-R"))
    rule = dict(joint_T32_P32=joint["T32-P32"]["delta"] >=
                R["joint_T32_P32"],
                joint_T32_R=joint["T32-R"]["delta"] >= R["joint_T32_R"],
                per_seed=seeds_ok)
    status = "passed_A_development" if all(rule.values()) else \
        "stopped_A_no_signal"
    return dict(status=status, rule=rule, rule_thresholds=R, family=family,
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
    """Решение A свежее и полное (для continue-b): файлы, prepare,
    пересчёт статуса по тем же данным."""
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
    for f, s in (g.get("files") or {}).items():
        if not os.path.exists(f) or pc.sha(f) != s:
            p.append("файлы A изменены после решения")
            break
    if not g.get("files") or len(g["files"]) != 2 * 1800:
        p.append("решение A не закрепляет все 1800 эпизодов")
    if not os.path.exists(OUT_A) or pc.sha(OUT_A) != g.get("analysis_sha1"):
        p.append("анализ A изменён после решения")
    if p:
        return p
    return [] if g.get("status") == "passed_A_development" else \
        [f"статус A: {g.get('status')}"]


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
    """Ожидаемая SE разницы при n_new случаев на задачу: по сидам и
    совместно (наблюдаемая корреляция сидов и консервативная r = 1)."""
    out = {}
    for x, y in (("T32", "P32"), ("T32", "R")):
        k = f"{x}-{y}"
        ds = {s: {t: Y[x][s][t].astype(float) - Y[y][s][t].astype(float)
                  for t in TASKS} for s in (0, 1)}
        var_s = {s: float(np.mean([ds[s][t].var(ddof=1) for t in TASKS]))
                 for s in (0, 1)}
        se_seed = {str(s): float(np.sqrt(var_s[s] / (len(TASKS) * n_new)))
                   for s in (0, 1)}
        a = np.concatenate([ds[0][t] for t in TASKS])
        b = np.concatenate([ds[1][t] for t in TASKS])
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 \
            else float("nan")
        vj = {}
        for nm, rr in (("observed", r), ("conservative_r1", 1.0)):
            rr = 0.0 if not np.isfinite(rr) else rr
            v = 0.25 * (var_s[0] + var_s[1] + 2 * rr * np.sqrt(
                var_s[0] * var_s[1]))
            vj[nm] = float(np.sqrt(v / (len(TASKS) * n_new)))
        q = {str(s): float(np.mean(np.concatenate(
            [np.abs(ds[s][t]) for t in TASKS]))) for s in (0, 1)}
        out[k] = dict(q_old=q, corr_seeds_old=r, se_seed=se_seed,
                      se_joint=vj,
                      halfwidth97_5_joint={kk: 2.241 * v for kk, v in
                                           vj.items()},
                      old_delta=dict({str(s): float(np.mean(
                          [ds[s][t].mean() for t in TASKS])) for s in (0, 1)}))
    return out


def cpu_audit(a):
    Y = k32_table()
    res = dict(n_new_per_task=100, source="K-32 SR, init 30…44, задачи 0/6/8 "
               "(q из K-33 не используется)", precision=precision(Y))
    pc.save_json(AUDIT, res)
    for k, v in res["precision"].items():
        print(f"  {k}: q K-32 {v['q_old']}, корреляция сидов "
              f"{v['corr_seeds_old']:.2f}; ожидаемая SE на сид "
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
    print("самопроверка k36_analyze пройдена: явный сигнал — оба теста "
          "семьи A и ресурсное правило; одинаковые руки — Δ 0, стоп, тест 2 "
          "не открывается; порог по сидам; общие индексы = прямому resample "
          "(в т. ч. неравные задачи); McNemar; консервативная SE ≥ "
          "наблюдаемой")
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
