#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-34: анализ основной исследовательской оценки (init 30–44, два сида).

Каждый блок проверяется заново (k34_protocol.block_problems), начальные
хеши всех рук совпадают; ровно 150 состояний (15 на задачу) на руку.
1) T: SR выходов 8/16/24 на каждом сиде; T8/T16 против T24 — точный
   McNemar, Holm-4 (2 сравнения × 2 сида);
2) T против F на одинаковых глубинах — McNemar, отдельное Holm-6;
3) T24 − R_T24, F24 − R_F24 и прочее — описательно.
Joint bootstrap: resample init внутри каждой задачи, одни и те же индексы
для всех рук и обоих сидов; 100 000 повторов, seed 34011; интервалы по
каждому сиду и совместный (условно на двух обучениях). Незначимость не
объявляется равенством/не-хужестью. Задержка — reports/k34/latency_eval.json
(иначе latency_pending). Сохраняется полная таблица исходов.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT = "reports/k34/analysis.json"
BOOT_N, BOOT_SEED = 100_000, 34011
N_TASK, N_INIT = 10, 15


def mcnemar_exact(b, c):
    import math
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1))
                     / (1 << n)))


def holm(ps):
    ps = np.asarray(ps, np.float64)
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj.tolist()


def labels(seed):
    import k34_policy as kp
    out = {}
    for m, s_ in (("parents_attn24", "T"), ("flat_attn24", "F")):
        for d in (8, 16, 24):
            out[f"{s_}{d}"] = kp.label(m, seed, d)
        out[f"R_{s_}24"] = kp.label(m, seed)
    return out


def boot_sr(Y, n=BOOT_N, seed=BOOT_SEED, chunk=20000):
    """Y: {arm: [10, 15] bool}. Одни и те же resample-индексы init внутри
    задачи для всех рук. -> {arm: [n]} SR с равными весами задач."""
    rng = np.random.default_rng(seed)
    arms = sorted(Y)
    A = np.stack([np.asarray(Y[a], bool) for a in arms])    # [K, 10, 15]
    out = np.zeros((len(arms), n))
    done = 0
    while done < n:
        m = min(chunk, n - done)
        idx = rng.integers(0, N_INIT, (m, N_TASK, N_INIT))
        acc = np.zeros((len(arms), m))
        for t in range(N_TASK):
            acc += A[:, t, :][:, idx[:, t, :]].mean(-1)
        out[:, done:done + m] = acc / N_TASK
        done += m
    return {a: out[i] for i, a in enumerate(arms)}


def compare(Y, boot, x, y):
    X, Z = np.asarray(Y[x], bool).ravel(), np.asarray(Y[y], bool).ravel()
    b, c = int((X & ~Z).sum()), int((~X & Z).sum())
    d = float(np.mean(Y[x]) - np.mean(Y[y]))
    db = boot[x] - boot[y]
    return dict(a=x, b=y, sr_a=float(np.mean(Y[x])), sr_b=float(np.mean(Y[y])),
                delta=d, a_only=b, b_only=c, p_mcnemar=mcnemar_exact(b, c),
                ci95=[float(np.quantile(db, 0.025)),
                      float(np.quantile(db, 0.975))])


def analyze(Y):
    """Y: {'s0:T8': [10, 15] bool, ...}."""
    boot = boot_sr(Y)
    fam4, fam6, desc = [], [], []
    for s in (0, 1):
        for d in (8, 16):
            fam4.append(compare(Y, boot, f"s{s}:T{d}", f"s{s}:T24"))
        for d in (8, 16, 24):
            fam6.append(compare(Y, boot, f"s{s}:T{d}", f"s{s}:F{d}"))
        for k in ("T", "F"):
            desc.append(compare(Y, boot, f"s{s}:{k}24", f"s{s}:R_{k}24"))
        for d in (8, 16):
            desc.append(compare(Y, boot, f"s{s}:F{d}", f"s{s}:F24"))
    for fam in (fam4, fam6):
        for r, pa in zip(fam, holm([r["p_mcnemar"] for r in fam])):
            r["p_holm"] = pa
            r["verdict"] = ("значимо выше" if pa < 0.05 and r["delta"] > 0
                            else "значимо ниже" if pa < 0.05 and
                            r["delta"] < 0 else
                            "различие не установлено (не равенство)")
    joint = {}
    for tag, pairs in (("T8-T24", [("T8", "T24")]), ("T16-T24", [("T16",
                                                                   "T24")]),
                       ("T8-F8", [("T8", "F8")]), ("T16-F16", [("T16", "F16")]),
                       ("T24-F24", [("T24", "F24")])):
        x, y = pairs[0]
        jd = np.mean([boot[f"s{s}:{x}"] - boot[f"s{s}:{y}"] for s in (0, 1)],
                     0)
        joint[tag] = dict(delta=float(np.mean(
            [np.mean(Y[f"s{s}:{x}"]) - np.mean(Y[f"s{s}:{y}"])
             for s in (0, 1)])), ci95=[float(np.quantile(jd, 0.025)),
                                       float(np.quantile(jd, 0.975))])
    sr = {k: float(np.mean(v)) for k, v in Y.items()}
    return dict(sr=sr, holm4=fam4, holm6=fam6, descriptive=desc,
                joint=joint, note="совместный интервал условен на двух "
                                  "обучениях; незначимость не равенство")


def collect(plan, plan_file, check=True):
    import k34_protocol as pc
    prob = []
    Y = {}
    for s in (0, 1):
        for k, lab in labels(s).items():
            Y[f"s{s}:{k}"] = np.zeros((N_TASK, N_INIT), bool)
    seen = {k: set() for k in Y}
    rev = {lab: f"s{s}:{k}" for s in (0, 1) for k, lab in labels(s).items()}
    for lab, t, b in plan["cells"]:
        f = pc.block_path("eval", lab, t, b)
        if not os.path.exists(f):
            prob.append(f"нет блока {lab} t{t} i{b}")
            continue
        if check:
            pr = pc.block_problems(plan, plan_file, "eval", lab, t, b, f)
            if pr:
                prob += [f"{lab} t{t} i{b}: {x}" for x in pr[:2]]
                continue
        for e in json.load(open(f, encoding="utf-8"))["episodes"]:
            i = int(e["init_state_id"]) - 30
            key = rev[lab]
            if (t, i) in seen[key]:
                prob.append(f"{lab} t{t}: init {i + 30} повторён")
            seen[key].add((t, i))
            Y[key][t, i] = bool(e["success"])
    for k, v in seen.items():
        if len(v) != N_TASK * N_INIT:
            prob.append(f"{k}: состояний {len(v)} из 150")
    if check:
        prob += pc.parity_problems(plan, "eval")
    return Y, prob


def main():
    if "--selftest" in sys.argv:
        return selftest()
    import k34_protocol as pc
    pf = pc.plan_path("eval")
    if not os.path.exists(pf):
        print(f"  ОТКАЗ: нет плана {pf}")
        return 3
    plan = json.load(open(pf, encoding="utf-8"))
    Y, prob = collect(plan, pf)
    if prob:
        print(f"  ОТКАЗ: неполные или непроверенные данные ({len(prob)}): "
              + "; ".join(prob[:5]))
        json.dump(dict(ok=False, problems=prob[:200]),
                  open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=1)
        return 3
    res = analyze(Y)
    lat = "reports/k34/latency_eval.json"
    res["latency"] = json.load(open(lat, encoding="utf-8")) \
        if os.path.exists(lat) else "latency_pending"
    res["outcomes"] = {k: v.astype(int).tolist() for k, v in Y.items()}
    res.update(ok=True, boot=dict(n=BOOT_N, seed=BOOT_SEED))
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    for fam, nm in ((res["holm4"], "Holm-4"), (res["holm6"], "Holm-6")):
        for r in fam:
            print(f"  [{nm}] {r['a']} − {r['b']}: {100 * r['sr_a']:.1f} − "
                  f"{100 * r['sr_b']:.1f} = {100 * r['delta']:+.1f} п.п. "
                  f"[{100 * r['ci95'][0]:+.1f}; {100 * r['ci95'][1]:+.1f}], "
                  f"p {r['p_mcnemar']:.4f}, Holm {r['p_holm']:.4f} — "
                  f"{r['verdict']}")
    print(f"  отчёт {OUT}")
    return 0


def selftest():
    assert abs(mcnemar_exact(10, 0) - 2 / 1024) < 1e-15
    assert holm([0.01, 0.04, 0.03, 0.2]) == [0.04, 0.09, 0.09, 0.2]
    rng = np.random.default_rng(0)
    Y = {}
    for s in (0, 1):
        for k in ("T8", "T16", "T24", "F8", "F16", "F24", "R_T24", "R_F24"):
            Y[f"s{s}:{k}"] = rng.random((N_TASK, N_INIT)) < 0.8
    Y["s0:T8"] = np.zeros((N_TASK, N_INIT), bool)            # явно хуже
    Y["s1:T16"] = Y["s1:T24"].copy()                         # равны
    b = boot_sr(Y, n=3000, seed=1)
    b2 = boot_sr(Y, n=3000, seed=1)
    assert all(np.array_equal(b[k], b2[k]) for k in b)
    assert np.array_equal(b["s1:T16"], b["s1:T24"])          # общий resample
    assert abs(b["s0:T24"].mean() - Y["s0:T24"].mean()) < 0.01
    # явные индексы того же генератора = прямой расчёт для каждой руки
    idx = np.random.default_rng(5).integers(0, N_INIT, (1, N_TASK, N_INIT))
    b1 = boot_sr(Y, n=1, seed=5)
    for k in Y:
        direct = np.mean([Y[k][t, idx[0, t]].mean() for t in range(N_TASK)])
        assert abs(b1[k][0] - direct) < 1e-12, k
    res = analyze(Y)
    r0 = [r for r in res["holm4"] if r["a"] == "s0:T8"][0]
    assert r0["delta"] < 0 and r0["verdict"] == "значимо ниже"
    r1 = [r for r in res["holm4"] if r["a"] == "s1:T16"][0]
    assert r1["delta"] == 0 and r1["verdict"].startswith("различие не")
    assert len(res["holm4"]) == 4 and len(res["holm6"]) == 6
    print("самопроверка k34_analyze пройдена: McNemar, Holm-4 и Holm-6 "
          "отдельно, общий resample init для всех рук и сидов (детерминизм "
          "seed), значимое ухудшение маркируется, равные руки — «не "
          "установлено»")
    return 0


if __name__ == "__main__":
    sys.exit(main())
