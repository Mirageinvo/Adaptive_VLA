#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-35: описательный анализ SR-пилота K35-A.

A/C/N/S по 50 эпизодов (10 задач × init 20–24). Сравнения A−C и A−N —
парные таблицы выигрышей/потерь, точный McNemar, Holm-2; S — описательно
(C−S, A−S). 95% парные интервалы: resample init внутри задачи, общие
индексы для всех рук, 100 000 повторов, seed 35011; равные веса задач.
Многократно использованный development-банк, один обучающий сид, малый
объём: результат не подтверждающий, NI из незначимого p не выводится.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from k34_analyze import holm, mcnemar_exact  # noqa: E402

N_TASK, N_INIT, INIT0 = 10, 5, 20
BOOT_N, BOOT_SEED = 100_000, 35011
OUT = "reports/k35/analysis.json"
ARMS = ("A", "C", "N", "S")


def boot_sr(Y, n=BOOT_N, seed=BOOT_SEED, chunk=20000):
    rng = np.random.default_rng(seed)
    arms = sorted(Y)
    A = np.stack([np.asarray(Y[a], bool) for a in arms])    # [K, 10, 5]
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
    db = boot[x] - boot[y]
    return dict(a=x, b=y, succ_a=int(X.sum()), succ_b=int(Z.sum()),
                sr_a=float(X.mean()), sr_b=float(Z.mean()),
                delta=float(X.mean() - Z.mean()), a_only=b, b_only=c,
                p_mcnemar=mcnemar_exact(b, c),
                ci95=[float(np.quantile(db, 0.025)),
                      float(np.quantile(db, 0.975))])


def analyze(Y, n_boot=BOOT_N):
    boot = boot_sr(Y, n=n_boot)
    fam = [compare(Y, boot, "A", "C"), compare(Y, boot, "A", "N")]
    for r, pa in zip(fam, holm([r["p_mcnemar"] for r in fam])):
        r["p_holm"] = pa
        r["verdict"] = ("значимо выше" if pa < 0.05 and r["delta"] > 0 else
                        "значимо ниже" if pa < 0.05 and r["delta"] < 0 else
                        "различие не установлено (не равенство)")
    desc = [compare(Y, boot, "C", "S"), compare(Y, boot, "A", "S"),
            compare(Y, boot, "N", "S")]
    per_task = {a: np.asarray(Y[a], bool).sum(1).tolist() for a in ARMS}
    return dict(sr={a: float(np.mean(Y[a])) for a in ARMS},
                successes={a: int(np.sum(Y[a])) for a in ARMS},
                holm2=fam, descriptive=desc, per_task_successes_of_5=per_task,
                note="описательный пилот: development-банк, один сид, 50 "
                     "эпизодов на руку; незначимость не равенство")


def collect(plan, plan_file, check=True):
    import k35_protocol as pc
    prob = []
    Y = {a: np.zeros((N_TASK, N_INIT), bool) for a in ARMS}
    seen = {a: set() for a in ARMS}
    for lab, t, b in plan["cells"]:
        f = pc.block_path("pilot", lab, t, b)
        if not os.path.exists(f):
            prob.append(f"нет блока {lab} t{t} i{b}")
            continue
        if check:
            pr = pc.block_problems(plan, plan_file, "pilot", lab, t, b, f)
            if pr:
                prob += [f"{lab} t{t} i{b}: {x}" for x in pr[:2]]
                continue
        key = lab[-1]
        for e in json.load(open(f, encoding="utf-8"))["episodes"]:
            i = int(e["init_state_id"]) - INIT0
            if (t, i) in seen[key]:
                prob.append(f"{lab} t{t}: init {i + INIT0} повторён")
            seen[key].add((t, i))
            Y[key][t, i] = bool(e["success"])
    for k, v in seen.items():
        if len(v) != N_TASK * N_INIT:
            prob.append(f"{k}: состояний {len(v)} из 50")
    if check:
        prob += pc.parity_problems(plan, "pilot")
    return Y, prob


def offline_summary():
    import k35_oracle as ko
    import k35_protocol as pc
    out = {}
    for nm, f in (("oracle_gate", ko.GATE), ("selector_gate", pc.SEL_GATE),
                  ("selectors", pc.SEL_REPORT),
                  ("train", "reports/k35/train_s0.json"),
                  ("real_check", "reports/k35/real_check.json"),
                  ("fitcheck", "reports/k35/fitcheck.json"),
                  ("smoke", "reports/k35/smoke_check.json"),
                  ("pilot_gate", pc.GATE_FILE)):
        if os.path.exists(f):
            d = json.load(open(f, encoding="utf-8"))
            out[nm] = {k: d.get(k) for k in ("status", "problems", "metrics",
                                             "c_const", "best_update", "best",
                                             "micro", "ok", "drop",
                                             "successes", "details")
                       if k in d}
    return out


def main():
    if "--selftest" in sys.argv:
        return selftest()
    import k35_protocol as pc
    pf = pc.plan_path("pilot")
    res = dict(offline=offline_summary())
    lat = "reports/k35/latency.json"
    res["latency"] = json.load(open(lat, encoding="utf-8")) \
        if os.path.exists(lat) else "latency_pending"
    if not os.path.exists(pf):
        res.update(ok=True, pilot="не исполнялся")
        json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=1, default=float)
        print(f"  пилота нет; офлайн-сводка — {OUT}")
        return 0
    plan = json.load(open(pf, encoding="utf-8"))
    Y, prob = collect(plan, pf)
    if prob:
        print(f"  ОТКАЗ: неполные или непроверенные данные ({len(prob)}): "
              + "; ".join(prob[:5]))
        json.dump(dict(res, ok=False, problems=prob[:200]),
                  open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=1, default=float)
        return 3
    res.update(analyze(Y))
    res["outcomes"] = {k: v.astype(int).tolist() for k, v in Y.items()}
    res.update(ok=True, boot=dict(n=BOOT_N, seed=BOOT_SEED))
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=float)
    for r in res["holm2"] + res["descriptive"]:
        tag = "Holm-2" if "p_holm" in r else "описательно"
        print(f"  [{tag}] {r['a']} − {r['b']}: {r['succ_a']} − {r['succ_b']}"
              f" из 50 = {100 * r['delta']:+.1f} п.п. [{100 * r['ci95'][0]:+.1f};"
              f" {100 * r['ci95'][1]:+.1f}], только {r['a']} {r['a_only']} / "
              f"только {r['b']} {r['b_only']}, p {r['p_mcnemar']:.4f}" +
              (f", Holm {r['p_holm']:.4f} — {r['verdict']}" if "p_holm" in r
               else ""))
    print(f"  отчёт {OUT}")
    return 0


def selftest():
    rng = np.random.default_rng(0)
    Y = {a: rng.random((N_TASK, N_INIT)) < p for a, p in
         (("A", 0.9), ("C", 0.85), ("N", 0.8), ("S", 0.88))}
    r = analyze(Y, n_boot=2000)
    assert set(r["sr"]) == set(ARMS) and len(r["holm2"]) == 2
    for x in r["holm2"]:
        assert x["p_holm"] >= x["p_mcnemar"] and \
            x["ci95"][0] <= x["delta"] <= x["ci95"][1] + 1e-12
    # одинаковые руки: нулевая разница, p = 1, интервал вырожден в 0
    Z = {a: Y["A"].copy() for a in ARMS}
    r = analyze(Z, n_boot=500)
    assert all(x["delta"] == 0 and x["p_mcnemar"] == 1.0 and
               x["ci95"] == [0.0, 0.0] for x in r["holm2"])
    # общие индексы resample для всех рук: детерминизм seed
    b1, b2 = boot_sr(Y, n=300), boot_sr(Y, n=300)
    assert all(np.array_equal(b1[a], b2[a]) for a in ARMS)
    assert mcnemar_exact(0, 6) < 0.05 and mcnemar_exact(3, 3) == 1.0
    print("самопроверка k35_analyze пройдена: McNemar A−C / A−N, Holm-2, "
          "общие индексы resample по рукам, одинаковые руки — нулевая "
          "разница и p = 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
