#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-33: анализ подтверждающей оценки на свежем банке (3000 сцен × 6 рук).

Вход: планы обеих машин (reports/k33/plan_A.json, plan_B.json — контракты
обязаны совпасть) и SR-блоки reports/k33/sr/<M>/<arm>/t<T>_i<S>.json. Каждый
блок проверяется заново (k33_eval.block_problems: 5 эпизодов, action SHA,
настройки, метаданные и журнал руки, post-wait хеши = банку). Требуется
ровно 3000 различных случаев на руку, 300 на задачу, у всех рук.

Основной вопрос (T − F), для каждого training seed s:
  успехи, SR с равными весами задач, ΔSR, rescue/harm, доля разногласий q,
  точный двусторонний McNemar по 3000 парам, Holm-2 по двум сидам (α 0.05),
  95% парный стратифицированный bootstrap ΔSR; совместный интервал по двум
  сидам — один и тот же resample случаев внутри задачи для всех шести рук.
Инженерное семейство (не основной вопрос): T0/T1/F0/F1 против B и BC —
  8 сравнений; NI при нижней односторонней bootstrap-границе (хвост α/8)
  > −3 п.п.; положительная точечная разность NI не заменяет.
Bootstrap: 6-битный совместный образец исходов сцены (до 64 значений);
  внутри задачи counts образцов пересэмплируются мультиномиально; все
  разности и оба сида считаются по одним counts. 100 000 повторов, seed
  33011. Код завершения зависит только от полноты и проверок, не от SR/p.
Задержка — из reports/k33/latency.json, иначе latency_pending.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k33_policy as kp  # noqa: E402

OUT = "reports/k33/analysis.json"
ARMS6 = kp.ORDER0                       # порядок битов образца
BITS = np.array([[(c >> k) & 1 for k in range(len(ARMS6))]
                 for c in range(2 ** len(ARMS6))], np.float64)   # [64, 6]
PAIRS_MAIN = (("T0", "F0", 0), ("T1", "F1", 1))
NI_PAIRS = tuple((x, y) for x in ("T0", "T1", "F0", "F1")
                 for y in ("B", "BC"))


# --- статистика ---------------------------------------------------------------------
def mcnemar_exact(b, c):
    """Точный двусторонний McNemar: биномиальный тест разногласий."""
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return float(min(1.0, (2 * tail) / (1 << n)))     # точное целое деление


def holm(pvals):
    """Скорректированные p по Holm (монотонные)."""
    p = np.asarray(pvals, np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj.tolist()


def patterns(Y):
    """Y [6, n] bool (руки ARMS6) -> коды образцов [n]."""
    Y = np.asarray(Y, np.int64)
    return (Y * (1 << np.arange(Y.shape[0]))[:, None]).sum(0)


def sr_from_counts(counts, n):
    """counts [..., 64] -> SR рук [..., 6]."""
    return counts @ BITS / float(n)


def bootstrap(Yt, n_boot, seed, chunk=20000):
    """Yt: {task: [6, n_t] bool}. -> SR с равными весами задач [n_boot, 6].
    Один и тот же resample случаев задачи для всех рук (через counts
    общих образцов)."""
    rng = np.random.default_rng(seed)
    tasks = sorted(Yt)
    acc = np.zeros((n_boot, len(ARMS6)))
    for t in tasks:
        n_t = Yt[t].shape[1]
        cnt = np.bincount(patterns(Yt[t]), minlength=64)
        p = cnt / n_t
        done = 0
        while done < n_boot:
            m = min(chunk, n_boot - done)
            draw = rng.multinomial(n_t, p, size=m)
            acc[done:done + m] += sr_from_counts(draw, n_t)
            done += m
    return acc / len(tasks)


def point_sr(Yt):
    tasks = sorted(Yt)
    return np.mean([np.asarray(Yt[t], np.float64).mean(1) for t in tasks], 0)


def main_rows(Yt, boot):
    """T − F по сидам: McNemar, Holm, интервалы; совместный интервал."""
    sr = point_sr(Yt)
    ix = {a: i for i, a in enumerate(ARMS6)}
    rows, ps = [], []
    for t_arm, f_arm, s in PAIRS_MAIN:
        T = np.concatenate([Yt[t][ix[t_arm]] for t in sorted(Yt)])
        F = np.concatenate([Yt[t][ix[f_arm]] for t in sorted(Yt)])
        b = int((T & ~F).sum())
        c = int((~T & F).sum())
        p = mcnemar_exact(b, c)
        ps.append(p)
        d = sr[ix[t_arm]] - sr[ix[f_arm]]
        db = boot[:, ix[t_arm]] - boot[:, ix[f_arm]]
        rows.append(dict(seed=s, T=t_arm, F=f_arm, n=int(len(T)),
                         succ_T=int(T.sum()), succ_F=int(F.sum()),
                         sr_T=float(sr[ix[t_arm]]), sr_F=float(sr[ix[f_arm]]),
                         delta=float(d), rescue=b, harm=c,
                         q=float((b + c) / len(T)), p_mcnemar=p,
                         ci95=[float(np.quantile(db, 0.025)),
                               float(np.quantile(db, 0.975))]))
    adj = holm(ps)
    for r, pa in zip(rows, adj):
        r["p_holm"] = pa
        sig = pa < 0.05
        r["verdict"] = ("преимущество дерева на этом сиде" if sig and
                        r["delta"] > 0 else
                        "ухудшение дерева на этом сиде" if sig and
                        r["delta"] < 0 else
                        "различие не установлено (не доказывает равенство)")
    jd = np.mean([boot[:, ix[t]] - boot[:, ix[f]] for t, f, _s in PAIRS_MAIN],
                 0)
    joint = dict(delta=float(np.mean([r["delta"] for r in rows])),
                 ci95=[float(np.quantile(jd, 0.025)),
                       float(np.quantile(jd, 0.975))],
                 note="совместный интервал — описательно; не третий способ "
                      "получить значимость")
    both = all(r["p_holm"] < 0.05 and r["delta"] > 0 for r in rows)
    return rows, joint, both


def ni_rows(Yt, boot, margin, tail):
    sr = point_sr(Yt)
    ix = {a: i for i, a in enumerate(ARMS6)}
    out = []
    for x, y in NI_PAIRS:
        d = boot[:, ix[x]] - boot[:, ix[y]]
        Xc = np.concatenate([Yt[t][ix[x]] for t in sorted(Yt)])
        Yc = np.concatenate([Yt[t][ix[y]] for t in sorted(Yt)])
        lower = float(np.quantile(d, tail))
        out.append(dict(arm=x, ref=y, delta=float(sr[ix[x]] - sr[ix[y]]),
                        rescue=int((Xc & ~Yc).sum()),
                        harm=int((~Xc & Yc).sum()), lower=lower,
                        tail=tail, margin=margin, ni=bool(lower > -margin)))
    return out


# --- сбор исходов ---------------------------------------------------------------------
def collect(plans, check=True):
    """-> ({task: [6, 300] bool}, проблемы)."""
    import k33_bank as kb
    import k33_eval as ke
    prob = []
    Y = {t: np.zeros((len(ARMS6), 300), bool) for t in range(kb.N_TASKS)}
    seen = {a: {t: set() for t in range(kb.N_TASKS)} for a in ARMS6}
    for m, (plan, pf) in plans.items():
        for t, arm, s in plan["queue_sr"]:
            f = ke.block_path(m, "sr", arm, t, s)
            if not os.path.exists(f):
                prob.append(f"нет блока {m} {arm} t{t} i{s}")
                continue
            if check:
                pr_ = ke.block_problems(plan, pf, "sr", arm, t, s, f)
                if pr_:
                    prob += [f"{m} {arm} t{t} i{s}: {x}" for x in pr_[:2]]
                    continue
            d = json.load(open(f, encoding="utf-8"))
            for e in d["episodes"]:
                i = int(e["init_state_id"])
                if i in seen[arm][t]:
                    prob.append(f"{arm} t{t}: случай {i} повторён")
                seen[arm][t].add(i)
                Y[t][ARMS6.index(arm), i] = bool(e["success"])
    for a in ARMS6:
        for t in range(kb.N_TASKS):
            if seen[a][t] != set(range(300)):
                prob.append(f"{a} t{t}: случаев {len(seen[a][t])} из 300")
    return Y, prob


def analyze(Y, stats):
    boot = bootstrap(Y, int(stats["boot_n"]), int(stats["boot_seed"]))
    rows, joint, both = main_rows(Y, boot)
    ni = ni_rows(Y, boot, float(stats["ni_margin"]), float(stats["ni_tail"]))
    sr = point_sr(Y)
    per_task = {str(t): {a: float(np.mean(Y[t][i]))
                         for i, a in enumerate(ARMS6)} for t in sorted(Y)}
    return dict(sr={a: float(sr[i]) for i, a in enumerate(ARMS6)},
                main=rows, joint=joint, advantage_both_seeds=both,
                ni=ni, per_task=per_task,
                note_tasks="t0/6/8 и прочие строки — описательно, не новый "
                           "основной тест",
                note_ni="NI-семейство инженерное; общего FWER с Holm-2 не "
                        "заявляется; NI и выигрыш flat не заменяют "
                        "свидетельства пользы дерева")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-33: анализ")
    ap.add_argument("--no-check", action="store_true",
                    help="не перепроверять блоки (только для отладки)")
    a = ap.parse_args()
    import k33_eval as ke
    plans = {}
    for m in ke.MACHINES:
        pf = ke.plan_path(m)
        if not os.path.exists(pf):
            print(f"  ОТКАЗ: нет плана {pf}")
            return 3
        plans[m] = (json.load(open(pf, encoding="utf-8")), pf)
    fps = {p["contract_fp"] for p, _f in plans.values()}
    if len(fps) != 1:
        print("  ОТКАЗ: контракты машин различаются")
        return 3
    Y, prob = collect(plans, check=not a.no_check)
    if prob:
        print(f"  ОТКАЗ: неполные или непроверенные данные ({len(prob)}): "
              + "; ".join(prob[:5]))
        json.dump(dict(ok=False, problems=prob[:200]),
                  open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=1)
        return 3
    stats = plans["A"][0]["stats"]
    res = analyze(Y, stats)
    lat = "reports/k33/latency.json"
    res["latency"] = (json.load(open(lat, encoding="utf-8"))
                      if os.path.exists(lat) else "latency_pending")
    res.update(ok=True, contract_fp=fps.pop(), stats=stats,
               n_cases=300 * len(Y))
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    for r in res["main"]:
        print(f"  {r['T']} − {r['F']}: SR {r['sr_T']:.3f} − {r['sr_F']:.3f} = "
              f"{100 * r['delta']:+.1f} п.п. [{100 * r['ci95'][0]:+.1f}; "
              f"{100 * r['ci95'][1]:+.1f}], rescue/harm {r['rescue']}/"
              f"{r['harm']}, q {r['q']:.3f}, McNemar p {r['p_mcnemar']:.4f}, "
              f"Holm {r['p_holm']:.4f} — {r['verdict']}")
    j = res["joint"]
    print(f"  совместно по сидам: {100 * j['delta']:+.1f} п.п. "
          f"[{100 * j['ci95'][0]:+.1f}; {100 * j['ci95'][1]:+.1f}]")
    for r in res["ni"]:
        print(f"  NI {r['arm']} к {r['ref']}: Δ {100 * r['delta']:+.1f} п.п., "
              f"нижняя граница {100 * r['lower']:+.1f} — "
              + ("не-хуже (маржа 3 п.п.)" if r["ni"] else "не-хужесть не "
                 "установлена"))
    print(f"  задержка: {'есть' if res['latency'] != 'latency_pending' else 'latency_pending'}; отчёт {OUT}")
    return 0


def selftest():
    # McNemar и Holm
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(10, 0) - 2 / 2 ** 10) < 1e-15
    assert abs(mcnemar_exact(3, 7) - mcnemar_exact(7, 3)) < 1e-15
    assert holm([0.01, 0.04]) == [0.02, 0.04]
    assert holm([0.04, 0.01]) == [0.04, 0.02]
    assert holm([0.03, 0.02]) == [0.04, 0.04]
    rng = np.random.default_rng(0)
    Y = {t: rng.random((6, 300)) < rng.uniform(0.6, 0.95, (6, 1))
         for t in range(10)}
    # агрегация по образцам = прямой расчёт на явных индексах
    idx = {t: rng.integers(0, 300, 300) for t in Y}
    direct = np.mean([Y[t][:, idx[t]].mean(1) for t in Y], 0)
    via = np.mean([sr_from_counts(np.bincount(patterns(Y[t][:, idx[t]]),
                                              minlength=64), 300) for t in Y],
                  0)
    assert np.allclose(direct, via, atol=1e-12), (direct, via)
    assert np.allclose(sr_from_counts(np.bincount(patterns(Y[0]),
                                                  minlength=64), 300),
                       Y[0].mean(1))
    # общий resample: разность рук из одних counts; детерминизм seed
    b1 = bootstrap(Y, 2000, 33011)
    b2 = bootstrap(Y, 2000, 33011)
    assert np.array_equal(b1, b2) and b1.shape == (2000, 6)
    assert abs(b1.mean(0) - point_sr(Y)).max() < 0.01
    # одинаковые руки -> разность ровно 0 в каждом повторе (общий resample)
    Yd = {t: np.vstack([Y[t][:1]] * 6) for t in Y}
    bd = bootstrap(Yd, 500, 1)
    assert np.all(bd[:, 0] == bd[:, 1])
    # вердикты: значимое ухудшение маркируется ухудшением; NI не из p > 0.05
    ix = {a: i for i, a in enumerate(ARMS6)}
    Yw = {t: Y[t].copy() for t in Y}
    for t in Yw:
        Yw[t][ix["T0"]] = True                   # T0 всегда успех
        Yw[t][ix["F0"]] = rng.random(300) < 0.5  # F0 хуже
        Yw[t][ix["T1"]] = rng.random(300) < 0.5  # T1 хуже F1
        Yw[t][ix["F1"]] = True
    res = analyze(Yw, dict(boot_n=2000, boot_seed=33011, ni_margin=0.03,
                           ni_tail=0.05 / 8))
    r0, r1 = res["main"]
    assert r0["delta"] > 0 and r0["verdict"].startswith("преимущество")
    assert r1["delta"] < 0 and r1["verdict"].startswith("ухудшение")
    assert res["advantage_both_seeds"] is False
    # NI: равные руки с шумом — граница ниже 0; NI определяется только границей
    for r in res["ni"]:
        assert r["ni"] == (r["lower"] > -0.03)
    Ye = {t: np.vstack([Y[t][1:2] if a in ("B", "BC") else Y[t][:1]
                        for a in ARMS6]) for t in Y}     # T = F на обоих сидах
    re_ = analyze(Ye, dict(boot_n=2000, boot_seed=33011, ni_margin=0.03,
                           ni_tail=0.05 / 8))
    assert all(r["verdict"].startswith("различие не установлено")
               for r in re_["main"])
    print("самопроверка k33_analyze пройдена: точный McNemar, Holm, "
          "агрегация по 6-битным образцам = прямому расчёту на явных "
          "индексах, общий resample всех рук и сидов (детерминизм seed), "
          "значимое ухудшение маркируется ухудшением, отсутствие "
          "значимости не равенство, NI только по нижней границе")
    return 0


if __name__ == "__main__":
    sys.exit(main())
