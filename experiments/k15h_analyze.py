#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15h: парный анализ свежего банка и выгрузка признаков.

Только проверенные блоки (k15h_protocol.check_run). Неполный набор —
код 3 и никакого научного вывода: печатается только покрытие.

ЕДИНИЦА — начальное состояние. Для каждой политики: SR по задачам и в
целом, главная оценка — среднее SR по задачам с равными весами; парная ΔSR
против q0ref, выигрыши и потери, число состояний, где импульс не был
достигнут (q0ref завершился раньше чанка 4; такой эпизод остаётся в SR).

ОСНОВНЫЕ СРАВНЕНИЯ (заданы планом): p4r3p − q0ref, learned − arm_dir,
learned − static. Точный двусторонний Мак-Немар, поправка Holm на три.
95% интервал ΔSR — парный бутстрэп, состояния пересэмплируются внутри
задачи, вектор исходов состояния берётся целиком. Вторичные: остальные
политики против q0ref, arm_dir − static, только достигнутые импульсы.

ИНТЕРПРЕТАЦИЯ — узкая: learned > arm_dir — польза формы поправки руки при
согласованной норме; arm_dir > static может отражать зависимость нормы
и/или схвата от состояния; отсутствие обнаруженного преимущества не
доказывает равенства; выигрыш p4r3p не доказывает ценность обучаемой
геометрии.

ПРИЗНАКИ (reports/k15h/features.npz) — по q0ref: h18, a0 и learned-базис
вызовов 1–4 с маской достижения вызова средой (по done_step; значения для
уже завершившейся среды помечены и в набор не входят), исходы пяти
политик, направление, задача, состояние.
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


def mcnemar_p(gain, loss):
    n = gain + loss
    if n == 0:
        return 1.0
    k = min(gain, loss)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1))
               / 2.0 ** n)


def holm(ps):
    """Поправка Holm: скорректированные p в исходном порядке."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def equal_weight(outc, states):
    """Среднее по задачам SR (равные веса задач)."""
    by = {}
    for k in states:
        by.setdefault(k[0], []).append(float(outc[k]))
    return float(np.mean([np.mean(v) for v in by.values()]))


def bootstrap_diff(a, b, states, n=10000, seed=20261013):
    """95% интервал ΔSR(a − b) с равными весами задач; состояния
    пересэмплируются внутри задачи (парно)."""
    rng = np.random.default_rng(seed)
    by = {}
    for k in states:
        by.setdefault(k[0], []).append(float(a[k]) - float(b[k]))
    tasks = sorted(by)
    arrs = [np.asarray(by[t]) for t in tasks]
    est = np.zeros(n)
    for x in arrs:
        idx = rng.integers(0, len(x), size=(n, len(x)))
        est += x[idx].mean(1)
    est /= len(arrs)
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def compare(a, b, states, boot_n, boot_seed):
    gain = sum(1 for k in states if a[k] and not b[k])
    loss = sum(1 for k in states if b[k] and not a[k])
    return dict(d_sr_equal_weight=equal_weight(a, states)
                - equal_weight(b, states),
                d_sr_pooled=(sum(map(float, (a[k] for k in states)))
                             - sum(map(float, (b[k] for k in states))))
                / len(states),
                gain=gain, loss=loss, mcnemar_p=mcnemar_p(gain, loss),
                ci95=bootstrap_diff(a, b, states, boot_n, boot_seed))


def summarize(outc, done, states, plan):
    """outc {метка: {(t, s): успех}}, done {(t, s): done_step q0ref}."""
    import k15h_protocol as pc
    q0 = outc["q0ref"]
    unreached = [k for k in states if not pc.reached(int(done[k]),
                                                     pc.MOMENT)]
    rows = {}
    for lab, o in outc.items():
        per_task = {}
        for k in states:
            v = per_task.setdefault(str(k[0]), [0, 0, 0, 0])
            v[0] += bool(o[k])
            v[1] += 1
            v[2] += bool(o[k] and not q0[k])
            v[3] += bool(q0[k] and not o[k])
        rows[lab] = dict(
            sr_pooled=float(np.mean([o[k] for k in states])),
            sr_equal_weight=equal_weight(o, states),
            d_sr=equal_weight(o, states) - equal_weight(q0, states),
            gain=sum(v[2] for v in per_task.values()),
            loss=sum(v[3] for v in per_task.values()),
            per_task=per_task)
    a_ = plan["analysis"]
    prim = []
    for x, y in a_["primary"]:
        prim.append(dict(a=x, b=y, **compare(outc[x], outc[y], states,
                                             a_["bootstrap_n"],
                                             a_["bootstrap_seed"])))
    adj = holm([c["mcnemar_p"] for c in prim])
    for c, p_ in zip(prim, adj):
        c["holm_p"] = p_
    sec = {}
    for lab in outc:
        if lab != "q0ref":
            sec[f"{lab} − q0ref"] = compare(outc[lab], q0, states,
                                            a_["bootstrap_n"],
                                            a_["bootstrap_seed"])
    sec["arm_dir_uniform − static_uniform"] = compare(
        outc["arm_dir_uniform"], outc["static_uniform"], states,
        a_["bootstrap_n"], a_["bootstrap_seed"])
    reach = [k for k in states if k not in set(unreached)]
    sec_reached = {}
    if reach:
        for lab in outc:
            if lab != "q0ref":
                g = sum(1 for k in reach if outc[lab][k] and not q0[k])
                lo = sum(1 for k in reach if q0[k] and not outc[lab][k])
                sec_reached[lab] = dict(states=len(reach), gain=g, loss=lo,
                                        mcnemar_p=mcnemar_p(g, lo))
    return dict(states=len(states), unreached=len(unreached),
                unreached_states=[list(k) for k in unreached], rows=rows,
                primary=prim, secondary=sec,
                secondary_reached_only=sec_reached,
                interpretation=__doc__.split("ИНТЕРПРЕТАЦИЯ")[1]
                .split("ПРИЗНАКИ")[0].strip())


def export_features(plan, ok, outc, out):
    """Признаки по q0ref + исходы пяти политик."""
    import k15h_protocol as pc
    H18, A0, LB, MASK, T, S, DIR, Y = [], [], [], [], [], [], [], []
    for (lab, t, b), pth in sorted(ok.items()):
        if lab != "q0ref":
            continue
        d, z = pc.load(pth)
        h = z["k15h_h18"]                                 # [k, B, n, D]
        for i, e in enumerate(d["episodes"]):
            s = int(e["init_state_id"])
            ds = int(z["done_step"][i])
            mask = [pc.reached(ds, k + 1) for k in range(h.shape[0])]
            hh = np.where(np.asarray(mask)[:, None, None], h[:, i],
                          np.nan).astype(np.float32)
            H18.append(hh)
            A0.append(np.where(np.asarray(mask)[:, None, None],
                               z["k15d_a0"][:h.shape[0], i], np.nan))
            LB.append(np.where(np.asarray(mask)[:, None, None],
                               z["k15f_basis"][:h.shape[0], i], np.nan))
            MASK.append(mask)
            T.append(t)
            S.append(s)
            DIR.append(plan["directions"][str(t)][str(s)])
            Y.append([bool(outc[x][(t, s)]) for x in plan["labels"]])
    np.savez_compressed(out, h18=np.stack(H18), a0=np.stack(A0),
                        learned_basis=np.stack(LB), call_reached=np.asarray(
                            MASK), task=np.asarray(T), state=np.asarray(S),
                        direction=np.asarray(DIR), outcomes=np.asarray(Y),
                        labels=np.asarray(plan["labels"]))
    return len(T)


def run(plan_path, run_dir, out, features):
    import k15h_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan, plan_path)
    if prob:
        raise SystemExit("входы плана K-15h не совпадают: " + "; ".join(
            prob[:4]))
    ok, gaps = pc.check_run(plan, plan_path, run_dir)
    states = sorted((int(t), s) for t, bl in plan["blocks"].items()
                    for b in bl for s in range(b, b + plan["n_envs"]))
    want = len(states) // plan["n_envs"] * len(plan["labels"])
    res = dict(plan=plan_path, run_dir=run_dir, verified=len(ok),
               planned=want, gaps=gaps, complete=(len(ok) == want
                                                  and not gaps))
    if not res["complete"]:
        return res
    outc, done = {lab: {} for lab in plan["labels"]}, {}
    for (lab, t, b), pth in ok.items():
        d = json.load(open(pth))
        for e in d["episodes"]:
            k = (t, int(e["init_state_id"]))
            outc[lab][k] = bool(e["success"])
            if lab == "q0ref":
                done[k] = int(e["done_step"])
    res.update(summarize(outc, done, states, plan))
    if features and not plan.get("smoke"):
        res["features"] = dict(path=features, episodes=export_features(
            plan, ok, outc, features))
    return res


def report(res):
    print(f"  проверенных блоков {res['verified']}/{res['planned']}; "
          f"непригодных или отсутствующих {len(res['gaps'])}")
    for g in [g for g in res["gaps"] if g["why"] != "не готов"][:10]:
        print(f"    ВНИМАНИЕ {g['label']} t{g['task']} i{g['block']}: "
              f"{g['why']}")
    if not res["complete"]:
        print("  НАБОР НЕПОЛНЫЙ: научного вывода нет")
        return
    print(f"  состояний {res['states']}; импульс не достигнут в "
          f"{res['unreached']}")
    for lab, r in res["rows"].items():
        pt = " ".join(f"t{t} {v[0]}/{v[1]}" for t, v in r["per_task"].items())
        print(f"    {lab:16} SR {100 * r['sr_equal_weight']:5.1f}% (равные "
              f"веса задач), ΔSR {100 * r['d_sr']:+5.1f} п.п., +{r['gain']}"
              f"/−{r['loss']}; {pt}")
    print("  ОСНОВНЫЕ СРАВНЕНИЯ (Мак-Немар, Holm на 3; 95% бутстрэп ΔSR):")
    for c in res["primary"]:
        print(f"    {c['a']} − {c['b']}: ΔSR {100 * c['d_sr_equal_weight']:+.1f}"
              f" п.п. [{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}],"
              f" +{c['gain']}/−{c['loss']}, p {c['mcnemar_p']:.4f}, Holm "
              f"{c['holm_p']:.4f}")
    print("  вторичные:")
    for k, c in res["secondary"].items():
        print(f"    {k}: ΔSR {100 * c['d_sr_equal_weight']:+.1f} п.п. "
              f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
              f"+{c['gain']}/−{c['loss']}, p {c['mcnemar_p']:.4f}")
    for k, c in res["secondary_reached_only"].items():
        print(f"    только достигнутые, {k}: {c['states']} состояний, "
              f"+{c['gain']}/−{c['loss']}, p {c['mcnemar_p']:.4f}")
    if res.get("features"):
        print(f"  признаки: {res['features']['path']} "
              f"({res['features']['episodes']} состояний)")


def selftest():
    assert mcnemar_p(0, 0) == 1.0 and abs(mcnemar_p(5, 0) - 2 / 32) < 1e-12
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert holm([0.5, 0.01, 0.9]) == [1.0, 0.03, 1.0]
    states = [(t, s) for t in range(2) for s in range(10)]
    a = {k: k[1] < 8 for k in states}
    b = {k: k[1] < 6 for k in states}
    assert abs(equal_weight(a, states) - 0.8) < 1e-12
    c = compare(a, b, states, 2000, 1)
    assert (c["gain"], c["loss"]) == (4, 0)
    assert abs(c["d_sr_equal_weight"] - 0.2) < 1e-12
    assert c["ci95"][0] <= 0.2 <= c["ci95"][1]
    # равные веса задач против объединённого среднего
    st2 = [(0, s) for s in range(10)] + [(1, s) for s in range(5)]
    a2 = {k: k[0] == 0 for k in st2}
    assert abs(equal_weight(a2, st2) - 0.5) < 1e-12
    # бутстрэп парный и внутри задач: одинаковые векторы — интервал [0, 0]
    assert bootstrap_diff(a, a, states, 500, 1) == [0.0, 0.0]
    # сводка
    import k15h_policy as hp
    plan = dict(analysis=dict(primary=[["p4r3p", "q0ref"],
                                       ["learned_uniform", "arm_dir_uniform"],
                                       ["learned_uniform", "static_uniform"]],
                              bootstrap_n=500, bootstrap_seed=1))
    outc = {lab: dict(b) for lab in hp.LABELS}
    outc["p4r3p"] = dict(a)
    done = {k: (-1 if k[1] else 10) for k in states}       # s=0 не дошёл
    r = summarize(outc, done, states, plan)
    assert r["unreached"] == 2 and r["rows"]["p4r3p"]["gain"] == 4
    assert r["primary"][0]["gain"] == 4 and r["primary"][1]["gain"] == 0
    assert r["primary"][0]["holm_p"] >= r["primary"][0]["mcnemar_p"]
    assert r["secondary_reached_only"]["p4r3p"]["states"] == 18
    print("самопроверка k15h_analyze пройдена: Мак-Немар, Holm, равные веса "
          "задач, парный бутстрэп внутри задач, недостигнутые импульсы")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-15h: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", default="reports/k15h/plan.json")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default="reports/k15h/analysis.json")
    ap.add_argument("--features", default="reports/k15h/features.npz")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    res = run(a.plan, a.run_dir, a.out, a.features)
    report(res)
    with open(a.out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print(f"  сводка: {a.out}")
    return 0 if res["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
