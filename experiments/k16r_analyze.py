#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16r: контроль направления против learned (goal_band_hist_event K-16).

Только проверенные блоки; неполный набор — код 3. 50 уникальных состояний:
три сида контроля — не 150 независимых примеров.
  * каждый сид: SR, против learned и против q0 (Мак-Немар, бутстрэп);
  * средний успех контроля по состоянию (среднее трёх сидов): парная
    разность learned − контроль по состояниям, среднее с равными весами
    задач и 95% бутстрэп внутри задач;
  * описательно — задача 8 и остальные.
Похожий результат значит «преимущество обученного направления не
обнаружено»; амплитуда, временная форма и момент у контроля — от learned.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BOOT_N, BOOT_SEED = 10000, 20261015


def boot_mean_diff(diff, states, n=BOOT_N, seed=BOOT_SEED):
    """diff {(t, s): число}; среднее с равными весами задач и 95% ДИ
    (пересэмплирование состояний внутри задач)."""
    rng = np.random.default_rng(seed)
    by = {}
    for k in states:
        by.setdefault(k[0], []).append(float(diff[k]))
    arrs = [np.asarray(by[t]) for t in sorted(by)]
    est = float(np.mean([x.mean() for x in arrs]))
    bs = np.zeros(n)
    for x in arrs:
        bs += x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    bs /= len(arrs)
    return est, [float(np.percentile(bs, 2.5)), float(np.percentile(bs,
                                                                     97.5))]


def summarize(ctrl, learned, q0, states):
    import k15h_analyze as ha
    res = dict(states=len(states), seeds={})
    for lab, o in ctrl.items():
        res["seeds"][lab] = dict(
            sr_equal_weight=ha.equal_weight(o, states),
            vs_learned=ha.compare(o, learned, states, BOOT_N, BOOT_SEED),
            vs_q0=ha.compare(o, q0, states, BOOT_N, BOOT_SEED))
    mean_c = {k: float(np.mean([ctrl[l][k] for l in ctrl])) for k in states}
    diff = {k: float(learned[k]) - mean_c[k] for k in states}
    est, ci = boot_mean_diff(diff, states)
    res["learned_minus_mean_control"] = dict(d_sr=est, ci95=ci)
    res["sr"] = dict(learned=ha.equal_weight(learned, states),
                     q0=ha.equal_weight(q0, states),
                     mean_control=float(np.mean([np.mean(
                         [mean_c[k] for k in states if k[0] == t])
                         for t in sorted({k[0] for k in states})])))
    t8 = [k for k in states if k[0] == 8]
    rest = [k for k in states if k[0] != 8]
    res["descriptive_task8"] = {
        part: dict(learned=sum(learned[k] for k in ks),
                   q0=sum(q0[k] for k in ks),
                   control_mean=float(sum(mean_c[k] for k in ks)),
                   n=len(ks)) for part, ks in (("task8", t8),
                                                ("others", rest))}
    return res


def run(plan_path, run_dir):
    import k16_protocol as p16
    import k16r_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan)
    if prob:
        raise SystemExit("входы плана K-16r не совпадают: " + "; ".join(
            prob[:4]))
    ok, gaps = pc.check_run(plan, plan_path, run_dir)
    want = len(plan["labels"]) * len(plan["tasks"])
    res = dict(plan=plan_path, run_dir=run_dir, verified=len(ok),
               planned=want, gaps=gaps, complete=len(ok) == want and
               not gaps)
    if not res["complete"]:
        return res

    def outcomes(paths):
        o = {}
        for t, pth in paths:
            d, _z = p16.load(pth)
            for e in d["episodes"]:
                o[(int(t), int(e["init_state_id"]))] = bool(e["success"])
        return o
    ctrl = {lab: outcomes([(t, ok[(lab, t)]) for t in plan["tasks"]])
            for lab in plan["labels"]}
    learned = outcomes([(t, plan["refs"][f"{pc.BASE}_t{t}"])
                        for t in plan["tasks"]])
    q0 = outcomes([(t, plan["refs"][f"q0_t{t}"]) for t in plan["tasks"]])
    states = sorted(q0)
    res.update(summarize(ctrl, learned, q0, states))
    return res


def report(res):
    print(f"  проверенных блоков {res['verified']}/{res['planned']}; "
          f"непригодных или отсутствующих {len(res['gaps'])}")
    for g in [g for g in res["gaps"] if g["why"] != "не готов"][:10]:
        print(f"    ВНИМАНИЕ {g['label']} t{g['task']}: {g['why']}")
    if not res["complete"]:
        print("  НАБОР НЕПОЛНЫЙ: вывода нет")
        return
    s = res["sr"]
    print(f"  SR: learned {100 * s['learned']:.1f}%, q0 {100 * s['q0']:.1f}%,"
          f" средний контроль {100 * s['mean_control']:.1f}%")
    for lab, r in res["seeds"].items():
        vl, vq = r["vs_learned"], r["vs_q0"]
        print(f"    {lab}: SR {100 * r['sr_equal_weight']:.1f}%; против "
              f"learned {100 * vl['d_sr_equal_weight']:+.1f} п.п. "
              f"[{100 * vl['ci95'][0]:+.1f}; {100 * vl['ci95'][1]:+.1f}], "
              f"+{vl['gain']}/−{vl['loss']}, p {vl['mcnemar_p']:.3f}; против "
              f"q0 +{vq['gain']}/−{vq['loss']}, p {vq['mcnemar_p']:.3f}")
    m = res["learned_minus_mean_control"]
    print(f"  learned − средний контроль (по состояниям, равные веса задач): "
          f"{100 * m['d_sr']:+.1f} п.п. [{100 * m['ci95'][0]:+.1f}; "
          f"{100 * m['ci95'][1]:+.1f}]")
    for part, v in res["descriptive_task8"].items():
        print(f"    {part}: learned {v['learned']}/{v['n']}, q0 {v['q0']}/"
              f"{v['n']}, средний контроль {v['control_mean']:.2f}/{v['n']}")


def selftest():
    states = [(t, s) for t in range(10) for s in range(25, 30)]
    rng = np.random.default_rng(0)
    q0 = {k: bool(rng.random() < 0.8) for k in states}
    learned = dict(q0)
    for k in [k for k in states if not q0[k]][:4]:
        learned[k] = True
    ctrl = {f"c{i}": dict(q0) for i in range(3)}
    r = summarize(ctrl, learned, q0, states)
    assert r["seeds"]["c0"]["vs_learned"]["loss"] == 4
    assert r["learned_minus_mean_control"]["d_sr"] > 0
    lo, hi = r["learned_minus_mean_control"]["ci95"]
    assert lo <= r["learned_minus_mean_control"]["d_sr"] <= hi
    same = summarize({"c": dict(learned)}, learned, q0, states)
    assert same["learned_minus_mean_control"]["d_sr"] == 0.0
    print("самопроверка k16r_analyze пройдена: сиды против learned и q0, "
          "средний контроль по состоянию, бутстрэп внутри задач")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-16r: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", default="reports/k16r/plan.json")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default="reports/k16r/analysis.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        res = run(a.plan, a.run_dir)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    report(res)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    print(f"  сводка: {a.out}")
    return 0 if res["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
