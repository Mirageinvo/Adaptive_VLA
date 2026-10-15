#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16s: контроль направления записанного вмешательства против learned.

Только проверенные блоки stored_rot_s1..s3 (stored_replay — техническая
проверка, в сравнение не входит); неполный набор — код 3. Статистика —
как у K-16r (k16r_analyze.summarize): каждый сид против learned и q0;
средний успех контроля по состоянию; learned − контроль по 50 состояниям.
Похожий результат значит «преимущество обученного направления не
обнаружено» для этого вмешательства на этих состояниях.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ROT = ("stored_rot_s1", "stored_rot_s2", "stored_rot_s3")


def run(plan_path, run_dir):
    import k16_protocol as p16
    import k16r_analyze as ra
    import k16s_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan)
    if prob:
        raise SystemExit("входы плана K-16s не совпадают: " + "; ".join(
            prob[:4]))
    pairs = [(lab, t) for lab in ROT for t in plan["tasks"]]
    ok, gaps = pc.check_run(plan, plan_path, run_dir, pairs)
    res = dict(plan=plan_path, run_dir=run_dir, verified=len(ok),
               planned=len(pairs), gaps=gaps,
               complete=len(ok) == len(pairs) and not gaps)
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
            for lab in ROT}
    learned = outcomes([(t, plan["refs"][f"{pc.BASE}_t{t}"])
                        for t in plan["tasks"]])
    q0 = outcomes([(t, plan["refs"][f"q0_t{t}"]) for t in plan["tasks"]])
    res.update(ra.summarize(ctrl, learned, q0, sorted(q0)))
    return res


def main():
    ap = argparse.ArgumentParser(description="K-16s: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", default="reports/k16s/plan.json")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default="reports/k16s/analysis.json")
    a = ap.parse_args()
    if a.selftest:
        import k16r_analyze as ra
        ra.selftest()
        print("самопроверка k16s_analyze пройдена (статистика K-16r)")
        return 0
    import k16r_analyze as ra
    try:
        res = run(a.plan, a.run_dir)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    ra.report(res)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    print(f"  сводка: {a.out}")
    return 0 if res["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
