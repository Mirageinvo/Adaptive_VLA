#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, этап 3: парное сравнение двух оценок на закреплённом банке
(по умолчанию flat24 seed 0, n_exec 4 против 8).

Задачи берутся только с действительной отметкой завершения
(rt2_eval.task_done_problems, n = числу случаев банка); набор (задача, seed)
каждой оценки обязан В ТОЧНОСТИ совпасть с банком — иначе отказ. Вывод: SR с
равными весами задач, по задачам, выигрыши/потери, точный Мак-Немар,
бутстрэп по парам внутри задач (rt2_compare.compare).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def bank_keys(bank):
    return {(t, int(c["seed"])) for t, v in bank["tasks"].items()
            for c in v["cases"]}


def main():
    import rt2_compare as rcmp
    ap = argparse.ArgumentParser(description="RoboTwin: n_exec по банку")
    ap.add_argument("--bank", default="reports/rt3/bank.json")
    ap.add_argument("--a", default="reports/rt3/eval/nexec4_s0_flat24")
    ap.add_argument("--b", default="reports/rt3/eval/nexec8_s0_flat24")
    ap.add_argument("--out", default="reports/rt3/compare_nexec4_nexec8.json")
    a = ap.parse_args()
    bank = json.load(open(a.bank, encoding="utf-8"))
    keys = bank_keys(bank)
    tasks = sorted(bank["tasks"])
    n = len(next(iter(bank["tasks"].values()))["cases"])
    A, pa = rcmp.load(a.a, tasks, n)
    B, pb = rcmp.load(a.b, tasks, n)
    if pa or pb:
        print("  НЕПОЛНО/ОТКАЗ: " + "; ".join((pa + pb)[:6]))
        return 3
    for nm, X in ((a.a, A), (a.b, B)):
        if set(X) != keys:
            print(f"  ОТКАЗ: {nm}: набор (задача, seed) ≠ банку "
                  f"({sorted(set(X) ^ keys)[:4]})")
            return 3
    r = rcmp.compare(A, B)
    print(f"  {os.path.basename(a.a)} − {os.path.basename(a.b)} ({r['n']} "
          f"пар): SR {100 * r['sr_a']:.1f}% / {100 * r['sr_b']:.1f}%, Δ "
          f"{100 * r['d']:+.1f} п.п. [{100 * r['ci95'][0]:+.1f}; "
          f"{100 * r['ci95'][1]:+.1f}], +{r['gain']}/−{r['loss']}, "
          f"Мак-Немар p {r['mcnemar_p']:.4f}")
    for t, v in r["per_task"].items():
        print(f"    {t:20} {v[0]}/{v[2]}  против  {v[1]}/{v[2]}")
    json.dump(dict(r, a=a.a, b=a.b, bank=a.bank,
                   bank_fingerprint=bank["fingerprint"],
                   note="один тест режима исполнения на закреплённом банке; "
                        "разведочно"),
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(f"  {a.out}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        b = dict(tasks=dict(x=dict(cases=[dict(seed=1), dict(seed=3)])))
        assert bank_keys(b) == {("x", 1), ("x", 3)}
        print("самопроверка rt3_compare пройдена")
        sys.exit(0)
    sys.exit(main())
