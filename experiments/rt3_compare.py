#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, этап 3: парное сравнение двух оценок на закреплённом банке
(по умолчанию flat24 seed 0, n_exec 4 против 8).

Задачи берутся только с действительными отметками RT2 и rt3 (rt3_check:
привязка к текущему банку, подтверждённые случаи, инструкции, манифест,
SHA журналов), n = числу случаев банка; набор (задача, seed) каждой оценки
обязан В ТОЧНОСТИ совпасть с банком — иначе отказ. Модели двух оценок
обязаны совпасть по всем ключам манифеста (чекпойнт, веса, prep_id,
policy, фабрика, seed, версия); единственное допустимое различие — n_exec,
и он обязан быть ровно --na и --nb. Вывод: SR с равными весами задач, по
задачам, выигрыши/потери, точный Мак-Немар, бутстрэп по парам внутри задач
(rt2_compare.compare).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def identity_problems(ea, eb, na, nb):
    """Манифесты двух оценок: всё, кроме n_exec, совпадает; n_exec = na/nb."""
    import rt2_policy as rp
    if ea is None or eb is None:
        return ["нет манифеста модели в одной из оценок"]
    p = [f"модели различаются по {k}" for k in rp.MODEL_KEYS
         if ea.get(k) != eb.get(k) or ea.get(k) is None]
    extra = sorted((set(ea) | set(eb)) - set(rp.MODEL_KEYS) - {"n_exec"})
    p += [f"лишний ключ манифеста {k} различается" for k in extra
          if ea.get(k) != eb.get(k)]
    if ea.get("n_exec") != na or eb.get("n_exec") != nb:
        p.append(f"n_exec {ea.get('n_exec')}/{eb.get('n_exec')}, ожидалось "
                 f"{na}/{nb}")
    return p


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
    ap.add_argument("--na", type=int, default=4)
    ap.add_argument("--nb", type=int, default=8)
    a = ap.parse_args()
    import rt3_check as rk
    bank = rk.load_bank(a.bank)
    keys = bank_keys(bank)
    A, pa = rk.load_outcomes(a.a, bank)
    B, pb = rk.load_outcomes(a.b, bank)
    if pa or pb:
        print("  НЕПОЛНО/ОТКАЗ: " + "; ".join((pa + pb)[:6]))
        return 3
    ea, eb = rk.expected_of(a.a), rk.expected_of(a.b)
    pi = identity_problems(ea, eb, a.na, a.nb)
    if pi:
        print("  ОТКАЗ: " + "; ".join(pi))
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
                   model=ea, n_exec=dict(a=a.na, b=a.nb),
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
        import rt2_policy as rp
        m = {k: f"v_{k}" for k in rp.MODEL_KEYS}
        assert identity_problems(dict(m, n_exec=4), dict(m, n_exec=8),
                                 4, 8) == []
        for k in rp.MODEL_KEYS:
            assert identity_problems(dict(m, n_exec=4),
                                     dict(m, **{k: "other"}, n_exec=8), 4, 8)
        assert identity_problems(dict(m, n_exec=8), dict(m, n_exec=8), 4, 8)
        assert identity_problems(dict(m, n_exec=8), dict(m, n_exec=4), 4, 8)
        assert identity_problems(None, dict(m, n_exec=8), 4, 8)
        print("самопроверка rt3_compare пройдена: ключи банка; модели — все "
              "ключи манифеста равны, различие только n_exec = 4/8")
        sys.exit(0)
    sys.exit(main())
