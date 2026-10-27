#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-26: анализ parents_attn24 против flat_attn24 по сидам 0 и 1.

Только проверенные блоки (новые — check_run обоих планов K-26 сида,
источники — заново проверенные планы с закреплёнными SHA; закреплённые
источники двух планов одного сида обязаны совпадать). Состояние —
(набор/задача, init_state_id), env seed 101; бутстрэп N = 10000, seed из
плана. По каждому сиду S:
  ОСНОВНОЕ — parents_attn24_sS − flat_attn24_sS: ΔSR (равные веса задач),
  выигрыши/потери, точный двусторонний Мак-Немар, 95% бутстрэп внутри
  задач; Holm по двум основным p;
  обязательные вторичные — parents_attn24 − parents24_sS и flat_attn24 −
  flat24_sS; взаимодействие (new_p − new_f) − (old_p − old_f) — описательно;
  инженерные — каждая новая модель против fullbar и fullbar_cached
  (консервативная граница не-хужести, маржа 3 п.п.); MR, cond и задача 8 —
  описательно.
Сводка двух сидов (всегда оба, без отбора): SR, среднее и ско каждой
модели; средняя парная разность; совместный парный бутстрэп (одни и те же
индексы состояний внутри задачи для всех политик и обоих сидов; 2×150 не
объединяются в 300). Решение — по заранее заданному правилу (раздел 15.2
плана): положительный разведочный сигнал, если новый parents лучше нового
flat в обоих сидах, средний выигрыш ≥ 2 п.п. и нет среднего ухудшения к
старому parents; иначе эффект дерева не установлен.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k23_analyze as a23  # noqa: E402

OUT = "reports/k26/analysis.json"
SIGNAL_PP = 0.02
LABS = dict(new_p="parents_attn24_s{}", new_f="flat_attn24_s{}",
            old_p="parents24_s{}", old_f="flat24_s{}",
            mr="parents_mr24_s{}", cond="parents_cond24_s{}")


def load_seed(seed):
    import k23_protocol as p23
    import k26_protocol as pc
    res, outc, refs_pin = {}, {}, None
    plans = {}
    for mode in pc.MODES:
        path = pc.plan_path(seed, mode)
        if not os.path.exists(path):
            res[mode] = dict(plan=path, verified=0, planned=30, gaps=[],
                             complete=False)
            continue
        plan = json.load(open(path, encoding="utf-8"))
        prob = pc.check_plan_inputs(plan)
        if prob:
            raise SystemExit(f"{path}: " + "; ".join(prob[:3]))
        ok, gaps = pc.check_run(plan, path, pc.run_dir(path))
        want = len(plan["cells"]) * len(plan["labels"])
        res[mode] = dict(plan=path, verified=len(ok), planned=want,
                         gaps=gaps, complete=len(ok) == want and not gaps)
        plans[mode] = plan
        outc.update(a23.outcomes(ok))
        pins = {k: (v["sha12"], sorted(v["blocks"]))
                for k, v in plan["references"].items()}
        if refs_pin is None:
            refs_pin = pins
        elif pins != refs_pin:
            raise SystemExit(f"seed {seed}: источники планов пары различаются")
    if not all(r["complete"] for r in res.values()):
        return plans, res, None
    refs = p23.verify_refs(plans["parents_attn24"])
    return plans, res, {**a23.outcomes(refs), **outc}


def lab(key, seed):
    return LABS[key].format(seed)


def interaction(outc, seed, states):
    """(new_p − new_f) − (old_p − old_f), равные веса задач."""
    import k15h_analyze as ha
    return (ha.equal_weight(outc[lab("new_p", seed)], states)
            - ha.equal_weight(outc[lab("new_f", seed)], states)
            - ha.equal_weight(outc[lab("old_p", seed)], states)
            + ha.equal_weight(outc[lab("old_f", seed)], states))


def per_seed(seed, plan, outc):
    import k15h_analyze as ha
    keys = ("new_p", "new_f", "old_p", "old_f", "mr", "cond")
    labs = [lab(k, seed) for k in keys] + ["fullbar", "fullbar_cached"]
    states = a23.same_states(outc, labs, 150)
    boot = plan["bootstrap"]
    np_, nf = lab("new_p", seed), lab("new_f", seed)
    return dict(
        seed=seed, states=len(states),
        sr={k: ha.equal_weight(outc[k], states) for k in labs},
        per_task={k: a23.per_task(outc[k], states) for k in labs},
        primary=a23.cmp(outc, np_, nf, states, boot),
        new_p_vs_old_p=a23.cmp(outc, np_, lab("old_p", seed), states, boot),
        new_f_vs_old_f=a23.cmp(outc, nf, lab("old_f", seed), states, boot),
        interaction=interaction(outc, seed, states),
        new_p_vs_fullbar=a23.cmp(outc, np_, "fullbar", states, boot, True),
        new_p_vs_cached=a23.cmp(outc, np_, "fullbar_cached", states, boot,
                                True),
        new_f_vs_fullbar=a23.cmp(outc, nf, "fullbar", states, boot, True),
        new_f_vs_cached=a23.cmp(outc, nf, "fullbar_cached", states, boot,
                                True),
        new_p_vs_mr=a23.cmp(outc, np_, lab("mr", seed), states, boot),
        new_p_vs_cond=a23.cmp(outc, np_, lab("cond", seed), states, boot),
        task8={k: a23.per_task(outc[k], states).get("10/8") for k in labs}), \
        states


def joint_boot(outs, terms, states, n, seed):
    """Средняя по сидам величина Σ c·SR(метка) (равные веса задач); одни и
    те же индексы состояний внутри задачи для всех меток и сидов.
    terms: [(коэффициент, ключ LABS)]."""
    rng = np.random.default_rng(seed)
    by = {}
    for k in states:
        by.setdefault(k[0], []).append(k)
    tasks = sorted(by)
    seeds = sorted(outs)
    D = {s: {t: sum(c * np.asarray([float(outs[s][lab(key, s)][k])
                                    for k in by[t]]) for c, key in terms)
             for t in tasks} for s in seeds}
    est = float(np.mean([np.mean([D[s][t].mean() for t in tasks])
                         for s in seeds]))
    bs = np.empty(n)
    for i in range(n):
        idx = {t: rng.integers(0, len(by[t]), len(by[t])) for t in tasks}
        bs[i] = np.mean([np.mean([D[s][t][idx[t]].mean() for t in tasks])
                         for s in seeds])
    return est, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


TERMS = dict(
    primary=[(1, "new_p"), (-1, "new_f")],
    new_p_vs_old_p=[(1, "new_p"), (-1, "old_p")],
    new_f_vs_old_f=[(1, "new_f"), (-1, "old_f")],
    interaction=[(1, "new_p"), (-1, "new_f"), (-1, "old_p"), (1, "old_f")])


def decision(summ):
    d = summ["primary"]["per_seed"]
    m = summ["primary"]["mean"]
    mp = summ["new_p_vs_old_p"]["mean"]
    mf = summ["new_f_vs_old_f"]["mean"]
    signal = all(v > 0 for v in d.values()) and m >= SIGNAL_PP and mp >= 0
    if signal:
        verdict = ("положительный разведочный сигнал: новый parents лучше "
                   "нового flat в обоих сидах, средний выигрыш ≥ 2 п.п., "
                   "без среднего ухудшения к старому parents (не "
                   "доказательство превосходства; следующий шаг — "
                   "отдельный план: абляция epsilon = 1 и независимая "
                   "проверка)")
    else:
        why = []
        if not all(v > 0 for v in d.values()):
            why.append("знак parents − flat не положителен в обоих сидах")
        if m < SIGNAL_PP:
            why.append(f"средний выигрыш {100 * m:+.1f} п.п. < 2")
        if mp < 0:
            why.append(f"новый parents в среднем ниже старого "
                       f"({100 * mp:+.1f} п.п.)")
        verdict = "эффект дерева не установлен: " + "; ".join(why)
    note = None
    if mp > 0 and mf > 0 and not signal:
        note = ("обе новые модели в среднем выше старых — это поддержка "
                "визуального подключения, а не иерархии")
    return dict(signal=signal, verdict=verdict, note=note)


def summary(per, outs, states, boot):
    import k15h_analyze as ha
    seeds = sorted(per)
    rows = {}
    for key in ("new_p", "new_f", "old_p", "old_f", "mr", "cond"):
        v = [per[s]["sr"][lab(key, s)] for s in seeds]
        rows[LABS[key].format("S")] = dict(
            per_seed=dict(zip(seeds, v)), mean=float(np.mean(v)),
            sd=float(np.std(v, ddof=1)) if len(v) > 1 else None)
    out = dict(seeds=seeds, rows=rows, holm_primary=dict(zip(
        seeds, ha.holm([per[s]["primary"]["mcnemar_p"] for s in seeds]))))
    for key, terms in TERMS.items():
        d = {s: (per[s][key]["d_sr_equal_weight"] if key != "interaction"
                 else per[s]["interaction"]) for s in seeds}
        est, ci = joint_boot(outs, terms, states, boot["n"], boot["seed"])
        out[key] = dict(per_seed=d, mean=float(np.mean(list(d.values()))),
                        joint_paired_bootstrap=dict(d=est, ci95=ci))
    out["decision"] = decision(out)
    out["note"] = ("разработочный банк 30–44 (разведочно); интервалы условны "
                   "на двух обученных реализациях; Holm внутри K-26 не "
                   "делает результат независимым подтверждением")
    return out


def report(res, per, summ):
    line = a23.line
    for s, r in res.items():
        for mode, x in r.items():
            print(f"  seed {s} {mode}: {x['plan']}: проверено "
                  f"{x['verified']}/{x['planned']}"
                  + ("" if x["complete"] else " — НЕПОЛНЫЙ"))
    for s, o in sorted(per.items()):
        print(f"\n  SEED {s} (150 состояний 30–44): " + ", ".join(
            f"{k} {100 * v:.1f}%" for k, v in o["sr"].items()))
        print(f"    ОСНОВНОЕ parents_attn24 − flat_attn24: "
              f"{line(o['primary'])}" + (f"; Holm {summ['holm_primary'][s]:.4f}"
                                         if summ else ""))
        print(f"    parents_attn24 − parents24: {line(o['new_p_vs_old_p'])}")
        print(f"    flat_attn24 − flat24: {line(o['new_f_vs_old_f'])}")
        print(f"    взаимодействие (описательно): "
              f"{100 * o['interaction']:+.1f} п.п.")
        for k, nm in (("new_p_vs_fullbar", "parents_attn24 − fullbar"),
                      ("new_p_vs_cached", "parents_attn24 − fullbar_cached"),
                      ("new_f_vs_fullbar", "flat_attn24 − fullbar"),
                      ("new_f_vs_cached", "flat_attn24 − fullbar_cached")):
            print(f"    {nm}: {line(o[k])}")
        print(f"    parents_attn24 − MR (описательно): {line(o['new_p_vs_mr'])}")
        print(f"    parents_attn24 − cond (описательно): "
              f"{line(o['new_p_vs_cond'])}")
        print("    задача 8 (описательно): " + ", ".join(
            f"{k} {v[0]}/{v[1]}" for k, v in o["task8"].items()))
    if summ:
        print("\n  СВОДКА ДВУХ СИДОВ (без отбора):")
        for a, r in summ["rows"].items():
            print(f"    {a:18} " + " ".join(f"s{s} {100 * v:.1f}%" for s, v in
                                           r["per_seed"].items())
                  + f" | среднее {100 * r['mean']:.1f}%, ско "
                  + (f"{100 * r['sd']:.1f}" if r["sd"] is not None else "—"))
        for key, nm in (("primary", "ОСНОВНОЕ parents_attn − flat_attn"),
                        ("new_p_vs_old_p", "parents_attn − parents24"),
                        ("new_f_vs_old_f", "flat_attn − flat24"),
                        ("interaction", "взаимодействие")):
            c = summ[key]
            jb = c["joint_paired_bootstrap"]
            print(f"    {nm}: " + ", ".join(
                f"s{s} {100 * v:+.1f}" for s, v in c["per_seed"].items())
                  + f"; среднее {100 * c['mean']:+.1f} п.п.; совместный "
                  f"бутстрэп [{100 * jb['ci95'][0]:+.1f}; "
                  f"{100 * jb['ci95'][1]:+.1f}]")
        dcs = summ["decision"]
        print(f"\n  РЕШЕНИЕ (правило 15.2): {dcs['verdict']}")
        if dcs["note"]:
            print(f"    {dcs['note']}")


def selftest():
    rng = np.random.default_rng(0)
    states = [(f"10/{t}", i) for t in range(10) for i in range(30, 45)]
    base = {k: bool(rng.random() < .85) for k in states}
    fails = [k for k in states if not base[k]]

    def make(gain_p):
        outs, per = {}, {}
        for s in (0, 1):
            o = {LABS[k].format(s): dict(base) for k in LABS}
            o["fullbar"], o["fullbar_cached"] = dict(base), dict(base)
            for k in fails[:gain_p[s]]:
                o[f"parents_attn24_s{s}"][k] = True
            outs[s] = o
            per[s], st = per_seed(s, dict(bootstrap=dict(n=300, seed=1)), o)
        return summary(per, outs, st, dict(n=300, seed=1))
    sm = make((5, 4))
    assert sm["primary"]["per_seed"][0] > 0 and sm["decision"]["signal"]
    jb = sm["primary"]["joint_paired_bootstrap"]
    assert abs(jb["d"] - sm["primary"]["mean"]) < 1e-12
    assert abs(sm["interaction"]["mean"] - sm["primary"]["mean"]) < 1e-12
    assert len(sm["holm_primary"]) == 2
    sm2 = make((5, 0))
    assert not sm2["decision"]["signal"] and "обоих" in \
        sm2["decision"]["verdict"]
    sm3 = make((1, 1))
    assert not sm3["decision"]["signal"] and "< 2" in sm3["decision"]["verdict"]
    print("самопроверка k26_analyze пройдена: основное parents_attn − "
          "flat_attn, Holm на 2, совместный бутстрэп = средняя Δ, "
          "взаимодействие, правило решения (сигнал / знак / порог 2 п.п.)")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-26: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    res, per, outs, st, boot = {}, {}, {}, None, None
    try:
        for s in a.seeds:
            plans, r, outc = load_seed(s)
            res[s] = r
            if outc is None:
                continue
            plan = plans["parents_attn24"]
            per[s], st = per_seed(s, plan, outc)
            outs[s] = outc
            boot = plan["bootstrap"]
        summ = summary(per, outs, st, boot) if len(per) == len(a.seeds) \
            and len(per) > 1 else None
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    report(res, per, summ)
    os.makedirs("reports/k26", exist_ok=True)
    out = OUT if summ else \
        f"reports/k26/analysis_s{'_'.join(map(str, a.seeds))}.json"
    json.dump(dict(plans=res, per_seed=per, summary=summ), open(
        out, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
        default=str)
    print(f"  сводка: {out}")
    complete = all(x["complete"] for r in res.values() for x in r.values())
    return 0 if complete else 3


if __name__ == "__main__":
    sys.exit(main())
