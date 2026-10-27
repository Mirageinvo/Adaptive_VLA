#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-26: анализ parents_attn24 и flat_attn24 по сидам 0 и 1.

Только проверенные блоки (новые — check_run обоих планов K-26 сида,
обязательные источники — заново проверенные планы с закреплёнными SHA;
обязательные источники двух планов сида обязаны совпадать). Состояние —
(набор/задача, init_state_id), env seed 101; бутстрэп N = 10000, seed из
плана. Все запланированные сравнения по каждому сиду S:
  механизм — parents_attn24_sS − flat_attn24_sS: ΔSR (равные веса задач),
  выигрыши/потери, точный двусторонний Мак-Немар, 95% бутстрэп внутри
  задач; Holm по двум p этого сравнения;
  parents_attn24 − parents24_sS и flat_attn24 − flat24_sS; взаимодействие
  (new_p − new_f) − (old_p − old_f) — описательно;
  каждая новая модель против fullbar и fullbar_cached (консервативная
  граница не-хужести, маржа 3 п.п.); MR и задача 8 — описательно; cond
  (K-25) — необязательно: при отсутствии полного проверенного набора
  «сравнение не выполнено» с причиной.
Сводка двух сидов (всегда оба): SR, среднее и ско; средние парные разности
и совместный парный бутстрэп (одни и те же индексы состояний внутри задачи
для всех политик и обоих сидов; 2×150 не объединяются в 300).
Выводы (раздел 15.2 с поправкой от 27.10) — ДВА независимых, без общего
сигнала и без порога 2 п.п.:
  project_result — SR дерева к исходному BAR (точечно и статистически:
  не-хужесть / превосходство — раздельно), задержка из сопоставимого замера
  K-26 (если выполнен), изменение к прежнему parents24;
  hierarchy_specific_evidence — дерево против нового flat (знак по сидам,
  Holm, совместный интервал, взаимодействие).
Результат flat_attn24 не является условием остановки дерева; SR и p-value
не влияют на код завершения (3 — только неполный/непроверенный набор).
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
LATENCY = "reports/k26/latency.json"
ALPHA = 0.05
LABS = dict(new_p="parents_attn24_s{}", new_f="flat_attn24_s{}",
            old_p="parents24_s{}", old_f="flat24_s{}",
            mr="parents_mr24_s{}", cond="parents_cond24_s{}")
REQUIRED = ("new_p", "new_f", "old_p", "old_f", "mr")


def lab(key, seed):
    return LABS[key].format(seed)


def load_seed(seed):
    import k23_protocol as p23
    import k26_protocol as pc
    res, outc, refs_pin, plans = {}, {}, None, {}
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
            raise SystemExit(f"seed {seed}: обязательные источники планов "
                             f"пары различаются")
    if not all(r["complete"] for r in res.values()):
        return plans, res, None, {}
    refs = p23.verify_refs(plans["parents_attn24"])
    outc = {**a23.outcomes(refs), **outc}
    # необязательный cond: только если оба плана закрепили один и тот же
    # набор и он заново прошёл проверку
    opt = {}
    o_p = plans["parents_attn24"].get("optional_references") or {}
    o_f = plans["flat_attn24"].get("optional_references") or {}
    for name in sorted(set(o_p) | set(o_f)):
        a_, b_ = o_p.get(name) or {}, o_f.get(name) or {}
        if a_ != b_:
            opt[name] = dict(status="не выполнено", why="необязательные "
                             "источники планов пары различаются")
            continue
        blocks, why = pc.verify_optional(dict(optional_references={
            name: a_}))[name]
        if blocks is None:
            opt[name] = dict(status="не выполнено", why=why)
            continue
        opt[name] = dict(status="выполнено")
        outc.update(a23.outcomes(blocks))
    return plans, res, outc, opt


def interaction(outc, seed, states):
    import k15h_analyze as ha
    return (ha.equal_weight(outc[lab("new_p", seed)], states)
            - ha.equal_weight(outc[lab("new_f", seed)], states)
            - ha.equal_weight(outc[lab("old_p", seed)], states)
            + ha.equal_weight(outc[lab("old_f", seed)], states))


def per_seed(seed, plan, outc, opt=None):
    import k15h_analyze as ha
    labs = [lab(k, seed) for k in REQUIRED] + ["fullbar", "fullbar_cached"]
    states = a23.same_states(outc, labs, 150)
    boot = plan["bootstrap"]
    np_, nf = lab("new_p", seed), lab("new_f", seed)
    o = dict(
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
        task8={k: a23.per_task(outc[k], states).get("10/8") for k in labs})
    cl = lab("cond", seed)
    st = (opt or {}).get(f"k25_s{seed}") or dict(
        status="не выполнено", why="необязательный набор не закреплён")
    if st["status"] == "выполнено" and cl in outc and \
            sorted(outc[cl]) == states:
        o["sr"][cl] = ha.equal_weight(outc[cl], states)
        o["new_p_vs_cond"] = dict(a23.cmp(outc, np_, cl, states, boot),
                                  status="выполнено")
    else:
        why = st.get("why") if st["status"] != "выполнено" else \
            "состояния cond ≠ состояниям K-26"
        o["new_p_vs_cond"] = dict(status="не выполнено", why=why)
    return o, states


def joint_boot(outs, terms, states, n, seed):
    """Средняя по сидам Σ c·SR(метка); одни и те же индексы состояний внутри
    задачи для всех меток и сидов. terms: [(коэффициент, ключ LABS или
    готовая метка)]."""
    rng = np.random.default_rng(seed)
    by = {}
    for k in states:
        by.setdefault(k[0], []).append(k)
    tasks = sorted(by)
    seeds = sorted(outs)

    def L(key, s):
        return lab(key, s) if key in LABS else key
    D = {s: {t: sum(c * np.asarray([float(outs[s][L(key, s)][k])
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
    interaction=[(1, "new_p"), (-1, "new_f"), (-1, "old_p"), (1, "old_f")],
    new_p_vs_fullbar=[(1, "new_p"), (-1, "fullbar")],
    new_p_vs_cached=[(1, "new_p"), (-1, "fullbar_cached")])


def read_latency(path=LATENCY):
    """Задержка из сопоставимого замера K-26 (если выполнен)."""
    if not os.path.exists(path):
        return dict(status="не выполнено", why=f"нет {path}")
    r = json.load(open(path, encoding="utf-8"))
    out = dict(status="выполнено", device=r.get("device"), batches={})
    for bk in [k for k in r if k.startswith("batch") and k != "batches"]:
        b = r[bk]
        if not isinstance(b, dict) or "parents_attn24" not in b:
            continue
        row = dict(parents_attn24_ms=b["parents_attn24"]["total"]["median"])
        for ref in ("fullbar", "fullbar_cached", "parents24"):
            if ref in b:
                x, y = b["parents_attn24"]["total"], b[ref]["total"]
                row[f"vs_{ref}_pct"] = dict(
                    median=100 * (x["median"] / y["median"] - 1),
                    p95=100 * (x["p95"] / y["p95"] - 1))
        out["batches"][bk] = row
    return out


def stat_flags(c):
    """Точечное и статистическое раздельно."""
    return dict(point=("выше" if c["d_sr_equal_weight"] > 0 else "равно"
                       if c["d_sr_equal_weight"] == 0 else "ниже"),
                non_inferior_3pp=c.get("non_inferior_margin_3pp"),
                superior=bool(c["mcnemar_p"] < ALPHA and c["ci95"][0] > 0))


def project_result(per, summ, latency):
    seeds = sorted(per)
    out = dict(per_seed={}, latency=latency)
    for s in seeds:
        o = per[s]
        out["per_seed"][s] = dict(
            sr_tree=o["sr"][lab("new_p", s)], sr_fullbar=o["sr"]["fullbar"],
            vs_fullbar=dict(d=o["new_p_vs_fullbar"]["d_sr_equal_weight"],
                            ci95=o["new_p_vs_fullbar"]["ci95"],
                            p=o["new_p_vs_fullbar"]["mcnemar_p"],
                            ni_lower=o["new_p_vs_fullbar"][
                                "ni_lower_conservative"],
                            **stat_flags(o["new_p_vs_fullbar"])),
            vs_cached=dict(d=o["new_p_vs_cached"]["d_sr_equal_weight"],
                           **stat_flags(o["new_p_vs_cached"])),
            vs_old_parents=dict(d=o["new_p_vs_old_p"]["d_sr_equal_weight"],
                                ci95=o["new_p_vs_old_p"]["ci95"],
                                p=o["new_p_vs_old_p"]["mcnemar_p"]))
    for k in ("new_p_vs_fullbar", "new_p_vs_cached", "new_p_vs_old_p"):
        out[k] = dict(mean=summ[k]["mean"],
                      joint_ci95=summ[k]["joint_paired_bootstrap"]["ci95"])
    ps = out["per_seed"]
    out["text"] = (
        "дерево к BAR: точечно " + ", ".join(
            f"s{s} {100 * ps[s]['vs_fullbar']['d']:+.1f} п.п. "
            f"({ps[s]['vs_fullbar']['point']}; не-хужесть "
            f"{'установлена' if ps[s]['vs_fullbar']['non_inferior_3pp'] else 'не установлена'}"
            f"; превосходство "
            f"{'установлено' if ps[s]['vs_fullbar']['superior'] else 'не установлено'})"
            for s in seeds)
        + f"; к прежнему parents24 в среднем "
          f"{100 * out['new_p_vs_old_p']['mean']:+.1f} п.п. "
          f"[{100 * out['new_p_vs_old_p']['joint_ci95'][0]:+.1f}; "
          f"{100 * out['new_p_vs_old_p']['joint_ci95'][1]:+.1f}]")
    return out


def hierarchy_evidence(summ):
    d = summ["primary"]["per_seed"]
    holm = summ["holm_primary"]
    ci = summ["primary"]["joint_paired_bootstrap"]["ci95"]
    m = summ["primary"]["mean"]
    if all(v > 0 for v in d.values()):
        sign = "точечно дерево выше нового flat в обоих сидах"
    elif all(v < 0 for v in d.values()):
        sign = "точечно новый flat выше дерева в обоих сидах"
    elif all(v == 0 for v in d.values()):
        sign = "точечно равны в обоих сидах"
    else:
        sign = "знак разности различается между сидами"
    established = all(h < ALPHA for h in holm.values()) and (
        ci[0] > 0 or ci[1] < 0) and len({np.sign(v) for v in d.values()}) \
        == 1
    if established:
        stat = ("различие статистически установлено (Holm < 0.05 в обоих "
                "сидах, совместный интервал не содержит 0) — "
                + ("в пользу дерева" if m > 0 else "в пользу нового flat"))
    else:
        stat = "статистически различие не установлено"
    if not (established and m > 0):
        what = ("дополнительное преимущество иерархии над новым flat не "
                "обнаружено")
    else:
        what = ("есть свидетельство пользы иерархии сверх визуального "
                "доступа (разведочно; наследование отдельно не выделено — "
                "нужна абляция epsilon = 1)")
    return dict(sign=sign, established=established, mean=m, joint_ci95=ci,
                holm=holm, interaction_mean=summ["interaction"]["mean"],
                text=f"{sign}; {stat}; {what}",
                note="сравнение с flat_attn24 ограничивает выводы о пользе "
                     "именно иерархии, но не является условием остановки "
                     "дерева")


def summary(per, outs, states, boot, latency=None):
    import k15h_analyze as ha
    seeds = sorted(per)
    rows = {}
    for key in ("new_p", "new_f", "old_p", "old_f", "mr"):
        v = [per[s]["sr"][lab(key, s)] for s in seeds]
        rows[LABS[key].format("S")] = dict(
            per_seed=dict(zip(seeds, v)), mean=float(np.mean(v)),
            sd=float(np.std(v, ddof=1)) if len(v) > 1 else None)
    for key in ("fullbar", "fullbar_cached"):
        v = [per[s]["sr"][key] for s in seeds]
        rows[key] = dict(per_seed=dict(zip(seeds, v)), mean=float(np.mean(v)),
                         sd=None)
    if all(per[s]["new_p_vs_cond"]["status"] == "выполнено" for s in seeds):
        v = [per[s]["sr"][lab("cond", s)] for s in seeds]
        rows[LABS["cond"].format("S")] = dict(
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
    out["project_result"] = project_result(per, out, latency or dict(
        status="не выполнено", why="замер не передан"))
    out["hierarchy_specific_evidence"] = hierarchy_evidence(out)
    out["note"] = ("разработочный банк 30–44 (разведочно); интервалы условны "
                   "на двух обученных реализациях; Holm внутри K-26 не "
                   "делает результат независимым подтверждением; "
                   "flat_attn24 — дополнительная архитектура для сравнения")
    return out


def exit_code(res):
    """Технический код: 3 только при неполном/непроверенном наборе; SR и
    p-value на код не влияют."""
    return 0 if all(x["complete"] for r in res.values()
                    for x in r.values()) else 3


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
        print(f"    механизм parents_attn24 − flat_attn24: "
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
        c = o["new_p_vs_cond"]
        print("    parents_attn24 − cond (необязательно): " + (
            line(c) if c["status"] == "выполнено" else
            f"сравнение не выполнено: {c['why']}"))
        print("    задача 8 (описательно): " + ", ".join(
            f"{k} {v[0]}/{v[1]}" for k, v in o["task8"].items()))
    if summ:
        print("\n  СВОДКА ДВУХ СИДОВ (без отбора):")
        for a, r in summ["rows"].items():
            print(f"    {a:20} " + " ".join(f"s{s} {100 * v:.1f}%" for s, v in
                                           r["per_seed"].items())
                  + f" | среднее {100 * r['mean']:.1f}%" + (
                      f", ско {100 * r['sd']:.1f}" if r["sd"] is not None
                      else ""))
        for key, nm in (("primary", "механизм parents_attn − flat_attn"),
                        ("new_p_vs_old_p", "parents_attn − parents24"),
                        ("new_f_vs_old_f", "flat_attn − flat24"),
                        ("interaction", "взаимодействие"),
                        ("new_p_vs_fullbar", "parents_attn − fullbar"),
                        ("new_p_vs_cached", "parents_attn − fullbar_cached")):
            c = summ[key]
            jb = c["joint_paired_bootstrap"]
            print(f"    {nm}: " + ", ".join(
                f"s{s} {100 * v:+.1f}" for s, v in c["per_seed"].items())
                  + f"; среднее {100 * c['mean']:+.1f} п.п.; совместный "
                  f"бутстрэп [{100 * jb['ci95'][0]:+.1f}; "
                  f"{100 * jb['ci95'][1]:+.1f}]")
        pr = summ["project_result"]
        print(f"\n  РЕЗУЛЬТАТ МОДЕЛИ (project_result): {pr['text']}")
        lt = pr["latency"]
        if lt["status"] == "выполнено":
            for bk, row in lt["batches"].items():
                print(f"    задержка {bk}: parents_attn24 "
                      f"{row['parents_attn24_ms']:.1f} мс; " + ", ".join(
                          f"к {k[3:-4]} {v['median']:+.1f}% (p95 "
                          f"{v['p95']:+.1f}%)" for k, v in row.items()
                          if k.startswith("vs_")))
        else:
            print(f"    задержка: не выполнено ({lt['why']})")
        he = summ["hierarchy_specific_evidence"]
        print(f"  ПОЛЬЗА ИЕРАРХИИ (hierarchy_specific_evidence): {he['text']}")
        print(f"    {he['note']}")


def selftest():
    def with_sr(n_succ):
        """n_succ успехов из 150, равномерно по задачам (первые по порядку)."""
        o = {}
        per_t = [n_succ // 10 + (1 if t < n_succ % 10 else 0)
                 for t in range(10)]
        for t in range(10):
            for j, i in enumerate(range(30, 45)):
                o[(f"10/{t}", i)] = j < per_t[t]
        return o

    def make(np_, nf, op=134, of=134, fb=122, cond=True):
        outs, per = {}, {}
        for s in (0, 1):
            o = {lab("new_p", s): with_sr(np_), lab("new_f", s): with_sr(nf),
                 lab("old_p", s): with_sr(op), lab("old_f", s): with_sr(of),
                 lab("mr", s): with_sr(op), "fullbar": with_sr(fb),
                 "fullbar_cached": with_sr(129)}
            if cond:
                o[lab("cond", s)] = with_sr(133)
            outs[s] = o
            opt = {f"k25_s{s}": dict(status="выполнено")} if cond else {
                f"k25_s{s}": dict(status="не выполнено", why="нет полного "
                                  "проверенного набора: проверено 20/30")}
            per[s], st = per_seed(s, dict(bootstrap=dict(n=300, seed=1)), o,
                                  opt)
        return summary(per, outs, st, dict(n=300, seed=1),
                       dict(status="не выполнено", why="тест")), per
    # 1. дерево выросло к parents24 (89 -> 94), новый flat тоже 94: рост
    # к прежней версии есть, преимущества над flat нет, ветка не закрыта
    sm, per = make(141, 141)
    assert "signal" not in sm and "decision" not in sm
    pr, he = sm["project_result"], sm["hierarchy_specific_evidence"]
    assert pr["new_p_vs_old_p"]["mean"] > 0.04
    assert pr["per_seed"][0]["vs_fullbar"]["point"] == "выше"
    assert "не обнаружено" in he["text"] and "равны" in he["sign"]
    assert "условием остановки" in he["note"]
    # 2. дерево 94, новый flat 95: сообщается, ветка не закрывается
    sm2, _p = make(141, 143)
    he2 = sm2["hierarchy_specific_evidence"]
    assert "flat выше" in he2["sign"] and not he2["established"]
    assert sm2["project_result"]["new_p_vs_old_p"]["mean"] > 0
    # 3. явное преимущество дерева в обоих сидах
    sm3, _p = make(148, 120)
    assert sm3["hierarchy_specific_evidence"]["established"]
    # 4. cond отсутствует -> «не выполнено», остальное считается
    sm4, per4 = make(141, 141, cond=False)
    assert per4[0]["new_p_vs_cond"]["status"] == "не выполнено" and \
        "20/30" in per4[0]["new_p_vs_cond"]["why"]
    assert "parents_cond24_sS" not in sm4["rows"] and \
        "parents_cond24_sS" in sm["rows"]
    # совместный бутстрэп = средняя Δ; Holm на 2
    jb = sm["primary"]["joint_paired_bootstrap"]
    assert abs(jb["d"] - sm["primary"]["mean"]) < 1e-12
    assert len(sm["holm_primary"]) == 2
    # код завершения — только полнота набора
    full = {0: dict(a=dict(complete=True)), 1: dict(a=dict(complete=True))}
    assert exit_code(full) == 0
    full[1]["a"]["complete"] = False
    assert exit_code(full) == 3
    # задержка: чтение сопоставимого замера
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "l.json")
        t = dict(total=dict(median=100.0, p95=120.0))
        json.dump(dict(device="V100", batch1=dict(
            parents_attn24=dict(total=dict(median=90.0, p95=100.0)),
            fullbar=t, parents24=dict(total=dict(median=85.0, p95=95.0)))),
            open(p, "w"))
        lt = read_latency(p)
        assert abs(lt["batches"]["batch1"]["vs_fullbar_pct"]["median"] + 10) \
            < 1e-9
        assert read_latency(os.path.join(td, "нет.json"))["status"] == \
            "не выполнено"
    print("самопроверка k26_analyze пройдена: два независимых вывода "
          "(project_result / hierarchy_specific_evidence) без общего "
          "сигнала; регрессии — одинаковый рост дерева и flat (89 -> 94), "
          "flat выше выросшего дерева (94 vs 95), явное преимущество "
          "дерева; cond необязателен («не выполнено» с причиной); "
          "совместный бутстрэп = средняя Δ, Holm на 2; код завершения — "
          "только полнота; чтение задержки")
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
            plans, r, outc, opt = load_seed(s)
            res[s] = r
            if outc is None:
                continue
            plan = plans["parents_attn24"]
            per[s], st = per_seed(s, plan, outc, opt)
            outs[s] = outc
            boot = plan["bootstrap"]
        summ = summary(per, outs, st, boot, read_latency()) \
            if len(per) == len(a.seeds) and len(per) > 1 else None
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
    return exit_code(res)


if __name__ == "__main__":
    sys.exit(main())
