#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-30: анализ H (parents_hybrid24) − C (parents_softplan24) по сидам 0 и 1.

Только проверенные блоки (check_run четырёх планов); 150 уникальных
состояний на вариант/сид (a23.outcomes отказывает при повторе, same_states —
при неполном наборе). Основное — H − C того же сида на 150 парах: SR с
равными весами 10 задач, Δ, выигрыши/потери, точный Мак-Немар; Holm по двум
p; совместный парный бутстрэп (одни индексы внутри задачи для обоих режимов
и обоих сидов; 10 000, seed 20261130). Описательно — H и C против исходного
K-25 (только при полном проверенном наборе того же чекпойнта), parents24,
flat24, fullbar, fullbar_cached. Короткая диагностика лучшей проверки val_sel
(root acc, энтропия, coarse, RMS уровней). Два вывода: project_result и
hard_plan_evidence. SR/p не влияют на код завершения (3 — неполный набор).
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

OUT = "reports/k30/analysis.json"
LATENCY = "reports/k30/latency.json"
ALPHA = 0.05


def lab(kind, seed):
    import k30_model as k30m
    if kind in ("hard", "soft"):
        return k30m.label_of(kind, seed)
    return dict(k25=f"parents_cond24_s{seed}", parents=f"parents24_s{seed}",
                flat=f"flat24_s{seed}").get(kind, kind)


def load_seed(seed):
    import k23_protocol as p23
    import k26_protocol as p26
    import k30_protocol as pc
    res, outc, opt = {}, {}, {}
    plans = {}
    for mode in ("hard", "soft"):
        path = pc.plan_path(seed, mode)
        if not os.path.exists(path):
            res[mode] = dict(plan=path, verified=0, planned=30,
                             complete=False)
            continue
        plan = json.load(open(path, encoding="utf-8"))
        prob = pc.check_plan_inputs(plan)
        if prob:
            raise SystemExit(f"{path}: " + "; ".join(prob[:3]))
        ok, gaps = pc.check_run(plan, path, pc.run_dir(path))
        want = len(plan["cells"]) * len(plan["labels"])
        res[mode] = dict(plan=path, verified=len(ok), planned=want, gaps=gaps,
                         complete=len(ok) == want and not gaps)
        if res[mode]["complete"]:
            outc.update(a23.outcomes(ok))
            plans[mode] = plan
    if not all(r["complete"] for r in res.values()):
        return res, None, {}, None
    plan = plans["hard"]
    outc.update(a23.outcomes(p23.verify_refs(plan)))
    nm = f"k25_s{seed}"
    blocks, why = p26.verify_optional(dict(optional_references={
        nm: plan["optional_references"].get(
            nm, dict(status="не выполнено", why="не закреплён"))}))[nm]
    if blocks is None:
        opt["k25"] = dict(status="не выполнено", why=why)
    else:
        outc.update(a23.outcomes(blocks))
        opt["k25"] = dict(status="выполнено")
    return res, outc, opt, plan


def per_seed(seed, plan, outc, opt):
    import k15h_analyze as ha
    H, C = lab("hard", seed), lab("soft", seed)
    states = a23.same_states(outc, [H, C], 150)
    boot = plan["bootstrap"]
    o = dict(seed=seed, states=len(states),
             sr={k: ha.equal_weight(outc[k], states) for k in (H, C)},
             primary=a23.cmp(outc, H, C, states, boot),
             per_task={k: a23.per_task(outc[k], states) for k in (H, C)},
             descriptive={})
    for kind in ("k25", "parents", "flat", "fullbar", "fullbar_cached"):
        L_ = lab(kind, seed)
        if kind == "k25" and opt.get("k25", {}).get("status") != "выполнено":
            o["descriptive"][kind] = dict(status="не выполнено",
                                          why=opt.get("k25", {}).get("why"))
            continue
        if L_ not in outc or sorted(outc[L_]) != states:
            o["descriptive"][kind] = dict(status="не выполнено",
                                          why="нет полного набора")
            continue
        o["sr"][L_] = ha.equal_weight(outc[L_], states)
        o["descriptive"][kind] = dict(
            status="выполнено",
            H=a23.cmp(outc, H, L_, states, boot,
                      kind in ("fullbar", "fullbar_cached")),
            C=a23.cmp(outc, C, L_, states, boot,
                      kind in ("fullbar", "fullbar_cached")))
    o["diagnostics"] = {}
    for mode in ("hard", "soft"):
        tr = f"reports/k30/train_s{seed}_{mode}.json"
        if os.path.exists(tr):
            r = json.load(open(tr, encoding="utf-8"))
            u = r["best"][0] if r.get("best") else None
            row = next((c["val"] for c in r["curve"]
                        if c["phase"] == "main" and c["u"] == u), None)
            if row:
                o["diagnostics"][mode] = {k: row.get(k) for k in (
                    "L_action", "l1_8", "root_acc", "entropy", "L_coarse",
                    "codes_used", "rms_G0", "rms_G1", "rms_G2")}
                o["diagnostics"][mode]["best_update"] = u
    return o, states


def joint_boot(pairs, states, boot):
    """pairs {seed: (H dict, C dict)}; одни индексы внутри задачи."""
    seeds = sorted(pairs)
    by = {}
    for k in states:
        by.setdefault(k[0], []).append(k)
    D = {s: {t: np.array([float(pairs[s][0][k]) - float(pairs[s][1][k])
                          for k in v]) for t, v in by.items()} for s in seeds}
    est = float(np.mean([np.mean([D[s][t].mean() for t in by])
                         for s in seeds]))
    rng = np.random.default_rng(boot["seed"])
    bs = np.empty(boot["n"])
    for i in range(boot["n"]):
        idx = {t: rng.integers(0, len(v), len(v)) for t, v in by.items()}
        bs[i] = np.mean([np.mean([D[s][t][idx[t]].mean() for t in by])
                         for s in seeds])
    return dict(d=est, ci95=[float(np.percentile(bs, 2.5)),
                             float(np.percentile(bs, 97.5))])


def read_latency(path=LATENCY):
    if not os.path.exists(path):
        return dict(status="не выполнено", why="латентность ожидается")
    return dict(status="выполнено", **json.load(open(path, encoding="utf-8")))


def summary(per, pairs, states, boot, latency):
    import k15h_analyze as ha
    seeds = sorted(per)
    d = {s: per[s]["primary"]["d_sr_equal_weight"] for s in seeds}
    holm = dict(zip(seeds, ha.holm([per[s]["primary"]["mcnemar_p"]
                                    for s in seeds])))
    jb = joint_boot(pairs, states, boot)
    if all(v > 0 for v in d.values()):
        sign = "точечно H выше C в обоих сидах"
    elif all(v < 0 for v in d.values()):
        sign = "точечно H ниже C в обоих сидах"
    elif all(v == 0 for v in d.values()):
        sign = "точечно равны в обоих сидах"
    else:
        sign = "знак разности различается между сидами"
    est = all(h < ALPHA for h in holm.values()) and \
        (jb["ci95"][0] > 0 or jb["ci95"][1] < 0) and \
        len({np.sign(v) for v in d.values()}) == 1
    return dict(
        seeds=seeds, d_per_seed=d, d_mean=float(np.mean(list(d.values()))),
        holm=holm, joint=jb,
        hard_plan_evidence=dict(
            sign=sign, established=est,
            text=f"{sign}; " + ("различие статистически установлено" if est
                                else "преимущество не обнаружено (интервал "
                                     "или Holm)")),
        project_result=dict(
            sr={s: per[s]["sr"] for s in seeds},
            descriptive={s: {k: (v.get("H", {}).get("d_sr_equal_weight"),
                                 v.get("C", {}).get("d_sr_equal_weight"))
                             if v["status"] == "выполнено" else v["status"]
                             for k, v in per[s]["descriptive"].items()}
                         for s in seeds},
            latency=latency),
        note="разведочный пилот на многократно использованном банке 30–44; "
             "интервал условен на двух обучениях и одном общем кодеке")


def report(res, per, summ):
    line = a23.line
    for s, r in res.items():
        print(f"  seed {s}: " + ", ".join(
            f"{m} {v['verified']}/{v['planned']}" + ("" if v["complete"]
                                                     else " НЕПОЛНО")
            for m, v in r.items()))
    for s, o in sorted(per.items()):
        print(f"\n  SEED {s} (150 состояний): " + ", ".join(
            f"{k} {100 * v:.1f}%" for k, v in o["sr"].items()))
        print(f"    основное H − C: {line(o['primary'])}" + (
            f"; Holm {summ['holm'][s]:.4f}" if summ else ""))
        H, C = lab("hard", s), lab("soft", s)
        print("    по задачам H/C: " + ", ".join(
            f"{t.split('/')[1]}: {v[0]}/{o['per_task'][C][t][0]}"
            for t, v in sorted(o["per_task"][H].items(),
                               key=lambda x: int(x[0].split('/')[1]))))
        for k, v in o["descriptive"].items():
            print(f"    описательно к {k}: " + (
                f"H {line(v['H'])}; C {line(v['C'])}"
                if v["status"] == "выполнено" else
                f"не выполнено ({v.get('why')})"))
        for m, dg in o.get("diagnostics", {}).items():
            print(f"    {m} @{dg['best_update']}: L_action "
                  f"{dg['L_action']:.4f}, root acc {dg['root_acc']:.3f}, "
                  f"энтропия {dg['entropy']:.2f}, coarse "
                  f"{dg['L_coarse']:.4f}, кодов {dg['codes_used']}, RMS "
                  f"{dg['rms_G0']:.3f}/{dg['rms_G1']:.3f}/{dg['rms_G2']:.3f}")
    if summ:
        j = summ["joint"]
        print("\n  ДВА СИДА: H − C " + ", ".join(
            f"s{s} {100 * v:+.1f}" for s, v in summ["d_per_seed"].items()) +
              f"; среднее {100 * summ['d_mean']:+.1f} п.п. [{100 * j['ci95'][0]:+.1f}; "
              f"{100 * j['ci95'][1]:+.1f}]; Holm {summ['holm']}")
        print(f"  СВИДЕТЕЛЬСТВО ЖЁСТКОГО ПЛАНА: "
              f"{summ['hard_plan_evidence']['text']}")
        lt = summ["project_result"]["latency"]
        print(f"  задержка: {lt.get('status')}" + (
            f" — H/C медиана b1 {lt.get('batch1', {}).get('ratio_h_c')}, b5 "
            f"{lt.get('batch5', {}).get('ratio_h_c')}"
            if lt.get("status") == "выполнено" else ""))


def selftest():
    def with_sr(n):
        per_t = [n // 10 + (1 if t < n % 10 else 0) for t in range(10)]
        return {(f"10/{t}", i): j < per_t[t] for t in range(10)
                for j, i in enumerate(range(30, 45))}
    outs = {}
    per = {}
    for s in (0, 1):
        o = {lab("hard", s): with_sr(136), lab("soft", s): with_sr(130),
             "fullbar": with_sr(122), "fullbar_cached": with_sr(129)}
        outs[s] = o
        plan = dict(bootstrap=dict(n=300, seed=1))
        per[s], st = per_seed(s, plan, o, dict(k25=dict(
            status="не выполнено", why="тест")))
    pairs = {s: (outs[s][lab("hard", s)], outs[s][lab("soft", s)])
             for s in (0, 1)}
    sm = summary(per, pairs, st, dict(n=300, seed=1),
                 dict(status="не выполнено"))
    assert abs(sm["joint"]["d"] - sm["d_mean"]) < 1e-12
    assert "выше" in sm["hard_plan_evidence"]["sign"]
    assert per[0]["descriptive"]["k25"]["status"] == "не выполнено"
    assert per[0]["descriptive"]["fullbar"]["status"] == "выполнено"
    bad = dict(outs[0])
    bad[lab("hard", 0)] = {k: v for k, v in list(bad[lab("hard", 0)].items())
                           [:-1]}
    try:
        per_seed(0, dict(bootstrap=dict(n=10, seed=1)), bad, {})
        raise AssertionError("неполный набор принят")
    except SystemExit:
        pass
    print("самопроверка k30_analyze пройдена: H − C, Holm, совместный "
          "бутстрэп = средняя Δ, описательные контроли с причиной "
          "отсутствия, неполный набор — отказ")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-30: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    res, per, pairs, st, boot = {}, {}, {}, None, None
    try:
        for s in a.seeds:
            r, outc, opt, plan = load_seed(s)
            res[s] = r
            if outc is None:
                continue
            per[s], st = per_seed(s, plan, outc, opt)
            pairs[s] = (outc[lab("hard", s)], outc[lab("soft", s)])
            boot = plan["bootstrap"]
        summ = summary(per, pairs, st, boot, read_latency()) \
            if len(per) == len(a.seeds) and len(per) > 1 else None
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    report(res, per, summ)
    os.makedirs("reports/k30", exist_ok=True)
    out = OUT if summ else \
        f"reports/k30/analysis_s{'_'.join(map(str, a.seeds))}.json"
    json.dump(dict(plans=res, per_seed=per, summary=summ),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=str)
    print(f"  сводка: {out}")
    return 0 if all(all(v["complete"] for v in r.values())
                    for r in res.values()) else 3


if __name__ == "__main__":
    sys.exit(main())
