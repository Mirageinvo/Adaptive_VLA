#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-18: анализ (итоговые 150 состояний, старты 30–44, чекпойнт 300 тыс.).

ОСНОВНОЕ (одно): basis4_rl − pca4_rl — парная ΔSR с равными весами задач,
выигрыши/потери, точный Мак-Немар, 95% парный бутстрэп внутри задач.
ВТОРИЧНЫЕ: basis4 − full48, pca4 − full48, каждый вариант − q0.
Промежуточные SR (старты 25–29) на отметках 100 и 300 тыс. Диагностика:
главные углы базиса basis4 к PCA, коэффициенты, нормы и множитель
ограничения, оценки критиков и TD, представление задач в replay,
задача 8 (описательно). Неполный набор — код 3 без вывода.
Интервалы относятся к этим обученным политикам (один сид обучения).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BOOT_N, BOOT_SEED = 10000, 20261017
VARIANTS = ("full48_rl", "pca4_rl", "basis4_rl")
PRIMARY = ("basis4_rl", "pca4_rl")
SECONDARY = (("basis4_rl", "full48_rl"), ("pca4_rl", "full48_rl"))


def load_eval(d, label, mark, bank):
    p = os.path.join(d, f"{label}_{mark}_{bank}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def outcomes(ev):
    return {(int(e["task"]), int(e["init_state_id"])): bool(e["success"])
            for e in ev["episodes"]}


def diag_summary(ev):
    dg = ev.get("diag") or []
    if not dg:
        return None
    nr = np.array([x["norm"] for x in dg])
    f = np.array([x["f"] for x in dg])
    c = np.array([np.linalg.norm(x["coef"]) for x in dg])
    out = dict(decisions=len(dg), norm_median=float(np.median(nr)),
               norm_p95=float(np.percentile(nr, 95)),
               shrink_rate=float((f < 1).mean()),
               coef_norm_median=float(np.median(c)),
               coef_norm_p95=float(np.percentile(c, 95)))
    ang = [x["angles"] for x in dg if x.get("angles")]
    if ang:
        A = np.array(ang)
        out["principal_angles_deg"] = dict(
            max_median=float(np.median(A.max(1))),
            max_p95=float(np.percentile(A.max(1), 95)),
            mean_median=float(np.median(A.mean(1))))
    pe = ev["episodes"]
    out["decisions_executed"] = int(sum(e.get("decisions_executed", 0)
                                        for e in pe))
    out["episodes_with_window"] = int(sum(e.get("decisions", 0) > 0
                                          for e in pe))
    return out


def run(eval_dir, waves_log):
    import k15h_analyze as ha
    need = [("q0", "q0", "final")] + [(v, "300k", "final")
                                      for v in VARIANTS]
    evs = {lab: load_eval(eval_dir, lab, m, b) for lab, m, b in need}
    miss = [k for k, v in evs.items() if v is None]
    res = dict(eval_dir=eval_dir, missing=miss, complete=not miss)
    if miss:
        return res
    outc = {k: outcomes(v) for k, v in evs.items()}
    states = sorted(outc["q0"])
    if len(states) != 150 or any(sorted(o) != states for o in
                                 outc.values()):
        res.update(complete=False, why="состояния оценок не совпадают или "
                   "их не 150")
        return res
    rows = {}
    for lab, o in outc.items():
        pt = {}
        for k in states:
            x = pt.setdefault(str(k[0]), [0, 0])
            x[0] += o[k]
            x[1] += 1
        rows[lab] = dict(sr_equal_weight=ha.equal_weight(o, states),
                         gain=sum(o[k] and not outc["q0"][k] for k in states),
                         loss=sum(outc["q0"][k] and not o[k] for k in states),
                         per_task=pt,
                         diag=diag_summary(evs[lab]) if lab != "q0" else None)
    a_, b_ = PRIMARY
    prim = dict(a=a_, b=b_, **ha.compare(outc[a_], outc[b_], states, BOOT_N,
                                         BOOT_SEED))
    sec = [dict(a=x, b=y, **ha.compare(outc[x], outc[y], states, BOOT_N,
                                       BOOT_SEED)) for x, y in SECONDARY]
    vs_q0 = {v: ha.compare(outc[v], outc["q0"], states, BOOT_N, BOOT_SEED)
             for v in VARIANTS}
    mid = {}
    for mark in ("100k", "300k"):
        for lab in VARIANTS:
            ev = load_eval(eval_dir, lab, mark, "mid")
            if ev:
                o = outcomes(ev)
                mid[f"{lab}@{mark}"] = ha.equal_weight(o, sorted(o))
    q0m = load_eval(eval_dir, "q0", "q0", "mid")
    if q0m:
        o = outcomes(q0m)
        mid["q0"] = ha.equal_weight(o, sorted(o))
    tr = {}
    if os.path.exists(waves_log):
        for ln in open(waves_log):
            r = json.loads(ln)
            t = tr.setdefault(r["variant"], dict(waves=0, transitions=0,
                                                 td=[], q=[], tasks={},
                                                 train_success=0, eps=0))
            t["waves"] += 1
            t["transitions"] += r["transitions"]
            t["tasks"][str(r["task"])] = t["tasks"].get(str(r["task"]), 0) \
                + r["transitions"]
            t["train_success"] += sum(r["success"])
            t["eps"] += len(r["success"])
            if r["updates"]["td"] is not None:
                t["td"].append(r["updates"]["td"])
                t["q"].append(r["updates"]["q"])
        for t in tr.values():
            for k in ("td", "q"):
                x = t.pop(k)
                t[k] = dict(first=x[:5], last=x[-5:]) if x else None
            t["train_sr_with_noise"] = t["train_success"] / max(t["eps"], 1)
    t8 = {lab: rows[lab]["per_task"].get("8") for lab in rows}
    res.update(states=len(states), rows=rows, primary=prim, secondary=sec,
               vs_q0=vs_q0, mid_sr=mid, training=tr, task8_descriptive=t8,
               note=("старты 30–44 отложены от RL-обучения K-18, но "
                     "использовались в проекте раньше; один сид обучения"))
    return res


def report(res):
    if not res["complete"]:
        print(f"  НАБОР НЕПОЛНЫЙ ({res.get('missing') or res.get('why')}): "
              f"вывода нет")
        return
    for lab, r in res["rows"].items():
        pt = " ".join(f"t{t} {v[0]}/{v[1]}" for t, v in r["per_task"].items())
        s = (f"    {lab:10} SR {100 * r['sr_equal_weight']:5.1f}%, к q0 "
             f"+{r['gain']}/−{r['loss']}")
        if r["diag"]:
            d = r["diag"]
            s += (f"; решений {d['decisions']} (исполнено "
                  f"{d['decisions_executed']}), ||δ|| медиана "
                  f"{d['norm_median']:.3f}, p95 {d['norm_p95']:.3f}, "
                  f"ограничение команд {100 * d['shrink_rate']:.1f}%")
            if "principal_angles_deg" in d:
                s += (f", главный угол к PCA (макс., медиана) "
                      f"{d['principal_angles_deg']['max_median']:.1f}°")
        print(s + f"; {pt}")
    c = res["primary"]
    print(f"  ОСНОВНОЕ {c['a']} − {c['b']}: ΔSR "
          f"{100 * c['d_sr_equal_weight']:+.1f} п.п. "
          f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
          f"+{c['gain']}/−{c['loss']}, Мак-Немар p {c['mcnemar_p']:.4f}")
    for c in res["secondary"]:
        print(f"  вторичное {c['a']} − {c['b']}: ΔSR "
              f"{100 * c['d_sr_equal_weight']:+.1f} п.п. "
              f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
              f"+{c['gain']}/−{c['loss']}, p {c['mcnemar_p']:.4f}")
    for k, c in res["vs_q0"].items():
        print(f"  вторичное {k} − q0: ΔSR {100 * c['d_sr_equal_weight']:+.1f}"
              f" п.п. [{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}],"
              f" +{c['gain']}/−{c['loss']}, p {c['mcnemar_p']:.4f}")
    print("  промежуточные SR (25–29): " + ", ".join(
        f"{k} {100 * v:.1f}%" for k, v in res["mid_sr"].items()))
    for v, t in res["training"].items():
        print(f"  обучение {v}: волн {t['waves']}, переходов "
              f"{t['transitions']}, SR обучающих эпизодов с шумом "
              f"{100 * t['train_sr_with_noise']:.1f}%, TD первые/последние "
              f"{t['td']}")
    print(f"  задача 8 (описательно): {res['task8_descriptive']}")


def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        base = {(t, s): bool(rng.random() < .8) for t in range(10)
                for s in range(30, 45)}
        for lab, m in [("q0", "q0")] + [(v, "300k") for v in VARIANTS]:
            o = dict(base)
            if lab == "basis4_rl":
                for k in [k for k in base if not base[k]][:6]:
                    o[k] = True
            json.dump(dict(episodes=[dict(task=t, init_state_id=s,
                                          success=v, decisions=3,
                                          decisions_executed=3)
                                     for (t, s), v in o.items()],
                           diag=[dict(pos=0, coef=[.1] * 4, norm=.1, f=1.0,
                                      angles=[1, 2, 3, 4])]),
                      open(os.path.join(td, f"{lab}_{m}_final.json"), "w"))
        r = run(td, os.path.join(td, "none.jsonl"))
        assert r["complete"] and r["primary"]["gain"] == 6
        assert r["primary"]["loss"] == 0 and r["vs_q0"]["pca4_rl"]["gain"] == 0
        os.remove(os.path.join(td, "pca4_rl_300k_final.json"))
        assert not run(td, "x")["complete"]
    print("самопроверка k18_analyze пройдена: основное сравнение, вторичные, "
          "неполный набор — без вывода")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-18: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--eval-dir", default="reports/k18/eval")
    ap.add_argument("--waves-log", default="logs/k18/waves.jsonl")
    ap.add_argument("--out", default="reports/k18/analysis.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    res = run(a.eval_dir, a.waves_log)
    report(res)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    print(f"  сводка: {a.out}")
    return 0 if res["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
