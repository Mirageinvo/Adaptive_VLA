#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-17: анализ пилота (блок 40 — старты 40–44 — всех задач, 50 парных
состояний; старты использовались в разработке).

Только проверенные блоки; неполный набор — код 3 без вывода.
ОСНОВНОЕ (одно, заранее заданное): held_goal − ctx_memory — парная ΔSR с
равными весами задач, выигрыши/потери, точный Мак-Немар, 95% парный
бутстрэп внутри задач.
ВТОРИЧНЫЕ: каждая модель против q0. По каждой политике: SR по задачам,
окна (начатые, число исполненных позиций, без события), нормы применённых
поправок по позициям и суммарная за окно, частота ограничения. Задача 8 —
описательно (не для выбора модели или момента). Сцепление батча: h18
первого вызова окна против q0 в том же вызове — до первого вмешательства
в батче и после соседа.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BOOT_N, BOOT_SEED = 10000, 20261016
PRIMARY = ("held_goal", "ctx_memory")


def summarize(outc, extra, states):
    import k15h_analyze as ha
    q0 = outc["q0"]
    rows = {}
    for lab, o in outc.items():
        pt = {}
        for k in states:
            v = pt.setdefault(str(k[0]), [0, 0])
            v[0] += bool(o[k])
            v[1] += 1
        rows[lab] = dict(sr_equal_weight=ha.equal_weight(o, states),
                         d_sr=ha.equal_weight(o, states)
                         - ha.equal_weight(q0, states),
                         gain=sum(1 for k in states if o[k] and not q0[k]),
                         loss=sum(1 for k in states if q0[k] and not o[k]),
                         per_task=pt, **extra.get(lab, {}))
    a_, b_ = PRIMARY
    prim = dict(a=a_, b=b_, **ha.compare(outc[a_], outc[b_], states, BOOT_N,
                                         BOOT_SEED))
    vs_q0 = {lab: ha.compare(o, q0, states, BOOT_N, BOOT_SEED)
             for lab, o in outc.items() if lab != "q0"}
    t8 = {lab: [sum(o[k] for k in states if k[0] == 8),
                sum(1 for k in states if k[0] == 8)]
          for lab, o in outc.items()}
    return dict(states=len(states), rows=rows, primary=prim, vs_q0=vs_q0,
                task8_descriptive=t8)


def window_stats(z, d):
    """Окна по средам: исполненные позиции, нормы по позициям, сумма."""
    import k16_protocol as p16
    cor = np.asarray(z["k17_corrected"], bool)
    wp = np.asarray(z["k17_win_pos"])
    na = np.asarray(z["k17_norm_applied"], np.float64)
    cl = np.asarray(z["k17_clipped"], bool) if "k17_clipped" in z else \
        np.zeros_like(cor)
    ds = np.asarray(z["done_step"])
    out = dict(started=0, no_event=0, executed_positions=[0, 0, 0, 0],
               norms=[[], [], []], clipped=[[], [], []], window_sum=[])
    for b in range(cor.shape[1]):
        cs = np.nonzero(cor[:, b])[0]
        if not len(cs):
            out["no_event"] += 1
            continue
        out["started"] += 1
        n_ex, s = 0, 0.0
        for c in cs:
            if p16.reached(int(ds[b]), int(c) + 1):
                n_ex += 1
                k = int(wp[c, b])
                out["norms"][k].append(float(na[c, b]))
                out["clipped"][k].append(bool(cl[c, b]))
                s += float(na[c, b])
        out["executed_positions"][n_ex] += 1
        out["window_sum"].append(s)
    return out


def coupling(z, zq):
    """h18 первого вызова окна против q0: [окон, h18 отличается,
    max|Δ|] до первого вмешательства в батче и после соседа."""
    cor = np.asarray(z["k17_corrected"], bool)
    wp = np.asarray(z["k17_win_pos"])
    first_any = min(np.nonzero(cor.any(1))[0].tolist() or [10 ** 9])
    out = {"before": [0, 0, 0.0], "after": [0, 0, 0.0]}
    for k, b in zip(*np.nonzero(cor & (wp == 0))):
        if k >= len(zq["k17_h18"]):
            continue
        dd = float(np.abs(z["k17_h18"][k, b].astype(np.float32)
                          - zq["k17_h18"][k, b].astype(np.float32)).max())
        key = "after" if k > first_any else "before"
        out[key][0] += 1
        out[key][1] += int(dd > 0)
        out[key][2] = max(out[key][2], dd)
    return out


def _agg(ws):
    def q(x):
        return dict(n=len(x), median=float(np.median(x)) if x else None,
                    p95=float(np.percentile(x, 95)) if x else None)
    return dict(windows_started=ws["started"], no_event=ws["no_event"],
                executed_positions_hist=ws["executed_positions"],
                norm_by_position=[q(x) for x in ws["norms"]],
                clip_rate_by_position=[float(np.mean(x)) if x else None
                                       for x in ws["clipped"]],
                window_norm_sum=q(ws["window_sum"]))


def run(plan_path, run_dir):
    import k16_protocol as p16
    import k17_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan)
    if prob:
        raise SystemExit("входы плана K-17 не совпадают: " + "; ".join(
            prob[:4]))
    ok, gaps = pc.check_run(plan, plan_path, run_dir)
    want = len(plan["labels"]) * len(plan["tasks"]) * len(plan["blocks"])
    res = dict(plan=plan_path, run_dir=run_dir, verified=len(ok),
               planned=want, gaps=gaps,
               complete=len(ok) == want and not gaps)
    if not res["complete"]:
        return res
    outc = {lab: {} for lab in plan["labels"]}
    acc = {lab: dict(started=0, no_event=0, executed_positions=[0] * 4,
                     norms=[[], [], []], clipped=[[], [], []],
                     window_sum=[]) for lab in plan["labels"] if lab != "q0"}
    t8 = {lab: dict(started=0, no_event=0, executed_positions=[0] * 4,
                    norms=[[], [], []], clipped=[[], [], []],
                    window_sum=[]) for lab in acc}
    coup = {lab: {"before": [0, 0, 0.0], "after": [0, 0, 0.0]}
            for lab in acc}
    for (lab, t, b), pth in ok.items():
        d, z = p16.load(pth)
        for i, e in enumerate(d["episodes"]):
            outc[lab][(t, int(e["init_state_id"]))] = bool(e["success"])
        if lab == "q0":
            continue
        ws = window_stats(z, d)
        for tgt in ([acc[lab], t8[lab]] if t == 8 else [acc[lab]]):
            tgt["started"] += ws["started"]
            tgt["no_event"] += ws["no_event"]
            for k in range(4):
                tgt["executed_positions"][k] += ws["executed_positions"][k]
            for k in range(3):
                tgt["norms"][k] += ws["norms"][k]
                tgt["clipped"][k] += ws["clipped"][k]
            tgt["window_sum"] += ws["window_sum"]
        _dq, zq = p16.load(ok[("q0", t, b)])
        cc = coupling(z, zq)
        for key in ("before", "after"):
            coup[lab][key][0] += cc[key][0]
            coup[lab][key][1] += cc[key][1]
            coup[lab][key][2] = max(coup[lab][key][2], cc[key][2])
    extra = {lab: _agg(v) for lab, v in acc.items()}
    res.update(summarize(outc, extra, sorted(outc["q0"])))
    res["task8_windows"] = {lab: _agg(v) for lab, v in t8.items()}
    res["batch_coupling_h18"] = coup
    res["note"] = ("старты 40–44 использовались в разработке; три позиции "
                   "обучены на демонстрациях, не на собственных изменённых "
                   "траекториях")
    return res


def report(res):
    print(f"  проверенных блоков {res['verified']}/{res['planned']}; "
          f"непригодных или отсутствующих {len(res['gaps'])}")
    for g in [g for g in res["gaps"] if g["why"] != "не готов"][:10]:
        print(f"    ВНИМАНИЕ {g['label']} t{g['task']} i{g['block']}: "
              f"{g['why']}")
    if not res["complete"]:
        print("  НАБОР НЕПОЛНЫЙ: вывода нет")
        return
    for lab, r in res["rows"].items():
        pt = " ".join(f"t{t} {v[0]}/{v[1]}" for t, v in r["per_task"].items())
        s = (f"    {lab:11} SR {100 * r['sr_equal_weight']:5.1f}%, ΔSR к q0 "
             f"{100 * r['d_sr']:+5.1f} п.п., +{r['gain']}/−{r['loss']}")
        if lab != "q0":
            nb = r["norm_by_position"]
            s += (f"; окон {r['windows_started']} (без события "
                  f"{r['no_event']}), исполнено позиций 0/1/2/3: "
                  f"{r['executed_positions_hist']}; ||δ|| медиана по "
                  f"позициям " + "/".join(
                      f"{x['median']:.3f}" if x["median"] is not None
                      else "—" for x in nb)
                  + f", сумма за окно {r['window_norm_sum']['median']}")
        print(s + f"; {pt}")
    c = res["primary"]
    print(f"  ОСНОВНОЕ {c['a']} − {c['b']}: ΔSR "
          f"{100 * c['d_sr_equal_weight']:+.1f} п.п. "
          f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
          f"+{c['gain']}/−{c['loss']}, Мак-Немар p {c['mcnemar_p']:.4f}")
    for k, c in res["vs_q0"].items():
        print(f"  вторичное {k} − q0: ΔSR {100 * c['d_sr_equal_weight']:+.1f}"
              f" п.п. [{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}],"
              f" +{c['gain']}/−{c['loss']}, p {c['mcnemar_p']:.4f}")
    print(f"  задача 8 (описательно): {res['task8_descriptive']}")
    print(f"  сцепление батча (h18 первого вызова окна против q0; [окон, "
          f"отличается, max|Δ|]): {res['batch_coupling_h18']}")


def selftest():
    states = [(t, s) for t in range(10) for s in range(40, 45)]
    rng = np.random.default_rng(0)
    q0 = {k: bool(rng.random() < 0.8) for k in states}
    outc = {lab: dict(q0) for lab in ("q0", "ctx_memory", "held_goal")}
    for k in [k for k in states if not q0[k]][:4]:
        outc["held_goal"][k] = True
    r = summarize(outc, {}, states)
    assert r["primary"]["gain"] == 4 and r["primary"]["loss"] == 0
    assert r["vs_q0"]["ctx_memory"]["gain"] == 0
    C, B = 6, 2
    cor = np.zeros((C, B), bool)
    wp = np.full((C, B), -1)
    cor[1:4, 0], wp[1:4, 0] = True, [0, 1, 2]
    cor[3:6, 1], wp[3:6, 1] = True, [0, 1, 2]
    z = {"k17_corrected": cor, "k17_win_pos": wp,
         "k17_h18": np.zeros((C, B, 2, 2), np.float16),
         "k17_norm_applied": np.where(cor, 0.1, np.nan),
         "k17_clipped": np.zeros((C, B), bool),
         "done_step": np.array([-1, 8 * 4 + 1])}
    zq = {"k17_h18": np.zeros((C, B, 2, 2), np.float16)}
    z["k17_h18"][3, 1] = 0.5
    z["k17_h18"][2, 0] = 0.7                     # позиция 1 — не в сцеплении
    c = coupling(z, zq)
    assert c["before"] == [1, 0, 0.0] and c["after"][:2] == [1, 1], c
    ws = window_stats(z, None)
    # среда 1 завершилась на шаге 33: исполнены вызовы 4 и 5 (позиции 0, 1)
    assert ws["executed_positions"] == [0, 0, 1, 1], ws
    assert len(ws["norms"][2]) == 1 and np.isclose(sum(ws["window_sum"]), .5)
    print("самопроверка k17_analyze пройдена: одно основное сравнение, "
          "исполненные позиции окна с учётом завершения, сцепление батча "
          "по первому вызову окна")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-17: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", default="reports/k17/plan.json")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default="reports/k17/analysis.json")
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
