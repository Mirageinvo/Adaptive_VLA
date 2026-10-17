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


BANKS = dict(final=tuple(range(30, 45)), mid=tuple(range(25, 30)))
TASKS = tuple(range(10))


def sha(p):
    import hashlib
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def load_eval(d, label, mark, bank):
    p = os.path.join(d, f"{label}_{mark}_{bank}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def outcomes(ev):
    return {(int(e["task"]), int(e["init_state_id"])): bool(e["success"])
            for e in ev["episodes"]}


def eval_problems(ev, label, mark, bank, config_sha, ckpt_check=True):
    """Банк ровно {задачи 0–9} × {старты банка}, без повторов; метка,
    отметка, конфигурация; для вариантов — чекпойнт (файл, SHA, вариант,
    отметка, конфигурация, отпечаток весов)."""
    p = []
    if (ev.get("label"), ev.get("mark"), ev.get("bank")) != (label, mark,
                                                           bank):
        p.append(f"{label}/{mark}/{bank}: метка, отметка или банк в файле "
                 f"{(ev.get('label'), ev.get('mark'), ev.get('bank'))}")
    if ev.get("config_sha1") != config_sha:
        p.append(f"{label}/{mark}/{bank}: другая конфигурация")
    keys = [(int(e["task"]), int(e["init_state_id"]))
            for e in ev["episodes"]]
    if len(keys) != len(set(keys)):
        p.append(f"{label}/{mark}/{bank}: повтор состояния")
    want = {(t, s) for t in TASKS for s in BANKS[bank]}
    if set(keys) != want:
        p.append(f"{label}/{mark}/{bank}: набор состояний не равен банку "
                 f"{bank}")
    if label != "q0" and ckpt_check:
        import torch
        import k18_prepare as kp
        ck = ev.get("ckpt") or {}
        f = ck.get("path")
        if not f or not os.path.exists(f) or sha(f) != ck.get("sha1"):
            p.append(f"{label}/{mark}: чекпойнт отсутствует или изменён")
        else:
            c = torch.load(f, map_location="cpu", weights_only=False)
            if (c.get("variant"), c.get("mark"), c.get("config_sha1")) != \
                    (label, mark, config_sha):
                p.append(f"{label}/{mark}: чекпойнт другого варианта, "
                         f"отметки или конфигурации")
            if not (kp.state_sha(c["actor"]) == c.get("actor_sha1")
                    == ck.get("actor_sha1")):
                p.append(f"{label}/{mark}: отпечаток весов не совпадает")
            if c.get("vla_build") != ev.get("vla_build"):
                p.append(f"{label}/{mark}: сборка q0 чекпойнта другая")
    return p


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


def run(eval_dir, state_path, config_path):
    import k15h_analyze as ha
    need = [("q0", "q0", "final")] + [(v, "300k", "final")
                                      for v in VARIANTS]
    evs = {lab: load_eval(eval_dir, lab, m, b) for lab, m, b in need}
    miss = [k for k, v in evs.items() if v is None]
    res = dict(eval_dir=eval_dir, missing=miss, complete=not miss)
    if miss:
        return res
    if not os.path.exists(config_path):
        res.update(complete=False, why="нет конфигурации")
        return res
    cs = sha(config_path)
    prob = []
    for lab, m, b in need:
        prob += eval_problems(evs[lab], lab, m, b, cs)
    mids = {}
    for lab, m in [("q0", "q0")] + [(v, mk) for v in VARIANTS
                                    for mk in ("100k", "300k")]:
        ev = load_eval(eval_dir, lab, m, "mid")
        if ev is not None:
            prob += eval_problems(ev, lab, m, "mid", cs)
            mids[(lab, m)] = ev
    builds = {e.get("vla_build") for e in list(evs.values())
              + list(mids.values())}
    if len(builds) != 1:
        prob.append(f"оценки разных сборок q0: {sorted(map(str, builds))}")
    if prob:
        res.update(complete=False, why="; ".join(prob[:6]))
        return res
    outc = {k: outcomes(v) for k, v in evs.items()}
    states = sorted(outc["q0"])
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
    for (lab, m), ev in mids.items():
        o = outcomes(ev)
        mid["q0" if lab == "q0" else f"{lab}@{m}"] = ha.equal_weight(
            o, sorted(o))
    tr = {}
    wlog = []
    if state_path and os.path.exists(state_path):
        import torch
        wlog = torch.load(state_path, map_location="cpu",
                          weights_only=False).get("wave_log", [])
    ids = [r["wave_id"] for r in wlog]
    if len(ids) != len(set(ids)):
        res.update(complete=False, why="повтор записи волны в состоянии")
        return res
    if True:
        for r in wlog:
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
    import shutil
    import tempfile
    import torch
    import k18_prepare as kp
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        cfgp = os.path.join(td, "config.json")
        json.dump(dict(kind="k18_config"), open(cfgp, "w"))
        cs = sha(cfgp)
        base = {(t, s): bool(rng.random() < .8) for t in range(10)
                for s in range(30, 45)}

        def write(lab, m, bank, o, ck=None, cfg_sha=cs, build="B"):
            json.dump(dict(label=lab, mark=m, bank=bank, config_sha1=cfg_sha,
                           vla_build=build, ckpt=ck,
                           episodes=[dict(task=t, init_state_id=s,
                                          success=v, decisions=3,
                                          decisions_executed=3)
                                     for (t, s), v in o],
                           diag=[dict(pos=0, coef=[.1] * 4, norm=.1, f=1.0,
                                      angles=[1, 2, 3, 4])]),
                      open(os.path.join(td, "ev", f"{lab}_{m}_{bank}.json"),
                           "w"))

        def ckpt(v, mark="300k", cfg_sha=cs):
            st = {"w": torch.full((2,), float(len(v)))}
            p = os.path.join(td, f"{v}_{mark}.pt")
            torch.save(dict(actor=st, variant=v, mark=mark,
                            config_sha1=cfg_sha, vla_build="B",
                            actor_sha1=kp.state_sha(st)), p)
            return dict(path=p, sha1=sha(p), actor_sha1=kp.state_sha(st))

        def full():
            os.makedirs(os.path.join(td, "ev"), exist_ok=True)
            write("q0", "q0", "final", list(base.items()))
            for v in VARIANTS:
                o = dict(base)
                if v == "basis4_rl":
                    for k in [k for k in base if not base[k]][:6]:
                        o[k] = True
                write(v, "300k", "final", list(o.items()), ckpt(v))
        ev_dir = os.path.join(td, "ev")
        full()
        r = run(ev_dir, None, cfgp)
        assert r["complete"], r.get("why")
        assert r["primary"]["gain"] == 6 and r["primary"]["loss"] == 0
        assert r["vs_q0"]["pca4_rl"]["gain"] == 0
        # регрессии: каждая даёт неполный набор (код 3)
        wrong = {(t, s): True for t in range(10) for s in range(0, 15)}
        cases = []
        cases.append(lambda: write("pca4_rl", "300k", "final",
                                   list(wrong.items()), ckpt("pca4_rl")))
        cases.append(lambda: write("pca4_rl", "300k", "final",
                                   list(base.items())
                                   + [((0, 30), True)], ckpt("pca4_rl")))
        cases.append(lambda: write("pca4_rl", "300k", "final",
                                   list(base.items()), ckpt("basis4_rl")))
        cases.append(lambda: write("pca4_rl", "300k", "final",
                                   list(base.items()), ckpt("pca4_rl"),
                                   cfg_sha="x"))
        cases.append(lambda: write("pca4_rl", "300k", "final",
                                   list(base.items()),
                                   ckpt("pca4_rl", cfg_sha="x")))
        cases.append(lambda: write("pca4_rl", "300k", "final",
                                   list(base.items()), ckpt("pca4_rl"),
                                   build="C"))
        cases.append(lambda: write("pca4_rl", "100k", "mid",
                                   [((t, s), True) for t in range(10)
                                    for s in range(0, 5)], ckpt("pca4_rl",
                                                                "100k")))
        cases.append(lambda: os.remove(os.path.join(ev_dir,
                                                    "pca4_rl_300k_final.json")))
        for i, mk in enumerate(cases):
            shutil.rmtree(ev_dir)
            full()
            mk()
            rr = run(ev_dir, None, cfgp)
            assert not rr["complete"], (i, rr.get("why"))
    print("самопроверка k18_analyze пройдена: основное и вторичные "
          "сравнения; неправильный банк, повтор, чужой чекпойнт, другая "
          "конфигурация (оценки и чекпойнта), другая сборка q0, неверный "
          "промежуточный банк, отсутствие оценки — без вывода")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-18: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--eval-dir", default="reports/k18/eval")
    ap.add_argument("--state", default="data/k18/run/state.pt")
    ap.add_argument("--config", default="data/k18/config.json")
    ap.add_argument("--out", default="reports/k18/analysis.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    res = run(a.eval_dir, a.state, a.config)
    report(res)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    print(f"  сводка: {a.out}")
    return 0 if res["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
