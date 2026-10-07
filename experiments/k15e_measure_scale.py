#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15e M0: анализ раскаток семейства a0 + α·δ на всех задачах.

РУКИ: q0 (канонический черновик), a000, a025, a050, a075, a100 (масштабы
поправки h18) и контроль am050, am100 (та же поправка в обратную сторону).

ЧТО СЧИТАЕТСЯ (равный вес задач, кластер = задача × состояние):
  * успех каждой руки, rescue/harm каждого α против q0;
  * оракул best-of-set по кластеру для наборов
      dir5  = {0, .25, .5, .75, 1}       — семейство из плана;
      dir3  = {0, .5, 1}  и  ctrl3 = {0, −.5, −1} — РАВНЫЕ по размеру
              наборы «по направлению δ» и «против него»;
  * разность оракулов dir3 − ctrl3 с 90 % интервалом — сколько оракульного
    выигрыша даёт именно направление δ, а не разнообразие траекторий;
  * для каждого кластера — набор α с успехом, гистограмма «лучшего» α
    (наименьший |α| среди успешных; при общем провале — нет);
  * тождества: доля эпизодов, где действия a000 побитово равны q0, и
    a100 — прежней руке h18 K-15d на тех же стартах (если даны её
    артефакты).

РЕШЕНИЕ О ДОРОГОМ ДАТАСЕТЕ (план §3.2, плюс контроль направления):
  перспективно  выигрыш оракула dir5 >= 5 п.п. И dir3 − ctrl3 >= 2 п.п.;
  закрыть       выигрыш dir5 <= 2 п.п. ИЛИ dir3 − ctrl3 <= 0;
  иначе         расширить probe до 10 состояний на задачу.
Это решение о сборе данных, а не критерий эффективности метода.

КОДЫ: 0 — набор исправен (решение в сводке); 3 — технический отказ.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIR5 = ("a000", "a025", "a050", "a075", "a100")
DIR3 = ("a000", "a050", "a100")
CTRL3 = ("a000", "am050", "am100")
ALL_ALPHA = DIR5 + ("am050", "am100")
DECISION = dict(promising_gain=0.05, promising_dir_minus_ctrl=0.02,
                close_gain=0.02, close_dir_minus_ctrl=0.0,
                registered="07.10.2026, до роллаутов M0")


def check_arm(d, arm):
    """Метка исполняет своё: q0 — depthrvq/fast, aXXX — k15e с этим α."""
    import k15e_candidates as kc
    j = d.get("joint") or {}
    if arm == "q0":
        ok = (d.get("policy") == "depthrvq" and d.get("levels") == 1
              and (d.get("argv") or {}).get("depth_rvq_mode") == "fast")
        return [] if ok else [f"q0: не depthrvq/fast ({d.get('policy')})"]
    want = kc.label_alpha(arm)
    bad = []
    if d.get("policy") != "k15e":
        bad.append(f"{arm}: policy {d.get('policy')!r}")
    if j.get("alpha") is None or abs(float(j["alpha"]) - want) > 1e-12:
        bad.append(f"{arm}: α {j.get('alpha')!r}, ожидалось {want}")
    if j.get("layers_per_call") != 18 or j.get("phase") != "h1p1":
        bad.append(f"{arm}: не h18 ({j.get('phase')}, "
                   f"{j.get('layers_per_call')} слоёв)")
    return bad


def oracle(sr, arms):
    """Успех хотя бы одной руки набора по кластеру."""
    return np.max(np.stack([sr[a] for a in arms]), axis=0)


def decide(gain5, dir_minus_ctrl, rule=DECISION):
    if gain5 <= rule["close_gain"] or dir_minus_ctrl <= \
            rule["close_dir_minus_ctrl"]:
        return "закрыть одномерное масштабирование"
    if gain5 >= rule["promising_gain"] and dir_minus_ctrl >= \
            rule["promising_dir_minus_ctrl"]:
        return "перспективно: строить replay gate и пилот датасета"
    return "неясно: расширить probe до 10 состояний на задачу"


def best_alpha(sr, clusters_n):
    """Наименьший |α| среди успешных кандидатов dir5 по кластеру."""
    import k15e_candidates as kc
    order = sorted(DIR5, key=lambda a: abs(kc.label_alpha(a)))
    hist = {a: 0 for a in DIR5}
    none = 0
    for i in range(clusters_n):
        hit = [a for a in order if sr[a][i] > 0.5]
        if hit:
            hist[hit[0]] += 1
        else:
            none += 1
    return hist, none


def action_identity(files_a, files_b):
    """Доля эпизодов с побитово равными исполненными действиями."""
    def hashes(files):
        out = {}
        for p in files:
            d = json.load(open(p))
            for e in d["episodes"]:
                out[(int(d["task_id"]), int(e["init_state_id"]))] = \
                    e.get("action_sha1")
        return out
    ha, hb = hashes(files_a), hashes(files_b)
    keys = sorted(set(ha) & set(hb))
    same = sum(ha[k] == hb[k] for k in keys)
    return dict(compared=len(keys), identical=int(same),
                share=(same / len(keys) if keys else None))


def main():
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import k14q_behavior as kb
    ap = argparse.ArgumentParser(description="K-15e M0: анализ масштабов")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--arts", nargs="*", default=[])
    ap.add_argument("--h18-arts", nargs="*", default=[],
                    help="артефакты руки h18 K-15d на тех же стартах — для "
                         "сверки a100")
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--states", default="0,1,2,3,4")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.arts or not a.out:
        raise SystemExit("нужны --arts и --out")
    obs, meta = kb.load(a.arts)
    arms = ["q0"] + [x for x in ALL_ALPHA if x in meta]
    if set(meta) != set(arms):
        raise SystemExit(f"посторонние руки: {sorted(set(meta) - set(arms))}")
    technical = []
    missing = [x for x in ALL_ALPHA if x not in meta]
    if missing and not a.allow_partial:
        technical.append(f"нет рук {missing}")
    for nm in arms:
        for p in meta[nm]["files"]:
            technical += [f"{os.path.basename(p)}: {x}"
                          for x in check_arm(json.load(open(p)), nm)]
            if str(json.load(open(p)).get("suite")) != "10":
                technical.append(f"{os.path.basename(p)}: не LIBERO-10")
        if len(meta[nm]["fingerprints"]) != 1:
            technical.append(f"у руки {nm} несколько моделей")
    shas = set()
    for nm in arms:
        shas |= set(meta[nm]["script_shas"])
    if len(shas) > 1:
        technical.append(f"разные версии харнесса: {sorted(shas)}")
    clusters, seeds = kb.align(obs, arms)
    want = sorted((int(t), int(s)) for t in a.tasks.split(",")
                  for s in a.states.split(","))
    if clusters != want and not a.allow_partial:
        technical.append(f"кластеров {len(clusters)} из {len(want)}")

    sr = {nm: kb.cluster_means(obs, nm, clusters, seeds) for nm in arms}
    base = sr["q0"]
    vals = {f"sr_{nm}": sr[nm] for nm in arms}
    for nm in arms[1:]:
        res, hrm = kb.pair_rates(obs, "q0", nm, clusters, seeds)
        vals[f"rescue_{nm}"], vals[f"harm_{nm}"] = res, hrm
    sets = {"dir5": [x for x in DIR5 if x in sr],
            "dir3": [x for x in DIR3 if x in sr],
            "ctrl3": [x for x in CTRL3 if x in sr]}
    for k, v in sets.items():
        if v:
            vals[f"oracle_{k}"] = oracle(sr, v)
            vals[f"gain_{k}"] = vals[f"oracle_{k}"] - base
    if sets["dir3"] == list(DIR3) and sets["ctrl3"] == list(CTRL3):
        vals["dir3_minus_ctrl3"] = vals["oracle_dir3"] - vals["oracle_ctrl3"]
    ci, _ = kb.boot(vals, clusters, qs=(kb.Q_LO, kb.Q_HI))
    point = {k: kb.point(v, clusters) for k, v in vals.items()}
    per_task = {}
    for t, idx in kb.strata(clusters).items():
        per_task[int(t)] = {k: float(np.asarray(v)[idx].mean())
                            for k, v in vals.items()
                            if k.startswith(("sr_", "oracle_"))}
    hist, none = (best_alpha(sr, len(clusters))
                  if sets["dir5"] == list(DIR5) else ({}, None))
    ident = {}
    if "a000" in meta:
        ident["a000_vs_q0"] = action_identity(meta["a000"]["files"],
                                              meta["q0"]["files"])
    if "a100" in meta and a.h18_arts:
        ident["a100_vs_h18_k15d"] = action_identity(meta["a100"]["files"],
                                                    a.h18_arts)
    gain5 = point.get("gain_dir5")
    dmc = point.get("dir3_minus_ctrl3")
    decision = (decide(gain5, dmc) if gain5 is not None and dmc is not None
                else "не определено: неполный набор рук")
    code = 3 if technical else 0

    print(f"  кластеров {len(clusters)}, сиды {seeds}, руки {arms}")
    print("  успех: " + ", ".join(f"{nm} {point['sr_' + nm]:.3f}"
                                  for nm in arms))
    for nm in arms[1:]:
        print(f"    {nm}: rescue {point['rescue_' + nm]:.3f}, harm "
              f"{point['harm_' + nm]:.3f}")
    for k in ("dir5", "dir3", "ctrl3"):
        if f"gain_{k}" in point:
            print(f"  оракул {k} {point['oracle_' + k]:.3f}, выигрыш к q0 "
                  f"{point['gain_' + k]:+.3f} [{ci['gain_' + k][0]:+.3f}, "
                  f"{ci['gain_' + k][1]:+.3f}]")
    if dmc is not None:
        print(f"  НАПРАВЛЕНИЕ: dir3 − ctrl3 = {dmc:+.3f} "
              f"[{ci['dir3_minus_ctrl3'][0]:+.3f}, "
              f"{ci['dir3_minus_ctrl3'][1]:+.3f}]")
    print(f"  лучший α (наименьший |α| с успехом): {hist}, без успеха "
          f"{none}")
    for k, v in ident.items():
        print(f"  тождество {k}: {v['identical']}/{v['compared']} эпизодов")
    print("  по задачам: " + "; ".join(
        f"{t}: q0 {v['sr_q0']:.2f}, dir5 {v.get('oracle_dir5', float('nan')):.2f}"
        for t, v in per_task.items()))
    print(f"  РЕШЕНИЕ M0: {decision}" + (f"; ТЕХНИЧЕСКИЙ ОТКАЗ {technical}"
                                         if technical else ""))
    out = dict(kind="k15e_m0_scale", technical=technical, code=code,
               decision=decision, decision_rule=DECISION, point=point,
               ci90=ci, per_task=per_task, best_alpha_hist=hist,
               no_success_clusters=none, identity=ident, arms=arms,
               clusters=len(clusters), seeds=seeds,
               script_sha1=sorted(shas), partial=bool(a.allow_partial),
               fingerprints={nm: sorted(meta[nm]["fingerprints"])
                             for nm in arms})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = a.out + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, allow_nan=False)
    os.replace(tmp, a.out)
    print(f"  сводка: {a.out}")
    return code


def selftest():
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    assert decide(0.06, 0.03).startswith("перспективно")
    assert decide(0.06, 0.0).startswith("закрыть")      # только разнообразие
    assert decide(0.02, 0.05).startswith("закрыть")
    assert decide(0.04, 0.01).startswith("неясно")
    sr = {"a": np.array([0, 1, 0.0]), "b": np.array([1, 0, 0.0])}
    assert oracle(sr, ["a", "b"]).tolist() == [1, 1, 0]
    s5 = {k: np.array([0.0, 0.0, 0.0]) for k in DIR5}
    s5["a050"][0] = 1.0
    s5["a100"][0] = 1.0
    s5["a100"][1] = 1.0
    h, n = best_alpha(s5, 3)
    assert h["a050"] == 1 and h["a100"] == 1 and n == 1
    q0 = dict(policy="depthrvq", levels=1, argv=dict(depth_rvq_mode="fast"))
    assert check_arm(q0, "q0") == []
    e = lambda al, ph="h1p1", ly=18: dict(policy="k15e", joint=dict(
        alpha=al, phase=ph, layers_per_call=ly))
    assert check_arm(e(0.25), "a025") == []
    assert check_arm(e(-0.5), "am050") == []
    assert check_arm(e(0.5), "a025")
    assert check_arm(e(0.25, ly=24), "a025")
    assert check_arm(dict(q0), "a000")
    print("самопроверка k15e_measure_scale пройдена")
    return 0


if __name__ == "__main__":
    sys.exit(main())
