#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15d: парный поведенческий анализ q0, D0 и H1.

ФУНКЦИИ K-14q ИМПОРТИРУЮТСЯ: загрузка артефактов k9h, парность по
(задача, состояние, сид) с поэпизодной сверкой стартов и сидов, кластерные
средние, rescue/harm, стратифицированный по задачам кластерный бутстрап с
равным весом задач.

РУКИ: q0; d0 (a_d, 24 слоя); h18 (a1 чекпойнта h1p1, проход до 18-го
слоя); h1 (a2 чекпойнта h1p2, 24 слоя). Состав задаётся --expected-arms и
проверяется ТОЧНО: пропущенная рука — технический отказ, а не тихое
сравнение оставшихся.

ПАРЫ: каждая рука против q0; h1 и h18 против d0; h1 против h18. Для каждой
— успех обеих рук, разность с двусторонним 90 % интервалом, rescue, harm и
по задачам.

РЕЖИМЫ (план §13):
    safety  задачи 8-9, 50 кластеров. Технический пилот: набор полон и
            парен, одна версия харнесса, одна модель у руки, действия
            конечны, и ни у D0, ни у H1 нет катастрофы — разность успеха
            против q0 не ниже CATASTROPHE (-0.15). Это НЕ вывод об
            эффективности.
    target  прицельный пилот на задачах --tasks (состояния 0-24): та же
            проверка катастрофы, что в safety, на задаче, где фильтр
            допуска нашёл риск.
    dev     250 кластеров. Зарегистрированного порога успеха в плане нет:
            печатаются величины, решение — по ним человеком. Код 0 при
            технической исправности.

КОДЫ: 0 — пройдено; 4 — катастрофа в safety; 3 — технический отказ.
"""
import argparse
import json
import os
import sys

import numpy as np

CATASTROPHE = -0.15
MODES = dict(safety=dict(tasks=[8, 9], states=list(range(25))),
             dev=dict(tasks=list(range(10)), states=list(range(25))),
             target=dict(tasks=None, states=list(range(25))))
SEEDS = [101]
ARMS = ("q0", "d0", "h18", "h1")
PAIRS = (("q0", "d0"), ("q0", "h18"), ("q0", "h1"), ("d0", "h18"),
         ("d0", "h1"), ("h18", "h1"))


def expected_clusters(mode, tasks=None):
    m = MODES[mode]
    tasks = m["tasks"] if m["tasks"] is not None else tasks
    if not tasks:
        raise SystemExit(f"режим {mode} требует --tasks")
    return sorted((t, s) for t in tasks for s in m["states"])


def parse_arms(text):
    arms = [x.strip() for x in str(text).split(",") if x.strip()]
    bad = [x for x in arms if x not in ARMS]
    if bad or "q0" not in arms or len(arms) < 2 or len(set(arms)) != len(arms):
        raise SystemExit(f"--expected-arms {text!r}: нужен q0 и хотя бы одна "
                         f"рука из {list(ARMS[1:])}, без повторов")
    return [x for x in ARMS if x in arms]


def arm_provenance(paths, arm):
    """Чекпойнт и исключение из фильтра у руки по её артефактам."""
    seen = set()
    for p in paths:
        j = (json.load(open(p)).get("joint") or {})
        seen.add(json.dumps(dict(
            checkpoint_sha1=j.get("checkpoint_sha1"), phase=j.get("phase"),
            layers_per_call=j.get("layers_per_call"),
            admission_override=j.get("admission_override")),
            sort_keys=True, ensure_ascii=False))
    return [json.loads(x) for x in sorted(seen)]


def safety_verdict(technical, deltas, catastrophe=CATASTROPHE):
    """deltas: {рука: разность успеха против q0}."""
    if technical:
        return dict(code=3, outcome=f"технический отказ: {technical}")
    bad = {k: v for k, v in deltas.items() if v < catastrophe}
    if bad:
        return dict(code=4, outcome=(
            "катастрофическое падение: " + ", ".join(
                f"{k} {v:+.3f}" for k, v in sorted(bad.items()))
            + f" ниже {catastrophe}"))
    return dict(code=0, outcome="технический пилот пройден; это НЕ вывод "
                                "об эффективности")


def level_summary(paths, arm):
    """Средний |поправки| по каналам и максимум |a| по блокам руки."""
    acc, n, amax, finite = None, 0, None, True
    for p in paths:
        d = json.load(open(p))
        if str(d.get("arm_label")) != arm:
            continue
        s = d.get("k15d_levels")
        if not s or not s.get("calls"):
            raise SystemExit(f"{p}: у руки {arm} нет k15d_levels")
        last = s["levels"][-1]
        v = np.asarray(s[f"mean_abs_delta_{last}"], np.float64)
        # средняя поправка ПОСЛЕДНЕГО уровня к предыдущему, взвешенная
        # числом вызовов
        acc = v * s["calls"] if acc is None else acc + v * s["calls"]
        n += int(s["calls"])
        m = np.asarray(s[f"absmax_{last}"], np.float64)
        amax = m if amax is None else np.maximum(amax, m)
        finite &= all(bool(s.get(f"finite_{lv}")) for lv in s["levels"])
    if not n:
        return None
    return dict(calls=n, mean_abs_delta_last=(acc / n).tolist(),
                absmax=amax.tolist(), finite=bool(finite))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import k14q_behavior as kb
    import k15c_behavior as kc
    ap = argparse.ArgumentParser(description="K-15d: q0, D0, H1 парно")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=tuple(MODES), default=None)
    ap.add_argument("--arts", nargs="*", default=[])
    ap.add_argument("--expected-arms", default=",".join(ARMS),
                    help="точный состав рук через запятую, например q0,h1")
    ap.add_argument("--tasks", default="",
                    help="для --mode target: номера задач через запятую")
    ap.add_argument("--allow-partial", action="store_true",
                    help="только для смоука: неполный набор — не результат")
    ap.add_argument("--allow-preflight", action="store_true")
    ap.add_argument("--boot", type=int, default=kb.N_BOOT)
    ap.add_argument("--out", default="")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.mode or not a.arts or not a.out:
        raise SystemExit("нужны --mode, --arts и --out")
    if os.path.exists(a.out) and not a.overwrite:
        raise SystemExit(f"{a.out} уже существует")
    if a.allow_preflight and not a.allow_partial:
        raise SystemExit("--allow-preflight только вместе с --allow-partial")
    for p_ in a.arts:
        if ((json.load(open(p_)).get("joint") or {}).get("preflight")
                and not a.allow_preflight):
            raise SystemExit(f"{p_}: артефакт предполётной проверки")
    obs, meta = kb.load(a.arts)
    arms = parse_arms(a.expected_arms)
    tasks = [int(x) for x in a.tasks.split(",") if x.strip()]
    want_cl = expected_clusters(a.mode, tasks)
    technical = []
    # СОСТАВ РУК — ТОЧНО ЗАЯВЛЕННЫЙ, в том числе в смоуке: лишняя или
    # пропавшая рука означает, что раннер и анализ говорят о разном.
    if set(meta) != set(arms):
        raise SystemExit(f"руки в артефактах {sorted(meta)}, заявлены "
                         f"{arms}")
    clusters, seeds = kb.align(obs, arms)
    if not a.allow_partial:
        if clusters != want_cl:
            technical.append(f"кластеров {len(clusters)} из {len(want_cl)}")
        if seeds != SEEDS:
            technical.append(f"сиды {seeds}, зарегистрирован {SEEDS}")
    shas = set()
    for nm in arms:
        shas |= set(meta[nm]["script_shas"])
        if len(meta[nm]["fingerprints"]) != 1:
            technical.append(f"у руки {nm} несколько моделей: "
                             f"{sorted(meta[nm]['fingerprints'])}")
    if len(shas) > 1:
        technical.append(f"артефакты сняты разными версиями k9h: "
                         f"{sorted(shas)}")
    act_max, levels, prov = {}, {}, {}
    for nm in arms:
        if nm != "q0":
            prov[nm] = arm_provenance(meta[nm]["files"], nm)
            if len(prov[nm]) != 1:
                technical.append(f"у руки {nm} разные чекпойнты или "
                                 f"исключения по блокам: {prov[nm]}")
        bad, act_max[nm] = kc.finite_actions(meta[nm]["files"])
        technical += bad
        if nm != "q0":
            levels[nm] = level_summary(meta[nm]["files"], nm)
            if levels[nm] and not levels[nm]["finite"]:
                technical.append(f"{nm}: уровни не конечны")

    sr = {nm: kb.cluster_means(obs, nm, clusters, seeds) for nm in arms}
    pairs = {}
    for b, c in PAIRS:
        if b not in arms or c not in arms:
            continue
        res, hrm = kb.pair_rates(obs, b, c, clusters, seeds)
        d = sr[c] - sr[b]
        ci, _r = kb.boot(dict(delta=d), clusters, n=int(a.boot),
                         qs=(kb.Q_LO, kb.Q_HI))
        per_task = {int(t): dict(
            clusters=int(len(idx)), sr_base=float(sr[b][idx].mean()),
            sr_cand=float(sr[c][idx].mean()), rescue=float(res[idx].mean()),
            harm=float(hrm[idx].mean()))
            for t, idx in kb.strata(clusters).items()}
        pairs[f"{c}_vs_{b}"] = dict(
            base=b, cand=c, sr_base=kb.point(sr[b], clusters),
            sr_cand=kb.point(sr[c], clusters), delta=kb.point(d, clusters),
            ci90_delta=ci["delta"], rescue=kb.point(res, clusters),
            harm=kb.point(hrm, clusters), per_task=per_task)
    if a.mode in ("safety", "target"):
        verdict = safety_verdict(technical, {
            c: v["delta"] for k, v in pairs.items()
            for c in [v["cand"]] if v["base"] == "q0"})
    else:
        verdict = (dict(code=3, outcome=f"технический отказ: {technical}")
                   if technical else
                   dict(code=0, outcome="набор исправен; порога успеха в "
                                        "плане нет — решение по величинам"))
    print(f"  кластеров {len(clusters)}, сиды {seeds}, руки {arms}, версия "
          f"k9h {sorted(shas)}")
    for k, v in pairs.items():
        print(f"  {v['cand']} против {v['base']}: успех {v['sr_base']:.4f} "
              f"-> {v['sr_cand']:.4f}, разность {v['delta']:+.4f}, 90 % "
              f"[{v['ci90_delta'][0]:+.4f}, {v['ci90_delta'][1]:+.4f}]; "
              f"rescue {v['rescue']:.4f}, harm {v['harm']:.4f}")
        print("    по задачам: " + "; ".join(
            f"{t}: {x['sr_base']:.2f}->{x['sr_cand']:.2f}"
            for t, x in v["per_task"].items()))
    for nm, pv in prov.items():
        for x in pv:
            if x.get("admission_override"):
                print(f"  ВНИМАНИЕ: рука {nm} НЕ допущена фильтром, "
                      f"исключение: {x['admission_override']}")
    for nm, s in levels.items():
        if s:
            print(f"  {nm}: вызовов {s['calls']}, средняя |поправка| "
                  f"последнего уровня " + ", ".join(
                      f"{x:.4f}" for x in s["mean_abs_delta_last"])
                  + "; max|a| " + ", ".join(f"{x:.3f}" for x in s["absmax"]))
    print(f"  ИСХОД ({a.mode}): {verdict['outcome']} (код {verdict['code']})")
    out = dict(kind="k15d_behavior", mode=a.mode, verdict=verdict,
               technical=technical, pairs=pairs, levels=levels,
               action_absmax=act_max, clusters=len(clusters), seeds=seeds,
               arms=arms, tasks=tasks or MODES[a.mode]["tasks"],
               provenance=prov, script_sha1=sorted(shas),
               fingerprints={nm: sorted(meta[nm]["fingerprints"])
                             for nm in arms},
               partial=bool(a.allow_partial),
               thresholds=dict(catastrophe=CATASTROPHE,
                               declared="план K-15d §13, до данных"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                exist_ok=True)
    tmp = a.out + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, allow_nan=False)
    os.replace(tmp, a.out)
    print(f"  сводка: {a.out}")
    return int(verdict["code"])


def selftest():
    assert len(expected_clusters("safety")) == 50
    assert len(expected_clusters("dev")) == 250
    assert len(expected_clusters("target", [3])) == 25
    try:
        expected_clusters("target")
        raise AssertionError("target без задач принят")
    except SystemExit:
        pass
    assert parse_arms("q0,h1") == ["q0", "h1"]
    assert parse_arms("h1,q0,d0,h18") == ["q0", "d0", "h18", "h1"]
    for bad in ("h1", "q0", "q0,x", "q0,h1,h1"):
        try:
            parse_arms(bad)
            raise AssertionError(f"принят состав {bad!r}")
        except SystemExit:
            pass
    assert safety_verdict(["x"], {"d0": 0.0})["code"] == 3
    assert safety_verdict([], {"d0": 0.0, "h1": -0.2})["code"] == 4
    assert safety_verdict([], {"d0": -0.15, "h1": 0.1})["code"] == 0
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for i, (arm, calls) in enumerate((("d0", 2), ("d0", 6), ("q0", 1))):
            json.dump(dict(arm_label=arm, k15d_levels=dict(
                calls=calls, levels=["d"], mean_abs_delta_d=[0.1 * (i + 1)] * 7,
                absmax_d=[0.5 + i] * 7, finite_d=True)),
                open(os.path.join(td, f"{i}.json"), "w"))
        paths = [os.path.join(td, f"{i}.json") for i in range(3)]
        s = level_summary(paths, "d0")
        assert s["calls"] == 8
        assert abs(s["mean_abs_delta_last"][0] - (0.1 * 2 + 0.2 * 6) / 8) \
            < 1e-12
        assert s["absmax"][0] == 1.5 and s["finite"]
        assert level_summary(paths, "h1") is None
    print("самопроверка k15d_behavior пройдена")
    return 0


if __name__ == "__main__":
    sys.exit(main())
