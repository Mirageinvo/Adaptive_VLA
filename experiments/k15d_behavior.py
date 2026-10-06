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
             confirm=dict(tasks=list(range(8)), states=list(range(25))),
             final=dict(tasks=list(range(10)), states=list(range(25, 50))),
             target=dict(tasks=None, states=list(range(25))))
# ПОДТВЕРЖДАЮЩЕЕ ПРАВИЛО, ЗАФИКСИРОВАННОЕ ДО РОЛЛАУТОВ (07.10.2026).
# Гипотеза «h18 лучше q0» родилась на safety (задачи 8-9, состояния 0-24),
# поэтому подтверждается на НЕ ВИДЕННЫХ кластерах: задачи 0-7, состояния
# 0-24, 200 кластеров. Единственная первичная пара — h18 против q0; равный
# вес восьми задач, стратифицированный по задачам кластерный бутстрап
# K-14q; успех — нижняя граница двустороннего 90 % интервала разности
# строго выше нуля. Всё остальное (другие руки и пары, задачи 8-9) —
# описательно. Менять правило после начала роллаутов confirm нельзя.
CONFIRM = dict(base="q0", cand="h18", tasks=list(range(8)),
               states=list(range(25)), interval="two-sided 90%",
               rule="ci90_lo > 0", registered="07.10.2026, до роллаутов",
               # параметры бутстрапа K-14q, сверяются с ним при анализе
               n_boot=10000, boot_seed=53, quantiles=[5, 95])
# ФИНАЛЬНАЯ ПРОВЕРКА, ЗАФИКСИРОВАННАЯ ДО РОЛЛАУТОВ (07.10.2026). Открывается
# только после пройденного confirm, на замороженном h18 (чекпойнт
# h1p1_s0.pt) и финальном банке K-14q: задачи 0-9, состояния 25-49, 250
# кластеров. Пара, вес задач, бутстрап и правило — те же, что в confirm.
FINAL = dict(CONFIRM, tasks=list(range(10)), states=list(range(25, 50)),
             requires="пройденный confirm с тем же исполнением h18",
             registered="07.10.2026, до роллаутов")
RULES = dict(confirm=CONFIRM, final=FINAL)


def check_confirm_args(*, allow_partial, boot, arms, tasks, kb_consts,
                       rule=CONFIRM):
    """confirm и final не допускают ни одного ручного отступления."""
    R = rule
    p = []
    if allow_partial:
        p.append("confirm запрещает --allow-partial")
    if int(boot) != R["n_boot"]:
        p.append(f"confirm требует {R['n_boot']} бутстрап-реплик, "
                 f"дано {boot}")
    if list(arms) != [R["base"], R["cand"]]:
        p.append(f"confirm требует ровно {R['base']},"
                 f"{R['cand']}, дано {list(arms)}")
    if tasks:
        p.append("--tasks в confirm запрещён: задачи фиксированы правилом")
    for k, v in (("n_boot", kb_consts.get("n_boot")),
                 ("boot_seed", kb_consts.get("boot_seed")),
                 ("quantiles", kb_consts.get("quantiles"))):
        if v != R[k]:
            p.append(f"бутстрап K-14q изменился: {k} {v!r}, правило "
                     f"{R[k]!r}")
    return p
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


# ЧТО ИМЕННО ИСПОЛНЯЕТ КАЖДАЯ МЕТКА. Проверяется по КАЖДОМУ артефакту
# fail-closed: артефакт h1p2 под меткой h18, h18 на 24 слоях или посторонняя
# политика под меткой q0 — технический отказ, а не тихое сравнение.
ARM_SPEC = {
    "q0":  dict(policy="depthrvq", depth_rvq_mode="fast", levels=1),
    "d0":  dict(policy="k15d", phase="d0", layers_per_call=24, level="d"),
    "h18": dict(policy="k15d", phase="h1p1", layers_per_call=18, level="1"),
    "h1":  dict(policy="k15d", phase="h1p2", layers_per_call=24, level="2"),
}
# Набор задач, в котором определены safety, confirm и dev.
MODE_SUITE = dict(safety="10", dev="10", confirm="10", final="10")


def check_arm_spec(d, arm):
    """Артефакт d действительно исполняет то, что значит метка arm."""
    spec = ARM_SPEC[arm]
    j = d.get("joint") or {}
    argv = d.get("argv") or {}
    got = dict(policy=d.get("policy"))
    if spec["policy"] == "depthrvq":
        got.update(depth_rvq_mode=argv.get("depth_rvq_mode"),
                   levels=d.get("levels"))
    else:
        got.update(phase=j.get("phase"),
                   layers_per_call=j.get("layers_per_call"),
                   level=j.get("level"))
    bad = {k: (got.get(k), v) for k, v in spec.items() if got.get(k) != v}
    return [f"метка {arm}: {k} = {g!r}, ожидалось {w!r}"
            for k, (g, w) in sorted(bad.items())]


def arm_provenance(paths, arm):
    """Чекпойнт, код и исключение из фильтра у руки по её артефактам.

    Сюда же попадают отпечатки модели, модулей уточнения и руки и версия
    харнесса: по ним раннер решает, открывает ли прошлый safety dev.
    """
    seen = set()
    for p in paths:
        d = json.load(open(p))
        j = d.get("joint") or {}
        seen.add(json.dumps(dict(
            checkpoint_sha1=j.get("checkpoint_sha1"), phase=j.get("phase"),
            layers_per_call=j.get("layers_per_call"), level=j.get("level"),
            model_fingerprint=j.get("model_fingerprint"),
            refine_module_sha1=j.get("refine_module_sha1"),
            policy_module_sha1=j.get("policy_module_sha1"),
            harness_sha1=d.get("script_sha1"),
            admission_override=j.get("admission_override")),
            sort_keys=True, ensure_ascii=False))
    return [json.loads(x) for x in sorted(seen)]


def check_actions_npz(paths):
    """Отпечаток npz действий, записанный в JSON, пересчитывается."""
    import hashlib
    bad = []
    for p in paths:
        d = json.load(open(p))
        npz, want = d.get("actions_npz"), d.get("actions_npz_sha1")
        if not npz or not want:
            bad.append(f"{os.path.basename(p)}: нет npz действий или его "
                       f"отпечатка")
            continue
        f = os.path.join(os.path.dirname(p), npz)
        if not os.path.exists(f):
            bad.append(f"{os.path.basename(p)}: нет {npz}")
            continue
        h = hashlib.sha1()
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
        if h.hexdigest()[:12] != want:
            bad.append(f"{os.path.basename(p)}: отпечаток {npz} не совпал с "
                       f"записанным")
    return bad


def confirm_verdict(technical, lo, hi, point_delta):
    """Первичная пара h18 против q0 на задачах 0-7."""
    if technical:
        return dict(code=3, outcome=f"технический отказ: {technical}")
    ok = lo > 0.0
    return dict(code=0 if ok else 4, outcome=(
        f"h18 против q0: разность {point_delta:+.4f}, 90 % [{lo:+.4f}, "
        f"{hi:+.4f}] — {'ПОДТВЕРЖДЕНО' if ok else 'НЕ подтверждено'} "
        f"(правило: нижняя граница > 0)"))


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
    if a.mode in RULES:
        problems = check_confirm_args(rule=RULES[a.mode],
            allow_partial=a.allow_partial, boot=a.boot, arms=arms,
            tasks=tasks, kb_consts=dict(n_boot=kb.N_BOOT,
                                        boot_seed=kb.BOOT_SEED,
                                        quantiles=[kb.Q_LO, kb.Q_HI]))
        if problems:
            raise SystemExit("; ".join(problems))
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
    # ЧТО ИСПОЛНЯЛА КАЖДАЯ МЕТКА, НАБОР ЗАДАЧ И ОТПЕЧАТКИ npz — по КАЖДОМУ
    # артефакту.
    want_suite = MODE_SUITE.get(a.mode)
    for nm in arms:
        for p_ in meta[nm]["files"]:
            d_ = json.load(open(p_))
            technical += [f"{os.path.basename(p_)}: {x}"
                          for x in check_arm_spec(d_, nm)]
            if want_suite is not None and str(d_.get("suite")) != want_suite:
                technical.append(f"{os.path.basename(p_)}: набор "
                                 f"{d_.get('suite')!r}, режим {a.mode} "
                                 f"определён на {want_suite}")
        technical += check_actions_npz(meta[nm]["files"])
    if a.mode in RULES and not {CONFIRM["base"], CONFIRM["cand"]} \
            <= set(arms):
        technical.append(f"в confirm нет первичной пары {CONFIRM['base']}, "
                         f"{CONFIRM['cand']}")
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
    if a.mode in RULES:
        prim = pairs.get(f"{CONFIRM['cand']}_vs_{CONFIRM['base']}")
        if prim is None:
            verdict = dict(code=3, outcome="нет первичной пары")
        else:
            verdict = confirm_verdict(technical, prim["ci90_delta"][0],
                                      prim["ci90_delta"][1], prim["delta"])
    elif a.mode in ("safety", "target"):
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
                  f"последнего уровня по выданным чанкам " + ", ".join(
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
                               declared="план K-15d §13, до данных"),
               confirm_rule=RULES.get(a.mode))
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
    assert len(expected_clusters("confirm")) == 200
    assert len(expected_clusters("final")) == 250
    assert set(expected_clusters("final")).isdisjoint(
        expected_clusters("dev"))
    assert FINAL["base"] == "q0" and FINAL["cand"] == "h18"
    kbc = dict(n_boot=10000, boot_seed=53, quantiles=[5, 95])
    assert check_confirm_args(allow_partial=False, boot=10000,
                              arms=["q0", "h18"], tasks=[], kb_consts=kbc,
                              rule=FINAL) == []
    assert check_confirm_args(allow_partial=True, boot=10000,
                              arms=["q0", "h18"], tasks=[], kb_consts=kbc,
                              rule=FINAL)
    ok_args = dict(allow_partial=False, boot=10000, arms=["q0", "h18"],
                   tasks=[], kb_consts=kbc)
    assert check_confirm_args(**ok_args) == []
    for mut in (dict(allow_partial=True), dict(boot=100),
                dict(arms=["q0", "h18", "h1"]), dict(arms=["q0", "d0"]),
                dict(tasks=[0, 1]),
                dict(kb_consts=dict(kbc, boot_seed=1)),
                dict(kb_consts=dict(kbc, quantiles=[10, 90]))):
        assert check_confirm_args(**dict(ok_args, **mut)), mut
    assert confirm_verdict([], 0.01, 0.2, 0.1)["code"] == 0
    assert confirm_verdict([], 0.0, 0.2, 0.1)["code"] == 4     # строго > 0
    assert confirm_verdict(["x"], 0.1, 0.2, 0.1)["code"] == 3
    # метка обязана исполнять своё
    k = lambda ph, ly, lv: dict(policy="k15d", joint=dict(
        phase=ph, layers_per_call=ly, level=lv))
    assert check_arm_spec(k("h1p1", 18, "1"), "h18") == []
    assert check_arm_spec(k("h1p2", 24, "2"), "h18")          # h1 под h18
    assert check_arm_spec(k("h1p1", 24, "1"), "h18")          # не 18 слоёв
    assert check_arm_spec(k("d0", 24, "d"), "d0") == []
    assert check_arm_spec(k("h1p2", 24, "2"), "h1") == []
    q0a = dict(policy="depthrvq", levels=1, argv=dict(depth_rvq_mode="fast"))
    assert check_arm_spec(q0a, "q0") == []
    assert check_arm_spec(dict(q0a, policy="fullbar"), "q0")
    assert check_arm_spec(dict(q0a, argv=dict(depth_rvq_mode="full")), "q0")
    assert check_arm_spec(k("d0", 24, "d"), "q0")
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
        # отпечаток npz пересчитывается
        np.savez(os.path.join(td, "a.actions.npz"), actions=np.zeros(3))
        import hashlib
        sha = hashlib.sha1(open(os.path.join(td, "a.actions.npz"),
                                "rb").read()).hexdigest()[:12]
        art = os.path.join(td, "a.json")
        json.dump(dict(actions_npz="a.actions.npz", actions_npz_sha1=sha),
                  open(art, "w"))
        assert check_actions_npz([art]) == []
        json.dump(dict(actions_npz="a.actions.npz",
                       actions_npz_sha1="000000000000"), open(art, "w"))
        assert check_actions_npz([art])
    print("самопроверка k15d_behavior пройдена")
    return 0


if __name__ == "__main__":
    sys.exit(main())
