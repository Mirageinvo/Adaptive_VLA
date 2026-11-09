#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-33: шесть политик без переобучения — существующие загрузчики и пути.

  T0, T1 — parents_attn24, training seed 0 / 1 (k26_policy.build_arm);
  F0, F1 — flat_attn24,    training seed 0 / 1 (k26_policy.build_arm);
  B      — fullbar: штатный generate и ActionCodec самого K-9h
           (--policy fullbar, руки-подмены нет);
  BC     — fullbar_cached: k21_cached_policy.build_arm (требует пройденного
           допуска k21_barcache для той же ревизии, подготовки и кода).

Спецификации T/F берутся из закреплённых планов K-26
(reports/k26/plan_s<S>_<mode>.json): путь и SHA best.pt, отпечаток весов,
шаг best, отчёт обучения. Здесь сверяются SHA плана-источника, SHA
чекпойнта = спецификации = best_sha1 отчёта обучения, шаг best = отчёту,
SHA отчёта, код K-26 = закреплённому в плане; загрузчик k26_policy при
построении руки сам сверяет режим, архитектуру, router, seed, prep_id,
код обучения и отпечаток весов. Веса, точность, процессор, нормировки и
декодер не меняются. Политике не передаются ни scene seed, ни case_id.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ARMS = {
    "T0": dict(kind="tree", mode="parents_attn24", seed=0,
               label="parents_attn24_s0", policy="k15f"),
    "F0": dict(kind="flat", mode="flat_attn24", seed=0,
               label="flat_attn24_s0", policy="k15f"),
    "T1": dict(kind="tree", mode="parents_attn24", seed=1,
               label="parents_attn24_s1", policy="k15f"),
    "F1": dict(kind="flat", mode="flat_attn24", seed=1,
               label="flat_attn24_s1", policy="k15f"),
    "B": dict(kind="bar", label="fullbar", policy="fullbar"),
    "BC": dict(kind="cached", label="fullbar_cached", policy="k15f"),
}
ORDER0 = ("T0", "F0", "BC", "T1", "F1", "B")
BASIS = "data/k15f/basis_s0.pt"
K26_PLAN = "reports/k26/plan_s{seed}_{mode}.json"


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def k26_plan_path(arm):
    a = ARMS[arm]
    return K26_PLAN.format(seed=a["seed"], mode=a["mode"])


def k9h_args(arm):
    """Аргументы K-9h руки (без общих настроек раскатки)."""
    a = ARMS[arm]
    out = ["--policy", a["policy"], "--arm-label", a["label"]]
    if a["policy"] == "k15f":
        out += ["--k15f-basis", BASIS]
    return out


def source_problems(arm, src):
    """Источник руки по записи плана K-33 (src) на этой машине."""
    import k21_protocol as pp
    a = ARMS[arm]
    p = []
    if a["kind"] in ("tree", "flat"):
        pl = src["k26_plan"]
        if not os.path.exists(pl) or sha(pl) != src["k26_plan_sha1"]:
            return [f"{arm}: план K-26 {pl} изменён или отсутствует"]
        plan = json.load(open(pl, encoding="utf-8"))
        spec = plan["specs"].get(a["label"])
        if spec is None or plan.get("train_seed") != a["seed"] or \
                plan.get("mode") != a["mode"]:
            return [f"{arm}: в плане K-26 нет {a['label']}"]
        if not os.path.exists(spec["ckpt"]) or \
                pp.full_sha(spec["ckpt"]) != spec["ckpt_sha1"]:
            p.append(f"{arm}: best.pt изменён или отсутствует")
        tr = spec["train_report"]
        if not os.path.exists(tr) or pp.full_sha(tr) != \
                spec["train_report_sha1"]:
            p.append(f"{arm}: отчёт обучения K-26 изменён или отсутствует")
        else:
            r = json.load(open(tr, encoding="utf-8"))
            if r.get("best_sha1") != spec["ckpt_sha1"] or \
                    r.get("best_weights_hash") != spec["weights_hash"]:
                p.append(f"{arm}: best.pt ≠ отчёту обучения")
            bu = r.get("best_update", spec["best_update"])
            if int(bu) != int(spec["best_update"]):
                p.append(f"{arm}: шаг best {spec['best_update']} ≠ отчёту {bu}")
        for f, s_ in plan["code"].items():
            q = os.path.join(HERE, f)
            if not os.path.exists(q) or sha(q) != s_:
                p.append(f"{arm}: код K-26 {f} изменён")
        for k in ("ckpt_sha1", "weights_hash", "best_update", "arch", "seed",
                  "prep_id"):
            if spec.get(k) != src["spec"].get(k):
                p.append(f"{arm}: спецификация {k} ≠ плану K-33")
    elif a["kind"] == "cached":
        import k21_barcache as bc
        import k21_data as d21
        prep = json.load(open(d21.REPORT, encoding="utf-8"))
        try:
            adm = bc.require_admitted(dict(prep_report=prep))
        except SystemExit as e:
            return [f"{arm}: допуск fullbar_cached: {e.code}"]
        for k in ("barcache_sha1", "equivalence"):
            if adm.get(k) != src["admission"].get(k):
                p.append(f"{arm}: допуск {k} ≠ плану K-33")
    return p


def make_source(arm):
    """Запись источника руки для плана K-33 (контентные отпечатки)."""
    import k21_protocol as pp
    a = ARMS[arm]
    if a["kind"] in ("tree", "flat"):
        pl = k26_plan_path(arm)
        plan = json.load(open(pl, encoding="utf-8"))
        spec = plan["specs"][a["label"]]
        r = json.load(open(spec["train_report"], encoding="utf-8"))
        src = dict(k26_plan=os.path.abspath(pl), k26_plan_sha1=sha(pl),
                   spec={k: spec[k] for k in (
                       "arch", "seed", "ckpt", "ckpt_sha1", "weights_hash",
                       "best_update", "val_L", "prep_id", "train_report",
                       "train_report_sha1", "router")},
                   train_best_sha1=r.get("best_sha1"),
                   k26_code=plan["code"])
        src["content"] = dict(ckpt_sha1=spec["ckpt_sha1"],
                              weights_hash=spec["weights_hash"],
                              best_update=spec["best_update"],
                              prep_id=spec["prep_id"],
                              k26_code=plan["code"])
    elif a["kind"] == "cached":
        import k21_barcache as bc
        import k21_data as d21
        prep = json.load(open(d21.REPORT, encoding="utf-8"))
        adm = bc.require_admitted(dict(prep_report=prep))
        src = dict(admission={k: adm.get(k) for k in (
            "barcache_sha1", "equivalence", "equivalence_confirmed",
            "kind", "ok")}, admit_report=os.path.abspath(bc.ADMIT_REPORT),
            policy_sha1=pp.full_sha(os.path.join(HERE,
                                                 "k21_cached_policy.py")))
        src["content"] = dict(admission=src["admission"],
                              policy_sha1=src["policy_sha1"])
    else:
        src = dict(content=dict(native="k9h fullbar"))
    if src.get("content") is not None:
        bad = source_problems(arm, src)
        if bad:
            raise SystemExit("; ".join(bad))
    return src


def install_arm(arm, src, plan_sha1, debug):
    """Подмена k15f_policy.build_arm для T/F/BC; B — штатный путь K-9h."""
    a = ARMS[arm]
    if a["policy"] != "k15f":
        return
    import k15f_policy
    adapter = sha(os.path.join(HERE, "k33_harness.py"))[:12]
    if a["kind"] in ("tree", "flat"):
        import k26_policy as kp
        spec = json.load(open(src["k26_plan"], encoding="utf-8"))[
            "specs"][a["label"]]
        k26_sha = src["k26_plan_sha1"]

        def build(device, basis, label, torch, **kw):
            if label != a["label"]:
                raise SystemExit(f"K-33 {arm}: метка {label}")
            # plan_sha1 руки — SHA плана K-26 (источник спецификации), как в
            # действующем холодном пути; K-33 добавляет свои поля
            arm_ = kp.build_arm(device, basis, label, torch, spec=spec,
                                plan_sha1=k26_sha, debug=debug)
            arm_.meta.update(harness_adapter_sha1=adapter, k33_arm=arm,
                             k33_plan_sha1=plan_sha1)
            return arm_
    else:
        import k21_cached_policy as kpc

        def build(device, basis, label, torch, **kw):
            if label != a["label"]:
                raise SystemExit(f"K-33 {arm}: метка {label}")
            arm_ = kpc.build_arm(device, basis, label, torch,
                                 plan_sha1=plan_sha1)
            arm_.meta.update(harness_adapter_sha1=adapter, k33_arm=arm,
                             k33_plan_sha1=plan_sha1)
            return arm_
    k15f_policy.build_arm = build


def joint_problems(arm, d, src, plan_sha1, debug):
    """Метаданные руки в JSON блока K-9h."""
    a = ARMS[arm]
    j = d.get("joint") or {}
    p = []
    if a["kind"] == "bar":
        return p
    want = dict(label=a["label"], k33_arm=arm, k33_plan_sha1=plan_sha1,
                harness_adapter_sha1=sha(os.path.join(
                    HERE, "k33_harness.py"))[:12])
    if a["kind"] in ("tree", "flat"):
        want.update(arch=a["mode"], train_seed=a["seed"],
                    plan_sha1=src["k26_plan_sha1"],
                    ckpt_sha1=src["spec"]["ckpt_sha1"],
                    weights_hash=src["spec"]["weights_hash"],
                    router_debug=bool(debug))
    else:
        want.update(arm="fullbar_cached", plan_sha1=plan_sha1,
                    barcache_sha1=src["admission"]["barcache_sha1"],
                    equivalence=src["admission"]["equivalence"])
    for k, v in want.items():
        if j.get(k) != v:
            p.append(f"{arm}: {k} = {j.get(k)!r}, ожидалось {v!r}")
    return p


def journal_problems(arm, z, bounds, debug):
    """Журнал вызовов руки: формы, конечность, router, исполненные действия
    = журнал (масштаб и знак схвата харнесса)."""
    import k15h_protocol as hpc
    import k21_protocol as pp
    a = ARMS[arm]
    if a["kind"] == "bar":
        return []
    lab = a["label"]
    if a["kind"] in ("tree", "flat"):
        import k26_protocol as pc
        p = pp.journal_problems(lab, z, bounds)
        if not p:
            p += pc.attn_problems(z, lab, a["mode"], debug=debug)
    else:
        import k21_cached_protocol as cp
        p = cp.journal_problems(lab, z)
    if not p:
        p += hpc.actions_problems(lab, z, hpc.harness_scale())
    return p


def selftest():
    assert set(ORDER0) == set(ARMS) and len(ORDER0) == 6
    assert k26_plan_path("T1") == "reports/k26/plan_s1_parents_attn24.json"
    assert k26_plan_path("F0") == "reports/k26/plan_s0_flat_attn24.json"
    assert k9h_args("B") == ["--policy", "fullbar", "--arm-label", "fullbar"]
    assert k9h_args("BC")[:4] == ["--policy", "k15f", "--arm-label",
                                  "fullbar_cached"]
    assert "--k15f-basis" in k9h_args("T0")
    assert {ARMS[a]["label"] for a in ARMS} == {
        "parents_attn24_s0", "parents_attn24_s1", "flat_attn24_s0",
        "flat_attn24_s1", "fullbar", "fullbar_cached"}
    d = dict(joint=dict(label="x"))
    assert joint_problems("B", d, {}, "p", False) == []
    assert joint_problems("BC", d, dict(admission=dict(
        barcache_sha1="a", equivalence="e")), "p", False)
    print("самопроверка k33_policy пройдена: шесть рук (T0/F0/T1/F1 из "
          "планов K-26, B штатный, BC cached), аргументы K-9h, метаданные "
          "руки в блоке")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; использование — через k33_harness/k33_protocol")
