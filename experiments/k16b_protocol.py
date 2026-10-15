#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16b: план и проверки (блоки 30 и 35 всех задач, q0 и две головы).

ПЛАН: метки q0, ctx_event, ctx_goal_event; задачи 0–9; блоки 30 и 35
(старты, другие, чем у пилота K-16, но уже использованные в проекте;
резерв 45–49 закрыт); контракт раскатки и руки; чекпойнт K-16b и отчёт
обучения; эталоны — q0ref K-15h тех же блоков (JSON и npz).

ПРОВЕРКА БЛОКА: строгая validate_block и контракт; проверки журнала K-16
(события по журналу; импульс ровно в первом событии; q0 — без импульса;
вне импульса = a0; схват = a0; δ журнала; предел); исполненные actions =
журнал; q0 — против q0ref K-15h (хеши действий, исходы, done_step);
головы — против q0 этого блока (старты, префикс, a0 и состояние до
импульса, вызов импульса = первое событие q0).
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

KIND = "k16b_plan"
PLAN = "reports/k16b/plan.json"
BLOCKS = (30, 35)
SMOKE = [(0, 30), (8, 30)]
MAP = {"q0": "q0", "ctx_event": "goal_band_hist_event",
       "ctx_goal_event": "goal_band_hist_event"}
CODE = ("k16b_policy.py", "k16b_harness.py", "k16b_protocol.py",
        "k16b_train.py", "k16b_analyze.py", "k16b_rollout.sh",
        "k16_policy.py", "k16_harness.py", "k16_protocol.py",
        "k16_models.py", "k9h_multiarm_gate.py", "k15f_policy.py",
        "k15f_continuous_refine.py", "k15f_check_identity.py",
        "k15g_local_probe.py", "k15h_protocol.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]


def make_plan(*, models, train_report, basis, identity, device):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import k16b_policy as bp
    import torch
    ck = torch.load(models, map_location="cpu", weights_only=False)
    rep = json.load(open(train_report))
    if rep["models"]["sha1"] != hashlib.sha1(open(models, "rb").read()
                                             ).hexdigest():
        raise SystemExit("отчёт обучения не от этого чекпойнта")
    c = sorted(glob.glob("reports/k15h/run/p*/"))
    if len(c) != 1:
        raise SystemExit(f"каталог K-15h неоднозначен: {c}")
    refs = {}
    for t in range(10):
        for b in BLOCKS:
            v = os.path.join(c[0], f"q0ref_t{t}_i{b}.json")
            if not os.path.exists(v):
                raise SystemExit(f"нет эталона {v}")
            refs[f"q0_t{t}_i{b}"] = v
            refs[f"q0_t{t}_i{b}__npz"] = os.path.join(
                c[0], json.load(open(v))["actions_npz"])
    contract = hpc.expected_contract(basis, identity)
    contract.pop("k15h_policy_sha1", None)
    contract.update(
        harness_adapter_sha1=sha(os.path.join(HERE, "k16b_harness.py")),
        k16b_policy_sha1=sha(os.path.join(HERE, "k16b_policy.py")),
        k16_models_sha1=sha(os.path.join(HERE, "k16_models.py")),
        models_sha1=sha(models))
    inputs = dict(models=models, train_report=train_report, basis=basis,
                  identity_report=identity,
                  labels=ck["provenance"]["labels"],
                  k16_models=ck["provenance"]["k16_models"],
                  harness_utils=hpc.HARNESS_UTILS)
    return dict(
        kind=KIND, created=datetime.datetime.now().isoformat(
            timespec="seconds"),
        labels=list(bp.LABELS), tasks=list(range(10)), blocks=list(BLOCKS),
        n_envs=5, smoke=SMOKE, device=str(device),
        setup=pr.expected_setup(), expected_contract=contract,
        model_rules=dict(alpha=ck["alpha"], clip_env_norm=ck["clip_env_norm"],
                         units=ck.get("units")),
        inputs={k: os.path.abspath(v) for k, v in inputs.items()},
        inputs_sha1={k: sha(v) for k, v in inputs.items()},
        refs={k: os.path.abspath(v) for k, v in refs.items()},
        refs_sha1={k: sha(v) for k, v in refs.items()},
        code={f: sha(os.path.join(HERE, f)) for f in CODE},
        note=("вопрос: помогает ли явное представление цели; цель "
              "вычисляется из того же скрытого вектора, что в контексте"))


def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план K-16b"]
    for grp in ("inputs", "refs"):
        for k, f in plan[grp].items():
            if not os.path.exists(f) or sha(f) != plan[f"{grp}_sha1"][k]:
                p.append(f"{grp} {k} изменён или отсутствует ({f})")
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    return p


def check_block(plan, plan_path, path, label, task, block, ref_q0=None):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import k16_protocol as p16
    p = pr.validate_block(path, label, int(task), int(block),
                          require_single=True, setup=plan["setup"])
    if p:
        return p
    d, z = p16.load(path)
    j = d.get("joint") or {}
    for k, v in plan["expected_contract"].items():
        if json.dumps(j.get(k), sort_keys=True) != json.dumps(
                v, sort_keys=True):
            p.append(f"{label}: контракт {k} = {j.get(k)!r}, план {v!r}")
    if j.get("label") != label or j.get("plan_sha1") != sha(plan_path):
        p.append(f"{label}: метка или план руки не из плана")
    m = MAP[label]
    z2 = dict(z)
    z2[f"k15d_level_{m}"] = z[f"k15d_level_{label}"]
    p += p16.check_arrays(m, z2, plan, [99] * plan["n_envs"])
    if not p:
        p += hpc.actions_problems(label, z, hpc.harness_scale())
    if not p:
        if label == "q0":
            p += p16.check_q0_vs_k15h(d, plan["refs"][f"q0_t{task}_i{block}"])
        else:
            if not ref_q0 or not os.path.exists(ref_q0):
                return p + [f"{label}: нет блока q0"]
            rd, rz = p16.load(ref_q0)
            p += p16.check_vs_q0(m, d, z2, rd, rz)
    return p


def check_run(plan, plan_path, run_dir, pairs=None):
    ok, gaps = {}, []
    pairs = pairs or [(t, b) for t in plan["tasks"] for b in plan["blocks"]]
    for t, b in pairs:
        ref = os.path.join(run_dir, f"q0_t{t}_i{b}.json")
        for lab in plan["labels"]:
            pth = os.path.join(run_dir, f"{lab}_t{t}_i{b}.json")
            if not os.path.exists(pth):
                gaps.append(dict(label=lab, task=t, block=b, why="не готов"))
                continue
            if lab != "q0" and ("q0", t, b) not in ok:
                gaps.append(dict(label=lab, task=t, block=b,
                                 why="нет проверенного q0"))
                continue
            prob = check_block(plan, plan_path, pth, lab, t, b, ref)
            if prob:
                gaps.append(dict(label=lab, task=t, block=b,
                                 why="; ".join(prob[:2])))
            else:
                ok[(lab, t, b)] = pth
    return ok, gaps


def selftest():
    assert set(MAP) == {"q0", "ctx_event", "ctx_goal_event"}
    assert MAP["ctx_event"] == MAP["ctx_goal_event"] == \
        "goal_band_hist_event"          # событийное расписание K-16
    import k16_policy as kp
    for v in MAP.values():
        assert v in kp.POLICIES
    print("самопроверка k16b_protocol пройдена: соответствие меток "
          "расписаниям K-16")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-16b: протокол")
    ap.add_argument("mode", choices=("plan", "check-plan", "check-block",
                                     "smoke-check", "selftest"))
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--models", default="data/k16b/models.pt")
    ap.add_argument("--train-report", default="reports/k16b/train.json")
    ap.add_argument("--basis", default="data/k15f/basis_s0.pt")
    ap.add_argument("--identity", default="reports/k15f/identity_cuda1.json")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--block", nargs=4, default=None,
                    metavar=("JSON", "МЕТКА", "ЗАДАЧА", "БЛОК"))
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    try:
        if a.mode == "plan":
            if os.path.exists(a.plan):
                raise SystemExit(f"{a.plan} уже зафиксирован")
            plan = make_plan(models=a.models, train_report=a.train_report,
                             basis=a.basis, identity=a.identity,
                             device=a.device)
            os.makedirs(os.path.dirname(a.plan), exist_ok=True)
            with open(a.plan + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=1)
            os.replace(a.plan + ".tmp", a.plan)
            print(f"  план {a.plan}: политик {len(plan['labels'])}, задач "
                  f"10, блоки {plan['blocks']}, процессов "
                  f"{len(plan['labels']) * 10 * len(plan['blocks'])}")
            return 0
        plan = json.load(open(a.plan))
        if a.mode == "check-plan":
            prob = check_plan_inputs(plan)
            print("\n".join(prob) if prob else "  входы плана сверены")
            return 3 if prob else 0
        if a.mode == "check-block":
            path, lab, t, b = a.block
            ref = os.path.join(os.path.dirname(path), f"q0_t{t}_i{b}.json")
            prob = check_block(plan, a.plan, path, lab, t, b, ref)
            if prob:
                print("; ".join(prob[:6]))
            return 2 if prob else 0
        ok, gaps = check_run(plan, a.plan, a.run_dir,
                             [tuple(x) for x in plan["smoke"]])
        n = len(plan["smoke"]) * len(plan["labels"])
        print(f"  smoke: проверено блоков {len(ok)} из {n}; проблем "
              f"{len(gaps)}" + (f": {gaps[:4]}" if gaps else ""))
        return 3 if gaps else 0
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
