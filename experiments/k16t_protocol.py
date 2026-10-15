#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16t: подтверждение K-16s на задаче 8 (блоки 30 и 35).

ЭТАПЫ (раннер k16t_rollout.sh):
  plan       план: задача 8, блоки 30 и 35; learned = неизменённая
             goal_band_hist_event K-16 (тот же чекпойнт); повороты
             stored_rot_s1..s3 (те же сиды и SO(3), что K-16r/K-16s); q0 —
             проверенные блоки K-16b тех же стартов (JSON и npz закреплены);
  learned    2 процесса;
  reference  таблица эталонов из ПРОВЕРЕННЫХ learned-блоков (вызов
             импульса, a0, действие learned), SHA блоков закреплены;
  rot        6 процессов;
  analyze    основной результат — по 10 новым состояниям; старые 5
             (K-16s, блок 25) — отдельно.

ПРОВЕРКА learned: строгая validate_block и контракт; журнал K-16 (событие
по журналу, один импульс в первом событии, вне импульса = a0, схват,
δ, предел); исполненные actions = журнал; против q0 K-16b (старты,
префикс и входы до импульса, вызов = первое событие q0).
ПРОВЕРКА поворота: то же для событийного импульса + эталон (вызов, a0
побитово, исходная δ, применённая = повёрнутая, матрицы по сиду).
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

KIND = "k16t_plan"
PLAN = "reports/k16t/plan.json"
REF = "reports/k16t/reference.npz"
TASK = 8
BLOCKS = (30, 35)
LEARNED = "goal_band_hist_event"
ROT = ("stored_rot_s1", "stored_rot_s2", "stored_rot_s3")
PLAN16 = "reports/k16/plan.json"
PLAN16B = "reports/k16b/plan.json"
CODE = ("k16t_protocol.py", "k16t_harness.py", "k16t_rollout.sh",
        "k16s_policy.py", "k16s_protocol.py", "k16r_policy.py",
        "k16r_analyze.py", "k16_policy.py", "k16_harness.py",
        "k16_protocol.py", "k16_models.py", "k16b_protocol.py",
        "k9h_multiarm_gate.py", "k15g_local_probe.py", "k15h_protocol.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]


def run_dir_of(plan_path, sub):
    return os.path.join(f"reports/{sub}/run", f"p{sha(plan_path)}")


def make_plan(device):
    import k15g_local_probe as pr
    import k16_protocol as p16
    import k16b_protocol as b16
    import k16s_policy as sp
    plan16 = json.load(open(PLAN16))
    plan16b = json.load(open(PLAN16B))
    if p16.check_plan_inputs(plan16) or b16.check_plan_inputs(plan16b):
        raise SystemExit("план K-16 или K-16b не проходит проверку входов")
    run16b = run_dir_of(PLAN16B, "k16b")
    refs = {}
    for b in BLOCKS:
        v = os.path.join(run16b, f"q0_t{TASK}_i{b}.json")
        prob = b16.check_block(plan16b, PLAN16B, v, "q0", TASK, b)
        prob += pr.validate_block(v, "q0", TASK, b, require_single=True,
                                  setup=plan16b["setup"])
        if prob:
            raise SystemExit(f"q0 K-16b {v} не прошёл проверку: {prob[:2]}")
        refs[f"q0_i{b}"] = v
        refs[f"q0_i{b}__npz"] = os.path.join(run16b, json.load(open(v))[
            "actions_npz"])
    models = plan16["inputs"]["models"]
    base = p16.expected_contract(plan16["inputs"]["basis"],
                                 plan16["inputs"]["identity_report"], models)
    c_learned = dict(base, harness_adapter_sha1=sha(os.path.join(
        HERE, "k16t_harness.py")))
    c_rot = dict(c_learned, k16s_policy_sha1=sha(os.path.join(
        HERE, "k16s_policy.py")))
    inputs = dict(plan16=PLAN16, plan16b=PLAN16B, models=models,
                  basis=plan16["inputs"]["basis"],
                  identity_report=plan16["inputs"]["identity_report"])
    return dict(
        kind=KIND, created=datetime.datetime.now().isoformat(
            timespec="seconds"),
        task=TASK, blocks=list(BLOCKS), n_envs=5, learned=LEARNED,
        rot=list(ROT), seeds={k: sp.SEEDS[k] for k in ROT},
        device=str(device), setup=plan16["setup"],
        model_rules=plan16["model_rules"],
        expected_contract=dict(learned=c_learned, rot=c_rot),
        inputs={k: os.path.abspath(v) for k, v in inputs.items()},
        inputs_sha1={k: sha(v) for k, v in inputs.items()},
        refs={k: os.path.abspath(v) for k, v in refs.items()},
        refs_sha1={k: sha(v) for k, v in refs.items()},
        code={f: sha(os.path.join(HERE, f)) for f in CODE},
        note=("подтверждение K-16s на задаче 8: основной результат — 10 "
              "новых состояний; блок 40 не добавляется"))


def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план K-16t"]
    for grp in ("inputs", "refs"):
        for k, f in plan[grp].items():
            if not os.path.exists(f) or sha(f) != plan[f"{grp}_sha1"][k]:
                p.append(f"{grp} {k} изменён или отсутствует ({f})")
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    return p


def common_checks(plan, plan_path, path, label, block, contract):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import k16_protocol as p16
    p = pr.validate_block(path, label, TASK, int(block), require_single=True,
                          setup=plan["setup"])
    if p:
        return p, None, None
    d, z = p16.load(path)
    j = d.get("joint") or {}
    for k, v in contract.items():
        if json.dumps(j.get(k), sort_keys=True) != json.dumps(
                v, sort_keys=True):
            p.append(f"{label}: контракт {k} = {j.get(k)!r}, план {v!r}")
    if j.get("label") != label or j.get("plan_sha1") != sha(plan_path):
        p.append(f"{label}: метка или план руки не из плана")
    z2 = dict(z)
    z2[f"k15d_level_{LEARNED}"] = z[f"k15d_level_{label}"]
    p += p16.check_arrays(LEARNED, z2, plan, [99] * plan["n_envs"])
    if not p:
        p += hpc.actions_problems(label, z, hpc.harness_scale())
    if not p:
        rd, rz = p16.load(plan["refs"][f"q0_i{block}"])
        p += p16.check_vs_q0(LEARNED, d, z2, rd, rz)
    return p, d, z


def check_learned(plan, plan_path, path, block):
    p, _d, _z = common_checks(plan, plan_path, path, LEARNED, block,
                              plan["expected_contract"]["learned"])
    return p


def build_reference(plan, plan_path, run_dir):
    """Таблица эталонов из проверенных learned-блоков."""
    import k16_protocol as p16
    calls, A0, LV, shas = [], [], [], {}
    for b in plan["blocks"]:
        pth = os.path.join(run_dir, f"{LEARNED}_t{TASK}_i{b}.json")
        prob = check_learned(plan, plan_path, pth, b)
        if prob:
            raise SystemExit(f"learned блока {b} не прошёл проверку: "
                             f"{prob[:2]}")
        d, z = p16.load(pth)
        imp = np.asarray(z["k16_impulse"], bool)
        c_, a_, l_ = [], [], []
        for e in range(plan["n_envs"]):
            k = np.nonzero(imp[:, e])[0]
            if len(k) != 1:
                raise SystemExit(f"блок {b}, среда {e}: импульсов {len(k)}")
            k = int(k[0])
            c_.append(k + 1)
            a_.append(z["k15d_a0"][k, e])
            l_.append(z[f"k15d_level_{LEARNED}"][k, e])
        calls.append(c_)
        A0.append(np.stack(a_))
        LV.append(np.stack(l_))
        shas[str(b)] = dict(json=sha(pth), npz=sha(os.path.join(
            run_dir, d["actions_npz"])))
    os.makedirs(os.path.dirname(REF), exist_ok=True)
    np.savez(REF, call=np.asarray(calls, np.int64),
             a0=np.stack(A0).astype(np.float32),
             level=np.stack(LV).astype(np.float32),
             blocks=np.asarray(plan["blocks"]))
    meta = dict(plan_sha1=sha(plan_path), learned_blocks=shas,
                reference_sha1=sha(REF))
    with open(REF[:-4] + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    return meta


def reference_ok(plan, plan_path, run_dir):
    """Эталон построен этим планом из неизменённых learned-блоков."""
    if not os.path.exists(REF) or not os.path.exists(REF[:-4] + ".json"):
        return ["нет таблицы эталонов (этап reference)"]
    m = json.load(open(REF[:-4] + ".json"))
    p = []
    if m["plan_sha1"] != sha(plan_path) or m["reference_sha1"] != sha(REF):
        p.append("таблица эталонов не от этого плана или изменена")
    for b, s_ in m["learned_blocks"].items():
        pth = os.path.join(run_dir, f"{LEARNED}_t{TASK}_i{b}.json")
        d = json.load(open(pth))
        if sha(pth) != s_["json"] or sha(os.path.join(
                run_dir, d["actions_npz"])) != s_["npz"]:
            p.append(f"learned блока {b} изменён после построения эталона")
    return p


def ref_for_block(R, block):
    i = R["blocks"].tolist().index(int(block))
    return dict(call=R["call"][i], a0=R["a0"][i], level=R["level"][i])


def check_rot(plan, plan_path, path, label, block, run_dir):
    import k16s_protocol as sp_
    import torch
    p = reference_ok(plan, plan_path, run_dir)
    if p:
        return p
    p, _d, z = common_checks(plan, plan_path, path, label, block,
                             plan["expected_contract"]["rot"])
    if p:
        return p
    R = np.load(REF)
    rb = ref_for_block(R, block)
    Rk = {k: v[None] for k, v in rb.items()}
    pseudo = dict(tasks=[TASK], n_envs=plan["n_envs"], block=int(block),
                  seeds=plan["seeds"])
    ck = torch.load(plan["inputs"]["models"], map_location="cpu",
                    weights_only=False)
    return sp_.reference_problems(label, z, pseudo, TASK, Rk,
                                  np.asarray(ck["scale_arm"], np.float64))


def analyze(plan, plan_path, run_dir):
    import k16_protocol as p16
    import k16r_analyze as ra
    ok, gaps = {}, []
    for b in plan["blocks"]:
        pth = os.path.join(run_dir, f"{LEARNED}_t{TASK}_i{b}.json")
        prob = check_learned(plan, plan_path, pth, b) if os.path.exists(
            pth) else ["не готов"]
        (gaps.append(dict(label=LEARNED, block=b, why=prob[:2])) if prob
         else ok.__setitem__((LEARNED, b), pth))
        for lab in plan["rot"]:
            pth = os.path.join(run_dir, f"{lab}_t{TASK}_i{b}.json")
            prob = check_rot(plan, plan_path, pth, lab, b, run_dir) \
                if os.path.exists(pth) else ["не готов"]
            (gaps.append(dict(label=lab, block=b, why=prob[:2])) if prob
             else ok.__setitem__((lab, b), pth))
    want = len(plan["blocks"]) * (1 + len(plan["rot"]))
    res = dict(verified=len(ok), planned=want, gaps=gaps,
               complete=len(ok) == want and not gaps)
    if not res["complete"]:
        return res

    def outc(paths):
        o = {}
        for pth in paths:
            d, _z = p16.load(pth)
            for e in d["episodes"]:
                o[(TASK, int(e["init_state_id"]))] = bool(e["success"])
        return o
    learned = outc([ok[(LEARNED, b)] for b in plan["blocks"]])
    ctrl = {lab: outc([ok[(lab, b)] for b in plan["blocks"]])
            for lab in plan["rot"]}
    q0 = outc([plan["refs"][f"q0_i{b}"] for b in plan["blocks"]])
    states = sorted(q0)
    res.update(ra.summarize(ctrl, learned, q0, states))
    # старые 5 состояний (K-16s, блок 25) — отдельно
    s_path = "reports/k16s/analysis.json"
    if os.path.exists(s_path):
        res["k16s_task8_block25"] = json.load(open(s_path)).get(
            "descriptive_task8", {}).get("task8")
    return res


def selftest():
    R = dict(blocks=np.array([30, 35]), call=np.array([[1] * 5, [2] * 5]),
             a0=np.zeros((2, 5, 8, 7)), level=np.ones((2, 5, 8, 7)))
    rb = ref_for_block(R, 35)
    assert (rb["call"] == 2).all() and rb["level"].shape == (5, 8, 7)
    assert set(ROT) == {"stored_rot_s1", "stored_rot_s2", "stored_rot_s3"}
    print("самопроверка k16t_protocol пройдена: эталон по блоку")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-16t: протокол")
    ap.add_argument("mode", choices=("plan", "check-plan", "check-learned",
                                     "reference", "check-rot", "analyze",
                                     "selftest"))
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--block", nargs=3, default=None,
                    metavar=("JSON", "МЕТКА", "БЛОК"))
    ap.add_argument("--out", default="reports/k16t/analysis.json")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    try:
        if a.mode == "plan":
            if os.path.exists(a.plan):
                raise SystemExit(f"{a.plan} уже зафиксирован")
            plan = make_plan(a.device)
            os.makedirs(os.path.dirname(a.plan), exist_ok=True)
            with open(a.plan + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=1)
            os.replace(a.plan + ".tmp", a.plan)
            print(f"  план {a.plan}: задача {TASK}, блоки {list(BLOCKS)}, "
                  f"процессов {len(BLOCKS) * (1 + len(ROT))}")
            return 0
        plan = json.load(open(a.plan))
        if a.mode == "check-plan":
            prob = check_plan_inputs(plan)
            print("\n".join(prob) if prob else "  входы плана сверены")
            return 3 if prob else 0
        if a.mode == "check-learned":
            path, _lab, b = a.block
            prob = check_learned(plan, a.plan, path, b)
            if prob:
                print("; ".join(prob[:6]))
            return 2 if prob else 0
        if a.mode == "check-rot":
            path, lab, b = a.block
            prob = check_rot(plan, a.plan, path, lab, b, a.run_dir)
            if prob:
                print("; ".join(prob[:6]))
            return 2 if prob else 0
        if a.mode == "reference":
            m = build_reference(plan, a.plan, a.run_dir)
            R = np.load(REF)
            print(f"  эталон {REF}: вызовы импульсов по блокам "
                  f"{dict(zip(R['blocks'].tolist(), R['call'].tolist()))}; "
                  f"sha {m['reference_sha1']}")
            return 0
        import k16r_analyze as ra
        res = analyze(plan, a.plan, a.run_dir)
        ra.report(res)
        if res.get("k16s_task8_block25"):
            print(f"  отдельно, старые 5 состояний (K-16s, блок 25): "
                  f"{res['k16s_task8_block25']}")
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
        print(f"  сводка: {a.out}")
        return 0 if res["complete"] else 3
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
