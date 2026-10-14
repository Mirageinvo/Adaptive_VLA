#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16r: план контроля направления и проверки блоков.

ПЛАН: три метки (сиды поворота заданы заранее), задачи 0–9, блок 25;
входы — план K-16 и его чекпойнт, базис, тождество; эталоны — блоки q0 и
goal_band_hist_event из каталога K-16 (переиспользуются, не
пересчитываются). Smoke — первая метка на всех десяти задачах.

ПРОВЕРКА БЛОКА: строгая validate_block и контракт; проверки K-16 для
goal_band_hist_event (события по журналу, один импульс в первом событии,
вне импульса = a0, схват = a0, δ журнала = действие − a0, предел) и
против q0 K-16 (старты, префикс и входы до импульса, вызов импульса =
первое событие q0); исполненные actions = журнал; матрицы поворота =
пересчитанным по сиду и состоянию; исходная δ и места импульсов = блоку
goal_band_hist_event K-16; применённая δ = повёрнутая исходная.
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

KIND = "k16r_plan"
PLAN = "reports/k16r/plan.json"
PLAN16 = "reports/k16/plan.json"
BASE = "goal_band_hist_event"
CODE = ("k16r_policy.py", "k16r_harness.py", "k16r_protocol.py",
        "k16r_analyze.py", "k16r_rollout.sh", "k16_policy.py",
        "k16_harness.py", "k16_protocol.py", "k16_models.py",
        "k9h_multiarm_gate.py", "k15g_local_probe.py", "k15h_protocol.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]


def k16_run_dir(plan16_path):
    d = os.path.join("reports/k16/run", f"p{sha(plan16_path)}")
    if not os.path.isdir(d):
        raise SystemExit(f"нет каталога K-16 {d}")
    return d


def make_plan(device):
    import k16_protocol as p16
    import k16r_policy as rp
    plan16 = json.load(open(PLAN16))
    if p16.check_plan_inputs(plan16):
        raise SystemExit("план K-16 не проходит проверку входов")
    run16 = k16_run_dir(PLAN16)
    refs = {}
    for t in plan16["tasks"]:
        for lab in ("q0", BASE):
            refs[f"{lab}_t{t}"] = os.path.join(
                run16, f"{lab}_t{t}_i{plan16['block']}.json")
    for v in refs.values():
        if not os.path.exists(v):
            raise SystemExit(f"нет блока K-16 {v}")
    contract = dict(plan16["expected_contract"])
    contract["harness_adapter_sha1"] = sha(os.path.join(HERE,
                                                        "k16r_harness.py"))
    contract["k16r_policy_sha1"] = sha(os.path.join(HERE, "k16r_policy.py"))
    inputs = dict(plan16=PLAN16, models=plan16["inputs"]["models"],
                  basis=plan16["inputs"]["basis"],
                  identity_report=plan16["inputs"]["identity_report"])
    return dict(
        kind=KIND, created=datetime.datetime.now().isoformat(
            timespec="seconds"),
        labels=list(rp.LABELS), seeds=dict(rp.ROT_SEEDS), base=BASE,
        tasks=plan16["tasks"], block=plan16["block"],
        n_envs=plan16["n_envs"], smoke_label=rp.LABELS[0],
        device=str(device), setup=plan16["setup"],
        expected_contract=contract,
        inputs={k: os.path.abspath(v) for k, v in inputs.items()},
        inputs_sha1={k: sha(v) for k, v in inputs.items()},
        refs={k: os.path.abspath(v) for k, v in refs.items()},
        refs_sha1={k: sha(v) for k, v in refs.items()},
        code={f: sha(os.path.join(HERE, f)) for f in CODE},
        rotation=("SO(3) отдельно для позиции и вращения, одна на все 8 "
                  "шагов, в исполняемых координатах; генератор "
                  "default_rng([сид, задача, состояние])"),
        note="контроль направления; 50 уникальных состояний, не 150")


def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план K-16r"]
    for grp in ("inputs", "refs"):
        for k, f in plan[grp].items():
            if not os.path.exists(f) or sha(f) != plan[f"{grp}_sha1"][k]:
                p.append(f"{grp} {k} изменён или отсутствует ({f})")
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    return p


def rotation_problems(label, z, plan, task, zb, scale):
    """Матрицы по сиду; исходная δ и места импульсов = блоку
    goal_band_hist_event K-16; применённая δ = повёрнутая исходная."""
    import k16r_policy as rp
    p = []
    seed = plan["seeds"][label]
    states = [plan["block"] + i for i in range(plan["n_envs"])]
    for b, s_ in enumerate(states):
        Rp, Rr = rp.rotations(seed, task, s_)
        if not (np.allclose(z["k16_rot_pos"][b], Rp) and
                np.allclose(z["k16_rot_rot"][b], Rr)):
            p.append(f"{label}: матрицы среды {b} не по сиду")
    imp, impb = np.asarray(z["k16_impulse"], bool), \
        np.asarray(zb["k16_impulse"], bool)
    if not np.array_equal(imp, impb):
        p.append(f"{label}: места импульсов не равны learned K-16")
        return p
    for k, b in zip(*np.nonzero(imp)):
        if not np.allclose(z["k16_d_codec_orig"][k, b],
                           zb["k16_d_codec"][k, b], atol=1e-6):
            p.append(f"{label}: исходная δ среды {b} не равна learned K-16")
        want = rp.rotate_delta(z["k16_d_codec_orig"][k, b], scale,
                               np.asarray(z["k16_rot_pos"][b]),
                               np.asarray(z["k16_rot_rot"][b]))
        if not np.allclose(z["k16_d_codec"][k, b], want, atol=1e-6):
            p.append(f"{label}: применённая δ среды {b} не равна повёрнутой")
    return p


def check_block(plan, plan_path, path, label, task):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import k16_protocol as p16
    import torch
    p = pr.validate_block(path, label, int(task), plan["block"],
                          require_single=True, setup=plan["setup"])
    if p:
        return p
    d, z = p16.load(path)
    j = d.get("joint") or {}
    for k, v in plan["expected_contract"].items():
        if json.dumps(j.get(k), sort_keys=True) != json.dumps(
                v, sort_keys=True):
            p.append(f"{label}: контракт {k} = {j.get(k)!r}, план {v!r}")
    if j.get("label") != label or j.get("plan_sha1") != sha(plan_path) or \
            j.get("rotation_seed") != plan["seeds"][label]:
        p.append(f"{label}: метка, план или сид руки не из плана")
    plan16 = json.load(open(plan["inputs"]["plan16"]))
    z2 = dict(z)
    z2[f"k15d_level_{BASE}"] = z[f"k15d_level_{label}"]
    p += p16.check_arrays(BASE, z2, plan16, [99] * plan["n_envs"])
    if not p:
        p += hpc.actions_problems(label, z, hpc.harness_scale())
    if not p:
        rd, rz = p16.load(plan["refs"][f"q0_t{task}"])
        p += p16.check_vs_q0(BASE, d, z2, rd, rz)
    if not p:
        _db, zb = p16.load(plan["refs"][f"{BASE}_t{task}"])
        ck = torch.load(plan["inputs"]["models"], map_location="cpu",
                        weights_only=False)
        p += rotation_problems(label, z, plan, int(task), zb,
                               np.asarray(ck["scale_arm"], np.float64))
    return p


def check_run(plan, plan_path, run_dir, labels=None):
    ok, gaps = {}, []
    for lab in (labels or plan["labels"]):
        for t in plan["tasks"]:
            pth = os.path.join(run_dir, f"{lab}_t{t}_i{plan['block']}.json")
            if not os.path.exists(pth):
                gaps.append(dict(label=lab, task=t, why="не готов"))
                continue
            prob = check_block(plan, plan_path, pth, lab, t)
            if prob:
                gaps.append(dict(label=lab, task=t, why="; ".join(prob[:2])))
            else:
                ok[(lab, t)] = pth
    return ok, gaps


def selftest():
    import k16r_policy as rp
    rng = np.random.default_rng(0)
    plan = dict(seeds=dict(rp.ROT_SEEDS), block=25, n_envs=2)
    lab = rp.LABELS[0]
    C, B = 4, 2
    scale = rng.uniform(0.3, 0.9, 6)
    rots = [rp.rotations(plan["seeds"][lab], 3, 25 + b) for b in range(B)]
    imp = np.zeros((C, B), bool)
    imp[2, 1] = True
    dco = np.full((C, B, 8, 6), np.nan)
    dco[2, 1] = rng.standard_normal((8, 6)) * 0.02
    dap = np.full((C, B, 8, 6), np.nan)
    dap[2, 1] = rp.rotate_delta(dco[2, 1], scale, *rots[1])
    z = {"k16_rot_pos": np.stack([r[0] for r in rots]),
         "k16_rot_rot": np.stack([r[1] for r in rots]),
         "k16_impulse": imp, "k16_d_codec_orig": dco, "k16_d_codec": dap}
    zb = {"k16_impulse": imp, "k16_d_codec": dco}
    assert rotation_problems(lab, z, plan, 3, zb, scale) == []
    zb2 = {"k16_impulse": imp, "k16_d_codec": dco * 1.1}
    assert rotation_problems(lab, z, plan, 3, zb2, scale)
    assert rotation_problems(lab, z, plan, 4, zb, scale)      # не та задача
    z3 = dict(z, k16_d_codec=dco)                              # без поворота
    assert rotation_problems(lab, z3, plan, 3, zb, scale)
    imp2 = imp.copy()
    imp2[1, 0] = True
    assert rotation_problems(lab, z, plan, 3, dict(zb, k16_impulse=imp2),
                             scale)
    print("самопроверка k16r_protocol пройдена: матрицы по сиду, исходная "
          "δ и места = learned K-16, применённая = повёрнутая")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-16r: протокол")
    ap.add_argument("mode", choices=("plan", "check-plan", "check-block",
                                     "smoke-check", "selftest"))
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--block", nargs=3, default=None,
                    metavar=("JSON", "МЕТКА", "ЗАДАЧА"))
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
            print(f"  план {a.plan}: метки {plan['labels']}, сиды "
                  f"{plan['seeds']}, задач {len(plan['tasks'])}, процессов "
                  f"{len(plan['labels']) * len(plan['tasks'])}")
            return 0
        plan = json.load(open(a.plan))
        if a.mode == "check-plan":
            prob = check_plan_inputs(plan)
            print("\n".join(prob) if prob else "  входы плана сверены")
            return 3 if prob else 0
        if a.mode == "check-block":
            path, lab, t = a.block
            prob = check_block(plan, a.plan, path, lab, t)
            if prob:
                print("; ".join(prob[:6]))
            return 2 if prob else 0
        ok, gaps = check_run(plan, a.plan, a.run_dir, [plan["smoke_label"]])
        print(f"  smoke: проверено блоков {len(ok)} из {len(plan['tasks'])};"
              f" проблем {len(gaps)}" + (f": {gaps[:6]}" if gaps else ""))
        return 3 if gaps else 0
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
