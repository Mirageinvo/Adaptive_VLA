#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16s: план контроля записанного вмешательства и проверки блоков.

ПЛАН: метки stored_replay (техническая) и stored_rot_s1..s3; задачи 0–9,
блок 25. Таблица эталонов (reports/k16s/reference.npz) строится из
learned-блоков goal_band_hist_event K-16 (каждый заново проверен штатной
проверкой K-16; SHA JSON и npz закреплены): вызов импульса, a0 и действие
learned на каждую среду. Smoke: stored_replay и stored_rot_s3 на задаче 2.

ПРОВЕРКА БЛОКА: строгая validate_block и контракт; проверки K-16 для
событийного импульса (по журналу) и против q0 K-16 (старты, префикс, a0
и состояние до импульса, вызов = первое событие q0); исполненные actions =
журнал; импульс каждой среды — ровно в эталонном вызове, a0 там побитово =
эталону; stored_replay — рука побитово = сохранённым командам learned;
stored_rot — матрицы по сиду, исходная δ = эталонной, применённая =
повёрнутой (с точностью округления float32).
SMOKE-ПРОВЕРКА stored_replay: исполненные actions, done_step и исходы
совпадают с learned K-16.
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

KIND = "k16s_plan"
PLAN = "reports/k16s/plan.json"
REF = "reports/k16s/reference.npz"
PLAN16 = "reports/k16/plan.json"
BASE = "goal_band_hist_event"
SMOKE = [("stored_replay", 2), ("stored_rot_s3", 2)]
CODE = ("k16s_policy.py", "k16s_harness.py", "k16s_protocol.py",
        "k16s_analyze.py", "k16s_rollout.sh", "k16r_policy.py",
        "k16r_analyze.py", "k16_policy.py", "k16_harness.py",
        "k16_protocol.py", "k16_models.py", "k9h_multiarm_gate.py",
        "k15g_local_probe.py", "k15h_protocol.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]


def make_plan(device):
    import k15g_local_probe as pr
    import k16_protocol as p16
    import k16r_protocol as r16
    import k16s_policy as sp
    plan16 = json.load(open(PLAN16))
    if p16.check_plan_inputs(plan16):
        raise SystemExit("план K-16 не проходит проверку входов")
    run16 = r16.k16_run_dir(PLAN16)
    blk = plan16["block"]
    refs, calls, A0, LV = {}, [], [], []
    for t in plan16["tasks"]:
        q0p = os.path.join(run16, f"q0_t{t}_i{blk}.json")
        for lab in ("q0", BASE):
            v = os.path.join(run16, f"{lab}_t{t}_i{blk}.json")
            prob = p16.check_block(plan16, PLAN16, v, lab, t, q0p)
            prob += pr.validate_block(v, lab, t, blk, require_single=True,
                                      setup=plan16["setup"])
            if prob:
                raise SystemExit(f"блок K-16 {v} не прошёл проверку: "
                                 f"{prob[:2]}")
            refs[f"{lab}_t{t}"] = v
            d = json.load(open(v))
            refs[f"{lab}_t{t}__npz"] = os.path.join(run16, d["actions_npz"])
        _d, z = p16.load(refs[f"{BASE}_t{t}"])
        imp = np.asarray(z["k16_impulse"], bool)
        c_, a_, l_ = [], [], []
        for b in range(plan16["n_envs"]):
            k = np.nonzero(imp[:, b])[0]
            if len(k) != 1:
                raise SystemExit(f"задача {t}, среда {b}: импульсов {len(k)}")
            k = int(k[0])
            c_.append(k + 1)
            a_.append(z["k15d_a0"][k, b])
            l_.append(z[f"k15d_level_{BASE}"][k, b])
        calls.append(c_)
        A0.append(np.stack(a_))
        LV.append(np.stack(l_))
    os.makedirs(os.path.dirname(REF), exist_ok=True)
    np.savez(REF, call=np.asarray(calls, np.int64),
             a0=np.stack(A0).astype(np.float32),
             level=np.stack(LV).astype(np.float32),
             tasks=np.asarray(plan16["tasks"]))
    contract = dict(plan16["expected_contract"])
    contract["harness_adapter_sha1"] = sha(os.path.join(HERE,
                                                        "k16s_harness.py"))
    contract["k16s_policy_sha1"] = sha(os.path.join(HERE, "k16s_policy.py"))
    inputs = dict(plan16=PLAN16, models=plan16["inputs"]["models"],
                  basis=plan16["inputs"]["basis"],
                  identity_report=plan16["inputs"]["identity_report"],
                  reference=REF)
    return dict(
        kind=KIND, created=datetime.datetime.now().isoformat(
            timespec="seconds"),
        labels=list(sp.LABELS), seeds=dict(sp.SEEDS), base=BASE,
        tasks=plan16["tasks"], block=blk, n_envs=plan16["n_envs"],
        smoke=SMOKE, device=str(device), setup=plan16["setup"],
        expected_contract=contract,
        inputs={k: os.path.abspath(v) for k, v in inputs.items()},
        inputs_sha1={k: sha(v) for k, v in inputs.items()},
        refs={k: os.path.abspath(v) for k, v in refs.items()},
        refs_sha1={k: sha(v) for k, v in refs.items()},
        code={f: sha(os.path.join(HERE, f)) for f in CODE},
        reference_rule=("δ_ref = (действие learned − его a0)[:8, :6] в "
                        "float64 из learned-блока K-16; выбор без исходов"),
        note=("контроль направления записанного вмешательства на этих 50 "
              "состояниях; не политика для неизвестных стартов"))


def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план K-16s"]
    for grp in ("inputs", "refs"):
        for k, f in plan[grp].items():
            if not os.path.exists(f) or sha(f) != plan[f"{grp}_sha1"][k]:
                p.append(f"{grp} {k} изменён или отсутствует ({f})")
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    return p


def reference_problems(label, z, plan, task, R, scale):
    import k16r_policy as rp
    import k16s_policy as sp
    p = []
    ti = plan["tasks"].index(int(task))
    imp = np.asarray(z["k16_impulse"], bool)
    lv = z[f"k15d_level_{label}"]
    for b in range(plan["n_envs"]):
        calls = (np.nonzero(imp[:, b])[0] + 1).tolist()
        if calls != [int(R["call"][ti, b])]:
            p.append(f"{label}: среда {b}: импульс {calls}, эталон "
                     f"{int(R['call'][ti, b])}")
            continue
        k = calls[0] - 1
        if not np.array_equal(z["k15d_a0"][k, b], R["a0"][ti, b]):
            p.append(f"{label}: среда {b}: a0 импульса ≠ эталону")
        dref = sp.ref_delta(R["level"][ti, b], R["a0"][ti, b])
        if not np.allclose(z["k16_d_codec_orig"][k, b], dref, atol=1e-7):
            p.append(f"{label}: среда {b}: исходная δ ≠ эталонной")
        if label == "stored_replay":
            if not np.array_equal(lv[k, b, :, :6], R["level"][ti, b][:, :6]):
                p.append(f"{label}: среда {b}: рука ≠ сохранённым командам")
        else:
            seed = plan["seeds"][label]
            Rp, Rr = rp.rotations(seed, int(task), plan["block"] + b)
            if not (np.allclose(z["k16_rot_pos"][b], Rp) and
                    np.allclose(z["k16_rot_rot"][b], Rr)):
                p.append(f"{label}: матрицы среды {b} не по сиду")
            want = rp.rotate_delta(dref, scale, Rp, Rr)
            got = lv[k, b, :, :6].astype(np.float64) - \
                z["k15d_a0"][k, b, :, :6].astype(np.float64)
            if not np.allclose(got, want, atol=1e-6):
                p.append(f"{label}: среда {b}: применённая δ ≠ повёрнутой")
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
            j.get("rotation_seed") != plan["seeds"].get(label) or \
            j.get("reference_sha1") != plan["inputs_sha1"]["reference"]:
        p.append(f"{label}: метка, план, сид или эталон руки не из плана")
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
        R = np.load(plan["inputs"]["reference"])
        ck = torch.load(plan["inputs"]["models"], map_location="cpu",
                        weights_only=False)
        p += reference_problems(label, z, plan, task, R,
                                np.asarray(ck["scale_arm"], np.float64))
    return p


def replay_matches_learned(plan, path, task):
    """stored_replay = learned K-16: исполненные actions, done_step, исходы."""
    import k16_protocol as p16
    d, z = p16.load(path)
    dl, zl = p16.load(plan["refs"][f"{BASE}_t{task}"])
    p = []
    for f in ("success", "done_step", "action_sha1"):
        if [str(e.get(f)) for e in d["episodes"]] != \
                [str(e.get(f)) for e in dl["episodes"]]:
            p.append(f"stored_replay: {f} ≠ learned K-16")
    if not np.array_equal(np.asarray(z["done_step"]),
                          np.asarray(zl["done_step"])):
        p.append("stored_replay: done_step npz ≠ learned K-16")
    return p


def check_run(plan, plan_path, run_dir, pairs):
    ok, gaps = {}, []
    for lab, t in pairs:
        pth = os.path.join(run_dir, f"{lab}_t{t}_i{plan['block']}.json")
        if not os.path.exists(pth):
            gaps.append(dict(label=lab, task=t, why="не готов"))
            continue
        prob = check_block(plan, plan_path, pth, lab, t)
        if not prob and lab == "stored_replay":
            prob = replay_matches_learned(plan, pth, t)
        if prob:
            gaps.append(dict(label=lab, task=t, why="; ".join(prob[:2])))
        else:
            ok[(lab, t)] = pth
    return ok, gaps


def selftest():
    import k16r_policy as rp
    import k16s_policy as sp
    rng = np.random.default_rng(0)
    plan = dict(tasks=[3], n_envs=2, block=25, seeds=dict(sp.SEEDS))
    C, B = 4, 2
    a0 = rng.uniform(-.5, .5, (C, B, 8, 7)).astype(np.float32)
    level_ref = a0[[1, 2], [0, 1]].copy()
    level_ref[..., :6] += np.float32(0.01)
    R = dict(call=np.array([[2, 3]]), a0=a0[[1, 2], [0, 1]][None],
             level=level_ref[None])
    scale = np.full(6, 0.9)

    def journal(label):
        lv = a0.copy()
        imp = np.zeros((C, B), bool)
        dor = np.full((C, B, 8, 6), np.nan, np.float32)
        for b, k in ((0, 1), (1, 2)):
            imp[k, b] = True
            dr = sp.ref_delta(R["level"][0, b], R["a0"][0, b])
            dor[k, b] = dr
            if label == "stored_replay":
                lv[k, b, :, :6] = R["level"][0, b][:, :6]
            else:
                Rp, Rr = rp.rotations(plan["seeds"][label], 3, 25 + b)
                lv[k, b, :, :6] = (a0[k, b, :, :6].astype(np.float64)
                                   + rp.rotate_delta(dr, scale, Rp, Rr)
                                   ).astype(np.float32)
        z = {"k16_impulse": imp, "k15d_a0": a0, f"k15d_level_{label}": lv,
             "k16_d_codec_orig": dor}
        if label != "stored_replay":
            rr = [rp.rotations(plan["seeds"][label], 3, 25 + b)
                  for b in range(B)]
            z["k16_rot_pos"] = np.stack([r[0] for r in rr])
            z["k16_rot_rot"] = np.stack([r[1] for r in rr])
        return z
    for lab in ("stored_replay", "stored_rot_s2"):
        assert reference_problems(lab, journal(lab), plan, 3, R, scale) == []
    z = journal("stored_rot_s2")
    z["k16_impulse"] = z["k16_impulse"].copy()
    z["k16_impulse"][3, 0] = True                     # второй импульс
    assert reference_problems("stored_rot_s2", z, plan, 3, R, scale)
    z = journal("stored_replay")
    z["k15d_level_stored_replay"] = z["k15d_level_stored_replay"].copy()
    z["k15d_level_stored_replay"][1, 0, 0, 0] += 1e-6  # не побитово
    assert reference_problems("stored_replay", z, plan, 3, R, scale)
    zz = journal("stored_rot_s1")
    zz["k16_rot_pos"] = journal("stored_rot_s2")["k16_rot_pos"]
    assert reference_problems("stored_rot_s1", zz, plan, 3, R, scale)
    Ra = dict(R, a0=R["a0"].copy())
    Ra["a0"][0, 1, 0, 0] += 1e-3
    assert reference_problems("stored_rot_s2", journal("stored_rot_s2"),
                              plan, 3, Ra, scale)
    print("самопроверка k16s_protocol пройдена: эталонный вызов и a0, "
          "воспроизведение побитово, поворот по сиду, второй импульс")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-16s: протокол")
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
            with open(a.plan + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=1)
            os.replace(a.plan + ".tmp", a.plan)
            R = np.load(REF)
            print(f"  план {a.plan}: метки {plan['labels']}, задач "
                  f"{len(plan['tasks'])}; эталонные вызовы по задачам: "
                  + "; ".join(f"t{t}: {R['call'][i].tolist()}"
                              for i, t in enumerate(plan["tasks"])))
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
        ok, gaps = check_run(plan, a.plan, a.run_dir,
                             [tuple(x) for x in plan["smoke"]])
        print(f"  smoke: проверено блоков {len(ok)} из {len(plan['smoke'])};"
              f" проблем {len(gaps)}" + (f": {gaps[:4]}" if gaps else ""))
        return 3 if gaps else 0
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
