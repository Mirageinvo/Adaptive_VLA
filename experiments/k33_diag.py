#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-33: диагностика расхождения bank-check (post-wait хеши ≠ банку).

Без политик, без изменения банка и k33_bank. Один процесс, случаи по
порядку; для каждого случая четыре трассы одним и тем же кодом k33_bank:
  G1, G2 — путь генерации (create -> seed -> native reset -> exec seed ->
           10 шагов ожидания), как k33_bank.generate;
  R1, R2 — путь воспроизведения (то же + set_init_state(S0) после reset),
           как k33_bank.restore_into/replay, но без отказов: всё пишется.
Снимки: после native reset, после set_init_state, после каждого шага
ожидания — flattened state, qacc_warmstart, ctrl, act, qfrc_applied,
состояние контроллера руки/хвата (goal_*, torques и т. п.), state
наблюдения и SHA кадров. Отчёт: хеши каждой трассы против банка и первая
стадия/поле, где G1 и R1 (а также G1 и G2) расходятся.
Шаги среды учитываются в журнале K-33 (scope checks).
  python experiments/k33_diag.py --cases 0:9:0,0:9:1,0:0:0,1:9:0
  (space:task:позиция случая в банке задачи)
  --post-fix: текущий k33_bank — generate дважды и replay одного seed,
  равенство post-wait хешей (проверка правки set_init_state в generate).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k33_bank as kb  # noqa: E402

OUT = "reports/k33/diag_bank_check.json"
TRACES = ("G1", "R1", "G2", "R2")
CTRL_ATTRS = ("goal_pos", "goal_ori", "goal_qpos", "goal_torque", "torques",
              "initial_joint", "initial_ee_pos", "initial_ee_ori_mat",
              "ee_pos", "ee_ori_mat", "joint_pos", "joint_vel",
              "relative_ori", "ori_ref", "current_action", "delta",
              "last_goal_pos", "last_goal_ori")


def _arr(x):
    try:
        a = np.asarray(x, np.float64)
        return a.copy() if a.dtype != object else None
    except Exception:                                        # noqa: BLE001
        return None


def controller_state(env):
    out = {}
    try:
        robots = env.env.robots
    except Exception:                                        # noqa: BLE001
        return out
    for ri, r in enumerate(robots):
        for cname in ("controller", "gripper_controller"):
            c = getattr(r, cname, None)
            if c is None:
                continue
            for a in CTRL_ATTRS:
                if hasattr(c, a):
                    v = _arr(getattr(c, a))
                    if v is not None:
                        out[f"r{ri}.{cname}.{a}"] = v
        for a in ("recent_qpos", "recent_actions", "recent_torques"):
            buf = getattr(r, a, None)
            v = _arr(getattr(buf, "current", None)) if buf is not None else None
            if v is not None:
                out[f"r{ri}.{a}.current"] = v
    return out


def snap(env, obs=None):
    d = env.sim.data
    s = dict(state=np.array(env.get_sim_state(), np.float64, copy=True),
             qacc_warmstart=np.array(d.qacc_warmstart, np.float64, copy=True),
             ctrl=np.array(d.ctrl, np.float64, copy=True),
             qfrc_applied=np.array(d.qfrc_applied, np.float64, copy=True),
             xfrc_applied=np.array(d.xfrc_applied, np.float64, copy=True))
    if getattr(d, "act", None) is not None and np.size(d.act):
        s["act"] = np.array(d.act, np.float64, copy=True)
    for k, v in controller_state(env).items():
        s["ctl:" + k] = v
    for name, ob in sorted(getattr(env.env, "_observables", {}).items()):
        for a in ("_time_since_last_sample", "_current_delay",
                  "_sampling_timestep"):
            v = getattr(ob, a, None)
            if v is not None:
                s[f"obt:{name}.{a}"] = np.asarray(v, np.float64).reshape(-1)
    if obs is not None:
        s["obs_state"] = np.asarray(obs["state"], np.float64).copy()
        for c in kb.CAMERAS:
            s[f"img:{c}"] = hashlib.sha1(np.ascontiguousarray(
                obs[f"{c}_image"]).tobytes()).hexdigest()[:16]
    return s


def trace(task, seed, S0=None):
    env, _i = kb._create(task, seed)
    try:
        kb.seed_all(seed)
        env.reset()
        st = dict(reset=snap(env))
        if S0 is not None:
            env.set_init_state(np.asarray(S0, np.float64))
            st["set"] = snap(env)
        kb.seed_all(kb.EXEC_SEED)
        obs = None
        for k in range(kb.WAIT):
            env.step(np.array(kb.DUMMY))
            obs = kb._obs_now(env)
            st[f"step{k}"] = snap(env, obs)
        ih, ihf = kb.post_wait_hashes(obs)
        return dict(snaps=st, init_hash=ih, init_hash_full=ihf)
    finally:
        env.close()


def diff(a, b):
    """Поля, различающиеся между двумя снимками: имя -> max|Δ| или 'sha'."""
    out = {}
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            out[k] = "нет в одном"
            continue
        x, y = a[k], b[k]
        if isinstance(x, str):
            if x != y:
                out[k] = "sha"
        elif x.shape != y.shape:
            out[k] = f"форма {x.shape} vs {y.shape}"
        elif not np.array_equal(x, y, equal_nan=True):
            dd = np.abs(x - y)
            out[k] = float(np.nanmax(dd)) if np.isfinite(dd).any() \
                else "NaN-маска"
    return out


def stage_diffs(A, B, stages):
    """Все стадии с различиями: стадия -> поля."""
    out = {}
    for s in stages:
        if s in A["snaps"] and s in B["snaps"]:
            d = diff(A["snaps"][s], B["snaps"][s])
            if d:
                out[s] = d
    return out


def first_diff(A, B, stages):
    for s in stages:
        if s in A["snaps"] and s in B["snaps"]:
            d = diff(A["snaps"][s], B["snaps"][s])
            if d:
                return dict(stage=s, fields=d)
    return None


def post_fix(specs, kp):
    """Текущий k33_bank: generate и replay одного seed (новый объект среды
    на каждый вызов) дают одинаковые post-wait хеши? Банк не читается для
    хешей — только seed случая."""
    L = kp.ledger()
    eid = f"checks/diag_bank_postfix/{int(time.time())}"
    L.reserve("checks", eid, amount=len(specs) * 3 * kb.WAIT)
    steps, out = 0, []
    for space, task, pos in specs:
        cases, _meta = kb.load_task(space, task)
        seed = cases[pos]["seed"]
        g = kb.generate(task, seed)
        g2 = kb.generate(task, seed)
        case = dict(case_id=kb.case_id(space, task, cases[pos]["idx"]),
                    seed=seed, S0=g["S0"], body_pos=g["body_pos"],
                    body_quat=g["body_quat"], sig=g["sig"])
        r = kb.replay(task, case, g["meta"])
        steps += 3 * kb.WAIT
        hg = [g["init_hash"], g["init_hash_full"]]
        row = dict(case_id=case["case_id"],
                   gen_eq_gen=hg == [g2["init_hash"], g2["init_hash_full"]],
                   gen_eq_replay=hg == [r["init_hash"], r["init_hash_full"]],
                   S0_eq_bank=bool(np.array_equal(g["S0"], cases[pos]["S0"])),
                   sig_eq_bank=g["sig"] == cases[pos]["sig"])
        out.append(row)
        print(f"  {row}", flush=True)
    L.commit(eid, steps)
    ok = all(r["gen_eq_gen"] and r["gen_eq_replay"] for r in out)
    print("  после правки: generate = replay " + ("во всех случаях" if ok
                                                    else "НЕ во всех"))
    kp.save_json(OUT.replace(".json", "_postfix.json"),
                 dict(cases=out, ok=ok))
    return 0 if ok else 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="0:9:0,0:9:1,0:0:0,1:9:0")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--post-fix", action="store_true")
    a = ap.parse_args()
    import k33_protocol as kp
    specs = [tuple(int(v) for v in s.split(":")) for s in a.cases.split(",")]
    if a.post_fix:
        return post_fix(specs, kp)
    L = kp.ledger()
    eid = f"checks/diag_bank/{int(time.time())}"
    L.reserve("checks", eid, amount=len(specs) * len(TRACES) * kb.WAIT)
    steps, rep = 0, dict(cases=[], order=a.cases)
    for space, task, pos in specs:
        cases, _meta = kb.load_task(space, task)
        c = cases[pos]
        T = {}
        for name in TRACES:
            T[name] = trace(task, c["seed"],
                            c["S0"] if name.startswith("R") else None)
            steps += kb.WAIT
        stages = ["reset", "set"] + [f"step{k}" for k in range(kb.WAIT)]
        r = dict(case_id=c["case_id"], seed=c["seed"],
                 bank=[c["init_hash"], c["init_hash_full"]],
                 hashes={n: [T[n]["init_hash"], T[n]["init_hash_full"]]
                         for n in TRACES})
        r["match_bank"] = {n: r["hashes"][n] == r["bank"] for n in TRACES}
        # S0 банка против состояния после native reset в этом процессе
        r["S0_vs_reset_G1"] = diff(dict(state=np.asarray(c["S0"])),
                                   dict(state=T["G1"]["snaps"]["reset"]
                                        ["state"]))
        r["G1_vs_G2"] = first_diff(T["G1"], T["G2"], stages)
        r["R1_vs_R2"] = first_diff(T["R1"], T["R2"], stages)
        # G1 vs R1: reset (что даёт native reset), затем set R1 против reset G1
        g, rr = T["G1"]["snaps"], T["R1"]["snaps"]
        r["G1reset_vs_R1reset"] = diff(g["reset"], rr["reset"])
        r["G1reset_vs_R1set"] = diff(g["reset"], rr["set"])
        r["G1_vs_R1_steps"] = first_diff(
            T["G1"], T["R1"], [f"step{k}" for k in range(kb.WAIT)])
        st_w = [f"step{k}" for k in range(kb.WAIT)]
        r["G1_vs_R1_all"] = {k: sorted(v) for k, v in
                             stage_diffs(T["G1"], T["R1"], st_w).items()}
        r["G1_vs_G2_hash"] = r["hashes"]["G1"] == r["hashes"]["G2"]
        r["R1_vs_R2_hash"] = r["hashes"]["R1"] == r["hashes"]["R2"]
        rep["cases"].append(r)
        print(f"  {c['case_id']}: банк={r['match_bank']}", flush=True)
        print(f"    хеши: G1=G2 {r['G1_vs_G2_hash']}, R1=R2 "
              f"{r['R1_vs_R2_hash']}", flush=True)
        print("    G1 vs R1 по шагам: " + ("; ".join(
            f"{k}: {','.join(v)}" for k, v in r["G1_vs_R1_all"].items())
            or "равны"), flush=True)
        for k in ("S0_vs_reset_G1", "G1reset_vs_R1reset", "G1reset_vs_R1set",
                  "G1_vs_G2", "R1_vs_R2", "G1_vs_R1_steps"):
            v = r[k]
            if isinstance(v, dict) and "fields" in v:
                v = dict(stage=v["stage"], fields=dict(
                    list(v["fields"].items())[:12]))
            elif isinstance(v, dict):
                v = dict(list(v.items())[:12])
            print(f"    {k}: {json.dumps(v, ensure_ascii=False) if v else 'равны'}",
                  flush=True)
    L.commit(eid, steps)
    kp.save_json(a.out, rep)
    print(f"  отчёт: {a.out}; шагов {steps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
