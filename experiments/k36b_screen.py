#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-36, этап B (§8.5): development-screen Tplus и гейт B.

Только после завершения ОБОИХ продолжений (k36b_train.train_problems) и при
свежем решении A. Tplus_s0/s1 исполняются на закреплённых 30 A-сценах
каждой задачи 0/6/8 (prepare: splits[t].screen): 90 на сид, 180 всего,
scope screen_b журнала K-36. R/T32 на этих же случаях берутся из полного
проверенного A и заново не исполняются. Эпизоды и их проверка —
k36_harness (стадия screen_b).
Гейт B (точные счётчики, без округления):
  1. joint Δ(Tplus−T32) ≥ +0.03 (равные веса задач и сидов);
  2. на каждом сиде успехов Tplus ≥ успехов T32 − 2 (из 90);
  3. на каждом сиде успехов Tplus ≥ успехов R − 2;
  4. actor изменился и принят хотя бы один новый update (оба сида);
  все проверки пройдены. Статус passed_B_candidate / stopped_B_no_gain.
Решение закрепляет SHA всех используемых эпизодов, отчётов продолжения и
final.pt; проверка решения пересчитывает статус по исходам.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k36_bank as kb36  # noqa: E402
import k36_harness as kh  # noqa: E402
import k36_protocol as pc  # noqa: E402
import k36b_train as kbt  # noqa: E402

GATE_B = "reports/k36/gate_b.json"
STAGE = "screen_b"
kh.STAGE_SCOPE.setdefault(STAGE, "screen_b")
TASKS = kb36.ADAPTED


def tplus_label(s):
    return f"Tplus_s{s}"


def tplus_hash(s):
    r = json.load(open(kbt.report_path(s), encoding="utf-8"))
    return r["actor_hash"], r["final_sha1"]


def screen_want(prep, s, t, cid, ah):
    sp = prep["splits"][str(t)]
    h = sp["init_hash"][cid]
    return kh.want_of(prep, STAGE, tplus_label(s), t, cid, sp["pos"][cid],
                      "bank", space=kb36.SPACE_CONF, case_id=cid,
                      want_hash=tuple(h), actor_hash=ah)


def screen_wants(prep, hashes):
    return [screen_want(prep, s, t, cid, hashes[s])
            for s in pc.parse_seeds() for t in TASKS
            for cid in prep["splits"][str(t)]["screen"]]


def load_tplus(torch, prep, s, H):
    ah, fsha = tplus_hash(s)
    if pc.sha(kbt.final_path(s)) != fsha:
        raise SystemExit(f"Tplus_s{s}: final.pt ≠ отчёту продолжения")
    a, _c = kh.load_actor(torch, prep, "T", s, H, path=kbt.final_path(s),
                          want_hash=ah)
    return a


def _screen_seed(torch, prep, C, L, sim, s, wants, device, loader=None,
                 actor=None):
    v = kh.load_source(torch, s, device, loader)
    act = actor if actor is not None else load_tplus(torch, prep, s,
                                                     v["mod"].H)
    arm = kh.make_arm(torch, prep, f"T32_s{s}", v, actor=act)
    out = []
    for w in wants:
        if w["seed"] == s:
            out.append((w, kh.run_ep(w, sim, v["io"], arm, L, C)))
    return out


def outcomes(prep, hashes, C):
    """-> (Y[name][s][t] массивы по screen-случаям, файлы, проблемы)."""
    Y = {nm: {s: {} for s in (0, 1)} for nm in ("Tplus", "T32", "R")}
    files, p = {}, []
    for s in pc.parse_seeds():
        for t in TASKS:
            ids = prep["splits"][str(t)]["screen"]
            for nm in Y:
                y = np.zeros(len(ids), bool)
                for i, cid in enumerate(ids):
                    w = screen_want(prep, s, t, cid, hashes[s]) \
                        if nm == "Tplus" else kh.a_want(prep, f"{nm}_s{s}",
                                                        t, cid)
                    got = kh.load_ep(w)
                    if got is None:
                        p.append(f"{w['eid']}: нет эпизода")
                        continue
                    pr = kh.episode_problems(*got, w, C)
                    if pr:
                        p.append(f"{w['eid']}: {pr[:2]}")
                    y[i] = bool(got[0]["summary"]["success"])
                    files.update(kh.ep_files(w))
                Y[nm][s][t] = y
    return Y, files, p


def rule_b(Y, train_reps):
    R = pc.RULE_B
    d = {t: np.mean([Y["Tplus"][s][t].astype(float) -
                     Y["T32"][s][t].astype(float) for s in (0, 1)], 0)
         for t in TASKS}
    joint = float(np.mean([d[t].mean() for t in TASKS]))
    cnt = {s: {nm: int(sum(Y[nm][s][t].sum() for t in TASKS))
               for nm in Y} for s in (0, 1)}
    flags = dict(
        joint=bool(joint >= R["joint_Tplus_T32"]),
        lag_T32=all(cnt[s]["Tplus"] >= cnt[s]["T32"] - R["per_seed_lag_T32"]
                    for s in (0, 1)),
        lag_R=all(cnt[s]["Tplus"] >= cnt[s]["R"] - R["per_seed_lag_R"]
                  for s in (0, 1)),
        actor_updated=all(r.get("actor_changed") is True and
                          int(r.get("accepted_added", 0)) >= 1
                          for r in train_reps.values()))
    st = "passed_B_candidate" if all(flags.values()) else "stopped_B_no_gain"
    return st, flags, dict(joint_delta=joint, successes=cnt)


def screen_b(a, sim=None, loader=None):
    sim = sim or kb36.BankSim("bank", kb36.SPACE_CONF)
    import torch
    import k36_analyze as an
    try:
        prep = pc.load_prepare()
        p = an.gate_a_problems()
        for s in pc.parse_seeds():
            p += [f"Tplus_s{s}: {x}" for x in kbt.train_problems(s, prep)]
        if p:
            raise SystemExit("screen-b не разрешён: " + "; ".join(p[:3]))
        pr = kh.resources_problems(torch, a.device)
        if pr:
            raise SystemExit("; ".join(pr))
        C = kh.consts(prep)
        L = pc.ledger()
        hashes = {s: tplus_hash(s)[0] for s in pc.parse_seeds()}
        wants = screen_wants(prep, hashes)
        for s in pc.parse_seeds():
            _screen_seed(torch, prep, C, L, sim, s, wants, a.device, loader)
            kh.free(torch)
        return decide_b(prep, hashes, C)
    finally:
        sim.close()


def decide_b(prep, hashes, C):
    Y, files, p = outcomes(prep, hashes, C)
    reps = {s: json.load(open(kbt.report_path(s), encoding="utf-8"))
            for s in pc.parse_seeds()}
    if p:
        pc.save_json(GATE_B, dict(status="technical_stop", problems=p[:50]))
        print("  гейт B: ТЕХНИЧЕСКИЙ СТОП — " + "; ".join(p[:3]))
        return 3
    st, flags, det = rule_b(Y, reps)
    for s in pc.parse_seeds():
        for f in (kbt.report_path(s), kbt.final_path(s)):
            files[f] = pc.sha(f)
    pc.save_json(GATE_B, dict(
        status=st, flags=flags, details=det, rule=pc.RULE_B,
        prepare_sha1=prep["sha1"], files=files,
        tplus_actor_hash={str(s): h for s, h in hashes.items()},
        table={f"{nm}_s{s}": {str(t): Y[nm][s][t].astype(int).tolist()
                              for t in TASKS} for nm in Y for s in (0, 1)}))
    c = det["successes"]
    print(f"  гейт B: joint Δ(Tplus−T32) {100 * det['joint_delta']:+.1f} "
          f"п.п.; успехи из 90 — " + "; ".join(
              f"s{s}: Tplus {c[s]['Tplus']}, T32 {c[s]['T32']}, R "
              f"{c[s]['R']}" for s in (0, 1)) + f"; {flags} — {st}")
    return 0


def gate_b_problems():
    """Решение B свежее: файлы, prepare, пересчёт статуса по исходам."""
    if not os.path.exists(GATE_B):
        return ["нет решения B"]
    g = json.load(open(GATE_B, encoding="utf-8"))
    try:
        prep = pc.load_prepare()
    except SystemExit as e:
        return [str(e.code)]
    if g.get("prepare_sha1") != prep["sha1"]:
        return ["решение B другого prepare"]
    for f, s in (g.get("files") or {}).items():
        if not os.path.exists(f) or pc.sha(f) != s:
            return ["файлы B изменены после решения"]
    C = kh.consts(prep)
    hashes = {s: g["tplus_actor_hash"][str(s)] for s in pc.parse_seeds()}
    Y, files, p = outcomes(prep, hashes, C)
    if p:
        return [f"данные B: {p[:2]}"]
    want = set(files) | {f for s in pc.parse_seeds() for f in
                         (kbt.report_path(s), kbt.final_path(s))}
    if set(g["files"]) != want:
        return ["решение B: набор файлов ≠ ожидаемому"]
    reps = {s: json.load(open(kbt.report_path(s), encoding="utf-8"))
            for s in pc.parse_seeds()}
    st, flags, _d = rule_b(Y, reps)
    if g.get("status") != st or g.get("flags") != flags:
        return [f"статус решения B ≠ пересчёту ({st})"]
    return [] if st == "passed_B_candidate" else [f"статус B: {st}"]


def selftest():
    rng = np.random.default_rng(0)
    n = 30

    def table(pt, p32, pr):
        Y = {nm: {s: {} for s in (0, 1)} for nm in ("Tplus", "T32", "R")}
        for t in TASKS:
            u = rng.random(n)
            for s in (0, 1):
                Y["Tplus"][s][t] = u < pt
                Y["T32"][s][t] = u < p32
                Y["R"][s][t] = u < pr
        return Y
    ok = {0: dict(actor_changed=True, accepted_added=3),
          1: dict(actor_changed=True, accepted_added=1)}
    st, f, d = rule_b(table(0.9, 0.7, 0.7), ok)
    assert st == "passed_B_candidate", (f, d)
    st, f, d = rule_b(table(0.7, 0.7, 0.7), ok)
    assert st == "stopped_B_no_gain" and not f["joint"]
    # ровно +6 чистых успехов из 180 — порог проходит (6/180 > 0.03)
    Y = table(0.7, 0.7, 0.7)
    for s in (0, 1):
        Y["Tplus"][s][0] = Y["T32"][s][0].copy()
        idx = np.flatnonzero(~Y["T32"][s][0])[:3]
        Y["Tplus"][s][0][idx] = True
        for t in (6, 8):
            Y["Tplus"][s][t] = Y["T32"][s][t].copy()
    st, f, d = rule_b(Y, ok)
    assert abs(d["joint_delta"] - 6 / 180) < 1e-12 and f["joint"]
    # на одном сиде Tplus отстаёт от R на 3 успеха — стоп
    Y = table(0.9, 0.7, 0.7)
    for t in TASKS:
        Y["Tplus"][1][t] = Y["R"][1][t].copy()
    Y["Tplus"][1][0][np.flatnonzero(Y["R"][1][0])[:3]] = False
    st, f, _d = rule_b(Y, ok)
    assert not f["lag_R"] and st == "stopped_B_no_gain"
    # actor не изменился / нет принятых обновлений — стоп
    st, f, _d = rule_b(table(0.9, 0.7, 0.7), {0: ok[0], 1: dict(
        actor_changed=False, accepted_added=0)})
    assert not f["actor_updated"] and st == "stopped_B_no_gain"
    assert kh.STAGE_SCOPE[STAGE] == "screen_b"
    print("самопроверка k36b_screen пройдена: правило B (joint ≥ 0.03, "
          "6 чистых успехов из 180 проходят, отставание по сиду > 2 — стоп, "
          "неизменённый actor — стоп)")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-36 B: screen")
    ap.add_argument("cmd", choices=("screen-b", "gate-b"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    try:
        if a.cmd == "gate-b":
            p = gate_b_problems()
            print("  решение B: " + ("passed_B_candidate, привязки сходятся"
                                     if not p else "; ".join(p)))
            return 0 if not p else 4
        return screen_b(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
