#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16: план пилота и проверки блоков.

ПЛАН фиксирует: шесть политик, задачи 0–9, блок 25 (состояния 25–29;
разработочные старты, резерв 45–49 закрыт), контракт раскатки, контракт
базиса и тождества, чекпойнт голов (SHA, единицы, α, предел), разметку и
отчёт обучения, код, таблицу доноров для goal_band_hist_time и эталоны
K-15h (q0ref блока 25 — воспроизведение; static_uniform блока 25 —
исторический ориентир).

ДОНОРЫ (момент для контроля времени): для состояния s ∈ 25..29 задачи t —
вызов первого события q0ref K-15h на состоянии 30 + (s − 25) той же задачи
(блок 30), при отсутствии события — 35 + (s − 25) (блок 35). Используется
только номер вызова события донора; исходы доноров и время события самого
проверяемого состояния не используются.

ПРОВЕРКА БЛОКА (check_block):
  * строгая validate_block и контракт; метка, план, чекпойнт и доноры руки;
  * журнал: нумерация вызовов; не больше одного импульса на среду; q0 — ни
    одного; события пересчитаны той же причинной функцией по журналу a0 и
    исполненных команд и совпадают с записанными; event — импульс ровно в
    первом событии; time — ровно в вызове донора (если вызов был);
  * вне импульса действие ПОБИТОВО = a0; схват = a0; в импульсе
    действие − a0 = записанная δ; норма после предела не выше предела;
  * исполненные actions = журнал руки до done_step;
  * против q0 того же блока: старты, префикс до импульса, h18 и a0 вызовов
    до импульса, у event-политик вызов импульса = первому событию q0;
  * q0 против q0ref K-15h того же блока: хеши действий, исходы, done_step.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

KIND = "k16_plan"
PLAN = "reports/k16/plan.json"
BLOCK = 25
TASKS = list(range(10))
SMOKE_TASKS = [0, 8]
H = 8
CODE = ("k16_policy.py", "k16_harness.py", "k16_protocol.py",
        "k16_models.py", "k16_analyze.py", "k16_rollout.sh",
        "k9h_multiarm_gate.py", "k15f_policy.py",
        "k15f_continuous_refine.py", "k15f_check_identity.py",
        "k15g_local_probe.py", "k15h_protocol.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]


def reached(done_step, call):
    return bool(done_step < 0 or done_step >= H * (call - 1))


def k15h_run_dir():
    c = sorted(glob.glob("reports/k15h/run/p*/"))
    if len(c) != 1:
        raise SystemExit(f"каталог K-15h неоднозначен: {c}")
    return c[0]


def donor_table(run_dir):
    """{задача: {состояние: (донор, вызов)}} по q0ref K-15h блоков 30/35."""
    import k16_passport as kp
    out, files = {}, {}
    for t in TASKS:
        ev = {}
        for b in (30, 35):
            f = os.path.join(run_dir, f"q0ref_t{t}_i{b}.json")
            files[f"t{t}_i{b}"] = f
            d = json.load(open(f))
            z = np.load(os.path.join(run_dir, d["actions_npz"]))
            a0, A = z["k15d_a0"], z["actions"]
            for k, e in enumerate(d["episodes"]):
                ds = int(z["done_step"][k])
                n_exec = ds + 1 if ds >= 0 else len(A)
                c, _ex = kp.first_event_call(-a0[:, k, :, 6], A[:, k, 6],
                                             a0.shape[0], n_exec)
                ev[int(e["init_state_id"])] = c
        out[str(t)] = {}
        for i in range(5):
            for donor in (30 + i, 35 + i):
                if ev.get(donor):
                    out[str(t)][str(BLOCK + i)] = [donor, int(ev[donor])]
                    break
            else:
                raise SystemExit(f"задача {t}: нет донора для {BLOCK + i}")
    return out, files


def make_plan(*, models, train_report, basis, identity, device):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import torch
    ck = torch.load(models, map_location="cpu", weights_only=False)
    run15 = k15h_run_dir()
    donors, dfiles = donor_table(run15)
    refs = {f"q0_t{t}": os.path.join(run15, f"q0ref_t{t}_i{BLOCK}.json")
            for t in TASKS}
    refs.update({f"static_t{t}": os.path.join(
        run15, f"static_uniform_t{t}_i{BLOCK}.json") for t in TASKS})
    refs.update({f"donor_{k}": v for k, v in dfiles.items()})
    for v in refs.values():
        if not os.path.exists(v):
            raise SystemExit(f"нет эталона {v}")
    rep = json.load(open(train_report))
    if rep["models"]["sha1"] != hashlib.sha1(open(models, "rb").read()
                                             ).hexdigest():
        raise SystemExit("отчёт обучения не от этого чекпойнта")
    inputs = dict(models=models, train_report=train_report, basis=basis,
                  identity_report=identity, labels=ck["provenance"]["labels"],
                  harness_utils=hpc.HARNESS_UTILS)
    import k16_policy as kpol
    return dict(
        kind=KIND, created=datetime.datetime.now().isoformat(
            timespec="seconds"),
        labels=list(kpol.LABELS), tasks=TASKS, block=BLOCK, n_envs=5,
        smoke_tasks=SMOKE_TASKS, device=str(device),
        setup=pr.expected_setup(),
        expected_contract=expected_contract(basis, identity, models),
        model_rules=dict(alpha=ck["alpha"], clip_env_norm=ck["clip_env_norm"],
                         units=ck.get("units"), n_low=ck["n_low"]),
        donors=donors,
        donor_rule=("вызов первого события q0ref K-15h на состоянии "
                    "30 + (s − 25), иначе 35 + (s − 25); исходы доноров не "
                    "используются"),
        inputs={k: os.path.abspath(v) for k, v in inputs.items()},
        inputs_sha1={k: sha(v) for k, v in inputs.items()},
        refs={k: os.path.abspath(v) for k, v in refs.items()},
        refs_sha1={k: sha(v) for k, v in refs.items()},
        code={f: sha(os.path.join(HERE, f)) for f in CODE},
        note=("разработочный пилот на уже использованных стартах 25–29; "
              "резерв 45–49 закрыт"))


def expected_contract(basis, identity, models):
    """Контракт руки K-16: базис и тождество (как K-15h), без полей
    K-15h-руки; адаптер, рука, модели K-16 и чекпойнт голов."""
    import k15h_protocol as hpc
    c = hpc.expected_contract(basis, identity)
    c.pop("k15h_policy_sha1", None)
    c.update(harness_adapter_sha1=sha(os.path.join(HERE, "k16_harness.py")),
             k16_policy_sha1=sha(os.path.join(HERE, "k16_policy.py")),
             k16_models_sha1=sha(os.path.join(HERE, "k16_models.py")),
             models_sha1=sha(models))
    return c


def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план K-16"]
    for grp in ("inputs", "refs"):
        for k, f in plan[grp].items():
            if not os.path.exists(f) or sha(f) != plan[f"{grp}_sha1"][k]:
                p.append(f"{grp} {k} изменён или отсутствует ({f})")
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    return p


# --- проверки блока ------------------------------------------------------------
def load(path):
    d = json.load(open(path))
    z = np.load(os.path.join(os.path.dirname(path), d["actions_npz"]),
                allow_pickle=True)
    return d, {k: z[k] for k in z.files}


def replay_events(a0, lv):
    """События по журналу той же причинной функцией: (события [C, B],
    первое событие [B])."""
    import k16_policy as kpol
    C, B = a0.shape[:2]
    tr = kpol.Tracker(B)
    ev = np.zeros((C, B), bool)
    for c in range(C):
        e, _t, _h, _m = tr.step(np.zeros((B, 8)), -a0[c, :, :, 6], None,
                                [-1] * B)
        ev[c] = e
        tr.after(np.zeros((B, 8)), -lv[c, :, :, 6])
    return ev, np.asarray(tr.first_event)


def check_arrays(label, z, plan, donors):
    import k16_policy as kpol
    p = []
    need = ("k15d_a0", f"k15d_level_{label}", "k15g_chunk", "k16_impulse",
            "k16_event", "k16_first_event", "k16_d_codec",
            "k16_norm_applied", "k16_state", "k16_h18")
    miss = [k for k in need if k not in z]
    if miss:
        return [f"{label}: в журнале нет {miss}"]
    a0, lv = z["k15d_a0"], z[f"k15d_level_{label}"]
    imp = np.asarray(z["k16_impulse"], bool)
    C = len(z["k15g_chunk"])
    if z["k15g_chunk"].tolist() != list(range(1, C + 1)):
        p.append(f"{label}: нумерация вызовов нарушена")
    if (imp.sum(0) > 1).any():
        p.append(f"{label}: больше одного импульса на среду")
    sched = kpol.POLICIES[label][3]
    if sched is None and imp.any():
        p.append(f"{label}: у q0 есть импульс")
    ev, fe = replay_events(a0, lv)
    if not np.array_equal(ev, z["k16_event"]) or not np.array_equal(
            fe, z["k16_first_event"]):
        p.append(f"{label}: события не совпали с пересчётом по журналу")
    for b in range(imp.shape[1]):
        calls = (np.nonzero(imp[:, b])[0] + 1).tolist()
        if sched == "event":
            want = [int(fe[b])] if fe[b] > 0 else []
            if calls != want:
                p.append(f"{label}: среда {b}: импульс {calls}, первое "
                         f"событие {fe[b]}")
        elif sched == "time":
            want = [donors[b]] if donors[b] <= C else []
            if calls != want:
                p.append(f"{label}: среда {b}: импульс {calls}, донор "
                         f"{donors[b]}")
    if (~imp).any() and not np.array_equal(lv[~imp], a0[~imp]):
        p.append(f"{label}: вне импульса действие не равно a0")
    if not np.array_equal(lv[..., 6], a0[..., 6]):
        p.append(f"{label}: схват изменён")
    if imp.any():
        d = lv[imp][..., :6].astype(np.float64) - a0[imp][..., :6]
        if not np.allclose(d, z["k16_d_codec"][imp], atol=2e-6):
            p.append(f"{label}: действие в импульсе ≠ a0 + δ журнала")
        clip = float(plan["model_rules"]["clip_env_norm"])
        if (z["k16_norm_applied"][imp] > clip * (1 + 1e-6)).any():
            p.append(f"{label}: норма поправки выше предела")
    return p


def check_vs_q0(label, d, z, rd, rz):
    """Против q0 того же блока: старты, префикс до импульса, входы до
    импульса, у event — вызов импульса = первое событие q0."""
    import k16_policy as kpol
    p = []
    for i, (e, r) in enumerate(zip(d["episodes"], rd["episodes"])):
        for f in ("init_state_id", "init_hash", "init_hash_full",
                  "rollout_seed"):
            if str(e.get(f)) != str(r.get(f)):
                p.append(f"{label}: среда {i}: {f} не равен q0")
    imp = np.asarray(z["k16_impulse"], bool)
    A, Ar = z["actions"], rz["actions"]
    dr = np.asarray(rz["done_step"])
    for b in range(A.shape[1]):
        c_imp = np.nonzero(imp[:, b])[0]
        c_imp = int(c_imp[0]) + 1 if len(c_imp) else None
        end = (H * (c_imp - 1) if c_imp else None)
        if end is None:
            end = int(dr[b]) + 1 if int(dr[b]) >= 0 else len(Ar)
            if not np.array_equal(A[:end, b], Ar[:end, b]) or \
                    d["episodes"][b].get("success") != \
                    rd["episodes"][b].get("success"):
                p.append(f"{label}: среда {b} без импульса разошлась с q0")
            continue
        if not np.array_equal(A[:end, b], Ar[:end, b]):
            p.append(f"{label}: префикс среды {b} до импульса не равен q0")
        for c in range(c_imp):
            if c >= len(rz["k16_h18"]) or not reached(int(dr[b]), c + 1):
                break
            if not (np.array_equal(z["k15d_a0"][c, b], rz["k15d_a0"][c, b])
                    and np.array_equal(z["k16_state"][c, b],
                                       rz["k16_state"][c, b])):
                p.append(f"{label}: входы вызова {c + 1} среды {b} не равны "
                         f"q0")
                break
        if kpol.POLICIES[label][3] == "event" and \
                c_imp != int(rz["k16_first_event"][b]):
            p.append(f"{label}: среда {b}: импульс в {c_imp}, первое событие "
                     f"q0 {int(rz['k16_first_event'][b])}")
    return p


def check_q0_vs_k15h(d, ref_path):
    rd = json.load(open(ref_path))
    p = []
    for f in ("action_sha1", "success", "done_step", "init_hash"):
        if [str(e.get(f)) for e in d["episodes"]] != \
                [str(e.get(f)) for e in rd["episodes"]]:
            p.append(f"q0: {f} не равен q0ref K-15h")
    return p


def check_block(plan, plan_path, path, label, task, ref_q0=None):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    p = pr.validate_block(path, label, int(task), plan["block"],
                          require_single=True, setup=plan["setup"])
    if p:
        return p
    d, z = load(path)
    j = d.get("joint") or {}
    for k, v in plan["expected_contract"].items():
        if json.dumps(j.get(k), sort_keys=True) != json.dumps(
                v, sort_keys=True):
            p.append(f"{label}: контракт {k} = {j.get(k)!r}, план {v!r}")
    donors = [plan["donors"][str(task)][str(plan["block"] + i)][1]
              for i in range(plan["n_envs"])]
    import k16_policy as kpol
    want_d = donors if kpol.POLICIES[label][3] == "time" else None
    if j.get("label") != label or j.get("plan_sha1") != sha(plan_path) or \
            j.get("donor_calls") != want_d:
        p.append(f"{label}: метка, план или доноры руки не из плана")
    p += check_arrays(label, z, plan, donors)
    if not p:
        p += hpc.actions_problems(label, z, hpc.harness_scale())
    if label == "q0":
        p += check_q0_vs_k15h(d, plan["refs"][f"q0_t{task}"])
    elif not p:
        if not ref_q0 or not os.path.exists(ref_q0):
            return p + [f"{label}: нет блока q0"]
        rd, rz = load(ref_q0)
        p += check_vs_q0(label, d, z, rd, rz)
    return p


def check_run(plan, plan_path, run_dir, tasks=None):
    ok, gaps = {}, []
    for t in (tasks if tasks is not None else plan["tasks"]):
        ref = os.path.join(run_dir, f"q0_t{t}_i{plan['block']}.json")
        for lab in plan["labels"]:
            pth = os.path.join(run_dir, f"{lab}_t{t}_i{plan['block']}.json")
            if not os.path.exists(pth):
                gaps.append(dict(label=lab, task=t, why="не готов"))
                continue
            if lab != "q0" and ("q0", t) not in ok:
                gaps.append(dict(label=lab, task=t, why="нет проверенного q0"))
                continue
            prob = check_block(plan, plan_path, pth, lab, t, ref)
            if prob:
                gaps.append(dict(label=lab, task=t, why="; ".join(prob[:2])))
            else:
                ok[(lab, t)] = pth
    return ok, gaps


# --- самопроверка ------------------------------------------------------------
def selftest():
    import k16_models as km
    rng = np.random.default_rng(0)
    C, B = 6, 3
    a0 = rng.uniform(-0.5, 0.5, (C, B, H, 7)).astype(np.float32)
    a0[..., 6] = 1.0                              # исполнение: −1 (открыт)
    a0[3, 0, 4:, 6] = -1.0                        # среда 0: событие в вызове 4
    a0[4:, 0, :, 6] = -1.0
    plan = dict(model_rules=dict(clip_env_norm=0.5))

    def journal(label, imp_calls, donors=(9, 9, 9)):
        lv = a0.copy()
        imp = np.zeros((C, B), bool)
        dc = np.full((C, B, H, 6), np.nan, np.float32)
        napp = np.full((C, B), np.nan)
        for b, c in imp_calls.items():
            imp[c - 1, b] = True
            _a, inf = km.apply_correction(a0[c - 1, b:b + 1], np.ones((1, H,
                                                                     6)),
                                          np.full(6, .05), np.ones(7), .5,
                                          .5)
            lv[c - 1, b] = _a[0]
            dc[c - 1, b] = inf["d_codec"][0]
            napp[c - 1, b] = inf["norm_applied"][0]
        ev, fe = replay_events(a0, lv)
        return {"k15d_a0": a0, f"k15d_level_{label}": lv,
                "k15g_chunk": np.arange(1, C + 1), "k16_impulse": imp,
                "k16_event": ev, "k16_first_event": fe, "k16_d_codec": dc,
                "k16_norm_applied": napp, "k16_state": np.zeros((C, B, 8)),
                "k16_h18": np.zeros((C, B, 2, 2))}
    z = journal("goal_band_hist_event", {0: 4})
    assert z["k16_first_event"].tolist() == [4, -1, -1]
    assert check_arrays("goal_band_hist_event", z, plan, [9] * 3) == []
    assert check_arrays("q0", journal("q0", {}), plan, [9] * 3) == []
    # импульс не в событии; два импульса; импульс у q0; не тот донор
    assert check_arrays("goal_band_hist_event", journal(
        "goal_band_hist_event", {0: 3}), plan, [9] * 3)
    zz = journal("goal_band_hist_event", {0: 4})
    zz["k16_impulse"][5, 0] = True
    assert any("больше одного" in x for x in check_arrays(
        "goal_band_hist_event", zz, plan, [9] * 3))
    assert check_arrays("q0", journal("q0", {1: 2}), plan, [9] * 3)
    zt = journal("goal_band_hist_time", {0: 2, 1: 2, 2: 2})
    assert check_arrays("goal_band_hist_time", zt, plan, [2, 2, 2]) == []
    assert check_arrays("goal_band_hist_time", zt, plan, [2, 3, 2])
    # донор позже последнего вызова — импульса нет
    zn = journal("goal_band_hist_time", {0: 2, 1: 2})
    assert check_arrays("goal_band_hist_time", zn, plan, [2, 2, 99]) == []
    # схват изменён / δ не совпала / записанные события подменены
    zg = journal("goal_band_hist_event", {0: 4})
    zg["k15d_level_goal_band_hist_event"] = \
        zg["k15d_level_goal_band_hist_event"].copy()
    zg["k15d_level_goal_band_hist_event"][3, 0, 0, 6] = 0.3
    assert check_arrays("goal_band_hist_event", zg, plan, [9] * 3)
    zd = journal("goal_band_hist_event", {0: 4})
    zd["k16_d_codec"] = zd["k16_d_codec"] * 2
    assert any("δ" in x for x in check_arrays("goal_band_hist_event", zd,
                                              plan, [9] * 3))
    ze = journal("goal_band_hist_event", {0: 4})
    ze["k16_event"] = ze["k16_event"].copy()
    ze["k16_event"][1, 1] = True
    assert any("события" in x for x in check_arrays(
        "goal_band_hist_event", ze, plan, [9] * 3))
    print("самопроверка k16_protocol пройдена: событие по журналу, один "
          "импульс, q0 без импульса, донор, схват, δ, подмена событий")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-16: протокол")
    ap.add_argument("mode", choices=("plan", "check-plan", "check-block",
                                     "smoke-check", "selftest"))
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--models", default="data/k16/models.pt")
    ap.add_argument("--train-report", default="reports/k16/train.json")
    ap.add_argument("--basis", default="data/k15f/basis_s0.pt")
    ap.add_argument("--identity", default="reports/k15f/identity_cuda1.json")
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
            plan = make_plan(models=a.models, train_report=a.train_report,
                             basis=a.basis, identity=a.identity,
                             device=a.device)
            os.makedirs(os.path.dirname(a.plan), exist_ok=True)
            with open(a.plan + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=1)
            os.replace(a.plan + ".tmp", a.plan)
            print(f"  план {a.plan}: политик {len(plan['labels'])}, задач "
                  f"{len(plan['tasks'])}, блок {plan['block']}, процессов "
                  f"{len(plan['labels']) * len(plan['tasks'])}")
            for t, dd in plan["donors"].items():
                print(f"    доноры задачи {t}: " + " ".join(
                    f"{s}←{v[0]}:в{v[1]}" for s, v in dd.items()))
            return 0
        plan = json.load(open(a.plan))
        if a.mode == "check-plan":
            prob = check_plan_inputs(plan)
            print("\n".join(prob) if prob else "  входы плана сверены")
            return 3 if prob else 0
        if a.mode == "check-block":
            path, lab, t = a.block
            ref = os.path.join(os.path.dirname(path),
                               f"q0_t{t}_i{plan['block']}.json")
            prob = check_block(plan, a.plan, path, lab, t, ref)
            if prob:
                print("; ".join(prob[:6]))
            return 2 if prob else 0
        ok, gaps = check_run(plan, a.plan, a.run_dir, plan["smoke_tasks"])
        print(f"  smoke: проверено блоков {len(ok)} из "
              f"{len(plan['smoke_tasks']) * len(plan['labels'])}; проблем "
              f"{len(gaps)}" + (f": {gaps[:6]}" if gaps else ""))
        return 3 if gaps else 0
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
