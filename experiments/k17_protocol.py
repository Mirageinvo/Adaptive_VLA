#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-17: план и проверки (блок 40 — старты 40–44 — всех задач; q0,
ctx_memory, held_goal).

ПЛАН: метки, задачи 0–9, блок 40 (старты использовались в разработке, в
том числе в K-16u; резерв 45–49 закрыт), контракт раскатки и руки,
чекпойнт K-17 и отчёт обучения, происхождение весов (окна, разметка K-16,
K-16, код обучения), файлы стартов LIBERO и q0 K-16u того же блока
(эталон для сверки q0).

ПРОВЕРКА БЛОКА: строгая validate_block и контракт; журнал K-17: события
по журналу, окно — ровно вызовы [первое событие, +1, +2] (усечённые
концом записи), у q0 окна нет; вне окна действие = a0; схват = a0;
действие в окне = a0 + δ журнала; предел нормы и единственное применение
на чанк (||δ_codec · scale|| = норма журнала); мировая цель (p*, R*)
постоянна внутри окна; геометрия held_goal = ошибка от сохранённой цели
до записанного состояния вызова; у ctx_memory геометрия — нули;
исполненные actions = журнал. q0 — против q0 K-16u (действия, успех,
шаг завершения, старты). Модели — против q0 этого блока до своего первого
вмешательства (старты, префикс действий, a0 и состояние, начало окна =
первое событие q0). Совпадение с q0 после первой поправки не требуется;
h18 после вмешательства соседа — в анализе (сцепление батча).
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

KIND = "k17_plan"
PLAN = "reports/k17/plan.json"
BLOCKS = (40,)
SMOKE = [(0, 40), (8, 40)]
H = 8
CODE = ("k17_policy.py", "k17_harness.py", "k17_protocol.py",
        "k17_models.py", "k17_train.py", "k17_analyze.py", "k17_prepare.py",
        "k17_rollout.sh", "k16_policy.py", "k16_harness.py",
        "k16_protocol.py", "k16_models.py", "k9h_multiarm_gate.py",
        "k15f_policy.py", "k15f_continuous_refine.py",
        "k15f_check_identity.py", "k15g_local_probe.py", "k15h_protocol.py")
K16U_PLAN = "reports/k16u/plan.json"


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]


def full_sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def provenance_problems(prov):
    p = []
    for k in ("windows", "k16_labels", "k16_models"):
        f = prov.get(k)
        if not f or not os.path.exists(f) or full_sha(f) != prov.get(
                k + "_sha1"):
            p.append(f"происхождение: {k} не совпадает с обучением ({f})")
    code = prov.get("code") or {}
    if not code:
        p.append("происхождение: нет отпечатков кода обучения")
    for f, s_ in code.items():
        fp = os.path.join(HERE, f)
        if not os.path.exists(fp) or full_sha(fp) != s_:
            p.append(f"происхождение: код обучения {f} изменён")
    return p


def reserve_problems(inv):
    bad = {t: sorted(int(x) for x in v if 45 <= int(x) <= 49)
           for t, v in inv["used"].items()}
    bad = {t: v for t, v in bad.items() if v}
    return [f"резерв 45–49 использован: {bad}"] if bad else []


def k16u_q0_refs(plan_path=K16U_PLAN):
    run = os.path.join("reports/k16u/run", f"p{sha(plan_path)}")
    refs = {}
    for t in range(10):
        f = os.path.join(run, f"q0_t{t}_i40.json")
        if not os.path.exists(f):
            raise SystemExit(f"нет q0 K-16u: {f}")
        refs[f"q0k16u_t{t}"] = f
    return refs


def make_plan(*, models, train_report, basis, identity, device):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import k17_policy as bp
    import torch
    ck = torch.load(models, map_location="cpu", weights_only=False)
    rep = json.load(open(train_report))
    if ck.get("kind") != "k17_models":
        raise SystemExit("чекпойнт не K-17")
    if rep["models"]["sha1"] != full_sha(models):
        raise SystemExit("отчёт обучения не от этого чекпойнта")
    prob = provenance_problems(ck["provenance"])
    if prob:
        raise SystemExit("; ".join(prob[:4]))
    inv = json.load(open("reports/k15h/inventory.json"))
    refs = {}
    for t, v in inv["libero_bank"].items():
        f = v["path"]
        if not os.path.exists(f) or full_sha(f) != v["sha1"]:
            raise SystemExit(f"файл стартов задачи {t} изменён: {f}")
        refs[f"init_t{t}"] = f
    prob = reserve_problems(inv)
    if prob:
        raise SystemExit(prob[0])
    refs.update(k16u_q0_refs())
    contract = hpc.expected_contract(basis, identity)
    contract.pop("k15h_policy_sha1", None)
    contract.update(
        harness_adapter_sha1=sha(os.path.join(HERE, "k17_harness.py")),
        k17_policy_sha1=sha(os.path.join(HERE, "k17_policy.py")),
        k17_models_sha1=sha(os.path.join(HERE, "k17_models.py")),
        k16_models_sha1=sha(os.path.join(HERE, "k16_models.py")),
        models_sha1=sha(models))
    inputs = dict(models=models, train_report=train_report, basis=basis,
                  identity_report=identity,
                  windows=ck["provenance"]["windows"],
                  k16_labels=ck["provenance"]["k16_labels"],
                  k16_models=ck["provenance"]["k16_models"],
                  harness_utils=hpc.HARNESS_UTILS,
                  inventory="reports/k15h/inventory.json",
                  k16u_plan=K16U_PLAN)
    return dict(
        kind=KIND, created=datetime.datetime.now().isoformat(
            timespec="seconds"),
        labels=list(bp.LABELS), tasks=list(range(10)), blocks=list(BLOCKS),
        n_envs=5, smoke=SMOKE, device=str(device),
        setup=pr.expected_setup(), expected_contract=contract,
        model_rules=dict(alpha=ck["alpha"], clip_env_norm=ck["clip_env_norm"],
                         units=ck.get("units"),
                         scale_arm=list(ck["scale_arm"]),
                         dpos_sigma=np.asarray(ck["norms"]["dpos_sigma"])
                         .tolist(), window_calls=3),
        inputs={k: os.path.abspath(v) for k, v in inputs.items()},
        inputs_sha1={k: sha(v) for k, v in inputs.items()},
        refs={k: os.path.abspath(v) for k, v in refs.items()},
        refs_sha1={k: sha(v) for k, v in refs.items()},
        code={f: sha(os.path.join(HERE, f)) for f in CODE},
        training_provenance=ck["provenance"],
        note=("K-17: сохраняемая цель и три последовательные коррекции; "
              "старты 40–44 использовались в разработке (в том числе "
              "K-16u); резерв 45–49 закрыт"))


def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план K-17"]
    for grp in ("inputs", "refs"):
        for k, f in plan[grp].items():
            if not os.path.exists(f) or sha(f) != plan[f"{grp}_sha1"][k]:
                p.append(f"{grp} {k} изменён или отсутствует ({f})")
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    prov = plan.get("training_provenance")
    if not prov:
        p.append("в плане нет происхождения весов")
    else:
        p += provenance_problems(prov)
    return p


def window_calls(fe, C):
    return [c for c in (fe, fe + 1, fe + 2) if c <= C] if fe > 0 else []


def check_arrays(label, z, rules):
    import k16_models as km
    import k16_protocol as p16
    import k17_models as m17
    p = []
    need = ("k15d_a0", f"k15d_level_{label}", "k15g_chunk", "k17_corrected",
            "k17_event", "k17_first_event", "k17_window_start",
            "k17_win_pos", "k17_d_codec", "k17_norm_applied", "k17_state",
            "k17_p_star", "k17_R_star", "k17_geo", "k17_h18")
    miss = [k for k in need if k not in z]
    if miss:
        return [f"{label}: в журнале нет {miss}"]
    a0, lv = z["k15d_a0"], z[f"k15d_level_{label}"]
    cor = np.asarray(z["k17_corrected"], bool)
    C, B = cor.shape
    if z["k15g_chunk"].tolist() != list(range(1, C + 1)):
        p.append(f"{label}: нумерация вызовов нарушена")
    ev, fe = p16.replay_events(a0, lv)
    if not np.array_equal(ev, z["k17_event"]) or not np.array_equal(
            fe, z["k17_first_event"]):
        p.append(f"{label}: события не совпали с пересчётом по журналу")
    ws = np.asarray(z["k17_window_start"])
    for b in range(B):
        calls = (np.nonzero(cor[:, b])[0] + 1).tolist()
        want = [] if label == "q0" else window_calls(int(fe[b]), C)
        if calls != want:
            p.append(f"{label}: среда {b}: окно {calls}, ожидалось {want}")
            continue
        want_ws = want[0] if want else -1
        if int(ws[b]) != want_ws:
            p.append(f"{label}: среда {b}: начало окна {ws[b]}")
        wp = np.asarray(z["k17_win_pos"])[:, b]
        exp_wp = np.full(C, -1)
        for k, c in enumerate(want):
            exp_wp[c - 1] = k
        if not np.array_equal(wp, exp_wp):
            p.append(f"{label}: среда {b}: позиции окна {wp.tolist()}")
        if want:
            c0 = want[0] - 1
            ps, Rs = z["k17_p_star"][:, b], z["k17_R_star"][:, b]
            if np.isfinite(ps[:c0]).any() or not np.isfinite(ps[c0:]).all() \
                    or not (ps[c0:] == ps[c0]).all() or \
                    not (Rs[c0:] == Rs[c0]).all():
                p.append(f"{label}: среда {b}: мировая цель не сохранена "
                         f"неизменной с начала окна")
            ds = np.asarray(rules["dpos_sigma"], np.float64)
            for c in want:
                st = z["k17_state"][c - 1, b].astype(np.float64)
                g = z["k17_geo"][c - 1, b]
                if label == "ctx_memory":
                    ok = not g.any()
                else:
                    e = m17.geo_error_np(ps[c - 1][None], Rs[c - 1][None],
                                         st[None, :3],
                                         km.aa_to_mat_np(st[None, 3:6]),
                                         ds)[0]
                    ok = np.allclose(g, e, atol=1e-4)
                if not ok:
                    p.append(f"{label}: среда {b}: геометрия вызова {c} не "
                             f"равна ошибке до сохранённой цели")
                    break
    if (~cor).any() and not np.array_equal(lv[~cor], a0[~cor]):
        p.append(f"{label}: вне окна действие не равно a0")
    if not np.array_equal(lv[..., 6], a0[..., 6]):
        p.append(f"{label}: схват изменён")
    if cor.any():
        d = lv[cor][..., :6].astype(np.float64) - a0[cor][..., :6]
        if not np.allclose(d, z["k17_d_codec"][cor], atol=2e-6):
            p.append(f"{label}: действие в окне ≠ a0 + δ журнала")
        clip = float(rules["clip_env_norm"])
        na = z["k17_norm_applied"][cor].astype(np.float64)
        if (na > clip * (1 + 1e-6)).any():
            p.append(f"{label}: норма поправки выше предела")
        sc = np.asarray(rules["scale_arm"], np.float64)[:6]
        n2 = np.linalg.norm((z["k17_d_codec"][cor].astype(np.float64)
                             * sc).reshape(int(cor.sum()), -1), axis=1)
        if not np.allclose(n2, na, rtol=1e-4, atol=1e-7):
            p.append(f"{label}: ограничение применено не один раз на чанк")
    return p


def check_vs_q0(label, d, z, rd, rz):
    """До первого вмешательства среды — как q0; окно — от первого
    события q0."""
    import k16_protocol as p16
    p = []
    for i, (e, r) in enumerate(zip(d["episodes"], rd["episodes"])):
        for f in ("init_state_id", "init_hash", "init_hash_full",
                  "rollout_seed"):
            if str(e.get(f)) != str(r.get(f)):
                p.append(f"{label}: среда {i}: {f} не равен q0")
    cor = np.asarray(z["k17_corrected"], bool)
    A, Ar = z["actions"], rz["actions"]
    dr = np.asarray(rz["done_step"])
    Cq = len(rz["k15d_a0"])
    for b in range(A.shape[1]):
        cs = np.nonzero(cor[:, b])[0]
        c1 = int(cs[0]) + 1 if len(cs) else None
        if c1 is None:
            end = int(dr[b]) + 1 if int(dr[b]) >= 0 else len(Ar)
            if not np.array_equal(A[:end, b], Ar[:end, b]) or \
                    d["episodes"][b].get("success") != \
                    rd["episodes"][b].get("success"):
                p.append(f"{label}: среда {b} без окна разошлась с q0")
            if int(rz["k17_first_event"][b]) > 0 and p16.reached(
                    int(dr[b]), int(rz["k17_first_event"][b])):
                p.append(f"{label}: среда {b}: у q0 событие "
                         f"{int(rz['k17_first_event'][b])}, окна нет")
            continue
        if not np.array_equal(A[:H * (c1 - 1), b], Ar[:H * (c1 - 1), b]):
            p.append(f"{label}: префикс среды {b} до окна не равен q0")
        for c in range(c1):
            if c >= Cq or not p16.reached(int(dr[b]), c + 1):
                break
            if not (np.array_equal(z["k15d_a0"][c, b], rz["k15d_a0"][c, b])
                    and np.array_equal(z["k17_state"][c, b],
                                       rz["k17_state"][c, b])):
                p.append(f"{label}: входы вызова {c + 1} среды {b} не равны "
                         f"q0")
                break
        if c1 != int(rz["k17_first_event"][b]):
            p.append(f"{label}: среда {b}: окно с {c1}, первое событие q0 "
                     f"{int(rz['k17_first_event'][b])}")
    return p


def check_q0_vs_ref(d, ref_path):
    rd = json.load(open(ref_path))
    p = []
    for f in ("action_sha1", "success", "done_step", "init_hash"):
        if [str(e.get(f)) for e in d["episodes"]] != \
                [str(e.get(f)) for e in rd["episodes"]]:
            p.append(f"q0: {f} не равен q0 K-16u того же блока")
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
    want_k = None if label == "q0" else label
    if j.get("model_kind") != want_k:
        p.append(f"{label}: модель {j.get('model_kind')!r}, ожидалась "
                 f"{want_k!r}")
    p += check_arrays(label, z, plan["model_rules"])
    if not p:
        p += hpc.actions_problems(label, z, hpc.harness_scale())
    if p:
        return p
    if label == "q0":
        p += check_q0_vs_ref(d, plan["refs"][f"q0k16u_t{task}"])
    else:
        if not ref_q0 or not os.path.exists(ref_q0):
            return p + [f"{label}: нет блока q0"]
        rd, rz = p16.load(ref_q0)
        p += check_vs_q0(label, d, z, rd, rz)
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


# --- самопроверка ------------------------------------------------------------
def _journal(label, rng, fe_calls, C=6, B=3, clip=0.5):
    """Синтетический журнал руки с окнами от заданных первых событий."""
    import k16_models as km
    import k17_models as m17
    a0 = rng.uniform(-0.5, 0.5, (C, B, H, 7)).astype(np.float32)
    a0[..., 6] = 1.0
    for b, c in fe_calls.items():                 # событие в вызове c
        a0[c - 1, b, 4:, 6] = -1.0
        a0[c:, b, :, 6] = -1.0
    ds = np.full(3, 0.1)
    sc = np.full(6, 0.9)
    lv = a0.copy()
    cor = np.zeros((C, B), bool)
    wp = np.full((C, B), -1)
    ws = np.full(B, -1)
    st = rng.standard_normal((C, B, 8)).astype(np.float32) * 0.1
    nan = lambda *s: np.full(s, np.nan, np.float32)   # noqa: E731
    dc, na = nan(C, B, H, 6), nan(C, B)
    ps, Rs, geo = nan(C, B, 3), nan(C, B, 3, 3), nan(C, B, 9)
    for b, c in fe_calls.items():
        if label == "q0":
            continue
        calls = window_calls(c, C)
        ws[b] = calls[0]
        p_ = rng.standard_normal(3).astype(np.float32)
        R_ = km.aa_to_mat_np(rng.standard_normal(3) * .3).astype(np.float32)
        for k, cc in enumerate(calls):
            cor[cc - 1, b] = True
            wp[cc - 1, b] = k
            ps[cc - 1:, b], Rs[cc - 1:, b] = p_, R_
            s_ = st[cc - 1, b].astype(np.float64)
            geo[cc - 1, b] = (np.zeros(9) if label == "ctx_memory" else
                              m17.geo_error_np(p_[None], R_[None],
                                               s_[None, :3],
                                               km.aa_to_mat_np(
                                                   s_[None, 3:6]), ds)[0])
            _a, inf = km.apply_correction(a0[cc - 1, b:b + 1],
                                          np.ones((1, H, 6)),
                                          np.full(6, .05),
                                          np.r_[sc, 1.0], .5, clip)
            lv[cc - 1, b] = _a[0]
            dc[cc - 1, b] = inf["d_codec"][0]
            na[cc - 1, b] = inf["norm_applied"][0]
    import k16_protocol as p16
    ev, fe = p16.replay_events(a0, lv)
    z = {"k15d_a0": a0, f"k15d_level_{label}": lv,
         "k15g_chunk": np.arange(1, C + 1), "k17_corrected": cor,
         "k17_event": ev, "k17_first_event": fe, "k17_window_start": ws,
         "k17_win_pos": wp, "k17_d_codec": dc, "k17_norm_applied": na,
         "k17_state": st, "k17_p_star": ps, "k17_R_star": Rs,
         "k17_geo": geo, "k17_h18": np.zeros((C, B, 1), np.float16)}
    rules = dict(clip_env_norm=clip, scale_arm=list(sc), dpos_sigma=list(ds))
    return z, rules


def selftest():
    import k17_policy as kp17
    assert set(kp17.LABELS) == {"q0", "ctx_memory", "held_goal"}
    assert window_calls(5, 6) == [5, 6] and window_calls(-1, 6) == []
    rng = np.random.default_rng(0)
    for lab in ("q0", "ctx_memory", "held_goal"):
        z, rules = _journal(lab, rng, {0: 2, 1: 5})
        assert check_arrays(lab, z, rules) == [], (lab, check_arrays(
            lab, z, rules))
    z, rules = _journal("held_goal", rng, {0: 2, 1: 5})
    # нарушения: окно короче, сдвинутая цель, геометрия, двойной предел,
    # поправка вне окна, ненулевая геометрия у ctx_memory
    muts = []
    z1 = dict(z, k17_corrected=z["k17_corrected"].copy())
    z1["k17_corrected"][3, 0] = False
    muts.append(z1)
    z2 = dict(z, k17_p_star=z["k17_p_star"].copy())
    z2["k17_p_star"][3, 0] += 0.01
    muts.append(z2)
    z3 = dict(z, k17_geo=z["k17_geo"].copy())
    z3["k17_geo"][2, 0] += 0.01
    muts.append(z3)
    z4 = dict(z, k17_norm_applied=z["k17_norm_applied"] * 0.5)
    muts.append(z4)
    lv = z["k15d_level_held_goal"].copy()
    lv[0, 2, :, :6] += 0.01
    muts.append(dict(z, k15d_level_held_goal=lv))
    for i, m in enumerate(muts):
        assert check_arrays("held_goal", m, rules), i
    zc, rc = _journal("ctx_memory", rng, {0: 2})
    zc["k17_geo"] = np.where(np.isnan(zc["k17_geo"]), np.nan, 0.5)
    assert check_arrays("ctx_memory", zc, rc)
    # у q0 нет окна
    zq, rq = _journal("held_goal", rng, {0: 2})
    zq["k15d_level_q0"] = zq.pop("k15d_level_held_goal")
    assert check_arrays("q0", zq, rq)
    # резерв и происхождение
    inv = dict(used={str(t): list(range(45)) for t in range(10)})
    assert reserve_problems(inv) == []
    inv["used"]["7"] = list(range(46))
    assert reserve_problems(inv)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        fs = {k: os.path.join(td, k) for k in ("w", "l", "m")}
        for f in fs.values():
            open(f, "w").write(f)
        f0 = "k17_models.py"
        good = dict(windows=fs["w"], windows_sha1=full_sha(fs["w"]),
                    k16_labels=fs["l"], k16_labels_sha1=full_sha(fs["l"]),
                    k16_models=fs["m"], k16_models_sha1=full_sha(fs["m"]),
                    code={f0: full_sha(os.path.join(HERE, f0))})
        assert provenance_problems(good) == []
        for mut in (dict(windows_sha1="0"), dict(k16_labels_sha1="0"),
                    dict(k16_models_sha1="0"), dict(code={f0: "0"}),
                    dict(code={})):
            assert provenance_problems(dict(good, **mut)), mut
    print("самопроверка k17_protocol пройдена: окно из трёх вызовов от "
          "события, сохранённая цель, геометрия, нули ctx_memory, предел "
          "один раз, вне окна = a0, у q0 окна нет, резерв, происхождение")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-17: протокол")
    ap.add_argument("mode", choices=("plan", "check-plan", "check-block",
                                     "smoke-check", "selftest"))
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--models", default="data/k17/models.pt")
    ap.add_argument("--train-report", default="reports/k17/train.json")
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
