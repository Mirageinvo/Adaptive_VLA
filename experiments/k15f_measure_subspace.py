#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15f M1: перепись q0, выбор кластеров и поведенческий замер базиса.

РЕЖИМЫ:
  smoke   технический: руки q0, z, l0p, r0p на задачах 0 и 8, состояния
          0-4. Обязательное тождество z = q0 на каждом эпизоде.
  census  перепись q0 на design-состояниях 0-9 всех 10 задач (100
          кластеров). Фиксирует ДО раскаток кандидатов:
            F — кластеры, проваленные q0;
            диагностические успехи — два наименьших успешных состояния
            каждой задачи (для harm и диапазона);
            блоки по 5 состояний, которые нужно раскатать.
          Сводка переписи пишется один раз; перезапись — только явно.
  m1      16 рук (l/r × 4 направления × ±) на блоках из переписи.
          Первичное (зарегистрировано до роллаутов, 08.10.2026):
            R_L — провалы q0, спасённые хотя бы одной learned-рукой,
            R_C — то же для контроля; парно: L-only и C-only.
            мощность:     F >= 12, иначе решения нет;
            перспективно: R_L >= ceil(F/3) И R_L − R_C >= 3;
            закрыть:      R_L <= 2 ИЛИ R_L <= R_C;
            иначе неясно (агент останавливается, без смены амплитуды,
            контроля и порогов).
          Это инженерный скрининг, а не доказательство улучшения SR.

ГЕЙТЫ (fail-closed, код 3): сид 101 в заголовке и каждом эпизоде, хеши
действий формата k9h, связь npz с JSON, метка исполняет свои c и
контроль, один контракт базиса у всех рук k15f, каждый эпизод кандидата
парен эпизоду q0 переписи (те же init_hash/init_hash_full/rollout_seed),
полное покрытие выбранных кластеров.
"""
import argparse
import datetime
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k15e_measure_scale as ms  # noqa: E402
import k15f_policy as kp  # noqa: E402
from k15d_behavior import check_actions_npz  # noqa: E402

SEEDS = [101]
DESIGN_STATES = list(range(10))
TASKS = list(range(10))
DIAG_PER_TASK = 2
RULE = dict(min_F=12, promising_frac=1.0 / 3.0, promising_margin=3,
            close_RL=2, grip_flip_ablation=0.10,
            registered="08.10.2026, до роллаутов M1")
# ЕСЛИ M1 ПЕРСПЕКТИВЕН, а хоть одно направление меняет решения схвата у
# доли > grip_flip_ablation исполняемых шагов (отчёт предобучения), перед
# архитектурным выводом ОБЯЗАТЕЛЬНА абляция arm-only против gripper-only.
# Контракт раскатки, общий для q0 и всех кандидатов:
SETUP = ("script_sha1", "suite", "horizon", "max_steps", "waiting_steps",
         "ensemble", "rollout_seed_mode", "ckpt", "seed", "n_envs")
# ОЖИДАЕМЫЙ контракт (зарегистрирован): одинаковость значений мало —
# согласованно неверное или отсутствующее у всех поле прошло бы.
# script_sha1 — отпечаток ТЕКУЩЕГО k9h_multiarm_gate.py, подставляется при
# анализе.
SETUP_EXPECTED = dict(suite="10", horizon=8, max_steps=600, waiting_steps=10,
                      ensemble="off", rollout_seed_mode="fixed", seed=101,
                      n_envs=5,
                      ckpt="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
CENSUS_FIELDS = ("kind", "code", "F", "fails", "diag", "blocks", "powered",
                 "rule", "q0_files")
CONTRACT = ("basis_sha1", "basis_state_sha1", "stats_sha1", "amp_factor",
            "control_seed", "identity_report_sha1", "refine_module_sha1",
            "k15f_policy_sha1", "frozen_content_sha", "joint_sha1",
            "code_version", "codec", "k15a_gate")
START = ("init_hash", "init_hash_full", "rollout_seed")


def load_arts(paths):
    """{метка: [(путь, артефакт)]} с базовыми проверками."""
    out = {}
    for p in paths:
        d = json.load(open(p))
        out.setdefault(str(d["arm_label"]), []).append((p, d))
    return out


def episodes(arts):
    """{(задача, состояние, сид): эпизод} с проверкой повторов."""
    ep, dup = {}, []
    for p, d in arts:
        for e in d["episodes"]:
            k = (int(d["task_id"]), int(e["init_state_id"]), int(d["seed"]))
            if k in ep:
                dup.append(k)
            ep[k] = e
    return ep, dup


def check_q0(arts):
    bad = []
    for p, d in arts:
        bad += [f"{os.path.basename(p)}: {x}" for x in ms.check_arm(d, "q0")]
        if str(d.get("suite")) != "10":
            bad.append(f"{os.path.basename(p)}: не LIBERO-10")
    return bad


def check_k15f(arts, label):
    c, control = kp.label_spec(label)
    bad, contracts = [], []
    for p, d in arts:
        j = d.get("joint") or {}
        if d.get("policy") != "k15f" or j.get("label") != label or \
                j.get("coeffs") != c or bool(j.get("control")) != control \
                or j.get("layers_per_call") != 18:
            bad.append(f"{os.path.basename(p)}: метка {label} исполняет не "
                       f"своё (c={j.get('coeffs')}, контроль "
                       f"{j.get('control')})")
        if str(d.get("suite")) != "10":
            bad.append(f"{os.path.basename(p)}: не LIBERO-10")
        contracts.append({k: j.get(k) for k in CONTRACT})
    return bad, contracts


def technical_common(by_label):
    bad = []
    files = [p for lst in by_label.values() for p, _d in lst]
    bad += check_actions_npz(files)
    bad += ms.check_episodes(files, seeds=SEEDS)
    return bad


def pair_check(q0_ep, arm_ep, keys, label):
    """Кандидат покрывает выбранные кластеры теми же стартами, что q0."""
    bad = []
    for k in keys:
        if k not in arm_ep:
            bad.append(f"{label}: нет эпизода {k}")
            continue
        if any(str(arm_ep[k].get(f)) != str(q0_ep[k].get(f)) for f in START):
            bad.append(f"{label}: эпизод {k} — другой старт")
    return bad


def census(q0_arts, expected_setup=None):
    """Провалы q0, диагностические успехи и нужные блоки."""
    technical = check_q0(q0_arts) + technical_common({"q0": q0_arts})
    technical += check_setup({"q0": q0_arts}, expected_setup)
    ep, dup = episodes(q0_arts)
    if dup:
        technical.append(f"повторяющиеся эпизоды q0: {dup[:3]}")
    want = [(t, s, SEEDS[0]) for t in TASKS for s in DESIGN_STATES]
    miss = [k for k in want if k not in ep]
    if miss:
        technical.append(f"перепись неполна: нет {len(miss)} кластеров, "
                         f"например {miss[:3]}")
    fails, diag = [], []
    for t in TASKS:
        succ = [s for s in DESIGN_STATES if (t, s, SEEDS[0]) in ep
                and ep[(t, s, SEEDS[0])]["success"]]
        fails += [[t, s] for s in DESIGN_STATES
                  if (t, s, SEEDS[0]) in ep
                  and not ep[(t, s, SEEDS[0])]["success"]]
        diag += [[t, s] for s in sorted(succ)[:DIAG_PER_TASK]]
    sel = sorted(set(map(tuple, fails)) | set(map(tuple, diag)))
    blocks = {}
    for t, s in sel:
        blocks.setdefault(int(t), set()).add(5 * (int(s) // 5))
    F = len(fails)
    return dict(kind="k15f_m1_census", technical=technical,
                code=3 if technical else 0, F=F, fails=fails, diag=diag,
                blocks={str(t): sorted(v) for t, v in sorted(blocks.items())},
                powered=F >= RULE["min_F"], rule=RULE,
                q0_files={os.path.basename(p): _sha(p) for p, _d in q0_arts},
                per_task_sr={str(t): float(np.mean(
                    [ep[(t, s, SEEDS[0])]["success"] for s in DESIGN_STATES
                     if (t, s, SEEDS[0]) in ep])) for t in TASKS},
                created=datetime.datetime.now().isoformat(timespec="seconds"))


def _sha(p):
    import hashlib
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]


def decide(F, RL, RC, rule=RULE):
    if F < rule["min_F"]:
        return "недостаточная мощность: решения нет"
    if RL <= rule["close_RL"] or RL <= RC:
        return "закрыть данный rank-4 imitation-initialized базис"
    if RL >= math.ceil(F * rule["promising_frac"]) and \
            RL - RC >= rule["promising_margin"]:
        return "перспективно: этап B (outcome-датасет, критик, actor)"
    return "неясно: агент останавливается"


def check_setup(by_label, expected=None):
    """Один контракт раскатки у всех артефактов И он равен ожидаемому."""
    bad = []
    if expected is not None:
        for lab, arts in by_label.items():
            for p, d in arts:
                for k, v in expected.items():
                    if d.get(k) != v:
                        bad.append(f"{lab}:{os.path.basename(p)}: {k} = "
                                   f"{d.get(k)!r}, ожидалось {v!r}")
        if bad:
            return bad[:10]
    seen = {}
    for lab, arts in by_label.items():
        for p, d in arts:
            key = json.dumps({k: d.get(k) for k in SETUP}, sort_keys=True)
            seen.setdefault(key, []).append(f"{lab}:{os.path.basename(p)}")
    if len(seen) > 1:
        return [f"разные контракты раскатки: {len(seen)} вариантов, "
                f"например {[v[0] for v in seen.values()][:3]}"]
    return []


def m1(cen, q0_arts, cand, max_grip_flip=None, expected_setup=None,
       basis_report=None):
    """cand: {метка: [(путь, артефакт)]} для 16 рук.

    Перепись ВЫЧИСЛЯЕТСЯ ЗАНОВО из q0-артефактов и обязана совпасть с
    зарегистрированной во всех полях: иначе правленый JSON переписи мог бы
    подменить популяцию кластеров при тех же q0.
    """
    technical = list(cen.get("technical") or [])
    re_cen = census(q0_arts, expected_setup)
    for k in CENSUS_FIELDS:
        a_, b_ = (json.dumps(re_cen.get(k), sort_keys=True),
                  json.dumps(cen.get(k), sort_keys=True))
        if a_ != b_:
            technical.append(f"перепись: поле {k} не совпало с "
                             f"вычисленным заново по q0")
    technical += check_setup(dict(cand, q0=q0_arts), expected_setup)
    labels = kp.all_labels()
    missing = [x for x in labels if x not in cand]
    if missing:
        technical.append(f"нет рук {missing}")
    extra = [x for x in cand if x not in labels]
    if extra:
        technical.append(f"посторонние руки {extra}")
    contracts = []
    for lab, arts in cand.items():
        if lab not in labels:
            continue
        b, c = check_k15f(arts, lab)
        technical += b
        contracts += c
    technical += technical_common(cand)
    seen = {json.dumps(c, sort_keys=True, ensure_ascii=False)
            for c in contracts}
    if len(seen) > 1:
        technical.append(f"руки k15f с разными контрактами базиса: "
                         f"{len(seen)}")
    for c in contracts:
        miss = [k for k in CONTRACT if not ms.present(c.get(k))
                and c.get(k) != 0]
        if miss:
            technical.append(f"пустые поля контракта {miss}")
            break
    q0_ep, _ = episodes(q0_arts)
    fails = [(t, s, SEEDS[0]) for t, s in cen["fails"]]
    diag = [(t, s, SEEDS[0]) for t, s in cen["diag"]]
    ep = {}
    for lab in labels:
        if lab in cand:
            ep[lab], dup = episodes(cand[lab])
            if dup:
                technical.append(f"{lab}: повторяющиеся эпизоды")
            technical += pair_check(q0_ep, ep[lab], fails + diag, lab)
    learned = [x for x in labels if x.startswith("l")]
    control = [x for x in labels if x.startswith("r")]

    def saved(fam, k):
        return any(ep.get(x, {}).get(k, {}).get("success") for x in fam)
    L = {k: saved(learned, k) for k in fails}
    Cc = {k: saved(control, k) for k in fails}
    RL, RC = sum(L.values()), sum(Cc.values())
    l_only = sum(1 for k in fails if L[k] and not Cc[k])
    c_only = sum(1 for k in fails if Cc[k] and not L[k])
    per_arm = {}
    for lab in labels:
        e = ep.get(lab, {})
        per_arm[lab] = dict(
            rescue=sum(1 for k in fails if e.get(k, {}).get("success")),
            harm_on_diag=sum(1 for k in diag
                             if k in e and not e[k].get("success")))
    per_task = {}
    for t in TASKS:
        ft = [k for k in fails if k[0] == t]
        per_task[str(t)] = dict(F=len(ft), RL=sum(L[k] for k in ft),
                                RC=sum(Cc[k] for k in ft))
    F = len(fails)
    # ОТЧЁТ ПРЕДОБУЧЕНИЯ СВЯЗАН С ИСПОЛНЯЕМЫМ ЧЕКПОЙНТОМ: его
    # checkpoint_sha1 обязан совпасть с basis_sha1 рук
    basis_shas = {c.get("basis_sha1") for c in contracts}
    if basis_report is not None:
        if {basis_report.get("checkpoint_sha1")} != basis_shas:
            technical.append(f"отчёт предобучения от чекпойнта "
                             f"{basis_report.get('checkpoint_sha1')}, руки "
                             f"исполняли {sorted(map(str, basis_shas))}")
            max_grip_flip = None
    pre = decide(F, RL, RC)
    if pre.startswith("перспективно") and max_grip_flip is None:
        technical.append("нет доли смены схвата из отчёта предобучения, "
                         "связанного с этим чекпойнтом")
    decision = ("НЕ ПРИНИМАЕТСЯ: технический отказ" if technical else pre)
    ablation_required = bool(
        decision.startswith("перспективно") and max_grip_flip is not None
        and max_grip_flip > RULE["grip_flip_ablation"])
    if ablation_required:
        decision += ("; до архитектурного вывода обязательна абляция "
                     "arm-only против gripper-only (смена схвата "
                     f"{100 * max_grip_flip:.1f}% > "
                     f"{100 * RULE['grip_flip_ablation']:.0f}%)")
    return dict(kind="k15f_m1", technical=technical,
                code=3 if technical else 0, F=F, R_L=RL, R_C=RC,
                L_only=l_only, C_only=c_only,
                oracle_gain_pp=100.0 * RL / (len(TASKS) * len(DESIGN_STATES)),
                per_arm=per_arm, per_task=per_task, decision=decision,
                rule=RULE, n_diag=len(diag),
                max_grip_flip_share=max_grip_flip,
                ablation_required=ablation_required)


def smoke(q0_arts, cand, expected_setup=None):
    technical = check_q0(q0_arts) + technical_common(
        dict(cand, q0=q0_arts))
    technical += check_setup(dict(cand, q0=q0_arts), expected_setup)
    for lab, arts in cand.items():
        b, _c = check_k15f(arts, lab)
        technical += b
    ident = None
    if "z" in cand:
        ident = ms.paired_identity([p for p, _d in cand["z"]],
                                   [p for p, _d in q0_arts])
        if not ident["passed"]:
            technical.append(f"тождество z = q0 нарушено: "
                             f"{ident['identical']}/{ident['episodes']}; "
                             f"{ident['problems']}")
    else:
        technical.append("нет руки z — тождество с q0 не проверено")
    return dict(kind="k15f_m1_smoke", technical=technical,
                code=3 if technical else 0, identity_z_q0=ident)


def main():
    ap = argparse.ArgumentParser(description="K-15f M1: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=("smoke", "census", "m1"))
    ap.add_argument("--q0-arts", nargs="*", default=[])
    ap.add_argument("--arts", nargs="*", default=[])
    ap.add_argument("--census", default="reports/k15f/m1_census.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--overwrite-census", action="store_true")
    ap.add_argument("--basis-report", default="reports/k15f/basis_s0.json",
                    help="отчёт предобучения: доля смены схвата")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    q0 = [(p, json.load(open(p))) for p in a.q0_arts]
    exp = dict(SETUP_EXPECTED, script_sha1=_sha(os.path.join(
        HERE, "k9h_multiarm_gate.py")))
    if a.mode == "census":
        if os.path.exists(a.census) and not a.overwrite_census:
            raise SystemExit(f"{a.census} уже зарегистрирована")
        res = census(q0, exp)
        out = a.census
        print(f"  перепись q0: F = {res['F']} провалов из 100; диагностика "
              f"{len(res['diag'])}; блоки {res['blocks']}; мощность "
              f"{'достаточна' if res['powered'] else 'НЕДОСТАТОЧНА'}")
        print("  успех q0 по задачам: " + ", ".join(
            f"{t}: {v:.1f}" for t, v in res["per_task_sr"].items()))
    else:
        by = load_arts(a.arts)
        if a.mode == "smoke":
            res = smoke(q0, by, exp)
            ident = res["identity_z_q0"] or {}
            print(f"  тождество z = q0: {ident.get('identical')}/"
                  f"{ident.get('episodes')} эпизодов")
        else:
            cen = json.load(open(a.census))
            flip, brep = None, None
            if os.path.exists(a.basis_report):
                brep = json.load(open(a.basis_report))
                flip = brep.get("max_grip_flip_share")
            res = m1(cen, q0, by, max_grip_flip=flip, expected_setup=exp,
                     basis_report=brep)
            print(f"  F = {res['F']}, R_L = {res['R_L']}, R_C = "
                  f"{res['R_C']}, только learned {res['L_only']}, только "
                  f"контроль {res['C_only']}; выигрыш оракула learned "
                  f"{res['oracle_gain_pp']:.1f} п.п.")
            print("  по рукам (спасения / вред на диагностике): " + ", ".join(
                f"{k} {v['rescue']}/{v['harm_on_diag']}"
                for k, v in res["per_arm"].items()))
            print("  по задачам: " + "; ".join(
                f"{t}: F {v['F']}, L {v['RL']}, C {v['RC']}"
                for t, v in res["per_task"].items() if v["F"]))
            print(f"  РЕШЕНИЕ M1: {res['decision']}")
        out = a.out
    if res["technical"]:
        print(f"  ТЕХНИЧЕСКИЙ ОТКАЗ: {res['technical'][:8]}")
    print(f"  код {res['code']}")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".",
                    exist_ok=True)
        with open(out + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
        os.replace(out + ".tmp", out)
        print(f"  сводка: {out}")
    return int(res["code"])


# --- самопроверка -----------------------------------------------------------
def _fake(td, label, tasks, states, succ_fn, joint=None, policy="k15f",
          seed=101, hash_fn=None):
    import hashlib
    paths = []
    for t in tasks:
        for blk in sorted({5 * (s // 5) for s in states}):
            eps = []
            for s in range(blk, blk + 5):
                h = (hash_fn(label, t, s) if hash_fn else
                     hashlib.sha1(f"{label}{t}{s}".encode()).hexdigest()[:16])
                eps.append(dict(success=bool(succ_fn(label, t, s)),
                                init_state_id=s, env_index=s - blk,
                                init_hash=f"i{t}{s}",
                                init_hash_full=f"I{t}{s}",
                                rollout_seed=seed, action_sha1=h))
            d = dict(episodes=eps, arm_label=label, task_id=t,
                     init_start=blk, seed=seed, n_envs=5, suite="10",
                     policy=policy, levels=1, script_sha1="K9H",
                     horizon=8, max_steps=600, waiting_steps=10,
                     ensemble="off", rollout_seed_mode="fixed", ckpt="X")
            if policy == "depthrvq":
                d.update(argv=dict(depth_rvq_mode="fast"),
                         joint=dict(model_fingerprint="q"))
            else:
                c, control = kp.label_spec(label)
                d["joint"] = dict(joint or {}, label=label, coeffs=c,
                                  control=control, layers_per_call=18)
            base = os.path.join(td, f"{label}_t{t}_i{blk}")
            np.savez(base + ".actions.npz", actions=np.zeros(2),
                     action_sha1=np.asarray([e["action_sha1"] for e in eps]),
                     init_state_id=np.asarray([e["init_state_id"]
                                               for e in eps]))
            d["actions_npz"] = os.path.basename(base) + ".actions.npz"
            d["actions_npz_sha1"] = _sha(base + ".actions.npz")
            json.dump(d, open(base + ".json", "w"))
            paths.append(base + ".json")
    return paths


def selftest():
    import tempfile
    assert decide(10, 9, 0).startswith("недостаточная")
    assert decide(15, 5, 2).startswith("перспективно")
    assert decide(15, 5, 3).startswith("неясно")
    assert decide(15, 2, 0).startswith("закрыть")
    assert decide(15, 4, 4).startswith("закрыть")
    cont = dict(basis_sha1="B", basis_state_sha1="S", stats_sha1="T",
                amp_factor=1.0, control_seed=7, identity_report_sha1="I",
                refine_module_sha1="R", k15f_policy_sha1="P",
                frozen_content_sha="F", joint_sha1="J",
                code_version={"x": 1}, codec={"c": 1}, k15a_gate={"g": 1})
    q0_fail = {(8, s) for s in range(10)} | {(9, 1), (9, 6), (3, 7)}

    def q0_succ(_l, t, s):
        return (t, s) not in q0_fail

    def cand_succ(label, t, s):
        if (t, s) not in q0_fail:
            return True
        return label.startswith("l") and (s % 2 == 0)   # learned спасает

    with tempfile.TemporaryDirectory() as td:
        q0p = _fake(td, "q0", TASKS, DESIGN_STATES, q0_succ,
                    policy="depthrvq")
        q0 = [(p, json.load(open(p))) for p in q0p]
        cen = census(q0)
        assert cen["code"] == 0, cen["technical"]
        assert cen["F"] == 13 and cen["powered"]
        # по два наименьших успешных состояния на задачу; у задачи 8 успехов
        # нет вовсе
        assert len(cen["diag"]) == 2 * 9, len(cen["diag"])
        cand = {}
        for lab in kp.all_labels():
            ps = _fake(td, lab, TASKS, DESIGN_STATES, cand_succ, joint=cont)
            cand[lab] = [(p, json.load(open(p))) for p in ps]
        r = m1(cen, q0, cand, max_grip_flip=0.05)
        assert r["code"] == 0, r["technical"]
        assert r["R_C"] == 0 and r["R_L"] >= 5, (r["R_L"], r["R_C"])
        assert r["decision"].startswith("перспективно"), r["decision"]
        assert not r["ablation_required"]
        r = m1(cen, q0, cand, max_grip_flip=0.3)
        assert r["ablation_required"] and "абляция" in r["decision"]
        exp = dict(SETUP_EXPECTED, script_sha1="K9H", ckpt="X")
        r = m1(cen, q0, cand, 0.05, expected_setup=exp,
               basis_report=dict(checkpoint_sha1="B"))
        assert r["code"] == 0, r["technical"]
        # поле удалено СРАЗУ У ВСЕХ рук и у q0 — одинаковость не спасает
        def drop(arts):
            out = []
            for p_, d_ in arts:
                d2 = dict(d_)
                d2.pop("horizon")
                out.append((p_, d2))
            return out
        cand_nh = {k: drop(v) for k, v in cand.items()}
        r = m1(cen, drop(q0), cand_nh, 0.05, expected_setup=exp)
        assert any("horizon" in x for x in r["technical"]), r["technical"]
        # отчёт предобучения от другого чекпойнта
        r = m1(cen, q0, cand, 0.05, expected_setup=exp,
               basis_report=dict(checkpoint_sha1="ДРУГОЙ"))
        assert any("отчёт предобучения" in x for x in r["technical"])
        assert r["decision"].startswith("НЕ ПРИНИМАЕТСЯ")
        # перспективно, но доли смены схвата нет -> не принимается
        r = m1(cen, q0, cand, None, expected_setup=exp)
        assert r["decision"].startswith("НЕ ПРИНИМАЕТСЯ"), r["decision"]
        # правленая перепись: другой состав провалов при тех же q0
        cen_bad = dict(cen, fails=cen["fails"][1:], F=cen["F"] - 1)
        assert any("перепись" in x for x in
                   m1(cen_bad, q0, cand, 0.05)["technical"])
        # другой контракт раскатки у одной руки
        bad = dict(cand)
        d = json.load(open(cand["l1p"][0][0]))
        d["max_steps"] = 300
        bad["l1p"] = [(cand["l1p"][0][0], d)] + cand["l1p"][1:]
        assert any("контракты раскатки" in x for x in
                   m1(cen, q0, bad, 0.05)["technical"])
        # мутации: метка не своя, другой контракт, другой старт, нет руки
        bad = dict(cand)
        d = json.load(open(cand["l0p"][0][0]))
        d["joint"]["coeffs"] = [0, 0, 0, 1.0]
        bad["l0p"] = [(cand["l0p"][0][0], d)] + cand["l0p"][1:]
        assert m1(cen, q0, bad)["code"] == 3
        bad = dict(cand)
        d = json.load(open(cand["r1m"][0][0]))
        d["joint"]["basis_sha1"] = "ДРУГОЙ"
        bad["r1m"] = [(cand["r1m"][0][0], d)] + cand["r1m"][1:]
        assert any("контракт" in x for x in m1(cen, q0, bad)["technical"])
        bad = dict(cand)
        bad.pop("l3m")
        assert any("нет рук" in x for x in m1(cen, q0, bad)["technical"])
        bad = dict(cand)
        d = json.load(open(cand["l2p"][-1][0]))
        for e in d["episodes"]:
            e["init_hash"] = "ИНОЙ"
        bad["l2p"] = cand["l2p"][:-1] + [(cand["l2p"][-1][0], d)]
        assert any("другой старт" in x for x in m1(cen, q0, bad)["technical"])
        # q0 подменён после переписи
        cen2 = dict(cen, q0_files=dict(cen["q0_files"]))
        k = next(iter(cen2["q0_files"]))
        cen2["q0_files"][k] = "000000000000"
        assert any("перепись: поле q0_files" in x
                   for x in m1(cen2, q0, cand)["technical"])
        # контроль спасает столько же -> закрыть
        def same(label, t, s):
            return True if (t, s) not in q0_fail else (s % 2 == 0)
        cand2 = {lab: [(p, json.load(open(p))) for p in
                       _fake(td, lab, TASKS, DESIGN_STATES, same,
                             joint=cont)] for lab in kp.all_labels()}
        assert m1(cen, q0, cand2)["decision"].startswith("закрыть")
        # smoke: z = q0 обязателен
        zp = _fake(td, "z", [0, 8], range(5), q0_succ, joint=cont,
                   hash_fn=lambda l, t, s: __import__("hashlib").sha1(
                       f"q0{t}{s}".encode()).hexdigest()[:16])
        q0s = [(p, json.load(open(p))) for p in q0p
               if os.path.basename(p) in ("q0_t0_i0.json", "q0_t8_i0.json")]
        rs = smoke(q0s, {"z": [(p, json.load(open(p))) for p in zp]})
        assert rs["code"] == 0, rs["technical"]
        zbad = _fake(td, "z", [0, 8], range(5), q0_succ, joint=cont)
        rs = smoke(q0s, {"z": [(p, json.load(open(p))) for p in zbad]})
        assert rs["code"] == 3
    print("самопроверка k15f_measure_subspace пройдена: перепись, правило "
          "M1, парность, контракт, тождество z = q0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
