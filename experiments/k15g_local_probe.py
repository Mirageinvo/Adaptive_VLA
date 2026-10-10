#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15g: протокол локальной поправки — план, проверки, агрегация.

РЕЖИМЫ:
  plan     фиксирует НЕИЗМЕНЯЕМЫЙ план запуска до первой раскатки: SHA
           базиса, отчёта гейта, переписи M1 и итогового отчёта M1 (с его
           вердиктом — дословно), SHA кода руки, адаптера и харнесса;
           модель, карта, dtype; сид и правило сида, горизонт, лимит шагов;
           задачи, блоки, провалы и диагностические успехи; полный список
           кандидатов; моменты и длительность импульса; правила анализа.
           Перезапись — только явно.
  check    сверка артефактов с планом и проверки протокола (технический
           отказ — код 3).
  analyze  агрегация: по моментам и вместе. Новый гейт «успеха» НЕ
           вводится, пороги M1 НЕ переносятся: это диагностика для
           следующего решения.

ОБЯЗАТЕЛЬНЫЕ ПРОВЕРКИ:
  1. q0ref воспроизводит q0 переписи M1: те же хеши исполненных действий,
     исход и шаг завершения на КАЖДОМ эпизоде плана.
  2. Общий префикс: до импульса ИСПОЛНЕННЫЕ действия (после преобразований
     харнесса) побитово равны q0ref.
  3. Один импульс: k15g_active истинен ровно в вызове с номером момента.
  4. Возврат к q0: вне импульса действие побитово равно текущему a0.
  5. Момент не достигнут (среда завершилась до чанка k) — отмечается и не
     считается проведённой проверкой поправки.
  6. Целостность: npz по отпечатку, сид 101 в каждом эпизоде, ожидаемый
     контракт раскатки, один контракт базиса у всех рук K-15g, метка
     исполняет своё.

ИНТЕРПРЕТАЦИЯ. Спасения показывают существование полезных кратких
вмешательств среди проверенных кандидатов; learned против контроля — их
полезность; различие моментов — чувствительность ко времени. Лучший
кандидат «по исходу» — oracle-покрытие проверенным набором, а не
результат автономной политики. Исход M1 не переименовывается: здесь
проверяется другая гипотеза.
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

import k15g_local_policy as lp  # noqa: E402

KIND = "k15g_local_plan"
H = 8
# Задачи — НЕ фиксированный список: все задачи с провалами q0 по переписи,
# на которой зафиксирован план (после восстановления — холодная M1-cold).
BASIS_CONTRACT = ("basis_sha1", "basis_state_sha1", "stats_sha1",
                  "amp_factor", "control_seed", "control_kind",
                  "identity_report_sha1", "refine_module_sha1",
                  "k15f_policy_sha1", "k15g_policy_sha1",
                  "harness_adapter_sha1", "frozen_content_sha", "joint_sha1",
                  "code_version", "codec", "k15a_gate")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]


SMOKE = dict(tasks=[0, 8], blocks={"0": [0], "8": [0]},
             candidates=["q0ref", "p2l0p", "p2r0p", "p4l0p", "p4r0p"])


def make_plan(*, basis, identity, census_path, m1_report, census_q0_dir,
              device, dtype, tasks=None, smoke=False):
    cen = json.load(open(census_path))
    if tasks is None:
        tasks = sorted({int(t) for t, _s in cen["fails"]})
    if smoke:
        # SMOKE: техническая цепочка до завершения M1; вердикта M1 нет
        tasks = SMOKE["tasks"]
        m1 = dict(decision="M1 не учитывается: smoke-план", code=None)
        blocks = dict(SMOKE["blocks"])
    else:
        m1 = json.load(open(m1_report))
        blocks = {str(t): cen["blocks"][str(t)] for t in tasks}
    fails = [x for x in cen["fails"] if x[0] in tasks]
    diag = [x for x in cen["diag"] if x[0] in tasks]
    code = {f: sha(os.path.join(HERE, f)) for f in (
        "k15g_local_policy.py", "k15g_harness.py", "k9h_multiarm_gate.py",
        "k15f_policy.py", "k15f_continuous_refine.py",
        "k15f_check_identity.py", "k15g_local_probe.py")}
    import k15f_measure_subspace as kms
    return dict(
        expected_contract=expected_contract(basis, identity),
        kind=KIND, created=datetime.datetime.now().isoformat(
            timespec="seconds"),
        basis=os.path.abspath(basis), basis_sha1=sha(basis),
        identity_report=os.path.abspath(identity),
        identity_report_sha1=sha(identity),
        census=os.path.abspath(census_path), census_sha1=sha(census_path),
        census_q0_dir=os.path.abspath(census_q0_dir),
        smoke=bool(smoke),
        m1_report=(None if smoke else os.path.abspath(m1_report)),
        m1_report_sha1=(None if smoke else sha(m1_report)),
        m1_verdict=dict(decision=m1.get("decision"), code=m1.get("code"),
                        R_L=m1.get("R_L"), R_C=m1.get("R_C")),
        code=code, device=str(device), compute_dtype=dtype,
        setup=dict(kms.SETUP_EXPECTED, script_sha1=code["k9h_multiarm_gate.py"]),
        tasks=tasks, blocks=blocks, fails=fails, diag=diag,
        candidates=(list(SMOKE["candidates"]) if smoke
                    else ["q0ref"] + lp.all_labels()),
        moments=list(lp.MOMENTS),
        pulse_chunks=1,
        rules=dict(
            chunk_numbering="первый вызов политики — чанк 1; waiting steps "
                            "не входят",
            not_reached="среда завершилась до чанка k — момент не "
                        "достигнут, не считается проверкой",
            no_success_gate="пороги M1 не переносятся; вердикт — "
                            "диагностический",
            infra_retry="повтор только распознанных инфраструктурных "
                        "сбоев (No CUDA GPUs are available)"))


def expected_contract(basis, identity):
    """Ожидаемые значения контракта базиса — из файлов на момент плана."""
    import torch
    ck = torch.load(basis, map_location="cpu", weights_only=False)
    rep = json.load(open(identity))
    return dict(
        basis_sha1=sha(basis), basis_state_sha1=ck["state_sha1"],
        stats_sha1=ck["stats_sha1"], amp_factor=float(ck["amp_factor"]),
        control_seed=int(ck["control_seed"]),
        control_kind=ck.get("control_kind"),
        identity_report_sha1=sha(identity),
        refine_module_sha1=sha(os.path.join(HERE,
                                            "k15f_continuous_refine.py")),
        k15f_policy_sha1=sha(os.path.join(HERE, "k15f_policy.py")),
        k15g_policy_sha1=sha(os.path.join(HERE, "k15g_local_policy.py")),
        harness_adapter_sha1=sha(os.path.join(HERE, "k15g_harness.py")),
        frozen_content_sha=rep.get("frozen_sha1"),
        joint_sha1=rep.get("joint_sha1"),
        code_version=rep.get("code_version"))


def validate_block(path, label=None, task=None, block=None):
    """Пара JSON+npz одного блока цела и согласована. Список проблем.

    Используется и проверкой протокола, и возобновлением раннера: блок
    считается готовым, только если проблем нет.
    """
    p = []
    name = os.path.basename(path)
    try:
        d = json.load(open(path))
        f = os.path.join(os.path.dirname(path), d["actions_npz"])
        if sha(f) != d["actions_npz_sha1"]:
            return [f"{name}: отпечаток npz не совпал"]
        z = np.load(f, allow_pickle=True)
        zz = {k: z[k] for k in z.files}
    except Exception as e:                  # noqa: BLE001 — любой сбой = отказ
        return [f"{name}: не читается ({type(e).__name__})"]
    eps = d.get("episodes") or []
    ids = [int(e["init_state_id"]) for e in eps]
    if label is not None and d.get("arm_label") != label:
        p.append(f"{name}: метка {d.get('arm_label')!r}")
    if task is not None and int(d.get("task_id", -1)) != int(task):
        p.append(f"{name}: задача {d.get('task_id')}")
    if block is not None and ids != list(range(int(block), int(block) + 5)):
        p.append(f"{name}: состояния {ids}, ожидался блок {block}")
    if [str(x) for x in zz.get("action_sha1", [])] != \
            [str(e.get("action_sha1")) for e in eps]:
        p.append(f"{name}: action_sha1 в npz не совпадает с JSON")
    if [int(x) for x in zz.get("init_state_id", [])] != ids:
        p.append(f"{name}: init_state_id в npz не совпадает с JSON")
    if [int(x) for x in zz.get("done_step", [])] != \
            [int(e.get("done_step", -2)) for e in eps]:
        p.append(f"{name}: done_step в npz не совпадает с JSON")
    if int(d.get("seed", -1)) != 101 or any(
            int(e.get("rollout_seed", -1)) != 101 for e in eps):
        p.append(f"{name}: сид не 101")
    return p


def active_amplitude(z, label):
    """(сумма квадратов по каналам, число шагов, переключений схвата) —
    ТОЛЬКО по исполненным шагам активного чанка: среда могла завершиться
    внутри него или раньше."""
    act = np.asarray(z.get("k15g_active", []), bool)
    if not act.any():
        return np.zeros(7), 0, 0
    ci = np.where(act)[0]
    ds = np.asarray(z["done_step"])
    t = ci[:, None] * H + np.arange(H)[None, :]                  # [A, 8]
    end = np.where(ds >= 0, ds, len(z["actions"]) - 1)            # [B]
    mask = t[:, None, :] <= end[None, :, None]                   # [A, B, 8]
    a0 = z["k15d_a0"][ci]
    lv = z[f"k15d_level_{label}"][ci]
    dlt = (lv - a0)[mask]                                         # [N, 7]
    gs = int((((lv[..., 6] > 0) != (a0[..., 6] > 0)) & mask).sum())
    return (dlt ** 2).sum(0), int(len(dlt)), gs


def check_plan_inputs(plan):
    """Входные артефакты на месте и совпадают с планом."""
    p = []
    for k, f in (("basis_sha1", plan["basis"]),
                 ("identity_report_sha1", plan["identity_report"]),
                 ("census_sha1", plan["census"]),
                 ("m1_report_sha1", plan["m1_report"])):
        if f is None and plan.get("smoke") and k == "m1_report_sha1":
            continue
        if not os.path.exists(f) or sha(f) != plan[k]:
            p.append(f"{k}: файл изменён или отсутствует ({f})")
    # ВКЛЮЧАЯ сам протокол: правка проверок после плана делает план
    # недействительным
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    return p


def load(path):
    d = json.load(open(path))
    z = np.load(os.path.join(os.path.dirname(path), d["actions_npz"]),
                allow_pickle=True)
    return d, {k: z[k] for k in z.files}


def episodes(files):
    out = {}
    for p in files:
        d = json.load(open(p))
        for i, e in enumerate(d["episodes"]):
            out[(int(d["task_id"]), int(e["init_state_id"]))] = (e, p, i)
    return out


def reached(done_step, moment):
    """Достигнут ли чанк moment (1-based) средой с данным done_step."""
    return bool(done_step < 0 or done_step >= H * (moment - 1))


def check_arm_block(d, z, label, ref_z, ref_d):
    """ref_d — JSON блока q0ref (эпизоды с исходом и шагом завершения)."""
    """Проверки протокола одного блока руки импульса против q0ref."""
    bad = []
    moment, c, control = lp.label_spec(label)
    j = d.get("joint") or {}
    if j.get("label") != label or j.get("coeffs") != c or \
            bool(j.get("control")) != control or j.get("moment") != moment:
        bad.append(f"{label}: метка исполняет не своё")
    chunk = np.asarray(z.get("k15g_chunk", []))
    active = np.asarray(z.get("k15g_active", []), bool)
    if len(chunk) == 0 or chunk.tolist() != list(range(1, len(chunk) + 1)):
        bad.append(f"{label}: нумерация чанков нарушена")
    want_active = (chunk == moment) if moment is not None else \
        np.zeros_like(active)
    if active.tolist() != want_active.tolist():
        bad.append(f"{label}: импульс не ровно в чанке {moment}")
    a0, lv = z.get("k15d_a0"), z.get(f"k15d_level_{label}")
    if a0 is None or lv is None:
        bad.append(f"{label}: нет журнала a0/действий")
    else:
        off = ~active
        if off.any() and not np.array_equal(a0[off], lv[off]):
            bad.append(f"{label}: вне импульса действие не равно a0")
    # общий префикс исполненных действий с q0ref
    if moment is not None:
        n_pre = H * (moment - 1)
        A, Ar = z["actions"], ref_z["actions"]
        ds, dr = np.asarray(z["done_step"]), np.asarray(ref_z["done_step"])
        ref_eps = (ref_d or {}).get("episodes") or []
        own_eps = d.get("episodes") or []
        for b in range(A.shape[1]):
            end = n_pre if reached(int(dr[b]), moment) else int(dr[b]) + 1
            if not reached(int(dr[b]), moment):
                # импульса не было: исход и завершение обязаны совпасть
                if int(ds[b]) != int(dr[b]):
                    bad.append(f"{label}: среда {b} завершилась до импульса "
                               f"не так, как q0ref")
                if ref_eps and own_eps and \
                        own_eps[b].get("success") != ref_eps[b].get("success"):
                    bad.append(f"{label}: среда {b} без импульса изменила "
                               f"исход")
            if not np.array_equal(A[:end, b], Ar[:end, b]):
                bad.append(f"{label}: префикс среды {b} до импульса не "
                           f"совпал с q0ref")
    return bad


def check(plan, run_dir):
    """Все проверки протокола. Возвращает (проблемы, покрытие)."""
    import k15f_measure_subspace as kms
    import k15e_measure_scale as ms
    from k15d_behavior import check_actions_npz
    technical = check_plan_inputs(plan)
    labels = plan["candidates"]
    want = {(int(t), int(b)) for t, bl in plan["blocks"].items() for b in bl}
    files = {lab: sorted(glob.glob(os.path.join(run_dir,
                                                f"{lab}_t*_i*.json")))
             for lab in labels}
    cover = {}
    for lab, fl in files.items():
        got = {(json.load(open(p))["task_id"], json.load(open(p))[
            "init_start"]) for p in fl}
        cover[lab] = len(got & want)
        extra = got - want
        if extra:
            technical.append(f"{lab}: блоки вне плана {sorted(extra)}")
    all_files = [p for fl in files.values() for p in fl]
    for lab, fl in files.items():
        keys = []
        for p in fl:
            d_ = json.load(open(p))
            technical += validate_block(p, lab, d_["task_id"],
                                        d_["init_start"])
            keys += [(d_["task_id"], e["init_state_id"])
                     for e in d_["episodes"]]
        if len(keys) != len(set(keys)):
            technical.append(f"{lab}: повторяющиеся эпизоды")
    exp_c = plan.get("expected_contract") or {}
    if not exp_c:
        technical.append("в плане нет ожидаемого контракта базиса")
    for p in all_files:
        j_ = json.load(open(p)).get("joint") or {}
        for k, v in exp_c.items():
            if json.dumps(j_.get(k), sort_keys=True) != json.dumps(
                    v, sort_keys=True):
                technical.append(f"{os.path.basename(p)}: {k} = "
                                 f"{j_.get(k)!r}, план {v!r}")
    technical += check_actions_npz(all_files)
    technical += ms.check_episodes(all_files, seeds=[101])
    technical += kms.check_setup(
        {lab: [(p, json.load(open(p))) for p in fl]
         for lab, fl in files.items() if fl}, plan["setup"])
    contracts = {json.dumps({k: (json.load(open(p)).get("joint") or {})
                             .get(k) for k in BASIS_CONTRACT},
                            sort_keys=True) for p in all_files}
    if len(contracts) > 1:
        technical.append(f"руки K-15g с разными контрактами базиса: "
                         f"{len(contracts)}")
    # 1. q0ref воспроизводит q0 переписи M1
    cen_q0 = episodes(sorted(glob.glob(os.path.join(
        plan["census_q0_dir"], "q0_t*_i*.json"))))
    ref = episodes(files["q0ref"])
    for k, (e, _p, _i) in ref.items():
        if k not in cen_q0:
            technical.append(f"q0ref {k}: нет в переписи M1")
            continue
        c_e = cen_q0[k][0]
        for f in ("action_sha1", "success", "done_step"):
            if str(e.get(f)) != str(c_e.get(f)):
                technical.append(f"q0ref {k}: {f} не совпал с q0 M1")
    # 1б. СТАРТ КАЖДОГО ЭПИЗОДА КАЖДОЙ РУКИ равен старту q0 переписи M1
    # (init_hash, init_hash_full, rollout_seed): в многоблочном процессе
    # reset на задаче 8 восстанавливал среду не полностью
    for lab in labels:
        for k, (e, _p, _i) in episodes(files[lab]).items():
            if k not in cen_q0:
                continue
            c_e = cen_q0[k][0]
            if any(str(e.get(f)) != str(c_e.get(f)) for f in
                   ("init_hash", "init_hash_full", "rollout_seed")):
                technical.append(f"{lab} {k}: старт не равен старту q0 "
                                 f"переписи")
    # 2-4. блоки рук импульса против q0ref того же блока
    ref_blocks = {(json.load(open(p))["task_id"],
                   json.load(open(p))["init_start"]): p
                  for p in files["q0ref"]}
    for lab in labels:
        if lab == "q0ref":
            continue
        for p in files[lab]:
            d, z = load(p)
            key = (d["task_id"], d["init_start"])
            if key not in ref_blocks:
                technical.append(f"{lab} {key}: нет блока q0ref")
                continue
            rd, rz = load(ref_blocks[key])
            technical += check_arm_block(d, z, lab, rz, rd)
    return technical, cover


def analyze(plan, run_dir, require_complete=True):
    technical, cover = check(plan, run_dir)
    labels = [x for x in plan["candidates"] if x != "q0ref"]
    fails = [tuple(x) for x in plan["fails"]]
    diag = [tuple(x) for x in plan["diag"]]
    eps = {lab: episodes(sorted(glob.glob(os.path.join(
        run_dir, f"{lab}_t*_i*.json")))) for lab in plan["candidates"]}
    complete = all(cover.get(lab, 0) == sum(len(v) for v in
                                            plan["blocks"].values())
                   for lab in plan["candidates"])

    def stats_for(labs):
        rescued, harmed, reached_n, tested = {}, {}, 0, 0
        for lab in labs:
            m, _c, _ctl = lp.label_spec(lab)
            for k in fails + diag:
                if k not in eps[lab]:
                    continue
                e = eps[lab][k][0]
                if not reached(int(e["done_step"]), m):
                    continue
                reached_n += 1
                if k in fails:
                    tested += 1
                    if e["success"]:
                        rescued.setdefault(k, []).append(lab)
                elif not e["success"]:
                    harmed.setdefault(k, []).append(lab)
        return rescued, harmed, reached_n, tested

    if require_complete and not complete:
        technical = technical + [f"набор неполный: {cover}"]
    out = dict(complete=complete, coverage=cover, technical=technical,
               moments={})
    for m in lp.MOMENTS + (None,):
        labs = [x for x in labels if m is None or x.startswith(f"p{m}")]
        L = [x for x in labs if x[2] == "l"]
        C = [x for x in labs if x[2] == "r"]
        rL, hL, nL, tL = stats_for(L)
        rC, hC, nC, tC = stats_for(C)
        per_task = {}
        for t in plan["tasks"]:
            ft = [k for k in fails if k[0] == t]
            per_task[str(t)] = dict(F=len(ft),
                                    learned=sum(k in rL for k in ft),
                                    control=sum(k in rC for k in ft))
        per_dir = {}
        for lab in L:
            r_, h_, _n, _t = stats_for([lab])
            rc_, hc_, _n2, _t2 = stats_for(["p" + lab[1] + "r" + lab[3:]])
            per_dir[lab[1:].replace("l", "", 1)] = dict(
                learned_rescues=len(r_), control_rescues=len(rc_),
                learned_harms=len(h_), control_harms=len(hc_))
        out["moments"]["all" if m is None else str(m)] = dict(
            reached_interventions=dict(learned=nL, control=nC),
            tested_failures=dict(learned=tL, control=tC),
            rescued_failures=dict(learned=len(rL), control=len(rC),
                                  learned_only=len(set(rL) - set(rC)),
                                  control_only=len(set(rC) - set(rL)),
                                  both=len(set(rL) & set(rC))),
            harmed_diag=dict(learned=len(hL), control=len(hC)),
            oracle_coverage_note=("лучший кандидат по исходу — покрытие "
                                  "проверенным набором, НЕ результат "
                                  "автономной политики"),
            per_task=per_task, per_direction=per_dir,
            rescued_by=({f"{k[0]}/{k[1]}": v for k, v in rL.items()}))
    # фактическая амплитуда и схват в активном чанке
    amp = {}
    for lab in labels:
        sq, n, gs = np.zeros(7), 0, 0
        for p in sorted(glob.glob(os.path.join(run_dir,
                                               f"{lab}_t*_i*.json"))):
            d, z = load(p)
            q_, n_, g_ = active_amplitude(z, lab)
            sq += q_
            n += n_
            gs += g_
        if n:
            amp[lab] = dict(rms_arm=float(np.sqrt(sq[:6].sum() / (6 * n))),
                            grip_switch_share=gs / n, steps=n)
    out["active_chunk_amplitude"] = amp
    out["m1_verdict_verbatim"] = plan["m1_verdict"]
    return out


def main():
    ap = argparse.ArgumentParser(description="K-15g: протокол")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=("plan", "check", "analyze"))
    ap.add_argument("--plan", default="reports/k15g/local_plan.json")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--basis", default="data/k15f/basis_s0.pt")
    ap.add_argument("--identity", default="reports/k15f/identity_cuda1.json")
    ap.add_argument("--census", default="reports/k15f/m1_census.json")
    ap.add_argument("--m1-report", default=None)
    ap.add_argument("--census-q0-dir", default=None)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", default="reports/k15g/local_analysis.json")
    ap.add_argument("--overwrite-plan", action="store_true")
    ap.add_argument("--validate-block", nargs=4, default=None,
                    metavar=("JSON", "МЕТКА", "ЗАДАЧА", "БЛОК"),
                    help="для раннера: 0 — блок цел, 2 — повреждён")
    ap.add_argument("--smoke", action="store_true",
                    help="малый план smoke: задачи 0 и 8, блок 0, пять рук")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.validate_block:
        path, lab, t, b = a.validate_block
        prob = validate_block(path, lab, int(t), int(b))
        if prob:
            print("; ".join(prob))
        return 2 if prob else 0
    if a.mode == "plan":
        if os.path.exists(a.plan) and not a.overwrite_plan:
            raise SystemExit(f"{a.plan} уже зафиксирован")
        if not a.census_q0_dir or (not a.m1_report and not a.smoke):
            raise SystemExit("нужны --census-q0-dir и (кроме smoke) "
                             "--m1-report")
        plan = make_plan(basis=a.basis, identity=a.identity,
                         census_path=a.census, m1_report=a.m1_report,
                         census_q0_dir=a.census_q0_dir, device=a.device,
                         dtype=a.dtype, smoke=a.smoke)
        os.makedirs(os.path.dirname(os.path.abspath(a.plan)), exist_ok=True)
        with open(a.plan + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(plan, fh, ensure_ascii=False, indent=1)
        os.replace(a.plan + ".tmp", a.plan)
        print(f"  план зафиксирован: {a.plan}; задачи {plan['tasks']}, "
              f"блоки {plan['blocks']}, провалов {len(plan['fails'])}, "
              f"диагностика {len(plan['diag'])}, кандидатов "
              f"{len(plan['candidates'])}; M1: "
              f"{plan['m1_verdict']['decision']}")
        return 0
    plan = json.load(open(a.plan))
    if plan.get("kind") != KIND:
        raise SystemExit("не план K-15g")
    if a.mode == "check":
        prob, cover = check(plan, a.run_dir)
        print(f"  покрытие: {cover}")
        print(f"  проблем: {len(prob)}" + (f"; {prob[:8]}" if prob else ""))
        return 3 if prob else 0
    res = analyze(plan, a.run_dir)
    for m, v in res["moments"].items():
        r = v["rescued_failures"]
        print(f"  момент {m}: вмешательств достигнуто L {v['reached_interventions']['learned']}"
              f" / C {v['reached_interventions']['control']}; спасено провалов "
              f"L {r['learned']}, C {r['control']} (только L {r['learned_only']}, "
              f"только C {r['control_only']}, общие {r['both']}); вред на "
              f"диагностике L {v['harmed_diag']['learned']}, C "
              f"{v['harmed_diag']['control']}")
    if res["technical"]:
        print(f"  ТЕХНИЧЕСКИЙ ОТКАЗ: {res['technical'][:8]}")
    if not res["complete"]:
        print("  набор неполный — сводка предварительная")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print(f"  сводка: {a.out}")
    return 3 if res["technical"] else 0


def selftest():
    assert reached(-1, 4) and reached(24, 4) and not reached(23, 4)
    assert reached(8, 2) and not reached(7, 2)
    # блок импульса: префикс, один импульс, возврат к a0
    C, B = 5, 2
    a0 = np.zeros((C, B, H, 7), np.float32)
    lv = a0.copy()
    lv[1] += 0.1                                   # импульс в чанке 2
    A = np.zeros((C * H, B, 7), np.float32)
    Ar = A.copy()
    A[H:2 * H] += 0.1                              # после префикса — иначе
    z = dict(k15g_chunk=np.arange(1, C + 1), k15g_active=np.arange(1, C + 1)
             == 2, k15d_a0=a0, k15d_level_p2l0p=lv, actions=A,
             done_step=np.array([-1, 20]))
    rz = dict(actions=Ar, done_step=np.array([-1, 20]))
    d = dict(joint=dict(label="p2l0p", coeffs=[1.0, 0, 0, 0], control=False,
                        moment=2))
    assert check_arm_block(d, z, "p2l0p", rz, {}) == []
    z2 = dict(z, k15g_active=np.arange(1, C + 1) == 3)
    assert any("ровно в чанке" in x for x in
               check_arm_block(d, z2, "p2l0p", rz, {}))
    A3 = A.copy()
    A3[3, 0] += 1.0                                # расхождение в префиксе
    assert any("префикс" in x for x in check_arm_block(
        d, dict(z, actions=A3), "p2l0p", rz, {}))
    lv4 = lv.copy()
    lv4[3] += 0.01                                 # вне импульса не a0
    assert any("не равно a0" in x for x in check_arm_block(
        d, dict(z, **{"k15d_level_p2l0p": lv4}), "p2l0p", rz, {}))
    d_bad = dict(joint=dict(d["joint"], moment=4))
    assert any("не своё" in x for x in check_arm_block(
        d_bad, z, "p2l0p", rz, {}))
    # среда завершилась до импульса — сравнивается вся траектория
    rz5 = dict(actions=Ar, done_step=np.array([3, 20]))
    z5 = dict(z, done_step=np.array([3, 20]))
    assert check_arm_block(d, z5, "p2l0p", rz5, {}) == []
    # без импульса (среда завершилась раньше) исход обязан совпасть
    ref_d = dict(episodes=[dict(success=True), dict(success=False)])
    own_d = dict(d, episodes=[dict(success=False), dict(success=False)])
    assert any("изменила исход" in x for x in check_arm_block(
        own_d, z5, "p2l0p", rz5, ref_d))
    assert check_arm_block(dict(d, episodes=ref_d["episodes"]), z5,
                           "p2l0p", rz5, ref_d) == []
    # амплитуда — только исполненные шаги: среда 0 завершилась на шаге 10,
    # то есть внутри активного чанка 2 (шаги 8..15) исполнены 8, 9, 10
    za = dict(z, done_step=np.array([10, -1]))
    sq_, n_, _g = active_amplitude(za, "p2l0p")
    assert n_ == 3 + 8, n_
    assert abs(sq_[0] - 0.01 * 11) < 1e-6
    # целостность пары JSON+npz для возобновления
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        eps = [dict(init_state_id=5 + k, action_sha1=f"{k:016x}",
                    done_step=20 + k, rollout_seed=101, success=True)
               for k in range(5)]

        def write(eps_, npz_kw):
            np.savez(os.path.join(td, "a.actions.npz"), **npz_kw)
            dd = dict(arm_label="p2l0p", task_id=8, init_start=5, seed=101,
                      episodes=eps_, actions_npz="a.actions.npz",
                      actions_npz_sha1=sha(os.path.join(td,
                                                        "a.actions.npz")))
            json.dump(dd, open(os.path.join(td, "a.json"), "w"))
            return os.path.join(td, "a.json")
        good = dict(action_sha1=np.asarray([e["action_sha1"] for e in eps]),
                    init_state_id=np.arange(5, 10),
                    done_step=np.arange(20, 25))
        pth = write(eps, good)
        assert validate_block(pth, "p2l0p", 8, 5) == []
        assert validate_block(pth, "p2l0p", 8, 0)          # не тот блок
        assert validate_block(pth, "p4l0p", 8, 5)          # не та метка
        pth = write(eps, dict(good, done_step=np.arange(21, 26)))
        assert any("done_step" in x for x in validate_block(pth))
        pth = write(eps, dict(good, action_sha1=np.asarray(["0" * 16] * 5)))
        assert any("action_sha1" in x for x in validate_block(pth))
        pth = write(eps, good)
        np.savez(os.path.join(td, "a.actions.npz"), x=np.ones(1))
        assert any("отпечаток" in x for x in validate_block(pth))
        open(os.path.join(td, "a.json"), "w").write("{обрыв")
        assert any("не читается" in x for x in validate_block(pth))
    assert len(lp.all_labels()) == 32
    print("самопроверка k15g_local_probe пройдена: момент достигнут/нет, "
          "один импульс, префикс, возврат к a0, метка")
    return 0


if __name__ == "__main__":
    sys.exit(main())
