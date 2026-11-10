#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-34: подготовка источников, планы раскаток, транзакции и проверки блоков.

prepare — reports/k34/prepare.json: четыре источника K-26 (parents_attn24 /
  flat_attn24 × seeds 0/1) из закреплённых планов K-26 со сверкой SHA плана,
  best.pt = спецификации = отчёту обучения, шага best, кода K-26
  (k33_policy.make_source); подготовка K-21 (prep_id), ревизия BAR.
  Неизменяема: другой источник — отказ.
plan --kind smoke|pilot|eval — reports/k34/plan_<kind>.json:
  smoke — T0 на глубинах 8/16/24 в состоянии init (источник + копии голов,
          до обучения), LIBERO-10 t0 init 20, отладочный снимок головы;
  pilot — T0_d8/T0_d16/T0_d24 (обученный best) и R_T0 (источник K-26),
          10 задач × init 20–24 (блок i20), 40 блоков, 200 эпизодов;
  eval  — на каждом сиде T8/T16/T24, F8/F16/F24, R_T24, R_F24; init 30–44
          (блоки i30/i35/i40), 480 блоков, 2400 эпизодов.
  Порядок задач — перестановка seed 34010, руки внутри задачи циклически
  сдвигаются; все руки одной задачи — одни и те же блоки.
block --op reserve|finish — reserve -> процесс K-9h -> атомарный JSON+npz ->
  finish (проверка блока, расход 5·(env_steps + 10), commit/settle; провал —
  карантин и маркер отказа, новая попытка той же ячейки только по решению).
  Журнал data/k34/ledger.json: checks 20 000, pilot 122 000, eval 1 464 000.
smoke-check — проверки блоков smoke + пересчёт головы каждой глубины из
  checkpoint на сохранённом входе первого вызова: на том же устройстве под
  тем же fp16-autocast харнесса (гейт: y, grip ≤ 1e-5, arm ≤ 1e-6) и
  CPU float32 (описательно); размер блока и прогноз диска.
pilot-gate — SR рук пилота (по 50 эпизодов) и правило §8; задержка — из
  reports/k34/latency_pilot.json (свободная карта), иначе awaiting_latency.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k34_model as k34m  # noqa: E402
import k34_policy as kp  # noqa: E402

PREPARE = "reports/k34/prepare.json"
OUT = "reports/k34"
LEDGER = "data/k34/ledger.json"
CAPS = dict(checks=20_000, pilot=122_000, eval=1_464_000)
BLOCK_CAP = 5 * 610
REJECTED = "reports/k34/rejected"
TASK_SEED = 34010
ARM_ID = {("parents_attn24", 0): "T0", ("flat_attn24", 0): "F0",
          ("parents_attn24", 1): "T1", ("flat_attn24", 1): "F1"}
SHORT = dict(parents_attn24="T", flat_attn24="F")
KINDS = ("smoke", "pilot", "eval")
SCOPE = dict(smoke="checks", pilot="pilot", eval="eval")
SMOKE_CELLS = ((0, 20),)
PILOT_BLOCKS = (20,)
EVAL_BLOCKS = (30, 35, 40)
GATE = dict(t24_vs_r=5, early_vs_t24=5, latency_ratio=0.85)
DBG_TOL = dict(y=1e-5, grip_logit=1e-5, arm=1e-6)
CODE = ("k34_model.py", "k34_train.py", "k34_policy.py", "k34_harness.py",
        "k34_protocol.py", "k34_analyze.py", "k34_latency.py", "k34_run.sh",
        "k9h_multiarm_gate.py", "k16_harness.py", "k16_protocol.py",
        "k15g_local_probe.py", "k15h_protocol.py", "k15f_policy.py",
        "k21_main_policy.py", "k21_policy.py", "k21_protocol.py",
        "k26_policy.py", "k26_protocol.py", "k26_model.py",
        "k26_attention.py", "k21_tree.py", "k21_model.py", "k21_data.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def save_json(path, d):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(path + ".tmp", path)


def ledger():
    import k31_collect as kcol
    return kcol.Ledger(LEDGER, CAPS)


def plan_path(kind):
    return os.path.join(OUT, f"plan_{kind}.json")


# --- подготовка ----------------------------------------------------------------------------
def make_prepare():
    import k21_data as d21
    import k33_policy as kp33
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    src = {}
    for (mode, seed), aid in ARM_ID.items():
        s = kp33.make_source(aid)
        src[f"{mode}_s{seed}"] = dict(spec=s["spec"], k26_plan=s["k26_plan"],
                                      k26_plan_sha1=s["k26_plan_sha1"],
                                      content=s["content"])
    return dict(kind="k34_prepare", sources=src, prep_id=prep["prep_id"],
                bar_revision=prep["bar_pin"]["revision"],
                bounds=prep["bounds_codec"],
                n_new_params_H768=k34m.n_new_params(768))


def prepare(a):
    import k33_policy as kp33
    new = make_prepare()
    if os.path.exists(PREPARE):
        old = json.load(open(PREPARE, encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "created"} != new:
            raise SystemExit(f"{PREPARE} закреплён с другими источниками")
        print(f"  {PREPARE}: закреплён, источники совпадают")
    else:
        save_json(PREPARE, dict(new, created=datetime.datetime.now()
                                .isoformat(timespec="seconds")))
        print(f"  {PREPARE}: источники записаны")
    for k, s in new["sources"].items():
        aid = ARM_ID[(s["spec"]["arch"], int(s["spec"]["seed"]))]
        pr = kp33.source_problems(aid, dict(s, spec=s["spec"]))
        if pr:
            raise SystemExit(f"источник {k}: " + "; ".join(pr))
        print(f"    {k}: best.pt {s['spec']['ckpt_sha1'][:12]}, шаг best "
              f"{s['spec']['best_update']}, веса {s['spec']['weights_hash'][:12]}")
    return 0


def source(mode, seed):
    p = json.load(open(PREPARE, encoding="utf-8"))
    return p["sources"][f"{mode}_s{seed}"]


# --- планы ---------------------------------------------------------------------------------
def task_order():
    return [int(t) for t in np.random.default_rng(TASK_SEED).permutation(10)]


def arms_for(kind):
    """[(метка, mode, seed, depth|None)] в базовом порядке."""
    if kind == "smoke":
        return [(kp.label("parents_attn24", 0, d), "parents_attn24", 0, d)
                for d in k34m.EXITS]
    if kind == "pilot":
        return [(kp.label("parents_attn24", 0, d), "parents_attn24", 0, d)
                for d in k34m.EXITS] + [(kp.label("parents_attn24", 0),
                                         "parents_attn24", 0, None)]
    out = []
    for seed in (0, 1):
        for mode in ("parents_attn24", "flat_attn24"):
            out += [(kp.label(mode, seed, d), mode, seed, d)
                    for d in k34m.EXITS]
        out += [(kp.label(m, seed), m, seed, None)
                for m in ("parents_attn24", "flat_attn24")]
    return out


def cells(kind):
    """[(метка, задача, блок)] в порядке исполнения."""
    arms = [x[0] for x in arms_for(kind)]
    if kind == "smoke":
        return [(lab, t, b) for t, b in SMOKE_CELLS for lab in arms]
    blocks = PILOT_BLOCKS if kind == "pilot" else EVAL_BLOCKS
    out = []
    for pos, t in enumerate(task_order()):
        k = pos % len(arms)
        rot = arms[k:] + arms[:k]
        for lab in rot:
            out += [(lab, t, b) for b in blocks]
    return out


def arm_spec(kind, mode, seed, depth):
    s = source(mode, seed)
    if depth is None:
        return dict(kind="R", **s["spec"])
    if kind == "smoke":
        return dict(kind="init", mode=mode, seed=seed, depth=depth,
                    source=s["spec"], ckpt=None, ckpt_sha1="init",
                    weights_hash=None)
    import k34_train as k34t
    rp = k34t.report_path(seed, mode)
    if not os.path.exists(rp):
        raise SystemExit(f"нет отчёта обучения K-34 {rp}")
    r = json.load(open(rp, encoding="utf-8"))
    ck = os.path.join(k34t.run_dir(seed, mode), "best.pt")
    if sha(ck) != r["best_sha1"]:
        raise SystemExit(f"{ck} ≠ отчёту обучения K-34")
    return dict(kind="K34", mode=mode, seed=seed, depth=depth,
                source=s["spec"], ckpt=os.path.abspath(ck),
                ckpt_sha1=r["best_sha1"], weights_hash=r["best_weights_hash"],
                best_update=r["best_update"], train_report=os.path.abspath(rp),
                train_report_sha1=sha(rp))


def make_plan(kind, device):
    import k15g_local_probe as pr
    import k21_data as d21
    if kind not in KINDS:
        raise SystemExit(f"вид плана {kind}")
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    p = json.load(open(PREPARE, encoding="utf-8"))
    arms = {lab: dict(mode=m, seed=s, depth=d,
                      spec=arm_spec(kind, m, s, d))
            for lab, m, s, d in arms_for(kind)}
    return dict(kind="k34_plan", plan_kind=kind,
                created=datetime.datetime.now().isoformat(timespec="seconds"),
                arms=arms, cells=[list(c) for c in cells(kind)],
                debug=kind == "smoke", scope=SCOPE[kind],
                setup=dict(pr.expected_setup(), ckpt=snap),
                bar_revision=prep["bar_pin"]["revision"],
                prep_id=prep["prep_id"], bounds=prep["bounds_codec"],
                prepare_sha1=sha(PREPARE), task_order=task_order(),
                n_envs=5, env_seed=101, gate=GATE,
                code={f: sha(os.path.join(HERE, f)) for f in CODE
                      if os.path.exists(os.path.join(HERE, f))},
                device=str(device))


def plan_cmd(a):
    p = plan_path(a.kind)
    if os.path.exists(p):
        old = json.load(open(p, encoding="utf-8"))
        pr = plan_problems(old)
        if pr:
            raise SystemExit(f"{p}: закреплённый план не сходится — "
                             + "; ".join(pr[:3]))
        print(f"  план {p}: закреплён")
        return 0
    save_json(p, make_plan(a.kind, a.device))
    print(f"  план {p}: {len(cells(a.kind))} блоков, руки "
          f"{list(dict.fromkeys(c[0] for c in cells(a.kind)))}")
    return 0


def plan_problems(plan):
    p = []
    if plan.get("kind") != "k34_plan":
        return ["не план K-34"]
    k = plan["plan_kind"]
    if plan["cells"] != [list(c) for c in cells(k)]:
        p.append("ячейки ≠ расписанию")
    if not os.path.exists(PREPARE) or sha(PREPARE) != plan["prepare_sha1"]:
        p.append("подготовка изменена")
    for f, s in plan["code"].items():
        q = os.path.join(HERE, f)
        if not os.path.exists(q) or sha(q) != s:
            p.append(f"код {f} изменён после плана")
    for lab, arm in plan["arms"].items():
        sp = arm["spec"]
        if sp["kind"] == "K34" and (not os.path.exists(sp["ckpt"]) or
                                    sha(sp["ckpt"]) != sp["ckpt_sha1"]):
            p.append(f"{lab}: checkpoint K-34 изменён или отсутствует")
    return p


# --- транзакции -----------------------------------------------------------------------------
def block_path(kind, lab, task, start):
    return os.path.join(OUT, kind, lab, f"t{task}_i{start}.json")


def eid(kind, lab, task, start):
    return f"{SCOPE[kind]}/k34/{kind}/{lab}/t{task}/b{start}"


def reject_path(e):
    return os.path.join(REJECTED, e.replace("/", "__") + ".json")


def quarantine(path):
    q = os.path.join(OUT, "quarantine", time.strftime("%Y%m%dT%H%M%S"))
    os.makedirs(q, exist_ok=True)
    for f in (path, path[:-5] + ".actions.npz"):
        if os.path.exists(f):
            shutil.move(f, os.path.join(q, os.path.basename(
                os.path.dirname(f)) + "_" + os.path.basename(f)))
    return q


def block_actual(path):
    d = json.load(open(path, encoding="utf-8"))
    eps = d["episodes"]
    steps = int(eps[0]["env_steps"])
    if len(eps) != 5 or any(int(e["env_steps"]) != steps for e in eps):
        raise SystemExit(f"{path}: эпизодов/env_steps")
    return 5 * (steps + int(d.get("waiting_steps", 10)))


def block_problems(plan, plan_file, kind, lab, task, start, path=None):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import k16_protocol as p16
    import k21_protocol as pp
    path = path or block_path(kind, lab, task, start)
    if not os.path.exists(path):
        return [f"нет блока {path}"]
    p = pr.validate_block(path, lab, int(task), int(start),
                          require_single=True, setup=plan["setup"])
    if p:
        return p
    d, z = p16.load(path)
    if os.path.basename(os.path.normpath(str(d.get("ckpt")))) != \
            plan["bar_revision"]:
        p.append(f"{lab}: снимок BAR блока не закреплённой ревизии")
    arm = plan["arms"][lab]
    sp = arm["spec"]
    j = d.get("joint") or {}
    if arm["depth"] is None:
        import k26_protocol as pc
        want = dict(label=lab, ckpt_sha1=sp["ckpt_sha1"],
                    weights_hash=sp["weights_hash"], k34_kind="R",
                    k34_plan_sha1=sha(plan_file))
        p += [f"{lab}: {k} = {j.get(k)!r}" for k, v in want.items()
              if j.get(k) != v]
        p += pp.journal_problems(lab, z, plan["bounds"])
        if not p:
            p += pc.attn_problems(z, lab, arm["mode"], debug=False)
    else:
        want = dict(label=lab, k34_kind="K34", k34_depth=arm["depth"],
                    k34_ckpt_sha1=sp["ckpt_sha1"],
                    k34_source_sha1=sp["source"]["ckpt_sha1"],
                    k34_debug=bool(plan["debug"]),
                    k34_plan_sha1=sha(plan_file))
        p += [f"{lab}: {k} = {j.get(k)!r}, ожидалось {v!r}"
              for k, v in want.items() if j.get(k) != v]
        p += pp.journal_problems(lab, z, plan["bounds"])
        p += kp.journal_problems(lab, z, debug=bool(plan["debug"]),
                                 bounds=plan["bounds"])
    if not p:
        p += hpc.actions_problems(lab, z, hpc.harness_scale())
    return p


def block_cmd(a):
    import k31_collect as kcol  # noqa: F401
    plan = json.load(open(a.plan, encoding="utf-8"))
    kind = plan["plan_kind"]
    pr_ = plan_problems(plan)
    if pr_:
        raise SystemExit("план: " + "; ".join(pr_[:3]))
    if [a.label, a.task, a.start] not in plan["cells"]:
        raise SystemExit(f"{a.label} t{a.task} i{a.start} не из плана")
    e = eid(kind, a.label, a.task, a.start)
    path = block_path(kind, a.label, a.task, a.start)
    L = ledger()
    if os.path.exists(reject_path(e)):
        raise SystemExit(f"{e}: результат был отвергнут — технический стоп")
    if a.op == "reserve":
        if os.path.exists(path):
            raise SystemExit(f"{path}: результат уже есть — нужен finish")
        if L.is_done(e):
            raise SystemExit(f"{e}: учтён, но результата нет")
        L.reserve(SCOPE[kind], e, amount=BLOCK_CAP)
        print(f"    резерв {e}")
        return 0
    pr_ = block_problems(plan, a.plan, kind, a.label, a.task, a.start, path)
    if pr_:
        q = quarantine(path) if os.path.exists(path) else None
        save_json(reject_path(e), dict(eid=e, reasons=pr_, quarantine=q))
        raise SystemExit(f"{a.label} t{a.task} i{a.start}: " +
                         "; ".join(pr_[:4]) + f" -> карантин {q}")
    act = block_actual(path)
    done = L.d["done"].get(e)
    pd = L.d["pending"]
    if done is None and not (pd and pd.get("id") == e):
        raise SystemExit(f"{e}: результат есть, но не учтён и не "
                         f"зарезервирован")
    L.settle(e, act)
    sc = SCOPE[kind]
    print(f"    {a.label} t{a.task} i{a.start}: проверен, шагов {act}; {sc} "
          f"{L.spent(sc)}/{CAPS[sc]}")
    return 0


def cells_cmd(a):
    plan = json.load(open(a.plan, encoding="utf-8"))
    pr_ = plan_problems(plan)
    if pr_:
        raise SystemExit("план: " + "; ".join(pr_[:3]))
    print(f"#{len(plan['cells'])}")
    for lab, t, b in plan["cells"]:
        print(f"{lab}:{t}:{b}:{int(plan['debug'])}")
    return 0


def parity_problems(plan, kind):
    """Одинаковые post-wait хеши у всех рук на каждом (задача, init)."""
    h = {}
    for lab, t, b in plan["cells"]:
        f = block_path(kind, lab, t, b)
        if not os.path.exists(f):
            continue
        for e in json.load(open(f, encoding="utf-8"))["episodes"]:
            h.setdefault((t, int(e["init_state_id"])), set()).add(
                (e["init_hash"], e["init_hash_full"]))
    return [f"t{k[0]} init {k[1]}: начальные хеши рук различаются"
            for k, v in h.items() if len(v) != 1]


# --- smoke ---------------------------------------------------------------------------------
def smoke_check(a):
    import torch
    plan = json.load(open(a.plan, encoding="utf-8"))
    if plan["plan_kind"] != "smoke":
        raise SystemExit("нужен план smoke")
    prob, rep = [], {}
    sizes = []
    for lab, t, b in plan["cells"]:
        f = block_path("smoke", lab, t, b)
        pr_ = block_problems(plan, a.plan, "smoke", lab, t, b, f)
        prob += pr_
        if pr_:
            continue
        import k16_protocol as p16
        _d, z = p16.load(f)
        arm = plan["arms"][lab]
        ck = arm["spec"]["source"]["ckpt"]
        depth = arm["depth"]
        L, U = np.asarray(plan["bounds"]["L"]), np.asarray(plan["bounds"]["U"])
        # init: головы 8/16 — копии readout24 источника
        same = kp.head_recompute(torch, z, ck, 24, L, U, device=a.device,
                                 autocast=a.device.startswith("cuda"))
        cpu = kp.head_recompute(torch, z, ck, 24, L, U)
        bad = {k: v for k, v in same.items() if not (v <= DBG_TOL[k])}
        if set(same) != set(DBG_TOL):
            bad["поля"] = sorted(same)
        if bad:
            prob.append(f"{lab}: пересчёт головы на том же устройстве — "
                        f"{bad}")
        sizes += [os.path.getsize(q) for q in (f, f[:-5] + ".actions.npz")]
        rep[lab] = dict(depth=depth, same_device=same, cpu_fp32=cpu)
    prob += parity_problems(plan, "smoke")
    per_block = sum(sizes) / max(1, len(sizes) // 2)
    rep.update(ok=not prob, problems=prob, bytes_per_block=per_block,
               disk_forecast_gb=dict(
                   pilot=per_block * len(cells("pilot")) / 2 ** 30,
                   eval=per_block * len(cells("eval")) / 2 ** 30),
               tolerances=DBG_TOL,
               note="гейт — пересчёт головы под тем же fp16-autocast "
                    "харнесса; CPU float32 — описательно")
    save_json(os.path.join(OUT, "smoke_check.json"), rep)
    for lab in [c[0] for c in plan["cells"]]:
        if lab in rep:
            print(f"  smoke {lab}: голова (то же устройство) {rep[lab]['same_device']}"
                  f"; CPU fp32 {rep[lab]['cpu_fp32']}")
    print(f"  smoke: блок ≈ {per_block / 2 ** 20:.2f} МБ; диск пилот "
          f"{rep['disk_forecast_gb']['pilot']:.2f} ГБ, оценка "
          f"{rep['disk_forecast_gb']['eval']:.2f} ГБ — " +
          ("пройден" if not prob else "ОТКАЗ: " + "; ".join(prob[:3])))
    return 0 if not prob else 3


# --- пилот -----------------------------------------------------------------------------------
def sr_of(kind, plan, lab):
    succ = []
    for l_, t, b in plan["cells"]:
        if l_ != lab:
            continue
        d = json.load(open(block_path(kind, lab, t, b), encoding="utf-8"))
        succ += [bool(e["success"]) for e in d["episodes"]]
    return int(sum(succ)), len(succ)


def gate_decision(s, lat):
    """s: {24, 8, 16, 'R'} -> число успехов из 50; lat: {d: p50 мс} или
    None. -> (статус, детали)."""
    c1 = s[24] >= s["R"] - GATE["t24_vs_r"]
    early = {}
    for d in (8, 16):
        q = s[24] - s[d] <= GATE["early_vs_t24"]
        r = None if lat is None else lat[d] <= GATE["latency_ratio"] * lat[24]
        early[d] = dict(quality=q, latency=r)
    if not c1 or not any(v["quality"] for v in early.values()):
        return "stop", dict(t24_vs_r=c1, early=early)
    if lat is None:
        return "awaiting_latency", dict(t24_vs_r=c1, early=early)
    ok = any(v["quality"] and v["latency"] for v in early.values())
    return ("passed" if ok else "stop"), dict(t24_vs_r=c1, early=early)


GATE_FILE = os.path.join(OUT, "pilot_gate.json")
LAT_PILOT = os.path.join(OUT, "latency_pilot.json")


def _block_shas(kind, plan):
    out = {}
    for lab, t, b in plan["cells"]:
        f = block_path(kind, lab, t, b)
        for q in (f, f[:-5] + ".actions.npz"):
            out[q] = sha(q) if os.path.exists(q) else None
    return out


def _successes(plan):
    s = {}
    for lab, arm in plan["arms"].items():
        k, n = sr_of("pilot", plan, lab)
        if n != 50:
            raise SystemExit(f"{lab}: эпизодов {n} из 50")
        s["R" if arm["depth"] is None else str(arm["depth"])] = k
    return s


def _latency(plan, plan_file, lat_path=None):
    """-> (p50 B1 по глубинам или None, привязка, проблемы). Чужой замер —
    проблема, а не «нет замера»."""
    lat_path = lat_path or LAT_PILOT
    if not os.path.exists(lat_path):
        return None, None, []
    r = json.load(open(lat_path, encoding="utf-8"))
    p = []
    if r.get("plan_sha1") != sha(plan_file):
        p.append("замер задержки другого плана")
    want = plan["arms"][kp.label("parents_attn24", 0, 24)]["spec"]["ckpt_sha1"]
    if (r.get("ckpt_sha1") or {}).get("parents_attn24") != want:
        p.append("замер задержки другого checkpoint")
    try:
        lat = {str(d): float(r["batch1"][f"d{d}"]["total"]["median"])
               for d in k34m.EXITS}
    except (KeyError, TypeError):
        return None, None, p + ["замер задержки неполон"]
    if not all(np.isfinite(v) and v > 0 for v in lat.values()):
        p.append("замер задержки не конечен")
    return lat, dict(path=os.path.abspath(lat_path), sha1=sha(lat_path),
                     plan_sha1=r.get("plan_sha1")), p


def _decide(s, lat):
    si = {(int(k) if k != "R" else k): v for k, v in s.items()}
    li = None if lat is None else {int(k): v for k, v in lat.items()}
    return gate_decision(si, li)


def pilot_gate(a):
    plan = json.load(open(a.plan, encoding="utf-8"))
    prob = plan_problems(plan)
    if plan.get("plan_kind") != "pilot":
        prob.append("не план пилота")
    for lab, t, b in plan["cells"]:
        prob += block_problems(plan, a.plan, "pilot", lab, t, b)
    prob += parity_problems(plan, "pilot")
    lat, lat_bind, lp = _latency(plan, a.plan)
    prob += lp
    if prob:
        save_json(GATE_FILE, dict(status="technical_stop", problems=prob[:50],
                                  plan_sha1=sha(a.plan)))
        print("  пилот: ОТКАЗ — " + "; ".join(prob[:4]))
        return 3
    s = _successes(plan)
    status, det = _decide(s, lat)
    rec = dict(status=status, successes=s, latency_p50_b1=lat,
               latency=lat_bind, details=det, rule=GATE,
               plan=os.path.abspath(a.plan), plan_sha1=sha(a.plan),
               ckpt={lab: arm["spec"]["ckpt_sha1"]
                     for lab, arm in plan["arms"].items()},
               blocks=_block_shas("pilot", plan))
    save_json(GATE_FILE, rec)
    print(f"  пилот: успехи из 50 — T24 {s['24']}, T16 {s['16']}, T8 {s['8']}"
          f", R {s['R']}; задержка B1 p50 {lat}; статус {status}")
    return 0


def pilot_verify_problems(plan_file, gate_file=None, lat_path=None):
    """Отчёт гейта = текущим артефактам: план, checkpoint, все блоки,
    замер задержки; успехи и статус пересчитываются заново."""
    gate_file = gate_file or GATE_FILE
    if not os.path.exists(gate_file):
        return ["нет отчёта гейта"]
    g = json.load(open(gate_file, encoding="utf-8"))
    plan = json.load(open(plan_file, encoding="utf-8"))
    p = []
    if g.get("plan_sha1") != sha(plan_file):
        p.append("гейт другого плана")
    p += plan_problems(plan)
    if g.get("ckpt") != {lab: arm["spec"]["ckpt_sha1"]
                         for lab, arm in plan["arms"].items()}:
        p.append("checkpoint гейта ≠ плану")
    want = _block_shas("pilot", plan)
    if g.get("blocks") != want or any(v is None for v in want.values()):
        p.append("блоки пилота отсутствуют или изменены после гейта")
    lat, bind, lp = _latency(plan, plan_file, lat_path)
    p += lp
    if (g.get("latency") or None) != bind:
        p.append("замер задержки ≠ закреплённому в гейте")
    if p:
        return p
    s = _successes(plan)
    if g.get("successes") != s:
        p.append("успехи ≠ блокам")
    st, _d = _decide(s, lat)
    if g.get("status") != st:
        p.append(f"статус ≠ пересчитанному ({st})")
    if st != "passed":
        p.append(f"статус {st}")
    return p


def pilot_verify(a):
    p = pilot_verify_problems(a.plan)
    print("  гейт пилота: " + ("passed, привязки сходятся" if not p else
                               "ОТКАЗ — " + "; ".join(p[:4])))
    return 0 if not p else 3


# --- самопроверка -----------------------------------------------------------------------------
def gate_regression():
    """Гейт пилота: поддельный {"status": "passed"}, чужой замер, изменённый
    блок — отказ; полный проверенный набор — passed."""
    import tempfile
    g = globals()
    keep = {k: g[k] for k in ("OUT", "GATE_FILE", "LAT_PILOT",
                              "plan_problems", "block_problems",
                              "parity_problems")}
    with tempfile.TemporaryDirectory() as td:
        try:
            g.update(OUT=td, GATE_FILE=os.path.join(td, "gate.json"),
                     LAT_PILOT=os.path.join(td, "lat.json"))
            g["plan_problems"] = lambda plan: []
            g["block_problems"] = lambda *aa, **kw: []
            g["parity_problems"] = lambda *aa, **kw: []
            arms = {lab: dict(mode=m, seed=s_, depth=d,
                              spec=dict(ckpt_sha1=f"c{d}"))
                    for lab, m, s_, d in arms_for("pilot")}
            plan = dict(plan_kind="pilot", arms=arms,
                        cells=[list(c) for c in cells("pilot")])
            pf = os.path.join(td, "plan.json")
            save_json(pf, plan)
            succ = {kp.label("parents_attn24", 0, 24): 4,
                    kp.label("parents_attn24", 0, 16): 4,
                    kp.label("parents_attn24", 0, 8): 2,
                    kp.label("parents_attn24", 0): 4}
            for lab, t, b in plan["cells"]:
                f = block_path("pilot", lab, t, b)
                os.makedirs(os.path.dirname(f), exist_ok=True)
                np.savez(f[:-5] + ".actions.npz", a=np.zeros(1))
                save_json(f, dict(episodes=[dict(success=i < succ[lab],
                                                 init_state_id=b + i)
                                            for i in range(5)]))
            lat = dict(plan_sha1=sha(pf), ckpt_sha1=dict(parents_attn24="c24"),
                       batch1={f"d{d}": dict(total=dict(median=m_)) for d, m_
                               in ((8, 50.0), (16, 80.0), (24, 100.0))})
            save_json(g["LAT_PILOT"], lat)
            a_ = argparse.Namespace(plan=pf)
            assert pilot_gate(a_) == 0
            assert json.load(open(g["GATE_FILE"]))["status"] == "passed"
            assert pilot_verify_problems(pf) == [], pilot_verify_problems(pf)
            # поддельный отчёт гейта
            save_json(g["GATE_FILE"], dict(status="passed"))
            assert pilot_verify_problems(pf)
            assert pilot_gate(a_) == 0 and pilot_verify_problems(pf) == []
            # изменённый блок после гейта
            f0 = block_path("pilot", *plan["cells"][0])
            d0 = json.load(open(f0))
            d0["episodes"][0]["success"] = not d0["episodes"][0]["success"]
            save_json(f0, d0)
            assert pilot_verify_problems(pf)
            d0["episodes"][0]["success"] = not d0["episodes"][0]["success"]
            save_json(f0, d0)
            assert pilot_gate(a_) == 0 and pilot_verify_problems(pf) == []
            # замер другого плана: гейт отказывает, verify тоже
            save_json(g["LAT_PILOT"], dict(lat, plan_sha1="чужой"))
            assert pilot_gate(a_) == 3
            assert pilot_verify_problems(pf)
            # замер другого checkpoint
            save_json(g["LAT_PILOT"], dict(lat, ckpt_sha1=dict(
                parents_attn24="x")))
            assert pilot_gate(a_) == 3
            # замер подменён после гейта
            save_json(g["LAT_PILOT"], lat)
            assert pilot_gate(a_) == 0
            lat2 = copy.deepcopy(lat)
            lat2["batch1"]["d16"]["total"]["median"] = 90.0
            save_json(g["LAT_PILOT"], lat2)
            assert pilot_verify_problems(pf)
            # без замера — awaiting_latency, продолжение не допускается
            os.remove(g["LAT_PILOT"])
            assert pilot_gate(a_) == 0
            assert json.load(open(g["GATE_FILE"]))["status"] == \
                "awaiting_latency"
            assert pilot_verify_problems(pf)
        finally:
            g.update(keep)


def selftest():
    # расписание
    assert sorted(task_order()) == list(range(10))
    c = cells("pilot")
    assert len(c) == 40 and len(set(c)) == 40
    assert {x[0] for x in c} == {"parents_attn24_s0_d8",
                                 "parents_attn24_s0_d16",
                                 "parents_attn24_s0_d24",
                                 "parents_attn24_s0"}
    for t in range(10):
        assert sum(1 for x in c if x[1] == t) == 4
    first = [x[0] for x in c[:4]]
    second = [x[0] for x in c[4:8]]
    assert first[1:] + first[:1] == second                 # сдвиг рук
    ce = cells("eval")
    assert len(ce) == 480 and len(set(ce)) == 480
    assert len({x[0] for x in ce}) == 16
    assert len(cells("smoke")) == 3
    assert 40 * BLOCK_CAP == CAPS["pilot"] and \
        480 * BLOCK_CAP == CAPS["eval"]
    assert 3 * BLOCK_CAP <= CAPS["checks"]
    # правило пилота
    s = {24: 40, 16: 36, 8: 30, "R": 42}
    st, _d = gate_decision(s, None)
    assert st == "awaiting_latency"
    assert gate_decision(s, {8: 50, 16: 80, 24: 100})[0] == "passed"
    assert gate_decision(s, {8: 50, 16: 90, 24: 100})[0] == "stop"
    assert gate_decision({**s, "R": 46}, None)[0] == "stop"     # T24 хуже R
    assert gate_decision({**s, 16: 34}, None)[0] == "stop"
    assert gate_decision({**s, 8: 35}, {8: 80, 16: 90, 24: 100})[0] \
        == "passed"
    gate_regression()
    print("самопроверка k34_protocol пройдена: расписание (пилот 40 блоков, "
          "оценка 480, сдвиг рук по задачам, бюджеты = блоки × 3050), "
          "правило пилота (T24 к R, ранний выход к T24, задержка ≤ 0.85; без "
          "задержки — awaiting_latency)")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-34: протокол")
    ap.add_argument("cmd", choices=("prepare", "plan", "block", "cells",
                                    "smoke-check", "pilot-gate",
                                    "pilot-verify"))
    ap.add_argument("--kind", choices=KINDS)
    ap.add_argument("--plan")
    ap.add_argument("--op", choices=("reserve", "finish"))
    ap.add_argument("--label")
    ap.add_argument("--task", type=int)
    ap.add_argument("--start", type=int)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    try:
        return dict(prepare=prepare, plan=plan_cmd, block=block_cmd,
                    cells=cells_cmd, **{"smoke-check": smoke_check,
                                        "pilot-gate": pilot_gate,
                                        "pilot-verify": pilot_verify})[a.cmd](a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
