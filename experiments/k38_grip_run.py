#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-38, скрининг g1: early / late против архивного T1 (K-37 c2).

Команды (через k38_grip_check.py или напрямую):
  prepare --run-id g1      plan.json + plan.sha256; копии входных T1
  gpuinfo --device D
  assign --part N --host H --uuid U [--state …]
  preflight --device D [--reference]
  collect --part N --device D
  status | merge | analyze
Окружение: OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 обязательны (проверяются
вместе с torch.get_num_threads() == 8 и пишутся в отчёты).

Сцены: c2, задачи 0/6/8; в задаче — порядок (sha256("k38-grip-screen-v1|
{task}|{case_id}"), case_id), первые 20; ранг r -> часть r % 4 (15 сцен,
60 новых эпизодов). В части: сид 0, затем 1; сцены по (r, задача);
порядок сдвигов — (early, late) при чётном r + сид, иначе (late, early).
T1 — архив c2: на prepare координатор проверяет запись (load_ep, проверка
эпизода K-32, журнал части) и копирует JSON/NPZ в inputs/ с SHA256 в плане;
сбор сверяет копии с планом.
Новый эпизод early/late: базовый журнал K-32 (по исходным предложениям и
фактическим k) + журнал K-38: итоговый чанк, tau, «выдано». Проверка:
пересборка вмешательства из предложений (поиск tau, одна подстановка),
физические действия = масштабу итоговых чанков, общий префикс с T1 до
первой изменённой команды побитово; не дошедший сдвиг — эпизод = T1
целиком (действия и исход).
Допуск карты: нулевой сдвиг на двух закреплённых архивных эпизодах вне
скрининга (по сиду) — побитово = архиву (k36_harness.compare_eps); на
эталонной — ещё early и late на smoke-случае (префикс, tau, флаги).
Анализ: пары сцена × сид (120); W/L/D, 100·D/120 п.п.; bootstrap по
сценам внутри задачи (20), 50 000, seed 38001, общие выборки. Правило:
D ≥ 6, D_s > 0 на обоих сидах, ≥ 4 сцены с d_s0 + d_s1 > 0.
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k31_collect as kcol  # noqa: E402
import k36_bank as kb36  # noqa: E402
import k36_harness as kh  # noqa: E402
import k36_protocol as pc  # noqa: E402
import k37_confirm as kcf  # noqa: E402
import k38_grip_check as kgc  # noqa: E402
import k38_grip_operator as kg  # noqa: E402

PROTOCOL = "k38g-v1"
RUNS = "reports/k38/runs"
TASKS = (0, 6, 8)
SEEDS = (0, 1)
KINDS = ("early", "late")
N_PER_TASK = 20
N_PARTS = 4
EP = 610
CAP_PART = dict(screen=62 * EP)
CAP_PF = dict(checks=7 * EP)
THREADS = 8
BOOT = dict(n=50_000, seed=38001)
RULE = dict(D_min=6, per_seed_gt=0, scenes_min=4)
STAGES = dict(screen="screen", pfzero="checks", pfsmoke="checks")
CODE = ("k38_grip_run.py", "k38_grip_check.py", "k38_grip_operator.py",
        "k37_confirm.py", "k36b_gate.py", "k36_harness.py", "k36_bank.py",
        "k36_protocol.py", "k32_policy.py", "k32_operator.py",
        "k32_harness.py", "k31_collect.py", "k33_bank.py")
OK, TECH, BUSY = 0, 3, 4


def sha256(p):
    return kcf.sha256(p)


def scene_key(task, cid):
    return (hashlib.sha256(f"k38-grip-screen-v1|{task}|{cid}".encode())
            .hexdigest(), cid)


def select_scenes(scenes):
    """c2 scenes -> [dict(…, rank, part)] по 20 на задачу."""
    out = []
    for t in TASKS:
        cs = sorted((s for s in scenes if s["task"] == t),
                    key=lambda s: scene_key(t, s["case_id"]))[:N_PER_TASK]
        if len(cs) != N_PER_TASK:
            raise SystemExit(f"t{t}: сцен {len(cs)} < {N_PER_TASK}")
        for r, s in enumerate(cs):
            out.append(dict(s, rank=r, part=r % N_PARTS))
    return out


def kind_order(rank, seed):
    return KINDS if (rank + seed) % 2 == 0 else KINDS[::-1]


def part_order(sel, part):
    """-> [(seed, scene, kind)] в порядке исполнения части."""
    scs = sorted((s for s in sel if s["part"] == part),
                 key=lambda s: (s["rank"], TASKS.index(s["task"])))
    return [(sd, s, k) for sd in SEEDS for s in scs
            for k in kind_order(s["rank"], sd)]


def install(out_dir):
    kh.OUT = out_dir
    kh.REJECTED = os.path.join(out_dir, "rejected")
    kh.STAGE_SCOPE.update(STAGES)


def g_want(plan_sha, k36, stage, kind, s, sc):
    """kind ∈ early / late / zero (нулевой сдвиг для допуска)."""
    kh.STAGE_SCOPE.update(STAGES)
    pl = dict(k36, sha1=plan_sha)
    w = kh.want_of(pl, stage, f"T32_s{s}", sc["task"], sc["case_id"],
                   sc["pos"], "bank", space=sc.get("space", 2),
                   case_id=sc["case_id"],
                   want_hash=None if sc["hash"] is None else
                   tuple(sc["hash"]))
    lab = f"{kind}_s{s}"
    return dict(w, label=lab, kind=f"G{kind}",
                actor_hash=f"{w['actor_hash']}+grip-{kind}-v1",
                eid=f"{w['scope']}/{stage}/{lab}/t{sc['task']}/"
                    f"{sc['case_id']}")


# --- журнал K-38 и проверки -------------------------------------------------------------
def save_g(want, ep):
    import k32_harness as kh32
    jp, np_ = kh.ep_paths(want["stage"], want["label"], want["task"],
                          want["key"])
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    calls = [dict(c, rec=dict(c["rec"], a_codec=c["rec"]["base_a_codec"]))
             for c in ep["calls"]]
    J = kh32.pack_journal([(0, calls)], 6)
    n = len(ep["calls"])
    K = dict(a2=np.stack([np.asarray(c["a_codec"], np.float32)
                          for c in ep["calls"]]) if n else
             np.zeros((0, 8, 7), np.float32),
             tau=np.array([int(c["rec"].get("k38_tau", -1))
                           for c in ep["calls"]], np.int64),
             issued=np.array([bool(c["rec"].get("k38_issued", False))
                              for c in ep["calls"]]))
    with open(np_ + ".tmp", "wb") as fh:
        np.savez_compressed(fh, actions=ep["actions"].astype(np.float32),
                            **{f"j_{k}": v for k, v in J.items()},
                            **{f"k_{k}": v for k, v in K.items()})
    os.replace(np_ + ".tmp", np_)
    pc.save_json(jp, dict({k: want[k] for k in kh.IDENT},
                          want_hash=want["want_hash"], npz_sha1=kh.sha(np_),
                          phys=int(ep["physical"]),
                          seconds=float(ep["seconds"]),
                          summary={k: ep[k] for k in kh.SUMMARY}))
    return int(ep["physical"])


def replay_expected(J, kind):
    """Пересборка из исходных предложений: -> (итоговые чанки, событие)."""
    prev, used, exp, ev = kg.PREV0, False, [], None
    for i in range(len(J["elapsed"])):
        el, k = int(J["elapsed"][i]), int(J["k"][i])
        a = np.asarray(J["a_codec"][i], np.float32)
        out = a
        if not used:
            tau = kg.find_event(kg.phys_grip(a), prev, kg.horizon(el))
            if tau is not None:
                used = True
                ev = dict(call=i + 1, elapsed=el, tau=int(tau), k=k)
                if kind in KINDS:
                    ev["first"] = kg.first_changed(kind, tau)
                    out = kg.substitute(a, tau, kind)
        exp.append(out)
        prev = float(kg.phys_grip(out)[k - 1])
    return (np.stack(exp) if exp else np.zeros((0, 8, 7), np.float32)), ev


def intervention_problems(J, K, A, kind, q):
    """Журнал K-38 = пересборке; исполненное = масштабу итоговых чанков."""
    import k29_policy as kp
    p = []
    n = len(J["elapsed"])
    if any(len(K.get(x, [])) != n for x in ("a2", "tau", "issued")):
        return ["журнал K-38 неполон"], None
    exp, ev = replay_expected(J, kind)
    if not np.array_equal(exp.astype(np.float32),
                          np.asarray(K["a2"], np.float32)):
        p.append("итоговые чанки ≠ пересборке вмешательства")
    if int(np.sum(K["issued"])) > 1:
        p.append("больше одного вмешательства")
    want_tau = np.full(n, -1)
    want_iss = np.zeros(n, bool)
    if ev is not None:
        want_tau[ev["call"] - 1] = ev["tau"]
        want_iss[ev["call"] - 1] = True
    if not (np.array_equal(K["tau"], want_tau) and
            np.array_equal(K["issued"], want_iss)):
        p.append("tau/«выдано» ≠ пересборке")
    if not np.array_equal(np.asarray(K["a2"])[:, :, :6],
                          np.asarray(J["a_codec"])[:, :, :6]):
        p.append("каналы руки ≠ предложению T1")
    ex = np.concatenate([kp.scale_exec(np.asarray(K["a2"][i])[None], q)[0]
                         [:int(J["k"][i])] for i in range(n)]) if n else \
        np.zeros((0, 7), np.float32)
    if not np.array_equal(ex.astype(np.float32), np.asarray(A, np.float32)):
        p.append("исполненное ≠ масштабу итоговых чанков")
    return p, ev


def prefix_problems(z, rec, ref, ev):
    """Префикс до первой изменённой команды = архивному T1; не дошедший
    сдвиг — эпизод = T1 целиком."""
    p = []
    A = np.asarray(z["actions"], np.float32)
    rz, rr = ref[1], ref[0]
    A1 = np.asarray(rz["actions"], np.float32)
    J1 = {k[2:]: v for k, v in rz.items() if k.startswith("j_")}
    _e, ev1 = replay_expected(J1, "zero")
    P = len(A) if ev is None else ev["elapsed"] + ev["first"]
    reached = ev is not None and ev["first"] < ev["k"]
    if (ev is None) != (ev1 is None) or (ev is not None and
            any(ev[k] != ev1[k] for k in ("call", "elapsed", "tau"))):
        p.append("событие ≠ событию архивного T1")
    elif ev is not None:
        c = ev["call"] - 1
        if not np.array_equal(z["j_a_codec"][c], J1["a_codec"][c]):
            p.append("предложение на вызове события ≠ архиву")
        if reached != (ev["first"] < ev1["k"]):
            p.append("достижение первой изменённой команды ≠ архиву")
    m = min(P, len(A))
    if not np.array_equal(A[:m], A1[:m]):
        p.append("общий префикс ≠ архивному T1")
    if not reached and (not np.array_equal(A, A1) or
                         rec["summary"] != rr["summary"]):
        p.append("сдвиг не исполнен, но эпизод ≠ T1")
    return p


def g_problems(rec, z, want, C, ref):
    import k29_policy as kp
    import k32_harness as kh32
    p = []
    bad = [k for k in kh.IDENT if rec.get(k) != want[k]]
    if bad:
        return [f"идентификаторы {bad}"]
    s = rec["summary"]
    wh = want["want_hash"]
    if wh is not None and (s["init_hash"] != wh[0] or
                           s["init_hash_full"] != wh[1]):
        p.append("начальный хеш ≠ банку")
    A = z.get("actions")
    if A is None or kcol.actions_sha(A) != s["actions_sha1"]:
        return p + ["действия ≠ SHA"]
    J = {k[2:]: v for k, v in z.items() if k.startswith("j_")}
    K = {k[2:]: v for k, v in z.items() if k.startswith("k_")}
    if not J or len(J.get("ep", [])) == 0:
        return p + ["журнал пуст"]
    q = C["q"]
    n = len(J["ep"])
    A_base = np.concatenate([kp.scale_exec(J["a_codec"][i][None], q)[0]
                             [:int(J["k"][i])] for i in range(n)])
    p += [f"база: {x}" for x in kh32.journal_problems(
        J, [A_base], [s["done_step"]], C["half"], C["mid"], C["L"], C["U"],
        q, "T")]
    kind = want["kind"][1:]
    ip, ev = intervention_problems(J, K, A, kind, q)
    p += ip
    p += kh.summary_problems(rec, s, len(A))
    if ref is not None and not ip:
        p += prefix_problems(z, rec, ref, ev)
    return p


def load_ref(rr, plan, s, sc):
    """Копия архивного T1 из inputs/ со сверкой SHA256 плана."""
    e = plan["refs"][f"{s}|{sc['case_id']}"]
    jp, np_ = os.path.join(rr, e["json"]), os.path.join(rr, e["npz"])
    if sha256(jp) != e["json_sha256"] or sha256(np_) != e["npz_sha256"]:
        raise SystemExit(f"копия T1 {sc['case_id']} s{s} ≠ плану")
    return pc.load_json(jp), dict(np.load(np_))


# --- план -------------------------------------------------------------------------------------
def code_shas():
    return {f: sha256(os.path.join(HERE, f)) for f in CODE}


def archive_ep(c2rr, c2, k36, case_part, s, sc, C):
    """Проверенная архивная запись T1 c2 -> (rec, z, json, npz)."""
    k = case_part[sc["case_id"]]
    pd = kcf.part_dir(c2rr, k)
    kcf.install(pd)
    w = kcf.make_want(c2["sha256"], k36, c2["heads"], "confirm", "T1", s,
                      sc, 2)
    got = kh.load_ep(w)
    if got is None:
        raise SystemExit(f"архив: нет {w['eid']}")
    pr = kh.episode_problems(*got, w, C)
    led = pc.load_json(os.path.join(pd, "ledger.json"))
    dn = led["done"].get(w["eid"])
    if (pr or led.get("caps") != kcf.CAP_PART or
            led.get("pending") is not None or not isinstance(dn, dict) or
            dn.get("scope") != w["scope"] or type(dn.get("actual")) is not int or
            dn["actual"] != got[0]["phys"]):
        raise SystemExit(f"архив {w['eid']}: {pr[:2]} / журнал {dn}")
    jp, np_ = kh.ep_paths("confirm", f"T32_s{s}", sc["task"],
                          sc["case_id"])
    return got, jp, np_


def make_plan(run_id, c2rr):
    rr = os.path.join(RUNS, run_id)
    c2, k36 = kcf.load_plan(c2rr)
    aud = pc.load_json(os.path.join(kgc.OUT, "audit.json"))
    if aud.get("status") != "coverage_ok" or \
            aud.get("plan_sha256") != c2["sha256"]:
        raise SystemExit("аудит K-38 не coverage_ok или от другого плана")
    C = kh.consts(k36)
    case_part = {sc["case_id"]: int(k) for k, v in c2["parts"].items()
                 for sc in v["scenes"]}
    sel = select_scenes(c2["scenes"])
    chosen = {sc["case_id"] for sc in sel}
    refs = {}

    def put(s, sc, tag):
        got, jp, np_ = archive_ep(c2rr, c2, k36, case_part, s, sc, C)
        rel_j = os.path.join("inputs", f"T32_s{s}",
                             f"t{sc['task']}_{sc['case_id']}.json")
        rel_n = rel_j[:-5] + ".npz"
        for src, rel in ((jp, rel_j), (np_, rel_n)):
            dst = os.path.join(rr, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copyfile(src, dst + ".tmp")
                os.replace(dst + ".tmp", dst)
        ref = dict(
            eid=got[0]["eid"], json=rel_j, npz=rel_n,
            json_sha256=sha256(os.path.join(rr, rel_j)),
            npz_sha256=sha256(os.path.join(rr, rel_n)),
            src_json_sha256=sha256(jp), src_npz_sha256=sha256(np_),
            success=bool(got[0]["summary"]["success"]), tag=tag)
        for typ in ("json", "npz"):
            if ref[f"{typ}_sha256"] != ref[f"src_{typ}_sha256"]:
                raise SystemExit(f"копия {ref[typ]} ≠ источнику")
        refs[f"{s}|{sc['case_id']}"] = ref
        return got
    for sc in sel:
        for s in SEEDS:
            put(s, sc, "screen")
    # допуск: по сиду один архивный T1 вне скрининга (по хешу); smoke —
    # первый вне скрининга с k > tau + 2 на вызове события
    rows = {r["eid"]: r for r in aud["rows"]}
    rest = [sc for sc in c2["scenes"] if sc["case_id"] not in chosen]

    def h(tag, s, sc):
        return hashlib.sha256(f"k38-grip-{tag}-v1|{s}|{sc['case_id']}"
                              .encode()).hexdigest()
    pf_zero = []
    for s in SEEDS:
        sc = min(rest, key=lambda x: h("pf", s, x))
        put(s, sc, "pf_zero")
        pf_zero.append(dict(seed=s, scene=sc))
    smoke = None
    cands = sorted(((s, sc) for s in SEEDS for sc in rest),
                   key=lambda x: h("smoke", *x))
    for strict in (True, False):
        for s, sc in cands:
            w = kcf.make_want(c2["sha256"], k36, c2["heads"], "confirm",
                              "T1", s, sc, 2)
            ev = rows.get(w["eid"], {}).get("ev")
            if not ev:
                continue
            if (strict and ev["k"] > ev["tau"] + 2) or \
                    (not strict and ev["early"]):
                smoke = dict(seed=s, scene=sc, ev=ev,
                             expect=dict(early=ev["tau"] - 2 < ev["k"],
                                         late=ev["tau"] < ev["k"]))
                break
        if smoke:
            break
    if smoke is None:
        raise SystemExit("нет smoke-случая")
    put(smoke["seed"], smoke["scene"], "pf_smoke")
    parts = {}
    for k in range(N_PARTS):
        order = part_order(sel, k)
        parts[str(k)] = dict(
            scenes=[sc["case_id"] for sc in sel if sc["part"] == k],
            eids=[g_want("X", k36, "screen", kind, s, sc)["eid"]
                  for s, sc, kind in order])
    cover = {}
    for t in TASKS:
        rs = [rows[kcf.make_want(c2["sha256"], k36, c2["heads"], "confirm",
                                 "T1", s, sc, 2)["eid"]]
              for sc in sel if sc["task"] == t for s in SEEDS]
        cover[str(t)] = dict(pairs=len(rs), t1_fail=sum(not r["success"]
                                                        for r in rs),
                             fail_reached=sum(not r["success"] and bool(
                                 r["ev"]) and r["ev"]["early"] for r in rs),
                             with_event=sum(bool(r["ev"]) for r in rs))
    k32 = k36["k32"]
    return dict(
        kind="k38_grip_plan", run_id=run_id, protocol=PROTOCOL,
        code=code_shas(), c2=dict(plan_sha256=c2["sha256"], run_root=c2rr,
                                  bank=c2["bank"]),
        audit_sha256=sha256(os.path.join(kgc.OUT, "audit.json")),
        k36_prepare_sha1=k36["sha1"],
        k32=dict(contract=k32["k32_contract"], seeds={str(s): dict(
            source_weights_hash=k32["seeds"][str(s)]["source"][
                "weights_hash"],
            T_actor_hash=k32["seeds"][str(s)]["T"]["actor_hash"])
            for s in SEEDS}),
        operator=dict(version="k38-grip-v1", shift=2, kinds=list(KINDS)),
        scenes=sel, parts=parts, refs=refs, pf_zero=pf_zero, smoke=smoke,
        coverage_selected=cover, caps=dict(part=CAP_PART, preflight=CAP_PF,
                                           reserve=EP),
        resource=dict(kcf.RES, preflight_need_gb=10.0, threads=THREADS),
        env=dict(B=1, seeds=list(SEEDS), exec_seed=101, waiting_steps=10,
                 max_steps=600, horizon=8),
        analysis=dict(BOOT, rule=RULE, unit="scene (both seeds)",
                      task_weights="equal"))


def plan_path(rr):
    return os.path.join(rr, "plan.json")


def prepare(a):
    rr = os.path.join(RUNS, a.run_id)
    shp = os.path.join(rr, "plan.sha256")
    new = make_plan(a.run_id, a.archive)
    sh = kcf.canon_sha(new)
    if os.path.exists(plan_path(rr)) or os.path.exists(shp):
        if not (os.path.exists(plan_path(rr)) and os.path.exists(shp)):
            raise SystemExit(f"{rr}: план или plan.sha256 отсутствует")
        old = json.load(open(plan_path(rr), encoding="utf-8"))
        if not (kcf.canon_sha(old) == open(shp).read().strip() == sh):
            raise SystemExit(f"{plan_path(rr)} закреплён иначе")
        print(f"  план закреплён, совпадает: {sh[:16]}")
        return OK
    pc.save_json(plan_path(rr), new)
    with open(shp, "w") as fh:
        fh.write(sh + "\n")
    pc.save_json(os.path.join(rr, "plan_meta.json"), dict(
        host=socket.gethostname(), time=time.strftime("%Y-%m-%dT%H:%M:%S")))
    cv = new["coverage_selected"]
    print(f"  план {plan_path(rr)}: sha256 {sh[:16]}; сцен "
          f"{len(new['scenes'])}, новых эпизодов "
          f"{sum(len(v['eids']) for v in new['parts'].values())}, частей "
          f"{N_PARTS}; smoke {new['smoke']['scene']['case_id']} s"
          f"{new['smoke']['seed']} (tau {new['smoke']['ev']['tau']}, k "
          f"{new['smoke']['ev']['k']})")
    print("  покрытие среди 120 пар: " + "; ".join(
        f"t{t}: неудач T1 {v['t1_fail']}, из них с исполнимым сдвигом "
        f"{v['fail_reached']}; с событием {v['with_event']}/{v['pairs']}"
        for t, v in cv.items()))
    return OK


def load_plan(rr, verify=True):
    p = json.load(open(plan_path(rr), encoding="utf-8"))
    sh = kcf.canon_sha(p)
    if sh != open(os.path.join(rr, "plan.sha256")).read().strip():
        raise SystemExit("plan.json ≠ plan.sha256")
    k36 = pc.load_prepare() if verify else None
    if verify:
        bad = [f for f, s in p["code"].items() if code_shas().get(f) != s]
        if bad:
            raise SystemExit(f"код ≠ плану: {bad}")
        if k36["sha1"] != p["k36_prepare_sha1"]:
            raise SystemExit("prepare K-36 ≠ плану")
        for f, s in p["c2"]["bank"]["files"].items():
            if not os.path.exists(f) or sha256(f) != s:
                raise SystemExit(f"банк {f} ≠ плану")
        for key, e in p["refs"].items():
            for x in ("json", "npz"):
                f = os.path.join(rr, e[x])
                if not os.path.exists(f) or sha256(f) != e[f"{x}_sha256"]:
                    raise SystemExit(f"копия T1 {key} ≠ плану")
    p["sha256"] = sh
    return p, k36


# --- ресурсы, потоки, назначения ---------------------------------------------------------------
def threads_problems(torch):
    p = []
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(v) != str(THREADS):
            p.append(f"{v}={os.environ.get(v)} (нужно {THREADS})")
    if torch.get_num_threads() != THREADS:
        p.append(f"torch.get_num_threads() = {torch.get_num_threads()}")
    return p


def env_record(torch):
    return dict(kcf.env_versions(), torch_threads=torch.get_num_threads(),
                omp=os.environ.get("OMP_NUM_THREADS"),
                mkl=os.environ.get("MKL_NUM_THREADS"))


def part_dir(rr, k):
    return os.path.join(rr, f"part-{int(k):02d}")


def assign(a):
    lk = os.path.join(a.run_root, ".assignments.lock")
    import fcntl
    with open(lk, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            plan, _k = load_plan(a.run_root, verify=False)
            path = os.path.join(a.run_root, "assignments.json")
            d = json.load(open(path)) if os.path.exists(path) else \
                dict(history=[])
            k = int(a.part)
            if not 0 <= k < N_PARTS:
                raise SystemExit(f"часть {k}")
            last = next((h for h in reversed(d["history"])
                         if h["part"] == k), None)
            st = a.state
            if st == "assigned" and last and last["state"] == "assigned" \
                    and (last["host"], last["uuid"]) != (a.host, a.uuid):
                raise SystemExit(f"part {k} уже назначен {last['host']}")
            if st != "assigned" and (not last or last["state"] !=
                                     "assigned"):
                raise SystemExit(f"part {k}: нет действующего назначения")
            n = 1 + sum(1 for h in d["history"] if h["part"] == k)
            rec = dict(part=k, host=a.host if st == "assigned" else
                       last["host"], uuid=a.uuid if st == "assigned" else
                       last["uuid"], n=n, state=st,
                       plan_sha256=plan["sha256"],
                       time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            d["history"].append(rec)
            pc.save_json(path, d)
            if st in ("assigned", "done"):
                pc.save_json(os.path.join(part_dir(a.run_root, k),
                                          "assignment.json"), rec)
            print(f"  part {k}: {st} -> {rec['host']} {rec['uuid']} (№ {n})")
            return OK
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def pf_dir(rr, host, uuid):
    return os.path.join(rr, "preflight", f"{host}__{uuid}")


def admitted(rr, plan, uuid, torch):
    p = os.path.join(pf_dir(rr, socket.gethostname(), uuid), "report.json")
    if not os.path.exists(p):
        raise SystemExit(f"карта {uuid} не прошла preflight K-38")
    r = pc.load_json(p)
    if r.get("admitted") is not True or r.get("plan_sha256") != \
            plan["sha256"]:
        raise SystemExit("preflight: не допущена или другой план")
    if r.get("env") != env_record(torch):
        raise SystemExit("preflight: версии среды или потоки изменились")
    return r


def make_arm(torch, k36, s, v, kind):
    return kg.GripArm(kh.make_arm(torch, k36, f"T32_s{s}", v), kind)


def run_one(w, L, sim, io, arm, val, allowed, disk_path, woe, saver, *,
            reference_eid):
    import k36b_gate as kgt
    if kh.load_ep(w) is None and not L.is_done(w["eid"]):
        kcf.check_disk(disk_path)

    def run_save():
        arm.episode = kh.episode_number(reference_eid)
        ep = kcol.run_episode(sim, io, arm, w["task"], w["reset_id"])
        return saver(w, ep)
    return kgt.transact(w, L, run_save, val, allowed, woe)


def zero_validator(C):
    """Нулевой сдвиг: идентификаторы — по want (kind Gzero), журнал и
    сводка — штатная проверка T32 (kind T)."""
    def val(r, z, w_):
        bad = [k for k in kh.IDENT if r.get(k) != w_[k]]
        if bad:
            return [f"идентификаторы {bad}"]
        return kh.episode_problems(dict(r, kind="T"), z, dict(w_, kind="T"),
                                   C)
    return val


def record_validator(rr, plan, C):
    check_zero = zero_validator(C)

    def validate(rec, z, w):
        if w["kind"] == "Gzero":
            return check_zero(rec, z, w)
        ref = load_ref(rr, plan, w["seed"], w)
        return g_problems(rec, z, w, C, ref)

    return validate


def zero_saver(w, ep):
    kh.save_ep(w, ep)
    return int(ep["physical"])


def preflight(a, sim=None, src_loader=None, gpu_runner=None,
              torch_check=True):
    import torch
    rr = a.run_root
    plan, k36 = load_plan(rr)
    tp = threads_problems(torch)
    if tp:
        raise SystemExit("потоки: " + "; ".join(tp))
    idx = kcf.dev_index(a.device)
    g = kcf.gpu_query(idx, gpu_runner)
    host = socket.gethostname()
    d = pf_dir(rr, host, g["uuid"])
    lk = kcf.Lock(os.path.join(kcf.lock_dir(), g["uuid"] + ".lock"),
                  f"карта {g['uuid']}", kcf.Busy)
    try:
        kcf.check_gpu(idx, plan["resource"]["preflight_need_gb"], gpu_runner)
        kcf.check_disk(rr)
        refp = os.path.join(rr, "preflight", "reference.json")
        if not a.reference and not os.path.exists(refp):
            raise SystemExit("нет эталонной карты: preflight --reference")
        install(d)
        C = kh.consts(k36)
        L = kcol.Ledger(os.path.join(d, "ledger.json"), CAP_PF)
        val = record_validator(rr, plan, C)
        sim = sim or kb36.BankSim("bank", 2, bank_dir=plan["c2"]["bank"]
                                  ["dir"])
        jobs = [(x["seed"], x["scene"], "zero") for x in plan["pf_zero"]]
        if a.reference:
            sm = plan["smoke"]
            jobs += [(sm["seed"], sm["scene"], k) for k in KINDS]
        W = {g_want(plan["sha256"], k36, "pfzero" if k == "zero" else
                    "pfsmoke", k, s, sc)["eid"]:
             g_want(plan["sha256"], k36, "pfzero" if k == "zero" else
                    "pfsmoke", k, s, sc) for s, sc, k in jobs}
        prob, binding = [], None
        try:
            if torch_check:
                binding = kcf.bind_device(torch, idx, g)
                torch.cuda.reset_peak_memory_stats(idx)
            for s in SEEDS:
                js = [j for j in jobs if j[0] == s]
                if not js:
                    continue
                kcf.check_gpu(idx, plan["resource"]["preflight_need_gb"],
                              gpu_runner)
                v = kh.load_source(torch, s, a.device, src_loader)
                for _s, sc, kind in js:
                    w = g_want(plan["sha256"], k36, "pfzero" if kind ==
                               "zero" else "pfsmoke", kind, s, sc)
                    sv = zero_saver if kind == "zero" else save_g
                    base_eid = plan["refs"][f"{s}|{sc['case_id']}"]["eid"]
                    got = run_one(w, L, sim, v["io"], make_arm(
                        torch, k36, s, v, kind), val,
                        [("checks", "checks/")], d, W.__getitem__, sv,
                        reference_eid=base_eid)
                    if kind == "zero":
                        pr = kh.compare_eps(load_ref(rr, plan, s, sc), got)
                        prob += [f"нулевой сдвиг {sc['case_id']} s{s}: {x}"
                                 for x in pr]
                    else:
                        _ip, ev = intervention_problems(
                            {k[2:]: v_ for k, v_ in got[1].items()
                             if k.startswith("j_")},
                            {k[2:]: v_ for k, v_ in got[1].items()
                             if k.startswith("k_")},
                            got[1]["actions"], kind, C["q"])
                        sm = plan["smoke"]
                        if ev is None or ev["tau"] != sm["ev"]["tau"] or \
                                ev["call"] != sm["ev"]["call"]:
                            prob.append(f"smoke {kind}: событие {ev} ≠ "
                                        f"ожидаемому {sm['ev']}")
                        else:
                            reached = ev["first"] < ev["k"]
                            if reached != sm["expect"][kind]:
                                prob.append(f"smoke {kind}: исполнение "
                                            f"{reached} ≠ ожидаемому")
                v = None
                kh.free(torch)
        except torch.cuda.OutOfMemoryError as e:
            raise kcf.Busy(f"CUDA OOM: {e}")
        finally:
            kcf.close_sim(sim)
        secs = [float(kh.load_ep(w)[0]["seconds"]) for w in W.values()]
        peak = torch.cuda.max_memory_reserved(idx) / 2 ** 30 \
            if torch_check else 0.0
        rep = dict(plan_sha256=plan["sha256"], host=host, uuid=g["uuid"],
                   gpu=g, binding=binding, env=env_record(torch),
                   peak_gb=float(peak), median_s=float(np.median(secs)),
                   n_episodes=len(W), spent=L.spent("checks"),
                   crashes=len(L.d["crashes"]), problems=prob,
                   admitted=not prob, reference=bool(a.reference),
                   time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        pc.save_json(os.path.join(d, "report.json"), rep)
        if a.reference and not prob:
            pc.save_json(refp, dict(dir=d, host=host, uuid=g["uuid"]))
        print(f"  preflight K-38 {host} {a.device}: эпизодов {len(W)}, "
              f"медиана {rep['median_s']:.1f} с, пик {peak:.1f} ГиБ, потоки "
              f"{rep['env']['torch_threads']}; " +
              ("ДОПУЩЕНА" if not prob else "НЕ допущена: " +
               "; ".join(prob[:3])))
        return OK if not prob else TECH
    finally:
        lk.release()


# --- сбор, статус, слияние, анализ --------------------------------------------------------------
def part_wants(plan, k36, k):
    by = {sc["case_id"]: sc for sc in plan["scenes"]}
    sel = [dict(by[c]) for c in plan["parts"][str(k)]["scenes"]]
    order = part_order([dict(s, part=k) for s in sel], k)
    return [(s, sc, kind, g_want(plan["sha256"], k36, "screen", kind, s, sc))
            for s, sc, kind in order]


def collect(a, sim=None, src_loader=None, gpu_runner=None, torch_check=True):
    import torch
    rr, k = a.run_root, int(a.part)
    plan, k36 = load_plan(rr)
    tp = threads_problems(torch)
    if tp:
        raise SystemExit("потоки: " + "; ".join(tp))
    pd = part_dir(rr, k)
    ap = os.path.join(pd, "assignment.json")
    if not os.path.exists(ap):
        raise SystemExit(f"part {k}: нет назначения")
    asg = pc.load_json(ap)
    idx = kcf.dev_index(a.device)
    g = kcf.gpu_query(idx, gpu_runner)
    if asg.get("plan_sha256") != plan["sha256"] or asg.get("part") != k or \
            asg.get("host") != socket.gethostname() or \
            asg.get("uuid") != g["uuid"] or asg.get("state") != "assigned":
        raise SystemExit(f"part {k}: назначение не этому хосту/карте/плану")
    locks = [kcf.Lock(os.path.join(pd, ".lock"), f"part {k}", SystemExit)]
    try:
        locks.append(kcf.Lock(os.path.join(kcf.lock_dir(),
                                           g["uuid"] + ".lock"),
                              f"карта {g['uuid']}", kcf.Busy))
        adm = admitted(rr, plan, g["uuid"], torch)
        need = max(kcf.RES["min_free_gb"], adm["peak_gb"] +
                   kcf.RES["margin_gb"])
        kcf.check_gpu(idx, need, gpu_runner)
        kcf.check_disk(pd)
        install(pd)
        C = kh.consts(k36)
        L = kcol.Ledger(os.path.join(pd, "ledger.json"), CAP_PART)
        val = record_validator(rr, plan, C)
        PW = part_wants(plan, k36, k)
        W = {w["eid"]: w for _s, _sc, _k, w in PW}
        sim = sim or kb36.BankSim("bank", 2,
                                  bank_dir=plan["c2"]["bank"]["dir"])
        try:
            if torch_check:
                kcf.bind_device(torch, idx, g)
            for s in SEEDS:
                kcf.check_gpu(idx, need, gpu_runner)
                v = kh.load_source(torch, s, a.device, src_loader)
                arms = {kind: make_arm(torch, k36, s, v, kind)
                        for kind in KINDS}
                for s_, sc, kind, w in PW:
                    if s_ != s:
                        continue
                    base_eid = plan["refs"][f"{s}|{sc['case_id']}"]["eid"]
                    run_one(w, L, sim, v["io"], arms[kind], val,
                            [("screen", "screen/")], pd, W.__getitem__, save_g,
                            reference_eid=base_eid)
                    n = sum(1 for e in W if L.is_done(e))
                    if n % 10 == 0:
                        print(f"  part {k}: {n}/{len(W)}; расход "
                              f"{L.spent('screen')}", flush=True)
                arms = v = None
                kh.free(torch)
        except torch.cuda.OutOfMemoryError as e:
            raise kcf.Busy(f"CUDA OOM: {e}")
        finally:
            kcf.close_sim(sim)
        rep = part_problems(plan, k36, rr, k, C)
        pc.save_json(os.path.join(pd, "part_report.json"), rep)
        print(f"  part {k}: " + ("готова, проверка пройдена" if rep["ok"]
                                 else "ПРОБЛЕМЫ: " +
                                 "; ".join(rep["problems"][:3])))
        return OK if rep["ok"] else TECH
    finally:
        for lk in reversed(locks):
            lk.release()


def part_problems(plan, k36, rr, k, C):
    pd = part_dir(rr, k)
    install(pd)
    lp = os.path.join(pd, "ledger.json")
    if not os.path.exists(lp):
        return dict(ok=False, problems=["нет журнала"], part=k)
    led = pc.load_json(lp)
    if led.get("caps") != CAP_PART:
        return dict(ok=False, problems=["другие лимиты журнала"], part=k)
    p = []
    PW = part_wants(plan, k36, k)
    W = {w["eid"]: (s, sc, kind, w) for s, sc, kind, w in PW}
    if led["pending"]:
        p.append(f"открытый резерв {led['pending']['id']}")
    if set(led["done"]) != set(W):
        p.append(f"набор eid журнала ≠ плану ({len(led['done'])} из "
                 f"{len(W)})")
    tot, rows = 0, []
    for e, (s, sc, kind, w) in W.items():
        dn = led["done"].get(e)
        if not dn:
            continue
        try:
            got = kh.load_ep(w)
        except SystemExit as ex:
            p.append(f"{e}: {ex}")
            continue
        if got is None:
            p.append(f"{e}: учтён, файла нет")
            continue
        ref = load_ref(rr, plan, s, sc)
        pr = g_problems(*got, w, C, ref)
        if dn["scope"] != "screen" or dn["actual"] != got[0]["phys"]:
            pr.append(f"журнал {dn} ≠ записи")
        p += [f"{e}: {x}" for x in pr]
        tot += int(dn["actual"])
        K = {kk[2:]: v for kk, v in got[1].items() if kk.startswith("k_")}
        J = {kk[2:]: v for kk, v in got[1].items() if kk.startswith("j_")}
        _ip, ev = intervention_problems(J, K, got[1]["actions"], kind,
                                        C["q"])
        rows.append(dict(eid=e, part=k, task=sc["task"],
                         case_id=sc["case_id"], seed=s, kind=kind,
                         success=bool(got[0]["summary"]["success"]),
                         t1_success=bool(ref[0]["summary"]["success"]),
                         reached=bool(ev and ev["first"] < ev["k"]),
                         event=bool(ev), phys=got[0]["phys"]))
    crash = sum(int(c["amount"]) for c in led["crashes"])
    if led["spent"].get("screen", 0) != tot + crash:
        p.append(f"расход {led['spent']} ≠ {tot} + аварии {crash}")
    return dict(ok=not p, problems=p[:50], part=k, n=len(rows),
                spent=led["spent"].get("screen", 0),
                crashes=len(led["crashes"]), rows=rows)


def status(a):
    plan, _k = load_plan(a.run_root, verify=False)
    for k in range(N_PARTS):
        lp = os.path.join(part_dir(a.run_root, k), "ledger.json")
        if not os.path.exists(lp):
            print(f"  part {k}: не начата")
            continue
        d = json.load(open(lp))
        print(f"  part {k}: {len(d['done'])}/{len(plan['parts'][str(k)]['eids'])}"
              f"; расход {d['spent']}; резерв "
              f"{(d['pending'] or {}).get('id', '—')}; аварий "
              f"{len(d['crashes'])}")
    return OK


def build_merge(rr, plan, k36):
    C = kh.consts(k36)
    ap = os.path.join(rr, "assignments.json")
    asg = json.load(open(ap)) if os.path.exists(ap) else dict(history=[])
    last = {h["part"]: h for h in asg["history"]}
    prob, rows = [], []
    for k in range(N_PARTS):
        local_p = os.path.join(part_dir(rr, k), "assignment.json")
        local = pc.load_json(local_p) if os.path.exists(local_p) else None
        h = last.get(k)
        keys = ("part", "host", "uuid", "n")
        if not h or not local or {x: local.get(x) for x in keys} != \
                {x: h.get(x) for x in keys}:
            prob.append(f"part {k}: назначение ≠ координатору")
        rep = part_problems(plan, k36, rr, k, C)
        prob += [f"part {k}: {x}" for x in rep["problems"]]
        rows += rep.get("rows", [])
    eids = [r["eid"] for r in rows]
    want = [e for k in range(N_PARTS) for e in plan["parts"][str(k)]["eids"]]
    if len(eids) != len(set(eids)) or set(eids) != set(want) or \
            len(want) != 2 * len(SEEDS) * len(plan["scenes"]):
        prob.append(f"набор новых eid: {len(set(eids))} из {len(want)}")
    pairs = {}
    for r in rows:
        pairs.setdefault((r["case_id"], r["seed"]), {})[r["kind"]] = r
    if len(pairs) != len(SEEDS) * len(plan["scenes"]) or \
            any(set(v) != set(KINDS) for v in pairs.values()):
        prob.append("неполные пары сцена × сид")
    for (c, s), v in pairs.items():
        t1 = plan["refs"].get(f"{s}|{c}")
        if not t1 or any(r["t1_success"] != t1["success"]
                         for r in v.values()):
            prob.append(f"T1 {c} s{s} ≠ плану")
    return prob, rows


def merge(a):
    plan, k36 = load_plan(a.run_root)
    prob, rows = build_merge(a.run_root, plan, k36)
    pc.save_json(os.path.join(a.run_root, "merged.json"), dict(
        plan_sha256=plan["sha256"], ok=not prob, problems=prob[:100],
        rows=rows if not prob else []))
    print("  merge: " + (f"полный набор {len(rows)} новых эпизодов, проверка "
                         f"пройдена" if not prob else
                         "ОТКАЗ: " + "; ".join(prob[:4])))
    return OK if not prob else TECH


def contrasts(table, n_boot=BOOT["n"], seed=BOOT["seed"]):
    """table: [dict(task, case_id, seed, T1, early, late)] -> статистика."""
    scenes = sorted({(r["task"], r["case_id"]) for r in table})
    if {t for t, _ in scenes} != set(TASKS):
        raise SystemExit("таблица исходов: набор задач ≠ плану")
    ix = {sc: i for i, sc in enumerate(scenes)}
    Y = {p_: np.full((len(scenes), len(SEEDS)), -1, int)
         for p_ in ("T1",) + KINDS}
    for r in table:
        for p_ in Y:
            Y[p_][ix[(r["task"], r["case_id"])], r["seed"]] = int(r[p_])
    if any((v < 0).any() for v in Y.values()):
        raise SystemExit("таблица исходов неполна")
    task = np.array([t for t, _c in scenes])
    rng = np.random.default_rng(seed)
    idx = {t: rng.integers(0, int((task == t).sum()),
                           (n_boot, int((task == t).sum()))) for t in TASKS}
    out = {}
    for a_, b_ in (("early", "T1"), ("late", "T1"), ("early", "late")):
        dif = Y[a_] - Y[b_]
        W = int((dif == 1).sum())
        L_ = int((dif == -1).sum())
        d = dif.mean(1)
        est = float(np.mean([d[task == t].mean() for t in TASKS]))
        bs = np.mean([d[task == t][idx[t]].mean(1) for t in TASKS], 0)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        per_seed = {str(s): int(dif[:, s].sum()) for s in SEEDS}
        scene_pos = int((dif.sum(1) > 0).sum())
        out[f"{a_}-{b_}"] = dict(
            W=W, L=L_, D=W - L_, pp=100.0 * (W - L_) / dif.size, delta=est,
            lo=float(lo), hi=float(hi), per_seed=per_seed,
            scenes_positive=scene_pos,
            by_task={str(t): int(dif[task == t].sum()) for t in TASKS},
            passes=bool(W - L_ >= RULE["D_min"] and
                        all(v > RULE["per_seed_gt"] for v in per_seed.values())
                        and scene_pos >= RULE["scenes_min"]))
    sr = {p_: float(Y[p_].mean()) for p_ in Y}
    rescue_any = int(((Y["T1"] == 0) & ((Y["early"] == 1) |
                                        (Y["late"] == 1))).sum())
    return out, sr, rescue_any


def analyze(a):
    plan, k36 = load_plan(a.run_root)
    mp = os.path.join(a.run_root, "merged.json")
    if not os.path.exists(mp):
        raise SystemExit("нет merged.json — сначала merge")
    m = pc.load_json(mp)
    if m.get("ok") is not True or m.get("plan_sha256") != plan["sha256"]:
        raise SystemExit("merge не пройден или от другого плана")
    prob, rows = build_merge(a.run_root, plan, k36)
    if prob:
        raise SystemExit("проверка при анализе: " + "; ".join(prob[:3]))
    key = lambda r: r["eid"]  # noqa: E731
    if sorted(rows, key=key) != sorted(m["rows"], key=key):
        raise SystemExit("merged.json ≠ проверенным записям")
    tab = {}
    for r in rows:
        x = tab.setdefault((r["case_id"], r["seed"]), dict(
            task=r["task"], case_id=r["case_id"], seed=r["seed"],
            T1=r["t1_success"]))
        x[r["kind"]] = r["success"]
    st, sr, rescue = contrasts(list(tab.values()))
    reached = {k: sum(r["reached"] for r in rows if r["kind"] == k)
               for k in KINDS}
    passing = [k for k in KINDS if st[f"{k}-T1"]["passes"]]
    best = None
    if passing:
        best = sorted(passing, key=lambda k: (-st[f"{k}-T1"]["D"],
                                              st[f"{k}-T1"]["L"],
                                              KINDS.index(k)))[0]
    status_ = "candidate" if best else "stopped_no_gain"
    pc.save_json(os.path.join(a.run_root, "analysis.json"), dict(
        plan_sha256=plan["sha256"], status=status_, best=best, sr=sr,
        contrasts=st, reached=reached, rescue_any_ceiling=rescue, rule=RULE,
        boot=BOOT))
    for k_, v in st.items():
        print(f"  {k_}: W {v['W']} L {v['L']} D {v['D']:+d} "
              f"({v['pp']:+.1f} п.п.); Δ {100 * v['delta']:+.2f} 95% "
              f"[{100 * v['lo']:+.2f}; {100 * v['hi']:+.2f}]; по сидам "
              f"{v['per_seed']}; сцен с плюсом {v['scenes_positive']}; по "
              f"задачам {v['by_task']}")
    print("  SR: " + ", ".join(f"{k} {100 * v:.1f}" for k, v in sr.items())
          + f"; исполненных сдвигов {reached}; T1-неудач, спасённых хотя бы "
          f"одним сдвигом, {rescue}")
    print(f"  K-38 g1: {status_}" + (f" (лучший {best})" if best else ""))
    return OK


# --- самопроверка --------------------------------------------------------------------------------
def selftest():
    from collections import Counter
    O, C = kg.OPEN, kg.CLOSE
    scenes = [dict(task=t, case_id=f"k33s2_t{t}_c{i:05d}", pos=i, hash=None)
              for t in TASKS for i in range(160)]
    sel = select_scenes(scenes)
    assert len(sel) == 60 and Counter(s["task"] for s in sel) == \
        {0: 20, 6: 20, 8: 20}
    for k in range(N_PARTS):
        o = part_order(sel, k)
        assert len(o) == 60 and len({s["case_id"] for _s, s, _k in o}) == 15
        assert Counter(s["task"] for _s, s, _k in o) == {0: 20, 6: 20, 8: 20}
        assert [x[0] for x in o] == [0] * 30 + [1] * 30
    assert kind_order(0, 0) == ("early", "late") and \
        kind_order(1, 0) == ("late", "early") == kind_order(0, 1)
    # пересборка вмешательства и проверки на синтетике
    import k29_policy as kp
    q = np.array([0.9] * 6 + [1.0])

    def chunk(g):
        a = np.zeros((8, 7), np.float32)
        a[:, :6] = 0.01
        a[:, 6] = -np.asarray(g, np.float32)
        return a
    prop = [chunk([O] * 8), chunk([O, O, O, O, C, C, C, C]), chunk([C] * 8)]
    J = dict(elapsed=np.array([0, 8, 16]), k=np.array([8, 8, 8]),
             a_codec=np.stack(prop), ep=np.zeros(3, int))
    for kind in KINDS:
        exp, ev = replay_expected(J, kind)
        K = dict(a2=exp, tau=np.array([-1, 4, -1]),
                 issued=np.array([False, True, False]))
        A = np.concatenate([kp.scale_exec(exp[i][None], q)[0]
                            for i in range(3)])
        assert intervention_problems(J, K, A, kind, q)[0] == []
        bad = dict(K, a2=K["a2"].copy())
        bad["a2"][1, 3, 6] *= -1
        assert intervention_problems(J, bad, A, kind, q)[0]
        assert intervention_problems(J, dict(K, issued=np.array(
            [True, True, False])), A, kind, q)[0]
        # префикс с архивным T1 (T1 = нулевой сдвиг тех же предложений)
        A1 = np.concatenate([kp.scale_exec(prop[i][None], q)[0]
                             for i in range(3)])
        ref = (dict(summary=dict(success=True)),
               dict(actions=A1, **{f"j_{k}": v for k, v in J.items()}))
        z = dict(actions=A, **{f"j_{k}": v for k, v in J.items()})
        assert prefix_problems(z, dict(summary=dict(success=False)), ref,
                               ev) == []
        A_bad = A.copy()
        A_bad[3, 0] += 1e-3                    # до первой изменённой
        assert prefix_problems(dict(z, actions=A_bad),
                               dict(summary=dict(success=False)), ref, ev)
        assert ev["first"] == (2 if kind == "early" else 4)
    # сдвиг не исполнен (k на вызове события ≤ первой изменённой): = T1
    J2 = dict(J, k=np.array([8, 2]), elapsed=J["elapsed"][:2],
              a_codec=J["a_codec"][:2], ep=np.zeros(2, int))
    exp2, ev2 = replay_expected(J2, "early")
    A2 = np.concatenate([kp.scale_exec(exp2[i][None], q)[0][:J2["k"][i]]
                         for i in range(2)])
    A2t1 = np.concatenate([kp.scale_exec(prop[i][None], q)[0][:J2["k"][i]]
                           for i in range(2)])
    assert np.array_equal(A2, A2t1)            # изменённые не исполнены
    ref2 = (dict(summary=dict(success=True)),
            dict(actions=A2t1, **{f"j_{k}": v for k, v in J2.items()}))
    z2 = dict(actions=A2, **{f"j_{k}": v for k, v in J2.items()})
    assert prefix_problems(z2, dict(summary=dict(success=True)), ref2,
                           ev2) == []
    assert prefix_problems(z2, dict(summary=dict(success=False)), ref2, ev2)
    assert prefix_problems(z2, dict(summary=dict(success=True)), ref2, None)
    J3 = dict(J2, k=np.array([8, 3]))
    exp3, ev3 = replay_expected(J3, "early")
    A3 = np.concatenate([kp.scale_exec(exp3[i][None], q)[0][:J3["k"][i]]
                         for i in range(2)])
    z3 = dict(actions=A3, **{f"j_{k}": v for k, v in J3.items()})
    assert prefix_problems(z3, dict(summary=dict(success=True)), ref2, ev3)
    # анализ и правило
    tab = []
    rng = np.random.default_rng(0)
    for t in TASKS:
        for i in range(20):
            for s in SEEDS:
                y1 = bool(rng.random() < 0.8)
                tab.append(dict(task=t, case_id=f"c{t}_{i}", seed=s, T1=y1,
                                early=y1 or (i < 4 and s in (0, 1)),
                                late=y1))
    st, sr, resc = contrasts(tab, n_boot=500)
    e = st["early-T1"]
    assert e["L"] == 0 and e["D"] == e["W"] and st["late-T1"]["D"] == 0
    assert st["late-T1"]["passes"] is False
    assert e["passes"] == (e["D"] >= 6 and all(v > 0 for v in
                                               e["per_seed"].values())
                           and e["scenes_positive"] >= 4)
    assert CAP_PART["screen"] * N_PARTS == 151280
    e2e_selftest()
    print("самопроверка k38_grip_run пройдена: 60 сцен (20 на задачу), "
          "4 × 15, все 4 записи на сцену в части, порядок сидов и "
          "чередование сдвигов; игрушечный допуск (нулевой сдвиг = архиву, "
          "smoke с реальным событием), сбор, слияние, отказ при подмене "
          "копии T1; пересборка вмешательства и отказы при порче "
          "канала/двойном вмешательстве; общий префикс с T1 и отказ при его "
          "порче; неисполненный сдвиг = T1; W/L/D и правило; лимит 151 280")
    return 0


def e2e_selftest():
    """Игрушка: ToySim, игрушечные K-25 и actor; «архив T1» из нулевого
    сдвига; эталонный допуск (нулевой сдвиг = архиву, smoke), сбор одной
    части, слияние, анализ; подмена копии T1 — отказ."""
    import tempfile
    import torch
    import k21_model as km
    import k25_model as k25m
    import k29_model as k29m
    import k32_model as km32
    import k32_ppo as kppo
    g_ = globals()
    k29m.no_tf32(torch)
    torch.manual_seed(0)
    M = k29m.build(torch)
    Lb = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    Ub = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    mod = k25m.build(torch)["K25"](km.fake_bar(torch), Lb, Ub, seed=0).eval()
    with torch.no_grad():
        gg = torch.Generator().manual_seed(3)
        for p_ in list(mod.cond.parameters()) + list(
                mod.readout.parameters()):
            p_.add_(torch.randn(p_.shape, generator=gg) * 0.05)
    for p_ in mod.parameters():
        p_.requires_grad_(False)
    dec = M["Decoder"](mod).freeze()
    H = mod.H
    norm = km32.fit_normalizer(np.random.default_rng(0).normal(
        size=(64, 6 * H + 72)), H)

    class FIO(kcol.FakeIO):
        def prep(self, obs, desc, task):
            x = float(obs["state"][0, 0])
            bt = km.fake_batch(torch, B=1, seed=int(abs(x) * 1e3) % 13)
            s8 = np.zeros((1, 8))
            s8[0, :3] = x
            return bt, 1, s8
    io = FIO()
    actor, _c = km32.build(torch, H, 6, 0, norm)
    with torch.no_grad():
        g2 = torch.Generator().manual_seed(7)
        for p_ in actor.out.parameters():
            p_.copy_(torch.randn(p_.shape, generator=g2))
    ah = kppo.tensor_hash(actor.state_dict())
    k36 = dict(sha1="PREP36", k32=dict(k32_contract="C", seeds={
        str(s): dict(source=dict(weights_hash="W"), T=dict(actor_hash=ah))
        for s in SEEDS}), k32_plan=dict(
        bounds=dict(L=list(Lb) + [-1.0], U=list(Ub) + [1.0]),
        scale_q=np.asarray(io.max_act_q).tolist()))
    src = dict(mod=mod, dec=dec, M=M, io=io, actors=dict(T=actor))
    sc_all = [dict(task=t, case_id=f"k33s2_t{t}_c{i:05d}", pos=i, hash=None,
                   rank=0, part=0) for t in TASKS for i in range(2)]
    sel = [sc_all[i] for i in (0, 2, 5)]  # по одной сцене каждой задачи
    pfz, smk = sc_all[3], sc_all[4]
    keep = {k: g_[k] for k in ("load_plan", "N_PARTS", "env_record")}
    keep_kh = dict(OUT=kh.OUT, REJECTED=kh.REJECTED, SS=dict(kh.STAGE_SCOPE))
    with tempfile.TemporaryDirectory() as td:
        os.environ["K37_LOCK_DIR"] = os.path.join(td, "locks")
        keep_disk = kcf._DISK_USAGE
        kcf._DISK_USAGE = lambda p: type("U", (), dict(free=100 * 2 ** 30))()
        try:
            rr = os.path.join(td, "g")
            arch = os.path.join(td, "arch")
            # «архив T1»: нулевой сдвиг, штатная запись K-36
            install(arch)
            kh.STAGE_SCOPE["confirm"] = "confirm"
            refs = {}
            for sc in sel + [pfz, smk]:
                for s in SEEDS:
                    w = kh.want_of(dict(k36, sha1="C2"), "confirm",
                                   f"T32_s{s}", sc["task"], sc["case_id"],
                                   sc["pos"], "bank", space=2,
                                   case_id=sc["case_id"])
                    arm = kg.GripArm(kh.make_arm(torch, k36, f"T32_s{s}",
                                                 src), "zero")
                    arm.episode = kh.episode_number(w["eid"])
                    ep = kcol.run_episode(kppo.ToySim(), io, arm, w["task"],
                                          w["reset_id"])
                    kh.save_ep(w, ep)
                    jp, np_ = kh.ep_paths("confirm", f"T32_s{s}", sc["task"],
                                          sc["case_id"])
                    rel = os.path.join("inputs", f"T32_s{s}",
                                       os.path.basename(jp))
                    os.makedirs(os.path.join(rr, "inputs", f"T32_s{s}"),
                                exist_ok=True)
                    shutil.copyfile(jp, os.path.join(rr, rel))
                    shutil.copyfile(np_, os.path.join(rr, rel[:-5] + ".npz"))
                    refs[f"{s}|{sc['case_id']}"] = dict(
                        eid=w["eid"], json=rel, npz=rel[:-5] + ".npz",
                        json_sha256=sha256(os.path.join(rr, rel)),
                        npz_sha256=sha256(os.path.join(rr, rel[:-5] +
                                                       ".npz")),
                        success=bool(ep["success"]))
            zs = dict(np.load(os.path.join(rr, refs[f"0|{smk['case_id']}"][
                "npz"])))
            ev_s, _n = kgc.episode_event(
                {k[2:]: v for k, v in zs.items() if k.startswith("j_")},
                zs["actions"])
            assert ev_s is not None, "в игрушке нет события для smoke"
            plan = dict(sha256="G1", scenes=sel, refs=refs,
                        parts={"0": dict(scenes=[s["case_id"] for s in sel])},
                        pf_zero=[dict(seed=0, scene=pfz),
                                 dict(seed=1, scene=pfz)],
                        smoke=dict(seed=0, scene=smk, ev=ev_s,
                                   expect=dict(early=ev_s["early"],
                                               late=ev_s["late"])),
                        c2=dict(bank=dict(dir=".")),
                        resource=dict(kcf.RES, preflight_need_gb=10.0))
            plan["parts"]["0"]["eids"] = [w["eid"] for *_x, w in
                                          part_wants(plan, k36, 0)]
            g_.update(load_plan=lambda rr_, verify=True: (dict(plan), k36),
                      N_PARTS=1, env_record=lambda t: dict(torch_threads=8))
            os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = \
                str(THREADS)
            torch.set_num_threads(THREADS)
            NS = argparse.Namespace
            kcf_dev = kcf.dev_index
            kcf.dev_index = lambda d: 0
            gr = lambda c: "0, GPU-u0, 00000000:01:00.0, 32510, 0, 32510, 0, V100"  # noqa
            try:
                # эталон: нулевой сдвиг = архиву, smoke по аудиту архива
                assert preflight(NS(run_root=rr, device="cpu",
                                    reference=True),
                                 sim=kh._NoClose(kppo.ToySim()),
                                 src_loader=lambda s: src, gpu_runner=gr,
                                 torch_check=False) == OK
                rep = pc.load_json(os.path.join(pf_dir(
                    rr, socket.gethostname(), "GPU-u0"), "report.json"))
                assert rep["admitted"] and rep["n_episodes"] == 4
                assign(NS(run_root=rr, part=0, host=socket.gethostname(),
                          uuid="GPU-u0", state="assigned"))
                assert collect(NS(run_root=rr, part=0, device="cpu"),
                               sim=kh._NoClose(kppo.ToySim()),
                               src_loader=lambda s: src, gpu_runner=gr,
                               torch_check=False) == OK
                assert merge(NS(run_root=rr)) == OK
                merged = pc.load_json(os.path.join(rr, "merged.json"))
                assert merged["ok"] and len(merged["rows"]) == 12
                assert analyze(NS(run_root=rr)) == OK
                report = pc.load_json(os.path.join(rr, "analysis.json"))
                assert all(np.isfinite(v[key])
                           for v in report["contrasts"].values()
                           for key in ("delta", "lo", "hi"))
                # подмена копии T1 — отказ проверки
                e0 = refs[f"0|{sel[0]['case_id']}"]
                fp = os.path.join(rr, e0["npz"])
                b = open(fp, "rb").read()
                open(fp, "wb").write(b[:-1] + bytes([b[-1] ^ 1]))
                try:
                    build_merge(rr, plan, k36)
                    raise AssertionError("подмена T1 прошла")
                except SystemExit:
                    pass
                open(fp, "wb").write(b)
            finally:
                kcf.dev_index = kcf_dev
        finally:
            g_.update(keep)
            kcf._DISK_USAGE = keep_disk
            kh.OUT, kh.REJECTED = keep_kh["OUT"], keep_kh["REJECTED"]
            kh.STAGE_SCOPE.clear()
            kh.STAGE_SCOPE.update(keep_kh["SS"])
            os.environ.pop("K37_LOCK_DIR", None)


def main(argv=None):
    if argv is None and "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-38 g1")
    ap.add_argument("cmd", choices=("prepare", "gpuinfo", "assign",
                                    "preflight", "collect", "status",
                                    "merge", "analyze"))
    ap.add_argument("--run-root", default=os.environ.get("K38_RUN_ROOT"))
    ap.add_argument("--run-id")
    ap.add_argument("--archive", default="reports/k37c/runs/c2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--part", type=int)
    ap.add_argument("--host")
    ap.add_argument("--uuid")
    ap.add_argument("--state", default="assigned",
                    choices=("assigned", "released", "done"))
    ap.add_argument("--reference", action="store_true")
    a = ap.parse_args(argv)
    fn = dict(prepare=prepare, gpuinfo=kcf.gpuinfo, assign=assign,
              preflight=preflight, collect=collect, status=status,
              merge=merge, analyze=analyze)[a.cmd]
    if a.cmd not in ("prepare", "gpuinfo") and not a.run_root:
        print("  ОТКАЗ: нужен --run-root или K38_RUN_ROOT")
        return TECH
    if a.cmd == "prepare" and not a.run_id:
        print("  ОТКАЗ: нужен --run-id")
        return TECH
    try:
        return fn(a)
    except kcf.Busy as e:
        print(f"  ВРЕМЕННО НЕТ РЕСУРСА: {e}", flush=True)
        return BUSY
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return TECH
        raise


if __name__ == "__main__":
    sys.exit(main())
