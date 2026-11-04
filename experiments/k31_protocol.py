#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-31: подготовка, real-check, fork-smoke, пилот A и гейт, добор B, набор,
fitcheck, обучение селекторов, план SR.

prepare — reports/k31/prepare.json (неизменяем; повтор только сверяет,
  конфликт — отказ до записи): источники K-25 s0/s1 (SHA, отпечаток, отчёт,
  код, prep_id, best update из отчёта), снимок BAR, пределы, банки,
  описания 10 задач (план A K-21 = LIBERO), холодный эталон начальных хешей
  30…44 всех задач (закреплённые блоки K-25), генератор/нормировка,
  селектор, бюджет, точность, SHA кода.
real-check DEV — reports/k31/real_check.json: оба источника на настоящем BAR
  по наблюдениям LIBERO (B = 1; B = 5 — только независимость примеров):
  legacy экспорт = K-25 побитово; нулевая правка каждого уровня = FP32
  reference побитово; кеш F (CPU-декодер) против online: |ΔZ| ≤ 1e−5,
  |Δarm| ≤ 1e−6; счётчики; правка родителя меняет детей, вне поддерева
  побитово; схват — reference; пул (нормы, m); FP32 под fp16 autocast;
  память и время reference/пула/селектора.
fork-smoke DEV — 4 случая (s0,t0,i0) (s0,t8,i1) (s1,t0,i0) (s1,t8,i1),
  пространство RNG 1: reference, noop, один tree, повтор того же tree, один
  flat; noop и повтор обязаны совпасть; ≤ 12 200 шагов (лимит checks 20 000).
pilot DEV — A обоих сидов (train 0…7, val 20…21 на задачах 0/6/8) и гейт:
  полнота + noop; ≥ 80% train-checkpoint каждого сида с tree-кандидатом;
  ≥ 2 различных train-пары (задача, init), спасённые tree, в объединении
  сидов. Иначе stopped_low_proposal_coverage / stopped_no_headroom (код 4).
collect DEV — B (train 8…19, val 22…24) только после допуска A.
dataset / fitcheck / train — CPU; train: 1000 обновлений, best по val CE;
  оба сида без активности tree на validation -> stopped_inactive_selector.
plan-eval — reports/k31/plan_eval_s<S>.json для SR (k31_harness).
"""
import argparse
import datetime
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k31_candidates as kc  # noqa: E402
import k31_collect as kcol  # noqa: E402

PREPARE = "reports/k31/prepare.json"
REAL_CHECK = "reports/k31/real_check.json"
SMOKE = "reports/k31/fork_smoke.json"
PILOT = "reports/k31/pilot.json"
COLLECT = "reports/k31/collect.json"
FITCHECK = "reports/k31/fitcheck.json"
TRAIN = "reports/k31/train.json"
LEDGER = "data/k31/ledger.json"
CASES = "data/k31/cases"
SEEDS = (0, 1)
PRIMARY = (0, 6, 8)
A_TRAIN, A_VAL = tuple(range(0, 8)), (20, 21)
B_TRAIN, B_VAL = tuple(range(8, 20)), (22, 23, 24)
EVAL_INIT = tuple(range(30, 45))
CLOSED = tuple(range(45, 50))
UNUSED = tuple(range(25, 30))
CAPS = dict(dev_s0=400000, dev_s1=400000, checks=20000, sr=549000)
A_LIMIT = 160000
SMOKE_CASES = ((0, 0, 0), (0, 8, 1), (1, 0, 0), (1, 8, 1))
GATE = dict(tree_coverage=0.8, rescued_pairs=2)
Z_TOL, ARM_TOL = 1e-5, 1e-6
DU_TOL, LAM_TOL = 1e-5, 1e-4     # пересборка: |ΔDu| абс., |Δλ|/λ
MODEL_CODE = ("k31_readout.py", "k31_candidates.py", "k31_policy.py",
              "k31_collect.py", "k31_data.py", "k31_train.py",
              "k31_harness.py", "k31_protocol.py", "k29_model.py",
              "k29_policy.py", "k29_dist.py", "k25_readout.py",
              "k25_model.py", "k25_policy.py", "k22_model.py", "k21_model.py",
              "k21_tree.py", "k21_data.py", "k21_train.py", "k20_data.py",
              "k18_collector.py", "k16_models.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def fp(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()
                        ).hexdigest()


def code_sha():
    return {f: sha(os.path.join(HERE, f)) for f in MODEL_CODE}


def save_json(path, d):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(path + ".tmp", path)


# --- подготовка -------------------------------------------------------------------
def banks():
    b = dict(pilot_train=list(A_TRAIN), pilot_val=list(A_VAL),
             more_train=list(B_TRAIN), more_val=list(B_VAL),
             eval=list(EVAL_INIT), tasks_dev=list(PRIMARY),
             tasks_eval=list(range(10)), closed=list(CLOSED),
             unused=list(UNUSED))
    sets = [set(A_TRAIN), set(A_VAL), set(B_TRAIN), set(B_VAL),
            set(EVAL_INIT)]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] & sets[j]:
                raise SystemExit("банки начальных состояний пересекаются")
    if any(x in CLOSED or x in UNUSED for s_ in sets for x in s_):
        raise SystemExit("банки задевают 25…29 или 45…49")
    return b


def k25_init_ref():
    import k23_protocol as p23
    import k26_protocol as p26
    pins, ref = {}, {}
    for s in SEEDS:
        r = p26.ref_k25(s)
        pins[f"k25_s{s}"] = p23.pin_ref(r)
        for (_l, _su, t, _b), f in r["blocks"].items():
            for e in json.load(open(f, encoding="utf-8"))["episodes"]:
                key = f"{int(t)}:{int(e['init_state_id'])}"
                if ref.setdefault(key, e["init_hash"]) != e["init_hash"]:
                    raise SystemExit(f"эталон {key} различается между сидами")
    want = {f"{t}:{i}" for t in range(10) for i in EVAL_INIT}
    if set(ref) != want:
        raise SystemExit("эталон стартов K-25 неполон")
    return pins, ref


def make_prepare():
    import k21_data as d21
    import k23_protocol as p23
    import k29_model as km
    import k29_policy as kp
    import k31_train as kt
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    srcs = {}
    for s in SEEDS:
        e = km.source_entry(s)
        if e["prep_id"] != prep["prep_id"]:
            raise SystemExit(f"seed {s}: источник другой подготовки K-21")
        tr = json.load(open(e["report"], encoding="utf-8"))
        best = tr.get("best_update")
        if best is None or int(best) != int(e["update"]):
            raise SystemExit(f"seed {s}: шаг best отчёта {best} ≠ чекпойнту "
                             f"{e['update']}")
        srcs[str(s)] = e
    rA = p23.ref_k21_main()
    desc = {}
    for (l_, _s, t_, _b), f in rA["blocks"].items():
        if l_ == "fullbar":
            desc.setdefault(int(t_), json.load(open(f, encoding="utf-8"))
                            .get("task_description"))
    from libero.libero import benchmark
    bm = benchmark.get_benchmark_dict()["libero_10"]()
    tasks = {}
    for t in range(10):
        lang = bm.get_task(t).language
        if desc.get(t) != lang:
            raise SystemExit(f"задача {t}: план A {desc.get(t)!r} ≠ LIBERO")
        if len(bm.get_task_init_states(t)) < 45:
            raise SystemExit(f"задача {t}: начальных состояний < 45")
        tasks[str(t)] = lang
    pins, init_ref = k25_init_ref()
    U, r = kp.harness_utils()
    q = np.maximum(np.abs(U.ACTION_Q99), np.abs(U.ACTION_Q01))
    return dict(
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        prep_id=prep["prep_id"], prepare_sha1=sha(d21.REPORT),
        bar_revision=prep["bar_pin"]["revision"], snapshot=snap,
        bounds=prep["bounds_codec"], state_norm=prep["state_norm"],
        sources=srcs, tasks=tasks, banks=banks(), env=kcol.ENV,
        candidates=kc.CONFIG, selector=kt.CFG, caps=CAPS, a_limit=A_LIMIT,
        gate=GATE, smoke_cases=[list(x) for x in SMOKE_CASES],
        precision=dict(trunk="fp16 autocast, no_grad, TF32 off", B=1,
                       reference="FP32 вне autocast", selector="FP32 CPU"),
        k25_refs=pins, init_ref=init_ref, scale_q=q.tolist(),
        offset_table_sha1=sha("data/pos_offset_table.json"),
        harness_cfg_sha1=sha(os.path.join(r, "config/eval/bar.yaml")),
        code=code_sha())


CONTRACT_KEYS = ("prep_id", "prepare_sha1", "bar_revision", "bounds",
                 "state_norm", "sources", "tasks", "banks", "env",
                 "candidates", "selector", "caps", "a_limit", "gate",
                 "smoke_cases", "precision", "init_ref", "scale_q",
                 "offset_table_sha1", "harness_cfg_sha1", "code")


def prepare():
    # сравнение — в JSON-форме (кортежи CFG после записи становятся списками)
    rep = json.loads(json.dumps(make_prepare(), default=str))
    if os.path.exists(PREPARE):
        old = json.load(open(PREPARE, encoding="utf-8"))
        diff = [k for k in CONTRACT_KEYS if old.get(k) != rep.get(k)]
        if diff:
            raise SystemExit(f"{PREPARE} закреплён; изменилось {diff} — "
                             f"новая постановка требует нового каталога")
        print(f"  {PREPARE}: закреплён, сверка пройдена")
        return 0
    save_json(PREPARE, rep)
    print(f"  {PREPARE}: источники " + ", ".join(
        f"s{s} {e['weights_hash'][:12]} (обн. {e['update']})"
        for s, e in rep["sources"].items()))
    return 0


def load_prepare():
    if not os.path.exists(PREPARE):
        raise SystemExit("нет подготовки: k31_protocol.py prepare")
    p = json.load(open(PREPARE, encoding="utf-8"))
    bad = [f for f, s in p["code"].items() if sha(os.path.join(HERE, f)) != s]
    if bad:
        raise SystemExit(f"код изменён после prepare: {bad}")
    return p


def contract(p):
    return fp({k: p[k] for k in CONTRACT_KEYS})


# --- общие загрузки ---------------------------------------------------------------
def source_entry_of(doc, seed):
    """Описание источника сида из prepare ИЛИ плана SR (один формат)."""
    try:
        return doc["sources"][str(seed)]
    except (KeyError, TypeError):
        raise SystemExit(f"в документе нет источника seed {seed}")


def load_source_dec(torch, doc, seed, device):
    import k29_model as km
    km.no_tf32(torch)
    e = source_entry_of(doc, seed)
    mod, bar, _prep = km.load_source(torch, e, device)
    if km.frozen_hash(mod) != e["weights_hash"]:
        raise SystemExit("источник после загрузки ≠ отпечатку")
    M = km.build(torch)
    return mod, bar, M, M["Decoder"](mod).freeze()


def free():
    """Сборка мусора и кеш CUDA; ссылки вызывающий удаляет сам (del)."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ledger():
    return kcol.Ledger(LEDGER, CAPS)


def capture(agent):
    la = agent.last
    return dict(F={k: v.detach().clone() for k, v in la["F"].items()},
                ref={k: v.detach().clone() for k, v in la["ref"].items()},
                s11n=la["s11n"].detach().clone(), call=la["rec"]["call"],
                elapsed=la["rec"]["elapsed"])


def make_pool_fn(arm):
    import k31_policy as kpol

    def pool_fn(payload, fork):
        if int(payload["call"]) != int(fork["call"]):
            raise kcol.Mismatch("вызов полезной нагрузки ≠ развилке")
        F, ref = payload["F"], payload["ref"]
        pr = arm.proposals(F, ref, int(fork["call"]))
        P = pr["P"]
        mean, pth = kpol.ctx_vectors(F, pr["path"])
        al = np.asarray(P["allowed"], bool)
        va = np.asarray(P["valid"], bool)
        Du = np.stack([np.asarray(r["Du"], np.float32) if va[i] else
                       np.full((kc.T, 6), np.nan, np.float32)
                       for i, r in enumerate(P["rows"])])
        lam = np.array([r["lam"] if (va[i] and i) else np.nan
                        for i, r in enumerate(P["rows"])])
        lam_raw = np.array([np.nan if r["lam"] is None else r["lam"]
                            for r in P["rows"]])
        info = pr["info"]
        npz = dict(valid=P["valid"], m=np.array(P["m"]), lam=lam,
                   lam_raw=lam_raw, Du=Du, u0=(pr["Z0"] / arm.half_cpu)
                   .numpy(), Z0=pr["Z0"].numpy(), Zhat=pr["Zhat"].numpy(),
                   d=pr["d"], flat=pr["flat"], path=np.array(pr["path"]),
                   ctx_mean=mean, ctx_path=pth,
                   s11n=payload["s11n"][0].float().cpu().numpy(),
                   node_center=np.array([info[k]["center"] for k in
                                         ("root", "mid", "leaf")]),
                   node_width=np.array([info[k]["width"] for k in
                                        ("root", "mid", "leaf")]),
                   node_mass=np.array([info[k]["mass"] for k in
                                       ("root", "mid", "leaf")]))
        if arm.space == 1:            # smoke: контекст CPU-пересборки
            npz.update({f"F_{k}": v.detach().cpu().numpy()
                        for k, v in F.items()})
        return dict(arms=P["arm"].numpy(), allowed=al, npz=npz,
                    json=dict(m=int(P["m"]), why=[r["why"] for r in
                                                  P["rows"]],
                              path=list(pr["path"]), info=info,
                              elapsed=int(payload["elapsed"])))
    return pool_fn


def case_dir(seed, split, task, init, root=CASES):
    return os.path.join(root, f"s{seed}", split, f"t{task}_i{init}")


def items_of(phase):
    tr, va = (A_TRAIN, A_VAL) if phase == "A" else (B_TRAIN, B_VAL)
    return [("train", t, i) for t in PRIMARY for i in tr] + \
        [("val", t, i) for t in PRIMARY for i in va]


def run_cases(device, jobs, *, space, scope_of, limit_of, root=CASES,
              smoke=False):
    """jobs: [(seed, split, task, init)] -> {(seed, split, task, init): case}.
    Среды — до CUDA; модели сидов — по очереди."""
    import k18_collector as k18c
    sim = k18c.Sim(1, "persistent")
    import torch
    import k29_policy as kp
    import k31_harness as kh
    import k31_policy as kpol
    p = load_prepare()
    L = ledger()
    out = {}
    try:
        for seed in sorted({j[0] for j in jobs}):
            mod, bar, M, dec = load_source_dec(torch, p, seed, device)
            arm = kpol.Arm(torch, mod, dec, M, mode="reference",
                           source_seed=seed, space=space)
            io = kh.IO(kp.Inputs(torch, bar["proc"], device))
            pool_fn = make_pool_fn(arm)
            for (s, split, t, i) in [j for j in jobs if j[0] == seed]:
                t0 = time.time()
                c = kcol.run_case(sim, io, arm, t, i, seed=s, space=space,
                                  cdir=case_dir(s, split, t, i, root),
                                  ledger=L, scope=scope_of(s),
                                  limit=limit_of(s), pool_fn=pool_fn,
                                  capture=capture,
                                  tag=f"{'smoke' if smoke else 'dev'}/s{s}/"
                                      f"{split}/t{t}_i{i}",
                                  max_branches=2 if smoke else None,
                                  repeat=smoke)
                out[(s, split, t, i)] = c
                lab = ", ".join(f"{k}{'+' if v > 0 else '−' if v < 0 else '0'}"
                                for k, v in c["labels"].items())
                print(f"  s{s} {split} t{t} i{i}: R0 {int(c['R0'])}, "
                      f"tau {c['fork']['tau']}, {lab or 'нет кандидатов'}; "
                      f"шагов {sum(c['physical'].values())}, "
                      f"{(time.time() - t0) / 60:.1f} мин", flush=True)
            del mod, bar, M, dec, arm, io, pool_fn
            free()
    finally:
        sim.close()
    return out


# --- real-check ---------------------------------------------------------------------
def real_check(a):
    import k18_collector as k18c
    sim = k18c.Sim(5, "persistent")
    import torch
    import k25_policy as kp25
    import k29_policy as kp
    import k31_policy as kpol
    import k31_readout as kr
    import k21_model as km
    sim1 = k18c.Sim(1, "persistent")
    p = load_prepare()
    rep = dict(contract=contract(p), seeds={})
    starts = [30, 31, 32, 33, 34]
    try:
        obs5 = sim.reset(0, starts)
        dummy = np.array([kcol.DUMMY] * 5)
        for _ in range(kcol.ENV["waiting_steps"]):
            obs5, _r, _d, _s = sim.step(dummy)
        o1 = sim1.reset(0, [30])
        for _ in range(kcol.ENV["waiting_steps"]):
            o1, _r, _d, _s = sim1.step(dummy[:1])
    finally:
        sim.close()
        sim1.close()
    L = ledger()
    eid = f"checks/real_check@{time.strftime('%Y%m%dT%H%M%S')}"
    steps_rc = 6 * kcol.ENV["waiting_steps"]
    L.reserve("checks", eid, amount=steps_rc)
    L.commit(eid, steps_rc)
    ih5, _f5 = k18c.init_hashes(obs5, 5)
    ih1, _f1 = k18c.init_hashes(o1, 1)
    want = [p["init_ref"][f"0:{i}"] for i in starts]
    rep["cold_start"] = dict(B5_equal_ref=ih5 == want,
                             B1_equal_ref=ih1[0] == want[0],
                             B1_equal_B5=ih1[0] == ih5[0])
    obs1 = {k: v[:1] for k, v in obs5.items()}
    for seed in SEEDS:
        mod, bar, M, dec = load_source_dec(torch, p, seed, a.device)
        dev = torch.device(a.device)
        inp = kp.Inputs(torch, bar["proc"], a.device)
        desc = inp.task_desc(0)
        E = kr.build(torch)
        ac = torch.autocast(dev.type, dtype=torch.float16)
        r = {}
        z_b1 = None
        for B, ob in ((1, obs1), (5, obs5)):
            batch, po, s8 = inp(ob, desc, 0)
            s11 = torch.as_tensor(km.state11(s8), device=dev)
            with torch.no_grad(), ac:
                F, s11n, cnt = M["export"](mod, batch, po, s11)
                legacy = dec(F, autocast=ac)
                real = mod(batch, po, s11)
            with torch.no_grad():
                ref = dec(F)
            r[f"legacy_B{B}"] = bool(
                torch.equal(legacy["arm"], real["arm"]) and
                torch.equal(legacy["grip_logit"], real["grip_logit"]) and
                torch.equal(legacy["Z"], real["y"]))
            r[f"counters_B{B}"] = kp25.counters_problems(cnt)
            if B == 5:
                F1 = {k: v[:1] for k, v in F.items()}
                with torch.no_grad():
                    ref1 = dec(F1)
                r["B5_vs_B1_maxZ"] = float((ref["Z"][:1] - z_b1).abs().max())
                r["B5_row_vs_slice_equal"] = bool(torch.equal(ref1["Z"],
                                                              ref["Z"][:1]))
                r["B5_finite"] = bool(torch.isfinite(ref["Z"]).all())
                continue
            z_b1 = ref["Z"].detach().clone()
            path = kr.path_of(F["W2"][0].float().cpu().numpy())
            zero = {}
            for lv, ix in zip(kr.LEVELS, path):
                z = E["edited"](dec, F, ref, [lv], [ix], np.zeros((1, 6)))
                zero[lv] = bool(torch.equal(z["Zhat"], ref["Z"]))
            r["zero_edit_bitwise"] = zero
            Fc = {k: v.detach().cpu() for k, v in F.items()}
            dcpu = M["Decoder"](mod).freeze().cpu()
            with torch.no_grad():
                rc = dcpu(Fc)
            r["cache_maxZ"] = float((rc["Z"] - ref["Z"].cpu()).abs().max())
            r["cache_maxarm"] = float((rc["arm"] - ref["arm"].cpu()).abs()
                                      .max())
            dvec = np.random.default_rng(seed).normal(size=6)
            dvec = (0.1 * dvec / np.linalg.norm(dvec)).astype(np.float32)
            o = E["edited"](dec, F, ref, list(kr.LEVELS), list(path),
                            np.stack([dvec] * 3))
            with torch.autocast(dev.type, dtype=torch.float16):
                o16 = E["edited"](dec, F, ref, list(kr.LEVELS), list(path),
                                  np.stack([dvec] * 3))
            r["fp32_under_autocast"] = bool(o16["Zhat"].dtype == torch.float32
                                            and torch.equal(o16["Zhat"],
                                                            o["Zhat"]))
            sub = {}
            for i, lv in enumerate(kr.LEVELS):
                _r, mids, leaves = kr.descendants(lv, path[i])
                ol = [x for x in range(16) if x not in leaves]
                om = [x for x in range(8) if x not in mids]
                sub[lv] = dict(
                    outside_bitwise=bool(
                        torch.equal(o["z2"][i, ol], ref["z2"][0, ol]) and
                        torch.equal(o["z1"][i, om], ref["z1"][0, om])),
                    inside_max=float((o["z2"][i, leaves] -
                                      ref["z2"][0, leaves]).abs().max()))
            r["subtree"] = sub
            rt = 2 * path[0]
            r["parent_changes_child_delta"] = float(
                (o["delta1"][0, rt] - ref["delta1"][0, rt]).abs().max())

            class ForceSel:
                def ctx(self, mean, pth):
                    return np.zeros(32, np.float32)

                def scores(self, X):
                    return np.zeros((7, 3), np.float32), np.array(
                        [0, .9, .8, .7, .6, .5, .4], np.float64)
            arm = kpol.Arm(torch, mod, dec, M, mode="repair",
                           source_seed=seed, space=2, selector=ForceSel())
            arm.begin(0, 0)
            ts = []
            for _ in range(3):
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                a_c, rec = arm.decide(batch, po, s8, 0)
                torch.cuda.synchronize(dev)
                ts.append((time.perf_counter() - t0) * 1e3)
                arm.begin(0, 0)
            g = np.where(rec["grip_logit"] > 0, 1.0, -1.0)
            r["grip_reference"] = bool(np.array_equal(a_c[:, 6], g))
            r["pool"] = dict(valid=rec["valid"].tolist(),
                             allowed=rec["allowed"].tolist(), m=int(rec["m"]),
                             lam=[None if not np.isfinite(x) else float(x)
                                  for x in rec["lam"]], why=rec["why"],
                             chosen=rec["chosen_name"])
            ra = kpol.Arm(torch, mod, dec, M, mode="reference",
                          source_seed=seed)
            ra.begin(0, 0)
            tr_ = []
            torch.cuda.reset_peak_memory_stats(dev)
            for _ in range(5):
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                ra.decide(batch, po, s8, 0)
                torch.cuda.synchronize(dev)
                tr_.append((time.perf_counter() - t0) * 1e3)
            r["ms_reference"] = float(np.median(tr_))
            r["ms_repair_eligible"] = float(np.median(ts))
            r["peak_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        rep["seeds"][str(seed)] = r
        del mod, bar, M, dec, arm, ra, E, F, ref, o, o16, dcpu, rc
        free()
    prob = []
    if not all(rep["cold_start"].values()):
        prob.append(f"холодный старт B = 1 ≠ эталону K-25: {rep['cold_start']}")
    for s, r in rep["seeds"].items():
        if not (r["legacy_B1"] and r["legacy_B5"]):
            prob.append(f"s{s}: legacy ≠ K-25")
        if r["counters_B1"] or r["counters_B5"]:
            prob.append(f"s{s}: счётчики")
        if not all(r["zero_edit_bitwise"].values()):
            prob.append(f"s{s}: нулевая правка ≠ reference")
        if not (r["cache_maxZ"] <= Z_TOL and r["cache_maxarm"] <= ARM_TOL):
            prob.append(f"s{s}: кеш F ≠ online ({r['cache_maxZ']:.2e}, "
                        f"{r['cache_maxarm']:.2e})")
        if not r["fp32_under_autocast"]:
            prob.append(f"s{s}: FP32 под autocast")
        if not all(v["outside_bitwise"] for v in r["subtree"].values()):
            prob.append(f"s{s}: вне поддерева изменено")
        if not all(v["inside_max"] > 0 for v in r["subtree"].values()):
            prob.append(f"s{s}: правка не изменила поддерево")
        if not r["grip_reference"]:
            prob.append(f"s{s}: схват ≠ reference")
    rep["ok"] = not prob
    rep["problems"] = prob
    rep["device"] = torch.cuda.get_device_name(torch.device(a.device))
    save_json(REAL_CHECK, rep)
    for s, r in rep["seeds"].items():
        print(f"  s{s}: legacy B1/B5 {r['legacy_B1']}/{r['legacy_B5']}; "
              f"нулевая правка {r['zero_edit_bitwise']}; кеш |ΔZ| "
              f"{r['cache_maxZ']:.2e}, |Δarm| {r['cache_maxarm']:.2e}; "
              f"пул m {r['pool']['m']}, λ {r['pool']['lam']}; reference "
              f"{r['ms_reference']:.0f} мс, eligible "
              f"{r['ms_repair_eligible']:.0f} мс; пик {r['peak_gb']:.1f} ГиБ",
              flush=True)
    print("  " + ("real-check K-31 пройден" if not prob else
                  "real-check НЕ пройден: " + "; ".join(prob)))
    return 0 if not prob else 3


# --- свежесть ---------------------------------------------------------------------
def fresh(kind):
    """Пустой список — этап свеж для текущих prepare/кода."""
    try:
        p = load_prepare()
    except SystemExit as e:
        return [str(e.code)]
    path = dict(real_check=REAL_CHECK, smoke=SMOKE, pilot=PILOT,
                collect=COLLECT, fitcheck=FITCHECK, train=TRAIN)[kind]
    if not os.path.exists(path):
        return [f"нет {path}"]
    d = json.load(open(path, encoding="utf-8"))
    pr = []
    if d.get("contract") != contract(p):
        pr.append(f"{path}: другой контракт")
    if kind in ("real_check", "smoke") and not d.get("ok"):
        pr.append(f"{path}: не пройден")
    if kind == "pilot" and d.get("status") not in ("passed",
                                                   "stopped_no_headroom",
                                                   "stopped_low_proposal_"
                                                   "coverage"):
        pr.append(f"{path}: статус {d.get('status')}")
    if kind == "fitcheck":
        if not d.get("ok"):
            pr.append("fitcheck не пройден")
        for s in SEEDS:
            dp = dataset_path(s)
            if not os.path.exists(dp) or sha(dp) != d.get("datasets", {}).get(
                    str(s)):
                pr.append(f"fitcheck другого набора s{s}")
    return pr


# --- независимая CPU-пересборка smoke ----------------------------------------------
FEAT = ("x0", "x1", "x2", "iv0", "iv1", "iv2", "W0", "W1", "W2")


def decoder_from_weights(torch, W):
    """FP32-декодер из закреплённых весов (readout.*, cond.* чекпойнта)."""
    import types
    import k21_tree as k21t
    import k25_readout as kr25
    import k29_model as km
    H = int(W["readout.ln.weight"].shape[0])
    L = W["readout.L"].double().numpy()
    U = W["readout.U"].double().numpy()
    with torch.random.fork_rng(devices=[]):
        rd = k21t.build(torch)["Readout"](H, L, U)
        cm = kr25.build(torch)["CondModules"](H)
    rd.load_state_dict({k[len("readout."):]: v for k, v in W.items()
                        if k.startswith("readout.")})
    cm.load_state_dict({k[len("cond."):]: v for k, v in W.items()
                        if k.startswith("cond.")})
    src = types.SimpleNamespace(readout=rd, cond=cm, _L=L, _U=U)
    return km.build(torch)["Decoder"](src).freeze()


def cpu_decoder(torch, entry):
    if sha(entry["ckpt"]) != entry["ckpt_sha1"]:
        raise SystemExit("чекпойнт источника ≠ подготовке")
    ck = torch.load(entry["ckpt"], map_location="cpu", weights_only=False)
    return decoder_from_weights(torch, ck["weights"])


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    return float(np.abs(a - b).max() / max(1.0, float(np.abs(b).max())))


def rebuild_problems(cdir, dec, seed, space, task, init, scale_q):
    """reference -> правка узла -> потомки -> Zhat -> нормировка -> команда
    из закреплённых весов и сохранённых признаков развилки (CPU, FP32);
    плюс валидность всех строк (в т. ч. отсечённых по m)."""
    import torch
    import k31_readout as kr
    z = dict(np.load(os.path.join(cdir, "ref.npz")))
    rj = json.load(open(os.path.join(cdir, "ref.json"), encoding="utf-8"))
    miss = [k for k in FEAT if f"F_{k}" not in z]
    if miss:
        return [f"нет контекста пересборки {miss}"], {}
    E = kr.build(torch)
    F = {k: torch.as_tensor(z[f"F_{k}"]) for k in FEAT}
    st, p = {}, []
    with torch.no_grad():
        ref = dec(F)
    st["Z0_rel"] = _rel(ref["Z"][0].numpy(), z["Z0"])
    if not st["Z0_rel"] <= Z_TOL:
        p.append(f"Z0 пересборки ≠ записанному ({st['Z0_rel']:.2e})")
    path = kr.path_of(F["W2"][0].float().numpy())
    if list(path) != [int(x) for x in z["path"]]:
        p.append(f"путь {path} ≠ записанному {z['path'].tolist()}")
    call = int(rj["fork"]["call"])
    d = kc.tree_direction(kc.rng(space, seed, task, init, call, "tree"))[0]
    fl = kc.flat_arrays(kc.rng(space, seed, task, init, call, "flat"))
    if not (np.array_equal(d, z["d"]) and np.array_equal(fl, z["flat"])):
        p.append("направление/плоские массивы ≠ ключу RNG")
    o = E["edited"](dec, F, ref, list(kr.LEVELS), list(path),
                    np.stack([d] * 3))
    st["Zhat_rel"] = _rel(o["Zhat"].numpy(), z["Zhat"])
    if not st["Zhat_rel"] <= Z_TOL:
        p.append(f"Zhat пересборки ≠ записанному ({st['Zhat_rel']:.2e})")
    h, m = E["half_mid"](dec._L, dec._U, "cpu")
    P = kc.pool(torch, ref["Z"][0], o["Zhat"], fl, h, m)
    if not np.array_equal(P["valid"], np.asarray(z["valid"], bool)) or \
            not np.array_equal(P["allowed"], np.asarray(z["allowed"], bool)) \
            or int(P["m"]) != int(z["m"]):
        p.append(f"валидность/m пересборки {P['valid'].tolist()}/{P['m']} ≠ "
                 f"записанным {z['valid'].tolist()}/{int(z['m'])}")
    da, du, dl = [], [], []
    zv = np.asarray(z["valid"], bool)
    lam_rec = np.asarray(z["lam"], np.float64)
    for i in range(1, 7):
        if not zv[i]:
            if np.isfinite(lam_rec[i]) or np.isfinite(z["Du"][i]).any():
                p.append(f"строка {i}: λ/Du у недопустимой строки")
            continue
        Dr = np.asarray(z["Du"][i], np.float64)
        if not (np.isfinite(Dr).all() and np.isfinite(lam_rec[i])):
            p.append(f"строка {i}: записанные Du/λ не конечны")
            continue
        rms = float(np.sqrt(np.mean(Dr[:kc.N8] ** 2)))
        if not (abs(rms - kc.TARGET_RMS) <= kc.RMS_TOL and
                lam_rec[i] <= kc.LAMBDA_MAX and
                np.abs(Dr[:kc.N8]).max() <= kc.DU_MAX):
            p.append(f"строка {i}: записанный Du вне нормировки")
        if P["valid"][i]:
            du.append(float(np.abs(np.asarray(P["rows"][i]["Du"],
                                              np.float64) - Dr).max()))
            dl.append(abs(float(P["rows"][i]["lam"]) - lam_rec[i]) /
                      max(abs(float(P["rows"][i]["lam"])), 1e-12))
        if P["allowed"][i] and np.asarray(z["allowed"], bool)[i]:
            da.append(float(np.abs(P["arm"][i].numpy() - z["arms"][i]).max()))
    st["Du_max"] = max(du) if du else None
    st["lam_rel_max"] = max(dl) if dl else None
    st["arm_max"] = max(da) if da else None
    if du and not max(du) <= DU_TOL:
        p.append(f"Du пересборки ≠ записанному ({max(du):.2e})")
    if dl and not max(dl) <= LAM_TOL:
        p.append(f"λ пересборки ≠ записанной ({max(dl):.2e})")
    if da and not max(da) <= ARM_TOL:
        p.append(f"команды пересборки ≠ записанным ({max(da):.2e})")
    g = np.where(ref["grip_logit"][0, :kc.N8].numpy() > 0, 1.0, -1.0)
    if not np.array_equal(g.astype(np.float32), z["fork_grip"]):
        p.append("схват развилки ≠ логиту reference")
    import k29_policy as kp
    tau = int(rj["fork"]["tau"])
    A_ref = np.asarray(z["actions"], np.float32)
    q = np.asarray(scale_q, np.float64)
    for n in json.load(open(os.path.join(cdir, "case.json")))["order"]:
        base = n.split("#")[0]
        bz = np.load(os.path.join(cdir, f"b_{n}.npz"))
        if base == "noop":
            want = np.asarray(z["fork_ref_codec"], np.float32)
        else:
            want = np.concatenate([z["arms"][kc.ROWS.index(base)],
                                   z["fork_grip"][:, None]], 1).astype(
                np.float32)
        if not np.array_equal(bz["fork_codec"], want):
            p.append(f"ветвь {n}: записанный чанк ≠ кандидату")
        A = np.asarray(bz["actions"], np.float32)
        k = min(kc.N8, len(A) - tau)
        if k < 1:
            p.append(f"ветвь {n}: нет исполненных шагов после развилки")
            continue
        if not np.array_equal(A[:tau], A_ref[:tau]):
            p.append(f"ветвь {n}: префикс ≠ reference")
        ex = kp.scale_exec(want[None], q)[0][:k].astype(np.float32)
        if not np.array_equal(A[tau:tau + k], ex):
            p.append(f"ветвь {n}: исполненные физические команды ≠ "
                     f"масштабу ожидаемого чанка")
    return p, st


# --- fork-smoke -------------------------------------------------------------------
def fork_smoke(a):
    pr = fresh("real_check")
    if pr:
        raise SystemExit("smoke без свежего real-check: " + "; ".join(pr))
    p = load_prepare()
    jobs = [(s, "smoke", t, i) for s, t, i in SMOKE_CASES]
    t0 = time.time()
    L0 = ledger().spent("checks")
    out = run_cases(a.device, jobs, space=1, scope_of=lambda s: "checks",
                    limit_of=lambda s: None, root="data/k31/smoke",
                    smoke=True)
    spent = ledger().spent("checks") - L0
    prob = []
    import torch
    rebuild = {}
    for s in SEEDS:
        dcpu = cpu_decoder(torch, p["sources"][str(s)])
        for (s_, split, t, i) in [k for k in out if k[0] == s]:
            pr, st = rebuild_problems(case_dir(s_, split, t, i,
                                               "data/k31/smoke"),
                                      dcpu, s_, 1, t, i, p["scale_q"])
            rebuild[f"s{s_}_t{t}_i{i}"] = dict(problems=pr, **st)
            prob += [f"s{s_} t{t} i{i}: {x}" for x in pr]
        del dcpu
    for k, c in out.items():
        if not c.get("repeat_equal", False):
            prob.append(f"{k}: повтор кандидата не воспроизвёлся")
        if len(c["order"]) > 4:
            prob.append(f"{k}: исполнений больше 5")
    rep = dict(contract=contract(p), ok=not prob, problems=prob,
               rebuild=rebuild,
               cases={f"s{k[0]}_t{k[2]}_i{k[3]}": v for k, v in out.items()},
               physical_steps=spent, seconds=time.time() - t0,
               sec_per_step=(time.time() - t0) / max(spent, 1))
    rep["forecast_hours_AB_upper"] = rep["sec_per_step"] * 2 * 366000 / 3600
    save_json(SMOKE, rep)
    for k, v in rebuild.items():
        print(f"  пересборка {k}: Z0 {v.get('Z0_rel', float('nan')):.2e}, "
              f"Zhat {v.get('Zhat_rel', float('nan')):.2e}, |Δкоманд| "
              f"{v.get('arm_max')}, |ΔDu| {v.get('Du_max')}, |Δλ|/λ "
              f"{v.get('lam_rel_max')}; "
              + ("ок" if not v["problems"] else "; ".join(v["problems"])))
    print(f"  fork-smoke: шагов {spent}, {rep['seconds'] / 60:.1f} мин; "
          f"{rep['sec_per_step'] * 1e3:.1f} мс/шаг; верхняя оценка A+B "
          f"{rep['forecast_hours_AB_upper']:.1f} ч; "
          + ("пройден" if not prob else "НЕ пройден: " + "; ".join(prob)))
    return 0 if not prob else 3


# --- пилот A и гейт ------------------------------------------------------------------
def gate_report(cases):
    """cases: {(seed, split, task, init): case}."""
    lv_tree = ("root", "mid", "leaf")
    per, rescued_pairs = {}, set()
    for s in SEEDS:
        tr = {k: c for k, c in cases.items() if k[0] == s and k[1] == "train"}
        va = {k: c for k, c in cases.items() if k[0] == s and k[1] == "val"}
        cov = np.mean([any(c["allowed"][1:4]) for c in tr.values()]) \
            if tr else 0.0
        resc = {n: 0 for n in kc.ROWS[1:]}
        harm = {n: 0 for n in kc.ROWS[1:]}
        only = {n: 0 for n in kc.ROWS[1:]}
        o_tree = o_flat = 0
        ms = []
        for k, c in tr.items():
            lab = c["labels"]
            for n, v in lab.items():
                resc[n] += int(v > 0)
                harm[n] += int(v < 0)
            rs = [n for n, v in lab.items() if v > 0]
            if len(rs) == 1:
                only[rs[0]] += 1
            if any(lab.get(n, 0) > 0 for n in lv_tree):
                rescued_pairs.add((k[2], k[3]))
            o_tree += int(c["R0"] or any(lab.get(n, 0) > 0 for n in lv_tree))
            o_flat += int(c["R0"] or any(v > 0 for n, v in lab.items()
                                         if n.startswith("flat")))
            ms.append(int(sum(c["allowed"][1:4])))
        n = max(len(tr), 1)
        per[str(s)] = dict(
            train=len(tr), val=len(va), tree_coverage=float(cov),
            R0=float(np.mean([c["R0"] for c in tr.values()])) if tr else None,
            oracle_tree=o_tree / n, oracle_flat=o_flat / n, rescue=resc,
            harm=harm, rescue_only_level=only,
            m_hist=np.bincount(ms, minlength=4).tolist(),
            physical=int(sum(sum(c["physical"].values())
                             for k, c in cases.items() if k[0] == s)))
    cov_ok = all(per[str(s)]["tree_coverage"] >= GATE["tree_coverage"]
                 for s in SEEDS)
    res_ok = len(rescued_pairs) >= GATE["rescued_pairs"]
    status = "passed" if (cov_ok and res_ok) else (
        "stopped_low_proposal_coverage" if not cov_ok else
        "stopped_no_headroom")
    tasks = sorted({t for t, _ in rescued_pairs})
    return dict(status=status, per_seed=per,
                rescued_train_pairs=sorted([list(x) for x in rescued_pairs]),
                rescued_tasks=tasks,
                note="oracle — верхняя оценка на конечном пуле, не SR модели")


def load_cases(seed, phase_items):
    out = {}
    for split, t, i in phase_items:
        cdir = case_dir(seed, split, t, i)
        pr = kcol.case_problems(cdir, seed=seed, space=0, task=t, init=i)
        if pr:
            raise SystemExit(f"s{seed} {split} t{t} i{i}: {pr[:3]}")
        out[(seed, split, t, i)] = json.load(open(os.path.join(
            cdir, "case.json"), encoding="utf-8"))
    return out


def pilot(a):
    for k in ("real_check", "smoke"):
        pr = fresh(k)
        if pr:
            raise SystemExit(f"пилот без свежего {k}: " + "; ".join(pr))
    p = load_prepare()
    jobs = [(s, split, t, i) for s in SEEDS for split, t, i in items_of("A")]
    run_cases(a.device, jobs, space=0, scope_of=lambda s: f"dev_s{s}",
              limit_of=lambda s: A_LIMIT)
    cases = {}
    for s in SEEDS:
        cases.update(load_cases(s, items_of("A")))
    rep = dict(gate_report(cases), contract=contract(p),
               budget={k: ledger().spent(k) for k in CAPS})
    save_json(PILOT, rep)
    print_gate(rep)
    return 0 if rep["status"] == "passed" else 4


def print_gate(rep):
    for s, v in rep["per_seed"].items():
        print(f"  s{s}: train {v['train']}, val {v['val']}; R0 "
              f"{v['R0']:.2f}; покрытие tree {v['tree_coverage']:.2f}; oracle "
              f"tree/flat {v['oracle_tree']:.2f}/{v['oracle_flat']:.2f}; "
              f"rescue {v['rescue']}; harm {v['harm']}; m {v['m_hist']}; "
              f"шагов {v['physical']}")
    print(f"  спасённые tree train-пары: {rep['rescued_train_pairs']} (задачи "
          f"{rep['rescued_tasks']})")
    print(f"  ГЕЙТ A: {rep['status']}")


def collect(a):
    pr = fresh("pilot")
    if pr:
        raise SystemExit("добор без пилота: " + "; ".join(pr))
    rep_a = json.load(open(PILOT, encoding="utf-8"))
    if rep_a["status"] != "passed":
        raise SystemExit(f"гейт A не пройден: {rep_a['status']}")
    p = load_prepare()
    jobs = [(s, split, t, i) for s in SEEDS for split, t, i in items_of("B")]
    run_cases(a.device, jobs, space=0, scope_of=lambda s: f"dev_s{s}",
              limit_of=lambda s: None)
    cases = {}
    for s in SEEDS:
        cases.update(load_cases(s, items_of("A") + items_of("B")))
    rep = dict(gate_report(cases), contract=contract(p),
               budget={k: ledger().spent(k) for k in CAPS},
               n_cases={str(s): sum(1 for k in cases if k[0] == s)
                        for s in SEEDS})
    rep["status"] = "complete"
    save_json(COLLECT, rep)
    print(f"  добор B: случаев {rep['n_cases']}; бюджет {rep['budget']}")
    return 0


# --- набор, fitcheck, обучение ------------------------------------------------------
def dataset_path(seed):
    return f"data/k31/dataset_s{seed}.npz"


def build_dataset(seed):
    import k31_data as kdat
    pr = fresh("collect")
    if pr:
        raise SystemExit("набор без добора: " + "; ".join(pr))
    items = items_of("A") + items_of("B")
    tr, va = [], []
    for split, t, i in items:
        c = kdat.load_case(case_dir(seed, split, t, i), seed=seed, space=0,
                           task=t, init=i)
        (tr if split == "train" else va).append(c)
    D = kdat.build_dataset(tr, va)
    path = dataset_path(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as fh:
        np.savez_compressed(
            fh, Xt=D["train"]["X"], yt=D["train"]["y"],
            at=D["train"]["allowed"], kt=np.array(D["train"]["keys"]),
            Xv=D["val"]["X"], yv=D["val"]["y"], av=D["val"]["allowed"],
            kv=np.array(D["val"]["keys"]), pm_mean=D["pca_mean"]["mean"],
            pm_comp=D["pca_mean"]["comp"], pp_mean=D["pca_path"]["mean"],
            pp_comp=D["pca_path"]["comp"], st_mean=D["std"]["mean"],
            st_std=D["std"]["std"], sha=np.array(D["sha"]))
    os.replace(path + ".tmp", path)
    return D


def load_dataset(seed):
    z = dict(np.load(dataset_path(seed)))
    D = dict(train=dict(X=z["Xt"], y=z["yt"], allowed=z["at"],
                        keys=[tuple(x) for x in z["kt"]]),
             val=dict(X=z["Xv"], y=z["yv"], allowed=z["av"],
                      keys=[tuple(x) for x in z["kv"]]),
             pca_mean=dict(mean=z["pm_mean"], comp=z["pm_comp"]),
             pca_path=dict(mean=z["pp_mean"], comp=z["pp_comp"]),
             std=dict(mean=z["st_mean"], std=z["st_std"]),
             sha=str(z["sha"]))
    tk = {k[1] for k in D["train"]["keys"]}
    vk = {k[1] for k in D["val"]["keys"]}
    if tk & vk or not tk <= set(A_TRAIN + B_TRAIN) or \
            not vk <= set(A_VAL + B_VAL):
        raise SystemExit("утечка между train и val по init-банку")
    return D


def fitcheck_all(a):
    import torch
    import k31_train as kt
    p = load_prepare()
    rep = dict(contract=contract(p), seeds={}, datasets={})
    for s in SEEDS:
        if not os.path.exists(dataset_path(s)):
            build_dataset(s)
        D = load_dataset(s)
        rep["datasets"][str(s)] = sha(dataset_path(s))
        rep["seeds"][str(s)] = kt.fitcheck(torch, D, s)
        v = rep["seeds"][str(s)]
        print(f"  fitcheck s{s}: CE {v['ce0']:.3f} -> {v['ce1']:.3f} "
              f"(n {v['n']}, классы {v['classes']}, базовая "
              f"{v['baseline_ce']:.3f}); {'ок' if v['ok'] else 'НЕТ'}")
    rep["ok"] = all(v["ok"] for v in rep["seeds"].values())
    save_json(FITCHECK, rep)
    return 0 if rep["ok"] else 3


def selector_path(seed):
    return f"data/k31/selector_s{seed}.pt"


def train_all(a):
    pr = fresh("fitcheck")
    if pr:
        raise SystemExit("обучение без свежего fitcheck: " + "; ".join(pr))
    import torch
    import k31_train as kt
    p = load_prepare()
    rep = dict(contract=contract(p), seeds={})
    for s in SEEDS:
        D = load_dataset(s)
        tr = kt.Trainer(torch, D, s, f"data/k31/sel_s{s}",
                        extra_contract=dict(code=p["code"],
                                            contract=contract(p)))
        best = tr.run()
        meta = kt.save_selector(torch, selector_path(s), best, D, s,
                                dict(contract=contract(p)))
        S = kt.Selector(torch, selector_path(s))
        ev = kt.evaluate(torch, S.mod, D["val"])
        rep["seeds"][str(s)] = dict(
            best_update=best["u"], val_ce=best["ce"], meta=meta,
            selector_sha1=sha(selector_path(s)), dataset_sha=D["sha"],
            val=ev, counts=dict(train=len(D["train"]["X"]),
                                val=len(D["val"]["X"])), log=tr.log)
        print(f"  s{s}: best {best['u']}, val CE {best['ce']:.4f}; "
              f"вмешательства на val tree/flat "
              f"{ev['would_intervene']['tree']}/"
              f"{ev['would_intervene']['flat']}")
    active = [s for s in SEEDS
              if rep["seeds"][str(s)]["val"]["would_intervene"]["tree"] > 0]
    rep["status"] = "trained" if active else "stopped_inactive_selector"
    rep["active_seeds"] = active
    save_json(TRAIN, rep)
    print(f"  СЕЛЕКТОР: {rep['status']} (активные сиды {active})")
    return 0 if active else 4


# --- план SR ------------------------------------------------------------------------
def eval_plan_path(seed):
    return f"reports/k31/plan_eval_s{seed}.json"


def make_eval_plan(p, trn, seed, half, mid, sel_sha1):
    """Чистая сборка плана SR (без файлов) — тот же формат источника, что
    у prepare: plan["sources"][str(seed)]."""
    import k31_harness as kh
    s = int(seed)
    L = np.asarray(p["bounds"]["L"], np.float64)[:6]
    U = np.asarray(p["bounds"]["U"], np.float64)[:6]
    sel = dict(path=selector_path(s), sha1=sel_sha1,
               weights_hash=trn["seeds"][str(s)]["meta"]["weights_hash"])
    if sel["sha1"] != trn["seeds"][str(s)]["selector_sha1"]:
        raise SystemExit(f"s{s}: селектор ≠ отчёту обучения")
    arms = {kh.label(k, s): dict(label=kh.label(k, s), seed=s,
                                 contract=contract(p)) for k in kh.KINDS}
    return dict(seed=s, sources={str(s): p["sources"][str(s)]},
                selector=sel, tasks=p["tasks"], init_ref=p["init_ref"],
                half=[float(x) for x in half], mid=[float(x) for x in mid],
                L=L.tolist(), U=U.tolist(), scale_q=p["scale_q"], arms=arms,
                contract=contract(p), code=p["code"],
                order=["ref s0", "ref s1", "repair s0", "flatrepair s0",
                       "repair s1", "flatrepair s1"])


def plan_eval(a):
    import torch
    import k31_readout as kr
    p = load_prepare()
    if not os.path.exists(TRAIN):
        raise SystemExit("нет обучения селекторов")
    trn = json.load(open(TRAIN, encoding="utf-8"))
    if trn.get("contract") != contract(p) or trn["status"] != "trained":
        raise SystemExit(f"обучение: статус {trn.get('status')} или другой "
                         f"контракт")
    E = kr.build(torch)
    L = np.asarray(p["bounds"]["L"], np.float64)[:6]
    U = np.asarray(p["bounds"]["U"], np.float64)[:6]
    h, m = E["half_mid"](L, U, "cpu")
    for s in SEEDS:
        path = eval_plan_path(s)
        plan = make_eval_plan(p, trn, s, h, m, sha(selector_path(s)))
        source_entry_of(plan, s)                  # тот же путь, что у SR
        if os.path.exists(path):
            old = json.load(open(path, encoding="utf-8"))
            if old != json.loads(json.dumps(plan)):
                raise SystemExit(f"{path} закреплён и отличается")
            print(f"  {path}: закреплён, сверка пройдена")
            continue
        save_json(path, plan)
        print(f"  {path}: записан")
    return 0


def load_eval_plan(seed):
    p = load_prepare()
    path = eval_plan_path(seed)
    if not os.path.exists(path):
        raise SystemExit(f"нет {path}")
    plan = json.load(open(path, encoding="utf-8"))
    if plan["contract"] != contract(p):
        raise SystemExit(f"{path}: другой контракт")
    ps = sha(path)
    for v in plan["arms"].values():
        v["plan_sha1"] = ps
    return plan


def load_selector(torch, plan, seed):
    import k31_train as kt
    s = plan["selector"]
    if sha(s["path"]) != s["sha1"]:
        raise SystemExit("селектор ≠ плану SR")
    S = kt.Selector(torch, s["path"])
    if S.meta["weights_hash"] != s["weights_hash"]:
        raise SystemExit("веса селектора ≠ плану")
    return S


def rebuild_selftest():
    import tempfile
    import torch
    import k21_model as km
    import k21_train as kt
    import k25_model as k25m
    import k29_model as k29m
    import k31_policy as kpol
    k29m.no_tf32(torch)
    M = k29m.build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    mod = k25m.build(torch)["K25"](km.fake_bar(torch), L, U, seed=0).eval()
    with torch.no_grad():
        g = torch.Generator().manual_seed(3)
        for q in list(mod.cond.parameters()) + list(mod.readout.parameters()):
            q.add_(torch.randn(q.shape, generator=g) * 0.05)
    for q in mod.parameters():
        q.requires_grad_(False)
    dec = M["Decoder"](mod).freeze()
    dw = decoder_from_weights(torch, kt.trainable_state(mod))
    bt = km.fake_batch(torch, B=1, seed=2)
    st8 = np.zeros((1, 8))
    arm = kpol.Arm(torch, mod, dec, M, mode="reference", source_seed=0,
                   space=1)
    arm.begin(8, 1)
    arm.decide(bt, 1, st8, 0)
    with torch.no_grad():
        assert torch.equal(dw(arm.last["F"])["Z"], dec(arm.last["F"])["Z"])
    pay = capture(arm)
    fork = dict(call=pay["call"], tau=0)
    pool = make_pool_fn(arm)(pay, fork)
    grip = arm.last["rec"]["a_codec"][:, 6].copy()
    import k29_policy as kp
    q = np.array([0.9375, 0.91, 0.9375, 0.2036, 0.2636, 0.375, 1.0])
    i = int(np.nonzero(pool["allowed"][1:])[0][0]) + 1
    name = kc.ROWS[i]
    ref_codec = arm.last["rec"]["a_codec"]
    prefix = np.random.default_rng(0).normal(size=(16, 7)).astype(np.float32)
    fork["tau"] = 16
    with tempfile.TemporaryDirectory() as td:
        def branch(nm, chunk, k=8, act_over=None):
            ex = kp.scale_exec(chunk[None], q)[0][:k].astype(np.float32)
            A = np.concatenate([prefix, ex])
            if act_over is not None:
                A = act_over(A.copy())
            np.savez(os.path.join(td, f"b_{nm}.npz"), actions=A,
                     fork_codec=chunk)

        def write(npz_over=None, act_over=None, k=8):
            npz = dict(pool["npz"], allowed=pool["allowed"],
                       arms=pool["arms"], fork_grip=grip,
                       fork_ref_codec=ref_codec,
                       actions=np.concatenate([prefix, np.zeros((5, 7),
                                                                np.float32)]))
            npz.update(npz_over or {})
            np.savez(os.path.join(td, "ref.npz"), **npz)
            json.dump(dict(fork=fork), open(os.path.join(td, "ref.json"),
                                            "w"))
            json.dump(dict(order=["noop", name]), open(os.path.join(
                td, "case.json"), "w"))
            branch("noop", ref_codec.astype(np.float32))
            cand = np.concatenate([pool["arms"][i], grip[:, None]],
                                  1).astype(np.float32)
            branch(name, cand, k=k, act_over=act_over)
            return rebuild_problems(td, dw, 0, 1, 8, 1, q)
        pr, st = write()
        assert pr == [], pr
        assert st["Z0_rel"] == 0.0 and st["arm_max"] in (0.0, None)
        assert write(k=3)[0] == []                 # раннее завершение
        Du2 = pool["npz"]["Du"].copy()
        Du2[i] *= 2.0
        assert write(dict(Du=Du2))[0]              # удвоенный Du
        l2 = pool["npz"]["lam"].copy()
        l2[i] = np.inf
        assert write(dict(lam=l2))[0]              # λ = Inf
        l3 = pool["npz"]["lam"].copy()
        l3[i] *= 1.01
        assert write(dict(lam=l3))[0]              # λ ≠ пересборке
        vcut = [j for j in range(1, 7) if pool["npz"]["valid"][j]]
        Du3 = pool["npz"]["Du"].copy()
        Du3[vcut[-1], 0, 0] += 1e-4                # последняя допустимая строка
        assert write(dict(Du=Du3))[0]

        def bump(A):
            A[16, 0] += 1e-6
            return A
        assert any("физические" in x for x in write(act_over=bump)[0])

        def bump_prefix(A):
            A[3, 1] += 1e-6
            return A
        assert any("префикс" in x for x in write(act_over=bump_prefix)[0])
        bad = pool["arms"].copy()
        bad[i, 0, 0] += 1e-5
        assert write(dict(arms=bad))[0]
        assert write(dict(d=-pool["npz"]["d"]))[0]
        zh = pool["npz"]["Zhat"].copy()
        zh[0, 0, 0] += 1.0
        assert write(dict(Zhat=zh))[0]
        assert write(dict(path=np.array([0, 0, 0])))[0] or \
            list(pool["npz"]["path"]) == [0, 0, 0]
        vv = pool["npz"]["valid"].copy()
        vv[i] = not vv[i]
        assert write(dict(valid=vv))[0]
        miss = {k: v for k, v in pool["npz"].items() if k != "F_W2"}
        np.savez(os.path.join(td, "ref.npz"), **miss, allowed=pool["allowed"],
                 arms=pool["arms"], fork_grip=grip)
        assert rebuild_problems(td, dw, 0, 1, 8, 1, q)[0]


def selftest():
    b = banks()
    assert set(b["pilot_train"]) == set(range(8)) and \
        b["more_val"] == [22, 23, 24]
    assert len(items_of("A")) == 30 and len(items_of("B")) == 45
    import k31_train as kt
    cfg = dict(selector=kt.CFG, candidates=kc.CONFIG, caps=CAPS, gate=GATE)
    written = json.loads(json.dumps(cfg))         # как prepare.json на диске
    fresh_ = json.loads(json.dumps(cfg, default=str))   # как в prepare()
    assert all(written[k] == fresh_[k] for k in cfg), \
        "повторная сверка prepare видит изменение без изменений"
    # гейт: покрытие, две различные спасённые пары, сиды не удваивают
    def case(R0, lab, allowed=(True,) * 7):
        return dict(R0=R0, labels=lab, allowed=list(allowed),
                    physical=dict(ref=100))
    base = dict(root=0, mid=0, leaf=0, flat0=0, flat1=0, flat2=0)
    cs = {}
    for s in SEEDS:
        for i in range(8):
            cs[(s, "train", 0, i)] = case(False, dict(base))
        cs[(s, "val", 0, 20)] = case(False, dict(base))
    assert gate_report(cs)["status"] == "stopped_no_headroom"
    cs[(0, "train", 0, 1)] = case(False, dict(base, mid=1))
    cs[(1, "train", 0, 1)] = case(False, dict(base, leaf=1))
    assert gate_report(cs)["status"] == "stopped_no_headroom"   # одна пара
    cs[(1, "train", 0, 2)] = case(False, dict(base, root=1))
    r = gate_report(cs)
    assert r["status"] == "passed" and r["rescued_train_pairs"] == [[0, 1],
                                                                   [0, 2]]
    cs[(0, "train", 0, 3)] = case(False, dict(base), (True, False, False,
                                                      False, True, True,
                                                      True))
    cs[(0, "train", 0, 4)] = case(False, dict(base), (True, False, False,
                                                      False, True, True,
                                                      True))
    assert gate_report(cs)["status"] == "stopped_low_proposal_coverage"
    # только flat-спасения не открывают гейт
    cs2 = {k: case(False, dict(base, flat0=1)) for k in cs}
    assert gate_report(cs2)["status"] == "stopped_no_headroom"
    assert set(MODEL_CODE) >= {"k31_protocol.py", "k31_harness.py"}
    # план SR -> загрузчик источника, которым пользуется SR (k31_harness
    # run_eval -> load_source_dec -> source_entry_of)
    src = {str(s): dict(seed=s, ckpt=f"c{s}", weights_hash=f"w{s}")
           for s in SEEDS}
    fake_p = dict({k: None for k in CONTRACT_KEYS}, sources=src,
                  bounds=dict(L=[-1.0] * 7, U=[1.0] * 7), tasks={"0": "x"},
                  init_ref={}, scale_q=[1.0] * 7, code={})
    trn = dict(seeds={str(s): dict(meta=dict(weights_hash=f"h{s}"),
                                   selector_sha1=f"S{s}") for s in SEEDS})
    for s in SEEDS:
        pl = make_eval_plan(fake_p, trn, s, [1.0] * 6, [0.0] * 6, f"S{s}")
        pl = json.loads(json.dumps(pl))           # как после записи в файл
        assert source_entry_of(pl, s) == src[str(s)]
        assert source_entry_of(fake_p, s) == src[str(s)]
        try:
            make_eval_plan(fake_p, trn, s, [1.0] * 6, [0.0] * 6, "other")
            raise AssertionError("чужой селектор принят")
        except SystemExit:
            pass
    try:
        source_entry_of(dict(source=src["0"]), 0)
        raise AssertionError("старый формат принят молча")
    except SystemExit:
        pass
    rebuild_selftest()
    print("самопроверка k31_protocol пройдена: CPU-декодер из весов = "
          "декодеру источника побитово; независимая пересборка развилки "
          "проходит (и при раннем завершении), подмена команд, удвоенный Du, "
          "λ = Inf/≠, Du допустимой строки, направление, Zhat, валидность, "
          "физическая команда при верном fork_codec, префикс — отказ; "
          "план SR -> загрузчик "
          "источника SR (тот же формат, что prepare), чужой селектор — отказ; "
          "банки не пересекаются и не "
          "задевают 25…29/45…49; гейт A: покрытие tree ≥ 80% каждого сида, "
          "≥ 2 различных спасённых train-пар (одна пара в двух сидах не "
          "удваивается, flat не считается)")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-31: протокол")
    ap.add_argument("cmd", choices=("prepare", "real-check", "fork-smoke",
                                    "pilot", "gate", "collect", "dataset",
                                    "fitcheck", "train", "plan-eval",
                                    "check-fresh", "status"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--kind", default=None)
    a = ap.parse_args()
    try:
        if a.cmd == "prepare":
            return prepare()
        if a.cmd == "real-check":
            return real_check(a)
        if a.cmd == "fork-smoke":
            return fork_smoke(a)
        if a.cmd == "pilot":
            return pilot(a)
        if a.cmd == "gate":
            rep = json.load(open(PILOT, encoding="utf-8"))
            print_gate(rep)
            return 0 if rep["status"] == "passed" else 4
        if a.cmd == "collect":
            return collect(a)
        if a.cmd == "dataset":
            for s in SEEDS:
                D = build_dataset(s)
                print(f"  s{s}: train {len(D['train']['X'])}, val "
                      f"{len(D['val']['X'])}, sha {D['sha'][:12]}")
            return 0
        if a.cmd == "fitcheck":
            return fitcheck_all(a)
        if a.cmd == "train":
            return train_all(a)
        if a.cmd == "plan-eval":
            return plan_eval(a)
        if a.cmd == "check-fresh":
            pr = fresh(a.kind)
            print("  " + ("свежо" if not pr else "; ".join(pr)))
            return 0 if not pr else 1
        if a.cmd == "status":
            if os.path.exists(TRAIN):
                print(json.load(open(TRAIN, encoding="utf-8"))["status"])
            return 0
    except kcol.Mismatch as e:
        print(f"  ОСТАНОВ (воспроизведение): {e}", flush=True)
        return 3
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
