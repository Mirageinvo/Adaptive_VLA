#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-37: быстрый пилот «родитель плюс лист» поверх T32 (K-32).

Команды:
  selftest
  prepare            — reports/k37/prepare.json (неизменяем): K-36 prepare
                       (входы K-32/K-25/банка проверены его загрузчиком),
                       15 dev-сцен (space 0, задачи 0/6/8 по 5), строки K-21,
                       лимиты, правило, код.
  audit              — CPU: по K-36 bank-smoke T32 (если перенесён): доли
                       root/mid/leaf, срабатывания ограничителя по j_D, на
                       снимках первых вызовов — быстрая формула = полной
                       пересборке листьев для заданных u.
  cache --device D   — признаки строк K-21 для обоих сидов (GPU).
  fit                — головки (CPU), data/k37/head_s<S>.pt.
  pilot --device D   — 15 сцен × 2 сида × {T1 = T32, T2 = T32 + лист} и две
                       проверки нулевой головки (s0 t0, s1 t8 — первый
                       dev-случай); B1, свежая среда, 10 waiting, до 600.
  analyze            — W/L/D по задачам и сидам, решение в целых числах.
  status
Журнал data/k37/ledger.json: pilot 60 эпизодов, checks 2 (по 610 шагов);
транзакции эпизодов — k36b_gate.transact (проверка до settle/commit).
Эпизоды T1 проверяет k36_harness.episode_problems; T2 — пересборка
команд из отдельных слагаемых (D_parent, D_leaf, ограничитель, tanh) и
связь с физическими действиями; база T2 — проверка журнала K-32.
"""
import argparse
import os
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
import k37_fit as kfit  # noqa: E402
import k37_operator as kop  # noqa: E402

OUT = "reports/k37"
DATA = "data/k37"
PREPARE = os.path.join(OUT, "prepare.json")
LEDGER = os.path.join(DATA, "ledger.json")
CAPS = dict(pilot=60 * pc.EP, checks=2 * pc.EP)
TASKS = (0, 6, 8)
N_DEV = 5
IDENT = ((0, 0), (1, 8))                      # (сид, задача) — первый случай
STAGES = dict(t1="pilot", t2="pilot", ident="checks")
RULE = dict(candidate=dict(D_min=2, per_seed_min=0, scenes_min=2),
            negative=dict(D_max=-2, per_seed_max=0))
CODE = ("k37_operator.py", "k37_fit.py", "k37_parent_leaf.py",
        "k36b_gate.py", "k36_harness.py", "k36_bank.py", "k32_policy.py",
        "k32_operator.py", "k32_model.py", "k31_readout.py",
        "k31_collect.py")


def install_paths():
    """Эпизоды K-37 — в reports/k37 (модули K-36 переиспользуются)."""
    kh.OUT = OUT
    kh.REJECTED = os.path.join(OUT, "rejected")
    for k, v in STAGES.items():
        kh.STAGE_SCOPE[k] = v


def ledger():
    return kcol.Ledger(LEDGER, CAPS)


def code_shas():
    return {f: pc.sha(os.path.join(HERE, f)) for f in CODE}


# --- prepare --------------------------------------------------------------------------
def dev_list():
    out = []
    for t in TASKS:
        for cid, pos, ih, ihf in kb36.dev_cases(t, N_DEV):
            out.append(dict(task=t, case_id=cid, pos=int(pos), hash=[ih, ihf]))
    return out


def make_prepare():
    import k21_data as d21
    k36 = pc.load_prepare()
    D = d21.load_prepared(require_teacher=False)
    sel = kfit.select_rows(D, k36["k32_plan"]["tasks"])
    return pc.jnorm(dict(
        kind="k37_prepare", k36_prepare_sha1=k36["sha1"], dev=dev_list(),
        ident=[list(x) for x in IDENT], rows=sel,
        rows_sha1=kfit.rows_sha(sel), caps=CAPS, rule=RULE, cfg=kfit.CFG,
        leaf_alpha=kop.LEAF_ALPHA, code=code_shas(),
        actors={str(s): k36["k32"]["seeds"][str(s)]["T"]["actor_hash"]
                for s in (0, 1)}))


def prepare(a):
    new = make_prepare()
    if os.path.exists(PREPARE):
        old = pc.load_json(PREPARE)
        diff = sorted(k for k in set(old) | set(new)
                      if k != "created" and old.get(k) != new.get(k))
        if diff:
            raise SystemExit(f"{PREPARE} закреплён иначе: {diff}")
        print(f"  {PREPARE}: закреплён, совпадает")
    else:
        pc.save_json(PREPARE, dict(new, created=time.strftime(
            "%Y-%m-%dT%H:%M:%S")))
        print(f"  {PREPARE}: записан")
    for t in TASKS:
        r = new["rows"][str(t)]
        print(f"    t{t}: строк обучения {len(r['train'])}, диагностики "
              f"{len(r['diag'])} (эпизодов {r['n_episodes']}); dev-сцен "
              f"{sum(1 for d in new['dev'] if d['task'] == t)}")
    return 0


def load_prepare():
    if not os.path.exists(PREPARE):
        raise SystemExit("нет prepare K-37")
    d = pc.load_json(PREPARE)
    k36 = pc.load_prepare()
    if d["k36_prepare_sha1"] != k36["sha1"]:
        raise SystemExit("prepare K-37 от другого prepare K-36")
    bad = [f for f, s in d["code"].items() if code_shas().get(f) != s]
    if bad:
        raise SystemExit(f"код изменён после prepare K-37: {bad}")
    d["sha1"] = pc.sha(PREPARE)
    return d, k36


# --- кеш и обучение -------------------------------------------------------------------
def cache_path(s):
    return os.path.join(DATA, f"cache_s{s}.npz")


def head_path(s):
    return os.path.join(DATA, f"head_s{s}.pt")


def _cache_seed(torch, p37, k36, D, s, device):
    import k20_data as kd20
    import k31_readout as kr
    v = kh.load_source(torch, s, device)
    base = kh.make_arm(torch, k36, f"T32_s{s}", v)
    t2 = kop.T2Arm(torch, base, kop.make_head(torch))
    E = kr.build(torch)
    keys = ("x", "w", "c", "Dp", "Z0", "tgt", "row", "split", "task")
    acc = {k: [] for k in keys}
    reasons, check = {}, None
    for t in TASKS:
        for split, nm in ((0, "train"), (1, "diag")):
            for r in p37["rows"][str(t)][nm]:
                batch = kd20.prompt_batch(v["bar"], [np.array(D["IMG"][r],
                                                              copy=True)],
                                          kd20.norm_states(np.asarray(
                                              D["state8"][[r]])),
                                          [D["task"][r]])
                t2.begin(t, 0, 0)
                a, rec = t2.decide(batch, int(D["pos_offset"][r]),
                                   np.asarray(D["state8"][[r]], np.float64),
                                   int(D["step"][r]))
                key = f"t{t}_{nm}"
                reasons.setdefault(key, {}).setdefault(
                    int(rec["k37_reason"]), 0)
                reasons[key][int(rec["k37_reason"])] += 1
                if rec["k37_reason"] != 0:
                    continue
                if check is None:          # сверка быстрой и полной сборки
                    u = np.array([0.7, -0.4, 0.2, 1.1, -0.9, 0.3], np.float32)
                    Dl = kop.leaf_delta(rec["k37_w"], rec["k37_c"], u)
                    Dref = kop.full_reference(
                        torch, E, v["dec"], t2.cap["F"], t2.cap["ref"],
                        rec["r"], rec["path"], rec["eta"],
                        rec["c"][rec["r"]], rec["k37_leaf"], u)
                    arm_f, _du, _l = kop.compose(torch, rec["Z0"], t2.half,
                                                 t2.mid, rec["D"], Dl)
                    Du_r, _l2 = __import__("k32_operator").limit(torch, Dref)
                    arm_r = t2.mid + t2.half * np.tanh(
                        np.asarray(rec["Z0"], np.float32)[:8] / t2.half +
                        Du_r.numpy())
                    check = dict(row=int(r), dD=float(np.abs(
                        np.asarray(rec["D"]) + Dl - Dref).max()),
                        darm=float(np.abs(arm_f - arm_r).max()))
                acc["x"].append(rec["k37_x"])
                acc["w"].append(rec["k37_w"])
                acc["c"].append(rec["k37_c"])
                acc["Dp"].append(np.asarray(rec["D"], np.float32))
                acc["Z0"].append(np.asarray(rec["Z0"], np.float32)[:8])
                acc["tgt"].append(np.asarray(D["act"][r], np.float32)[:8, :6])
                acc["row"].append(r)
                acc["split"].append(split)
                acc["task"].append(t)
    out = {k: np.asarray(v_) for k, v_ in acc.items()}
    out["half"], out["mid"] = t2.half, t2.mid
    return out, reasons, check


def cache(a):
    import torch
    import k21_data as d21
    install_paths()
    p37, k36 = load_prepare()
    D = d21.load_prepared(require_teacher=False)
    rep = {}
    for s in (0, 1):
        z, reasons, check = _cache_seed(torch, p37, k36, D, s, a.device)
        kh.free(torch)
        prob = []
        if check is None:
            prob.append("нет подходящих вызовов — обучение невозможно")
        elif not (check["dD"] <= kfit.CHECK_TOL["D"] and
                  check["darm"] <= kfit.CHECK_TOL["arm"]):
            prob.append(f"быстрая сборка ≠ полной {check}")
        os.makedirs(DATA, exist_ok=True)
        with open(cache_path(s) + ".tmp", "wb") as fh:
            np.savez(fh, **z)
        os.replace(cache_path(s) + ".tmp", cache_path(s))
        rep[str(s)] = dict(reasons=reasons, check=check, problems=prob,
                           n=int(len(z["row"])),
                           sha1=pc.sha(cache_path(s)))
        print(f"  кеш s{s}: подходящих {len(z['row'])} (причины "
              f"{reasons}); сверка {check}; " +
              ("ок" if not prob else "; ".join(prob)), flush=True)
    rep.update(prepare_sha1=p37["sha1"],
               ok=all(not v["problems"] for v in rep.values()
                      if isinstance(v, dict)))
    pc.save_json(os.path.join(OUT, "cache.json"), rep)
    return 0 if rep["ok"] else 3


def fit(a):
    import torch
    p37, _k36 = load_prepare()
    c = pc.load_json(os.path.join(OUT, "cache.json"))
    if c.get("ok") is not True or c.get("prepare_sha1") != p37["sha1"]:
        raise SystemExit("кеш не пройден или от другого prepare")
    for s in (0, 1):
        if pc.sha(cache_path(s)) != c[str(s)]["sha1"]:
            raise SystemExit(f"кеш s{s} изменён")
        z = dict(np.load(cache_path(s)))
        head, rep = kfit.train_head(torch, z, s)
        hh = kfit.head_hash(torch, head)
        torch.save(dict(head=head.state_dict(), head_hash=hh, seed=s,
                        prepare_sha1=p37["sha1"],
                        cache_sha1=c[str(s)]["sha1"]), head_path(s) + ".tmp")
        os.replace(head_path(s) + ".tmp", head_path(s))
        rep.update(head_hash=hh, head_sha1=pc.sha(head_path(s)),
                   prepare_sha1=p37["sha1"])
        pc.save_json(os.path.join(OUT, f"fit_s{s}.json"), rep)
        print(f"  головка s{s}: train {rep['n_train']} / diag "
              f"{rep['n_diag']}; потеря diag {rep['diag']} (нулевая "
              f"{rep['diag_zero']}); изменение весов "
              f"{rep['weight_change']:.3g}", flush=True)
    return 0


def load_head(torch, p37, s):
    f = pc.load_json(os.path.join(OUT, f"fit_s{s}.json"))
    if f.get("prepare_sha1") != p37["sha1"] or \
            pc.sha(head_path(s)) != f.get("head_sha1"):
        raise SystemExit(f"головка s{s} ≠ отчёту обучения")
    ck = torch.load(head_path(s), map_location="cpu", weights_only=False)
    head = kop.make_head(torch)
    head.load_state_dict(ck["head"])
    if kfit.head_hash(torch, head) != f["head_hash"]:
        raise SystemExit(f"веса головки s{s} ≠ отчёту")
    return head, f["head_hash"]


# --- эпизоды пилота ---------------------------------------------------------------------
def wants(p37, k36, s, d, head_hash):
    pl = dict(k36, sha1=p37["sha1"])
    w1 = kh.want_of(pl, "t1", f"T32_s{s}", d["task"], d["case_id"], d["pos"],
                    "bank", space=kb36.SPACE_DEV, case_id=d["case_id"],
                    want_hash=None if d["hash"] is None else tuple(d["hash"]))

    def t2(stage, hh):
        return dict(w1, stage=stage, scope=STAGES[stage], label=f"T2_s{s}",
                    kind="T2", actor_hash=f"{w1['actor_hash']}+{hh}",
                    eid=f"{STAGES[stage]}/{stage}/T2_s{s}/t{d['task']}/"
                        f"{d['case_id']}")
    return w1, t2("t2", head_hash), t2("ident", "zero")


def save_t2(want, ep):
    import k32_harness as kh32
    jp, np_ = kh.ep_paths(want["stage"], want["label"], want["task"],
                          want["key"])
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    calls = [dict(c, rec=dict(c["rec"], a_codec=c["rec"]["base_a_codec"]))
             for c in ep["calls"]]
    J = kh32.pack_journal([(0, calls)], 6)
    n = len(ep["calls"])
    K = dict(a2=np.zeros((n, 8, 7), np.float32), reason=np.zeros(n, int),
             leaf=np.full(n, -1), w=np.zeros((n, 8), np.float32),
             c=np.zeros(n, np.float32), u=np.zeros((n, 6), np.float32),
             Dl=np.zeros((n, 8, 6), np.float32),
             Du2=np.zeros((n, 8, 6), np.float32))
    for i, c in enumerate(ep["calls"]):
        r = c["rec"]
        K["a2"][i] = c["a_codec"]
        K["reason"][i], K["leaf"][i] = r["k37_reason"], r["k37_leaf"]
        K["w"][i], K["c"][i], K["u"][i] = r["k37_w"], r["k37_c"], r["k37_u"]
        K["Dl"][i], K["Du2"][i] = r["k37_Dl"], r["k37_Du2"]
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


def t2_problems(rec, z, want, C):
    """T2: база — журнал K-32; добавка листа и ограничитель пересобраны из
    слагаемых; исполненное = масштабу итоговых команд; сводка."""
    import torch
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
    need = ("a2", "reason", "leaf", "w", "c", "u", "Dl", "Du2")
    if not J or any(k not in K for k in need) or len(J.get("ep", [])) == 0:
        return p + ["журнал T2 неполон или пуст"]
    n = len(J["ep"])
    if any(len(K[k]) != n for k in need):
        return p + ["журнал T2: длины ≠ числу вызовов"]
    for k in ("a2", "w", "c", "u", "Dl", "Du2"):
        if not np.isfinite(K[k]).all():
            return p + [f"журнал T2: {k} не конечен"]
    q = C["q"]
    A_base = np.concatenate([kp.scale_exec(J["a_codec"][i][None], q)[0]
                             [:int(J["k"][i])] for i in range(n)])
    p += [f"база: {x}" for x in kh32.journal_problems(
        J, [A_base], [s["done_step"]], C["half"], C["mid"], C["L"], C["U"],
        q, "T")]
    for i in range(n):
        base_a, a2 = J["a_codec"][i], K["a2"][i]
        if not np.array_equal(a2[:, 6], base_a[:, 6]):
            p.append(f"вызов {i}: схват ≠ T32")
        if int(K["reason"][i]) == 0 and K["u"][i].any():
            Dl = kop.leaf_delta(K["w"][i], float(K["c"][i]), K["u"][i])
            arm, Du2, _l = kop.compose(torch, J["Z0"][i], C["half"],
                                       C["mid"], J["D"][i], Dl)
            if not (np.array_equal(Dl, K["Dl"][i]) and
                    np.array_equal(Du2, K["Du2"][i]) and
                    np.array_equal(arm.astype(np.float32), a2[:, :6])):
                p.append(f"вызов {i}: команда T2 ≠ пересборке из слагаемых")
            if K["leaf"][i] < 0 or not J["has_actor"][i] or J["zero"][i]:
                p.append(f"вызов {i}: активная правка без родителя/листа")
        elif not np.array_equal(a2, base_a):
            p.append(f"вызов {i}: неактивный вызов ≠ команде T32")
    ex = np.concatenate([kp.scale_exec(K["a2"][i][None], q)[0]
                         [:int(J["k"][i])] for i in range(n)])
    if not np.array_equal(ex.astype(np.float32), np.asarray(A, np.float32)):
        p.append("исполненное ≠ масштабу итоговых команд T2")
    p += kh.summary_problems(rec, s, len(A))
    return p


def ident_case(p37, s, d):
    """Проверка нулевой головки: (сид, задача) из p37["ident"] и первый
    dev-случай этой задачи."""
    first = next(x["case_id"] for x in p37["dev"] if x["task"] == d["task"])
    return [s, d["task"]] in p37["ident"] and d["case_id"] == first


def all_wants(p37, k36, hh):
    """eid -> want по обоим сидам (для проверки открытого резерва)."""
    out = {}
    for s in (0, 1):
        for d in p37["dev"]:
            w1, w2, wz = wants(p37, k36, s, d, hh[s])
            out[w1["eid"]], out[w2["eid"]] = w1, w2
            if ident_case(p37, s, d):
                out[wz["eid"]] = wz
    return out


def validator(C):
    def val(rec, z, w):
        if w["kind"] == "T2":
            return t2_problems(rec, z, w, C)
        return kh.episode_problems(rec, z, w, C)
    return val


def pilot(a, sim=None, src_loader=None, heads=None):
    """По сиду: на каждой сцене T1, (проверка нулевой головки), T2."""
    import torch
    import k36b_gate as kg
    install_paths()
    p37, k36 = load_prepare()
    C = kh.consts(k36)
    L = ledger()
    H = heads or {s: load_head(torch, p37, s) for s in (0, 1)}
    hh = {s: H[s][1] for s in (0, 1)}
    W = all_wants(p37, k36, hh)
    val = validator(C)
    allowed = [("pilot", "pilot/"), ("checks", "checks/ident/")]
    sim = sim or kb36.BankSim("bank", kb36.SPACE_DEV)
    try:
        for s in (0, 1):
            v = kh.load_source(torch, s, a.device, src_loader)
            arms = dict(
                T1=kh.make_arm(torch, k36, f"T32_s{s}", v),
                T2=kop.T2Arm(torch, kh.make_arm(torch, k36, f"T32_s{s}", v),
                             H[s][0]),
                Z=kop.T2Arm(torch, kh.make_arm(torch, k36, f"T32_s{s}", v),
                            kop.make_head(torch)))
            io = v["io"]
            for d in p37["dev"]:
                w1, w2, wz = wants(p37, k36, s, d, hh[s])
                plan = [(w1, arms["T1"], None)]
                if ident_case(p37, s, d):
                    plan.append((wz, arms["Z"], save_t2))
                plan.append((w2, arms["T2"], save_t2))
                for w, arm, save in plan:
                    def run_save(w=w, arm=arm, save=save, io=io):
                        arm.episode = kh.episode_number(w["eid"])
                        ep = kcol.run_episode(sim, io, arm, w["task"],
                                              w["reset_id"])
                        if save is None:
                            kh.save_ep(w, ep)
                            return int(ep["physical"])
                        return save(w, ep)
                    kg.transact(w, L, run_save, val, allowed, W.__getitem__)
                if ident_case(p37, s, d):
                    pr = kh.compare_eps(kh.load_ep(w1), kh.load_ep(wz))
                    if pr:
                        raise SystemExit(f"нулевая головка ≠ T1 (s{s} "
                                         f"t{d['task']}): {pr[:4]}")
                    print(f"  s{s} t{d['task']}: нулевая головка = T1 "
                          f"побитово", flush=True)
                print(f"  s{s} t{d['task']} {d['case_id']}: пара готова; "
                      f"pilot {L.spent('pilot')}/{CAPS['pilot']}, checks "
                      f"{L.spent('checks')}/{CAPS['checks']}", flush=True)
            del arms, v, io
            kh.free(torch)
        return 0
    finally:
        sim.close()


# --- анализ ---------------------------------------------------------------------------------
def decide(pairs):
    """pairs: [(сид, задача, case, y1, y2)] -> (статус, детали)."""
    W = sum(1 for _s, _t, _c, y1, y2 in pairs if y2 and not y1)
    Ls = sum(1 for _s, _t, _c, y1, y2 in pairs if y1 and not y2)
    Dd = W - Ls
    per = {s: sum(int(y2) - int(y1) for s_, _t, _c, y1, y2 in pairs
                  if s_ == s) for s in (0, 1)}
    scenes = {}
    for s, t, c, y1, y2 in pairs:
        scenes[(t, c)] = scenes.get((t, c), 0) + int(y2) - int(y1)
    pos = sum(1 for v in scenes.values() if v > 0)
    if Dd >= 2 and all(v >= 0 for v in per.values()) and pos >= 2:
        st = "candidate"
    elif Dd <= -2 and all(v <= 0 for v in per.values()):
        st = "negative"
    else:
        st = "undetermined"
    return st, dict(W=W, L=Ls, D=Dd, per_seed=per, scenes_positive=pos,
                    delta_pp=100.0 * Dd / max(len(pairs), 1))


def analyze(a):
    install_paths()
    p37, k36 = load_prepare()
    C = kh.consts(k36)
    pairs, prob, per_task = [], [], {}
    for s in (0, 1):
        hh = pc.load_json(os.path.join(OUT, f"fit_s{s}.json"))["head_hash"]
        for d in p37["dev"]:
            w1, w2, _wz = wants(p37, k36, s, d, hh)
            g1, g2 = kh.load_ep(w1), kh.load_ep(w2)
            if g1 is None or g2 is None:
                prob.append(f"s{s} t{d['task']} {d['case_id']}: нет пары")
                continue
            prob += [f"T1 {w1['eid']}: {x}" for x in
                     kh.episode_problems(*g1, w1, C)]
            prob += [f"T2 {w2['eid']}: {x}" for x in t2_problems(*g2, w2, C)]
            zz = g2[1]
            act = int(((zz["k_reason"] == 0) & zz["k_u"].any(1)).sum())
            calls_n = int(len(zz["k_reason"]))
            y1 = bool(g1[0]["summary"]["success"])
            y2 = bool(g2[0]["summary"]["success"])
            pairs.append((s, d["task"], d["case_id"], y1, y2))
            k = f"s{s}_t{d['task']}"
            pt = per_task.setdefault(k, dict(T1=0, T2=0, W=0, L=0,
                                             calls=0, active=0))
            pt["calls"] += calls_n
            pt["active"] += act
            pt["T1"] += y1
            pt["T2"] += y2
            pt["W"] += int(y2 and not y1)
            pt["L"] += int(y1 and not y2)
    if prob or len(pairs) != 30:
        pc.save_json(os.path.join(OUT, "analysis.json"), dict(
            status="technical_stop", problems=prob[:50],
            n_pairs=len(pairs)))
        print("  K-37: ТЕХНИЧЕСКИЙ СТОП — " + "; ".join(prob[:3]))
        return 3
    st, det = decide(pairs)
    pc.save_json(os.path.join(OUT, "analysis.json"), dict(
        status=st, details=det, per_task=per_task, rule=RULE,
        pairs=[list(x) for x in pairs], prepare_sha1=p37["sha1"]))
    print(f"  K-37: W {det['W']}, L {det['L']}, D {det['D']} "
          f"({det['delta_pp']:+.1f} п.п.), по сидам {det['per_seed']}, "
          f"сцен с плюсом {det['scenes_positive']}; по задачам {per_task} — "
          f"{st}")
    return 0


# --- аудит (CPU) ------------------------------------------------------------------------------
def audit(a):
    """CPU, только K-36 dev bank-smoke T32: доли маршрутов, срабатывания
    общего ограничителя по j_D; на сохранённых первых вызовах — быстрая
    формула (j_D[0] + D_leaf) против полной пересборки листьев для u =
    ±e_ось (12 векторов) и одного большого u (насыщение cap)."""
    import torch
    import k31_protocol as p31
    import k31_readout as kr
    import k32_operator as ko
    import k32_protocol as pc32
    smoke = os.path.join(pc.OUT, "smoke")
    E = kr.build(torch)
    base = pc32.load_base()
    rep, prob = {}, []
    for s in (0, 1):
        d = os.path.join(smoke, f"T32_s{s}")
        if not os.path.isdir(d):
            prob.append(f"нет {d}")
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
        routes, capped, clipped, n_calls, n_nz = np.zeros(3, int), 0, 0, 0, 0
        worst, n_chk, skipped = 0.0, 0, {}
        dcpu = p31.cpu_decoder(torch, base["sources"][str(s)])
        for f in files:
            z = dict(np.load(os.path.join(d, f)))
            ha = z["j_has_actor"].astype(bool)
            routes += np.bincount(z["j_r"][ha], minlength=3)
            n_calls += int(len(ha))
            for D_ in z["j_D"][ha & ~z["j_zero"].astype(bool)]:
                n_nz += 1
                _du, lim = ko.limit(torch, D_)
                capped += int(lim["capped"])
                clipped += int(lim["clipped"])
            if not any(k.startswith("F_") for k in z):
                skipped[f] = "нет снимка"
                continue
            F = {k[2:]: torch.as_tensor(v) for k, v in z.items()
                 if k.startswith("F_")}
            with torch.no_grad():
                ref = dcpu(F)
            r0, path = int(z["f_r"]), [int(x) for x in z["f_path"]]
            W2 = F["W2"][0].double().numpy()
            leaf, why = kop.choose_leaf(W2, r0, path)
            if leaf is None or bool(z["f_zero"]):
                skipped[f] = "zero" if bool(z["f_zero"]) else \
                    {v: k for k, v in kop.REASON.items()}[why]
                continue
            pi = ko.path_info(F)
            w, c = kop.leaf_profile(W2, leaf)
            us = []
            for ax in range(6):
                for sg in (1.0, -1.0):
                    u = np.zeros(6, np.float32)
                    u[ax] = sg
                    us.append(u)
            us.append(np.full(6, 5.0, np.float32))
            for u in us:
                Dl = kop.leaf_delta(w, c, u)
                Dref = kop.full_reference(torch, E, dcpu, F, ref, r0, path,
                                          z["f_eta"], pi["c"][r0], leaf, u)
                worst = max(worst, float(np.abs(z["j_D"][0] + Dl -
                                                Dref).max()))
                n_chk += 1
        rep[str(s)] = dict(files=len(files), calls=n_calls,
                           routes=routes.tolist(), nonzero=n_nz,
                           capped=capped, clipped=clipped, checks=n_chk,
                           worst_dD=worst, skipped=skipped)
        print(f"  аудит s{s}: эпизодов {len(files)}, вызовов {n_calls}; "
              f"root/mid/leaf {routes.tolist()}; ненулевых {n_nz}, cap "
              f"{capped}, clip {clipped}; сверок снимок×u {n_chk}, max "
              f"|ΔD| {worst:.2e}; пропуски {skipped}", flush=True)
        if n_chk and worst > kfit.CHECK_TOL["D"]:
            prob.append(f"s{s}: быстрая ≠ полной ({worst:.2e})")
    pc.save_json(os.path.join(OUT, "audit.json"), dict(seeds=rep,
                                                         problems=prob))
    if prob:
        print("  аудит: " + "; ".join(prob))
        return 3
    return 0


def status(a):
    install_paths()
    L = ledger()
    print(f"  K-37: расход {dict(L.d['spent'])} из {CAPS}; аварий "
          f"{len(L.d['crashes'])}")
    for f in ("cache.json", "fit_s0.json", "fit_s1.json", "analysis.json"):
        print(f"    {f}: {'есть' if os.path.exists(os.path.join(OUT, f)) else 'нет'}")
    return 0


def e2e_selftest():
    """Пилот на игрушке: ToySim, FakeIO, игрушечный K-25 и actor; T1, T2
    (ненулевая головка) и проверка нулевой головки; повтор не исполняет
    заново; подмена команды T2 — отказ проверки."""
    import tempfile
    import torch
    import k21_model as km
    import k25_model as k25m
    import k29_model as k29m
    import k31_readout as kr
    import k32_model as km32
    import k32_ppo as kppo
    g_ = globals()
    k29m.no_tf32(torch)
    M = k29m.build(torch)
    Lb = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    Ub = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    mod = k25m.build(torch)["K25"](km.fake_bar(torch), Lb, Ub, seed=0).eval()
    with torch.no_grad():
        g = torch.Generator().manual_seed(3)
        for p_ in list(mod.cond.parameters()) + list(
                mod.readout.parameters()):
            p_.add_(torch.randn(p_.shape, generator=g) * 0.05)
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
    head = kop.make_head(torch)
    with torch.no_grad():
        head.bias.copy_(torch.tensor([1.5, -1.0, 0.8, 2.0, -1.7, 0.6]))
    hh = kfit.head_hash(torch, head)
    k36 = dict(sha1="PREP36", k32=dict(k32_contract="C", seeds={
        str(s): dict(source=dict(weights_hash="W"), T=dict(actor_hash=ah))
        for s in (0, 1)}), k32_plan=dict(
        bounds=dict(L=list(Lb) + [-1.0], U=list(Ub) + [1.0]),
        scale_q=np.asarray(io.max_act_q).tolist()))
    dev = [dict(task=t, case_id=f"k33s0_t{t}_c0000{i}", pos=i, hash=None)
           for t in TASKS for i in (1, 2)][:4]
    p37 = dict(sha1="PREP37", dev=dev, ident=[list(x) for x in IDENT])
    src = dict(mod=mod, dec=dec, M=M, io=io, actors=dict(T=actor))
    keep = {k: g_[k] for k in ("OUT", "DATA", "LEDGER", "load_prepare")}
    keep_kh = dict(OUT=kh.OUT, REJECTED=kh.REJECTED,
                   SS=dict(kh.STAGE_SCOPE))
    with tempfile.TemporaryDirectory() as td:
        try:
            g_.update(OUT=os.path.join(td, "r"), DATA=os.path.join(td, "d"),
                      LEDGER=os.path.join(td, "d", "ledger.json"),
                      load_prepare=lambda: (p37, k36))
            a_ = argparse.Namespace(device="cpu")
            heads = {s: (head, hh) for s in (0, 1)}
            assert pilot(a_, sim=kh._NoClose(kppo.ToySim()),
                         src_loader=lambda s: src, heads=heads) == 0
            L = ledger()
            n_pairs = 2 * len(dev)
            n_ident = sum(1 for s in (0, 1) for d in dev
                          if ident_case(p37, s, d))
            assert n_ident == 1                         # (s0, t0) в наборе
            assert len(L.d["done"]) == 2 * n_pairs + n_ident, L.d["done"]
            sp = dict(L.d["spent"])
            assert pilot(a_, sim=kh._NoClose(kppo.ToySim()),
                         src_loader=lambda s: src, heads=heads) == 0
            assert dict(L.d["spent"]) == sp == dict(ledger().d["spent"])
            C = kh.consts(k36)
            n_act = 0
            for s in (0, 1):
                for d in dev:
                    w1, w2, _wz = wants(p37, k36, s, d, hh)
                    rec, z = kh.load_ep(w2)
                    assert t2_problems(rec, z, w2, C) == []
                    n_act += int(((z["k_reason"] == 0) &
                                  z["k_u"].any(1)).sum())
                    z2 = dict(z)
                    z2["k_a2"] = z["k_a2"].copy()
                    z2["k_a2"][0, 0, 0] += 1e-4
                    assert t2_problems(rec, z2, w2, C)
                    z3 = dict(z)
                    z3["k_Dl"] = z["k_Dl"] * 1.01
                    if (z["k_reason"] == 0).any() and z["k_u"].any():
                        assert t2_problems(rec, z3, w2, C)
            assert n_act > 0, "на игрушке нет активных правок листа"
        finally:
            g_.update(keep)
            kh.OUT, kh.REJECTED = keep_kh["OUT"], keep_kh["REJECTED"]
            kh.STAGE_SCOPE.clear()
            kh.STAGE_SCOPE.update(keep_kh["SS"])
    _ = kr
    return n_act


def selftest():
    # правило решения в целых числах
    def P(lst):
        return [(s, t, f"c{i}", bool(a_), bool(b_))
                for i, (s, t, a_, b_) in enumerate(lst)]
    base = [(s, t, 1, 1) for s in (0, 1) for t in (0, 6, 8) for _ in range(5)]
    st, det = decide(P(base))
    assert st == "undetermined" and det["D"] == 0
    win = list(base)
    win[0] = (0, 0, 0, 1)
    win[16] = (1, 6, 0, 1)
    st, det = decide(P(win))
    assert det["D"] == 2 and st == "candidate", det
    one = list(base)
    one[0] = (0, 0, 0, 1)
    one[1] = (0, 0, 0, 1)
    one[15] = (1, 0, 1, 0)
    assert decide(P(one))[0] == "undetermined"          # сид 1 < 0
    neg = list(base)
    neg[0] = (0, 0, 1, 0)
    neg[20] = (1, 6, 1, 0)
    assert decide(P(neg))[0] == "negative"
    assert sum(CAPS.values()) == 62 * pc.EP
    e2e_selftest()
    print("самопроверка k37_parent_leaf пройдена: решение пилота в целых "
          "(D ≥ 2, сиды ≥ 0, ≥ 2 сцены — кандидат; D ≤ −2 и сиды ≤ 0 — "
          "отрицательно; прочее — неопределённо); бюджет 62 эпизода; "
          "игрушечный пилот: T1/T2/нулевая головка через транзакции, "
          "нулевая = T1 побитово, повтор не исполняет заново, подмена "
          "команды T2 или D_leaf — отказ проверки")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-37: родитель плюс лист")
    ap.add_argument("cmd", choices=("prepare", "audit", "cache", "fit",
                                    "pilot", "analyze", "status"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    fn = dict(prepare=prepare, audit=audit, cache=cache, fit=fit,
              pilot=pilot, analyze=analyze, status=status)[a.cmd]
    try:
        return fn(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
