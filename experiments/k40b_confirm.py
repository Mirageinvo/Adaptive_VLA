#!/usr/bin/env python3
"""K-40 B: подтверждение на резерве — T32, TreeB-Mix, FlatB-Mix.

Сцены: Confirmation reserve базового плана K-40 B (384 физические сцены
банка space 5, по 128 на задачу; не раскатывались). Оба сида — 768 пар на
контраст, 2304 эпизода.
Веса: ровно те, что прошли Screen, — план головок K-40 B закрепляет
четыре чекпойнта, PCA и код; здесь закрепляются SHA плана головок, его
чекпойнтов и PCA (и сверяются при каждом запуске).
Части: 12, по рангу сцены внутри задачи (rank − 32) mod 12 — задачи
распределены равномерно; все три политики и оба сида сцены — в одной
части. Порядок политик вращается по (номер сцены в части + задача +
сид) mod 3.
Исполнение: T32 — k40b_eval_base.run_t32; TreeB/FlatB —
k40b_eval_heads.run_head; запись и проверка шардов — их save_ep /
ep_problems; журнал части — k40b_eval_base.ledger_problems.
Допуск карты: нулевая голова = базовый T32 Screen побитово (оба сида).
Анализ — в заранее заданном порядке: (1) TreeB − T32, (2) TreeB − FlatB
(проверяется, только если подтверждён (1)); подтверждение — нижняя граница
парного 95% ДИ > 0 (bootstrap 50 000 по физическим сценам внутри задач,
оба сида вместе, задачи с равным весом). Плюс FlatB − T32 описательно.
Команды: prepare | preflight --device D | collect --part K --device D |
status | analyze | --selftest.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PROTOCOL = "k40b-confirm-v1"
POLICIES = ("T32", "TreeB", "FlatB")
N_PARTS = 12
ROOT = Path("reports/k40b/eval/confirm")
SHARDS = Path("data/k40b/eval/confirm/shards")
CODE = ("k40b_confirm.py", "k40b_eval_base.py", "k40b_eval_heads.py",
        "k40b_train.py", "k40b_collect.py", "k40a_transfer.py",
        "k31_collect.py", "k32_harness.py", "k32_policy.py",
        "k36_harness.py")
BOOT = dict(n=50_000, seed=40003)
ORDER = (("tree_t32", "TreeB", "T32"), ("tree_flat", "TreeB", "FlatB"))
EP = 610
OK, TECH, BUSY = 0, 3, 4


def kb():
    import k40b_eval_base as m
    return m


def keh():
    import k40b_eval_heads as m
    return m


def sha256(p):
    import k37_confirm as kcf
    return kcf.sha256(p)


def eid_of(policy, seed, sc):
    return f"k40b-confirm|{policy}|s{seed}|t{sc['task']}|{sc['case_id']}"


def reserve_scenes():
    """Резерв из банка (с хешами), сверенный с базовым планом."""
    b = kb()
    bp = b.load_plan()
    _m, screen, reserve = b.scenes()
    want = {(r["task"], r["case_id"], r["pos"], r["rank"])
            for r in bp["reserve"]}
    got = {(r["task"], r["case_id"], r["pos"], r["rank"]) for r in reserve}
    if want != got or len(reserve) != 384:
        raise SystemExit("резерв банка ≠ базовому плану")
    if {(s["task"], s["case_id"]) for s in screen} & \
            {(r["task"], r["case_id"]) for r in reserve}:
        raise SystemExit("резерв пересекается со Screen")
    out = []
    for r in sorted(reserve, key=lambda r: (r["task"], r["rank"])):
        out.append(dict(r, part=(r["rank"] - b.N_SCREEN) % N_PARTS))
    return bp, out


def queue(plan, part):
    out = []
    for s in (0, 1):
        for sc in sorted((x for x in plan["scenes"] if x["part"] == part),
                         key=lambda x: (x["rank"], x["task"])):
            k = ((sc["rank"] - 32) // N_PARTS + sc["task"] + s) % 3
            for pol in POLICIES[k:] + POLICIES[:k]:
                out.append(dict(eid=eid_of(pol, s, sc), policy=pol, seed=s,
                                scene=sc))
    return out


def plan_path():
    return ROOT / "plan.json"


def shard_path(eid):
    return SHARDS / (eid.replace("|", "__") + ".npz")


def prepare(a):
    import k36_protocol as pc
    import k37_confirm as kcf
    h = keh()
    hp = h.load_plan()                           # сверка весов, PCA, кода
    bp, scenes = reserve_scenes()
    plan = dict(kind="k40b_confirm_plan", protocol=PROTOCOL,
                base_plan=bp["sha256"], heads_plan=hp["sha256"],
                checkpoints=hp["checkpoints"], pca=hp["pca"],
                scenes=scenes,
                code={f: sha256(os.path.join(HERE, f)) for f in CODE},
                order=[list(x) for x in ORDER], boot=BOOT,
                rule="нижняя граница 95% ДИ > 0; (2) только при (1)",
                parts={str(k): dict(n=0, cap=0) for k in range(N_PARTS)})
    for k in range(N_PARTS):
        n = len(queue(plan, k))
        plan["parts"][str(k)] = dict(n=n, cap=(n + 1) * EP)
    plan["episodes"] = [j["eid"] for k in range(N_PARTS)
                        for j in queue(plan, k)]
    sh = kcf.canon_sha(plan)
    p = plan_path()
    if p.exists():
        old = pc.load_json(str(p))
        if kcf.canon_sha({k: v for k, v in old.items() if k != "sha256"}) \
                != sh:
            raise SystemExit("план подтверждения закреплён иначе")
        print(f"  план закреплён: {sh[:16]}")
        return OK
    plan["sha256"] = sh
    p.parent.mkdir(parents=True, exist_ok=True)
    pc.save_json(str(p), plan)
    from collections import Counter
    per = Counter((s["part"], s["task"]) for s in scenes)
    print(f"  план подтверждения {sh[:16]}: сцен {len(scenes)}, эпизодов "
          f"{len(plan['episodes'])}; частей {N_PARTS}, сцен на часть и "
          f"задачу: {sorted(set(per.values()))}")
    return OK


def load_plan(verify=True):
    import k36_protocol as pc
    import k37_confirm as kcf
    p = pc.load_json(str(plan_path()))
    if kcf.canon_sha({k: v for k, v in p.items() if k != "sha256"}) != \
            p["sha256"]:
        raise SystemExit("plan.json ≠ своему SHA")
    if verify:
        bad = [f for f, s in p["code"].items()
               if sha256(os.path.join(HERE, f)) != s]
        hp = keh().load_plan()
        if hp["sha256"] != p["heads_plan"]:
            bad.append("план головок")
        bad += [k for k, v in p["checkpoints"].items()
                if sha256(v["path"]) != v["sha256"]]
        if sha256(p["pca"]["path"]) != p["pca"]["sha256"]:
            bad.append("PCA")
        if bad:
            raise SystemExit(f"≠ плану подтверждения: {bad}")
    return p


def check_shard(j, plan_sha, C, done="_no"):
    path = shard_path(j["eid"])
    if not path.exists():
        return None, ["нет шарда"]
    with np.load(path, allow_pickle=False) as z:
        zz = dict(z)
    meta = json.loads(str(zz["meta"]))
    if meta.get("eid") != j["eid"] or meta.get("plan") != plan_sha:
        return meta, ["шард чужого эпизода или плана"]
    if j["policy"] == "T32":
        p = kb().ep_problems(zz, meta, j, C)
    else:
        if meta.get("policy") != j["policy"]:
            return meta, ["политика шарда ≠ заданию"]
        p = keh().ep_problems(zz, meta, j, C)
    if done != "_no" and done != dict(scope="base",
                                      actual=meta["summary"]["physical"]):
        p.append(f"журнал {done} ≠ записи")
    return meta, p


def save(j, ep, plan_sha):
    if j["policy"] == "T32":
        return kb().save_ep(shard_path(j["eid"]), j, ep, plan_sha)
    return keh().save_ep(shard_path(j["eid"]), j, ep, plan_sha)


# --- допуск и сбор ----------------------------------------------------------------------------------
def admission_path(card):
    return ROOT / "preflight" / f"{socket.gethostname()}__{card.g['uuid']}.json"


def preflight(a):
    """Нулевая голова = базовый T32 Screen побитово на обоих сидах."""
    import k36_harness as kh
    import k36_protocol as pc
    import k38_grip_run as kgr
    b, h = kb(), keh()
    plan = load_plan()
    bp = b.load_plan()
    k36 = pc.load_prepare()
    C = kh.consts(k36)
    sc = sorted((x for x in bp["screen"] if x["part"] == 0),
                key=lambda x: (x["rank"], x["task"]))[0]
    card = b.Card(a.device)
    prob, info = [], {}
    try:
        card.start()
        sim = card.sim(b.SPACE, b.BANK_DIR)
        P = dict(np.load(plan["pca"]["path"]))
        for s in (0, 1):
            v, arm = card.source(k36, s)
            ref = b.shard_path(b.eid_of("T32", s, sc))
            with np.load(ref) as z:
                A_ref = z["actions"]
                m_ref = json.loads(str(z["meta"]))
            ep0 = h.run_head(sim, v["io"], arm, v["mod"].bar,
                             h.zero_head(card.torch, P, C), C, sc["task"],
                             sc["pos"], b.eid_of("T32", s, sc))
            if not np.array_equal(np.asarray(ep0["actions"], np.float32),
                                  A_ref) or \
                    {k: ep0[k] for k in b.SUMMARY} != m_ref["summary"] or \
                    ep0["heavy"] != m_ref["heavy"]:
                prob.append(f"s{s}: нулевая голова ≠ базовому T32")
            info[f"zero_s{s}_seconds"] = float(ep0["seconds"])
            arm = v = None
            kh.free(card.torch)
        rep = dict(admitted=not prob, problems=prob, info=info,
                   plan=plan["sha256"], env=kgr.env_record(card.torch),
                   uuid=card.g["uuid"], host=socket.gethostname())
        admission_path(card).parent.mkdir(parents=True, exist_ok=True)
        pc.save_json(str(admission_path(card)), rep)
    finally:
        card.close()
    print(f"  preflight K-40 B confirm {a.device}: " +
          ("ДОПУЩЕНА" if not prob else "НЕ допущена: " + "; ".join(prob)) +
          f"; {info}")
    return OK if not prob else TECH


def part_dir(k):
    return ROOT / f"part-{int(k)}"


def collect(a):
    import k31_collect as kcol
    import k36_harness as kh
    import k36_protocol as pc
    import k38_grip_run as kgr
    b, h = kb(), keh()
    plan = load_plan()
    hp = h.load_plan()
    k36 = pc.load_prepare()
    C = kh.consts(k36)
    k = int(a.part)
    if str(k) not in plan["parts"]:
        raise SystemExit(f"часть {k}")
    card = b.Card(a.device)
    pd = part_dir(k)
    pd.mkdir(parents=True, exist_ok=True)
    plock = card.kcf.Lock(str(pd / ".lock"), f"part {k}", card.kcf.Busy)
    try:
        adm = pc.load_json(str(admission_path(card)))
        if not adm.get("admitted") or adm.get("plan") != plan["sha256"] or \
                adm.get("env") != kgr.env_record(card.torch):
            raise SystemExit("карта не допущена для этого плана/среды")
        L = kcol.Ledger(str(pd / "ledger.json"),
                        dict(base=plan["parts"][str(k)]["cap"]))
        jobs = queue(plan, k)
        lp_ = b.ledger_problems(L.d, plan, k, jobs, final=False)
        if lp_:
            raise SystemExit(f"журнал part {k}: {lp_[:3]}")
        by = {j["eid"]: j for j in jobs}
        pdg = L.d["pending"]
        if pdg:
            m, pr = check_shard(by[pdg["id"]], plan["sha256"], C)
            if m is not None and not pr:
                L.settle(pdg["id"], m["summary"]["physical"])
            elif m is not None:
                raise SystemExit(f"резерв {pdg['id']}: {pr[:3]}")
        card.start()
        sim = card.sim(b.SPACE, b.BANK_DIR)
        for s in (0, 1):
            js = [j for j in jobs if j["seed"] == s]
            v = arm = None
            heads = {}
            for j in js:
                if shard_path(j["eid"]).exists():
                    _m, pr = check_shard(j, plan["sha256"], C,
                                         L.d["done"].get(j["eid"]))
                    if pr:
                        raise SystemExit(f"{j['eid']}: {pr[:3]}")
                    continue
                if L.is_done(j["eid"]):
                    raise SystemExit(f"{j['eid']}: учтён, шарда нет")
                if v is None:
                    v, arm = card.source(k36, s)
                    heads = {pol: h.load_head(card.torch, hp, pol, s, C)
                             for pol in ("TreeB", "FlatB")}
                card.kcf.check_disk(str(ROOT))
                L.reserve("base", j["eid"], amount=EP)
                sc = j["scene"]
                if j["policy"] == "T32":
                    ep = b.run_t32(sim, v["io"], arm, sc["task"], sc["pos"],
                                   j["eid"])
                else:
                    ep = h.run_head(sim, v["io"], arm, v["mod"].bar,
                                    heads[j["policy"]], C, sc["task"],
                                    sc["pos"], j["eid"])
                meta = save(j, ep, plan["sha256"])
                _m, pr = check_shard(j, plan["sha256"], C)
                if pr:
                    print(f"  {j['eid']}: ПРОБЛЕМЫ {pr[:3]} — резерв открыт",
                          flush=True)
                    return TECH
                L.commit(j["eid"], meta["summary"]["physical"])
                n = sum(L.is_done(x) for x in by)
                print(f"  {j['eid']}: успех {meta['summary']['success']}; "
                      f"{n}/{len(jobs)}", flush=True)
            if v is not None:
                arm = v = None
                kh.free(card.torch)
    finally:
        plock.release()
        card.close()
    print(f"  part {k}: готова", flush=True)
    return OK


def status(a):
    plan = load_plan(verify=False)
    for k, v in plan["parts"].items():
        lp = part_dir(k) / "ledger.json"
        if not lp.exists():
            print(f"  part {k}: не начата (0/{v['n']})")
            continue
        d = json.load(open(lp))
        print(f"  part {k}: {len(d['done'])}/{v['n']}; аварий "
              f"{len(d['crashes'])}")
    return OK


# --- анализ --------------------------------------------------------------------------------------------
def decide(res):
    """Порядок: (1) TreeB − T32; (2) TreeB − FlatB только при (1)."""
    c1 = res["tree_t32"]["lo"] > 0
    c2 = c1 and res["tree_flat"]["lo"] > 0
    return dict(tree_t32_confirmed=bool(c1),
                tree_flat_confirmed=bool(c2),
                tree_flat_tested=bool(c1))


def analyze(a):
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    b, h = kb(), keh()
    plan = load_plan()
    C = kh.consts(pc.load_prepare())
    Y, prob, locks = {}, [], []
    try:
        for k in range(N_PARTS):
            locks.append(kcf.Lock(str(part_dir(k) / ".lock"), f"part {k}",
                                  kcf.Busy))
            lp = part_dir(k) / "ledger.json"
            if not lp.exists():
                prob.append(f"part {k}: нет журнала")
                continue
            d = json.load(open(lp))
            jobs = queue(plan, k)
            prob += [f"part {k}: {x}" for x in
                     b.ledger_problems(d, plan, k, jobs, final=True)]
            tot = 0
            for j in jobs:
                m, pr = check_shard(j, plan["sha256"], C,
                                    d["done"].get(j["eid"]))
                prob += [f"{j['eid']}: {x}" for x in pr]
                if m is None or pr:
                    continue
                tot += m["summary"]["physical"]
                sc = j["scene"]
                Y[(j["policy"], j["seed"], sc["task"], sc["case_id"])] = \
                    int(m["summary"]["success"])
            if d["spent"] != dict(base=tot + EP * len(d["crashes"])):
                prob.append(f"part {k}: расход ≠ сумме")
    finally:
        for lk in locks:
            lk.release()
    if len(Y) != len(plan["episodes"]):
        prob.append(f"эпизодов {len(Y)} ≠ {len(plan['episodes'])}")
    if prob:
        raise SystemExit("анализ: " + "; ".join(prob[:4]))
    tasks = sorted({s["task"] for s in plan["scenes"]})
    scenes = sorted({(s["task"], s["case_id"]) for s in plan["scenes"]})
    con = lambda x, y: h.contrast(Y, x, y, scenes, tasks,  # noqa: E731
                                  n_boot=BOOT["n"], seed=BOOT["seed"])
    res = dict(sr={p: {str(t): float(np.mean([Y[(p, s, t, c)]
                                             for (tt, c) in scenes if tt == t
                                             for s in (0, 1)]))
                       for t in tasks} for p in POLICIES},
               tree_t32=con("TreeB", "T32"), tree_flat=con("TreeB", "FlatB"),
               flat_t32=con("FlatB", "T32"))
    res["decision"] = decide(res)
    pc.save_json(str(ROOT / "analysis.json"), dict(res, plan=plan["sha256"]))
    for p in POLICIES:
        print(f"  SR {p}: {res['sr'][p]}")
    for n_ in ("tree_t32", "tree_flat", "flat_t32"):
        c = res[n_]
        print(f"  {n_}: W {c['W']} L {c['L']} D {c['D']:+d} (N {c['N']}); "
              f"Δ {c['delta_pp']:+.1f} п.п. 95% [{c['lo']:+.1f}; "
              f"{c['hi']:+.1f}]; задачи {c['by_task']}; сиды {c['by_seed']};"
              f" расхождений {100 * c['disc']:.0f}%")
    d = res["decision"]
    print(f"  (1) TreeB − T32: {'ПОДТВЕРЖДЕНО' if d['tree_t32_confirmed'] else 'не подтверждено'}")
    print("  (2) TreeB − FlatB: " + (
        ("ПОДТВЕРЖДЕНО" if d["tree_flat_confirmed"] else "не подтверждено")
        if d["tree_flat_tested"] else "не проверяется (1 не подтверждено)"))
    return OK


def selftest():
    from collections import Counter
    # части: задачи равномерно, все варианты сцены в части
    scenes = [dict(task=t, case_id=f"c{t}_{r}", pos=r, rank=r, hash=["a", "b"])
              for t in (0, 6, 8) for r in range(32, 160)]
    for s in scenes:
        s["part"] = (s["rank"] - 32) % N_PARTS
    plan = dict(scenes=scenes)
    qs = [queue(plan, k) for k in range(N_PARTS)]
    assert sum(len(q) for q in qs) == 2304
    assert len({j["eid"] for q in qs for j in q}) == 2304
    for q in qs:
        c = Counter(j["scene"]["task"] for j in q)
        assert max(c.values()) - min(c.values()) == 0
        cs = Counter(j["scene"]["case_id"] for j in q)
        assert set(cs.values()) == {6}
    for q in qs:
        first = Counter(q[i]["policy"] for i in range(0, len(q), 3))
        assert max(first.values()) - min(first.values()) <= 2, first
    # порядок проверки
    mk = lambda l1, l2: dict(tree_t32=dict(lo=l1), tree_flat=dict(lo=l2))  # noqa
    assert decide(mk(0.5, 0.2)) == dict(tree_t32_confirmed=True,
                                        tree_flat_confirmed=True,
                                        tree_flat_tested=True)
    assert decide(mk(-0.1, 3.0))["tree_flat_confirmed"] is False
    assert decide(mk(-0.1, 3.0))["tree_flat_tested"] is False
    assert decide(mk(0.1, -0.2))["tree_flat_confirmed"] is False
    print("самопроверка k40b_confirm пройдена: 12 частей, задачи поровну, "
          "все 6 вариантов сцены в части, 2304 эпизода; порядок проверки "
          "(2) только при (1)")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("prepare", "preflight", "collect",
                                    "status", "analyze"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--part", type=int, default=0)
    a = ap.parse_args()
    import k37_confirm as kcf
    fn = dict(prepare=prepare, preflight=preflight, collect=collect,
              status=status, analyze=analyze)[a.cmd]
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
