#!/usr/bin/env python3
"""K-40 B, оценка — фаза I: банк space 5, план, базовый T32 на Screen.

Банк: LIBERO-10, задачи 0/6/8, по 160 допустимых сцен (space 5), генерация
и отбор — k33_bank (через k37_bank.run_chunk); исключаются сигнатуры и
seed официальных стартов и всех банков из фактических манифестов
(K-33 spaces 0/1, K-37 space 2, K-39 space 3, K-39 pulse space 4) и
повторы внутри банка; холодный повтор обоих начальных хешей; SHA-манифест.
Разбиение до раскаток: в задаче — сортировка по
sha256("k40b-eval-screen-v1|task|case_id"); первые 32 — Screen, остальные
128 — Confirmation reserve. Часть Screen = ранг в задаче mod 4 (оба сида
сцены в одной части).
Базовая политика: штатный T32 (k31_collect.run_episode: один тяжёлый
вызов на чанк, до 8 действий B), источники T32_s0/T32_s1.
Эпизод: шард npz — журнал K-32 (pack_journal), действия, сводка, число
тяжёлых вызовов, время; проверка — журнал против номинала, исполненное =
масштаб B, начальные хеши = банку, тяжёлых = чанков, сводка согласована.
Команды: bank [--workers N] | prepare | preflight --device D |
collect --part K --device D | status | finalize | --selftest.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PROTOCOL = "k40b-eval-base-v1"
TASKS = (0, 6, 8)
SPACE = 5
N_SCENES, N_SCREEN = 160, 32
MAX_CAND, CHUNK = 480, 48
N_PARTS = 4
SEEDS = (0, 1)
EP = 610
BANK_DIR = "data/k40b/eval/bank"
WORK = "data/k40b/eval/bank_work"
MANIFEST = "reports/k40b/eval/bank_manifest.json"
BANK_LEDGER = "data/k40b/eval/bank_ledger.json"
ROOT = Path("reports/k40b/eval/base")
SHARDS = Path("data/k40b/eval/base/shards")
PREV = ("reports/k33/bank_manifest.json", "reports/k37c/bank_manifest.json",
        "reports/k39/bank_manifest.json",
        "reports/k39/pulse/b1/bank_manifest.json")
CODE = ("k40b_eval_base.py", "k31_collect.py", "k32_harness.py",
        "k32_policy.py", "k36_harness.py", "k36_bank.py", "k33_bank.py",
        "k37_bank.py", "k37_confirm.py", "k38_grip_run.py", "k29_policy.py")
IDENT = ((0, 0), (1, 8))                       # допуск: (сид, задача) c2
OK, TECH, BUSY = 0, 3, 4


def sha256(p):
    import k37_confirm as kcf
    return kcf.sha256(p)


def h256(s):
    return hashlib.sha256(s.encode()).hexdigest()


# --- банк -------------------------------------------------------------------------------------
def file_hash_ok(p, want):
    """Хеш файла тем алгоритмом, которым его записал манифест: 40 hex —
    SHA1 (K-33), 64 hex — SHA256 (K-37, K-39)."""
    alg = {40: hashlib.sha1, 64: hashlib.sha256}.get(len(str(want)))
    if alg is None:
        raise SystemExit(f"неизвестный формат хеша {want!r}")
    h = alg()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest() == want


def prev_seen():
    """Подписи и seed всех прежних банков по их манифестам (с проверкой
    SHA файлов)."""
    sigs, seeds, spaces = set(), set(), set()
    for mp in PREV:
        if not os.path.exists(mp):
            raise SystemExit(f"нет манифеста прежнего банка {mp}")
        m = json.load(open(mp, encoding="utf-8"))
        if "space" in m:
            spaces.add(int(m["space"]))
        for p, s in m["files"].items():
            if not os.path.exists(p) or not file_hash_ok(p, s):
                raise SystemExit(f"{mp}: файл {p} отсутствует или изменён")
            z = np.load(p, allow_pickle=False)
            sigs.update(str(x) for x in z["sig"])
            seeds.update(int(x) for x in z["seed"])
    if SPACE in spaces:
        raise SystemExit(f"space {SPACE} уже занят прежним банком")
    return frozenset(sigs), frozenset(seeds)


def bank(a):
    import k31_collect as kcol
    import k33_bank as kb
    import k33_protocol as p33
    import k37_bank as k37b
    if os.path.exists(MANIFEST):
        m = json.load(open(MANIFEST, encoding="utf-8"))
        bad = [p for p, s in m["files"].items()
               if not os.path.exists(p) or sha256(p) != s]
        if bad:
            raise SystemExit(f"банк закреплён, файлы изменены: {bad}")
        print(f"  {MANIFEST}: закреплён (bank_id {m['bank_id'][:12]})")
        return OK
    seen_prev, seeds0 = prev_seen()
    seen_build = set(seen_prev)          # растёт по мере отбора задач
    L = kcol.Ledger(BANK_LEDGER, dict(bank=10 * (len(TASKS) * MAX_CAND +
                                                 len(TASKS) * N_SCENES)))
    per, files = {}, {}
    for t in TASKS:
        official = set(kb.official_joint_sigs(t))
        results, start = [], 0
        while True:
            cases, rej = kb.select(results, N_SCENES, set(seen_build),
                                   official)
            if len(cases) >= N_SCENES:
                break
            if start >= MAX_CAND:
                raise SystemExit(f"t{t}: допустимых {len(cases)} < "
                                 f"{N_SCENES} за {start} кандидатов")
            n = min(CHUNK, MAX_CAND - start)
            jobs = [(SPACE, t, i) for i in range(start, start + n)]
            results += k37b.run_chunk(L, f"bank/gen/t{t}/c{start}", jobs,
                                      kb._gen_job, a.workers,
                                      os.path.join(WORK,
                                                   f"gen_t{t}_c{start}.pkl"))
            start += n
        cases, rej = kb.select(results, N_SCENES, seen_build, official)
        if len({json.dumps(c["meta"], sort_keys=True) for c in cases}) != 1:
            raise SystemExit(f"t{t}: разные модели сцен")
        p = kb.task_path(SPACE, t, BANK_DIR)
        if os.path.exists(p):            # продолжение после частичной записи
            old, _m = kb.load_task(SPACE, t, BANK_DIR)
            if [(c["case_id"], c["init_hash"], c["init_hash_full"])
                    for c in old] != [(c["case_id"], c["init_hash"],
                                       c["init_hash_full"]) for c in cases]:
                raise SystemExit(f"{p}: записанная задача ≠ пересчёту из "
                                 f"кеша генерации — нужен разбор")
        else:
            p = kb.save_task(SPACE, t, cases, cases[0]["meta"], BANK_DIR)
        files[p] = sha256(p)
        per[str(t)] = dict(cases=len(cases), candidates=start,
                           reject_counts={w: sum(1 for r in rej
                                                 if r["why"] == w)
                                          for w in {r["why"] for r in rej}})
        print(f"  t{t}: принято {len(cases)} из {start}; отклонено "
              f"{per[str(t)]['reject_counts']}", flush=True)
    seeds = [int(x) for p_ in files for x in np.load(p_)["seed"]]
    sigs = [str(x) for p_ in files for x in np.load(p_)["sig"]]
    if len(set(seeds)) != len(seeds) or set(seeds) & seeds0:
        raise SystemExit("коллизия seed (внутри банка или с прежними)")
    if len(set(sigs)) != len(sigs) or set(sigs) & seen_prev:
        raise SystemExit("коллизия подписей (внутри банка или с прежними)")
    jobs = []
    for t in TASKS:
        cs, meta = kb.load_task(SPACE, t, BANK_DIR)
        jobs += [(SPACE, t, c, meta) for c in cs]
    res = []
    for k in range(0, len(jobs), CHUNK):
        res += k37b.run_chunk(L, f"bank/replay/p{k}", jobs[k:k + CHUNK],
                              kb._replay_job, a.workers,
                              os.path.join(WORK, f"replay_p{k}.pkl"))
    by = {r["case_id"]: r for r in res}
    bad = []
    for _sp, _t, c, _m in jobs:
        r = by[c["case_id"]]
        if not r.get("ok"):
            bad.append((c["case_id"], r.get("why")))
        elif (r["init_hash"], r["init_hash_full"]) != \
                (c["init_hash"], c["init_hash_full"]):
            bad.append((c["case_id"], "post-wait хеши ≠ банку"))
        elif r["done"] or r["success"] or not r["finite"]:
            bad.append((c["case_id"], "успех/done/нечисловое"))
    man = dict(kind="k40b_eval_bank", space=SPACE, tasks=list(TASKS),
               n_per_task=N_SCENES, recipe=kb.RECIPE, files=files, per=per,
               prev=list(PREV), replay=dict(n=len(jobs), bad=len(bad),
                                            examples=bad[:20]),
               versions=p33.versions(), budget=L.spent("bank"),
               created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    man["bank_id"] = h256(json.dumps(dict(recipe=kb.RECIPE, files=files),
                                     sort_keys=True, default=str))
    if bad:
        p33.save_json(MANIFEST + ".failed.json", man)
        print(f"  холодный повтор: {len(bad)} расхождений {bad[:3]}")
        return TECH
    p33.save_json(MANIFEST, man)
    print(f"  банк space {SPACE}: {len(jobs)} сцен, холодный повтор совпал; "
          f"bank_id {man['bank_id'][:12]}")
    return OK


def scenes():
    """-> (манифест, Screen [dict], Reserve [dict])."""
    import k33_bank as kb
    m = json.load(open(MANIFEST, encoding="utf-8"))
    bad = [p for p, s in m["files"].items()
           if not os.path.exists(p) or sha256(p) != s]
    if bad or m.get("replay", {}).get("bad") != 0:
        raise SystemExit(f"банк: файлы изменены {bad} или повтор не пройден")
    screen, reserve = [], []
    for t in TASKS:
        cs, _meta = kb.load_task(SPACE, t, BANK_DIR)
        if len(cs) != N_SCENES:
            raise SystemExit(f"t{t}: сцен {len(cs)} ≠ {N_SCENES}")
        rows = [dict(task=t, case_id=c["case_id"], pos=pos,
                     hash=[c["init_hash"], c["init_hash_full"]])
                for pos, c in enumerate(cs)]
        rows.sort(key=lambda r: h256(f"k40b-eval-screen-v1|{t}|"
                                     f"{r['case_id']}"))
        for rk, r in enumerate(rows):
            r["rank"] = rk
            if rk < N_SCREEN:
                r["part"] = rk % N_PARTS
                screen.append(r)
            else:
                reserve.append(r)
    return m, screen, reserve


def eid_of(policy, seed, sc):
    return f"k40b-eval|{policy}|s{seed}|t{sc['task']}|{sc['case_id']}"


def queue(plan, part):
    out = []
    for s in SEEDS:
        for sc in sorted((x for x in plan["screen"] if x["part"] == part),
                         key=lambda x: (x["rank"], x["task"])):
            out.append(dict(eid=eid_of("T32", s, sc), seed=s, scene=sc))
    return out


# --- план -----------------------------------------------------------------------------------------
def plan_path():
    return ROOT / "plan.json"


def prepare(a):
    import k36_protocol as pc
    import k37_confirm as kcf
    m, screen, reserve = scenes()
    k36 = pc.load_prepare()
    plan = dict(kind="k40b_eval_base_plan", protocol=PROTOCOL,
                bank=dict(bank_id=m["bank_id"], files=m["files"],
                          space=SPACE, dir=BANK_DIR,
                          manifest_sha256=sha256(MANIFEST)),
                screen=screen, reserve=[dict(task=r["task"],
                                             case_id=r["case_id"],
                                             pos=r["pos"], rank=r["rank"])
                                        for r in reserve],
                k36_prepare_sha1=k36["sha1"],
                t32={str(s): dict(
                    actor=k36["k32"]["seeds"][str(s)]["T"]["actor_hash"],
                    source=k36["k32"]["seeds"][str(s)]["source"][
                        "weights_hash"]) for s in SEEDS},
                code={f: sha256(os.path.join(HERE, f)) for f in CODE},
                env=dict(B=1, exec_seed=101, waiting_steps=10, max_steps=600,
                         horizon=8, threads=8),
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
            raise SystemExit("план закреплён иначе — новый каталог")
        print(f"  план закреплён: {sh[:16]}")
        return OK
    plan["sha256"] = sh
    pc.save_json(str(p), plan)
    print(f"  план {p}: SHA {sh[:16]}; Screen {len(screen)} сцен, "
          f"{len(plan['episodes'])} эпизодов T32; резерв {len(reserve)}")
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
        if bad:
            raise SystemExit(f"код ≠ плану: {bad}")
        if pc.load_prepare()["sha1"] != p["k36_prepare_sha1"]:
            raise SystemExit("prepare K-36 ≠ плану")
        for f, s in p["bank"]["files"].items():
            if sha256(f) != s:
                raise SystemExit(f"банк {f} ≠ плану")
    return p


# --- эпизод ---------------------------------------------------------------------------------------
SUMMARY = ("success", "success_step", "done_step", "own_steps", "steps",
           "physical", "init_hash", "init_hash_full", "actions_sha1")


def run_t32(sim, io, arm, task, pos, eid):
    """Штатный T32 через k31_collect.run_episode; считает тяжёлые вызовы."""
    import k31_collect as kcol
    from k36_harness import episode_number
    heavy = dict(n=0)
    fwd = arm.forward

    def counted(*x, **kw):
        heavy["n"] += 1
        return fwd(*x, **kw)
    arm.forward = counted
    arm.episode = episode_number(eid)
    try:
        ep = kcol.run_episode(sim, io, arm, task, pos)
    finally:
        arm.forward = fwd
    ep["heavy"] = heavy["n"]
    return ep


def shard_path(eid):
    return SHARDS / (eid.replace("|", "__") + ".npz")


def save_ep(path, j, ep, plan_sha):
    import k32_harness as kh32
    J = kh32.pack_journal([(0, ep["calls"])], 6)
    meta = dict(eid=j["eid"], plan=plan_sha, heavy=int(ep["heavy"]),
                seconds=float(ep["seconds"]),
                summary={k: ep[k] for k in SUMMARY})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, meta=np.array(json.dumps(meta)),
                            actions=np.asarray(ep["actions"], np.float32),
                            **{f"j_{k}": v for k, v in J.items()})
    tmp.replace(path)
    return meta


def ep_problems(z, meta, j, C):
    import k29_policy as kp
    import k31_collect as kcol
    import k32_harness as kh32
    p = []
    A = z["actions"]
    s = meta["summary"]
    if kcol.actions_sha(A) != s["actions_sha1"]:
        return ["действия ≠ SHA"]
    if [s["init_hash"], s["init_hash_full"]] != list(j["scene"]["hash"]):
        p.append("начальный хеш ≠ банку")
    J = {k[2:]: v for k, v in z.items() if k.startswith("j_")}
    n = len(J.get("ep", []))
    if n == 0:
        return p + ["журнал пуст"]
    q = C["q"]
    A_nom = np.concatenate([kp.scale_exec(J["a_codec"][i][None], q)[0]
                            [:int(J["k"][i])] for i in range(n)])
    p += [f"журнал: {x}" for x in kh32.journal_problems(
        J, [A_nom], [s["done_step"]], C["half"], C["mid"], C["L"], C["U"],
        q, "T")[:3]]
    if not np.array_equal(A_nom.astype(np.float32),
                          np.asarray(A, np.float32)):
        p.append("исполненное ≠ масштабу B")
    if meta["heavy"] != n:
        p.append(f"тяжёлых вызовов {meta['heavy']} ≠ чанков {n}")
    ss, ds, st_ = s["success_step"], s["done_step"], s["steps"]
    if s["physical"] != 10 + len(A) or st_ != len(A) or s["physical"] > EP:
        p.append("расход ≠ шагам")
    if s["success"]:
        if not (0 <= ss < st_ and (ds < 0 or ss <= ds)):
            p.append("success_step несогласован")
    elif ss != -1:
        p.append("success_step при неуспехе")
    if ds >= 0:
        if ds + 1 != st_ or s["own_steps"] != st_:
            p.append("done_step/steps несогласованы")
    elif st_ != 600:
        p.append("без terminal эпизод не дошёл до лимита")
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
    p = ep_problems(zz, meta, j, C)
    if done != "_no" and done != dict(scope="base",
                                      actual=meta["summary"]["physical"]):
        p.append(f"журнал {done} ≠ записи")
    return meta, p


# --- карта, допуск, сбор ----------------------------------------------------------------------------
class Card:
    def __init__(self, device):
        import torch
        import k37_confirm as kcf
        from k38_grip_run import threads_problems
        if pr := threads_problems(torch):
            raise SystemExit("потоки: " + "; ".join(pr))
        self.torch, self.kcf = torch, kcf
        self.idx = kcf.dev_index(device)
        self.g = kcf.gpu_query(self.idx)
        self.lock = kcf.Lock(os.path.join(kcf.lock_dir(), self.g["uuid"] +
                                          ".lock"), f"карта {self.g['uuid']}",
                             kcf.Busy)
        self.device = device
        self.sims = []

    def start(self):
        self.kcf.check_gpu(self.idx, self.kcf.RES["min_free_gb"])
        self.kcf.check_disk(str(ROOT))
        self.kcf.bind_device(self.torch, self.idx, self.g)

    def sim(self, space, bank_dir):
        import k36_bank as kb36
        s = kb36.BankSim("bank", space, bank_dir=bank_dir)
        self.sims.append(s)
        return s

    def source(self, k36, s):
        import k36_harness as kh
        self.kcf.check_gpu(self.idx, self.kcf.RES["min_free_gb"])
        v = kh.load_source(self.torch, s, self.device)
        return v, kh.make_arm(self.torch, k36, f"T32_s{s}", v)

    def close(self):
        for s in self.sims:
            self.kcf.close_sim(s)
        self.lock.release()


def admission_path(card):
    return ROOT / "preflight" / f"{socket.gethostname()}__{card.g['uuid']}.json"


def preflight(a):
    """Штатный T32 воспроизводит архивный T1 c2 на двух эпизодах."""
    import k36_harness as kh
    import k36_protocol as pc
    import k37_bank as k37b
    import k37_confirm as kcf
    import k38_grip_run as kgr
    import k39_failure_probe as kfp
    plan = load_plan()
    k36 = pc.load_prepare()
    archive = Path(a.archive)
    c2, _k = kcf.load_plan(str(archive))
    merged = pc.load_json(str(archive / "merged.json"))
    rows = sorted((r for r in merged["rows"] if r["policy"] == "T1"),
                  key=lambda r: r["eid"])
    wants = {}
    for k in range(kcf.N_PARTS):
        for w in kcf.part_wants(c2, k36, k):
            wants[w["eid"]] = (k, w)
    card = Card(a.device)
    prob, got = [], []
    try:
        card.start()
        sim2 = card.sim(k37b.SPACE, k37b.BANK_DIR)
        for s, t in IDENT:
            r = next(x for x in rows if x["seed"] == s and x["task"] == t)
            part, w = wants[r["eid"]]
            rec, z = kfp.reference(archive, part, w)
            v, arm = card.source(k36, s)
            ep = run_t32(sim2, v["io"], arm, t, w["reset_id"], r["eid"])
            q = rec["summary"]
            same = np.array_equal(np.asarray(ep["actions"], np.float32),
                                  np.asarray(z["actions"], np.float32)) and \
                all(ep[k] == q[k] for k in ("success", "steps", "done_step",
                                            "init_hash", "init_hash_full"))
            if not same:
                prob.append(f"s{s} t{t}: T32 ≠ архиву c2")
            if ep["heavy"] != len(ep["calls"]):
                prob.append(f"s{s} t{t}: тяжёлых {ep['heavy']} ≠ чанков")
            got.append(dict(eid=r["eid"], same=bool(same),
                            heavy=int(ep["heavy"]),
                            seconds=float(ep["seconds"])))
            arm = v = None
            kh.free(card.torch)
        rep = dict(admitted=not prob, problems=prob, episodes=got,
                   plan=plan["sha256"], env=kgr.env_record(card.torch),
                   uuid=card.g["uuid"], host=socket.gethostname())
        pc.save_json(str(admission_path(card)), rep)
    finally:
        card.close()
    print(f"  preflight K-40 B base {a.device}: " +
          ("ДОПУЩЕНА" if not prob else "НЕ допущена: " + "; ".join(prob)) +
          f"; {[round(x['seconds']) for x in got]} с на эпизод")
    return OK if not prob else TECH


def part_dir(k):
    return ROOT / f"part-{int(k)}"


def ledger_problems(d, plan, k, jobs, final):
    """Журнал части: лимиты, резерв, аварии (id/область/сумма), done."""
    p = []
    ids = {j["eid"] for j in jobs}
    if d.get("caps") != dict(base=plan["parts"][str(k)]["cap"]):
        p.append(f"лимиты {d.get('caps')}")
    if set(d.get("spent", {})) - {"base"}:
        p.append(f"чужие области расхода {d['spent']}")
    pd_ = d.get("pending")
    if pd_ is not None and (final or pd_.get("id") not in ids or
                            pd_.get("scope") != "base" or
                            pd_.get("amount") != EP or
                            pd_.get("id") in d["done"]):
        p.append(f"открытый резерв {pd_}")
    for c in d.get("crashes", []):
        if c.get("id") not in ids or c.get("scope") != "base" or \
                c.get("amount") != EP:
            p.append(f"чужая авария {c}")
    if set(d["done"]) - ids:
        p.append("учтены эпизоды вне части")
    if any(v.get("scope") != "base" for v in d["done"].values()):
        p.append("учёт в чужой области")
    if final and set(d["done"]) != ids:
        p.append(f"учтено {len(d['done'])} из {len(ids)}")
    return p


def collect(a):
    import k31_collect as kcol
    import k36_harness as kh
    import k36_protocol as pc
    import k38_grip_run as kgr
    plan = load_plan()
    k36 = pc.load_prepare()
    C = kh.consts(k36)
    k = int(a.part)
    if str(k) not in plan["parts"]:
        raise SystemExit(f"часть {k}")
    card = Card(a.device)
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
        lp_ = ledger_problems(L.d, plan, k, jobs, final=False)
        if lp_:
            raise SystemExit(f"журнал part {k}: {lp_[:3]}")
        by = {j["eid"]: j for j in jobs}
        pdg = L.d["pending"]
        if pdg:
            if pdg["id"] not in by or pdg["scope"] != "base" or \
                    pdg["amount"] != EP:
                raise SystemExit(f"открытый резерв {pdg['id']} не из части")
            m, pr = check_shard(by[pdg["id"]], plan["sha256"], C)
            if m is not None and not pr:
                L.settle(pdg["id"], m["summary"]["physical"])
            elif m is not None:
                raise SystemExit(f"резерв {pdg['id']}: {pr[:3]}")
        card.start()
        sim = card.sim(SPACE, BANK_DIR)
        for s in SEEDS:
            js = [j for j in jobs if j["seed"] == s]
            v = arm = None
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
                card.kcf.check_disk(str(ROOT))
                L.reserve("base", j["eid"], amount=EP)
                ep = run_t32(sim, v["io"], arm, j["scene"]["task"],
                             j["scene"]["pos"], j["eid"])
                meta = save_ep(shard_path(j["eid"]), j, ep, plan["sha256"])
                _m, pr = check_shard(j, plan["sha256"], C)
                if pr:
                    print(f"  {j['eid']}: ПРОБЛЕМЫ {pr[:3]} — резерв открыт",
                          flush=True)
                    return TECH
                L.commit(j["eid"], meta["summary"]["physical"])
                n = sum(L.is_done(x) for x in by)
                print(f"  {j['eid']}: успех {meta['summary']['success']}, "
                      f"шагов {meta['summary']['steps']}; {n}/{len(jobs)}",
                      flush=True)
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
        print(f"  part {k}: {len(d['done'])}/{v['n']}; резерв "
              f"{(d['pending'] or {}).get('id', '—')}; аварий "
              f"{len(d['crashes'])}")
    return OK


def finalize(a):
    """Все 192 шарда проверены, журналы закрыты (строго) ->
    baseline_manifest.json; существующий манифест обязан совпасть."""
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    plan = load_plan()
    C = kh.consts(pc.load_prepare())
    out, prob, locks = {}, [], []
    try:
        for k in range(N_PARTS):
            pd = part_dir(k)
            pd.mkdir(parents=True, exist_ok=True)
            locks.append(kcf.Lock(str(pd / ".lock"), f"part {k}", kcf.Busy))
            lp = pd / "ledger.json"
            if not lp.exists():
                raise SystemExit(f"part {k}: нет журнала")
            d = json.load(open(lp))
            jobs = queue(plan, k)
            lpr = ledger_problems(d, plan, k, jobs, final=True)
            if lpr:
                prob += [f"part {k}: {x}" for x in lpr]
                continue
            tot = 0
            for j in jobs:
                m, pr = check_shard(j, plan["sha256"], C, d["done"][j["eid"]])
                prob += [f"{j['eid']}: {x}" for x in pr]
                if m is None or pr:
                    continue
                tot += m["summary"]["physical"]
                out[j["eid"]] = dict(sha256=sha256(shard_path(j["eid"])),
                                     success=m["summary"]["success"],
                                     task=j["scene"]["task"], seed=j["seed"],
                                     case_id=j["scene"]["case_id"],
                                     part=k, steps=m["summary"]["steps"])
            if d["spent"] != dict(base=tot + EP * len(d["crashes"])):
                prob.append(f"part {k}: расход ≠ сумме")
    finally:
        for lk in locks:
            lk.release()
    if set(out) != set(plan["episodes"]):
        prob.append(f"эпизодов {len(out)} ≠ {len(plan['episodes'])}")
    if prob:
        print("  finalize: ОТКАЗ " + "; ".join(prob[:4]))
        return TECH
    man = dict(kind="k40b_eval_baseline", plan=plan["sha256"],
               episodes=out)
    mp = ROOT / "baseline_manifest.json"
    if mp.exists():
        if pc.load_json(str(mp)) != man:
            raise SystemExit("baseline_manifest.json уже есть и ≠ пересчёту")
        print("  baseline_manifest: уже закреплён, совпадает")
    else:
        pc.save_json(str(mp), man)
    sr = {t: np.mean([v["success"] for v in out.values() if v["task"] == t])
          for t in TASKS}
    print(f"  baseline_manifest: {len(out)} эпизодов; SR T32 по задачам "
          f"{ {t: round(100 * x, 1) for t, x in sr.items()} }")
    return OK


# --- самопроверка ------------------------------------------------------------------------------------
def selftest():
    from collections import Counter
    import torch
    import k21_model as km21
    import k25_model as k25m
    import k29_model as k29m
    import k32_model as km32
    import k32_ppo as kppo
    import k31_collect as kcol
    import k36_harness as kh
    k29m.no_tf32(torch)
    torch.manual_seed(0)
    # разбиение: 32 Screen на задачу, части по 8, оба сида в части
    sc = []
    for t in TASKS:
        rows = [dict(task=t, case_id=f"c{t}_{i}", pos=i, hash=["a", "b"])
                for i in range(N_SCENES)]
        rows.sort(key=lambda r: h256(f"k40b-eval-screen-v1|{t}|"
                                     f"{r['case_id']}"))
        for rk, r in enumerate(rows[:N_SCREEN]):
            sc.append(dict(r, rank=rk, part=rk % N_PARTS))
    plan = dict(screen=sc)
    qs = [queue(plan, k) for k in range(N_PARTS)]
    assert [len(q) for q in qs] == [48] * 4
    assert len({j["eid"] for q in qs for j in q}) == 192
    for q in qs:
        assert Counter(j["scene"]["task"] for j in q) == {t: 16 for t in TASKS}
        assert set(Counter(j["scene"]["case_id"] for j in q).values()) == {2}
    # игрушечный эпизод T32: шард проходит, порча — отказ
    M = k29m.build(torch)
    Lb = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    Ub = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    mod = k25m.build(torch)["K25"](km21.fake_bar(torch), Lb, Ub,
                                   seed=0).eval()
    for p_ in mod.parameters():
        p_.requires_grad_(False)
    dec = M["Decoder"](mod).freeze()
    H = mod.H
    norm = km32.fit_normalizer(np.random.default_rng(0).normal(
        size=(64, 6 * H + 72)), H)
    actor, _c = km32.build(torch, H, 6, 0, norm)

    class Sim(kppo.ToySim):
        def _obs(self):
            o = super()._obs()
            for k in ("agentview_image", "robot0_eye_in_hand_image"):
                o[k] = np.broadcast_to(o[k][:, :1, :1], (1, 224, 224, 3)
                                       ).copy()
            return o

    class FIO(kcol.FakeIO):
        @property
        def inp(self):
            return self

        def state8(self, obs):
            return np.asarray(obs["state"][:, :8], np.float64)

        def prep(self, obs, desc, task):
            x = float(obs["state"][0, 0])
            bt = km21.fake_batch(torch, B=1, seed=int(abs(x) * 1e3) % 13)
            s8 = np.zeros((1, 8))
            s8[0, :3] = x
            return bt, 1, s8
    io = FIO()
    k36 = dict(k32=dict(k32_contract="C", seeds={"0": dict(
        source=dict(weights_hash="W"), T=dict(actor_hash="A"))}),
        k32_plan=dict(bounds=dict(L=list(Lb) + [-1.0], U=list(Ub) + [1.0]),
                      scale_q=np.asarray(io.max_act_q).tolist()))
    C = kh.consts(k36)
    src = dict(mod=mod, dec=dec, M=M, io=io, actors=dict(T=actor))
    arm = kh.make_arm(torch, k36, "T32_s0", src)
    sim = Sim()
    ep = run_t32(sim, io, arm, 0, 3, "e")
    assert ep["heavy"] == len(ep["calls"]) > 0
    j = dict(eid="k40b-eval|T32|s0|t0|toy", seed=0,
             scene=dict(task=0, case_id="toy", pos=3,
                        hash=[ep["init_hash"], ep["init_hash_full"]]))
    g_ = globals()
    keep = g_["SHARDS"]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        g_["SHARDS"] = Path(td)
        try:
            save_ep(shard_path(j["eid"]), j, ep, "P")
            m, pr = check_shard(j, "P", C)
            assert pr == [], pr
            assert check_shard(j, "Q", C)[1]
            assert check_shard(dict(j, scene=dict(j["scene"],
                                                  hash=["x", "y"])),
                               "P", C)[1]
            assert check_shard(j, "P", C, None)[1]
            with np.load(shard_path(j["eid"])) as z:
                zz = dict(z)
            meta = json.loads(str(zz["meta"]))
            assert ep_problems(zz, dict(meta, heavy=meta["heavy"] + 1), j, C)
            zz["actions"] = zz["actions"].copy()
            zz["actions"][0, 0] += 1
            assert ep_problems(zz, meta, j, C)
        finally:
            g_["SHARDS"] = keep
    # алгоритм хеша прежних манифестов
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "f.bin"
        fp.write_bytes(b"abc")
        assert file_hash_ok(fp, hashlib.sha1(b"abc").hexdigest())
        assert file_hash_ok(fp, hashlib.sha256(b"abc").hexdigest())
        assert not file_hash_ok(fp, hashlib.sha1(b"x").hexdigest())
    # строгий журнал части
    pl = dict(parts={"0": dict(cap=49 * EP)})
    jb = [dict(eid="a"), dict(eid="b")]
    good = dict(caps=dict(base=49 * EP), spent=dict(base=100),
                pending=None, crashes=[], done={
                    "a": dict(scope="base", actual=50),
                    "b": dict(scope="base", actual=50)})
    assert ledger_problems(good, pl, 0, jb, True) == []
    for bad in (dict(good, crashes=[dict(id="x", scope="base", amount=EP)]),
                dict(good, crashes=[dict(id="a", scope="z", amount=EP)]),
                dict(good, crashes=[dict(id="a", scope="base", amount=5)]),
                dict(good, caps=dict(base=1)),
                dict(good, pending=dict(id="a", scope="base", amount=EP)),
                dict(good, done={"a": dict(scope="base", actual=50)})):
        assert ledger_problems(bad, pl, 0, jb, True), bad
    print("самопроверка k40b_eval_base пройдена: SHA1/SHA256 прежних "
          "манифестов; строгий журнал части; Screen 3 × 32, 4 части × 48, "
          "оба сида сцены в части; игрушечный T32: шард проходит, чужой план, "
          "хеш, тяжёлые вызовы, действия и запись без журнала — отказ")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("bank", "prepare", "preflight", "collect",
                                    "status", "finalize"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--archive", default="reports/k37c/runs/c2")
    a = ap.parse_args()
    import k37_confirm as kcf
    fn = dict(bank=bank, prepare=prepare, preflight=preflight,
              collect=collect, status=status, finalize=finalize)[a.cmd]
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
