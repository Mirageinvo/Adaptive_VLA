"""K-43, этап 3: парный пилот «T32» против «T32 + второй перенос» (v0).

96 свежих физических сцен t8 (банк space 7), один сид T32 s0. На каждой
сцене — T32 с наблюдателем; если развилка есть и правило места дало цель,
вторая ветвь из общего префикса (свежий сброс, переисполнение A[:tau],
сверка хеша развилки). Иначе итог ветви тождествен T32 (не исполняется).
Контраст по сценам: W — спасённые провалы, L — потерянные успехи, D = W − L,
Δ п.п. и бутстреп-ДИ 95% по сценам. Правило (закреплено до пилота): D ≥ 5.

Команды (из k43_handoff.py):
  bank [--workers 8]         банк space 7 (рецепт K-33/K-42, исключения —
                             официальные старты и прежние банки, включая c2)
  prepare                    план: сцены, части, сид, код, config/templates/
                             audit (SHA), бюджет развилки, правило
  collect --part K --device  сбор части (журнал, атомарные шарды, проверка
                             до commit, возобновление)
  status | analyze
Допуск карты — smoke k43_transfer на этой карте (хост, uuid, среда) с теми
же config и templates.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np

TASK = 8
SPACE = 7
N_SCENES = 96
MAX_CAND, CHUNK = 312, 48
N_PARTS = 2
SEED = 0
WAIT, MAX_STEPS, H8 = 10, 600, 8
EP = WAIT + MAX_STEPS
BUDGET = 192
PROTOCOL = "k43-handoff-pilot-v1"
BRANCHES = ("T32", "Transfer")
ROOT = Path("reports/k43/handoff/v1/pilot")
DATA = Path("data/k43/handoff/v1")
BANK_DIR = str(DATA / "bank")
WORK = str(DATA / "bank_work")
MANIFEST = str(ROOT / "bank_manifest.json")
BANK_LEDGER = str(DATA / "bank_ledger.json")
SHARDS = DATA / "shards"
PREV = ("reports/k33/bank_manifest.json", "reports/k37c/bank_manifest.json",
        "reports/k39/bank_manifest.json",
        "reports/k39/pulse/b1/bank_manifest.json",
        "reports/k40b/eval/bank_manifest.json")
PREV_OPT = ("reports/k42/recovery/v1/bank_manifest.json",)
CODE = ("k43_pilot.py", "k43_transfer.py", "k43_handoff.py", "k42_recovery.py",
        "k42_audit.py", "k40b_eval_base.py", "k31_collect.py", "k36_harness.py",
        "k36_bank.py", "k33_bank.py", "k37_bank.py", "k37_confirm.py",
        "k38_grip_run.py", "k39_cache.py", "k29_policy.py", "k32_harness.py")
BOOT = dict(n=50_000, seed=43001)
RULE = dict(D_min=5)
OK, TECH = 0, 3
SUMMARY = ("success", "success_step", "done_step", "own_steps", "steps",
           "physical", "init_hash", "init_hash_full", "actions_sha1")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def h256(s):
    return hashlib.sha256(s.encode()).hexdigest()


def jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist()
                                 if hasattr(o, "tolist") else str(o)))


# --- банк -------------------------------------------------------------------------------------
def prev_seen():
    import k40b_eval_base as kb40
    import k42_recovery as kr
    sigs, seeds, spaces = set(), set(), set()
    for mp in PREV + tuple(p for p in PREV_OPT if os.path.exists(p)):
        if not os.path.exists(mp):
            raise SystemExit(f"нет манифеста прежнего банка {mp}")
        m = json.load(open(mp, encoding="utf-8"))
        spaces |= kr.spaces_of(m)
        for p, s in m["files"].items():
            if not os.path.exists(p) or not kb40.file_hash_ok(p, s):
                raise SystemExit(f"{mp}: файл {p} отсутствует или изменён")
            z = np.load(p, allow_pickle=False)
            sigs.update(str(x) for x in z["sig"])
            seeds.update(int(x) for x in z["seed"])
    if SPACE in spaces or SPACE == kr.SPACE:
        raise SystemExit(f"space {SPACE} уже занят прежним банком")
    return frozenset(sigs), frozenset(seeds), sorted(spaces)


def bank(a):
    """Рецепт банка K-42 (k33_bank select/save, холодный повтор)."""
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
    seen_prev, seeds0, spaces = prev_seen()
    print(f"  занятые пространства по манифестам: {spaces}; беру {SPACE}")
    L = kcol.Ledger(BANK_LEDGER, dict(bank=10 * (MAX_CAND + N_SCENES)))
    official = set(kb.official_joint_sigs(TASK))
    results, start = [], 0
    while True:
        cases, rej = kb.select(results, N_SCENES, set(seen_prev), official)
        if len(cases) >= N_SCENES:
            break
        if start >= MAX_CAND:
            raise SystemExit(f"t{TASK}: допустимых {len(cases)} < {N_SCENES}")
        n = min(CHUNK, MAX_CAND - start)
        jobs = [(SPACE, TASK, i) for i in range(start, start + n)]
        results += k37b.run_chunk(L, f"bank/gen/t{TASK}/c{start}", jobs,
                                  kb._gen_job, a.workers,
                                  os.path.join(WORK, f"gen_c{start}.pkl"))
        start += n
    cases, rej = kb.select(results, N_SCENES, set(seen_prev), official)
    if len({json.dumps(c["meta"], sort_keys=True) for c in cases}) != 1:
        raise SystemExit("разные модели сцен")
    p = kb.task_path(SPACE, TASK, BANK_DIR)
    if os.path.exists(p):
        old, _m = kb.load_task(SPACE, TASK, BANK_DIR)
        if [(c["case_id"], c["init_hash"], c["init_hash_full"]) for c in old] \
                != [(c["case_id"], c["init_hash"], c["init_hash_full"]) for c in cases]:
            raise SystemExit(f"{p}: записанная задача ≠ пересчёту — нужен разбор")
    else:
        p = kb.save_task(SPACE, TASK, cases, cases[0]["meta"], BANK_DIR)
    files = {p: sha256(p)}
    z = np.load(p)
    seeds, sigs = [int(x) for x in z["seed"]], [str(x) for x in z["sig"]]
    if len(set(seeds)) != len(seeds) or set(seeds) & seeds0:
        raise SystemExit("коллизия seed (внутри банка или с прежними)")
    if len(set(sigs)) != len(sigs) or set(sigs) & seen_prev:
        raise SystemExit("коллизия подписей (внутри банка или с прежними)")
    cs, meta = kb.load_task(SPACE, TASK, BANK_DIR)
    jobs = [(SPACE, TASK, c, meta) for c in cs]
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
        elif (r["init_hash"], r["init_hash_full"]) != (c["init_hash"], c["init_hash_full"]):
            bad.append((c["case_id"], "post-wait хеши ≠ банку"))
        elif r["done"] or r["success"] or not r["finite"]:
            bad.append((c["case_id"], "успех/done/нечисловое"))
    man = dict(kind="k43_pilot_bank", space=SPACE, tasks=[TASK], n_per_task=N_SCENES,
               recipe=kb.RECIPE, files=files, candidates=start,
               prev=list(PREV) + [p for p in PREV_OPT if os.path.exists(p)],
               prev_spaces=spaces, replay=dict(n=len(jobs), bad=len(bad),
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
    print(f"  банк space {SPACE}: {len(jobs)} сцен t{TASK}, холодный повтор совпал; "
          f"bank_id {man['bank_id'][:12]}")
    return OK


def split_rows(rows):
    """Ранг по фиксированному хешу (без отбора по исходу), часть = ранг % 2."""
    rows = sorted(rows, key=lambda r: h256(f"{PROTOCOL}|{TASK}|{r['case_id']}"))
    return [dict(r, rank=k, part=k % N_PARTS) for k, r in enumerate(rows)]


def scenes():
    import k33_bank as kb
    m = json.load(open(MANIFEST, encoding="utf-8"))
    bad = [p for p, s in m["files"].items() if not os.path.exists(p) or sha256(p) != s]
    if bad or m.get("replay", {}).get("bad") != 0:
        raise SystemExit(f"банк: файлы изменены {bad} или повтор не пройден")
    cs, _meta = kb.load_task(SPACE, TASK, BANK_DIR)
    if len(cs) != N_SCENES:
        raise SystemExit(f"сцен {len(cs)} ≠ {N_SCENES}")
    rows = [dict(task=TASK, case_id=c["case_id"], pos=pos,
                 hash=[c["init_hash"], c["init_hash_full"]]) for pos, c in enumerate(cs)]
    return m, split_rows(rows)


def eid_of(branch, sc):
    return f"k43|{branch}|s{SEED}|t{sc['task']}|{sc['case_id']}"


# --- план -------------------------------------------------------------------------------------
def plan_path():
    return ROOT / "plan.json"


def prepare(a):
    import k36_protocol as pc
    import k37_confirm as kcf
    import k43_transfer as kt
    m, pilot = scenes()
    k36 = pc.load_prepare()
    cfg = kt.load_cfg()
    for p in (kt.TEMPLATES, kt.OUT / "audit.json"):
        if not p.exists():
            raise SystemExit(f"нет {p}")
    tpl = pc.load_json(str(kt.TEMPLATES))
    if tpl["audit_sha256"] != sha256(kt.OUT / "audit.json"):
        raise SystemExit("шаблоны построены по другому audit.json")
    plan = dict(kind="k43_pilot_plan", protocol=PROTOCOL,
                bank=dict(bank_id=m["bank_id"], files=m["files"], space=SPACE,
                          dir=BANK_DIR, manifest_sha256=sha256(MANIFEST)),
                pilot=pilot, config=cfg,
                config_sha256=sha256(kt.CONFIG) if kt.CONFIG.exists() else None,
                templates_sha256=sha256(kt.TEMPLATES),
                audit_sha256=sha256(kt.OUT / "audit.json"), open_min=tpl["open_min"],
                budget=BUDGET, k36_prepare_sha1=k36["sha1"],
                t32=dict(actor=k36["k32"]["seeds"][str(SEED)]["T"]["actor_hash"],
                         source=k36["k32"]["seeds"][str(SEED)]["source"]["weights_hash"],
                         seed=SEED),
                code={f: sha256(os.path.join(os.path.dirname(__file__), f)) for f in CODE},
                env=dict(B=1, waiting_steps=WAIT, max_steps=MAX_STEPS, horizon=H8),
                rule=RULE, boot=BOOT, parts={})
    for k in range(N_PARTS):
        n = sum(1 for x in pilot if x["part"] == k)
        plan["parts"][str(k)] = dict(n_scenes=n, cap=(2 * n + 2) * EP)
    sh = kcf.canon_sha(plan)
    p = plan_path()
    if p.exists():
        old = pc.load_json(str(p))
        if kcf.canon_sha({k: v for k, v in old.items() if k != "sha256"}) != sh:
            raise SystemExit("план закреплён иначе — новый каталог")
        print(f"  план закреплён: {sh[:16]}")
        return OK
    plan["sha256"] = sh
    pc.save_json(str(p), plan)
    print(f"  план {p}: SHA {sh[:16]}; {len(pilot)} сцен t{TASK}, s{SEED}, бюджет "
          f"развилки {BUDGET}, правило D ≥ {RULE['D_min']}; не более {2 * len(pilot)} "
          f"эпизодов")
    return OK


def load_plan(verify=True):
    import k36_protocol as pc
    import k37_confirm as kcf
    import k43_transfer as kt
    plan = pc.load_json(str(plan_path()))
    if kcf.canon_sha({k: v for k, v in plan.items() if k != "sha256"}) != plan["sha256"]:
        raise SystemExit("plan.json изменён")
    if verify:
        bad = [f for f, s in plan["code"].items()
               if sha256(os.path.join(os.path.dirname(__file__), f)) != s]
        if bad:
            raise SystemExit(f"код ≠ плану: {bad}")
        if sha256(kt.TEMPLATES) != plan["templates_sha256"] or \
                kt.load_cfg() != plan["config"]:
            raise SystemExit("templates/config ≠ плану")
    return plan


# --- шарды и проверки ---------------------------------------------------------------------------
def shard_path(eid):
    return SHARDS / (eid.replace("|", "__") + ".npz")


def save_ep(path, j, ep, plan_sha):
    cl = ep["calls"]
    arr = dict(actions=np.asarray(ep["actions"], np.float32),
               on_hist=np.asarray(ep["on_hist"], bool),
               grasp_hist=np.asarray(ep["grasp_hist"], np.int8),
               pot_hist=np.asarray(ep["pot_hist"], np.float64),
               state_hist=np.asarray(ep["state_hist"], np.float64),
               c_tau=np.array([c["tau"] for c in cl], np.int64),
               c_k=np.array([c["k"] for c in cl], np.int64),
               c_opt=np.array([bool(c["rec"].get("option")) for c in cl], bool),
               c_hash=np.array([c["hash"] for c in cl], dtype="<U40"))
    meta = dict(eid=j["eid"], plan=plan_sha, branch=j["branch"],
                summary={k: ep[k] for k in SUMMARY}, trigger=ep["trigger"],
                goal=ep["plan"], fork_hash=ep["fork_hash"],
                option=ep["option"], heavy=int(ep["heavy"]),
                t32_after=int(ep["t32_after"]),
                ended_in_option=bool(ep["ended_in_option"]),
                on_end=ep["on_end"], seconds=float(ep["seconds"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, meta=np.array(json.dumps(jsonable(meta))), **arr)
    tmp.replace(path)
    return meta


def read_shard(path):
    with np.load(path, allow_pickle=False) as z:
        zz = dict(z)
    return zz, json.loads(str(zz["meta"]))


def fork_of(meta):
    """Развилка, на которой исполняется ветвь: триггер и цель найдена."""
    t, g = meta["trigger"], meta["goal"]
    return t if t is not None and g is not None and g.get("ok") else None


def common_problems(zz, meta, j, plan, start=0):
    import k31_collect as kcol
    p = []
    s = meta["summary"]
    A = zz["actions"]
    if meta["eid"] != j["eid"] or meta["plan"] != plan["sha256"] or \
            meta["branch"] != j["branch"]:
        return ["шард другого эпизода/плана"]
    if kcol.actions_sha(A) != s["actions_sha1"]:
        return ["действия ≠ SHA"]
    if [s["init_hash"], s["init_hash_full"]] != list(j["scene"]["hash"]):
        p.append("начальный хеш ≠ банку")
    if s["steps"] != len(A) or s["physical"] != WAIT + len(A) or s["physical"] > EP:
        p.append("расход ≠ шагам")
    if s["done_step"] >= 0 and s["done_step"] + 1 != s["steps"]:
        p.append("done_step ≠ шагам")
    if s["done_step"] < 0 and s["steps"] != MAX_STEPS:
        p.append("без terminal не до лимита")
    n = len(A) + 1
    for k, sh in (("on_hist", (n, 2)), ("grasp_hist", (n, 2)), ("pot_hist", (n, 2, 3)),
                  ("state_hist", (n, 9))):
        if tuple(zz[k].shape) != sh:
            p.append(f"{k} {zz[k].shape} ≠ {sh}")
    if len(zz["c_tau"]) and (int(zz["c_tau"][0]) != start or any(
            int(zz["c_tau"][i]) + int(zz["c_k"][i]) != int(zz["c_tau"][i + 1])
            for i in range(len(zz["c_tau"]) - 1))):
        p.append("последовательность вызовов")
    return p


def t32_problems(zz, meta, j, plan):
    import k43_handoff as kh
    p = common_problems(zz, meta, j, plan)
    if p and p[0] in ("шард другого эпизода/плана", "действия ≠ SHA"):
        return p
    if zz["c_opt"].any():
        p.append("опция в T32")
    if meta["heavy"] != len(zz["c_tau"]):
        p.append("тяжёлых ≠ вызовов")
    tr = dict(A=zz["actions"], state=zz["state_hist"], on=zz["on_hist"],
              grasp=zz["grasp_hist"])
    h = kh.handoff(tr, plan["open_min"], dict(kh.CFG, budget=plan["budget"]))
    t = meta["trigger"]
    if h["tau"] != (None if t is None else t["tau"]) or \
            (t is not None and (t["first"], t["second"]) != (h["first"], h["second"])):
        p.append("развилка ≠ правилу аудита по записанной трассе")
    if t is not None:
        idx = np.nonzero(zz["c_tau"] == t["tau"])[0]
        if len(idx) != 1 or str(zz["c_hash"][idx[0]]) != meta["fork_hash"]:
            p.append("хеш развилки ≠ границе вызова")
        if meta["goal"] is None:
            p.append("нет записи цели на развилке")
    return p


def transfer_problems(zz, meta, j, plan, z0, m0):
    f0 = fork_of(m0)
    if f0 is None:
        return ["ветвь без развилки T32"]
    tau = int(f0["tau"])
    p = common_problems(zz, meta, j, plan, start=tau)
    if p and p[0] in ("шард другого эпизода/плана", "действия ≠ SHA"):
        return p
    if meta["trigger"] != m0["trigger"] or meta["fork_hash"] != m0["fork_hash"] or \
            meta["goal"] != m0["goal"]:
        p.append("развилка/цель ветви ≠ T32")
    if not np.array_equal(zz["actions"][:tau], z0["actions"][:tau]) or \
            not np.array_equal(zz["on_hist"][:tau + 1], z0["on_hist"][:tau + 1]) or \
            not np.array_equal(zz["state_hist"][:tau + 1], z0["state_hist"][:tau + 1]):
        p.append("префикс ≠ T32")
    if meta["option"] is None:
        p.append("нет опции")
    opt_tau = zz["c_tau"][zz["c_opt"]]
    if len(opt_tau) and int(opt_tau[0]) != tau:
        p.append("первый чанк опции не на развилке")
    if meta["heavy"] != int((~zz["c_opt"]).sum()):
        p.append("тяжёлых ≠ вызовов T32")
    return p


def check_shard(j, plan, t32=None):
    p = shard_path(j["eid"])
    if not p.exists():
        return None, ["нет шарда"]
    zz, m = read_shard(p)
    if j["branch"] == "T32":
        return m, t32_problems(zz, m, j, plan)
    return m, transfer_problems(zz, m, j, plan, *t32)


# --- сбор ---------------------------------------------------------------------------------------
def part_dir(k):
    return ROOT / f"part-{int(k)}"


def jobs_of(plan, k):
    out = []
    for sc in sorted((x for x in plan["pilot"] if x["part"] == k), key=lambda x: x["rank"]):
        for b in BRANCHES:
            out.append(dict(eid=eid_of(b, sc), branch=b, scene=sc))
    return out


def part_loop(L, plan, k, runner, log=print):
    """Сцены части по рангу: T32 -> (если развилка с целью) Transfer. Шард и
    проверка до commit; открытый резерв с готовым проверенным шардом — settle."""
    by = {j["eid"]: j for j in jobs_of(plan, k)}
    t32 = {}

    def t32_of(sc):
        e = eid_of("T32", sc)
        if e not in t32:
            t32[e] = read_shard(shard_path(e))
        return t32[e]

    def check(j):
        if j["branch"] == "T32":
            return check_shard(j, plan)
        return check_shard(j, plan, t32=t32_of(j["scene"]))
    pdg = L.d["pending"]
    if pdg:
        if pdg["id"] not in by:
            raise SystemExit(f"открытый резерв {pdg['id']} не из части")
        m, pr = check(by[pdg["id"]])
        if m is not None and not pr:
            L.settle(pdg["id"], m["summary"]["physical"])

    def ensure(j, prefix=None, expect=None):
        if shard_path(j["eid"]).exists():
            m, pr = check(j)
            if pr:
                raise SystemExit(f"{j['eid']}: {pr[:3]}")
            if not L.is_done(j["eid"]):
                L.settle(j["eid"], m["summary"]["physical"])
            return
        if L.is_done(j["eid"]):
            raise SystemExit(f"{j['eid']}: учтён, шарда нет")
        L.reserve("base", j["eid"], amount=EP)
        ep = runner(j, prefix, expect)
        meta = save_ep(shard_path(j["eid"]), j, ep, plan["sha256"])
        _m, pr = check(j)
        if pr:
            raise SystemExit(f"{j['eid']}: ПРОБЛЕМЫ {pr[:3]} — резерв открыт")
        L.commit(j["eid"], meta["summary"]["physical"])
        log(f"  {j['eid']}: успех {meta['summary']['success']}, шагов "
            f"{meta['summary']['steps']}, развилка "
            f"{(meta['trigger'] or {}).get('tau')}, опция "
            f"{((meta['option'] or {}).get('info') or {}).get('stop')}; учтено "
            f"{len(L.d['done'])}")
    for j in by.values():
        if j["branch"] != "T32":
            continue
        ensure(j)
        sc = j["scene"]
        z0, m0 = t32_of(sc)
        f0 = fork_of(m0)
        if f0 is None:
            continue
        ex = dict(tau=f0["tau"], hash=m0["fork_hash"])
        ensure(by[eid_of("Transfer", sc)], prefix=z0["actions"][:f0["tau"]], expect=ex)
    return OK


def admission_ok(card, plan):
    import k36_protocol as pc
    import k38_grip_run as kgr
    import k43_transfer as kt
    sp = kt.OUT / "smoke" / f"{socket.gethostname()}__{card.g['uuid']}.json"
    if not sp.exists():
        return False, f"нет smoke этой карты {sp}"
    r = pc.load_json(str(sp))
    if not r.get("admitted"):
        return False, "smoke карты не допущен"
    if r.get("templates_sha256") != plan["templates_sha256"] or \
            r.get("config") != plan["config"]:
        return False, "smoke карты — с другими templates/config"
    if r.get("env") != kgr.env_record(card.torch):
        return False, "среда карты ≠ smoke"
    return True, ""


def collect(a):
    import k31_collect as kcol
    import k36_harness as kh
    import k36_protocol as pc
    import k40b_eval_base as kb40
    import k43_transfer as kt
    plan = load_plan()
    k36 = pc.load_prepare()
    C = kh.consts(k36)
    tpl = pc.load_json(str(kt.TEMPLATES))
    cfg = plan["config"]
    k = int(a.part)
    if str(k) not in plan["parts"]:
        raise SystemExit(f"часть {k}")
    card = kb40.Card(a.device)
    pd = part_dir(k)
    pd.mkdir(parents=True, exist_ok=True)
    plock = card.kcf.Lock(str(pd / ".lock"), f"part {k}", card.kcf.Busy)
    try:
        ok, why = admission_ok(card, plan)
        if not ok:
            raise SystemExit(f"карта не допущена: {why}")
        L = kcol.Ledger(str(pd / "ledger.json"), dict(base=plan["parts"][str(k)]["cap"]))
        card.start()
        sim = kt.probe_sim(SPACE, BANK_DIR)
        card.sims.append(sim)
        src = {}

        def runner(j, prefix, expect):
            if not src:
                src["v"], src["arm"] = card.source(k36, SEED)
            card.kcf.check_disk(str(ROOT))
            sc = j["scene"]
            return kt.run_branch(sim, src["v"]["io"], src["arm"], C, cfg, tpl,
                                 plan["open_min"], j["branch"], sc["task"], sc["pos"],
                                 eid_of("T32", sc), prefix=prefix, expect=expect,
                                 budget=plan["budget"])
        try:
            part_loop(L, plan, k, runner, log=lambda s: print(s, flush=True))
        except kcol.Mismatch as e:
            print(f"  техническое расхождение воспроизведения: {e} — резерв открыт, "
                  f"нужен разбор", flush=True)
            return TECH
        src.clear()
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
            print(f"  part {k}: не начата (0/{v['n_scenes']} сцен)")
            continue
        d = json.load(open(lp))
        n_t = sum(1 for e in d["done"] if e.split("|")[1] == "T32")
        n_b = sum(1 for e in d["done"] if e.split("|")[1] == "Transfer")
        print(f"  part {k}: T32 {n_t}/{v['n_scenes']}, ветвей {n_b}; резерв "
              f"{(d['pending'] or {}).get('id', '—')}; аварий {len(d['crashes'])}")
    return OK


# --- анализ -------------------------------------------------------------------------------------
def paired(ya, yb, n_boot=BOOT["n"], seed=BOOT["seed"]):
    d = np.asarray(ya, np.float64) - np.asarray(yb, np.float64)
    W, L_ = int((d == 1).sum()), int((d == -1).sum())
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return dict(W=W, L=L_, D=W - L_, N=len(d), disc=float(np.mean(d != 0)),
                delta_pp=100 * float(d.mean()), lo=100 * float(lo), hi=100 * float(hi))


def summarize(rows, n_boot=BOOT["n"]):
    """rows: dict(T32=(успех, meta), Transfer=(успех, meta|None), fork)."""
    import k43_transfer as kt
    y0 = np.array([int(r["T32"][0]) for r in rows])
    y1 = np.array([int(r["Transfer"][0]) for r in rows])
    c = paired(y1, y0, n_boot=n_boot)
    fk = np.array([r["fork"] is not None for r in rows])
    trg = np.array([r["T32"][1]["trigger"] is not None for r in rows])
    fail = y0 == 0
    br = [r for r in rows if r["Transfer"][1] is not None]
    infos = [(r, r["Transfer"][1]["option"]["info"]) for r in br]
    res = dict(n=len(rows), sr=dict(T32=float(y0.mean()), Transfer=float(y1.mean())),
               contrast=c, decision="signal" if c["D"] >= RULE["D_min"] else "no_signal",
               coverage=dict(trigger=int(trg.sum()), fork=int(fk.sum()),
                             fork_on_fail=int((fk & fail).sum()),
                             fork_on_success=int((fk & ~fail).sum()),
                             fail_T32=int(fail.sum()),
                             goal_refused=dict(_count(r["T32"][1]["goal"]["why"]
                                                      for r in rows
                                                      if r["T32"][1]["trigger"] and
                                                      not r["T32"][1]["goal"]["ok"]))),
               option=dict(full=sum(kt.full_transfer(dict(m, success=bool(
                   m["summary"]["success"]))) for r in br for m in [r["Transfer"][1]]),
                   stops=dict(_count(i.get("stop") for _r, i in infos)),
                   first_left_on=sum(bool(i.get("first_left_on")) or
                                     i.get("stop") == "первая кофеварка вышла из On"
                                     for _r, i in infos),
                   chunks=np.quantile([r["Transfer"][1]["option"]["chunks"] for r in br],
                                      [0, .5, .9, 1]).tolist() if br else None),
               saves=[r["case_id"] for r in rows if r["Transfer"][0] and not r["T32"][0]],
               losses=[dict(case_id=r["case_id"],
                            stop=r["Transfer"][1]["option"]["info"].get("stop"))
                       for r in rows if r["T32"][0] and not r["Transfer"][0]],
               cost=dict(physical=int(sum(r["T32"][1]["summary"]["physical"] +
                                          (r["Transfer"][1]["summary"]["physical"]
                                           if r["Transfer"][1] else 0) for r in rows)),
                         heavy=int(sum(r["T32"][1]["heavy"] + (r["Transfer"][1]["heavy"]
                                                              if r["Transfer"][1] else 0)
                                       for r in rows)),
                         seconds=float(sum(r["T32"][1]["seconds"] +
                                           (r["Transfer"][1]["seconds"]
                                            if r["Transfer"][1] else 0) for r in rows))))
    return res


def _count(it):
    out = {}
    for x in it:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


def collect_rows(plan, final=True):
    import k40b_eval_base as kb40
    rows, prob = [], []
    for k in sorted(int(x) for x in plan["parts"]):
        lp = part_dir(k) / "ledger.json"
        if not lp.exists():
            prob.append(f"part {k}: нет журнала")
            continue
        d = json.load(open(lp))
        want = []
        for sc in sorted((x for x in plan["pilot"] if x["part"] == k),
                         key=lambda x: x["rank"]):
            j0 = dict(eid=eid_of("T32", sc), branch="T32", scene=sc)
            m0, pr = check_shard(j0, plan)
            prob += [f"{j0['eid']}: {x}" for x in pr]
            if m0 is None or pr:
                continue
            want.append(j0)
            z0, m0 = read_shard(shard_path(j0["eid"]))
            f0 = fork_of(m0)
            r = dict(case_id=sc["case_id"], fork=f0, T32=(bool(m0["summary"]["success"]), m0))
            if f0 is None:
                r["Transfer"] = (r["T32"][0], None)
            else:
                j1 = dict(eid=eid_of("Transfer", sc), branch="Transfer", scene=sc)
                m1, pr = check_shard(j1, plan, t32=(z0, m0))
                prob += [f"{j1['eid']}: {x}" for x in pr]
                if m1 is None or pr:
                    continue
                want.append(j1)
                r["Transfer"] = (bool(m1["summary"]["success"]), m1)
            rows.append(r)
        prob += [f"part {k}: {x}" for x in kb40.ledger_problems(d, plan, k, want, final)]
        tot = sum(d["done"][j["eid"]]["actual"] for j in want if j["eid"] in d["done"])
        if d["spent"] != dict(base=tot + EP * len(d["crashes"])):
            prob.append(f"part {k}: расход ≠ сумме")
    if final and len(rows) != N_SCENES:
        prob.append(f"сцен {len(rows)} ≠ {N_SCENES}")
    return rows, prob


def analyze(a):
    import k36_protocol as pc
    plan = load_plan()
    if plan["t32"]["seed"] != SEED:
        raise SystemExit("пилот закреплён на T32 s0")
    rows, prob = collect_rows(plan)
    if prob:
        raise SystemExit("анализ: " + "; ".join(prob[:4]))
    res = summarize(rows)
    res.update(plan=plan["sha256"], budget=plan["budget"])
    pc.save_json(str(ROOT / "analysis.json"), jsonable(res))
    c = res["contrast"]
    cv, op = res["coverage"], res["option"]
    print(f"  T32 s{SEED}, t{TASK}, {res['n']} сцен; SR T32 {100 * res['sr']['T32']:.1f}%, "
          f"T32+перенос {100 * res['sr']['Transfer']:.1f}%")
    print(f"  W {c['W']} L {c['L']} D {c['D']:+d}; Δ {c['delta_pp']:+.1f} п.п. 95% "
          f"[{c['lo']:+.1f}; {c['hi']:+.1f}]; расхождений {100 * c['disc']:.0f}%")
    print(f"  охват: триггер {cv['trigger']}, развилка с целью {cv['fork']} (провалы "
          f"{cv['fork_on_fail']}/{cv['fail_T32']}, успехи {cv['fork_on_success']}); "
          f"отказ цели {cv['goal_refused']}")
    print(f"  опция: полных переносов {op['full']}, остановки {op['stops']}, первая "
          f"выходила из On {op['first_left_on']}, чанков (min/p50/p90/max) {op['chunks']}")
    print(f"  спасения {res['saves']}; потери {res['losses']}")
    print(f"  стоимость: физ. шагов {res['cost']['physical']}, тяжёлых "
          f"{res['cost']['heavy']}, {res['cost']['seconds'] / 3600:.2f} ч")
    print(f"  решение: {res['decision']} (порог D ≥ {RULE['D_min']})")
    return OK


# --- самопроверка ---------------------------------------------------------------------------------
def selftest():
    import tempfile
    import k31_collect as kcol
    import k43_transfer as kt
    g = globals()
    keep = {k: g[k] for k in ("ROOT", "SHARDS")}
    q = np.array([1.0] * 7)
    L_, U_ = -np.ones(6) * 2, np.ones(6) * 2
    C = dict(q=q, L=L_, U=U_)

    class SIO(kcol.FakeIO):
        max_act_q = q

        def prep(self, obs, desc, task):
            return dict(obs, toy_pots=np.array([self._sim.pots.copy()])), 0, \
                np.asarray(obs["state"][:, :8], np.float64)
    sim = kt.ToyPots()
    io = SIO()
    io._sim = sim
    cfg = json.loads(json.dumps(kt.CFG0))
    cfg["pos_m"] = kt.ToyPots.POSM
    tpl = dict(templates=[dict(rel_pre=kt.pq(kt.pose([0, 0, 0.12], [0, 0, 0, 1])),
                               rel_grasp=kt.pq(kt.pose([0, 0, 0.05], [0, 0, 0, 1])),
                               rel_place=kt.pq(kt.pose(p_, [0, 0, 0, 1])))
                          for p_ in ([-0.04, 0.0, 0.0], [0.05, 0.0, 0.0])],
               carry_z=0.15, d_min=0.06, open_min=0.06)
    # init 0 — T32 ставит одну (ветвь спасает), 1 — обе, 2 — как 0
    hs = {}
    for i in range(3):
        e = kcol.run_episode(sim, io, kt.ToyT32(q, L_, U_), TASK, i)
        hs[i] = [e["init_hash"], e["init_hash_full"]]
    pilot = [dict(task=TASK, case_id=f"c{i}", pos=i, hash=hs[i], rank=i, part=0)
             for i in range(3)]
    with tempfile.TemporaryDirectory() as td:
        g["ROOT"], g["SHARDS"] = Path(td) / "r", Path(td) / "s"
        try:
            plan = dict(sha256="P", pilot=pilot, parts={"0": dict(cap=8 * EP)},
                        open_min=0.06, budget=BUDGET, config=cfg)
            n_run = dict(n=0, crash=None)

            def runner(j, prefix, expect):
                n_run["n"] += 1
                sc = j["scene"]
                ep = kt.run_branch(sim, io, kt.ToyT32(q, L_, U_), C, cfg, tpl, 0.06,
                                   j["branch"], TASK, sc["pos"], eid_of("T32", sc),
                                   prefix=prefix, expect=expect, wait=10, budget=BUDGET)
                return ep
            pd = part_dir(0)
            pd.mkdir(parents=True)
            Lg = kcol.Ledger(str(pd / "ledger.json"), dict(base=8 * EP))
            # авария: шард записан, commit не успел
            orig = Lg.commit
            calls = dict(n=0)

            def crash_commit(eid, actual):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("авария до commit")
                return orig(eid, actual)
            Lg.commit = crash_commit
            try:
                part_loop(Lg, plan, 0, runner, log=lambda s: None)
                raise AssertionError("авария не сработала")
            except RuntimeError:
                pass
            Lg = kcol.Ledger(str(pd / "ledger.json"), dict(base=8 * EP))
            n0 = n_run["n"]
            part_loop(Lg, plan, 0, runner, log=lambda s: None)
            rows, prob = collect_rows(plan, final=False)
            done = set(Lg.d["done"])
            assert not prob, prob
            assert len(rows) == 3, rows
            # c0 и c2: развилка, ветвь спасает; c1: успех T32
            assert rows[0]["fork"] is not None and rows[0]["Transfer"][0] and \
                not rows[0]["T32"][0]
            assert rows[1]["T32"][0]
            res = summarize(rows, n_boot=200)
            assert res["contrast"]["W"] >= 2 and res["contrast"]["L"] == 0, res["contrast"]
            assert res["option"]["full"] >= 2
            # повтор ничего не исполняет
            n1 = n_run["n"]
            part_loop(kcol.Ledger(str(pd / "ledger.json"), dict(base=8 * EP)), plan, 0,
                      runner, log=lambda s: None)
            assert n_run["n"] == n1 and n1 > n0
            assert done == {j["eid"] for j in jobs_of(plan, 0)
                            if j["branch"] == "T32" or fork_of(read_shard(shard_path(
                                eid_of("T32", j["scene"])))[1])}
            # порча шардов ловится
            j0 = dict(eid=eid_of("T32", pilot[0]), branch="T32", scene=pilot[0])
            z0, m0 = read_shard(shard_path(j0["eid"]))
            assert t32_problems(z0, m0, j0, plan) == []
            bad = dict(z0, actions=z0["actions"].copy())
            bad["actions"][0, 0] += 1
            assert t32_problems(bad, m0, j0, plan)
            mt = json.loads(json.dumps(m0))
            mt["trigger"]["tau"] += 8
            assert t32_problems(z0, mt, j0, plan)
            j1 = dict(eid=eid_of("Transfer", pilot[0]), branch="Transfer", scene=pilot[0])
            z1, m1 = read_shard(shard_path(j1["eid"]))
            assert transfer_problems(z1, m1, j1, plan, z0, m0) == []
            b1 = dict(z1, state_hist=z1["state_hist"].copy())
            b1["state_hist"][3, 2] += 1e-3
            assert "префикс ≠ T32" in transfer_problems(b1, m1, j1, plan, z0, m0)
            mg = json.loads(json.dumps(m1))
            mg["goal"]["place_i"] = 0
            assert transfer_problems(z1, mg, j1, plan, z0, m0)
            assert check_shard(j1, dict(plan, sha256="Q"), t32=(z0, m0))[1]
        finally:
            g.update(keep)
    # правило
    assert paired([1] * 6 + [0] * 90, [0] * 6 + [0] * 90, n_boot=100)["D"] == 6
    print("k43_pilot selftest ok")
