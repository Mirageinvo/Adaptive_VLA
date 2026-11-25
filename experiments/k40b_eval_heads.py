#!/usr/bin/env python3
"""K-40 B, оценка — фаза II: TreeB-Mix / FlatB-Mix на Screen и анализ.

План закрепляет базовый план и baseline_manifest (фаза I), PCA (pca.npz +
pca.json), четыре чекпойнта (TreeB_Mix_s0/s1, FlatB_Mix_s0/s1; сопоставление
T32_s → *_s), отчёт обучения, план сбора B и SHA кода (k40b_train — тот же,
что обучал: train_code_sha256 чекпойнтов).
Политика: на каждом чанке один проход T32 (Arm.decide) с перехватом
признаков всех узлов и визуальных токенов того же прохода (VisTap);
z0 = featurize(группы токенов, state11 текущего наблюдения); голова:
model.ctx → model.goals(z0) → model.decode([z0, ĝ2, ĝ4, ĝ6, ĝ8]) → Δ;
итоговый чанк — k40b_train.compose_np(B, Δ) (клип L/U, схват из B).
Никаких будущих наблюдений, повторных вызовов T32 и триггеров.
Эпизод — k31_collect.run_episode с обёрткой агента: журнал K-32 хранит
номинальный B, исполняется итоговый чанк; по вызовам пишутся B, Δ, итог,
подцели и число исполненных шагов.
Анализ: T32 (фаза I), TreeB, FlatB — 576 эпизодов; контрасты TreeB−T32,
TreeB−FlatB, FlatB−T32: SR, W/L/D, Δ п.п., 95% ДИ (50 000, сцены внутри
задач, оба сида вместе, задачи с равным весом), задачи, сиды, доля
расхождений; статистика поправок. Правило отбора: D(TreeB−T32) ≥ 6, по
обоим сидам ≥ 0, D(TreeB−FlatB) > 0.
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

PROTOCOL = "k40b-eval-heads-v1"
POLICIES = ("TreeB", "FlatB")
CKPT = {"TreeB": "TreeB_Mix_s{}.pt", "FlatB": "FlatB_Mix_s{}.pt"}
MODELS = Path("data/k40b/models/v1")
PCA = Path("data/k40b/pca.npz")
PCA_REP = Path("reports/k40b/pca.json")
TRAIN_REP = Path("reports/k40b/train_v1.json")
COLLECT_PLAN = Path("data/k40b/collect/v1/plan.json")
ROOT = Path("reports/k40b/eval/heads")
SHARDS = Path("data/k40b/eval/heads/shards")
CODE = ("k40b_eval_heads.py", "k40b_eval_base.py", "k40b_train.py",
        "k40b_collect.py", "k40a_transfer.py", "k31_collect.py",
        "k32_harness.py", "k32_policy.py", "k36_harness.py")
BOOT = dict(n=50_000, seed=40002)
RULE = dict(tree_minus_t32_D=6, seeds_nonneg=True, tree_minus_flat_D=1)
EP = 610
OK, TECH, BUSY = 0, 3, 4


def base():
    import k40b_eval_base as kb
    return kb


def sha256(p):
    import k37_confirm as kcf
    return kcf.sha256(p)


# --- голова ---------------------------------------------------------------------------------------
class Head:
    """Обученная голова на CPU (float32): z0 + признаки узлов -> Δ."""

    def __init__(self, torch, ckpt, P, C):
        import k40b_train as kt
        self.torch, self.kt, self.P = torch, kt, P
        M = kt.build(torch)
        self.model = (M["Tree"]() if ckpt["kind"] == "tree" else
                      M["Flat"](ckpt["params"]["flat_hidden"]))
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()
        self.C = C
        self.K = kt.Consts(torch, C, "cpu")

    def __call__(self, F, vis_tokens, s8, B):
        """F — признаки узлов (numpy, без батча), vis_tokens — выход
        энкодера+коннектора, s8 [1, 8], B [8, 7] -> (Δ [8, 6], подцели
        [4, 75], z0 [75])."""
        import k21_model as km
        import k40b_collect as kc
        torch, kt = self.torch, self.kt
        v = kc.group_vis(vis_tokens).reshape(1, 1, -1)
        s11 = km.state11(np.asarray(s8, np.float64)).reshape(1, 1, -1)
        z0 = kt.featurize(v, s11, self.P)[0, 0]
        if not np.isfinite(z0).all():
            raise SystemExit("z0 не конечен")
        with torch.no_grad():
            Ft = {k: torch.as_tensor(np.asarray(F[k], np.float32))[None]
                  for k in kt.F_LIST}
            Bt = torch.as_tensor(np.asarray(B, np.float32))[None]
            zt = torch.as_tensor(z0)[None]
            c = self.model.ctx(Ft)
            Bn = Bt[..., :6] / self.K.half
            g = self.model.goals(c, zt, Bn)
            d = self.model.decode(c, [zt] + g, Bn)
        d = d[0].numpy().astype(np.float32)
        goals = torch.stack(g, 1)[0].numpy().astype(np.float32)
        if d.shape != (8, 6) or not np.isfinite(d).all() or \
                np.abs(d).max() > kt.CAP + 1e-6:
            raise SystemExit(f"Δ формы {d.shape} или вне CAP")
        return d, goals, z0


class HeadArm:
    """Агент для run_episode: штатный decide T32 с перехватом признаков,
    голова, итоговый чанк. rec["a_codec"] — номинальный B (журнал K-32)."""

    def __init__(self, arm, bar, head, C):
        self.arm, self.bar, self.head, self.C = arm, bar, head, C
        self.log = []
        self.heavy = 0

    @property
    def call(self):
        return self.arm.call

    @call.setter
    def call(self, v):
        self.arm.call = v

    @property
    def episode(self):
        return self.arm.episode

    @episode.setter
    def episode(self, v):
        self.arm.episode = v

    @property
    def last(self):
        return self.arm.last

    def begin(self, task, init, episode=None):
        self.log = []
        self.heavy = 0
        self.arm.begin(task, init, episode)

    def decide(self, batch, po, s8, elapsed):
        import k40a_transfer as ka
        import k40b_collect as kc
        import k40b_train as kt
        got = {}
        fwd = self.arm.forward

        def wrapped(*x, **kw):
            self.heavy += 1
            out = fwd(*x, **kw)
            got["F"] = {k: out[0][k][0].detach().float().cpu().numpy()
                        for k in ka.F_KEYS}
            return out
        self.arm.forward = wrapped
        try:
            with kc.VisTap(self.bar) as tap:
                B, rec = self.arm.decide(batch, po, s8, elapsed)
        finally:
            self.arm.forward = fwd
        B = np.asarray(B, np.float32)
        d, goals, z0 = self.head(got["F"], tap.got, s8, B)
        final = kt.compose_np(B, d, self.C["half"], self.C["L"], self.C["U"])
        self.log.append(dict(B=B, delta=d, final=final, goals=goals,
                             z0=z0, elapsed=int(elapsed)))
        return final, rec

    def executed(self, k):
        self.arm.executed(k)
        self.log[-1]["k"] = int(k)


def zero_head(torch, P, C, kind="tree", flat_hidden=64):
    """Голова с нулевой инициализацией: Δ ≡ 0 (проверка тождества T32)."""
    import k40b_train as kt
    M = kt.build(torch)
    m = M["Tree"]() if kind == "tree" else M["Flat"](flat_hidden)
    ck = dict(kind=kind, params=dict(flat_hidden=flat_hidden),
              state=m.state_dict())
    return Head(torch, ck, P, C)


# --- план ---------------------------------------------------------------------------------------------
def eid_of(policy, seed, sc):
    return base().eid_of(policy, seed, sc)


def queue(plan, part):
    out = []
    bp = plan["base_plan_obj"]
    for s in (0, 1):
        for sc in sorted((x for x in bp["screen"] if x["part"] == part),
                         key=lambda x: (x["rank"], x["task"])):
            for pol in (POLICIES if (sc["rank"] + s) % 2 == 0
                        else POLICIES[::-1]):
                out.append(dict(eid=eid_of(pol, s, sc), policy=pol, seed=s,
                                scene=sc))
    return out


def plan_path():
    return ROOT / "plan.json"


def verify_ckpt(ck, name, pca_sha, train_sha):
    pr = []
    if ck.get("pca_sha256") != pca_sha:
        pr.append("PCA ≠ закреплённому")
    if ck.get("train_code_sha256") != train_sha:
        pr.append("код обучения ≠ k40b_train.py")
    if ck.get("data") != "Mix" or ck.get("kind") != dict(
            TreeB="tree", FlatB="flat")[name]:
        pr.append(f"вид/данные {ck.get('kind')}/{ck.get('data')}")
    return pr


def prepare(a):
    import torch
    import k36_protocol as pc
    import k37_confirm as kcf
    kb = base()
    bp = kb.load_plan()
    bm_p = kb.ROOT / "baseline_manifest.json"
    if not bm_p.exists():
        raise SystemExit("нет baseline_manifest — сначала фаза I finalize")
    bm = pc.load_json(str(bm_p))
    if bm.get("plan") != bp["sha256"] or set(bm["episodes"]) != \
            set(bp["episodes"]):
        raise SystemExit("baseline_manifest ≠ базовому плану")
    pca_sha = sha256(PCA)
    rep = pc.load_json(str(PCA_REP))
    if rep.get("out_sha256") != pca_sha:
        raise SystemExit("PCA ≠ своему отчёту")
    train_sha = sha256(Path(HERE) / "k40b_train.py")
    ck = {}
    for pol in POLICIES:
        for s in (0, 1):
            p = MODELS / CKPT[pol].format(s)
            c = torch.load(p, map_location="cpu")
            pr = verify_ckpt(c, pol, pca_sha, train_sha)
            if pr or c.get("seed") != s:
                raise SystemExit(f"{p}: {pr} сид {c.get('seed')}")
            ck[f"{pol}_s{s}"] = dict(path=str(p), sha256=sha256(p))
    plan = dict(kind="k40b_eval_heads_plan", protocol=PROTOCOL,
                base_plan=bp["sha256"], baseline_manifest=sha256(bm_p),
                pca=dict(path=str(PCA), sha256=pca_sha,
                         report_sha256=sha256(PCA_REP)),
                train_report_sha256=sha256(TRAIN_REP),
                collect_plan_sha256=sha256(COLLECT_PLAN),
                checkpoints=ck, seed_map={"0": "s0", "1": "s1"},
                code={f: sha256(os.path.join(HERE, f)) for f in CODE},
                rule=RULE, boot=BOOT,
                parts={str(k): dict(n=0, cap=0) for k in range(kb.N_PARTS)})
    pl = dict(plan, base_plan_obj=bp)
    for k in range(kb.N_PARTS):
        n = len(queue(pl, k))
        plan["parts"][str(k)] = dict(n=n, cap=(n + 1) * EP)
    plan["episodes"] = [j["eid"] for k in range(kb.N_PARTS)
                        for j in queue(pl, k)]
    sh = kcf.canon_sha(plan)
    p = plan_path()
    if p.exists():
        old = pc.load_json(str(p))
        if kcf.canon_sha({k: v for k, v in old.items() if k != "sha256"}) \
                != sh:
            raise SystemExit("план головок закреплён иначе")
        print(f"  план закреплён: {sh[:16]}")
        return OK
    plan["sha256"] = sh
    pc.save_json(str(p), plan)
    print(f"  план головок {sh[:16]}: эпизодов {len(plan['episodes'])} "
          f"(TreeB и FlatB × 96 сцен × 2 сида)")
    return OK


def load_plan(verify=True):
    import k36_protocol as pc
    import k37_confirm as kcf
    p = pc.load_json(str(plan_path()))
    if kcf.canon_sha({k: v for k, v in p.items() if k != "sha256"}) != \
            p["sha256"]:
        raise SystemExit("plan.json ≠ своему SHA")
    kb = base()
    bp = kb.load_plan(verify=verify)
    if bp["sha256"] != p["base_plan"]:
        raise SystemExit("базовый план ≠ плану головок")
    if verify:
        bad = [f for f, s in p["code"].items()
               if sha256(os.path.join(HERE, f)) != s]
        bad += [k for k, v in p["checkpoints"].items()
                if sha256(v["path"]) != v["sha256"]]
        if sha256(PCA) != p["pca"]["sha256"]:
            bad.append("PCA")
        if sha256(kb.ROOT / "baseline_manifest.json") != \
                p["baseline_manifest"]:
            bad.append("baseline_manifest")
        if bad:
            raise SystemExit(f"≠ плану: {bad}")
    p["base_plan_obj"] = bp
    return p


# --- эпизод ----------------------------------------------------------------------------------------------
def run_head(sim, io, arm, bar, head, C, task, pos, eid):
    import k31_collect as kcol
    from k36_harness import episode_number
    ha = HeadArm(arm, bar, head, C)
    arm.episode = episode_number(eid)
    ep = kcol.run_episode(sim, io, ha, task, pos)
    ep["heavy"] = ha.heavy
    ep["log"] = ha.log
    return ep


def shard_path(eid):
    return SHARDS / (eid.replace("|", "__") + ".npz")


def save_ep(path, j, ep, plan_sha):
    import k32_harness as kh32
    kb = base()
    calls = [dict(c, rec=c["rec"]) for c in ep["calls"]]
    J = kh32.pack_journal([(0, calls)], 6)
    lg = ep["log"]
    H = dict(B=np.stack([x["B"] for x in lg]),
             delta=np.stack([x["delta"] for x in lg]),
             final=np.stack([x["final"] for x in lg]),
             goals=np.stack([x["goals"] for x in lg]),
             z0=np.stack([x["z0"] for x in lg]),
             k=np.array([x.get("k", -1) for x in lg], np.int64))
    meta = dict(eid=j["eid"], plan=plan_sha, policy=j["policy"],
                heavy=int(ep["heavy"]), seconds=float(ep["seconds"]),
                summary={k: ep[k] for k in kb.SUMMARY})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, meta=np.array(json.dumps(meta)),
                            actions=np.asarray(ep["actions"], np.float32),
                            **{f"j_{k}": v for k, v in J.items()},
                            **{f"h_{k}": v for k, v in H.items()})
    tmp.replace(path)
    return meta


def ep_problems(z, meta, j, C):
    import k29_policy as kp
    import k31_collect as kcol
    import k32_harness as kh32
    import k40b_train as kt
    p = []
    A = z["actions"]
    s = meta["summary"]
    if kcol.actions_sha(A) != s["actions_sha1"]:
        return ["действия ≠ SHA"]
    if [s["init_hash"], s["init_hash_full"]] != list(j["scene"]["hash"]):
        p.append("начальный хеш ≠ банку")
    J = {k[2:]: v for k, v in z.items() if k.startswith("j_")}
    H = {k[2:]: v for k, v in z.items() if k.startswith("h_")}
    n = len(J.get("ep", []))
    if n == 0 or any(len(H.get(k, [])) != n for k in
                     ("B", "delta", "final", "goals", "z0", "k")):
        return p + ["журнал или записи головы неполны"]
    q = C["q"]
    A_nom = np.concatenate([kp.scale_exec(J["a_codec"][i][None], q)[0]
                            [:int(J["k"][i])] for i in range(n)])
    p += [f"журнал: {x}" for x in kh32.journal_problems(
        J, [A_nom], [s["done_step"]], C["half"], C["mid"], C["L"], C["U"],
        q, "T")[:3]]
    if not np.array_equal(H["B"], J["a_codec"]) or \
            not np.array_equal(H["k"], J["k"]):
        p.append("B/k головы ≠ журналу")
    d = H["delta"]
    if d.shape != (n, 8, 6) or not np.isfinite(d).all() or \
            np.abs(d).max() > kt.CAP + 1e-6:
        p.append("Δ формы/конечности/CAP")
    re = kt.compose_np(H["B"], d, C["half"], C["L"], C["U"])
    if not np.array_equal(re, H["final"]):
        p.append("итог ≠ compose(B, Δ)")
    if not np.array_equal(H["final"][..., 6], H["B"][..., 6]):
        p.append("схват изменён")
    ex = np.concatenate([kp.scale_exec(H["final"][i][None], q)[0]
                         [:int(J["k"][i])] for i in range(n)])
    if not np.array_equal(ex.astype(np.float32), np.asarray(A, np.float32)):
        p.append("исполненное ≠ масштабу итоговых чанков")
    if meta["heavy"] != n:
        p.append(f"тяжёлых вызовов {meta['heavy']} ≠ чанков {n}")
    if not np.isfinite(H["goals"]).all() or H["goals"].shape != (n, 4, 75):
        p.append("подцели формы/конечности")
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
    if meta.get("eid") != j["eid"] or meta.get("plan") != plan_sha or \
            meta.get("policy") != j["policy"]:
        return meta, ["шард чужого эпизода или плана"]
    p = ep_problems(zz, meta, j, C)
    if done != "_no" and done != dict(scope="base",
                                      actual=meta["summary"]["physical"]):
        p.append(f"журнал {done} ≠ записи")
    return meta, p


# --- допуск и сбор --------------------------------------------------------------------------------------
def load_head(torch, plan, pol, s, C):
    """Чекпойнт сида s (SHA сверен в load_plan) -> Head на CPU."""
    P = dict(np.load(PCA))
    ck = torch.load(plan["checkpoints"][f"{pol}_s{s}"]["path"],
                    map_location="cpu")
    if not ck.get("state"):
        raise SystemExit("пустой чекпойнт")
    return Head(torch, ck, P, C)


def admission_path(card):
    return ROOT / "preflight" / f"{socket.gethostname()}__{card.g['uuid']}.json"


def preflight(a):
    """Нулевая голова = T32 побитово на базовом эпизоде; обученные головы
    дают конечные Δ в пределах CAP на том же старте."""
    import k36_harness as kh
    import k36_protocol as pc
    import k38_grip_run as kgr
    kb = base()
    plan = load_plan()
    k36 = pc.load_prepare()
    C = kh.consts(k36)
    bp = plan["base_plan_obj"]
    sc = sorted((x for x in bp["screen"] if x["part"] == 0),
                key=lambda x: (x["rank"], x["task"]))[0]
    card = kb.Card(a.device)
    prob, info = [], {}
    try:
        card.start()
        sim = card.sim(kb.SPACE, kb.BANK_DIR)
        s = 0
        v, arm = card.source(k36, s)
        ref = kb.shard_path(kb.eid_of("T32", s, sc))
        with np.load(ref) as z:
            A_ref = z["actions"]
        P = dict(np.load(PCA))
        ep0 = run_head(sim, v["io"], arm, v["mod"].bar,
                       zero_head(card.torch, P, C), C, sc["task"],
                       sc["pos"], kb.eid_of("T32", s, sc))
        if not np.array_equal(np.asarray(ep0["actions"], np.float32), A_ref):
            prob.append("нулевая голова ≠ базовому T32")
        info["zero_seconds"] = float(ep0["seconds"])
        for pol in POLICIES:
            hd = load_head(card.torch, plan, pol, s, C)
            ep = run_head(sim, v["io"], arm, v["mod"].bar, hd, C, sc["task"],
                          sc["pos"], eid_of(pol, s, sc))
            d = np.stack([x["delta"] for x in ep["log"]])
            info[pol] = dict(success=bool(ep["success"]),
                             delta_rms=float(np.sqrt((d ** 2).mean())),
                             seconds=float(ep["seconds"]))
            if ep["heavy"] != len(ep["calls"]):
                prob.append(f"{pol}: тяжёлых ≠ чанков")
        arm = v = None
        kh.free(card.torch)
        rep = dict(admitted=not prob, problems=prob, info=info,
                   plan=plan["sha256"], env=kgr.env_record(card.torch),
                   uuid=card.g["uuid"], host=socket.gethostname())
        pc.save_json(str(admission_path(card)), rep)
    finally:
        card.close()
    print(f"  preflight K-40 B heads {a.device}: " +
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
    kb = base()
    plan = load_plan()
    k36 = pc.load_prepare()
    C = kh.consts(k36)
    k = int(a.part)
    if str(k) not in plan["parts"]:
        raise SystemExit(f"часть {k}")
    card = kb.Card(a.device)
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
        lp_ = kb.ledger_problems(L.d, plan, k, jobs, final=False)
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
        sim = card.sim(kb.SPACE, kb.BANK_DIR)
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
                    heads = {pol: load_head(card.torch, plan, pol, s, C)
                             for pol in POLICIES}
                card.kcf.check_disk(str(ROOT))
                L.reserve("base", j["eid"], amount=EP)
                ep = run_head(sim, v["io"], arm, v["mod"].bar,
                              heads[j["policy"]], C, j["scene"]["task"],
                              j["scene"]["pos"], j["eid"])
                meta = save_ep(shard_path(j["eid"]), j, ep, plan["sha256"])
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


# --- анализ -------------------------------------------------------------------------------------------------
def contrast(Y, a, b, scenes, tasks, n_boot=BOOT["n"], seed=BOOT["seed"]):
    """Y[(policy, seed, task, case)] -> W/L/D, Δ, ДИ по сценам в задачах."""
    dif = {}
    for (t, c) in scenes:
        for s in (0, 1):
            dif[(t, c, s)] = Y[(a, s, t, c)] - Y[(b, s, t, c)]
    v = np.array(list(dif.values()))
    W, L_ = int((v == 1).sum()), int((v == -1).sum())
    per_t = {t: np.array([np.mean([dif[(t, c, s)] for s in (0, 1)])
                          for (tt, c) in scenes if tt == t]) for t in tasks}
    est = float(np.mean([per_t[t].mean() for t in tasks]))
    rng = np.random.default_rng(seed)
    bs = np.mean([per_t[t][rng.integers(0, len(per_t[t]),
                                        (n_boot, len(per_t[t])))].mean(1)
                  for t in tasks], 0)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return dict(W=W, L=L_, D=W - L_, N=len(v), disc=float(np.mean(v != 0)),
                delta_pp=100 * est, lo=100 * float(lo), hi=100 * float(hi),
                by_task={str(t): int(sum(dif[k] for k in dif if k[0] == t))
                         for t in tasks},
                by_seed={str(s): int(sum(dif[k] for k in dif if k[2] == s))
                         for s in (0, 1)})


def analyze(a):
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    import k40b_train as kt
    kb = base()
    plan = load_plan()
    bp = plan["base_plan_obj"]
    C = kh.consts(pc.load_prepare())
    bm = pc.load_json(str(kb.ROOT / "baseline_manifest.json"))
    Y, prob, dstat = {}, [], {p: [] for p in POLICIES}
    for e, v in bm["episodes"].items():
        if sha256(kb.shard_path(e)) != v["sha256"]:
            prob.append(f"{e}: базовый шард ≠ манифесту")
        Y[("T32", v["seed"], v["task"], v["case_id"])] = int(v["success"])
    locks = []
    try:
        for k in range(kb.N_PARTS):
            locks.append(kcf.Lock(str(part_dir(k) / ".lock"), f"part {k}",
                                  kcf.Busy))
            d = json.load(open(part_dir(k) / "ledger.json"))
            jobs = queue(plan, k)
            lpr = kb.ledger_problems(d, plan, k, jobs, final=True)
            prob += [f"part {k}: {x}" for x in lpr]
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
                with np.load(shard_path(j["eid"])) as z:
                    dstat[j["policy"]].append(z["h_delta"].reshape(-1))
            if d["spent"] != dict(base=tot + EP * len(d["crashes"])):
                prob.append(f"part {k}: расход ≠ сумме")
    finally:
        for lk in locks:
            lk.release()
    if len(Y) != 576:
        prob.append(f"эпизодов {len(Y)} ≠ 576")
    if prob:
        raise SystemExit("анализ: " + "; ".join(prob[:4]))
    tasks = sorted({sc["task"] for sc in bp["screen"]})
    scenes = sorted({(sc["task"], sc["case_id"]) for sc in bp["screen"]})
    res = dict(
        sr={p: {str(t): float(np.mean([Y[(p, s, t, c)] for (tt, c) in scenes
                                       if tt == t for s in (0, 1)]))
                for t in tasks} for p in ("T32", *POLICIES)},
        tree_t32=contrast(Y, "TreeB", "T32", scenes, tasks),
        tree_flat=contrast(Y, "TreeB", "FlatB", scenes, tasks),
        flat_t32=contrast(Y, "FlatB", "T32", scenes, tasks),
        delta={p: dict(rms=float(np.sqrt(np.mean(np.concatenate(v) ** 2))),
                       at_cap=float(np.mean(np.abs(np.concatenate(v)) >=
                                            0.999 * kt.CAP)))
               for p, v in dstat.items()})
    tt, tf = res["tree_t32"], res["tree_flat"]
    res["pass"] = bool(tt["D"] >= RULE["tree_minus_t32_D"] and
                       all(x >= 0 for x in tt["by_seed"].values()) and
                       tf["D"] >= RULE["tree_minus_flat_D"])
    pc.save_json(str(ROOT / "analysis.json"), dict(res, plan=plan["sha256"]))
    for p in ("T32", *POLICIES):
        print(f"  SR {p}: {res['sr'][p]}")
    for n_ in ("tree_t32", "tree_flat", "flat_t32"):
        c = res[n_]
        print(f"  {n_}: W {c['W']} L {c['L']} D {c['D']:+d} (N {c['N']}); "
              f"Δ {c['delta_pp']:+.1f} п.п. 95% [{c['lo']:+.1f}; "
              f"{c['hi']:+.1f}]; задачи {c['by_task']}; сиды {c['by_seed']};"
              f" расхождений {100 * c['disc']:.0f}%")
    print(f"  поправки: {res['delta']}")
    print(f"  правило отбора: {'ПРОЙДЕНО' if res['pass'] else 'не пройдено'}")
    return OK


# --- самопроверка ---------------------------------------------------------------------------------------------
def selftest():
    import torch
    import k21_model as km21
    import k25_model as k25m
    import k29_model as k29m
    import k31_collect as kcol
    import k32_model as km32
    import k32_ppo as kppo
    import k36_harness as kh
    import k40b_collect as kc
    import k40b_train as kt
    k29m.no_tf32(torch)
    torch.manual_seed(0)
    # контраст: известные W/L, по сценам
    tasks = (0, 6)
    scenes = [(t, f"c{i}") for t in tasks for i in range(4)]
    Y = {}
    for (t, c) in scenes:
        for s in (0, 1):
            Y[("T32", s, t, c)] = 0 if c == "c0" else 1
            Y[("TreeB", s, t, c)] = 1 if c in ("c0", "c1") else 1
            Y[("FlatB", s, t, c)] = 0
    r = contrast(Y, "TreeB", "T32", scenes, tasks, n_boot=200)
    assert (r["W"], r["L"], r["D"], r["N"]) == (4, 0, 4, 16)
    assert abs(r["delta_pp"] - 25.0) < 1e-9
    # игрушечная политика: нулевая голова = T32, обученная — проверки
    M = k29m.build(torch)
    Lb = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    Ub = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    mod = k25m.build(torch)["K25"](km21.fake_bar(torch), Lb, Ub,
                                   seed=0).eval()
    for p_ in mod.parameters():
        p_.requires_grad_(False)
    dec = M["Decoder"](mod).freeze()
    H_ = mod.H
    norm = km32.fit_normalizer(np.random.default_rng(0).normal(
        size=(64, 6 * H_ + 72)), H_)
    actor, _c = km32.build(torch, H_, 6, 0, norm)

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
    # визуальные признаки игрушечной VLA: D
    bt = km21.fake_batch(torch, B=1, seed=0)
    with torch.no_grad():
        img = mod.bar.vlm.get_image_features(bt["pixel_values"],
                                             bt.get("pixel_attention_mask"))
    D = int(img.shape[-1])
    rng = np.random.default_rng(0)
    P = dict(mean=np.zeros(4 * D, np.float32),
             comps=rng.normal(size=(kt.N_VIS, 4 * D)).astype(np.float32),
             proj_std=np.ones(kt.N_VIS, np.float32),
             s11_mean=np.zeros(11, np.float32),
             s11_std=np.ones(11, np.float32))
    keep_h = kt.H_DIM
    kt.H_DIM = H_                       # ширина узлов игрушечной VLA
    try:
        _toy_heads(torch, kh, kcol, kt, k36, src, Sim, io, mod, P, C)
    finally:
        kt.H_DIM = keep_h
    del kc
    print("самопроверка k40b_eval_heads пройдена: нулевая голова = T32 "
          "побитово, тяжёлых = чанков; обученная: Δ в CAP, итог = compose, "
          "схват из B, исполненное = итог; порча шарда — отказ; контраст "
          "W/L/D и Δ по сценам")
    return 0


def _toy_heads(torch, kh, kcol, kt, k36, src, Sim, io, mod, P, C):
    import tempfile
    arm = kh.make_arm(torch, k36, "T32_s0", src)
    ep_t = kcol.run_episode(Sim(), io, arm, 0, 3)
    arm = kh.make_arm(torch, k36, "T32_s0", src)
    ep0 = run_head(Sim(), io, arm, mod.bar, zero_head(torch, P, C), C, 0, 3,
                   "e0")
    assert np.array_equal(ep0["actions"], ep_t["actions"]), "нулевая ≠ T32"
    assert ep0["heavy"] == len(ep0["calls"])
    # обученная (ненулевая) голова: Δ ≠ 0, проверки шарда
    hz = zero_head(torch, P, C)
    with torch.no_grad():
        hz.model.dec[-1].bias.fill_(0.3)
    arm = kh.make_arm(torch, k36, "T32_s0", src)
    ep1 = run_head(Sim(), io, arm, mod.bar, hz, C, 0, 3, "e1")
    dd = np.stack([x["delta"] for x in ep1["log"]])
    assert np.abs(dd).max() > 0 and np.abs(dd).max() <= kt.CAP + 1e-6
    j = dict(eid="k40b-eval|TreeB|s0|t0|toy", policy="TreeB", seed=0,
             scene=dict(task=0, case_id="toy", pos=3,
                        hash=[ep1["init_hash"], ep1["init_hash_full"]]))
    g_ = globals()
    keep = g_["SHARDS"]
    with tempfile.TemporaryDirectory() as td:
        g_["SHARDS"] = Path(td)
        try:
            save_ep(shard_path(j["eid"]), j, ep1, "P")
            m, pr = check_shard(j, "P", C)
            assert pr == [], pr
            with np.load(shard_path(j["eid"])) as z:
                zz = dict(z)
            meta = json.loads(str(zz["meta"]))
            for k_, f in (("h_final", lambda x: x + 0.01),
                          ("h_delta", lambda x: x * 2),
                          ("actions", lambda x: x + 1)):
                bad = dict(zz)
                bad[k_] = f(zz[k_].copy())
                assert ep_problems(bad, meta, j, C), k_
            bad = dict(zz)
            bad["h_final"] = zz["h_final"].copy()
            bad["h_final"][..., 6] *= -1
            assert ep_problems(bad, meta, j, C)
            assert ep_problems(zz, dict(meta, heavy=meta["heavy"] + 1), j, C)
            assert check_shard(dict(j, policy="FlatB"), "P", C)[1]
        finally:
            g_["SHARDS"] = keep


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
