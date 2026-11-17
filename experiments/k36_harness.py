#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-36: исполнение B1 (R / T32 / P32), эпизодные транзакции, проверки.

Эпизод — неизменённый k31_collect.run_episode (seed 101, 10 шагов ожидания,
защёлка, до 600 команд) с k32_policy.Arm (детерминированно: argmax
маршрута, η = μ, шум выключен; схват — reference). Среда — k36_bank.BankSim
(bank / legacy) или старый путь k18_collector.Sim (legacy-check).
Транзакция эпизода: reserve(610) -> исполнение -> атомарный файл
(json + npz: журнал вызовов K-32, действия, при необходимости контекст
первого вызова) -> commit -> проверка; записанный эпизод переживает
перезапуск (settle, без повтора). Отвергнутый завершённый эпизод —
карантин и маркер reports/k36/rejected (новая попытка — только по решению).
Проверка эпизода (CPU): метка/стадия/scope/eid/prepare/контракт K-32/actor
hash/источник; npz SHA; начальные хеши = банку (или холодному эталону K-32
для legacy init 30); действия = SHA; журнал — k32_harness.journal_problems
(формы и конечность, схват = знак логита reference, рука reference =
tanh(Z0), r/η = детерминированному правилу, η = 0 -> рука reference,
D/Du/рука пересобраны побитово, исполненное = масштабу команд, k и
elapsed последовательны, число шагов = done/600).

Команды: real-check DEV, legacy-check DEV, bank-smoke DEV, cold-repeat DEV,
confirm-a DEV, status. До confirm-a — четыре свежих технических гейта.
Промежуточные SR этапов не печатаются (§17): только число валидных
эпизодов, время и расход.
"""
import argparse
import gc
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

import k31_collect as kcol  # noqa: E402
import k36_bank as kb36  # noqa: E402
import k36_protocol as pc  # noqa: E402

OUT = pc.OUT
REJECTED = os.path.join(OUT, "rejected")
STAGE_SCOPE = dict(legacy="checks", smoke="checks", repeat="checks",
                   confirm_a="confirm_a")
GATES = dict(real_check=os.path.join(OUT, "real_check.json"),
             legacy=os.path.join(OUT, "legacy_check.json"),
             smoke=os.path.join(OUT, "bank_smoke.json"),
             repeat=os.path.join(OUT, "cold_repeat.json"))
LEGACY_TASKS, LEGACY_INIT = (0, 8), 30
SMOKE_TASKS, SMOKE_N = kb36.ADAPTED, 2
SUMMARY = ("success", "success_step", "done_step", "own_steps", "steps",
           "physical", "init_hash", "init_hash_full", "actions_sha1")
FIRST_KEYS = ("Z0", "path", "c", "r", "eta", "Du", "a_codec", "zero")
ACTOR_PARAMS, CRITIC_PARAMS = 203_271, 607_489      # при H = 768
Z_TOL, ARM_TOL, DU_TOL = 1e-5, 1e-6, 1e-5


def sha(p):
    return pc.sha(p)


# --- пути и записи ---------------------------------------------------------------------
def ep_paths(stage, lab, task, key):
    d = os.path.join(OUT, stage, lab)
    return (os.path.join(d, f"t{task}_{key}.json"),
            os.path.join(d, f"t{task}_{key}.npz"))


def eid_of(stage, lab, task, key):
    return f"{STAGE_SCOPE[stage]}/{stage}/{lab}/t{task}/{key}"


def reject_path(eid):
    return os.path.join(REJECTED, eid.replace("/", "__") + ".json")


def episode_number(eid):
    return int(hashlib.sha1(eid.encode()).hexdigest()[:8], 16)


IDENT = ("eid", "scope", "stage", "label", "kind", "seed", "task", "key",
         "reset_id", "sim_mode", "space", "case_id", "prepare_sha1",
         "contract", "actor_hash", "source_weights_hash")


def want_of(prep, stage, lab, task, key, reset_id, sim_mode, space=None,
            case_id=None, want_hash=None, actor_hash=None):
    name, kind, s = pc.parse_arm(lab)
    eid = eid_of(stage, lab, task, key)
    return dict(eid=eid, scope=STAGE_SCOPE[stage], stage=stage, label=lab,
                kind=kind, seed=s, task=int(task), key=str(key),
                reset_id=int(reset_id), sim_mode=sim_mode,
                space=None if space is None else int(space), case_id=case_id,
                prepare_sha1=prep["sha1"],
                contract=prep["k32"]["k32_contract"],
                actor_hash=actor_hash if actor_hash is not None else
                (None if kind == "R" else
                 prep["k32"]["seeds"][str(s)][kind]["actor_hash"]),
                source_weights_hash=prep["k32"]["seeds"][str(s)]["source"][
                    "weights_hash"],
                want_hash=None if want_hash is None else list(want_hash))


def save_ep(want, ep, first=None):
    import k32_harness as kh
    import k32_operator as ko
    jp, np_ = ep_paths(want["stage"], want["label"], want["task"],
                       want["key"])
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    d = ko.DIM.get(want["kind"], 6)
    J = kh.pack_journal([(0, ep["calls"])], d)
    arr = dict(actions=ep["actions"].astype(np.float32),
               **{f"j_{k}": v for k, v in J.items()})
    if first is not None:
        F, rec = first
        arr.update({f"F_{k}": v.numpy() for k, v in F.items()})
        arr.update({f"f_{k}": np.asarray(rec[k]) for k in FIRST_KEYS
                    if k in rec})
        arr["f_has_zhat"] = np.array(rec.get("Zhat") is not None)
        arr["f_Zhat"] = np.asarray(rec["Zhat"] if rec.get("Zhat") is not None
                                   else np.zeros((20, 6), np.float32))
    with open(np_ + ".tmp", "wb") as fh:
        np.savez_compressed(fh, **arr)
    os.replace(np_ + ".tmp", np_)
    rec = dict({k: want[k] for k in IDENT}, want_hash=want["want_hash"],
               npz_sha1=sha(np_), phys=int(ep["physical"]),
               seconds=float(ep["seconds"]),
               summary={k: ep[k] for k in SUMMARY})
    pc.save_json(jp, rec)
    return rec


def load_ep(want):
    jp, np_ = ep_paths(want["stage"], want["label"], want["task"],
                       want["key"])
    if not (os.path.exists(jp) and os.path.exists(np_)):
        return None
    rec = json.load(open(jp, encoding="utf-8"))
    bad = [k for k in IDENT if rec.get(k) != want[k]]
    if bad or rec.get("want_hash") != want["want_hash"]:
        raise SystemExit(f"{jp}: сохранённый эпизод чужой ({bad})")
    if sha(np_) != rec.get("npz_sha1"):
        raise SystemExit(f"{jp}: npz изменён")
    return rec, dict(np.load(np_))


def consts(prep):
    """half/mid/L/U/q закреплённого плана K-32 (CPU)."""
    import torch
    import k31_readout as kr
    b = prep["k32_plan"]
    L = np.asarray(b["bounds"]["L"])[:6]
    U = np.asarray(b["bounds"]["U"])[:6]
    h, m = [v.numpy() for v in kr.build(torch)["half_mid"](L, U, "cpu")]
    return dict(half=h, mid=m, L=L, U=U, q=np.asarray(b["scale_q"]))


def episode_problems(rec, z, want, C):
    import k32_harness as kh
    p = []
    bad = [k for k in IDENT if rec.get(k) != want[k]]
    if bad:
        return [f"идентификаторы {bad}"]
    s = rec["summary"]
    wh = want["want_hash"]
    if wh is not None:
        if s["init_hash"] != wh[0] or (wh[1] is not None and
                                       s["init_hash_full"] != wh[1]):
            p.append("начальный хеш ≠ банку/эталону")
    A = z.get("actions")
    if A is None or kcol.actions_sha(A) != s["actions_sha1"]:
        return p + ["действия ≠ SHA"]
    J = {k[2:]: v for k, v in z.items() if k.startswith("j_")}
    if not J or len(J.get("ep", [])) == 0:
        return p + ["пустой журнал"]
    if (J["ep"] != 0).any():
        return p + ["журнал: чужие эпизоды"]
    p += kh.journal_problems(J, [A], [s["done_step"]], C["half"], C["mid"],
                             C["L"], C["U"], C["q"], want["kind"])
    if int(rec["phys"]) != int(s["physical"]) or int(s["physical"]) != \
            kcol.ENV["waiting_steps"] + int(s["steps"]):
        p.append("расход ≠ шагам")
    return p


def quarantine(want, reasons):
    jp, np_ = ep_paths(want["stage"], want["label"], want["task"],
                       want["key"])
    q = os.path.join(OUT, "quarantine", time.strftime("%Y%m%dT%H%M%S"))
    os.makedirs(q, exist_ok=True)
    for f in (jp, np_):
        if os.path.exists(f):
            shutil.move(f, os.path.join(q, want["label"] + "_" +
                                        os.path.basename(f)))
    pc.save_json(reject_path(want["eid"]), dict(eid=want["eid"],
                                                reasons=reasons[:20],
                                                quarantine=q))
    return q


def ledger_owner_problems(L, eid, phys, scope):
    dn = L.d["done"].get(eid)
    pd = L.d["pending"] or {}
    if dn is not None:
        return [] if dn["scope"] == scope and int(dn["actual"]) == int(
            phys) else [f"{eid}: учтён иначе ({dn})"]
    if pd.get("id") == eid and pd.get("scope") == scope:
        return []
    return [f"{eid}: результат не учтён в бюджете"]


def run_ep(want, sim, io, arm, L, C, keep_first=False, crash=None):
    """-> (rec, z). Транзакция §14; проверка после commit."""
    if os.path.exists(reject_path(want["eid"])):
        raise SystemExit(f"{want['eid']}: был отвергнут — технический стоп")
    got = load_ep(want)
    if got is None:
        if L.is_done(want["eid"]):
            raise SystemExit(f"{want['eid']}: учтён, но результата нет")
        L.reserve(want["scope"], want["eid"], amount=pc.EP)
        arm.episode = episode_number(want["eid"])
        arm.keep_first = bool(keep_first)
        ep = kcol.run_episode(sim, io, arm, want["task"], want["reset_id"])
        first = None
        if keep_first:
            f0 = ep["calls"][0]["rec"]
            first = (arm.first_F, f0)
        save_ep(want, ep, first)
        if crash == "after_write":
            raise KeyboardInterrupt("после записи эпизода")
        L.commit(want["eid"], ep["physical"])
        got = load_ep(want)
    rec, z = got
    pr = ledger_owner_problems(L, want["eid"], rec["phys"], want["scope"])
    if pr:
        raise SystemExit(pr[0])
    L.settle(want["eid"], rec["phys"])
    pr = episode_problems(rec, z, want, C)
    if pr:
        q = quarantine(want, pr)
        raise SystemExit(f"{want['eid']}: эпизод отвергнут ({pr[:3]}) -> "
                         f"карантин {q}")
    return rec, z


# --- источники и руки -------------------------------------------------------------------
class Sources:
    """Один источник K-25 на GPU за раз (§12)."""

    def __init__(self, torch, device, loader=None, io_of=None):
        self.torch, self.device = torch, device
        self.loader, self.io_of = loader, io_of
        self.seed = None
        self.v = None
        self.arms = {}

    def get(self, seed):
        if self.seed != seed:
            self.release()
            if self.loader is None:
                import k29_policy as kp
                import k31_harness as kh31
                import k32_protocol as pc32
                mod, bar, M, dec = pc32.load_source_dec(
                    self.torch, pc32.load_base(), seed, self.device)
                io = kh31.IO(kp.Inputs(self.torch, bar["proc"],
                                       self.device))
            else:
                mod, bar, M, dec, io = self.loader(seed)
            self.v = dict(mod=mod, bar=bar, M=M, dec=dec, io=io)
            self.seed = seed
        return self.v

    def release(self):
        self.v, self.seed, self.arms = None, None, {}
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def load_actor(torch, prep, kind, seed, H, path=None, want_hash=None):
    import k32_model as km32
    import k32_operator as ko
    import k32_ppo as kppo
    import k32_protocol as pc32
    key = f"k32_{kind}_final_s{seed}"
    path = path or prep["placement"][key]["path"]
    if path == prep["placement"].get(key, {}).get("path") and \
            sha(path) != prep["placement"][key]["sha1"]:
        raise SystemExit(f"{path}: final.pt ≠ prepare")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    actor, critic = km32.build(torch, H, ko.DIM[kind], seed,
                               pc32.load_norm(seed))
    actor.load_state_dict(ck["actor"])
    want = want_hash or prep["k32"]["seeds"][str(seed)][kind]["actor_hash"]
    if kppo.tensor_hash(actor.state_dict()) != want:
        raise SystemExit(f"{kind} s{seed}: actor ≠ prepare")
    actor.eval()
    return actor, critic


def make_arm(torch, prep, lab, src, actor=None):
    import k32_policy as kpol
    _name, kind, s = pc.parse_arm(lab)
    if kind == "R":
        return kpol.Arm(torch, src["mod"], src["dec"], src["M"], kind="R",
                        seed=s)
    if actor is None:
        actor, _c = load_actor(torch, prep, kind, s, src["mod"].H)
    return kpol.Arm(torch, src["mod"], src["dec"], src["M"], kind=kind,
                    seed=s, actor=actor)


# --- гейты --------------------------------------------------------------------------------
def gate_problems(kind, prep):
    path = GATES[kind]
    if not os.path.exists(path):
        return [f"нет {path}"]
    g = json.load(open(path, encoding="utf-8"))
    p = []
    if g.get("ok") is not True or g.get("problems") != []:
        p.append(f"{kind}: не пройден")
    if g.get("prepare_sha1") != prep["sha1"]:
        p.append(f"{kind}: другой prepare")
    for f, s in (g.get("files") or {}).items():
        if not os.path.exists(f) or sha(f) != s:
            p.append(f"{kind}: файл {f} изменён или отсутствует")
            break
    if kind != "real_check" and not g.get("files"):
        p.append(f"{kind}: нет закреплённых файлов эпизодов")
    return p


def ep_files(want):
    jp, np_ = ep_paths(want["stage"], want["label"], want["task"],
                       want["key"])
    return {jp: sha(jp), np_: sha(np_)}


def checks_problems(prep):
    p = []
    for k in GATES:
        p += gate_problems(k, prep)
    return p


# --- legacy-check -----------------------------------------------------------------------
def labels_of(seed):
    return [f"R_s{seed}", f"T32_s{seed}", f"P32_s{seed}"]


def compare_eps(a, b):
    """Пара эпизодов (rec, z): хеши, действия, исходы, журнал — побитово."""
    p = []
    sa, sb = a[0]["summary"], b[0]["summary"]
    for k in SUMMARY:
        if sa[k] != sb[k]:
            p.append(k)
    za, zb = a[1], b[1]
    for k in sorted(set(za) | set(zb)):
        if not (k.startswith("j_") or k == "actions"):
            continue
        if k not in za or k not in zb or not np.array_equal(za[k], zb[k]):
            p.append(k)
    return p


def legacy_check(a, sims=None, src=None):
    import k18_collector as k18c
    sims = sims or dict(old=k18c.Sim(1, "persistent"),
                        new=kb36.BankSim("legacy"))
    import torch
    try:
        prep = pc.load_prepare()
        C = consts(prep)
        L = pc.ledger()
        src = src or Sources(torch, a.device)
        files, prob, pairs = {}, [], {}
        for s in pc.parse_seeds():
            v = src.get(s)
            for lab in labels_of(s):
                arm = make_arm(torch, prep, lab, v)
                for t in LEGACY_TASKS:
                    wh = (prep["k32_plan"]["init_ref"][f"{t}:{LEGACY_INIT}"],
                          None)
                    res = {}
                    for path in ("old", "new"):
                        w = want_of(prep, "legacy", lab, t,
                                    f"i{LEGACY_INIT}_{path}", LEGACY_INIT,
                                    "k18" if path == "old" else "legacy",
                                    want_hash=wh)
                        res[path] = run_ep(w, sims[path], v["io"], arm, L, C)
                        files.update(ep_files(w))
                    d = compare_eps(res["old"], res["new"])
                    pairs[f"{lab}/t{t}"] = d
                    if d:
                        prob.append(f"{lab} t{t}: старый ≠ новый путь {d[:4]}")
        src.release()
        rep = dict(ok=not prob, problems=prob, pairs=pairs, files=files,
                   prepare_sha1=prep["sha1"], n_episodes=len(files) // 2,
                   spent_checks=L.spent("checks"))
        pc.save_json(GATES["legacy"], rep)
        print(f"  legacy-check: эпизодов {rep['n_episodes']}, пар "
              f"{len(pairs)}; " + ("старый = новый путь побитово (хеши, "
                                   "действия, исходы, журнал)" if not prob
                                   else "ОТКАЗ: " + "; ".join(prob[:3])))
        return 0 if not prob else 3
    finally:
        for s_ in sims.values():
            s_.close()


# --- bank-smoke / cold-repeat ---------------------------------------------------------------
def rebuild_problems(torch, prep, kind, seed, F, first):
    """Независимая CPU FP32-пересборка первого вызова из закреплённых весов
    (допуски K-32)."""
    import k31_protocol as p31
    import k32_protocol as pc32
    b = pc32.load_base()
    if kind != "R":
        return pc32.rebuild_problems(torch, [(F, first)], b, kind, seed)
    dcpu = p31.cpu_decoder(torch, b["sources"][str(seed)])
    with torch.no_grad():
        ref = dcpu(F)
    z = p31._rel(ref["Z"][0].numpy(), first["Z0"])
    arm = ref["arm"][0, :8].float().numpy()
    e = float(np.abs(arm - np.asarray(first["a_codec"])[:, :6]).max())
    p = []
    if not z <= Z_TOL:
        p.append(f"пересборка Z0 R: {z:.2e}")
    if not e <= ARM_TOL:
        p.append(f"пересборка руки R: {e:.2e}")
    return p, dict(Z0=z, arm=e)


def first_of(z):
    F = {k[2:]: v for k, v in z.items() if k.startswith("F_")}
    first = {k: z[f"f_{k}"] for k in FIRST_KEYS if f"f_{k}" in z}
    first["Zhat"] = z["f_Zhat"] if bool(z.get("f_has_zhat", False)) else None
    if "r" in first:
        first["r"] = int(first["r"])
    if "zero" in first:
        first["zero"] = bool(first["zero"])
    return F, first


def smoke_wants(prep, stage, labels):
    out = []
    for t in SMOKE_TASKS:
        for cid, pos, ih, ihf in kb36.dev_cases(t, SMOKE_N):
            for lab in labels:
                out.append(want_of(prep, stage, lab, t, cid, pos, "bank",
                                   space=kb36.SPACE_DEV, case_id=cid,
                                   want_hash=(ih, ihf)))
    return out


def bank_smoke(a, sim=None, src=None):
    sim = sim or kb36.BankSim("bank", kb36.SPACE_DEV)
    import torch
    try:
        prep = pc.load_prepare()
        p0 = gate_problems("legacy", prep)
        if p0:
            raise SystemExit("bank-smoke без legacy-check: " + "; ".join(p0))
        C = consts(prep)
        L = pc.ledger()
        src = src or Sources(torch, a.device)
        files, prob, rb, secs = {}, [], {}, {}
        for s in pc.parse_seeds():
            v = src.get(s)
            for w in smoke_wants(prep, "smoke", labels_of(s)):
                arm = make_arm(torch, prep, w["label"], v)
                rec, z = run_ep(w, sim, v["io"], arm, L, C, keep_first=True)
                files.update(ep_files(w))
                F, first = first_of(z)
                if not F or "a_codec" not in first:
                    prob.append(f"{w['eid']}: нет контекста первого вызова")
                    continue
                Ft = {k: torch.as_tensor(x) for k, x in F.items()}
                pr, st = rebuild_problems(torch, prep, w["kind"], s, Ft,
                                          first)
                if not np.array_equal(first["a_codec"], z["j_a_codec"][0]):
                    pr.append("снимок не первого вызова журнала")
                rb[w["eid"]] = st
                prob += [f"{w['eid']}: {x}" for x in pr]
                secs.setdefault(str(w["task"]), []).append(
                    (rec["seconds"], rec["phys"]))
        src.release()
        fc = {}
        for t, v in secs.items():
            sec = np.array([x[0] for x in v])
            ph = np.array([x[1] for x in v])
            fc[t] = dict(median_s_per_ep=float(np.median(sec)),
                         p90_s_per_ep=float(np.percentile(sec, 90)),
                         s_per_step=float(sec.sum() / ph.sum()))
        per_ep = np.median([x[0] for v in secs.values() for x in v])
        rep = dict(ok=not prob, problems=prob, rebuild=rb, files=files,
                   timing=fc, forecast_A_hours=float(per_ep * 1800 / 3600),
                   prepare_sha1=prep["sha1"], n_episodes=len(files) // 2,
                   spent_checks=L.spent("checks"))
        pc.save_json(GATES["smoke"], rep)
        print(f"  bank-smoke: эпизодов {rep['n_episodes']}; начальные хеши = "
              f"банку; пересборка первого вызова "
              f"{'пройдена' if not prob else 'ОТКАЗ'}; медиана "
              f"{per_ep:.0f} с/эпизод, прогноз A ≈ "
              f"{rep['forecast_A_hours']:.0f} ч (по dev-сценам); "
              + ("пройден" if not prob else "; ".join(prob[:3])))
        return 0 if not prob else 3
    finally:
        sim.close()


def cold_repeat(a, sim=None, src=None):
    sim = sim or kb36.BankSim("bank", kb36.SPACE_DEV)
    import torch
    try:
        prep = pc.load_prepare()
        p0 = gate_problems("smoke", prep)
        if p0:
            raise SystemExit("cold-repeat без bank-smoke: " + "; ".join(p0))
        C = consts(prep)
        L = pc.ledger()
        src = src or Sources(torch, a.device)
        files, prob = {}, []
        for s in pc.parse_seeds():
            v = src.get(s)
            for lab in (f"T32_s{s}", f"P32_s{s}"):
                arm = make_arm(torch, prep, lab, v)
                for t in SMOKE_TASKS:
                    cid, pos, ih, ihf = kb36.dev_cases(t, 1)[0]
                    w = want_of(prep, "repeat", lab, t, cid, pos, "bank",
                                space=kb36.SPACE_DEV, case_id=cid,
                                want_hash=(ih, ihf))
                    r1 = run_ep(w, sim, v["io"], arm, L, C)
                    files.update(ep_files(w))
                    w0 = want_of(prep, "smoke", lab, t, cid, pos, "bank",
                                 space=kb36.SPACE_DEV, case_id=cid,
                                 want_hash=(ih, ihf))
                    r0 = load_ep(w0)
                    if r0 is None:
                        prob.append(f"{lab} t{t}: нет эпизода bank-smoke")
                        continue
                    z0 = {k: x for k, x in r0[1].items()
                          if not k.startswith(("F_", "f_"))}
                    d = compare_eps((r0[0], z0), r1)
                    if d:
                        prob.append(f"{lab} t{t}: повтор ≠ smoke {d[:4]}")
        src.release()
        rep = dict(ok=not prob, problems=prob, files=files,
                   prepare_sha1=prep["sha1"], n_episodes=len(files) // 2,
                   spent_checks=L.spent("checks"))
        pc.save_json(GATES["repeat"], rep)
        print(f"  cold-repeat: эпизодов {rep['n_episodes']}; " +
              ("хеши, действия, исходы и журнал = bank-smoke побитово"
               if not prob else "ОТКАЗ: " + "; ".join(prob[:3])))
        return 0 if not prob else 3
    finally:
        sim.close()


# --- real-check -------------------------------------------------------------------------
def real_check(a):
    """Сохранённые входы K-21 (без физических шагов): новый путь K-36 =
    путь K-32; μ/log π по кешу при minibatch 8/32/64; источник заморожен;
    число параметров; actor state.pt (начало продолжения) = T32."""
    import torch
    import k20_data as kd20
    import k21_data as d21
    import k32_dist as kd
    import k32_model as km32
    import k32_policy as kpol
    import k32_ppo as kppo
    import k32_protocol as pc32
    prep = pc.load_prepare()
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    rows = [int(r) for r in np.asarray(D["val_rows"])[:5]]
    rep, prob = dict(seeds={}), []
    torch.cuda.reset_peak_memory_stats(dev)
    rep["mem_before_gb"] = torch.cuda.memory_allocated(dev) / 2 ** 30
    for s in pc.parse_seeds():
        r = {}
        mod, bar, M, dec = pc32.load_source_dec(torch, pc32.load_base(), s,
                                                a.device)
        vers = {n: p_._version for n, p_ in mod.named_parameters()}
        ins = []
        for row in rows:
            batch = kd20.prompt_batch(bar, [np.array(D["IMG"][row],
                                                     copy=True)],
                                      kd20.norm_states(np.asarray(
                                          D["state8"][[row]])),
                                      [D["task"][row]])
            ins.append((batch, int(D["pos_offset"][row]),
                        np.asarray(D["state8"][[row]], np.float64)))
        for kind in ("R", "T", "P"):
            plan32 = pc32.load_eval_plan_kind(kind, s)
            old, _d = pc32.make_eval_arm(torch, plan32, kind, s, mod, dec, M)
            new = make_arm(torch, prep, f"{dict(R='R', T='T32', P='P32')[kind]}"
                                        f"_s{s}", dict(mod=mod, dec=dec, M=M))
            worst = []
            for i, (bt, po, s8) in enumerate(ins):
                for el in (0, 8, 16):
                    old.begin(0, i, 0)
                    new.begin(0, i, 0)
                    ao, ro = old.decide(bt, po, s8, el)
                    an, rn = new.decide(bt, po, s8, el)
                    same = np.array_equal(ao, an)
                    for k in ("Z0", "ref_arm", "grip_logit", "logits", "mu",
                              "r", "eta", "D", "Du", "path", "c", "S"):
                        if k in ro or k in rn:
                            same &= k in ro and k in rn and np.array_equal(
                                np.asarray(ro[k]), np.asarray(rn[k]))
                    worst.append(bool(same))
            r[f"new_eq_old_{kind}"] = all(worst)
            if not all(worst):
                prob.append(f"s{s} {kind}: новый путь ≠ K-32")
            if kind == "R":
                continue
            # μ / log π по кешу (стохастический сбор того же actor)
            actor, critic = load_actor(torch, prep, kind, s, mod.H)
            r[f"params_{kind}"] = dict(actor=km32.n_params(actor),
                                       critic=km32.n_params(critic))
            if mod.H == 768 and (r[f"params_{kind}"]["actor"] != ACTOR_PARAMS
                                 or r[f"params_{kind}"]["critic"] !=
                                 CRITIC_PARAMS):
                prob.append(f"s{s} {kind}: число параметров ≠ паспорту")
            sto = kpol.Arm(torch, mod, dec, M, kind=kind, seed=s, actor=actor,
                           stochastic=True, space=36)
            recs = []
            for i, (bt, po, s8) in enumerate(ins):
                sto.begin(0, i, 900 + i)
                for el in (0, 8, 16):
                    recs.append(sto.decide(bt, po, s8, el)[1])
            S = torch.as_tensor(np.stack([q["S"] for q in recs]))
            X = km32.actor_inputs(torch, S, mod.H)
            mus = {}
            for mb in (8, 32, 64):
                lg, mu = kppo.policy_all(torch, actor, X, mb)
                lp = kd.log_prob(torch, lg.double(), mu.double(),
                                 torch.as_tensor([q["r"] for q in recs]),
                                 torch.as_tensor(np.stack([q["eta"] for q in
                                                           recs])).double())
                mus[mb] = dict(
                    mu=float((mu - torch.as_tensor(np.stack(
                        [q["mu"] for q in recs]))).abs().max()),
                    logp=float((lp - torch.as_tensor(
                        [q["logpi"] for q in recs])).abs().max()))
            r[f"cache_{kind}"] = mus
            if not all(v["mu"] <= 1e-6 and v["logp"] <= 1e-4
                       for v in mus.values()):
                prob.append(f"s{s} {kind}: μ/log π кеша > допуска {mus}")
            if kind == "T":
                st = torch.load(prep["placement"][f"k32_T_state_s{s}"]["path"],
                                map_location="cpu", weights_only=False)
                a2, _c2 = km32.build(torch, mod.H, 6, s, pc32.load_norm(s))
                a2.load_state_dict(st["actor"])
                a2.eval()
                cont = make_arm(torch, prep, f"T32_s{s}",
                                dict(mod=mod, dec=dec, M=M), actor=a2)
                eq = []
                for i, (bt, po, s8) in enumerate(ins):
                    new.begin(0, i, 0)
                    cont.begin(0, i, 0)
                    eq.append(np.array_equal(new.decide(bt, po, s8, 0)[0],
                                             cont.decide(bt, po, s8, 0)[0]))
                r["tplus_start_eq_T32"] = all(eq) and kppo.tensor_hash(
                    a2.state_dict()) == prep["k32"]["seeds"][str(s)]["T"][
                    "actor_hash"]
                if not r["tplus_start_eq_T32"]:
                    prob.append(f"s{s}: actor state.pt ≠ T32")
                del st, a2, cont
            del actor, critic, sto
        r["source_frozen"] = all(not p_.requires_grad for p_ in
                                 mod.parameters()) and all(
            p_._version == vers[n] for n, p_ in mod.named_parameters())
        if not r["source_frozen"]:
            prob.append(f"s{s}: источник изменён/с градиентами")
        r["peak_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        del mod, bar, M, dec, old, new, ins
        gc.collect()
        torch.cuda.empty_cache()
        r["mem_after_free_gb"] = torch.cuda.memory_allocated(dev) / 2 ** 30
        rep["seeds"][str(s)] = r
        print(f"  real-check s{s}: новый = K-32 R/T/P "
              f"{r['new_eq_old_R']}/{r['new_eq_old_T']}/{r['new_eq_old_P']};"
              f" кеш μ/log π T {r['cache_T']}; параметры {r['params_T']}; "
              f"начало продолжения = T32 {r['tplus_start_eq_T32']}; "
              f"источник заморожен {r['source_frozen']}; пик "
              f"{r['peak_gb']:.1f} ГБ, после {r['mem_after_free_gb']:.2f}",
              flush=True)
    rep.update(ok=not prob, problems=prob, prepare_sha1=prep["sha1"],
               files={}, rows=rows,
               device=torch.cuda.get_device_name(dev))
    pc.save_json(GATES["real_check"], rep)
    print("  real-check K-36 " + ("пройден" if not prob else
                                  "НЕ пройден: " + "; ".join(prob[:3])))
    return 0 if not prob else 3


# --- этап A -------------------------------------------------------------------------------
def block_path(lab, task, g):
    return os.path.join(OUT, "confirm_a", lab, f"t{task}_b{g:02d}.json")


def a_want(prep, lab, task, cid):
    sp = prep["splits"][str(task)]
    return want_of(prep, "confirm_a", lab, task, cid, sp["pos"][cid], "bank",
                   space=kb36.SPACE_CONF, case_id=cid,
                   want_hash=tuple(sp["init_hash"][cid]))


def block_problems(prep, lab, task, g, cases, C):
    bp = block_path(lab, task, g)
    if not os.path.exists(bp):
        return ["нет блока"]
    d = json.load(open(bp, encoding="utf-8"))
    if d.get("cases") != cases or d.get("prepare_sha1") != prep["sha1"]:
        return ["блок другого плана"]
    p = []
    for cid in cases:
        w = a_want(prep, lab, task, cid)
        got = load_ep(w)
        if got is None:
            p.append(f"{cid}: нет эпизода")
            continue
        jp, np_ = ep_paths("confirm_a", lab, task, cid)
        if d["files"].get(jp) != sha(jp) or d["files"].get(np_) != sha(np_):
            p.append(f"{cid}: файл изменён после блока")
        p += [f"{cid}: {x}" for x in episode_problems(*got, w, C)]
    return p


def confirm_a(a, sim=None, src=None, cells=None):
    sim = sim or kb36.BankSim("bank", kb36.SPACE_CONF)
    import torch
    try:
        prep = pc.load_prepare()
        p0 = checks_problems(prep)
        if p0:
            raise SystemExit("этап A без технических гейтов: " +
                             "; ".join(p0[:4]))
        cells = cells or pc.cells_a(prep["splits"])
        if hashlib.sha1(json.dumps(cells).encode()).hexdigest() != \
                prep["cells_A_sha1"] and not getattr(a, "toy", False):
            raise SystemExit("очередь A ≠ закреплённой в prepare")
        C = consts(prep)
        L = pc.ledger()
        src = src or Sources(torch, a.device)
        t0, done0, n_new = time.time(), 0, 0
        for i, (lab, t, g, cases) in enumerate(cells):
            if os.path.exists(block_path(lab, t, g)):
                pr = block_problems(prep, lab, t, g, cases, C)
                if pr:
                    raise SystemExit(f"{lab} t{t} b{g}: готовый блок не "
                                     f"прошёл проверку: {pr[:2]}")
                done0 += 1
                continue
            _n, _k, s = pc.parse_arm(lab)
            v = src.get(s)
            if lab not in src.arms:
                src.arms[lab] = make_arm(torch, prep, lab, v)
            arm = src.arms[lab]
            files = {}
            for cid in cases:
                w = a_want(prep, lab, t, cid)
                run_ep(w, sim, v["io"], arm, L, C)
                files.update(ep_files(w))
            pc.save_json(block_path(lab, t, g), dict(
                label=lab, task=t, block=g, cases=cases, files=files,
                prepare_sha1=prep["sha1"]))
            n_new += 1
            el = (time.time() - t0) / 3600
            left = len(cells) - i - 1
            print(f"  A: блок {i + 1}/{len(cells)} ({lab} t{t} b{g}) "
                  f"готов; новых {n_new} за {el:.1f} ч, осталось {left} "
                  f"(~{el / n_new * left:.1f} ч); confirm_a "
                  f"{L.spent('confirm_a')}/{pc.CAPS['confirm_a']}",
                  flush=True)
        src.release()
        print(f"  этап A: все {len(cells)} блоков готовы и проверены "
              f"(ранее готовых {done0})")
        return 0
    finally:
        sim.close()


def status(a):
    prep = pc.load_prepare()
    L = pc.ledger()
    cells = pc.cells_a(prep["splits"])
    nb = sum(os.path.exists(block_path(l_, t, g)) for l_, t, g, _c in cells)
    print("  гейты: " + ", ".join(
        f"{k} {'ok' if not gate_problems(k, prep) else 'нет'}"
        for k in GATES))
    print(f"  этап A: блоков {nb}/{len(cells)}; расход " + ", ".join(
        f"{k} {L.spent(k)}/{v}" for k, v in pc.CAPS.items() if L.spent(k)))
    print(f"  аварийных списаний: {len(L.d['crashes'])}; отвергнутых: "
          f"{len(os.listdir(REJECTED)) if os.path.isdir(REJECTED) else 0}")
    return 0


# --- самопроверка -------------------------------------------------------------------------
def selftest():
    import tempfile
    import types
    import torch
    import k21_model as km
    import k25_model as k25m
    import k29_model as k29m
    import k31_readout as kr
    import k32_model as km32
    import k32_ppo as kppo
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
    h, m = [v.numpy() for v in kr.build(torch)["half_mid"](dec._L, dec._U,
                                                         "cpu")]
    actor, _c = km32.build(torch, H, 6, 0, norm)
    with torch.no_grad():
        g2 = torch.Generator().manual_seed(7)
        for p_ in actor.out.parameters():
            p_.copy_(torch.randn(p_.shape, generator=g2))
    ah = kppo.tensor_hash(actor.state_dict())
    prep = dict(sha1="PREP", k32=dict(k32_contract="C", seeds={"0": dict(
        source=dict(weights_hash="W"), T=dict(actor_hash=ah),
        P=dict(actor_hash=ah))}))
    C = dict(half=h, mid=m, L=Lb, U=Ub, q=io.max_act_q)
    import k32_policy as kpol
    g_ = globals()
    keep = {k: g_[k] for k in ("OUT", "REJECTED")}
    with tempfile.TemporaryDirectory() as td:
        try:
            g_.update(OUT=os.path.join(td, "r"),
                      REJECTED=os.path.join(td, "r", "rejected"))
            L = kcol.Ledger(os.path.join(td, "led.json"), pc.CAPS)
            arms = dict(R=kpol.Arm(torch, mod, dec, M, kind="R", seed=0),
                        T=kpol.Arm(torch, mod, dec, M, kind="T", seed=0,
                                   actor=actor),
                        P=kpol.Arm(torch, mod, dec, M, kind="P", seed=0,
                                   actor=actor))
            got = {}
            for kind, lab in (("R", "R_s0"), ("T", "T32_s0"),
                              ("P", "P32_s0")):
                w = want_of(prep, "confirm_a", lab, 0, "k33s1_t0_c00007", 7,
                            "bank", space=1, case_id="k33s1_t0_c00007")
                got[kind] = run_ep(w, kppo.ToySim(), io, arms[kind], L, C,
                                   keep_first=True)
                rec, z = got[kind]
                assert episode_problems(rec, z, w, C) == []
                # повтор: эпизод не исполняется второй раз, расход один раз
                sp0 = L.spent("confirm_a")
                assert run_ep(w, kppo.ToySim(), io, arms[kind], L, C)[0] \
                    == rec and L.spent("confirm_a") == sp0
            # подмены — отказ
            w = want_of(prep, "confirm_a", "T32_s0", 0, "k33s1_t0_c00007", 7,
                        "bank", space=1, case_id="k33s1_t0_c00007")
            rec, z = got["T"]
            for nm, f in (
                    ("рука", lambda zz: zz["j_a_codec"].__setitem__(
                        (0, 0, 0), zz["j_a_codec"][0, 0, 0] + 1e-5)),
                    ("η", lambda zz: zz["j_eta"].__setitem__(
                        (0, 0), zz["j_eta"][0, 0] + 1e-3)),
                    ("NaN", lambda zz: zz["j_mu"].__setitem__((0, 0, 0),
                                                              np.nan)),
                    ("действие", lambda zz: zz["actions"].__setitem__(
                        (1, 0), zz["actions"][1, 0] + 1e-6)),
                    ("последний вызов", lambda zz: [zz.__setitem__(
                        k, v[:-1]) for k, v in list(zz.items())
                        if k.startswith("j_")]),
                    ("пустой", lambda zz: [zz.__setitem__(k, v[:0]) for k, v
                                           in list(zz.items())
                                           if k.startswith("j_")])):
                zz = {k: np.array(v, copy=True) for k, v in z.items()}
                f(zz)
                assert episode_problems(rec, zz, w, C), nm
            for k, v in (("actor_hash", "x"), ("contract", "D"),
                         ("prepare_sha1", "Q"), ("case_id", "c2")):
                assert episode_problems(dict(rec, **{k: v}), z, w, C), k
            assert episode_problems(rec, z, dict(w, want_hash=["0" * 16,
                                                               None]), C)
            # первый вызов: контекст пересборки сохранён и = журналу
            F, first = first_of(got["T"][1])
            assert F and np.array_equal(first["a_codec"],
                                        got["T"][1]["j_a_codec"][0])
            # авария между записью и commit: settle без повтора и списания
            w2 = want_of(prep, "confirm_a", "T32_s0", 0, "k33s1_t0_c00011",
                         11, "bank", space=1, case_id="k33s1_t0_c00011")
            try:
                run_ep(w2, kppo.ToySim(), io, arms["T"], L, C,
                       crash="after_write")
                raise AssertionError("авария не сработала")
            except KeyboardInterrupt:
                pass
            L2 = kcol.Ledger(os.path.join(td, "led.json"), pc.CAPS)
            sp0 = L2.spent("confirm_a")
            r2 = run_ep(w2, kppo.ToySim(), io, arms["T"], L2, C)
            assert L2.spent("confirm_a") == sp0 + r2[0]["phys"] and \
                not L2.d["crashes"]
            # отвергнутый эпизод: карантин и маркер; повтор — стоп
            w3 = want_of(prep, "confirm_a", "P32_s0", 0, "k33s1_t0_c00013",
                         13, "bank", space=1, case_id="k33s1_t0_c00013",
                         want_hash=("f" * 16, None))
            try:
                run_ep(w3, kppo.ToySim(), io, arms["P"], L2, C)
                raise AssertionError("неверный хеш принят")
            except SystemExit:
                pass
            assert os.path.exists(reject_path(w3["eid"]))
            try:
                run_ep(w3, kppo.ToySim(), io, arms["P"], L2, C)
                raise AssertionError("повтор отвергнутого")
            except SystemExit:
                pass
            # чужой сохранённый эпизод — отказ, файлы не трогаются
            w4 = dict(w2, actor_hash="другой")
            try:
                run_ep(w4, kppo.ToySim(), io, arms["T"], L2, C)
                raise AssertionError("чужой эпизод принят")
            except SystemExit:
                pass
            # сравнение пары: побитово равные — без отличий; подмена — есть
            assert compare_eps(got["T"], got["T"]) == []
            zz = dict(got["T"][1], j_Du=got["T"][1]["j_Du"] + 1e-7)
            assert compare_eps(got["T"], (got["T"][0], zz))
            # гейт: подмена закреплённого файла — отказ; голый passed — отказ
            fp = os.path.join(td, "x.json")
            open(fp, "w").write("{}")
            gpath = os.path.join(td, "gate.json")
            keepg = dict(GATES)
            GATES["legacy"] = gpath
            try:
                pc.save_json(gpath, dict(ok=True, problems=[],
                                         prepare_sha1="PREP",
                                         files={fp: sha(fp)}))
                assert gate_problems("legacy", prep) == []
                open(fp, "w").write("{1}")
                assert gate_problems("legacy", prep)
                pc.save_json(gpath, dict(ok=True, problems=[],
                                         prepare_sha1="PREP"))
                assert gate_problems("legacy", prep)       # голый passed
                pc.save_json(gpath, dict(ok=True, problems=[],
                                         prepare_sha1="OLD",
                                         files={fp: sha(fp)}))
                assert gate_problems("legacy", prep)
            finally:
                GATES.update(keepg)
        finally:
            g_.update(keep)
    assert types
    print("самопроверка k36_harness пройдена: эпизоды R/T/P (ненулевой actor) "
          "проходят независимую проверку журнала; повтор не исполняет и не "
          "списывает второй раз; подмены руки/η/NaN/действия/последнего "
          "вызова/пустой журнал, чужие actor/контракт/prepare/случай, "
          "неверный начальный хеш — отказ; контекст первого вызова = "
          "журналу; авария между записью и commit — settle без списания; "
          "отвергнутый — карантин, маркер, повтор запрещён; чужой "
          "сохранённый эпизод — отказ; сравнение пар побитово; гейт — "
          "подмена файла, голый passed, другой prepare — отказ")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-36: исполнение")
    ap.add_argument("cmd", choices=("real-check", "legacy-check",
                                    "bank-smoke", "cold-repeat", "confirm-a",
                                    "status"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    fn = {"real-check": real_check, "legacy-check": legacy_check,
          "bank-smoke": bank_smoke, "cold-repeat": cold_repeat,
          "confirm-a": confirm_a, "status": status}[a.cmd]
    try:
        return fn(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
