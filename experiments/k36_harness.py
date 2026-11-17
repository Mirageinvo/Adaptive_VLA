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
SOURCE_H = 768
Z_TOL, ARM_TOL, DU_TOL = 1e-5, 1e-6, 1e-5
MIN_FREE_GB = 10.0


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
    p += summary_problems(rec, s, len(A))
    return p


def summary_problems(rec, s, n_actions):
    """Типы, конечность и согласованность сводки эпизода с длиной
    действий; без terminal эпизод обязан дойти до лимита 600."""
    p = []
    ints = ("success_step", "done_step", "own_steps", "steps", "physical")
    if not isinstance(s.get("success"), bool):
        return ["success не bool"]
    for k in ints:
        v = s.get(k)
        if isinstance(v, bool) or not isinstance(v, int):
            return [f"{k} не целое"]
    for k in ("init_hash", "init_hash_full", "actions_sha1"):
        if not isinstance(s.get(k), str) or not s[k]:
            return [f"{k} не строка"]
    if isinstance(rec.get("phys"), bool) or not isinstance(rec.get("phys"),
                                                            int):
        return ["phys не целое"]
    ss, ds, own, st = (s["success_step"], s["done_step"], s["own_steps"],
                       s["steps"])
    mx = kcol.ENV["max_steps"]
    if not 0 < st <= mx:
        p.append(f"steps {st} вне 1…{mx}")
    if n_actions != st:
        p.append(f"действий {n_actions} ≠ steps {st}")
    if ds >= 0:
        if not (ds + 1 == st == own):
            p.append("done_step/steps/own_steps несогласованы")
    else:
        if ds != -1 or own != st or st != mx:
            p.append("без terminal эпизод не дошёл до лимита")
    if s["success"]:
        if not 0 <= ss < st or (ds >= 0 and ss > ds):
            p.append("success_step несогласован с успехом")
    elif ss != -1:
        p.append("success_step при неуспехе")
    if rec["phys"] != s["physical"] or s["physical"] != \
            kcol.ENV["waiting_steps"] + st:
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
def free(torch):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_source(torch, seed, device, loader=None):
    """Один источник K-25 -> dict(mod, bar, M, dec, io). Вызывается только
    внутри функции обработки сида/куска очереди (§12): наружу — CPU."""
    if loader is not None:
        return loader(seed)
    import k29_policy as kp
    import k31_harness as kh31
    import k32_protocol as pc32
    mod, bar, M, dec = pc32.load_source_dec(torch, pc32.load_base(), seed,
                                            device)
    io = kh31.IO(kp.Inputs(torch, bar["proc"], device))
    return dict(mod=mod, bar=bar, M=M, dec=dec, io=io)


def need_free_gb(prep=None):
    """max(10 ГиБ, измеренный пик real-check + 2 ГиБ)."""
    peak = 0.0
    if os.path.exists(GATES["real_check"]):
        g = json.load(open(GATES["real_check"], encoding="utf-8"))
        peak = max([float((v or {}).get("peak_gb", 0.0))
                    for v in (g.get("seeds") or {}).values()] or [0.0])
    return max(MIN_FREE_GB, peak + 2.0)


def resources_problems(torch, device):
    dev = torch.device(device)
    if dev.type != "cuda":
        return []
    free_gb = torch.cuda.mem_get_info(dev)[0] / 2 ** 30
    need = need_free_gb()
    return [] if free_gb >= need else \
        [f"свободно GPU {free_gb:.1f} ГиБ < {need:.1f}"]


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
    if actor is None and "actors" in src:            # самопроверка
        actor = src["actors"][kind]
    if actor is None:
        actor, _c = load_actor(torch, prep, kind, s, src["mod"].H)
    return kpol.Arm(torch, src["mod"], src["dec"], src["M"], kind=kind,
                    seed=s, actor=actor)


# --- ожидаемые наборы эпизодов ------------------------------------------------------------
def labels_of(seed):
    return [f"R_s{seed}", f"T32_s{seed}", f"P32_s{seed}"]


def legacy_wants(prep):
    out = []
    for s in pc.parse_seeds():
        for lab in labels_of(s):
            for t in LEGACY_TASKS:
                wh = (prep["k32_plan"]["init_ref"][f"{t}:{LEGACY_INIT}"],
                      None)
                for path in ("old", "new"):
                    out.append(want_of(prep, "legacy", lab, t,
                                       f"i{LEGACY_INIT}_{path}", LEGACY_INIT,
                                       "k18" if path == "old" else "legacy",
                                       want_hash=wh))
    return out


def smoke_wants(prep, stage, labels, n=SMOKE_N):
    out = []
    for t in SMOKE_TASKS:
        for cid, pos, ih, ihf in kb36.dev_cases(t, n):
            for lab in labels:
                out.append(want_of(prep, stage, lab, t, cid, pos, "bank",
                                   space=kb36.SPACE_DEV, case_id=cid,
                                   want_hash=(ih, ihf)))
    return out


def all_smoke_wants(prep):
    return [w for s in pc.parse_seeds()
            for w in smoke_wants(prep, "smoke", labels_of(s))]


def repeat_wants(prep, stage="repeat"):
    return [w for s in pc.parse_seeds()
            for w in smoke_wants(prep, stage, [f"T32_s{s}", f"P32_s{s}"],
                                 n=1)]


def ep_files(want):
    jp, np_ = ep_paths(want["stage"], want["label"], want["task"],
                       want["key"])
    return {jp: sha(jp), np_: sha(np_)}


def files_of(wants):
    out = set()
    for w in wants:
        out.update(ep_paths(w["stage"], w["label"], w["task"], w["key"]))
    return out


# --- гейты --------------------------------------------------------------------------------
REBUILD_TOL = dict(Z0=Z_TOL, Zhat=Z_TOL, Du=DU_TOL, arm=ARM_TOL, c=1e-6)


def rebuild_entry_problems(kind, st):
    """Запись пересборки первого вызова: обязательные ошибки (R: Z0, arm;
    T/P: Z0, Zhat, Du, arm, c) конечны и 0 ≤ ошибка ≤ допуска K-32."""
    keys = ("Z0", "arm") if kind == "R" else ("Z0", "Zhat", "Du", "arm", "c")
    if not isinstance(st, dict):
        return ["нет пересборки"]
    for k in keys:
        v = st.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or \
                not np.isfinite(v) or not 0 <= v <= REBUILD_TOL[k]:
            return [f"пересборка {k} = {v!r} нет/вне [0; {REBUILD_TOL[k]}]"]
    return []


def realcheck_content_problems(g):
    p = []
    seeds = g.get("seeds") or {}
    for s in pc.parse_seeds():
        r = seeds.get(str(s))
        if not isinstance(r, dict):
            p.append(f"real-check: нет результатов s{s}")
            continue
        for k in ("new_eq_old_R", "new_eq_old_T", "new_eq_old_P",
                  "tplus_start_eq_T32", "source_frozen"):
            if r.get(k) is not True:
                p.append(f"real-check s{s}: {k} не True")
        for kind in ("T", "P"):
            c = r.get(f"cache_{kind}") or {}
            if set(c) != {"8", "32", "64"} or not all(
                    isinstance(v, dict) and v.get("mu") is not None and
                    v["mu"] <= 1e-6 and v.get("logp") is not None and
                    v["logp"] <= 1e-4 for v in c.values()):
                p.append(f"real-check s{s} {kind}: кеш μ/log π неполон или "
                         f"вне допуска")
            par = r.get(f"params_{kind}") or {}
            if r.get("H") != SOURCE_H or par.get("actor") != ACTOR_PARAMS or \
                    par.get("critic") != CRITIC_PARAMS:
                p.append(f"real-check s{s} {kind}: размерность источника "
                         f"{r.get('H')} или число параметров {par} ≠ "
                         f"паспорту (H {SOURCE_H}: {ACTOR_PARAMS} / "
                         f"{CRITIC_PARAMS})")
        pk = r.get("peak_gb")
        if not isinstance(pk, (int, float)) or not np.isfinite(pk):
            p.append(f"real-check s{s}: нет пика памяти")
    return p


def episodes_content_problems(kind, g, prep, C):
    wants = dict(legacy=legacy_wants, smoke=all_smoke_wants,
                 repeat=repeat_wants)[kind](prep)
    files = g.get("files") or {}
    p = []
    if set(files) != files_of(wants):
        return [f"{kind}: набор файлов ≠ ожидаемому ({len(files)} из "
                f"{2 * len(wants)})"]
    got = {}
    for w in wants:
        try:
            r = load_ep(w)
        except SystemExit as e:
            return [f"{kind}: {e.code}"]
        if r is None:
            return [f"{kind}: нет {w['eid']}"]
        pr = episode_problems(*r, w, C)
        if pr:
            return [f"{kind} {w['eid']}: {pr[:2]}"]
        got[w["eid"]] = r
    if kind == "legacy":
        for w in wants:
            if w["key"].endswith("_old"):
                w2 = dict(w, key=w["key"][:-4] + "_new")
                d = compare_eps(got[w["eid"]], got[eid_of(
                    "legacy", w["label"], w["task"], w2["key"])])
                if d:
                    p.append(f"legacy {w['label']} t{w['task']}: {d[:3]}")
    elif kind == "smoke":
        rb = g.get("rebuild") or {}
        for w in wants:
            pr = rebuild_entry_problems(w["kind"], rb.get(w["eid"]))
            if pr:
                p.append(f"smoke {w['eid']}: {pr[0]}")
    elif kind == "repeat":
        for w in wants:
            w0 = dict(w, stage="smoke", scope="checks",
                      eid=eid_of("smoke", w["label"], w["task"], w["key"]))
            r0 = load_ep(w0)
            if r0 is None:
                p.append(f"repeat {w['eid']}: нет пары smoke")
                continue
            z0 = {k: x for k, x in r0[1].items()
                  if not k.startswith(("F_", "f_"))}
            d = compare_eps((r0[0], z0), got[w["eid"]])
            if d:
                p.append(f"repeat {w['eid']}: ≠ smoke {d[:3]}")
    return p


def gate_problems(kind, prep, C=None):
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
    if p:
        return p
    if kind == "real_check":
        return realcheck_content_problems(g)
    return episodes_content_problems(kind, g, prep, C or consts(prep))


def checks_problems(prep):
    C = consts(prep)
    p = []
    for k in GATES:
        p += gate_problems(k, prep, C)
    return p


# --- legacy-check -----------------------------------------------------------------------
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


def _legacy_seed(torch, prep, C, L, sims, s, device, loader):
    """Область жизни источника s; наружу — только CPU-эпизоды."""
    v = load_source(torch, s, device, loader)
    out = []
    for w in legacy_wants(prep):
        if w["seed"] != s:
            continue
        arm = make_arm(torch, prep, w["label"], v)
        sim = sims["old" if w["key"].endswith("_old") else "new"]
        out.append((w, run_ep(w, sim, v["io"], arm, L, C)))
    return out


def legacy_check(a, sims=None, loader=None):
    import k18_collector as k18c
    sims = sims or dict(old=k18c.Sim(1, "persistent"),
                        new=kb36.BankSim("legacy"))
    import torch
    try:
        prep = pc.load_prepare()
        C = consts(prep)
        L = pc.ledger()
        pr = resources_problems(torch, a.device)
        if pr:
            raise SystemExit("; ".join(pr))
        got = {}
        for s in pc.parse_seeds():
            for w, r in _legacy_seed(torch, prep, C, L, sims, s, a.device,
                                     loader):
                got[w["eid"]] = (w, r)
            free(torch)
        files, prob, pairs = {}, [], {}
        for eid, (w, r) in got.items():
            files.update(ep_files(w))
            if w["key"].endswith("_old"):
                e2 = eid_of("legacy", w["label"], w["task"],
                            w["key"][:-4] + "_new")
                d = compare_eps(r, got[e2][1])
                pairs[f"{w['label']}/t{w['task']}"] = d
                if d:
                    prob.append(f"{w['label']} t{w['task']}: старый ≠ новый "
                                f"путь {d[:4]}")
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


def _wants_seed(torch, prep, C, L, sim, s, wants, device, loader,
                keep_first=False):
    """Область жизни источника s: эпизоды wants этого сида -> CPU."""
    v = load_source(torch, s, device, loader)
    arms, out = {}, []
    for w in wants:
        if w["seed"] != s:
            continue
        if w["label"] not in arms:
            arms[w["label"]] = make_arm(torch, prep, w["label"], v)
        out.append((w, run_ep(w, sim, v["io"], arms[w["label"]], L, C,
                              keep_first=keep_first)))
    return out


def bank_smoke(a, sim=None, loader=None):
    sim = sim or kb36.BankSim("bank", kb36.SPACE_DEV)
    import torch
    try:
        prep = pc.load_prepare()
        C = consts(prep)
        p0 = gate_problems("legacy", prep, C)
        if p0:
            raise SystemExit("bank-smoke без legacy-check: " + "; ".join(p0))
        pr = resources_problems(torch, a.device)
        if pr:
            raise SystemExit("; ".join(pr))
        L = pc.ledger()
        wants = all_smoke_wants(prep)
        res = []
        for s in pc.parse_seeds():
            res += _wants_seed(torch, prep, C, L, sim, s, wants, a.device,
                               loader, keep_first=True)
            free(torch)
        files, prob, rb, secs = {}, [], {}, {}
        for w, (rec, z) in res:
            files.update(ep_files(w))
            F, first = first_of(z)
            if not F or "a_codec" not in first:
                prob.append(f"{w['eid']}: нет контекста первого вызова")
                continue
            Ft = {k: torch.as_tensor(x) for k, x in F.items()}
            pr_, st = rebuild_problems(torch, prep, w["kind"], w["seed"], Ft,
                                       first)
            if not np.array_equal(first["a_codec"], z["j_a_codec"][0]):
                pr_.append("снимок не первого вызова журнала")
            rb[w["eid"]] = st
            prob += [f"{w['eid']}: {x}" for x in pr_]
            secs.setdefault(str(w["task"]), []).append(
                (rec["seconds"], rec["phys"]))
        fc = {}
        for t, v in secs.items():
            sec = np.array([x[0] for x in v])
            ph = np.array([x[1] for x in v])
            fc[t] = dict(median_s_per_ep=float(np.median(sec)),
                         p90_s_per_ep=float(np.percentile(sec, 90)),
                         s_per_step=float(sec.sum() / ph.sum()))
        per_ep = float(np.median([x[0] for v in secs.values() for x in v]))
        rep = dict(ok=not prob, problems=prob, rebuild=rb, files=files,
                   timing=fc, forecast_A_hours=per_ep * 1800 / 3600,
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


def cold_repeat(a, sim=None, loader=None):
    sim = sim or kb36.BankSim("bank", kb36.SPACE_DEV)
    import torch
    try:
        prep = pc.load_prepare()
        C = consts(prep)
        p0 = gate_problems("smoke", prep, C)
        if p0:
            raise SystemExit("cold-repeat без bank-smoke: " + "; ".join(p0))
        pr = resources_problems(torch, a.device)
        if pr:
            raise SystemExit("; ".join(pr))
        L = pc.ledger()
        wants = repeat_wants(prep)
        res = []
        for s in pc.parse_seeds():
            res += _wants_seed(torch, prep, C, L, sim, s, wants, a.device,
                               loader)
            free(torch)
        files, prob = {}, []
        for w, r1 in res:
            files.update(ep_files(w))
            w0 = dict(w, stage="smoke", scope="checks",
                      eid=eid_of("smoke", w["label"], w["task"], w["key"]))
            r0 = load_ep(w0)
            if r0 is None:
                prob.append(f"{w['eid']}: нет эпизода bank-smoke")
                continue
            z0 = {k: x for k, x in r0[1].items()
                  if not k.startswith(("F_", "f_"))}
            d = compare_eps((r0[0], z0), r1)
            if d:
                prob.append(f"{w['label']} t{w['task']}: повтор ≠ smoke "
                            f"{d[:4]}")
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
def _real_check_seed(torch, prep, s, D, rows, device):
    """Область жизни источника s; наружу — только CPU-числа."""
    import k20_data as kd20
    import k32_dist as kd
    import k32_model as km32
    import k32_policy as kpol
    import k32_ppo as kppo
    import k32_protocol as pc32
    dev = torch.device(device)
    torch.cuda.reset_peak_memory_stats(dev)
    r, prob = {}, []
    mod, bar, M, dec = pc32.load_source_dec(torch, pc32.load_base(), s,
                                            device)
    r["H"] = int(mod.H)
    vers = {n: p_._version for n, p_ in mod.named_parameters()}
    src = dict(mod=mod, dec=dec, M=M)
    ins = []
    for row in rows:
        batch = kd20.prompt_batch(bar, [np.array(D["IMG"][row], copy=True)],
                                  kd20.norm_states(np.asarray(
                                      D["state8"][[row]])),
                                  [D["task"][row]])
        ins.append((batch, int(D["pos_offset"][row]),
                    np.asarray(D["state8"][[row]], np.float64)))
    names = dict(R="R", T="T32", P="P32")
    for kind in ("R", "T", "P"):
        plan32 = pc32.load_eval_plan_kind(kind, s)
        old, _d = pc32.make_eval_arm(torch, plan32, kind, s, mod, dec, M)
        new = make_arm(torch, prep, f"{names[kind]}_s{s}", src)
        same_all = []
        for i, (bt, po, s8) in enumerate(ins):
            for el in (0, 8, 16):
                old.begin(0, i, 0)
                new.begin(0, i, 0)
                ao, ro = old.decide(bt, po, s8, el)
                an, rn = new.decide(bt, po, s8, el)
                same = np.array_equal(ao, an)
                for k in ("Z0", "ref_arm", "grip_logit", "logits", "mu", "r",
                          "eta", "D", "Du", "path", "c", "S"):
                    if k in ro or k in rn:
                        same &= k in ro and k in rn and np.array_equal(
                            np.asarray(ro[k]), np.asarray(rn[k]))
                same_all.append(bool(same))
        r[f"new_eq_old_{kind}"] = all(same_all)
        if not all(same_all):
            prob.append(f"s{s} {kind}: новый путь ≠ K-32")
        if kind == "R":
            continue
        actor, critic = load_actor(torch, prep, kind, s, mod.H)
        r[f"params_{kind}"] = dict(actor=km32.n_params(actor),
                                   critic=km32.n_params(critic))
        if mod.H != SOURCE_H or \
                r[f"params_{kind}"]["actor"] != ACTOR_PARAMS or \
                r[f"params_{kind}"]["critic"] != CRITIC_PARAMS:
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
            mus[str(mb)] = dict(
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
            cont = make_arm(torch, prep, f"T32_s{s}", src, actor=a2)
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
    r["source_frozen"] = all(not p_.requires_grad for p_ in
                             mod.parameters()) and all(
        p_._version == vers[n] for n, p_ in mod.named_parameters())
    if not r["source_frozen"]:
        prob.append(f"s{s}: источник изменён/с градиентами")
    r["peak_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    return r, prob


def real_check(a):
    """Сохранённые входы K-21 (без физических шагов): новый путь K-36 =
    путь K-32; μ/log π по кешу при minibatch 8/32/64; источник заморожен;
    число параметров; actor state.pt (начало продолжения) = T32. Каждый
    сид — в своей области жизни модели."""
    import torch
    import k21_data as d21
    prep = pc.load_prepare()
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    rows = [int(r) for r in np.asarray(D["val_rows"])[:5]]
    rep, prob = dict(seeds={}), []
    rep["mem_before_gb"] = torch.cuda.memory_allocated(dev) / 2 ** 30
    for s in pc.parse_seeds():
        r, pr = _real_check_seed(torch, prep, s, D, rows, a.device)
        free(torch)
        r["mem_after_free_gb"] = torch.cuda.memory_allocated(dev) / 2 ** 30
        rep["seeds"][str(s)] = r
        prob += pr
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
                                  "НЕ пройден: " + "; ".join(prob[:3])) +
          f"; порог свободной памяти для этапов {need_free_gb():.1f} ГиБ")
    return 0 if not prob else 3


# --- этап A -------------------------------------------------------------------------------
def block_path(lab, task, g):
    return os.path.join(OUT, "confirm_a", lab, f"t{task}_b{g:02d}.json")


def a_want(prep, lab, task, cid):
    sp = prep["splits"][str(task)]
    h = sp["init_hash"].get(cid)
    if not (isinstance(h, list) and len(h) == 2 and all(
            isinstance(x, str) and x for x in h)):
        raise SystemExit(f"{cid}: нет начальных хешей банка в prepare")
    return want_of(prep, "confirm_a", lab, task, cid, sp["pos"][cid], "bank",
                   space=kb36.SPACE_CONF, case_id=cid, want_hash=tuple(h))


def a_files(prep):
    """Ожидаемый набор файлов всех 1800 эпизодов A."""
    out = set()
    for lab, t, _g, cases in pc.cells_a(prep["splits"]):
        for cid in cases:
            out.update(ep_paths("confirm_a", lab, t, cid))
    return out


def block_problems(prep, lab, task, g, cases, C):
    bp = block_path(lab, task, g)
    if not os.path.exists(bp):
        return ["нет блока"]
    d = json.load(open(bp, encoding="utf-8"))
    if d.get("cases") != cases or d.get("prepare_sha1") != prep["sha1"]:
        return ["блок другого плана"]
    want_files = set()
    for cid in cases:
        want_files.update(ep_paths("confirm_a", lab, task, cid))
    if set(d.get("files") or {}) != want_files:
        return ["набор файлов блока ≠ ожидаемому"]
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


def seed_chunks(cells):
    """Последовательные клетки одного сида -> [(сид, [клетки])] в порядке
    очереди (порядок исполнения не меняется)."""
    out = []
    for c in cells:
        s = pc.parse_arm(c[0])[2]
        if out and out[-1][0] == s:
            out[-1][1].append(c)
        else:
            out.append((s, [c]))
    return out


def _a_chunk(torch, prep, C, L, sim, s, chunk, device, loader):
    """Область жизни источника s: клетки куска -> число новых блоков."""
    v = load_source(torch, s, device, loader)
    arms, n = {}, 0
    for lab, t, g, cases in chunk:
        if lab not in arms:
            arms[lab] = make_arm(torch, prep, lab, v)
        files = {}
        for cid in cases:
            w = a_want(prep, lab, t, cid)
            run_ep(w, sim, v["io"], arms[lab], L, C)
            files.update(ep_files(w))
        pc.save_json(block_path(lab, t, g), dict(
            label=lab, task=t, block=g, cases=cases, files=files,
            prepare_sha1=prep["sha1"]))
        n += 1
    return n


def confirm_a(a, sim=None, loader=None, cells=None, prep=None, L=None,
              C=None):
    sim = sim or kb36.BankSim("bank", kb36.SPACE_CONF)
    import torch
    try:
        if prep is None:
            prep = pc.load_prepare()
            p0 = checks_problems(prep)
            if p0:
                raise SystemExit("этап A без технических гейтов: " +
                                 "; ".join(p0[:4]))
            cells = pc.cells_a(prep["splits"])
            if hashlib.sha1(json.dumps(cells).encode()).hexdigest() != \
                    prep["cells_A_sha1"]:
                raise SystemExit("очередь A ≠ закреплённой в prepare")
            pr = resources_problems(torch, a.device)
            if pr:
                raise SystemExit("; ".join(pr))
        C = C or consts(prep)
        L = L or pc.ledger()
        todo, done0 = [], 0
        for lab, t, g, cases in cells:
            if os.path.exists(block_path(lab, t, g)):
                pr = block_problems(prep, lab, t, g, cases, C)
                if pr:
                    raise SystemExit(f"{lab} t{t} b{g}: готовый блок не "
                                     f"прошёл проверку: {pr[:2]}")
                done0 += 1
                continue
            todo.append((lab, t, g, cases))
        t0, n_new = time.time(), 0
        for s, chunk in seed_chunks(todo):
            n_new += _a_chunk(torch, prep, C, L, sim, s, chunk, a.device,
                              loader)
            free(torch)
            el = (time.time() - t0) / 3600
            left = len(cells) - done0 - n_new
            print(f"  A: готово блоков {done0 + n_new}/{len(cells)}; новых "
                  f"{n_new} за {el:.1f} ч, осталось {left} "
                  f"(~{el / max(n_new, 1) * left:.1f} ч); confirm_a "
                  f"{L.spent('confirm_a')}/{pc.CAPS['confirm_a']}",
                  flush=True)
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
    C = consts(prep)
    print("  гейты: " + ", ".join(
        f"{k} {'ok' if not gate_problems(k, prep, C) else 'нет'}"
        for k in GATES))
    print(f"  этап A: блоков {nb}/{len(cells)}; расход " + ", ".join(
        f"{k} {L.spent(k)}/{v}" for k, v in pc.CAPS.items() if L.spent(k)))
    print(f"  аварийных списаний: {len(L.d['crashes'])}; отвергнутых: "
          f"{len(os.listdir(REJECTED)) if os.path.isdir(REJECTED) else 0}")
    return 0

def extra_selftests(torch, td, prep, C, io, mod, dec, M, actor, rec, got):
    """Замечания ревью 691c3cd: сводка эпизода, содержание гейтов, области
    жизни источников."""
    import copy
    import weakref
    import k32_ppo as kppo
    # 3. сводка: незавершённый короткий эпизод, success NaN, own_steps
    s0 = rec["summary"]
    n = int(s0["steps"])
    assert summary_problems(rec, s0, n) == []
    for nm, s1, na in (
            ("незавершённый 1 шаг", dict(s0, done_step=-1, own_steps=1,
                                          steps=1, physical=11,
                                          success=False, success_step=-1),
             1),
            ("success NaN", dict(s0, success=float("nan")), n),
            ("own_steps 999", dict(s0, own_steps=999), n),
            ("длина действий", s0, n + 1),
            ("успех без шага", dict(s0, success=True, success_step=-1), n),
            ("шаг успеха без успеха", dict(s0, success=False,
                                            success_step=3), n),
            ("steps не int", dict(s0, steps=float(n)), n)):
        r1 = dict(rec, phys=int(s1.get("physical", 0)) if isinstance(
            s1.get("physical"), int) else rec["phys"])
        assert summary_problems(r1, s1, na), nm
    # 1. real-check: без результатов / неполный кеш — отказ
    good = {str(s): dict(new_eq_old_R=True, new_eq_old_T=True,
                         new_eq_old_P=True, tplus_start_eq_T32=True,
                         source_frozen=True, H=768, peak_gb=7.5,
                         params_T=dict(actor=ACTOR_PARAMS,
                                       critic=CRITIC_PARAMS),
                         params_P=dict(actor=ACTOR_PARAMS,
                                       critic=CRITIC_PARAMS),
                         cache_T={m: dict(mu=0.0, logp=0.0)
                                  for m in ("8", "32", "64")},
                         cache_P={m: dict(mu=0.0, logp=0.0)
                                  for m in ("8", "32", "64")})
            for s in (0, 1)}
    assert realcheck_content_problems(dict(seeds=good)) == []
    assert realcheck_content_problems(dict())
    # без H и с actor = critic = 1 — отказ (обход проверки параметров)
    gg = copy.deepcopy(good)
    for s in ("0", "1"):
        gg[s].pop("H")
        gg[s]["params_T"] = dict(actor=1, critic=1)
        gg[s]["params_P"] = dict(actor=1, critic=1)
    assert realcheck_content_problems(dict(seeds=gg))
    gg = copy.deepcopy(good)
    gg["0"].pop("H")
    assert realcheck_content_problems(dict(seeds=gg))      # только без H
    gg = copy.deepcopy(good)
    gg["1"]["H"] = 512
    assert realcheck_content_problems(dict(seeds=gg))
    # пересборка: Zhat обязателен для T/P; конечность и 0 ≤ e ≤ допуска
    okT = dict(Z0=1e-7, Zhat=2e-6, Du=1e-7, arm=1e-7, c=0.0)
    assert rebuild_entry_problems("T", okT) == []
    assert rebuild_entry_problems("P", okT) == []
    assert rebuild_entry_problems("R", dict(Z0=0.0, arm=0.0)) == []
    for kind in ("T", "P"):
        for bad in ({k: v for k, v in okT.items() if k != "Zhat"},
                    dict(okT, Zhat=float("nan")), dict(okT, Zhat=1.0),
                    dict(okT, Du=-1e-9), dict(okT, arm=float("inf")),
                    dict(okT, c=True), None):
            assert rebuild_entry_problems(kind, bad), (kind, bad)
    assert rebuild_entry_problems("R", dict(Z0=float("nan"), arm=0.0))
    for mut in (lambda g: g["1"].pop("cache_P"),
                lambda g: g["0"]["cache_T"].pop("64"),
                lambda g: g["0"]["cache_T"]["8"].update(logp=1e-3),
                lambda g: g["1"].update(tplus_start_eq_T32=None),
                lambda g: g["0"]["params_T"].update(actor=1)):
        gg = copy.deepcopy(good)
        mut(gg)
        assert realcheck_content_problems(dict(seeds=gg)), mut
    # 1. гейт эпизодов: точный набор, содержание, пары old/new
    G = globals()
    keepw = G["legacy_wants"]
    try:
        lw = [want_of(prep, "legacy", "T32_s0", 0, f"i30_{p_}", 30,
                      "k18" if p_ == "old" else "legacy")
              for p_ in ("old", "new")]
        G["legacy_wants"] = lambda prep_: lw
        L = kcol.Ledger(os.path.join(td, "led_g.json"), pc.CAPS)
        import k32_policy as kpol
        arm = kpol.Arm(torch, mod, dec, M, kind="T", seed=0, actor=actor)
        files = {}
        for w in lw:
            run_ep(w, kppo.ToySim(), io, arm, L, C)
            files.update(ep_files(w))
        g = dict(ok=True, problems=[], prepare_sha1=prep["sha1"],
                 files=files)
        assert episodes_content_problems("legacy", g, prep, C) == []
        one = dict(g, files=dict(list(files.items())[:1]))
        assert episodes_content_problems("legacy", one, prep, C)
        jp, np_ = ep_paths("legacy", "T32_s0", 0, "i30_new")
        zz = dict(np.load(np_))
        zz["actions"] = zz["actions"].copy()
        zz["actions"][0, 0] += 1e-6
        with open(np_, "wb") as fh:
            np.savez_compressed(fh, **zz)
        assert episodes_content_problems("legacy", g, prep, C)
    finally:
        G["legacy_wants"] = keepw
    # 2. области жизни: предыдущий источник освобождён до загрузки
    # следующего; исполнение этапа A кусками по сидам
    refs = []

    class Big:
        pass

    def loader(seed):
        gc.collect()
        assert all(r() is None for r in refs), "прежний источник жив"
        m2 = copy.deepcopy(mod)
        tag = Big()
        m2.tag = tag
        refs.append(weakref.ref(tag))
        d2 = M["Decoder"](m2).freeze()
        return dict(mod=m2, bar=None, M=M, dec=d2, io=io,
                    actors=dict(T=actor, P=actor))
    ep0 = kcol.run_episode(kppo.ToySim(), kcol.FakeIO(), kcol.FakeAgent(),
                           0, 7)
    ids = [f"k33s1_t0_c{i:05d}" for i in range(10)]
    prep2 = dict(prep, splits={"0": dict(
        pos={c: 7 for c in ids},
        init_hash={c: [ep0["init_hash"], ep0["init_hash_full"]]
                   for c in ids})})
    prep2["k32"] = copy.deepcopy(prep["k32"])
    prep2["k32"]["seeds"]["1"] = prep2["k32"]["seeds"]["0"]
    cells = [("T32_s0", 0, 0, ids[:2]), ("R_s1", 0, 0, ids[:2]),
             ("P32_s1", 0, 0, ids[2:4]), ("T32_s0", 0, 1, ids[4:6])]
    assert [c[0] for c in seed_chunks(cells)] == [0, 1, 0]
    L3 = kcol.Ledger(os.path.join(td, "led_a.json"), pc.CAPS)
    a_ = argparse.Namespace(device="cpu")
    assert confirm_a(a_, sim=_NoClose(kppo.ToySim()), loader=loader,
                     cells=cells, prep=prep2, L=L3, C=C) == 0
    assert len(refs) == 3
    for lab, t, g_, cases in cells:
        assert block_problems(prep2, lab, t, g_, cases, C) == []
    # повтор: готовые блоки не исполняются, источник не загружается
    n_ref = len(refs)
    assert confirm_a(a_, sim=_NoClose(kppo.ToySim()), loader=loader,
                     cells=cells, prep=prep2, L=L3, C=C) == 0
    assert len(refs) == n_ref
    # блок с лишним файлом — отказ
    bp = block_path(*cells[0][:3])
    d = json.load(open(bp))
    d["files"]["x"] = "y"
    pc.save_json(bp, d)
    assert block_problems(prep2, *cells[0], C)


class _NoClose:
    """ToySim с методом close (как у BankSim)."""

    def __init__(self, s):
        self.s = s

    def reset(self, *a):
        return self.s.reset(*a)

    def step(self, *a):
        return self.s.step(*a)

    def close(self):
        pass


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
        P=dict(actor_hash=ah))}),
        k32_plan=dict(init_ref={"0:30": "0" * 16, "8:30": "0" * 16}))
    prep["k32"]["seeds"]["1"] = prep["k32"]["seeds"]["0"]
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
                # произвольный закреплённый файл — не ожидаемый набор
                assert gate_problems("legacy", prep, C)
                open(fp, "w").write("{1}")
                assert gate_problems("legacy", prep, C)
                pc.save_json(gpath, dict(ok=True, problems=[],
                                         prepare_sha1="PREP"))
                assert gate_problems("legacy", prep, C)    # голый passed
                pc.save_json(gpath, dict(ok=True, problems=[],
                                         prepare_sha1="OLD",
                                         files={fp: sha(fp)}))
                assert gate_problems("legacy", prep, C)
            finally:
                GATES.update(keepg)
            extra_selftests(torch, td, prep, C, io, mod, dec, M, actor, rec,
                            got)
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
          "подмена файла, голый passed, другой prepare — отказ; сводка "
          "эпизода (незавершённый короткий, NaN, own_steps, длина) — отказ; "
          "real-check без результатов / без H / с чужим числом параметров и "
          "гейт с произвольным файлом — отказ; пересборка T/P без Zhat, с "
          "NaN/Inf, отрицательной или сверх допуска ошибкой — отказ; "
          "прежний источник освобождён до загрузки следующего; этап A "
          "кусками по сидам, повтор без исполнения")
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
