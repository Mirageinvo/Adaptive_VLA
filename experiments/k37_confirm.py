#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-37, подтверждение: T1 (T32), T2 (T32 + головка позднего листа),
T2P (T32 + контрольная головка с профилем родителя) на 480 новых сценах
(space 2, задачи 0/6/8 по 160) × 2 сида = 2 880 эпизодов, 12 очередей.

Команды (общий --run-root = reports/k37c/runs/<run_id>):
  selftest
  ctrl-cache --device D | ctrl-fit       контрольная головка (k37_control)
  bank [--workers N]                     банк space 2 (k37_bank)
  prepare --run-id ID                    plan.json + plan.sha256 (неизменяем)
  gpuinfo --device D                     хост и UUID карты (для assign)
  assign --part N --host H --uuid U [--state assigned|released|done]
  preflight --device D [--reference]     допуск карты: 22 эпизода dev
  collect --part N --device D            сбор очереди (продолжение с места)
  status | merge | analyze
Коды: 0 — норма; 3 — технический отказ; 4 — временно нет ресурса (память
карты, диск, карта занята другим процессом K-37) — повторить тот же part.

Очереди: позиция i сцены в задаче j (0, 6, 8 -> 0, 1, 2) -> part
(i + 4j) % 12; 40 сцен (14/13/13), 240 эпизодов; порядок внутри части —
сид 0, затем 1; сцены по (i, j); политики T1, T2, T2P. Журнал, эпизоды,
лог, блокировка — в part-NN; блокировка карты — по UUID в K37_LOCK_DIR
(по умолчанию logs/k37_gpu_locks). Резерв 610 шагов на эпизод; лимит
части 243·610 (три аварийных резерва запаса).
Анализ: единица — сцена; d = среднее по сидам разностей успеха; средние
по сценам в задаче, задачи с равным весом; 95% ДИ — парный
стратифицированный bootstrap по сценам внутри задачи (50 000 реплик,
seed 37001, одни выборки для обоих контрастов, перцентили numpy linear).
H1: T2 − T1, нижняя граница > 0. H2: T2 − T2P — тот же критерий, только
если H1 пройдена; иначе описательно.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k31_collect as kcol  # noqa: E402
import k36_bank as kb36  # noqa: E402
import k36_harness as kh  # noqa: E402
import k36_protocol as pc  # noqa: E402
import k37_bank as kbk  # noqa: E402
import k37_control as kc  # noqa: E402
import k37_fit as kfit  # noqa: E402
import k37_operator as kop  # noqa: E402
import k37_parent_leaf as kpl  # noqa: E402

PROTOCOL = "k37c-v1"
RUNS = "reports/k37c/runs"
TASKS = (0, 6, 8)
SEEDS = (0, 1)
POLICIES = ("T1", "T2", "T2P")
N_PARTS = 12
EP = 610
CAP_PART = dict(confirm=243 * EP)
CAP_PREFLIGHT = dict(checks=25 * EP)
IDENT = ((0, 0), (1, 8))
BOOT = dict(n=50_000, seed=37001, level=0.95, quantile="numpy linear")
RES = dict(min_free_gb=10.0, margin_gb=2.0, disk_gb=12.0)
TOL = dict(arm=1e-6, D=1e-5)
CODE = ("k37_confirm.py", "k37_control.py", "k37_bank.py", "k37_latency.py",
        "k37_operator.py", "k37_fit.py", "k37_parent_leaf.py",
        "k36b_gate.py", "k36_harness.py", "k36_bank.py", "k36_protocol.py",
        "k32_policy.py", "k32_operator.py", "k32_model.py", "k32_harness.py",
        "k31_readout.py", "k31_collect.py", "k33_bank.py")
OK, TECH, BUSY = 0, 3, 4


class Busy(Exception):
    """Временно нет ресурса -> код 4."""


def sha256(p):
    return kbk.sha256(p)


def canon_sha(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), default=str)
                          .encode()).hexdigest()


# --- очереди -----------------------------------------------------------------------------
def part_of(i, j):
    return (int(i) + 4 * int(j)) % N_PARTS


def parts_of(scenes):
    """scenes (по задачам, pos) -> {part: [сцены в порядке (i, j)]}."""
    out = {k: [] for k in range(N_PARTS)}
    for sc in scenes:
        j = TASKS.index(sc["task"])
        out[part_of(sc["pos"], j)].append((sc["pos"], j, sc))
    return {k: [x[2] for x in sorted(v, key=lambda y: (y[0], y[1]))]
            for k, v in out.items()}


def label(pol, s):
    return f"T32_s{s}" if pol == "T1" else f"{pol}_s{s}"


STAGES = dict(confirm="confirm", dev="checks", zero="checks")


def make_want(plan_sha, k36, heads, stage, pol, s, sc, space, zero=False):
    kh.STAGE_SCOPE.update(STAGES)
    pl = dict(k36, sha1=plan_sha)
    w = kh.want_of(pl, stage, f"T32_s{s}", sc["task"], sc["case_id"],
                   sc["pos"], "bank", space=space, case_id=sc["case_id"],
                   want_hash=None if sc["hash"] is None else tuple(sc["hash"]))
    if pol == "T1":
        return w
    hh = "zero" if zero else heads[pol][str(s)]["head_hash"]
    lab = label(pol, s)
    return dict(w, label=lab, kind=pol, actor_hash=f"{w['actor_hash']}+{hh}",
                eid=f"{w['scope']}/{stage}/{lab}/t{sc['task']}/"
                    f"{sc['case_id']}")


def install(out_dir):
    kh.OUT = out_dir
    kh.REJECTED = os.path.join(out_dir, "rejected")
    kh.STAGE_SCOPE.update(STAGES)


def part_wants(plan, k36, part):
    """-> [want] в порядке исполнения части."""
    out = []
    for s in SEEDS:
        for sc in plan["parts"][str(part)]["scenes"]:
            for pol in POLICIES:
                out.append(make_want(plan["sha256"], k36, plan["heads"],
                                     "confirm", pol, s, sc, kbk.SPACE))
    return out


def validator(C):
    def val(rec, z, w):
        if w["kind"] == "T":
            return kh.episode_problems(rec, z, w, C)
        p = kpl.t2_problems(rec, z, w, C)
        if not p:
            p += (kc.parent_journal_problems(z) if w["kind"] == "T2P"
                  else kc.leaf_journal_problems(z))
        return p
    return val


def saver(w, ep):
    if w["kind"] == "T":
        kh.save_ep(w, ep)
        return int(ep["physical"])
    return kpl.save_t2(w, ep)


# --- план -----------------------------------------------------------------------------------
def code_shas():
    return {f: sha256(os.path.join(HERE, f)) for f in CODE}


def heads_info(torch, p37):
    out = dict(T2={}, T2P={})
    an = pc.load_json(os.path.join(kpl.OUT, "analysis.json"))
    for s in SEEDS:
        _h, hh = kpl.load_head(torch, p37, s)
        d = os.path.join(kpl.OUT, "t2", f"T2_s{s}")
        recs = [pc.load_json(os.path.join(d, f)) for f in sorted(
            os.listdir(d)) if f.endswith(".json")] if os.path.isdir(d) else []
        n_pairs = sum(1 for x in an.get("pairs", []) if x[0] == s)
        if len(recs) != n_pairs or not recs or any(
                str(r.get("actor_hash", "")).rsplit("+", 1)[-1] != hh
                for r in recs):
            raise SystemExit(f"головка T2 s{s} ≠ головке в эпизодах пилота "
                             f"({len(recs)} записей из {n_pairs})")
        out["T2"][str(s)] = dict(path=kpl.head_path(s), head_hash=hh,
                                 sha256=sha256(kpl.head_path(s)))
        r = pc.load_json(kc.report_path(s))
        if r.get("ok") is not True or r.get("seed") != s or \
                r.get("prepare_sha1") != p37["sha1"] or \
                sha256(kc.head_path(s)) != r.get("head_sha256"):
            raise SystemExit(f"контрольная головка s{s} не прошла проверку")
        out["T2P"][str(s)] = dict(path=kc.head_path(s),
                                  head_hash=r["head_hash"],
                                  sha256=r["head_sha256"])
    return out


def make_plan(run_id):
    import torch
    k36 = pc.load_prepare()
    p37, _k = kpl.load_prepare()
    an = pc.load_json(os.path.join(kpl.OUT, "analysis.json"))
    if an.get("status") != "candidate" or an.get("prepare_sha1") != \
            p37["sha1"]:
        raise SystemExit("пилот K-37 не в статусе candidate")
    m = kbk.load_manifest()
    scenes = kbk.scenes(m)
    P = parts_of(scenes)
    k32 = k36["k32"]
    plan = dict(
        kind="k37_confirm_plan", run_id=str(run_id), protocol=PROTOCOL,
        code=code_shas(), k36_prepare_sha1=k36["sha1"],
        k37_pilot=dict(prepare_sha1=p37["sha1"],
                       analysis_sha1=pc.sha(os.path.join(kpl.OUT,
                                                         "analysis.json"))),
        k32=dict(contract=k32["k32_contract"],
                 seeds={str(s): dict(
                     source_weights_hash=k32["seeds"][str(s)]["source"][
                         "weights_hash"],
                     T_actor_hash=k32["seeds"][str(s)]["T"]["actor_hash"])
                     for s in SEEDS}),
        bar_revision=k36["k32_plan"]["bar_revision"],
        heads=heads_info(torch, p37),
        bank=dict(origin="k37 native-reset, k33_bank recipe, space 2",
                  bank_id=m["bank_id"], files=m["files"], space=kbk.SPACE,
                  dir=kbk.BANK_DIR),
        env=dict(B=1, seeds=list(SEEDS), exec_seed=101, waiting_steps=10,
                 max_steps=600, horizon=8, collector="k31_collect"),
        scenes=scenes, policies=list(POLICIES), ident=[list(x)
                                                      for x in IDENT],
        parts={str(k): dict(scenes=v, n_episodes=6 * len(v))
               for k, v in P.items()},
        caps=dict(per_part=CAP_PART, preflight=CAP_PREFLIGHT,
                  reserve_per_episode=EP),
        resource=dict(RES, preflight_need_gb=float(kh.need_free_gb())),
        tol=TOL, analysis=dict(BOOT, unit="scene", task_weights="equal",
                               H1="T2-T1 lower > 0",
                               H2="T2-T2P lower > 0 if H1"))
    for k in range(N_PARTS):
        ws = part_wants(dict(plan, sha256="X"), k36, k)
        plan["parts"][str(k)]["eids"] = [w["eid"] for w in ws]
    return plan


def plan_path(rr):
    return os.path.join(rr, "plan.json")


def prepare(a):
    rr = os.path.join(RUNS, a.run_id)
    new = make_plan(a.run_id)
    sh = canon_sha(new)
    shp = os.path.join(rr, "plan.sha256")
    if os.path.exists(plan_path(rr)) or os.path.exists(shp):
        if not (os.path.exists(plan_path(rr)) and os.path.exists(shp)):
            raise SystemExit(f"{rr}: план или plan.sha256 отсутствует — "
                             f"каталог повреждён, нужен новый run_id")
        old = json.load(open(plan_path(rr), encoding="utf-8"))
        rec = open(shp).read().strip()
        if not (canon_sha(old) == rec == sh):
            raise SystemExit(f"{plan_path(rr)} закреплён иначе или "
                             f"plan.sha256 не сходится")
        print(f"  план закреплён, совпадает: {sh[:16]}")
        return OK
    os.makedirs(rr, exist_ok=True)
    pc.save_json(plan_path(rr), new)
    with open(os.path.join(rr, "plan.sha256"), "w") as fh:
        fh.write(sh + "\n")
    pc.save_json(os.path.join(rr, "plan_meta.json"), dict(
        host=socket.gethostname(), time=time.strftime("%Y-%m-%dT%H:%M:%S"),
        cwd=os.getcwd()))
    n = sum(len(v["eids"]) for v in new["parts"].values())
    print(f"  план {plan_path(rr)}: sha256 {sh[:16]}; сцен "
          f"{len(new['scenes'])}, эпизодов {n}, очередей {N_PARTS}")
    return OK


def load_plan(rr, verify=True):
    """-> (plan с полем sha256, k36). Сверка содержимого, кода, головок,
    банка и входов K-36 до загрузки моделей."""
    p = json.load(open(plan_path(rr), encoding="utf-8"))
    sh = canon_sha(p)
    want = open(os.path.join(rr, "plan.sha256")).read().strip()
    if sh != want:
        raise SystemExit("plan.json ≠ plan.sha256")
    k36 = pc.load_prepare() if verify else None
    if verify:
        bad = [f for f, s in p["code"].items() if code_shas().get(f) != s]
        if bad:
            raise SystemExit(f"код ≠ плану: {bad}")
        if k36["sha1"] != p["k36_prepare_sha1"]:
            raise SystemExit("prepare K-36 ≠ плану")
        for pol, by in p["heads"].items():
            for s, h in by.items():
                if not os.path.exists(h["path"]) or \
                        sha256(h["path"]) != h["sha256"]:
                    raise SystemExit(f"головка {pol} s{s} ≠ плану")
        for f, s in p["bank"]["files"].items():
            if not os.path.exists(f) or sha256(f) != s:
                raise SystemExit(f"банк {f} ≠ плану")
    p["sha256"] = sh
    return p, k36


def load_heads(torch, plan):
    out = {}
    for pol in ("T2", "T2P"):
        out[pol] = {}
        for s in SEEDS:
            h = plan["heads"][pol][str(s)]
            ck = torch.load(h["path"], map_location="cpu", weights_only=False)
            head = kop.make_head(torch)
            head.load_state_dict(ck["head"])
            if kfit.head_hash(torch, head) != h["head_hash"] or \
                    int(ck.get("seed", s)) != s:
                raise SystemExit(f"веса головки {pol} s{s} ≠ плану")
            out[pol][s] = head
    return out


# --- ресурсы и блокировки ------------------------------------------------------------------
def dev_index(device):
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        raise SystemExit("CUDA_VISIBLE_DEVICES задан — карта по UUID не "
                         "сопоставима; снимите переменную")
    if not str(device).startswith("cuda:"):
        raise SystemExit(f"устройство {device}")
    return int(str(device).split(":")[1])


def gpu_query(idx, runner=None):
    """-> dict(uuid, total, used, free, util); N/A/ошибка — SystemExit."""
    cmd = ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,memory.total,"
           "memory.used,memory.free,utilization.gpu,name",
           "--format=csv,noheader,nounits", "-i", str(idx)]
    try:
        out = (runner or (lambda c: subprocess.check_output(
            c, text=True, timeout=60)))(cmd).strip()
    except Exception as e:                                   # noqa: BLE001
        raise SystemExit(f"nvidia-smi: {type(e).__name__}: {e}")
    f = [x.strip() for x in out.split(",")]
    if len(f) != 8:
        raise SystemExit(f"nvidia-smi: неожиданный ответ {out!r}")
    try:
        nums = [int(x) for x in f[3:7]]
        int(f[0])
    except ValueError:
        raise SystemExit(f"nvidia-smi: не число в {out!r}")
    if not f[1].startswith("GPU-") or int(f[0]) != int(idx) or \
            f[2].count(":") != 2:
        raise SystemExit(f"nvidia-smi: UUID/индекс/PCI {out!r}")
    return dict(index=int(f[0]), uuid=f[1], pci=f[2], total=nums[0],
                used=nums[1], free=nums[2], util=nums[3], name=f[7])


def check_gpu(idx, need_gb, runner=None):
    g = gpu_query(idx, runner)
    if g["free"] < need_gb * 1024:
        raise Busy(f"cuda:{idx}: свободно {g['free']} МиБ < "
                   f"{need_gb * 1024:.0f}")
    return g


_DISK_USAGE = None                     # самопроверка подставляет свой


def check_disk(path, need_gb=RES["disk_gb"], usage=None):
    free = (usage or _DISK_USAGE or shutil.disk_usage)(path).free
    if free < need_gb * 2 ** 30:
        raise Busy(f"диск: свободно {free / 2 ** 30:.1f} ГиБ < {need_gb}")
    return free


def lock_dir():
    return os.environ.get("K37_LOCK_DIR",
                          os.path.join(ROOT, "logs", "k37_gpu_locks"))


class Lock:
    def __init__(self, path, what, busy_exc):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "a+")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.fh.close()
            raise busy_exc(f"{what} уже занят(а) другим процессом K-37")
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(f"{socket.gethostname()} pid {os.getpid()} "
                      f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        self.fh.flush()

    def release(self):
        try:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
        finally:
            self.fh.close()


def env_versions():
    import torch
    import k33_protocol as p33
    v = dict(p33.versions(), torch=torch.__version__,
             cuda=torch.version.cuda,
             cudnn=str(torch.backends.cudnn.version()),
             python=sys.version.split()[0],
             image=os.environ.get("K37_IMAGE", "не задан"))
    return v


def bind_device(torch, idx, g, props=None):
    """До загрузки моделей: CUDA-контекст на cuda:idx и сверка с картой
    nvidia-smi (той же, что проверка памяти и блокировка): UUID, а если torch
    его не даёт — PCI domain/bus/device. Нечем сверить — отказ."""
    if props is None:
        torch.zeros(1, device=torch.device(f"cuda:{idx}"))
        props = torch.cuda.get_device_properties(idx)
    u = getattr(props, "uuid", None)
    if u is not None:
        if str(u).replace("GPU-", "").lower() != \
                g["uuid"].replace("GPU-", "").lower():
            raise SystemExit(f"cuda:{idx} в torch ≠ {g['uuid']} (UUID {u})")
        return "uuid"
    bus = getattr(props, "pci_bus_id", None)
    dv = getattr(props, "pci_device_id", None)
    dom = getattr(props, "pci_domain_id", None)
    if bus is None or dv is None:
        raise SystemExit(f"cuda:{idx}: torch не даёт ни UUID, ни PCI bus ID "
                         f"— привязку к карте подтвердить нельзя")
    d_, b_, rest = g["pci"].split(":")
    ok = int(b_, 16) == int(bus) and int(rest.split(".")[0], 16) == int(dv) \
        and (dom is None or int(d_, 16) == int(dom))
    if not ok:
        raise SystemExit(f"cuda:{idx} в torch (PCI {dom}:{bus}:{dv}) ≠ "
                         f"{g['pci']}")
    return "pci"


# --- назначения ------------------------------------------------------------------------------
def part_dir(rr, k):
    return os.path.join(rr, f"part-{int(k):02d}")


def assign(a):
    lk = os.path.join(a.run_root, ".assignments.lock")
    with open(lk, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)              # ждём другой assign
        try:
            return _assign(a)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _assign(a):
    plan, _k = load_plan(a.run_root, verify=False)
    path = os.path.join(a.run_root, "assignments.json")
    d = json.load(open(path)) if os.path.exists(path) else dict(history=[])
    k = int(a.part)
    if not 0 <= k < N_PARTS:
        raise SystemExit(f"часть {k}")
    last = next((h for h in reversed(d["history"]) if h["part"] == k), None)
    st = a.state
    if st == "assigned" and last and last["state"] == "assigned" and \
            (last["host"], last["uuid"]) != (a.host, a.uuid):
        raise SystemExit(f"part {k} уже назначен {last['host']} "
                         f"{last['uuid']}; сначала --state released")
    if st in ("released", "done") and (not last or last["state"] !=
                                       "assigned"):
        raise SystemExit(f"part {k}: нет действующего назначения")
    n = 1 + sum(1 for h in d["history"] if h["part"] == k)
    rec = dict(part=k, host=a.host if st == "assigned" else last["host"],
               uuid=a.uuid if st == "assigned" else last["uuid"], n=n,
               state=st, plan_sha256=plan["sha256"],
               time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    d["history"].append(rec)
    pc.save_json(path, d)
    if st in ("assigned", "done"):
        pc.save_json(os.path.join(part_dir(a.run_root, k), "assignment.json"),
                     rec)
    print(f"  part {k:02d}: {st} -> {rec['host']} {rec['uuid']} (№ {n})")
    return OK


def gpuinfo(a):
    g = gpu_query(dev_index(a.device))
    print(f"  host {socket.gethostname()}  {a.device}  uuid {g['uuid']}  "
          f"{g['name']}  занято {g['used']}/{g['total']} МиБ, загрузка "
          f"{g['util']}%")
    return OK


# --- исполнение очереди эпизодов ------------------------------------------------------------
def run_wants(torch, wants, L, sim, src, arms_of, val, allowed, disk_path,
              woe, on_done=None):
    """Последовательно: для новых — проверка диска до резерва; транзакция
    k36b_gate.transact (проверка до commit)."""
    import k36b_gate as kg
    for w in wants:
        arm = arms_of(w)
        if kh.load_ep(w) is None and not L.is_done(w["eid"]):
            check_disk(disk_path)

        def run_save(w=w, arm=arm):
            arm.episode = kh.episode_number(w["eid"])
            ep = kcol.run_episode(sim, src["io"], arm, w["task"],
                                  w["reset_id"])
            return saver(w, ep)
        got = kg.transact(w, L, run_save, val, allowed, woe)
        if on_done:
            on_done(w, got)


def make_arms(torch, k36, s, v, heads, zero=False):
    mk = lambda: kh.make_arm(torch, k36, f"T32_s{s}", v)  # noqa: E731
    arms = dict(T1=mk(), T2=kop.T2Arm(torch, mk(), heads["T2"][s]),
                T2P=kc.ParentArm(torch, mk(), heads["T2P"][s]))
    if zero:
        arms.update(Z_T2=kop.T2Arm(torch, mk(), kop.make_head(torch)),
                    Z_T2P=kc.ParentArm(torch, mk(), kop.make_head(torch)))
    return arms


def collect(a, sim=None, src_loader=None, heads=None, gpu_runner=None,
            disk_usage=None, torch_check=True):
    import torch
    rr, k = a.run_root, int(a.part)
    plan, k36 = load_plan(rr)
    pd = part_dir(rr, k)
    asg_p = os.path.join(pd, "assignment.json")
    if not os.path.exists(asg_p):
        raise SystemExit(f"part {k}: нет назначения (assign на cds2)")
    asg = pc.load_json(asg_p)
    idx = dev_index(a.device)
    g = gpu_query(idx, gpu_runner)
    if asg.get("plan_sha256") != plan["sha256"] or asg.get("part") != k or \
            asg.get("host") != socket.gethostname() or \
            asg.get("uuid") != g["uuid"] or asg.get("state") != "assigned":
        raise SystemExit(f"part {k}: назначение не этому хосту/карте/плану "
                         f"({asg.get('host')} {asg.get('uuid')})")
    locks = [Lock(os.path.join(pd, ".lock"), f"part {k}", SystemExit)]
    try:
        locks.append(Lock(os.path.join(lock_dir(), g["uuid"] + ".lock"),
                          f"карта {g['uuid']}", Busy))
        adm = admitted(rr, plan, g["uuid"])
        need = max(RES["min_free_gb"], adm["peak_gb"] + RES["margin_gb"])
        check_gpu(idx, need, gpu_runner)
        check_disk(pd, usage=disk_usage)
        install(pd)
        C = kh.consts(k36)
        L = kcol.Ledger(os.path.join(pd, "ledger.json"), CAP_PART)
        W = {w["eid"]: w for w in part_wants(plan, k36, k)}
        val = validator(C)
        H = heads or load_heads(torch, plan)
        sim = sim or kb36.BankSim("bank", kbk.SPACE,
                                  bank_dir=plan["bank"]["dir"])
        t0, n0 = time.time(), sum(1 for e in W if L.is_done(e))
        try:
            if torch_check:
                bind_device(torch, idx, g)
            for s in SEEDS:
                check_gpu(idx, need, gpu_runner)
                v = kh.load_source(torch, s, a.device, src_loader)
                arms = make_arms(torch, k36, s, v, H)
                ws = [w for w in W.values() if w["seed"] == s]

                def done(w, got):
                    n = sum(1 for e in W if L.is_done(e))
                    if n % 12 == 0:
                        el = time.time() - t0
                        rate = (n - n0) / el if el > 0 and n > n0 else 0
                        eta = (len(W) - n) / rate / 3600 if rate else None
                        print(f"  part {k:02d}: {n}/{len(W)}; расход "
                              f"{L.spent('confirm')}/{CAP_PART['confirm']}"
                              + (f"; ETA {eta:.1f} ч" if eta else ""),
                              flush=True)
                run_wants(torch, ws, L, sim, v,
                          lambda w: arms[w["kind"] if w["kind"] != "T"
                                         else "T1"],
                          val, [("confirm", "confirm/")], pd, W.__getitem__,
                          done)
                arms = v = None
                kh.free(torch)
        except torch.cuda.OutOfMemoryError as e:
            raise Busy(f"CUDA OOM: {e}")
        finally:
            sim.close()
        rep = part_problems(plan, k36, rr, k, C)
        pc.save_json(os.path.join(pd, "part_report.json"), rep)
        print(f"  part {k:02d}: " + ("готова, проверка пройдена"
                                     if rep["ok"] else
                                     "ПРОБЛЕМЫ: " + "; ".join(
                                         rep["problems"][:3])))
        return OK if rep["ok"] else TECH
    finally:
        for lk in reversed(locks):
            lk.release()


def part_problems(plan, k36, rr, k, C):
    """Полная проверка части: точное множество eid, нет резерва, done =
    валидные записи (scope, расход), расход = сумма + аварии."""
    pd = part_dir(rr, k)
    install(pd)
    p = []
    lp = os.path.join(pd, "ledger.json")
    if not os.path.exists(lp):
        return dict(ok=False, problems=["нет журнала"])
    L = kcol.Ledger(lp, CAP_PART)
    W = {w["eid"]: w for w in part_wants(plan, k36, k)}
    if L.d["pending"]:
        p.append(f"открытый резерв {L.d['pending']['id']}")
    extra = sorted(set(L.d["done"]) - set(W))
    miss = sorted(set(W) - set(L.d["done"]))
    if extra:
        p.append(f"лишние eid {extra[:3]}")
    if miss:
        p.append(f"нет {len(miss)} eid, напр. {miss[:2]}")
    val = validator(C)
    tot, succ = 0, {}
    for e, w in W.items():
        if e not in L.d["done"]:
            continue
        try:
            got = kh.load_ep(w)
        except SystemExit as ex:
            p.append(f"{e}: {ex}")
            continue
        if got is None:
            p.append(f"{e}: учтён, файла нет")
            continue
        pr = val(*got, w)
        dn = L.d["done"][e]
        if dn["scope"] != "confirm" or int(dn["actual"]) != got[0]["phys"]:
            pr.append(f"журнал {dn} ≠ записи ({got[0]['phys']})")
        p += [f"{e}: {x}" for x in pr]
        tot += int(dn["actual"])
        succ[e] = bool(got[0]["summary"]["success"])
    crash = sum(int(c["amount"]) for c in L.d["crashes"])
    if L.spent("confirm") != tot + crash:
        p.append(f"расход {L.spent('confirm')} ≠ {tot} + аварии {crash}")
    rej = os.path.join(pd, "rejected")
    n_rej = len(os.listdir(rej)) if os.path.isdir(rej) else 0
    return dict(ok=not p, problems=p[:50], part=k, n=len(succ),
                spent=L.spent("confirm"), crashes=len(L.d["crashes"]),
                rejected=n_rej, plan_sha256=plan["sha256"])


# --- допуск карты (preflight) ---------------------------------------------------------------
def pf_dir(rr, host, uuid):
    return os.path.join(rr, "preflight", f"{host}__{uuid}")


def admitted(rr, plan, uuid, host=None):
    host = host or socket.gethostname()
    p = os.path.join(pf_dir(rr, host, uuid), "report.json")
    if not os.path.exists(p):
        raise SystemExit(f"карта {uuid} на {host} не прошла preflight")
    r = pc.load_json(p)
    if r.get("admitted") is not True or r.get("plan_sha256") != \
            plan["sha256"]:
        raise SystemExit(f"preflight {uuid}: не допущена или другой план")
    if r.get("versions") != env_versions():
        raise SystemExit(f"preflight {uuid}: версии среды изменились")
    return r


def dev_scenes():
    out = []
    for t in TASKS:
        cid, pos, ih, ihf = kb36.dev_cases(t, 1)[0]
        out.append(dict(task=t, case_id=cid, pos=int(pos), hash=[ih, ihf]))
    return out


def pf_wants(plan, k36, s, sc):
    main = [make_want(plan["sha256"], k36, plan["heads"], "dev", pol, s, sc,
                      kb36.SPACE_DEV) for pol in POLICIES]
    zero = []
    if [s, sc["task"]] in plan["ident"]:
        zero = [make_want(plan["sha256"], k36, plan["heads"], "zero", pol, s,
                          sc, kb36.SPACE_DEV, zero=True)
                for pol in ("T2", "T2P")]
    return main, zero


def first_call(z):
    """Первый вызов: базовая команда T32, D, итоговая команда, причина."""
    a2 = z["k_a2"][0] if "k_a2" in z else z["j_a_codec"][0]
    return dict(base=np.asarray(z["j_a_codec"][0], np.float64),
                D=np.asarray(z["j_D"][0], np.float64),
                a=np.asarray(a2, np.float64),
                reason=int(z["k_reason"][0]) if "k_reason" in z else -1)


def preflight(a, sim=None, src_loader=None, heads=None, gpu_runner=None,
              torch_check=True):
    import torch
    rr = a.run_root
    plan, k36 = load_plan(rr)
    idx = dev_index(a.device)
    g = gpu_query(idx, gpu_runner)
    host = socket.gethostname()
    d = pf_dir(rr, host, g["uuid"])
    lk = Lock(os.path.join(lock_dir(), g["uuid"] + ".lock"),
              f"карта {g['uuid']}", Busy)
    try:
        check_gpu(idx, plan["resource"]["preflight_need_gb"], gpu_runner)
        check_disk(rr)
        refp = os.path.join(rr, "preflight", "reference.json")
        if a.reference:
            if os.path.exists(refp) and pc.load_json(refp).get("dir") != d:
                raise SystemExit("эталон уже назначен другой карте")
        elif not os.path.exists(refp):
            raise SystemExit("нет эталонной карты: preflight --reference "
                             "на cds2 cuda:0")
        install(d)
        C = kh.consts(k36)
        L = kcol.Ledger(os.path.join(d, "ledger.json"), CAP_PREFLIGHT)
        H = heads or load_heads(torch, plan)
        val = validator(C)
        sim = sim or kb36.BankSim("bank", kb36.SPACE_DEV)
        scs = dev_scenes() if not hasattr(a, "dev") else a.dev
        allw = {}
        for s in SEEDS:
            for sc in scs:
                m, z = pf_wants(plan, k36, s, sc)
                for w in m + z:
                    allw[w["eid"]] = w
        eps, prob = {}, []
        uuid_state = None
        try:
            if torch_check:
                uuid_state = bind_device(torch, idx, g)   # контекст создан
                torch.cuda.reset_peak_memory_stats(idx)
            for s in SEEDS:
                check_gpu(idx, plan["resource"]["preflight_need_gb"],
                          gpu_runner)
                v = kh.load_source(torch, s, a.device, src_loader)
                arms = make_arms(torch, k36, s, v, H, zero=True)
                for sc in scs:
                    m, z = pf_wants(plan, k36, s, sc)
                    for w in m + z:
                        key = ("Z_" + w["kind"]) if w["stage"] == "zero" \
                            else ("T1" if w["kind"] == "T" else w["kind"])
                        run_wants(torch, [w], L, sim, v,
                                  lambda _w, key=key: arms[key], val,
                                  [("checks", "checks/")], d,
                                  allw.__getitem__)
                    if z:
                        t1 = kh.load_ep(m[0])
                        for wz in z:
                            pr = kpl.zero_head_problems(t1, kh.load_ep(wz))
                            prob += [f"нулевая {wz['eid']}: {x}" for x in pr]
                arms = v = None
                kh.free(torch)
        except torch.cuda.OutOfMemoryError as e:
            raise Busy(f"CUDA OOM: {e}")
        finally:
            sim.close()
        secs = []
        for w in allw.values():
            rec, z = kh.load_ep(w)
            fc = first_call(z)
            secs.append(float(rec["seconds"]))     # время исполнения эпизода
            eps[w["eid"]] = dict(
                success=bool(rec["summary"]["success"]),
                init_hash=rec["summary"]["init_hash"],
                init_hash_full=rec["summary"]["init_hash_full"],
                actions_sha1=rec["summary"]["actions_sha1"],
                steps=rec["summary"]["steps"], reason=fc["reason"],
                first_base=fc["base"].tolist(), first_D=fc["D"].tolist(),
                first_a=fc["a"].tolist())
        peak = (torch.cuda.max_memory_reserved(idx) / 2 ** 30
                if torch_check else 0.0)
        rep = dict(plan_sha256=plan["sha256"], host=host, uuid=g["uuid"],
                   gpu=g, device=a.device, torch_uuid=uuid_state,
                   versions=env_versions(), peak_gb=float(peak),
                   median_s=float(np.median(secs)) if secs else None,
                   n_episodes=len(eps), episodes=eps,
                   spent=L.spent("checks"), crashes=len(L.d["crashes"]),
                   time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   reference=bool(a.reference))
        if a.reference:
            cmp_ = dict(ok=True)
        else:
            cmp_ = compare_reference(pc.load_json(os.path.join(
                pc.load_json(refp)["dir"], "report.json")), rep)
        prob += cmp_.get("problems", [])
        rep.update(compare=cmp_, problems=prob, admitted=not prob)
        pc.save_json(os.path.join(d, "report.json"), rep)
        if a.reference and not prob:
            pc.save_json(refp, dict(dir=d, host=host, uuid=g["uuid"]))
        print(f"  preflight {host} {a.device} {g['uuid']}: эпизодов "
              f"{len(eps)}, медиана {rep['median_s']:.1f} с, пик "
              f"{peak:.1f} ГиБ; " + ("ДОПУЩЕНА" if not prob else
                                     "НЕ допущена: " + "; ".join(prob[:3])))
        return OK if not prob else TECH
    finally:
        lk.release()


def compare_reference(ref, cur):
    """Жёстко: тот же план, тот же набор eid, post-wait хеши, причина и
    команды первого вызова (|Δarm| ≤ 1e−6, |ΔD| ≤ 1e−5), исходы.
    Описательно: совпадение траекторий (SHA действий)."""
    p = []
    if ref["plan_sha256"] != cur["plan_sha256"]:
        return dict(ok=False, problems=["эталон другого плана"])
    if set(ref["episodes"]) != set(cur["episodes"]):
        return dict(ok=False, problems=["другой набор эпизодов"])
    same_traj, worst = 0, dict(arm=0.0, D=0.0)
    for e, r in ref["episodes"].items():
        c = cur["episodes"][e]
        if (r["init_hash"], r["init_hash_full"]) != (c["init_hash"],
                                                     c["init_hash_full"]):
            p.append(f"{e}: post-wait хеши ≠ эталону")
            continue
        if r["reason"] != c["reason"]:
            p.append(f"{e}: применимость правки ≠ эталону")
        da = max(float(np.abs(np.subtract(r["first_base"], c["first_base"]))
                       .max()),
                 float(np.abs(np.subtract(r["first_a"], c["first_a"])).max()))
        dD = float(np.abs(np.subtract(r["first_D"], c["first_D"])).max())
        worst["arm"], worst["D"] = max(worst["arm"], da), max(worst["D"], dD)
        if not (da <= TOL["arm"] and dD <= TOL["D"]):
            p.append(f"{e}: команды первого вызова ≠ эталону ({da:.1e}, "
                     f"{dD:.1e})")
        if r["success"] != c["success"]:
            p.append(f"{e}: исход ≠ эталону")
        same_traj += int(r["actions_sha1"] == c["actions_sha1"])
    return dict(ok=not p, problems=p, worst=worst,
                identical_trajectories=same_traj, n=len(ref["episodes"]))


# --- статус, слияние, анализ ----------------------------------------------------------------
def status(a):
    plan, _k = load_plan(a.run_root, verify=False)
    ap = os.path.join(a.run_root, "assignments.json")
    asg = json.load(open(ap)) if os.path.exists(ap) else dict(history=[])
    last = {}
    for h in asg["history"]:
        last[h["part"]] = h
    tot = 0
    for k in range(N_PARTS):
        pd = part_dir(a.run_root, k)
        lp = os.path.join(pd, "ledger.json")
        h = last.get(k)
        who = f"{h['state']} {h['host']} {h['uuid'][:12]}" if h else "—"
        if not os.path.exists(lp):
            print(f"  part {k:02d}: не начата; {who}")
            continue
        d = json.load(open(lp))
        n = len(d["done"])
        tot += n
        ts = sorted(os.path.getmtime(os.path.join(dp, f))
                    for dp, _dn, fs in os.walk(pd) for f in fs
                    if f.endswith(".json") and f.startswith("t"))
        rate = sum(1 for x in ts if x > time.time() - 3600)
        eta = (len(plan["parts"][str(k)]["eids"]) - n) / rate if rate else None
        print(f"  part {k:02d}: {n}/240; расход {d['spent']}; резерв "
              f"{(d['pending'] or {}).get('id', '—')}; аварий "
              f"{len(d['crashes'])}; за час {rate}"
              + (f", ETA {eta:.1f} ч" if eta else "") + f"; {who}")
    print(f"  всего {tot}/{N_PARTS * 240}")
    return OK


def build_merge(rr, plan, k36):
    """Полная проверка всех частей -> (problems, rows, parts)."""
    C = kh.consts(k36)
    ap = os.path.join(rr, "assignments.json")
    asg = json.load(open(ap)) if os.path.exists(ap) else dict(history=[])
    last = {}
    for h in asg["history"]:
        last[h["part"]] = h
    prob, rows, parts = [], [], {}
    for k in range(N_PARTS):
        pd = part_dir(rr, k)
        ap_ = os.path.join(pd, "assignment.json")
        local = pc.load_json(ap_) if os.path.exists(ap_) else None
        h = last.get(k)
        if not h or not local or {x: local.get(x) for x in
                                  ("part", "host", "uuid", "n")} != \
                {x: h.get(x) for x in ("part", "host", "uuid", "n")}:
            prob.append(f"part {k}: назначение ≠ последнему у координатора")
        rep_ = part_problems(plan, k36, rr, k, C)
        parts[str(k)] = {x: rep_.get(x) for x in ("n", "spent", "crashes",
                                                  "rejected", "ok")}
        prob += [f"part {k}: {x}" for x in rep_["problems"]]
        if rep_["ok"]:
            install(pd)
            for w in part_wants(plan, k36, k):
                rec, z = kh.load_ep(w)
                act = int(((z["k_reason"] == 0) & z["k_u"].any(1)).sum()) \
                    if "k_reason" in z else 0
                rows.append(dict(eid=w["eid"], part=k, task=w["task"],
                                 case_id=w["case_id"], seed=w["seed"],
                                 policy=dict(T="T1").get(w["kind"],
                                                         w["kind"]),
                                 success=bool(rec["summary"]["success"]),
                                 phys=rec["phys"], calls=int(len(z["j_ep"])),
                                 active=act))
    prob += rows_problems(rows, plan)
    return prob, rows, parts


def rows_problems(rows, plan):
    """Точное множество eid плана, без дублей; каждая сцена — все 6."""
    p = []
    eids = [r["eid"] for r in rows]
    want = [e for k in range(N_PARTS) for e in plan["parts"][str(k)]["eids"]]
    n_want = 6 * len(plan["scenes"])
    if len(want) != n_want or len(set(want)) != n_want:
        p.append(f"план: eid {len(set(want))} из {n_want}")
    if len(eids) != len(set(eids)):
        p.append(f"дубли eid: {len(eids) - len(set(eids))}")
    if set(eids) != set(want):
        p.append(f"набор eid ≠ плану ({len(set(eids))} из {len(set(want))})")
    by = {}
    for r in rows:
        by.setdefault((r["task"], r["case_id"]), []).append(
            (r["seed"], r["policy"]))
    full = sorted((s_, p_) for s_ in SEEDS for p_ in POLICIES)
    nb = sum(1 for v in by.values() if sorted(v) != full)
    if nb or len(by) != len(plan["scenes"]):
        p.append(f"сцен с неполным/дублированным набором {nb}; сцен "
                 f"{len(by)}")
    return p


def merge(a):
    plan, k36 = load_plan(a.run_root)
    prob, rows, parts = build_merge(a.run_root, plan, k36)
    out = dict(plan_sha256=plan["sha256"], ok=not prob, problems=prob[:100],
               parts=parts, rows=rows if not prob else [],
               time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    pc.save_json(os.path.join(a.run_root, "merged.json"), out)
    print("  merge: " + (f"полный набор {len(rows)}, проверка пройдена"
                         if not prob else "ОТКАЗ: " + "; ".join(prob[:4])))
    return OK if not prob else TECH


def scene_table(rows, scenes):
    """-> Y[pol] [n_scenes, 2] (порядок plan.scenes), task [n_scenes]."""
    ix = {(sc["task"], sc["case_id"]): i for i, sc in enumerate(scenes)}
    Y = {p_: np.full((len(scenes), len(SEEDS)), -1, int) for p_ in POLICIES}
    for r in rows:
        cell = (ix[(r["task"], r["case_id"])], r["seed"])
        if Y[r["policy"]][cell] != -1:
            raise SystemExit(f"дубль исхода {r['policy']} {r['case_id']} "
                             f"s{r['seed']}")
        Y[r["policy"]][cell] = int(r["success"])
    if any((v < 0).any() for v in Y.values()):
        raise SystemExit("таблица исходов неполна")
    return Y, np.array([sc["task"] for sc in scenes])


def contrast_stats(Y, task, contrasts, n_boot=BOOT["n"], seed=BOOT["seed"]):
    """Сцена — единица; задачи с равным весом; общий bootstrap."""
    rng = np.random.default_rng(seed)
    idx = {t: rng.integers(0, int((task == t).sum()),
                           (n_boot, int((task == t).sum())))
           for t in TASKS}
    out = {}
    for a_, b_ in contrasts:
        d = (Y[a_] - Y[b_]).mean(1)
        est = float(np.mean([d[task == t].mean() for t in TASKS]))
        bs = np.mean([d[task == t][idx[t]].mean(1) for t in TASKS], 0)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        pairs = Y[a_].ravel(), Y[b_].ravel()
        W = int(((pairs[0] == 1) & (pairs[1] == 0)).sum())
        L_ = int(((pairs[0] == 0) & (pairs[1] == 1)).sum())
        out[f"{a_}-{b_}"] = dict(
            delta=est, lo=float(lo), hi=float(hi), W=W, L=L_, net=W - L_,
            discordant_share=(W + L_) / pairs[0].size,
            by_task={str(t): float(d[task == t].mean()) for t in TASKS},
            by_seed={str(s): float(np.mean([(Y[a_][task == t, s] -
                                             Y[b_][task == t, s]).mean()
                                            for t in TASKS]))
                     for s in SEEDS})
    return out


def analyze(a):
    plan, k36 = load_plan(a.run_root)
    mp = os.path.join(a.run_root, "merged.json")
    if not os.path.exists(mp):
        raise SystemExit("нет merged.json — сначала merge")
    m = pc.load_json(mp)
    if m.get("ok") is not True or m.get("plan_sha256") != plan["sha256"]:
        raise SystemExit("merge не пройден или от другого плана")
    prob, rows, _parts = build_merge(a.run_root, plan, k36)   # заново
    if prob:
        raise SystemExit("проверка частей при анализе: " + "; ".join(
            prob[:3]))
    key = lambda r: r["eid"]  # noqa: E731
    if sorted(rows, key=key) != sorted(m["rows"], key=key):
        raise SystemExit("merged.json ≠ проверенным записям частей")
    Y, task = scene_table(rows, plan["scenes"])
    m = dict(m, rows=rows)
    st = contrast_stats(Y, task, [("T2", "T1"), ("T2", "T2P"),
                                  ("T2P", "T1")])
    h1 = st["T2-T1"]["lo"] > 0
    h2 = (st["T2-T2P"]["lo"] > 0) if h1 else None
    sr = {p_: dict(all=float(Y[p_].mean()),
                   by_seed={str(s): float(Y[p_][:, s].mean()) for s in SEEDS},
                   by_task={str(t): float(Y[p_][task == t].mean())
                            for t in TASKS}) for p_ in POLICIES}
    cov = {}
    for p_ in ("T2", "T2P"):
        rs = [r for r in m["rows"] if r["policy"] == p_]
        cov[p_] = sum(r["active"] for r in rs) / max(1, sum(r["calls"]
                                                            for r in rs))
    out = dict(plan_sha256=plan["sha256"], sr=sr, contrasts=st,
               H1=dict(passed=bool(h1), rule="T2-T1 lower > 0"),
               H2=dict(passed=h2, rule="T2-T2P lower > 0 (only if H1)",
                       descriptive=h2 is None),
               active_share=cov, parts=m["parts"], boot=BOOT)
    pc.save_json(os.path.join(a.run_root, "analysis.json"), out)
    for k_, v in st.items():
        print(f"  {k_}: Δ {100 * v['delta']:+.2f} п.п. 95% "
              f"[{100 * v['lo']:+.2f}; {100 * v['hi']:+.2f}]; W {v['W']} L "
              f"{v['L']} net {v['net']:+d}; расхождений "
              f"{100 * v['discordant_share']:.1f}%")
    print("  SR: " + ", ".join(f"{p_} {100 * sr[p_]['all']:.1f}"
                                for p_ in POLICIES))
    print(f"  H1 (T2 − T1): {'подтверждена' if h1 else 'не подтверждена'}; "
          f"H2 (T2 − T2P): " + ("описательно" if h2 is None else
                                ("подтверждена" if h2 else
                                 "не подтверждена")))
    return OK


# --- самопроверка ---------------------------------------------------------------------------
def selftest():
    import tempfile
    from collections import Counter
    # очереди: 12 по 40 сцен, 14/13/13, все 480 ровно один раз
    scenes = [dict(task=t, case_id=f"k33s2_t{t}_c{i:05d}", pos=i, hash=None)
              for t in TASKS for i in range(160)]
    P = parts_of(scenes)
    assert all(len(v) == 40 for v in P.values())
    assert sorted(sc["case_id"] for v in P.values() for sc in v) == \
        sorted(sc["case_id"] for sc in scenes)
    for k, v in P.items():
        c = Counter(sc["task"] for sc in v)
        assert sorted(c.values()) == [13, 13, 14], (k, c)
    assert CAP_PART["confirm"] == 148230 and 12 * CAP_PART["confirm"] == \
        1778760
    # канонический хеш не зависит от порядка ключей
    assert canon_sha(dict(a=1, b=[1, 2])) == canon_sha(dict(b=[1, 2], a=1))
    # nvidia-smi: N/A и не число — отказ; мало памяти — Busy
    ok_line = ("1, GPU-abc, 00000000:3B:00.0, 32510, 13213, 19297, 93, "
               "Tesla V100")
    g = gpu_query(1, lambda c: ok_line)
    assert g["free"] == 19297 and g["uuid"] == "GPU-abc"
    for bad in ("1, GPU-abc, 00000000:3B:00.0, 32510, N/A, 19297, 93, V100",
                "1, GPU-abc, 32510",
                "0, GPU-abc, 00000000:3B:00.0, 32510, 1, 19297, 93, V100",
                "1, GPU-abc, [N/A], 32510, 1, 19297, 93, V100"):
        try:
            gpu_query(1, lambda c, b=bad: b)
            raise AssertionError(bad)
        except SystemExit:
            pass
    check_gpu(1, 12.0, lambda c: ok_line)        # чужой процесс, но хватает
    try:
        check_gpu(1, 20.0, lambda c: ok_line)
        raise AssertionError("мало памяти прошло")
    except Busy:
        pass
    try:
        check_disk(".", usage=lambda p: type("U", (), dict(free=2 ** 30))())
        raise AssertionError("мало диска прошло")
    except Busy:
        pass
    # привязка к карте до загрузки моделей
    P = lambda **k: type("P", (), k)()  # noqa: E731
    assert bind_device(None, 1, g, P(uuid="abc")) == "uuid"
    assert bind_device(None, 1, g, P(pci_bus_id=0x3B, pci_device_id=0,
                                     pci_domain_id=0)) == "pci"
    for bad_p in (P(uuid="zzz"), P(pci_bus_id=0x3C, pci_device_id=0), P()):
        try:
            bind_device(None, 1, g, bad_p)
            raise AssertionError("привязка к чужой/непроверяемой карте")
        except SystemExit:
            pass
    # блокировки: вторая на тот же файл — отказ
    with tempfile.TemporaryDirectory() as td:
        l1 = Lock(os.path.join(td, "g.lock"), "карта", Busy)
        try:
            Lock(os.path.join(td, "g.lock"), "карта", Busy)
            raise AssertionError("двойная блокировка")
        except Busy:
            pass
        l1.release()
        Lock(os.path.join(td, "g.lock"), "карта", Busy).release()
    # bootstrap: известный сдвиг ловится, нулевой — нет; общие выборки
    rng = np.random.default_rng(1)
    n = 480
    task = np.repeat(TASKS, 160)
    base = (rng.random((n, 2)) < 0.8).astype(int)
    Y = dict(T1=base, T2P=base.copy(), T2=base.copy())
    flip = rng.random((n, 2)) < 0.12
    Y["T2"] = np.where(flip & (base == 0), 1, base)
    st = contrast_stats(Y, task, [("T2", "T1"), ("T2P", "T1")], n_boot=4000)
    assert st["T2-T1"]["lo"] > 0 and st["T2-T1"]["L"] == 0
    assert st["T2P-T1"]["delta"] == 0 and st["T2P-T1"]["lo"] == 0
    st2 = contrast_stats(Y, task, [("T2", "T1")], n_boot=4000)
    assert st2["T2-T1"]["lo"] == st["T2-T1"]["lo"]      # тот же seed
    e2e_selftest()
    print("самопроверка k37_confirm пройдена: 12 очередей по 40 сцен "
          "(14/13/13), лимиты; nvidia-smi N/A/не число — отказ, чужой "
          "процесс при достатке памяти — допуск, нехватка памяти/диска — "
          "код 4; блокировка карты эксклюзивна; bootstrap по сценам; "
          "игрушечный сбор/слияние/анализ (см. e2e)")
    return 0


def e2e_selftest():
    """Игрушка: 2 части × 2 сцены, ToySim; preflight эталона и второй
    карты; сбор с аварией до записи и после записи до commit; слияние
    отвергает дубль, пропуск и чужую головку; анализ."""
    import tempfile
    import torch
    import k21_model as km
    import k25_model as k25m
    import k29_model as k29m
    import k32_model as km32
    import k32_ppo as kppo
    g_ = globals()
    k29m.no_tf32(torch)
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
    heads = {}
    for pol, bias in (("T2", [1.5, -1.0, 0.8, 2.0, -1.7, 0.6]),
                      ("T2P", [-1.2, 0.9, 1.4, -0.6, 1.8, -2.0])):
        heads[pol] = {}
        for s in SEEDS:
            hd = kop.make_head(torch)
            with torch.no_grad():
                hd.bias.copy_(torch.tensor(bias))
            heads[pol][s] = hd
    hinfo = {pol: {str(s): dict(head_hash=kfit.head_hash(torch, heads[pol][s]))
                   for s in SEEDS} for pol in heads}
    k36 = dict(sha1="PREP36", k32=dict(k32_contract="C", seeds={
        str(s): dict(source=dict(weights_hash="W"), T=dict(actor_hash=ah))
        for s in SEEDS}), k32_plan=dict(
        bounds=dict(L=list(Lb) + [-1.0], U=list(Ub) + [1.0]),
        scale_q=np.asarray(io.max_act_q).tolist()))
    scenes = [dict(task=t, case_id=f"k33s2_t{t}_c{i:05d}", pos=i, hash=None)
              for t in TASKS for i in range(2)]
    plan = dict(sha256="PLANSHA", heads=hinfo, ident=[list(x) for x in IDENT],
                scenes=scenes,
                parts={"0": dict(scenes=scenes[:3]), "1": dict(scenes=scenes[3:])},
                resource=dict(RES, preflight_need_gb=10.0))
    for k in ("0", "1"):
        plan["parts"][k]["eids"] = [w["eid"] for w in
                                    part_wants(plan, k36, int(k))]
    src = dict(mod=mod, dec=dec, M=M, io=io, actors=dict(T=actor))
    ok_line = "{i}, GPU-{u}, 00000000:3B:00.0, 32510, 0, 32510, 0, Tesla V100"
    keep = {k: g_[k] for k in ("load_plan", "env_versions", "N_PARTS",
                               "dev_scenes", "_DISK_USAGE", "make_plan",
                               "RUNS")}
    keep_kh = dict(OUT=kh.OUT, REJECTED=kh.REJECTED, SS=dict(kh.STAGE_SCOPE))
    with tempfile.TemporaryDirectory() as td:
        os.environ["K37_LOCK_DIR"] = os.path.join(td, "locks")
        try:
            g_.update(load_plan=lambda rr, verify=True: (dict(plan), k36),
                      env_versions=lambda: dict(torch="x"), N_PARTS=2,
                      _DISK_USAGE=lambda p: type("U", (), dict(
                          free=100 * 2 ** 30))())
            rr = os.path.join(td, "run")
            os.makedirs(rr)
            host = socket.gethostname()
            dev = [dict(task=t, case_id=f"k33s0_t{t}_c00000", pos=0, hash=None)
                   for t in TASKS]
            NS = argparse.Namespace
            # preflight: эталон, затем вторая «карта»
            for i, ref in ((0, True), (1, False)):
                a_ = NS(run_root=rr, device="cpu", reference=ref, dev=dev)
                g_["dev_index"] = lambda d, i=i: i
                rc = preflight(a_, sim=kh._NoClose(kppo.ToySim()),
                               src_loader=lambda s: src, heads=heads,
                               gpu_runner=lambda c, i=i: ok_line.format(
                                   i=i, u=f"u{i}"), torch_check=False)
                assert rc == OK, rc
            r1 = pc.load_json(os.path.join(pf_dir(rr, host, "GPU-u1"),
                                           "report.json"))
            assert r1["admitted"] and r1["n_episodes"] == 22 and \
                r1["compare"]["identical_trajectories"] == \
                r1["compare"]["n"] == 22
            # prepare: повтор — только при целом plan.sha256
            g_.update(make_plan=lambda rid: dict(run_id=rid, scenes=[],
                                                 parts={}),
                      RUNS=os.path.join(td, "runs"))
            assert prepare(NS(run_id="p")) == OK == prepare(NS(run_id="p"))
            os.remove(os.path.join(td, "runs", "p", "plan.sha256"))
            try:
                prepare(NS(run_id="p"))
                raise AssertionError("повтор без plan.sha256 прошёл")
            except SystemExit:
                pass
            # назначения: параллельные assign не теряют записи
            import threading
            ths = [threading.Thread(target=assign, args=(NS(
                run_root=rr, part=k, host=host, uuid="GPU-u0",
                state="assigned"),)) for k in (0, 1)]
            for th in ths:
                th.start()
            for th in ths:
                th.join()
            hist = json.load(open(os.path.join(rr, "assignments.json")))
            assert sorted(h["part"] for h in hist["history"]) == [0, 1]
            g_["dev_index"] = lambda d: 0
            gr = lambda c: ok_line.format(i=0, u="u0")  # noqa: E731

            def run_part(k, sim=None):
                return collect(NS(run_root=rr, part=k, device="cpu"),
                               sim=sim or kh._NoClose(kppo.ToySim()),
                               src_loader=lambda s: src, heads=heads,
                               gpu_runner=gr, torch_check=False)
            # авария до записи результата: исключение внутри эпизода
            class Crash(kh._NoClose):
                n = 0

                def step(self, *a):
                    Crash.n += 1
                    if Crash.n == 30:
                        raise RuntimeError("авария")
                    return super().step(*a)
            try:
                run_part(0, Crash(kppo.ToySim()))
                raise AssertionError("авария не прервала")
            except RuntimeError:
                pass
            L0 = kcol.Ledger(os.path.join(part_dir(rr, 0), "ledger.json"),
                             CAP_PART)
            assert L0.d["pending"] is not None
            assert run_part(0) == OK
            L0 = kcol.Ledger(os.path.join(part_dir(rr, 0), "ledger.json"),
                             CAP_PART)
            assert len(L0.d["crashes"]) == 1 and not L0.d["pending"]
            # запись есть, commit не успел: снять done последнего eid
            assert run_part(1) == OK
            lp1 = os.path.join(part_dir(rr, 1), "ledger.json")
            d1 = json.load(open(lp1))
            e_last = plan["parts"]["1"]["eids"][-1]
            amt = d1["done"].pop(e_last)
            d1["spent"]["confirm"] -= amt["actual"]
            d1["pending"] = dict(scope="confirm", id=e_last, amount=EP)
            json.dump(d1, open(lp1, "w"))
            assert run_part(1) == OK
            d1 = json.load(open(lp1))
            assert e_last in d1["done"] and not d1["pending"] and \
                not d1["crashes"]
            # слияние полного набора игрушки (2 части)
            pr_m, rows_m, _pp = build_merge(rr, plan, k36)
            assert not pr_m and len(rows_m) == 36, pr_m[:3]
            # противоречащий дубль исхода -> отказ и в слиянии, и в таблице
            dup = rows_m + [dict(rows_m[0], success=not rows_m[0]["success"])]
            assert rows_problems(dup, plan)
            try:
                scene_table(dup, plan["scenes"])
                raise AssertionError("дубль принят таблицей")
            except SystemExit:
                pass
            # назначение подменено локально -> отказ
            ap0 = os.path.join(part_dir(rr, 0), "assignment.json")
            a0 = pc.load_json(ap0)
            pc.save_json(ap0, dict(a0, n=99))
            assert build_merge(rr, plan, k36)[0]
            pc.save_json(ap0, a0)
            # пропуск: удалить файл эпизода -> отказ
            w0 = part_wants(plan, k36, 0)[4]
            install(part_dir(rr, 0))
            jp, np_ = kh.ep_paths(w0["stage"], w0["label"], w0["task"],
                                  w0["key"])
            os.replace(np_, np_ + ".bak")
            assert build_merge(rr, plan, k36)[0]
            os.replace(np_ + ".bak", np_)
            # чужая головка: подмена хеша головки в плане -> отказ
            bad = json.loads(json.dumps(plan))
            bad["heads"]["T2"]["0"]["head_hash"] = "другая"
            assert build_merge(rr, bad, k36)[0]
            # вторая очередь на ту же карту при занятой блокировке
            lk = Lock(os.path.join(lock_dir(), "GPU-u0.lock"), "карта", Busy)
            try:
                run_part(0)
                raise AssertionError("карта занята, а сбор пошёл")
            except Busy:
                pass
            lk.release()
            # анализ на игрушечном полном наборе
            _pr, rows_m, _pp = build_merge(rr, plan, k36)
            Y, task = scene_table(rows_m, plan["scenes"])
            st = contrast_stats(Y, task, [("T2", "T1"), ("T2", "T2P")],
                                n_boot=200)
            assert set(st) == {"T2-T1", "T2-T2P"}
        finally:
            g_.update(keep)
            g_.pop("dev_index", None)
            g_["dev_index"] = _dev_index_orig
            kh.OUT, kh.REJECTED = keep_kh["OUT"], keep_kh["REJECTED"]
            kh.STAGE_SCOPE.clear()
            kh.STAGE_SCOPE.update(keep_kh["SS"])
            os.environ.pop("K37_LOCK_DIR", None)


_dev_index_orig = dev_index


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-37: подтверждение")
    ap.add_argument("cmd", choices=("ctrl-cache", "ctrl-fit", "bank",
                                    "prepare", "gpuinfo", "assign",
                                    "preflight", "collect", "status",
                                    "merge", "analyze"))
    ap.add_argument("--run-root", default=os.environ.get("K37_RUN_ROOT"))
    ap.add_argument("--run-id")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--part", type=int)
    ap.add_argument("--host")
    ap.add_argument("--uuid")
    ap.add_argument("--state", default="assigned",
                    choices=("assigned", "released", "done"))
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    fn = {"ctrl-cache": kc.ctrl_cache, "ctrl-fit": kc.ctrl_fit,
          "bank": kbk.build, "prepare": prepare, "gpuinfo": gpuinfo,
          "assign": assign, "preflight": preflight, "collect": collect,
          "status": status, "merge": merge, "analyze": analyze}[a.cmd]
    need_rr = ("assign", "preflight", "collect", "status", "merge", "analyze")
    if a.cmd in need_rr and not a.run_root:
        print("  ОТКАЗ: нужен --run-root или K37_RUN_ROOT")
        return TECH
    if a.cmd == "prepare" and not a.run_id:
        print("  ОТКАЗ: нужен --run-id")
        return TECH
    try:
        return fn(a)
    except Busy as e:
        print(f"  ВРЕМЕННО НЕТ РЕСУРСА: {e}", flush=True)
        return BUSY
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return TECH
        raise


if __name__ == "__main__":
    sys.exit(main())
