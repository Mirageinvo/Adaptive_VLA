#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-42: повторный захват после подтверждённого промаха (детерминированный пилот).

Банк: LIBERO-10, задача 8, свежее пространство 6 (проверяется по фактическим
манифестам: K-33 0/1, K-37c 2, K-39 3, K-39 pulse 4, K-40b eval 5), 104
допустимые сцены (k33_bank через k37_bank.run_chunk), исключены официальные
старты и подписи/seed всех прежних банков; холодный повтор хешей.
Разбиение до раскаток: сортировка по sha256("k42-recovery-v1|8|case_id");
первые 96 — пилот (часть = ранг mod 2, три ветви сцены в одной части),
остальные 8 — dev/калибровка (smoke, calibrate).

Ветви (T32_s0, детерминированный, сид 0):
  T32        штатный эпизод (k31_collect.run_episode) с наблюдателем;
  RetryOld   свежая среда, переисполнение физического префикса до развилки
             (prefix) со сверкой хешей, опция повтора с прежней целью;
  RetryLive  то же, цель — локализатор по RGB agentview (+ смещение),
             при неуверенности/смещении > 40 мм — ровно прежняя цель.
После опции — тот же T32 до terminal или лимита 600. Без триггера исходы
RetryOld и RetryLive равны T32 (эпизоды не исполняются).

Наблюдатель (только полученные данные; тонкая обёртка sim.step): буфер 32
точек (исполненное действие, позиция/кватернион кисти, пальцы); первое
закрытие — три open -> три close по командам среды через границы чанков
(как k39_pulse.grip_event, +1 close, −1 open); на границах чанков в окне 16
шагов после close: команда close непрерывна, подъём кисти ≥ 20 мм, апертура
пальцев ≤ empty_max k шагов подряд -> подтверждённый промах; ≥ held_min —
захват; между — неопределённость. Один триггер на эпизод; отказ с причиной
(неопределённость, нет подъёма, схват открыт, мало шагов < 56).
Опция (≤ 48 шагов, 6 чанков): отвод вверх, отвод к точке подхода,
выравнивание над целью, подход к высоте закрытия, закрытие (если поза
достигнута, иначе возврат T32), подъём; P-регулятор позы кисти в дельта-
командах OSC с ограничением амплитуды, обратный перевод в формат кодека
(обратный scale_exec с проверкой round-trip), схват ±1.

Команды: bank [--workers N] | smoke --device D | calibrate | prepare |
collect --part K --device D | status | analyze | --selftest.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PROTOCOL = "k42-recovery-v1"
TASK = 8
SPACE = 6
N_PILOT, N_DEV = 96, 8
N_SCENES = N_PILOT + N_DEV
MAX_CAND, CHUNK = 312, 48
N_PARTS = 2
SEED = 0
EP = 610
WAIT, MAX_STEPS, H8 = 10, 600, 8
BRANCHES = ("T32", "RetryOld", "RetryLive")
RETRY = ("RetryOld", "RetryLive")
ROOT = Path("reports/k42/recovery/v1")
DATA = Path("data/k42/recovery/v1")
BANK_DIR = str(DATA / "bank")
WORK = str(DATA / "bank_work")
MANIFEST = str(ROOT / "bank_manifest.json")
BANK_LEDGER = str(DATA / "bank_ledger.json")
SHARDS = DATA / "shards"
SMOKE = DATA / "smoke"
CALIB = DATA / "localizer.npz"
CONFIG = ROOT / "config.json"
PREV = ("reports/k33/bank_manifest.json", "reports/k37c/bank_manifest.json",
        "reports/k39/bank_manifest.json",
        "reports/k39/pulse/b1/bank_manifest.json",
        "reports/k40b/eval/bank_manifest.json")
CODE = ("k42_recovery.py", "k40b_eval_base.py", "k31_collect.py",
        "k32_harness.py", "k32_policy.py", "k36_harness.py", "k36_bank.py",
        "k33_bank.py", "k37_bank.py", "k37_confirm.py", "k38_grip_run.py",
        "k38_grip_operator.py", "k39_cache.py", "k29_policy.py")
BOOT = dict(n=50_000, seed=42001)
RULE = dict(live_t32_D=5, live_old_D=3, old_t32_D=5)
OK, TECH, BUSY = 0, 3, 4
OPEN, CLOSE = -1.0, 1.0                      # физическая команда схвата
PHASES = ("retreat_up", "retreat_back", "align", "approach", "close", "lift")

# Начальная конфигурация; config.json (после smoke) переопределяет ключи,
# prepare закрепляет итог в плане.
CFG0 = dict(
    buffer=32, window=16, lift_m=0.020, k_ap=3, empty_max=0.004,
    held_min=0.010, min_left=56, n_pre=3, n_post=3,
    pos_m=0.05, rot_rad=0.5, gain=1.0, umax=0.6, rot_gain=0.0,
    umax_rot=0.3, clear_m=0.06, lift_opt_m=0.06, tol_xy=0.012,
    tol_z=0.012, max_shift_m=0.040, approach_back=16,
    guard_m=0.010, clear_tol=0.010, live_min_m=0.003)


def sha256(p):
    import k37_confirm as kcf
    return kcf.sha256(p)


def h256(s):
    return hashlib.sha256(s.encode()).hexdigest()


def jsonable(x):
    return json.loads(json.dumps(x, default=lambda v: np.asarray(v).tolist()))


def load_cfg():
    cfg = dict(CFG0)
    if CONFIG.exists():
        extra = json.load(open(CONFIG, encoding="utf-8"))
        bad = set(extra) - set(CFG0)
        if bad:
            raise SystemExit(f"{CONFIG}: неизвестные ключи {sorted(bad)}")
        cfg.update(extra)
    return cfg


# --- банк -------------------------------------------------------------------------------------
def spaces_of(m):
    """Пространства манифеста: поле space и каталоги s<N> путей файлов."""
    out = set()
    if "space" in m:
        out.add(int(m["space"]))
    for p in m.get("files", {}):
        r = re.search(r"(?:^|/)s(\d+)/t\d+\.npz$", str(p))
        if r:
            out.add(int(r.group(1)))
    return out


def prev_seen():
    """Подписи, seed и пространства прежних банков по манифестам (SHA1/
    SHA256 файлов — k40b_eval_base.file_hash_ok)."""
    import k40b_eval_base as kb40
    sigs, seeds, spaces = set(), set(), set()
    for mp in PREV:
        if not os.path.exists(mp):
            raise SystemExit(f"нет манифеста прежнего банка {mp}")
        m = json.load(open(mp, encoding="utf-8"))
        spaces |= spaces_of(m)
        for p, s in m["files"].items():
            if not os.path.exists(p) or not kb40.file_hash_ok(p, s):
                raise SystemExit(f"{mp}: файл {p} отсутствует или изменён")
            z = np.load(p, allow_pickle=False)
            sigs.update(str(x) for x in z["sig"])
            seeds.update(int(x) for x in z["seed"])
    if SPACE in spaces:
        raise SystemExit(f"space {SPACE} уже занят прежним банком")
    return frozenset(sigs), frozenset(seeds), sorted(spaces)


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
            raise SystemExit(f"t{TASK}: допустимых {len(cases)} < {N_SCENES}"
                             f" за {start} кандидатов")
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
    if os.path.exists(p):                     # продолжение после частичной записи
        old, _m = kb.load_task(SPACE, TASK, BANK_DIR)
        if [(c["case_id"], c["init_hash"], c["init_hash_full"]) for c in old] \
                != [(c["case_id"], c["init_hash"], c["init_hash_full"])
                    for c in cases]:
            raise SystemExit(f"{p}: записанная задача ≠ пересчёту из кеша "
                             f"генерации — нужен разбор")
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
        elif (r["init_hash"], r["init_hash_full"]) != \
                (c["init_hash"], c["init_hash_full"]):
            bad.append((c["case_id"], "post-wait хеши ≠ банку"))
        elif r["done"] or r["success"] or not r["finite"]:
            bad.append((c["case_id"], "успех/done/нечисловое"))
    man = dict(kind="k42_recovery_bank", space=SPACE, tasks=[TASK],
               n_per_task=N_SCENES, recipe=kb.RECIPE, files=files,
               per={str(TASK): dict(cases=len(cases), candidates=start,
                                    reject_counts={w: sum(1 for r in rej
                                                          if r["why"] == w)
                                                   for w in {r["why"]
                                                             for r in rej}})},
               prev=list(PREV), prev_spaces=spaces,
               replay=dict(n=len(jobs), bad=len(bad), examples=bad[:20]),
               versions=p33.versions(), budget=L.spent("bank"),
               created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    man["bank_id"] = h256(json.dumps(dict(recipe=kb.RECIPE, files=files),
                                     sort_keys=True, default=str))
    if bad:
        p33.save_json(MANIFEST + ".failed.json", man)
        print(f"  холодный повтор: {len(bad)} расхождений {bad[:3]}")
        return TECH
    p33.save_json(MANIFEST, man)
    print(f"  банк space {SPACE}: {len(jobs)} сцен t{TASK}, холодный повтор "
          f"совпал; bank_id {man['bank_id'][:12]}")
    return OK


def split_rows(rows):
    """rows [dict(case_id, ...)] -> (пилот, dev) по хешу; ранг, часть,
    порядок ветвей восстановления (вращение по рангу)."""
    rows = sorted(rows, key=lambda r: h256(f"{PROTOCOL}|{TASK}|"
                                           f"{r['case_id']}"))
    pilot, dev = [], []
    for rk, r in enumerate(rows):
        r = dict(r, rank=rk)
        if rk < N_PILOT:
            r["part"] = rk % N_PARTS
            r["order"] = list(RETRY if (rk // N_PARTS) % 2 == 0
                              else RETRY[::-1])
            pilot.append(r)
        else:
            dev.append(r)
    return pilot, dev


def scenes():
    """-> (манифест, пилот [dict], dev [dict])."""
    import k33_bank as kb
    m = json.load(open(MANIFEST, encoding="utf-8"))
    bad = [p for p, s in m["files"].items()
           if not os.path.exists(p) or sha256(p) != s]
    if bad or m.get("replay", {}).get("bad") != 0:
        raise SystemExit(f"банк: файлы изменены {bad} или повтор не пройден")
    cs, _meta = kb.load_task(SPACE, TASK, BANK_DIR)
    if len(cs) != N_SCENES:
        raise SystemExit(f"сцен {len(cs)} ≠ {N_SCENES}")
    rows = [dict(task=TASK, case_id=c["case_id"], pos=pos,
                 hash=[c["init_hash"], c["init_hash_full"]])
            for pos, c in enumerate(cs)]
    pilot, dev = split_rows(rows)
    return m, pilot, dev


def eid_of(branch, sc):
    return f"k42|{branch}|s{SEED}|t{sc['task']}|{sc['case_id']}"


def jobs_of(plan, part, triggered=None):
    """Эпизоды части по рангу: T32, затем ветви в порядке сцены.
    triggered — множество case_id с триггером (None — все возможные)."""
    out = []
    for sc in sorted((x for x in plan["pilot"] if x["part"] == part),
                     key=lambda x: x["rank"]):
        out.append(dict(eid=eid_of("T32", sc), branch="T32", scene=sc))
        if triggered is None or sc["case_id"] in triggered:
            for b in sc["order"]:
                out.append(dict(eid=eid_of(b, sc), branch=b, scene=sc))
    return out


# --- геометрия, команды ------------------------------------------------------------------------
def st9(state):
    s = np.asarray(state, np.float64).reshape(-1)
    if s.shape != (9,) or not np.isfinite(s).all():
        raise SystemExit(f"state: форма {s.shape} или нечисловые значения")
    return s


def aperture(s):
    """Апертура пальцев: qpos левого минус qpos правого (Panda: [0, 0.04]
    и [−0.04, 0])."""
    return float(s[0] - s[1])


def qmul(a, b):
    """Произведение кватернионов (x, y, z, w) — порядок robosuite."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def rot_err(q_tgt, q_cur):
    """Ось-угол поворота q_tgt·q_cur⁻¹ (мировая система), рад."""
    qc = np.asarray(q_cur, np.float64)
    e = qmul(np.asarray(q_tgt, np.float64), np.array([-qc[0], -qc[1],
                                                       -qc[2], qc[3]]))
    if e[3] < 0:
        e = -e
    v = e[:3]
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3)
    return v / n * 2.0 * np.arctan2(n, e[3])


def pctl(p, q_cur, T, q_tgt, grip, cfg):
    """P-регулятор на чанк: постоянная дельта-команда OSC [8, 7] (рука в
    единицах входа контроллера, схват ±1)."""
    u = np.clip(cfg["gain"] * (np.asarray(T) - p) / (H8 * cfg["pos_m"]),
                -cfg["umax"], cfg["umax"])
    r = np.clip(cfg["rot_gain"] * rot_err(q_tgt, q_cur) /
                (H8 * cfg["rot_rad"]), -cfg["umax_rot"], cfg["umax_rot"])
    E = np.zeros((H8, 7), np.float64)
    E[:, :3], E[:, 3:6], E[:, 6] = u, r, float(grip)
    return E


def to_codec(E, q, L, U, tol=1e-5):
    """Команда среды [8, 7] -> формат кодека (обратный k29_policy.scale_exec:
    рука / q, схват −g), клип в пределы кодека; round-trip через scale_exec.
    -> (codec float32, clipped)."""
    import k29_policy as kp
    E = np.asarray(E, np.float64)
    q = np.asarray(q, np.float64)
    c = np.zeros((H8, 7), np.float64)
    c[:, :6] = E[:, :6] / q[:6]
    c[:, 6] = -E[:, 6]
    cl = np.clip(c[:, :6], np.asarray(L) + 1e-6, np.asarray(U) - 1e-6)
    clipped = bool(np.abs(cl - c[:, :6]).max() > 0)
    c[:, :6] = cl
    codec = c.astype(np.float32)
    back = kp.scale_exec(codec[None], q)[0]
    if not set(np.unique(codec[:, 6]).tolist()) <= {-1.0, 1.0}:
        raise SystemExit("схват опции не ±1")
    if not clipped and np.abs(back - E).max() > tol:
        raise SystemExit(f"round-trip масштаба: {np.abs(back - E).max():.2e}")
    return codec, clipped


# --- наблюдатель ----------------------------------------------------------------------------------
class Observer:
    """Онлайн-детектор промаха захвата. Питается только из обёртки
    sim.step (start — наблюдение после ожидания, step — исполненное
    действие и полученное состояние); логика читает только буфер последних
    cfg["buffer"] точек. Границы чанков — каждые 8 управляющих шагов.
    force_tau — техническая развилка на этой границе (smoke/самопроверка)."""

    def __init__(self, cfg, force_tau=None):
        self.cfg = cfg
        self.force_tau = None if force_tau is None else int(force_tau)
        self.buf = deque(maxlen=int(cfg["buffer"]))
        self.frames = deque(maxlen=int(cfg["n_post"]) + 1)
        self.cmd = deque(maxlen=int(cfg["buffer"]))
        self.t = -1
        self.events, self.trigger, self.refusals = [], None, []
        self.watch = None
        self.search_from = 0
        self.stopped = False
        self.n_held = 0

    def pt(self, t):
        for b in reversed(self.buf):
            if b["t"] == t:
                return b
        raise SystemExit(f"наблюдатель: точки {t} нет в буфере")

    def g_cmd(self, i):
        """Физическая команда схвата шага i (до начала — open ожидания)."""
        if i < 0:
            return OPEN
        for t, g in reversed(self.cmd):
            if t == i:
                return g
        raise SystemExit(f"наблюдатель: команды {i} нет в буфере")

    def start(self, state, frame=None):
        self.t = 0
        s = st9(state)
        self.buf.append(dict(t=0, a=None, pos=s[2:5].copy(),
                             quat=s[5:9].copy(), fing=s[:2].copy()))
        self.frames.append((0, frame))

    def step(self, a, state, done=False, frame=None):
        if self.stopped or self.t < 0:
            return
        a = np.asarray(a, np.float64).reshape(-1)
        s = st9(state)
        self.cmd.append((self.t, float(a[6])))
        self.t += 1
        self.buf.append(dict(t=self.t, a=a.copy(), pos=s[2:5].copy(),
                             quat=s[5:9].copy(), fing=s[:2].copy()))
        self.frames.append((self.t, frame))
        if done:
            self.stopped = True
            if self.watch is not None:
                self._resolve("done", "эпизод завершён в окне")
            return
        self._detect()
        if self.t % H8 == 0:
            self._boundary(self.t)

    def _detect(self):
        if self.watch is not None:
            return
        npre, npost = int(self.cfg["n_pre"]), int(self.cfg["n_post"])
        g = self.t - npost
        if g < self.search_from:
            return
        if all(self.g_cmd(i) == OPEN for i in range(g - npre, g)) and \
                all(self.g_cmd(i) == CLOSE for i in range(g, g + npost)):
            p0 = self.pt(g)
            lo = max(0, g - int(self.cfg["approach_back"]))
            ap_pt = max((self.pt(i) for i in range(lo, g + 1)),
                        key=lambda b: (b["pos"][2], -b["t"]))
            fr = next((f for t, f in self.frames if t == g), None)
            self.watch = dict(idx=len(self.events) + 1, g=g,
                              detected=self.t, pot_guess=self.n_held + 1,
                              close_pos=p0["pos"].tolist(),
                              close_quat=p0["quat"].tolist(),
                              approach_pos=ap_pt["pos"].tolist())
            self.frame_close = fr

    def _resolve(self, outcome, reason, b=None, extra=None):
        ev = dict(self.watch, outcome=outcome, reason=reason,
                  resolved=self.t if b is None else b, **(extra or {}))
        self.events.append(ev)
        if outcome == "held":
            self.n_held += 1
        self.search_from = ev["g"] + 1
        self.watch = None
        return ev

    def _fire(self, ev, b, forced, frame_fork):
        if MAX_STEPS - b < int(self.cfg["min_left"]):
            self.refusals.append(dict(event=ev["idx"], tau=b,
                                      reason="мало шагов"))
            return
        self.trigger = dict(tau=b, g=ev["g"], event=ev["idx"],
                            pot_guess=ev["pot_guess"], forced=bool(forced),
                            close_pos=ev["close_pos"],
                            close_quat=ev["close_quat"],
                            approach_pos=ev["approach_pos"],
                            lift=ev.get("lift"), ap=ev.get("ap"))
        self.trigger_frames = dict(close=getattr(self, "frame_close", None),
                                   fork=frame_fork)

    def _boundary(self, b):
        cfg = self.cfg
        fr_b = self.frames[-1][1]
        if self.force_tau is not None and b == self.force_tau and \
                self.trigger is None:
            if self.watch is not None:
                ev = dict(self.watch)
            elif self.events:
                ev = dict(self.events[-1])
            else:
                p = self.pt(b)
                lo = max(0, b - int(cfg["approach_back"]))
                ap_pt = max((self.pt(i) for i in range(lo, b + 1)),
                            key=lambda x: (x["pos"][2], -x["t"]))
                ev = dict(idx=0, g=b, pot_guess=self.n_held + 1,
                          close_pos=p["pos"].tolist(),
                          close_quat=p["quat"].tolist(),
                          approach_pos=ap_pt["pos"].tolist())
                self.frame_close = fr_b
            self._fire(ev, b, True, fr_b)
        w = self.watch
        if w is None:
            return
        g = w["g"]
        if b < w["detected"]:
            return
        last = b + H8 > g + int(cfg["window"])
        if any(self.g_cmd(i) != CLOSE for i in range(g, b)):
            self._resolve("released", "схват открыт в окне", b)
            return
        lift = float(self.pt(b)["pos"][2] - self.pt(g)["pos"][2])
        K = int(cfg["k_ap"])
        aps = [float(self.pt(t)["fing"][0] - self.pt(t)["fing"][1])
               for t in range(b - K + 1, b + 1)]
        extra = dict(lift=lift, ap=aps)
        if lift >= cfg["lift_m"] and max(aps) <= cfg["empty_max"]:
            ev = self._resolve("miss", "подтверждённый промах", b, extra)
            if self.trigger is None:
                self._fire(ev, b, False, fr_b)
            return
        if lift >= cfg["lift_m"] and min(aps) >= cfg["held_min"]:
            self._resolve("held", "захват", b, extra)
            return
        if last:
            why = "нет подъёма" if lift < cfg["lift_m"] else \
                "неопределённая апертура"
            self._resolve("uncertain", why, b, extra)

    def export(self):
        return jsonable(dict(events=self.events, trigger=self.trigger,
                             refusals=self.refusals, force_tau=self.force_tau,
                             pending=self.watch))

    @classmethod
    def replay(cls, cfg, S, A, done, force_tau=None):
        """Повтор по записанной траектории (S [n+1, 9], A [n, 7])."""
        o = cls(cfg, force_tau)
        o.start(S[0])
        for i in range(len(A)):
            o.step(A[i], S[i + 1], done=bool(done and i == len(A) - 1))
        return o.export()


class RecSim:
    """Тонкая обёртка sim: снимает переданные действия и полученные
    состояния (кадры — в кольцо наблюдателя), ничего не меняет."""

    def __init__(self, sim, observer):
        self.sim, self.ob = sim, observer
        self.obs = None

    def reset(self, task, starts):
        self.n = 0
        self.S, self.A = [], []
        self.obs = self.sim.reset(task, starts)
        return self.obs

    def step(self, actions):
        obs, r, d, s = self.sim.step(actions)
        self.obs = obs
        self.n += 1
        st = np.asarray(obs["state"], np.float64)[0]
        fr = np.asarray(obs["agentview_image"])[0]
        if self.n == WAIT:
            self.S.append(st.copy())
            self.ob.start(st, fr)
        elif self.n > WAIT:
            self.A.append(np.asarray(actions, np.float64)[0].copy())
            self.S.append(st.copy())
            self.ob.step(actions[0], st, bool(d[0]), fr)
        return obs, r, d, s

    def close(self):
        return self.sim.close()


def frames_of(obs):
    return np.stack([np.asarray(obs["agentview_image"])[0],
                     np.asarray(obs["robot0_eye_in_hand_image"])[0]]
                    ).astype(np.uint8)


# --- локализатор ------------------------------------------------------------------------------------
class NullLocalizer:
    """Честная заглушка: калибровки нет — всегда «неопределённо»."""
    kind = "null"
    offset = np.zeros(2)

    def locate(self, rgb, ref_xy):
        return dict(ok=False, why="калибровка недоступна", xy=None,
                    conf=None, second=None, px=None)

    def describe(self):
        return dict(kind=self.kind)


class RecordedLocalizer:
    """Повтор записанного ответа (проверка шарда)."""

    def __init__(self, rec, offset):
        self.rec, self.offset = rec, np.asarray(offset, np.float64)
        self.kind = "recorded"

    def locate(self, rgb, ref_xy):
        return dict(self.rec)


def project(P, xyz, H, flip_v):
    """Мир -> (row, col) кадра наблюдения. P [3, 4] (u = столбец, v =
    строка сырого рендера, как robosuite camera_utils)."""
    w = np.asarray(P, np.float64) @ np.append(np.asarray(xyz, np.float64), 1)
    u, v = w[0] / w[2], w[1] / w[2]
    return (H - 1 - v if flip_v else v), u


def backproject(P, row, col, z, H, flip_v):
    """(row, col) кадра -> мировые (x, y) на плоскости высоты z."""
    P = np.asarray(P, np.float64)
    v = H - 1 - row if flip_v else row
    u = col
    r1, r2 = P[0] - u * P[2], P[1] - v * P[2]
    A = np.array([[r1[0], r1[1]], [r2[0], r2[1]]])
    bb = -np.array([r1[2] * z + r1[3], r2[2] * z + r2[3]])
    return np.linalg.solve(A, bb)


def ncc_map(win, tpl):
    """Нормированная взаимная корреляция всех положений шаблона в окне."""
    from numpy.lib.stride_tricks import sliding_window_view
    W = sliding_window_view(win, tpl.shape)
    Wc = W - W.mean(axis=(-1, -2), keepdims=True)
    Tc = tpl - tpl.mean()
    num = (Wc * Tc).sum(axis=(-1, -2))
    den = np.sqrt((Wc ** 2).sum(axis=(-1, -2)) * (Tc ** 2).sum()) + 1e-9
    return num / den


class TemplateLocalizer:
    """Сопоставление RGB-шаблонов в agentview около ожидаемой точки:
    ориентир — пиксель максимума NCC, луч на плоскость высоты ориентира.
    Неопределённость: низкий максимум, два близких максимума, максимум на
    краю области поиска, точка вне кадра."""
    kind = "template"

    def __init__(self, P, flip_v, z_plane, offset, templates, conf_min=0.6,
                 margin_min=0.05, radius_px=24, H=224, W=224):
        self.P = np.asarray(P, np.float64)[:3]
        self.flip_v, self.z = bool(flip_v), float(z_plane)
        self.offset = np.asarray(offset, np.float64)
        self.tpl = [np.asarray(t, np.float64).mean(-1) for t in templates]
        if not self.tpl:
            raise SystemExit("локализатор: нет шаблонов")
        self.conf_min, self.margin_min = float(conf_min), float(margin_min)
        self.R, self.H, self.W = int(radius_px), int(H), int(W)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z["P"], bool(z["flip_v"]), float(z["z_plane"]),
                   z["offset"], list(z["templates"]), float(z["conf_min"]),
                   float(z["margin_min"]), int(z["radius_px"]),
                   int(z["H"]), int(z["W"]))

    def describe(self):
        return dict(kind=self.kind, flip_v=self.flip_v, z_plane=self.z,
                    offset=self.offset.tolist(), n_templates=len(self.tpl),
                    conf_min=self.conf_min, margin_min=self.margin_min,
                    radius_px=self.R)

    def locate(self, rgb, ref_xy):
        no = dict(ok=False, xy=None, conf=None, second=None, px=None)
        if rgb is None:
            return dict(no, why="нет кадра")
        img = np.asarray(rgb, np.float64)
        if img.shape[:2] != (self.H, self.W):
            return dict(no, why=f"кадр {img.shape} ≠ калибровке")
        g = img.mean(-1)
        r0, c0 = project(self.P, [ref_xy[0], ref_xy[1], self.z], self.H,
                         self.flip_v)
        th, tw = self.tpl[0].shape
        if any(t.shape != (th, tw) for t in self.tpl):
            return dict(no, why="шаблоны разной формы")
        top = int(round(r0)) - self.R - th // 2
        left = int(round(c0)) - self.R - tw // 2
        hgt, wid = 2 * self.R + th, 2 * self.R + tw
        if top < 0 or left < 0 or top + hgt > self.H or left + wid > self.W:
            return dict(no, why="область поиска вне кадра")
        win = g[top:top + hgt, left:left + wid]
        S = np.max(np.stack([ncc_map(win, t) for t in self.tpl]), 0)
        i, j = np.unravel_index(int(np.argmax(S)), S.shape)
        best = float(S[i, j])
        M = S.copy()
        M[max(0, i - th // 2):i + th // 2 + 1,
          max(0, j - tw // 2):j + tw // 2 + 1] = -np.inf
        second = float(M.max()) if np.isfinite(M).any() else -1.0
        row, col = top + i + th // 2, left + j + tw // 2
        px = [int(row), int(col)]
        base = dict(conf=best, second=second, px=px)
        if best < self.conf_min:
            return dict(no, **base, why="низкая уверенность")
        if best - second < self.margin_min:
            return dict(no, **base, why="два близких максимума")
        if i in (0, S.shape[0] - 1) or j in (0, S.shape[1] - 1):
            return dict(no, **base, why="максимум на краю области")
        xy = backproject(self.P, row, col, self.z, self.H, self.flip_v)
        return dict(ok=True, xy=xy.tolist(), why="", **base)


def load_localizer(plan_loc):
    if plan_loc is None or plan_loc.get("kind") == "null":
        return NullLocalizer()
    if sha256(plan_loc["path"]) != plan_loc["sha256"]:
        raise SystemExit("калибровка локализатора ≠ плану")
    return TemplateLocalizer.load(plan_loc["path"])


def object_guard(loc, frame_close, frame_fork, old_xy, cfg):
    """Защита по объекту до отвода (только RGB agentview): ориентир цели
    локализуется на кадре закрытия g и на кадре развилки; вмешательство
    разрешено, только если оба уверены и ориентир сдвинулся не более
    guard_m. Неопределённость — отказ."""
    ref = np.asarray(old_xy, np.float64) - np.asarray(loc.offset)
    r0 = loc.locate(frame_close, ref)
    r1 = loc.locate(frame_fork, ref)
    out = dict(close=jsonable(r0), fork=jsonable(r1), kind=loc.kind,
               shift_m=None)
    if not r0.get("ok") or not r1.get("ok"):
        why = r0.get("why") if not r0.get("ok") else r1.get("why")
        return dict(out, ok=False, why=f"объект не локализован: {why}")
    sh = float(np.linalg.norm(np.asarray(r1["xy"]) - np.asarray(r0["xy"])))
    out["shift_m"] = sh
    if sh > cfg["guard_m"]:
        return dict(out, ok=False, why="объект сдвинулся")
    return dict(out, ok=True, why="")


def fork_of(meta):
    """Действующая развилка эпизода T32: триггер наблюдателя, прошедший
    защиту по объекту (принудительные развилки smoke — без защиты)."""
    tr = meta["observer"]["trigger"]
    if tr is None:
        return None
    g = meta.get("guard")
    if tr.get("forced") or (g is not None and g["ok"]):
        return tr
    return None


# --- опция повтора ---------------------------------------------------------------------------------
class RetryOption:
    """До 6 чанков по 8 шагов: отвод вверх, отвод к точке подхода,
    выравнивание над целью, подход, закрытие (если поза достигнута),
    подъём. Цель выбирается на наблюдении после отвода. Null — пустая
    опция (проверка тождества T32)."""

    def __init__(self, trig, cfg, branch, loc, q, L, U, held_min):
        self.cfg, self.branch, self.loc = cfg, branch, loc
        self.q, self.L, self.U, self.held_min = q, L, U, held_min
        self.n = 0 if branch == "Null" else len(PHASES)
        self.cp = np.asarray(trig["close_pos"], np.float64)
        self.cq = np.asarray(trig["close_quat"], np.float64)
        self.ap = np.asarray(trig["approach_pos"], np.float64)
        self.z_clear = max(float(self.ap[2]), float(self.cp[2]) +
                           cfg["clear_m"])
        self.i, self.steps, self.log = 0, 0, []
        self.frames = {}
        self.tgt = None
        self.info = dict(branch=branch, old_xy=self.cp[:2].tolist(),
                         target_xy=None, live=None, fallback=None,
                         shift_m=None, stop=None, retreat_hash=None,
                         result=None, reached=None, z_clear=self.z_clear)
        self.finished = False

    def _stop(self, why):
        self.info["stop"] = why
        self.info["phases_done"] = self.i
        self.finished = True
        return None

    def _choose(self, obs):
        old = self.cp[:2]
        if self.branch != "RetryLive":
            self.tgt = old.copy()
            return
        rgb = None if obs is None else np.asarray(obs["agentview_image"])[0]
        r = self.loc.locate(rgb, old - self.loc.offset)
        live = jsonable(dict(r, offset=np.asarray(self.loc.offset).tolist()))
        self.info["live"] = live
        if not r.get("ok"):
            self.info["fallback"] = r.get("why") or "неопределённо"
            self.tgt = old.copy()
            return
        t = np.asarray(r["xy"], np.float64) + self.loc.offset
        sh = float(np.linalg.norm(t - old))
        self.info["shift_m"] = sh
        if sh > self.cfg["max_shift_m"]:
            self.info["fallback"] = "смещение > предела"
            self.tgt = old.copy()
            return
        self.tgt = t

    def next(self, state, elapsed, obs=None):
        """Наблюдение на границе -> codec [8, 7] | None (возврат T32)."""
        if self.finished:
            return None
        s = st9(state)
        p, qc, ap = s[2:5], s[5:9], aperture(s)
        if self.log:
            self.log[-1]["end_pos"] = p.tolist()
        i = self.i
        if i >= self.n:
            if self.n:
                self.info["result"] = dict(
                    ap=ap, lift=float(p[2] - self.cp[2]),
                    held=bool(ap >= self.held_min))
                if obs is not None:
                    self.frames["end"] = frames_of(obs)
            return self._stop("завершена")
        if MAX_STEPS - int(elapsed) < H8 * (self.n - i):
            return self._stop("лимит шагов")
        if PHASES[i] in ("retreat_back", "align") and \
                float(p[2]) < self.z_clear - self.cfg["clear_tol"]:
            self.info["clearance"] = dict(phase=PHASES[i], z=float(p[2]),
                                          z_clear=self.z_clear)
            if obs is not None:
                self.frames["end"] = frames_of(obs)
            return self._stop("зазор не достигнут")
        if i == 2:
            if obs is not None:
                import k31_collect as kcol
                self.info["retreat_hash"] = kcol.obs_hash(obs)
                self.frames["retreat"] = frames_of(obs)
            self._choose(obs)
            self.info["target_xy"] = self.tgt.tolist()
        if i == 4:
            exy = float(np.linalg.norm(p[:2] - self.tgt))
            ez = float(abs(p[2] - self.cp[2]))
            self.info["reached"] = dict(err_xy=exy, err_z=ez,
                                        err_rot=float(np.linalg.norm(
                                            rot_err(self.cq, qc))))
            if exy > self.cfg["tol_xy"] or ez > self.cfg["tol_z"]:
                if obs is not None:
                    self.frames["end"] = frames_of(obs)
                return self._stop("поза не достигнута")
        xy = p[:2] if i == 0 else (self.ap[:2] if i == 1 else self.tgt)
        z = dict(retreat_up=max(self.z_clear, float(p[2])),
                 retreat_back=self.z_clear, align=self.z_clear,
                 approach=float(self.cp[2]), close=float(self.cp[2]),
                 lift=float(self.cp[2]) + self.cfg["lift_opt_m"])[PHASES[i]]
        T = np.array([xy[0], xy[1], z])
        grip = CLOSE if PHASES[i] in ("close", "lift") else OPEN
        E = pctl(p, qc, T, self.cq, grip, self.cfg)
        codec, clipped = to_codec(E, self.q, self.L, self.U)
        self.log.append(dict(phase=PHASES[i], elapsed=int(elapsed),
                             start_pos=p.tolist(), target=T.tolist(),
                             cmd=E[0, :6].tolist(), clipped=clipped,
                             ap=ap))
        self.i += 1
        return codec

    def executed(self, k):
        self.steps += int(k)
        self.log[-1]["k"] = int(k)


# --- агент -------------------------------------------------------------------------------------------
class RecoveryArm:
    """Агент для run_episode поверх штатного Arm: состояние опции живёт
    здесь и сбрасывается в begin. T32 — только наблюдение; ветви —
    опция на границе триггера, затем штатный T32. expect = dict(tau,
    hash) — развилка из эпизода T32."""

    def __init__(self, arm, rs, C, cfg, branch, loc, expect=None):
        if branch not in BRANCHES + ("Null",):
            raise ValueError(branch)
        self.arm, self.rs, self.C, self.cfg = arm, rs, C, cfg
        self.branch, self.loc, self.expect = branch, loc, expect
        self.begin_state()

    def begin_state(self):
        self.heavy, self.log = 0, []
        self.opt, self.started, self.fork_hash = None, False, None
        self.cur, self.guard = None, None

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
        self.begin_state()
        self.arm.begin(task, init, episode)

    def _guard(self, tr):
        ob = self.rs.ob
        fr = getattr(ob, "trigger_frames", {}) or {}
        self.guard = object_guard(self.loc, fr.get("close"), fr.get("fork"),
                                  tr["close_pos"][:2], self.cfg)

    def _start(self, elapsed):
        import k31_collect as kcol
        tr = self.rs.ob.trigger
        h = kcol.obs_hash(self.rs.obs)
        if self.expect is not None and (int(self.expect["tau"]) != elapsed or
                                        self.expect["hash"] != h):
            raise kcol.Mismatch(f"развилка ветви ({elapsed}) ≠ T32 "
                                f"({self.expect['tau']}) или хеш")
        self._guard(tr)
        if not tr.get("forced") and not self.guard["ok"]:
            raise kcol.Mismatch(f"ветвь при отказе защиты: "
                                f"{self.guard['why']}")
        if self.expect is not None and "guard_ok" in self.expect and \
                bool(self.expect["guard_ok"]) != bool(self.guard["ok"]):
            raise kcol.Mismatch("защита по объекту ≠ эпизоду T32")
        self.fork_hash, self.started = h, True
        self.fork_frames = frames_of(self.rs.obs)
        self.opt = RetryOption(tr, self.cfg, self.branch, self.loc,
                               self.C["q"], self.C["L"], self.C["U"],
                               self.cfg["held_min"])

    def decide(self, batch, po, s8, elapsed):
        import k31_collect as kcol
        tr = self.rs.ob.trigger
        if self.branch == "T32" and tr is not None and self.guard is None \
                and tr["tau"] == elapsed:
            self._guard(tr)
        if self.branch != "T32" and not self.started and tr is not None:
            if tr["tau"] != elapsed:
                raise kcol.Mismatch(f"триггер ветви {tr['tau']} ≠ границе "
                                    f"{elapsed}")
            self._start(elapsed)
        if self.branch != "T32" and not self.started and \
                self.expect is not None and elapsed >= self.expect["tau"]:
            raise kcol.Mismatch("триггер не воспроизвёлся на префиксе")
        if self.opt is not None and not self.opt.finished:
            st = np.asarray(self.rs.obs["state"], np.float64)[0]
            ch = self.opt.next(st, elapsed, self.rs.obs)
            if ch is not None:
                self.cur = "opt"
                self.log.append(dict(kind="opt", elapsed=int(elapsed),
                                     codec=ch))
                return ch, dict(option=True, phase=self.opt.log[-1]["phase"],
                                elapsed=int(elapsed))
        self.cur = "t32"
        fwd = self.arm.forward

        def counted(*x, **kw):
            self.heavy += 1
            return fwd(*x, **kw)
        self.arm.forward = counted
        try:
            B, rec = self.arm.decide(batch, po, s8, elapsed)
        finally:
            self.arm.forward = fwd
        B = np.asarray(B, np.float32)
        self.log.append(dict(kind="t32", elapsed=int(elapsed), codec=B))
        return B, rec

    def executed(self, k):
        if self.cur == "opt":
            self.opt.executed(k)
        else:
            self.arm.executed(k)
        self.log[-1]["k"] = int(k)


def run_branch(sim, io, arm, C, cfg, loc, branch, task, pos, eid_t32, *,
               prefix=None, expect=None, force_tau=None):
    """Эпизод ветви через k31_collect.run_episode (обёртки sim и агента)."""
    import k31_collect as kcol
    from k36_harness import episode_number
    q_io = np.asarray(getattr(io, "inp", io).max_act_q, np.float64)
    if not np.array_equal(q_io, np.asarray(C["q"], np.float64)):
        raise SystemExit("масштаб io ≠ scale_q плана K-32")
    ob = Observer(cfg, force_tau)
    rs = RecSim(sim, ob)
    ag = RecoveryArm(arm, rs, C, cfg, branch, loc, expect)
    arm.episode = episode_number(eid_t32)
    ep = kcol.run_episode(rs, io, ag, task, pos, prefix=prefix)
    if branch != "T32" and not ag.started:
        raise kcol.Mismatch("развилка не достигнута")
    tau = int(expect["tau"]) if expect is not None else (
        ob.trigger["tau"] if (branch != "T32" and ob.trigger) else 0)
    ep.update(heavy=ag.heavy, observer=ob.export(), branch=branch,
              guard=None if ag.guard is None else jsonable(ag.guard),
              S=np.stack(rs.S), log=ag.log, tau=tau, fork_hash=ag.fork_hash,
              option=None if ag.opt is None else dict(
                  info=ag.opt.info, log=ag.opt.log, steps=ag.opt.steps),
              frames={})
    if branch == "T32" and ob.trigger is not None:
        fr = getattr(ob, "trigger_frames", {})
        ep["frames"] = {k: v for k, v in fr.items() if v is not None}
        h = [c["hash"] for c in ep["calls"] if c["tau"] == ob.trigger["tau"]]
        ep["fork_hash"] = h[0] if h else None
    if branch != "T32" and ag.opt is not None:
        ep["frames"] = dict(ag.opt.frames, fork=ag.fork_frames)
    return ep


# --- шарды ---------------------------------------------------------------------------------------------
SUMMARY = ("success", "success_step", "done_step", "own_steps", "steps",
           "physical", "init_hash", "init_hash_full", "actions_sha1")


def shard_path(eid):
    return SHARDS / (eid.replace("|", "__") + ".npz")


def save_ep(path, j, ep, plan_sha):
    import k32_harness as kh32
    t32 = [c for c in ep["calls"] if not c["rec"].get("option")]
    J = kh32.pack_journal([(0, t32)], 6)
    cl = ep["calls"]
    arr = dict(actions=np.asarray(ep["actions"], np.float32),
               tr_state=np.asarray(ep["S"], np.float64),
               c_tau=np.array([c["tau"] for c in cl], np.int64),
               c_k=np.array([c["k"] for c in cl], np.int64),
               c_opt=np.array([bool(c["rec"].get("option")) for c in cl],
                              bool),
               c_codec=np.stack([c["a_codec"] for c in cl]).astype(np.float32)
               if cl else np.zeros((0, H8, 7), np.float32),
               c_hash=np.array([c["hash"] for c in cl], dtype="<U40"))
    for k, v in ep["frames"].items():
        arr[f"f_{k}"] = np.asarray(v, np.uint8)
    meta = dict(eid=j["eid"], plan=plan_sha, branch=j["branch"],
                heavy=int(ep["heavy"]), seconds=float(ep["seconds"]),
                summary={k: ep[k] for k in SUMMARY}, tau=int(ep["tau"]),
                fork_hash=ep["fork_hash"], observer=ep["observer"],
                option=jsonable(ep["option"]), guard=ep.get("guard"),
                prefix_heavy=int(j.get("prefix_heavy", 0)))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, meta=np.array(json.dumps(meta)),
                            **arr, **{f"j_{k}": v for k, v in J.items()})
    tmp.replace(path)
    return meta


def read_shard(path):
    with np.load(path, allow_pickle=False) as z:
        zz = dict(z)
    return zz, json.loads(str(zz["meta"]))


def summary_problems(s, n_actions):
    p = []
    ss, ds, st_ = s["success_step"], s["done_step"], s["steps"]
    if s["physical"] != WAIT + n_actions or st_ != n_actions or \
            s["physical"] > EP:
        p.append("расход ≠ шагам")
    if s["success"]:
        if not (0 <= ss < st_ and (ds < 0 or ss <= ds)):
            p.append("success_step несогласован")
    elif ss != -1:
        p.append("success_step при неуспехе")
    if ds >= 0:
        if ds + 1 != st_ or s["own_steps"] != st_:
            p.append("done_step/steps несогласованы")
    elif st_ != MAX_STEPS:
        p.append("без terminal эпизод не дошёл до лимита")
    return p


def t32_calls_problems(J, C):
    """Каждый вызов T32 по отдельности — штатная проверка журнала K-32
    (в ветвях последовательность вызовов прерывается опцией)."""
    import k29_policy as kp
    import k32_harness as kh32
    n = len(J.get("ep", []))
    if n == 0:
        return []
    J2 = dict(J, ep=np.arange(n, dtype=np.int64),
              elapsed=np.zeros(n, np.int64))
    A = [kp.scale_exec(J["a_codec"][i][None], C["q"])[0][:int(J["k"][i])]
         for i in range(n)]
    return kh32.journal_problems(J2, A, [-1] * n, C["half"], C["mid"],
                                 C["L"], C["U"], C["q"], "T")


def trace_problems(zz, meta, cfg):
    """Траектория наблюдателя: форма, конечность, повтор решения."""
    S, A = zz["tr_state"], zz["actions"]
    if S.shape != (len(A) + 1, 9) or not np.isfinite(S).all():
        return [f"траектория формы {S.shape}"]
    ob = meta["observer"]
    re_ = Observer.replay(cfg, S, A, meta["summary"]["done_step"] >= 0,
                          ob.get("force_tau"))
    if re_ != ob:
        return ["повтор наблюдателя ≠ записи"]
    return []


def t32_problems(zz, meta, j, C, cfg):
    import k40b_eval_base as kb40
    p = kb40.ep_problems(zz, meta, j, C)
    if p and p[0] == "действия ≠ SHA":
        return p
    p += trace_problems(zz, meta, cfg)
    if not np.array_equal(zz["c_opt"], np.zeros(len(zz["c_opt"]), bool)):
        p.append("опция в T32")
    if len(zz["c_tau"]) != len(zz["j_ep"]) or \
            not np.array_equal(zz["c_codec"], zz["j_a_codec"]):
        p.append("таблица вызовов ≠ журналу")
    tr = meta["observer"]["trigger"]
    if tr is not None:
        idx = np.nonzero(zz["c_tau"] == tr["tau"])[0]
        if len(idx) != 1 or str(zz["c_hash"][idx[0]]) != meta["fork_hash"]:
            p.append("хеш развилки ≠ границе вызова")
    elif meta["fork_hash"] is not None:
        p.append("хеш развилки без триггера")
    return p


def option_replay(meta, zz, cfg, C):
    """Пересборка команд опции по состояниям границ (цель RetryLive — по
    записанному ответу локализатора)."""
    o = meta["option"]
    tr = meta["observer"]["trigger"]
    live = o["info"].get("live")
    loc = RecordedLocalizer(live or {}, (live or {}).get("offset", [0, 0]))
    R = RetryOption(tr, cfg, meta["branch"], loc, C["q"], C["L"], C["U"],
                    cfg["held_min"])
    out = []
    for i in np.nonzero(zz["c_opt"])[0]:
        t = int(zz["c_tau"][i])
        ch = R.next(zz["tr_state"][t], t)
        if ch is None:
            return None
        out.append(ch)
        R.executed(int(zz["c_k"][i]))
    return out


def branch_problems(zz, meta, j, C, cfg, z0, m0):
    """Ветвь восстановления против своего эпизода T32 (z0, m0)."""
    import k31_collect as kcol
    import k29_policy as kp
    p = []
    A, s = zz["actions"], meta["summary"]
    if kcol.actions_sha(A) != s["actions_sha1"]:
        return ["действия ≠ SHA"]
    if [s["init_hash"], s["init_hash_full"]] != list(j["scene"]["hash"]):
        p.append("начальный хеш ≠ банку")
    tr0 = m0["observer"]["trigger"]
    if tr0 is None:
        return p + ["ветвь без триггера T32"]
    if fork_of(m0) is None:
        return p + ["ветвь при отказе защиты по объекту в T32"]
    if meta.get("guard") != m0.get("guard"):
        p.append("защита по объекту ветви ≠ T32")
    tau = int(meta["tau"])
    if tau != int(tr0["tau"]) or meta["fork_hash"] != m0["fork_hash"]:
        p.append("развилка ≠ T32")
    if not np.array_equal(A[:tau], z0["actions"][:tau]) or \
            not np.array_equal(zz["tr_state"][:tau + 1],
                               z0["tr_state"][:tau + 1]):
        p.append("префикс ≠ T32")
    p += trace_problems(zz, meta, cfg)
    if meta["observer"]["trigger"] != tr0:
        p.append("триггер ветви ≠ T32")
    ct, ck, co, cc = zz["c_tau"], zz["c_k"], zz["c_opt"], zz["c_codec"]
    n = len(ct)
    if n == 0 or int(ct[0]) != tau or any(int(ct[i]) + int(ck[i]) !=
                                          int(ct[i + 1])
                                          for i in range(n - 1)):
        return p + ["последовательность вызовов"]
    if str(zz["c_hash"][0]) != meta["fork_hash"]:
        p.append("хеш первой границы ≠ развилке")
    ex = np.concatenate([kp.scale_exec(cc[i][None], C["q"])[0][:int(ck[i])]
                         for i in range(n)])
    if not np.array_equal(ex.astype(np.float32), A[tau:]):
        p.append("исполненное ≠ масштабу команд")
    n_opt = int(co.sum())
    if not np.array_equal(co, np.arange(n) < n_opt):
        p.append("вызовы опции не подряд с развилки")
    o = meta["option"]
    if o is None or len(o["log"]) != n_opt or n_opt > len(PHASES) or \
            int(ck[co].sum()) != int(o["steps"]) or o["steps"] > 48:
        p.append("журнал опции ≠ вызовам или > 48 шагов")
    if n_opt:
        oc = cc[co]
        if not set(np.unique(oc[..., 6]).tolist()) <= {-1.0, 1.0} or \
                (oc[..., :6] < C["L"] - 1e-4).any() or \
                (oc[..., :6] > C["U"] + 1e-4).any():
            p.append("команды опции вне пределов")
        if o is not None:
            rb = option_replay(meta, zz, cfg, C)
            if rb is None or not all(np.array_equal(a, b)
                                     for a, b in zip(rb, oc)):
                p.append("команды опции ≠ пересборке")
    J = {k[2:]: v for k, v in zz.items() if k.startswith("j_")}
    nt = n - n_opt
    if len(J.get("ep", [])) != nt or (nt and (
            not np.array_equal(J["a_codec"], cc[~co]) or
            not np.array_equal(J["k"], ck[~co]))):
        p.append("журнал T32 ≠ таблице вызовов")
    else:
        p += [f"журнал: {x}" for x in t32_calls_problems(J, C)[:3]]
    if meta["heavy"] != nt:
        p.append(f"тяжёлых вызовов {meta['heavy']} ≠ T32-чанков {nt}")
    if meta["prefix_heavy"] != int((z0["c_tau"] < tau).sum()):
        p.append("тяжёлые вызовы префикса ≠ T32")
    p += summary_problems(s, len(A))
    return p


def check_shard(j, plan_sha, C, cfg, done="_no", t32=None):
    """-> (meta, проблемы); t32 = (zz, meta) эпизода T32 сцены для ветви."""
    path = shard_path(j["eid"])
    if not path.exists():
        return None, ["нет шарда"]
    zz, meta = read_shard(path)
    if meta.get("eid") != j["eid"] or meta.get("plan") != plan_sha or \
            meta.get("branch") != j["branch"]:
        return meta, ["шард чужого эпизода или плана"]
    if j["branch"] == "T32":
        p = t32_problems(zz, meta, j, C, cfg)
    else:
        if t32 is None:
            return meta, ["нет эпизода T32 для проверки ветви"]
        p = branch_problems(zz, meta, j, C, cfg, t32[0], t32[1])
    if done != "_no" and done != dict(scope="base",
                                      actual=meta["summary"]["physical"]):
        p.append(f"журнал {done} ≠ записи")
    return meta, p


def pair_problems(mo, ml, zo, zl):
    """RetryOld и RetryLive одной сцены: общий отвод; при возврате к
    прежней цели — побитово одинаковые эпизоды."""
    p = []
    if mo["tau"] != ml["tau"] or mo["fork_hash"] != ml["fork_hash"]:
        p.append("разные развилки")
    ro = (mo["option"] or {}).get("info", {}).get("retreat_hash")
    rl = (ml["option"] or {}).get("info", {}).get("retreat_hash")
    if (ro is None) != (rl is None) or ro != rl:
        p.append("хеш после отвода различается")
    io_, il = mo["option"]["info"], ml["option"]["info"]
    if il.get("target_xy") is not None and \
            il["target_xy"] == io_.get("target_xy"):
        if not np.array_equal(zo["actions"], zl["actions"]) or \
                mo["summary"] != ml["summary"]:
            p.append("одинаковая цель, но эпизоды различаются")
    return p


# --- план ------------------------------------------------------------------------------------------
def plan_path():
    return ROOT / "plan.json"


def calib_entry():
    rp = CALIB.with_suffix(".json")
    if not CALIB.exists() or not rp.exists():
        return dict(kind="null", why="калибровки нет")
    r = json.load(open(rp, encoding="utf-8"))
    if not r.get("usable"):
        return dict(kind="null", why=f"калибровка непригодна: "
                    f"{r.get('why')}", report_sha256=sha256(rp))
    return dict(kind="template", path=str(CALIB), sha256=sha256(CALIB),
                report_sha256=sha256(rp), errors=r.get("errors"))


def prepare(a):
    import k36_protocol as pc
    import k37_confirm as kcf
    loc_e = calib_entry()
    if loc_e["kind"] == "null":
        raise SystemExit(f"локализатор null ({loc_e.get('why')}): RetryLive "
                         f"совпадёт с RetryOld, защита по объекту откажет "
                         f"везде — prepare отказан")
    live_p, live_sha = live_admission(loc_e)
    m, pilot, dev = scenes()
    k36 = pc.load_prepare()
    cfg = load_cfg()
    plan = dict(kind="k42_recovery_plan", protocol=PROTOCOL,
                bank=dict(bank_id=m["bank_id"], files=m["files"],
                          space=SPACE, dir=BANK_DIR,
                          manifest_sha256=sha256(MANIFEST)),
                pilot=pilot, dev=dev, config=cfg,
                config_sha256=sha256(CONFIG) if CONFIG.exists() else None,
                localizer=loc_e,
                live_smoke=dict(path=live_p, sha256=live_sha),
                k36_prepare_sha1=k36["sha1"],
                t32=dict(actor=k36["k32"]["seeds"][str(SEED)]["T"][
                    "actor_hash"], source=k36["k32"]["seeds"][str(SEED)][
                    "source"]["weights_hash"], seed=SEED),
                code={f: sha256(os.path.join(HERE, f)) for f in CODE},
                env=dict(B=1, exec_seed=101, waiting_steps=WAIT,
                         max_steps=MAX_STEPS, horizon=H8, threads=8),
                rule=RULE, boot=BOOT, parts={})
    for k in range(N_PARTS):
        n = sum(1 for x in pilot if x["part"] == k)
        plan["parts"][str(k)] = dict(n_scenes=n, cap=(3 * n + 1) * EP)
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
    print(f"  план {p}: SHA {sh[:16]}; пилот {len(pilot)} сцен t{TASK}, dev "
          f"{len(dev)}; локализатор {plan['localizer']['kind']}; не более "
          f"{3 * len(pilot)} эпизодов")
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
        if p["localizer"]["kind"] != "null" and \
                sha256(p["localizer"]["path"]) != p["localizer"]["sha256"]:
            raise SystemExit("калибровка локализатора ≠ плану")
    return p


# --- карта, smoke ------------------------------------------------------------------------------------
def card(device):
    import k40b_eval_base as kb40
    return kb40.Card(device)


def controller_job(args):
    """В отдельном процессе (до CUDA): контроллер руки и камера agentview
    свежей среды K33Env (явные проверки API robosuite)."""
    task, space, bank_dir, pos = args
    try:
        import k33_bank as kb
        K = kb.make_env_class()
        env = K(task_id=int(task), image_size=224, k33_mode="bank",
                k33_space=int(space), k33_bank_dir=bank_dir)
        try:
            env.reset(init_state_id=int(pos))
            base = getattr(env, "env", None)
            rs = getattr(base, "env", None)
            robots = getattr(rs, "robots", None)
            if not robots:
                raise RuntimeError("нет env.env.env.robots — другой API")
            c = robots[0].controller
            out = dict(ok=True, cls=type(c).__name__,
                       name=str(getattr(c, "name", None)))
            for k in ("input_max", "input_min", "output_max", "output_min",
                      "use_delta", "control_delta", "kp", "control_ori"):
                v = getattr(c, k, None)
                out[k] = None if v is None else np.asarray(v).tolist()
            out["control_freq"] = float(getattr(rs, "control_freq", -1))
            from robosuite.utils import camera_utils as cu
            out["P_agentview"] = np.asarray(cu.get_camera_transform_matrix(
                base.sim, "agentview", 224, 224))[:3].tolist()
            return out
        finally:
            env.close()
    except Exception as e:                                   # noqa: BLE001
        import traceback
        return dict(ok=False, why=f"{type(e).__name__}: {e}",
                    tb=traceback.format_exc()[-2000:])


def controller_problems(ci, cfg):
    if not ci.get("ok"):
        return [f"контроллер: {ci.get('why')}"]
    p = []
    if "OSC" not in ci["cls"] and "OSC" not in ci["name"]:
        p.append(f"контроллер {ci['cls']}/{ci['name']} — не OSC")
    if ci.get("use_delta") is False or ci.get("control_delta") is False:
        p.append("контроллер не в режиме дельты")
    imax = np.asarray(ci.get("input_max") or [np.nan], np.float64).reshape(-1)
    omax = np.asarray(ci.get("output_max") or [np.nan],
                      np.float64).reshape(-1)
    if imax.size == 1:
        imax = np.full(6, imax[0])
    if omax.size == 1:
        omax = np.full(6, omax[0])
    if not (np.isfinite(imax).all() and np.isfinite(omax).all()):
        p.append("нет input_max/output_max")
    return p


def probe_axes(sim, io, C, cfg, task, pos, u=0.3):
    """Короткая проба на одной сцене (k39_cache.Replay): пустое закрытие и
    знаки/масштаб по осям (+u 8 шагов, −u 8 шагов), команды — через
    to_codec и io.scale (штатный путь)."""
    import k39_cache as kc39
    rp = kc39.Replay(sim, io, task, pos)
    out = dict(empty_ap=[], axes={})

    def run(E):
        codec, cl = to_codec(E, C["q"], C["L"], C["U"])
        ex = io.scale(codec[None])[0]
        traj = []
        for t in range(H8):
            rp.step(ex[t])
            traj.append(st9(np.asarray(rp.obs["state"])[0]))
        return np.stack(traj), cl
    s0 = st9(np.asarray(rp.obs["state"])[0])
    E = np.zeros((H8, 7))
    E[:, 6] = CLOSE
    tr, _ = run(E)
    tr2, _ = run(E)
    out["empty_ap"] = [aperture(s) for s in np.concatenate([tr, tr2])]
    E[:, 6] = OPEN
    run(E)
    for ax in range(3):
        for sg in (1.0, -1.0):
            E = np.zeros((H8, 7))
            E[:, 6] = OPEN
            E[:, ax] = sg * u
            p0 = st9(np.asarray(rp.obs["state"])[0])[2:5]
            tr, cl = run(E)
            d = tr[-1, 2:5] - p0
            out["axes"][f"{'xyz'[ax]}{'+' if sg > 0 else '-'}"] = dict(
                disp=d.tolist(), per_unit_step=float(d[ax] / (sg * u * H8)),
                clipped=cl)
    out["start"] = s0.tolist()
    return out


def live_example(forced, cfg):
    """Есть ли принудительная развилка smoke, где RGB сместил цель
    RetryLive, исполненные действия ≠ RetryOld, а опция RetryLive прошла
    все фазы (поза достигнута, закрытие и подъём исполнены)."""
    out = dict(ok=False, case_id=None, n=len(forced), why="нет развилок")
    for f in forced:
        o, l_ = f.get("RetryOld"), f.get("RetryLive")
        if not o or not l_ or not l_.get("info"):
            continue
        il = l_["info"]
        sh = il.get("shift_m")
        if il.get("fallback") or sh is None or sh < cfg["live_min_m"]:
            out["why"] = f"цель не сменилась ({il.get('fallback') or sh})"
            continue
        if o["actions_sha1"] == l_["actions_sha1"]:
            out["why"] = "действия RetryLive = RetryOld"
            continue
        if il.get("stop") != "завершена" or il.get("phases_done") != \
                len(PHASES):
            out["why"] = f"опция не дошла до подъёма: {il.get('stop')}"
            continue
        return dict(ok=True, case_id=f["case_id"], n=len(forced), why="",
                    shift_m=sh, reached=il.get("reached"),
                    result=il.get("result"))
    return out


def live_admission(loc_entry):
    """Калибровочный smoke (до плана) с этой калибровкой и живым
    примером -> (путь, SHA) | SystemExit."""
    for f in sorted((ROOT / "smoke").glob("*.json")):
        r = json.load(open(f, encoding="utf-8"))
        if r.get("plan") is None and \
                r.get("calib_sha256") == loc_entry["sha256"] and \
                (r.get("live_example") or {}).get("ok") and \
                not r.get("problems"):
            return str(f), sha256(f)
    raise SystemExit("нет smoke с этой калибровкой и живым примером "
                     "RetryLive (цель сменилась, действия ≠ RetryOld, "
                     "опция дошла до подъёма) — prepare отказан")


def smoke_path(c):
    return ROOT / "smoke" / f"{socket.gethostname()}__{c.g['uuid']}.json"


def smoke(a):
    """Технический допуск и калибровочные измерения (до и после prepare)."""
    import multiprocessing as mp
    import k36_harness as kh
    import k36_protocol as pc
    import k38_grip_run as kgr
    import k40b_eval_base as kb40
    import k31_collect as kcol
    k36 = pc.load_prepare()
    C = kh.consts(k36)
    planned = plan_path().exists()
    plan = load_plan() if planned else None
    cfg = plan["config"] if planned else load_cfg()
    loc = load_localizer(plan["localizer"]) if planned else (
        TemplateLocalizer.load(CALIB) if calib_entry()["kind"] != "null"
        else NullLocalizer())
    _m, _pilot, dev = scenes()
    with mp.get_context("forkserver").Pool(1) as pool:
        ci = pool.apply(controller_job, ((TASK, SPACE,
                                          os.path.abspath(BANK_DIR),
                                          dev[0]["pos"]),))
    prob = controller_problems(ci, cfg)
    rep = dict(controller={k: v for k, v in ci.items() if k != "tb"},
               config=cfg, plan=plan["sha256"] if planned else None)
    c = card(a.device)
    try:
        c.start()
        # 1) база: архивный T32 K-40 B (s0, t8) с наблюдателем
        bp = kb40.load_plan(verify=False)
        sc5 = sorted((x for x in bp["screen"] if x["task"] == TASK),
                     key=lambda x: x["rank"])[0]
        e5 = kb40.eid_of("T32", 0, sc5)
        z5, m5 = read_shard(kb40.shard_path(e5))
        sim5 = c.sim(kb40.SPACE, kb40.BANK_DIR)
        v, arm = c.source(k36, SEED)
        io = v["io"]
        ep = run_branch(sim5, io, arm, C, cfg, loc, "T32", TASK, sc5["pos"],
                        e5)
        same = np.array_equal(np.asarray(ep["actions"], np.float32),
                              z5["actions"]) and \
            {k: ep[k] for k in kb40.SUMMARY} == m5["summary"] and \
            ep["heavy"] == m5["heavy"]
        if not same:
            prob.append("T32 с наблюдателем ≠ архиву K-40 B")
        rep["base"] = dict(eid=e5, same=bool(same), observer=ep["observer"],
                           seconds=float(ep["seconds"]))
        sim = c.sim(SPACE, BANK_DIR)
        # 2) проба контроллера и пустого закрытия
        rep["probe"] = probe_axes(sim, io, C, cfg, TASK, dev[0]["pos"])
        for k_, d in rep["probe"]["axes"].items():
            if d["per_unit_step"] <= 0:
                prob.append(f"ось {k_}: перемещение против команды")
        # 3) dev-эпизоды T32 с наблюдателем, технические ветви
        rows, probes = [], []
        SMOKE.mkdir(parents=True, exist_ok=True)
        for sc in dev[:a.n_dev]:
            eid = f"k42-smoke|T32|{sc['case_id']}"
            ep = run_branch(sim, io, arm, C, cfg, loc, "T32", TASK,
                            sc["pos"], eid)
            np.savez_compressed(SMOKE / f"dev_{sc['case_id']}.npz",
                                meta=np.array(json.dumps(dict(
                                    case_id=sc["case_id"], pos=sc["pos"],
                                    observer=ep["observer"],
                                    summary={k: ep[k] for k in SUMMARY}))),
                                actions=np.asarray(ep["actions"], np.float32),
                                tr_state=np.asarray(ep["S"]))
            rows.append(dict(case_id=sc["case_id"], success=ep["success"],
                             observer=ep["observer"],
                             seconds=float(ep["seconds"])))
            evs = ep["observer"]["events"]
            if len(probes) < a.n_probe and evs:
                g = evs[0]["g"]
                tau = H8 * ((g + int(cfg["n_post"]) + H8 - 1) // H8)
                if MAX_STEPS - tau >= cfg["min_left"]:
                    probes.append((sc, ep, tau))
        rep["dev"] = rows
        rep["forced"] = []
        for sc, ep0, tau in probes:
            eid = f"k42-smoke|T32|{sc['case_id']}"
            e0 = run_branch(sim, io, arm, C, cfg, loc, "T32", TASK,
                            sc["pos"], eid, force_tau=tau)
            if not np.array_equal(e0["actions"], ep0["actions"]):
                prob.append(f"{sc['case_id']}: принудительная развилка "
                            f"изменила T32")
            ex = dict(tau=tau, hash=e0["fork_hash"])
            out = dict(case_id=sc["case_id"], tau=tau)
            got = {}
            for br in ("Null", "RetryOld", "RetryLive"):
                try:
                    eb = run_branch(sim, io, arm, C, cfg, loc, br, TASK,
                                    sc["pos"], eid,
                                    prefix=np.asarray(e0["actions"])[:tau],
                                    expect=ex, force_tau=tau)
                except kcol.Mismatch as e:
                    prob.append(f"{sc['case_id']} {br}: {e}")
                    continue
                got[br] = eb
                o = eb["option"] or {}
                mv = []
                for r in o.get("log", []):
                    if "end_pos" in r:
                        want = np.asarray(r["target"]) - \
                            np.asarray(r["start_pos"])
                        did = np.asarray(r["end_pos"]) - \
                            np.asarray(r["start_pos"])
                        mv.append(dict(phase=r["phase"],
                                       want=want.tolist(), did=did.tolist(),
                                       cos=float(want @ did / (
                                           np.linalg.norm(want) *
                                           np.linalg.norm(did) + 1e-12))))
                out[br] = dict(success=eb["success"], info=o.get("info"),
                               guard=eb.get("guard"),
                               actions_sha1=kcol.actions_sha(
                                   np.asarray(eb["actions"], np.float32)),
                               moves=mv, steps=o.get("steps"),
                               heavy=eb["heavy"],
                               t32_after=sum(1 for x in eb["log"]
                                             if x["kind"] == "t32"))
            if "Null" in got and not (
                    np.array_equal(got["Null"]["actions"], e0["actions"]) and
                    got["Null"]["success"] == e0["success"] and
                    got["Null"]["done_step"] == e0["done_step"] and
                    got["Null"]["heavy"] == sum(1 for x in e0["calls"]
                                                if x["tau"] >= tau)):
                prob.append(f"{sc['case_id']}: нулевая опция ≠ T32")
            if "RetryOld" in got and "RetryLive" in got:
                io_ = got["RetryOld"]["option"]["info"]
                il = got["RetryLive"]["option"]["info"]
                if io_["retreat_hash"] != il["retreat_hash"]:
                    prob.append(f"{sc['case_id']}: отвод различается")
                if il["target_xy"] == io_["target_xy"] and not \
                        np.array_equal(got["RetryOld"]["actions"],
                                       got["RetryLive"]["actions"]):
                    prob.append(f"{sc['case_id']}: одинаковая цель, разные "
                                f"эпизоды")
            rep["forced"].append(out)
        arm = v = None
        kh.free(c.torch)
        rep["live_example"] = live_example(rep["forced"], cfg)
        ce = calib_entry()
        rep["calib_sha256"] = ce.get("sha256")
        rep["aperture"] = aperture_summary(rows, rep["probe"])
        rep["suggest"] = suggest_cfg(rep, ci)
        if planned and abs(rep["suggest"]["pos_m_measured"] - cfg["pos_m"]) \
                > 0.5 * cfg["pos_m"]:
            prob.append(f"pos_m плана {cfg['pos_m']} ≠ измеренному "
                        f"{rep['suggest']['pos_m_measured']:.4f}")
        rep.update(admitted=not prob, problems=prob,
                   env=kgr.env_record(c.torch), uuid=c.g["uuid"],
                   host=socket.gethostname())
        pc.save_json(str(smoke_path(c)), rep)
    finally:
        c.close()
    print(f"  smoke {a.device}: " + ("ДОПУЩЕНА" if not prob else
                                     "НЕ допущена: " + "; ".join(prob[:4])))
    print(f"  живой пример RetryLive: {rep.get('live_example')}")
    print(f"  апертура: {rep.get('aperture')}")
    print("  оси (м на единицу за шаг): " + str({k: round(
        d["per_unit_step"], 4) for k, d in rep["probe"]["axes"].items()}))
    print(f"  предложение для config.json: {rep.get('suggest')}")
    if not planned:
        print("  плана нет: это калибровочный smoke; допуск карты — "
              "повторный smoke после prepare")
    return OK if not prob else TECH


def suggest_cfg(rep, ci):
    """Числа для config.json по пробе: фактический метр на единицу за шаг
    (среднее по осям и знакам), номинал контроллера, порог пустого
    закрытия (+3 мм к контрольному)."""
    ax = rep["probe"]["axes"]
    meas = float(np.mean([abs(d["per_unit_step"]) for d in ax.values()]))
    imax = np.asarray(ci.get("input_max") or [np.nan], np.float64).reshape(-1)
    omax = np.asarray(ci.get("output_max") or [np.nan],
                      np.float64).reshape(-1)
    held = rep["aperture"]["held"]["ap_q"]
    return dict(pos_m_measured=meas,
                pos_m_nominal=float(omax[0] / imax[0]),
                empty_max=float(rep["aperture"]["empty_close_last"] + 0.003),
                held_min_hint=None if held is None else held[0])


def aperture_summary(rows, probe):
    ev = [e for r in rows for e in r["observer"]["events"]]
    out = dict(empty_close_last=probe["empty_ap"][-1],
               empty_close_min=min(probe["empty_ap"]))
    for oc in ("miss", "held", "uncertain", "released", "done"):
        aps = [max(e["ap"]) for e in ev if e["outcome"] == oc and
               e.get("ap")]
        out[oc] = dict(n=sum(1 for e in ev if e["outcome"] == oc),
                       ap_q=np.quantile(aps, [0, .5, 1]).tolist()
                       if aps else None)
    lifts = [e["lift"] for e in ev if e.get("lift") is not None]
    out["lift_q"] = np.quantile(lifts, [0, .5, 1]).tolist() if lifts else None
    return out


# --- калибровка локализатора ----------------------------------------------------------------------
def calib_job(args):
    """В отдельном процессе: K33Env той же сцены, ожидание, переисполнение
    действий; кадры t = 0, t = g и после отвода опции (2 чанка
    retreat_up/retreat_back с раскрытым схватом, как в пилоте), позы тел
    moka, матрица камеры."""
    task, space, bank_dir, pos, A, g, S, ev, cfg = args
    try:
        import k33_bank as kb
        from robosuite.utils import camera_utils as cu
        K = kb.make_env_class()
        env = K(task_id=int(task), image_size=224, k33_mode="bank",
                k33_space=int(space), k33_bank_dir=bank_dir)
        try:
            env.reset(init_state_id=int(pos))
            sim = env.env.sim
            names = [str(sim.model.body_id2name(i))
                     for i in range(sim.model.nbody)]
            moka = [n for n in names if "moka" in n.lower()]
            main = [n for n in moka if n.endswith("_main")] or moka
            if not main:
                raise RuntimeError(f"нет тел moka среди {names[:40]}")
            ids = [names.index(n) for n in main]

            def bodies():
                return {n: np.array(sim.data.body_xpos[i], copy=True).tolist()
                        for n, i in zip(main, ids)}
            obs = None
            for _ in range(kb.WAIT):
                obs, _r, _d, _i = env.step(np.array(kb.DUMMY, np.float64))
            out = dict(ok=True, frame0=np.asarray(obs["agentview_image"]),
                       bodies0=bodies(), state_ok=[])
            dev = [float(np.abs(np.asarray(obs["state"]) - S[0]).max())]
            for t in range(g):
                obs, _r, _d, _i = env.step(np.asarray(A[t], np.float64))
                dev.append(float(np.abs(np.asarray(obs["state"]) -
                                        S[t + 1]).max()))
            out.update(frame_g=np.asarray(obs["agentview_image"]),
                       bodies_g=bodies(), max_state_dev=max(dev))
            cp = np.asarray(ev["close_pos"], np.float64)
            ap_ = np.asarray(ev["approach_pos"], np.float64)
            z_clear = max(float(ap_[2]), float(cp[2]) + cfg["clear_m"])
            q0 = st9(obs["state"])[5:9]
            for ph in ("retreat_up", "retreat_back"):
                s9 = st9(obs["state"])
                p_ = s9[2:5]
                T = np.array([*(p_[:2] if ph == "retreat_up" else ap_[:2]),
                              max(z_clear, float(p_[2])) if ph == "retreat_up"
                              else z_clear])
                E = pctl(p_, s9[5:9], T, q0, OPEN, cfg)
                for t in range(H8):
                    obs, _r, _d, _i = env.step(E[t])
            out.update(frame_r=np.asarray(obs["agentview_image"]),
                       bodies_r=bodies(),
                       z_after_retreat=float(st9(obs["state"])[4]),
                       z_clear=z_clear,
                       P=np.asarray(cu.get_camera_transform_matrix(
                           sim, "agentview", 224, 224))[:3])
            return out
        finally:
            env.close()
    except Exception as e:                                   # noqa: BLE001
        return dict(ok=False, why=f"{type(e).__name__}: {e}")


def crop(img, row, col, half=12):
    r, c = int(round(row)), int(round(col))
    if r - half < 0 or c - half < 0 or r + half > img.shape[0] or \
            c + half > img.shape[1]:
        return None
    return np.asarray(img[r - half:r + half, c - half:c + half], np.uint8)


def nearest_body(bodies, xy):
    return min(bodies.items(), key=lambda kv: np.linalg.norm(
        np.asarray(kv[1][:2]) - np.asarray(xy)))


def build_calib(rows, conf_min=0.6, margin_min=0.05, radius_px=24):
    """rows: dict(frame_g, bodies_g, frame_r, bodies_r, P, close_pos)
    удачных захватов -> (лучшая калибровка по ориентации кадра, отчёт).
    Шаблоны — из кадров после отвода (r, как у RetryLive) и кадров закрытия
    (g, как у защиты по объекту); оценка — отложенная по сцене (шаблоны
    других сцен) отдельно на r и на g, мм."""
    best = None
    for flip in (True, False):
        P = rows[0]["P"]
        tgt = [nearest_body(r["bodies_g"], r["close_pos"][:2])[0]
               for r in rows]
        z = float(np.median([r["bodies_g"][n][2] for r, n in zip(rows, tgt)]))
        offs = np.array([np.asarray(r["close_pos"][:2]) -
                         np.asarray(r["bodies_g"][n][:2])
                         for r, n in zip(rows, tgt)])
        off = np.median(offs, 0)
        tpls = []
        for r, n in zip(rows, tgt):
            per = []
            for fk, bk in (("frame_r", "bodies_r"), ("frame_g", "bodies_g")):
                b = r[bk][n]
                rr, cc = project(P, [b[0], b[1], z], 224, flip)
                per.append(crop(r[fk], rr, cc))
            tpls.append(per)
        stat = {}
        for fk, bk in (("frame_r", "bodies_r"), ("frame_g", "bodies_g")):
            err, oks = [], []
            for i, (r, n) in enumerate(zip(rows, tgt)):
                others = [t for k, per in enumerate(tpls) if k != i
                          for t in per if t is not None]
                if not others:
                    continue
                loc = TemplateLocalizer(P, flip, z, off, others, conf_min,
                                        margin_min, radius_px)
                b = np.asarray(r[bk][n][:2])
                res = loc.locate(r[fk], b + np.array([0.015, -0.010]))
                oks.append(bool(res["ok"]))
                if res["ok"]:
                    err.append(1000 * float(np.linalg.norm(
                        np.asarray(res["xy"]) - b)))
            stat[fk[-1]] = dict(ok_frac=float(np.mean(oks)) if oks else 0.0,
                                err_mm_q=np.quantile(err, [.5, .9]).tolist()
                                if err else None)
        rep = dict(flip_v=flip, z_plane=z, offset=off.tolist(),
                   offset_spread_mm=(1000 * np.std(offs, 0)).tolist(),
                   n=len(rows), retreat=stat["r"], close=stat["g"],
                   ok_frac=stat["r"]["ok_frac"],
                   err_mm_q=stat["r"]["err_mm_q"],
                   retreat_z_short=int(sum(
                       r["z_after_retreat"] < r["z_clear"] - 0.010
                       for r in rows)))
        score = (rep["ok_frac"], -(rep["err_mm_q"] or [1e9])[0])
        if best is None or score > best[0]:
            best = (score, rep, [t for per in tpls for t in per
                                 if t is not None], P)
    _s, rep, tpls, P = best
    rep["usable"] = bool(rep["ok_frac"] >= 0.8 and rep["err_mm_q"] and
                         rep["err_mm_q"][0] <= 10.0)
    rep["why"] = "" if rep["usable"] else \
        "доля/ошибка локализации после отвода"
    calib = dict(P=np.asarray(P), flip_v=np.array(rep["flip_v"]),
                 z_plane=np.array(rep["z_plane"]),
                 offset=np.asarray(rep["offset"]),
                 templates=np.stack(tpls) if tpls else
                 np.zeros((0, 24, 24, 3), np.uint8),
                 conf_min=np.array(conf_min), margin_min=np.array(margin_min),
                 radius_px=np.array(radius_px), H=np.array(224),
                 W=np.array(224))
    return calib, rep


def calibrate(a):
    """Шаблоны и смещение по удачным захватам dev-эпизодов smoke (CPU)."""
    import multiprocessing as mp
    import k33_protocol as p33
    if plan_path().exists():
        raise SystemExit("план уже закреплён — калибровка меняется только "
                         "в новом каталоге")
    _m, _pilot, dev = scenes()
    pos = {d["case_id"]: d["pos"] for d in dev}
    jobs, keys = [], []
    for f in sorted(SMOKE.glob("dev_*.npz")):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        for e in m["observer"]["events"]:
            if e["outcome"] == "held":
                jobs.append((TASK, SPACE, os.path.abspath(BANK_DIR),
                             pos[m["case_id"]], z["actions"], e["g"],
                             z["tr_state"], e, load_cfg()))
                keys.append((m["case_id"], e))
    if len(jobs) < 3:
        raise SystemExit(f"удачных захватов в smoke {len(jobs)} < 3 — нужен "
                         f"больший --n-dev")
    with mp.get_context("forkserver").Pool(min(4, len(jobs))) as pool:
        res = pool.map(calib_job, jobs)
    bad = [(k[0], r.get("why")) for k, r in zip(keys, res) if not r["ok"]]
    if bad:
        raise SystemExit(f"калибровка: {bad[:3]}")
    rows = [dict(r, close_pos=k[1]["close_pos"]) for k, r in zip(keys, res)]
    calib, rep = build_calib(rows)
    rep["max_state_dev"] = max(r["max_state_dev"] for r in res)
    rep["events"] = [dict(case_id=k[0], g=k[1]["g"]) for k in keys]
    rep["errors"] = rep["err_mm_q"]
    CALIB.parent.mkdir(parents=True, exist_ok=True)
    with open(str(CALIB) + ".tmp", "wb") as fh:
        np.savez(fh, **calib)
    os.replace(str(CALIB) + ".tmp", CALIB)
    p33.save_json(str(CALIB.with_suffix(".json")), rep)
    print(f"  калибровка: {len(rows)} захватов, flip_v {rep['flip_v']}, "
          f"смещение {np.round(rep['offset'], 4).tolist()} м, доля "
          f"локализаций после отвода {rep['ok_frac']:.2f}, ошибка мм "
          f"{rep['err_mm_q']}; на кадре закрытия (защита) {rep['close']}; "
          f"отвод не набрал высоту {rep['retreat_z_short']}/{rep['n']}; "
          f"расхождение повтора state {rep['max_state_dev']:.1e}; "
          f"{'ПРИГОДНА' if rep['usable'] else 'НЕ пригодна (RetryLive = '}"
          f"{'' if rep['usable'] else 'fallback)'}")
    return OK


# --- сбор --------------------------------------------------------------------------------------------
def part_dir(k):
    return ROOT / f"part-{int(k)}"


def admission_ok(c, plan):
    import k36_protocol as pc
    import k38_grip_run as kgr
    sp = smoke_path(c)
    if not sp.exists():
        return False
    adm = pc.load_json(str(sp))
    return bool(adm.get("admitted") and adm.get("plan") == plan["sha256"] and
                adm.get("env") == kgr.env_record(c.torch))


def triggered_of(plan, k, C, cfg):
    """case_id сцен части с триггером по проверенным шардам T32."""
    out = set()
    for j in jobs_of(plan, k, set()):
        p = shard_path(j["eid"])
        if p.exists():
            _zz, m = read_shard(p)
            if fork_of(m) is not None:
                out.add(j["scene"]["case_id"])
    return out


def part_loop(L, plan, k, C, cfg, runner, log=print):
    """Сцены части по рангу: T32 -> ветви (если триггер). Шард и проверка
    до commit; открытый резерв с готовым шардом — settle. runner(job,
    prefix, expect) -> эпизод."""
    import k40b_eval_base as kb40
    jobs_all = jobs_of(plan, k)
    lp_ = kb40.ledger_problems(L.d, plan, k, jobs_all, final=False)
    if lp_:
        raise SystemExit(f"журнал part {k}: {lp_[:3]}")
    by = {j["eid"]: j for j in jobs_all}
    t32 = {}

    def t32_of(sc):
        e = eid_of("T32", sc)
        if e not in t32:
            t32[e] = read_shard(shard_path(e))
        return t32[e]

    def check(j, done="_no"):
        if j["branch"] == "T32":
            return check_shard(j, plan["sha256"], C, cfg, done)
        return check_shard(j, plan["sha256"], C, cfg, done,
                           t32=t32_of(j["scene"]))
    pdg = L.d["pending"]
    if pdg:
        if pdg["id"] not in by:
            raise SystemExit(f"открытый резерв {pdg['id']} не из части")
        m, pr = check(by[pdg["id"]])
        if m is not None and not pr:
            L.settle(pdg["id"], m["summary"]["physical"])
        elif m is not None:
            raise SystemExit(f"резерв {pdg['id']}: {pr[:3]}")

    def ensure(j, prefix=None, expect=None):
        if shard_path(j["eid"]).exists():
            _m, pr = check(j, L.d["done"].get(j["eid"]))
            if pr:
                raise SystemExit(f"{j['eid']}: {pr[:3]}")
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
            f"{meta['summary']['steps']}, триггер "
            f"{(meta['observer']['trigger'] or {}).get('tau')}; учтено "
            f"{len(L.d['done'])}")
    for j in jobs_all:
        if j["branch"] != "T32":
            continue
        ensure(j)
        sc = j["scene"]
        z0, m0 = t32_of(sc)
        tr = fork_of(m0)
        if tr is None:
            continue
        ex = dict(tau=tr["tau"], hash=m0["fork_hash"], guard_ok=True)
        ph = int((z0["c_tau"] < tr["tau"]).sum())
        for b in sc["order"]:
            jb = dict(by[eid_of(b, sc)], prefix_heavy=ph)
            ensure(jb, prefix=z0["actions"][:tr["tau"]], expect=ex)
        mo = read_shard(shard_path(eid_of("RetryOld", sc)))
        ml = read_shard(shard_path(eid_of("RetryLive", sc)))
        pp = pair_problems(mo[1], ml[1], mo[0], ml[0])
        if pp:
            raise SystemExit(f"{sc['case_id']}: пара ветвей {pp}")
    return OK


def collect(a):
    import k31_collect as kcol
    import k36_harness as kh
    import k36_protocol as pc
    plan = load_plan()
    k36 = pc.load_prepare()
    C = kh.consts(k36)
    cfg = plan["config"]
    loc = load_localizer(plan["localizer"])
    k = int(a.part)
    if str(k) not in plan["parts"]:
        raise SystemExit(f"часть {k}")
    c = card(a.device)
    pd = part_dir(k)
    pd.mkdir(parents=True, exist_ok=True)
    plock = c.kcf.Lock(str(pd / ".lock"), f"part {k}", c.kcf.Busy)
    try:
        if not admission_ok(c, plan):
            raise SystemExit("карта не допущена smoke для этого плана/среды")
        L = kcol.Ledger(str(pd / "ledger.json"),
                        dict(base=plan["parts"][str(k)]["cap"]))
        c.start()
        sim = c.sim(SPACE, BANK_DIR)
        src = {}

        def runner(j, prefix, expect):
            if not src:
                src["v"], src["arm"] = c.source(k36, SEED)
            c.kcf.check_disk(str(ROOT))
            sc = j["scene"]
            return run_branch(sim, src["v"]["io"], src["arm"], C, cfg, loc,
                              j["branch"], sc["task"], sc["pos"],
                              eid_of("T32", sc), prefix=prefix,
                              expect=expect)
        try:
            part_loop(L, plan, k, C, cfg, runner,
                      log=lambda s: print(s, flush=True))
        except kcol.Mismatch as e:
            print(f"  техническое расхождение воспроизведения: {e} — резерв "
                  f"открыт, нужен разбор", flush=True)
            return TECH
        src.clear()
        kh.free(c.torch)
    finally:
        plock.release()
        c.close()
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
        print(f"  part {k}: T32 {n_t}/{v['n_scenes']}, всего эпизодов "
              f"{len(d['done'])}; резерв {(d['pending'] or {}).get('id', '—')}"
              f"; аварий {len(d['crashes'])}")
    return OK


# --- анализ ---------------------------------------------------------------------------------------------
def paired(ya, yb, n_boot=BOOT["n"], seed=BOOT["seed"]):
    """Парный контраст по сценам (одна задача, один сид)."""
    d = np.asarray(ya, np.float64) - np.asarray(yb, np.float64)
    W, L_ = int((d == 1).sum()), int((d == -1).sum())
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return dict(W=W, L=L_, D=W - L_, N=len(d), disc=float(np.mean(d != 0)),
                delta_pp=100 * float(d.mean()), lo=100 * float(lo),
                hi=100 * float(hi))


def decide_rule(c_old, c_live, c_lo, rule=RULE):
    if c_live["D"] >= rule["live_t32_D"] and c_lo["D"] >= rule["live_old_D"]:
        return "freeze_directed"
    if c_old["D"] >= rule["old_t32_D"]:
        return "confirm_simple"
    if c_live["D"] >= rule["live_t32_D"]:
        return "confirm_directed_tree_open"
    return "close_option"


RULE_TEXT = dict(
    freeze_directed="заморозить направленный повтор; подтвердить SR",
    confirm_simple="первым подтверждать простой повтор",
    confirm_directed_tree_open="подтвердить замороженный направленный "
                               "контроллер; польза дерева открыта",
    close_option="закрыть опцию для t8 в данном бюджете")


def summarize(rows):
    """rows: по сцене dict(T32, RetryOld, RetryLive -> (успех, meta|None),
    trigger) -> таблица результата."""
    Y = {b: np.array([int(r[b][0]) for r in rows]) for b in BRANCHES}
    trg = np.array([r["trigger"] is not None for r in rows])
    fail = Y["T32"] == 0
    res = dict(n=len(rows), sr={b: float(Y[b].mean()) for b in BRANCHES},
               old_t32=paired(Y["RetryOld"], Y["T32"]),
               live_t32=paired(Y["RetryLive"], Y["T32"]),
               live_old=paired(Y["RetryLive"], Y["RetryOld"]))
    res["decision"] = decide_rule(res["old_t32"], res["live_t32"],
                                  res["live_old"])
    nf, ns = int(fail.sum()), int((~fail).sum())
    cov = dict(n_trigger=int(trg.sum()), fail_T32=nf,
               trig_on_fail=int((trg & fail).sum()),
               trig_on_success=int((trg & ~fail).sum()),
               c_f=float((trg & fail).sum() / nf) if nf else None,
               c_s=float((trg & ~fail).sum() / ns) if ns else None,
               ceiling_pp=100 * float((trg & fail).sum()) / len(rows),
               by_pot={}, by_event={})
    for r in rows:
        if r["trigger"] is not None:
            kp_ = str(r["trigger"]["pot_guess"])
            ke = str(r["trigger"]["event"])
            cov["by_pot"][kp_] = cov["by_pot"].get(kp_, 0) + 1
            cov["by_event"][ke] = cov["by_event"].get(ke, 0) + 1
    br = {}
    for b in RETRY:
        ms = [r[b][1] for r in rows if r[b][1] is not None]
        inf = [m["option"]["info"] for m in ms]
        saves = int(sum(1 for r in rows if r[b][1] is not None and
                        r[b][0] and not r["T32"][0]))
        losses = int(sum(1 for r in rows if r[b][1] is not None and
                         not r[b][0] and r["T32"][0]))
        br[b] = dict(
            n=len(ms), saves=saves, losses=losses,
            r=saves / cov["trig_on_fail"] if cov["trig_on_fail"] else None,
            l=losses / cov["trig_on_success"] if cov["trig_on_success"]
            else None,
            reached_close=sum(1 for x in inf if x.get("stop") == "завершена"),
            stop={s: sum(1 for x in inf if x.get("stop") == s)
                  for s in {x.get("stop") for x in inf}},
            held_after_lift=sum(1 for x in inf
                                if (x.get("result") or {}).get("held")),
            fallback=sum(1 for x in inf if x.get("fallback")),
            live_ok=sum(1 for x in inf if (x.get("live") or {}).get("ok")),
            shift_mm_q=np.quantile([1000 * x["shift_m"] for x in inf
                                    if x.get("shift_m") is not None],
                                   [0, .5, 1]).tolist()
            if any(x.get("shift_m") is not None for x in inf) else None,
            option_steps=float(np.mean([m["option"]["steps"] for m in ms]))
            if ms else None,
            heavy=int(sum(m["heavy"] + m["prefix_heavy"] for m in ms)),
            physical=int(sum(m["summary"]["physical"] for m in ms)))
    res["coverage"], res["branches"] = cov, br
    res["refusals"] = {}
    for r in rows:
        for x in r["refusals"]:
            res["refusals"][x["reason"]] = res["refusals"].get(
                x["reason"], 0) + 1
    res["outcomes"] = {}
    for r in rows:
        for e in r["events"]:
            k_ = f"{e['outcome']}:{e['reason']}"
            res["outcomes"][k_] = res["outcomes"].get(k_, 0) + 1
    return res


def analyze(a):
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    import k40b_eval_base as kb40
    plan = load_plan()
    if plan["t32"]["seed"] != SEED or SEED != 0:
        raise SystemExit(f"пилот закреплён на T32 s0, в плане "
                         f"s{plan['t32']['seed']}")
    C = kh.consts(pc.load_prepare())
    cfg = plan["config"]
    rows, prob, locks, cost = [], [], [], dict(physical=0, heavy=0, sec=0.0)
    try:
        for k in range(N_PARTS):
            locks.append(kcf.Lock(str(part_dir(k) / ".lock"), f"part {k}",
                                  kcf.Busy))
            d = json.load(open(part_dir(k) / "ledger.json"))
            trig = triggered_of(plan, k, C, cfg)
            jobs = jobs_of(plan, k, trig)
            prob += [f"part {k}: {x}" for x in kb40.ledger_problems(
                d, plan, k, jobs, final=True)]
            tot = 0
            by = {}
            for j in jobs:
                t0 = by.get(eid_of("T32", j["scene"]))
                m, pr = check_shard(j, plan["sha256"], C, cfg,
                                    d["done"].get(j["eid"]),
                                    t32=None if t0 is None else t0)
                prob += [f"{j['eid']}: {x}" for x in pr]
                if m is None or pr:
                    continue
                by[j["eid"]] = read_shard(shard_path(j["eid"]))
                tot += m["summary"]["physical"]
                cost["heavy"] += m["heavy"]
                cost["sec"] += m["seconds"]
            cost["physical"] += tot
            if d["spent"] != dict(base=tot + EP * len(d["crashes"])):
                prob.append(f"part {k}: расход ≠ сумме")
            for sc in sorted((x for x in plan["pilot"] if x["part"] == k),
                             key=lambda x: x["rank"]):
                e0 = eid_of("T32", sc)
                if e0 not in by:
                    continue
                z0, m0 = by[e0]
                tr = fork_of(m0)
                g_ = m0.get("guard")
                ref_ = list(m0["observer"]["refusals"])
                if m0["observer"]["trigger"] is not None and tr is None:
                    ref_.append(dict(reason=f"защита: {g_['why']}"
                                     if g_ else "защита: нет записи"))
                r = dict(case_id=sc["case_id"], trigger=tr,
                         events=m0["observer"]["events"], refusals=ref_,
                         T32=(bool(m0["summary"]["success"]), m0))
                for b in RETRY:
                    if tr is None:
                        r[b] = (r["T32"][0], None)
                    else:
                        zb = by.get(eid_of(b, sc))
                        r[b] = (bool(zb[1]["summary"]["success"]), zb[1]) \
                            if zb else (False, None)
                if tr is not None:
                    zo, zl = by.get(eid_of("RetryOld", sc)), \
                        by.get(eid_of("RetryLive", sc))
                    if zo and zl:
                        prob += [f"{sc['case_id']}: {x}" for x in
                                 pair_problems(zo[1], zl[1], zo[0], zl[0])]
                    else:
                        prob.append(f"{sc['case_id']}: нет ветвей")
                rows.append(r)
    finally:
        for lk in locks:
            lk.release()
    if len(rows) != N_PILOT:
        prob.append(f"сцен {len(rows)} ≠ {N_PILOT}")
    if prob:
        raise SystemExit("анализ: " + "; ".join(prob[:4]))
    res = summarize(rows)
    res.update(plan=plan["sha256"], cost=cost, t32_seed=plan["t32"]["seed"],
               localizer=plan["localizer"]["kind"])
    pc.save_json(str(ROOT / "analysis.json"), res)
    print(f"  T32 сид s{res['t32_seed']}, задача t{TASK}")
    print("  SR (96 сцен): " + ", ".join(f"{b} {100 * res['sr'][b]:.1f}%"
                                         for b in BRANCHES))
    for n_ in ("old_t32", "live_t32", "live_old"):
        c_ = res[n_]
        print(f"  {n_}: W {c_['W']} L {c_['L']} D {c_['D']:+d}; Δ "
              f"{c_['delta_pp']:+.1f} п.п. 95% [{c_['lo']:+.1f}; "
              f"{c_['hi']:+.1f}]; расхождений {100 * c_['disc']:.0f}%")
    cv = res["coverage"]
    print(f"  охват: триггеров {cv['n_trigger']} (на провалах T32 "
          f"{cv['trig_on_fail']}/{cv['fail_T32']}, на успехах "
          f"{cv['trig_on_success']}); по кофеварке {cv['by_pot']}; потолок "
          f"{cv['ceiling_pp']:.1f} п.п.")
    for b in RETRY:
        x = res["branches"][b]
        print(f"  {b}: спасений {x['saves']}, потерь {x['losses']}, дошли "
              f"до конца опции {x['reached_close']}, захват после подъёма "
              f"{x['held_after_lift']}, fallback {x['fallback']}, остановки "
              f"{x['stop']}, шагов опции {x['option_steps']}")
    print(f"  отказы триггера: {res['refusals']}; локализатор "
          f"{res['localizer']}")
    print(f"  стоимость: физ. шагов {cost['physical']}, тяжёлых вызовов "
          f"{cost['heavy']}, {cost['sec'] / 3600:.2f} ч")
    print(f"  решение: {res['decision']} — {RULE_TEXT[res['decision']]}")
    return OK


# --- самопроверка ------------------------------------------------------------------------------------------
def _toy_P():
    return np.array([[400.0, 0, 0, 112], [0, -400.0, 0, 112], [0, 0, 0, 1]])


def _draw(objs, H=224):
    """Сырой рендер игрушки: объект — светлый квадрат с тёмным центром."""
    img = np.full((H, H, 3), 60, np.uint8)
    for o in objs:
        r, c = project(_toy_P(), o, H, False)
        r, c = int(round(r)), int(round(c))
        img[r - 5:r + 6, c - 5:c + 6] = 220
        img[r - 1:r + 2, c - 1:c + 2] = 20
        img[r - 5:r - 3, c - 5:c + 6] = 140
    return img


class GraspSim:
    """Игрушечная среда захвата: кисть интегрирует дельта-команды
    (0.05 м на единицу, слежение 0.6), пальцы закрываются по 0.01 за шаг
    до объекта (апертура held_ap) или до 0; удерживаемый объект едет с
    кистью; успех — объект отпущен у цели. Кадр agentview перевёрнут, как
    в k33_bank._obs_now. Сценарий — по init: 0 попадание, 1 промах 20 мм,
    2 неопределённая апертура, 3 поздний промах, 4 закрытие в воздухе,
    затем промах."""
    GOAL = np.array([0.15, 0.10])

    def reset(self, task, starts):
        self.sc = int(starts[0]) % 10
        self.t, self.held = 0, False
        self.eef = np.array([0.0, 0.0, 0.15])
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.ap = 0.08
        self.obj = np.array([0.06, -0.04, 0.0])
        self.held_ap = 0.007 if self.sc == 2 else 0.02
        return self._obs()

    def near(self):
        return np.linalg.norm(self.eef[:2] - self.obj[:2]) < 0.012 and \
            abs(self.eef[2] - self.obj[2]) < 0.03

    def _obs(self):
        st = np.zeros((1, 9), np.float64)
        st[0, 0], st[0, 1] = self.ap / 2, -self.ap / 2
        st[0, 2:5], st[0, 5:9] = self.eef, self.quat
        av = _draw([self.obj])[::-1][None].copy()
        return dict(state=st, agentview_image=av,
                    robot0_eye_in_hand_image=np.zeros((1, 224, 224, 3),
                                                      np.uint8),
                    toy_obj=self.obj.copy()[None], toy_held=np.array(
                        [self.held]))

    def step(self, a):
        a = np.asarray(a, np.float64)[0]
        self.t += 1
        if self.t > WAIT:
            self.eef = self.eef + np.clip(a[:3], -1, 1) * 0.05 * 0.6
            self.eef[2] = max(self.eef[2], 0.0)
        if a[6] > 0:
            tgt = self.held_ap if (self.near() or self.held) else 0.0
            self.ap = max(self.ap - 0.01, tgt)
            if self.near() and self.ap <= self.held_ap + 1e-9:
                self.held = True
        else:
            self.ap = min(self.ap + 0.01, 0.08)
            if self.held and self.ap > self.held_ap + 0.005:
                self.held = False
                self.obj[2] = 0.0
        if self.held:
            self.obj = self.eef.copy()
        s = bool(self.t > WAIT and not self.held and self.obj[2] <= 1e-9 and
                 np.linalg.norm(self.obj[:2] - self.GOAL) < 0.03)
        return self._obs(), np.array([float(s)]), np.array([s]), \
            np.array([s])

    def close(self):
        pass


class ScriptArm:
    """Марковский скрипт «T32» для игрушки: решение — функция наблюдения
    и elapsed (переисполнение префикса без вызовов не меняет поведение)."""

    def __init__(self, q, L, U, cfg):
        self.q, self.L, self.U, self.cfg = q, L, U, cfg
        self.call, self.episode, self.last = 0, 0, None

    def begin(self, task, init, episode=None):
        self.sc, self.call = int(init) % 10, 0

    def forward(self):
        return None

    def decide(self, batch, po, s8, elapsed):
        self.forward()
        self.call += 1
        o = batch
        s = st9(o["state"][0])
        p, ap = s[2:5], aperture(s)
        obj, held = o["toy_obj"][0], bool(o["toy_held"][0])
        miss = np.array([0.02, 0.0]) if self.sc in (1, 3, 4) else np.zeros(2)
        tgt = obj[:2] + miss
        T, grip = p.copy(), OPEN
        if self.sc == 3 and elapsed < 552:
            pass
        elif self.sc == 4 and 8 <= elapsed < 16:
            grip = CLOSE
        elif held:
            grip = CLOSE
            if np.linalg.norm(p[:2] - GraspSim.GOAL) > 0.01:
                T = np.array([*GraspSim.GOAL, 0.12])
            elif p[2] > 0.03:
                T = np.array([*GraspSim.GOAL, 0.0])
            else:
                grip = OPEN
        elif ap < 0.079:
            if p[2] < 0.10:
                T, grip = np.array([p[0], p[1], 0.12]), CLOSE
        elif np.linalg.norm(p[:2] - tgt) > 0.008:
            T = np.array([tgt[0], tgt[1], 0.12])
        elif p[2] > 0.022:
            T = np.array([tgt[0], tgt[1], 0.02])
        else:
            grip = CLOSE
        E = pctl(p, s[5:9], T, s[5:9], grip, self.cfg)
        codec, _c = to_codec(E, self.q, self.L, self.U)
        return codec, dict(call=self.call, elapsed=int(elapsed))

    def executed(self, k):
        pass


def _toy_calib(cfg):
    img = _draw([np.array([0.0, 0.0, 0.0])])[::-1]
    r, c = project(_toy_P(), [0, 0, 0], 224, True)
    return TemplateLocalizer(_toy_P(), True, 0.0, [0.0, 0.0],
                             [crop(img, r, c)], 0.6, 0.05, 24)


def selftest():
    import tempfile
    import torch
    import k21_model as km21
    import k25_model as k25m
    import k29_model as k29m
    import k29_policy as kp
    import k31_collect as kcol
    import k32_model as km32
    import k36_harness as kh
    import k39_pulse as k39p
    import k40b_eval_base as kb40
    k29m.no_tf32(torch)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    # детектор закрытия = k39_pulse.grip_event на первом событии
    for _ in range(300):
        A = np.zeros((40, 7))
        A[:, 6] = np.where(rng.random(40) < 0.5, OPEN, CLOSE)
        ob = Observer(dict(CFG0, buffer=64))
        S = np.zeros((41, 9))
        ob.start(S[0])
        g = None
        for i in range(40):
            ob.step(A[i], S[i + 1])
            if ob.watch is not None and g is None:
                g = ob.watch["g"]
        assert g == k39p.grip_event(A), (g, k39p.grip_event(A))
    # разбиение: 96 пилот + 8 dev, части по рангу, порядок ветвей вращается
    rows = [dict(task=TASK, case_id=f"k33s6_t8_c{i:05d}", pos=i,
                 hash=["a", "b"]) for i in range(N_SCENES)]
    pilot, dev = split_rows(rows)
    assert len(pilot) == 96 and len(dev) == 8
    assert {r["case_id"] for r in pilot}.isdisjoint(r["case_id"]
                                                    for r in dev)
    pl = dict(pilot=pilot)
    q0, q1 = jobs_of(pl, 0), jobs_of(pl, 1)
    assert len(q0) == len(q1) == 144 and len({j["eid"] for j in q0 + q1}) \
        == 288
    for q in (q0, q1):
        by = {}
        for j in q:
            by.setdefault(j["scene"]["case_id"], []).append(j["branch"])
        assert all(v[0] == "T32" and sorted(v) == sorted(BRANCHES)
                   for v in by.values())
    assert sum(r["order"][0] == "RetryOld" for r in pilot) == 48
    assert len(jobs_of(pl, 0, set())) == 48
    assert spaces_of(dict(files={"data/k33/bank/s1/t3.npz": "x",
                                 "data/k39/pulse/b1/bank/s4/t0.npz": "y"})) \
        == {1, 4}
    assert spaces_of(dict(space=5, files={})) == {5}
    # масштаб: round-trip и клип
    q = np.array([0.9, 0.8, 0.95, 0.3, 0.4, 0.5, 1.0])
    Lb = np.full(6, -1.5)
    Ub = np.full(6, 1.5)
    E = np.zeros((8, 7))
    E[:, :3] = [0.3, -0.2, 0.5]
    E[:, 6] = CLOSE
    cdc, cl = to_codec(E, q, Lb, Ub)
    assert not cl and np.abs(kp.scale_exec(cdc[None], q)[0] - E).max() < 1e-6
    assert (cdc[:, 6] == -1).all()                     # кодек: −g
    E[:, 0] = 5.0
    _c, cl = to_codec(E, q, Lb, Ub)
    assert cl
    # P-регулятор: направление, ограничение
    cfg = dict(CFG0, pos_m=0.03)
    Ep = pctl(np.zeros(3), [1, 0, 0, 0], [0.01, -0.5, 0.0], [1, 0, 0, 0],
              OPEN, cfg)
    assert Ep[0, 0] > 0 and Ep[0, 1] == -cfg["umax"] and \
        (Ep[:, 3:6] == 0).all()
    assert np.allclose(rot_err([0, 0, np.sin(0.1), np.cos(0.1)],
                               [0, 0, 0, 1]), [0, 0, 0.2])
    # локализатор: точность, неоднозначность, низкая уверенность, заглушка
    loc = _toy_calib(cfg)
    img = _draw([np.array([0.05, -0.03, 0])])[::-1]
    r = loc.locate(img, np.array([0.04, -0.02]))
    assert r["ok"] and np.abs(np.asarray(r["xy"]) - [0.05, -0.03]).max() \
        < 0.004, r
    r2 = loc.locate(_draw([np.array([0.05, -0.03, 0]),
                           np.array([0.05, -0.08, 0])])[::-1],
                    np.array([0.05, -0.055]))
    assert not r2["ok"] and r2["why"] == "два близких максимума", r2
    r3 = loc.locate(np.full((224, 224, 3), 60, np.uint8), np.zeros(2))
    assert not r3["ok"] and r3["why"] == "низкая уверенность"
    assert not NullLocalizer().locate(img, np.zeros(2))["ok"]
    rr, cc = project(_toy_P(), [0.02, 0.03, 0.0], 224, True)
    assert np.allclose(backproject(_toy_P(), rr, cc, 0.0, 224, True),
                       [0.02, 0.03])
    # калибровка: ориентация кадра и смещение по удачным захватам
    rows_c = []
    for x, y in ((0.03, 0.02), (-0.04, 0.05), (0.06, -0.05), (-0.02, -0.03),
                 (0.0, 0.07)):
        o = np.array([x, y, 0.0])
        fr = _draw([o])[::-1].copy()
        bd = {"moka_pot_1_main": o.tolist()}
        rows_c.append(dict(frame_r=fr, bodies_r=bd, frame_g=fr, bodies_g=bd,
                           P=_toy_P(), close_pos=(o + [0.01, 0.0, 0.02])
                           .tolist(), z_after_retreat=0.1, z_clear=0.1))
    cal, crep = build_calib(rows_c)
    assert crep["flip_v"] and crep["usable"], crep
    assert crep["close"]["ok_frac"] == 1.0 and crep["retreat_z_short"] == 0
    assert np.allclose(crep["offset"], [0.01, 0.0])
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "c.npz")
        np.savez(fp, **cal)
        lc = TemplateLocalizer.load(fp)
        r = lc.locate(_draw([np.array([0.02, 0.01, 0])])[::-1],
                      np.array([0.03, 0.0]))
        assert r["ok"] and np.abs(np.asarray(r["xy"]) - [0.02, 0.01]).max() \
            < 0.004, r
    _selftest_script(cfg, q, Lb, Ub, loc, kcol)
    # настоящий игрушечный Arm (журнал K-32): T32 и ветвь — шарды
    M = k29m.build(torch)
    Lb2 = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    Ub2 = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    mod = k25m.build(torch)["K25"](km21.fake_bar(torch), Lb2, Ub2,
                                   seed=0).eval()
    for p_ in mod.parameters():
        p_.requires_grad_(False)
    dec = M["Decoder"](mod).freeze()
    Hm = mod.H
    norm = km32.fit_normalizer(np.random.default_rng(0).normal(
        size=(64, 6 * Hm + 72)), Hm)
    actor, _c = km32.build(torch, Hm, 6, 0, norm)

    class FIO(kcol.FakeIO):
        @property
        def inp(self):
            return self

        def prep(self, obs, desc, task):
            x = float(obs["state"][0, 2])
            bt = km21.fake_batch(torch, B=1, seed=int(abs(x) * 1e3) % 13)
            return bt, 1, np.asarray(obs["state"][:, :8], np.float64)
    io = FIO()
    k36 = dict(k32=dict(k32_contract="C", seeds={"0": dict(
        source=dict(weights_hash="W"), T=dict(actor_hash="A"))}),
        k32_plan=dict(bounds=dict(L=list(Lb2) + [-1.0],
                                  U=list(Ub2) + [1.0]),
                      scale_q=np.asarray(io.max_act_q).tolist()))
    C = kh.consts(k36)
    src = dict(mod=mod, dec=dec, M=M, io=io, actors=dict(T=actor))
    arm = kh.make_arm(torch, k36, "T32_s0", src)
    sc = dict(task=TASK, case_id="toy", pos=0, rank=0, part=0,
              order=list(RETRY))
    e0 = eid_of("T32", sc)
    ep = run_branch(GraspSim(), io, arm, C, cfg, loc, "T32",
                    TASK, 0, e0, force_tau=24)
    assert ep["observer"]["trigger"]["forced"] and ep["heavy"] == \
        len(ep["calls"])
    sc["hash"] = [ep["init_hash"], ep["init_hash_full"]]
    plan = dict(sha256="P", pilot=[sc], parts={"0": dict(cap=4 * EP)},
                config=cfg)
    g_ = globals()
    keep = g_["SHARDS"]
    with tempfile.TemporaryDirectory() as td:
        g_["SHARDS"] = Path(td)
        try:
            j0 = dict(eid=e0, branch="T32", scene=sc)
            save_ep(shard_path(e0), j0, ep, "P")
            m0, pr = check_shard(j0, "P", C, cfg)
            assert pr == [], pr
            z0, m0 = read_shard(shard_path(e0))
            tr = m0["observer"]["trigger"]
            ex = dict(tau=tr["tau"], hash=m0["fork_hash"])
            ph = int((z0["c_tau"] < tr["tau"]).sum())
            for br, lc in (("RetryOld", loc), ("RetryLive", loc)):
                eb = run_branch(GraspSim(), io, arm, C, cfg, lc, br, TASK, 0,
                                e0, prefix=z0["actions"][:tr["tau"]],
                                expect=ex, force_tau=24)
                jb = dict(eid=eid_of(br, sc), branch=br, scene=sc,
                          prefix_heavy=ph)
                save_ep(shard_path(jb["eid"]), jb, eb, "P")
                mb, pr = check_shard(jb, "P", C, cfg, t32=(z0, m0))
                assert pr == [], (br, pr)
                assert eb["heavy"] == int((~read_shard(shard_path(
                    jb["eid"]))[0]["c_opt"]).sum())
            zb, mb = read_shard(shard_path(eid_of("RetryOld", sc)))
            jb = dict(eid=eid_of("RetryOld", sc), branch="RetryOld",
                      scene=sc, prefix_heavy=ph)
            zl, ml = read_shard(shard_path(eid_of("RetryLive", sc)))
            assert pair_problems(mb, ml, zb, zl) == []
            ml2 = json.loads(json.dumps(ml))
            ml2["option"]["info"]["retreat_hash"] = "x"
            assert pair_problems(mb, ml2, zb, zl)
            # порча: префикс, команды опции, тяжёлые, триггер, журнал
            bad = dict(zb)
            bad["c_codec"] = zb["c_codec"].copy()
            bad["c_codec"][0, 0, 0] += 0.01
            assert branch_problems(bad, mb, jb, C, cfg, z0, m0)
            bad = dict(zb, actions=zb["actions"].copy())
            bad["actions"][0, 0] += 1
            assert branch_problems(bad, mb, jb, C, cfg, z0, m0)
            assert branch_problems(zb, dict(mb, heavy=mb["heavy"] + 1), jb,
                                   C, cfg, z0, m0)
            assert branch_problems(zb, dict(mb, tau=mb["tau"] + 8), jb, C,
                                   cfg, z0, m0)
            mg = json.loads(json.dumps(mb))
            mg["guard"]["shift_m"] = 1.0
            assert "защита по объекту ветви ≠ T32" in branch_problems(
                zb, mg, jb, C, cfg, z0, m0)
            mt = json.loads(json.dumps(mb))
            mt["observer"]["trigger"]["g"] += 1
            assert branch_problems(zb, mt, jb, C, cfg, z0, m0)
            mo = json.loads(json.dumps(mb))
            mo["option"]["log"] = mo["option"]["log"][:-1]
            assert branch_problems(zb, mo, jb, C, cfg, z0, m0)
            z0b = dict(z0, tr_state=z0["tr_state"].copy())
            z0b["tr_state"][tr["g"], 3] += 1e-3
            assert t32_problems(z0b, m0, j0, C, cfg)
            assert t32_problems(z0, dict(m0, fork_hash="x"), j0, C, cfg)
            assert check_shard(jb, "Q", C, cfg, t32=(z0, m0))[1]
            # Ledger: строгая проверка с ожидаемым набором (триггер есть)
            jobs = jobs_of(plan, 0, {"toy"})
            good = dict(caps=dict(base=4 * EP), spent=dict(base=1),
                        pending=None, crashes=[],
                        done={j["eid"]: dict(scope="base", actual=1)
                              for j in jobs})
            assert kb40.ledger_problems(good, plan, 0, jobs, True) == []
            assert kb40.ledger_problems(good, plan, 0, jobs_of(plan, 0,
                                                               set()), True)
        finally:
            g_["SHARDS"] = keep
    _selftest_resume(cfg, q, Lb, Ub, loc)
    # контрасты и правило решения
    T = [1] * 70 + [0] * 26
    O = list(T)
    Lv = list(T)
    for i in range(70, 76):
        Lv[i] = 1
    O[70] = 1
    Lv[0] = 0
    rr_ = [dict(case_id=str(i), trigger=None, events=[], refusals=[],
                T32=(bool(T[i]), None), RetryOld=(bool(O[i]), None),
                RetryLive=(bool(Lv[i]), None)) for i in range(96)]
    c_ = paired(Lv, T, n_boot=500)
    assert (c_["W"], c_["L"], c_["D"], c_["N"]) == (6, 1, 5, 96)
    assert abs(c_["delta_pp"] - 100 * 5 / 96) < 1e-9 and c_["lo"] <= \
        c_["delta_pp"] <= c_["hi"]
    s_ = summarize(rr_)
    assert s_["live_old"]["D"] == 4 and s_["old_t32"]["D"] == 1
    assert s_["decision"] == "freeze_directed"
    assert decide_rule(dict(D=5), dict(D=5), dict(D=2)) == "confirm_simple"
    assert decide_rule(dict(D=4), dict(D=6), dict(D=2)) == \
        "confirm_directed_tree_open"
    assert decide_rule(dict(D=4), dict(D=4), dict(D=9)) == "close_option"
    # prepare без калибровки и без живого примера — отказ
    g_ = globals()
    keep_c, keep_r = g_["CALIB"], g_["ROOT"]
    with tempfile.TemporaryDirectory() as td:
        try:
            g_["CALIB"], g_["ROOT"] = Path(td) / "none.npz", Path(td)
            for ent in (None, dict(kind="template", sha256="s")):
                try:
                    if ent is None:
                        prepare(None)
                    else:
                        live_admission(ent)
                    raise AssertionError("prepare принят")
                except SystemExit as e:
                    assert ("null" if ent is None else "живым") in str(e), e
        finally:
            g_["CALIB"], g_["ROOT"] = keep_c, keep_r
    good = dict(case_id="a", RetryOld=dict(actions_sha1="x"),
                RetryLive=dict(actions_sha1="y", info=dict(
                    shift_m=0.01, fallback=None, stop="завершена",
                    phases_done=len(PHASES))))
    assert live_example([good], cfg)["ok"]
    same = json.loads(json.dumps(good))
    same["RetryLive"]["actions_sha1"] = "x"
    fb = json.loads(json.dumps(good))
    fb["RetryLive"]["info"]["fallback"] = "низкая уверенность"
    stp = json.loads(json.dumps(good))
    stp["RetryLive"]["info"].update(stop="поза не достигнута", phases_done=4)
    for bad_ in (same, fb, stp):
        assert not live_example([bad_], cfg)["ok"]
    print("самопроверка k42_recovery пройдена: детектор закрытия = "
          "grip_event; пустое закрытие -> промах и триггер, занятое -> "
          "захват, неопределённая апертура и мало шагов -> отказ с причиной, "
          "вторая попытка после закрытия в воздухе; RetryOld = RetryLive "
          "побитово при одинаковой цели, общий отвод, RetryLive по шаблону "
          "спасает промах; нулевая опция = T32 побитово; префикс и хеши; "
          "round-trip масштаба; шарды T32/ветвей, порча — отказ; журнал и "
          "возобновление; контрасты W/L/D и правило решения; защита по "
          "объекту (сдвиг/нет калибровки -> нет ветви); нет бокового "
          "движения без зазора; prepare отказывает без калибровки и живого "
          "примера")
    return 0


def _selftest_script(cfg, q, Lb, Ub, loc, kcol):
    """Семантика на скриптовой политике и игрушечном захвате."""
    C = dict(q=q, L=Lb, U=Ub)

    class SIO(kcol.FakeIO):
        max_act_q = q

        def prep(self, obs, desc, task):
            return obs, 0, np.asarray(obs["state"][:, :8], np.float64)
    io = SIO()

    def run(init, branch="T32", lc=None, prefix=None, expect=None):
        arm = ScriptArm(q, Lb, Ub, cfg)
        return run_branch(GraspSim(), io, arm, C, cfg, lc or loc,
                          branch, TASK, init, "toy", prefix=prefix,
                          expect=expect)

    class GuardOnly:
        """Защита проходит по шаблону, локализация после отвода — нет."""
        kind, offset = "guard_only", loc.offset

        def __init__(self):
            self.n = 0

        def locate(self, rgb, ref_xy):
            self.n += 1
            return loc.locate(rgb, ref_xy) if self.n <= 2 else \
                NullLocalizer().locate(rgb, ref_xy)
    hit, miss, unc, late, air = (run(i) for i in range(5))
    assert hit["success"] and hit["observer"]["trigger"] is None
    assert [e["outcome"] for e in hit["observer"]["events"]][0] == "held"
    tr = miss["observer"]["trigger"]
    assert not miss["success"] and tr is not None and tr["event"] == 1 and \
        not tr["forced"]
    assert tr["tau"] - tr["g"] <= cfg["window"] and tr["lift"] >= 0.02
    assert unc["observer"]["trigger"] is None and \
        unc["observer"]["events"][0]["reason"] == "неопределённая апертура"
    assert late["observer"]["trigger"] is None and \
        late["observer"]["refusals"][0]["reason"] == "мало шагов"
    ta = air["observer"]["trigger"]
    assert air["observer"]["events"][0]["outcome"] != "miss" and \
        ta is not None and ta["event"] == 2
    ex = dict(tau=tr["tau"], hash=miss["fork_hash"])
    pre = miss["actions"][:tr["tau"]]
    null = run(1, "Null", prefix=pre, expect=ex)
    assert np.array_equal(null["actions"], miss["actions"]) and \
        {k: null[k] for k in SUMMARY} == {k: miss[k] for k in SUMMARY}
    assert miss["guard"]["ok"] and miss["guard"]["shift_m"] < 1e-3
    old = run(1, "RetryOld", prefix=pre, expect=ex)
    live0 = run(1, "RetryLive", lc=GuardOnly(), prefix=pre, expect=ex)
    assert np.array_equal(old["actions"], live0["actions"])
    assert live0["option"]["info"]["fallback"] == "калибровка недоступна"
    assert old["guard"] == miss["guard"]
    # без калибровки защита отказывает -> ветви нет (вмешательства нет)
    try:
        run(1, "RetryOld", lc=NullLocalizer(), prefix=pre, expect=ex)
        raise AssertionError("ветвь при отказе защиты принята")
    except kcol.Mismatch as e:
        assert "защиты" in str(e)
    m_n = run(1, "T32", lc=NullLocalizer())
    assert m_n["observer"]["trigger"] is not None and \
        fork_of(m_n) is None and not m_n["guard"]["ok"]
    assert fork_of(miss) is not None
    # сдвинутый объект: защита отказывает
    f0 = _draw([np.array([0.06, -0.04, 0.0])])[::-1].copy()
    f1 = _draw([np.array([0.06, -0.02, 0.0])])[::-1].copy()
    cp_ = np.array([0.06, -0.04]) + loc.offset
    assert object_guard(loc, f0, f0, cp_, cfg)["ok"]
    gm = object_guard(loc, f0, f1, cp_, cfg)
    assert not gm["ok"] and gm["why"] == "объект сдвинулся", gm
    assert fork_of(dict(observer=dict(trigger=dict(forced=False)),
                        guard=gm)) is None
    live = run(1, "RetryLive", lc=loc, prefix=pre, expect=ex)
    oi, li = old["option"]["info"], live["option"]["info"]
    assert oi["retreat_hash"] == li["retreat_hash"] and li["fallback"] is None
    assert np.array_equal(old["actions"][:tr["tau"] + 16],
                          live["actions"][:tr["tau"] + 16])
    assert abs(li["shift_m"] - 0.02) < 0.004
    assert not oi["result"]["held"] and li["result"]["held"], (oi, li)
    assert not old["success"] and live["success"]
    for e in (old, live):
        assert e["option"]["steps"] <= 48 and \
            len(e["option"]["log"]) <= len(PHASES)
        assert e["heavy"] == sum(1 for x in e["log"] if x["kind"] == "t32")
    try:
        run(1, "RetryOld", prefix=pre, expect=dict(ex, hash="x"))
        raise AssertionError("чужой хеш развилки принят")
    except kcol.Mismatch:
        pass
    try:
        run(0, "RetryOld", prefix=hit["actions"][:24], expect=dict(
            tau=24, hash="x"))
        raise AssertionError("ветвь без триггера принята")
    except kcol.Mismatch:
        pass
    # недостижимая поза -> досрочный возврат T32 с причиной
    cf2 = dict(cfg, gain=0.2)
    arm = ScriptArm(q, Lb, Ub, cfg)
    e2 = run_branch(GraspSim(), io, arm, C, cf2, loc, "RetryOld",
                    TASK, 1, "toy", prefix=pre, expect=ex)
    assert e2["option"]["info"]["stop"] in ("поза не достигнута",
                                            "зазор не достигнут"), \
        e2["option"]["info"]
    # высота отвода не набрана -> бокового движения нет
    cf3 = dict(cfg, clear_m=0.5)
    e3 = run_branch(GraspSim(), io, ScriptArm(q, Lb, Ub, cfg), C, cf3, loc,
                    "RetryOld", TASK, 1, "toy", prefix=pre, expect=ex)
    i3 = e3["option"]["info"]
    assert i3["stop"] == "зазор не достигнут" and \
        [r["phase"] for r in e3["option"]["log"]] == ["retreat_up"], i3
    lg = e3["option"]["log"][0]
    assert np.allclose(lg["target"][:2], lg["start_pos"][:2])


def _selftest_resume(cfg, q, Lb, Ub, loc):
    """part_loop на скриптовой игрушке: авария после записи шарда до
    commit -> settle; повторный проход ничего не исполняет."""
    import tempfile
    import k31_collect as kcol
    C = dict(q=q, L=Lb, U=Ub)

    class SIO(kcol.FakeIO):
        max_act_q = q

        def prep(self, obs, desc, task):
            return obs, 0, np.asarray(obs["state"][:, :8], np.float64)
    io = SIO()
    sc = [dict(task=TASK, case_id=f"c{i}", pos=i, rank=i, part=0,
               order=list(RETRY if i % 2 else RETRY[::-1])) for i in (0, 1)]
    for s in sc:
        e = run_branch(GraspSim(), io, ScriptArm(q, Lb, Ub, cfg), C, cfg,
                       NullLocalizer(), "T32", TASK, s["pos"], "toy")
        s["hash"] = [e["init_hash"], e["init_hash_full"]]
    plan = dict(sha256="P", pilot=sc, parts={"0": dict(cap=7 * EP)},
                config=cfg)
    n_run = dict(n=0)

    def runner(j, prefix, expect):
        n_run["n"] += 1
        s = j["scene"]
        return run_branch(GraspSim(), io, ScriptArm(q, Lb, Ub, cfg), C, cfg,
                          loc, j["branch"], TASK, s["pos"], eid_of("T32", s),
                          prefix=prefix, expect=expect)
    import k32_harness as kh32
    g_ = globals()
    keep = (g_["SHARDS"], g_["t32_calls_problems"], g_["t32_problems"],
            kh32.pack_journal)
    with tempfile.TemporaryDirectory() as td:
        g_["SHARDS"] = Path(td) / "sh"
        # скрипт не пишет журнал K-32: проверки журнала здесь отключены
        g_["t32_calls_problems"] = lambda J, C: []
        g_["t32_problems"] = lambda zz, meta, j, C, cfg: trace_problems(
            zz, meta, cfg)
        kh32.pack_journal = lambda ec, d: dict(ep=np.zeros(sum(
            len(c) for _e, c in ec), np.int64), a_codec=np.stack(
            [c["a_codec"] for _e, cl in ec for c in cl]), k=np.array(
            [c["k"] for _e, cl in ec for c in cl], np.int64))
        try:
            L = kcol.Ledger(str(Path(td) / "ledger.json"),
                            dict(base=plan["parts"]["0"]["cap"]))
            real_commit = L.commit
            boom = dict(n=0)

            def crash_commit(eid, actual):
                boom["n"] += 1
                if boom["n"] == 2:
                    raise RuntimeError("авария до commit")
                return real_commit(eid, actual)
            L.commit = crash_commit
            try:
                part_loop(L, plan, 0, C, cfg, runner, log=lambda s: None)
                raise AssertionError("авария не произошла")
            except RuntimeError:
                pass
            L2 = kcol.Ledger(str(Path(td) / "ledger.json"),
                             dict(base=plan["parts"]["0"]["cap"]))
            assert L2.d["pending"] is not None
            n0 = n_run["n"]
            part_loop(L2, plan, 0, C, cfg, runner, log=lambda s: None)
            trig = {s["case_id"] for s in sc if read_shard(shard_path(
                eid_of("T32", s)))[1]["observer"]["trigger"]}
            assert trig == {"c1"}, trig
            assert set(L2.d["done"]) == {j["eid"] for j in
                                         jobs_of(plan, 0, trig)}
            assert n_run["n"] == n0 + (len(L2.d["done"]) - 2)
            assert L2.d["crashes"] == []
            n1 = n_run["n"]
            part_loop(L2, plan, 0, C, cfg, runner, log=lambda s: None)
            assert n_run["n"] == n1
        finally:
            (g_["SHARDS"], g_["t32_calls_problems"], g_["t32_problems"],
             kh32.pack_journal) = keep


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("bank", "smoke", "calibrate", "prepare",
                                    "collect", "status", "analyze"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-dev", type=int, default=N_DEV)
    ap.add_argument("--n-probe", type=int, default=2)
    a = ap.parse_args()
    import k37_confirm as kcf
    fn = dict(bank=bank, smoke=smoke, calibrate=calibrate, prepare=prepare,
              collect=collect, status=status, analyze=analyze)[a.cmd]
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
