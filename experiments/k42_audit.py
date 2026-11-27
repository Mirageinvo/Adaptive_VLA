#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-42: офлайн-аудит промахов захвата T1 (= T32) задачи 8 по архиву c2.

Модель не загружается. Каждый эпизод T1 задачи 8 архива K-37 c2 (провалы
и успехи) переисполняется в BankSim по записанным командам среды, как
k39_cache.Replay / k39_failure_probe.replay: свежая среда того же старта,
10 шагов ожидания, команды архива. Начальные хеши и terminal (успех, шаг
успеха, done, число шагов) обязаны совпасть с архивом, иначе отказ.

После ожидания и после каждой команды из рабочего процесса среды читается
служебная проба (не входит в наблюдение политики): body_xpos кофеварок
(obj_body_id LIBERO, имена из bddl: moka_pot_1, moka_pot_2), контакт
захвата robosuite _check_grasp и предикат On(кофеварка, cook_region плиты).
Из наблюдения берутся state = gripper_qpos(2) + eef_pos(3) + eef_quat(4)
(k33_bank._obs_now). Трассы пишутся в traces/*.npz; классификация —
отдельной командой report по сохранённым трассам (пороги меняются без
повторного прогона).

Попытка захвата — каждое устойчивое закрытие по командам среды: три open
(−1) и затем три close (+1); первое совпадает с k39_pulse.grip_event.
Цель попытки — кофеварка с контактом захвата в окне, иначе ближайшая по
xy к кисти в момент первого close. Окно разметки — до window_gt шагов,
пока команда остаётся close. Класс попытки:
  lifted — кофеварка поднялась ≥ pot_lift;
  miss   — кисть поднялась ≥ lift при закрытой команде, кофеварка нет
           (промах с подъёмом — событие плана K-42);
  nolift — ни того, ни другого (раскрыл раньше или не поднимал).
Для lifted — placed, если цель на плите в течение settle шагов после
первой open, иначе drop (held_end — закрытие до конца эпизода).
Режим провала — первая попытка, которая не lifted + placed, со стадией
(сколько кофеварок уже на плите); иначе no_attempt.

Онлайн-триггер плана (§2): граница чанка b (кратна 8) в (g, g + 16],
все команды g…b−1 — close, подъём кисти ≥ lift, апертура пальцев
|q0| + |q1| ≤ порога hold_k наблюдений подряд, цель сдвинулась не более
чем на stay (прокси «ориентир остался на столе» — по sim, т. е. верхняя
оценка для зрения), цель не на плите, осталось ≥ min_left шагов. Только
первое срабатывание на эпизод. Порог апертуры — --aperture или авто:
середина между p95 пустых (miss) и p5 удерживаемых (lifted).

--frames N: для первых N провалов части с закрытием сохраняется PNG кадров
(как они приходят в наблюдении, без переворота) на g, g+8, g+16, g+40 с
проекцией центров кофеварок через калибровку agentview robosuite
(get_camera_transform_matrix): красный — (строка, столбец) как есть, синий —
с вертикальным переворотом, зелёный — с горизонтальным. Какой маркер ложится
на кофеварку, та и ориентация кадров нашей обёртки.

Команды:
  python3 experiments/k42_audit.py --selftest
  python3 experiments/k42_audit.py run [--shard I/N] [--frames N]
  python3 experiments/k42_audit.py report [--aperture M] [--partial]
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

TASK = 8
OBJECTS = ("moka_pot_1", "moka_pot_2")
REGION = "flat_stove_1_cook_region"
CAMERA = "agentview"
IMAGE = 224
OPEN, CLOSE = -1.0, 1.0
N8 = 8
MAX_STEPS = 600
CFG = dict(lift=0.020, pot_lift=0.015, window_gt=40, window_trig=16, hold_k=3,
           stay=0.010, min_left=56, settle=16)
BASE = ("no_close", "lifted", "miss", "nolift")
OUT = Path("reports/k42/audit/v1")


# --- события и классификация (без среды) -------------------------------------------
def close_events(A):
    """Все устойчивые закрытия: индексы первой close (+1) после трёх open
    (−1; до начала — команда ожидания), за которой ещё две close. После
    события поиск продолжается с конца серии close. Первое = grip_event."""
    import k31_collect as kcol
    g = np.concatenate([[kcol.DUMMY[6]] * 3, np.asarray(A, np.float64)[:, 6]])
    out, i = [], 3
    while i < len(g) - 2:
        if (g[i - 3:i] == OPEN).all() and (g[i:i + 3] == CLOSE).all():
            out.append(i - 3)
            while i < len(g) and g[i] == CLOSE:
                i += 1
        else:
            i += 1
    return out


def close_end(A, g):
    """Индекс первой команды не close начиная с g (len(A), если нет)."""
    c = np.asarray(A, np.float64)[g:, 6] != CLOSE
    return int(g + np.argmax(c)) if c.any() else len(A)


def aperture(state):
    s = np.asarray(state, np.float64)
    return np.abs(s[..., 0]) + np.abs(s[..., 1])


def attempt(tr, g, cfg=CFG):
    """Разметка одной попытки по трассе: индексы наблюдений — число уже
    исполненных команд (0 — после ожидания)."""
    A, eef, pot = tr["A"], tr["state"][:, 2:5], tr["pot"]
    n = len(A)
    end = close_end(A, g)
    hi = min(end, g + cfg["window_gt"], n)
    win = np.arange(g + 1, hi + 1)
    grasp = tr["grasp"][win] if len(win) else np.zeros((0, pot.shape[1]))
    held = (grasp == 1).sum(0)
    if held.max(initial=0) > 0:
        tgt, by = int(np.argmax(held)), "contact"
    else:
        tgt = int(np.argmin(np.linalg.norm(pot[g, :, :2] - eef[g, :2], axis=1)))
        by = "nearest"
    eef_rise = float(eef[win, 2].max() - eef[g, 2]) if len(win) else 0.0
    pot_rise = float(pot[win, tgt, 2].max() - pot[g, tgt, 2]) if len(win) else 0.0
    if pot_rise >= cfg["pot_lift"]:
        kind = "lifted"
    elif eef_rise >= cfg["lift"]:
        kind = "miss"
    else:
        kind = "nolift"
    lo_h, hi_h = g + 4, min(end, g + cfg["window_trig"], n)
    ap = aperture(tr["state"])
    ap_hold = float(np.median(ap[lo_h:hi_h + 1])) if hi_h >= lo_h else \
        float(ap[min(g + 1, n)])
    out = dict(g=int(g), end=int(end), target=tgt, by=by, kind=kind,
               eef_rise=eef_rise, pot_rise=pot_rise, ap_hold=ap_hold,
               on_before=bool(tr["on"][g, tgt]))
    if kind == "lifted":
        if end >= n:
            out["fate"] = "held_end"
        else:
            on = tr["on"][end:min(n, end + cfg["settle"]) + 1, tgt]
            out["fate"] = "placed" if on.any() else "drop"
    return out


def boundary(tr, g, b, thr, cfg=CFG):
    """Проверка на границе b попытки g только по наблюдениям ≤ b ->
    (a, b, причина a). (a) — проприоцепция как у раннера; (b) — (a) и цель
    (ближайшая к кисти в g) осталась на месте и не на плите (по sim)."""
    eef, pot = tr["state"][:, 2:5], tr["pot"]
    ap = aperture(tr["state"])
    k0 = b - cfg["hold_k"] + 1
    if eef[b, 2] - eef[g, 2] < cfg["lift"]:
        return False, False, "no_lift"
    if k0 < g + 1 or (ap[k0:b + 1] > thr).any():
        return False, False, "aperture"
    if MAX_STEPS - b < cfg["min_left"]:
        return False, False, "time"
    tgt = int(np.argmin(np.linalg.norm(pot[g, :, :2] - eef[g, :2], axis=1)))
    ok_b = (np.linalg.norm(pot[b, tgt] - pot[g, tgt]) <= cfg["stay"] and
            not tr["on"][b, tgt])
    return True, bool(ok_b), "ok"


def fork_outcome(tr, g, b, cfg=CFG):
    """Разметка после развилки b (вне триггера): self_recovered — та же
    кофеварка позже поднялась ≥ pot_lift без вмешательства."""
    eef, pot = tr["state"][:, 2:5], tr["pot"]
    tgt = int(np.argmin(np.linalg.norm(pot[g, :, :2] - eef[g, :2], axis=1)))
    rise = float(pot[b + 1:, tgt, 2].max() - pot[b, tgt, 2]) if b + 1 < len(pot) else 0.0
    return dict(b=int(b), g=int(g), target=tgt, stage=int(tr["on"][b].sum()),
                outcome="self_recovered" if rise >= cfg["pot_lift"] else "not_lifted")


def causal_triggers(tr, thr, cfg=CFG):
    """Первая штатная граница (кратна 8) в (g, g + window_trig] любой
    попытки, где все команды g…b−1 — close и условие выполнено по данным
    до b. Границы на terminal (b = числу исполненных команд) исключены.
    -> dict(a=развилка | None, b=развилка | None, why=причины у (a))."""
    A = tr["A"]
    n = len(A)
    out, why = dict(a=None, b=None), Counter()
    for g in close_events(A):
        end = close_end(A, g)
        b = (g // N8 + 1) * N8
        while b <= g + cfg["window_trig"] and b <= end and b < n:
            ok_a, ok_b, w = boundary(tr, g, b, thr, cfg)
            why[w] += 1
            if ok_a and out["a"] is None:
                out["a"] = fork_outcome(tr, g, b, cfg)
            if ok_b and out["b"] is None:
                out["b"] = fork_outcome(tr, g, b, cfg)
            if out["a"] is not None and out["b"] is not None:
                return dict(out, why=dict(why))
            b += N8
    return dict(out, why=dict(why))


def classify(tr, thr, cfg=CFG):
    """Эпизод -> попытки (описательно), класс первой, режим провала и
    причинные триггеры (a)/(b) с исходом после развилки."""
    events = close_events(tr["A"])
    atts = [attempt(tr, g, cfg) for g in events]
    placed, mode = set(), None
    for at in atts:
        at["stage"] = len(placed)
        if at["kind"] == "lifted" and at.get("fate") == "placed":
            placed.add(at["target"])
        elif mode is None and at["target"] not in placed:
            kind = at["kind"] if at["kind"] != "lifted" else at["fate"]
            mode = dict(kind=kind, stage=at["stage"])
    for i, at in enumerate(atts):
        if at["kind"] == "miss":
            at["regrasp"] = any(b["target"] == at["target"] and b["kind"] == "lifted"
                                for b in atts[i + 1:])
    if mode is None:
        mode = dict(kind="no_attempt", stage=len(placed))
    return dict(attempts=atts, first=atts[0]["kind"] if atts else "no_close",
                mode=mode, trigger=causal_triggers(tr, thr, cfg),
                n_on_end=int(tr["on"][-1].sum()))


def need_failures(n_starts, r, gain=0.05):
    """Минимум провалов с триггером, чтобы r · n > gain · N."""
    return int(np.floor(gain * n_starts / r + 1e-9)) + 1


def auto_threshold(rows):
    """Порог апертуры: середина между p95 пустых и p5 удерживаемых."""
    held = [a["ap_hold"] for r in rows for a in r["attempts"] if a["kind"] == "lifted"]
    empty = [a["ap_hold"] for r in rows for a in r["attempts"] if a["kind"] == "miss"]
    if not held:
        raise SystemExit("нет удерживаемых захватов — задайте --aperture")
    hi = float(np.quantile(held, 0.05))
    if not empty:
        return hi / 2, dict(held_p5=hi, empty_p95=None, overlap=False)
    lo = float(np.quantile(empty, 0.95))
    return ((lo + hi) / 2 if lo < hi else lo), dict(held_p5=hi, empty_p95=lo,
                                                    overlap=bool(lo >= hi))


def ceiling(f, c_f, r, c_s=0.0, ell=0.0):
    """Формула §5 плана: Δ = f·c_f·r − (1 − f)·c_s·ℓ (доля)."""
    return f * c_f * r - (1 - f) * c_s * ell


# --- среда с пробой ----------------------------------------------------------------
_LIVE = [None]


def _probe(data):
    live = _LIVE[0]
    if live is None or getattr(live, "env", None) is None:
        raise RuntimeError("проба до создания среды")
    inner = live.env.env
    sim = inner.sim
    if "camera" in data:
        from robosuite.utils.camera_utils import get_camera_transform_matrix
        return np.asarray(get_camera_transform_matrix(
            sim, data["camera"], data["h"], data["w"]), np.float64).tolist()
    names, region = data["names"], data["region"]
    missing = [n for n in names if n not in inner.obj_body_id]
    if missing:
        raise RuntimeError(f"нет тел {missing}; есть: {sorted(inner.obj_body_id)}")
    states = inner.object_states_dict
    if region not in states:
        raise RuntimeError(f"нет региона {region}; есть: {sorted(states)}")
    pos = [np.array(sim.data.body_xpos[inner.obj_body_id[n]], np.float64).tolist()
           for n in names]
    grasp = []
    for n in names:
        try:
            grasp.append(int(bool(inner._check_grasp(
                gripper=inner.robots[0].gripper, object_geoms=inner.get_object(n)))))
        except Exception:                                    # noqa: BLE001
            grasp.append(-1)
    on = [bool(states[region].check_ontop(states[n])) for n in names]
    return dict(pos=pos, grasp=grasp, on=on)


class _ProbeConn:
    """Канал воркера K-36: команды probe обслуживаются здесь, остальные
    уходят в неизменённый k36_bank._worker."""

    def __init__(self, conn, probe=_probe):
        self.conn, self.probe = conn, probe

    def recv(self):
        while True:
            cmd, data = self.conn.recv()
            if cmd != "probe":
                return cmd, data
            try:
                self.conn.send(("ok", self.probe(data)))
            except Exception as e:                           # noqa: BLE001
                self.conn.send(("err", f"{e}\n{traceback.format_exc()}"))

    def send(self, msg):
        self.conn.send(msg)


def _probe_worker(conn):
    """Воркер BankSim, запоминающий живой объект K33Env для пробы."""
    import k33_bank as kb
    import k36_bank as kb36
    base = kb.make_env_class

    def recording():
        cls = base()

        class Recorded(cls):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                _LIVE[0] = self
        return Recorded
    kb.make_env_class = recording
    kb36._worker(_ProbeConn(conn))


def audit_sim(space, bank_dir):
    """BankSim того же рецепта, но с воркером _probe_worker."""
    import k36_bank as kb36

    class AuditSim(kb36.BankSim):
        def probe(self, names=OBJECTS, region=REGION):
            return self._call("probe", dict(names=list(names), region=region))

        def camera(self, name=CAMERA, h=IMAGE, w=IMAGE):
            return np.asarray(self._call("probe", dict(camera=name, h=h, w=w)))

    orig = kb36._worker
    kb36._worker = _probe_worker
    try:
        return AuditSim("bank", space, bank_dir=bank_dir)
    finally:
        kb36._worker = orig


class _SeedIO:
    """io для k39_cache.Replay без модели: среда в отдельном процессе,
    поэтому сид основного процесса на физику не влияет."""

    def seed(self, s):
        import random
        np.random.seed(int(s))
        random.seed(int(s))


# --- архив --------------------------------------------------------------------------
def archive_jobs(archive):
    """Все T1 задачи 8 c2 со сверкой merged.json, журналов частей и записей."""
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    from k39_failure_probe import reference

    plan, k36 = kcf.load_plan(str(archive))
    C = kh.consts(k36)
    merged = pc.load_json(archive / "merged.json")
    if merged.get("ok") is not True or merged.get("plan_sha256") != plan["sha256"]:
        raise SystemExit("merged.json не подтверждает архивный план")
    indexed = {r["eid"]: r for r in merged["rows"]
               if r["policy"] == "T1" and r["task"] == TASK}
    jobs = []
    for part in range(kcf.N_PARTS):
        ledger = pc.load_json(Path(kcf.part_dir(str(archive), part)) / "ledger.json")
        if ledger["pending"] is not None or ledger["caps"] != kcf.CAP_PART:
            raise SystemExit(f"часть {part}: открытый резерв или неверные лимиты")
        for w in kcf.part_wants(plan, k36, part):
            if w["task"] != TASK or w["kind"] != "T":
                continue
            r = indexed.pop(w["eid"], None)
            rec, z = reference(archive, part, w)
            problems = kh.episode_problems(rec, z, w, C)
            if (r is None or problems or r["success"] is not rec["summary"]["success"]
                    or ledger["done"].get(w["eid"]) != {"scope": "confirm",
                                                       "actual": rec["phys"]}):
                raise SystemExit(f"{w['eid']}: архив не прошёл сверку ({problems})")
            jobs.append(dict(want=w, part=part, success=rec["summary"]["success"],
                             npz_sha1=rec["npz_sha1"]))
    if indexed:
        raise SystemExit("в merged.json остались записи T1 t8 вне плана")
    return sorted(jobs, key=lambda j: j["want"]["eid"])


def trace_name(want):
    return f"s{want['seed']}_{want['case_id']}"


def mark(img, pix, color):
    h, w = img.shape[:2]
    r, c = int(round(pix[0])), int(round(pix[1]))
    for d in range(-4, 5):
        for rr, cc in ((r + d, c), (r, c + d)):
            if 0 <= rr < h and 0 <= cc < w:
                img[rr, cc] = color


def project(M, xyz):
    """Мир -> (строка, столбец) по матрице robosuite."""
    p = M @ np.append(np.asarray(xyz, np.float64), 1.0)
    return p[1] / p[2], p[0] / p[2]


def marked_frame(obs, M, pots):
    from k39_cache import CAMERAS
    av = np.array(obs[CAMERAS[0]][0], np.uint8, copy=True)
    h, w = av.shape[:2]
    for xyz in pots:
        r, c = project(M, xyz)
        mark(av, (r, c), (255, 0, 0))
        mark(av, (h - 1 - r, c), (0, 0, 255))
        mark(av, (r, w - 1 - c), (0, 255, 0))
    return [av, np.array(obs[CAMERAS[1]][0], np.uint8, copy=True)]


def replay_trace(sim, job, rec, z, frames_png=None):
    """Повтор архивного T1 с пробой на каждом шаге -> dict трассы."""
    from k39_cache import Replay
    w = job["want"]
    rp = Replay(sim, _SeedIO(), w["task"], w["reset_id"])
    q = rec["summary"]
    if (rp.ih[0][0], rp.ih[1][0]) != (q["init_hash"], q["init_hash_full"]):
        raise SystemExit(f"{w['eid']}: начальные хеши ≠ архиву")
    A = np.asarray(z["actions"], np.float32)
    M = sim.camera()
    ev = close_events(A)
    want_frames = set()
    if frames_png is not None and ev:
        want_frames = {min(ev[0] + d, len(A)) for d in (0, 8, 16, 40)}
    states, pots, grasp, on, frames = [], [], [], [], {}

    def look(step):
        p = sim.probe()
        states.append(np.asarray(rp.obs["state"][0], np.float64))
        pots.append(p["pos"])
        grasp.append(p["grasp"])
        on.append(p["on"])
        if step in want_frames:
            frames[step] = marked_frame(rp.obs, M, p["pos"])
    look(0)
    for step, a in enumerate(A, 1):
        rp.step(a)
        look(step)
    if (rp.reward >= 1, rp.succ_step, rp.done_step, rp.steps) != (
            q["success"], q["success_step"], q["done_step"], q["steps"]):
        raise SystemExit(f"{w['eid']}: повтор T1 ≠ архивному исходу или terminal")
    if frames:
        from k39_failure_probe import save_frames
        save_frames(frames_png, frames)
    return dict(A=A, state=np.stack(states), pot=np.asarray(pots, np.float64),
                grasp=np.asarray(grasp, np.int8), on=np.asarray(on, np.bool_), cam=M)


def save_trace(path, tr, job):
    w = job["want"]
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **tr, eid=np.array(w["eid"]), npz_sha1=np.array(job["npz_sha1"]),
             success=np.array(bool(job["success"])), seed=np.array(int(w["seed"])),
             case_id=np.array(w["case_id"]), names=np.array(OBJECTS))
    os.replace(tmp, path)


def load_trace(path):
    z = dict(np.load(path, allow_pickle=False))
    return dict(z, eid=str(z["eid"]), npz_sha1=str(z["npz_sha1"]),
                success=bool(z["success"]), seed=int(z["seed"]),
                case_id=str(z["case_id"]))


def run(a):
    import k37_bank as kbk
    import k37_confirm as kcf
    from k39_failure_probe import reference
    i, n = (int(x) for x in a.shard.split("/"))
    jobs = [j for k, j in enumerate(archive_jobs(a.archive)) if k % n == i]
    tdir, fdir = a.out / "traces", a.out / "frames"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    print(f"  часть {i}/{n}: {len(jobs)} эпизодов T1 t{TASK}", flush=True)
    sim, n_frames, t0 = None, 0, time.time()
    try:
        sim = audit_sim(kbk.SPACE, kbk.BANK_DIR)
        for k, j in enumerate(jobs):
            name = trace_name(j["want"])
            path = tdir / f"{name}.npz"
            if path.exists():
                old = load_trace(path)
                if old["eid"] != j["want"]["eid"] or old["npz_sha1"] != j["npz_sha1"]:
                    raise SystemExit(f"{path}: трасса от другого эпизода")
                continue
            rec, z = reference(a.archive, j["part"], j["want"])
            png = None
            if not j["success"] and n_frames < a.frames:
                png, n_frames = fdir / f"{name}.png", n_frames + 1
            tr = replay_trace(sim, j, rec, z, png)
            save_trace(path, tr, j)
            ev = close_events(tr["A"])
            print(f"  [{k + 1}/{len(jobs)}] {name}: T1 {int(j['success'])}, "
                  f"закрытий {len(ev)}, {time.time() - t0:.0f} с", flush=True)
    finally:
        if sim is not None:
            kcf.close_sim(sim)


# --- отчёт --------------------------------------------------------------------------
def quant(x):
    if not len(x):
        return "—"
    q = np.quantile(np.asarray(x, np.float64), [0.05, 0.5, 0.95])
    return f"n={len(x)} p5 {q[0] * 1e3:.1f} / p50 {q[1] * 1e3:.1f} / p95 {q[2] * 1e3:.1f} мм"


def trigger_stats(rows, cond):
    """Срабатывания условия cond на провалах/успехах и самовосстановления."""
    out = {}
    for g, s in (("fail", False), ("success", True)):
        forks = [r["trigger"][cond] for r in rows if r["success"] is s and r["trigger"][cond]]
        out[g] = dict(n=sum(1 for r in rows if r["success"] is s), trig=len(forks),
                      self_recovered=sum(f["outcome"] == "self_recovered" for f in forks),
                      stage=dict(Counter(f["stage"] for f in forks)))
    return out


def summarize(rows, thr, info, cfg=CFG):
    fail = [r for r in rows if not r["success"]]
    succ = [r for r in rows if r["success"]]
    N, F = len(rows), len(fail)
    out = dict(n=N, fail=F, success=len(succ), f=F / N if N else 0.0, threshold=thr,
               threshold_info=info, cfg=cfg)
    out["first"] = {g: dict(Counter(r["first"] for r in grp))
                    for g, grp in (("fail", fail), ("success", succ))}
    out["mode"] = dict(Counter(f"{r['mode']['kind']}@{r['mode']['stage']}" for r in fail))
    miss = [(r, a) for r in rows for a in r["attempts"] if a["kind"] == "miss"]
    out["regrasp_after_miss"] = {g: [sum(a["regrasp"] for r, a in miss if r["success"] is s),
                                     sum(1 for r, _a in miss if r["success"] is s)]
                                 for g, s in (("fail", False), ("success", True))}
    out["episodes_with_miss"] = {g: sum(any(a["kind"] == "miss" for a in r["attempts"])
                                        for r in grp)
                                 for g, grp in (("fail", fail), ("success", succ))}
    seeds = sorted({r["seed"] for r in rows})
    out["by_seed"] = {}
    for sd in ["all", *seeds]:
        grp = rows if sd == "all" else [r for r in rows if r["seed"] == sd]
        d = dict(n=len(grp), fail=sum(not r["success"] for r in grp),
                 need={"1.0": need_failures(len(grp), 1.0),
                       "0.4": need_failures(len(grp), 0.4)})
        for cond in ("a", "b"):
            st = trigger_stats(grp, cond)
            st["meets"] = {k: st["fail"]["trig"] >= v for k, v in d["need"].items()}
            d[cond] = st
        out["by_seed"][str(sd)] = d
    out["why_a"] = dict(sum((Counter(r["trigger"]["why"]) for r in rows), Counter()))
    out["ap_hold"] = {k: quant([a["ap_hold"] for r in rows for a in r["attempts"]
                                if a["kind"] == k]) for k in ("lifted", "miss")}
    out["eef_rise_miss"] = quant([a["eef_rise"] for _r, a in miss])
    out["contact_target"] = dict(Counter(a["by"] for r in rows for a in r["attempts"]))
    return out


def print_summary(s):
    print(f"\n  T1 t{TASK}: эпизодов {s['n']}, провалов {s['fail']} (f = {s['f']:.3f}), "
          f"успехов {s['success']}")
    print(f"  порог апертуры {s['threshold'] * 1e3:.1f} мм ({s['threshold_info']})")
    print("\n  Первая попытка захвата (описательно)  провалы   успехи")
    for k in BASE:
        print(f"    {k:<34} {s['first']['fail'].get(k, 0):>8} "
              f"{s['first']['success'].get(k, 0):>8}")
    print("\n  Режим провала (вид@стадия: 0 — первая кофеварка, 1 — вторая)")
    for k, v in sorted(s["mode"].items(), key=lambda x: -x[1]):
        print(f"    {k:<34} {v:>8}")
    print(f"\n  Эпизоды с miss хоть раз: провалы {s['episodes_with_miss']['fail']}, "
          f"успехи {s['episodes_with_miss']['success']}; повторный захват той же "
          f"кофеварки после miss [да, всего]: {s['regrasp_after_miss']}")
    print("\n  Причинный триггер: (a) проприоцепция как у раннера; "
          "(b) (a) + объект на месте и не на плите (sim)")
    print("  сид  усл.  провалы: триг (самовосст.)  успехи: триг (самовосст.)  "
          "нужно при r=1 / r=0.4  выполнено")
    for sd, d in s["by_seed"].items():
        for cond in ("a", "b"):
            st = d[cond]
            fl, sc = st["fail"], st["success"]
            print(f"  {sd:<4} ({cond})   {fl['trig']:>3}/{fl['n']:<4} ({fl['self_recovered']:>2})"
                  f"            {sc['trig']:>3}/{sc['n']:<4} ({sc['self_recovered']:>2})"
                  f"          ≥{d['need']['1.0']} / ≥{d['need']['0.4']}"
                  f"         r=1: {'да' if st['meets']['1.0'] else 'нет'}, "
                  f"r=0.4: {'да' if st['meets']['0.4'] else 'нет'}")
    print(f"  проверки границ (a) по причинам: {s['why_a']}")
    print(f"\n  Апертура в удержании: lifted {s['ap_hold']['lifted']}; miss {s['ap_hold']['miss']}")
    print(f"  Подъём кисти у miss: {s['eef_rise_miss']}")
    print(f"  Цель попытки определена по: {s['contact_target']}")


def report(a):
    import k36_protocol as pc
    paths = sorted((a.out / "traces").glob("*.npz"))
    paths = [p for p in paths if not p.name.endswith(".tmp.npz")]
    if not paths:
        raise SystemExit(f"нет трасс в {a.out / 'traces'}")
    trs = [load_trace(p) for p in paths]
    if not a.partial:
        want = {j["want"]["eid"]: j["npz_sha1"] for j in archive_jobs(a.archive)}
        got = {t["eid"]: t["npz_sha1"] for t in trs}
        if got != want:
            raise SystemExit(f"трассы ≠ архиву: есть {len(got)}, нужно {len(want)} "
                             "(или --partial)")
    cams = {np.asarray(t["cam"]).tobytes() for t in trs}
    rows0 = [dict(classify(t, np.inf), success=t["success"], seed=t["seed"]) for t in trs]
    thr, info = (a.aperture, "задан") if a.aperture is not None else auto_threshold(rows0)
    rows = [dict(classify(t, thr), success=t["success"], seed=t["seed"], eid=t["eid"])
            for t in trs]
    s = summarize(rows, thr, info)
    s["camera_constant"] = len(cams) == 1
    print_summary(s)
    print(f"  калибровка {CAMERA} одинакова во всех эпизодах: {s['camera_constant']}")
    pc.save_json(str(a.out / "summary.json"), dict(s, rows=rows))
    print(f"\n  записано {a.out / 'summary.json'}")


# --- самопроверка ---------------------------------------------------------------------
def _toy(script, n=600):
    """Синтетическая трасса: script — список (шаг, что) событий.
    что: ('close', pot) — начать закрытие у pot; ('lift', pot|None, мм) —
    подъём кисти (pot — вместе с кофеваркой); ('open', place) — раскрыть,
    place: True — кофеварка на плите; ('push', pot) — сдвиг кофеварки."""
    A = np.zeros((n, 7), np.float32)
    A[:, 6] = OPEN
    state = np.zeros((n + 1, 9))
    state[:, 0], state[:, 1] = 0.04, -0.04
    pot = np.zeros((n + 1, 2, 3))
    pot[:, 0] = (0.0, 0.25, 0.9)
    pot[:, 1] = (0.05, 0.05, 0.9)
    eef = np.zeros((n + 1, 3))
    eef[:] = pot[0, 0] + (0, 0, 0.05)
    grasp = np.zeros((n + 1, 2), np.int8)
    on = np.zeros((n + 1, 2), np.bool_)
    cur, held, ap_c = None, False, 0.0
    for t, ev in sorted(script, key=lambda x: x[0]):
        if ev[0] == "close":
            cur, ap_c = ev[1], ev[2]
            eef[t:] = pot[t, cur] + (0, 0, 0.05)
            A[t:, 6] = CLOSE
            state[t + 1:, 0], state[t + 1:, 1] = ap_c / 2, -ap_c / 2
        elif ev[0] == "lift":
            k, mm = ev[1], ev[2] / 1e3
            for s in range(t, min(t + 10, n + 1)):
                eef[s:, 2] += mm / 10
                if k is not None:
                    pot[s:, k, 2] += mm / 10
                    grasp[s:, k] = 1
            held = k is not None
        elif ev[0] == "open":
            A[t:, 6] = OPEN
            state[t + 1:, 0], state[t + 1:, 1] = 0.04, -0.04
            if held and ev[1]:
                on[t + 1:, cur] = True
            if held:
                grasp[t + 1:, cur] = 0
                pot[t + 1:, cur, 2] = 0.9
            held = False
        elif ev[0] == "push":
            pot[t:, ev[1], 0] += 0.03
    state[:, 2:5] = eef
    return dict(A=A, state=state, pot=pot, grasp=grasp, on=on)


def selftest():
    from k39_pulse import grip_event
    rng = np.random.default_rng(42)
    for _ in range(300):
        A = np.zeros((int(rng.integers(5, 60)), 7))
        A[:, 6] = np.where(rng.random(len(A)) < 0.5, OPEN, CLOSE)
        ev = close_events(A)
        assert (ev[:1] or [None])[0] == grip_event(A)
        assert all(b > a for a, b in zip(ev, ev[1:]))
    thr = 0.005
    T = lambda r, c: r["trigger"][c]  # noqa: E731
    ok = _toy([(20, ("close", 0, 0.02)), (24, ("lift", 0, 60)), (60, ("open", True)),
               (100, ("close", 1, 0.02)), (104, ("lift", 1, 60)), (140, ("open", True))])
    r = classify(ok, thr)
    assert r["first"] == "lifted" and r["mode"]["kind"] == "no_attempt"
    assert r["mode"]["stage"] == 2 and r["n_on_end"] == 2
    assert T(r, "a") is None and T(r, "b") is None
    assert [a["by"] for a in r["attempts"]] == ["contact", "contact"]
    # промах первой: пустой подъём; триггер на границе 32 по данным до неё;
    # позже T32 сам захватывает ту же кофеварку -> исход самовосстановление
    miss = _toy([(20, ("close", 0, 0.0)), (25, ("lift", None, 40)), (50, ("open", False)),
                 (80, ("close", 0, 0.02)), (84, ("lift", 0, 60)), (120, ("open", True))])
    r = classify(miss, thr)
    assert r["first"] == "miss" and r["mode"] == dict(kind="miss", stage=0)
    assert T(r, "a")["b"] == 32 and T(r, "a")["g"] == 20 and T(r, "a")["stage"] == 0
    assert T(r, "a")["outcome"] == "self_recovered" and T(r, "b")["b"] == 32
    # случай ревью: триггер на 32, кофеварка поднята на 44 той же попыткой
    late = _toy([(20, ("close", 0, 0.0)), (25, ("lift", None, 40)),
                 (40, ("close", 0, 0.02)), (44, ("lift", 0, 60)), (80, ("open", True))])
    r = classify(late, thr)
    assert T(r, "a") is not None and T(r, "a")["b"] == 32
    assert T(r, "a")["outcome"] == "self_recovered"
    # без подъёма: раскрыл до границы
    nolift = _toy([(20, ("close", 0, 0.0)), (26, ("open", False))])
    r = classify(nolift, thr)
    assert r["first"] == "nolift" and T(r, "a") is None
    # уронил вне плиты: захват удачный -> триггера нет
    drop = _toy([(20, ("close", 0, 0.02)), (24, ("lift", 0, 60)), (60, ("open", False))])
    r = classify(drop, thr)
    assert r["first"] == "lifted" and r["mode"] == dict(kind="drop", stage=0)
    assert r["attempts"][0]["fate"] == "drop" and T(r, "a") is None
    assert classify(_toy([]), thr)["first"] == "no_close"
    # пальцы раскрыты при подъёме (не пусто) -> нет триггера, причина aperture
    r = classify(_toy([(20, ("close", 0, 0.02)), (25, ("lift", None, 40)),
                       (50, ("open", False))]), thr)
    assert T(r, "a") is None and r["trigger"]["why"].get("aperture", 0) > 0
    # мало времени
    r = classify(_toy([(550, ("close", 0, 0.0)), (551, ("lift", None, 40)),
                       (590, ("open", False))]), thr)
    assert T(r, "a") is None and r["trigger"]["why"].get("time", 0) > 0
    # сдвинутая кофеварка: (a) срабатывает, (b) — нет
    r = classify(_toy([(20, ("close", 0, 0.0)), (25, ("lift", None, 40)),
                       (28, ("push", 0)), (50, ("open", False))]), thr)
    assert T(r, "a") is not None and T(r, "b") is None
    # промах на второй кофеварке после установки первой
    r = classify(_toy([(20, ("close", 0, 0.02)), (24, ("lift", 0, 60)), (60, ("open", True)),
                       (100, ("close", 1, 0.0)), (101, ("lift", None, 40)),
                       (130, ("open", False))]), thr)
    assert T(r, "a")["stage"] == 1 and r["mode"] == dict(kind="miss", stage=1)
    # медленный подъём: к границам 24 и 32 подъёма ещё нет, окно 16 закрыто
    r = classify(_toy([(20, ("close", 0, 0.0)), (36, ("lift", None, 40)),
                       (70, ("open", False))]), thr)
    assert T(r, "a") is None and r["trigger"]["why"].get("no_lift", 0) > 0
    # граница на terminal (b = числу команд) не учитывается
    r = classify(_toy([(20, ("close", 0, 0.0)), (21, ("lift", None, 40))], n=32), thr)
    assert T(r, "a") is None
    r = classify(_toy([(20, ("close", 0, 0.0)), (21, ("lift", None, 40))], n=33), thr)
    assert T(r, "a") is not None and T(r, "a")["b"] == 32
    # порог по данным, сводка по сидам, нужное число провалов
    rows = [dict(classify(t, np.inf), success=s_, seed=0)
            for t, s_ in ((ok, True), (miss, False))]
    th, info = auto_threshold(rows)
    assert 0 < th < 0.02 and not info["overlap"], (th, info)
    s = summarize([dict(classify(t, th), success=x, seed=sd) for t, x, sd in
                   ((ok, True, 0), (miss, False, 0), (drop, False, 1))], th, info)
    assert s["by_seed"]["all"]["a"]["fail"]["trig"] == 1
    assert s["by_seed"]["0"]["a"]["fail"] == dict(n=1, trig=1, self_recovered=1,
                                                  stage={0: 1})
    assert s["by_seed"]["1"]["a"]["fail"]["trig"] == 0
    assert need_failures(160, 1.0) == 9 and need_failures(160, 0.4) == 21
    assert abs(ceiling(0.27, 5 / (96 * 0.27 * 0.40), 0.40) - 5 / 96) < 1e-12
    # канал пробы: probe обслуживается, остальное — дальше
    class Fake:
        def __init__(self, msgs):
            self.msgs, self.sent = list(msgs), []

        def recv(self):
            return self.msgs.pop(0)

        def send(self, m):
            self.sent.append(m)
    f = Fake([("probe", 1), ("probe", "bad"), ("step", 7)])
    pc = _ProbeConn(f, probe=lambda d: d + 1)
    assert pc.recv() == ("step", 7)
    assert f.sent[0] == ("ok", 2) and f.sent[1][0] == "err"
    M = np.zeros((4, 4))
    M[0, 0], M[1, 1], M[2, 3], M[3, 3] = 100, 50, 1, 1
    assert project(M, (1.0, 2.0, 0.0)) == (100.0, 100.0)
    print("k42_audit selftest ok")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    ap = argparse.ArgumentParser(description="K-42: офлайн-аудит промахов захвата t8")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--archive", type=Path, default=Path("reports/k37c/runs/c2"))
        p.add_argument("--out", type=Path, default=OUT)
        if name == "run":
            p.add_argument("--shard", default="0/1", help="I/N: эпизоды с номером ≡ I mod N")
            p.add_argument("--frames", type=int, default=6,
                           help="PNG кадров с проекцией для первых N провалов части")
        else:
            p.add_argument("--aperture", type=float, default=None,
                           help="порог пустой апертуры, м (по умолчанию — авто)")
            p.add_argument("--partial", action="store_true",
                           help="разрешить неполный набор трасс")
    a = ap.parse_args()
    if a.out.resolve().is_relative_to(a.archive.resolve()):
        raise SystemExit("каталог аудита должен быть вне архива c2")
    if a.cmd == "run":
        run(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
