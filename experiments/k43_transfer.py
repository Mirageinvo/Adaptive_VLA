"""K-43, этап 2: второй перенос кофеварки t8 из состояния T32 (v0).

Диагностическая ветвь: позы кофеварок, плиты и контакт берутся из
симулятора (проба в воркере BankSim). Подкоманды вызываются из
k43_handoff.py:

  fit-templates  повтор успешных архивных T1 (c2, s0 и s1) с пробой в
                 ключевых шагах второго переноса -> каталог шаблонов:
                 поза кисти в системе второй кофеварки перед захватом и при
                 закрытии, высота переноса, поза поставленной кофеварки в
                 системе плиты и расстояние между кофеварками на плите.
                 Dev-сцены smoke выбираются до шаблонов и исключаются из
                 источников по case_id (оба сида).
  smoke          GPU, живой T32 s0 на 3–5 dev-сценах c2 с натуральным
                 префиксом: T32 с наблюдателем = архив побитово, развилка
                 наблюдателя = аудиту, ветвь «второй перенос» из префикса
                 (хеш развилки), пробы осей и поворота, прерывание опции с
                 возвратом T32; длительности фаз для бюджета.

Опция (каждый чанк — 8 шагов постоянной дельта-команды P-регулятора K-42,
pctl/to_codec, команда пересчитывается по новому наблюдению и пробе):
  up       свободная кисть до высоты переноса z_carry;
  over     над позой подхода к второй кофеварке (только после зазора);
  pre      поза подхода (позиция и поворот из шаблона);
  descend  поза захвата;
  close    закрытие; проверка удержания нужной кофеварки (проба grasp);
  lift     подъём до z_carry; проверка, что кофеварка поднялась;
  carry    над целевой позой (поза кисти = T_goal · inv(T_eef_obj),
           T_eef_obj измерена после захвата; только после зазора);
  lower    опускание к целевой позе;
  release  раскрытие; retreat — подъём кисти.
Остановка — бюджет фазы, срыв захвата, нет подъёма, недостаточный зазор;
управление возвращается T32 из фактического состояния.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np

TASK = 8
H8 = 8
MAX_STEPS = 600
OPEN, CLOSE = -1.0, 1.0
OBJECTS = ("moka_pot_1", "moka_pot_2")
STOVE = "flat_stove_1"
REGION = "flat_stove_1_cook_region"
OUT = Path("reports/k43/handoff/v1")
TEMPLATES = OUT / "templates.json"
CONFIG = OUT / "config.json"
OK, TECH = 0, 3

PHASES = ("up", "over", "pre", "descend", "close", "lift", "carry", "lower",
          "release", "retreat")
CFG0 = dict(
    # контроллер (как pctl K-42; pos_m — м на единицу команды за шаг)
    gain=1.0, pos_m=0.05, umax=0.6, rot_gain=1.0, rot_rad=0.5, umax_rot=0.3,
    # допуски и запасы
    tol_xy=0.010, tol_z=0.010, tol_rot=0.20, clear_tol=0.015, pot_lift=0.015,
    drop_m=0.010, retreat_m=0.06, pair_margin=0.9,
    # максимум чанков на фазу
    max_chunks=dict(up=3, over=4, pre=3, descend=3, close=1, lift=3, carry=5,
                    lower=3, release=1, retreat=1),
    max_option_chunks=None)


def load_cfg():
    cfg = json.loads(json.dumps(CFG0))
    if CONFIG.exists():
        extra = json.load(open(CONFIG, encoding="utf-8"))
        bad = set(extra) - set(CFG0)
        if bad:
            raise SystemExit(f"{CONFIG}: неизвестные ключи {sorted(bad)}")
        for k, v in extra.items():
            if isinstance(v, dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist()
                                 if hasattr(o, "tolist") else str(o)))


# --- позы (кватернион x, y, z, w — порядок robosuite) ---------------------------------
def q2m(q):
    x, y, z, w = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def m2q(R):
    R = np.asarray(R, np.float64)
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[i] = s / 4
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    q = np.asarray(q)
    return q if q[3] >= 0 else -q


def wxyz2xyzw(q):
    q = np.asarray(q, np.float64)
    return np.array([q[1], q[2], q[3], q[0]])


def pose(p, q):
    return (np.asarray(p, np.float64), q2m(q))


def pmul(a, b):
    return (a[0] + a[1] @ b[0], a[1] @ b[1])


def pinv(a):
    return (-a[1].T @ a[0], a[1].T)


def pq(a):
    return a[0].tolist(), m2q(a[1]).tolist()


def ang(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.arccos(np.clip(c, -1, 1)))


# --- проба среды ----------------------------------------------------------------------
_LIVE = [None]


def _probe43(data):
    live = _LIVE[0]
    if live is None or getattr(live, "env", None) is None:
        raise RuntimeError("проба до создания среды")
    inner = live.env.env
    sim = inner.sim
    names, region = list(data.get("names", OBJECTS)), data.get("region", REGION)
    missing = [n for n in names if n not in inner.obj_body_id]
    if missing:
        raise RuntimeError(f"нет тел {missing}; есть: {sorted(inner.obj_body_id)}")
    states = inner.object_states_dict
    if region not in states:
        raise RuntimeError(f"нет региона {region}; есть: {sorted(states)}")
    out = dict(pos=[], quat=[], grasp=[], on=[])
    for n in names:
        i = inner.obj_body_id[n]
        out["pos"].append(np.array(sim.data.body_xpos[i], np.float64).tolist())
        out["quat"].append(wxyz2xyzw(sim.data.body_xquat[i]).tolist())
        try:
            out["grasp"].append(int(bool(inner._check_grasp(
                gripper=inner.robots[0].gripper, object_geoms=inner.get_object(n)))))
        except Exception:                                    # noqa: BLE001
            out["grasp"].append(-1)
        out["on"].append(bool(states[region].check_ontop(states[n])))
    st = data.get("stove", STOVE)
    if st in inner.obj_body_id:
        i = inner.obj_body_id[st]
        out["stove"] = dict(pos=np.array(sim.data.body_xpos[i]).tolist(),
                            quat=wxyz2xyzw(sim.data.body_xquat[i]).tolist())
    else:
        out["stove"] = None
    try:
        sid = sim.model.site_name2id(region)
        out["region"] = dict(pos=np.array(sim.data.site_xpos[sid]).tolist(),
                             size=np.array(sim.model.site_size[sid]).tolist(),
                             mat=np.array(sim.data.site_xmat[sid]).reshape(3, 3).tolist())
    except Exception:                                        # noqa: BLE001
        out["region"] = None
    try:
        rb = inner.robots[0]
        sid = rb.eef_site_id if not isinstance(rb.eef_site_id, dict) else \
            list(rb.eef_site_id.values())[0]
        out["eef"] = dict(pos=np.array(sim.data.site_xpos[sid]).tolist(),
                          quat=m2q(np.array(sim.data.site_xmat[sid]).reshape(3, 3)).tolist())
    except Exception as e:                                   # noqa: BLE001
        out["eef"] = dict(err=f"{type(e).__name__}: {e}")
    return out


def _worker43(conn):
    import k33_bank as kb
    import k36_bank as kb36
    import k42_audit as ka
    base = kb.make_env_class

    def recording():
        cls = base()

        class Recorded(cls):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                _LIVE[0] = self
        return Recorded
    kb.make_env_class = recording
    kb36._worker(ka._ProbeConn(conn, probe=_probe43))


def probe_sim(space, bank_dir):
    """BankSim того же рецепта с пробой K-43 в воркере."""
    import k36_bank as kb36

    class ProbeSim(kb36.BankSim):
        def probe(self):
            return self._call("probe", dict(names=list(OBJECTS), region=REGION))

    orig = kb36._worker
    kb36._worker = _worker43
    try:
        return ProbeSim("bank", space, bank_dir=bank_dir)
    finally:
        kb36._worker = orig


# --- шаблоны --------------------------------------------------------------------------
def second_transfer_steps(tr, h, cfg):
    """Индексы второго переноса в успешной трассе после развилки h ->
    dict(g2, a2, r2, m2) | None. g2 — закрытие, после которого вторая
    кофеварка удерживается и поднимается; r2 — первая open после g2, после
    которой вторая On до конца; a2 — самая высокая точка кисти за 24 шага до
    g2; m2 — самая высокая точка кисти между g2 и r2."""
    import k42_audit as ka
    A, gr, on, pot = tr["A"], tr["grasp"], tr["on"], tr["pot"]
    eef = tr["state"][:, 2:5]
    j, tau = h["second"], h["tau"]
    for g in ka.close_events(A):
        if g < tau:
            continue
        end = ka.close_end(A, g)
        if not (gr[g:end + 1, j] == 1).any():
            continue
        if pot[g:end + 1, j, 2].max() - pot[g, j, 2] < cfg["pot_lift"]:
            continue
        r2 = end
        if r2 >= len(A) or not on[r2 + 1:, j].all() or not on[-1].all():
            return None
        lo = max(tau, g - 24)
        a2 = int(lo + np.argmax(eef[lo:g + 1, 2]))
        m2 = int(g + np.argmax(eef[g:r2 + 1, 2]))
        return dict(g2=int(g), a2=a2, r2=int(r2), m2=m2, end=len(A))
    return None


def choose_dev(rows, n_fail=2, n_succ=2):
    """Dev-сцены smoke (s0, с развилкой): провал с самой поздней развилкой,
    ещё провал и успехи по фиксированному хешу eid."""
    def hh(r):
        return hashlib.sha256(r["eid"].encode()).hexdigest()
    s0 = [r for r in rows if r["seed"] == 0 and r["handoff"]["tau"] is not None]
    fail = sorted((r for r in s0 if not r["success"]), key=hh)
    succ = sorted((r for r in s0 if r["success"]), key=hh)
    late = max(fail, key=lambda r: (r["handoff"]["tau"], hh(r)), default=None)
    out = [late] if late else []
    out += [r for r in fail if r is not late][:n_fail - len(out)]
    out += succ[:n_succ]
    return out


def template_job(args):
    """Повтор архивного эпизода с пробой на шагах steps -> dict."""
    sim, job, rec, z, steps = args
    from k39_cache import Replay
    import k42_audit as ka
    w = job["want"]
    rp = Replay(sim, ka._SeedIO(), w["task"], w["reset_id"])
    q = rec["summary"]
    if (rp.ih[0][0], rp.ih[1][0]) != (q["init_hash"], q["init_hash_full"]):
        raise SystemExit(f"{w['eid']}: начальные хеши ≠ архиву")
    A = np.asarray(z["actions"], np.float32)
    want = set(steps)
    got = {}
    if 0 in want:
        got[0] = dict(sim.probe(), state=np.asarray(rp.obs["state"][0]).tolist())
    for s, a in enumerate(A, 1):
        rp.step(a)
        if s in want:
            got[s] = dict(sim.probe(), state=np.asarray(rp.obs["state"][0]).tolist())
    if (rp.reward >= 1, rp.steps) != (q["success"], q["steps"]):
        raise SystemExit(f"{w['eid']}: повтор ≠ архиву")
    return got


def eef_pose(p):
    s = np.asarray(p["state"], np.float64)
    return pose(s[2:5], s[5:9])


def obj_pose(p, k):
    return pose(p["pos"][k], p["quat"][k])


def stove_pose(p):
    st = p.get("stove")
    return pose(st["pos"], st["quat"]) if st else (np.zeros(3), np.eye(3))


def template_from(probes, ix, h):
    """Позы одного источника -> запись каталога."""
    j, k = h["second"], h["first"]
    pa, pg, pr, pm, pe = (probes[ix[x]] for x in ("a2", "g2", "r2", "m2", "end"))
    To = obj_pose(pg, j)
    rel_pre = pmul(pinv(To), eef_pose(pa))
    rel_grasp = pmul(pinv(To), eef_pose(pg))
    Tr = obj_pose(pr, j)
    eef_obj = pmul(pinv(eef_pose(pr)), Tr)
    Tend = obj_pose(pe, j)
    rel_place = pmul(pinv(stove_pose(pe)), Tend)
    d_pair = float(np.linalg.norm(np.asarray(pe["pos"][j])[:2] -
                                  np.asarray(pe["pos"][k])[:2]))
    eef_chk = None
    if "pos" in pg.get("eef", {}):
        ep = eef_pose(pg)
        es = pose(pg["eef"]["pos"], pg["eef"]["quat"])
        eef_chk = dict(dpos=float(np.linalg.norm(ep[0] - es[0])),
                       drot=ang(ep[1], es[1]))
    return dict(second=j, first=k, rel_pre=pq(rel_pre), rel_grasp=pq(rel_grasp),
                eef_obj_release=pq(eef_obj), rel_place=pq(rel_place),
                carry_z=float(np.asarray(pm["state"])[4]), d_pair=d_pair,
                stove_known=pe.get("stove") is not None, eef_check=eef_chk,
                dur=dict(grasp=ix["g2"] - h["tau"], release=ix["r2"] - h["tau"],
                         end=ix["end"] - h["tau"]))


def fit_templates(a):
    import k36_protocol as pc
    import k37_bank as kbk
    import k37_confirm as kcf
    import k42_audit as ka
    import k43_handoff as kh43
    from k39_failure_probe import reference
    aud = pc.load_json(str(OUT / "audit.json"))
    if aud.get("partial"):
        raise SystemExit("audit.json неполный — нужен полный аудит")
    rows = aud["rows"]
    dev = choose_dev(rows)
    dev_cases = {r["case_id"] for r in dev}
    trs = {}
    for p in sorted(Path(a.traces).glob("*.npz")):
        if not p.name.endswith(".tmp.npz"):
            t = ka.load_trace(p)
            trs[t["eid"]] = t
    jobs = {j["want"]["eid"]: j for j in ka.archive_jobs(Path(a.archive))}
    cands = []
    for r in rows:
        if not r["success"] or r["handoff"]["tau"] is None or \
                r["case_id"] in dev_cases:
            continue
        tr = trs[r["eid"]]
        h = kh43.handoff(tr, aud["open_min"])
        if h != r["handoff"]:
            raise SystemExit(f"{r['eid']}: развилка ≠ audit.json")
        ix = second_transfer_steps(tr, h, CFG0)
        if ix is not None:
            cands.append((hashlib.sha256(r["eid"].encode()).hexdigest(), r, h, ix))
    cands.sort(key=lambda x: x[0])
    cands = cands[:a.n_src]
    if len(cands) < 4:
        raise SystemExit(f"источников шаблонов {len(cands)} < 4")
    print(f"  dev-сцены: {[(r['eid'], r['success'], r['handoff']['tau']) for r in dev]}")
    print(f"  источников шаблонов: {len(cands)} (сцен {len({c[1]['case_id'] for c in cands})})",
          flush=True)
    sims = [probe_sim(kbk.SPACE, kbk.BANK_DIR) for _ in range(a.workers)]
    try:
        def work(i_c):
            i, (_h, r, h, ix) = i_c
            j = jobs[r["eid"]]
            rec, z = reference(Path(a.archive), j["part"], j["want"])
            steps = sorted({ix["a2"], ix["g2"], ix["r2"], ix["m2"], ix["end"]})
            got = template_job((sims[i % len(sims)], j, rec, z, steps))
            return template_from(got, ix, h)
        # каждой среде — свой поток; задачи одной среды идут последовательно
        out = [None] * len(cands)
        groups = [[(i, c) for i, c in enumerate(cands) if i % len(sims) == s]
                  for s in range(len(sims))]

        def run_group(g):
            for i, c in g:
                out[i] = work((i, c))
                print(f"  шаблон {i + 1}/{len(cands)}: {c[1]['eid']}", flush=True)
        with ThreadPoolExecutor(len(sims)) as ex:
            list(ex.map(run_group, groups))
    finally:
        for s in sims:
            kcf.close_sim(s)
    for t, c in zip(out, cands):
        t.update(eid=c[1]["eid"], case_id=c[1]["case_id"], seed=c[1]["seed"])
    rep = dict(kind="k43_templates", open_min=aud["open_min"],
               audit_sha256=sha256(OUT / "audit.json"),
               dev=[dict(eid=r["eid"], case_id=r["case_id"], seed=r["seed"],
                         success=r["success"], tau=r["handoff"]["tau"]) for r in dev],
               templates=out,
               carry_z=float(np.median([t["carry_z"] for t in out])),
               d_min=float(min(t["d_pair"] for t in out)),
               dur={k: np.quantile([t["dur"][k] for t in out], [0, .5, .9, 1]).tolist()
                    for k in ("grasp", "release", "end")},
               eef_check=[t["eef_check"] for t in out],
               stove_known=all(t["stove_known"] for t in out))
    pc.save_json(str(TEMPLATES), jsonable(rep))
    print(f"  шаблонов {len(out)}; carry_z {rep['carry_z']:.3f} м; d_min "
          f"{rep['d_min'] * 1e3:.0f} мм; плита известна {rep['stove_known']}")
    print(f"  длительность от tau (min/p50/p90/max): захват {rep['dur']['grasp']}, "
          f"отпускание {rep['dur']['release']}, конец {rep['dur']['end']}")
    print(f"  сверка позы кисти state vs site: {rep['eef_check'][:3]}")
    print(f"  записано {TEMPLATES}")
    return OK


# --- выбор цели и опция -----------------------------------------------------------------
def plan_goal(tpl, probe, first, second, eef_q, cfg):
    """Фиксированное правило v0: шаблон захвата с наименьшим поворотом
    кисти; место — первая запись каталога (порядок шаблонов), где вторая не
    ближе d_min · pair_margin к первой. -> dict | dict(ok=False, why)."""
    T = tpl["templates"]
    To = obj_pose(probe, second)
    Re = q2m(eef_q)
    best = None
    for i, t in enumerate(T):
        G = pmul(To, pose(*t["rel_grasp"]))
        a_ = ang(Re, G[1])
        if best is None or a_ < best[0]:
            best = (a_, i)
    gi = best[1]
    St = stove_pose(probe)
    p1 = np.asarray(probe["pos"][first])[:2]
    dmin = tpl["d_min"] * cfg["pair_margin"]
    for i, t in enumerate(T):
        Tg = pmul(St, pose(*t["rel_place"]))
        if np.linalg.norm(Tg[0][:2] - p1) >= dmin:
            return dict(ok=True, grasp_i=gi, place_i=i, goal=pq(Tg),
                        pre=pq(pmul(To, pose(*T[gi]["rel_pre"]))),
                        grasp=pq(pmul(To, pose(*T[gi]["rel_grasp"]))),
                        rot_to_grasp=best[0], first=int(first), second=int(second))
    return dict(ok=False, why="нет свободного места в каталоге")


class Transfer:
    """Опция второго переноса. next(state, probe, elapsed) -> codec [8, 7]
    | None (возврат T32)."""

    def __init__(self, plan, tpl, cfg, q, L, U):
        self.plan, self.cfg = plan, cfg
        self.q, self.L, self.U = q, L, U
        self.z_carry = float(tpl["carry_z"])
        self.i, self.n_phase, self.n_total = 0, 0, 0
        self.log, self.info = [], dict(stop=None, phases=[], eef_obj=None)
        self.finished = False
        self.eef_obj = None
        self.z_pot0 = None

    def _stop(self, why):
        self.info["stop"] = why
        self.finished = True
        return None

    def _advance(self):
        self.info["phases"].append(dict(phase=PHASES[self.i], chunks=self.n_phase))
        self.i += 1
        self.n_phase = 0

    def next(self, state, probe, elapsed):
        import k42_recovery as kr
        if self.finished:
            return None
        s = kr.st9(state)
        p, qc = s[2:5], s[5:9]
        cfg, pl = self.cfg, self.plan
        j = pl["second"]
        if self.log:
            self.log[-1]["end_pos"] = p.tolist()
        mc = cfg.get("max_option_chunks")
        if mc is not None and self.n_total >= mc:
            return self._stop("прерывание: лимит чанков опции")
        if MAX_STEPS - int(elapsed) < H8:
            return self._stop("лимит шагов")
        while True:
            if self.i >= len(PHASES):
                return self._stop("завершена")
            ph = PHASES[self.i]
            pre = pose(*pl["pre"])
            gr = pose(*pl["grasp"])
            T, qt, grip = None, None, OPEN
            done = False
            if ph == "up":
                done = p[2] >= self.z_carry - cfg["tol_z"]
                T, qt = np.array([p[0], p[1], self.z_carry]), qc
            elif ph == "over":
                if p[2] < self.z_carry - cfg["clear_tol"]:
                    return self._stop("зазор не достигнут перед over")
                done = np.linalg.norm(p[:2] - pre[0][:2]) <= cfg["tol_xy"]
                T, qt = np.array([pre[0][0], pre[0][1], self.z_carry]), m2q(pre[1])
            elif ph in ("pre", "descend"):
                tg = pre if ph == "pre" else gr
                e_ = np.abs(p - tg[0])
                done = e_[:2].max() <= cfg["tol_xy"] and e_[2] <= cfg["tol_z"] and \
                    ang(q2m(qc), tg[1]) <= cfg["tol_rot"]
                T, qt = tg[0], m2q(tg[1])
            elif ph == "close":
                if self.n_phase >= 1:
                    if int(probe["grasp"][j]) != 1:
                        return self._stop("нет удержания второй кофеварки")
                    self.z_pot0 = float(probe["pos"][j][2])
                    done = True
                T, qt, grip = gr[0], m2q(gr[1]), CLOSE
            elif ph == "lift":
                grip = CLOSE
                if int(probe["grasp"][j]) != 1:
                    return self._stop("срыв захвата при подъёме")
                done = p[2] >= self.z_carry - cfg["tol_z"]
                if done and float(probe["pos"][j][2]) - self.z_pot0 < cfg["pot_lift"]:
                    return self._stop("кофеварка не поднялась")
                T, qt = np.array([p[0], p[1], self.z_carry]), qc
            elif ph in ("carry", "lower"):
                grip = CLOSE
                if int(probe["grasp"][j]) != 1:
                    return self._stop(f"срыв захвата ({ph})")
                if self.eef_obj is None:
                    self.eef_obj = pmul(pinv(pose(p, qc)), obj_pose(probe, j))
                    self.info["eef_obj"] = pq(self.eef_obj)
                G = pmul(pose(*pl["goal"]), pinv(self.eef_obj))
                if ph == "carry":
                    if p[2] < self.z_carry - cfg["clear_tol"]:
                        return self._stop("зазор не достигнут перед carry")
                    done = np.linalg.norm(p[:2] - G[0][:2]) <= cfg["tol_xy"]
                    T = np.array([G[0][0], G[0][1], self.z_carry])
                else:
                    zt = G[0][2] + cfg["drop_m"]
                    done = abs(p[2] - zt) <= cfg["tol_z"] and \
                        np.linalg.norm(p[:2] - G[0][:2]) <= cfg["tol_xy"]
                    T = np.array([G[0][0], G[0][1], zt])
                qt = m2q(G[1])
            elif ph == "release":
                if self.n_phase >= 1:
                    done = True
                T, qt, grip = p.copy(), qc, OPEN
            elif ph == "retreat":
                if self.n_phase >= 1:
                    done = True
                T, qt = np.array([p[0], p[1], p[2] + cfg["retreat_m"]]), qc
            if done:
                self._advance()
                continue
            if self.n_phase >= cfg["max_chunks"][ph]:
                return self._stop(f"бюджет фазы {ph}")
            break
        E = kr.pctl(p, qc, T, qt, grip, cfg)
        codec, clipped = kr.to_codec(E, self.q, self.L, self.U)
        self.log.append(dict(phase=ph, elapsed=int(elapsed), start_pos=p.tolist(),
                             target=np.asarray(T).tolist(), grip=grip,
                             clipped=clipped))
        self.n_phase += 1
        self.n_total += 1
        return codec


# --- наблюдатель, обёртки ------------------------------------------------------------------
class Observer:
    """Точка передачи онлайн по правилу аудита (k43_handoff.boundary_flags)
    на данных ≤ текущей границы; триггер — один раз."""

    def __init__(self, open_min, cfg):
        self.open_min, self.cfg = open_min, cfg
        self.A, self.S, self.ON, self.GR, self.P = [], [], [], [], []
        self.trigger = None
        self.stopped = False

    def start(self, state, probe):
        self.S.append(np.asarray(state, np.float64))
        self.ON.append(probe["on"])
        self.GR.append(probe["grasp"])
        self.P.append(probe)

    def step(self, a, state, probe, done):
        import k43_handoff as kh
        if self.stopped:
            return
        self.A.append(np.asarray(a, np.float64))
        self.start(state, probe)
        t = len(self.A)
        if done:
            self.stopped = True
            return
        if self.trigger is None and t % H8 == 0:
            tr = dict(A=np.stack(self.A), state=np.stack(self.S),
                      on=np.asarray(self.ON, bool), grasp=np.asarray(self.GR, np.int8))
            f = kh.boundary_flags(tr, t, self.open_min, self.cfg)
            if f["progress"] and f["release"] and f["time"]:
                self.trigger = dict(tau=t, first=f["first"], second=1 - f["first"])


class RecSim:
    """Обёртка ProbeSim: проба после каждого шага после ожидания."""

    def __init__(self, sim, ob, wait):
        self.sim, self.ob, self.wait = sim, ob, wait
        self.obs = self.probe = None

    def reset(self, task, starts):
        self.n = 0
        self.obs = self.sim.reset(task, starts)
        return self.obs

    def step(self, actions):
        obs, r, d, s = self.sim.step(actions)
        self.obs = obs
        self.n += 1
        if self.n >= self.wait:
            self.probe = self.sim.probe()
            st = np.asarray(obs["state"], np.float64)[0]
            if self.n == self.wait:
                self.ob.start(st, self.probe)
            else:
                self.ob.step(np.asarray(actions, np.float64)[0], st, self.probe,
                             bool(d[0]))
        return obs, r, d, s

    def close(self):
        return self.sim.close()


class HandoffArm:
    """Агент run_episode поверх штатного Arm. T32 — только наблюдение (и
    план цели на развилке для записи); Transfer — опция с развилки, затем
    штатный T32. expect = dict(tau, hash) — развилка эпизода T32."""

    def __init__(self, arm, rs, C, cfg, tpl, branch, expect=None):
        self.arm, self.rs, self.C, self.cfg, self.tpl = arm, rs, C, cfg, tpl
        self.branch, self.expect = branch, expect
        self.begin_state()

    def begin_state(self):
        self.heavy, self.log = 0, []
        self.opt, self.started, self.plan, self.fork_hash = None, False, None, None
        self.cur = None

    call = property(lambda s: s.arm.call, lambda s, v: setattr(s.arm, "call", v))
    episode = property(lambda s: s.arm.episode,
                       lambda s, v: setattr(s.arm, "episode", v))

    def begin(self, task, init, episode=None):
        self.begin_state()
        self.arm.begin(task, init, episode)

    def decide(self, batch, po, s8, elapsed):
        import k31_collect as kcol
        tr = self.rs.ob.trigger
        if tr is not None and self.plan is None and tr["tau"] == elapsed:
            st = np.asarray(self.rs.obs["state"], np.float64)[0]
            self.plan = plan_goal(self.tpl, self.rs.probe, tr["first"], tr["second"],
                                  st[5:9], self.cfg)
            self.fork_hash = kcol.obs_hash(self.rs.obs)
            if self.branch == "Transfer":
                if self.expect is not None and (int(self.expect["tau"]) != elapsed or
                                                self.expect["hash"] != self.fork_hash):
                    raise kcol.Mismatch("развилка ветви ≠ T32")
                if self.plan["ok"]:
                    self.opt = Transfer(self.plan, self.tpl, self.cfg, self.C["q"],
                                        self.C["L"], self.C["U"])
                self.started = True
        if self.branch == "Transfer" and not self.started and \
                self.expect is not None and elapsed >= self.expect["tau"]:
            raise kcol.Mismatch("развилка не воспроизвелась на префиксе")
        if self.opt is not None and not self.opt.finished:
            st = np.asarray(self.rs.obs["state"], np.float64)[0]
            ch = self.opt.next(st, self.rs.probe, elapsed)
            if ch is not None:
                self.cur = "opt"
                self.log.append(dict(kind="opt", elapsed=int(elapsed)))
                return ch, dict(option=True, phase=self.opt.log[-1]["phase"])
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
        self.log.append(dict(kind="t32", elapsed=int(elapsed)))
        return np.asarray(B, np.float32), rec

    def executed(self, k):
        if self.cur == "opt":
            pass
        else:
            self.arm.executed(k)
        self.log[-1]["k"] = int(k)


def run_branch(sim, io, arm, C, cfg, tpl, open_min, branch, task, pos, eid, *,
               prefix=None, expect=None, wait=None):
    import k31_collect as kcol
    import k43_handoff as kh
    from k36_harness import episode_number
    ob = Observer(open_min, dict(kh.CFG))
    rs = RecSim(sim, ob, kcol.ENV["waiting_steps"] if wait is None else wait)
    ag = HandoffArm(arm, rs, C, cfg, tpl, branch, expect)
    arm.episode = episode_number(eid)
    ep = kcol.run_episode(rs, io, ag, task, pos, prefix=prefix)
    if branch == "Transfer" and not ag.started:
        raise kcol.Mismatch("развилка не достигнута")
    last = rs.probe or {}
    ep.update(heavy=ag.heavy, trigger=ob.trigger, plan=ag.plan, fork_hash=ag.fork_hash,
              option=None if ag.opt is None else dict(info=ag.opt.info, log=ag.opt.log,
                                                      chunks=ag.opt.n_total),
              log=ag.log, on_end=last.get("on"), branch=branch,
              t32_after=sum(1 for x in ag.log if x["kind"] == "t32" and
                            ag.opt is not None and x["elapsed"] >
                            (ob.trigger or {}).get("tau", 10 ** 9)))
    return ep


# --- smoke ---------------------------------------------------------------------------
def probe_rot(sim, io, C, cfg, task, pos, u=0.3):
    """Знак и масштаб поворота: +u по оси z 8 шагов; угол рысканья."""
    import k39_cache as kc39
    import k42_recovery as kr
    rp = kc39.Replay(sim, io, task, pos)
    q0 = kr.st9(np.asarray(rp.obs["state"])[0])[5:9]
    E = np.zeros((H8, 7))
    E[:, 6], E[:, 5] = OPEN, u
    codec, cl = kr.to_codec(E, C["q"], C["L"], C["U"])
    ex = io.scale(codec[None])[0]
    for t in range(H8):
        rp.step(ex[t])
    q1 = kr.st9(np.asarray(rp.obs["state"])[0])[5:9]
    e = kr.rot_err(q1, q0)
    return dict(rot_err=e.tolist(), per_unit_step_z=float(e[2] / (u * H8)), clipped=cl)


def smoke(a):
    import k36_harness as kh
    import k36_protocol as pc
    import k37_bank as kbk
    import k38_grip_run as kgr
    import k40b_eval_base as kb40
    import k31_collect as kcol
    import k42_audit as ka
    import k42_recovery as kr
    from k39_failure_probe import reference
    tpl = pc.load_json(str(TEMPLATES))
    if tpl.get("audit_sha256") != sha256(OUT / "audit.json"):
        raise SystemExit("шаблоны построены по другому audit.json")
    cfg = load_cfg()
    k36 = pc.load_prepare()
    C = kh.consts(k36)
    jobs = {j["want"]["eid"]: j for j in ka.archive_jobs(Path(a.archive))}
    aud = pc.load_json(str(OUT / "audit.json"))
    aud_tau = {r["eid"]: r["handoff"]["tau"] for r in aud["rows"]}
    dev = tpl["dev"][:a.n_dev]
    prob, rep = [], dict(config=cfg, templates_sha256=sha256(TEMPLATES), scenes=[])
    card = kb40.Card(a.device)
    try:
        card.start()
        sim = probe_sim(kbk.SPACE, kbk.BANK_DIR)
        card.sims.append(sim)
        v, arm = card.source(k36, 0)
        io = v["io"]
        j0 = jobs[dev[0]["eid"]]
        rep["axes"] = kr.probe_axes(sim, io, C, cfg, TASK, j0["want"]["reset_id"])
        rep["rot"] = probe_rot(sim, io, C, cfg, TASK, j0["want"]["reset_id"])
        pm = [d["per_unit_step"] for d in rep["axes"]["axes"].values()]
        rep["pos_m_measured"] = float(np.median(pm))
        if min(pm) <= 0:
            prob.append("ось против команды")
        if rep["rot"]["per_unit_step_z"] <= 0:
            prob.append("поворот z против команды (порядок кватерниона/знак)")
        print(f"  оси м/ед/шаг {[round(x, 4) for x in pm]}; поворот z рад/ед/шаг "
              f"{rep['rot']['per_unit_step_z']:.4f}", flush=True)
        n_full = 0
        interrupted = False
        for d in dev:
            j = jobs[d["eid"]]
            w = j["want"]
            rec, z = reference(Path(a.archive), j["part"], w)
            e0 = run_branch(sim, io, arm, C, cfg, tpl, tpl["open_min"], "T32", TASK,
                            w["reset_id"], d["eid"])
            q = rec["summary"]
            same = np.array_equal(np.asarray(e0["actions"], np.float32),
                                  np.asarray(z["actions"], np.float32)) and \
                all(e0[k] == q[k] for k in ("success", "steps", "done_step"))
            if not same:
                prob.append(f"{d['eid']}: T32 с наблюдателем ≠ архиву")
            tau = (e0["trigger"] or {}).get("tau")
            if tau != aud_tau[d["eid"]]:
                prob.append(f"{d['eid']}: развилка {tau} ≠ аудиту {aud_tau[d['eid']]}")
            row = dict(eid=d["eid"], t32_success=e0["success"], same=bool(same),
                       tau=tau, plan=e0["plan"], seconds=float(e0["seconds"]))
            if tau is not None:
                h = [c["hash"] for c in e0["calls"] if c["tau"] == tau]
                ex = dict(tau=tau, hash=h[0] if h else e0["fork_hash"])
                variants = [("full", cfg)]
                if not interrupted:
                    variants.append(("interrupt", dict(cfg, max_option_chunks=2)))
                    interrupted = True
                for name, cf in variants:
                    try:
                        eb = run_branch(sim, io, arm, C, cf, tpl, tpl["open_min"],
                                        "Transfer", TASK, w["reset_id"], d["eid"],
                                        prefix=np.asarray(e0["actions"])[:tau], expect=ex)
                    except kcol.Mismatch as e:
                        prob.append(f"{d['eid']} {name}: {e}")
                        continue
                    o = eb["option"] or {}
                    info = o.get("info", {})
                    full = info.get("stop") == "завершена" and \
                        bool(eb["on_end"] and all(eb["on_end"]))
                    if name == "full" and (full or eb["success"]):
                        n_full += int(full or eb["success"])
                    if name == "interrupt" and eb["t32_after"] < 1 and \
                            eb["done_step"] < 0:
                        prob.append(f"{d['eid']}: после прерывания T32 не продолжил")
                    row[name] = dict(success=eb["success"], steps=eb["steps"],
                                     stop=info.get("stop"), phases=info.get("phases"),
                                     chunks=o.get("chunks"), on_end=eb["on_end"],
                                     t32_after=eb["t32_after"], heavy=eb["heavy"],
                                     log=o.get("log"))
                    print(f"  {d['eid']} {name}: успех {eb['success']} (T32 "
                          f"{e0['success']}), остановка {info.get('stop')}, чанков "
                          f"{o.get('chunks')}, On {eb['on_end']}", flush=True)
            rep["scenes"].append(row)
        if n_full < 1:
            prob.append("ни одного полного второго переноса")
        arm = v = None
        kh.free(card.torch)
        rep.update(admitted=not prob, problems=prob, n_full=n_full,
                   env=kgr.env_record(card.torch), uuid=card.g["uuid"],
                   host=socket.gethostname(), time=time.time())
        dst = OUT / "smoke" / f"{socket.gethostname()}__{card.g['uuid']}.json"
        pc.save_json(str(dst), jsonable(rep))
    finally:
        card.close()
    print(f"  smoke {a.device}: " + ("ДОПУЩЕН" if not prob else
                                     "НЕ допущен: " + "; ".join(prob[:4])))
    print(f"  измеренный pos_m {rep.get('pos_m_measured')}; в конфиге {cfg['pos_m']}")
    return OK if not prob else TECH


# --- самопроверка ---------------------------------------------------------------------
class ToyPots:
    """Две кофеварки и плита. Кисть интегрирует команду (pos_m 0.03 м на
    единицу за шаг), поворот не меняется. Захват — close в 1 см от точки
    кофеварка + 5 см; удерживаемая едет с кистью; open — падает на стол
    (z = 0), On — в круге плиты r 0.1."""
    STOVE = np.array([0.0, 0.15, 0.0])
    POSM = 0.03

    def reset(self, task, starts):
        self.init = int(starts[0])
        self.eef = np.array([0.0, -0.15, 0.15])
        self.pots = np.array([[0.1, -0.1, 0.0], [-0.1, -0.1, 0.0]])
        self.held, self.ap, self.t = None, 0.08, 0
        return self._obs()

    def on(self):
        return [bool(np.linalg.norm(p[:2] - self.STOVE[:2]) < 0.1 and p[2] < 1e-6
                     and self.held != i) for i, p in enumerate(self.pots)]

    def probe(self):
        return dict(pos=self.pots.tolist(), quat=[[0, 0, 0, 1.0]] * 2,
                    grasp=[int(self.held == i) for i in range(2)], on=self.on(),
                    stove=dict(pos=self.STOVE.tolist(), quat=[0, 0, 0, 1.0]),
                    region=None, eef=dict(pos=self.eef.tolist(), quat=[0, 0, 0, 1.0]))

    def _obs(self):
        st = np.array([[self.ap / 2, -self.ap / 2, *self.eef, 0, 0, 0, 1.0]])
        img = np.zeros((1, 4, 4, 3), np.uint8)
        img[0, 0, 0, :2] = (np.clip(self.pots[:, 0] * 100 + 100, 0, 255)).astype(np.uint8)
        return dict(state=st, agentview_image=img, robot0_eye_in_hand_image=img.copy(),
                    toy_on=np.array([self.on()]), toy_held=np.array([
                        -1 if self.held is None else self.held]))

    def step(self, actions):
        a = np.asarray(actions, np.float64)[0]
        self.t += 1
        self.eef = self.eef + a[:3] * self.POSM
        if a[6] == CLOSE:
            if self.held is None:
                for i, p in enumerate(self.pots):
                    if np.linalg.norm(self.eef - (p + [0, 0, 0.05])) < 0.012:
                        self.held = i
                self.ap = 0.03 if self.held is not None else 0.0
        else:
            if self.held is not None:
                self.pots[self.held, 2] = 0.0
            self.held, self.ap = None, 0.08
        if self.held is not None:
            self.pots[self.held] = self.eef - [0, 0, 0.05]
        on = self.on()
        r = 1.0 if all(on) else 0.0
        return self._obs(), np.array([r]), np.array([r >= 1]), np.array([r >= 1])

    def close(self):
        pass


class ToyT32:
    """Скрипт «T32»: несёт кофеварку 0 на плиту в (−0.04, 0.15); init 1 —
    затем и кофеварку 1 в (0.05, 0.15); иначе после первой стоит на месте."""

    def __init__(self, q, L, U):
        self.q, self.L, self.U = q, L, U
        self.call, self.episode, self.last = 0, 0, None

    def begin(self, task, init, episode=None):
        self.init = int(init)

    def forward(self):
        return None

    def decide(self, batch, po, s8, elapsed):
        import k42_recovery as kr
        self.forward()
        o = batch
        s = kr.st9(o["state"][0])
        p = s[2:5]
        on, held = list(o["toy_on"][0]), int(o["toy_held"][0])
        goals = [np.array([-0.04, 0.15]), np.array([0.05, 0.15])]
        todo = [0] + ([1] if self.init == 1 else [])
        k = next((i for i in todo if not on[i]), None)
        grip, T = OPEN, p.copy()
        if held >= 0:
            grip = CLOSE
            g = goals[held]
            if np.linalg.norm(p[:2] - g) > 0.008:
                T = np.array([g[0], g[1], 0.15])
            elif p[2] > 0.06:
                T = np.array([g[0], g[1], 0.05])
            else:
                grip = OPEN
        elif k is not None:
            pot = o["toy_pots"][0][k]
            tg = pot + [0, 0, 0.05]
            if np.linalg.norm(p[:2] - tg[:2]) > 0.008:
                T = np.array([tg[0], tg[1], 0.15])
            elif p[2] > tg[2] + 0.008:
                T = tg
            else:
                grip, T = CLOSE, tg
        cfg = dict(CFG0, pos_m=ToyPots.POSM)
        E = kr.pctl(p, s[5:9], T, s[5:9], grip, cfg)
        return kr.to_codec(E, self.q, self.L, self.U)[0], {}

    def executed(self, k):
        pass


def selftest():
    import k31_collect as kcol
    import k43_handoff as kh
    # позы: композиция и обращение
    rng = np.random.default_rng(0)
    for _ in range(50):
        qa, qb = rng.normal(size=4), rng.normal(size=4)
        A_, B_ = pose(rng.normal(size=3), qa), pose(rng.normal(size=3), qb)
        I_ = pmul(A_, pinv(A_))
        assert np.allclose(I_[0], 0, atol=1e-9) and np.allclose(I_[1], np.eye(3))
        assert np.allclose(q2m(m2q(B_[1])), B_[1], atol=1e-9)
        C_ = pmul(pmul(A_, B_), pinv(B_))
        assert np.allclose(C_[0], A_[0]) and np.allclose(C_[1], A_[1])
    assert np.allclose(wxyz2xyzw([1, 0, 0, 0]), [0, 0, 0, 1])

    # игрушечная среда через run_episode
    q = np.array([1.0] * 6 + [1.0])
    L, U = -np.ones(6) * 2, np.ones(6) * 2
    C = dict(q=q, L=L, U=U)

    class SIO(kcol.FakeIO):
        max_act_q = q

        def prep(self, obs, desc, task):
            return dict(obs, toy_pots=np.array([self._sim.pots.copy()])), 0, \
                np.asarray(obs["state"][:, :8], np.float64)
    sim = ToyPots()
    io = SIO()
    io._sim = sim
    cfg = json.loads(json.dumps(CFG0))
    cfg["pos_m"] = ToyPots.POSM
    tpl = dict(templates=[dict(rel_pre=pq(pose([0, 0, 0.12], [0, 0, 0, 1])),
                               rel_grasp=pq(pose([0, 0, 0.05], [0, 0, 0, 1])),
                               rel_place=pq(pose(p_, [0, 0, 0, 1])))
                          for p_ in ([-0.04, 0.0, 0.0], [0.05, 0.0, 0.0],
                                     [0.0, -0.05, 0.0])],
               carry_z=0.15, d_min=0.06, open_min=0.06)
    om = 0.06
    arm = ToyT32(q, L, U)
    e0 = run_branch(sim, io, arm, C, cfg, tpl, om, "T32", TASK, 0, "toy", wait=10)
    assert not e0["success"] and e0["trigger"] is not None, e0["trigger"]
    tau = e0["trigger"]["tau"]
    assert e0["trigger"]["first"] == 0 and e0["plan"]["ok"]
    # место 0 занято первой (−0.04) -> выбрано место 1
    assert e0["plan"]["place_i"] == 1, e0["plan"]
    # нулевая ветвь (наблюдение) = T32 без обёртки
    e_plain = kcol.run_episode(sim, io, ToyT32(q, L, U), TASK, 0)
    assert np.array_equal(e_plain["actions"], e0["actions"])
    # тот же результат офлайн-правилом аудита по трассе наблюдателя
    h = [c["hash"] for c in e0["calls"] if c["tau"] == tau]
    ex = dict(tau=tau, hash=h[0])
    eb = run_branch(sim, io, ToyT32(q, L, U), C, cfg, tpl, om, "Transfer", TASK, 0,
                    "toy", prefix=np.asarray(e0["actions"])[:tau], expect=ex, wait=10)
    info = eb["option"]["info"]
    assert eb["success"], (info, eb["option"]["log"][-3:])
    assert [x["phase"] for x in info["phases"]][:7] == list(PHASES[:7]), info
    assert np.array_equal(eb["actions"][:tau], e0["actions"][:tau])
    # прерывание: лимит чанков -> возврат T32 из фактического состояния
    cf2 = dict(cfg, max_option_chunks=2)
    ei = run_branch(sim, io, ToyT32(q, L, U), C, cf2, tpl, om, "Transfer", TASK, 0,
                    "toy", prefix=np.asarray(e0["actions"])[:tau], expect=ex, wait=10)
    assert ei["option"]["info"]["stop"].startswith("прерывание") and ei["t32_after"] > 0
    assert ei["heavy"] == sum(1 for x in ei["log"] if x["kind"] == "t32")
    # чужой хеш развилки -> отказ
    try:
        run_branch(sim, io, ToyT32(q, L, U), C, cfg, tpl, om, "Transfer", TASK, 0,
                   "toy", prefix=np.asarray(e0["actions"])[:tau],
                   expect=dict(ex, hash="x"), wait=10)
        raise AssertionError("чужой хеш принят")
    except kcol.Mismatch:
        pass
    # успешный T32 (init 1): развилка есть, ветвь не обязана её портить
    s1 = run_branch(sim, io, ToyT32(q, L, U), C, cfg, tpl, om, "T32", TASK, 1, "toy",
                    wait=10)
    assert s1["success"] and s1["trigger"] is not None
    # нет места: каталог только на месте первой
    tpl2 = dict(tpl, templates=tpl["templates"][:1])
    pr = sim.probe()
    pr["pos"][0] = [-0.04, 0.15, 0.0]
    assert not plan_goal(tpl2, pr, 0, 1, [0, 0, 0, 1], cfg)["ok"]
    # зазор: кисть ниже z_carry перед over -> остановка без бокового движения
    tr = Transfer(e0["plan"], dict(tpl, carry_z=0.5), dict(cfg, max_chunks=dict(
        cfg["max_chunks"], up=1)), q, L, U)
    st = np.array([0.04, -0.04, 0.0, 0.0, 0.15, 0, 0, 0, 1.0])
    assert tr.next(st, pr, 300) is not None
    assert tr.next(st, pr, 308) is None and tr.info["stop"] == "бюджет фазы up"
    assert kh.CFG["stable"] == 8
    print("k43_transfer selftest ok")
