#!/usr/bin/env python3
"""K-40 B: повтор 384 архивных эпизодов c2 с захватом окон для TreeB.

Выборка — как в k40b_audit (только проверенные записи): в группе
задача × сид × split m = min(спасений, ⌊успехов T1/2⌋); роли R (спасение
T2P, иначе T2), T1a (общие успехи T1), T1b (успехи T1 только для
контроля TreeB-T1), первые по хешу.
Окна: M — число полных (k = 8) чанков журнала; i_k = ⌊k(M−1)/7⌋, k = 0…7.
В окне с началом t:
  точка 0 — один штатный проход T32 (Arm.decide) с перехватом признаков
  всех узлов (x0/x1/x2, W0–W2, iv0–iv2), S0, прогноз B (= j_a_codec[i]) и
  визуальных признаков того же прохода;
  точки 2/4/6/8 — только get_image_features (энкодер + коннектор);
  визуальные токены в порядке подачи → 4 последовательные группы → средние
  (4 × D, fp32); state 9, state8 модели и state11 в пяти точках; кадры
  обеих камер (uint8, без потерь);
  цель: T1 — j_a_codec[i], T2/T2P — k_a2[i]; масштаб цели = исполненным
  действиям архива на шагах t…t+7.
Повтор целиком сверяется с архивом (начальные хеши, исход, terminal).
Команды: --prepare-only | --part K --device D | --status | --selftest.
"""

import argparse
from collections import Counter
from pathlib import Path
import socket

import numpy as np

PROTOCOL = "k40b-collect-v1"
RESERVE = 610
SCOPE = "k40b_collect"
N_PARTS = 2
N_WIN, WIN = 8, 8
POINTS = (0, 2, 4, 6, 8)
N_GROUPS = 4
F_KEYS = ("x0", "x1", "x2", "W0", "W1", "W2", "iv0", "iv1", "iv2")
F_SHAPES = dict(x0=(4, 768), x1=(8, 768), x2=(16, 768), W0=(20, 4),
                W1=(20, 8), W2=(20, 16), iv0=(4, 2), iv1=(8, 2),
                iv2=(16, 2))
S_DIM = 6 * 768 + 72
CAMS = ("agentview_image", "robot0_eye_in_hand_image")
IMG = (224, 224, 3)
CODE = ("k40b_collect.py", "k40b_audit.py", "k40a_transfer.py",
        "k39_cache.py")


def rank(key):
    import k40b_audit as ka
    return ka.rank(key)


def window_starts(J):
    """Журнал -> индексы вызовов окон (полные чанки, равномерно)."""
    full = [i for i in np.argsort(J["elapsed"]) if int(J["k"][i]) == WIN]
    M = len(full)
    if M < N_WIN:
        raise SystemExit(f"полных чанков {M} < {N_WIN}")
    return [int(full[(k * (M - 1)) // (N_WIN - 1)]) for k in range(N_WIN)]


def group_vis(img):
    """Визуальные токены [.., n, D] в порядке подачи -> [4, D] средних."""
    x = np.asarray(img, np.float32)
    x = x.reshape(-1, x.shape[-1])
    if len(x) < N_GROUPS:
        raise SystemExit(f"визуальных токенов {len(x)} < {N_GROUPS}")
    return np.stack([g.mean(0) for g in np.array_split(x, N_GROUPS)])


def select(rows):
    """Проверенные строки -> [dict(eid, role, split, task, seed, case_id,
    policy)] по правилу аудита."""
    import k40b_audit as ka
    from collections import defaultdict
    by = defaultdict(dict)
    for r in rows:
        by[(r["task"], r["case_id"], r["seed"])][r["policy"]] = r
    split = ka.split_scenes(rows)
    resc, t1s = defaultdict(list), defaultdict(list)
    for (t, c, s), d in sorted(by.items()):
        g = (t, s, split[c])
        if d["T1"]["success"]:
            t1s[g].append(d["T1"])
            continue
        pick = "T2P" if d["T2P"]["success"] else (
            "T2" if d["T2"]["success"] else None)
        if pick:
            resc[g].append(d[pick])
    out = []
    for g in sorted(set(resc) | set(t1s)):
        m = min(len(resc[g]), len(t1s[g]) // 2)
        R = sorted(resc[g], key=lambda r: rank(r["eid"]))[:m]
        T = sorted(t1s[g], key=lambda r: rank(r["eid"]))
        for role, xs in (("R", R), ("T1a", T[:m]), ("T1b", T[m:2 * m])):
            out += [dict(eid=r["eid"], role=role, split=g[2], task=r["task"],
                         seed=r["seed"], case_id=r["case_id"],
                         policy=r["policy"], arch_part=r["part"],
                         phys=r["phys"]) for r in xs]
    return out


def prepare(a):
    import k36_harness as kh
    import k37_confirm as kcf
    import k40b_audit as ka
    rep = dict(problems=[])
    rows = ka.verified_rows(a.archive, rep)
    if rows is None or rep["problems"]:
        raise SystemExit("архив: " + "; ".join(rep["problems"][:3]))
    plan, k36 = kcf.load_plan(a.archive)
    sel = select(rows)
    wants = {}
    for k in range(kcf.N_PARTS):
        for w in kcf.part_wants(plan, k36, k):
            wants[w["eid"]] = w
    val = kcf.validator(kh.consts(k36))
    eps = []
    for e in sorted(sel, key=lambda e: (e["seed"], rank(e["eid"]))):
        w = wants[e["eid"]]
        kcf.install(kcf.part_dir(a.archive, e["arch_part"]))
        got = kh.load_ep(w)
        if got is None:
            raise SystemExit(f"{e['eid']}: нет записи")
        rec, z = got
        pr = val(rec, z, w)
        if pr:
            raise SystemExit(f"{e['eid']}: {pr[:2]}")
        J = {k[2:]: z[k] for k in z if k.startswith("j_")}
        e.update(want=w, npz_sha1=rec["npz_sha1"],
                 success=bool(rec["summary"]["success"]),
                 windows=window_starts(J),
                 t=[int(J["elapsed"][i]) for i in window_starts(J)])
        if (e["role"] == "R") != (e["policy"] != "T1") or not e["success"]:
            raise SystemExit(f"{e['eid']}: роль/политика/успех не согласованы")
        eps.append(e)
    for i, e in enumerate(eps):
        e["part"] = i % N_PARTS
    here = Path(__file__).resolve().parent
    cnt = Counter((e["role"], e["split"]) for e in eps)
    return dict(protocol=PROTOCOL, archive=str(a.archive),
                archive_plan=plan["sha256"],
                code={n: kcf.sha256(here / n) for n in CODE},
                episodes=eps, counts={f"{r}_{s}": n for (r, s), n in
                                      sorted(cnt.items())},
                caps={SCOPE: (len(eps) // N_PARTS + 1) * RESERVE}), k36


# --- сбор ---------------------------------------------------------------------------------
class VisTap:
    """Перехват get_image_features на время прохода T32."""

    def __init__(self, bar):
        self.vlm, self.got = bar.vlm, None
        self.orig = bar.vlm.get_image_features

    def __enter__(self):
        def tap(*x, **kw):
            out = self.orig(*x, **kw)
            self.got = out.detach().float().cpu().numpy()
            return out
        self.vlm.get_image_features = tap
        return self

    def __exit__(self, *exc):
        del self.vlm.get_image_features             # вернуть метод класса
        if self.vlm.get_image_features != self.orig:
            raise SystemExit("get_image_features не восстановлен")


def vis_only(arm, bar, io, desc, task, obs):
    """-> визуальные токены; их число обязано совпасть с числом позиций
    изображения в обработанном промпте."""
    torch = arm.torch
    batch, _po, _s8 = io.prep(obs, desc, task)
    with torch.no_grad(), arm.autocast:
        img = bar.vlm.get_image_features(batch["pixel_values"],
                                         batch.get("pixel_attention_mask"))
    img = img.detach().float().cpu().numpy()
    tid = getattr(bar, "image_token_id", None)
    if tid is None:
        tid = bar.config.vlm_config.image_token_id
    n_pos = int((batch["input_ids"] == tid).sum())
    n_tok = int(np.prod(img.shape[:-1]))
    if n_tok != n_pos:
        raise SystemExit(f"визуальных токенов {n_tok} ≠ позиций изображения "
                         f"в промпте {n_pos}")
    return img


def point_map(ts):
    """шаг -> [(начало окна, точка)]; точка 8 окна может совпасть с точкой 0
    следующего."""
    out = {}
    for t in ts:
        for p in POINTS:
            out.setdefault(t + p, []).append((t, p))
    return out


def point_record(io, obs, desc, task):
    import k21_model as km
    _b, _po, s8 = io.prep(obs, desc, task)
    st_ = np.asarray(obs["state"], np.float64)
    if st_.shape != (1, 9) or not np.isfinite(st_).all():
        raise SystemExit(f"state {st_.shape} или не конечен")
    s8 = np.asarray(s8, np.float64)
    return dict(state=st_[0].copy(), state8=s8[0].copy(),
                state11=km.state11(s8)[0].copy(),
                **{c: np.array(obs[c][0], copy=True) for c in CAMS})


def replay_windows(sim, io, arm, bar, e, rec, z, q):
    import k29_policy as kp
    from k36_harness import episode_number
    from k39_cache import Replay
    import k40a_transfer as ka
    w = e["want"]
    task, init, eid = w["task"], w["reset_id"], w["eid"]
    rp = Replay(sim, io, task, init)
    s = rec["summary"]
    if (rp.ih[0][0], rp.ih[1][0]) != (s["init_hash"], s["init_hash_full"]):
        raise SystemExit(f"{eid}: начальные хеши ≠ архиву")
    A = np.asarray(z["actions"], np.float32)
    J = {k[2:]: z[k] for k in z if k.startswith("j_")}
    tgt_all = z["k_a2"] if e["policy"] != "T1" else J["a_codec"]
    starts = dict(zip(e["t"], e["windows"]))
    need = point_map(e["t"])
    W = {t: dict(pts={}) for t in e["t"]}
    desc = io.task_desc(task)
    for step in range(len(A) + 1):
        for t_, p_ in need.get(step, []):
            W[t_]["pts"][p_] = point_record(io, rp.obs, desc, task)
            if p_ == 0:
                i = starts[t_]
                with VisTap(bar) as tap:
                    B, rec0, F = ka.decide_rec(arm, io, desc, task, init,
                                               episode_number(eid), i,
                                               rp.obs, t_, keep_F=True)
                if not np.array_equal(B, J["a_codec"][i]):
                    raise SystemExit(f"{eid} t{t_}: B ≠ j_a_codec")
                tgt = np.asarray(tgt_all[i], np.float32)
                if not np.array_equal(kp.scale_exec(tgt[None], q)[0]
                                      .astype(np.float32), A[t_:t_ + WIN]):
                    raise SystemExit(f"{eid} t{t_}: масштаб цели ≠ "
                                     f"исполненному")
                W[t_].update(B=B, target=tgt, S0=np.asarray(rec0["S"],
                                                            np.float32),
                             F=F, vis0=group_vis(tap.got))
            else:
                W[t_]["pts"][p_]["vis"] = group_vis(
                    vis_only(arm, bar, io, desc, task, rp.obs))
        if step < len(A):
            rp.step(A[step])
    if (rp.reward >= 1, rp.succ_step, rp.done_step, rp.steps) != (
            s["success"], s["success_step"], s["done_step"], s["steps"]):
        raise SystemExit(f"{eid}: повтор ≠ архивному исходу или terminal")
    out = dict(t=np.array(e["t"], np.int64),
               call=np.array(e["windows"], np.int64))
    ts = e["t"]
    out["B"] = np.stack([W[t]["B"] for t in ts])
    out["target"] = np.stack([W[t]["target"] for t in ts])
    out["S0"] = np.stack([W[t]["S0"] for t in ts])
    for k in F_KEYS:
        out[f"F_{k}"] = np.stack([np.asarray(W[t]["F"][k], np.float32)
                                  for t in ts])
    for t in ts:
        W[t]["pts"][0]["vis"] = W[t]["vis0"]
    for k in ("state", "state8", "state11", "vis", *CAMS):
        out[f"p_{k}"] = np.stack([np.stack([W[t]["pts"][p][k]
                                            for p in POINTS]) for t in ts])
    out["actions"] = np.stack([A[t:t + WIN] for t in ts])
    return out


DATA = dict(t=(N_WIN,), call=(N_WIN,), B=(N_WIN, 8, 7),
            target=(N_WIN, 8, 7), S0=(N_WIN, S_DIM),
            actions=(N_WIN, WIN, 7), p_state=(N_WIN, 5, 9),
            p_state8=(N_WIN, 5, 8), p_state11=(N_WIN, 5, 11),
            **{f"F_{k}": (N_WIN, *v) for k, v in F_SHAPES.items()})


def data_problems(z, e, q, vis_dim=None):
    import k29_policy as kp
    p = []
    miss = [k for k in (*DATA, "p_vis", *(f"p_{c}" for c in CAMS))
            if k not in z]
    if miss:
        return [f"нет ключей {miss[:4]}"]
    for k, shp in DATA.items():
        if z[k].shape != shp:
            p.append(f"{k}: {z[k].shape} ≠ {shp}")
        elif z[k].dtype.kind == "f" and not np.isfinite(z[k]).all():
            p.append(f"{k}: не конечен")
    v = z["p_vis"]
    if v.ndim != 4 or v.shape[:3] != (N_WIN, 5, N_GROUPS) or \
            v.shape[3] < 1 or (vis_dim is not None and v.shape[3] != vis_dim) \
            or not np.isfinite(v).all():
        p.append(f"p_vis: {v.shape} (ожидалось D = {vis_dim}) или не конечен")
    for c in CAMS:
        im = z[f"p_{c}"]
        if im.dtype != np.uint8 or im.shape != (N_WIN, 5, *IMG):
            p.append(f"p_{c}: {im.dtype} {im.shape} ≠ uint8 "
                     f"{(N_WIN, 5, *IMG)}")
    if p:
        return p
    if list(z["t"]) != list(e["t"]) or list(z["call"]) != list(e["windows"]):
        p.append("окна ≠ плану")
    for i in range(N_WIN):
        ex = kp.scale_exec(z["target"][i][None], q)[0].astype(np.float32)
        if not np.array_equal(ex, z["actions"][i]):
            p.append(f"окно {i}: масштаб цели ≠ действиям")
            break
    if e["policy"] == "T1" and not np.array_equal(z["B"], z["target"]):
        p.append("T1: цель ≠ B")
    return p


def part_dir(out, k):
    return out / f"part-{int(k)}"


def ep_files(out, e):
    d = part_dir(out, e["part"])
    name = f"{e['role']}_{e['eid'].replace('/', '__')}"
    return d / f"{name}.json", d / f"{name}.npz"


def checked(out, e, plan, q):
    import k36_protocol as pc
    import k37_confirm as kcf
    jp, dp = ep_files(out, e)
    row = pc.load_json(str(jp))
    ident = dict(eid=e["eid"], plan=plan["sha256"], role=e["role"],
                 policy=e["policy"], npz_sha1=e["npz_sha1"], phys=e["phys"])
    if any(row.get(k) != v for k, v in ident.items()):
        raise SystemExit(f"{jp}: запись не соответствует плану")
    if not dp.is_file() or kcf.sha256(dp) != row.get("data_sha256"):
        raise SystemExit(f"{dp}: данные отсутствуют или изменены")
    vd = row.get("vis_dim")
    if type(vd) is not int or vd < 1:
        raise SystemExit(f"{jp}: vis_dim {vd!r}")
    with np.load(dp, allow_pickle=False) as z:
        pr = data_problems(dict(z), e, q, vd)
    if pr:
        raise SystemExit(f"{dp}: {pr[:3]}")
    return row


def restore(out, plan, part, ledger, q):
    eps = {e["eid"]: e for e in plan["episodes"] if e["part"] == part}
    done, pending, pid = ledger.d["done"], ledger.d["pending"], \
        ledger.pending_id()
    if not set(done).issubset(eps):
        raise SystemExit("ledger: учтённые эпизоды вне части")
    if pending is not None and (pid not in eps or pid in done or pending !=
                                dict(id=pid, scope=SCOPE, amount=RESERVE)):
        raise SystemExit("ledger: открытый резерв не из части")
    for c in ledger.d["crashes"]:
        if c.get("id") not in eps or c.get("scope") != SCOPE or \
                c.get("amount") != RESERVE:
            raise SystemExit("ledger: авария не из части")
    rows = {}
    for eid, e in eps.items():
        jp, _dp = ep_files(out, e)
        if jp.exists():
            row = checked(out, e, plan, q)
            if eid != pid and done.get(eid) != dict(scope=SCOPE,
                                                    actual=e["phys"]):
                raise SystemExit(f"{jp}: журнал ≠ записи")
            rows[eid] = row
        elif eid in done:
            raise SystemExit(f"{jp}: учтён, записи нет")
    spent = sum(eps[k]["phys"] for k in done) + \
        RESERVE * len(ledger.d["crashes"])
    if ledger.d["spent"] != {SCOPE: spent}:
        raise SystemExit(f"ledger: расход {ledger.d['spent']} ≠ {spent}")
    dims = {r["vis_dim"] for r in rows.values()}
    if len(dims) > 1:
        raise SystemExit(f"разная размерность визуальных признаков {dims}")
    if pid in rows:
        ledger.settle(pid, eps[pid]["phys"])
    return rows


def collect(a, plan, k36):
    import torch
    import k31_collect as kcol
    import k36_bank as kb36
    import k36_harness as kh
    import k36_protocol as pc
    import k37_bank as kbk
    import k37_confirm as kcf
    from k38_grip_run import threads_problems
    C = kh.consts(k36)
    q = C["q"]
    pd_ = part_dir(a.out, a.part)
    pd_.mkdir(parents=True, exist_ok=True)
    plock = kcf.Lock(str(pd_ / ".part.lock"), f"part {a.part}", kcf.Busy)
    try:
        ledger = kcol.Ledger(str(pd_ / "ledger.json"), plan["caps"])
        rows = restore(a.out, plan, a.part, ledger, q)
        mine = [e for e in plan["episodes"] if e["part"] == a.part]
        todo = [e for e in mine if e["eid"] not in rows]
        if not todo:
            print(f"  part {a.part}: готова ({len(rows)})", flush=True)
            return
        if pr := threads_problems(torch):
            raise SystemExit(f"потоки: {pr}")
        idx = kcf.dev_index(a.device)
        gpu = kcf.gpu_query(idx)
        lock = kcf.Lock(str(Path(kcf.lock_dir()) / (gpu["uuid"] + ".lock")),
                        gpu["uuid"], kcf.Busy)
        sim = None
        val = kcf.validator(C)
        try:
            kcf.check_gpu(idx, kcf.RES["min_free_gb"])
            binding = kcf.bind_device(torch, idx, gpu)
            adm = dict(host=socket.gethostname(), uuid=gpu["uuid"],
                       binding=binding, threads=torch.get_num_threads())
            sim = kb36.BankSim("bank", kbk.SPACE, bank_dir=kbk.BANK_DIR)
            for seed in (0, 1):
                src = arm = None
                for e in todo:
                    if e["seed"] != seed:
                        continue
                    if src is None:
                        kcf.check_gpu(idx, kcf.RES["min_free_gb"])
                        src = kh.load_source(torch, seed, a.device)
                        arm = kh.make_arm(torch, k36, f"T32_s{seed}", src)
                    kcf.install(kcf.part_dir(str(a.archive),
                                             e["arch_part"]))
                    rec, z = kh.load_ep(e["want"])
                    if rec["npz_sha1"] != e["npz_sha1"] or \
                            val(rec, z, e["want"]):
                        raise SystemExit(f"{e['eid']}: архив изменился")
                    kcf.check_disk(a.out)
                    if rows:
                        vd0 = next(iter(rows.values()))["vis_dim"]
                    else:
                        vd0 = None
                    ledger.reserve(SCOPE, e["eid"], amount=RESERVE)
                    data = replay_windows(sim, src["io"], arm,
                                          src["mod"].bar, e, rec, z, q)
                    pr = data_problems(data, e, q, vd0)
                    if pr:
                        raise SystemExit(f"{e['eid']}: {pr[:3]}")
                    jp, dp = ep_files(a.out, e)
                    tmp = dp.with_suffix(".tmp.npz")
                    with open(tmp, "wb") as fh:
                        np.savez_compressed(fh, **data)
                    tmp.replace(dp)
                    row = dict(eid=e["eid"], plan=plan["sha256"],
                               role=e["role"], policy=e["policy"],
                               npz_sha1=e["npz_sha1"], phys=e["phys"],
                               data_sha256=kcf.sha256(dp),
                               vis_dim=int(data["p_vis"].shape[-1]), gpu=adm)
                    pc.save_json(str(jp), row)
                    checked(a.out, e, plan, q)
                    ledger.commit(e["eid"], e["phys"])
                    rows[e["eid"]] = row
                    print(f"  {e['role']} {e['eid']}: окон {N_WIN}, vis "
                          f"{row['vis_dim']}; {len(rows)}/{len(mine)}",
                          flush=True)
                if src is not None:
                    src = arm = None
                    kh.free(torch)
        finally:
            if sim is not None:
                kcf.close_sim(sim)
            lock.release()
        print(f"  part {a.part}: готова", flush=True)
    finally:
        plock.release()


def status(a, plan):
    import json
    dims = set()
    for k in range(N_PARTS):
        for jp in part_dir(a.out, k).glob("*.json"):
            if jp.name != "ledger.json":
                dims.add(json.load(open(jp)).get("vis_dim"))
        n = sum(e["part"] == k for e in plan["episodes"])
        lp = part_dir(a.out, k) / "ledger.json"
        d = json.load(open(lp)) if lp.exists() else dict(done={}, crashes=[])
        print(f"  part {k}: {len(d['done'])}/{n}; аварий "
              f"{len(d['crashes'])}")
    print(f"  размерность визуальных признаков: {sorted(dims, key=str)}" +
          ("" if len(dims) <= 1 else "  — РАЗНАЯ, данные не годятся"))


def selftest():
    # окна
    J = dict(elapsed=np.arange(20) * 8, k=np.array([8] * 19 + [3]))
    ws = window_starts(J)
    assert ws == [(k * 18) // 7 for k in range(8)] and ws[0] == 0 and \
        ws[-1] == 18
    J8 = dict(elapsed=np.arange(8) * 8, k=np.full(8, 8))
    assert window_starts(J8) == list(range(8))
    try:
        window_starts(dict(elapsed=np.arange(7) * 8, k=np.full(7, 8)))
        raise AssertionError
    except SystemExit:
        pass
    # группы визуальных токенов: порядок подачи, 4 последовательные группы
    img = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
    gv = group_vis(img)
    flat = img.reshape(-1, 3)
    assert gv.shape == (4, 3) and np.allclose(gv[0], flat[:3].mean(0)) and \
        np.allclose(gv[3], flat[9:].mean(0))
    # отбор: роли, равенство групп
    rows = []
    for t in (0, 6, 8):
        for c in range(20):
            for s in (0, 1):
                fail = (c + s + t) % 4 == 0
                for pol in ("T1", "T2", "T2P"):
                    ok = not fail if pol == "T1" else (
                        (pol == "T2P" and c % 2 == 0) or
                        (pol == "T2" and c % 3 == 0))
                    rows.append(dict(eid=f"{t}/{c}/{s}/{pol}", task=t,
                                     case_id=f"t{t}_c{c}", seed=s,
                                     policy=pol, success=ok, part=0,
                                     phys=300))
    sel = select(rows)
    cnt = Counter(e["role"] for e in sel)
    assert cnt["R"] == cnt["T1a"] == cnt["T1b"] > 0
    assert all((e["role"] == "R") == (e["policy"] != "T1") for e in sel)
    assert len({e["eid"] for e in sel}) == len(sel)
    import k40b_audit as ka
    assert cnt["R"] == ka.budget(rows)["totals"]["m"]
    # данные: валидные проходят, порча отвергается
    q = np.ones(7)
    rng = np.random.default_rng(0)
    tgt = rng.uniform(-.5, .5, (N_WIN, 8, 7)).astype(np.float32)
    import k29_policy as kp
    acts = np.stack([kp.scale_exec(tgt[i][None], q)[0] for i in range(N_WIN)]
                    ).astype(np.float32)
    data = {k: np.ones(shp, np.float32) for k, shp in DATA.items()}
    data.update(t=np.arange(N_WIN, dtype=np.int64) * 8,
                call=np.arange(N_WIN, dtype=np.int64), B=tgt.copy(),
                target=tgt, actions=acts,
                p_vis=np.ones((N_WIN, 5, 4, 6), np.float32),
                **{f"p_{c}": np.zeros((N_WIN, 5, *IMG), np.uint8)
                   for c in CAMS})
    e = dict(t=list(range(0, 64, 8)), windows=list(range(8)), policy="T1")
    assert data_problems(data, e, q) == []
    for bad in (dict(data, actions=acts + 1),
                dict(data, B=tgt + 0.1),
                dict(data, S0=np.ones((N_WIN, 10), np.float32)),
                dict(data, p_vis=np.full((N_WIN, 5, 4, 6), np.nan,
                                         np.float32)),
                {k: v for k, v in data.items() if k != "p_" + CAMS[1]},
                dict(data, p_vis=np.ones((N_WIN, 5, 4, 0), np.float32)),
                dict(data, **{f"p_{CAMS[0]}": np.zeros((N_WIN, 5, 224, 224,
                                                        1), np.uint8)})):
        assert data_problems(bad, e, q), "порча принята"
    assert data_problems(data, e, q, vis_dim=7), "чужая D принята"
    assert data_problems(data, e, q, vis_dim=6) == []
    assert data_problems(dict(data, B=tgt + 0.1), dict(e, policy="T2P"),
                         q) == []
    pm = point_map([0, 8, 24])
    assert pm[8] == [(0, 8), (8, 0)] and pm[24] == [(24, 0)] and \
        sum(len(v) for v in pm.values()) == 15
    # журнал: общий расход и чужой резерв
    import tempfile
    import k31_collect as kcol
    plan_ = dict(sha256="P", episodes=[dict(eid="e1", part=0, phys=84, role="R"),
                                       dict(eid="e2", part=0, phys=90, role="T1a")])
    with tempfile.TemporaryDirectory() as td:
        L = kcol.Ledger(str(Path(td) / "l.json"), {SCOPE: 2 * RESERVE})
        assert restore(Path(td), plan_, 0, L, q) == {}
        L.d["spent"] = {SCOPE: 74}                 # расход без учтённых
        try:
            restore(Path(td), plan_, 0, L, q)
            raise AssertionError("расход без учтённых принят")
        except SystemExit:
            pass
        L.d["spent"] = {SCOPE: 0}
        L.d["pending"] = dict(id="e1", scope=SCOPE, amount=5)
        try:
            restore(Path(td), plan_, 0, L, q)
            raise AssertionError("чужой резерв принят")
        except SystemExit:
            pass
        L.d["pending"] = dict(id="x", scope=SCOPE, amount=RESERVE)
        try:
            restore(Path(td), plan_, 0, L, q)
            raise AssertionError("резерв вне части принят")
        except SystemExit:
            pass
    print("самопроверка k40b_collect пройдена: окна ⌊k(M−1)/7⌋ по полным "
          "чанкам; 4 последовательные группы токенов; роли R/T1a/T1b = "
          "аудиту; цель T1 = B, у T2/T2P — k_a2; масштаб цели = действиям; "
          "формы кадров и признаков, единая D, журнал (расход, резерв) — "
          "порча отвергается")
    return 0


def main():
    import sys
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", default="reports/k37c/runs/c2_cds2")
    ap.add_argument("--out", type=Path, default=Path("data/k40b/collect/v1"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--part", type=int)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    import k36_protocol as pc
    import k37_confirm as kcf
    pp = a.out / "plan.json"
    if a.prepare_only:
        plan, _k36 = prepare(a)
        plan["sha256"] = kcf.canon_sha(plan)
        a.out.mkdir(parents=True, exist_ok=True)
        if pp.exists() and pc.load_json(str(pp)) != plan:
            raise SystemExit("план изменился, укажите новый --out")
        if not pp.exists():
            pc.save_json(str(pp), plan)
        print(f"  K-40 B сбор: эпизодов {len(plan['episodes'])} "
              f"{plan['counts']}; SHA {plan['sha256'][:16]}", flush=True)
        return 0
    if not pp.exists():
        raise SystemExit("плана нет — сначала --prepare-only")
    plan = pc.load_json(str(pp))
    if kcf.canon_sha({k: v for k, v in plan.items() if k != "sha256"}) != \
            plan["sha256"]:
        raise SystemExit("plan.json ≠ своему SHA")
    here = Path(__file__).resolve().parent
    bad = [n for n, s in plan["code"].items() if kcf.sha256(here / n) != s]
    if bad:
        raise SystemExit(f"код ≠ плану: {bad}")
    c2, k36 = kcf.load_plan(plan["archive"])
    if c2["sha256"] != plan["archive_plan"]:
        raise SystemExit("архив ≠ плану")
    if a.status:
        status(a, plan)
        return 0
    if a.part not in range(N_PARTS):
        raise SystemExit("нужен --part 0|1")
    a.archive = plan["archive"]
    collect(a, plan, k36)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
