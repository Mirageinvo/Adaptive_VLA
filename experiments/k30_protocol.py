#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-30: четыре плана SR (H/C × seed 0/1), проверка блоков с независимой
пересборкой журнала.

plan --seed S --mode hard|soft — reports/k30/plan_s<S>_<mode>.json: метка,
  спецификация (best.pt, SHA, полный отпечаток весов, источник K-25,
  словарь, код контракта), setup/описания/pos_offset из плана A K-21,
  LIBERO-10, задачи 0–9 × блоки 30/35/40, smoke 0/8 блока 30, env seed 101;
  закреплённые источники: план A K-21, fullbar_cached, K-22 (s0) или повтор
  K-23 (s1); описательный K-25 того же сида — если полный набор проверен.
check-block — validate_block (холодный K-9h: начальные хеши, хеши
  действий, сид, одиночный процесс, setup) + журнал K-30 (CPU, float64):
  1) формы, dtype ID, конечность — до сравнений;
  2) ids = argmax записанных logits; hard: b_used = P[ids] точно; soft:
     |softmax(logits)@P − b_used| ≤ 2e−6; b_soft — так же;
  3) C0, z1/z2, Z из журнала: ошибка ≤ 1e−5·max(1, max|ref|); рука
     (a − mid)/half ≤ 1e−5 абсолютно; пределы; схват = знак Σ W2·y2[6];
  4) исполненные действия = журналу побитово (k15h_protocol);
  5) в smoke — CPU-повтор root_ln/classifier/plan_proj/Expand1 из весов
     best.pt по записанному h8: допуск 1e−4·max(1,|ref|); Expand1 — с явным
     воспроизведением fp16 autocast (вход, веса и выход Linear в fp16,
     накопление fp32, затем (h + side) + Linear во float32); сверх допуска
     принимается только ровно один fp16-шаг выхода Linear (флип округления
     из-за порядка накопления GEMM); равенство ID — при разрыве top1 − top2
     > 1e−3, иначе сверяются logits.
Везде `not err <= tol` — нечисловая ошибка тоже отказ.
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k21_protocol as pp  # noqa: E402
import k23_protocol as p23  # noqa: E402
import k24_protocol as p24  # noqa: E402
import k25_protocol  # noqa: E402,F401  (регистрирует k24_s*)
import k26_protocol as p26  # noqa: E402  (k25_s*)
import k30_model as k30m  # noqa: E402

KIND = "k30_plan"
BOOT = dict(n=10000, seed=20261130)
TOL_PLAN, TOL_RAW, TOL_ARM, TOL_DBG, GAP = 2e-6, 1e-5, 1e-5, 1e-4, 1e-3
CODE = ("k30_protocol.py", "k30_policy.py", "k30_harness.py",
        "k30_readout.py", "k30_model.py", "k30_data.py", "k30_codec.py",
        "k21_main_policy.py", "k21_policy.py", "k16_harness.py",
        "k16_policy.py", "k9h_multiarm_gate.py", "k15g_local_probe.py",
        "k15h_protocol.py", "k21_protocol.py")
sha, full_sha = pp.sha, pp.full_sha


def plan_path(seed, mode):
    return f"reports/k30/plan_s{seed}_{mode}.json"


def run_dir(plan_path_):
    return os.path.join("reports/k30/run", f"p{sha(plan_path_)}")


def block_name(lab, suite, t, b):
    return f"{lab}_{suite}_t{t}_i{b}.json"


# --- план --------------------------------------------------------------------------
def make_plan(seed, mode, device):
    import torch
    import k21_data as d21
    import k21_train as kt
    import k30_data as kd
    import k30_train as k30t
    p = k30t.load_prepare()
    trp = k30t.report_path(seed, mode)
    tr = json.load(open(trp, encoding="utf-8"))
    ck = os.path.join(k30t.outdir(seed, mode), "best.pt")
    if full_sha(ck) != tr["best_sha1"]:
        raise SystemExit(f"{ck} ≠ отчёту обучения")
    c = torch.load(ck, map_location="cpu", weights_only=False)
    if c["weights_hash"] != tr["best_weights_hash"] or \
            kt.state_hash(c["weights"]) != c["weights_hash"] or \
            c.get("mode") != mode or int(c.get("seed", -1)) != seed or \
            tr.get("prepare") != k30t.fp(p):
        raise SystemExit("best.pt: отпечаток, режим, seed или prepare")
    u = int(c["update"])
    if not (k30t.EVAL_EVERY <= u <= k30t.MAIN and u % k30t.EVAL_EVERY == 0):
        raise SystemExit(f"best не из main 250…4000: {u}")
    C = kd.load()
    lab = k30m.label_of(mode, seed)
    spec = dict(mode=mode, seed=seed, ckpt=os.path.abspath(ck),
                ckpt_sha1=full_sha(ck), weights_hash=c["weights_hash"],
                best_update=u, crit=float(c["crit"]),
                source=p["sources"][str(seed)], codec_sha=C["sha"],
                code=p["code"], train_report=os.path.abspath(trp),
                train_report_sha1=full_sha(trp))
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    rA = p23.ref_k21_main()
    A = rA["plan"]
    refs = {"k21_main_A": p23.pin_ref(rA),
            "fullbar_cached": p23.pin_ref(p23.ref_cached())}
    if seed == 0:
        refs["k22"] = p23.pin_ref(p23.ref_k22())
    else:
        refs[f"k23_repeat_s{seed}"] = p23.pin_ref(p24.ref_k23_repeat(seed))
    opt = {}
    try:
        r25 = p26.ref_k25(seed)
        ok25 = all(full_sha(s_["ckpt"]) == s_["ckpt_sha1"] and
                   s_["ckpt_sha1"] == p["sources"][str(seed)]["ckpt_sha1"]
                   for s_ in r25["plan"]["specs"].values())
        opt[f"k25_s{seed}"] = dict(status="pinned", ref=p23.pin_ref(r25)) \
            if ok25 else dict(status="не выполнено",
                              why="чекпойнт K-25 плана ≠ источнику K-30")
    except (SystemExit, OSError, KeyError) as ex:
        opt[f"k25_s{seed}"] = dict(status="не выполнено", why=str(
            getattr(ex, "code", ex)))
    desc = {}
    for (l_, s_, t_, b_), f in rA["blocks"].items():
        if l_ == "fullbar":
            desc.setdefault(t_, json.load(open(f, encoding="utf-8")).get(
                "task_description"))
    cells = [["10", t, b] for t in A["tasks"] for b in A["blocks"]]
    if any(int(x[2]) + 4 >= 45 for x in cells):
        raise SystemExit("блоки задевают резерв 45–49")
    inp = dict(prepare=d21.REPORT, k30_prepare=k30t.PREPARE, train_report=trp,
               codec_report=kd.REPORT)
    return dict(
        kind=KIND, name=f"k30_s{seed}_{mode}", train_seed=seed, plan_mode=mode,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        labels=[lab], policy={lab: "k15f"}, specs={lab: spec},
        pair=k30m.label_of("soft" if mode == "hard" else "hard", seed),
        suites=["10"], tasks=list(A["tasks"]), blocks=list(A["blocks"]),
        cells=cells, smoke=[["10", t, b] for t, b in A["smoke"]],
        setups={"10": A["setup"]},
        suite_tasks={"10": [desc[t] for t in sorted(desc)]},
        refs=A["refs"], refs_sha1=A["refs_sha1"], device=str(device),
        snapshot_path=snap, prep_id=prep["prep_id"],
        bar_revision=prep["bar_pin"]["revision"], bounds=prep["bounds_codec"],
        n_envs=5, env_seed=101, bootstrap=dict(BOOT), references=refs,
        optional_references=opt,
        inputs={k: os.path.abspath(v) for k, v in inp.items()},
        inputs_sha1={k: sha(v) for k, v in inp.items()},
        code={f: sha(os.path.join(HERE, f)) for f in CODE},
        tolerances=dict(plan=TOL_PLAN, raw=TOL_RAW, arm=TOL_ARM, dbg=TOL_DBG,
                        gap=GAP),
        note="разведочный парный пилот на многократно использованном банке "
             "30–44; основное — H − C того же сида")


def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план K-30"]
    for grp in ("inputs", "refs"):
        for k, f in (plan.get(grp) or {}).items():
            if not os.path.exists(f) or sha(f) != plan[f"{grp}_sha1"][k]:
                p.append(f"{grp} {k} изменён или отсутствует ({f})")
    for lab, s in plan["specs"].items():
        if not os.path.exists(s["ckpt"]) or full_sha(s["ckpt"]) != \
                s["ckpt_sha1"]:
            p.append(f"чекпойнт {lab} изменён или отсутствует")
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    return p


# --- журнал K-30 -------------------------------------------------------------------
SHAPES = dict(logits=(4, 128), ids=(4,), b_soft=(4, 6), b_used=(4, 6),
              C0=(4, 6), g1=(8,), g2=(16,), delta1=(8, 20, 6),
              y2=(16, 20, 7), Z=(20, 6), grip=(20,))


def _rel(a, b):
    return float(np.abs(a - b).max() / max(1.0, float(np.abs(b).max())))


def geometry_np():
    import torch
    import k30_readout as kr
    G = kr.build(torch)["geometry"]()
    return {k: v.double().numpy() for k, v in G.items()}


def journal_problems(z, label, mode, bounds, P, spec=None, smoke=False):
    import k30_readout as kr
    need = [f"k21_k30_{k}" for k in SHAPES] + [f"k15d_level_{label}"]
    miss = [k for k in need if k not in z]
    if miss:
        return [f"{label}: в журнале нет {miss}"]
    g = {k: np.asarray(z[f"k21_k30_{k}"]) for k in SHAPES}
    lv0 = np.asarray(z[f"k15d_level_{label}"], np.float64)
    C, B = g["Z"].shape[:2] if g["Z"].ndim == 4 else (None, None)
    if C is None:
        return [f"{label}: Z формы {g['Z'].shape}"]
    p = [f"{label}: {k} формы {g[k].shape}" for k, t in SHAPES.items()
         if g[k].shape != (C, B) + t]
    if lv0.shape != (C, B, 8, 7):
        p.append(f"{label}: журнал команд формы {lv0.shape}")
    if g["ids"].dtype.kind not in "iu":
        p.append(f"{label}: ids dtype {g['ids'].dtype}")
    if p:
        return p
    bad = [k for k, v in g.items() if k != "ids" and
           not np.isfinite(v.astype(np.float64)).all()]
    if not np.isfinite(lv0).all():
        bad.append("level")
    if bad:
        return [f"{label}: нечисловые значения в {bad}"]
    if g["ids"].min() < 0 or g["ids"].max() >= k30m.K:
        return [f"{label}: ID вне 0…127"]
    f = {k: v.reshape((C * B,) + v.shape[2:]).astype(np.float64)
         if k != "ids" else v.reshape((C * B,) + v.shape[2:])
         for k, v in g.items()}
    lg = f["logits"]
    if not np.array_equal(np.argmax(lg, -1), f["ids"]):
        p.append(f"{label}: ids ≠ argmax(logits)")
    Pf = np.asarray(P, np.float32)
    _i, soft = kr.np_plan(lg, Pf, "soft")
    d_soft = float(np.abs(soft - f["b_soft"]).max())
    if not d_soft <= TOL_PLAN:
        p.append(f"{label}: b_soft ≠ softmax@P ({d_soft:.2e})")
    if mode == "hard":
        if not np.array_equal(g["b_used"].reshape(C * B, 4, 6),
                              Pf[f["ids"]]):
            p.append(f"{label}: hard b_used ≠ P[ids]")
    else:
        d = float(np.abs(soft - f["b_used"]).max())
        if not d <= TOL_PLAN:
            p.append(f"{label}: soft b_used ≠ softmax@P ({d:.2e})")
    G = geometry_np()
    ff = dict(b_used=f["b_used"], g1=f["g1"], g2=f["g2"], delta1=f["delta1"],
              y2=f["y2"], W0=np.broadcast_to(G["W0"], (C * B, 20, 4)),
              W1=np.broadcast_to(G["W1"], (C * B, 20, 8)),
              W2=np.broadcast_to(G["W2"], (C * B, 20, 16)))
    rec = kr.np_readout(ff, bounds["L"], bounds["U"])
    for nm, a, b in (("C0", rec["C0"], f["C0"]), ("Z", rec["Z"], f["Z"]),
                     ("G0+G1+G2", rec["G0"] + rec["G1"] + rec["G2"], f["Z"]),
                     ("grip", rec["grip"], f["grip"])):
        d = _rel(a, b)
        if not d <= TOL_RAW:
            p.append(f"{label}: {nm} пересборки ≠ журналу ({d:.2e})")
    lv = lv0.reshape(C * B, 8, 7)
    half, mid = rec["half"], rec["mid"]
    dn = float(np.abs((rec["arm"][:, :8] - mid) / half -
                      (lv[..., :6] - mid) / half).max())
    if not dn <= TOL_ARM:
        p.append(f"{label}: рука (a−mid)/half ≠ исполненным ({dn:.2e})")
    L, U = np.asarray(bounds["L"])[:6], np.asarray(bounds["U"])[:6]
    if (lv[..., :6] < L - 1e-4).any() or (lv[..., :6] > U + 1e-4).any():
        p.append(f"{label}: рука вне пределов")
    if not np.array_equal(np.where(rec["grip"][:, :8] > 0, 1.0, -1.0),
                          lv[..., 6]):
        p.append(f"{label}: схват ≠ знаку Σ W2·y2[6]")
    if smoke and not p:
        p += debug_problems(z, label, spec, g, G)
    return p


def debug_problems(z, label, spec, g, G):
    """CPU-повтор root_ln/classifier/plan_proj/Expand1 из весов best.pt."""
    import torch
    keys = [f"k21_k30_dbg_{k}" for k in ("h8", "hrp", "mid_init")]
    if any(k not in z for k in keys):
        return [f"{label}: в smoke нет debug-фрагмента"]
    h8, hrp, mid = (np.asarray(z[k], np.float64)[0] for k in keys)
    ck = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    W = {k: v.double().numpy() for k, v in ck["weights"].items()
         if k.startswith(("root_ln.", "root_logits.", "plan_proj.", "exp1."))}
    mu = h8.mean(-1, keepdims=True)
    var = ((h8 - mu) ** 2).mean(-1, keepdims=True)
    v0 = (h8 - mu) / np.sqrt(var + 1e-5) * W["root_ln.weight"] + \
        W["root_ln.bias"]
    lg = v0 @ W["root_logits.weight"].T + W["root_logits.bias"]
    lg_rec = np.asarray(g["logits"][0], np.float64)
    p = []
    d = _rel(lg, lg_rec)
    if not d <= TOL_DBG:
        p.append(f"{label}: logits CPU ≠ GPU ({d:.2e})")
    srt = np.sort(lg, -1)
    clear = (srt[..., -1] - srt[..., -2]) > GAP
    if not np.array_equal(np.argmax(lg, -1)[clear], g["ids"][0][clear]):
        p.append(f"{label}: ID CPU ≠ GPU при явном разрыве top1−top2")
    bu = np.asarray(g["b_used"][0], np.float64)
    hrp_c = h8 + bu @ W["plan_proj.weight"].T
    d = _rel(hrp_c, hrp)
    if not d <= TOL_DBG:
        p.append(f"{label}: h8 + plan_proj ≠ GPU ({d:.2e})")
    half = bool(np.asarray(z["k21_k30_dbg_half"])[0].max() > 0) \
        if "k21_k30_dbg_half" in z else False
    mid_c, w16 = expand1_fp16(torch, hrp, W, G, half)
    err = np.abs(mid_c - mid)
    tol = TOL_DBG * np.maximum(1.0, np.abs(mid))
    over = ~(err <= tol)
    # единственное допущение сверх общего допуска: ровно один шаг fp16
    # выхода Linear (флип округления из-за порядка накопления в GEMM)
    flip = over & np.isclose(err, np.abs(np.spacing(w16.astype(np.float16))
                                         .astype(np.float64)), rtol=1e-3,
                             atol=0)
    if (over & ~flip).any():
        p.append(f"{label}: Expand1 CPU ≠ GPU (max {float(err.max()):.2e}, "
                 f"вне допуска {int((over & ~flip).sum())})")
    return p


def expand1_fp16(torch, hrp, W, G, half=False):
    """CPU-повтор Expand1 с приведениями fp16 autocast и порядком сложений
    GPU: Linear(φ) — вход и веса округлены до fp16, накопление fp32, выход
    fp16; затем (h + side) + Linear в float32. hrp [B,4,H] (float32)."""
    import k21_tree as kt
    TT = kt.build(torch)
    iv = torch.as_tensor(G["IV1"], dtype=torch.float32)[None]
    l_, r_ = iv[..., 0] / 20, iv[..., 1] / 20
    phi = TT["phi"]
    fe = torch.cat([phi(l_), phi(r_), phi((l_ + r_) / 2), phi(r_ - l_)],
                   -1)[0].float()
    Wh = torch.as_tensor(W["exp1.W.weight"]).half().float()
    bh = torch.as_tensor(W["exp1.W.bias"]).half().float()
    w16 = (fe.half().float() @ Wh.T + bh).half()                  # [8, H]
    dt = torch.float16 if half else torch.float32   # dtype h на GPU
    hp = torch.as_tensor(hrp, dtype=torch.float32).to(dt).repeat_interleave(
        2, 1)
    side = torch.as_tensor(W["exp1.e_side"], dtype=torch.float32).repeat(
        4, 1).to(dt)
    mid = ((hp + side[None]) + w16.to(dt)[None]).float()
    return mid.double().numpy(), w16.float().numpy()[None].repeat(
        len(hrp), 0)


def check_block(plan, plan_path_, path, label, suite, task, block):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import k16_protocol as p16
    import k30_data as kd
    if label not in plan["labels"]:
        return [f"метка {label!r} не из плана"]
    if [suite, int(task), int(block)] not in [[c[0], int(c[1]), int(c[2])]
                                              for c in plan["cells"]]:
        return [f"ячейка {suite}/{task}/{block} не из плана"]
    p = pr.validate_block(path, label, int(task), int(block),
                          require_single=True, setup=plan["setups"][suite])
    if p:
        return p
    d, z = p16.load(path)
    if os.path.basename(os.path.normpath(str(d.get("ckpt")))) != \
            plan["bar_revision"]:
        p.append(f"{label}: снимок BAR блока не закреплённой ревизии")
    if d.get("task_description") != plan["suite_tasks"][suite][int(task)]:
        p.append(f"{label}: задача блока ≠ задаче плана")
    spec = plan["specs"][label]
    mode, seed = k30m.parse_label(label)
    smoke = [suite, int(task), int(block)] in [
        [c[0], int(c[1]), int(c[2])] for c in plan["smoke"]]
    j = d.get("joint") or {}
    want = dict(label=label, arch=k30m.ARCH, plan_mode=mode, train_seed=seed,
                plan_sha1=sha(plan_path_), prep_id=plan["prep_id"],
                bar_revision=plan["bar_revision"], ckpt_sha1=spec["ckpt_sha1"],
                weights_hash=spec["weights_hash"], codec_sha=spec["codec_sha"],
                k30_policy_sha1=full_sha(os.path.join(HERE, "k30_policy.py")),
                harness_adapter_sha1=sha(os.path.join(HERE, "k30_harness.py")))
    for k, v in want.items():
        if j.get(k) != v:
            p.append(f"{label}: {k} = {j.get(k)!r}, ожидалось {v!r}")
    p += pp.journal_problems(label, z, plan["bounds"])
    if not p:
        C = kd.load()
        p += journal_problems(z, label, mode, plan["bounds"], C["P32"], spec,
                              smoke)
    if not p:
        p += hpc.actions_problems(label, z, hpc.harness_scale())
    return p


def check_run(plan, plan_path_, rdir, cells=None):
    ok, gaps = {}, []
    for s, t, b in cells or plan["cells"]:
        for lab in plan["labels"]:
            pth = os.path.join(rdir, block_name(lab, s, t, b))
            if not os.path.exists(pth):
                gaps.append(dict(label=lab, suite=s, task=t, block=b,
                                 why="не готов"))
                continue
            prob = check_block(plan, plan_path_, pth, lab, s, t, b)
            if prob:
                gaps.append(dict(label=lab, suite=s, task=t, block=b,
                                 why="; ".join(prob[:2])))
            else:
                ok[(lab, s, int(t), int(b))] = pth
    return ok, gaps


def selftest():
    import torch
    import k21_model as km
    import k30_policy as kp
    M = k30m.build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    P = np.random.default_rng(0).uniform(-0.8, 0.8, (128, 6)).astype(
        np.float32)
    bounds = dict(L=L.tolist(), U=U.tolist())
    for mode in ("hard", "soft"):
        mod = M["K30"](km.fake_bar(torch), L, U, P, np.full(6, 0.3), seed=0,
                       plan_mode=mode).eval()
        with torch.no_grad():
            mod.root_logits.weight.normal_(0, 0.5)
            mod.cond.gate.bias.fill_(0.0)
        lab = k30m.label_of(mode, 0)
        rows = []
        for c in range(3):
            with torch.no_grad():
                out = mod(km.fake_batch(torch, B=3, seed=c), 1,
                          torch.randn(3, 11, generator=torch.Generator()
                                      .manual_seed(c)), debug=c == 0)
            rec = kp.record(out, False)
            g = torch.where(out["grip_logit"][:, :8] > 0, 1.0, -1.0)
            rec["level"] = torch.cat([out["arm"][:, :8], g[..., None]],
                                     -1).numpy()
            rows.append(rec)
        z = {f"k21_{k}": np.stack([r[k] for r in rows]) for k in rows[0]
             if k != "level"}
        z[f"k15d_level_{lab}"] = np.stack([r["level"] for r in rows])
        assert journal_problems(z, lab, mode, bounds, P) == [], \
            journal_problems(z, lab, mode, bounds, P)
        for key, mut, word in (
                ("k21_k30_ids", lambda a: (a + 1) % 128, "argmax"),
                ("k21_k30_b_used", lambda a: a + 1e-3, "b_used"),
                ("k21_k30_logits", lambda a: a * 1.01, ""),
                ("k21_k30_C0", lambda a: a + 1e-3, "C0"),
                ("k21_k30_Z", lambda a: a + 1e-3, "Z"),
                ("k21_k30_grip", lambda a: -a, ""),
                (f"k15d_level_{lab}", lambda a: a + 1e-3, "рука"),
                ("k21_k30_delta1", lambda a: a * np.nan, "нечисловые"),
                ("k21_k30_g1", lambda a: a[..., :4], "формы")):
            zb = dict(z)
            zb[key] = mut(np.array(z[key], copy=True))
            pb = journal_problems(zb, lab, mode, bounds, P)
            assert pb and (not word or any(word in x for x in pb)), \
                (mode, key, pb)
        # режим плана: soft-журнал не проходит как hard
        if mode == "soft":
            assert journal_problems(z, lab, "hard", bounds, P)
        # smoke: CPU-повтор по debug и весам
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            import k21_train as kt
            ckp = os.path.join(td, "best.pt")
            torch.save(dict(weights=kt.trainable_state(mod)), ckp)
            with torch.no_grad():
                o = mod(km.fake_batch(torch, B=3, seed=0), 1,
                        torch.randn(3, 11, generator=torch.Generator()
                                    .manual_seed(0)), debug=True)
            dbg = kp.record(o, True)
            zd = dict(z)
            for k in ("h8", "hrp", "mid_init", "half"):
                zd[f"k21_k30_dbg_{k}"] = dbg[f"k30_dbg_{k}"][None]
            for k in SHAPES:
                zd[f"k21_k30_{k}"] = np.concatenate(
                    [dbg[f"k30_{k}"][None], z[f"k21_k30_{k}"][1:]])
            assert journal_problems(zd, lab, mode, bounds, P,
                                    dict(ckpt=ckp), smoke=True) == []
            zb = dict(zd)
            zb["k21_k30_dbg_mid_init"] = zd["k21_k30_dbg_mid_init"] + 0.01
            assert any("Expand1" in x for x in journal_problems(
                zb, lab, mode, bounds, P, dict(ckpt=ckp), smoke=True))
    # Expand1: сокращающиеся слагаемые с большими весами (квантование fp16
    # весов и входа даёт ошибку больше прежнего допуска), «GPU» — честный
    # fp16-расчёт; повтор с явными приведениями проходит, внесённая ошибка
    # 1e−3 обнаруживается
    Hs = 64
    gg = torch.Generator().manual_seed(7)
    W_ = {"exp1.W.weight": (torch.randn(Hs, 36, generator=gg) * 40).double()
          .numpy(), "exp1.W.bias": (torch.randn(Hs, generator=gg) * 40)
          .double().numpy(), "exp1.e_side": (torch.randn(2, Hs, generator=gg)
                                             * 1e-3).double().numpy()}
    G_ = geometry_np()
    import k21_tree as kt_
    TT = kt_.build(torch)
    iv = torch.as_tensor(G_["IV1"], dtype=torch.float32)[None]
    l_, r_ = iv[..., 0] / 20, iv[..., 1] / 20
    fe = torch.cat([TT["phi"](l_), TT["phi"](r_), TT["phi"]((l_ + r_) / 2),
                    TT["phi"](r_ - l_)], -1)[0].float()
    hrp = torch.randn(2, 4, Hs, generator=gg).float()
    # «GPU»: вход и веса fp16, накопление fp32 в ДРУГОМ порядке (обратном,
    # как может cuBLAS), смещение fp16, выход fp16
    fh = fe.half().float()
    Whh = torch.as_tensor(W_["exp1.W.weight"]).half().float()
    acc = torch.zeros(fh.shape[0], Whh.shape[0])
    for k_ in reversed(range(fh.shape[1])):
        acc = acc + fh[:, k_:k_ + 1] * Whh[:, k_][None]
    lin = (acc + torch.as_tensor(W_["exp1.W.bias"]).half().float()).half()
    gpu_mid = (hrp.repeat_interleave(2, 1) + torch.as_tensor(
        W_["exp1.e_side"]).float().repeat(4, 1)[None]) + lin.float()[None]
    f64 = (fe.double() @ torch.as_tensor(W_["exp1.W.weight"]).T +
           torch.as_tensor(W_["exp1.W.bias"]))
    old_err = float((torch.as_tensor(gpu_mid).double() - (
        hrp.double().repeat_interleave(2, 1) + torch.as_tensor(
            W_["exp1.e_side"]).repeat(4, 1)[None] + f64[None])).abs().max())
    assert old_err > 1e-4, old_err          # прежняя пересборка отвергла бы
    mid_c, w16 = expand1_fp16(torch, hrp.numpy(), W_, G_)
    gm = gpu_mid.double().numpy()
    err = np.abs(mid_c - gm)
    tol = TOL_DBG * np.maximum(1.0, np.abs(gm))
    flip = np.isclose(err, np.abs(np.spacing(w16.astype(np.float16))
                                  .astype(np.float64)), rtol=1e-3, atol=0)
    assert ((err <= tol) | flip).all(), float(err.max())
    err2 = np.abs(mid_c - (gm + 1e-3))
    assert not ((err2 <= tol) | np.isclose(err2, np.abs(np.spacing(
        w16.astype(np.float16)).astype(np.float64)), rtol=1e-3,
        atol=0)).all()
    assert len(set(CODE)) == len(CODE)
    print("самопроверка k30_protocol пройдена: журнал игрушечной руки (hard и "
          "soft) проходит; порча ID, b_used, logits, C0, Z, схвата, "
          "действий, NaN, форм и подмена режима — отказ; smoke: CPU-повтор "
          "root_ln/classifier/plan_proj/Expand1 проходит, порча — отказ; Expand1 с "
          "сокращающимися слагаемыми и честным fp16: явный повтор приведений "
          "проходит (прежний допуск отверг бы), внесённая ошибка 1e−3 — "
          "отказ")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-30: планы и проверки")
    ap.add_argument("cmd", choices=("plan", "check-plan", "check-block",
                                    "smoke-check", "cells", "info",
                                    "selftest"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=tuple(k30m.MODES), default="hard")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--block", nargs=5, default=None)
    ap.add_argument("--which", choices=("smoke", "all"), default="all")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    try:
        if a.cmd == "plan":
            pth = plan_path(a.seed, a.mode)
            if os.path.exists(pth):
                print(f"  план {pth} уже зафиксирован")
                return 0
            plan = make_plan(a.seed, a.mode, a.device)
            with open(pth + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=1)
            os.replace(pth + ".tmp", pth)
            print(f"  план {pth}: {plan['labels']}, ячеек {len(plan['cells'])}"
                  f"; K-25 описательно: "
                  f"{plan['optional_references'][f'k25_s{a.seed}']['status']}")
            return 0
        plan = json.load(open(a.plan, encoding="utf-8"))
        if a.cmd == "check-plan":
            prob = check_plan_inputs(plan)
            if not prob:
                p23.verify_refs(plan)
            print("\n".join(prob) if prob else "  входы и источники сверены")
            return 3 if prob else 0
        if a.cmd == "info":
            print(plan["snapshot_path"])
            return 0
        if a.cmd == "cells":
            for s, t, b in (plan["smoke"] if a.which == "smoke"
                            else plan["cells"]):
                for lab in plan["labels"]:
                    print(f"{lab}:{s}:{t}:{b}")
            return 0
        if a.cmd == "check-block":
            path, lab, s, t, b = a.block
            prob = check_block(plan, a.plan, path, lab, s, t, b)
            if prob:
                print("; ".join(prob[:6]))
            return 2 if prob else 0
        ok, gaps = check_run(plan, a.plan, run_dir(a.plan),
                             [tuple(x) for x in plan["smoke"]])
        print(f"  smoke: проверено {len(ok)} из {len(plan['smoke'])}; проблем "
              f"{len(gaps)}" + (f": {gaps[:4]}" if gaps else ""))
        return 3 if gaps else 0
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
