#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16u: обучение четырёх вариантов пары G+C (одна конфигурация).

Данные — половина B K-16 (обучение обоих модулей), выбор эпохи — по
val_sel (чистая L_r, при равенстве — более ранняя эпоха); A не
используется; val_confirm не открывается. Полная проверка разметки и кэша
(verify_h18=True), непересечение эпизодов B и val_sel.

L = L_r + 0.5·L_goal(C.aux) + 0.5·L_goal(G); L_r — MSE(r̂_norm, r*/r_σ).
Adam, batch 256, wd 1e-4, 40 эпох; lr C: косинус 1e-3 -> 1e-5; lr G:
косинус 1e-4 -> 1e-6. Сид 0; одинаковый порядок батчей у всех вариантов;
у трёх вариантов ctx_goal начальные веса C побитово равны. Входы C —
собственные предсказания G текущими весами (ничего не кэшируется, кроме
h18, a0 и признаков состояния).

Нормировки и правило поправки — K-16: dpos_sigma (A), r_sigma (должна
совпасть), scale_arm, α = 0.5, предел — из data/k16/models.pt.

ВЫХОД: data/k16u/models.pt (каждый вариант — своя пара G+C),
reports/k16u/train.json.
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k16_models as km  # noqa: E402
import k16u_models as um  # noqa: E402

SEED = 0
EPOCHS, BATCH, WD = 40, 256, 1e-4
LR_C, LR_G = (1e-3, 1e-5), (1e-4, 1e-6)
VARIANTS = ("frozen_goal", "separate_goal", "joint_goal", "joint_ctx")
EVAL_ROWS = 4096
OUT = "data/k16u/models.pt"
REPORT = "reports/k16u/train.json"
K16 = "data/k16/models.pt"
CODE = ("k16u_models.py", "k16u_train.py", "k16b_train.py",
        "k16_models.py", "k16_train.py", "k16_prepare.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def cos_lr(ep, epochs, lr0, lr1):
    return lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * (ep - 1)
                                                   / max(epochs - 1, 1)))


def batch_t(torch, D, idx, dev):
    import k16_train as kt
    return kt.to_t(torch, D, idx, dev, ("h18", "a0", "sf_hist", "rstar",
                                        "dpos", "drot6d", "grip", "etype"))


def losses(torch, out, b, r_sigma, dpos_sigma):
    import k16_train as kt
    lab = {k: (b[k].float() if k != "etype" else b[k])
           for k in ("dpos", "drot6d", "grip", "etype")}
    l_r = ((out["r"] - b["rstar"].float() / r_sigma) ** 2).mean()
    l_c = kt.goal_loss(torch, out["aux"], lab, dpos_sigma)
    l_g = kt.goal_loss(torch, out["goal"], lab, dpos_sigma)
    return l_r, l_c, l_g


def make_opt(torch, pair):
    groups = [dict(params=list(pair.C.parameters()), lr=LR_C[0], name="C")]
    if pair.route != "frozen":
        groups.append(dict(params=list(pair.G.parameters()), lr=LR_G[0],
                           name="G"))
    return torch.optim.Adam(groups, weight_decay=WD)


def evaluate(torch, pair, D, idx, dev, r_sigma, dpos_sigma):
    pair.eval()
    acc = np.zeros(3)
    with torch.no_grad():
        for s in range(0, len(idx), 1024):
            i = idx[s:s + 1024]
            b = batch_t(torch, D, i, dev)
            out = pair(b["h18"], b["a0"].float(), b["sf_hist"].float())
            lr_, lc, lg = losses(torch, out, b, r_sigma, dpos_sigma)
            acc += np.array([float(lr_), float(lc), float(lg)]) * len(i)
    acc /= max(len(idx), 1)
    return dict(L_r=acc[0], goal_C=acc[1], goal_G=acc[2])


def grad_norms(torch, pair, D, idx, dev, r_sigma, dpos_sigma):
    """Нормы градиента G от L_r и от ошибки цели G (диагностический
    батч)."""
    out = {}
    pair.train()
    for name in ("L_r", "goal_G"):
        pair.zero_grad(set_to_none=True)
        b = batch_t(torch, D, idx, dev)
        o = pair(b["h18"], b["a0"].float(), b["sf_hist"].float())
        l_r, _lc, l_g = losses(torch, o, b, r_sigma, dpos_sigma)
        loss = l_r if name == "L_r" else l_g
        if loss.requires_grad:
            loss.backward()
        g = [p.grad for p in pair.G.parameters() if p.grad is not None]
        out[name] = float(torch.sqrt(sum((x ** 2).sum() for x in g))) \
            if g else 0.0
    pair.zero_grad(set_to_none=True)
    return out


def train_variant(torch, mode, g_state, B, V, dev, r_sigma, dpos_sigma,
                  d_model, n_pos, epochs=EPOCHS, log=print):
    Pair = um.build(torch, d_model, n_pos)
    pair = Pair(mode, g_state, c_seed=SEED).to(dev)
    g0 = {k: v.detach().clone() for k, v in pair.G.state_dict().items()}
    opt = make_opt(torch, pair)
    rng = np.random.default_rng(SEED)              # одинаковый порядок
    ev_rng = np.random.default_rng(20261016)
    ev_tr = np.sort(ev_rng.choice(len(B["age"]), min(EVAL_ROWS,
                                                     len(B["age"])),
                                  replace=False))
    vidx = np.arange(len(V["age"]))
    diag = np.arange(min(256, len(B["age"])))
    rep = dict(mode=mode, grad_norms_init=grad_norms(
        torch, pair, B, diag, dev, r_sigma, dpos_sigma),
        initial=dict(train=evaluate(torch, pair, B, ev_tr, dev, r_sigma,
                                    dpos_sigma),
                     val=evaluate(torch, pair, V, vidx, dev, r_sigma,
                                  dpos_sigma)), curve=[])
    best = None
    n = len(B["age"])
    for ep in range(1, epochs + 1):
        for g in opt.param_groups:
            lr0, lr1 = LR_C if g["name"] == "C" else LR_G
            g["lr"] = cos_lr(ep, epochs, lr0, lr1)
        pair.train()
        perm = rng.permutation(n)
        for s in range(0, n, BATCH):
            b = batch_t(torch, B, perm[s:s + BATCH], dev)
            out = pair(b["h18"], b["a0"].float(), b["sf_hist"].float())
            l_r, l_c, l_g = losses(torch, out, b, r_sigma, dpos_sigma)
            loss = l_r + 0.5 * l_c + (0.5 * l_g if pair.route != "frozen"
                                      else 0.0)
            if not bool(torch.isfinite(loss)):
                raise SystemExit(f"{mode}: нечисловая ошибка на эпохе {ep}")
            opt.zero_grad()
            loss.backward()
            opt.step()
        tr = evaluate(torch, pair, B, ev_tr, dev, r_sigma, dpos_sigma)
        va = evaluate(torch, pair, V, vidx, dev, r_sigma, dpos_sigma)
        if not all(np.isfinite(list(tr.values()) + list(va.values()))):
            raise SystemExit(f"{mode}: нечисловая проверка на эпохе {ep}")
        rep["curve"].append(dict(epoch=ep, train=tr, val=va))
        if best is None or va["L_r"] < best[1]:
            best = (ep, va["L_r"], {k: v.detach().clone()
                                    for k, v in pair.state_dict().items()})
    pair.load_state_dict(best[2])
    dG = math.sqrt(sum(float(((pair.G.state_dict()[k].float() - v.float())
                              ** 2).sum()) for k, v in g0.items()))
    nG = math.sqrt(sum(float((v.float() ** 2).sum()) for v in g0.values()))
    rep.update(best_epoch=best[0], best_val_L_r=best[1],
               best_is_last=best[0] == epochs,
               params_total=sum(p.numel() for p in pair.parameters()),
               params_trainable=sum(p.numel() for p in pair.parameters()
                                    if p.requires_grad),
               g_rel_change=dG / max(nG, 1e-12))
    log(f"  {mode}: лучшая эпоха {best[0]}"
        + (" (последняя — завершённость не установлена)"
           if best[0] == epochs else "")
        + f", val L_r {best[1]:.4f}, изменение весов G {dG / nG:.3e}")
    return pair, rep


def metrics(torch, pair, V, B, dev, r_sigma, dpos_sigma, scale, alpha,
            clip):
    """skill, геометрия цели, нормы применённой поправки."""
    import k16_prepare as kp
    pair.eval()
    R, Gp = [], []
    with torch.no_grad():
        for s in range(0, len(V["age"]), 1024):
            i = np.arange(s, min(s + 1024, len(V["age"])))
            b = batch_t(torch, V, i, dev)
            o = pair(b["h18"], b["a0"].float(), b["sf_hist"].float())
            R.append(o["r"].cpu().numpy())
            Gp.append(o["goal"].cpu().numpy())
    rn = np.concatenate(R)                          # норм. единицы
    g = np.concatenate(Gp)
    rs = np.asarray(r_sigma, np.float64)
    tn = V["rstar"] / rs
    mse_n = float(((rn - tn) ** 2).mean())
    meanB = (B["rstar"] / rs).mean(0)
    by = {}
    for t, r in zip(B["task"], B["rstar"] / rs):
        by.setdefault(t, []).append(r)
    by = {t: np.mean(v, 0) for t, v in by.items()}
    task_pred = np.stack([by.get(t, meanB) for t in V["task"]])
    base = dict(zero=float((tn ** 2).mean()),
                mean=float(((meanB - tn) ** 2).mean()),
                task_mean=float(((task_pred - tn) ** 2).mean()))
    rc = rn * rs
    mse_codec = float(((rc - V["rstar"]) ** 2).mean())
    _a, info = km.apply_correction(V["a0"], rn, rs, scale, alpha, clip)
    ds = np.asarray(dpos_sigma)
    pe = np.linalg.norm(g[:, :3] * ds - V["dpos"], axis=1)
    ang = np.degrees(kp.rot_angle(kp.sixd_to_mat(g[:, 3:9]),
                                  kp.sixd_to_mat(V["drot6d"])))
    return dict(
        mse_norm=mse_n, mse_codec=mse_codec,
        skill_norm={k: 1 - mse_n / max(v, 1e-12) for k, v in base.items()},
        baselines_norm=base,
        goal=dict(pos_err_m_mean=float(pe.mean()),
                  pos_err_m_median=float(np.median(pe)),
                  rot_err_deg_mean=float(ang.mean()),
                  rot_err_deg_median=float(np.median(ang)),
                  grip_acc=float(((g[:, 9] > 0) == (V["grip"] > .5)).mean()),
                  type_acc=float((g[:, 10:13].argmax(1) == V["etype"])
                                 .mean())),
        applied_norm_exec=dict(median=float(np.median(info["norm_applied"])),
                               p95=float(np.percentile(info["norm_applied"],
                                                       95))),
        clip_rate=float(info["clipped"].mean()))


def fitcheck_summary(path="reports/k16/fitcheck.json"):
    if not os.path.exists(path):
        return None
    f = json.load(open(path))
    return dict(baselines=f["baselines"], heads={
        k: dict(overfit=v["overfit"], best_epoch=v["long"]["best_epoch"],
                best_val=v["long"]["best_val"],
                train_last=v["long"]["curve"][-1]["train"])
        for k, v in f["heads"].items()},
        note="диагностический контекст; веса data/k16fit не используются")


def main():
    ap = argparse.ArgumentParser(description="K-16u: обучение")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--labels", default="reports/k16/labels.npz")
    ap.add_argument("--h18-cache", default="data/k15f/h18_cache")
    ap.add_argument("--k16-models", default=K16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--report", default=REPORT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import torch
    import k15h_protocol as pc
    import k16_train as kt
    fs = fitcheck_summary()
    if fs:
        print("  fitcheck (контекст): " + "; ".join(
            f"{k}: 512 строк {v['overfit']['first']:.3f}->"
            f"{v['overfit']['last']:.3f}, лучшая эпоха {v['best_epoch']}, "
            f"val {v['best_val']:.4f}, train {v['train_last']:.3f}"
            for k, v in fs["heads"].items()))
    try:
        data, meta = kt.load_inputs(a.labels, a.h18_cache, verify=True)
        B, V = data["B"], data["V"]
        if set(B["epi"].tolist()) & set(V["epi"].tolist()):
            raise SystemExit("эпизоды B и val_sel пересекаются")
        ck16 = torch.load(a.k16_models, map_location="cpu",
                          weights_only=False)
        if ck16["provenance"]["labels_sha1"] != sha(a.labels):
            raise SystemExit("goal_hist K-16 обучен на другой разметке")
        r_sigma = np.maximum(B["rstar"].reshape(-1, 6).std(0), 1e-5)
        if not np.allclose(r_sigma, ck16["norms"]["r_sigma"], rtol=1e-5):
            raise SystemExit("r_sigma не равна K-16")
        dpos_sigma = np.asarray(ck16["norms"]["dpos_sigma"])
        dev = torch.device(a.device)
        rs_t = torch.tensor(r_sigma, dtype=torch.float32, device=dev)
        ds_t = torch.tensor(dpos_sigma, dtype=torch.float32, device=dev)
        g_state = ck16["goals"]["goal_hist"]["state"]
        d_model, n_pos = int(B["h18"].shape[2]), int(B["h18"].shape[1])
        scale = pc.harness_scale()
        variants, rep = {}, {}
        for mode in VARIANTS:
            pair, r = train_variant(torch, mode, g_state, B, V, dev, rs_t,
                                    ds_t, d_model, n_pos)
            r["metrics_val_sel"] = metrics(
                torch, pair, V, B, dev, r_sigma, dpos_sigma, scale,
                float(ck16["alpha"]), float(ck16["clip_env_norm"]))
            variants[mode] = dict(state=pair.state_dict(), mode=mode,
                                  use_goal=um.MODES[mode][0],
                                  route=um.MODES[mode][1],
                                  best_epoch=r["best_epoch"])
            rep[mode] = r
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    code = {f: sha(os.path.join(HERE, f)) for f in CODE}
    ck = dict(variants=variants, norms=dict(r_sigma=r_sigma,
                                            dpos_sigma=dpos_sigma),
              alpha=float(ck16["alpha"]),
              clip_env_norm=float(ck16["clip_env_norm"]),
              scale_arm=list(ck16["scale_arm"]), units=ck16.get("units"),
              d_model=d_model, n_pos=n_pos, seed=SEED,
              config=dict(epochs=EPOCHS, batch=BATCH, wd=WD, lr_C=LR_C,
                          lr_G=LR_G, loss="L_r + 0.5 goal_C + 0.5 goal_G"),
              provenance=dict(labels=os.path.abspath(a.labels),
                              labels_sha1=sha(a.labels),
                              h18_manifest_sha1=meta["h18_manifest_sha1"],
                              k16_models=os.path.abspath(a.k16_models),
                              k16_models_sha1=sha(a.k16_models), code=code))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(ck, a.out + ".tmp")
    os.replace(a.out + ".tmp", a.out)
    res = dict(fitcheck_summary=fs, variants=rep,
               models=dict(path=os.path.abspath(a.out), sha1=sha(a.out)),
               code=code, config=ck["config"],
               note=("MSE_norm — в нормировке r/r_σ; MSE_codec — в единицах "
                     "кодека; skill — только по нормированной"))
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1,
                  default=lambda x: x.tolist() if hasattr(x, "tolist")
                  else str(x))
    for mode, r in rep.items():
        m = r["metrics_val_sel"]
        print(f"  {mode}: эпоха {r['best_epoch']}, L_r val "
              f"{r['best_val_L_r']:.4f}; skill (норм.) к нулю "
              f"{100 * m['skill_norm']['zero']:+.1f}%, к средней "
              f"{100 * m['skill_norm']['mean']:+.1f}%, к средней по задаче "
              f"{100 * m['skill_norm']['task_mean']:+.1f}%; цель "
              f"{100 * m['goal']['pos_err_m_mean']:.1f} см / "
              f"{m['goal']['rot_err_deg_mean']:.1f}°, схват "
              f"{100 * m['goal']['grip_acc']:.1f}%, тип "
              f"{100 * m['goal']['type_acc']:.1f}%; ||δ|| исп. "
              f"{m['applied_norm_exec']['median']:.4f}; ∇G от L_r "
              f"{r['grad_norms_init']['L_r']:.2e}, от цели "
              f"{r['grad_norms_init']['goal_G']:.2e}; ΔG "
              f"{r['g_rel_change']:.2e}")
    print(f"  модели: {a.out}")
    return 0


# --- самопроверка ------------------------------------------------------------
def selftest():
    import tempfile
    import torch
    import k16_train as kt
    rng = np.random.default_rng(0)
    B, V = kt._synth(200, rng), kt._synth(60, rng)
    C = km.build(torch)
    torch.manual_seed(5)
    g_state = C["GoalNet"]().state_dict()
    dev = torch.device("cpu")
    rs = torch.full((6,), 0.05)
    ds = torch.full((3,), 0.1)
    Pair = um.build(torch)
    pairs = {m: Pair(m, g_state, c_seed=SEED) for m in VARIANTS}
    # одинаковая инициализация C у трёх вариантов ctx_goal; G — копии
    cg = [pairs[m].C for m in ("frozen_goal", "separate_goal",
                               "joint_goal")]
    for c in cg[1:]:
        assert all(torch.equal(p, q) for p, q in zip(cg[0].parameters(),
                                                     c.parameters()))
    ids = {id(p) for m in pairs for p in pairs[m].G.parameters()}
    assert len(ids) == len(list(pairs["joint_goal"].G.parameters())) * 4
    idx = np.arange(32)
    # backward только от L_r: G получает градиент лишь в совместных
    for m, p in pairs.items():
        gn = grad_norms(torch, p, B, idx, dev, rs, ds)
        if m in ("joint_goal", "joint_ctx"):
            assert gn["L_r"] > 0 and np.isfinite(gn["L_r"]), (m, gn)
        else:
            assert gn["L_r"] == 0.0, (m, gn)
        if m == "frozen_goal":
            assert gn["goal_G"] == 0.0
        else:
            assert gn["goal_G"] > 0
    # несколько шагов: G frozen побитово сохранён, у остальных изменился
    with tempfile.TemporaryDirectory():
        for m in VARIANTS:
            pair, rep = train_variant(torch, m, g_state, B, V, dev, rs, ds,
                                      768, 16, epochs=2, log=lambda *x: None)
            same = all(torch.equal(pair.G.state_dict()[k], v)
                       for k, v in g_state.items())
            assert same == (m == "frozen_goal"), m
            assert len(rep["curve"]) == 2
    # будущая разметка не влияет на forward
    p = pairs["joint_goal"].eval()
    b1 = batch_t(torch, B, idx, dev)
    B2 = dict(B, dpos=B["dpos"] + 9, rstar=B["rstar"] * 3)
    b2 = batch_t(torch, B2, idx, dev)
    with torch.no_grad():
        o1 = p(b1["h18"], b1["a0"].float(), b1["sf_hist"].float())
        o2 = p(b2["h18"], b2["a0"].float(), b2["sf_hist"].float())
    assert torch.equal(o1["r"], o2["r"]) and torch.equal(o1["goal"],
                                                         o2["goal"])
    # сохранение и загрузка восстанавливают выходы
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "p.pt")
        torch.save(p.state_dict(), f)
        q = Pair("joint_goal", g_state, c_seed=99).eval()
        q.load_state_dict(torch.load(f))
        with torch.no_grad():
            o3 = q(b1["h18"], b1["a0"].float(), b1["sf_hist"].float())
        assert torch.equal(o1["r"], o3["r"])
    # NaN — отказ
    bad = dict(B, rstar=B["rstar"] * np.nan)
    try:
        train_variant(torch, "joint_goal", g_state, bad, V, dev, rs, ds,
                      768, 16, epochs=1, log=lambda *x: None)
        raise AssertionError("принята нечисловая ошибка")
    except SystemExit:
        pass
    # поправка: ноль -> a0; схват и хвост из a0; предел — один множитель
    a0 = rng.uniform(-.5, .5, (3, 20, 7)).astype(np.float32)
    az, _ = km.apply_correction(a0, np.zeros((3, 8, 6)), np.ones(6),
                                np.ones(7), 0.5, 1.0)
    assert np.array_equal(az, a0)
    ab, inf = km.apply_correction(a0, np.ones((3, 8, 6)), np.full(6, .1),
                                  np.ones(7), 0.5, 0.1)
    assert np.allclose(inf["norm_applied"], 0.1)
    assert np.array_equal(ab[..., 6], a0[..., 6]) and np.array_equal(
        ab[:, 8:], a0[:, 8:])
    print("самопроверка k16u_train пройдена: маршрутизация градиентов G "
          "(L_r и цель), заморозка, копии G, одинаковая инициализация C, "
          "разметка не в forward, сохранение, NaN, поправка")
    return 0


if __name__ == "__main__":
    sys.exit(main())
