#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-17: обучение G (A) и двух пар G+C (B), одна конфигурация.

ДАННЫЕ — окна reports/k17/windows.npz (A, B, val_sel), h18 и a0 — из кэша
K-15f по сохранённым строкам; связь окон с кэшем и разметкой K-16
проверяется (SHA, эпизоды строк, r* = act − a0). val_confirm не
открывается.

ВЕСА ОКОН: каждая задача — равный суммарный вес, внутри задачи эпизоды
равны, внутри эпизода вес делится между окнами; три позиции окна — один
вес. Нормировано к среднему 1. Потери и проверочные метрики — взвешенные;
обычное среднее по окнам — дополнительно.

1. G на A: вход — наблюдение начала окна; цель — (Δp / dpos_sigma, 6D);
   dpos_sigma — по A. Adam, wd 1e-4, батч 256 окон, 40 эпох, lr косинус
   1e-3 -> 1e-5; эпоха — по взвешенной ошибке позы на val_sel.
2. Две копии G -> пары ctx_memory и held_goal на B: батч 128 окон (384
   позиции), 40 эпох, lr C 1e-3 -> 1e-5, lr G 1e-4 -> 1e-6; одинаковые
   порядок окон и начальные веса C.
   L = L_r + 0.5 L_goal,G + 0.5 L_goal,C; L_r — MSE(r̂, r*/r_sigma),
   среднее по трём позициям; цель G — поза конца окна относительно начала;
   вспомогательная цель C — та же поза относительно состояния позиции.
   G вычисляется один раз на окно; прогноз и скрытый вектор используются
   на всех позициях без отсоединения. Эпоха — по взвешенной L_r на
   val_sel (при равенстве — более ранняя).
r_sigma — по B (все позиции); α, предел и единицы — из data/k16/models.pt.
Перед полным обучением — проверка подгонки фиксированной подвыборки B
(256 окон, 100 эпох, постоянный lr); отсутствие снижения — отказ.

ВЫХОД: data/k17/models.pt, reports/k17/train.json.
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
import k17_models as m17  # noqa: E402

SEED = 0
EPOCHS, WD = 40, 1e-4
BATCH_G, BATCH_PAIR = 256, 128
LR_GPRE, LR_C, LR_G = (1e-3, 1e-5), (1e-3, 1e-5), (1e-4, 1e-6)
FIT_N, FIT_EPOCHS = 256, 100
EVAL_N = 4096
OUT = "data/k17/models.pt"
REPORT = "reports/k17/train.json"
K16 = "data/k16/models.pt"
WINDOWS = "reports/k17/windows.npz"
SPLITS = (("A", "train"), ("B", "train"), ("V", "val_sel"))
CODE = ("k17_models.py", "k17_train.py", "k17_prepare.py", "k16_models.py",
        "k16_prepare.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def cos_lr(ep, epochs, lr0, lr1):
    return lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * (ep - 1)
                                                   / max(epochs - 1, 1)))


def window_weights(task, episode):
    """Равный вес задач; внутри — эпизодов; внутри эпизода — окон."""
    task = np.asarray(task)
    episode = np.asarray(episode)
    w = np.zeros(len(task))
    for t in np.unique(task):
        mt = task == t
        eps = np.unique(episode[mt])
        for e in eps:
            me = mt & (episode == e)
            w[me] = 1.0 / (len(eps) * me.sum())
    return w * len(w) / max(w.sum(), 1e-300)


def wmean(x, w):
    x, w = np.asarray(x, np.float64), np.asarray(w, np.float64)
    return float((x * w).sum() / max(w.sum(), 1e-300))


# --- данные ------------------------------------------------------------------
def derive(D):
    """Признаки и цели из окон (numpy): sf, p, R, вспомогательные цели C."""
    N = len(D["t"])
    sf = np.stack([km.state_features(D["states"][:, k], D["hist"][:, k],
                                     D["hist_mask"][:, k], D["age"][:, k],
                                     True) for k in range(3)], 1)
    p = D["states"][..., :3].astype(np.float64)
    R = km.aa_to_mat_np(D["states"][..., 3:6])
    Rg = R[:, 0] @ m17.sixd_to_mat_np(D["drot6d"])
    goal_p = p[:, 0] + D["dpos"]
    aux_dp = goal_p[:, None] - p
    aux6d = km.mat_to_6d_np(np.swapaxes(R, -1, -2) @ Rg[:, None])
    D.update(sf=sf.astype(np.float32), p=p.astype(np.float32),
             R=R.astype(np.float32), aux_dp=aux_dp.astype(np.float32),
             aux6d=aux6d.astype(np.float32),
             w=window_weights(D["task"], D["episode"]).astype(np.float32))
    assert D["sf"].shape == (N, 3, km.STATE_FEAT)
    return D


def link_problems(wj, windows, cache, labels16):
    p = []
    if wj.get("kind") != "k17_windows" or not wj.get("ready"):
        p.append("windows.json не того вида или не готов")
    if wj.get("out", {}).get("sha1") != sha(windows):
        p.append("windows.npz изменён после подготовки")
    if wj.get("inputs", {}).get("h18_manifest_sha1") != sha(
            os.path.join(cache, "manifest.json")):
        p.append("кэш h18 не тот, по которому построены окна")
    if wj.get("inputs", {}).get("k16_labels_sha1") != sha(labels16):
        p.append("разметка K-16 не та, по которой построены окна")
    return p


def load_data(windows, cache, labels16, verify=True):
    import k15f_build_cache as kb
    wj = json.load(open(windows[:-4] + ".json"))
    prob = link_problems(wj, windows, cache, labels16)
    if prob:
        raise SystemExit("; ".join(prob))
    Z = dict(np.load(windows, allow_pickle=False))
    c = kb.load_cache(cache, verify_h18=verify)
    data = {}
    for sp, part in SPLITS:
        D = {k[len(sp) + 2:]: v for k, v in Z.items()
             if k.startswith(sp + "__")}
        rows = np.asarray(D["rows"], np.int64)
        if rows.ndim != 2 or rows.shape[1] != 3 or not len(rows):
            raise SystemExit(f"{sp}: строки окон формы {rows.shape}")
        if not np.array_equal(c[part]["epi"][rows],
                              np.repeat(D["episode"][:, None], 3, 1)):
            raise SystemExit(f"{sp}: эпизоды строк кэша не равны окнам")
        if not np.array_equal(c[part]["rows"][rows], D["cache_rows"]):
            raise SystemExit(f"{sp}: строки K-9a не равны окнам")
        rs_ = (c[part]["act"][rows][..., :8, :6].astype(np.float64)
               - c[part]["a0"][rows][..., :8, :6].astype(np.float64))
        if not np.allclose(rs_, D["rstar"], atol=1e-6):
            raise SystemExit(f"{sp}: r* окон не равен act − a0 кэша")
        u, inv = np.unique(rows.reshape(-1), return_inverse=True)
        D["h18U"] = np.asarray(c[part]["h18"][u], np.float16)
        D["a0U"] = np.asarray(c[part]["a0"][u][:, :8], np.float32)
        D["ridx"] = inv.reshape(rows.shape)
        D["rstar"] = D["rstar"].astype(np.float32)
        data[sp] = derive(D)
    eps = {sp: set(data[sp]["episode"].tolist()) for sp in data}
    if eps["A"] & eps["B"] or eps["A"] & eps["V"] or eps["B"] & eps["V"]:
        raise SystemExit("эпизоды A, B, val_sel пересекаются")
    return data, wj


def batch(torch, D, idx, dev):
    idx = np.asarray(idx)
    r = D["ridx"][idx]
    t = lambda x: torch.as_tensor(np.asarray(x), device=dev)  # noqa: E731
    return dict(h18=t(D["h18U"][r]), a0=t(D["a0U"][r]).float(),
                sf=t(D["sf"][idx]).float(), p=t(D["p"][idx]).float(),
                R=t(D["R"][idx]).float(), rstar=t(D["rstar"][idx]).float(),
                dpos=t(D["dpos"][idx]).float(),
                drot6d=t(D["drot6d"][idx]).float(),
                aux_dp=t(D["aux_dp"][idx]).float(),
                aux6d=t(D["aux6d"][idx]).float(), w=t(D["w"][idx]).float())


def pose_loss(g, dp, r6, ds):
    """Ошибка позы на окно: MSE положения (норм.) + MSE 6D."""
    return ((g[..., :3] - dp / ds) ** 2).mean(-1) + \
        ((g[..., 3:9] - r6) ** 2).mean(-1)


def run_pair(pair, b):
    """Окно -> выходы трёх позиций (G один раз)."""
    import torch
    st = pair.start(b["h18"][:, 0], b["a0"][:, 0], b["sf"][:, 0],
                    b["p"][:, 0], b["R"][:, 0])
    rs, auxs, geos = [], [], []
    for k in range(3):
        pos = torch.full((len(b["w"]),), k, device=b["w"].device)
        r, aux, geo = pair.correct(b["h18"][:, k], b["a0"][:, k],
                                   b["sf"][:, k], st["hid"], pos,
                                   st["p_star"], st["R_star"], b["p"][:, k],
                                   b["R"][:, k])
        rs.append(r)
        auxs.append(aux)
        geos.append(geo)
    return dict(st, r=torch.stack(rs, 1), aux=torch.stack(auxs, 1),
                geo=torch.stack(geos, 1))


def pair_losses(o, b, r_sigma, ds):
    """Потери на окно: L_r, L_goal,G, L_goal,C, L_r по позициям."""
    l_rk = ((o["r"] - b["rstar"] / r_sigma) ** 2).mean((-1, -2))
    l_g = pose_loss(o["goal"], b["dpos"], b["drot6d"], ds)
    l_c = pose_loss(o["aux"], b["aux_dp"], b["aux6d"], ds).mean(1)
    return l_rk.mean(1), l_g, l_c, l_rk


def wavg(torch, x, w):
    return (x * w).sum() / w.sum()


# --- обучение G ----------------------------------------------------------------
def eval_g(torch, G, D, dev, ds, idx=None):
    G.eval()
    idx = np.arange(len(D["w"])) if idx is None else idx
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 1024):
            b = batch(torch, D, idx[s:s + 1024], dev)
            g, _h = G(b["h18"][:, 0], b["a0"][:, 0], b["sf"][:, 0])
            out.append(g.float().cpu().numpy())
    return np.concatenate(out)


def g_loss_np(g, D, ds, idx=None):
    idx = np.arange(len(D["w"])) if idx is None else idx
    l = ((g[:, :3] - D["dpos"][idx] / ds) ** 2).mean(1) + \
        ((g[:, 3:9] - D["drot6d"][idx]) ** 2).mean(1)
    return wmean(l, D["w"][idx]), float(l.mean())


def train_g(torch, A, V, dev, ds_np, epochs=EPOCHS, log=print):
    M = m17.build(torch, int(A["h18U"].shape[2]), int(A["h18U"].shape[1]))
    torch.manual_seed(SEED)
    G = M["G"]().to(dev)
    ds = torch.tensor(ds_np, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(G.parameters(), lr=LR_GPRE[0], weight_decay=WD)
    rng = np.random.default_rng(SEED)
    ev = np.sort(np.random.default_rng(20261016).choice(
        len(A["w"]), min(EVAL_N, len(A["w"])), replace=False))
    curve, best = [], None
    for ep in range(1, epochs + 1):
        for g_ in opt.param_groups:
            g_["lr"] = cos_lr(ep, epochs, *LR_GPRE)
        G.train()
        perm = rng.permutation(len(A["w"]))
        for s in range(0, len(perm), BATCH_G):
            b = batch(torch, A, perm[s:s + BATCH_G], dev)
            g, _h = G(b["h18"][:, 0], b["a0"][:, 0], b["sf"][:, 0])
            loss = wavg(torch, pose_loss(g, b["dpos"], b["drot6d"], ds),
                        b["w"])
            if not bool(torch.isfinite(loss)):
                raise SystemExit(f"G: нечисловая ошибка на эпохе {ep}")
            opt.zero_grad()
            loss.backward()
            opt.step()
        tr = g_loss_np(eval_g(torch, G, A, dev, ds, ev), A, ds_np, ev)
        va = g_loss_np(eval_g(torch, G, V, dev, ds), V, ds_np)
        if not np.isfinite(tr + va).all():
            raise SystemExit(f"G: нечисловая проверка на эпохе {ep}")
        curve.append(dict(epoch=ep, train_w=tr[0], train_rows=tr[1],
                          val_w=va[0], val_rows=va[1]))
        if best is None or va[0] < best[1]:
            best = (ep, va[0], {k: v.detach().clone()
                                for k, v in G.state_dict().items()})
    G.load_state_dict(best[2])
    log(f"  G: лучшая эпоха {best[0]}"
        + (" (последняя)" if best[0] == epochs else "")
        + f", взвешенная ошибка позы val_sel {best[1]:.4f}")
    return G, dict(best_epoch=best[0], best_val_w=best[1],
                   best_is_last=best[0] == epochs, curve=curve,
                   params=m17.n_params(G))


def goal_metrics(g, D, ds):
    """Ошибка цели (см, градусы) против «остаться в текущей позе»."""
    ds = np.asarray(ds, np.float64)
    w = D["w"]
    pe = np.linalg.norm(g[:, :3] * ds - D["dpos"], axis=1) * 100
    pst = np.linalg.norm(D["dpos"], axis=1) * 100
    Rp = m17.sixd_to_mat_np(g[:, 3:9])
    Rt = m17.sixd_to_mat_np(D["drot6d"])
    ang = np.degrees(np.arccos(np.clip((np.trace(
        np.swapaxes(Rp, -1, -2) @ Rt, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    ast = np.degrees(np.arccos(np.clip((np.trace(Rt, axis1=-2, axis2=-1)
                                        - 1) / 2, -1, 1)))

    def s(x):
        return dict(w_mean=wmean(x, w), mean=float(x.mean()),
                    median=float(np.median(x)))
    return dict(pos_cm=s(pe), pos_cm_stay=s(pst), rot_deg=s(ang),
                rot_deg_stay=s(ast),
                pos_skill_vs_stay=1 - wmean(pe ** 2, w) / max(
                    wmean(pst ** 2, w), 1e-12),
                rot_skill_vs_stay=1 - wmean(ang ** 2, w) / max(
                    wmean(ast ** 2, w), 1e-12))


# --- обучение пар ----------------------------------------------------------------
def make_opt(torch, pair, lrs=None):
    lc, lg = lrs or (LR_C[0], LR_G[0])
    return torch.optim.Adam([dict(params=list(pair.C.parameters()), lr=lc,
                                  name="C"),
                             dict(params=list(pair.G.parameters()), lr=lg,
                                  name="G")], weight_decay=WD)


def eval_pair(torch, pair, D, dev, rs, ds, idx=None, keep=False):
    pair.eval()
    idx = np.arange(len(D["w"])) if idx is None else idx
    acc = dict(L_r=[], goal_G=[], goal_C=[], L_rk=[])
    outs = dict(r=[], goal=[], geo=[], aux=[])
    with torch.no_grad():
        for s in range(0, len(idx), 512):
            b = batch(torch, D, idx[s:s + 512], dev)
            o = run_pair(pair, b)
            l_r, l_g, l_c, l_rk = pair_losses(o, b, rs, ds)
            for k, v in zip(("L_r", "goal_G", "goal_C", "L_rk"),
                            (l_r, l_g, l_c, l_rk)):
                acc[k].append(v.float().cpu().numpy())
            if keep:
                for k in outs:
                    outs[k].append(o[k].float().cpu().numpy())
    acc = {k: np.concatenate(v) for k, v in acc.items()}
    w = D["w"][idx]
    res = {f"{k}_w": wmean(acc[k], w) for k in ("L_r", "goal_G", "goal_C")}
    res.update({f"{k}_rows": float(acc[k].mean())
                for k in ("L_r", "goal_G", "goal_C")})
    res["L_r_pos_w"] = [wmean(acc["L_rk"][:, k], w) for k in range(3)]
    if keep:
        return res, {k: np.concatenate(v) for k, v in outs.items()}, acc
    return res


def train_pair(torch, kind, g_state, B, V, dev, rs, ds, epochs=EPOCHS,
               rows=None, lrs=None, cosine=True, log=print, select=True):
    M = m17.build(torch, int(B["h18U"].shape[2]), int(B["h18U"].shape[1]))
    pair = M["Pair"](kind, ds.cpu().numpy(), g_state, c_seed=SEED).to(dev)
    opt = make_opt(torch, pair, lrs)
    rng = np.random.default_rng(SEED)              # одинаковый порядок
    rows = np.arange(len(B["w"])) if rows is None else rows
    ev = rows if len(rows) <= EVAL_N else np.sort(
        np.random.default_rng(20261016).choice(rows, EVAL_N, replace=False))
    curve, best = [], None
    init = eval_pair(torch, pair, B, dev, rs, ds, ev)
    for ep in range(1, epochs + 1):
        if cosine:
            for g_ in opt.param_groups:
                lr0, lr1 = LR_C if g_["name"] == "C" else LR_G
                g_["lr"] = cos_lr(ep, epochs, lr0, lr1)
        pair.train()
        perm = rng.permutation(rows)
        for s in range(0, len(perm), BATCH_PAIR):
            b = batch(torch, B, perm[s:s + BATCH_PAIR], dev)
            o = run_pair(pair, b)
            l_r, l_g, l_c, _k = pair_losses(o, b, rs, ds)
            loss = wavg(torch, l_r + 0.5 * l_g + 0.5 * l_c, b["w"])
            if not bool(torch.isfinite(loss)):
                raise SystemExit(f"{kind}: нечисловая ошибка на эпохе {ep}")
            opt.zero_grad()
            loss.backward()
            opt.step()
        tr = eval_pair(torch, pair, B, dev, rs, ds, ev)
        va = eval_pair(torch, pair, V, dev, rs, ds) if select else None
        vals = list(tr.values()) + (list(va.values()) if va else [])
        if not np.isfinite(np.hstack([np.ravel(x) for x in vals])).all():
            raise SystemExit(f"{kind}: нечисловая проверка на эпохе {ep}")
        curve.append(dict(epoch=ep, train=tr, val=va))
        if select and (best is None or va["L_r_w"] < best[1]):
            best = (ep, va["L_r_w"], {k: v.detach().clone()
                                      for k, v in pair.state_dict().items()})
    rep = dict(kind=kind, initial_train=init, curve=curve)
    if select:
        pair.load_state_dict(best[2])
        rep.update(best_epoch=best[0], best_val_L_r_w=best[1],
                   best_is_last=best[0] == epochs,
                   params=m17.n_params(pair))
        log(f"  {kind}: лучшая эпоха {best[0]}"
            + (" (последняя — бюджет мог быть мал)"
               if best[0] == epochs else "")
            + f", взвешенная L_r val_sel {best[1]:.4f}")
    return pair, rep


def fitcheck(torch, g_state, B, dev, rs, ds, log=print, epochs=FIT_EPOCHS):
    rows = np.sort(np.random.default_rng(20261016).choice(
        len(B["w"]), min(FIT_N, len(B["w"])), replace=False))
    out = {}
    for kind in m17.KINDS:
        _p, rep = train_pair(torch, kind, g_state, B, None, dev, rs, ds,
                             epochs=epochs, rows=rows, cosine=False,
                             select=False, log=log)
        f = rep["initial_train"]["L_r_w"]
        last = rep["curve"][-1]["train"]["L_r_w"]
        out[kind] = dict(rows=len(rows), epochs=epochs, L_r_first=f,
                         L_r_last=last, ratio=last / max(f, 1e-12))
        log(f"  подгонка {kind}: {len(rows)} окон, L_r {f:.4f} -> "
            f"{last:.4f}")
        if not last < f:
            raise SystemExit(f"подгонка {kind}: ошибка не снижается")
    return out


def corr_metrics(torch, pair, D, Bref, dev, rs_np, ds_np, scale, alpha,
                 clip):
    """Поправка против ориентиров (по B), позиции, нормы, срезы, задачи."""
    rs = torch.tensor(rs_np, dtype=torch.float32, device=dev)
    ds = torch.tensor(ds_np, dtype=torch.float32, device=dev)
    res, o, acc = eval_pair(torch, pair, D, dev, rs, ds, keep=True)
    rsn = np.asarray(rs_np, np.float64)
    tn = D["rstar"] / rsn                                  # [N, 3, 8, 6]
    Bn = Bref["rstar"] / rsn
    mean = Bn.reshape(-1, 8, 6).mean(0)
    by = {t: Bn[Bref["task"] == t].reshape(-1, 8, 6).mean(0)
          for t in np.unique(Bref["task"])}
    tm = np.stack([by.get(t, mean) for t in D["task"]])[:, None]
    w = D["w"]
    err = ((o["r"] - tn) ** 2).mean((-1, -2))              # [N, 3]
    base = dict(zero=(tn ** 2).mean((-1, -2)),
                mean=((mean - tn) ** 2).mean((-1, -2)),
                task_mean=((tm - tn) ** 2).mean((-1, -2)))

    def skill(m):
        e = err[m].mean(1)
        return {k: 1 - wmean(e, w[m]) / max(wmean(v[m].mean(1), w[m]),
                                            1e-12) for k, v in base.items()}
    allm = np.ones(len(w), bool)
    pos = [{k: 1 - wmean(err[:, j], w) / max(wmean(v[:, j], w), 1e-12)
            for k, v in base.items()} for j in range(3)]
    a0 = D["a0U"][D["ridx"]]                               # [N, 3, 8, 7]
    napp, clp = [], []
    for j in range(3):
        _a, inf = km.apply_correction(a0[:, j], o["r"][:, j], rsn, scale,
                                      alpha, clip)
        napp.append(inf["norm_applied"])
        clp.append(inf["clipped"])
    napp, clp = np.stack(napp, 1), np.stack(clp, 1)
    gs = D["grip_switch"].astype(bool)
    tasks = {}
    for t in np.unique(D["task"]):
        m = D["task"] == t
        tasks[str(t)] = dict(windows=int(m.sum()),
                             episodes=len(set(D["episode"][m].tolist())),
                             skill_zero=skill(m)["zero"],
                             L_r=float(err[m].mean()))
    return dict(
        losses=res, skill_w=skill(allm),
        skill_rows={k: 1 - float(err.mean()) / max(float(v.mean()), 1e-12)
                    for k, v in base.items()},
        skill_by_position=pos,
        grip_switch={"with": dict(windows=int(gs.sum()), **skill(gs)),
                     "without": dict(windows=int((~gs).sum()),
                                     **skill(~gs))},
        applied_norm_exec=[dict(median=float(np.median(napp[:, j])),
                                p95=float(np.percentile(napp[:, j], 95)),
                                clip_rate=float(clp[:, j].mean()))
                           for j in range(3)],
        window_norm_sum=dict(median=float(np.median(napp.sum(1))),
                             p95=float(np.percentile(napp.sum(1), 95))),
        goal=goal_metrics(o["goal"], D, ds_np), per_task=tasks)


def main():
    ap = argparse.ArgumentParser(description="K-17: обучение")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--windows", default=WINDOWS)
    ap.add_argument("--h18-cache", default="data/k15f/h18_cache")
    ap.add_argument("--k16-labels", default="reports/k16/labels.npz")
    ap.add_argument("--k16-models", default=K16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--report", default=REPORT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import torch
    import k15h_protocol as pc
    try:
        if os.path.exists(a.out):
            raise SystemExit(f"{a.out} уже есть (новые веса — в новый файл)")
        data, wj = load_data(a.windows, a.h18_cache, a.k16_labels, True)
        A, B, V = data["A"], data["B"], data["V"]
        ck16 = torch.load(a.k16_models, map_location="cpu",
                          weights_only=False)
        scale = pc.harness_scale()
        if not np.allclose(np.asarray(ck16["scale_arm"]), scale[:6]):
            raise SystemExit("scale_arm K-16 не равен масштабу харнесса")
        alpha, clip = float(ck16["alpha"]), float(ck16["clip_env_norm"])
        dev = torch.device(a.device)
        ds_np = np.maximum(A["dpos"].std(0), 1e-4)
        rs_np = np.maximum(B["rstar"].reshape(-1, 6).std(0), 1e-5)
        ds = torch.tensor(ds_np, dtype=torch.float32, device=dev)
        rs = torch.tensor(rs_np, dtype=torch.float32, device=dev)
        print(f"  окон: A {len(A['w'])}, B {len(B['w'])}, val_sel "
              f"{len(V['w'])}; dpos_sigma {np.round(ds_np, 4).tolist()}")
        G, rep_g = train_g(torch, A, V, dev, ds_np)
        g_state = {k: v.detach().cpu().clone()
                   for k, v in G.state_dict().items()}
        rep_g["val_sel"] = goal_metrics(eval_g(torch, G, V, dev, ds), V,
                                        ds_np)
        rep_g["B_before_joint"] = goal_metrics(eval_g(torch, G, B, dev, ds),
                                               B, ds_np)
        fit = fitcheck(torch, g_state, B, dev, rs, ds)
        pairs, rep = {}, {}
        for kind in m17.KINDS:
            pair, r = train_pair(torch, kind, g_state, B, V, dev, rs, ds)
            r["val_sel"] = corr_metrics(torch, pair, V, B, dev, rs_np, ds_np,
                                        scale, alpha, clip)
            pairs[kind] = dict(state={k: v.detach().cpu() for k, v in
                                      pair.state_dict().items()},
                               kind=kind, best_epoch=r["best_epoch"])
            rep[kind] = r
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    code = {f: sha(os.path.join(HERE, f)) for f in CODE}
    d_model, n_pos = int(B["h18U"].shape[2]), int(B["h18U"].shape[1])
    ck = dict(kind="k17_models", G_pre=g_state, pairs=pairs,
              norms=dict(r_sigma=rs_np, dpos_sigma=ds_np), alpha=alpha,
              clip_env_norm=clip, scale_arm=list(ck16["scale_arm"]),
              units=ck16.get("units"), d_model=d_model, n_pos=n_pos,
              seed=SEED,
              config=dict(epochs=EPOCHS, wd=WD, batch_g=BATCH_G,
                          batch_pair=BATCH_PAIR, lr_g_pre=LR_GPRE,
                          lr_C=LR_C, lr_G=LR_G,
                          loss="L_r + 0.5 goal_G + 0.5 goal_C",
                          weights="задачи равны, эпизоды равны, окна "
                                  "делят вес эпизода"),
              provenance=dict(windows=os.path.abspath(a.windows),
                              windows_sha1=sha(a.windows),
                              k16_labels=os.path.abspath(a.k16_labels),
                              k16_labels_sha1=sha(a.k16_labels),
                              k16_models=os.path.abspath(a.k16_models),
                              k16_models_sha1=sha(a.k16_models),
                              h18_manifest_sha1=sha(os.path.join(
                                  a.h18_cache, "manifest.json")),
                              windows_code_sha1=wj.get("code"), code=code))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(ck, a.out + ".tmp")
    os.replace(a.out + ".tmp", a.out)
    res = dict(G=rep_g, fitcheck=fit, pairs=rep,
               models=dict(path=os.path.abspath(a.out), sha1=sha(a.out)),
               code=code, config=ck["config"],
               norms=dict(r_sigma=rs_np.tolist(), dpos_sigma=ds_np.tolist()),
               note=("w — веса окон (задачи и эпизоды равны); rows — "
                     "обычное среднее; L_r — в нормировке r/r_sigma; "
                     "ориентиры средней поправки — только по B"))
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1,
                  default=lambda x: x.tolist() if hasattr(x, "tolist")
                  else str(x))
    gv = rep_g["val_sel"]
    print(f"  G val_sel: {gv['pos_cm']['w_mean']:.2f} см (остаться "
          f"{gv['pos_cm_stay']['w_mean']:.2f}), {gv['rot_deg']['w_mean']:.2f}"
          f"° (остаться {gv['rot_deg_stay']['w_mean']:.2f}°)")
    for kind, r in rep.items():
        m = r["val_sel"]
        print(f"  {kind}: эпоха {r['best_epoch']}"
              + (" (последняя)" if r["best_is_last"] else "")
              + f"; skill (взв.) к нулю {100 * m['skill_w']['zero']:+.1f}%, "
              f"к средней {100 * m['skill_w']['mean']:+.1f}%, к средней по "
              f"задаче {100 * m['skill_w']['task_mean']:+.1f}%; по позициям "
              + "/".join(f"{100 * p['zero']:+.1f}"
                         for p in m["skill_by_position"])
              + f"; цель {m['goal']['pos_cm']['w_mean']:.2f} см / "
              f"{m['goal']['rot_deg']['w_mean']:.2f}°; ||δ|| исп. медиана "
              + "/".join(f"{x['median']:.3f}"
                         for x in m["applied_norm_exec"])
              + ", ограничение "
              + "/".join(f"{100 * x['clip_rate']:.0f}%"
                         for x in m["applied_norm_exec"]))
    print(f"  модели: {a.out}; отчёт: {a.report}")
    return 0


# --- самопроверка ------------------------------------------------------------
def synth(n, rng, n_ep=6, tasks=("a", "b", "c")):
    """Синтетические окна той же структуры, что load_data."""
    U = 3 * n
    D = dict(t=np.arange(n), episode=rng.integers(0, n_ep, n) +
             1000 * rng.integers(0, 1, n),
             task=np.asarray([tasks[i % len(tasks)] for i in range(n)]),
             states=np.concatenate([rng.standard_normal((n, 3, 3)) * .1,
                                    rng.standard_normal((n, 3, 3)) * .3,
                                    rng.uniform(0, .04, (n, 3, 2))], -1),
             hist=rng.standard_normal((n, 3, 3, 8)) * .1,
             hist_mask=np.ones((n, 3, 3), bool),
             age=np.tile([0, 8, 16], (n, 1)) + 24,
             dpos=rng.standard_normal((n, 3)) * .1,
             rstar=(rng.standard_normal((n, 3, 8, 6)) * .05)
             .astype(np.float32),
             grip_switch=rng.random(n) < .3,
             h18U=rng.standard_normal((U, 16, 768)).astype(np.float16),
             a0U=rng.uniform(-.5, .5, (U, 8, 7)).astype(np.float32),
             ridx=np.arange(U).reshape(n, 3))
    D["episode"] = D["episode"] + 100 * (np.arange(n) % len(tasks))
    D["drot6d"] = km.mat_to_6d_np(km.aa_to_mat_np(
        rng.standard_normal((n, 3)) * .2))
    return derive(D)


def selftest():
    import tempfile
    import torch
    rng = np.random.default_rng(0)
    # веса: равные задачи, эпизоды, окна
    w = window_weights(["a", "a", "a", "b"], [1, 1, 2, 3])
    assert np.allclose(w / w.sum(), [.125, .125, .25, .5])
    B, V = synth(48, rng), synth(24, rng)
    dev = torch.device("cpu")
    ds_np = np.maximum(B["dpos"].std(0), 1e-4)
    rs_np = np.maximum(B["rstar"].reshape(-1, 6).std(0), 1e-5)
    ds, rs = torch.tensor(ds_np).float(), torch.tensor(rs_np).float()
    # вспомогательная цель позиции 0 совпадает с целью G
    assert np.allclose(B["aux_dp"][:, 0], B["dpos"], atol=1e-6)
    assert np.allclose(B["aux6d"][:, 0], B["drot6d"], atol=1e-5)
    G, rg = train_g(torch, B, V, dev, ds_np, epochs=2, log=lambda *x: None)
    assert len(rg["curve"]) == 2
    g_state = {k: v.clone() for k, v in G.state_dict().items()}
    M = m17.build(torch)
    pairs = {k: M["Pair"](k, ds_np, g_state, c_seed=SEED)
             for k in m17.KINDS}
    pa, ph = pairs["ctx_memory"], pairs["held_goal"]
    assert all(torch.equal(p, q) for p, q in zip(pa.C.parameters(),
                                                 ph.C.parameters()))
    assert all(torch.equal(p, q) for p, q in zip(pa.G.parameters(),
                                                 ph.G.parameters()))
    assert {id(p) for p in pa.G.parameters()}.isdisjoint(
        {id(p) for p in ph.G.parameters()})
    idx = np.arange(16)
    b = batch(torch, B, idx, dev)
    # ctx_memory не использует геометрию; обе используют память G
    o = run_pair(pa, b)
    assert not o["geo"].any()
    oh = run_pair(ph, b)
    assert oh["geo"].abs().sum() > 0
    # градиент L_r каждой позиции доходит до G и C у обеих моделей
    for p_ in (pa, ph):
        for k in range(3):
            p_.zero_grad(set_to_none=True)
            o = run_pair(p_, b)
            l = ((o["r"][:, k] - b["rstar"][:, k] / rs) ** 2).mean()
            l.backward()
            gG = sum(float(x.grad.abs().sum()) for x in p_.G.parameters()
                     if x.grad is not None)
            gC = sum(float(x.grad.abs().sum()) for x in p_.C.parameters()
                     if x.grad is not None)
            assert gG > 0 and gC > 0, (p_.kind, k)
    # будущие цели и r* не влияют на выходы
    B2 = dict(B, dpos=B["dpos"] + 5, rstar=B["rstar"] * 3,
              aux_dp=B["aux_dp"] * 0, drot6d=B["drot6d"][:, ::-1].copy())
    b2 = batch(torch, B2, idx, dev)
    with torch.no_grad():
        o1, o2 = run_pair(ph.eval(), b), run_pair(ph, b2)
    assert torch.equal(o1["r"], o2["r"]) and torch.equal(o1["goal"],
                                                         o2["goal"])
    # цель сохраняется в мировых координатах на всех позициях: geo по
    # позиции k = ошибка от p*, R* окна до состояния позиции k
    pn, Rn = o1["p_star"].numpy(), o1["R_star"].numpy()
    for k in range(3):
        e = m17.geo_error_np(pn, Rn, B["p"][idx, k], B["R"][idx, k], ds_np)
        assert np.allclose(o1["geo"][:, k].numpy(), e, atol=1e-5)
    # короткое обучение, одинаковый порядок, подгонка, метрики
    for kind in m17.KINDS:
        p_, r_ = train_pair(torch, kind, g_state, B, V, dev, rs, ds,
                            epochs=2, log=lambda *x: None)
        assert len(r_["curve"]) == 2 and "best_epoch" in r_
    fit = fitcheck(torch, g_state, B, dev, rs, ds, log=lambda *x: None,
                   epochs=30)
    assert all(v["L_r_last"] < v["L_r_first"] for v in fit.values())
    m = corr_metrics(torch, p_, V, B, dev, rs_np, ds_np, np.ones(7) * .9,
                     .5, .3)
    assert len(m["skill_by_position"]) == 3
    assert all(x["p95"] <= .3 + 1e-9 for x in m["applied_norm_exec"])
    gm = goal_metrics(np.concatenate([V["dpos"] / ds_np, V["drot6d"]], 1),
                      V, ds_np)
    assert gm["pos_cm"]["mean"] < 1e-6 and gm["rot_deg"]["mean"] < 1e-3
    # сохранение и загрузка восстанавливают выходы
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "p.pt")
        torch.save(ph.state_dict(), f)
        q = M["Pair"]("held_goal", ds_np, None, c_seed=99).eval()
        q.load_state_dict(torch.load(f))
        with torch.no_grad():
            o3 = run_pair(q, b)
        assert torch.equal(o1["r"], o3["r"])
    # NaN — отказ
    bad = dict(B, rstar=B["rstar"] * np.nan)
    try:
        train_pair(torch, "held_goal", g_state, bad, V, dev, rs, ds,
                   epochs=1, log=lambda *x: None)
        raise AssertionError("принята нечисловая ошибка")
    except SystemExit:
        pass
    print("самопроверка k17_train пройдена: веса задач/эпизодов/окон, "
          "копии G, одинаковые C, нули геометрии у ctx_memory, градиент "
          "каждой позиции в G и C, разметка не в forward, мировая цель на "
          "трёх позициях, подгонка, метрики, сохранение, NaN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
