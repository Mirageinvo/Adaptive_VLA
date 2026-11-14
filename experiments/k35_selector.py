#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-35: два малых селектора ветвления по доступным признакам (§11).

PCA-32 (без whitening, знак компоненты — её максимальная по модулю
координата положительна) отдельно для корней и средних, fit только на
oracle-train (средние — по всем a с равными весами).
Вход первого: 4 корня × (32 PCA + l/20 + r/20) = 136.
Вход второго: 136 + 4 средних × (32 PCA + l/20 + r/20 + mid_id/7) = 140 +
one-hot a (6) = 282. Непрерывные признаки стандартизованы по train (std
снизу 1e−6), one-hot — нет.
Сеть: Linear(in, 128) → GELU → Linear(128, 6), последний Linear = 0;
параметров 18 310 и 36 998 (при p = 32).
Цели: mid[i, a, b] = C_full[i, 6a+b] − min_b; первый селектор — стоимость
out-of-fold продолжения: cost_root[i, a] = C_full[i, 6a + b_oof[i, a]] − min_a
(5 folds по эпизодам; PCA/стандартизация fold — только по его обучающей
части). Масштаб — медиана положительных относительных стоимостей train
(снизу 1e−6). MSE шести нормированных стоимостей. CPU FP32, batch 128,
AdamW lr 1e−3, wd 1e−4, clip 1.0; второй селектор (OOF и финальный) —
ровно 1000 обновлений, последний; первый — 2000, проверка каждые 50 и на
0, best — по РЕАЛЬНОЙ составной offline-потере C_full[i, 6â + b̂(i, â)] на
oracle-val с финальным вторым селектором (ничья — ранний шаг).
Решение: меньший ID среди стоимостей ≤ min + 1e−6 (нормированные
единицы); NaN/Inf — технический отказ.
Гейт обучаемости (oracle-val): capture = (L_C − L_A)/(L_C − L_O) ≥ 0.25 и
L_A < L_C; L_A ≤ L_N; arm-8 A не хуже arm-8 C; маршруты допустимы.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k35_oracle as ko  # noqa: E402

P_PCA = 32
HIDDEN = 128
TIE = 1e-6
N_FOLDS = 5
FOLD_SEED = 35501
SEL_SEED = 35502
BATCH, LR, WD, CLIP = 128, 1e-3, 1e-4, 1.0
MID_UPDATES, ROOT_UPDATES, ROOT_EVAL = 1000, 2000, 50
D_FIRST, D_SECOND = 4 * (P_PCA + 2), 4 * (P_PCA + 2) + 4 * (P_PCA + 3) + 6
GATE_RULE = dict(capture=0.25)


def n_params(d_in, h=HIDDEN, k=6):
    return d_in * h + h + h * k + k


# --- PCA и признаки ---------------------------------------------------------------
def pca_fit(X, p):
    X = np.asarray(X, np.float64)
    mu = X.mean(0)
    _u, _s, vt = np.linalg.svd(X - mu, full_matrices=False)
    comp = vt[:p].copy()
    for i in range(comp.shape[0]):
        j = int(np.argmax(np.abs(comp[i])))
        if comp[i, j] < 0:
            comp[i] = -comp[i]
    return dict(mean=mu.astype(np.float32), comp=comp.astype(np.float32))


def pca_apply(P, X):
    return (np.asarray(X, np.float32) - P["mean"]) @ P["comp"].T


def raw_first(P_root, roots, iv_root):
    """roots [N, 4, H], iv [N, 4, 2] -> [N, 136] (без стандартизации)."""
    z = pca_apply(P_root, roots.reshape(-1, roots.shape[-1])).reshape(
        roots.shape[0], 4, -1)
    return np.concatenate([z, iv_root / 20.0], -1).reshape(roots.shape[0], -1)


def raw_second(P_root, P_mid, roots, iv_root, mids, iv_mid, mid_ids, a):
    """mids [N, 4, H] (для выбранного a), mid_ids [N, 4], a [N] ->
    (непрерывные [N, 276], one-hot [N, 6])."""
    f1 = raw_first(P_root, roots, iv_root)
    z = pca_apply(P_mid, mids.reshape(-1, mids.shape[-1])).reshape(
        mids.shape[0], 4, -1)
    f2 = np.concatenate([z, iv_mid / 20.0, (np.asarray(mid_ids, np.float32)
                                            / 7.0)[..., None]], -1)
    oh = np.eye(6, dtype=np.float32)[np.asarray(a)]
    return np.concatenate([f1, f2.reshape(mids.shape[0], -1)], -1), oh


def std_fit(X):
    return dict(mu=X.mean(0).astype(np.float32),
                sd=np.maximum(X.std(0), 1e-6).astype(np.float32))


def std_apply(S, X):
    return (X - S["mu"]) / S["sd"]


def mid_ids_of(a):
    import k35_topology as tp
    return np.asarray([tp.CUTS[6 * int(x)]["mids"] for x in np.asarray(a)])


# --- обучение ------------------------------------------------------------------------
def make_net(torch, d_in, seed):
    g = torch.Generator().manual_seed(int(seed))
    rng = torch.random.get_rng_state()
    try:
        net = torch.nn.Sequential(torch.nn.Linear(d_in, HIDDEN),
                                  torch.nn.GELU(),
                                  torch.nn.Linear(HIDDEN, 6))
        with torch.no_grad():
            b = 1.0 / np.sqrt(d_in)
            net[0].weight.copy_((torch.rand(net[0].weight.shape,
                                            generator=g) * 2 - 1) * b)
            net[0].bias.copy_((torch.rand(HIDDEN, generator=g) * 2 - 1) * b)
            net[2].weight.zero_()
            net[2].bias.zero_()
    finally:
        torch.random.set_rng_state(rng)
    return net


def train_net(torch, X, Y, updates, seed, evaluate=None, every=None):
    """X [n, d], Y [n, 6] (нормированные относительные стоимости)."""
    torch.set_num_threads(4)
    net = make_net(torch, X.shape[1], seed)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    Yt = torch.as_tensor(Y, dtype=torch.float32)
    rng = np.random.default_rng(seed + 7)
    best = None
    hist = []
    for u in range(updates + 1):
        if evaluate is not None and u % every == 0:
            v = float(evaluate(net))
            hist.append((u, v))
            if best is None or v < best[1]:
                best = (u, v, {k: t.detach().clone() for k, t in
                               net.state_dict().items()})
        if u == updates:
            break
        idx = rng.integers(0, len(X), min(BATCH, len(X)))
        loss = ((net(Xt[idx]) - Yt[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step()
    if best is not None:
        net.load_state_dict(best[2])
        return net, dict(best_update=best[0], best_value=best[1], curve=hist)
    return net, dict(best_update=updates, curve=hist)


def decide(scores):
    """[n, 6] -> решения: меньший ID среди ≤ min + TIE; нечисловые — отказ."""
    s = np.asarray(scores, np.float64)
    if not np.isfinite(s).all():
        raise SystemExit("селектор: нечисловые scores")
    m = s.min(1, keepdims=True)
    return np.argmax(s <= m + TIE, 1)


def predict(torch, net, X):
    with torch.no_grad():
        return net(torch.as_tensor(X, dtype=torch.float32)).numpy()


def rel(C):
    """[..., 6] -> относительно минимума."""
    return C - C.min(-1, keepdims=True)


def scale_of(R):
    pos = R[R > 0]
    return float(max(np.median(pos), 1e-6)) if pos.size else 1e-6


def folds_by_episode(episodes, k=N_FOLDS, seed=FOLD_SEED):
    eps = np.unique(episodes)
    if len(eps) < k:
        raise SystemExit(f"эпизодов {len(eps)} < {k} folds")
    perm = np.random.default_rng(seed).permutation(len(eps))
    fold_of_ep = {int(eps[j]): i % k for i, j in enumerate(perm)}
    f = np.asarray([fold_of_ep[int(e)] for e in episodes])
    for i in range(k):
        if (f != i).sum() == 0 or (f == i).sum() == 0:
            raise SystemExit("пустой fold")
    return f


def fit_transforms(O, idx):
    """PCA корней и средних + стандартизации по строкам idx."""
    roots, mids = O["roots"][idx], O["mids"][idx]
    P_root = pca_fit(roots.reshape(-1, roots.shape[-1]), P_PCA)
    P_mid = pca_fit(mids.reshape(-1, mids.shape[-1]), P_PCA)
    f1 = raw_first(P_root, roots, O["iv_root"][idx])
    S1 = std_fit(f1)
    cont = []
    for a in range(6):
        aa = np.full(len(idx), a)
        c, _oh = raw_second(P_root, P_mid, roots, O["iv_root"][idx],
                            mids[:, a], O["iv_mid"][idx, a], mid_ids_of(aa),
                            aa)
        cont.append(c)
    S2 = std_fit(np.concatenate(cont, 0))
    return dict(P_root=P_root, P_mid=P_mid, S1=S1, S2=S2)


def X_first(Tr, O, idx):
    return std_apply(Tr["S1"], raw_first(Tr["P_root"], O["roots"][idx],
                                         O["iv_root"][idx]))


def X_second(Tr, O, idx, a):
    a = np.asarray(a)
    c, oh = raw_second(Tr["P_root"], Tr["P_mid"], O["roots"][idx],
                       O["iv_root"][idx], O["mids"][idx, a],
                       O["iv_mid"][idx, a], mid_ids_of(a), a)
    return np.concatenate([std_apply(Tr["S2"], c), oh], -1)


def mid_dataset(Tr, O, idx, C):
    X, Y = [], []
    for a in range(6):
        X.append(X_second(Tr, O, idx, np.full(len(idx), a)))
        Y.append(rel(C[idx][:, 6 * a:6 * a + 6]))
    return np.concatenate(X, 0), np.concatenate(Y, 0)


def b_for_all_a(torch, net, Tr, O, idx):
    return np.stack([decide(predict(torch, net, X_second(
        Tr, O, idx, np.full(len(idx), a)))) for a in range(6)], 1)


def train_selectors(torch, Otr, Ova, rows_tr, rows_va, episodes_tr, task,
                    log=print):
    """Полный протокол §11.3–11.4. -> модели, преобразования, отчёт."""
    C = Otr["C"]["full"]
    n = len(rows_tr)
    folds = folds_by_episode(episodes_tr)
    b_oof = np.zeros((n, 6), np.int64)
    fold_rep = []
    for k in range(N_FOLDS):
        tr_i = np.flatnonzero(folds != k)
        te_i = np.flatnonzero(folds == k)
        Tr_k = fit_transforms(Otr, tr_i)
        Xm, Ym = mid_dataset(Tr_k, Otr, tr_i, C)
        sc = scale_of(Ym)
        net, _h = train_net(torch, Xm, Ym / sc, MID_UPDATES,
                            SEL_SEED + 10 * k + 1)
        b_oof[te_i] = b_for_all_a(torch, net, Tr_k, Otr, te_i)
        fold_rep.append(dict(fold=k, train_rows=int(len(tr_i)),
                             test_rows=int(len(te_i)),
                             episodes=int(len(set(episodes_tr[te_i])))))
    Tr = fit_transforms(Otr, np.arange(n))
    Xm, Ym = mid_dataset(Tr, Otr, np.arange(n), C)
    sc_mid = scale_of(Ym)
    net_mid, _h = train_net(torch, Xm, Ym / sc_mid, MID_UPDATES, SEL_SEED + 1)
    Yr = root_targets(C, b_oof)
    sc_root = scale_of(Yr)
    Xr = X_first(Tr, Otr, np.arange(n))
    nv = len(rows_va)
    Cv = Ova["C"]["full"]
    bv = b_for_all_a(torch, net_mid, Tr, Ova, np.arange(nv))
    Xv = X_first(Tr, Ova, np.arange(nv))

    def composite(net):
        av = decide(predict(torch, net, Xv))
        return ko.task_mean(Cv[np.arange(nv), 6 * av + bv[np.arange(nv), av]],
                            rows_va, task)
    net_root, hist = train_net(torch, Xr, Yr / sc_root, ROOT_UPDATES,
                               SEL_SEED + 2, evaluate=composite,
                               every=ROOT_EVAL)
    log(f"  селектор корней: best шаг {hist['best_update']}, составная "
        f"val-потеря {hist['best_value']:.4f}")
    return dict(net_root=net_root, net_mid=net_mid, Tr=Tr,
                scales=dict(mid=sc_mid, root=sc_root), b_oof=b_oof,
                folds=folds, fold_report=fold_rep, root_history=hist)


def choose(torch, S, O, idx):
    """Решения A на строках idx -> (a, b, c)."""
    a = decide(predict(torch, S["net_root"], X_first(S["Tr"], O, idx)))
    b = decide(predict(torch, S["net_mid"], X_second(S["Tr"], O, idx, a)))
    return a, b, 6 * a + b


def root_targets(C, b_oof):
    """Стоимость реально исполнимого продолжения: C[i, 6a + b_oof[i, a]]
    (не оптимистичный min_b) -> относительно минимума по a."""
    n = C.shape[0]
    return rel(np.stack([C[np.arange(n), 6 * a + b_oof[:, a]]
                         for a in range(6)], 1))


def selector_gate(Cv, rows_va, task, c_A, c_const, arm8):
    nv = len(rows_va)
    full = Cv
    L_C = ko.task_mean(full[:, c_const], rows_va, task)
    L_O = ko.task_mean(full.min(1), rows_va, task)
    L_N = ko.task_mean(full.mean(1), rows_va, task)
    L_A = ko.task_mean(full[np.arange(nv), c_A], rows_va, task)
    a8_A = ko.task_mean(arm8[np.arange(nv), c_A], rows_va, task)
    a8_C = ko.task_mean(arm8[:, c_const], rows_va, task)
    cap = (L_C - L_A) / max(L_C - L_O, 1e-8)
    met = dict(L_A=L_A, L_C=L_C, L_O=L_O, L_N=L_N, capture=cap, arm8_A=a8_A,
               arm8_C=a8_C, regret=L_A - L_O,
               cut_freq=np.bincount(c_A, minlength=36).tolist())
    p = []
    if not (cap >= GATE_RULE["capture"] and L_A < L_C):
        p.append(f"capture {cap:.3f} (< 0.25) или L_A ≥ L_C")
    if not L_A <= L_N:
        p.append("L_A > L_N")
    if not a8_A <= a8_C:
        p.append("arm-8 A хуже C")
    if (c_A < 0).any() or (c_A > 35).any():
        p.append("недопустимый маршрут")
    return met, p


def bundle_state(S):
    """Сохраняемый набор: преобразования (numpy) и веса двух сетей (CPU)."""
    return dict(Tr={k: {kk: np.asarray(vv) for kk, vv in v.items()}
                    for k, v in S["Tr"].items()},
                net_root={k: v.detach().cpu().clone() for k, v in
                          S["net_root"].state_dict().items()},
                net_mid={k: v.detach().cpu().clone() for k, v in
                         S["net_mid"].state_dict().items()},
                d_in=(D_FIRST, D_SECOND), hidden=HIDDEN, p_pca=P_PCA,
                scales=dict(S.get("scales") or {}))


def bundle_hash(B):
    h = hashlib.sha1()
    for k in sorted(B["Tr"]):
        for kk in sorted(B["Tr"][k]):
            h.update(f"{k}.{kk}".encode())
            h.update(np.ascontiguousarray(B["Tr"][k][kk], np.float32)
                     .tobytes())
    for nm in ("net_root", "net_mid"):
        for k in sorted(B[nm]):
            h.update(f"{nm}.{k}".encode())
            h.update(B[nm][k].float().numpy().tobytes())
    return h.hexdigest()


def save_bundle(torch, S, path, meta):
    B = dict(bundle_state(S), meta=meta)
    B["bundle_hash"] = bundle_hash(B)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(B, path + ".tmp")
    os.replace(path + ".tmp", path)
    return B["bundle_hash"]


def nets_of(torch, B):
    out = {}
    for nm, d in (("net_root", D_FIRST), ("net_mid", D_SECOND)):
        net = make_net(torch, d, 0)
        net.load_state_dict(B[nm])
        net.eval()
        out[nm] = net
    return out


def load_bundle(torch, path, want_hash=None):
    B = torch.load(path, map_location="cpu", weights_only=False)
    h = bundle_hash(B)
    if h != B.get("bundle_hash") or (want_hash is not None and
                                     h != want_hash):
        raise SystemExit(f"{path}: отпечаток селекторов ≠ закреплённому")
    return B


def init_bundle(torch, H):
    """Набор инициализации (collector-smoke): нулевой последний слой ->
    scores = 0, решение по правилу ничьей. Не обученная политика."""
    comp = np.eye(P_PCA, H, dtype=np.float32)
    Tr = dict(P_root=dict(mean=np.zeros(H, np.float32), comp=comp),
              P_mid=dict(mean=np.zeros(H, np.float32), comp=comp.copy()),
              S1=dict(mu=np.zeros(D_FIRST, np.float32),
                      sd=np.ones(D_FIRST, np.float32)),
              S2=dict(mu=np.zeros(D_SECOND - 6, np.float32),
                      sd=np.ones(D_SECOND - 6, np.float32)))
    S = dict(Tr=Tr, net_root=make_net(torch, D_FIRST, SEL_SEED + 2),
             net_mid=make_net(torch, D_SECOND, SEL_SEED + 1))
    B = dict(bundle_state(S), meta=dict(kind="init"))
    B["bundle_hash"] = bundle_hash(B)
    return B


def selftest():
    import torch
    assert n_params(136) == 18310 and n_params(282) == 36998
    # решения: ничьи, почти равные, NaN
    s = np.array([[1.0, 0.5, 0.5, 2.0, 0.5 + 5e-7, 3.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [2.0, 1.0, 1.0 - 2e-6, 4.0, 5.0, 6.0]])
    assert decide(s).tolist() == [1, 0, 2]
    try:
        decide(np.array([[np.nan, 0, 0, 0, 0, 0]]))
        raise AssertionError("NaN принят")
    except SystemExit:
        pass
    # PCA: знак, воспроизводимость
    X = np.random.default_rng(0).normal(size=(200, 40))
    P1, P2 = pca_fit(X, 8), pca_fit(X.copy(), 8)
    assert np.array_equal(P1["comp"], P2["comp"])
    for row in P1["comp"]:
        assert row[np.argmax(np.abs(row))] > 0
    # синтетический оракул: выбор зависит от состояния
    rng = np.random.default_rng(1)
    n, H = 300, 40
    ep = np.arange(n) // 3
    task = np.array([f"t{i % 3}" for i in range(n)])
    roots = rng.normal(size=(n, 4, H)).astype(np.float32)
    mids = rng.normal(size=(n, 6, 4, H)).astype(np.float32)
    best_a = (roots[:, 0, 0] > 0).astype(int) * 5
    C = np.ones((n, 36))
    for i in range(n):
        bb = int(mids[i, best_a[i], 0, 0] > 0) * 3
        C[i, 6 * best_a[i] + bb] = 0.1
        C[i, 6 * best_a[i]:6 * best_a[i] + 6] -= 0.3
    O = dict(C=dict(full=C, arm8=C), roots=roots, mids=mids,
             iv_root=np.tile(np.array([[0, 5], [5, 10], [10, 15], [15, 20]],
                                      np.float32), (n, 1, 1)),
             iv_mid=np.zeros((n, 6, 4, 2), np.float32))
    tr = np.arange(0, 240)
    va = np.arange(240, 300)
    Otr = {k: (v[tr] if k != "C" else {kk: vv[tr] for kk, vv in v.items()})
           for k, v in O.items()}
    Ova = {k: (v[va] if k != "C" else {kk: vv[va] for kk, vv in v.items()})
           for k, v in O.items()}
    g = globals()
    keep = dict(P_PCA=g["P_PCA"], MID_UPDATES=g["MID_UPDATES"],
                ROOT_UPDATES=g["ROOT_UPDATES"])
    try:
        g.update(P_PCA=8, MID_UPDATES=300, ROOT_UPDATES=400)
        S = train_selectors(torch, Otr, Ova, list(tr), list(va), ep[tr],
                            task, log=lambda *a: None)
        assert S["b_oof"].shape == (240, 6)
        a, b, c = choose(torch, S, Ova, np.arange(len(va)))
        cc, _m = ko.c_const_of(C[tr], list(tr), task)
        met, pr = selector_gate(C[va], list(va), task, c, cc, C[va])
        assert met["L_A"] < met["L_C"] and met["capture"] > 0.25, met
        # 13. folds не делят эпизод
        f = S["folds"]
        for e in np.unique(ep[tr]):
            assert len(set(f[ep[tr] == e])) == 1
        # 14. изменение val не меняет train-преобразования и OOF
        Ova2 = dict(Ova, roots=Ova["roots"] * 2)
        S2 = train_selectors(torch, Otr, Ova2, list(tr), list(va), ep[tr],
                             task, log=lambda *a: None)
        assert np.array_equal(S2["b_oof"], S["b_oof"]) and np.array_equal(
            S2["Tr"]["P_root"]["comp"], S["Tr"]["P_root"]["comp"])
    finally:
        g.update(keep)
    # 15. цели первого селектора различают OOF-продолжение и min_b:
    # при ошибочном втором решении для a = 0 лучшим становится другой a
    Cs = np.full((1, 36), 1.0)
    Cs[0, 0] = 0.0          # a = 0 хорош только при b = 0
    Cs[0, 6:12] = 0.4       # a = 1 стабильно средний при любом b
    b_bad = np.array([[3, 0, 0, 0, 0, 0]])   # второй селектор ошибается на a = 0
    t_oof = root_targets(Cs, b_bad)
    t_min = rel(np.stack([Cs[:, 6 * a:6 * a + 6].min(1) for a in range(6)],
                         1))
    assert np.argmin(t_min[0]) == 0 and np.argmin(t_oof[0]) == 1
    # независимый от состояния шум — запас около нуля у C, A не хуже N
    try:
        folds_by_episode(np.array([0, 0, 1, 1]))
        raise AssertionError("мало эпизодов принято")
    except SystemExit:
        pass
    print("самопроверка k35_selector пройдена: число параметров 18 310 / "
          "36 998, правило ничьей и отказ на NaN, PCA со знаком и "
          "воспроизводимостью; на синтетике с зависимостью от состояния "
          "селекторы снижают потерю ниже C и capture > 0.25; folds не делят "
          "эпизоды; цели первого — OOF-продолжение; изменение val не меняет "
          "train-преобразования и OOF; мало эпизодов — отказ")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; запуск — через k35_protocol.py")
