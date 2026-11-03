#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-31: набор развилок, PCA-контекст и стандартизация признаков селектора.

Случай (data/k31/cases/s<S>/<split>/t<T>_i<I>/): ref.json/ref.npz (reference,
развилка, пул: валидность, λ, Du, руки, контекст), b_<name>.json/npz,
case.json (R0, Rc, метки Rc − R0). Набор = все обязательные случаи сида;
split задаётся только init-банком (train/val), кандидаты одного случая не
делятся.
PCA (float64, только train-checkpoint, центр — train mean, без whitening):
16 компонент для [mean x0, mean x1, mean x2] и 16 для [x0_root*, x1_mid*,
x2_leaf*]; знак компоненты — так, чтобы максимальный по модулю элемент был
положителен; ранг < 16 — информационный стоп.
Стандартизация: по train-кандидатам (валидные строки, включая noop) с
равным весом checkpoint; std < 1e−6 -> 1. Итоговые признаки — float32.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k31_candidates as kc  # noqa: E402

N_PCA = 16
STD_FLOOR = 1e-6
CLASSES = ("harm", "neutral", "rescue")


def arr_sha(*arrs):
    h = hashlib.sha1()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class InformationalStop(Exception):
    pass


def pca_fit(X):
    """X [n, d] float64 -> dict(mean, comp [16, d]); знак канонизирован."""
    X = np.asarray(X, np.float64)
    if X.shape[0] < N_PCA + 1:
        raise InformationalStop(f"строк {X.shape[0]} < {N_PCA + 1}")
    mu = X.mean(0)
    _u, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    tol = s.max() * max(X.shape) * np.finfo(np.float64).eps
    if int((s > tol).sum()) < N_PCA:
        raise InformationalStop(f"ранг {(s > tol).sum()} < {N_PCA}")
    comp = vt[:N_PCA].copy()
    for i in range(N_PCA):
        j = int(np.argmax(np.abs(comp[i])))
        if comp[i, j] < 0:
            comp[i] = -comp[i]
    return dict(mean=mu, comp=comp)


def pca_apply(p, x):
    return (np.asarray(x, np.float64) - p["mean"]) @ p["comp"].T


def label_class(d):
    return {-1: 0, 0: 1, 1: 2}[int(d)]


def load_case(cdir, *, seed, space, task, init):
    """-> dict случая; повреждённый/незавершённый — технический отказ
    (общая проверка k31_collect.case_problems: id, файлы, исходы, R0,
    метки Rc − R0, подтверждение noop)."""
    import k31_collect as kcol
    pr = kcol.case_problems(cdir, seed=seed, space=space, task=task,
                            init=init)
    if pr:
        raise SystemExit(f"{cdir}: {pr[:3]}")
    case = json.load(open(os.path.join(cdir, "case.json"), encoding="utf-8"))
    rj = json.load(open(os.path.join(cdir, "ref.json"), encoding="utf-8"))
    z = dict(np.load(os.path.join(cdir, "ref.npz")))
    return dict(case=case, ref=rj, z=z)


def raw_rows(c):
    """Случай -> (строки кандидатов с признаками без контекста PCA). Для
    каждой разрешённой строки: индекс, метка (для non-noop), Du/λ/узел."""
    z = c["z"]
    allowed = np.asarray(z["allowed"], bool)
    rows = []
    for i in range(7):
        if not allowed[i]:
            continue
        name = kc.ROWS[i]
        lab = None if i == 0 else c["case"]["labels"][name]
        rows.append(dict(i=i, name=name, label=lab))
    return rows


def case_features(c, pca_mean, pca_path):
    """-> X [7,149] сырые признаки (недопустимые строки — нули) и маска."""
    z = c["z"]
    ctx = np.concatenate([pca_apply(pca_mean, z["ctx_mean"]),
                          pca_apply(pca_path, z["ctx_path"])])
    allowed = np.asarray(z["allowed"], bool)
    P = dict(allowed=allowed, rows=[
        dict(Du=z["Du"][i], lam=float(z["lam"][i])) for i in range(7)])
    info = {lv: dict(center=float(z["node_center"][j]),
                     width=float(z["node_width"][j]),
                     mass=float(z["node_mass"][j]))
            for j, lv in enumerate(("root", "mid", "leaf"))}
    X = kc.features(ctx, z["s11n"], int(c["ref"]["fork"]["tau"]),
                    z["u0"], P, info)
    return X, allowed


def standardizer(Xs, masks):
    """Xs — список [7,149] train-checkpoint, masks — разрешённые строки;
    равный вес checkpoint."""
    w_sum = 0.0
    m = np.zeros(kc.N_FEAT)
    for X, a in zip(Xs, masks):
        r = np.asarray(X, np.float64)[a]
        m += r.mean(0)
        w_sum += 1.0
    m /= w_sum
    v = np.zeros(kc.N_FEAT)
    for X, a in zip(Xs, masks):
        r = np.asarray(X, np.float64)[a]
        v += ((r - m) ** 2).mean(0)
    sd = np.sqrt(v / w_sum)
    sd = np.where(sd < STD_FLOOR, 1.0, sd)
    return dict(mean=m, std=sd)


def standardize(st, X):
    return ((np.asarray(X, np.float64) - st["mean"]) / st["std"]).astype(
        np.float32)


def build_dataset(cases_train, cases_val):
    """-> dict(pca_mean, pca_path, std, train/val: X [n,7,149] float32,
    allowed [n,7], y [n,7] (-1 — нет метки), keys)."""
    A = np.stack([np.asarray(c["z"]["ctx_mean"], np.float64)
                  for c in cases_train])
    B = np.stack([np.asarray(c["z"]["ctx_path"], np.float64)
                  for c in cases_train])
    pm, pp = pca_fit(A), pca_fit(B)

    def mats(cs):
        X, al, y, keys = [], [], [], []
        for c in cs:
            x, a = case_features(c, pm, pp)
            yy = np.full(7, -1, np.int64)
            for r in raw_rows(c):
                if r["label"] is not None:
                    yy[r["i"]] = label_class(r["label"])
            X.append(x)
            al.append(a)
            y.append(yy)
            keys.append((c["case"]["task"], c["case"]["init"]))
        return X, al, y, keys
    Xt, at, yt, kt = mats(cases_train)
    st = standardizer(Xt, at)
    Xv, av, yv, kv = mats(cases_val)
    out = dict(pca_mean=pm, pca_path=pp, std=st)
    for nm, X, al, y, keys in (("train", Xt, at, yt, kt),
                               ("val", Xv, av, yv, kv)):
        out[nm] = dict(X=np.stack([standardize(st, x) for x in X])
                       if X else np.zeros((0, 7, kc.N_FEAT), np.float32),
                       allowed=np.array(al, bool).reshape(-1, 7),
                       y=np.array(y, np.int64).reshape(-1, 7), keys=keys)
    out["sha"] = arr_sha(pm["mean"], pm["comp"], pp["mean"], pp["comp"],
                         st["mean"], st["std"], out["train"]["X"],
                         out["train"]["y"], out["val"]["X"], out["val"]["y"])
    return out


def selftest():
    rng = np.random.default_rng(0)
    # PCA: ранг, знак, только train
    X = rng.normal(size=(40, 60)) @ rng.normal(size=(60, 60))
    p = pca_fit(X)
    assert p["comp"].shape == (N_PCA, 60)
    assert np.allclose(p["comp"] @ p["comp"].T, np.eye(N_PCA), atol=1e-8)
    for i in range(N_PCA):
        assert p["comp"][i, np.argmax(np.abs(p["comp"][i]))] > 0
    p2 = pca_fit(X.copy())
    assert np.array_equal(p["comp"], p2["comp"])
    low = rng.normal(size=(40, 5)) @ rng.normal(size=(5, 60))
    try:
        pca_fit(low)
        raise AssertionError("ранг < 16 принят")
    except InformationalStop:
        pass
    # стандартизация: равный вес checkpoint, нижняя граница std
    Xs = [np.full((7, kc.N_FEAT), float(i), np.float32) for i in range(3)]
    ms = [np.array([True] * 7), np.array([True] + [False] * 6),
          np.array([True, True] + [False] * 5)]
    st = standardizer(Xs, ms)
    assert np.allclose(st["mean"], 1.0) and np.allclose(
        st["std"], np.sqrt(2 / 3))
    st1 = standardizer([np.zeros((7, kc.N_FEAT), np.float32)], ms[:1])
    assert (st1["std"] == 1.0).all()
    # набор из игрушечных случаев: метки, маски, нет утечки val в PCA/std
    H = 24

    def fake(task, init, labels, allowed, seed):
        g = np.random.default_rng(seed)
        z = dict(allowed=np.array(allowed), ctx_mean=g.normal(size=3 * H),
                 ctx_path=g.normal(size=3 * H), s11n=g.normal(size=11),
                 u0=g.normal(size=(20, 6)),
                 Du=g.normal(size=(7, 20, 6)) * 0.01,
                 lam=np.abs(g.normal(size=7)) + 0.1,
                 node_center=np.array([10.0, 5.0, 2.5]),
                 node_width=np.array([20.0, 10.0, 5.0]),
                 node_mass=np.array([1.0, 0.5, 0.25]))
        return dict(case=dict(task=task, init=init, labels=labels),
                    ref=dict(fork=dict(tau=16)), z=z)
    al = [True, True, True, True, True, True, True]
    lab = dict(root=1, mid=0, leaf=-1, flat0=0, flat1=0, flat2=1)
    tr = [fake(0, i, lab, al, i) for i in range(20)]
    va = [fake(0, 20 + i, lab, al, 100 + i) for i in range(3)]
    D = build_dataset(tr, va)
    assert D["train"]["X"].shape == (20, 7, kc.N_FEAT)
    assert D["train"]["y"][0].tolist() == [-1, 2, 1, 0, 1, 1, 2]
    va2 = [fake(0, 20 + i, lab, al, 999 + i) for i in range(3)]
    D2 = build_dataset(tr, va2)
    assert np.array_equal(D["pca_mean"]["comp"], D2["pca_mean"]["comp"]) \
        and np.array_equal(D["std"]["std"], D2["std"]["std"])
    assert not np.array_equal(D["val"]["X"], D2["val"]["X"])
    assert D["sha"] != D2["sha"]
    print("самопроверка k31_data пройдена: PCA (ортонормированность, знак, "
          "детерминизм, ранг < 16 — информационный стоп), стандартизация с "
          "равным весом checkpoint и нижней границей std, метки "
          "harm/neutral/rescue, PCA/std не видят validation")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
