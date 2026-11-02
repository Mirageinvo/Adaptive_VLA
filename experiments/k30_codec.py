#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-30: грубые планы корней и словарь прототипов (K = 128), CPU, float64.

План корня r (r = 0…3, участок I_r = {5r, …, 5r+4} чанка из 20 команд):
  u[t,c] = clip((a[t,c] − mid[c]) / half[c], −1, 1)  (только рука, единицы
  кодека K-21; mid = (U+L)/2, half = (U−L)/2);
  n_r = Σ_{t∈I_r} m[t];  root_valid[r] = n_r > 0;
  b[r,c] = Σ m[t]·u[t,c] / n_r;  при n_r = 0 — нули и метка −1.
Кластеризация: точки (строка, корень) с root_valid, вес w_i / n_valid_i
(w_i — экспертный вес «задачи → эпизоды → строки»), нормированы; μ и std —
взвешенно по train; s = max(std, 0.05); x = (b − μ)/s. Weighted k-means++
(один запуск, np.random.default_rng(20261102), D²-выбор по весу), Lloyd ≤ 30
итераций, остановка при относительном улучшении ≤ 1e−6 после ≥ 2 итераций;
пустой кластер — точка с максимальной взвешенной ошибкой (ничья — меньший
(row_id, root_id)). Центры -> b-координаты, лексикографическая сортировка;
float32 P — то, что использует модель. Метки — по float32 P, float64 на CPU,
d = Σ_c ((b_c − P_kc)/s_c)², argmin с меньшим ID.
Допуск к обучению VLA: взвешенный val MSE ближайшего прототипа ≤ 0.5 ·
MSE постоянного прототипа (μ, т. е. 0 в нормированных координатах).
"""
import hashlib
import math
import sys

import numpy as np

K = 128
T = 20
SEG = 5
R = 4
CH = 6
S_FLOOR = 0.05
SEED = 20261102
MAX_IT = 30
REL_TOL = 1e-6
BLOCK = 8192
GATE = 0.5
CONFIG = dict(K=K, T=T, seg=SEG, roots=R, channels=CH, s_floor=S_FLOOR,
              seed=SEED, max_iter=MAX_IT, rel_tol=REL_TOL, block=BLOCK,
              gate_val_mse_ratio=GATE,
              point_weight="w_i / n_valid_roots_i, нормировано",
              init="weighted k-means++ (D² по весу), один запуск",
              sort="лексикографически по 6 координатам b",
              labels="float32 P, float64 CPU, argmin меньший ID")


def mid_half(L, U):
    L = np.asarray(L, np.float64)[:CH]
    U = np.asarray(U, np.float64)[:CH]
    half = (U - L) / 2
    if not (half > 0).all():
        raise SystemExit("half ≤ 0 в каком-то канале")
    return (U + L) / 2, half


def plan_targets(arm, mask, L, U):
    """arm [N, 20, ≥6] (единицы кодека), mask [N, 20] -> (b [N,4,6] float64,
    root_valid [N,4] bool, доля clipping u)."""
    mid, half = mid_half(L, U)
    a = np.asarray(arm, np.float64)[..., :CH]
    m = np.asarray(mask, np.float64)
    if a.shape[1] != T or m.shape != a.shape[:2]:
        raise SystemExit(f"формы целей {a.shape} / маски {m.shape}")
    raw = (a - mid) / half
    u = np.clip(raw, -1, 1)
    clipped = (raw != u) & (m[..., None] > 0)
    clip_frac = float(clipped.sum() / max(1.0, (m.sum() * CH)))
    uu = u.reshape(len(u), R, SEG, CH)
    mm = m.reshape(len(m), R, SEG)
    n = mm.sum(-1)
    rv = n > 0
    b = np.where(rv[..., None], (uu * mm[..., None]).sum(2) /
                 np.where(rv, n, 1.0)[..., None], 0.0)
    has = m.sum(1) > 0
    if (has & ~rv.any(1)).any():
        raise SystemExit("строка с командами без действительного корня")
    return b, rv, clip_frac


def point_weights(w_rows, rv):
    w = np.asarray(w_rows, np.float64)
    n = rv.sum(1)
    pw = np.where(rv, (w / np.maximum(n, 1))[:, None], 0.0)
    tot = pw.sum()
    if not tot > 0:
        raise SystemExit("нулевая сумма весов точек")
    return pw / tot


def wstats(b, pw):
    """Взвешенные μ, std по каналам (pw [N,4], b [N,4,6])."""
    wsum = pw.sum()
    mu = (b * pw[..., None]).sum((0, 1)) / wsum
    var = (((b - mu) ** 2) * pw[..., None]).sum((0, 1)) / wsum
    return mu, np.sqrt(var)


def _flat(b, rv, pw, row_ids):
    idx = np.nonzero(rv)
    X = b[idx]                                 # [M, 6]
    return X, pw[idx], np.stack([np.asarray(row_ids)[idx[0]], idx[1]], 1)


def _assign(X, C):
    """-> (метки, квадраты расстояний) блоками; ничья — меньший ID."""
    lab = np.empty(len(X), np.int64)
    d2 = np.empty(len(X))
    cc = (C * C).sum(1)
    for s in range(0, len(X), BLOCK):
        x = X[s:s + BLOCK]
        d = (x * x).sum(1)[:, None] - 2 * x @ C.T + cc[None]
        d = np.maximum(d, 0.0)
        lab[s:s + BLOCK] = np.argmin(d, 1)
        d2[s:s + BLOCK] = d[np.arange(len(x)), lab[s:s + BLOCK]]
    return lab, d2


def _exact_d2(X, C, lab):
    return ((X - C[lab]) ** 2).sum(1)


def kmeanspp(X, w, k, rng):
    n = len(X)
    p = w / w.sum()
    first = int(rng.choice(n, p=p))
    C = [X[first]]
    d2 = ((X - X[first]) ** 2).sum(1)
    for _ in range(1, k):
        q = w * d2
        if not q.sum() > 0:
            raise SystemExit("k-means++: все точки совпали с центрами")
        j = int(rng.choice(n, p=q / q.sum()))
        C.append(X[j])
        d2 = np.minimum(d2, ((X - X[j]) ** 2).sum(1))
    return np.array(C)


def lloyd(X, w, C, keys):
    """-> центры, история objective, число итераций, пересозданные."""
    hist, reseed = [], 0
    order = np.lexsort((keys[:, 1], keys[:, 0]))      # ничья: (row, root)
    it = 0
    for it in range(1, MAX_IT + 1):
        lab, _ = _assign(X, C)
        d2 = _exact_d2(X, C, lab)
        obj = float((w * d2).sum())
        hist.append(obj)
        C = C.copy()
        cnt = np.bincount(lab, weights=w, minlength=len(C))
        sums = np.zeros_like(C)
        np.add.at(sums, lab, X * w[:, None])
        nz = cnt > 0
        C[nz] = sums[nz] / cnt[nz, None]
        for kk in np.nonzero(~nz)[0]:               # пустой кластер
            err = w * d2
            best = order[np.argmax(err[order])]      # первый максимум
            C[kk] = X[best]
            d2[best] = 0.0
            reseed += 1
        if it >= 2 and (hist[-2] - hist[-1]) <= REL_TOL * max(hist[-2],
                                                               1e-300):
            break
    lab, _ = _assign(X, C)
    hist.append(float((w * _exact_d2(X, C, lab)).sum()))
    return C, hist, it, reseed


def fit(b, rv, w_rows, row_ids):
    """Обучение словаря на train -> dict(P64, P32, mu, s, ...)."""
    pw = point_weights(w_rows, rv)
    mu, std = wstats(b, pw)
    s = np.maximum(std, S_FLOOR)
    Xb, wp, keys = _flat(b, rv, pw, row_ids)
    if len(np.unique(Xb, axis=0)) < K:
        raise SystemExit(f"уникальных train-точек {len(np.unique(Xb, axis=0))}"
                         f" < K = {K}")
    X = (Xb - mu) / s
    rng = np.random.default_rng(SEED)
    C0 = kmeanspp(X, wp, K, rng)
    C, hist, iters, reseed = lloyd(X, wp, C0, keys)
    P64 = C * s + mu
    order = np.lexsort(tuple(P64[:, c] for c in range(CH - 1, -1, -1)))
    P64 = P64[order]
    return dict(P64=P64, P32=P64.astype(np.float32), mu=mu, std=std, s=s,
                objective=hist, iters=iters, reseeded=reseed,
                n_points=int(len(X)))


def labels(b, rv, P32, s):
    """Метки по float32 P (float64 CPU); −1 для недействительных корней."""
    P = np.asarray(P32, np.float32).astype(np.float64)
    s = np.asarray(s, np.float64)
    out = np.full(rv.shape, -1, np.int64)
    idx = np.nonzero(rv)
    X = b[idx]
    lab = np.empty(len(X), np.int64)
    for st in range(0, len(X), BLOCK):
        x = X[st:st + BLOCK]
        d = (((x[:, None, :] - P[None]) / s) ** 2).sum(-1)
        lab[st:st + BLOCK] = np.argmin(d, 1)            # первый — меньший ID
    out[idx] = lab
    return out


def metrics(b, rv, ids, w_rows, P32, mu, s):
    """Взвешенный MSE (нормированные координаты, среднее по каналам)
    ближайшего прототипа и постоянного (μ); MAE по каналам; использование."""
    pw = point_weights(w_rows, rv)
    idx = np.nonzero(rv)
    x = (b[idx] - mu) / s
    pr = (np.asarray(P32, np.float64)[ids[idx]] - mu) / s
    w = pw[idx]
    mse = float((w * ((x - pr) ** 2).mean(1)).sum())
    mse0 = float((w * (x ** 2).mean(1)).sum())
    mae = (w[:, None] * np.abs(b[idx] - np.asarray(P32, np.float64)[
        ids[idx]])).sum(0).tolist()
    use = np.bincount(ids[idx], weights=w, minlength=K)
    used = int((use > 0).sum())
    q = use[use > 0] / use.sum()
    perp = float(math.exp(-(q * np.log(q)).sum()))
    return dict(mse=mse, mse_const=mse0,
                ratio=mse / mse0 if mse0 > 1e-8 else None,
                mae_b=mae, codes_used=used, perplexity=perp,
                n_points=int(len(x)))


def sha_arr(*arrs):
    h = hashlib.sha1()
    for a in arrs:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def codec_problems(z):
    """Проверка артефакта словаря (формы, dtype, конечность, SHA)."""
    p = []
    need = ("P32", "P64", "mu", "s", "sha")
    if any(k not in z for k in need):
        return [f"в словаре нет {[k for k in need if k not in z]}"]
    if z["P32"].shape != (K, CH) or z["P32"].dtype != np.float32:
        p.append(f"P формы {z['P32'].shape} {z['P32'].dtype}")
    if not (np.isfinite(z["P32"]).all() and np.isfinite(z["s"]).all()):
        p.append("словарь не конечен")
    if np.abs(z["P32"]).max() > 1 + 1e-6:
        p.append("прототипы вне [−1, 1]")
    if str(z["sha"]) != sha_arr(z["P32"], z["mu"], z["s"]):
        p.append("SHA словаря не совпадает")
    return p


def selftest():
    L, U = -np.ones(6) * 2.0, np.ones(6) * 4.0
    mid, half = mid_half(L, U)
    rng = np.random.default_rng(0)
    # средние участков и хвосты длины 1/5/6/19
    N = 6
    arm = rng.uniform(-3, 5, (N, T, 7))
    m = np.ones((N, T))
    for i, ln in enumerate((1, 5, 6, 19)):
        m[i + 1, ln:] = 0
    b, rv, cf = plan_targets(arm, m, L, U)
    u = np.clip((arm[..., :6] - mid) / half, -1, 1)
    assert np.allclose(b[0, 2], u[0, 10:15].mean(0))
    assert rv[1].tolist() == [True, False, False, False]
    assert rv[2].tolist() == [True, False, False, False]
    assert rv[3].tolist() == [True, True, False, False]
    assert np.allclose(b[3, 1], u[3, 5])
    assert rv[4].tolist() == [True, True, True, True]
    assert np.allclose(b[4, 3], u[4, 15:19].mean(0))
    assert (b[1, 1:] == 0).all() and cf > 0
    # изменение недействительного хвоста не меняет цели
    arm2 = arm.copy()
    arm2[1, 1:] += 7
    b2, rv2, _ = plan_targets(arm2, m, L, U)
    assert np.array_equal(b2, b) and np.array_equal(rv2, rv)
    # словарь: воспроизводимость, сортировка, метки, ничья
    M = 3000
    arm = np.concatenate([rng.normal(0, 0.3, (M, T, 6)) +
                          rng.integers(-1, 2, (M, 1, 6)) * 1.0,
                          np.zeros((M, T, 1))], -1)
    mk = np.ones((M, T))
    mk[::7, 13:] = 0
    b, rv, _ = plan_targets(arm, mk, L, U)
    w = rng.uniform(0.5, 2, M)
    ids_row = np.arange(M) * 3 + 11
    f1 = fit(b, rv, w, ids_row)
    f2 = fit(b, rv, w, ids_row)
    assert np.array_equal(f1["P64"], f2["P64"])
    P = f1["P64"]
    assert all(tuple(P[i]) <= tuple(P[i + 1]) for i in range(K - 1))
    assert np.abs(f1["P32"]).max() <= 1
    lab = labels(b, rv, f1["P32"], f1["s"])
    assert (lab[~rv] == -1).all() and lab[rv].min() >= 0
    # ничья — меньший ID
    P2 = np.zeros((K, CH), np.float32)
    P2[5] = 0.1
    P2[7] = 0.1
    bb = np.zeros((1, R, CH))
    bb[0, :] = 0.1
    l2 = labels(bb, np.ones((1, R), bool), P2, np.ones(CH))
    assert (l2 == 5).all()
    # val не влияет на словарь (fit зависит только от train)
    met = metrics(b, rv, lab, w, f1["P32"], f1["mu"], f1["s"])
    assert met["ratio"] < GATE and met["codes_used"] > 50
    # порча артефакта — отказ
    z = dict(P32=f1["P32"], P64=f1["P64"], mu=f1["mu"], s=f1["s"],
             sha=sha_arr(f1["P32"], f1["mu"], f1["s"]))
    assert codec_problems(z) == []
    zb = dict(z, P32=z["P32"].copy())
    zb["P32"][0, 0] += 1e-3
    assert codec_problems(zb)
    assert codec_problems(dict(z, P32=z["P32"][:64]))
    print(f"самопроверка k30_codec пройдена: средние участков, хвосты "
          f"1/5/6/19, пустые корни −1, недействительный хвост не влияет, "
          f"словарь воспроизводим и отсортирован, метки по float32 P, "
          f"ничья — меньший ID, порча артефакта — отказ; игрушка: MSE/конст "
          f"{met['ratio']:.3f}, кодов {met['codes_used']}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; подготовка — k30_data.py")
