#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-13a: базис траекторного многообразия для HiCoRA-T.

ЧТО СТРОИТСЯ. Одно линейное подпространство ранга 64 в пространстве ЦЕЛОГО
остатка чанка размерности 16*512 = 8192. В исходной HiCoRA базис строится в
пространстве ОДНОЙ позиции (512), и каждая из 16 позиций получает свои
коэффициенты независимо — всего 16*32 = 512 степеней свободы. Здесь их 64 на
весь чанк, и они связаны: один коэффициент меняет траекторию согласованно.

РАЗНИЦА С K-11a — РОВНО В ОДНОЙ СТРОКЕ. Там остаток разворачивается в
`(-1, 512)`, то есть по строке на позицию; здесь — в `(-1, 8192)`, по строке на
чанк. Всё остальное (что считать остатком, как строить грамиан, как выбирать
предел) переиспользуется без изменений, чтобы разница между HiCoRA и HiCoRA-T
была архитектурной, а не следствием другого рецепта.

ОТ ЧЕГО ОТСЧИТЫВАЕТСЯ ОСТАТОК. От ПРЕДСКАЗАННОГО черновика q0hat слоя 12, а не
от истинного кода нулевого уровня:

    R = (E0[k0] + E1[k1] + E2[k2]) - E0[q0hat].

Поправка обязана чинить и ошибку квантования, и ошибку предсказания черновика —
именно так устроена HiCoRA, и менять это здесь нельзя, иначе цель обучения
станет другой.

БАЗИС БЕЗ ЦЕНТРИРОВАНИЯ. Подпространство проходит через ноль, поэтому
`u = 0` даёт `dZ = 0` ТОЧНО. На этом держится тождество HiCoRA-T с coarse24 при
нулевой инициализации последнего слоя; центрированный PCA добавил бы
постоянный сдвиг, и тождество исчезло бы.

ГРАМИАН ПОТОКОМ. Остаток целиком — это 150000 x 8192 x 4 байта, около 5 ГБ;
он не хранится. Накапливается только матрица 8192 x 8192, а собственные
векторы грамиана совпадают с правыми сингулярными векторами остатка.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

N_POS, D_LAT, RANK = 16, 512, 64
RHO_PCT, RHO_COVER = 95.0, 0.95


def residual_rows(E, k_true, q0hat, idx, batch=4096):
    """Остаток чанка построчно: по строке на наблюдение, размерность 8192.

    Генератор, а не массив: полный остаток не помещается в памяти, а для
    грамиана он и не нужен целиком.
    """
    for a in range(0, len(idx), batch):
        s = idx[a:a + batch]
        z_full = E[0][k_true[s, 0, :].astype(np.int64)]
        for lvl in range(1, E.shape[0]):
            z_full = z_full + E[lvl][k_true[s, lvl, :].astype(np.int64)]
        z0 = E[0][q0hat[s].astype(np.int64)]
        yield (z_full - z0).reshape(len(s), N_POS * D_LAT)


def gram_stream(E, k_true, q0hat, idx, batch=4096, log=print):
    """Грамиан 8192 x 8192 по train-строкам. Остаток не хранится."""
    d = N_POS * D_LAT
    G = np.zeros((d, d), np.float64)
    n = 0
    for r in residual_rows(E, k_true, q0hat, idx, batch):
        rd = r.astype(np.float64)
        G += rd.T @ rd
        n += len(rd)
        if n % (batch * 8) == 0:
            log(f"    грамиан: {n} строк из {len(idx)}")
    if n != len(idx):
        raise SystemExit(f"в грамиан вошло {n} строк вместо {len(idx)}")
    return G, n


def basis_from_gram(G, rank):
    """Ортонормированный базис из грамиана. БЕЗ ЦЕНТРИРОВАНИЯ.

    Собственные векторы симметричного грамиана — это правые сингулярные
    векторы самого остатка, поэтому подпространство то же, что дал бы
    uncentered SVD, но без хранения остатка.
    """
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    if float(w[0]) <= 0:
        raise SystemExit("грамиан вырожден: остаток тождественно нулевой")
    neg = float(w[w < 0].sum())
    if neg < -1e-6 * float(w[0]):
        raise SystemExit(f"у грамиана заметно отрицательные собственные "
                         f"значения (сумма {neg:.3e}): накопление испорчено")
    B = V[:, :int(rank)].T.copy()          # (rank, 8192)
    return np.ascontiguousarray(B, np.float32), np.maximum(w, 0.0)


def explained(w, rank):
    tot = float(w.sum())
    return float(w[:int(rank)].sum() / tot) if tot > 0 else 0.0


def coeffs_stream(E, k_true, q0hat, idx, B, batch=4096):
    """Целевые коэффициенты c* = B r для заданных строк."""
    out = []
    for r in residual_rows(E, k_true, q0hat, idx, batch):
        out.append((r.astype(np.float32) @ B.T))
    return np.concatenate(out) if out else np.zeros((0, B.shape[0]), np.float32)


def rho_joint(coef, base_q=RHO_PCT, cover=RHO_COVER):
    """Предел по СОВМЕСТНОМУ покрытию — тот же рецепт, что в K-11a.

    Покоординатный процентиль оставляет вне предела долю (1-q) по КАЖДОЙ
    координате, и при 64 координатах вне бокса оказывается большинство
    наблюдений. Здесь: покоординатный масштаб как базовый процентиль, затем
    общий множитель как процентиль величины max_i |a_i| / s_i — тогда внутрь
    бокса целиком попадает ровно `cover` наблюдений при любом ранге.
    """
    a = np.abs(np.asarray(coef, np.float64))
    s_ = np.percentile(a, base_q, axis=0)
    if not np.isfinite(s_).all() or (s_ <= 0).any():
        raise SystemExit("базовый масштаб вырожден хотя бы по одной "
                         "координате: остаток не покрывает базис")
    m = (a / s_[None]).max(axis=1)
    alpha = float(np.percentile(m, 100.0 * cover))
    return (alpha * s_).astype(np.float32), alpha


def apply_correction(B, rho, u):
    """dZ = reshape(B^T (rho * tanh(u))). Форма (..., 16, 512)."""
    c = rho * np.tanh(u)
    dz = c @ B
    return dz.reshape(*u.shape[:-1], N_POS, D_LAT)


# ------------------------------ самопроверка ------------------------------

def selftest():
    rng = np.random.default_rng(0)
    d = N_POS * D_LAT

    # --- 1. БАЗИС ИЗ ГРАМИАНА == UNCENTERED SVD ---------------------------
    # Проверяется на малой задаче, где прямой SVD выполним: подпространства
    # обязаны совпасть, иначе поточное накопление считает не то.
    n, dd, r = 200, 30, 5
    X = rng.normal(size=(n, dd)) @ rng.normal(size=(dd, dd))
    G = X.T @ X
    B, w = basis_from_gram(G, r)
    _U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # знаки собственных векторов произвольны — сравниваем проекторы
    P1 = B.T.astype(np.float64) @ B.astype(np.float64)
    P2 = Vt[:r].T @ Vt[:r]
    assert np.abs(P1 - P2).max() < 1e-6, np.abs(P1 - P2).max()
    assert np.allclose(w[:r], S[:r] ** 2, rtol=1e-6), (w[:r], S[:r] ** 2)
    # ортонормированность
    assert np.abs(B @ B.T - np.eye(r)).max() < 1e-5

    # --- 2. ЦЕНТРИРОВАНИЕ БЫЛО БЫ ДРУГИМ ПОДПРОСТРАНСТВОМ -----------------
    # При ненулевом среднем centered PCA даёт иной базис; наш проходит через
    # ноль, и это условие тождества u=0 -> dZ=0
    Xs = X + 5.0
    Bc, _ = basis_from_gram((Xs - Xs.mean(0)).T @ (Xs - Xs.mean(0)), r)
    Bu, _ = basis_from_gram(Xs.T @ Xs, r)
    assert np.abs(Bc.T @ Bc - Bu.T @ Bu).max() > 1e-3, \
        "центрированный и нецентрированный базисы совпали: тест пуст"

    # --- 3. ОСТАТОК СЧИТАЕТСЯ ОТ ПРЕДСКАЗАННОГО ЧЕРНОВИКА -----------------
    V_, N_ = 7, 5
    E = rng.normal(size=(3, V_, D_LAT)).astype(np.float32)
    k_true = rng.integers(0, V_, size=(N_, 3, N_POS)).astype(np.int16)
    q0hat = rng.integers(0, V_, size=(N_, N_POS)).astype(np.int16)
    rows = np.concatenate(list(residual_rows(E, k_true, q0hat,
                                             np.arange(N_), batch=2)))
    assert rows.shape == (N_, d), rows.shape
    exp0 = (E[0][k_true[0, 0]] + E[1][k_true[0, 1]] + E[2][k_true[0, 2]]
            - E[0][q0hat[0]]).reshape(-1)
    assert np.abs(rows[0] - exp0).max() < 1e-5
    # если бы отсчитывали от ИСТИННОГО нулевого уровня, вышло бы другое
    alt = (E[1][k_true[0, 1]] + E[2][k_true[0, 2]]).reshape(-1)
    assert np.abs(rows[0] - alt).max() > 1e-6, \
        "остаток совпал с чисто квантовым: черновик не учтён"

    # --- 4. ТОЖДЕСТВО u=0 -> dZ=0 ТОЧНОЕ ----------------------------------
    Bt = np.linalg.qr(rng.normal(size=(d, 8)))[0].T.astype(np.float32)
    rho = np.abs(rng.normal(size=8)).astype(np.float32) + 0.1
    dz = apply_correction(Bt, rho, np.zeros((3, 8), np.float32))
    assert dz.shape == (3, N_POS, D_LAT), dz.shape
    assert np.abs(dz).max() == 0.0, "нулевые коэффициенты дали ненулевую поправку"
    # а ненулевые — дают
    assert np.abs(apply_correction(Bt, rho, np.ones((1, 8), np.float32))).max() > 0

    # --- 5. ПРЕДЕЛ ПОКРЫВАЕТ ЗАЯВЛЕННУЮ ДОЛЮ НАБЛЮДЕНИЙ -------------------
    coef = rng.normal(size=(4000, 16)).astype(np.float32)
    rho_v, alpha = rho_joint(coef, RHO_PCT, 0.95)
    inside = (np.abs(coef) <= rho_v[None]).all(axis=1).mean()
    assert 0.93 <= inside <= 0.97, inside
    assert alpha > 0 and np.all(rho_v > 0)
    # покоординатный процентиль покрыл бы заметно меньше — ради этого и
    # введена совместная схема
    s_only = np.percentile(np.abs(coef), RHO_PCT, axis=0)
    assert (np.abs(coef) <= s_only[None]).all(axis=1).mean() < inside - 0.1

    # --- 6. КОЭФФИЦИЕНТЫ ВОССТАНАВЛИВАЮТ ПРОЕКЦИЮ ОСТАТКА -----------------
    G2, _ = gram_stream(E, k_true, q0hat, np.arange(N_), batch=2,
                        log=lambda *_: None)
    B2, _w2 = basis_from_gram(G2, 3)
    c2 = coeffs_stream(E, k_true, q0hat, np.arange(N_), B2, batch=2)
    assert c2.shape == (N_, 3), c2.shape
    proj = c2 @ B2
    resid = np.concatenate(list(residual_rows(E, k_true, q0hat,
                                              np.arange(N_), batch=2)))
    # проекция ортогональна невязке — признак правильного базиса
    assert np.abs((resid - proj) @ B2.T).max() < 1e-3

    # --- 7. ГРАМИАН ТОЛЬКО ПО ЗАДАННЫМ СТРОКАМ ----------------------------
    Ga, na = gram_stream(E, k_true, q0hat, np.arange(3), batch=2,
                         log=lambda *_: None)
    Gb, _ = gram_stream(E, k_true, q0hat, np.arange(N_), batch=2,
                        log=lambda *_: None)
    assert na == 3 and np.abs(Ga - Gb).max() > 1e-3, \
        "грамиан не зависит от набора строк: split игнорируется"
    print("самопроверка k13a_build_trajectory_basis пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--rho-pct", type=float, default=RHO_PCT)
    ap.add_argument("--rho-cover", type=float, default=RHO_COVER)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", default="data/k13a_traj_basis")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import k12b_protocol as kb

    need = [f"{a.cache}.ktrue.npy", f"{a.cache}.q0hat.npy",
            f"{a.cache}.codebooks.npy", f"{a.cache}.split.npy",
            f"{a.cache}.meta.json"]
    miss = [f for f in need if not os.path.exists(f)]
    if miss:
        raise SystemExit(f"нет файлов кэша: {miss}")

    k_true = np.load(f"{a.cache}.ktrue.npy", mmap_mode="r")
    q0hat = np.load(f"{a.cache}.q0hat.npy", mmap_mode="r")
    E = np.load(f"{a.cache}.codebooks.npy")
    split = np.load(f"{a.cache}.split.npy")
    cmeta = json.load(open(f"{a.cache}.meta.json"))

    if k_true.shape[1:] != (E.shape[0], N_POS):
        raise SystemExit(f"коды формы {k_true.shape}, ожидалось "
                         f"(N, {E.shape[0]}, {N_POS})")
    if E.shape[-1] != D_LAT:
        raise SystemExit(f"латент {E.shape[-1]}, ожидалось {D_LAT}")
    if q0hat.shape != (k_true.shape[0], N_POS):
        raise SystemExit(f"черновик формы {q0hat.shape}")

    tr = np.flatnonzero(split == 0)
    va = np.flatnonzero(split == 1)
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit(f"пустой split: train {len(tr)}, val {len(va)}")
    print(f"  кэш {a.cache}: {k_true.shape[0]} наблюдений, уровней "
          f"{E.shape[0]}, словарь {E.shape[1]}")
    print(f"  train {len(tr)}, val {len(va)}; базис строится ТОЛЬКО по train")

    G, n = gram_stream(E, k_true, q0hat, tr, a.batch)
    B, w = basis_from_gram(G, a.rank)
    ev = explained(w, a.rank)
    print(f"\n  базис ранга {a.rank} из {N_POS * D_LAT} измерений; "
          f"объяснённая доля энергии остатка {100 * ev:.2f}%")
    print(f"    ортонормированность: max|B B^T - I| = "
          f"{np.abs(B @ B.T - np.eye(a.rank)).max():.2e}")

    c_tr = coeffs_stream(E, k_true, q0hat, tr, B, a.batch)
    rho, alpha = rho_joint(c_tr, a.rho_pct, a.rho_cover)
    inside = float((np.abs(c_tr) <= rho[None]).all(axis=1).mean())
    print(f"\n  предел rho: процентиль {a.rho_pct} по координате, общий "
          f"множитель alpha={alpha:.3f}")
    print(f"    внутрь бокса целиком попадает {100 * inside:.2f}% train-"
          f"наблюдений (цель {100 * a.rho_cover:.0f}%)")
    print(f"    ||rho|| = {float(np.linalg.norm(rho)):.4f}")

    c_va = coeffs_stream(E, k_true, q0hat, va, B, a.batch)
    inside_va = float((np.abs(c_va) <= rho[None]).all(axis=1).mean())
    print(f"    на val внутрь попадает {100 * inside_va:.2f}% — предел "
          f"переносится, а не подогнан")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.save(f"{a.out}.basis.npy", B)
    np.save(f"{a.out}.rho.npy", rho)
    np.save(f"{a.out}.coef_train.npy", c_tr.astype(np.float32))
    np.save(f"{a.out}.coef_val.npy", c_va.astype(np.float32))
    meta = dict(
        kind="trajectory", rank=int(a.rank), d_flat=int(N_POS * D_LAT),
        n_pos=N_POS, d_latent=D_LAT, method="uncentered_svd_via_gram",
        centered=False, n_train_rows=int(n), n_val_rows=int(len(va)),
        explained=ev, rho_pct=a.rho_pct, rho_cover=a.rho_cover,
        rho_alpha=alpha, rho_norm=float(np.linalg.norm(rho)),
        cover_train=inside, cover_val=inside_va,
        residual_from="predicted_draft_q0hat",
        cache=a.cache,
        cache_meta_sha1=hashlib.sha1(
            open(f"{a.cache}.meta.json", "rb").read()).hexdigest()[:12],
        ktrue_sha1=hashlib.sha1(
            open(f"{a.cache}.ktrue.npy", "rb").read()).hexdigest()[:12],
        codebooks_sha1=cmeta.get("codebooks_sha1"),
        manifest=cmeta.get("manifest"), split_sha1=hashlib.sha1(
            open(f"{a.cache}.split.npy", "rb").read()).hexdigest()[:12],
        basis_sha1=hashlib.sha1(B.tobytes()).hexdigest()[:12],
        rho_sha1=hashlib.sha1(rho.tobytes()).hexdigest()[:12],
        code_version=kb.code_version([os.path.abspath(__file__)]),
        script_sha1=hashlib.sha1(
            open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12])
    json.dump(meta, open(f"{a.out}.meta.json", "w"), ensure_ascii=False,
              indent=1)
    print(f"\n  сохранено: {a.out}.basis.npy {B.shape}, .rho.npy {rho.shape}, "
          f"коэффициенты train/val, .meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
