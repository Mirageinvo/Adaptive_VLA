#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-13a: базис траекторного многообразия для HiCoRA-T.

ЧТО СТРОИТСЯ. Одно линейное подпространство ранга 64 в пространстве ЦЕЛОГО
остатка чанка размерности 16*512 = 8192. В исходной HiCoRA базис строится в
пространстве ОДНОЙ позиции (512), и каждая из 16 позиций получает свои
коэффициенты независимо. Здесь их 64 на весь чанк, и они связаны: один
коэффициент меняет траекторию согласованно.

РАЗНИЦА С K-11a — РОВНО В ОДНОЙ СТРОКЕ. Там остаток разворачивается в
`(-1, 512)`, по строке на позицию; здесь — в `(-1, 8192)`, по строке на чанк.
Всё остальное (что считать остатком, как строить грамиан, как выбирать предел)
переиспользуется без изменений, чтобы разница между HiCoRA и HiCoRA-T была
архитектурной, а не следствием другого рецепта.

ОТ ЧЕГО ОТСЧИТЫВАЕТСЯ ОСТАТОК. От ПРЕДСКАЗАННОГО черновика q0hat слоя 12, а не
от истинного кода нулевого уровня:

    R = (E0[k0] + E1[k1] + E2[k2]) - E0[q0hat].

Поправка обязана чинить и ошибку квантования, и ошибку предсказания черновика —
именно так устроена HiCoRA, и менять это здесь нельзя, иначе цель обучения
станет другой.

БАЗИС БЕЗ ЦЕНТРИРОВАНИЯ. Подпространство проходит через ноль, поэтому `u = 0`
даёт `dZ = 0` ТОЧНО. На этом держится тождество HiCoRA-T с coarse24 при нулевой
инициализации последнего слоя; центрированный PCA добавил бы постоянный сдвиг.

ГРАМИАН ПОТОКОМ. Остаток целиком — 150000 x 8192 x 4 байта, около 5 ГБ; он не
хранится. Накапливается только матрица 8192 x 8192, её собственные векторы
совпадают с правыми сингулярными векторами остатка.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

N_POS, D_LAT, RANK = 16, 512, 64
RHO_PCT, RHO_COVER = 95.0, 0.95
SPLITS = ("train", "val", "test")


def file_sha(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:12]


def array_sha(a):
    return hashlib.sha1(
        np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def load_split(path, n_rows):
    """Разбиение из кэша K-11a. ЭТО СТРОКИ, а не числа.

    K-11a пишет `np.asarray(d["split"]).astype(str)`, то есть train/val/test.
    Сравнение с числом 0 дало бы пустые наборы — ошибка, которая обнаружилась
    бы только на настоящем кэше, потому что синтетические тесты своё
    разбиение задавали сами.
    """
    sp = np.load(path, allow_pickle=True).astype(str)
    if sp.ndim != 1 or len(sp) != n_rows:
        raise SystemExit(f"split формы {sp.shape}, ожидался вектор длины "
                         f"{n_rows}")
    bad = sorted(set(sp.tolist()) - set(SPLITS))
    if bad:
        raise SystemExit(f"в split неизвестные значения {bad[:5]}, "
                         f"допустимы {SPLITS}")
    idx = {name: np.flatnonzero(sp == name) for name in SPLITS}
    if len(idx["train"]) == 0 or len(idx["val"]) == 0:
        raise SystemExit(f"пустой split: train {len(idx['train'])}, val "
                         f"{len(idx['val'])}")
    return idx, sp


def residual_rows(E, k_true, q0hat, idx, batch=4096, n_pos=None, d_lat=None):
    """Остаток чанка построчно: по строке на наблюдение, размерность n_pos*d.

    Генератор, а не массив: полный остаток не помещается в памяти, а для
    грамиана он и не нужен целиком.
    """
    n_pos = k_true.shape[2] if n_pos is None else n_pos
    d_lat = E.shape[-1] if d_lat is None else d_lat
    for a in range(0, len(idx), batch):
        s = idx[a:a + batch]
        z_full = E[0][k_true[s, 0, :].astype(np.int64)]
        for lvl in range(1, E.shape[0]):
            z_full = z_full + E[lvl][k_true[s, lvl, :].astype(np.int64)]
        z0 = E[0][q0hat[s].astype(np.int64)]
        yield (z_full - z0).reshape(len(s), n_pos * d_lat)


def gram_stream(E, k_true, q0hat, idx, batch=4096, log=print,
                n_pos=None, d_lat=None):
    """Грамиан (n_pos*d) x (n_pos*d) по заданным строкам. Остаток не хранится."""
    n_pos = k_true.shape[2] if n_pos is None else n_pos
    d_lat = E.shape[-1] if d_lat is None else d_lat
    d = n_pos * d_lat
    G = np.zeros((d, d), np.float64)
    n = 0
    for r in residual_rows(E, k_true, q0hat, idx, batch, n_pos, d_lat):
        rd = r.astype(np.float64)
        G += rd.T @ rd
        n += len(rd)
        if batch and n % (batch * 8) == 0:
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
    rank = int(rank)
    if not 1 <= rank <= G.shape[0]:
        raise SystemExit(f"ранг {rank} вне [1, {G.shape[0]}]")
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    if float(w[0]) <= 0:
        raise SystemExit("грамиан вырожден: остаток тождественно нулевой")
    neg = float(w[w < 0].sum())
    if neg < -1e-6 * float(w[0]):
        raise SystemExit(f"у грамиана заметно отрицательные собственные "
                         f"значения (сумма {neg:.3e}): накопление испорчено")
    B = V[:, :rank].T.copy()
    return np.ascontiguousarray(B, np.float32), np.maximum(w, 0.0)


def explained(w, rank):
    tot = float(w.sum())
    return float(w[:int(rank)].sum() / tot) if tot > 0 else 0.0


def coeffs_stream(E, k_true, q0hat, idx, B, batch=4096, n_pos=None,
                  d_lat=None):
    """Целевые коэффициенты c* = B r для заданных строк."""
    out = []
    for r in residual_rows(E, k_true, q0hat, idx, batch, n_pos, d_lat):
        out.append(r.astype(np.float32) @ B.T)
    return (np.concatenate(out) if out
            else np.zeros((0, B.shape[0]), np.float32))


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


def apply_correction(B, rho, u, n_pos=N_POS, d_lat=D_LAT):
    """dZ = reshape(B^T (rho * tanh(u))). Форма (..., n_pos, d_lat)."""
    c = rho * np.tanh(u)
    dz = c @ B
    return dz.reshape(*u.shape[:-1], n_pos, d_lat)


def save_atomic(path, saver):
    """Запись через временное имя. Файловым объектом, а не именем.

    `np.save` дописывает `.npy`, если имя им не кончается, поэтому
    `saver(tmp)` создавал бы `...tmp.123.npy`, а os.replace искал бы
    `...tmp.123` и падал. Передача открытого файла снимает это поведение.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        saver(fh)
    os.replace(tmp, path)


# ------------------------------ самопроверка ------------------------------

def _tiny_cache(tmp, n=24, n_pos=3, d_lat=8, vocab=7, levels=3, seed=0):
    """Маленький кэш ТОЙ ЖЕ формы, что настоящий, включая СТРОКОВЫЙ split."""
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(levels, vocab, d_lat)).astype(np.float32)
    k_true = rng.integers(0, vocab, size=(n, levels, n_pos)).astype(np.int16)
    q0hat = rng.integers(0, vocab, size=(n, n_pos)).astype(np.int16)
    sp = np.array(["train"] * (n - 8) + ["val"] * 4 + ["test"] * 4)
    base = os.path.join(tmp, "cache")
    np.save(f"{base}.codebooks.npy", E)
    np.save(f"{base}.ktrue.npy", k_true)
    np.save(f"{base}.q0hat.npy", q0hat)
    np.save(f"{base}.split.npy", sp)
    json.dump(dict(codebooks_sha1=array_sha(E), manifest=dict(n_episodes=n),
                   d_latent=d_lat),
              open(f"{base}.meta.json", "w"))
    return base, E, k_true, q0hat, sp


def selftest():
    import subprocess
    import tempfile
    rng = np.random.default_rng(0)
    NP_, DL_ = 3, 8                     # тестовые размерности, не production

    # --- 1. БАЗИС ИЗ ГРАМИАНА == UNCENTERED SVD ---------------------------
    n, dd, r = 200, 30, 5
    X = rng.normal(size=(n, dd)) @ rng.normal(size=(dd, dd))
    B, w = basis_from_gram(X.T @ X, r)
    _U, S, Vt = np.linalg.svd(X, full_matrices=False)
    P1 = B.T.astype(np.float64) @ B.astype(np.float64)
    P2 = Vt[:r].T @ Vt[:r]
    assert np.abs(P1 - P2).max() < 1e-6, np.abs(P1 - P2).max()
    assert np.allclose(w[:r], S[:r] ** 2, rtol=1e-6)
    assert np.abs(B @ B.T - np.eye(r)).max() < 1e-5
    for bad in (0, dd + 1):
        try:
            basis_from_gram(X.T @ X, bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"ранг {bad} принят")

    # --- 2. ЦЕНТРИРОВАНИЕ ДАЛО БЫ ДРУГОЕ ПОДПРОСТРАНСТВО ------------------
    Xs = X + 5.0
    Bc, _ = basis_from_gram((Xs - Xs.mean(0)).T @ (Xs - Xs.mean(0)), r)
    Bu, _ = basis_from_gram(Xs.T @ Xs, r)
    assert np.abs(Bc.T @ Bc - Bu.T @ Bu).max() > 1e-3, \
        "центрированный и нецентрированный базисы совпали: тест пуст"

    # --- 3. ОСТАТОК ОТ ПРЕДСКАЗАННОГО ЧЕРНОВИКА ---------------------------
    tmp = tempfile.mkdtemp(prefix="k13a_")
    base, E, k_true, q0hat, sp = _tiny_cache(tmp, n_pos=NP_, d_lat=DL_)
    rows = np.concatenate(list(residual_rows(E, k_true, q0hat,
                                             np.arange(len(k_true)), 5,
                                             NP_, DL_)))
    assert rows.shape == (len(k_true), NP_ * DL_), rows.shape
    exp0 = (E[0][k_true[0, 0]] + E[1][k_true[0, 1]] + E[2][k_true[0, 2]]
            - E[0][q0hat[0]]).reshape(-1)
    assert np.abs(rows[0] - exp0).max() < 1e-5
    alt = (E[1][k_true[0, 1]] + E[2][k_true[0, 2]]).reshape(-1)
    assert np.abs(rows[0] - alt).max() > 1e-6, \
        "остаток совпал с чисто квантовым: черновик не учтён"

    # --- 4. РАЗБИЕНИЕ — СТРОКИ. Это и был пропущенный блокер. -------------
    idx, sp2 = load_split(f"{base}.split.npy", len(k_true))
    assert len(idx["train"]) == len(k_true) - 8 and len(idx["val"]) == 4
    assert sp2.dtype.kind in "US", sp2.dtype
    # числовое сравнение дало бы пустоту — ровно та ошибка
    assert len(np.flatnonzero(sp2 == 0)) == 0
    np.save(os.path.join(tmp, "bad.npy"), np.array(["train", "лишнее"]))
    for path, nrow, needle in ((os.path.join(tmp, "bad.npy"), 2,
                                "неизвестные значения"),
                               (f"{base}.split.npy", 999, "ожидался вектор")):
        try:
            load_split(path, nrow)
        except SystemExit as e:
            assert needle in str(e), e
        else:
            raise AssertionError(f"принято: {path}")
    np.save(os.path.join(tmp, "noval.npy"),
            np.array(["train"] * 3 + ["test"]))
    try:
        load_split(os.path.join(tmp, "noval.npy"), 4)
    except SystemExit as e:
        assert "пустой split" in str(e), e
    else:
        raise AssertionError("split без val принят")

    # --- 5. ТОЖДЕСТВО u=0 -> dZ=0 ТОЧНОЕ ----------------------------------
    d = NP_ * DL_
    Bt = np.linalg.qr(rng.normal(size=(d, 4)))[0].T.astype(np.float32)
    rho = np.abs(rng.normal(size=4)).astype(np.float32) + 0.1
    dz = apply_correction(Bt, rho, np.zeros((3, 4), np.float32), NP_, DL_)
    assert dz.shape == (3, NP_, DL_), dz.shape
    assert np.abs(dz).max() == 0.0, "нулевые коэффициенты дали поправку"
    assert np.abs(apply_correction(Bt, rho, np.ones((1, 4), np.float32),
                                   NP_, DL_)).max() > 0

    # --- 6. ПРЕДЕЛ ПОКРЫВАЕТ ЗАЯВЛЕННУЮ ДОЛЮ ------------------------------
    coef = rng.normal(size=(4000, 16)).astype(np.float32)
    rho_v, alpha = rho_joint(coef, RHO_PCT, 0.95)
    inside = (np.abs(coef) <= rho_v[None]).all(axis=1).mean()
    assert 0.93 <= inside <= 0.97, inside
    s_only = np.percentile(np.abs(coef), RHO_PCT, axis=0)
    assert (np.abs(coef) <= s_only[None]).all(axis=1).mean() < inside - 0.1

    # --- 7. ГРАМИАН ТОЛЬКО ПО ЗАДАННЫМ СТРОКАМ, коэффициенты ортогональны --
    G2, n2 = gram_stream(E, k_true, q0hat, idx["train"], 5,
                         lambda *_: None, NP_, DL_)
    assert n2 == len(idx["train"])
    B2, _w2 = basis_from_gram(G2, 3)
    c2 = coeffs_stream(E, k_true, q0hat, idx["train"], B2, 5, NP_, DL_)
    resid = np.concatenate(list(residual_rows(E, k_true, q0hat,
                                              idx["train"], 5, NP_, DL_)))
    assert np.abs((resid - c2 @ B2) @ B2.T).max() < 1e-3
    Ga, _ = gram_stream(E, k_true, q0hat, idx["val"], 5, lambda *_: None,
                        NP_, DL_)
    assert np.abs(Ga - G2).max() > 1e-3, "грамиан не зависит от набора строк"

    # --- 8. ИНТЕГРАЦИОННЫЙ ПРОГОН НАСТОЯЩЕГО main НА МАЛЕНЬКОМ КЭШЕ -------
    # Математические тесты выше не поймали строковый split: они задавали
    # разбиение сами. Ловит только запуск того пути, которым идёт команда.
    out = os.path.join(tmp, "basis")
    rc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--cache", base,
         "--rank", "3", "--batch", "5", "--expect-d-latent", str(DL_),
         "--out", out], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stdout[-800:] + rc.stderr[-800:]
    Bf = np.load(f"{out}.basis.npy")
    mf = json.load(open(f"{out}.meta.json"))
    assert Bf.shape == (3, NP_ * DL_), Bf.shape
    assert mf["centered"] is False and mf["rank"] == 3
    assert mf["n_train_rows"] == len(idx["train"])
    assert mf["q0hat_sha1"] == array_sha(q0hat), "хэш черновика не сохранён"
    assert mf["codebooks_sha1"] == array_sha(E)
    assert not os.path.exists(f"{out}.meta.json.tmp.{os.getpid()}")
    # повреждённые коды отвергаются до счёта
    np.save(f"{base}.ktrue.npy",
            (k_true.astype(np.int32) + 1000).astype(np.int32))
    rc2 = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--cache", base,
         "--rank", "3", "--batch", "5", "--expect-d-latent", str(DL_),
         "--out", os.path.join(tmp, "b2")], capture_output=True, text=True)
    assert rc2.returncode != 0 and "вне словаря" in (rc2.stdout + rc2.stderr), \
        rc2.stdout[-400:] + rc2.stderr[-400:]
    print("самопроверка k13a_build_trajectory_basis пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--rho-pct", type=float, default=RHO_PCT)
    ap.add_argument("--rho-cover", type=float, default=RHO_COVER)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--expect-d-latent", type=int, default=D_LAT)
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
    cmeta = json.load(open(f"{a.cache}.meta.json"))

    n_obs, n_lvl, n_pos = k_true.shape
    d_lat = int(E.shape[-1])
    if n_lvl != E.shape[0]:
        raise SystemExit(f"уровней в кодах {n_lvl}, в книгах {E.shape[0]}")
    if d_lat != a.expect_d_latent:
        raise SystemExit(f"латент {d_lat}, ожидался {a.expect_d_latent}")
    if q0hat.shape != (n_obs, n_pos):
        raise SystemExit(f"черновик формы {q0hat.shape}, ожидалось "
                         f"{(n_obs, n_pos)}")
    # ДИАПАЗОНЫ КОДОВ — ДО СЧЁТА: код вне словаря дал бы обращение за границу
    # или молча завернулся отрицательным индексом
    for nm, arr in (("k_true", k_true), ("q0hat", q0hat)):
        lo, hi = int(np.asarray(arr).min()), int(np.asarray(arr).max())
        if lo < 0 or hi >= E.shape[1]:
            raise SystemExit(f"{nm}: коды [{lo}, {hi}] вне словаря "
                             f"[0, {E.shape[1] - 1}]")
    if not 1 <= a.rank <= n_pos * d_lat:
        raise SystemExit(f"ранг {a.rank} вне [1, {n_pos * d_lat}]")

    # ХЭШИ ПО ФАКТИЧЕСКИ ЗАГРУЖЕННЫМ МАССИВАМ, а не переписанные из чужой
    # meta: базис зависит от q0hat и кодовых книг напрямую, и если они не те,
    # обученная голова будет несовместима с прогоном
    cb_sha = array_sha(E)
    q0_sha = array_sha(np.asarray(q0hat))
    if cmeta.get("codebooks_sha1") not in (None, cb_sha):
        raise SystemExit(f"кодовые книги на диске {cb_sha}, а в meta кэша "
                         f"{cmeta.get('codebooks_sha1')}")

    idx, _sp = load_split(f"{a.cache}.split.npy", n_obs)
    tr, va = idx["train"], idx["val"]
    print(f"  кэш {a.cache}: {n_obs} наблюдений, уровней {n_lvl}, позиций "
          f"{n_pos}, латент {d_lat}, словарь {E.shape[1]}")
    print(f"  split: train {len(tr)}, val {len(va)}, test "
          f"{len(idx['test'])}; базис строится ТОЛЬКО по train")

    G, n = gram_stream(E, k_true, q0hat, tr, a.batch, n_pos=n_pos,
                       d_lat=d_lat)
    B, w = basis_from_gram(G, a.rank)
    ev = explained(w, a.rank)
    print(f"\n  базис ранга {a.rank} из {n_pos * d_lat} измерений; "
          f"объяснённая доля энергии остатка {100 * ev:.2f}%")
    print(f"    ОПИСАТЕЛЬНО: сравнивать эту долю с локальным базисом K-11a "
          f"напрямую нельзя —\n    там 16*rank степеней свободы против {a.rank} "
          f"здесь, и энергия латента\n    не равна влиянию на действия. "
          f"Решение принимается по rollout, не по этой доле.")
    print(f"    ортонормированность: max|B B^T - I| = "
          f"{np.abs(B @ B.T - np.eye(a.rank)).max():.2e}")

    c_tr = coeffs_stream(E, k_true, q0hat, tr, B, a.batch, n_pos, d_lat)
    rho, alpha = rho_joint(c_tr, a.rho_pct, a.rho_cover)
    inside = float((np.abs(c_tr) <= rho[None]).all(axis=1).mean())
    print(f"\n  предел rho: процентиль {a.rho_pct} по координате, общий "
          f"множитель alpha={alpha:.3f}")
    print(f"    внутрь бокса целиком попадает {100 * inside:.2f}% train-"
          f"наблюдений (цель {100 * a.rho_cover:.0f}%)")
    print(f"    ||rho|| = {float(np.linalg.norm(rho)):.4f}")

    c_va = coeffs_stream(E, k_true, q0hat, va, B, a.batch, n_pos, d_lat)
    inside_va = float((np.abs(c_va) <= rho[None]).all(axis=1).mean())
    print(f"    на val внутрь попадает {100 * inside_va:.2f}% — предел "
          f"переносится, а не подогнан")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    save_atomic(f"{a.out}.basis.npy", lambda fh: np.save(fh, B))
    save_atomic(f"{a.out}.rho.npy", lambda fh: np.save(fh, rho))
    save_atomic(f"{a.out}.coef_train.npy",
                lambda fh: np.save(fh, c_tr.astype(np.float32)))
    save_atomic(f"{a.out}.coef_val.npy",
                lambda fh: np.save(fh, c_va.astype(np.float32)))
    meta = dict(
        kind="trajectory", rank=int(a.rank), d_flat=int(n_pos * d_lat),
        n_pos=int(n_pos), d_latent=d_lat, n_levels=int(n_lvl),
        method="uncentered_svd_via_gram", centered=False,
        n_train_rows=int(n), n_val_rows=int(len(va)),
        explained=ev, rho_pct=a.rho_pct, rho_cover=a.rho_cover,
        rho_alpha=alpha, rho_norm=float(np.linalg.norm(rho)),
        cover_train=inside, cover_val=inside_va,
        residual_from="predicted_draft_q0hat",
        cache=a.cache,
        cache_meta_sha1=file_sha(f"{a.cache}.meta.json"),
        ktrue_sha1=file_sha(f"{a.cache}.ktrue.npy"),
        q0hat_sha1=q0_sha, codebooks_sha1=cb_sha,
        codebooks_sha1_in_cache_meta=cmeta.get("codebooks_sha1"),
        split_sha1=file_sha(f"{a.cache}.split.npy"),
        manifest=cmeta.get("manifest"),
        basis_sha1=array_sha(B), rho_sha1=array_sha(rho),
        code_version=kb.code_version([os.path.abspath(__file__)]),
        script_sha1=file_sha(os.path.abspath(__file__)))
    # META ПОСЛЕДНЕЙ: её наличие означает, что все массивы уже на диске
    save_atomic(f"{a.out}.meta.json",
                lambda fh: fh.write(json.dumps(meta, ensure_ascii=False,
                                               indent=1).encode()))
    print(f"\n  сохранено: {a.out}.basis.npy {B.shape}, .rho.npy {rho.shape}, "
          f"коэффициенты train/val, .meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
