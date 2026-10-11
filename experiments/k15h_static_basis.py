#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15h (подготовка): базисы для сравнения с фиксированным PCA4.

Только преобразования базиса и проверки на CPU, без раскаток. Координаты —
касательное пространство K-15f (k15f_continuous_refine): t ∈ R^{8×7},
рука t[k, c] = Δa / σ_c, схват — приращение логита / σ_g. Базис [B, 4, 56]
уже включает амплитудный множитель f (как в журнале k15f_basis).

ДВА ЯВНО РАЗЛИЧЁННЫХ ВАРИАНТА:

  static       ПОЛНОСТЬЮ СТАТИЧНЫЙ БАЗИС: δ_j = f·ρ_j·U_j из того же
               чекпойнта, без обучаемой поправки w_j(s). От входа не зависит
               вообще (ни рука, ни схват, ни норма).

  arm_dir      КОНТРОЛЬ АДАПТАЦИИ НАПРАВЛЕНИЯ РУКИ: направление руки —
               фиксированная часть руки U_j (в нормированных координатах);
               норма поправки руки В ЕДИНИЦАХ ДЕЙСТВИЯ (‖σ ⊙ δ_руки‖ по 8
               шагам × 6 каналам) равна норме learned при том же входном
               состоянии; координаты схвата — от learned при том же
               состоянии без изменений. Норма и схват ЗАВИСЯТ от состояния:
               это НЕ полностью статичная голова, а проверка узкого
               вопроса — приносит ли пользу адаптация направления руки.

Знак направления руки — знак U_j (привязан структурно, как у learned).
"""
import argparse
import sys

import numpy as np

H_EXEC, N_CH, K = 8, 7, 4
D_T = H_EXEC * N_CH
ARM = np.array([k * N_CH + c for k in range(H_EXEC) for c in range(6)])
GRIP = np.array([k * N_CH + 6 for k in range(H_EXEC)])
VARIANTS = ("static", "arm_dir")


def sigma_vec(sigma_arm):
    """σ по 48 координатам руки [шаг][канал]."""
    return np.tile(np.asarray(sigma_arm, np.float64), H_EXEC)


def static_basis(U, rho, f, batch=None):
    """δ_j = f·ρ_j·U_j: [4, 56] или [B, 4, 56] (копии одного и того же)."""
    U = np.asarray(U, np.float64)
    rho = np.asarray(rho, np.float64)
    if U.shape != (K, D_T) or rho.shape != (K,):
        raise ValueError(f"U {U.shape}, ρ {rho.shape}")
    b = float(f) * rho[:, None] * U
    return b if batch is None else np.broadcast_to(b, (batch, K, D_T)).copy()


def arm_action_norm(basis, sigma_arm):
    """‖σ ⊙ δ_руки‖ в единицах действия: [..., 4]."""
    arm = np.asarray(basis, np.float64)[..., ARM]
    return np.linalg.norm(arm * sigma_vec(sigma_arm), axis=-1)


def arm_dir_basis(learned, U, sigma_arm):
    """learned [B, 4, 56] -> вариант arm_dir [B, 4, 56]."""
    learned = np.asarray(learned, np.float64)
    U = np.asarray(U, np.float64)
    u_arm = U[:, ARM]                                     # [4, 48]
    n_u = np.linalg.norm(u_arm * sigma_vec(sigma_arm), axis=-1)  # [4]
    if (n_u < 1e-12).any():
        raise ValueError(f"у U нет части руки: {n_u}")
    target = arm_action_norm(learned, sigma_arm)          # [B, 4]
    out = np.zeros_like(learned)
    out[..., ARM] = (target / n_u)[..., None] * u_arm[None]
    out[..., GRIP] = learned[..., GRIP]
    return out


def arm_cos(learned, U):
    """Косинус части руки learned с частью руки U_j (нормированные
    координаты): [B, 4]. Диагностика для журналов k15f_basis: отрицательный
    косинус означал бы, что arm_dir меняет знак поправки руки."""
    la = np.asarray(learned, np.float64)[..., ARM]
    ua = np.asarray(U, np.float64)[:, ARM]
    return (la * ua[None]).sum(-1) / np.maximum(
        np.linalg.norm(la, axis=-1) * np.linalg.norm(ua, axis=-1)[None],
        1e-300)


def make(variant, learned, U, rho, f, sigma_arm):
    if variant == "static":
        return static_basis(U, rho, f, batch=len(learned))
    if variant == "arm_dir":
        return arm_dir_basis(learned, U, sigma_arm)
    raise ValueError(f"вариант {variant!r}")


def checks(variant, learned, U, rho, f, sigma_arm, tol=1e-9):
    """Свойства варианта на конкретном наборе learned-базисов. Список
    нарушений."""
    p = []
    out = make(variant, learned, U, rho, f, sigma_arm)
    if variant == "static":
        if not np.allclose(out, out[:1], rtol=0, atol=tol):
            p.append("статичный базис зависит от входа")
        want = static_basis(U, rho, f)
        if not np.allclose(out[0], want, rtol=0, atol=tol):
            p.append("статичный базис не равен f·ρ·U")
    else:
        if not np.array_equal(out[..., GRIP],
                              np.asarray(learned, np.float64)[..., GRIP]):
            p.append("схват не сохранён от learned")
        nl = arm_action_norm(learned, sigma_arm)
        no = arm_action_norm(out, sigma_arm)
        if not np.allclose(no, nl, rtol=1e-9, atol=tol):
            p.append(f"норма руки не согласована: max|Δ| "
                     f"{float(np.abs(no - nl).max()):.3e}")
        ua = np.asarray(U, np.float64)[:, ARM]
        dirs = out[..., ARM]
        nz = np.linalg.norm(dirs, axis=-1) > tol
        cos = (dirs * ua[None]).sum(-1) / np.maximum(
            np.linalg.norm(dirs, axis=-1) * np.linalg.norm(ua, axis=-1)[None],
            1e-300)
        if not np.allclose(cos[nz], 1.0, atol=1e-9):
            p.append("направление руки не равно фиксированному U_j")
    return p


def zero_identity(basis, a0, sigma_arm, sigma_g):
    """Тождество действия при нулевых коэффициентах — через штатные
    compose/combine K-15f (нужен torch)."""
    import torch
    sys.path.insert(0, __import__("os").path.dirname(
        __import__("os").path.abspath(__file__)))
    import k15f_continuous_refine as kf
    b = torch.tensor(np.asarray(basis), dtype=torch.float32)
    a0_t = torch.tensor(np.asarray(a0), dtype=torch.float32)
    t = kf.compose(b, torch.zeros(K))
    a = kf.combine(a0_t, t, torch.tensor(np.asarray(sigma_arm),
                                         dtype=torch.float32),
                   float(sigma_g))
    return bool(torch.equal(a, a0_t))


def selftest():
    rng = np.random.default_rng(0)
    U = np.linalg.qr(rng.standard_normal((D_T, K)))[0].T     # [4, 56]
    rho = np.array([2.0, 1.5, 1.0, 0.8])
    f = 1.0
    sigma_arm = rng.uniform(0.02, 0.2, 6)
    sigma_g = 0.7
    B = 6
    # learned: U_j + w_j(s), w ⊥ U_j, ‖w‖ <= 1, масштаб f·ρ_j
    w = rng.standard_normal((B, K, D_T)) * 0.3
    w -= (w * U[None]).sum(-1, keepdims=True) * U[None]
    learned = f * rho[None, :, None] * (U[None] + w)
    for v in VARIANTS:
        assert checks(v, learned, U, rho, f, sigma_arm) == [], v
    st = static_basis(U, rho, f, batch=B)
    # статичный не зависит от входа: другой learned — тот же результат
    assert np.array_equal(make("static", learned * 3.0, U, rho, f,
                               sigma_arm), st)
    ad = arm_dir_basis(learned, U, sigma_arm)
    # согласование нормы в единицах действия, а не в нормированных
    # координатах (при разных σ по каналам они различаются)
    assert np.allclose(arm_action_norm(ad, sigma_arm),
                       arm_action_norm(learned, sigma_arm))
    assert not np.allclose(np.linalg.norm(ad[..., ARM], axis=-1),
                           np.linalg.norm(learned[..., ARM], axis=-1))
    # схват сохранён побитово; при другом схвате learned — другой схват
    assert np.array_equal(ad[..., GRIP], learned[..., GRIP])
    l2 = learned.copy()
    l2[..., GRIP] += 1.0
    assert not np.array_equal(arm_dir_basis(l2, U, sigma_arm)[..., GRIP],
                              ad[..., GRIP])
    # норма зависит от состояния — вариант НЕ статичный
    assert not np.allclose(ad, ad[:1])
    # мутации ловятся
    bad = ad.copy()
    bad[..., GRIP] += 1e-3
    orig = globals()["arm_dir_basis"]
    try:
        globals()["arm_dir_basis"] = lambda l_, U_, s_: bad
        assert any("схват" in x for x in checks("arm_dir", learned, U, rho,
                                                f, sigma_arm))
        sc = ad.copy()
        sc[..., ARM] *= 1.01
        globals()["arm_dir_basis"] = lambda l_, U_, s_: sc
        assert any("норма" in x for x in checks("arm_dir", learned, U, rho,
                                                f, sigma_arm))
        rot = ad.copy()
        rot[..., ARM] = rot[..., ARM][..., ::-1]
        globals()["arm_dir_basis"] = lambda l_, U_, s_: rot
        assert any("направление" in x for x in checks(
            "arm_dir", learned, U, rho, f, sigma_arm))
    finally:
        globals()["arm_dir_basis"] = orig
    assert np.allclose(arm_cos(ad, U)[np.linalg.norm(ad[..., ARM], axis=-1)
                                      > 0], 1.0)
    # нулевая норма learned -> нулевая рука, без деления на ноль
    z = learned.copy()
    z[0, 1] = 0.0
    assert np.all(arm_dir_basis(z, U, sigma_arm)[0, 1] == 0.0)
    try:
        U0 = U.copy()
        U0[2, ARM] = 0.0
        arm_dir_basis(learned, U0, sigma_arm)
        raise AssertionError("принят U без части руки")
    except ValueError:
        pass
    try:
        make("иной", learned, U, rho, f, sigma_arm)
        raise AssertionError("принят неизвестный вариант")
    except ValueError:
        pass
    # тождество действия при c = 0 для обоих вариантов (штатный combine)
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        a0 = rng.uniform(-0.9, 0.9, (B, 24, 7))
        for v in VARIANTS:
            assert zero_identity(make(v, learned, U, rho, f, sigma_arm), a0,
                                 sigma_arm, sigma_g), v
        ident = "тождество при c = 0 проверено"
    else:
        ident = "БЕЗ torch: тождество при c = 0 НЕ проверено"
    print(f"самопроверка k15h_static_basis пройдена: статичный не зависит "
          f"от входа, arm_dir — норма руки в единицах действия, схват и "
          f"направление; {ident}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-15h: статичный базис")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    raise SystemExit("модуль преобразований; раскатки — после полного K-15g")


if __name__ == "__main__":
    sys.exit(main())
