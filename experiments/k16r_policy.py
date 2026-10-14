#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16r: контроль направления поправки goal_band_hist_event.

Тот же событийный вызов и та же голова, что у goal_band_hist_event (рука
K-16 строится без изменений): в вызове импульса берётся её обычная
поправка δ (после α и предела нормы), переводится в координаты
исполняемых команд (δ_exec = δ_codec × scale_arm) и поворачивается:
  позиция (каналы 0–2)  — матрица R_pos ∈ SO(3),
  вращение (каналы 3–5) — матрица R_rot ∈ SO(3),
одинаково на всех восьми шагах; затем обратно в единицы кодека. Схват и
хвост чанка — из a0; вне импульса действие побитово равно a0.

Сохраняются: норма каждого шага (и всего массива — предел остаётся
выполненным), энергия временных разностей, энергия каждой частоты DCT,
суммированная по группе каналов. Меняется пространственное направление.

МАТРИЦЫ: равномерные на SO(3) (QR гауссовой матрицы, знаки по диагонали
R, det = +1), генератор — np.random.default_rng([сид, задача, состояние]);
сид — один из трёх заранее заданных (ROT_SEEDS); сид среды остаётся 101.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BASE = "goal_band_hist_event"
ROT_SEEDS = {"goal_band_rot_s1": 20261015, "goal_band_rot_s2": 20261016,
             "goal_band_rot_s3": 20261017}
LABELS = tuple(ROT_SEEDS)
H = 8


def so3(rng):
    """Равномерная матрица из SO(3)."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))[None]
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def rotations(seed, task, state):
    rng = np.random.default_rng([int(seed), int(task), int(state)])
    return so3(rng), so3(rng)


def rotate_delta(d_codec, scale_arm, R_pos, R_rot):
    """d_codec [8, 6] -> повёрнутая в исполняемых координатах [8, 6]."""
    s = np.asarray(scale_arm, np.float64)
    e = np.asarray(d_codec, np.float64) * s
    out = np.concatenate([e[:, :3] @ R_pos.T, e[:, 3:6] @ R_rot.T], 1)
    return out / s


def build_arm(device, basis, label, torch, *, models_path, plan_sha1,
              task, states, identity_report=None):
    import k15f_build_cache as kb
    import k16_policy as kp
    if label not in ROT_SEEDS:
        raise SystemExit(f"рука k16r: метка {label!r}")
    seed = ROT_SEEDS[label]
    arm = kp.build_arm(device, basis, BASE, torch, models_path=models_path,
                       plan_sha1=plan_sha1, identity_report=identity_report)
    rots = [rotations(seed, task, s) for s in states]
    ck = torch.load(models_path, map_location="cpu", weights_only=False)
    scale = np.asarray(ck["scale_arm"], np.float64)
    inner, log = arm.act, arm.log
    kp_bound = kp.ACTION_CLIP_BOUND
    log.label = label

    def act(batch, pos_off, autocast, first):
        a, q0 = inner(batch, pos_off, autocast, first)
        imp = np.asarray(log.rows["impulse"][-1], bool)
        a0 = np.asarray(log.rows["a0"][-1])
        d_orig = np.asarray(log.rows["d_codec"][-1], np.float32).copy()
        d_rot = np.full_like(d_orig, np.nan)
        for b in np.nonzero(imp)[0]:
            Rp, Rr = rots[b]
            dr = rotate_delta(d_orig[b], scale, Rp, Rr)
            a[b, :H, :6] = (a0[b, :, :6].astype(np.float64) + dr).astype(
                np.float32)
            d_rot[b] = dr.astype(np.float32)
        # журнал: применённая δ = повёрнутая; исходная — отдельно
        log.rows["act"][-1] = a[:, :H].astype(np.float32)
        log.rows["d_codec"][-1] = np.where(imp[:, None, None], d_rot,
                                           np.nan).astype(np.float32)
        log.rows["d_codec_orig"].append(d_orig)
        if (~imp).any() and not np.array_equal(a[~imp][:, :H], a0[~imp]):
            raise SystemExit("рука k16r: без импульса действие не равно a0")
        # после поворота — те же проверки допустимого действия, что у K-16
        # (обрезание изменило бы задуманный контроль: технический отказ)
        if not np.isfinite(a).all():
            raise SystemExit("рука k16r: действие не конечно после поворота")
        if float(np.abs(a[:, :H]).max()) > kp_bound:
            raise SystemExit(f"рука k16r: |действие| выше {kp_bound} после "
                             f"поворота")
        for b in np.nonzero(imp)[0]:          # нормы шагов сохранены
            e0, e1 = d_orig[b] * scale, d_rot[b] * scale
            for g in (slice(0, 3), slice(3, 6)):
                if not np.allclose(np.linalg.norm(e0[:, g], axis=1),
                                   np.linalg.norm(e1[:, g], axis=1),
                                   atol=1e-6):
                    raise SystemExit("рука k16r: поворот изменил норму")
        return a, q0
    log.extra["rot_pos"] = np.stack([r[0] for r in rots]).astype(np.float64)
    log.extra["rot_rot"] = np.stack([r[1] for r in rots]).astype(np.float64)
    arm.act = act
    m = arm.meta
    m.update(label=label, base_label=BASE, rotation_seed=seed,
             k16r_policy_sha1=kb.sha_file(os.path.abspath(__file__)))
    m["model_fingerprint"] = hashlib.sha1("|".join(str(m[k]) for k in (
        "model_fingerprint", "label", "rotation_seed", "k16r_policy_sha1"))
        .encode()).hexdigest()[:12]
    return arm


def selftest():
    import k16_models as km
    rng = np.random.default_rng(0)
    R = so3(rng)
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R),
                                                           1.0)
    # воспроизводимо по сиду и ключу; разные сиды/состояния — разные
    a1, a2 = rotations(1, 3, 25), rotations(1, 3, 25)
    assert np.array_equal(a1[0], a2[0]) and np.array_equal(a1[1], a2[1])
    assert not np.allclose(rotations(2, 3, 25)[0], a1[0])
    assert not np.allclose(rotations(1, 3, 26)[0], a1[0])
    assert not np.allclose(a1[0], a1[1])          # две разные матрицы
    # сохранение: норма шага по группам, временные разности, энергия частот
    scale = rng.uniform(0.2, 0.9, 6)
    d = rng.standard_normal((8, 6)) * 0.05
    dr = rotate_delta(d, scale, *a1)
    e, er = d * scale, dr * scale
    for g in (slice(0, 3), slice(3, 6)):
        assert np.allclose(np.linalg.norm(e[:, g], axis=1),
                           np.linalg.norm(er[:, g], axis=1))
        assert np.isclose(np.sum(np.diff(e[:, g], axis=0) ** 2),
                          np.sum(np.diff(er[:, g], axis=0) ** 2))
        D = km.dct_matrix()
        assert np.allclose((np.einsum("ft,tc->fc", D, e[:, g]) ** 2).sum(1),
                           (np.einsum("ft,tc->fc", D, er[:, g]) ** 2).sum(1))
    assert np.isclose(np.linalg.norm(e), np.linalg.norm(er))
    assert not np.allclose(dr, d)                 # направление меняется
    _wrapper_test()
    print("самопроверка k16r_policy пройдена: SO(3), воспроизводимость по "
          "сиду и состоянию, нормы шагов, временные разности, энергия "
          "частот по группам")
    return 0


def _wrapper_test():
    """Обёртка на поддельной внутренней руке: импульс в вызове 2 у среды 1;
    вне импульса — a0; в импульсе — a0 + повёрнутая δ; журнал согласован."""
    import tempfile
    from types import SimpleNamespace
    import torch
    import k16_policy as kp
    B = 3
    rng = np.random.default_rng(1)
    a0s = [rng.uniform(-.5, .5, (B, 20, 7)).astype(np.float32)
           for _ in range(3)]
    d = (rng.standard_normal((H, 6)) * 0.02).astype(np.float32)
    log = kp.K16Log(BASE)

    def fake_act(batch, pos_off, autocast, first):
        c = 0 if first else len(log.rows["a0"])
        a0 = a0s[c]
        imp = np.zeros(B, bool)
        dc = np.full((B, H, 6), np.nan, np.float32)
        a = a0.copy()
        if c == 1:
            imp[1] = True
            dc[1] = d
            a[1, :H, :6] = (a0[1, :H, :6].astype(np.float64) + d).astype(
                np.float32)
        log.add(a0=a0[:, :H], act=a[:, :H], chunk=c + 1, impulse=imp,
                d_codec=dc)
        return a, np.zeros((B, 4))
    saved = kp.build_arm
    kp.build_arm = lambda *a_, **k_: SimpleNamespace(
        act=fake_act, log=log, meta=dict(model_fingerprint="x"))
    try:
        with tempfile.TemporaryDirectory() as td:
            mp = os.path.join(td, "m.pt")
            torch.save(dict(scale_arm=[0.9] * 6), mp)
            arm = build_arm("cpu", "b", "goal_band_rot_s1", torch,
                            models_path=mp, plan_sha1="P", task=3,
                            states=[25, 26, 27])
            outs = [arm.act(None, 0, None, c == 0)[0] for c in range(3)]
    finally:
        kp.build_arm = saved
    summ, z = log.take()
    lv, a0 = z["k15d_level_goal_band_rot_s1"], z["k15d_a0"]
    imp = z["k16_impulse"]
    assert imp.sum() == 1 and imp[1, 1]
    assert np.array_equal(lv[~imp], a0[~imp])
    Rp, Rr = rotations(ROT_SEEDS["goal_band_rot_s1"], 3, 26)
    want = rotate_delta(d, [0.9] * 6, Rp, Rr)
    assert np.allclose(lv[1, 1, :, :6] - a0[1, 1, :, :6], want, atol=1e-6)
    assert np.allclose(z["k16_d_codec"][1, 1], want, atol=1e-6)
    assert np.allclose(z["k16_d_codec_orig"][1, 1], d)
    assert np.array_equal(lv[..., 6], a0[..., 6])
    assert np.array_equal(outs[1][:, :H], lv[1])
    assert z["k16_rot_pos"].shape == (3, 3, 3)
    # после поворота действие вне предела — технический отказ
    big = (np.ones((H, 6)) * 5.0).astype(np.float32)
    log2 = kp.K16Log(BASE)

    def fake2(batch, pos_off, autocast, first):
        a0 = a0s[0]
        imp = np.array([True, False, False])
        dc = np.full((B, H, 6), np.nan, np.float32)
        dc[0] = big
        a = a0.copy()
        log2.add(a0=a0[:, :H], act=a[:, :H], chunk=1, impulse=imp,
                 d_codec=dc)
        return a, np.zeros((B, 4))
    kp.build_arm = lambda *a_, **k_: SimpleNamespace(
        act=fake2, log=log2, meta=dict(model_fingerprint="x"))
    try:
        with tempfile.TemporaryDirectory() as td:
            mp = os.path.join(td, "m.pt")
            torch.save(dict(scale_arm=[0.9] * 6), mp)
            arm = build_arm("cpu", "b", "goal_band_rot_s1", torch,
                            models_path=mp, plan_sha1="P", task=3,
                            states=[25, 26, 27])
            try:
                arm.act(None, 0, None, True)
                raise AssertionError("принято действие вне предела")
            except SystemExit as e:
                assert "выше" in str(e)
    finally:
        kp.build_arm = saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="K-16r: рука")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    raise SystemExit("модуль руки; запуск — через k16r_harness.py")
