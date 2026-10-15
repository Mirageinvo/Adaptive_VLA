#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16s: контроль направления ЗАПИСАННОГО вмешательства learned K-16.

Опорная поправка состояния — из learned-блока goal_band_hist_event K-16:
δ_ref = (действие learned − его a0)[:8, :6], разность в float64. Таблица
(вызов импульса, a0 и действие learned) строится при фиксации плана.

Рука — рука q0 K-16 (без изменений; та же причинная функция события) и
обёртка: в первом событийном вызове среды строго проверяется, что вызов
равен эталонному и a0 побитово равен эталонному a0 (иначе технический
отказ); затем
  stored_replay     рука = сохранённые команды руки learned (побитово),
                    техническая проверка воспроизведения;
  stored_rot_s<k>   δ_ref повёрнута в исполняемых координатах (те же две
                    SO(3) и сиды, что K-16r), a = a0 + δ_rot (float64 ->
                    float32).
α и предел повторно не применяются (норма поворотом сохраняется). Схват и
хвост — из a0; вне импульса действие = a0. После изменения — проверки
конечности и |a| <= 1.5 (без обрезания).

Это контроль направления записанного вмешательства на этих 50 состояниях,
а не политика для неизвестных стартов: опорная поправка выбрана без
использования исходов.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k16r_policy as rp  # noqa: E402

H = 8
LABELS = ("stored_replay", "stored_rot_s1", "stored_rot_s2",
          "stored_rot_s3")
SEEDS = {"stored_rot_s1": rp.ROT_SEEDS["goal_band_rot_s1"],
         "stored_rot_s2": rp.ROT_SEEDS["goal_band_rot_s2"],
         "stored_rot_s3": rp.ROT_SEEDS["goal_band_rot_s3"]}


def ref_delta(level, a0):
    return (np.asarray(level, np.float64)[:, :6]
            - np.asarray(a0, np.float64)[:, :6])


def wrap(arm, label, ref, scale, task, states):
    """ref: dict(call [B], a0 [B, 8, 7], level [B, 8, 7])."""
    import k16_policy as kp
    if label not in LABELS:
        raise SystemExit(f"рука k16s: метка {label!r}")
    seed = SEEDS.get(label)
    rots = ([rp.rotations(seed, task, s) for s in states] if seed
            else None)
    inner, log = arm.act, arm.log
    log.label = label
    B = len(states)
    used = np.zeros(B, bool)
    seen = np.zeros(B, bool)

    def act(batch, pos_off, autocast, first):
        nonlocal used, seen
        if first:
            used, seen = np.zeros(B, bool), np.zeros(B, bool)
        a, q0 = inner(batch, pos_off, autocast, first)
        call = int(np.asarray(log.rows["chunk"][-1]))
        ev = np.asarray(log.rows["event"][-1], bool)
        a0 = np.asarray(log.rows["a0"][-1])
        imp = np.zeros(B, bool)
        d_app = np.full((B, H, 6), np.nan, np.float32)
        d_ref = np.full((B, H, 6), np.nan, np.float32)
        napp = np.full(B, np.nan, np.float32)
        for b in range(B):
            if not ev[b] or seen[b]:
                continue
            seen[b] = True                 # первое событие этой среды
            if call != int(ref["call"][b]) or not np.array_equal(
                    a0[b], ref["a0"][b]):
                raise SystemExit(f"рука k16s: эталон не воспроизведён "
                                 f"(среда {b}, вызов {call}, эталон "
                                 f"{int(ref['call'][b])})")
            dr = ref_delta(ref["level"][b], ref["a0"][b])
            if label == "stored_replay":
                a[b, :H, :6] = np.asarray(ref["level"][b])[:, :6]
            else:
                drot = rp.rotate_delta(dr, scale, *rots[b])
                a[b, :H, :6] = (a0[b, :, :6].astype(np.float64)
                                + drot).astype(np.float32)
            imp[b] = True
            used[b] = True
            d_app[b] = (a[b, :H, :6].astype(np.float64)
                        - a0[b, :, :6].astype(np.float64))
            d_ref[b] = dr
            napp[b] = np.linalg.norm(d_app[b] * scale)
        if not np.isfinite(a).all():
            raise SystemExit("рука k16s: действие не конечно")
        if float(np.abs(a[:, :H]).max()) > kp.ACTION_CLIP_BOUND:
            raise SystemExit("рука k16s: |действие| выше предела")
        if (~imp).any() and not np.array_equal(a[~imp][:, :H], a0[~imp]):
            raise SystemExit("рука k16s: без импульса действие не равно a0")
        log.rows["act"][-1] = a[:, :H].astype(np.float32)
        log.rows["impulse"][-1] = imp
        log.rows["d_codec"][-1] = d_app
        log.rows["norm_applied"][-1] = napp
        log.rows["norm_raw"][-1] = napp
        log.rows["d_codec_orig"].append(d_ref)
        if imp.any():
            print(f"    k16s: {label}: импульс в вызове {call} у сред "
                  f"{np.nonzero(imp)[0].tolist()}", flush=True)
        return a, q0
    if rots:
        log.extra["rot_pos"] = np.stack([r[0] for r in rots])
        log.extra["rot_rot"] = np.stack([r[1] for r in rots])
    arm.act = act
    return arm


def build_arm(device, basis, label, torch, *, models_path, plan_sha1, task,
              states, ref, ref_sha1, identity_report=None):
    import k15f_build_cache as kb
    import k16_policy as kp
    arm = kp.build_arm(device, basis, "q0", torch, models_path=models_path,
                       plan_sha1=plan_sha1, identity_report=identity_report)
    ck = torch.load(models_path, map_location="cpu", weights_only=False)
    wrap(arm, label, ref, np.asarray(ck["scale_arm"], np.float64), task,
         states)
    m = arm.meta
    m.update(label=label, base_label="q0", reference_sha1=ref_sha1,
             rotation_seed=SEEDS.get(label),
             k16s_policy_sha1=kb.sha_file(os.path.abspath(__file__)))
    m["model_fingerprint"] = hashlib.sha1("|".join(str(m[k]) for k in (
        "model_fingerprint", "label", "rotation_seed", "reference_sha1",
        "k16s_policy_sha1")).encode()).hexdigest()[:12]
    return arm


def selftest():
    from types import SimpleNamespace
    import k16_policy as kp
    rng = np.random.default_rng(0)
    B, C = 3, 4
    a0s = [rng.uniform(-.5, .5, (B, 20, 7)).astype(np.float32)
           for _ in range(C)]
    ev_at = {1: [0, 2], 2: [1]}               # событие: вызов 2 — среды 0, 2
    level = a0s[1][:, :H].copy()
    level[:, :, :6] += 0.01
    ref = dict(call=np.array([2, 3, 2]),
               a0=np.stack([a0s[1][0, :H], a0s[2][1, :H], a0s[1][2, :H]]),
               level=np.stack([level[0], a0s[2][1, :H] + np.float32(0.02) *
                               np.r_[np.ones(6), 0].astype(np.float32),
                               level[2]]))

    def mk():
        log = kp.K16Log("q0")

        def fake(batch, pos_off, autocast, first):
            c = 0 if first else len(log.rows["a0"])
            a0 = a0s[c]
            ev = np.zeros(B, bool)
            ev[ev_at.get(c, [])] = True
            log.add(a0=a0[:, :H], act=a0[:, :H], chunk=c + 1, event=ev,
                    impulse=np.zeros(B, bool),
                    d_codec=np.full((B, H, 6), np.nan, np.float32),
                    norm_applied=np.full(B, np.nan),
                    norm_raw=np.full(B, np.nan))
            return a0.copy(), np.zeros((B, 4))
        return SimpleNamespace(act=fake, log=log, meta={})
    scale = np.full(6, 0.9)
    # воспроизведение: рука = сохранённые команды побитово
    arm = wrap(mk(), "stored_replay", ref, scale, 3, [25, 26, 27])
    outs = [arm.act(None, 0, None, c == 0)[0] for c in range(C)]
    _s, z = arm.log.take()
    imp = z["k16_impulse"]
    assert imp[1, 0] and imp[1, 2] and imp[2, 1] and imp.sum() == 3
    assert np.array_equal(outs[1][0, :H, :6], ref["level"][0][:, :6])
    assert np.array_equal(outs[1][0, :H, 6], a0s[1][0, :H, 6])
    assert np.array_equal(z["k15d_level_stored_replay"][~imp],
                          z["k15d_a0"][~imp])
    # поворот: норма сохранена, направление другое
    arm = wrap(mk(), "stored_rot_s3", ref, scale, 3, [25, 26, 27])
    outs = [arm.act(None, 0, None, c == 0)[0] for c in range(C)]
    _s, z = arm.log.take()
    dref = ref_delta(ref["level"][0], ref["a0"][0])
    dapp = outs[1][0, :H, :6].astype(np.float64) - a0s[1][0, :H, :6]
    assert np.isclose(np.linalg.norm(dapp * scale),
                      np.linalg.norm(dref * scale), rtol=1e-5)
    assert not np.allclose(dapp, dref, atol=1e-4)
    assert np.allclose(z["k16_d_codec_orig"][1, 0], dref, atol=1e-7)
    # эталон не воспроизведён (другой a0) — технический отказ
    bad = dict(ref, a0=ref["a0"].copy())
    bad["a0"][0, 0, 0] += 1e-3
    arm = wrap(mk(), "stored_rot_s1", bad, scale, 3, [25, 26, 27])
    try:
        for c in range(C):
            arm.act(None, 0, None, c == 0)
        raise AssertionError("принят невоспроизведённый эталон")
    except SystemExit as e:
        assert "эталон" in str(e)
    print("самопроверка k16s_policy пройдена: воспроизведение побитово, "
          "поворот сохраняет норму, отказ при невоспроизведённом эталоне")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="K-16s: рука")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    raise SystemExit("модуль руки; запуск — через k16s_harness.py")
