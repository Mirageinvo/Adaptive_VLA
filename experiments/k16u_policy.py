#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16u: рука — один событийный импульс пары G+C варианта обучения.

Метки: q0, frozen_goal_event, separate_goal_event, joint_goal_event,
joint_ctx_event. У каждого варианта — СВОЯ пара G+C из data/k16u/models.pt.
Событие (первая смена команды схвата в a0, причинно), трекер, журнал,
apply_correction (α и предел — из чекпойнта), проверки действия — как у
руки K-16. Предсказанный схват G для выбора момента не используется и не
исполняется. Состояние руки — из обёртки process_state.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k16_models as km  # noqa: E402
import k16_policy as kp  # noqa: E402

H = 8
POLICIES = {"q0": (None, None),
            "frozen_goal_event": ("frozen_goal", "event"),
            "separate_goal_event": ("separate_goal", "event"),
            "joint_goal_event": ("joint_goal", "event"),
            "joint_ctx_event": ("joint_ctx", "event")}
LABELS = tuple(POLICIES)


def load_models(path, torch, dev):
    import k16u_models as um
    ck = torch.load(path, map_location="cpu", weights_only=False)
    Pair = um.build(torch, int(ck["d_model"]), int(ck["n_pos"]))
    pairs = {}
    for k, v in ck["variants"].items():
        p = Pair(v["mode"], {kk[2:]: vv for kk, vv in v["state"].items()
                             if kk.startswith("G.")})
        p.load_state_dict(v["state"])
        pairs[k] = p.to(dev).eval()
    return ck, pairs


def build_arm(device, basis, label, torch, *, models_path, plan_sha1,
              identity_report=None):
    import k15_context
    import k15d_policy as kdp
    import k15f_build_cache as kb
    import k15f_check_identity as ki
    import k15f_continuous_refine as kf
    import k15f_policy as kfp
    if label not in POLICIES:
        raise SystemExit(f"рука k16u: метка {label!r}")
    head_k, sched = POLICIES[label]
    tag = str(device).replace(":", "")
    identity_report = identity_report or f"reports/k15f/identity_{tag}.json"
    g15a, _g = kdp.gate_paths(device)
    for f_ in (basis, identity_report, g15a, models_path):
        if not os.path.exists(f_):
            raise SystemExit(f"нет {f_}")
    basis_sha = kb.sha_file(basis)
    rep = json.load(open(identity_report))
    prob = kfp.check_identity_report(rep, basis_sha=basis_sha, device=device,
                                     code=kb.sha_file(kf.__file__),
                                     gate_sha=kb.sha_file(ki.__file__))
    ck_b = torch.load(basis, map_location="cpu", weights_only=False)
    prob += ki.check_basis_ckpt(ck_b)
    if prob:
        raise SystemExit("рука K-16u не собрана: " + "; ".join(prob))
    ctx = k15_context.build(kdp.context_namespace(device, g15a))
    model, dev = ctx.model, ctx.dev
    k15t = k15_context.k15t
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    frozen, _n, _e = k15t.frozen_content_sha(model, torch, set())
    if frozen != rep.get("frozen_sha1"):
        raise SystemExit("замороженное не то, что в гейте тождества")
    ck, pairs = load_models(models_path, torch, dev)
    r_sigma = np.asarray(ck["norms"]["r_sigma"], np.float64)
    dpos_sigma = np.asarray(ck["norms"]["dpos_sigma"], np.float64)
    scale = np.asarray(list(ck["scale_arm"]) + [1.0], np.float64)
    alpha, clip = float(ck["alpha"]), float(ck["clip_env_norm"])
    log = kp.K16Log(label)
    st = dict(tr=None, nbuf=0)

    def act(batch, pos_off, autocast, first):
        if first:
            st["nbuf"] = len(kp.STATE_BUF) - 1
        if len(kp.STATE_BUF) != st["nbuf"] + 1:
            raise SystemExit("рука k16u: нет состояния текущего вызова")
        st["nbuf"] = len(kp.STATE_BUF)
        state = np.asarray(kp.STATE_BUF[-1], np.float64)
        with torch.no_grad(), autocast:
            v, p_ = model.build_inputs(position_offset=pos_off, **batch)
            q0, a0, h18 = kf.plain_h18(
                model, vlm_inputs_embeds=v,
                attention_mask=batch.get("attention_mask"), position_ids=p_,
                decode=ctx.decode_fp32)
        a0_np = a0.float().cpu().numpy()
        B = a0_np.shape[0]
        if first:
            st["tr"] = kp.Tracker(B)
        tr = st["tr"]
        if state.shape != (B, 8):
            raise SystemExit(f"рука k16u: состояние формы {state.shape}")
        ev, trig, hist, mask = tr.step(state, -a0_np[:, :H, 6], sched,
                                       [-1] * B)
        sf = km.state_features(state, hist, mask,
                               np.full(B, 8 * (tr.call - 1)), True)
        a = a0_np.copy()
        nan = lambda *s: np.full(s, np.nan, np.float32)   # noqa: E731
        goal_log, gm_log = nan(B, km.GOAL_DIM), nan(B, 3)
        out_log, dc_log = nan(B, H, 6), nan(B, H, 6)
        nraw, napp, clp = nan(B), nan(B), np.zeros(B, bool)
        hid_log = nan(B, km.GOAL_HIDDEN)
        t0 = time.perf_counter()
        if trig.any():
            idx = np.nonzero(trig)[0]
            with torch.no_grad():
                h_ = h18[idx]
                a0_t = a0[idx, :H].float()
                sf_t = torch.as_tensor(sf[idx], device=dev)
                o = pairs[head_k](h_, a0_t, sf_t)
                r_out, g_out = o["r"], o["goal"]
                hid_log[idx] = o["hid"].float().cpu().numpy()
            r_np = r_out.float().cpu().numpy()
            a_new, info = km.apply_correction(a0_np[idx], r_np, r_sigma,
                                              scale, alpha, clip)
            a[idx] = a_new
            g_np = g_out.float().cpu().numpy()
            goal_log[idx], gm_log[idx] = g_np, g_np[:, :3] * dpos_sigma
            out_log[idx], dc_log[idx] = r_np, info["d_codec"]
            nraw[idx], napp[idx] = info["norm_raw"], info["norm_applied"]
            clp[idx] = info["clipped"]
        head_ms = (time.perf_counter() - t0) * 1e3
        keep = ~trig
        if not np.array_equal(a[keep], a0_np[keep]):
            raise SystemExit("рука k16u: без импульса действие не равно a0")
        if not np.isfinite(a).all():
            raise SystemExit("рука k16u: действие не конечно")
        if float(np.abs(a[:, :H]).max()) > kp.ACTION_CLIP_BOUND:
            raise SystemExit("рука k16u: |действие| выше предела")
        tr.after(state, -a[:, :H, 6])
        if first:
            print(f"    проверка k16u: {label}: голова {head_k}", flush=True)
        if trig.any():
            print(f"    k16u: импульс в вызове {tr.call} у сред "
                  f"{np.nonzero(trig)[0].tolist()}, ||δ|| исп. "
                  f"{np.nanmax(napp):.4f}", flush=True)
        log.add(a0=a0_np[:, :H].astype(np.float32),
                act=a[:, :H].astype(np.float32), chunk=tr.call,
                state=state.astype(np.float32), hist_mask=mask, event=ev,
                impulse=trig, goal=goal_log, goal_m=gm_log, out=out_log,
                d_codec=dc_log, norm_raw=nraw, norm_applied=napp,
                clipped=clp, head_ms=head_ms, sf=sf, goal_hidden=hid_log,
                h18=(h18.to(torch.float16).cpu().numpy()
                     if label == "q0" or trig.any() else
                     np.zeros((B,) + tuple(h18.shape[1:]), np.float16)))
        log.extra["first_event"] = np.asarray(tr.first_event, np.int64)
        return a.astype(np.float32), q0.cpu().numpy()

    meta = dict(
        arm="k16u", label=label, head=head_k, variant=head_k,
        schedule=sched,
        plan_sha1=plan_sha1, models=os.path.abspath(models_path),
        models_sha1=kb.sha_file(models_path), alpha=alpha,
        clip_env_norm=clip, units=ck.get("units"), donor_calls=None,
        coeffs=[0.0] * 4, control=False, basis=os.path.abspath(basis),
        basis_sha1=basis_sha, basis_state_sha1=ck_b["state_sha1"],
        stats_sha1=ck_b["stats_sha1"], amp_factor=float(ck_b["amp_factor"]),
        control_seed=int(ck_b["control_seed"]),
        control_kind=ck_b.get("control_kind"),
        identity_report=os.path.abspath(identity_report),
        identity_report_sha1=kb.sha_file(identity_report),
        refine_module_sha1=kb.sha_file(kf.__file__),
        k15f_policy_sha1=kb.sha_file(kfp.__file__),
        k16u_policy_sha1=kb.sha_file(os.path.abspath(__file__)),
        k16_models_sha1=kb.sha_file(km.__file__),
        frozen_content_sha=frozen, joint_sha1=ctx.joint_sha,
        code_version=ctx.code_version, codec=ctx.codec_fp,
        k15a_gate=ctx.gate_info, layers_per_call=18, preflight=False,
        admission_override=None, decodes_per_call=1)
    meta = json.loads(json.dumps(meta, default=k15t.json_scalar))
    meta["model_fingerprint"] = hashlib.sha1("|".join(str(meta[k]) for k in (
        "label", "models_sha1", "k16u_policy_sha1", "plan_sha1",
        "frozen_content_sha", "joint_sha1", "basis_sha1")).encode()
    ).hexdigest()[:12]
    return SimpleNamespace(model=model, proc=ctx.proc, codec=ctx.codec,
                           act=act, log=log, meta=meta, ctx=ctx)


def selftest():
    import tempfile
    import torch
    import k16_models as km_
    import k16u_models as um
    C = km_.build(torch)
    torch.manual_seed(0)
    g_state = C["GoalNet"]().state_dict()
    Pair = um.build(torch)
    variants = {m: dict(state=Pair(m, g_state).state_dict(), mode=m)
                for m in um.MODES}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.pt")
        torch.save(dict(variants=variants, d_model=768, n_pos=16), p)
        _ck, pairs = load_models(p, torch, torch.device("cpu"))
    assert set(pairs) == set(um.MODES)
    assert len({id(pp.G) for pp in pairs.values()}) == 4
    assert POLICIES["q0"] == (None, None) and len(LABELS) == 5
    print("самопроверка k16u_policy пройдена: свои пары G+C, метки")
    return 0


def integration():
    """Игрушечная модель: q0 = a0; один событийный импульс; поправка =
    журнал δ; у ctx и ctx_goal входы до импульса совпадают с q0."""
    import tempfile
    import torch
    import k15_context
    import k15d_policy as kdp
    import k15d_train as kt
    import k15f_build_cache as kbc
    import k15f_check_identity as ki
    import k15f_continuous_refine as kf
    import k15f_pretrain_basis as kpb
    import k16b_train as bt
    saved = (k15_context.build, sys.argv, kdp.gate_paths)
    ctx = kt._fake_ctx()
    ctx.model.action_expert.norm.variance_epsilon = 1e-6
    ctx.vocab = int(ctx.model.book0.shape[0])
    k15_context.build = lambda a: ctx
    try:
        with tempfile.TemporaryDirectory() as td:
            cache = os.path.join(td, "h18_cache")
            sys.argv = ["x", "--device", "cpu", "--out", cache]
            assert kbc.main() == 0
            basis0 = os.path.join(td, "basis0.pt")
            sys.argv = ["x", "--mode", "full", "--device", "cpu", "--cache",
                        cache, "--epochs", "1", "--out", basis0, "--report",
                        os.path.join(td, "b.json")]
            assert kpb.main() == 0
            ckb = torch.load(basis0, map_location="cpu", weights_only=False)
            stb = dict(ckb["state"])
            stb["out.weight"] = torch.randn(
                stb["out.weight"].shape,
                generator=torch.Generator().manual_seed(1)) * 0.05
            hb = kf.BasisHead(*ki.C_norm(ckb), ckb["d_model"], ckb["n_pos"],
                              ckb["stats"], hp=ckb["hp"])
            hb.load_state_dict(stb)
            basis = os.path.join(td, "basis.pt")
            torch.save(dict(ckb, state=stb, state_sha1=kf.state_sha(hb),
                            selected="epoch1"), basis)
            ident = os.path.join(td, "identity.json")
            sys.argv = ["x", "--device", "cpu", "--basis", basis,
                        "--h18-cache", cache, "--out", ident,
                        "--gate-batches", "2"]
            assert ki.main() == 0
            gate = os.path.join(td, "k15a.json")
            open(gate, "w").write("{}")
            kdp.gate_paths = lambda device: (gate, gate)
            po, sel = ctx.parts_full["val_sel"][0]
            batch = ctx.build_batch(po, sel)
            ac = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
            with torch.no_grad():
                v_, p__ = ctx.model.build_inputs(position_offset=po, **batch)
                _q, _a, h_ = kf.plain_h18(
                    ctx.model, vlm_inputs_embeds=v_,
                    attention_mask=batch.get("attention_mask"),
                    position_ids=p__, decode=ctx.decode_fp32)
            Bn, n_pos, d_model = h_.shape
            import k16u_models as um
            C, _CtxHead = bt.build_heads(torch, d_model, n_pos)
            torch.manual_seed(0)
            g_state = C["GoalNet"]().state_dict()
            Pair = um.build(torch, d_model, n_pos)
            ck = dict(
                variants={m: dict(state=Pair(m, g_state).state_dict(),
                                  mode=m) for m in um.MODES},
                norms=dict(r_sigma=np.full(6, 0.05),
                           dpos_sigma=np.full(3, 0.1)),
                scale_arm=[0.9] * 6, alpha=0.5, clip_env_norm=0.3,
                units=km.UNITS, d_model=d_model, n_pos=n_pos)
            mp = os.path.join(td, "m.pt")
            torch.save(ck, mp)
            arrs = {}
            for lab in LABELS:
                arm = build_arm("cpu", basis, lab, torch, models_path=mp,
                                plan_sha1="P", identity_report=ident)
                for c in range(5):
                    kp.STATE_BUF.append(np.random.default_rng(c)
                                        .standard_normal((Bn, 8)) * 0.1)
                    arm.act(batch, po, ac, c == 0)
                arrs[lab] = arm.log.take()[1]
            for lab, a in arrs.items():
                lv, a0 = a[f"k15d_level_{lab}"], a["k15d_a0"]
                imp = a["k16_impulse"]
                assert (imp.sum(0) <= 1).all()
                assert np.array_equal(lv[~imp], a0[~imp])
                assert np.array_equal(lv[..., 6], a0[..., 6])
                assert np.array_equal(a0[0], arrs["q0"]["k15d_a0"][0])
                for k, b in zip(*np.nonzero(imp)):
                    assert np.allclose(lv[k, b, :, :6] - a0[k, b, :, :6],
                                       a["k16_d_codec"][k, b], atol=1e-6)
                if lab == "q0":
                    assert not imp.any()
    finally:
        k15_context.build, sys.argv, kdp.gate_paths = saved
    print("интеграция k16u_policy пройдена: q0 = a0, один импульс, схват из "
          "a0, поправка = журнал δ")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="K-16u: рука")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--integration", action="store_true")
    a = ap.parse_args()
    if a.integration:
        sys.exit(integration())
    if a.selftest:
        sys.exit(selftest())
    raise SystemExit("модуль руки; запуск — через k16u_harness.py")
