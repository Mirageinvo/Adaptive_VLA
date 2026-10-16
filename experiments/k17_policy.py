#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-17: рука — одно окно из трёх последовательных коррекций.

Метки: q0, ctx_memory, held_goal (у каждой модели своя пара G+C из
data/k17/models.pt). Начало окна — прежнее причинное событие (первая смена
команды схвата в a0 относительно последней исполненной, Tracker K-16).
В событийном вызове G один раз предсказывает цель по текущему наблюдению;
цель переводится в мировую позу (p*, R*) и вместе со скрытым вектором
сохраняется. В этом и двух следующих вызовах C получает ТЕКУЩИЕ h18, a0,
состояние и ошибку до сохранённой цели (у ctx_memory — нули) и вычисляет
новую поправку; затем — q0. Поправка применяется apply_correction K-16
(α и предел — из чекпойнта) отдельно к каждому чанку, один раз; схват и
хвост — из a0.
"""
import argparse
import collections
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
import k17_models as m17  # noqa: E402

H = 8
POLICIES = {"q0": None, "ctx_memory": "ctx_memory",
            "held_goal": "held_goal"}
LABELS = tuple(POLICIES)


class K17Log:
    def __init__(self, label):
        self.label = label
        self.reset()

    def reset(self):
        self.rows = collections.defaultdict(list)
        self.extra = {}

    def add(self, **kw):
        for k, v in kw.items():
            self.rows[k].append(np.asarray(v))

    def take(self):
        if not self.rows:
            self.reset()
            return dict(calls=0), {}
        arr = {k: np.stack(v) for k, v in self.rows.items()}
        arr.update(self.extra)
        out = {"k15d_a0": arr.pop("a0"),
               f"k15d_level_{self.label}": arr.pop("act"),
               "k15g_chunk": arr.pop("chunk")}
        out.update({f"k17_{k}": v for k, v in arr.items()})
        summ = dict(calls=int(len(out["k15g_chunk"])), levels=[self.label],
                    corrections=int(out["k17_corrected"].sum()))
        self.reset()
        return summ, out


def load_models(path, torch, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("kind") != "k17_models":
        raise SystemExit(f"{path}: не модели K-17")
    M = m17.build(torch, int(ck["d_model"]), int(ck["n_pos"]))
    ds = np.asarray(ck["norms"]["dpos_sigma"], np.float32)
    pairs = {}
    for k, v in ck["pairs"].items():
        p = M["Pair"](v["kind"], ds)
        p.load_state_dict(v["state"])
        pairs[k] = p.to(dev).eval()
    return ck, pairs


class Window:
    """Позиция окна каждой среды: −1 — не начато, 0..2 — в окне, 3 —
    завершено. Сохранённые мировая цель и скрытый вектор."""

    def __init__(self, B):
        self.pos = np.full(B, -1)
        self.start_call = np.full(B, -1)
        self.hid = None
        self.p_star = np.full((B, 3), np.nan, np.float32)
        self.R_star = np.full((B, 3, 3), np.nan, np.float32)
        self.goal = np.full((B, m17.G_OUT), np.nan, np.float32)

    def step(self, trig, call):
        """trig — начало окна сейчас. Возвращает (новые [B], активные
        [B], позиция [B])."""
        new = np.asarray(trig, bool) & (self.pos < 0)
        self.pos[(self.pos >= 0) & (self.pos < 3)] += 1
        self.pos[new] = 0
        self.start_call[new] = call
        self.pos[self.pos > 3] = 3
        act = (self.pos >= 0) & (self.pos <= 2)
        return new, act, np.where(act, self.pos, -1)


def build_arm(device, basis, label, torch, *, models_path, plan_sha1,
              identity_report=None):
    import k15_context
    import k15d_policy as kdp
    import k15f_build_cache as kb
    import k15f_check_identity as ki
    import k15f_continuous_refine as kf
    import k15f_policy as kfp
    if label not in POLICIES:
        raise SystemExit(f"рука k17: метка {label!r}")
    kind = POLICIES[label]
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
        raise SystemExit("рука K-17 не собрана: " + "; ".join(prob))
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
    scale = np.asarray(list(ck["scale_arm"]) + [1.0], np.float64)
    alpha, clip = float(ck["alpha"]), float(ck["clip_env_norm"])
    log = K17Log(label)
    st = dict(tr=None, win=None, nbuf=0)
    T = lambda x: torch.as_tensor(np.asarray(x, np.float32),  # noqa: E731
                                  device=dev)

    def act(batch, pos_off, autocast, first):
        if first:
            st["nbuf"] = len(kp.STATE_BUF) - 1
        if len(kp.STATE_BUF) != st["nbuf"] + 1:
            raise SystemExit("рука k17: нет состояния текущего вызова")
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
            st["tr"], st["win"] = kp.Tracker(B), Window(B)
        tr, win = st["tr"], st["win"]
        if state.shape != (B, 8):
            raise SystemExit(f"рука k17: состояние формы {state.shape}")
        ev, trig, hist, mask = tr.step(state, -a0_np[:, :H, 6],
                                       "event" if kind else None, [-1] * B)
        new, active, wpos = win.step(trig, tr.call)
        sf = km.state_features(state, hist, mask,
                               np.full(B, 8 * (tr.call - 1)), True)
        p_cur = state[:, :3]
        R_cur = km.aa_to_mat_np(state[:, 3:6])
        a = a0_np.copy()
        nan = lambda *s: np.full(s, np.nan, np.float32)   # noqa: E731
        geo_log, out_log, dc_log = nan(B, 9), nan(B, H, 6), nan(B, H, 6)
        nraw, napp, clp = nan(B), nan(B), np.zeros(B, bool)
        t0 = time.perf_counter()
        if active.any():
            pr = pairs[kind]
            with torch.no_grad():
                if new.any():
                    i = np.nonzero(new)[0]
                    s_ = pr.start(h18[i], a0[i, :H].float(), T(sf[i]),
                                  T(p_cur[i]), T(R_cur[i]))
                    if win.hid is None:
                        win.hid = torch.zeros((B, s_["hid"].shape[1]),
                                              device=dev)
                    win.hid[i] = s_["hid"].float()
                    win.p_star[i] = s_["p_star"].float().cpu().numpy()
                    win.R_star[i] = s_["R_star"].float().cpu().numpy()
                    win.goal[i] = s_["goal"].float().cpu().numpy()
                i = np.nonzero(active)[0]
                r_t, _aux, geo = pr.correct(
                    h18[i], a0[i, :H].float(), T(sf[i]), win.hid[i],
                    torch.as_tensor(wpos[i], device=dev),
                    T(win.p_star[i]), T(win.R_star[i]), T(p_cur[i]),
                    T(R_cur[i]))
            r_np = r_t.float().cpu().numpy()
            a_new, info = km.apply_correction(a0_np[i], r_np, r_sigma, scale,
                                              alpha, clip)
            a[i] = a_new
            geo_log[i] = geo.float().cpu().numpy()
            out_log[i], dc_log[i] = r_np, info["d_codec"]
            nraw[i], napp[i] = info["norm_raw"], info["norm_applied"]
            clp[i] = info["clipped"]
        head_ms = (time.perf_counter() - t0) * 1e3
        if not np.array_equal(a[~active], a0_np[~active]):
            raise SystemExit("рука k17: вне окна действие не равно a0")
        if not np.array_equal(a[..., 6], a0_np[..., 6]) or \
                not np.array_equal(a[:, H:], a0_np[:, H:]):
            raise SystemExit("рука k17: схват или хвост изменены")
        if not np.isfinite(a).all():
            raise SystemExit("рука k17: действие не конечно")
        if float(np.abs(a[:, :H]).max()) > kp.ACTION_CLIP_BOUND:
            raise SystemExit("рука k17: |действие| выше предела")
        tr.after(state, -a[:, :H, 6])
        if first:
            print(f"    проверка k17: {label}: модель {kind}", flush=True)
        if active.any():
            print(f"    k17: вызов {tr.call}, позиции окна "
                  f"{dict(zip(np.nonzero(active)[0].tolist(), wpos[active].tolist()))}"
                  f", ||δ|| исп. {np.nanmax(napp):.4f}", flush=True)
        in_win = win.pos >= 0
        log.add(a0=a0_np[:, :H].astype(np.float32),
                act=a[:, :H].astype(np.float32), chunk=tr.call,
                state=state.astype(np.float32), hist_mask=mask, event=ev,
                trigger=trig, corrected=active, win_pos=wpos,
                p_star=np.where(in_win[:, None], win.p_star, np.nan),
                R_star=np.where(in_win[:, None, None], win.R_star, np.nan),
                goal=np.where(in_win[:, None], win.goal, np.nan),
                geo=geo_log, out=out_log, d_codec=dc_log, norm_raw=nraw,
                norm_applied=napp, clipped=clp, head_ms=head_ms, sf=sf,
                h18=(h18.to(torch.float16).cpu().numpy()
                     if label == "q0" or active.any() else
                     np.zeros((B,) + tuple(h18.shape[1:]), np.float16)))
        log.extra["first_event"] = np.asarray(tr.first_event, np.int64)
        log.extra["window_start"] = win.start_call.copy()
        return a.astype(np.float32), q0.cpu().numpy()

    meta = dict(
        arm="k17", label=label, model_kind=kind, schedule="window3",
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
        k17_policy_sha1=kb.sha_file(os.path.abspath(__file__)),
        k17_models_sha1=kb.sha_file(m17.__file__),
        k16_models_sha1=kb.sha_file(km.__file__),
        frozen_content_sha=frozen, joint_sha1=ctx.joint_sha,
        code_version=ctx.code_version, codec=ctx.codec_fp,
        k15a_gate=ctx.gate_info, layers_per_call=18, preflight=False,
        admission_override=None, decodes_per_call=1)
    meta = json.loads(json.dumps(meta, default=k15t.json_scalar))
    meta["model_fingerprint"] = hashlib.sha1("|".join(str(meta[k]) for k in (
        "label", "models_sha1", "k17_policy_sha1", "plan_sha1",
        "frozen_content_sha", "joint_sha1", "basis_sha1")).encode()
    ).hexdigest()[:12]
    return SimpleNamespace(model=model, proc=ctx.proc, codec=ctx.codec,
                           act=act, log=log, meta=meta, ctx=ctx)


def _toy_ckpt(torch, d_model=768, n_pos=16, seed=0):
    M = m17.build(torch, d_model, n_pos)
    ds = np.full(3, 0.1, np.float32)
    torch.manual_seed(seed)
    g = M["G"]().state_dict()
    pairs = {}
    for k in m17.KINDS:
        p = M["Pair"](k, ds, g, c_seed=seed)
        with torch.no_grad():                    # заметная поправка
            p.C.r.weight.mul_(20.0)
        pairs[k] = dict(state=p.state_dict(), kind=k, best_epoch=1)
    return dict(kind="k17_models", pairs=pairs, d_model=d_model,
                n_pos=n_pos, norms=dict(r_sigma=np.full(6, 0.05),
                                        dpos_sigma=ds),
                scale_arm=[0.9] * 6, alpha=0.5, clip_env_norm=0.3,
                units=km.UNITS)


def selftest():
    import tempfile
    import torch
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.pt")
        torch.save(_toy_ckpt(torch), p)
        _ck, pairs = load_models(p, torch, torch.device("cpu"))
    assert set(pairs) == set(m17.KINDS)
    assert pairs["ctx_memory"].G is not pairs["held_goal"].G
    w = Window(3)
    seq = []
    for c, tg in enumerate([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0],
                            [0, 0, 0], [1, 1, 1]], 1):
        n_, a_, p_ = w.step(np.asarray(tg, bool), c)
        seq.append(p_.tolist())
    assert seq == [[-1, -1, -1], [0, -1, -1], [1, 0, -1], [2, 1, -1],
                   [-1, 2, -1], [-1, -1, 0]], seq
    assert w.start_call.tolist() == [2, 3, 6]
    assert POLICIES["q0"] is None and len(LABELS) == 3
    print("самопроверка k17_policy пройдена: свои пары G+C, окно из трёх "
          "вызовов от события, одно окно на среду")
    return 0


def integration():
    """Игрушечная модель: q0 = a0; окно из трёх вызовов от первого
    события; поправка = журнал δ; мировая цель сохраняется; геометрия
    пересчитывается по текущему состоянию; ctx_memory — нули."""
    import tempfile
    import torch
    import k15_context
    import k15d_policy as kdp
    import k15d_train as kt
    import k15f_build_cache as kbc
    import k15f_check_identity as ki
    import k15f_continuous_refine as kf
    import k15f_pretrain_basis as kpb
    saved = (k15_context.build, sys.argv, kdp.gate_paths)
    ctx = kt._fake_ctx()
    ctx.model.action_expert.norm.variance_epsilon = 1e-6
    ctx.vocab = int(ctx.model.book0.shape[0])
    k15_context.build = lambda a: ctx
    n_corr = 0
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
            mp = os.path.join(td, "m.pt")
            ck = _toy_ckpt(torch, d_model, n_pos)
            torch.save(ck, mp)
            ds = np.asarray(ck["norms"]["dpos_sigma"], np.float64)
            arrs = {}
            # чередование знака схвата a0 через вызовы не управляется
            # игрушкой — события берутся какие есть; проверки структурные
            for lab in LABELS:
                arm = build_arm("cpu", basis, lab, torch, models_path=mp,
                                plan_sha1="P", identity_report=ident)
                for c in range(7):
                    kp.STATE_BUF.append(np.random.default_rng(c)
                                        .standard_normal((Bn, 8)) * 0.1)
                    arm.act(batch, po, ac, c == 0)
                arrs[lab] = arm.log.take()[1]
            for lab, a in arrs.items():
                lv, a0 = a[f"k15d_level_{lab}"], a["k15d_a0"]
                cor = a["k17_corrected"]
                fe = a["k17_first_event"]
                C = len(lv)
                for b in range(Bn):
                    calls = (np.nonzero(cor[:, b])[0] + 1).tolist()
                    want = ([c for c in (fe[b], fe[b] + 1, fe[b] + 2)
                             if c <= C] if fe[b] > 0 and lab != "q0"
                            else [])
                    assert calls == want, (lab, b, calls, fe[b])
                    if calls:
                        ps = a["k17_p_star"][np.asarray(calls) - 1, b]
                        assert (ps == ps[0]).all()
                        for c in calls:
                            st_ = a["k17_state"][c - 1, b].astype(np.float64)
                            e = m17.geo_error_np(
                                a["k17_p_star"][c - 1, b][None],
                                a["k17_R_star"][c - 1, b][None],
                                st_[None, :3],
                                km.aa_to_mat_np(st_[None, 3:6]), ds)[0]
                            g_ = a["k17_geo"][c - 1, b]
                            if lab == "ctx_memory":
                                assert not g_.any()
                            else:
                                assert np.allclose(g_, e, atol=1e-4)
                n_corr += int(cor.sum())
                assert np.array_equal(lv[~cor], a0[~cor])
                assert np.array_equal(lv[..., 6], a0[..., 6])
                assert np.array_equal(a0[0], arrs["q0"]["k15d_a0"][0])
                for k, b in zip(*np.nonzero(cor)):
                    assert np.allclose(lv[k, b, :, :6] - a0[k, b, :, :6],
                                       a["k17_d_codec"][k, b], atol=1e-6)
                if lab == "q0":
                    assert not cor.any()
                # валидатор протокола принимает журнал правильной руки
                import k17_protocol as pc17
                rules = dict(clip_env_norm=ck["clip_env_norm"],
                             scale_arm=ck["scale_arm"], dpos_sigma=list(ds),
                             r_sigma=list(ck["norms"]["r_sigma"]),
                             alpha=ck["alpha"])
                pr_ = pc17.check_arrays(lab, a, rules)
                assert pr_ == [], (lab, pr_)
    finally:
        k15_context.build, sys.argv, kdp.gate_paths = saved
    print(f"интеграция k17_policy пройдена: q0 = a0, окно из трёх вызовов "
          f"от первого события, мировая цель сохранена, геометрия по "
          f"текущему состоянию, нули у ctx_memory, схват из a0, поправка = "
          f"журнал δ (исправленных чанков {n_corr})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="K-17: рука")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--integration", action="store_true")
    a = ap.parse_args()
    if a.integration:
        sys.exit(integration())
    if a.selftest:
        sys.exit(selftest())
    raise SystemExit("модуль руки; запуск — через k17_harness.py")
