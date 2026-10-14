#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16: рука — один импульс корректирующей головы на эпизод.

ПОЛИТИКИ:
  q0                    без вмешательства (журнал событий и h18 всех вызовов);
  direct_hist_event     прямая голова (история + скрытый вектор goal_hist);
  goal_flat_hist_event  цель goal_hist -> goal_flat;
  goal_band_hist_event  цель goal_hist -> goal_band (основной кандидат);
  goal_band_now_event   цель goal_now -> goal_band;
  goal_band_hist_time   те же веса, что goal_band_hist_event; момент —
                        заранее из таблицы доноров (события q0 других
                        состояний той же задачи).

МОМЕНТ «event»: первый вызов, в восьми командах a0 которого знак схвата
(в единицах исполнения: −a0) отличается от предыдущей ИСПОЛНЕННОЙ команды
этой среды (в вызове 1 — от первого ненулевого знака самого чанка; ноль
сохраняет знак) — та же причинная функция, что в паспорте. done_step не
используется. Не более одного импульса на среду.

ПОПРАВКА: только в вызове импульса; k16_models.apply_correction (единицы и
предел — из чекпойнта): рука a0 + δ, схват и хвост чанка — a0. В
остальных вызовах и у остальных сред действие ПОБИТОВО равно a0.

СОСТОЯНИЕ РУКИ: наблюдаемое (pos, axis-angle, пальцы) — из обёртки
utils.process_state, которую ставит адаптер (k16_harness): она возвращает
тот же результат и копирует его в STATE_BUF. История — состояния
предыдущих вызовов (8, 16, 24 команды назад) с маской; возраст — 8·(вызов−1).
Номер старта, будущий успех и состояния предметов головам не передаются.
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

H = 8
ACTION_CLIP_BOUND = 1.5
# метка -> (голова, источник цели, признаки состояния, расписание)
POLICIES = {
    "q0": (None, None, "sf_hist", None),
    "direct_hist_event": ("direct", "feat_goal_hist", "sf_hist", "event"),
    "goal_flat_hist_event": ("goal_flat", "pred_goal_hist", "sf_hist",
                             "event"),
    "goal_band_hist_event": ("goal_band", "pred_goal_hist", "sf_hist",
                             "event"),
    "goal_band_now_event": ("goal_band_now", "pred_goal_now", "sf_now",
                            "event"),
    "goal_band_hist_time": ("goal_band", "pred_goal_hist", "sf_hist",
                            "time"),
}
LABELS = tuple(POLICIES)
STATE_BUF = []                     # заполняет обёртка process_state


def sign_seq(x, prev=0):
    out = []
    for v in np.asarray(x, np.float64):
        s = 1 if v > 0 else -1 if v < 0 else prev
        out.append(s)
        prev = s
    return np.asarray(out)


def event_now(a0_grip_exec, ref):
    """Есть ли смена знака относительно ref в чанке (единицы исполнения)."""
    return bool(ref != 0 and (sign_seq(a0_grip_exec, prev=ref) != ref).any())


def first_ref(a0_grip_exec):
    s = sign_seq(a0_grip_exec)
    nz = np.nonzero(s)[0]
    return int(s[nz[0]]) if len(nz) else 0


class Tracker:
    """Состояние каждой среды: история, последняя исполненная команда,
    первое событие, использован ли импульс."""

    def __init__(self, B):
        self.B = B
        self.hist = [collections.deque(maxlen=3) for _ in range(B)]
        self.last = [0] * B
        self.first_event = [-1] * B
        self.used = [False] * B
        self.call = 0

    def step(self, state, a0_exec_grip, schedule, donor_calls):
        """state [B, 8]; a0_exec_grip [B, 8]. Возвращает (событие сейчас
        [B], импульс сейчас [B], история [B, 3, 8], маска [B, 3])."""
        self.call += 1
        c = self.call
        ev, trig = np.zeros(self.B, bool), np.zeros(self.B, bool)
        hist = np.zeros((self.B, 3, 8))
        mask = np.zeros((self.B, 3), bool)
        for b in range(self.B):
            for k, s_ in enumerate(reversed(self.hist[b])):
                hist[b, k] = s_
                mask[b, k] = True
            ref = first_ref(a0_exec_grip[b]) if c == 1 else self.last[b]
            ev[b] = event_now(a0_exec_grip[b], ref)
            if ev[b] and self.first_event[b] < 0:
                self.first_event[b] = c
            if self.used[b] or schedule is None:
                continue
            if schedule == "event" and ev[b] and self.first_event[b] == c:
                trig[b] = True
            elif schedule == "time" and donor_calls[b] == c:
                trig[b] = True
            if trig[b]:
                self.used[b] = True
        return ev, trig, hist, mask

    def after(self, state, exec_grip, ref_carry=True):
        """После выбора действия: история и последняя исполненная команда
        (exec_grip [B, 8] — первые 8 команд в единицах исполнения)."""
        for b in range(self.B):
            self.hist[b].append(np.asarray(state[b], np.float64))
            prev = self.last[b] if self.call > 1 else first_ref(exec_grip[b])
            self.last[b] = int(sign_seq(exec_grip[b], prev=prev)[-1])


class K16Log:
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
        arr = {}
        for k, v in self.rows.items():
            arr[k] = np.stack(v)
        arr.update(self.extra)
        lab = self.label
        out = {"k15d_a0": arr.pop("a0"), f"k15d_level_{lab}": arr.pop("act"),
               "k15g_chunk": arr.pop("chunk")}
        out.update({f"k16_{k}": v for k, v in arr.items()})
        summ = dict(calls=int(len(out["k15g_chunk"])), levels=[lab],
                    impulses=int(out["k16_impulse"].sum()))
        self.reset()
        return summ, out


def load_models(path, torch, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    C = km.build(torch, d_model=int(ck.get("d_model", 768)),
                 n_pos=int(ck.get("n_pos", 16)))
    goals = {}
    for k, v in ck["goals"].items():
        m = C["GoalNet"]()
        m.load_state_dict(v["state"])
        goals[k] = m.to(dev).eval()
    heads = {}
    for k, v in ck["heads"].items():
        m = C[v["cls"]](v["hidden"])
        m.load_state_dict(v["state"])
        heads[k] = m.to(dev).eval()
    return ck, goals, heads


def build_arm(device, basis, label, torch, *, models_path, plan_sha1,
              donor_calls=None, identity_report=None):
    """donor_calls — момент (номер вызова) для сред блока (только time)."""
    import k15_context
    import k15d_policy as kdp
    import k15f_build_cache as kb
    import k15f_check_identity as ki
    import k15f_continuous_refine as kf
    import k15f_policy as kfp
    if label not in POLICIES:
        raise SystemExit(f"рука k16: метка {label!r}")
    head_k, goal_k, sf_k, sched = POLICIES[label]
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
        raise SystemExit("рука K-16 не собрана: " + "; ".join(prob))
    ctx = k15_context.build(kdp.context_namespace(device, g15a))
    model, dev = ctx.model, ctx.dev
    k15t = k15_context.k15t
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    frozen, _n, _e = k15t.frozen_content_sha(model, torch, set())
    if frozen != rep.get("frozen_sha1"):
        raise SystemExit("замороженное не то, что в гейте тождества")
    ck, goals, heads = load_models(models_path, torch, dev)
    r_sigma = np.asarray(ck["norms"]["r_sigma"], np.float64)
    dpos_sigma = np.asarray(ck["norms"]["dpos_sigma"], np.float64)
    scale = np.asarray(ck["scale_arm"] + [1.0], np.float64)
    alpha, clip = float(ck["alpha"]), float(ck["clip_env_norm"])
    if sched == "time" and donor_calls is None:
        raise SystemExit("рука k16: для time нужна таблица доноров")
    log = K16Log(label)
    st = dict(tr=None, nbuf=0)

    def act(batch, pos_off, autocast, first):
        if first:
            st["nbuf"] = len(STATE_BUF) - 1
        # СОСТОЯНИЕ ТЕКУЩЕГО ВЫЗОВА: ровно одна новая запись обёртки
        if len(STATE_BUF) != st["nbuf"] + 1:
            raise SystemExit("рука k16: нет состояния текущего вызова "
                             "(обёртка process_state не установлена?)")
        st["nbuf"] = len(STATE_BUF)
        state = np.asarray(STATE_BUF[-1], np.float64)
        with torch.no_grad(), autocast:
            v, p_ = model.build_inputs(position_offset=pos_off, **batch)
            q0, a0, h18 = kf.plain_h18(
                model, vlm_inputs_embeds=v,
                attention_mask=batch.get("attention_mask"), position_ids=p_,
                decode=ctx.decode_fp32)
        a0_np = a0.float().cpu().numpy()
        B = a0_np.shape[0]
        if first:
            st["tr"] = Tracker(B)
        tr = st["tr"]
        if state.shape != (B, 8):
            raise SystemExit(f"рука k16: состояние формы {state.shape}")
        ev, trig, hist, mask = tr.step(
            state, -a0_np[:, :H, 6], sched,
            donor_calls if donor_calls is not None else [-1] * B)
        age = np.full(B, 8 * (tr.call - 1))
        sf = km.state_features(state, hist, mask, age, sf_k == "sf_hist")
        a = a0_np.copy()
        nan = lambda *s: np.full(s, np.nan, np.float32)   # noqa: E731
        goal_log, gm_log = nan(B, km.GOAL_DIM), nan(B, 3)
        out_log, dc_log = nan(B, H, 6), nan(B, H, 6)
        nraw, napp, clp = nan(B), nan(B), np.zeros(B, bool)
        lo_n, hi_n = nan(B), nan(B)
        t0 = time.perf_counter()
        if trig.any():
            idx = np.nonzero(trig)[0]
            with torch.no_grad():
                h_ = h18[idx]
                a0_t = a0[idx, :H].float()
                sf_t = torch.as_tensor(sf[idx], device=dev)
                gnet = goals["goal_now" if goal_k == "pred_goal_now"
                             else "goal_hist"]
                hid = gnet.hidden(h_, a0_t, sf_t)
                g_out = gnet.body[-1](hid)
                goal_in = hid if goal_k == "feat_goal_hist" else g_out
                hm = heads[head_k]
                r_out, _aux = hm(h_, a0_t, sf_t, goal_in)
                if hasattr(hm, "parts"):
                    lo, hi, _c = hm.parts(h_, a0_t, sf_t, goal_in)
                    rs = torch.as_tensor(r_sigma, device=dev,
                                         dtype=torch.float32)
                    lo_n[idx] = (lo * rs).flatten(1).norm(dim=1).cpu()
                    hi_n[idx] = (hi * rs).flatten(1).norm(dim=1).cpu()
            r_np = r_out.float().cpu().numpy()
            a_new, info = km.apply_correction(a0_np[idx], r_np, r_sigma,
                                              scale, alpha, clip)
            a[idx] = a_new
            g_np = g_out.float().cpu().numpy()
            goal_log[idx] = g_np
            gm_log[idx] = g_np[:, :3] * dpos_sigma
            out_log[idx] = r_np
            dc_log[idx] = info["d_codec"]
            nraw[idx], napp[idx] = info["norm_raw"], info["norm_applied"]
            clp[idx] = info["clipped"]
        head_ms = (time.perf_counter() - t0) * 1e3
        keep = ~trig
        if not np.array_equal(a[keep], a0_np[keep]):
            raise SystemExit("рука k16: без импульса действие не равно a0")
        if not np.isfinite(a).all():
            raise SystemExit("рука k16: действие не конечно")
        if float(np.abs(a[:, :H]).max()) > ACTION_CLIP_BOUND:
            raise SystemExit("рука k16: |действие| выше предела")
        tr.after(state, -a[:, :H, 6])
        if first:
            print(f"    проверка k16: {label}: расписание {sched}, голова "
                  f"{head_k}, цель {goal_k}", flush=True)
        if trig.any():
            print(f"    k16: импульс в вызове {tr.call} у сред "
                  f"{np.nonzero(trig)[0].tolist()}, ||δ|| исп. "
                  f"{np.nanmax(napp):.4f}", flush=True)
        is_q0 = label == "q0"
        log.add(a0=a0_np[:, :H].astype(np.float32),
                act=a[:, :H].astype(np.float32), chunk=tr.call,
                state=state.astype(np.float32), hist_mask=mask,
                event=ev, impulse=trig, goal=goal_log, goal_m=gm_log,
                out=out_log, d_codec=dc_log, norm_raw=nraw,
                norm_applied=napp, clipped=clp, band_low_norm=lo_n,
                band_high_norm=hi_n, head_ms=head_ms, sf=sf,
                h18=(h18.to(torch.float16).cpu().numpy() if is_q0 or
                     trig.any() else np.zeros((B,) + tuple(h18.shape[1:]),
                                              np.float16)))
        log.extra["first_event"] = np.asarray(tr.first_event, np.int64)
        log.extra["donor_calls"] = np.asarray(
            donor_calls if donor_calls is not None else [-1] * B, np.int64)
        return a.astype(np.float32), q0.cpu().numpy()

    meta = dict(
        arm="k16", label=label, head=head_k, goal_source=goal_k,
        state_features=sf_k, schedule=sched, plan_sha1=plan_sha1,
        models=os.path.abspath(models_path),
        models_sha1=kb.sha_file(models_path), alpha=alpha,
        clip_env_norm=clip, units=ck.get("units"),
        donor_calls=(list(map(int, donor_calls)) if donor_calls is not None
                     else None),
        coeffs=[0.0] * 4, control=False,
        basis=os.path.abspath(basis), basis_sha1=basis_sha,
        basis_state_sha1=ck_b["state_sha1"], stats_sha1=ck_b["stats_sha1"],
        amp_factor=float(ck_b["amp_factor"]),
        control_seed=int(ck_b["control_seed"]),
        control_kind=ck_b.get("control_kind"),
        identity_report=os.path.abspath(identity_report),
        identity_report_sha1=kb.sha_file(identity_report),
        refine_module_sha1=kb.sha_file(kf.__file__),
        k15f_policy_sha1=kb.sha_file(kfp.__file__),
        k16_policy_sha1=kb.sha_file(os.path.abspath(__file__)),
        k16_models_sha1=kb.sha_file(km.__file__),
        frozen_content_sha=frozen, joint_sha1=ctx.joint_sha,
        code_version=ctx.code_version, codec=ctx.codec_fp,
        k15a_gate=ctx.gate_info, layers_per_call=18, preflight=False,
        admission_override=None, decodes_per_call=1)
    meta = json.loads(json.dumps(meta, default=k15t.json_scalar))
    meta["model_fingerprint"] = hashlib.sha1("|".join(str(meta[k]) for k in (
        "label", "models_sha1", "k16_policy_sha1", "k16_models_sha1",
        "plan_sha1", "frozen_content_sha", "joint_sha1", "basis_sha1",
        "donor_calls")).encode()).hexdigest()[:12]
    return SimpleNamespace(model=model, proc=ctx.proc, codec=ctx.codec,
                           act=act, log=log, meta=meta, ctx=ctx)


# --- самопроверки -------------------------------------------------------------
def selftest():
    # событие: как в паспорте
    assert first_ref([0, 0, -1, -1]) == -1
    assert event_now(np.array([-1, -1, 1, 1]), -1)
    assert not event_now(np.array([-1, 0, 0, -1]), -1)
    assert not event_now(np.array([1, 1]), 0)
    # трекер: одно событие, один импульс, история и сброс
    tr = Tracker(3)
    grips = [np.full((3, 8), -1.0) for _ in range(6)]
    grips[2][0, 4:] = 1.0                  # среда 0: событие в вызове 3
    for c_ in (3, 4, 5):
        grips[c_][0, :] = 1.0              # дальше схват остаётся закрытым
    grips[4][1, :] = 1.0                   # среда 1: событие в вызове 5
    trig_calls, ev_calls = [], []
    for c in range(6):
        st_ = np.full((3, 8), float(c))
        ev, trig, hist, mask = tr.step(st_, grips[c], "event", [-1] * 3)
        trig_calls.append(trig.copy())
        ev_calls.append(ev.copy())
        if c == 3:
            assert mask[0].tolist() == [True, True, True]
            assert hist[0, 0, 0] == 2.0 and hist[0, 2, 0] == 0.0
        if c == 0:
            assert not mask.any()
        tr.after(st_, grips[c])
    tc = np.array(trig_calls)
    assert tc[:, 0].tolist() == [0, 0, 1, 0, 0, 0]
    assert tc[:, 1].tolist() == [0, 0, 0, 0, 1, 0]
    assert not tc[:, 2].any()
    assert tr.first_event == [3, 5, -1]
    # после исполненной смены знака событие уже не повторяется
    assert not np.array(ev_calls)[3:, 0].any()
    # время: импульс в назначенный вызов
    tr2 = Tracker(2)
    for c in range(5):
        _e, trig, _h, _m = tr2.step(np.zeros((2, 8)), np.full((2, 8), -1.0),
                                    "time", [3, 9])
        tr2.after(np.zeros((2, 8)), np.full((2, 8), -1.0))
        assert trig.tolist() == [c == 2, False]
    # q0: событий журналируются, импульсов нет
    tr3 = Tracker(1)
    g = np.full((1, 8), -1.0)
    g[0, 3:] = 1
    ev, trig, _h, _m = tr3.step(np.zeros((1, 8)), g, None, [-1])
    assert ev[0] and not trig[0]
    # метки и расписания
    assert POLICIES["goal_band_hist_time"][0] == \
        POLICIES["goal_band_hist_event"][0]
    assert len(LABELS) == 6
    print("самопроверка k16_policy пройдена: событие (как в паспорте), один "
          "импульс на среду, история и маска, время по донору, q0 без "
          "импульса")
    return 0


def integration():
    """Игрушечная модель: шесть политик; q0 побитово = a0; импульс один,
    в событийном вызове; поправка = apply_correction; h18/a0 до импульса
    одинаковы у всех политик."""
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
            # игрушечные модели K-16 в размерах игрушечного h18
            with torch.no_grad():
                v_, p__ = ctx.model.build_inputs(position_offset=po, **batch)
                _q, a0_, h_ = kf.plain_h18(
                    ctx.model, vlm_inputs_embeds=v_,
                    attention_mask=batch.get("attention_mask"),
                    position_ids=p__, decode=ctx.decode_fp32)
            B, n_pos, d_model = h_.shape
            C = km.build(torch, d_model=d_model, n_pos=n_pos)
            torch.manual_seed(0)
            _t, w = km.matched_widths(torch, C)
            ck = dict(
                goals={k: dict(state=C["GoalNet"]().state_dict(), sf=s)
                       for k, s in (("goal_now", "sf_now"),
                                    ("goal_hist", "sf_hist"))},
                heads={"direct": dict(state=C["Direct"](w["Direct"])
                                      .state_dict(), cls="Direct",
                                      hidden=w["Direct"]),
                       "goal_flat": dict(state=C["GoalFlat"](w["GoalFlat"])
                                         .state_dict(), cls="GoalFlat",
                                         hidden=w["GoalFlat"]),
                       "goal_band": dict(state=C["GoalBand"](w["GoalBand"])
                                         .state_dict(), cls="GoalBand",
                                         hidden=w["GoalBand"]),
                       "goal_band_now": dict(
                           state=C["GoalBand"](w["GoalBand"]).state_dict(),
                           cls="GoalBand", hidden=w["GoalBand"])},
                norms=dict(r_sigma=np.full(6, 0.05),
                           dpos_sigma=np.full(3, 0.1)),
                scale_arm=[0.9] * 6, alpha=0.5, clip_env_norm=0.3,
                units=km.UNITS, d_model=d_model, n_pos=n_pos)
            mp = os.path.join(td, "models.pt")
            torch.save(ck, mp)
            arrs = {}
            for lab in LABELS:
                arm = build_arm("cpu", basis, lab, torch, models_path=mp,
                                plan_sha1="P", donor_calls=[2] * B,
                                identity_report=ident)
                for c in range(5):
                    STATE_BUF.append(np.random.default_rng(c).standard_normal(
                        (B, 8)) * 0.1)
                    arm.act(batch, po, ac, c == 0)
                arrs[lab] = arm.log.take()[1]
            for lab, a in arrs.items():
                lv, a0 = a[f"k15d_level_{lab}"], a["k15d_a0"]
                imp = a["k16_impulse"]
                assert (imp.sum(0) <= 1).all(), lab
                assert np.array_equal(lv[~imp], a0[~imp]), lab
                assert np.array_equal(lv[..., 6], a0[..., 6]), lab
                if lab == "q0":
                    assert not imp.any()
                    assert a["k16_h18"].shape[:2] == (5, B)
                if lab == "goal_band_hist_time":
                    assert imp[1].all() and imp.sum() == B
                # h18 и a0 вызова 1 одинаковы у всех политик
                assert np.array_equal(a0[0], arrs["q0"]["k15d_a0"][0])
                k_, b_ = np.nonzero(imp)
                for k, b in zip(k_, b_):
                    d = lv[k, b, :, :6] - a0[k, b, :, :6]
                    assert np.allclose(d, a["k16_d_codec"][k, b], atol=1e-6)
                    assert a["k16_norm_applied"][k, b] <= 0.3 + 1e-6
    finally:
        k15_context.build, sys.argv, kdp.gate_paths = saved
    print("интеграция k16_policy пройдена: q0 = a0, один импульс, схват из "
          "a0, поправка = журнал δ, предел нормы, время по донору")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="K-16: рука")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--integration", action="store_true")
    a = ap.parse_args()
    if a.integration:
        sys.exit(integration())
    if a.selftest:
        sys.exit(selftest())
    raise SystemExit("модуль руки; запуск — через k16_harness.py")
