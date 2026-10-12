#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15h: пять политик одного импульса в чанке 4 на свежем банке.

МЕТКИ:
  q0ref            обычная политика, без вмешательства;
  p4r3p            точное воспроизведение кандидата K-15g: контроль (базис
                   learned, повёрнутый R), c = +e_3, чанк 4 — тем же путём
                   вычислений, что k15g_local_policy (c формы [4]);
  learned_uniform  базис learned, направление ±e_j среды — из таблицы плана;
  arm_dir_uniform  то же j и знак; рука — фиксированный шаблон U_j,arm с
                   нормой learned в координатах действия K-15f (σ_arm по 8
                   шагам), схват — из learned того же состояния;
  static_uniform   то же j и знак; полностью статичный базис f·ρ·U.

Коэффициенты uniform-вариантов — [B, 4]: у пяти сред блока направления
разные. Таблица направлений читается из плана (не выбирается заново).

ОДИН обычный проход модели и один декод на вызов (plain_h18); h18 берётся
из этого же прохода. Вне чанка 4 действие ПОБИТОВО равно текущему a0.

ЖУРНАЛ (npz блока через харнесс): a0 и действие (k15d_a0,
k15d_level_<метка>), learned-базис до преобразования (k15f_basis),
фактически использованный базис (k15h_basis_used), номер чанка и признак
импульса (k15g_chunk, k15g_active), запланированные и применённые
коэффициенты (k15h_coeff_planned [B, 4], k15h_coeff_applied [C, B, 4]),
h18 вызовов 1–4 ДО поправки — FP32-копия значений прохода
(k15h_h18 [≤4, B, n_pos, D]), исходный dtype — в сводке журнала.
"""
import argparse
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MOMENT = 4
LABELS = ("q0ref", "p4r3p", "learned_uniform", "arm_dir_uniform",
          "static_uniform")
UNIFORM = ("learned_uniform", "arm_dir_uniform", "static_uniform")
DIRS = [f"{j}{s}" for j in range(4) for s in ("p", "m")]
H18_CALLS = 4                     # h18 сохраняется для вызовов 1..4
ACTION_CLIP_BOUND = 1.5
H_EXEC, N_CH = 8, 7
ARM_IDX = [k * N_CH + c for k in range(H_EXEC) for c in range(6)]
GRIP_IDX = [k * N_CH + 6 for k in range(H_EXEC)]


def file_sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]


def dir_coeff(d):
    """'3p' -> [0, 0, 0, +1]."""
    if d not in DIRS:
        raise ValueError(f"направление {d!r}")
    c = [0.0] * 4
    c[int(d[0])] = 1.0 if d[1] == "p" else -1.0
    return c


def planned_coeffs(label, directions):
    """directions — список направлений сред блока (по порядку env). [B, 4]."""
    B = len(directions)
    if label == "q0ref":
        return np.zeros((B, 4), np.float32)
    if label == "p4r3p":
        return np.tile(np.asarray(dir_coeff("3p"), np.float32), (B, 1))
    if label in UNIFORM:
        return np.asarray([dir_coeff(d) for d in directions], np.float32)
    raise ValueError(f"метка {label!r}")


class Schedule:
    def __init__(self, moment):
        self.moment, self.call = moment, 0

    def step(self, first):
        self.call = 1 if first else self.call + 1
        return self.call, (self.moment is not None
                           and self.call == self.moment)


# --- преобразования базиса (Torch; эталон формул — k15h_static_basis) -------
def static_basis_t(U, rho, f, B, torch):
    b = (float(f) * rho[:, None] * U).float()            # [4, 56]
    return b[None].expand(B, -1, -1).contiguous()


def arm_dir_t(L, U, sigma_arm, torch):
    """L [B, 4, 56] learned (с амплитудой) -> arm_dir [B, 4, 56]."""
    arm = torch.tensor(ARM_IDX, device=L.device)
    grip = torch.tensor(GRIP_IDX, device=L.device)
    sig = sigma_arm.float().repeat(H_EXEC)                # [48]
    u_arm = U.float()[:, arm]                             # [4, 48]
    n_u = (u_arm * sig).norm(dim=-1)                      # [4]
    if bool((n_u < 1e-12).any()):
        raise SystemExit(f"рука k15h: у U нет части руки: {n_u.tolist()}")
    target = (L.float()[..., arm] * sig).norm(dim=-1)     # [B, 4]
    out = torch.zeros_like(L, dtype=torch.float32)
    out[..., arm] = (target / n_u)[..., None] * u_arm[None]
    out[..., grip] = L.float()[..., grip]
    return out


def arm_cos_t(L, U, sigma_arm, torch):
    """Косинус руки learned с рукой U_j: нормированные координаты и после
    умножения на σ. [B, 4] каждый."""
    arm = torch.tensor(ARM_IDX, device=L.device)
    sig = sigma_arm.float().repeat(H_EXEC)
    la, ua = L.float()[..., arm], U.float()[:, arm][None]

    def cos(x, y):
        return (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1)
                                  ).clamp_min(1e-30)
    return cos(la, ua), cos(la * sig, ua * sig)


class H15Log:
    def __init__(self, label, h_exec):
        self.label, self.h = label, int(h_exec)
        self.reset()

    def reset(self):
        self.a0, self.act, self.learned, self.used = [], [], [], []
        self.chunk, self.active, self.applied, self.h18 = [], [], [], []
        self.cos_n, self.cos_s = [], []
        self.planned = None
        self.h18_dtype = None

    def add(self, a0, act, learned, used, chunk, active, applied, planned,
            h18=None, h18_dtype=None, cos=None):
        self.a0.append(np.asarray(a0[:, :self.h], np.float32))
        self.act.append(np.asarray(act[:, :self.h], np.float32))
        self.learned.append(np.asarray(learned, np.float32))
        self.used.append(np.asarray(used, np.float32))
        self.chunk.append(int(chunk))
        self.active.append(bool(active))
        self.applied.append(np.asarray(applied, np.float32))
        self.planned = np.asarray(planned, np.float32)
        if h18 is not None:
            self.h18.append(np.asarray(h18, np.float32))
            self.h18_dtype = str(h18_dtype)
        if cos is not None:
            self.cos_n.append(np.asarray(cos[0], np.float32))
            self.cos_s.append(np.asarray(cos[1], np.float32))

    def take(self):
        if not self.a0:
            self.reset()
            return dict(calls=0), {}
        a0, act = np.stack(self.a0), np.stack(self.act)
        arrays = {"k15d_a0": a0, f"k15d_level_{self.label}": act,
                  "k15f_basis": np.stack(self.learned),
                  "k15h_basis_used": np.stack(self.used),
                  "k15g_chunk": np.asarray(self.chunk, np.int64),
                  "k15g_active": np.asarray(self.active, bool),
                  "k15h_coeff_planned": self.planned,
                  "k15h_coeff_applied": np.stack(self.applied),
                  "k15h_arm_cos_norm": np.stack(self.cos_n),
                  "k15h_arm_cos_sigma": np.stack(self.cos_s)}
        if self.h18:
            arrays["k15h_h18"] = np.stack(self.h18)
        summ = dict(calls=int(len(self.chunk)), levels=[self.label],
                    active_calls=int(sum(self.active)),
                    h18_calls=len(self.h18), h18_source_dtype=self.h18_dtype,
                    h18_shape=(list(self.h18[0].shape) if self.h18 else None),
                    **{f"finite_{self.label}": bool(np.isfinite(act).all())})
        self.reset()
        return summ, arrays


def build_arm(device, basis, label, torch, *, directions, plan_sha1,
              identity_report=None):
    """directions — направления сред блока по порядку env (из плана)."""
    import k15_context
    import k15d_policy as kdp
    import k15f_build_cache as kb
    import k15f_check_identity as ki
    import k15f_continuous_refine as kf
    import k15f_policy as kfp
    if label not in LABELS:
        raise SystemExit(f"рука k15h: метка {label!r}")
    planned = planned_coeffs(label, directions)
    moment = None if label == "q0ref" else MOMENT
    tag = str(device).replace(":", "")
    identity_report = identity_report or f"reports/k15f/identity_{tag}.json"
    g15a, _g = kdp.gate_paths(device)
    for f_ in (basis, identity_report, g15a):
        if not os.path.exists(f_):
            raise SystemExit(f"нет {f_}")
    basis_sha = kb.sha_file(basis)
    rep = json.load(open(identity_report))
    prob = kfp.check_identity_report(rep, basis_sha=basis_sha, device=device,
                                     code=kb.sha_file(kf.__file__),
                                     gate_sha=kb.sha_file(ki.__file__))
    ck = torch.load(basis, map_location="cpu", weights_only=False)
    prob += ki.check_basis_ckpt(ck)
    if prob:
        raise SystemExit("рука K-15h не собрана: " + "; ".join(prob))
    ctx = k15_context.build(kdp.context_namespace(device, g15a))
    model, dev = ctx.model, ctx.dev
    k15t = k15_context.k15t
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    frozen, _n, _e = k15t.frozen_content_sha(model, torch, set())
    if frozen != rep.get("frozen_sha1"):
        raise SystemExit("замороженное не то, что в гейте тождества")
    head = ki.load_head(torch, ck, model, dev)
    f = float(ck["amp_factor"])
    R = kf.rotation(int(ck["control_seed"])).to(dev)
    U, rho = head.U, head.rho
    c_vec = torch.tensor(dir_coeff("3p"), device=dev)     # путь K-15g
    c_env = torch.tensor(planned, device=dev)             # [B, 4]
    zero_c = torch.zeros(kf.K_BASIS, device=dev)
    sched = Schedule(moment)
    log = H15Log(label, kf.H_EXEC)

    def act(batch, pos_off, autocast, first):
        chunk, active = sched.step(first)
        with torch.no_grad(), autocast:
            v, p_ = model.build_inputs(position_offset=pos_off, **batch)
            q0, a0, h18 = kf.plain_h18(
                model, vlm_inputs_embeds=v,
                attention_mask=batch.get("attention_mask"), position_ids=p_,
                decode=ctx.decode_fp32)
        B = a0.shape[0]
        if B != len(directions):
            raise SystemExit(f"рука k15h: сред {B}, направлений в плане "
                             f"{len(directions)}")
        with torch.no_grad():
            basis_, _u = head(h18, a0)
            basis_ = basis_ * f                            # learned
            learned = basis_
            if label == "p4r3p":
                used = basis_ @ R
                cc = c_vec if active else zero_c
            elif label == "arm_dir_uniform":
                used = arm_dir_t(learned, U, head.sigma_arm, torch)
                cc = c_env if active else zero_c
            elif label == "static_uniform":
                used = static_basis_t(U, rho, f, B, torch).to(dev)
                cc = c_env if active else zero_c
            else:                                          # learned, q0ref
                used = learned
                cc = c_env if (active and label != "q0ref") else zero_c
            a = kf.combine(a0, kf.compose(used, cc), head.sigma_arm,
                           head.sigma_g)
            applied = (cc[None].expand(B, -1) if cc.ndim == 1 else cc)
            sv = torch.linalg.svdvals(learned)
            cond = float((sv[:, 0] / sv[:, -1].clamp_min(1e-12)).max())
            un = learned / learned.norm(dim=-1, keepdim=True).clamp_min(
                1e-12)
            gram = (un @ un.transpose(1, 2)).abs()
            gram = gram - torch.diag_embed(torch.diagonal(gram, dim1=1,
                                                          dim2=2))
            pc = float(gram.max())
            cos = arm_cos_t(learned, U, head.sigma_arm, torch)
        if cond > kf.GEOMETRY["cond_max"] or pc > kf.GEOMETRY["pair_cos_max"]:
            raise SystemExit(f"рука k15h: геометрия базиса вне пределов "
                             f"(обусловленность {cond:.2f}, косинус {pc:.3f})")
        if not active and not torch.equal(a, a0):
            raise SystemExit(f"рука k15h: чанк {chunk} не активен, но "
                             f"действие не равно a0")
        if not bool(torch.isfinite(a).all()):
            raise SystemExit("рука k15h: действие не конечно")
        amax = float(a[:, :kf.H_EXEC].abs().max())
        if amax > ACTION_CLIP_BOUND:
            raise SystemExit(f"рука k15h: |действие| {amax:.3f} выше "
                             f"{ACTION_CLIP_BOUND}")
        if first:
            print(f"    проверка k15h: {label}: импульс в чанке {moment}, "
                  f"направления сред {list(directions)}", flush=True)
        if active:
            print(f"    k15h: импульс исполнен в чанке {chunk}, max|a-a0| "
                  f"{float((a - a0).abs().max()):.4f}", flush=True)
        keep_h18 = chunk <= H18_CALLS
        log.add(a0.cpu().numpy(), a.cpu().numpy(),
                learned.float().cpu().numpy(), used.float().cpu().numpy(),
                chunk, active, applied.float().cpu().numpy(), planned,
                h18=(h18.float().cpu().numpy() if keep_h18 else None),
                h18_dtype=(h18.dtype if keep_h18 else None),
                cos=[x.cpu().numpy() for x in cos])
        return a.float().cpu().numpy(), q0.cpu().numpy()

    meta = dict(
        arm="k15h", label=label, moment=moment, pulse_chunks=1,
        directions=list(directions), coeffs_planned=planned.tolist(),
        plan_sha1=plan_sha1, control=(label == "p4r3p"),
        basis=os.path.abspath(basis), basis_sha1=basis_sha,
        basis_state_sha1=ck["state_sha1"], stats_sha1=ck["stats_sha1"],
        amp_factor=f, control_seed=int(ck["control_seed"]),
        control_kind=ck.get("control_kind"),
        identity_report=os.path.abspath(identity_report),
        identity_report_sha1=kb.sha_file(identity_report),
        refine_module_sha1=kb.sha_file(kf.__file__),
        k15f_policy_sha1=kb.sha_file(kfp.__file__),
        k15h_policy_sha1=kb.sha_file(os.path.abspath(__file__)),
        frozen_content_sha=frozen, joint_sha1=ctx.joint_sha,
        code_version=ctx.code_version, codec=ctx.codec_fp,
        k15a_gate=ctx.gate_info, layers_per_call=18, preflight=False,
        admission_override=None, decodes_per_call=1,
        # совместимость с выводом харнесса для --policy k15f
        coeffs=planned.tolist())
    meta = json.loads(json.dumps(meta, default=k15t.json_scalar))
    meta["model_fingerprint"] = hashlib.sha1("|".join(str(meta[k]) for k in (
        "basis_sha1", "basis_state_sha1", "refine_module_sha1",
        "k15f_policy_sha1", "k15h_policy_sha1", "label", "directions",
        "moment", "amp_factor", "control_seed", "frozen_content_sha",
        "joint_sha1", "plan_sha1")).encode()).hexdigest()[:12]
    return SimpleNamespace(model=model, proc=ctx.proc, codec=ctx.codec,
                           act=act, log=log, meta=meta, ctx=ctx)


# --- самопроверки -------------------------------------------------------------
def selftest():
    import torch
    import k15h_static_basis as sb
    assert dir_coeff("3p") == [0, 0, 0, 1.0] and dir_coeff("0m")[0] == -1.0
    for bad in ("4p", "0x", "3"):
        try:
            dir_coeff(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    dirs = ["0p", "3m", "1p", "2m", "0p"]
    pc = planned_coeffs("learned_uniform", dirs)
    assert pc.shape == (5, 4) and pc[1, 3] == -1 and pc[2, 1] == 1
    assert len({tuple(r) for r in pc}) == 4            # разные по средам
    assert (planned_coeffs("p4r3p", dirs) == [0, 0, 0, 1]).all()
    assert not planned_coeffs("q0ref", dirs).any()
    s = Schedule(MOMENT)
    seq = [s.step(i == 0) for i in range(6)]
    assert [a for _c, a in seq] == [False, False, False, True, False, False]
    # Torch-преобразования совпадают с эталоном k15h_static_basis
    rng = np.random.default_rng(0)
    U = np.linalg.qr(rng.standard_normal((56, 4)))[0].T
    rho = np.array([2.0, 1.5, 1.0, 0.8])
    sig = rng.uniform(0.02, 0.2, 6)
    w = rng.standard_normal((5, 4, 56)) * 0.3
    w -= (w * U[None]).sum(-1, keepdims=True) * U[None]
    L = 1.3 * rho[None, :, None] * (U[None] + w)
    Ut, rt, st = (torch.tensor(U), torch.tensor(rho), torch.tensor(sig))
    ad = arm_dir_t(torch.tensor(L), Ut, st, torch).double().numpy()
    assert np.allclose(ad, sb.arm_dir_basis(L, U, sig), atol=1e-5)
    stc = static_basis_t(Ut, rt, 1.3, 5, torch).double().numpy()
    assert np.allclose(stc, sb.static_basis(U, rho, 1.3, batch=5), atol=1e-5)
    assert np.allclose(stc, stc[:1])
    # двойная амплитуда ловится сравнением с эталоном
    assert not np.allclose(arm_dir_t(torch.tensor(L * 1.3), Ut, st,
                                     torch).double().numpy(),
                           sb.arm_dir_basis(L, U, sig), atol=1e-5)
    cn, cs = arm_cos_t(torch.tensor(L), Ut, st, torch)
    assert cn.shape == (5, 4) and cs.shape == (5, 4)
    # журнал
    lg = H15Log("learned_uniform", 8)
    a0 = np.zeros((5, 20, 7), np.float32)
    for k in range(6):
        lg.add(a0, a0, np.ones((5, 4, 56)), np.ones((5, 4, 56)), k + 1,
               k == 3, np.zeros((5, 4)), pc,
               h18=(np.ones((5, 16, 8)) if k < 4 else None),
               h18_dtype=("torch.bfloat16" if k < 4 else None),
               cos=[np.ones((5, 4)), np.ones((5, 4))])
    summ, arr = lg.take()
    assert arr["k15h_h18"].shape == (4, 5, 16, 8) and summ["h18_calls"] == 4
    assert summ["h18_source_dtype"] == "torch.bfloat16"
    assert arr["k15g_active"].tolist() == [False] * 3 + [True, False, False]
    assert arr["k15h_coeff_planned"].shape == (5, 4)
    assert lg.take()[0]["calls"] == 0
    print("самопроверка k15h_policy пройдена: направления по средам, чанк 4, "
          "Torch-преобразования = эталон, двойная амплитуда, журнал h18")
    return 0


def integration():
    """Игрушечная модель: пять рук, две раскатки; p4r3p побитово равна руке
    K-15g p4r3p; uniform — индивидуальные направления по средам; вне чанка
    4 действие = a0; h18 вызовов 1–4 одинаков у всех рук."""
    import tempfile
    import torch
    import k15_context
    import k15d_policy as kdp
    import k15d_train as kt
    import k15f_build_cache as kbc
    import k15f_check_identity as ki
    import k15f_continuous_refine as kf
    import k15f_pretrain_basis as kpb
    import k15g_local_policy as lp
    import k15h_static_basis as sb
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
            ck = torch.load(basis0, map_location="cpu", weights_only=False)
            st = dict(ck["state"])
            st["out.weight"] = torch.randn(
                st["out.weight"].shape,
                generator=torch.Generator().manual_seed(1)) * 0.05
            head = kf.BasisHead(*ki.C_norm(ck), ck["d_model"], ck["n_pos"],
                                ck["stats"], hp=ck["hp"])
            head.load_state_dict(st)
            basis = os.path.join(td, "basis.pt")
            torch.save(dict(ck, state=st, state_sha1=kf.state_sha(head),
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
            B = int(batch["attention_mask"].shape[0]) if "attention_mask" \
                in batch else None
            ac = torch.autocast(device_type="cpu", dtype=torch.bfloat16)

            def roll(arm, n=6):
                for i in range(n):
                    arm.act(batch, po, ac, i == 0)
                return arm.log.take()[1]
            old = lp.build_arm("cpu", basis, "p4r3p", torch,
                               identity_report=ident)
            ref_old = None
            for _ in range(2):
                for i in range(6):
                    old.act(batch, po, ac, i == 0)
                ref_old = old.log.take()[1]
            Bn = ref_old["k15d_a0"].shape[1]
            dirs = [DIRS[(3 * i + 1) % 8] for i in range(Bn)]
            arrs = {}
            for lab in LABELS:
                arm = build_arm("cpu", basis, lab, torch, directions=dirs,
                                plan_sha1="P", identity_report=ident)
                roll(arm)
                arrs[lab] = roll(arm)                 # вторая раскатка
                a = arrs[lab]
                assert a["k15g_chunk"].tolist() == [1, 2, 3, 4, 5, 6]
                want = [k == MOMENT and lab != "q0ref" for k in range(1, 7)]
                assert a["k15g_active"].tolist() == want, lab
                off = ~a["k15g_active"]
                lv = a[f"k15d_level_{lab}"]
                assert np.array_equal(lv[off], a["k15d_a0"][off]), lab
                assert a["k15h_h18"].shape[0] == H18_CALLS
            # p4r3p побитово = рука K-15g
            assert np.array_equal(arrs["p4r3p"]["k15d_level_p4r3p"],
                                  ref_old["k15d_level_p4r3p"])
            # h18 и a0 вызовов 1–4 одинаковы у всех рук (вход одинаков)
            for lab in LABELS:
                assert np.array_equal(arrs[lab]["k15h_h18"],
                                      arrs["q0ref"]["k15h_h18"])
                assert np.array_equal(arrs[lab]["k15d_a0"],
                                      arrs["q0ref"]["k15d_a0"])
            # uniform: индивидуальные направления; формулы базиса
            hU = ck["stats"]["U"].numpy().astype(np.float64)
            hr = ck["stats"]["rho"].numpy().astype(np.float64)
            sg = ck["stats"]["sigma_arm"].numpy().astype(np.float64)
            f = float(ck["amp_factor"])
            Lb = arrs["learned_uniform"]["k15f_basis"].astype(np.float64)
            on = arrs["learned_uniform"]["k15g_active"]
            ap = arrs["learned_uniform"]["k15h_coeff_applied"]
            assert np.array_equal(ap[on][0], planned_coeffs(
                "learned_uniform", dirs))
            assert not ap[~on].any()
            assert np.allclose(arrs["arm_dir_uniform"]["k15h_basis_used"],
                               sb.arm_dir_basis(
                                   arrs["arm_dir_uniform"]["k15f_basis"]
                                   .reshape(-1, 4, 56), hU, sg)
                               .reshape(Lb.shape), atol=1e-5)
            assert np.allclose(arrs["static_uniform"]["k15h_basis_used"],
                               sb.static_basis(hU, hr, f)[None, None],
                               atol=1e-5)
            if Bn > 1:
                dl = (arrs["learned_uniform"]["k15d_level_learned_uniform"]
                      - arrs["learned_uniform"]["k15d_a0"])[on][0]
                assert not np.allclose(dl[0], dl[1])
            assert B is None or B == Bn
    finally:
        k15_context.build, sys.argv, kdp.gate_paths = saved
    print("интеграция k15h_policy пройдена: p4r3p побитово = K-15g, "
          "направления по средам, формулы базиса, чанк 4, h18/a0 до "
          "импульса одинаковы")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="K-15h: рука")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--integration", action="store_true")
    a = ap.parse_args()
    if a.integration:
        sys.exit(integration())
    if a.selftest:
        sys.exit(selftest())
    raise SystemExit("модуль руки; запуск — через k15h_harness.py")
