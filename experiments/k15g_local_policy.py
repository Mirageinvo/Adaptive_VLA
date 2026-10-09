#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15g: локальная поправка — ноль -> один импульс в чанке k -> ноль.

РАСПИСАНИЕ. Вызовы политики нумеруются с 1 внутри каждой раскатки (10
стартовых waiting steps в нумерацию не входят; первый вызов — чанк №1,
шаги 0..7). В чанке k применяется c = ±e_j, во всех остальных c = 0: до
импульса исполняется q0, после — снова q0, пересчитанная по НОВЫМ
наблюдениям (а не старые действия исходной траектории).

МЕТКИ РУК:
  q0ref         c = 0 во всех чанках (обязана воспроизвести q0);
  p<k><f><j><s> импульс в чанке k ∈ {2, 4}, семейство f ∈ {l (learned),
                r (контроль)}, направление j ∈ 0..3, знак s ∈ {p, m}.

ТОТ ЖЕ БАЗИС, АМПЛИТУДА И КОНТРОЛЬНАЯ МАТРИЦА, что в K-15f M1: рука
собирается теми же проверками (`k15f_policy`: гейт тождества с кодом 0,
чекпойнт базиса, замороженное), один проход `plain_h18`, один декод a0 —
базис считается в каждом вызове (и логируется), но действие отличается от
a0 только в активном чанке; вне его действие ПОБИТОВО равно a0 (проверка
на каждом вызове).

ЖУРНАЛ ПО ВЫЗОВАМ (в npz блока через харнесс): a0 и исполняемое действие
(k15d_a0, k15d_level_<метка>), базис (k15f_basis), номер чанка
(k15g_chunk) и признак активного вмешательства (k15g_active).
"""
import argparse
import hashlib
import json
import os
import re
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOMENTS = (2, 4)
LABEL_RE = re.compile(r"^(q0ref|p[24][lr][0-3][pm])$")
ACTION_CLIP_BOUND = 1.5


def label_spec(label):
    """Метка -> (чанк импульса или None, c [4], контроль)."""
    if not LABEL_RE.match(str(label)):
        raise ValueError(f"метка {label!r}")
    if label == "q0ref":
        return None, [0.0] * 4, False
    k, fam, j, s = int(label[1]), label[2], int(label[3]), label[4]
    c = [0.0] * 4
    c[j] = 1.0 if s == "p" else -1.0
    return k, c, fam == "r"


def all_labels():
    out = []
    for k in MOMENTS:
        for fam in ("l", "r"):
            for j in range(4):
                for s in ("p", "m"):
                    out.append(f"p{k}{fam}{j}{s}")
    return out


def is_label(label):
    return bool(LABEL_RE.match(str(label)))


class Schedule:
    """Номер вызова внутри раскатки; first=True начинает новую раскатку."""

    def __init__(self, moment):
        self.moment = moment
        self.call = 0

    def step(self, first):
        self.call = 1 if first else self.call + 1
        return self.call, (self.moment is not None
                           and self.call == self.moment)


class PulseLog:
    """Журнал вызовов: a0, действие, базис, номер чанка, активность."""

    def __init__(self, label, h_exec):
        self.label, self.h = label, int(h_exec)
        self.reset()

    def reset(self):
        self.a0, self.act, self.basis, self.chunk, self.active = \
            [], [], [], [], []

    def add(self, a0, act, basis, chunk, active):
        self.a0.append(np.asarray(a0[:, :self.h], np.float32))
        self.act.append(np.asarray(act[:, :self.h], np.float32))
        self.basis.append(np.asarray(basis, np.float32))
        self.chunk.append(int(chunk))
        self.active.append(bool(active))

    def take(self):
        if not self.a0:
            self.reset()
            return dict(calls=0), {}
        a0, act = np.stack(self.a0), np.stack(self.act)
        arrays = {"k15d_a0": a0, f"k15d_level_{self.label}": act,
                  "k15f_basis": np.stack(self.basis),
                  "k15g_chunk": np.asarray(self.chunk, np.int64),
                  "k15g_active": np.asarray(self.active, bool)}
        summ = dict(calls=int(len(self.chunk)), levels=[self.label],
                    active_calls=int(sum(self.active)),
                    **{f"mean_abs_delta_{self.label}": np.abs(act - a0)
                       .reshape(-1, 7).mean(0).tolist(),
                       f"absmax_{self.label}": np.abs(act).reshape(-1, 7)
                       .max(0).tolist(),
                       f"finite_{self.label}": bool(np.isfinite(act).all())})
        self.reset()
        return summ, arrays


def build_arm(device, basis, label, torch, *, identity_report=None):
    """Рука импульса на том же базисе, что K-15f M1, и с теми же
    проверками привязки (`k15f_policy`)."""
    import k15_context
    import k15d_policy as kdp
    import k15f_build_cache as kb
    import k15f_check_identity as ki
    import k15f_continuous_refine as kf
    import k15f_policy as kfp
    moment, c_list, control = label_spec(label)
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
        raise SystemExit("рука K-15g не собрана: " + "; ".join(prob))
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
    c = torch.tensor(c_list, device=dev)
    zero_c = torch.zeros(kf.K_BASIS, device=dev)
    sched = Schedule(moment)
    log = PulseLog(label, kf.H_EXEC)

    def act(batch, pos_off, autocast, first):
        chunk, active = sched.step(first)
        with torch.no_grad(), autocast:
            v, p_ = model.build_inputs(position_offset=pos_off, **batch)
            q0, a0, h18 = kf.plain_h18(
                model, vlm_inputs_embeds=v,
                attention_mask=batch.get("attention_mask"), position_ids=p_,
                decode=ctx.decode_fp32)
        with torch.no_grad():
            basis_, _u = head(h18, a0)
            basis_ = basis_ * f
            if control:
                basis_ = basis_ @ R
            cc = c if active else zero_c
            a = kf.combine(a0, kf.compose(basis_, cc), head.sigma_arm,
                           head.sigma_g)
            sv = torch.linalg.svdvals(basis_)
            cond = float((sv[:, 0] / sv[:, -1].clamp_min(1e-12)).max())
            un = basis_ / basis_.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            gram = (un @ un.transpose(1, 2)).abs()
            gram = gram - torch.diag_embed(torch.diagonal(gram, dim1=1,
                                                          dim2=2))
            pc = float(gram.max())
        if cond > kf.GEOMETRY["cond_max"] or pc > kf.GEOMETRY["pair_cos_max"]:
            raise SystemExit(f"рука k15g: геометрия базиса вне пределов "
                             f"(обусловленность {cond:.2f}, косинус {pc:.3f})")
        if not active and not torch.equal(a, a0):
            raise SystemExit(f"рука k15g: чанк {chunk} не активен, но "
                             f"действие не равно a0")
        if not bool(torch.isfinite(a).all()):
            raise SystemExit("рука k15g: действие не конечно")
        amax = float(a[:, :kf.H_EXEC].abs().max())
        if amax > ACTION_CLIP_BOUND:
            raise SystemExit(f"рука k15g: |действие| {amax:.3f} выше "
                             f"{ACTION_CLIP_BOUND}")
        if first:
            print(f"    проверка k15g: {label}: импульс в чанке {moment}, "
                  f"c={c_list}, контроль={control}", flush=True)
        if active:
            print(f"    k15g: импульс исполнен в чанке {chunk}, max|a-a0| "
                  f"{float((a - a0).abs().max()):.4f}", flush=True)
        log.add(a0.cpu().numpy(), a.cpu().numpy(),
                basis_.float().cpu().numpy(), chunk, active)
        return a.float().cpu().numpy(), q0.cpu().numpy()

    meta = dict(
        arm="k15g_pulse", label=label, moment=moment, pulse_chunks=1,
        coeffs=c_list, control=bool(control), basis=os.path.abspath(basis),
        basis_sha1=basis_sha, basis_state_sha1=ck["state_sha1"],
        stats_sha1=ck["stats_sha1"], amp_factor=f,
        control_seed=int(ck["control_seed"]),
        control_kind=ck.get("control_kind"),
        identity_report=os.path.abspath(identity_report),
        identity_report_sha1=kb.sha_file(identity_report),
        refine_module_sha1=kb.sha_file(kf.__file__),
        k15f_policy_sha1=kb.sha_file(kfp.__file__),
        k15g_policy_sha1=kb.sha_file(os.path.abspath(__file__)),
        frozen_content_sha=frozen, joint_sha1=ctx.joint_sha,
        code_version=ctx.code_version, codec=ctx.codec_fp,
        k15a_gate=ctx.gate_info, layers_per_call=18, preflight=False,
        admission_override=None, decodes_per_call=1)
    meta = json.loads(json.dumps(meta, default=k15t.json_scalar))
    meta["model_fingerprint"] = hashlib.sha1("|".join(str(meta[k]) for k in (
        "basis_sha1", "basis_state_sha1", "refine_module_sha1",
        "k15f_policy_sha1", "k15g_policy_sha1", "coeffs", "control",
        "moment", "amp_factor", "control_seed", "frozen_content_sha",
        "joint_sha1")).encode()).hexdigest()[:12]
    return SimpleNamespace(model=model, proc=ctx.proc, codec=ctx.codec,
                           act=act, log=log, meta=meta, ctx=ctx)


def selftest():
    assert label_spec("q0ref") == (None, [0.0] * 4, False)
    assert label_spec("p2l1m") == (2, [0.0, -1.0, 0.0, 0.0], False)
    assert label_spec("p4r3p") == (4, [0.0, 0.0, 0.0, 1.0], True)
    for bad in ("p3l0p", "p2x0p", "p2l4p", "q0", "z", "l0p"):
        assert not is_label(bad), bad
    labs = all_labels()
    assert len(labs) == 32 and len(set(labs)) == 32
    # расписание: нумерация с 1 и сброс на first; импульс ровно в чанке k
    s = Schedule(2)
    seq = [s.step(i == 0) for i in range(5)]
    assert [c for c, _ in seq] == [1, 2, 3, 4, 5]
    assert [a for _, a in seq] == [False, True, False, False, False]
    seq = [s.step(i == 0) for i in range(3)]     # новая раскатка
    assert [c for c, _ in seq] == [1, 2, 3] and seq[1][1]
    s0 = Schedule(None)
    assert not any(s0.step(i == 0)[1] for i in range(6))
    # журнал
    lg = PulseLog("p2l0p", 8)
    a0 = np.zeros((5, 20, 7), np.float32)
    for k in range(3):
        lg.add(a0, a0 + (0.1 if k == 1 else 0.0), np.ones((5, 4, 56)),
               k + 1, k == 1)
    summ, arr = lg.take()
    assert summ["calls"] == 3 and summ["active_calls"] == 1
    assert arr["k15g_active"].tolist() == [False, True, False]
    assert arr["k15g_chunk"].tolist() == [1, 2, 3]
    assert arr["k15d_level_p2l0p"].shape == (3, 5, 8, 7)
    print("самопроверка k15g_local_policy пройдена: метки, расписание, "
          "журнал")
    return 0


def integration():
    """Рука импульса на игрушечной среде: базис, гейт с кодом 0, рука,
    расписание по двум раскаткам, вне импульса действие = a0."""
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
            ac = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
            for lab in ("q0ref", "p2l0p", "p4r1m"):
                arm = build_arm("cpu", basis, lab, torch,
                                identity_report=ident)
                moment, _c, _ctl = label_spec(lab)
                for _ in range(2):                    # две раскатки подряд
                    for i in range(5):
                        arm.act(batch, po, ac, i == 0)
                    summ, arr = arm.log.take()
                    assert arr["k15g_chunk"].tolist() == [1, 2, 3, 4, 5]
                    want = [k == moment for k in range(1, 6)]
                    assert arr["k15g_active"].tolist() == want, (lab, want)
                    off = ~arr["k15g_active"]
                    lv = arr[f"k15d_level_{lab}"]
                    assert np.array_equal(lv[off], arr["k15d_a0"][off])
                    if moment is not None:
                        on = arr["k15g_active"]
                        assert not np.array_equal(lv[on],
                                                  arr["k15d_a0"][on])
                assert arm.meta["moment"] == moment
    finally:
        k15_context.build, sys.argv, kdp.gate_paths = saved
    print("интеграция k15g_local_policy пройдена: импульс ровно в чанке k "
          "в каждой раскатке, вне импульса действие = a0")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="K-15g: локальная поправка")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--integration", action="store_true")
    a = ap.parse_args()
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    if a.integration:
        sys.exit(integration())
    if a.selftest:
        sys.exit(selftest())
    raise SystemExit("модуль руки; запуск — через k15g_harness.py")
