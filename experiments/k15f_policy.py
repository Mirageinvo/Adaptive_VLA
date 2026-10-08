#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15f M1: рука с фиксированными коэффициентами c на весь эпизод.

НА КАЖДОМ ВЫЗОВЕ: чистый проход до 18-го слоя (`plain_h18`), базис
δ(s_t) = голова(h18, a0) · f (f — зарегистрированный множитель амплитуды),
для контроля — δ(s_t)·R, затем a_t = combine(a0_t, Σ c_j δ_j(s_t)).
Коэффициенты фиксированы на весь эпизод, базис пересчитывается при каждом
наблюдении.

МЕТКИ РУК (общие для раннера и анализа):
  z        c = 0 (обязана побитово совпасть с q0);
  l<j><s>  learned, c = ±e_j, j = 0..3, s = p (+1) или m (−1);
  r<j><s>  контроль: тот же c на базисе, повёрнутом R.

ПРИВЯЗКА: чекпойнт базиса (завершённый, full, технически исправный, с
выбранным множителем) и пройденный отчёт гейта тождества k15f для ЭТОЙ
карты, ЭТОГО файла базиса и ЭТИХ версий кода. На каждом вызове проверяются
конечность и |a| <= 1.5; у руки z — побитовое равенство a0.
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
LABEL_RE = re.compile(r"^(z|[lr][0-3][pm])$")
ACTION_CLIP_BOUND = 1.5


def label_spec(label):
    """Метка -> (c [4], контроль)."""
    if not LABEL_RE.match(str(label)):
        raise ValueError(f"метка {label!r}")
    c = [0.0] * 4
    if label == "z":
        return c, False
    c[int(label[1])] = 1.0 if label[2] == "p" else -1.0
    return c, label[0] == "r"


def all_labels():
    out = []
    for fam in ("l", "r"):
        for j in range(4):
            for s in ("p", "m"):
                out.append(f"{fam}{j}{s}")
    return out


def check_identity_report(rep, *, basis_sha, device, code,
                          gate_sha=None):
    p = []
    if gate_sha is not None and (rep.get("code_sha") or {}).get(
            "k15f_check_identity") != gate_sha:
        p.append("гейт снят другой версией k15f_check_identity")
    if rep.get("kind") != "k15f_identity" or rep.get("passed") is not True:
        p.append("гейт тождества k15f не пройден")
    if rep.get("code") != 0 or rep.get("basis_state_dependent") is not True:
        p.append("базис не зависит от h18 (глобальная PCA, код гейта "
                 f"{rep.get('code')}): это не иерархический K-15f")
    if rep.get("basis_sha1") != basis_sha:
        p.append("гейт снят для другого файла базиса")
    if rep.get("device") != str(device):
        p.append(f"гейт снят на {rep.get('device')}, рука на {device}")
    if (rep.get("code_sha") or {}).get("k15f_continuous_refine") != code:
        p.append("гейт снят другой версией k15f_continuous_refine")
    return p


def build_arm(device, basis, label, torch, *, identity_report=None):
    import k15_context
    import k15d_policy as kp
    import k15f_build_cache as kb
    import k15f_check_identity as ki
    import k15f_continuous_refine as kf
    c_list, control = label_spec(label)
    tag = str(device).replace(":", "")
    identity_report = identity_report or f"reports/k15f/identity_{tag}.json"
    g15a, _g15d = kp.gate_paths(device)
    for f_ in (basis, identity_report, g15a):
        if not os.path.exists(f_):
            raise SystemExit(f"нет {f_}")
    basis_sha = kb.sha_file(basis)
    rep = json.load(open(identity_report))
    prob = check_identity_report(rep, basis_sha=basis_sha, device=device,
                                 code=kb.sha_file(kf.__file__),
                                 gate_sha=kb.sha_file(ki.__file__))
    ck = torch.load(basis, map_location="cpu", weights_only=False)
    prob += ki.check_basis_ckpt(ck)
    if prob:
        raise SystemExit("рука K-15f не собрана: " + "; ".join(prob))
    ctx = k15_context.build(kp.context_namespace(device, g15a))
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
    zero = not any(c_list)
    log = kp.LevelLog([label], kf.H_EXEC)

    def act(batch, pos_off, autocast, first):
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
            a = kf.combine(a0, kf.compose(basis_, c), head.sigma_arm,
                           head.sigma_g)
        # ГЕОМЕТРИЯ ЖИВОГО БАЗИСА на каждом вызове (вращение контроля её
        # сохраняет, поэтому достаточно learned): SVD 4×56 почти бесплатна
        with torch.no_grad():
            sv = torch.linalg.svdvals(basis_ if not control
                                      else basis_ @ R.T)
            cond = float((sv[:, 0] / sv[:, -1].clamp_min(1e-12)).max())
            un = basis_ / basis_.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            gram = (un @ un.transpose(1, 2)).abs()
            gram = gram - torch.diag_embed(torch.diagonal(gram, dim1=1,
                                                          dim2=2))
            pc = float(gram.max())
        if cond > kf.GEOMETRY["cond_max"] or pc > kf.GEOMETRY["pair_cos_max"]:
            raise SystemExit(f"рука k15f: геометрия базиса вне пределов "
                             f"(обусловленность {cond:.2f}, попарный "
                             f"косинус {pc:.3f})")
        if zero and not torch.equal(a, a0):
            raise SystemExit("рука k15f z: c = 0, но действие не равно a0")
        if not bool(torch.isfinite(a).all()):
            raise SystemExit("рука k15f: действие не конечно")
        amax = float(a[:, :kf.H_EXEC].abs().max())
        if amax > ACTION_CLIP_BOUND:
            raise SystemExit(f"рука k15f: |действие| {amax:.3f} выше "
                             f"{ACTION_CLIP_BOUND}")
        if first:
            print(f"    проверка k15f: {label} c={c_list} контроль={control},"
                  f" слоёв 18, max|a-a0| {float((a - a0).abs().max()):.4f},"
                  f" max|a| {amax:.3f}", flush=True)
        log.add(a0.cpu().numpy(), {label: a.cpu().numpy()})
        return a.float().cpu().numpy(), q0.cpu().numpy()

    meta = dict(
        arm="k15f", label=label, coeffs=c_list, control=bool(control),
        basis=os.path.abspath(basis), basis_sha1=basis_sha,
        basis_state_sha1=ck["state_sha1"], basis_selected=ck.get("selected"),
        stats_sha1=ck["stats_sha1"], amp_factor=f,
        control_seed=int(ck["control_seed"]),
        identity_report=os.path.abspath(identity_report),
        identity_report_sha1=kb.sha_file(identity_report),
        refine_module_sha1=kb.sha_file(kf.__file__),
        k15f_policy_sha1=kb.sha_file(os.path.abspath(__file__)),
        frozen_content_sha=frozen, joint_sha1=ctx.joint_sha,
        code_version=ctx.code_version, codec=ctx.codec_fp,
        k15a_gate=ctx.gate_info, layers_per_call=18, preflight=False,
        admission_override=None, decodes_per_call=1)
    meta = json.loads(json.dumps(meta, default=k15t.json_scalar))
    meta["model_fingerprint"] = hashlib.sha1("|".join(str(meta[k]) for k in (
        "basis_sha1", "basis_state_sha1", "refine_module_sha1",
        "k15f_policy_sha1", "coeffs", "control", "amp_factor",
        "control_seed", "frozen_content_sha", "joint_sha1"
    )).encode()).hexdigest()[:12]
    return SimpleNamespace(model=model, proc=ctx.proc, codec=ctx.codec,
                           act=act, log=log, meta=meta, ctx=ctx)


def selftest():
    assert label_spec("z") == ([0.0] * 4, False)
    assert label_spec("l2m") == ([0.0, 0.0, -1.0, 0.0], False)
    assert label_spec("r0p") == ([1.0, 0.0, 0.0, 0.0], True)
    for bad in ("l4p", "x0p", "l0", "zz", "r1q"):
        try:
            label_spec(bad)
            raise AssertionError(f"принята метка {bad}")
        except ValueError:
            pass
    labs = all_labels()
    assert len(labs) == 16 and len(set(labs)) == 16
    # поле code отчёта — код исхода гейта; отпечатки кода — в code_sha
    rep = dict(kind="k15f_identity", passed=True, code=0,
               basis_state_dependent=True, basis_sha1="B", device="cuda:1",
               code_sha=dict(k15f_continuous_refine="K"))
    kw = dict(basis_sha="B", device="cuda:1", code="K")
    assert check_identity_report(rep, **kw) == []
    rep_g = dict(rep, code_sha=dict(k15f_continuous_refine="K",
                                    k15f_check_identity="G"))
    assert check_identity_report(rep_g, gate_sha="G", **kw) == []
    assert check_identity_report(rep_g, gate_sha="ИНОЙ", **kw)
    for mut in (dict(passed=False), dict(basis_sha1="X"),
                dict(device="cuda:0"),
                dict(code_sha=dict(k15f_continuous_refine="Y")),
                dict(code=4), dict(basis_state_dependent=False)):
        assert check_identity_report(dict(rep, **mut), **kw), mut
    assert np.isfinite(ACTION_CLIP_BOUND)
    print("самопроверка k15f_policy пройдена")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="K-15f: рука с фиксированным c")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    if a.selftest:
        sys.exit(selftest())
    raise SystemExit("это модуль руки для k9h_multiarm_gate --policy k15f")
