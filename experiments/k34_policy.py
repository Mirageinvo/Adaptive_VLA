#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-34: рука для штатного холодного харнесса K-9h — выбранная глубина
выхода (8/16/24) одного обученного checkpoint K-34 либо неизменённый
источник K-26 (R, глубина 24, рука k26_policy).

Рука K-34 = k21_main_policy.build_arm с подменённым загрузчиком: источник
K-26 (k34_train.load_source: сверка спецификации и отпечатка), затем веса
best.pt K-34 (сверка отпечатка), mod.exit_depth = d — глубина фиксирована на
весь эпизод и не выбирается по задаче/исходу. На каждом вызове: счётчики
выбранной глубины (k34_model.counters_problems), агрегаты router по
выполненным точкам (padding 0, суммы, конечность), журнал: глубина, число
слоёв, router, Expand, голов. В отладочных блоках (collector-smoke) на
первом вызове сохраняются вход и выход головы: x_d = norm(h_d), интервалы
узлов, веса, y, arm, логит схвата — для независимого пересчёта головы из
checkpoint.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k26_attention as ka  # noqa: E402
import k34_model as k34m  # noqa: E402


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def label(mode, seed, depth=None):
    """Метки: T/F K-34 — <mode>_s<S>_d<d>; R — метка K-26 <mode>_s<S>."""
    return f"{mode}_s{seed}" + ("" if depth is None else f"_d{depth}")


def parse(lab):
    parts = lab.split("_")
    if parts[-1].startswith("d") and parts[-1][1:].isdigit():
        d = int(parts[-1][1:])
        mode, seed = "_".join(parts[:-2]), int(parts[-2][1:])
        return mode, seed, d
    return "_".join(parts[:-1]), int(parts[-1][1:]), None


def load_trained(torch, mode, seed, depth, spec, device):
    """K-34: источник + веса best.pt со сверкой отпечатка."""
    import k21_data as d21
    import k21_train as ktr
    import k34_train as k34t
    mod, bar = k34t.load_source(torch, mode, seed, spec["source"], device,
                                checkpoint=False)
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    if spec.get("ckpt") is None:
        # состояние init (collector-smoke): источник + копии голов
        if spec.get("kind") != "init":
            raise SystemExit("нет checkpoint K-34")
        mod.exit_depth = int(depth)
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)
        return mod, bar, dict(weights_hash=None, update=None), prep
    ck = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if cfg.get("architecture") != k34m.ARCH[mode] or \
            int(cfg.get("seed", -1)) != seed or \
            cfg.get("source", {}).get("ckpt_sha1") != \
            spec["source"]["ckpt_sha1"]:
        raise SystemExit(f"{label(mode, seed)}: checkpoint K-34 — "
                         f"архитектура, seed или источник")
    if ck["weights_hash"] != spec["weights_hash"]:
        raise SystemExit("отпечаток best.pt K-34 ≠ плану")
    own = dict(mod.named_parameters())
    own.update(dict(mod.named_buffers()))
    with torch.no_grad():
        for k, v in ck["weights"].items():
            if k not in own:
                raise SystemExit(f"в модели нет {k}")
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    if ktr.state_hash(ktr.trainable_state(mod)) != ck["weights_hash"]:
        raise SystemExit("веса после загрузки ≠ отпечатку best.pt K-34")
    mod.exit_depth = int(depth)
    mod.eval()
    for p in mod.parameters():
        p.requires_grad_(False)
    return mod, bar, ck, prep


def router_record(out, n_points):
    infos = out.get("router") or []
    if len(infos) != n_points:
        raise SystemExit(f"router: агрегатов {len(infos)}, ожидалось "
                         f"{n_points}")
    f = [ka.info_floats(x) for x in infos]
    pad = np.full(6, -1.0, np.float32)
    se = np.full(6, -1.0, np.float32)
    fin = np.full(6, -1.0, np.float32)
    for i, x in enumerate(f):
        pad[i], se[i], fin[i] = x["pad_mass"], x["sum_err"], float(
            x["finite"])
    return dict(attn_pad=pad, attn_sumerr=se, attn_finite=fin)


def build_arm(device, basis, lab, torch, *, spec, plan_sha1, debug=False):
    """spec: для K-34 — mode, seed, depth, source, ckpt, weights_hash;
    R — через k26_policy (spec K-26)."""
    import k21_main_policy as kmp
    mode, seed, depth = parse(lab)
    if depth is None:
        import k26_policy as kp26
        arm = kp26.build_arm(device, basis, lab, torch, spec=spec,
                             plan_sha1=plan_sha1, debug=False)
        arm.meta.update(k34_kind="R", k34_depth=24)
        return arm
    orig = kmp.load_policy
    kmp.load_policy = lambda torch_, l_, c_, dev_: load_trained(
        torch_, mode, seed, depth, spec, dev_)
    try:
        # init (collector-smoke): файл для метаданных базовой руки —
        # checkpoint источника; собственный отпечаток K-34 — "init"
        arm = kmp.build_arm(device, basis, lab, torch,
                            ckpt=spec["ckpt"] or spec["source"]["ckpt"],
                            plan_sha1=plan_sha1)
    finally:
        kmp.load_policy = orig
    mod = arm.k21
    mod.collect = True
    n_pts = len(k34m.expected_counters(mode, depth)["router"])
    last, st = {}, dict(dbg_done=False, dbg=None)
    fwd0 = mod.forward

    def forward(*a, **k):
        out = fwd0(*a, **k)
        last["out"] = out
        return out
    mod.forward = forward

    def head(d, h, iv, w, cnt):
        cnt["heads"].append(d)
        x = mod.bar.action_expert.norm(h)
        out = getattr(mod, k34m.HEADS[d])(x, iv, w)
        if st["dbg"] is not None:
            st["dbg"].update(
                dbg_x=x.float().cpu().numpy(), dbg_iv=iv.float().cpu().numpy(),
                dbg_w=w.float().cpu().numpy(),
                dbg_y=out["y"].float().cpu().numpy(),
                dbg_arm=out["arm"].float().cpu().numpy(),
                dbg_grip=out["grip_logit"].float().cpu().numpy(),
                dbg_depth=np.int32(d))
        return out
    mod._head = head
    act0 = arm.act

    def act(batch, pos_off, autocast, first):
        last.clear()
        keep = debug and not st["dbg_done"]
        st["dbg"] = {} if keep else None
        res = act0(batch, pos_off, autocast, first)
        prob = k34m.counters_problems(mode, mod.counters, depth)
        if prob or "out" not in last:
            raise SystemExit(f"рука {lab}: " + "; ".join(prob or [
                "нет выхода модели"]))
        rec = router_record(last["out"], n_pts)
        bad = [i for i in range(n_pts) if rec["attn_pad"][i] != 0 or
               rec["attn_sumerr"][i] > ka.SUM_TOL or
               rec["attn_finite"][i] != 1]
        if bad:
            raise SystemExit(f"рука {lab}: router в точках {bad}")
        c = mod.counters
        rec.update(exit_depth=np.int32(depth), n_text=np.int32(sum(c["text"])),
                   n_router=np.int32(len(c["router"])),
                   n_expand=np.int32(c["expand"]),
                   n_heads=np.int32(len(c["heads"])),
                   head_depth=np.int32(c["heads"][0]))
        if keep:
            rec.update(st["dbg"])
            st["dbg_done"] = True
        st["dbg"] = None
        arm.log.add(**rec)
        return res
    arm.act = act
    arm.meta.update(k34_kind="K34", k34_mode=mode, k34_seed=seed,
                    k34_depth=int(depth), k34_ckpt_sha1=spec["ckpt_sha1"],
                    k34_weights_hash=spec["weights_hash"],
                    k34_source_sha1=spec["source"]["ckpt_sha1"],
                    k34_debug=bool(debug), arch=k34m.ARCH[mode],
                    layers_per_call=int(depth),
                    train_seed=seed, router_version=ka.VERSION,
                    k34_policy_sha1=sha(os.path.abspath(__file__)))
    return arm


def journal_problems(lab, z, debug=False):
    """Журнал выбранной глубины: глубина, число слоёв/router/Expand/голов —
    ожидаемые на каждом вызове; router по выполненным точкам."""
    mode, _seed, depth = parse(lab)
    if depth is None:
        return []
    want = k34m.expected_counters(mode, depth)
    need = ("k21_exit_depth", "k21_n_text", "k21_n_router", "k21_n_expand",
            "k21_n_heads", "k21_head_depth", "k21_attn_pad",
            "k21_attn_sumerr", "k21_attn_finite")
    miss = [k for k in need if k not in z]
    if miss:
        return [f"{lab}: в журнале нет {miss}"]
    p = []
    C = len(z["k15g_chunk"])
    for k, v in (("k21_exit_depth", depth), ("k21_n_text", depth),
                 ("k21_n_router", len(want["router"])),
                 ("k21_n_expand", want["expand"]), ("k21_n_heads", 1),
                 ("k21_head_depth", depth)):
        a = np.asarray(z[k])
        if a.shape[0] != C or not (a == v).all():
            p.append(f"{lab}: {k[4:]} ≠ {v} на части вызовов")
    n = len(want["router"])
    pad = np.asarray(z["k21_attn_pad"])[:, :n]
    if (pad != 0).any() or (np.asarray(z["k21_attn_sumerr"])[:, :n] >
                            ka.SUM_TOL).any() or \
            (np.asarray(z["k21_attn_finite"])[:, :n] != 1).any():
        p.append(f"{lab}: router — padding/суммы/конечность")
    if debug:
        dk = ("k21_dbg_x", "k21_dbg_iv", "k21_dbg_w", "k21_dbg_y",
              "k21_dbg_arm", "k21_dbg_grip", "k21_dbg_depth")
        if any(k not in z for k in dk):
            p.append(f"{lab}: нет отладочного снимка головы")
        elif int(np.asarray(z["k21_dbg_depth"]).reshape(-1)[0]) != depth:
            p.append(f"{lab}: снимок головы другой глубины")
    return p


def head_recompute(torch, z, ckpt_path, depth, L, U, device="cpu",
                   autocast=False):
    """Голова из checkpoint (без модели) на сохранённом входе первого
    вызова -> max|Δ| y, arm, grip к записанным."""
    import k21_tree as kt
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pre = k34m.HEADS[depth] + "."
    sd = {k[len(pre):]: v for k, v in ck["weights"].items()
          if k.startswith(pre)}
    x = torch.as_tensor(np.asarray(z["k21_dbg_x"])[0])
    H = x.shape[-1]
    R = kt.build(torch)["Readout"](H, L, U).to(device)
    R.load_state_dict(sd)
    dev = torch.device(device)
    iv = torch.as_tensor(np.asarray(z["k21_dbg_iv"])[0], device=dev)
    w = torch.as_tensor(np.asarray(z["k21_dbg_w"])[0], device=dev)
    xx = x.to(dev)
    ctx = torch.autocast("cuda", dtype=torch.float16) if autocast else None
    with torch.no_grad():
        if ctx is not None:
            with ctx:                      # Readout сам переводит вход в float
                o = R(xx, iv, w)
        else:
            o = R(xx, iv, w)
    out = {}
    for k, nm in (("y", "k21_dbg_y"), ("arm", "k21_dbg_arm"),
                  ("grip_logit", "k21_dbg_grip")):
        ref = np.asarray(z[nm])[0]
        out[k] = float(np.abs(o[k].float().cpu().numpy() - ref).max())
    return out


def selftest():
    assert parse("parents_attn24_s0_d8") == ("parents_attn24", 0, 8)
    assert parse("flat_attn24_s1_d24") == ("flat_attn24", 1, 24)
    assert parse("flat_attn24_s1") == ("flat_attn24", 1, None)
    assert label("parents_attn24", 1, 16) == "parents_attn24_s1_d16"
    C = 3
    for mode in k34m.MODES:
        for d in k34m.EXITS:
            lab = label(mode, 0, d)
            want = k34m.expected_counters(mode, d)
            n = len(want["router"])
            pad = np.full((C, 6), -1.0, np.float32)
            pad[:, :n] = 0.0
            z = dict(k15g_chunk=np.arange(1, C + 1),
                     k21_exit_depth=np.full(C, d),
                     k21_n_text=np.full(C, d),
                     k21_n_router=np.full(C, n),
                     k21_n_expand=np.full(C, want["expand"]),
                     k21_n_heads=np.ones(C), k21_head_depth=np.full(C, d),
                     k21_attn_pad=pad, k21_attn_sumerr=pad.copy(),
                     k21_attn_finite=np.where(pad == 0, 1.0, -1.0))
            assert journal_problems(lab, z) == [], (mode, d)
            bad = dict(z, k21_n_text=np.r_[np.full(C - 1, d), d - 1])
            assert journal_problems(lab, bad), (mode, d)
            assert journal_problems(lab, z, debug=True)    # нет снимка
    assert n == 6 and k34m.expected_counters("parents_attn24", 8)[
        "router"] == [4, 4]
    print("самопроверка k34_policy пройдена: метки T/F/R, журнал выбранной "
          "глубины (глубина, слои, router, Expand, головы на каждом вызове; "
          "отладочный снимок головы)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль руки; запуск — через k34_harness.py")
