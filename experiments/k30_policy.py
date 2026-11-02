#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-30: рука parents_hybrid24_s<S> / parents_softplan24_s<S> для штатного
холодного харнесса (как K-25: k21_main_policy.build_arm с подменённым
загрузчиком; журнал, пределы, конечность, схват ±1 — базовая рука).

Загрузка: спецификация плана (режим, seed, best.pt, SHA, отпечаток весов,
источник K-25, словарь) -> k30_model.make_model (источник K-25 и словарь со
сверкой) -> веса best.pt -> полный отпечаток (включая буферы словаря и
геометрии) = плану; код контракта обучения не изменён. Режим плана — из
спецификации, не из CLI.
Журнал каждого вызова (в npz с префиксом k21_): k30_logits [B,4,128],
k30_ids, k30_b_soft, k30_b_used, k30_C0, k30_g1, k30_g2, k30_delta1, k30_y2,
k30_Z, k30_grip; на первом вызове блока — ограниченный debug: h8, h_root+
(после plan_proj) и начальные средние (после Expand1), float32.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k30_model as k30m  # noqa: E402

LOG_KEYS = ("logits", "ids", "b_soft", "b_used", "C0", "g1", "g2", "delta1",
            "y2", "Z", "grip")
DBG_KEYS = ("h8", "hrp", "mid_init", "half")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def prep_view():
    import k21_data as d21
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    return prep, dict(L=np.asarray(prep["bounds_codec"]["L"]),
                      U=np.asarray(prep["bounds_codec"]["U"]),
                      state_mean=prep["state_norm"]["mean"],
                      state_std=prep["state_norm"]["std"], prep_report=prep)


def load_policy(torch, label, spec, device):
    import k21_train as kt
    import k30_data as kd
    mode, seed = k30m.parse_label(label)
    p = []
    if spec.get("mode") != mode or int(spec.get("seed", -1)) != seed:
        p.append(f"спецификация {spec.get('mode')}/{spec.get('seed')}")
    if sha(spec["ckpt"]) != spec["ckpt_sha1"]:
        p.append("best.pt ≠ SHA плана")
    ck = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    if ck.get("mode") != mode or int(ck.get("seed", -1)) != seed:
        p.append(f"чекпойнт {ck.get('mode')}/{ck.get('seed')}")
    if ck["weights_hash"] != spec["weights_hash"] or \
            kt.state_hash(ck["weights"]) != ck["weights_hash"]:
        p.append("отпечаток весов чекпойнта")
    C = kd.load()
    if C["sha"] != spec["codec_sha"] or ck.get("codec_sha") != C["sha"]:
        p.append("словарь ≠ плану/чекпойнту")
    bad = [f for f, s in spec["code"].items()
           if sha(os.path.join(HERE, f)) != s]
    if bad:
        p.append(f"код модели изменён после обучения: {bad}")
    if p:
        raise SystemExit(f"{label}: " + "; ".join(p))
    prep, D = prep_view()
    dev = torch.device(device)
    mod, bar = k30m.make_model(torch, seed, mode, spec["source"],
                               dict(P32=C["P32"], s=C["s"]), D, dev,
                               checkpoint=False)
    own = dict(mod.named_parameters())
    own.update(dict(mod.named_buffers()))
    with torch.no_grad():
        for k, v in ck["weights"].items():
            if k not in own:
                raise SystemExit(f"в модели нет {k}")
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    if k30m.full_hash(mod) != spec["weights_hash"]:
        raise SystemExit("полный отпечаток после загрузки ≠ плану")
    if mod.plan_mode != mode:
        raise SystemExit("режим плана модели ≠ метке")
    mod.eval()
    for q in mod.parameters():
        q.requires_grad_(False)
    return mod, bar, ck, prep


def counters_problems(c):
    if c is None:
        return ["счётчики не записаны"]
    p = []
    if c.get("vision") != 1:
        p.append(f"vision {c.get('vision')}")
    if any(x != 1 for x in c.get("text", [])) or len(c.get("text", [])) != 24:
        p.append("слои VLM не по одному разу")
    if c.get("active") != k30m.ACTIVE:
        p.append(f"активные query {c.get('active')}")
    if c.get("classifier") != 1:
        p.append(f"классификатор корней вызван {c.get('classifier')} раз")
    return p


def record(out, debug):
    pl, cd = out["plan"], out["cond"]
    src = dict(logits=pl["logits"], ids=pl["ids"], b_soft=pl["b_soft"],
               b_used=pl["b_used"], C0=cd["C0"], g1=cd["g1"], g2=cd["g2"],
               delta1=cd["delta1"], y2=cd["y2"], Z=cd["Z"],
               grip=out["grip_logit"])
    rec = {}
    for k, v in src.items():
        x = v.detach().cpu()
        rec[f"k30_{k}"] = x.numpy() if k == "ids" else x.float().numpy()
    if debug and out.get("debug"):
        for k in DBG_KEYS:
            rec[f"k30_dbg_{k}"] = out["debug"][k].cpu().numpy()
    return rec


def build_arm(device, basis, label, torch, *, spec, plan_sha1):
    import k21_main_policy as kmp
    mode, seed = k30m.parse_label(label)
    orig = kmp.load_policy
    kmp.load_policy = lambda torch_, label_, ckpt_, device_: load_policy(
        torch_, label, spec, device_)
    try:
        arm = kmp.build_arm(device, basis, label, torch, ckpt=spec["ckpt"],
                            plan_sha1=plan_sha1)
    finally:
        kmp.load_policy = orig
    mod = arm.k21
    fwd0 = mod.forward
    last = {}
    state = dict(calls=0)

    def forward(*a, **k):
        out = fwd0(*a, debug=state["calls"] == 0, **k)
        last["out"] = out
        return out
    mod.forward = forward
    act0 = arm.act

    def act(batch, pos_off, autocast, first):
        last.clear()
        if first:
            state["calls"] = 0
        res = act0(batch, pos_off, autocast, first)
        prob = counters_problems(mod.counters)
        if prob or "out" not in last:
            raise SystemExit(f"рука {label}: " + "; ".join(prob or [
                "нет выхода модели"]))
        arm.log.add(**record(last["out"], state["calls"] == 0))
        state["calls"] += 1
        return res
    arm.act = act
    arm.meta.update(arch=k30m.ARCH, plan_mode=mode, train_seed=seed,
                    codec_sha=spec["codec_sha"],
                    weights_hash=spec["weights_hash"],
                    k30_policy_sha1=sha(os.path.abspath(__file__)))
    return arm


def selftest():
    import torch
    import k21_model as km
    M = k30m.build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    P = np.random.default_rng(0).uniform(-0.8, 0.8, (128, 6)).astype(
        np.float32)
    mod = M["K30"](km.fake_bar(torch), L, U, P, np.full(6, 0.3), seed=0,
                   plan_mode="soft").eval()
    with torch.no_grad():
        out = mod(km.fake_batch(torch, B=3), 1, torch.zeros(3, 11),
                  debug=True)
    rec = record(out, True)
    assert set(rec) == {f"k30_{k}" for k in LOG_KEYS} | \
        {f"k30_dbg_{k}" for k in DBG_KEYS}
    assert rec["k30_ids"].dtype.kind == "i" and \
        rec["k30_logits"].shape == (3, 4, 128)
    assert counters_problems(mod.counters) == []
    assert counters_problems(dict(mod.counters, classifier=2))
    assert k30m.parse_label("parents_softplan24_s1") == ("soft", 1)
    print("самопроверка k30_policy пройдена: журнал плана/уточнений и "
          "ограниченный debug, счётчики (классификатор ровно 1), метки")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль руки; запуск — через k30_harness.py")
