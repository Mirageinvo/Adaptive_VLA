#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-25: рука parents_cond24_s<S> для штатного холодного харнесса.

Спецификация (seed, путь и SHA чекпойнта, отпечаток весов) — только из
плана. Загрузка сверяет режим, архитектуру k25_parents_cond24, training seed
контракта, prep_id, SHA кода контракта, отпечаток весов = плану и после
загрузки. Рука — k21_main_policy.build_arm с подменённым загрузчиком
(журнал, пределы, конечность, схват ±1); дополнительно на КАЖДОМ вызове:
счётчики (vision 1, слои по 1, активные 4/12/28) и запись в журнал
C0/g1/g2/δ1/y2/Z/G0/G1/G2/W0/W1/W2 и делений split1/split2 (float32; root
записывает базовая рука) для независимой пересборки при проверке блока.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k25_protocol as pc  # noqa: E402
import k25_readout as kr  # noqa: E402

ACTIVE = [4] * 8 + [12] * 8 + [28] * 8
COND_KEYS = ("C0", "g1", "g2", "delta1", "y2", "Z", "G0", "G1", "G2",
             "W0", "W1", "W2")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def spec_problems(label, spec, cfg, weights_hash):
    arch, seed = pc.parse_label(label)
    p = []
    if spec.get("arch") != arch or int(spec.get("seed", -1)) != seed:
        p.append(f"{label}: спецификация плана {spec.get('arch')}/"
                 f"{spec.get('seed')}")
    if cfg.get("mode") != "parents_cond24":
        p.append(f"{label}: чекпойнт режима {cfg.get('mode')}")
    if int(cfg.get("seed", -1)) != seed:
        p.append(f"{label}: training seed контракта {cfg.get('seed')}")
    if cfg.get("architecture") != "k25_parents_cond24":
        p.append(f"{label}: архитектура контракта {cfg.get('architecture')}")
    if (cfg.get("readout") or {}).get("version") != kr.VERSION:
        p.append(f"{label}: версия readout {cfg.get('readout')}")
    if weights_hash != spec.get("weights_hash"):
        p.append(f"{label}: отпечаток весов ≠ плану")
    return p


def load_policy(torch, label, spec, device):
    import k21_data as d21
    import k21_train as kt
    import k25_model as k25m
    ck = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    cfg = ck["config"]
    prob = spec_problems(label, spec, cfg, ck["weights_hash"])
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    if prep.get("prep_id") != cfg["prep_id"]:
        prob.append("чекпойнт обучен на другой подготовке")
    bad = [f for f, s in cfg["code"].items()
           if sha(os.path.join(HERE, f)) != s]
    if bad:
        prob.append(f"код модели изменён после обучения: {bad}")
    if prob:
        raise SystemExit("; ".join(prob))
    dev = torch.device(device)
    bar = d21.load_bar(dev, prep["bar_pin"])
    m = bar["model"]
    m.action_expert.float()
    _arch, seed = pc.parse_label(label)
    mod = k25m.build(torch)["K25"](
        m, np.asarray(prep["bounds_codec"]["L"]),
        np.asarray(prep["bounds_codec"]["U"]), prep["state_norm"]["mean"],
        prep["state_norm"]["std"], checkpoint=False, seed=seed).to(dev)
    own = dict(mod.named_parameters())
    own.update(dict(mod.named_buffers()))
    with torch.no_grad():
        for k, v in ck["weights"].items():
            if k not in own:
                raise SystemExit(f"в модели нет {k}")
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    if kt.state_hash(kt.trainable_state(mod)) != ck["weights_hash"]:
        raise SystemExit("веса после загрузки ≠ отпечатку чекпойнта")
    mod.eval()
    for p in mod.parameters():
        p.requires_grad_(False)
    return mod, bar, ck, prep


def counters_problems(c):
    if c is None:
        return ["счётчики не записаны"]
    p = []
    if c.get("vision") != 1:
        p.append(f"vision {c.get('vision')}")
    if any(x != 1 for x in c.get("text", [])) or len(c.get("text", [])) != 24:
        p.append("слои VLM не по одному разу")
    if any(x != 1 for x in c.get("expert", [])) or \
            len(c.get("expert", [])) != 24:
        p.append("слои эксперта не по одному разу")
    if c.get("active") != ACTIVE:
        p.append(f"активные query {c.get('active')}")
    return p


def build_arm(device, basis, label, torch, *, spec, plan_sha1):
    import k21_main_policy as kmp
    _arch, seed = pc.parse_label(label)
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

    def forward(*a, **k):
        out = fwd0(*a, **k)
        last["out"] = out
        return out
    mod.forward = forward
    act0 = arm.act

    def act(batch, pos_off, autocast, first):
        last.clear()
        res = act0(batch, pos_off, autocast, first)
        prob = counters_problems(mod.counters)
        if prob or "out" not in last:
            raise SystemExit(f"рука {label}: " + "; ".join(prob or [
                "нет выхода модели"]))
        out = last["out"]
        rec = {f"cond_{k}": out["cond"][k].detach().float().cpu().numpy()
               for k in COND_KEYS}
        rec["cond_split1"] = out["split1"].float().cpu().numpy()
        rec["cond_split2"] = out["split2"].float().cpu().numpy()
        arm.log.add(**rec)
        return res
    arm.act = act
    arm.meta.update(arch="parents_cond24", train_seed=seed,
                    readout_version=kr.VERSION,
                    k25_policy_sha1=sha(os.path.abspath(__file__)))
    return arm


def selftest():
    spec = dict(arch="parents_cond24", seed=1, weights_hash="w")
    cfg = dict(mode="parents_cond24", seed=1, architecture="k25_parents_cond24",
               readout=dict(version=kr.VERSION))
    lb = "parents_cond24_s1"
    assert spec_problems(lb, spec, cfg, "w") == []
    assert spec_problems(lb, spec, dict(cfg, seed=0), "w")
    assert spec_problems(lb, spec, cfg, "x")
    assert spec_problems(lb, spec, dict(cfg, mode="parents_mr24"), "w")
    assert spec_problems(lb, spec, dict(cfg, readout=dict(version="mr1")),
                         "w")
    good = dict(vision=1, text=[1] * 24, expert=[1] * 24, active=ACTIVE)
    assert counters_problems(good) == []
    assert counters_problems(dict(good, active=[16] * 24))
    print("самопроверка k25_policy пройдена: сверка спецификации и "
          "контракта (режим, seed, архитектура, версия readout, веса), "
          "счётчики 4/12/28")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль руки; запуск — через k25_harness.py")
