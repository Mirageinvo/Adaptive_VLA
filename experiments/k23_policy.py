#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-23: рука однопроходной модели с явными архитектурой и training seed.

Метка руки — <архитектура>_s<seed> (flat24_s1, parents24_s0, …); спецификацию
(архитектура, seed, путь чекпойнта, отпечаток весов) рука получает только из
плана — никакой глобальной «последней» модели. Загрузка сверяет режим,
архитектуру, training seed контракта, prep_id, SHA кода контракта обучения,
отпечаток весов чекпойнта = плану и после загрузки. Сама рука — та же, что
k21_main_policy (журнал, пределы, конечность, схват ±1): её build_arm
вызывается с подменённым загрузчиком (k21_main_policy не меняется). На
КАЖДОМ вызове сверяются счётчики: vision 1, каждый слой VLM и эксперта 1,
активные query (flat24 — 16 на всех слоях; parents24 — 4/12/28).
"""
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ARCHS = ("flat24", "parents24", "tree_adapt24")
ACTIVE = dict(flat24=[16] * 24, tree_adapt24=[4] * 8 + [8] * 8 + [16] * 8,
              parents24=[4] * 8 + [12] * 8 + [28] * 8)
CONTRACT_ARCH = dict(flat24=(None, "k21_flat24"),
                     tree_adapt24=(None,),
                     parents24=("k22_parents24",))


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def parse_label(label):
    m = re.fullmatch(r"(flat24|parents24|tree_adapt24)_s(\d+)", str(label))
    if not m:
        raise SystemExit(f"метка {label!r}: нужна <архитектура>_s<seed>")
    return m.group(1), int(m.group(2))


def spec_problems(label, spec, cfg, weights_hash):
    """Сверка чекпойнта со спецификацией плана (без загрузки модели)."""
    arch, seed = parse_label(label)
    p = []
    if spec.get("arch") != arch or int(spec.get("seed", -1)) != seed:
        p.append(f"{label}: спецификация плана {spec.get('arch')}/"
                 f"{spec.get('seed')}")
    if cfg.get("mode") != arch:
        p.append(f"{label}: чекпойнт режима {cfg.get('mode')}")
    if int(cfg.get("seed", -1)) != seed:
        p.append(f"{label}: training seed контракта {cfg.get('seed')}")
    if cfg.get("architecture") not in CONTRACT_ARCH[arch]:
        p.append(f"{label}: архитектура контракта {cfg.get('architecture')}")
    if weights_hash != spec.get("weights_hash"):
        p.append(f"{label}: отпечаток весов чекпойнта ≠ плану")
    return p


def load_policy(torch, label, spec, device):
    import k21_data as d21
    import k21_model as km
    import k21_train as kt
    import k22_model as k22
    arch, seed = parse_label(label)
    ckpt = spec["ckpt"]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    prob = spec_problems(label, spec, cfg, ck["weights_hash"])
    prep = json.load(open(d21.REPORT))
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
    L = np.asarray(prep["bounds_codec"]["L"])
    U = np.asarray(prep["bounds_codec"]["U"])
    mean, std = prep["state_norm"]["mean"], prep["state_norm"]["std"]
    if arch == "parents24":
        mod = k22.build(torch)["K22"](m, L, U, mean, std, checkpoint=False,
                                      seed=seed)
    else:
        mod = km.build(torch)["K21"](m, arch, L, U, mean, std,
                                     checkpoint=False, seed=seed)
    mod = mod.to(dev)
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


def counters_problems(arch, c):
    p = []
    if c is None:
        return ["счётчики не записаны"]
    if c.get("vision") != 1:
        p.append(f"vision {c.get('vision')}")
    if any(x != 1 for x in c.get("text", [])) or len(c.get("text", [])) != 24:
        p.append("слои VLM не по одному разу")
    if any(x != 1 for x in c.get("expert", [])) or \
            len(c.get("expert", [])) != 24:
        p.append("слои эксперта не по одному разу")
    if c.get("active") != ACTIVE[arch]:
        p.append(f"активные query {c.get('active')}")
    return p


def build_arm(device, basis, label, torch, *, spec, plan_sha1):
    import k21_main_policy as kmp
    arch, seed = parse_label(label)
    orig = kmp.load_policy
    kmp.load_policy = lambda torch_, label_, ckpt_, device_: load_policy(
        torch_, label, spec, device_)
    try:
        arm = kmp.build_arm(device, basis, label, torch, ckpt=spec["ckpt"],
                            plan_sha1=plan_sha1)
    finally:
        kmp.load_policy = orig
    act0 = arm.act

    def act(batch, pos_off, autocast, first):
        out = act0(batch, pos_off, autocast, first)
        prob = counters_problems(arch, arm.k21.counters)
        if prob:
            raise SystemExit(f"рука {label}: " + "; ".join(prob))
        return out
    arm.act = act
    arm.meta.update(arch=arch, train_seed=seed,
                    k23_policy_sha1=sha(os.path.abspath(__file__)))
    return arm


def selftest():
    assert parse_label("parents24_s0") == ("parents24", 0)
    assert parse_label("flat24_s1") == ("flat24", 1)
    for bad in ("parents24", "flat24_1", "tree_s1"):
        try:
            parse_label(bad)
            raise AssertionError(bad)
        except SystemExit:
            pass
    spec = dict(arch="parents24", seed=0, weights_hash="w")
    cfg = dict(mode="parents24", seed=0, architecture="k22_parents24")
    assert spec_problems("parents24_s0", spec, cfg, "w") == []
    assert spec_problems("parents24_s0", spec, cfg, "x")          # веса
    assert spec_problems("parents24_s0", spec, dict(cfg, seed=1), "w")
    assert spec_problems("parents24_s0", spec, dict(cfg, mode="flat24"), "w")
    assert spec_problems("flat24_s1", dict(arch="flat24", seed=1,
                                           weights_hash="w"),
                         dict(mode="flat24", seed=1,
                              architecture="k21_flat24"), "w") == []
    assert spec_problems("flat24_s1", dict(arch="flat24", seed=1,
                                           weights_hash="w"),
                         dict(mode="flat24", seed=1,
                              architecture="k22_parents24"), "w")
    good = dict(vision=1, text=[1] * 24, expert=[1] * 24,
                active=ACTIVE["parents24"])
    assert counters_problems("parents24", good) == []
    assert counters_problems("flat24", good)
    assert counters_problems("parents24", dict(good, vision=2))
    print("самопроверка k23_policy пройдена: метки с seed, сверка "
          "спецификации плана с контрактом, счётчики по архитектуре")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль руки; запуск — через k23_harness.py")
