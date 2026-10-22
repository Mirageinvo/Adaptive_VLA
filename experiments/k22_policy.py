#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-22: рука parents24 для оценки на стартах 30–44.

Та же рука, что k21_main_policy (тот же журнал, проверки выхода, пределы,
снимок BAR), но модель — K22 из k22_model, чекпойнт data/k22/runs.
Загрузка сверяет режим, архитектуру, prep_id, SHA кода контракта обучения
и отпечаток весов. k21_main_policy не меняется (закреплён в плане A K-21):
его build_arm вызывается с подменённым загрузчиком.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

LABEL = "parents24"


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def load_policy(torch, label, ckpt, device):
    import k21_data as d21
    import k21_train as kt
    import k22_model as k22
    if label != LABEL:
        raise SystemExit(f"метка {label!r}: эта рука только для {LABEL}")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if cfg["mode"] != LABEL or cfg.get("architecture") != "k22_parents24":
        raise SystemExit(f"чекпойнт режима {cfg['mode']} / "
                         f"{cfg.get('architecture')}, нужен {LABEL}")
    prep = json.load(open(d21.REPORT))
    if prep.get("prep_id") != cfg["prep_id"]:
        raise SystemExit("чекпойнт обучен на другой подготовке")
    bad = [f for f, s in cfg["code"].items()
           if sha(os.path.join(HERE, f)) != s]
    if bad:
        raise SystemExit(f"код модели изменён после обучения: {bad}")
    dev = torch.device(device)
    bar = d21.load_bar(dev, prep["bar_pin"])
    m = bar["model"]
    m.action_expert.float()
    M = k22.build(torch)
    mod = M["K22"](m, np.asarray(prep["bounds_codec"]["L"]),
                   np.asarray(prep["bounds_codec"]["U"]),
                   prep["state_norm"]["mean"], prep["state_norm"]["std"],
                   checkpoint=False, seed=kt.SEED).to(dev)
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


def build_arm(device, basis, label, torch, *, ckpt, plan_sha1):
    import k21_main_policy as kmp
    orig = kmp.load_policy
    kmp.load_policy = load_policy
    try:
        arm = kmp.build_arm(device, basis, label, torch, ckpt=ckpt,
                            plan_sha1=plan_sha1)
    finally:
        kmp.load_policy = orig
    arm.meta["k22_policy_sha1"] = sha(os.path.abspath(__file__))
    arm.meta["architecture"] = "k22_parents24"
    return arm


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        import k21_main_policy as kmp
        assert callable(kmp.build_arm) and callable(kmp.load_policy)
        print("самопроверка k22_policy пройдена")
        sys.exit(0)
    raise SystemExit("модуль руки; запуск — через k22_harness.py")
