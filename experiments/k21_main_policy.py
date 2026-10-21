#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21 (расширенная оценка): рука flat24 / tree_adapt24 / tree_fixed24.

То же, что k21_policy (тот же журнал, та же загрузка из закреплённого
снимка, сверка отпечатка весов, prep_id и SHA кода контракта обучения),
но с режимом tree_fixed24. k21_policy закреплён в плане пилота и не
меняется."""
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

import k21_policy as base  # noqa: E402

LABELS = ("flat24", "tree_adapt24", "tree_fixed24")


def load_policy(torch, label, ckpt, device):
    import k21_data as d21
    import k21_model as km
    import k21_train as kt
    if label not in LABELS:
        raise SystemExit(f"метка {label!r}")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if cfg["mode"] != label:
        raise SystemExit(f"чекпойнт режима {cfg['mode']}, метка {label}")
    prep = json.load(open(d21.REPORT))
    if prep.get("prep_id") != cfg["prep_id"]:
        raise SystemExit("чекпойнт обучен на другой подготовке")
    bad = [f for f, s in cfg["code"].items()
           if base.sha(os.path.join(HERE, f)) != s]
    if bad:
        raise SystemExit(f"код модели изменён после обучения: {bad}")
    dev = torch.device(device)
    bar = d21.load_bar(dev, prep["bar_pin"])
    m = bar["model"]
    m.action_expert.float()
    M = km.build(torch)
    mod = M["K21"](m, label, np.asarray(prep["bounds_codec"]["L"]),
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
    import k16_policy as kp
    import k21_model as km
    mod, bar, ck, prep = load_policy(torch, label, ckpt, device)
    dev = torch.device(device)
    L = torch.as_tensor(prep["bounds_codec"]["L"], device=dev)
    U = torch.as_tensor(prep["bounds_codec"]["U"], device=dev)
    log = base.K21Log(label)
    st = dict(nbuf=0, call=0)

    def act(batch, pos_off, autocast, first):
        if first:
            st["nbuf"] = len(kp.STATE_BUF) - 1
            st["call"] = 0
        if len(kp.STATE_BUF) != st["nbuf"] + 1:
            raise SystemExit("рука k21: нет состояния текущего вызова")
        st["nbuf"] = len(kp.STATE_BUF)
        st["call"] += 1
        state8 = np.asarray(kp.STATE_BUF[-1], np.float64)
        B = state8.shape[0]
        s11 = torch.as_tensor(km.state11(state8), device=dev)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        with torch.no_grad(), autocast:
            out = mod(batch, int(pos_off), s11)
        torch.cuda.synchronize(dev)
        ms = (time.perf_counter() - t0) * 1e3
        arm = out["arm"].float()
        if not (torch.isfinite(arm).all() and torch.isfinite(
                out["grip_logit"]).all()):
            raise SystemExit("рука k21: выход не конечен")
        if not bool(((arm >= L - 1e-4) & (arm <= U + 1e-4)).all()):
            raise SystemExit("рука k21: выход вне пределов")
        g = torch.where(out["grip_logit"].float() > 0, 1.0, -1.0)
        a = torch.cat([arm, g[..., None]], -1).cpu().numpy().astype(
            np.float32)
        near = ((arm - L).abs() < 0.01 * (U - L)) | \
            ((U - arm).abs() < 0.01 * (U - L))
        rec = dict(act=a[:, :8], chunk=st["call"],
                   state=state8.astype(np.float32),
                   grip_logit=out["grip_logit"][:, :8].float().cpu().numpy(),
                   saturated=near.float().mean((1, 2)).cpu().numpy(), ms=ms)
        if "root" in out:
            rec["root"] = out["root"].float().cpu().numpy()
        log.add(**rec)
        if first:
            print(f"    проверка k21: {label}, вызов на {B} сред {ms:.1f} мс,"
                  f" активные query {mod.counters['active'][::8]}",
                  flush=True)
        return a, np.zeros((B, 16), np.int64)

    meta = dict(
        arm="k21", label=label, plan_sha1=plan_sha1,
        ckpt=os.path.abspath(ckpt), ckpt_sha1=base.sha(ckpt),
        weights_hash=ck["weights_hash"], best_update=ck["update"],
        prep_id=prep["prep_id"], bar_revision=prep["bar_pin"]["revision"],
        k21_policy_sha1=base.sha(os.path.abspath(__file__)),
        coeffs=[0.0], control=False, basis_sha1=None, amp_factor=None,
        decodes_per_call=0, layers_per_call=24)
    meta["model_fingerprint"] = hashlib.sha1("|".join(str(meta[k]) for k in (
        "label", "ckpt_sha1", "weights_hash", "prep_id", "k21_policy_sha1",
        "plan_sha1")).encode()).hexdigest()[:12]
    return SimpleNamespace(model=mod.bar, proc=bar["proc"], act=act,
                           log=log, meta=meta, k21=mod)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        assert set(LABELS) >= set(base.LABELS)
        print("самопроверка k21_main_policy пройдена")
        sys.exit(0)
    raise SystemExit("модуль руки; запуск — через k21_main_harness.py")
