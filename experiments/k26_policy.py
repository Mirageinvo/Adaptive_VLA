#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-26: рука parents_attn24_s<S> / flat_attn24_s<S> для штатного холодного
харнесса.

Спецификация (seed, путь и SHA чекпойнта, отпечаток весов) — только из
плана. Загрузка сверяет режим, архитектуру k26_<режим>, конфигурацию router,
training seed контракта, prep_id, SHA кода контракта, отпечаток весов =
плану и после загрузки. Рука — k21_main_policy.build_arm с подменённым
загрузчиком (журнал, пределы, конечность, схват ±1); дополнительно на КАЖДОМ
вызове: счётчики (vision 1, слои по 1, router 6 с прежними длинами, K/V 1)
и агрегаты router по шести точкам — масса на padding, ошибка сумм P,
конечность, RMS поправки, энтропия и KL по уровням, число токенов банка;
у parents — деления split1/split2 (root пишет базовая рука). Полные карты P
в журнал не пишутся; в smoke-блоках на первом вызове — ограниченный
debug-фрагмент (S, P, log-prior, маска одной среды после слоя 24).
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
import k26_model as k26m  # noqa: E402
import k26_protocol as pc  # noqa: E402


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def spec_problems(label, spec, cfg, weights_hash):
    arch, seed = pc.parse_label(label)
    p = []
    if spec.get("arch") != arch or int(spec.get("seed", -1)) != seed:
        p.append(f"{label}: спецификация плана {spec.get('arch')}/"
                 f"{spec.get('seed')}")
    if cfg.get("mode") != arch:
        p.append(f"{label}: чекпойнт режима {cfg.get('mode')}")
    if int(cfg.get("seed", -1)) != seed:
        p.append(f"{label}: training seed контракта {cfg.get('seed')}")
    if cfg.get("architecture") != k26m.ARCH[arch]:
        p.append(f"{label}: архитектура контракта {cfg.get('architecture')}")
    if cfg.get("router") != ka.CONFIG:
        p.append(f"{label}: конфигурация router контракта")
    if weights_hash != spec.get("weights_hash"):
        p.append(f"{label}: отпечаток весов ≠ плану")
    return p


def load_policy(torch, label, spec, device):
    import k21_data as d21
    import k21_train as kt
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
    arch, seed = pc.parse_label(label)
    mod = k26m.build(torch)["make"](
        arch, m, np.asarray(prep["bounds_codec"]["L"]),
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


def journal_record(out, mode, keep_debug=False):
    """Агрегаты router одного вызова -> массивы журнала (float32)."""
    infos = out.get("router")
    if not infos or len(infos) != len(ka.AFTER):
        raise SystemExit(f"router: агрегатов {0 if not infos else len(infos)}"
                         f", ожидалось {len(ka.AFTER)}")
    f = [ka.info_floats(x) for x in infos]
    H = np.full((6, 3), -1.0, np.float32)
    KL = np.full((6, 3), -1.0, np.float32)
    for i, x in enumerate(f):
        for lv, y in enumerate(x["levels"]):
            H[i, lv] = y["entropy"]
            if y["kl_to_prior"] is not None:
                KL[i, lv] = y["kl_to_prior"]
    rec = dict(attn_pad=np.float32([x["pad_mass"] for x in f]),
               attn_sumerr=np.float32([x["sum_err"] for x in f]),
               attn_finite=np.float32([float(x["finite"]) for x in f]),
               attn_rms=np.float32([x["rms_ratio"] for x in f]),
               attn_H=H, attn_KL=KL,
               attn_ntok=np.int32(out["bank"]["counts"].tolist()))
    if mode == "parents_attn24":
        rec["attn_split1"] = out["split1"].float().cpu().numpy()
        rec["attn_split2"] = out["split2"].float().cpu().numpy()
    if keep_debug:
        mp = infos[-1].get("maps")
        if mp is None:
            raise SystemExit("router: нет debug-карт")
        rec.update(attn_dbg_S=mp["S"][0].numpy(), attn_dbg_P=mp["P"][0].numpy(),
                   attn_dbg_log_prior=np.where(
                       mp["valid"][0].numpy()[None, None], mp["log_prior"][0]
                       .numpy(), 0.0).astype(np.float32),
                   attn_dbg_valid=mp["valid"][0].numpy())
    return rec


def build_arm(device, basis, label, torch, *, spec, plan_sha1, debug=False):
    import k21_main_policy as kmp
    arch, seed = pc.parse_label(label)
    orig = kmp.load_policy
    kmp.load_policy = lambda torch_, label_, ckpt_, device_: load_policy(
        torch_, label, spec, device_)
    try:
        arm = kmp.build_arm(device, basis, label, torch, ckpt=spec["ckpt"],
                            plan_sha1=plan_sha1)
    finally:
        kmp.load_policy = orig
    mod = arm.k21
    mod.collect = True
    fwd0 = mod.forward
    last = {}
    st = dict(dbg_done=False)

    def forward(*a, **k):
        out = fwd0(*a, **k)
        last["out"] = out
        return out
    mod.forward = forward
    act0 = arm.act

    def act(batch, pos_off, autocast, first):
        last.clear()
        keep = debug and not st["dbg_done"]
        mod.keep_maps = keep
        try:
            res = act0(batch, pos_off, autocast, first)
        finally:
            mod.keep_maps = False
        prob = k26m.counters_problems(arch, mod.counters)
        if prob or "out" not in last:
            raise SystemExit(f"рука {label}: " + "; ".join(prob or [
                "нет выхода модели"]))
        rec = journal_record(last["out"], arch, keep_debug=keep)
        bad = [i for i in range(6) if rec["attn_pad"][i] != 0 or
               rec["attn_sumerr"][i] > ka.SUM_TOL or
               rec["attn_finite"][i] != 1]
        if bad:
            raise SystemExit(f"рука {label}: router в точках {bad} — padding "
                             f"{rec['attn_pad'].tolist()}, суммы "
                             f"{rec['attn_sumerr'].tolist()}")
        if keep:
            st["dbg_done"] = True
            nb = sum(v.nbytes for k, v in rec.items() if "dbg" in k)
            print(f"    debug-фрагмент router: {nb / 2 ** 20:.2f} МБ "
                  f"({rec['attn_dbg_S'].shape})", flush=True)
        arm.log.add(**rec)
        return res
    arm.act = act
    arm.meta.update(arch=arch, train_seed=seed, router_version=ka.VERSION,
                    router_debug=bool(debug),
                    k26_policy_sha1=sha(os.path.abspath(__file__)))
    return arm


def selftest():
    import torch
    spec = dict(arch="flat_attn24", seed=1, weights_hash="w")
    cfg = dict(mode="flat_attn24", seed=1, architecture="k26_flat_attn24",
               router=ka.CONFIG)
    lb = "flat_attn24_s1"
    assert spec_problems(lb, spec, cfg, "w") == []
    assert spec_problems(lb, spec, dict(cfg, seed=0), "w")
    assert spec_problems(lb, spec, cfg, "x")
    assert spec_problems(lb, spec, dict(cfg, mode="parents_attn24"), "w")
    assert spec_problems(lb, spec, dict(cfg, router=dict(ka.CONFIG,
                                                         eps=1.0)), "w")
    assert spec_problems("parents_attn24_s1", spec, cfg, "w")
    for mode in k26m.MODES:
        z, lab, _b = pc.fake_journal(torch, mode, C=2, B=3)
        assert z["k21_attn_pad"].shape == (2, 6)
        assert z["k21_attn_dbg_S"].shape[0] == 1
        assert pc.attn_problems(z, lab, mode, debug=True) == []
    print("самопроверка k26_policy пройдена: сверка спецификации и "
          "контракта (режим, seed, архитектура, router, веса), журнал "
          "router и debug-фрагмент проходят проверку протокола")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль руки; запуск — через k26_harness.py")
