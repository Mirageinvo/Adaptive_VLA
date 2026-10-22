#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fullbar_cached: полный BAR с кэшем префикса как рука харнесса k9h.

Самостоятельный эталон (эквивалентность штатному generate не
подтверждена). Те же исходные веса из закреплённого снимка (fp16, без
autocast — как штатная рука fullbar), тот же greedy blockwise выбор трёх
блоков по 16 кодов, та же сборка трёх уровней ActionCodec; префикс
считается один раз (k21_barcache.generate_cached). Харнесс получает готовый
чанк [B, T, 7] в нормированной шкале кодека (ActionChunk) и исполняет его
так же, как декодированные коды fullbar. На каждом вызове проверяются
счётчики (vision 1, каждый слой VLM 1, эксперт 3) и конечность.
"""
import contextlib
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

LABEL = "fullbar_cached"


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def build_arm(device, basis, label, torch, *, plan_sha1):
    import k21_barcache as bc
    import k21_data as d21
    import k21_policy as base
    if label != LABEL:
        raise SystemExit(f"метка {label!r}: эта рука только для {LABEL}")
    prep = json.load(open(d21.REPORT))
    adm = bc.require_admitted(dict(prep_report=prep))
    dev = torch.device(device)
    bar = d21.load_bar(dev, prep["bar_pin"])
    model = bar["model"]
    log = base.K21Log(label)
    st = dict(call=0)

    def act(batch, pos_off, autocast, first):
        _ = autocast            # штатный fullbar исполняется без autocast
        if first:
            st["call"] = 0
        st["call"] += 1
        B = int(batch["input_ids"].shape[0])
        cnt = dict(vision=0, text=[0] * 24, expert=[0] * 24)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        with contextlib.nullcontext():
            toks = bc.generate_cached(torch, model, batch, int(pos_off), cnt)
            with torch.no_grad():
                x = bc.decode(torch, bar["E"], bar["codec"], toks, B)
        torch.cuda.synchronize(dev)
        ms = (time.perf_counter() - t0) * 1e3
        if not bc.counters_ok(cnt, model.num_blocks):
            raise SystemExit(f"рука fullbar_cached: счётчики {cnt}")
        if not bool(torch.isfinite(x).all()):
            raise SystemExit("рука fullbar_cached: выход не конечен")
        a = x.cpu().numpy().astype(np.float32)
        t = toks.cpu().numpy()
        log.add(act=a[:, :8], chunk=st["call"], tokens=t, ms=ms)
        if first:
            print(f"    проверка fullbar_cached: вызов на {B} сред {ms:.1f} "
                  f"мс; vision {cnt['vision']}, слои VLM {set(cnt['text'])},"
                  f" эксперт {set(cnt['expert'])}; {adm['equivalence']}",
                  flush=True)
        return a, t[:, :model.block_size]

    meta = dict(
        arm="fullbar_cached", label=label, plan_sha1=plan_sha1,
        prep_id=prep["prep_id"], bar_revision=prep["bar_pin"]["revision"],
        barcache_sha1=adm["barcache_sha1"], reference_kind="standalone",
        equivalence=adm["equivalence"],
        equivalence_confirmed=adm["equivalence_confirmed"],
        k21_cached_policy_sha1=sha(os.path.abspath(__file__)),
        coeffs=[0.0], control=False, basis_sha1=None, amp_factor=None,
        decodes_per_call=1, layers_per_call=24)
    meta["model_fingerprint"] = hashlib.sha1("|".join(str(meta[k]) for k in (
        "label", "prep_id", "bar_revision", "barcache_sha1",
        "k21_cached_policy_sha1", "plan_sha1")).encode()).hexdigest()[:12]
    return SimpleNamespace(model=model, proc=bar["proc"], act=act, log=log,
                           meta=meta)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        import k21_barcache as bc
        assert callable(bc.generate_cached) and callable(bc.require_admitted)
        print("самопроверка k21_cached_policy пройдена")
        sys.exit(0)
    raise SystemExit("модуль руки; запуск — через k21_cached_harness.py")
