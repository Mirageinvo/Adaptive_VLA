#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15e M0: рука с фиксированным масштабом поправки h18, a0 + α·δ.

ЧТО ИСПОЛНЯЕТСЯ НА КАЖДОМ ВЫЗОВЕ: аттестованный путь руки h18 из K-15d
(чекпойнт h1p1, проход до 18-го слоя, `k15d_policy.load_refiner`), затем
`k15e_candidates.make_candidates` для α ∈ {0, α_руки, 1}. Исполняется
кандидат α_руки. На КАЖДОМ вызове проверяется:

  * кандидат α=0 побитово равен a0, кандидат α=1 — выходу h18;
  * действие конечно и |a| <= 1.5 на исполняемых шагах.

ПРИВЯЗКА — та же, что у руки h18 K-15d: завершённый допущенный чекпойнт
h1p1, гейт K-15d этой карты, отчёт проверки вывода K-15d для этого файла
этими версиями k15d_depth_refine/k15d_policy. K-15d не меняется: модуль
импортирует его загрузчики. Отпечаток модуля кандидатов и α входят в
отпечаток модели руки.
"""
import argparse
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H18_CKPT = "data/k15d/h1p1_s0.pt"


def build_arm(device, alpha, torch, *, checkpoint=H18_CKPT,
              inference_report=None):
    import k15_context
    import k15d_depth_refine as dr
    import k15d_policy as kp
    import k15e_candidates as kc
    alpha = float(alpha)
    if not -1.0 <= alpha <= 1.0:
        raise SystemExit(f"α {alpha} вне [-1, 1]")
    g15a, g15d = kp.gate_paths(device)
    tag = str(device).replace(":", "")
    inference_report = inference_report or \
        f"reports/k15d/inference_h1p1_s0_{tag}.json"
    for f in (checkpoint, inference_report, g15a, g15d):
        if not os.path.exists(f):
            raise SystemExit(f"нет {f}")
    report = json.load(open(inference_report, encoding="utf-8"))
    ck_sha = kp.sha_file(checkpoint)
    problems = kp.check_report(report, ck_sha=ck_sha,
                               refine_sha=kp.sha_file(dr.__file__),
                               policy_sha=kp.sha_file(kp.__file__))
    if problems:
        raise SystemExit("рука K-15e не собрана: " + "; ".join(problems))
    ctx = k15_context.build(kp.context_namespace(device, g15a))
    L = kp.load_refiner(ctx, checkpoint, k15d_gate=g15d)
    if L.level != "1" or L.layers != 18:
        raise SystemExit(f"чекпойнт {checkpoint}: уровень {L.level}, "
                         f"слоёв {L.layers}; K-15e строится на h18")
    model, ref = ctx.model, L.ref
    label = kc.alpha_label(alpha)
    log = kp.LevelLog([label], L.h_exec)
    grid = (0.0, alpha, 1.0)

    def act(batch, pos_off, autocast, first):
        with torch.no_grad(), autocast:
            v, p_ = model.build_inputs(position_offset=pos_off, **batch)
            out = ref.run(model, vlm_inputs_embeds=v,
                          attention_mask=batch.get("attention_mask"),
                          position_ids=p_, decode=ctx.decode_fp32,
                          stop_after="1")
        a0, a1 = out["a0"], out["actions"]["1"]
        with torch.no_grad():
            c = kc.make_candidates(a0, a1, out["logit0"], out["logits"]["1"],
                                   grid, L.h_exec)
        # ИНВАРИАНТЫ НА КАЖДОМ ВЫЗОВЕ, а не только на первом.
        if not torch.equal(c[:, 0], a0):
            raise SystemExit("рука k15e: кандидат α=0 не равен a0")
        if not torch.equal(c[:, 2], a1):
            raise SystemExit("рука k15e: кандидат α=1 не равен h18")
        a = c[:, 1]
        if not bool(torch.isfinite(a).all()):
            raise SystemExit("рука k15e: действие не конечно")
        amax = float(a[:, :L.h_exec].abs().max())
        if amax > kc.ACTION_CLIP_BOUND:
            raise SystemExit(f"рука k15e: |действие| {amax:.3f} выше "
                             f"{kc.ACTION_CLIP_BOUND}")
        if first:
            print(f"    проверка k15e: α={alpha} ({label}), слоёв 18, "
                  f"max|a-a0| {float((a - a0).abs().max()):.4f}, "
                  f"max|a| {amax:.3f}", flush=True)
        log.add(a0.cpu().numpy(), {label: a.cpu().numpy()})
        return a.float().cpu().numpy(), out["q0"].cpu().numpy()

    ck = L.ck
    meta = dict(
        arm="k15e", alpha=alpha, alpha_label=label,
        phase=ck["phase"], level=L.level, levels=[label],
        checkpoint=os.path.abspath(checkpoint), checkpoint_sha1=ck_sha,
        selected_tag=ck.get("selected_tag"),
        selected_level_sha1=ck.get("selected_level_sha1"),
        refine_module_sha1=kp.sha_file(dr.__file__),
        policy_module_sha1=kp.sha_file(kp.__file__),
        k15e_policy_sha1=kp.sha_file(os.path.abspath(__file__)),
        candidates_module_sha1=kp.sha_file(kc.__file__),
        inference_report=os.path.abspath(inference_report),
        inference_report_sha1=kp.sha_file(inference_report),
        k15d_gate_run_id=L.gate.get("run_id"),
        stats_sha1=L.stats["sha1"], frozen_content_sha=L.frozen,
        k15a_gate=ctx.gate_info, joint_sha1=ctx.joint_sha,
        code_version=ctx.code_version,
        codec=ctx.codec_fp, admission_override=None, preflight=False,
        decodes_per_call=1, layers_per_call=18)
    import k15_train_depth_rvq as k15t
    meta = json.loads(json.dumps(meta, default=k15t.json_scalar))
    meta["model_fingerprint"] = hashlib.sha1("|".join(
        str(meta[k]) for k in ("checkpoint_sha1", "refine_module_sha1",
                               "policy_module_sha1", "k15e_policy_sha1",
                               "candidates_module_sha1", "alpha",
                               "selected_level_sha1", "frozen_content_sha",
                               "joint_sha1")).encode()).hexdigest()[:12]
    return SimpleNamespace(model=model, proc=ctx.proc, codec=ctx.codec,
                           act=act, log=log, meta=meta, ctx=ctx)


def selftest():
    import k15e_candidates as kc
    assert kc.alpha_label(0.25) == "a025"
    # отпечаток модели обязан различать α
    m = dict(checkpoint_sha1="C", refine_module_sha1="R",
             policy_module_sha1="P", k15e_policy_sha1="E",
             candidates_module_sha1="K", selected_level_sha1="L",
             frozen_content_sha="F", joint_sha1="J")
    fp = {}
    for al in (0.0, 0.5):
        mm = dict(m, alpha=al)
        fp[al] = hashlib.sha1("|".join(str(mm[k]) for k in (
            "checkpoint_sha1", "refine_module_sha1", "policy_module_sha1",
            "k15e_policy_sha1", "candidates_module_sha1", "alpha",
            "selected_level_sha1", "frozen_content_sha", "joint_sha1")
        ).encode()).hexdigest()[:12]
    assert fp[0.0] != fp[0.5]
    assert np.isfinite(0.0)
    print("самопроверка k15e_policy пройдена")
    return 0


def integration():
    """Настоящий act руки на игрушечной среде K-15d (CPU).

    Обучается h1p1 (full на игрушке), снимается проверка вывода K-15d,
    собирается рука K-15e при нескольких α; проверяются инварианты α=0 и
    α=1 и то, что α реально масштабирует поправку.
    """
    import tempfile
    import torch
    import k15_context
    import k15d_check_inference as ki
    import k15d_check_init_identity as kg
    import k15d_policy as kp
    import k15d_train as kt
    saved = (k15_context.build, kt.REF_Q0_VAL_RMS, sys.argv, kp.gate_paths)
    ctx = kt._fake_ctx()
    k15_context.build = lambda a: ctx
    try:
        with tempfile.TemporaryDirectory() as td:
            gate = os.path.join(td, "gate.json")
            sys.argv = ["x", "--out", gate, "--device", "cpu",
                        "--gate-batches", "2"]
            assert kg.main() == 0
            kp.gate_paths = lambda device: (gate, gate)

            def train(mode):
                sys.argv = ["x", "--phase", "h1p1", "--mode", mode,
                            "--device", "cpu", "--k15d-gate", gate,
                            "--out", os.path.join(td, f"h1p1_{mode}.pt"),
                            "--report", os.path.join(td,
                                                     f"h1p1_{mode}.json"),
                            "--report-every", "5"]
                return kt.main()
            assert train("smoke") == 0
            rep = json.load(open(os.path.join(td, "h1p1_smoke.json")))
            kt.REF_Q0_VAL_RMS = rep["final"]["rms"]["a0"]
            assert train("full") == 0
            ck = os.path.join(td, "h1p1_full.pt")
            inf = os.path.join(td, "inf.json")
            sys.argv = ["x", "--checkpoint", ck, "--device", "cpu",
                        "--train-report", os.path.join(td, "h1p1_full.json"),
                        "--out", inf]
            assert ki.main() == 0
            po, sel = ctx.parts_full["val_sel"][0]
            ac = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
            acts = {}
            for al in (0.0, 0.5, 1.0, -1.0):
                arm = build_arm("cpu", al, torch, checkpoint=ck,
                                inference_report=inf)
                a_np, q0c = arm.act(ctx.build_batch(po, sel), po, ac, True)
                acts[al] = a_np
                assert np.array_equal(q0c, ctx.q0_can[np.asarray(sel)])
                s_, arr = arm.log.take()
                assert s_["calls"] == 1 and arm.meta["alpha"] == al
            d = acts[1.0][:, :8, :6] - acts[0.0][:, :8, :6]
            assert np.abs(d).max() > 0, "h18 не сдвинул действие"
            assert np.allclose(acts[0.5][:, :8, :6],
                               acts[0.0][:, :8, :6] + 0.5 * d, atol=1e-5)
            assert np.allclose(acts[-1.0][:, :8, :6],
                               acts[0.0][:, :8, :6] - d, atol=1e-5)
            # отпечатки модели различают α
            fps = {build_arm("cpu", al, torch, checkpoint=ck,
                             inference_report=inf).meta["model_fingerprint"]
                   for al in (0.0, 0.5)}
            assert len(fps) == 2
    finally:
        (k15_context.build, kt.REF_Q0_VAL_RMS, sys.argv,
         kp.gate_paths) = saved
    print("интеграция k15e_policy пройдена: α=0/α=1 тождественны, α "
          "масштабирует поправку h18 линейно, контроль α<0 — зеркало")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="K-15e: рука a0 + α·δ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--integration", action="store_true")
    a = ap.parse_args()
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    if a.integration:
        sys.exit(integration())
    if a.selftest:
        sys.exit(selftest())
    raise SystemExit("это модуль руки для k9h_multiarm_gate --policy k15e")
