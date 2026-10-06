#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15d: проверка вывода руки настоящим проходом по val_sel.

ЗАЧЕМ. Тренер оценивал уточнение своим циклом. Рука в роллауте исполняет
`k15d_policy.make_act` — функцию, собранную из чекпойнта с диска, на карте
роллаута, с гейтом этой карты. Здесь эта самая функция проходит весь
val_sel (в smoke — те же 40 батчей, что тренер) и сверяется с тем, что
тренер записал для выбранной точки:

    q0                       побитово с каноническим на каждом батче
    строки val_sel           тот же порядок, что в файле строк тренера
    ошибка строки a0 и a_k   |live - rec| <= COST_ATOL + COST_RTOL * |rec|
    RMS итогового уровня     с отчётом тренера, отн. RMS_REL
    диапазон                 |a| <= 1.5 на исполняемых шагах (в act)
    замороженное             побитово до и после

Ошибка строки сверяется с допуском: на другой карте (D0 обучен на cuda:0,
роллаут может идти на cuda:1) ядра fp16 могут различаться в последних
разрядах. q0 при этом обязан совпасть побитово — это гарантирует гейт
K-15a обеих карт.

Отчёт требует рука: `checkpoint_sha1`, `refine_module_sha1`,
`policy_module_sha1`, `smoke`. Коды: 0 — воспроизведено; 3 — нет.
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

COST_RTOL = 1e-4
COST_ATOL = 1e-7
RMS_REL = 1e-5
SMOKE_VAL_BATCHES = 40       # как SMOKE["val_batches"] в k15d_train


def compare_rows(live, rec, rtol=COST_RTOL, atol=COST_ATOL):
    lv = np.asarray(live, np.float64)
    rv = np.asarray(rec, np.float64)
    if lv.shape != rv.shape:
        return None, None, -1
    if lv.size == 0:
        return 0.0, 0.0, 0
    d = np.abs(lv - rv)
    rel = d / np.maximum(np.abs(rv), 1e-12)
    return float(d.max()), float(rel.max()), int(
        (d > atol + rtol * np.abs(rv)).sum())


def default_train_report(ck):
    suffix = "" if ck["mode"] == "full" else f"_{ck['mode']}"
    return f"reports/k15d/{ck['phase']}_s{ck['seed']}{suffix}.json"


def main():
    import k15d_policy as kp
    ap = argparse.ArgumentParser(description="K-15d: проверка вывода руки")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--train-report", default=None,
                    help="отчёт тренера этого чекпойнта (по умолчанию — "
                         "канонический путь по фазе, сиду и режиму)")
    ap.add_argument("--allow-smoke", action="store_true",
                    help="smoke-чекпойнт: отчёт помечается smoke и годится "
                         "только для предполётной проверки роллаута")
    ap.add_argument("--allow-failed-admission", default=None,
                    metavar="ПРИЧИНА",
                    help="чекпойнт, НЕ допущенный фильтром, проверяется и "
                         "получает право на руку только с этой текстовой "
                         "причиной; она и проваленные гейты пишутся в отчёт")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.checkpoint:
        ap.error("--checkpoint обязателен")
    import torch
    import k15_context
    import k15d_depth_refine as dr
    import k15_train_depth_rvq as k15t
    t0 = time.time()
    ck0 = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    smoke = ck0.get("mode") == "smoke"
    if smoke and not a.allow_smoke:
        raise SystemExit("smoke-чекпойнт: нужен --allow-smoke")
    train_report = a.train_report or default_train_report(ck0)
    rep_t = json.load(open(train_report))
    if rep_t.get("run_id") != ck0.get("run_id"):
        raise SystemExit(f"отчёт тренера {train_report} от прогона "
                         f"{rep_t.get('run_id')}, чекпойнт — "
                         f"{ck0.get('run_id')}")
    rows_file = rep_t["rows_file"]
    tag = os.path.splitext(os.path.basename(a.checkpoint))[0]
    out = a.out or f"reports/k15d/inference_{tag}_{a.device.replace(':', '')}.json"

    g15a, g15d = kp.gate_paths(a.device)
    ctx = k15_context.build(kp.context_namespace(a.device, g15a))
    torch_, model, dev = ctx.torch, ctx.model, ctx.dev
    L = kp.load_refiner(ctx, a.checkpoint, k15d_gate=g15d, preflight=smoke,
                        override=a.allow_failed_admission)
    if L.exception is not None:
        print(f"  ИСКЛЮЧЕНИЕ ИЗ ФИЛЬТРА ДОПУСКА: {L.exception['reason']}; "
              f"проваленные гейты {L.exception['failed']}")
    act = kp.make_act(ctx, L, verbose=True)
    lv = L.level
    val_all = list(ctx.parts_full["val_sel"])
    if smoke:
        import k15d_check_init_identity as kg
        batches = [val_all[i] for i in kg.strided(len(val_all),
                                                  SMOKE_VAL_BATCHES)]
    else:
        batches = val_all
    rec = np.load(rows_file, allow_pickle=True)
    rows_rec = np.asarray(rec["rows"], np.int64)
    rows_live = np.concatenate([np.asarray(s, np.int64) for _p, s in batches])
    checks = {}
    checks["rows_same_order"] = bool(np.array_equal(rows_rec, rows_live))
    q0_can = torch_.as_tensor(np.asarray(ctx.q0_can))
    ac16 = torch_.autocast(device_type=dev.type, dtype=ctx.dt)
    err0, errk, q0_bad = [], [], 0
    horizon = int(L.stats["horizon"])
    for i, (po, sel) in enumerate(batches):
        b = ctx.build_batch(po, sel)
        a_np, q0 = act(b, po, ac16, i == 0)
        q0_bad += int((torch_.as_tensor(q0)
                       != q0_can[np.asarray(sel, np.int64)]).sum())
        t_ = torch_.from_numpy(np.asarray(ctx.ACT[sel], np.float32)
                               [:, :horizon, :7]).to(dev)
        a_t = torch_.from_numpy(a_np).to(dev)
        with torch_.no_grad():
            z0 = model.depth_aligned_book(0)[torch_.as_tensor(q0).to(dev)]
            a0 = ctx.decode_fp32(z0).float()
        r0, _ = k15t.weighted_row_error(a0, t_, ctx.weights_gate, torch_)
        rk, _ = k15t.weighted_row_error(a_t, t_, ctx.weights_gate, torch_)
        err0.append(r0.cpu().numpy())
        errk.append(rk.cpu().numpy())
        if (i + 1) % 100 == 0:
            print(f"  батч {i + 1}/{len(batches)}", flush=True)
    err0, errk = np.concatenate(err0), np.concatenate(errk)
    checks["q0_canonical"] = q0_bad == 0
    cmp = {}
    for key, live in (("a0", err0), (lv, errk)):
        mx_a, mx_r, over = compare_rows(live, rec[f"err_{key}"])
        cmp[key] = dict(max_abs=mx_a, max_rel=mx_r, rows_over=over)
        checks[f"rows_match_{key}"] = over == 0
    rms_live = float(np.sqrt(errk.mean()))
    rms_rec = float(rep_t["final"]["rms"][lv])
    rms_rel = abs(rms_live - rms_rec) / max(rms_rec, 1e-12)
    checks["rms_matches_trainer"] = rms_rel <= RMS_REL
    rms0_live = float(np.sqrt(err0.mean()))
    frozen1, _n, _e = k15t.frozen_content_sha(model, torch_, set())
    checks["frozen_unchanged"] = frozen1 == L.frozen
    passed = all(checks.values())
    print(f"  {lv}: RMS вживую {rms_live:.6f}, у тренера {rms_rec:.6f} "
          f"(отн. {rms_rel:.1e}); a0 {rms0_live:.6f}; строки: " + "; ".join(
              f"{k} макс {v['max_abs']:.2e} абс / {v['max_rel']:.2e} отн, "
              f"за допуском {v['rows_over']}" for k, v in cmp.items()))
    rep = dict(
        kind=kp.REPORT_KIND, smoke=bool(smoke),
        verdict=dict(passed=bool(passed),
                     failed=sorted(k for k, v in checks.items() if not v)),
        checks=checks, compare=cmp, rms_live=rms_live,
        rms_trainer=rms_rec, rms_rel=rms_rel, rms_a0_live=rms0_live,
        rows=int(len(errk)), checkpoint=os.path.abspath(a.checkpoint),
        checkpoint_sha1=kp.sha_file(a.checkpoint),
        checkpoint_run_id=ck0.get("run_id"), phase=ck0.get("phase"),
        level=lv, layers_per_call=L.layers,
        admission_override=L.exception,
        device=str(dev), train_report=train_report,
        train_report_sha1=kp.sha_file(train_report),
        refine_module_sha1=kp.sha_file(dr.__file__),
        policy_module_sha1=kp.sha_file(kp.__file__),
        k15d_gate_run_id=L.gate.get("run_id"),
        trained_gate_run_id=ck0.get("gate_run_id"),
        frozen_sha1=L.frozen, git_head=ctx.git_head,
        tolerances=dict(cost_rtol=COST_RTOL, cost_atol=COST_ATOL,
                        rms_rel=RMS_REL),
        finished=datetime.datetime.now().isoformat(timespec="seconds"),
        seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False,
                  default=k15t.json_scalar)
    os.replace(tmp, out)
    print(f"ПРОВЕРКА ВЫВОДА {tag} на {dev}: "
          f"{'ПРОЙДЕНА' if passed else 'НЕ ПРОЙДЕНА ' + str(rep['verdict']['failed'])}"
          f"; {out}")
    return 0 if passed else 3


def selftest():
    mx, rel, n = compare_rows([1.0, 2.0, 3.0001], [1.0, 2.0, 3.0])
    assert n == 0 and abs(mx - 1e-4) < 1e-9, (mx, rel, n)
    assert compare_rows([1.0, 2.1], [1.0, 2.0])[2] == 1
    assert compare_rows([2e-8], [1e-9])[2] == 0
    assert compare_rows([5e-7], [1e-9])[2] == 1
    assert compare_rows([1.0], [1.0, 2.0])[2] == -1
    ck = dict(phase="d0", seed=0, mode="full")
    assert default_train_report(ck) == "reports/k15d/d0_s0.json"
    assert default_train_report(dict(ck, mode="smoke")) == \
        "reports/k15d/d0_s0_smoke.json"
    print("самопроверка k15d_check_inference пройдена")
    return 0


if __name__ == "__main__":
    sys.exit(main())
