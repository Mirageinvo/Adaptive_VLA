#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16: диагностика обученных голов без раскаток и проходов VLM.

По чекпойнту data/k16/models.pt, разметке и кэшу h18 (уже проверенному
обучением):
  * ошибка НУЛЕВОГО предсказания в той же нормировке, что MSE обучения
    (r / r_sigma, r_sigma — по половине B), на B и val_sel;
  * нормы r* и ошибки корректоров на B и val_sel, по задачам;
  * срез «экспертное переключение в следующих 8 командах» (тип не
    «конец», event_cmd_idx − t < 8) против остальных строк — только
    диагностика разметки, для выбора момента в среде не используется;
  * энергия r* в плавной (первые 2 коэффициента DCT по 8 шагам) и
    остальной частях и качество предсказания каждой части;
  * кривые обучения из отчёта обучения.
Нормы частей goal_band в отчёте обучения посчитаны ДО α и в единицах
кодека; здесь все нормы — в единицах кодека, отдельно указано.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k16_models as km  # noqa: E402

H = 8


def band_split(r):
    """r [N, 8, 6] -> (плавная, остальная) части во времени и энергии."""
    D = km.dct_matrix()
    c = np.einsum("ft,ntc->nfc", D, r)
    lo = np.einsum("ft,nfc->ntc", D[:km.N_LOW], c[:, :km.N_LOW])
    return lo, r - lo


def skill(pred, true):
    """Улучшение относительно нулевого предсказания."""
    m0 = float((true ** 2).mean())
    return 1.0 - float(((pred - true) ** 2).mean()) / max(m0, 1e-12)


def predict(torch, goals, heads, D, dev):
    out = {}
    for name, m in goals.items():
        sfk = "sf_now" if name == "goal_now" else "sf_hist"
        P, F = [], []
        with torch.no_grad():
            for s in range(0, len(D["age"]), 1024):
                i = slice(s, s + 1024)
                h = torch.as_tensor(np.asarray(D["h18"][i]), device=dev)
                a0 = torch.as_tensor(D["a0"][i], device=dev).float()
                sf = torch.as_tensor(D[sfk][i], device=dev)
                hid = m.hidden(h, a0, sf)
                F.append(hid.cpu().numpy())
                P.append(m.body[-1](hid).cpu().numpy())
        D[f"pred_{name}"], D[f"feat_{name}"] = np.concatenate(P), \
            np.concatenate(F)
    gsrc = dict(direct="feat_goal_hist", goal_flat="pred_goal_hist",
                goal_band="pred_goal_hist", goal_band_now="pred_goal_now")
    for name, m in heads.items():
        sfk = "sf_now" if name == "goal_band_now" else "sf_hist"
        R = []
        with torch.no_grad():
            for s in range(0, len(D["age"]), 1024):
                i = slice(s, s + 1024)
                r, _ = m(torch.as_tensor(np.asarray(D["h18"][i]), device=dev),
                         torch.as_tensor(D["a0"][i], device=dev).float(),
                         torch.as_tensor(D[sfk][i], device=dev),
                         torch.as_tensor(D[gsrc[name]][i], device=dev))
                R.append(r.cpu().numpy())
        out[name] = np.concatenate(R)
    return out


def metrics(D, pred_norm, r_sigma, L, part, idx):
    rs = r_sigma
    true = np.asarray(D["rstar"], np.float64)
    res = dict(rows=int(len(true)),
               zero_mse_norm=float(((true / rs) ** 2).mean()),
               rstar_norm_codec=dict(
                   median=float(np.median(np.linalg.norm(
                       true.reshape(len(true), -1), axis=1))),
                   p95=float(np.percentile(np.linalg.norm(
                       true.reshape(len(true), -1), axis=1), 95))))
    lo_t, hi_t = band_split(true)
    res["energy_share_low"] = float((lo_t ** 2).sum() / max(
        (true ** 2).sum(), 1e-12))
    # срез «экспертное переключение в следующих 8 командах»
    ecmd = np.asarray(L[f"{part}__event_cmd_idx"])[idx]
    t = np.asarray(L[f"{part}__step"])[idx]
    et = np.asarray(L[f"{part}__event_type"])[idx]
    near = (et != 2) & (ecmd - t < H)
    res["near_switch_rows"] = int(near.sum())
    for name, pn in pred_norm.items():
        p = pn * rs
        lo_p, hi_p = band_split(p)
        e = dict(mse_norm=float(((p / rs - true / rs) ** 2).mean()),
                 skill_vs_zero=skill(p, true),
                 skill_low=skill(lo_p, lo_t), skill_high=skill(hi_p, hi_t),
                 pred_norm_codec_median=float(np.median(np.linalg.norm(
                     p.reshape(len(p), -1), axis=1))),
                 near_switch_skill=(skill(p[near], true[near])
                                    if near.any() else None),
                 other_skill=(skill(p[~near], true[~near])
                              if (~near).any() else None))
        per_task = {}
        for tk in sorted(set(D["task"].tolist())):
            m = D["task"] == tk
            per_task[tk] = dict(rows=int(m.sum()),
                                skill=skill(p[m], true[m]),
                                rstar_norm_median=float(np.median(
                                    np.linalg.norm(true[m].reshape(
                                        int(m.sum()), -1), axis=1))))
        e["per_task"] = per_task
        res[name] = e
    return res


def main():
    ap = argparse.ArgumentParser(description="K-16: диагностика голов")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--labels", default="reports/k16/labels.npz")
    ap.add_argument("--h18-cache", default="data/k15f/h18_cache")
    ap.add_argument("--models", default="data/k16/models.pt")
    ap.add_argument("--train-report", default="reports/k16/train.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="reports/k16/diag.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import torch
    import k16_policy as kp
    import k16_train as kt
    data, _meta = kt.load_inputs(a.labels, a.h18_cache, verify=False)
    L = dict(np.load(a.labels, allow_pickle=False))
    dev = torch.device(a.device)
    ck, goals, heads = kp.load_models(a.models, torch, dev)
    rs = np.asarray(ck["norms"]["r_sigma"], np.float64)
    res = {}
    for name, part, key in (("B", "train", "selB"), ("V", "val_sel",
                                                     "selV")):
        D = data[name]
        pred = predict(torch, goals, heads, D, dev)
        res[name] = metrics(D, pred, rs, L, part, np.asarray(L[key]))
    rep = json.load(open(a.train_report))
    res["curves"] = {k: v.get("history") for k, v in rep["train"].items()
                     if isinstance(v, dict) and "history" in v}
    res["note"] = ("нормы — в единицах кодека, до α; срез около "
                   "экспертного переключения — диагностика разметки")
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    for name in ("B", "V"):
        r = res[name]
        print(f"  {name}: строк {r['rows']}, MSE нуля (норм.) "
              f"{r['zero_mse_norm']:.3f}, ||r*|| медиана "
              f"{r['rstar_norm_codec']['median']:.4f}, p95 "
              f"{r['rstar_norm_codec']['p95']:.4f}; доля энергии плавной "
              f"части {r['energy_share_low']:.3f}; строк у переключения "
              f"{r['near_switch_rows']}")
        for h in ("direct", "goal_flat", "goal_band", "goal_band_now"):
            e = r[h]
            ns = e["near_switch_skill"]
            print(f"    {h:14} MSE {e['mse_norm']:.3f}, skill "
                  f"{100 * e['skill_vs_zero']:+.1f}% (плавн. "
                  f"{100 * e['skill_low']:+.1f}%, ост. "
                  f"{100 * e['skill_high']:+.1f}%), у переключения "
                  f"{(100 * ns if ns is not None else float('nan')):+.1f}%, "
                  f"||r̂|| {e['pred_norm_codec_median']:.4f}")
    print("  кривые (val): " + "; ".join(
        f"{k}: " + ",".join(f"{x:.3f}" for x in v)
        for k, v in res["curves"].items()))
    print(f"  сводка: {a.out}")
    return 0


def selftest():
    rng = np.random.default_rng(0)
    r = rng.standard_normal((50, 8, 6))
    lo, hi = band_split(r)
    assert np.allclose(lo + hi, r)
    D = km.dct_matrix()
    c_hi = np.einsum("ft,ntc->nfc", D, hi)
    assert np.allclose(c_hi[:, :km.N_LOW], 0, atol=1e-10)
    assert np.isclose((lo ** 2).sum() + (hi ** 2).sum(), (r ** 2).sum())
    assert skill(r, r) == 1.0 and abs(skill(np.zeros_like(r), r)) < 1e-12
    L = {"x__event_cmd_idx": np.array([5, 20, 9]),
         "x__step": np.array([0, 0, 5]),
         "x__event_type": np.array([0, 1, 2])}
    Dd = dict(rstar=rng.standard_normal((3, 8, 6)),
              task=np.array(["a", "a", "b"]))
    m = metrics(Dd, dict(h=np.zeros((3, 8, 6))), np.ones(6), L, "x",
                np.arange(3))
    assert m["near_switch_rows"] == 1                    # конец не считается
    assert m["h"]["skill_vs_zero"] == 0.0
    print("самопроверка k16_diag пройдена: разбиение по DCT, энергия, "
          "skill, срез у переключения без «конца»")
    return 0


if __name__ == "__main__":
    sys.exit(main())
