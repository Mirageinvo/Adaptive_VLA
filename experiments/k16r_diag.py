#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16r: масштаб расхождения входов головы и исходной δ (без раскаток).

По готовым блокам K-16r (диагностические артефакты) и learned-блокам K-16:
для каждого импульса — max|Δh18|, max|Δδ|, относительная разность норм δ
в исполняемых координатах; отдельно импульсы ДО первого вмешательства в
батче (ни у одной среды блока не было импульса раньше — ни в этом прогоне,
ни в learned) и ПОСЛЕ вмешательства соседа. Причина (выравнивание батча и
т. п.) здесь не устанавливается — только масштаб.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def load(p):
    d = json.load(open(p))
    z = np.load(os.path.join(os.path.dirname(p), d["actions_npz"]))
    return d, {k: z[k] for k in z.files}


def impulse_rows(zr, zl, scale):
    imp_r = np.asarray(zr["k16_impulse"], bool)
    imp_l = np.asarray(zl["k16_impulse"], bool)
    first_any = min([c for c in (np.nonzero(imp_r.any(1))[0].tolist()
                                 + np.nonzero(imp_l.any(1))[0].tolist())]
                    or [10 ** 9])
    rows = []
    for k, b in zip(*np.nonzero(imp_r)):
        kl = np.nonzero(imp_l[:, b])[0]
        if not len(kl) or kl[0] != k:
            rows.append(dict(env=int(b), call=int(k) + 1, matched=False))
            continue
        dr = np.asarray(zr["k16_d_codec_orig"][k, b], np.float64)
        dl = np.asarray(zl["k16_d_codec"][k, b], np.float64)
        hr = zr["k16_h18"][k, b].astype(np.float32)
        hl = zl["k16_h18"][k, b].astype(np.float32)
        er, el = dr * scale, dl * scale
        rows.append(dict(
            env=int(b), call=int(k) + 1, matched=True,
            after_neighbor=bool(k > first_any),
            h18_maxdiff=float(np.abs(hr - hl).max()),
            delta_maxdiff=float(np.abs(dr - dl).max()),
            delta_max=float(np.abs(dl).max()),
            norm_rel_diff=float(abs(np.linalg.norm(er) - np.linalg.norm(el))
                                / max(np.linalg.norm(el), 1e-12))))
    return rows


def summarize(rows):
    out = {}
    for name, sel in (("до вмешательства в батче",
                       [r for r in rows if r["matched"]
                        and not r["after_neighbor"]]),
                      ("после вмешательства соседа",
                       [r for r in rows if r["matched"]
                        and r["after_neighbor"]])):
        if not sel:
            out[name] = dict(n=0)
            continue
        h = np.array([r["h18_maxdiff"] for r in sel])
        dd = np.array([r["delta_maxdiff"] for r in sel])
        nr = np.array([r["norm_rel_diff"] for r in sel])
        out[name] = dict(n=len(sel), h18_differs=int((h > 0).sum()),
                         delta_differs=int((dd > 0).sum()),
                         h18_maxdiff_max=float(h.max()),
                         delta_maxdiff_max=float(dd.max()),
                         norm_rel_diff_max=float(nr.max()),
                         norm_rel_diff_median=float(np.median(nr)))
    out["unmatched_impulses"] = sum(1 for r in rows if not r["matched"])
    return out


def main():
    ap = argparse.ArgumentParser(description="K-16r: диагностика")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="reports/k16r/diag_batch.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import torch
    R = sorted(glob.glob("reports/k16r/run/p*/"))[0]
    L = sorted(glob.glob("reports/k16/run/p*/"))[0]
    plan = json.load(open("reports/k16r/plan.json"))
    ck = torch.load(plan["inputs"]["models"], map_location="cpu",
                    weights_only=False)
    scale = np.asarray(ck["scale_arm"], np.float64)
    rows = []
    for p in sorted(glob.glob(R + "goal_band_rot_s*_t*_i25.json")):
        d, zr = load(p)
        t = int(d["task_id"])
        _dl, zl = load(L + f"goal_band_hist_event_t{t}_i25.json")
        for r in impulse_rows(zr, zl, scale):
            rows.append(dict(r, block=os.path.basename(p)))
    res = dict(blocks=len({r["block"] for r in rows}), impulses=len(rows),
               summary=summarize(rows), rows=rows,
               note="масштаб расхождения; причина не устанавливается")
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(f"  блоков {res['blocks']}, импульсов {res['impulses']}")
    for k, v in res["summary"].items():
        print(f"    {k}: {v}")
    print(f"  сводка: {a.out}")
    return 0


def selftest():
    C, B = 6, 3
    z = {"k16_impulse": np.zeros((C, B), bool),
         "k16_d_codec": np.zeros((C, B, 8, 6)),
         "k16_d_codec_orig": np.zeros((C, B, 8, 6)),
         "k16_h18": np.zeros((C, B, 2, 2), np.float16)}
    zl = {k: v.copy() for k, v in z.items()}
    for zz in (z, zl):
        zz["k16_impulse"][2, 0] = True
        zz["k16_impulse"][4, 1] = True
    zl["k16_d_codec"][2, 0] = 0.01
    z["k16_d_codec_orig"][2, 0] = 0.01
    zl["k16_d_codec"][4, 1] = 0.02
    z["k16_d_codec_orig"][4, 1] = 0.0201
    z["k16_h18"][4, 1] = 0.5
    rows = impulse_rows(z, zl, np.ones(6))
    s = summarize(rows)
    assert s["до вмешательства в батче"]["n"] == 1
    assert s["до вмешательства в батче"]["delta_differs"] == 0
    assert s["после вмешательства соседа"]["h18_differs"] == 1
    assert s["после вмешательства соседа"]["delta_differs"] == 1
    print("самопроверка k16r_diag пройдена: разделение до/после соседа, "
          "расхождения h18 и δ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
