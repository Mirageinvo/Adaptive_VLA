#!/usr/bin/env python3
"""K-40 B: использует ли декодер подцели. Офлайн, на val-окнах сбора B.

Для TreeB_Mix и FlatB_Mix (оба сида обучения) входы и веса прежние; меняется
только источник подцелей, подаваемых декодеру:
  pred  — собственный прогноз модели (как при исполнении);
  copy  — копии z0 на всех горизонтах;
  true  — фактически наблюдавшиеся z2/z4/z6/z8.
Для каждого варианта: ошибка действий после compose против цели (доли
half², среднее), отдельно на спасениях (R) и успехах T1 (T1a ∪ T1b);
ошибка B против цели; RMS поправки; RMS разницы действий между вариантами.
Только чтение: загрузчик k40b_train.load_windows (полная проверка сбора и
PCA), чекпойнты сверяются с отчётом обучения. Выход —
reports/k40b/goalcheck_v1.json.
"""
import argparse
from pathlib import Path


MODELS = ("TreeB_Mix", "FlatB_Mix")
GROUPS = dict(rescue=("R",), t1=("T1a", "T1b"))


def run(torch, kt, model, A, idx, K):
    F, B, T, Z = kt.tensors(torch, A, idx, K, "cpu")
    out = {}
    with torch.no_grad():
        c = model.ctx(F)
        z0 = Z[:, 0]
        Bn = B[..., :6] / K.half
        g = model.goals(c, z0, Bn)
        src = dict(pred=g, copy=[z0] * 4, true=[Z[:, k] for k in (1, 2, 3, 4)])
        acts = {}
        for name, gs in src.items():
            d = model.decode(c, [z0] + list(gs), Bn)
            a = kt.compose(torch, B, d, K.half, K.L, K.U)
            acts[name] = a
            out[name] = dict(
                action_err=float(kt.action_err(torch, B, d, T, K)),
                delta_rms=float(torch.sqrt((d ** 2).mean())))
        out["B_err"] = float(kt.action_err(torch, B, torch.zeros_like(
            B[..., :6]), T, K))
        for x, y in (("pred", "copy"), ("pred", "true"), ("copy", "true")):
            dd = (acts[x][..., :6] - acts[y][..., :6]) / K.half
            out[f"diff_{x}_{y}_rms"] = float(torch.sqrt((dd ** 2).mean()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect", type=Path,
                    default=Path("data/k40b/collect/v1"))
    ap.add_argument("--pca", type=Path, default=Path("data/k40b/pca.npz"))
    ap.add_argument("--models", type=Path, default=Path("data/k40b/models/v1"))
    ap.add_argument("--train-report", type=Path,
                    default=Path("reports/k40b/train_v1.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("reports/k40b/goalcheck_v1.json"))
    a = ap.parse_args()
    import torch
    import k36_protocol as pc
    import k37_confirm as kcf
    import k40b_train as kt
    A, meta, C, plan_sha, P, _prep = kt.load_windows(a.collect, a.pca)
    K = kt.Consts(torch, C, "cpu")
    trep = pc.load_json(str(a.train_report))
    if trep.get("collect_plan") != plan_sha:
        raise SystemExit("отчёт обучения ≠ плану сбора")
    M = kt.build(torch)
    rep = dict(kind="k40b_goalcheck", collect_plan=plan_sha,
               pca_sha256=kcf.sha256(a.pca), results={})
    for name in MODELS:
        for s in (0, 1):
            p = a.models / f"{name}_s{s}.pt"
            ck = torch.load(p, map_location="cpu")
            for k in ("collect_plan", "pca_sha256", "config", "params",
                      "train_code_sha256"):
                if ck.get(k) != trep.get(k):
                    raise SystemExit(f"{p}: {k} ≠ отчёту обучения")
            m = (M["Tree"]() if ck["kind"] == "tree" else
                 M["Flat"](ck["params"]["flat_hidden"]))
            m.load_state_dict(ck["state"])
            m.eval()
            for grp, roles in GROUPS.items():
                idx = kt.split_idx(meta, roles, "val")
                r = run(torch, kt, m, A, idx, K)
                r["n"] = int(len(idx))
                rep["results"][f"{name}_s{s}_{grp}"] = r
                print(f"  {name} s{s} [{grp}, {len(idx)} окон]: B "
                      f"{r['B_err']:.6f}; действия pred/copy/true "
                      f"{r['pred']['action_err']:.6f} / "
                      f"{r['copy']['action_err']:.6f} / "
                      f"{r['true']['action_err']:.6f}; Δrms "
                      f"{r['pred']['delta_rms']:.4f} / "
                      f"{r['copy']['delta_rms']:.4f} / "
                      f"{r['true']['delta_rms']:.4f}; разница действий "
                      f"pred−copy {r['diff_pred_copy_rms']:.5f}, pred−true "
                      f"{r['diff_pred_true_rms']:.5f}", flush=True)
    pc.save_json(str(a.out), rep)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
