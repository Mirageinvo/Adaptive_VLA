#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16: достаточно ли обучения (без раскаток).

Для голов ctx и ctx_goal (K-16b) и goal_band (K-16), на тех же входах
(замороженный goal_hist K-16, половина B, val_sel):
  1. ЁМКОСТЬ: обучение с нуля на фиксированной подвыборке B (512 строк,
     200 эпох) — насколько падает обучающая ошибка;
  2. ОРИЕНТИРЫ, посчитанные только по B: нулевая поправка, средняя r*,
     средняя r* по задаче (для задач без строк — общая средняя); MSE на
     B и val_sel в той же нормировке (r / r_sigma);
  3. ДЛИННОЕ ОБУЧЕНИЕ: 40 эпох, косинусное убывание lr 1e-3 -> 1e-5;
     по эпохам — обучающая (фиксированная подвыборка B, без вспомогательной
     ошибки) и проверочная ошибки; выбор эпохи — по val_sel.
Новые веса — data/k16fit/<голова>.pt; прежние чекпойнты и планы не
меняются. Отчёт — reports/k16/fitcheck.json.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k16_models as km  # noqa: E402

OVERFIT_ROWS, OVERFIT_EPOCHS = 512, 200
LONG_EPOCHS, LR0, LR1 = 40, 1e-3, 1e-5
EVAL_ROWS = 4096
OUTDIR = "data/k16fit"


def baselines(B, V, r_sigma):
    """Нулевая, средняя и средняя по задаче поправки (по B)."""
    rs = np.asarray(r_sigma)
    tb, tv = B["rstar"] / rs, V["rstar"] / rs
    mean = tb.mean(0)
    by = {}
    for t, r in zip(B["task"], tb):
        by.setdefault(t, []).append(r)
    by = {t: np.mean(v, 0) for t, v in by.items()}

    def mse(pred, true):
        return float(((pred - true) ** 2).mean())
    out = {}
    for name, D, T in (("B", B, tb), ("val_sel", V, tv)):
        task_pred = np.stack([by.get(t, mean) for t in D["task"]])
        out[name] = dict(zero=mse(np.zeros_like(T), T),
                         mean=mse(np.broadcast_to(mean, T.shape), T),
                         task_mean=mse(task_pred, T))
    out["val_sel_tasks_without_B_rows"] = sorted(
        set(V["task"].tolist()) - set(by))
    return out


def make_step(torch, kind, dev, r_sigma, dpos_sigma, aux):
    import k16_train as kt
    keys_g = ("dpos", "drot6d", "grip", "etype")
    gk = "ctxin" if kind in ("ctx", "ctx_goal") else "pred_goal_hist"

    def step(model, d, idx, with_aux=True):
        b = kt.to_t(torch, d, idx, dev, ("h18", "a0", "rstar", "sf_hist", gk)
                    + keys_g)
        r, a = model(b["h18"], b["a0"].float(), b["sf_hist"].float(),
                     b[gk].float())
        loss = ((r - b["rstar"].float() / r_sigma) ** 2).mean()
        if aux and with_aux and model.training and a is not None:
            loss = loss + km.AUX * kt.goal_loss(
                torch, a, {k: (b[k].float() if k != "etype" else b[k])
                           for k in keys_g}, dpos_sigma)
        return loss
    return step


def eval_loss(torch, model, step, D, idx):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for s in range(0, len(idx), 1024):
            i = idx[s:s + 1024]
            tot += float(step(model, D, i, with_aux=False)) * len(i)
    return tot / max(len(idx), 1)


def train(torch, model, step, D, V, epochs, lr0, lr1, rows=None, seed=0,
          batch=256, eval_rows=None, dev=None):
    """Adam с косинусным lr; кривые обучающей и проверочной ошибок."""
    rng = np.random.default_rng(seed)
    rows = np.arange(len(D["age"])) if rows is None else rows
    ev_tr = rows if eval_rows is None else eval_rows
    opt = torch.optim.Adam(model.parameters(), lr=lr0, weight_decay=1e-4)
    curve, best = [], None
    vidx = np.arange(len(V["age"])) if V is not None else None
    for ep in range(1, epochs + 1):
        lr = lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * (ep - 1)
                                                     / max(epochs - 1, 1)))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        perm = rng.permutation(rows)
        for s in range(0, len(perm), batch):
            loss = step(model, D, perm[s:s + batch])
            if not bool(torch.isfinite(loss)):
                raise SystemExit(f"нечисловая ошибка на эпохе {ep}")
            opt.zero_grad()
            loss.backward()
            opt.step()
        tr = eval_loss(torch, model, step, D, ev_tr)
        va = eval_loss(torch, model, step, V, vidx) if V is not None \
            else None
        curve.append(dict(epoch=ep, lr=lr, train=tr, val=va))
        if va is not None and (best is None or va < best[1]):
            best = (ep, va, {k: v.detach().clone()
                             for k, v in model.state_dict().items()})
    return curve, best


def build_model(torch, kind, d_model, n_pos):
    import k16_train as kt
    import k16b_train as bt
    C, CtxHead = bt.build_heads(torch, d_model, n_pos)
    if kind == "goal_band":
        _t, w = km.matched_widths(torch, C)
        return kt.new_model(torch, C["GoalBand"], w["GoalBand"])
    return kt.new_model(torch, CtxHead, bt.HIDDEN,
                        use_goal=(kind == "ctx_goal"))


def run_all(torch, B, V, dev, r_sigma, dpos_sigma, kinds, log=print,
            overfit_epochs=OVERFIT_EPOCHS, long_epochs=LONG_EPOCHS,
            outdir=OUTDIR):
    d_model, n_pos = int(B["h18"].shape[2]), int(B["h18"].shape[1])
    rs = torch.tensor(r_sigma, dtype=torch.float32, device=dev)
    ds = torch.tensor(dpos_sigma, dtype=torch.float32, device=dev)
    rng = np.random.default_rng(20261016)
    small = np.sort(rng.choice(len(B["age"]), min(OVERFIT_ROWS,
                                                  len(B["age"])),
                               replace=False))
    ev_rows = np.sort(rng.choice(len(B["age"]), min(EVAL_ROWS,
                                                    len(B["age"])),
                                 replace=False))
    res = dict(baselines=baselines(B, V, r_sigma), heads={})
    for kind in kinds:
        aux = kind in ("ctx", "ctx_goal")
        step = make_step(torch, kind, dev, rs, ds, aux)
        m = build_model(torch, kind, d_model, n_pos).to(dev)
        c_small, _b = train(torch, m, step, B, None, overfit_epochs, LR0,
                            LR0, rows=small, dev=dev)
        m = build_model(torch, kind, d_model, n_pos).to(dev)
        c_long, best = train(torch, m, step, B, V, long_epochs, LR0, LR1,
                             eval_rows=ev_rows, dev=dev)
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{kind}.pt")
        torch.save(dict(state=best[2], kind=kind, best_epoch=best[0],
                        best_val=best[1]), path)
        res["heads"][kind] = dict(
            overfit=dict(rows=len(small), epochs=overfit_epochs,
                         first=c_small[0]["train"],
                         last=c_small[-1]["train"],
                         ratio=c_small[-1]["train"] / max(
                             c_small[0]["train"], 1e-12)),
            long=dict(curve=c_long, best_epoch=best[0], best_val=best[1]),
            weights=path, params=km.n_params(m))
        log(f"  {kind}: подвыборка {len(small)} — ошибка "
            f"{c_small[0]['train']:.3f} -> {c_small[-1]['train']:.3f}; "
            f"длинное обучение — лучшая эпоха {best[0]}, val "
            f"{best[1]:.4f}, train в конце {c_long[-1]['train']:.3f}")
    return res


def main():
    ap = argparse.ArgumentParser(description="K-16: достаточность обучения")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--labels", default="reports/k16/labels.npz")
    ap.add_argument("--h18-cache", default="data/k15f/h18_cache")
    ap.add_argument("--k16-models", default="data/k16/models.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="reports/k16/fitcheck.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import torch
    import k16_policy as kp
    import k16_train as kt
    import k16b_train as bt
    dev = torch.device(a.device)
    data, _m = kt.load_inputs(a.labels, a.h18_cache, verify=False)
    ck16, goals, _h = kp.load_models(a.k16_models, torch, dev)
    gh = goals["goal_hist"]
    B, V = data["B"], data["V"]
    for D in (B, V):
        bt.add_goal_inputs(torch, gh, D, dev)
    res = run_all(torch, B, V, dev, np.asarray(ck16["norms"]["r_sigma"]),
                  np.asarray(ck16["norms"]["dpos_sigma"]),
                  ("ctx", "ctx_goal", "goal_band"))
    b = res["baselines"]
    print(f"  ориентиры (MSE, норм.): val_sel ноль {b['val_sel']['zero']:.4f}"
          f", средняя {b['val_sel']['mean']:.4f}, по задаче "
          f"{b['val_sel']['task_mean']:.4f}; B ноль {b['B']['zero']:.4f}, "
          f"средняя {b['B']['mean']:.4f}, по задаче {b['B']['task_mean']:.4f}")
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(f"  сводка: {a.out}")
    return 0


def selftest():
    import tempfile
    import torch
    import k16_train as kt
    import k16b_train as bt
    rng = np.random.default_rng(0)
    B, V = kt._synth(200, rng), kt._synth(60, rng)
    C = km.build(torch)
    gh = kt.new_model(torch, C["GoalNet"])
    for D in (B, V):
        bt.add_goal_inputs(torch, gh, D, torch.device("cpu"))
    bl = baselines(B, V, np.full(6, 0.05))
    assert bl["B"]["mean"] <= bl["B"]["zero"]
    assert bl["B"]["task_mean"] <= bl["B"]["mean"] + 1e-12
    with tempfile.TemporaryDirectory() as td:
        res = run_all(torch, B, V, torch.device("cpu"), np.full(6, 0.05),
                      np.full(3, 0.1), ("ctx", "goal_band"),
                      log=lambda *x: None, overfit_epochs=30, long_epochs=3,
                      outdir=td)
        for k in ("ctx", "goal_band"):
            h = res["heads"][k]
            assert h["overfit"]["last"] < h["overfit"]["first"]
            assert len(h["long"]["curve"]) == 3
            assert os.path.exists(h["weights"])
            lrs = [c["lr"] for c in h["long"]["curve"]]
            assert lrs[0] > lrs[-1]
    print("самопроверка k16_fitcheck пройдена: ориентиры, переобучение "
          "подвыборки, кривые и убывание lr, отдельное сохранение")
    return 0


if __name__ == "__main__":
    sys.exit(main())
