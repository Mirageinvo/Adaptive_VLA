#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16b: «контекст» против «контекст + явная цель» (без частотного
разделения).

ВОПРОС: помогает ли ЯВНОЕ представление цели использовать доступную
информацию для коррекции действий. Явная цель вычисляется из того же
скрытого вектора goal_hist, что есть в общем контексте, — новой
информации она не добавляет.

ГОЛОВЫ (обучаются на половине B K-16, выбор эпохи — по val_sel):
  ctx       h18 (проекция по позициям), a0, признаки состояния с историей,
            скрытый вектор ЗАМОРОЖЕННОГО goal_hist K-16 -> поправка руки
            r [8, 6] (нормирована по каналу) + вспомогательный выход цели;
  ctx_goal  то же + явная ПРЕДСКАЗАННАЯ цель goal_hist (Δpos, 6D, схват
            через tanh, тип через softmax).
Одинаковые выборки, вспомогательные потери (вес AUX), сид, оптимизатор,
эпохи, ширина (бюджеты параметров отличаются лишь на вход цели, < 2%).
goal_hist не переобучается (берётся из data/k16/models.pt).
Правило применения поправки — то же, что K-16 (α, предел, единицы — из
чекпойнта K-16; k16_models.apply_correction).

ВЫХОД: data/k16b/models.pt, reports/k16b/train.json.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k16_models as km  # noqa: E402

HIDDEN = 208
OUT = "data/k16b/models.pt"
REPORT = "reports/k16b/train.json"
K16_MODELS = "data/k16/models.pt"
HEADS = ("ctx", "ctx_goal")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def build_heads(torch, d_model=768, n_pos=16):
    nn = torch.nn
    C = km.build(torch, d_model=d_model, n_pos=n_pos)

    class CtxHead(nn.Module):
        """Общий контекст (+ явная цель при use_goal) -> r [8, 6], aux."""

        def __init__(self, hidden=HIDDEN, use_goal=False):
            super().__init__()
            self.use_goal = bool(use_goal)
            self.enc = C["GoalNet"]().enc.__class__()
            d = (self.enc.out_dim + km.H * 7 + km.STATE_FEAT
                 + km.GOAL_HIDDEN + (km.GOAL_DIM if use_goal else 0))
            self.trunk = nn.Sequential(nn.Linear(d, hidden), nn.GELU())
            self.r = nn.Sequential(nn.GELU(), nn.Linear(hidden,
                                                        km.H * km.N_ARM))
            self.aux = nn.Sequential(nn.GELU(), nn.Linear(hidden,
                                                          km.GOAL_DIM))

        def forward(self, h18, a0, sf, goal):
            """goal: [feat (GOAL_HIDDEN) | предсказанная цель (GOAL_DIM)]."""
            feat = goal[:, :km.GOAL_HIDDEN]
            parts = [self.enc(h18), a0.flatten(1), sf, feat]
            if self.use_goal:
                g = goal[:, km.GOAL_HIDDEN:]
                parts.append(torch.cat([g[:, :9], torch.tanh(g[:, 9:10]),
                                        torch.softmax(g[:, 10:13], -1)], 1))
            z = self.trunk(torch.cat(parts, 1))
            return self.r(z).view(-1, km.H, km.N_ARM), self.aux(z)
    return C, CtxHead


def add_goal_inputs(torch, goal_hist, D, dev):
    """Скрытый вектор и прогноз замороженного goal_hist -> D["ctxin"]
    [N, GOAL_HIDDEN + GOAL_DIM]."""
    F, P = [], []
    with torch.no_grad():
        for s in range(0, len(D["age"]), 1024):
            i = slice(s, s + 1024)
            h = torch.as_tensor(np.asarray(D["h18"][i]), device=dev)
            a0 = torch.as_tensor(D["a0"][i], device=dev).float()
            sf = torch.as_tensor(D["sf_hist"][i], device=dev)
            hid = goal_hist.hidden(h, a0, sf)
            F.append(hid.cpu().numpy())
            P.append(goal_hist.body[-1](hid).cpu().numpy())
    D["pred_goal_hist"] = np.concatenate(P)
    D["feat_goal_hist"] = np.concatenate(F)
    D["ctxin"] = np.concatenate([D["feat_goal_hist"], D["pred_goal_hist"]],
                                1).astype(np.float32)


def train_heads(torch, B, V, dev, d_model, n_pos, epochs=None, log=print):
    import k16_train as kt
    epochs = epochs or kt.EPOCHS
    _C, CtxHead = build_heads(torch, d_model, n_pos)
    dpos_sigma = torch.tensor(np.maximum(B["dpos"].std(0), 1e-4),
                              dtype=torch.float32, device=dev)
    r_sigma = torch.tensor(np.maximum(B["rstar"].reshape(-1, 6).std(0),
                                      1e-5), dtype=torch.float32, device=dev)
    keys_g = ("dpos", "drot6d", "grip", "etype")
    heads, rep = {}, {}
    for name in HEADS:
        m = kt.new_model(torch, CtxHead, HIDDEN,
                         use_goal=(name == "ctx_goal")).to(dev)

        def step(model, d, idx):
            b = kt.to_t(torch, d, idx, dev, ("h18", "a0", "rstar", "sf_hist",
                                             "ctxin") + keys_g)
            r, aux = model(b["h18"], b["a0"].float(), b["sf_hist"].float(),
                           b["ctxin"].float())
            loss = ((r - b["rstar"].float() / r_sigma) ** 2).mean()
            if model.training:
                loss = loss + km.AUX * kt.goal_loss(
                    torch, aux, {k: (b[k].float() if k != "etype" else b[k])
                                 for k in keys_g}, dpos_sigma)
            return loss
        best, h = kt.fit(torch, m, B, V, step, dev, epochs)
        heads[name] = (m, "ctxin", "sf_hist")
        rep[name] = dict(best_epoch=best[0], val_r_mse_norm=best[1],
                         history=h, params=km.n_params(m), hidden=HIDDEN,
                         use_goal=(name == "ctx_goal"))
        log(f"  {name}: эпоха {best[0]}, MSE поправки (норм.) на val_sel "
            f"{best[1]:.4f}, параметров {km.n_params(m)}")
    return heads, rep, dict(dpos_sigma=dpos_sigma.cpu().numpy(),
                            r_sigma=r_sigma.cpu().numpy())


def main():
    ap = argparse.ArgumentParser(description="K-16b: обучение")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--labels", default="reports/k16/labels.npz")
    ap.add_argument("--h18-cache", default="data/k15f/h18_cache")
    ap.add_argument("--k16-models", default=K16_MODELS)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--report", default=REPORT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import torch
    import k16_policy as kp
    import k16_train as kt
    try:
        data, meta = kt.load_inputs(a.labels, a.h18_cache, verify=True)
        dev = torch.device(a.device)
        ck16, goals, _h = kp.load_models(a.k16_models, torch, dev)
        if ck16["provenance"]["labels_sha1"] != sha(a.labels):
            raise SystemExit("goal_hist K-16 обучен на другой разметке")
        gh = goals["goal_hist"]
        for p in gh.parameters():
            p.requires_grad_(False)
        B, V = data["B"], data["V"]
        for D in (B, V):
            add_goal_inputs(torch, gh, D, dev)
        d_model, n_pos = int(B["h18"].shape[2]), int(B["h18"].shape[1])
        heads, rep, norms = train_heads(torch, B, V, dev, d_model, n_pos)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    if not np.allclose(norms["r_sigma"], ck16["norms"]["r_sigma"],
                       rtol=1e-5):
        print("  ОТКАЗ: нормировка поправки не равна K-16 (другая B?)")
        return 3
    import k15h_protocol as pc
    scale = pc.harness_scale()
    ev = kt.evaluate(torch, {"goal_hist": (gh, "sf_hist")}, heads, norms, V,
                     data["A"], scale, float(ck16["clip_env_norm"]), dev)
    code = {f: sha(os.path.join(HERE, f))[:12] for f in (
        "k16b_train.py", "k16_models.py", "k16_train.py")}
    ck = dict(
        heads={k: dict(state=m.state_dict(), use_goal=(k == "ctx_goal"),
                       hidden=HIDDEN, cls="CtxHead")
               for k, (m, _g, _s) in heads.items()},
        goal_hist=dict(state=gh.state_dict()), norms=norms,
        clip_env_norm=float(ck16["clip_env_norm"]),
        alpha=float(ck16["alpha"]), units=ck16.get("units"),
        scale_arm=list(ck16["scale_arm"]), d_model=d_model, n_pos=n_pos,
        provenance=dict(labels=os.path.abspath(a.labels),
                        labels_sha1=sha(a.labels),
                        h18_manifest_sha1=meta["h18_manifest_sha1"],
                        k16_models=os.path.abspath(a.k16_models),
                        k16_models_sha1=sha(a.k16_models), code=code))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(ck, a.out + ".tmp")
    os.replace(a.out + ".tmp", a.out)
    res = dict(train=rep, val_sel={k: v for k, v in ev.items()
                                   if k in HEADS},
               models=dict(path=os.path.abspath(a.out), sha1=sha(a.out)),
               code=code,
               note=("явная цель вычисляется из того же скрытого вектора — "
                     "проверяется представление, а не новая информация"))
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1,
                  default=lambda x: x.tolist() if hasattr(x, "tolist")
                  else str(x))
    print(json.dumps(res["val_sel"], ensure_ascii=False, indent=1))
    print(f"  модели: {a.out}")
    return 0


def selftest():
    import torch
    import k16_train as kt
    rng = np.random.default_rng(0)
    B, V = kt._synth(300, rng), kt._synth(100, rng)
    C = km.build(torch)
    gh = kt.new_model(torch, C["GoalNet"])
    for D in (B, V):
        add_goal_inputs(torch, gh, D, torch.device("cpu"))
        assert D["ctxin"].shape[1] == km.GOAL_HIDDEN + km.GOAL_DIM
    heads, rep, _n = train_heads(torch, B, V, torch.device("cpu"), 768, 16,
                                 epochs=2, log=lambda *x: None)
    pc, pg = rep["ctx"]["params"], rep["ctx_goal"]["params"]
    assert pg > pc and (pg - pc) / pc < 0.02
    # ctx не видит явную цель: изменение её части входа не меняет выход
    m = heads["ctx"][0]
    x = torch.tensor(B["ctxin"][:4])
    x2 = x.clone()
    x2[:, km.GOAL_HIDDEN:] += 5.0
    h = torch.tensor(B["h18"][:4])
    a0 = torch.tensor(B["a0"][:4])
    sf = torch.tensor(B["sf_hist"][:4])
    with torch.no_grad():
        assert torch.equal(m(h, a0, sf, x)[0], m(h, a0, sf, x2)[0])
        mg = heads["ctx_goal"][0]
        assert not torch.equal(mg(h, a0, sf, x)[0], mg(h, a0, sf, x2)[0])
    # одинаковая инициализация общей части при одном сиде
    _C, CtxHead = build_heads(torch)
    m1 = kt.new_model(torch, CtxHead, HIDDEN)
    m2 = kt.new_model(torch, CtxHead, HIDDEN)
    assert all(torch.equal(p1, p2) for p1, p2 in zip(m1.parameters(),
                                                     m2.parameters()))
    print("самопроверка k16b_train пройдена: контекст со скрытым вектором, "
          "ctx не видит явную цель, ctx_goal видит, бюджеты в пределах 2%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
