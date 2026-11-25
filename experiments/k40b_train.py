#!/usr/bin/env python3
"""K-40 B: обучение TreeB / FlatB на окнах k40b_collect и офлайн-отчёт.

Данные: полный набор 384 эпизодов сбора (журналы закрыты, каждая запись
проверена k40b_collect.checked, единая размерность D), PCA и нормировка —
k40b_pca (K-21 train). Окно: вход — признаки всех узлов T32 в o0
(x0/x1/x2 + временные веса W0–W2), z0 и исходный план B; цели — команды
руки окна (T1: B, спасения: k_a2) и подцели z2/z4/z6/z8.
z = [PCA64(4 группы визуальных токенов)/std, (state11 − μ)/σ] ∈ R^75.
Контекст: для команды c уровня l — Σ_n W_l[c, n]·P_l(x_l[n]); контекст
интервала [a, b) — среднее по командам.
TreeB: ĝ8 = z0 + G(h_0:8, z0, B); ĝ4 = (z0 + ĝ8)/2 + M(h_0:8, z0, ĝ8);
       ĝ2 = (z0 + ĝ4)/2 + M(h_0:4, z0, ĝ4); ĝ6 = (ĝ4 + ĝ8)/2 + M(h_4:8, ĝ4, ĝ8).
FlatB: [ĝ2, ĝ4, ĝ6, ĝ8] = z0 + H(h_0:8, z0, B), сопоставимое число
       параметров.
Общий декодер: Δ_seg = cap·tanh(D(h_seg, z_нач, z_кон, B_seg)), рука =
B + Δ·half (клип в L/U), схват — из B; последний слой инициализирован 0.
Потери: actions (MSE в долях half) + λ·goals (на каждом горизонте:
полусумма средних ошибок визуальной и физической частей) + β·Δ².
Сначала WARM эпох декодер получает истинные подцели, затем — собственные
предсказанные (дети дерева всегда получают предсказанных родителей).
Наборы (train-окна): Mix = R + T1a; T1 = T1a + T1b. Модели: TreeB-Mix,
FlatB-Mix, TreeB-T1; два сида обучения.
Отчёт (val-окна): ошибка действий модели и B против цели — отдельно на
спасениях и T1; ошибки подцелей по горизонтам (и копии z0); по задачам и
сидам; параметры. Чекпойнты — data/k40b/models.
"""
import argparse
import json
from pathlib import Path

import numpy as np

H_DIM, N_VIS, N_S11 = 768, 64, 11
Z_DIM = N_VIS + N_S11
D_CTX = 128
HID = 256
CAP = 0.25
LAMBDA, BETA = 0.5, 0.01
EPOCHS, WARM, LR, WD, BS = 300, 60, 1e-3, 1e-4, 256
MODELS = dict(TreeB_Mix=("tree", "Mix"), FlatB_Mix=("flat", "Mix"),
              TreeB_T1=("tree", "T1"))
ROLES = dict(Mix=("R", "T1a"), T1=("T1a", "T1b"))
SEEDS = (0, 1)
PTS = (0, 2, 4, 6, 8)


# --- данные -------------------------------------------------------------------------------------
def load_windows(out, pca_path):
    """-> dict массивов по окнам + метаданные; полная проверка сбора."""
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    import k40b_collect as kc
    plan = pc.load_json(str(out / "plan.json"))
    if kcf.canon_sha({k: v for k, v in plan.items() if k != "sha256"}) != \
            plan["sha256"]:
        raise SystemExit("plan.json сбора ≠ своему SHA")
    _c2, k36 = kcf.load_plan(plan["archive"])
    C = kh.consts(k36)
    eps = {e["eid"]: e for e in plan["episodes"]}
    for k in range(kc.N_PARTS):
        led = json.load(open(kc.part_dir(out, k) / "ledger.json"))
        mine = {e for e, v in eps.items() if v["part"] == k}
        if led["pending"] is not None or set(led["done"]) != mine:
            raise SystemExit(f"сбор part {k} не завершён")
        spent = sum(eps[e]["phys"] for e in mine) + \
            kc.RESERVE * len(led["crashes"])
        if led["spent"] != {kc.SCOPE: spent}:
            raise SystemExit(f"сбор part {k}: расход не сходится")
    P = np.load(pca_path)
    cols = {k: [] for k in ("B", "target", "vis", "s11", *(
        f"F_{x}" for x in ("x0", "x1", "x2", "W0", "W1", "W2")))}
    meta, dims = [], set()
    for e in plan["episodes"]:
        row = kc.checked(out, e, plan, C["q"])
        dims.add(row["vis_dim"])
        _jp, dp = kc.ep_files(out, e)
        with np.load(dp, allow_pickle=False) as z:
            for i in range(kc.N_WIN):
                cols["B"].append(z["B"][i])
                cols["target"].append(z["target"][i])
                cols["vis"].append(z["p_vis"][i].reshape(5, -1))
                cols["s11"].append(z["p_state11"][i])
                for x in ("x0", "x1", "x2", "W0", "W1", "W2"):
                    cols[f"F_{x}"].append(z[f"F_{x}"][i])
                meta.append(dict(eid=e["eid"], role=e["role"],
                                 split=e["split"], task=e["task"],
                                 seed=e["seed"], win=i))
    if len(dims) != 1:
        raise SystemExit(f"разная D {dims}")
    if 4 * dims.pop() != P["mean"].shape[0]:
        raise SystemExit("D сбора ≠ D PCA")
    A = {k: np.stack(v).astype(np.float32) for k, v in cols.items()}
    A["z"] = featurize(A["vis"], A["s11"], P)
    del A["vis"], A["s11"]
    return A, meta, C, plan["sha256"]


def featurize(vis, s11, P):
    """vis [N, 5, 4D], s11 [N, 5, 11] -> z [N, 5, 75]."""
    v = (vis - P["mean"]) @ P["comps"].T / P["proj_std"]
    s = (s11 - P["s11_mean"]) / P["s11_std"]
    return np.concatenate([v, s], -1).astype(np.float32)


# --- модели ------------------------------------------------------------------------------------
def build(torch):
    nn = torch.nn

    def mlp(i, o, h=HID, zero=True):
        m = nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h),
                          nn.GELU(), nn.Linear(h, o))
        if zero:
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)
        return m

    class Ctx(nn.Module):
        def __init__(self):
            super().__init__()
            self.P = nn.ModuleList([nn.Sequential(nn.Linear(H_DIM, D_CTX),
                                                  nn.LayerNorm(D_CTX))
                                    for _ in range(3)])

        def forward(self, F):
            """-> контекст команд [N, 8, 3·D_CTX]."""
            out = []
            for l, (x, W) in enumerate(((F["x0"], F["W0"]),
                                        (F["x1"], F["W1"]),
                                        (F["x2"], F["W2"]))):
                out.append(torch.einsum("bcn,bnd->bcd", W[:, :8],
                                        self.P[l](x)))
            return torch.cat(out, -1)

    def span(c, a, b):
        return c[:, a:b].mean(1)

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.ctx = Ctx()
            self.dec = mlp(3 * D_CTX + 2 * Z_DIM + 12, 12)

        def decode(self, c, goals, Bn):
            """goals: [z0, g2, g4, g6, g8] -> Δ [N, 8, 6] (доли half)."""
            out = []
            for s in range(4):
                a, b = 2 * s, 2 * s + 2
                x = torch.cat([span(c, a, b), goals[s], goals[s + 1],
                               Bn[:, a:b].flatten(1)], -1)
                out.append(CAP * torch.tanh(self.dec(x)).view(-1, 2, 6))
            return torch.cat(out, 1)

    class Tree(Base):
        def __init__(self):
            super().__init__()
            self.G = mlp(3 * D_CTX + Z_DIM + 48, Z_DIM)
            self.M = mlp(3 * D_CTX + 2 * Z_DIM + 2, Z_DIM)

        def goals(self, c, z0, Bn):
            g8 = z0 + self.G(torch.cat([span(c, 0, 8), z0, Bn.flatten(1)],
                                       -1))
            one = lambda k: torch.nn.functional.one_hot(  # noqa: E731
                torch.full((len(z0),), k, device=z0.device), 2).float()

            def mid(a, b, za, zb, lev):
                return (za + zb) / 2 + self.M(torch.cat(
                    [span(c, a, b), za, zb, one(lev)], -1))
            g4 = mid(0, 8, z0, g8, 0)
            g2 = mid(0, 4, z0, g4, 1)
            g6 = mid(4, 8, g4, g8, 1)
            return [g2, g4, g6, g8]

    class Flat(Base):
        def __init__(self, hid):
            super().__init__()
            self.H = mlp(3 * D_CTX + Z_DIM + 48, 4 * Z_DIM, h=hid)

        def goals(self, c, z0, Bn):
            o = self.H(torch.cat([span(c, 0, 8), z0, Bn.flatten(1)], -1))
            return [z0 + o[:, k * Z_DIM:(k + 1) * Z_DIM] for k in range(4)]

    def nparams(m):
        return sum(p.numel() for p in m.parameters())

    def flat_matched():
        """Ширина H, при которой параметров не меньше, чем у G + M."""
        t = nparams(Tree())
        for h in range(64, 2048, 8):
            if nparams(Flat(h)) >= t:
                return h
        raise SystemExit("ширина Flat не подобрана")
    return dict(Tree=Tree, Flat=Flat, nparams=nparams,
                flat_matched=flat_matched)


def tensors(torch, A, idx, half, dev):
    F = {k: torch.as_tensor(A[f"F_{k}"][idx], device=dev)
         for k in ("x0", "x1", "x2", "W0", "W1", "W2")}
    hb = torch.as_tensor(half, dtype=torch.float32, device=dev)
    Bn = torch.as_tensor(A["B"][idx][:, :, :6], device=dev) / hb
    Tn = torch.as_tensor(A["target"][idx][:, :, :6], device=dev) / hb
    Z = torch.as_tensor(A["z"][idx], device=dev)
    return F, Bn, Tn, Z


def forward(model, F, Bn, Z, true_goals):
    c = model.ctx(F)
    z0 = Z[:, 0]
    g = model.goals(c, z0, Bn)
    use = [Z[:, k] for k in (1, 2, 3, 4)] if true_goals else g
    d = model.decode(c, [z0] + use, Bn)
    return g, d


def goal_err(g, Z):
    """-> [4 горизонта, 2 части] средняя квадратичная ошибка."""
    out = []
    for k, gk in enumerate(g):
        e = (gk - Z[:, k + 1]) ** 2
        out.append([e[:, :N_VIS].mean(), e[:, N_VIS:].mean()])
    return out


def losses(model, F, Bn, Tn, Z, true_goals):
    g, d = forward(model, F, Bn, Z, true_goals)
    la = ((Bn + d - Tn) ** 2).mean()
    lg = sum(0.5 * (v + s) for v, s in goal_err(g, Z)) / 4
    lk = (d ** 2).mean()
    return la + LAMBDA * lg + BETA * lk, dict(a=la, g=lg, k=lk)


def train_one(torch, M, kind, A, idx, half, seed, dev, flat_hid,
              epochs=EPOCHS, warm=WARM, log=print):
    torch.manual_seed(seed)
    model = (M["Tree"]() if kind == "tree" else M["Flat"](flat_hid)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    F, Bn, Tn, Z = tensors(torch, A, idx, half, dev)
    rng = np.random.default_rng(seed)
    n = len(idx)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, BS):
            b = torch.as_tensor(perm[i:i + BS], device=dev)
            loss, _p = losses(model, {k: v[b] for k, v in F.items()},
                              Bn[b], Tn[b], Z[b], ep < warm)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if log and (ep + 1) % 50 == 0:
            with torch.no_grad():
                _l, p = losses(model, F, Bn, Tn, Z, False)
            log(f"    эпоха {ep + 1}: действия {float(p['a']):.5f}, "
                f"подцели {float(p['g']):.4f}, Δ² {float(p['k']):.5f}")
    return model


def evaluate(torch, model, A, idx, half, dev):
    model.eval()
    F, Bn, Tn, Z = tensors(torch, A, idx, half, dev)
    with torch.no_grad():
        g, d = forward(model, F, Bn, Z, False)
        a_model = ((Bn + d - Tn) ** 2).mean((1, 2)).cpu().numpy()
        a_base = ((Bn - Tn) ** 2).mean((1, 2)).cpu().numpy()
        ge = [[float(v), float(s)] for v, s in goal_err(g, Z)]
        gc = [[float(v), float(s)] for v, s in goal_err(
            [Z[:, 0]] * 4, Z)]
        dn = float((d ** 2).mean())
    return dict(a_model=a_model, a_base=a_base, goals=ge, goals_copy=gc,
                delta2=dn)


def split_idx(meta, roles, split):
    return np.array([i for i, m in enumerate(meta)
                     if m["role"] in roles and m["split"] == split])


def report_block(torch, model, A, meta, half, dev, roles):
    out = {}
    for name, rs in (("rescue", ("R",)), ("t1", ("T1a", "T1b"))):
        idx = split_idx(meta, rs, "val")
        if not len(idx):
            continue
        r = evaluate(torch, model, A, idx, half, dev)
        by_task = {}
        for t in sorted({meta[i]["task"] for i in idx}):
            sel = np.array([meta[i]["task"] == t for i in idx])
            by_task[str(t)] = dict(model=float(r["a_model"][sel].mean()),
                                   base=float(r["a_base"][sel].mean()))
        by_seed = {}
        for s in (0, 1):
            sel = np.array([meta[i]["seed"] == s for i in idx])
            by_seed[str(s)] = dict(model=float(r["a_model"][sel].mean()),
                                   base=float(r["a_base"][sel].mean()))
        out[name] = dict(n=int(len(idx)), action_model=float(
            r["a_model"].mean()), action_base=float(r["a_base"].mean()),
            goals=r["goals"], goals_copy_z0=r["goals_copy"],
            delta2=r["delta2"], by_task=by_task, by_seed=by_seed)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect", type=Path, default=Path("data/k40b/collect/v1"))
    ap.add_argument("--pca", type=Path, default=Path("data/k40b/pca.npz"))
    ap.add_argument("--out", type=Path, default=Path("data/k40b/models/v1"))
    ap.add_argument("--report", type=Path,
                    default=Path("reports/k40b/train_v1.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import torch
    import k36_protocol as pc
    import k37_confirm as kcf
    if a.out.exists():
        raise SystemExit(f"{a.out} уже есть — укажите новый --out")
    A, meta, C, plan_sha = load_windows(a.collect, a.pca)
    half = np.asarray(C["half"], np.float32)
    M = build(torch)
    fh = M["flat_matched"]()
    params = dict(TreeB=M["nparams"](M["Tree"]()),
                  FlatB=M["nparams"](M["Flat"](fh)), flat_hidden=fh)
    print(f"  окон {len(meta)}; параметры {params}", flush=True)
    a.out.mkdir(parents=True)
    rep = dict(kind="k40b_train", collect_plan=plan_sha,
               pca_sha256=kcf.sha256(a.pca), params=params,
               config=dict(D_CTX=D_CTX, HID=HID, CAP=CAP, LAMBDA=LAMBDA,
                           BETA=BETA, EPOCHS=EPOCHS, WARM=WARM, LR=LR, WD=WD,
                           BS=BS), models={})
    for name, (kind, data) in MODELS.items():
        idx = split_idx(meta, ROLES[data], "train")
        rep["models"][name] = dict(train_windows=int(len(idx)), seeds={})
        for s in SEEDS:
            print(f"  {name} сид {s}: окон {len(idx)}", flush=True)
            model = train_one(torch, M, kind, A, idx, half, s, a.device, fh)
            torch.save(dict(state=model.state_dict(), kind=kind, data=data,
                            seed=s, flat_hidden=fh, collect_plan=plan_sha),
                       a.out / f"{name}_s{s}.pt")
            rep["models"][name]["seeds"][str(s)] = report_block(
                torch, model, A, meta, half, a.device, ROLES[data])
    pc.save_json(str(a.report), rep)
    print_report(rep)
    return 0


def print_report(rep):
    for name, r in rep["models"].items():
        for part in ("rescue", "t1"):
            v = [r["seeds"][str(s)].get(part) for s in SEEDS]
            if not all(v):
                continue
            am = np.mean([x["action_model"] for x in v])
            ab = v[0]["action_base"]
            g = np.mean([[h[0] + h[1] for h in x["goals"]] for x in v], 0)
            gc = [h[0] + h[1] for h in v[0]["goals_copy_z0"]]
            print(f"  {name} [{part}, val {v[0]['n']} окон]: действия "
                  f"{am:.5f} (B {ab:.5f}); подцели 2/4/6/8 "
                  f"{np.round(g, 3).tolist()} (копия z0 "
                  f"{np.round(gc, 3).tolist()})")


def selftest():
    import torch
    torch.manual_seed(0)
    M = build(torch)
    T, Fl = M["Tree"](), M["Flat"](M["flat_matched"]())
    assert M["nparams"](Fl) >= M["nparams"](T)
    rng = np.random.default_rng(0)
    N = 64
    A = dict(B=rng.uniform(-.5, .5, (N, 8, 7)).astype(np.float32),
             z=rng.normal(size=(N, 5, Z_DIM)).astype(np.float32),
             **{f"F_x{l}": rng.normal(size=(N, n, H_DIM)).astype(np.float32)
                for l, n in ((0, 4), (1, 8), (2, 16))},
             **{f"F_W{l}": rng.dirichlet(np.ones(n), (N, 20)).astype(
                 np.float32) for l, n in ((0, 4), (1, 8), (2, 16))})
    A["target"] = A["B"].copy()
    A["target"][:, :, :6] += 0.05
    half = np.ones(6, np.float32)
    idx = np.arange(N)
    F, Bn, Tn, Z = tensors(torch, A, idx, half, "cpu")
    # нулевая инициализация: Δ = 0, ĝ дерева = z0 и середины
    g, d = forward(T, F, Bn, Z, False)
    assert float(d.abs().max()) == 0.0
    assert torch.allclose(g[3], Z[:, 0]) and torch.allclose(g[1], Z[:, 0])
    # дети дерева зависят от предсказанного родителя
    with torch.no_grad():
        T.G[-1].bias.add_(1.0)
    g2, _d = forward(T, F, Bn, Z, False)
    assert not torch.allclose(g2[0], g[0]) and not torch.allclose(g2[2], g[2])
    # подцели на входе — не будущие наблюдения: ĝ не зависит от Z[:, 1:]
    Z2 = Z.clone()
    Z2[:, 1:] += 5.0
    g3, _d = forward(T, F, Bn, Z2, False)
    assert all(torch.allclose(x, y) for x, y in zip(g2, g3))
    # обучение уменьшает ошибку действий
    m = train_one(torch, M, "tree", A, idx, half, 0, "cpu", 64, epochs=40,
                  warm=10, log=None)
    r = evaluate(torch, m, A, idx, half, "cpu")
    assert r["a_model"].mean() < r["a_base"].mean()
    mf = train_one(torch, M, "flat", A, idx, half, 0, "cpu",
                   M["flat_matched"](), epochs=40, warm=10, log=None)
    rf = evaluate(torch, mf, A, idx, half, "cpu")
    assert rf["a_model"].mean() < rf["a_base"].mean()
    # поправка ограничена CAP при любом выходе декодера
    with torch.no_grad():
        T.dec[-1].bias.fill_(100.0)
    _g, dd = forward(T, F, Bn, Z, False)
    assert float(dd.abs().max()) <= CAP + 1e-6
    P = dict(mean=np.zeros(8, np.float32), comps=np.eye(N_VIS, 8,
                                                        dtype=np.float32),
             proj_std=np.ones(N_VIS, np.float32),
             s11_mean=np.zeros(11, np.float32),
             s11_std=np.ones(11, np.float32))
    z = featurize(np.ones((3, 5, 8), np.float32), np.ones((3, 5, 11),
                                                          np.float32), P)
    assert z.shape == (3, 5, Z_DIM)
    print("самопроверка k40b_train пройдена: нулевая инициализация (Δ = 0, "
          "ĝ = z0/середины), дети дерева от предсказанного родителя, "
          "будущие наблюдения не на входе, Flat ≥ параметров Tree, обучение "
          "снижает ошибку действий, featurize 75")
    return 0


if __name__ == "__main__":
    import sys
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
