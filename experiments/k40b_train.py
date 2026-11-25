#!/usr/bin/env python3
"""K-40 B: обучение TreeB / FlatB на окнах k40b_collect и офлайн-отчёт.

Данные: только завершённый сбор — план сбора (SHA, код, архив c2),
k40b_collect.restore по обеим частям (закрытые журналы, расход, каждая
запись checked), ровно 384 эпизода и равные числа ролей R/T1a/T1b в
train и val; единая размерность D = D PCA. PCA и нормировка — k40b_pca
(pca.json: SHA npz, источник = отпечаток весов T32_s0 и prepare K-36;
формы, конечность, положительные масштабы).
Окно: вход — признаки всех узлов T32 в o0 (x0/x1/x2 + временные веса
W0–W2), z0, исходный план B; цели — команды руки окна (T1: B,
спасения: k_a2) и подцели z2/z4/z6/z8.
z = [PCA64(4 группы визуальных токенов)/std, (state11 − μ)/σ] ∈ R^75.
Контекст: для команды c уровня l — Σ_n W_l[c, n]·P_l(x_l[n]); контекст
интервала [a, b) — среднее по командам.
TreeB: ĝ8 = z0 + G(h_0:8, z0, B); ĝ4 = (z0 + ĝ8)/2 + M(h_0:8, z0, ĝ8);
       ĝ2 = (z0 + ĝ4)/2 + M(h_0:4, z0, ĝ4); ĝ6 = (ĝ4 + ĝ8)/2 + M(h_4:8, ĝ4, ĝ8).
FlatB: [ĝ2, ĝ4, ĝ6, ĝ8] = z0 + H(h_0:8, h_0:4, h_4:8, z0, B) — те же
       интервальные контексты, без обусловливания родителями; параметров
       не меньше, чем у TreeB.
Действие (общее для обучения, оценки и исполнителя — compose): рука =
clip(B + Δ·half, L, U), схват = B; Δ = 0 даёт B побитово.
Потери: L_a/s_a + λ·L_g/s_g + β·Δ², где s_a, s_g — масштабы на train
при Δ = 0 и подцелях-копиях z0 (одинаково для всех моделей). WARM эпох
декодер получает истинные подцели, затем собственные; дети дерева всегда
получают предсказанных родителей.
Выбор чекпойнта: после WARM каждые EVAL эпох — та же нормированная потеря
на val-окнах своих ролей в режиме собственных предсказаний; берётся
минимум.
TreeB-T1 (T1a + T1b): у всех целей T1 = B, при Δ, начатом с 0, градиент
действий нулевой — это проверка сохранения T32, а не контроль обучения.
"""
import argparse
from pathlib import Path

import numpy as np

H_DIM, N_VIS, N_S11 = 768, 64, 11
Z_DIM = N_VIS + N_S11
D_CTX = 128
HID = 256
CAP = 0.25
LAMBDA, BETA = 0.5, 0.01
EPOCHS, WARM, EVAL, LR, WD, BS = 300, 60, 10, 1e-3, 1e-4, 256
MODELS = dict(TreeB_Mix=("tree", "Mix"), FlatB_Mix=("flat", "Mix"),
              TreeB_T1=("tree", "T1"))
ROLES = dict(Mix=("R", "T1a"), T1=("T1a", "T1b"))
SEEDS = (0, 1)
N_EPISODES = 384
F_LIST = ("x0", "x1", "x2", "W0", "W1", "W2")


# --- данные -------------------------------------------------------------------------------------
def load_pca(path, k36):
    import k36_protocol as pc
    import k37_confirm as kcf
    rp = Path("reports/k40b/pca.json") if path == Path(
        "data/k40b/pca.npz") else path.with_suffix(".json")
    r = pc.load_json(str(rp))
    if r.get("kind") != "k40b_pca" or r.get("out_sha256") != \
            kcf.sha256(path):
        raise SystemExit("PCA: отчёт отсутствует или SHA npz ≠ отчёту")
    if r.get("k36_prepare_sha1") != k36["sha1"] or \
            r.get("source_weights") != \
            k36["k32"]["seeds"]["0"]["source"]["weights_hash"] or \
            not r.get("source_weights"):
        raise SystemExit("PCA: источник визуальных признаков ≠ T32_s0")
    P = dict(np.load(path))
    D4 = P["mean"].shape[0] if P["mean"].ndim == 1 else -1
    want = dict(mean=(D4,), comps=(N_VIS, D4), proj_std=(N_VIS,),
                s11_mean=(N_S11,), s11_std=(N_S11,))
    for k, shp in want.items():
        if k not in P or P[k].shape != shp or not np.isfinite(P[k]).all():
            raise SystemExit(f"PCA: {k} формы {P.get(k, np.zeros(0)).shape}"
                             f" ≠ {shp} или не конечен")
    if (P["proj_std"] <= 0).any() or (P["s11_std"] <= 0).any():
        raise SystemExit("PCA: неположительный масштаб")
    return P, r


def load_windows(out, pca_path):
    """-> (массивы по окнам, метаданные, C, sha плана сбора, PCA, отчёт)."""
    import k31_collect as kcol
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    import k40b_collect as kc
    plan = pc.load_json(str(out / "plan.json"))
    if kcf.canon_sha({k: v for k, v in plan.items() if k != "sha256"}) != \
            plan["sha256"]:
        raise SystemExit("plan.json сбора ≠ своему SHA")
    here = Path(__file__).resolve().parent
    bad = [n for n, s in plan["code"].items() if kcf.sha256(here / n) != s]
    if bad:
        raise SystemExit(f"код сбора ≠ плану: {bad}")
    c2, k36 = kcf.load_plan(plan["archive"])
    if c2["sha256"] != plan["archive_plan"]:
        raise SystemExit("архив c2 ≠ плану сбора")
    C = kh.consts(k36)
    eps = plan["episodes"]
    if len(eps) != N_EPISODES or len({e["eid"] for e in eps}) != N_EPISODES:
        raise SystemExit(f"эпизодов в плане {len(eps)} ≠ {N_EPISODES}")
    for sp in ("train", "val"):
        n = {r: sum(e["role"] == r and e["split"] == sp for e in eps)
             for r in ("R", "T1a", "T1b")}
        if len(set(n.values())) != 1 or not n["R"]:
            raise SystemExit(f"{sp}: роли не равны {n}")
    rows = {}
    for k in range(kc.N_PARTS):
        lp = kc.part_dir(out, k) / "ledger.json"
        if not lp.exists():
            raise SystemExit(f"сбор part {k}: нет журнала")
        led = kcol.Ledger(str(lp), plan["caps"])
        if led.pending_id() is not None:
            raise SystemExit(f"сбор part {k}: открытый резерв")
        rr = kc.restore(out, plan, k, led, C["q"])
        if len(rr) != sum(e["part"] == k for e in eps):
            raise SystemExit(f"сбор part {k} не завершён")
        rows.update(rr)
    if len(rows) != N_EPISODES:
        raise SystemExit(f"записей {len(rows)} ≠ {N_EPISODES}")
    P, prep = load_pca(pca_path, k36)
    dims = {r["vis_dim"] for r in rows.values()}
    if len(dims) != 1 or 4 * next(iter(dims)) != P["mean"].shape[0]:
        raise SystemExit(f"D сбора {dims} ≠ D PCA {P['mean'].shape[0] // 4}")
    cols = {k: [] for k in ("B", "target", "vis", "s11",
                            *(f"F_{x}" for x in F_LIST))}
    meta = []
    for e in eps:
        _jp, dp = kc.ep_files(out, e)
        with np.load(dp, allow_pickle=False) as z:
            for i in range(kc.N_WIN):
                cols["B"].append(z["B"][i])
                cols["target"].append(z["target"][i])
                cols["vis"].append(z["p_vis"][i].reshape(5, -1))
                cols["s11"].append(z["p_state11"][i])
                for x in F_LIST:
                    cols[f"F_{x}"].append(z[f"F_{x}"][i])
                meta.append(dict(eid=e["eid"], role=e["role"],
                                 split=e["split"], task=e["task"],
                                 seed=e["seed"], win=i))
    A = {k: np.stack(v).astype(np.float32) for k, v in cols.items()}
    A["z"] = featurize(A["vis"], A["s11"], P)
    if not np.isfinite(A["z"]).all():
        raise SystemExit("признаки подцелей не конечны")
    del A["vis"], A["s11"]
    return A, meta, C, plan["sha256"], P, prep


def featurize(vis, s11, P):
    """vis [N, 5, 4D], s11 [N, 5, 11] -> z [N, 5, 75]."""
    v = (vis - P["mean"]) @ P["comps"].T / P["proj_std"]
    s = (s11 - P["s11_mean"]) / P["s11_std"]
    return np.concatenate([v, s], -1).astype(np.float32)


# --- действие -------------------------------------------------------------------------------------
def compose(torch, B, d, half, L, U):
    """B [N, 8, 7] (кодек), Δ [N, 8, 6] (доли half) -> итоговые команды:
    рука clip(B + Δ·half, min(L, B), max(U, B)), схват из B. Δ = 0 -> B
    побитово (B + 0 = B; B внутри своих границ, даже если T32 вышел за L/U
    в пределах допуска), градиент по Δ не блокируется."""
    b = B[..., :6]
    arm = torch.minimum(torch.maximum(b + d * half, torch.minimum(L, b)),
                        torch.maximum(U, b))
    return torch.cat([arm, B[..., 6:]], -1)


def compose_np(B, d, half, L, U):
    """То же для исполнителя (numpy, float32)."""
    B = np.asarray(B, np.float32)
    d = np.asarray(d, np.float32)
    b = B[..., :6]
    L = np.asarray(L, np.float32)
    U = np.asarray(U, np.float32)
    arm = np.minimum(np.maximum(b + d * np.asarray(half, np.float32),
                                np.minimum(L, b)), np.maximum(U, b))
    return np.concatenate([arm.astype(np.float32), B[..., 6:]], -1)


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
            self.H = mlp(9 * D_CTX + Z_DIM + 48, 4 * Z_DIM, h=hid)

        def goals(self, c, z0, Bn):
            o = self.H(torch.cat([span(c, 0, 8), span(c, 0, 4),
                                  span(c, 4, 8), z0, Bn.flatten(1)], -1))
            return [z0 + o[:, k * Z_DIM:(k + 1) * Z_DIM] for k in range(4)]

    def nparams(m):
        return sum(p.numel() for p in m.parameters())

    def flat_matched():
        """Наименьшая ширина H, при которой параметров не меньше, чем у
        TreeB."""
        t = nparams(Tree())
        for h in range(64, 4096, 8):
            if nparams(Flat(h)) >= t:
                return h
        raise SystemExit("ширина Flat не подобрана")
    return dict(Tree=Tree, Flat=Flat, nparams=nparams,
                flat_matched=flat_matched)


class Consts:
    def __init__(self, torch, C, dev):
        f = lambda x: torch.as_tensor(np.asarray(x, np.float32),  # noqa
                                      device=dev)
        self.half, self.L, self.U = f(C["half"]), f(C["L"]), f(C["U"])


def tensors(torch, A, idx, K, dev):
    F = {k: torch.as_tensor(A[f"F_{k}"][idx], device=dev) for k in F_LIST}
    B = torch.as_tensor(A["B"][idx], device=dev)
    T = torch.as_tensor(A["target"][idx], device=dev)
    Z = torch.as_tensor(A["z"][idx], device=dev)
    return F, B, T, Z


def forward(model, F, B, Z, K, true_goals):
    c = model.ctx(F)
    z0 = Z[:, 0]
    Bn = B[..., :6] / K.half
    g = model.goals(c, z0, Bn)
    use = [Z[:, k] for k in (1, 2, 3, 4)] if true_goals else g
    d = model.decode(c, [z0] + use, Bn)
    return g, d


def goal_err(g, Z):
    """-> [4 горизонта][визуальная, физическая] — средние квадраты."""
    out = []
    for k, gk in enumerate(g):
        e = (gk - Z[:, k + 1]) ** 2
        out.append([e[:, :N_VIS].mean(), e[:, N_VIS:].mean()])
    return out


def action_err(torch, B, d, T, K, per_window=False):
    a = compose(torch, B, d, K.half, K.L, K.U)
    e = ((a[..., :6] - T[..., :6]) / K.half) ** 2
    return e.mean((1, 2)) if per_window else e.mean()


def scales(torch, A, idx, K, dev):
    """Масштабы потерь на train при Δ = 0 и подцелях-копиях z0."""
    F, B, T, Z = tensors(torch, A, idx, K, dev)
    s_a = float(action_err(torch, B, torch.zeros_like(B[..., :6]), T, K))
    s_g = float(sum(0.5 * (v + s) for v, s in goal_err([Z[:, 0]] * 4, Z))
                / 4)
    return dict(s_a=s_a if s_a > 0 else 1.0, s_g=s_g if s_g > 0 else 1.0,
                raw_a=s_a, raw_g=s_g)


def losses(torch, model, F, B, T, Z, K, S, true_goals):
    g, d = forward(model, F, B, Z, K, true_goals)
    la = action_err(torch, B, d, T, K)
    lg = sum(0.5 * (v + s) for v, s in goal_err(g, Z)) / 4
    lk = (d ** 2).mean()
    tot = la / S["s_a"] + LAMBDA * lg / S["s_g"] + BETA * lk
    return tot, dict(a=la, g=lg, k=lk)


def train_one(torch, M, kind, A, idx, vidx, K, S, seed, dev, flat_hid,
              epochs=EPOCHS, warm=WARM, every=EVAL, log=print):
    import copy
    torch.manual_seed(seed)
    model = (M["Tree"]() if kind == "tree" else M["Flat"](flat_hid)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    F, B, T, Z = tensors(torch, A, idx, K, dev)
    Fv, Bv, Tv, Zv = tensors(torch, A, vidx, K, dev)
    rng = np.random.default_rng(seed)
    n = len(idx)
    best = (np.inf, -1, None)
    curve = []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, BS):
            b = torch.as_tensor(perm[i:i + BS], device=dev)
            loss, _p = losses(torch, model, {k: v[b] for k, v in F.items()},
                              B[b], T[b], Z[b], K, S, ep < warm)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep + 1 >= warm and (ep + 1) % every == 0:
            model.eval()
            with torch.no_grad():
                lv, pv = losses(torch, model, Fv, Bv, Tv, Zv, K, S, False)
            curve.append((ep + 1, float(lv), float(pv["a"]), float(pv["g"])))
            if float(lv) < best[0]:
                best = (float(lv), ep + 1, copy.deepcopy(model.state_dict()))
            if log and (ep + 1) % 50 == 0:
                log(f"    эпоха {ep + 1}: val {float(lv):.4f} (действия "
                    f"{float(pv['a']):.5f}, подцели {float(pv['g']):.4f})")
    model.load_state_dict(best[2])
    return model, dict(best_epoch=best[1], best_val=best[0], curve=curve)


def evaluate(torch, model, A, idx, K, dev):
    model.eval()
    F, B, T, Z = tensors(torch, A, idx, K, dev)
    with torch.no_grad():
        g, d = forward(model, F, B, Z, K, False)
        am = action_err(torch, B, d, T, K, True).cpu().numpy()
        ab = action_err(torch, B, torch.zeros_like(d), T, K,
                        True).cpu().numpy()
        ge = [[float(v), float(s)] for v, s in goal_err(g, Z)]
        gc = [[float(v), float(s)] for v, s in goal_err([Z[:, 0]] * 4, Z)]
        dn = float((d ** 2).mean())
        dmax = float(d.abs().max())
    return dict(a_model=am, a_base=ab, goals=ge, goals_copy=gc, delta2=dn,
                delta_max=dmax)


def split_idx(meta, roles, split):
    return np.array([i for i, m in enumerate(meta)
                     if m["role"] in roles and m["split"] == split], int)


def report_block(torch, model, A, meta, K, dev):
    out = {}
    for name, rs in (("rescue", ("R",)), ("t1", ("T1a", "T1b"))):
        idx = split_idx(meta, rs, "val")
        r = evaluate(torch, model, A, idx, K, dev)
        cut = {}
        for key in ("task", "seed"):
            for v in sorted({meta[i][key] for i in idx}):
                sel = np.array([meta[i][key] == v for i in idx])
                cut[f"{key}{v}"] = dict(model=float(r["a_model"][sel].mean()),
                                        base=float(r["a_base"][sel].mean()))
        out[name] = dict(n=int(len(idx)), action_model=float(
            r["a_model"].mean()), action_base=float(r["a_base"].mean()),
            goals=r["goals"], goals_copy_z0=r["goals_copy"],
            delta2=r["delta2"], delta_max=r["delta_max"], cuts=cut)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect", type=Path,
                    default=Path("data/k40b/collect/v1"))
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
    if a.out.exists() or a.report.exists():
        raise SystemExit(f"{a.out} или {a.report} уже есть — новый --out")
    A, meta, C, plan_sha, _P, prep = load_windows(a.collect, a.pca)
    K = Consts(torch, C, a.device)
    M = build(torch)
    fh = M["flat_matched"]()
    params = dict(TreeB=M["nparams"](M["Tree"]()),
                  FlatB=M["nparams"](M["Flat"](fh)), flat_hidden=fh)
    cfg = dict(D_CTX=D_CTX, HID=HID, CAP=CAP, LAMBDA=LAMBDA, BETA=BETA,
               EPOCHS=EPOCHS, WARM=WARM, EVAL=EVAL, LR=LR, WD=WD, BS=BS)
    prov = dict(collect_plan=plan_sha, pca_sha256=kcf.sha256(a.pca),
                pca_source=prep["source_weights"],
                train_code_sha256=kcf.sha256(Path(__file__).resolve()),
                config=cfg, params=params)
    print(f"  окон {len(meta)}; параметры {params}", flush=True)
    a.out.mkdir(parents=True)
    rep = dict(kind="k40b_train", **prov, models={})
    for name, (kind, data) in MODELS.items():
        idx = split_idx(meta, ROLES[data], "train")
        vidx = split_idx(meta, ROLES[data], "val")
        S = scales(torch, A, idx, K, a.device)
        rep["models"][name] = dict(train_windows=int(len(idx)),
                                   val_windows=int(len(vidx)), scales=S,
                                   seeds={})
        for s in SEEDS:
            print(f"  {name} сид {s}: train {len(idx)}, val {len(vidx)}; "
                  f"масштабы {S}", flush=True)
            model, tr = train_one(torch, M, kind, A, idx, vidx, K, S, s,
                                  a.device, fh)
            torch.save(dict(state=model.state_dict(), kind=kind, data=data,
                            seed=s, scales=S, train=tr, **prov),
                       a.out / f"{name}_s{s}.pt")
            rep["models"][name]["seeds"][str(s)] = dict(
                train=tr, **report_block(torch, model, A, meta, K, a.device))
    pc.save_json(str(a.report), rep)
    print_report(rep)
    return 0


def print_report(rep):
    for name, r in rep["models"].items():
        be = [r["seeds"][str(s)]["train"]["best_epoch"] for s in SEEDS]
        for part in ("rescue", "t1"):
            v = [r["seeds"][str(s)][part] for s in SEEDS]
            am = np.mean([x["action_model"] for x in v])
            ab = v[0]["action_base"]
            g = np.mean([[h[0] + h[1] for h in x["goals"]] for x in v], 0)
            gc = [h[0] + h[1] for h in v[0]["goals_copy_z0"]]
            print(f"  {name} [{part}, val {v[0]['n']} окон, эпохи {be}]: "
                  f"действия {am:.5f} (B {ab:.5f}); подцели 2/4/6/8 "
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
    C = dict(half=np.ones(6), L=-np.ones(6), U=np.ones(6))
    K = Consts(torch, C, "cpu")
    idx = np.arange(N)
    F, B, Tt, Z = tensors(torch, A, idx, K, "cpu")
    with torch.no_grad():
        g, d = forward(T, F, B, Z, K, False)
    # нулевая инициализация: Δ = 0 -> B побитово; ĝ = z0 и середины
    assert float(d.abs().max()) == 0.0
    assert torch.equal(compose(torch, B, d, K.half, K.L, K.U), B)
    assert torch.allclose(g[3], Z[:, 0]) and torch.allclose(g[1], Z[:, 0])
    # композиция: клип в L/U, схват из B; numpy-версия совпадает
    Bc = torch.full((1, 8, 7), 0.99)
    dc = torch.full((1, 8, 6), 0.25)
    out = compose(torch, Bc, dc, K.half, K.L, K.U)
    assert torch.allclose(out[..., :6], torch.ones(1, 8, 6)) and \
        torch.equal(out[..., 6], Bc[..., 6])
    assert float(action_err(torch, Bc, dc, torch.ones(1, 8, 7), K)) == 0.0
    assert np.array_equal(compose_np(Bc.numpy(), dc.numpy(), C["half"],
                                     C["L"], C["U"]), out.numpy())
    # B чуть за границей (допуск T32) при Δ = 0 сохраняется побитово
    Bo = torch.full((1, 8, 7), 1.00005)
    assert torch.equal(compose(torch, Bo, torch.zeros(1, 8, 6), K.half,
                               K.L, K.U), Bo)
    # градиент по Δ проходит при нулевой поправке
    dz = torch.zeros(1, 8, 6, requires_grad=True)
    action_err(torch, Bc * 0, dz, torch.full((1, 8, 7), 0.1), K).backward()
    assert float(dz.grad.abs().sum()) > 0
    # дети дерева зависят от предсказанного родителя
    with torch.no_grad():
        T.G[-1].bias.add_(1.0)
        g2, _d = forward(T, F, B, Z, K, False)
    assert not torch.allclose(g2[0], g[0]) and not torch.allclose(g2[2], g[2])
    # будущие наблюдения не на входе
    Z2 = Z.clone()
    Z2[:, 1:] += 5.0
    with torch.no_grad():
        g3, _d = forward(T, F, B, Z2, K, False)
    assert all(torch.allclose(x, y) for x, y in zip(g2, g3))
    # Flat различает половины чанка при том же среднем контексте
    with torch.no_grad():
        Fl.H[-1].weight.normal_()
        c = Fl.ctx(F)
        cs = torch.cat([c[:, 4:8], c[:, 0:4]], 1)
        Bn = B[..., :6] / K.half
        ga = Fl.goals(c, Z[:, 0], Bn)
        Fl.ctx.forward = lambda F_: cs                 # noqa: E731
        gb = Fl.goals(cs, Z[:, 0], Bn)
    assert not torch.allclose(ga[0], gb[0])
    # поправка ограничена CAP
    with torch.no_grad():
        T.dec[-1].bias.fill_(100.0)
        _g, dd = forward(T, F, B, Z, K, False)
    assert float(dd.abs().max()) <= CAP + 1e-6
    # масштабы и обучение с выбором чекпойнта (подцели предсказуемы:
    # будущее = z0 + сдвиг, чтобы проверка шла по действиям)
    A["z"][:, 1:] = A["z"][:, :1] + 0.3
    S = scales(torch, A, idx, K, "cpu")
    assert S["raw_a"] > 0 and S["raw_g"] > 0
    m, tr = train_one(torch, M, "tree", A, idx[:48], idx[48:], K, S, 0,
                      "cpu", 64, epochs=40, warm=10, every=5, log=None)
    assert tr["best_epoch"] >= 10 and tr["best_epoch"] % 5 == 0
    r = evaluate(torch, m, A, idx, K, "cpu")
    assert r["a_model"].mean() < r["a_base"].mean()
    # T1-данные (цель = B): Δ остаётся нулевым — вырожденный контроль
    A1 = dict(A, target=A["B"].copy())
    S1 = scales(torch, A1, idx, K, "cpu")
    m1, _t = train_one(torch, M, "tree", A1, idx[:48], idx[48:], K, S1, 0,
                       "cpu", 64, epochs=20, warm=5, every=5, log=None)
    r1 = evaluate(torch, m1, A1, idx, K, "cpu")
    assert r1["delta_max"] == 0.0
    # featurize
    P = dict(mean=np.zeros(8, np.float32),
             comps=np.eye(N_VIS, 8, dtype=np.float32),
             proj_std=np.ones(N_VIS, np.float32),
             s11_mean=np.zeros(11, np.float32),
             s11_std=np.ones(11, np.float32))
    z = featurize(np.ones((3, 5, 8), np.float32),
                  np.ones((3, 5, 11), np.float32), P)
    assert z.shape == (3, 5, Z_DIM)
    print("самопроверка k40b_train пройдена: Δ = 0 → B побитово; клип L/U и "
          "схват из B (torch = numpy); дети дерева от предсказанного "
          "родителя; будущие наблюдения не на входе; Flat различает половины "
          "чанка и ≥ параметров Tree; CAP; масштабы потерь; выбор чекпойнта "
          "после WARM; T1-контроль вырожден (Δ ≡ 0)")
    return 0


if __name__ == "__main__":
    import sys
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
