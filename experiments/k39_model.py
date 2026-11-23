#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-39 f2: лёгкие головы поверх кеша (энкодер, генератор, селектор, flat).

Команды:
  fit --run-id f2 --seed S [--device D]    обучение для сида S
  offline --run-id f2 [--device D]          отчёт на diag-сценах и сёстрах
  selftest
Данные: шарды k39_cache (data_plan закреплён). Строки:
  nom  — номинальное окно (o0, o4, B, Y);
  pair — две сестры ± из одного o0, обе с has_y и has_o4 (Y₁, Y₂);
  sis1 — единственная живая сестра (обычная строка с одной меткой).
Пара принимается, только если t, B и obs0_hash сестёр совпадают между
собой и с номинальным окном того же эпизода и шага.
Кадры: [:, ::-1] по ширине (как k29_policy.Inputs), центр 196, bilinear
antialias до 96, /255. Состояние: k21_model.state11, стандартизация по
train своего сида.
Энкодер: 3 × Conv3×3 s2 (16/24/32, SiLU), avgpool 2, Linear -> 32 на
камеру, общие веса камер и моментов; предобучение с MLP 75→96→28 на B[:4]
по o0 и Y[:4] по o4 (сцены генератора), затем заморозка.
x0 = [v0 64, s11 11, B 8×7 (рука (a−mid)/half, схват — знак кодека), t/600]
= 132; xmid = x0 + [v 64, s11 11] выбранного наблюдения = 207.
Генератор 132→96→56: C1, C2 (рука clip(b + 2·tanh r, −1, 1), схват — логит).
Пара: min(L(C1,Y₁)+L(C2,Y₂), L(C1,Y₂)+L(C2,Y₁)) / 2; одна метка:
min(L(C1,Y), L(C2,Y)); L = SmoothL1(β 0.05) руки + 0.25·BCE схвата.
Селектор 207→32→3 (B, C1, C2): Σ pₖ·(dₖ − min d), d — среднее по 4 шагам
(MAE руки + 0.25·[знак схвата ≠]); исполнение — argmax, равенство -> B, C1.
Flat 207→106→28. Контроль числа ветвей: одна альтернатива (132→96→28)
и селектор 207→32→2. Режимы входа селектора/flat: old/old, fresh/fresh,
fresh RGB/old state, old RGB/fresh state (равные веса).
Каждая сестра в энкодере, селекторе и flat имеет половину веса её пары.
Генераторы и flat делят порог повышенного веса по номинальным gen train.
Визуальные признаки стандартизуются по train генератора (o0 и o4), как
и состояние; выход селектора стартует с нулей (равные вероятности).
Сцены: gen — энкодер и генератор; sel — селектор; flat — gen + sel;
внутренняя проверка — по хешу 4 из 32 gen и 2 из 16 sel на задачу;
лучшая эпоха по ней. diag — только отчёт.
"""
import argparse
import hashlib
import io
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

IMG = 96
VIS = 32
SEEDS = (0, 1)
TASKS = (0, 6, 8)
CFG = dict(lr=1e-3, wd=1e-4, batch=64, beta=0.05, grip_w=0.25,
           epochs=dict(enc=15, gen=25, sel=15, flat=25), hard_w=2.0,
           val_gen=4, val_sel=2)
MODES = ((0, 0), (1, 1), (1, 0), (0, 1))
# Варианты обучения голов на том же кеше: base — исходный (эпохи CFG);
# long — головы до 100 эпох, выбор по внутренней проверке; nearb — тот же
# бюджет, выходы генератора и flat стартуют около B (малые разные веса),
# логит схвата = 2·sign(B_grip) + Δ.
VARIANTS = dict(base=dict(head_epochs=None, nearb=False),
                long=dict(head_epochs=100, nearb=False),
                nearb=dict(head_epochs=100, nearb=True))     # (rgb, state): 0 — o0, 1 — o4


def h256(s):
    return hashlib.sha256(s.encode()).hexdigest()


# --- кадры и строки ---------------------------------------------------------------------------
def prep_rgb(torch, rgb):
    """[2, H, W, 3] uint8 -> [2, 3, 96, 96] uint8."""
    out = []
    for c in range(rgb.shape[0]):
        x = torch.tensor(np.asarray(rgb[c])[:, ::-1].copy()).permute(
            2, 0, 1)[None].float()
        H, W = x.shape[-2:]
        top, left = int(round((H - 196) / 2)), int(round((W - 196) / 2))
        x = x[..., top:top + 196, left:left + 196]
        x = torch.nn.functional.interpolate(x, size=(IMG, IMG),
                                            mode="bilinear",
                                            align_corners=False,
                                            antialias=True)
        out.append(torch.clamp(torch.round(x), 0, 255).to(torch.uint8)[0])
    return torch.stack(out).numpy()


def load_rows(torch, plan, shard, ok):
    """plan — data_plan; shard(job) -> путь; ok(path, job) -> meta|None|False.
    -> (rows, counts)."""
    rows, cnt = [], dict(nom=0, pair=0, sis1=0, dead=0, pair_reject=0)
    nom_idx = {}
    for e in plan["episodes"]:
        if e["split"] not in ("gen", "sel", "diag"):
            continue
        job = f"cache/{e['eid']}"
        p = shard(job)
        if not ok(p, job):
            raise SystemExit(f"шард {job} отсутствует или не проверен")
        with np.load(p, allow_pickle=False) as z:
            times = z["t"]
            if not np.array_equal(times, e["windows"]):
                raise SystemExit(f"шард {job}: окна ≠ data_plan")
            if not len(times):
                continue
            data = {k: z[k] for k in ("rgb0", "rgb4", "s0", "s4", "B", "Y",
                                      "obs0_hash")}
        for i, t in enumerate(times):
            r = dict(kind="nom", seed=e["seed"], task=e["task"],
                     case=e["case_id"], split=e["split"], eid=e["eid"],
                     t=int(t), img0=prep_rgb(torch, data["rgb0"][i]),
                     img4=prep_rgb(torch, data["rgb4"][i]),
                     s0=data["s0"][i].astype(np.float64),
                     s4=data["s4"][i].astype(np.float64),
                     B=data["B"][i].astype(np.float32),
                     Y=data["Y"][i, :4].astype(np.float32),
                     h0=str(data["obs0_hash"][i]))
            nom_idx[(e["eid"], r["t"])] = len(rows)
            rows.append(r)
            cnt["nom"] += 1
    groups = {}
    for s in plan["sisters"]:
        if not s["missing"]:
            groups.setdefault((s["eid"], s["t"]), []).append(s)
    by = {e["eid"]: e for e in plan["episodes"]}
    for (eid, t), specs in sorted(groups.items()):
        e = by[eid]
        sis = []
        for s in sorted(specs, key=lambda x: -x["sign"]):
            job = f"cache/{s['job']}"
            p = shard(job)
            if not ok(p, job):
                raise SystemExit(f"шард {job} отсутствует или не проверен")
            with np.load(p, allow_pickle=False) as z:
                x = dict(sign=int(z["sign"]), t=int(z["t"]),
                         B=z["B"].astype(np.float32), h0=str(z["obs0_hash"]),
                         live=bool(z["has_y"]) and bool(z["has_o4"]))
                if x["live"]:
                    x.update(img4=prep_rgb(torch, z["rgb4"]),
                             s4=z["s4"].astype(np.float64),
                             Y=z["Y"][:4].astype(np.float32))
                sis.append(x)
        n = rows[nom_idx[(eid, t)]] if (eid, t) in nom_idx else None
        if (n is None or [x["sign"] for x in sis] != [1, -1] or
                any(x["t"] != t or x["h0"] != n["h0"] or
                    not np.array_equal(x["B"], n["B"]) for x in sis)):
            cnt["pair_reject"] += 1
            continue
        live = [x for x in sis if x["live"]]
        cnt["dead"] += len(sis) - len(live)
        base = dict(seed=e["seed"], task=e["task"], case=e["case_id"],
                    split=e["split"], eid=eid, t=t, img0=n["img0"],
                    s0=n["s0"], B=n["B"], h0=n["h0"])
        if len(live) == 2:
            rows.append(dict(base, kind="pair", img4=live[0]["img4"],
                             s4=live[0]["s4"], Y=live[0]["Y"],
                             img4b=live[1]["img4"], s4b=live[1]["s4"],
                             Yb=live[1]["Y"]))
            cnt["pair"] += 1
        elif len(live) == 1:
            rows.append(dict(base, kind="sis1", img4=live[0]["img4"],
                             s4=live[0]["s4"], Y=live[0]["Y"]))
            cnt["sis1"] += 1
    return rows, cnt


def val_scenes(plan):
    """Внутренняя проверка: по хешу 4 из gen и 2 из sel на задачу."""
    out = set()
    for t in TASKS:
        sp = plan["splits"][str(t)]
        for nm, n in (("gen", CFG["val_gen"]), ("sel", CFG["val_sel"])):
            ids = sorted(sp[nm], key=lambda c: h256(f"k39-f2-val|{t}|{c}"))
            out |= set(ids[:n])
    return out


# --- представление действий ------------------------------------------------------------------
def norm_arm(a, C):
    return (np.asarray(a, np.float64)[..., :6] - C["mid"]) / C["half"]


def chunk_feat(B, C):
    """[8, 7] кодек -> [56]: рука нормирована, схват — знак."""
    f = np.concatenate([norm_arm(B, C), np.sign(B[..., 6:7])], -1)
    return f.reshape(*B.shape[:-2], 56).astype(np.float32)


def tail_dist(arm_n, grip_sign, Y, C):
    """Расстояние к учителю: среднее по 4 шагам (MAE руки + 0.25·[знак ≠])."""
    ya = norm_arm(Y, C)
    return (np.abs(arm_n - ya).mean(-1) +
            CFG["grip_w"] * (np.sign(grip_sign) != np.sign(Y[..., 6]))
            ).mean(-1)


# --- модели --------------------------------------------------------------------------------------
def build(torch):
    nn = torch.nn

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1), nn.SiLU(),
                nn.Conv2d(16, 24, 3, 2, 1), nn.SiLU(),
                nn.Conv2d(24, 32, 3, 2, 1), nn.SiLU(), nn.AvgPool2d(2))
            self.p = nn.Linear(32 * (IMG // 16) ** 2, VIS)

        def forward(self, x):                   # [N, 2, 3, H, W] float
            N = x.shape[0]
            y = self.c(x.reshape(N * 2, 3, IMG, IMG)).reshape(N * 2, -1)
            return self.p(y).reshape(N, 2 * VIS)

    def mlp(i, h, o):
        return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))
    return dict(Enc=Enc, mlp=mlp)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def tail_from(torch, r, gl, b_arm, b_grip=None):
    """r [..,4,6], gl [..,4], b_arm [..,4,6] -> (arm_n, grip_logit).
    b_grip (знак штатного схвата) задан — логит = 2·b_grip + gl (nearb)."""
    if b_grip is not None:
        gl = gl + 2 * b_grip
    return torch.clamp(b_arm + 2 * torch.tanh(r), -1, 1), gl


def init_near_b(torch, net, tag):
    """Выходной слой: малые разные веса (≈ 1e−3), нулевой bias."""
    g = torch.Generator().manual_seed(int(h256(tag)[:8], 16))
    with torch.no_grad():
        w = net[2].weight
        w.copy_((torch.randn(w.shape, generator=g) * 1e-3).to(w.device))
        net[2].bias.zero_()


def row_loss(torch, arm, gl, ya, yg):
    """[..,4,6], [..,4] против нормированной руки и метки схвата {0,1}."""
    F = torch.nn.functional
    la = F.smooth_l1_loss(arm, ya, beta=CFG["beta"], reduction="none"
                          ).mean((-2, -1))
    lg = F.binary_cross_entropy_with_logits(gl, yg, reduction="none"
                                            ).mean(-1)
    return la + CFG["grip_w"] * lg


def pair_loss(torch, alts, Y1, Y2):
    """alts: ((arm1, gl1), (arm2, gl2)); Y = (ya, yg). Перестановка меток
    не меняет значение."""
    l11 = row_loss(torch, *alts[0], *Y1)
    l22 = row_loss(torch, *alts[1], *Y2)
    l12 = row_loss(torch, *alts[0], *Y2)
    l21 = row_loss(torch, *alts[1], *Y1)
    return torch.minimum(l11 + l22, l12 + l21) / 2


def single_loss(torch, alts, Y):
    ls = torch.stack([row_loss(torch, *a, *Y) for a in alts], -1)
    return ls.min(-1).values


def regret_loss(torch, logits, d):
    """Σ pₖ·(dₖ − min d)."""
    p = torch.softmax(logits, -1)
    return (p * (d - d.min(-1, keepdim=True).values)).sum(-1)


def action_errors(arm, grip, ya, yg):
    """Ошибки руки и схвата по четырём шагам."""
    arm_error = (arm - ya).abs().mean((-2, -1))
    grip_error = ((grip > 0) != (yg > 0)).float().mean(-1)
    return arm_error, grip_error


def action_distance(arm, grip, ya, yg):
    arm_error, grip_error = action_errors(arm, grip, ya, yg)
    return arm_error + CFG["grip_w"] * grip_error


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum()


# --- обучение ------------------------------------------------------------------------------------
def tensors(torch, rows, C, s11_stats, enc, dev, v_stats=None):
    """Признаки строк: x0, (v, s11) для o0, o4 (и o4b у пар), цели."""
    import k21_model as km

    def vis(imgs):
        out = []
        with torch.no_grad():
            for i in range(0, len(imgs), 256):
                x = torch.as_tensor(np.stack(imgs[i:i + 256])).to(dev
                                                                 ).float() / 255
                out.append(enc(x).cpu())
        v = torch.cat(out) if out else torch.zeros(0, 2 * VIS)
        if v_stats is not None:
            v = (v - v_stats[0]) / v_stats[1]
        return v

    def s11(s):
        v = km.state11(np.stack(s))
        return torch.as_tensor((v - s11_stats[0]) / s11_stats[1],
                               dtype=torch.float32)
    T = dict(n=len(rows), weight=torch.as_tensor(
        [r.get("weight", 1.0) for r in rows], dtype=torch.float32))
    T["v0"], T["s0"] = vis([r["img0"] for r in rows]), s11([r["s0"] for r in rows])
    T["v4"], T["s4"] = vis([r["img4"] for r in rows]), s11([r["s4"] for r in rows])
    B = np.stack([r["B"] for r in rows])
    T["Bf"] = torch.as_tensor(chunk_feat(B, C))
    T["b_arm"] = torch.as_tensor(norm_arm(B[:, 4:8], C), dtype=torch.float32)
    T["b_grip"] = torch.as_tensor(np.sign(B[:, 4:8, 6]), dtype=torch.float32)
    T["tt"] = torch.as_tensor([[r["t"] / 600.0] for r in rows],
                              dtype=torch.float32)
    T["x0"] = torch.cat([T["v0"], T["s0"], T["Bf"], T["tt"]], 1)
    Y = np.stack([r["Y"] for r in rows])
    T["ya"] = torch.as_tensor(norm_arm(Y, C), dtype=torch.float32)
    T["yg"] = torch.as_tensor((Y[..., 6] > 0).astype(np.float32))
    pr = [i for i, r in enumerate(rows) if r["kind"] == "pair"]
    T["pair_idx"] = torch.as_tensor(pr, dtype=torch.long)
    if pr:
        Yb = np.stack([rows[i]["Yb"] for i in pr])
        T["yb_a"] = torch.as_tensor(norm_arm(Yb, C), dtype=torch.float32)
        T["yb_g"] = torch.as_tensor((Yb[..., 6] > 0).astype(np.float32))
    return T


def xmid(torch, T, mode, idx=None):
    sel = slice(None) if idx is None else idx
    v = T["v4"] if mode[0] else T["v0"]
    s = T["s4"] if mode[1] else T["s0"]
    return torch.cat([T["x0"][sel], v[sel], s[sel]], 1)


def gen_alts(torch, gen, x0, b_arm, n_alt, b_grip=None):
    o = gen(x0).reshape(-1, n_alt, 28)
    r = o[..., :24].reshape(-1, n_alt, 4, 6)
    gl = o[..., 24:]
    return [tail_from(torch, r[:, k], gl[:, k], b_arm, b_grip)
            for k in range(n_alt)]


def fit_loop(torch, model, loss_fn, n, val_fn, epochs, seed):
    """Мини-батчи по индексам 0…n−1; лучшая эпоха по val_fn()."""
    if n == 0:
        raise SystemExit("нет строк для обучения")
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                            weight_decay=CFG["wd"])
    best, best_sd, hist = float("inf"), None, []
    for ep in range(epochs):
        perm = np.random.default_rng([39010 + seed, ep]).permutation(n)
        model.train()
        tot = 0.0
        for i in range(0, n, CFG["batch"]):
            idx = torch.as_tensor(perm[i:i + CFG["batch"]])
            loss = loss_fn(idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            v = float(val_fn())
        if not np.isfinite(v) or not np.isfinite(tot):
            raise SystemExit(f"эпоха {ep}: нечисловая потеря")
        hist.append(dict(epoch=ep, train=tot / max(n, 1), val=v))
        if v < best:
            best = v
            best_sd = {k: x.detach().clone() for k, x in
                       model.state_dict().items()}
    model.load_state_dict(best_sd)
    return dict(best_val=best, history=hist)


def baseline_error(r, C):
    """Ошибка B в нормированных координатах, симметричная для пары."""
    tail = r["B"][4:8]
    arm, grip = norm_arm(tail, C), tail[:, 6]
    d = tail_dist(arm, grip, r["Y"], C)
    if r["kind"] == "pair":
        d = (d + tail_dist(arm, grip, r["Yb"], C)) / 2
    return float(d)


def with_hard_weights(rows, C, cutoff):
    return [dict(r, weight=CFG["hard_w"] if baseline_error(r, C) >= cutoff else 1.0)
            for r in rows]


def fit_seed(torch, rows, C, seed, dev, n_alt_list=(2, 1), variant="base"):
    """-> (модели, отчёт). rows — строки ОДНОГО сида."""
    V = VARIANTS[variant]
    nearb = V["nearb"]
    ep_head = {k: V["head_epochs"] or CFG["epochs"][k]
               for k in ("gen", "sel", "flat")}
    B_ = build(torch)
    torch.manual_seed(39010 + seed)
    vals = rows[0]["_val"] if rows and "_val" in rows[0] else set()
    gen_tr = [r for r in rows if r["split"] == "gen" and r["case"] not in vals]
    gen_va = [r for r in rows if r["split"] == "gen" and r["case"] in vals]
    sel_tr = [r for r in rows if r["split"] == "sel" and r["case"] not in vals]
    sel_va = [r for r in rows if r["split"] == "sel" and r["case"] in vals]
    if not all((gen_tr, gen_va, sel_tr, sel_va)):
        raise SystemExit(f"s{seed}: нужны отдельные непустые train и val для gen и sel")
    import k21_model as km
    pre_tr, pre_va = expand_pairs(gen_tr), expand_pairs(gen_va)
    s_all = km.state11(np.stack([r[key] for r in pre_tr for key in ("s0", "s4")]))
    s_weights = np.repeat([r["weight"] for r in pre_tr], 2)
    s_mean = np.average(s_all, axis=0, weights=s_weights)
    s_std = np.sqrt(np.average((s_all - s_mean) ** 2, axis=0, weights=s_weights))
    st = (s_mean, np.where(s_std < 1e-6, 1.0, s_std))
    rep = dict(seed=seed, n=dict(gen_tr=len(gen_tr), gen_va=len(gen_va),
                                 sel_tr=len(sel_tr), sel_va=len(sel_va)))
    # 1) энкодер: B[:4] по o0 и Y[:4] по o4 (+ сёстры)
    enc = B_["Enc"]().to(dev)
    head = B_["mlp"](2 * VIS + 11, 96, 28).to(dev)

    def enc_batch(rs):
        imgs, s, ya, yg, w = [], [], [], [], []
        for r in rs:
            for img, ss, tgt in ((r["img0"], r["s0"], r["B"][:4]),
                                 (r["img4"], r["s4"], r["Y"])):
                imgs.append(img)
                s.append(ss)
                ya.append(norm_arm(tgt, C))
                yg.append(tgt[:, 6] > 0)
                w.append(r["weight"])
        x = torch.as_tensor(np.stack(imgs)).to(dev).float() / 255
        sv = torch.as_tensor((km.state11(np.stack(s)) - st[0]) / st[1],
                             dtype=torch.float32, device=dev)
        targets = [torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
                   for a in (ya, yg, w)]
        return x, sv, *targets

    class Pre(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc, self.head = enc, head

    pre = Pre()

    def enc_loss(rs):
        x, sv, ya, yg, weight = enc_batch(rs)
        o = pre.head(torch.cat([pre.enc(x), sv], 1))
        loss = row_loss(torch, torch.tanh(o[:, :24].reshape(-1, 4, 6)),
                        o[:, 24:], ya, yg)
        return weighted_mean(loss, weight)
    rep["enc"] = fit_loop(torch, pre, lambda idx: enc_loss(
        [pre_tr[i] for i in idx.tolist()]), len(pre_tr),
        lambda: enc_loss(pre_va), CFG["epochs"]["enc"], seed)
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    rep["params"] = dict(enc=n_params(enc))

    # нормировка визуальных признаков по train генератора (o0 и o4)
    raw = tensors(torch, pre_tr, C, st, enc, dev, None)
    vv = torch.cat([raw["v0"], raw["v4"]])
    vw = raw["weight"].repeat(2)[:, None]
    vmean = (vv * vw).sum(0) / vw.sum()
    vstd = (((vv - vmean).square() * vw).sum(0) / vw.sum()).sqrt()
    vst = (vmean, vstd.clamp_min(1e-6))

    def T_of(rs):
        return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in
                tensors(torch, rs, C, st, enc, dev, vst).items()}
    nominal_d = [baseline_error(r, C) for r in gen_tr if r["kind"] == "nom"]
    if not nominal_d:
        raise SystemExit(f"s{seed}: нет номинальных строк gen train")
    cutoff = float(np.quantile(nominal_d, 0.75))
    rep["hard_cutoff"] = cutoff
    gen_weighted = with_hard_weights(gen_tr, C, cutoff)
    Tg, Tgv = T_of(gen_weighted), T_of(gen_va)
    Ts, Tsv = T_of(expand_pairs(sel_tr)), T_of(expand_pairs(sel_va))
    models = dict(enc=enc, s11_stats=st, v_stats=vst, variant=variant)
    # покрытие покомпонентных поправок руки |Y − B[4:8]| (норм.), gen train
    corr = np.abs(np.stack([norm_arm(r["Y"], C) - norm_arm(r["B"][4:8], C)
                            for r in gen_tr if r["kind"] == "nom"]))
    rep["arm_correction"] = dict(
        **{f"share_le_{x}": float((corr <= x).mean())
           for x in (0.05, 0.1, 0.25, 0.5, 1.0)},
        p50=float(np.median(corr)), p90=float(np.quantile(corr, 0.9)),
        p99=float(np.quantile(corr, 0.99)), max=float(corr.max()))
    rep["variant"], rep["head_epochs"] = variant, ep_head

    def bg(T, idx=None):
        if not nearb:
            return None
        return T["b_grip"] if idx is None else T["b_grip"][idx]
    for n_alt in n_alt_list:
        # 2) генератор
        gen = B_["mlp"](132, 96, 28 * n_alt).to(dev)
        if nearb:
            init_near_b(torch, gen, f"k39-gen{n_alt}|{seed}")
        def gloss(T, idx):
            alts = gen_alts(torch, gen, T["x0"][idx], T["b_arm"][idx], n_alt,
                            bg(T, idx))
            ls = single_loss(torch, alts, (T["ya"][idx], T["yg"][idx]))
            if len(T["pair_idx"]):
                pos = {int(j): q for q, j in enumerate(T["pair_idx"].tolist())}
                m = [k for k, j in enumerate(idx.tolist()) if j in pos]
                if m:
                    jj = torch.as_tensor([pos[int(idx[k])] for k in m],
                                         device=dev)
                    mi = torch.as_tensor(m, device=dev)
                    sub = [(a[0][mi], a[1][mi]) for a in alts]
                    if n_alt == 2:
                        pl = pair_loss(torch, sub, (T["ya"][idx][mi],
                                                    T["yg"][idx][mi]),
                                       (T["yb_a"][jj], T["yb_g"][jj]))
                    else:
                        pl = (row_loss(torch, *sub[0], T["ya"][idx][mi],
                                       T["yg"][idx][mi]) +
                              row_loss(torch, *sub[0], T["yb_a"][jj],
                                       T["yb_g"][jj])) / 2
                    ls = ls.clone()
                    ls[mi] = pl
            return weighted_mean(ls, T["weight"][idx])
        rg = fit_loop(torch, gen, lambda idx: gloss(Tg, idx.to(dev)),
                      Tg["n"], lambda: gloss(Tgv, torch.arange(
                          Tgv["n"], device=dev)), ep_head["gen"], seed)
        for p in gen.parameters():
            p.requires_grad_(False)
        gen.eval()
        # 3) селектор: строки sel; сёстры — отдельные строки с весом 0.5
        sel = B_["mlp"](207, 32, 1 + n_alt).to(dev)
        with torch.no_grad():               # старт с равных вероятностей
            sel[2].weight.zero_()
            sel[2].bias.zero_()

        def cand_d(T, idx):
            with torch.no_grad():
                alts = gen_alts(torch, gen, T["x0"][idx], T["b_arm"][idx],
                                n_alt, bg(T, idx))
            ya, yg = T["ya"][idx], T["yg"][idx]
            candidates = [(T["b_arm"][idx], T["b_grip"][idx]), *alts]
            return torch.stack([action_distance(a, g, ya, yg)
                                for a, g in candidates], -1)

        def sloss(T, idx):
            d = cand_d(T, idx)
            tot = 0.0
            for mode in MODES:
                loss = regret_loss(torch, sel(xmid(torch, T, mode, idx)), d)
                tot = tot + weighted_mean(loss, T["weight"][idx])
            return tot / len(MODES)
        rs_ = fit_loop(torch, sel, lambda idx: sloss(Ts, idx.to(dev)),
                       Ts["n"], lambda: sloss(Tsv, torch.arange(
                           Tsv["n"], device=dev)), ep_head["sel"], seed)
        models[f"gen{n_alt}"], models[f"sel{n_alt}"] = gen, sel
        rep[f"gen{n_alt}"], rep[f"sel{n_alt}"] = rg, rs_
        rep["params"][f"tree{n_alt}"] = n_params(gen) + n_params(sel)
    # 4) flat: gen + sel, все режимы
    flat = B_["mlp"](207, 106, 28).to(dev)
    if nearb:
        init_near_b(torch, flat, f"k39-flat|{seed}")
    Tf = T_of(expand_pairs(gen_weighted + with_hard_weights(sel_tr, C, cutoff)))
    Tfv = T_of(expand_pairs(gen_va + sel_va))

    def floss(T, idx):
        tot = 0.0
        for mode in MODES:
            o = flat(xmid(torch, T, mode, idx))
            a, g = tail_from(torch, o[:, :24].reshape(-1, 4, 6), o[:, 24:],
                             T["b_arm"][idx], bg(T, idx))
            loss = row_loss(torch, a, g, T["ya"][idx], T["yg"][idx])
            tot = tot + weighted_mean(loss, T["weight"][idx])
        return tot / len(MODES)
    rep["flat"] = fit_loop(torch, flat, lambda idx: floss(Tf, idx.to(dev)),
                           Tf["n"], lambda: floss(Tfv, torch.arange(
                               Tfv["n"], device=dev)), ep_head["flat"],
                           seed)
    models["flat"] = flat
    rep["params"]["flat"] = n_params(flat)
    return models, rep


# --- офлайн-оценка --------------------------------------------------------------------------------
def evaluate(torch, models, rows, C, dev, n_alt=2):
    """Строки diag (nom/sis1/pair) -> метрики по строкам."""
    enc, st, vst = models["enc"], models["s11_stats"], models["v_stats"]
    gen, sel, flat = models[f"gen{n_alt}"], models[f"sel{n_alt}"], \
        models["flat"]
    T = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in
         tensors(torch, rows, C, st, enc, dev, vst).items()}
    bgr = T["b_grip"] if VARIANTS[models.get("variant", "base")]["nearb"] \
        else None
    with torch.no_grad():
        alts = gen_alts(torch, gen, T["x0"], T["b_arm"], n_alt, bgr)
        arms = [T["b_arm"]] + [a for a, _g in alts]
        grips = [T["b_grip"]] + [torch.where(g > 0, 1.0, -1.0)
                                 for _a, g in alts]

        ya, yg = T["ya"], T["yg"]
        ar, gr = zip(*(action_errors(a, g, ya, yg) for a, g in zip(arms, grips)))
        arm_err, grip_mis = torch.stack(ar, -1), torch.stack(gr, -1)
        out = dict(d=(arm_err + CFG["grip_w"] * grip_mis).cpu().numpy(),
                   arm_err=arm_err.cpu().numpy(), grip_mis=grip_mis.cpu().numpy())
        # хвосты кандидатов в кодеке float32 — та же арифметика, что в
        # k39_run.Heads.start (кандидат 0 — B[4:8] побитово)
        tails = [np.stack([r["B"][4:8] for r in rows]).astype(np.float32)]
        for a_, g_ in alts:
            c = np.zeros((len(rows), 4, 7), np.float32)
            c[..., :6] = (a_.cpu().numpy() * C["half"] + C["mid"]
                          ).astype(np.float32)
            c[..., 6] = np.where(g_.cpu().numpy() > 0, 1.0, -1.0)
            tails.append(c)
        out["tails"] = np.stack(tails, 1)
        for nm, mode in (("old", (0, 0)), ("fresh", (1, 1)),
                         ("rgb_only", (1, 0)), ("state_only", (0, 1))):
            ch = sel(xmid(torch, T, mode)).argmax(-1)
            out[f"choice_{nm}"] = ch.cpu().numpy()
            o = flat(xmid(torch, T, mode))
            a, g = tail_from(torch, o[:, :24].reshape(-1, 4, 6), o[:, 24:],
                             T["b_arm"], bgr)
            af, gf = action_errors(a, g, ya, yg)
            out[f"d_flat_{nm}"] = (af + CFG["grip_w"] * gf).cpu().numpy()
            out[f"arm_flat_{nm}"] = af.cpu().numpy()
            out[f"grip_flat_{nm}"] = gf.cpu().numpy()
    return out


def summarize_eval(rows, ev):
    idx_all = np.arange(len(rows))
    weights = np.asarray([r.get("weight", 1.0) for r in rows])
    rep = {}

    def block(ix):
        if len(ix) == 0:
            return None
        w = weights[ix]
        d = ev["d"][ix]
        r = dict(n=int(len(ix)), weight_sum=float(w.sum()),
                 d_B=float(np.average(d[:, 0], weights=w)),
                 d_oracle=float(np.average(d.min(1), weights=w)))
        for nm in ("old", "fresh", "rgb_only", "state_only"):
            ch = ev[f"choice_{nm}"][ix]
            r[f"d_tree_{nm}"] = float(np.average(ev["d"][ix, ch], weights=w))
            r[f"share_alt_{nm}"] = float(np.average(ch != 0, weights=w))
            r[f"d_flat_{nm}"] = float(np.average(ev[f"d_flat_{nm}"][ix], weights=w))
            r[f"arm_tree_{nm}"] = float(np.average(ev["arm_err"][ix, ch], weights=w))
            r[f"grip_tree_{nm}"] = float(np.average(ev["grip_mis"][ix, ch], weights=w))
            r[f"arm_flat_{nm}"] = float(np.average(ev[f"arm_flat_{nm}"][ix], weights=w))
            r[f"grip_flat_{nm}"] = float(np.average(ev[f"grip_flat_{nm}"][ix], weights=w))
        r["arm_err"] = np.average(ev["arm_err"][ix], axis=0, weights=w).round(5).tolist()
        r["grip_mis"] = np.average(ev["grip_mis"][ix], axis=0, weights=w).round(4).tolist()
        r["oracle_share"] = (np.bincount(d.argmin(1), weights=w,
                                         minlength=d.shape[1]) / w.sum()).tolist()
        r["fresh_ne_old"] = float(np.average(ev["choice_fresh"][ix] !=
                                             ev["choice_old"][ix], weights=w))
        return r
    rep["all"] = block(idx_all)
    for key, f in (("task", "task"), ("seed", "seed"), ("kind", "kind")):
        rep[f"by_{key}"] = {str(v): block(np.array(
            [i for i in idx_all if rows[i][f] == v]))
            for v in sorted({r[f] for r in rows}, key=str)}
    return rep


def nominal_changes(ev, nom):
    """Штатные окна: разница d и ФАКТИЧЕСКИ изменённые команды (хвост 4×7
    float32 ≠ B[4:8] / fresh ≠ old), а также смена индекса ветви."""
    ix = np.flatnonzero(nom)
    tl = ev["tails"][ix]
    rr = np.arange(len(ix))
    cf, co = ev["choice_fresh"][ix], ev["choice_old"][ix]
    tf, to = tl[rr, cf], tl[rr, co]
    b = tl[:, 0]
    diff = lambda x, y: (x != y).reshape(len(ix), -1).any(1)  # noqa: E731
    return dict(
        n=int(len(ix)),
        d_fresh_minus_old=float(ev["d"][ix, cf].mean() -
                                ev["d"][ix, co].mean()),
        changed_fresh=int(diff(tf, b).sum()),
        changed_old=int(diff(to, b).sum()),
        fresh_ne_old=int(diff(tf, to).sum()),
        index_ne0_fresh=int((cf != 0).sum()),
        index_ne0_old=int((co != 0).sum()),
        index_fresh_ne_old=int((cf != co).sum()))


def expand_pairs(rows):
    """Пара -> две строки с общим pair_id, каждая с половиной веса пары."""
    out = []
    for i, r in enumerate(rows):
        if r["kind"] != "pair":
            out.append(dict(r, weight=r.get("weight", 1.0)))
            continue
        half = r.get("weight", 1.0) / 2
        out.append(dict(r, kind="sisA", pair_id=i, weight=half))
        out.append(dict(r, kind="sisB", pair_id=i, weight=half,
                        img4=r["img4b"], s4=r["s4b"], Y=r["Yb"]))
    return out


def sister_pref(rows, ev):
    """Разные предпочтения и выигрыш oracle над одним хвостом на пару."""
    by = {}
    for i, r in enumerate(rows):
        if r["kind"] in ("sisA", "sisB"):
            by.setdefault(r["pair_id"], []).append(ev["d"][i])
    pairs = [v for v in by.values() if len(v) == 2]
    if not pairs:
        return dict(n=0, share=None, oracle_gain_mean=None)
    d = np.asarray(pairs)
    idx = np.argmin(d, axis=-1)
    gain = d.mean(1).min(-1) - d.min(-1).mean(1)
    return dict(n=len(pairs), share=float((idx[:, 0] != idx[:, 1]).mean()),
                oracle_gain_mean=float(gain.mean()))


def pair_magnitudes(rows, C):
    """Величина различий Y+ / Y− в парах."""
    ds, mx = [], []
    for r in rows:
        if r["kind"] != "pair":
            continue
        ds.append(float(tail_dist(norm_arm(r["Y"], C), r["Y"][:, 6],
                                  r["Yb"], C)))
        mx.append(float(np.abs(norm_arm(r["Y"], C) - norm_arm(r["Yb"], C)
                               ).max()))
    if not ds:
        return dict(n=0)
    d, m = np.array(ds), np.array(mx)
    return dict(n=len(d), d_mean=float(d.mean()), d_median=float(np.median(d)),
                d_p90=float(np.quantile(d, 0.9)), maxabs_median=float(
                     np.median(m)), maxabs_p90=float(np.quantile(m, 0.9)),
                 maxabs_max=float(m.max()),
                share_d_gt_0_01=float((d > 0.01).mean()),
                share_d_gt_0_05=float((d > 0.05).mean()))


# --- команды ---------------------------------------------------------------------------------------
def model_dir(run, s, variant="base"):
    if variant == "base":
        return os.path.join("data/k39", run, "models", f"s{s}")
    return os.path.join("data/k39", run, "models", variant, f"s{s}")


def fit_name(s, variant="base"):
    return f"fit_s{s}.json" if variant == "base" else \
        f"fit_{variant}_s{s}.json"


def _setup(run):
    import k36_harness as kh
    import k36_protocol as pc
    import k39_cache as kc
    plan = kc.load_data_plan(run)
    k36 = pc.load_prepare()
    if k36["sha1"] != plan["k36_prepare_sha1"]:
        raise SystemExit("prepare K-36 ≠ data_plan")
    C = kh.consts(k36)
    return plan, C, kc


def _rows(torch, run, plan, kc, seed=None):
    import k36_protocol as pc

    if seed is not None:                    # только шарды этого сида
        eids = {e["eid"] for e in plan["episodes"] if e["seed"] == seed}
        plan = dict(plan, episodes=[e for e in plan["episodes"]
                                    if e["eid"] in eids],
                    sisters=[x for x in plan["sisters"] if x["eid"] in eids])
    lp = os.path.join(kc.rep_dir(run), "ledger.json")
    if not os.path.exists(lp):
        raise SystemExit(f"нет журнала {lp}")
    ledger = pc.load_json(lp)
    if ledger.get("caps") != plan["caps"]:
        raise SystemExit("лимиты журнала ≠ data_plan")
    expected = {kc.job_name("cache", k, e)
                for k, e in kc.jobs_of(plan, kc.TRAIN_SPLITS, True)}
    done = ledger["done"]
    pending_id = (ledger["pending"] or {}).get("id")
    missing = sorted(expected - done.keys())
    if missing or pending_id in expected:
        raise SystemExit(f"кеш ещё не принят журналом: "
                         f"{len(missing)} незакрытых заданий, "
                         f"первое {missing[0] if missing else pending_id}")

    def accepted(path, job):
        meta = kc.shard_ok(path, job, plan["sha256"])
        if meta and done[job] == dict(scope="cache", actual=meta["phys"]):
            return meta
        return False

    rows, cnt = load_rows(torch, plan, lambda job: kc.shard_path(run, job),
                          accepted)
    vs = val_scenes(plan)
    for r in rows:
        r["_val"] = vs
    if seed is not None:
        rows = [r for r in rows if r["seed"] == seed]
    return rows, cnt


def fit_cmd(a):
    import torch
    import k36_protocol as pc
    plan, C, kc = _setup(a.run_id)
    rows, cnt = _rows(torch, a.run_id, plan, kc, a.seed)
    t0 = time.time()
    models, rep = fit_seed(torch, rows, C, a.seed, torch.device(a.device),
                           variant=a.variant)
    md = model_dir(a.run_id, a.seed, a.variant)
    os.makedirs(md, exist_ok=True)
    path = os.path.join(md, "heads.pt")
    stamp = dict(run_id=a.run_id, seed=a.seed, model_code=kc.sha256(__file__),
                 plan=plan["sha256"], cfg=CFG, variant=a.variant)
    ck = {k: (v.state_dict() if hasattr(v, "state_dict") else v)
          for k, v in models.items()}
    torch.save(dict(ck, fit=stamp), path + ".tmp")
    os.replace(path + ".tmp", path)
    eh = hashlib.sha256()
    for k_, v_ in sorted(models["enc"].state_dict().items()):
        eh.update(k_.encode())
        eh.update(v_.detach().cpu().numpy().tobytes())
    rep.update(stamp, counts=cnt, seconds=time.time() - t0,
               sha256=kc.sha256(path), enc_sha256=eh.hexdigest())
    pc.save_json(os.path.join(kc.rep_dir(a.run_id),
                              fit_name(a.seed, a.variant)), rep)
    print(f"  fit s{a.seed}: строк gen {rep['n']['gen_tr']}+{rep['n']['gen_va']}"
          f", sel {rep['n']['sel_tr']}+{rep['n']['sel_va']}; пары "
          f"{cnt['pair']}, одиночные сёстры {cnt['sis1']}, отказ пар "
          f"{cnt['pair_reject']}; параметры {rep['params']}; "
          f"{rep['seconds']:.0f} с; вариант {a.variant}, лучшие эпохи "
          f"{ {k: min(rep[k]['history'], key=lambda x: x['val'])['epoch'] for k in ('gen2', 'sel2', 'flat')} }; "
          f"поправки руки ≤0.25: {rep['arm_correction']['share_le_0.25']:.3f}")
    return 0


def load_models(torch, run, s, dev, plan_sha=None, weights_sha=None,
                variant="base"):
    import k39_cache as kc

    B_ = build(torch)
    path = os.path.join(model_dir(run, s, variant), "heads.pt")
    if not os.path.isfile(path):
        raise SystemExit(f"нет весов s{s}: {path}")
    with open(path, "rb") as fh:
        body = fh.read()
    if weights_sha is not None and hashlib.sha256(body).hexdigest() != weights_sha:
        raise SystemExit(f"s{s}: heads.pt ≠ отпечатку весов")
    ck = torch.load(io.BytesIO(body), map_location="cpu", weights_only=False)
    want = dict(run_id=run, seed=s, model_code=kc.sha256(__file__), cfg=CFG,
                variant=variant)
    if plan_sha is not None:
        want["plan"] = plan_sha
    stamp = ck.get("fit", {})
    if any(stamp.get(k) != v for k, v in want.items()):
        raise SystemExit(f"s{s}: происхождение весов ≠ текущему fit")
    enc = B_["Enc"]().to(dev)
    enc.load_state_dict(ck["enc"])
    m = dict(enc=enc.eval(), s11_stats=ck["s11_stats"],
             v_stats=ck["v_stats"], variant=variant)
    for k, (i, h, o) in dict(gen2=(132, 96, 56), sel2=(207, 32, 3),
                             gen1=(132, 96, 28), sel1=(207, 32, 2),
                             flat=(207, 106, 28)).items():
        if k not in ck:
            raise SystemExit(f"s{s}: нет {k} в heads.pt")
        net = B_["mlp"](i, h, o).to(dev)
        net.load_state_dict(ck[k])
        m[k] = net.eval()
    return m


def offline_cmd(a):
    import torch
    import k36_protocol as pc
    plan, C, kc = _setup(a.run_id)
    dev = torch.device(a.device)
    fits = {}
    for s in SEEDS:
        path = os.path.join(kc.rep_dir(a.run_id), fit_name(s, a.variant))
        if not os.path.isfile(path):
            raise SystemExit(f"нет отчёта fit для s{s}")
        fit = pc.load_json(path)
        if fit.get("plan") != plan["sha256"] or fit.get("seed") != s:
            raise SystemExit(f"s{s}: отчёт fit ≠ data_plan/сиду")
        fits[s] = fit["sha256"]
    rows_all, cnt = _rows(torch, a.run_id, plan, kc)
    diag_rows = [r for r in rows_all if r["split"] == "diag"]
    rep = dict(plan=plan["sha256"], counts=cnt,
               pairs_all=pair_magnitudes(rows_all, C),
               pairs_diag=pair_magnitudes(diag_rows, C), weights_sha256=fits)
    for s in SEEDS:
        m = load_models(torch, a.run_id, s, dev, plan["sha256"], fits[s],
                        a.variant)
        diag = expand_pairs([r for r in diag_rows if r["seed"] == s])
        if not diag:
            raise SystemExit(f"s{s}: нет диагностических строк")
        for n_alt in (2, 1):
            ev = evaluate(torch, m, diag, C, dev, n_alt)
            sm = summarize_eval(diag, ev)
            sm["sister_pref_differs"] = sister_pref(diag, ev)
            nom = np.array([r["kind"] == "nom" for r in diag])
            sm["nominal"] = nominal_changes(ev, nom)
            rep[f"s{s}_alt{n_alt}"] = sm
    rep["variant"] = a.variant
    name = "offline.json" if a.variant == "base" else \
        f"offline_{a.variant}.json"
    pc.save_json(os.path.join(kc.rep_dir(a.run_id), name), rep)
    pm = rep["pairs_all"]
    print(f"  пары Y+/Y−: {pm}")
    for k, v in rep.items():
        if k.startswith("s") and "_alt" in k and v["all"]:
            x = v["all"]
            print(f"  {k}: d_B {x['d_B']:.4f}, лучший {x['d_oracle']:.4f}, "
                  f"дерево old {x['d_tree_old']:.4f} / fresh "
                  f"{x['d_tree_fresh']:.4f}, flat fresh {x['d_flat_fresh']:.4f}"
                  f"; доля альтернатив fresh {x['share_alt_fresh']:.4f}, "
                  f"fresh≠old {x['fresh_ne_old']:.4f}")
            nm_ = v["nominal"]
            print(f"      штатные окна n {nm_['n']}: d_fresh − d_old "
                  f"{nm_['d_fresh_minus_old']:+.5f}; фактически изменённых "
                  f"хвостов fresh {nm_['changed_fresh']}, old "
                  f"{nm_['changed_old']}, fresh≠old {nm_['fresh_ne_old']} "
                  f"(индекс ≠ 0: {nm_['index_ne0_fresh']}/"
                  f"{nm_['index_ne0_old']}); выигрыш раздельного "
                  f"выбора в парах "
                  f"{v['sister_pref_differs'].get('oracle_gain_mean')}")
    return 0


# --- самопроверка ---------------------------------------------------------------------------------
def selftest():
    import tempfile
    import torch
    torch.manual_seed(0)
    C = dict(mid=np.zeros(6), half=np.ones(6))
    # 1) парная потеря инвариантна к перестановке меток
    a1 = (torch.randn(5, 4, 6), torch.randn(5, 4))
    a2 = (torch.randn(5, 4, 6), torch.randn(5, 4))
    Y1 = (torch.randn(5, 4, 6), (torch.rand(5, 4) > 0.5).float())
    Y2 = (torch.randn(5, 4, 6), (torch.rand(5, 4) > 0.5).float())
    assert torch.allclose(pair_loss(torch, (a1, a2), Y1, Y2),
                          pair_loss(torch, (a1, a2), Y2, Y1))
    assert torch.allclose(pair_loss(torch, (a1, a2), Y1, Y2),
                          pair_loss(torch, (a2, a1), Y1, Y2))
    # 2) сожаление: две одинаково хорошие ветви — равнодушие между ними
    d = torch.tensor([[0.3, 0.1, 0.1]])
    l1 = float(regret_loss(torch, torch.tensor([[-10.0, 5.0, -5.0]]), d))
    l2 = float(regret_loss(torch, torch.tensor([[-10.0, -5.0, 5.0]]), d))
    assert l1 < 1e-4 and abs(l1 - l2) < 1e-9
    assert float(regret_loss(torch, torch.tensor([[5.0, -5.0, -5.0]]), d)) \
        > 0.19
    # 3) параметры по плану
    B_ = build(torch)
    assert n_params(B_["mlp"](132, 96, 56)) + n_params(B_["mlp"](207, 32, 3)) \
        == 24955
    assert n_params(B_["mlp"](207, 106, 28)) == 25044
    # 4) загрузчик: пары по (eid, t), маска has_y, отказ при чужом B / o0
    with tempfile.TemporaryDirectory() as td:
        def sp(job):
            return os.path.join(td, job.replace("/", "__") + ".npz")
        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 255, (2, 2, 224, 224, 3), dtype=np.uint8)
        B = rng.normal(size=(2, 8, 7)).astype(np.float32)
        B[..., 6] = 1.0
        Y = rng.normal(size=(2, 8, 7)).astype(np.float32)
        np.savez(sp("cache/e1"), t=np.array([0, 8]), rgb0=rgb, rgb4=rgb,
                 s0=np.zeros((2, 8)), s4=np.zeros((2, 8)), B=B, Y=Y,
                 obs0_hash=np.array(["h0", "h8"]))

        def sis(job, sign, has_y, t=8, b=None, h="h8"):
            np.savez(sp(job), sign=np.array(sign), t=np.array(t),
                     B=B[1] if b is None else b, obs0_hash=np.array(h),
                     has_y=np.array(has_y), has_o4=np.array(has_y),
                     rgb4=rgb[1], s4=np.zeros(8),
                     Y=(Y[1] + sign) if has_y else np.zeros((8, 7),
                                                            np.float32))
        plan = dict(episodes=[dict(eid="e1", seed=0, task=0, case_id="c1",
                                    split="gen", windows=[0, 8])],
                    sisters=[dict(eid="e1", t=8, sign=1, missing=False,
                                  job="sis/e1/8/+1"),
                             dict(eid="e1", t=8, sign=-1, missing=False,
                                  job="sis/e1/8/-1")])
        ok = lambda p, job: os.path.exists(p)  # noqa: E731
        sis("cache/sis/e1/8/+1", 1, True)
        sis("cache/sis/e1/8/-1", -1, True)
        rows, cnt = load_rows(torch, plan, sp, ok)
        assert cnt == dict(nom=2, pair=1, sis1=0, dead=0, pair_reject=0)
        pr = [r for r in rows if r["kind"] == "pair"][0]
        assert np.allclose(pr["Y"], Y[1][:4] + 1) and \
            np.allclose(pr["Yb"], Y[1][:4] - 1)
        assert pr["img4"].shape == (2, 3, IMG, IMG)
        sis("cache/sis/e1/8/-1", -1, False)            # обрыв: has_y False
        rows, cnt = load_rows(torch, plan, sp, ok)
        assert cnt["sis1"] == 1 and cnt["pair"] == 0 and cnt["dead"] == 1
        sis("cache/sis/e1/8/-1", -1, True, b=B[0])     # чужой B
        assert load_rows(torch, plan, sp, ok)[1]["pair_reject"] == 1
        sis("cache/sis/e1/8/-1", -1, True, h="hX")     # чужой o0
        assert load_rows(torch, plan, sp, ok)[1]["pair_reject"] == 1
        sis("cache/sis/e1/8/-1", 1, True)             # повтор знака +
        assert load_rows(torch, plan, sp, ok)[1]["pair_reject"] == 1
    # 5) Проверка масштаба: B и обе метки исходно в координатах кодека.
    scaled = dict(mid=np.linspace(0.25, 0.75, 6), half=np.full(6, 0.5))
    Bc = np.tile(np.r_[scaled["mid"], 1.0], (8, 1))
    Yc = Bc[:4].copy()
    Yc[:, :6] += 0.1
    assert abs(baseline_error(dict(B=Bc, Y=Yc, kind="nom"), scaled) - 0.2) < 1e-9
    # 6) Только сёстры дают противоположные хвосты, сцены не пересекаются.
    rows = []
    for i in range(90):
        img0 = np.full((2, 3, IMG, IMG), 128, np.uint8)
        Bc = np.zeros((8, 7), np.float32)
        Bc[:, 6] = 1.0
        Yc = Bc[:4].copy()
        Yp, Yn = Yc.copy(), Yc.copy()
        Yp[:, 0], Yn[:, 0] = 0.8, -0.8
        base = dict(seed=0, task=(0, 6, 8)[(i // 3) % 3], case=f"c{i}",
                     split=("gen", "sel", "diag")[i % 3], eid=f"e{i}", t=0, img0=img0,
                    s0=np.zeros(8), B=Bc, h0="x")
        rows.append(dict(base, kind="nom", img4=img0, s4=base["s0"], Y=Yc))
        rows.append(dict(base, kind="pair", img4=np.full_like(img0, 188),
                         s4=base["s0"], Y=Yp, img4b=np.full_like(img0, 68),
                         s4b=base["s0"], Yb=Yn))
    for r in rows:
        r["_val"] = {f"c{j}" for j in range(12) if j % 3 != 2}
    CFG["epochs"].update(enc=6, gen=80, sel=80, flat=80)
    models, rep = fit_seed(torch, rows, C, 0, torch.device("cpu"),
                           n_alt_list=(2,))
    diag = expand_pairs([r for r in rows if r["split"] == "diag"])
    ev = evaluate(torch, models, diag, C, torch.device("cpu"), 2)
    sm = summarize_eval(diag, ev)["all"]
    # старое наблюдение одинаково у обеих «сторон» -> один выбор на всех
    assert len(set(ev["choice_old"].tolist())) == 1
    # свежее различает стороны и улучшает расстояние
    assert len(set(ev["choice_fresh"].tolist())) >= 2, sm
    assert sm["d_tree_fresh"] < sm["d_tree_old"], sm
    assert sm["d_flat_fresh"] < sm["d_flat_old"], sm
    assert sister_pref(diag, ev)["oracle_gain_mean"] > 0
    # вариант nearb проходит тот же путь обучения и оценки
    mnb, rnb = fit_seed(torch, rows, C, 0, torch.device("cpu"),
                        n_alt_list=(2,), variant="nearb")
    evaluate(torch, mnb, diag, C, torch.device("cpu"), 2)
    assert rnb["variant"] == "nearb" and rnb["head_epochs"]["gen"] == 100
    assert 0 <= rnb["arm_correction"]["share_le_0.25"] <= 1
    # при o4 = o0 свежий и старый режимы совпадают побитово
    same = [dict(r, img4=r["img0"], s4=r["s0"]) for r in diag]
    ev2 = evaluate(torch, models, same, C, torch.device("cpu"), 2)
    assert np.array_equal(ev2["choice_old"], ev2["choice_fresh"])
    # nearb: старт около B — рука ≈ B, знак схвата = знак B
    gtmp = B_["mlp"](132, 96, 56)
    init_near_b(torch, gtmp, "t")
    x0 = torch.randn(7, 132)
    b_arm = torch.rand(7, 4, 6) * 1.6 - 0.8
    b_g = torch.where(torch.rand(7, 4) > 0.5, 1.0, -1.0)
    with torch.no_grad():
        alts_nb = gen_alts(torch, gtmp, x0, b_arm, 2, b_g)
    for arm, gl in alts_nb:
        assert float((arm - b_arm).abs().max()) < 0.05
        assert torch.equal(torch.where(gl > 0, 1.0, -1.0), b_g)
    al = gen_alts(torch, gtmp, x0, b_arm, 2, b_g)
    assert not torch.equal(al[0][0], al[1][0])          # ветви различны
    tl = np.zeros((3, 3, 4, 7), np.float32)
    tl[:, 2] = 1.0                       # C2 отличается от B, C1 = B побитово
    evn = dict(tails=tl, choice_fresh=np.array([1, 2, 0]),
               choice_old=np.array([0, 0, 0]), d=np.zeros((3, 3)))
    nc = nominal_changes(evn, np.array([True, True, True]))
    assert nc["changed_fresh"] == 1 and nc["index_ne0_fresh"] == 2 and \
        nc["fresh_ne_old"] == 1 and nc["index_fresh_ne_old"] == 2
    ex = expand_pairs([dict(kind="pair", Y=1, Yb=2, img4="a", img4b="b",
                            s4=0, s4b=1), dict(kind="nom")])
    assert [r["kind"] for r in ex] == ["sisA", "sisB", "nom"] and \
        ex[1]["Y"] == 2 and ex[1]["img4"] == "b"
    assert [r["weight"] for r in ex] == [0.5, 0.5, 1.0]
    sp_ = sister_pref(ex, dict(d=np.array([[0, 1, 2], [2, 1, 0], [0, 0, 0]])))
    assert sp_ == dict(n=1, share=1.0, oracle_gain_mean=1.0)
    print("самопроверка k39_model пройдена: парная потеря инвариантна к "
          "перестановке; сожаление равнодушно к равным ветвям; параметры "
          "24 955 / 25 044; загрузчик: пары по (eid, t), маска has_y, отказ "
          "при чужом B или o0; игрушка: старое наблюдение — один выбор, "
          f"свежее — разные хвосты (d {sm['d_tree_old']:.3f} -> "
          f"{sm['d_tree_fresh']:.3f}); o4 = o0 -> одинаковые решения")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-39 модели")
    ap.add_argument("cmd", choices=("fit", "offline"))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--variant", choices=tuple(VARIANTS), default="base")
    a = ap.parse_args()
    if a.cmd == "fit" and a.seed is None:
        print("  ОТКАЗ: нужен --seed")
        return 3
    try:
        return dict(fit=fit_cmd, offline=offline_cmd)[a.cmd](a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
