#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16 (шаг 3): обучение двух предсказателей цели и четырёх голов.

ПОРЯДОК:
  1. разметка (k16_prepare) и кэш h18 проверяются: отпечатки разметки,
     ПОЛНАЯ проверка кэша h18 (verify_h18=True), строки разметки = строки
     кэша;
  2. goal_now и goal_hist обучаются на половине A, эпоха — по ошибке цели
     на val_sel; затем замораживаются;
  3. их СОБСТВЕННЫЕ прогнозы на B и val_sel — вход корректирующих голов
     (не истинная будущая цель);
  4. на половине B обучаются direct (история + вспомогательная цель),
     goal_flat и goal_band (цель goal_hist), goal_band_now (цель
     goal_now); одна и та же основная ошибка — MSE поправки r [8, 6] в
     нормированных по каналу единицах; эпоха — по ней на val_sel.
Один сид, Adam, не больше 10 эпох, одна заранее заданная конфигурация.

ВЫХОД: data/k16/models.pt (веса, ширины, нормировки, предел нормы —
95-й процентиль ||r*|| в координатах исполняемых команд на B, α = 0.5,
отпечатки входов и кода) и reports/k16/train.json (ошибки цели и поправки
на val_sel, базовый прогноз цели по задаче и возрасту, нормы частей,
доля срабатываний предела).
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

SEED = 0
EPOCHS = 10
BATCH = 256
LR = 1e-3
WD = 1e-4
ALPHA = 0.5
CLIP_Q = 95.0
OUT = "data/k16/models.pt"
REPORT = "reports/k16/train.json"


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def split_arrays(L, part, idx, h18):
    """Входы и разметка для строк idx части part."""
    g = lambda k: np.asarray(L[f"{part}__{k}"])[idx]   # noqa: E731
    hist = g("hist")                                     # [n, 4, 8]
    mask = g("hist_mask").astype(bool)
    return dict(
        h18=np.asarray(h18[idx], np.float16), a0=None,
        state=hist[:, 0], hist=hist[:, 1:], hist_mask=mask[:, 1:],
        age=g("age"), dpos=g("dpos"), drot6d=g("drot6d"),
        grip=(g("event_grip") > 0).astype(np.float32),
        etype=g("event_type").astype(np.int64), rstar=g("rstar"),
        task=g("task"), epi=g("epi"))


def goal_loss(torch, out, y, dpos_sigma):
    F = torch.nn.functional
    l_pos = ((out[:, :3] - y["dpos"] / dpos_sigma) ** 2).mean()
    l_rot = ((out[:, 3:9] - y["drot6d"]) ** 2).mean()
    l_g = F.binary_cross_entropy_with_logits(out[:, 9], y["grip"])
    l_t = F.cross_entropy(out[:, 10:13], y["etype"])
    return l_pos + l_rot + l_g + l_t


def to_t(torch, d, idx, dev, keys):
    return {k: torch.as_tensor(np.asarray(d[k][idx]), device=dev)
            for k in keys}


def new_model(torch, ctor, *args, seed=SEED, **kw):
    """Сид — ДО конструктора: инициализация воспроизводима по сиду."""
    torch.manual_seed(seed)
    return ctor(*args, **kw)


def fit(torch, model, train, val, step_fn, dev, epochs=EPOCHS, seed=SEED):
    """Обучение с выбором эпохи по ошибке на val (взвешенной по строкам);
    нечисловая ошибка — технический отказ."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    n = len(train["age"])
    best, best_state, hist = None, None, []
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        for s in range(0, n, BATCH):
            loss = step_fn(model, train, perm[s:s + BATCH])
            if not bool(torch.isfinite(loss)):
                raise SystemExit(f"нечисловая ошибка обучения на эпохе {ep}")
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        nv = len(val["age"])
        with torch.no_grad():
            tot = 0.0
            for s in range(0, nv, 1024):
                i = np.arange(s, min(s + 1024, nv))
                tot += float(step_fn(model, val, i)) * len(i)
            vl = tot / nv
        if not np.isfinite(vl):
            raise SystemExit(f"нечисловая ошибка проверки на эпохе {ep}")
        hist.append(vl)
        if best is None or vl < best[1]:
            best = (ep, vl)
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best, hist


def train_all(torch, data, dev, epochs=EPOCHS, log=print):
    """data: A, B, V — словари массивов (+ sf_hist, sf_now). Возвращает
    модели и отчёт."""
    C = km.build(torch, d_model=int(data["A"]["h18"].shape[2]),
                 n_pos=int(data["A"]["h18"].shape[1]))
    rep = {}
    A, B, V = data["A"], data["B"], data["V"]
    dpos_sigma = torch.tensor(np.maximum(A["dpos"].std(0), 1e-4),
                              dtype=torch.float32, device=dev)
    r_sigma = torch.tensor(np.maximum(B["rstar"].reshape(-1, 6).std(0),
                                      1e-5), dtype=torch.float32, device=dev)
    keys_g = ("h18", "a0", "dpos", "drot6d", "grip", "etype")

    def goal_step(sfk):
        def f(model, d, idx):
            b = to_t(torch, d, idx, dev, keys_g + (sfk,))
            out = model(b["h18"], b["a0"].float(), b[sfk].float())
            return goal_loss(torch, out, {k: (b[k].float() if k != "etype"
                                              else b[k]) for k in keys_g},
                             dpos_sigma)
        return f
    goals = {}
    for name, sfk in (("goal_now", "sf_now"), ("goal_hist", "sf_hist")):
        m = new_model(torch, C["GoalNet"]).to(dev)
        best, h = fit(torch, m, A, V, goal_step(sfk), dev, epochs)
        for p in m.parameters():
            p.requires_grad_(False)
        goals[name] = (m, sfk)
        rep[name] = dict(best_epoch=best[0], val_loss=best[1], history=h,
                         params=km.n_params(m))
        log(f"  {name}: эпоха {best[0]}, ошибка цели на val_sel "
            f"{best[1]:.4f}")
    # СОБСТВЕННЫЕ ПРОГНОЗЫ замороженных предсказателей — вход голов
    for name, (m, sfk) in goals.items():
        for D_ in (B, V):
            outs, feats = [], []
            with torch.no_grad():
                for s in range(0, len(D_["age"]), 1024):
                    i = np.arange(s, min(s + 1024, len(D_["age"])))
                    b = to_t(torch, D_, i, dev, ("h18", "a0", sfk))
                    hid = m.hidden(b["h18"], b["a0"].float(),
                                   b[sfk].float())
                    feats.append(hid.cpu().numpy())
                    outs.append(m.body[-1](hid).cpu().numpy())
            D_[f"pred_{name}"] = np.concatenate(outs)
            D_[f"feat_{name}"] = np.concatenate(feats)
    target, widths = km.matched_widths(torch, C)
    specs = dict(
        direct=("Direct", widths["Direct"], "feat_goal_hist", "sf_hist"),
        goal_flat=("GoalFlat", widths["GoalFlat"], "pred_goal_hist",
                   "sf_hist"),
        goal_band=("GoalBand", widths["GoalBand"], "pred_goal_hist",
                   "sf_hist"),
        goal_band_now=("GoalBand", widths["GoalBand"], "pred_goal_now",
                       "sf_now"))
    heads = {}
    for name, (cls, hid, gk, sfk) in specs.items():
        m = new_model(torch, C[cls], hid).to(dev)

        def step(model, d, idx, gk=gk, sfk=sfk, name=name):
            ks = ("h18", "a0", "rstar", sfk, gk) + (
                keys_g[2:] if name == "direct" else ())
            b = to_t(torch, d, idx, dev, ks)
            r, aux = model(b["h18"], b["a0"].float(), b[sfk].float(),
                           b[gk].float())
            loss = (((r - b["rstar"].float() / r_sigma) ** 2).mean())
            if name == "direct" and model.training:
                loss = loss + km.AUX * goal_loss(
                    torch, aux, {k: (b[k].float() if k != "etype" else b[k])
                                 for k in keys_g[2:]}, dpos_sigma)
            return loss
        best, h = fit(torch, m, B, V, step, dev, epochs)
        heads[name] = (m, gk, sfk)
        rep[name] = dict(best_epoch=best[0], val_r_mse_norm=best[1],
                         history=h, params=km.n_params(m), hidden=hid,
                         goal_input=gk, state_features=sfk)
        log(f"  {name}: эпоха {best[0]}, MSE поправки (норм.) на val_sel "
            f"{best[1]:.4f}, параметров {km.n_params(m)}")
    rep["param_target_direct"] = target
    return goals, heads, rep, dict(dpos_sigma=dpos_sigma.cpu().numpy(),
                                   r_sigma=r_sigma.cpu().numpy())


def evaluate(torch, goals, heads, norms, V, A, scale, clip, dev):
    """Ошибки на val_sel: цель (м, градусы, точность схвата и типа),
    базовый прогноз цели по задаче и возрасту, поправка (R²), нормы частей
    goal_band, доля срабатываний предела."""
    import k16_prepare as kp
    out = {}
    ds = norms["dpos_sigma"]
    for name in goals:
        g = V[f"pred_{name}"]
        pe = np.linalg.norm(g[:, :3] * ds - V["dpos"], axis=1)
        Rp = kp.sixd_to_mat(g[:, 3:9])
        Rt = kp.sixd_to_mat(V["drot6d"])
        ang = np.degrees(kp.rot_angle(Rp, Rt))
        out[name] = dict(
            pos_err_m=dict(mean=float(pe.mean()), median=float(
                np.median(pe))), rot_err_deg=dict(
                mean=float(ang.mean()), median=float(np.median(ang))),
            grip_acc=float(((g[:, 9] > 0) == (V["grip"] > 0.5)).mean()),
            type_acc=float((g[:, 10:13].argmax(1) == V["etype"]).mean()))
    # базовый прогноз: среднее АБСОЛЮТНОЕ положение цели по (задача,
    # корзина возраста) на A минус текущее положение руки
    bins = lambda a: (np.asarray(a) // 40).astype(int)   # noqa: E731
    tab = {}
    absA = A["state"][:, :3] + A["dpos"]
    for t, b_, g_ in zip(A["task"], bins(A["age"]), absA):
        tab.setdefault((t, b_), []).append(g_)
    glob_ = absA.mean(0)
    absP = np.stack([np.mean(tab[(t, b_)], 0) if (t, b_) in tab else glob_
                     for t, b_ in zip(V["task"], bins(V["age"]))])
    pe = np.linalg.norm(absP - V["state"][:, :3] - V["dpos"], axis=1)
    out["baseline_task_age"] = dict(pos_err_m=dict(
        mean=float(pe.mean()), median=float(np.median(pe))))
    rs = norms["r_sigma"]
    var0 = float((V["rstar"] ** 2).mean())
    varc = float(((V["rstar"] - V["rstar"].mean(0)) ** 2).mean())
    for name, (m, gk, sfk) in heads.items():
        rr, lo_n, hi_n = [], [], []
        with torch.no_grad():
            for s in range(0, len(V["age"]), 1024):
                i = np.arange(s, min(s + 1024, len(V["age"])))
                b = to_t(torch, V, i, dev, ("h18", "a0", sfk, gk))
                goal = b[gk].float()
                r, _ = m(b["h18"], b["a0"].float(), b[sfk].float(), goal)
                rr.append(r.cpu().numpy() * rs)
                if hasattr(m, "parts"):
                    lo, hi, _c = m.parts(b["h18"], b["a0"].float(),
                                         b[sfk].float(), goal)
                    lo_n.append(np.linalg.norm((lo.cpu().numpy() * rs)
                                               .reshape(len(i), -1), axis=1))
                    hi_n.append(np.linalg.norm((hi.cpu().numpy() * rs)
                                               .reshape(len(i), -1), axis=1))
        r = np.concatenate(rr)
        mse = float(((r - V["rstar"]) ** 2).mean())
        _a, info = km.apply_correction(V["a0"], r / rs, rs, scale, ALPHA,
                                       clip)
        e = dict(r_mse=mse, skill_vs_zero=1.0 - mse / max(var0, 1e-12),
                 r2=1.0 - mse / max(varc, 1e-12),
                 clip_rate=float(info["clipped"].mean()),
                 applied_norm_median=float(np.median(info["norm_applied"])))
        if lo_n:
            e["low_norm_median"] = float(np.median(np.concatenate(lo_n)))
            e["high_norm_median"] = float(np.median(np.concatenate(hi_n)))
        out[name] = e
    return out


def load_inputs(labels, cache, verify=True):
    import k15f_build_cache as kb
    L = dict(np.load(labels, allow_pickle=False))
    meta = json.load(open(labels[:-4] + ".json"))
    if meta.get("kind") != "k16_labels" or meta.get("problems"):
        raise SystemExit("разметка K-16 не того вида или с проблемами")
    if sha(labels) != meta["out"]["sha1"]:
        raise SystemExit("разметка K-16 изменена после подготовки")
    if sha(os.path.join(cache, "manifest.json")) != \
            meta["h18_manifest_sha1"]:
        raise SystemExit("кэш h18 не тот, по которому построена разметка")
    c = kb.load_cache(cache, verify_h18=verify)
    for p in kb.PARTS:
        if not np.array_equal(L[f"{p}__rows"], c[p]["rows"]):
            raise SystemExit(f"строки разметки {p} не равны кэшу h18")
        rs_ = (c[p]["act"][:, :8, :6].astype(np.float64)
               - c[p]["a0"][:, :8, :6].astype(np.float64))
        if not np.allclose(L[f"{p}__rstar"], rs_, atol=1e-6):
            raise SystemExit(f"rstar разметки {p} не равен act − a0 кэша")
    data = {}
    for name, part, key in (("A", "train", "selA"), ("B", "train", "selB"),
                            ("V", "val_sel", "selV")):
        idx = np.asarray(L[key])
        d = split_arrays(L, part, idx, c[part]["h18"])
        d["a0"] = np.asarray(c[part]["a0"][idx], np.float32)
        d["sf_hist"] = km.state_features(d["state"], d["hist"],
                                         d["hist_mask"], d["age"], True)
        d["sf_now"] = km.state_features(d["state"], d["hist"],
                                        d["hist_mask"], d["age"], False)
        data[name] = d
    return data, meta


def main():
    ap = argparse.ArgumentParser(description="K-16: обучение")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--labels", default="reports/k16/labels.npz")
    ap.add_argument("--h18-cache", default="data/k15f/h18_cache")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--report", default=REPORT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import torch
    import k15h_protocol as pc
    try:
        data, meta = load_inputs(a.labels, a.h18_cache, verify=True)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    dev = torch.device(a.device)
    try:
        goals, heads, rep, norms = train_all(torch, data, dev)
    except SystemExit as e:              # нечисловая ошибка — без чекпойнта
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    scale = pc.harness_scale()
    B = data["B"]
    clip = float(np.percentile(np.linalg.norm(
        (B["rstar"] * scale[:6]).reshape(len(B["rstar"]), -1), axis=1),
        CLIP_Q))
    ev = evaluate(torch, goals, heads, norms, data["V"], data["A"], scale,
                  clip, dev)
    code = {f: sha(os.path.join(HERE, f))[:12] for f in (
        "k16_prepare.py", "k16_models.py", "k16_train.py")}
    ck = dict(
        goals={k: dict(state=m.state_dict(), sf=sfk)
               for k, (m, sfk) in goals.items()},
        heads={k: dict(state=m.state_dict(), goal=gk, sf=sfk,
                       cls=type(m).__name__, hidden=rep[k]["hidden"])
               for k, (m, gk, sfk) in heads.items()},
        norms=norms, clip_env_norm=clip, clip_q=CLIP_Q, alpha=ALPHA,
        n_low=km.N_LOW, seed=SEED, units=km.UNITS,
        d_model=int(data["A"]["h18"].shape[2]),
        n_pos=int(data["A"]["h18"].shape[1]),
        scale_arm=np.asarray(scale[:6]).tolist(),
        provenance=dict(labels=os.path.abspath(a.labels),
                        labels_sha1=sha(a.labels),
                        h18_manifest_sha1=meta["h18_manifest_sha1"],
                        code=code))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(ck, a.out + ".tmp")
    os.replace(a.out + ".tmp", a.out)
    res = dict(train=rep, val_sel=ev, clip_env_norm=clip, alpha=ALPHA,
               models=dict(path=os.path.abspath(a.out), sha1=sha(a.out)),
               code=code,
               note=("ошибки на val_sel; в val_sel нет трёх задач пилота "
                     "(см. labels.json)"))
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1,
                  default=lambda x: x.tolist() if hasattr(x, "tolist")
                  else str(x))
    print(json.dumps(res["val_sel"], ensure_ascii=False, indent=1))
    print(f"  предел нормы поправки (p{CLIP_Q:.0f}, исполняемые команды): "
          f"{clip:.4f}; модели: {a.out}")
    return 0


# --- самопроверка ------------------------------------------------------------
def _synth(n, rng):
    st = rng.standard_normal((n, 8)) * 0.1
    hist = st[:, None] + rng.standard_normal((n, 3, 8)) * 0.01
    mask = np.ones((n, 3), bool)
    mask[: n // 10, 1:] = False
    age = rng.integers(0, 300, n)
    a0 = rng.uniform(-0.5, 0.5, (n, 8, 7)).astype(np.float32)
    h18 = rng.standard_normal((n, 16, 768)).astype(np.float16)
    dpos = (st[:, :3] * -1 + 0.05).astype(np.float64)
    import k16_prepare as kp
    R = kp.aa_to_mat(rng.standard_normal((n, 3)) * 0.3)
    d = dict(h18=h18, a0=a0, state=st, hist=hist, hist_mask=mask, age=age,
             dpos=dpos, drot6d=kp.mat_to_6d(R).astype(np.float32),
             grip=(rng.random(n) > 0.5).astype(np.float32),
             etype=rng.integers(0, 3, n),
             rstar=(np.tile(dpos[:, None, :3], (1, 8, 2)) * 0.3
                    ).astype(np.float32),
             task=np.asarray([f"t{i % 3}" for i in range(n)]),
             epi=np.arange(n) // 5)
    d["sf_hist"] = km.state_features(st, hist, mask, age, True)
    d["sf_now"] = km.state_features(st, hist, mask, age, False)
    return d


def selftest():
    import torch
    rng = np.random.default_rng(0)
    # признаки: вариант без истории — те же маски и возраст, значения 0
    d = _synth(50, rng)
    sh, sn = d["sf_hist"], d["sf_now"]
    assert sh.shape[1] == km.STATE_FEAT
    assert np.array_equal(sh[:, :11], sn[:, :11]) and np.array_equal(
        sh[:, -1], sn[:, -1])
    hv = sn[:, 11:-1].reshape(-1, 3, 12)
    assert not hv[..., :11].any() and np.array_equal(
        hv[..., 11], d["hist_mask"].astype(np.float32))
    # будущая разметка не входит в признаки
    d2 = dict(d, dpos=d["dpos"] * 0 + 9)
    assert np.array_equal(km.state_features(d2["state"], d2["hist"],
                                            d2["hist_mask"], d2["age"]),
                          sh)
    # DCT и частотная голова: части = обратное DCT коэффициентов; высокая
    # ветка получает ПРЕДСКАЗАНИЕ низкой
    C = km.build(torch)
    gb = C["GoalBand"](64)
    g = torch.randn(4, km.GOAL_DIM)
    sf = torch.tensor(sh[:4])
    a0 = torch.tensor(d["a0"][:4])
    lo, hi, coef = gb.parts(None, a0, sf, g)
    Dm = torch.tensor(km.dct_matrix(), dtype=torch.float32)
    assert torch.allclose(lo + hi, torch.einsum("ft,bfc->btc", Dm, coef),
                          atol=1e-5)
    full = torch.einsum("tf,bfc->btc", Dm, (lo + hi))     # прямое DCT
    assert torch.allclose(full, coef, atol=1e-5)
    with torch.no_grad():
        gb.low[-1].bias += 1.0                             # меняем низкую ветку
    lo2, hi2, _ = gb.parts(None, a0, sf, g)
    assert not torch.allclose(hi, hi2)                     # высокая её видит
    # параметры корректоров выровнены (direct — с вектором цели)
    t_, w_ = km.matched_widths(torch, C)
    for k, h in w_.items():
        assert abs(km.n_params(C[k](h)) - t_) <= 0.1 * t_
    # сквозное обучение на синтетике: головы получают ПРОГНОЗ цели
    data = dict(A=_synth(300, rng), B=_synth(300, rng), V=_synth(100, rng))
    goals, heads, rep, norms = train_all(torch, data, torch.device("cpu"),
                                         epochs=2, log=lambda *x: None)
    gnow = goals["goal_now"][0]
    with torch.no_grad():
        p = gnow(torch.tensor(data["B"]["h18"][:5]),
                 torch.tensor(data["B"]["a0"][:5]),
                 torch.tensor(data["B"]["sf_now"][:5])).numpy()
    assert np.allclose(p, data["B"]["pred_goal_now"][:5], atol=1e-5)
    assert not np.allclose(data["B"]["pred_goal_hist"][:, :3],
                           data["B"]["dpos"] / norms["dpos_sigma"])
    assert heads["goal_band_now"][1] == "pred_goal_now"
    assert heads["goal_flat"][1] == heads["goal_band"][1] == \
        "pred_goal_hist"
    assert heads["direct"][1] == "feat_goal_hist" and \
        heads["direct"][2] == "sf_hist"
    assert data["B"]["feat_goal_hist"].shape[1] == km.GOAL_HIDDEN
    # сид до конструктора: одинаковая инициализация при одинаковом сиде
    m1 = new_model(torch, C["GoalNet"])
    m2 = new_model(torch, C["GoalNet"])
    assert all(torch.equal(p1, p2) for p1, p2 in zip(m1.parameters(),
                                                     m2.parameters()))
    # нечисловая ошибка — отказ
    bad = dict(data["A"], dpos=data["A"]["dpos"] * np.nan)
    try:
        train_all(torch, dict(A=bad, B=data["B"], V=data["V"]),
                  torch.device("cpu"), epochs=1, log=lambda *x: None)
        raise AssertionError("принята нечисловая ошибка")
    except SystemExit:
        pass
    # единицы выхода и предел: один множитель, схват и хвост из a0
    a0x = np.random.default_rng(1).uniform(-.5, .5, (3, 20, 7)).astype(
        np.float32)
    outx = np.ones((3, 8, 6))
    ax, inf = km.apply_correction(a0x, outx, np.full(6, 0.1),
                                  np.full(7, 2.0), 0.5, 0.5)
    raw = 0.5 * 0.1 * 2.0 * np.sqrt(48)
    assert np.allclose(inf["norm_raw"], raw) and inf["clipped"].all()
    assert np.allclose(inf["norm_applied"], 0.5)
    assert np.array_equal(ax[:, :, 6], a0x[:, :, 6])
    assert np.array_equal(ax[:, 8:], a0x[:, 8:])
    assert np.allclose(ax[:, :8, :6] - a0x[:, :8, :6],
                       0.05 * 0.5 / raw * np.ones((3, 8, 6)), atol=1e-6)
    a0z, _ = km.apply_correction(a0x, np.zeros((3, 8, 6)), np.ones(6),
                                 np.ones(7), 0.5, 1.0)
    assert np.array_equal(a0z, a0x)                       # ноль -> a0
    for m, _s in goals.values():
        assert not any(p_.requires_grad for p_ in m.parameters())
    ev = evaluate(torch, goals, heads, norms, data["V"], data["A"],
                  np.ones(7), 1.0, torch.device("cpu"))
    assert "baseline_task_age" in ev and "low_norm_median" in \
        ev["goal_band"]
    print("самопроверка k16_train пройдена: признаки без будущего, now — "
          "маски без значений, DCT и обратное, высокая ветка видит "
          "прогноз низкой, параметры ±10%, головы на ПРОГНОЗЕ цели, "
          "предсказатели заморожены")
    return 0


if __name__ == "__main__":
    sys.exit(main())
