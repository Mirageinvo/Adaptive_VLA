#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-37, подтверждение: контроль T2-parent («профиль родителя»).

Тот же замороженный источник, actor T32 сида, маршрут, условие
допустимости и выбор позднего листа, что у T2 (k37_operator.T2Arm); тот же
вход x96 (признаки и профиль ВЫБРАННОГО ЛИСТА). Отличие — распределение
добавки по времени:
  D_extra[t] = 0.025 · w_p[t] / c_p · tanh(head(x96)), t < 8,
w_p, c_p — профиль и нормировка выбранного узла T32 (k32_operator.path_info,
rec["w_r"], rec["c"][r]; FP32 — те же значения в добавке и в журнале).
D_total = D_parent + D_extra -> общий ограничитель K-32 -> рука из Z0;
схват T32; недопустимый вызов или u ≡ 0 -> команда T32 побитово.

Команды (через k37_confirm.py):
  ctrl-cache --device D — по строкам prepare K-37 (те же, что у пилота):
    один экспорт на строку, сбор w_p, c_p, r; сверка с кешем пилота
    (строки, split, задачи, допустимость, маршрут, лист — точно;
    |Δx|, |ΔD_parent| ≤ 1e−5; |Δw_leaf|, |Δc_leaf| ≤ 1e−6).
  ctrl-fit — головка Linear(96, 6) на сид: k37_fit.train_head без изменений
    (AdamW, 8 эпох, batch 64, то же перемешивание, α 0.025, рег. 0.001);
    входы, цели, D_parent, Z0 — из кеша пилота; w, c — родительские.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k37_fit as kfit  # noqa: E402
import k37_operator as kop  # noqa: E402

OUT = "reports/k37c"
DATA = "data/k37c"
TOL = dict(x=1e-5, Dp=1e-5, w=1e-6, c=1e-6)


def cache_path(s):
    return os.path.join(DATA, f"ctrl_cache_s{s}.npz")


def head_path(s):
    return os.path.join(DATA, f"head_parent_s{s}.pt")


def report_path(s):
    return os.path.join(OUT, f"ctrl_fit_s{s}.json")


class ParentArm(kop.T2Arm):
    """T32 + головка; добавка по профилю выбранного узла T32."""

    def decide(self, batch, po, state8, elapsed):
        torch = self.torch
        a0, rec = self.base.decide(batch, po, state8, elapsed)
        F, s11n = self.cap["F"], self.cap["s11n"]
        r = int(rec.get("r", 2))
        ext = dict(base_a_codec=a0.copy(), k37_reason=kop.REASON["leaf_route"],
                   k37_leaf=-1, k37_w=np.zeros(kop.N8, np.float32),
                   k37_c=np.float32(0), k37_u=np.zeros(6, np.float32),
                   k37_Dl=np.zeros((kop.N8, 6), np.float32),
                   k37_Du2=np.zeros((kop.N8, 6), np.float32))
        if "logits" not in rec or rec.get("zero", True):
            ext["k37_reason"] = kop.REASON["zero_parent"] if "logits" in rec \
                else kop.REASON["leaf_route"]
            rec.update(ext)
            return a0, rec
        W2 = np.asarray(F["W2"][0].float().cpu().numpy(), np.float64)
        leaf, why = kop.choose_leaf(W2, r, rec["path"])
        ext["k37_reason"] = why
        if leaf is None:
            rec.update(ext)
            return a0, rec
        wl64, cl64 = kop.leaf_profile(W2, leaf)
        wl, cl = wl64.astype(np.float32), np.float32(cl64)
        x = kop.leaf_actor_input(F, s11n[0].float().cpu().numpy(), elapsed,
                                 rec["Z0"], self.half, leaf, cl)
        h = kop.actor_hidden(torch, self.base.actor, x)
        x96 = kop.head_input(h, rec["eta"], rec["Du"], wl, float(cl), r)
        u = kop.head_u(torch, self.head, x96)
        wp = np.asarray(rec["w_r"], np.float32)[:kop.N8].copy()
        cp = np.float32(np.asarray(rec["c"], np.float32)[r])
        ext.update(k37_leaf=int(leaf), k37_w=wp, k37_c=cp, k37_u=u,
                   k37_x=x96, k37_wleaf=wl, k37_cleaf=cl)
        if not u.any():
            rec.update(ext)
            return a0, rec
        Dl = kop.leaf_delta(wp, float(cp), u)
        arm, Du2, _lim = kop.compose(torch, rec["Z0"], self.half, self.mid,
                                     rec["D"], Dl)
        a = np.concatenate([arm, a0[:, 6:7]], 1).astype(np.float32)
        if not np.isfinite(a).all():
            raise SystemExit("K-37 T2-parent: команда не конечна")
        ext.update(k37_Dl=Dl, k37_Du2=Du2)
        rec.update(ext)
        rec["a_codec"] = a
        self.base.last = dict(rec=rec)
        return a, rec


def parent_journal_problems(z):
    """T2-parent: применённый профиль = профилю узла T32 из журнала K-32
    (w_r и c[r]) на каждом активном вызове."""
    J = {k[2:]: v for k, v in z.items() if k.startswith("j_")}
    K = {k[2:]: v for k, v in z.items() if k.startswith("k_")}
    p = []
    for i in range(len(K.get("reason", []))):
        if int(K["reason"][i]) != 0:
            continue
        r = int(J["r"][i])
        if not (np.array_equal(K["w"][i], J["w"][i]) and
                np.float32(K["c"][i]) == np.float32(J["c"][i][r])):
            p.append(f"вызов {i}: профиль ≠ профилю узла T32")
    return p


def leaf_journal_problems(z):
    """T2: на активном вызове профиль — листа (≠ профилю узла T32 хотя бы
    где-то не обязателен); проверяется только, что лист — потомок узла."""
    import k31_readout as kr
    J = {k[2:]: v for k, v in z.items() if k.startswith("j_")}
    K = {k[2:]: v for k, v in z.items() if k.startswith("k_")}
    p = []
    for i in range(len(K.get("reason", []))):
        if int(K["reason"][i]) != 0:
            continue
        r = int(J["r"][i])
        _a, _b, leaves = kr.descendants(kr.LEVELS[r], int(J["path"][i][r]))
        if int(K["leaf"][i]) not in leaves:
            p.append(f"вызов {i}: лист не потомок узла")
    return p


# --- кеш и обучение ---------------------------------------------------------------------
def _ctrl_seed(torch, p37, k36, D, s, device, pilot):
    import k20_data as kd20
    import k36_harness as kh
    v = kh.load_source(torch, s, device)
    t2 = kop.T2Arm(torch, kh.make_arm(torch, k36, f"T32_s{s}", v),
                   kop.make_head(torch))
    acc = dict(row=[], split=[], task=[], wp=[], cp=[], r=[], leaf=[],
               x=[], Dp=[], w=[], c=[])
    for t in (0, 6, 8):
        for split, nm in ((0, "train"), (1, "diag")):
            for r in p37["rows"][str(t)][nm]:
                batch = kd20.prompt_batch(
                    v["bar"], [np.array(D["IMG"][r], copy=True)],
                    kd20.norm_states(np.asarray(D["state8"][[r]])),
                    [D["task"][r]])
                t2.begin(t, 0, 0)
                _a, rec = t2.decide(batch, int(D["pos_offset"][r]),
                                    np.asarray(D["state8"][[r]], np.float64),
                                    int(D["step"][r]))
                if rec["k37_reason"] != 0:
                    continue
                rr = int(rec["r"])
                acc["row"].append(r)
                acc["split"].append(split)
                acc["task"].append(t)
                acc["wp"].append(np.asarray(rec["w_r"], np.float32)[:8])
                acc["cp"].append(np.float32(np.asarray(rec["c"],
                                                       np.float32)[rr]))
                acc["r"].append(rr)
                acc["leaf"].append(int(rec["k37_leaf"]))
                acc["x"].append(rec["k37_x"])
                acc["Dp"].append(np.asarray(rec["D"], np.float32))
                acc["w"].append(rec["k37_w"])
                acc["c"].append(rec["k37_c"])
    del v, t2
    kh.free(torch)
    z = {k: np.asarray(x) for k, x in acc.items()}
    return z, compare_with_pilot(z, pilot)


def compare_with_pilot(z, pilot):
    """-> (problems, max-расхождения). Строки/split/задачи — точно."""
    p = []
    for k in ("row", "split", "task"):
        if not np.array_equal(np.asarray(z[k]), np.asarray(pilot[k])):
            p.append(f"{k}: состав подходящих строк ≠ кешу пилота")
    if p:
        return p, {}
    d = {}
    for k, pk in (("x", "x"), ("Dp", "Dp"), ("w", "w"), ("c", "c")):
        d[k] = float(np.abs(np.asarray(z[k], np.float64) -
                            np.asarray(pilot[pk], np.float64)).max()) \
            if len(z[k]) else 0.0
        if not d[k] <= TOL[k]:
            p.append(f"{k}: |Δ| {d[k]:.2e} > {TOL[k]:.0e}")
    if not (np.isfinite(z["wp"]).all() and np.isfinite(z["cp"]).all() and
            (np.asarray(z["cp"]) >= 0.05 - 1e-7).all()):
        p.append("w_p/c_p не конечны или c_p < 0.05")
    return p, d


def ctrl_cache(a):
    import torch
    import k21_data as d21
    import k36_protocol as pc
    import k37_parent_leaf as kpl
    p37, k36 = kpl.load_prepare()
    c = pc.load_json(os.path.join(kpl.OUT, "cache.json"))
    if c.get("ok") is not True or c.get("prepare_sha1") != p37["sha1"]:
        raise SystemExit("кеш пилота не пройден или от другого prepare")
    D = d21.load_prepared(require_teacher=False)
    rep, ok = {}, True
    for s in (0, 1):
        if pc.sha(kpl.cache_path(s)) != c[str(s)]["sha1"]:
            raise SystemExit(f"кеш пилота s{s} изменён")
        pilot = dict(np.load(kpl.cache_path(s)))
        z, (prob, diff) = _ctrl_seed(torch, p37, k36, D, s, a.device, pilot)
        os.makedirs(DATA, exist_ok=True)
        with open(cache_path(s) + ".tmp", "wb") as fh:
            np.savez(fh, **z)
        os.replace(cache_path(s) + ".tmp", cache_path(s))
        lev = np.bincount(np.asarray(z["r"], int), minlength=3).tolist()
        rep[str(s)] = dict(n=int(len(z["row"])), diff=diff, problems=prob,
                           routes_root_mid_leaf=lev,
                           pilot_cache_sha1=c[str(s)]["sha1"],
                           sha1=pc.sha(cache_path(s)))
        ok &= not prob
        print(f"  контрольный кеш s{s}: строк {len(z['row'])}, маршруты "
              f"root/mid/leaf {lev}; расхождения с пилотом {diff}; " +
              ("ок" if not prob else "; ".join(prob)), flush=True)
    rep.update(prepare_sha1=p37["sha1"], ok=bool(ok), tol=TOL)
    pc.save_json(os.path.join(OUT, "ctrl_cache.json"), rep)
    return 0 if ok else 3


def ctrl_fit(a):
    import torch
    import k36_protocol as pc
    import k37_parent_leaf as kpl
    p37, _k36 = kpl.load_prepare()
    cc = pc.load_json(os.path.join(OUT, "ctrl_cache.json"))
    pcache = pc.load_json(os.path.join(kpl.OUT, "cache.json"))
    if cc.get("ok") is not True or cc.get("prepare_sha1") != p37["sha1"]:
        raise SystemExit("контрольный кеш не пройден")
    for s in (0, 1):
        if pc.sha(cache_path(s)) != cc[str(s)]["sha1"] or \
                pc.sha(kpl.cache_path(s)) != pcache[str(s)]["sha1"]:
            raise SystemExit(f"кеш s{s} изменён после проверки")
        zp = dict(np.load(kpl.cache_path(s)))
        zc = dict(np.load(cache_path(s)))
        if not np.array_equal(zp["row"], zc["row"]):
            raise SystemExit(f"s{s}: строки кешей разошлись")
        z = dict(zp, w=zc["wp"], c=zc["cp"])     # распределение — родителя
        head, rep = kfit.train_head(torch, z, s)
        prob = []
        for k, v in head.state_dict().items():
            if not bool(torch.isfinite(v).all()):
                prob.append(f"{k} не конечен")
        if not rep["grad_init"] > 0:
            prob.append("нулевой начальный градиент")
        if not rep["weight_change"] > 0:
            prob.append("веса не изменились")
        hh = kfit.head_hash(torch, head)
        os.makedirs(DATA, exist_ok=True)
        torch.save(dict(head=head.state_dict(), head_hash=hh, seed=s,
                        kind="T2P", prepare_sha1=p37["sha1"],
                        ctrl_cache_sha1=cc[str(s)]["sha1"],
                        pilot_cache_sha1=pcache[str(s)]["sha1"]),
                   head_path(s) + ".tmp")
        os.replace(head_path(s) + ".tmp", head_path(s))
        import k36_inputs as kin
        cov = {str(t): dict(train=int(((zp["task"] == t) &
                                       (zp["split"] == 0)).sum()),
                            diag=int(((zp["task"] == t) &
                                      (zp["split"] == 1)).sum()))
               for t in (0, 6, 8)}
        rep.update(kind="T2P", seed=s, head_hash=hh,
                   head_sha1=pc.sha(head_path(s)),
                   head_sha256=kin.sha256(head_path(s)),
                   prepare_sha1=p37["sha1"], rows_sha1=p37["rows_sha1"],
                   coverage=cov, problems=prob, ok=not prob)
        pc.save_json(report_path(s), rep)
        print(f"  контрольная головка s{s}: train {rep['n_train']} / diag "
              f"{rep['n_diag']}; потеря diag {rep['diag']} (нулевая "
              f"{rep['diag_zero']}); изменение весов "
              f"{rep['weight_change']:.3g}" +
              ("" if not prob else " — " + "; ".join(prob)), flush=True)
        if prob:
            return 3
    return 0


def selftest():
    import torch
    import k21_model as km
    import k25_model as k25m
    import k29_model as k29m
    import k31_readout as kr
    import k32_model as km32
    import k32_policy as kpol
    k29m.no_tf32(torch)
    torch.manual_seed(0)                     # игрушка не зависит от RNG
    M = k29m.build(torch)
    E = kr.build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    mod = k25m.build(torch)["K25"](km.fake_bar(torch), L, U, seed=0).eval()
    with torch.no_grad():
        g = torch.Generator().manual_seed(3)
        for p in list(mod.cond.parameters()) + list(mod.readout.parameters()):
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    for p in mod.parameters():
        p.requires_grad_(False)
    dec = M["Decoder"](mod).freeze()
    H = mod.H
    norm = km32.fit_normalizer(np.random.default_rng(0).normal(
        size=(64, 6 * H + 72)), H)
    actor, _c = km32.build(torch, H, 6, 0, norm)
    with torch.no_grad():
        g2 = torch.Generator().manual_seed(7)
        for p in actor.out.parameters():
            p.copy_(torch.randn(p.shape, generator=g2))
    n_act, worst, differ = 0, 0.0, 0
    for seed in range(60):
        if n_act >= 3 and differ >= 1:
            break
        mk = lambda: kpol.Arm(torch, mod, dec, M, kind="T", seed=0,  # noqa
                              actor=actor)
        head = kop.make_head(torch)
        bt = km.fake_batch(torch, B=1, seed=seed)
        s8 = np.random.default_rng(seed).normal(size=(1, 8)) * 0.1
        base = mk()
        base.begin(0, 0, 0)
        a_t32, _r = base.decide(bt, 1, s8, 8)
        pa = ParentArm(torch, mk(), head)
        pa.begin(0, 0, 0)
        a_zero, rz = pa.decide(bt, 1, s8, 8)
        assert np.array_equal(a_zero, a_t32), "нулевая T2-parent ≠ T32"
        if rz["k37_reason"] != 0:
            continue
        n_act += 1
        with torch.no_grad():
            head.bias.copy_(torch.as_tensor(
                np.random.default_rng(seed).normal(size=6) * 2.0))
        la = kop.T2Arm(torch, mk(), head)
        la.begin(0, 0, 0)
        a_leaf, rl = la.decide(bt, 1, s8, 8)
        pa.begin(0, 0, 0)
        a_par, rp = pa.decide(bt, 1, s8, 8)
        # тот же вход x96 и тот же u; разный профиль добавки
        assert np.array_equal(rl["k37_x"], rp["k37_x"])
        assert np.array_equal(rl["k37_u"], rp["k37_u"])
        assert np.array_equal(rp["k37_w"], rp["w_r"][:8])
        assert rp["k37_c"] == np.float32(rp["c"][rp["r"]])
        differ += int(not np.array_equal(a_leaf, a_par))
        # эталон: профиль узла P32-формулой = D_extra
        w = np.asarray(rp["w_r"], np.float64)[:8]
        ref = (w[:, None] / float(rp["c"][rp["r"]]) * 0.025 *
               np.tanh(np.asarray(rp["k37_u"], np.float64))[None])
        worst = max(worst, float(np.abs(ref - rp["k37_Dl"]).max()))
        assert np.array_equal(a_par[:, 6], a_t32[:, 6])
        assert not np.array_equal(a_par[:, :6], a_t32[:, :6])
        _ = E
    assert n_act >= 1 and differ >= 1, (n_act, differ)
    assert worst <= 1e-7, worst
    # сверка с кешем пилота: подмена строки/задачи — отказ; допуски
    zp = dict(row=np.arange(4), split=np.zeros(4, int), task=np.zeros(4, int),
              x=np.zeros((4, 96), np.float32),
              Dp=np.zeros((4, 8, 6), np.float32),
              w=np.zeros((4, 8), np.float32), c=np.full(4, 0.1, np.float32))
    zc = dict(zp, wp=np.full((4, 8), 0.1, np.float32),
              cp=np.full(4, 0.1, np.float32))
    assert compare_with_pilot(zc, zp)[0] == []
    assert compare_with_pilot(dict(zc, row=np.arange(1, 5)), zp)[0]
    bad = dict(zc, x=zc["x"] + 1e-4)
    assert compare_with_pilot(bad, zp)[0]
    print(f"самопроверка k37_control пройдена: нулевая T2-parent = T32 "
          f"побитово; тот же x96/u, что у T2, профиль — узла T32 (w_r, c[r]); "
          f"добавка = эталону (|Δ| ≤ {worst:.1e}); команды T2 и T2-parent "
          f"различаются на {differ}/{n_act} активных; сверка кеша отвергает "
          f"подмену строк и |Δx| > 1e−5")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; запуск — через k37_confirm.py")
