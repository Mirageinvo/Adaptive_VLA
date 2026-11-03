#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-31: рука на замороженном parents_cond24 (B = 1): reference, repair
(noop + древесные кандидаты) и flatrepair (noop + плоские кандидаты).

Вызов: один тяжёлый проход (k29_model.export под fp16 autocast, no_grad,
TF32 выкл.), FP32 reference k29_model.Decoder (autocast выкл.) — Z0, рука
reference = ref["arm"][:8], схват = знак логита reference (±1) всегда.
repair/flatrepair, пока право на ремонт не использовано: пул k31_candidates
(путь по W2, правки k31_readout.edited, плоские AR(1), нормировка, общий m),
149 признаков, CPU-селектор (score = p(rescue) − p(harm), noop = 0),
выбор в своей семье по порогу 0.10. Выбранная поправка заменяет руку на
этом чанке; после исполнения хотя бы одной её команды used_repair = True и
до конца эпизода — только reference. used_repair — на эпизод (begin()
сбрасывает), не глобальный.
Журнал вызова (rec): задача/старт/вызов/elapsed, used до/после, путь и
узлы, Z0/рука/логит схвата reference, валидность/m/λ/Du/руки пула,
признаки [7,149], логиты/score селектора, выбранный ID, команды.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k31_candidates as kc  # noqa: E402
import k31_readout as kr  # noqa: E402

N8 = kc.N8
MODES = ("reference", "repair", "flatrepair")
FAMILY = dict(repair="tree", flatrepair="flat")


def ctx_vectors(F, path):
    """F одного наблюдения -> (средние [3H], путь [3H]) float32 numpy."""
    x0 = F["x0"][0].float().cpu().numpy()
    x1 = F["x1"][0].float().cpu().numpy()
    x2 = F["x2"][0].float().cpu().numpy()
    mean = np.concatenate([x0.mean(0), x1.mean(0), x2.mean(0)])
    pth = np.concatenate([x0[path[0]], x1[path[1]], x2[path[2]]])
    return mean.astype(np.float32), pth.astype(np.float32)


class Arm:
    def __init__(self, torch, mod, dec, M, *, mode, source_seed, space=0,
                 selector=None):
        if mode not in MODES:
            raise ValueError(mode)
        if mode != "reference" and selector is None:
            raise SystemExit(f"{mode}: нужен селектор")
        self.torch, self.mod, self.dec, self.M = torch, mod, dec, M
        self.E = kr.build(torch)
        self.mode, self.seed, self.space = mode, int(source_seed), int(space)
        self.selector = selector
        self.dev = next(mod.parameters()).device
        self.L = np.asarray(mod._L, np.float64)[:6]
        self.U = np.asarray(mod._U, np.float64)[:6]
        h, m = self.E["half_mid"](dec._L, dec._U, "cpu")
        self.half_cpu, self.mid_cpu = h, m
        self.autocast = torch.autocast(self.dev.type, dtype=torch.float16,
                                       enabled=self.dev.type == "cuda")
        self.task = self.init = None
        self.used = False
        self.call = 0
        self.last = None

    def begin(self, task, init):
        self.task, self.init = int(task), int(init)
        self.used = False
        self.call = 0
        self.last = None

    def forward(self, batch, po, state8):
        import k21_model as km
        import k25_policy as kp25
        torch = self.torch
        s11 = torch.as_tensor(km.state11(np.asarray(state8, np.float64)),
                              device=self.dev)
        with torch.no_grad(), self.autocast:
            F, s11n, cnt = self.M["export"](self.mod, batch, po, s11)
        pr = kp25.counters_problems(cnt)
        if pr:
            raise SystemExit("тяжёлый проход: " + "; ".join(pr))
        with torch.no_grad():
            ref = self.dec(F)
        return F, s11n, ref, cnt

    def proposals(self, F, ref, call):
        """Пул одного наблюдения (RNG — по задаче/старту/вызову эпизода)."""
        torch = self.torch
        W2 = F["W2"][0].float().cpu().numpy()
        path = kr.path_of(W2)
        info = kr.node_info(F, path)
        d, g = kc.tree_direction(kc.rng(self.space, self.seed, self.task,
                                        self.init, call, "tree"))
        flat = kc.flat_arrays(kc.rng(self.space, self.seed, self.task,
                                     self.init, call, "flat"))
        o = self.E["edited"](self.dec, F, ref, list(kr.LEVELS), list(path),
                             np.stack([d] * 3))
        Z0 = ref["Z"][0].detach().float().cpu()
        Zhat = o["Zhat"].detach().float().cpu()
        P = kc.pool(torch, Z0, Zhat, flat, self.half_cpu, self.mid_cpu)
        return dict(path=path, info=info, d=d, g=g, flat=flat, Z0=Z0,
                    Zhat=Zhat, P=P)

    def decide(self, batch, po, state8, elapsed):
        torch = self.torch
        self.call += 1
        F, s11n, ref, cnt = self.forward(batch, po, state8)
        ref_arm = ref["arm"][0, :N8].detach().float().cpu()
        gl = ref["grip_logit"][0, :N8].detach().float().cpu()
        grip = torch.where(gl > 0, 1.0, -1.0)
        rec = dict(task=self.task, init=self.init, call=self.call,
                   elapsed=int(elapsed), used_before=bool(self.used),
                   Z0=ref["Z"][0].detach().float().cpu().numpy(),
                   ref_arm=ref_arm.numpy(), grip_logit=gl.numpy(),
                   chosen=0, chosen_name="noop")
        arm = ref_arm
        if self.mode != "reference" and not self.used:
            pr = self.proposals(F, ref, self.call)
            P = pr["P"]
            mean, pth = ctx_vectors(F, pr["path"])
            ctx = self.selector.ctx(mean, pth)
            X = kc.features(ctx, s11n[0].float().cpu().numpy(), elapsed,
                            (pr["Z0"] / self.half_cpu).numpy(), P,
                            pr["info"])
            logits, scores = self.selector.scores(X)
            scores = np.where(P["allowed"], scores, -np.inf)
            scores[0] = 0.0
            ch = kc.choose(scores, P["allowed"], FAMILY[self.mode])
            if ch:
                arm = P["arm"][ch]
            rec.update(path=list(pr["path"]), info=pr["info"], d=pr["d"],
                       flat=pr["flat"], valid=P["valid"],
                       allowed=P["allowed"], m=P["m"],
                       lam=np.array([r["lam"] if (P["valid"][i] and i)
                                     else np.nan
                                     for i, r in enumerate(P["rows"])],
                                    np.float64),
                       lam_raw=[r["lam"] for r in P["rows"]],
                       why=[r["why"] for r in P["rows"]],
                       Du=np.stack([np.asarray(r["Du"], np.float32)
                                    if P["valid"][i] else
                                    np.full((kc.T, 6), np.nan, np.float32)
                                    for i, r in enumerate(P["rows"])]),
                       pool_arm=P["arm"].numpy(), X=X,
                       sel_logits=logits, scores=scores, chosen=int(ch),
                       chosen_name=kc.ROWS[ch])
        a = torch.cat([arm, grip[:, None]], -1).numpy().astype(np.float32)
        if not np.isfinite(a).all():
            raise SystemExit("команды не конечны")
        if ((a[:, :6] < self.L - 1e-4) | (a[:, :6] > self.U + 1e-4)).any():
            raise SystemExit("рука вне пределов кодека")
        rec["a_codec"] = a
        self.last = dict(F=F, s11n=s11n, ref=ref, rec=rec)
        return a, rec

    def executed(self, k):
        """После шага среды: исполнено k ≥ 1 команд выбранного чанка."""
        rec = self.last["rec"]
        if rec["chosen"] and int(k) >= 1:
            self.used = True
        rec["k"] = int(k)
        rec["used_after"] = bool(self.used)


def selftest():
    import torch
    import k21_model as km
    import k25_model as k25m
    import k29_model as k29m
    k29m.no_tf32(torch)
    M = k29m.build(torch)
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

    class Sel:
        def __init__(self, s):
            self.s = np.asarray(s, np.float64)

        def ctx(self, mean, pth):
            assert mean.shape == pth.shape
            return np.zeros(32, np.float32)

        def scores(self, X):
            assert X.shape == (7, kc.N_FEAT)
            return np.zeros((7, 3), np.float32), self.s.copy()
    st8 = np.concatenate([np.random.default_rng(0).normal(0, .1, (1, 3)),
                          np.random.default_rng(1).normal(0, .5, (1, 3)),
                          np.zeros((1, 2))], 1)
    bt = km.fake_batch(torch, B=1, seed=2)
    ref_arm = Arm(torch, mod, dec, M, mode="reference", source_seed=0)
    ref_arm.begin(8, 3)
    a_ref, r_ref = ref_arm.decide(bt, 1, st8, 0)
    with torch.no_grad():
        F, _s, _c = M["export"](mod, bt, 1, torch.as_tensor(
            km.state11(st8)))
        o = dec(F)
    assert np.array_equal(a_ref[:, :6], o["arm"][0, :8].numpy())
    assert set(np.unique(a_ref[:, 6])) <= {-1.0, 1.0}
    # repair: выбирается tree с наибольшим score; схват = reference
    sel = Sel([0, .5, .3, .2, .9, .1, .1])
    rp = Arm(torch, mod, dec, M, mode="repair", source_seed=0, selector=sel)
    rp.begin(8, 3)
    a1, r1 = rp.decide(bt, 1, st8, 0)
    ok_tree = [i for i in (1, 2, 3) if r1["allowed"][i]]
    assert r1["chosen"] in ok_tree and r1["chosen"] in (1, 2, 3)
    assert np.array_equal(a1[:, 6], a_ref[:, 6])
    assert not np.array_equal(a1[:, :6], a_ref[:, :6])
    assert np.array_equal(a1[:, :6], r1["pool_arm"][r1["chosen"]])
    # тот же вызов того же эпизода -> те же кандидаты (RNG по ключу)
    rp2 = Arm(torch, mod, dec, M, mode="repair", source_seed=0, selector=sel)
    rp2.begin(8, 3)
    a1b, r1b = rp2.decide(bt, 1, st8, 0)
    assert np.array_equal(a1, a1b) and np.array_equal(r1["Du"], r1b["Du"],
                                                      equal_nan=True)
    # после исполнения выбранного чанка — только reference
    rp.executed(8)
    assert rp.used
    a2, r2 = rp.decide(bt, 1, st8, 8)
    assert r2["chosen"] == 0 and r2["used_before"] and \
        np.array_equal(a2, a_ref) and "X" not in r2
    # если среда не исполнила ни одной команды — право не потрачено
    rp.begin(8, 3)
    rp.decide(bt, 1, st8, 0)
    rp.executed(0)
    assert not rp.used
    # begin сбрасывает право (новый эпизод), другой эпизод не влияет
    rp.begin(8, 4)
    assert not rp.used and rp.call == 0
    # flatrepair выбирает только плоские; порог не пройден -> noop
    fr = Arm(torch, mod, dec, M, mode="flatrepair", source_seed=0,
             selector=sel)
    fr.begin(8, 3)
    a3, r3 = fr.decide(bt, 1, st8, 0)
    assert r3["chosen"] in (4, 5, 6)
    lo = Arm(torch, mod, dec, M, mode="repair", source_seed=0,
             selector=Sel([0, .1, .1, .1, .9, .9, .9]))
    lo.begin(8, 3)
    a4, r4 = lo.decide(bt, 1, st8, 0)
    assert r4["chosen"] == 0 and np.array_equal(a4, a_ref)
    lo.executed(8)
    assert not lo.used
    try:
        Arm(torch, mod, dec, M, mode="repair", source_seed=0)
        raise AssertionError("repair без селектора принят")
    except SystemExit:
        pass
    print("самопроверка k31_policy пройдена: reference = FP32 reference "
          "K-29 побитово, схват — знак логита reference во всех режимах; "
          "repair/flatrepair выбирают только свою семью по порогу, "
          "кандидаты воспроизводимы по ключу вызова; после исполнения "
          "ремонта — только reference; k = 0 не тратит право; begin "
          "сбрасывает право на эпизод")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
