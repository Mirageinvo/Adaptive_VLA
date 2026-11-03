#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-31: CPU-селектор полезности (149 -> 64 -> 32 -> 3, GELU), обучение,
fitcheck, проверка активности и загрузка для руки.

Классы: harm, neutral, rescue (Rc − R0 = −1/0/+1); score = p(rescue) −
p(harm); noop = 0 напрямую. Последний Linear — нули, остальные — локальный
U(±1/√fan_in) от torch.Generator(31000 + seed); глобальный RNG не трогается.
Обучение: float32, AdamW lr 1e−3 (постоянный), wd 1e−4, clip 1.0, 1000
обновлений; шаг — 16 train-checkpoint (равномерно с возвращением по
локальному np.random.Generator(31100 + seed)), внутри — средняя CE по
валидным non-noop кандидатам, без перевзвешивания классов. Проверки: 0 и
каждые 50; best — минимальная validation CE (равный вес checkpoint), при
равенстве раньше. Сохранение каждые 50: веса, оптимизатор, update, RNG,
best, лог, отпечаток набора/кода; возобновление = непрерывному обучению.
fitcheck: временная копия на ≤ 64 train-кандидатах (детерминированно, все
представленные классы), 500 шагов, lr 3e−3: конечность, применённые
обновления, CE падает минимум вдвое.
"""
import copy
import hashlib
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k31_candidates as kc  # noqa: E402
import k31_data as kdat  # noqa: E402

CFG = dict(hidden=(64, 32), n_out=3, updates=1000, batch=16, lr=1e-3,
           wd=1e-4, clip=1.0, check_every=50, save_every=50, fit_n=64,
           fit_steps=500, fit_lr=3e-3, init_seed_base=31000,
           sample_seed_base=31100, threshold=kc.THRESH)
N_PARAMS = 149 * 64 + 64 + 64 * 32 + 32 + 32 * 3 + 3        # 11 779


def make_model(torch, seed):
    nn = torch.nn
    g = torch.Generator().manual_seed(CFG["init_seed_base"] + int(seed))

    class Sel(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(kc.N_FEAT, 64)
            self.l2 = nn.Linear(64, 32)
            self.l3 = nn.Linear(32, 3)
            with torch.no_grad():
                for m in (self.l1, self.l2):
                    b = 1.0 / math.sqrt(m.in_features)
                    m.weight.copy_((torch.rand(m.weight.shape, generator=g)
                                    * 2 - 1) * b)
                    m.bias.copy_((torch.rand(m.bias.shape, generator=g)
                                  * 2 - 1) * b)
                self.l3.weight.zero_()
                self.l3.bias.zero_()

        def forward(self, x):
            g_ = nn.functional.gelu
            return self.l3(g_(self.l2(g_(self.l1(x)))))
    with torch.random.fork_rng(devices=[]):   # nn.Linear трогает глобальный
        return Sel()


def weights_hash(mod):
    h = hashlib.sha1()
    for k, v in sorted(mod.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def gain(torch, logits):
    p = torch.softmax(logits, -1)
    return p[..., 2] - p[..., 0]


def group_ce(torch, mod, X, y):
    """X [n,7,149], y [n,7] (−1 / 0 / класс; noop строка 0 не учится) ->
    CE по группам [n] (среднее по валидным non-noop) и маска групп."""
    m = (y >= 0)
    m[:, 0] = False
    lg = mod(torch.as_tensor(X))
    ce = torch.nn.functional.cross_entropy(
        lg.reshape(-1, 3), torch.as_tensor(np.where(m, y, 0)).reshape(-1),
        reduction="none").reshape(m.shape)
    mt = torch.as_tensor(m, dtype=torch.float32)
    cnt = mt.sum(1)
    has = cnt > 0
    per = (ce * mt).sum(1) / cnt.clamp_min(1)
    return per, has, lg


def evaluate(torch, mod, part):
    X, y, al = part["X"], part["y"], part["allowed"]
    if len(X) == 0:
        return dict(ce=None, n=0)
    with torch.no_grad():
        per, has, lg = group_ce(torch, mod, X, y)
        gh = gain(torch, lg).numpy()
    ce = float(per[has].mean()) if bool(has.any()) else None
    m = (y >= 0)
    m[:, 0] = False
    pred = lg.argmax(-1).numpy()
    cls = {}
    for c, nm in enumerate(kdat.CLASSES):
        sel = m & (y == c)
        cls[nm] = dict(n=int(sel.sum()), acc=float((pred[sel] == c).mean())
                       if sel.any() else None)
    lab = np.where(m, y - 1, 0).astype(np.float64)
    mse = float(((gh - lab)[m] ** 2).mean()) if m.any() else None
    act = dict(tree=0, flat=0)
    for i in range(len(X)):
        s = np.where(al[i], gh[i], -np.inf)
        s[0] = 0.0
        for fam in ("tree", "flat"):
            act[fam] += int(kc.choose(s, al[i], fam) > 0)
    q = np.quantile(gh[m], [0.05, 0.5, 0.95]).tolist() if m.any() else None
    return dict(ce=ce, n=int(len(X)), groups_without_label=int((~has).sum()),
                classes=cls, mse_gain=mse, gain_q=q, would_intervene=act)


class Trainer:
    def __init__(self, torch, D, seed, out_dir, extra_contract=None):
        self.torch, self.D, self.seed = torch, D, int(seed)
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "state.pt")
        self.contract = dict(cfg=CFG, seed=self.seed, dataset_sha=D["sha"],
                             **(extra_contract or {}))
        self.mod = make_model(torch, seed)
        self.opt = torch.optim.AdamW(self.mod.parameters(), lr=CFG["lr"],
                                     weight_decay=CFG["wd"])
        self.gen = np.random.Generator(np.random.PCG64(
            CFG["sample_seed_base"] + self.seed))
        self.u = 0
        self.best = None
        self.log = []
        m = (D["train"]["y"] >= 0)
        m[:, 0] = False
        self.pool = np.nonzero(m.any(1))[0]
        if len(self.pool) == 0:
            raise kdat.InformationalStop("нет train-checkpoint с метками")
        if os.path.exists(self.path):
            self._load()

    def _state(self):
        return dict(model=self.mod.state_dict(), opt=self.opt.state_dict(),
                    u=self.u, best=self.best, log=self.log,
                    gen=self.gen.bit_generator.state,
                    contract=self.contract)

    def _save(self):
        self.torch.save(self._state(), self.path + ".tmp")
        os.replace(self.path + ".tmp", self.path)

    def _load(self):
        st = self.torch.load(self.path, map_location="cpu",
                             weights_only=False)
        if st["contract"] != self.contract:
            raise SystemExit(f"{self.path}: другой контракт (набор/код/seed)")
        self.mod.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        self.u, self.best, self.log = st["u"], st["best"], st["log"]
        self.gen.bit_generator.state = st["gen"]

    def check(self):
        ev = evaluate(self.torch, self.mod, self.D["val"])
        tr = evaluate(self.torch, self.mod, self.D["train"])
        row = dict(u=self.u, val=ev, train_ce=tr["ce"])
        self.log.append(row)
        if ev["ce"] is not None and (self.best is None or
                                     ev["ce"] < self.best["ce"]):
            self.best = dict(u=self.u, ce=ev["ce"],
                             model=copy.deepcopy(self.mod.state_dict()),
                             hash=weights_hash(self.mod))
        return row

    def step(self):
        torch = self.torch
        idx = self.gen.choice(self.pool, size=CFG["batch"], replace=True)
        X, y = self.D["train"]["X"][idx], self.D["train"]["y"][idx]
        self.mod.train()
        per, has, _lg = group_ce(torch, self.mod, X, y)
        loss = per[has].mean()
        if not bool(torch.isfinite(loss)):
            raise SystemExit("CE не конечна")
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mod.parameters(), CFG["clip"])
        self.opt.step()
        self.mod.eval()
        self.u += 1
        return float(loss.detach())

    def run(self, stop_at=None, verbose=True):
        if self.u == 0 and not self.log:
            r = self.check()
            self._save()
            if verbose:
                print(f"    s{self.seed} обн. 0: val CE {r['val']['ce']}",
                      flush=True)
        end = CFG["updates"] if stop_at is None else min(stop_at,
                                                         CFG["updates"])
        while self.u < end:
            self.step()
            if self.u % CFG["check_every"] == 0:
                r = self.check()
                if verbose:
                    v = r["val"]
                    print(f"    s{self.seed} обн. {self.u}: val CE "
                          f"{v['ce']:.4f}, train CE {r['train_ce']:.4f}, "
                          f"вмешательства tree/flat "
                          f"{v['would_intervene']['tree']}/"
                          f"{v['would_intervene']['flat']}; best "
                          f"{self.best['u']}", flush=True)
            if self.u % CFG["save_every"] == 0:
                self._save()
        return self.best


def fitcheck(torch, D, seed):
    """-> dict(ok, ce0, ce1, finite, applied, classes)."""
    y = D["train"]["y"]
    m = (y >= 0)
    m[:, 0] = False
    cand = np.argwhere(m)
    by = {c: [tuple(x) for x in cand if y[x[0], x[1]] == c] for c in range(3)}
    pick, k = [], 0
    while len(pick) < min(CFG["fit_n"], len(cand)):
        for c in range(3):
            if k < len(by[c]) and len(pick) < CFG["fit_n"]:
                pick.append(by[c][k])
        k += 1
        if k > len(cand):
            break
    Xf = np.stack([D["train"]["X"][i, j] for i, j in pick])
    yf = np.array([y[i, j] for i, j in pick])
    mod = make_model(torch, seed)
    h0 = weights_hash(mod)
    opt = torch.optim.AdamW(mod.parameters(), lr=CFG["fit_lr"],
                            weight_decay=CFG["wd"])
    Xt, yt = torch.as_tensor(Xf), torch.as_tensor(yf)
    ce = torch.nn.functional.cross_entropy
    with torch.no_grad():
        c0 = float(ce(mod(Xt), yt))
    finite = True
    for _ in range(CFG["fit_steps"]):
        loss = ce(mod(Xt), yt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), CFG["clip"])
        opt.step()
        finite &= bool(torch.isfinite(loss))
    with torch.no_grad():
        c1 = float(ce(mod(Xt), yt))
    cls = {kdat.CLASSES[c]: int((yf == c).sum()) for c in range(3)}
    applied = weights_hash(mod) != h0
    p = np.bincount(yf, minlength=3) / len(yf)
    base = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return dict(ok=bool(finite and applied and c1 <= c0 / 2), ce0=c0, ce1=c1,
                finite=finite, applied=applied, classes=cls, n=len(yf),
                baseline_ce=base)


class Selector:
    """Рантайм: PCA-контекст, стандартизация, MLP (float32, CPU)."""

    def __init__(self, torch, path):
        st = torch.load(path, map_location="cpu", weights_only=False)
        self.torch, self.meta = torch, st["meta"]
        self.mod = make_model(torch, st["meta"]["seed"])
        self.mod.load_state_dict(st["model"])
        self.mod.eval()
        if weights_hash(self.mod) != st["meta"]["weights_hash"]:
            raise SystemExit("селектор: отпечаток весов ≠ записанному")
        self.pm, self.pp, self.std = st["pca_mean"], st["pca_path"], st["std"]

    def ctx(self, mean, pth):
        return np.concatenate([kdat.pca_apply(self.pm, mean),
                               kdat.pca_apply(self.pp, pth)])

    def scores(self, X):
        Xs = kdat.standardize(self.std, X)
        with self.torch.no_grad():
            lg = self.mod(self.torch.as_tensor(Xs))
            g = gain(self.torch, lg).numpy().astype(np.float64)
        g[0] = 0.0
        return lg.numpy(), g


def save_selector(torch, path, best, D, seed, extra):
    mod = make_model(torch, seed)
    mod.load_state_dict(best["model"])
    meta = dict(seed=int(seed), update=best["u"], val_ce=best["ce"],
                weights_hash=weights_hash(mod), dataset_sha=D["sha"],
                cfg=CFG, **extra)
    st = dict(model=mod.state_dict(), pca_mean=D["pca_mean"],
              pca_path=D["pca_path"], std=D["std"], meta=meta)
    torch.save(st, path + ".tmp")
    os.replace(path + ".tmp", path)
    return meta


def selftest():
    import tempfile
    import torch
    assert sum(p.numel() for p in make_model(torch, 0).parameters()) == \
        N_PARAMS == 11779
    ts = torch.random.get_rng_state().clone()
    a, b = make_model(torch, 0), make_model(torch, 0)
    assert weights_hash(a) == weights_hash(b) != weights_hash(
        make_model(torch, 1))
    assert torch.equal(torch.random.get_rng_state(), ts)
    with torch.no_grad():
        g0 = gain(torch, a(torch.zeros(7, kc.N_FEAT)))
    assert torch.equal(g0, torch.zeros(7))            # нулевой выход

    def part(n, s):
        g = np.random.default_rng(s)
        X = g.normal(size=(n, 7, kc.N_FEAT)).astype(np.float32)
        y = np.full((n, 7), -1, np.int64)
        al = np.ones((n, 7), bool)
        for i in range(n):
            for j in range(1, 7):
                y[i, j] = 2 if X[i, j, 0] > 0.8 else (0 if X[i, j, 0] < -0.8
                                                      else 1)
        return dict(X=X, y=y, allowed=al, keys=[(0, i) for i in range(n)])
    D = dict(train=part(60, 1), val=part(15, 2), sha="toy",
             pca_mean=None, pca_path=None, std=None)
    fc = fitcheck(torch, D, 0)
    assert fc["ok"] and fc["finite"] and fc["applied"], fc
    # непрерывное обучение = обучение с остановкой и возобновлением
    orig = dict(CFG)
    CFG.update(updates=120)
    try:
        with tempfile.TemporaryDirectory() as td:
            t1 = Trainer(torch, D, 0, os.path.join(td, "a"))
            b1 = t1.run(verbose=False)
            t2 = Trainer(torch, D, 0, os.path.join(td, "b"))
            t2.run(stop_at=50, verbose=False)
            t3 = Trainer(torch, D, 0, os.path.join(td, "b"))
            assert t3.u == 50
            b3 = t3.run(verbose=False)
            assert weights_hash(t1.mod) == weights_hash(t3.mod)
            assert b1["u"] == b3["u"] and b1["hash"] == b3["hash"]
            assert [r["u"] for r in t3.log] == [0, 50, 100]
            # другой набор -> отказ возобновления
            try:
                Trainer(torch, dict(D, sha="other"), 0, os.path.join(td, "b"))
                raise AssertionError("чужой набор принят при возобновлении")
            except SystemExit:
                pass
            # изменение меток меняет обновление
            D2 = dict(D, train=dict(D["train"], y=np.where(
                D["train"]["y"] == 2, 0, D["train"]["y"])), sha="toy2")
            t4 = Trainer(torch, D2, 0, os.path.join(td, "c"))
            t4.run(stop_at=50, verbose=False)
            t5 = Trainer(torch, D, 0, os.path.join(td, "d"))
            t5.run(stop_at=50, verbose=False)
            assert weights_hash(t4.mod) != weights_hash(t5.mod)
            # рантайм-селектор: noop = 0, сохранённые веса = лучшим
            p = os.path.join(td, "sel.pt")
            st = dict(mean=np.zeros(kc.N_FEAT), std=np.ones(kc.N_FEAT))
            Dp = dict(D, pca_mean=dict(mean=np.zeros(6), comp=np.eye(16, 6)),
                      pca_path=dict(mean=np.zeros(6), comp=np.eye(16, 6)),
                      std=st)
            meta = save_selector(torch, p, b1, Dp, 0, {})
            S = Selector(torch, p)
            lg, sc = S.scores(D["val"]["X"][0])
            assert sc[0] == 0.0 and lg.shape == (7, 3)
            assert meta["weights_hash"] == b1["hash"]
            assert S.ctx(np.ones(6), np.ones(6)).shape == (32,)
    finally:
        CFG.clear()
        CFG.update(orig)
    # данные без меток rescue/harm: fitcheck честно сообщает классы
    Dn = dict(D, train=dict(D["train"], y=np.where(D["train"]["y"] >= 0, 1,
                                                   -1)))
    fn = fitcheck(torch, Dn, 0)
    assert fn["classes"] == {"harm": 0, "neutral": fn["n"], "rescue": 0}
    print("самопроверка k31_train пройдена: 11 779 параметров, локальная "
          "инициализация (нулевой последний слой, gain 0), fitcheck, "
          "возобновление = непрерывному обучению (веса, best, проверки), "
          "чужой набор — отказ, метки меняют обновление, рантайм-селектор "
          "(noop = 0, веса = best)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
