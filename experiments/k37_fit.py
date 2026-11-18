#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-37: строки K-21, кеш замороженных признаков, обучение головки листа.

Строки (§4): только D["train_rows"] K-21 задач 0/6/8 (сопоставление по
закреплённым текстам LIBERO), 0 ≤ step < 600, step кратен 8, восемь
действительных целей. Эпизоды задачи делятся 80/20 по стабильному хешу
sha1("k37|<задача>|<эпизод>"); внутри части строки берутся по кругу по
эпизодам (порядок хеша), в эпизоде — по возрастанию step; до 192 строк
обучения и 64 диагностики на задачу. Одинаковы для обоих сидов.
Кеш (GPU, на сид): штатный T32 (k36_harness.make_arm) в обёртке T2 с
нулевой головкой — один экспорт на строку; elapsed = step. Для подходящих
вызовов сохраняются вход головки (96), Z0[:8], D_parent (до ограничителя),
half, mid, w, c_l, эксперт D["act"][row, :8, :6]. На первой подходящей
строке сида — сверка быстрой суммы с полной пересборкой листьев на рабочем
устройстве (|ΔD| ≤ 1e−5, |Δarm| ≤ 1e−6).
Обучение (CPU, на сид): Linear(96, 6), нули; AdamW lr 3e−3, wd 1e−4, batch
64 (последний неполный включён), 8 эпох, порядок — default_rng([37, seed,
эпоха]); последний checkpoint. Потеря: mean smooth_l1(beta 0.05) по
(arm − эксперт)/half на 8 шагах + 0.001·mean(tanh(u)²). Тензорный
ограничитель: clip ±0.08, rms = √max(mean D1², 1e−12), s = min(1, 0.04/rms).
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k32_operator as ko  # noqa: E402
import k37_operator as kop  # noqa: E402

TASKS = (0, 6, 8)
N_TRAIN, N_DIAG = 192, 64
CFG = dict(lr=3e-3, wd=1e-4, batch=64, epochs=8, beta=0.05, reg=1e-3,
           rms_floor=1e-12)
CHECK_TOL = dict(D=1e-5, arm=1e-6)


def ep_hash(t, e):
    return hashlib.sha1(f"k37|{int(t)}|{int(e)}".encode()).hexdigest()


def select_rows(D, task_text):
    """-> {задача: dict(train=[...], diag=[...])}."""
    tr = np.asarray(D["train_rows"])
    out = {}
    for t in TASKS:
        rows = [int(r) for r in tr
                if str(D["task"][r]) == task_text[str(t)] and
                0 <= int(D["step"][r]) < 600 and int(D["step"][r]) % 8 == 0
                and bool(np.asarray(D["valid"][r])[:8].all())]
        by_ep = {}
        for r in rows:
            by_ep.setdefault(int(D["episode"][r]), []).append(r)
        eps = sorted(by_ep, key=lambda e: ep_hash(t, e))
        n_tr = int(round(0.8 * len(eps)))
        parts = dict(train=eps[:n_tr], diag=eps[n_tr:])
        res = {}
        for nm, ee in parts.items():
            lim = N_TRAIN if nm == "train" else N_DIAG
            lists = [sorted(by_ep[e], key=lambda r: int(D["step"][r]))
                     for e in ee]
            sel, k = [], 0
            while len(sel) < lim and any(k < len(x) for x in lists):
                for x in lists:
                    if k < len(x) and len(sel) < lim:
                        sel.append(int(x[k]))
                k += 1
            res[nm] = sel
        if set(res["train"]) & set(res["diag"]):
            raise SystemExit(f"t{t}: строки train/diag пересекаются")
        out[str(t)] = dict(res, n_episodes=len(eps), n_train_eps=n_tr)
    return out


def rows_sha(sel):
    return hashlib.sha1(json.dumps(sel, sort_keys=True).encode()).hexdigest()


# --- тензорный ограничитель и потеря ------------------------------------------------------
def limit_t(torch, D, floor=CFG["rms_floor"]):
    D1 = torch.clamp(D, -ko.CLIP, ko.CLIP)
    ms = torch.clamp(torch.mean(D1 ** 2, dim=(-2, -1)), min=floor)
    s = torch.clamp(ko.RMS_CAP / torch.sqrt(ms), max=1.0)
    return D1 * s[..., None, None]


def batch_loss(torch, head, b, cfg=CFG):
    u = head(b["x"])
    th = torch.tanh(u)
    Dl = b["w"][:, :, None] / b["c"][:, None, None] * kop.LEAF_ALPHA * \
        th[:, None, :]
    Du = limit_t(torch, b["Dp"] + Dl)
    arm = b["mid"] + b["half"] * torch.tanh(b["Z0"] / b["half"] + Du)
    err = torch.nn.functional.smooth_l1_loss(
        (arm - b["tgt"]) / b["half"], torch.zeros_like(arm),
        beta=cfg["beta"], reduction="none")
    return err.mean() + cfg["reg"] * (th ** 2).mean(), err


def to_t(torch, z, idx=None):
    sl = slice(None) if idx is None else idx
    f = lambda k: torch.as_tensor(np.asarray(z[k])[sl], dtype=torch.float32)  # noqa: E731
    return dict(x=f("x"), w=f("w"), c=f("c"), Dp=f("Dp"), Z0=f("Z0"),
                tgt=f("tgt"), half=torch.as_tensor(np.asarray(z["half"],
                                                              np.float32)),
                mid=torch.as_tensor(np.asarray(z["mid"], np.float32)))


def train_head(torch, z, seed, cfg=CFG):
    """z: кеш (x, w, c, Dp, Z0, tgt, half, mid, split). -> (head, отчёт)."""
    head = kop.make_head(torch)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["wd"])
    tr = np.flatnonzero(np.asarray(z["split"]) == 0)
    dg = np.flatnonzero(np.asarray(z["split"]) == 1)
    if len(tr) == 0:
        raise SystemExit(f"s{seed}: нет подходящих обучающих строк")
    w0 = {k: v.detach().clone() for k, v in head.state_dict().items()}
    b0 = to_t(torch, z, tr[:min(64, len(tr))])
    loss0, _e = batch_loss(torch, head, b0, cfg)
    loss0.backward()
    g0 = float(head.weight.grad.abs().sum() + head.bias.grad.abs().sum())
    head.zero_grad(set_to_none=True)
    hist = []
    for ep in range(cfg["epochs"]):
        perm = np.random.default_rng([37, int(seed), ep]).permutation(tr)
        tot, n = 0.0, 0
        for s0 in range(0, len(perm), cfg["batch"]):
            b = to_t(torch, z, perm[s0:s0 + cfg["batch"]])
            loss, _e = batch_loss(torch, head, b, cfg)
            if not bool(torch.isfinite(loss)):
                raise SystemExit("K-37: нечисловая потеря головки")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(perm[s0:s0 + cfg["batch"]])
            n += len(perm[s0:s0 + cfg["batch"]])
        hist.append(dict(epoch=ep, loss=tot / n))

    def evaluate(idx, zero=False):
        if len(idx) == 0:
            return None
        hd = kop.make_head(torch) if zero else head
        with torch.no_grad():
            _l, err = batch_loss(torch, hd, to_t(torch, z, idx), cfg)
        return dict(all=float(err.mean()), first4=float(err[:, :4].mean()),
                    last4=float(err[:, 4:].mean()))
    rep = dict(n_train=int(len(tr)), n_diag=int(len(dg)), grad_init=g0,
               history=hist, train=evaluate(tr), diag=evaluate(dg),
               train_zero=evaluate(tr, True), diag_zero=evaluate(dg, True),
               weight_change=float(sum(float((v - w0[k]).abs().sum())
                                       for k, v in head.state_dict().items())))
    return head, rep


def head_hash(torch, head):
    h = hashlib.sha1()
    for k, v in sorted(head.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def selftest():
    import torch
    rng = np.random.default_rng(0)
    # тензорный ограничитель = K-32 (forward), включая нулевую матрицу
    for scale in (0.0, 0.01, 0.05, 0.3):
        D = (rng.normal(size=(8, 6)) * scale).astype(np.float32)
        a = limit_t(torch, torch.as_tensor(D)[None])[0].numpy()
        b = ko.limit(torch, D)[0].numpy()
        assert np.abs(a - b).max() <= 1e-7, (scale, np.abs(a - b).max())
    # синтетический кеш: обучение снижает потерю, ненулевой градиент при
    # нулевой инициализации, веса меняются
    n = 300
    half = np.full(6, 1.5, np.float32)
    mid = np.zeros(6, np.float32)
    w = np.abs(rng.normal(size=(n, 8))).astype(np.float32) * 0.3
    c = np.maximum(0.05, np.sqrt((w.astype(np.float64) ** 2).mean(1))
                   ).astype(np.float32)
    x = rng.normal(size=(n, 96)).astype(np.float32)
    Z0 = (rng.normal(size=(n, 8, 6)) * 0.2).astype(np.float32)
    Dp = (rng.normal(size=(n, 8, 6)) * 0.005).astype(np.float32)
    true_u = x[:, :6] * 0.8
    with torch.no_grad():
        Dl = w[:, :, None] / c[:, None, None] * 0.025 * np.tanh(true_u)[
            :, None, :]
        Du = limit_t(torch, torch.as_tensor(Dp + Dl)).numpy()
    tgt = (mid + half * np.tanh(Z0 / half + Du)).astype(np.float32)
    z = dict(x=x, w=w, c=c, Dp=Dp, Z0=Z0, tgt=tgt, half=half, mid=mid,
             split=np.r_[np.zeros(240, int), np.ones(60, int)])
    head, rep = train_head(torch, z, 0, dict(CFG, epochs=30))
    assert rep["grad_init"] > 0 and rep["weight_change"] > 0
    assert rep["diag"]["all"] < rep["diag_zero"]["all"], rep
    # строки: детерминизм, без пересечений, ограничения step/valid
    N = 4000
    Dd = dict(train_rows=np.arange(N), task=np.array(
        ["a", "b", "c", "d"] * (N // 4)), step=np.arange(N) % 80 * 4,
        episode=np.arange(N) // 40, valid=np.ones((N, 20), bool))
    tt = {"0": "a", "6": "b", "8": "c"}
    s1, s2 = select_rows(Dd, tt), select_rows(Dd, tt)
    assert s1 == s2 and rows_sha(s1) == rows_sha(s2)
    for t in ("0", "6", "8"):
        allr = s1[t]["train"] + s1[t]["diag"]
        assert all(Dd["step"][r] % 8 == 0 for r in allr)
        assert len(s1[t]["train"]) <= N_TRAIN and len(s1[t]["diag"]) <= N_DIAG
    print("самопроверка k37_fit пройдена: тензорный ограничитель = K-32 "
          "(в т. ч. ноль); при нулевой инициализации градиент ненулевой, "
          "обучение снижает диагностическую потерю ниже нулевой головки; "
          "отбор строк детерминирован, шаг кратен 8, лимиты")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; запуск — через k37_parent_leaf.py")
