#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: fitcheck, warmup и основное обучение flat24 / tree_adapt24
(tree_fixed24 — тем же кодом), выбор чекпойнта по val_sel, возобновление.

РЕЖИМ (раздел 8 плана):
  warmup  1000 обновлений; обучаются только новые модули (query, проекция
          состояния, головы границ, расширения, readout), AdamW lr 3e−4;
  main    отдельный AdamW: эксперт 2e−5, LoRA 1e−5, новые 1e−4; линейный
          разогрев 200, косинус до 0.1·lr на объявленном максимуме 16 000;
          первоначально до 8 000; проверка каждые 500 (включая 0).
  Добор до 16 000 (обе модели), если хотя бы у одной лучшая проверка среди
  последних четырёх до 8000 и их минимум улучшился ≥ 1% к минимуму
  предыдущих четырёх. Чекпойнт — минимум взвешенной L на val_sel (при
  равенстве — более ранний).
  Общие: эффективный батч 32 (microbatch × накопление), wd 0.01, clip 1.0,
  сид 0; источники: 3 экспертных обновления : 1 учительское; расписание
  индексов заранее, отдельный генератор (одинаково у вариантов).
ВЕСА СТРОК: задачи равны -> эпизоды равны -> строки внутри эпизода равны
  (и в сэмплере, и в проверке); обычные средние — дополнительно.
ПОТЕРЯ: L = L1 руки (действительные позиции) + 0.1·BCE схвата.
FITCHECK: копия tree_adapt24 на 128 фиксированных train-наблюдениях (есть
  смены схвата и повороты), до 500 обновлений; порог: L1 руки снизилась
  ≥ 4 раз от стартовой, ниже постоянного прогноза, схват ≥ 95%. Затем
  модели создаются заново из исходного чекпойнта (сверяется отпечаток
  начальных весов).
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SEED = 0
EFF_BATCH = 32
WARM_UPDATES, WARM_LR = 1000, 3e-4
MAIN_UPDATES, MAIN_MAX = 8000, 16000
LR = dict(expert=2e-5, lora=1e-5, new=1e-4)
MAIN_WARM, LR_FLOOR = 200, 0.1
WD, CLIP = 0.01, 1.0
EVAL_EVERY = 500
SAVE_EVERY = 250
TEACHER_EVERY = 4                 # каждое 4-е обновление — учительское
PANEL_N = 256
FIT_N, FIT_UPDATES = 128, 500
FIT_RATIO, FIT_GRIP = 4.0, 0.95
OUTDIR = "data/k21/runs"


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def row_weights(task, episode):
    task, episode = np.asarray(task), np.asarray(episode)
    w = np.zeros(len(task))
    for t in np.unique(task):
        mt = task == t
        eps = np.unique(episode[mt])
        for e in eps:
            me = mt & (episode == e)
            w[me] = 1.0 / (len(eps) * me.sum())
    return w * len(w) / max(w.sum(), 1e-300)


def schedule(D, n_updates, micro, seed=SEED, start=0):
    """Заранее заданная очередь: для обновления u — источник и индексы
    microbatch (эксперт: строки train по весам; учитель: индексы набора)."""
    tr = np.asarray(D["train_rows"])
    if "_p_train" not in D:
        w = row_weights(D["task"][tr], D["episode"][tr])
        D["_p_train"] = w / w.sum()
    p = D["_p_train"]
    acc = EFF_BATCH // micro
    out = []
    for u in range(start, n_updates):
        g = np.random.default_rng([seed, 7, u])
        if u % TEACHER_EVERY == TEACHER_EVERY - 1:
            idx = g.integers(0, len(D["teacher_rows"]), EFF_BATCH)
            out.append(("teacher", idx.reshape(acc, micro)))
        else:
            idx = g.choice(len(tr), EFF_BATCH, replace=True, p=p)
            out.append(("expert", tr[idx].reshape(acc, micro)))
    return out


def lr_main(u, base):
    if u < MAIN_WARM:
        return base * (u + 1) / MAIN_WARM
    t = (u - MAIN_WARM) / max(MAIN_MAX - MAIN_WARM, 1)
    return base * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(
        math.pi * min(t, 1.0))))


def extension_rule(curves):
    """curves: {вариант: [(шаг, val_L)]} до 8000 -> продолжать ли обе."""
    for c in curves.values():
        c = [x for x in c if x[0] <= MAIN_UPDATES]
        if len(c) < 8:
            continue
        vals = [v for _s, v in c]
        best_i = int(np.argmin(vals))
        last4, prev4 = vals[-4:], vals[-8:-4]
        if best_i >= len(vals) - 4 and min(last4) <= 0.99 * min(prev4):
            return True
    return False


# --- модель и шаги -----------------------------------------------------------------
def make_model(torch, mode, D, dev, pin, checkpoint=True):
    import k21_data as d21
    import k21_model as km
    bar = d21.load_bar(dev, pin)
    m = bar["model"]
    m.action_expert.float()
    M = km.build(torch)
    mod = M["K21"](m, mode, D["L"], D["U"], D["state_mean"], D["state_std"],
                   checkpoint=checkpoint, seed=SEED).to(dev)
    return mod, bar


def trainable_state(mod):
    """Только обучаемые и новые тензоры (без замороженного BAR)."""
    keep = {}
    for n, p in mod.named_parameters():
        if p.requires_grad or not n.startswith("bar.") or \
                n.endswith((".A", ".B")):
            keep[n] = p.detach().cpu().clone()
    for n, b in mod.named_buffers():
        if not n.startswith("bar."):
            keep[n] = b.detach().cpu().clone()
    for n, p in mod.bar.action_expert.named_parameters():
        keep[f"bar.action_expert.{n}"] = p.detach().cpu().clone()
    return keep


def state_hash(st):
    h = hashlib.sha1()
    for k in sorted(st):
        h.update(k.encode())
        h.update(np.ascontiguousarray(st[k].float().numpy()).tobytes())
    return h.hexdigest()


def set_phase(mod, phase):
    """warmup — только новые модули; main — эксперт, LoRA, новые."""
    gr = mod.groups_all
    for k, v in gr.items():
        on = (k == "new") or phase == "main"
        for p in v:
            p.requires_grad_(on)


def make_opt(torch, mod, phase):
    gr = mod.groups_all
    if phase == "warmup":
        return torch.optim.AdamW([dict(params=gr["new"], lr=WARM_LR,
                                       name="new", base=WARM_LR)],
                                 weight_decay=WD)
    return torch.optim.AdamW([dict(params=gr[k], lr=LR[k], name=k,
                                   base=LR[k]) for k in ("expert", "lora",
                                                         "new")],
                             weight_decay=WD)


def run_micro(torch, km, mod, bar, D, rows, src, dev, dt):
    import k21_data as d21
    if src == "teacher":
        tr_rows = D["teacher_rows"][rows]
        batch, po, st, tgt, val = d21.make_batch(D, tr_rows, bar, dev,
                                                 teacher_idx=rows)
    else:
        batch, po, st, tgt, val = d21.make_batch(D, rows, bar, dev)
    with torch.autocast("cuda", dtype=dt, enabled=dev.type == "cuda"):
        out = mod(batch, po, st)
        loss, l1, bce = km.loss_fn(torch, out, tgt, val)
    return out, loss, l1, bce, tgt, val


def evaluate(torch, km, mod, bar, D, rows, dev, dt, micro, weights=None,
             tree_diag=False):
    """Метрики на строках (эксперт): L, L1 руки 20/8, схват; взвешенные и
    обычные, по задачам; насыщение; для дерева — интервалы и массы."""
    mod.eval()
    rows = np.asarray(rows)
    L_, l1s, l18, gacc, sat, roots, leafm = [], [], [], [], [], [], []
    Lt = torch.as_tensor(D["L"], device=dev)
    Ut = torch.as_tensor(D["U"], device=dev)
    with torch.no_grad():
        for s0 in range(0, len(rows), micro):
            r = rows[s0:s0 + micro]
            out, loss, l1, bce, tgt, val = run_micro(torch, km, mod, bar, D, r,
                                                     "expert", dev, dt)
            if not (torch.isfinite(loss).all() and torch.isfinite(
                    out["arm"]).all()):
                raise SystemExit("нечисловая проверка")
            v = val.float()
            e = (out["arm"] - tgt[..., :6]).abs().mean(-1)
            L_.append(loss.float().cpu().numpy())
            l1s.append(((e * v).sum(1) / v.sum(1)).cpu().numpy())
            v8 = v[:, :8]
            l18.append(((e[:, :8] * v8).sum(1) / v8.sum(1).clamp_min(1))
                       .cpu().numpy())
            g = ((out["grip_logit"] > 0) == (tgt[..., 6] > 0)).float()
            gacc.append(((g * v).sum(1) / v.sum(1)).cpu().numpy())
            near = ((out["arm"] - Lt).abs() < 0.01 * (Ut - Lt)) | \
                ((Ut - out["arm"]).abs() < 0.01 * (Ut - Lt))
            sat.append(near.float().mean((1, 2)).cpu().numpy())
            if tree_diag and "root" in out:
                roots.append((out["root"][..., 1] - out["root"][..., 0])
                             .cpu().numpy())
                leafm.append(out["weights"].sum(1).cpu().numpy())
    w = np.ones(len(rows)) if weights is None else np.asarray(weights)

    def wm(x):
        x = np.concatenate(x)
        return float((x * w).sum() / w.sum()), float(x.mean())
    res = dict(L=wm(L_), l1_arm_20=wm(l1s), l1_arm_8=wm(l18),
               grip_acc=wm(gacc), saturation=wm(sat))
    task = D["task"][rows]
    Lc = np.concatenate(L_)
    res["per_task_L"] = {str(t): float(Lc[task == t].mean())
                         for t in np.unique(task)}
    if roots:
        R = np.concatenate(roots)
        LM = np.concatenate(leafm)
        res["tree"] = dict(
            root_durations_q=np.percentile(R, [5, 25, 50, 75, 95], 0)
            .tolist(), root_dev_from_uniform=float(np.abs(R - 5).mean()),
            leaf_mass_mean=LM.mean(0).tolist(),
            leaves_mass_lt_0_1=float((LM < 0.1).sum(1).mean()))
    mod.train()
    return res


def grad_norms(mod):
    out = {}
    for k, v in mod.groups_all.items():
        gs = [p.grad.float() for p in v if p.grad is not None]
        out[k] = float(np.sqrt(sum(float((g ** 2).sum()) for g in gs))) \
            if gs else 0.0
    ae = mod.bar.action_expert.layers
    for i in (0, 23):
        g = ae[i].self_attn.q_proj.weight.grad
        out[f"expert_layer{i + 1}"] = float(g.norm()) if g is not None else 0.0
    if hasattr(mod, "dur"):
        for nm in ("dur", "split1", "split2", "exp1", "exp2"):
            gs = [p.grad.float() for p in getattr(mod, nm).parameters()
                  if p.grad is not None]
            out[nm] = float(np.sqrt(sum(float((g ** 2).sum()) for g in gs))) \
                if gs else 0.0
    return out


class Trainer:
    def __init__(self, torch, a, D, mode, tag):
        import k21_model as km
        self.torch, self.km, self.a, self.D = torch, km, a, D
        self.mode, self.tag = mode, tag
        self.dev = torch.device(a.device)
        self.dt = getattr(torch, a.dtype)
        self.dir = os.path.join(OUTDIR, tag)
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, "resume.pt")
        self.mod, self.bar = make_model(torch, mode, D, self.dev,
                                        D["prep_report"]["bar_pin"])
        g = self.mod.groups()
        self.mod.groups_all = g
        self.init_hash = state_hash(trainable_state(self.mod))
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.dt == torch.float16 and self.dev.type == "cuda"))
        vr = np.asarray(D["val_rows"])
        self.val_rows = vr
        self.val_w = row_weights(D["task"][vr], D["episode"][vr])
        tr = np.asarray(D["train_rows"])
        rg = np.random.default_rng(SEED + 5)
        pw = row_weights(D["task"][tr], D["episode"][tr])
        self.panel = np.sort(rg.choice(tr, PANEL_N, replace=False,
                                       p=pw / pw.sum()))
        self.st = dict(phase="warmup", u=0, curve=[], best=None,
                       limit=MAIN_UPDATES, init_hash=self.init_hash,
                       acc={})

    # --- состояние -------------------------------------------------------------
    def save(self, opt):
        torch = self.torch
        st = dict(self.st, weights=trainable_state(self.mod),
                  opt=opt.state_dict(), scaler=self.scaler.state_dict(),
                  rng=torch.random.get_rng_state(),
                  config=self.config())
        torch.save(st, self.state_path + ".tmp")
        os.replace(self.state_path + ".tmp", self.state_path)

    def config(self):
        return dict(mode=self.mode, micro=self.a.micro, dtype=self.a.dtype,
                    prep_id=self.D["prep_report"]["prep_id"], seed=SEED,
                    eff_batch=EFF_BATCH, lr=LR, warm=WARM_UPDATES,
                    main_max=MAIN_MAX)

    def load(self):
        torch = self.torch
        if not os.path.exists(self.state_path):
            return None
        st = torch.load(self.state_path, map_location="cpu",
                        weights_only=False)
        if st["config"] != self.config():
            raise SystemExit(f"{self.state_path}: другая конфигурация")
        if st["init_hash"] != self.init_hash:
            raise SystemExit("начальные веса ≠ закреплённым")
        own = dict(self.mod.named_parameters())
        own.update(dict(self.mod.named_buffers()))
        with torch.no_grad():
            for k, v in st["weights"].items():
                own[k].copy_(v.to(own[k].device, own[k].dtype))
        torch.random.set_rng_state(st["rng"])
        self.scaler.load_state_dict(st["scaler"])
        opt_state = st.pop("opt")
        for k in ("weights", "scaler", "rng", "config"):
            st.pop(k)
        self.st = st
        print(f"  {self.tag}: продолжение — фаза {st['phase']}, обновление "
              f"{st['u']}", flush=True)
        return opt_state

    # --- проверка -----------------------------------------------------------
    def check(self, opt, u):
        a = self.a
        t0 = time.time()
        val = evaluate(self.torch, self.km, self.mod, self.bar, self.D,
                       self.val_rows, self.dev, self.dt, a.eval_micro,
                       self.val_w, tree_diag=self.mode != "flat24")
        pan = evaluate(self.torch, self.km, self.mod, self.bar, self.D,
                       self.panel, self.dev, self.dt, a.eval_micro)
        tr = {k: {kk: vv / v["n"] for kk, vv in v.items() if kk != "n"}
              for k, v in self.st["acc"].items() if v.get("n")}
        self.st["acc"] = {}
        rec = dict(update=u, val=val, panel=pan, train=tr,
                   grad=self.st.get("last_grad"), seconds=time.time() - t0)
        self.st["curve"].append(rec)
        v = val["L"][0]
        if self.st["best"] is None or v < self.st["best"][1]:
            self.st["best"] = (u, v)
            st = trainable_state(self.mod)
            p = os.path.join(self.dir, "best.pt")
            self.torch.save(dict(weights=st, update=u, val_L=v,
                                 config=self.config(),
                                 init_hash=self.init_hash,
                                 weights_hash=state_hash(st)), p + ".tmp")
            os.replace(p + ".tmp", p)
        print(f"  {self.tag} main {u}: val L {v:.4f} (рука-20 "
              f"{val['l1_arm_20'][0]:.4f}, -8 {val['l1_arm_8'][0]:.4f}, "
              f"схват {100 * val['grip_acc'][0]:.1f}%), панель train L "
              f"{pan['L'][0]:.4f}; лучшая {self.st['best']}; проверка "
              f"{rec['seconds']:.0f} с", flush=True)

    # --- обучение -------------------------------------------------------------
    def run(self, until_main):
        torch = self.torch
        opt_state = self.load()
        phase = self.st["phase"]
        set_phase(self.mod, phase)
        opt = make_opt(torch, self.mod, phase)
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        self.st["limit"] = max(self.st.get("limit", MAIN_UPDATES), until_main)
        t_start, n_seen = time.time(), 0
        while True:
            phase = self.st["phase"]
            if phase == "warmup" and self.st["u"] >= WARM_UPDATES:
                self.st.update(phase="main", u=0)
                set_phase(self.mod, "main")
                opt = make_opt(torch, self.mod, "main")
                print(f"  {self.tag}: warmup завершён -> основной режим "
                      f"(новый оптимизатор)", flush=True)
                self.check(opt, 0)
                self.save(opt)
                continue
            if phase == "main" and self.st["u"] >= until_main:
                break
            u = self.st["u"]
            sched = schedule(self.D, u + 1, self.a.micro, start=u)[0]
            if phase == "main":
                for gr in opt.param_groups:
                    gr["lr"] = lr_main(u, gr["base"])
                sched = schedule(self.D, WARM_UPDATES + u + 1, self.a.micro,
                                 start=WARM_UPDATES + u)[0]
            src, idx = sched
            opt.zero_grad(set_to_none=True)
            accd = self.st["acc"].setdefault(src, {})
            for rows in idx:
                out, loss, l1, bce, tgt, val = run_micro(
                    torch, self.km, self.mod, self.bar, self.D, rows, src,
                    self.dev, self.dt)
                lm = loss.mean() / len(idx)
                if not bool(torch.isfinite(lm)):
                    raise SystemExit(f"{self.tag}: нечисловая потеря")
                self.scaler.scale(lm).backward()
                for k, x in (("L", loss), ("l1", l1), ("bce", bce)):
                    accd[k] = accd.get(k, 0.0) + float(x.mean()) / len(idx)
                n_seen += len(rows)
            accd["n"] = accd.get("n", 0) + 1
            self.scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], CLIP)
            if not bool(torch.isfinite(gn)) and self.scaler.is_enabled() \
                    is False:
                raise SystemExit(f"{self.tag}: нечисловой градиент")
            if (u + 1) % EVAL_EVERY == 0 or u < 2:
                self.st["last_grad"] = dict(grad_norms(self.mod),
                                            total=float(gn))
            self.scaler.step(opt)
            self.scaler.update()
            self.st["u"] = u + 1
            if self.st["u"] == 100 and phase == "main" or \
                    (self.st["u"] == 100 and phase == "warmup"):
                el = time.time() - t_start
                mem = torch.cuda.max_memory_allocated(self.dev) / 2 ** 30 \
                    if self.dev.type == "cuda" else 0
                rate = n_seen / max(el, 1e-9)
                left = (WARM_UPDATES - self.st["u"] if phase == "warmup"
                        else 0) + until_main - (self.st["u"] if phase ==
                                                "main" else 0)
                print(f"  {self.tag} ({phase}): {rate:.1f} набл./с, память "
                      f"{mem:.1f} ГБ; до {until_main} основных обновлений ~"
                      f"{left * EFF_BATCH / rate / 3600:.1f} ч (без проверок)",
                      flush=True)
            if phase == "main" and self.st["u"] % EVAL_EVERY == 0:
                self.check(opt, self.st["u"])
            if self.st["u"] % SAVE_EVERY == 0:
                self.save(opt)
        self.save(opt)
        return self.st


# --- fitcheck ----------------------------------------------------------------------
def fit_rows(D, n=FIT_N, seed=SEED + 9):
    """128 train-наблюдений: со сменой схвата, с поворотами, прочие."""
    tr = np.asarray(D["train_rows"])
    act = D["act"][tr]
    sw = (np.sign(act[:, 1:, 6]) != np.sign(act[:, :-1, 6])).any(1)
    rot = np.abs(act[..., 3:6]).max((1, 2))
    rng = np.random.default_rng(seed)
    a = rng.choice(tr[sw], min(n // 4, sw.sum()), replace=False)
    b = rng.choice(tr[np.argsort(-rot)[:4000]], n // 4, replace=False)
    rest = np.setdiff1d(tr, np.r_[a, b])
    c = rng.choice(rest, n - len(a) - len(b), replace=False)
    return np.sort(np.unique(np.r_[a, b, c]))[:n]


def fitcheck(torch, a, D):
    import k21_model as km
    rows = fit_rows(D)
    T = Trainer(torch, a, D, "tree_adapt24", "fitcheck")
    set_phase(T.mod, "main")
    opt = make_opt(torch, T.mod, "main")
    for g in opt.param_groups:
        g["lr"] = g["base"] * 3                    # быстрая подгонка
    dev, dt = T.dev, T.dt
    act = D["act"][rows]
    const = float(np.abs(act[..., :6] - act[..., :6].mean(0)).mean())
    ev0 = evaluate(torch, km, T.mod, T.bar, D, rows, dev, dt, a.eval_micro)
    curve = [dict(update=0, l1=ev0["l1_arm_20"][1], grip=ev0["grip_acc"][1])]
    rng = np.random.default_rng(SEED + 3)
    for u in range(1, FIT_UPDATES + 1):
        idx = rng.choice(len(rows), EFF_BATCH, replace=False).reshape(-1,
                                                                      a.micro)
        opt.zero_grad(set_to_none=True)
        for r in idx:
            out, loss, l1, bce, tgt, val = run_micro(torch, km, T.mod, T.bar,
                                                     D, rows[r], "expert", dev,
                                                     dt)
            T.scaler.scale(loss.mean() / len(idx)).backward()
        T.scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], CLIP)
        T.scaler.step(opt)
        T.scaler.update()
        if u % 100 == 0:
            ev = evaluate(torch, km, T.mod, T.bar, D, rows, dev, dt,
                          a.eval_micro)
            curve.append(dict(update=u, l1=ev["l1_arm_20"][1],
                              grip=ev["grip_acc"][1]))
            print(f"  fitcheck {u}: L1 руки {ev['l1_arm_20'][1]:.4f} (старт "
                  f"{curve[0]['l1']:.4f}, постоянный {const:.4f}), схват "
                  f"{100 * ev['grip_acc'][1]:.1f}%", flush=True)
    fin = curve[-1]
    ok = (fin["l1"] <= curve[0]["l1"] / FIT_RATIO and fin["l1"] < const
          and fin["grip"] >= FIT_GRIP)
    rep = dict(rows=rows.tolist(), curve=curve, constant_l1=const, ok=ok,
               criteria=dict(ratio=FIT_RATIO, grip=FIT_GRIP))
    os.makedirs("reports/k21", exist_ok=True)
    json.dump(rep, open("reports/k21/fitcheck.json", "w"), indent=1)
    init_hash = T.init_hash
    del T, opt
    torch.cuda.empty_cache()
    # модели создаются заново: начальные веса совпадают с исходными
    T2 = Trainer(torch, a, D, "tree_adapt24", "fitcheck_reinit")
    same = T2.init_hash == init_hash
    del T2
    torch.cuda.empty_cache()
    print(f"  fitcheck: {'ПРОЙДЕН' if ok else 'НЕ ПРОЙДЕН'}; начальные веса "
          f"после пересоздания {'совпадают' if same else 'НЕ совпадают'}",
          flush=True)
    return 0 if (ok and same) else 3


def main_train(torch, a, D):
    modes = a.modes
    trainers = {}
    for m in modes:
        tr = Trainer(torch, a, D, m, m)
        tr.run(MAIN_UPDATES)
        trainers[m] = [(c["update"], c["val"]["L"][0])
                       for c in tr.st["curve"]]
        del tr
        torch.cuda.empty_cache()
    ext = extension_rule(trainers)
    print(f"  правило добора: {'продолжить обе до 16000' if ext else 'стоп на 8000'}",
          flush=True)
    if ext:
        for m in modes:
            tr = Trainer(torch, a, D, m, m)
            tr.run(MAIN_MAX)
            del tr
            torch.cuda.empty_cache()
    rep = {}
    for m in modes:
        st = torch.load(os.path.join(OUTDIR, m, "resume.pt"),
                        map_location="cpu", weights_only=False)
        b = torch.load(os.path.join(OUTDIR, m, "best.pt"),
                       map_location="cpu", weights_only=False)
        rep[m] = dict(best=st["best"], updates=st["u"], extension=ext,
                      best_is_last=st["best"][0] == st["u"],
                      best_sha1=sha(os.path.join(OUTDIR, m, "best.pt")),
                      best_weights_hash=b["weights_hash"],
                      curve=st["curve"], config=st["config"])
    json.dump(rep, open("reports/k21/train.json", "w"), ensure_ascii=False,
              indent=1, default=float)
    print("  отчёт: reports/k21/train.json")
    return 0


def selftest():
    D = dict(train_rows=np.arange(100), task=np.array(["a", "b"] * 60),
             episode=np.arange(120) // 6, teacher_rows=np.arange(0, 40, 2))
    s1 = schedule(D, 12, 8)
    s2 = schedule(D, 12, 8)
    assert all(a[0] == b[0] and np.array_equal(a[1], b[1])
               for a, b in zip(s1, s2))
    assert [x[0] for x in s1[:4]] == ["expert"] * 3 + ["teacher"]
    assert s1[0][1].shape == (4, 8)
    assert np.array_equal(schedule(D, 12, 8, start=5)[0][1], s1[5][1])
    assert np.array_equal(schedule(D, 12, 16)[0][1].reshape(-1),
                          s1[0][1].reshape(-1))
    # LR: разогрев, без скачка, пол 0.1
    assert lr_main(0, 1.0) < lr_main(199, 1.0) <= 1.0
    lrs = [lr_main(u, 1.0) for u in range(200, MAIN_MAX)]
    assert all(b <= a + 1e-12 for a, b in zip(lrs, lrs[1:]))
    assert abs(lr_main(MAIN_MAX, 1.0) - LR_FLOOR) < 1e-6
    # правило добора
    flat = [(500 * i, 1.0 - 0.01 * i) for i in range(17)]
    stall = [(500 * i, 1.0) for i in range(17)]
    assert extension_rule(dict(a=flat, b=stall))
    assert not extension_rule(dict(a=stall, b=stall))
    worse = [(500 * i, 1.0 + 0.01 * i) for i in range(17)]
    assert not extension_rule(dict(a=worse))
    w = row_weights(["a", "a", "a", "b"], [1, 1, 2, 3])
    assert np.allclose(w / w.sum(), [.125, .125, .25, .5])
    print("самопроверка k21_train пройдена: расписание индексов (повторяемо, "
          "3:1, не зависит от microbatch и возобновления), LR без скачка до "
          "пола 0.1, правило добора, веса строк")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-21: обучение")
    ap.add_argument("mode", choices=("fitcheck", "train", "selftest"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--eval-micro", type=int, default=8)
    ap.add_argument("--modes", nargs="+", default=["flat24", "tree_adapt24"])
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    if EFF_BATCH % a.micro:
        print("  ОТКАЗ: эффективный батч 32 не делится на microbatch")
        return 3
    import torch
    import k21_data as d21
    try:
        D = d21.load_prepared(require_teacher=True)
        if a.mode == "fitcheck":
            return fitcheck(torch, a, D)
        return main_train(torch, a, D)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
