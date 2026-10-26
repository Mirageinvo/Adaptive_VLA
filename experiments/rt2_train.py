#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, baseline: тренер flat24 (и будущих архитектур через фабрику).

Закреплённый протокол (общий для baseline и будущего дерева):
  - данные — подготовка rt1 (8 задач, 45/5 эпизодов, prep_id-контракт),
    только ЭКСПЕРТНЫЕ обновления (учителя на 14 каналах нет);
  - выборка: задача -> эпизод -> строка, равные веса на каждом уровне; на
    обновление — ОДНА общая очередь из 32 строк (seed, обновление), затем
    разбиение на micro (накопление 32/micro); seed передаётся явно;
  - bf16 autocast, обучаемые параметры float32, без GradScaler,
    checkpointing; TF32 выключен явно (записан в контракте);
  - warmup 1000 обновлений — только группа new (state_proj, readout, query
    и прочие новые модули), AdamW lr 3e-4; main ровно 8000 — эксперт 2e-5,
    LoRA 1e-5, new 1e-4, разогрев 200, косинус с горизонтом 8000 до 0.1;
    WD 0.01, clip 1.0; без автоматического добора;
  - потеря — L1 12 суставов + 1.0·L1 двух схватов по действительным
    позициям; проверка каждые 500 (взвешенная и обычная потеря val, MAE руки
    и схватов на 20 и первых 8 шагах, насыщение, по задачам, train-панель
    256 строк), атомарное сохранение каждые 250; best — минимум взвешенной
    val-потери, при равенстве — ранний;
  - фабрика модели задаётся явно (--factory rt1_model:flat24); контракт
    включает только её зависимости, а не все файлы каталога.
Режимы: fitcheck (128 строк, 16 на задачу, 500 обновлений, LR×3 без
косинуса; критерии: потеря ≥ 4× ниже стартовой и ниже постоянного
прогноза, MAE схватов ≤ 0.10; затем пересоздание и сверка init), train,
selftest. Возобновление сверяет контракт; чужой best/resume не затирается.
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

EFF_BATCH = 32
WARM_UPDATES, WARM_LR = 1000, 3e-4
MAIN_UPDATES = 8000
LR = dict(expert=2e-5, lora=1e-5, new=1e-4)
MAIN_WARM, HORIZON, LR_FLOOR = 200, 8000, 0.1
WD, CLIP = 0.01, 1.0
EVAL_EVERY, SAVE_EVERY = 500, 250
PANEL_N, PANEL_SEED = 256, 5
FIT_PER_TASK, FIT_UPDATES, FIT_LR_MULT = 16, 500, 3.0
FIT_RATIO, FIT_GRIP_MAE = 4.0, 0.10
SCHED_TAG, FIT_TAG = 23, 29
REALCHECK = "reports/rt1/realcheck.json"
FACTORIES = {"rt1_model:flat24": ("rt1_model.py", "k21_model.py",
                                  "k21_tree.py"),
             "rt1_model:parents24": ("rt1_model.py", "k21_model.py",
                                     "k21_tree.py", "k22_model.py")}
COMMON_DEPS = ("rt2_train.py", "rt1_common.py", "rt1_data.py",
               "rt1_prepare.py", "k21_data.py", "k23_smoke.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def outdir(seed, arch):
    return f"data/rt2/runs/s{seed}/{arch}"


def report_path(seed, arch):
    return f"reports/rt2/train_s{seed}_{arch}.json"


# --- выборка -----------------------------------------------------------------
def row_weights(ix, rows):
    """Равные веса задач, внутри задачи — эпизодов, внутри эпизода — строк."""
    task, ep = ix["task"][rows], ix["ep"][rows]
    w = np.zeros(len(rows))
    tasks = np.unique(task)
    for t in tasks:
        mt = task == t
        eps = np.unique(ep[mt])
        for e in eps:
            me = mt & (ep == e)
            w[me] = 1.0 / (len(tasks) * len(eps) * me.sum())
    return w


def schedule(train_rows, w, seed, u, micro):
    """Общая очередь 32 строк обновления u (с возвращением по весам) ->
    список micro-частей."""
    if EFF_BATCH % micro:
        raise SystemExit(f"micro {micro} не делит {EFF_BATCH}")
    g = np.random.default_rng([seed, SCHED_TAG, u])
    rows = g.choice(train_rows, EFF_BATCH, replace=True, p=w / w.sum())
    return rows.reshape(-1, micro)


def lr_main(u, base):
    if u < MAIN_WARM:
        return base * (u + 1) / MAIN_WARM
    t = min((u - MAIN_WARM) / max(HORIZON - MAIN_WARM, 1), 1.0)
    return base * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(
        math.pi * t)))


def fit_rows(D, seed=0):
    """16 строк на задачу: разные эпизоды; где есть смена схвата в ближайших
    20 шагах — строка у перехода, иначе равномерная."""
    import rt1_common as rc
    g = np.random.default_rng([seed, FIT_TAG])
    tr = D.rows("train")
    out = []
    for t in np.unique(D.ix["task"][tr]):
        eps = np.unique(D.ix["ep"][tr][D.ix["task"][tr] == t])
        pick = g.choice(eps, min(FIT_PER_TASK, len(eps)), replace=False)
        for e in pick:
            r = tr[D.ix["ep"][tr] == e]
            D.sample(int(r[0]))                     # кэш каналов эпизода
            act = D._ch[int(e)][1]
            gch = np.abs(np.diff(act[:, list(rc.GRIP_IDX)], axis=0)).max(1)
            sw = np.where(gch > 0.05)[0]
            if len(sw):
                t0 = int(max(0, g.choice(sw) - g.integers(0, 10)))
            else:
                t0 = int(g.integers(0, len(act)))
            out.append(int(r[min(t0, len(r) - 1)]))
    return np.asarray(out)


# --- модель и контракт ----------------------------------------------------------
def factory(spec):
    mod_name, arch = spec.split(":")
    if spec not in FACTORIES:
        raise SystemExit(f"фабрика {spec!r} не закреплена")
    import importlib
    return importlib.import_module(mod_name), arch


def realcheck_micro(a):
    """micro обязан совпадать с общим micro закреплённого real-check."""
    if not os.path.exists(REALCHECK):
        raise SystemExit(f"нет {REALCHECK}: сначала rt1_realcheck.py")
    r = json.load(open(REALCHECK, encoding="utf-8"))
    if r.get("problems"):
        raise SystemExit("real-check не пройден")
    if r.get("common_micro") != a.micro:
        raise SystemExit(f"micro {a.micro} ≠ общему micro real-check "
                         f"{r.get('common_micro')}")
    return r, sha(REALCHECK)


def contract(a, D, mode):
    import torch
    import transformers
    import rt1_common as rc
    deps = COMMON_DEPS + FACTORIES[a.factory]
    rr, rsha = realcheck_micro(a)
    return dict(
        mode=mode, factory=a.factory, seed=a.seed, prep_id=D.prep["prep_id"],
        code={f: sha(os.path.join(HERE, f)) for f in deps},
        realcheck_sha1=rsha, micro=a.micro, eff_batch=EFF_BATCH,
        precision="bfloat16 autocast, параметры float32, без GradScaler",
        tf32=dict(matmul=torch.backends.cuda.matmul.allow_tf32,
                  cudnn=torch.backends.cudnn.allow_tf32),
        budget=dict(warmup=WARM_UPDATES, main=MAIN_UPDATES, extension=False),
        lr=dict(warm=WARM_LR, **LR), schedule=dict(
            main_warm=MAIN_WARM, horizon=HORIZON, floor=LR_FLOOR),
        wd=WD, clip=CLIP, loss=dict(grip_w=rc.GRIP_W, kind="L1"),
        eval_every=EVAL_EVERY, save_every=SAVE_EVERY,
        panel=dict(n=PANEL_N, seed=PANEL_SEED), sampler="task->episode->row,"
        " общая очередь 32 строк на обновление",
        bar_revision=rc.BAR_REVISION, torch=torch.__version__,
        transformers=transformers.__version__,
        fit=dict(per_task=FIT_PER_TASK, updates=FIT_UPDATES,
                 lr_mult=FIT_LR_MULT, ratio=FIT_RATIO,
                 grip_mae=FIT_GRIP_MAE) if mode == "fitcheck" else None)


def build(torch, a, D, dev):
    import k21_train as ktr
    import rt1_common as rc
    mod_m, arch = factory(a.factory)
    bar = rc.load_bar(dev, "bfloat16")
    m = bar["model"]
    m.action_expert.float()
    M = mod_m.build(torch)
    mod = M[arch](m, D.L, D.U, checkpoint=True, seed=a.seed).to(dev)
    return mod, bar, ktr.state_hash(ktr.trainable_state(mod))


def set_phase(mod, phase):
    for k, v in mod.groups_all.items():
        on = (k == "new") or phase == "main"
        for p in v:
            p.requires_grad_(on)


def make_opt(torch, mod, phase, mult=1.0):
    gr = mod.groups_all
    if phase == "warmup":
        return torch.optim.AdamW([dict(params=gr["new"], lr=WARM_LR,
                                       base=WARM_LR, name="new")],
                                 weight_decay=WD)
    return torch.optim.AdamW([dict(params=gr[k], lr=LR[k] * mult,
                                   base=LR[k] * mult, name=k)
                              for k in ("expert", "lora", "new")],
                             weight_decay=WD)


def forward_loss(torch, rm, mod, D, bar, rows, dev):
    batch, po, st, tgt, val = D.make_batch(rows, bar, dev, torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = mod(batch, po, st)
    loss, la, lg = rm.loss_fn(torch, out, tgt, val)
    return out, loss, la, lg, tgt, val


def evaluate(torch, rm, mod, D, bar, rows, dev, micro, weights=None):
    import rt1_common as rc
    mod.eval()
    rows = np.asarray(rows)
    acc = dict(L=[], arm=[], arm8=[], grip=[], grip8=[], sat=[])
    Lt = torch.as_tensor(D.L[list(rc.ARM_IDX)], device=dev)
    Ut = torch.as_tensor(D.U[list(rc.ARM_IDX)], device=dev)
    with torch.no_grad():
        for s0 in range(0, len(rows), micro):
            out, loss, la, lg, tgt, val = forward_loss(
                torch, rm, mod, D, bar, rows[s0:s0 + micro], dev)
            if not torch.isfinite(loss).all():
                raise SystemExit("нечисловая потеря на проверке")
            v = val.float()
            v8 = v[:, :8]
            ea = (out["arm"] - tgt[..., list(rc.ARM_IDX)]).abs().mean(-1)
            eg = (out["grip"] - tgt[..., list(rc.GRIP_IDX)]).abs().mean(-1)
            acc["L"].append(loss.float().cpu().numpy())
            acc["arm"].append(((ea * v).sum(1) / v.sum(1)).cpu().numpy())
            acc["arm8"].append(((ea[:, :8] * v8).sum(1) /
                                v8.sum(1).clamp_min(1)).cpu().numpy())
            acc["grip"].append(((eg * v).sum(1) / v.sum(1)).cpu().numpy())
            acc["grip8"].append(((eg[:, :8] * v8).sum(1) /
                                 v8.sum(1).clamp_min(1)).cpu().numpy())
            a = out["arm"]
            near = ((a - Lt).abs() < 0.01 * (Ut - Lt)) | \
                ((Ut - a).abs() < 0.01 * (Ut - Lt))
            acc["sat"].append(near.float().mean((1, 2)).cpu().numpy())
    mod.train()
    w = np.ones(len(rows)) if weights is None else np.asarray(weights)
    res = {}
    for k, v in acc.items():
        x = np.concatenate(v)
        res[k] = [float((x * w).sum() / w.sum()), float(x.mean())]
    task = D.ix["task"][rows]
    Lc = np.concatenate(acc["L"])
    res["per_task_L"] = {str(t): float((Lc[task == t] * w[task == t]).sum() /
                                       w[task == t].sum())
                         for t in np.unique(task)}
    return res


class Trainer:
    def __init__(self, torch, a, D):
        import rt1_model as rm
        self.torch, self.a, self.D, self.rm = torch, a, D, rm
        self.dev = torch.device(a.device)
        _m, self.arch = factory(a.factory)
        self.dir = outdir(a.seed, self.arch)
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, "resume.pt")
        self.mod, self.bar, self.init_hash = build(torch, a, D, self.dev)
        self.mod.groups_all = self.mod.groups()
        self.train_rows = D.rows("train")
        self.w = row_weights(D.ix, self.train_rows)
        self.val_rows = D.rows("val")
        self.val_w = row_weights(D.ix, self.val_rows)
        g = np.random.default_rng(PANEL_SEED)
        self.panel = np.sort(g.choice(self.train_rows, PANEL_N, replace=False,
                                      p=self.w / self.w.sum()))
        self.contract = contract(a, D, "train")
        self.st = dict(phase="warmup", u=0, attempted=0, curve=[], best=None,
                       init_hash=self.init_hash, nonfinite=0)

    def save(self, opt):
        import k21_train as ktr
        torch = self.torch
        st = dict(self.st, weights=ktr.trainable_state(self.mod),
                  opt=opt.state_dict(), rng=torch.random.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state(self.dev),
                  contract=self.contract)
        torch.save(st, self.state_path + ".tmp")
        os.replace(self.state_path + ".tmp", self.state_path)

    def load(self):
        torch = self.torch
        if not os.path.exists(self.state_path):
            return None
        st = torch.load(self.state_path, map_location="cpu",
                        weights_only=False)
        if st["contract"] != self.contract:
            diff = sorted(k for k in set(st["contract"]) | set(self.contract)
                          if st["contract"].get(k) != self.contract.get(k))
            raise SystemExit(f"{self.state_path}: другой контракт ({diff}); "
                             f"существующее состояние не затирается")
        if st["init_hash"] != self.init_hash:
            raise SystemExit("начальные веса ≠ закреплённым")
        own = dict(self.mod.named_parameters())
        own.update(dict(self.mod.named_buffers()))
        with torch.no_grad():
            for k, v in st["weights"].items():
                own[k].copy_(v.to(own[k].device, own[k].dtype))
        torch.random.set_rng_state(st["rng"])
        torch.cuda.set_rng_state(st["cuda_rng"], self.dev)
        opt_state = st.pop("opt")
        for k in ("weights", "rng", "cuda_rng", "contract"):
            st.pop(k)
        self.st = st
        print(f"  продолжение: фаза {st['phase']}, обновление {st['u']}",
              flush=True)
        return opt_state

    def check(self, u):
        import k21_train as ktr
        torch = self.torch
        t0 = time.time()
        val = evaluate(torch, self.rm, self.mod, self.D, self.bar,
                       self.val_rows, self.dev, self.a.eval_micro, self.val_w)
        pan = evaluate(torch, self.rm, self.mod, self.D, self.bar,
                       self.panel, self.dev, self.a.eval_micro)
        rec = dict(update=u, val=val, panel=pan, seconds=time.time() - t0)
        self.st["curve"].append(rec)
        v = val["L"][0]
        if self.st["best"] is None or v < self.st["best"][1]:
            self.st["best"] = (u, v)
            st = ktr.trainable_state(self.mod)
            p = os.path.join(self.dir, "best.pt")
            torch.save(dict(weights=st, update=u, val_L=v,
                            contract=self.contract, init_hash=self.init_hash,
                            weights_hash=ktr.state_hash(st)), p + ".tmp")
            os.replace(p + ".tmp", p)
        print(f"  {self.arch} main {u}: val L {v:.4f} (рука {val['arm'][0]:.4f},"
              f" первые 8 {val['arm8'][0]:.4f}; схват {val['grip'][0]:.4f}); "
              f"панель {pan['L'][0]:.4f}; насыщение {val['sat'][0]:.3f}; "
              f"лучшая {self.st['best']}; проверка {rec['seconds']:.0f} с",
              flush=True)

    def run(self):
        torch = self.torch
        opt_state = self.load()
        set_phase(self.mod, self.st["phase"])
        opt = make_opt(torch, self.mod, self.st["phase"])
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        tm = dict(data=0.0, fb=0.0, n=0, t0=time.time())
        params = lambda: [p for g in opt.param_groups for p in g["params"]]
        while True:
            ph = self.st["phase"]
            if ph == "warmup" and self.st["u"] >= WARM_UPDATES:
                self.st.update(phase="main", u=0)
                set_phase(self.mod, "main")
                opt = make_opt(torch, self.mod, "main")
                print("  warmup завершён -> основной режим", flush=True)
                self.check(0)
                self.save(opt)
                continue
            if ph == "main" and self.st["u"] >= MAIN_UPDATES:
                break
            u = self.st["u"]
            gu = u if ph == "warmup" else WARM_UPDATES + u
            if ph == "main":
                for g in opt.param_groups:
                    g["lr"] = lr_main(u, g["base"])
            parts = schedule(self.train_rows, self.w, self.a.seed, gu,
                             self.a.micro)
            opt.zero_grad(set_to_none=True)
            for rows in parts:
                t0 = time.time()
                batch = self.D.make_batch(rows, self.bar, self.dev,
                                          torch.bfloat16)
                torch.cuda.synchronize(self.dev)
                t1 = time.time()
                b, po, st_, tgt, val = batch
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self.mod(b, po, st_)
                loss, _la, _lg = self.rm.loss_fn(torch, out, tgt, val)
                lm = loss.mean() / len(parts)
                if not bool(torch.isfinite(lm)):
                    raise SystemExit(f"нечисловая потеря на обновлении {gu}")
                lm.backward()
                torch.cuda.synchronize(self.dev)
                tm["data"] += t1 - t0
                tm["fb"] += time.time() - t1
            gn = torch.nn.utils.clip_grad_norm_(params(), CLIP)
            if not bool(torch.isfinite(gn)):
                raise SystemExit(f"нечисловой градиент на обновлении {gu}")
            opt.step()
            self.st["u"] = u + 1
            self.st["attempted"] += 1
            tm["n"] += 1
            if tm["n"] == 100:
                el = time.time() - tm["t0"]
                left = (WARM_UPDATES - self.st["u"] if ph == "warmup" else 0) \
                    + MAIN_UPDATES - (self.st["u"] if ph == "main" else 0)
                print(f"  {ph}: загрузка {tm['data'] / 100:.2f} с/обн, "
                      f"forward/backward {tm['fb'] / 100:.2f} с/обн, "
                      f"{3600 * 100 / el:.0f} обн/ч; до конца ~"
                      f"{left * el / 100 / 3600:.1f} ч (без проверок)",
                      flush=True)
            if ph == "main" and self.st["u"] % EVAL_EVERY == 0:
                self.check(self.st["u"])
            if self.st["u"] % SAVE_EVERY == 0:
                self.save(opt)
        self.save(opt)
        return self.st


def train(a):
    import torch
    import k21_train as ktr
    import rt1_data as rd
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    D = rd.Data()
    tr = Trainer(torch, a, D)
    have = os.path.exists(tr.state_path)
    print(f"  RT2 {tr.arch} seed {a.seed}: micro {a.micro} (накопление "
          f"{EFF_BATCH // a.micro}); " + ("продолжение" if have else
                                          "новый запуск"), flush=True)
    st = tr.run()
    b = torch.load(os.path.join(tr.dir, "best.pt"), map_location="cpu",
                   weights_only=False)
    rep = dict(arch=tr.arch, factory=a.factory, seed=a.seed,
               updates=st["u"], phase=st["phase"], best=st["best"],
               best_sha1=sha(os.path.join(tr.dir, "best.pt")),
               best_weights_hash=b["weights_hash"], init_hash=tr.init_hash,
               curve=st["curve"], contract=tr.contract,
               note="SR ещё не выполнен")
    if st["u"] != MAIN_UPDATES or st["phase"] != "main":
        raise SystemExit("обучение не дошло до бюджета")
    os.makedirs("reports/rt2", exist_ok=True)
    json.dump(rep, open(report_path(a.seed, tr.arch), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"  лучшая {st['best']}; отчёт {report_path(a.seed, tr.arch)}; "
          f"SR ещё не выполнен", flush=True)
    del ktr
    return 0


def fitcheck(a):
    import torch
    import rt1_data as rd
    import rt1_model as rm
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    D = rd.Data()
    dev = torch.device(a.device)
    rows = fit_rows(D, a.seed)
    mod, bar, h0 = build(torch, a, D, dev)
    mod.groups_all = mod.groups()
    set_phase(mod, "main")
    opt = make_opt(torch, mod, "main", mult=FIT_LR_MULT)
    # постоянный прогноз: средние цели по 128 строкам (действительные позиции)
    tg, vl = [], []
    for r in rows:
        s = D.sample(int(r))
        tg.append(s[3])
        vl.append(s[4])
    tg, vl = np.stack(tg), np.stack(vl).astype(float)
    mean = (tg * vl[..., None]).sum((0, 1)) / vl.sum()
    import rt1_common as rc
    ea = np.abs(tg - mean)[..., list(rc.ARM_IDX)].mean(-1)
    eg = np.abs(tg - mean)[..., list(rc.GRIP_IDX)].mean(-1)
    const = float((((ea + rc.GRIP_W * eg) * vl).sum(1) / vl.sum(1)).mean())

    def ev(m_, b_):
        r = evaluate(torch, rm, m_, D, b_, rows, dev, a.eval_micro)
        return dict(L=r["L"][1], arm=r["arm"][1], grip=r["grip"][1])
    curve = [dict(update=0, **ev(mod, bar))]
    g = np.random.default_rng([a.seed, FIT_TAG, 1])
    for u in range(1, FIT_UPDATES + 1):
        idx = rows[g.choice(len(rows), EFF_BATCH, replace=False)]
        opt.zero_grad(set_to_none=True)
        for part in idx.reshape(-1, a.micro):
            _o, loss, _la, _lg, _t, _v = forward_loss(torch, rm, mod, D, bar,
                                                      part, dev)
            (loss.mean() / (EFF_BATCH // a.micro)).backward()
        torch.nn.utils.clip_grad_norm_([p for gr in opt.param_groups
                                        for p in gr["params"]], CLIP)
        opt.step()
        if u % 100 == 0:
            curve.append(dict(update=u, **ev(mod, bar)))
            c = curve[-1]
            print(f"  fitcheck {u}: L {c['L']:.4f} (старт {curve[0]['L']:.4f},"
                  f" постоянный {const:.4f}); рука {c['arm']:.4f}, схват MAE "
                  f"{c['grip']:.4f}", flush=True)
    fin = curve[-1]
    ok = (fin["L"] <= curve[0]["L"] / FIT_RATIO and fin["L"] < const
          and fin["grip"] <= FIT_GRIP_MAE)
    del mod, bar, opt
    torch.cuda.empty_cache()
    _m2, _b2, h1 = build(torch, a, D, dev)
    same = h1 == h0
    rep = dict(rows=rows.tolist(), curve=curve, constant_L=const, ok=ok,
               init_same=same, criteria=dict(ratio=FIT_RATIO,
                                             grip_mae=FIT_GRIP_MAE),
               contract=contract(a, D, "fitcheck"))
    os.makedirs("reports/rt2", exist_ok=True)
    json.dump(rep, open("reports/rt2/fitcheck.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"  fitcheck: {'ПРОЙДЕН' if ok else 'НЕ ПРОЙДЕН'}; начальные веса "
          f"после пересоздания {'совпадают' if same else 'НЕ совпадают'}",
          flush=True)
    return 0 if ok and same else 3


def check_fresh(a):
    p = "reports/rt2/fitcheck.json"
    if not os.path.exists(p):
        return 1
    import rt1_data as rd
    r = json.load(open(p, encoding="utf-8"))
    D = rd.Data(check_files=False)
    want = contract(a, D, "fitcheck")
    ok = r.get("ok") and r.get("init_same") and r.get("contract") == want
    print(f"  {p}: " + ("пройден тем же контрактом" if ok else "пересчитать"))
    return 0 if ok else 1


def selftest():
    ix = dict(task=np.array([0] * 6 + [1] * 2), ep=np.array([0, 0, 0, 1, 1, 1,
                                                             2, 2]))
    rows = np.arange(8)
    w = row_weights(ix, rows)
    assert abs(w.sum() - 1) < 1e-12
    assert abs(w[ix["task"] == 0].sum() - 0.5) < 1e-12
    assert abs(w[:3].sum() - w[3:6].sum()) < 1e-12
    s1 = schedule(rows, w, 0, 5, 8)
    assert s1.shape == (4, 8) and np.array_equal(s1, schedule(rows, w, 0, 5,
                                                             8))
    assert np.array_equal(s1.reshape(-1), schedule(rows, w, 0, 5,
                                                   4).reshape(-1))
    assert not np.array_equal(s1, schedule(rows, w, 1, 5, 8))
    try:
        schedule(rows, w, 0, 0, 3)
        raise AssertionError("micro 3 принят")
    except SystemExit:
        pass
    assert abs(lr_main(0, 1.0) - 1 / MAIN_WARM) < 1e-12
    assert abs(lr_main(MAIN_WARM, 1.0) - 1.0) < 1e-12
    assert abs(lr_main(HORIZON, 1.0) - LR_FLOOR) < 1e-12
    assert set(FACTORIES) == {"rt1_model:flat24", "rt1_model:parents24"}
    print("самопроверка rt2_train пройдена: иерархические веса, общая "
          "очередь 32 строк (одинакова при любом micro), seed меняет "
          "очередь, косинус с горизонтом 8000 до 0.1, закреплённые фабрики")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="RoboTwin: тренер baseline")
    ap.add_argument("cmd", choices=("fitcheck", "train", "check-fresh"))
    ap.add_argument("--factory", default="rt1_model:flat24")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro", type=int, required=True)
    ap.add_argument("--eval-micro", type=int, default=4)
    a = ap.parse_args()
    if a.seed not in (0, 1):
        raise SystemExit("training seed RoboTwin — только 0 или 1")
    try:
        return dict(fitcheck=fitcheck, train=train,
                    **{"check-fresh": check_fresh})[a.cmd](a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
