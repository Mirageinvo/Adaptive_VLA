#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-27: обучение parents_mem24 (seed 0 и 1) поверх замороженного parents24
того же сида; real-check, fitcheck, решение о microbatch, check-fresh.

Обучается ТОЛЬКО self.mem (AdamW lr 3e-4, wd 0.01, clip 1.0; разогрев 100,
косинус до 0.1 на горизонте 2000; 2000 попыток обновления без добора;
эффективный батч 32 цепочки; float16 + GradScaler; параметры и состояния
AdamW — float32). Цепочка t−16, t−8, t (или t−8, t): ранние вызовы —
текущей моделью, eval/no_grad, каждый строит собственный снимок из своего
прогноза; текущий t — с градиентом модуля (checkpointing), потеря — прежняя
L1 руки + 0.1 BCE схвата только по экспертной цели текущей строки. Снимки
detach (truncated). Учитель K-21 не используется. Веса цепочек: задачи ->
эпизоды -> текущие строки. Очередь — numpy-генератор (seed, 27, u).
Проверка и сохранение каждые 250 (атомарно); лучшая — минимум взвешенной L на
панели val (≤ 512 цепочек; базовая L без памяти — один раз). База сверяется
с исходным отпечатком после загрузки, после каждого сохранения и в конце.
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

import k27_memory as kmem  # noqa: E402
import k27_model as k27m  # noqa: E402

SEEDS = (0, 1)
MODE = k27m.MODE
BUDGET = 2000
EFF = 32
LR, WD, CLIP = 3e-4, 0.01, 1.0
WARM, HORIZON, FLOOR = 100, 2000, 0.1
EVAL_EVERY = SAVE_EVERY = 250
MICROS = (8, 16, 32)
FIT_UPDATES, FIT_DROP = 200, 0.10
K_STEP = 8
PREPARE = "reports/k27/prepare.json"
MICRO_FILE = "reports/k27/micro.json"
CODE = ("k27_memory.py", "k27_model.py", "k27_train.py", "k27_data.py",
        "k22_model.py", "k21_model.py", "k21_tree.py", "k21_data.py",
        "k20_data.py", "k16_models.py", "k21_train.py", "k23_train.py")
HP = dict(budget=BUDGET, eff_batch=EFF, lr=LR, wd=WD, clip=CLIP, warmup=WARM,
          horizon=HORIZON, floor=FLOOR, eval_every=EVAL_EVERY,
          save_every=SAVE_EVERY, k_step=K_STEP, loss="L1 arm + 0.1 BCE grip",
          sources="expert current rows only", truncated=True)


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def outdir(seed):
    return f"data/k27/runs/s{seed}/{MODE}"


def report_path(seed):
    return f"reports/k27/train_s{seed}.json"


def lr_at(u):
    if u < WARM:
        return LR * (u + 1) / WARM
    t = min((u - WARM) / max(HORIZON - WARM, 1), 1.0)
    return LR * (FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * t)))


def chain_probs(D, ix):
    import k21_train as kt
    cur = ix["cur"]
    w = kt.row_weights(np.asarray(D["task"])[cur], np.asarray(D["episode"])
                       [cur])
    return w / w.sum()


def schedule(D, ix, seed, u, n=EFF):
    p = chain_probs(D, ix)
    g = np.random.default_rng([seed, 27, u])
    return g.choice(len(ix["cur"]), n, replace=True, p=p)


def schedule_fp(D, ix, seed, points=(0, 1, 99, 100, 1999)):
    h = hashlib.sha1()
    for u in points:
        h.update(np.ascontiguousarray(schedule(D, ix, seed, u)).tobytes())
    return h.hexdigest()


def seed_entry(seed):
    import k27_protocol as pc
    return pc.seed_entry(seed)


# --- модель -----------------------------------------------------------------
def load_model(torch, seed, D, dev, entry, checkpoint=True, memory="on"):
    """K27 с весами исторического parents24 сида (сверка SHA файла и
    отпечатка общих тензоров)."""
    import k21_data as d21
    import k21_protocol as pp
    import k21_train as kt
    ck_path = entry["source"]["ckpt"]
    if pp.full_sha(ck_path) != entry["source"]["ckpt_sha1"]:
        raise SystemExit(f"{ck_path} ≠ закреплённому SHA")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if cfg.get("mode") != "parents24" or int(cfg.get("seed", -1)) != seed:
        raise SystemExit(f"источник: режим {cfg.get('mode')}, seed "
                         f"{cfg.get('seed')} (нужен parents24 seed {seed})")
    bar = d21.load_bar(dev, D["prep_report"]["bar_pin"])
    m = bar["model"]
    m.action_expert.float()
    mod = k27m.build(torch)["K27"](
        m, D["L"], D["U"], D["state_mean"], D["state_std"],
        checkpoint=checkpoint, seed=seed, memory=memory).to(dev)
    own = dict(mod.named_parameters())
    own.update(dict(mod.named_buffers()))
    with torch.no_grad():
        for k, v in ck["weights"].items():
            if k not in own or k.startswith("mem."):
                raise SystemExit(f"ключ источника {k}")
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    h = base_hash(mod)
    if h != entry["source"]["weights_hash"] or h != ck["weights_hash"]:
        raise SystemExit(f"база после загрузки {h} ≠ отпечатку источника")
    if any(p.requires_grad for n, p in mod.named_parameters()
           if not n.startswith("mem.")):
        raise SystemExit("база не заморожена")
    _ = kt
    return mod, bar


def base_hash(mod):
    import k21_train as kt
    return kt.state_hash(k27m.shared_keys(kt.trainable_state(mod)))


def mem_state(mod):
    return {k: v.detach().cpu().clone() for k, v in mod.mem.state_dict()
            .items()}


def mem_hash(mod):
    import k21_train as kt
    return kt.state_hash(mem_state(mod))


def mem_from_rows(torch, rows, k=K_STEP):
    if not any(r is not None for r in rows):
        return None
    r0 = next(r for r in rows if r is not None)
    snap, _m = kmem.stack_snapshot(torch, rows, {kk: v[None] for kk, v in
                                                 r0.items()})
    return dict(snap=snap, use=[r is not None for r in rows],
                k=[k] * len(rows))


def chain_forward(torch, mod, bar, D, ix, ids, dev, dt, grad, use_memory=True,
                  cnt=None):
    """Цепочки ids: ранние вызовы — no_grad, собственные снимки; текущий —
    с градиентом (grad) -> out, loss [n], l1, bce, tgt, val."""
    import k21_data as d21
    import k21_model as km
    cur, p1, p2 = ix["cur"][ids], ix["prev1"][ids], ix["prev2"][ids]
    n = len(ids)
    ac = torch.autocast("cuda", dtype=dt, enabled=dev.type == "cuda")
    mem = None
    if use_memory:
        was = mod.training
        mod.eval()
        snaps = [None] * n
        with torch.no_grad():
            h3 = np.where(p2 >= 0)[0]
            if len(h3):
                b, po, st, _t, _v = d21.make_batch(D, p2[h3], bar, dev)
                with ac:
                    o = mod(b, po, st, None)
                for i, r in zip(h3, kmem.split_snapshot(o["snapshot"])):
                    snaps[i] = r
                del o
                if cnt is not None:
                    cnt["forwards"] += 1
                    cnt["observations"] += len(h3)
            b, po, st, _t, _v = d21.make_batch(D, p1, bar, dev)
            with ac:
                o = mod(b, po, st, mem_from_rows(torch, snaps))
            snaps = kmem.split_snapshot(o["snapshot"])
            del o
            if cnt is not None:
                cnt["forwards"] += 1
                cnt["observations"] += n
        if was:
            mod.train()
        mem = mem_from_rows(torch, snaps)
    b, po, st, tgt, val = d21.make_batch(D, cur, bar, dev)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, ac:
        out = mod(b, po, st, mem)
        loss, l1, bce = km.loss_fn(torch, out, tgt, val)
    if cnt is not None:
        cnt["forwards"] += 1
        cnt["observations"] += n
    return out, loss, l1, bce, tgt, val


def evaluate(torch, mod, bar, D, ix, ids, dev, dt, micro, use_memory=True):
    """Взвешенная (задачи -> эпизоды -> строки) L и L1 руки по цепочкам ids;
    агрегаты памяти: доля строк с памятью, квантили гейтов, RMS поправок."""
    import k21_train as kt
    ids = np.asarray(ids)
    cur = ix["cur"][ids]
    w = kt.row_weights(np.asarray(D["task"])[cur], np.asarray(D["episode"])
                       [cur])
    was = mod.training
    mod.eval()
    L_, l1_, has, g, rms = [], [], [], [], []
    with torch.no_grad():
        for s0 in range(0, len(ids), micro):
            out, loss, l1, _b, _t, _v = chain_forward(
                torch, mod, bar, D, ix, ids[s0:s0 + micro], dev,
                torch.float16 if dev.type == "cuda" else torch.float32,
                False, use_memory)
            if not (torch.isfinite(loss).all() and torch.isfinite(out["arm"])
                    .all()):
                raise SystemExit("нечисловая проверка")
            L_.append(loss.float().cpu().numpy())
            l1_.append(l1.float().cpu().numpy())
            mi = out.get("mem")
            if mi is not None:
                has.append(mi["has_root"].any(-1).float().cpu().numpy())
                g.append(mi["g_root"][mi["has_root"]].float().cpu().numpy())
                if "g_mid" in mi:
                    g.append(mi["g_mid"][mi["has_mid"]].float().cpu().numpy())
                rms.append([float(mi["rms_root"]),
                            float(mi.get("rms_mid", float("nan")))])
    if was:
        mod.train()
    L_, l1_ = np.concatenate(L_), np.concatenate(l1_)
    res = dict(L=float((L_ * w).sum() / w.sum()),
               l1_arm=float((l1_ * w).sum() / w.sum()), n=int(len(ids)))
    if use_memory:
        gg = np.concatenate(g) if g else np.zeros(0)
        res.update(share_with_memory=float(np.concatenate(has).mean())
                   if has else 0.0,
                   gate_q=[float(np.quantile(gg, q)) for q in (.05, .5, .95)]
                   if len(gg) else None,
                   rms_root=float(np.nanmean([r[0] for r in rms]))
                   if rms else None,
                   rms_mid=float(np.nanmean([r[1] for r in rms]))
                   if rms else None)
    return res


# --- контракт ----------------------------------------------------------------
def run_contract(seed, a, entry, seqrep):
    import k21_data as d21
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    return dict(
        mode=MODE, architecture=k27m.ARCHITECTURE, seed=seed,
        source=entry["source"], seed_entry_fp=entry["fingerprint"],
        prep_id=prep["prep_id"], bar_revision=prep["bar_pin"]["revision"],
        sequences=dict(npz_sha1=seqrep["npz_sha1"],
                       train=seqrep["splits"]["train"]["index_sha1"],
                       val=seqrep["splits"]["val"]["index_sha1"],
                       panel_seed=seqrep["panel"]["seed"]),
        memory=kmem.CONFIG, hp=HP, micro=a.micro, dtype=a.dtype,
        code={f: sha(os.path.join(HERE, f)) for f in CODE})


def check_contract(kind, a):
    import torch
    import k21_data as d21
    import k21_protocol as pp
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    c = dict(kind=kind, code={f: sha(os.path.join(HERE, f)) for f in CODE},
             prep_id=prep["prep_id"], prepare_sha1=pp.full_sha(d21.REPORT),
             bar_revision=prep["bar_pin"]["revision"],
             sequences_sha1=pp.full_sha("reports/k27/sequences.json"),
             k27_prepare=[seed_entry(s)["fingerprint"] for s in SEEDS],
             precision=f"{a.dtype} autocast + GradScaler",
             device=torch.cuda.get_device_name(torch.device(a.device)),
             memory=kmem.CONFIG)
    if kind == "real-check":
        c.update(micros=sorted(a.micros))
    else:
        c.update(fit=dict(updates=FIT_UPDATES, drop=FIT_DROP, lr=LR,
                          micro=a.micro, seed=0))
    return c


def check_fresh(a):
    p = "reports/k27/real_check.json" if a.kind == "real-check" else \
        "reports/k27/fitcheck.json"
    if not os.path.exists(p):
        print(f"  {p}: нет")
        return 1
    r = json.load(open(p, encoding="utf-8"))
    passed = r.get("problems") == [] if a.kind == "real-check" else \
        bool(r.get("ok"))
    want, got = check_contract(a.kind, a), r.get("contract") or {}
    diff = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
    ok = passed and not diff
    print(f"  {p}: " + ("пройден тем же контрактом" if ok else
                        f"пересчитать (пройден: {passed}; расхождения: "
                        f"{diff})"))
    return 0 if ok else 1


# --- тренер ------------------------------------------------------------------
class Trainer:
    def __init__(self, torch, a, seed, D, ix, ix_val, panel, entry, seqrep,
                 odir=None, make=None):
        import k23_train as k23t
        self.torch, self.a, self.seed, self.D, self.ix = torch, a, seed, D, ix
        self.ix_val, self.panel = ix_val, np.asarray(panel)
        self.dev = torch.device(a.device)
        self.dt = getattr(torch, a.dtype)
        self.dir = odir or outdir(seed)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "resume.pt")
        self.mod, self.bar = (make or load_model)(torch, seed, D, self.dev,
                                                  entry)
        self.base0 = base_hash(self.mod)
        self.init_mem = mem_hash(self.mod)
        self.opt = torch.optim.AdamW(self.mod.groups()["new"], lr=LR,
                                     weight_decay=WD)
        Sc = k23t.counting_scaler(torch.cuda.amp.GradScaler)
        self.scaler = Sc(enabled=self.dt == torch.float16 and
                         self.dev.type == "cuda")
        self.entry, self.seqrep = entry, seqrep
        self.st = dict(u=0, curve=[], best=None, baseline=None,
                       cnt=dict(forwards=0, observations=0), seconds=0.0)

    def config(self):
        return run_contract(self.seed, self.a, self.entry, self.seqrep)

    def save(self):
        torch = self.torch
        st = dict(self.st, mem=mem_state(self.mod),
                  opt=self.opt.state_dict(), scaler=self.scaler.state_dict(),
                  rng=torch.random.get_rng_state(), config=self.config(),
                  init_mem=self.init_mem, base_hash=self.base0)
        torch.save(st, self.path + ".tmp")
        os.replace(self.path + ".tmp", self.path)
        if base_hash(self.mod) != self.base0:
            raise SystemExit("база изменилась во время обучения")

    def load(self):
        torch = self.torch
        if not os.path.exists(self.path):
            return False
        st = torch.load(self.path, map_location="cpu", weights_only=False)
        if st["config"] != self.config():
            raise SystemExit(f"{self.path}: другая конфигурация")
        if st["init_mem"] != self.init_mem or st["base_hash"] != self.base0:
            raise SystemExit("начальные веса памяти или база ≠ закреплённым")
        self.mod.mem.load_state_dict({k: v.to(self.dev) for k, v in
                                      st["mem"].items()})
        self.opt.load_state_dict(st["opt"])
        self.scaler.load_state_dict(st["scaler"])
        torch.random.set_rng_state(st["rng"])
        self.st = {k: st[k] for k in self.st}
        print(f"  продолжение с обновления {self.st['u']}", flush=True)
        return True

    def check(self, u):
        t0 = time.time()
        if self.st["baseline"] is None:
            self.st["baseline"] = evaluate(
                self.torch, self.mod, self.bar, self.D, self.ix_val,
                self.panel, self.dev, self.dt, self.a.eval_micro,
                use_memory=False)
        ev = evaluate(self.torch, self.mod, self.bar, self.D, self.ix_val,
                      self.panel, self.dev, self.dt, self.a.eval_micro)
        sc = dict(getattr(self.scaler, "_k23", {}))
        rec = dict(update=u, val=ev, scaler=sc, cnt=dict(self.st["cnt"]),
                   seconds=time.time() - t0)
        self.st["curve"].append(rec)
        if self.st["best"] is None or ev["L"] < self.st["best"][1]:
            self.st["best"] = (u, ev["L"])
            p = os.path.join(self.dir, "best.pt")
            self.torch.save(dict(mem=mem_state(self.mod), update=u,
                                 val_L=ev["L"], config=self.config(),
                                 mem_hash=mem_hash(self.mod),
                                 base_hash=self.base0,
                                 init_mem=self.init_mem), p + ".tmp")
            os.replace(p + ".tmp", p)
        b = self.st["baseline"]
        gq = "/".join(f"{x:.3f}" for x in ev["gate_q"]) if ev["gate_q"] \
            else "—"
        print(f"  {MODE} s{self.seed} {u}: панель L {ev['L']:.4f} (без памяти "
              f"{b['L']:.4f}), L1 руки {ev['l1_arm']:.4f}; память у "
              f"{100 * ev['share_with_memory']:.0f}% строк, гейт q5/50/95 "
              f"{gq}, RMS поправки {ev['rms_root']:.4f}/"
              f"{ev['rms_mid']:.4f}; шагов {sc.get('steps')}, пропусков "
              f"{sc.get('skips')}; лучшая {self.st['best']} "
              f"({rec['seconds']:.0f} с)", flush=True)

    def step(self, ids, lr):
        torch = self.torch
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.zero_grad(set_to_none=True)
        mb = self.a.micro
        for s0 in range(0, len(ids), mb):
            sub = ids[s0:s0 + mb]
            out, loss, *_r = chain_forward(torch, self.mod, self.bar, self.D,
                                           self.ix, sub, self.dev, self.dt,
                                           True, cnt=self.st["cnt"])
            lm = loss.mean() * len(sub) / len(ids)
            if not bool(torch.isfinite(lm)):
                raise SystemExit("нечисловая потеря")
            self.scaler.scale(lm).backward()
            del out, loss
        bad = [n for n, p in self.mod.named_parameters()
               if not n.startswith("mem.") and p.grad is not None]
        if bad:
            raise SystemExit(f"градиент у замороженной базы: {bad[:3]}")
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.mod.groups()["new"], CLIP)
        self.scaler.step(self.opt)
        self.scaler.update()

    def run(self, until):
        self.load()
        self.mod.train()
        t_start, u0 = time.time(), self.st["u"]
        if self.st["u"] == 0 and not self.st["curve"]:
            self.check(0)
            self.save()
        while self.st["u"] < until:
            u = self.st["u"]
            self.step(schedule(self.D, self.ix, self.seed, u), lr_at(u))
            self.st["u"] = u + 1
            if self.st["u"] == u0 + 100:
                el = time.time() - t_start
                left = (until - self.st["u"]) * el / 100
                mem = self.torch.cuda.max_memory_allocated(self.dev) / 2 ** 30 \
                    if self.dev.type == "cuda" else 0.0
                print(f"  {MODE} s{self.seed}: {100 / el:.2f} обн./с, VLA "
                      f"forward {self.st['cnt']['forwards']}, память {mem:.1f} "
                      f"ГБ; до {until} ~{left / 3600:.1f} ч (без проверок)",
                      flush=True)
            if self.st["u"] % EVAL_EVERY == 0:
                self.check(self.st["u"])
            if self.st["u"] % SAVE_EVERY == 0:
                self.st["seconds"] += time.time() - t_start
                t_start = time.time()
                self.save()
        self.save()
        return self.st


# --- режимы ------------------------------------------------------------------
def decided_micro():
    if not os.path.exists(MICRO_FILE):
        return None
    return int(json.load(open(MICRO_FILE, encoding="utf-8"))["micro"])


def train(a):
    import torch
    import k21_data as d21
    import k27_data as kd27
    if a.seed not in SEEDS:
        raise SystemExit(f"seed {a.seed}: в K-27 только {SEEDS}")
    if decided_micro() != a.micro:
        raise SystemExit(f"micro {a.micro} ≠ решению {MICRO_FILE}")
    e = seed_entry(a.seed)
    D = d21.load_prepared(require_teacher=False)
    S = kd27.load()
    ix = S["train"]
    tr = Trainer(torch, a, a.seed, D, ix, S["val"], S["panel"], e,
                 S["report"])
    print(f"  K-27 seed {a.seed}: бюджет {BUDGET}, micro {a.micro} × "
          f"{EFF // a.micro}; цепочек train {len(ix['cur'])}, панель "
          f"{len(S['panel'])}; база {tr.base0[:12]} (источник "
          f"{e['source']['ckpt']})", flush=True)
    st = tr.run(BUDGET)
    if base_hash(tr.mod) != e["source"]["weights_hash"]:
        raise SystemExit("база в конце ≠ источнику")
    b = torch.load(os.path.join(tr.dir, "best.pt"), map_location="cpu",
                   weights_only=False)
    rep = dict(seed=a.seed, mode=MODE, budget=BUDGET, updates=st["u"],
               best=st["best"], best_sha1=sha(os.path.join(tr.dir,
                                                           "best.pt")),
               best_mem_hash=b["mem_hash"], base_hash=tr.base0,
               init_mem=tr.init_mem, baseline=st["baseline"],
               counters=st["cnt"], scaler=dict(getattr(tr.scaler, "_k23",
                                                       {})),
               seconds=st["seconds"], curve=st["curve"], config=tr.config(),
               note="адаптация только модуля памяти поверх обученного "
                    "parents24; не равна по compute обучению с нуля")
    os.makedirs("reports/k27", exist_ok=True)
    json.dump(rep, open(report_path(a.seed), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"  K-27 seed {a.seed}: лучшая {st['best']} (без памяти "
          f"{st['baseline']['L']:.4f}); отчёт {report_path(a.seed)}",
          flush=True)
    return 0


def fitcheck(a):
    import torch
    import k21_data as d21
    import k27_data as kd27
    e = seed_entry(0)
    D = d21.load_prepared(require_teacher=False)
    S = kd27.load()
    ix = S["train"]
    fit = S["fit"]
    dev = torch.device(a.device)
    mod, bar = load_model(torch, 0, D, dev, e)
    import k23_train as k23t
    opt = torch.optim.AdamW(mod.groups()["new"], lr=LR, weight_decay=WD)
    scaler = k23t.counting_scaler(torch.cuda.amp.GradScaler)()
    h0, m0 = base_hash(mod), mem_hash(mod)
    ev0 = evaluate(torch, mod, bar, D, ix, fit, dev, torch.float16,
                   a.eval_micro)
    curve = [dict(update=0, L=ev0["L"])]
    rng = np.random.default_rng(20261030)
    mod.train()
    for u in range(1, FIT_UPDATES + 1):
        ids = rng.choice(fit, EFF, replace=len(fit) < EFF)
        opt.zero_grad(set_to_none=True)
        for s0 in range(0, EFF, a.micro):
            sub = ids[s0:s0 + a.micro]
            out, loss, *_r = chain_forward(torch, mod, bar, D, ix, sub, dev,
                                           torch.float16, True)
            scaler.scale(loss.mean() * len(sub) / EFF).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(mod.groups()["new"], CLIP)
        scaler.step(opt)
        scaler.update()
        if u % 50 == 0:
            ev = evaluate(torch, mod, bar, D, ix, fit, dev, torch.float16,
                          a.eval_micro)
            curve.append(dict(update=u, L=ev["L"]))
            print(f"  fitcheck {u}: L {ev['L']:.4f} (старт {ev0['L']:.4f}); "
                  f"шагов {scaler._k23['steps']}, пропусков "
                  f"{scaler._k23['skips']}", flush=True)
    fin = curve[-1]["L"]
    checks = dict(finite=bool(np.isfinite(fin)),
                  optimizer_applied=scaler._k23["steps"] > 0,
                  base_unchanged=base_hash(mod) == h0,
                  memory_trained=mem_hash(mod) != m0,
                  drop_10pct=fin <= (1 - FIT_DROP) * ev0["L"])
    ok = all(checks.values())
    rep = dict(ok=ok, checks=checks, curve=curve, start=ev0, fit_n=len(fit),
               scaler=scaler._k23, contract=check_contract("fitcheck", a))
    os.makedirs("reports/k27", exist_ok=True)
    json.dump(rep, open("reports/k27/fitcheck.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"  fitcheck: {'ПРОЙДЕН' if ok else 'НЕ ПРОЙДЕН'} {checks}; L "
          f"{ev0['L']:.4f} -> {fin:.4f} ({100 * (1 - fin / ev0['L']):.1f}%)",
          flush=True)
    return 0 if ok else 3


def real_check(a):
    import torch
    import k21_data as d21
    import k21_model as km
    import k21_train as kt
    import k22_model as k22
    import k27_data as kd27
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    S = kd27.load()
    ix = S["train"]
    prob, rep = [], dict(micros=sorted(a.micros))
    if a.dtype != "float16":
        prob.append("real-check только для float16 + GradScaler")
    rows8 = ix["cur"][:: max(1, len(ix["cur"]) // 8)][:8]
    # 1. оба сида: база = источнику, заморозка, memory off / нулевой Wo =
    # parents24 (замороженный, как в руке SR) на настоящих строках
    for s in SEEDS:
        e = seed_entry(s)
        mod, bar = load_model(torch, s, D, dev, e, checkpoint=False)
        r = dict(base_hash=base_hash(mod), source=e["source"]["weights_hash"],
                 mem_init=mem_hash(mod),
                 n_new=sum(p.numel() for p in mod.groups()["new"]),
                 n_new_expected=kmem.n_new_params(mod.H))
        if r["n_new"] != r["n_new_expected"]:
            prob.append(f"seed {s}: новых параметров {r['n_new']}")
        # эталон — отдельный K22 на собственном BAR (K22 оборачивает LoRA
        # при создании; общий BAR испортил бы проверяемую модель)
        b2 = d21.load_bar(dev, D["prep_report"]["bar_pin"])
        b2["model"].action_expert.float()
        ref = k22.build(torch)["K22"](
            b2["model"], D["L"], D["U"], D["state_mean"], D["state_std"],
            checkpoint=False, seed=s).to(dev)
        ck = torch.load(e["source"]["ckpt"], map_location="cpu",
                        weights_only=False)
        own = dict(ref.named_parameters())
        own.update(dict(ref.named_buffers()))
        with torch.no_grad():
            for k, v in ck["weights"].items():
                own[k].copy_(v.to(own[k].device, own[k].dtype))
        for p in ref.parameters():
            p.requires_grad_(False)
        mod.eval()
        ref.eval()
        batch, po, st, _t, _v = d21.make_batch(D, rows8, bar, dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            o_ref = ref(batch, po, st)
            o1 = mod(batch, po, st, None)
            snaps = kmem.split_snapshot(o1["snapshot"])
            o2 = mod(batch, po, st, mem_from_rows(torch, snaps))
        r["memory_off_vs_parents24"] = max(
            float((o_ref[k].float() - o1[k].float()).abs().max())
            for k in ("arm", "grip_logit"))
        r["zero_wo_with_memory_vs_parents24"] = max(
            float((o_ref[k].float() - o2[k].float()).abs().max())
            for k in ("arm", "grip_logit"))
        r["mem_reads"] = mod.counters["mem_reads"]
        if r["base_hash"] != r["source"]:
            prob.append(f"seed {s}: база ≠ источнику")
        if r["memory_off_vs_parents24"] != 0 or \
                r["zero_wo_with_memory_vs_parents24"] != 0:
            prob.append(f"seed {s}: memory off / нулевой Wo ≠ parents24 "
                        f"({r['memory_off_vs_parents24']:.2e}, "
                        f"{r['zero_wo_with_memory_vs_parents24']:.2e})")
        if r["mem_reads"] != ["root", "mid"]:
            prob.append(f"seed {s}: чтения памяти {r['mem_reads']}")
        rep[f"seed_{s}"] = r
        print(f"  seed {s}: {json.dumps(r, default=float)}", flush=True)
        del mod, bar, ref, b2, o_ref, o1, o2
        torch.cuda.empty_cache()
    # 2. путь обучения seed 0: micro 8/16/32, цепочки из 2 и 3 наблюдений
    e = seed_entry(0)
    mod, bar = load_model(torch, 0, D, dev, e)
    import k23_train as k23t
    opt = torch.optim.AdamW(mod.groups()["new"], lr=LR, weight_decay=WD)
    scaler = k23t.counting_scaler(torch.cuda.amp.GradScaler)()
    h0 = base_hash(mod)
    i3 = np.where(ix["prev2"] >= 0)[0]
    i2 = np.where(ix["prev2"] < 0)[0]
    mod.train()
    for mb in sorted(a.micros):
        half = mb // 2
        ids = np.r_[i3[:mb - half], i2[:half]] if len(i2) else i3[:mb]
        r = {}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            cnt = dict(forwards=0, observations=0)
            out, loss, *_r = chain_forward(torch, mod, bar, D, ix, ids, dev,
                                           torch.float16, True, cnt=cnt)
            scaler.scale(loss.mean()).backward()
            frozen_grad = [n for n, p in mod.named_parameters()
                           if not n.startswith("mem.") and p.grad is not None]
            sc = float(scaler.get_scale())
            gn = {n: float(getattr(mod.mem, n).weight.grad.float().norm()) / sc
                  for n in ("Wq", "Wk", "Wv", "Wo", "gate")}
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(mod.groups()["new"], CLIP)
            before = float(scaler.get_scale())
            scaler.step(opt)
            scaler.update()
            applied = float(scaler.get_scale()) >= before
            torch.cuda.synchronize(dev)
            arm = out["arm"].float()
            checks = dict(
                finite=bool(torch.isfinite(arm).all() and
                            torch.isfinite(loss).all()),
                frozen_no_grad=not frozen_grad,
                base_unchanged=base_hash(mod) == h0,
                counters=mod.counters["vision"] == 1 and
                all(x == 1 for x in mod.counters["text"]) and
                mod.counters["mem_reads"] == ["root", "mid"],
                chains_2_and_3=bool(len(i3) and len(i2)),
                params_finite=all(bool(torch.isfinite(p).all())
                                  for p in mod.groups()["new"]))
            r = dict(checks=checks, seconds=time.time() - t0,
                     peak_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                     forwards=cnt["forwards"], observations=cnt["observations"],
                     grad_norms=gn, optimizer_step_applied=applied,
                     loss=float(loss.mean()))
            bad = [k for k, v in checks.items() if not v]
            if bad:
                prob.append(f"micro {mb}: {bad}")
            del out, loss
        except torch.cuda.OutOfMemoryError:
            r = dict(oom=True)
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        rep[f"micro_{mb}"] = r
        print(f"  micro {mb}: {json.dumps({k: r.get(k) for k in ('checks', 'peak_gb', 'seconds', 'forwards', 'oom')}, default=float)}",
              flush=True)
    # 3. путь градиента после применённого шага: Q/K/V/gate получают градиент
    tries = 0
    while float(mod.mem.Wo.weight.detach().abs().sum()) == 0 and tries < 8:
        opt.zero_grad(set_to_none=True)
        _o, loss, *_r = chain_forward(torch, mod, bar, D, ix, i3[:8], dev,
                                      torch.float16, True)
        scaler.scale(loss.mean()).backward()
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()
        tries += 1
    opt.zero_grad(set_to_none=True)
    _o, loss, *_r = chain_forward(torch, mod, bar, D, ix, i3[:8], dev,
                                  torch.float16, True)
    sc = float(scaler.get_scale())
    scaler.scale(loss.mean()).backward()
    gn = {n: float(getattr(mod.mem, n).weight.grad.float().norm()) / sc
          for n in ("Wq", "Wk", "Wv", "Wo", "gate")}
    rep["grad_after_step"] = dict(grads=gn, extra_steps=tries,
                                  scaler=dict(scaler._k23), grad_scale=sc,
                                  wo_norm=float(mod.mem.Wo.weight.detach()
                                                .float().norm()))
    if not all(np.isfinite(v) and v > 0 for v in gn.values()):
        prob.append(f"градиент памяти после шага: {gn}")
    if base_hash(mod) != h0:
        prob.append("база изменилась после шагов")
    print(f"  после шага: {rep['grad_after_step']}", flush=True)
    ok_m = [m for m in sorted(a.micros) if (rep.get(f"micro_{m}") or {})
            .get("checks") and all(rep[f"micro_{m}"]["checks"].values())]
    if not ok_m:
        prob.append("нет пригодного microbatch")
    elif any((rep.get(f"micro_{m}") or {}).get("oom") for m in a.micros
             if m < max(ok_m)):
        prob.append("OOM на меньшем microbatch при пригодном большем")
    rep["micro_choice"] = max(ok_m) if ok_m else None
    rep["problems"] = prob
    rep["contract"] = check_contract("real-check", a)
    os.makedirs("reports/k27", exist_ok=True)
    json.dump(rep, open("reports/k27/real_check.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    if not prob:
        old = decided_micro()
        if old is not None and old != rep["micro_choice"]:
            prob.append(f"решение micro уже записано: {old}")
        elif old is None:
            json.dump(dict(micro=rep["micro_choice"],
                           accumulation=EFF // rep["micro_choice"],
                           peak_gb={m: (rep.get(f"micro_{m}") or {})
                                    .get("peak_gb") for m in a.micros}),
                      open(MICRO_FILE, "w", encoding="utf-8"), indent=1)
    _ = (km, kt)
    print("  " + ("ОТКАЗ: " + "; ".join(prob) if prob else
                  f"real-check K-27 пройден; micro {rep['micro_choice']} × "
                  f"{EFF // rep['micro_choice']}") + "; reports/k27/"
          "real_check.json", flush=True)
    return 3 if prob else 0


def selftest():
    import copy
    import tempfile
    import types
    import torch
    import k21_data as d21
    import k21_model as km
    import k27_data as kd27
    M = k27m.build(torch)
    bar0 = km.fake_bar(torch)
    L, U = -np.ones(6) * 1.1, np.ones(6) * 4.0
    # игрушечные данные: 4 задачи × 3 эпизода × шаги 0..40 через 8
    task, ep, stp = [], [], []
    for t in range(4):
        for e in range(3):
            for s in range(0, 48, 8):
                task.append(f"t{t}")
                ep.append(10 * t + e)
                stp.append(s)
    N = len(task)
    rows = np.arange(N)
    tr = rows[np.asarray(ep) % 10 < 2]
    va = rows[np.asarray(ep) % 10 == 2]
    D = dict(task=np.asarray(task), episode=np.asarray(ep),
             step=np.asarray(stp), train_rows=tr, val_rows=va, L=L, U=U,
             state_mean=None, state_std=None, prep_report=dict(bar_pin=None))
    ixt = kd27.build_index(D["task"], D["episode"], D["step"], tr)
    ixv = kd27.build_index(D["task"], D["episode"], D["step"], va)
    assert (ixt["prev2"] >= 0).any() and (ixt["prev2"] < 0).any()

    def fake_mb(D_, rows_, bar, dev, teacher_idx=None):
        rows_ = np.asarray(rows_)
        b = km.fake_batch(torch, B=len(rows_), seed=int(rows_.sum()) % 997)
        g = torch.Generator().manual_seed(int(rows_.sum()) + 7)
        st_ = torch.randn(len(rows_), 11, generator=g)
        tgt = torch.rand(len(rows_), 20, 7, generator=g) * 2
        tgt[..., 6] = torch.where(tgt[..., 6] > 1, 1.0, -1.0)
        return b, 1, st_, tgt, torch.ones(len(rows_), 20, dtype=torch.bool)
    base = M["K22"](copy.deepcopy(bar0), L, U, seed=1)
    with torch.no_grad():
        for p in base.parameters():
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=torch.Generator()
                                   .manual_seed(p.numel())) * 0.02)
    import k21_train as kt
    sb = kt.trainable_state(base)

    def fake_make(torch_, seed, D_, dev, entry, checkpoint=True,
                  memory="on"):
        m = M["K27"](copy.deepcopy(bar0), L, U, seed=seed, memory=memory)
        own = dict(m.named_parameters())
        own.update(dict(m.named_buffers()))
        with torch.no_grad():
            for k, v in sb.items():
                own[k].copy_(v)
        return m, dict(model=m)
    saved = (d21.make_batch, globals()["EVAL_EVERY"], globals()["SAVE_EVERY"],
             globals()["run_contract"])
    entry = dict(source=dict(ckpt="x", weights_hash=kt.state_hash(sb)),
                 fingerprint="f")
    seqrep = dict(npz_sha1="n", splits=dict(train=dict(index_sha1="a"),
                                            val=dict(index_sha1="b")),
                  panel=dict(seed=1))
    a_ = types.SimpleNamespace(device="cpu", dtype="float32", micro=4,
                               eval_micro=4)
    try:
        d21.make_batch = fake_mb
        globals()["EVAL_EVERY"] = globals()["SAVE_EVERY"] = 2
        globals()["run_contract"] = lambda seed, a__, e_, s_: dict(seed=seed)
        res = {}
        with tempfile.TemporaryDirectory() as td:
            for name, stops in (("cont", (4,)), ("resume", (2, 4))):
                for until in stops:
                    T = Trainer(torch, a_, 1, D, ixt, ixv,
                                np.arange(len(ixv["cur"]))[:6], entry,
                                seqrep, odir=os.path.join(td, name),
                                make=fake_make)
                    T.run(until)
                res[name] = (mem_hash(T.mod), base_hash(T.mod),
                             [(c["update"], round(c["val"]["L"], 10))
                              for c in T.st["curve"]], T.st["cnt"])
            assert res["cont"] == res["resume"], res
            assert res["cont"][1] == kt.state_hash(sb)
            assert res["cont"][0] != T.init_mem
            # 2 цепочки по 3 и 2 наблюдения: прямых вызовов больше, чем строк
            assert T.st["cnt"]["forwards"] > 4
    finally:
        d21.make_batch, g1, g2, rc = saved
        globals()["EVAL_EVERY"], globals()["SAVE_EVERY"] = g1, g2
        globals()["run_contract"] = rc
    # очередь: от seed зависит, повторяема; веса задач равны
    q0 = schedule(D, ixt, 0, 5)
    assert np.array_equal(q0, schedule(D, ixt, 0, 5)) and \
        not np.array_equal(q0, schedule(D, ixt, 1, 5))
    p = chain_probs(D, ixt)
    tk = D["task"][ixt["cur"]]
    assert np.allclose([p[tk == t].sum() for t in np.unique(tk)], 0.25)
    # LR: разогрев 100, косинус до 0.1 на 2000
    assert lr_at(0) < lr_at(99) <= LR and abs(lr_at(2000) - LR * FLOOR) < 1e-12
    # снимки из истории отделены; ранний вызов не получает градиент
    m, _b = fake_make(torch, 0, D, torch.device("cpu"), entry)
    m.train()
    saved_mb = d21.make_batch
    try:
        d21.make_batch = fake_mb
        ids = np.where(ixt["prev2"] >= 0)[0][:3]
        out, loss, *_r = chain_forward(torch, m, dict(model=m), D, ixt, ids,
                                       torch.device("cpu"), torch.float32,
                                       True)
        assert out["mem"]["use"].all() and m.counters["mem_reads"] == [
            "root", "mid"]
        loss.mean().backward()
        assert all(p_.grad is None for n_, p_ in m.named_parameters()
                   if not n_.startswith("mem."))
        assert float(m.mem.Wo.weight.grad.abs().sum()) > 0
        # без памяти — путь parents24
        out0, *_r = chain_forward(torch, m, dict(model=m), D, ixt, ids,
                                  torch.device("cpu"), torch.float32, False,
                                  use_memory=False)
        assert "mem" not in out0
    finally:
        d21.make_batch = saved_mb
    # настоящая фабрика: K27 нужного seed, отказ при чужом источнике
    saved_lb = d21.load_bar
    import k21_protocol as pp
    try:
        d21.load_bar = lambda dev, pin: dict(model=copy.deepcopy(bar0))
        with tempfile.TemporaryDirectory() as td:
            ckp = os.path.join(td, "best.pt")
            for s_ in SEEDS:
                cfg = dict(mode="parents24", seed=s_)
                torch.save(dict(weights=sb, weights_hash=kt.state_hash(sb),
                                config=cfg), ckp)
                en = dict(source=dict(ckpt=ckp, ckpt_sha1=pp.full_sha(ckp),
                                      weights_hash=kt.state_hash(sb)))
                mf, _bf = load_model(torch, s_, dict(D, prep_report=dict(
                    bar_pin=None)), torch.device("cpu"), en)
                assert type(mf).__name__ == "K27" and mf.mode == MODE and \
                    mf.memory_mode == "on"
                assert base_hash(mf) == kt.state_hash(sb)
                try:
                    load_model(torch, 1 - s_, dict(D, prep_report=dict(
                        bar_pin=None)), torch.device("cpu"), en)
                    raise AssertionError("источник чужого seed принят")
                except SystemExit:
                    pass
    finally:
        d21.load_bar = saved_lb
    print("самопроверка k27_train пройдена: непрерывное = остановка + "
          "возобновление (память, база, кривая, счётчики), база неизменна, "
          "очередь зависит от seed и повторяема, веса задач равны, LR "
          "100/косинус 2000/пол 0.1, цепочки 2/3 наблюдения с собственными "
          "снимками, градиент только у mem.*, без памяти — путь parents24, "
          "настоящая фабрика (K27, seed, отказ при источнике чужого seed)")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-27: обучение и проверки")
    ap.add_argument("cmd", choices=("real-check", "fitcheck", "train",
                                    "check-fresh", "micro"))
    ap.add_argument("--kind", choices=("real-check", "fitcheck"),
                    default="real-check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--eval-micro", type=int, default=8)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--micros", type=int, nargs="+", default=list(MICROS))
    a = ap.parse_args()
    try:
        if a.cmd == "real-check":
            return real_check(a)
        if a.cmd == "fitcheck":
            if decided_micro() != a.micro:
                raise SystemExit(f"fitcheck micro {a.micro} ≠ решению "
                                 f"{decided_micro()}")
            return fitcheck(a)
        if a.cmd == "check-fresh":
            return check_fresh(a)
        if a.cmd == "micro":
            m = decided_micro()
            if m is None:
                raise SystemExit("решения о microbatch нет")
            print(m)
            return 0
        return train(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
