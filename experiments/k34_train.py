#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-34: дообучение источника K-26 с тремя исполнимыми выходами 8/16/24.

Источник — best.pt закреплённого плана K-26 того же режима и seed (запись
reports/k34/prepare.json): загрузка весов, сверка отпечатка, затем
attach_exits (readout8/readout16 — копии readout24).
Потеря каждого выхода — неизменённая k21_model.loss_fn (L1 руки + 0.1·BCE
схвата, прежние valid и маска нулевого схвата учителя); общая —
(L8 + L16 + L24)/3; один forward_all на microbatch.
Режим:
  warmup 500 обновлений: обучаются ТОЛЬКО readout8/readout16 (весь источник,
         включая readout24 и router, заморожен), AdamW lr 3e−4;
  main   4000 обновлений, новый AdamW (момент источника не загружается):
         эксперт 2e−5, LoRA 1e−5, new K-26 1e−4, новые головы 1e−4;
         разогрев 200, косинус до 0.1 на 4000 (собственный scheduler, не
         MAIN_MAX старого тренера); wd 0.01, clip 1.0; эффективный батч 32.
Очередь: k21_train.schedule с training seed S и абсолютными номерами
  u_abs = 200000 + u, u = 0…4499 (первые 500 — warmup); учитель — по u_abs
  mod 4 (одинаковые строки у T и F одного seed); отпечаток очереди в
  контракте. Проверка — main 0 и каждые 250: по каждой глубине L, рука-20,
  рука-8, схват (веса строк K-21); best — минимум среднего (L8+L16+L24)/3,
  при равенстве — более ранний. Без добора, подбора весов, вариантов.
Продолжение: resume.pt (веса, оптимизатор, GradScaler, RNG, счётчики) с
  проверкой контракта (источник, головы, очередь, код, константы) до записи.
fitcheck: 64 фиксированные train-строки, 200 обновлений только ранних
  голов (lr 3e−4); у каждой ранней головы L снизилась ≥ 30%, веса
  изменились; выход 24 и все веса источника побитово прежние.
real-check: до обучения выход 24 нового экземпляра = исходной руке K-26
  побитово (B = 1 и 5); усечённые 8/16 = forward_all (arm ≤ 1e-6, grip ≤
  1e-5); счётчики; один main-шаг меняет обе ранние головы и допустимые
  группы, градиенты конечны, vision/base text не меняются; память/скорость.
"""
import argparse
import copy
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

import k34_model as k34m  # noqa: E402

SEEDS = (0, 1)
MODES = k34m.MODES
EFF_BATCH = 32
WARM_UPDATES, WARM_LR = 500, 3e-4
MAIN_UPDATES = 4000
LR = dict(expert=2e-5, lora=1e-5, new=1e-4, heads=1e-4)
MAIN_WARM, LR_FLOOR = 200, 0.1
WD, CLIP = 0.01, 1.0
EVAL_EVERY = SAVE_EVERY = 250
U_OFFSET = 200_000
FIT_N, FIT_UPDATES, FIT_DROP = 64, 200, 0.30
FIT_ROWS_SEED = 34009
PREPARE = "reports/k34/prepare.json"
OUTDIR = "data/k34/runs"
LOSS = "(L8 + L16 + L24) / 3, L_d = k21_model.loss_fn (L1 + 0.1·BCE)"
CODE = ("k34_model.py", "k34_train.py", "k26_model.py", "k26_attention.py",
        "k22_model.py", "k21_model.py", "k21_tree.py", "k21_train.py",
        "k21_data.py", "k20_data.py", "k21_fastload.py", "k23_train.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def run_dir(seed, mode):
    return os.path.join(OUTDIR, f"s{seed}", mode)


def report_path(seed, mode):
    return f"reports/k34/train_s{seed}_{mode}.json"


def lr_main(u, base):
    if u < MAIN_WARM:
        return base * (u + 1) / MAIN_WARM
    t = (u - MAIN_WARM) / max(MAIN_UPDATES - MAIN_WARM, 1)
    return base * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(
        math.pi * min(t, 1.0))))


def sched_entry(D, u_abs, micro, seed):
    import k21_train as kt
    return kt.schedule(D, u_abs + 1, micro, seed=seed, start=u_abs)[0]


def schedule_fp(D, micro, seed, points=(0, 1, 2, 3, 499, 500, 501, 4499)):
    h = hashlib.sha1()
    for u in points:
        src, idx = sched_entry(D, U_OFFSET + u, micro, seed)
        h.update(src.encode())
        h.update(np.ascontiguousarray(np.asarray(idx, np.int64)).tobytes())
    return h.hexdigest()


# --- источник ---------------------------------------------------------------------------
def source_entry(seed, mode):
    if not os.path.exists(PREPARE):
        raise SystemExit(f"нет {PREPARE}: сначала k34_protocol.py prepare")
    p = json.load(open(PREPARE, encoding="utf-8"))
    e = (p.get("sources") or {}).get(f"{mode}_s{seed}")
    if e is None:
        raise SystemExit(f"в подготовке нет источника {mode}_s{seed}")
    return e


def load_source(torch, mode, seed, spec, device, checkpoint=True):
    """Модель K-34 с весами best.pt источника K-26 (сверка отпечатка) и
    головами 8/16 — копиями readout24."""
    import k21_data as d21
    import k21_train as ktr
    import k26_policy as kp26
    ck = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    prob = kp26.spec_problems(f"{mode}_s{seed}", spec, ck["config"],
                              ck["weights_hash"])
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    if prep.get("prep_id") != ck["config"]["prep_id"]:
        prob.append("источник обучен на другой подготовке")
    if prob:
        raise SystemExit("; ".join(prob))
    dev = torch.device(device)
    bar = d21.load_bar(dev, prep["bar_pin"])
    m = bar["model"]
    m.action_expert.float()
    mod = k34m.build(torch)["make"](
        mode, m, np.asarray(prep["bounds_codec"]["L"]),
        np.asarray(prep["bounds_codec"]["U"]), prep["state_norm"]["mean"],
        prep["state_norm"]["std"], checkpoint=checkpoint, seed=seed).to(dev)
    own = dict(mod.named_parameters())
    own.update(dict(mod.named_buffers()))
    with torch.no_grad():
        for k, v in ck["weights"].items():
            if k not in own:
                raise SystemExit(f"в модели нет {k}")
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    if ktr.state_hash(ktr.trainable_state(mod)) != ck["weights_hash"]:
        raise SystemExit("веса после загрузки ≠ отпечатку источника")
    mod.attach_exits()
    return mod, bar


# --- шаг, проверка ------------------------------------------------------------------------
def multi_loss(torch, km, out, tgt, val):
    """-> (общая [B], {d: (L, l1, bce)})."""
    per = {d: km.loss_fn(torch, out["exits"][d], tgt, val)
           for d in k34m.EXITS}
    tot = sum(per[d][0] for d in k34m.EXITS) / len(k34m.EXITS)
    return tot, per


def set_phase(G, phase):
    for k, ps in G.items():
        on = k == "heads" or phase == "main"
        for p in ps:
            p.requires_grad_(on)


def make_opt(torch, G, phase):
    if phase == "warmup":
        return torch.optim.AdamW([dict(params=G["heads"], lr=WARM_LR,
                                       name="heads", base=WARM_LR)],
                                 weight_decay=WD)
    return torch.optim.AdamW([dict(params=G[k], lr=LR[k], name=k,
                                   base=LR[k]) for k in ("expert", "lora",
                                                         "new", "heads")],
                             weight_decay=WD)


def evaluate(torch, km, mod, bar, D, rows, dev, dt, micro, mb, weights=None):
    """По каждой глубине: L, рука-20, рука-8, схват (взвешенно)."""
    was = mod.training
    mod.eval()
    rows = np.asarray(rows)
    acc = {d: dict(L=[], l1_20=[], l1_8=[], grip=[]) for d in k34m.EXITS}
    with torch.no_grad():
        for s0 in range(0, len(rows), micro):
            r = rows[s0:s0 + micro]
            batch, po, st, tgt, val = mb(D, r, bar, dev)
            with torch.autocast("cuda", dtype=dt, enabled=dev.type == "cuda"):
                out = mod.forward_all(batch, po, st)
                _tot, per = multi_loss(torch, km, out, tgt, val)
            v = val.float()
            for d in k34m.EXITS:
                o = out["exits"][d]
                if not (torch.isfinite(per[d][0]).all() and
                        torch.isfinite(o["arm"]).all() and
                        torch.isfinite(o["grip_logit"]).all()):
                    raise SystemExit(f"нечисловая проверка (глубина {d})")
                e = (o["arm"].float() - tgt[..., :6]).abs().mean(-1)
                acc[d]["L"].append(per[d][0].float().cpu().numpy())
                acc[d]["l1_20"].append(((e * v).sum(1) / v.sum(1))
                                       .cpu().numpy())
                v8 = v[:, :8]
                acc[d]["l1_8"].append(((e[:, :8] * v8).sum(1) /
                                       v8.sum(1).clamp_min(1)).cpu().numpy())
                g = ((o["grip_logit"] > 0) == (tgt[..., 6] > 0)).float()
                acc[d]["grip"].append(((g * v).sum(1) / v.sum(1))
                                      .cpu().numpy())
    if was:
        mod.train()
    w = np.ones(len(rows)) if weights is None else np.asarray(weights)
    res = {str(d): {k: float((np.concatenate(x) * w).sum() / w.sum())
                    for k, x in a.items()} for d, a in acc.items()}
    res["mean_L"] = float(np.mean([res[str(d)]["L"] for d in k34m.EXITS]))
    return res


class Trainer:
    def __init__(self, torch, a, D, mode, seed, make_model=None, mb=None,
                 outdir=None, contract_extra=None):
        import k21_model as km
        import k21_train as kt
        import k21_data as d21
        self.torch, self.km, self.a, self.D = torch, km, a, D
        self.mode, self.seed = mode, seed
        self.dev = torch.device(a.device)
        self.dt = getattr(torch, a.dtype)
        self.mb = mb or d21.make_batch
        self.dir = outdir or run_dir(seed, mode)
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, "resume.pt")
        if make_model is None:
            spec = source_entry(seed, mode)["spec"]
            self.mod, self.bar = load_source(torch, mode, seed, spec,
                                             a.device)
            self.source = dict(ckpt_sha1=spec["ckpt_sha1"],
                               weights_hash=spec["weights_hash"],
                               best_update=spec["best_update"])
        else:
            self.mod, self.bar = make_model()
            self.source = dict(fake=True)
        self.G = k34m.groups(self.mod)
        st0 = kt.trainable_state(self.mod)
        self.init_hash = kt.state_hash(st0)
        self.source_hash = kt.state_hash({k: v for k, v in st0.items()
                                          if not k.startswith(
                                              ("readout8.", "readout16."))})
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.dt == torch.float16 and self.dev.type == "cuda"))
        vr = np.asarray(D["val_rows"])
        self.val_rows = vr
        self.val_w = kt.row_weights(D["task"][vr], D["episode"][vr])
        self.extra = contract_extra or {}
        self.st = dict(phase="warmup", u=0, curve=[], best=None,
                       init_hash=self.init_hash, acc={})

    def config(self):
        return dict(
            mode=self.mode, seed=self.seed, architecture=k34m.ARCH[self.mode],
            source=self.source, exits=list(k34m.EXITS), loss=LOSS,
            heads=dict(names=["readout8", "readout16"], init="deepcopy "
                       "readout24 после загрузки источника",
                       n_new=k34m.n_new_params(self.mod.H)),
            micro=self.a.micro, dtype=self.a.dtype, eff_batch=EFF_BATCH,
            warm=dict(updates=WARM_UPDATES, lr=WARM_LR, groups=["heads"]),
            main=dict(updates=MAIN_UPDATES, lr=LR, warm=MAIN_WARM,
                      floor=LR_FLOOR), wd=WD, clip=CLIP, u_offset=U_OFFSET,
            eval_every=EVAL_EVERY, schedule_fp=schedule_fp(
                self.D, self.a.micro, self.seed),
            code={f: sha(os.path.join(HERE, f)) for f in CODE
                  if os.path.exists(os.path.join(HERE, f))}, **self.extra)

    def save(self, opt):
        import k21_train as kt
        torch = self.torch
        st = dict(self.st, weights=kt.trainable_state(self.mod),
                  opt=opt.state_dict(), scaler=self.scaler.state_dict(),
                  rng=torch.random.get_rng_state(), config=self.config())
        torch.save(st, self.state_path + ".tmp")
        os.replace(self.state_path + ".tmp", self.state_path)

    def load(self):
        torch = self.torch
        if not os.path.exists(self.state_path):
            return None
        st = torch.load(self.state_path, map_location="cpu",
                        weights_only=False)
        if st["config"] != self.config():
            diff = sorted(k for k in set(st["config"]) | set(self.config())
                          if st["config"].get(k) != self.config().get(k))
            raise SystemExit(f"{self.state_path}: другой контракт {diff}")
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
        print(f"  K-34 {self.mode} s{self.seed}: продолжение — фаза "
              f"{st['phase']}, обновление {st['u']}", flush=True)
        return opt_state

    def check(self, u):
        import k21_train as kt
        t0 = time.time()
        val = evaluate(self.torch, self.km, self.mod, self.bar, self.D,
                       self.val_rows, self.dev, self.dt, self.a.eval_micro,
                       self.mb, self.val_w)
        tr = {k: {kk: vv / v["n"] for kk, vv in v.items() if kk != "n"}
              for k, v in self.st["acc"].items() if v.get("n")}
        self.st["acc"] = {}
        self.st["curve"].append(dict(update=u, val=val, train=tr,
                                     grad=self.st.get("last_grad"),
                                     seconds=time.time() - t0))
        v = val["mean_L"]
        if self.st["best"] is None or v < self.st["best"][1]:
            self.st["best"] = (u, v)
            st = kt.trainable_state(self.mod)
            p = os.path.join(self.dir, "best.pt")
            self.torch.save(dict(weights=st, update=u, val=val,
                                 config=self.config(),
                                 init_hash=self.init_hash,
                                 weights_hash=kt.state_hash(st)), p + ".tmp")
            os.replace(p + ".tmp", p)
        print(f"  K-34 {self.mode} s{self.seed} main {u}: val L "
              + ", ".join(f"d{d} {val[str(d)]['L']:.4f}" for d in k34m.EXITS)
              + f"; среднее {v:.4f}; лучшая {self.st['best']}", flush=True)

    def step(self, opt, src, idx):
        torch = self.torch
        opt.zero_grad(set_to_none=True)
        accd = self.st["acc"].setdefault(src, {})
        n_seen = 0
        for rows in idx:
            if src == "teacher":
                batch, po, st, tgt, val = self.mb(
                    self.D, self.D["teacher_rows"][rows], self.bar, self.dev,
                    teacher_idx=rows)
            else:
                batch, po, st, tgt, val = self.mb(self.D, rows, self.bar,
                                                  self.dev)
            with torch.autocast("cuda", dtype=self.dt,
                                enabled=self.dev.type == "cuda"):
                out = self.mod.forward_all(batch, po, st)
                tot, per = multi_loss(torch, self.km, out, tgt, val)
            lm = tot.mean() / len(idx)
            if not bool(torch.isfinite(lm)):
                raise SystemExit("нечисловая потеря")
            self.scaler.scale(lm).backward()
            accd["L"] = accd.get("L", 0.0) + float(tot.mean()) / len(idx)
            for d in k34m.EXITS:
                accd[f"L{d}"] = accd.get(f"L{d}", 0.0) + \
                    float(per[d][0].mean()) / len(idx)
            n_seen += len(rows)
        accd["n"] = accd.get("n", 0) + 1
        self.scaler.unscale_(opt)
        ps = [p for g in opt.param_groups for p in g["params"]]
        gn = torch.nn.utils.clip_grad_norm_(ps, CLIP)
        if not bool(torch.isfinite(gn)) and not self.scaler.is_enabled():
            raise SystemExit("нечисловой градиент")
        self.scaler.step(opt)
        self.scaler.update()
        return n_seen, float(gn)

    def run(self, until_main=MAIN_UPDATES):
        torch = self.torch
        opt_state = self.load()
        phase = self.st["phase"]
        set_phase(self.G, phase)
        opt = make_opt(torch, self.G, phase)
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        self.mod.train()
        t_start, n_seen = time.time(), 0
        while True:
            phase = self.st["phase"]
            if phase == "warmup" and self.st["u"] >= WARM_UPDATES:
                self.st.update(phase="main", u=0)
                set_phase(self.G, "main")
                opt = make_opt(torch, self.G, "main")
                print(f"  K-34 {self.mode} s{self.seed}: warmup завершён -> "
                      f"main (новый оптимизатор)", flush=True)
                self.check(0)
                self.save(opt)
                t_start, n_seen = time.time(), 0
                continue
            if phase == "main" and self.st["u"] >= until_main:
                break
            u = self.st["u"]
            u_abs = U_OFFSET + (u if phase == "warmup" else WARM_UPDATES + u)
            if phase == "main":
                for g in opt.param_groups:
                    g["lr"] = lr_main(u, g["base"])
            src, idx = sched_entry(self.D, u_abs, self.a.micro, self.seed)
            ns, gn = self.step(opt, src, idx)
            n_seen += ns
            self.st["u"] = u + 1
            if (u + 1) % EVAL_EVERY == 0 or u < 2:
                self.st["last_grad"] = dict(total=gn, phase=phase)
            if self.st["u"] == 100 and phase == "main":
                el = time.time() - t_start
                mem = torch.cuda.max_memory_allocated(self.dev) / 2 ** 30 \
                    if self.dev.type == "cuda" else 0.0
                rate = n_seen / max(el, 1e-9)
                left = (until_main - 100) * EFF_BATCH / rate / 3600
                print(f"  K-34 {self.mode} s{self.seed}: {rate:.1f} набл./с, "
                      f"память {mem:.1f} ГБ; до {until_main} main ~"
                      f"{left:.1f} ч (без проверок)", flush=True)
            if phase == "main" and self.st["u"] % EVAL_EVERY == 0:
                self.check(self.st["u"])
            if self.st["u"] % SAVE_EVERY == 0:
                self.save(opt)
        self.save(opt)
        return self.st


def train(a):
    import torch
    import k21_data as d21
    import k21_train as kt
    import k23_train as k23t
    if a.mode not in MODES or a.seed not in SEEDS:
        raise SystemExit("режим или seed")
    if a.micro is None:
        import k26_train as k26t          # то же накопление, что у источника
        a.micro = k26t.decided_micro()
        if a.micro is None:
            raise SystemExit("нет решения о microbatch K-26 и не задан "
                             "--micro")
    if EFF_BATCH % a.micro:
        raise SystemExit("эффективный батч 32 не делится на microbatch")
    kt.schedule = k23t.seeded_schedule(kt.schedule, a.seed)
    if not a.no_fastload:
        import k21_fastload as fl
        fl.install()
    D = d21.load_prepared(require_teacher=True)
    tr = Trainer(torch, a, D, a.mode, a.seed)
    print(f"  K-34 {a.mode} s{a.seed}: новых параметров "
          f"{k34m.n_new_params(tr.mod.H)}; warmup {WARM_UPDATES} + main "
          f"{MAIN_UPDATES}; micro {a.micro}", flush=True)
    st = tr.run(MAIN_UPDATES)
    if st["phase"] != "main" or st["u"] != MAIN_UPDATES:
        raise SystemExit("обучение не дошло до конца")
    b = torch.load(os.path.join(tr.dir, "best.pt"), map_location="cpu",
                   weights_only=False)
    rep = dict(mode=a.mode, seed=a.seed, best=st["best"],
               best_update=int(b["update"]),
               best_sha1=sha(os.path.join(tr.dir, "best.pt")),
               best_weights_hash=b["weights_hash"], best_val=b["val"],
               updates=dict(warmup=WARM_UPDATES, main=st["u"]),
               curve=st["curve"], config=tr.config(),
               init_hash=tr.init_hash, source_hash=tr.source_hash)
    os.makedirs("reports/k34", exist_ok=True)
    json.dump(rep, open(report_path(a.seed, a.mode), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"  K-34 {a.mode} s{a.seed}: best main {rep['best_update']}, "
          f"среднее val L {st['best'][1]:.4f}; отчёт "
          f"{report_path(a.seed, a.mode)}")
    return 0


# --- fitcheck ---------------------------------------------------------------------------
def fit_rows(D, n=None, seed=FIT_ROWS_SEED):
    n = FIT_N if n is None else n
    tr = np.asarray(D["train_rows"])
    return np.sort(np.random.default_rng(seed).choice(tr, n, replace=False))


def fitcheck_run(torch, a, D, mod, bar, mb):
    """200 обновлений только ранних голов на 64 строках."""
    import k21_model as km
    import k21_train as kt
    rows = fit_rows(D)
    dev, dt = torch.device(a.device), getattr(torch, a.dtype)
    G = k34m.groups(mod)
    set_phase(G, "warmup")
    st0 = kt.trainable_state(mod)
    src0 = {k: v for k, v in st0.items()
            if not k.startswith(("readout8.", "readout16."))}
    heads0 = {k: v for k, v in st0.items()
              if k.startswith(("readout8.", "readout16."))}
    ev0 = evaluate(torch, km, mod, bar, D, rows, dev, dt, a.eval_micro, mb)
    with torch.no_grad():
        mod.eval()
        b0 = mb(D, rows[:a.eval_micro], bar, dev)
        o24_0 = mod(b0[0], b0[1], b0[2], exit_depth=24)["arm"].clone()
    opt = make_opt(torch, G, "warmup")
    scaler = torch.cuda.amp.GradScaler(
        enabled=(dt == torch.float16 and dev.type == "cuda"))
    rng = np.random.default_rng(FIT_ROWS_SEED + 1)
    mod.train()
    for _u in range(int(FIT_UPDATES)):
        idx = rng.choice(len(rows), EFF_BATCH, replace=True).reshape(
            -1, a.micro)
        opt.zero_grad(set_to_none=True)
        for r in idx:
            batch, po, st, tgt, val = mb(D, rows[r], bar, dev)
            with torch.autocast("cuda", dtype=dt, enabled=dev.type == "cuda"):
                out = mod.forward_all(batch, po, st)
                tot, _per = multi_loss(torch, km, out, tgt, val)
            scaler.scale(tot.mean() / len(idx)).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(G["heads"], CLIP)
        scaler.step(opt)
        scaler.update()
    ev1 = evaluate(torch, km, mod, bar, D, rows, dev, dt, a.eval_micro, mb)
    st1 = kt.trainable_state(mod)
    with torch.no_grad():
        mod.eval()
        o24_1 = mod(b0[0], b0[1], b0[2], exit_depth=24)["arm"]
    drop = {d: 1 - ev1[str(d)]["L"] / ev0[str(d)]["L"] for d in (8, 16)}
    changed = {nm: any(not torch.equal(st1[k], heads0[k]) for k in heads0
                       if k.startswith(nm + "."))
               for nm in ("readout8", "readout16")}
    src_same = all(torch.equal(st1[k], v) for k, v in src0.items())
    out24_same = bool(torch.equal(o24_0, o24_1))
    ok = all(drop[d] >= FIT_DROP for d in drop) and all(changed.values()) \
        and src_same and out24_same
    return dict(rows=rows.tolist(), L_before={d: ev0[str(d)]["L"]
                                              for d in (8, 16, 24)},
                L_after={d: ev1[str(d)]["L"] for d in (8, 16, 24)},
                drop=drop, heads_changed=changed, source_unchanged=src_same,
                out24_unchanged=out24_same, criterion=dict(
                    drop=FIT_DROP, updates=FIT_UPDATES, n=FIT_N,
                    lr=WARM_LR), ok=ok)


def fitcheck(a):
    import torch
    import k21_data as d21
    if a.micro is None:
        import k26_train as k26t
        a.micro = k26t.decided_micro() or 8
    D = d21.load_prepared(require_teacher=False)
    rep = {}
    for mode in MODES:
        spec = source_entry(0, mode)["spec"]
        mod, bar = load_source(torch, mode, 0, spec, a.device)
        r = fitcheck_run(torch, a, D, mod, bar, d21.make_batch)
        rep[mode] = r
        print(f"  fitcheck {mode} s0: снижение L d8 {100 * r['drop'][8]:.1f}%"
              f", d16 {100 * r['drop'][16]:.1f}% (порог "
              f"{100 * FIT_DROP:.0f}%); головы изменились {r['heads_changed']}"
              f"; источник и выход 24 прежние: {r['source_unchanged']}/"
              f"{r['out24_unchanged']} — " + ("пройден" if r["ok"] else
                                             "НЕ ПРОЙДЕН"), flush=True)
        del mod, bar
        torch.cuda.empty_cache()
    rep["ok"] = all(v["ok"] for v in rep.values() if isinstance(v, dict))
    rep["contract"] = stage_contract("fitcheck", a.device, a.micro, a.dtype)
    os.makedirs("reports/k34", exist_ok=True)
    json.dump(rep, open(FITCHECK, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    return 0 if rep["ok"] else 3


# --- контракты этапов и проверка готовых результатов -----------------------------------
REAL_CHECK = "reports/k34/real_check.json"
FITCHECK = "reports/k34/fitcheck.json"
RC_KEYS = ("B1_legacy_equal", "B5_legacy_equal", "B1_d8_vs_all",
           "B1_d16_vs_all", "B5_d8_vs_all", "B5_d16_vs_all", "ms_B5",
           "main_step", "save_load", "peak_gb", "mem_before_load_gb",
           "n_new", "source")


def code_now():
    return {f: sha(os.path.join(HERE, f)) for f in CODE
            if os.path.exists(os.path.join(HERE, f))}


def source_contract():
    if not os.path.exists(PREPARE):
        raise SystemExit(f"нет {PREPARE}")
    p = json.load(open(PREPARE, encoding="utf-8"))
    return {k: {kk: v["spec"][kk] for kk in ("ckpt_sha1", "weights_hash",
                                              "best_update")}
            for k, v in sorted(p["sources"].items())}


def stage_contract(kind, device=None, micro=None, dtype=None):
    """То, что определяет результат этапа: подготовка и источники, код,
    точность, microbatch (fitcheck/обучение), устройство (GPU-проверки)."""
    c = dict(kind=kind, prepare_sha1=sha(PREPARE), sources=source_contract(),
             code=code_now(), dtype=dtype)
    if kind in ("fitcheck", "train"):
        c["micro"] = micro
    if kind in ("real_check", "fitcheck") and device is not None:
        import torch
        c["device_name"] = torch.cuda.get_device_name(torch.device(device))
    return c


def real_check_problems(r, want):
    p = []
    if r.get("contract") != want:
        diff = sorted(k for k in set(want) | set(r.get("contract") or {})
                      if want.get(k) != (r.get("contract") or {}).get(k))
        p.append(f"контракт real-check другой {diff}")
    if r.get("problems") != []:
        p.append("real-check не пройден")
    res = r.get("results") or {}
    tags = [f"{m}_s{s}" for m in MODES for s in SEEDS]
    for t in tags:
        x = res.get(t)
        if not isinstance(x, dict):
            p.append(f"нет результата {t}")
            continue
        miss = [k for k in RC_KEYS if k not in x]
        if miss:
            p.append(f"{t}: нет {miss}")
            continue
        if not (x["B1_legacy_equal"] and x["B5_legacy_equal"]):
            p.append(f"{t}: legacy ≠")
        for k in ("B1_d8_vs_all", "B1_d16_vs_all", "B5_d8_vs_all",
                  "B5_d16_vs_all"):
            e = x[k]
            if not (e.get("arm", 1) <= 1e-6 and e.get("grip", 1) <= 1e-5):
                p.append(f"{t}: {k} {e}")
        ms = x["main_step"]
        if not (ms.get("finite") and ms.get("heads8") and ms.get("heads16")
                and ms.get("frozen_unchanged") and all(
                    (ms.get("changed") or {}).values()) and
                set(ms.get("changed") or {}) == {"expert", "lora", "new",
                                                 "heads"}):
            p.append(f"{t}: main-шаг {ms}")
        if x["save_load"] is not True:
            p.append(f"{t}: save/load различающихся голов")
    return p


def fitcheck_problems(r, want):
    p = []
    if r.get("contract") != want:
        p.append("контракт fitcheck другой")
    if r.get("ok") is not True:
        p.append("fitcheck не пройден")
    for m in MODES:
        x = r.get(m)
        dr = {str(k): v for k, v in ((x or {}).get("drop") or {}).items()} \
            if isinstance(x, dict) else {}
        if not isinstance(x, dict) or x.get("ok") is not True or \
                set(dr) != {"8", "16"} or not all(
                    np.isfinite(v) and v >= FIT_DROP for v in dr.values()) \
                or x.get("source_unchanged") is not True or \
                x.get("out24_unchanged") is not True:
            p.append(f"fitcheck {m}: нет полного пройденного результата")
    return p


def check_fresh(a):
    """Готовый отчёт пройден ровно тем же контрактом и полон -> 0."""
    if a.kind == "real_check":
        path, fn = REAL_CHECK, real_check_problems
        want = stage_contract("real_check", a.device, dtype="float16")
    else:
        path, fn = FITCHECK, fitcheck_problems
        if a.micro is None:
            import k26_train as k26t
            a.micro = k26t.decided_micro() or 8
        want = stage_contract("fitcheck", a.device, a.micro, a.dtype)
    if not os.path.exists(path):
        print(f"  {path}: нет")
        return 1
    pr = fn(json.load(open(path, encoding="utf-8")), want)
    print(f"  {path}: " + ("пройден тем же контрактом" if not pr else
                           "пересчитать — " + "; ".join(pr[:3])))
    return 0 if not pr else 1


def train_problems(seed, mode, micro, dtype):
    """Готовое обучение: отчёт, best.pt и их контракт = текущим."""
    rp = report_path(seed, mode)
    ck = os.path.join(run_dir(seed, mode), "best.pt")
    if not os.path.exists(rp) or not os.path.exists(ck):
        return ["нет отчёта или best.pt"]
    r = json.load(open(rp, encoding="utf-8"))
    p = []
    if sha(ck) != r.get("best_sha1"):
        p.append("best.pt ≠ отчёту")
    c = r.get("config") or {}
    src = source_contract().get(f"{mode}_s{seed}")
    for k, v in (("mode", mode), ("seed", seed), ("micro", micro),
                 ("dtype", dtype), ("code", code_now()),
                 ("architecture", k34m.ARCH[mode])):
        if c.get(k) != v:
            p.append(f"конфигурация {k} ≠ текущей")
    if c.get("source") != src:
        p.append("источник обучения ≠ подготовке")
    if (r.get("updates") or {}).get("main") != MAIN_UPDATES:
        p.append("обучение не дошло до 4000 main")
    return p


def train_valid(a):
    if a.micro is None:
        import k26_train as k26t
        a.micro = k26t.decided_micro()
    pr = train_problems(a.seed, a.mode, a.micro, a.dtype)
    print(f"  обучение {a.mode} s{a.seed}: " + ("готово, контракт тот же"
                                               if not pr else
                                               "; ".join(pr[:3])))
    return 0 if not pr else 1


def trained_ckpt_problems(ck, mode, seed, spec, code=None):
    """Checkpoint K-34 перед использованием: архитектура, seed, источник,
    отпечаток, контракт кода = текущему."""
    c = ck.get("config") or {}
    p = []
    if c.get("architecture") != k34m.ARCH[mode] or int(c.get("seed", -1)) != \
            seed:
        p.append("архитектура или seed")
    if (c.get("source") or {}).get("ckpt_sha1") != \
            spec["source"]["ckpt_sha1"]:
        p.append("источник")
    if ck.get("weights_hash") != spec.get("weights_hash"):
        p.append("отпечаток ≠ плану")
    if c.get("code") != (code if code is not None else code_now()):
        p.append("контракт кода checkpoint ≠ текущему")
    if c.get("exits") != list(k34m.EXITS) or c.get("loss") != LOSS:
        p.append("выходы или потеря")
    return p


# --- real-check --------------------------------------------------------------------------
def _gpu_free(torch):
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def _legacy_outputs(torch, D, rows, tag, spec, device):
    """Исходная рука K-26 -> CPU-выходы B = 1 и 5; модель освобождается
    внутри функции."""
    import k21_data as d21
    import k26_policy as kp26
    dev = torch.device(device)
    ref, bar, _ck, _p = kp26.load_policy(torch, tag, spec, device)
    out = {}
    with torch.no_grad():
        for B in (1, 5):
            batch, po, st, _t, _v = d21.make_batch(D, rows[:B], bar, dev)
            with torch.autocast("cuda", dtype=torch.float16):
                o = ref(batch, po, st)
            out[B] = (o["arm"].float().cpu(), o["grip_logit"].float().cpu())
    return out


def _new_checks(torch, D, rows, mode, seed, spec, device, ref_out):
    """Новый экземпляр K-34: побитовая сверка, усечение, счётчики,
    скорость, main-шаг, сохранение/загрузка различающихся голов. Только
    CPU-результаты наружу."""
    import tempfile
    import k21_data as d21
    import k21_model as km
    import k21_train as kt
    tag = f"{mode}_s{seed}"
    dev = torch.device(device)
    r, prob = {}, []
    r["mem_before_load_gb"] = torch.cuda.memory_allocated(dev) / 2 ** 30
    torch.cuda.reset_peak_memory_stats(dev)
    mod, bar = load_source(torch, mode, seed, spec, device, checkpoint=False)
    mod.eval()
    with torch.no_grad():
        for B in (1, 5):
            batch, po, st, _t, _v = d21.make_batch(D, rows[:B], bar, dev)
            with torch.autocast("cuda", dtype=torch.float16):
                o24 = mod(batch, po, st, exit_depth=24)
                c24 = copy.deepcopy(mod.counters)
                oe, ce = {}, {}
                for d in (8, 16):
                    oe[d] = mod(batch, po, st, exit_depth=d)
                    ce[d] = copy.deepcopy(mod.counters)
                oa = mod.forward_all(batch, po, st)
                ca = copy.deepcopy(mod.counters)
            same = torch.equal(o24["arm"].float().cpu(), ref_out[B][0]) and \
                torch.equal(o24["grip_logit"].float().cpu(), ref_out[B][1])
            r[f"B{B}_legacy_equal"] = bool(same)
            if not same:
                prob.append(f"{tag} B{B}: выход 24 ≠ K-26 побитово")
            for d in (8, 16):
                da = float((oe[d]["arm"] - oa["exits"][d]["arm"]).abs().max())
                dg = float((oe[d]["grip_logit"] -
                            oa["exits"][d]["grip_logit"]).abs().max())
                r[f"B{B}_d{d}_vs_all"] = dict(arm=da, grip=dg)
                if not (da <= 1e-6 and dg <= 1e-5):
                    prob.append(f"{tag} B{B} d{d}: усечённый ≠ forward_all "
                                f"({da:.2e}, {dg:.2e})")
            for d, c in ((24, c24), (8, ce[8]), (16, ce[16])):
                prob += [f"{tag} d{d}: {x}" for x in
                         k34m.counters_problems(mode, c, d)]
            prob += [f"{tag} all: {x}" for x in
                     k34m.counters_problems(mode, ca, 24, True)]
            del o24, oe, oa
    batch, po, st, tgt, val = d21.make_batch(D, rows, bar, dev)
    sp = {}
    with torch.no_grad():
        for d in k34m.EXITS:
            for _ in range(3):
                with torch.autocast("cuda", dtype=torch.float16):
                    mod(batch, po, st, exit_depth=d)
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            for _ in range(10):
                with torch.autocast("cuda", dtype=torch.float16):
                    mod(batch, po, st, exit_depth=d)
            torch.cuda.synchronize(dev)
            sp[d] = (time.perf_counter() - t0) / 10 * 1e3
    r["ms_B5"] = sp
    # один main-шаг
    G = k34m.groups(mod)
    frozen = {n: p.detach().cpu().clone() for n, p in mod.named_parameters()
              if not p.requires_grad}
    before = {k: [p.detach().cpu().clone() for p in G[k]] for k in G}
    set_phase(G, "main")
    opt = make_opt(torch, G, "main")
    mod.train()
    mod.checkpoint = True
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        out = mod.forward_all(batch, po, st)
        tot, _per = multi_loss(torch, km, out, tgt, val)
    tot.mean().backward()
    del out, tot, _per
    fin = all(bool(torch.isfinite(p.grad).all()) for k in G for p in G[k]
              if p.grad is not None)
    opt.step()
    chg = {k: any(not torch.equal(p.detach().cpu(), q) for p, q in
                  zip(G[k], before[k])) for k in G}
    nh = len(list(mod.readout8.parameters()))
    h8 = any(not torch.equal(p.detach().cpu(), q) for p, q in
             zip(mod.readout8.parameters(), before["heads"][:nh]))
    h16 = any(not torch.equal(p.detach().cpu(), q) for p, q in
              zip(mod.readout16.parameters(), before["heads"][nh:]))
    fz = all(torch.equal(p.detach().cpu(), frozen[n]) for n, p in
             mod.named_parameters() if n in frozen)
    del opt, G, before, frozen
    r["main_step"] = dict(finite=bool(fin), changed=chg, heads8=bool(h8),
                          heads16=bool(h16), frozen_unchanged=bool(fz))
    if not (fin and all(chg.values()) and h8 and h16 and fz):
        prob.append(f"{tag}: main-шаг — {r['main_step']}")
    # сохранение и загрузка уже различающихся голов 8/16/24
    mod.eval()
    mod.checkpoint = False
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            o1 = mod.forward_all(batch, po, st)
        ref1 = {d: o1["exits"][d]["arm"].float().cpu() for d in k34m.EXITS}
        del o1
        diff_heads = not torch.equal(mod.readout8.mlp[0].weight,
                                     mod.readout16.mlp[0].weight) and \
            not torch.equal(mod.readout16.mlp[0].weight,
                            mod.readout.mlp[0].weight)
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "s.pt")
            torch.save(kt.trainable_state(mod), f)
            for nm in ("readout8", "readout16", "readout"):
                for p_ in getattr(mod, nm).parameters():
                    p_.add_(0.01)
            own = dict(mod.named_parameters())
            own.update(dict(mod.named_buffers()))
            for k, v in torch.load(f, weights_only=False).items():
                own[k].copy_(v.to(own[k].device, own[k].dtype))
        with torch.autocast("cuda", dtype=torch.float16):
            o2 = mod.forward_all(batch, po, st)
        sl = diff_heads and all(torch.equal(o2["exits"][d]["arm"].float()
                                            .cpu(), ref1[d])
                                for d in k34m.EXITS)
        del o2
    r["save_load"] = bool(sl)
    if not sl:
        prob.append(f"{tag}: сохранение/загрузка различающихся голов")
    r.update(peak_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
             n_new=k34m.n_new_params(mod.H),
             source=dict(ckpt_sha1=spec["ckpt_sha1"],
                         weights_hash=spec["weights_hash"]))
    if r["n_new"] != sum(p.numel() for p in mod.heads_params()):
        prob.append(f"{tag}: число новых параметров ≠ формуле")
    del mod, bar, batch, po, st, tgt, val
    return r, prob


def real_check(a):
    import torch
    import k21_data as d21
    D = d21.load_prepared(require_teacher=False)
    rows = np.asarray(D["val_rows"])[:5]
    rep, prob = {}, []
    for mode in MODES:
        for seed in SEEDS:
            tag = f"{mode}_s{seed}"
            spec = source_entry(seed, mode)["spec"]
            ref_out = _legacy_outputs(torch, D, rows, tag, spec, a.device)
            _gpu_free(torch)
            r, pr = _new_checks(torch, D, rows, mode, seed, spec, a.device,
                                ref_out)
            _gpu_free(torch)
            r["mem_after_free_gb"] = torch.cuda.memory_allocated(
                torch.device(a.device)) / 2 ** 30
            rep[tag], prob = r, prob + pr
            print(f"  real-check {tag}: legacy B1/B5 {r['B1_legacy_equal']}/"
                  f"{r['B5_legacy_equal']}, d8/d16 к forward_all "
                  f"{r['B5_d8_vs_all']}/{r['B5_d16_vs_all']}, мс B5 "
                  + ", ".join(f"d{d} {v:.1f}" for d, v in r["ms_B5"].items())
                  + f", память до загрузки {r['mem_before_load_gb']:.2f} ГБ, "
                  f"пик {r['peak_gb']:.1f} ГБ, после освобождения "
                  f"{r['mem_after_free_gb']:.2f} ГБ; save/load "
                  f"{r['save_load']}", flush=True)
    out = dict(problems=prob, results=rep,
               contract=stage_contract("real_check", a.device,
                                       dtype="float16"))
    os.makedirs("reports/k34", exist_ok=True)
    json.dump(out, open(REAL_CHECK, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print("  real-check K-34 " + ("пройден" if not prob else "НЕ пройден: "
                                   + "; ".join(prob[:4])))
    return 0 if not prob else 3


# --- самопроверка ---------------------------------------------------------------------
def reuse_regression():
    """Готовые результаты принимаются только своего контракта и полными."""
    import tempfile
    want = dict(kind="real_check", prepare_sha1="P", sources={"a": 1},
                code={"x": "1"}, dtype="float16", device_name="V100")
    full = {f"{m}_s{s_}": dict(
        B1_legacy_equal=True, B5_legacy_equal=True,
        B1_d8_vs_all=dict(arm=0.0, grip=0.0),
        B1_d16_vs_all=dict(arm=0.0, grip=0.0),
        B5_d8_vs_all=dict(arm=0.0, grip=0.0),
        B5_d16_vs_all=dict(arm=0.0, grip=0.0), ms_B5={}, peak_gb=1.0,
        mem_before_load_gb=0.0, n_new=1, source={},
        main_step=dict(finite=True, heads8=True, heads16=True,
                       frozen_unchanged=True, changed=dict(
                           expert=True, lora=True, new=True, heads=True)),
        save_load=True) for m in MODES for s_ in SEEDS}
    ok = dict(problems=[], results=full, contract=want)
    assert real_check_problems(ok, want) == []
    assert real_check_problems(dict(ok, results={}), want)       # пустой
    assert real_check_problems(ok, dict(want, device_name="A100"))
    assert real_check_problems(ok, dict(want, prepare_sha1="Q"))
    bad = copy.deepcopy(full)
    bad["flat_attn24_s1"]["save_load"] = False
    assert real_check_problems(dict(ok, results=bad), want)
    bad = copy.deepcopy(full)
    del bad["parents_attn24_s0"]["main_step"]
    assert real_check_problems(dict(ok, results=bad), want)
    fw = dict(kind="fitcheck", micro=8)
    fr = {m: dict(ok=True, drop={"8": 0.5, "16": 0.4},
                  source_unchanged=True, out24_unchanged=True)
          for m in MODES}
    assert fitcheck_problems(dict(fr, ok=True, contract=fw), fw) == []
    assert fitcheck_problems(dict(fr, ok=True, contract=fw),
                             dict(fw, micro=4))
    fr2 = copy.deepcopy(fr)
    fr2["flat_attn24"]["drop"]["16"] = float("nan")
    assert fitcheck_problems(dict(fr2, ok=True, contract=fw), fw)
    # checkpoint с чужим контрактом кода — отказ до использования
    spec = dict(source=dict(ckpt_sha1="S"), weights_hash="W")
    ck = dict(weights_hash="W", config=dict(
        architecture=k34m.ARCH["parents_attn24"], seed=0,
        source=dict(ckpt_sha1="S"), code={"k": "1"}, exits=list(k34m.EXITS),
        loss=LOSS))
    assert trained_ckpt_problems(ck, "parents_attn24", 0, spec,
                                 code={"k": "1"}) == []
    assert trained_ckpt_problems(ck, "parents_attn24", 0, spec,
                                 code={"k": "2"})
    assert trained_ckpt_problems(ck, "parents_attn24", 1, spec,
                                 code={"k": "1"})
    assert trained_ckpt_problems(dict(ck, weights_hash="V"),
                                 "parents_attn24", 0, spec, code={"k": "1"})
    # готовое обучение: только с тем же контрактом и полным отчётом
    g = globals()
    keep = {k: g[k] for k in ("run_dir", "report_path", "source_contract",
                              "code_now")}
    with tempfile.TemporaryDirectory() as td:
        try:
            g["run_dir"] = lambda s_, m: os.path.join(td, m)
            g["report_path"] = lambda s_, m: os.path.join(td, m + ".json")
            g["source_contract"] = lambda: {"parents_attn24_s0": {"c": 1}}
            g["code_now"] = lambda: {"k": "1"}
            os.makedirs(os.path.join(td, "parents_attn24"))
            bp = os.path.join(td, "parents_attn24", "best.pt")
            open(bp, "wb").write(b"w")
            rep = dict(best_sha1=sha(bp), updates=dict(main=MAIN_UPDATES),
                       config=dict(mode="parents_attn24", seed=0, micro=8,
                                   dtype="float16", code={"k": "1"},
                                   architecture=k34m.ARCH["parents_attn24"],
                                   source={"c": 1}))
            json.dump(rep, open(os.path.join(td, "parents_attn24.json"), "w"))
            assert train_problems(0, "parents_attn24", 8, "float16") == []
            assert train_problems(0, "parents_attn24", 4, "float16")
            g["code_now"] = lambda: {"k": "2"}
            assert train_problems(0, "parents_attn24", 8, "float16")
            g["code_now"] = lambda: {"k": "1"}
            open(bp, "wb").write(b"x")                      # подмена best.pt
            assert train_problems(0, "parents_attn24", 8, "float16")
        finally:
            g.update(keep)


def selftest():
    import tempfile
    import types
    import torch
    import k21_train as kt
    import k26_model as k26m
    M = k34m.build(torch)
    bar0 = k26m.fake_bar(torch)
    L, U = -np.ones(6) * 1.1, np.ones(6) * 4.0
    nr = 64
    DT = dict(train_rows=np.arange(48), val_rows=np.arange(48, 56),
              teacher_rows=np.arange(48, 56), task=np.arange(nr) % 4,
              episode=np.arange(nr) % 8)

    def fake_mb(D_, rows, bar, dev, teacher_idx=None):
        rows = np.asarray(rows)
        b = k26m.fake_batch(torch, B=len(rows), seed=int(rows.sum()) % 1000)
        g = torch.Generator().manual_seed(int(rows.sum()) + 7)
        st_ = torch.randn(len(rows), 11, generator=g)
        tgt = torch.rand(len(rows), 20, 7, generator=g) * 2
        tgt[..., 6] = torch.where(tgt[..., 6] > 1, 1.0, -1.0)
        return b, 1, st_, tgt, torch.ones(len(rows), 20, dtype=torch.bool)

    def make_model_for(mode):
        def mk():
            m = M["make"](mode, copy.deepcopy(bar0), L, U, seed=1,
                          checkpoint=False)
            m.attach_exits()
            return m, dict(model=m.bar)
        return mk
    # LR и очередь
    assert lr_main(0, 1.0) < lr_main(199, 1.0) <= 1.0
    lrs = [lr_main(u, 1.0) for u in range(200, MAIN_UPDATES + 1)]
    assert all(b <= a_ + 1e-12 for a_, b in zip(lrs, lrs[1:]))
    assert abs(lr_main(MAIN_UPDATES, 1.0) - LR_FLOOR) < 1e-9
    s0 = sched_entry(DT, U_OFFSET + 3, 8, 0)
    s1 = sched_entry(DT, U_OFFSET + 3, 8, 1)
    assert s0[0] == "teacher" and sched_entry(DT, U_OFFSET, 8, 0)[0] == \
        "expert"
    assert not np.array_equal(s0[1], s1[1])               # seed различает
    assert np.array_equal(sched_entry(DT, U_OFFSET + 3, 8, 0)[1], s0[1])
    assert schedule_fp(DT, 8, 0) == schedule_fp(DT, 8, 0) != \
        schedule_fp(DT, 8, 1)
    g = globals()
    keep = {k: g[k] for k in ("WARM_UPDATES", "MAIN_UPDATES", "EVAL_EVERY",
                              "SAVE_EVERY", "EFF_BATCH", "FIT_UPDATES",
                              "FIT_N")}
    a_ = types.SimpleNamespace(device="cpu", dtype="float32", micro=4,
                               eval_micro=4)
    try:
        g.update(WARM_UPDATES=3, MAIN_UPDATES=4, EVAL_EVERY=2, SAVE_EVERY=1,
                 EFF_BATCH=8)
        for mode in MODES:
            with tempfile.TemporaryDirectory() as td:
                # warmup меняет только головы; main — допустимые группы
                tr = Trainer(torch, a_, DT, mode, 1, make_model_for(mode),
                             fake_mb, os.path.join(td, "w"))
                src0 = {k: v for k, v in kt.trainable_state(tr.mod).items()
                        if not k.startswith(("readout8.", "readout16."))}
                set_phase(tr.G, "warmup")
                opt = make_opt(torch, tr.G, "warmup")
                tr.mod.train()
                h0 = [p.detach().clone() for p in tr.G["heads"]]
                tr.step(opt, *sched_entry(DT, U_OFFSET, 4, 1))
                s1_ = kt.trainable_state(tr.mod)
                assert all(torch.equal(s1_[k], v) for k, v in src0.items()), \
                    "warmup изменил источник"
                assert any(not torch.equal(p, q) for p, q in
                           zip(tr.G["heads"], h0))
                set_phase(tr.G, "main")
                opt = make_opt(torch, tr.G, "main")
                b0 = {k: [p.detach().clone() for p in tr.G[k]]
                      for k in tr.G}
                tr.step(opt, *sched_entry(DT, U_OFFSET + 1, 4, 1))
                for k in ("expert", "lora", "new", "heads"):
                    assert any(not torch.equal(p, q) for p, q in
                               zip(tr.G[k], b0[k])), (mode, k)
                # непрерывный прогон = прерванный и продолженный
                res = {}
                for name, stops in (("cont", (4,)), ("resume", (2, 4))):
                    for until in stops:
                        t = Trainer(torch, a_, DT, mode, 1,
                                    make_model_for(mode), fake_mb,
                                    os.path.join(td, name))
                        t.run(until)
                    res[name] = (kt.state_hash(kt.trainable_state(t.mod)),
                                 [(c["update"], round(c["val"]["mean_L"], 10))
                                  for c in t.st["curve"]], t.st["best"])
                assert res["cont"] == res["resume"], (mode, res)
                assert [c[0] for c in res["cont"][1]] == [0, 2, 4]
                # best — минимум среднего трёх глубин, при равенстве ранний
                st_ = torch.load(os.path.join(td, "cont", "resume.pt"),
                                 weights_only=False)
                means = [c["val"]["mean_L"] for c in st_["curve"]]
                bi = int(np.argmin(means))
                assert st_["best"] == (st_["curve"][bi]["update"], means[bi])
                b = torch.load(os.path.join(td, "cont", "best.pt"),
                               weights_only=False)
                assert b["update"] == st_["best"][0]
                for d in k34m.EXITS:
                    assert np.isfinite(st_["curve"][-1]["val"][str(d)]["L"])
                # чужой контракт — отказ до записи
                t = Trainer(torch, a_, DT, mode, 1, make_model_for(mode),
                            fake_mb, os.path.join(td, "cont"),
                            contract_extra=dict(other=1))
                h = sha(os.path.join(td, "cont", "resume.pt"))
                try:
                    t.run(4)
                    raise AssertionError("чужой контракт принят")
                except SystemExit:
                    pass
                assert sha(os.path.join(td, "cont", "resume.pt")) == h
                # fitcheck на игрушке: источник и выход 24 прежние
                g.update(FIT_UPDATES=5, FIT_N=8)
                m, bar = make_model_for(mode)()
                DF = dict(DT, train_rows=np.arange(48))
                r = fitcheck_run(torch, a_, DF, m, bar, fake_mb)
                assert r["source_unchanged"] and r["out24_unchanged"] and \
                    all(r["heads_changed"].values()), r
                g.update(FIT_UPDATES=keep["FIT_UPDATES"], FIT_N=keep["FIT_N"])
    finally:
        g.update(keep)
    reuse_regression()
    print("самопроверка k34_train пройдена: LR (разогрев, косинус до 0.1 на "
          "4000), очередь u_abs = 200000 + u (учитель по u_abs mod 4, seed "
          "различает, отпечаток), warmup меняет только головы 8/16, main — "
          "expert/LoRA/new/heads, прерванный и продолженный прогон = "
          "непрерывному, best — минимум среднего трёх глубин, чужой контракт "
          "— отказ без записи, fitcheck не меняет источник и выход 24")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-34: обучение")
    ap.add_argument("cmd", choices=("train", "fitcheck", "real-check",
                                    "check-fresh", "train-valid"))
    ap.add_argument("--kind", choices=("real_check", "fitcheck"))
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--micro", type=int, default=None,
                    help="по умолчанию — решение K-26 (reports/k26/micro.json)")
    ap.add_argument("--eval-micro", type=int, default=8)
    ap.add_argument("--no-fastload", action="store_true")
    a = ap.parse_args()
    try:
        return dict(train=train, fitcheck=fitcheck,
                    **{"real-check": real_check, "check-fresh": check_fresh,
                       "train-valid": train_valid})[a.cmd](a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
