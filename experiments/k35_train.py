#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-35: адаптация общего декодера смешанных срезов (K35-A, seed 0).

Источник — K-34 T0 (parents_attn24 s0), best.pt K-34 (best main 3000) по
закреплённой подготовке reports/k35/prepare.json: K-35-модель строится
фабрикой k35_model, головы attach_exits, затем веса best.pt K-34 со
сверкой отпечатка и контракта (k34_train.trained_ckpt_problems).
Потеря — k21_model.loss_fn ОДНОГО итогового смешанного чанка.
Срезы: логическая строка g = u_abs·32 + j эффективного батча получает c =
perm_group[g % 36], perm_group = default_rng([35, seed, g // 36])
.permutation(36) — до microbatch-разбиения (смена micro не меняет c).
Режим: warmup 500 — только три головы (readout8/16/24), lr 3e−4;
main 4000 — новый AdamW: expert 2e−5, LoRA 1e−5, new 1e−4, heads 1e−4;
разогрев 200, косинус до 0.1; wd 0.01, clip 1.0; u_abs = 300000 + u; учитель
по u_abs mod 4 (k21_train.schedule с seed 0).
Проверка: панель checkpoint-val (до 8 строк на задачу), шесть срезов
[0, 7, 14, 21, 28, 35]; main 0 и каждые 250; best — минимум среднего L
шести срезов (равные веса задач), ничья — ранний шаг.
fitcheck: 64 train-строки, 200 обновлений трёх голов, срезы по правилу
выше; средний L 36 срезов снизился ≥ 30%, все три головы изменились, всё
прочее побитово прежнее, конечность.
real-check: source_forward фабрики K-35 = настоящая рука K-34 T0 побитово
(B = 1, 5); 36 принудительных c (B = 1, 5, смешанные c внутри батча):
формы, конечность, счётчики, сумма весов; mapped-router на сохранённых
входах — медленный эталон (5e−6); сборка из сохранённых partial y
(y/grip 1e−5, arm 1e−6); один main-шаг; save/load различающихся голов;
выбор micro из 32/16/8 — первый без OOM с пиком ≤ 14 ГиБ.
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

import k35_model as k35m  # noqa: E402
import k35_oracle as ko  # noqa: E402
import k35_topology as tp  # noqa: E402

SEED = 0
EFF_BATCH = 32
WARM_UPDATES, WARM_LR = 500, 3e-4
MAIN_UPDATES = 4000
LR = dict(expert=2e-5, lora=1e-5, new=1e-4, heads=1e-4)
MAIN_WARM, LR_FLOOR = 200, 0.1
WD, CLIP = 0.01, 1.0
EVAL_EVERY = SAVE_EVERY = 250
U_OFFSET = 300_000
FIT_N, FIT_UPDATES, FIT_DROP = 64, 200, 0.30
FIT_ROWS_SEED = 35009
MICROS = (32, 16, 8)
PEAK_LIMIT_GB = 14.0
PREPARE = "reports/k35/prepare.json"
RUN_DIR = "data/k35/runs/s0"
REPORT = "reports/k35/train_s0.json"
REAL_CHECK = "reports/k35/real_check.json"
FITCHECK = "reports/k35/fitcheck.json"
LOSS = "k21_model.loss_fn(смешанный чанк): L1 руки + 0.1·BCE схвата"
CODE = ("k35_topology.py", "k35_model.py", "k35_train.py", "k35_oracle.py",
        "k34_model.py", "k34_train.py", "k26_model.py", "k26_attention.py",
        "k22_model.py", "k21_model.py", "k21_tree.py", "k21_train.py",
        "k21_data.py", "k20_data.py", "k21_fastload.py", "k23_train.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def code_now():
    return {f: sha(os.path.join(HERE, f)) for f in CODE
            if os.path.exists(os.path.join(HERE, f))}


def lr_main(u, base):
    if u < MAIN_WARM:
        return base * (u + 1) / MAIN_WARM
    t = (u - MAIN_WARM) / max(MAIN_UPDATES - MAIN_WARM, 1)
    return base * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(
        math.pi * min(t, 1.0))))


def cuts_for(u_abs, n=None, seed=SEED):
    """c для логических строк эффективного батча обновления u_abs."""
    n = EFF_BATCH if n is None else n
    out = []
    for j in range(n):
        g = u_abs * n + j
        perm = np.random.default_rng([35, seed, g // 36]).permutation(36)
        out.append(int(perm[g % 36]))
    return np.asarray(out, np.int64)


def sched_entry(D, u_abs, micro):
    import k21_train as kt
    return kt.schedule(D, u_abs + 1, micro, seed=SEED, start=u_abs)[0]


def schedule_fp(D, micro, points=(0, 1, 2, 3, 499, 500, 4499)):
    h = hashlib.sha1()
    for u in points:
        src, idx = sched_entry(D, U_OFFSET + u, micro)
        h.update(src.encode())
        h.update(np.ascontiguousarray(np.asarray(idx, np.int64)).tobytes())
        h.update(cuts_for(U_OFFSET + u).tobytes())
    return h.hexdigest()


# --- источник ---------------------------------------------------------------------
def prepare_entry():
    if not os.path.exists(PREPARE):
        raise SystemExit(f"нет {PREPARE}: сначала k35_protocol.py prepare")
    return json.load(open(PREPARE, encoding="utf-8"))


def load_k35(torch, device, checkpoint=True, prep=None):
    """K-35 на весах best.pt K-34 T0 (сверка отпечатка и контракта)."""
    import k21_data as d21
    import k21_train as ktr
    import k34_train as k34t
    prep = prep or prepare_entry()
    src = prep["source"]
    if sha(src["ckpt"]) != src["ckpt_sha1"]:
        raise SystemExit("best.pt K-34 ≠ подготовке K-35")
    ck = torch.load(src["ckpt"], map_location="cpu", weights_only=False)
    pr = k34t.trained_ckpt_problems(ck, "parents_attn24", 0, src["spec"])
    if pr:
        raise SystemExit("источник K-34: " + "; ".join(pr))
    p21 = json.load(open(d21.REPORT, encoding="utf-8"))
    if p21.get("prep_id") != prep["prep_id"]:
        raise SystemExit("подготовка K-21 ≠ закреплённой")
    dev = torch.device(device)
    bar = d21.load_bar(dev, p21["bar_pin"])
    m = bar["model"]
    m.action_expert.float()
    mod = k35m.build(torch)["make"](
        m, np.asarray(p21["bounds_codec"]["L"]),
        np.asarray(p21["bounds_codec"]["U"]), p21["state_norm"]["mean"],
        p21["state_norm"]["std"], checkpoint=checkpoint, seed=0).to(dev)
    mod.attach_exits()
    own = dict(mod.named_parameters())
    own.update(dict(mod.named_buffers()))
    with torch.no_grad():
        for k, v in ck["weights"].items():
            if k not in own:
                raise SystemExit(f"в модели нет {k}")
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    if ktr.state_hash(ktr.trainable_state(mod)) != ck["weights_hash"]:
        raise SystemExit("веса после загрузки ≠ отпечатку best.pt K-34")
    return mod, bar


# --- шаг, проверка ----------------------------------------------------------------
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


def evaluate(torch, km, mod, bar, D, rows, dev, dt, micro, mb, cuts):
    """Средние по срезам cuts: L, рука-20, рука-8, схват; равные веса задач."""
    was = mod.training
    mod.eval()
    rows = np.asarray(rows)
    res = {}
    with torch.no_grad():
        for c in cuts:
            acc = {k: [] for k in ("full", "arm20", "arm8", "grip")}
            for s0 in range(0, len(rows), micro):
                r = rows[s0:s0 + micro]
                batch, po, st, tgt, val = mb(D, r, bar, dev)
                with torch.autocast("cuda", dtype=dt,
                                    enabled=dev.type == "cuda"):
                    o = mod(batch, po, st, cuts=[c] * len(r))
                rc = ko.row_costs(torch, km, o, tgt, val)
                for k in acc:
                    acc[k].append(rc[k])
            res[str(c)] = {k: ko.task_mean(np.concatenate(v), rows, D["task"])
                           for k, v in acc.items()}
            if not all(np.isfinite(v) for v in res[str(c)].values()):
                raise SystemExit(f"нечисловая проверка (c = {c})")
    if was:
        mod.train()
    res["mean_L"] = float(np.mean([res[str(c)]["full"] for c in cuts]))
    return res


class Trainer:
    def __init__(self, torch, a, D, panel_rows, make_model=None, mb=None,
                 outdir=None, contract_extra=None):
        import k21_data as d21
        import k21_model as km
        import k21_train as kt
        self.torch, self.km, self.a, self.D = torch, km, a, D
        self.dev = torch.device(a.device)
        self.dt = getattr(torch, a.dtype)
        self.mb = mb or d21.make_batch
        self.dir = outdir or RUN_DIR
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, "resume.pt")
        self.panel = np.asarray(panel_rows)
        if make_model is None:
            self.mod, self.bar = load_k35(torch, a.device)
            p = prepare_entry()
            self.source = dict(ckpt_sha1=p["source"]["ckpt_sha1"],
                               weights_hash=p["source"]["weights_hash"])
        else:
            self.mod, self.bar = make_model()
            self.source = dict(fake=True)
        self.G = k35m.groups(self.mod)
        self.init_hash = kt.state_hash(kt.trainable_state(self.mod))
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.dt == torch.float16 and self.dev.type == "cuda"))
        self.extra = contract_extra or {}
        self.st = dict(phase="warmup", u=0, curve=[], best=None,
                       init_hash=self.init_hash, acc={})

    def config(self):
        return dict(source=self.source, loss=LOSS, micro=self.a.micro,
                    dtype=self.a.dtype, eff_batch=EFF_BATCH,
                    warm=dict(updates=WARM_UPDATES, lr=WARM_LR,
                              groups=["heads"]),
                    main=dict(updates=MAIN_UPDATES, lr=LR, warm=MAIN_WARM,
                              floor=LR_FLOOR), wd=WD, clip=CLIP,
                    u_offset=U_OFFSET, seed=SEED, ckpt_cuts=list(ko.CKPT_CUTS),
                    panel_sha1=ko.sha_arr(self.panel),
                    schedule_fp=schedule_fp(self.D, self.a.micro),
                    topology=dict(n_cuts=tp.N_CUTS, pairs=list(tp.PAIRS)),
                    code=code_now(), **self.extra)

    def save(self, opt):
        import k21_train as kt
        torch = self.torch
        st = dict(self.st, weights=kt.trainable_state(self.mod),
                  opt=opt.state_dict(), scaler=self.scaler.state_dict(),
                  rng=torch.random.get_rng_state(),
                  cuda_rng=(torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available() else None),
                  config=self.config())
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
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        self.scaler.load_state_dict(st["scaler"])
        opt_state = st.pop("opt")
        for k in ("weights", "scaler", "rng", "cuda_rng", "config"):
            st.pop(k, None)
        self.st = st
        print(f"  K-35: продолжение — фаза {st['phase']}, обновление "
              f"{st['u']}", flush=True)
        return opt_state

    def check(self, u):
        import k21_train as kt
        val = evaluate(self.torch, self.km, self.mod, self.bar, self.D,
                       self.panel, self.dev, self.dt, self.a.eval_micro,
                       self.mb, ko.CKPT_CUTS)
        tr = {k: v / max(self.st["acc"].get("n", 1), 1)
              for k, v in self.st["acc"].items() if k != "n"}
        self.st["acc"] = {}
        self.st["curve"].append(dict(update=u, val=val, train=tr))
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
        print(f"  K-35 main {u}: панель L (6 срезов) {v:.4f}; лучшая "
              f"{self.st['best']}", flush=True)

    def step(self, opt, src, idx, cuts):
        torch = self.torch
        opt.zero_grad(set_to_none=True)
        n_seen, j = 0, 0
        for rows in idx:
            k = len(rows)
            cs = cuts[j:j + k]
            j += k
            if src == "teacher":
                batch, po, st, tgt, val = self.mb(
                    self.D, self.D["teacher_rows"][rows], self.bar, self.dev,
                    teacher_idx=rows)
            else:
                batch, po, st, tgt, val = self.mb(self.D, rows, self.bar,
                                                  self.dev)
            with torch.autocast("cuda", dtype=self.dt,
                                enabled=self.dev.type == "cuda"):
                out = self.mod(batch, po, st, cuts=cs)
                loss = self.km.loss_fn(torch, out, tgt, val)[0]
            lm = loss.mean() / len(idx)
            if not bool(torch.isfinite(lm)):
                raise SystemExit("нечисловая потеря")
            self.scaler.scale(lm).backward()
            self.st["acc"]["L"] = self.st["acc"].get("L", 0.0) + \
                float(loss.mean()) / len(idx)
            n_seen += k
        if j != len(cuts):
            raise SystemExit("срезы не совпали с числом строк")
        self.st["acc"]["n"] = self.st["acc"].get("n", 0) + 1
        self.scaler.unscale_(opt)
        ps = [p for g in opt.param_groups for p in g["params"]]
        gn = torch.nn.utils.clip_grad_norm_(ps, CLIP)
        if not bool(torch.isfinite(gn)) and not self.scaler.is_enabled():
            raise SystemExit("нечисловой градиент")
        self.scaler.step(opt)
        self.scaler.update()
        return n_seen

    def run(self, until_main=None):
        until_main = MAIN_UPDATES if until_main is None else until_main
        torch = self.torch
        opt_state = self.load()
        phase = self.st["phase"]
        set_phase(self.G, phase)
        opt = make_opt(torch, self.G, phase)
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        self.mod.train()
        t0, seen = time.time(), 0
        while True:
            phase = self.st["phase"]
            if phase == "warmup" and self.st["u"] >= WARM_UPDATES:
                self.st.update(phase="main", u=0)
                set_phase(self.G, "main")
                opt = make_opt(torch, self.G, "main")
                print("  K-35: warmup завершён -> main (новый оптимизатор)",
                      flush=True)
                self.check(0)
                self.save(opt)
                t0, seen = time.time(), 0
                continue
            if phase == "main" and self.st["u"] >= until_main:
                break
            u = self.st["u"]
            u_abs = U_OFFSET + (u if phase == "warmup" else WARM_UPDATES + u)
            if phase == "main":
                for g in opt.param_groups:
                    g["lr"] = lr_main(u, g["base"])
            src, idx = sched_entry(self.D, u_abs, self.a.micro)
            if np.asarray(idx).size != EFF_BATCH:
                raise SystemExit("очередь k21 ≠ эффективному батчу K-35")
            seen += self.step(opt, src, idx, cuts_for(u_abs))
            self.st["u"] = u + 1
            if self.st["u"] == 100 and phase == "main":
                el = time.time() - t0
                mem = torch.cuda.max_memory_allocated(self.dev) / 2 ** 30 \
                    if self.dev.type == "cuda" else 0.0
                rate = seen / max(el, 1e-9)
                print(f"  K-35: {rate:.1f} набл./с, память {mem:.1f} ГБ; до "
                      f"{until_main} main ~"
                      f"{(until_main - 100) * EFF_BATCH / rate / 3600:.1f} ч",
                      flush=True)
            if phase == "main" and self.st["u"] % EVAL_EVERY == 0:
                self.check(self.st["u"])
            if self.st["u"] % SAVE_EVERY == 0:
                self.save(opt)
        self.save(opt)
        return self.st


def micro_now():
    if os.path.exists(REAL_CHECK):
        r = json.load(open(REAL_CHECK, encoding="utf-8"))
        if r.get("problems") == [] and r.get("micro"):
            return int(r["micro"])
    return None


def fresh_problems(kind):
    """Отчёт real-check / fitcheck пройден и с текущим контрактом."""
    path = dict(real_check=REAL_CHECK, fitcheck=FITCHECK)[kind]
    if not os.path.exists(path):
        return [f"нет {path}"]
    r = json.load(open(path, encoding="utf-8"))
    p = []
    ok = r.get("problems") == [] if kind == "real_check" else \
        r.get("ok") is True
    if not ok:
        p.append(f"{kind} не пройден")
    c = r.get("contract") or {}
    if c.get("code") != code_now():
        p.append(f"{kind}: код изменён после проверки")
    if not os.path.exists(PREPARE) or c.get("prepare_sha1") != sha(PREPARE):
        p.append(f"{kind}: другая подготовка")
    if kind == "fitcheck" and c.get("micro") != micro_now():
        p.append("fitcheck: micro ≠ решению real-check")
    return p


def resources_problems(torch, dev):
    """Перед обучением: ≥ 14 ГиБ свободной GPU и ≥ пик micro + 2 ГиБ;
    ≥ 12 ГБ диска."""
    import shutil
    r = json.load(open(REAL_CHECK, encoding="utf-8"))
    peak = (r["results"].get("micro_profile") or {}).get(str(r["micro"]))
    if peak is None:
        peak = (r["results"].get("micro_profile") or {}).get(r["micro"])
    free = torch.cuda.mem_get_info(dev)[0] / 2 ** 30
    need = max(14.0, float(peak or 0.0) + 2.0)
    p = []
    if free < need:
        p.append(f"свободно GPU {free:.1f} ГиБ < {need:.1f}")
    disk = shutil.disk_usage(".").free / 1e9
    if disk < 12:
        p.append(f"свободно диска {disk:.1f} ГБ < 12")
    return p


def train(a):
    import torch
    import k21_data as d21
    import k21_train as kt
    import k23_train as k23t
    pr = fresh_problems("real_check") + fresh_problems("fitcheck")
    if pr:
        raise SystemExit("; ".join(pr))
    a.micro = a.micro or micro_now()
    if not a.micro:
        raise SystemExit("нет решения о microbatch (real-check)")
    if a.micro != micro_now():
        raise SystemExit("micro ≠ решению real-check")
    pr = resources_problems(torch, torch.device(a.device))
    if pr:
        raise SystemExit("; ".join(pr))
    kt.schedule = k23t.seeded_schedule(kt.schedule, SEED)
    if not a.no_fastload:
        import k21_fastload as fl
        fl.install()
    D = d21.load_prepared(require_teacher=True)
    pan = json.load(open(ko.PANELS, encoding="utf-8"))
    tr = Trainer(torch, a, D, pan["checkpoint_val"]["rows"])
    st = tr.run()
    if st["phase"] != "main" or st["u"] != MAIN_UPDATES:
        raise SystemExit("обучение не дошло до конца")
    b = torch.load(os.path.join(tr.dir, "best.pt"), map_location="cpu",
                   weights_only=False)
    rep = dict(best=st["best"], best_update=int(b["update"]),
               best_sha1=sha(os.path.join(tr.dir, "best.pt")),
               best_weights_hash=b["weights_hash"], best_val=b["val"],
               updates=dict(warmup=WARM_UPDATES, main=st["u"]),
               curve=st["curve"], config=tr.config(), init_hash=tr.init_hash)
    os.makedirs("reports/k35", exist_ok=True)
    json.dump(rep, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=float)
    print(f"  K-35: best main {rep['best_update']}, панель L {st['best'][1]:.4f}"
          f"; отчёт {REPORT}")
    return 0


def train_problems(micro, dtype):
    if not os.path.exists(REPORT):
        return ["нет отчёта обучения"]
    r = json.load(open(REPORT, encoding="utf-8"))
    ck = os.path.join(RUN_DIR, "best.pt")
    p = []
    if not os.path.exists(ck) or sha(ck) != r.get("best_sha1"):
        p.append("best.pt ≠ отчёту")
    c = r.get("config") or {}
    for k, v in (("micro", micro), ("dtype", dtype), ("code", code_now())):
        if c.get(k) != v:
            p.append(f"конфигурация {k} ≠ текущей")
    if (r.get("updates") or {}).get("main") != MAIN_UPDATES:
        p.append("обучение не дошло до 4000 main")
    return p


def load_trained(torch, device, checkpoint=False):
    """K-35 best (общий декодер A/C/N) со сверкой отпечатка."""
    import k21_train as ktr
    r = json.load(open(REPORT, encoding="utf-8"))
    ck_path = os.path.join(RUN_DIR, "best.pt")
    if sha(ck_path) != r["best_sha1"]:
        raise SystemExit("best.pt K-35 ≠ отчёту")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    if ck["config"].get("code") != code_now():
        raise SystemExit("контракт кода best.pt K-35 ≠ текущему")
    mod, bar = load_k35(torch, device, checkpoint=checkpoint)
    own = dict(mod.named_parameters())
    own.update(dict(mod.named_buffers()))
    with torch.no_grad():
        for k, v in ck["weights"].items():
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    if ktr.state_hash(ktr.trainable_state(mod)) != ck["weights_hash"]:
        raise SystemExit("веса после загрузки ≠ отпечатку best.pt K-35")
    mod.eval()
    for p in mod.parameters():
        p.requires_grad_(False)
    return mod, bar, ck


# --- fitcheck ---------------------------------------------------------------------
def fit_rows(D, n=None):
    n = FIT_N if n is None else n
    tr = np.asarray(D["train_rows"])
    return np.sort(np.random.default_rng(FIT_ROWS_SEED).choice(tr, n,
                                                               replace=False))


def fitcheck_run(torch, a, D, mod, bar, mb):
    import k21_model as km
    import k21_train as kt
    rows = fit_rows(D)
    dev, dt = torch.device(a.device), getattr(torch, a.dtype)
    G = k35m.groups(mod)
    set_phase(G, "main")              # полный набор ключей для сравнения
    st0 = kt.trainable_state(mod)
    set_phase(G, "warmup")
    heads = ("readout8.", "readout16.", "readout.")
    src0 = {k: v for k, v in st0.items() if not k.startswith(heads)}
    ev0 = evaluate(torch, km, mod, bar, D, rows, dev, dt, a.eval_micro, mb,
                   range(36))
    opt = make_opt(torch, G, "warmup")
    scaler = torch.cuda.amp.GradScaler(
        enabled=(dt == torch.float16 and dev.type == "cuda"))
    rng = np.random.default_rng(FIT_ROWS_SEED + 1)
    mod.train()
    for u in range(int(FIT_UPDATES)):
        idx = rng.choice(len(rows), EFF_BATCH, replace=True)
        cuts = cuts_for(10_000_000 + u)
        opt.zero_grad(set_to_none=True)
        for s0 in range(0, EFF_BATCH, a.micro):
            r = rows[idx[s0:s0 + a.micro]]
            batch, po, st, tgt, val = mb(D, r, bar, dev)
            with torch.autocast("cuda", dtype=dt, enabled=dev.type == "cuda"):
                o = mod(batch, po, st, cuts=cuts[s0:s0 + len(r)])
                loss = km.loss_fn(torch, o, tgt, val)[0]
            scaler.scale(loss.mean() * len(r) / EFF_BATCH).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(G["heads"], CLIP)
        scaler.step(opt)
        scaler.update()
    ev1 = evaluate(torch, km, mod, bar, D, rows, dev, dt, a.eval_micro, mb,
                   range(36))
    set_phase(G, "main")
    st1 = kt.trainable_state(mod)
    if set(st1) != set(st0):
        raise SystemExit("fitcheck: набор тензоров изменился")
    drop = 1 - ev1["mean_L"] / ev0["mean_L"]
    changed = {h[:-1]: any(not torch.equal(st1[k], st0[k]) for k in st0
                           if k.startswith(h)) for h in heads}
    src_same = all(torch.equal(st1[k], v) for k, v in src0.items())
    ok = bool(np.isfinite(drop) and drop >= FIT_DROP and all(changed.values())
              and src_same)
    return dict(rows=rows.tolist(), L_before=ev0["mean_L"],
                L_after=ev1["mean_L"], drop=drop, heads_changed=changed,
                rest_unchanged=src_same, ok=ok,
                criterion=dict(drop=FIT_DROP, updates=FIT_UPDATES, n=FIT_N))


def fitcheck(a):
    import torch
    import k21_data as d21
    a.micro = a.micro or micro_now() or 8
    D = d21.load_prepared(require_teacher=False)
    mod, bar = load_k35(torch, a.device)
    r = fitcheck_run(torch, a, D, mod, bar, d21.make_batch)
    r["contract"] = dict(code=code_now(), prepare_sha1=sha(PREPARE),
                         micro=a.micro, dtype=a.dtype)
    os.makedirs("reports/k35", exist_ok=True)
    json.dump(r, open(FITCHECK, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=float)
    print(f"  fitcheck K-35: средний L 36 срезов {r['L_before']:.4f} -> "
          f"{r['L_after']:.4f} (снижение {100 * r['drop']:.1f}%, порог 30%); "
          f"головы {r['heads_changed']}; прочее прежнее {r['rest_unchanged']}"
          f" — " + ("пройден" if r["ok"] else "НЕ ПРОЙДЕН"))
    return 0 if r["ok"] else 3


# --- real-check ------------------------------------------------------------------
def slow_probs(S, valid, sizes, parents, eps):
    """Медленный эталон (float64, циклы) по сохранённым входам."""
    S = np.asarray(S, np.float64)
    B, h, A, N = S.shape
    P = np.zeros_like(S)
    for b in range(B):
        v = np.asarray(valid[b], bool)
        lu = -np.log(v.sum())
        for hh in range(h):
            maps, off = [], 0
            for lev, n in enumerate(sizes):
                for j in range(n):
                    s = S[b, hh, off + j]
                    if lev == 0:
                        pr = np.full(N, lu)
                    else:
                        pp = maps[off - sizes[lev - 1] +
                                  int(parents[lev - 1][b][j])]
                        pr = np.log((1 - eps) * pp + eps * np.exp(lu))
                    x = np.where(v, s + pr, -np.inf)
                    x = np.exp(x - x.max())
                    p = x / x.sum()
                    maps.append(p)
                    P[b, hh, off + j] = p
                off += n
            maps.clear()
    return P


def real_check(a):
    """GPU: источник, 36 срезов, router, сборка, шаг, save/load, micro."""
    import gc
    import tempfile
    import torch
    import k21_data as d21
    import k21_model as km
    import k21_train as kt
    import k34_policy as kp34
    D = d21.load_prepared(require_teacher=False)
    dev = torch.device(a.device)
    prep = prepare_entry()
    rows = np.asarray(D["val_rows"])[:5]
    prob, rep = [], {}
    # 1) настоящая рука K-34 T0 (отдельная область жизни)
    spec = prep["source"]["spec"]
    ref, bar, _ck, _p = kp34.load_trained(torch, "parents_attn24", 0, 24,
                                          spec, a.device)
    ref_out, inputs = {}, {}
    with torch.no_grad():
        for B in (1, 5):
            batch, po, st, tgt, val = d21.make_batch(D, rows[:B], bar, dev)
            inputs[B] = (dict((k, v.cpu()) for k, v in batch.items()),
                         po.cpu(), st.cpu(), tgt.cpu(), val.cpu())
            with torch.autocast("cuda", dtype=torch.float16):
                o = ref(batch, po, st, exit_depth=24)
            ref_out[B] = (o["arm"].float().cpu(), o["grip_logit"].float().cpu())
    del ref, bar
    gc.collect()
    torch.cuda.empty_cache()

    def to_dev(B):
        b, po, st, tgt, val = inputs[B]
        return ({k: v.to(dev) for k, v in b.items()}, po.to(dev),
                st.to(dev), tgt.to(dev), val.to(dev))
    rep["mem_before_load_gb"] = torch.cuda.memory_allocated(dev) / 2 ** 30
    torch.cuda.reset_peak_memory_stats(dev)
    mod, bar = load_k35(torch, a.device, checkpoint=False)
    mod.eval()
    with torch.no_grad():
        for B in (1, 5):
            batch, po, st, tgt, val = to_dev(B)
            with torch.autocast("cuda", dtype=torch.float16):
                o = mod.source_forward(batch, po, st)
            same = torch.equal(o["arm"].float().cpu(), ref_out[B][0]) and \
                torch.equal(o["grip_logit"].float().cpu(), ref_out[B][1])
            rep[f"B{B}_source_equal"] = bool(same)
            if not same:
                prob.append(f"B{B}: source_forward ≠ K-34 T0 побитово")
            worst = dict(w_sum=0.0, router=0.0, y=0.0, arm=0.0)
            cut_lists = [[c] * B for c in range(36)]
            if B == 5:
                cut_lists.append([0, 17, 35, 9, 26])
            for cs in cut_lists:
                mod.router_capture = []
                with torch.autocast("cuda", dtype=torch.float16):
                    o = mod(batch, po, st, cuts=cs)
                cap = mod.router_capture
                mod.router_capture = None
                prob += [f"c={cs[0]}: {x}" for x in
                         k35m.counters_problems(mod.counters)]
                if not (torch.isfinite(o["arm"]).all() and
                        torch.isfinite(o["y"]).all()):
                    prob.append(f"c={cs}: нечисловой выход")
                worst["w_sum"] = max(worst["w_sum"], float(
                    (o["w_sum"] - 1).abs().max()))
                last = cap[-1]
                Ps = slow_probs(last["S"].numpy(), last["valid"].numpy(),
                                last["sizes"], [x.numpy() for x in
                                                last["parents"]], 0.1)
                worst["router"] = max(worst["router"], float(
                    np.abs(Ps - last["P"].numpy()).max()))
                y, arm, grip = k35m.assemble_np(
                    {k: v.cpu().numpy() for k, v in o["partial"].items()},
                    mod.readout.L.cpu().numpy(), mod.readout.U.cpu().numpy())
                worst["y"] = max(worst["y"], float(np.abs(
                    y - o["y"].cpu().numpy()).max()))
                worst["arm"] = max(worst["arm"], float(np.abs(
                    arm - o["arm"].cpu().numpy()).max()))
            rep[f"B{B}_worst"] = worst
            if not (worst["w_sum"] <= 5e-6 and worst["router"] <= 5e-6 and
                    worst["y"] <= 1e-5 and worst["arm"] <= 1e-6):
                prob.append(f"B{B}: допуски {worst}")
    # 2) main-шаг, save/load
    G = k35m.groups(mod)
    frozen = {n: p.detach().cpu().clone() for n, p in mod.named_parameters()
              if not p.requires_grad}
    before = {k: [p.detach().cpu().clone() for p in G[k]] for k in G}
    set_phase(G, "main")
    opt = make_opt(torch, G, "main")
    mod.train()
    mod.checkpoint = True
    batch, po, st, tgt, val = to_dev(5)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        o = mod(batch, po, st, cuts=[0, 17, 35, 9, 26])
        loss = km.loss_fn(torch, o, tgt, val)[0]
    loss.mean().backward()
    fin = all(bool(torch.isfinite(p.grad).all()) for k in G for p in G[k]
              if p.grad is not None)
    opt.step()
    chg = {k: any(not torch.equal(p.detach().cpu(), q) for p, q in
                  zip(G[k], before[k])) for k in G}
    fz = all(torch.equal(p.detach().cpu(), frozen[n]) for n, p in
             mod.named_parameters() if n in frozen)
    del opt, G, before, frozen, o, loss
    rep["main_step"] = dict(finite=bool(fin), changed=chg,
                            frozen_unchanged=bool(fz))
    if not (fin and all(chg.values()) and fz):
        prob.append(f"main-шаг {rep['main_step']}")
    mod.eval()
    mod.checkpoint = False
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            r1 = mod(batch, po, st, cuts=[3, 14, 25, 30, 1])["arm"].float()\
                .cpu()
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "s.pt")
            torch.save(kt.trainable_state(mod), f)
            for nm in ("readout8", "readout16", "readout"):
                for p_ in getattr(mod, nm).parameters():
                    p_.add_(0.01)
            for p_ in mod.bar.action_expert.layers[0].parameters():
                p_.add_(0.001)
            own = dict(mod.named_parameters())
            own.update(dict(mod.named_buffers()))
            for k, v in torch.load(f, weights_only=False).items():
                own[k].copy_(v.to(own[k].device, own[k].dtype))
        with torch.autocast("cuda", dtype=torch.float16):
            r2 = mod(batch, po, st, cuts=[3, 14, 25, 30, 1])["arm"].float()\
                .cpu()
    rep["save_load"] = bool(torch.equal(r1, r2))
    if not rep["save_load"]:
        prob.append("save/load")
    rep["peak_gb_eval"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    # 3) micro: первый из 32/16/8 без OOM и с пиком ≤ 14 ГиБ на полном шаге
    mod.train()
    mod.checkpoint = True
    G = k35m.groups(mod)
    set_phase(G, "main")
    opt = make_opt(torch, G, "main")
    micro, prof = None, {}
    vr = np.asarray(D["train_rows"])[:32]
    for mc in MICROS:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            opt.zero_grad(set_to_none=True)
            for s0 in range(0, 32, mc):
                b2, po2, st2, tg2, va2 = d21.make_batch(D, vr[s0:s0 + mc], bar,
                                                        dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    o = mod(b2, po2, st2,
                            cuts=list(cuts_for(0)[s0:s0 + mc]))
                    loss = km.loss_fn(torch, o, tg2, va2)[0]
                (loss.mean() * mc / 32).backward()
                del o, loss
            pk = torch.cuda.max_memory_allocated(dev) / 2 ** 30
            prof[mc] = pk
            if pk <= PEAK_LIMIT_GB:
                micro = mc
                break
        except torch.OutOfMemoryError:
            prof[mc] = "OOM"
            opt.zero_grad(set_to_none=True)
    rep.update(micro=micro, micro_profile=prof)
    if micro is None:
        prob.append("ни один micro из 32/16/8 не помещается в 14 ГиБ")
    del mod, bar, opt, G
    gc.collect()
    torch.cuda.empty_cache()
    rep["mem_after_free_gb"] = torch.cuda.memory_allocated(dev) / 2 ** 30
    out = dict(problems=prob, micro=micro, results=rep,
               contract=dict(code=code_now(), prepare_sha1=sha(PREPARE),
                             device=torch.cuda.get_device_name(dev)))
    os.makedirs("reports/k35", exist_ok=True)
    json.dump(out, open(REAL_CHECK, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"  real-check K-35: источник B1/B5 {rep.get('B1_source_equal')}/"
          f"{rep.get('B5_source_equal')}; худшие B5 {rep.get('B5_worst')}; "
          f"main-шаг {rep['main_step']}; save/load {rep['save_load']}; micro "
          f"{micro} ({prof}); память до {rep['mem_before_load_gb']:.2f} / "
          f"после {rep['mem_after_free_gb']:.2f} ГБ — "
          + ("пройден" if not prob else "НЕ пройден: " + "; ".join(prob[:3])))
    return 0 if not prob else 3


# --- самопроверка ---------------------------------------------------------------
def selftest():
    import tempfile
    import types
    import torch
    import k21_train as kt
    import k26_model as k26m
    # срезы: независимость от micro, баланс в группах из 36, детерминизм
    c0 = cuts_for(U_OFFSET)
    assert np.array_equal(c0, cuts_for(U_OFFSET)) and c0.min() >= 0 and \
        c0.max() < 36
    # группы по 36 логических строк выровнены по g, не по обновлениям:
    # 9 обновлений с u_abs ≡ 0 (mod 9) — ровно 8 полных групп
    u9 = (U_OFFSET // 9 + 1) * 9
    allc = np.concatenate([cuts_for(u9 + u) for u in range(9)])
    assert len(allc) == 288 and (np.bincount(allc, minlength=36) == 8).all()
    for k in range(8):
        assert sorted(allc[36 * k:36 * k + 36]) == list(range(36))
    assert lr_main(0, 1.0) < lr_main(199, 1.0) <= 1.0
    assert abs(lr_main(MAIN_UPDATES, 1.0) - LR_FLOOR) < 1e-9
    # 17 (часть). медленный эталон router = mapped на случайных входах
    M = k35m.build(torch)
    g = torch.Generator().manual_seed(0)
    S = torch.randn(2, 4, 12, 7, generator=g)
    valid = torch.ones(2, 7, dtype=torch.bool)
    valid[1, 5:] = False
    pm = torch.tensor([[0, 0, 3, 3], [1, 1, 2, 2]])
    pl = torch.tensor([[0, 0, 1, 1], [2, 2, 3, 3]])
    P = M["mapped_probs"](S, valid, (4, 4, 4), [pm, pl], 0.1)["P"].numpy()
    Ps = slow_probs(S.numpy(), valid.numpy(), (4, 4, 4), [pm.numpy(),
                                                          pl.numpy()], 0.1)
    assert np.abs(P - Ps).max() < 5e-6
    # игрушечное обучение: warmup — только головы; resume = непрерывному
    bar0 = k26m.fake_bar(torch)
    L, U = -np.ones(6) * 1.1, np.ones(6) * 4.0
    nr = 64
    DT = dict(train_rows=np.arange(48), val_rows=np.arange(48, 56),
              teacher_rows=np.arange(48, 56),
              task=np.array([f"t{i % 4}" for i in range(nr)]),
              episode=np.arange(nr) // 2)

    def fake_mb(D_, rows, bar, dev, teacher_idx=None):
        rows = np.asarray(rows)
        b = k26m.fake_batch(torch, B=len(rows), seed=int(rows.sum()) % 1000)
        gg = torch.Generator().manual_seed(int(rows.sum()) + 7)
        st_ = torch.randn(len(rows), 11, generator=gg)
        tgt = torch.rand(len(rows), 20, 7, generator=gg) * 2
        tgt[..., 6] = torch.where(tgt[..., 6] > 1, 1.0, -1.0)
        return b, 1, st_, tgt, torch.ones(len(rows), 20, dtype=torch.bool)

    def mk():
        m = M["make"](copy.deepcopy(bar0), L, U, seed=0, checkpoint=False)
        m.attach_exits()
        return m, dict(model=m.bar)
    gl = globals()
    keep = {k: gl[k] for k in ("WARM_UPDATES", "MAIN_UPDATES", "EVAL_EVERY",
                               "SAVE_EVERY")}
    a_ = types.SimpleNamespace(device="cpu", dtype="float32", micro=4,
                               eval_micro=4)
    try:
        gl.update(WARM_UPDATES=2, MAIN_UPDATES=4, EVAL_EVERY=2, SAVE_EVERY=1)
        with tempfile.TemporaryDirectory() as td:
            tr = Trainer(torch, a_, DT, np.arange(48, 56), mk, fake_mb,
                         os.path.join(td, "w"))
            heads = ("readout8.", "readout16.", "readout.")
            s0 = {k: v for k, v in kt.trainable_state(tr.mod).items()
                  if not k.startswith(heads)}
            set_phase(tr.G, "warmup")
            opt = make_opt(torch, tr.G, "warmup")
            tr.mod.train()
            tr.step(opt, *sched_entry(DT, U_OFFSET, 4),
                    cuts_for(U_OFFSET))
            s1 = kt.trainable_state(tr.mod)
            assert all(torch.equal(s1[k], v) for k, v in s0.items())
            res = {}
            for name, stops in (("cont", (4,)), ("resume", (2, 4))):
                for until in stops:
                    t = Trainer(torch, a_, DT, np.arange(48, 56), mk, fake_mb,
                                os.path.join(td, name))
                    t.run(until)
                res[name] = (kt.state_hash(kt.trainable_state(t.mod)),
                             t.st["best"])
            assert res["cont"] == res["resume"], res
            # 12. смена micro не меняет row -> c (c назначаются до разбиения)
            mc = {}
            for m_ in (4, 8, 16):
                src_, idx_ = sched_entry(DT, U_OFFSET + 3, m_)
                mc[m_] = (np.asarray(idx_).reshape(-1), cuts_for(U_OFFSET + 3))
            assert all(np.array_equal(mc[4][0], v[0]) and
                       np.array_equal(mc[4][1], v[1]) for v in mc.values())
            t = Trainer(torch, a_, DT, np.arange(48, 56), mk, fake_mb,
                        os.path.join(td, "cont"), contract_extra=dict(x=1))
            try:
                t.run(4)
                raise AssertionError("чужой контракт принят")
            except SystemExit:
                pass
            gl.update(FIT_UPDATES=3, FIT_N=8)
            m, b_ = mk()
            r = fitcheck_run(torch, a_, dict(DT, train_rows=np.arange(48)),
                             m, b_, fake_mb)
            assert r["rest_unchanged"] and all(r["heads_changed"].values()), r
    finally:
        gl.update(keep)
        gl.update(FIT_UPDATES=200, FIT_N=64)
    print("самопроверка k35_train пройдена: срезы строк детерминированы и "
          "сбалансированы (каждые 36 логических строк — все срезы), LR, "
          "медленный эталон router = mapped, warmup меняет только три головы, "
          "прерванное и продолженное обучение = непрерывному, чужой контракт "
          "— отказ, fitcheck не трогает ничего вне голов")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-35: обучение")
    ap.add_argument("cmd", choices=("train", "fitcheck", "real-check",
                                    "check-fresh", "train-valid"))
    ap.add_argument("--kind", choices=("real_check", "fitcheck"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--micro", type=int, default=None)
    ap.add_argument("--eval-micro", type=int, default=8)
    ap.add_argument("--no-fastload", action="store_true")
    a = ap.parse_args()
    try:
        if a.cmd == "check-fresh":
            p = fresh_problems(a.kind)
            print(f"  {a.kind}: " + ("свежий, пройден" if not p else
                                     "нужен заново — " + "; ".join(p)))
            return 0 if not p else 1
        if a.cmd == "train-valid":
            p = train_problems(micro_now(), a.dtype)
            print("  обучение K-35: " + ("соответствует контракту" if not p
                                         else "; ".join(p)))
            return 0 if not p else 1
        return dict(train=train, fitcheck=fitcheck,
                    **{"real-check": real_check})[a.cmd](a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
