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


def base_config(source, micro, dtype, panel):
    """Контракт обучения без очереди (schedule_fp зависит от данных);
    JSON-нормализован (кортежи -> списки)."""
    return json.loads(json.dumps(dict(
        source=source, loss=LOSS, micro=micro, dtype=dtype,
        eff_batch=EFF_BATCH,
        warm=dict(updates=WARM_UPDATES, lr=WARM_LR, groups=["heads"]),
        main=dict(updates=MAIN_UPDATES, lr=LR, warm=MAIN_WARM,
                  floor=LR_FLOOR), wd=WD, clip=CLIP, u_offset=U_OFFSET,
        seed=SEED, ckpt_cuts=list(ko.CKPT_CUTS),
        panel_sha1=ko.sha_arr(panel),
        topology=dict(n_cuts=tp.N_CUTS, pairs=[list(x) for x in tp.PAIRS]),
        code=code_now())))


def source_binding():
    p = prepare_entry()
    return dict(ckpt_sha1=p["source"]["ckpt_sha1"],
                weights_hash=p["source"]["weights_hash"])


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
            self.source = source_binding()
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
        return dict(base_config(self.source, self.a.micro, self.a.dtype,
                                self.panel),
                    schedule_fp=schedule_fp(self.D, self.a.micro),
                    **self.extra)

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
        if not realcheck_result_problems(r):
            return int(r["micro"])
    return None


def device_contract(torch, dev, dtype):
    return dict(code=code_now(), prepare_sha1=sha(PREPARE),
                device=torch.cuda.get_device_name(dev), dtype=dtype)


def realcheck_result_problems(r):
    """Полнота и прохождение всех результатов real-check."""
    p = []
    if r.get("problems") != []:
        p.append("real-check: есть проблемы")
    R = r.get("results") or {}
    for B in (1, 5):
        if R.get(f"B{B}_source_equal") is not True:
            p.append(f"real-check: нет/не пройдено B{B}_source_equal")
        if R.get(f"B{B}_route_ok") is not True:
            p.append(f"real-check: нет/не пройден маршрут B{B}")
        w = R.get(f"B{B}_worst") or {}
        if set(w) != set(RC_TOL) or not all(
                isinstance(w[k], (int, float)) and np.isfinite(w[k]) and
                w[k] <= RC_TOL[k] for k in RC_TOL):
            p.append(f"real-check: допуски B{B} неполны или превышены")
    if R.get("route_hooks_ok") is not True:
        p.append("real-check: нет перехвата маршрута")
    ms = R.get("main_step") or {}
    ch = ms.get("changed") or {}
    if ms.get("finite") is not True or ms.get("frozen_unchanged") is not \
            True or set(ch) != {"expert", "lora", "new", "heads"} or \
            not all(v is True for v in ch.values()):
        p.append("real-check: main-шаг неполон или не пройден")
    if R.get("save_load") is not True:
        p.append("real-check: save/load")
    mc = r.get("micro")
    prof = R.get("micro_profile") or {}
    pk = prof.get(str(mc))
    if mc not in MICROS or R.get("micro") != mc or \
            not isinstance(pk, (int, float)) or not pk <= PEAK_LIMIT_GB:
        p.append("real-check: micro или его профиль неполны")
    else:
        for m_ in MICROS:
            if m_ == mc:
                break
            v = prof.get(str(m_))
            if not (v == "OOM" or (isinstance(v, (int, float)) and
                                   v > PEAK_LIMIT_GB)):
                p.append(f"real-check: micro {m_} не исключён измерением")
    return p


def fitcheck_result_problems(r):
    p = []
    try:
        drop = 1 - float(r["L_after"]) / float(r["L_before"])
    except (KeyError, TypeError, ZeroDivisionError, ValueError):
        return ["fitcheck: нет L до/после"]
    if not (np.isfinite(drop) and abs(drop - float(r.get("drop", np.nan)))
            < 1e-9 and drop >= FIT_DROP):
        p.append("fitcheck: снижение L не подтверждается или < 30%")
    hc = r.get("heads_changed") or {}
    if set(hc) != {"readout8", "readout16", "readout"} or \
            not all(v is True for v in hc.values()):
        p.append("fitcheck: не все три головы изменились")
    if r.get("rest_unchanged") is not True:
        p.append("fitcheck: изменилось что-то вне голов")
    if r.get("criterion") != dict(drop=FIT_DROP, updates=FIT_UPDATES,
                                  n=FIT_N) or len(r.get("rows") or []) != \
            FIT_N:
        p.append("fitcheck: критерий или строки ≠ плану")
    if r.get("ok") is not True:
        p.append("fitcheck: не пройден")
    return p


def fresh_problems(kind, device_name=None, dtype="float16"):
    """Отчёт real-check / fitcheck полон, пройден и с текущим контрактом
    (код, подготовка, устройство, dtype, micro)."""
    path = dict(real_check=REAL_CHECK, fitcheck=FITCHECK)[kind]
    if not os.path.exists(path):
        return [f"нет {path}"]
    r = json.load(open(path, encoding="utf-8"))
    p = realcheck_result_problems(r) if kind == "real_check" else \
        fitcheck_result_problems(r)
    c = r.get("contract") or {}
    if c.get("code") != code_now():
        p.append(f"{kind}: код изменён после проверки")
    if not os.path.exists(PREPARE) or c.get("prepare_sha1") != sha(PREPARE):
        p.append(f"{kind}: другая подготовка")
    if c.get("dtype") != dtype:
        p.append(f"{kind}: dtype {c.get('dtype')} ≠ {dtype}")
    if device_name is not None and c.get("device") != device_name:
        p.append(f"{kind}: устройство {c.get('device')} ≠ {device_name}")
    if kind == "fitcheck":
        rc = json.load(open(REAL_CHECK, encoding="utf-8")) \
            if os.path.exists(REAL_CHECK) else {}
        if c.get("micro") != micro_now() or c.get("device") != \
                (rc.get("contract") or {}).get("device"):
            p.append("fitcheck: micro/устройство ≠ решению real-check")
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
    nm = torch.cuda.get_device_name(torch.device(a.device))
    pr = fresh_problems("real_check", nm, a.dtype) + \
        fresh_problems("fitcheck", nm, a.dtype)
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
    """Отчёт обучения полон и соответствует контракту: источник, seed,
    панель, потеря, режим, очередь, код, micro/dtype, 4000 main, best."""
    if not os.path.exists(REPORT):
        return ["нет отчёта обучения"]
    r = json.load(open(REPORT, encoding="utf-8"))
    ck = os.path.join(RUN_DIR, "best.pt")
    p = []
    if not os.path.exists(ck) or sha(ck) != r.get("best_sha1"):
        p.append("best.pt ≠ отчёту")
    if micro is None:
        return p + ["нет решения о micro (real-check)"]
    try:
        pan = json.load(open(ko.PANELS, encoding="utf-8"))
        want = base_config(source_binding(), micro, dtype,
                           pan["checkpoint_val"]["rows"])
        init = prepare_entry()["source"]["weights_hash"]
    except (OSError, KeyError, SystemExit) as e:
        return p + [f"нет подготовки/панелей: {e}"]
    c = dict(r.get("config") or {})
    if not isinstance(c.pop("schedule_fp", None), str):
        p.append("в конфигурации нет отпечатка очереди")
    diff = sorted(k for k in set(c) | set(want) if c.get(k) != want.get(k))
    if diff:
        p.append(f"конфигурация обучения ≠ контракту: {diff}")
    if r.get("init_hash") != init:
        p.append("начальные веса обучения ≠ источнику K-34 T0")
    if r.get("updates") != dict(warmup=WARM_UPDATES, main=MAIN_UPDATES):
        p.append("обучение не дошло до 500 + 4000")
    ups = [x.get("update") for x in r.get("curve") or []]
    if ups != list(range(0, MAIN_UPDATES + 1, EVAL_EVERY)):
        p.append("кривая проверок неполна")
    else:
        vals = [x["val"]["mean_L"] for x in r["curve"]]
        bu = ups[int(np.argmin(vals))]
        if r.get("best_update") != bu or (r.get("best") or [None])[0] != bu:
            p.append("best ≠ минимуму кривой (ничья — ранний шаг)")
    return p


def load_trained(torch, device, checkpoint=False):
    """K-35 best (общий декодер A/C/N) со сверкой отпечатка."""
    import k21_train as ktr
    r = json.load(open(REPORT, encoding="utf-8"))
    ck_path = os.path.join(RUN_DIR, "best.pt")
    if sha(ck_path) != r["best_sha1"]:
        raise SystemExit("best.pt K-35 ≠ отчёту")
    pr = train_problems(micro_now(), "float16")
    if pr:
        raise SystemExit("обучение K-35: " + "; ".join(pr))
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    if ck.get("config") != r["config"] or ck.get("weights_hash") != \
            r["best_weights_hash"] or ck.get("update") != r["best_update"] \
            or ck.get("init_hash") != r["init_hash"]:
        raise SystemExit("best.pt K-35: конфигурация/отпечаток ≠ отчёту")
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
    nm = torch.cuda.get_device_name(torch.device(a.device))
    pr = fresh_problems("real_check", nm, a.dtype)
    if pr:
        raise SystemExit("fitcheck требует свежего real-check: " +
                         "; ".join(pr))
    a.micro = micro_now()
    D = d21.load_prepared(require_teacher=False)
    mod, bar = load_k35(torch, a.device)
    r = fitcheck_run(torch, a, D, mod, bar, d21.make_batch)
    r["contract"] = dict(device_contract(torch, torch.device(a.device),
                                         a.dtype), micro=a.micro)
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


RC_TOL = dict(w_sum=5e-6, router=5e-6, y=1e-5, arm=1e-6)
ROUTE_IV_TOL = 0.05        # вложенность интервалов под fp16 (единицы шагов)
SMOKE_CUTS_RC = ([0, 17, 35, 9, 26], [35] * 5, [17] * 5)


def _free(torch):
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def _ref_outputs(torch, D, rows, spec, device):
    """Настоящая рука K-34 T0 -> CPU-входы и выходы B = 1, 5; модель
    освобождается внутри функции."""
    import k21_data as d21
    import k34_policy as kp34
    dev = torch.device(device)
    ref, bar, _ck, _p = kp34.load_trained(torch, "parents_attn24", 0, 24,
                                          spec, device)
    ref_out, inputs = {}, {}
    with torch.no_grad():
        for B in (1, 5):
            batch, po, st, tgt, val = d21.make_batch(D, rows[:B], bar, dev)
            inputs[B] = (dict((k, v.cpu()) for k, v in batch.items()),
                         po.cpu() if hasattr(po, "cpu") else po, st.cpu(),
                         tgt.cpu(), val.cpu())
            with torch.autocast("cuda", dtype=torch.float16):
                o = ref(batch, po, st, exit_depth=24)
            ref_out[B] = (o["arm"].float().cpu(),
                          o["grip_logit"].float().cpu())
    del ref, bar, o, batch
    return inputs, ref_out


def _to_dev(torch, inp, dev):
    b, po, st, tgt, val = inp
    return ({k: v.to(dev) for k, v in b.items()},
            po.to(dev) if hasattr(po, "to") else po, st.to(dev), tgt.to(dev),
            val.to(dev))


def _k35_checks(torch, inputs, ref_out, device):
    """Источник, 36 срезов (B1/B5, смешанные), фактический маршрут, router,
    сборка, main-шаг, save/load. Возвращает только CPU-результаты."""
    import tempfile
    import k21_model as km
    import k21_train as kt
    dev = torch.device(device)
    rep, prob = {}, []
    rep["mem_before_load_gb"] = torch.cuda.memory_allocated(dev) / 2 ** 30
    torch.cuda.reset_peak_memory_stats(dev)
    mod, _bar = load_k35(torch, device, checkpoint=False)
    rep["mem_loaded_gb"] = torch.cuda.memory_allocated(dev) / 2 ** 30
    mod.eval()
    TAB = tp.tables()
    eps = float(mod.router.eps)
    with torch.no_grad():
        for B in (1, 5):
            batch, po, st, tgt, val = _to_dev(torch, inputs[B], dev)
            with torch.autocast("cuda", dtype=torch.float16):
                o = mod.source_forward(batch, po, st)
            same = torch.equal(o["arm"].float().cpu(), ref_out[B][0]) and \
                torch.equal(o["grip_logit"].float().cpu(), ref_out[B][1])
            rep[f"B{B}_source_equal"] = bool(same)
            if not same:
                prob.append(f"B{B}: source_forward ≠ K-34 T0 побитово")
            worst = dict(w_sum=0.0, router=0.0, y=0.0, arm=0.0)
            route_bad = []
            cut_lists = [[c] * B for c in range(36)]
            if B == 5:
                cut_lists += [list(x) for x in SMOKE_CUTS_RC]
            for cs in cut_lists:
                mod.router_capture = []
                with torch.autocast("cuda", dtype=torch.float16):
                    o = mod(batch, po, st, cuts=cs)
                cap, mod.router_capture = mod.router_capture, None
                prob += [f"c={cs[0]}: {x}" for x in
                         k35m.counters_problems(mod.counters)]
                if not (torch.isfinite(o["arm"]).all() and
                        torch.isfinite(o["y"]).all()):
                    prob.append(f"c={cs}: нечисловой выход")
                cc = np.asarray(cs)
                if not np.array_equal(o["c"].numpy(), cc):
                    route_bad.append(f"c {cs}")
                for k, v in o["route"].items():
                    if not np.array_equal(v.cpu().numpy(), TAB[k][cc]):
                        route_bad.append(f"{k} при c {cs}")
                worst["w_sum"] = max(worst["w_sum"], float(
                    (o["w_sum"] - 1).abs().max()))
                for last in cap:
                    Ps = slow_probs(last["S"].numpy(), last["valid"].numpy(),
                                    last["sizes"], [x.numpy() for x in
                                                    last["parents"]], eps)
                    worst["router"] = max(worst["router"], float(
                        np.abs(Ps - last["P"].numpy()).max()))
                y, arm, grip = k35m.assemble_np(
                    {k: v.cpu().numpy() for k, v in o["partial"].items()},
                    mod.readout.L.cpu().numpy(), mod.readout.U.cpu().numpy())
                worst["y"] = max(worst["y"], float(np.abs(
                    y - o["y"].cpu().numpy()).max()), float(np.abs(
                        grip - o["grip_logit"].cpu().numpy()).max()))
                worst["arm"] = max(worst["arm"], float(np.abs(
                    arm - o["arm"].cpu().numpy()).max()))
            rep[f"B{B}_worst"] = worst
            rep[f"B{B}_route_ok"] = not route_bad
            if route_bad:
                prob.append(f"B{B}: исполненный маршрут ≠ топологии: "
                            f"{route_bad[:3]}")
            if not all(worst[k] <= RC_TOL[k] for k in RC_TOL):
                prob.append(f"B{B}: допуски {worst}")
        # перехват аргументов Expand / router / RoPE / голов на настоящей
        # модели (те же проверки, что в самопроверке k35_model)
        batch, po, st, tgt, val = _to_dev(torch, inputs[5], dev)
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                k35m.route_check(torch, mod, batch, st,
                                 [list(x) for x in SMOKE_CUTS_RC],
                                 po=po, iv_tol=ROUTE_IV_TOL)
            rep["route_hooks_ok"] = True
        except AssertionError as e:
            rep["route_hooks_ok"] = False
            prob.append(f"перехват маршрута: {str(e)[:200]}")
    # main-шаг
    G = k35m.groups(mod)
    set_phase(G, "main")
    frozen = {n: p.detach().cpu().clone() for n, p in mod.named_parameters()
              if not p.requires_grad}
    before = {k: [p.detach().cpu().clone() for p in G[k]] for k in G}
    opt = make_opt(torch, G, "main")
    mod.train()
    mod.checkpoint = True
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        o = mod(batch, po, st, cuts=list(SMOKE_CUTS_RC[0]))
        loss = km.loss_fn(torch, o, tgt, val)[0]
    loss.mean().backward()
    fin = all(bool(torch.isfinite(p.grad).all()) for k in G for p in G[k]
              if p.grad is not None)
    opt.step()
    chg = {k: any(not torch.equal(p.detach().cpu(), q) for p, q in
                  zip(G[k], before[k])) for k in G}
    fz = all(torch.equal(p.detach().cpu(), frozen[n]) for n, p in
             mod.named_parameters() if n in frozen)
    rep["main_step"] = dict(finite=bool(fin), changed=chg,
                            frozen_unchanged=bool(fz))
    if not (fin and all(chg.values()) and fz):
        prob.append(f"main-шаг {rep['main_step']}")
    del opt, before, frozen, o, loss
    # save/load: порча голов и общего веса, восстановление, те же выходы
    mod.eval()
    mod.checkpoint = False
    cs = [3, 14, 25, 30, 1]
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            r1 = mod(batch, po, st, cuts=cs)["arm"].float().cpu()
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "s.pt")
            torch.save(kt.trainable_state(mod), f)
            for nm in ("readout8", "readout16", "readout"):
                for p_ in getattr(mod, nm).parameters():
                    p_.add_(0.01)
            for p_ in mod.bar.action_expert.layers[0].parameters():
                p_.add_(0.001)
            with torch.autocast("cuda", dtype=torch.float16):
                r_bad = mod(batch, po, st, cuts=cs)["arm"].float().cpu()
            own = dict(mod.named_parameters())
            own.update(dict(mod.named_buffers()))
            for k, v in torch.load(f, weights_only=False).items():
                own[k].copy_(v.to(own[k].device, own[k].dtype))
            del own
        with torch.autocast("cuda", dtype=torch.float16):
            r2 = mod(batch, po, st, cuts=cs)["arm"].float().cpu()
    rep["save_load"] = bool(torch.equal(r1, r2) and not torch.equal(r1,
                                                                    r_bad))
    if not rep["save_load"]:
        prob.append("save/load")
    rep["peak_gb_checks"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    del mod, _bar, G, batch, st, tgt, val
    return rep, prob


def _micro_profile(torch, D, device, mc):
    """Отдельная область жизни: модель -> полный обучающий шаг (backward на
    32 строках, GradScaler, clip, optimizer.step с состояниями AdamW) ->
    пик памяти (CPU-число) или 'OOM'."""
    import k21_data as d21
    import k21_model as km
    dev = torch.device(device)
    mod = bar = opt = G = scaler = None
    try:
        torch.cuda.reset_peak_memory_stats(dev)
        mod, bar = load_k35(torch, device, checkpoint=True)
        G = k35m.groups(mod)
        set_phase(G, "main")
        opt = make_opt(torch, G, "main")
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        mod.train()
        vr = np.asarray(D["train_rows"])[:EFF_BATCH]
        cuts = cuts_for(U_OFFSET)
        opt.zero_grad(set_to_none=True)
        for s0 in range(0, EFF_BATCH, mc):
            b2, po2, st2, tg2, va2 = d21.make_batch(D, vr[s0:s0 + mc], bar,
                                                    dev)
            with torch.autocast("cuda", dtype=torch.float16):
                o = mod(b2, po2, st2, cuts=list(cuts[s0:s0 + mc]))
                loss = km.loss_fn(torch, o, tg2, va2)[0]
            scaler.scale(loss.mean() * mc / EFF_BATCH).backward()
            del o, loss, b2, st2, tg2, va2
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups
                                        for p in g["params"]], CLIP)
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize(dev)
        return float(torch.cuda.max_memory_allocated(dev) / 2 ** 30)
    except torch.OutOfMemoryError:
        return "OOM"
    finally:
        del mod, bar, opt, G, scaler
        _free(torch)


def real_check(a):
    """GPU: источник, 36 срезов, маршрут, router, сборка, шаг, save/load,
    micro. Каждая модель — в своей функции; наружу только CPU-числа."""
    import torch
    import k21_data as d21
    D = d21.load_prepared(require_teacher=False)
    dev = torch.device(a.device)
    prep = prepare_entry()
    rows = np.asarray(D["val_rows"])[:5]
    inputs, ref_out = _ref_outputs(torch, D, rows, prep["source"]["spec"],
                                   a.device)
    _free(torch)
    rep, prob = _k35_checks(torch, inputs, ref_out, a.device)
    _free(torch)
    rep["mem_after_checks_gb"] = torch.cuda.memory_allocated(dev) / 2 ** 30
    micro, prof = None, {}
    for mc in MICROS:
        pk = _micro_profile(torch, D, a.device, mc)
        prof[str(mc)] = pk
        if pk != "OOM" and pk <= PEAK_LIMIT_GB:
            micro = mc
            break
    rep.update(micro=micro, micro_profile=prof)
    if micro is None:
        prob.append("ни один micro из 32/16/8 не помещается в 14 ГиБ")
    _free(torch)
    rep["mem_after_free_gb"] = torch.cuda.memory_allocated(dev) / 2 ** 30
    out = dict(problems=prob, micro=micro, results=rep,
               contract=device_contract(torch, dev, a.dtype))
    os.makedirs("reports/k35", exist_ok=True)
    json.dump(out, open(REAL_CHECK, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"  real-check K-35: источник B1/B5 {rep.get('B1_source_equal')}/"
          f"{rep.get('B5_source_equal')}; маршрут B1/B5/перехват "
          f"{rep.get('B1_route_ok')}/{rep.get('B5_route_ok')}/"
          f"{rep.get('route_hooks_ok')}; худшие B5 {rep.get('B5_worst')}; "
          f"main-шаг {rep['main_step']}; save/load {rep['save_load']}; micro "
          f"{micro} ({prof}); память до {rep['mem_before_load_gb']:.2f} / "
          f"после {rep['mem_after_free_gb']:.2f} ГБ — "
          + ("пройден" if not prob else "НЕ пройден: " + "; ".join(prob[:3])))
    return 0 if not prob else 3


def good_realcheck():
    prof = {"32": "OOM", "16": 13.0}
    R = dict(B1_source_equal=True, B5_source_equal=True, B1_route_ok=True,
             B5_route_ok=True, route_hooks_ok=True,
             B1_worst={k: 0.0 for k in RC_TOL},
             B5_worst={k: 0.0 for k in RC_TOL},
             main_step=dict(finite=True, frozen_unchanged=True,
                            changed=dict(expert=True, lora=True, new=True,
                                         heads=True)),
             save_load=True, micro=16, micro_profile=prof)
    return dict(problems=[], micro=16, results=R)


def validators_regression():
    import copy as cp
    import tempfile
    g = good_realcheck()
    assert realcheck_result_problems(g) == [], realcheck_result_problems(g)
    for mut in (lambda r: r["results"].pop("B5_worst"),
                lambda r: r["results"].pop("route_hooks_ok"),
                lambda r: r["results"]["main_step"]["changed"].pop("lora"),
                lambda r: r["results"]["B1_worst"].update(arm=1e-3),
                lambda r: r["results"]["micro_profile"].update({"32": 12.0}),
                lambda r: r["results"].pop("save_load"),
                lambda r: r.update(problems=None),
                lambda r: r.update(results={})):
        b = cp.deepcopy(g)
        mut(b)
        assert realcheck_result_problems(b), b
    f = dict(L_before=1.0, L_after=0.6, drop=0.4, ok=True,
             heads_changed=dict(readout8=True, readout16=True, readout=True),
             rest_unchanged=True, rows=list(range(FIT_N)),
             criterion=dict(drop=FIT_DROP, updates=FIT_UPDATES, n=FIT_N))
    assert fitcheck_result_problems(f) == []
    for mut in (lambda r: r.pop("heads_changed"),
                lambda r: r.update(L_after=0.9),
                lambda r: r.update(rest_unchanged=None),
                lambda r: r.update(rows=[1, 2]),
                lambda r: r.pop("L_before")):
        b = cp.deepcopy(f)
        mut(b)
        assert fitcheck_result_problems(b), b
    # отчёт обучения: чужие source / seed / panel / loss — отказ
    G = globals()
    keep = {k: G[k] for k in ("REPORT", "RUN_DIR", "PREPARE", "REAL_CHECK")}
    keep_pan = ko.PANELS
    with tempfile.TemporaryDirectory() as td:
        try:
            G.update(REPORT=os.path.join(td, "r.json"),
                     RUN_DIR=os.path.join(td, "run"),
                     PREPARE=os.path.join(td, "prep.json"),
                     REAL_CHECK=os.path.join(td, "rc.json"))
            ko.PANELS = os.path.join(td, "pan.json")
            os.makedirs(G["RUN_DIR"])
            json.dump(dict(source=dict(ckpt_sha1="c", weights_hash="w"),
                           prep_id="x"), open(G["PREPARE"], "w"))
            json.dump(dict(checkpoint_val=dict(rows=[4, 8, 15])),
                      open(ko.PANELS, "w"))
            ckf = os.path.join(G["RUN_DIR"], "best.pt")
            open(ckf, "wb").write(b"best")
            cfg = dict(base_config(dict(ckpt_sha1="c", weights_hash="w"), 16,
                                   "float16", [4, 8, 15]), schedule_fp="q")
            curve = [dict(update=u, val=dict(mean_L=1.0 - u / 1e4 if u <=
                                             2000 else 0.9))
                     for u in range(0, MAIN_UPDATES + 1, EVAL_EVERY)]
            rep = dict(best_sha1=sha(ckf), config=cfg, init_hash="w",
                       updates=dict(warmup=WARM_UPDATES, main=MAIN_UPDATES),
                       curve=curve, best_update=2000, best=[2000, 0.8])
            json.dump(rep, open(G["REPORT"], "w"))
            assert train_problems(16, "float16") == [], \
                train_problems(16, "float16")
            for mut in (
                    lambda r: r["config"]["source"].update(ckpt_sha1="z"),
                    lambda r: r["config"].update(seed=1),
                    lambda r: r["config"].update(panel_sha1="p"),
                    lambda r: r["config"].update(loss="другая"),
                    lambda r: r["config"].pop("schedule_fp"),
                    lambda r: r.update(init_hash="v"),
                    lambda r: r.update(best_update=2250),
                    lambda r: r.update(curve=r["curve"][:-1]),
                    lambda r: r["updates"].update(main=3999)):
                b = cp.deepcopy(rep)
                mut(b)
                json.dump(b, open(G["REPORT"], "w"))
                assert train_problems(16, "float16"), mut
            json.dump(rep, open(G["REPORT"], "w"))
            assert train_problems(32, "float16")         # другой micro
            assert train_problems(None, "float16")
        finally:
            G.update(keep)
            ko.PANELS = keep_pan


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
    validators_regression()
    print("самопроверка k35_train пройдена: срезы строк детерминированы и "
          "сбалансированы (каждые 36 логических строк — все срезы), LR, "
          "медленный эталон router = mapped, warmup меняет только три головы, "
          "прерванное и продолженное обучение = непрерывному, чужой контракт "
          "— отказ, fitcheck не трогает ничего вне голов; валидаторы real-check / "
          "fitcheck / обучения отвергают неполные и чужие отчёты")
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
            import torch
            nm = torch.cuda.get_device_name(torch.device(a.device)) \
                if torch.cuda.is_available() else None
            p = fresh_problems(a.kind, nm, a.dtype)
            if nm is None:
                p.append("CUDA недоступна — устройство не сверено")
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
