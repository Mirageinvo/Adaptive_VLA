#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-30: парное SFT-дообучение H (hard) и C (soft) от K-25 того же сида;
base prepare, real-check (+ решение micro), fitcheck, train, check-fresh.

Потери (у H и C одинаковые; по строке, затем среднее):
  L_action = L1 руки (маски K-21) + 0.1·BCE схвата (пропуск target_grip = 0);
  L_code   = среднее по root_valid CE(logits_r, label_r);
  L_coarse = среднее по root_valid и каналам SmoothL1((b_soft − b_target)/s,
             beta = 1) — b_soft у ОБОИХ вариантов;
  L_total  = L_action + 0.02·L_code + 0.1·L_coarse.
Бюджет: warmup 500 (только root_ln/root_logits/plan_proj, AdamW 3e−4) +
main 4000 (отдельный AdamW: эксперт 1e−5, LoRA 5e−6, прочие используемые
старые и новые 1e−4; линейный разогрев 200, косинус до 0.1 на 4000), wd
0.01, clip 1.0, эффективный batch 32, 3 expert : 1 teacher. Очередь: q =
0…4499 (warmup + main, q не сбрасывается), u_abs = 100000 + q, RNG
[seed, 7, u_abs] (k21_train.schedule), teacher при u_abs % 4 == 3.
Проверка val_sel: warmup 0/500 и каждые 250 main; best — минимум L_action
(взвешенный) на main 250…4000, при равенстве — ранний; без добора.
GradScaler: применённые шаги и пропуски отдельно; 50 подряд — отказ.
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

import k30_model as k30m  # noqa: E402

SEEDS = (0, 1)
EFF = 32
WARM, MAIN = 500, 4000
LR_WARM = 3e-4
LR = dict(expert=1e-5, lora=5e-6, other=1e-4, new=1e-4)
MAIN_WARM, HORIZON, FLOOR = 200, 4000, 0.1
WD, CLIP = 0.01, 1.0
Q_OFFSET = 100000
EVAL_EVERY = SAVE_EVERY = 250
W_CODE, W_COARSE, GRIP_W = 0.02, 0.1, 0.1
MAX_SKIPS = 50
STEP_TRIES = 8
FIT_N, FIT_UPDATES, FIT_LR_MULT = 128, 500, 3.0
MICROS = (8, 32)
PREPARE = "reports/k30/prepare.json"
REAL_CHECK = "reports/k30/real_check.json"
MICRO_FILE = "reports/k30/micro.json"
FITCHECK = "reports/k30/fitcheck.json"
CODE = ("k30_codec.py", "k30_data.py", "k30_readout.py", "k30_model.py",
        "k30_train.py", "k25_readout.py", "k25_model.py", "k25_policy.py",
        "k25_protocol.py", "k22_model.py", "k21_model.py", "k21_tree.py",
        "k21_data.py", "k21_train.py", "k20_data.py", "k16_models.py",
        "k21_fastload.py", "k23_train.py")
HP = dict(eff=EFF, warm=WARM, main=MAIN, lr_warm=LR_WARM, lr=LR,
          main_warm=MAIN_WARM, horizon=HORIZON, floor=FLOOR, wd=WD, clip=CLIP,
          q_offset=Q_OFFSET, teacher="u_abs % 4 == 3", eval_every=EVAL_EVERY,
          w_code=W_CODE, w_coarse=W_COARSE, grip_w=GRIP_W,
          best="min weighted val_sel L_action, main 250…4000",
          micros=list(MICROS))


def sha(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def fp(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()
                        ).hexdigest()


def outdir(seed, mode):
    return f"data/k30/runs/s{seed}/{k30m.MODES[mode]}"


def report_path(seed, mode):
    return f"reports/k30/train_s{seed}_{mode}.json"


def code_sha():
    return {f: sha(os.path.join(HERE, f)) for f in CODE}


# --- base prepare ------------------------------------------------------------------
def make_prepare():
    import torch
    import transformers
    import k21_data as d21
    import k30_codec as kc
    import k30_data as kd
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    tj = json.load(open(d21.TEACHER_REPORT, encoding="utf-8"))
    C = kd.load()
    if not C["report"]["gate"]["passed"]:
        raise SystemExit("допуск кодека не пройден — обучение VLA не "
                         "запускается")
    if C["report"]["k21_prep_id"] != prep["prep_id"]:
        raise SystemExit("кодек от другой подготовки K-21")
    src = {}
    for s in SEEDS:
        e = k30m.source_entry(s)
        if e["prep_id"] != prep["prep_id"]:
            raise SystemExit(f"seed {s}: источник другой подготовки")
        src[str(s)] = e
    return dict(
        k21=dict(prep_id=prep["prep_id"], prepare_sha1=sha(d21.REPORT),
                 teacher_sha1=tj["out"]["sha1"],
                 teacher_report_sha1=sha(d21.TEACHER_REPORT)),
        bar_revision=prep["bar_pin"]["revision"], sources=src,
        codec=dict(sha=C["sha"], codec_file_sha1=C["report"]["codec_file_sha1"],
                   labels_file_sha1=C["report"]["labels_file_sha1"],
                   report_sha1=sha(kd.REPORT), config=kc.CONFIG,
                   gate=C["report"]["gate"]),
        rows=C["report"]["rows"], arch=k30m.ARCH, modes=k30m.MODES,
        plan=dict(hard="argmax + STE (прямой — P[ids])", soft="softmax @ P",
                  eps_plan=1e-3, seed_space=k30m.SEED_SPACE, K=k30m.K),
        geometry="фиксированная: корни по 5, деления пополам, τ K-21",
        precision="тяжёлые слои fp16 autocast + GradScaler + checkpointing; "
                  "план/readout float32 вне autocast; TF32 off",
        hp=HP, code=code_sha(), versions=dict(
            numpy=np.__version__, torch=torch.__version__,
            transformers=transformers.__version__))


def prepare():
    rep = make_prepare()
    if os.path.exists(PREPARE):
        old = json.load(open(PREPARE, encoding="utf-8"))
        if fp(old) != fp(rep):
            diff = sorted(k for k in set(old) | set(rep)
                          if old.get(k) != rep.get(k))
            raise SystemExit(f"{PREPARE} уже закреплён; изменилось {diff} — "
                             f"новая постановка требует нового каталога")
        print(f"  {PREPARE}: закреплён, сверка пройдена")
        return 0
    os.makedirs("reports/k30", exist_ok=True)
    with open(PREPARE + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(PREPARE + ".tmp", PREPARE)
    print(f"  {PREPARE}: источники " + ", ".join(
        f"s{s} {e['weights_hash'][:12]} (@{e['update']})"
        for s, e in rep["sources"].items()) + f"; словарь {rep['codec']['sha'][:12]}")
    return 0


def load_prepare():
    if not os.path.exists(PREPARE):
        raise SystemExit("нет base prepare: k30_train.py prepare")
    p = json.load(open(PREPARE, encoding="utf-8"))
    bad = [f for f, s in p["code"].items() if sha(os.path.join(HERE, f)) != s]
    if bad:
        raise SystemExit(f"код изменён после prepare: {bad}")
    return p


# --- очередь ------------------------------------------------------------------------
def make_schedule(seed):
    import k21_train as kt

    def sched(D, n_updates, micro, seed_=None, start=0):
        """n_updates/start — в пространстве u_abs."""
        return kt.schedule(D, n_updates, micro, seed=seed, start=start)
    return sched


def batch_at(sched, D, q, micro):
    u = Q_OFFSET + q
    return sched(D, u + 1, micro, start=u)[0]


def queue_sha(sched, D, micro):
    h = hashlib.sha1()
    for q in range(WARM + MAIN):
        src, idx = batch_at(sched, D, q, micro)
        h.update(src.encode())
        h.update(np.ascontiguousarray(idx).tobytes())
    return h.hexdigest()


def lr_main(u, base):
    if u < MAIN_WARM:
        return base * (u + 1) / MAIN_WARM
    t = min((u - MAIN_WARM) / max(HORIZON - MAIN_WARM, 1), 1.0)
    return base * (FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * t)))


# --- потери --------------------------------------------------------------------------
def losses(torch, out, tgt, val, b_t, rv, ids, s):
    """-> (L_total [B], компоненты [B])."""
    import k21_model as km
    F = torch.nn.functional
    dev = out["arm"].device
    with torch.autocast(dev.type, enabled=False):
        la, l1, bce = km.loss_fn(torch, {k: out[k].float() for k in
                                         ("arm", "grip_logit")},
                                 tgt.float(), val, grip_w=GRIP_W)
        rvt = torch.as_tensor(rv, device=dev).float()
        idt = torch.as_tensor(np.where(rv, ids, 0), device=dev).long()
        bt = torch.as_tensor(b_t, device=dev, dtype=torch.float32)
        st = torch.as_tensor(np.asarray(s, np.float32), device=dev)
        lg = out["plan"]["logits"].float()
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), idt.reshape(-1),
                             reduction="none").reshape(rvt.shape)
        nr = rvt.sum(1).clamp_min(1)
        l_code = (ce * rvt).sum(1) / nr
        sl = F.smooth_l1_loss((out["plan"]["b_soft"].float() - bt) / st,
                              torch.zeros_like(bt), beta=1.0,
                              reduction="none").mean(-1)
        l_coarse = (sl * rvt).sum(1) / nr
        acc = (((lg.argmax(-1) == idt).float() * rvt).sum(1) / nr)
        tot = la + W_CODE * l_code + W_COARSE * l_coarse
    return tot, dict(L_action=la, l1=l1, bce=bce, L_code=l_code,
                     L_coarse=l_coarse, root_acc=acc)


# --- модель и данные -------------------------------------------------------------------
def load_data():
    import k21_data as d21
    import k30_data as kd
    D = d21.load_prepared(require_teacher=True)
    C = kd.load()
    return D, C


def check_codec(p, C):
    if C["sha"] != p["codec"]["sha"]:
        raise SystemExit("словарь ≠ закреплённому в prepare")


def make(torch, seed, mode, p, D, C, dev, checkpoint=True, bar=None):
    check_codec(p, C)
    return k30m.make_model(torch, seed, mode, p["sources"][str(seed)],
                           dict(P32=C["P32"], s=C["s"]), D, dev,
                           checkpoint=checkpoint, bar=bar)


def micro_batch(torch, mod, bar, D, C, src, rows, dev, ac, make_batch=None):
    import k21_data as d21
    import k30_data as kd
    mb = make_batch or d21.make_batch
    if src == "teacher":
        tr = D["teacher_rows"][rows]
        batch, po, st, tgt, val = mb(D, tr, bar, dev, teacher_idx=rows)
        b_t, rv, ids = kd.targets(C, rows=tr, teacher_idx=rows)
    else:
        batch, po, st, tgt, val = mb(D, rows, bar, dev)
        b_t, rv, ids = kd.targets(C, rows=rows)
    with ac:
        out = mod(batch, po, st)
    tot, comp = losses(torch, out, tgt, val, b_t, rv, ids, C["s"])
    return out, tot, comp, tgt, val


def evaluate(torch, mod, bar, D, C, rows, dev, micro, make_batch=None):
    """Взвешенные (K-21) метрики val_sel за один проход."""
    import k21_train as kt
    rows = np.asarray(rows)
    w = kt.row_weights(D["task"][rows], D["episode"][rows])
    ac = torch.autocast(dev.type, dtype=torch.float16,
                        enabled=dev.type == "cuda")
    was = mod.training
    mod.eval()
    acc = {k: [] for k in ("L_action", "l1", "l1_8", "bce", "grip_acc",
                           "L_code", "L_coarse", "root_acc", "entropy",
                           "rms_G0", "rms_G1", "rms_G2", "g1", "g2", "recon",
                           "sat")}
    used = np.zeros(k30m.K)
    Lt = torch.as_tensor(np.asarray(D["L"])[:6], device=dev).float()
    Ut = torch.as_tensor(np.asarray(D["U"])[:6], device=dev).float()
    with torch.no_grad():
        for s0 in range(0, len(rows), micro):
            r = rows[s0:s0 + micro]
            out, tot, comp, tgt, val = micro_batch(
                torch, mod, bar, D, C, "expert", r, dev, ac, make_batch)
            if not torch.isfinite(tot).all():
                raise SystemExit("нечисловая проверка")
            v = val.float()
            e = (out["arm"].float() - tgt[..., :6].float()).abs().mean(-1)
            acc["l1_8"].append(((e[:, :8] * v[:, :8]).sum(1) /
                                v[:, :8].sum(1).clamp_min(1)).cpu().numpy())
            g = ((out["grip_logit"] > 0) == (tgt[..., 6] > 0)).float()
            acc["grip_acc"].append(((g * v).sum(1) / v.sum(1)).cpu().numpy())
            for k in ("L_action", "l1", "bce", "L_code", "L_coarse",
                      "root_acc"):
                acc[k].append(comp[k].float().cpu().numpy())
            p_ = out["plan"]["p"].float()
            acc["entropy"].append((-(p_ * (p_ + 1e-12).log()).sum(-1)
                                   .mean(-1)).cpu().numpy())
            used += np.bincount(out["plan"]["ids"].reshape(-1).cpu().numpy(),
                                minlength=k30m.K)
            cd = out["cond"]
            for nm in ("G0", "G1", "G2"):
                acc[f"rms_{nm}"].append(cd[nm].float().pow(2).mean((1, 2))
                                        .sqrt().cpu().numpy())
            acc["g1"].append(cd["g1"].float().mean(1).cpu().numpy())
            acc["g2"].append(cd["g2"].float().mean(1).cpu().numpy())
            sc = torch.stack([cd[k].abs().flatten(1).max(1).values for k in
                              ("Z", "G0", "G1", "G2")], 1).max(1).values \
                .clamp_min(1.0)
            acc["recon"].append(((cd["G0"] + cd["G1"] + cd["G2"] - cd["Z"])
                                 .abs().flatten(1).max(1).values / sc)
                                .cpu().numpy())
            arm = out["arm"].float()
            near = ((arm - Lt).abs() < 0.01 * (Ut - Lt)) | \
                ((Ut - arm).abs() < 0.01 * (Ut - Lt))
            acc["sat"].append(near.float().mean((1, 2)).cpu().numpy())
    if was:
        mod.train()
    res = {}
    for k, v in acc.items():
        x = np.concatenate(v)
        res[k] = float(x.max()) if k == "recon" else \
            float((x * w).sum() / w.sum())
    res["codes_used"] = int((used > 0).sum())
    res["rows"] = int(len(rows))
    return res


# --- тренер -------------------------------------------------------------------------
def counting_scaler(torch):
    import k23_train as k23t
    return k23t.counting_scaler(torch.cuda.amp.GradScaler)


def frozen_hash(torch, mod):
    """Отпечаток того, что не должно меняться: словарь, геометрия, vision,
    базовый текст VLM (без LoRA), dur/split источника."""
    h = hashlib.sha1()
    for n, t in list(mod.named_buffers()) + list(mod.named_parameters()):
        if n.startswith(("codec_", "geo_", "dur.", "split1.", "split2.")) or \
                (n.startswith("bar.") and not n.startswith(
                    "bar.action_expert.") and not n.endswith((".A", ".B"))):
            h.update(n.encode())
            h.update(np.ascontiguousarray(t.detach().float().cpu().numpy())
                     .tobytes())
    return h.hexdigest()


class Trainer:
    def __init__(self, torch, a, seed, mode, p, D, C, odir=None, makef=None,
                 fastload=True):
        self.torch, self.a, self.seed, self.mode = torch, a, seed, mode
        self.p, self.D, self.C = p, D, C
        self.dev = torch.device(a.device)
        self.dir = odir or outdir(seed, mode)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "resume.pt")
        self.mod, self.bar = (makef or make)(torch, seed, mode, p, D, C,
                                             self.dev)
        self.init_hash = k30m.full_hash(self.mod)
        self.frozen0 = frozen_hash(torch, self.mod)
        self.sched = make_schedule(seed)
        self.mb = None
        if fastload and self.dev.type == "cuda":
            import k21_data as d21
            import k21_fastload as fl
            self.pf = fl.Prefetcher(D, self.bar, d21.make_batch, self.sched)
            self.mb = self.pf.make_batch
        self.scaler = counting_scaler(torch)(enabled=self.dev.type == "cuda")
        self.st = dict(phase="warmup", u=0, q=0, curve=[], best=None,
                       cnt=dict(attempted=0), seconds=0.0)
        self.opt = None
        self.ac = torch.autocast(self.dev.type, dtype=torch.float16,
                                 enabled=self.dev.type == "cuda")

    def config(self):
        return dict(mode=self.mode, seed=self.seed, prepare=fp(self.p),
                    micro=self.a.micro, init_hash=self.init_hash,
                    real_check=self.a.__dict__.get("_rc_sha"),
                    micro_file=self.a.__dict__.get("_micro_sha"))

    def make_opt(self, phase):
        torch = self.torch
        self.mod.set_trainable(phase)
        g = self.mod.groups30()
        if phase == "warmup":
            return torch.optim.AdamW([dict(params=g["new"], lr=LR_WARM,
                                           base=LR_WARM, name="new")],
                                     weight_decay=WD)
        return torch.optim.AdamW([dict(params=g[k], lr=LR[k], base=LR[k],
                                       name=k) for k in ("expert", "lora",
                                                         "other", "new")
                                  if g[k]], weight_decay=WD)

    def save(self):
        import k21_train as kt
        torch = self.torch
        if frozen_hash(torch, self.mod) != self.frozen0:
            raise SystemExit("замороженное изменилось")
        st = dict(self.st, weights=kt.trainable_state(self.mod),
                  opt=self.opt.state_dict(), scaler=self.scaler.state_dict(),
                  rng=torch.random.get_rng_state(), config=self.config())
        torch.save(st, self.path + ".tmp")
        os.replace(self.path + ".tmp", self.path)

    def load(self):
        torch = self.torch
        if not os.path.exists(self.path):
            return None
        st = torch.load(self.path, map_location="cpu", weights_only=False)
        if st["config"] != self.config():
            raise SystemExit(f"{self.path}: другой контракт — новый каталог")
        own = dict(self.mod.named_parameters())
        own.update(dict(self.mod.named_buffers()))
        with torch.no_grad():
            for k, v in st["weights"].items():
                own[k].copy_(v.to(own[k].device, own[k].dtype))
        self.scaler.load_state_dict(st["scaler"])
        torch.random.set_rng_state(st["rng"])
        self.st = {k: st[k] for k in self.st}
        print(f"  продолжение: {self.st['phase']} {self.st['u']} (q "
              f"{self.st['q']})", flush=True)
        return st["opt"]

    def save_best(self, crit):
        import k21_train as kt
        torch = self.torch
        st = kt.trainable_state(self.mod)
        p = os.path.join(self.dir, "best.pt")
        torch.save(dict(weights=st, update=self.st["u"], crit=crit,
                        weights_hash=kt.state_hash(st), config=self.config(),
                        mode=self.mode, seed=self.seed,
                        codec_sha=self.C["sha"]), p + ".tmp")
        os.replace(p + ".tmp", p)

    def check(self, tag, candidate):
        t0 = time.time()
        ev = evaluate(self.torch, self.mod, self.bar, self.D, self.C,
                      self.D["val_rows"], self.dev, self.a.eval_micro,
                      self.mb)
        rec = dict(tag=tag, phase=self.st["phase"], u=self.st["u"],
                   q=self.st["q"], val=ev, cnt=dict(self.st["cnt"]),
                   scaler=dict(getattr(self.scaler, "_k23", {})),
                   grad=self.st.get("last_grad"), seconds=time.time() - t0)
        self.st["curve"].append(rec)
        if candidate and (self.st["best"] is None or
                          ev["L_action"] < self.st["best"][1]):
            self.st["best"] = (self.st["u"], ev["L_action"])
            self.save_best(ev["L_action"])
        print(f"  {k30m.MODES[self.mode]} s{self.seed} {tag}: L_action "
              f"{ev['L_action']:.4f} (L1 {ev['l1']:.4f}, 8 {ev['l1_8']:.4f}, "
              f"схват {ev['grip_acc']:.3f}); CE {ev['L_code']:.3f}, root acc "
              f"{ev['root_acc']:.3f}, coarse {ev['L_coarse']:.4f}, энтропия "
              f"{ev['entropy']:.2f}, кодов {ev['codes_used']}; RMS G0/G1/G2 "
              f"{ev['rms_G0']:.3f}/{ev['rms_G1']:.3f}/{ev['rms_G2']:.3f}; "
              f"шагов {rec['scaler'].get('steps')}, пропусков "
              f"{rec['scaler'].get('skips')}; лучшая {self.st['best']} "
              f"({rec['seconds']:.0f} с)", flush=True)

    def step(self, lr_of):
        torch = self.torch
        for g in self.opt.param_groups:
            g["lr"] = lr_of(g)
        src, idx = batch_at(self.sched, self.D, self.st["q"], self.a.micro)
        if self.mb is not None:
            self.pf.note_schedule(Q_OFFSET + self.st["q"], src, idx)
        self.opt.zero_grad(set_to_none=True)
        for rows in idx:
            _o, tot, _c, _t, _v = micro_batch(
                torch, self.mod, self.bar, self.D, self.C, src, rows, self.dev,
                self.ac, self.mb)
            L = tot.mean()
            if not bool(torch.isfinite(L)):
                raise SystemExit("нечисловая потеря")
            self.scaler.scale(L / len(idx)).backward()
        self.scaler.unscale_(self.opt)
        ps = [p for g in self.opt.param_groups for p in g["params"]]
        gn = float(torch.nn.utils.clip_grad_norm_(ps, CLIP))
        self.scaler.step(self.opt)
        self.scaler.update()
        c = getattr(self.scaler, "_k23", {})
        if c.get("consecutive_skips", 0) >= MAX_SKIPS:
            raise SystemExit(f"{MAX_SKIPS} пропусков подряд")
        self.st["cnt"]["attempted"] += 1
        self.st["last_grad"] = gn

    def first_batches_check(self):
        """Первый expert и первый teacher microbatch оставшейся очереди:
        fastload = прежний путь, метки кеша = пересчёту из целей."""
        if self.mb is None:
            return
        import k21_data as d21
        import k21_fastload as fl
        import k30_codec as kc
        import k30_data as kd
        seen = set()
        for q in range(self.st["q"], WARM + MAIN):
            src, idx = batch_at(self.sched, self.D, q, self.a.micro)
            if src in seen:
                continue
            seen.add(src)
            rows = idx[0]
            ti = rows if src == "teacher" else None
            rr = self.D["teacher_rows"][rows] if src == "teacher" else rows
            self.pf.note_schedule(Q_OFFSET + q, src, idx)
            a_ = self.pf.make_batch(self.D, rr, self.bar, self.dev, ti)
            b_ = d21.make_batch(self.D, rr, self.bar, self.dev, ti)
            if not fl.same_batch(a_, b_):
                raise SystemExit(f"fastload ≠ прежнему пути ({src})")
            tgt, val = b_[3].cpu().numpy(), b_[4].cpu().numpy()
            bt, rv, _ = kc.plan_targets(tgt, val, self.D["L"], self.D["U"])
            b0, rv0, _i = kd.targets(self.C, rows=rr, teacher_idx=ti)
            if not (np.array_equal(rv, rv0) and np.allclose(bt, b0,
                                                            atol=1e-6)):
                raise SystemExit(f"кеш меток ≠ пересчёту ({src})")
            if len(seen) == 2:
                break
        print(f"  очередь: первые microbatch {sorted(seen)} — fastload и "
              f"метки сверены", flush=True)

    def run(self):
        opt_state = self.load()
        self.opt = self.make_opt(self.st["phase"])
        if opt_state is not None:
            self.opt.load_state_dict(opt_state)
        self.mod.train()
        self.first_batches_check()
        if self.st["phase"] == "warmup" and self.st["u"] == 0 and \
                not self.st["curve"]:
            self.check("warmup 0", False)
        t0 = time.time()
        while True:
            ph, u = self.st["phase"], self.st["u"]
            if ph == "warmup" and u >= WARM:
                self.check("warmup 500", False)
                self.st.update(phase="main", u=0)
                self.opt = self.make_opt("main")
                self.save()
                continue
            if ph == "main" and u >= MAIN:
                break
            if ph == "warmup":
                self.step(lambda g: LR_WARM)
            else:
                self.step(lambda g, u=u: lr_main(u, g["base"]))
            self.st["u"] = u + 1
            self.st["q"] += 1
            if self.st["cnt"]["attempted"] == 50:
                el = time.time() - t0
                print(f"  {k30m.MODES[self.mode]} s{self.seed}: "
                      f"{50 / el:.2f} обн./с; до конца ~"
                      f"{(WARM + MAIN - self.st['q']) * el / 50 / 3600:.1f} ч "
                      f"без проверок", flush=True)
            if ph == "main" and self.st["u"] % EVAL_EVERY == 0:
                self.check(f"main {self.st['u']}", True)
            if self.st["u"] % SAVE_EVERY == 0:
                self.st["seconds"] += time.time() - t0
                t0 = time.time()
                self.save()
        self.save()
        return self.st


# --- micro / real-check / fitcheck ----------------------------------------------------
def rc_contract(a):
    import torch
    p = load_prepare()
    return dict(prepare=fp(p), code=code_sha(),
                device=torch.cuda.get_device_name(torch.device(a.device)),
                precision="fp16 autocast + float32 план/readout",
                micros=list(MICROS))


def fresh_problems(kind, a):
    """Свежесть real-check (и micro.json) или fitcheck против ТЕКУЩИХ
    prepare, кода, устройства и SHA отчётов. Пустой список — свежо. Одна
    функция для check-fresh, fitcheck и train (до создания модели)."""
    try:
        p = load_prepare()
    except SystemExit as e:
        return [str(e.code)]
    pr = []
    if not (os.path.exists(REAL_CHECK) and os.path.exists(MICRO_FILE)):
        return ["нет real_check.json/micro.json"]
    r = json.load(open(REAL_CHECK, encoding="utf-8"))
    m = json.load(open(MICRO_FILE, encoding="utf-8"))
    if r.get("problems") != []:
        pr.append("real-check не пройден")
    if r.get("contract") != rc_contract(a):
        pr.append("real-check другого контракта (prepare/код/устройство)")
    if m.get("prepare") != fp(p) or m.get("real_check_sha1") != \
            sha(REAL_CHECK) or m.get("micro") != r.get("micro_choice"):
        pr.append("micro.json не согласован с real_check/prepare")
    if kind == "fitcheck":
        if not os.path.exists(FITCHECK):
            return pr + ["нет fitcheck.json"]
        f = json.load(open(FITCHECK, encoding="utf-8"))
        if not f.get("ok"):
            pr.append("fitcheck не пройден")
        want = dict(rc_contract(a), micro=m.get("micro"),
                    micro_file_sha1=sha(MICRO_FILE),
                    real_check_sha1=sha(REAL_CHECK))
        if f.get("contract") != want:
            pr.append("fitcheck другого контракта")
    return pr


def decided_micro(a=None):
    """micro из micro.json — только при свежем real-check текущего
    контракта (при a) или при согласии файлов с prepare (без a)."""
    if not (os.path.exists(MICRO_FILE) and os.path.exists(REAL_CHECK)):
        return None
    if a is not None:
        pr = fresh_problems("real-check", a)
        if pr:
            raise SystemExit("; ".join(pr))
    else:
        m = json.load(open(MICRO_FILE, encoding="utf-8"))
        r = json.load(open(REAL_CHECK, encoding="utf-8"))
        if r.get("problems") != [] or m.get("prepare") != fp(load_prepare()) \
                or m.get("real_check_sha1") != sha(REAL_CHECK) or \
                m.get("micro") != r.get("micro_choice"):
            raise SystemExit("micro.json не согласован с real_check/prepare")
    return int(json.load(open(MICRO_FILE, encoding="utf-8"))["micro"])


def applied_step(torch, mod, opt, scaler, fn):
    """До STEP_TRIES попыток; -> (применён, нормы после unscale)."""
    import k21_train as kt
    h0 = kt.state_hash({n: p.detach().cpu() for n, p in
                        mod.named_parameters() if p.requires_grad})
    gn = None
    for _ in range(STEP_TRIES):
        opt.zero_grad(set_to_none=True)
        steps0 = scaler._k23["steps"]
        L = fn()
        scaler.scale(L).backward()
        scaler.unscale_(opt)
        gn = {g["name"]: float(sum(float((p.grad.float() ** 2).sum())
                                   for p in g["params"] if p.grad is not None)
                               ** 0.5) for g in opt.param_groups}
        ok = all(math.isfinite(v) for v in gn.values())
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in
                                        g["params"]], CLIP)
        scaler.step(opt)
        scaler.update()
        if scaler._k23["steps"] > steps0 and ok:
            h1 = kt.state_hash({n: p.detach().cpu() for n, p in
                                mod.named_parameters() if p.requires_grad})
            return h1 != h0, gn
    return False, gn


def free_cuda(torch):
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cpu_batch(batch):
    return {k: (v.detach().cpu() if hasattr(v, "detach") else v)
            for k, v in batch.items()}


def _to_dev(batch, dev):
    return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}


def source_part(torch, s, p, D, C, dev, ac, rows5):
    """Источник seed s: (1) настоящий K-25 cond (k25_policy) на B = 1/5 ->
    выходы на CPU, модель освобождается; (2) H и C фабрики K-30 по очереди:
    K25.forward на общих весах (source_forward) = (1) побитово; основной
    hybrid-forward — счётчики, конечность, пределы, пересборка, FP32, план;
    отпечаток полного state_dict — на CPU. Возвращает только CPU-данные."""
    import k21_data as d21
    import k21_train as kt
    import k25_policy as kp25
    e = p["sources"][str(s)]
    r, prob = {}, []
    spec = dict(arch="parents_cond24", seed=s, ckpt=e["ckpt"],
                weights_hash=e["weights_hash"])
    ref, bar_r, _ck, _pr = kp25.load_policy(
        torch, f"parents_cond24_s{s}", spec, str(dev))
    inputs, ref_out = {}, {}
    for B in (1, 5):
        batch, po, st, _t, _v = d21.make_batch(D, rows5[:B], bar_r, dev)
        with torch.no_grad(), ac:
            o = ref(batch, po, st)
        inputs[B] = (_cpu_batch(batch), po.detach().cpu(), st.detach().cpu())
        ref_out[B] = (o["arm"].detach().cpu(), o["grip_logit"].detach().cpu())
        del o, batch
    del ref, bar_r, _ck
    free_cuda(torch)
    hashes = {}
    for mode in ("hard", "soft"):
        m_, b_ = make(torch, s, mode, p, D, C, dev, checkpoint=False)
        m_.eval()
        hashes[mode] = kt.state_hash({k: v.detach().cpu() for k, v in
                                      m_.state_dict().items()})
        if mode == "hard":
            n_new = sum(q.numel() for n, q in m_.named_parameters()
                        if n.startswith(k30m.NEW_PREFIX))
            n_src = sum(v.numel() for v in torch.load(
                e["ckpt"], map_location="cpu", weights_only=False)[
                    "weights"].values())
            r["new_params"], r["new_share"] = n_new, n_new / n_src
            if n_new != k30m.new_param_count(m_.H) or r["new_share"] >= 0.01:
                prob.append(f"s{s}: новых параметров {n_new}")
        for B in (1, 5):
            bc, po, st = inputs[B]
            batch = _to_dev(bc, dev)
            po, st = po.to(dev), st.to(dev)
            with torch.no_grad(), ac:
                ol = m_.source_forward(batch, po, st)
                eq = bool(torch.equal(ol["arm"].cpu(), ref_out[B][0]) and
                          torch.equal(ol["grip_logit"].cpu(), ref_out[B][1]))
                del ol
                o = m_(batch, po, st)
            r[f"source_legacy_{mode}_B{B}"] = eq
            if not eq:
                prob.append(f"s{s} {mode} B{B}: K25.forward на весах фабрики "
                            f"K-30 ≠ настоящему K-25")
            cn, pl, cd = m_.counters, o["plan"], o["cond"]
            L = torch.as_tensor(np.asarray(D["L"])[:6], device=dev)
            U = torch.as_tensor(np.asarray(D["U"])[:6], device=dev)
            want_b = C["P32"][pl["ids"].cpu().numpy()] if mode == "hard" \
                else (pl["p"].double().cpu().numpy() @
                      C["P32"].astype(np.float64))
            rr = dict(
                classifier=cn["classifier"], vision=cn["vision"],
                layers_once=all(x == 1 for x in cn["text"]),
                active_ok=cn["active"] == k30m.ACTIVE,
                finite=all(bool(torch.isfinite(x).all()) for x in
                           (o["arm"], o["grip_logit"], pl["logits"])),
                in_bounds=bool(((o["arm"] >= L - 1e-4) &
                                (o["arm"] <= U + 1e-4)).all()),
                recon=float((cd["G0"] + cd["G1"] + cd["G2"] - cd["Z"]).abs()
                            .max()),
                fp32=all(x.dtype == torch.float32 for x in (
                    pl["logits"], pl["p"], pl["b_used"], cd["Z"], o["arm"])),
                plan_diff=float(np.abs(pl["b_used"].double().cpu().numpy() -
                                       want_b).max()))
            r[f"{mode}_B{B}"] = rr
            bad = [k for k, v in rr.items() if v is False]
            if rr["classifier"] != 1:
                bad.append(f"classifier {rr['classifier']}")
            if not rr["recon"] <= 1e-4:
                bad.append("recon")
            if not rr["plan_diff"] <= 2e-6:
                bad.append("plan")
            if bad:
                prob.append(f"s{s} {mode} B{B}: {bad}")
            del o, pl, cd, batch
        del m_, b_
        free_cuda(torch)
    r["pair_init_equal"] = hashes["hard"] == hashes["soft"]
    if not r["pair_init_equal"]:
        prob.append(f"s{s}: начальные тензоры H ≠ C")
    return r, prob


def step_part(torch, s, mode, mb, p, D, C, dev, ac):
    """Отдельная область жизни: модель создаётся, шаги warmup и main,
    память и время; только CPU-результат; освобождение до следующего."""
    free_cuda(torch)
    base = torch.cuda.memory_allocated(dev) / 2 ** 30
    m_, b_ = make(torch, s, mode, p, D, C, dev)
    r = step_check(torch, m_, b_, D, C, dev, ac, mb, p, s, mode)
    r["alloc_before_gb"] = base
    del m_, b_
    free_cuda(torch)
    return r


def write_conflicts(p, choice):
    """Конфликты, проверяемые ДО записи real_check.json и micro.json:
    другой micro для того же prepare; уже начатое обучение."""
    conf = []
    if os.path.exists(MICRO_FILE):
        old = json.load(open(MICRO_FILE, encoding="utf-8"))
        if old.get("prepare") == fp(p) and old.get("micro") != choice:
            conf.append(f"micro уже выбран для этого prepare: "
                        f"{old.get('micro')} ≠ {choice}")
    started = [f for f in (report_path(s, m) for s in SEEDS
                           for m in k30m.MODES) if os.path.exists(f)] + \
        [d for d in (outdir(s, m) for s in SEEDS for m in k30m.MODES)
         if os.path.exists(os.path.join(d, "resume.pt"))]
    if started:
        conf.append(f"обучение уже начато ({started[:2]}): real-check не "
                    f"перезаписывается")
    return conf, started


def real_check(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = load_prepare()
    D, C = load_data()
    check_codec(p, C)
    dev = torch.device(a.device)
    ac = torch.autocast("cuda", dtype=torch.float16)
    prob, rep = [], dict(seeds={"0": {}, "1": {}})
    tr = np.asarray(D["train_rows"])
    rows5 = tr[np.argsort(D["task"][tr], kind="stable")][:5]
    for s in SEEDS:
        r, pr = source_part(torch, s, p, D, C, dev, ac, rows5)
        rep["seeds"][str(s)].update(r)
        prob += pr
        print(f"  seed {s}: источник/инициализация " + json.dumps(
            r, default=float)[:1200], flush=True)
    for mode in ("hard", "soft"):
        for mb in MICROS:
            r = step_part(torch, 0, mode, mb, p, D, C, dev, ac)
            rep["seeds"]["0"][f"{mode}_micro{mb}"] = r
            if r.get("bad"):
                prob.append(f"s0 {mode} micro {mb}: {r['bad']}")
            print(f"  s0 {mode} micro {mb}: " + json.dumps(
                {k: (v if k != "warmup" and k != "main" else
                     {kk: vv for kk, vv in v.items() if kk in
                      ("peak_gb", "seconds", "bad")})
                 for k, v in r.items()}, default=float), flush=True)
    ok_m = [mb for mb in MICROS
            if all(not rep["seeds"]["0"][f"{m}_micro{mb}"].get("bad") and
                   not rep["seeds"]["0"][f"{m}_micro{mb}"].get("oom")
                   for m in ("hard", "soft"))]
    if 8 not in ok_m:
        prob.append("micro 8 не прошёл")
    choice = 32 if 32 in ok_m else (8 if 8 in ok_m else None)
    rep["micro_choice"] = choice
    if choice:
        for mode in ("hard", "soft"):
            r = step_part(torch, 1, mode, choice, p, D, C, dev, ac)
            rep["seeds"]["1"][f"{mode}_micro{choice}"] = r
            if r.get("bad") or r.get("oom"):
                prob.append(f"s1 {mode} micro {choice}: "
                            f"{'OOM' if r.get('oom') else r['bad']}")
    # все конфликты — до записи обоих файлов
    conf, started = write_conflicts(p, choice)
    prob += conf
    rep["problems"] = prob
    rep["contract"] = rc_contract(a)
    if started:
        print("  ОТКАЗ: " + "; ".join(prob), flush=True)
        return 3
    os.makedirs("reports/k30", exist_ok=True)
    with open(REAL_CHECK + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(REAL_CHECK + ".tmp", REAL_CHECK)
    if not prob:
        with open(MICRO_FILE + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(dict(micro=choice, accumulation=EFF // choice,
                           prepare=fp(p), real_check_sha1=sha(REAL_CHECK)),
                      fh, indent=1)
        os.replace(MICRO_FILE + ".tmp", MICRO_FILE)
    print("  " + ("ОТКАЗ: " + "; ".join(prob) if prob else
                  f"real-check K-30 пройден; micro {choice}"), flush=True)
    return 3 if prob else 0


def step_check(torch, mod, bar, D, C, dev, ac, mb, p, seed, mode):
    """Шаг warmup и шаг main на mb строках: применён, конечен, меняются
    разрешённые группы, замороженное неизменно; память, время."""
    import k21_train as kt
    res = {}
    rows = np.asarray(D["train_rows"])[::max(1, len(D["train_rows"]) //
                                             mb)][:mb]
    fz = frozen_hash(torch, mod)
    try:
        for phase in ("warmup", "main"):
            mod.train()
            mod.set_trainable(phase)
            g = mod.groups30()
            opt = torch.optim.AdamW(
                [dict(params=g[k], lr=(LR_WARM if phase == "warmup" else
                                       LR[k]), name=k)
                 for k in ("expert", "lora", "other", "new") if g[k]],
                weight_decay=WD)
            sc = counting_scaler(torch)()
            before = {k: kt.state_hash({str(i): p.detach().cpu() for i, p in
                                        enumerate(v)}) for k, v in g.items()
                      if v}
            def untrained():
                """Необучаемые в этой фазе (включая эксперт и LoRA в
                warmup); базовый VLM — в frozen_hash."""
                return kt.state_hash(
                    {n: q.detach().cpu() for n, q in mod.named_parameters()
                     if not q.requires_grad and (
                         not n.startswith("bar.") or
                         n.startswith("bar.action_expert.") or
                         n.endswith((".A", ".B")))})
            old_frozen_all = untrained()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.time()

            def fn():
                _o, tot, _c, _t, _v = micro_batch(torch, mod, bar, D, C,
                                                  "expert", rows, dev, ac)
                return tot.mean()
            applied, gn = applied_step(torch, mod, opt, sc, fn)
            torch.cuda.synchronize(dev)
            after = {k: kt.state_hash({str(i): p.detach().cpu() for i, p in
                                       enumerate(v)}) for k, v in g.items()
                     if v}
            changed = {k: before[k] != after[k] for k in before}
            same_untrained = untrained() == old_frozen_all
            r = dict(applied=applied, grad_norms=gn, changed=changed,
                     untrained_same=same_untrained,
                     frozen_same=frozen_hash(torch, mod) == fz,
                     finite=all(bool(torch.isfinite(q).all())
                                for q in mod.parameters() if q.requires_grad),
                     seconds=time.time() - t0,
                     peak_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30)
            bad = [k for k in ("applied", "untrained_same", "frozen_same",
                               "finite") if not r[k]]
            bad += [f"не изменилась группа {k}" for k, v in changed.items()
                    if not v]
            if gn and not all(math.isfinite(v) for v in gn.values()):
                bad.append("градиенты не конечны")
            r["bad"] = bad
            res[phase] = r
            del opt
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return dict(oom=True, bad=[])
    res["bad"] = [f"{ph}: {b}" for ph in ("warmup", "main")
                  for b in res[ph]["bad"]]
    return res


def fit_rows(D, n=FIT_N, seed=20261103):
    tr = np.asarray(D["train_rows"])
    task = np.asarray(D["task"])[tr]
    ts = sorted(set(task.tolist()))
    rng = np.random.default_rng(seed)
    per = max(1, n // len(ts))
    out = []
    for t in ts:
        c = tr[task == t]
        out += rng.choice(c, min(per, len(c)), replace=False).tolist()
    rest = np.setdiff1d(tr, out)
    out += rng.choice(rest, max(0, n - len(out)), replace=False).tolist()
    return np.sort(np.asarray(out[:n]))


def fitcheck(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    p = load_prepare()
    micro = decided_micro(a)
    if micro is None or micro != a.micro:
        raise SystemExit(f"micro {a.micro} ≠ решению real-check {micro}")
    D, C = load_data()
    dev = torch.device(a.device)
    ac = torch.autocast("cuda", dtype=torch.float16)
    rows = fit_rows(D)
    act = np.asarray(D["act"])[rows][..., :6]
    val = np.asarray(D["valid"])[rows].astype(bool)
    const = np.array([act[..., c][val].mean() for c in range(6)])
    l1_const = float(np.mean([np.abs(act[i][val[i]] - const).mean()
                              for i in range(len(rows))]))
    rep, prob = dict(rows=rows.tolist(), l1_const=l1_const), []
    for mode in ("hard", "soft"):
        mod, bar = make(torch, 0, mode, p, D, C, dev)
        init = k30m.full_hash(mod)
        fz = frozen_hash(torch, mod)
        mod.set_trainable("main")
        g = mod.groups30()
        opt = torch.optim.AdamW([dict(params=g[k], lr=LR[k] * FIT_LR_MULT,
                                      name=k) for k in ("expert", "lora",
                                                        "other", "new")
                                 if g[k]], weight_decay=WD)
        sc = counting_scaler(torch)()

        def ev():
            mod.eval()
            out = {}
            with torch.no_grad():
                acc = {k: [] for k in ("l1", "grip", "ce")}
                for s0 in range(0, len(rows), micro):
                    r_ = rows[s0:s0 + micro]
                    o, tot, comp, tgt, v = micro_batch(
                        torch, mod, bar, D, C, "expert", r_, dev, ac)
                    acc["l1"].append(comp["l1"].float().cpu().numpy())
                    gg = ((o["grip_logit"] > 0) == (tgt[..., 6] > 0)).float()
                    vv = v.float()
                    acc["grip"].append(((gg * vv).sum(1) / vv.sum(1)).cpu()
                                       .numpy())
                    acc["ce"].append(comp["L_code"].float().cpu().numpy())
                out = {k: float(np.concatenate(v).mean()) for k, v in
                       acc.items()}
            mod.train()
            return out
        e0 = ev()
        rng = np.random.default_rng(20261104)
        mod.train()
        gseen = {}
        for u in range(FIT_UPDATES):
            ids = rng.choice(rows, EFF, replace=False)
            opt.zero_grad(set_to_none=True)
            for s0 in range(0, EFF, micro):
                _o, tot, _c, _t, _v = micro_batch(
                    torch, mod, bar, D, C, "expert", ids[s0:s0 + micro], dev,
                    ac)
                sc.scale(tot.mean() * micro / EFF).backward()
            sc.unscale_(opt)
            if u == 0:
                gseen = {k: any(p_.grad is not None and float(p_.grad.abs()
                                                              .sum()) > 0
                                for p_ in v) for k, v in g.items() if v}
            torch.nn.utils.clip_grad_norm_([q for v in g.values() for q in v],
                                           CLIP)
            sc.step(opt)
            sc.update()
            if (u + 1) % 100 == 0:
                e = ev()
                print(f"  fitcheck {k30m.MODES[mode]} {u + 1}: L1 {e['l1']:.4f} "
                      f"(старт {e0['l1']:.4f}, константа {l1_const:.4f}), "
                      f"схват {e['grip']:.3f}, CE {e['ce']:.3f} (старт "
                      f"{e0['ce']:.3f})", flush=True)
        e1 = ev()
        checks = dict(
            finite=all(math.isfinite(v) for v in e1.values()),
            l1_halved=e1["l1"] <= e0["l1"] / 2,
            below_const=e1["l1"] < l1_const,
            grip_95=e1["grip"] >= 0.95,
            ce_drop_20=e1["ce"] <= 0.8 * e0["ce"],
            grads_new=bool(gseen.get("new")),
            frozen_same=frozen_hash(torch, mod) == fz,
            applied=sc._k23["steps"] > 0)
        mod = bar = opt = None                  # освободить до пересоздания
        torch.cuda.empty_cache()
        m2, _b2 = make(torch, 0, mode, p, D, C, dev)
        checks["reinit_same"] = k30m.full_hash(m2) == init
        del m2, _b2
        torch.cuda.empty_cache()
        rep[mode] = dict(start=e0, end=e1, checks=checks)
        bad = [k for k, v in checks.items() if not v]
        if bad:
            prob.append(f"{mode}: {bad}")
        print(f"  fitcheck {k30m.MODES[mode]}: {checks}", flush=True)
    rep["ok"] = not prob
    rep["problems"] = prob
    rep["contract"] = dict(rc_contract(a), micro=micro,
                           micro_file_sha1=sha(MICRO_FILE),
                           real_check_sha1=sha(REAL_CHECK))
    with open(FITCHECK + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(FITCHECK + ".tmp", FITCHECK)
    print("  " + ("fitcheck ПРОЙДЕН" if not prob else
                  "fitcheck НЕ ПРОЙДЕН: " + "; ".join(prob)), flush=True)
    return 0 if not prob else 3


def check_fresh(a):
    pr = fresh_problems(a.kind, a)
    print(f"  {a.kind}: " + ("пройден тем же контрактом" if not pr else
                             "пересчитать: " + "; ".join(pr)))
    return 0 if not pr else 1


def train(a):
    import torch
    import k21_train as kt
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if a.seed not in SEEDS or a.mode not in k30m.MODES:
        raise SystemExit("seed/режим")
    p = load_prepare()
    pr = fresh_problems("fitcheck", a)          # до создания модели
    if pr:
        raise SystemExit("проверки не свежи: " + "; ".join(pr))
    micro = decided_micro(a)
    if micro != a.micro:
        raise SystemExit(f"micro {a.micro} ≠ решению {micro}")
    a._rc_sha, a._micro_sha = sha(REAL_CHECK), sha(MICRO_FILE)
    D, C = load_data()
    tr = Trainer(torch, a, a.seed, a.mode, p, D, C)
    qsha = queue_sha(tr.sched, D, micro)
    print(f"  K-30 {k30m.label_of(a.mode, a.seed)}: warmup {WARM} + main "
          f"{MAIN}, micro {micro} × {EFF // micro}; очередь {qsha[:12]}",
          flush=True)
    st = tr.run()
    b = torch.load(os.path.join(tr.dir, "best.pt"), map_location="cpu",
                   weights_only=False)
    rep = dict(seed=a.seed, mode=a.mode, label=k30m.label_of(a.mode, a.seed),
               best=st["best"], best_sha1=sha(os.path.join(tr.dir, "best.pt")),
               best_weights_hash=b["weights_hash"], init_hash=tr.init_hash,
               counters=st["cnt"], scaler=dict(getattr(tr.scaler, "_k23", {})),
               curve=st["curve"], queue_sha=qsha, seconds=st["seconds"],
               config=tr.config(), prepare=fp(p))
    if kt.state_hash(b["weights"]) != b["weights_hash"]:
        raise SystemExit("best.pt повреждён")
    with open(report_path(a.seed, a.mode), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=float)
    print(f"  {rep['label']}: лучшая {st['best']}; "
          f"{report_path(a.seed, a.mode)}", flush=True)
    return 0


def selftest():
    import tempfile
    import types
    import torch
    import k21_data as d21
    import k21_model as km
    import k21_train as kt
    import k30_codec as kc
    torch.backends.cuda.matmul.allow_tf32 = False
    M = k30m.build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    rng = np.random.default_rng(0)
    N = 80
    act = np.concatenate([rng.uniform(-1, 1, (N, 20, 6)),
                          np.sign(rng.normal(size=(N, 20, 1)))], -1) \
        .astype(np.float32)
    valid = np.ones((N, 20), np.float32)
    valid[::6, 14:] = 0
    tt = act[np.arange(0, 40, 4)] + 0.01
    D = dict(train_rows=np.arange(60), val_rows=np.arange(60, 72),
             teacher_rows=np.arange(0, 40, 4), act=act, valid=valid,
             task=np.array([f"t{i % 4}" for i in range(N)]),
             episode=np.arange(N) // 5, L=L, U=U, teacher_target=tt,
             state_mean=None, state_std=None, prep_report=dict(prep_id="P"))
    P = (rng.uniform(-0.8, 0.8, (k30m.K, 6))).astype(np.float32)
    s = np.full(6, 0.3)
    b_all, rv_all, _ = kc.plan_targets(act, valid, L, U)
    ids_all = kc.labels(b_all, rv_all, P, s)
    b_te, rv_te, _ = kc.plan_targets(tt, np.ones((10, 20)), L, U)
    C = dict(P32=P, s=s, sha="toy", pos=np.arange(N), b=b_all, rv=rv_all,
             ids=ids_all, b_teacher=b_te, rv_teacher=rv_te,
             id_teacher=kc.labels(b_te, rv_te, P, s),
             teacher_rows=D["teacher_rows"])
    bar0 = km.fake_bar(torch)

    def fake_mb(D_, rows, bar, dev, teacher_idx=None):
        rows = np.asarray(rows)
        b = km.fake_batch(torch, B=len(rows), seed=int(rows.sum()) % 97)
        st_ = torch.as_tensor(rng.normal(size=(len(rows), 11)) * 0,
                              dtype=torch.float32) + float(rows.sum() % 7)
        if teacher_idx is None:
            tgt = torch.as_tensor(act[rows])
            val = torch.as_tensor(valid[rows])
        else:
            tgt = torch.as_tensor(tt[np.asarray(teacher_idx)])
            val = torch.ones(tgt.shape[:2])
        return b, 1, st_, tgt, val

    def fake_make(torch_, seed, mode, p_, D_, C_, dev, checkpoint=True,
                  bar=None):
        torch.manual_seed(0)
        m = M["K30"](copy.deepcopy(bar0), L, U, C_["P32"], C_["s"],
                     seed=seed, plan_mode=mode)
        m.set_trainable("main")
        return m, dict(model=m)
    # потери: метки не входят в forward; L_coarse по b_soft у обоих
    m, _b = fake_make(torch, 0, "hard", None, D, C, "cpu")
    m.eval()
    saved = d21.make_batch
    try:
        d21.make_batch = fake_mb
        o, tot, comp, tgt, val = micro_batch(
            torch, m, None, D, C, "expert", np.arange(4),
            torch.device("cpu"), torch.autocast("cpu", enabled=False))
        C2 = dict(C, ids=(C["ids"] + 1) % k30m.K)
        o2, tot2, comp2, _t, _v = micro_batch(
            torch, m, None, D, C2, "expert", np.arange(4),
            torch.device("cpu"), torch.autocast("cpu", enabled=False))
        assert torch.equal(o["arm"], o2["arm"]) and not torch.equal(
            comp["L_code"], comp2["L_code"])
        want = comp["L_action"] + W_CODE * comp["L_code"] + \
            W_COARSE * comp["L_coarse"]
        assert torch.allclose(tot, want)
        # тренер: непрерывный = прерванный + возобновление
        g = globals()
        old = (g["WARM"], g["MAIN"], g["EVAL_EVERY"], g["SAVE_EVERY"],
               g["EFF"], g["HORIZON"], g["MAIN_WARM"])
        g["WARM"], g["MAIN"], g["EVAL_EVERY"], g["SAVE_EVERY"], g["EFF"], \
            g["HORIZON"], g["MAIN_WARM"] = 2, 4, 2, 1, 8, 4, 2
        a_ = types.SimpleNamespace(device="cpu", micro=4, eval_micro=4)
        res = {}
        with tempfile.TemporaryDirectory() as td:
            for name, stop in (("cont", None), ("resume", 3)):
                odir = os.path.join(td, name)
                if stop:
                    T_ = Trainer(torch, a_, 0, "hard", {}, D, C, odir=odir,
                                 makef=fake_make, fastload=False)
                    calls = dict(n=0)
                    orig = T_.step

                    def step(*x, _o=orig, _c=calls):
                        _c["n"] += 1
                        if _c["n"] > stop:
                            raise KeyboardInterrupt
                        return _o(*x)
                    T_.step = step
                    try:
                        T_.run()
                    except KeyboardInterrupt:
                        pass
                T2 = Trainer(torch, a_, 0, "hard", {}, D, C, odir=odir,
                             makef=fake_make, fastload=False)
                T2.run()
                res[name] = (k30m.full_hash(T2.mod), T2.st["best"],
                             [(c["tag"], round(c["val"]["L_action"], 9))
                              for c in T2.st["curve"]])
            assert res["cont"] == res["resume"], res
            assert res["cont"][1][0] in (2, 4)        # только main
        # каждая обучаемая — ровно в одной группе
        tr_ = Trainer(torch, a_, 0, "soft", {}, D, C, odir=tempfile.mkdtemp(),
                      makef=fake_make, fastload=False)
        opt = tr_.make_opt("main")
        ids_ = [id(p_) for gg in opt.param_groups for p_ in gg["params"]]
        assert len(ids_) == len(set(ids_)) == sum(
            1 for p_ in tr_.mod.parameters() if p_.requires_grad)
        opt = tr_.make_opt("warmup")
        assert len(opt.param_groups) == 1 and \
            len(opt.param_groups[0]["params"]) == 5
    finally:
        d21.make_batch = saved
        if "old" in dir():
            (g["WARM"], g["MAIN"], g["EVAL_EVERY"], g["SAVE_EVERY"], g["EFF"],
             g["HORIZON"], g["MAIN_WARM"]) = old
    # очередь: u_abs = 100000 + q, teacher при u_abs % 4 == 3, не зависит
    # от режима, различается по сиду
    D2 = dict(D, train_rows=np.arange(60))
    s0 = make_schedule(0)
    seq = [batch_at(s0, D2, q, 8)[0] for q in range(8)]
    assert seq == ["teacher" if (Q_OFFSET + q) % 4 == 3 else "expert"
                   for q in range(8)]
    assert not np.array_equal(batch_at(s0, D2, 0, 8)[1],
                              batch_at(make_schedule(1), D2, 0, 8)[1])
    assert abs(lr_main(HORIZON, 1.0) - FLOOR) < 1e-12 and \
        lr_main(0, 1.0) == 1.0 / MAIN_WARM
    _ = kt
    # свежесть: устаревший fitcheck не пропускает train до создания модели;
    # конфликт micro и начатое обучение — до записи real-check
    with tempfile.TemporaryDirectory() as td:
        names = ("PREPARE", "REAL_CHECK", "MICRO_FILE", "FITCHECK",
                 "load_prepare", "rc_contract", "Trainer", "outdir",
                 "report_path")
        keep = {n: g[n] for n in names}
        try:
            g["PREPARE"] = os.path.join(td, "prepare.json")
            g["REAL_CHECK"] = os.path.join(td, "rc.json")
            g["MICRO_FILE"] = os.path.join(td, "micro.json")
            g["FITCHECK"] = os.path.join(td, "fit.json")
            pfake = dict(x=1)
            g["load_prepare"] = lambda: pfake
            g["rc_contract"] = lambda a__: dict(prepare=fp(pfake), dev="D")
            g["outdir"] = lambda s_, m_: os.path.join(td, f"run_{s_}_{m_}")
            g["report_path"] = lambda s_, m_: os.path.join(td,
                                                           f"tr_{s_}_{m_}.json")

            def boom(*x, **k):
                raise AssertionError("Trainer создан при несвежих проверках")
            g["Trainer"] = boom
            json.dump(dict(problems=[], micro_choice=32,
                           contract=dict(prepare=fp(pfake), dev="D")),
                      open(g["REAL_CHECK"], "w"))
            json.dump(dict(micro=32, prepare=fp(pfake),
                           real_check_sha1=sha(g["REAL_CHECK"])),
                      open(g["MICRO_FILE"], "w"))
            good = dict(dict(prepare=fp(pfake), dev="D"), micro=32,
                        micro_file_sha1=sha(g["MICRO_FILE"]),
                        real_check_sha1=sha(g["REAL_CHECK"]))
            json.dump(dict(ok=True, contract=good), open(g["FITCHECK"], "w"))
            a2 = types.SimpleNamespace(device="cpu", micro=32, seed=0,
                                       mode="hard")
            assert fresh_problems("fitcheck", a2) == []
            json.dump(dict(ok=True, contract=dict(good, micro=8)),
                      open(g["FITCHECK"], "w"))
            assert fresh_problems("fitcheck", a2)
            try:
                train(a2)
                raise AssertionError("train принял устаревший fitcheck")
            except SystemExit as e:
                assert "не свежи" in str(e.code)
            # micro.json от другого real-check — несвежо
            json.dump(dict(micro=32, prepare=fp(pfake),
                           real_check_sha1="0" * 40),
                      open(g["MICRO_FILE"], "w"))
            assert fresh_problems("real-check", a2)
            # конфликты до записи
            json.dump(dict(micro=32, prepare=fp(pfake)),
                      open(g["MICRO_FILE"], "w"))
            c1, _s = write_conflicts(pfake, 8)
            assert c1 and write_conflicts(pfake, 32)[0] == []
            os.makedirs(g["outdir"](0, "hard"))
            open(os.path.join(g["outdir"](0, "hard"), "resume.pt"), "w") \
                .close()
            assert write_conflicts(pfake, 32)[1]
        finally:
            for n, v in keep.items():
                g[n] = v
    print("самопроверка k30_train пройдена: метки не входят в forward, "
          "L_total = L_action + 0.02·L_code + 0.1·L_coarse, непрерывное = "
          "прерванное + возобновление (веса, кривая, best только в main), "
          "каждая обучаемая — в одной группе, warmup — только новые, "
          "очередь u_abs/teacher/сид, LR; свежесть: устаревший fitcheck и "
          "чужой micro.json отвергаются до создания модели, конфликты micro и "
          "начатого обучения — до записи real-check")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-30: обучение и проверки")
    ap.add_argument("cmd", choices=("prepare", "real-check", "fitcheck",
                                    "train", "check-fresh", "micro"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=tuple(k30m.MODES), default="hard")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--eval-micro", type=int, default=16)
    ap.add_argument("--kind", choices=("real-check", "fitcheck"),
                    default="real-check")
    a = ap.parse_args()
    try:
        if a.cmd == "prepare":
            return prepare()
        if a.cmd == "micro":
            m = decided_micro()
            if m is None:
                raise SystemExit("решения о micro нет")
            print(m)
            return 0
        return {"real-check": real_check, "fitcheck": fitcheck,
                "train": train, "check-fresh": check_fresh}[a.cmd](a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
