#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-28: обучение надстройки parents_goal_fb24 (seed 0 и 1) поверх
замороженного parents24 того же сида; real-check, fitcheck, check-fresh.

Строки — только экспертные train K-21 (учитель не используется); выборка
задачи -> эпизоды -> строки; эффективный батч 32, micro из real-check.
Обновление: один вызов замороженной базы на начальных наблюдениях (fp16
autocast, no_grad), затем новые модули во float32 вне autocast; контроллер
на шагах j = 0…19 с реальными S[t+j] (k28_data) и экспертными A[t+j].
Потери: L_pose = 0.5 L_knots + 0.5 L_dense (ell = SmoothL1((p − p*)/σ_p) +
‖R − R*‖²_F / (4σ_R²); уровни корни/средние/листья с равным весом, пустые
уровни пропускаются с перенормировкой; dense j = 1…20); L_arm — L1
исправленной руки к экспертной команде (единицы кодека) по действительным
шагам; L_corr = mean (Δa/κ)²; L = L_arm + 0.1 L_pose + 0.001 L_corr.
Бюджет: разогрев 500 (только головы поз, AdamW 3e−4, L_pose), основная 2000
(все новые, AdamW 1e−4, разогрев LR 100, косинус 2000 до 0.1, wd 0.01, clip
1.0). GradScaler: применённые шаги и пропуски — отдельно; 50 подряд без
шага — отказ. Проверки: до/после разогрева и каждые 250 основных; best —
среди основных 250…2000 по взвешенной L1 первых 8 шагов на val_sel (при
равенстве — ранняя). База сверяется с исходным отпечатком при сохранении
и в конце.
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

import k28_control as kc  # noqa: E402
import k28_model as k28m  # noqa: E402

SEEDS = (0, 1)
EFF = 32
WARM, MAIN = 500, 2000
LR_WARM, LR_MAIN, WD, CLIP = 3e-4, 1e-4, 0.01, 1.0
MAIN_WARM, HORIZON, FLOOR = 100, 2000, 0.1
EVAL_EVERY = SAVE_EVERY = 250
W_POSE, W_CORR = 0.1, 0.001
MAX_SKIPS = 50
FIT_N, FIT_UPDATES, FIT_LR = 128, 500, 3e-4
FIT_L1_DROP, FIT_POSE_RATIO = 0.10, 2.0
MICRO_FILE = "reports/k28/micro.json"
REAL_CHECK = "reports/k28/real_check.json"
STEP_TRIES = 8
CODE = ("k28_geometry.py", "k28_control.py", "k28_model.py", "k28_train.py",
        "k28_data.py", "k22_model.py", "k21_model.py", "k21_tree.py",
        "k21_data.py", "k20_data.py", "k16_models.py", "k21_train.py",
        "k23_train.py", "k21_fastload.py")
HP = dict(eff=EFF, warm=WARM, main=MAIN, lr_warm=LR_WARM, lr_main=LR_MAIN,
          wd=WD, clip=CLIP, main_warm=MAIN_WARM, horizon=HORIZON,
          floor=FLOOR, eval_every=EVAL_EVERY, w_pose=W_POSE, w_corr=W_CORR,
          best="min weighted val_sel L1 first 8, main 250..2000",
          sources="expert train rows K-21 only (no teacher)")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def outdir(seed):
    return f"data/k28/runs/s{seed}/{k28m.MODE}"


def report_path(seed):
    return f"reports/k28/train_s{seed}.json"


def lr_main(u):
    if u < MAIN_WARM:
        return LR_MAIN * (u + 1) / MAIN_WARM
    t = min((u - MAIN_WARM) / max(HORIZON - MAIN_WARM, 1), 1.0)
    return LR_MAIN * (FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * t)))


def row_probs(D, rows):
    import k21_train as kt
    w = kt.row_weights(np.asarray(D["task"])[rows],
                       np.asarray(D["episode"])[rows])
    return w / w.sum()


def make_schedule(D, seed, micro):
    """Очередь: обновление u (сквозная нумерация разогрев+основная) ->
    [("expert", строки [EFF/micro, micro])]; формат k21_fastload."""
    tr = np.asarray(D["train_rows"])
    p = row_probs(D, tr)

    def sched(D_, n_updates, micro_, seed_=None, start=0):
        out = []
        for u in range(start, n_updates):
            g = np.random.default_rng([seed, 28, u])
            idx = g.choice(len(tr), EFF, replace=True, p=p)
            out.append(("expert", tr[idx].reshape(EFF // micro_, micro_)))
        return out
    return sched


def schedule_fp(sched, D, micro, points=(0, 1, 499, 500, 2499)):
    h = hashlib.sha1()
    for u in points:
        h.update(np.ascontiguousarray(sched(D, u + 1, micro, start=u)[0][1])
                 .tobytes())
    return h.hexdigest()


def seed_entry(seed):
    import k28_protocol as pc
    return pc.seed_entry(seed)


def base_hash(mod):
    import k21_train as ktr
    return ktr.state_hash(k28m.shared_keys(ktr.trainable_state(mod)))


def new_state(mod):
    import k21_train as ktr
    return k28m.new_weights(ktr.trainable_state(mod))


def new_hash(mod):
    import k21_train as ktr
    return ktr.state_hash(new_state(mod))


# --- модель ---------------------------------------------------------------------
def load_model(torch, seed, D, S, dev, entry, checkpoint=False):
    import k21_data as d21
    import k21_protocol as pp
    src = entry["source"]
    if pp.full_sha(src["ckpt"]) != src["ckpt_sha1"]:
        raise SystemExit(f"{src['ckpt']} ≠ закреплённому SHA")
    ck = torch.load(src["ckpt"], map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if cfg.get("mode") != "parents24" or int(cfg.get("seed", -1)) != seed:
        raise SystemExit(f"источник: {cfg.get('mode')} seed {cfg.get('seed')}")
    if cfg["prep_id"] != D["prep_report"]["prep_id"]:
        raise SystemExit("источник обучен на другой подготовке")
    bar = d21.load_bar(dev, D["prep_report"]["bar_pin"])
    m = bar["model"]
    m.action_expert.float()
    mod = k28m.build(torch)["K28"](
        m, D["L"], D["U"], D["state_mean"], D["state_std"],
        checkpoint=checkpoint, seed=seed, sigma_p=S["sigma_p"],
        sigma_R=S["sigma_R"]).to(dev)
    own = dict(mod.named_parameters())
    own.update(dict(mod.named_buffers()))
    with torch.no_grad():
        for k, v in ck["weights"].items():
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    h = base_hash(mod)
    if h != src["weights_hash"] or h != ck["weights_hash"]:
        raise SystemExit(f"база {h} ≠ отпечатку источника")
    return mod, bar


# --- потери ---------------------------------------------------------------------
def pose_losses(torch, mod, pl, seq, ok):
    import k28_data as kd
    G = mod_geo(torch)
    sp = mod.sigma_p
    sR = mod.sigma_R
    F = torch.nn.functional

    def ell(p, R, pt, Rt):
        return F.smooth_l1_loss((p - pt) / sp, torch.zeros_like(p),
                                reduction="none").mean(-1) + \
            G["frob2"](R, Rt) / (4 * sR * sR)
    lv, parts, cnt = pl["levels"], [], {}
    for name in ("root", "mid", "leaf"):
        t_, p_, R_ = lv[name]
        pt, Rt, v = kd.state_at(torch, seq, ok, t_.detach().float().cpu())
        v = v.to(p_.device)
        cnt[name] = int(v.sum())
        if cnt[name]:
            parts.append(ell(p_, R_, pt.to(p_.device).float(),
                             Rt.to(p_.device).float())[v].mean())
    L_knots = torch.stack(parts).mean() if parts else \
        pl["ref_p"].sum() * 0
    okd = torch.as_tensor(ok[:, 1:21], device=pl["ref_p"].device)
    st = torch.as_tensor(seq[:, 1:21], dtype=torch.float32,
                         device=pl["ref_p"].device)
    pt, Rt = G["pose_of_state8"](st)
    if bool(okd.any()):
        L_dense = ell(pl["ref_p"][:, 1:21], pl["ref_R"][:, 1:21], pt,
                      Rt)[okd].mean()
    else:
        L_dense = pl["ref_p"].sum() * 0
    return 0.5 * L_knots + 0.5 * L_dense, dict(
        knots=float(L_knots.detach()), dense=float(L_dense.detach()), counts=cnt,
        empty_levels=[k for k, v in cnt.items() if v == 0])


_GEO = {}


def mod_geo(torch):
    if "G" not in _GEO:
        import k28_geometry as kg
        _GEO["G"] = kg.build(torch)
    return _GEO["G"]


def full_loss(torch, mod, pl, seq, ok, tgt, val, warm):
    L_pose, pinfo = pose_losses(torch, mod, pl, seq, ok)
    if warm:
        return L_pose, dict(L_pose=float(L_pose.detach()), **pinfo)
    ca = mod.control_all(pl, seq)
    okj = torch.as_tensor(ok[:, :20], device=tgt.device) & val.bool()
    v = okj.float()
    nv = v.sum(1).clamp_min(1)
    l1 = (ca["a"] - tgt[..., :6].float()).abs().mean(-1)
    L_arm = ((l1 * v).sum(1) / nv).mean()
    k_, _L, _U = kc.build(torch)["kappa"](mod._L, mod._U, tgt.device)
    L_corr = (((ca["da"] / k_) ** 2).mean(-1) * v).sum() / v.sum().clamp_min(1)
    L = L_arm + W_POSE * L_pose + W_CORR * L_corr
    return L, dict(L=float(L.detach()), L_arm=float(L_arm.detach()),
                   L_pose=float(L_pose.detach()),
                   L_corr=float(L_corr.detach()), **pinfo)


def grad_norms(mod, scale):
    """Нормы градиентов по группам новых модулей (делённые на масштаб)."""
    out = {}
    for k, v in mod.new_modules().items():
        t = sum(float((p.grad.float() ** 2).sum()) for mm in v
                for p in mm.parameters() if p.grad is not None)
        out[k] = math.sqrt(t) / scale
    return out


# --- оценка ---------------------------------------------------------------------
def evaluate(torch, mod, bar, D, S, rows, dev, micro, make_batch=None):
    """Взвешенные (задачи -> эпизоды -> строки) L1 руки 8/20 с поправкой и
    без (база), ошибки поз, величины поправок, насыщения."""
    import k21_data as d21
    import k21_train as kt
    mb = make_batch or d21.make_batch
    rows = np.asarray(rows)
    w = kt.row_weights(np.asarray(D["task"])[rows],
                       np.asarray(D["episode"])[rows])
    G = mod_geo(torch)
    ac = torch.autocast("cuda", dtype=torch.float16,
                        enabled=dev.type == "cuda")
    was = mod.training
    mod.eval()
    acc = {k: [] for k in ("l1_8", "l1_20", "b8", "b20")}
    pe, re_, da_, sat, clip, Lp = [], [], [], [], [], []
    k_, _L, _U = kc.build(torch)["kappa"](mod._L, mod._U, dev)
    with torch.no_grad():
        for s0 in range(0, len(rows), micro):
            r = rows[s0:s0 + micro]
            batch, po, st, tgt, val = mb(D, r, bar, dev)
            seq = S["state_seq8"][S["pos"][r]]
            ok = S["state_valid"][S["pos"][r]]
            st8 = torch.as_tensor(np.asarray(D["state8"])[r], dtype=torch.float32,
                                  device=dev)
            pl = mod.plan(batch, po, st, st8, autocast=ac)
            ca = mod.control_all(pl, seq)
            okj = torch.as_tensor(ok[:, :20], device=dev) & val.bool()
            v = okj.float()
            for nm, a in (("", ca["a"]), ("b", pl["a0"])):
                l1 = (a - tgt[..., :6].float()).abs().mean(-1)
                acc[("l1_8" if not nm else "b8")].append(
                    ((l1[:, :8] * v[:, :8]).sum(1) / v[:, :8].sum(1)
                     .clamp_min(1)).cpu().numpy())
                acc[("l1_20" if not nm else "b20")].append(
                    ((l1 * v).sum(1) / v.sum(1).clamp_min(1)).cpu().numpy())
            okd = torch.as_tensor(ok[:, 1:21], device=dev)
            stt = torch.as_tensor(seq[:, 1:21], dtype=torch.float32,
                                  device=dev)
            ptr, Rtr = G["pose_of_state8"](stt)
            pe.append((pl["ref_p"][:, 1:21] - ptr).norm(dim=-1)[okd].cpu())
            re_.append(G["rot_err"](pl["ref_R"][:, 1:21], Rtr).norm(dim=-1)
                       [okd].cpu())
            m8 = okj[:, :8]
            da_.append((ca["da"][:, :8].abs() / k_)[m8].flatten().cpu())
            sat.append(ca["kappa_sat"][:, :8][m8].float().flatten().cpu())
            clip.append(ca["clip"][:, :8][m8].float().flatten().cpu())
            Lp.append(float(pose_losses(torch, mod, pl, seq, ok)[0]))
    if was:
        mod.train()

    def wm(x):
        x = np.concatenate(x)
        return float((x * w).sum() / w.sum())

    def q(x):
        x = torch.cat(x).float().numpy() if x else np.zeros(1)
        return [float(np.median(x)), float(np.percentile(x, 95))]
    return dict(l1_8=wm(acc["l1_8"]), l1_20=wm(acc["l1_20"]),
                base_l1_8=wm(acc["b8"]), base_l1_20=wm(acc["b20"]),
                pose_err_m=q(pe), rot_err_rad=q(re_), da_over_kappa=q(da_),
                kappa_sat=float(torch.cat(sat).mean()) if sat else 0.0,
                clip=float(torch.cat(clip).mean()) if clip else 0.0,
                L_pose=float(np.mean(Lp)), rows=int(len(rows)))


# --- контракт -------------------------------------------------------------------
def run_contract(seed, a, entry, S):
    import torch
    import transformers
    import k21_data as d21
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    return dict(mode=k28m.MODE, architecture=k28m.ARCHITECTURE, seed=seed,
                source=entry["source"], seed_entry_fp=entry["fingerprint"],
                base_hash=entry["source"]["weights_hash"],
                prep_id=prep["prep_id"], k28_prep_id=S["report"]["k28_prep_id"],
                bar_revision=prep["bar_pin"]["revision"],
                sigma_p=[float(x) for x in S["sigma_p"]],
                sigma_R=float(S["sigma_R"]), control=kc.CONFIG,
                sizes=dict(head_hid=k28m.HEAD_HID, ctx_h=k28m.CTX_H,
                           nodes=k28m.N_NODES), hp=HP, micro=a.micro,
                dtype=a.dtype, torch=torch.__version__,
                transformers=transformers.__version__,
                code={f: sha(os.path.join(HERE, f)) for f in CODE})


def check_contract(kind, a):
    import torch
    import k21_data as d21
    import k21_protocol as pp
    import k28_data as kd
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    c = dict(kind=kind, code={f: sha(os.path.join(HERE, f)) for f in CODE},
             prep_id=prep["prep_id"], prepare_sha1=pp.full_sha(d21.REPORT),
             k28_prepare_sha1=pp.full_sha(kd.REPORT),
             bar_revision=prep["bar_pin"]["revision"],
             entries=[seed_entry(s)["fingerprint"] for s in SEEDS],
             precision=f"{a.dtype} autocast база, float32 надстройка",
             device=torch.cuda.get_device_name(torch.device(a.device)),
             control=kc.CONFIG)
    if kind == "real-check":
        c.update(micros=sorted(a.micros))
    else:
        c.update(fit=dict(n=FIT_N, updates=FIT_UPDATES, lr=FIT_LR,
                          drop=FIT_L1_DROP, pose_ratio=FIT_POSE_RATIO,
                          micro=a.micro, seed=0))
    return c


def write_json(path, obj):
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(path + ".tmp", path)


def read_micro():
    if not os.path.exists(MICRO_FILE):
        return None
    return json.load(open(MICRO_FILE, encoding="utf-8"))


def micro_problems(rep, mrec, want=None):
    """Согласие отчёта real-check, решения micro и (при want) текущего
    контракта. Пустой список — решение действительно."""
    p = []
    if rep is None:
        return ["нет отчёта real-check"]
    if rep.get("problems") != []:
        p.append("real-check не пройден")
    if mrec is None:
        return p + ["нет решения micro"]
    if mrec.get("micro") != rep.get("micro_choice"):
        p.append(f"micro {mrec.get('micro')} ≠ выбору отчёта "
                 f"{rep.get('micro_choice')}")
    if mrec.get("contract") != rep.get("contract"):
        p.append("решение micro от другого контракта real-check")
    if want is not None:
        got = rep.get("contract") or {}
        diff = sorted(k for k in set(want) | set(got)
                      if want.get(k) != got.get(k))
        if diff:
            p.append(f"контракт изменился: {diff}")
    return p


def check_fresh(a):
    p = REAL_CHECK if a.kind == "real-check" else \
        "reports/k28/fitcheck.json"
    if not os.path.exists(p):
        print(f"  {p}: нет")
        return 1
    r = json.load(open(p, encoding="utf-8"))
    want = check_contract(a.kind, a)
    if a.kind == "real-check":
        prob = micro_problems(r, read_micro(), want)
    else:
        got = r.get("contract") or {}
        diff = sorted(k for k in set(want) | set(got)
                      if want.get(k) != got.get(k))
        prob = ([] if r.get("ok") else ["fitcheck не пройден"]) + \
            ([f"контракт изменился: {diff}"] if diff else [])
    print(f"  {p}: " + ("пройден тем же контрактом" if not prob else
                        "пересчитать: " + "; ".join(prob)))
    return 0 if not prob else 1


# --- тренер ---------------------------------------------------------------------
class Trainer:
    def __init__(self, torch, a, seed, D, S, entry, odir=None, make=None,
                 fastload=True):
        import k23_train as k23t
        self.torch, self.a, self.seed, self.D, self.S = torch, a, seed, D, S
        self.dev = torch.device(a.device)
        self.entry = entry
        self.dir = odir or outdir(seed)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "resume.pt")
        self.mod, self.bar = (make or load_model)(torch, seed, D, S,
                                                  self.dev, entry)
        self.base0 = base_hash(self.mod)
        self.init_new = new_hash(self.mod)
        self.sched = make_schedule(D, seed, a.micro)
        self.mb = None
        if fastload and self.dev.type == "cuda":
            import k21_data as d21
            import k21_fastload as fl
            self.pf = fl.Prefetcher(D, self.bar, d21.make_batch, self.sched)
            self.mb = self.pf.make_batch
        Sc = k23t.counting_scaler(torch.cuda.amp.GradScaler)
        self.scaler = Sc(enabled=self.dev.type == "cuda")
        self.st = dict(phase="warmup", u=0, curve=[], best=None,
                       cnt=dict(attempted=0), seconds=0.0)
        self.opt = None
        self.ac = torch.autocast("cuda", dtype=torch.float16,
                                 enabled=self.dev.type == "cuda")

    def make_opt(self, phase):
        torch = self.torch
        if phase == "warmup":
            ps = self.mod.pose_params()
            return torch.optim.AdamW(ps, lr=LR_WARM, weight_decay=WD)
        return torch.optim.AdamW(self.mod.groups()["new"], lr=LR_MAIN,
                                 weight_decay=WD)

    def set_trainable(self, phase):
        for p in self.mod.groups()["new"]:
            p.requires_grad_(phase == "main")
        for p in self.mod.pose_params():
            p.requires_grad_(True)

    def config(self):
        return run_contract(self.seed, self.a, self.entry, self.S)

    def save(self):
        torch = self.torch
        st = dict(self.st, new=new_state(self.mod),
                  opt=self.opt.state_dict(), scaler=self.scaler.state_dict(),
                  rng=torch.random.get_rng_state(), config=self.config(),
                  init_new=self.init_new, base_hash=self.base0)
        torch.save(st, self.path + ".tmp")
        os.replace(self.path + ".tmp", self.path)
        if base_hash(self.mod) != self.base0:
            raise SystemExit("база изменилась во время обучения")

    def load(self):
        torch = self.torch
        if not os.path.exists(self.path):
            return None
        st = torch.load(self.path, map_location="cpu", weights_only=False)
        if st["config"] != self.config():
            raise SystemExit(f"{self.path}: другая конфигурация")
        if st["init_new"] != self.init_new or st["base_hash"] != self.base0:
            raise SystemExit("начальные новые веса или база ≠ закреплённым")
        own = dict(self.mod.named_parameters())
        with torch.no_grad():
            for k, v in st["new"].items():
                own[k].copy_(v.to(own[k].device, own[k].dtype))
        self.scaler.load_state_dict(st["scaler"])
        torch.random.set_rng_state(st["rng"])
        self.st = {k: st[k] for k in self.st}
        print(f"  продолжение: фаза {self.st['phase']}, обновление "
              f"{self.st['u']}", flush=True)
        return st["opt"]

    def check(self, tag, candidate):
        t0 = time.time()
        ev = evaluate(self.torch, self.mod, self.bar, self.D, self.S,
                      self.D["val_rows"], self.dev, self.a.eval_micro,
                      self.mb)
        sc = dict(getattr(self.scaler, "_k23", {}))
        rec = dict(tag=tag, phase=self.st["phase"], u=self.st["u"], val=ev,
                   scaler=sc, cnt=dict(self.st["cnt"]),
                   grad=self.st.get("last_grad"), seconds=time.time() - t0,
                   base_unchanged=base_hash(self.mod) == self.base0)
        self.st["curve"].append(rec)
        if not rec["base_unchanged"]:
            raise SystemExit("база изменилась")
        if candidate and (self.st["best"] is None or
                          ev["l1_8"] < self.st["best"][1]):
            self.st["best"] = (self.st["u"], ev["l1_8"])
            p = os.path.join(self.dir, "best.pt")
            self.torch.save(dict(weights=new_state(self.mod),
                                 update=self.st["u"], crit=ev["l1_8"],
                                 config=self.config(),
                                 new_hash=new_hash(self.mod),
                                 base_hash=self.base0,
                                 init_new=self.init_new), p + ".tmp")
            os.replace(p + ".tmp", p)
        print(f"  {k28m.MODE} s{self.seed} {tag}: val L1-8 {ev['l1_8']:.4f} "
              f"(база {ev['base_l1_8']:.4f}), L1-20 {ev['l1_20']:.4f} "
              f"({ev['base_l1_20']:.4f}); поза {ev['pose_err_m'][0]:.4f}/"
              f"{ev['pose_err_m'][1]:.4f} м, {ev['rot_err_rad'][0]:.4f}/"
              f"{ev['rot_err_rad'][1]:.4f} рад; |Δa|/κ "
              f"{ev['da_over_kappa'][0]:.3f}/{ev['da_over_kappa'][1]:.3f}, "
              f"κ-насыщение {ev['kappa_sat']:.3f}, clip {ev['clip']:.3f}; "
              f"шагов {sc.get('steps')}, пропусков {sc.get('skips')}; лучшая "
              f"{self.st['best']} ({rec['seconds']:.0f} с)", flush=True)

    def step(self, u_global, warm, lr):
        torch = self.torch
        for g in self.opt.param_groups:
            g["lr"] = lr
        src, idx = self.sched(self.D, u_global + 1, self.a.micro,
                              start=u_global)[0]
        if self.mb is not None:
            self.pf.note_schedule(u_global, src, idx)
        import k21_data as d21
        mb = self.mb or d21.make_batch
        self.opt.zero_grad(set_to_none=True)
        for rows in idx:
            batch, po, st, tgt, val = mb(self.D, rows, self.bar, self.dev)
            seq = self.S["state_seq8"][self.S["pos"][rows]]
            ok = self.S["state_valid"][self.S["pos"][rows]]
            st8 = torch.as_tensor(np.asarray(self.D["state8"])[rows],
                                  dtype=torch.float32, device=self.dev)
            pl = self.mod.plan(batch, po, st, st8, autocast=self.ac)
            L, info = full_loss(torch, self.mod, pl, seq, ok, tgt, val, warm)
            if not bool(torch.isfinite(L)):
                raise SystemExit(f"нечисловая потеря: {info}")
            self.scaler.scale(L / len(idx)).backward()
        if any(p.grad is not None for n, p in self.mod.named_parameters()
               if not n.startswith(k28m.NEW_PREFIXES)):
            raise SystemExit("градиент у замороженной базы")
        self.scaler.unscale_(self.opt)
        ps = [p for g in self.opt.param_groups for p in g["params"]]
        gn = torch.nn.utils.clip_grad_norm_(ps, CLIP)
        self.scaler.step(self.opt)
        self.scaler.update()
        c = getattr(self.scaler, "_k23", {})
        if c.get("consecutive_skips", 0) >= MAX_SKIPS:
            raise SystemExit(f"{MAX_SKIPS} попыток подряд без шага")
        self.st["cnt"]["attempted"] += 1
        if (u_global + 1) % EVAL_EVERY == 0 or u_global < 2:
            self.st["last_grad"] = dict(total=float(gn), **grad_norms(
                self.mod, 1.0))

    def run(self):
        opt_state = self.load()
        self.mod.train()
        self.set_trainable(self.st["phase"])
        self.opt = self.make_opt(self.st["phase"])
        if opt_state is not None:
            self.opt.load_state_dict(opt_state)
        t_start = time.time()
        if self.st["phase"] == "warmup" and self.st["u"] == 0 and \
                not self.st["curve"]:
            self.check("начало", False)
        while True:
            ph, u = self.st["phase"], self.st["u"]
            if ph == "warmup" and u >= WARM:
                self.check("конец разогрева", False)
                self.st.update(phase="main", u=0)
                self.set_trainable("main")
                self.opt = self.make_opt("main")
                self.save()
                continue
            if ph == "main" and u >= MAIN:
                break
            ug = u if ph == "warmup" else WARM + u
            self.step(ug, ph == "warmup",
                      LR_WARM if ph == "warmup" else lr_main(u))
            self.st["u"] = u + 1
            if self.st["cnt"]["attempted"] == 100:
                el = time.time() - t_start
                print(f"  {k28m.MODE} s{self.seed}: {100 / el:.2f} обн./с; "
                      f"до конца ~{(WARM + MAIN - ug - 1) * el / 100 / 3600:.1f}"
                      f" ч без проверок", flush=True)
            if ph == "main" and self.st["u"] % EVAL_EVERY == 0:
                self.check(f"основная {self.st['u']}", True)
            if self.st["u"] % SAVE_EVERY == 0:
                self.st["seconds"] += time.time() - t_start
                t_start = time.time()
                self.save()
        self.save()
        return self.st


# --- режимы ----------------------------------------------------------------------
def decided_micro():
    """Решение micro, только если оно согласовано с пройденным отчётом
    real-check того же контракта; иначе отказ."""
    m = read_micro()
    if m is None:
        return None
    rep = json.load(open(REAL_CHECK, encoding="utf-8")) \
        if os.path.exists(REAL_CHECK) else None
    p = micro_problems(rep, m)
    if p:
        raise SystemExit(f"{MICRO_FILE}: " + "; ".join(p))
    return int(m["micro"])


def load_data():
    import k21_data as d21
    import k28_data as kd
    D = d21.load_prepared(require_teacher=False)
    S = kd.load()
    return D, S


def train(a):
    import torch
    if a.seed not in SEEDS:
        raise SystemExit(f"seed {a.seed}: в K-28 только {SEEDS}")
    if decided_micro() != a.micro:
        raise SystemExit(f"micro {a.micro} ≠ решению {MICRO_FILE}")
    e = seed_entry(a.seed)
    D, S = load_data()
    tr = Trainer(torch, a, a.seed, D, S, e)
    print(f"  K-28 seed {a.seed}: разогрев {WARM} + основная {MAIN}, micro "
          f"{a.micro} × {EFF // a.micro}; база {tr.base0[:12]}; новых "
          f"параметров {k28m.new_param_count(tr.mod)}", flush=True)
    st = tr.run()
    if base_hash(tr.mod) != e["source"]["weights_hash"]:
        raise SystemExit("база в конце ≠ источнику")
    b = torch.load(os.path.join(tr.dir, "best.pt"), map_location="cpu",
                   weights_only=False)
    rep = dict(seed=a.seed, mode=k28m.MODE, warm=WARM, main=MAIN,
               best=st["best"], best_sha1=sha(os.path.join(tr.dir, "best.pt")),
               best_new_hash=b["new_hash"], base_hash=tr.base0,
               init_new=tr.init_new, counters=st["cnt"],
               scaler=dict(getattr(tr.scaler, "_k23", {})),
               seconds=st["seconds"], curve=st["curve"], config=tr.config(),
               schedule_fp=schedule_fp(tr.sched, D, a.micro),
               note="надстройка над замороженным parents24; источники новых "
                    "голов — экспертные строки и реальные будущие состояния; "
                    "учитель не используется")
    os.makedirs("reports/k28", exist_ok=True)
    json.dump(rep, open(report_path(a.seed), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"  K-28 seed {a.seed}: лучшая {st['best']}; {report_path(a.seed)}",
          flush=True)
    return 0


def fit_rows(D, n=None, seed=20261031):
    n = FIT_N if n is None else n
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
    out += rng.choice(rest, n - len(out), replace=False).tolist()
    return np.sort(np.asarray(out[:n]))


def fresh_state_used(torch, mod, bar, D, S, rows, dev, ac):
    """Поправка шагов j ≥ 1 меняется при сдвиге наблюдаемых состояний
    (план и шаг 0 — нет): контроллер читает свежее состояние."""
    import k21_data as d21
    mod.eval()
    with torch.no_grad():
        batch, po, st, _t, _v = d21.make_batch(D, rows, bar, dev)
        seq = S["state_seq8"][S["pos"][rows]]
        st8 = torch.as_tensor(np.asarray(D["state8"])[rows],
                              dtype=torch.float32, device=dev)
        pl = mod.plan(batch, po, st, st8, autocast=ac)
        seq2 = seq.copy()
        seq2[:, 1:, :3] += 0.01
        a1 = mod.control_all(pl, seq)["a"]
        a2 = mod.control_all(pl, seq2)["a"]
    mod.train()
    return bool(torch.equal(a1[:, 0], a2[:, 0]) and
                not torch.equal(a1[:, 1:8], a2[:, 1:8]))


def fitcheck(a):
    import torch
    import k23_train as k23t
    e = seed_entry(0)
    D, S = load_data()
    dev = torch.device(a.device)
    mod, bar = load_model(torch, 0, D, S, dev, e)
    rows = fit_rows(D)
    h0, n0 = base_hash(mod), new_hash(mod)
    ev0 = evaluate(torch, mod, bar, D, S, rows, dev, a.eval_micro)
    opt = torch.optim.AdamW(mod.groups()["new"], lr=FIT_LR, weight_decay=WD)
    scaler = k23t.counting_scaler(torch.cuda.amp.GradScaler)()
    ac = torch.autocast("cuda", dtype=torch.float16)
    rng = np.random.default_rng(20261032)
    mod.train()
    for p in mod.groups()["new"]:
        p.requires_grad_(True)
    import k21_data as d21
    curve = [dict(u=0, **{k: ev0[k] for k in ("l1_8", "base_l1_8",
                                              "L_pose")})]
    for u in range(1, FIT_UPDATES + 1):
        ids = rng.choice(rows, EFF, replace=False)
        opt.zero_grad(set_to_none=True)
        for s0 in range(0, EFF, a.micro):
            r = ids[s0:s0 + a.micro]
            batch, po, st, tgt, val = d21.make_batch(D, r, bar, dev)
            seq = S["state_seq8"][S["pos"][r]]
            ok = S["state_valid"][S["pos"][r]]
            st8 = torch.as_tensor(np.asarray(D["state8"])[r],
                                  dtype=torch.float32, device=dev)
            pl = mod.plan(batch, po, st, st8, autocast=ac)
            L, _i = full_loss(torch, mod, pl, seq, ok, tgt, val, False)
            scaler.scale(L * len(r) / EFF).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(mod.groups()["new"], CLIP)
        scaler.step(opt)
        scaler.update()
        if u % 100 == 0:
            ev = evaluate(torch, mod, bar, D, S, rows, dev, a.eval_micro)
            curve.append(dict(u=u, **{k: ev[k] for k in (
                "l1_8", "base_l1_8", "L_pose", "da_over_kappa")}))
            print(f"  fitcheck {u}: L1-8 {ev['l1_8']:.4f} (база "
                  f"{ev['base_l1_8']:.4f}), L_pose {ev['L_pose']:.4f} (старт "
                  f"{ev0['L_pose']:.4f}), |Δa|/κ {ev['da_over_kappa']}; шагов "
                  f"{scaler._k23['steps']}, пропусков {scaler._k23['skips']}",
                  flush=True)
    fin = curve[-1]
    checks = dict(
        l1_8_drop=fin["l1_8"] <= (1 - FIT_L1_DROP) * ev0["base_l1_8"],
        pose_halved=fin["L_pose"] <= ev0["L_pose"] / FIT_POSE_RATIO,
        finite=bool(np.isfinite([fin["l1_8"], fin["L_pose"]]).all()),
        base_unchanged=base_hash(mod) == h0, new_trained=new_hash(mod) != n0,
        corrections_nonzero=fin["da_over_kappa"][1] > 0,
        corrections_bounded=fin["da_over_kappa"][1] <= 1.0 + 1e-6,
        optimizer_applied=scaler._k23["steps"] > 0)
    checks["fresh_state_used"] = fresh_state_used(torch, mod, bar, D, S,
                                                  rows[:8], dev, ac)
    ok = all(checks.values())
    # пересоздание: начальные новые веса воспроизводятся
    del mod, bar, opt
    torch.cuda.empty_cache()
    m2, _b2 = load_model(torch, 0, D, S, dev, e)
    checks["reinit_same"] = new_hash(m2) == n0
    ok = ok and checks["reinit_same"]
    rep = dict(ok=ok, checks=checks, start=ev0, curve=curve,
               rows=rows.tolist(), scaler=scaler._k23,
               contract=check_contract("fitcheck", a))
    os.makedirs("reports/k28", exist_ok=True)
    json.dump(rep, open("reports/k28/fitcheck.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"  fitcheck: {'ПРОЙДЕН' if ok else 'НЕ ПРОЙДЕН'} {checks}; L1-8 "
          f"{ev0['base_l1_8']:.4f} -> {fin['l1_8']:.4f}, L_pose "
          f"{ev0['L_pose']:.4f} -> {fin['L_pose']:.4f}", flush=True)
    return 0 if ok else 3


def train_step_check(torch, mod, bar, D, S, dev, ac, mb):
    """Полный шаг обучения на mb строках: forward, backward, AdamW. До
    STEP_TRIES попыток с GradScaler (пропуск при inf/NaN — повтор с меньшим
    масштабом), иначе отказ. Нормы градиентов групп — после unscale_, до
    clipping; требуются конечные и ненулевые (головы поз, ctrl), реально
    применённый шаг (счётчик GradScaler и изменённый отпечаток новых
    весов), неизменная база. Время со сборкой батча и без, пик памяти."""
    import k21_data as d21
    import k23_train as k23t
    h0 = base_hash(mod)
    opt = torch.optim.AdamW(mod.groups()["new"], lr=LR_MAIN, weight_decay=WD)
    scaler = k23t.counting_scaler(torch.cuda.amp.GradScaler)()
    mod.train()
    for p in mod.groups()["new"]:
        p.requires_grad_(True)
    rows = np.asarray(D["train_rows"])[::max(1, len(D["train_rows"]) //
                                             mb)][:mb]
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        batch, po, st, tgt, val = d21.make_batch(D, rows, bar, dev)
        t_batch = time.time() - t0
        seq = S["state_seq8"][S["pos"][rows]]
        ok = S["state_valid"][S["pos"][rows]]
        st8 = torch.as_tensor(np.asarray(D["state8"])[rows],
                              dtype=torch.float32, device=dev)
        attempts, applied, n0 = [], False, new_hash(mod)
        for _k in range(STEP_TRIES):
            opt.zero_grad(set_to_none=True)
            steps0 = scaler._k23["steps"]
            scale0 = float(scaler.get_scale())
            t1 = time.time()
            pl = mod.plan(batch, po, st, st8, autocast=ac)
            L, info = full_loss(torch, mod, pl, seq, ok, tgt, val, False)
            scaler.scale(L).backward()
            frozen_grad = [n for n, p in mod.named_parameters()
                           if not n.startswith(k28m.NEW_PREFIXES) and
                           p.grad is not None]
            scaler.unscale_(opt)
            gn = grad_norms(mod, 1.0)            # после unscale_, до clipping
            g_fin = all(math.isfinite(v) for v in gn.values())
            torch.nn.utils.clip_grad_norm_(mod.groups()["new"], CLIP)
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize(dev)
            applied = scaler._k23["steps"] > steps0
            attempts.append(dict(scale=scale0, applied=applied,
                                 grads_finite=g_fin, grad_norms=gn))
            if applied:
                break
        checks = dict(finite=bool(torch.isfinite(L)),
                      frozen_no_grad=not frozen_grad,
                      grads_finite=g_fin,
                      grads_nonzero=g_fin and gn["pose"] > 0 and
                      gn["ctrl"] > 0,
                      applied=applied,
                      params_changed=new_hash(mod) != n0,
                      base_unchanged=base_hash(mod) == h0,
                      params_finite=all(bool(torch.isfinite(p).all())
                                        for p in mod.groups()["new"]))
        r = dict(checks=checks, attempts=attempts, seconds=time.time() - t1,
                 seconds_with_batch=time.time() - t1 + t_batch,
                 peak_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                 loss=info, grad_norms=gn,
                 params=dict((k, sum(p.numel() for mm in v
                                     for p in mm.parameters()))
                             for k, v in mod.new_modules().items()),
                 bad=[k for k, v in checks.items() if not v])
        del batch, pl, L
    except torch.cuda.OutOfMemoryError:
        r = dict(oom=True, bad=[])
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    print(f"  micro {mb}: {json.dumps({k: r.get(k) for k in ('checks', 'peak_gb', 'seconds', 'seconds_with_batch', 'grad_norms', 'params', 'oom')}, default=float)}"
          + (f"; попыток {len(r['attempts'])}" if "attempts" in r else ""),
          flush=True)
    return r


def free_cuda(torch):
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def nominal_reference(torch, s, e, D, dev, ac, batch, po, st):
    """Исходный parents24 сида на тех же входах -> (рука, логит схвата) на
    CPU. Модель и её тензоры живут только внутри функции."""
    import k21_data as d21
    import k22_model as k22
    b2 = d21.load_bar(dev, D["prep_report"]["bar_pin"])
    b2["model"].action_expert.float()
    ref = k22.build(torch)["K22"](b2["model"], D["L"], D["U"],
                                  D["state_mean"], D["state_std"],
                                  seed=s).to(dev)
    ck = torch.load(e["source"]["ckpt"], map_location="cpu",
                    weights_only=False)
    own = dict(ref.named_parameters())
    own.update(dict(ref.named_buffers()))
    with torch.no_grad():
        for k, v in ck["weights"].items():
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    with torch.no_grad(), ac:
        o = ref(batch, po, st)
    out = (o["arm"].float().cpu(), o["grip_logit"].float().cpu())
    del o, own, ref, b2, ck
    return out


def source_check(torch, s, D, S, dev, ac, C, rows8):
    """Проверки источника seed s на настоящих строках -> (результаты,
    проблемы). Модели освобождаются при выходе."""
    import k21_data as d21
    prob = []
    e = seed_entry(s)
    mod, bar = load_model(torch, s, D, S, dev, e)
    mod.eval()
    batch, po, st, tgt, val = d21.make_batch(D, rows8, bar, dev)
    seq = S["state_seq8"][S["pos"][rows8]]
    ok = S["state_valid"][S["pos"][rows8]]
    st8 = torch.as_tensor(np.asarray(D["state8"])[rows8],
                          dtype=torch.float32, device=dev)
    ref_arm, ref_grip = nominal_reference(torch, s, e, D, dev, ac, batch, po,
                                          st)
    free_cuda(torch)
    t0 = time.time()
    with torch.no_grad():
        pl = mod.plan(batch, po, st, st8, autocast=ac)
        torch.cuda.synchronize(dev)
        t_plan = time.time() - t0
        ca = mod.control_all(pl, seq)
        pl2 = mod.plan(batch, po, st, st8, autocast=ac)
    r = dict(base_hash=base_hash(mod), source=e["source"]["weights_hash"],
             new_params=k28m.new_param_count(mod),
             new_share=k28m.new_param_count(mod) / 272344970,
             nominal_vs_parents24=max(
                 float((pl["a0"].cpu() - ref_arm).abs().max()),
                 float((pl["grip_logit"].cpu() - ref_grip).abs().max())),
             zero_last_layer_vs_nominal=float(
                 (ca["a"] - pl["a0"]).abs().max()),
             plan_repeatable=bool(torch.equal(pl2["ref_p"], pl["ref_p"]) and
                                  torch.equal(pl2["c"], pl["c"])),
             counters=mod.counters, plan_seconds_b8=t_plan,
             new_init_hash=new_hash(mod))
    if r["base_hash"] != r["source"]:
        prob.append(f"seed {s}: база ≠ источнику")
    if r["nominal_vs_parents24"] != 0 or \
            r["zero_last_layer_vs_nominal"] != 0:
        prob.append(f"seed {s}: номинал/нулевой слой ≠ parents24 "
                    f"({r['nominal_vs_parents24']:.2e}, "
                    f"{r['zero_last_layer_vs_nominal']:.2e})")
    if not r["plan_repeatable"]:
        prob.append(f"seed {s}: повторный план на тех же входах иной")
    c = mod.counters
    if c["vision"] != 1 or any(x != 1 for x in c["text"]) or \
            c["active"] != [4] * 8 + [12] * 8 + [28] * 8:
        prob.append(f"seed {s}: счётчики {c}")
    if r["new_share"] >= 0.01:
        prob.append(f"seed {s}: новых параметров {r['new_share']:.2%}")
    # CPU-контроллер = GPU на тех же планах и свежих состояниях
    with torch.no_grad():
        m_ = mod.ctrl.l3.weight
        m_.normal_(0, 0.5)
        ca_g = mod.control_all(pl, seq)
        cpu = mod.to_cpu_plan(pl)
        cpu["plan_id"] = f"rc{s}"
        rt = C["Runtime"](mod.ctrl.state_dict())
        rt.begin(cpu)
        dmax = 0.0
        for j in range(8):
            okj = ok[:, j]
            a_c, _d = rt.step(cpu, seq[:, j], j, okj)
            d_ = np.abs(a_c[okj, :6] - ca_g["a"][:, j][torch.as_tensor(
                okj, device=dev)].cpu().numpy())
            dmax = max(dmax, float(d_.max()) if d_.size else 0.0)
        m_.zero_()
    r["cpu_vs_gpu_controller"] = dmax
    if not dmax <= 1e-4:
        prob.append(f"seed {s}: CPU ≠ GPU контроллер ({dmax:.2e})")
    print(f"  seed {s}: {json.dumps({k: v for k, v in r.items() if k != 'counters'}, default=float)}",
          flush=True)
    return r, prob


def real_check(a):
    import torch
    D, S = load_data()
    dev = torch.device(a.device)
    C = kc.build(torch)
    prob, rep = [], dict(micros=sorted(a.micros))
    if a.dtype != "float16":
        prob.append("real-check только для float16 (как исходная политика)")
    rows8 = np.asarray(D["train_rows"])[::max(1, len(D["train_rows"]) // 8)][:8]
    ac = torch.autocast("cuda", dtype=torch.float16)
    for s in SEEDS:
        r, p_ = source_check(torch, s, D, S, dev, ac, C, rows8)
        rep[f"seed_{s}"] = r
        prob += p_
        free_cuda(torch)
    # путь обучения seed 0: micro 8 и 32, полный шаг AdamW
    mod, bar = load_model(torch, 0, D, S, dev, seed_entry(0))
    for mb in sorted(a.micros):
        r = train_step_check(torch, mod, bar, D, S, dev, ac, mb)
        if r.get("bad"):
            prob.append(f"micro {mb}: {r['bad']}")
        rep[f"micro_{mb}"] = r
    ok_m = [m for m in sorted(a.micros)
            if (rep.get(f"micro_{m}") or {}).get("checks")
            and not rep[f"micro_{m}"]["bad"]]
    if 8 not in ok_m:
        prob.append("micro 8 не прошёл")
    rep["micro_choice"] = max(ok_m) if ok_m else None
    del mod, bar
    free_cuda(torch)
    # seed 1: источник сверен выше; forward/backward на выбранном micro
    if rep["micro_choice"]:
        mod, bar = load_model(torch, 1, D, S, dev, seed_entry(1))
        r = train_step_check(torch, mod, bar, D, S, dev, ac,
                             rep["micro_choice"])
        rep["seed1_step"] = r
        if r.get("oom") or r.get("bad"):
            prob.append(f"seed 1, micro {rep['micro_choice']}: "
                        f"{'OOM' if r.get('oom') else r['bad']}")
        del mod, bar
        free_cuda(torch)
    rep["contract"] = check_contract("real-check", a)
    old = read_micro()
    if old is not None and old.get("micro") != rep["micro_choice"]:
        prob.append(f"решение micro уже записано: {old.get('micro')} ≠ "
                    f"{rep['micro_choice']} — micro не меняется после "
                    f"решения")
    rep["problems"] = prob          # все проверки — до записи
    os.makedirs("reports/k28", exist_ok=True)
    if not prob:
        write_json(MICRO_FILE, dict(micro=rep["micro_choice"],
                                    accumulation=EFF // rep["micro_choice"],
                                    contract=rep["contract"]))
    write_json(REAL_CHECK, rep)
    print("  " + ("ОТКАЗ: " + "; ".join(prob) if prob else
                  f"real-check K-28 пройден; micro {rep['micro_choice']}") +
          f"; {REAL_CHECK}", flush=True)
    return 3 if prob else 0


def selftest():
    import copy
    import tempfile
    import types
    import torch
    import k21_data as d21
    import k21_model as km
    import k21_train as kt
    M = k28m.build(torch)
    bar0 = km.fake_bar(torch)
    L, U = -np.ones(6) * 1.1, np.ones(6) * 4.0
    N = 64
    rng = np.random.default_rng(0)
    D = dict(train_rows=np.arange(48), val_rows=np.arange(48, 56),
             task=np.arange(N) % 4, episode=np.arange(N) % 8, L=L, U=U,
             state_mean=None, state_std=None,
             state8=np.concatenate([rng.normal(0, .1, (N, 3)),
                                    rng.normal(0, .3, (N, 3)),
                                    np.zeros((N, 2))], 1),
             prep_report=dict(prep_id="P", bar_pin=None))
    seq = np.repeat(D["state8"][:, None], 21, 1).astype(np.float32)
    seq[:, :, :3] += np.cumsum(rng.normal(0, 0.005, (N, 21, 3)), 1)
    okm = np.ones((N, 21), bool)
    okm[::5, 18:] = False
    S = dict(state_seq8=seq, state_valid=okm, pos=np.arange(N),
             sigma_p=np.full(3, 0.01, np.float32), sigma_R=0.02,
             report=dict(k28_prep_id="K"))

    def fake_mb(D_, rows, bar, dev, teacher_idx=None):
        rows = np.asarray(rows)
        b = km.fake_batch(torch, B=len(rows), seed=int(rows.sum()) % 997)
        g = torch.Generator().manual_seed(int(rows.sum()) + 7)
        st_ = torch.randn(len(rows), 11, generator=g)
        tgt = torch.rand(len(rows), 20, 7, generator=g)
        return b, 1, st_, tgt, torch.ones(len(rows), 20, dtype=torch.bool)
    base = M["K22"](copy.deepcopy(bar0), L, U, seed=1)
    with torch.no_grad():
        for p in base.parameters():
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=torch.Generator()
                                   .manual_seed(p.numel())) * 0.02)
    sb = kt.trainable_state(base)

    def fake_make(torch_, seed, D_, S_, dev, entry, checkpoint=False):
        m = M["K28"](copy.deepcopy(bar0), L, U, seed=seed, sigma_p=S_[
            "sigma_p"], sigma_R=S_["sigma_R"])
        own = dict(m.named_parameters())
        own.update(dict(m.named_buffers()))
        with torch.no_grad():
            for k, v in sb.items():
                own[k].copy_(v)
        return m, dict(model=m)
    g = globals()
    saved = (d21.make_batch, g["WARM"], g["MAIN"], g["EVAL_EVERY"],
             g["SAVE_EVERY"], g["run_contract"], g["EFF"])
    a_ = types.SimpleNamespace(device="cpu", dtype="float32", micro=4,
                               eval_micro=4)
    try:
        d21.make_batch = fake_mb
        g["WARM"], g["MAIN"], g["EVAL_EVERY"], g["SAVE_EVERY"], g["EFF"] = \
            2, 4, 2, 1, 8
        g["run_contract"] = lambda seed, a__, e_, S_: dict(seed=seed)
        res = {}
        with tempfile.TemporaryDirectory() as td:
            for name, stops in (("cont", None), ("resume", 3)):
                odir = os.path.join(td, name)
                if stops is not None:
                    # прерывание после 3 попыток (разогрев 2 + основная 1)
                    T_ = Trainer(torch, a_, 1, D, S, {}, odir=odir,
                                 make=fake_make, fastload=False)
                    orig_step = T_.step
                    calls = dict(n=0)

                    def step(*x, _o=orig_step, _c=calls):
                        _c["n"] += 1
                        if _c["n"] > stops:
                            raise KeyboardInterrupt
                        return _o(*x)
                    T_.step = step
                    try:
                        T_.run()
                    except KeyboardInterrupt:
                        pass
                T = Trainer(torch, a_, 1, D, S, {}, odir=odir,
                            make=fake_make, fastload=False)
                T.run()
                res[name] = (new_hash(T.mod), base_hash(T.mod),
                             [(c["tag"], round(c["val"]["l1_8"], 9))
                              for c in T.st["curve"]], T.st["best"])
            assert res["cont"][0] == res["resume"][0], res
            assert res["cont"][1] == kt.state_hash(sb)
            assert res["cont"][3][0] in (2, 4)          # только основная
            assert res["cont"][0] != T.init_new
    finally:
        (d21.make_batch, g["WARM"], g["MAIN"], g["EVAL_EVERY"],
         g["SAVE_EVERY"], g["run_contract"], g["EFF"]) = saved
    # очередь: от seed, повторяема, не зависит от micro, веса задач равны
    s0 = make_schedule(D, 0, 8)
    q = s0(D, 4, 8, start=3)[0][1]
    assert np.array_equal(q.reshape(-1), make_schedule(D, 0, 4)(
        D, 4, 4, start=3)[0][1].reshape(-1))
    assert not np.array_equal(q, make_schedule(D, 1, 8)(D, 4, 8, start=3)
                              [0][1])
    p = row_probs(D, D["train_rows"])
    tk = D["task"][D["train_rows"]]
    assert np.allclose([p[tk == t].sum() for t in range(4)], 0.25)
    assert lr_main(0) < lr_main(99) <= LR_MAIN and \
        abs(lr_main(2000) - LR_MAIN * FLOOR) < 1e-12
    # потери: пустой уровень не даёт NaN
    m = fake_make(torch, 0, D, S, None, None)[0]
    m.train()
    b, po, st, tgt, val = fake_mb(D, np.arange(3), None, None)
    pl = m.plan(b, po, st, torch.as_tensor(D["state8"][:3], dtype=torch.float32))
    ok0 = np.zeros((3, 21), bool)
    Lp, info = pose_losses(torch, m, pl, seq[:3], ok0)
    assert torch.isfinite(Lp) and set(info["empty_levels"]) == {
        "root", "mid", "leaf"}
    Lf, inf2 = full_loss(torch, m, pl, seq[:3], okm[:3], tgt, val, False)
    assert torch.isfinite(Lf) and inf2["L_corr"] == 0.0
    # настоящая фабрика (load_model) с игрушечным BAR и источником
    saved_lb = d21.load_bar
    import k21_protocol as pp
    try:
        d21.load_bar = lambda dev, pin: dict(model=copy.deepcopy(bar0))
        with tempfile.TemporaryDirectory() as td:
            ckp = os.path.join(td, "best.pt")
            for s_ in SEEDS:
                torch.save(dict(weights=sb, weights_hash=kt.state_hash(sb),
                                config=dict(mode="parents24", seed=s_,
                                            prep_id="P")), ckp)
                en = dict(source=dict(ckpt=ckp, ckpt_sha1=pp.full_sha(ckp),
                                      weights_hash=kt.state_hash(sb)))
                mf, _b = load_model(torch, s_, D, S, torch.device("cpu"), en)
                assert type(mf).__name__ == "K28" and mf.fb_mode == "fb"
                assert base_hash(mf) == kt.state_hash(sb)
                try:
                    load_model(torch, 1 - s_, D, S, torch.device("cpu"), en)
                    raise AssertionError("источник чужого seed принят")
                except SystemExit:
                    pass
    finally:
        d21.load_bar = saved_lb
    # решение micro согласовано с отчётом real-check и контрактом
    good = dict(problems=[], micro_choice=8, contract=dict(c=1))
    assert micro_problems(good, dict(micro=8, contract=dict(c=1)),
                          dict(c=1)) == []
    for rep_, m_, w_ in ((good, dict(micro=32, contract=dict(c=1)), None),
                         (good, dict(micro=8, contract=dict(c=2)), None),
                         (good, dict(micro=8, contract=dict(c=1)), dict(c=2)),
                         (dict(good, problems=["x"]),
                          dict(micro=8, contract=dict(c=1)), None),
                         (good, None, None), (None, None, None)):
        assert micro_problems(rep_, m_, w_), (rep_, m_, w_)
    rows = fit_rows(dict(D, train_rows=np.arange(48)), n=16)
    assert len(set(D["task"][rows].tolist())) == 4 and len(rows) == 16
    print("самопроверка k28_train пройдена: непрерывное = прерванное + "
          "возобновление (новые веса, кривая, best только среди основной), "
          "база неизменна, очередь (seed, повторяемость, micro, веса задач), "
          "LR, потери без NaN при пустых уровнях, настоящая фабрика "
          "(K28, отказ при источнике чужого seed), строки fitcheck по задачам, "
          "решение micro ↔ отчёт real-check ↔ контракт")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-28: обучение и проверки")
    ap.add_argument("cmd", choices=("real-check", "fitcheck", "train",
                                    "check-fresh", "micro"))
    ap.add_argument("--kind", choices=("real-check", "fitcheck"),
                    default="real-check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--eval-micro", type=int, default=8)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--micros", type=int, nargs="+", default=[8, 32])
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
