#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-29: PPO условного декодера parents_cond24 (arm-only) — обновление,
тренер с бюджетом, валидацией и возобновлением.

Обновление на свежем batch (1–2 полных цикла):
  V_old = sigmoid(critic(F)) до обновлений; A = R − V_old, нормировка по
  всему batch (std ≤ 1e-8 -> actor пропускается, причина в журнале);
  μ_old — пересчёт текущим actor (= policy_old) по кешу рабочими minibatch;
  сверка с μ, записанным при сборе: max|Δμ| ≤ 1e−5, max|Δ log π| ≤ 1e−3,
  иначе технический отказ;
  до 4 эпох, minibatch 32 (последний меньше), clip 0.2, grad clip 1.0
  (actor и critic отдельно), actor: −mean(min(r·A, clip(r)·A)) +
  0.01·mean KL(θ‖reference) (сумма по окну, без деления на длину эпизода);
  critic: BCEWithLogits к R; перед эпохой — снимок actor и его optimizer;
  после — точный средний KL(θ‖old) на всём batch; > 0.03 -> откат actor и
  optimizer к началу эпохи, дальнейшие эпохи на batch не выполняются.
Тренер: циклы 0 -> 6 -> 8 (волна = 5 эпизодов, init без повторов из 0…19),
batch = 2 цикла (последний может быть 1); кап 200 000 физических шагов на
сид, новый цикл — только если 9150 помещается; резерв цикла пишется в
журнал до начала. Валидация (детерминированный actor, задачи 0/6/8, init
20…24): в начале, после первого принятого обновления за 100 000 и в конце —
только описательно. Итог — последний принятый actor.
RNG: init ID — default_rng([seed, 1, волна]); шум — [seed, 2, волна];
перемешивание — [seed, 3, обновление, эпоха]; критик — torch seed 29000+seed.
Сохранение на границах обновления (actor, critic, оба optimizer, счётчики,
хеши); полные циклы ждут обновления в pending/ с отпечатком policy_old.
"""
import argparse
import copy
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k29_collect as kc  # noqa: E402
import k29_dist as kd  # noqa: E402

SEEDS = (0, 1)
CAP = 200_000
MID_MARK = 100_000
TRAIN_INIT = tuple(range(0, 20))
VAL_INIT = tuple(range(20, 25))
CFG = dict(lr_actor=1e-5, wd_actor=0.0, lr_critic=3e-4, epochs=4, minibatch=32,
           clip=0.2, grad_clip=1.0, lambda_ref=0.01, kl_old_max=0.03,
           mu_tol=1e-5, logp_tol=1e-3, adv_std_min=1e-8, cycles_per_batch=2,
           cap=CAP, cycle_max=kc.CYCLE_MAX, tasks=list(kc.CYCLE_TASKS),
           train_init=list(TRAIN_INIT), val_init=list(VAL_INIT),
           mid_mark=MID_MARK, dist=kd.CONFIG)


def run_dir(seed):
    return f"data/k29/runs/s{seed}"


def report_path(seed):
    return f"reports/k29/train_s{seed}.json"


def to_dev(F, idx, dev):
    return {k: v[idx].to(dev) for k, v in F.items()}


# --- одно обновление PPO -----------------------------------------------------------
def mu_all(torch, actor, M, buf, dev, mb):
    out = []
    N = len(buf["k"])
    with torch.no_grad():
        for s in range(0, N, mb):
            idx = np.arange(s, min(N, s + mb))
            out.append(M["mu_of"](actor(to_dev(buf["F"], idx, dev))).float()
                       .cpu())
    return torch.cat(out)


def ppo_update(torch, M, actor, critic, opt_a, opt_c, buf, dev, rng_seed,
               cfg=CFG, check=True):
    """-> статистика обновления (dict). buf — concat циклов (CPU)."""
    N = len(buf["k"])
    k = torch.as_tensor(buf["k"])
    Y, mu_ref = buf["Y"].float(), buf["mu_ref"].float()
    R = torch.as_tensor(buf["R"], dtype=torch.float32)
    el = torch.as_tensor(buf["elapsed"], dtype=torch.float32)
    mb = int(cfg["minibatch"])
    st = dict(n=N, episodes=int(len(set(buf["episode"].tolist()))),
              R_mean=float(R.mean()), epochs=0, rollback=None,
              actor_skipped=None)
    # V_old и μ_old до всех обновлений
    critic.eval()
    with torch.no_grad():
        V = torch.cat([torch.sigmoid(critic(
            to_dev(buf["F"], np.arange(s, min(N, s + mb)), dev),
            buf["s11n"][s:s + mb].to(dev), el[s:s + mb].to(dev))).cpu()
            for s in range(0, N, mb)])
    mu_old = mu_all(torch, actor, M, buf, dev, mb)
    lp_old = kd.log_prob(torch, Y, mu_old, k)
    lp_rec = kd.log_prob(torch, Y, buf["mu"].float(), k)
    st["mu_recompute_max"] = float((mu_old - buf["mu"].float()).abs().max())
    st["logp_recompute_max"] = float((lp_old - lp_rec).abs().max())
    if check and not (st["mu_recompute_max"] <= cfg["mu_tol"] and
                      st["logp_recompute_max"] <= cfg["logp_tol"]):
        raise SystemExit(f"пересчёт μ/log π по кешу ≠ записанному при сборе: "
                         f"{st['mu_recompute_max']:.2e} / "
                         f"{st['logp_recompute_max']:.2e}")
    A = R - V
    st["V_mean"], st["adv_std_raw"] = float(V.mean()), float(A.std())
    actor_on = bool(A.std() > cfg["adv_std_min"])
    if actor_on:
        A = (A - A.mean()) / A.std()
    else:
        st["actor_skipped"] = "std(A) ≤ 1e-8: actor не обновляется"
    kl_ref0 = kd.kl(torch, mu_old, mu_ref, k)
    st["kl_ref_before"] = float(kl_ref0.mean())
    bce = torch.nn.BCEWithLogitsLoss()
    hist = []
    for ep in range(int(cfg["epochs"])):
        snap_a = copy.deepcopy(actor.state_dict())
        snap_o = copy.deepcopy(opt_a.state_dict())
        perm = np.random.default_rng([*rng_seed, ep]).permutation(N)
        actor.train()
        critic.train()
        la_s = lc_s = 0.0
        nb = 0
        for s in range(0, N, mb):
            idx = perm[s:s + mb]
            Fm = to_dev(buf["F"], idx, dev)
            if actor_on:
                mu = M["mu_of"](actor(Fm)).float()
                lp = kd.log_prob(torch, Y[idx].to(dev), mu, k[idx].to(dev))
                ratio = torch.exp(lp - lp_old[idx].to(dev))
                Am = A[idx].to(dev)
                surr = torch.min(ratio * Am, torch.clamp(
                    ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * Am)
                klr = kd.kl(torch, mu, mu_ref[idx].to(dev), k[idx].to(dev))
                la = -surr.mean() + cfg["lambda_ref"] * klr.mean()
                if not bool(torch.isfinite(la)):
                    raise SystemExit("нечисловая потеря actor")
                opt_a.zero_grad(set_to_none=True)
                la.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(),
                                               cfg["grad_clip"])
                opt_a.step()
                la_s += float(la.detach())
            v = critic(Fm, buf["s11n"][idx].to(dev), el[idx].to(dev))
            lc = bce(v, R[idx].to(dev))
            if not bool(torch.isfinite(lc)):
                raise SystemExit("нечисловая потеря critic")
            opt_c.zero_grad(set_to_none=True)
            lc.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(),
                                           cfg["grad_clip"])
            opt_c.step()
            lc_s += float(lc.detach())
            nb += 1
        mu_new = mu_all(torch, actor, M, buf, dev, mb)
        kl_old = float(kd.kl(torch, mu_new, mu_old, k).mean())
        rec = dict(epoch=ep, loss_actor=la_s / nb, loss_critic=lc_s / nb,
                   kl_old=kl_old,
                   kl_ref=float(kd.kl(torch, mu_new, mu_ref, k).mean()))
        if actor_on and kl_old > cfg["kl_old_max"]:
            actor.load_state_dict(snap_a)
            opt_a.load_state_dict(snap_o)
            rec["rolled_back"] = True
            st["rollback"] = dict(epoch=ep, kl_old=kl_old)
            hist.append(rec)
            break
        hist.append(rec)
        st["epochs"] = ep + 1
    for p in list(actor.parameters()) + list(critic.parameters()):
        if not bool(torch.isfinite(p).all()):
            raise SystemExit("нечисловые веса после обновления")
    st["history"] = hist
    st["accepted"] = bool(actor_on and st["epochs"] > 0)
    actor.eval()
    critic.eval()
    return st


# --- тренер ----------------------------------------------------------------------
def pick_init(seed, wave_id, bank=TRAIN_INIT, n=5):
    g = np.random.default_rng([seed, 1, wave_id])
    return sorted(int(x) for x in g.choice(np.asarray(bank), n, replace=False))


class Trainer:
    def __init__(self, torch, seed, M, mod, actor, reference, critic, sim,
                 inputs, policy_cls, dev, contract, odir=None, cap=CAP,
                 cfg=CFG, log=print):
        if seed not in SEEDS:
            raise SystemExit(f"training seed {seed}: только {SEEDS}")
        self.torch, self.seed, self.M = torch, seed, M
        self.mod, self.actor, self.ref, self.critic = mod, actor, reference, \
            critic
        self.sim, self.inputs, self.policy_cls = sim, inputs, policy_cls
        self.dev, self.contract, self.cfg, self.log = dev, contract, cfg, log
        self.dir = odir or run_dir(seed)
        os.makedirs(os.path.join(self.dir, "pending"), exist_ok=True)
        self.ledger = kc.Ledger(os.path.join(self.dir, "ledger.json"), cap)
        self.opt_a = torch.optim.AdamW(actor.parameters(), lr=cfg["lr_actor"],
                                       weight_decay=cfg["wd_actor"])
        self.opt_c = torch.optim.Adam(critic.parameters(), lr=cfg["lr_critic"])
        self.state_path = os.path.join(self.dir, "state.pt")
        self.st = dict(wave=0, cycle=0, update=0, accepted=0, val={},
                       updates=[], episodes=0, transitions=0, phys_train=0,
                       policy_steps=0, finished=False)
        self.ref_hash = self.M_hash(reference)
        self.src_hash = None

    def M_hash(self, m):
        import k29_model as km
        return km.tensor_hash(m.state_dict())

    def actor_hash(self):
        return self.M_hash(self.actor)

    def _crash(self, point):
        """Точка аварии для самопроверок (в работе не срабатывает)."""
        if getattr(self, "crash_at", None) == point:
            raise KeyboardInterrupt(point)

    # --- сохранение ----------------------------------------------------------------
    def save(self):
        torch = self.torch
        self.check_frozen()
        d = dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                 opt_a=self.opt_a.state_dict(), opt_c=self.opt_c.state_dict(),
                 st=self.st, contract=self.contract, ref_hash=self.ref_hash,
                 actor_hash=self.actor_hash())
        torch.save(d, self.state_path + ".tmp")
        os.replace(self.state_path + ".tmp", self.state_path)

    def load(self):
        torch = self.torch
        if not os.path.exists(self.state_path):
            return False
        d = torch.load(self.state_path, map_location="cpu", weights_only=False)
        if d["contract"] != self.contract:
            raise SystemExit(f"{self.state_path}: другой контракт — нужен "
                             f"новый каталог")
        if d["ref_hash"] != self.ref_hash:
            raise SystemExit("reference ≠ сохранённому")
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.opt_a.load_state_dict(d["opt_a"])
        self.opt_c.load_state_dict(d["opt_c"])
        self.st = d["st"]
        if self.actor_hash() != d["actor_hash"]:
            raise SystemExit("actor после загрузки ≠ сохранённому")
        self.log(f"  продолжение: обновление {self.st['update']}, цикл "
                 f"{self.st['cycle']}, волна {self.st['wave']}; бюджет "
                 f"{self.ledger.d['train_spent']}/{self.ledger.cap}")
        return True

    def check_frozen(self):
        if self.M_hash(self.ref) != self.ref_hash:
            raise SystemExit("reference изменился")
        if self.src_hash is not None:
            import k29_model as km
            if km.frozen_hash(self.mod) != self.src_hash:
                raise SystemExit("замороженный источник изменился")

    # --- сбор -------------------------------------------------------------------------
    def pending(self):
        out = []
        used = {c for u in self.st["updates"] for c in u["cycles"]}
        for f in sorted(glob.glob(os.path.join(self.dir, "pending",
                                               "cycle_*.pt"))):
            d = self.torch.load(f, map_location="cpu", weights_only=False)
            if d["cycle"] in used:            # авария после сохранения
                os.remove(f)                  # обновления, до удаления
                continue
            if d["policy_old"] != self.actor_hash():
                raise SystemExit(f"{f}: собран другим policy_old")
            if d["cycle"] >= self.st["cycle"]:    # авария до сохранения
                self.st["cycle"] = d["cycle"] + 1  # счётчиков цикла: всё —
                self.st["wave"] = max(self.st["wave"], d["wave_end"])
                for k_ in ("episodes", "transitions", "policy_steps"):
                    self.st[k_] += int(d["counts"][k_])   # из данных цикла
                self.st["phys_train"] = self.ledger.d["train_spent"]
                self.save()                       # одно начисление
            out.append((f, d))
        return out

    def collect_cycle(self):
        torch = self.torch
        cid = self.st["cycle"]
        self.ledger.reserve(f"cycle {cid}")
        pol = self.policy_cls(torch, self.mod, self.actor, self.ref, self.M,
                              "actor")
        bufs, eps_all, phys = [], [], 0
        cnt = dict(policy_steps=0, transitions=0, episodes=0)
        t0 = time.time()
        for task in self.cfg["tasks"]:
            w = self.st["wave"]
            starts = pick_init(self.seed, w)
            eps, buf, _log, c = kc.run_wave(
                self.sim, self.inputs, pol, task, starts,
                noise_gen=np.random.default_rng([self.seed, 2, w]))
            self.st["wave"] = w + 1
            phys += c["physical"]
            for k_ in cnt:
                cnt[k_] += c[k_]
            bufs.append(buf)
            eps_all += [dict(e, wave=w) for e in eps]
        if phys > self.cfg["cycle_max"]:
            raise SystemExit(f"цикл {phys} > {self.cfg['cycle_max']}")
        d = dict(policy_old=self.actor_hash(), cycle=cid, bufs=bufs,
                 episodes=eps_all, phys=phys, counts=cnt,
                 wave_end=self.st["wave"])
        p = os.path.join(self.dir, "pending", f"cycle_{cid:04d}.pt")
        torch.save(d, p + ".tmp")
        os.replace(p + ".tmp", p)
        self._crash("after_pending")
        self.ledger.commit(phys, dict(cycle=cid, success=float(np.mean(
            [e["success"] for e in eps_all])), **cnt))
        self._crash("after_commit")
        self.st["cycle"] = cid + 1
        self.st["episodes"] += cnt["episodes"]
        self.st["transitions"] += cnt["transitions"]
        self.st["policy_steps"] += cnt["policy_steps"]
        self.st["phys_train"] = self.ledger.d["train_spent"]
        sr = np.mean([e["success"] for e in eps_all])
        self.log(f"  цикл {cid}: {phys} шагов ({time.time() - t0:.0f} с), "
                 f"успех стохастической политики {sr:.2f}, переходов "
                 f"{cnt['transitions']}; бюджет {self.ledger.d['train_spent']}"
                 f"/{self.ledger.cap}")
        self.save()                       # счётчики волн/циклов — с pending
        return p

    def validate(self, tag):
        if tag in self.st["val"]:
            return
        pol = self.policy_cls(self.torch, self.mod, self.actor, self.ref,
                              self.M, "actor")
        res, phys = {}, 0
        for task in self.cfg["tasks"]:
            eps, _b, _l, c = kc.run_wave(self.sim, self.inputs, pol, task,
                                         list(self.cfg["val_init"]),
                                         record=False)
            phys += c["physical"]
            res[str(task)] = [bool(e["success"]) for e in eps]
        self.ledger.add_other("validation", phys)
        sr = float(np.mean([np.mean(v) for v in res.values()]))
        self.st["val"][tag] = dict(sr=sr, per_task=res, phys=phys,
                                   at_train_steps=self.ledger.d["train_spent"],
                                   update=self.st["update"],
                                   actor_hash=self.actor_hash())
        self.log(f"  валидация {tag} (описательно, 20–24): SR {sr:.3f} "
                 f"{ {k: sum(v) for k, v in res.items()} }")
        self.save()

    def update(self, items):
        torch = self.torch
        bufs = [b for _f, d in items for b in d["bufs"]]
        buf = kc.concat(bufs)
        u = self.st["update"]
        t0 = time.time()
        st = ppo_update(torch, self.M, self.actor, self.critic, self.opt_a,
                        self.opt_c, buf, self.dev, (self.seed, 3, u), self.cfg)
        st.update(update=u, cycles=[d["cycle"] for _f, d in items],
                  seconds=time.time() - t0,
                  train_steps=self.ledger.d["train_spent"])
        self.st["update"] = u + 1
        self.st["accepted"] += int(st["accepted"])
        self.st["updates"].append({k: v for k, v in st.items()})
        self.save()
        for f, _d in items:
            os.remove(f)
        self.log(f"  обновление {u}: переходов {st['n']}, R {st['R_mean']:.2f},"
                 f" эпох {st['epochs']}" + (f", ОТКАТ на эпохе "
                                           f"{st['rollback']['epoch']} (KL "
                                           f"{st['rollback']['kl_old']:.3f})"
                                           if st["rollback"] else "")
                 + (f", {st['actor_skipped']}" if st["actor_skipped"] else "")
                 + f"; KL old/ref {st['history'][-1]['kl_old']:.4f}/"
                 f"{st['history'][-1]['kl_ref']:.4f}; пересчёт μ "
                 f"{st['mu_recompute_max']:.1e} ({st['seconds']:.0f} с)")
        return st

    def run(self):
        self.load()
        self.check_frozen()
        self.validate("start")
        while True:
            pend = self.pending()
            last = not self.ledger.can_cycle()
            if len(pend) >= self.cfg["cycles_per_batch"] or (last and pend):
                su = self.update(pend)
                # первое ПРИНЯТОЕ обновление, сделанное за отметкой 100 000
                if su["accepted"] and "mid" not in self.st["val"] and \
                        su["train_steps"] >= self.cfg["mid_mark"]:
                    self.validate("mid")
                continue
            if last:
                break
            self.collect_cycle()
        self.validate("end")
        self.st["finished"] = True
        self.save()
        return self.st


def selftest():
    import tempfile
    import torch
    import k21_model as km
    import k25_model as k25m
    import k29_model as k29m
    import k29_policy as kp
    k29m.no_tf32(torch)
    M = k29m.build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])

    torch.manual_seed(0)
    base = k25m.build(torch)["K25"](km.fake_bar(torch), L, U, seed=0).eval()

    def make(seed=0):
        mod = copy.deepcopy(base)              # один ствол, как чекпойнт
        for p in mod.parameters():
            p.requires_grad_(False)
        actor = M["Decoder"](mod)
        ref = M["Decoder"](mod).freeze()
        critic = M["Critic"](mod.H, torch.Generator().manual_seed(29000 + seed))
        return mod, actor, ref, critic

    class ToyInputs(kc.FakeInputs):
        def __call__(self, obs, desc, task):
            x = obs["state"][:, 0]
            b = km.fake_batch(torch, B=len(x), seed=int(1000 * x.sum()) % 97)
            return b, 1, np.concatenate(
                [np.asarray(obs["state"][:, :3], np.float64),
                 np.zeros((len(x), 5))], 1)

    class ToyPolicy(kp.Policy):
        def chunk(self, batch, po, state8, noise=None):
            out = super().chunk(batch, po, state8, noise)
            # игрушка: «рука» по x — подставляем знак, чтобы x рос
            out["a_codec"][..., 0] = np.abs(out["a_codec"][..., 0]) + 0.3
            return out

    # --- обновление: ratio = 1 в начале, откат по KL, пропуск при std = 0
    mod, actor, ref, critic = make()
    sim = kc.FakeSim(thr=0.25, flicker=False)
    pol = ToyPolicy(torch, mod, actor, ref, M, "actor")
    bufs = []
    for w, st_ in enumerate(([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])):
        _e, b, _l, _c = kc.run_wave(sim, ToyInputs(), pol, 0, st_,
                                    noise_gen=np.random.default_rng(w))
        bufs.append(b)
    buf = kc.concat(bufs)
    buf["R"] = (np.arange(len(buf["k"])) % 2).astype(np.float32)  # разброс
    opt_a = torch.optim.AdamW(actor.parameters(), lr=1e-5, weight_decay=0)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-4)
    h_ref = k29m.tensor_hash(ref.state_dict())
    actor0, critic0 = copy.deepcopy(actor), copy.deepcopy(critic)  # policy_old
    st = ppo_update(torch, M, actor, critic, opt_a, opt_c, buf, "cpu",
                    (0, 3, 0))
    # буфер старой политики после обновления не принимается (пересчёт μ)
    try:
        ppo_update(torch, M, actor, critic, opt_a, opt_c, buf, "cpu",
                   (0, 3, 0))
        raise AssertionError("буфер чужой политики принят")
    except SystemExit:
        pass
    actor, critic = actor0, critic0
    assert st["mu_recompute_max"] <= 1e-5 and st["epochs"] >= 1
    assert k29m.tensor_hash(ref.state_dict()) == h_ref
    # откат: огромный LR -> KL > порога -> веса и optimizer как до эпохи
    a2 = copy.deepcopy(actor)
    o2 = torch.optim.AdamW(a2.parameters(), lr=1.0, weight_decay=0)
    before = k29m.tensor_hash(a2.state_dict())
    st2 = ppo_update(torch, M, a2, copy.deepcopy(critic), o2,
                     torch.optim.Adam(critic.parameters(), lr=3e-4), buf,
                     "cpu", (0, 3, 1))
    assert st2["rollback"] and st2["rollback"]["epoch"] == 0 and \
        k29m.tensor_hash(a2.state_dict()) == before and not st2["accepted"]
    assert all(len(v) == 0 for v in o2.state.values()) or not o2.state
    # std(A) = 0 -> actor пропускается, critic учится
    b3 = dict(buf, R=np.ones(len(buf["k"]), np.float32))
    a3, c3 = copy.deepcopy(actor), copy.deepcopy(critic)
    h3, hc3 = k29m.tensor_hash(a3.state_dict()), k29m.tensor_hash(
        c3.state_dict())
    oc3 = torch.optim.Adam(c3.parameters(), lr=3e-4)
    oc3_ = torch.optim.AdamW(a3.parameters(), lr=1e-5)
    # V_old = 0.5 у нулевого критика -> A = 0.5 у всех -> std 0
    c0 = make()[3]
    st3 = ppo_update(torch, M, a3, c0, oc3_, torch.optim.Adam(
        c0.parameters(), lr=3e-4), b3, "cpu", (0, 3, 2))
    assert st3["actor_skipped"] and k29m.tensor_hash(a3.state_dict()) == h3
    _ = (hc3, oc3)
    # подмена награды меняет обновление, но не μ при сборе (нет утечки)
    a4, a5 = copy.deepcopy(actor), copy.deepcopy(actor)
    c4, c5 = copy.deepcopy(critic), copy.deepcopy(critic)
    st4 = ppo_update(torch, M, a4, c4, torch.optim.AdamW(a4.parameters(), lr=1e-5),
               torch.optim.Adam(c4.parameters(), lr=3e-4), buf, "cpu",
               (0, 3, 9))
    b5 = dict(buf, R=1 - buf["R"])
    st5 = ppo_update(torch, M, a5, c5, torch.optim.AdamW(a5.parameters(), lr=1e-5),
               torch.optim.Adam(c5.parameters(), lr=3e-4), b5, "cpu",
               (0, 3, 9))
    assert st4["accepted"] and st5["accepted"]
    assert k29m.tensor_hash(a4.state_dict()) != k29m.tensor_hash(
        a5.state_dict())
    # порча записанного μ -> технический отказ пересчёта
    b6 = dict(buf, mu=buf["mu"] + 1e-3)
    try:
        ppo_update(torch, M, copy.deepcopy(actor), copy.deepcopy(critic),
                   torch.optim.AdamW(actor.parameters(), lr=1e-5),
                   torch.optim.Adam(critic.parameters(), lr=3e-4), b6, "cpu",
                   (0, 3, 3))
        raise AssertionError("пересчёт μ не поймал порчу")
    except SystemExit:
        pass

    # --- тренер: непрерывный = прерванный + возобновление; авария в цикле
    def trainer(td, cap, make_seed=0):
        mod, actor, ref, critic = make(make_seed)
        return Trainer(torch, 0, M, mod, actor, ref, critic,
                       kc.FakeSim(thr=0.25, flicker=False), ToyInputs(),
                       ToyPolicy, "cpu", dict(toy=1), odir=td, cap=cap,
                       log=lambda *a: None)
    small = dict(CFG, cycle_max=kc.CYCLE_MAX, mid_mark=5000)
    with tempfile.TemporaryDirectory() as td1, \
            tempfile.TemporaryDirectory() as td2:
        cap = 4 * kc.CYCLE_MAX
        t1 = trainer(td1, cap)
        t1.cfg = small
        s1 = t1.run()
        assert s1["finished"] and s1["update"] >= 2
        assert t1.ledger.d["train_spent"] <= cap and \
            t1.ledger.remaining() < kc.CYCLE_MAX
        assert set(s1["val"]) >= {"start", "end"}
        # прерывание после первого цикла (pending сохранён) и продолжение
        t2 = trainer(td2, cap)
        t2.cfg = small
        t2.load()
        t2.validate("start")
        t2.collect_cycle()
        t3 = trainer(td2, cap)
        t3.cfg = small
        s3 = t3.run()
        assert k29m.tensor_hash(t3.actor.state_dict()) == \
            k29m.tensor_hash(t1.actor.state_dict())
        assert s3["update"] == s1["update"] and \
            t3.ledger.d["train_spent"] == t1.ledger.d["train_spent"]
    with tempfile.TemporaryDirectory() as td4:
        # авария после сохранения обновления, до удаления pending
        t6 = trainer(td4, 4 * kc.CYCLE_MAX)
        t6.cfg = small
        t6.load()
        t6.validate("start")
        t6.collect_cycle()
        t6.collect_cycle()
        pend = t6.pending()
        import shutil
        keep = [f + ".bak" for f, _d in pend]
        for (f, _d), k_ in zip(pend, keep):
            shutil.copy(f, k_)
        t6.update(pend)
        for (f, _d), k_ in zip(pend, keep):
            shutil.move(k_, f)                # «не успели удалить»
        t7 = trainer(td4, 4 * kc.CYCLE_MAX)
        t7.cfg = small
        t7.load()
        assert t7.pending() == [] and not glob.glob(os.path.join(
            td4, "pending", "cycle_*.pt"))
    # аварии после записи pending и после ledger.commit, до state.pt:
    # счётчики восстанавливаются из данных цикла и ledger, один раз
    for point in ("after_pending", "after_commit"):
        with tempfile.TemporaryDirectory() as td5:
            t8 = trainer(td5, 4 * kc.CYCLE_MAX)
            t8.cfg = small
            t8.load()
            t8.validate("start")
            t8.crash_at = point
            try:
                t8.collect_cycle()
                raise AssertionError("авария не сработала")
            except KeyboardInterrupt:
                pass
            t9 = trainer(td5, 4 * kc.CYCLE_MAX)
            t9.cfg = small
            t9.load()
            pend = t9.pending()
            assert len(pend) == 1
            cnt = pend[0][1]["counts"]
            for _ in range(2):                    # повторный вызов — без
                t9.pending()                      # двойного начисления
                t10 = trainer(td5, 4 * kc.CYCLE_MAX)
                t10.cfg = small
                t10.load()
                t10.pending()
                for k_ in ("episodes", "transitions", "policy_steps"):
                    assert t10.st[k_] == cnt[k_] > 0, (point, k_, t10.st[k_])
                assert t10.st["cycle"] == 1 and t10.st["wave"] == 3
                assert t10.st["phys_train"] == t10.ledger.d["train_spent"]
                if point == "after_commit":
                    assert t10.ledger.d["train_spent"] == pend[0][1]["phys"]
                else:                             # резерв списан целиком
                    assert t10.ledger.d["train_spent"] == kc.CYCLE_MAX
    with tempfile.TemporaryDirectory() as td3:
        t4 = trainer(td3, 4 * kc.CYCLE_MAX)
        t4.cfg = small
        t4.ledger.reserve("cycle 0")          # авария внутри цикла
        t5 = trainer(td3, 4 * kc.CYCLE_MAX)
        assert t5.ledger.d["train_spent"] == kc.CYCLE_MAX and \
            t5.ledger.d["crashes"]
    # init: без повторов внутри волны, из 0…19, воспроизводимо
    for w in range(50):
        s_ = pick_init(0, w)
        assert len(set(s_)) == 5 and all(0 <= x < 20 for x in s_)
        assert s_ == pick_init(0, w)
    print("самопроверка k29_ppo пройдена: пересчёт μ по кешу = записанному, "
          "reference неизменен, откат по KL восстанавливает actor и "
          "optimizer, std(A) = 0 -> actor пропущен, награда влияет на "
          "обновление, порча μ -> отказ; тренер: бюджет и последний неполный "
          "batch, валидации start/end, прерванный + продолжение = непрерывный,"
          " авария внутри цикла списывает резерв, авария после обновления до "
          "удаления pending не блокирует продолжение, аварии после pending и "
          "после ledger.commit восстанавливают счётчики один раз; init без "
          "повторов")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    raise SystemExit("модуль; запуск обучения — k29_protocol.py train")
