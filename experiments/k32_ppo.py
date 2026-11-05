#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-32: PPO латентной политики (r, η) и тренер с бюджетом/возобновлением.

Обновление на свежем batch (1–2 полных цикла, только переходы old-actor):
  logits_old/μ_old — пересчёт текущим actor (= old) по кешу S; сверка с
  записанными при сборе: |Δμ| ≤ 1e−6, |Δlog π| ≤ 1e−4, независимо от
  minibatch 8/32/64; V_old = critic(S) до обновлений; A = R − V_old,
  нормировка по batch (std ≤ 1e−8 — actor пропускается, critic учится);
  4 эпохи, minibatch 32, clip 0.2, grad clip 1.0 отдельно;
  actor: −mean(min(ratio·A, clip·A)) + 0.01·mean KL_joint(θ‖init) (init —
  равномерный маршрут, μ = 0); после КАЖДОГО шага actor — точный средний
  KL_joint(θ‖old) на всём batch: конечный и ≤ 0.03 — принят, иначе откат
  шага (actor + полное состояние AdamW) и остановка actor на batch;
  нечисловой KL — технический отказ; critic — MSE к R, все 4 эпохи.
Тренер (оператор × seed): цикл = задачи 0/6/8 по волне из 5 эпизодов;
batch = 2 полных цикла (последний может быть 1); кап 200 000 физических
шагов; новый цикл — только если 9150 помещается; волна: reserve 3050 ->
исполнение -> атомарный pending-файл (id, расход, old-hash) -> commit;
записанный до commit результат закрывается settle. Обновление — только на
полных циклах; состояние после обновления сохраняется до удаления pending,
повтор не делает optimizer.step второй раз. Валидация (детерминированно,
init 20…24): start, после первого принятого обновления за 100 000, end.
"""
import copy
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k32_collect as kcol32  # noqa: E402
import k32_dist as kd  # noqa: E402
import k32_model as km32  # noqa: E402

CAP = 200_000
MID_MARK = 100_000
VAL_CAP = 3 * len(kcol32.TASKS) * len(kcol32.VAL_INIT) * 610      # 27 450
CFG = dict(lr_actor=1e-4, wd_actor=0.0, lr_critic=3e-4, epochs=4, minibatch=32,
           clip=0.2, grad_clip=1.0, lambda_ref=0.01, kl_old_max=0.03,
           mu_tol=1e-6, logp_tol=1e-4, adv_std_min=1e-8, cycles_per_batch=2,
           cap=CAP, cycle_max=kcol32.CYCLE_MAX, wave_max=kcol32.WAVE_MAX,
           tasks=list(kcol32.TASKS), train_init=list(kcol32.TRAIN_INIT),
           val_init=list(kcol32.VAL_INIT), mid_mark=MID_MARK, val_cap=VAL_CAP,
           critic_loss="MSE", sigma=kd.SIGMA)


def policy_all(torch, actor, X, mb):
    lg, mu = [], []
    with torch.no_grad():
        for s in range(0, len(X), mb):
            a, raw = actor(X[s:s + mb])
            lg.append(a)
            mu.append(kd.mu_of(torch, raw))
    return torch.cat(lg), torch.cat(mu)


def ppo_update(torch, actor, critic, opt_a, opt_c, buf, H, rng_seed, cfg=CFG):
    N = len(buf["r"])
    S = torch.as_tensor(buf["S"], dtype=torch.float32)
    X = km32.actor_inputs(torch, S, H)
    r = torch.as_tensor(buf["r"])
    eta = torch.as_tensor(buf["eta"], dtype=torch.float32)
    R = torch.as_tensor(buf["R"], dtype=torch.float32)
    d = eta.shape[1]
    mb = int(cfg["minibatch"])
    st = dict(n=N, episodes=int(len(set(buf["episode"].tolist()))),
              R_mean=float(R.mean()), epochs=0, attempts=0, accepted_steps=0,
              stop=None, actor_skipped=None)
    actor.eval()
    critic.eval()
    lg_old, mu_old = policy_all(torch, actor, X, mb)
    for m2 in (8, 64):
        l2, u2 = policy_all(torch, actor, X, m2)
        st[f"mb{m2}_mu_max"] = float((u2 - mu_old).abs().max())
        if not st[f"mb{m2}_mu_max"] <= cfg["mu_tol"]:
            raise SystemExit(f"μ зависит от minibatch ({m2})")
    lp_old = kd.log_prob(torch, lg_old.double(), mu_old.double(), r,
                         eta.double())
    st["mu_recompute_max"] = float((mu_old - torch.as_tensor(
        buf["mu"])).abs().max())
    st["logp_recompute_max"] = float((lp_old - torch.as_tensor(
        buf["logpi"])).abs().max())
    if not (st["mu_recompute_max"] <= cfg["mu_tol"] and
            st["logp_recompute_max"] <= cfg["logp_tol"]):
        raise SystemExit(f"пересчёт μ/log π по кешу ≠ записанному: "
                         f"{st['mu_recompute_max']:.2e} / "
                         f"{st['logp_recompute_max']:.2e}")
    with torch.no_grad():
        V = torch.cat([critic(S[s:s + mb]) for s in range(0, N, mb)])
    A = R - V
    st["V_mean"], st["adv_std_raw"] = float(V.mean()), float(A.std())
    actor_on = bool(A.std() > cfg["adv_std_min"])
    if actor_on:
        A = (A - A.mean()) / A.std()
    else:
        st["actor_skipped"] = st["stop"] = "std(A) ≤ 1e-8"
    lg0, mu0 = kd.init_reference(torch, N, d)
    st["kl_init_before"] = float(kd.kl_joint(torch, lg_old, mu_old, lg0,
                                             mu0).mean())
    kl_last = 0.0

    def kl_old_batch():
        lg_n, mu_n = policy_all(torch, actor, X, mb)
        return float(kd.kl_joint(torch, lg_n, mu_n, lg_old, mu_old).mean())
    hist = []
    for ep in range(int(cfg["epochs"])):
        perm = np.random.default_rng([*rng_seed, ep]).permutation(N)
        la_s = lc_s = 0.0
        na = nb = 0
        for s in range(0, N, mb):
            idx = torch.as_tensor(perm[s:s + mb])
            if actor_on:
                actor.train()
                lg, raw = actor(X[idx])
                mu = kd.mu_of(torch, raw)
                lp = kd.log_prob(torch, lg.double(), mu.double(), r[idx],
                                 eta[idx].double())
                ratio = torch.exp(lp - lp_old[idx])
                Am = A[idx].double()
                surr = torch.min(ratio * Am, torch.clamp(
                    ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * Am)
                klr = kd.kl_joint(torch, lg, mu, lg0[idx], mu0[idx])
                la = -surr.mean() + cfg["lambda_ref"] * klr.double().mean()
                if not bool(torch.isfinite(la)):
                    raise SystemExit("нечисловая потеря actor")
                snap_a = copy.deepcopy(actor.state_dict())
                snap_o = copy.deepcopy(opt_a.state_dict())
                opt_a.zero_grad(set_to_none=True)
                la.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(),
                                               cfg["grad_clip"])
                opt_a.step()
                actor.eval()
                st["attempts"] += 1
                kl_try = kl_old_batch()
                if not np.isfinite(kl_try):
                    raise SystemExit("нечисловой KL после шага actor")
                if kl_try <= cfg["kl_old_max"]:
                    st["accepted_steps"] += 1
                    kl_last = kl_try
                    la_s += float(la.detach())
                    na += 1
                    lim = cfg.get("actor_step_limit")
                    if lim is not None and st["accepted_steps"] >= lim:
                        actor_on = False
                        st["stop"] = "лимит шагов (самопроверка)"
                else:
                    actor.load_state_dict(snap_a)
                    opt_a.load_state_dict(snap_o)
                    actor_on = False
                    st["kl_rejected"] = kl_try
                    st["stop"] = (f"KL {kl_try:.4f} > {cfg['kl_old_max']} на "
                                  f"шаге {st['attempts']} (эпоха {ep}); шаг "
                                  f"откатан, actor на batch остановлен")
            critic.train()
            v = critic(S[idx])
            lc = torch.mean((v - R[idx]) ** 2)
            if not bool(torch.isfinite(lc)):
                raise SystemExit("нечисловая потеря critic")
            opt_c.zero_grad(set_to_none=True)
            lc.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(),
                                           cfg["grad_clip"])
            opt_c.step()
            critic.eval()
            lc_s += float(lc.detach())
            nb += 1
        hist.append(dict(epoch=ep, accepted=na,
                         loss_actor=la_s / na if na else None,
                         loss_critic=lc_s / nb))
        st["epochs"] = ep + 1
    kl_fin = kl_old_batch()
    st["kl_old_final"] = kl_fin
    lg_f, mu_f = policy_all(torch, actor, X, mb)
    st["kl_init_final"] = float(kd.kl_joint(torch, lg_f, mu_f, lg0,
                                            mu0).mean())
    if st["accepted_steps"] and abs(kl_fin - kl_last) > 1e-6 * max(1.0,
                                                                  kl_fin):
        raise SystemExit("KL сохранённого actor ≠ KL последнего принятого шага")
    for p in list(actor.parameters()) + list(critic.parameters()):
        if not bool(torch.isfinite(p).all()):
            raise SystemExit("нечисловые веса после обновления")
    st["history"] = hist
    st["accepted"] = st["accepted_steps"] >= 1
    st["stop"] = st["stop"] or "все эпохи выполнены"
    st["route_share"] = np.bincount(buf["r"], minlength=3).tolist()
    st["clip_share"] = float(np.mean(buf["clipped"]))
    st["cap_share"] = float(np.mean(buf["capped"]))
    return st


def tensor_hash(sd):
    import hashlib
    h = hashlib.sha1()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


class Trainer:
    def __init__(self, torch, *, seed, op, H, actor, critic, make_arm, sim,
                 io, ledger, odir, contract, cfg=CFG, log=print):
        self.torch, self.seed, self.op, self.H = torch, int(seed), op, H
        self.actor, self.critic, self.make_arm = actor, critic, make_arm
        self.sim, self.io, self.ledger = sim, io, ledger
        self.dir, self.contract, self.cfg, self.log = odir, contract, cfg, log
        os.makedirs(os.path.join(odir, "pending"), exist_ok=True)
        self.scope = f"train_{op}_s{seed}"
        self.vscope = f"val_{op}_s{seed}"
        self.opt_a = torch.optim.AdamW(actor.parameters(), lr=cfg["lr_actor"],
                                       weight_decay=cfg["wd_actor"])
        self.opt_c = torch.optim.Adam(critic.parameters(),
                                      lr=cfg["lr_critic"])
        self.state_path = os.path.join(odir, "state.pt")
        self.st = dict(wave=0, update=0, accepted=0, updates=[], val={},
                       used_waves=[], discarded=[], finished=False)
        self.init_hash = tensor_hash(actor.state_dict())

    def _crash(self, point):
        if getattr(self, "crash_at", None) == point:
            raise KeyboardInterrupt(point)

    def actor_hash(self):
        return tensor_hash(self.actor.state_dict())

    def save(self):
        d = dict(actor=self.actor.state_dict(),
                 critic=self.critic.state_dict(),
                 opt_a=self.opt_a.state_dict(), opt_c=self.opt_c.state_dict(),
                 st=self.st, contract=self.contract,
                 actor_hash=self.actor_hash(), init_hash=self.init_hash)
        self.torch.save(d, self.state_path + ".tmp")
        os.replace(self.state_path + ".tmp", self.state_path)

    def load(self):
        if not os.path.exists(self.state_path):
            return False
        d = self.torch.load(self.state_path, map_location="cpu",
                            weights_only=False)
        if d["contract"] != self.contract or d["init_hash"] != self.init_hash:
            raise SystemExit(f"{self.state_path}: другой контракт/начальный "
                             f"actor — нужен новый каталог")
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.opt_a.load_state_dict(d["opt_a"])
        self.opt_c.load_state_dict(d["opt_c"])
        self.st = d["st"]
        if self.actor_hash() != d["actor_hash"]:
            raise SystemExit("actor после загрузки ≠ сохранённому")
        return True

    def spent(self):
        return self.ledger.spent(self.scope)

    def wave_path(self, w):
        return os.path.join(self.dir, "pending", f"wave_{w:05d}.pt")

    def wave_eid(self, w):
        return f"k32/{self.op}/s{self.seed}/wave{w}"

    def ledger_problems(self, eid, phys, scope):
        """Запись бюджета eid принадлежит этому запуску (scope) и расходу."""
        dn = self.ledger.d["done"].get(eid)
        pd = self.ledger.d["pending"] or {}
        if dn is not None:
            if dn["scope"] != scope or int(dn["actual"]) != int(phys):
                return [f"{eid}: учтён в {dn['scope']} ({dn['actual']}) ≠ "
                        f"{scope} ({phys})"]
            return []
        if pd.get("id") == eid:
            return [] if pd["scope"] == scope else \
                [f"{eid}: открытый резерв в {pd['scope']} ≠ {scope}"]
        return [f"{eid}: результат не учтён в бюджете"]

    def pending(self):
        """Волны текущего old-actor. Сначала ВСЕ файлы проверяются (контракт,
        оператор, seed, область бюджета, номер волны, задача, eid,
        принадлежность бюджетной записи, old-actor), и только затем —
        удаление использованных и settle."""
        n = len(self.cfg["tasks"])
        files = []
        for f in sorted(glob.glob(os.path.join(self.dir, "pending",
                                               "wave_*.pt"))):
            d = self.torch.load(f, map_location="cpu", weights_only=False)
            w = int(os.path.basename(f)[5:10])
            bad = [k for k, v in (("contract", self.contract), ("op", self.op),
                                  ("seed", self.seed), ("scope", self.scope),
                                  ("wave", w),
                                  ("task", self.cfg["tasks"][w % n]),
                                  ("eid", self.wave_eid(w)))
                   if d.get(k) != v]
            if bad:
                raise SystemExit(f"{f}: чужая волна ({bad})")
            pr = self.ledger_problems(d["eid"], d["phys"], self.scope)
            if pr:
                raise SystemExit(f"{f}: {pr[0]}")
            if w not in self.st["used_waves"] and \
                    d["policy_old"] != self.actor_hash():
                raise SystemExit(f"{f}: собран другим old-actor")
            files.append((f, w, d))
        out = {}
        for f, w, d in files:
            if w in self.st["used_waves"]:
                os.remove(f)                   # авария после сохранения
                continue                       # обновления, до удаления
            self.ledger.settle(d["eid"], d["phys"])
            out[w] = d
        if out:
            self.st["wave"] = max(self.st["wave"], max(out) + 1)
        return out

    def cycles(self, pend):
        """-> [[волны полного цикла]], номер текущей неполной волны цикла."""
        n = len(self.cfg["tasks"])
        full, cur = [], []
        for w in sorted(pend):
            cur.append(w)
            if len(cur) == n:
                full.append(cur)
                cur = []
        return full, cur

    def collect_wave(self):
        w = self.st["wave"]
        n = len(self.cfg["tasks"])
        task = self.cfg["tasks"][w % n]
        eid = self.wave_eid(w)
        self.ledger.reserve(self.scope, eid, amount=self.cfg["wave_max"],
                            limit=self.cfg["cap"])
        arm = self.make_arm(self.actor, stochastic=True)
        starts = kcol32.pick_init(self.seed, w)
        eps, buf, phys, sec = kcol32.run_wave(self.sim, self.io, arm, task,
                                              starts,
                                              episode_base=w * kcol32.WAVE)
        if buf is None:
            raise SystemExit(f"волна {w}: нет переходов")
        d = dict(wave=w, task=task, eid=eid, phys=phys, buf=buf, eps=eps,
                 policy_old=self.actor_hash(), seconds=sec,
                 contract=self.contract, op=self.op, seed=self.seed,
                 scope=self.scope)
        p = self.wave_path(w)
        self.torch.save(d, p + ".tmp")
        os.replace(p + ".tmp", p)
        self._crash("after_write")
        self.ledger.commit(eid, phys)
        self._crash("after_commit")
        self.st["wave"] = w + 1
        self.save()
        sr = np.mean([e["success"] for e in eps])
        self.log(f"  {self.op} s{self.seed} волна {w} (t{task}): успех "
                 f"{sr:.2f}, переходов {len(buf['r'])}, {phys} шагов; бюджет "
                 f"{self.spent()}/{self.cfg['cap']}")

    def update(self, items):
        u = self.st["update"]
        buf = kcol32.concat([d["buf"] for d in items])
        t0 = time.time()
        st = ppo_update(self.torch, self.actor, self.critic, self.opt_a,
                        self.opt_c, buf, self.H,
                        (32, self.seed, 3, u), self.cfg)
        st.update(update=u, waves=[d["wave"] for d in items],
                  seconds=time.time() - t0, train_steps=self.spent())
        self.st["update"] = u + 1
        self.st["accepted"] += int(st["accepted"])
        self.st["used_waves"] += [d["wave"] for d in items]
        self.st["updates"].append(st)
        self.save()
        self._crash("after_update_save")
        for d in items:
            os.remove(self.wave_path(d["wave"]))
        self.log(f"  {self.op} s{self.seed} обновление {u}: переходов "
                 f"{st['n']}, R {st['R_mean']:.2f}, шагов actor "
                 f"{st['accepted_steps']}/{st['attempts']}; {st['stop']}; KL "
                 f"old/init {st['kl_old_final']:.4f}/{st['kl_init_final']:.4f}")
        return st

    def validate(self, tag):
        if tag in self.st["val"]:
            return
        arm = self.make_arm(self.actor, stochastic=False)
        res, phys = {}, 0
        vdir = os.path.join(self.dir, "val")
        os.makedirs(vdir, exist_ok=True)
        ah = self.actor_hash()
        for i, task in enumerate(self.cfg["tasks"]):
            eid = f"k32/{self.op}/s{self.seed}/val_{tag}_t{task}"
            vf = os.path.join(vdir, f"{tag}_t{task}.json")
            if os.path.exists(vf):            # результат пережил аварию
                v = json.load(open(vf, encoding="utf-8"))
                bad = [k for k, x in (("eid", eid), ("actor_hash", ah),
                                      ("contract", self.contract),
                                      ("scope", self.vscope))
                       if v.get(k) != x]
                pr = bad or self.ledger_problems(eid, v["phys"], self.vscope)
                if pr:
                    raise SystemExit(f"{vf}: {pr}")
                self.ledger.settle(eid, v["phys"])
            else:
                if self.ledger.is_done(eid):
                    raise SystemExit(f"{eid}: учтён, но результата нет")
                self.ledger.reserve(self.vscope, eid,
                                    amount=self.cfg["wave_max"])
                eps, _b, ph, _s = kcol32.run_wave(
                    self.sim, self.io, arm, task, list(self.cfg["val_init"]),
                    episode_base=10 ** 6 + i * 10, record=False)
                v = dict(eid=eid, actor_hash=ah, contract=self.contract,
                         scope=self.vscope, phys=int(ph),
                         success=[bool(e["success"]) for e in eps],
                         episodes=eps)
                with open(vf + ".tmp", "w", encoding="utf-8") as fh:
                    json.dump(v, fh, ensure_ascii=False, default=float)
                os.replace(vf + ".tmp", vf)
                self._crash("val_after_write")
                self.ledger.commit(eid, ph)
            phys += int(v["phys"])
            res[str(task)] = list(v["success"])
        sr = float(np.mean([np.mean(v) for v in res.values()]))
        self.st["val"][tag] = dict(sr=sr, per_task=res, phys=phys,
                                   at_train_steps=self.spent(),
                                   update=self.st["update"],
                                   actor_hash=self.actor_hash())
        self.save()
        self.log(f"  {self.op} s{self.seed} валидация {tag}: SR {sr:.3f} "
                 f"{ {k: sum(v) for k, v in res.items()} }")

    def run(self):
        self.load()
        self.validate("start")
        n = len(self.cfg["tasks"])
        while True:
            pend = self.pending()
            full, cur = self.cycles(pend)
            room = self.cfg["cap"] - self.spent()
            can_new_cycle = room >= self.cfg["cycle_max"]
            if cur:                                   # неполный цикл
                need = (n - len(cur)) * self.cfg["wave_max"]
                if room >= need:
                    self.collect_wave()
                    continue
                for w in cur:                          # не помещается —
                    os.remove(self.wave_path(w))       # отбросить
                    self.st["discarded"].append(w)
                self.st["used_waves"] += cur
                self.save()
                continue
            if len(full) >= self.cfg["cycles_per_batch"] or \
                    (full and not can_new_cycle):
                items = [pend[w] for c in full[:self.cfg["cycles_per_batch"]]
                         for w in c]
                su = self.update(items)
                if su["accepted"] and "mid" not in self.st["val"] and \
                        su["train_steps"] >= self.cfg["mid_mark"]:
                    self.validate("mid")
                continue
            if not can_new_cycle:
                break
            self.collect_wave()
        if "mid" not in self.st["val"]:
            self.st["mid_note"] = "отметка 100 000 не пересечена принятым " \
                                  "обновлением"
        self.validate("end")
        self.st["finished"] = True
        self.st["no_actor_update"] = self.st["accepted"] == 0
        self.save()
        return self.st


# --- самопроверка -----------------------------------------------------------------
class ToySim:
    """Эпизод: done на 12-м шаге; успех зависит от init и команд."""

    def reset(self, task, starts):
        self.t, self.init, self.acc = 0, int(starts[0]), 0.0
        return self._obs()

    def _obs(self):
        st = np.zeros((1, 9), np.float32)
        st[0, 0] = self.init * 0.01 + self.acc
        img = np.full((1, 4, 4, 3), self.t % 7, np.uint8)
        return dict(state=st, agentview_image=img,
                    robot0_eye_in_hand_image=img)

    def step(self, a):
        self.t += 1
        self.acc += float(np.asarray(a)[0, 0])
        if self.t <= 10:
            return self._obs(), np.array([0.0]), np.array([False]), \
                np.array([False])
        d = self.t >= 22
        s = bool(d and (self.init + int(abs(self.acc) * 1e3)) % 2 == 0)
        return self._obs(), np.array([float(s)]), np.array([d]), np.array([s])


def selftest():
    import tempfile
    import torch
    import k21_model as km
    import k25_model as k25m
    import k29_model as k29m
    import k31_collect as kcol
    import k32_policy as kpol
    k29m.no_tf32(torch)
    M = k29m.build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    mod = k25m.build(torch)["K25"](km.fake_bar(torch), L, U, seed=0).eval()
    with torch.no_grad():
        g = torch.Generator().manual_seed(3)
        for p in list(mod.cond.parameters()) + list(mod.readout.parameters()):
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    for p in mod.parameters():
        p.requires_grad_(False)
    dec = M["Decoder"](mod).freeze()
    H = mod.H
    rng = np.random.default_rng(0)
    norm = km32.fit_normalizer(rng.normal(size=(64, 6 * H + 72)), H)

    class FIO(kcol.FakeIO):
        def prep(self, obs, desc, task):
            x = float(obs["state"][0, 0])
            bt = km.fake_batch(torch, B=1, seed=int(abs(x) * 1e3) % 13)
            s8 = np.zeros((1, 8))
            s8[0, :3] = x
            return bt, 1, s8
    io = FIO()

    def make(op, d):
        a, c = km32.build(torch, H, d, 0, norm)

        def make_arm(actor, stochastic):
            return kpol.Arm(torch, mod, dec, M, kind=op, seed=0, actor=actor,
                            stochastic=stochastic)
        return a, c, make_arm

    # 1. сбор + обновление: пересчёт по кешу, шаги, откат по KL
    a, c, mk = make("T", 6)
    eps, buf, phys, _s = kcol32.run_wave(ToySim(), io, mk(a, True), 0,
                                         [0, 1, 2, 3, 4], episode_base=0)
    eps2, buf2, _p, _s = kcol32.run_wave(ToySim(), io, mk(a, True), 6,
                                         [5, 6, 7, 8, 9], episode_base=5)
    B = kcol32.concat([buf, buf2])
    assert kcol32.buf_problems(B, 6, H) == []
    assert len(set(B["R"].tolist())) == 2, "в игрушке нужны оба исхода"
    oa = torch.optim.AdamW(a.parameters(), lr=1e-2)
    oc = torch.optim.Adam(c.parameters(), lr=3e-4)
    cfg = dict(CFG, lr_actor=1e-2)
    a0 = copy.deepcopy(a.state_dict())
    st = ppo_update(torch, a, c, oa, oc, B, H, (0,), cfg)
    assert st["mu_recompute_max"] <= 1e-6 and st["attempts"] >= 1
    assert st["accepted_steps"] >= 1 and tensor_hash(a.state_dict()) != \
        tensor_hash(a0)
    assert st["kl_old_final"] <= 0.03
    # откат: порог 0 -> первый шаг откатан вместе с AdamW, critic учится
    a2, c2, _m = make("T", 6)
    oa2 = torch.optim.AdamW(a2.parameters(), lr=1e-2)
    oc2 = torch.optim.Adam(c2.parameters(), lr=3e-4)
    s_a, s_o = copy.deepcopy(a2.state_dict()), copy.deepcopy(oa2.state_dict())
    c_before = tensor_hash(c2.state_dict())
    st2 = ppo_update(torch, a2, c2, oa2, oc2, B, H, (0,),
                     dict(cfg, kl_old_max=0.0))
    assert st2["accepted_steps"] == 0 and st2["attempts"] == 1
    assert tensor_hash(a2.state_dict()) == tensor_hash(s_a)
    assert oa2.state_dict()["state"] == s_o["state"]
    assert tensor_hash(c2.state_dict()) != c_before and st2["epochs"] == 4
    # изменение наград меняет обновление
    a3, c3, _m = make("T", 6)
    oa3 = torch.optim.AdamW(a3.parameters(), lr=1e-2)
    oc3 = torch.optim.Adam(c3.parameters(), lr=3e-4)
    ppo_update(torch, a3, c3, oa3, oc3, dict(B, R=1 - B["R"]), H, (0,), cfg)
    assert tensor_hash(a3.state_dict()) != tensor_hash(a.state_dict())
    # подмена записанного μ — отказ
    try:
        a4, c4, _m = make("T", 6)
        ppo_update(torch, a4, c4, torch.optim.AdamW(a4.parameters()),
                   torch.optim.Adam(c4.parameters()),
                   dict(B, mu=B["mu"] + 1e-3), H, (0,), cfg)
        raise AssertionError("подменённый μ принят")
    except SystemExit:
        pass
    # 2. тренер: полный прогон и возобновление после аварий
    caps = {"train_T_s0": 4 * kcol32.CYCLE_MAX + 100,
            "val_T_s0": VAL_CAP}
    small = dict(CFG, cap=caps["train_T_s0"], mid_mark=10 ** 9)

    def trainer(td, crash=None):
        a_, c_, mk_ = make("T", 6)
        L_ = kcol.Ledger(os.path.join(td, "ledger.json"), caps)
        t = Trainer(torch, seed=0, op="T", H=H, actor=a_, critic=c_,
                    make_arm=mk_, sim=ToySim(), io=io, ledger=L_,
                    odir=os.path.join(td, "run"), contract=dict(c=1),
                    cfg=small, log=lambda *x: None)
        t.crash_at = crash
        return t
    with tempfile.TemporaryDirectory() as td:
        t = trainer(td)
        st_full = t.run()
        h_full = t.actor_hash()
        spent_full = t.spent()
        n_upd = st_full["update"]
        assert n_upd >= 2 and st_full["finished"]
        assert set(st_full["val"]) >= {"start", "end"}
        assert spent_full <= small["cap"]
    for crash in ("after_write", "after_commit", "after_update_save"):
        with tempfile.TemporaryDirectory() as td:
            t = trainer(td, crash)
            try:
                t.run()
                raise AssertionError(f"авария {crash} не сработала")
            except KeyboardInterrupt:
                pass
            t2 = trainer(td)
            st2 = t2.run()
            assert t2.actor_hash() == h_full, crash
            assert st2["update"] == n_upd, crash
            assert t2.spent() == spent_full, crash
            assert not t2.ledger.d["crashes"], crash
    # авария валидации между записью и commit: повтор без исполнения
    with tempfile.TemporaryDirectory() as td:
        t = trainer(td, "val_after_write")
        try:
            t.run()
            raise AssertionError("авария валидации не сработала")
        except KeyboardInterrupt:
            pass
        t2 = trainer(td)
        st2 = t2.run()
        assert t2.actor_hash() == h_full and st2["update"] == n_upd
        assert t2.ledger.spent("val_T_s0") == sum(
            v["phys"] for v in st2["val"].values()), "валидация задвоена"
        assert not t2.ledger.d["crashes"]
    # чужая волна T в тренере P при одинаковом начальном actor — отказ до
    # каких-либо изменений бюджета/файлов
    with tempfile.TemporaryDirectory() as td:
        t = trainer(td, "after_commit")
        try:
            t.run()
        except KeyboardInterrupt:
            pass
        caps_p = dict(caps, train_P_s0=caps["train_T_s0"], val_P_s0=VAL_CAP)
        a_, c_, mk_ = make("P", 6)
        Lp = kcol.Ledger(os.path.join(td, "ledger_p.json"), caps_p)
        tp = Trainer(torch, seed=0, op="P", H=H, actor=a_, critic=c_,
                     make_arm=mk_, sim=ToySim(), io=io, ledger=Lp,
                     odir=os.path.join(td, "run"), contract=dict(c=1),
                     cfg=dict(small, cap=caps["train_T_s0"]),
                     log=lambda *x: None)
        before = json.dumps(Lp.d, sort_keys=True)
        files = sorted(os.listdir(os.path.join(td, "run", "pending")))
        try:
            tp.pending()
            raise AssertionError("волна T принята тренером P")
        except SystemExit:
            pass
        assert json.dumps(Lp.d, sort_keys=True) == before
        assert sorted(os.listdir(os.path.join(td, "run", "pending"))) == files
    # другой контракт — отказ возобновления
    with tempfile.TemporaryDirectory() as td:
        t = trainer(td, "after_commit")
        try:
            t.run()
        except KeyboardInterrupt:
            pass
        a_, c_, mk_ = make("T", 6)
        try:
            Trainer(torch, seed=0, op="T", H=H, actor=a_, critic=c_,
                    make_arm=mk_, sim=ToySim(), io=io,
                    ledger=kcol.Ledger(os.path.join(td, "ledger.json"), caps),
                    odir=os.path.join(td, "run"), contract=dict(c=2),
                    cfg=small, log=lambda *x: None).run()
            raise AssertionError("чужой контракт принят")
        except SystemExit:
            pass
    print("самопроверка k32_ppo пройдена: пересчёт μ/log π по кешу "
          "(minibatch 8/32/64), принятые шаги и KL ≤ 0.03, откат шага с "
          "AdamW при KL выше порога (critic продолжает 4 эпохи), награды "
          "меняют обновление, подмена μ — отказ; тренер: полный прогон в "
          "капе, возобновление после аварий (запись/commit/обновление) даёт "
          "тот же actor, число обновлений и расход без аварийных списаний; "
          "авария валидации между записью и commit — результат сохранён, "
          "расход один раз; волна T в тренере P (одинаковый actor) — отказ "
          "до изменения бюджета и файлов; чужой контракт — отказ")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
