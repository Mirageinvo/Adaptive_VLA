#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-28: рука для пошагового цикла (k28_rollout_core).

parents_goal_fb24_s<S> — план (один тяжёлый проход замороженного parents24
сида + дерево поз + контексты) и восемь лёгких шагов CPU-контроллера по
свежему состоянию; parents_goal_fb24legacy_s<S> — тот же K28 в режиме
legacy: только номинальный чанк, контроллер и дерево не строятся (сверка с
историческим parents24).
Загрузка: база — исторический parents24 сида (SHA файла, отпечаток весов,
режим/seed контракта, prep_id); новые модули — best.pt обучения K-28
(SHA, отпечаток новых тензоров, контракт: режим, seed, отпечаток базы,
k28_prep_id, SHA кода). Всё заморожено, eval. Контроллер копируется на CPU
и сверяется с чекпойнтом.
Номинальный чанк — как у руки K-21/K-22: рука float32, схват ±1 по знаку
логита; вход — тот же батч харнесса, pos_offset задачи, state11 из сырого
state8 (process_state), fp16 autocast.
"""
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k28_control as kc  # noqa: E402
import k28_model as k28m  # noqa: E402

T = 20


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def parse_label(label):
    m = re.fullmatch(r"parents_goal_fb24(legacy)?_s(\d+)", str(label))
    if not m:
        raise SystemExit(f"метка {label!r}: нужна parents_goal_fb24_s<seed> "
                         f"или parents_goal_fb24legacy_s<seed>")
    return ("legacy" if m.group(1) else "fb"), int(m.group(2))


def state_hash_new(mod):
    import k21_train as ktr
    return ktr.state_hash(k28m.new_weights(ktr.trainable_state(mod)))


def load_model(torch, label, spec, device):
    """K28 с весами базы и (fb) новыми модулями; сверки происхождения."""
    import k21_data as d21
    import k21_protocol as pp
    import k21_train as ktr
    mode, seed = parse_label(label)
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    src = spec["source"]
    if pp.full_sha(src["ckpt"]) != src["ckpt_sha1"]:
        raise SystemExit(f"{label}: база {src['ckpt']} ≠ SHA плана")
    ck = torch.load(src["ckpt"], map_location="cpu", weights_only=False)
    if ck["config"].get("mode") != "parents24" or \
            int(ck["config"].get("seed", -1)) != seed or \
            ck["config"]["prep_id"] != prep["prep_id"]:
        raise SystemExit(f"{label}: источник не parents24 seed {seed}")
    nck = None
    if mode == "fb":
        if pp.full_sha(spec["ckpt"]) != spec["ckpt_sha1"]:
            raise SystemExit(f"{label}: чекпойнт K-28 ≠ SHA плана")
        nck = torch.load(spec["ckpt"], map_location="cpu",
                         weights_only=False)
        cfg = nck["config"]
        p = []
        if cfg.get("mode") != k28m.MODE or int(cfg.get("seed", -1)) != seed:
            p.append("режим/seed контракта")
        if cfg.get("base_hash") != src["weights_hash"]:
            p.append("отпечаток базы контракта")
        if nck["new_hash"] != spec.get("new_hash"):
            p.append("отпечаток новых модулей ≠ плану")
        bad = [f for f, s in cfg["code"].items()
               if sha(os.path.join(HERE, f)) != s]
        if bad:
            p.append(f"код изменён после обучения: {bad}")
        if p:
            raise SystemExit(f"{label}: " + "; ".join(p))
        sig = (cfg["sigma_p"], cfg["sigma_R"])
    else:
        sig = ((0.01, 0.01, 0.01), 0.02)
    dev = torch.device(device)
    bar = d21.load_bar(dev, prep["bar_pin"])
    m = bar["model"]
    m.action_expert.float()
    mod = k28m.build(torch)["K28"](
        m, np.asarray(prep["bounds_codec"]["L"]),
        np.asarray(prep["bounds_codec"]["U"]), prep["state_norm"]["mean"],
        prep["state_norm"]["std"], checkpoint=False, seed=seed,
        sigma_p=sig[0], sigma_R=sig[1], mode=mode).to(dev)
    own = dict(mod.named_parameters())
    own.update(dict(mod.named_buffers()))
    with torch.no_grad():
        for k, v in ck["weights"].items():
            own[k].copy_(v.to(own[k].device, own[k].dtype))
        if nck is not None:
            for k, v in nck["weights"].items():
                if not k.startswith(k28m.NEW_PREFIXES):
                    raise SystemExit(f"в чекпойнте K-28 чужой ключ {k}")
                own[k].copy_(v.to(own[k].device, own[k].dtype))
    if ktr.state_hash(k28m.shared_keys(ktr.trainable_state(mod))) != \
            src["weights_hash"]:
        raise SystemExit("база после загрузки ≠ отпечатку источника")
    if nck is not None and state_hash_new(mod) != nck["new_hash"]:
        raise SystemExit("новые модули после загрузки ≠ отпечатку")
    mod.eval()
    for p_ in mod.parameters():
        p_.requires_grad_(False)
    return mod, bar, prep, mode, seed


class Arm:
    """plan(...) — один тяжёлый вызов; step(...) — лёгкий шаг (fb) или
    номинальная команда (legacy). Журнал — словарь списков."""

    def __init__(self, torch, mod, bar, prep, mode, label, meta):
        self.torch, self.mod, self.bar, self.mode = torch, mod, bar, mode
        self.label, self.meta = label, meta
        self.dev = next(mod.parameters()).device
        self.runtime = None
        if mode == "fb":
            C = kc.build(torch)
            self.runtime = C["Runtime"](mod.ctrl.state_dict())
            h_cpu = self.runtime.weights_hash()
            h = hashlib.sha1()
            for k, v in sorted(mod.ctrl.state_dict().items()):
                h.update(k.encode())
                h.update(np.ascontiguousarray(v.detach().float().cpu()
                                              .numpy()).tobytes())
            if h_cpu != h.hexdigest():
                raise SystemExit("CPU-контроллер ≠ весам модели")
            self.meta["ctrl_cpu_hash"] = h_cpu
        self.n_plan = 0
        self.log = {}
        self.reset_log()

    def reset_log(self):
        self.log = {k: [] for k in (
            "plan_a", "plan_grip_logit", "plan_nodes_t", "plan_nodes_p",
            "plan_nodes_R", "plan_ref_p", "plan_ref_R", "plan_c", "plan_id",
            "step_plan", "step_j", "step_state8", "step_e", "step_e_norm",
            "step_r", "step_da", "step_a0", "step_a", "step_active")}

    def plan(self, batch, pos_off, state8, autocast):
        import k21_model as km
        torch = self.torch
        s11 = torch.as_tensor(km.state11(np.asarray(state8, np.float64)),
                              device=self.dev)
        pid = f"{self.label}:{self.n_plan}"
        with torch.no_grad():
            pl = self.mod.plan(batch, int(pos_off), s11,
                               torch.as_tensor(np.asarray(state8, np.float32),
                                               device=self.dev), plan_id=pid,
                               autocast=autocast)
        arm = pl["a0"].float()
        a = torch.cat([arm, pl["grip"][..., None]], -1).cpu().numpy() \
            .astype(np.float32)
        if not np.isfinite(a).all():
            raise SystemExit("номинальный чанк не конечен")
        self.n_plan += 1
        cpu = None
        if self.mode == "fb":
            cpu = self.mod.to_cpu_plan(pl)
            self.runtime.begin(cpu)
            for k, v in (("plan_nodes_t", pl["nodes"]["t"]),
                         ("plan_nodes_p", pl["nodes"]["p"]),
                         ("plan_nodes_R", pl["nodes"]["R"]),
                         ("plan_ref_p", cpu["ref_p"]),
                         ("plan_ref_R", cpu["ref_R"]),
                         ("plan_c", cpu["c"])):
                self.log[k].append(v.detach().float().cpu().numpy())
        self.log["plan_a"].append(a)
        self.log["plan_grip_logit"].append(
            pl["grip_logit"].float().cpu().numpy())
        self.log["plan_id"].append(self.n_plan - 1)
        return dict(a=a, cpu=cpu, plan_id=pid, index=self.n_plan - 1)

    def step(self, plan, state8, j, active):
        if self.mode == "legacy":
            return plan["a"][:, j]
        a, dbg = self.runtime.step(plan["cpu"], state8, j, active)
        a = a.astype(np.float32)
        L = self.log
        L["step_plan"].append(plan["index"])
        L["step_j"].append(int(j))
        L["step_state8"].append(np.asarray(state8, np.float32))
        for k in ("e", "e_norm", "r", "da", "a0"):
            L[f"step_{k}"].append(np.asarray(dbg[k], np.float32))
        L["step_a"].append(a)
        L["step_active"].append(np.asarray(active, bool))
        return a

    def take(self):
        """Журнал блока -> массивы npz (префикс k28_)."""
        out = {}
        for k, v in self.log.items():
            if v:
                out[f"k28_{k}"] = np.stack([np.asarray(x) for x in v])
        self.reset_log()
        return out


def build_arm(torch, label, spec, device, plan_sha1):
    mod, bar, prep, mode, seed = load_model(torch, label, spec, device)
    meta = dict(arm="k28", label=label, arch="parents_goal_fb24",
                mode=mode, train_seed=seed, plan_sha1=plan_sha1,
                ckpt_sha1=spec.get("ckpt_sha1") or spec["source"]["ckpt_sha1"],
                base_weights_hash=spec["source"]["weights_hash"],
                new_hash=spec.get("new_hash"), prep_id=prep["prep_id"],
                bar_revision=prep["bar_pin"]["revision"],
                control=kc.CONFIG, heavy_stride=8, control_stride=1,
                k28_policy_sha1=sha(os.path.abspath(__file__)))
    meta["model_fingerprint"] = hashlib.sha1("|".join(str(meta[k]) for k in (
        "label", "ckpt_sha1", "base_weights_hash", "new_hash", "prep_id",
        "k28_policy_sha1", "plan_sha1")).encode()).hexdigest()[:12]
    return Arm(torch, mod, bar, prep, mode, label, meta)


def selftest():
    import copy
    import contextlib
    import torch
    import k21_model as km
    assert parse_label("parents_goal_fb24_s1") == ("fb", 1)
    assert parse_label("parents_goal_fb24legacy_s0") == ("legacy", 0)
    M = k28m.build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    bar0 = km.fake_bar(torch)
    st8 = np.concatenate([np.random.default_rng(0).normal(0, 0.1, (3, 3)),
                          np.random.default_rng(1).normal(0, 0.5, (3, 3)),
                          np.zeros((3, 2))], 1)
    bt = km.fake_batch(torch, B=3, pad=2)
    outs = {}
    for mode in ("legacy", "fb"):
        mod = M["K28"](copy.deepcopy(bar0), L, U, seed=0, mode=mode).eval()
        for p in mod.parameters():
            p.requires_grad_(False)
        arm = Arm(torch, mod, dict(model=bar0), None, mode,
                  f"parents_goal_fb24{'legacy' if mode == 'legacy' else ''}"
                  f"_s0", {})
        pl = arm.plan(bt, 1, st8, contextlib.nullcontext())
        rows = [arm.step(pl, st8, j, np.ones(3, bool)) for j in range(8)]
        outs[mode] = (pl["a"], np.stack(rows))
        z = arm.take()
        if mode == "fb":
            assert z["k28_step_j"].tolist() == list(range(8))
            assert z["k28_plan_c"].shape == (1, 3, 20, kc.CTX)
    # нулевой последний слой: шаги fb = номинал legacy побитово
    assert np.array_equal(outs["fb"][0], outs["legacy"][0])
    assert np.array_equal(outs["fb"][1], outs["legacy"][1])
    print("самопроверка k28_policy пройдена: метки fb/legacy, план — один "
          "тяжёлый вызов, 8 шагов CPU-контроллера, журнал плана/шагов, "
          "нулевой последний слой -> команды = legacy побитово")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль руки; запуск — через k28_harness.py")
