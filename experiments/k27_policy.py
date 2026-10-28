#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-27: рука parents_mem24_s<S> (память) и parents_mem24off_s<S> (тот же
K27 с memory="off" — техническая сверка с историческим parents24) для
штатного холодного харнесса.

Учёт сред (не по флагу first): адаптер оборачивает utils.get_envs —
reset векторной среды выдаёт каждой среде новый episode_id и обнуляет её
счётчик шагов, step увеличивает счётчики. На каждом вызове рука берёт
env_id (индекс в векторе), episode_id и фактически исполненный шаг; память
среды — из MemoryBank; k = шаг − шаг снимка. Новый эпизод (в т. ч. с тем же
env_id), k ≥ 20 — пустая память; шаг назад — отказ.
Загрузка: база — исторический parents24 сида (SHA файла, отпечаток), память —
best.pt обучения K-27 (SHA, отпечаток модуля, контракт: режим, seed, SHA
кода); всё заморожено, eval. Журнал каждого вызова: env/эпизод/шаг/k,
использование памяти, живые узлы, средние по головам соответствия A0/A1,
гейты, интервалы средних, RMS поправок; в smoke-блоках на первом вызове с
памятью — debug-фрагмент (S0/S1, интервалы, старые W/чанк, k, A0/A1 по
головам) для независимой CPU-проверки сдвига, масок и соответствий.
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

import k27_memory as kmem  # noqa: E402
import k27_model as k27m  # noqa: E402

TRACK = dict(episode=None, steps=None, next_episode=0, resets=0)


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def parse_label(label):
    m = re.fullmatch(r"parents_mem24(off)?_s(\d+)", str(label))
    if not m:
        raise SystemExit(f"метка {label!r}: нужна parents_mem24_s<seed> или "
                         f"parents_mem24off_s<seed>")
    return ("off" if m.group(1) else "on"), int(m.group(2))


# --- учёт сред -----------------------------------------------------------------
class TrackedEnvs:
    def __init__(self, envs):
        self._envs = envs

    def reset(self, *a, **k):
        obs = self._envs.reset(*a, **k)
        n = len(obs["state"])
        TRACK["episode"] = np.arange(TRACK["next_episode"],
                                     TRACK["next_episode"] + n)
        TRACK["next_episode"] += n
        TRACK["steps"] = np.zeros(n, np.int64)
        TRACK["resets"] += 1
        return obs

    def step(self, action):
        out = self._envs.step(action)
        if TRACK["steps"] is None:
            raise SystemExit("учёт сред: step до reset")
        TRACK["steps"] += 1
        return out

    def __getattr__(self, name):
        return getattr(self._envs, name)


def install_env_tracking(root="third_party/actioncodec"):
    r = os.path.abspath(root)
    if r not in sys.path:
        sys.path.insert(0, r)
    import utils
    if getattr(utils.get_envs, "_k27", False):
        return
    orig = utils.get_envs

    def get_envs(*a, **k):
        envs, desc = orig(*a, **k)
        return TrackedEnvs(envs), desc
    get_envs._k27 = True
    utils.get_envs = get_envs


def env_view(B):
    if TRACK["episode"] is None or len(TRACK["episode"]) != B:
        raise SystemExit(f"учёт сред: сред {None if TRACK['episode'] is None else len(TRACK['episode'])}, батч {B}")
    return list(range(B)), TRACK["episode"].tolist(), TRACK["steps"].tolist()


# --- загрузка --------------------------------------------------------------------
def spec_problems(label, spec, cfg=None, mem_hash=None):
    mode, seed = parse_label(label)
    p = []
    if spec.get("memory") != mode or int(spec.get("seed", -1)) != seed:
        p.append(f"{label}: спецификация {spec.get('memory')}/"
                 f"{spec.get('seed')}")
    if mode == "on":
        if cfg is None or cfg.get("mode") != k27m.MODE or \
                int(cfg.get("seed", -1)) != seed or \
                cfg.get("memory") != kmem.CONFIG:
            p.append(f"{label}: контракт чекпойнта памяти")
        if mem_hash != spec.get("mem_hash"):
            p.append(f"{label}: отпечаток памяти ≠ плану")
    return p


def load_policy(torch, label, spec, device):
    import k21_data as d21
    import k21_protocol as pp
    import k21_train as kt
    mode, seed = parse_label(label)
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    src = spec["source"]
    if pp.full_sha(src["ckpt"]) != src["ckpt_sha1"]:
        raise SystemExit(f"{label}: база {src['ckpt']} ≠ SHA плана")
    ck = torch.load(src["ckpt"], map_location="cpu", weights_only=False)
    if ck["config"].get("mode") != "parents24" or \
            int(ck["config"].get("seed", -1)) != seed or \
            ck["config"]["prep_id"] != prep["prep_id"]:
        raise SystemExit(f"{label}: источник не parents24 seed {seed} этой "
                         f"подготовки")
    mck = None
    if mode == "on":
        if pp.full_sha(spec["ckpt"]) != spec["ckpt_sha1"]:
            raise SystemExit(f"{label}: чекпойнт памяти ≠ SHA плана")
        mck = torch.load(spec["ckpt"], map_location="cpu",
                         weights_only=False)
        prob = spec_problems(label, spec, mck["config"], mck["mem_hash"])
        bad = [f for f, s in mck["config"]["code"].items()
               if sha(os.path.join(HERE, f)) != s]
        if bad:
            prob.append(f"код модели изменён после обучения: {bad}")
        if prob:
            raise SystemExit("; ".join(prob))
    else:
        prob = spec_problems(label, spec)
        if prob:
            raise SystemExit("; ".join(prob))
    dev = torch.device(device)
    bar = d21.load_bar(dev, prep["bar_pin"])
    m = bar["model"]
    m.action_expert.float()
    mod = k27m.build(torch)["K27"](
        m, np.asarray(prep["bounds_codec"]["L"]),
        np.asarray(prep["bounds_codec"]["U"]), prep["state_norm"]["mean"],
        prep["state_norm"]["std"], checkpoint=False, seed=seed,
        memory=mode).to(dev)
    own = dict(mod.named_parameters())
    own.update(dict(mod.named_buffers()))
    with torch.no_grad():
        for k, v in ck["weights"].items():
            own[k].copy_(v.to(own[k].device, own[k].dtype))
        if mck is not None:
            mod.mem.load_state_dict({k: v.to(dev) for k, v in
                                     mck["mem"].items()})
    if kt.state_hash(k27m.shared_keys(kt.trainable_state(mod))) != \
            src["weights_hash"]:
        raise SystemExit("база после загрузки ≠ отпечатку источника")
    if mck is not None and kt.state_hash({k: v.detach().cpu() for k, v in
                                          mod.mem.state_dict().items()}) \
            != mck["mem_hash"]:
        raise SystemExit("память после загрузки ≠ отпечатку")
    mod.eval()
    for p in mod.parameters():
        p.requires_grad_(False)
    # для журнала k21_main_policy: отпечаток «весов» — база + память
    ck_view = dict(ck, weights_hash=spec.get("weights_hash",
                                             src["weights_hash"]),
                   update=(mck or {}).get("update", ck.get("update")))
    return mod, bar, ck_view, prep


# --- журнал -----------------------------------------------------------------------
def journal_record(out, B, env, ks, use, keep_debug=False, mem_in=None):
    """-> массивы фиксированной формы на вызов (без памяти — −1/0)."""
    mi = out.get("mem")
    rec = dict(mem_env=np.int32(env), mem_episode=np.int64(ks["episode"]),
               mem_step=np.int64(ks["steps"]),
               mem_k=np.int32([k if u else -1 for k, u in zip(ks["k"],
                                                              use)]),
               mem_use=np.float32(use),
               mem_iv_mid=out["mid"].detach().float().cpu().numpy())
    if mi is None:
        rec.update(mem_alive_root=np.zeros((B, 4), np.float32),
                   mem_alive_mid=np.zeros((B, 8), np.float32),
                   mem_A0=np.zeros((B, 4, 4), np.float32),
                   mem_A1=np.zeros((B, 8, 8), np.float32),
                   mem_g_root=np.full((B, 4), -1, np.float32),
                   mem_g_mid=np.full((B, 8), -1, np.float32),
                   mem_rms=np.float32([0.0, 0.0]))
        return rec

    def c(x):
        return x.detach().float().cpu().numpy()
    rec.update(mem_alive_root=c(mi["alive_root"]),
               mem_alive_mid=c(mi.get("alive_mid", mi["alive_root"]
                                      .new_zeros(B, 8))),
               mem_A0=c(mi["A0"].mean(1)),
               mem_A1=c(mi["A1"].mean(1)) if "A1" in mi else
               np.zeros((B, 8, 8), np.float32),
               mem_g_root=np.where(c(mi["has_root"]) > 0, c(mi["g_root"]),
                                   -1).astype(np.float32),
               mem_g_mid=np.where(c(mi["has_mid"]) > 0, c(mi["g_mid"]),
                                  -1).astype(np.float32) if "g_mid" in mi
               else np.full((B, 8), -1, np.float32),
               mem_rms=np.float32([float(mi["rms_root"]),
                                   float(mi.get("rms_mid", 0.0))]))
    if keep_debug:
        sn = mem_in["snap"]
        rec.update(
            mem_dbg_S0=c(mi["S0"]), mem_dbg_A0=c(mi["A0"]),
            mem_dbg_S1=c(mi["S1"]) if "S1" in mi else
            np.zeros((B, kmem.N_HEADS, 8, 8), np.float32),
            mem_dbg_A1=c(mi["A1"]) if "A1" in mi else
            np.zeros((B, kmem.N_HEADS, 8, 8), np.float32),
            mem_dbg_root_now=c(out["root"]), mem_dbg_mid_now=c(out["mid"]),
            mem_dbg_old_iv_root=c(sn["iv_root"]),
            mem_dbg_old_iv_mid=c(sn["iv_mid"]), mem_dbg_old_W0=c(sn["W0"]),
            mem_dbg_old_W1=c(sn["W1"]), mem_dbg_old_chunk=c(sn["chunk"]),
            mem_dbg_k=np.float32(mem_in["k"]),
            mem_dbg_use=np.float32(mem_in["use"]))
    return rec


def wrap_arm(arm, torch, mode, label, debug=False):
    """Память, проверки и журнал вокруг руки k21_main_policy (arm.k21 —
    модель K27, arm.act — базовый вызов). В режиме off память не выбирается
    и банк не пополняется: use = 0, mem_in = None на каждом вызове."""
    mod = arm.k21
    fwd0 = mod.forward
    bank = kmem.MemoryBank()
    last, st = {}, dict(dbg_done=False, calls=0)

    def forward(batch, po, state):
        out = fwd0(batch, po, state, last.get("mem_in"))
        last["out"] = out
        return out
    mod.forward = forward
    act0 = arm.act

    def act(batch, pos_off, autocast, first):
        last.clear()
        B = int(batch["input_ids"].shape[0])
        env, eps, steps = env_view(B)
        if mode == "on":
            use, ks_, _rows = bank.select(env, eps, steps)
            mem_in = k27m.mem_input(torch, bank, env, eps, steps)
        else:
            use, ks_, mem_in = [False] * B, [0] * B, None
        last["mem_in"] = mem_in
        res = act0(batch, pos_off, autocast, first)
        out = last.get("out")
        c = mod.counters
        prob = []
        if out is None:
            prob.append("нет выхода модели")
        elif c["vision"] != 1 or any(x != 1 for x in c["text"]) or \
                c["active"] != [4] * 8 + [12] * 8 + [28] * 8:
            prob.append(f"счётчики {c}")
        want = ["root", "mid"] if (mem_in is not None and
                                   "alive_mid" in (out.get("mem") or {})) \
            else (["root"] if mem_in is not None else [])
        if out is not None and c["mem_reads"] != want:
            prob.append(f"чтения памяти {c['mem_reads']} ≠ {want}")
        if any(u and k != kmem.K_EXEC for u, k in zip(use, ks_)):
            prob.append(f"исполнено шагов {ks_} (ожидалось {kmem.K_EXEC})")
        if mode == "off" and out is not None and "mem" in out:
            prob.append("memory off использовал память")
        if prob:
            raise SystemExit(f"рука {label}: " + "; ".join(prob))
        keep = debug and mode == "on" and mem_in is not None and \
            not st["dbg_done"]
        rec = journal_record(out, B, env, dict(episode=eps, steps=steps,
                                               k=ks_), use, keep, mem_in)
        if keep:
            st["dbg_done"] = True
        arm.log.add(**rec)
        if mode == "on":
            bank.update(env, eps, steps, kmem.split_snapshot(out["snapshot"]))
        st["calls"] += 1
        return res
    arm.act = act
    return arm


def build_arm(device, basis, label, torch, *, spec, plan_sha1, debug=False):
    import k21_main_policy as kmp
    mode, seed = parse_label(label)
    orig = kmp.load_policy
    kmp.load_policy = lambda torch_, label_, ckpt_, device_: load_policy(
        torch_, label, spec, device_)
    try:
        arm = kmp.build_arm(device, basis, label, torch,
                            ckpt=spec.get("ckpt") or spec["source"]["ckpt"],
                            plan_sha1=plan_sha1)
    finally:
        kmp.load_policy = orig
    wrap_arm(arm, torch, mode, label, debug)
    arm.meta.update(arch="parents_mem24", memory=mode, train_seed=seed,
                    memory_version=kmem.VERSION, memory_debug=bool(debug),
                    k27_policy_sha1=sha(os.path.abspath(__file__)))
    return arm


def selftest():
    import copy
    import torch
    import k21_model as km
    assert parse_label("parents_mem24_s1") == ("on", 1)
    assert parse_label("parents_mem24off_s0") == ("off", 0)
    spec = dict(memory="on", seed=1, mem_hash="h")
    cfg = dict(mode=k27m.MODE, seed=1, memory=kmem.CONFIG)
    assert spec_problems("parents_mem24_s1", spec, cfg, "h") == []
    assert spec_problems("parents_mem24_s1", spec, dict(cfg, seed=0), "h")
    assert spec_problems("parents_mem24_s1", spec, cfg, "x")
    assert spec_problems("parents_mem24off_s1", spec, cfg, "h")
    # учёт сред: reset -> новые эпизоды, step -> шаги
    class FakeEnvs:
        def reset(self, options=None):
            return dict(state=np.zeros((len(options), 8)))

        def step(self, a):
            return None

        n = 5
    te = TrackedEnvs(FakeEnvs())
    te.reset(options=[{}] * 5)
    e0 = TRACK["episode"].copy()
    for _ in range(10):
        te.step(None)
    assert TRACK["steps"].tolist() == [10] * 5 and te.n == 5
    te.reset(options=[{}] * 5)
    assert not set(e0) & set(TRACK["episode"].tolist())
    assert TRACK["steps"].tolist() == [0] * 5
    # журнал игрушечной модели: формы постоянны с памятью и без
    M = k27m.build(torch)
    L = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    U = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    m = M["K27"](copy.deepcopy(km.fake_bar(torch)), L, U, seed=0).eval()
    bt = km.fake_batch(torch, B=3)
    st_ = torch.randn(3, 11)
    bank = kmem.MemoryBank()
    shapes = None
    for i, step_ in enumerate((10, 18, 26)):
        mi = k27m.mem_input(torch, bank, [0, 1, 2], [7, 7, 8], [step_] * 3)
        use, ks_, _r = bank.select([0, 1, 2], [7, 7, 8], [step_] * 3)
        with torch.no_grad():
            o = m(bt, 1, st_, mi)
        rec = journal_record(o, 3, [0, 1, 2], dict(episode=[7, 7, 8],
                                                   steps=[step_] * 3, k=ks_),
                             use, keep_debug=i == 1, mem_in=mi)
        sh = {k: np.asarray(v).shape for k, v in rec.items()
              if "dbg" not in k}
        assert shapes is None or sh == shapes, (sh, shapes)
        shapes = sh
        if i == 1:
            assert rec["mem_k"].tolist() == [8, 8, 8] and \
                "mem_dbg_S0" in rec
        bank.update([0, 1, 2], [7, 7, 8], [step_] * 3,
                    kmem.split_snapshot(o["snapshot"]))
    # настоящая обёртка руки: три вызова на блок для off и on; журнал
    # проходит проверку протокола (off — без памяти и без пополнения банка)
    import types
    import k21_policy as kp21
    import k27_protocol as pc
    for mode_ in ("off", "on"):
        lab = f"parents_mem24{'off' if mode_ == 'off' else ''}_s0"
        mm = M["K27"](copy.deepcopy(km.fake_bar(torch)), L, U, seed=0,
                      memory=mode_).eval()
        if mode_ == "on":
            with torch.no_grad():
                mm.mem.Wo.weight.normal_(0, 0.5, generator=torch.Generator()
                                         .manual_seed(2))
        log = kp21.K21Log(lab)
        calls = dict(n=0)

        def act0(batch, pos_off, autocast, first, _m=mm, _log=log, _c=calls):
            _c["n"] += 1
            with torch.no_grad():
                o = _m(batch, int(pos_off), torch.randn(
                    5, 11, generator=torch.Generator().manual_seed(_c["n"])))
            g_ = torch.where(o["grip_logit"] > 0, 1.0, -1.0)
            a_ = torch.cat([o["arm"], g_[..., None]], -1).numpy()
            _log.add(act=a_[:, :8], chunk=_c["n"], root=o["root"].numpy())
            return a_, None
        arm = types.SimpleNamespace(k21=mm, act=act0, log=log, meta={})
        wrap_arm(arm, torch, mode_, lab, debug=True)
        te = TrackedEnvs(FakeEnvs())
        te.reset(options=[{}] * 5)
        for _w in range(10):
            te.step(None)
        for i in range(3):
            arm.act(km.fake_batch(torch, B=5, seed=i), 1, None, i == 0)
            for _s in range(8):
                te.step(None)
        _summ, z = log.take()
        pb = pc.mem_problems(z, lab, mode_, debug=mode_ == "on")
        assert pb == [], (mode_, pb)
        if mode_ == "off":
            assert (z["k21_mem_use"] == 0).all() and \
                (z["k21_mem_k"] == -1).all()
        else:
            assert (z["k21_mem_use"][1:] == 1).all() and \
                (z["k21_mem_k"][1:] == 8).all()
    print("самопроверка k27_policy пройдена: метки on/off, сверка "
          "спецификации и контракта памяти, учёт сред (reset -> новые "
          "эпизоды и нулевые шаги, step -> шаги), журнал постоянной формы с "
          "памятью и без, debug-фрагмент; настоящая обёртка руки — 3 вызова "
          "off (память не используется, банк не пополняется) и on — журнал "
          "проходит проверку протокола")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль руки; запуск — через k27_harness.py")
