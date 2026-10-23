#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-23, пакет B: независимый повтор обучения flat24 и parents24 с явным
training seed (тот же k21_train; файлы K-21/K-22 не меняются).

Что делает training seed S (и только он):
  - инициализация собственных модулей: конструкторы K21/K22 получают seed=S
    (все случайные тензоры — LoRA A, slot, state_proj, readout, расширения
    дерева — из локальных генераторов от seed; нулевые B, головы границ,
    q0 = BOS и предобученный эксперт от seed не зависят);
  - очередь обучающих строк: k21_train.schedule вызывается с seed=S.
    В k21_train default seed=SEED захвачен при определении, а обёртка
    k21_fastload передаёт свой захваченный default; поэтому здесь schedule
    заменяется ДО установки fastload функцией, которая всегда использует S
    (её же получает предзагрузчик для построения очереди наперёд);
  - контракт запуска: seed, архитектура, отпечаток очереди, SHA кода K-23.
k21_train.SEED (= 0) НЕ меняется: от него зависит только выбор диагностической
train-панели (SEED + 5) — она остаётся той же, что у K-21/K-22, для
сравнимости кривых. Подготовка, учитель, train/val и 5120 учительских
наблюдений общие для всех сидов.

Режимы:
  smoke --seed S        — настоящий BAR: воспроизводимость инициализации,
                          равенство общих тензоров flat/parents, отличие от
                          seed 0, forward/backward + AdamW (lr 0) на micro 32,
                          счётчики; отдельный временный запуск, обучение
                          потом создаётся заново из той же инициализации;
  train --seed S        — пара (flat24, parents24) в data/k23/runs/s<S>/:
                          обе до 8000, общее правило добора, при срабатывании
                          обе до 16000 (k21_train.main_train); fastload;
                          счётчик шагов и пропусков GradScaler;
  --selftest            — игрушечная модель: непрерывное обучение = остановка
                          + возобновление, семантика seed, очередь, отчёты.
Разрешён только seed 1 (единственный новый повтор плана K-23).
"""
import argparse
import builtins
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k21_train as kt  # noqa: E402

ALLOWED_SEEDS = (1,)
MODES = ("flat24", "parents24")
ARCH = dict(flat24="k21_flat24", parents24="k22_parents24")
CODE_EXTRA = dict(flat24=("k23_train.py", "k21_fastload.py"),
                  parents24=("k23_train.py", "k21_fastload.py",
                             "k22_model.py"))
MAX_CONSEC_SKIPS = 50
# точки очереди для отпечатка: начало, переход warmup/main, конец бюджета 8000
# и максимального 16000 (индексы обновлений k21_train.schedule)
FP_POINTS = (0, 1, 2, 998, 999, 1000, 1001, 1002, 8998, 8999, 9000, 16999)


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def outdir(seed):
    return f"data/k23/runs/s{seed}"


def report_path(seed):
    return f"reports/k23/train_s{seed}.json"


# --- seed ---------------------------------------------------------------------
def seeded_schedule(orig, seed):
    """Очередь всегда с training seed (переданный seed игнорируется: это
    захваченный default 0 из k21_train/k21_fastload)."""
    train_seed = seed

    def wrapper(D, n_updates, micro, seed=None, start=0):
        return orig(D, n_updates, micro, seed=train_seed, start=start)
    wrapper.k23_seed = train_seed
    return wrapper


def schedule_fp(sched, D, micro, points=FP_POINTS):
    h = hashlib.sha1()
    for u in points:
        src, idx = sched(D, u + 1, micro, start=u)[0]
        h.update(src.encode())
        h.update(np.ascontiguousarray(np.asarray(idx, np.int64)).tobytes())
    return h.hexdigest()


def build_module(torch, mode, bar_model, D, seed, checkpoint=True):
    import k21_model as km
    import k22_model as k22
    if mode == "flat24":
        return km.build(torch)["K21"](
            bar_model, "flat24", D["L"], D["U"], D["state_mean"],
            D["state_std"], checkpoint=checkpoint, seed=seed)
    if mode == "parents24":
        return k22.build(torch)["K22"](
            bar_model, D["L"], D["U"], D["state_mean"], D["state_std"],
            checkpoint=checkpoint, seed=seed)
    raise SystemExit(f"K-23 не обучает режим {mode!r}")


def make_model_fn(seed):
    def make_model(torch, mode, D, dev, pin, checkpoint=True):
        import k21_data as d21
        bar = d21.load_bar(dev, pin)
        m = bar["model"]
        m.action_expert.float()
        mod = build_module(torch, mode, m, D, seed, checkpoint).to(dev)
        return mod, bar
    return make_model


# --- GradScaler: число шагов и пропусков ------------------------------------
def counting_scaler(base):
    class CountingScaler(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._k23 = dict(steps=0, skips=0, consecutive_skips=0)
            self._before = None

        def step(self, opt, *a, **k):
            self._before = self.get_scale() if self.is_enabled() else None
            return super().step(opt, *a, **k)

        def update(self, *a, **k):
            super().update(*a, **k)
            c = self._k23
            if self._before is not None and self.get_scale() < self._before:
                c["skips"] += 1
                c["consecutive_skips"] += 1
                if c["consecutive_skips"] >= MAX_CONSEC_SKIPS:
                    raise SystemExit(f"GradScaler: {MAX_CONSEC_SKIPS} "
                                     f"пропусков подряд — обновления не "
                                     f"выполняются")
            else:
                c["steps"] += 1
                c["consecutive_skips"] = 0
            self._before = None

        def state_dict(self):
            d = dict(super().state_dict())
            d["_k23"] = dict(self._k23)
            return d

        def load_state_dict(self, d):
            d = dict(d)
            k = d.pop("_k23", None)
            if d:
                super().load_state_dict(d)
            if k is not None:
                self._k23 = dict(k)
    return CountingScaler


# --- установка подмен ---------------------------------------------------------
def install(seed, torch=None, fastload=True):
    if seed not in ALLOWED_SEEDS:
        raise SystemExit(f"training seed {seed}: в K-23 разрешён только "
                         f"{ALLOWED_SEEDS} (seed 0 — исторический запуск, "
                         f"третьего сида нет)")
    sched = seeded_schedule(kt.schedule, seed)
    kt.schedule = sched                          # до fastload: его захватит
    kt.make_model = make_model_fn(seed)
    kt.OUTDIR = outdir(seed)
    redirect = {"reports/k21/train.json": report_path(seed)}

    def _open(path, *a, **k):
        n = os.path.normpath(str(path))
        for src, dst in redirect.items():
            if n == os.path.normpath(src) and a and "w" in a[0]:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                path = dst
        return builtins.open(path, *a, **k)
    kt.open = _open
    orig_contract = kt.run_contract
    fp_cache = {}

    def run_contract(mode, a, D):
        if mode not in MODES:
            raise SystemExit(f"K-23 не обучает режим {mode!r}")
        c = orig_contract(mode, a, D)
        if a.micro not in fp_cache:
            fp_cache[a.micro] = schedule_fp(sched, D, a.micro)
        c.update(seed=seed, architecture=ARCH[mode], k23=dict(
            train_seed=seed, schedule_fp=fp_cache[a.micro],
            panel="train-панель K-21 (k21_train.SEED + 5), не зависит от "
                  "training seed"))
        c["code"] = dict(c["code"], **{f: sha(os.path.join(HERE, f))
                                       for f in CODE_EXTRA[mode]})
        return c
    kt.run_contract = run_contract
    orig_check = kt.Trainer.check

    def check(self, opt, u):
        orig_check(self, opt, u)
        sc = self.scaler
        c = dict(getattr(sc, "_k23", {}))
        c["scale"] = float(sc.get_scale()) if sc.is_enabled() else None
        self.st["curve"][-1]["scaler"] = c
        print(f"    {self.tag}: шагов оптимизатора {c.get('steps')}, "
              f"пропусков GradScaler {c.get('skips')}, масштаб "
              f"{c['scale']}", flush=True)
    kt.Trainer.check = check
    if torch is not None:
        torch.cuda.amp.GradScaler = counting_scaler(
            torch.cuda.amp.GradScaler)
    st = None
    if fastload:
        import k21_fastload as fl
        st = fl.install()
    return dict(fastload=st, schedule=sched)


def finalize_report(torch, seed):
    """Дописать в собственный отчёт K-23: seed, отпечатки, счётчики."""
    p = report_path(seed)
    rep = json.load(open(p))
    for m in MODES:
        st = torch.load(os.path.join(outdir(seed), m, "resume.pt"),
                        map_location="cpu", weights_only=False)
        sc = (st.get("scaler") or {}).get("_k23")
        rep[m].update(train_seed=seed, architecture=ARCH[m],
                      init_hash=st["init_hash"], scaler=sc,
                      schedule_fp=st["config"]["k23"]["schedule_fp"],
                      budget_main_updates=st["u"])
    fps = {rep[m]["schedule_fp"] for m in MODES}
    rep["pair"] = dict(seed=seed, same_schedule=len(fps) == 1,
                       extension=rep[MODES[0]]["extension"])
    json.dump(rep, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"  отчёт дополнен: {p}; очередь пары "
          f"{'общая' if len(fps) == 1 else 'РАЗНАЯ'}", flush=True)
    return 0 if len(fps) == 1 else 3


# --- сравнения начальных весов -------------------------------------------------
def init_compare(st_a, st_b):
    """Общие ключи равной формы: равны/различны; ключи разной формы."""
    common = sorted(set(st_a) & set(st_b))
    same, diff, shape = [], [], []
    for k in common:
        a, b = st_a[k], st_b[k]
        if tuple(a.shape) != tuple(b.shape):
            shape.append(k)
        elif bool((a == b).all()):
            same.append(k)
        else:
            diff.append(k)
    return dict(same=same, diff=diff, shape=shape,
                only_a=sorted(set(st_a) - set(st_b)),
                only_b=sorted(set(st_b) - set(st_a)))


RANDOM_KEYS = ("slot", "state_proj.weight", ".A", "readout.mlp.0.weight")


def seed_change_problems(cmp_s_vs_0):
    """При смене seed обязаны отличаться случайные тензоры; предобученные и
    намеренно нулевые — нет."""
    p = []
    for key in RANDOM_KEYS:
        hit = [k for k in cmp_s_vs_0["same"] if k.endswith(key) or
               (key.startswith(".") and k.endswith(key))]
        if hit:
            p.append(f"при смене seed не изменились {hit[:3]}")
    bad = [k for k in cmp_s_vs_0["diff"] if k.startswith(
        "bar.action_expert.") or k.endswith(".B") or k.endswith("q0")]
    if bad:
        p.append(f"от seed зависят предобученные/нулевые тензоры {bad[:3]}")
    return p


def pair_problems(cmp_flat_parents):
    p = []
    if cmp_flat_parents["diff"]:
        p.append(f"общие тензоры flat/parents различаются: "
                 f"{cmp_flat_parents['diff'][:4]}")
    exp_shape = {"q0", "slot"}
    other = [k for k in cmp_flat_parents["shape"] if k not in exp_shape]
    if other:
        p.append(f"неожиданно разная форма: {other[:4]}")
    return p


# --- smoke на настоящем BAR ---------------------------------------------------
def smoke(a):
    import torch
    import k21_data as d21
    import k21_model as km
    seed = a.seed
    if seed not in ALLOWED_SEEDS:
        raise SystemExit(f"training seed {seed} не разрешён")
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=True)
    pin = D["prep_report"]["bar_pin"]
    prob, rep = [], dict(seed=seed)

    def built(mode, s):
        bar = d21.load_bar(dev, pin)
        m = bar["model"]
        m.action_expert.float()
        mod = build_module(torch, mode, m, D, s).to(dev)
        return mod, bar

    states = {}
    for tag, mode, s in (("flat_s", "flat24", seed), ("flat_s_again",
                                                      "flat24", seed),
                         ("flat_0", "flat24", 0),
                         ("parents_s", "parents24", seed)):
        mod, bar = built(mode, s)
        states[tag] = kt.trainable_state(mod)
        rep[f"init_hash_{tag}"] = kt.state_hash(states[tag])
        if tag in ("flat_s", "parents_s"):
            opt = torch.optim.AdamW([p for g in mod.groups().values()
                                     for p in g], lr=0.0)
            rows = D["train_rows"][:: max(1, len(D["train_rows"]) // 64)]
            batch, po, st_, tgt, val = d21.make_batch(
                D, rows[:a.micro], bar, dev)
            torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.time()
            mod.train()
            with torch.autocast("cuda", dtype=torch.float16):
                out = mod(batch, po, st_)
                loss, _l1, _b = km.loss_fn(torch, out, tgt, val)
            loss.mean().backward()
            opt.step()
            torch.cuda.synchronize(dev)
            c = mod.counters
            want = [16] * 24 if mode == "flat24" else \
                [4] * 8 + [12] * 8 + [28] * 8
            ok = dict(
                counters=c["active"] == want and c["vision"] == 1
                and all(x == 1 for x in c["text"]),
                finite=bool(torch.isfinite(out["arm"]).all()
                            and torch.isfinite(loss).all()),
                grads=all(p.grad is not None and bool(
                    torch.isfinite(p.grad).all()) for g in
                    mod.groups().values() for p in g if p.requires_grad))
            rep[f"step_{tag}"] = dict(ok, seconds=time.time() - t0,
                                      peak_gb=torch.cuda.max_memory_allocated(
                                          dev) / 2 ** 30,
                                      loss=float(loss.mean()))
            bad = [k for k, v in ok.items() if not v]
            if bad:
                prob.append(f"{tag}: {bad}")
            del opt
        del mod, bar
        torch.cuda.empty_cache()
    if rep["init_hash_flat_s"] != rep["init_hash_flat_s_again"]:
        prob.append("повторное построение с тем же seed даёт другие веса")
    cp = init_compare(states["flat_s"], states["parents_s"])
    prob += pair_problems(cp)
    c0 = init_compare(states["flat_s"], states["flat_0"])
    prob += seed_change_problems(c0)
    rep["pair_compare"] = {k: len(v) for k, v in cp.items()}
    rep["seed_compare"] = {k: len(v) for k, v in c0.items()}
    rep["seed_changed_keys"] = c0["diff"][:40]
    # seed 0 против исторической инициализации (справочно, не гейт)
    hist = {}
    for mode, path in (("flat24", "data/k21/runs/flat24/resume.pt"),):
        if os.path.exists(path):
            h = torch.load(path, map_location="cpu", weights_only=False)
            hist[mode] = dict(historical=h["init_hash"],
                              rebuilt_seed0=rep["init_hash_flat_0"],
                              equal=h["init_hash"] == rep["init_hash_flat_0"])
    rep["historical_seed0"] = hist
    # очередь: seed S против исходной функции с тем же seed; seed 0 = старая
    orig = kt.schedule
    s_fn = seeded_schedule(orig, seed)
    fp_s = schedule_fp(s_fn, D, a.micro)
    fp_direct = schedule_fp(lambda D_, n, m, start=0: orig(
        D_, n, m, seed=seed, start=start), D, a.micro)
    fp_0 = schedule_fp(seeded_schedule(orig, 0), D, a.micro)
    fp_hist = schedule_fp(lambda D_, n, m, start=0: orig(D_, n, m,
                                                          start=start),
                          D, a.micro)
    rep["schedule"] = dict(fp_seed=fp_s, fp_seed0=fp_0, fp_hist=fp_hist)
    if fp_s != fp_direct:
        prob.append("очередь seed S ≠ исходной функции с seed S")
    if fp_0 != fp_hist:
        prob.append("seed 0 не воспроизводит историческую очередь")
    if fp_s == fp_0:
        prob.append("очередь не зависит от seed")
    rep["problems"] = prob
    os.makedirs("reports/k23", exist_ok=True)
    out = f"reports/k23/smoke_s{seed}.json"
    json.dump(rep, open(out, "w"), ensure_ascii=False, indent=1,
              default=float)
    print(json.dumps({k: v for k, v in rep.items() if k not in (
        "seed_changed_keys",)}, ensure_ascii=False, indent=1, default=float))
    print("  " + ("ОТКАЗ: " + "; ".join(prob) if prob else
                  f"smoke K-23 seed {seed} пройден") + f"; отчёт {out}")
    return 3 if prob else 0


# --- самопроверка (игрушечная модель) ----------------------------------------
def selftest():
    import copy
    import tempfile
    import types
    import torch
    import k21_data as d21
    import k21_model as km
    # 1) семантика seed и пары на игрушечном BAR
    bar0 = km.fake_bar(torch)
    L, U = -np.ones(6) * 1.1, np.ones(6) * 4.0
    D0 = dict(L=L, U=U, state_mean=None, state_std=None)

    def st(mode, s):
        return kt.trainable_state(build_module(
            torch, mode, copy.deepcopy(bar0), D0, s, checkpoint=False))
    f1, f1b, f0, p1 = st("flat24", 1), st("flat24", 1), st("flat24", 0), \
        st("parents24", 1)
    assert kt.state_hash(f1) == kt.state_hash(f1b)
    assert pair_problems(init_compare(f1, p1)) == []
    assert seed_change_problems(init_compare(f1, f0)) == [], \
        seed_change_problems(init_compare(f1, f0))
    assert seed_change_problems(init_compare(f1, f1b))      # без смены — ловит
    # 2) очередь: всегда training seed, seed 0 = историческая
    rng = np.random.default_rng(0)
    D = dict(train_rows=np.arange(400), val_rows=np.arange(400, 440),
             teacher_rows=np.arange(400, 420),
             task=rng.integers(0, 4, 440), episode=rng.integers(0, 20, 440))
    orig = kt.schedule
    s1 = seeded_schedule(orig, 1)
    assert np.array_equal(s1(D, 5, 8, seed=0, start=4)[0][1],
                          orig(D, 5, 8, seed=1, start=4)[0][1])
    assert schedule_fp(seeded_schedule(orig, 0), D, 8, (0, 3, 7)) == \
        schedule_fp(lambda D_, n, m, start=0: orig(D_, n, m, start=start),
                    D, 8, (0, 3, 7))
    assert schedule_fp(s1, D, 8, (0, 3, 7)) != \
        schedule_fp(seeded_schedule(orig, 0), D, 8, (0, 3, 7))
    # 3) счётчик GradScaler: пропуск = уменьшение масштаба
    class FakeScaler:
        def __init__(self, enabled=True):
            self.s, self.inf = 8.0, False

        def is_enabled(self):
            return True

        def get_scale(self):
            return self.s

        def step(self, opt):
            return None

        def update(self):
            self.s = self.s / 2 if self.inf else self.s

        def state_dict(self):
            return dict(scale=self.s)

        def load_state_dict(self, d):
            self.s = d["scale"]
    CS = counting_scaler(FakeScaler)
    sc = CS()
    sc.step(None)
    sc.update()
    sc.inf = True
    sc.step(None)
    sc.update()
    assert sc._k23["steps"] == 1 and sc._k23["skips"] == 1
    sc2 = CS()
    sc2.load_state_dict(sc.state_dict())
    assert sc2._k23 == sc._k23 and sc2.s == 4.0
    # 4) непрерывное обучение = остановка + возобновление (настоящий Trainer)
    saved = {k: getattr(kt, k) for k in (
        "WARM_UPDATES", "MAIN_UPDATES", "EVAL_EVERY", "SAVE_EVERY",
        "PANEL_N", "EFF_BATCH", "OUTDIR", "make_model", "run_contract",
        "schedule")}
    saved_mb = d21.make_batch
    nr = 64
    DT = dict(train_rows=np.arange(48), val_rows=np.arange(48, 56),
              teacher_rows=np.arange(48, 56), task=np.arange(nr) % 4,
              episode=np.arange(nr) % 8, L=L, U=U, state_mean=None,
              state_std=None, prep_report=dict(bar_pin=None))

    def fake_mb(D_, rows, bar, dev, teacher_idx=None):
        rows = np.asarray(rows)
        b = km.fake_batch(torch, B=len(rows), seed=int(rows.sum()) % 1000)
        g = torch.Generator().manual_seed(int(rows.sum()) + 7)
        st_ = torch.randn(len(rows), 11, generator=g)
        tgt = torch.rand(len(rows), 20, 7, generator=g) * 2
        tgt[..., 6] = torch.where(tgt[..., 6] > 1, 1.0, -1.0)
        return b, 1, st_, tgt, torch.ones(len(rows), 20, dtype=torch.bool)

    def fake_make_model(torch_, mode, D_, dev, pin, checkpoint=True):
        m = copy.deepcopy(bar0)
        return build_module(torch_, mode, m, D_, 1, checkpoint=False), \
            dict(model=m)
    try:
        kt.WARM_UPDATES, kt.MAIN_UPDATES = 2, 4
        kt.EVAL_EVERY, kt.SAVE_EVERY, kt.PANEL_N, kt.EFF_BATCH = 2, 1, 8, 8
        kt.make_model = fake_make_model
        kt.run_contract = lambda mode, a_, D_: dict(mode=mode, seed=1)
        kt.schedule = seeded_schedule(saved["schedule"], 1)
        d21.make_batch = fake_mb
        a_ = types.SimpleNamespace(device="cpu", dtype="float32", micro=4,
                                   eval_micro=4)
        res = {}
        with tempfile.TemporaryDirectory() as td:
            for name, stops in (("cont", (4,)), ("resume", (2, 4))):
                kt.OUTDIR = os.path.join(td, name)
                for until in stops:
                    tr = kt.Trainer(torch, a_, DT, "parents24", "p")
                    tr.run(until)
                res[name] = (kt.state_hash(kt.trainable_state(tr.mod)),
                             [(c["update"], round(c["val"]["L"][0], 10))
                              for c in tr.st["curve"]])
        assert res["cont"] == res["resume"], res
    finally:
        for k, v in saved.items():
            setattr(kt, k, v)
        d21.make_batch = saved_mb
    print("самопроверка k23_train пройдена: seed -> инициализация "
          "(повтор равен, общие тензоры пары равны, случайные меняются, "
          "предобученные/нулевые — нет), очередь всегда с training seed и "
          "seed 0 = историческая, счётчик GradScaler, непрерывное = "
          "остановка + возобновление (Trainer, игрушечная модель)")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-23: повтор обучения")
    ap.add_argument("cmd", choices=("smoke", "train"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--expect", choices=("new", "resume", "auto"),
                    default="auto", help="новый запуск или продолжение")
    a, rest = ap.parse_known_args()
    if a.cmd == "smoke":
        return smoke(a)
    import torch
    if a.seed not in ALLOWED_SEEDS:
        raise SystemExit(f"training seed {a.seed} не разрешён")
    have = [m for m in MODES if os.path.exists(os.path.join(
        outdir(a.seed), m, "resume.pt"))]
    if a.expect == "new" and have:
        raise SystemExit(f"--expect new, но есть состояние {have}")
    if a.expect == "resume" and not have:
        raise SystemExit("--expect resume, но состояния нет")
    print(f"  K-23 seed {a.seed}: " + (f"продолжение ({have})" if have
                                       else "новый запуск"), flush=True)
    st = install(a.seed, torch)
    sys.argv = [sys.argv[0], "train", "--device", a.device, "--micro",
                str(a.micro), "--dtype", a.dtype, "--modes", *MODES] + rest
    code = kt.main()
    pf = st["fastload"]["pf"] if st["fastload"] else None
    if pf is not None:
        print(f"  fastload: из предзагрузки {pf.hits}, синхронно "
              f"{pf.misses}", flush=True)
    if code == 0:
        code = finalize_report(torch, a.seed)
    return code


if __name__ == "__main__":
    sys.exit(main())
