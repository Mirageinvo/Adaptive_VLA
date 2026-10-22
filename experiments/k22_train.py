#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-22: real-check, fitcheck и обучение parents24 тем же k21_train.

Сопоставимость с K-21: те же данные и подготовка (prep_id), учитель,
потеря, оптимизатор, расписание индексов, бюджет (warmup 1000 + 8000,
правило добора), выбор по val_sel и ускоренная загрузка k21_fastload.
Подмены в k21_train (файл не меняется): построитель модели -> K22, каталог
data/k22/runs, отчёты reports/k22/, в контракт добавлены SHA k22_model.py и
этого файла.
"""
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

REDIRECT = {"reports/k21/train.json": "reports/k22/train.json",
            "reports/k21/fitcheck.json": "reports/k22/fitcheck.json"}


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def _open(path, *a, **k):
    n = os.path.normpath(str(path))
    for src, dst in REDIRECT.items():
        if n == os.path.normpath(src) and a and "w" in a[0]:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            path = dst
    return builtins.open(path, *a, **k)


def make_model(torch, mode, D, dev, pin, checkpoint=True):
    # fitcheck k21_train строит «tree_adapt24» — здесь это K-22
    if mode not in ("parents24", "tree_adapt24"):
        raise SystemExit(f"K-22 не обучает режим {mode!r}")
    import k21_data as d21
    import k22_model as k22
    bar = d21.load_bar(dev, pin)
    m = bar["model"]
    m.action_expert.float()
    M = k22.build(torch)
    mod = M["K22"](m, D["L"], D["U"], D["state_mean"], D["state_std"],
                   checkpoint=checkpoint, seed=kt.SEED).to(dev)
    return mod, bar


def install():
    orig_contract = kt.run_contract

    def run_contract(mode, a, D):
        c = orig_contract(mode, a, D)
        c["architecture"] = "k22_parents24"
        c["code"] = dict(c["code"], **{f: sha(os.path.join(HERE, f)) for f in
                                       ("k22_model.py", "k22_train.py",
                                        "k21_fastload.py")})
        return c
    kt.run_contract = run_contract
    kt.make_model = make_model
    kt.OUTDIR = "data/k22/runs"
    kt.open = _open
    import k21_fastload as fl
    return fl.install()


def real_check(a):
    """Настоящий BAR: forward/backward + шаг AdamW (lr 0); счётчики
    4/12/28, конечность, пределы, ненулевые градиенты; код 3 при отказе."""
    import torch
    import k21_data as d21
    import k21_model as km
    import k22_model as k22
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    mod, bar = make_model(torch, "parents24", D, dev,
                          D["prep_report"]["bar_pin"])
    print(f"  K-22: {json.dumps(mod.report(), ensure_ascii=False)}",
          flush=True)
    opt = torch.optim.AdamW([p for g in mod.groups().values() for p in g],
                            lr=0.0)
    rows = D["train_rows"][:: max(1, len(D["train_rows"]) // 64)][:64]
    prob, sizes = [], {}
    for mb in a.microbatches:
        batch, po, st, tgt, val = d21.make_batch(D, rows[:mb], bar, dev)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
        t0 = time.time()
        try:
            mod.train()
            with torch.autocast("cuda", dtype=torch.float16):
                out = mod(batch, po, st)
                loss, _l1, _b = km.loss_fn(torch, out, tgt, val)
            loss.mean().backward()
            opt.step()
            torch.cuda.synchronize(dev)
            ok = True
        except torch.cuda.OutOfMemoryError:
            ok = False
        r = dict(ok=ok, seconds=time.time() - t0,
                 peak_gb_with_adamw=torch.cuda.max_memory_allocated(dev)
                 / 2 ** 30)
        if ok:
            c = mod.counters
            gn = {k: float(np.sqrt(sum(float((p.grad.float() ** 2).sum())
                                       for p in v if p.grad is not None)))
                  for k, v in mod.groups().items()}
            Lt = torch.as_tensor(D["L"], device=dev)
            Ut = torch.as_tensor(D["U"], device=dev)
            checks = dict(
                counters=c["active"] == [4] * 8 + [12] * 8 + [28] * 8
                and c["vision"] == 1 and all(x == 1 for x in c["text"]),
                finite=bool(torch.isfinite(out["arm"]).all()
                            and torch.isfinite(loss).all()),
                in_bounds=bool(((out["arm"] >= Lt - 1e-4)
                                & (out["arm"] <= Ut + 1e-4)).all()),
                grads=all(v > 0 and np.isfinite(v) for v in gn.values()))
            r.update(checks=checks, grad_norms=gn, loss=float(loss.mean()))
            bad = [k for k, v in checks.items() if not v]
            if bad:
                prob.append(f"microbatch {mb}: {bad}")
        mod.zero_grad(set_to_none=True)
        sizes[mb] = r
        print(f"    microbatch {mb}: {r}", flush=True)
        if not ok:
            break
    if not any(v["ok"] for v in sizes.values()):
        prob.append("нет пригодного microbatch")
    os.makedirs("reports/k22", exist_ok=True)
    json.dump(dict(microbatch=sizes, problems=prob,
                   query_layer_ops=k22.query_layer_ops()),
              open("reports/k22/real_check.json", "w"), indent=1,
              default=float)
    print("  " + ("ОТКАЗ: " + "; ".join(prob) if prob else
                  "K-22 real-check пройден"), flush=True)
    return 3 if prob else 0


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cwd = os.getcwd()
        os.chdir(td)
        try:
            os.makedirs("reports/k21")
            h = _open("reports/k21/train.json", "w")
            h.write("x")
            h.close()
            assert os.path.exists("reports/k22/train.json")
            assert not os.path.exists("reports/k21/train.json")
        finally:
            os.chdir(cwd)
    print("самопроверка k22_train пройдена: перенаправление отчётов")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    if "real-check" in sys.argv:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--device", default="cuda:0")
        ap.add_argument("--microbatches", type=int, nargs="+",
                        default=[8, 32])
        sys.exit(real_check(ap.parse_args()))
    if "--modes" in sys.argv:
        i = sys.argv.index("--modes")
        if sys.argv[i + 1:i + 2] != ["parents24"] or (
                len(sys.argv) > i + 2 and not sys.argv[i + 2].startswith("--")):
            raise SystemExit("этот запускатель обучает только parents24")
    else:
        sys.argv += ["--modes", "parents24"]
    st = install()
    code = kt.main()
    pf = st["pf"]
    if pf is not None:
        print(f"  fastload: из предзагрузки {pf.hits}, синхронно "
              f"{pf.misses}", flush=True)
    sys.exit(code)
