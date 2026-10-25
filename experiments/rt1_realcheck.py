#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, этап 1: real-check пары flat24 / parents24 на настоящем BAR.

Для каждой архитектуры (тот же seed, BAR загружается заново): bf16 autocast,
checkpointing, microbatch 1/2/4 на одних и тех же обучающих строках разных
задач — forward, потеря, backward, clip 1.0 и ПОЛНЫЙ шаг AdamW с
настоящими LR групп K-21 (эксперт 2e-5, LoRA 1e-5, новые 1e-4; wd 0.01).
Проверки: счётчики (vision 1, слои по 1, активные query), конечность,
суставы в пределах робота и схваты в [0, 1], градиенты по группам (без
градиента — только структурно неиспользуемые параметры, k23_smoke),
пик памяти с состоянием оптимизатора; общие начальные тензоры пары равны.
Итог — наибольший microbatch, прошедший у ОБЕИХ моделей (эффективный батч
32 набирается накоплением). Отчёт reports/rt1/realcheck.json; код 3 при
технической проблеме или если ни один microbatch не прошёл у обеих.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

LR = dict(expert=2e-5, lora=1e-5, new=1e-4)
WD, CLIP, EFF_BATCH = 0.01, 1.0, 32
OUT = "reports/rt1/realcheck.json"


def pick_rows(D, n):
    """По строке из разных задач и эпизодов, детерминированно."""
    tr = D.rows("train")
    task = D.ix["task"][tr]
    out, rng = [], np.random.default_rng(20261025)
    for k in range(n):
        cand = tr[task == (k % len(np.unique(task)))]
        out.append(int(rng.choice(cand)))
    return out


def main():
    ap = argparse.ArgumentParser(description="RoboTwin: real-check пары")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--micros", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    import k21_train as ktr
    import k23_smoke as sm
    import k23_train as k23t
    import rt1_common as rc
    import rt1_data as rd
    import rt1_model as rm
    dev = torch.device(a.device)
    dt = getattr(torch, a.dtype)
    D = rd.Data()
    M = rm.build(torch)
    rows = pick_rows(D, max(a.micros))
    free0, total = torch.cuda.mem_get_info(dev)
    rep = dict(device=torch.cuda.get_device_name(dev), dtype=a.dtype,
               seed=a.seed, prep_id=D.prep["prep_id"], rows=rows,
               free_gb_at_start=free0 / 2 ** 30, total_gb=total / 2 ** 30,
               archs={})
    prob, states = [], {}
    want = dict(flat24=[16] * 24, parents24=[4] * 8 + [12] * 8 + [28] * 8)
    for arch in rm.ARCHS:
        bar = rc.load_bar(dev, a.dtype)
        rep["bar_revision"] = bar["revision"]
        m = bar["model"]
        m.action_expert.float()
        mod = M[arch](m, D.L, D.U, checkpoint=True, seed=a.seed).to(dev)
        states[arch] = ktr.trainable_state(mod)
        gr = mod.groups()
        gn = sm.groups_named(mod)
        named = [(n, p) for n, p in mod.named_parameters() if p.requires_grad]
        opt = torch.optim.AdamW([dict(params=gr[k], lr=LR[k]) for k in LR],
                                weight_decay=WD)
        last = len(m.vlm.text_model.layers) - 1
        res = {}
        for mb in a.micros:
            r = dict(ok=False)
            try:
                batch, po, st, tgt, val = D.make_batch(rows[:mb], bar, dev, dt)
                r["prefix_len"] = int(batch["input_ids"].shape[1])
                r["pixel_values"] = list(batch["pixel_values"].shape)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
                torch.cuda.synchronize(dev)
                t0 = time.time()
                mod.train()
                with torch.autocast("cuda", dtype=dt):
                    out = mod(batch, po, st)
                    loss, la, lg = rm.loss_fn(torch, out, tgt, val)
                loss.mean().backward()
                gp, none, norms = sm.grad_problems(named, gn, last)
                torch.nn.utils.clip_grad_norm_([p for _n, p in named], CLIP)
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(dev)
                c = mod.counters
                arm, grip = out["arm"].float(), out["grip"].float()
                Lt = torch.as_tensor(D.L[list(rc.ARM_IDX)], device=dev)
                Ut = torch.as_tensor(D.U[list(rc.ARM_IDX)], device=dev)
                checks = dict(
                    counters=c["active"] == want[arch] and c["vision"] == 1
                    and all(x == 1 for x in c["text"]),
                    finite=bool(torch.isfinite(out["action"]).all()
                                and torch.isfinite(loss).all()),
                    bounds=bool(((arm >= Lt - 1e-4) & (arm <= Ut + 1e-4)).all()
                                and (grip >= 0).all() and (grip <= 1).all()),
                    grads=not gp)
                r.update(ok=all(checks.values()), checks=checks,
                         seconds=time.time() - t0,
                         peak_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                         loss=float(loss.mean()), l1_arm=float(la.mean()),
                         l1_grip=float(lg.mean()), group_grad_norms=norms,
                         no_grad_params=none, grad_problems=gp)
                bad = [k for k, v in checks.items() if not v]
                if bad:
                    prob.append(f"{arch} micro {mb}: {bad}")
            except torch.cuda.OutOfMemoryError:
                r.update(ok=False, oom=True)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            res[str(mb)] = r
            print(f"  {arch} micro {mb}: " + (
                f"пик {r['peak_gb']:.1f} ГБ, {r['seconds']:.2f} с, префикс "
                f"{r.get('prefix_len')}, потеря {r['loss']:.4f}, проверки "
                f"{r['checks']}" if "peak_gb" in r else "НЕ ПОМЕСТИЛСЯ (OOM)"),
                flush=True)
            if r.get("oom"):
                break
        rep["archs"][arch] = res
        del mod, bar, m, opt
        torch.cuda.empty_cache()
    cp = k23t.init_compare(states["flat24"], states["parents24"])
    prob += k23t.pair_problems(cp)
    rep["pair_compare"] = {k: len(v) for k, v in cp.items()}
    ok_both = [mb for mb in a.micros
               if all(rep["archs"][x].get(str(mb), {}).get("ok")
                      for x in rm.ARCHS)]
    common = max(ok_both) if ok_both else None
    rep["common_micro"] = common
    rep["accumulation"] = EFF_BATCH // common if common and \
        EFF_BATCH % common == 0 else None
    if common is None:
        prob.append("ни один microbatch не прошёл у обеих моделей")
    rep["problems"] = prob
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rep, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=float)
    print(f"  пара: {rep['pair_compare']}; общий micro {common} "
          f"(накопление {rep['accumulation']}); свободно в начале "
          f"{rep['free_gb_at_start']:.1f} ГБ")
    print("  " + ("ОТКАЗ: " + "; ".join(prob[:6]) if prob else
                  "real-check RoboTwin пройден") + f"; отчёт {OUT}")
    return 3 if prob else 0


if __name__ == "__main__":
    sys.exit(main())
