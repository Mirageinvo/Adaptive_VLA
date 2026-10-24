#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-23: smoke тренера с исправленной проверкой градиентов.

k23_train.smoke требовал градиент у КАЖДОГО обучаемого параметра. Группа
expert — все параметры bar.action_expert, включая таблицу эмбеддингов
токенов (и выходную голову, если есть), которые однопроходная модель не
использует (на вход эксперта идут собственные query) — у них grad is None, и
проверка отказывала ложно (так же было в K-21/K-22: AdamW такие параметры
пропускает; real-check K-22 считал нормы групп, пропуская None).

Здесь, остальное — как в k23_train.smoke (тот же построитель, те же
сравнения инициализации и очереди):
  - все существующие градиенты конечны;
  - норма градиента каждой группы (expert, lora, new) > 0 и конечна;
  - параметры без градиента перечисляются; допустимы только структурно не
    влияющие на выход: эмбеддинги/выходная голова эксперта и LoRA o_proj
    последнего слоя VLM (unused_ok), иначе — отказ.
k23_train.py и k23_run.sh не меняются (закреплены в плане пакета A).
Отчёт: reports/k23/smoke_s<S>_grads.json (прежний smoke_s<S>.json
сохраняется как есть).
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k21_train as kt  # noqa: E402
import k23_train as t  # noqa: E402

UNUSED_OK = re.compile(r"^bar\.action_expert\.(embed_tokens|lm_head)\.")


def unused_ok(n, last_layer):
    """Структурно не влияющие на выход параметры: эмбеддинги/голова
    эксперта и LoRA o_proj ПОСЛЕДНЕГО слоя VLM (его выход обновляет только
    состояния префикса, которые дальше никуда не идут: query действий читают
    префикс через K/V, посчитанные из входа слоя)."""
    return bool(UNUSED_OK.match(n)) or n in (
        f"bar.vlm.text_model.layers.{last_layer}.self_attn.o_proj.A",
        f"bar.vlm.text_model.layers.{last_layer}.self_attn.o_proj.B")


def grad_problems(named, groups_named, last_layer=23):
    """named [(имя, параметр)] обучаемых; groups_named {группа: [имена]}."""
    p, none, norms = [], [], {}
    by = dict(named)
    for n, prm in named:
        if prm.grad is None:
            none.append(n)
        elif not bool(prm.grad.isfinite().all()):
            p.append(f"нечисловой градиент {n}")
    bad_none = [n for n in none if not unused_ok(n, last_layer)]
    if bad_none:
        p.append(f"нет градиента у используемых параметров: {bad_none[:6]}")
    for g, names in groups_named.items():
        gs = [by[n].grad.float() for n in names if by[n].grad is not None]
        v = float(np.sqrt(sum(float((x ** 2).sum()) for x in gs))) if gs \
            else 0.0
        norms[g] = v
        if not (v > 0 and np.isfinite(v)):
            p.append(f"норма градиента группы {g} = {v}")
    return p, none, norms


def groups_named(mod):
    ids = {id(p): n for n, p in mod.named_parameters()}
    return {g: [ids[id(p)] for p in v] for g, v in mod.groups().items()}


def smoke(a):
    import torch
    import k21_data as d21
    import k21_model as km
    seed = a.seed
    if seed not in t.ALLOWED_SEEDS:
        raise SystemExit(f"training seed {seed} не разрешён")
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=True)
    pin = D["prep_report"]["bar_pin"]
    prob, rep = [], dict(seed=seed, fix="проверка градиентов по группам")

    def built(mode, s):
        bar = d21.load_bar(dev, pin)
        m = bar["model"]
        m.action_expert.float()
        return t.build_module(torch, mode, m, D, s).to(dev), bar

    states = {}
    for tag, mode, s in (("flat_s", "flat24", seed),
                         ("flat_s_again", "flat24", seed),
                         ("flat_0", "flat24", 0),
                         ("parents_s", "parents24", seed)):
        mod, bar = built(mode, s)
        states[tag] = kt.trainable_state(mod)
        rep[f"init_hash_{tag}"] = kt.state_hash(states[tag])
        if tag in ("flat_s", "parents_s"):
            gn = groups_named(mod)
            named = [(n, p) for n, p in mod.named_parameters()
                     if p.requires_grad]
            opt = torch.optim.AdamW([p for _n, p in named], lr=0.0)
            rows = D["train_rows"][:: max(1, len(D["train_rows"]) // 64)]
            batch, po, st_, tgt, val = d21.make_batch(D, rows[:a.micro],
                                                      bar, dev)
            torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.time()
            mod.train()
            with torch.autocast("cuda", dtype=torch.float16):
                out = mod(batch, po, st_)
                loss, _l1, _b = km.loss_fn(torch, out, tgt, val)
            loss.mean().backward()
            last = len(mod.bar.vlm.text_model.layers) - 1
            gp, none, norms = grad_problems(named, gn, last)
            opt.step()
            torch.cuda.synchronize(dev)
            c = mod.counters
            want = [16] * 24 if mode == "flat24" else \
                [4] * 8 + [12] * 8 + [28] * 8
            ok = dict(counters=c["active"] == want and c["vision"] == 1
                      and all(x == 1 for x in c["text"]),
                      finite=bool(torch.isfinite(out["arm"]).all()
                                  and torch.isfinite(loss).all()),
                      grads=not gp)
            rep[f"step_{tag}"] = dict(
                ok, seconds=time.time() - t0, loss=float(loss.mean()),
                peak_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                group_grad_norms=norms, no_grad_params=none,
                grad_problems=gp)
            for k, v in ok.items():
                if not v:
                    prob.append(f"{tag}: {k}" + (f" ({gp[:2]})" if k ==
                                                 "grads" else ""))
            del opt
        del mod, bar
        torch.cuda.empty_cache()
    if rep["init_hash_flat_s"] != rep["init_hash_flat_s_again"]:
        prob.append("повторное построение с тем же seed даёт другие веса")
    cp = t.init_compare(states["flat_s"], states["parents_s"])
    prob += t.pair_problems(cp)
    c0 = t.init_compare(states["flat_s"], states["flat_0"])
    prob += t.seed_change_problems(c0)
    rep["pair_compare"] = {k: len(v) for k, v in cp.items()}
    rep["seed_compare"] = {k: len(v) for k, v in c0.items()}
    hist = {}
    path = "data/k21/runs/flat24/resume.pt"
    if os.path.exists(path):
        h = torch.load(path, map_location="cpu", weights_only=False)
        hist["flat24"] = dict(historical=h["init_hash"],
                              rebuilt_seed0=rep["init_hash_flat_0"],
                              equal=h["init_hash"] == rep["init_hash_flat_0"])
    rep["historical_seed0"] = hist
    orig = kt.schedule
    fp_s = t.schedule_fp(t.seeded_schedule(orig, seed), D, a.micro)
    fp_direct = t.schedule_fp(lambda D_, n, m, start=0: orig(
        D_, n, m, seed=seed, start=start), D, a.micro)
    fp_0 = t.schedule_fp(t.seeded_schedule(orig, 0), D, a.micro)
    fp_hist = t.schedule_fp(lambda D_, n, m, start=0: orig(
        D_, n, m, start=start), D, a.micro)
    rep["schedule"] = dict(fp_seed=fp_s, fp_seed0=fp_0, fp_hist=fp_hist)
    if fp_s != fp_direct:
        prob.append("очередь seed S ≠ исходной функции с seed S")
    if fp_0 != fp_hist:
        prob.append("seed 0 не воспроизводит историческую очередь")
    if fp_s == fp_0:
        prob.append("очередь не зависит от seed")
    rep["problems"] = prob
    os.makedirs("reports/k23", exist_ok=True)
    out = f"reports/k23/smoke_s{seed}_grads.json"
    json.dump(rep, open(out, "w"), ensure_ascii=False, indent=1,
              default=float)
    for tag in ("flat_s", "parents_s"):
        r = rep[f"step_{tag}"]
        print(f"  {tag}: нормы групп {r['group_grad_norms']}; без градиента "
              f"{len(r['no_grad_params'])}: {r['no_grad_params'][:4]}; "
              f"пик {r['peak_gb']:.1f} ГБ")
    print(f"  пара: {rep['pair_compare']}; смена seed: {rep['seed_compare']};"
          f" seed 0 = историческая инициализация: {hist}")
    print("  " + ("ОТКАЗ: " + "; ".join(prob) if prob else
                  f"smoke K-23 seed {seed} пройден") + f"; отчёт {out}")
    return 3 if prob else 0


def selftest():
    import torch

    class P:
        def __init__(self, g):
            self.grad = g
    named = [("bar.action_expert.layers.0.w", P(torch.ones(2))),
             ("bar.action_expert.embed_tokens.weight", P(None)),
             ("bar.vlm.l.A", P(torch.ones(2))), ("slot", P(torch.ones(1)))]
    gn = dict(expert=[named[0][0], named[1][0]], lora=[named[2][0]],
              new=[named[3][0]])
    p, none, norms = grad_problems(named, gn)
    assert p == [] and none == ["bar.action_expert.embed_tokens.weight"]
    named2 = named + [("readout.w", P(None))]
    gn2 = dict(gn, new=gn["new"] + ["readout.w"])
    assert grad_problems(named2, gn2)[0]                # используемый без grad
    named3 = [named[0], named[1], ("bar.vlm.l.A", P(None)), named[3]]
    assert any("lora" in x for x in grad_problems(named3, gn)[0])
    named5 = named + [("bar.vlm.text_model.layers.23.self_attn.o_proj.B",
                       P(None)),
                      ("bar.vlm.text_model.layers.22.self_attn.o_proj.B",
                       P(None))]
    gn5 = dict(gn, lora=gn["lora"] + [named5[-2][0], named5[-1][0]])
    pr5 = grad_problems(named5, gn5, 23)[0]
    assert pr5 and "layers.22" in pr5[0] and "layers.23" not in pr5[0], pr5
    named4 = [("bar.action_expert.layers.0.w",
               P(torch.tensor([float("inf")])))] + named[1:]
    assert grad_problems(named4, gn)[0]
    print("самопроверка k23_smoke пройдена: неиспользуемые эмбеддинги "
          "эксперта и LoRA o_proj последнего слоя без градиента допустимы, "
          "предпоследнего и прочие используемые — нет, нулевая "
          "группа и нечисловой градиент — отказ")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-23: smoke тренера")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro", type=int, default=32)
    return smoke(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
