#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: честный замер времени — fullbar (штатный путь харнесса: generate
трёх уровней + сборка ActionCodec), flat24, tree_adapt24 (один проход).

Фиксированная панель настоящих наблюдений val_sel (кадры, подсказка,
состояние), батчи 1 и 5, одна карта, политики загружены одновременно;
50 прогревов, затем ≥ 300 измерений на политику, чередуя политики
блоками по 10. Части: подготовка входа (процессор + перенос, wall-clock),
vision encoder (CUDA events), вычисление политики до готового чанка
(CUDA events), полный путь (wall-clock с синхронизацией). Медиана, p95,
пропускная способность, пик памяти. Отладочный журнал выключен. LoRA не
сливается (численное соответствие слияния отдельно не проверялось).
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

WARM, N_MEAS, BLOCK = 50, 300, 10
PANEL = 40


def q(x):
    x = np.asarray(x)
    return dict(median=float(np.median(x)), p95=float(np.percentile(x, 95)),
                mean=float(x.mean()))


def main():
    ap = argparse.ArgumentParser(description="K-21: латентность")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--n", type=int, default=N_MEAS)
    ap.add_argument("--out", default="reports/k21/latency.json")
    a = ap.parse_args()
    import torch
    import k20_data as kd
    import k21_data as d21
    import k21_model as km
    import k21_policy as kp21
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    prep = D["prep_report"]
    bar = d21.load_bar(dev, prep["bar_pin"])
    pols = {}
    for lab in ("flat24", "tree_adapt24"):
        mod, b2, _ck, _p = kp21.load_policy(
            torch, lab, f"data/k21/runs/{lab}/best.pt", a.device)
        pols[lab] = mod
    va = np.asarray(D["val_rows"])
    rng = np.random.default_rng(20261021)
    panel = np.sort(rng.choice(va, PANEL, replace=False))
    E, codec = bar["E"], bar["codec"]
    ac = torch.autocast("cuda", dtype=torch.float16)

    def ev():
        return torch.cuda.Event(enable_timing=True)

    def run(lab, rows):
        t0 = time.perf_counter()
        batch = kd.prompt_batch(bar, [np.asarray(D["IMG"][i]) for i in rows],
                                kd.norm_states(np.asarray(D["state8"][rows])),
                                D["task"][rows].tolist())
        po = int(D["pos_offset"][rows[0]])
        s11 = torch.as_tensor(km.state11(np.asarray(D["state8"][rows])),
                              device=dev)
        torch.cuda.synchronize(dev)
        t_prep = time.perf_counter() - t0
        e0, e1 = ev(), ev()
        with torch.no_grad():
            e0.record()
            img = bar["model"].vlm.get_image_features(
                batch["pixel_values"], batch.get("pixel_attention_mask"))
            e1.record()
            del img
            g0, g1 = ev(), ev()
            g0.record()
            if lab == "fullbar":
                tk = bar["model"].generate(**batch, position_offset=po,
                                           do_sample=False,
                                           initial_position_shift=1)
                K = tk.reshape(len(rows), 3, 16)
                z = sum(E[j][K[:, j, :].long()] for j in range(3))
                x, _ = codec._decode(z, embodiment_ids=0)
                out = x[..., :7]
            else:
                with ac:
                    out = pols[lab](batch, po, s11)["arm"]
            g1.record()
        torch.cuda.synchronize(dev)
        total = time.perf_counter() - t0
        return dict(prep=t_prep * 1e3, vision=e0.elapsed_time(e1),
                    policy=g0.elapsed_time(g1), total=total * 1e3,
                    finite=bool(torch.isfinite(out.float()).all()))
    labs = ["fullbar", "flat24", "tree_adapt24"]
    res = {}
    for bs in a.batches:
        sel = [panel[(i * bs) % PANEL:(i * bs) % PANEL + bs]
               for i in range(PANEL)]
        sel = [s for s in sel if len(s) == bs]
        for lab in labs:
            for i in range(WARM):
                run(lab, sel[i % len(sel)])
        meas = {lab: [] for lab in labs}
        torch.cuda.reset_peak_memory_stats(dev)
        i = 0
        while min(len(v) for v in meas.values()) < a.n:
            for lab in labs:
                for _ in range(BLOCK):
                    r = run(lab, sel[i % len(sel)])
                    if not r["finite"]:
                        raise SystemExit(f"{lab}: нечисловой выход")
                    meas[lab].append(r)
                    i += 1
        res[f"batch{bs}"] = {lab: dict(
            {k: q([m[k] for m in v]) for k in ("prep", "vision", "policy",
                                                "total")},
            throughput_obs_s=float(bs * 1e3 / np.median([m["total"]
                                                        for m in v])),
            n=len(v)) for lab, v in meas.items()}
        res[f"batch{bs}"]["peak_memory_gb_all_loaded"] = \
            torch.cuda.max_memory_allocated(dev) / 2 ** 30
        for lab in labs:
            r = res[f"batch{bs}"][lab]
            print(f"  batch {bs} {lab:13}: полный путь медиана "
                  f"{r['total']['median']:.1f} мс (p95 {r['total']['p95']:.1f})"
                  f", политика {r['policy']['median']:.1f} мс, vision "
                  f"{r['vision']['median']:.1f} мс, подготовка "
                  f"{r['prep']['median']:.1f} мс; "
                  f"{r['throughput_obs_s']:.1f} набл./с", flush=True)
        fb = res[f"batch{bs}"]["fullbar"]["total"]["median"]
        for lab in ("flat24", "tree_adapt24"):
            x = res[f"batch{bs}"][lab]["total"]["median"]
            print(f"    {lab} против fullbar: {100 * (1 - x / fb):+.1f}% "
                  f"медианы полного пути", flush=True)
    res["note"] = ("fullbar — штатный generate (на каждом из трёх блоков "
                   "префикс пересчитывается) + сборка ActionCodec; "
                   "flat24/tree_adapt24 — один проход с fp16 autocast, LoRA "
                   "не слита; vision входит и в 'policy'")
    res["device"] = torch.cuda.get_device_name(dev)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"  отчёт: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
