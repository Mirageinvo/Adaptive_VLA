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
SEED = 20261021


def q(x):
    x = np.asarray(x)
    return dict(median=float(np.median(x)), p95=float(np.percentile(x, 95)),
                mean=float(x.mean()))


def make_queue(task, va, bs, n_batches, seed=SEED):
    """Очередь входов, одинаковая для всех политик: батч — из одной задачи
    (общее pos_offset), задачи по кругу."""
    rng = np.random.default_rng([seed, bs])
    by = {}
    for r in va:
        by.setdefault(str(task[r]), []).append(int(r))
    tasks = sorted(t for t, v in by.items() if len(v) >= bs)
    out = []
    for i in range(n_batches):
        t = tasks[i % len(tasks)]
        out.append(np.sort(rng.choice(by[t], bs, replace=False)))
    return out


class VisionTimer:
    """Замер vision encoder ВНУТРИ единственного реального вызова политики
    (обёртка get_image_features; счётчик вызовов)."""

    def __init__(self, torch, vlm):
        self.torch, self.vlm = torch, vlm
        self.orig = vlm.get_image_features
        self.events, self.calls = [], 0

        def wrapped(*a, **k):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            out = self.orig(*a, **k)
            e1.record()
            self.events.append((e0, e1))
            self.calls += 1
            return out
        vlm.get_image_features = wrapped

    def take(self):
        ms = sum(a.elapsed_time(b) for a, b in self.events)
        n = self.calls
        self.events, self.calls = [], 0
        return ms, n


def main():
    ap = argparse.ArgumentParser(description="K-21: латентность")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--n", type=int, default=N_MEAS)
    ap.add_argument("--out", default="reports/k21/latency.json")
    ap.add_argument("--with-cached", action="store_true",
                    help="добавить fullbar с кэшем префикса (k21_barcache)")
    a = ap.parse_args()
    import torch
    import k20_data as kd
    import k21_data as d21
    import k21_model as km
    import k21_policy as kp21
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    prep = D["prep_report"]
    if a.with_cached:
        import k21_barcache as bc
        try:
            bc.require_verified(D)
        except SystemExit as e:
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
    bar = d21.load_bar(dev, prep["bar_pin"])
    models = dict(fullbar=bar["model"])
    for lab in ("flat24", "tree_adapt24"):
        mod, _b2, _ck, _p = kp21.load_policy(
            torch, lab, f"data/k21/runs/{lab}/best.pt", a.device)
        models[lab] = mod
    if a.with_cached:
        models["fullbar_cached"] = bar["model"]
    vt = {}
    timers = {}
    for lab, m in models.items():           # общий vlm — один таймер
        vlm = m.vlm if lab.startswith("fullbar") else m.bar.vlm
        if id(vlm) not in vt:
            vt[id(vlm)] = VisionTimer(torch, vlm)
        timers[lab] = vt[id(vlm)]
    E, codec = bar["E"], bar["codec"]
    ac = torch.autocast("cuda", dtype=torch.float16)
    va = np.asarray(D["val_rows"])

    cache = {}

    def run(lab, rows):
        """Наблюдение -> готовый чанк [B, 20, 7] на CPU (как отдаёт харнессу).
        Кадры — из заранее заполненного кэша в памяти (вне таймера)."""
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        batch = kd.prompt_batch(bar, [cache[int(i)] for i in rows],
                                kd.norm_states(np.asarray(D["state8"][rows])),
                                D["task"][rows].tolist())
        pos = np.unique(D["pos_offset"][rows])
        if len(pos) != 1:
            raise SystemExit("в батче разные pos_offset")
        po = int(pos[0])
        s11 = torch.as_tensor(km.state11(np.asarray(D["state8"][rows])),
                              device=dev)
        torch.cuda.synchronize(dev)
        t_prep = time.perf_counter() - t0
        g0 = torch.cuda.Event(enable_timing=True)
        g1 = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            g0.record()
            if lab == "fullbar_cached":
                import k21_barcache as bc
                tk = bc.generate_cached(torch, models[lab], batch, po)
            if lab == "fullbar":
                tk = models[lab].generate(**batch, position_offset=po,
                                          do_sample=False,
                                          initial_position_shift=1)
            if lab.startswith("fullbar"):
                K = tk.reshape(len(rows), 3, 16)
                z = sum(E[j][K[:, j, :].long()] for j in range(3))
                x, _ = codec._decode(z, embodiment_ids=0)
                chunk = x[..., :7].float()
            else:
                with ac:
                    o = models[lab](batch, po, s11)
                g = torch.where(o["grip_logit"].float() > 0, 1.0, -1.0)
                chunk = torch.cat([o["arm"].float(), g[..., None]], -1)
            g1.record()
        a_np = chunk.cpu().numpy()
        t_total = time.perf_counter() - t0
        v_ms, v_calls = timers[lab].take()
        return dict(prep=t_prep * 1e3, vision=v_ms, vision_calls=v_calls,
                    policy_gpu=g0.elapsed_time(g1), total=t_total * 1e3,
                    shape=list(a_np.shape),
                    finite=bool(np.isfinite(a_np).all()))
    labs = ["fullbar", "flat24", "tree_adapt24"] + (
        ["fullbar_cached"] if a.with_cached else [])
    res = {}
    for bs in a.batches:
        queue = make_queue(D["task"], va, bs, max(WARM, a.n))
        # все уникальные кадры очереди — в RAM до любых измерений, чтобы ни
        # одна политика не платила за первое чтение с диска
        for r in np.unique(np.concatenate(queue)):
            if int(r) not in cache:
                cache[int(r)] = np.array(D["IMG"][int(r)], copy=True)
        print(f"  batch {bs}: кадров в памяти {len(cache)} "
              f"({sum(x.nbytes for x in cache.values()) / 2 ** 20:.0f} МБ)",
              flush=True)
        for lab in labs:
            for j in range(WARM):
                run(lab, queue[j % len(queue)])
        meas = {lab: [] for lab in labs}
        torch.cuda.reset_peak_memory_stats(dev)
        pos = {lab: 0 for lab in labs}          # своя позиция в общей очереди
        while min(len(v) for v in meas.values()) < a.n:
            for lab in labs:
                for _ in range(BLOCK):
                    if len(meas[lab]) >= a.n:
                        break
                    r = run(lab, queue[pos[lab] % len(queue)])
                    pos[lab] += 1
                    if not r["finite"] or r["vision_calls"] != 1 or \
                            r["shape"][1:] != [20, 7]:
                        raise SystemExit(f"{lab}: выход {r['shape']}, вызовов "
                                         f"vision {r['vision_calls']}, "
                                         f"конечен {r['finite']}")
                    meas[lab].append(r)
        res[f"batch{bs}"] = {lab: dict(
            {k: q([m[k] for m in v]) for k in ("prep", "vision",
                                                "policy_gpu", "total")},
            throughput_obs_s=float(bs * 1e3 / np.median([m["total"]
                                                        for m in v])),
            n=len(v)) for lab, v in meas.items()}
        res[f"batch{bs}"]["peak_memory_gb_all_loaded"] = \
            torch.cuda.max_memory_allocated(dev) / 2 ** 30
        for lab in labs:
            r = res[f"batch{bs}"][lab]
            print(f"  batch {bs} {lab:13}: полный путь медиана "
                  f"{r['total']['median']:.1f} мс (p95 {r['total']['p95']:.1f})"
                  f", GPU политики {r['policy_gpu']['median']:.1f} мс (в т.ч. "
                  f"vision {r['vision']['median']:.1f}), подготовка "
                  f"{r['prep']['median']:.1f} мс; "
                  f"{r['throughput_obs_s']:.1f} набл./с", flush=True)
        fb = res[f"batch{bs}"]["fullbar"]["total"]["median"]
        refs = ["fullbar"] + (["fullbar_cached"] if a.with_cached else [])
        for ref in refs:
            fb = res[f"batch{bs}"][ref]["total"]["median"]
            for lab in [x for x in labs if x not in ("fullbar", ref)]:
                x = res[f"batch{bs}"][lab]["total"]["median"]
                print(f"    {lab} против {ref}: {100 * (1 - x / fb):+.1f}% "
                      f"медианы полного пути", flush=True)
    res["note"] = ("полный путь: подготовка входа -> готовый чанк [B, 20, 7] "
                   "на CPU; кадры заранее в RAM (вне таймера); vision — внутри "
                   "единственного вызова политики; "
                   "очередь входов одинакова для всех политик, батч — из одной "
                   "задачи; fullbar — штатный generate трёх уровней + сборка "
                   "ActionCodec; flat24/tree_adapt24 — один проход, fp16 "
                   "autocast, LoRA не слита")
    res["device"] = torch.cuda.get_device_name(dev)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"  отчёт: {a.out}")
    return 0


def selftest():
    task = np.array(["a"] * 7 + ["b"] * 3 + ["c"] * 6)
    q5 = make_queue(task, np.arange(16), 5, 6)
    assert all(len(set(task[x])) == 1 and len(x) == 5 for x in q5)
    assert {task[x][0] for x in q5} == {"a", "c"}
    assert all(np.array_equal(x, y) for x, y in zip(q5, make_queue(
        task, np.arange(16), 5, 6)))
    print("самопроверка k21_latency пройдена: общая очередь, батч из одной "
          "задачи")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main())
