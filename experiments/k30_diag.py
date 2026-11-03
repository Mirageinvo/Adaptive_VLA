#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-30: описательная диагностика плана корней по журналам раскаток (CPU).

Только полностью проверенные блоки (k30_protocol.check_run того же плана);
закреплённые файлы не читаются на запись. Отдельный отчёт
reports/k30/diag.json; на SR/анализ/цепочку не влияет.

Единица наблюдения — эпизод (среда блока): сначала среднее по активным
вызовам эпизода и четырём корням, затем разбивка по задачам и исходам.
Вызов c (0-based) активен для среды с done_step d, если d < 0 или d ≥ H·c
(k15g_local_probe.reached).
1) Уверенность классификатора: max p, энтропия (наты), разрыв top1 − top2
   (в вероятностях), p = softmax(logits) записанных logits.
2) Переключения argmax-кода — отдельно для каждого корня внутри эпизода,
   по соседним активным вызовам: доля переключений (на пару вызовов) и
   расстояние между прототипами при переключении.
3) Сборка из одних и тех же записанных logits: b_hard = P[argmax p],
   b_soft = Σ_k p_k P_k; расстояние ‖b_hard − b_soft‖.
Расстояния — в масштабе нормировки кодека: sqrt(Σ_c ((Δ_c)/s_c)²) по шести
каналам руки (s — из словаря). Шкала для сравнения — медиана расстояния
между случайной парой прототипов и до ближайшего соседа.
Это описание чувствительности к жёсткому выбору, а не установление причины
провалов; траектории H и C после первых действий различны, поэтому прямое
сравнение их планов по (задача, старт, номер вызова) не делается.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT = "reports/k30/diag.json"
H_CALL = 8
METRICS = ("maxp", "entropy", "gap", "switch_rate", "switch_dist",
           "hard_soft")


def softmax(lg):
    lg = np.asarray(lg, np.float64)
    e = np.exp(lg - lg.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def ndist(a, b, s):
    """Расстояние в нормировке кодека по последней оси (6 каналов)."""
    return np.sqrt((((np.asarray(a, np.float64) - b) / s) ** 2).sum(-1))


def active_calls(C, done_step, h=H_CALL):
    """[C] bool: вызов c активен для среды с данным done_step."""
    c = np.arange(C)
    return (done_step < 0) | (done_step >= h * c)


def episode_stats(logits, act, P, s):
    """logits [C,4,K] одной среды, act [C] bool -> метрики эпизода.
    switch_dist — среднее по всем переключениям эпизода (NaN, если их нет)."""
    lg = np.asarray(logits, np.float64)[act]                   # [A,4,K]
    A = len(lg)
    if A == 0:
        return None
    p = softmax(lg)
    srt = np.sort(p, -1)
    ent = -(p * np.log(np.clip(p, 1e-300, None))).sum(-1)
    ids = np.argmax(lg, -1)                                     # [A,4]
    Pd = np.asarray(P, np.float64)
    b_hard = Pd[ids]                                            # [A,4,6]
    b_soft = p @ Pd
    r = dict(calls=A, maxp=float(srt[..., -1].mean()),
             entropy=float(ent.mean()),
             gap=float((srt[..., -1] - srt[..., -2]).mean()),
             hard_soft=float(ndist(b_hard, b_soft, s).mean()))
    if A >= 2:
        sw = ids[1:] != ids[:-1]                                # [A-1,4]
        r["switch_rate"] = float(sw.mean())
        r["switch_rate_root"] = [float(x) for x in sw.mean(0)]
        r["switches"] = int(sw.sum())
        if sw.any():
            d = ndist(Pd[ids[1:]], Pd[ids[:-1]], s)[sw]
            r["switch_dist"] = float(d.mean())
        else:
            r["switch_dist"] = float("nan")
    else:
        r.update(switch_rate=float("nan"), switches=0,
                 switch_dist=float("nan"),
                 switch_rate_root=[float("nan")] * 4)
    return r


def block_episodes(d, z, P, s):
    lg = np.asarray(z["k21_k30_logits"])                        # [C,B,4,K]
    ds = np.asarray(z["done_step"])
    out = []
    for i, e in enumerate(d["episodes"]):
        r = episode_stats(lg[:, i], active_calls(lg.shape[0], int(ds[i])),
                          P, s)
        if r is None:
            continue
        r.update(task=int(d["task_id"]), state=int(e["init_state_id"]),
                 success=bool(e["success"]))
        out.append(r)
    return out


def scale_refs(P, s):
    Pd = np.asarray(P, np.float64)
    D = ndist(Pd[:, None], Pd[None], s)
    iu = np.triu_indices(len(Pd), 1)
    nn = np.where(np.eye(len(Pd), dtype=bool), np.inf, D).min(1)
    return dict(pair_median=float(np.median(D[iu])),
                nn_median=float(np.median(nn)))


def summarize(eps):
    def agg(rows):
        o = dict(n=len(rows))
        for m in METRICS:
            v = np.array([r[m] for r in rows], np.float64)
            v = v[np.isfinite(v)]
            o[m] = dict(mean=float(v.mean()) if len(v) else None,
                        median=float(np.median(v)) if len(v) else None,
                        n=int(len(v)))
        return o
    res = dict(all=agg(eps),
               success=agg([r for r in eps if r["success"]]),
               fail=agg([r for r in eps if not r["success"]]),
               per_task={})
    for t in sorted({r["task"] for r in eps}):
        rows = [r for r in eps if r["task"] == t]
        res["per_task"][t] = dict(
            sr=f"{sum(r['success'] for r in rows)}/{len(rows)}",
            **{m: agg(rows)[m]["mean"] for m in METRICS})
    sw = np.array([r["switch_rate_root"] for r in eps], np.float64)
    res["switch_rate_root"] = [float(np.nanmean(sw[:, k])) for k in range(4)]
    return res


def load_verified(seed, mode):
    import k16_protocol as p16
    import k30_protocol as pc
    path = pc.plan_path(seed, mode)
    if not os.path.exists(path):
        return None, dict(status="нет плана")
    plan = json.load(open(path, encoding="utf-8"))
    prob = pc.check_plan_inputs(plan)
    if prob:
        raise SystemExit(f"{path}: " + "; ".join(prob[:3]))
    ok, gaps = pc.check_run(plan, path, pc.run_dir(path))
    blocks = [p16.load(f) for _k, f in sorted(ok.items())]
    return blocks, dict(status="выполнено", verified=len(ok),
                        planned=len(plan["cells"]) * len(plan["labels"]),
                        gaps=len(gaps))


def fmt(x, nd=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def report(res):
    rf = res["scale"]
    print(f"  шкала расстояний (нормировка кодека): медиана пары прототипов "
          f"{rf['pair_median']:.2f}, до ближайшего {rf['nn_median']:.2f}")
    for key, r in res["runs"].items():
        info = r["info"]
        if info["status"] != "выполнено":
            print(f"\n  {key}: {info['status']}")
            continue
        S = r["summary"]
        print(f"\n  {key}: блоков {info['verified']}/{info['planned']} "
              f"проверено, эпизодов {S['all']['n']} (успех "
              f"{S['success']['n']}, провал {S['fail']['n']})")
        print("    метрика          все      успех    провал   (средние по "
              "эпизодам)")
        for m in METRICS:
            print(f"    {m:<14} " + "  ".join(
                f"{fmt(S[g][m]['mean']):>7}" for g in ("all", "success",
                                                       "fail")))
        print("    переключения по корням 0–3: " + ", ".join(
            fmt(x) for x in S["switch_rate_root"]))
        print("    по задачам (SR | maxp | энтропия | переключ. | "
              "расст. переключ. | hard−soft):")
        for t, v in S["per_task"].items():
            print(f"      t{t}: {v['sr']:>5} | {fmt(v['maxp'])} | "
                  f"{fmt(v['entropy'])} | {fmt(v['switch_rate'])} | "
                  f"{fmt(v['switch_dist'], 2)} | {fmt(v['hard_soft'], 2)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import k30_data as kd
    C = kd.load()
    P, s = C["P32"], np.asarray(C["s"], np.float64)
    res = dict(scale=scale_refs(P, s), runs={},
               note="описательно; эпизод — единица; не причинный вывод")
    for seed in [int(x) for x in a.seeds.split(",") if x != ""]:
        for mode in ("hard", "soft"):
            blocks, info = load_verified(seed, mode)
            key = f"s{seed} {mode}"
            if not blocks:
                res["runs"][key] = dict(info=dict(info, status=info.get(
                    "status") if blocks is None else "нет проверенных блоков"))
                continue
            eps = [e for d, z in blocks for e in block_episodes(d, z, P, s)]
            res["runs"][key] = dict(info=info, summary=summarize(eps),
                                    episodes=eps)
    report(res)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, OUT)
    print(f"\n  отчёт: {OUT}")
    return 0


def selftest():
    rng = np.random.default_rng(0)
    P = rng.uniform(-0.8, 0.8, (128, 6)).astype(np.float32)
    s = np.full(6, 0.3)
    # активные вызовы: done_step 15 при H = 8 -> вызовы 0, 1 (16 > 15)
    assert active_calls(4, 15).tolist() == [True, True, False, False]
    assert active_calls(3, -1).all()
    assert active_calls(3, 16).tolist() == [True, True, True]
    # уверенный и постоянный план: maxp ≈ 1, нет переключений, hard ≈ soft
    lg = np.full((5, 4, 128), -50.0)
    lg[..., 7] = 50.0
    r = episode_stats(lg, np.ones(5, bool), P, s)
    assert r["maxp"] > 0.999 and r["switch_rate"] == 0 and \
        r["hard_soft"] < 1e-6 and np.isnan(r["switch_dist"])
    # равномерный: энтропия ln K, разрыв 0, hard−soft = расстояние до среднего
    r = episode_stats(np.zeros((3, 4, 128)), np.ones(3, bool), P, s)
    assert abs(r["entropy"] - np.log(128)) < 1e-9 and r["gap"] < 1e-12
    want = ndist(P[0].astype(np.float64), P.astype(np.float64).mean(0), s)
    assert abs(r["hard_soft"] - want) < 1e-9
    # переключение одного корня 7 -> 9 на втором вызове из трёх
    lg = np.full((3, 4, 128), -50.0)
    lg[..., 7] = 50.0
    lg[1:, 2, 7] = -50.0
    lg[1:, 2, 9] = 50.0
    r = episode_stats(lg, np.ones(3, bool), P, s)
    assert r["switches"] == 1 and abs(r["switch_rate"] - 1 / 8) < 1e-12
    assert r["switch_rate_root"] == [0.0, 0.0, 0.5, 0.0]
    assert abs(r["switch_dist"] - ndist(P[9].astype(np.float64),
                                        P[7].astype(np.float64), s)) < 1e-9
    # неактивные вызовы не учитываются (переключение после done)
    act = np.array([True, False, False])
    r = episode_stats(lg, act, P, s)
    assert r["calls"] == 1 and np.isnan(r["switch_rate"])
    # блок и сводка: эпизоды по средам, исходы
    Cn = 4
    lgb = np.zeros((Cn, 5, 4, 128))
    d = dict(task_id=3, episodes=[dict(init_state_id=30 + i,
                                       success=i % 2 == 0) for i in range(5)])
    z = {"k21_k30_logits": lgb, "done_step": np.array([7, -1, 31, 0, 8])}
    eps = block_episodes(d, z, P, s)
    assert [e["calls"] for e in eps] == [1, 4, 4, 1, 2]
    S = summarize(eps)
    assert S["all"]["n"] == 5 and S["success"]["n"] == 3 and \
        S["per_task"][3]["sr"] == "3/5"
    rf = scale_refs(P, s)
    assert 0 < rf["nn_median"] < rf["pair_median"]
    print("самопроверка k30_diag пройдена: активные вызовы по done_step, "
          "уверенность (max p, энтропия, разрыв), переключения по корням с "
          "расстоянием прототипов, hard−soft из одних logits, эпизод как "
          "единица, сводка по исходам и задачам")
    return 0


if __name__ == "__main__":
    sys.exit(main())
