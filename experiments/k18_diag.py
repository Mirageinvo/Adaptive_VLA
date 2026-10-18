#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-18: диагностика по готовым данным (CPU, без раскаток).

1. Поправки на оценках: ‖δ_exec‖, ‖c‖, доля ограничения команд, главные
   углы basis4 к PCA — по каждому файлу оценки (отметки 100k и 300k).
2. Настоящие r* (окна K-17, половины B и val_sel): ‖r* · scale‖ по 48
   числам одной позиции окна — исполняемые единицы K-18 (те же, что у
   δ_exec и ρ); без α и r_sigma.
3. Replay каждого варианта: нормы записанных поправок по этапам —
   общий q0 (нулевые), начальное исследование (актёр не обновлялся, первые
   64 эпизода), далее три равные по версиям поведения трети.
4. Критики финального состояния: Q1, Q2 на записанных и на текущих
   детерминированных действиях актёра; доля оценок вне [0, 1]; средняя
   оценка терминальных переходов с успехом и без.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def q(x, ps=(10, 25, 50, 75, 90, 95, 99)):
    x = np.asarray(x, np.float64)
    if not len(x):
        return None
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def eval_diag(eval_dir):
    out = {}
    for fn in sorted(os.listdir(eval_dir)):
        if not fn.endswith(".json") or fn.startswith("q0_"):
            continue
        d = json.load(open(os.path.join(eval_dir, fn)))
        g = d.get("diag") or []
        if not g:
            continue
        n = np.array([x["norm"] for x in g])
        c = np.array([np.linalg.norm(x["coef"]) for x in g])
        f = np.array([x["f"] for x in g])
        r = dict(decisions=len(g), norm=q(n), coef_norm=q(c),
                 shrink_rate=float((f < 1).mean()),
                 by_pos={str(p): q(n[[x["pos"] == p for x in g]], (10, 50,
                                                                    90))
                         for p in (0, 1, 2)})
        a = [x["angles"] for x in g if x.get("angles")]
        if a:
            A = np.array(a)
            r["principal_angles_deg"] = dict(max=q(A.max(1), (10, 50, 90)),
                                             mean=q(A.mean(1), (10, 50, 90)))
        out[fn[:-5]] = r
    return out


def rstar_norms(windows, scale, rho):
    Z = np.load(windows, allow_pickle=False)
    out = {}
    for sp in ("B", "V"):
        r = np.asarray(Z[f"{sp}__rstar"], np.float64)        # [N, 3, 8, 6]
        n = np.linalg.norm((r * np.asarray(scale[:6])).reshape(
            len(r), 3, -1), axis=-1)
        out[sp] = dict(windows=len(r), norm=q(n.ravel()),
                       by_pos={str(k): q(n[:, k], (10, 50, 90))
                               for k in range(3)},
                       share_above_rho=float((n > rho).mean()),
                       share_above_0_9rho=float((n > .9 * rho).mean()))
    out["units"] = ("‖r* · scale_arm‖ по 8 × 6 числам одной позиции окна; "
                    "r* = act − a0 в единицах кодека; scale — max|Q| "
                    "харнесса; α и r_sigma не применяются (единицы δ_exec "
                    "K-18 и ρ)")
    return out


def replay_and_critics(state_path, cfg, torch):
    import k18_td3 as td
    st = torch.load(state_path, map_location="cpu", weights_only=False)
    U = np.load(cfg["pca"]["path"])["U"]
    b = cfg["bounds"]
    td3 = dict(cfg["td3"], **{k: cfg["noise"][k]
                              for k in ("target_sigma", "target_cap")})
    out = {}
    init_eps = cfg["td3"]["init_noise_episodes"]
    for v in cfg["variants"]:
        d = st["replays"][v]["d"]
        n = int(st["replays"][v]["n"])
        dl = np.asarray(d["delta"][:n], np.float64)
        ver = np.asarray(d["version"][:n])
        nr = np.linalg.norm(dl, axis=1)
        zero = ~np.any(dl != 0, axis=1) & (ver == 0)
        # версия поведения = номер волны варианта; начальное исследование —
        # волны до набора init_noise_episodes собственных эпизодов
        init_v = int(np.ceil(init_eps / cfg["n_envs"]))
        stages = dict(q0_shared=zero, init_explore=~zero & (ver < init_v))
        rest = ~zero & (ver >= init_v)
        if rest.any():
            edges = np.quantile(ver[rest], [1 / 3, 2 / 3])
            stages["learn_1"] = rest & (ver <= edges[0])
            stages["learn_2"] = rest & (ver > edges[0]) & (ver <= edges[1])
            stages["learn_3"] = rest & (ver > edges[1])
        rep = {k: dict(n=int(m.sum()), versions=[int(ver[m].min()),
                                                  int(ver[m].max())]
                       if m.any() else None, norm=q(nr[m], (10, 50, 90)),
                       share_near_rho=float((nr[m] > .95 * cfg["rho"])
                                            .mean()) if m.any() else None)
               for k, m in stages.items()}
        ag = td.Agent(torch, v, U, cfg["rho"], np.asarray(cfg["scale"]),
                      b["lo"], b["hi"], td3, "cpu")
        ag.load(st["agents"][v])
        obs = torch.as_tensor(np.asarray(d["obs"][:n], np.float32))
        with torch.no_grad():
            act = ag.actor(obs)["delta"]
            res = {}
            for nm, dd in (("recorded", torch.as_tensor(dl, dtype=torch.float32)),
                           ("actor", act)):
                q1 = ag.c1(obs, dd).numpy()
                q2 = ag.c2(obs, dd).numpy()
                qm = np.minimum(q1, q2)
                res[nm] = dict(q1=q(q1, (1, 50, 99)), qmin=q(qm, (1, 50, 99)),
                               share_q1_below_0=float((q1 < 0).mean()),
                               share_q1_above_1=float((q1 > 1).mean()),
                               min=float(q1.min()), max=float(q1.max()))
            q1r = ag.c1(obs, torch.as_tensor(dl, dtype=torch.float32)).numpy()
            q1a = ag.c1(obs, act).numpy()
        an = np.linalg.norm(act.numpy(), axis=1)
        done = np.asarray(d["done"][:n]) > .5
        rew = np.asarray(d["rew"][:n])
        res["actor_minus_recorded_q1_mean"] = float((q1a - q1r).mean())
        res["actor_norm_on_replay_obs"] = q(an, (10, 50, 90))
        res["terminal_mean_q1_recorded"] = dict(
            success=float(q1r[done & (rew > .5)].mean())
            if (done & (rew > .5)).any() else None,
            failure=float(q1r[done & (rew < .5)].mean())
            if (done & (rew < .5)).any() else None,
            share_success=float(rew[done].mean()) if done.any() else None)
        out[v] = dict(replay=rep, critics=res, n=n)
    return out


def report(res):
    rho = res["rho"]
    print(f"  ρ = {rho:.4f} (исполняемые единицы)")
    print("  1. Поправки на оценках (‖δ‖ p10/p50/p90; ‖c‖ p50; ограничение):")
    for k, r in res["eval"].items():
        s = (f"    {k:26} ‖δ‖ {r['norm']['p10']:.3f}/{r['norm']['p50']:.3f}/"
             f"{r['norm']['p90']:.3f}  ‖c‖ {r['coef_norm']['p50']:.3f}  "
             f"огр. {100 * r['shrink_rate']:.1f}%")
        if "principal_angles_deg" in r:
            pa = r["principal_angles_deg"]["max"]
            s += f"  макс. угол к PCA p50 {pa['p50']:.1f}°"
        print(s)
    print("  2. Настоящие r* (‖r*·scale‖, позиция окна):")
    for sp in ("B", "V"):
        r = res["rstar"][sp]
        n = r["norm"]
        print(f"    {sp}: окон {r['windows']}, p10 {n['p10']:.3f} p25 "
              f"{n['p25']:.3f} p50 {n['p50']:.3f} p75 {n['p75']:.3f} p90 "
              f"{n['p90']:.3f} p95 {n['p95']:.3f} p99 {n['p99']:.3f}; > ρ "
              f"{100 * r['share_above_rho']:.1f}%, > 0.9ρ "
              f"{100 * r['share_above_0_9rho']:.1f}%")
    print("  3–4. Replay и критики:")
    for v, r in res.get("train", {}).items():
        print(f"    {v}: переходов {r['n']}")
        for k, x in r["replay"].items():
            if x["n"]:
                print(f"      {k:12} n {x['n']:5d} версии {x['versions']}  "
                      f"‖δ‖ p10/p50/p90 {x['norm']['p10']:.3f}/"
                      f"{x['norm']['p50']:.3f}/{x['norm']['p90']:.3f}  "
                      f"> 0.95ρ {100 * x['share_near_rho']:.1f}%")
        c = r["critics"]
        for nm in ("recorded", "actor"):
            x = c[nm]
            print(f"      Q1 на действиях {nm:8}: p1/p50/p99 "
                  f"{x['q1']['p1']:.3f}/{x['q1']['p50']:.3f}/"
                  f"{x['q1']['p99']:.3f}, мин {x['min']:.3f}, макс "
                  f"{x['max']:.3f}, < 0 {100 * x['share_q1_below_0']:.1f}%, "
                  f"> 1 {100 * x['share_q1_above_1']:.1f}%")
        t = c["terminal_mean_q1_recorded"]
        print(f"      Q1(актёр) − Q1(записанное) в среднем "
              f"{c['actor_minus_recorded_q1_mean']:+.3f}; ‖δ актёра‖ на "
              f"наблюдениях replay p50 "
              f"{c['actor_norm_on_replay_obs']['p50']:.3f}; терминальные: "
              f"Q1 успех {t['success']:.3f}, неуспех {t['failure']:.3f} "
              f"(доля успеха {t['share_success']:.3f})")


def selftest():
    rng = np.random.default_rng(0)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "w.npz")
        np.savez(p, B__rstar=rng.standard_normal((20, 3, 8, 6)) * .1,
                 V__rstar=rng.standard_normal((5, 3, 8, 6)) * .1)
        r = rstar_norms(p, np.ones(7), 0.7)
        assert r["B"]["windows"] == 20 and 0 < r["B"]["norm"]["p50"] < 1.5
        os.makedirs(os.path.join(td, "ev"))
        json.dump(dict(diag=[dict(pos=k % 3, norm=.7, coef=[.9, .1, 0, 0],
                                  f=1.0, angles=[1, 2, 3, 80])
                             for k in range(9)]),
                  open(os.path.join(td, "ev", "basis4_rl_300k_final.json"),
                       "w"))
        e = eval_diag(os.path.join(td, "ev"))
        assert e["basis4_rl_300k_final"]["principal_angles_deg"]["max"][
            "p50"] == 80
    print("самопроверка k18_diag пройдена")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-18: диагностика (CPU)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--config", default="data/k18/config.json")
    ap.add_argument("--state", default="data/k18/run/state.pt")
    ap.add_argument("--eval-dir", default="reports/k18/eval")
    ap.add_argument("--out", default="reports/k18/diag.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import torch
    cfg = json.load(open(a.config))
    res = dict(rho=float(cfg["rho"]), eval=eval_diag(a.eval_dir),
               rstar=rstar_norms(cfg["inputs"]["windows"], cfg["scale"],
                                 float(cfg["rho"])))
    res["train"] = replay_and_critics(a.state, cfg, torch)
    report(res)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    print(f"  сводка: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
