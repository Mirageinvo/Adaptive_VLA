#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: диагностика обучения и иерархии.

CPU (по reports/k21/train.json, без вычислений): лучшая отметка, кривые,
отдельные агрегаты экспертных и учительских батчей, нормы градиентов
(эксперт слой 1/24, LoRA, новые модули, головы длительностей и делений,
расширения), насыщение пределов, L по задачам; для дерева — квантили
корневых длительностей, отклонение от равномерных, массы листьев.

--gpu: распределения ВЛОЖЕННЫХ делений выбранного чекпойнта tree_adapt24
на фиксированной выборке val_sel (корневые длительности, относительные
точки делений уровней 4 -> 8 и 8 -> 16, ширины листьев, массы листьев);
равномерное дерево: длительности 5, деления 0.5, ширины 1.25.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def cpu_summary(tr):
    out = {}
    for m, r in tr.items():
        c = r["curve"]
        best = r["best"]
        rec = dict(best_update=best[0], best_val_L=best[1],
                   updates=r["updates"], best_is_last=r["best_is_last"],
                   extension=r["extension"])
        b = next(x for x in c if x["update"] == best[0])
        v = b["val"]
        rec["best_val"] = dict(L=v["L"], l1_arm_20=v["l1_arm_20"],
                               l1_arm_8=v["l1_arm_8"], grip=v["grip_acc"],
                               saturation=v["saturation"])
        rec["per_task_L_best"] = v.get("per_task_L")
        rec["train_sources_last"] = c[-1].get("train")
        rec["grad"] = [dict(update=x["update"], **(x.get("grad") or {}))
                       for x in c if x.get("grad")]
        if "tree" in v:
            rec["tree_best"] = v["tree"]
            rec["tree_curve"] = [dict(update=x["update"],
                                      dev=x["val"]["tree"][
                                          "root_dev_from_uniform"],
                                      leaves_lt_0_1=x["val"]["tree"][
                                          "leaves_mass_lt_0_1"])
                                 for x in c if "tree" in x["val"]]
        out[m] = rec
    return out


def print_cpu(s):
    for m, r in s.items():
        v = r["best_val"]
        print(f"  {m}: лучшая {r['best_update']} из {r['updates']}"
              + (" (последняя)" if r["best_is_last"] else "")
              + f"; val L {v['L'][0]:.4f} (по строкам {v['L'][1]:.4f}), "
              f"рука-20 {v['l1_arm_20'][0]:.4f}, -8 {v['l1_arm_8'][0]:.4f}, "
              f"схват {100 * v['grip'][0]:.1f}%, насыщение "
              f"{100 * v['saturation'][0]:.2f}%")
        ts = r["train_sources_last"] or {}
        print("    последние агрегаты обучения: " + ", ".join(
            f"{k}: L {x.get('L', float('nan')):.4f}" for k, x in ts.items()))
        g = r["grad"][-1] if r["grad"] else {}
        print("    нормы градиентов (последняя запись): " + ", ".join(
            f"{k} {x:.3g}" for k, x in g.items() if k != "update"))
        if "tree_best" in r:
            t = r["tree_best"]
            q = np.asarray(t["root_durations_q"])
            print(f"    корень: квантили 5/25/50/75/95% длительностей по слотам"
                  f" {np.round(q, 2).tolist()}; среднее |d − 5| "
                  f"{t['root_dev_from_uniform']:.3f}")
            print(f"    массы листьев (среднее по val): "
                  f"{np.round(t['leaf_mass_mean'], 2).tolist()}; листьев с "
                  f"массой < 0.1: {t['leaves_mass_lt_0_1']:.2f}")
            print("    отклонение корня по проверкам: " + ", ".join(
                f"{x['update']}: {x['dev']:.3f}" for x in r["tree_curve"]))
        pt = r["per_task_L_best"] or {}
        if pt:
            worst = sorted(pt.items(), key=lambda kv: -kv[1])[:5]
            print("    худшие задачи (L): " + "; ".join(
                f"{k[:40]} {v:.3f}" for k, v in worst))


def gpu_splits(a):
    import torch
    import k21_data as d21
    import k21_policy as kp21
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    mod, bar, _ck, _p = kp21.load_policy(
        torch, "tree_adapt24", "data/k21/runs/tree_adapt24/best.pt",
        a.device)
    va = np.asarray(D["val_rows"])
    rng = np.random.default_rng(20261021)
    rows = np.sort(rng.choice(va, min(a.n, len(va)), replace=False))
    R, S1, S2, W, M = [], [], [], [], []
    ac = torch.autocast("cuda", dtype=torch.float16)
    with torch.no_grad():
        for s0 in range(0, len(rows), 8):
            r = rows[s0:s0 + 8]
            po = D["pos_offset"][r]
            batch, po_t, st, _t, _v = d21.make_batch(D, r, bar, dev)
            with ac:
                o = mod(batch, po_t, st)
            root = o["root"].float()
            mid = o["mid"].float()
            leaves = o["leaves"].float()
            R.append((root[..., 1] - root[..., 0]).cpu().numpy())
            S1.append(((o["split1"].float() - root[..., 0])
                       / (root[..., 1] - root[..., 0])).cpu().numpy())
            S2.append(((o["split2"].float() - mid[..., 0])
                       / (mid[..., 1] - mid[..., 0])).cpu().numpy())
            W.append((leaves[..., 1] - leaves[..., 0]).cpu().numpy())
            M.append(o["weights"].float().sum(1).cpu().numpy())
            _ = po
    R, S1, S2 = np.concatenate(R), np.concatenate(S1), np.concatenate(S2)
    W, M = np.concatenate(W), np.concatenate(M)

    def qq(x):
        return np.round(np.percentile(x, [5, 25, 50, 75, 95]), 3).tolist()
    res = dict(n=int(len(rows)),
               root_durations=dict(q=qq(R), mean_abs_dev=float(np.abs(
                   R - 5).mean()), per_slot_median=np.median(R, 0).tolist()),
               split_4_to_8=dict(q=qq(S1), mean_abs_dev=float(np.abs(
                   S1 - .5).mean()), per_node_median=np.median(S1, 0)
                   .tolist()),
               split_8_to_16=dict(q=qq(S2), mean_abs_dev=float(np.abs(
                   S2 - .5).mean()), per_node_median=np.median(S2, 0)
                   .tolist()),
               leaf_widths=dict(q=qq(W), mean_abs_dev=float(np.abs(
                   W - 1.25).mean())),
               leaf_mass=dict(mean=M.mean(0).tolist(),
                              lt_0_1=float((M < 0.1).sum(1).mean())),
               spread=dict(root_sd_across_obs=float(R.std(0).mean()),
                           split1_sd=float(S1.std(0).mean()),
                           split2_sd=float(S2.std(0).mean())))
    print(f"  вложенные деления tree_adapt24 на {res['n']} наблюдениях val_sel:")
    for k in ("root_durations", "split_4_to_8", "split_8_to_16",
              "leaf_widths"):
        print(f"    {k}: квантили 5–95% {res[k]['q']}, среднее отклонение от "
              f"равномерного {res[k]['mean_abs_dev']:.3f}")
    print(f"    разброс между наблюдениями (sd): корень "
          f"{res['spread']['root_sd_across_obs']:.3f}, деления 4→8 "
          f"{res['spread']['split1_sd']:.3f}, 8→16 "
          f"{res['spread']['split2_sd']:.3f}")
    print(f"    листьев с массой < 0.1 в среднем {res['leaf_mass']['lt_0_1']:.2f}")
    return res


def main():
    ap = argparse.ArgumentParser(description="K-21: диагностика")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", default="reports/k21/diag.json")
    a = ap.parse_args()
    tr = json.load(open("reports/k21/train.json"))
    s = cpu_summary(tr)
    print_cpu(s)
    out = dict(cpu=s)
    if a.gpu:
        out["nested_splits"] = gpu_splits(a)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1,
              default=float)
    print(f"  отчёт: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
