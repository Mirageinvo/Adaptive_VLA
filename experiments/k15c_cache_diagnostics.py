#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15c: диагностика по готовому кэшу. Без модели VLA, минуты.

ДВЕ ЧАСТИ, ОБЕ — ОПИСАНИЕ, НИ ОДНА НЕ ЯВЛЯЕТСЯ ГЕЙТОМ.

1. ТОЧНАЯ ПЕРЕОЦЕНКА ЧЕКПОЙНТОВ на train и на val_sel — для выбранной и для
   последней эпохи. В истории обучения `train_regret` — среднее по батчам
   ПО ХОДУ изменения параметров, и по нему не отличить переобучение от
   слабой головы. Здесь каждое состояние оценивается целиком:

       hard RMS; доля разрыва (val_sel, с учителем) или доля оракульного
       запаса (train, где учителя нет: черновик -> лучший из восьми);
       ожидаемый regret softmax; доля ранга 0 и гистограмма;
       построчно относительно ранга 0 — где выбор лучше, где хуже и
       насколько; по задачам.

   Развилка, которую это различает:
       train заметно лучше, val на ранге 0  -> переобучение или сдвиг;
       и train, и val на ранге 0            -> objective/голова считают
                                               уход от ранга 0 невыгодным;
       отдельные задачи с сильным выигрышем -> возможен task-aware путь.

2. ПЕРЕНОС РЕШЕНИЯ НА СОСЕДНИЕ КАДРЫ. Берётся лучший ранг кадра t и
   применяется к затратам кадра t+d того же эпизода, d = 1, 2, 4, 8.
   Считаются RMS, доля оракульного запаса (и доля разрыва на val_sel), с
   бутстрапом по ЭПИЗОДАМ и по задачам; совпадение лучшего ранга у
   соседей — рядом, против случайного sum p^2.

   ЭТО НЕ ГЕЙТ ДЛЯ LoRA. Номер ранга у соседних кадров означает разные
   коды (порядок кандидатов меняется), чанки действий соседей сдвинуты и
   перекрываются, а цель может чередоваться между кадрами и при этом
   точно определяться состоянием. Замер отвечает на узкий вопрос: полезна
   ли сама стратегия переноса решения во времени.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

NEIGHBOUR_D = (1, 2, 4, 8)
N_BOOT = 1000
BOOT_SEED = 15


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64))))


def share(r_base, r_target, r_value):
    """(base - value) / (base - target); None при неположительном разрыве."""
    gap = float(r_base) - float(r_target)
    if gap <= 1e-12:
        return None
    return float((float(r_base) - float(r_value)) / gap)


def row_vs_rank0(costs, pick):
    """Построчно: выбор против ранга 0. Доли и средние величины."""
    c = np.asarray(costs, np.float64)
    sel = c[np.arange(len(c)), np.asarray(pick)]
    d = sel - c[:, 0]
    better, worse = d < 0, d > 0
    return dict(
        share_better=float(better.mean()), share_worse=float(worse.mean()),
        share_same=float((d == 0).mean()),
        mean_gain_where_better=(float(-d[better].mean())
                                if better.any() else None),
        mean_loss_where_worse=(float(d[worse].mean())
                               if worse.any() else None),
        net_mean=float(d.mean()))


def neighbour_pairs(episodes, steps, d):
    """Индексы (i, j): строка j — тот же эпизод, шаг на d позже строки i."""
    ep = np.asarray(episodes, np.int64)
    st = np.asarray(steps, np.int64)
    where = {(int(e), int(s)): k for k, (e, s) in enumerate(zip(ep, st))}
    ii, jj = [], []
    for k, (e, s) in enumerate(zip(ep.tolist(), st.tolist())):
        j = where.get((e, s + int(d)))
        if j is not None:
            ii.append(k)
            jj.append(j)
    return np.asarray(ii, np.int64), np.asarray(jj, np.int64)


def transfer_metrics(costs, i, j):
    """Лучший ранг кадра i, применённый к затратам кадра j."""
    c = np.asarray(costs, np.float64)
    best = c.argmin(1)
    cj = c[j]
    moved = cj[np.arange(len(j)), best[i]]
    r0, rt, ro = rms(cj[:, 0]), rms(moved), rms(cj.min(1))
    p = np.bincount(best, minlength=c.shape[1]) / float(len(best))
    return dict(pairs=int(len(i)), rms_rank0=r0, rms_transfer=rt,
                rms_oracle=ro, oracle_share=share(r0, ro, rt),
                same_best_rank=float((best[i] == best[j]).mean()),
                chance_same=float((p ** 2).sum()))


def boot_by_episode(fn, episodes, n=N_BOOT, seed=BOOT_SEED):
    """Бутстрап по эпизодам: пересэмплируются эпизоды, а не строки."""
    ep = np.asarray(episodes)
    uniq = np.unique(ep)
    by = {e: np.flatnonzero(ep == e) for e in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[e] for e in pick])
        v = fn(idx)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]


def selftest():
    # --- ПОСТРОЧНО ОТНОСИТЕЛЬНО РАНГА 0 ---------------------------------
    c = np.array([[1.0, 0.5, 2.0], [0.2, 0.9, 0.1], [0.3, 0.3, 0.3]])
    r = row_vs_rank0(c, [1, 1, 2])
    assert abs(r["share_better"] - 1 / 3) < 1e-12
    assert abs(r["share_worse"] - 1 / 3) < 1e-12
    assert abs(r["mean_gain_where_better"] - 0.5) < 1e-12
    assert abs(r["mean_loss_where_worse"] - 0.7) < 1e-12
    assert row_vs_rank0(c, [0, 0, 0])["mean_gain_where_better"] is None

    # --- СОСЕДИ ----------------------------------------------------------
    ep = np.array([0, 0, 0, 1, 1, 0])
    st = np.array([0, 1, 3, 0, 1, 2])
    i, j = neighbour_pairs(ep, st, 1)
    pairs = sorted(zip(i.tolist(), j.tolist()))
    assert pairs == [(0, 1), (1, 5), (3, 4), (5, 2)], pairs
    i2, j2 = neighbour_pairs(ep, st, 2)
    assert sorted(zip(i2.tolist(), j2.tolist())) == [(0, 5), (1, 2)]

    # --- ПЕРЕНОС: ИДЕАЛЬНО ГЛАДКАЯ ЦЕЛЬ ДАЁТ ОРАКУЛ, СЛУЧАЙНАЯ — НЕТ -----
    rng = np.random.default_rng(0)
    n = 400
    epi = np.repeat(np.arange(40), 10)
    stp = np.tile(np.arange(10), 40)
    smooth = rng.uniform(0.5, 1.0, (n, 8))
    lab = np.repeat(rng.integers(0, 8, 40), 10)       # лучший ранг по эпизоду
    smooth[np.arange(n), lab] = 0.1
    ii, jj = neighbour_pairs(epi, stp, 1)
    tm = transfer_metrics(smooth, ii, jj)
    assert abs(tm["oracle_share"] - 1.0) < 1e-9 and tm["same_best_rank"] == 1
    noisy = rng.uniform(0.5, 1.0, (n, 8))
    noisy[np.arange(n), rng.integers(0, 8, n)] = 0.1
    tn = transfer_metrics(noisy, ii, jj)
    assert tn["oracle_share"] < 0.3, tn["oracle_share"]
    assert abs(tn["chance_same"] - 1 / 8) < 0.05

    # --- БУТСТРАП ПО ЭПИЗОДАМ ------------------------------------------
    ci = boot_by_episode(lambda idx: float(np.mean(idx)), epi, n=200)
    assert ci is not None and ci[0] <= ci[1]
    assert share(1.0, 1.0, 0.5) is None and share(2.0, 1.0, 1.5) == 0.5
    print("самопроверка k15c_cache_diagnostics пройдена")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    ap = argparse.ArgumentParser(
        description="K-15c: диагностика по кэшу, без модели")
    ap.add_argument("--cache", default="data/k15c/rank_cache")
    ap.add_argument("--c1", default="data/k15b/c1_selected.pt")
    ap.add_argument("--k11a-cache", default="data/k11a_joint12")
    ap.add_argument("--summaries", nargs="*", default=None,
                    help="сводки тренера; по умолчанию все несмоук-сводки "
                         "reports/k15c/selector*_s0.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default="reports/k15c/diagnostics.json")
    ap.add_argument("--allow-smoke", action="store_true",
                    help="только для проверки самого скрипта на smoke-кэше")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if os.path.exists(a.out) and not a.overwrite:
        raise SystemExit(f"{a.out} уже существует")
    import torch
    import k15c_build_rank_cache as cb
    import k15c_rank_selector as rs
    import k15c_train_rank_selector as tr
    import k15_train_depth_rvq as k15t

    man, problems = cb.validate_cache(a.cache, allow_smoke=a.allow_smoke)
    if problems:
        raise SystemExit("кэш не принят: " + "; ".join(problems[:6]))
    book = torch.load(a.c1, map_location="cpu", weights_only=False)
    C1 = book["c1"].detach().float().cpu()
    if book.get("c1_sha1") != man["c1_sha1"]:
        raise SystemExit("книга не та, что у кэша")
    arr, data, teacher_va = tr.load_features(a.cache, man, C1, torch, rs)
    va, trn = data["val_sel"], data["train"]
    dev = torch.device(a.device)
    with open(f"{a.k11a_cache}.meta.json", encoding="utf-8") as fh:
        m11 = json.load(fh)
    d11 = np.load(m11["cache"], allow_pickle=True)
    epi_all, stp_all = np.asarray(d11["episode"]), np.asarray(d11["step"])
    vocab = man.get("task_vocab") or []
    out = dict(kind="k15c_cache_diagnostics",
               cache_manifest_sha1=cb.sha_file(os.path.join(
                   a.cache, "manifest.json")), reevaluation={},
               neighbours={})

    # --- 1. ТОЧНАЯ ПЕРЕОЦЕНКА ---------------------------------------------
    summaries = a.summaries
    if summaries is None:
        summaries = sorted(p for p in glob.glob("reports/k15c/selector*_s0"
                                                ".json")
                           if "_smoke" not in p)
    print(f"  сводок тренера: {len(summaries)}")
    for sp in summaries:
        summ = json.load(open(sp, encoding="utf-8"))
        for name, res in (summ.get("results") or {}).items():
            if not res.get("trained") or not os.path.exists(
                    res.get("checkpoint", "")):
                continue
            ck = torch.load(res["checkpoint"], map_location="cpu",
                            weights_only=False)
            states = ck.get("all_states") or {}
            last = max(states) if states else None
            src = "h18" if name.startswith("h18") else "h24"
            full = name in rs.NEEDS_FULL_H24
            codes_need = name in rs.NEEDS_BOOK
            inps = {}
            for part, dd in (("train", trn), ("val_sel", va)):
                inps[part] = tr.Inputs(
                    torch, dd["ctx18"] if src == "h18" else dd["ctx24"],
                    dd["emb"], dd["feat"],
                    h_full=dd["h24"] if full else None, device=dev,
                    cand_codes=dd["codes"] if codes_need else None)
            head = rs.build_head(name, int(ck["d_model"]), int(ck["e_dim"]),
                                 torch, proj=int(ck["proj"]),
                                 book=C1).to(dev)
            key = f"{os.path.basename(sp)}:{name}"
            out["reevaluation"][key] = {}
            for label, st in (("selected", ck["state"]),
                              ("last", states.get(last))):
                if st is None:
                    continue
                head.load_state_dict(st)
                row = {}
                for part, dd in (("train", trn), ("val_sel", va)):
                    sc = tr.scores_for(head, inps[part],
                                       np.arange(len(dd["costs"])), torch)
                    teacher = (teacher_va if part == "val_sel"
                               else dd["costs"].min(1))
                    ev = tr.evaluate_scores(sc, dd["costs"], dd["draft"],
                                            teacher, task_ids=dd["tasks"],
                                            task_vocab=vocab)
                    ev["vs_rank0"] = row_vs_rank0(dd["costs"],
                                                  sc.argmax(1))
                    ev["share_meaning"] = (
                        "доля разрыва черновик -> учитель" if
                        part == "val_sel" else
                        "доля оракульного запаса черновик -> лучший из 8")
                    row[part] = ev
                row["epoch"] = int(ck["selected_epoch"] if label ==
                                   "selected" else last)
                out["reevaluation"][key][label] = row
                t_, v_ = row["train"], row["val_sel"]
                print(f"  {key} [{label}, эпоха {row['epoch']}]: train RMS "
                      f"{t_['rms']:.6f} (доля запаса "
                      f"{100 * (t_['capture'] or 0):.1f} %, ранг 0 у "
                      f"{100 * t_['share_rank0']:.1f} %); val RMS "
                      f"{v_['rms']:.6f} ({100 * (v_['capture'] or 0):.1f} %, "
                      f"ранг 0 у {100 * v_['share_rank0']:.1f} %)")

    # --- 2. ПЕРЕНОС НА СОСЕДНИЕ КАДРЫ -------------------------------------
    for part, dd in (("train", trn), ("val_sel", va)):
        ep = epi_all[dd["rows"]]
        st = stp_all[dd["rows"]]
        out["neighbours"][part] = {}
        for d in NEIGHBOUR_D:
            i, j = neighbour_pairs(ep, st, d)
            if len(i) == 0:
                out["neighbours"][part][str(d)] = dict(pairs=0)
                print(f"  {part}, d={d}: пар соседей нет")
                continue
            tm = transfer_metrics(dd["costs"], i, j)
            ep_pair = ep[j]

            def share_of(idx, i=i, j=j):
                return transfer_metrics(dd["costs"], i[idx],
                                        j[idx])["oracle_share"]

            tm["oracle_share_ci90"] = boot_by_episode(share_of, ep_pair,
                                                      n=a.boot)
            if part == "val_sel":
                tm["capture"] = share(rms(dd["draft"][j]),
                                      rms(teacher_va[j]),
                                      tm["rms_transfer"])
                tm["capture_rank0"] = share(rms(dd["draft"][j]),
                                            rms(teacher_va[j]),
                                            tm["rms_rank0"])
            per = {}
            for tid in np.unique(dd["tasks"][j]):
                m = dd["tasks"][j] == tid
                nm = vocab[int(tid)] if int(tid) < len(vocab) else str(tid)
                per[nm] = transfer_metrics(dd["costs"], i[m], j[m])
            tm["per_task"] = per
            out["neighbours"][part][str(d)] = tm
            ci = tm["oracle_share_ci90"]
            print(f"  {part}, d={d}: пар {tm['pairs']}; перенос RMS "
                  f"{tm['rms_transfer']:.6f} против ранга 0 "
                  f"{tm['rms_rank0']:.6f} и оракула {tm['rms_oracle']:.6f}"
                  f" -> доля запаса "
                  f"{100 * (tm['oracle_share'] or 0):.1f} %"
                  + ("" if ci is None else
                     f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]")
                  + (f", доля разрыва {100 * (tm['capture'] or 0):.1f} %"
                     if part == "val_sel" else "")
                  + f"; тот же лучший ранг {100 * tm['same_best_rank']:.1f} "
                    f"% при случайном {100 * tm['chance_same']:.1f} %")
    out["note"] = ("ОПИСАНИЕ, НЕ ГЕЙТ. Перенос решения на соседей отвечает "
                   "лишь на вопрос, полезна ли сама стратегия переноса; "
                   "номер ранга у соседей означает разные коды, а их чанки "
                   "действий сдвинуты и перекрываются")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                exist_ok=True)
    tmp = a.out + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, allow_nan=False,
                  default=k15t.json_scalar)
    os.replace(tmp, a.out)
    print(f"  сводка: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
