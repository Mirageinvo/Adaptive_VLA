#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-35: панели экспертных строк, плотный оракул 36 срезов, c_const, гейт.

Панели (§9), только экспертные строки, без teacher:
  oracle-train — до 64 строк на каждую присутствующую train-задачу;
  oracle-val   — до 16 строк на каждую присутствующую val_sel-задачу;
  checkpoint-val — до первых 8 строк каждой задачи из oracle-val.
Отбор: задачи — отсортированный список (исходные ID); для задачи с позицией
task_index — RNG SeedSequence([35, 20261114, split_id, task_index]);
эпизоды перемешиваются, строки каждого эпизода перемешиваются тем же RNG;
обход по кругу (первая строка каждого эпизода, затем вторая, …) до лимита.
Средние — с равными весами задач, внутри задачи — строк.

Оракул (§10): замороженный best K35-A; для каждой строки — 36 настоящих
прореженных forward (cuts = c); матрицы [N, 36]: C_full (L1 + 0.1·BCE на
доступных шагах), C_arm20, C_arm8 (первые min(8, valid)), C_grip; признаки:
нормированные корни после слоя 8 [N, 4, H], средние после слоя 16 для
каждого a [N, 6, 4, H], интервалы. Префикс одинаков для всех c (сверка);
средние одного a одинаковы для всех b (сверка).
c_const = argmin_c Mean_task(C_full_train[:, c]) (ничья — меньший c).
Гейт потенциала (oracle-val): все строки и 36 срезов конечны; gap_full =
(L_C − L_O)/L_C ≥ 0.10; относительное улучшение arm-8 того же оракула ≥
0.05; L_C ≤ 1.25·L_S (L_S — полный выход 24 источника K-34 на той же
панели). Иначе stopped_low_oracle.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k35_topology as tp  # noqa: E402

PANELS = "reports/k35/panels.json"
ORACLE_DIR = "data/k35/oracle"
GATE = "reports/k35/oracle_gate.json"
LIMITS = dict(oracle_train=64, oracle_val=16, checkpoint_val=8)
SPLIT_ID = dict(oracle_train=0, oracle_val=1)
CKPT_CUTS = (0, 7, 14, 21, 28, 35)
GATE_RULE = dict(gap_full=0.10, arm8_gain=0.05, c_vs_s=1.25)


def sha_arr(a):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a, np.int64))
                        .tobytes()).hexdigest()


def pick_rows(rows, task, episode, step, split_id, limit):
    """Детерминированный отбор панели. -> {задача: [row_id, …]}."""
    rows = np.asarray(rows)
    tasks = sorted({str(t) for t in np.asarray(task)[rows]})
    out = {}
    for ti, t in enumerate(tasks):
        rng = np.random.default_rng(np.random.SeedSequence(
            [35, 20261114, int(split_id), ti]))
        mt = rows[np.asarray([str(x) for x in np.asarray(task)[rows]]) == t]
        eps = np.unique(np.asarray(episode)[mt])
        eps = eps[rng.permutation(len(eps))]
        lists = []
        for e in eps:
            r = mt[np.asarray(episode)[mt] == e]
            r = r[np.argsort(np.asarray(step)[r], kind="stable")]
            lists.append(list(r[rng.permutation(len(r))]))
        sel, k = [], 0
        while len(sel) < limit and any(k < len(x) for x in lists):
            for x in lists:
                if k < len(x) and len(sel) < limit:
                    sel.append(int(x[k]))
            k += 1
        out[t] = sel
    return out


def make_panels(D):
    task, ep = D["task"], D["episode"]
    step = D.get("step", np.zeros(len(task), np.int64))
    tr = np.asarray(D["train_rows"])
    va = np.asarray(D["val_rows"])
    if set(np.asarray(ep)[tr].tolist()) & set(np.asarray(ep)[va].tolist()):
        raise SystemExit("train и val_sel делят эпизоды")
    p_tr = pick_rows(tr, task, ep, step, SPLIT_ID["oracle_train"],
                     LIMITS["oracle_train"])
    p_va = pick_rows(va, task, ep, step, SPLIT_ID["oracle_val"],
                     LIMITS["oracle_val"])
    p_ck = {t: v[:LIMITS["checkpoint_val"]] for t, v in p_va.items()}
    out = {}
    for nm, p in (("oracle_train", p_tr), ("oracle_val", p_va),
                  ("checkpoint_val", p_ck)):
        flat = [r for t in sorted(p) for r in p[t]]
        if len(flat) != len(set(flat)):
            raise SystemExit(f"{nm}: дубликаты row_id")
        out[nm] = dict(by_task=p, rows=flat, sha1=sha_arr(flat),
                       n_tasks=len(p), n_rows=len(flat),
                       n_episodes=len({int(ep[r]) for r in flat}),
                       per_task={t: len(v) for t, v in p.items()})
    out["rule"] = dict(limits=LIMITS, split_id=SPLIT_ID,
                       rng="SeedSequence([35, 20261114, split_id, task_index])",
                       order="episodes shuffled, rows within episode shuffled,"
                             " round robin")
    return out


def task_mean(values, rows, task):
    """Среднее с равными весами задач (внутри задачи — строк)."""
    values = np.asarray(values, np.float64)
    t = np.asarray([str(task[r]) for r in rows])
    return float(np.mean([values[t == u].mean(0) for u in sorted(set(t))],
                         0)) if values.ndim == 1 else \
        np.mean([values[t == u].mean(0) for u in sorted(set(t))], 0)


def row_costs(torch, km, out, tgt, val):
    """-> dict(full, arm20, arm8, grip) [B] float64."""
    full, l1, bce = km.loss_fn(torch, out, tgt, val)
    v = val.float()
    e = (out["arm"].float() - tgt[..., :6]).abs().mean(-1)
    v8 = v[:, :8]
    arm8 = (e[:, :8] * v8).sum(1) / v8.sum(1).clamp_min(1)
    return dict(full=full.double().cpu().numpy(),
                arm20=l1.double().cpu().numpy(),
                arm8=arm8.double().cpu().numpy(),
                grip=bce.double().cpu().numpy())


def oracle_rows(torch, km, mod, bar, D, rows, dev, dt, micro, mb,
                tol_feat=0.0):
    """36 прореженных forward на строку -> матрицы и признаки."""
    rows = np.asarray(rows)
    n = len(rows)
    C = {k: np.full((n, 36), np.nan) for k in ("full", "arm20", "arm8",
                                               "grip")}
    roots = iv_root = None
    mids, iv_mid = None, None
    mod.eval()
    with torch.no_grad():
        for s0 in range(0, n, micro):
            r = rows[s0:s0 + micro]
            batch, po, st, tgt, val = mb(D, r, bar, dev)
            for c in range(36):
                with torch.autocast("cuda", dtype=dt,
                                    enabled=dev.type == "cuda"):
                    o = mod(batch, po, st, cuts=[c] * len(r))
                rc = row_costs(torch, km, o, tgt, val)
                for k in C:
                    C[k][s0:s0 + len(r), c] = rc[k]
                sn = o["snap"]
                rn = sn["roots_n"].float().cpu().numpy()
                mn = sn["mids_n"].float().cpu().numpy()
                if roots is None:
                    H = rn.shape[-1]
                    roots = np.zeros((n, 4, H), np.float32)
                    iv_root = np.zeros((n, 4, 2), np.float32)
                    mids = np.zeros((n, 6, 4, H), np.float32)
                    iv_mid = np.zeros((n, 6, 4, 2), np.float32)
                    done_r = np.zeros(n, bool)
                    done_m = np.zeros((n, 6), bool)
                sl = slice(s0, s0 + len(r))
                a, b = divmod(c, 6)
                if not done_r[sl].all():
                    roots[sl] = rn
                    iv_root[sl] = sn["iv_root"].float().cpu().numpy()
                    done_r[sl] = True
                elif np.abs(roots[sl] - rn).max() > tol_feat:
                    raise SystemExit("признаки корней зависят от среза")
                if not done_m[sl, a].all():
                    mids[sl, a] = mn
                    iv_mid[sl, a] = sn["iv_mid"].float().cpu().numpy()
                    done_m[sl, a] = True
                elif np.abs(mids[sl, a] - mn).max() > tol_feat:
                    raise SystemExit("признаки средних зависят от b")
    for k, v in C.items():
        if not np.isfinite(v).all():
            raise SystemExit(f"оракул: нечисловые {k}")
    return dict(C=C, roots=roots, iv_root=iv_root, mids=mids, iv_mid=iv_mid)


def source_costs(torch, km, src, bar, D, rows, dev, dt, micro, mb):
    """L полного выхода 24 замороженного источника K-34 на тех же строках."""
    rows = np.asarray(rows)
    out = {k: np.zeros(len(rows)) for k in ("full", "arm20", "arm8", "grip")}
    src.eval()
    with torch.no_grad():
        for s0 in range(0, len(rows), micro):
            r = rows[s0:s0 + micro]
            batch, po, st, tgt, val = mb(D, r, bar, dev)
            with torch.autocast("cuda", dtype=dt, enabled=dev.type == "cuda"):
                o = src(batch, po, st, exit_depth=24)
            rc = row_costs(torch, km, o, tgt, val)
            for k in out:
                out[k][s0:s0 + len(r)] = rc[k]
    return out


def c_const_of(C_full_train, rows, task):
    m = task_mean(C_full_train, rows, task)              # [36]
    return int(np.flatnonzero(m == m.min())[0]), m


def gate_problems(Cv, rows_v, task, c_const, L_S):
    """-> (метрики, проблемы) гейта потенциала на oracle-val."""
    p = []
    full, arm8 = Cv["full"], Cv["arm8"]
    if full.shape[1] != 36 or not np.isfinite(full).all() or \
            not np.isfinite(arm8).all():
        return {}, ["оракул неполон или не конечен"]
    o = np.argmin(full, 1)                               # ничья — меньший c
    L_C = task_mean(full[:, c_const], rows_v, task)
    L_O = task_mean(full[np.arange(len(o)), o], rows_v, task)
    L_N = task_mean(full.mean(1), rows_v, task)
    a8_C = task_mean(arm8[:, c_const], rows_v, task)
    a8_O = task_mean(arm8[np.arange(len(o)), o], rows_v, task)
    gap = (L_C - L_O) / max(L_C, 1e-8)
    gain8 = (a8_C - a8_O) / max(a8_C, 1e-8)
    met = dict(L_C=L_C, L_O=L_O, L_N=L_N, L_S=L_S, gap_full=gap,
               arm8_C=a8_C, arm8_O=a8_O, arm8_gain=gain8, c_const=c_const,
               oracle_cut_freq=np.bincount(o, minlength=36).tolist())
    if gap < GATE_RULE["gap_full"]:
        p.append(f"gap_full {gap:.3f} < {GATE_RULE['gap_full']}")
    if gain8 < GATE_RULE["arm8_gain"]:
        p.append(f"arm8 улучшение {gain8:.3f} < {GATE_RULE['arm8_gain']}")
    if not L_C <= GATE_RULE["c_vs_s"] * L_S:
        p.append(f"L_C {L_C:.4f} > 1.25·L_S {L_S:.4f}")
    return met, p


def selftest():
    # панели: детерминизм, лимиты, без дубликатов и пересечений эпизодов
    n = 400
    rng = np.random.default_rng(0)
    task = np.array([f"t{i % 5}" for i in range(n)])
    episode = np.arange(n) // 4
    step = np.arange(n) % 4
    tr = np.arange(0, 300)
    va = np.arange(300, 400)
    D = dict(task=task, episode=episode, step=step, train_rows=tr,
             val_rows=va)
    p1, p2 = make_panels(D), make_panels(D)
    assert p1 == p2
    for nm, lim in (("oracle_train", 64), ("oracle_val", 16),
                    ("checkpoint_val", 8)):
        assert all(v <= lim for v in p1[nm]["per_task"].values())
        assert len(p1[nm]["rows"]) == len(set(p1[nm]["rows"]))
    assert set(p1["oracle_train"]["rows"]) <= set(tr.tolist())
    assert all(p1["checkpoint_val"]["by_task"][t] ==
               p1["oracle_val"]["by_task"][t][:8] for t in
               p1["checkpoint_val"]["by_task"])
    # обход по кругу: сначала по одной строке каждого эпизода
    t0 = p1["oracle_val"]["by_task"]["t0"]
    assert len({int(episode[r]) for r in t0[:5]}) == 5
    # изменение val не меняет train-панель
    D2 = dict(D, val_rows=va[:-20])
    assert make_panels(D2)["oracle_train"] == p1["oracle_train"]
    try:
        make_panels(dict(D, val_rows=np.arange(296, 400)))
        raise AssertionError("пересечение эпизодов принято")
    except SystemExit:
        pass
    # c_const: равные веса задач, ничья — меньший c
    rows = p1["oracle_train"]["rows"]
    Cf = rng.random((len(rows), 36))
    Cf[:, 9] -= 0.5
    Cf[:, 3] -= 0.5
    cc, m = c_const_of(Cf, rows, task)
    assert cc in (3, 9)
    Cf[:, 3] = Cf[:, 9]
    assert c_const_of(Cf, rows, task)[0] == 3
    # 19. одинаковые стоимости — нулевой запас; зависимость от состояния —
    # положительный; независимый от состояния шум — около нуля
    rv = p1["oracle_val"]["rows"]
    same = np.tile(rng.random(36), (len(rv), 1))
    met, _p = gate_problems(dict(full=same, arm8=same), rv, task, 0, 1.0)
    cbest = int(np.argmin(same[0]))
    met, _p = gate_problems(dict(full=same, arm8=same), rv, task, cbest, 1.0)
    assert abs(met["gap_full"]) < 1e-12
    dep = np.ones((len(rv), 36))
    for i in range(len(rv)):
        dep[i, i % 36] = 0.2
    met, pr = gate_problems(dict(full=dep, arm8=dep), rv, task, 0, 1.0)
    assert met["gap_full"] > 0.5 and pr == []
    met, pr = gate_problems(dict(full=dep, arm8=dep), rv, task, 0, 0.5)
    assert pr                                              # L_C > 1.25·L_S
    bad = dep.copy()
    bad[0, 0] = np.nan
    assert gate_problems(dict(full=bad, arm8=dep), rv, task, 0, 1.0)[1]
    print("самопроверка k35_oracle пройдена: панели детерминированы, "
          "лимиты, без дубликатов, обход по эпизодам, изменение val не "
          "меняет train, пересечение эпизодов — отказ; c_const с равными "
          "весами задач и ничьёй к меньшему c; одинаковые стоимости — "
          "нулевой запас, зависимость от состояния — положительный, NaN — "
          "отказ, L_C > 1.25·L_S — отказ")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; запуск — через k35_protocol.py")
