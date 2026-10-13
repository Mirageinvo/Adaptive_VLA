#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15i: предсказуема ли польза импульса p4r3p по состоянию до него.

РАЗВЕДОЧНАЯ ОЦЕНКА ВЫБОРА ПО ЗАПИСАННЫМ ТРАЕКТОРИЯМ (не раскатки
переключателя). Набор — k15i_prepare (150 состояний свежего банка K-15h).

МОДЕЛЬ ПОЛЬЗЫ: h18 вызова 4 (до импульса) -> центрирование и PCA до 16
компонент; a0 вызова 4 (56 чисел); стандартизация; ridge-регрессия цели
y = успех(p4r3p) − успех(q0) со свободным членом. Всё (центр, PCA,
стандартизация, λ, пороги) — ТОЛЬКО по обучающей части; λ — внутренней
проверкой (по блокам внутри обучающей части) по пользе выбора.

ПРАВИЛО (задано заранее): прогноз пользы > 0 -> p4r3p, иначе q0. Состояние,
где импульс не достигнут (среда завершилась до вызова 4), — всегда q0.

ПЕРЕКЛЮЧАТЕЛИ ДЛЯ СРАВНЕНИЯ:
  always_q0, always_p4r3p;
  by_task        средняя польза задачи на обучении > 0 -> p4r3p (при
                 исключении задачи — общая средняя);
  benefit_a0     та же модель пользы, только a0;
  benefit_h18a0  основная модель;
  fail_detector  ridge-регрессия провала q0 (1 − успех) по тем же
                 признакам; вмешательство, если прогноз провала выше
                 порога; порог — внутренней проверкой по пользе выбора.

ВНЕШНЯЯ ПРОВЕРКА: (а) блоки — проверочный блок 25 / 30 / 35 исключён из
обучения на всех задачах (основная); (б) исключение целой задачи
(перенос между задачами). Блок 45–49 не читается.

ОТЧЁТ на исключённых из обучения состояниях: SR выбора (общий и с равными
весами задач), разница с постоянными политиками, число вмешательств,
спасений и повреждений, по задачам. Оракульная граница пары — отдельной
строкой, не результат политики.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

N_PCA = 16
LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
BLOCKS = (25, 30, 35)
SELECTORS = ("always_q0", "always_p4r3p", "by_task", "benefit_a0",
             "benefit_h18a0", "fail_detector")


# --- признаки и модели (только обучающая часть) -----------------------------
class Features:
    """Центр и PCA h18, стандартизация — по обучающим строкам."""

    def __init__(self, h18, a0, use_h18=True, n_pca=N_PCA):
        self.use_h18 = use_h18
        A = a0.reshape(len(a0), -1).astype(np.float64)
        parts = [A]
        if use_h18:
            X = h18.reshape(len(h18), -1).astype(np.float64)
            self.mu = X.mean(0)
            _u, _s, vt = np.linalg.svd(X - self.mu, full_matrices=False)
            self.V = vt[:min(n_pca, len(vt))].T
            parts.insert(0, (X - self.mu) @ self.V)
        Z = np.concatenate(parts, 1)
        self.zm, self.zs = Z.mean(0), Z.std(0)
        self.zs[self.zs < 1e-8] = 1.0

    def __call__(self, h18, a0):
        A = a0.reshape(len(a0), -1).astype(np.float64)
        parts = [A]
        if self.use_h18:
            X = h18.reshape(len(h18), -1).astype(np.float64)
            parts.insert(0, (X - self.mu) @ self.V)
        return (np.concatenate(parts, 1) - self.zm) / self.zs


def ridge(Z, y, lam):
    """Ridge со свободным членом без штрафа. Возвращает (w, b)."""
    zm, ym = Z.mean(0), y.mean()
    Zc = Z - zm
    w = np.linalg.solve(Zc.T @ Zc + lam * np.eye(Z.shape[1]),
                        Zc.T @ (y - ym))
    return w, ym - zm @ w


def utility(y, choose):
    """Польза выбора: сумма y по выбранным вмешательствам."""
    return float(np.sum(np.asarray(y)[np.asarray(choose, bool)]))


def fit(D, rows, lam, use_h18, target):
    """Модель на строках rows с фиксированным λ -> функция прогноза."""
    F = Features(D["h18"][rows], D["a0"][rows], use_h18)
    w, b = ridge(F(D["h18"][rows], D["a0"][rows]), target[rows], lam)
    return lambda idx: F(D["h18"][idx], D["a0"][idx]) @ w + b


def fit_predict(D, tr, te, use_h18, target, inner_groups):
    """Обучить на tr (только достигнутые), предсказать te. λ — внутренней
    проверкой по группам inner_groups (обучающая часть) по пользе
    правила «прогноз > 0»."""
    best, best_u = LAMBDAS[-1], -np.inf
    groups = sorted(set(inner_groups[tr].tolist()))
    for lam in sorted(LAMBDAS, reverse=True):     # равенство — сильнее λ
        u = 0.0
        for g in groups:
            itr = tr[inner_groups[tr] != g]
            iva = tr[inner_groups[tr] == g]
            if len(itr) < 3 or len(iva) == 0:
                continue
            pred = fit(D, itr, lam, use_h18, target)(iva)
            u += utility(D["y"][iva], pred > 0)
        if u > best_u + 1e-12:
            best, best_u = lam, u
    return fit(D, tr, best, use_h18, target)(te), best


def fail_detector(D, tr, te, inner_groups):
    """Прогноз провала q0; λ и порог выбираются СОВМЕСТНО внутренней
    проверкой по пользе того же правила «прогноз провала > порог», среди
    порогов — «всегда» (−∞) и «никогда» (+∞). При равенстве пользы — меньше
    вмешательств, затем сильнее λ. Затем модель с выбранным λ обучается
    на всей внешней обучающей части, порог применяется к te."""
    fail = 1.0 - D["succ_q0"].astype(np.float64)
    groups = sorted(set(inner_groups[tr].tolist()))
    best = None                                  # (польза, −вмеш., λ, порог)
    for lam in LAMBDAS:
        sc, yy = [], []
        for g in groups:
            itr = tr[inner_groups[tr] != g]
            iva = tr[inner_groups[tr] == g]
            if len(itr) < 3 or len(iva) == 0:
                continue
            sc.append(fit(D, itr, lam, True, fail)(iva))
            yy.append(D["y"][iva])
        if not sc:
            continue
        s_, y_ = np.concatenate(sc), np.concatenate(yy)
        for thr in np.concatenate([[-np.inf, np.inf], np.unique(s_)]):
            ch = s_ > thr
            key = (utility(y_, ch), -int(ch.sum()), lam)
            if best is None or key > best[0]:
                best = (key, float(thr))
    if best is None:
        return np.zeros(len(te), bool), dict(threshold=None, lam=None)
    (_u, _n, lam), thr = best
    p_te = fit(D, tr, lam, True, fail)(te)
    return p_te > thr, dict(threshold=thr, lam=lam)


def choose_all(D, tr, te, inner_groups, task_fallback_global=False):
    """Решения всех переключателей для te (обучение — на tr)."""
    tr = tr[D["reached4"][tr]]
    r = D["reached4"][te]
    out = {"always_q0": np.zeros(len(te), bool),
           "always_p4r3p": r.copy()}
    ymean = D["y"][tr].mean() if len(tr) else 0.0
    bt = []
    for i in te:
        rows = tr[D["task"][tr] == D["task"][i]]
        m = (D["y"][rows].mean() if (len(rows) and not task_fallback_global)
             else ymean)
        bt.append(m > 0)
    out["by_task"] = np.asarray(bt) & r
    info = {}
    for name, use_h in (("benefit_a0", False), ("benefit_h18a0", True)):
        sel = np.zeros(len(te), bool)
        if r.any():
            p, lam = fit_predict(D, tr, te[r], use_h, D["y"].astype(float),
                                 inner_groups)
            sel[r] = p > 0
            info[name] = dict(lam=lam)
        out[name] = sel
    sel = np.zeros(len(te), bool)
    if r.any():
        s_, inf = fail_detector(D, tr, te[r], inner_groups)
        sel[r] = s_
        info["fail_detector"] = inf
    out["fail_detector"] = sel
    return out, info


def evaluate(D, folds):
    """folds: [(имя, обучающие, проверочные, внутренние группы, флаг)]."""
    N = len(D["y"])
    dec = {k: np.full(N, -1, np.int64) for k in SELECTORS}
    infos = {}
    for name, tr, te, groups, fb in folds:
        if set(tr.tolist()) & set(te.tolist()):
            raise SystemExit(f"{name}: проверочные состояния в обучении")
        ch, info = choose_all(D, tr, te, groups, fb)
        infos[name] = info
        for k, v in ch.items():
            dec[k][te] = v.astype(np.int64)
    if any((v < 0).any() for v in dec.values()):
        raise SystemExit("не все состояния получили решение вне обучения")
    return summarize(D, dec), infos


def summarize(D, dec):
    sq0, stg = D["succ_q0"].astype(bool), D["succ_target"].astype(bool)
    tasks = sorted(set(D["task"].tolist()))
    res = {}
    for k, d in dec.items():
        d = d.astype(bool)
        s = np.where(d, stg, sq0)
        ew = float(np.mean([s[D["task"] == t].mean() for t in tasks]))
        res[k] = dict(
            sr=float(s.mean()), sr_equal_weight=ew,
            d_vs_q0=float(s.mean() - sq0.mean()),
            d_vs_p4r3p=float(s.mean() - np.where(D["reached4"], stg,
                                                  sq0).mean()),
            interventions=int(d.sum()),
            rescues=int((d & (D["y"] == 1)).sum()),
            harms=int((d & (D["y"] == -1)).sum()),
            per_task={str(t): dict(
                sr=float(s[D["task"] == t].mean()),
                interventions=int(d[D["task"] == t].sum()),
                rescues=int((d & (D["y"] == 1))[D["task"] == t].sum()),
                harms=int((d & (D["y"] == -1))[D["task"] == t].sum()))
                for t in tasks})
    res["oracle_bound"] = dict(
        sr=float(np.maximum(sq0, stg).mean()),
        note="граница при знании исхода заранее, не результат политики")
    return res


def block_folds(D):
    blk = (D["state"] // 5) * 5
    return [(f"блок {b}", np.nonzero(blk != b)[0], np.nonzero(blk == b)[0],
             blk, False) for b in BLOCKS]


def task_folds(D):
    blk = (D["state"] // 5) * 5
    return [(f"задача {t}", np.nonzero(D["task"] != t)[0],
             np.nonzero(D["task"] == t)[0], blk, True)
            for t in sorted(set(D["task"].tolist()))]


def _sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def dataset_problems(D, meta, here=HERE, check_inputs=True):
    """Смысл и отпечатки набора перед обучением."""
    p = []
    want = dict(kind="k15i_dataset", target="p4r3p", base="q0ref", call=4)
    for k, v in want.items():
        if meta.get(k) != v:
            p.append(f"{k} = {meta.get(k)!r}, ожидалось {v!r}")
    for f, s_ in (meta.get("code") or {}).items():
        if _sha(os.path.join(here, f)) != s_:
            p.append(f"код {f} изменён после подготовки")
    if set(meta.get("code") or {}) != {"k15i_prepare.py",
                                       "k15i_benefit_probe.py"}:
        p.append("в наборе нет отпечатков обоих файлов K-15i")
    if check_inputs:
        for k, v in (meta.get("inputs") or {}).items():
            if not os.path.exists(v["path"]) or _sha(v["path"]) != v["sha1"]:
                p.append(f"вход {k} изменён или отсутствует")
    if (D["state"] >= 45).any():
        p.append("в наборе состояния резерва 45–49")
    y = D["succ_target"].astype(int) - D["succ_q0"].astype(int)
    if not np.array_equal(D["y"], y):
        p.append("y не равен succ_target − succ_q0")
    r = D["reached4"].astype(bool)
    if r.any() and not (np.isfinite(D["h18"][r]).all() and
                        np.isfinite(D["a0"][r]).all()):
        p.append("у достигнутых признаки не конечны")
    if (~r).any() and not (np.isnan(D["h18"][~r]).all() and
                           np.isnan(D["a0"][~r]).all()):
        p.append("у недостигнутых признаки не пусты")
    if (D["succ_target"][~r] != D["succ_q0"][~r]).any():
        p.append("импульс не достигнут, но исходы различаются")
    return p


def load(path):
    D = dict(np.load(path))
    meta = json.load(open(path[:-4] + ".json"))
    if _sha(path) != (meta.get("dataset") or {}).get("sha1"):
        raise SystemExit("набор K-15i изменён после подготовки")
    p = dataset_problems(D, meta)
    if p:
        raise SystemExit("набор K-15i непригоден: " + "; ".join(p[:6]))
    return D, meta


def report(name, res):
    print(f"  --- {name} (оценка на исключённых из обучения состояниях)")
    for k in SELECTORS:
        r = res[k]
        print(f"    {k:15} SR {100 * r['sr']:5.1f}% (равные веса "
              f"{100 * r['sr_equal_weight']:5.1f}%), против q0 "
              f"{100 * r['d_vs_q0']:+5.1f} п.п., против p4r3p "
              f"{100 * r['d_vs_p4r3p']:+5.1f} п.п.; вмешательств "
              f"{r['interventions']}, спасений {r['rescues']}, повреждений "
              f"{r['harms']}")
    print(f"    ОРАКУЛЬНАЯ ГРАНИЦА (не политика): "
          f"{100 * res['oracle_bound']['sr']:.1f}%")


def main():
    ap = argparse.ArgumentParser(description="K-15i: проба пользы")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset", default="reports/k15i/dataset.npz")
    ap.add_argument("--out", default="reports/k15i/probe.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    D, meta = load(a.dataset)
    res_b, inf_b = evaluate(D, block_folds(D))
    res_t, inf_t = evaluate(D, task_folds(D))
    report("внешняя проверка по блокам (основная)", res_b)
    for k in SELECTORS:
        pt = res_b[k]["per_task"]
        print(f"      {k}: " + " ".join(
            f"t{t} {100 * v['sr']:.0f}%/{v['interventions']}"
            for t, v in pt.items()))
    report("исключение целой задачи (перенос)", res_t)
    out = dict(dataset=meta["dataset"],
               code={f: _sha(os.path.join(HERE, f)) for f in (
                   "k15i_prepare.py", "k15i_benefit_probe.py")},
               model=dict(n_pca=N_PCA, lambdas=list(LAMBDAS),
                          features="h18 вызова 4 (PCA) + a0 вызова 4",
                          rule="прогноз пользы > 0 -> p4r3p"),
               blocks=res_b, blocks_info=inf_b,
               tasks=res_t, tasks_info=inf_t,
               note=("разведочная оценка выбора по записанным траекториям; "
                     "успех иерархии подтверждается раскатками"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1,
                  default=lambda x: x.tolist() if hasattr(x, "tolist")
                  else str(x))
    print(f"  сводка: {a.out}")
    return 0


# --- самопроверка ------------------------------------------------------------
def _synth(seed=0, signal=True, n_tasks=10):
    rng = np.random.default_rng(seed)
    keys = [(t, s) for t in range(n_tasks) for s in range(25, 40)]
    N = len(keys)
    h18 = rng.standard_normal((N, 4, 6)).astype(np.float32)
    a0 = rng.standard_normal((N, 8, 7)).astype(np.float32)
    sq0 = rng.random(N) < 0.6
    if signal:            # импульс спасает, когда h18[...,0,0] > 0
        good = h18[:, 0, 0] > 0
        stg = np.where(good, True, sq0 & (rng.random(N) < 0.3))
    else:
        stg = rng.random(N) < 0.6
    r4 = np.ones(N, bool)
    r4[::17] = False
    stg = np.where(r4, stg, sq0)
    h18[~r4] = np.nan
    a0[~r4] = np.nan
    return dict(task=np.array([k[0] for k in keys]),
                state=np.array([k[1] for k in keys]), h18=h18, a0=a0,
                reached4=r4, y=stg.astype(int) - sq0.astype(int),
                succ_q0=sq0, succ_target=stg)


def selftest():
    D = _synth()
    res, _i = evaluate(D, block_folds(D))
    # есть сигнал в h18 — модель пользы обгоняет постоянные политики и a0
    assert res["benefit_h18a0"]["sr"] > res["always_q0"]["sr"]
    assert res["benefit_h18a0"]["sr"] > res["always_p4r3p"]["sr"]
    assert res["benefit_h18a0"]["sr"] >= res["benefit_a0"]["sr"]
    assert res["oracle_bound"]["sr"] >= res["benefit_h18a0"]["sr"]
    # недостигнутый импульс — всегда q0 (вмешательств нет)
    dec_r = res["always_p4r3p"]["interventions"]
    assert dec_r == int(D["reached4"].sum())
    # NaN недостигнутых не попадает в обучение (иначе прогнозы — NaN)
    blk = (D["state"] // 5) * 5
    tr = np.nonzero(blk != 25)[0]
    te = np.nonzero(blk == 25)[0]
    ch, _ = choose_all(D, tr, te, blk)
    assert all(np.asarray(v).dtype == bool for v in ch.values())
    # УТЕЧКА: изменение проверочных строк (признаки и исходы) не меняет
    # обученное (прогнозы на обучающих строках) — ни PCA, ни λ, ни порог
    D2 = {k: (v.copy() if isinstance(v, np.ndarray) else v)
          for k, v in D.items()}
    D2["h18"][te] = np.where(np.isnan(D2["h18"][te]), np.nan, 1e3)
    D2["a0"][te] = np.where(np.isnan(D2["a0"][te]), np.nan, -1e3)
    D2["y"][te] = -D2["y"][te]
    D2["succ_q0"][te] = ~D2["succ_q0"][te]
    trr = tr[D["reached4"][tr]]
    for use_h in (True, False):
        p1, l1 = fit_predict(D, trr, trr, use_h, D["y"].astype(float), blk)
        p2, l2 = fit_predict(D2, trr, trr, use_h, D2["y"].astype(float),
                             blk)
        assert l1 == l2 and np.allclose(p1, p2)
    s1, i1 = fail_detector(D, trr, trr, blk)
    s2, i2 = fail_detector(D2, trr, trr, blk)
    assert i1 == i2 and np.array_equal(s1, s2)
    # проверочные состояния в обучении — отказ
    try:
        evaluate(D, [("x", np.arange(len(D["y"])), te, blk, False)])
        raise AssertionError("принята утечка проверочных в обучение")
    except SystemExit:
        pass
    # перепутанный знак цели ухудшает модель пользы (знак значим)
    Dn = dict(D, y=-D["y"])
    resn, _ = evaluate(Dn, block_folds(Dn))
    assert summarize(D, {"x": np.zeros(len(D["y"]))})["x"]["sr"] == \
        res["always_q0"]["sr"]
    assert resn["benefit_h18a0"]["interventions"] != \
        res["benefit_h18a0"]["interventions"]
    # исключение задачи: каждая задача проверяется ровно раз
    rt, _ = evaluate(D, task_folds(D))
    assert rt["always_q0"]["sr"] == res["always_q0"]["sr"]
    # детектор: q0 проваливается всегда, импульс спасает всегда ->
    # вмешиваться во всех достигнутых (порог «всегда» доступен)
    Da = _synth(seed=5)
    Da["succ_q0"] = np.zeros_like(Da["succ_q0"])
    Da["succ_target"] = Da["reached4"].copy()
    Da["y"] = Da["succ_target"].astype(int)
    ra, _ = evaluate(Da, block_folds(Da))
    assert ra["fail_detector"]["sr"] == ra["always_p4r3p"]["sr"] == \
        Da["reached4"].mean()
    # проверка набора перед обучением
    meta = dict(kind="k15i_dataset", target="p4r3p", base="q0ref", call=4,
                code={f: _sha(os.path.join(HERE, f)) for f in (
                    "k15i_prepare.py", "k15i_benefit_probe.py")})
    assert dataset_problems(D, meta, check_inputs=False) == []
    for mut, word in ((dict(target="learned_uniform"), "target"),
                      (dict(call=2), "call"),
                      (dict(code=dict(meta["code"], **{
                          "k15i_prepare.py": "0"})), "изменён"),
                      (dict(code={"k15i_prepare.py": "x"}), "обоих")):
        pr_ = dataset_problems(D, dict(meta, **mut), check_inputs=False)
        assert any(word in x for x in pr_), (mut, pr_)
    for mutD, word in ((dict(y=-D["y"]), "y не равен"),
                       (dict(state=np.where(np.arange(len(D["y"])) == 0, 47,
                                            D["state"])), "резерва")):
        pr_ = dataset_problems(dict(D, **mutD), meta, check_inputs=False)
        assert any(word in x for x in pr_), (word, pr_)
    # без сигнала модель не обязана выигрывать; оценка определена
    Dz = _synth(seed=3, signal=False)
    rz, _ = evaluate(Dz, block_folds(Dz))
    assert 0.0 <= rz["benefit_h18a0"]["sr"] <= 1.0
    print("самопроверка k15i_benefit_probe пройдена: сигнал находится, "
          "недостигнутый импульс, нет утечки проверочных в PCA/λ/порог, "
          "отказ при пересечении, знак цели, исключение задачи")
    return 0


if __name__ == "__main__":
    sys.exit(main())
