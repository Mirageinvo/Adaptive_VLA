#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-27: индекс последовательностей наблюдений с шагом 8 из подготовки K-21.

Ключ строки — (task, episode, step); строки НЕ считаются последовательными
по номеру массива. Отдельно для train и val_sel: текущая строка t
допустима, если в том же разбиении, задаче и эпизоде есть точно step − 8;
если есть и step − 16 — цепочка из трёх наблюдений, иначе из двух. Без
интерполяции, ближайших кадров, переходов через reset или между задачами.
Повтор ключа — отказ. Подготовка K-21 не переписывается, кадры не
копируются: сохраняются номера строк и SHA индекса.

prepare -> data/k27/sequences.npz, reports/k27/sequences.json: цепочки из
2/3 наблюдений по разбиениям, покрытие по задачам и эпизодам, доля строк с
историей, train-задачи без единой пары (с причиной), панель val (до 512
цепочек, заранее по задачам -> эпизодам), 64 train-цепочки для fitcheck с
покрытием задач.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

STRIDE = 8
PANEL_N, FIT_N = 512, 64
PANEL_SEED, FIT_SEED = 20261028, 20261029
OUT = "data/k27/sequences.npz"
REPORT = "reports/k27/sequences.json"


def build_index(task, episode, step, rows, stride=STRIDE):
    """-> dict(cur, prev1, prev2) (номера строк; prev2 = −1 при цепочке из
    двух наблюдений)."""
    task, episode, step = (np.asarray(x) for x in (task, episode, step))
    rows = np.sort(np.asarray(rows, np.int64))
    key = {}
    for r in rows:
        k = (str(task[r]), int(episode[r]), int(step[r]))
        if k in key:
            raise SystemExit(f"повтор ключа {k}: строки {key[k]} и {r}")
        key[k] = int(r)
    cur, p1, p2 = [], [], []
    for r in rows:
        t, e, s = str(task[r]), int(episode[r]), int(step[r])
        a = key.get((t, e, s - stride))
        if a is None:
            continue
        b = key.get((t, e, s - 2 * stride))
        cur.append(int(r))
        p1.append(a)
        p2.append(-1 if b is None else b)
    return dict(cur=np.asarray(cur, np.int64), prev1=np.asarray(p1, np.int64),
                prev2=np.asarray(p2, np.int64))


def check_index(ix, task, episode, step, rows, stride=STRIDE):
    """Независимая проверка: те же задача/эпизод, точный шаг, строки
    только своего разбиения, нет повтора текущих строк."""
    task, episode, step = (np.asarray(x) for x in (task, episode, step))
    ok = set(int(r) for r in rows)
    p = []
    if len(set(ix["cur"].tolist())) != len(ix["cur"]):
        p.append("повтор текущей строки")
    for c, a, b in zip(ix["cur"], ix["prev1"], ix["prev2"]):
        chain = [(c, 0), (a, stride)] + ([(b, 2 * stride)] if b >= 0 else [])
        for r, d in chain:
            if int(r) not in ok:
                p.append(f"строка {r} вне разбиения")
            elif str(task[r]) != str(task[c]) or episode[r] != episode[c] \
                    or int(step[r]) != int(step[c]) - d:
                p.append(f"цепочка {c}: строка {r} — другая задача/эпизод "
                         f"или шаг")
        if p:
            break
    return p


def coverage(ix, task, episode, rows):
    task, episode = np.asarray(task), np.asarray(episode)
    rows = np.asarray(rows)
    out = {}
    for t in sorted({str(x) for x in task[rows]}):
        rt = rows[np.asarray([str(x) == t for x in task[rows]])]
        m = np.asarray([str(task[c]) == t for c in ix["cur"]], bool)
        cur, p2 = ix["cur"][m], ix["prev2"][m]
        out[t] = dict(rows=int(len(rt)),
                      episodes=int(len(set(episode[rt].tolist()))),
                      chains=int(len(cur)), chains2=int((p2 < 0).sum()),
                      chains3=int((p2 >= 0).sum()),
                      episodes_with_chain=int(len(set(episode[cur].tolist()))),
                      share_rows_with_history=float(len(cur) / max(len(rt),
                                                                   1)))
    return out


def stratified(ix, task, episode, n, seed):
    """До n цепочек: задачи по кругу, внутри задачи — эпизоды по кругу,
    порядок фиксирован сидом (не по трудности/исходу)."""
    task, episode = np.asarray(task), np.asarray(episode)
    rng = np.random.default_rng(seed)
    by = {}
    for i, c in enumerate(ix["cur"]):
        by.setdefault(str(task[c]), {}).setdefault(int(episode[c]), []
                                                   ).append(i)
    queues = {}
    for t, eps in by.items():
        order = list(eps)
        rng.shuffle(order)
        lists = [list(rng.permutation(eps[e])) for e in order]
        q = []
        while any(lists):
            for lst in lists:
                if lst:
                    q.append(int(lst.pop()))
        queues[t] = q
    out, tasks = [], sorted(queues)
    while len(out) < n and any(queues[t] for t in tasks):
        for t in tasks:
            if queues[t] and len(out) < n:
                out.append(queues[t].pop(0))
    return np.asarray(sorted(out), np.int64)


def index_sha(ix):
    h = hashlib.sha1()
    for k in ("cur", "prev1", "prev2"):
        h.update(k.encode())
        h.update(np.ascontiguousarray(ix[k]).tobytes())
    return h.hexdigest()


def load(path=OUT, report=REPORT):
    """Сверка индекса с отчётом и подготовкой K-21."""
    import k21_data as d21
    import k21_protocol as pp
    if not os.path.exists(path) or not os.path.exists(report):
        raise SystemExit("prepare K-27 не выполнен")
    r = json.load(open(report, encoding="utf-8"))
    if pp.full_sha(path) != r["npz_sha1"]:
        raise SystemExit(f"{path} изменён после prepare")
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    if prep["prep_id"] != r["prep_id"]:
        raise SystemExit("индекс K-27 построен на другой подготовке K-21")
    z = dict(np.load(path))
    out = {}
    for sp in ("train", "val"):
        out[sp] = {k: z[f"{sp}_{k}"] for k in ("cur", "prev1", "prev2")}
        if index_sha(out[sp]) != r["splits"][sp]["index_sha1"]:
            raise SystemExit(f"индекс {sp} ≠ отчёту")
    out["panel"] = z["panel"]
    out["fit"] = z["fit"]
    out["report"] = r
    return out


def prepare():
    import k21_data as d21
    import k21_protocol as pp
    D = d21.load_prepared(require_teacher=False)
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    task, ep, st = D["task"], D["episode"], D["step"]
    rep = dict(created=datetime.datetime.now().isoformat(timespec="seconds"),
               prep_id=prep["prep_id"], prepare_sha1=pp.full_sha(d21.REPORT),
               stride=STRIDE, splits={})
    arrays = {}
    for sp, rows in (("train", D["train_rows"]), ("val", D["val_rows"])):
        ix = build_index(task, ep, st, rows)
        prob = check_index(ix, task, ep, st, rows)
        if prob:
            raise SystemExit(f"{sp}: " + "; ".join(prob[:3]))
        cov = coverage(ix, task, ep, rows)
        rep["splits"][sp] = dict(
            rows=int(len(rows)), chains=int(len(ix["cur"])),
            chains2=int((ix["prev2"] < 0).sum()),
            chains3=int((ix["prev2"] >= 0).sum()),
            share_rows_with_history=float(len(ix["cur"]) / max(len(rows), 1)),
            tasks=len(cov), tasks_without_pair=[t for t, c in cov.items()
                                                if c["chains"] == 0],
            index_sha1=index_sha(ix), per_task=cov)
        for k, v in ix.items():
            arrays[f"{sp}_{k}"] = v
        if sp == "train":
            tr_ix = ix
        else:
            va_ix = ix
    if set(arrays["train_cur"].tolist()) & set(arrays["val_cur"].tolist()):
        raise SystemExit("train и val пересекаются по текущим строкам")
    panel = stratified(va_ix, task, ep, PANEL_N, PANEL_SEED)
    fit = stratified(tr_ix, task, ep, FIT_N, FIT_SEED)
    arrays.update(panel=panel, fit=fit)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT + ".tmp.npz", **arrays)
    os.replace(OUT + ".tmp.npz", OUT)
    rep.update(npz=os.path.abspath(OUT), npz_sha1=pp.full_sha(OUT),
               panel=dict(n=int(len(panel)), seed=PANEL_SEED,
                          tasks=len({str(task[va_ix["cur"][i]])
                                     for i in panel})),
               fit=dict(n=int(len(fit)), seed=FIT_SEED,
                        tasks=len({str(task[tr_ix["cur"][i]]) for i in fit})))
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    json.dump(rep, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    for sp, r in rep["splits"].items():
        print(f"  {sp}: строк {r['rows']}, цепочек {r['chains']} (из 2 — "
              f"{r['chains2']}, из 3 — {r['chains3']}); доля строк с историей "
              f"{100 * r['share_rows_with_history']:.1f}%; задач {r['tasks']}"
              + (f"; БЕЗ ПАРЫ: {r['tasks_without_pair']}"
                 if r["tasks_without_pair"] else ""))
        low = sorted(r["per_task"].items(), key=lambda kv:
                     kv[1]["share_rows_with_history"])[:3]
        print("    меньше всего истории: " + "; ".join(
            f"{t[:40]}: {100 * c['share_rows_with_history']:.0f}% "
            f"({c['chains']} цеп., эпизодов {c['episodes_with_chain']}/"
            f"{c['episodes']})" for t, c in low))
    print(f"  панель val: {rep['panel']['n']} цепочек, задач "
          f"{rep['panel']['tasks']}; fitcheck: {rep['fit']['n']} цепочек, "
          f"задач {rep['fit']['tasks']}")
    print(f"  {OUT} (SHA {rep['npz_sha1'][:12]}), {REPORT}")
    miss = rep["splits"]["train"]["tasks_without_pair"]
    if miss:
        print(f"  ВНИМАНИЕ: train-задачи без единой пары t−8 -> t: {miss}; "
              f"обучение памяти для них невозможно (stride и состав задач "
              f"не меняются)")
    return 0


def selftest():
    # 3 задачи, по 2 эпизода; шаги с пропусками; строки перемешаны
    rows, task, ep, st = [], [], [], []
    r = 0
    for t in ("a", "b", "c"):
        for e in (0, 1):
            steps = [0, 8, 16, 24, 30, 40] if t != "c" else [0, 5, 13]
            for s in steps:
                task.append(t)
                ep.append(e + (10 if t == "b" else 0))
                st.append(s)
                rows.append(r)
                r += 1
    perm = np.random.default_rng(0).permutation(len(rows))
    task = np.asarray(task, object)[perm]
    ep, st = np.asarray(ep)[perm], np.asarray(st)[perm]
    tr = np.arange(len(rows))
    ix = build_index(task, ep, st, tr)
    assert check_index(ix, task, ep, st, tr) == []
    cov = coverage(ix, task, ep, tr)
    # a: шаги 8,16,24 имеют пары, 16 и 24 — тройки; 40 — нет (32 нет)
    assert cov["a"]["chains"] == 6 and cov["a"]["chains3"] == 4
    assert cov["c"]["chains"] == 2 and cov["c"]["chains3"] == 0  # 13 <- 5
    for c, a, b in zip(ix["cur"], ix["prev1"], ix["prev2"]):
        assert st[a] == st[c] - 8 and task[a] == task[c] and ep[a] == ep[c]
        if b >= 0:
            assert st[b] == st[c] - 16
    # строки другого разбиения не берутся
    half = tr[: len(tr) // 2]
    ixh = build_index(task, ep, st, half)
    assert set(ixh["prev1"].tolist()) <= set(half.tolist())
    # повтор ключа и неверный шаг ловятся
    st2 = st.copy()
    st2[1] = st2[0]
    ep2 = ep.copy()
    ep2[1], task2 = ep2[0], task.copy()
    task2[1] = task2[0]
    try:
        build_index(task2, ep2, st2, tr)
        raise AssertionError("повтор ключа принят")
    except SystemExit:
        pass
    bad = {k: v.copy() for k, v in ix.items()}
    bad["prev1"][0] = bad["cur"][0]
    assert check_index(bad, task, ep, st, tr)
    bad = {k: v.copy() for k, v in ix.items()}
    bad["prev1"][0] = [x for x in tr if task[x] != task[bad["cur"][0]]][0]
    assert check_index(bad, task, ep, st, tr)
    # панель: покрывает задачи, повторяема, не зависит от исходов
    p1 = stratified(ix, task, ep, 4, 1)
    p2 = stratified(ix, task, ep, 4, 1)
    assert np.array_equal(p1, p2) and len({task[ix["cur"][i]] for i in p1}) \
        == 3
    assert len(stratified(ix, task, ep, 999, 1)) == len(ix["cur"])
    assert index_sha(ix) != index_sha(bad)
    print("самопроверка k27_data пройдена: точные пары/тройки по ключу "
          "(задача, эпизод, шаг) при перемешанных строках и пропусках шагов, "
          "только своё разбиение, повтор ключа и неверная строка цепочки — "
          "отказ, покрытие по задачам, стратифицированная повторяемая панель")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-27: последовательности")
    ap.add_argument("cmd", choices=("prepare", "selftest"))
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    try:
        return prepare()
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
