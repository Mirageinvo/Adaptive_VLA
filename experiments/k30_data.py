#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-30, пакет A: этап кодека на данных K-21 (CPU) и загрузка меток.

codec — только экспертный train строит словарь (k30_codec.fit); val_sel и
  teacher (teacher_target, маска — единицы на 20 позициях) проходят через
  уже замороженный float32 P. Разрешённые индексы — train_rows, val_rows,
  teacher_rows ⊆ train_rows (как в K-21); val_confirm и старты 45–49 не
  читаются. Артефакты: data/k30/root_codec.npz, data/k30/root_labels.npz,
  reports/k30/codec.json (один отчёт: размеры и SHA, objective, MSE и MAE,
  использование кодов, perplexity, сравнение с постоянным прототипом, доли
  clipping, покрытие задач val_sel, допуск).
  Готовые артефакты не перезаписываются: повтор пересчитывает и требует
  побитового совпадения SHA.
load() — словарь и метки со сверкой SHA; индексы: expert по row_id
(train ∪ val), teacher по teacher_idx (позиция в teacher_rows).
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k30_codec as kc  # noqa: E402

CODEC = "data/k30/root_codec.npz"
LABELS = "data/k30/root_labels.npz"
REPORT = "reports/k30/codec.json"


def sha(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def build(D):
    """D — подготовка K-21 (с учителем) -> (codec, labels, отчёт)."""
    import k21_train as kt
    tr = np.asarray(D["train_rows"])
    va = np.asarray(D["val_rows"])
    te = np.asarray(D["teacher_rows"])
    if not np.isin(te, tr).all():
        raise SystemExit("teacher_rows ⊄ train_rows")
    if np.intersect1d(tr, va).size:
        raise SystemExit("train и val_sel пересекаются")
    L, U = D["L"], D["U"]
    b_tr, rv_tr, cf_tr = kc.plan_targets(D["act"][tr], D["valid"][tr], L, U)
    w_tr = kt.row_weights(D["task"][tr], D["episode"][tr])
    f = kc.fit(b_tr, rv_tr, w_tr, tr)
    P32, mu, s = f["P32"], f["mu"], f["s"]
    id_tr = kc.labels(b_tr, rv_tr, P32, s)
    b_va, rv_va, cf_va = kc.plan_targets(D["act"][va], D["valid"][va], L, U)
    w_va = kt.row_weights(D["task"][va], D["episode"][va])
    id_va = kc.labels(b_va, rv_va, P32, s)
    tt = np.asarray(D["teacher_target"])
    b_te, rv_te, cf_te = kc.plan_targets(tt, np.ones(tt.shape[:2]), L, U)
    id_te = kc.labels(b_te, rv_te, P32, s)
    codec = dict(P32=P32, P64=f["P64"], mu=mu, std=f["std"], s=s,
                 objective=np.asarray(f["objective"]),
                 sha=kc.sha_arr(P32, mu, s))
    lab = dict(train_rows=tr, b_train=b_tr, rv_train=rv_tr, id_train=id_tr,
               val_rows=va, b_val=b_va, rv_val=rv_va, id_val=id_va,
               teacher_rows=te, teacher_idx=np.arange(len(te)), b_teacher=b_te,
               rv_teacher=rv_te, id_teacher=id_te)
    m_tr = kc.metrics(b_tr, rv_tr, id_tr, w_tr, P32, mu, s)
    m_va = kc.metrics(b_va, rv_va, id_va, w_va, P32, mu, s)
    m_te = kc.metrics(b_te, rv_te, id_te, np.ones(len(te)), P32, mu, s)
    tasks_tr = sorted(set(str(x) for x in np.asarray(D["task"])[tr]))
    tasks_va = sorted(set(str(x) for x in np.asarray(D["task"])[va]))
    gate = m_va["ratio"] is not None and m_va["ratio"] <= kc.GATE
    rep = dict(
        config=kc.CONFIG, k21_prep_id=D["prep_report"]["prep_id"],
        rows=dict(train=int(len(tr)), val=int(len(va)), teacher=int(len(te))),
        valid_roots=dict(train=int(rv_tr.sum()), val=int(rv_va.sum()),
                         teacher=int(rv_te.sum())),
        clip_frac=dict(train=cf_tr, val=cf_va, teacher=cf_te),
        mu=mu.tolist(), std=f["std"].tolist(), s=s.tolist(),
        objective=f["objective"], iters=f["iters"], reseeded=f["reseeded"],
        metrics=dict(train=m_tr, val=m_va, teacher=m_te),
        coverage=dict(train_tasks=len(tasks_tr), val_tasks=len(tasks_va),
                      val_missing=[t for t in tasks_tr if t not in tasks_va]),
        gate=dict(rule=f"val MSE ≤ {kc.GATE} · MSE постоянного прототипа",
                  ratio=m_va["ratio"], passed=bool(gate)),
        codec_sha=codec["sha"])
    return codec, lab, rep


def labels_sha(lab):
    return kc.sha_arr(*(lab[k] for k in sorted(lab)))


def codec_stage():
    import k21_data as d21
    D = d21.load_prepared(require_teacher=True)
    codec, lab, rep = build(D)
    lsha = labels_sha(lab)
    if os.path.exists(CODEC) or os.path.exists(LABELS):
        z = dict(np.load(CODEC))
        zl = dict(np.load(LABELS))
        if str(z["sha"]) != codec["sha"] or labels_sha(
                {k: v for k, v in zl.items()}) != lsha:
            raise SystemExit("кодек/метки уже закреплены и не совпадают с "
                             "пересчётом — изменение требует нового каталога")
        print(f"  {CODEC}, {LABELS}: закреплены, пересчёт совпал побитово")
    else:
        os.makedirs(os.path.dirname(CODEC), exist_ok=True)
        for path, d in ((CODEC, codec), (LABELS, lab)):
            with open(path + ".tmp", "wb") as fh:
                np.savez(fh, **d)
            os.replace(path + ".tmp", path)
    rep.update(codec_file=os.path.abspath(CODEC), codec_file_sha1=sha(CODEC),
               labels_file=os.path.abspath(LABELS),
               labels_file_sha1=sha(LABELS), labels_sha=lsha,
               code={f: sha(os.path.join(HERE, f))
                     for f in ("k30_codec.py", "k30_data.py")})
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(REPORT + ".tmp", REPORT)
    m = rep["metrics"]
    print(f"  строк train {rep['rows']['train']}, val {rep['rows']['val']}, "
          f"teacher {rep['rows']['teacher']}; действительных корней "
          f"{rep['valid_roots']}; clipping u {rep['clip_frac']}")
    print(f"  словарь K = {kc.K}: s {np.round(rep['s'], 3).tolist()}, Lloyd "
          f"{rep['iters']} итераций, пересоздано пустых {rep['reseeded']}")
    for k in ("train", "val", "teacher"):
        print(f"    {k}: MSE {m[k]['mse']:.4f} / постоянный "
              f"{m[k]['mse_const']:.4f} = {m[k]['ratio']:.3f}; кодов "
              f"{m[k]['codes_used']}, perplexity {m[k]['perplexity']:.1f}")
    print(f"  покрытие val_sel: {rep['coverage']['val_tasks']} из "
          f"{rep['coverage']['train_tasks']} задач")
    print("  ДОПУСК: " + ("пройден" if rep["gate"]["passed"] else
                          "НЕ пройден — обучение VLA не запускается") +
          f" (отношение val {rep['gate']['ratio']}); {REPORT}")
    return 0 if rep["gate"]["passed"] else 3


def load():
    """-> dict(P32, mu, s, sha, expert: row_id -> (b, rv, id), teacher)."""
    if not (os.path.exists(CODEC) and os.path.exists(LABELS) and
            os.path.exists(REPORT)):
        raise SystemExit("кодек K-30 не подготовлен: k30_data.py codec")
    rep = json.load(open(REPORT, encoding="utf-8"))
    if sha(CODEC) != rep["codec_file_sha1"] or \
            sha(LABELS) != rep["labels_file_sha1"]:
        raise SystemExit("кодек/метки изменены после подготовки")
    z = dict(np.load(CODEC))
    pr = kc.codec_problems(z)
    if pr:
        raise SystemExit("; ".join(pr))
    lab = dict(np.load(LABELS))
    if labels_sha(lab) != rep["labels_sha"]:
        raise SystemExit("метки изменены")
    rows = np.concatenate([lab["train_rows"], lab["val_rows"]])
    pos = np.full(int(rows.max()) + 1, -1, np.int64)
    pos[rows] = np.arange(len(rows))
    return dict(P32=z["P32"], mu=z["mu"], s=z["s"], sha=str(z["sha"]),
                report=rep, pos=pos,
                b=np.concatenate([lab["b_train"], lab["b_val"]]),
                rv=np.concatenate([lab["rv_train"], lab["rv_val"]]),
                ids=np.concatenate([lab["id_train"], lab["id_val"]]),
                b_teacher=lab["b_teacher"], rv_teacher=lab["rv_teacher"],
                id_teacher=lab["id_teacher"],
                teacher_rows=lab["teacher_rows"])


def targets(C, rows=None, teacher_idx=None):
    """Метки батча: expert — по row_id; teacher — по teacher_idx (строки
    учителя сверяются)."""
    if teacher_idx is not None:
        ti = np.asarray(teacher_idx)
        if rows is not None and not np.array_equal(
                C["teacher_rows"][ti], np.asarray(rows)):
            raise SystemExit("строки учителя ≠ меткам")
        return C["b_teacher"][ti], C["rv_teacher"][ti], C["id_teacher"][ti]
    r = np.asarray(rows)
    p = C["pos"][r] if r.max() < len(C["pos"]) else np.full(len(r), -1)
    if (p < 0).any():
        raise SystemExit("метки запрошены для неразрешённой строки")
    return C["b"][p], C["rv"][p], C["ids"][p]


def selftest():
    rng = np.random.default_rng(1)
    N = 400
    act = np.concatenate([rng.normal(0, 0.5, (N, 20, 6)) +
                          rng.integers(-1, 2, (N, 1, 6)), np.zeros((N, 20, 1))],
                         -1)
    valid = np.ones((N, 20))
    valid[::9, 15:] = 0
    D = dict(train_rows=np.arange(0, 300), val_rows=np.arange(300, 380),
             teacher_rows=np.arange(10, 70, 2), act=act, valid=valid,
             task=np.array([f"task {i % 5}" for i in range(N)]),
             episode=np.arange(N) // 7, L=-np.ones(6) * 2, U=np.ones(6) * 3,
             teacher_target=act[np.arange(10, 70, 2)] + 0.01,
             prep_report=dict(prep_id="P"))
    old_k = kc.K
    try:
        kc.K = 16
        codec, lab, rep = build(D)
        # val не влияет на словарь
        D2 = dict(D, act=act.copy())
        D2["act"][300:] += 5
        c2, _l2, _r2 = build(D2)
        assert np.array_equal(c2["P32"], codec["P32"])
        # expert и teacher не смешиваются
        C = dict(P32=codec["P32"], pos=np.full(400, -1), b=None)
        rows = np.concatenate([lab["train_rows"], lab["val_rows"]])
        C["pos"][rows] = np.arange(len(rows))
        C.update(b=np.concatenate([lab["b_train"], lab["b_val"]]),
                 rv=np.concatenate([lab["rv_train"], lab["rv_val"]]),
                 ids=np.concatenate([lab["id_train"], lab["id_val"]]),
                 b_teacher=lab["b_teacher"], rv_teacher=lab["rv_teacher"],
                 id_teacher=lab["id_teacher"],
                 teacher_rows=lab["teacher_rows"])
        be, _r, _i = targets(C, rows=[10])
        bt, _r, _i = targets(C, rows=[10], teacher_idx=[0])
        assert not np.allclose(be, bt)
        try:
            targets(C, rows=[12], teacher_idx=[0])
            raise AssertionError("несовпадение строк учителя принято")
        except SystemExit:
            pass
        try:
            targets(C, rows=[395])
            raise AssertionError("неразрешённая строка принята")
        except SystemExit:
            pass
        assert rep["coverage"]["val_tasks"] == 5 and rep["gate"]["passed"]
    finally:
        kc.K = old_k
    print("самопроверка k30_data пройдена: словарь только из train (val не "
          "влияет), метки expert/teacher не смешиваются, неразрешённые строки "
          "— отказ, задачи — строки")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    if len(sys.argv) > 1 and sys.argv[1] == "codec":
        try:
            return codec_stage()
        except SystemExit as e:
            if isinstance(e.code, str):
                print(f"  ОТКАЗ: {e.code}")
                return 3
            raise
    raise SystemExit("использование: k30_data.py codec | --selftest")


if __name__ == "__main__":
    sys.exit(main())
