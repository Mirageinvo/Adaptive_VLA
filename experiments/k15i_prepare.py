#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15i (подготовка): набор для пробы «поможет ли импульс» по выгрузке K-15h.

ВХОДЫ: план K-15h, каталог его раскаток, features.npz и analysis.json
анализатора K-15h. Перед сборкой:
  * входы плана K-15h сверены (k15h_protocol.check_plan_inputs);
  * все блоки плана проверены заново (k15h_protocol.check_run), набор полон;
  * анализ K-15h полон и построен по этому плану и каталогу;
  * строки выгрузки — ровно состояния банка плана; ни одного состояния
    резерва (45–49) и вне банка;
  * исходы выгрузки совпадают с JSON проверенных блоков ПО ИМЕНАМ политик;
  * h18 и a0 выгрузки совпадают с q0ref тех же блоков на достигнутых
    вызовах; на достигнутом вызове 4 нет NaN.

ЦЕЛЬ для основного объекта p4r3p: y = успех(p4r3p) − успех(q0ref) ∈
{−1, 0, +1}. ПРИЗНАКИ — из q0ref ПЕРЕД чанком 4 (вызов 4, до импульса):
h18 [n_pos, D] и a0 [8, 7]. Номер задачи и старта и будущие исходы в
признаки основной модели не входят (задача хранится отдельно — только для
базового переключателя «по задаче» и группировки). Если среда завершилась
до вызова 4, импульса не было: решение — q0, признаков нет (не заполняются).

ОРАКУЛЬНАЯ ГРАНИЦА пары (q0ref, p4r3p): max исходов по состоянию — граница
при знании исхода заранее, НЕ результат политики.

Выход: reports/k15i/dataset.npz и dataset.json (SHA входов и кода).
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

TARGET = "p4r3p"
BASE = "q0ref"
CALL = 4                                   # признаки перед чанком 4
RESERVED = range(45, 50)
OUT = "reports/k15i/dataset.npz"
CODE = ("k15i_prepare.py", "k15i_benefit_probe.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def outcomes_by_name(labels, outcomes, name):
    """Столбец исходов по ИМЕНИ политики (а не по позиции)."""
    labels = [str(x) for x in labels]
    if labels.count(name) != 1:
        raise SystemExit(f"в выгрузке политика {name!r} встречается "
                         f"{labels.count(name)} раз")
    return np.asarray(outcomes)[:, labels.index(name)].astype(bool)


def build(feat, plan, json_outcomes=None, q0_blocks=None):
    """feat — содержимое features.npz. Возвращает набор или SystemExit."""
    p = []
    task, state = np.asarray(feat["task"]), np.asarray(feat["state"])
    keys = list(zip(task.tolist(), state.tolist()))
    want = {(int(t), s) for t, bl in plan["blocks"].items() for b in bl
            for s in range(int(b), int(b) + int(plan["n_envs"]))}
    if any(s in RESERVED for _t, s in keys):
        p.append("в выгрузке есть состояния резерва 45–49")
    if len(keys) != len(set(keys)):
        p.append("повторяющиеся состояния в выгрузке")
    if set(keys) != want:
        p.append(f"состояния выгрузки ({len(set(keys))}) не равны банку "
                 f"плана ({len(want)})")
    labels = [str(x) for x in feat["labels"]]
    if sorted(labels) != sorted(plan["labels"]):
        p.append(f"политики выгрузки {labels} не равны плану")
    if p:
        raise SystemExit("выгрузка K-15h непригодна: " + "; ".join(p))
    y_t = outcomes_by_name(labels, feat["outcomes"], TARGET)
    y_b = outcomes_by_name(labels, feat["outcomes"], BASE)
    if json_outcomes is not None:
        for i, k in enumerate(keys):
            if json_outcomes[TARGET][k] != bool(y_t[i]) or \
                    json_outcomes[BASE][k] != bool(y_b[i]):
                p.append(f"исход {k} в выгрузке не равен JSON блока")
    reached = np.asarray(feat["call_reached"], bool)
    if reached.shape[1] < CALL:
        p.append("в выгрузке меньше 4 позиций вызовов")
    r4 = reached[:, CALL - 1]
    h18 = np.asarray(feat["h18"], np.float32)[:, CALL - 1]
    a0 = np.asarray(feat["a0"], np.float32)[:, CALL - 1]
    if r4.any() and (not np.isfinite(h18[r4]).all()
                     or not np.isfinite(a0[r4]).all()):
        p.append("на достигнутом вызове 4 в признаках NaN")
    if (~r4).any() and np.isfinite(h18[~r4]).any():
        p.append("у недостигнутого вызова 4 признаки не пусты")
    # недостигнутый импульс: исходы обязаны совпасть (протокол K-15h)
    if (y_t[~r4] != y_b[~r4]).any():
        p.append("импульс не достигнут, но исход p4r3p отличается от q0")
    if q0_blocks is not None:
        for i, k in enumerate(keys):
            ref = q0_blocks.get(k)
            if ref is None:
                p.append(f"нет q0ref для {k}")
                continue
            h_ref, a_ref, m_ref = ref
            # ЭТАЛОННАЯ МАСКА РОВНО ИЗ 4 ПОЗИЦИЙ: вызова, которого в q0ref
            # не было, — False; сверяются все четыре позиции
            m4 = [c < len(m_ref) and bool(m_ref[c]) for c in range(CALL)]
            for c in range(CALL):
                if m4[c] != bool(reached[i, c]):
                    p.append(f"{k}: маска вызова {c + 1} не равна q0ref")
                elif m4[c]:
                    if not (np.array_equal(feat["h18"][i, c], h_ref[c]) and
                            np.array_equal(feat["a0"][i, c], a_ref[c])):
                        p.append(f"{k}: h18/a0 вызова {c + 1} не равны "
                                 f"q0ref")
                elif not (np.isnan(feat["h18"][i, c]).all() and
                          np.isnan(feat["a0"][i, c]).all()):
                    p.append(f"{k}: у недостигнутого вызова {c + 1} "
                             f"признаки не NaN")
    if p:
        raise SystemExit("выгрузка K-15h непригодна: " + "; ".join(p[:6]))
    y = y_t.astype(np.int64) - y_b.astype(np.int64)
    return dict(task=task.astype(np.int64), state=state.astype(np.int64),
                h18=np.where(r4[:, None, None], h18, np.nan),
                a0=np.where(r4[:, None, None], a0, np.nan),
                reached4=r4, y=y, succ_q0=y_b, succ_target=y_t,
                oracle=np.maximum(y_b, y_t))


def from_k15h(plan_path, run_dir, features, analysis):
    """Проверенные входы K-15h -> аргументы build()."""
    import k15h_protocol as pc
    plan = json.load(open(plan_path))
    prob = pc.check_plan_inputs(plan, plan_path)
    if prob:
        raise SystemExit("входы плана K-15h не совпадают: " + "; ".join(
            prob[:4]))
    an = json.load(open(analysis))
    if not an.get("complete") or os.path.abspath(an.get("run_dir", "")) != \
            os.path.abspath(run_dir) or os.path.abspath(
                an.get("plan", "")) != os.path.abspath(plan_path):
        raise SystemExit("анализ K-15h неполон или по другим плану/каталогу")
    ok, gaps = pc.check_run(plan, plan_path, run_dir)
    if gaps:
        raise SystemExit(f"K-15h: непроверенных блоков {len(gaps)}")
    outc, q0b = {lab: {} for lab in plan["labels"]}, {}
    for (lab, t, b), pth in ok.items():
        d, z = pc.load(pth)
        for i, e in enumerate(d["episodes"]):
            k = (t, int(e["init_state_id"]))
            outc[lab][k] = bool(e["success"])
            if lab == BASE:
                ds = int(z["done_step"][i])
                n = z["k15h_h18"].shape[0]
                q0b[k] = (z["k15h_h18"][:, i], z["k15d_a0"][:n, i],
                          [pc.reached(ds, c + 1) for c in range(n)])
    feat = dict(np.load(features, allow_pickle=False))
    return plan, feat, outc, q0b


def main():
    ap = argparse.ArgumentParser(description="K-15i: подготовка")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", default="reports/k15h/plan.json")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--features", default="reports/k15h/features.npz")
    ap.add_argument("--analysis", default="reports/k15h/analysis.json")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    rd = a.run_dir
    if rd is None:
        cand = sorted(glob.glob("reports/k15h/run/p*/"))
        if len(cand) != 1:
            raise SystemExit(f"укажите --run-dir: найдено {cand}")
        rd = cand[0]
    plan, feat, outc, q0b = from_k15h(a.plan, rd, a.features, a.analysis)
    ds = build(feat, plan, outc, q0b)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **ds)
    meta = dict(
        kind="k15i_dataset", target=TARGET, base=BASE, call=CALL,
        inputs={k: dict(path=os.path.abspath(v), sha1=sha(v)) for k, v in
                (("plan", a.plan), ("features", a.features),
                 ("analysis", a.analysis))},
        run_dir=os.path.abspath(rd),
        code={f: sha(os.path.join(HERE, f)) for f in CODE},
        dataset=dict(path=os.path.abspath(a.out), sha1=sha(a.out)),
        n=int(len(ds["y"])), reached4=int(ds["reached4"].sum()),
        y_counts={str(v): int((ds["y"] == v).sum()) for v in (-1, 0, 1)},
        sr=dict(q0=float(ds["succ_q0"].mean()),
                target=float(ds["succ_target"].mean()),
                oracle_bound=float(ds["oracle"].mean())),
        note=("оракульная граница — max(q0, p4r3p) при знании исхода "
              "заранее, не результат политики; блок 45–49 не читался"))
    with open(a.out[:-4] + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print(f"  набор K-15i: {meta['n']} состояний, импульс достигнут в "
          f"{meta['reached4']}; y: {meta['y_counts']}; SR q0 "
          f"{100 * meta['sr']['q0']:.1f}%, p4r3p "
          f"{100 * meta['sr']['target']:.1f}%, ОРАКУЛЬНАЯ ГРАНИЦА "
          f"{100 * meta['sr']['oracle_bound']:.1f}%")
    print(f"  {a.out}")
    return 0


# --- самопроверка ------------------------------------------------------------
def _fake(n_tasks=3, rng=None, labels=None):
    rng = rng or np.random.default_rng(0)
    labels = labels or ["q0ref", "p4r3p", "learned_uniform",
                        "arm_dir_uniform", "static_uniform"]
    plan = dict(n_envs=5, labels=["q0ref", "p4r3p", "learned_uniform",
                                  "arm_dir_uniform", "static_uniform"],
                blocks={str(t): [25, 30, 35] for t in range(n_tasks)})
    keys = [(t, s) for t in range(n_tasks) for s in range(25, 40)]
    N = len(keys)
    reached = np.ones((N, 4), bool)
    reached[0, 3] = False                      # среда 0 завершилась до вызова 4
    h18 = rng.standard_normal((N, 4, 3, 5)).astype(np.float32)
    a0 = rng.standard_normal((N, 4, 8, 7)).astype(np.float32)
    h18[~reached] = np.nan
    a0[~reached] = np.nan
    succ = {lab: rng.random(N) < 0.6 for lab in labels}
    succ["p4r3p"][0] = succ["q0ref"][0]
    out = np.stack([succ[lab] for lab in labels], 1)
    feat = dict(task=np.array([k[0] for k in keys]),
                state=np.array([k[1] for k in keys]), labels=np.array(labels),
                outcomes=out, call_reached=reached, h18=h18, a0=a0)
    json_out = {lab: {k: bool(succ[lab][i]) for i, k in enumerate(keys)}
                for lab in labels}
    q0b = {k: (h18[i], a0[i], reached[i].tolist())
           for i, k in enumerate(keys)}
    return plan, feat, json_out, q0b, succ


def selftest():
    plan, feat, jo, q0b, succ = _fake()
    ds = build(feat, plan, jo, q0b)
    want = succ["p4r3p"].astype(int) - succ["q0ref"].astype(int)
    assert np.array_equal(ds["y"], want)
    assert not ds["reached4"][0] and np.isnan(ds["h18"][0]).all()
    assert np.isfinite(ds["h18"][1]).all()
    assert np.array_equal(ds["oracle"], succ["p4r3p"] | succ["q0ref"])
    # переставленные столбцы исходов: y по ИМЕНАМ не меняется
    lab2 = ["p4r3p", "static_uniform", "q0ref", "learned_uniform",
            "arm_dir_uniform"]
    plan2, feat2, jo2, q0b2, succ2 = _fake(labels=lab2)
    ds2 = build(feat2, plan2, jo2, q0b2)
    assert np.array_equal(ds2["y"], succ2["p4r3p"].astype(int)
                          - succ2["q0ref"].astype(int))
    # знак цели: спасение = +1
    i = int(np.nonzero(ds["y"] == 1)[0][0])
    assert succ["p4r3p"][i] and not succ["q0ref"][i]

    def bad(mut, word):
        f = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in feat.items()}
        j = {k: dict(v) for k, v in jo.items()}
        q = dict(q0b)
        mut(f, j, q)
        try:
            build(f, plan, j, q)
            raise AssertionError(f"принята выгрузка: {word}")
        except SystemExit as e:
            assert word in str(e), (word, str(e))
    # неполная выгрузка
    bad(lambda f, j, q: [f.update({k: f[k][:-1] for k in (
        "task", "state", "outcomes", "call_reached", "h18", "a0")})],
        "не равны банку")
    # состояние резерва
    bad(lambda f, j, q: f["state"].__setitem__(-1, 47), "резерва")
    # исход выгрузки не равен JSON блока
    def flip(f, j, q):
        f["outcomes"][3, list(f["labels"]).index("p4r3p")] ^= True
    bad(flip, "не равен JSON")
    # NaN на достигнутом вызове 4
    bad(lambda f, j, q: f["h18"].__setitem__((2, 3, 0, 0), np.nan),
        "NaN")
    # недостигнутый импульс с другим исходом
    def unreached(f, j, q):
        col = list(f["labels"]).index("p4r3p")
        f["outcomes"][0, col] = not f["outcomes"][0, col]
        j["p4r3p"][(0, 25)] = bool(f["outcomes"][0, col])
    bad(unreached, "не достигнут")
    # признаки не равны q0ref
    def h_bad(f, j, q):
        f["h18"][5, 2, 0, 0] += 1
    bad(h_bad, "не равны q0ref")
    # исходный q0ref содержит два вызова, выгрузка приписывает 3-й и 4-й
    def two_calls(f, j, q):
        k = (1, 30)
        i = [tuple(x) for x in zip(f["task"].tolist(),
                                   f["state"].tolist())].index(k)
        h, a_, _m = q[k]
        q[k] = (h[:2], a_[:2], [True, True])
        f["call_reached"][i] = [True, True, True, True]
    bad(two_calls, "маска вызова 3")
    # недостигнутый вызов с непустыми признаками
    def ghost(f, j, q):
        h, a_, _m = q[(0, 25)]
        f["h18"][0, 3] = 0.0
    bad(ghost, "не пусты")
    # дублированная политика
    def dup(f, j, q):
        lab = list(f["labels"])
        lab[2] = "p4r3p"
        f["labels"] = np.array(lab)
    bad(dup, "политики выгрузки")
    print("самопроверка k15i_prepare пройдена: цель по именам и знак, "
          "недостигнутый импульс, неполная/повреждённая выгрузка, резерв, "
          "сверка с JSON и q0ref")
    return 0


if __name__ == "__main__":
    sys.exit(main())
