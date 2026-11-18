#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-36: точные решения переходов A -> B и B -> C, принадлежность резервов.

Решение A -> B (замечание ревью 5fa9b85: граница +3 п.п. в float давала
0.029999… < 0.03). Ресурсное правило §7.3 эквивалентно целочисленному при
100 случаях на задачу и равных весах: для каждого из сравнений T32−P32 и
T32−R сумма (выигрыши − потери) по обоим сидам и задачам 0/6/8 ≥ 18
(= 0.03·600) и на каждом сиде ≥ −6 (= −0.02·300). Проверки полноты и SHA —
те же, что в k36_analyze.gate_a_problems (точный набор 1800 эпизодов, их
SHA, prepare, неизменный файл анализа); исходы собираются заново
(k36_analyze.collect_a: блоки, журналы, хеши).
Статусы решений: passed (код 0), stopped — проверенное отрицательное
решение (код 4), technical — отсутствующие/повреждённые данные (код 3).
Принадлежность резерва: перед любым reserve открытый резерв журнала
обязан принадлежать той же стадии (scope и префикс eid), иначе — отказ до
списания (чужой незакрытый резерв не «лечится» аварийным списанием).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k36_protocol as pc  # noqa: E402

GATE_A_EXACT = "reports/k36/gate_a_exact.json"
TASKS = (0, 6, 8)
JOINT_MIN, SEED_MIN = 18, -6          # 0.03·600 и −0.02·300 (точно)


def net(Y, x, y, seeds=(0, 1)):
    """Σ (выигрыши − потери) X против Y по сидам и задачам (целое)."""
    return int(sum(int(Y[x][s][t].sum()) - int(Y[y][s][t].sum())
                   for s in seeds for t in TASKS))


def rule_a_exact(Y):
    n = {s: {k: net(Y, *k.split("-"), seeds=(s,))
             for k in ("T32-P32", "T32-R")} for s in (0, 1)}
    j = {k: net(Y, *k.split("-")) for k in ("T32-P32", "T32-R")}
    flags = dict(joint_T32_P32=j["T32-P32"] >= JOINT_MIN,
                 joint_T32_R=j["T32-R"] >= JOINT_MIN,
                 per_seed=all(v >= SEED_MIN for s in (0, 1)
                              for v in n[s].values()))
    st = "passed_A_development" if all(flags.values()) else \
        "stopped_A_no_signal"
    return st, flags, dict(net_joint=j, net_per_seed={str(s): n[s]
                                                      for s in n})


def decision_a():
    """-> (вид: passed|stopped|technical, статус, детали/проблемы)."""
    import k36_analyze as ka
    import k36_harness as kh
    if not os.path.exists(ka.GATE_A) or not os.path.exists(ka.OUT_A):
        return "technical", None, ["нет решения/анализа A (analyze-a)"]
    g = json.load(open(ka.GATE_A, encoding="utf-8"))
    try:
        prep = pc.load_prepare()
    except SystemExit as e:
        return "technical", None, [str(e.code)]
    p = []
    if g.get("prepare_sha1") != prep["sha1"]:
        p.append("решение A другого prepare")
    files = g.get("files") or {}
    if set(files) != kh.a_files(prep):
        p.append(f"набор файлов решения A ≠ 1800 эпизодам ({len(files)})")
    for f, sh in files.items():
        if not os.path.exists(f) or pc.sha(f) != sh:
            p.append("файлы A изменены после решения")
            break
    if pc.sha(ka.OUT_A) != g.get("analysis_sha1"):
        p.append("анализ A изменён после решения")
    if p:
        return "technical", None, p
    try:
        Y, pr, _f, _p = ka.collect_a()
    except SystemExit as e:
        return "technical", None, [f"сбор A: {e.code}"]
    if pr:
        return "technical", None, [f"данные A: {pr[:3]}"]
    st, flags, det = rule_a_exact(Y)
    return ("passed" if st == "passed_A_development" else "stopped"), st, \
        dict(flags=flags, **det, float_status=g.get("status"))


def write_decision_a():
    kind, st, det = decision_a()
    rec = dict(kind=kind, status=st, details=det, rule=dict(
        joint_net_min=JOINT_MIN, per_seed_net_min=SEED_MIN,
        note="целочисленная форма §7.3 при 100 случаях на задачу"))
    if kind != "technical":
        import k36_analyze as ka
        rec.update(gate_a_sha1=pc.sha(ka.GATE_A),
                   analysis_sha1=pc.sha(ka.OUT_A))
    pc.save_json(GATE_A_EXACT, rec)
    return kind, st, det


def gate_a_problems():
    """Для continue-b: [] только при проверенном passed_A_development."""
    kind, st, det = decision_a()
    if kind == "passed":
        return []
    return det if kind == "technical" else [f"статус A: {st}"]


def code_of(kind):
    return dict(passed=0, stopped=4, technical=3)[kind]


TRAIN_PENDING = [(f"train_tplus_s{s}", f"k36/tplus/s{s}/") for s in (0, 1)] \
    + [(f"val_tplus_s{s}", f"k32/T/s{s}/val_") for s in (0, 1)]
SCREEN_PENDING = [("screen_b", "screen_b/screen_b/")]
C_PENDING = [("confirm_c", "confirm_c/confirm_c/"),
             ("bar_c", "bar_c/confirm_c/")]


def bar_pending(kind):
    return [("checks", f"checks/{kind}/")]


# --- резервы ---------------------------------------------------------------------
def pending_problems(L, allowed):
    """Открытый резерв журнала принадлежит этой стадии или его нет.
    allowed — [(scope, префикс eid)]."""
    pd = L.d.get("pending")
    if not pd:
        return []
    if any(pd.get("scope") == sc and str(pd.get("id", "")).startswith(pre)
           for sc, pre in allowed):
        return []
    return [f"открыт чужой резерв {pd.get('id')} ({pd.get('scope')}) — "
            f"сначала завершите его стадию; списание не выполняется"]


def selftest():
    import tempfile
    import k31_collect as kcol
    rng = np.random.default_rng(0)
    names = ("R", "T32", "P32")

    def table(nets_TP):
        """nets_TP[s][t] — чистый выигрыш T32 над P32 и над R."""
        Y = {nm: {s: {} for s in (0, 1)} for nm in names}
        for s in (0, 1):
            for t in TASKS:
                base = rng.random(100) < 0.5
                d = nets_TP[s][TASKS.index(t)]
                T = base.copy()
                idx = np.flatnonzero(~T) if d > 0 else np.flatnonzero(T)
                T[idx[:abs(d)]] = d > 0
                Y["T32"][s][t] = T
                Y["P32"][s][t] = base
                Y["R"][s][t] = base
        return Y
    # ровно +18/600: +8, +9, −8 на каждом сиде — проходит (раньше отказ)
    Y = table({0: (8, 9, -8), 1: (8, 9, -8)})
    st, f, det = rule_a_exact(Y)
    assert det["net_joint"]["T32-P32"] == 18 and st == "passed_A_development"
    # +17 — стоп
    st, f, det = rule_a_exact(table({0: (8, 9, -8), 1: (8, 8, -8)}))
    assert det["net_joint"]["T32-P32"] == 17 and st == "stopped_A_no_signal"
    # ровно −6 на сиде — проходит; −7 — стоп
    st, f, det = rule_a_exact(table({0: (20, 10, 0), 1: (-6, 0, 0)}))
    assert f["per_seed"] and st == "passed_A_development"
    st, f, det = rule_a_exact(table({0: (20, 10, 0), 1: (-7, 0, 0)}))
    assert not f["per_seed"]
    # чужой открытый резерв — отказ; свой — допустим
    with tempfile.TemporaryDirectory() as td:
        L = kcol.Ledger(os.path.join(td, "l.json"), pc.CAPS)
        bar = [("checks", "checks/bar_legacy/")]
        assert pending_problems(L, bar) == []
        L.reserve("confirm_a", "confirm_a/confirm_a/R_s0/t0/x")
        assert pending_problems(L, bar)
        assert pending_problems(L, [("confirm_a",
                                     "confirm_a/confirm_a/")]) == []
    assert code_of("stopped") == 4 and code_of("technical") == 3
    print("самопроверка k36b_gate пройдена: правило A в целых (ровно +18 и "
          "−6 проходят, +17 и −7 — стоп), чужой открытый резерв — отказ")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-36: точное решение A")
    ap.add_argument("cmd", choices=("gate-a",))
    ap.parse_args()
    kind, st, det = write_decision_a()
    if kind == "technical":
        print("  решение A: ТЕХНИЧЕСКИЙ СТОП — " + "; ".join(det[:3]))
    else:
        print(f"  решение A (целочисленное правило): {st}; чистые выигрыши "
              f"совместно {det['net_joint']}, по сидам "
              f"{det['net_per_seed']}; float-статус analyze-a "
              f"{det['float_status']}")
    return code_of(kind)


if __name__ == "__main__":
    sys.exit(main())
