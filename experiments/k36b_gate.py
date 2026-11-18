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
    try:
        g = json.load(open(ka.GATE_A, encoding="utf-8"))
        an = json.load(open(ka.OUT_A, encoding="utf-8"))
        if not isinstance(g, dict) or not isinstance(an, dict):
            raise ValueError("не объект")
    except (OSError, ValueError) as e:
        return "technical", None, [f"решение/анализ A не читается: {e}"]
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
    fst, frule = ka.resource_rule(Y)
    if g.get("status") != fst or g.get("rule") != frule or \
            an.get("status") != fst or an.get("rule") != frule:
        return "technical", None, [
            f"сохранённые float-статус/флаги решения и анализа A ≠ "
            f"пересчёту по исходам ({fst})"]
    st, flags, det = rule_a_exact(Y)
    return ("passed" if st == "passed_A_development" else "stopped"), st, \
        dict(flags=flags, **det, float_status=fst, float_rule=frule)


def write_decision_a():
    """Всегда перезаписывает gate_a_exact.json (прежний статус не
    остаётся при сбое); любой сбой чтения — technical."""
    try:
        kind, st, det = decision_a()
    except Exception as e:                                   # noqa: BLE001
        kind, st, det = "technical", None, [f"{type(e).__name__}: {e}"]
    rec = dict(kind=kind, status=st, details=det, rule=dict(
        joint_net_min=JOINT_MIN, per_seed_net_min=SEED_MIN,
        note="целочисленная форма §7.3 при 100 случаях на задачу"))
    if kind != "technical":
        import k36_analyze as ka
        rec.update(gate_a_sha1=pc.sha(ka.GATE_A),
                   analysis_sha1=pc.sha(ka.OUT_A))
    if os.path.exists(GATE_A_EXACT):
        os.remove(GATE_A_EXACT)
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


def _paths_of(eid):
    """eid эпизода '<scope>/<stage>/<метка>/t<задача>/<ключ>' -> пути."""
    import k36_harness as kh
    parts = str(eid).split("/")
    if len(parts) != 5 or not parts[3].startswith("t"):
        return None
    return kh.ep_paths(parts[1], parts[2], int(parts[3][1:]), parts[4])


def before_reserve(L, want, allowed, want_of_eid, validate):
    """Непосредственно перед НОВЫМ reserve эпизода want.
    Открытый резерв:
      - тот же eid — scope обязан совпасть (иначе отказ);
      - эпизод этой стадии (точные scope и префикс eid) с записанным
        результатом — результат проходит ПОЛНУЮ проверку эпизода против
        ожидаемого контракта (want_of_eid -> validate: идентификаторы,
        prepare, контракт, actor, хеши, журнал, расход = 10 + шагам = длине
        действий) и только тогда settle фактическим расходом;
      - без результата — reserve спишет его как аварию;
      - чужая стадия или непроверяемый результат — отказ без изменений."""
    import k36_harness as kh
    pd = L.d.get("pending")
    if not pd:
        return
    if pd.get("id") == want["eid"]:
        if pd.get("scope") != want["scope"]:
            raise SystemExit(f"{want['eid']}: открытый резерв в scope "
                             f"{pd.get('scope')} ≠ {want['scope']}")
        return
    pr = pending_problems(L, allowed)
    if pr:
        raise SystemExit("; ".join(pr))
    paths = _paths_of(pd["id"])
    if paths is None:
        raise SystemExit(f"открытый резерв {pd['id']}: не эпизод стадии")
    jp, np_ = paths
    if not (os.path.exists(jp) and os.path.exists(np_)):
        return
    try:
        w2 = want_of_eid(pd["id"])
        got = kh.load_ep(w2)
    except (SystemExit, KeyError, ValueError) as e:
        raise SystemExit(f"результат открытого резерва {pd['id']} не "
                         f"сходится с ожидаемым: {e} — ручное решение")
    if got is None or w2["scope"] != pd["scope"] or w2["eid"] != pd["id"]:
        raise SystemExit(f"результат открытого резерва {pd['id']} не "
                         f"сходится с ожидаемым — ручное решение")
    rec, z = got
    p2 = validate(rec, z, w2)
    phys = rec.get("phys")
    if p2 or not isinstance(phys, int) or not \
            0 < phys <= int(pd.get("amount", 0)):
        raise SystemExit(f"результат открытого резерва {pd['id']} не "
                         f"прошёл проверку ({p2[:2]}, phys {phys}) — ручное "
                         f"решение")
    L.settle(pd["id"], phys)


def parse_eid(eid):
    """'<scope>/<stage>/<метка>/t<задача>/<ключ>' -> (stage, label, task,
    key)."""
    parts = str(eid).split("/")
    if len(parts) != 5 or not parts[3].startswith("t"):
        raise ValueError(f"eid {eid}")
    return parts[1], parts[2], int(parts[3][1:]), parts[4]


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
    # записанный, но не учтённый результат ДРУГОГО эпизода стадии —
    # settle его фактическим расходом, а не аварийное списание 610
    import k36_harness as kh
    with tempfile.TemporaryDirectory() as td:
        keep = kh.OUT
        try:
            kh.OUT = os.path.join(td, "r")
            L = kcol.Ledger(os.path.join(td, "l2.json"), pc.CAPS)
            e1 = "bar_c/confirm_c/BAR/t0/k33s1_t0_c00001"
            L.reserve("bar_c", e1)
            jp, np_ = kh.ep_paths("confirm_c", "BAR", 0, "k33s1_t0_c00001")
            os.makedirs(os.path.dirname(jp))
            np.savez(np_, a=np.zeros(1))
            pc.save_json(jp, dict(eid=e1, scope="bar_c", phys=32,
                                  npz_sha1=pc.sha(np_)))
            w2 = dict(eid="bar_c/confirm_c/BAR/t0/k33s1_t0_c00002",
                      scope="bar_c")

            def woe(eid):
                st_, lab, t_, key = parse_eid(eid)
                return dict(eid=eid, scope="bar_c", stage=st_, label=lab,
                            task=t_, key=key)
            vals = []

            def val(rec, z, w):
                vals.append(rec["eid"])
                return [] if rec.get("ok", True) else ["контракт"]
            keep_le = kh.load_ep
            kh.load_ep = lambda w: (json.load(open(kh.ep_paths(
                w["stage"], w["label"], w["task"], w["key"])[0])),
                {}) if os.path.exists(kh.ep_paths(
                    w["stage"], w["label"], w["task"], w["key"])[0]) else None
            before_reserve(L, w2, C_PENDING, woe, val)
            assert vals == [e1]
            assert L.d["pending"] is None and L.spent("bar_c") == 32 and \
                not L.d["crashes"]
            # результата нет — допускается аварийное списание при reserve
            L.reserve("bar_c", "bar_c/confirm_c/BAR/t0/k33s1_t0_c00003")
            before_reserve(L, w2, C_PENDING, woe, val)
            assert L.d["pending"]["id"].endswith("00003")
            # записанный результат с чужим контрактом или phys −7 — отказ
            for bad in (dict(ok=False, phys=32), dict(phys=-7)):
                e4 = "bar_c/confirm_c/BAR/t0/k33s1_t0_c00004"
                L.d["pending"] = dict(scope="bar_c", id=e4, amount=610)
                jp4, np4 = kh.ep_paths("confirm_c", "BAR", 0,
                                       "k33s1_t0_c00004")
                np.savez(np4, a=np.zeros(1))
                pc.save_json(jp4, dict(eid=e4, scope="bar_c",
                                       npz_sha1=pc.sha(np4), **bad))
                before = json.dumps(L.d, sort_keys=True)
                try:
                    before_reserve(L, w2, C_PENDING, woe, val)
                    raise AssertionError(f"чужой результат принят {bad}")
                except SystemExit:
                    pass
                assert json.dumps(L.d, sort_keys=True) == before
            # тот же eid в другом scope — отказ
            L.d["pending"] = dict(scope="confirm_c", id=w2["eid"],
                                  amount=610)
            try:
                before_reserve(L, w2, C_PENDING, woe, val)
                raise AssertionError("тот же eid в другом scope принят")
            except SystemExit:
                pass
            kh.load_ep = keep_le
            L.d["pending"] = None
            # чужая стадия — отказ без изменений
            L.d["pending"] = dict(scope="confirm_a", id="confirm_a/x",
                                  amount=610)
            before = json.dumps(L.d, sort_keys=True)
            try:
                before_reserve(L, w2, C_PENDING, woe, val)
                raise AssertionError("чужой резерв принят")
            except SystemExit:
                pass
            assert json.dumps(L.d, sort_keys=True) == before
        finally:
            kh.OUT = keep
    # повреждённое решение A: technical, прежний passed не остаётся;
    # подменённый float-статус — technical
    import k36_analyze as ka
    G = globals()
    keep = dict(GATE_A_EXACT=G["GATE_A_EXACT"])
    keep_ka = (ka.GATE_A, ka.OUT_A, ka.collect_a)
    keep_lp, keep_af = pc.load_prepare, kh.a_files
    with tempfile.TemporaryDirectory() as td:
        try:
            G["GATE_A_EXACT"] = os.path.join(td, "exact.json")
            ka.GATE_A, ka.OUT_A = (os.path.join(td, "g.json"),
                                   os.path.join(td, "a.json"))
            pc.save_json(G["GATE_A_EXACT"], dict(kind="passed"))
            open(ka.GATE_A, "w").write("{испорчен")
            open(ka.OUT_A, "w").write("{}")
            kind, st, det = write_decision_a()
            assert kind == "technical" and json.load(open(
                G["GATE_A_EXACT"]))["kind"] == "technical"
            Yp = table({0: (8, 9, -8), 1: (8, 9, -8)})
            fst, frule = ka.resource_rule(Yp)
            fs = [os.path.join(td, "e.json")]
            open(fs[0], "w").write("x")
            pc.load_prepare = lambda: dict(sha1="P")
            kh.a_files = lambda prep: set(fs)
            ka.collect_a = lambda: (Yp, [], {}, None)
            pc.save_json(ka.OUT_A, dict(status=fst, rule=frule))
            gate = dict(status=fst, rule=frule, prepare_sha1="P",
                        files={fs[0]: pc.sha(fs[0])},
                        analysis_sha1=pc.sha(ka.OUT_A))
            pc.save_json(ka.GATE_A, gate)
            kind, st, det = write_decision_a()
            assert kind == "passed" and det["float_status"] == fst
            pc.save_json(ka.OUT_A, dict(status="technical_stop", rule=frule))
            pc.save_json(ka.GATE_A, dict(gate, status="technical_stop",
                                         analysis_sha1=pc.sha(ka.OUT_A)))
            kind, st, det = write_decision_a()
            assert kind == "technical"
        finally:
            G.update(keep)
            ka.GATE_A, ka.OUT_A, ka.collect_a = keep_ka
            pc.load_prepare, kh.a_files = keep_lp, keep_af
    assert code_of("stopped") == 4 and code_of("technical") == 3
    print("самопроверка k36b_gate пройдена: правило A в целых (ровно +18 и "
          "−6 проходят, +17 и −7 — стоп), чужой открытый резерв — отказ; "
          "записанный результат другого эпизода стадии — settle, а не "
          "аварийное списание")
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
