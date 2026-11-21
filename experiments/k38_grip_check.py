#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-38: проверки и аудит сдвига закрытия схвата.

Первая итерация — только read-only CPU-аудит архива T1 K-37 c2:
  python3 experiments/k38_grip_check.py audit --archive reports/k37c/runs/c2
План c2 загружается штатно (k37_confirm.load_plan: SHA плана, кода,
головок, банка, входов K-36). Для каждой из 960 записей T1 (480 сцен ×
2 сида): k36_harness.load_ep (идентификаторы, хеши, SHA NPZ), полная
проверка эпизода K-32 (k36_harness.episode_problems) и запись журнала
части (scope confirm, расход = phys). По вызовам j_elapsed: тот же поиск
tau, что у обёртки (prev — actions[elapsed−1, 6] или DUMMY[6]); первое
событие; достижимость early (tau−2 < k) и late (tau < k) по архивному k.
Решение о расходе: U = неудачи T1 с достигнутой первой изменённой командой;
оба числителя < 48 -> stopped_low_coverage, иначе coverage_ok.
Отчёт reports/k38/audit.json; архив K-37 только читается.
Сбор (prepare / preflight / assign / collect / merge / analyze) — следующая
итерация, только при coverage_ok.
"""
import argparse
import collections
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k38_grip_operator as kg  # noqa: E402

OUT = "reports/k38"
NEED = 48                         # +5 п.п. из 960
N_EXPECTED = 960


def episode_event(J, A, max_steps=kg.MAX_STEPS):
    """Архивная запись T1 -> (первое событие | None, число вызовов с
    событием). J — j_* поля журнала, A — физические действия."""
    A = np.asarray(A, np.float64)
    first, n_calls = None, 0
    for i in range(len(J["elapsed"])):
        el, k = int(J["elapsed"][i]), int(J["k"][i])
        prev = float(A[el - 1, 6]) if el > 0 else kg.PREV0
        tau = kg.find_event(kg.phys_grip(J["a_codec"][i]), prev,
                            kg.horizon(el, max_steps))
        if tau is None:
            continue
        n_calls += 1
        if first is None:
            first = dict(call=i + 1, elapsed=el, tau=int(tau), k=k,
                         early=bool(tau - 2 < k), late=bool(tau < k))
    return first, n_calls


def transitions(A):
    """Позиции исполненных переходов открыто -> закрыто."""
    g = np.asarray(A, np.float64)[:, 6]
    prev = np.r_[kg.PREV0, g[:-1]]
    return np.flatnonzero((prev == kg.OPEN) & (g == kg.CLOSE))


def boundary_lost(A, max_steps=kg.MAX_STEPS):
    """True — переход есть, но ни один не помещается окном tau−2…tau+2 в
    свой чанк; None — переходов нет."""
    tr = transitions(A)
    if len(tr) == 0:
        return None
    ok = [2 <= p % 8 and p % 8 + 2 < kg.horizon(p - p % 8, max_steps)
          for p in tr]
    return not any(ok)


def summarize(rows):
    """rows: dict(task, seed, success, ev, n_calls, lost) -> отчёт."""
    def block(rs):
        ev = [r for r in rs if r["ev"]]
        fail = [r for r in rs if not r["success"]]
        out = dict(episodes=len(rs), successes=sum(r["success"] for r in rs),
                   with_event=len(ev),
                   calls_with_event=sum(r["n_calls"] for r in rs))
        for kind in ("early", "late"):
            rr = [r for r in ev if r["ev"][kind]]
            out[kind] = dict(reached=len(rr),
                             reached_fail=sum(not r["success"] for r in rr),
                             reached_success=sum(r["success"] for r in rr))
        out["event_fail"] = sum(not r["success"] for r in ev)
        out["fail"] = len(fail)
        out["cut_before_early"] = sum(not r["ev"]["early"] for r in ev)
        out["cut_before_late"] = sum(not r["ev"]["late"] for r in ev)
        out["boundary_lost"] = sum(r["lost"] is True for r in rs)
        out["boundary_lost_fail"] = sum(r["lost"] is True and
                                        not r["success"] for r in rs)
        out["no_transition"] = sum(r["lost"] is None for r in rs)
        return out
    rep = dict(all=block(rows))
    rep["by_task"] = {str(t): block([r for r in rows if r["task"] == t])
                      for t in sorted({r["task"] for r in rows})}
    rep["by_seed"] = {str(s): block([r for r in rows if r["seed"] == s])
                      for s in sorted({r["seed"] for r in rows})}
    rep["tau_hist"] = dict(sorted(collections.Counter(
        r["ev"]["tau"] for r in rows if r["ev"]).items()))
    rep["call_hist"] = dict(sorted(collections.Counter(
        r["ev"]["call"] for r in rows if r["ev"]).items()))
    n = len(rows)
    rep["U"] = {k: rep["all"][k]["reached_fail"] / max(n, 1)
                for k in ("early", "late")}
    rep["need"] = NEED
    rep["status"] = ("coverage_ok" if max(rep["all"]["early"]["reached_fail"],
                                          rep["all"]["late"]["reached_fail"])
                     >= NEED else "stopped_low_coverage")
    return rep


def audit(a):
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    rr = a.archive
    plan, k36 = kcf.load_plan(rr)
    C = kh.consts(k36)
    rows, prob = [], []
    for k in range(kcf.N_PARTS):
        pd = kcf.part_dir(rr, k)
        lp = os.path.join(pd, "ledger.json")
        try:
            ledger = pc.load_json(lp)
        except (OSError, ValueError) as exc:
            prob.append(f"part {k}: журнал не читается ({exc})")
            continue
        if (not isinstance(ledger, dict) or
                ledger.get("caps") != kcf.CAP_PART or
                not isinstance(ledger.get("done"), dict)):
            prob.append(f"part {k}: неверный журнал или лимиты")
            continue
        if ledger.get("pending") is not None:
            prob.append(f"part {k}: открыт резерв {ledger['pending']}")
            continue
        kcf.install(pd)
        for w in kcf.part_wants(plan, k36, k):
            if w["kind"] != "T":
                continue
            got = kh.load_ep(w)
            if got is None:
                prob.append(f"{w['eid']}: нет записи")
                continue
            rec, z = got
            pr = kh.episode_problems(rec, z, w, C)
            dn = ledger["done"].get(w["eid"])
            if (not isinstance(dn, dict) or dn.get("scope") != "confirm" or
                    type(dn.get("actual")) is not int or
                    dn["actual"] != rec["phys"]):
                pr.append(f"журнал части {dn} ≠ записи ({rec['phys']})")
            if pr:
                prob += [f"{w['eid']}: {x}" for x in pr[:3]]
                continue
            J = {kk[2:]: v for kk, v in z.items() if kk.startswith("j_")}
            ev, nc = episode_event(J, z["actions"])
            rows.append(dict(eid=w["eid"], part=k, task=w["task"],
                             case_id=w["case_id"], seed=w["seed"],
                             success=bool(rec["summary"]["success"]),
                             steps=int(rec["summary"]["steps"]),
                             ev=ev, n_calls=nc,
                             lost=boundary_lost(z["actions"])))
    if len(rows) != N_EXPECTED:
        prob.append(f"записей T1 {len(rows)} из {N_EXPECTED}")
    if len({r["eid"] for r in rows}) != len(rows):
        prob.append("повторный eid T1 в архиве")
    mp = os.path.join(rr, "merged.json")
    m_succ = None
    if os.path.exists(mp):
        try:
            m = pc.load_json(mp)
            merged_t1 = [(r["eid"], r["success"]) for r in m["rows"]
                         if r["policy"] == "T1"]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            prob.append(f"merged.json не читается ({exc})")
        else:
            raw_t1 = [(r["eid"], r["success"]) for r in rows]
            m_succ = sum(ok is True for _, ok in merged_t1)
            if m.get("ok") is not True or m.get("plan_sha256") != plan["sha256"]:
                prob.append("merged.json: незавершённое слияние или другой план")
            if (any(not isinstance(eid, str) or type(ok) is not bool
                    for eid, ok in merged_t1) or
                    sorted(merged_t1) != sorted(raw_t1)):
                prob.append("merged.json: исходы T1 по eid ≠ записям эпизодов")
    rep = summarize(rows) if rows else dict(status="technical_stop")
    if prob:
        rep["status"] = "technical_stop"
    rep.update(archive=rr, plan_sha256=plan["sha256"], problems=prob[:50],
               merged_T1_successes=m_succ,
               time=time.strftime("%Y-%m-%dT%H:%M:%S"),
               rows=[{kk: r[kk] for kk in ("eid", "task", "seed", "success",
                                            "ev", "n_calls", "lost")}
                     for r in rows])
    pc.save_json(os.path.join(OUT, "audit.json"), rep)
    A = rep.get("all", {})
    if prob:
        print("  K-38 аудит: ТЕХНИЧЕСКИЙ СТОП — " + "; ".join(prob[:3]))
        return 3
    print(f"  T1: эпизодов {A['episodes']}, успехов {A['successes']} "
          f"(merged.json {m_succ}); с событием {A['with_event']} "
          f"(вызовов {A['calls_with_event']})")
    for kind in ("early", "late"):
        x = A[kind]
        print(f"  {kind}: исполнено {x['reached']} (неудач T1 "
              f"{x['reached_fail']}, успехов {x['reached_success']}); "
              f"U = {100 * rep['U'][kind]:.2f} п.п. (нужно ≥ {NEED} неудач)")
    print(f"  обрыв до первой изменённой команды: early "
          f"{A['cut_before_early']}, late {A['cut_before_late']}; только "
          f"переходы у границ чанков {A['boundary_lost']} (из них неудач "
          f"{A['boundary_lost_fail']}); без перехода {A['no_transition']}")
    for nm in ("by_task", "by_seed"):
        print(f"  {nm}: " + "; ".join(
            f"{k}: событие {v['with_event']}, early/late неудач "
            f"{v['early']['reached_fail']}/{v['late']['reached_fail']} из "
            f"{v['fail']}" for k, v in rep[nm].items()))
    print(f"  tau {rep['tau_hist']}; вызов {rep['call_hist']}")
    print(f"  K-38 аудит: {rep['status']} -> {os.path.join(OUT, 'audit.json')}")
    return 0


def selftest():
    O, C = kg.OPEN, kg.CLOSE

    def chunk(g):
        a = np.zeros((8, 7), np.float32)
        a[:, 6] = -np.asarray(g, np.float32)
        return a
    # синтетическая запись: выбор аудита = выбору обёртки
    seq = [chunk([O] * 8), chunk([O, O, O, O, C, C, C, C]),
           chunk([C] * 8)]
    ks = [8, 8, 3]
    J = dict(elapsed=np.array([0, 8, 16]), k=np.array(ks),
             a_codec=np.stack(seq))
    A = np.concatenate([np.c_[np.zeros((k, 6)), -s[:k, 6]]
                        for s, k in zip(seq, ks)])
    ev, nc = episode_event(J, A)
    assert ev == dict(call=2, elapsed=8, tau=4, k=8, early=True, late=True)
    assert nc == 1

    class Base:
        def __init__(self):
            self.i, self.episode, self.call, self.last = 0, 0, 0, None

        def begin(self, *a):
            self.i = 0

        def decide(self, *a):
            c = seq[self.i]
            self.i += 1
            self.call += 1
            return c.copy(), {}

        def executed(self, k):
            pass
    arm = kg.GripArm(Base(), "zero")
    arm.begin(0, 0)
    for j, k in enumerate(ks):
        arm.decide(None, 0, None, 8 * j)
        arm.executed(k)
    assert arm.event["call"] == ev["call"] and arm.event["tau"] == ev["tau"] and \
        arm.event["elapsed"] == ev["elapsed"]
    # обрыв: событие на вызове, где k ≤ tau−2 — ни один сдвиг не исполнен
    J2 = dict(J, k=np.array([8, 2, 3]))
    A2 = np.concatenate([np.c_[np.zeros((k, 6)), -s[:k, 6]]
                         for s, k in zip(seq, [8, 2])])
    ev2, _ = episode_event(dict(J2, elapsed=J2["elapsed"][:2],
                                k=J2["k"][:2], a_codec=J2["a_codec"][:2]), A2)
    assert ev2["early"] is False and ev2["late"] is False
    # граница: переход только на позиции 0 или 7 чанка — потерян
    A3 = np.zeros((16, 7))
    A3[:, 6] = [O] * 7 + [C] * 9                   # переход на p = 7
    assert boundary_lost(A3) is True
    A4 = np.zeros((16, 7))
    A4[:, 6] = [O] * 4 + [C] * 12                  # p = 4 — помещается
    assert boundary_lost(A4) is False
    assert boundary_lost(np.full((8, 7), O)) is None
    # сводка и правило 48
    rows = []
    for i in range(960):
        fail = i < 150
        ev_ = dict(call=1, tau=4, k=8, early=i < 60, late=i < 30)
        rows.append(dict(task=(0, 6, 8)[i % 3], seed=i % 2, success=not fail,
                         ev=ev_ if i < 500 else None, n_calls=1 if i < 500
                         else 0, lost=None))
    rep = summarize(rows)
    assert rep["all"]["successes"] == 810
    assert rep["all"]["early"]["reached_fail"] == 60 and \
        rep["status"] == "coverage_ok"
    for r in rows:
        if r["ev"]:
            r["ev"]["early"] = r["ev"]["late"] = False
    assert summarize(rows)["status"] == "stopped_low_coverage"
    print("самопроверка k38_grip_check пройдена: аудит на синтетической "
          "записи выбирает то же событие, что обёртка; обрыв до подстановки; "
          "граница чанка (позиции 0/7 теряются, 4 — нет); правило 48 "
          "неудач")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    if len(sys.argv) > 1 and sys.argv[1] in (
            "prepare", "gpuinfo", "assign", "preflight", "collect",
            "status", "merge", "analyze"):
        import k38_grip_run as kgr
        return kgr.main(sys.argv[1:])
    ap = argparse.ArgumentParser(description="K-38: схват")
    ap.add_argument("cmd", choices=("audit",))
    ap.add_argument("--archive", default="reports/k37c/runs/c2")
    a = ap.parse_args()
    try:
        return audit(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
