#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15f M1: восстановление вторых блоков после утечки состояния среды.

Код и протокол M1 не меняются. Скрипт только:

  cold-check  сверяет холодные (одиночный процесс, только блок 5) прогоны
              q0 с переписью M1 по init_hash, init_hash_full, action_sha1
              ВСЕЙ исполненной траектории, success и done_step — для задач
              2, 8 и 9. Холодные q0 лежат отдельно (reports/k15f/diag/);
              перепись не трогается. Любое расхождение — код 3: восстановление
              останавливается, перепись НЕ подменяется;
  quarantine  переносит блок 5 указанных задач у всех 16 рук в карантин и
              пишет журнал: какие пары JSON/NPZ, их отпечатки, причина;
  journal     после дозапуска M1 дописывает в журнал отпечатки и исходы
              новых пар.

Журнал: reports/k15f/m1_recovery.json (дополняется, не перезаписывается).
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

LABELS = [f"{f}{j}{s}" for f in ("l", "r") for j in range(4)
          for s in ("p", "m")]
FIELDS = ("init_hash", "init_hash_full", "action_sha1", "success",
          "done_step")
JOURNAL = "reports/k15f/m1_recovery.json"
REASON = ("утечка состояния среды между блоками одного процесса "
          "k9h (задача 8: старт блока 5 зависел от блока 0); вторые блоки "
          "пересчитываются одиночными процессами симметрично у всех рук")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]


def m1_dir():
    return sorted(glob.glob("reports/k15f/m1/s101/*/"),
                  key=os.path.getmtime)[-1]


def compare_cold(census_json, cold_json):
    """Холодный q0 против переписи: список расхождений по эпизодам."""
    c = {e["init_state_id"]: e for e in json.load(open(census_json))
         ["episodes"]}
    d = {e["init_state_id"]: e for e in json.load(open(cold_json))
         ["episodes"]}
    bad = []
    if set(c) != set(d):
        bad.append(f"состав эпизодов {sorted(d)} против {sorted(c)}")
    for s in sorted(set(c) & set(d)):
        for f in FIELDS:
            if str(c[s].get(f)) != str(d[s].get(f)):
                bad.append(f"состояние {s}: {f} {d[s].get(f)!r} против "
                           f"переписи {c[s].get(f)!r}")
    return bad


def load_journal():
    if os.path.exists(JOURNAL):
        return json.load(open(JOURNAL))
    return dict(kind="k15f_m1_recovery", entries=[])


def save_journal(j):
    os.makedirs(os.path.dirname(JOURNAL), exist_ok=True)
    with open(JOURNAL + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(j, fh, ensure_ascii=False, indent=1)
    os.replace(JOURNAL + ".tmp", JOURNAL)


def cold_check(tasks, diag_dir):
    D = m1_dir()
    res, ok = {}, True
    for t in tasks:
        cen = os.path.join(D, f"q0_t{t}_i5.json")
        cold = os.path.join(diag_dir, f"q0_t{t}_i5.json")
        if not os.path.exists(cold):
            res[t] = ["нет холодного прогона"]
            ok = False
            continue
        res[t] = compare_cold(cen, cold)
        ok &= not res[t]
        print(f"  задача {t}: холодный q0 "
              f"{'ВОСПРОИЗВОДИТ перепись' if not res[t] else 'НЕ совпал'}"
              + (f": {res[t][:3]}" if res[t] else ""))
    j = load_journal()
    j["entries"].append(dict(
        step="cold_check", time=_now(), m1_dir=D, tasks=tasks,
        cold_files={str(t): sha(os.path.join(diag_dir, f"q0_t{t}_i5.json"))
                    for t in tasks if os.path.exists(
                        os.path.join(diag_dir, f"q0_t{t}_i5.json"))},
        census_files={str(t): sha(os.path.join(D, f"q0_t{t}_i5.json"))
                      for t in tasks},
        result={str(k): v for k, v in res.items()}, passed=ok))
    save_journal(j)
    return 0 if ok else 3


def quarantine(tasks):
    D = m1_dir()
    Q = os.path.join("reports/k15f/m1_quarantine",
                     datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
    os.makedirs(Q)
    moved = []
    for t in tasks:
        for lab in LABELS:
            js = os.path.join(D, f"{lab}_t{t}_i5.json")
            npz = os.path.join(D, f"{lab}_t{t}_i5.actions.npz")
            if not os.path.exists(js):
                moved.append(dict(arm=lab, task=t, missing=True))
                continue
            rec = dict(arm=lab, task=t, block=5, json_sha1=sha(js),
                       npz_sha1=sha(npz) if os.path.exists(npz) else None,
                       outcomes=[e["success"] for e in
                                 json.load(open(js))["episodes"]])
            shutil.move(js, Q)
            if os.path.exists(npz):
                shutil.move(npz, Q)
            moved.append(rec)
    j = load_journal()
    j["entries"].append(dict(step="quarantine", time=_now(), m1_dir=D,
                             quarantine_dir=Q, tasks=tasks, reason=REASON,
                             moved=moved))
    save_journal(j)
    print(f"  в карантине {Q}: {sum(1 for m in moved if not m.get('missing'))}"
          f" пар; журнал {JOURNAL}")
    return 0


def journal_new(tasks):
    D = m1_dir()
    new = []
    for t in tasks:
        for lab in LABELS:
            js = os.path.join(D, f"{lab}_t{t}_i5.json")
            if not os.path.exists(js):
                new.append(dict(arm=lab, task=t, missing=True))
                continue
            d = json.load(open(js))
            new.append(dict(arm=lab, task=t, block=5, json_sha1=sha(js),
                            npz_sha1=sha(os.path.join(D, d["actions_npz"])),
                            outcomes=[e["success"] for e in d["episodes"]],
                            init_starts=d.get("init_start")))
    j = load_journal()
    j["entries"].append(dict(step="new_results", time=_now(), m1_dir=D,
                             tasks=tasks, results=new))
    save_journal(j)
    miss = [n for n in new if n.get("missing")]
    print(f"  новых пар {len(new) - len(miss)}, отсутствует {len(miss)}")
    return 3 if miss else 0


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ep = [dict(init_state_id=s, init_hash=f"h{s}", init_hash_full=f"H{s}",
                   action_sha1=f"{s:016x}", success=s % 2 == 0, done_step=10)
              for s in range(5, 10)]
        a, b = os.path.join(td, "a.json"), os.path.join(td, "b.json")
        json.dump(dict(episodes=ep), open(a, "w"))
        json.dump(dict(episodes=ep), open(b, "w"))
        assert compare_cold(a, b) == []
        for f, v in (("action_sha1", "f" * 16), ("success", None),
                     ("done_step", 11), ("init_hash", "x")):
            ep2 = [dict(e) for e in ep]
            ep2[1][f] = (not ep2[1][f]) if f == "success" else v
            json.dump(dict(episodes=ep2), open(b, "w"))
            assert any(f in x for x in compare_cold(a, b)), f
        json.dump(dict(episodes=ep[:4]), open(b, "w"))
        assert any("состав" in x for x in compare_cold(a, b))
    assert len(LABELS) == 16
    print("самопроверка k15f_m1_recovery пройдена")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-15f M1: восстановление")
    ap.add_argument("mode", choices=("cold-check", "quarantine", "journal",
                                     "selftest"))
    ap.add_argument("--tasks", default="2,8,9")
    ap.add_argument("--diag-dir", default="reports/k15f/diag")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    tasks = [int(x) for x in a.tasks.split(",")]
    return dict(cold_check=lambda: cold_check(tasks, a.diag_dir),
                quarantine=lambda: quarantine(tasks),
                journal=lambda: journal_new(tasks))[
        a.mode.replace("-", "_")]()


if __name__ == "__main__":
    sys.exit(main())
