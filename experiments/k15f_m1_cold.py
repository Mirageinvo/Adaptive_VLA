#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15f M1-cold: восстановление M1 на холодных стартах (отдельный набор).

ПРИЧИНА (техническое отклонение от первоначального способа запуска,
принятое после обнаружения ошибки). В M1 блоки 0 и 5 одной руки шли в
одном процессе k9h; reset между блоками восстанавливал среду не полностью.
Холодный (одиночный) прогон q0 блока 5 разошёлся с переписью на задачах 2
и 9 во всех пяти состояниях, на задаче 8 совпал, но у 14 рук старт блока 5
зависел от их блока 0. Поэтому все ВТОРЫЕ блоки пересчитываются
одиночными процессами в отдельном каталоге; старые результаты сохраняются.

ЧТО НЕ МЕНЯЕТСЯ: базис, амплитуда, контроль, сид 101, контракт раскатки,
штатные census() и m1() из k15f_measure_subspace, правило RULE.

РЕЖИМЫ:
  census    собирает в каталог M1-cold q0 блока 0 из исходного M1 (первый
            блок своего процесса — холодный; проверяется по argv) и
            холодные q0 блока 5 всех 10 задач (одиночный процесс,
            init_starts == "5"), сверяет контракт и строит НОВУЮ перепись
            штатной census(). Пишет её отдельно, не трогая исходную, и
            печатает отличия от исходной (F, провалы, диагностика, блоки).
  plan      фиксирует план восстановления: исходные M1 и перепись с
            отпечатками, новая перепись, переиспользуемые блоки 0 рук (с
            отпечатками и проверкой холодности), задания на блоки 5 по
            НОВОЙ переписи для всех 16 рук, неизменные параметры.
  assemble  копирует переиспользуемые блоки 0 рук в каталог M1-cold со
            сверкой отпечатков.
  jobs      печатает «задача рука блок» для раннера.
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

COLD_DIR = "reports/k15f/m1cold"
DIAG = "reports/k15f/diag"
CENSUS_OLD = "reports/k15f/m1_census.json"
CENSUS_NEW = "reports/k15f/m1cold_census.json"
PLAN = "reports/k15f/m1cold_plan.json"
TASKS = list(range(10))
LABELS = [f"{f}{j}{s}" for f in ("l", "r") for j in range(4)
          for s in ("p", "m")]


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]


def m1_dir():
    return sorted(glob.glob("reports/k15f/m1/s101/*/"),
                  key=os.path.getmtime)[-1]


def first_in_process(d):
    """Блок шёл ПЕРВЫМ в своём процессе (по argv харнесса)."""
    starts = str((d.get("argv") or {}).get("init_starts") or
                 d.get("init_start"))
    first = starts.replace(",", " ").split()[0]
    return int(first) == int(d["init_start"])


def single_block(d):
    starts = str((d.get("argv") or {}).get("init_starts") or "")
    return starts.replace(",", " ").split() == [str(d["init_start"])]


def copy_pair(src_json, dst_dir):
    d = json.load(open(src_json))
    npz = os.path.join(os.path.dirname(src_json), d["actions_npz"])
    os.makedirs(dst_dir, exist_ok=True)
    for f in (src_json, npz):
        dst = os.path.join(dst_dir, os.path.basename(f))
        if os.path.exists(dst) and sha(dst) != sha(f):
            raise SystemExit(f"{dst} уже есть и отличается")
        shutil.copy2(f, dst)
    return dict(json=os.path.basename(src_json), json_sha1=sha(src_json),
                npz_sha1=sha(npz))


def census_mode():
    import k15f_measure_subspace as kms
    src = m1_dir()
    os.makedirs(COLD_DIR, exist_ok=True)
    prov, problems = [], []
    for t in TASKS:
        b0 = os.path.join(src, f"q0_t{t}_i0.json")
        d0 = json.load(open(b0))
        if not first_in_process(d0):
            problems.append(f"q0 t{t} блок 0 не первый в процессе")
        prov.append(dict(task=t, block=0, source=b0, cold_by="первый блок "
                         "процесса", **copy_pair(b0, COLD_DIR)))
        b5 = os.path.join(DIAG, f"q0_t{t}_i5.json")
        if not os.path.exists(b5):
            problems.append(f"нет холодного q0 t{t} блок 5 ({b5})")
            continue
        d5 = json.load(open(b5))
        if not single_block(d5) or d5.get("policy") != "depthrvq":
            problems.append(f"q0 t{t} блок 5: не одиночный процесс q0")
        prov.append(dict(task=t, block=5, source=b5,
                         cold_by="одиночный процесс", **copy_pair(b5,
                                                                COLD_DIR)))
    if problems:
        raise SystemExit("перепись M1-cold не собрана: " + "; ".join(
            problems))
    exp = dict(kms.SETUP_EXPECTED, script_sha1=kms._sha(os.path.join(
        HERE, "k9h_multiarm_gate.py")))
    q0 = [(p, json.load(open(p))) for p in
          sorted(glob.glob(os.path.join(COLD_DIR, "q0_t*_i*.json")))]
    new = kms.census(q0, exp)
    old = json.load(open(CENSUS_OLD))
    new.update(m1cold=dict(
        source_m1_dir=src, old_census=CENSUS_OLD,
        old_census_sha1=sha(CENSUS_OLD), q0_provenance=prov,
        reason=__doc__.split("ПРИЧИНА")[1].split("ЧТО НЕ МЕНЯЕТСЯ")[0]
        .strip(),
        diff=dict(F=[old["F"], new["F"]],
                  fails_added=sorted(map(list, set(map(tuple, new["fails"]))
                                         - set(map(tuple, old["fails"])))),
                  fails_removed=sorted(map(list, set(map(tuple,
                                                         old["fails"]))
                                           - set(map(tuple, new["fails"])))),
                  diag_changed=old["diag"] != new["diag"],
                  blocks=[old["blocks"], new["blocks"]])))
    with open(CENSUS_NEW + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(new, fh, ensure_ascii=False, indent=1)
    os.replace(CENSUS_NEW + ".tmp", CENSUS_NEW)
    dd = new["m1cold"]["diff"]
    print(f"  перепись M1-cold: F {dd['F'][0]} -> {dd['F'][1]}; провалы "
          f"добавлены {dd['fails_added']}, убраны {dd['fails_removed']}; "
          f"диагностика изменилась: {dd['diag_changed']}")
    print(f"  блоки: {new['blocks']}; мощность "
          f"{'достаточна' if new['powered'] else 'НЕДОСТАТОЧНА'}; код "
          f"{new['code']}" + (f"; {new['technical'][:4]}"
                             if new["technical"] else ""))
    print("  успех q0 по задачам: " + ", ".join(
        f"{t}: {v:.1f}" for t, v in new["per_task_sr"].items()))
    return int(new["code"])


def plan_mode(basis, identity, basis_report):
    if os.path.exists(PLAN):
        raise SystemExit(f"{PLAN} уже зафиксирован")
    src = m1_dir()
    new = json.load(open(CENSUS_NEW))
    reuse, jobs, problems = [], [], []
    for t_s, blocks in new["blocks"].items():
        t = int(t_s)
        for b in blocks:
            for lab in LABELS:
                if b == 0:
                    p = os.path.join(src, f"{lab}_t{t}_i0.json")
                    if not os.path.exists(p):
                        problems.append(f"нет блока 0 {lab} t{t}")
                        continue
                    if not first_in_process(json.load(open(p))):
                        problems.append(f"{lab} t{t} блок 0 не первый")
                    d = json.load(open(p))
                    reuse.append(dict(arm=lab, task=t, block=0, source=p,
                                      json_sha1=sha(p),
                                      npz_sha1=sha(os.path.join(
                                          src, d["actions_npz"]))))
                else:
                    jobs.append(dict(arm=lab, task=t, block=int(b)))
    if problems:
        raise SystemExit("план не собран: " + "; ".join(problems[:6]))
    import k15f_measure_subspace as kms
    plan = dict(
        kind="k15f_m1cold_plan",
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        source_m1_dir=src, old_census_sha1=sha(CENSUS_OLD),
        new_census=CENSUS_NEW, new_census_sha1=sha(CENSUS_NEW),
        cold_dir=COLD_DIR, reused_blocks=reuse, jobs=jobs,
        unchanged=dict(basis=basis, basis_sha1=sha(basis),
                       identity_report=identity,
                       identity_report_sha1=sha(identity),
                       basis_report=basis_report,
                       basis_report_sha1=sha(basis_report), seed=101,
                       rule=kms.RULE, setup=kms.SETUP_EXPECTED,
                       harness_sha1=sha(os.path.join(
                           HERE, "k9h_multiarm_gate.py"))),
        note=("старые вторые блоки рук в анализ M1-cold не входят; выбор "
              "блоков — по правилу новой переписи, без учёта прежних "
              "результатов кандидатов"))
    with open(PLAN + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(plan, fh, ensure_ascii=False, indent=1)
    os.replace(PLAN + ".tmp", PLAN)
    print(f"  план M1-cold: переиспользуется блоков 0 — {len(reuse)}, "
          f"заданий на холодные вторые блоки — {len(jobs)}; {PLAN}")
    return 0


def assemble_mode():
    plan = json.load(open(PLAN))
    for r in plan["reused_blocks"]:
        if sha(r["source"]) != r["json_sha1"]:
            raise SystemExit(f"{r['source']} изменился после плана")
        copy_pair(r["source"], plan["cold_dir"])
    print(f"  скопировано блоков 0 рук: {len(plan['reused_blocks'])}")
    return 0


def jobs_mode():
    plan = json.load(open(PLAN))
    for j in plan["jobs"]:
        print(j["task"], j["arm"], j["block"])
    return 0


def selftest():
    assert first_in_process(dict(init_start=0, argv=dict(init_starts="0,5")))
    assert not first_in_process(dict(init_start=5,
                                     argv=dict(init_starts="0,5")))
    assert single_block(dict(init_start=5, argv=dict(init_starts="5")))
    assert not single_block(dict(init_start=5, argv=dict(init_starts="0,5")))
    assert len(LABELS) == 16
    print("самопроверка k15f_m1_cold пройдена")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-15f M1-cold")
    ap.add_argument("mode", choices=("census", "plan", "assemble", "jobs",
                                     "selftest"))
    ap.add_argument("--basis", default="data/k15f/basis_s0.pt")
    ap.add_argument("--identity", default="reports/k15f/identity_cuda1.json")
    ap.add_argument("--basis-report", default="reports/k15f/basis_s0.json")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    if a.mode == "census":
        return census_mode()
    if a.mode == "plan":
        return plan_mode(a.basis, a.identity, a.basis_report)
    if a.mode == "assemble":
        return assemble_mode()
    return jobs_mode()


if __name__ == "__main__":
    sys.exit(main())
