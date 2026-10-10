#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15f M1-cold: неизменяемый снимок завершённого результата.

  make    ЗАКЛЮЧИТЕЛЬНАЯ проверка на исходниках прогона (план M1-cold
          проверяется штатной verify_plan(): все 13 зафиксированных
          исходников обязаны совпасть) и запись манифеста: коммит прогона,
          SHA плана, переписи, результата, базиса, отчётов и КАЖДОЙ пары
          JSON+npz, вошедшей в анализ. Официальный анализ M1 повторяется
          в процессе и обязан дать в точности сохранённый результат.
          Делается ОДИН раз, до любых правок исходников; повторно снимок
          не перезаписывается;
  verify  проверка снимка: все перечисленные файлы на месте и не изменены,
          состав каталога M1-cold совпадает с манифестом, итог технически
          корректен. ТЕКУЩИЙ КОД НЕ СВЕРЯЕТСЯ: его фиксирует план
          потребителя (K-15g).

Воспроизведение анализа исходным кодом: git checkout <run_commit> и
команда из поля reproduce.

План, перепись и результат M1-cold не изменяются.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SNAP = "reports/k15f/m1cold_snapshot.json"
RESULT = "reports/k15f/m1cold_result.json"
BASIS_REPORT = "reports/k15f/basis_s0.json"
RUN_LOG = "logs/k15f/m1cold_run.log"
KIND = "k15f_m1cold_snapshot"


def sha_full(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def git(*args):
    """ТОЛЬКО чтение git."""
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=False)


def run_commit(log=RUN_LOG):
    """Коммит, на котором шёл run: из строки СТАРТ лога."""
    commits = re.findall(r"режим run, .*коммит (\w+)", open(log).read())
    if not commits or len(set(commits)) != 1:
        raise SystemExit(f"{log}: коммит прогона не однозначен: "
                         f"{sorted(set(commits))}")
    return commits[0]


def cold_jsons(cold_dir, labels):
    """JSON, входящие в официальный анализ (как в k15f_m1_cold.sh)."""
    q0 = sorted(glob.glob(os.path.join(cold_dir, "q0_t*_i*.json")))
    arts = []
    for lab in labels:
        arts += sorted(glob.glob(os.path.join(cold_dir, f"{lab}_t*_i*.json")))
    return q0, arts


def official(census_path, q0_files, art_files, basis_report):
    """Официальный анализ M1 с теми же входами, что в k15f_m1_cold.sh."""
    import k15f_measure_subspace as kms
    exp = dict(kms.SETUP_EXPECTED, script_sha1=kms._sha(os.path.join(
        HERE, "k9h_multiarm_gate.py")))
    brep = json.load(open(basis_report))
    q0 = [(p, json.load(open(p))) for p in q0_files]
    res = kms.m1(json.load(open(census_path)), q0, kms.load_arts(art_files),
                 max_grip_flip=brep.get("max_grip_flip_share"),
                 expected_setup=exp, basis_report=brep)
    return json.loads(json.dumps(res))


def pair_entry(p):
    d = json.load(open(p))
    npz = os.path.join(os.path.dirname(p), d["actions_npz"])
    return dict(json=p, json_sha1=sha_full(p), npz=npz,
                npz_sha1=sha_full(npz), label=str(d["arm_label"]),
                task=int(d["task_id"]), block=int(d["init_start"]),
                argv_init_starts=str((d.get("argv") or {}).get(
                    "init_starts")))


def make(skip_git=False):
    import k15f_m1_cold as mc
    import k15g_local_probe as pr
    if os.path.exists(SNAP):
        raise SystemExit(f"{SNAP} уже записан: снимок не перезаписывается")
    plan = mc.verify_plan()        # SystemExit, если исходники изменились
    p = []
    commit = head = None
    if not skip_git:
        commit = run_commit()
        head = git("rev-parse", "HEAD").stdout.strip()
        files = [os.path.join("experiments", f) for f in mc.CODE_FILES]
        if git("diff", "--quiet", commit, "HEAD", "--", *files).returncode:
            p.append(f"зафиксированные исходники HEAD отличаются от "
                     f"коммита прогона {commit}")
        if git("diff", "--quiet", "HEAD", "--", *files).returncode:
            p.append("зафиксированные исходники изменены в рабочем дереве")
    cen = json.load(open(plan["new_census"]))
    res = json.load(open(RESULT))
    if res.get("code") != 0 or res.get("technical"):
        p.append(f"итог M1-cold: код {res.get('code')}, "
                 f"{res.get('technical')}")
    if res.get("F") != cen.get("F"):
        p.append("F итога не равно F переписи")
    q0, arts = cold_jsons(plan["cold_dir"], mc.LABELS)
    for f in q0 + arts:
        d = json.load(open(f))
        blk = int(d["init_start"])
        p += pr.validate_block(f, d["arm_label"], d["task_id"], blk,
                               require_single=blk != 0,
                               setup=pr.expected_setup())
        if blk == 0 and not mc.first_in_process(d):
            p.append(f"{os.path.basename(f)}: блок 0 не первый в процессе")
    re_ = official(plan["new_census"], q0, arts, plan["unchanged"][
        "basis_report"])
    if re_ != res:
        diff = sorted(k for k in set(re_) | set(res)
                      if re_.get(k) != res.get(k))
        p.append(f"официальный анализ не воспроизвёл результат: {diff}")
    if p:
        raise SystemExit("снимок M1-cold не записан: " + "; ".join(p[:8]))
    u = plan["unchanged"]
    snap = dict(
        kind=KIND, created=datetime.datetime.now().isoformat(
            timespec="seconds"),
        run_commit=commit, head_at_snapshot=head,
        plan=mc.PLAN, plan_sha1=sha_full(mc.PLAN),
        census=plan["new_census"], census_sha1=sha_full(plan["new_census"]),
        old_census=mc.CENSUS_OLD, old_census_sha1=sha_full(mc.CENSUS_OLD),
        result=RESULT, result_sha1=sha_full(RESULT),
        basis=u["basis"], basis_sha1=sha_full(u["basis"]),
        identity_report=u["identity_report"],
        identity_report_sha1=sha_full(u["identity_report"]),
        basis_report=u["basis_report"],
        basis_report_sha1=sha_full(u["basis_report"]),
        cold_dir=plan["cold_dir"], code_sha1_at_run=u.get("code_sha1"),
        pairs=[pair_entry(f) for f in q0 + arts],
        verdict=dict(code=res["code"], F=res["F"], R_L=res["R_L"],
                     R_C=res["R_C"], L_only=res["L_only"],
                     C_only=res["C_only"], decision=res["decision"]),
        reproduce=dict(
            checkout=commit,
            argv=["python3", "experiments/k15f_measure_subspace.py",
                  "--mode", "m1", "--census", plan["new_census"],
                  "--q0-arts", *q0, "--arts", *arts, "--basis-report",
                  u["basis_report"], "--out", "<новый путь>"]))
    os.makedirs(os.path.dirname(SNAP), exist_ok=True)
    with open(SNAP + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(snap, fh, ensure_ascii=False, indent=1)
    os.replace(SNAP + ".tmp", SNAP)
    print(f"  снимок M1-cold: пар {len(snap['pairs'])}, коммит прогона "
          f"{commit}, итог {snap['verdict']}")
    print(f"  {SNAP} sha1 {sha_full(SNAP)}")
    return 0


def problems(path=SNAP):
    """Проверка снимка без сверки текущего кода. Список проблем."""
    try:
        s = json.load(open(path))
    except Exception as e:                  # noqa: BLE001 — любой сбой = отказ
        return [f"{path}: не читается ({type(e).__name__})"]
    if s.get("kind") != KIND:
        return [f"{path}: не снимок M1-cold"]
    p = []
    for k in ("plan", "census", "old_census", "result", "basis",
              "identity_report", "basis_report"):
        f = s[k]
        if not os.path.exists(f) or sha_full(f) != s[k + "_sha1"]:
            p.append(f"{k} {f}: изменён или отсутствует")
    listed = set()
    for e in s["pairs"]:
        listed.add(os.path.normpath(e["json"]))
        for k in ("json", "npz"):
            if not os.path.exists(e[k]) or sha_full(e[k]) != e[k + "_sha1"]:
                p.append(f"{e[k]}: изменён или отсутствует")
    now = {os.path.normpath(f) for f in glob.glob(os.path.join(
        s["cold_dir"], "*_t*_i*.json"))}
    if now != listed:
        p.append(f"состав {s['cold_dir']} изменился: лишние "
                 f"{sorted(now - listed)[:3]}, нет {sorted(listed - now)[:3]}")
    if not p:
        res = json.load(open(s["result"]))
        cen = json.load(open(s["census"]))
        if res.get("code") != 0 or res.get("technical") \
                or cen.get("code") != 0 or cen.get("technical"):
            p.append("итог или перепись M1-cold технически не корректны")
        v = s["verdict"]
        if any(res.get(k) != v[k] for k in v):
            p.append("итог не совпадает с вердиктом снимка")
    return p


def verify(path=SNAP):
    """Снимок или SystemExit."""
    p = problems(path)
    if p:
        raise SystemExit("снимок M1-cold не прошёл проверку: " + "; ".join(
            p[:6]))
    return json.load(open(path))


def selftest():
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cold = os.path.join(td, "cold")
        os.makedirs(cold)
        pairs = []
        for lab in ("q0", "l0p"):
            js = os.path.join(cold, f"{lab}_t8_i5.json")
            npz = os.path.join(cold, f"{lab}_t8_i5.actions.npz")
            open(npz, "wb").write(lab.encode())
            json.dump(dict(arm_label=lab, task_id=8, init_start=5,
                           actions_npz=os.path.basename(npz),
                           argv=dict(init_starts="5")), open(js, "w"))
            pairs.append(pair_entry(js))
        res = dict(code=0, technical=[], F=14, R_L=12, R_C=11, L_only=1,
                   C_only=0, decision="неясно")
        files = {}
        for k, obj in (("plan", {}), ("census", dict(code=0, technical=[])),
                       ("old_census", {}), ("result", res),
                       ("basis_report", {}), ("identity_report", {})):
            files[k] = os.path.join(td, f"{k}.json")
            json.dump(obj, open(files[k], "w"))
        files["basis"] = os.path.join(td, "basis.pt")
        open(files["basis"], "wb").write(b"B")
        snap = dict(kind=KIND, cold_dir=cold, pairs=pairs,
                    verdict={k: res[k] for k in ("code", "F", "R_L", "R_C",
                                                 "L_only", "C_only",
                                                 "decision")})
        for k, f in files.items():
            snap[k], snap[k + "_sha1"] = f, sha_full(f)
        sp = os.path.join(td, "snap.json")
        json.dump(snap, open(sp, "w"))
        assert problems(sp) == [], problems(sp)
        # изменение ЛЮБОГО артефакта — отказ
        for f in list(files.values()) + [e["json"] for e in pairs] + [
                e["npz"] for e in pairs]:
            orig = open(f, "rb").read()
            open(f, "ab").write(b" ")
            assert problems(sp), f
            open(f, "wb").write(orig)
        assert problems(sp) == []
        # лишний или пропавший блок в каталоге — отказ
        extra = os.path.join(cold, "r0p_t8_i5.json")
        shutil.copy(pairs[1]["json"], extra)
        assert any("состав" in x for x in problems(sp))
        os.remove(extra)
        # итог с техническим отказом не принимается, даже если снимок
        # переписан под него
        bad = dict(snap, verdict=dict(snap["verdict"], code=3))
        json.dump(dict(res, code=3, technical=["x"]),
                  open(files["result"], "w"))
        bad["result_sha1"] = sha_full(files["result"])
        json.dump(bad, open(sp, "w"))
        assert any("технически" in x for x in problems(sp))
        assert problems(os.path.join(td, "нет.json"))
    # коммит прогона — однозначно из лога
    with tempfile.TemporaryDirectory() as td:
        lg = os.path.join(td, "run.log")
        open(lg, "w").write("=== СТАРТ x === режим run, карта cuda:1, коммит "
                            "0307626\n…\n=== СТАРТ y === режим run, карта "
                            "cuda:1, коммит 0307626\n")
        assert run_commit(lg) == "0307626"
        open(lg, "a").write("=== СТАРТ z === режим run, карта cuda:1, "
                            "коммит abcdef0\n")
        try:
            run_commit(lg)
            raise AssertionError("принят неоднозначный коммит")
        except SystemExit:
            pass
    print("самопроверка k15f_m1cold_snapshot пройдена: изменение каждого "
          "артефакта, состав каталога, технический отказ, коммит прогона")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-15f M1-cold: снимок")
    ap.add_argument("mode", choices=("make", "verify", "selftest"))
    ap.add_argument("--snapshot", default=SNAP)
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    if a.mode == "make":
        return make()
    s = verify(a.snapshot)
    print(f"  снимок M1-cold проверен: пар {len(s['pairs'])}, итог "
          f"{s['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
