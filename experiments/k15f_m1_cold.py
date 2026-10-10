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
    """Блок шёл ПЕРВЫМ в своём процессе — ТОЛЬКО по явному argv харнесса.
    Отсутствие argv подтверждением не считается."""
    starts = (d.get("argv") or {}).get("init_starts")
    if not starts:
        return False
    first = str(starts).replace(",", " ").split()[0]
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
    # ПЕРЕПИСЬ, НА КОТОРУЮ УЖЕ ССЫЛАЕТСЯ ПЛАН, НЕ ПЕРЕЗАПИСЫВАЕТСЯ
    if os.path.exists(PLAN):
        raise SystemExit(f"{PLAN} уже зафиксирован и ссылается на "
                         f"{CENSUS_NEW}: перепись не пересобирается")
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
    if new.get("code") != 0 or new.get("technical"):
        raise SystemExit(f"перепись технически не корректна: "
                         f"{new.get('technical')}")
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
    jobs = derive_jobs(new)
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
    verify_plan()
    print(f"  план M1-cold: переиспользуется блоков 0 — {len(reuse)}, "
          f"заданий на холодные вторые блоки — {len(jobs)}; {PLAN}")
    return 0


def assemble_mode():
    import k15g_local_probe as pr
    plan = verify_plan()
    for r in plan["reused_blocks"]:
        d = json.load(open(r["source"]))
        npz = os.path.join(os.path.dirname(r["source"]), d["actions_npz"])
        if sha(r["source"]) != r["json_sha1"] or sha(npz) != r["npz_sha1"]:
            raise SystemExit(f"{r['source']} или его npz изменились после "
                             f"плана")
        got = copy_pair(r["source"], plan["cold_dir"])
        if got["json_sha1"] != r["json_sha1"] or \
                got["npz_sha1"] != r["npz_sha1"]:
            raise SystemExit(f"копия {r['source']} не совпала с планом")
        dst = os.path.join(plan["cold_dir"], os.path.basename(r["source"]))
        prob = pr.validate_block(dst, r["arm"], r["task"], 0,
                                 setup=pr.expected_setup())
        if prob:
            raise SystemExit(f"собранная пара не прошла проверку: {prob[:3]}")
    print(f"  скопировано и проверено блоков 0 рук: "
          f"{len(plan['reused_blocks'])}")
    return 0


def derive_jobs(census):
    jobs = []
    for t_s, blocks in sorted(census["blocks"].items(), key=lambda x:
                              int(x[0])):
        for b in blocks:
            if int(b) != 0:
                for lab in LABELS:
                    jobs.append(dict(arm=lab, task=int(t_s), block=int(b)))
    return jobs


def verify_plan():
    """Перед assemble/run/analyze: все входы и код совпадают с планом,
    перепись технически корректна и воспроизводится штатной census() по
    q0 каталога M1-cold, задания получены из неё. Иначе — отказ."""
    import k15f_measure_subspace as kms
    import k15g_local_probe as pr
    plan = json.load(open(PLAN))
    p = []
    u = plan["unchanged"]
    for path, key in ((plan["new_census"], plan["new_census_sha1"]),
                      (u["basis"], u["basis_sha1"]),
                      (u["identity_report"], u["identity_report_sha1"]),
                      (u["basis_report"], u["basis_report_sha1"])):
        if not os.path.exists(path) or sha(path) != key:
            p.append(f"{path}: изменён или отсутствует")
    if sha(os.path.join(HERE, "k9h_multiarm_gate.py")) != u["harness_sha1"]:
        p.append("харнесс изменён после плана")
    if u["setup"] != kms.SETUP_EXPECTED or u["rule"] != kms.RULE:
        p.append("контракт раскатки или правило изменились после плана")
    cen = json.load(open(plan["new_census"]))
    if cen.get("code") != 0 or cen.get("technical"):
        p.append(f"перепись технически не корректна: {cen.get('technical')}")
    q0 = [(f, json.load(open(f))) for f in sorted(glob.glob(os.path.join(
        plan["cold_dir"], "q0_t*_i*.json")))]
    for f, d in q0:
        cold = (first_in_process(d) if int(d["init_start"]) == 0
                else single_block(d))
        if not cold:
            p.append(f"{os.path.basename(f)}: не холодный старт")
        p += pr.validate_block(f, "q0", d["task_id"], d["init_start"],
                               require_single=int(d["init_start"]) != 0,
                               setup=pr.expected_setup())
    re_ = kms.census(q0, pr.expected_setup())
    for k in ("F", "fails", "diag", "blocks", "q0_files", "code"):
        if json.dumps(re_.get(k), sort_keys=True) != json.dumps(
                cen.get(k), sort_keys=True):
            p.append(f"перепись не воспроизводится по q0: поле {k}")
    if plan["jobs"] != derive_jobs(cen):
        p.append("задания не совпадают с выведенными из переписи")
    if p:
        raise SystemExit("план M1-cold не прошёл проверку: " + "; ".join(
            p[:6]))
    return plan


def common_args():
    """Аргументы раскатки — из ЗАФИКСИРОВАННОГО плана."""
    st = verify_plan()["unchanged"]["setup"]
    return (f"--ckpt {st['ckpt']} --task-suite {st['suite']} --n-envs "
            f"{st['n_envs']} --seed {st['seed']} --rollout-seed-mode "
            f"{st['rollout_seed_mode']} --ensemble {st['ensemble']} "
            f"--horizon {st['horizon']} --max-steps {st['max_steps']} "
            f"--waiting-steps {st['waiting_steps']}")


def jobs_mode():
    plan = verify_plan()
    for j in plan["jobs"]:
        print(j["task"], j["arm"], j["block"])
    return 0


def selftest():
    assert first_in_process(dict(init_start=0, argv=dict(init_starts="0,5")))
    assert not first_in_process(dict(init_start=0))      # нет argv — нет
    cen = dict(blocks={"0": [0], "8": [0, 5], "2": [0, 5]})
    jb = derive_jobs(cen)
    assert len(jb) == 32 and {j["task"] for j in jb} == {2, 8}
    assert all(j["block"] == 5 for j in jb)
    assert not first_in_process(dict(init_start=5,
                                     argv=dict(init_starts="0,5")))
    assert single_block(dict(init_start=5, argv=dict(init_starts="5")))
    assert not single_block(dict(init_start=5, argv=dict(init_starts="0,5")))
    assert len(LABELS) == 16
    _chain_test()
    print("самопроверка k15f_m1_cold пройдена: холодность по argv, задания, "
          "сквозная цепочка перепись -> план -> проверка -> сборка, "
          "мутации")
    return 0


def _fake_art(dirpath, label, t, blk, starts, succ, policy="depthrvq"):
    """Синтетическая пара JSON+npz в формате харнесса."""
    import numpy as np
    import k15f_measure_subspace as kms
    import k15g_local_probe as pr
    rng = np.random.default_rng(hash((label, t, blk)) % 2**32)
    A = rng.standard_normal((16, 5, 7)).astype(np.float32)
    ds = np.full(5, 12)
    eps = []
    for k in range(5):
        h, end = pr.episode_action_sha(A, k, ds[k])
        eps.append(dict(success=bool(succ(t, blk + k)),
                        init_state_id=blk + k, env_index=k,
                        init_hash=f"i{t}{blk + k}",
                        init_hash_full=f"I{t}{blk + k}", rollout_seed=101,
                        done_step=int(ds[k]), own_steps=end,
                        action_sha1=h))
    base = os.path.join(dirpath, f"{label}_t{t}_i{blk}")
    np.savez(base + ".actions.npz", actions=A, done_step=ds,
             init_state_id=np.arange(blk, blk + 5),
             action_sha1=np.asarray([e["action_sha1"] for e in eps]))
    d = dict(kms.SETUP_EXPECTED, script_sha1=sha(os.path.join(
        HERE, "k9h_multiarm_gate.py")), episodes=eps, arm_label=label,
        task_id=t, init_start=blk, policy=policy, levels=1,
        argv=dict(init_starts=starts, depth_rvq_mode="fast"),
        actions_npz=os.path.basename(base) + ".actions.npz",
        joint=dict(model_fingerprint=f"fp{label}"))
    d["actions_npz_sha1"] = sha(base + ".actions.npz")
    json.dump(d, open(base + ".json", "w"))
    return base + ".json"


def _chain_test():
    import tempfile
    g = globals()
    keep = {k: g[k] for k in ("COLD_DIR", "DIAG", "CENSUS_OLD",
                              "CENSUS_NEW", "PLAN", "m1_dir")}
    fails = {(8, 0), (8, 6), (9, 7), (2, 9)}

    def succ(t, s_):
        return (t, s_) not in fails

    def cold_succ(t, s_):                   # холодный блок 5: другой исход
        return succ(t, s_) if (t, s_) != (2, 8) else False
    with tempfile.TemporaryDirectory() as td:
        m1 = os.path.join(td, "m1") + os.sep
        os.makedirs(m1)
        diag = os.path.join(td, "diag")
        os.makedirs(diag)
        for t in TASKS:
            _fake_art(m1, "q0", t, 0, "0,5", succ)
            _fake_art(m1, "q0", t, 5, "0,5", succ)
            _fake_art(diag, "q0", t, 5, "5", cold_succ)
            for lab in LABELS:
                _fake_art(m1, lab, t, 0, "0,5", succ, policy="k15f")
        old = dict(F=len(fails), fails=sorted(map(list, fails)), diag=[],
                   blocks={})
        oldp = os.path.join(td, "old.json")
        json.dump(old, open(oldp, "w"))
        for f in ("basis.pt", "ident.json", "brep.json"):
            open(os.path.join(td, f), "w").write(f)
        g.update(COLD_DIR=os.path.join(td, "cold"), DIAG=diag,
                 CENSUS_OLD=oldp, CENSUS_NEW=os.path.join(td, "new.json"),
                 PLAN=os.path.join(td, "plan.json"), m1_dir=lambda: m1)
        try:
            assert census_mode() == 0
            new = json.load(open(g["CENSUS_NEW"]))
            assert [2, 8] in new["m1cold"]["diff"]["fails_added"]
            assert plan_mode(os.path.join(td, "basis.pt"),
                             os.path.join(td, "ident.json"),
                             os.path.join(td, "brep.json")) == 0
            pl = json.load(open(g["PLAN"]))
            assert {j["task"] for j in pl["jobs"]} == {2, 8, 9}
            assert assemble_mode() == 0
            # повторная перепись при зафиксированном плане — отказ
            try:
                census_mode()
                raise AssertionError("перепись перезаписана при плане")
            except SystemExit as e:
                assert "не пересобирается" in str(e)
            # подменённый npz блока 0 руки после плана — отказ сборки
            src = pl["reused_blocks"][0]["source"]
            dd = json.load(open(src))
            import numpy as np
            np.savez(os.path.join(m1, dd["actions_npz"]), x=np.ones(1))
            try:
                assemble_mode()
                raise AssertionError("сборка приняла изменённый npz")
            except SystemExit as e:
                assert "изменились" in str(e)
            # правка переписи после плана — проверка плана отказывает
            cen = json.load(open(g["CENSUS_NEW"]))
            cen["F"] += 1
            json.dump(cen, open(g["CENSUS_NEW"], "w"))
            try:
                verify_plan()
                raise AssertionError("правка переписи прошла проверку")
            except SystemExit as e:
                assert "изменён" in str(e)
        finally:
            g.update(keep)


def main():
    ap = argparse.ArgumentParser(description="K-15f M1-cold")
    ap.add_argument("mode", choices=("census", "plan", "assemble", "jobs",
                                     "verify", "common-args", "selftest"))
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
    if a.mode == "verify":
        verify_plan()
        print("  план M1-cold проверен")
        return 0
    if a.mode == "common-args":
        print(common_args())
        return 0
    return jobs_mode()


if __name__ == "__main__":
    sys.exit(main())
