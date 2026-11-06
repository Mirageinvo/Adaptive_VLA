#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-33: построение и проверка банка сцен (без политик).

bank --workers N — development (5/задача), затем confirmation (300/задача):
  кандидаты по seed SeedSequence([33, 20261106, space, task, idx]) в
  порядке idx, генерация в свежих объектах среды (пул forkserver), отбор
  первых допустимых (k33_bank.select): дедупликация по подписи внутри и
  между пространствами, против официальных init — по qpos суставов.
  Учёт шагов (10 на кандидата) — журнал бюджета (scope bank, 100 000):
  резерв на пачку -> исполнение -> атомарный файл результатов -> commit;
  записанный до commit результат закрывается settle.
  Манифест reports/k33/bank_manifest.json неизменяем: повтор только
  сверяет SHA файлов банка.
bank-check --workers N — два независимых холодных воспроизведения всех
  случаев в разном порядке (задачи в обратном порядке; случайная
  перестановка с seed 33012): native reset того же рецепта, сверка
  имён/размерностей/отпечатка модели и runtime-поз тел/фикстур, S0 и
  подписи, затем 10 шагов ожидания и post-wait init_hash / init_hash_full
  = банку; нет успеха/done; конечность. Отчёт reports/k33/bank_check.json.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k33_bank as kb  # noqa: E402

LEDGER = "data/k33/ledger.json"
CAPS = dict(bank=100_000, checks=350_000, sr=10_980_000)
WORK = "data/k33/bank_work"
CHECK = "reports/k33/bank_check.json"
CHUNK_MARGIN = 8
MAX_FACTOR = 3


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def save_json(path, d):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(path + ".tmp", path)


def ledger():
    import k31_collect as kcol
    return kcol.Ledger(LEDGER, CAPS)


def versions():
    out = {}
    for m in ("robosuite", "mujoco", "gym", "libero", "libero2gym", "numpy"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", None)
        except Exception as e:                               # noqa: BLE001
            out[m] = f"нет: {type(e).__name__}"
    try:
        import subprocess
        import libero
        root = os.path.dirname(os.path.dirname(os.path.abspath(
            libero.__file__)))
        out["libero_commit"] = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except Exception as e:                                   # noqa: BLE001
        out["libero_commit"] = f"нет: {type(e).__name__}"
    return out


def task_files():
    from libero.libero import benchmark, get_libero_path
    bm = benchmark.get_benchmark_dict()["libero_10"]()
    out = {}
    for t in range(kb.N_TASKS):
        task = bm.get_task(t)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                            task.bddl_file)
        init = os.path.join(get_libero_path("init_states"),
                            task.problem_folder, task.init_states_file)
        out[str(t)] = dict(language=task.language, bddl=bddl, bddl_sha1=sha(bddl),
                           init=init, init_sha1=sha(init))
    return out


def pool(workers):
    import multiprocessing as mp
    return mp.get_context("forkserver").Pool(int(workers))


def run_chunk(L, eid, jobs, fn, workers, work_file):
    """reserve -> исполнение -> файл -> commit; готовый файл — settle."""
    if os.path.exists(work_file):
        d = pickle.load(open(work_file, "rb"))
        if d["eid"] != eid:
            raise SystemExit(f"{work_file}: чужая пачка")
        L.settle(eid, d["steps"])
        return d["res"]
    if L.is_done(eid):
        raise SystemExit(f"{eid}: учтён, но результата нет")
    L.reserve("bank", eid, amount=kb.WAIT * len(jobs))
    with pool(workers) as p:
        res = p.map(fn, jobs)
    steps = int(sum(r.get("steps", kb.WAIT) for r in res))
    os.makedirs(os.path.dirname(work_file), exist_ok=True)
    with open(work_file + ".tmp", "wb") as fh:
        pickle.dump(dict(eid=eid, res=res, steps=steps), fh)
    os.replace(work_file + ".tmp", work_file)
    L.commit(eid, steps)
    return res


def bank(a):
    if os.path.exists(kb.MANIFEST):
        m = json.load(open(kb.MANIFEST, encoding="utf-8"))
        bad = [p for p, s in m["files"].items()
               if not os.path.exists(p) or sha(p) != s]
        if bad:
            raise SystemExit(f"манифест банка закреплён, файлы изменены: "
                             f"{bad[:3]}")
        print(f"  {kb.MANIFEST}: закреплён, файлы совпадают "
              f"(bank_id {m['bank_id'][:12]})")
        return 0
    L = ledger()
    tf = task_files()
    seen = set()
    per, files = {}, {}
    official = {}
    for t in range(kb.N_TASKS):
        official[t] = set(kb.official_joint_sigs(t))
    for space in sorted(kb.SPACES):
        need = kb.SPACES[space]
        for t in range(kb.N_TASKS):
            results, start = [], 0
            while True:
                cases, rej = kb.select(results, need, set(seen),
                                       official[t])
                if len(cases) >= need:
                    break
                if start >= need * MAX_FACTOR + CHUNK_MARGIN:
                    raise SystemExit(f"s{space} t{t}: допустимых {len(cases)} "
                                     f"< {need} за {start} кандидатов")
                n = need - len(cases) + CHUNK_MARGIN
                jobs = [(space, t, i) for i in range(start, start + n)]
                eid = f"bank/gen/s{space}/t{t}/c{start}"
                wf = os.path.join(WORK, f"gen_s{space}_t{t}_c{start}.pkl")
                results += run_chunk(L, eid, jobs, kb._gen_job, a.workers, wf)
                start += n
            cases, rej = kb.select(results, need, seen, official[t])
            metas = {json.dumps(c["meta"], sort_keys=True) for c in cases}
            if len(metas) != 1:
                raise SystemExit(f"s{space} t{t}: разные модели сцен")
            p = kb.save_task(space, t, cases, cases[0]["meta"])
            files[p] = sha(p)
            per[f"s{space}_t{t}"] = dict(
                cases=len(cases), candidates=start, rejects=rej,
                reject_counts={w: sum(1 for r in rej if r["why"] == w)
                               for w in {r["why"] for r in rej}},
                model_fingerprint=cases[0]["meta"]["fingerprint"])
            print(f"  s{space} t{t}: случаев {len(cases)} из {start} "
                  f"кандидатов; отбраковано "
                  f"{per[f's{space}_t{t}']['reject_counts']}", flush=True)
    man = dict(recipe=kb.RECIPE, spaces=kb.SPACES, versions=versions(),
               tasks=tf, files=files, per=per,
               created=time.strftime("%Y-%m-%dT%H:%M:%S"),
               budget=dict(bank=L.spent("bank")))
    man["bank_id"] = hashlib.sha1(json.dumps(
        dict(recipe=man["recipe"], files=files, tasks=tf),
        sort_keys=True).encode()).hexdigest()
    save_json(kb.MANIFEST, man)
    print(f"  банк: {kb.MANIFEST}, bank_id {man['bank_id'][:12]}, шагов "
          f"{L.spent('bank')}")
    return 0


def bank_check(a):
    m = json.load(open(kb.MANIFEST, encoding="utf-8"))
    bad = [p for p, s in m["files"].items() if sha(p) != s]
    if bad:
        raise SystemExit(f"файлы банка изменены: {bad[:3]}")
    L = ledger()
    rep = dict(bank_id=m["bank_id"], rounds={})
    prob = []
    all_jobs = []
    for space in sorted(kb.SPACES):
        for t in range(kb.N_TASKS):
            cases, meta = kb.load_task(space, t)
            all_jobs += [(space, t, c, meta) for c in cases]
    orders = dict(r1=sorted(all_jobs, key=lambda j: (-j[1], j[0],
                                                     j[2]["idx"])),
                  r2=[all_jobs[i] for i in np.random.default_rng(
                      33012).permutation(len(all_jobs))])
    for rname, jobs in orders.items():
        res = []
        step = 200
        for k in range(0, len(jobs), step):
            part = jobs[k:k + step]
            eid = f"bank/{rname}/{m['bank_id'][:12]}/p{k}"
            wf = os.path.join(WORK, f"check_{rname}_p{k}.pkl")
            res += run_chunk(L, eid, part, kb._replay_job, a.workers, wf)
        by = {r["case_id"]: r for r in res}
        bad_r = []
        for (space, t, c, _meta) in jobs:
            r = by[c["case_id"]]
            why = None
            if not r.get("ok"):
                why = r.get("why")
            elif r["init_hash"] != c["init_hash"] or \
                    r["init_hash_full"] != c["init_hash_full"]:
                why = "post-wait хеши ≠ банку"
            elif r["done"] or r["success"] or not r["finite"]:
                why = "успех/done/нечисловое после восстановления"
            if why:
                bad_r.append(dict(case_id=c["case_id"], why=why))
        rep["rounds"][rname] = dict(n=len(jobs), bad=len(bad_r),
                                    examples=bad_r[:20])
        prob += [f"{rname}: {x['case_id']}: {x['why']}" for x in bad_r[:5]]
        print(f"  воспроизведение {rname}: {len(jobs) - len(bad_r)}/"
              f"{len(jobs)} совпали", flush=True)
    rep["ok"] = all(v["bad"] == 0 for v in rep["rounds"].values())
    rep["problems"] = prob
    rep["budget"] = dict(bank=L.spent("bank"))
    save_json(CHECK, rep)
    print("  " + ("bank-check пройден" if rep["ok"] else
                  "bank-check НЕ пройден: " + "; ".join(prob[:3])))
    return 0 if rep["ok"] else 3


def _blocks(d):
    out = {}
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.endswith(".json"):
            j = json.load(open(os.path.join(d, f), encoding="utf-8"))
            if "episodes" not in j:
                continue
            npz = os.path.join(d, f[:-5] + ".actions.npz")
            A = np.load(npz)["actions"] if os.path.exists(npz) else None
            out[f] = (j, A)
    return out


EP_KEYS = ("success", "success_step", "done_step", "own_steps", "init_hash",
           "init_hash_full", "action_sha1", "env_steps", "policy_calls")


def legacy_compare(a):
    """Старый холодный путь против адаптера legacy: эпизоды и действия."""
    cur, new = _blocks(a.cur), _blocks(a.new)
    prob, rows = [], []
    if not cur or set(cur) != set(new):
        prob.append(f"наборы блоков различаются: {sorted(cur)} / "
                    f"{sorted(new)}")
    for f in sorted(set(cur) & set(new)):
        (jc, Ac), (jn, An) = cur[f], new[f]
        for i, (ec, en) in enumerate(zip(jc["episodes"], jn["episodes"])):
            diff = [k for k in EP_KEYS if ec.get(k) != en.get(k)]
            rows.append(dict(block=f, ep=i, diff=diff))
            if diff:
                prob.append(f"{f} эпизод {i}: {diff}")
        if Ac is None or An is None or not np.array_equal(Ac, An):
            prob.append(f"{f}: исполненные действия не совпали побитово")
    rep = dict(ok=not prob, problems=prob, rows=rows)
    save_json(os.path.join("reports/k33", "legacy_compare.json"), rep)
    print("  legacy-smoke: " + ("совпадение побитово (сцены, хеши, действия, "
                                "исходы)" if not prob else
                                "РАСХОЖДЕНИЕ: " + "; ".join(prob[:4])))
    return 0 if not prob else 3


def dev_compare(a):
    """Харнесс в режиме bank: post-wait хеши эпизодов = банку."""
    prob, n = [], 0
    for f, (j, _A) in _blocks(a.new).items():
        t = int(j["task_id"])
        cases, _m = kb.load_task(int(a.space), t)
        for e in j["episodes"]:
            c = cases[int(e["init_state_id"])]
            n += 1
            if e["init_hash"] != c["init_hash"] or \
                    e["init_hash_full"] != c["init_hash_full"]:
                prob.append(f"{f} {c['case_id']}: post-wait хеш ≠ банку")
    print(f"  dev-replay: эпизодов {n}; " + ("хеши = банку" if not prob
                                            else "; ".join(prob[:4])))
    return 0 if (n and not prob) else 3


def ledger_cmd(a):
    """Резерв/фиксация физических шагов проверочного блока (scope checks).
    commit: расход = n_envs·(шагов + ожидание) из JSON блока."""
    L = ledger()
    if a.op == "reserve":
        L.reserve("checks", a.eid, amount=5 * 610)
        return 0
    j = json.load(open(a.json_path, encoding="utf-8"))
    e0 = j["episodes"][0]
    actual = len(j["episodes"]) * (int(e0["env_steps"]) + 10)
    L.commit(a.eid, actual)
    print(f"    шагов {actual}; checks {L.spent('checks')}/{CAPS['checks']}")
    return 0


def selftest():
    import tempfile
    import k31_collect as kcol
    with tempfile.TemporaryDirectory() as td:
        L = kcol.Ledger(os.path.join(td, "l.json"), CAPS)
        calls = dict(n=0)

        def fn(j):
            calls["n"] += 1
            return dict(steps=kb.WAIT, j=j)
        orig = pool

        class P:
            def __init__(self, w):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def map(self, f, jobs):
                return [f(j) for j in jobs]
        g = globals()
        g["pool"] = P
        try:
            wf = os.path.join(td, "w.pkl")
            r1 = run_chunk(L, "e1", [1, 2, 3], fn, 1, wf)
            assert L.spent("bank") == 30 and calls["n"] == 3
            r2 = run_chunk(kcol.Ledger(os.path.join(td, "l.json"), CAPS),
                           "e1", [1, 2, 3], fn, 1, wf)
            assert r2 == r1 and calls["n"] == 3      # без повторного исполнения
            # авария после записи до commit -> settle без повторного расхода
            L2 = kcol.Ledger(os.path.join(td, "l.json"), CAPS)
            L2.reserve("bank", "e2", amount=20)
            with open(os.path.join(td, "w2.pkl"), "wb") as fh:
                pickle.dump(dict(eid="e2", res=[], steps=20), fh)
            run_chunk(kcol.Ledger(os.path.join(td, "l.json"), CAPS), "e2",
                      [1, 2], fn, 1, os.path.join(td, "w2.pkl"))
            L3 = kcol.Ledger(os.path.join(td, "l.json"), CAPS)
            assert L3.spent("bank") == 50 and not L3.d["crashes"]
            try:
                run_chunk(L3, "e9", [1], fn, 1, wf)
                raise AssertionError("чужая пачка принята")
            except SystemExit:
                pass
        finally:
            g["pool"] = orig
    print("самопроверка k33_protocol пройдена: пачка банка — резерв, файл до "
          "commit, повтор без исполнения и без двойного расхода, авария "
          "между записью и commit закрывается settle, чужая пачка — отказ")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-33: банк сцен")
    ap.add_argument("cmd", choices=("bank", "bank-check", "legacy-compare",
                                    "dev-compare", "ledger"))
    ap.add_argument("--op", choices=("reserve", "commit"))
    ap.add_argument("--eid")
    ap.add_argument("--json-path")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cur")
    ap.add_argument("--new")
    ap.add_argument("--space", type=int, default=0)
    a = ap.parse_args()
    try:
        return dict(bank=bank, **{"bank-check": bank_check,
                                  "legacy-compare": legacy_compare,
                                  "dev-compare": dev_compare,
                                  "ledger": ledger_cmd})[a.cmd](a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
