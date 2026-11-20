#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-37, подтверждение: новый банк native-reset, пространство 2.

Рецепт и код сцены — k33_bank без изменений (generate / select / replay /
save_task / load_task, scene_seed(2, task, idx)), каталог data/k37c/bank.
Задачи 0/6/8, первые 160 допустимых кандидатов по возрастанию idx.
Допуск (k33_bank.select): корректный reset, конечность, нет успеха/done
до управления и во время ожидания, нет дубликата физической сцены — среди
отобранных И среди всех сцен банка K-33 (s0, s1) — и нет совпадения с
официальными init. Отбор завершается до исходов политик.
Холодный повтор (новый объект среды на сцену): post-wait хеши = банку, нет
успеха/done, конечно. Пачки кандидатов — файлы data/k37c/bank_work/*.pkl
(продолжение с места); расход — журнал data/k37c/bank_ledger.json.
Манифест reports/k37c/bank_manifest.json: SHA256 файлов, принятые и
отклонённые idx с причинами, число кандидатов, результат повтора, bank_id.
"""
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

SPACE = 2
TASKS = (0, 6, 8)
N_SCENES = 160
CHUNK = 48
MAX_CAND = 480                       # на задачу; больше — отказ
BANK_DIR = "data/k37c/bank"
WORK = "data/k37c/bank_work"
MANIFEST = "reports/k37c/bank_manifest.json"
LEDGER = "data/k37c/bank_ledger.json"
CAPS = dict(bank=10 * (len(TASKS) * MAX_CAND + len(TASKS) * N_SCENES))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def k33_seen():
    """Подписи и seed всех сцен банка K-33 (пространства 0 и 1)."""
    import k33_bank as kb
    sigs, seeds = set(), set()
    for sp in sorted(kb.SPACES):
        for t in range(kb.N_TASKS):
            z = np.load(kb.task_path(sp, t), allow_pickle=False)
            sigs.update(str(x) for x in z["sig"])
            seeds.update(int(x) for x in z["seed"])
    return sigs, seeds


def run_chunk(L, eid, jobs, fn, workers, wf):
    """reserve -> исполнение -> файл -> commit; готовый файл — settle."""
    import k33_protocol as p33
    key = [list(map(str, j[:3])) if len(j) == 3 else
           [str(j[0]), str(j[1]), j[2]["case_id"]] for j in jobs]
    if os.path.exists(wf):
        d = pickle.load(open(wf, "rb"))
        if d.get("eid") != eid or d.get("jobs") != key:
            raise SystemExit(f"{wf}: чужая пачка")
        if not L.is_done(eid):
            L.settle(eid, d["steps"])
        return d["res"]
    if L.is_done(eid):
        raise SystemExit(f"{eid}: учтён, но результата нет")
    import k33_bank as kb
    L.reserve("bank", eid, amount=kb.WAIT * len(jobs))
    with p33.pool(workers) as p:
        res = p.map(fn, jobs)
    steps = int(sum(r.get("steps", kb.WAIT) for r in res))
    os.makedirs(os.path.dirname(wf), exist_ok=True)
    with open(wf + ".tmp", "wb") as fh:
        pickle.dump(dict(eid=eid, jobs=key, res=res, steps=steps), fh)
    os.replace(wf + ".tmp", wf)
    L.commit(eid, steps)
    return res


def build(a):
    import k31_collect as kcol
    import k33_bank as kb
    import k33_protocol as p33
    if os.path.exists(MANIFEST):
        m = json.load(open(MANIFEST, encoding="utf-8"))
        bad = [p for p, s in m["files"].items()
               if not os.path.exists(p) or sha256(p) != s]
        if bad:
            raise SystemExit(f"банк закреплён, файлы изменены: {bad}")
        print(f"  {MANIFEST}: закреплён, файлы совпадают (bank_id "
              f"{m['bank_id'][:12]})")
        return 0
    L = kcol.Ledger(LEDGER, CAPS)
    seen, k33_seeds = k33_seen()
    per, files = {}, {}
    for t in TASKS:
        official = set(kb.official_joint_sigs(t))
        results, start = [], 0
        while True:
            cases, rej = kb.select(results, N_SCENES, set(seen), official)
            if len(cases) >= N_SCENES:
                break
            if start >= MAX_CAND:
                raise SystemExit(f"t{t}: допустимых {len(cases)} < "
                                 f"{N_SCENES} за {start} кандидатов")
            n = min(CHUNK, MAX_CAND - start)
            jobs = [(SPACE, t, i) for i in range(start, start + n)]
            results += run_chunk(L, f"bank/gen/t{t}/c{start}", jobs,
                                 kb._gen_job, a.workers,
                                 os.path.join(WORK, f"gen_t{t}_c{start}.pkl"))
            start += n
        cases, rej = kb.select(results, N_SCENES, seen, official)
        if len({json.dumps(c["meta"], sort_keys=True) for c in cases}) != 1:
            raise SystemExit(f"t{t}: разные модели сцен")
        p = kb.save_task(SPACE, t, cases, cases[0]["meta"], BANK_DIR)
        files[p] = sha256(p)
        per[str(t)] = dict(cases=len(cases), candidates=start,
                           accepted_idx=[int(c["idx"]) for c in cases],
                           rejects=rej,
                           reject_counts={w: sum(1 for r in rej
                                                 if r["why"] == w)
                                          for w in {r["why"] for r in rej}})
        print(f"  t{t}: принято {len(cases)} из {start} кандидатов; "
              f"отклонено {per[str(t)]['reject_counts']}", flush=True)
    seeds = [int(x) for p_ in files for x in np.load(p_)["seed"]]
    if len(set(seeds)) != len(seeds) or set(seeds) & k33_seeds:
        raise SystemExit("коллизия seed (внутри банка или с K-33)")
    # холодный повтор: новый объект среды на сцену
    jobs = []
    for t in TASKS:
        cs, meta = kb.load_task(SPACE, t, BANK_DIR)
        jobs += [(SPACE, t, c, meta) for c in cs]
    res = []
    for k in range(0, len(jobs), CHUNK):
        res += run_chunk(L, f"bank/replay/p{k}", jobs[k:k + CHUNK],
                         kb._replay_job, a.workers,
                         os.path.join(WORK, f"replay_p{k}.pkl"))
    by = {r["case_id"]: r for r in res}
    bad = []
    for _sp, _t, c, _m in jobs:
        r = by[c["case_id"]]
        if not r.get("ok"):
            bad.append((c["case_id"], r.get("why")))
        elif (r["init_hash"], r["init_hash_full"]) != \
                (c["init_hash"], c["init_hash_full"]):
            bad.append((c["case_id"], "post-wait хеши ≠ банку"))
        elif r["done"] or r["success"] or not r["finite"]:
            bad.append((c["case_id"], "успех/done/нечисловое"))
    man = dict(kind="k37_bank", space=SPACE, tasks=list(TASKS),
               n_per_task=N_SCENES, recipe=kb.RECIPE, files=files, per=per,
               replay=dict(n=len(jobs), bad=len(bad), examples=bad[:20]),
               versions=p33.versions(), budget=L.spent("bank"),
               created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    man["bank_id"] = hashlib.sha256(json.dumps(
        dict(recipe=kb.RECIPE, files=files), sort_keys=True,
        default=str).encode()).hexdigest()
    if bad:
        p33.save_json(MANIFEST + ".failed.json", man)
        print(f"  холодный повтор: {len(bad)} расхождений {bad[:3]}")
        return 3
    p33.save_json(MANIFEST, man)
    print(f"  банк space 2: {sum(v['cases'] for v in per.values())} сцен, "
          f"холодный повтор {len(jobs)}/{len(jobs)} совпали; bank_id "
          f"{man['bank_id'][:12]}; шагов {L.spent('bank')}")
    return 0


def load_manifest():
    if not os.path.exists(MANIFEST):
        raise SystemExit("нет банка подтверждения: k37_confirm.py bank")
    m = json.load(open(MANIFEST, encoding="utf-8"))
    bad = [p for p, s in m["files"].items()
           if not os.path.exists(p) or sha256(p) != s]
    if bad or m.get("replay", {}).get("bad") != 0:
        raise SystemExit(f"банк подтверждения: файлы изменены {bad} или "
                         f"повтор не пройден")
    return m


def scenes(m=None):
    """-> [dict(task, case_id, pos, hash=[ih, ihf], idx)] по задачам, в
    порядке позиции в файле."""
    import k33_bank as kb
    m = m or load_manifest()
    out = []
    for t in TASKS:
        cs, _meta = kb.load_task(SPACE, t, BANK_DIR)
        if len(cs) != N_SCENES:
            raise SystemExit(f"t{t}: сцен {len(cs)} ≠ {N_SCENES}")
        for pos, c in enumerate(cs):
            out.append(dict(task=t, case_id=c["case_id"], pos=pos,
                            idx=int(c["idx"]),
                            hash=[c["init_hash"], c["init_hash_full"]]))
    return out


def selftest():
    import k33_bank as kb
    assert kb.case_id(SPACE, 6, 12) == "k33s2_t6_c00012"
    s = {kb.scene_seed(SPACE, t, i) for t in TASKS for i in range(200)}
    s01 = {kb.scene_seed(sp, t, i) for sp in (0, 1) for t in TASKS
           for i in range(400)}
    assert len(s) == 3 * 200 and not s & s01
    assert CAPS["bank"] == 10 * (3 * 480 + 3 * 160)
    print("самопроверка k37_bank пройдена: case_id k33s2_*, seed space 2 "
          "уникальны и не пересекаются с s0/s1 на первых индексах; лимит "
          "расхода")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; запуск — через k37_confirm.py bank")
