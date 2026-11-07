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
# §14, поправка агента 2026-11-08: банк v1 (генерация без set_init_state(S0))
# архивирован, его расход сохранён в scope bank, закрытом для исполнений;
# исправленная генерация и два воспроизведения — scope bank_v2.
CAPS_V1 = dict(bank=100_000, checks=350_000, sr=10_980_000)
CAPS = dict(CAPS_V1, bank_v2=100_000)
BANK_SCOPE = "bank_v2"
WORK = "data/k33/bank_v2_work"
WORK_V1 = "data/k33/bank_work"
CHECK = "reports/k33/bank_check.json"
ARCH_V1 = "data/k33/archive_v1"
COMPARE_V1 = "reports/k33/bank_v1_v2_compare.json"
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


BANK_CODE = ("k33_bank.py", "k18_collector.py")


def bank_contract():
    """Контракт банка: код sampler/restore/отбора (только k33_bank и формула
    хешей), рецепт, размеры пространств, версии среды, BDDL/init — без кода
    моделей и анализа."""
    c = dict(code={f: sha(os.path.join(HERE, f)) for f in BANK_CODE},
             recipe=kb.RECIPE, spaces={str(k): v for k, v in kb.SPACES.items()},
             versions=versions(), tasks=task_files())
    c = json.loads(json.dumps(c, default=str))
    return c, hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()


def pool(workers):
    import multiprocessing as mp
    return mp.get_context("forkserver").Pool(int(workers))


def jobs_key(jobs):
    """Упорядоченный ключ заданий пачки (seed/индексы/case_id)."""
    out = []
    for j in jobs:
        if isinstance(j, tuple) and len(j) == 3 and not isinstance(j[2], dict):
            sp, t, i = j
            out.append([int(sp), int(t), int(i), kb.scene_seed(sp, t, i)])
        elif isinstance(j, tuple) and len(j) == 4:
            out.append([int(j[0]), int(j[1]), j[2]["case_id"], int(j[2]["seed"])])
        else:
            out.append(json.loads(json.dumps(j, default=str)))
    return out


def scope_problems(L, eid, scope):
    """Пачка банка исполняется только в scope bank_v2 и в его namespace;
    запись журнала по этому eid (done или pending) — того же scope."""
    p = []
    if scope != BANK_SCOPE:
        p.append(f"scope {scope} закрыт для исполнений (только {BANK_SCOPE})")
    if not str(eid).startswith(BANK_SCOPE + "/"):
        p.append(f"eid {eid} вне namespace {BANK_SCOPE}/")
    done = L.d["done"].get(eid)
    if done is not None and done.get("scope") != scope:
        p.append(f"{eid}: учтён в scope {done.get('scope')}")
    pd = L.d["pending"]
    if pd and pd.get("id") == eid and pd.get("scope") != scope:
        p.append(f"{eid}: открытый резерв в scope {pd.get('scope')}")
    return p


def run_chunk(L, eid, jobs, fn, workers, work_file, cfp="test",
              scope=BANK_SCOPE):
    """reserve -> исполнение -> файл -> commit; готовый файл — settle.
    Файл пачки закрепляет eid, scope, отпечаток контракта и упорядоченный
    список заданий; всё (и scope записи журнала) сверяется ДО settle."""
    key = jobs_key(jobs)
    pr = scope_problems(L, eid, scope)
    if pr:
        raise SystemExit("; ".join(pr))
    if os.path.exists(work_file):
        d = pickle.load(open(work_file, "rb"))
        bad = [k for k, v in (("eid", eid), ("scope", scope),
                              ("contract", cfp), ("jobs", key))
               if d.get(k) != v]
        if bad:
            raise SystemExit(f"{work_file}: чужая пачка ({bad})")
        L.settle(eid, d["steps"])
        return d["res"]
    if L.is_done(eid):
        raise SystemExit(f"{eid}: учтён, но результата нет")
    L.reserve(scope, eid, amount=kb.WAIT * len(jobs))
    with pool(workers) as p:
        res = p.map(fn, jobs)
    steps = int(sum(r.get("steps", kb.WAIT) for r in res))
    os.makedirs(os.path.dirname(work_file), exist_ok=True)
    with open(work_file + ".tmp", "wb") as fh:
        pickle.dump(dict(eid=eid, scope=scope, contract=cfp, jobs=key,
                         res=res, steps=steps), fh)
    os.replace(work_file + ".tmp", work_file)
    L.commit(eid, steps)
    return res


def leftover_problems(cpath):
    """Новое построение (нет contract.json v2) при непустых каталогах
    прежнего банка — отказ; продолжение построения v2 — допустимо."""
    left = [p_ for p_ in (kb.BANK_DIR, WORK_V1) if os.path.isdir(p_) and
            os.listdir(p_)]
    if not os.path.exists(cpath) and left:
        return f"остатки прежнего банка {left} — сначала archive-v1"
    return None


def bank(a):
    contract, cfp = bank_contract()
    if os.path.exists(kb.MANIFEST):
        m = json.load(open(kb.MANIFEST, encoding="utf-8"))
        bad = [p for p, s in m["files"].items()
               if not os.path.exists(p) or sha(p) != s]
        if bad or m.get("contract_fp") != cfp:
            diff = [k for k in contract if m.get("contract", {}).get(k) !=
                    contract[k]]
            raise SystemExit(f"манифест банка закреплён; файлы изменены "
                             f"{bad[:3]} или контракт другой {diff}")
        print(f"  {kb.MANIFEST}: закреплён, файлы и контракт совпадают "
              f"(bank_id {m['bank_id'][:12]})")
        return 0
    cpath = os.path.join(WORK, "contract.json")
    pr = leftover_problems(cpath)
    if pr:
        raise SystemExit(pr)
    if os.path.exists(cpath):
        if json.load(open(cpath, encoding="utf-8")).get("fp") != cfp:
            raise SystemExit(f"{cpath}: незавершённое построение другого "
                             f"контракта — нужен новый каталог")
    else:
        save_json(cpath, dict(fp=cfp, contract=contract))
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
                eid = f"{BANK_SCOPE}/gen/s{space}/t{t}/c{start}"
                wf = os.path.join(WORK, f"gen_s{space}_t{t}_c{start}.pkl")
                results += run_chunk(L, eid, jobs, kb._gen_job, a.workers, wf,
                                     cfp)
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
    seeds_all = [int(x) for p_ in files for x in np.load(p_)["seed"]]
    if len(set(seeds_all)) != len(seeds_all):
        raise SystemExit("коллизия seed среди выбранных случаев")
    man = dict(recipe=kb.RECIPE, spaces=kb.SPACES, versions=versions(),
               tasks=tf, files=files, per=per, contract=contract,
               contract_fp=cfp, n_selected=len(seeds_all),
               created=time.strftime("%Y-%m-%dT%H:%M:%S"),
               budget={BANK_SCOPE: L.spent(BANK_SCOPE),
                       "bank_v1": L.spent("bank")})
    man["bank_id"] = hashlib.sha1(json.dumps(
        dict(recipe=man["recipe"], files=files, tasks=tf),
        sort_keys=True).encode()).hexdigest()
    save_json(kb.MANIFEST, man)
    print(f"  банк: {kb.MANIFEST}, bank_id {man['bank_id'][:12]}, шагов "
          f"{BANK_SCOPE} {L.spent(BANK_SCOPE)}/{CAPS[BANK_SCOPE]}")
    return 0


def bank_check(a):
    m = json.load(open(kb.MANIFEST, encoding="utf-8"))
    bad = [p for p, s in m["files"].items() if sha(p) != s]
    _c, cfp = bank_contract()
    if bad or m.get("contract_fp") != cfp:
        raise SystemExit(f"файлы банка изменены {bad[:3]} или контракт другой")
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
            eid = f"{BANK_SCOPE}/replay/{rname}/{m['bank_id'][:12]}/p{k}"
            wf = os.path.join(WORK, f"check_{rname}_p{k}.pkl")
            res += run_chunk(L, eid, part, kb._replay_job, a.workers, wf, cfp)
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
    rep["budget"] = {BANK_SCOPE: L.spent(BANK_SCOPE),
                     "bank_v1": L.spent("bank")}
    save_json(CHECK, rep)
    print("  " + ("bank-check пройден" if rep["ok"] else
                  "bank-check НЕ пройден: " + "; ".join(prob[:3])))
    return 0 if rep["ok"] else 3


# --- поправка §14: миграция журнала, архив банка v1, сверка v1/v2 ----------------
def _v1_work_file(eid):
    """Файл результата пачки банка v1 по её eid (для settle при миграции)."""
    parts = str(eid).split("/")
    if len(parts) == 5 and parts[:2] == ["bank", "gen"]:
        return os.path.join(WORK_V1, f"gen_{parts[2]}_{parts[3]}_"
                            f"{parts[4]}.pkl")
    if len(parts) == 4 and parts[0] == "bank" and parts[1] in ("r1", "r2"):
        return os.path.join(WORK_V1, f"check_{parts[1]}_{parts[3]}.pkl")
    return None


def migrate_ledger_file(path, backup_dir, work_of=_v1_work_file):
    """Однократно: лимиты CAPS_V1 -> CAPS (только новый bank_v2, расход 0).
    Открытый резерв сначала урегулируется штатно: записанный результат
    пачки v1 — settle, иначе — списание как авария (как при следующем
    reserve). Резервная копия журнала и её SHA — до изменения. Повтор
    безопасен: журнал уже с CAPS — ничего не меняется."""
    import shutil
    import k31_collect as kcol
    raw = json.load(open(path, encoding="utf-8"))
    want = {k: int(v) for k, v in CAPS.items()}
    if raw["caps"] == want:
        if "bank_v2" not in raw["spent"]:
            raise SystemExit("лимиты v2 есть, но нет spent.bank_v2")
        return dict(status="already", migrations=raw.get("migrations", []))
    if raw["caps"] != {k: int(v) for k, v in CAPS_V1.items()}:
        raise SystemExit(f"{path}: неожиданные лимиты {raw['caps']}")
    settled = None
    if raw.get("pending"):
        pd = dict(raw["pending"])
        L1 = kcol.Ledger(path, CAPS_V1)
        wf = work_of(pd["id"])
        if wf and os.path.exists(wf):
            d = pickle.load(open(wf, "rb"))
            if d.get("eid") != pd["id"]:
                raise SystemExit(f"{wf}: чужая пачка для {pd['id']}")
            L1.settle(pd["id"], d["steps"])
            settled = dict(pending=pd, how="settle", steps=int(d["steps"]))
        else:
            L1._crash_pending()
            settled = dict(pending=pd, how="crash")
        raw = json.load(open(path, encoding="utf-8"))
    os.makedirs(backup_dir, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    bak = os.path.join(backup_dir, f"ledger_pre_bank_v2_{ts}.json")
    shutil.copy2(path, bak)
    bak_sha = sha(bak)
    if bak_sha != sha(path):
        raise SystemExit("резервная копия журнала ≠ журналу")
    before = dict(caps=dict(raw["caps"]), spent=dict(raw["spent"]),
                  n_done=len(raw["done"]), n_crashes=len(raw["crashes"]))
    raw["caps"] = want
    raw["spent"]["bank_v2"] = 0
    raw.setdefault("migrations", []).append(dict(
        kind="bank_v2", time=time.strftime("%Y-%m-%dT%H:%M:%S"),
        backup=bak, backup_sha1=bak_sha, before=before, pending=settled,
        reason="§14: банк v1 без set_init_state(S0) архивирован; bank закрыт"))
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(raw, fh, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    L = kcol.Ledger(path, CAPS)                 # штатное чтение с новыми лимитами
    for k, v in before["spent"].items():
        assert L.spent(k) == v, k
    assert len(L.d["done"]) == before["n_done"] and \
        len(L.d["crashes"]) == before["n_crashes"] and not L.d["pending"]
    return dict(status="migrated", backup=bak, backup_sha1=bak_sha,
                before=before, pending=settled)


def migrate_ledger(a):
    r = migrate_ledger_file(LEDGER, "data/k33/ledger_backups")
    if r["status"] == "already":
        print("  миграция журнала: уже выполнена (лимиты v2), без изменений")
    else:
        print(f"  миграция журнала: добавлен bank_v2 = {CAPS['bank_v2']}; "
              f"прежний расход {r['before']['spent']}; копия {r['backup']} "
              f"(sha1 {r['backup_sha1'][:12]}); открытый резерв: "
              f"{r['pending'] or 'нет'}")
    return 0


def _tree_sha(paths):
    out = {}
    for p_ in paths:
        if os.path.isdir(p_):
            for root, _d, fs in os.walk(p_):
                for f in sorted(fs):
                    q = os.path.join(root, f)
                    out[q] = sha(q)
        elif os.path.exists(p_):
            out[p_] = sha(p_)
    return out


def archive_integrity(A):
    """Архив v1: каждый файл на месте назначения с SHA до переноса; снимок
    журнала не изменён. -> проблемы."""
    moved = {}
    for path_ in A["sha_before"]:
        for it in A["items"]:
            if path_ == it["src"] or path_.startswith(it["src"].rstrip("/")
                                                     + "/"):
                moved[path_] = it["dst"] + path_[len(it["src"]):]
    pr = [f"{q}: нет или SHA ≠" for p_, q in moved.items()
          if not os.path.exists(q) or sha(q) != A["sha_before"][p_]]
    if len(moved) != len(A["sha_before"]):
        pr.append("не все файлы сопоставлены местам архива")
    if not os.path.exists(A["ledger_snapshot"]) or \
            sha(A["ledger_snapshot"]) != A["ledger_snapshot_sha1"]:
        pr.append("снимок журнала изменён")
    return pr, len(moved)


def archive_v1(a):
    """Перенос (не удаление) банка v1: файлы банка, bank_work, манифест,
    отчёт bank-check; снимок журнала. Сверка SHA до и после переноса.
    Повтор безопасен: завершённый архив только сверяется (файлы банка v2
    на прежних путях при этом не трогаются); незавершённый — доносится."""
    import shutil
    items = [(kb.BANK_DIR, os.path.join(ARCH_V1, "bank")),
             (WORK_V1, os.path.join(ARCH_V1, "bank_work")),
             (kb.MANIFEST, os.path.join(ARCH_V1, "bank_manifest.json")),
             (CHECK, os.path.join(ARCH_V1, "bank_check.json"))]
    af = os.path.join(ARCH_V1, "archive.json")
    if os.path.exists(af):
        A = json.load(open(af, encoding="utf-8"))
        if A.get("done"):
            pr, n = archive_integrity(A)
            if pr:
                raise SystemExit("архив v1 повреждён: " + "; ".join(pr[:3]))
            print(f"  архив v1 уже завершён: bank_id {A['bank_id'][:12]}, "
                  f"файлов {n}, SHA совпадают; рабочие файлы v2 не тронуты")
            return 0
    else:
        if not os.path.exists(kb.MANIFEST):
            raise SystemExit("нет манифеста банка v1 — архивировать нечего")
        m = json.load(open(kb.MANIFEST, encoding="utf-8"))
        _c, cfp = bank_contract()
        if m.get("contract_fp") == cfp:
            raise SystemExit("манифест — банк ТЕКУЩЕГО контракта; v1 не "
                             "архивируется этим режимом")
        os.makedirs(ARCH_V1, exist_ok=True)
        snap = os.path.join(ARCH_V1, "ledger_snapshot.json")
        shutil.copy2(LEDGER, snap)
        src = [s_ for s_, _d in items]
        A = dict(bank_id=m["bank_id"], contract_fp=m.get("contract_fp"),
                 items=[dict(src=s_, dst=d_) for s_, d_ in items],
                 sha_before=_tree_sha(src), ledger_snapshot=snap,
                 ledger_snapshot_sha1=sha(snap), done=False,
                 created=time.strftime("%Y-%m-%dT%H:%M:%S"))
        save_json(af, A)
    for it in A["items"]:
        s_, d_ = it["src"], it["dst"]
        if os.path.exists(s_) and os.path.exists(d_):
            raise SystemExit(f"{s_} и {d_} существуют одновременно")
        if os.path.exists(s_):
            os.makedirs(os.path.dirname(d_), exist_ok=True)
            shutil.move(s_, d_)
    pr, n = archive_integrity(A)
    if pr:
        raise SystemExit("архив v1: " + "; ".join(pr[:3]))
    A["done"], A["n_files"] = True, n
    save_json(af, A)
    print(f"  архив v1: bank_id {A['bank_id'][:12]}, файлов {n} "
          f"перенесено в {ARCH_V1} (SHA совпадают), снимок журнала "
          f"{A['ledger_snapshot_sha1'][:12]}")
    return 0


def v1_bank_source(cfp):
    """Банк v1 для post-fix: завершённый архив, иначе действующий банк
    прежнего контракта. -> (bank_id, каталог банка)."""
    af = os.path.join(ARCH_V1, "archive.json")
    if os.path.exists(af):
        A = json.load(open(af, encoding="utf-8"))
        if not A.get("done"):
            raise SystemExit("архив v1 не завершён")
        pr, _n = archive_integrity(A)
        if pr:
            raise SystemExit("архив v1 повреждён: " + "; ".join(pr[:3]))
        return A["bank_id"], os.path.join(ARCH_V1, "bank")
    if not os.path.exists(kb.MANIFEST):
        raise SystemExit("нет банка v1 (ни архива, ни манифеста)")
    m = json.load(open(kb.MANIFEST, encoding="utf-8"))
    if m.get("contract_fp") == cfp:
        raise SystemExit("действующий манифест — уже банк текущего контракта")
    bad = [p_ for p_, h in m["files"].items()
           if not os.path.exists(p_) or sha(p_) != h]
    if bad:
        raise SystemExit(f"файлы банка v1 изменены {bad[:3]}")
    return m["bank_id"], kb.BANK_DIR


CMP_ARRAYS = ("S0", "body_pos", "body_quat")


def compare_v1_problems(old_dir, new_dir):
    """Все случаи: одинаковые ID, idx, seed, S0, позы тел и подписи; модели
    задач одинаковые. -> (проблемы, сводка)."""
    pr, n, hdiff = [], 0, 0
    for space in sorted(kb.SPACES):
        for t in range(kb.N_TASKS):
            co, mo = kb.load_task(space, t, old_dir)
            cn, mn = kb.load_task(space, t, new_dir)
            if mo != mn:
                pr.append(f"s{space} t{t}: модель/имена ≠")
            if [c["case_id"] for c in co] != [c["case_id"] for c in cn]:
                pr.append(f"s{space} t{t}: наборы ID ≠ "
                          f"({len(co)} против {len(cn)})")
                continue
            for a_, b_ in zip(co, cn):
                n += 1
                why = [k for k in ("idx", "seed", "sig") if a_[k] != b_[k]]
                why += [k for k in CMP_ARRAYS
                        if not np.array_equal(a_[k], b_[k])]
                if why:
                    pr.append(f"{a_['case_id']}: {','.join(why)} ≠")
                if a_["init_hash"] != b_["init_hash"] or \
                        a_["init_hash_full"] != b_["init_hash_full"]:
                    hdiff += 1
    return pr, dict(n_cases=n, post_wait_hash_changed=hdiff)


def compare_v1(a):
    old = os.path.join(ARCH_V1, "bank")
    if not os.path.isdir(old) or not os.path.exists(kb.MANIFEST):
        raise SystemExit("нет архива v1 или манифеста нового банка")
    pr, summ = compare_v1_problems(old, kb.BANK_DIR)
    want = sum(kb.SPACES.values()) * kb.N_TASKS
    if summ["n_cases"] != want:
        pr.append(f"сверено {summ['n_cases']} случаев, нужно {want}")
    m = json.load(open(kb.MANIFEST, encoding="utf-8"))
    rep = dict(bank_id_new=m["bank_id"], ok=not pr, problems=pr[:50],
               n_problems=len(pr), **summ)
    save_json(COMPARE_V1, rep)
    print(f"  v1↔v2: случаев {summ['n_cases']}, ID/seed/S0/подписи/позы "
          + ("совпадают" if not pr else f"РАСХОЖДЕНИЕ ({len(pr)}): "
             + "; ".join(pr[:3]))
          + f"; post-wait хеши изменились у {summ['post_wait_hash_changed']}")
    return 0 if not pr else 3


EP_KEYS = ("success", "success_step", "done_step", "own_steps", "init_hash",
           "init_hash_full", "action_sha1", "env_steps", "policy_calls",
           "init_state_id", "rollout_seed")
LEGACY_SET = ((0, 30), (8, 30))
DEV_SET = ((0, 0), (8, 0))


def block_file(d, task, start):
    return os.path.join(d, f"fullbar_t{task}_i{start}.json")


def harness_block_problems(d, task, start):
    """Один блок K-9h: настройки, 5 эпизодов start…start+4 по порядку,
    обязательные поля, действия (форма, конечность) и пересчёт action SHA."""
    import k15g_local_probe as pr
    f = block_file(d, task, start)
    if not os.path.exists(f):
        return [f"нет блока {os.path.basename(f)}"], None, None
    j = json.load(open(f, encoding="utf-8"))
    p = []
    argv = j.get("argv") or {}
    for k, v in (("seed", 101), ("task_id", task), ("init_start", start)):
        if j.get(k) != v:
            p.append(f"{os.path.basename(f)}: {k} = {j.get(k)}")
    for k, v in (("rollout_seed_mode", "fixed"), ("n_envs", 5)):
        if argv.get(k) != v:
            p.append(f"{os.path.basename(f)}: argv {k} = {argv.get(k)}")
    eps = j.get("episodes") or []
    if [e.get("init_state_id") for e in eps] != list(range(start, start + 5)):
        p.append(f"{os.path.basename(f)}: эпизоды "
                 f"{[e.get('init_state_id') for e in eps]}")
    miss = [k for e in eps for k in EP_KEYS if k not in e]
    if miss:
        p.append(f"{os.path.basename(f)}: нет полей {sorted(set(miss))}")
    if any(e.get("rollout_seed") != 101 for e in eps):
        p.append(f"{os.path.basename(f)}: rollout_seed ≠ 101")
    npz = f[:-5] + ".actions.npz"
    A = np.load(npz)["actions"] if os.path.exists(npz) else None
    if A is None or A.ndim != 3 or A.shape[1:] != (5, 7) or \
            not np.isfinite(A).all():
        p.append(f"{os.path.basename(f)}: действия отсутствуют/форма/"
                 f"конечность")
    elif not p:
        for i, e in enumerate(eps):
            h, _end = pr.episode_action_sha(A, i, e["done_step"])
            if h != e["action_sha1"]:
                p.append(f"{os.path.basename(f)} эпизод {i}: action SHA ≠ JSON")
    return p, j, A


def legacy_problems(cur, new):
    prob = []
    for t, b in LEGACY_SET:
        pc_, jc, Ac = harness_block_problems(cur, t, b)
        pn_, jn, An = harness_block_problems(new, t, b)
        prob += [f"старый путь: {x}" for x in pc_] + \
            [f"адаптер: {x}" for x in pn_]
        if pc_ or pn_:
            continue
        for i, (ec, en) in enumerate(zip(jc["episodes"], jn["episodes"])):
            for grp in (("init_hash", "init_hash_full"),
                        ("action_sha1", "env_steps", "policy_calls"),
                        ("success", "success_step", "done_step",
                         "own_steps")):
                diff = [k for k in grp if ec[k] != en[k]]
                if diff:
                    prob.append(f"t{t} эпизод {i}: {diff}")
                    break
        if not np.array_equal(Ac, An):
            prob.append(f"t{t}: исполненные действия не совпали побитово")
    return prob


def legacy_compare(a):
    """Старый холодный путь против адаптера legacy (сцена -> действия ->
    исходы) на точном наборе блоков t0/t8 i30."""
    prob = legacy_problems(a.cur, a.new)
    save_json(os.path.join("reports/k33", "legacy_compare.json"),
              dict(ok=not prob, problems=prob, set=LEGACY_SET))
    print("  legacy-smoke: " + ("совпадение побитово (хеши сцен, действия, "
                                "исходы; 2 блока × 5)" if not prob else
                                "РАСХОЖДЕНИЕ: " + "; ".join(prob[:4])))
    return 0 if not prob else 3


def dev_problems(new, space, load=None):
    load = load or kb.load_task
    prob = []
    for t, b in DEV_SET:
        p_, j, _A = harness_block_problems(new, t, b)
        prob += p_
        if p_:
            continue
        cases, _m = load(int(space), t)
        for e in j["episodes"]:
            c = cases[int(e["init_state_id"])]
            if e["init_hash"] != c["init_hash"] or \
                    e["init_hash_full"] != c["init_hash_full"]:
                prob.append(f"t{t} {c['case_id']}: post-wait хеш ≠ банку")
    return prob


def dev_compare(a):
    """Харнесс в режиме bank: точный набор t0/t8 блок 0, хеши = банку."""
    prob = dev_problems(a.new, a.space)
    print("  dev-replay: " + ("10 эпизодов, хеши = банку" if not prob
                              else "; ".join(prob[:4])))
    return 0 if not prob else 3


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


def strict_selftest():
    import tempfile
    import k15g_local_probe as pr

    def write(d, task, start, n=5, ids=None, mutate=None):
        os.makedirs(d, exist_ok=True)
        rng = np.random.default_rng(task * 100 + start)
        A = rng.normal(size=(30, 5, 7)).astype(np.float32)
        ds = [10, -1, 5, 29, 3]
        eps = []
        for i in range(n):
            h, _e = pr.episode_action_sha(A, i, ds[i])
            eps.append(dict(success=bool(i % 2), success_step=-1,
                            done_step=ds[i], own_steps=ds[i] + 1,
                            init_hash=f"h{task}_{start + i}",
                            init_hash_full=f"f{task}_{start + i}",
                            action_sha1=h, env_steps=30, policy_calls=4,
                            init_state_id=(ids or list(range(start,
                                                             start + 5)))[i],
                            rollout_seed=101))
        j = dict(seed=101, task_id=task, init_start=start,
                 argv=dict(rollout_seed_mode="fixed", n_envs=5), episodes=eps)
        if mutate:
            mutate(j)
        f = block_file(d, task, start)
        json.dump(j, open(f, "w"))
        np.savez(f[:-5] + ".actions.npz", actions=A)
    with tempfile.TemporaryDirectory() as td:
        cur, new = os.path.join(td, "c"), os.path.join(td, "n")
        for t, b in LEGACY_SET:
            write(cur, t, b)
            write(new, t, b)
        assert legacy_problems(cur, new) == []
        # пустые эпизоды в обоих — отказ
        for t, b in LEGACY_SET:
            write(cur, t, b, mutate=lambda j: j.__setitem__("episodes", []))
            write(new, t, b, mutate=lambda j: j.__setitem__("episodes", []))
        assert legacy_problems(cur, new)
        # 5 против 4 — отказ
        for t, b in LEGACY_SET:
            write(cur, t, b)
            write(new, t, b, n=4)
        assert legacy_problems(cur, new)
        # нет блока t8 в обоих — отказ
        for t, b in LEGACY_SET:
            write(new, t, b)
        os.remove(block_file(cur, 8, 30))
        os.remove(block_file(new, 8, 30))
        assert legacy_problems(cur, new)
        write(cur, 8, 30)
        write(new, 8, 30)
        # неверный режим seed / порядок id / подмена action SHA — отказ
        write(new, 0, 30, mutate=lambda j: j["argv"].__setitem__(
            "rollout_seed_mode", "block"))
        assert legacy_problems(cur, new)
        write(new, 0, 30, ids=[31, 30, 32, 33, 34])
        assert legacy_problems(cur, new)
        write(new, 0, 30, mutate=lambda j: j["episodes"][2].__setitem__(
            "action_sha1", "0" * 16))
        assert legacy_problems(cur, new)
        write(new, 0, 30)
        assert legacy_problems(cur, new) == []
        # dev: точный набор t0/t8 блок 0; один эпизод одной задачи — отказ
        dev = os.path.join(td, "d")

        def load(space, t):
            return [dict(case_id=f"c{t}_{i}", init_hash=f"h{t}_{i}",
                         init_hash_full=f"f{t}_{i}") for i in range(5)], {}
        for t, b in DEV_SET:
            write(dev, t, b)
        assert dev_problems(dev, 0, load) == []
        os.remove(block_file(dev, 8, 0))
        write(dev, 0, 0, n=1)
        assert dev_problems(dev, 0, load)
        write(dev, 0, 0)
        write(dev, 8, 0, mutate=lambda j: j["episodes"][1].__setitem__(
            "init_hash", "x"))
        assert dev_problems(dev, 0, load)


def migration_selftest():
    import tempfile
    import k31_collect as kcol
    with tempfile.TemporaryDirectory() as td:
        lp = os.path.join(td, "l.json")
        L = kcol.Ledger(lp, CAPS_V1)
        L.reserve("bank", "bank/gen/s0/t0/c0", amount=50)
        L.commit("bank/gen/s0/t0/c0", 50)
        L.reserve("checks", "checks/a", amount=30)
        L.commit("checks/a", 30)
        L.reserve("checks", "checks/b", amount=7)        # открытый резерв
        L._crash_pending()                                # прошлая авария
        L.reserve("bank", "bank/r1/abc/p0", amount=20)   # открытый, с файлом
        wv1 = os.path.join(td, "v1")
        os.makedirs(wv1)
        with open(os.path.join(wv1, "check_r1_p0.pkl"), "wb") as fh:
            pickle.dump(dict(eid="bank/r1/abc/p0", steps=20), fh)

        def work_of(eid):
            f = _v1_work_file(eid)
            return f and os.path.join(wv1, os.path.basename(f))
        r = migrate_ledger_file(lp, os.path.join(td, "b"), work_of)
        assert r["status"] == "migrated" and r["pending"]["how"] == "settle"
        L2 = kcol.Ledger(lp, CAPS)
        assert L2.spent("bank") == 70 and L2.spent("checks") == 37 and \
            L2.spent("bank_v2") == 0 and len(L2.d["crashes"]) == 1 and \
            L2.is_done("bank/r1/abc/p0") and not L2.d["pending"]
        bak = json.load(open(r["backup"], encoding="utf-8"))
        assert bak["caps"] == CAPS_V1 and sha(r["backup"]) == r["backup_sha1"]
        h = sha(lp)
        assert migrate_ledger_file(lp, os.path.join(td, "b"))["status"] == \
            "already" and sha(lp) == h
        # открытый резерв без файла результата — авария
        lp2 = os.path.join(td, "l2.json")
        L = kcol.Ledger(lp2, CAPS_V1)
        L.reserve("bank", "bank/gen/s1/t2/c8", amount=40)
        r = migrate_ledger_file(lp2, os.path.join(td, "b"), work_of)
        L3 = kcol.Ledger(lp2, CAPS)
        assert r["pending"]["how"] == "crash" and L3.spent("bank") == 40
        # чужие лимиты — отказ
        lp3 = os.path.join(td, "l3.json")
        kcol.Ledger(lp3, dict(CAPS_V1, checks=1))
        try:
            migrate_ledger_file(lp3, os.path.join(td, "b"))
            raise AssertionError("чужие лимиты приняты")
        except SystemExit:
            pass
        # сверка v1/v2 на игрушечных банках
        meta = dict(bodies=["a"], joints=["j"], nq=1, nv=1, fingerprint="f")

        def mk(d, mut=None):
            for space, n in kb.SPACES.items():
                for t in range(kb.N_TASKS):
                    cases = [dict(seed=kb.scene_seed(space, t, i), idx=i,
                                  S0=np.full(4, float(i)),
                                  body_pos=np.zeros((2, 3)),
                                  body_quat=np.tile([1.0, 0, 0, 0], (2, 1)),
                                  sig=f"s{space}{t}{i}", init_hash=f"h{i}",
                                  init_hash_full=f"f{i}")
                             for i in range(min(n, 3))]
                    if mut and (space, t) == (1, 9):
                        mut(cases)
                    kb.save_task(space, t, cases, meta, d)
        o, nw = os.path.join(td, "o"), os.path.join(td, "n")
        mk(o)
        mk(nw, lambda c: c[1].__setitem__("init_hash", "x"))
        pr, sm = compare_v1_problems(o, nw)
        assert pr == [] and sm["post_wait_hash_changed"] == 1
        for mut in (lambda c: c[2].__setitem__("S0", np.full(4, 9.0)),
                    lambda c: c[0].__setitem__("sig", "z"),
                    lambda c: c[1].__setitem__("body_pos", np.ones((2, 3))),
                    lambda c: c.pop()):
            mk(nw, mut)
            assert compare_v1_problems(o, nw)[0], "расхождение не замечено"
        # архив v1: перенос со сверкой SHA, повтор безопасен
        g = globals()
        keep = {k: g[k] for k in ("ARCH_V1", "WORK_V1", "WORK", "CHECK",
                                  "LEDGER", "bank_contract")}
        keep_kb = (kb.BANK_DIR, kb.MANIFEST)
        try:
            g.update(ARCH_V1=os.path.join(td, "arch"),
                     WORK_V1=os.path.join(td, "wk"),
                     WORK=os.path.join(td, "wk2"),
                     CHECK=os.path.join(td, "chk.json"), LEDGER=lp,
                     bank_contract=lambda: ({}, "new"))
            kb.BANK_DIR, kb.MANIFEST = o, os.path.join(td, "man.json")
            os.makedirs(g["WORK_V1"])
            open(os.path.join(g["WORK_V1"], "gen.pkl"), "wb").write(b"x")
            save_json(kb.MANIFEST, dict(bank_id="old", contract_fp="old"))
            save_json(g["CHECK"], dict(ok=False))
            before = _tree_sha([o, g["WORK_V1"], kb.MANIFEST, g["CHECK"]])
            assert archive_v1(None) == 0 and archive_v1(None) == 0
            assert not os.path.exists(o) and not os.path.exists(kb.MANIFEST)
            after = _tree_sha([g["ARCH_V1"]])
            assert sorted(before.values()) == sorted(
                v for k, v in after.items()
                if not k.endswith(("archive.json", "ledger_snapshot.json")))
            assert v1_bank_source("new") == ("old", os.path.join(
                g["ARCH_V1"], "bank"))
            # регрессия: архив -> частичное построение v2 -> повтор цепочки
            cp2 = os.path.join(g["WORK"], "contract.json")
            assert leftover_problems(cp2) is None
            save_json(cp2, dict(fp="new"))
            kb.save_task(0, 0, [dict(seed=1, idx=0, S0=np.zeros(4),
                                     body_pos=np.zeros((2, 3)),
                                     body_quat=np.zeros((2, 4)), sig="v2",
                                     init_hash="a", init_hash_full="b")],
                         meta, o)
            v2f = _tree_sha([o, g["WORK"]])
            assert archive_v1(None) == 0
            assert _tree_sha([o, g["WORK"]]) == v2f, "файлы v2 тронуты"
            assert leftover_problems(cp2) is None     # продолжение v2
            os.remove(cp2)
            assert leftover_problems(cp2), "остатки без contract v2 приняты"
            save_json(cp2, dict(fp="new"))
            # повреждение архива — отказ
            af1 = os.path.join(g["ARCH_V1"], "bank", "s0", "t1.npz")
            with open(af1, "ab") as fh:
                fh.write(b"x")
            try:
                archive_v1(None)
                raise AssertionError("повреждённый архив принят")
            except SystemExit:
                pass
            # банк текущего контракта не архивируется
            g["ARCH_V1"] = os.path.join(td, "arch2")
            save_json(kb.MANIFEST, dict(bank_id="cur", contract_fp="new"))
            try:
                archive_v1(None)
                raise AssertionError("банк текущего контракта архивирован")
            except SystemExit:
                pass
        finally:
            g.update(keep)
            kb.BANK_DIR, kb.MANIFEST = keep_kb


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
            V = BANK_SCOPE + "/"

            def refused(f, *aa, **kw):
                try:
                    f(*aa, **kw)
                except SystemExit:
                    return True
                return False
            wf = os.path.join(td, "w.pkl")
            r1 = run_chunk(L, V + "e1", [1, 2, 3], fn, 1, wf)
            assert L.spent(BANK_SCOPE) == 30 and L.spent("bank") == 0 and \
                calls["n"] == 3
            r2 = run_chunk(kcol.Ledger(os.path.join(td, "l.json"), CAPS),
                           V + "e1", [1, 2, 3], fn, 1, wf)
            assert r2 == r1 and calls["n"] == 3      # без повторного исполнения
            # авария после записи до commit -> settle без повторного расхода
            L2 = kcol.Ledger(os.path.join(td, "l.json"), CAPS)
            L2.reserve(BANK_SCOPE, V + "e2", amount=20)
            with open(os.path.join(td, "w2.pkl"), "wb") as fh:
                pickle.dump(dict(eid=V + "e2", scope=BANK_SCOPE,
                                 contract="test", jobs=jobs_key([1, 2]),
                                 res=[], steps=20), fh)
            run_chunk(kcol.Ledger(os.path.join(td, "l.json"), CAPS), V + "e2",
                      [1, 2], fn, 1, os.path.join(td, "w2.pkl"))
            L3 = kcol.Ledger(os.path.join(td, "l.json"), CAPS)
            assert L3.spent(BANK_SCOPE) == 50 and not L3.d["crashes"]
            assert refused(run_chunk, L3, V + "e9", [1], fn, 1, wf), \
                "чужая пачка принята"
            # тот же eid, другое задание (новый seed) или контракт — отказ
            wg = os.path.join(td, "g.pkl")
            run_chunk(L3, V + "eg", [(1, 0, 5)], fn, 1, wg, "C1")
            n0 = calls["n"]
            for jobs, cf in (([(1, 0, 6)], "C1"), ([(1, 0, 5)], "C2")):
                assert refused(run_chunk, L3, V + "eg", jobs, fn, 1, wg, cf), \
                    "подменённая пачка принята"
            # scope bank закрыт; eid вне namespace; файл пачки без scope v2;
            # eid, учтённый в другом scope, — отказ без исполнения
            assert refused(run_chunk, L3, "bank/gen/x", [1], fn, 1,
                           os.path.join(td, "x.pkl"), scope="bank")
            assert refused(run_chunk, L3, "bank/gen/x", [1], fn, 1,
                           os.path.join(td, "x.pkl"))
            wo = os.path.join(td, "old.pkl")
            with open(wo, "wb") as fh:
                pickle.dump(dict(eid=V + "e3", contract="test",
                                 jobs=jobs_key([1]), res=[], steps=10), fh)
            assert refused(run_chunk, L3, V + "e3", [1], fn, 1, wo)
            L3.d["done"][V + "e4"] = dict(scope="bank", actual=10)
            assert refused(run_chunk, L3, V + "e4", [1], fn, 1,
                           os.path.join(td, "e4.pkl"))
            assert calls["n"] == n0
        finally:
            g["pool"] = orig
    migration_selftest()
    strict_selftest()
    print("самопроверка k33_protocol пройдена: миграция журнала v1 -> "
          "bank_v2 (копия и SHA, расход/done/crashes сохранены, резерв "
          "урегулирован settle или аварией, повтор без изменений, чужие "
          "лимиты — отказ), архив v1 переносом со сверкой SHA и повтором, "
          "сверка v1/v2 ловит ID/seed/S0/подпись/позы; scope bank закрыт, "
          "namespace bank_v2/, scope файла и записи журнала сверяются; "
          "строгие legacy/dev-сверки "
          "(пустые эпизоды, 5 против 4, один эпизод, неверные id/настройки, "
          "action SHA — отказ), пачка закрепляет контракт и задания (новый "
          "seed при том же eid — отказ); пачка банка — резерв, файл до "
          "commit, повтор без исполнения и без двойного расхода, авария "
          "между записью и commit закрывается settle, чужая пачка — отказ")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-33: банк сцен")
    ap.add_argument("cmd", choices=("bank", "bank-check", "legacy-compare",
                                    "dev-compare", "ledger", "migrate-ledger",
                                    "archive-v1", "compare-v1"))
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
                                  "ledger": ledger_cmd,
                                  "migrate-ledger": migrate_ledger,
                                  "archive-v1": archive_v1,
                                  "compare-v1": compare_v1})[a.cmd](a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
