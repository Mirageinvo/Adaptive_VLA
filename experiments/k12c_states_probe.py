#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-12c: сколько НЕЗАВИСИМЫХ начальных состояний даёт сюита LIBERO.

ЗАЧЕМ. Протокол финальной оценки опирается на «80 эпизодов на задачу». Эпизод
здесь — это НАЧАЛЬНОЕ СОСТОЯНИЕ. Если различных состояний в сюите меньше, то
«лишние эпизоды» — это повторы одного состояния с другим сидом раскатки, а они
независимыми наблюдениями не являются: при нулевой дисперсии политики это
буквально тот же эпизод, при положительной — наблюдения, коррелированные через
общее состояние. Пока число различных состояний не измерено, число эпизодов в
протоколе ничем не обосновано, а доверительный интервал по нему занижен.

ЧТО ПРОВЕРЯЕТСЯ
  1. граница допустимых init_state_id — и не клампится ли слишком большой id
     МОЛЧА к существующему состоянию (тогда отказа не будет, а эпизод окажется
     копией; это опаснее исключения);
  2. сколько среди допустимых id РАЗЛИЧНЫХ состояний (дубликаты — по хэшу);
  3. меняет ли сид раскатки начальное состояние (если нет, новых эпизодов сид
     не создаёт, сколько бы их ни запускать);
  4. совпадают ли хэши с записанными в готовых артефактах K-11e/K-11g — тогда
     отображение id -> состояние стабильно между прогонами, и прежние пары
     действительно парные.

ХЭШ СЧИТАЕТСЯ ТОЙ ЖЕ ФОРМУЛОЙ И ПОСЛЕ ТОГО ЖЕ ЧИСЛА ХОЛОСТЫХ ШАГОВ, ЧТО В
K-11g. Иначе он несравним с записанным, и п.4 не проверял бы ничего. Записи с
другим waiting_steps сравниваются отдельно, а не подмешиваются.

МОДЕЛЬ НЕ НУЖНА: действие — холостое. Ни чекпойнтов, ни GPU.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

PROBE_IDS = [0, 1, 2, 19, 39, 44, 49, 50, 63, 64, 99, 100, 127, 128, 199, 200,
             255, 511, 10000]


def ep_hash(parts):
    """Тот же хэш, что init_hash_full в K-11g, бит в бит."""
    return hashlib.sha1(np.ascontiguousarray(
        np.concatenate(parts).astype(np.float32)).tobytes()).hexdigest()[:16]


def obs_hashes(obs, n):
    """(init_hash, init_hash_full) для каждой среды — формулы K-11g."""
    short = [ep_hash([obs["state"][i].ravel(),
                      obs["agentview_image"][i].ravel() / 255.0])
             for i in range(n)]
    full = [ep_hash([obs["state"][i].ravel(),
                     obs["agentview_image"][i].ravel() / 255.0,
                     obs["robot0_eye_in_hand_image"][i].ravel() / 255.0])
            for i in range(n)]
    return short, full


def dedup(hash_by_id):
    """Группы id с ОДНИМ И ТЕМ ЖЕ состоянием.

    Возвращает n_distinct и сами группы: без них нельзя сказать, дубликат —
    это кламп большого id к нулевому состоянию или два одинаковых состояния
    внутри файла сюиты. Это разные дефекты.
    """
    by_hash = {}
    for i in sorted(hash_by_id):
        by_hash.setdefault(hash_by_id[i], []).append(int(i))
    groups = sorted([g for g in by_hash.values() if len(g) > 1])
    return dict(n_ids=len(hash_by_id), n_distinct=len(by_hash),
                dup_groups=groups,
                n_dup_ids=sum(len(g) - 1 for g in groups))


def walk_records(obj, inherited=None, out=None, src=""):
    """Все записи эпизодов в произвольном артефакте.

    Форматы K-11e и K-11g различаются, и угадывать структуру нельзя: обход
    ищет любой словарь с init_state_id и init_hash_full, а task_id / suite /
    waiting_steps берёт из БЛИЖАЙШЕГО охватывающего словаря, где они есть.
    """
    inherited = dict(inherited or {})
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k in ("task_id", "suite", "task_suite", "waiting_steps"):
            if k in obj and not isinstance(obj[k], (dict, list)):
                inherited["suite" if k == "task_suite" else k] = obj[k]
        if "init_state_id" in obj and "init_hash_full" in obj:
            out.append(dict(src=src,
                            task_id=inherited.get("task_id"),
                            suite=(None if inherited.get("suite") is None
                                   else str(inherited["suite"])),
                            waiting_steps=inherited.get("waiting_steps"),
                            init_state_id=int(obj["init_state_id"]),
                            init_hash_full=str(obj["init_hash_full"]),
                            init_hash=obj.get("init_hash")))
        for v in obj.values():
            walk_records(v, inherited, out, src)
    elif isinstance(obj, list):
        for v in obj:
            walk_records(v, inherited, out, src)
    return out


def collect_recorded(paths):
    rows = []
    for p in paths:
        try:
            obj = json.load(open(p))
        except Exception as e:                        # noqa: BLE001
            rows.append(dict(src=p, error=f"{type(e).__name__}: {e}"))
            continue
        rows.extend(walk_records(obj, None, None, os.path.basename(p)))
    return rows


def crosscheck(recorded, observed, waiting):
    """Сверка записанных хэшей с измеренными сейчас.

    НЕСОВПАДЕНИЕ — это не «мелкое расхождение»: оно означает, что id больше не
    задаёт то же состояние, и значит прежние пары эпизодов между руками
    сравнивались не на общих состояниях.
    """
    res = dict(checked=0, matched=0, mismatched=[], skipped_waiting=0,
               skipped_no_obs=0, bad_files=[], recorded_conflicts=[])
    seen = {}
    for r in recorded:
        if r.get("error"):
            res["bad_files"].append(r)
            continue
        if r.get("waiting_steps") is not None and \
                int(r["waiting_steps"]) != int(waiting):
            res["skipped_waiting"] += 1
            continue
        key = (r.get("suite"), r.get("task_id"), r["init_state_id"])
        # ОДИН id В ОДНОЙ ЗАДАЧЕ НЕ МОЖЕТ ИМЕТЬ ДВУХ РАЗНЫХ ХЭШЕЙ в прежних
        # артефактах: это противоречие внутри уже опубликованных данных.
        if key in seen and seen[key] != r["init_hash_full"]:
            res["recorded_conflicts"].append(
                dict(key=[str(x) for x in key], a=seen[key],
                     b=r["init_hash_full"], src=r["src"]))
        seen[key] = r["init_hash_full"]
    for key, h in sorted(seen.items(), key=lambda kv: str(kv[0])):
        okey = (key[1], key[2])
        if okey not in observed:
            res["skipped_no_obs"] += 1
            continue
        res["checked"] += 1
        if observed[okey] == h:
            res["matched"] += 1
        else:
            res["mismatched"].append(dict(task_id=key[1], init_state_id=key[2],
                                          recorded=h, observed=observed[okey]))
    return res


def budget(n_distinct, used_ids, need_per_task):
    """Сколько СВЕЖИХ состояний осталось и хватает ли их.

    «Свежие» — не использованные ни в K-11e (0..39), ни в K-11g (40..44), ни в
    пилоте. Повторное использование состояния, на котором уже выбирались sigma
    и чекпойнт, превращает финальную оценку в оценку на обучающей выборке.
    """
    used = sorted({int(i) for i in used_ids})
    fresh = sorted(set(range(n_distinct)) - set(used))
    return dict(n_distinct=int(n_distinct), n_used=len(used),
                n_fresh=len(fresh), need_per_task=int(need_per_task),
                enough=bool(len(fresh) >= int(need_per_task)),
                fresh_first=fresh[:8], fresh_last=fresh[-4:])


# ----------------------------- измерение ---------------------------------

def probe_block(get_envs, seed_everything, suite, task_id, ids, waiting,
                seed, image_size=224):
    """Сброс к заданным id и хэши после holostyh шагов. Блоком — одна среда
    на id, чтобы расход ГСЧ и порядок совпадали с прогонами K-11g."""
    seed_everything(int(seed))
    envs, desc = get_envs(suite, {"task_id": int(task_id),
                                  "image_size": int(image_size)}, len(ids))
    try:
        obs = envs.reset(options=[{"init_state_id": int(j)} for j in ids])
        dummy = np.array([[0, 0, 0, 0, 0, 0, -1]] * len(ids))
        for _ in range(int(waiting)):
            obs, _r, _d, _i = envs.step(dummy)
        short, full = obs_hashes(obs, len(ids))
    finally:
        envs.close()
    return desc, short, full


def probe_ids_safe(get_envs, seed_everything, suite, task_id, ids, waiting,
                   seed, n_envs):
    """Хэши для списка id; при отказе блок разбирается по одному.

    Отказ на одном id не должен прятать результаты остальных — иначе граница
    допустимых id измерялась бы с точностью до размера блока.
    """
    out, errs, desc = {}, {}, None
    i = 0
    while i < len(ids):
        block = [int(x) for x in ids[i:i + n_envs]]
        i += n_envs
        try:
            desc, _s, full = probe_block(get_envs, seed_everything, suite,
                                         task_id, block, waiting, seed)
            for j, h in zip(block, full):
                out[j] = h
            print(f"    id {block[0]}..{block[-1]}: ок", flush=True)
        except Exception as e:                         # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"[:200]
            if len(block) == 1:
                errs[block[0]] = msg
                print(f"    id {block[0]}: ОТКАЗ {msg}", flush=True)
                continue
            print(f"    блок {block[0]}..{block[-1]} отказал, по одному",
                  flush=True)
            for j in block:
                try:
                    desc, _s, full = probe_block(get_envs, seed_everything,
                                                 suite, task_id, [j], waiting,
                                                 seed)
                    out[j] = full[0]
                except Exception as e2:                # noqa: BLE001
                    errs[j] = f"{type(e2).__name__}: {e2}"[:200]
                    print(f"    id {j}: ОТКАЗ {errs[j]}", flush=True)
    return desc, out, errs


def find_bound(hashes, errs, probe_ids):
    """Что известно о границе id по результатам разведки.

    Если большие id принимаются, но дают состояние, уже виденное на меньшем
    id, — это МОЛЧАЛИВЫЙ КЛАМП, и именно он делает «160 эпизодов» иллюзией.
    """
    ok = sorted(hashes)
    bad = sorted(errs)
    clamp = []
    first_by_hash = {}
    for j in ok:
        h = hashes[j]
        if h in first_by_hash:
            clamp.append([first_by_hash[h], j])
        else:
            first_by_hash[h] = j
    return dict(probe_ids=list(probe_ids), accepted=ok, refused=bad,
                first_refused=(bad[0] if bad else None),
                max_accepted=(ok[-1] if ok else None),
                silent_clamp_pairs=clamp,
                verdict=("кламп" if clamp else
                         ("граница" if bad else "граница не найдена")))


# ------------------------------ самопроверка ------------------------------

def selftest():
    # 1. дубликаты и их группировка
    d = dedup({0: "a", 1: "b", 2: "a", 3: "c", 4: "a"})
    assert d["n_ids"] == 5 and d["n_distinct"] == 3, d
    assert d["dup_groups"] == [[0, 2, 4]], d
    assert d["n_dup_ids"] == 2, d
    assert dedup({0: "a", 1: "b"})["dup_groups"] == []

    # 2. обход артефакта произвольной формы: task_id берётся из охватывающей
    #    записи, а не угадывается
    art = dict(task_id=7, suite=10, waiting_steps=10, episodes=[
        dict(init_state_id=0, init_hash_full="h0", success=True),
        dict(init_state_id=1, init_hash_full="h1", success=False)])
    rows = walk_records(art, None, None, "a.json")
    assert len(rows) == 2 and {r["task_id"] for r in rows} == {7}, rows
    assert {r["suite"] for r in rows} == {"10"}, rows
    nested = dict(arms={"fullbar": dict(cells=[
        dict(task_id=3, waiting_steps=10,
             episodes=[dict(init_state_id=5, init_hash_full="z")])])})
    rows = walk_records(nested, None, None, "b.json")
    assert rows == [dict(src="b.json", task_id=3, suite=None,
                         waiting_steps=10, init_state_id=5,
                         init_hash_full="z", init_hash=None)], rows
    # запись без init_hash_full не подхватывается: сверять нечего
    assert walk_records(dict(task_id=1, episodes=[dict(init_state_id=0)])) == []

    # 3. сверка: совпадение, расхождение, пропуск по waiting_steps
    rec = [dict(src="a", task_id=1, suite="10", waiting_steps=10,
                init_state_id=0, init_hash_full="h0"),
           dict(src="a", task_id=1, suite="10", waiting_steps=10,
                init_state_id=1, init_hash_full="WRONG"),
           dict(src="a", task_id=1, suite="10", waiting_steps=3,
                init_state_id=2, init_hash_full="h2"),
           dict(src="a", task_id=1, suite="10", waiting_steps=10,
                init_state_id=9, init_hash_full="h9")]
    obs = {(1, 0): "h0", (1, 1): "h1", (1, 2): "h2"}
    r = crosscheck(rec, obs, waiting=10)
    assert r["checked"] == 2 and r["matched"] == 1, r
    assert len(r["mismatched"]) == 1 and r["mismatched"][0]["init_state_id"] == 1
    assert r["skipped_waiting"] == 1, r
    assert r["skipped_no_obs"] == 1, r          # id 9 сейчас не измеряли
    # противоречие ВНУТРИ прежних артефактов ловится отдельно
    rec2 = rec + [dict(src="b", task_id=1, suite="10", waiting_steps=10,
                       init_state_id=0, init_hash_full="other")]
    assert crosscheck(rec2, obs, 10)["recorded_conflicts"], "конфликт пропущен"

    # 4. граница: молчаливый кламп и честный отказ различаются
    b = find_bound({0: "a", 1: "b", 50: "a"}, {}, [0, 1, 50])
    assert b["verdict"] == "кламп" and b["silent_clamp_pairs"] == [[0, 50]], b
    b = find_bound({0: "a", 1: "b"}, {50: "IndexError"}, [0, 1, 50])
    assert b["verdict"] == "граница" and b["first_refused"] == 50, b
    b = find_bound({0: "a", 1: "b"}, {}, [0, 1])
    assert b["verdict"] == "граница не найдена", b

    # 5. бюджет состояний
    bg = budget(50, list(range(40)) + [40, 41, 42, 43, 44], 80)
    assert bg["n_fresh"] == 5 and not bg["enough"], bg
    assert budget(200, range(45), 80)["enough"], budget(200, range(45), 80)
    # повторное использование занятого id не считается свежим
    assert budget(10, [0, 0, 1], 1)["n_fresh"] == 8

    # 6. хэш воспроизводим и зависит от ВСЕХ частей
    a = np.arange(6, dtype=np.float64)
    assert ep_hash([a]) == ep_hash([a.copy()])
    assert ep_hash([a, a]) != ep_hash([a, a + 1e-6])
    print("самопроверка k12c_states_probe пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--suite", default="10")
    ap.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--waiting-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-b", type=int, default=7,
                    help="второй сид: проверка, создаёт ли сид новые состояния")
    ap.add_argument("--enum-max", type=int, default=60,
                    help="докуда перечислять id после разведки границы")
    ap.add_argument("--stage", default="all",
                    choices=["bound", "all"])
    ap.add_argument("--used-ids", default="0-44",
                    help="занятые прежними прогонами id, например 0-39,40-44")
    ap.add_argument("--need-per-task", type=int, default=80)
    ap.add_argument("--recorded", default="",
                    help="через запятую: артефакты K-11e/K-11g для сверки")
    ap.add_argument("--out", default="data/k12c_states.json")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    sys.path.insert(0, os.path.abspath("experiments"))
    from utils import get_envs, seed_everything   # noqa: E402

    import inspect
    try:
        src_file = inspect.getsourcefile(get_envs)
    except Exception:                              # noqa: BLE001
        src_file = None
    print(f"get_envs из {src_file}\n--- исходник get_envs ---", flush=True)
    try:
        print(inspect.getsource(get_envs), flush=True)
    except Exception as e:                         # noqa: BLE001
        print(f"  исходник недоступен: {e}", flush=True)
    print("--- конец исходника ---", flush=True)

    used = []
    for part in args.used_ids.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            used += list(range(int(a), int(b) + 1))
        else:
            used.append(int(part))

    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()]
    out = dict(suite=args.suite, waiting_steps=args.waiting_steps,
               seed=args.seed, seed_b=args.seed_b, n_envs=args.n_envs,
               enum_max=args.enum_max, used_ids=sorted(set(used)),
               need_per_task=args.need_per_task, get_envs_file=src_file,
               script_sha1=hashlib.sha1(
                   open(os.path.abspath(__file__), "rb").read()
               ).hexdigest()[:12], tasks={})

    observed_all = {}
    for t in task_ids:
        print(f"\n[задача {t}] разведка границы id", flush=True)
        desc, h0, e0 = probe_ids_safe(get_envs, seed_everything, args.suite, t,
                                      PROBE_IDS, args.waiting_steps,
                                      args.seed, 1)
        bound = find_bound(h0, e0, PROBE_IDS)
        rec = dict(task_description=desc, bound=bound, errors=e0)
        print(f"  граница: {bound['verdict']}, принято до "
              f"{bound['max_accepted']}, первый отказ "
              f"{bound['first_refused']}", flush=True)

        if args.stage == "all":
            lim = args.enum_max
            if bound["first_refused"] is not None:
                lim = min(lim, int(bound["first_refused"]))
            ids = list(range(lim))
            print(f"  перечисление id 0..{lim - 1}", flush=True)
            _d, hs, es = probe_ids_safe(get_envs, seed_everything, args.suite,
                                        t, ids, args.waiting_steps, args.seed,
                                        args.n_envs)
            hs.update({k: v for k, v in h0.items() if k < lim})
            rec["enum"] = dedup(hs)
            rec["enum_errors"] = es
            rec["budget"] = budget(rec["enum"]["n_distinct"], used,
                                   args.need_per_task)
            for j, h in hs.items():
                observed_all[(t, int(j))] = h
            print(f"  различных состояний {rec['enum']['n_distinct']} из "
                  f"{rec['enum']['n_ids']}; свежих "
                  f"{rec['budget']['n_fresh']}, нужно "
                  f"{args.need_per_task} -> "
                  f"{'хватает' if rec['budget']['enough'] else 'НЕ ХВАТАЕТ'}",
                  flush=True)

            # ДРУГОЙ СИД: если хэши те же, сид новых эпизодов не создаёт
            sb_ids = ids[:min(5, len(ids))]
            _d, hb, _e = probe_ids_safe(get_envs, seed_everything, args.suite,
                                        t, sb_ids, args.waiting_steps,
                                        args.seed_b, args.n_envs)
            same = [int(j) for j in sb_ids if hb.get(j) == hs.get(j)]
            rec["seed_effect"] = dict(
                ids=sb_ids, same=same, n_same=len(same),
                verdict=("сид не меняет начальное состояние"
                         if len(same) == len(sb_ids) else
                         "сид меняет начальное состояние"))
            print(f"  сид {args.seed_b}: совпало {len(same)}/{len(sb_ids)} — "
                  f"{rec['seed_effect']['verdict']}", flush=True)
        out["tasks"][str(t)] = rec

    paths = [p for p in args.recorded.split(",") if p.strip()]
    if paths:
        rows = collect_recorded(paths)
        out["crosscheck"] = crosscheck(rows, observed_all, args.waiting_steps)
        out["crosscheck"]["n_recorded_rows"] = len(rows)
        cc = out["crosscheck"]
        print(f"\nсверка с прежними артефактами: сверено {cc['checked']}, "
              f"совпало {cc['matched']}, расхождений "
              f"{len(cc['mismatched'])}, конфликтов внутри артефактов "
              f"{len(cc['recorded_conflicts'])}", flush=True)

    if out["tasks"] and all("budget" in r for r in out["tasks"].values()):
        mn = min(r["budget"]["n_fresh"] for r in out["tasks"].values())
        mnd = min(r["enum"]["n_distinct"] for r in out["tasks"].values())
        out["summary"] = dict(
            min_distinct=mnd, min_fresh=mn,
            enough_everywhere=all(r["budget"]["enough"]
                                  for r in out["tasks"].values()),
            any_silent_clamp=any(r["bound"]["silent_clamp_pairs"]
                                 for r in out["tasks"].values()),
            seed_creates_states=any(
                r.get("seed_effect", {}).get("n_same", 0)
                < len(r.get("seed_effect", {}).get("ids", []))
                for r in out["tasks"].values()))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    tmp = args.out + ".tmp"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"\nсохранено: {args.out}")
    if "summary" in out:
        s = out["summary"]
        print(f"ИТОГ: минимум различных состояний на задачу {s['min_distinct']},"
              f" свежих {s['min_fresh']} при нужных {args.need_per_task}; "
              f"молчаливый кламп: {s['any_silent_clamp']}; сид создаёт "
              f"состояния: {s['seed_creates_states']}")


if __name__ == "__main__":
    main()
