#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14: общие загрузчики и план батчей. ОДНА реализация на всех потребителей.

ЗАЧЕМ ОТДЕЛЬНЫМ МОДУЛЕМ. Загрузка состояний и строгая загрузка Joint12 были
написаны трижды — в K-11a, в тренере и в построителе q0, — и каждый раз новый
потребитель оказывался слабее предыдущего: в тренере проверки усилили, а в
построителе канонического q0, где они важнее всего, осталась только сверка
ключей. Дублирование логики означает, что усиление приходится повторять, и
кто-то один всегда остаётся fail-open.

ПЛАН БАТЧЕЙ ЗДЕСЬ ЖЕ. Он стал частью определения задачи (решения 5.1 и 5.2):
forward-размер батча 8 входит в идентичность, набор частей — тоже.
"""
import hashlib
import json
import os

import numpy as np

SCHEMA_VERSION = "k14-plan-1"
CANONICAL_PARTS = ("train", "val_sel", "val_confirm")
CANONICAL_BATCH = 8
SPLIT_SEED = 61
SEL_FRAC = 0.4


def arr_sha(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def file_sha(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:12]


# ----------------------------- план батчей --------------------------------

def make_plan(parts, offs, batch, declared=CANONICAL_PARTS):
    """Батчи из строк с одинаковым смещением. ДЕТЕРМИНИРОВАН.

    Состав задаётся раз и навсегда: строки сортируются, группы по возрастанию
    смещения, нарезка последовательная. Ни сид, ни порядок словаря не влияют.

    `declared` — ОБЪЯВЛЕННЫЙ набор частей. Он входит в идентичность плана даже
    если какая-то часть пуста: иначе удаление пустой части оставило бы тот же
    отпечаток при другом определении задачи.
    """
    miss = [p for p in declared if p not in parts]
    if miss:
        raise SystemExit(f"в наборе нет объявленных частей {miss}")
    extra = [p for p in parts if p not in declared]
    if extra:
        raise SystemExit(f"части вне объявленного набора: {extra}")
    plan = []
    for name in declared:
        rows = np.sort(np.asarray(parts[name], np.int64))
        for po in sorted(set(int(x) for x in offs[rows])):
            sel = rows[offs[rows] == po]
            for i in range(0, len(sel), int(batch)):
                plan.append((str(name), int(po),
                             np.asarray(sel[i:i + int(batch)], np.int64)))
    return plan


def plan_identity(plan, batch, declared=CANONICAL_PARTS,
                  schema=SCHEMA_VERSION):
    """Отпечаток ПЛАНА ЦЕЛИКОМ, включая размер батча и набор частей.

    ОТПЕЧАТОК ТОЛЬКО ПО СТРОКАМ НЕДОСТАТОЧЕН. Если в каждой группе по одной
    строке, батчи 1, 2, 4 и 8 дают одинаковую нарезку и одинаковый отпечаток —
    а задача при этом разная. Поэтому в хеш входят версия схемы, размер батча,
    объявленный набор частей и для каждого батча его часть, смещение, длина и
    сами строки.
    """
    h = hashlib.sha1()
    h.update(str(schema).encode())
    h.update(b"|batch=")
    h.update(str(int(batch)).encode())
    h.update(b"|parts=")
    h.update(json.dumps(list(declared)).encode())
    for name, po, rows in plan:
        h.update(b"|")
        h.update(str(name).encode())
        h.update(str(int(po)).encode())
        h.update(str(int(len(rows))).encode())
        h.update(np.ascontiguousarray(np.asarray(rows, np.int64)).tobytes())
    return h.hexdigest()[:12]


def plan_arrays(plan):
    """Плоское представление плана С ГРАНИЦАМИ БАТЧЕЙ.

    Плоских строк без границ мало: по ним нельзя восстановить, где кончается
    один батч и начинается следующий, а состав батчей и есть предмет
    фиксации.
    """
    rows = np.concatenate([r for _n, _o, r in plan]) if plan \
        else np.zeros(0, np.int64)
    ptr = np.zeros(len(plan) + 1, np.int64)
    for i, (_n, _o, r) in enumerate(plan):
        ptr[i + 1] = ptr[i] + len(r)
    return dict(plan_rows=np.asarray(rows, np.int64), batch_ptr=ptr,
                batch_part=np.array([n for n, _o, _r in plan]),
                batch_offset=np.array([o for _n, o, _r in plan], np.int64))


def plan_stats(plan):
    out = {}
    for name, _po, rows in plan:
        d = out.setdefault(name, dict(batches=0, rows=0))
        d["batches"] += 1
        d["rows"] += int(len(rows))
    return out


# --------------------------- загрузка данных ------------------------------

def check_gate_r(path, *, expect_plan_sha1=None, expect_batch=CANONICAL_BATCH,
                 expect_parts=CANONICAL_PARTS):
    """Требовать пройденный Gate R. ОТСУТСТВИЕ АРТЕФАКТА — ОТКАЗ.

    Gate R утверждает: план батчей достаточен, чтобы q0 воспроизводился в
    другом процессе при другом порядке исполнения. Без этого утверждения цели,
    построенные от q0, не определены однозначно: тренер с другим сидом
    переставляет батчи и решал бы другую задачу. Проверка вынесена сюда, чтобы
    все трое потребителей (K-14a, K-14b, тренер) требовали одного и того же, а
    не каждый по-своему.
    """
    if not path:
        raise SystemExit(
            "не указан артефакт Gate R. Целями может быть только q0, про "
            "который доказано, что он воспроизводится при другом порядке "
            "исполнения того же плана")
    if not os.path.exists(path):
        raise SystemExit(f"нет {path}: Gate R не проводился")
    r = json.load(open(path))
    if r.get("kind") != "k14_gate_r":
        raise SystemExit(f"{path} описывает {r.get('kind')}, а нужен "
                         f"k14_gate_r")
    if r.get("passed") is not True:
        raise SystemExit(f"Gate R не пройден: {r.get('failures')}")
    if r.get("failures"):
        raise SystemExit(f"Gate R помечен пройденным при отказах "
                         f"{r['failures']}: артефакт противоречив")
    if int(r.get("batch", -1)) != int(expect_batch):
        raise SystemExit(f"Gate R проведён при batch {r.get('batch')}, нужен "
                         f"{expect_batch}")
    parts = r.get("parts") or {}
    miss = [p for p in expect_parts if p not in parts]
    if miss:
        raise SystemExit(f"Gate R не покрывает части {miss}")
    bad = [p for p in expect_parts if not parts[p].get("same")]
    if bad:
        raise SystemExit(f"Gate R: q0 разошёлся по частям {bad}")
    seeds = r.get("exec_order_seeds") or []
    if len(seeds) != 2 or seeds[0] == seeds[1]:
        raise SystemExit(f"Gate R по порядкам {seeds}: это не два разных "
                         f"исполнения")
    if expect_plan_sha1 and r.get("plan_sha1") != expect_plan_sha1:
        raise SystemExit(f"Gate R проведён по плану {r.get('plan_sha1')}, а "
                         f"данные построены по {expect_plan_sha1}")
    return dict(gate_r_path=path, gate_r_sha1=file_sha(path),
                plan_sha1=r.get("plan_sha1"), exec_order_seeds=seeds,
                run_ids=r.get("run_ids"))


def load_states(src, n_obs, dataset_repo, dataset_revision, keys_sha,
                state_q01, state_q99):
    """Состояния наблюдений. ВСЕ поля меты обязательны и сверяются точно.

    Состояния входят в промпт, то есть определяют вход целиком. Собранные из
    другой ревизии датасета или для других ключей, они дали бы другую модельную
    выдачу при совпадающих отпечатках всего остального.

    ОТКУДА БЕРУТСЯ ОЖИДАЕМЫЕ РЕПОЗИТОРИЙ И РЕВИЗИЯ. Не из `meta` исходного
    npz: K-9a туда их не пишет — он записывает лишь ПУТЬ к манифесту
    разбиения, а сами поля лежат в манифесте. Оттуда их переносит K-11a в
    `meta["manifest"]`, и вынимать их положено `k11b.dataset_source`, которая
    заодно ловит противоречие между вложенной и верхнеуровневой ревизией.
    Прежняя версия сверялась с `meta` исходного кэша и потому не могла пройти
    никогда: она требовала поля, которых там нет по построению. Проверка,
    которая не выполняется ни на каких данных, не строже отсутствующей — она
    просто заменяет собой настоящую.
    """
    if not dataset_repo or not dataset_revision:
        raise SystemExit(
            "не заданы ожидаемые репозиторий и ревизия данных: их берут из "
            "meta кэша K-11a через k11b.dataset_source, а не угадывают")
    st_p, stm_p = src + ".state.npy", src + ".state.json"
    if not (os.path.exists(st_p) and os.path.exists(stm_p)):
        raise SystemExit(
            f"нет {st_p}: состояния собираются K-11a рядом с ИСХОДНЫМ кэшем. "
            f"Пересобирать их другим кодом нельзя — получился бы другой вход")
    sm = json.load(open(stm_p))
    need = ("keys_sha1", "n_obs", "dataset_repo", "dataset_revision", "dim")
    miss = [k for k in need if sm.get(k) is None]
    if miss:
        raise SystemExit(f"в {stm_p} нет полей {miss}")
    bad = []
    if sm["keys_sha1"] != keys_sha:
        bad.append(f"ключи {sm['keys_sha1']} против {keys_sha}")
    if int(sm["n_obs"]) != int(n_obs):
        bad.append(f"наблюдений {sm['n_obs']}, ожидалось ровно {n_obs}")
    for k, want in (("dataset_repo", dataset_repo),
                    ("dataset_revision", dataset_revision)):
        if str(sm[k]) != str(want):
            bad.append(f"{k}: состояния {sm[k]}, кэш собран на {want}")
    if bad:
        raise SystemExit("состояния не от тех наблюдений: " + "; ".join(bad))
    raw = np.load(st_p)
    if raw.ndim != 2 or raw.shape[0] != int(n_obs):
        raise SystemExit(f"состояния формы {raw.shape}, ожидалось "
                         f"({n_obs}, dim)")
    if int(sm["dim"]) != int(raw.shape[1]) or raw.shape[1] != len(state_q01):
        raise SystemExit(f"размерность {raw.shape[1]}, в мете {sm['dim']}, "
                         f"нормировка ждёт {len(state_q01)}")
    if not np.isfinite(raw).all():
        raise SystemExit("в состояниях есть nan или inf")
    norm = (raw - state_q01) / (state_q99 - state_q01) * 2.0 - 1.0
    return norm, sm, dict(state_npy=file_sha(st_p),
                          state_json=file_sha(stm_p))


def load_canonical_q0(path, *, gate_r_path, n_obs, keys_sha,
                      cache_meta_sha1=None):
    """Черновик q0 как АРТЕФАКТ K-14d, а не как файл сентябрьского кэша.

    Кэш K-11a сегодня побитово не воспроизводится: три независимых пути счёта
    совпадают между собой и одинаково расходятся с ним. Причина — план
    батчей, а не метод. Поэтому канонический q0 строится один раз своим планом
    и сопровождается доказательством Gate R, что план достаточен. Загружать
    его без этого доказательства значило бы вернуть ту же неопределённость,
    ради устранения которой всё и делалось.
    """
    man_p = path[:-4] + ".manifest.json" if path.endswith(".npz") \
        else path + ".manifest.json"
    npz_p = path if path.endswith(".npz") else path + ".npz"
    for q in (npz_p, man_p):
        if not os.path.exists(q):
            raise SystemExit(f"нет {q}")
    man = json.load(open(man_p))
    if man.get("kind") != "k14_q0_by_plan":
        raise SystemExit(f"{man_p} описывает {man.get('kind')}")
    need = ("plan_sha1", "batch", "npz_sha1", "keys_sha1", "limit",
            "git_dirty", "git_head", "run_id", "q0_dtype", "parts")
    miss = [k for k in need if man.get(k) is None]
    if miss:
        raise SystemExit(f"в {man_p} нет полей {miss}")
    gr = check_gate_r(gate_r_path, expect_plan_sha1=man["plan_sha1"],
                      expect_batch=int(man["batch"]))
    if man["limit"]:
        raise SystemExit(f"q0 построен с ограничением --limit {man['limit']}: "
                         f"это не полный канонический план")
    if man["git_dirty"]:
        raise SystemExit("q0 построен при незакоммиченных изменениях")
    got = file_sha(npz_p)
    if got != man["npz_sha1"]:
        raise SystemExit(f"{npz_p} имеет sha {got}, в манифесте "
                         f"{man['npz_sha1']}")
    if man["keys_sha1"] != keys_sha:
        raise SystemExit(f"q0 построен по ключам {man['keys_sha1']}, поданы "
                         f"{keys_sha}")
    if cache_meta_sha1 and man.get("cache_meta_sha1") != cache_meta_sha1:
        raise SystemExit(f"q0 построен по мете {man.get('cache_meta_sha1')}, "
                         f"подана {cache_meta_sha1}")
    with np.load(npz_p, allow_pickle=True) as z:
        q0 = np.asarray(z["q0"])
    if q0.shape != (int(n_obs), 16):
        raise SystemExit(f"q0 формы {q0.shape}, ожидалась ({n_obs}, 16)")
    if str(q0.dtype) != str(man["q0_dtype"]):
        raise SystemExit(f"q0 типа {q0.dtype}, в манифесте {man['q0_dtype']}")
    # СТРОКИ ВНЕ КАНОНИЧЕСКИХ ЧАСТЕЙ ОСТАЛИСЬ НЕПОСЧИТАННЫМИ и помечены -1.
    # Это не ошибка: план покрывает только части. Но потребитель обязан знать,
    # какие строки определены, иначе -1 молча уйдёт в метки как код 65535.
    defined = q0.min(axis=1) >= 0
    for nm, pm in man["parts"].items():
        if int(pm.get("n_rows", -1)) < 0:
            raise SystemExit(f"в манифесте нет числа строк части {nm}")
    prov = dict(q0_source="k14d_plan", q0_npz=npz_p, q0_npz_sha1=got,
                q0_manifest_sha1=file_sha(man_p), plan_sha1=man["plan_sha1"],
                plan_batch=int(man["batch"]), q0_run_id=man["run_id"],
                q0_git_head=man["git_head"],
                q0_parts={k: v.get("q0_sha1") for k, v in
                          man["parts"].items()},
                **{k: v for k, v in gr.items() if k != "plan_sha1"})
    return q0.astype(np.int64), defined, man, prov


def load_joint12_strict(model, path, expect_depth, expect_sha, torch, dev):
    """Строгая загрузка Joint12: лишние, недостающие, формы, конечность.

    Частично применённый чекпойнт даёт правдоподобные, но неверные числа, и
    ни одна метрика этого не покажет.
    """
    got = file_sha(path)
    if expect_sha and got != expect_sha:
        raise SystemExit(f"Joint12 {got}, ожидался {expect_sha}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if int(obj.get("depth", -1)) != int(expect_depth):
        raise SystemExit(f"чекпойнт глубины {obj.get('depth')}, ожидалась "
                         f"{expect_depth}")
    state = obj["state"]
    own = dict(model.named_parameters())
    stray = [k for k in state
             if not any(k.startswith(p) or k == p.rstrip(".")
                        for p in model.trainable_prefixes)]
    if stray:
        raise SystemExit(f"{len(stray)} ключей вне белого списка: {stray[:5]}")
    missing = [k for k in own if own[k].requires_grad and k not in state]
    if missing:
        raise SystemExit(f"нет {len(missing)} обучаемых весов: {missing[:5]}")
    with torch.no_grad():
        for k, v in state.items():
            if k not in own:
                raise SystemExit(f"ключ {k} отсутствует в модели")
            if tuple(own[k].shape) != tuple(v.shape):
                raise SystemExit(f"форма {k}: {tuple(v.shape)} против "
                                 f"{tuple(own[k].shape)}")
            if not torch.isfinite(v).all():
                raise SystemExit(f"в {k} есть nan или inf")
            own[k].data = v.to(dev, torch.float32)
    return got, len(state)


def selftest():
    offs = np.array([3, 3, 4, 4, 4, 3, 4, 3], np.int64)
    parts = dict(train=np.array([0, 1, 2, 5]), val_sel=np.array([3, 4]),
                 val_confirm=np.array([6, 7]))
    p1 = make_plan(parts, offs, 2)
    rev = dict(val_confirm=np.array([7, 6]), train=np.array([5, 2, 1, 0]),
               val_sel=np.array([4, 3]))
    assert plan_identity(p1, 2) == plan_identity(make_plan(rev, offs, 2), 2)
    for _n, po, rows in p1:
        assert len(set(int(x) for x in offs[rows])) == 1

    # РАЗМЕР БАТЧА ВХОДИТ В ОТПЕЧАТОК даже когда нарезка совпала: здесь в
    # каждой группе по одной строке, и батчи 1 и 8 дают ОДИН план.
    one = dict(train=np.array([0]), val_sel=np.array([3]),
               val_confirm=np.array([6]))
    a1, a8 = make_plan(one, offs, 1), make_plan(one, offs, 8)
    assert [list(r) for _n, _o, r in a1] == [list(r) for _n, _o, r in a8]
    assert plan_identity(a1, 1) != plan_identity(a8, 8), \
        "размер батча не вошёл в отпечаток"

    # НАБОР ЧАСТЕЙ ВХОДИТ ТОЖЕ, включая пустую часть
    empty = dict(one, val_confirm=np.zeros(0, np.int64))
    pe = make_plan(empty, offs, 2)
    assert plan_identity(pe, 2) != plan_identity(
        pe, 2, declared=("train", "val_sel", "val_confirm", "extra"))
    try:
        make_plan(dict(train=np.array([0])), offs, 2)
    except SystemExit as e:
        assert "объявленных частей" in str(e), e
    else:
        raise AssertionError("неполный набор частей принят")
    try:
        make_plan(dict(one, лишняя=np.array([1])), offs, 2)
    except SystemExit as e:
        assert "вне объявленного" in str(e), e
    else:
        raise AssertionError("лишняя часть принята")

    # ГРАНИЦЫ БАТЧЕЙ ВОССТАНАВЛИВАЮТ СОСТАВ
    ar = plan_arrays(p1)
    assert len(ar["batch_ptr"]) == len(p1) + 1
    for i, (nm, po, rows) in enumerate(p1):
        a_, b_ = ar["batch_ptr"][i], ar["batch_ptr"][i + 1]
        assert list(ar["plan_rows"][a_:b_]) == list(rows)
        assert ar["batch_part"][i] == nm and ar["batch_offset"][i] == po
    import tempfile
    ok = dict(kind="k14_gate_r", passed=True, failures=[], batch=CANONICAL_BATCH,
              exec_order_seeds=[0, 7], plan_sha1="P", run_ids=["A", "B"],
              parts={p: dict(same=True) for p in CANONICAL_PARTS})
    with tempfile.TemporaryDirectory() as td:
        def w(obj, nm="g.json"):
            q = os.path.join(td, nm)
            json.dump(obj, open(q, "w"))
            return q
        assert check_gate_r(w(ok), expect_plan_sha1="P")["plan_sha1"] == "P"
        for patch, why in (
                ({"passed": False}, "не пройден"),
                ({"failures": ["x"]}, "противоречив"),
                ({"batch": 16}, "batch"),
                ({"exec_order_seeds": [0, 0]}, "не два разных"),
                ({"kind": "other"}, "нужен"),
                ({"parts": {p: dict(same=True) for p in CANONICAL_PARTS[:2]}},
                 "не покрывает")):
            try:
                check_gate_r(w(dict(ok, **patch)))
            except SystemExit as e:
                assert why in str(e), (why, e)
            else:
                raise AssertionError(f"Gate R принят при {patch}")
        try:
            check_gate_r(w(ok), expect_plan_sha1="Q")
        except SystemExit as e:
            assert "по плану" in str(e)
        else:
            raise AssertionError("принят Gate R по чужому плану")
        for bad in ("", os.path.join(td, "нет.json")):
            try:
                check_gate_r(bad)
            except SystemExit:
                pass
            else:
                raise AssertionError("принято отсутствие Gate R")

        # --- канонический q0 ------------------------------------------------
        grp = w(ok)
        q0 = np.full((7, 16), 3, np.int32)
        q0[5:] = -1                       # строки вне частей не считались
        base = os.path.join(td, "q0")
        np.savez_compressed(open(base + ".npz", "wb"), q0=q0)
        def wm(**kw):
            m = dict(kind="k14_q0_by_plan", plan_sha1="P",
                     batch=CANONICAL_BATCH, npz_sha1=file_sha(base + ".npz"),
                     keys_sha1="KS", limit=0, git_dirty=False, git_head="H",
                     run_id="R", q0_dtype="int32", cache_meta_sha1="CM",
                     parts={p_: dict(n_rows=2, q0_sha1="s" + p_)
                            for p_ in CANONICAL_PARTS})
            m.update(kw)
            json.dump(m, open(base + ".manifest.json", "w"))
            return base + ".npz"
        got_q0, defined, man_, prov = load_canonical_q0(
            wm(), gate_r_path=grp, n_obs=7, keys_sha="KS",
            cache_meta_sha1="CM")
        assert got_q0.dtype == np.int64 and got_q0.shape == (7, 16)
        assert defined.sum() == 5, defined.sum()
        assert prov["plan_sha1"] == "P" and prov["gate_r_sha1"]
        for kw, why in (({"keys_sha1": "OTHER"}, "по ключам"),
                        ({"limit": 99}, "--limit"),
                        ({"git_dirty": True}, "незакоммиченных"),
                        ({"npz_sha1": "beef"}, "в манифесте"),
                        ({"plan_sha1": "Q"}, "по плану"),
                        ({"cache_meta_sha1": "XX"}, "по мете"),
                        ({"q0_dtype": "int64"}, "типа"),
                        ({"kind": "other"}, "описывает")):
            try:
                load_canonical_q0(wm(**kw), gate_r_path=grp, n_obs=7,
                                  keys_sha="KS", cache_meta_sha1="CM")
            except SystemExit as e:
                assert why in str(e), (why, e)
            else:
                raise AssertionError(f"q0 принят при {kw}")
        try:
            load_canonical_q0(wm(), gate_r_path=grp, n_obs=9, keys_sha="KS")
        except SystemExit as e:
            assert "формы" in str(e), e
        else:
            raise AssertionError("q0 чужого размера принят")
        # без доказательства Gate R q0 не грузится вовсе
        try:
            load_canonical_q0(wm(), gate_r_path="", n_obs=7, keys_sha="KS")
        except SystemExit:
            pass
        else:
            raise AssertionError("q0 принят без Gate R")

    # --- СОСТОЯНИЯ: ПОЛОЖИТЕЛЬНЫЙ ПУТЬ ОБЯЗАТЕЛЕН -------------------------
    # Первая версия этой проверки сверялась не с тем источником и не могла
    # пройти НИ НА КАКИХ данных. Отрицательные случаи её пропускали: они все
    # ожидали отказа, а она отказывала всегда. Ловится это только тем, что
    # корректный вход обязан приниматься.
    with tempfile.TemporaryDirectory() as td:
        q01 = np.array([0.0, -1.0, 0.0], np.float64)
        q99 = np.array([2.0, 1.0, 4.0], np.float64)
        src = os.path.join(td, "cache.npz")
        raw = np.array([[1.0, 0.0, 2.0], [0.0, -1.0, 0.0],
                        [2.0, 1.0, 4.0], [1.0, 0.5, 1.0]])
        np.save(src + ".state.npy", raw)
        def wsm(**kw):
            m = dict(keys_sha1="KS", n_obs=4, dataset_repo="repo/x",
                     dataset_revision="v2.0", dim=3)
            m.update(kw)
            json.dump(m, open(src + ".state.json", "w"))
        wsm()
        norm, sm_, shas = load_states(src, 4, "repo/x", "v2.0", "KS", q01, q99)
        assert norm.shape == (4, 3)
        assert abs(float(norm[1].min()) + 1.0) < 1e-12 and \
            abs(float(norm[2].max()) - 1.0) < 1e-12, norm
        assert shas["state_npy"] and shas["state_json"]
        for kw, args_, why in (
                ({"keys_sha1": "OTHER"}, ("repo/x", "v2.0", "KS"), "ключи"),
                ({"n_obs": 5}, ("repo/x", "v2.0", "KS"), "ровно"),
                ({"dim": 4}, ("repo/x", "v2.0", "KS"), "размерность"),
                ({}, ("repo/y", "v2.0", "KS"), "dataset_repo"),
                ({}, ("repo/x", "v1.0", "KS"), "dataset_revision"),
                ({"dataset_repo": None}, ("repo/x", "v2.0", "KS"),
                 "нет полей")):
            wsm(**kw)
            try:
                load_states(src, 4, args_[0], args_[1], args_[2], q01, q99)
            except SystemExit as e:
                assert why in str(e), (why, e)
            else:
                raise AssertionError(f"состояния приняты при {kw} {args_}")
        wsm()
        for bad in (("", "v2.0"), ("repo/x", "")):
            try:
                load_states(src, 4, bad[0], bad[1], "KS", q01, q99)
            except SystemExit as e:
                assert "не заданы" in str(e), e
            else:
                raise AssertionError("принято пустое происхождение данных")
        # число наблюдений сверяется ТОЧНО, а не «не меньше»
        try:
            load_states(src, 3, "repo/x", "v2.0", "KS", q01, q99)
        except SystemExit as e:
            assert "формы" in str(e) or "ровно" in str(e), e
        else:
            raise AssertionError("принят другой размер набора")

    print("самопроверка k14_common пройдена")


if __name__ == "__main__":
    selftest()
