#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-13c: детерминированное сравнение четырёх политик на общих состояниях.

ЧЕТЫРЕ РУКИ, А НЕ ТРИ:

    fast12          — исходная политика раннего выхода, БЕЗ поправки.
                      Именно к ней HiCoRA и HiCoRA-T добавляют поправку, и
                      именно с ней HiCoRA-T с нулевой головой совпадает
                      побитово. Без неё возможен двусмысленный исход:
                      HiCoRA-T лучше HiCoRA, но обе хуже политики без поправки.
    coarse24        — generate на 24 слоя, нулевой уровень. Цена раннего
                      выхода измеряется разностью coarse24 и fast12.
    hicora_d1_det   — существующая поправка: 16 независимых наборов
                      коэффициентов, по одному на позицию чанка.
    hicora_t_d1_det — траекторная поправка: один набор из 64 на весь чанк.

ГЛАВНОЕ СРАВНЕНИЕ ПОПАРНО ПО ГОЛОВАМ: T_s минус D1_s. Опора — детерминированная
D1, а не лучшая точка шумной RL-лестницы: та выбрана постфактум среди шести
оценок на одном dev-наборе и не воспроизвелась ни другой головой, ни
последующими шагами.

ТОЖДЕСТВО ПРОВЕРЯЕТСЯ ИСПОЛНЕНИЕМ, А НЕ РАССУЖДЕНИЕМ. На первом настоящем
батче сверяются q0, dZ, Z и ДЕКОДИРОВАННЫЕ ДЕЙСТВИЯ при обнулённой голове, а
также счётчики: один проход VLA и один вызов декодера. Проверка `dZ == 0` на
сохранённых латентах этого не даёт — она пройдёт и при неверном пути получения
черновика, и при чужом декодировании.

ЧЕГО ТОЖДЕСТВО НЕ ПРОВЕРЯЕТ: правильность базиса и rho. При нулевых
коэффициентах любой базис исчезает, поэтому их происхождение подтверждается
только отпечатками, и они сверяются отдельно.

coarse24 СЧИТАЕТСЯ ЧЕРЕЗ only_blocks(1). Полный generate выполняет три
BAR-блока, из которых нужен лишь первый: успех от этого не меняется, но
стоимость завышена втрое. На smoke-батче коды сверяются с первым блоком
полного generate.
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys
import time

import numpy as np

H_EXEC = 8
ARMS = ("fast12", "coarse24", "hicora_d1_det", "hicora_t_d1_det")
SHARED_ARMS = ("fast12", "coarse24")     # не зависят от головы D1


def arm_needs_head(arm):
    return arm in ("hicora_d1_det", "hicora_t_d1_det")


# pos_offset СЮДА НЕ ВХОДИТ. Оно берётся из таблицы по паре (набор, задача) и
# у разных задач разное: требовать одно значение на все ячейки значило бы
# требовать одинаковой длины промпта у разных задач. Вместо этого сверяется
# таблица целиком, а совпадение по задаче — отдельной проверкой ниже.
FIELDS_SAME = ("ckpt", "max_steps", "horizon", "rollout_seed_mode",
               "waiting_steps", "joint_sha1", "preprocess",
               "image_size", "res_norm_sha1", "precision_mode",
               "offset_table_sha1", "suite", "dtype", "head_precision",
               "stage", "script_sha1", "code_version")

# ПОЛЯ, ПОСТОЯННЫЕ ВНУТРИ РУКИ, НО РАЗНЫЕ МЕЖДУ РУКАМИ. Отпечаток головы у
# fast12 и у HiCoRA-T не обязан совпадать — это разные руки. А вот внутри
# одной руки он обязан: смена головы посреди девяти ячеек означала бы, что
# «рука» собрана из двух разных политик, и её успех ничей.
FIELDS_SAME_ARM = ("head_sha1", "basis_sha1", "rho_sha1", "head_seed")

# offset_table_sha1 ЗАКОННО ПУСТ, если смещение задано явным --pos-offset.
# Тогда оно пусто у всех ячеек, и проверка равенства всё равно поймает смесь.
FIELDS_REQUIRED = tuple(f for f in FIELDS_SAME if f != "offset_table_sha1")

# СПИСОК КЛЮЧЕЙ СВОДКИ ВЫВОДИТСЯ ИЗ FIELDS_SAME, А НЕ ПИШЕТСЯ ОТДЕЛЬНО.
# Раньше он задавался руками и разошёлся: поля, которых в нём не было,
# доставались как None, сравнивались None с None и подтверждали «одинаковые
# условия», ничего не проверив.
META_KEYS = tuple(dict.fromkeys(FIELDS_SAME + FIELDS_SAME_ARM
                                + ("identity",)))


def index_episodes(eps, arm):
    """{(задача, состояние): (успех, хэш)} с отказом на дублях.

    Словарь молча перезаписывал бы повтор, и одна ячейка, посчитанная дважды,
    осталась бы незамеченной.
    """
    out = {}
    for e in eps:
        k = (int(e["task_id"]), int(e["state_id"]))
        if k in out:
            raise SystemExit(f"{arm}: эпизод {k} встречается дважды")
        out[k] = (bool(e["success"]), str(e.get("init_hash_full") or ""))
    return out


def check_same_states(idx_by_arm):
    """У всех рук ОДНО И ТО ЖЕ начальное состояние в каждой паре.

    Совпадения (задача, состояние) недостаточно: если хэш начального
    состояния различается, это разные состояния под одним номером, и парная
    разность ничего не измеряет.
    """
    bad = []
    arms = sorted(idx_by_arm)
    ref = arms[0]
    for k, (_s, h) in sorted(idx_by_arm[ref].items()):
        if not h:
            bad.append(f"{ref}: у {k} нет init_hash_full")
            continue
        for other in arms[1:]:
            if k not in idx_by_arm[other]:
                continue
            h2 = idx_by_arm[other][k][1]
            if h2 != h:
                bad.append(f"{k}: {ref} {h} против {other} {h2}")
    if bad:
        raise SystemExit("начальные состояния расходятся между руками:\n  - "
                         + "\n  - ".join(bad[:8]))
    return True


def pair_table(by_arm, want_states, log=print, final=False):
    """Парные разности по общим (задача, состояние).

    НЕДОСЧИТАННАЯ РУКА ИСКЛЮЧАЕТСЯ С ОБЪЯСНЕНИЕМ, а не роняет отчёт: пока
    одна рука считается, остальные сравнивать можно и нужно. А вот РАСХОЖДЕНИЕ
    состояний при полном покрытии — отказ: это разные наборы, и разность по
    ним ничего не измеряет.

    В ИТОГОВОМ РЕЖИМЕ (`final=True`) исключение запрещено. Сводка, из которой
    молча выпала рука, выглядит точно так же, как полная, и разность рук в ней
    посчитана не по тому набору, который заявлен. Промежуточный отчёт этим
    пользоваться может, итоговый — нет.
    """
    idx_by_arm = {arm: index_episodes(eps, arm)
                  for arm, eps in by_arm.items()}
    sets = {arm: set(d) for arm, d in idx_by_arm.items()}
    full = {arm: k for arm, k in sets.items()
            if not want_states or len(k) >= want_states}
    partial = {arm: len(k) for arm, k in sets.items() if arm not in full}
    if partial and final:
        raise SystemExit(
            "итоговый режим: недосчитаны руки "
            + ", ".join(f"{a} ({n} из {want_states})"
                        for a, n in sorted(partial.items()))
            + ". Сводка без руки неотличима от полной")
    for arm, n in sorted(partial.items()):
        log(f"    рука {arm} исключена: {n} пар из {want_states} — ещё "
            f"считается или прервана")
    if not full:
        raise SystemExit("ни одна рука не покрывает нужные состояния")
    keys = None
    for k in full.values():
        keys = k if keys is None else (keys & k)
    extra = {arm: sorted(k - keys)[:5] for arm, k in full.items()}
    bad = {a: m for a, m in extra.items() if m}
    if bad:
        raise SystemExit(f"руки с полным покрытием считаны на РАЗНЫХ "
                         f"состояниях: {bad}")
    if want_states and len(keys) != want_states:
        raise SystemExit(f"общих пар {len(keys)}, ожидалось {want_states}")
    check_same_states({a: idx_by_arm[a] for a in full})
    by_arm = {a: by_arm[a] for a in full}
    succ = {arm: {k: idx_by_arm[arm][k][0] for k in sorted(keys)}
            for arm in by_arm}
    return succ, sorted(keys)


def compare(a, b):
    """a минус b: восстановления, потери, эффект. Точный McNemar."""
    import math
    rec = sum(1 for k in a if a[k] and not b[k])
    los = sum(1 for k in a if b[k] and not a[k])
    n = len(a)
    nd = rec + los
    p = (sum(math.comb(nd, i) for i in range(rec, nd + 1)) / 2.0 ** nd
         if nd else 1.0)
    return dict(n=n, recovered=rec, lost=los,
                success_a=sum(a.values()) / n, success_b=sum(b.values()) / n,
                effect=(rec - los) / n, discord=nd / n, p_one_sided=p)


def verdict(delta_pp, n=45):
    """Инженерный порог из плана. Это go/no-go, а не вывод о значимости.

    СЧИТАЕТСЯ ЧИСТЫЙ ДЕФИЦИТ, А НЕ ЧИСЛО ПОТЕРЬ. Потери есть всегда: при
    дискордантности около четверти (§39) пары расходятся в обе стороны, и
    «потерь ноль» — неверная формулировка даже при положительной разности.
    Порог смотрит на сальдо, и называть его надо так же.
    """
    lost = -round(delta_pp * n / 100.0)
    if lost <= 3:
        return "continue", f"чистый дефицит {max(lost, 0)} исходов из {n} <= 3"
    if lost <= 5:
        return "check", f"чистый дефицит {lost} исходов из {n} в диапазоне 4..5"
    return "fix_head", f"чистый дефицит {lost} исходов из {n} > 5"


def check_pos_offsets(cells):
    """Одна задача — одно смещение позиций у ВСЕХ рук.

    Смещение задаёт, с какой позиции модель читает свой вход, и разное
    смещение на одной задаче означало бы, что руки решали её из разных
    начальных условий. Равенства «по всем ячейкам» здесь требовать нельзя:
    у разных задач смещение разное по построению.
    """
    by_task = {}
    for c in cells:
        k = (str(c.get("suite")), int(c["task_id"]))
        by_task.setdefault(k, {}).setdefault(
            int(c["pos_offset"]), []).append(c.get("_path"))
    bad = {k: v for k, v in by_task.items() if len(v) > 1}
    if bad:
        (suite, task), v = sorted(bad.items())[0]
        raise SystemExit(
            f"набор {suite}, задача {task}: разные pos_offset "
            f"{sorted(v)} — руки читали вход с разных позиций. "
            f"Например {sorted(vv[0] for vv in v.values())}")
    return {k: next(iter(v)) for k, v in by_task.items()}


def check_final_composition(cells, meta_by_arm, heads, expect_cells,
                            log=print):
    """Состав итоговой сводки: какие руки, сколько ячеек, чем подтверждены.

    ВЫНЕСЕНО ФУНКЦИЕЙ РАДИ САМОПРОВЕРКИ. Проверка, живущая только внутри
    main, испытывается лишь настоящим прогоном — то есть тогда, когда ошибаться
    уже поздно.
    """
    # ТОЧНОЕ МНОЖЕСТВО РУК, А НЕ ИХ ЧИСЛО. Полностью отсутствующая голова
    # не попадает в «недосчитанные»: её просто нет, основное сравнение её
    # молча пропускает, и сводка выглядит целой. Проверять надо состав.
    want_arms = set(SHARED_ARMS) | {f"{arm}:{h}" for h in heads
                                    for arm in ("hicora_d1_det",
                                                "hicora_t_d1_det")}
    got_arms = set(meta_by_arm)
    if got_arms != want_arms:
        raise SystemExit(
            f"итоговый режим: нет рук {sorted(want_arms - got_arms)}, "
            f"лишние {sorted(got_arms - want_arms)}")
    per_arm = expect_cells // len(want_arms)
    wrong = {k: len(v) for k, v in meta_by_arm.items()
             if len(v) != per_arm}
    if wrong:
        raise SystemExit(f"итоговый режим: ячеек на руку не по "
                         f"{per_arm}: {wrong}")
    if len(cells) != expect_cells:
        raise SystemExit(
            f"итоговый режим: ячеек {len(cells)}, ожидалось "
            f"{expect_cells} ({len(want_arms)} рук по {per_arm})")
    # ОТПЕЧАТКИ ОБУЧЕННЫХ РУК ОБЯЗАНЫ БЫТЬ НЕПУСТЫ. Внутри руки
    # отсутствующие head_sha1/basis_sha1/rho_sha1 одинаковы как None и
    # проходят проверку равенства, не подтвердив ничего.
    for key, ms in meta_by_arm.items():
        if key in SHARED_ARMS:
            continue
        empty = [f for f in FIELDS_SAME_ARM
                 if any(m.get(f) in (None, "None") for m in ms)]
        if empty:
            raise SystemExit(
                f"итоговый режим: у руки {key} пусты {empty} — "
                f"происхождение головы не подтверждено")
    t_cells_f = [c for c in cells if c.get("arm") == "hicora_t_d1_det"]
    n_t = len(t_cells_f)
    n_id = len([c for c in t_cells_f
                if (c.get("identity") or {}).get("ok")])
    want_t = per_arm * len(heads)
    if n_t != want_t:
        raise SystemExit(f"итоговый режим: ячеек HiCoRA-T {n_t}, "
                         f"ожидалось {want_t}")
    if n_t != n_id:
        raise SystemExit(f"итоговый режим: тождество подтверждено в "
                         f"{n_id} из {n_t} ячеек HiCoRA-T")
    log(f"  итоговый режим: {len(cells)} ячеек, {len(want_arms)} рук по "
        f"{per_arm}, тождество {n_id}/{n_t}, отпечатки голов непусты")
    return want_arms


def selftest():
    # --- КАЖДОЕ ПРОВЕРЯЕМОЕ ПОЛЕ ДОЛЖНО ДОЙТИ ДО ПРОВЕРКИ -----------------
    # Ровно на этом сводка и обманулась: список ключей задавался отдельно от
    # FIELDS_SAME, отстал от него, и половина проверок сравнивала None с None.
    assert set(FIELDS_SAME) <= set(META_KEYS), \
        sorted(set(FIELDS_SAME) - set(META_KEYS))
    assert "precision_mode" in FIELDS_SAME
    assert "pos_offset" not in FIELDS_SAME, "смещение зависит от задачи"
    for f in ("script_sha1", "code_version", "suite", "dtype",
              "head_precision", "stage"):
        assert f in FIELDS_SAME, f
    assert not set(FIELDS_SAME) & set(FIELDS_SAME_ARM)
    assert set(FIELDS_SAME_ARM) <= set(META_KEYS)

    # --- ИТОГОВЫЙ СОСТАВ: отсутствующая рука не «недосчитана», её нет ----
    def _mkcells(arms, per=9, ok=True, blank=()):
        out = []
        for arm, head in arms:
            for i in range(per):
                c = dict(arm=arm, head=head, _path=f"{arm}_{head}_{i}.json",
                         head_sha1="hh", basis_sha1="bb", rho_sha1="rr",
                         head_seed=0)
                for f in blank:
                    c[f] = None
                if arm == "hicora_t_d1_det":
                    c["identity"] = dict(ok=ok)
                out.append(c)
        return out

    six = [("fast12", None), ("coarse24", None),
           ("hicora_d1_det", "s0"), ("hicora_d1_det", "s1"),
           ("hicora_t_d1_det", "s0"), ("hicora_t_d1_det", "s1")]

    def _meta(cs):
        mb = {}
        for c in cs:
            key = (c["arm"] if c["arm"] in SHARED_ARMS
                   else f"{c['arm']}:{c['head']}")
            mb.setdefault(key, []).append(c)
        return mb

    good = _mkcells(six)
    check_final_composition(good, _meta(good), ["s0", "s1"], 54,
                            log=lambda *_: None)
    for cs, why in (
            (_mkcells([x for x in six if x != ("hicora_t_d1_det", "s1")]),
             "нет рук"),
            (_mkcells(six, per=8), "ячеек"),
            (_mkcells(six, ok=False), "тождество подтверждено"),
            (_mkcells(six, blank=("basis_sha1",)), "пусты")):
        try:
            check_final_composition(cs, _meta(cs), ["s0", "s1"],
                                    54, log=lambda *_: None)
        except SystemExit as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"итоговый режим пропустил: {why}")


    # --- смещение позиций: по задаче, а не по всему прогону ---------------
    ok_cells = [dict(suite="libero_10", task_id=t, pos_offset=o,
                     _path=f"{arm}_t{t}.json")
                for arm in ("fast12", "coarse24")
                for t, o in ((3, 3), (6, 4), (8, 4))]
    got = check_pos_offsets(ok_cells)
    assert got[("libero_10", 3)] == 3 and got[("libero_10", 6)] == 4, got
    bad = ok_cells + [dict(suite="libero_10", task_id=3, pos_offset=9,
                           _path="t_s0_t3.json")]
    try:
        check_pos_offsets(bad)
    except SystemExit as e:
        assert "задача 3" in str(e), e
    else:
        raise AssertionError("разное смещение на одной задаче пропущено")

    # --- недосчитанная рука исключается, а не роняет отчёт ----------------
    mk9 = lambda ok: [dict(task_id=3, state_id=i, success=(i in ok),
                           init_hash_full=f"h3_{i}") for i in range(30, 35)]
    part = {"fast12": mk9({30, 31}), "coarse24": [
        dict(task_id=3, state_id=30, success=True, init_hash_full="h3_30")]}
    msgs = []
    sp, kp = pair_table(part, 5, log=msgs.append)
    assert set(sp) == {"fast12"}, sp
    assert any("coarse24 исключена" in m for m in msgs), msgs
    # тот же набор в ИТОГОВОМ режиме — отказ, а не исключение руки
    try:
        pair_table(part, 5, log=lambda *_: None, final=True)
    except SystemExit as e:
        assert "недосчитаны" in str(e), e
    else:
        raise AssertionError("итоговый режим принял неполную руку")
    # но РАСХОЖДЕНИЕ при полном покрытии — по-прежнему отказ
    diff = {"fast12": mk9({30}), "coarse24": [
        dict(task_id=3, state_id=i, success=True, init_hash_full=f"h3_{i}")
        for i in range(40, 45)]}
    try:
        pair_table(diff, 5, log=lambda *_: None)
    except SystemExit as e:
        assert "РАЗНЫХ состояниях" in str(e), e
    else:
        raise AssertionError("расхождение при полном покрытии принято")

    # --- парность и отказ на расхождении ----------------------------------
    mk = lambda ok: [dict(task_id=3, state_id=i, success=(i in ok),
                          init_hash_full=f"h3_{i}") for i in range(30, 35)]
    by = {"fast12": mk({30, 31, 32}), "hicora_t_d1_det": mk({30, 31, 32, 33})}
    succ, keys = pair_table(by, 5)
    assert len(keys) == 5
    c = compare(succ["hicora_t_d1_det"], succ["fast12"])
    assert c["recovered"] == 1 and c["lost"] == 0 and c["n"] == 5, c
    assert abs(c["effect"] - 0.2) < 1e-12
    bad = dict(by)
    bad["coarse24"] = [dict(task_id=3, state_id=i, success=True,
                            init_hash_full=f"h3_{i}") for i in range(40, 45)]
    try:
        pair_table(bad, 5)
    except SystemExit as e:
        assert "на РАЗНЫХ состояниях" in str(e), e
    else:
        raise AssertionError("непарные руки приняты")

    # --- ДУБЛЬ ЭПИЗОДА — ОТКАЗ, а не молчаливая перезапись ----------------
    dup = {"fast12": mk({30}) + [dict(task_id=3, state_id=30, success=False,
                                      init_hash_full="h3_30")]}
    try:
        pair_table(dup, 5, log=lambda *_: None)
    except SystemExit as e:
        assert "встречается дважды" in str(e), e
    else:
        raise AssertionError("дубль эпизода принят")

    # --- РАЗНЫЕ НАЧАЛЬНЫЕ СОСТОЯНИЯ ПОД ОДНИМ НОМЕРОМ — отказ -------------
    other = [dict(task_id=3, state_id=i, success=True,
                  init_hash_full="ДРУГОЙ") for i in range(30, 35)]
    try:
        pair_table({"fast12": mk({30}), "coarse24": other}, 5,
                   log=lambda *_: None)
    except SystemExit as e:
        assert "начальные состояния расходятся" in str(e), e
    else:
        raise AssertionError("разные состояния под одним номером приняты")
    # и отсутствие хэша тоже
    nohash = [dict(task_id=3, state_id=i, success=True) for i in range(30, 35)]
    try:
        pair_table({"fast12": nohash, "coarse24": nohash}, 5,
                   log=lambda *_: None)
    except SystemExit as e:
        assert "нет init_hash_full" in str(e), e
    else:
        raise AssertionError("эпизоды без хэша приняты")

    # --- порог go/no-go ---------------------------------------------------
    assert verdict(0.0)[0] == "continue"
    assert verdict(-3 * 100 / 45)[0] == "continue"      # ровно 3 исхода
    assert verdict(-4 * 100 / 45)[0] == "check"
    assert verdict(-6 * 100 / 45)[0] == "fix_head"
    assert verdict(+8.9)[0] == "continue"

    # --- McNemar ----------------------------------------------------------
    a = {i: True for i in range(10)}
    b = {i: i > 2 for i in range(10)}
    c2 = compare(a, b)
    assert c2["recovered"] == 3 and c2["lost"] == 0
    assert abs(c2["p_one_sided"] - 0.125) < 1e-9, c2["p_one_sided"]
    print("самопроверка k13c_hicora_t_rollout пройдена")


def report(res, want_pairs=45):
    """Таблица результата. Всё попарно и на общих состояниях."""
    succ = res["success"]
    print(f"\n  ДЕТЕРМИНИРОВАННОЕ СРАВНЕНИЕ, {res['n_pairs']} пар на руку, "
          f"задачи {res['tasks']}, состояния {res['states'][0]}.."
          f"{res['states'][-1]}")
    # ЭФФЕКТ БЕЗ ВОССТАНОВЛЕНИЙ И ПОТЕРЬ ОБМАНЧИВ: +6.67 пп из трёх чистых
    # исходов и из тринадцати против десяти — разные утверждения
    print(f"    {'рука':<20}{'успех':>9}"
          f"{'к fast12':>11}{'в/п':>9}{'p':>7}"
          f"{'к coarse24':>12}{'в/п':>9}{'p':>7}")
    base = succ.get("fast12")
    c24 = succ.get("coarse24")

    def col(s_, ref):
        if not ref:
            return f"{'—':>11}{'—':>9}{'—':>7}"
        c = compare(s_, ref)
        return (f"{100 * c['effect']:>+10.2f}"
                f"{c['recovered']:>5}/{c['lost']:<3}"
                f"{c['p_one_sided']:>7.3f}")
    for arm in res["arms"]:
        s = succ[arm]
        print(f"    {arm:<20}{100 * sum(s.values()) / len(s):>8.2f}%"
              + col(s, base) + col(s, c24))
    print(f"\n  ГЛАВНОЕ СРАВНЕНИЕ, попарно по головам:")
    for head, pair in res["head_pairs"].items():
        t, d = pair["t"], pair["d1"]
        if t not in succ or d not in succ:
            continue
        c = compare(succ[t], succ[d])
        vd, why = verdict(c["effect"] * 100, c["n"])
        print(f"    {head}: T {100 * c['success_a']:.2f}% против D1 "
              f"{100 * c['success_b']:.2f}%  ->  {100 * c['effect']:+.2f} пп "
              f"(восст {c['recovered']}, потер {c['lost']}, "
              f"p={c['p_one_sided']:.3f})")
        print(f"        решение: {vd} — {why}")
    print("\n    Порог инженерный, а не статистический: 45 пар различают "
          "только очень крупные\n    эффекты. Значимость — предмет "
          "зарегистрированного эксперимента.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cells", default="data/k13c/cells")
    ap.add_argument("--tasks", default="3,6,8")
    ap.add_argument("--states", default="30,35,40")
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--heads", default="s0,s1")
    ap.add_argument("--d1-cells", action="append", default=[],
                    help="маска готовых ячеек D1 и голова через '=', "
                         "например 'data/k12j/s0_*/eval_d1_det/*.json=s0'")
    ap.add_argument("--out", default="data/k13c/summary.json")
    ap.add_argument("--final", action="store_true",
                    help="итоговый режим: требовать ровно --expect-cells "
                         "ячеек, все руки полностью, тождество во всех "
                         "ячейках HiCoRA-T")
    ap.add_argument("--expect-cells", type=int, default=54)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    import glob
    cells = []
    for p in sorted(glob.glob(os.path.join(a.cells, "*.json"))):
        c = json.load(open(p))
        c["_path"] = p
        cells.append(c)
    # ГОТОВЫЕ ЯЧЕЙКИ D1 ПЕРЕИСПОЛЬЗУЮТСЯ, А НЕ ПЕРЕСЧИТЫВАЮТСЯ. В K-12j рука
    # называлась `baseline` (детерминированная D1 при sigma=0) — это та же
    # политика при тех же условиях, и условия сверяются ниже наравне со всеми.
    # Файлы НЕ переписываются: имя руки подставляется при чтении.
    for spec in a.d1_cells:
        if "=" not in spec:
            raise SystemExit(f"--d1-cells ждёт вид 'маска=голова', дано {spec}")
        pat, head = spec.rsplit("=", 1)
        found = sorted(glob.glob(pat))
        if not found:
            raise SystemExit(f"по маске {pat} нет ячеек")
        for p in found:
            c = json.load(open(p))
            if c.get("arm") != "baseline" or float(c.get("sigma", -1)) != 0.0:
                raise SystemExit(
                    f"{p}: рука {c.get('arm')} при sigma {c.get('sigma')} — "
                    f"это не детерминированная D1")
            c["arm"], c["head"], c["_path"] = "hicora_d1_det", head, p
            cells.append(c)
    if not cells:
        raise SystemExit(f"в {a.cells} нет ячеек")

    heads = [h for h in a.heads.split(",") if h.strip()]
    by_arm, meta_by_arm = {}, {}
    for c in cells:
        arm = c["arm"]
        key = arm if arm in SHARED_ARMS else f"{arm}:{c['head']}"
        by_arm.setdefault(key, []).extend(c["episodes"])
        m = {k: c.get(k) for k in META_KEYS}
        m["_path"] = c.get("_path")
        meta_by_arm.setdefault(key, []).append(m)
    # ОТСУТСТВУЮЩЕЕ ПОЛЕ ПРОХОДИТ ЛЮБУЮ ПРОВЕРКУ НА РАВЕНСТВО. Поэтому его
    # отсутствие — отказ, а не молчаливое согласие: ячейка, снятая до того,
    # как условие стали записывать, не сравнима с остальными, и разница в
    # условиях выглядела бы как разница рук.
    for c in cells:
        miss = [f for f in FIELDS_REQUIRED if c.get(f) is None]
        if miss:
            raise SystemExit(
                f"{c.get('_path')}: нет полей {miss}. Их отсутствие нельзя "
                f"считать совпадением с остальными руками")
    offs = check_pos_offsets(cells)
    print(f"  смещение позиций: {len(offs)} задач, по одному значению на "
          f"каждую у всех рук")
    # ОДИНАКОВЫЕ УСЛОВИЯ У ВСЕХ РУК: иначе разность рук смешана с разницей
    # горизонта, предела шагов или чекпойнта
    for key, ms in meta_by_arm.items():
        for fld in FIELDS_SAME + FIELDS_SAME_ARM:
            vals = {str(m.get(fld)) for m in ms}
            if len(vals) > 1:
                raise SystemExit(f"{key}: разные {fld} в ячейках {vals}")
    allf = [m for ms in meta_by_arm.values() for m in ms]
    for fld in FIELDS_SAME:
        vals = {str(m.get(fld)) for m in allf}
        if len(vals) > 1:
            raise SystemExit(f"руки считаны при разных {fld}: {vals}")
    # ТОЧНОСТЬ ИСПОЛНЕНИЯ — ЧАСТЬ СРАВНЕНИЯ, А НЕ ДЕТАЛЬ. Ствол под autocast и
    # голова в fp32; иначе сравнивались бы архитектуры вместе с разной
    # точностью. Равенство режимов уже проверено выше через FIELDS_SAME —
    # здесь закрепляется, что режим именно тот, а не просто общий.
    modes = {str(m.get("precision_mode")) for m in allf}
    if modes != {"trunk_autocast_head_fp32"}:
        raise SystemExit(
            f"руки считаны в разных режимах точности {modes}: ожидался "
            f"единый trunk_autocast_head_fp32. Ячейки без этого поля сняты до "
            f"разделения точности и непригодны для сравнения")
    # ТОЖДЕСТВО — УСЛОВИЕ ПРИГОДНОСТИ, А НЕ ЗАМЕЧАНИЕ
    # ТОЖДЕСТВО ТРЕБУЕТСЯ ОТ КАЖДОЙ ЯЧЕЙКИ T, А НЕ ОТ ХОТЯ БЫ ОДНОЙ. Проверка
    # снимается в самой ячейке на её первом батче; ячейка без неё исполнялась
    # неизвестно чем, и одной удачной проверки в соседней ячейке это не
    # заменяет.
    t_cells = [c for c in cells if c.get("arm") == "hicora_t_d1_det"]
    no_id = [c["_path"] for c in t_cells if not c.get("identity")]
    if no_id:
        raise SystemExit(
            f"{len(no_id)} ячеек HiCoRA-T без проверки тождества: "
            f"{no_id[:5]}. Тождество проверяется в каждой ячейке")
    bad_id = [c["_path"] for c in t_cells if not c["identity"].get("ok")]
    if bad_id:
        raise SystemExit(f"тождество не выполнено в {bad_id[:5]}")
    # МЕТКА ГОЛОВЫ ПРОТИВ СИДА В ЧЕКПОЙНТЕ
    for c in cells:
        if not c.get("head"):
            continue
        sd = c.get("head_seed")
        if sd is not None and f"s{int(sd)}" != c["head"]:
            raise SystemExit(f"{c['_path']}: метка {c['head']}, а сид "
                             f"чекпойнта {sd}")

    tasks = [int(x) for x in a.tasks.split(",")]
    starts = [int(x) for x in a.states.split(",")]
    want = len(tasks) * len(starts) * a.n_envs
    if a.final:
        check_final_composition(
            cells, meta_by_arm, heads, a.expect_cells)
    succ, keys = pair_table(by_arm, want, final=a.final)
    want_keys = {(t, s0) for t in tasks for st in starts
                 for s0 in range(st, st + a.n_envs)}
    if set(keys) != want_keys:
        miss = sorted(want_keys - set(keys))[:6]
        extra = sorted(set(keys) - want_keys)[:6]
        raise SystemExit(
            f"фактический набор не равен запрошенному: не хватает {miss} "
            f"({len(want_keys - set(keys))}), лишних {extra} "
            f"({len(set(keys) - want_keys)})")

    # РУКИ БЕРУТСЯ ИЗ ПОСЧИТАННОГО, а не из прочитанного: исключённая
    # недосчитанная рука в by_arm остаётся, и отчёт спотыкался бы о неё
    arms_present = list(succ)
    res = dict(arms=sorted(arms_present), success=succ, n_pairs=len(keys),
               tasks=tasks, states=sorted({k[1] for k in keys}),
               head_pairs={h: dict(t=f"hicora_t_d1_det:{h}",
                                   d1=f"hicora_d1_det:{h}") for h in heads},
               identity=[m.get("identity") for m in allf
                         if m.get("identity")],
               cells=[c["_path"] for c in cells])
    report(res)

    ident = [i for i in res["identity"] if i]
    if ident:
        ok = all(i.get("ok") for i in ident)
        print(f"\n  тождество HiCoRA-T с fast12 при обнулённой голове: "
              f"{'подтверждено' if ok else 'НЕ ПОДТВЕРЖДЕНО'} "
              f"({len(ident)} проверок)")
    else:
        print("\n  ВНИМАНИЕ: ни одна ячейка не несёт проверки тождества")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = f"{a.out}.tmp.{os.getpid()}"
    json.dump({k: v for k, v in res.items() if k != "success"},
              open(tmp, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, a.out)
    print(f"\n  сохранено: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
