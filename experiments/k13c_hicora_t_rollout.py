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


def pair_table(by_arm, want_states, log=print):
    """Парные разности по общим (задача, состояние).

    НЕДОСЧИТАННАЯ РУКА ИСКЛЮЧАЕТСЯ С ОБЪЯСНЕНИЕМ, а не роняет отчёт: пока
    одна рука считается, остальные сравнивать можно и нужно. А вот РАСХОЖДЕНИЕ
    состояний при полном покрытии — отказ: это разные наборы, и разность по
    ним ничего не измеряет.
    """
    sets = {arm: {(int(e["task_id"]), int(e["state_id"])) for e in eps}
            for arm, eps in by_arm.items()}
    full = {arm: k for arm, k in sets.items()
            if not want_states or len(k) >= want_states}
    partial = {arm: len(k) for arm, k in sets.items() if arm not in full}
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
    by_arm = {a: by_arm[a] for a in full}
    succ = {}
    for arm, eps in by_arm.items():
        d = {(int(e["task_id"]), int(e["state_id"])): bool(e["success"])
             for e in eps}
        succ[arm] = {k: d[k] for k in sorted(keys)}
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
    """Инженерный порог из плана. Это go/no-go, а не вывод о значимости."""
    lost = -round(delta_pp * n / 100.0)
    if lost <= 3:
        return "continue", f"проигрыш {max(lost, 0)} исходов из {n} <= 3"
    if lost <= 5:
        return "check", f"проигрыш {lost} исходов из {n} в диапазоне 4..5"
    return "fix_head", f"проигрыш {lost} исходов из {n} > 5"


def selftest():
    # --- недосчитанная рука исключается, а не роняет отчёт ----------------
    mk9 = lambda ok: [dict(task_id=3, state_id=i, success=(i in ok))
                      for i in range(30, 35)]
    part = {"fast12": mk9({30, 31}), "coarse24": [
        dict(task_id=3, state_id=30, success=True)]}
    msgs = []
    sp, kp = pair_table(part, 5, log=msgs.append)
    assert set(sp) == {"fast12"}, sp
    assert any("coarse24 исключена" in m for m in msgs), msgs
    # но РАСХОЖДЕНИЕ при полном покрытии — по-прежнему отказ
    diff = {"fast12": mk9({30}), "coarse24": [
        dict(task_id=3, state_id=i, success=True) for i in range(40, 45)]}
    try:
        pair_table(diff, 5, log=lambda *_: None)
    except SystemExit as e:
        assert "РАЗНЫХ состояниях" in str(e), e
    else:
        raise AssertionError("расхождение при полном покрытии принято")

    # --- парность и отказ на расхождении ----------------------------------
    mk = lambda ok: [dict(task_id=3, state_id=i, success=(i in ok))
                     for i in range(30, 35)]
    by = {"fast12": mk({30, 31, 32}), "hicora_t_d1_det": mk({30, 31, 32, 33})}
    succ, keys = pair_table(by, 5)
    assert len(keys) == 5
    c = compare(succ["hicora_t_d1_det"], succ["fast12"])
    assert c["recovered"] == 1 and c["lost"] == 0 and c["n"] == 5, c
    assert abs(c["effect"] - 0.2) < 1e-12
    bad = dict(by)
    bad["coarse24"] = [dict(task_id=3, state_id=i, success=True)
                       for i in range(40, 45)]
    try:
        pair_table(bad, 5)
    except SystemExit as e:
        assert "на РАЗНЫХ состояниях" in str(e), e
    else:
        raise AssertionError("непарные руки приняты")

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
    print(f"    {'рука':<20}{'успех':>9}{'к fast12':>11}{'к coarse24':>12}")
    base = succ.get("fast12")
    c24 = succ.get("coarse24")
    for arm in res["arms"]:
        s = succ[arm]
        v1 = compare(s, base)["effect"] * 100 if base else None
        v2 = compare(s, c24)["effect"] * 100 if c24 else None
        print(f"    {arm:<20}{100 * sum(s.values()) / len(s):>8.2f}%"
              + (f"{v1:>+10.2f}" if v1 is not None else f"{'—':>11}")
              + (f"{v2:>+11.2f}" if v2 is not None else f"{'—':>12}"))
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
        meta_by_arm.setdefault(key, []).append(
            {k: c.get(k) for k in ("ckpt", "max_steps", "horizon",
                                   "rollout_seed_mode", "code_version",
                                   "head_sha1", "basis_sha1", "rho_sha1",
                                   "identity")})
    # ОДИНАКОВЫЕ УСЛОВИЯ У ВСЕХ РУК: иначе разность рук смешана с разницей
    # горизонта, предела шагов или чекпойнта
    for key, ms in meta_by_arm.items():
        for fld in ("ckpt", "max_steps", "horizon", "rollout_seed_mode"):
            vals = {str(m.get(fld)) for m in ms}
            if len(vals) > 1:
                raise SystemExit(f"{key}: разные {fld} в ячейках {vals}")
    allf = [m for ms in meta_by_arm.values() for m in ms]
    for fld in ("ckpt", "max_steps", "horizon", "rollout_seed_mode"):
        vals = {str(m.get(fld)) for m in allf}
        if len(vals) > 1:
            raise SystemExit(f"руки считаны при разных {fld}: {vals}")

    tasks = [int(x) for x in a.tasks.split(",")]
    starts = [int(x) for x in a.states.split(",")]
    want = len(tasks) * len(starts) * a.n_envs
    succ, keys = pair_table(by_arm, want)

    arms_present = [k for k in by_arm]
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
