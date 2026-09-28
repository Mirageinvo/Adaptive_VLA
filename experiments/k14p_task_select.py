#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14p: отбор задач по правилу §53.4 и определение банков состояний.

ЗАЧЕМ МЕХАНИЧЕСКИ, А НЕ НА ГЛАЗ. Смоук на задаче 3 дал 100% успеха у всех
трёх рук: спасать нечего, и по пределу выполнимости §37
(discord <= 2*p_fail - delta) такие задачи разбавляют p_fail и делают
невыполнимым ВЕСЬ набор. Отбор, сделанный после просмотра результатов
сравнения, был бы подгонкой, поэтому правило записано в §53.4 до прогона, а
здесь применяется кодом.

ПРАВИЛО (§53.4): отбор по руке q0, состояния 0-9 каждой задачи; задача
включается, если провалов НЕ МЕНЕЕ 2 из 10. Эти состояния израсходованы.

БАНКИ: dev 10-39 (30 состояний на задачу), final 40-49 (10 состояний), одно
открытие. Execution seeds повторов 101-104, внутри повтора один для всех рук,
режим сида раскатки `fixed`.

ЧТО ЭТОТ СКРИПТ НЕ ДЕЛАЕТ. Не считает rescue/harm и не сравнивает руки: это
результат §53. Здесь только отбор и определение банков.
"""
import argparse
import json
import os
import sys

MIN_FAIL = 2            # из N_SEL состояний; §53.4
N_SEL = 10              # состояния 0..9 на отбор
DEV_RANGE = (10, 40)    # [10, 40) — 30 состояний
FINAL_RANGE = (40, 50)  # [40, 50) — 10 состояний
REPEAT_SEEDS = (101, 102, 103, 104)
BLOCK = 5               # сред за вызов; ограничение памяти хоста, §53.4
SEED_MODE = "fixed"


def load_runs(paths, *, expect_policy=None):
    """Артефакты k9h одной руки. Состояния не должны пересекаться."""
    runs, seen = [], {}
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"нет {p}")
        d = json.load(open(p))
        for k in ("episodes", "task_id", "init_start", "n_envs", "policy",
                  "seed", "rollout_seed_mode", "arm_label"):
            if d.get(k) is None:
                raise SystemExit(f"{p}: нет поля {k}")
        if expect_policy and str(d["policy"]) != str(expect_policy):
            raise SystemExit(f"{p}: политика {d['policy']}, отбор идёт по "
                             f"{expect_policy}")
        if str(d["rollout_seed_mode"]) != SEED_MODE:
            raise SystemExit(
                f"{p}: режим сида {d['rollout_seed_mode']}, §53.4 требует "
                f"{SEED_MODE}: иначе один init_state_id получает разные сиды "
                f"в блоках с разным началом")
        t, i0, n = int(d["task_id"]), int(d["init_start"]), int(d["n_envs"])
        if len(d["episodes"]) != n:
            raise SystemExit(f"{p}: эпизодов {len(d['episodes'])} при "
                             f"n_envs {n}")
        for j in range(n):
            key = (t, i0 + j)
            if key in seen:
                raise SystemExit(
                    f"{p}: состояние {i0 + j} задачи {t} уже покрыто "
                    f"{seen[key]} — блоки пересекаются, и доля провалов "
                    f"считалась бы по одному состоянию дважды")
            seen[key] = p
        d["_path"] = p
        runs.append(d)
    return runs


def select(runs):
    """Доли провалов по задачам и решение по каждой. Чистая функция."""
    by = {}
    for d in runs:
        t = int(d["task_id"])
        b = by.setdefault(t, dict(states={}, seeds=set(), paths=[]))
        b["paths"].append(d["_path"])
        b["seeds"].add(int(d["seed"]))
        for j, e in enumerate(d["episodes"]):
            b["states"][int(d["init_start"]) + j] = bool(e["success"])
    out = {}
    for t, b in sorted(by.items()):
        st = b["states"]
        miss = [s for s in range(N_SEL) if s not in st]
        extra = sorted(s for s in st if s >= N_SEL)
        if miss:
            raise SystemExit(
                f"задача {t}: нет состояний {miss}; отбор считается по "
                f"ПОЛНЫМ 0..{N_SEL - 1}, иначе доля провалов не та")
        if extra:
            raise SystemExit(
                f"задача {t}: покрыты состояния {extra[:5]} вне диапазона "
                f"отбора — они принадлежат банкам и тратиться на отбор не "
                f"должны")
        if len(b["seeds"]) != 1:
            raise SystemExit(f"задача {t}: сиды {sorted(b['seeds'])}")
        n_fail = sum(1 for s in range(N_SEL) if not st[s])
        out[t] = dict(n_fail=n_fail, n=N_SEL,
                      p_fail=n_fail / float(N_SEL),
                      included=bool(n_fail >= MIN_FAIL),
                      seed=sorted(b["seeds"])[0], sources=sorted(b["paths"]))
    return out


def feasible(p_fail, discord, delta):
    """Предел §37: discord <= 2*p_fail - delta. None, если нет данных."""
    if discord is None:
        return None
    return bool(float(discord) <= 2.0 * float(p_fail) - float(delta))


def main():
    ap = argparse.ArgumentParser(description="Отбор задач и банки, §53.4")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--runs", nargs="*", default=[],
                    help="артефакты k9h руки q0 на состояниях 0..9")
    ap.add_argument("--expect-policy", default="fast")
    ap.add_argument("--delta", type=float, default=0.05,
                    help="целевой прирост успеха для проверки §37")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", default="reports/k14p/banks.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not a.runs:
        raise SystemExit("нужны --runs")
    if os.path.exists(a.out) and not a.overwrite:
        raise SystemExit(f"{a.out} уже существует: банки не переопределяются "
                         f"молча (--overwrite осознанно)")

    runs = load_runs(a.runs, expect_policy=a.expect_policy)
    res = select(runs)
    inc = sorted(t for t, v in res.items() if v["included"])
    n_all = sum(v["n_fail"] for v in res.values())
    p_all = n_all / float(N_SEL * len(res))
    p_inc = (sum(res[t]["n_fail"] for t in inc) / float(N_SEL * len(inc))
             if inc else 0.0)

    print(f"\n  задач {len(res)}, состояний на задачу {N_SEL}, порог "
          f"провалов {MIN_FAIL}")
    print(f"  {'задача':>7} {'провалов':>9} {'p_fail':>7}  решение")
    for t in sorted(res):
        v = res[t]
        print(f"  {t:7d} {v['n_fail']:9d} {v['p_fail']:7.2f}  "
              f"{'включена' if v['included'] else 'исключена'}")
    print(f"\n  p_fail по всем задачам:      {p_all:.3f}")
    print(f"  p_fail по отобранным:        {p_inc:.3f}"
          if inc else "  отобранных задач НЕТ")
    # ПРЕДЕЛ ВЫПОЛНИМОСТИ §37 ПЕЧАТАЕТСЯ КАК ГРАНИЦА, А НЕ КАК ВЕРДИКТ:
    # дискордантность ещё не измерена, она появится только в §53.
    for nm, pf in (("все", p_all), ("отобранные", p_inc)):
        if pf > 0:
            print(f"  §37 на наборе «{nm}»: при delta={a.delta:.02f} "
                  f"допустимая дискордантность не выше "
                  f"{2 * pf - a.delta:+.3f}")
    if not inc:
        print("\n  НИ ОДНА ЗАДАЧА НЕ ПРОШЛА ПОРОГ. Вторичный набор §53.1 "
              "пуст; основной результат по всем десяти задачам остаётся, но "
              "восстанавливать на них почти нечего")

    banks = dict(
        kind="k14p_banks", rule=dict(min_fail=MIN_FAIL, n_sel=N_SEL,
                                     expect_policy=a.expect_policy),
        per_task=res, included_tasks=inc, all_tasks=sorted(res),
        p_fail_all=p_all, p_fail_included=p_inc, delta=float(a.delta),
        selection_states=list(range(N_SEL)),
        dev=dict(states=list(range(*DEV_RANGE)), opens="сколько угодно"),
        final=dict(states=list(range(*FINAL_RANGE)), opens="ОДНО"),
        repeat_seeds=list(REPEAT_SEEDS), block=BLOCK,
        rollout_seed_mode=SEED_MODE,
        note=("состояния 0..9 израсходованы на отбор; вторичный результат "
              "§53 считается по included_tasks, основной — по all_tasks"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    t_ = a.out + f".tmp.{os.getpid()}"
    json.dump(banks, open(t_, "w"), ensure_ascii=False, indent=1)
    os.replace(t_, a.out)
    print(f"\n  банки зафиксированы: {a.out}")
    print(f"    dev   состояния {DEV_RANGE[0]}..{DEV_RANGE[1] - 1}")
    print(f"    final состояния {FINAL_RANGE[0]}..{FINAL_RANGE[1] - 1}, "
          f"одно открытие")
    print(f"    сиды повторов {list(REPEAT_SEEDS)}, блок {BLOCK}, режим "
          f"{SEED_MODE}")
    return 0


def _run(task, i0, succ, seed=100, policy="fast", **kw):
    d = dict(task_id=task, init_start=i0, n_envs=len(succ), policy=policy,
             seed=seed, rollout_seed_mode=SEED_MODE, arm_label="fast_s0",
             episodes=[dict(env_index=j, success=bool(s))
                       for j, s in enumerate(succ)])
    d.update(kw)
    return d


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        def w(d, nm):
            q = os.path.join(td, nm)
            json.dump(d, open(q, "w"))
            return q

        # задача 0: 2 провала -> включена; задача 1: 1 провал -> исключена
        ps = [w(_run(0, 0, [1, 0, 1, 1, 1]), "a.json"),
              w(_run(0, 5, [1, 1, 0, 1, 1]), "b.json"),
              w(_run(1, 0, [1, 1, 1, 1, 1]), "c.json"),
              w(_run(1, 5, [1, 1, 1, 0, 1]), "d.json")]
        r = select(load_runs(ps, expect_policy="fast"))
        assert r[0]["n_fail"] == 2 and r[0]["included"]
        assert r[1]["n_fail"] == 1 and not r[1]["included"]
        assert abs(r[0]["p_fail"] - 0.2) < 1e-12

        # ПЕРЕСЕЧЕНИЕ БЛОКОВ — отказ: иначе состояние учтётся дважды
        try:
            load_runs([ps[0], w(_run(0, 3, [1, 1, 1, 1, 1]), "e.json")])
        except SystemExit as e:
            assert "пересекаются" in str(e), e
        else:
            raise AssertionError("приняты пересекающиеся блоки")
        # НЕПОЛНОЕ ПОКРЫТИЕ — отказ
        try:
            select(load_runs([ps[0]]))
        except SystemExit as e:
            assert "нет состояний" in str(e), e
        else:
            raise AssertionError("принято неполное покрытие 0..9")
        # СОСТОЯНИЯ ИЗ БАНКОВ на отбор не годятся. Покрытие 0..9 при этом
        # полное — иначе сработала бы проверка неполноты, и тест проверял бы
        # не то, что заявлено.
        try:
            select(load_runs([ps[0], ps[1],
                              w(_run(0, 10, [1] * 5), "f.json")]))
        except SystemExit as e:
            assert "вне диапазона" in str(e), e
        else:
            raise AssertionError("приняты состояния банков")
        # ЧУЖОЙ РЕЖИМ СИДА — отказ
        try:
            load_runs([w(_run(0, 0, [1] * 5, rollout_seed_mode="block"),
                         "g.json")])
        except SystemExit as e:
            assert "режим сида" in str(e), e
        else:
            raise AssertionError("принят режим block")
        # ЧУЖАЯ ПОЛИТИКА — отказ
        try:
            load_runs([w(_run(0, 0, [1] * 5, policy="depthrvq"), "h.json")],
                      expect_policy="fast")
        except SystemExit as e:
            assert "политика" in str(e), e
        else:
            raise AssertionError("принята чужая политика")
        # РАЗНЫЕ СИДЫ внутри задачи — отказ
        try:
            select(load_runs([ps[0], w(_run(0, 5, [1] * 5, seed=7), "i.json")]))
        except SystemExit as e:
            assert "сиды" in str(e), e
        else:
            raise AssertionError("приняты разные сиды в одной задаче")

    # предел §37
    assert feasible(0.10, 0.20, 0.05) is False
    assert feasible(0.25, 0.40, 0.05) is True
    assert feasible(0.10, None, 0.05) is None
    print("самопроверка k14p_task_select пройдена")


if __name__ == "__main__":
    sys.exit(main())
