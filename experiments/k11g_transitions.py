"""K-11g: разбор переходов исхода при шуме. Воспроизводимо, а не командой.

ВОПРОС. Существуют ли провалы детерминированной головы, которые ИСПРАВЛЯЮТСЯ
локальной поправкой? Если да, у head-only RL есть за что цепляться. Если шум
только теряет успехи, дело не в численном режиме обновления, и следующим шагом
будет LoRA последних слоёв.

ЧТО СЧИТАЕТСЯ. По парам (голова, начальное состояние) сравниваются исходы при
sigma=0 и при заданной sigma: восстановлено (провал -> успех), потеряно
(успех -> провал), успех сохранён, провал сохранён. Пары сопоставляются по
`pair_key`, и СОВПАДЕНИЕ `init_hash_full` ПРОВЕРЯЕТСЯ: номер состояния сам по
себе не доказывает, что состояние то же.

ЭТО ОПИСАТЕЛЬНАЯ СТАТИСТИКА, А НЕ СТАТИСТИЧЕСКИЙ ВЫВОД. Две головы
проверялись на ОБЩИХ состояниях и ОБЩЕМ потоке шума, эпизоды сгруппированы по
задачам — независимых испытаний здесь нет, и биномиальный интервал на
объединённой доле был бы неправомерен. Доли печатаются как доли; интервалы
печатаются с пометкой «описательно» и только по отдельной голове.

ЧЕГО ИЗ РЕЗУЛЬТАТА НЕ СЛЕДУЕТ. Что детерминированная голова «лишь немного
промахнулась»: шум применялся к КАЖДОМУ вызову политики, и восстановление
могло быть многошаговым. Корректная формулировка — «для N комбинаций
голова x исходно провальное состояние конкретная случайная траектория при
данной sigma оказалась успешной», то есть в head-only классе существуют
успешные стохастические траектории из части провальных состояний.

ПОБОЧНЫЙ, НО НУЖНЫЙ РЕЗУЛЬТАТ. Печатается и сохраняется СПИСОК состояний, где
исход переключился. Это заготовка failure bank для RL: при успехе около 90%
равномерная выборка состояний даёт градиент почти только от сохранённых
успехов.

Запуск:
    python3 experiments/k11g_transitions.py --selftest
    python3 experiments/k11g_transitions.py --cells data/k11g/cells \\
        --sigma 0.10 --out data/k11g/analysis/transitions.json
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

F2S, S2F, S2S, F2F = "f2s", "s2f", "s2s", "f2f"


def cp_interval(k, n, alpha=0.05):
    """Интервал Клоппера-Пирсона. ОПИСАТЕЛЬНО: эпизоды не независимы."""
    if n == 0:
        return (0.0, 1.0)

    def cdf(p, kk):
        if kk < 0:
            return 0.0
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i)
                   for i in range(kk + 1))

    lo, hi = alpha / 2, 1 - alpha / 2
    if k == 0:
        l = 0.0
    else:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            a, b = (m, b) if cdf(m, k - 1) > hi else (a, m)
        l = (a + b) / 2
    if k == n:
        h = 1.0
    else:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            a, b = (a, m) if cdf(m, k) < lo else (m, b)
        h = (a + b) / 2
    return (l, h)


def classify(det, sto):
    """Тип перехода по паре исходов."""
    if not det and sto:
        return F2S
    if det and not sto:
        return S2F
    return S2S if det else F2F


def transitions(det, sto):
    """Переходы по (голова, задача). Вход — словари (голова, pair_key) -> строка.

    Строка эпизода обязана содержать `success`, `init_hash_full` и `task_id`.
    Несовпадение хеша — ОТКАЗ: иначе «то же состояние» остаётся подписью.
    """
    bad, per, lists = [], defaultdict(lambda: defaultdict(int)), defaultdict(list)
    pairs = 0
    for key, s in sorted(sto.items()):
        d = det.get(key)
        if d is None:
            bad.append(f"{key}: нет детерминированной опоры")
            continue
        if d["init_hash_full"] != s["init_hash_full"]:
            bad.append(f"{key}: init_hash_full различается, состояние НЕ то же")
            continue
        if d["task_id"] != s["task_id"]:
            bad.append(f"{key}: разные задачи {d['task_id']} и {s['task_id']}")
            continue
        t = classify(bool(d["success"]), bool(s["success"]))
        per[(key[0], int(s["task_id"]))][t] += 1
        if t in (F2S, S2F):
            lists[(key[0], t)].append(key[1])
        pairs += 1
    miss = sorted(set(det) - set(sto))
    if miss:
        bad.append(f"{len(miss)} пар есть только у детерминированной руки: "
                   f"{miss[:3]}")
    if bad:
        raise SystemExit("ПАРЫ НЕ СОПОСТАВИМЫ:\n    " + "\n    ".join(bad[:8])
                         + ("\n    ..." if len(bad) > 8 else ""))
    if not pairs:
        raise SystemExit("ни одной пары: нечего сравнивать")
    return dict(per_head_task={f"{h}|{t}": dict(v) for (h, t), v in per.items()},
                raw=per, lists={f"{h}|{t}": sorted(v)
                                for (h, t), v in lists.items()},
                pairs=pairs)


def totals(res):
    """Итоги по голове и доли. Доли — описательные."""
    out = {}
    by_head = defaultdict(lambda: defaultdict(int))
    for (h, _t), v in res["raw"].items():
        for k, n in v.items():
            by_head[h][k] += n
    for h, v in sorted(by_head.items()):
        f2s, s2f = v[F2S], v[S2F]
        s2s, f2f = v[S2S], v[F2F]
        fails, succ = f2s + f2f, s2s + s2f
        out[h] = dict(f2s=f2s, s2f=s2f, s2s=s2s, f2f=f2f,
                      failures=fails, successes=succ,
                      recovered_frac=(f2s / fails) if fails else None,
                      recovered_ci=cp_interval(f2s, fails) if fails else None,
                      lost_frac=(s2f / succ) if succ else None,
                      lost_ci=cp_interval(s2f, succ) if succ else None,
                      net=f2s - s2f)
    return out


def tasks_with_transitions(res, heads=("s0", "s1")):
    """Где исход переключался. Важно: считать ПО КАЖДОЙ голове отдельно."""
    per_head = {h: sorted({t for (hh, t), v in res["raw"].items()
                           if hh == h and (v[F2S] or v[S2F])})
                for h in heads}
    all_t = sorted({t for (_h, t) in res["raw"]})
    any_ = sorted(set().union(*per_head.values())) if per_head else []
    return dict(per_head=per_head, none_in_any=[t for t in all_t
                                                if t not in any_],
                any=any_, all_tasks=all_t)


def selftest():
    # --- классификация ------------------------------------------------------
    assert classify(False, True) == F2S
    assert classify(True, False) == S2F
    assert classify(True, True) == S2S
    assert classify(False, False) == F2F

    def ep(t, h, ok):
        return dict(task_id=t, init_hash_full=h, success=ok)

    det = {("s0", "10|0|40"): ep(0, "H0", False),
           ("s0", "10|0|41"): ep(0, "H1", True),
           ("s0", "10|1|40"): ep(1, "H2", True),
           ("s1", "10|0|40"): ep(0, "H0", False)}
    sto = {("s0", "10|0|40"): ep(0, "H0", True),     # восстановлено
           ("s0", "10|0|41"): ep(0, "H1", False),    # потеряно
           ("s0", "10|1|40"): ep(1, "H2", True),     # успех сохранён
           ("s1", "10|0|40"): ep(0, "H0", False)}    # провал сохранён
    r = transitions(det, sto)
    assert r["pairs"] == 4
    assert r["per_head_task"]["s0|0"] == {F2S: 1, S2F: 1}
    assert r["per_head_task"]["s0|1"] == {S2S: 1}
    assert r["per_head_task"]["s1|0"] == {F2F: 1}
    assert r["lists"][f"s0|{F2S}"] == ["10|0|40"]
    assert r["lists"][f"s0|{S2F}"] == ["10|0|41"]

    t = totals(r)
    assert t["s0"] == dict(f2s=1, s2f=1, s2s=1, f2f=0, failures=1,
                           successes=2, recovered_frac=1.0,
                           recovered_ci=cp_interval(1, 1), lost_frac=0.5,
                           lost_ci=cp_interval(1, 2), net=0), t["s0"]
    assert t["s1"]["recovered_frac"] == 0.0 and t["s1"]["failures"] == 1

    # ЗАДАЧИ СЧИТАЮТСЯ ПО КАЖДОЙ ГОЛОВЕ ОТДЕЛЬНО. Прежний одноразовый разбор
    # свёл их в один список и дал неверное «сигнал только в 0, 6, 8».
    tw = tasks_with_transitions(r)
    assert tw["per_head"] == {"s0": [0], "s1": []}, tw["per_head"]
    assert tw["none_in_any"] == [1], tw["none_in_any"]

    # --- отказы -------------------------------------------------------------
    bad_hash = dict(sto)
    bad_hash[("s0", "10|0|40")] = ep(0, "ДРУГОЙ", True)
    try:
        transitions(det, bad_hash)
    except SystemExit as e:
        assert "init_hash_full различается" in str(e), str(e)
    else:
        raise AssertionError("чужое состояние принято")
    bad_task = dict(sto)
    bad_task[("s0", "10|1|40")] = ep(7, "H2", True)
    try:
        transitions(det, bad_task)
    except SystemExit as e:
        assert "разные задачи" in str(e), str(e)
    else:
        raise AssertionError("чужая задача принята")
    try:
        transitions({k: v for k, v in det.items() if k != ("s1", "10|0|40")},
                    sto)
    except SystemExit as e:
        assert "нет детерминированной опоры" in str(e), str(e)
    else:
        raise AssertionError("пара без опоры принята")
    try:
        transitions(det, {k: v for k, v in sto.items()
                          if k != ("s1", "10|0|40")})
    except SystemExit as e:
        assert "только у детерминированной" in str(e), str(e)
    else:
        raise AssertionError("лишняя опора принята")
    try:
        transitions({}, {})
    except SystemExit:
        pass
    else:
        raise AssertionError("пустой вход принят")

    # --- интервал -----------------------------------------------------------
    l, h = cp_interval(4, 6)
    assert abs(l - 0.2224) < 1e-3 and abs(h - 0.9567) < 1e-3, (l, h)
    assert cp_interval(0, 5)[0] == 0.0 and cp_interval(5, 5)[1] == 1.0

    print("самопроверка k11g_transitions пройдена: переходы классифицируются "
          "по четырём типам,\n  пары сверяются по init_hash_full и задаче, "
          "отсутствие опоры и лишняя опора —\n  отказ, задачи считаются ПО "
          "КАЖДОЙ ГОЛОВЕ отдельно, интервал Клоппера-Пирсона\n  совпадает с "
          "табличным")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cells", default="data/k11g/cells")
    ap.add_argument("--proto", default="data/k11g/protocol.json")
    ap.add_argument("--sigma", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    import k11g_protocol as kp
    proto = kp.load_json(a.proto, "протокол") if os.path.exists(a.proto) \
        else None
    det, sto = {}, {}
    files = sorted(glob.glob(os.path.join(a.cells, "*.json")))
    if not files:
        raise SystemExit(f"нет ячеек в {a.cells}")
    seen = set()
    for f in files:
        c = kp.load_json(f, "ячейка") if proto else json.load(open(f))
        sg = float(c["sigma"])
        if sg not in (0.0, a.sigma):
            continue
        if proto is not None:
            kp.check_cell(c, proto, c["head"], sg, c["task_id"])
        tgt = det if sg == 0.0 else sto
        for e in c["episodes"]:
            k = (c["head"], e["pair_key"])
            if k in tgt:
                raise SystemExit(f"дубль пары {k} при sigma={sg}")
            tgt[k] = dict(task_id=c["task_id"],
                          init_hash_full=e["init_hash_full"],
                          success=e["success"])
        seen.add((c["head"], sg, c["task_id"]))
    print(f"  ячеек прочитано: {len(seen)}; пар при sigma=0 {len(det)}, "
          f"при sigma={a.sigma} {len(sto)}"
          + ("; каждая сверена с протоколом" if proto else
             "; ПРОТОКОЛА НЕТ, сверка пропущена"))

    res = transitions(det, sto)
    tot = totals(res)
    tw = tasks_with_transitions(res, heads=tuple(sorted(
        {h for (h, _t) in res["raw"]})))

    print(f"\n  ПЕРЕХОДЫ по (голова, задача), sigma={a.sigma} против sigma=0")
    print(f"    {'гол':>5}{'зад':>5}{'f->s':>6}{'s->f':>6}{'s->s':>6}"
          f"{'f->f':>6}")
    for (h, t) in sorted(res["raw"]):
        v = res["raw"][(h, t)]
        print(f"    {h:>5}{t:>5}{v[F2S]:>6}{v[S2F]:>6}{v[S2S]:>6}{v[F2F]:>6}")

    print("\n  ИТОГИ по голове (доли ОПИСАТЕЛЬНЫЕ, не статистический вывод)")
    for h, v in sorted(tot.items()):
        rl, rh = v["recovered_ci"] or (0, 1)
        ll, lh = v["lost_ci"] or (0, 1)
        print(f"    {h}: восстановлено {v['f2s']}/{v['failures']}"
              f" = {100 * (v['recovered_frac'] or 0):.0f}% "
              f"[{100 * rl:.0f}%, {100 * rh:.0f}%], потеряно "
              f"{v['s2f']}/{v['successes']} = "
              f"{100 * (v['lost_frac'] or 0):.1f}% "
              f"[{100 * ll:.1f}%, {100 * lh:.1f}%], чистое {v['net']:+d}")

    print("\n  ЗАДАЧИ С ПЕРЕХОДАМИ (по каждой голове отдельно)")
    for h, ts in sorted(tw["per_head"].items()):
        print(f"    {h}: {ts}")
    print(f"    без переходов ни у одной головы: {tw['none_in_any']}")

    print("\n  СОСТОЯНИЯ ДЛЯ FAILURE BANK (исход переключился)")
    for k, v in sorted(res["lists"].items()):
        print(f"    {k}: {len(v)} шт")
        for pk in v:
            print(f"      {pk}")

    print("\n  ЧТО ЭТО ЗНАЧИТ. Для "
          f"{sum(v['f2s'] for v in tot.values())} комбинаций «голова x "
          f"исходно провальное состояние»\n  конкретная случайная траектория "
          f"при sigma={a.sigma} оказалась успешной. Значит в\n  head-only "
          f"классе СУЩЕСТВУЮТ успешные стохастические траектории из части "
          f"провальных\n  состояний. Что детерминированная голова «лишь "
          f"немного промахнулась», отсюда НЕ\n  следует: шум применялся к "
          f"каждому вызову, и восстановление могло быть многошаговым.")
    print("  Интервалы печатаются по ОТДЕЛЬНОЙ голове и описательно: две "
          "головы шли на общих\n  состояниях и общем потоке шума, эпизоды "
          "сгруппированы по задачам, независимых\n  испытаний здесь нет.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        json.dump(dict(sigma=a.sigma, pairs=res["pairs"],
                       per_head_task=res["per_head_task"],
                       totals=tot, tasks=tw, lists=res["lists"],
                       cells_read=len(seen), protocol_checked=bool(proto)),
                  open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {a.out}")


if __name__ == "__main__":
    main()
