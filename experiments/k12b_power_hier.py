"""K-12b: иерархический расчёт мощности. Задачи, D1-сиды, RL-сиды, правило гейта.

ЧТО ИСПРАВЛЯЕТ ОТНОСИТЕЛЬНО K-12a. Та симуляция предполагала одинаковые
`r_rec` и `r_loss` на всех задачах, разброс только у `r_rec`, отсутствие
разброса `r_loss` и ОДНУ обученную реплику. Для нашего случая это особенно
неудачно: K-11g показал сильную неоднородность по задачам и головам, а
ограничивает эксперимент именно `r_loss`. И четыре реплики `2 D1 x 2 RL` в
мощности не моделировались вовсе: при требовании «каждая проходит»
индивидуальная мощность 0.92-0.94 превращается в 0.72-0.78.

ОПОРА — ТА, ПРОТИВ КОТОРОЙ ИДЁТ ГЕЙТ. K-12a считала против `joint12` (89.5%),
тогда как зарегистрированное сравнение — обученная голова против СВОЕЙ
детерминированной D1: `hicora_s0` 92.0% и `hicora_s1` 90.25%. Потолок
улучшения равен доле провалов, поэтому опора решает всё: при 92% успеха
эффект +5 пп требует исправить больше провалов, чем есть, как только
дискордантность достигает 13.5%.

ДИСКОРДАНТНОСТЬ — НЕ ДАННОСТЬ, А ТО, ЧЕМ МЕТОД УПРАВЛЯЕТ. При 5% против
`hicora_s0` нужно восстанавливать 63% провалов при нулевых потерях — ровно
столько, сколько даёт случайный шум в §32. При 13.5% недостижимо ни при каких
долях. Значит эксперимент осуществим тогда и только тогда, когда обученная
политика меняет поведение почти исключительно там, где это помогает, и низкая
дискордантность есть ЦЕЛЬ, а не допущение.

ИЕРАРХИЯ РАЗБРОСА. Замысел задаёт средние `r_rec` и `r_loss` через калибровку
по (эффект, дискордантность). Дальше:
  уровень реплики (D1-сид x RL-сид) — свои `r_rec` и `r_loss`;
  уровень задачи внутри реплики — свои `r_rec` и `r_loss`.
Все разбросы разыгрываются бета-распределением с ТОЧНО заданным средним:
обрезанная нормаль поднимала бы среднее и завышала эффект.

ПРАВИЛО ГЕЙТА ЗАДАЁТСЯ ЯВНО, потому что от него зависит мощность:
  mean  — первичная величина есть среднее по четырём заранее фиксированным
          репликам; бутстрап по задачам, реплики фиксированы;
  mean_rep — то же, но реплики ТОЖЕ ресэмплируются: это цена обобщения с
          «эти четыре прогона» на «метод»;
  all   — каждая реплика обязана пройти сама.

Запуск:
    python3 experiments/k12b_power_hier.py --selftest
    python3 experiments/k12b_power_hier.py --cells 'data/k11g/cells/*.json' \\
        --out data/k12a/power_hier.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k12a_power import (ALPHA, TARGET_POWER, calibrate, cluster_lower,  # noqa
                        draw_r_rec, net_effect, r_loss_max)

RULES = ("mean", "mean_rep", "all")


# Сколько раз разброс пришлось урезать и до чего. Разброс ДОЛИ ограничен
# сверху самим средним: для доли в (0,1) дисперсия не может достигать
# m(1-m). Задавать sd абсолютным числом независимо от среднего нельзя — при
# r_loss=0.002 запрос sd=0.10 невозможен, и перебор падал посреди таблицы.
CLAMPS = {}


CV_MAX = 0.5      # предельный коэффициент вариации доли


def feasible_sd(mean, sd, margin=0.95, cv_max=CV_MAX):
    """Возможный разброс доли: ограничен И выполнимостью, И относительно.

    Два ограничения, и второе не формальность. Выполнимость: дисперсия доли не
    может достигать m(1-m). Относительное: при малом среднем абсолютный
    разброс бессмыслен — бета со средним 0.002 и sd 0.042 ВЫРОЖДАЕТСЯ, почти
    вся масса садится в нуль, и «разброс по задачам» превращается в «у
    большинства задач ноль, у одной много». Поэтому sd дополнительно
    ограничен долей самого среднего.
    """
    m = float(mean)
    if m <= 0.0 or m >= 1.0:
        return 0.0
    top = min(margin * (m * (1.0 - m)) ** 0.5, cv_max * m)
    return min(float(sd), top)


def draw(mean, sd, rng, tag=""):
    """Бета с точно заданным средним; разброс УРЕЗАЕТСЯ до возможного.

    Урезание не молчаливое: оно считается и печатается, потому что «разброс
    по задачам 0.10» при среднем 0.002 фактически означает 0.04, и читать
    таблицу, не зная этого, нельзя.
    """
    eff = feasible_sd(mean, sd)
    if sd > 0 and eff < float(sd) - 1e-12:
        k = (tag, round(float(mean), 5))
        CLAMPS[k] = (CLAMPS.get(k, (0, eff))[0] + 1, eff)
    return draw_r_rec(mean, eff, rng)


def sim_replica(rates, n_ep, r_rec, r_loss, task_sd, rng):
    """Одна реплика: по задаче свои доли, затем парные исходы.

    Возвращает список (средняя разность по задаче, число пар) и фактическую
    дискордантность — её надо видеть, потому что именно она ограничивает
    достижимость.
    """
    out, disc = [], []
    for p in rates:
        rr = draw(r_rec, task_sd, rng, "задача/r_rec")
        rl = draw(r_loss, task_sd, rng, "задача/r_loss")
        base = rng.random(n_ep) < p
        flip = np.where(base, rng.random(n_ep) < rl, rng.random(n_ep) < rr)
        new = np.where(flip, ~base, base)
        out.append((float((new.astype(float) - base.astype(float)).mean()),
                    n_ep))
        disc.append(float(flip.mean()))
    return out, float(np.mean(disc))


def sim_study(rates_by_head, n_ep, r_rec, r_loss, rep_sd, task_sd, rng,
              n_rl=2):
    """Все реплики: по каждой D1-голове и каждому RL-сиду."""
    reps, discs = [], []
    for _h, rates in sorted(rates_by_head.items()):
        for _s in range(n_rl):
            rr = draw(r_rec, rep_sd, rng, "реплика/r_rec")
            rl = draw(r_loss, rep_sd, rng, "реплика/r_loss")
            d, dc = sim_replica(rates, n_ep, rr, rl, task_sd, rng)
            reps.append(d)
            discs.append(dc)
    return reps, discs


def lower_mean(reps, n_boot, rng, resample_replicas=False, alpha=ALPHA):
    """Нижняя граница СРЕДНЕГО эффекта по репликам, бутстрап по задачам.

    `resample_replicas` добавляет ресэмплирование реплик: без него вывод
    относится к ЭТИМ четырём прогонам, с ним — к методу. Цена обобщения видна
    как разница двух границ, и замалчивать её нельзя.
    """
    k = len(reps[0])
    if k < 2:
        raise SystemExit("меньше двух задач: кластерный бутстрап невозможен")
    n_rep = len(reps)
    D = np.array([[x[0] for x in r] for r in reps], float)      # [rep, task]
    W = np.array([[x[1] for x in r] for r in reps], float)
    ti = rng.integers(0, k, size=(n_boot, k))
    if resample_replicas:
        ri = rng.integers(0, n_rep, size=(n_boot, n_rep))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        rows = ri[b] if resample_replicas else np.arange(n_rep)
        num = (D[np.ix_(rows, ti[b])] * W[np.ix_(rows, ti[b])]).sum()
        den = W[np.ix_(rows, ti[b])].sum()
        boots[b] = num / max(den, 1e-12)
    return float(np.percentile(boots, 100 * alpha))


def lower_all(reps, n_boot, rng, alpha=ALPHA):
    """Минимум из индивидуальных границ реплик: правило «каждая проходит»."""
    return min(cluster_lower(r, n_boot, rng, alpha) for r in reps)


def power_study(rates_by_head, n_ep, delta, discord, rule="mean",
                rep_sd=0.05, task_sd=0.10, n_sim=300, n_boot=300, seed=0,
                n_rl=2, alpha=ALPHA):
    """Мощность при заданном правиле гейта. Калибровка — по ХУДШЕЙ опоре.

    Калибровка делается по голове с НАИМЕНЬШЕЙ долей провалов: если для неё
    комбинация (эффект, дискордантность) недостижима, недостижим и общий
    результат, и признать это надо до прогона, а не после.
    """
    if rule not in RULES:
        raise SystemExit(f"правило {rule!r} не из {RULES}")
    p_fails = {h: 1.0 - float(np.mean(r)) for h, r in rates_by_head.items()}
    worst = min(p_fails, key=lambda h: p_fails[h])
    cal = calibrate(p_fails[worst], delta, discord)
    if cal is None:
        # ДВЕ РАЗНЫЕ ПРИЧИНЫ, и путать их нельзя: дискордантность по
        # определению не меньше модуля эффекта (иначе перевернувшихся пар не
        # хватит даже на сам сдвиг), а сверх того требуемая доля
        # восстановлений может превысить единицу.
        why = ("дискордантность меньше самого эффекта: "
               f"{100 * discord:.1f}% < {100 * delta:.1f} пп"
               if discord < abs(delta) else
               f"требуется восстановить больше провалов, чем есть: опора "
               f"{worst}, p_fail={100 * p_fails[worst]:.1f}%")
        return dict(power=None, reason=why, calibrated_on=worst,
                    p_fails=p_fails)
    rng = np.random.default_rng(seed)
    hit, lows, dcs = 0, [], []
    for _ in range(n_sim):
        reps, dd = sim_study(rates_by_head, n_ep, cal["r_rec"], cal["r_loss"],
                             rep_sd, task_sd, rng, n_rl)
        dcs += dd
        if rule == "all":
            lo = lower_all(reps, n_boot, rng, alpha)
        else:
            lo = lower_mean(reps, n_boot, rng,
                            resample_replicas=(rule == "mean_rep"),
                            alpha=alpha)
        lows.append(lo)
        hit += int(lo > 0.0)
    return dict(power=hit / n_sim, n_sim=n_sim, rule=rule,
                lower_median=float(np.median(lows)),
                discord_mean=float(np.mean(dcs)),
                r_rec=cal["r_rec"], r_loss=cal["r_loss"],
                calibrated_on=worst, p_fails=p_fails,
                n_replicas=len(rates_by_head) * n_rl)


def load_rates_k11g(pattern, sigma=0.0):
    """Успешность по задачам для КАЖДОЙ головы из ячеек K-11g при данной sigma.

    Именно это и есть опора гейта: детерминированная D1 каждой головы.
    """
    per = {}
    for f in sorted(glob.glob(pattern)):
        c = json.load(open(f))
        if abs(float(c["sigma"]) - sigma) > 1e-12:
            continue
        key = (c["head"], int(c["task_id"]))
        a, b = per.get(key, (0, 0))
        per[key] = (a + sum(bool(e["success"]) for e in c["episodes"]),
                    b + len(c["episodes"]))
    if not per:
        raise SystemExit(f"нет ячеек с sigma={sigma} по {pattern}")
    heads = sorted({h for h, _t in per})
    tasks = sorted({t for _h, t in per})
    bad = [(h, t) for h in heads for t in tasks if (h, t) not in per]
    if bad:
        raise SystemExit(f"неполный набор: нет ячеек {bad[:5]}")
    return {h: [per[(h, t)][0] / per[(h, t)][1] for t in tasks]
            for h in heads}, per


def selftest():
    rng = np.random.default_rng(0)
    rates = {"s0": [0.92] * 10, "s1": [0.90] * 10}

    # --- реплика: средние сохраняются --------------------------------------
    d, dc = sim_replica([0.9] * 200, 50, 0.5, 0.0, 0.0, rng)
    got = float(np.mean([x[0] for x in d]))
    assert abs(got - 0.1 * 0.5) < 0.01, got
    assert abs(dc - 0.1 * 0.5) < 0.01, dc

    # --- иерархия: число реплик и сохранение среднего ------------------------
    reps, dcs = sim_study(rates, 200, 0.5, 0.0, 0.05, 0.10, rng, n_rl=2)
    assert len(reps) == 4 and len(dcs) == 4
    m = float(np.mean([x[0] for r in reps for x in r]))
    # p_fail ~ 0.09 при успешности 0.92/0.90, r_rec = 0.5, потерь нет ->
    # эффект ~ 0.09 * 0.5 = 0.045. (В первой редакции теста я написал 0.085,
    # перепутав множитель; код был прав.)
    assert abs(m - 0.045) < 0.01, m

    # --- РАЗБРОС УРЕЗАЕТСЯ, А НЕ ПАДАЕТ ------------------------------------
    # Прежде при r_loss=0.002 и sd=0.10 перебор падал посреди таблицы.
    # при малом среднем связывает ОТНОСИТЕЛЬНОЕ ограничение
    assert abs(feasible_sd(0.002, 0.10) - 0.5 * 0.002) < 1e-12
    # при среднем 0.5 запрошенные 0.10 проходят целиком
    assert abs(feasible_sd(0.5, 0.10) - 0.10) < 1e-12
    # при среднем 0.1 относительный предел 0.05
    assert abs(feasible_sd(0.1, 0.10) - 0.05) < 1e-12
    assert feasible_sd(0.0, 0.10) == 0.0
    # распределение НЕ вырождено: среднее сохранено, масса не в нуле
    g2 = np.random.default_rng(5)
    dd = [draw(0.002, 0.10, g2, "t") for _ in range(5000)]
    assert abs(float(np.mean(dd)) - 0.002) < 2e-4, float(np.mean(dd))
    assert min(dd) > 0.0 and float(np.median(dd)) > 1e-4
    CLAMPS.clear()
    v = draw(0.002, 0.10, rng, "тест")
    assert 0.0 < v < 1.0
    assert CLAMPS and list(CLAMPS.values())[0][0] == 1
    CLAMPS.clear()
    draw(0.5, 0.01, rng, "тест")
    assert not CLAMPS, "урезание там, где не нужно"
    # И весь перебор с крошечным r_loss доходит до конца
    out = power_study({"a": [0.9] * 6}, 20, 0.02, 0.025, "mean", 0.10, 0.10,
                      n_sim=20, n_boot=20, seed=9)
    assert out["power"] is not None, out

    # --- ДВЕ ПРИЧИНЫ ОТКАЗА РАЗЛИЧАЮТСЯ ------------------------------------
    r1 = power_study(rates, 40, 0.05, 0.03, "mean", n_sim=5, n_boot=5)
    assert "меньше самого эффекта" in r1["reason"], r1["reason"]
    r2 = power_study(rates, 40, 0.05, 0.16, "mean", n_sim=5, n_boot=5)
    assert "больше провалов, чем есть" in r2["reason"], r2["reason"]

    # --- правило гейта: «каждая» строже «среднего» --------------------------
    mk = [[(0.05, 40)] * 10, [(0.05, 40)] * 10,
          [(0.05, 40)] * 10, [(-0.02, 40)] * 10]
    g_mean = lower_mean(mk, 500, np.random.default_rng(1))
    g_all = lower_all(mk, 500, np.random.default_rng(1))
    assert g_all < g_mean, (g_all, g_mean)
    assert g_all < 0 < g_mean, (g_all, g_mean)

    # --- РЕСЭМПЛИРОВАНИЕ РЕПЛИК РАСШИРЯЕТ ИНТЕРВАЛ --------------------------
    # Это цена обобщения с «эти четыре прогона» на «метод».
    het = [[(0.08, 40)] * 10, [(0.06, 40)] * 10,
           [(0.02, 40)] * 10, [(0.00, 40)] * 10]
    a = lower_mean(het, 2000, np.random.default_rng(2), False)
    b = lower_mean(het, 2000, np.random.default_rng(2), True)
    assert b < a, (a, b)

    # --- мощность -----------------------------------------------------------
    p_lo = power_study(rates, 40, 0.05, 0.05, "mean", 0.02, 0.02,
                       n_sim=150, n_boot=150, seed=3)
    p_hi = power_study(rates, 160, 0.05, 0.05, "mean", 0.02, 0.02,
                       n_sim=150, n_boot=150, seed=3)
    assert p_hi["power"] >= p_lo["power"], (p_lo["power"], p_hi["power"])
    # ПРАВИЛО «КАЖДАЯ» ДАЁТ МОЩНОСТЬ НЕ ВЫШЕ, чем «среднее»
    p_all = power_study(rates, 40, 0.05, 0.05, "all", 0.02, 0.02,
                        n_sim=150, n_boot=150, seed=3)
    assert p_all["power"] <= p_lo["power"], (p_all["power"], p_lo["power"])
    # НЕОДНОРОДНОСТЬ ПО ЗАДАЧАМ СНИЖАЕТ МОЩНОСТЬ при высокой исходной
    p_het = power_study(rates, 160, 0.05, 0.05, "mean", 0.02, 0.20,
                        n_sim=150, n_boot=150, seed=3)
    assert p_het["power"] <= p_hi["power"], (p_het["power"], p_hi["power"])
    # НЕДОСТИЖИМАЯ КОМБИНАЦИЯ ОБЪЯВЛЯЕТСЯ ДО ПРОГОНА, а не считается
    bad = power_study(rates, 40, 0.05, 0.16, "mean", n_sim=5, n_boot=5)
    assert bad["power"] is None and "провалов" in bad["reason"]
    assert bad["calibrated_on"] == "s0", bad["calibrated_on"]
    try:
        power_study(rates, 40, 0.05, 0.05, "чужое")
    except SystemExit:
        pass
    else:
        raise AssertionError("чужое правило принято")

    # --- чтение опоры -------------------------------------------------------
    import tempfile
    d_ = tempfile.mkdtemp()
    for h in ("s0", "s1"):
        for t in (0, 1):
            json.dump(dict(head=h, sigma=0.0, task_id=t,
                           episodes=[dict(success=(i < 4)) for i in range(5)]),
                      open(os.path.join(d_, f"{h}{t}.json"), "w"))
    r, raw = load_rates_k11g(os.path.join(d_, "*.json"))
    assert r == {"s0": [0.8, 0.8], "s1": [0.8, 0.8]}
    os.remove(os.path.join(d_, "s10.json"))
    try:
        load_rates_k11g(os.path.join(d_, "*.json"))
    except SystemExit as e:
        assert "неполный набор" in str(e), str(e)
    else:
        raise AssertionError("неполный набор принят")

    print("самопроверка k12b пройдена: разбросы по реплике и задаче сохраняют "
          "среднее,\n  правило «каждая реплика» строго жёстче «среднего», "
          "ресэмплирование реплик\n  расширяет интервал, неоднородность по "
          "задачам снижает мощность, недостижимая\n  комбинация объявляется "
          "ДО прогона и калибруется по ХУДШЕЙ опоре, неполный\n  набор ячеек "
          "отвергается")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cells", default="data/k11g/cells/*.json")
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--delta", default="5.0,3.0,2.0")
    ap.add_argument("--discord", default="0.03,0.05,0.08,0.10,0.135")
    ap.add_argument("--episodes", default="40,80,160")
    ap.add_argument("--rep-sd", type=float, default=0.05)
    ap.add_argument("--task-sd", type=float, default=0.10)
    ap.add_argument("--rules", default="mean,mean_rep,all")
    ap.add_argument("--n-sim", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    rates, raw = load_rates_k11g(a.cells, a.sigma)
    print(f"  опора: детерминированная D1 при sigma={a.sigma}, ячеек "
          f"{len(raw)}")
    for h, r in sorted(rates.items()):
        pf = 1.0 - float(np.mean(r))
        print(f"    {h}: успешность {100 * np.mean(r):.2f}%, провалов "
              f"{100 * pf:.1f}%, предел r_loss при +5 пп "
              f"{100 * (r_loss_max(pf, 0.05) or 0):.2f}%")
    print(f"  разброс: по реплике sd={a.rep_sd}, по задаче sd={a.task_sd}; "
          f"реплик {2 * len(rates)}")

    res = {}
    print(f"\n  МОЩНОСТЬ. Правило гейта указано столбцом; «каждая» требует "
          f"прохождения\n  всех реплик, «среднее+» ресэмплирует реплики "
          f"(обобщение на метод).")
    print(f"    {'эфф':>5}{'дискорд':>9}{'эп':>5}"
          + "".join(f"{r:>12}" for r in a.rules.split(",")))
    for dl in [float(x) for x in a.delta.split(",")]:
        for dsc in [float(x) for x in a.discord.split(",")]:
            row, skip = [], None
            for n_ep in [int(x) for x in a.episodes.split(",")]:
                cells = []
                for rule in a.rules.split(","):
                    p = power_study(rates, n_ep, dl / 100.0, dsc, rule,
                                    a.rep_sd, a.task_sd, a.n_sim, a.n_boot,
                                    a.seed)
                    res[f"{dl}|{dsc}|{n_ep}|{rule}"] = p
                    if p["power"] is None:
                        skip = p["reason"]
                        cells.append("  —")
                    else:
                        cells.append(f"{p['power']:.2f}")
                row.append((n_ep, cells))
            if skip:
                print(f"    {dl:>+4.0f}{100 * dsc:>8.1f}%   НЕДОСТИЖИМО: "
                      f"{skip}")
                continue
            for n_ep, cells in row:
                print(f"    {dl:>+4.0f}{100 * dsc:>8.1f}%{n_ep:>5}"
                      + "".join(f"{c:>12}" for c in cells))

    if CLAMPS:
        # СЖАТО ПО ВИДУ, а не по каждому значению среднего: иначе отчёт даёт
        # сотни строк и вытесняет саму таблицу мощности из вывода.
        print("\n  РАЗБРОС УРЕЗАН ДО ВОЗМОЖНОГО (для доли он ограничен и "
              "выполнимостью, и\n  коэффициентом вариации "
              f"{CV_MAX}):")
        agg_ = {}
        for (tag, m), (n, eff) in CLAMPS.items():
            a_ = agg_.setdefault(tag, [0, 1.0, 0.0, 1.0, 0.0])
            a_[0] += n
            a_[1] = min(a_[1], eff)
            a_[2] = max(a_[2], eff)
            a_[3] = min(a_[3], m)
            a_[4] = max(a_[4], m)
        for tag, (n, e1, e2, m1, m2) in sorted(agg_.items()):
            print(f"    {tag}: {n} раз, средние {m1:.4f}-{m2:.4f}, "
                  f"фактический разброс {e1:.4f}-{e2:.4f}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        json.dump(dict(rates=rates, sigma=a.sigma, rep_sd=a.rep_sd,
                       task_sd=a.task_sd, rules=a.rules.split(","),
                       results={k: v for k, v in res.items()},
                       n_sim=a.n_sim, n_boot=a.n_boot, seed=a.seed,
                       clamps={f"{k[0]}|{k[1]}": v
                               for k, v in CLAMPS.items()}),
                  open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {a.out}")
    print("\n  ЧТО ЭТО ДОПУСКАЕТ. Что разбросы по реплике и задаче бета и "
          "независимы; что\n  дискордантность одинакова в среднем по задачам; "
          "что эпизоды независимы —\n  последнее ТРЕБУЕТ свежих начальных "
          "состояний, повторы одного состояния с\n  разными сидами раскатки "
          "независимыми эпизодами не являются.")
    print("  «Нижняя граница выше нуля при истинном эффекте +5 пп» — это тест "
          "ПРЕВОСХОДСТВА,\n  где +5 пп служит альтернативой для расчёта "
          "мощности. Это НЕ то же, что\n  «граница выше 5 пп», то есть "
          "доказательство улучшения минимум на 5 пп.")


if __name__ == "__main__":
    main()
