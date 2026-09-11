"""K-12a: расчёт мощности для RL-эксперимента. Без карты, только numpy.

ЗАЧЕМ ДО КОДА RL. K-11e показал цену незарегистрированного «стало лучше»: при
десяти задачах и двух сидах разброс равен эффекту. Прежде чем писать обучение,
надо знать, осуществим ли вообще статистический вывод на этом наборе задач, и
если нет — сколько задач нужно. Результат может изменить дизайн, поэтому
считается первым.

МОДЕЛЬ ПОРОЖДЕНИЯ ВЗЯТА ИЗ ИЗМЕРЕННОГО, А НЕ ПРИДУМАНА. В §32 измерены ровно
те две величины, которыми RL может двигать успех: доля восстановленных провалов
`r_rec` (провал -> успех) и доля потерянных успехов `r_loss`. Поэтому
симуляция порождает парные исходы так: берётся эмпирическая успешность задачи,
разыгрывается базовый исход, затем провал переворачивается с вероятностью
`r_rec`, а успех — с вероятностью `r_loss`. Чистый эффект тогда

    delta = p_fail * r_rec - (1 - p_fail) * r_loss,

и ПОТОЛОК улучшения равен доле провалов: выше головы не прыгнуть. Это не
тонкость расчёта, а главное ограничение задачи: при успешности 89% эффект
+5 пп требует исправить почти половину всех провалов и при этом не потерять
НИ ОДНОГО успеха.

РАЗБРОС ПО СИДАМ ВХОДИТ В СИМУЛЯЦИЮ. K-11e дал разницу голов 1.75 пп при
идентичной конструкции, K-11g — разную устойчивость к своему шуму. Поэтому
`r_rec` получает случайную добавку на каждую реплику (D1-сид x RL-сид):
политика, обученная иначе, восстанавливает иную долю провалов.

СТАТИСТИКА ТА ЖЕ, ЧТО В ГЕЙТАХ: парная разность, кластерный бутстрап по
задачам, нижняя односторонняя граница 5%. Отвержение = граница выше нуля.

Запуск:
    python3 experiments/k12a_power.py --selftest
    python3 experiments/k12a_power.py --rates 'data/k11e/cells/*.json' \\
        --arm joint12 --out data/k12a/power.json
"""

import argparse
import glob
import json
import os

import numpy as np

ALPHA = 0.05          # односторонний уровень
TARGET_POWER = 0.80
PRIMARY_PP = 5.0      # минимальный практически значимый эффект
SECONDARY_PP = 2.0    # вторичная оценка, не гейт


def net_effect(p_fail, r_rec, r_loss):
    """Чистый эффект в долях. Потолок — сама доля провалов."""
    return p_fail * r_rec - (1.0 - p_fail) * r_loss


def r_rec_for(delta, p_fail, r_loss):
    """Какая доля восстановлений нужна для заданного эффекта.

    Возвращает None, если эффект недостижим даже при r_rec = 1: потолок
    улучшения равен доле провалов минус потери.
    """
    if p_fail <= 0:
        return None
    need = (delta + (1.0 - p_fail) * r_loss) / p_fail
    return None if need > 1.0 else max(need, 0.0)


def calibrate(p_fail, delta, discord):
    """Подобрать (r_rec, r_loss) по ЧИСТОМУ ЭФФЕКТУ И ДИСКОРДАНТНОСТИ.

    ЗАЧЕМ. При r_loss = 0 модель порождает дискордантность p_fail*r_rec — в
    наших условиях около 5%, тогда как в K-11e наблюдалось 15.5%, а в K-11g при
    sigma=0.10 14-20%. Дискордантность и есть главный источник дисперсии парной
    разности: она входит в неё ДВУМЯ направлениями, которые почти сокращаются в
    среднем, но складываются в разбросе. Поэтому оценка мощности при r_loss = 0
    оптимистична по построению и НЕ сопоставима с наблюдённой в K-11e
    стандартной ошибкой.

    Система:  p_fail*r_rec - p_succ*r_loss = delta
              p_fail*r_rec + p_succ*r_loss = discord
    откуда p_fail*r_rec = (discord + delta)/2 и p_succ*r_loss =
    (discord - delta)/2. Возвращает None, если требуется доля вне [0, 1].
    """
    p_succ = 1.0 - p_fail
    if discord < abs(delta):
        return None
    a = (discord + delta) / 2.0
    b = (discord - delta) / 2.0
    if p_fail <= 0 or p_succ <= 0:
        return None
    r_rec, r_loss = a / p_fail, b / p_succ
    if not (0.0 <= r_rec <= 1.0 and 0.0 <= r_loss <= 1.0):
        return None
    return dict(r_rec=r_rec, r_loss=r_loss,
                net=net_effect(p_fail, r_rec, r_loss), discord=discord)


def cluster_lower(diff_by_task, n_boot, rng, alpha=ALPHA):
    """Нижняя односторонняя граница парной разности, бутстрап ПО ЗАДАЧАМ.

    Задачи — кластеры: ресэмплируются они, а не эпизоды. Иначе интервал
    окажется уже настоящего, и это ровно та ошибка, от которой K-6h
    предостерегал.
    """
    k = len(diff_by_task)
    if k < 2:
        raise SystemExit("меньше двух задач: кластерный бутстрап невозможен")
    idx = rng.integers(0, k, size=(n_boot, k))
    # Каждая задача даёт свою среднюю разность и своё число пар; общая оценка
    # взвешена по числу пар, как micro в гейтах.
    d = np.array([x[0] for x in diff_by_task], float)
    w = np.array([x[1] for x in diff_by_task], float)
    num = (d[idx] * w[idx]).sum(axis=1)
    den = w[idx].sum(axis=1)
    boots = num / np.maximum(den, 1e-12)
    return float(np.percentile(boots, 100 * alpha))


def draw_r_rec(r_rec, seed_sd, rng):
    """Добавка на реплику, СОХРАНЯЮЩАЯ СРЕДНЕЕ.

    Первая версия брала `clip(normal(r_rec, sd), 0, 1)`. При малом r_rec
    обрезание снизу отсекает отрицательный хвост и ПОДНИМАЕТ среднее: при
    r_rec=0.06 и sd=0.30 мощность с разбросом оказывалась ВЫШЕ, чем без него,
    то есть модель завышала эффект. Бета-распределение с заданными средним и
    стандартным отклонением лежит в (0,1) по построению, среднее у него ровно
    r_rec, и обрезать ничего не нужно.
    """
    if seed_sd <= 0 or r_rec <= 0 or r_rec >= 1:
        return float(r_rec)
    m, v = float(r_rec), float(seed_sd) ** 2
    lim = m * (1.0 - m)
    if v >= lim:
        raise SystemExit(
            f"разброс sd={seed_sd} невозможен при среднем {m:.3f}: "
            f"дисперсия {v:.4f} не меньше предела {lim:.4f}. Для доли в (0,1) "
            f"стандартное отклонение ограничено сверху самим средним.")
    k = lim / v - 1.0
    return float(rng.beta(m * k, (1.0 - m) * k))


def simulate_once(rates, n_ep, r_rec, r_loss, rng, seed_sd=0.0):
    """Один прогон: парные исходы и нижняя граница.

    `rates` — успешность по задачам. Провал переворачивается с вероятностью
    r_rec, успех — с r_loss; обе доли общие для прогона, но r_rec получает
    добавку на реплику, если задан seed_sd.
    """
    rr = draw_r_rec(r_rec, seed_sd, rng)
    out = []
    for p in rates:
        base = rng.random(n_ep) < p
        flip = np.where(base, rng.random(n_ep) < r_loss,
                        rng.random(n_ep) < rr)
        new = np.where(flip, ~base, base)
        out.append((float((new.astype(float) - base.astype(float)).mean()),
                    n_ep))
    return out


def power(rates, n_ep, r_rec, r_loss, n_sim=400, n_boot=400, seed=0,
          seed_sd=0.0, alpha=ALPHA):
    """Доля прогонов, в которых нижняя граница выше нуля."""
    rng = np.random.default_rng(seed)
    hit = 0
    lows = []
    for _ in range(n_sim):
        d = simulate_once(rates, n_ep, r_rec, r_loss, rng, seed_sd)
        lo = cluster_lower(d, n_boot, rng, alpha)
        lows.append(lo)
        hit += int(lo > 0.0)
    return dict(power=hit / n_sim, n_sim=n_sim,
                lower_median=float(np.median(lows)),
                lower_q10=float(np.percentile(lows, 10)))


def needed_tasks(rates, n_ep, r_rec, r_loss, target=TARGET_POWER,
                 n_sim=200, n_boot=200, seed=0, seed_sd=0.0, cap=200):
    """Сколько задач нужно для заданной мощности. Задачи повторяются циклом.

    ЧТО ЭТО ДОПУСКАЕТ: что новые задачи похожи на имеющиеся по разбросу
    успешности. Если новые сюиты окажутся труднее или однороднее, число
    изменится. Это допущение, а не измерение.
    """
    base = list(rates)
    k = len(base)
    for n in range(k, cap + 1, max(1, k // 2)):
        rs = [base[i % k] for i in range(n)]
        p = power(rs, n_ep, r_rec, r_loss, n_sim, n_boot, seed, seed_sd)
        if p["power"] >= target:
            return dict(n_tasks=n, power=p["power"])
    return dict(n_tasks=None, power=None)


def load_rates(pattern, arm, field="arm_label"):
    """Успешность по задачам из ячеек гейта."""
    per = {}
    for f in sorted(glob.glob(pattern)):
        c = json.load(open(f))
        if str(c.get(field)) != arm:
            continue
        t = int(c["task_id"])
        a, b = per.get(t, (0, 0))
        per[t] = (a + sum(bool(e["success"]) for e in c["episodes"]),
                  b + len(c["episodes"]))
    if not per:
        raise SystemExit(f"нет ячеек с {field}={arm!r} по {pattern}")
    return {t: s / n for t, (s, n) in sorted(per.items())}, per


def selftest():
    # --- потолок улучшения --------------------------------------------------
    assert abs(net_effect(0.11, 1.0, 0.0) - 0.11) < 1e-12
    assert abs(net_effect(0.11, 0.0, 0.0)) < 1e-12
    assert net_effect(0.11, 0.5, 0.10) < 0      # потери съедают всё
    # СКОЛЬКО ВОССТАНОВЛЕНИЙ НУЖНО
    assert abs(r_rec_for(0.05, 0.11, 0.0) - 0.05 / 0.11) < 1e-12
    assert r_rec_for(0.15, 0.11, 0.0) is None   # выше потолка
    # ПОТЕРИ ПОДНИМАЮТ ТРЕБОВАНИЕ К ВОССТАНОВЛЕНИЯМ, и в какой-то момент
    # делают эффект недостижимым. При потерях 5% нужно уже 86%, при 10% —
    # больше единицы, то есть нельзя.
    assert abs(r_rec_for(0.05, 0.11, 0.05) - 0.8591) < 1e-3
    assert r_rec_for(0.05, 0.11, 0.10) is None
    assert r_rec_for(0.0, 0.11, 0.0) == 0.0

    # --- калибровка по эффекту И дискордантности ----------------------------
    c = calibrate(0.105, 0.05, 0.155)
    assert c is not None
    assert abs(c["net"] - 0.05) < 1e-12
    assert abs(0.105 * c["r_rec"] + 0.895 * c["r_loss"] - 0.155) < 1e-12
    # ПРИ НАБЛЮДЁННОЙ ДИСКОРДАНТНОСТИ +5 пп ТРЕБУЕТ ПОЧТИ ВСЕХ ВОССТАНОВЛЕНИЙ
    assert c["r_rec"] > 0.95, c["r_rec"]
    assert 0.05 < c["r_loss"] < 0.07, c["r_loss"]
    # дискордантность не может быть меньше модуля эффекта
    assert calibrate(0.105, 0.05, 0.03) is None
    # и не может требовать доли выше единицы
    assert calibrate(0.105, 0.09, 0.30) is None
    # при нулевых потерях дискордантность равна эффекту
    c0 = calibrate(0.105, 0.05, 0.05)
    assert abs(c0["r_loss"]) < 1e-12 and abs(c0["r_rec"] - 0.05 / 0.105) < 1e-9

    rng = np.random.default_rng(0)
    # --- бутстрап по кластерам ---------------------------------------------
    same = [(0.1, 10)] * 8
    lo = cluster_lower(same, 500, rng)
    assert abs(lo - 0.1) < 1e-9, lo          # нет разброса -> граница = оценке
    mixed = [(0.3, 10)] * 4 + [(-0.1, 10)] * 4
    lo2 = cluster_lower(mixed, 2000, np.random.default_rng(1))
    assert lo2 < 0.1, lo2                    # разброс опускает границу
    try:
        cluster_lower([(0.1, 10)], 10, rng)
    except SystemExit:
        pass
    else:
        raise AssertionError("одна задача принята")
    # ВЗВЕШИВАНИЕ ПО ЧИСЛУ ПАР: задача с большим числом эпизодов весит больше
    w = cluster_lower([(1.0, 100), (0.0, 1)] * 4, 2000,
                      np.random.default_rng(2))
    assert w > 0.5, w

    # --- симуляция ----------------------------------------------------------
    rates = [0.9] * 10
    # ПОД НУЛЕВОЙ ГИПОТЕЗОЙ мощность должна быть около уровня, не выше
    p0 = power(rates, 40, 0.0, 0.0, n_sim=200, n_boot=200, seed=1)
    assert p0["power"] <= 0.02, p0           # ровно ноль эффекта, границы <= 0
    # ЭФФЕКТ РАСТЁТ -> МОЩНОСТЬ РАСТЁТ
    # ЭФФЕКТ БЕРЁТСЯ МАЛЫМ, иначе обе точки упираются в мощность 1.0 и тест
    # ничего не различает: при r_rec=0.30 и десяти задачах эффект 3 пп уже
    # обнаруживается всегда.
    pa = power(rates, 20, 0.02, 0.0, n_sim=300, n_boot=200, seed=2)
    pb = power(rates, 20, 0.30, 0.0, n_sim=300, n_boot=200, seed=2)
    assert pa["power"] < 0.9, pa["power"]
    assert pb["power"] > pa["power"], (pa["power"], pb["power"])
    # БОЛЬШЕ ЗАДАЧ -> МОЩНОСТЬ РАСТЁТ
    p10 = power([0.9] * 10, 10, 0.04, 0.0, n_sim=300, n_boot=200, seed=3)
    p40 = power([0.9] * 40, 10, 0.04, 0.0, n_sim=300, n_boot=200, seed=3)
    assert p10["power"] < 0.9 and p40["power"] > p10["power"], \
        (p10["power"], p40["power"])
    # ДОБАВКА НА РЕПЛИКУ СОХРАНЯЕТ СРЕДНЕЕ. Прежняя версия с clip завышала
    # его при малом r_rec, и мощность с разбросом выходила ВЫШЕ, чем без.
    g = np.random.default_rng(7)
    draws = [draw_r_rec(0.06, 0.03, g) for _ in range(20000)]
    assert abs(float(np.mean(draws)) - 0.06) < 2e-3, float(np.mean(draws))
    assert abs(float(np.std(draws)) - 0.03) < 3e-3, float(np.std(draws))
    assert min(draws) > 0.0 and max(draws) < 1.0
    assert draw_r_rec(0.06, 0.0, g) == 0.06
    try:
        draw_r_rec(0.06, 0.5, g)
    except SystemExit:
        pass
    else:
        raise AssertionError("невозможный разброс принят")

    # РАЗБРОС ПО РЕПЛИКАМ СНИЖАЕТ МОЩНОСТЬ НЕ ВСЕГДА, И ЭТО ВАЖНО ЗНАТЬ.
    # При УЖЕ ВЫСОКОЙ мощности разброс её снижает: часть реплик падает ниже
    # порога. При НИЗКОЙ — может повысить, потому что часть реплик получает
    # эффект выше порога, а остальные и так не проходили. Измерено: при
    # эффекте 0.4 пп мощность растёт 0.14 -> 0.18, при 0.6 пп падает
    # 0.33 -> 0.30, при 1.0 пп 0.64 -> 0.60. Поэтому заявлять «разброс всегда
    # снижает мощность» нельзя, и в выводе скрипта это оговорено.
    ps0 = power(rates, 40, 0.10, 0.0, n_sim=400, n_boot=200, seed=11)
    ps1 = power(rates, 40, 0.10, 0.0, n_sim=400, n_boot=200, seed=11,
                seed_sd=0.05)
    assert ps0["power"] > 0.5, ps0["power"]
    assert ps1["power"] < ps0["power"], (ps0["power"], ps1["power"])
    lo0 = power(rates, 40, 0.04, 0.0, n_sim=400, n_boot=200, seed=11)
    lo1 = power(rates, 40, 0.04, 0.0, n_sim=400, n_boot=200, seed=11,
                seed_sd=0.05)
    assert lo1["power"] >= lo0["power"], "инверсия при низкой мощности"
    # ПОТЕРИ СНИЖАЮТ МОЩНОСТЬ: при них чистый эффект меньше.
    pl = power(rates, 40, 0.10, 0.02, n_sim=400, n_boot=200, seed=11)
    assert pl["power"] < ps0["power"], (pl["power"], ps0["power"])

    # --- чтение ставок ------------------------------------------------------
    import tempfile
    d = tempfile.mkdtemp()
    for t in (0, 1):
        json.dump(dict(arm_label="x", task_id=t,
                       episodes=[dict(success=(i < 4)) for i in range(5)]),
                  open(os.path.join(d, f"c{t}.json"), "w"))
    r, raw = load_rates(os.path.join(d, "*.json"), "x")
    assert r == {0: 0.8, 1: 0.8} and raw[0] == (4, 5)
    try:
        load_rates(os.path.join(d, "*.json"), "нет")
    except SystemExit:
        pass
    else:
        raise AssertionError("отсутствие руки принято")

    print("самопроверка k12a пройдена: потолок улучшения равен доле провалов, "
          "эффект выше\n  потолка объявляется недостижимым; бутстрап "
          "ресэмплирует ЗАДАЧИ и взвешивает по\n  числу пар; под нулевой "
          "гипотезой мощность не превышает уровня; мощность растёт\n  с "
          "эффектом и числом задач, падает от потерь, а от разброса по "
          "репликам падает\n  при высокой мощности и РАСТЁТ при низкой — "
          "и это проверено в обе стороны;\n  добавка на реплику сохраняет "
          "среднее (бета, не обрезанная нормаль);\n  калибровка по эффекту И "
          "дискордантности отвергает недостижимые комбинации")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rates", default="data/k11e/cells/t*_i*_*.json")
    ap.add_argument("--arm", default="joint12")
    ap.add_argument("--field", default="arm_label")
    ap.add_argument("--r-loss", type=float, default=0.0,
                    help="доля потерянных успехов. 0 — ОПТИМИСТИЧНЫЙ предел: "
                         "он порождает дискордантность вдвое-втрое ниже "
                         "наблюдённой и потому завышает мощность")
    ap.add_argument("--discord", default="0.05,0.14,0.155",
                    help="наблюдённые доли дискордантных пар для калибровки "
                         "(K-11e: 0.135-0.155; K-11g при sigma=0.10: "
                         "0.14-0.20)")
    ap.add_argument("--seed-sd", type=float, default=0.10,
                    help="стд добавки к r_rec на реплику (D1-сид x RL-сид). "
                         "Разыгрывается бета-распределением с ТОЧНО таким "
                         "средним: нормаль с обрезанием завышала бы среднее")
    ap.add_argument("--episodes", default="20,40,80")
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    rates, raw = load_rates(a.rates, a.arm, a.field)
    vals = list(rates.values())
    p_succ = float(np.mean(vals))
    p_fail = 1.0 - p_succ
    print(f"  опора {a.arm!r}: задач {len(rates)}, успешность "
          f"{100 * p_succ:.1f}%, разброс по задачам "
          f"{100 * min(vals):.0f}-{100 * max(vals):.0f}%")
    print(f"  успешность по задачам: "
          + " ".join(f"{t}:{100 * v:.0f}%" for t, v in rates.items()))
    print(f"\n  ПОТОЛОК УЛУЧШЕНИЯ = доля провалов = {100 * p_fail:.1f} пп. "
          f"Выше него эффект недостижим\n  при любой доле восстановлений.")
    for pp in (PRIMARY_PP, SECONDARY_PP):
        need = r_rec_for(pp / 100.0, p_fail, a.r_loss)
        if need is None:
            print(f"    эффект +{pp:.0f} пп при потерях "
                  f"{100 * a.r_loss:.0f}%: НЕДОСТИЖИМ")
        else:
            print(f"    эффект +{pp:.0f} пп при потерях "
                  f"{100 * a.r_loss:.0f}%: нужно восстанавливать "
                  f"{100 * need:.0f}% провалов")
    # Измеренное в §32 для справки: восстановления 60-67% при потерях 7-16%.
    meas = net_effect(p_fail, 0.63, 0.11)
    print(f"    при ИЗМЕРЕННЫХ в K-11g долях (восстановлено 63%, потеряно "
          f"11%) чистый эффект\n    составил бы {100 * meas:+.1f} пп — то "
          f"есть случайный шум сам по себе даёт минус.")

    res = {}
    print(f"\n  МОЩНОСТЬ (односторонний уровень {ALPHA}, кластерный бутстрап "
          f"по задачам, разброс\n  по репликам sd={a.seed_sd}, потери "
          f"{100 * a.r_loss:.0f}%)")
    print(f"    {'эффект':>8}{'r_rec':>8}{'эпизодов':>10}{'задач':>7}"
          f"{'мощность':>10}{'ниж.гр. медиана':>17}")
    for pp in (PRIMARY_PP, SECONDARY_PP):
        rr = r_rec_for(pp / 100.0, p_fail, a.r_loss)
        if rr is None:
            continue
        for n_ep in [int(x) for x in a.episodes.split(",")]:
            p = power(list(rates.values()), n_ep, rr, a.r_loss,
                      a.n_sim, a.n_boot, a.seed, a.seed_sd)
            print(f"    {pp:>+7.0f}{100 * rr:>7.0f}%{n_ep:>10}"
                  f"{len(rates):>7}{p['power']:>10.2f}"
                  f"{100 * p['lower_median']:>+16.2f}")
            res[f"{pp}|{n_ep}|{len(rates)}"] = p
        nt = needed_tasks(list(rates.values()), 40, rr, a.r_loss,
                          n_sim=max(a.n_sim // 2, 100),
                          n_boot=max(a.n_boot // 2, 100), seed=a.seed,
                          seed_sd=a.seed_sd)
        if nt["n_tasks"] is None:
            print(f"      для +{pp:.0f} пп мощности {TARGET_POWER} не "
                  f"достичь и на 200 задачах")
        else:
            print(f"      для +{pp:.0f} пп мощность {TARGET_POWER} "
                  f"достигается при ~{nt['n_tasks']} задачах "
                  f"(по 40 эпизодов)")
        res[f"needed|{pp}"] = nt

    print(f"\n  КАЛИБРОВКА ПО ДИСКОРДАНТНОСТИ. При r_loss=0 модель даёт "
          f"дискордантность\n  p_fail*r_rec, то есть около "
          f"{100 * p_fail * (r_rec_for(PRIMARY_PP / 100, p_fail, 0) or 0):.0f}"
          f"% для +{PRIMARY_PP:.0f} пп — против наблюдённых 13.5-15.5% в "
          f"K-11e.\n  Дискордантность входит в разброс парной разности ДВУМЯ "
          f"направлениями, которые\n  почти сокращаются в среднем, но "
          f"складываются в дисперсии.")
    print(f"    {'эффект':>8}{'дискорд':>9}{'r_rec':>8}{'r_loss':>8}"
          f"{'эпизодов':>10}{'мощность':>10}{'ниж.гр.':>10}")
    for pp in (PRIMARY_PP, SECONDARY_PP):
        for dsc in [float(x) for x in a.discord.split(",")]:
            cal = calibrate(p_fail, pp / 100.0, dsc)
            if cal is None:
                print(f"    {pp:>+7.0f}{100 * dsc:>8.1f}%  НЕДОСТИЖИМО при "
                      f"такой дискордантности")
                res[f"cal|{pp}|{dsc}"] = None
                continue
            # ЭПИЗОДЫ ПЕРЕБИРАЮТСЯ, А НЕ ЗАШИТЫ. Эпизоды уменьшают только
            # ВНУТРИЗАДАЧНУЮ составляющую дисперсии; межзадачная остаётся, и
            # надо видеть, где она начинает доминировать — иначе непонятно,
            # можно ли заменить новые задачи более длинным прогоном.
            for n_ep in [int(x) for x in a.episodes.split(",")]:
                pw = power(list(rates.values()), n_ep, cal["r_rec"],
                           cal["r_loss"], a.n_sim, a.n_boot, a.seed,
                           a.seed_sd)
                print(f"    {pp:>+7.0f}{100 * dsc:>8.1f}%"
                      f"{100 * cal['r_rec']:>7.0f}%{100 * cal['r_loss']:>7.1f}%"
                      f"{n_ep:>10}{pw['power']:>10.2f}"
                      f"{100 * pw['lower_median']:>+10.2f}")
                res[f"cal|{pp}|{dsc}|{n_ep}"] = dict(cal=cal, power=pw)
            nt = needed_tasks(list(rates.values()), 40, cal["r_rec"],
                              cal["r_loss"], n_sim=max(a.n_sim // 2, 100),
                              n_boot=max(a.n_boot // 2, 100), seed=a.seed,
                              seed_sd=a.seed_sd)
            res[f"calneed|{pp}|{dsc}"] = nt
            if nt["n_tasks"] is None:
                print(f"      мощности {TARGET_POWER} не достичь и на 200 "
                      f"задачах")
            elif nt["n_tasks"] > len(rates):
                print(f"      мощность {TARGET_POWER} требует ~{nt['n_tasks']} "
                      f"задач")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        json.dump(dict(arm=a.arm, rates=rates, raw=raw, p_succ=p_succ,
                       r_loss=a.r_loss, seed_sd=a.seed_sd, results=res,
                       alpha=ALPHA, target_power=TARGET_POWER),
                  open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {a.out}")
    print("\n  ОСОБЕННОСТЬ ЧТЕНИЯ. Разброс по репликам снижает мощность только "
          "когда она уже\n  высока; при низкой мощности он её ПОВЫШАЕТ, "
          "потому что часть реплик перескакивает\n  порог, а остальные и так "
          "не проходили. Поэтому строку с низкой мощностью нельзя\n  читать "
          "как «с разбросом будет ещё хуже».")
    print("\n  ЧТО ЭТО ДОПУСКАЕТ. Что новые задачи похожи на имеющиеся по "
          "разбросу успешности;\n  что эффект действует через восстановление "
          "провалов и потерю успехов с ОДНИМИ\n  долями на всех задачах; что "
          "добавка на реплику нормальна. Первое проверяемо\n  только новыми "
          "задачами, второе §32 опровергает частично (задача 8 ведёт себя\n  "
          "иначе), третье — удобное допущение. Поэтому числа — ориентир для "
          "дизайна, а не\n  обещание.")


if __name__ == "__main__":
    main()
