"""K-5c: разбор кривых устаревания. Правила чтения записаны ДО прихода данных.

ГЛАВНЫЙ ВОПРОС. Связан ли измеренный дрейф с провалом задачи? Если нет, то
дрейф — хороший диагностический сигнал о несогласии политики с собой, но
негодная метка для выбора горизонта, и обучать на нём нечего.

ПОЧЕМУ НЕ ПО ЗАДАЧАМ. Напрашивается сопоставить средний дрейф задачи со
штрафом Success(H=4) − Success(H=20) из K-5b. Так делать нельзя: при n = 10
критическая корреляция на уровне 0.05 равна 0.632, а сам штраф шумный
(при p ≈ 0.9 и 50 эпизодах ошибка разности ~6 пунктов). Нулевой результат
получится почти наверняка независимо от истины, то есть тест бессилен.
Он всё же считается — как вторичный, с явной пометкой о мощности.

ПОЧЕМУ ПО ЭПИЗОДАМ. Зонд при H_exec = 20 сам является прогоном с горизонтом
20 и сам даёт исход каждого эпизода. Значит вопрос ставится напрямую: чаще ли
проваливаются эпизоды с высоким дрейфом. При 200 эпизодах критическая
корреляция ~0.14, мощность на порядок выше. И не нужно джойнить прогоны с
разных хостов, где жадный argmax мог дать другие траектории.

ОБРАТНАЯ ПРИЧИННОСТЬ — ГЛАВНАЯ ЛОВУШКА. Дрейф меряется вдоль той самой
траектории, чей исход предсказывается. Разваливающийся эпизод проходит через
странные состояния, и дрейф там высок ПОТОМУ ЧТО он разваливается. Поэтому
дрейф считается в РАННЕМ окне (--early-steps), а предсказывается исход,
который определится позже. Без этого корреляция ничего не значит.

ЧТО СЧИТАЕТСЯ МЕТКОЙ. Сырое D(j) между задачами несравнимо: масштаб движения
разный. Кандидаты, все безразмерные или нормированные:
  ratio   = D(j) / D_держать(j)   — доля сдвига, НЕ уловленная планом
  excess  = sqrt(D(j)^2 - D(1)^2) — очищенное от постоянного несогласия
  cosdef  = 1 - cos(старое, свежее)
Ни один не объявляется главным заранее: считаются все три, сравниваются по
AUC. Квадратурная очистка — эвристика, а не доказанная декомпозиция: D(1)
содержит и несогласие политики, и один шаг настоящего устаревания.

Запуск:
    python3 experiments/k5c_analyze.py --selftest
    python3 experiments/k5c_analyze.py --dir data/k5c_drift
"""

import argparse
import glob
import json
import os

import numpy as np

CHUNK = 20


def auc(score, label):
    """AUC ранговым способом. label=1 — провал (то, что предсказываем)."""
    score, label = np.asarray(score, float), np.asarray(label, int)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    n1, n0 = int(label.sum()), int((1 - label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # средние ранги для связок — иначе дискретные метрики дают смещение
    s = score[order]
    i = 0
    while i < len(s):
        k = i
        while k + 1 < len(s) and s[k + 1] == s[i]:
            k += 1
        if k > i:
            ranks[order[i:k + 1]] = (i + 1 + k + 1) / 2.0
        i = k + 1
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auc_within(score, label, group):
    """AUC ВНУТРИ групп: сравниваются только пары из одной задачи.

    ЗАЧЕМ. Объединённый по задачам AUC может целиком объясняться тем, что у
    трудной задачи дрейф выше — то есть переизмерять межзадачную корреляцию.
    Но межзадачная неоднородность лечится таблицей из десяти чисел, без
    всякого предсказателя (см. вывод k5b_summarize). Методу нужно различение
    ВНУТРИ задачи, и меряет его только эта величина.

    Агрегация — по числу дискордантных пар, а не простым средним: задачи с
    одним провалом иначе весили бы столько же, сколько задачи с пятью.
    """
    score, label = np.asarray(score, float), np.asarray(label, int)
    group = np.asarray(group)
    num = den = 0.0
    for g in np.unique(group):
        m = group == g
        n1, n0 = int(label[m].sum()), int((1 - label[m]).sum())
        if n1 == 0 or n0 == 0:
            continue
        a = auc(score[m], label[m])
        if np.isfinite(a):
            num += a * n1 * n0
            den += n1 * n0
    return (num / den) if den > 0 else float("nan"), int(den)


def perm_null_within(score, label, group, n=2000, seed=0):
    """Нуль для стратифицированного AUC: метки мешаются ВНУТРИ каждой задачи.

    Перемешивать глобально нельзя — это разрушило бы и межзадачную структуру,
    и нуль оказался бы шире правды.
    """
    rng = np.random.default_rng(seed)
    group = np.asarray(group)
    lab = np.asarray(label).copy()
    idx = [np.where(group == g)[0] for g in np.unique(group)]
    out = np.empty(n)
    for i in range(n):
        for ii in idx:
            lab[ii] = rng.permutation(lab[ii])
        out[i] = auc_within(score, lab, group)[0]
    return out


def perm_null(score, label, n=2000, seed=0):
    """Распределение AUC при перемешанных метках. Даёт честный порог."""
    rng = np.random.default_rng(seed)
    lab = np.asarray(label).copy()
    out = np.empty(n)
    for i in range(n):
        rng.shuffle(lab)
        out[i] = auc(score, lab)
    return out


def episode_table(path, early_steps):
    """Сводка по эпизодам одного npz: ранний дрейф и исход."""
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    succ = z["success_by_env"]                      # (k_set, n_envs)
    T, j, env, rnd = z["T"], z["j"], z["env"], z["round"]
    done = z["done"]
    rows = []
    # РАННЕЕ ОКНО и только НЕзавершённые шаги: после done траектории нет.
    base = (T < early_steps) & (done == 0) & (j > 0)
    for k in range(succ.shape[0]):
        for e in range(succ.shape[1]):
            m = base & (rnd == k) & (env == e)
            if m.sum() < 50:                        # слишком мало — не считаем
                continue
            d, h = z["pose_l2"][m], z["hold_l2"][m]
            rec = dict(
                task=meta["task_id"], suite=meta["suite"], round=k, env=e,
                exec_horizon=meta["exec_horizon"],
                init_state_id=e + k * succ.shape[1],
                fail=int(succ[k, e] == 0), n_rows=int(m.sum()),
                d_mean=float(d.mean()),
                ratio=float(d.sum() / max(h.sum(), 1e-12)),
                cosdef=float(1.0 - np.nanmean(z["pose_cos"][m])),
                cum=float(np.nanmean(z["cum_pose_l2"][m])),
            )
            dj1 = z["pose_l2"][base & (rnd == k) & (env == e) & (j == 1)]
            fl = float(dj1.mean()) if dj1.size else 0.0
            rec["excess"] = float(np.sqrt(max(rec["d_mean"] ** 2 - fl ** 2, 0)))
            rows.append(rec)
    return rows, meta


def task_curves(path):
    """Кривые по смещению для одной задачи."""
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    j, done = z["j"], z["done"]
    out = {}
    for key in ("pose_l2", "hold_l2", "pose_cos"):
        v = np.full(CHUNK, np.nan)
        for k in range(CHUNK):
            m = (j == k) & (done == 0)
            if m.any():
                v[k] = np.nanmean(z[key][m])
        out[key] = v
    with np.errstate(invalid="ignore", divide="ignore"):
        out["ratio"] = out["pose_l2"] / out["hold_l2"]
    out["meta"] = meta
    return out


def selftest():
    # 1. AUC на известных ответах
    assert abs(auc([1, 2, 3, 4], [0, 0, 1, 1]) - 1.0) < 1e-12, "идеальный AUC"
    assert abs(auc([4, 3, 2, 1], [0, 0, 1, 1]) - 0.0) < 1e-12, "обратный AUC"
    assert abs(auc([1, 1, 1, 1], [0, 0, 1, 1]) - 0.5) < 1e-12, \
        "полностью связанные значения обязаны давать ровно 0.5"
    # связка считается за половину: пары (2,1)=1, (2,2)=0.5, (3,1)=1, (3,2)=1
    assert abs(auc([1, 2, 2, 3], [0, 0, 1, 1]) - 0.875) < 1e-12, "связки"

    # 2. Перестановочный нуль центрирован на 0.5 и НЕ вырожден. Проверка нужна
    #    потому, что в этом проекте нули уже дважды насыщались и переставали
    #    что-либо различать.
    rng = np.random.default_rng(1)
    lab = (rng.random(200) < 0.15).astype(int)
    null = perm_null(rng.normal(size=200), lab, n=500)
    assert abs(null.mean() - 0.5) < 0.02, f"нуль смещён: {null.mean():.3f}"
    assert 0.03 < null.std() < 0.15, \
        f"нуль вырожден или слишком широк: sd={null.std():.3f}"

    # 3. МОЩНОСТЬ. Синтетика, где связь ЗАДАНА: провал вероятнее при высоком
    #    дрейфе. Тест обязан это увидеть на 200 точках и НЕ увидеть на 10.
    x = rng.normal(size=200)
    y = (rng.random(200) < 1 / (1 + np.exp(-(x - 1.2)))).astype(int)
    a_big = auc(x, y)
    assert a_big > 0.65, f"на 200 точках связь обязана быть видна: {a_big:.3f}"
    r10 = []
    for s in range(200):
        g = np.random.default_rng(s)
        idx = g.choice(200, 10, replace=False)
        if 0 < y[idx].sum() < 10:
            r10.append(auc(x[idx], y[idx]))
    frac = float(np.mean(np.array(r10) > 0.75))
    assert frac < 0.8, \
        (f"по десяти точкам тест обязан быть НЕнадёжным, а уверенно "
         f"срабатывает в {frac:.0%} — проверка мощности бессмысленна")
    # 4. СТРАТИФИКАЦИЯ ОБЯЗАНА УБИВАТЬ ЧИСТО МЕЖЗАДАЧНЫЙ СИГНАЛ.
    #    Строим случай, где ВНУТРИ задачи связи нет вовсе: у задачи A дрейф
    #    низкий и провалов мало, у задачи B дрейф высокий и провалов много,
    #    но внутри каждой провал назначается СЛУЧАЙНО. Объединённый AUC
    #    обязан быть высоким (он ловит разницу задач), стратифицированный —
    #    около 0.5. Если это не так, колонка «внутри» бесполезна.
    g2 = np.r_[np.zeros(100, int), np.ones(100, int)]
    sc2 = np.r_[rng.normal(0, 1, 100), rng.normal(3, 1, 100)]
    lb2 = np.r_[(rng.random(100) < 0.10).astype(int),
                (rng.random(100) < 0.50).astype(int)]
    a_pool = auc(sc2, lb2)
    a_within, npair = auc_within(sc2, lb2, g2)
    assert a_pool > 0.65, \
        f"объединённый AUC обязан поймать межзадачную разницу: {a_pool:.3f}"
    assert abs(a_within - 0.5) < 0.10, \
        (f"стратифицированный AUC обязан быть около 0.5, получено "
         f"{a_within:.3f} — стратификация не работает")
    nw = perm_null_within(sc2, lb2, g2, n=300)
    assert abs(nw.mean() - 0.5) < 0.03, f"нуль внутри смещён: {nw.mean():.3f}"

    print("самопроверка пройдена:")
    print("  AUC точен на известных случаях, связки обрабатываются")
    print(f"  перестановочный нуль: среднее {null.mean():.3f}, sd {null.std():.3f}")
    print(f"  заданная связь видна на 200 точках (AUC {a_big:.3f}) и теряется "
          f"на 10 (уверенно лишь в {frac:.0%} подвыборок)")
    print(f"  чисто МЕЖзадачный сигнал: общий AUC {a_pool:.3f}, "
          f"внутри задач {a_within:.3f} ({npair} пар) — стратификация его "
          f"снимает")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dir", default="data/k5c_drift")
    ap.add_argument("--early-steps", type=int, default=100,
                    help="дрейф считается ТОЛЬКО на первых N шагах эпизода. "
                         "Иначе разваливающийся эпизод даёт высокий дрейф "
                         "потому что разваливается — обратная причинность")
    ap.add_argument("--exec-horizon", type=int, default=20)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()

    files = sorted(glob.glob(os.path.join(args.dir, "*.npz")))
    if not files:
        raise SystemExit(f"нет npz в {args.dir}")

    eps, curves = [], {}
    for f in files:
        try:
            rows, meta = episode_table(f, args.early_steps)
        except Exception as e:                       # noqa: BLE001
            print(f"  пропуск {os.path.basename(f)}: {e}")
            continue
        if meta["exec_horizon"] != args.exec_horizon:
            continue
        eps += rows
        curves[meta["task_id"]] = task_curves(f)

    if not eps:
        raise SystemExit(f"нет эпизодов с exec_horizon={args.exec_horizon}")

    print(f"\nэпизодов: {len(eps)}, задач: {len(curves)}, "
          f"раннее окно: {args.early_steps} шагов")

    # ---- форма кривой по задачам: есть ли РАЗНАЯ скорость старения ----------
    print("\n" + "=" * 74)
    print("КРИВЫЕ ПО ЗАДАЧАМ (доля сдвига, НЕ уловленная планом)")
    print(f"  {'задача':>7}" + "".join(f"{f'j={k}':>8}" for k in (1, 4, 8, 12, 19))
          + f"{'успех':>8}")
    for tid in sorted(curves):
        r = curves[tid]["ratio"]
        s = np.mean([1 - e["fail"] for e in eps if e["task"] == tid])
        print(f"  {tid:>7}" + "".join(f"{r[k]:>8.2f}" for k in (1, 4, 8, 12, 19))
              + f"{s:>8.0%}")
    sp = np.array([curves[t]["ratio"][19] for t in sorted(curves)])
    if len(sp) < 3:
        print(f"\n  задач всего {len(sp)} — о разбросе между задачами говорить "
              f"рано.\n  Кривая одной задачи оценена плотно (десятки тысяч "
              f"строк) и ей верить можно,\n  но межзадачная неоднородность "
              f"требует хотя бы трёх-четырёх.")
    else:
        print(f"\n  разброс ratio(19) между задачами: "
              f"{sp.min():.2f}–{sp.max():.2f}, sd {sp.std():.3f}")
        print("  Одинаковые кривые у всех задач = стареть все планы стареют "
              "одинаково,\n  и адаптировать горизонт по состоянию не по чему.")

    # ---- ГЛАВНЫЙ ГЕЙТ: предсказывает ли ранний дрейф провал -----------------
    fail = np.array([e["fail"] for e in eps])
    print("\n" + "=" * 74)
    print(f"ГЛАВНЫЙ ГЕЙТ: ранний дрейф против провала  "
          f"(провалов {fail.sum()} из {len(fail)})")
    if fail.sum() < 5 or fail.sum() > len(fail) - 5:
        print("  СЛИШКОМ ПЕРЕКОШЕНЫ ИСХОДЫ — гейт не считается.")
    else:
        grp = np.array([e["task"] for e in eps])
        print(f"  {'метка':>10}{'AUC общ':>9}{'нуль 95%':>14}"
              f"{'AUC внутри':>12}{'нуль 95%':>14}{'пар':>7}")
        for key in ("d_mean", "ratio", "excess", "cosdef", "cum"):
            sc = np.array([e[key] for e in eps], float)
            a = auc(sc, fail)
            nl = perm_null(sc, fail, n=2000)
            lo, hi = np.quantile(nl, [0.025, 0.975])
            aw, npair = auc_within(sc, fail, grp)
            nw = perm_null_within(sc, fail, grp, n=2000)
            wlo, whi = np.quantile(nw, [0.025, 0.975])
            mark = "*" if np.isfinite(aw) and (aw > whi or aw < wlo) else " "
            print(f"  {key:>10}{a:>9.3f}{lo:>7.3f}–{hi:<6.3f}"
                  f"{aw:>12.3f}{wlo:>8.3f}–{whi:<6.3f}{npair:>6}{mark}")
        print("\n  ЗВЁЗДОЧКА — значим ВНУТРИ задач. Именно эта колонка решает.")
        print("  Общий AUC может целиком объясняться тем, что у трудной задачи")
        print("  дрейф выше; такая межзадачная связь лечится таблицей из")
        print("  десяти чисел и предсказателя не требует.")
        print("\n  ЧИТАТЬ ТАК, правило записано до данных.")
        print("  Порог из K-5к: AUC < 0.60 — дрейф НЕ связан с исходом, и как")
        print("  метка горизонта он негоден, сколь угодно красиво ни росла бы")
        print("  кривая. Тогда результат работы — «дрейф плохой прокси»,")
        print("  а не adaptive horizon.")

    # ---- вторичное: по задачам, с честной оговоркой о мощности -------------
    tids = sorted(curves)
    if len(tids) >= 3:
        x = np.array([curves[t]["ratio"][19] for t in tids])
        y = np.array([np.mean([e["fail"] for e in eps if e["task"] == t])
                      for t in tids])
        if x.std() > 1e-9 and y.std() > 1e-9:
            r = float(np.corrcoef(x, y)[0, 1])
            print(f"\n  вторично, по задачам (n={len(tids)}): r = {r:+.3f}")
            print(f"  критическое |r| при n={len(tids)} на уровне 0.05 ≈ "
                  f"{2.0 / np.sqrt(len(tids)):.2f} — тест почти бессилен,")
            print("  и НУЛЕВОЙ результат здесь ничего не опровергает.")


if __name__ == "__main__":
    main()
