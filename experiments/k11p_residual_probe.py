"""K-11p. Линейный зонд: несёт ли h24 сведения об остатке сверх z0.

ЧТО ЭТО ЗА ВОПРОС. Голова поправки видит два входа: позднее состояние
`res_norm(h24)` и латент предсказанного черновика `z0`. Если весь остаток
восстановим уже по одному `z0`, то поздняя ветвь ничего не добавляет, а
«коррекция по h24» оказалась бы названием, а не механизмом. Зонд отвечает на
это ДО обучения головы и стоит минуты вместо суток.

ЧТО ЗОНД НЕ РЕШАЕТ. Отрицательный ответ линейного зонда НЕ закрывает ветвь:
линейная модель не обязана видеть то, что видит трёхслойная голова с tanh.
Правило чтения ниже намеренно АСИММЕТРИЧНО: положительный ответ разрешает
идти дальше с текущим источником, отрицательный отправляет к нелинейному
зонду или к парному кэшу считываний, но не к закрытию направления.

ТРИ ВАРИАНТА ВХОДА, ОДИН ПРОХОД ПО ДАННЫМ. Грамиан копится по объединённому
вектору [res_norm(h24), z0, 1], а варианты — это подматрицы: так все три
модели гарантированно обучены на одних и тех же строках, и разница между
ними не может оказаться разницей выборок.

ДВЕ МИШЕНИ. Сырые коэффициенты `a = (z* - z0) @ B` дают R^2 и сопоставимы с
обычной регрессией. Но голова решает ДРУГУЮ задачу: она выдаёт
`rho * tanh(...)`, то есть ограниченную величину, поэтому её мишень —
`clip(a, -rho, rho) / rho`. Именно по ней считается основная оценка.

ОСНОВНАЯ ОЦЕНКА — ПОСЛЕ ДЕКОДЕРА, а не в латенте. Уменьшение ошибки
коэффициентов не переводится в улучшение действий линейно: декодер
нелинеен, а шаги 0-7 чанка — это то, что реально исполняется. Поза,
вращение и знак схвата считаются РАЗДЕЛЬНО: их смешение в одно число уже
стоило нам гейта, неспособного подтверждать (K-10g).

РАЗБИЕНИЕ И БУТСТРАП — ПО ЭПИЗОДАМ. Соседние наблюдения одного эпизода
почти повторяют друг друга; доверительный интервал по наблюдениям был бы
уже истинного в разы, и любая разница выглядела бы значимой.
"""
import argparse
import copy
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k11a_build_hicora_cache as k11a  # noqa: E402
import k11b_hicora_identity as k11b  # noqa: E402

N_POS = k11a.N_POS
H_EXEC = k11a.H_EXEC
# Варианты входа. Порядок значим: он же порядок столбцов в таблицах.
VARIANTS = ("draft", "oracle", "z0", "h24", "both")
STAT_KEYS = ("pos_sq", "pos_n", "rot_sq", "rot_n", "grip_bad", "grip_n")
N_STAT = len(STAT_KEYS)
LAMBDAS = tuple(float(x) for x in np.logspace(-6.0, 2.0, 17))
N_BOOT = 1000
BOOT_SEED = 21
INNER_SEED = 13
INNER_FRAC = 0.2
# Пре-регистрированный допуск маршрутизации: схват варианта «both» не
# должен быть хуже варианта «z0» больше чем на полпроцентного пункта.
GRIP_TOL = 0.005


# --------------------------------------------------------------------------
# ошибки по наблюдениям
# --------------------------------------------------------------------------
def err_per_obs(a, ref):
    """То же, что `k11a.err_sums`, но БЕЗ свёртки по наблюдениям.

    Бутстрап идёт по эпизодам, поэтому суммы нужны не одним числом на батч,
    а по строкам, чтобы их можно было разложить по эпизодам. Сходимость с
    `k11a.err_sums` проверяется в самопроверке: расхождение означало бы, что
    зонд и диагностика меряют разные величины и их доли несопоставимы.
    """
    A = np.asarray(a, np.float64)[:, :H_EXEC]
    R = np.asarray(ref, np.float64)[:, :H_EXEC]
    d = A - R
    n = A.shape[0]
    out = np.empty((n, N_STAT), np.float64)
    out[:, 0] = (d[..., :3] ** 2).sum((1, 2))
    out[:, 1] = d[..., :3].size / max(n, 1)
    out[:, 2] = (d[..., 3:6] ** 2).sum((1, 2))
    out[:, 3] = d[..., 3:6].size / max(n, 1)
    out[:, 4] = (np.sign(A[..., 6]) != np.sign(R[..., 6])).sum(1)
    out[:, 5] = A[..., 6].size / max(n, 1)
    return out


def finish(stat):
    """RMS и доля ошибок знака из накопленных сумм.

    Корень берётся ОДИН раз в конце: среднее корней по батчам не равно
    корню общего среднего и зависело бы от `--batch`.
    """
    s = np.asarray(stat, np.float64)
    return dict(pos=float(np.sqrt(s[0] / max(s[1], 1e-9))),
                rot=float(np.sqrt(s[2] / max(s[3], 1e-9))),
                grip=float(s[4] / max(s[5], 1e-9)))


def gains_from(stat):
    """Доли возвращённого улучшения для всех вариантов из одного набора сумм.

    Знаменатель — ошибка черновика: `D(z*)` есть сама опора, её ошибка равна
    нулю по построению, поэтому доля равна `1 - e_v / e_draft`. Это тот же
    знаменатель, что в K-11a, и числа прямо сопоставимы с таблицей выбора
    ранга.
    """
    s = np.asarray(stat, np.float64)
    base = finish(s[VARIANTS.index("draft")])
    out = {}
    for i, v in enumerate(VARIANTS):
        e = finish(s[i])
        out[v] = dict(
            pos=float(1.0 - e["pos"] / base["pos"]) if base["pos"] > 0 else None,
            rot=float(1.0 - e["rot"] / base["rot"]) if base["rot"] > 0 else None,
            grip=e["grip"], err_pos=e["pos"], err_rot=e["rot"])
    return out


# --------------------------------------------------------------------------
# гребневая регрессия из грамиана
# --------------------------------------------------------------------------
def ridge_from_gram(G, C, idx, lam, bias_i):
    """Решение через нормальные уравнения на СТАНДАРТИЗОВАННЫХ признаках.

    Штраф `lam` без стандартизации означал бы разное для `h24` и `z0`: у них
    разный масштаб, и одна и та же lam давила бы один вход сильнее другого.
    Тогда сравнение вариантов было бы сравнением силы штрафа, а не входов.
    Свободный член не штрафуется.

    Средние и вторые моменты берутся ИЗ грамиана (столбец единиц), поэтому
    второго прохода по данным не нужно.
    """
    idx = list(idx)
    n = float(G[bias_i, bias_i])
    if n <= 1.0:
        raise ValueError("в грамиане меньше двух строк")
    sub = G[np.ix_(idx, idx)] / n
    mean = G[idx, bias_i] / n
    M = sub - np.outer(mean, mean)
    var = np.clip(np.diag(M), 0.0, None)
    std = np.sqrt(var)
    dead = std <= 1e-12
    std = np.where(dead, 1.0, std)
    Ms = M / np.outer(std, std)
    Ms[dead, :] = 0.0
    Ms[:, dead] = 0.0
    ybar = C[bias_i] / n
    Cc = C[idx] / n - np.outer(mean, ybar)
    Cs = Cc / std[:, None]
    Cs[dead, :] = 0.0
    A = Ms + float(lam) * np.eye(len(idx))
    w_s = np.linalg.solve(A, Cs)
    w = w_s / std[:, None]
    w[dead, :] = 0.0
    b = ybar - mean @ w
    return w, b


def ss_res_from_gram(G, C, S, idx, bias_i, w, b):
    """Сумма квадратов остатков на держанном наборе — тоже из грамиана.

    Позволяет перебрать все `lam` по ОДНОМУ проходу по данным: держанный
    набор входит только своими моментами.
    """
    idx = list(idx)
    W = np.vstack([np.asarray(w, np.float64), np.asarray(b, np.float64)[None]])
    ii = idx + [bias_i]
    Ga = G[np.ix_(ii, ii)]
    Ca = C[ii]
    return np.einsum("ik,ij,jk->k", W, Ga, W) - 2.0 * np.einsum(
        "ik,ik->k", W, Ca) + np.asarray(S, np.float64)


def r2_from(ss_res, S, C, n, bias_i):
    """R^2 по каждой координате мишени, затем среднее.

    Общая сумма квадратов считается вокруг среднего ДЕРЖАННОГО набора, а не
    обучающего: иначе «объяснённая доля» включала бы сдвиг среднего.
    """
    ybar = C[bias_i] / n
    ss_tot = np.asarray(S, np.float64) - n * ybar ** 2
    ok = ss_tot > 1e-12
    r2 = np.where(ok, 1.0 - ss_res / np.where(ok, ss_tot, 1.0), np.nan)
    return r2, float(np.nanmean(r2))


# --------------------------------------------------------------------------
# бутстрап по эпизодам
# --------------------------------------------------------------------------
def boot_draws(S_ep, n_boot=N_BOOT, seed=BOOT_SEED):
    """Суммы по бутстрап-выборкам ЭПИЗОДОВ, а не наблюдений.

    Возвращает массив (n_boot, ...) сумм в той же раскладке, что S_ep без
    первой оси. Один и тот же розыгрыш используется для ВСЕХ вариантов, что
    делает сравнение парным: разброс общей выборки эпизодов не попадает в
    разницу.
    """
    S = np.asarray(S_ep, np.float64)
    n_ep = S.shape[0]
    if n_ep < 2:
        raise ValueError("для бутстрапа нужно хотя бы два эпизода")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_ep, size=(int(n_boot), n_ep))
    return S[idx].sum(axis=1)


def ci(vals, lo=2.5, hi=97.5):
    v = np.asarray(vals, np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (None, None)
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi)))


# --------------------------------------------------------------------------
# пре-регистрированное правило маршрутизации
# --------------------------------------------------------------------------
def read_routing(d_pos_ci, d_rot_ci, d_grip_ci, tol=GRIP_TOL):
    """АСИММЕТРИЧНОЕ правило: подтверждает источник или отправляет дальше.

    Условие продолжения: нижняя граница 95% интервала парной разницы
    (both - z0) строго выше нуля И по положению, И по вращению, а ВЕРХНЯЯ
    граница парной разницы по знаку схвата не выше `tol`.

    СХВАТ ПРОВЕРЯЕТСЯ ПО ИНТЕРВАЛУ, А НЕ ПО ТОЧЕЧНОЙ РАЗНИЦЕ. Прежняя версия
    сравнивала с допуском только точечную оценку: разница 0.4 п.п. проходила
    бы и при верхней границе интервала 1.2 п.п., то есть жёсткое условие
    было бы жёстким только на словах. Позиция и вращение требуют, чтобы
    преимущество БЫЛО; схват требует, чтобы ухудшения НЕ БЫЛО, — поэтому у
    первых берётся нижняя граница, у второго верхняя.

    Отрицательный исход НЕ является стоп-гейтом. Линейная модель не обязана
    видеть то, что видит трёхслойная голова с tanh, поэтому отсутствие
    линейного преимущества переводит к нелинейному зонду или к парному кэшу
    считываний. Закрывать ветвь по одному линейному зонду ЗАПРЕЩЕНО.
    """
    lo_p, lo_r = d_pos_ci[0], d_rot_ci[0]
    hi_g = d_grip_ci[1]
    # Допуск сравнивается с запасом в 1e-12: разность двух долей, равная
    # допуску точно, в двоичной записи оказывается больше него, и решение
    # переворачивалось бы на шуме представления, а не на данных.
    grip_ok = hi_g is not None and hi_g <= tol + 1e-12
    if lo_p is not None and lo_r is not None and lo_p > 0 and lo_r > 0 \
            and grip_ok:
        return True, (
            "ПРОДОЛЖАЕМ С ТЕКУЩИМ ИСТОЧНИКОМ: h24 добавляет к z0 и по "
            "положению, и по вращению (нижние границы интервалов парной "
            f"разницы {lo_p:+.3f} и {lo_r:+.3f} выше нуля), ВЕРХНЯЯ граница "
            f"разницы по знаку схвата {hi_g:+.4f} не выше допуска "
            f"{tol:.1%}. Поздняя ветвь несёт сведения, которых нет в "
            "черновике")
    why = []
    if lo_p is None or lo_p <= 0:
        why.append("по положению нижняя граница не выше нуля")
    if lo_r is None or lo_r <= 0:
        why.append("по вращению нижняя граница не выше нуля")
    if not grip_ok:
        why.append("верхняя граница разницы по знаку схвата "
                   + ("не определена" if hi_g is None else f"{hi_g:+.4f}")
                   + f" выше допуска {tol:.1%}")
    return False, (
        f"ЛИНЕЙНОГО ПРЕИМУЩЕСТВА НЕТ ({'; '.join(why)}). Это НЕ повод "
        "закрывать ветвь: линейная модель не обязана видеть то, что видит "
        "голова с tanh. Следующий шаг — нелинейный зонд той же ёмкости, что "
        "у головы, или небольшой парный кэш считываний. Вывод «h24 "
        "бесполезен» по одному линейному зонду делать НЕЛЬЗЯ")


def read_source(gain_both, gain_oracle):
    """Доля от ОГРАНИЧЕННОГО оракула, а не от полного восстановления."""
    out = {}
    for k in ("pos", "rot"):
        a, b = gain_both.get(k), gain_oracle.get(k)
        out[k] = None if (a is None or b is None or abs(b) < 1e-9) \
            else float(a / b)
    return out


# --------------------------------------------------------------------------
def check_stamp(stamp, arrays, res_norm_sha, tap, cache_meta_sha):
    """Зонд обязан читать РОВНО те файлы, на которых K-11b сверил вход.

    Тождество проверяет ВЫХОД и при dz == 0 от h24 не зависит вовсе. Единая
    точка, где подтверждается правильность позднего ВХОДА, — сверка живого
    прохода с кэшем в K-11b. Она действительна только для файлов, лежавших
    на диске в тот момент, поэтому здесь требуется совпадение их отпечатков
    и sha `res_norm`. Без этого зонд мог бы исследовать другой отвод или
    другую норму, а вывод об источнике относился бы к другому входу.
    """
    need = ("res_norm_sha1", "arrays", "tap", "cache_meta_sha1")
    miss = [k for k in need if stamp.get(k) is None]
    if miss:
        raise SystemExit(
            f"в отпечатках нет полей {miss}: файл собран версией K-11b, "
            f"которая не сверяла живой проход с кэшем. Перезапустите K-11b")
    if int(stamp["tap"]) != int(tap):
        raise SystemExit(f"отпечатки для отвода {stamp['tap']}, а зонд "
                         f"читает {tap}")
    if stamp["res_norm_sha1"] != res_norm_sha:
        raise SystemExit(
            f"res_norm sha {res_norm_sha}, а тождество подтверждено на "
            f"{stamp['res_norm_sha1']}: голова видела бы другую норму")
    if stamp["cache_meta_sha1"] != cache_meta_sha:
        raise SystemExit("meta кэша изменилась после подтверждения входа")
    bad = [k for k, v in sorted(arrays.items())
           if stamp["arrays"].get(k) != v]
    if bad:
        raise SystemExit(
            f"массивы {bad} изменились после подтверждения входа в K-11b: "
            f"зонд читал бы не то, что было сверено с живым проходом")
    lv = stamp.get("live_vs_cache_normed") or {}
    if not lv.get("ok"):
        raise SystemExit(
            "в отпечатках вход головы НЕ подтверждён сверкой с живым "
            "проходом: запускать зонд не на чем")
    return True


def bucket_edges(frac, n_bucket=3):
    """Границы по ДОЛЕ совпавших позиций черновика, считаются на TRAIN.

    Наблюдение с полностью верным q0 практически не встречается: согласие с
    истинным грубым кодом измерено на уровне позиций, а не наблюдений.
    Поэтому «верный или неверный черновик» огрубляется до терцилей доли
    совпадений, и границы берутся с train, чтобы val на них не влиял.
    """
    q = np.linspace(0.0, 100.0, n_bucket + 1)[1:-1]
    return np.percentile(np.asarray(frac, np.float64), q)


def assign_bucket(frac, edges):
    return np.searchsorted(np.asarray(edges, np.float64),
                           np.asarray(frac, np.float64), side="right")


# --------------------------------------------------------------------------
def _integration(h_matters, seed=0, n_ep=40, per_ep=30, d_h=8, d_z=6, rank=4):
    """Весь конвейер на синтетике с ИЗВЕСТНЫМ ответом.

    Отдельные части проверены поштучно, но собранными они могут давать
    неверный вывод: например правило маршрутизации получало бы разницу,
    посчитанную по разным розыгрышам, и «преимущество» возникало бы из
    разброса выборки. Здесь связь задана явно — при `h_matters=True` остаток
    зависит от h, иначе не зависит вовсе, — и конвейер обязан ответить
    соответственно. Возвращает исход правила и парные разницы.
    """
    rng = np.random.default_rng(seed)
    n = n_ep * per_ep
    ep = np.repeat(np.arange(n_ep), per_ep)
    hh = rng.normal(size=(n, N_POS, d_h))
    zz = rng.normal(size=(n, N_POS, d_z))
    Az = rng.normal(size=(d_z, rank))
    Ah = rng.normal(size=(d_h, rank)) * (1.0 if h_matters else 0.0)
    a = zz @ Az + hh @ Ah + 0.1 * rng.normal(size=(n, N_POS, rank))
    rho_ = np.percentile(np.abs(a).reshape(-1, rank), 95.0, axis=0)
    a_cl = np.clip(a, -rho_, rho_)
    # «Декодер»: фиксированное линейное отображение коэффициентов в действия.
    M = rng.normal(size=(rank, H_EXEC * 7)) / np.sqrt(rank)

    def dec(c):
        return (c.sum(1) @ M).reshape(-1, H_EXEC, 7)

    X = np.concatenate([hh, zz, np.ones_like(zz[..., :1])], -1)
    d_all = d_h + d_z + 1
    bias_i = d_all - 1
    idx = dict(z0=list(range(d_h, d_h + d_z)), both=list(range(d_h + d_z)))
    tr = ep < n_ep // 2
    va = ~tr
    out = {}
    for part, mask in (("tr", tr), ("va", va)):
        Xf = X[mask].reshape(-1, d_all)
        Y = (a_cl[mask] / rho_).reshape(-1, rank)
        out[part] = dict(G=Xf.T @ Xf, C=Xf.T @ Y, S=(Y ** 2).sum(0))
    S_ep = np.zeros((n_ep // 2, len(VARIANTS), N_STAT))
    ref = dec(a[va])
    pred_dec = dict(draft=dec(np.zeros_like(a[va])), oracle=dec(a_cl[va]))
    for v in ("z0", "both"):
        w, b = ridge_from_gram(out["tr"]["G"], out["tr"]["C"], idx[v], 1e-4,
                               bias_i)
        p = np.clip(X[va].reshape(-1, d_all)[:, idx[v]] @ w + b, -1.0, 1.0)
        pred_dec[v] = dec(p.reshape(-1, N_POS, rank) * rho_)
    pred_dec["h24"] = pred_dec["z0"]
    ev = ep[va] - ep[va].min()
    for vi, v in enumerate(VARIANTS):
        per = err_per_obs(pred_dec[v], ref)
        np.add.at(S_ep, (ev, vi), per)
    draws = boot_draws(S_ep, n_boot=300, seed=7)
    gb = [gains_from(x) for x in draws]
    dp = ci([g["both"]["pos"] - g["z0"]["pos"] for g in gb])
    dr = ci([g["both"]["rot"] - g["z0"]["rot"] for g in gb])
    dg = ci([g["both"]["grip"] - g["z0"]["grip"] for g in gb])
    g_all = gains_from(S_ep.sum(0))
    ok, _ = read_routing(dp, dr, dg)
    return ok, g_all, dp, dr


def selftest():
    # --- суммы по наблюдениям сходятся с диагностикой ----------------------
    rng = np.random.default_rng(0)
    a = rng.normal(size=(7, 20, 7))
    r = rng.normal(size=(7, 20, 7))
    per = err_per_obs(a, r)
    ref = k11a.err_sums(a, r)
    got = per.sum(0)
    for i, k in enumerate(STAT_KEYS):
        assert abs(got[i] - ref[k]) < 1e-8 * max(1.0, abs(ref[k])), (k, got[i],
                                                                    ref[k])
    # контроль: подмена опоры обязана менять суммы, иначе проверка пустая
    assert abs(err_per_obs(a, r + 1.0).sum(0)[0] - got[0]) > 1e-6
    # разбиение по строкам обязано давать ту же сумму, что целиком
    assert np.allclose(per[:3].sum(0) + per[3:].sum(0), got)
    # шаги после H_EXEC не влияют: срез именно по шагам чанка
    a2 = a.copy()
    a2[:, H_EXEC:] += 100.0
    assert np.allclose(err_per_obs(a2, r).sum(0), got)

    # --- гребень восстанавливает известную линейную связь -------------------
    n, d, k = 4000, 6, 3
    X = rng.normal(size=(n, d)) * np.array([1.0, 10.0, 0.1, 5.0, 1.0, 2.0])
    Wt = rng.normal(size=(d, k))
    bt = rng.normal(size=k)
    Y = X @ Wt + bt
    Xa = np.hstack([X, np.ones((n, 1))])
    G = Xa.T @ Xa
    C = Xa.T @ Y
    S = (Y ** 2).sum(0)
    w, b = ridge_from_gram(G, C, list(range(d)), 1e-10, d)
    assert np.abs(w - Wt).max() < 1e-4, np.abs(w - Wt).max()
    assert np.abs(b - bt).max() < 1e-4
    # контроль: только часть признаков НЕ восстанавливает связь целиком
    w2, b2 = ridge_from_gram(G, C, [0, 1], 1e-10, d)
    assert np.abs(w2 - Wt[:2]).max() > 1e-2
    # штраф обязан сжимать: разные lam дают разные веса
    w3, _ = ridge_from_gram(G, C, list(range(d)), 10.0, d)
    assert np.abs(w3).max() < np.abs(w).max()

    # --- остаток из грамиана равен прямому ---------------------------------
    ssr = ss_res_from_gram(G, C, S, list(range(d)), d, w, b)
    direct = ((X @ w + b - Y) ** 2).sum(0)
    assert np.abs(ssr - direct).max() < 1e-4 * max(1.0, direct.max())
    # и для НЕоптимальных весов тоже — иначе проверка ловила бы только ноль
    wj = w + 0.3
    ssr2 = ss_res_from_gram(G, C, S, list(range(d)), d, wj, b)
    direct2 = ((X @ wj + b - Y) ** 2).sum(0)
    assert np.abs(ssr2 - direct2).max() < 1e-3 * max(1.0, direct2.max())

    # --- R^2 = 1 при точной подгонке и 0 при предсказании среднего ---------
    r2v, r2m = r2_from(ssr, S, C, n, d)
    assert r2m > 0.999999, r2m
    w0 = np.zeros((d, k))
    b0 = C[d] / n
    ss0 = ss_res_from_gram(G, C, S, list(range(d)), d, w0, b0)
    _, r2z = r2_from(ss0, S, C, n, d)
    assert abs(r2z) < 1e-8, r2z

    # --- накопление грамиана не зависит от размера батча --------------------
    G2 = np.zeros_like(G)
    for i, j in k11a.plan_batches(n, 137):
        G2 += Xa[i:j].T @ Xa[i:j]
    assert np.abs(G2 - G).max() < 1e-6 * max(1.0, np.abs(G).max())

    # --- доли улучшения ----------------------------------------------------
    st = np.zeros((len(VARIANTS), N_STAT))
    st[:, 1] = st[:, 3] = st[:, 5] = 100.0
    st[VARIANTS.index("draft"), 0] = 4.0 * 100.0     # rms 2
    st[VARIANTS.index("draft"), 2] = 4.0 * 100.0
    st[VARIANTS.index("oracle")] [0] = 0.0            # rms 0 -> доля 1
    st[VARIANTS.index("both"), 0] = 1.0 * 100.0       # rms 1 -> доля 0.5
    st[VARIANTS.index("both"), 2] = 1.0 * 100.0
    st[VARIANTS.index("z0"), 0] = 4.0 * 100.0         # как черновик -> 0
    st[VARIANTS.index("z0"), 2] = 4.0 * 100.0
    g = gains_from(st)
    assert abs(g["draft"]["pos"]) < 1e-12
    assert abs(g["oracle"]["pos"] - 1.0) < 1e-12
    assert abs(g["both"]["pos"] - 0.5) < 1e-12
    assert abs(g["z0"]["pos"]) < 1e-12
    # доля от ограниченного оракула
    src = read_source(g["both"], g["oracle"])
    assert abs(src["pos"] - 0.5) < 1e-12

    # --- бутстрап по эпизодам ----------------------------------------------
    S_ep = np.repeat(st[None], 40, axis=0) / 40.0
    dr = boot_draws(S_ep, n_boot=200, seed=1)
    gs = np.array([gains_from(x)["both"]["pos"] for x in dr])
    assert gs.std() < 1e-9, "одинаковые эпизоды обязаны давать нулевой разброс"
    # контроль: один выбивающийся эпизод обязан расширить интервал
    S_bad = S_ep.copy()
    S_bad[0, VARIANTS.index("both"), 0] *= 50.0
    dr2 = boot_draws(S_bad, n_boot=200, seed=1)
    gs2 = np.array([gains_from(x)["both"]["pos"] for x in dr2])
    assert gs2.std() > 1e-3, "бутстрап не заметил выбросового эпизода"
    lo, hi = ci(gs2)
    assert lo < hi

    # --- правило маршрутизации ---------------------------------------------
    ok, txt = read_routing((0.02, 0.09), (0.03, 0.11), (-0.004, 0.002))
    assert ok and "ПРОДОЛЖАЕМ" in txt
    ok, txt = read_routing((-0.01, 0.09), (0.03, 0.11), (-0.004, 0.002))
    assert not ok and "НЕ повод" in txt and "закрывать" in txt
    # схват хуже допуска перевешивает выигрыш в позе — отдельное жёсткое условие
    ok, txt = read_routing((0.02, 0.09), (0.03, 0.11), (0.001, 0.022))
    assert not ok and "схват" in txt
    # ГЛАВНОЕ: точечная разница внутри допуска, а ВЕРХНЯЯ граница вне его.
    # Прежняя версия правила это пропускала, потому что смотрела на точку.
    ok, txt = read_routing((0.02, 0.09), (0.03, 0.11), (-0.001, 0.012))
    assert not ok and "верхняя граница" in txt.lower(), txt
    # ровно на допуске — ещё проходит
    ok, _ = read_routing((0.02, 0.09), (0.03, 0.11), (-0.002, 0.005))
    assert ok
    # интервал не определён — отказ, а не пропуск
    ok, txt = read_routing((0.02, 0.09), (0.03, 0.11), (None, None))
    assert not ok and "не определена" in txt
    for bad in (read_routing((-0.01, 0.0), (-0.02, 0.0), (0.0, 0.1))[1],):
        assert "закрывать ветвь" in bad or "закрывать" in bad
        assert "бесполезен" in bad  # правило обязано называть запрещённый вывод

    # --- отпечатки входа ----------------------------------------------------
    good_stamp = dict(res_norm_sha1="aa11", tap=24, cache_meta_sha1="mm",
                      arrays=dict(h24="h1", q0hat="q1", ktrue="k1",
                                  split="s1"),
                      live_vs_cache_normed=dict(ok=True))
    assert check_stamp(good_stamp, dict(h24="h1", q0hat="q1"), "aa11", 24, "mm")
    for kw, msg in ((dict(res_norm_sha="bb22"), "норма"),
                    (dict(tap=18), "отвод"),
                    (dict(cache_meta_sha="zz"), "meta"),
                    (dict(arrays=dict(h24="ДРУГОЙ")), "массив")):
        a = dict(arrays=dict(h24="h1", q0hat="q1"), res_norm_sha="aa11",
                 tap=24, cache_meta_sha="mm")
        a.update(kw)
        try:
            check_stamp(good_stamp, a["arrays"], a["res_norm_sha"], a["tap"],
                        a["cache_meta_sha"])
        except SystemExit:
            pass
        else:
            raise AssertionError(f"подмена принята: {msg}")
    # неподтверждённый вход — отказ, даже если все хеши сошлись
    bad_stamp = dict(good_stamp, live_vs_cache_normed=dict(ok=False))
    try:
        check_stamp(bad_stamp, dict(h24="h1"), "aa11", 24, "mm")
    except SystemExit:
        pass
    else:
        raise AssertionError("неподтверждённый вход принят")
    # отпечатки старой версии K-11b — отказ, а не пропуск
    try:
        check_stamp({k: v for k, v in good_stamp.items()
                     if k != "res_norm_sha1"}, dict(h24="h1"), "aa11", 24,
                    "mm")
    except SystemExit:
        pass
    else:
        raise AssertionError("отпечатки без sha нормы приняты")

    # --- корзины по доле совпадений q0 -------------------------------------
    tr_frac = np.linspace(0.0, 1.0, 900)
    ed = bucket_edges(tr_frac, 3)
    bk = assign_bucket(np.array([0.0, 0.5, 1.0]), ed)
    assert bk.tolist() == [0, 1, 2], bk
    assert len(ed) == 2 and ed[0] < ed[1]
    # границы с train не двигаются от состава val
    ed2 = bucket_edges(tr_frac, 3)
    assert np.allclose(ed, ed2)

    # --- сквозная проверка конвейера на известном ответе --------------------
    ok_yes, g_yes, dp_yes, _ = _integration(True, seed=3)
    assert ok_yes, ("конвейер не увидел связи, которая заложена явно: "
                    f"разница {dp_yes}")
    assert g_yes["both"]["pos"] > g_yes["z0"]["pos"] + 0.05, g_yes
    # КОНТРОЛЬ: остаток не зависит от h — преимущества быть НЕ должно.
    # Без этой половины проверка ловила бы только «что-то нашлось».
    ok_no, g_no, dp_no, _ = _integration(False, seed=3)
    assert not ok_no, ("конвейер нашёл преимущество там, где связи нет: "
                       f"разница {dp_no}, доли {g_no['both']}, {g_no['z0']}")
    assert abs(g_no["both"]["pos"] - g_no["z0"]["pos"]) < 0.05, g_no
    # и оракул обязан быть верхней границей в обоих случаях
    for g_ in (g_yes, g_no):
        assert g_["oracle"]["pos"] >= g_["both"]["pos"] - 1e-9, g_

    print("самопроверка k11p пройдена (версия «вход сверяется с отпечатками "
          "K-11b, схват по интервалу»): "
          "суммы по наблюдениям сходятся с диагностикой и не зависят от "
          "разбиения, гребень восстанавливает известную связь и не "
          "восстанавливает её по части признаков, остаток и R^2 считаются из "
          "грамиана и совпадают с прямым счётом на неоптимальных весах, "
          "накопление не зависит от размера батча, доля улучшения равна нулю "
          "у черновика и единице у оракула, бутстрап по эпизодам замечает "
          "выбросовый эпизод, правило маршрутизации требует ВЕРХНЮЮ границу "
          "разницы по схвату и запрещает закрывать ветвь, отпечатки входа "
          "отвергают подмену нормы, отвода, meta и массивов, а собранный "
          "конвейер находит заложенную связь и НЕ находит её там, где её "
          "нет")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", help="префикс кэша K-11a")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gram-batch", type=int, default=512,
                    help="наблюдений на шаг накопления грамиана")
    ap.add_argument("--train-n", type=int, default=0,
                    help="ограничить train (0 — весь); выборка случайная по "
                         "ЭПИЗОДАМ, а не префикс")
    ap.add_argument("--val-n", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--buckets", type=int, default=3)
    ap.add_argument("--allow-module-drift", action="store_true")
    ap.add_argument("--out", default="data/k11p_probe.json")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()

    for f in ("cache", "out"):
        if getattr(args, f):
            setattr(args, f, os.path.abspath(getattr(args, f)))
    args.root = os.path.abspath(args.root)
    sys.path.insert(0, args.root)
    sha = k11a.file_sha1(__file__)
    print(f"k11p sha1 {sha}")
    for need, why in ((args.ckpt, "--ckpt"), (args.cache, "--cache")):
        if not need:
            raise SystemExit(f"нужен {why}")

    import torch
    import actioncodec  # noqa: F401  регистрирует action_codec
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if "action_codec" not in CONFIG_MAPPING:
        raise SystemExit("тип «action_codec» не зарегистрирован")
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import get_cfg, seed_everything
    from joint12_vla import make_joint12_class
    import joint12_vla as jv
    import hicora_vla as hv

    # ДОСТУПНОСТЬ GPU ПРОВЕРЯЕТСЯ ПЕРВОЙ: контейнер периодически теряет карты
    # при живом драйвере, и падение наступало бы уже после сбора моментов.
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                "torch.cuda.is_available() == False: контейнер потерял "
                "видеокарты. Проверьте nvidia-smi; лечится перезапуском "
                "докера с хоста")
        gi = int(args.device.split(":")[1]) if ":" in args.device else 0
        if gi >= torch.cuda.device_count():
            raise SystemExit(f"запрошено {args.device}, а видно "
                             f"{torch.cuda.device_count()} устройств")
        free_b, total_b = torch.cuda.mem_get_info(gi)
        print(f"  {args.device}: свободно {free_b / 2 ** 30:.1f} ГиБ из "
              f"{total_b / 2 ** 30:.1f}")
    seed_everything(0)

    prefix = args.cache
    meta = json.load(open(prefix + ".meta.json"))
    diag = json.load(open(prefix + ".diag.json"))

    # --- АРТЕФАКТЫ ПРИВЯЗЫВАЮТСЯ К ДИАГНОСТИКЕ ДО ЗАГРУЗКИ МОДЕЛИ ----------
    basis_p, rho_p = prefix + ".basis.npy", prefix + ".rho.npy"
    for p_ in (basis_p, rho_p):
        if not os.path.exists(p_):
            raise SystemExit(f"нет {p_}: сначала диагностика K-11a")
    B = np.load(basis_p).astype(np.float64)
    rho = np.load(rho_p).astype(np.float64)
    k11b.check_artifacts(diag, prefix, k11a.file_sha1(basis_p),
                         k11a.file_sha1(rho_p), B.shape[1],
                         k11a.file_sha1(prefix + ".meta.json"))
    # РАСХОЖДЕНИЕ ВЕРСИЙ МОДУЛЕЙ НЕ ПРОГЛАТЫВАЕТСЯ МОЛЧА. Прежде возвращённый
    # словарь расхождений никуда не шёл: флаг `--allow-module-drift` разрешал
    # дрейф И СКРЫВАЛ его, то есть отчёт не отличался от прогона без дрейфа.
    mod_drift = k11b.check_cache_fields(
        meta, diag, dict(ckpt=args.ckpt),
        dict(hicora_vla_sha1=k11a.file_sha1(hv.__file__),
             joint12_vla_sha1=k11a.file_sha1(jv.__file__)),
        allow_drift=args.allow_module_drift)
    if mod_drift:
        print(f"  ВНИМАНИЕ: версии модулей разошлись с кэшем: {mod_drift}. "
              f"Разрешено флагом; расхождение записано в отчёт")
    if len(rho) != B.shape[1]:
        raise SystemExit(f"rho длины {len(rho)} против ранга {B.shape[1]}")
    # ОРТОНОРМАЛЬНОСТЬ ПРОВЕРЯЕТСЯ ЗДЕСЬ, А НЕ ПРИНИМАЕТСЯ НА ВЕРУ: только с
    # ортонормальным базисом rho ограничивает норму поправки, а без этого
    # «в пределах rho» ничего не значит.
    dev_i = float(np.abs(B.T @ B - np.eye(B.shape[1])).max())
    if dev_i > 1e-4:
        raise SystemExit(f"базис не ортонормален: max|B^T B - I| = {dev_i:.2e}")
    rank = int(B.shape[1])
    print(f"  ранг {rank}, ||rho|| = {float(np.linalg.norm(rho)):.4f}, "
          f"max|B^T B - I| = {dev_i:.1e}; артефакты привязаны к диагностике")
    ds_repo, ds_rev = k11b.dataset_source(meta)
    print(f"  данные: {ds_repo}@{ds_rev} (из meta)")
    stamp_p = prefix + ".artifacts.json"
    if not os.path.exists(stamp_p):
        raise SystemExit(
            f"нет {stamp_p}: правильность ПОЗДНЕГО ВХОДА не подтверждена. "
            f"Тождество проверяет выход, а при dz == 0 выход от h24 не "
            f"зависит вовсе — запустите K-11b текущей версии, он сверит "
            f"живой проход с кэшем и запишет отпечатки")
    stamp = json.load(open(stamp_p))

    dev = torch.device(args.device)
    dt = getattr(torch, args.dtype)

    # --- ИСХОДНАЯ ФИНАЛЬНАЯ НОРМА ------------------------------------------
    # Голова читает h24 ПОСЛЕ неё, поэтому зонд обязан видеть тот же вход.
    # Дрейф 0.883 из аудита K-11a измерен ДО нормы и величиной дрейфа ПОСЛЕ
    # неё не является; связи между ними никто не мерил, и предполагать её
    # отсутствие оснований ровно столько же, сколько предполагать наличие.
    # Модель снимается сразу после извлечения нормы: две модели в памяти
    # одновременно уже стоили нам падения по памяти.
    cfg = get_cfg(os.path.join(args.root, args.cfg_path))
    # ЧЕКПОЙНТ ПОДСТАВЛЯЕТСЯ В cfg, А НЕ ЧИТАЕТСЯ ИЗ yaml. Без этих двух
    # строк `from_pretrained(**cfg.MODEL.vlm.kwargs)` поднимает модель,
    # прописанную в конфиге, и норма была бы снята с ДРУГОГО чекпойнта —
    # молча, потому что форма и имена совпали бы.
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt
    Cls = make_joint12_class(SmolVLABlockwiseAR)
    m0 = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    m0.init_joint_fast(depth=int(meta["depth"]), head_dtype=dt)
    res_norm = copy.deepcopy(m0.action_expert.norm).to(dev).eval()
    for p_ in res_norm.parameters():
        p_.requires_grad_(False)
    rn_sha = k11a.state_sha1(res_norm)
    del m0
    torch.cuda.empty_cache()
    print(f"  res_norm снята с исходного чекпойнта, sha {rn_sha}; модель "
          f"выгружена до тяжёлой части")

    # ОТПЕЧАТКИ СВЕРЯЮТСЯ, А НЕ ПРОСТО ПЕЧАТАЮТСЯ. Прежде sha нормы только
    # записывалась в отчёт: другая норма прошла бы весь зонд, и вывод об
    # источнике относился бы к входу, которого HiCoRA не увидит.
    tap_st = max(meta["saved_taps"])
    arr_sha = {f"h{tap_st}": k11a.file_sha1(f"{prefix}.h{tap_st}.npy")}
    for nm in ("q0hat", "ktrue", "split", "codebooks"):
        p_ = f"{prefix}.{nm}.npy"
        if os.path.exists(p_):
            arr_sha[nm] = k11a.file_sha1(p_)
    check_stamp(stamp, arr_sha, rn_sha, tap_st,
                k11a.file_sha1(prefix + ".meta.json"))
    lv = stamp["live_vs_cache_normed"]
    print(f"  вход подтверждён K-11b ({stamp.get('script_sha1')}): невязка "
          f"живого прохода против кэша после нормы {lv['rel']:.2e}, "
          f"контроль {lv['rel_pos_shift']:.2e}; отпечатки {len(arr_sha)} "
          f"массивов совпали")

    _, codec, E, _ = k11a.load_codec(args)
    Esav = np.load(prefix + ".codebooks.npy")
    Ecur = E.cpu().numpy()
    if Esav.shape != Ecur.shape or float(np.abs(Esav - Ecur).max()) > 1e-4:
        raise SystemExit("кодовые книги разошлись с кэшем: другой кодек")
    k11a.check_fingerprints(meta, dict(
        codebooks_sha1=hashlib.sha1(np.ascontiguousarray(
            Esav.astype(np.float32)).tobytes()).hexdigest()[:12],
        decoder_probe=k11a.decoder_probe(codec, E, dev),
        codec_state_sha1=k11a.state_sha1(codec)))
    print("  книги и декодер сверены с кэшем")

    # --- ДАННЫЕ -------------------------------------------------------------
    q0 = np.load(prefix + ".q0hat.npy")
    Kt = np.load(prefix + ".ktrue.npy")
    split = np.load(prefix + ".split.npy", allow_pickle=True).astype(str)
    tap = tap_st
    H = np.load(f"{prefix}.h{tap}.npy", mmap_mode="r")
    d_cache = np.load(meta["cache"], allow_pickle=True)
    epi = np.asarray(d_cache["episode"]).astype(np.int64)
    if len(epi) != len(q0) or H.shape[0] != len(q0):
        raise SystemExit(f"длины не сходятся: эпизодов {len(epi)}, кодов "
                         f"{len(q0)}, отвод {H.shape}")
    D_H, D_Z = int(H.shape[2]), int(E.shape[-1])
    tr = np.where(split == "train")[0]
    va = np.where(split == "val")[0]
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit("нет train или val")
    # ЭПИЗОДЫ TRAIN И VAL НЕ ПЕРЕСЕКАЮТСЯ — иначе зонд подсматривал бы.
    inter = np.intersect1d(np.unique(epi[tr]), np.unique(epi[va]))
    if len(inter):
        raise SystemExit(f"эпизоды {inter[:5]} есть и в train, и в val")

    def sub_by_episode(ids, limit, seed):
        """Ограничение выборки — по ЭПИЗОДАМ и случайно, а не префиксом.

        Кэш упорядочен по эпизодам, поэтому `ids[:n]` — это первые задачи.
        """
        if not limit or len(ids) <= limit:
            return ids
        eps = np.unique(epi[ids])
        rng = np.random.default_rng(seed)
        keep, take = set(), 0
        for e in rng.permutation(eps):
            keep.add(int(e))
            take += int((epi[ids] == e).sum())
            if take >= limit:
                break
        sel = ids[np.isin(epi[ids], list(keep))]
        print(f"    выборка {len(sel)} из {len(ids)} по {len(keep)} эпизодам "
              f"(сид {seed}), не префикс")
        return sel

    tr = sub_by_episode(tr, args.train_n, 31)
    va = sub_by_episode(va, args.val_n, 32)
    print(f"зонд: train {len(tr)}, val {len(va)}, отвод h{tap}, d_h={D_H}, "
          f"d_z={D_Z}, ранг {rank}")

    # внутреннее разбиение train для выбора lam — тоже по эпизодам
    tr_eps = np.unique(epi[tr])
    rng_in = np.random.default_rng(INNER_SEED)
    perm = rng_in.permutation(tr_eps)
    n_iv = max(1, int(round(len(perm) * INNER_FRAC)))
    iv_eps = set(int(x) for x in perm[:n_iv])
    is_iv = np.isin(epi[tr], list(iv_eps))
    print(f"  внутреннее разбиение для выбора lam: {len(tr_eps) - n_iv} "
          f"эпизодов на подгонку, {n_iv} на выбор (сид {INNER_SEED})")

    Bt = torch.as_tensor(B, dtype=torch.float32, device=dev)
    rho_t = torch.as_tensor(rho, dtype=torch.float32, device=dev)
    Et = E

    def z_of(codes0, all_levels=None):
        z = Et[0][torch.as_tensor(np.asarray(codes0)).long().to(dev)]
        if all_levels is not None:
            k = torch.as_tensor(np.asarray(all_levels)).long().to(dev)
            for l in range(1, Et.shape[0]):
                z = z + Et[l][k[:, l, :]]
        return z

    def feats_and_targets(sel):
        """[res_norm(h24), z0, 1] и обе мишени для набора наблюдений."""
        hb = torch.from_numpy(np.asarray(H[sel])).to(dev, dt)
        with torch.no_grad():
            hn = res_norm(hb).float()
            z0b = z_of(q0[sel])
            zsb = z_of(Kt[sel, 0, :], Kt[sel])
            a = (zsb - z0b) @ Bt
            t_clip = torch.clamp(a, -rho_t, rho_t) / rho_t
            X = torch.cat([hn, z0b,
                           torch.ones_like(z0b[..., :1])], dim=-1)
        return (X.reshape(-1, D_H + D_Z + 1).double(),
                a.reshape(-1, rank).double(),
                t_clip.reshape(-1, rank).double(), z0b, zsb, a)

    d_all = D_H + D_Z + 1
    bias_i = d_all - 1
    IDX = dict(h24=list(range(D_H)),
               z0=list(range(D_H, D_H + D_Z)),
               both=list(range(D_H + D_Z)))

    # --- ОДИН ПРОХОД ПО TRAIN: моменты для подгонки и для выбора lam --------
    acc = {p: dict(G=torch.zeros(d_all, d_all, dtype=torch.float64, device=dev),
                   Craw=torch.zeros(d_all, rank, dtype=torch.float64,
                                    device=dev),
                   Cclip=torch.zeros(d_all, rank, dtype=torch.float64,
                                     device=dev),
                   Sraw=torch.zeros(rank, dtype=torch.float64, device=dev),
                   Sclip=torch.zeros(rank, dtype=torch.float64, device=dev))
           for p in ("it", "iv")}
    n_done = 0
    for i, j in k11a.plan_batches(len(tr), args.gram_batch):
        sel = tr[i:j]
        X, y_raw, y_clip, *_ = feats_and_targets(sel)
        mask = np.repeat(is_iv[i:j], N_POS)
        for part, mm in (("iv", mask), ("it", ~mask)):
            if not mm.any():
                continue
            mt = torch.as_tensor(mm, device=dev)
            Xp, Yr, Yc = X[mt], y_raw[mt], y_clip[mt]
            a_ = acc[part]
            a_["G"] += Xp.T @ Xp
            a_["Craw"] += Xp.T @ Yr
            a_["Cclip"] += Xp.T @ Yc
            a_["Sraw"] += (Yr ** 2).sum(0)
            a_["Sclip"] += (Yc ** 2).sum(0)
        n_done += len(sel)
        if (i // max(args.gram_batch, 1)) % 20 == 0:
            print(f"    грамиан {n_done}/{len(tr)}", flush=True)
    A = {p: {k: v.cpu().numpy() for k, v in acc[p].items()} for p in acc}
    full = {k: A["it"][k] + A["iv"][k] for k in A["it"]}
    n_it, n_iv_rows = float(A["it"]["G"][bias_i, bias_i]), \
        float(A["iv"]["G"][bias_i, bias_i])
    print(f"  моменты собраны: {int(n_it)} строк на подгонку, "
          f"{int(n_iv_rows)} на выбор lam (строка = латентная позиция)")
    if n_it < d_all or n_iv_rows < 1:
        raise SystemExit("строк меньше, чем признаков: гребень выродится")

    # --- ВЫБОР lam НА ВНУТРЕННЕМ ДЕРЖАННОМ НАБОРЕ --------------------------
    print(f"\n  выбор lam на внутреннем держанном наборе (мишень головы, "
          f"{len(LAMBDAS)} значений):")
    best = {}
    for v in ("z0", "h24", "both"):
        sc = []
        for lam in LAMBDAS:
            w, b = ridge_from_gram(A["it"]["G"], A["it"]["Cclip"], IDX[v], lam,
                                   bias_i)
            ssr = ss_res_from_gram(A["iv"]["G"], A["iv"]["Cclip"],
                                   A["iv"]["Sclip"], IDX[v], bias_i, w, b)
            sc.append(float(ssr.sum() / n_iv_rows / rank))
        k_ = int(np.argmin(sc))
        best[v] = LAMBDAS[k_]
        edge = " (на КРАЮ сетки — расширить)" if k_ in (0, len(LAMBDAS) - 1) \
            else ""
        print(f"    {v:>5}: lam = {best[v]:.3g}, MSE = {sc[k_]:.6f}{edge}")

    # --- ПОДГОНКА НА ВСЁМ TRAIN --------------------------------------------
    W = {}
    for v in ("z0", "h24", "both"):
        W[v] = dict(
            clip=ridge_from_gram(full["G"], full["Cclip"], IDX[v], best[v],
                                 bias_i),
            raw=ridge_from_gram(full["G"], full["Craw"], IDX[v], best[v],
                                bias_i))

    # --- R^2 НА VAL, ОПИСАТЕЛЬНО -------------------------------------------
    accv = dict(G=torch.zeros(d_all, d_all, dtype=torch.float64, device=dev),
                Craw=torch.zeros(d_all, rank, dtype=torch.float64, device=dev),
                Cclip=torch.zeros(d_all, rank, dtype=torch.float64, device=dev),
                Sraw=torch.zeros(rank, dtype=torch.float64, device=dev),
                Sclip=torch.zeros(rank, dtype=torch.float64, device=dev))
    for i, j in k11a.plan_batches(len(va), args.gram_batch):
        X, y_raw, y_clip, *_ = feats_and_targets(va[i:j])
        accv["G"] += X.T @ X
        accv["Craw"] += X.T @ y_raw
        accv["Cclip"] += X.T @ y_clip
        accv["Sraw"] += (y_raw ** 2).sum(0)
        accv["Sclip"] += (y_clip ** 2).sum(0)
    V = {k: v.cpu().numpy() for k, v in accv.items()}
    n_va_rows = float(V["G"][bias_i, bias_i])
    r2 = {}
    print(f"\n  R^2 на val ({int(n_va_rows)} строк), ОПИСАТЕЛЬНО: "
          f"коэффициенты — не действия")
    print(f"    {'вход':>6}{'R2 сырой':>12}{'R2 мишень головы':>20}")
    for v in ("z0", "h24", "both"):
        rr = {}
        for tag, Ck, Sk in (("raw", "Craw", "Sraw"),
                            ("clip", "Cclip", "Sclip")):
            w, b = W[v][tag]
            ssr = ss_res_from_gram(V["G"], V[Ck], V[Sk], IDX[v], bias_i, w, b)
            _, rr[tag] = r2_from(ssr, V[Sk], V[Ck], n_va_rows, bias_i)
        r2[v] = rr
        print(f"    {v:>6}{rr['raw']:>12.4f}{rr['clip']:>20.4f}")

    # --- ОСНОВНАЯ ОЦЕНКА: ПОСЛЕ ДЕКОДЕРА, ПО ЭПИЗОДАМ ----------------------
    q_match = (q0 == Kt[:, 0, :]).mean(1)
    edges = bucket_edges(q_match[tr], args.buckets)
    bk_va = assign_bucket(q_match[va], edges)
    print(f"\n  корзины по доле совпавших позиций q0 (границы с TRAIN): "
          + ", ".join(f"{e:.3f}" for e in edges))
    for b_ in range(args.buckets):
        s_ = bk_va == b_
        if s_.any():
            print(f"    корзина {b_}: {int(s_.sum())} наблюдений val, доля "
                  f"совпадений {q_match[va][s_].mean():.3f}")

    ep_va = epi[va]
    ep_ids = np.unique(ep_va)
    ep_pos = {int(e): i for i, e in enumerate(ep_ids)}
    S_ep = np.zeros((len(ep_ids), args.buckets + 1, len(VARIANTS), N_STAT))

    Wt_ = {v: (torch.as_tensor(W[v]["clip"][0], dtype=torch.float32,
                               device=dev),
               torch.as_tensor(W[v]["clip"][1], dtype=torch.float32,
                               device=dev))
           for v in ("z0", "h24", "both")}
    col = {v: torch.as_tensor(IDX[v], device=dev) for v in IDX}

    for i, j in k11a.plan_batches(len(va), args.batch):
        sel = va[i:j]
        X, _, _, z0b, zsb, a = feats_and_targets(sel)
        nb = len(sel)
        Xf = X.float()
        with torch.no_grad():
            As = codec._decode(zsb, embodiment_ids=0)[0][..., :7].float()
            As_n = As.cpu().numpy()
            dec = {}
            dec["draft"] = codec._decode(
                z0b, embodiment_ids=0)[0][..., :7].float().cpu().numpy()
            c_or = torch.clamp(a, -rho_t, rho_t)
            dec["oracle"] = codec._decode(
                z0b + c_or @ Bt.T,
                embodiment_ids=0)[0][..., :7].float().cpu().numpy()
            for v in ("z0", "h24", "both"):
                w_, b_ = Wt_[v]
                pred = Xf.index_select(1, col[v]) @ w_ + b_
                # Голова физически способна выдать только rho*tanh(...),
                # поэтому предсказание зонда ограничивается тем же пределом:
                # иначе зонду разрешалось бы то, чего голова не может.
                c_hat = torch.clamp(pred, -1.0, 1.0).reshape(
                    nb, N_POS, rank) * rho_t
                dec[v] = codec._decode(
                    z0b + c_hat @ Bt.T,
                    embodiment_ids=0)[0][..., :7].float().cpu().numpy()
        bk = bk_va[i:j]
        for vi, v in enumerate(VARIANTS):
            per = err_per_obs(dec[v], As_n)
            for o in range(nb):
                pe = ep_pos[int(ep_va[i + o])]
                S_ep[pe, 0, vi] += per[o]
                S_ep[pe, 1 + int(bk[o]), vi] += per[o]
        if (i // max(args.batch, 1)) % 20 == 0:
            print(f"    декодирование {i}/{len(va)}", flush=True)

    tot = S_ep.sum(0)
    g_all = gains_from(tot[0])
    print(f"\n  доля возвращённого улучшения на val ({len(va)} наблюдений, "
          f"{len(ep_ids)} эпизодов), ПОСЛЕ декодера, шаги 0-{H_EXEC - 1}:")
    print(f"    {'вариант':>9}{'поз.ошиб':>11}{'доля поз':>10}"
          f"{'вр.ошиб':>10}{'доля вр':>10}{'знак':>8}")
    for v in VARIANTS:
        g = g_all[v]
        f = lambda x: "—" if x is None else f"{x:.1%}"
        print(f"    {v:>9}{g['err_pos']:>11.5f}{f(g['pos']):>10}"
              f"{g['err_rot']:>10.5f}{f(g['rot']):>10}{g['grip']:>7.1%}")

    src = read_source(g_all["both"], g_all["oracle"])
    print(f"    доля от ОГРАНИЧЕННОГО оракула у «both»: положение "
          f"{src['pos']:.1%}, вращение {src['rot']:.1%}"
          if src["pos"] is not None and src["rot"] is not None else "")

    # --- БУТСТРАП ПО ЭПИЗОДАМ, ПАРНЫЕ РАЗНИЦЫ ------------------------------
    draws = boot_draws(S_ep[:, 0], n_boot=args.n_boot, seed=BOOT_SEED)
    gb = [gains_from(x) for x in draws]
    boot = {}
    for v in VARIANTS:
        boot[v] = {k: ci([g[v][k] for g in gb]) for k in ("pos", "rot")}
        boot[v]["grip"] = ci([g[v]["grip"] for g in gb])
    d_pos = [g["both"]["pos"] - g["z0"]["pos"] for g in gb]
    d_rot = [g["both"]["rot"] - g["z0"]["rot"] for g in gb]
    # СХВАТ — ТОЖЕ ПАРНАЯ РАЗНИЦА С ИНТЕРВАЛОМ, а не точечная величина:
    # разница 0.4 п.п. при верхней границе 1.2 п.п. допуск не выдерживает,
    # и жёсткое условие было бы жёстким только на словах.
    d_grip = [g["both"]["grip"] - g["z0"]["grip"] for g in gb]
    d_pos_ci, d_rot_ci, d_grip_ci = ci(d_pos), ci(d_rot), ci(d_grip)
    print(f"\n  95% интервалы по {args.n_boot} бутстрап-выборкам ЭПИЗОДОВ:")
    for v in ("z0", "h24", "both", "oracle"):
        print(f"    {v:>9}: поз [{boot[v]['pos'][0]:.1%}, "
              f"{boot[v]['pos'][1]:.1%}], вр [{boot[v]['rot'][0]:.1%}, "
              f"{boot[v]['rot'][1]:.1%}]")
    print(f"    ПАРНАЯ разница both - z0: поз {np.mean(d_pos):+.3f} "
          f"[{d_pos_ci[0]:+.3f}, {d_pos_ci[1]:+.3f}], вр "
          f"{np.mean(d_rot):+.3f} [{d_rot_ci[0]:+.3f}, {d_rot_ci[1]:+.3f}]")
    print(f"    ПАРНАЯ разница по знаку схвата: {np.mean(d_grip):+.4f} "
          f"[{d_grip_ci[0]:+.4f}, {d_grip_ci[1]:+.4f}]; решение принимается "
          f"по ВЕРХНЕЙ границе при допуске {GRIP_TOL:.1%}")

    # --- РАЗБИВКА ПО КАЧЕСТВУ ЧЕРНОВИКА ------------------------------------
    print(f"\n  разбивка по доле совпавших позиций q0 (корзина 0 — худший "
          f"черновик):")
    print(f"    {'корзина':>8}{'вариант':>9}{'доля поз':>10}{'доля вр':>10}"
          f"{'знак':>8}")
    per_bucket = {}
    for b_ in range(args.buckets):
        if S_ep[:, 1 + b_, VARIANTS.index("draft"), 1].sum() <= 0:
            print(f"    корзина {b_} пуста — пропущена")
            continue
        gbk = gains_from(tot[1 + b_])
        per_bucket[str(b_)] = gbk
        for v in ("z0", "both", "oracle"):
            g = gbk[v]
            f = lambda x: "—" if x is None else f"{x:.1%}"
            print(f"    {b_:>8}{v:>9}{f(g['pos']):>10}{f(g['rot']):>10}"
                  f"{g['grip']:>7.1%}")

    ok, verdict = read_routing(d_pos_ci, d_rot_ci, d_grip_ci)
    print(f"\n  {verdict}")
    print("  ЧИТАТЬ ТАК: зонд отвечает на вопрос об ИСТОЧНИКЕ, а не о том, "
          "улучшит ли\n  обученная голова успех в симуляторе. Оракул здесь — "
          "верхняя граница\n  ранга r с пределом rho, обученной головой "
          "недостижимая.")

    out = dict(script_sha1=sha, cache=prefix, rank=rank,
               tap=int(tap), d_h=D_H, d_latent=D_Z,
               res_norm_sha1=rn_sha, module_drift=mod_drift,
               input_verified_by=stamp.get("script_sha1"),
               live_vs_cache_normed=stamp.get("live_vs_cache_normed"),
               array_sha1=arr_sha,
               n_train=int(len(tr)), n_val=int(len(va)),
               n_val_episodes=int(len(ep_ids)),
               lam=best, lam_grid=list(LAMBDAS), r2=r2,
               gains={v: g_all[v] for v in VARIANTS},
               source_fraction=src,
               boot={v: {k: list(boot[v][k]) for k in boot[v]}
                     for v in boot},
               paired_diff=dict(pos=dict(mean=float(np.mean(d_pos)),
                                         ci=list(d_pos_ci)),
                                rot=dict(mean=float(np.mean(d_rot)),
                                         ci=list(d_rot_ci)),
                                grip=dict(mean=float(np.mean(d_grip)),
                                          ci=list(d_grip_ci))),
               bucket_edges=[float(x) for x in edges],
               per_bucket=per_bucket, n_boot=int(args.n_boot),
               grip_tol=GRIP_TOL, routing_ok=bool(ok), routing=verdict,
               basis_sha1=diag.get("basis_sha1"),
               rho_sha1=diag.get("rho_sha1"),
               cache_meta_sha1=k11a.file_sha1(prefix + ".meta.json"),
               hicora_vla_sha1=k11a.file_sha1(hv.__file__),
               k11a_sha1=k11a.file_sha1(k11a.__file__))
    tmp = args.out + ".tmp"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
