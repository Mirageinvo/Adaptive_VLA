#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15b: геометрия книги C1. Без GPU, без модели, секунды.

ЗАЧЕМ. В замере M1 относительное расстояние строки книги до её медоида
вышло 0.92-0.99 при любом K от 8 до 64, и я заключил из этого, что строки
книги «почти ортогональны». ЭТО НЕВЕРНО: для двух ортогональных векторов
равной нормы

    ||x - m|| / ||x|| = sqrt(2) ~ 1.414,

а не 1. Значение около единицы объясняется иначе и гораздо скучнее —
медоид может иметь малую норму и стоять около начала координат, то есть
«поправка представителя» просто равна почти нулю. Один гигантский кластер
на 1073-1864 кода — ровно та картина, которую даёт центральный
малонормовый медоид. Здесь это проверяется, а не предполагается.

ЧТО СЧИТАЕТСЯ

  1. Нормы строк и нормы медоидов отдельно: если медоиды систематически
     короче своих членов, объяснение найдено.
  2. Косинус до медоида рядом с относительным расстоянием: одно без
     другого неразличимо между «далеко в ту же сторону» и «близко к нулю».
  3. Распределение БЛИЖАЙШЕГО СОСЕДА по евклиду и по косинусу — это и есть
     предел, доступный любому разбиению с настоящей строкой-представителем.
  4. НИЖНЯЯ ГРАНИЦА. Разбиение на K < V групп заставляет минимум V-K
     строк пользоваться представителем, отличным от себя, и каждая такая
     строка платит не меньше, чем до своего ближайшего соседа. Значит для
     любого разбиения

         mean_k rel(k) >= (сумма V-K наименьших rel до соседа) / V,

     а при неравномерном использовании кодов — то же с весами, где
     отбрасываются K наибольших слагаемых p_k * rel(k).

     ЧТО ИМЕННО ЭТА ГРАНИЦА ОГРАНИЧИВАЕТ, И ЧТО НЕТ. Она ограничивает
     СРЕДНЕЕ ОТНОСИТЕЛЬНОЕ РАССТОЯНИЕ В ЛАТЕНТЕ при представителе —
     настоящей строке книги. Она НЕ ограничивает action RMS на реально
     встречающихся состояниях, и перехода между этими двумя величинами
     здесь не доказано: далёкие латенты могут давать близкие действия,
     чувствительность декодера зависит от z0, и численный порог ниже с
     долей разрыва никак не связан. Поэтому граница НЕ закрывает ни
     грубую иерархию вообще, ни action-aware разбиение, ни центроиды и
     обучаемые представители, ни мягкий coarse-to-fine интерфейс, и она
     НЕ является гейтом ни для одного из них.
  5. Перезапуски k-medoids с разными seed: вырожденные размеры могут быть
     свойством данных, а могут — одной неудачной инициализации.
  6. Кластеризация по НОРМАЛИЗОВАННЫМ строкам (то есть по косинусу), с
     оценкой результата в ИСХОДНОМ пространстве: важно не то, насколько
     похожи направления, а сколько поправки доживает до декодера.

ЭТОТ ЗАМЕР НИЧЕГО НЕ РЕШАЕТ И ВСЕГДА ВОЗВРАЩАЕТ 0. Он описывает книгу.
Численный ориентир 0.50 объявлен до данных, 03.10.2026, и формулируется
ровно так: «достижимо ли при K <= 64 и представителе-настоящей-строке
среднее по строкам книги относительное латентное искажение ниже 0.50».
Ответ на этот вопрос — утверждение о латентной геометрии, и ни запускать,
ни отменять M2 или action-aware разбиение он не может.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

KS = (8, 16, 32, 64)
RESTART_SEEDS = (0, 1, 2, 3, 4)
BOUND_MAX = 0.50


def nearest_other(D):
    """Индекс и расстояние до ближайшей ДРУГОЙ точки. D — квадраты."""
    D = np.asarray(D, np.float64)
    n = int(D.shape[0])
    if D.shape != (n, n):
        raise ValueError(f"матрица {D.shape} не квадратная")
    M = D.copy()
    np.fill_diagonal(M, np.inf)
    idx = M.argmin(1)
    return idx.astype(np.int64), np.sqrt(M[np.arange(n), idx])


def partition_lower_bound(rel_nn, K):
    """Нижняя граница среднего относительного расстояния для ЛЮБОГО
    разбиения на K групп, где представитель — настоящая строка.

    Минимум V-K строк обязаны взять представителем не себя, и дешевле
    ближайшего соседа ни одна из них не отделается. Берутся V-K
    НАИМЕНЬШИХ значений — это самый выгодный возможный расклад.
    """
    rel = np.sort(np.asarray(rel_nn, np.float64))
    V = int(rel.size)
    K = int(K)
    if not (1 <= K <= V):
        raise ValueError(f"K={K} вне [1, {V}]")
    if K == V:
        return 0.0
    return float(rel[:V - K].sum() / V)


def partition_lower_bound_weighted(rel_nn, weights, K):
    """То же при неравномерном использовании кодов.

    Вклад строки, не ставшей представителем, не меньше p_k * rel(k), а
    представителями выгоднее всего сделать K строк с наибольшим таким
    вкладом. При равных весах совпадает с невзвешенной границей.
    """
    rel = np.asarray(rel_nn, np.float64)
    w = np.asarray(weights, np.float64)
    if rel.shape != w.shape:
        raise ValueError(f"формы {rel.shape} и {w.shape}")
    if float(w.sum()) <= 0.0:
        raise ValueError("нулевые веса")
    V, K = int(rel.size), int(K)
    if not (1 <= K <= V):
        raise ValueError(f"K={K} вне [1, {V}]")
    term = np.sort(rel * (w / w.sum()))
    return float(term[:V - K].sum()) if K < V else 0.0


def cosine_nearest(C, cos):
    """Ближайший сосед ПО КОСИНУСУ: максимум косинуса при j != k.

    Прежняя версия обещала это в заголовке, а считала косинус до
    ЕВКЛИДОВА ближайшего соседа — другая величина.
    """
    M = np.asarray(cos, np.float64).copy()
    np.fill_diagonal(M, -np.inf)
    idx = M.argmax(1)
    return idx.astype(np.int64), M[np.arange(M.shape[0]), idx]


def cosine_to(X, idx):
    """Косинус между каждой строкой и строкой `idx[k]`."""
    X = np.asarray(X, np.float64)
    nrm = np.linalg.norm(X, axis=-1)
    denom = np.maximum(nrm * nrm[np.asarray(idx, np.int64)], 1e-300)
    return (X * X[np.asarray(idx, np.int64)]).sum(-1) / denom


def describe(v, name):
    v = np.asarray(v, np.float64)
    return {name: dict(mean=float(v.mean()), median=float(np.median(v)),
                       p05=float(np.percentile(v, 5)),
                       p95=float(np.percentile(v, 95)),
                       min=float(v.min()), max=float(v.max()))}


def selftest():
    import k15b_measure_interface as mi

    # --- БЛИЖАЙШИЙ ДРУГОЙ -----------------------------------------------
    X = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    D = mi.pairwise_sq(X)
    idx, dist = nearest_other(D)
    assert idx.tolist() == [1, 0, 1], idx
    assert np.allclose(dist, [1.0, 1.0, 4.0]), dist
    try:
        nearest_other(np.zeros((2, 3)))
    except ValueError as e:
        assert "квадратная" in str(e), e
    else:
        raise AssertionError("принята неквадратная матрица")

    # --- ГРАНИЦА --------------------------------------------------------
    rel = np.array([0.1, 0.2, 0.3, 0.4])
    assert partition_lower_bound(rel, 4) == 0.0
    # K=3: одна строка обязана взять чужого представителя, самая дешёвая
    assert abs(partition_lower_bound(rel, 3) - 0.1 / 4) < 1e-12
    assert abs(partition_lower_bound(rel, 2) - 0.3 / 4) < 1e-12
    assert abs(partition_lower_bound(rel, 1) - 0.6 / 4) < 1e-12
    # ГРАНИЦА МОНОТОННА ПО K И НЕ ЗАВИСИТ ОТ ПОРЯДКА
    perm = partition_lower_bound(rel[::-1], 2)
    assert abs(perm - partition_lower_bound(rel, 2)) < 1e-15
    assert all(partition_lower_bound(rel, k) >= partition_lower_bound(rel,
                                                                     k + 1)
               for k in (1, 2, 3))
    for bad in (0, 5):
        try:
            partition_lower_bound(rel, bad)
        except ValueError as e:
            assert "вне" in str(e), e
        else:
            raise AssertionError(f"принято K={bad}")
    # ГРАНИЦА ДЕЙСТВИТЕЛЬНО НЕ ПРЕВОСХОДИТ ФАКТ: сверка с k-medoids
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(40, 6))
    Dy = mi.pairwise_sq(Y)
    ny = np.linalg.norm(Y, axis=-1)
    _i, dy = nearest_other(Dy)
    rel_nn = dy / ny
    for K in (2, 5, 10, 20):
        lab, med, _info = mi.kmedoids(Dy, K)
        fact = float((np.sqrt(Dy[np.arange(40), med[lab]]) / ny).mean())
        bound = partition_lower_bound(rel_nn, K)
        assert bound <= fact + 1e-12, (K, bound, fact)

    # --- КОСИНУС --------------------------------------------------------
    Z = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [2.0, 0.0]])
    c = cosine_to(Z, [3, 3, 3, 3])
    assert np.allclose(c, [1.0, 0.0, -1.0, 1.0]), c
    # ОРТОГОНАЛЬНЫЕ ВЕКТОРЫ РАВНОЙ НОРМЫ ДАЮТ rel = sqrt(2), А НЕ 1 —
    # то самое, на чём я ошибся
    d2 = float(mi.pairwise_sq(Z[[0, 1]])[0, 1])
    assert abs(np.sqrt(d2) / 1.0 - np.sqrt(2.0)) < 1e-12
    # А rel ~ 1 даёт КОРОТКИЙ представитель около нуля
    short = np.array([[1.0, 0.0], [0.02, 0.0]])
    r = np.sqrt(float(mi.pairwise_sq(short)[0, 1])) / 1.0
    assert abs(r - 0.98) < 1e-12, r

    # --- ВЗВЕШЕННАЯ ГРАНИЦА СОВПАДАЕТ С НЕВЗВЕШЕННОЙ ПРИ РАВНЫХ ВЕСАХ --
    for K in (1, 2, 3, 4):
        a_ = partition_lower_bound(rel, K)
        b_ = partition_lower_bound_weighted(rel, np.ones_like(rel), K)
        assert abs(a_ - b_) < 1e-15, (K, a_, b_)
    # ПРИ ПЕРЕКОШЕННЫХ ВЕСАХ ПРЕДСТАВИТЕЛЯМИ ВЫГОДНЕЕ ЧАСТЫЕ СТРОКИ
    wt = np.array([100.0, 1.0, 1.0, 1.0])
    bw = partition_lower_bound_weighted(rel, wt, 1)
    assert abs(bw - (0.2 + 0.3 + 0.4) / 103.0) < 1e-12, bw
    for bad_w in (np.zeros(4), ):
        try:
            partition_lower_bound_weighted(rel, bad_w, 2)
        except ValueError as e:
            assert "веса" in str(e), e
        else:
            raise AssertionError("приняты нулевые веса")
    try:
        partition_lower_bound_weighted(rel, np.ones(3), 2)
    except ValueError as e:
        assert "формы" in str(e), e
    else:
        raise AssertionError("приняты веса другой длины")

    # --- БЛИЖАЙШИЙ ПО КОСИНУСУ ОТЛИЧАЕТСЯ ОТ БЛИЖАЙШЕГО ПО ЕВКЛИДУ ----
    W = np.array([[1.0, 0.0], [10.0, 0.1], [1.2, 0.5]])
    Dw = mi.pairwise_sq(W)
    cosw = (W @ W.T) / np.outer(np.linalg.norm(W, axis=-1),
                                np.linalg.norm(W, axis=-1))
    e_idx, _e_d = nearest_other(Dw)
    c_idx, c_val = cosine_nearest(W, cosw)
    assert e_idx[0] == 2, e_idx          # по евклиду ближе короткая строка
    assert c_idx[0] == 1, c_idx          # по косинусу ближе длинная
    assert c_val[0] > 0.999, c_val
    assert cosine_nearest(W, cosw)[1].shape == (3,)

    d = describe([1.0, 2.0, 3.0], "x")
    assert d["x"]["median"] == 2.0 and d["x"]["max"] == 3.0
    print(f"самопроверка k15b_book_geometry пройдена: порог границы при "
          f"K=64 — {BOUND_MAX}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import k15b_measure_interface as mi

    ap = argparse.ArgumentParser(
        description="K-15b: геометрия книги C1, без GPU")
    ap.add_argument("--c1", default="data/k15b/c1_selected.pt")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--target", default="data/k15b/rankpath_target_train.npz",
                    help="кэш целей; по нему берутся веса использования "
                         "кодов. Если файла нет, считается только "
                         "равномерный вариант")
    ap.add_argument("--summary", default="reports/k15b/book_geometry.json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if os.path.exists(a.summary) and not a.overwrite:
        raise SystemExit(f"{a.summary} уже существует: без --overwrite не "
                         f"перезаписываю")
    import torch

    obj = torch.load(a.c1, map_location="cpu", weights_only=False)
    if obj.get("kind") != "k15b_c1_selected" \
            or obj.get("accepted") is not True:
        raise SystemExit(f"{a.c1}: kind {obj.get('kind')!r}, accepted "
                         f"{obj.get('accepted')!r}")
    C = np.asarray(obj["c1"].detach().float().cpu().numpy(), np.float64)
    got = hashlib.sha1(np.ascontiguousarray(
        np.asarray(C, np.float32)).tobytes()).hexdigest()[:12]
    if got != obj["c1_sha1"]:
        raise SystemExit(f"отпечаток книги {got}, в файле {obj['c1_sha1']}")
    V, dim = C.shape
    print(f"  книга {got} из состояния {obj.get('source_state')!r}: "
          f"{V} строк, размерность {dim}")

    # ТА ЖЕ ЛИ ЭТО КНИГА, ЧТО У КОДЕКА. Проверяется без модели: книги
    # кодека лежат рядом с кэшем, и probe выбрал именно исходную.
    books_path = f"{a.cache}.codebooks.npy"
    codec_gap = None
    if os.path.exists(books_path):
        E = np.load(books_path)
        if E.shape[1:] != C.shape:
            raise SystemExit(f"книги кодека {E.shape}, книга {C.shape}")
        codec_gap = float(np.abs(np.asarray(E[1], np.float64) - C).max())
        print(f"  расхождение с книгой кодека уровня 1: {codec_gap:.3e} "
              + ("(это ИСХОДНАЯ книга кодека)" if codec_gap <= 1e-5
                 else "(книга НЕ исходная)"))
    else:
        print(f"  {books_path} нет: сверку с кодеком пропускаю")

    nrm = np.linalg.norm(C, axis=-1)
    D = mi.pairwise_sq(C)
    nn_idx, nn_d = nearest_other(D)
    rel_nn = nn_d / np.maximum(nrm, 1e-300)
    cos_to_euclid_nn = cosine_to(C, nn_idx)
    cosmat = (C @ C.T) / np.maximum(np.outer(nrm, nrm), 1e-300)
    cnn_idx, cnn_val = cosine_nearest(C, cosmat)
    rel_to_cosine_nn = np.sqrt(D[np.arange(V), cnn_idx]) / np.maximum(
        nrm, 1e-300)
    stats = {}
    stats.update(describe(nrm, "row_norm"))
    stats.update(describe(rel_nn, "nearest_other_rel"))
    stats.update(describe(cos_to_euclid_nn, "cosine_to_euclid_nearest"))
    # БЛИЖАЙШИЙ ПО КОСИНУСУ — ОТДЕЛЬНАЯ ВЕЛИЧИНА, а не косинус до
    # евклидова соседа: в заголовке обещалась именно она.
    stats.update(describe(cnn_val, "cosine_nearest_value"))
    stats.update(describe(rel_to_cosine_nn, "rel_to_cosine_nearest"))
    stats.update(describe((cnn_idx != nn_idx).astype(np.float64),
                          "cosine_nn_differs_from_euclid_nn"))
    print(f"\n  НОРМЫ СТРОК: медиана {stats['row_norm']['median']:.4f}, "
          f"p05 {stats['row_norm']['p05']:.4f}, p95 "
          f"{stats['row_norm']['p95']:.4f}, минимум "
          f"{stats['row_norm']['min']:.4f}")
    print(f"  ДО БЛИЖАЙШЕЙ ПО ЕВКЛИДУ: относительное расстояние медиана "
          f"{stats['nearest_other_rel']['median']:.4f}, p05 "
          f"{stats['nearest_other_rel']['p05']:.4f}; косинус до неё "
          f"медиана {stats['cosine_to_euclid_nearest']['median']:.4f}")
    print(f"  ДО БЛИЖАЙШЕЙ ПО КОСИНУСУ: косинус медиана "
          f"{stats['cosine_nearest_value']['median']:.4f}, p95 "
          f"{stats['cosine_nearest_value']['p95']:.4f}; относительное "
          f"расстояние до неё медиана "
          f"{stats['rel_to_cosine_nearest']['median']:.4f}; это другая "
          f"строка, чем евклидов сосед, в "
          f"{100 * stats['cosine_nn_differs_from_euclid_nn']['mean']:.1f} "
          f"% случаев")
    print(f"  для справки: у ортогональных векторов равной нормы "
          f"относительное расстояние {np.sqrt(2.0):.4f}, а не 1")

    # --- ВЕСА ИСПОЛЬЗОВАНИЯ КОДОВ ---------------------------------------
    usage = None
    if os.path.exists(a.target):
        cache = np.load(a.target, allow_pickle=True)
        codes = np.asarray(cache["codes"], np.int64).reshape(-1)
        usage = np.bincount(codes, minlength=V).astype(np.float64)
        print(f"\n  веса использования взяты из {a.target}: "
              f"{codes.size} целевых кодов, различных "
              f"{int((usage > 0).sum())}/{V}")
    else:
        print(f"\n  {a.target} нет: взвешенная граница не считается")

    bounds = {int(K): partition_lower_bound(rel_nn, K) for K in KS}
    bounds_w = ({int(K): partition_lower_bound_weighted(rel_nn, usage, K)
                 for K in KS} if usage is not None else None)
    print("  НИЖНЯЯ ГРАНИЦА СРЕДНЕГО ОТНОСИТЕЛЬНОГО ЛАТЕНТНОГО "
          "ИСКАЖЕНИЯ\n  (представитель — настоящая строка книги; про "
          "action RMS это НЕ утверждение):")
    for K in KS:
        extra = ("" if bounds_w is None
                 else f", с весами использования {bounds_w[int(K)]:.4f}")
        print(f"    K={K:3d}: не ниже {bounds[K]:.4f}{extra}")

    # --- ФАКТИЧЕСКИЕ РАЗБИЕНИЯ, ПЕРЕЗАПУСКИ, КОСИНУСНАЯ ВЕРСИЯ ---------
    Cn = C / np.maximum(nrm, 1e-300)[:, None]
    Dn = mi.pairwise_sq(Cn)
    per_k = {}
    for K in KS:
        restarts = []
        for seed in RESTART_SEEDS:
            lab, med, info = mi.kmedoids(D, K, seed=seed)
            rel = np.sqrt(D[np.arange(V), med[lab]]) / np.maximum(nrm,
                                                                  1e-300)
            restarts.append(dict(
                seed=int(seed), size_max=int(info["size_max"]),
                size_min=int(info["size_min"]),
                inertia=float(info["inertia"]),
                rel_mean=float(rel.mean()),
                medoid_norm_median=float(np.median(nrm[med])),
                cosine_to_medoid_mean=float(cosine_to(C, med[lab]).mean())))
        lab_n, med_n, info_n = mi.kmedoids(Dn, K)
        rel_n = np.sqrt(D[np.arange(V), med_n[lab_n]]) / np.maximum(nrm,
                                                                    1e-300)
        cosine = dict(
            size_min=int(info_n["size_min"]), size_max=int(info_n["size_max"]),
            size_median=float(info_n["size_median"]),
            rel_mean_in_original_space=float(rel_n.mean()),
            cosine_to_medoid_mean=float(cosine_to(C, med_n[lab_n]).mean()),
            medoid_norm_median=float(np.median(nrm[med_n])))
        per_k[int(K)] = dict(lower_bound=bounds[int(K)],
                             euclidean_restarts=restarts,
                             cosine_clustering=cosine)
        best = min(restarts, key=lambda r: r["rel_mean"])
        print(f"\n  K={K}")
        print(f"    евклид, {len(restarts)} перезапусков: rel "
              f"{min(r['rel_mean'] for r in restarts):.4f}.."
              f"{max(r['rel_mean'] for r in restarts):.4f}, крупнейший "
              f"кластер {min(r['size_max'] for r in restarts)}.."
              f"{max(r['size_max'] for r in restarts)}")
        print(f"    лучший перезапуск: медиана нормы медоида "
              f"{best['medoid_norm_median']:.4f} против медианы по книге "
              f"{stats['row_norm']['median']:.4f}; средний косинус до "
              f"медоида {best['cosine_to_medoid_mean']:.4f}")
        print(f"    косинусная кластеризация: размеры "
              f"{cosine['size_min']}..{cosine['size_max']}, медиана "
              f"{cosine['size_median']:.0f}; rel в ИСХОДНОМ пространстве "
              f"{cosine['rel_mean_in_original_space']:.4f}; косинус до "
              f"медоида {cosine['cosine_to_medoid_mean']:.4f}")

    bound64 = bounds[max(KS)]
    statement = dict(
        question=(f"достижимо ли при K <= {max(KS)} и представителе — "
                  f"настоящей строке книги среднее по строкам книги "
                  f"относительное ЛАТЕНТНОЕ искажение ниже {BOUND_MAX}"),
        bound_at_max_k=float(bound64),
        bound_below_threshold=bool(bound64 < BOUND_MAX),
        threshold=float(BOUND_MAX),
        weighted_bound_at_max_k=(None if bounds_w is None
                                 else float(bounds_w[max(KS)])),
        decides_nothing=("это утверждение о латентной геометрии. Перехода "
                         "к action RMS на встречающихся состояниях здесь "
                         "НЕ доказано: далёкие латенты могут давать "
                         "близкие действия, чувствительность декодера "
                         "зависит от z0, а порог с долей разрыва не "
                         "связан. Ни M2, ни action-aware разбиение, ни "
                         "центроиды, ни мягкий coarse-to-fine этот замер "
                         "не открывает и не закрывает"))
    print(f"\n  УТВЕРЖДЕНИЕ: {statement['question']} — "
          + ("ДА" if statement["bound_below_threshold"] else "НЕТ")
          + f" (граница {bound64:.4f})")
    print("  ЭТО НЕ ГЕЙТ. " + statement["decides_nothing"])

    payload = dict(
        kind="k15b_book_geometry", statement=statement, decides=None,
        threshold=dict(bound_max=BOUND_MAX,
                       declared="до данных, 03.10.2026"),
        c1_file=os.path.abspath(a.c1), c1_sha1=got,
        source_state=obj.get("source_state"),
        source_epoch=obj.get("source_epoch"),
        codec_book_gap=codec_gap, vocab=int(V), dim=int(dim),
        stats=stats, lower_bounds={str(k): v for k, v in bounds.items()},
        lower_bounds_weighted=(None if bounds_w is None
                               else {str(k): v
                                     for k, v in bounds_w.items()}),
        usage_file=(os.path.abspath(a.target)
                    if usage is not None else None),
        per_k={str(k): v for k, v in per_k.items()},
        ks=[int(k) for k in KS], restart_seeds=[int(s) for s in RESTART_SEEDS],
        orthogonal_reference_rel=float(np.sqrt(2.0)),
        note=("ЭТОТ ЗАМЕР НИЧЕГО НЕ РЕШАЕТ: он описывает книгу и всегда "
              "возвращает 0. "
              "Относительное расстояние до представителя около единицы НЕ "
              "означает ортогональность строк: у ортогональных векторов "
              "равной нормы оно равно sqrt(2). Значение около 1 "
              "совместимо с коротким представителем около начала "
              "координат, и нормы медоидов здесь приведены именно для "
              "этой проверки. Нижняя граница относится ко всем "
              "разбиениям, где представитель — настоящая строка книги, и "
              "НЕ относится ни к центроидам, ни к мягким средним, ни к "
              "обучаемым представителям, ни к разбиению, построенному по "
              "ошибке ДЕЙСТВИЯ, и вообще не является утверждением об "
              "action RMS"))
    os.makedirs(os.path.dirname(os.path.abspath(a.summary)) or ".",
                exist_ok=True)
    tmp = a.summary + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1,
                  allow_nan=False)
    os.replace(tmp, a.summary)
    print(f"  сводка: {a.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
