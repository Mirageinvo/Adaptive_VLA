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
  4. НИЖНЯЯ ГРАНИЦА ДЛЯ ЛЮБОГО РАЗБИЕНИЯ. Разбиение на K < V групп
     заставляет минимум V-K строк пользоваться представителем, отличным от
     себя, и каждая такая строка платит не меньше, чем до своего
     ближайшего соседа. Значит для ЛЮБОГО разбиения

         mean_k rel(k) >= (сумма V-K наименьших rel до соседа) / V.

     Граница не зависит ни от алгоритма, ни от сбалансированности, ни от
     метрики кластеризации — только от того, что представитель обязан быть
     настоящей строкой книги. Она НЕ распространяется на центроиды и
     обучаемые представители: там вопрос переносится в опору декодера, а
     мягкие пути её, по измерению M1, проходят с запасом.
  5. Перезапуски k-medoids с разными seed: вырожденные размеры могут быть
     свойством данных, а могут — одной неудачной инициализации.
  6. Кластеризация по НОРМАЛИЗОВАННЫМ строкам (то есть по косинусу), с
     оценкой результата в ИСХОДНОМ пространстве: важно не то, насколько
     похожи направления, а сколько поправки доживает до декодера.

ПОРОГ ОБЪЯВЛЕН ДО ДАННЫХ, 03.10.2026: если нижняя граница при K = 64 ниже
0.50, то жёсткое разбиение с настоящей строкой-представителем ещё имеет
запас и его стоит искать лучшим алгоритмом — код 0. Если нет, это
семейство исключено целиком, независимо от алгоритма — код 4, и остаются
только мягкие или обучаемые представители.
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
    cos_nn = cosine_to(C, nn_idx)
    stats = {}
    stats.update(describe(nrm, "row_norm"))
    stats.update(describe(rel_nn, "nearest_other_rel"))
    stats.update(describe(cos_nn, "nearest_other_cosine"))
    print(f"\n  НОРМЫ СТРОК: медиана {stats['row_norm']['median']:.4f}, "
          f"p05 {stats['row_norm']['p05']:.4f}, p95 "
          f"{stats['row_norm']['p95']:.4f}, минимум "
          f"{stats['row_norm']['min']:.4f}")
    print(f"  ДО БЛИЖАЙШЕЙ ДРУГОЙ СТРОКИ: относительное расстояние "
          f"медиана {stats['nearest_other_rel']['median']:.4f}, p05 "
          f"{stats['nearest_other_rel']['p05']:.4f}; косинус медиана "
          f"{stats['nearest_other_cosine']['median']:.4f}")
    print(f"  для справки: у ортогональных векторов равной нормы "
          f"относительное расстояние {np.sqrt(2.0):.4f}, а не 1")

    bounds = {int(K): partition_lower_bound(rel_nn, K) for K in KS}
    print("\n  НИЖНЯЯ ГРАНИЦА ДЛЯ ЛЮБОГО РАЗБИЕНИЯ С НАСТОЯЩИМ "
          "ПРЕДСТАВИТЕЛЕМ:")
    for K in KS:
        print(f"    K={K:3d}: среднее относительное расстояние не ниже "
              f"{bounds[K]:.4f}")

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
    if bound64 < BOUND_MAX:
        verdict = dict(code=0, outcome=(
            f"нижняя граница при K={max(KS)} равна {bound64:.4f} < "
            f"{BOUND_MAX}: у жёсткого разбиения с настоящей строкой-"
            f"представителем есть запас, и неудача medoid-варианта в M1 "
            f"относится к алгоритму, а не к семейству"))
    else:
        verdict = dict(code=4, outcome=(
            f"нижняя граница при K={max(KS)} равна {bound64:.4f} >= "
            f"{BOUND_MAX}: ЛЮБОЕ разбиение на {max(KS)} и меньше групп с "
            f"настоящей строкой-представителем теряет столько же, "
            f"независимо от алгоритма, метрики и сбалансированности. "
            f"Остаются мягкие и обучаемые представители"))
    print(f"\n  ИСХОД: {verdict['outcome']} (код {verdict['code']})")

    payload = dict(
        kind="k15b_book_geometry", verdict=verdict,
        threshold=dict(bound_max=BOUND_MAX,
                       declared="до данных, 03.10.2026"),
        c1_file=os.path.abspath(a.c1), c1_sha1=got,
        source_state=obj.get("source_state"),
        source_epoch=obj.get("source_epoch"),
        codec_book_gap=codec_gap, vocab=int(V), dim=int(dim),
        stats=stats, lower_bounds={str(k): v for k, v in bounds.items()},
        per_k={str(k): v for k, v in per_k.items()},
        ks=[int(k) for k in KS], restart_seeds=[int(s) for s in RESTART_SEEDS],
        orthogonal_reference_rel=float(np.sqrt(2.0)),
        note=("Относительное расстояние до представителя около единицы НЕ "
              "означает ортогональность строк: у ортогональных векторов "
              "равной нормы оно равно sqrt(2). Значение около 1 "
              "совместимо с коротким представителем около начала "
              "координат, и нормы медоидов здесь приведены именно для "
              "этой проверки. Нижняя граница относится ко всем "
              "разбиениям, где представитель — настоящая строка книги, и "
              "НЕ относится к центроидам, мягким средним и обучаемым "
              "представителям"))
    os.makedirs(os.path.dirname(os.path.abspath(a.summary)) or ".",
                exist_ok=True)
    tmp = a.summary + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1,
                  allow_nan=False)
    os.replace(tmp, a.summary)
    print(f"  сводка: {a.summary}")
    return int(verdict["code"])


if __name__ == "__main__":
    sys.exit(main())
