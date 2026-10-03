#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15b, замер M1: каков интерфейс между h18 и замороженной книгой C1.

ЗАЧЕМ. Фаза `q1_reader` закрыта отрицательно: обучение 2048-классового
читателя снизило кросс-энтропию на отложенной части (4.9294 -> 4.6984) и
при этом УХУДШИЛО исполняемое действие (0.142997 -> 0.147169). Отсюда НЕ
следует, что в h18 мало информации: кросс-энтропия одного читателя
ограничивает `H(Y|H)` только сверху, потому что

    CE = H(Y|H) + E_h KL(p || q),

и `log2(2048) - CE` было бы взаимной информацией лишь при равномерных
целевых кодах И точном воспроизведении условного распределения. Ни то, ни
другое не проверено. Поэтому прежнее утверждение «в h18 только 4.2 бита»
снято, и здесь измеряется то, что измеримо.

ЧТО ИЗМЕРЯЕТСЯ, ОДИН ПРОХОД ПО val_sel, БЕЗ ОБУЧЕНИЯ

  1. ИЗВЛЕЧЁННАЯ ИНФОРМАЦИЯ ОТНОСИТЕЛЬНО BASELINE. Маргинальный
     предсказатель строится ПО ПОЗИЦИЯМ на целях train из кэша, его CE
     считается на тех же val-целях, что и у читателя. Величина
     `CE_marginal - CE_reader` — извлечённая информация, вариационная
     нижняя оценка, и называется она именно так, а не «информацией в
     h18». Рядом: эмпирическая энтропия целей, арифметическое
     `mean(p_target)` ОТДЕЛЬНО от геометрического `exp(-CE)`, и точность
     top-1 маргинала как настоящий baseline для согласия читателя (прежде
     8.47 % сравнивались с 1/2048, то есть не с тем).

  2. РАНЖИРОВАНИЕ. recall@{1,4,16,32} читателя против целевого кода.

  3. ГРУБЫЙ ИНТЕРФЕЙС. Книга C1 разбивается на K кластеров, представитель
     кластера — НАСТОЯЩАЯ строка книги (медоид), а не центроид: центроид
     вывел бы латент с многообразия сумм книг кодека, то есть ровно туда,
     где обученная книга в probe провалила опору декодера. Для каждого K
     четыре пути:

        oracle_hard    кластер известен из цели, применяется медоид
        oracle_soft    кластер известен, внутри него мягкое среднее
        reader_hard    вероятности читателя суммируются по кластерам,
                       argmax по кластерам, применяется медоид
        reader_soft    кластер выбран читателем, внутри него мягкое среднее

     Первые два — потолок интерфейса. Вторые два отвечают без обучения на
     вопрос, лежит ли полезная грубая информация уже в имеющихся логитах.

  4. МЕХАНИЗМ. Утверждение «точное попадание даёт всё, промах ничего» не
     было доказано: `agree` — средняя точность по ОТДЕЛЬНЫМ кодовым
     позициям, а 0.072892 — свойство всей шестнадцатикодовой
     последовательности, и одно из другого не выводится. Здесь ошибка
     действия стратифицируется по числу совпавших позиций, и отдельно
     строится контрфактическая кривая: целевой код подставляется в m
     случайных позиций из 16 при m = 0, 4, 8, 12, 16. Выпуклая кривая
     означает, что зачёт только за полное попадание; близкая к линейной —
     что частичное попадание помогает.

  5. МЯГКИЙ ПУТЬ в доступной точке с опорой и диапазоном: прежде был
     известен только его RMS без гейтов.

ЧЕГО ЭТОТ ЗАМЕР НЕ МОЖЕТ. Обученного читателя. Прогон 02.10.2026 сохранил
только выбранную точку, ею оказалась НЕОБУЧЕННАЯ e0s0, и состояния с CE
4.6984 после выхода процесса не существуют. Поэтому «нынешний читатель»
здесь — читатель из K-14, у которого CE 4.9294 и согласие 8.47 % против
baseline маргинала. Если в чекпойнте есть `all_states` (их пишет
исправленный тренер), можно взять любую точку через `--point`.

ПОРОГИ ОБЪЯВЛЕНЫ ДО ДАННЫХ, 03.10.2026:

    потолок грубого интерфейса   capture(oracle_hard) >= 0.50
    грубая информация уже есть   capture(reader_hard) >= capture(a1_pol) + 0.02

K проходит, если выполнены ОБА. Направление открыто, если прошло хотя бы
одно K из {8, 16, 32, 64}; код возврата 0. Если ни одно — код 4, и дальше
мягкий путь или success-rate/RL, а не третья попытка с той же головой.
Технические отказы — исключение, а не код.

СТРУКТУРНЫЙ ИНВАРИАНТ. В свёртке участвует K = 2048, где каждый кластер
одноэлементный. Тогда по построению

    oracle_hard == a1_rank,  reader_hard == a1_pol,
    oracle_soft == oracle_hard,  reader_soft == reader_hard,

и это сверяется побитово с путями, посчитанными независимо. Любая ошибка
в агрегации вероятностей, в метках кластеров или в подстановке медоидов
ломает именно это равенство.
"""
import argparse
import hashlib
import inspect
import json
import os
import sys
import time

import numpy as np

KS_DEFAULT = (8, 16, 32, 64)
CLUSTER_SEED = 0
CLUSTER_ITERS = 100
MARGINAL_ALPHA = 1.0            # сглаживание Лапласа для маргинала
RECALL_KS = (1, 4, 16, 32)
SUBSTITUTE_M = (0, 4, 8, 12, 16)
MATCH_BUCKETS = ((0, 0), (1, 2), (3, 5), (6, 16))
ORACLE_CAPTURE_MIN = 0.50
READER_CAPTURE_MARGIN = 0.02
TEACHER_REL_LIMIT = 1e-4
PATH_REL_LIMIT = 1e-3           # предел для СТРУКТУРНЫХ сверок путей


# --- ЧИСТЫЕ ФУНКЦИИ ------------------------------------------------------

def pairwise_sq(X):
    """Полная матрица квадратов евклидовых расстояний, 2048 x 2048.

    Считается один раз через грам-матрицу: наивная форма (n, K, d)
    при большой размерности латента занимала бы сотни мегабайт.
    """
    X = np.asarray(X, np.float64)
    if X.ndim != 2:
        raise ValueError(f"ожидалась матрица, получено {X.shape}")
    g = X @ X.T
    n2 = np.diag(g).copy()
    return np.maximum(n2[:, None] - 2.0 * g + n2[None, :], 0.0)


def kmeanspp_medoids(D, K, rng):
    """k-means++ по готовой матрице расстояний. Индексы РАЗЛИЧНЫ всегда.

    Выбор идёт только из ещё не взятых точек: прежняя версия тянула из
    всех и на вырожденных данных (одинаковые строки книги, нулевые
    расстояния) возвращала дубли, из которых потом получался пустой
    кластер.
    """
    n = int(D.shape[0])
    if not (1 <= K <= n):
        raise ValueError(f"K={K} вне [1, {n}]")
    first = int(rng.integers(n))
    chosen = [first]
    d2 = D[:, first].copy()
    while len(chosen) < K:
        taken = set(chosen)
        rest = np.asarray([i for i in range(n) if i not in taken], np.int64)
        total = float(d2[rest].sum())
        if total <= 0.0:
            # Все оставшиеся точки совпадают с уже выбранными: берём
            # детерминированно, иначе выбор по нулевым вероятностям падает.
            nxt = int(rest[0])
        else:
            nxt = int(rng.choice(rest, p=d2[rest] / total))
        chosen.append(nxt)
        d2 = np.minimum(d2, D[:, nxt])
    if len(set(chosen)) != K:
        raise ValueError("инициализация выдала дубли представителей")
    return np.asarray(chosen, np.int64)


def assign_nonempty(D, med):
    """Метки по ближайшему представителю, БЕЗ пустых кластеров.

    Повторное взятие argmin пустой кластер не лечит: при точных совпадениях
    расстояний все строки снова уходят в первый из равных. Поэтому пустой
    кластер забирает конкретную точку принудительно — самую далёкую от
    своего представителя среди тех, чей кластер не станет пустым и кто сам
    не представитель. `med` правится на месте.
    """
    n, K = int(D.shape[0]), int(len(med))
    labels = D[:, med].argmin(1).astype(np.int64)
    for c in range(K):
        if int((labels == c).sum()):
            continue
        sizes = np.bincount(labels, minlength=K)
        cost = D[np.arange(n), med[labels]].astype(np.float64).copy()
        cost[sizes[labels] <= 1] = -np.inf
        cost[np.asarray(med, np.int64)] = -np.inf
        i = int(np.argmax(cost))
        if not np.isfinite(cost[i]):
            raise ValueError(f"K={K}: нечего отдать кластеру {c}")
        labels[i] = c
        med[c] = i
    return labels


def kmedoids(D, K, seed=CLUSTER_SEED, iters=CLUSTER_ITERS):
    """Альтернирующий k-medoids по матрице расстояний. Детерминированный.

    Возвращает (labels, medoids, info). `medoids[labels[i]]` — индекс
    представителя точки i, и это НАСТОЯЩАЯ строка книги.
    """
    n = int(D.shape[0])
    if D.shape != (n, n):
        raise ValueError(f"матрица расстояний {D.shape} не квадратная")
    if not (1 <= int(K) <= n):
        raise ValueError(f"K={K} вне [1, {n}]")
    K = int(K)
    rng = np.random.default_rng(int(seed))
    med = kmeanspp_medoids(D, K, rng)
    labels = assign_nonempty(D, med)
    moved, it = 0, 0
    for it in range(int(iters)):
        new_med = med.copy()
        for c in range(K):
            members = np.flatnonzero(labels == c)
            sub = D[np.ix_(members, members)]
            new_med[c] = int(members[int(sub.sum(1).argmin())])
        new_labels = assign_nonempty(D, new_med)
        moved = int((new_labels != labels).sum())
        same = bool(np.array_equal(new_med, med)) and moved == 0
        med, labels = new_med, new_labels
        if same:
            break
    sizes = np.bincount(labels, minlength=K)
    if int(sizes.min()) <= 0:
        raise ValueError(f"K={K}: остался пустой кластер")
    if len(set(med.tolist())) != K:
        raise ValueError(f"K={K}: представители не различны")
    own = D[np.arange(n), med[labels]]
    info = dict(K=K, iterations=int(it + 1), last_moved=int(moved),
                size_min=int(sizes.min()), size_max=int(sizes.max()),
                size_median=float(np.median(sizes)),
                inertia=float(own.sum()),
                mean_sq_to_medoid=float(own.mean()))
    return labels.astype(np.int64), med.astype(np.int64), info


def clustering_fingerprint(labels, medoids):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(np.asarray(labels, np.int64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(medoids, np.int64)).tobytes())
    return h.hexdigest()[:12]


def marginal_logprob(counts, alpha=MARGINAL_ALPHA):
    """Логарифм маргинала ПО ПОЗИЦИЯМ со сглаживанием Лапласа.

    `counts` — (позиции, словарь). Сглаживание обязательно: без него
    невиденный на train целевой код дал бы на val бесконечную CE, то есть
    baseline зависел бы от одной строки.
    """
    c = np.asarray(counts, np.float64)
    if c.ndim != 2:
        raise ValueError(f"ожидалась матрица счётчиков, получено {c.shape}")
    if float(alpha) <= 0.0:
        raise ValueError("сглаживание должно быть положительным")
    p = (c + float(alpha)) / (c.sum(1, keepdims=True)
                              + float(alpha) * c.shape[1])
    return np.log(p)


def entropy_bits(p, axis=-1):
    """Энтропия в битах. Нули не дают вклада."""
    p = np.asarray(p, np.float64)
    q = np.where(p > 0.0, p, 1.0)
    return float(-(p * np.log2(q)).sum(axis=axis).mean())


def extracted_bits(ce_baseline, ce_model):
    """(CE_baseline - CE_model) в битах: ИЗВЛЕЧЁННАЯ информация.

    Это вариационная нижняя оценка выигрыша над контекстно-независимым
    предсказателем, а НЕ взаимная информация между h18 и целью и не предел
    возможностей представления.
    """
    return float((float(ce_baseline) - float(ce_model)) / np.log(2.0))


def cluster_choice(probs, labels, K, torch):
    """Вероятность КЛАСТЕРА и выбор читателя: (B,T,V) -> (B,T,K), (B,T).

    Суммирование идёт по меткам книги, поэтому при тождественном
    разбиении результат — та же самая раскладка в том же порядке, и
    argmax по кластерам совпадает с argmax по кодам даже на точных ничьих.
    """
    if int(labels.numel()) != int(probs.shape[-1]):
        raise ValueError(f"метк {int(labels.numel())}, кодов "
                         f"{int(probs.shape[-1])}")
    pc = torch.zeros(tuple(probs.shape[:-1]) + (int(K),),
                     device=probs.device, dtype=probs.dtype)
    pc.index_add_(-1, labels, probs)
    return pc, pc.argmax(-1)


def soft_inside(probs, labels, cluster, book, fallback, torch, eps=1e-12):
    """Мягкое среднее книги ВНУТРИ выбранного кластера.

    Где масса на кластере вырождена, берётся `fallback` — эмбеддинг
    представителя: нормировать на ноль нельзя, а вернуть ноль значило бы
    подменить поправку её отсутствием и улучшить результат по ошибке.
    Возвращает (эмбеддинг, маска вырожденных позиций).
    """
    shape = [1] * (int(probs.dim()) - 1) + [-1]
    msk = labels.view(*shape) == cluster.unsqueeze(-1)
    w = probs * msk
    den = w.sum(-1, keepdim=True)
    deg = den.squeeze(-1) <= float(eps)
    emb = (w / den.clamp_min(float(eps))) @ book
    return torch.where(deg.unsqueeze(-1), fallback, emb), deg


def recall_at(logits, target, ks, torch):
    """Доля позиций, где цель попала в top-k. `logits` (..., vocab)."""
    vocab = int(logits.shape[-1])
    kmax = max(int(k) for k in ks)
    if kmax > vocab:
        raise ValueError(f"k={kmax} больше словаря {vocab}")
    top = logits.topk(kmax, dim=-1).indices
    hit = top == target.unsqueeze(-1)
    out = {}
    for k in ks:
        out[int(k)] = float(hit[..., :int(k)].any(-1).float().mean())
    return out


def match_bucket_index(n_match, buckets=MATCH_BUCKETS):
    """Номер корзины по числу совпавших кодовых позиций."""
    for i, (lo, hi) in enumerate(buckets):
        if lo <= int(n_match) <= hi:
            return i
    raise ValueError(f"{n_match} не попало ни в одну корзину {buckets}")


def random_position_mask(n_rows, n_pos, m, gen, torch):
    """Маска из m случайных кодовых позиций НА СТРОКУ. Воспроизводимая."""
    m = int(m)
    if m < 0 or m > int(n_pos):
        raise ValueError(f"m={m} вне [0, {n_pos}]")
    order = torch.rand(int(n_rows), int(n_pos),
                       generator=gen).argsort(dim=-1)
    mask = torch.zeros(int(n_rows), int(n_pos), dtype=torch.bool)
    if m:
        mask.scatter_(1, order[:, :m], True)
    return mask


def cluster_decision(per_k, capture_hard,
                     oracle_min=ORACLE_CAPTURE_MIN,
                     margin=READER_CAPTURE_MARGIN):
    """Исход замера по объявленным порогам. Чистая функция.

    `per_k` — {K: {"capture_oracle_hard": x, "capture_reader_hard": y}},
    вырожденное K словаря в решении НЕ участвует.
    """
    if capture_hard is None:
        return dict(code=3, outcome="доля разрыва у нынешнего пути не "
                                    "определена: разрыв неположителен",
                    passed_ks=[])
    need_reader = float(capture_hard) + float(margin)
    rows, passed = [], []
    for K in sorted(per_k):
        o = per_k[K].get("capture_oracle_hard")
        r = per_k[K].get("capture_reader_hard")
        ok_o = o is not None and float(o) >= float(oracle_min)
        # Допуск 1e-12 — это шум двоичного представления суммы
        # `capture + margin`, а не ослабление порога: 0.034 + 0.02 даёт
        # 0.054000000000000006, и значение ровно 0.054 иначе не проходило бы.
        ok_r = r is not None and float(r) >= need_reader - 1e-12
        rows.append(dict(K=int(K), oracle=o, reader=r,
                         oracle_passed=bool(ok_o),
                         reader_passed=bool(ok_r),
                         passed=bool(ok_o and ok_r)))
        if ok_o and ok_r:
            passed.append(int(K))
    if passed:
        return dict(code=0, outcome=(f"грубый интерфейс открыт при K="
                                     f"{passed}: потолок есть и нынешние "
                                     f"логиты уже выбирают кластер"),
                    passed_ks=passed, rows=rows,
                    oracle_min=float(oracle_min),
                    reader_required=float(need_reader))
    return dict(code=4, outcome=("ни одно K не прошло оба порога: грубый "
                                 "интерфейс в этом виде закрыт, дальше "
                                 "мягкое уточнение или success-rate"),
                passed_ks=[], rows=rows, oracle_min=float(oracle_min),
                reader_required=float(need_reader))


def selftest():
    import torch
    import k15b_train_stagewise as trainer

    # --- РАССТОЯНИЯ И КЛАСТЕРИЗАЦИЯ -------------------------------------
    X = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0],
                  [0.0, 20.0], [1.0, 20.0]])
    D = pairwise_sq(X)
    assert D.shape == (6, 6)
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(D, D.T)
    assert abs(D[0, 2] - 100.0) < 1e-9, D[0, 2]
    labels, med, info = kmedoids(D, 3)
    groups = {}
    for i, c in enumerate(labels):
        groups.setdefault(int(c), []).append(i)
    assert sorted(sorted(v) for v in groups.values()) == [
        [0, 1], [2, 3], [4, 5]], groups
    assert info["size_min"] == 2 and info["size_max"] == 2, info
    assert set(med.tolist()) <= set(range(6))
    # ПРЕДСТАВИТЕЛЬ — НАСТОЯЩАЯ ТОЧКА, И ОН ЛЕЖИТ В СВОЁМ КЛАСТЕРЕ
    for c, m in enumerate(med):
        assert int(labels[int(m)]) == int(c), (c, m, labels[int(m)])
    # K = n: каждая точка сама себе представитель. Это тот инвариант, на
    # котором держится сверка вырожденного K со путями a1_rank и a1_pol.
    l_n, m_n, i_n = kmedoids(D, 6)
    assert i_n["size_min"] == 1 and i_n["size_max"] == 1, i_n
    for i in range(6):
        assert int(m_n[int(l_n[i])]) == i, (i, m_n, l_n)
    assert abs(i_n["inertia"]) < 1e-12, i_n
    # ДЕТЕРМИНИРОВАННОСТЬ ПРИ ОДНОМ SEED И ОТПЕЧАТОК
    l2, m2, _i2 = kmedoids(D, 3)
    assert np.array_equal(l2, labels) and np.array_equal(m2, med)
    assert clustering_fingerprint(labels, med) == \
        clustering_fingerprint(l2, m2)
    assert clustering_fingerprint(labels, med) != \
        clustering_fingerprint(l_n, m_n)
    # ПУСТОЙ КЛАСТЕР НЕВОЗМОЖЕН ДАЖЕ НА ВЫРОЖДЕННЫХ ДАННЫХ
    Dd = pairwise_sq(np.zeros((8, 3)))
    ld, md, idd = kmedoids(Dd, 4)
    assert idd["size_min"] >= 1 and len(set(md.tolist())) == 4, idd
    assert int(np.bincount(ld, minlength=4).min()) >= 1
    for bad in (0, 7):
        try:
            kmedoids(D, bad)
        except ValueError as e:
            assert "вне" in str(e), e
        else:
            raise AssertionError(f"принято K={bad}")

    # --- МАРГИНАЛ И ИНФОРМАЦИЯ ------------------------------------------
    counts = np.array([[9.0, 1.0], [5.0, 5.0]])
    lp = marginal_logprob(counts, alpha=1.0)
    assert np.allclose(np.exp(lp).sum(1), 1.0)
    assert abs(float(np.exp(lp[0, 0])) - 10.0 / 12.0) < 1e-12
    assert abs(float(np.exp(lp[1, 0])) - 0.5) < 1e-12
    # НЕВИДЕННЫЙ КОД НЕ ДАЁТ БЕСКОНЕЧНОСТИ
    lp0 = marginal_logprob(np.array([[10.0, 0.0]]), alpha=1.0)
    assert np.isfinite(lp0).all() and float(np.exp(lp0[0, 1])) > 0.0
    try:
        marginal_logprob(counts, alpha=0.0)
    except ValueError as e:
        assert "положительным" in str(e), e
    else:
        raise AssertionError("принято нулевое сглаживание")
    assert abs(entropy_bits(np.array([[0.5, 0.5]])) - 1.0) < 1e-12
    assert abs(entropy_bits(np.array([[1.0, 0.0]]))) < 1e-12
    assert abs(entropy_bits(np.full((1, 2048), 1.0 / 2048))
               - 11.0) < 1e-9
    # ИЗВЛЕЧЁННОЕ СЧИТАЕТСЯ ОТ BASELINE, А НЕ ОТ log2(V)
    assert abs(extracted_bits(np.log(2048.0), 4.6984) - 4.222) < 1e-3
    assert abs(extracted_bits(5.0, 5.0)) < 1e-12
    assert extracted_bits(4.0, 5.0) < 0.0

    # --- RECALL ---------------------------------------------------------
    lg = torch.tensor([[[0.0, 5.0, 1.0, 2.0]]])
    assert recall_at(lg, torch.tensor([[1]]), (1, 2), torch) == {1: 1.0,
                                                                2: 1.0}
    assert recall_at(lg, torch.tensor([[3]]), (1, 2, 3), torch) == {
        1: 0.0, 2: 1.0, 3: 1.0}
    assert recall_at(lg, torch.tensor([[0]]), (1, 4), torch)[4] == 1.0
    try:
        recall_at(lg, torch.tensor([[0]]), (5,), torch)
    except ValueError as e:
        assert "больше словаря" in str(e), e
    else:
        raise AssertionError("принят k больше словаря")

    # --- АГРЕГАЦИЯ В КЛАСТЕРЫ И МЯГКОЕ СРЕДНЕЕ ВНУТРИ --------------------
    pr = torch.tensor([[[0.5, 0.2, 0.25, 0.05],
                        [0.1, 0.1, 0.1, 0.7]]])
    lb = torch.tensor([0, 0, 1, 1])
    bk = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 5.0]])
    pc, choice = cluster_choice(pr, lb, 2, torch)
    assert torch.allclose(pc, torch.tensor([[[0.7, 0.3], [0.2, 0.8]]]))
    assert choice.tolist() == [[0, 1]], choice
    med = torch.tensor([0, 2])
    fb = bk[med[choice]]
    emb, deg = soft_inside(pr, lb, choice, bk, fb, torch)
    assert not bool(deg.any())
    # позиция 0: кластер 0, веса 0.5 и 0.2 -> (0.5*1 + 0.2*3)/0.7
    assert abs(float(emb[0, 0, 0]) - (0.5 + 0.6) / 0.7) < 1e-6, emb
    assert abs(float(emb[0, 0, 1])) < 1e-9
    # позиция 1: кластер 1, веса 0.1 и 0.7 -> (0.1*1 + 0.7*5)/0.8
    assert abs(float(emb[0, 1, 1]) - (0.1 + 3.5) / 0.8) < 1e-6, emb
    # ВЫРОЖДЕННАЯ МАССА -> ПРЕДСТАВИТЕЛЬ, А НЕ НОЛЬ
    pr0 = torch.tensor([[[0.0, 0.0, 0.6, 0.4]]])
    ch0 = torch.tensor([[0]])
    fb0 = bk[torch.tensor([[0]])]
    emb0, deg0 = soft_inside(pr0, lb, ch0, bk, fb0, torch)
    assert bool(deg0.all()) and torch.equal(emb0, fb0), (emb0, deg0)
    # ТОЖДЕСТВЕННОЕ РАЗБИЕНИЕ: это и есть структурный инвариант прогона
    ident = torch.arange(4)
    pc_i, ch_i = cluster_choice(pr, ident, 4, torch)
    assert torch.equal(pc_i, pr) and torch.equal(ch_i, pr.argmax(-1))
    fb_i = bk[ident[ch_i]]
    emb_i, deg_i = soft_inside(pr, ident, ch_i, bk, fb_i, torch)
    assert not bool(deg_i.any())
    assert torch.equal(emb_i, bk[pr.argmax(-1)]), emb_i
    try:
        cluster_choice(pr, torch.arange(3), 2, torch)
    except ValueError as e:
        assert "кодов" in str(e), e
    else:
        raise AssertionError("принято неверное число меток")

    # --- КОРЗИНЫ И МАСКИ ------------------------------------------------
    assert match_bucket_index(0) == 0 and match_bucket_index(2) == 1
    assert match_bucket_index(5) == 2 and match_bucket_index(16) == 3
    for bad in (-1, 17):
        try:
            match_bucket_index(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"принято {bad}")
    g = torch.Generator().manual_seed(7)
    mk = random_position_mask(5, 16, 6, g, torch)
    assert mk.shape == (5, 16) and mk.sum(1).tolist() == [6] * 5
    g2 = torch.Generator().manual_seed(7)
    assert torch.equal(random_position_mask(5, 16, 6, g2, torch), mk)
    g3 = torch.Generator().manual_seed(8)
    assert not torch.equal(random_position_mask(5, 16, 6, g3, torch), mk)
    g4 = torch.Generator().manual_seed(1)
    assert int(random_position_mask(3, 16, 0, g4, torch).sum()) == 0
    assert int(random_position_mask(3, 16, 16, g4, torch).sum()) == 48
    try:
        random_position_mask(3, 16, 17, g4, torch)
    except ValueError as e:
        assert "вне" in str(e), e
    else:
        raise AssertionError("принято m больше числа позиций")

    # --- ИСХОД ----------------------------------------------------------
    per = {8: dict(capture_oracle_hard=0.80, capture_reader_hard=0.10),
           16: dict(capture_oracle_hard=0.70, capture_reader_hard=0.02),
           32: dict(capture_oracle_hard=0.40, capture_reader_hard=0.30)}
    d = cluster_decision(per, 0.034)
    assert d["code"] == 0 and d["passed_ks"] == [8], d
    assert abs(d["reader_required"] - 0.054) < 1e-12, d
    # ПОТОЛОК БЕЗ ЧИТАТЕЛЯ НЕ ПРОХОДИТ, И ЧИТАТЕЛЬ БЕЗ ПОТОЛКА ТОЖЕ
    only = {16: dict(capture_oracle_hard=0.70, capture_reader_hard=0.02),
            32: dict(capture_oracle_hard=0.40, capture_reader_hard=0.30)}
    assert cluster_decision(only, 0.034)["code"] == 4
    # РОВНО НА ПОРОГЕ — ПРОХОДИТ
    edge = {16: dict(capture_oracle_hard=0.50, capture_reader_hard=0.054)}
    assert cluster_decision(edge, 0.034)["code"] == 0
    tiny = {16: dict(capture_oracle_hard=0.50, capture_reader_hard=0.0539)}
    assert cluster_decision(tiny, 0.034)["code"] == 4
    assert cluster_decision(per, None)["code"] == 3
    assert cluster_decision({16: dict(capture_oracle_hard=None,
                                      capture_reader_hard=None)},
                            0.034)["code"] == 4

    # --- ЧУЖИЕ ЧИСТЫЕ ФУНКЦИИ БЕРУТСЯ ИЗ ОДНОГО МЕСТА -------------------
    # Доля разрыва и белый список фазы импортируются из тренера, а не
    # копируются: расходиться им нечем.
    assert abs(trainer.capture(0.145469, 0.072892, 0.142997)
               - 0.034061) < 1e-5
    assert trainer.ZERO_TAG == "e0s0"
    assert 2048 not in KS_DEFAULT

    # --- КЛЮЧИ АРТЕФАКТА: КОЛЛИЗИЯ С gate_info НЕ ДОЛЖНА БЫТЬ ВОЗМОЖНА -
    # `payload` собирается как dict(... , **ctx.gate_info), и явный ключ,
    # который уже есть в gate_info, даёт `TypeError: dict() got multiple
    # values for keyword argument` — ПОСЛЕ всего прохода. Этот класс уже
    # случался с именем `rows`. Разбор дерева ловит его без GPU.
    import ast
    gate_keys = ("init_gate", "init_gate_sha1", "init_gate_run_id",
                 "init_gate_n_batches", "init_gate_causal",
                 "init_gate_code_version", "decoder_context")
    tree = ast.parse(open(os.path.abspath(__file__), encoding="utf-8")
                     .read())
    found = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "dict"
                and any(k.arg == "kind" for k in node.keywords
                        if isinstance(k, ast.keyword))):
            continue
        explicit = [k.arg for k in node.keywords if k.arg]
        assert len(explicit) == len(set(explicit)), explicit
        clash = sorted(set(explicit) & set(gate_keys))
        assert not clash, f"строка {node.lineno}: {clash} уже в gate_info"
        found += 1
    assert found >= 1, "не нашёлся сбор артефакта"

    print(f"самопроверка k15b_measure_interface пройдена: "
          f"K по умолчанию {list(KS_DEFAULT)}, пороги "
          f"oracle >= {ORACLE_CAPTURE_MIN}, reader >= capture + "
          f"{READER_CAPTURE_MARGIN}")


# --- ПРОВЕРКА ЧЕКПОЙНТА ЧИТАТЕЛЯ -----------------------------------------

READER_KINDS = ("k15b_q1_reader",)
READER_FIELDS = ("kind", "phase", "trainable_names", "state",
                 "selected_tag", "selected_state_sha1", "c1_sha1",
                 "target_content_sha1", "frozen_content_sha", "codec",
                 "code_version", "joint_sha1", "q0_prov", "history")


def check_reader(obj, ctx, book, cache_meta, frozen_sha, train_names):
    """Совместимость чекпойнта читателя с текущей обстановкой. Fail-closed."""
    problems = []
    missing = [f for f in READER_FIELDS if f not in obj]
    if missing:
        problems.append(f"нет обязательных полей {missing}")

    def same(name, got, want):
        if got != want:
            problems.append(f"{name}: в чекпойнте {got!r}, сейчас {want!r}")

    same("codec", obj.get("codec"), ctx.codec_fp)
    same("code_version", obj.get("code_version"), ctx.code_version)
    same("joint_sha1", obj.get("joint_sha1"), ctx.joint_sha)
    same("phase", str(obj.get("phase")), "q1_reader")
    for key in ("plan_sha1", "q0_manifest_sha1"):
        same(f"q0_prov.{key}", (obj.get("q0_prov") or {}).get(key),
             ctx.q0_prov.get(key))
    # ОДНА И ТА ЖЕ КНИГА И ОДИН И ТОТ ЖЕ КЭШ ЦЕЛИ: иначе маргинал строился
    # бы по целям, которых читатель не видел, а кластеры — по другой книге.
    same("c1_sha1", obj.get("c1_sha1"), book["c1_sha1"])
    same("target_content_sha1", obj.get("target_content_sha1"),
         cache_meta.get("content_sha1"))
    same("frozen_content_sha", obj.get("frozen_content_sha"), frozen_sha)
    if set(obj.get("trainable_names") or []) != set(train_names):
        problems.append("белый список чекпойнта не совпал с фазой")
    return problems


def reader_points(obj):
    """Какие точки доступны в чекпойнте и какая выбрана по умолчанию."""
    sel = str(obj.get("selected_tag"))
    all_states = obj.get("all_states") or None
    if all_states:
        if sel not in all_states:
            raise SystemExit(
                f"в all_states нет выбранной точки {sel!r}: "
                f"{sorted(all_states)}")
        return sel, dict(all_states), True
    return sel, {sel: obj["state"]}, False


def main():
    import k15_context
    import k15b_probe_and_extract as probe
    import k15b_train_stagewise as trainer
    import k15b_build_rankpath_cache as cachelib
    from k15_train_depth_rvq import H_EXEC

    ap = argparse.ArgumentParser(
        description="K-15b: замер интерфейса между h18 и книгой C1")
    k15_context.add_common_arguments(ap)
    ap.add_argument("--c1", default="data/k15b/c1_selected.pt")
    ap.add_argument("--target", default="data/k15b/rankpath_target_train.npz")
    ap.add_argument("--checkpoint", default="data/k15b/q1_reader_s0.pt")
    ap.add_argument("--point", default="",
                    help="тег точки читателя; пусто — выбранная в чекпойнте")
    ap.add_argument("--ks", default=",".join(str(k) for k in KS_DEFAULT),
                    help="размеры грубой книги через запятую")
    ap.add_argument("--batches", type=int, default=0,
                    help="батчей val_sel; 0 — все (сверка учителя с probe "
                         "возможна только при всех)")
    ap.add_argument("--support-every", type=int, default=5,
                    help="опора и диапазон грубых путей — каждый n-й батч")
    ap.add_argument("--out", default="data/k15b/coarse_clusters.pt")
    ap.add_argument("--summary",
                    default="reports/k15b/interface_measure.json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if int(a.limit) != 0:
        raise SystemExit("--limit не применяется: берите --batches")
    ks = sorted({int(x) for x in str(a.ks).split(",") if x.strip()})
    if not ks or min(ks) < 2:
        raise SystemExit(f"--ks {a.ks!r}: нужны размеры >= 2")
    for path in (a.out, a.summary):
        if os.path.exists(path) and not a.overwrite:
            raise SystemExit(f"{path} уже существует: без --overwrite не "
                             f"перезаписываю")
    archived = {p: probe.archive_existing(p) for p in (a.out, a.summary)}
    for path, dest in archived.items():
        if dest:
            print(f"  прежний {path} перенесён в {dest}")

    ctx = k15_context.build(a)
    torch = ctx.torch
    import torch.nn.functional as F
    dev = ctx.dev
    model, codec = ctx.model, ctx.codec
    k15t = k15_context.k15t
    vocab = int(ctx.vocab)
    if vocab in ks:
        raise SystemExit(f"--ks содержит {vocab}: вырожденное K добавляется "
                         f"автоматически как структурный инвариант")
    ks_all = ks + [vocab]

    # --- КНИГА ----------------------------------------------------------
    book = torch.load(a.c1, map_location="cpu", weights_only=False)
    if book.get("kind") != "k15b_c1_selected" \
            or book.get("accepted") is not True:
        raise SystemExit(
            f"{a.c1}: kind {book.get('kind')!r}, accepted "
            f"{book.get('accepted')!r}. Книга, не прошедшая гейты probe, "
            f"предметом замера быть не может")
    for key, want in (("codec", ctx.codec_fp),
                      ("code_version", ctx.code_version),
                      ("joint_sha1", ctx.joint_sha)):
        if book.get(key) != want:
            raise SystemExit(f"{a.c1}: {key} выбран в другой обстановке")

    def book_sha():
        return hashlib.sha1(np.ascontiguousarray(
            model.depth_aligned_book(1).detach().float().cpu().numpy()
        ).tobytes()).hexdigest()[:12]

    if book_sha() != book["c1_sha1"]:
        with torch.no_grad():
            model.depth_aligned_c1.copy_(book["c1"].to(
                model.depth_aligned_c1.device, model.depth_aligned_c1.dtype))
    if book_sha() != book["c1_sha1"]:
        raise SystemExit("после загрузки книга не та")
    print(f"  книга {book['c1_sha1']} из состояния "
          f"{book.get('source_state')!r}, цель "
          f"{book.get('teacher_target')}")

    all_names = list(ctx.info["names"])
    train_names = trainer.phase_names(all_names, "q1_reader")
    frozen_sha, n_frozen, n_elem = k15t.frozen_content_sha(
        model, torch, set(train_names))
    print(f"  замороженного {n_frozen} тензоров ({n_elem} значений), "
          f"отпечаток {frozen_sha}")

    # --- КЭШ ЦЕЛИ: ПО НЕМУ СТРОИТСЯ МАРГИНАЛ ----------------------------
    if not os.path.exists(a.target):
        raise SystemExit(f"нет {a.target}")
    cache = np.load(a.target, allow_pickle=True)
    meta = json.loads(str(cache["meta"]))
    n_pos_model = int(np.asarray(ctx.q0_can).shape[1])
    expect = dict(
        q0_prov=ctx.q0_prov, plan_sha1=ctx.q0_prov.get("plan_sha1"),
        topk=int(book["topk"]), codec=ctx.codec_fp,
        code_version=ctx.code_version, joint_sha1=ctx.joint_sha,
        channel_weights=[float(x) for x in
                         ctx.weights_gate.detach().cpu().numpy()],
        decoder_context=ctx.decoder_context,
        rank_candidates_file_sha1=k15_context.sha12(
            inspect.getfile(probe.rank_candidates)),
        action_error_positions=int(H_EXEC),
        code_target_positions=n_pos_model)
    problems, tmetric = trainer.check_cache_contract(meta, book, expect)
    if problems:
        raise SystemExit("кэш цели не подходит: " + "; ".join(problems[:6]))
    rows_t = np.asarray(cache["rows"], np.int64)
    codes_t = np.asarray(cache["codes"], np.int64)
    ranks_t = np.asarray(cache["ranks"], np.int16)
    plan_rows, _slot = cachelib.build_row_index(ctx.parts_full["train"])
    if not np.array_equal(rows_t, plan_rows):
        raise SystemExit("порядок строк кэша не совпал с планом")
    if cachelib.cache_fingerprint(rows_t, codes_t, ranks_t) \
            != meta["content_sha1"]:
        raise SystemExit("отпечаток содержимого кэша не совпал")
    n_pos = int(codes_t.shape[1])
    if n_pos != n_pos_model:
        raise SystemExit(f"кодовых позиций в кэше {n_pos}, у модели "
                         f"{n_pos_model}")
    if n_pos > MATCH_BUCKETS[-1][1]:
        raise SystemExit(
            f"кодовых позиций {n_pos}, а корзины совпадений покрывают "
            f"только до {MATCH_BUCKETS[-1][1]}: часть строк не попала бы "
            f"ни в одну и стратификация молча потеряла бы их")
    topk = int(meta["topk"])
    print(f"  кэш {meta['content_sha1']}: {rows_t.size} строк train, "
          f"{n_pos} кодовых позиций, topk {topk}")

    # --- МАРГИНАЛ ПО ПОЗИЦИЯМ, ПО ЦЕЛЯМ TRAIN ---------------------------
    counts = np.zeros((n_pos, vocab), np.float64)
    for t in range(n_pos):
        counts[t] = np.bincount(codes_t[:, t], minlength=vocab)
    lp_np = marginal_logprob(counts, alpha=MARGINAL_ALPHA)
    p_np = np.exp(lp_np)
    h_train_bits = entropy_bits(p_np)
    marg_top1 = p_np.argmax(1)
    used_train = int(np.unique(codes_t).size)
    print(f"  маргинал цели на train: энтропия {h_train_bits:.3f} бит из "
          f"{np.log2(vocab):.3f} возможных, различных кодов "
          f"{used_train}/{vocab}, сглаживание alpha={MARGINAL_ALPHA}")
    lp_dev = torch.as_tensor(lp_np, device=dev, dtype=torch.float32)
    marg_top1_dev = torch.as_tensor(marg_top1, device=dev,
                                    dtype=torch.long)

    # --- ЧИТАТЕЛЬ -------------------------------------------------------
    if not os.path.exists(a.checkpoint):
        raise SystemExit(f"нет {a.checkpoint}")
    obj = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    kind = str(obj.get("kind"))
    if kind not in READER_KINDS:
        raise SystemExit(
            f"{a.checkpoint} описывает {kind!r}: smoke-чекпойнт и чужие "
            f"артефакты предметом замера быть не могут")
    problems = check_reader(obj, ctx, book, meta, frozen_sha, train_names)
    if problems:
        raise SystemExit("чекпойнт читателя собран в другой обстановке: "
                         + "; ".join(problems[:6]))
    sel_tag, states, has_all = reader_points(obj)
    point = str(a.point) or sel_tag
    if point not in states:
        raise SystemExit(
            f"точки {point!r} в чекпойнте нет. Доступны {sorted(states)}"
            + ("" if has_all else
               ". Прогон 02.10.2026 сохранил только выбранную точку; "
               "остальные состояния пишет исправленный тренер в all_states"))
    own = dict(model.state_dict())
    with torch.no_grad():
        for k_, v_ in states[point].items():
            if tuple(v_.shape) != tuple(own[k_].shape):
                raise SystemExit(f"{point}.{k_}: форма не та")
            if not torch.isfinite(v_).all():
                raise SystemExit(f"{point}.{k_}: нечисловые значения")
            own[k_].copy_(v_.to(own[k_].device, own[k_].dtype))
    point_sha = ctx.k14c.state_sha(
        {k_: model.state_dict()[k_].detach().float().cpu().numpy()
         for k_ in train_names})
    if point == sel_tag and point_sha != str(obj["selected_state_sha1"]):
        raise SystemExit(f"после загрузки отпечаток {point_sha}, в "
                         f"чекпойнте {obj['selected_state_sha1']}")
    again, _n, _e = k15t.frozen_content_sha(model, torch, set(train_names))
    if again != frozen_sha:
        raise SystemExit(f"подстановка тронула не только белый список: "
                         f"{again} против {frozen_sha}")
    hist = {str(h.get("tag")): h for h in (obj.get("history") or [])}
    hp = hist.get(point) or {}
    print(f"  читатель: точка {point} ({point_sha}), в истории RMS "
          f"{hp.get('val_rms_a1_pol')}, CE {hp.get('val_ce')}; "
          f"{'есть all_states' if has_all else 'ТОЛЬКО выбранная точка'}")
    if point != sel_tag:
        print(f"    отпечаток точки {point} вычислен, но сверять его не с "
              f"чем: тренер пишет selected_state_sha1 только для выбранной")
    if point == sel_tag and point == trainer.ZERO_TAG:
        print("    ВНИМАНИЕ: это НЕОБУЧЕННАЯ точка — инициализация K-14. "
              "Обученные состояния прогона 02.10.2026 не сохранялись")
    model.eval()

    # --- КЛАСТЕРИЗАЦИЯ КНИГИ --------------------------------------------
    # Метрика — квадрат евклидовой в латенте, та же, по которой кодек и
    # токенизатор выбирают код. Представитель — медоид, то есть НАСТОЯЩАЯ
    # строка книги: центроид вывел бы латент с многообразия сумм книг
    # кодека, где обученная книга в probe и провалила опору.
    c1_np = model.depth_aligned_book(1).detach().float().cpu().numpy()
    D = pairwise_sq(c1_np)
    row_norm = np.linalg.norm(np.asarray(c1_np, np.float64), axis=-1)
    clusters, cl_info = {}, {}
    t_cl = time.time()
    for K in ks_all:
        if K == vocab:
            # ВЫРОЖДЕННЫЙ СЛУЧАЙ СТРОИТСЯ, А НЕ ИЩЕТСЯ. Тождественное
            # разбиение нужно как структурный инвариант, и порядок метк
            # должен совпадать с порядком кодов: иначе argmax по кластерам
            # и argmax по логитам ломались бы на точных ничьих.
            labels = np.arange(vocab, dtype=np.int64)
            med = np.arange(vocab, dtype=np.int64)
            info = dict(K=int(K), construction="identity", iterations=0,
                        last_moved=0, size_min=1, size_max=1,
                        size_median=1.0, inertia=0.0, mean_sq_to_medoid=0.0)
        else:
            labels, med, info = kmedoids(D, K)
            info["construction"] = "kmedoids"
        if int(np.bincount(labels, minlength=K).min()) <= 0:
            raise SystemExit(f"K={K}: есть пустой кластер")
        if len(set(med.tolist())) != K:
            raise SystemExit(f"K={K}: представители не различны")
        for c in range(K):
            if int(labels[int(med[c])]) != c:
                raise SystemExit(f"K={K}: представитель {med[c]} лежит не "
                                 f"в своём кластере")
        rel = np.sqrt(D[np.arange(vocab), med[labels]]) / np.maximum(
            row_norm, 1e-12)
        info.update(fingerprint=clustering_fingerprint(labels, med),
                    rel_to_medoid_mean=float(rel.mean()),
                    rel_to_medoid_p95=float(np.percentile(rel, 95)),
                    rel_to_medoid_max=float(rel.max()))
        clusters[K] = (labels, med)
        cl_info[K] = info
        print(f"  K={K:5d} ({info['construction']}): размеры "
              f"{info['size_min']}..{info['size_max']}, медиана "
              f"{info['size_median']:.0f}; относительное расстояние до "
              f"представителя: среднее {info['rel_to_medoid_mean']:.4f}, "
              f"p95 {info['rel_to_medoid_p95']:.4f}; отпечаток "
              f"{info['fingerprint']}")
    print(f"  кластеризация заняла {time.time() - t_cl:.0f} с")
    lab_dev = {K: torch.as_tensor(clusters[K][0], device=dev,
                                  dtype=torch.long) for K in ks_all}
    med_dev = {K: torch.as_tensor(clusters[K][1], device=dev,
                                  dtype=torch.long) for K in ks_all}
    lpm_cl = {}
    for K in ks_all:
        labels = clusters[K][0]
        agg = np.zeros((n_pos, K), np.float64)
        for c in range(K):
            agg[:, c] = p_np[:, labels == c].sum(1)
        lpm_cl[K] = torch.as_tensor(np.log(np.maximum(agg, 1e-300)),
                                    device=dev, dtype=torch.float32)
        cl_info[K]["marginal_entropy_bits"] = entropy_bits(agg)

    # --- ОДИН ПРОХОД ----------------------------------------------------
    batch_list = ctx.parts["val_sel"]
    take = probe.strided(len(batch_list), int(a.batches))
    chosen = [batch_list[i] for i in take]
    full_val = len(chosen) == len(batch_list)
    rows_all = np.unique(np.concatenate(
        [np.asarray(sel, np.int64) for _po, sel in chosen]))
    rows_sha = hashlib.sha1(
        np.ascontiguousarray(rows_all).tobytes()).hexdigest()[:12]
    if rows_all.size != sum(len(sel) for _po, sel in chosen):
        raise SystemExit("строки val_sel повторяются: цель зависела бы от "
                         "порядка обхода")
    print(f"  проход по {len(chosen)} батчам val_sel из {len(batch_list)}, "
          f"{rows_all.size} строк, отпечаток {rows_sha}"
          + ("" if full_val else "  (НЕ вся часть: сверка учителя с probe "
                                 "будет пропущена)"))

    PATHS = ["a0", "a1_tok", "a1_rank", "a1_pol", "a1_soft"]
    CL_PATHS = ("oracle_hard", "oracle_soft", "reader_hard", "reader_soft")
    for K in ks_all:
        PATHS += [f"K{K}_{nm}" for nm in CL_PATHS]
    PATHS += [f"sub{m}" for m in SUBSTITUTE_M]
    acc = {k: 0.0 for k in PATHS}
    scal = dict(ce_reader=0.0, ce_marg=0.0, p_target=0.0, agree=0.0,
                marg_agree=0.0)
    rec = {int(k): 0.0 for k in RECALL_KS}
    cl_scal = {K: dict(ce_reader=0.0, ce_marg=0.0, agree=0.0,
                       oracle_soft_degenerate=0.0,
                       reader_soft_degenerate=0.0) for K in ks_all}
    buckets = [dict(sum=0.0, n=0) for _ in MATCH_BUCKETS]
    sup = {"reference": []}
    rng_abs = {}
    for nm in ["a1_pol", "a1_soft"] + [f"K{K}_{p}" for K in ks_all
                                       for p in CL_PATHS]:
        sup[nm] = []
        rng_abs[nm] = []
    q0_dev = torch.as_tensor(np.asarray(ctx.q0_can), device=dev)
    ac16 = torch.autocast(device_type=dev.type, dtype=ctx.dt)
    c1 = model.depth_aligned_book(1)
    gen = torch.Generator().manual_seed(int(a.seed))
    every = max(int(a.support_every), 1)
    rank_hist, n_rows, forecast = [], 0, None
    t0 = time.time()
    with torch.no_grad():
        for bi, (po, sel) in enumerate(chosen, start=1):
            b = ctx.build_batch(po, sel)
            am = b.get("attention_mask")
            with ac16:
                v, p_ids = model.build_inputs(position_offset=po, **b)
                out = model.forward_depth_aligned_rvq(
                    vlm_inputs_embeds=v, attention_mask=am,
                    position_ids=p_ids, mode="full", tau=1.0)
            bad = int((out["pred_codes"][0]
                       != q0_dev[torch.as_tensor(sel, device=dev)]).sum())
            if bad:
                raise SystemExit(f"q0 разошёлся с каноническим в {bad} "
                                 f"позициях")
            B = len(sel)
            n_rows += B
            w = float(B)
            z0 = out["policy_embeddings"][0]
            c1_pol = out["policy_embeddings"][1]
            probs = out["policy_probabilities"][1].float()
            logits = out["logits"][1].float()
            pred = out["pred_codes"][1]
            action = torch.from_numpy(
                np.asarray(ctx.ACT[sel], np.float32)).to(dev)[..., :7]
            z_e = codec._encode(action.float(), embodiment_ids=0).float()
            d1 = ctx.tok.mean_squared_distances(z_e - z0, c1)
            _lg, _em, i1, _p = ctx.tok.quantize_residual(
                z_e - z0, c1, temperature=1.0)
            near = probe.rank_candidates(d1, i1, topk, torch)
            errs, decoded = [], []
            for j in range(near.shape[-1]):
                dec_j = ctx.decode_fp32(z0 + c1[near[..., j]])
                rows_j, _m = k15t.weighted_row_error(
                    dec_j, action, ctx.weights_gate, torch)
                errs.append(rows_j)
                decoded.append(dec_j)
            stack = torch.stack(errs, 0)
            best_j = stack.argmin(0)
            idx_ = best_j.view(-1, 1, 1).expand(-1, near.shape[1], 1)
            target = near.gather(-1, idx_).squeeze(-1)
            rank_hist.append(best_j.cpu().numpy())

            def record(name, act, latent=None, keep=False):
                rows_, mean_ = k15t.weighted_row_error(
                    act, action, ctx.weights_gate, torch)
                acc[name] += float(mean_) * w
                if keep:
                    sup[name].append(k15t.decoder_support(
                        latent.detach(), codec, torch, ctx.quantizers,
                        ctx.nearest_code, ctx.code_contribution)[1])
                    rng_abs[name].append(act[:, :H_EXEC].abs()
                                         .reshape(-1, 7).cpu().numpy())
                return rows_

            record("a0", ctx.decode_fp32(z0))
            record("a1_tok", decoded[0])
            rows_idx = torch.arange(B, device=best_j.device)
            record("a1_rank", torch.stack(decoded, 0)[best_j, rows_idx])
            lat_pol = z0 + c1_pol
            rows_pol = record("a1_pol", ctx.decode_fp32(lat_pol), lat_pol,
                              True)
            lat_soft = z0.float() + probs @ c1.float()
            record("a1_soft", ctx.decode_fp32(lat_soft), lat_soft, True)
            sup["reference"].append(k15t.decoder_support(
                z_e.detach(), codec, torch, ctx.quantizers,
                ctx.nearest_code, ctx.code_contribution)[1])

            # --- ВЕРОЯТНОСТНЫЕ ЧИСЛА ЧИТАТЕЛЯ И BASELINE ---------------
            ce_r = F.cross_entropy(logits.reshape(-1, vocab),
                                   target.reshape(-1))
            scal["ce_reader"] += float(ce_r) * w
            lp_t = lp_dev.unsqueeze(0).expand(B, -1, -1).gather(
                -1, target.unsqueeze(-1)).squeeze(-1)
            scal["ce_marg"] += float((-lp_t).mean()) * w
            scal["p_target"] += float(probs.gather(
                -1, target.unsqueeze(-1)).squeeze(-1).mean()) * w
            scal["agree"] += float((pred == target).float().mean()) * w
            scal["marg_agree"] += float(
                (marg_top1_dev.unsqueeze(0) == target).float().mean()) * w
            for k_, v_ in recall_at(logits, target, RECALL_KS,
                                    torch).items():
                rec[int(k_)] += v_ * w

            # --- МЕХАНИЗМ: СТРАТИФИКАЦИЯ И КОНТРФАКТИЧЕСКАЯ КРИВАЯ -----
            nmatch = (pred == target).sum(1).cpu().numpy()
            rp = rows_pol.detach().cpu().numpy()
            for r_, nm_ in zip(rp, nmatch):
                sl = buckets[match_bucket_index(int(nm_))]
                sl["sum"] += float(r_)
                sl["n"] += 1
            for m in SUBSTITUTE_M:
                msk = random_position_mask(B, n_pos, m, gen, torch).to(dev)
                code_m = torch.where(msk, target, pred)
                if m == 0 and not torch.equal(code_m, pred):
                    raise SystemExit("подстановка нуля позиций изменила коды")
                if m == n_pos and not torch.equal(code_m, target):
                    raise SystemExit("подстановка всех позиций не дала цель")
                record(f"sub{m}", ctx.decode_fp32(z0 + c1[code_m]))

            # --- ГРУБЫЙ ИНТЕРФЕЙС -------------------------------------
            keep = (bi % every == 0)
            for K in ks_all:
                lab, md = lab_dev[K], med_dev[K]
                true_cl = lab[target]
                pc, read_cl = cluster_choice(probs, lab, K, torch)
                for nm, cl in (("oracle", true_cl), ("reader", read_cl)):
                    code = md[cl]
                    if K == vocab:
                        want = target if nm == "oracle" else pred
                        if not torch.equal(code, want):
                            raise SystemExit(
                                f"вырожденное K={K}: {nm}_hard дал не те "
                                f"коды — агрегация вероятностей или метки "
                                f"кластеров собраны неверно")
                    lat_h = z0 + c1[code]
                    record(f"K{K}_{nm}_hard", ctx.decode_fp32(lat_h),
                           lat_h, keep)
                    emb, deg = soft_inside(probs, lab, cl, c1.float(),
                                           c1[code].float(), torch)
                    cl_scal[K][f"{nm}_soft_degenerate"] += \
                        float(deg.float().mean()) * w
                    lat_s = z0.float() + emb
                    record(f"K{K}_{nm}_soft", ctx.decode_fp32(lat_s),
                           lat_s, keep)
                ce_cl = -torch.log(pc.gather(
                    -1, true_cl.unsqueeze(-1)).squeeze(-1)
                    .clamp_min(1e-30)).mean()
                cl_scal[K]["ce_reader"] += float(ce_cl) * w
                lm = lpm_cl[K].unsqueeze(0).expand(B, -1, -1).gather(
                    -1, true_cl.unsqueeze(-1)).squeeze(-1)
                cl_scal[K]["ce_marg"] += float((-lm).mean()) * w
                cl_scal[K]["agree"] += float(
                    (read_cl == true_cl).float().mean()) * w

            if bi == min(20, len(chosen)):
                forecast = k15t.forecast_runtime(time.time() - t0, bi,
                                                 len(chosen), 1)
                print(f"    прогноз: {forecast['per_batch_s']:.2f} с/батч, "
                      f"весь проход {forecast['total_h']:.2f} ч", flush=True)
            if bi % 100 == 0:
                print(f"    {bi}/{len(chosen)} батчей, "
                      f"{time.time() - t0:.0f} с", flush=True)
    elapsed = time.time() - t0
    n = max(n_rows, 1)
    rms = {k: float(np.sqrt(v / n)) for k, v in acc.items()}
    print(f"  проход занял {elapsed / 60:.1f} мин, {n_rows} строк")

    # --- СВЕРКИ ---------------------------------------------------------
    draft, teach = rms["a0"], rms["a1_rank"]
    book_teacher = book.get("teacher_rms")
    if full_val:
        if book_teacher is None:
            raise SystemExit(f"{a.c1}: нет teacher_rms, сверить учителя не "
                             f"с чем")
        rel = abs(teach - float(book_teacher)) / max(float(book_teacher),
                                                     1e-12)
        if rel > TEACHER_REL_LIMIT:
            raise SystemExit(
                f"учитель {teach!r} против записанного probe "
                f"{book_teacher!r} (относительно {rel:.2e}): считается не "
                f"то же самое")
        print(f"  учитель сошёлся с probe ({book_teacher:.6f}, "
              f"относительно {rel:.1e})")
    else:
        rel = None
        print("  сверка учителя с probe пропущена: взята не вся val_sel")

    consistency = {}
    for name, other, why in (
            ("sub0", "a1_pol", "подстановка нуля позиций — это и есть "
                                "исполняемый путь"),
            (f"sub{n_pos}", "a1_rank", "подстановка всех позиций — это и "
                                       "есть путь учителя"),
            (f"K{vocab}_oracle_hard", "a1_rank",
             "одноэлементные кластеры: представитель равен целевому коду"),
            (f"K{vocab}_reader_hard", "a1_pol",
             "одноэлементные кластеры: argmax по кластерам равен argmax "
             "по логитам"),
            (f"K{vocab}_oracle_soft", f"K{vocab}_oracle_hard",
             "мягкое среднее по одному элементу равно ему самому"),
            (f"K{vocab}_reader_soft", f"K{vocab}_reader_hard",
             "мягкое среднее по одному элементу равно ему самому")):
        got = abs(rms[name] - rms[other]) / max(rms[other], 1e-12)
        consistency[name] = dict(against=other, rel=float(got), why=why,
                                 limit=PATH_REL_LIMIT,
                                 passed=bool(got <= PATH_REL_LIMIT))
    # ЭТИ СВЕРКИ НЕ ВЫЗЫВАЮТ ОТКАЗ ПРОЦЕССА, А МЕНЯЮТ ИСХОД НА КОД 3.
    # Проход стоит полчаса, и выбрасывать его числа после него же — худшее
    # из решений: артефакт сохраняется, диагноз печатается, решение по
    # грубому интерфейсу при сломанном инварианте не принимается. Точное
    # равенство кодов при вырожденном K проверяется на КАЖДОМ батче и
    # останавливает прогон сразу, поэтому агрегация вероятностей прикрыта
    # отдельно и раньше. Здесь остаются сверки ДЕЙСТВИЙ, которые не могут
    # быть побитовыми: `policy_embeddings` приходит из autocast в dtype
    # модели, а `c1[code]` берётся в dtype книги.
    broken = sorted(k for k, v in consistency.items() if not v["passed"])
    worst = max(v["rel"] for v in consistency.values())
    if broken:
        print(f"  СТРУКТУРНЫЕ ИНВАРИАНТЫ НЕ СОШЛИСЬ: {broken}")
        for k_ in broken:
            v_ = consistency[k_]
            print(f"    {k_} против {v_['against']}: относительно "
                  f"{v_['rel']:.2e} при пределе {PATH_REL_LIMIT} — "
                  f"{v_['why']}")
    else:
        print(f"  структурные инварианты сошлись: {len(consistency)} "
              f"сверок, худшая относительная разность {worst:.2e}")

    # --- ОПОРА И ДИАПАЗОН -----------------------------------------------
    ref_all = torch.cat(sup["reference"])
    ref_p95 = float(torch.quantile(ref_all, 0.95))
    ref_p99 = float(torch.quantile(ref_all, 0.99))

    def sup_stats(name):
        if not sup.get(name):
            return None
        v = torch.cat(sup[name])
        p95 = float(torch.quantile(v, 0.95))
        return dict(n=int(v.numel()), mean=float(v.mean()), p95=p95,
                    p99=float(torch.quantile(v, 0.99)),
                    reference_p95=ref_p95, reference_p99=ref_p99,
                    passed=bool(p95 <= ref_p95),
                    rule="p95 остатка пути <= p95 остатка кодека на "
                         "истинном латенте")

    def rng_stats(name):
        if not rng_abs.get(name):
            return None
        flat = np.concatenate(rng_abs[name], axis=0)
        p99 = [float(x) for x in np.percentile(flat, 99.0, axis=0)]
        amax = [float(x) for x in flat.max(axis=0)]
        return dict(rows=int(flat.shape[0]), p99_candidate=p99,
                    absmax_candidate=amax,
                    p99_dataset=[float(x) for x in ctx.act_p99_dataset],
                    **probe.range_ok(p99, ctx.act_p99_dataset, amax))

    # --- ИНФОРМАЦИЯ -----------------------------------------------------
    ce_reader = scal["ce_reader"] / n
    ce_marg = scal["ce_marg"] / n
    information = dict(
        ce_reader=float(ce_reader), ce_marginal=float(ce_marg),
        extracted_bits=extracted_bits(ce_marg, ce_reader),
        marginal_entropy_bits_train=float(h_train_bits),
        upper_bound_bits=float(np.log2(vocab)),
        p_target_arithmetic_mean=float(scal["p_target"] / n),
        p_target_geometric_mean=float(np.exp(-ce_reader)),
        agree_top1_reader=float(scal["agree"] / n),
        agree_top1_marginal=float(scal["marg_agree"] / n),
        recall={int(k): float(v / n) for k, v in rec.items()},
        distinct_target_codes_train=int(used_train),
        vocab=int(vocab), smoothing_alpha=float(MARGINAL_ALPHA),
        note=("`extracted_bits` = (CE_marginal - CE_reader)/ln2 — это "
              "ВАРИАЦИОННАЯ НИЖНЯЯ ОЦЕНКА выигрыша над контекстно-"
              "независимым предсказателем, а НЕ взаимная информация между "
              "h18 и целью и не предел возможностей представления. "
              "Прежняя величина log2(2048) - CE была неверна: она "
              "предполагала равномерность целевых кодов и точность "
              "условного распределения читателя. Геометрическое среднее "
              "вероятности цели — это exp(-CE); арифметическое "
              "агрегируется отдельно и с ним не совпадает"))

    # --- МЕХАНИЗМ -------------------------------------------------------
    mechanism = dict(
        buckets=[dict(matched_positions=f"{lo}-{hi}" if lo != hi
                      else str(lo), rows=int(b["n"]),
                      share=float(b["n"] / n),
                      rms=(float(np.sqrt(b["sum"] / b["n"]))
                           if b["n"] else None))
                 for (lo, hi), b in zip(MATCH_BUCKETS, buckets)],
        substitution=[dict(m=int(m), rms=float(rms[f"sub{m}"]),
                           capture=trainer.capture(draft, teach,
                                                   rms[f"sub{m}"]))
                      for m in SUBSTITUTE_M],
        positions=int(n_pos),
        note=("Корзины — ошибка исполняемого пути в зависимости от числа "
              "совпавших КОДОВЫХ позиций из 16; это не то же, что средняя "
              "позиционная точность. Кривая подстановки: целевой код "
              "ставится в m случайных позиций из 16, остальные остаются "
              "предсказанными. Выпуклая кривая означает зачёт только за "
              "полное попадание, близкая к линейной — что частичное "
              "попадание помогает. m=0 и m=16 совпадают с a1_pol и "
              "a1_rank по построению и сверены как инварианты"))

    # --- ГРУБЫЙ ИНТЕРФЕЙС -----------------------------------------------
    capture_hard = trainer.capture(draft, teach, rms["a1_pol"])
    capture_soft = trainer.capture(draft, teach, rms["a1_soft"])
    per_k = {}
    for K in ks_all:
        row = dict(**cl_info[K])
        for nm in CL_PATHS:
            key = f"K{K}_{nm}"
            row[f"rms_{nm}"] = float(rms[key])
            row[f"capture_{nm}"] = trainer.capture(draft, teach, rms[key])
            row[f"support_{nm}"] = sup_stats(key)
            row[f"range_{nm}"] = rng_stats(key)
        row["ce_reader_cluster"] = float(cl_scal[K]["ce_reader"] / n)
        row["ce_marginal_cluster"] = float(cl_scal[K]["ce_marg"] / n)
        row["extracted_bits_cluster"] = extracted_bits(
            cl_scal[K]["ce_marg"] / n, cl_scal[K]["ce_reader"] / n)
        row["agree_cluster"] = float(cl_scal[K]["agree"] / n)
        row["upper_bound_bits_cluster"] = float(np.log2(K))
        row["oracle_soft_degenerate_share"] = float(
            cl_scal[K]["oracle_soft_degenerate"] / n)
        row["reader_soft_degenerate_share"] = float(
            cl_scal[K]["reader_soft_degenerate"] / n)
        per_k[K] = row
    decision = cluster_decision(
        {K: per_k[K] for K in ks}, capture_hard)
    if broken:
        decision = dict(code=3, outcome=(
            f"структурные инварианты путей не сошлись ({broken}): решение "
            f"по грубому интерфейсу не принимается, числа сохранены для "
            f"разбора"), passed_ks=[], broken=broken,
            would_have_been=decision)

    rc = np.concatenate(rank_hist)
    rank_stats = dict(
        share_rank1=float((rc == 0).mean()),
        share_last_rank=float((rc == topk - 1).mean()),
        mean_rank=float(rc.mean()),
        histogram={int(k): int(v) for k, v in
                   zip(*np.unique(rc, return_counts=True))},
        note=("`share_last_rank` — доля строк, у которых оптимум ВНУТРИ "
              "окна достигается на его границе. Это сигнал о возможном "
              "упоре в окно, но НЕ оценка частоты обрезания ни снизу, ни "
              "сверху: ошибка действия не обязана монотонно зависеть от "
              "латентного ранга, поэтому ни лучший ранг 10 не доказывает, "
              "что за окном есть лучший код, ни лучший ранг 3 этого не "
              "исключает"))

    # --- ОТЧЁТ ----------------------------------------------------------
    print(f"\n  ЧЕРНОВИК {draft:.6f}; латентная цель {rms['a1_tok']:.6f}; "
          f"УЧИТЕЛЬ rank-path {teach:.6f}")
    print(f"  нынешний жёсткий путь {rms['a1_pol']:.6f} (доля разрыва "
          f"{100 * (capture_hard or 0):.1f} %), мягкий "
          f"{rms['a1_soft']:.6f} ({100 * (capture_soft or 0):.1f} %)")
    print(f"\n  ИНФОРМАЦИЯ (всё на одних и тех же {n_rows} строках):")
    print(f"    CE маргинала по позициям {ce_marg:.4f}, CE читателя "
          f"{ce_reader:.4f} -> извлечено "
          f"{information['extracted_bits']:.3f} бит над baseline")
    print(f"    энтропия цели на train {h_train_bits:.3f} бит (не "
          f"{np.log2(vocab):.0f}); различных кодов {used_train}/{vocab}")
    print(f"    вероятность цели: арифметическое среднее "
          f"{100 * information['p_target_arithmetic_mean']:.3f} %, "
          f"геометрическое "
          f"{100 * information['p_target_geometric_mean']:.3f} %")
    print(f"    согласие top-1: читатель "
          f"{100 * information['agree_top1_reader']:.2f} %, маргинал "
          f"{100 * information['agree_top1_marginal']:.2f} % (вот это и "
          f"есть baseline, а не 1/{vocab})")
    print("    recall: " + ", ".join(
        f"@{k} {100 * information['recall'][k]:.2f} %"
        for k in sorted(information["recall"])))
    print("\n  МЕХАНИЗМ. Ошибка по числу совпавших кодовых позиций:")
    for row in mechanism["buckets"]:
        print(f"    {row['matched_positions']:>5s} из {n_pos}: "
              f"{row['rows']:6d} строк ({100 * row['share']:5.1f} %), RMS "
              + ("—" if row["rms"] is None else f"{row['rms']:.6f}"))
    print("    кривая подстановки целевого кода в m позиций: " + ", ".join(
        f"m={r['m']} {r['rms']:.6f} ({100 * (r['capture'] or 0):.0f} %)"
        for r in mechanism["substitution"]))
    print(f"\n  ГРУБЫЙ ИНТЕРФЕЙС (порог потолка {ORACLE_CAPTURE_MIN}, "
          f"порог читателя {decision.get('reader_required')}):")
    for K in ks_all:
        r = per_k[K]
        mark = "" if K in ks else "   [вырожденное, инвариант]"
        print(f"    K={K:5d}{mark}")
        print(f"      потолок:  hard {r['rms_oracle_hard']:.6f} "
              f"({100 * (r['capture_oracle_hard'] or 0):5.1f} %), soft "
              f"{r['rms_oracle_soft']:.6f} "
              f"({100 * (r['capture_oracle_soft'] or 0):5.1f} %)")
        print(f"      читатель: hard {r['rms_reader_hard']:.6f} "
              f"({100 * (r['capture_reader_hard'] or 0):5.1f} %), soft "
              f"{r['rms_reader_soft']:.6f} "
              f"({100 * (r['capture_reader_soft'] or 0):5.1f} %)")
        print(f"      кластер: согласие {100 * r['agree_cluster']:.2f} %, "
              f"CE {r['ce_reader_cluster']:.4f} против маргинала "
              f"{r['ce_marginal_cluster']:.4f} -> "
              f"{r['extracted_bits_cluster']:.3f} бит из "
              f"{r['upper_bound_bits_cluster']:.2f} возможных")
        for nm in CL_PATHS:
            s_, g_ = r[f"support_{nm}"], r[f"range_{nm}"]
            if s_ is not None:
                print(f"      {nm:12s} опора p95 {s_['p95']:.4f} при "
                      f"эталоне {s_['reference_p95']:.4f} "
                      f"({'ok' if s_['passed'] else 'ОТКАЗ'}), диапазон "
                      f"{'ok' if g_['passed'] else 'ОТКАЗ'} по "
                      f"{g_['rows']} строкам")
    print(f"\n  ИСХОД: {decision['outcome']} (код {decision['code']})")

    # --- СОХРАНЕНИЕ -----------------------------------------------------
    payload = dict(
        kind="k15b_coarse_clusters",
        accepted=bool(decision["code"] == 0),
        accepted_ks=[int(k) for k in decision["passed_ks"]],
        thresholds=dict(oracle_capture_min=ORACLE_CAPTURE_MIN,
                        reader_capture_margin=READER_CAPTURE_MARGIN,
                        declared="до данных, 03.10.2026"),
        metric=dict(space="latent", distance="squared euclidean",
                    representative="medoid (настоящая строка книги C1)",
                    seed=int(CLUSTER_SEED), iters=int(CLUSTER_ITERS)),
        clusters={int(K): dict(
            labels=np.asarray(clusters[K][0], np.int64),
            medoids=np.asarray(clusters[K][1], np.int64),
            **{k_: v_ for k_, v_ in cl_info[K].items()})
            for K in ks_all},
        per_k={int(K): {k_: v_ for k_, v_ in per_k[K].items()}
               for K in ks_all},
        decision=decision, information=information, mechanism=mechanism,
        rank_stats=rank_stats, rms=rms, consistency=consistency,
        capture_hard=capture_hard, capture_soft=capture_soft,
        teacher=dict(draft=float(draft), latent=float(rms["a1_tok"]),
                     rank=float(teach),
                     probe_teacher_rms=(None if book_teacher is None
                                        else float(book_teacher)),
                     relative_to_probe=rel, full_val_sel=bool(full_val)),
        support=dict(a1_pol=sup_stats("a1_pol"),
                     a1_soft=sup_stats("a1_soft"),
                     reference_p95=ref_p95, reference_p99=ref_p99),
        range={nm: rng_stats(nm) for nm in ("a1_pol", "a1_soft")},
        reader=dict(checkpoint=os.path.abspath(a.checkpoint), point=point,
                    point_state_sha1=point_sha, selected_tag=sel_tag,
                    has_all_states=bool(has_all),
                    trained=bool(point != trainer.ZERO_TAG),
                    history_entry=hp,
                    note=("прогон 02.10.2026 сохранил только выбранную "
                          "точку, и ею оказалась необученная e0s0; "
                          "обученные состояния того прогона не "
                          "существуют")),
        rows=int(n_rows), rows_sha1=rows_sha,
        batches=len(chosen), batches_in_part=len(batch_list),
        seconds=float(elapsed), forecast=forecast,
        support_every=int(every), ks=[int(k) for k in ks],
        degenerate_k=int(vocab), topk=int(topk),
        code_target_positions=int(n_pos),
        action_error_positions=int(H_EXEC),
        recall_ks=[int(k) for k in RECALL_KS],
        substitute_m=[int(m) for m in SUBSTITUTE_M],
        c1_file=os.path.abspath(a.c1), c1_sha1=book["c1_sha1"],
        target_file=os.path.abspath(a.target),
        target_content_sha1=meta["content_sha1"],
        frozen_content_sha=frozen_sha, codec=ctx.codec_fp,
        code_version=ctx.code_version, q0_prov=ctx.q0_prov,
        joint_sha1=ctx.joint_sha,
        # `decoder_context` НЕ передаётся отдельно: он уже внутри
        # `ctx.gate_info`, и явное дублирование дало бы `TypeError: dict()
        # got multiple values for keyword argument` — ту же ошибку, что
        # когда-то дало имя `rows`, и тоже спустя весь проход.
        git_head=ctx.git_head, git_dirty=bool(ctx.dirty),
        archived={k: v for k, v in archived.items() if v},
        **ctx.gate_info)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                exist_ok=True)
    tmp = a.out + f".tmp.{os.getpid()}"
    torch.save(payload, tmp)
    back = torch.load(tmp, map_location="cpu", weights_only=False)
    for K in ks_all:
        for fld in ("labels", "medoids"):
            if not np.array_equal(back["clusters"][int(K)][fld],
                                  payload["clusters"][int(K)][fld]):
                os.unlink(tmp)
                raise SystemExit(f"после чтения clusters[{K}].{fld} "
                                 f"изменился")
    os.replace(tmp, a.out)
    print(f"  сохранено: {a.out} (обратное чтение сошлось)")

    light = {k: v for k, v in payload.items() if k != "clusters"}
    light["clusters"] = {int(K): {k_: v_ for k_, v_ in cl_info[K].items()}
                         for K in ks_all}
    os.makedirs(os.path.dirname(os.path.abspath(a.summary)) or ".",
                exist_ok=True)
    tmp = a.summary + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(light, fh, ensure_ascii=False, indent=1, allow_nan=False,
                  default=k15t.json_scalar)
    os.replace(tmp, a.summary)
    print(f"  сводка: {a.summary}")
    return int(decision["code"])


if __name__ == "__main__":
    sys.exit(main())
