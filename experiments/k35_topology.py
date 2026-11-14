#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-35: топология фиксированного бюджета 4 -> 8 -> 12 и 36 допустимых срезов.

Канонические ID: корни r = 0…3; средние m = 2r + side (0…7); листья
l = 2m + side (0…15). PAIRS — пары из четырёх в лексикографическом порядке.
Первое решение a выбирает два корня PAIRS[a]; из них создаются четыре
средних, отсортированных по m. Второе решение b выбирает две ЛОКАЛЬНЫЕ
позиции PAIRS[b] среди этих четырёх средних; из них — четыре листа,
отсортированных по l. Срез c = 6a + b (36 срезов), a = c // 6, b = c % 6.
Конечный срез: 2 корня + 2 средних + 4 листа = 8 узлов; каждый из 16
атомарных листовых участков покрыт ровно одним конечным предком.
RoPE-слоты — как в K-22: корни (1, 5, 9, 13), средний m — 2m, лист l — l.
parent_local индексирует предыдущий компактный уровень: для среднего —
позицию корня среди четырёх корней, для листа — позицию среднего среди
четырёх созданных средних; канонический родитель хранится отдельно.
Q action-query-layer = 8·4 + 8·8 + 8·12 = 192; router-Q = 4+4+8+8+12+12 = 48.
"""
import sys

import numpy as np

PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
N_CUTS = 36
ROOT_SLOTS = (1, 5, 9, 13)
STAGE_ACTIVE = (4, 8, 12)
ACTIVE = [4] * 8 + [8] * 8 + [12] * 8
ROUTER_ACTIVE = [4, 4, 8, 8, 12, 12]
Q_OPS, ROUTER_Q = 192, 48


def encode(a, b):
    return 6 * int(a) + int(b)


def decode(c):
    c = int(c)
    if not 0 <= c < N_CUTS:
        raise SystemExit(f"срез {c} вне 0…35")
    return c // 6, c % 6


def cut(c):
    """Полное описание среза c (канонические ID, родители, слоты)."""
    a, b = decode(c)
    split_roots = list(PAIRS[a])
    term_roots = [r for r in range(4) if r not in split_roots]
    mids = sorted(2 * r + s for r in split_roots for s in (0, 1))
    mid_parent = [m // 2 for m in mids]               # канонический корень
    mid_parent_local = list(mid_parent)               # корни все четыре
    split_mid_local = list(PAIRS[b])
    split_mids = [mids[i] for i in split_mid_local]
    term_mid_local = [i for i in range(4) if i not in split_mid_local]
    term_mids = [mids[i] for i in term_mid_local]
    leaves = sorted(2 * m + s for m in split_mids for s in (0, 1))
    leaf_parent = [l // 2 for l in leaves]
    leaf_parent_local = [mids.index(p) for p in leaf_parent]
    return dict(
        c=int(c), a=a, b=b, split_roots=split_roots, term_roots=term_roots,
        mids=mids, mid_parent=mid_parent, mid_parent_local=mid_parent_local,
        split_mid_local=split_mid_local, split_mids=split_mids,
        term_mid_local=term_mid_local, term_mids=term_mids, leaves=leaves,
        leaf_parent=leaf_parent, leaf_parent_local=leaf_parent_local,
        slots2=list(ROOT_SLOTS) + [2 * m for m in mids],
        slots3=list(ROOT_SLOTS) + [2 * m for m in mids] + list(leaves),
        terminal=dict(roots=term_roots, mids=term_mids, leaves=leaves))


CUTS = [cut(c) for c in range(N_CUTS)]


def tables():
    """Целочисленные таблицы [36, k] для батчевого gather."""
    keys = ("split_roots", "term_roots", "mids", "mid_parent_local",
            "split_mid_local", "term_mid_local", "leaves",
            "leaf_parent_local", "slots2", "slots3")
    return {k: np.asarray([C[k] for C in CUTS], np.int64) for k in keys}


def atomic_cover(c):
    """Какой конечный узел покрывает каждый из 16 атомарных участков ->
    список длиной 16 (уровень, канонический ID)."""
    C = CUTS[c]
    out = [None] * 16
    for r in C["term_roots"]:
        for l in range(4 * r, 4 * r + 4):
            out[l] = (out[l] or []) + [("root", r)]
    for m in C["term_mids"]:
        for l in (2 * m, 2 * m + 1):
            out[l] = (out[l] or []) + [("mid", m)]
    for l in C["leaves"]:
        out[l] = (out[l] or []) + [("leaf", l)]
    return out


def selftest():
    import itertools
    # 1. 36 уникальных срезов, кодирование обратимо
    assert len(CUTS) == 36 and len({(tuple(C["term_roots"]),
                                     tuple(C["term_mids"]),
                                     tuple(C["leaves"])) for C in CUTS}) == 36
    for c in range(36):
        assert encode(*decode(c)) == c
    try:
        decode(36)
        raise AssertionError("c = 36 принят")
    except SystemExit:
        pass
    assert list(PAIRS) == list(itertools.combinations(range(4), 2))
    for C in CUTS:
        # 2. 2 корня + 2 средних + 4 листа
        assert len(C["term_roots"]) == 2 and len(C["term_mids"]) == 2 and \
            len(C["leaves"]) == 4
        # 3. каждый атомарный участок покрыт ровно одним конечным предком
        cov = atomic_cover(C["c"])
        assert all(x is not None and len(x) == 1 for x in cov), (C["c"], cov)
        # 5. канонические ID, родители, слоты
        assert C["mids"] == sorted(C["mids"]) and \
            C["leaves"] == sorted(C["leaves"])
        assert all(m // 2 in C["split_roots"] for m in C["mids"])
        assert all(l // 2 in C["split_mids"] for l in C["leaves"])
        assert [C["mids"][i] for i in C["leaf_parent_local"]] == \
            C["leaf_parent"]
        assert C["mid_parent_local"] == C["mid_parent"]
        assert len(C["slots2"]) == 8 and len(C["slots3"]) == 12
        assert C["slots2"][4:] == [2 * m for m in C["mids"]]
    # срез, где разделяются не первые корни и не первые средние
    C = CUTS[encode(5, 5)]
    assert C["split_roots"] == [2, 3] and C["mids"] == [4, 5, 6, 7] and \
        C["split_mids"] == [6, 7] and C["leaves"] == [12, 13, 14, 15] and \
        C["leaf_parent_local"] == [2, 2, 3, 3]
    assert sum(ACTIVE) == Q_OPS and sum(ROUTER_ACTIVE) == ROUTER_Q
    T = tables()
    assert T["slots3"].shape == (36, 12) and T["leaf_parent_local"].shape \
        == (36, 4)
    print("самопроверка k35_topology пройдена: 36 уникальных срезов, c ↔ "
          "(a, b) обратимо, 2+2+4 конечных узла, каждый из 16 атомарных "
          "участков покрыт ровно одним конечным предком, канонические ID, "
          "явные родители (локальные и канонические), слоты K-22, Q = 192, "
          "router-Q = 48")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
