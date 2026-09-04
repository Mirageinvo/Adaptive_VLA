"""Единственный источник правды о разметке событийных целей.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ. Разметка была продублирована в K-10c, K-10d и K-10e и
разъехалась: K-10e брал объединение и `side="left"`, K-10d — только схват и
`side="right"`, поэтому остатки, бакеты и пороги относились к разным
определениям цели, а сравнение было недействительным. Дублирования больше нет:
все три скрипта импортируют отсюда.

ГРАНИЦА ФАЗЫ. Действие, которым цель достигается, принадлежит ЗАВЕРШАЮЩЕЙСЯ
фазе. При `side="right"` сам момент переключения схвата получал бы уже
следующую цель, то есть обучался бы на чужой. Здесь `side="left"`, и в момент
события остаток равен нулю.

СЛИЯНИЕ ПРИВЯЗЫВАЕТСЯ К СХВАТУ. Когда остановка и схват попадают в одно окно,
объединённое событие ставится в момент СХВАТА, а не в самый ранний из двух:
иначе `target_grip_sign` брался бы за несколько шагов до переключения и
оказывался бы старым.
"""

import numpy as np

EVENT_KINDS = ("grip", "stop", "union")
# Типы событий в разметке. TERMINAL — конец эпизода, он не событие, но целью
# быть обязан, иначе хвост остался бы без цели.
EV_GRIP, EV_STOP, EV_TERMINAL = 1, 2, 0


def grip_events(actions):
    """Индексы смены знака команды схвата (канал 6)."""
    s = np.sign(np.asarray(actions)[:, 6])
    s[s == 0] = 1.0
    return np.where(np.diff(s) != 0)[0] + 1


def stop_events(state_xyz, speed_frac=0.3, min_dwell=3, min_travel=0.02):
    """Остановки: участки НИЗКОЙ скорости достаточной длительности.

    ПОЧЕМУ НЕ «ЛОКАЛЬНЫЙ МИНИМУМ». Поиск точки, где скорость не больше
    соседних с допуском, при РАВНОМЕРНОМ движении принимает каждую точку: на
    прямой из 300 шагов находилось 73 несуществующие остановки. Здесь порог
    абсолютный — доля от медианной скорости эпизода, и равномерное движение
    даёт ровно ноль событий.
    """
    xyz = np.asarray(state_xyz, np.float64)
    v = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    if len(v) < min_dwell:
        return np.array([], np.int64)
    med = float(np.median(v))
    if med <= 0:
        return np.array([], np.int64)
    slow = v < speed_frac * med
    ev, last, i, n = [], 0, 0, len(slow)
    while i < n:
        if not slow[i]:
            i += 1
            continue
        j = i
        while j < n and slow[j]:
            j += 1
        if j - i >= min_dwell:
            c = (i + j) // 2
            if np.linalg.norm(xyz[c] - xyz[last]) >= min_travel:
                ev.append(c)
                last = c
        i = j
    return np.asarray(ev, np.int64)


def merge_events(grip, stop, tol):
    """Объединение с привязкой слитого события к моменту СХВАТА.

    Возвращает (индексы, типы, число слияний). Схват и остановка часто
    отмечают один момент — рука замедляется, чтобы схватить, — и тогда
    согласие двух отдельных оценок не является независимым подтверждением.
    """
    g = np.asarray(grip, np.int64)
    s = np.asarray(stop, np.int64)
    used = np.zeros(len(s), bool)
    idx, typ, dup = list(g), [EV_GRIP] * len(g), 0
    for i, e in enumerate(s):
        if len(g) and np.abs(g - e).min() <= tol:
            used[i] = True
            dup += 1                       # слито в ближайший схват
        else:
            idx.append(int(e)); typ.append(EV_STOP)
    o = np.argsort(np.asarray(idx, np.int64))
    return np.asarray(idx, np.int64)[o], np.asarray(typ, np.int64)[o], dup


def label(actions, state_xyz, kind="union", speed_frac=0.3, min_dwell=3,
          min_travel=0.02, merge_tol=4):
    """События и их типы для эпизода. Единая точка входа для всех скриптов."""
    if kind not in EVENT_KINDS:
        raise ValueError(f"неизвестный вид события: {kind}")
    g = grip_events(actions)
    if kind == "grip":
        return g, np.full(len(g), EV_GRIP, np.int64), 0
    s = stop_events(state_xyz, speed_frac, min_dwell, min_travel)
    if kind == "stop":
        return s, np.full(len(s), EV_STOP, np.int64), 0
    return merge_events(g, s, merge_tol)


def targets(actions, events, ev_types):
    """Для каждого момента: индекс цели, её тип и признак конца эпизода.

    ГРАНИЦА У ЗАВЕРШАЮЩЕЙСЯ ФАЗЫ (`side="left"`): в момент события остаток
    равен нулю. После последнего события целью становится конец эпизода, и он
    помечается TERMINAL — иначе конец был бы неотличим от схвата.
    """
    a = np.asarray(actions)
    n = len(a)
    t = np.arange(n)
    ev = np.asarray(events, np.int64)
    if len(ev) == 0:
        tau = np.full(n, n - 1, np.int64)
        return tau, np.full(n, EV_TERMINAL, np.int64), tau - t
    nxt = np.searchsorted(ev, t, side="left")
    inside = nxt < len(ev)
    tau = np.where(inside, ev[np.minimum(nxt, len(ev) - 1)], n - 1)
    typ = np.where(inside, np.asarray(ev_types)[np.minimum(nxt, len(ev) - 1)],
                   EV_TERMINAL)
    return tau.astype(np.int64), typ.astype(np.int64), (tau - t).astype(np.int64)


def bucket_names(edges):
    return ([f"<= {edges[0]}"]
            + [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)]
            + [f"> {edges[-1]}"])


def bucketize(remaining, edges):
    return np.searchsorted(np.asarray(edges), np.asarray(remaining),
                           side="right")


def selftest():
    # --- схват -------------------------------------------------------------
    a = np.zeros((20, 7)); a[:, 6] = 1.0; a[7:, 6] = -1.0
    assert list(grip_events(a)) == [7]

    # --- остановки ---------------------------------------------------------
    line = np.stack([np.linspace(0, 1, 300)] * 3, 1)
    assert len(stop_events(line)) == 0, "равномерное движение — ноль событий"
    seg, pause = 40, 10
    parts, pos = [], np.zeros(3)
    fast, slow = np.array([0.01, 0, 0]), np.array([0.001, 0, 0])
    for k in range(4):
        parts.append(pos + np.arange(1, seg + 1)[:, None] * fast)
        pos = parts[-1][-1]
        if k < 3:
            parts.append(pos + np.arange(1, pause + 1)[:, None] * slow)
            pos = parts[-1][-1]
    soft = np.concatenate(parts)
    # ПАУЗА С НЕНУЛЕВОЙ СКОРОСТЬЮ различает значение speed_frac. Идеальная
    # пауза в ноль этого не может, и именно поэтому ошибка «0.02 вместо 0.3»
    # прожила полдня.
    assert len(stop_events(soft, speed_frac=0.3, min_travel=0.05)) == 3
    assert len(stop_events(soft, speed_frac=0.02, min_travel=0.05)) == 0

    # --- слияние привязано к схвату ----------------------------------------
    idx, typ, dup = merge_events([10, 50], [12, 90], tol=4)
    assert list(idx) == [10, 50, 90] and dup == 1, (list(idx), dup)
    assert list(typ) == [EV_GRIP, EV_GRIP, EV_STOP]
    # Слитое событие стоит в момент СХВАТА (10), а не остановки (12).
    assert 12 not in list(idx)

    # --- граница фазы ------------------------------------------------------
    acts = np.zeros((15, 7))
    tau, typ2, rem = targets(acts, [5, 12], [EV_GRIP, EV_GRIP])
    assert rem[5] == 0 and rem[12] == 0, "в момент события остаток нулевой"
    assert rem[0] == 5 and rem[6] == 6
    assert typ2[13] == EV_TERMINAL and rem[14] == 0
    # Без событий вся траектория целится в конец и помечена TERMINAL.
    tau0, typ0, rem0 = targets(np.zeros((4, 7)), [], [])
    assert list(rem0) == [3, 2, 1, 0] and set(typ0) == {EV_TERMINAL}

    # --- единая точка входа ------------------------------------------------
    acts2 = np.zeros((300, 7)); acts2[:, 6] = 1.0; acts2[150:, 6] = -1.0
    for kind in EVENT_KINDS:
        e, t_, _ = label(acts2, soft[:300], kind=kind)
        assert len(e) == len(t_)
    e_g, _, _ = label(acts2, soft[:300], kind="grip")
    assert list(e_g) == [150]

    assert bucket_names([8, 16])[0] == "<= 8"
    assert list(bucketize([0, 9, 30], [8, 16, 32])) == [0, 1, 2]
    print("самопроверка goal_events пройдена: схват, равномерное движение "
          "ноль, мягкая пауза различает порог, слияние привязано к схвату, "
          "граница у завершающейся фазы, TERMINAL отделён")


if __name__ == "__main__":
    selftest()
