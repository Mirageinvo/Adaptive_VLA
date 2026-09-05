"""Сборка троек «состояние, цель, действие». Общая для K-10d и K-10f.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ. Разметка событий уже дважды расходилась между
скриптами, пока не была вынесена в `goal_events`. Сборка входа контроллера —
то же самое место риска: диагностика ёмкости обязана строить ровно тот вход,
на котором получен отрицательный результат, иначе она измеряет другую задачу.

ВРАЩЕНИЯ НЕ ВЫЧИТАЮТСЯ: разность ориентаций считается через матрицы поворота
с устойчивой ветвью около pi.
"""

import math

import numpy as np

# Части полного входа. Разделены, чтобы измерить вклад каждой отдельно:
# «rich помог» само по себе ничего не говорит, потому что меняет четыре
# фактора разом. И один из них, `remaining`, ПРИВИЛЕГИРОВАННЫЙ — он
# известен только потому, что мы знаем, когда произойдёт событие;
# развёрнутая система его не знает без монитора, которого нет.
RICH_PARTS = ("dstate", "prevact", "remaining", "task")
ORACLE_PARTS = ("remaining",)


def aa_to_R(v):
    """Axis-angle -> матрица поворота (Родриг), пакетно."""
    v = np.asarray(v, np.float64)
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    k = np.divide(v, np.where(th > 1e-12, th, 1.0))
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    I = np.broadcast_to(np.eye(3), K.shape).copy()
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return I + s * K + (1 - c) * (K @ K)


def R_to_aa(R):
    """Матрица поворота -> axis-angle, с устойчивой ветвью около pi.

    Кососимметричная часть обращается в ноль И при theta=0, И при theta=pi;
    одна ветка «малый угол» возвращала бы для поворота на pi нулевой вектор.
    """
    R = np.asarray(R, np.float64)
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    th = np.arccos(tr)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2],
                  R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = v / np.where(n > 1e-12, n, 1.0) * th[..., None]
    flat, Rf = out.reshape(-1, 3), R.reshape(-1, 3, 3)
    trf, nf, vf = tr.reshape(-1), n.reshape(-1), v.reshape(-1, 3)
    for i in np.where(nf < 1e-8)[0]:
        if trf[i] > 0:
            flat[i] = vf[i] / 2.0
            continue
        A = (Rf[i] + np.eye(3)) / 2.0        # при theta=pi это a a^T
        k = int(np.argmax(np.diag(A)))
        ak = math.sqrt(max(A[k, k], 0.0))
        if ak < 1e-12:
            flat[i] = 0.0
            continue
        u = A[:, k] / ak
        flat[i] = u / max(np.linalg.norm(u), 1e-12) * np.pi
    return flat.reshape(out.shape)


def rel_goal(pos_t, aa_t, pos_g, aa_g):
    """Цель относительно текущей позы: смещение и поворот R_g * R_t^T."""
    dR = aa_to_R(aa_g) @ np.swapaxes(aa_to_R(aa_t), -1, -2)
    return np.concatenate([pos_g - pos_t, R_to_aa(dR)], axis=-1)


def build(a, st, tau, ttyp, rem, steps, state_norm=None, task_id=None,
          rich=False, parts=None, n_task=64):
    """Тройки для выбранных шагов эпизода.

    rich=False воспроизводит вход отрицательного результата K-10d: только
    нормированное состояние и цель. rich=True добавляет то, чего контроллеру
    не хватало и без чего задача недоопределена: приращение состояния,
    предыдущее действие, ОСТАТОК до цели и идентификатор задачи. Без
    идентификатора сорок задач требуют разного поведения от одинаковых
    (состояние, цель) — это невозможно по построению, и часть провала могла
    объясняться именно этим, а не отсутствием зрения.
    """
    t = np.asarray(steps, np.int64)
    g = rel_goal(st[t, :3], st[t, 3:6], st[tau[t], :3], st[tau[t], 3:6])
    oh = np.zeros((len(t), 3))
    oh[np.arange(len(t)), np.clip(ttyp[t], 0, 2)] = 1.0
    g = np.concatenate([g, np.sign(a[tau[t], 6:7]), oh], 1)
    sn = state_norm(st[t]) if state_norm is not None else st[t]
    if parts is None:
        parts = RICH_PARTS if rich else ()
    parts = tuple(parts)
    bad = [p for p in parts if p not in RICH_PARTS]
    if bad:
        raise ValueError(f"неизвестные части входа: {bad}; доступны "
                         f"{list(RICH_PARTS)}")
    if not parts:
        return sn.astype(np.float32), g.astype(np.float32), a[t].astype(np.float32)
    # НУЛЕВАЯ СТРОКА НЕ ИМЕЕТ ПРЕДЫДУЩЕГО ДЕЙСТВИЯ. При `prev = max(t-1, 0)`
    # на t=0 получалось `предыдущее действие = a[0] = таргет`, то есть прямая
    # утечка целевой величины во вход. Здесь такие строки получают нули и
    # отдельный признак «предыдущего нет».
    has_prev = (t > 0).astype(np.float64)[:, None]
    prev = np.maximum(t - 1, 0)
    # ВРАЩЕНИЕ НЕ ВЫЧИТАЕТСЯ: приращение позы считается через матрицы, как и
    # цель. Прямое вычитание axis-angle — та же ошибка, которую мы уже
    # исправляли в определении цели.
    dpose = rel_goal(st[prev, :3], st[prev, 3:6], st[t, :3], st[t, 3:6])
    dst = np.concatenate([dpose, st[t, 6:] - st[prev, 6:]], 1)
    dst = dst * has_prev
    pact = a[prev] * has_prev
    # ОСТАТОК ДАЁТСЯ ЯВНО: без него контроллер не знает, сколько у него шагов,
    # и не может распределить движение — величина команды в демонстрации от
    # этого зависит напрямую.
    rr = rem[t][:, None].astype(np.float64)
    extra = []
    if "dstate" in parts:
        extra.append(dst)
    if "prevact" in parts:
        extra += [pact, has_prev]
    if "remaining" in parts:
        extra.append(np.log1p(rr))
    if "task" in parts:
        # ОТКАЗ, А НЕ МОЛЧАЛИВЫЙ ПРОПУСК. Прежде при `task_id is None` признак
        # просто исчезал, и K-10f с task давал вход шириной как без него, а
        # K-10d — тем более, потому что вовсе не передавал идентификатор.
        # Две конфигурации с одинаковыми флагами решали разные задачи.
        if task_id is None:
            raise ValueError("запрошена часть «task», но task_id не передан")
        ti = int(task_id)
        # ОСТАТОК ОТ ДЕЛЕНИЯ УБРАН: он молча склеивал бы разные задачи в один
        # столбец при n_task меньше числа задач.
        if not 0 <= ti < n_task:
            raise ValueError(f"task_id {ti} вне диапазона [0, {n_task})")
        oh_t = np.zeros((len(t), n_task))
        oh_t[np.arange(len(t)), ti] = 1.0
        extra.append(oh_t)
    if not extra:
        return sn.astype(np.float32), g.astype(np.float32), a[t].astype(np.float32)
    sn = np.concatenate([sn] + extra, 1)
    return sn.astype(np.float32), g.astype(np.float32), a[t].astype(np.float32)


def make_ctrl():
    """Сеть контроллера. Общая, потому что её строят и K-10d, и K-10h.

    Раньше класс был объявлен внутри `main()` K-10d, и замкнутый цикл обязан
    был бы повторить его своей копией. Копии архитектуры расходятся молча:
    веса грузятся, ключи совпадают, поведение другое. Torch импортируется
    внутри, чтобы numpy-путь модуля оставался лёгким.
    """
    import torch.nn as nn

    class Ctrl(nn.Module):
        """Общий ствол, отдельные головы позы и знака схвата."""

        def __init__(self, din, hid):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(din, hid), nn.GELU(),
                                      nn.Linear(hid, hid), nn.GELU())
            self.pose = nn.Linear(hid, 6)
            self.grip = nn.Linear(hid, 1)

        def forward(self, x):
            h = self.body(x)
            return self.pose(h), self.grip(h).squeeze(-1)

    return Ctrl


def online_features(state, prev_state, prev_act, goal_pose, goal_grip_sign,
                    goal_type, parts, state_norm=None, task_id=None,
                    n_task=64, remaining=None):
    """Вход контроллера для ОДНОГО шага замкнутого цикла.

    СОБИРАЕТСЯ ТЕМ ЖЕ `build`, а не своей формулой. Порядок столбцов,
    обнуление отсутствующего предыдущего действия, кодирование типа события и
    перевод вращений через матрицы — всё это должно совпасть с обучением
    побитово, а повторённая вручную сборка разошлась бы при первой же правке
    одной из сторон. Поэтому здесь строится микроэпизод из трёх строк, где
    строка 0 — предыдущий шаг, строка 1 — текущий, строка 2 — цель, и у него
    запрашивается ровно один шаг.

    ОСТАТОК — ВЕЛИЧИНА ИЗ БУДУЩЕГО. Если он входит в `parts`, вызывающий
    обязан передать `remaining`; в замкнутом цикле честного источника у него
    нет, и рука превращается в верхнюю границу.
    """
    parts = tuple(parts)
    if "remaining" in parts and remaining is None:
        raise ValueError("часть «remaining» запрошена, но значение не "
                         "передано: в замкнутом цикле её неоткуда взять")
    cur = np.asarray(state, np.float64)
    gp = np.asarray(goal_pose, np.float64)
    if gp.shape[-1] < 6:
        raise ValueError(f"поза цели короче шести чисел: {gp.shape}")
    prev = cur if prev_state is None else np.asarray(prev_state, np.float64)
    st = np.zeros((3, len(cur)))
    st[0], st[1] = prev, cur
    st[2, :6] = gp[:6]
    a = np.zeros((3, 7))
    a[0] = np.zeros(7) if prev_act is None else np.asarray(prev_act,
                                                           np.float64)
    a[2, 6] = float(goal_grip_sign)
    t = 0 if prev_state is None else 1
    tau = np.full(3, 2, np.int64)
    ttyp = np.full(3, int(goal_type), np.int64)
    rem = np.zeros(3, np.int64)
    if remaining is not None:
        rem[:] = int(remaining)
    s_, g_, _ = build(a, st, tau, ttyp, rem, np.array([t]),
                      state_norm=state_norm, parts=parts, task_id=task_id,
                      n_task=n_task)
    return np.concatenate([s_[0], g_[0]]).astype(np.float32)


def selftest():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 0.5, size=(40, 3))
    v = v / np.linalg.norm(v, axis=-1, keepdims=True) * rng.uniform(
        0.05, 2.5, size=(40, 1))
    assert np.abs(R_to_aa(aa_to_R(v)) - v).max() < 1e-8
    for ax in ([0, 0, 1.0], [1.0, 0, 0], [0.6, 0.8, 0.0]):
        w = np.array([ax]) * np.pi
        got = R_to_aa(aa_to_R(w))
        assert abs(np.linalg.norm(got) - np.pi) < 1e-6
        assert np.abs(aa_to_R(got) - aa_to_R(w)).max() < 1e-9
    p = rng.normal(size=(8, 3)); q = rng.normal(0, 0.3, size=(8, 3))
    assert np.abs(rel_goal(p, q, p, q)).max() < 1e-8
    sh = rng.normal(size=(1, 3))
    assert np.abs(rel_goal(p, q, p + 1.0, q)
                  - rel_goal(p + sh, q, p + sh + 1.0, q)).max() < 1e-9

    # rich добавляет ровно то, что заявлено, и не трогает цель и таргет.
    n = 20
    a = rng.normal(0, 0.2, size=(n, 7)); st = rng.normal(size=(n, 8))
    tau = np.full(n, n - 1); ttyp = np.zeros(n, np.int64)
    rem = tau - np.arange(n)
    s0, g0, y0 = build(a, st, tau, ttyp, rem, np.arange(n - 1))
    s1, g1, y1 = build(a, st, tau, ttyp, rem, np.arange(n - 1), rich=True,
                       task_id=3, n_task=64)
    assert np.allclose(g0, g1) and np.allclose(y0, y1)
    assert s1.shape[1] == s0.shape[1] + 8 + 7 + 1 + 1 + 64, s1.shape
    assert np.allclose(s1[:, :s0.shape[1]], s0)

    # --- онлайновая сборка обязана совпасть с обучающей ---------------------
    # ЭТО ГЛАВНАЯ ПРОВЕРКА МОДУЛЯ ДЛЯ ЗАМКНУТОГО ЦИКЛА: если вход разойдётся,
    # контроллер загрузится и будет работать, выдавая осмысленные с виду, но
    # чужие команды.
    for pset in ((), ("dstate",), ("dstate", "prevact"),
                 ("dstate", "prevact", "task"),
                 ("dstate", "prevact", "remaining", "task")):
        tid = 3 if "task" in pset else None
        S_, G_, _ = build(a, st, tau, ttyp, rem, np.arange(n - 1),
                          parts=pset, task_id=tid, n_task=64)
        for tt in (0, 1, 5, n - 2):
            want = np.concatenate([S_[tt], G_[tt]])
            got = online_features(
                st[tt], None if tt == 0 else st[tt - 1],
                None if tt == 0 else a[tt - 1],
                st[int(tau[tt]), :6], np.sign(a[int(tau[tt]), 6]),
                int(ttyp[tt]), pset, task_id=tid, n_task=64,
                remaining=int(rem[tt]) if "remaining" in pset else None)
            assert got.shape == want.shape, (pset, tt, got.shape, want.shape)
            assert np.abs(got - want).max() < 1e-6, (
                pset, tt, float(np.abs(got - want).max()))
    # Остаток без значения — отказ, а не молчаливый ноль.
    try:
        online_features(st[1], st[0], a[0], st[-1, :6], 1.0, 0,
                        ("remaining",))
        raise AssertionError("remaining без значения прошёл")
    except ValueError:
        pass
    # Идентификатор задачи попал в свою позицию и только в неё.
    oh = s1[:, -64:]
    assert oh.sum() == len(s1) and oh[:, 3].all()
    # НЕТ УТЕЧКИ ТАРГЕТА В НУЛЕВОЙ СТРОКЕ: предыдущего действия там нет.
    d0 = s0.shape[1]
    prev_act = s1[:, d0 + 8:d0 + 8 + 7]
    assert np.abs(prev_act[0]).max() == 0.0, "нулевая строка несёт таргет"
    assert not np.allclose(prev_act[1], y1[1]), "сдвиг на шаг потерян"
    assert np.allclose(prev_act[1], y1[0]), "предыдущее действие не то"
    has_prev = s1[:, d0 + 8 + 7]
    assert has_prev[0] == 0.0 and has_prev[1:].all()
    # ПРИРАЩЕНИЕ СОСТОЯНИЯ в нулевой строке тоже обнулено.
    assert np.abs(s1[0, d0:d0 + 8]).max() == 0.0
    # ЧАСТИ ВКЛЮЧАЮТСЯ ПО ОТДЕЛЬНОСТИ и дают предсказуемую ширину входа.
    w0 = s0.shape[1]
    # ЗАПРОС task БЕЗ ИДЕНТИФИКАТОРА — ОТКАЗ.
    try:
        build(a, st, tau, ttyp, rem, np.arange(n - 1), parts=("task",))
    except ValueError:
        pass
    else:
        raise AssertionError("«task» без task_id должен отвергаться")
    try:
        build(a, st, tau, ttyp, rem, np.arange(n - 1), parts=("task",),
              task_id=99, n_task=64)
    except ValueError:
        pass
    else:
        raise AssertionError("task_id вне диапазона должен отвергаться")

    for pp, add in ((("dstate",), 8), (("prevact",), 8), (("remaining",), 1),
                    (("task",), 64), (("dstate", "remaining"), 9)):
        sp, gp, yp = build(a, st, tau, ttyp, rem, np.arange(n - 1), parts=pp,
                           task_id=3, n_task=64)
        assert sp.shape[1] == w0 + add, (pp, sp.shape[1], w0 + add)
        assert np.allclose(gp, g0) and np.allclose(yp, y0)
    # Пустой набор частей эквивалентен plain.
    assert build(a, st, tau, ttyp, rem, np.arange(n - 1),
                 parts=())[0].shape[1] == w0
    # Неизвестная часть — отказ, а не молчаливое игнорирование.
    try:
        build(a, st, tau, ttyp, rem, np.arange(n - 1), parts=("нет",))
    except ValueError:
        pass
    else:
        raise AssertionError("неизвестная часть входа должна отвергаться")
    assert "remaining" in ORACLE_PARTS, "привилегированная часть помечена"

    print("самопроверка goal_dataset пройдена (версия «онлайновый вход»): "
          "перегон axis-angle с pi, инвариантность цели к сдвигу, rich "
          "расширяет вход не меняя цель и таргет, онлайновая сборка "
          "совпадает с обучающей на пяти наборах частей")


if __name__ == "__main__":
    selftest()
