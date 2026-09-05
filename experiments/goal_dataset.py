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
          rich=False, n_task=64):
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
    if not rich:
        return sn.astype(np.float32), g.astype(np.float32), a[t].astype(np.float32)
    prev = np.maximum(t - 1, 0)
    dst = st[t] - st[prev]
    pact = a[prev]
    # ОСТАТОК ДАЁТСЯ ЯВНО: без него контроллер не знает, сколько у него шагов,
    # и не может распределить движение — величина команды в демонстрации от
    # этого зависит напрямую.
    rr = rem[t][:, None].astype(np.float64)
    extra = [dst, pact, np.log1p(rr)]
    if task_id is not None:
        oh_t = np.zeros((len(t), n_task))
        oh_t[np.arange(len(t)), int(task_id) % n_task] = 1.0
        extra.append(oh_t)
    sn = np.concatenate([sn] + extra, 1)
    return sn.astype(np.float32), g.astype(np.float32), a[t].astype(np.float32)


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
    assert s1.shape[1] == s0.shape[1] + 8 + 7 + 1 + 64, s1.shape
    assert np.allclose(s1[:, :s0.shape[1]], s0)
    # Идентификатор задачи попал в свою позицию и только в неё.
    oh = s1[:, -64:]
    assert oh.sum() == len(s1) and oh[:, 3].all()
    print("самопроверка goal_dataset пройдена: перегон axis-angle с pi, "
          "инвариантность цели к сдвигу, rich расширяет вход не меняя цель "
          "и таргет")


if __name__ == "__main__":
    selftest()
