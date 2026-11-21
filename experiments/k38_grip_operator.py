#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-38: сдвиг первого подходящего закрытия схвата на два шага.

Знаки: физическая команда схвата g = −a_codec[:, 6] (k29_policy.scale_exec
инвертирует канал кодека); −1 — открыть, +1 — закрыть. Предыдущая
физическая команда в начале эпизода — k31_collect.DUMMY[6] (−1), далее —
последний ИСПОЛНЕННЫЙ шаг предыдущего чанка.

Событие (find_event): минимальный tau в чанке с
  tau ≥ 2, tau + 2 < h, h = min(8, 600 − elapsed);
  g[tau−2:tau] = открыть ×2; g[tau:tau+3] = закрыть ×3;
  команда перед tau−2 (prev при tau = 2, иначе g[tau−3]) — открыть.
Подстановка (один раз за эпизод, на первом вызове с событием):
  early: g[tau−2:tau] -> закрыть (первая изменённая команда tau−2);
  late:  g[tau:tau+2] -> открыть (первая изменённая команда tau).
Рука (каналы 0…5) — без изменений; канал 6 = −g; zero — базовый чанк
побитово (событие только записывается).
"""
import sys

import numpy as np

OPEN, CLOSE = -1.0, 1.0
N8 = 8
MAX_STEPS = 600
KINDS = ("zero", "early", "late")
PREV0 = -1.0                         # k31_collect.DUMMY[6]


def phys_grip(a_codec):
    return -np.asarray(a_codec, np.float64)[:, 6]


def horizon(elapsed, max_steps=MAX_STEPS):
    return int(min(N8, max_steps - int(elapsed)))


def find_event(g, prev, h):
    """g [8] физические команды, prev — предыдущая исполненная, h — длина
    доступного префикса -> tau | None."""
    g = np.asarray(g, np.float64)
    for tau in range(2, N8):
        if not tau + 2 < h:
            break
        if not ((g[tau - 2:tau] == OPEN).all() and
                (g[tau:tau + 3] == CLOSE).all()):
            continue
        before = prev if tau == 2 else g[tau - 3]
        if before == OPEN:
            return tau
    return None


def first_changed(kind, tau):
    return tau - 2 if kind == "early" else tau


def substitute(a_codec, tau, kind):
    """-> копия чанка [8, 7] float32 с одним сдвигом закрытия."""
    a = np.array(a_codec, np.float32, copy=True)
    g = phys_grip(a)
    if kind == "early":
        g[tau - 2:tau] = CLOSE
    elif kind == "late":
        g[tau:tau + 2] = OPEN
    else:
        raise ValueError(kind)
    a[:, 6] = (-g).astype(np.float32)
    return a


class GripArm:
    """Обёртка над k32_policy.Arm: begin / decide / executed. Один вызов
    base.decide на decide; не более одного вмешательства за эпизод."""

    def __init__(self, base, kind):
        if kind not in KINDS:
            raise ValueError(kind)
        self.base, self.kind = base, kind
        self._reset()

    def _reset(self):
        self.prev = PREV0
        self.used = False
        self.last_a = None
        self.event = None             # dict(call, elapsed, tau, first, …)

    # счётчики коллектора — в базу
    @property
    def episode(self):
        return self.base.episode

    @episode.setter
    def episode(self, v):
        self.base.episode = v

    @property
    def call(self):
        return self.base.call

    @call.setter
    def call(self, v):
        self.base.call = v

    @property
    def keep_first(self):
        return getattr(self.base, "keep_first", False)

    @keep_first.setter
    def keep_first(self, v):
        self.base.keep_first = v

    def begin(self, task, init, episode=None):
        self.base.begin(task, init, episode)
        self._reset()

    def decide(self, batch, po, state8, elapsed):
        a0, rec = self.base.decide(batch, po, state8, elapsed)
        a0 = np.asarray(a0, np.float32)
        out = a0
        ext = dict(base_a_codec=a0.copy(), k38_kind=self.kind, k38_tau=-1,
                   k38_issued=False, k38_first=-1)
        if not self.used:
            tau = find_event(phys_grip(a0), self.prev, horizon(elapsed))
            if tau is not None:
                self.used = True
                self.event = dict(call=int(getattr(self.base, "call", -1)),
                                  elapsed=int(elapsed), tau=int(tau),
                                  first=first_changed(self.kind, tau)
                                  if self.kind != "zero" else -1,
                                  issued=self.kind != "zero", reached=None)
                ext.update(k38_tau=int(tau))
                if self.kind != "zero":
                    out = substitute(a0, tau, self.kind)
                    ext.update(k38_issued=True,
                               k38_first=self.event["first"])
        if isinstance(rec, dict):
            rec.update(ext)
            rec["a_codec"] = out
            if getattr(self.base, "last", None) is not None:
                self.base.last = dict(rec=rec)
        self.last_a = out
        self._pending = self.event is not None and \
            self.event.get("reached") is None and ext["k38_tau"] >= 0
        return out, rec

    def executed(self, k):
        self.base.executed(k)
        k = int(k)
        if k > 0:
            self.prev = float(phys_grip(self.last_a)[k - 1])
        if getattr(self, "_pending", False):
            ev = self.event
            ev["k"] = k
            ev["reached"] = bool(ev["issued"] and ev["first"] < k)
            self._pending = False


def selftest():
    O, C = OPEN, CLOSE

    def chunk(g):                    # физические -> кодек
        a = np.zeros((8, 7), np.float32)
        a[:, :6] = np.arange(48, dtype=np.float32).reshape(8, 6) * 0.01
        a[:, 6] = -np.asarray(g, np.float32)
        return a
    # знаки: кодек +1 -> физическое открытие −1 (scale_exec)
    import k29_policy as kp
    a = chunk([O] * 8)
    assert (kp.scale_exec(a[None], np.ones(7))[0][:, 6] == O).all()
    assert (phys_grip(a) == O).all()
    # пример плана, tau = 4
    g = [O, O, O, O, C, C, C, C]
    assert find_event(g, O, 8) == 4
    e = phys_grip(substitute(chunk(g), 4, "early"))
    lt = phys_grip(substitute(chunk(g), 4, "late"))
    assert list(e) == [O, O, C, C, C, C, C, C]
    assert list(lt) == [O, O, O, O, O, O, C, C]
    # рука не меняется, меняется ровно две команды
    for kind in ("early", "late"):
        s = substitute(chunk(g), 4, kind)
        assert np.array_equal(s[:, :6], chunk(g)[:, :6])
        assert int((s[:, 6] != chunk(g)[:, 6]).sum()) == 2
    # границы tau = 2 и tau = 5
    assert find_event([O, O, C, C, C, O, O, O], O, 8) == 2
    assert find_event([O, O, C, C, C, O, O, O], C, 8) is None   # prev закрыт
    assert find_event([O, O, O, O, O, C, C, C], O, 8) == 5
    # короткий префикс по лимиту: tau = 5 требует h ≥ 8
    assert find_event([O, O, O, O, O, C, C, C], O, 7) is None
    assert horizon(596) == 4 and horizon(0) == 8
    # нет закрытия / короткая серия закрытия / только два открытия до
    assert find_event([O] * 8, O, 8) is None
    assert find_event([O, O, O, C, C, O, O, O], O, 8) is None
    assert find_event([C, O, O, C, C, C, C, C], O, 8) is None   # g[0] закрыт
    assert find_event([O, C, O, O, C, C, C, O], O, 8) is None   # перед — C
    # обёртка на синтетической базе
    class Base:
        def __init__(self, chunks):
            self.chunks, self.i, self.episode, self.call = chunks, 0, 0, 0
            self.last, self.ks = None, []

        def begin(self, task, init, episode=None):
            self.i = 0

        def decide(self, batch, po, s8, elapsed):
            c = self.chunks[self.i]
            self.i += 1
            self.call += 1
            rec = dict(a_codec=c.copy())
            self.last = dict(rec=rec)
            return c.copy(), rec

        def executed(self, k):
            self.ks.append(k)
    seq = [chunk([O] * 8), chunk(g), chunk(g), chunk([C] * 8)]
    outs = {}
    for kind in KINDS:
        arm = GripArm(Base(seq), kind)
        arm.begin(0, 0)
        res = []
        for j in range(4):
            a_, r_ = arm.decide(None, 0, None, 8 * j)
            arm.executed(8)
            res.append(a_)
        outs[kind] = (res, dict(arm.event))
    for j in range(4):                               # нулевой = T1 побитово
        assert np.array_equal(outs["zero"][0][j], seq[j])
    ev = outs["early"][1]
    assert ev["tau"] == 4 and ev["elapsed"] == 8 and ev["reached"]
    # второй подходящий вызов не трогается
    assert np.array_equal(outs["early"][0][2], seq[2])
    assert np.array_equal(outs["late"][0][2], seq[2])
    # новый эпизод сбрасывает состояние
    arm = GripArm(Base(seq), "early")
    arm.begin(0, 0)
    arm.decide(None, 0, None, 0)
    arm.executed(8)
    arm.decide(None, 0, None, 8)
    arm.executed(8)
    assert arm.used
    arm.base.i = 1
    arm.begin(0, 1)
    assert not arm.used and arm.prev == PREV0 and arm.event is None
    # ранний terminal: команда выдана, но не исполнена
    for kind, k, want in (("early", 2, False), ("early", 3, True),
                          ("late", 4, False), ("late", 5, True)):
        arm = GripArm(Base([chunk(g)]), kind)
        arm.begin(0, 0)
        arm.decide(None, 0, None, 0)
        arm.executed(k)
        assert arm.event["issued"] and arm.event["reached"] is want, \
            (kind, k)
    # prev = последняя исполненная (с учётом подстановки) команда
    arm = GripArm(Base([chunk([O, O, O, O, O, O, O, C]), chunk(g)]), "early")
    arm.begin(0, 0)
    arm.decide(None, 0, None, 0)
    arm.executed(8)
    assert arm.prev == C
    arm.decide(None, 0, None, 8)                     # tau = 4: перед — O ок
    assert arm.event is not None and arm.event["tau"] == 4
    print("самопроверка k38_grip_operator пройдена: знаки кодек/физика, "
          "пример tau = 4, границы tau = 2/5, лимит 600, нет закрытия, "
          "короткая серия, закрытая предыдущая; нулевой сдвиг = T1 "
          "побитово; одно вмешательство на эпизод; сброс в begin; ранний "
          "terminal — «выдано» ≠ «исполнено»")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; запуск — через k38_grip_check.py")
