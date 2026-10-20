#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: ускоренная загрузка батчей без изменения обучения.

1. Строка кадра читается одним pread (те же байты, что через mmap).
2. Батчи следующих обновлений собираются заранее в фоновых потоках
   (расписание индексов детерминировано и известно заранее).
3. Подмена — только функций k21_data.make_batch и k21_train.schedule
   (обёртка, сообщающая, какие обновления понадобятся дальше); файлы
   контракта обучения не меняются.
4. Эквивалентность: первые CHECK_N батчей собираются и прежним путём;
   любое побитовое расхождение тензоров — отказ.
"""
import concurrent.futures as cf
import copy
import os
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

AHEAD = 3            # обновлений вперёд
WORKERS = 6
CHECK_N = 4


class RowReader:
    """Чтение строк .npy одним pread на строку."""

    def __init__(self, path):
        mm = np.load(path, mmap_mode="r")
        self.shape, self.dtype = mm.shape[1:], mm.dtype
        self.offset = mm.offset
        self.nbytes = int(np.prod(self.shape)) * mm.dtype.itemsize
        self.n = mm.shape[0]
        self.fd = os.open(path, os.O_RDONLY)
        del mm

    def __getitem__(self, i):
        i = int(i)
        if not 0 <= i < self.n:
            raise IndexError(i)
        b = os.pread(self.fd, self.nbytes, self.offset + i * self.nbytes)
        if len(b) != self.nbytes:
            raise SystemExit(f"неполное чтение строки {i}")
        return np.frombuffer(b, self.dtype).reshape(self.shape)


def cpu_part(D, rows, proc_bar, reader, teacher_idx=None):
    """Всё, что make_batch делает до переноса на GPU."""
    import k20_data as kd
    import utils as U
    rows = np.asarray(rows)
    if teacher_idx is not None and not np.array_equal(
            D["teacher_rows"][np.asarray(teacher_idx)], rows):
        raise SystemExit("строки батча ≠ строкам учителя")
    cfg, proc = proc_bar["cfg"], proc_bar["proc"]
    msgs = []
    states = kd.norm_states(np.asarray(D["state8"][rows]))
    for s, t in zip(states, D["task"][rows].tolist()):
        m = U.prompt_template(
            s, None, t, mode=cfg.MODEL.vla_processor.kwargs.mode,
            action_vocab_size=cfg.MODEL.action_processor.vocab_size,
            action_token_len=cfg.MODEL.action_processor.token_len)
        m[1]["content"] = m[1]["content"][1:]
        msgs.append(m)
    texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
    b = proc(text=texts, images=[[np.asarray(reader[i])] for i in rows],
             return_tensors="pt", padding=True, padding_side="left",
             action_processor_kwargs={"embodiment_ids": 0})
    return b


def to_device(D, b, rows, bar, dev, teacher_idx=None):
    import torch
    import utils as U
    rows = np.asarray(rows)
    batch = U.dict_apply(lambda x: x.to(bar["E"].device, torch.float16), b)
    po = torch.as_tensor(D["pos_offset"][rows], device=dev)
    st = torch.as_tensor(D["state11"][rows], device=dev)
    if teacher_idx is None:
        tgt = torch.as_tensor(D["act"][rows], device=dev)
        val = torch.as_tensor(D["valid"][rows], device=dev)
    else:
        tgt = torch.as_tensor(D["teacher_target"][np.asarray(teacher_idx)],
                              device=dev)
        val = torch.ones(tgt.shape[:2], device=dev)
    return batch, po, st, tgt, val


def same_batch(x, y):
    import torch
    bx, by = x[0], y[0]
    if set(bx.keys()) != set(by.keys()):
        return False
    for k in bx.keys():
        if not torch.equal(bx[k], by[k]):
            return False
    return all(torch.equal(a, b) for a, b in zip(x[1:], y[1:]))


class Prefetcher:
    """Предзагрузка по фактически запрошенным батчам: номер обновления
    определяется по недавним вызовам расписания, содержащим запрошенные
    строки; готовятся следующие AHEAD обновлений того же ряда; неиспользованные
    задачи отбрасываются."""

    def __init__(self, D, bar, original_make_batch, sched_fn, micro_hint=None,
                 workers=WORKERS, cpu_fn=None):
        self.D, self.bar = D, bar
        self.orig = original_make_batch
        self.sched = sched_fn
        self.cpu_fn = cpu_fn or cpu_part
        self.reader = RowReader(D["prep_report"]["inputs"]["images"]["path"]) \
            if cpu_fn is None else None
        self.pool = cf.ThreadPoolExecutor(workers)
        self.local = threading.local()
        self.futs = {}           # key -> (update, future)
        self.recent = {}         # update -> [keys]
        self.lock = threading.Lock()
        self.checked = 0
        self.hits = self.misses = 0

    def _bar(self):
        if not hasattr(self.local, "bar"):
            self.local.bar = dict(cfg=self.bar["cfg"],
                                  proc=copy.deepcopy(self.bar["proc"]))
        return self.local.bar

    @staticmethod
    def key(rows, teacher_idx):
        return (tuple(int(r) for r in np.asarray(rows)),
                None if teacher_idx is None else
                tuple(int(t) for t in np.asarray(teacher_idx)))

    def keys_of(self, src, idx):
        out = []
        for r in idx:
            if src == "teacher":
                out.append((self.key(self.D["teacher_rows"][r], r),
                            (self.D["teacher_rows"][r], r)))
            else:
                out.append((self.key(r, None), (r, None)))
        return out

    def note_schedule(self, start, src, idx):
        with self.lock:
            self.recent[start] = [k for k, _ in self.keys_of(src, idx)]
            if len(self.recent) > 16:
                for u in sorted(self.recent)[:-16]:
                    del self.recent[u]

    def _work(self, rows, teacher_idx):
        if self.reader is None:
            return self.cpu_fn(self.D, rows, None, None, teacher_idx)
        return self.cpu_fn(self.D, rows, self._bar(), self.reader, teacher_idx)

    def _ahead(self, u, micro):
        for t in range(u + 1, u + 1 + AHEAD):
            src, idx = self.sched(self.D, t + 1, micro, start=t)[0]
            for k, (rows, ti) in self.keys_of(src, idx):
                with self.lock:
                    if k in self.futs:
                        continue
                    self.futs[k] = (t, self.pool.submit(self._work, rows, ti))
        with self.lock:
            for k in [k for k, (t, _f) in self.futs.items() if t <= u - 1]:
                self.futs.pop(k)[1].cancel()

    def make_batch(self, D, rows, bar, dev, teacher_idx=None):
        k = self.key(rows, teacher_idx)
        with self.lock:
            hit = self.futs.pop(k, None)
            u = hit[0] if hit else next((t for t, ks in self.recent.items()
                                         if k in ks), None)
        micro = len(np.asarray(rows))
        if hit is None:
            self.misses += 1
            b = self._work(rows, teacher_idx)
        else:
            self.hits += 1
            b = hit[1].result()
        if u is not None:
            self._ahead(u, micro)
        out = to_device(D, b, rows, bar, dev, teacher_idx)
        if self.checked < CHECK_N and self.orig is not None:
            ref = self.orig(D, rows, bar, dev, teacher_idx=teacher_idx)
            if not same_batch(out, ref):
                raise SystemExit("ускоренная сборка батча ≠ прежней")
            self.checked += 1
            print(f"  fastload: батч {self.checked}/{CHECK_N} совпал с "
                  f"прежним путём побитово", flush=True)
        return out


def install(_unused=None):
    """Подмена make_batch и обёртка schedule (вызывается до k21_train.main)."""
    import k21_data as d21
    import k21_train as kt
    orig_make = d21.make_batch
    orig_sched = kt.schedule
    state = dict(pf=None)

    def make_batch(D, rows, bar, dev, teacher_idx=None):
        if state["pf"] is None:
            state["pf"] = Prefetcher(D, bar, orig_make, orig_sched)
            for u0, (src, idx) in state.get("pending", {}).items():
                state["pf"].note_schedule(u0, src, idx)
        return state["pf"].make_batch(D, rows, bar, dev, teacher_idx)

    def schedule(D, n_updates, micro, seed=kt.SEED, start=0):
        out = orig_sched(D, n_updates, micro, seed=seed, start=start)
        if n_updates == start + 1:
            if state["pf"] is not None:
                state["pf"].note_schedule(start, *out[0])
            else:
                state.setdefault("pending", {})[start] = out[0]
        return out
    d21.make_batch = make_batch
    kt.schedule = schedule
    return state


def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        X = rng.integers(0, 255, (50, 3, 8, 16), dtype=np.uint8)
        f = td + "/x.npy"
        np.save(f, X)
        r = RowReader(f)
        mm = np.load(f, mmap_mode="r")
        for i in (0, 7, 49):
            assert np.array_equal(r[i], mm[i])
        try:
            r[50]
            raise AssertionError
        except IndexError:
            pass
    # предзагрузка: следующие обновления готовятся, лишние отбрасываются
    D = dict(teacher_rows=np.arange(100, 140))
    calls = []

    def sched(D_, n, micro, start=0):
        g = np.random.default_rng(start)
        if start % 4 == 3:
            return [("teacher", g.integers(0, 40, (1, micro)))]
        return [("expert", g.integers(0, 99, (1, micro)))]

    def cpu(D_, rows, b, r, ti):
        calls.append(tuple(np.asarray(rows).tolist()))
        return dict(rows=np.asarray(rows))
    pf = Prefetcher(D, None, None, sched, cpu_fn=cpu, workers=2)
    import k21_fastload as me
    me_to = me.to_device
    me.to_device = lambda D_, b, rows, bar, dev, ti=None: b
    globals()["to_device"] = me.to_device
    try:
        for u in range(0, 10):
            src, idx = sched(D, u + 1, 4, start=u)[0]
            pf.note_schedule(u, src, idx)
            for rr in idx:
                if src == "teacher":
                    o = pf.make_batch(D, D["teacher_rows"][rr], None, None, rr)
                else:
                    o = pf.make_batch(D, rr, None, None)
                want = D["teacher_rows"][rr] if src == "teacher" else rr
                assert np.array_equal(o["rows"], want)
        assert pf.hits >= 8 and pf.misses <= 2, (pf.hits, pf.misses)
        assert len(pf.futs) <= AHEAD
    finally:
        me.to_device = me_to
        globals()["to_device"] = me_to
    print("самопроверка k21_fastload пройдена: pread-строка = mmap-строке, "
          "предзагрузка следующих обновлений (попадания, без накопления)")
    return 0


if __name__ == "__main__":
    sys.exit(selftest())
