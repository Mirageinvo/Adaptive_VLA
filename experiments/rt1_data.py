#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, этап 1: строки обучения/проверки -> батч модели.

Строка индекса — (задача, эпизод, шаг t). Вход: три кадра шага t
(decode_image_bit, RGB) -> склейка 224×672; состояние state[t] (14 каналов)
нормировано по q01/q99 обучающих эпизодов; инструкция — одна из
инструкций эпизода, выбор детерминирован по (эпизод, шаг). Цель —
action[t : t+20] в исходных единицах (рад, схваты [0, 1]); за концом
эпизода — последнее действие с маской 0. Промпт и обработка — как у
BAR-LIBERO (rt1_common.prompt_batch).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rt1_common as rc  # noqa: E402
import rt1_prepare as rp  # noqa: E402


def verify_prep(prep, index, files=True, log=print):
    """Контракт подготовки: prep_id = отпечатку содержимого, index.npz и
    SHA реальных HDF5 (один раз за процесс). Проблемы -> SystemExit."""
    if prep.get("problems"):
        raise SystemExit("подготовка с проблемами: "
                         + "; ".join(prep["problems"][:3]))
    if rp.contract_id(prep) != prep.get("prep_id"):
        raise SystemExit("prep.json изменён после подготовки (prep_id ≠ "
                         "содержимому)")
    if rc.sha(index) != prep["index_sha1"]:
        raise SystemExit("index.npz ≠ подготовке")
    if files:
        t0 = __import__("time").time()
        for i, f in enumerate(prep["files"]):
            p = os.path.join(prep["data_root"], f["path"])
            if not os.path.exists(p) or rc.sha(p) != f["sha1"]:
                raise SystemExit(f"файл данных изменён или отсутствует: {p}")
        log(f"  данные сверены: {len(prep['files'])} файлов, "
            f"{__import__('time').time() - t0:.0f} с", flush=True)


class Data:
    def __init__(self, prep=rp.PREP, index=rp.INDEX, check=True,
                 check_files=True):
        self.prep = json.load(open(prep, encoding="utf-8"))
        if check:
            verify_prep(self.prep, index, files=check_files)
        z = np.load(index)
        self.ix = {k: z[k] for k in z.files}
        st = self.prep["stats_train"]["state"]
        self.q01, self.q99 = np.asarray(st["q01"]), np.asarray(st["q99"])
        r = self.prep["robot"]
        self.L, self.U = np.asarray(r["L"]), np.asarray(r["U"])
        self.files = self.prep["files"]
        self.root = self.prep["data_root"]
        self.decode = rc.decoder()
        self._h = {}
        self._ch = {}          # fid -> (state[T,14], action[T,14], инструкции)

    def rows(self, split):
        return np.where(self.ix["split"] == (1 if split == "val" else 0))[0]

    def _file(self, fid):
        if fid not in self._h:
            import h5py
            if len(self._h) > 16:
                for k in list(self._h)[:8]:
                    self._h.pop(k).close()
            self._h[fid] = h5py.File(os.path.join(
                self.root, self.files[fid]["path"]), "r")
        return self._h[fid]

    def sample(self, row):
        fid, t = int(self.ix["ep"][row]), int(self.ix["t"][row])
        h = self._file(fid)
        if fid not in self._ch:          # малые массивы — в памяти
            self._ch[fid] = (rc.channels(h, "state"), rc.channels(h, "action"),
                             rp.instructions(h) or [self.files[fid]["task"]])
        st, act, ins = self._ch[fid]
        frames = [self.decode(h[f"vision/{c}/colors"][t]) for c in rc.CAMS]
        img = rc.tile(frames)
        s_n = rc.normalize(st[t], self.q01, self.q99)
        tgt, valid = rc.chunk_target(act, t)
        text = ins[(fid * 100003 + t) % len(ins)]
        return img, s_n, text, tgt, valid

    def make_batch(self, rows, bar, device, dtype):
        import torch
        S = [self.sample(r) for r in rows]
        batch = rc.prompt_batch(bar, [s[0] for s in S], [s[1] for s in S],
                                [s[2] for s in S], device, dtype)
        B = len(S)
        po = torch.full((B,), rc.POS_OFFSET, device=device, dtype=torch.long)
        st = torch.as_tensor(np.stack([s[1] for s in S]), device=device,
                             dtype=torch.float32)
        tgt = torch.as_tensor(np.stack([s[3] for s in S]), device=device)
        val = torch.as_tensor(np.stack([s[4] for s in S]), device=device)
        return batch, po, st, tgt, val
