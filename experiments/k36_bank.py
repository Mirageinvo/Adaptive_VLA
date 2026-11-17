#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-36: разбиение готового банка K-33 v2 и среда B1 для k31_collect.

Разбиение (§5.2): для задачи t — 300 настоящих case ID пространства 1
(k33_bank.load_task, по возрастанию idx), позиции перемешиваются
PCG64(SeedSequence([36, 20261117, t])); первые 100 — A, следующие 100 — C,
последние 100 — резерв H (v1 не открывает). B-screen — первые 30 случаев
списка A каждой адаптированной задачи 0/6/8; C для остальных семи — первые
20 их списка C. Case ID читаются из банка; 0…299 не подставляются.

Среда (§6): BankSim — один долго живущий CPU-воркер (forkserver, создаётся
до CUDA) с K33Env того же рецепта K-33; на каждом reset прежний объект
среды закрывается и создаётся новый. Режимы: bank (space 0 — dev, 1 —
confirmation; reset(task, [позиция случая в файле банка])) и legacy
(штатный init_state_id: create seed 0 -> native reset -> init file). API —
как k18_collector.Sim (reset/step/close), поэтому эпизод — неизменённый
k31_collect.run_episode: seed 101, ровно 10 шагов ожидания в управляющем
цикле, защёлка done/успеха, до 600 команд. Ожидание не дублируется:
восстановление K-33 возвращает наблюдение S0 до ожидания.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SEQ = (36, 20261117)
N_CASES = 300
N_A, N_C, N_H = 100, 100, 100
ADAPTED = (0, 6, 8)
REST = tuple(t for t in range(10) if t not in ADAPTED)
N_SCREEN, N_C_REST = 30, 20
BANK_ID = "ddf3f626f0b5"
SPACE_CONF, SPACE_DEV = 1, 0


def ids_hash(ids):
    return hashlib.sha1("\n".join(ids).encode()).hexdigest()


def split_ids(ids, task):
    """ids — case ID задачи по возрастанию idx -> dict(A, C, H)."""
    ids = list(ids)
    if len(ids) != N_CASES or len(set(ids)) != N_CASES:
        raise SystemExit(f"t{task}: случаев {len(ids)} (уникальных "
                         f"{len(set(ids))}) ≠ {N_CASES}")
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [*SEQ, int(task)])))
    perm = g.permutation(N_CASES)
    order = [ids[int(i)] for i in perm]
    return dict(A=order[:N_A], C=order[N_A:N_A + N_C],
                H=order[N_A + N_C:])


def task_split(task, bank_dir=None):
    """Разбиение задачи из банка: ID в порядке idx, позиции в файле."""
    import k33_bank as kb
    cases, _meta = kb.load_task(SPACE_CONF, task,
                                bank_dir or kb.BANK_DIR)
    by_idx = sorted(cases, key=lambda c: int(c["idx"]))
    ids = [c["case_id"] for c in by_idx]
    pos = {c["case_id"]: i for i, c in enumerate(cases)}
    hashes = {c["case_id"]: [c["init_hash"], c["init_hash_full"]]
              for c in cases}
    s = split_ids(ids, task)
    return dict(task=int(task), ids_sorted=ids, ids_sha1=ids_hash(ids),
                pos=pos, hashes=hashes, **s)


def make_splits(bank_dir=None):
    out = {}
    for t in range(10):
        sp = task_split(t, bank_dir)
        out[str(t)] = dict(
            ids_sha1=sp["ids_sha1"], A=sp["A"], C=sp["C"],
            H_sha1=ids_hash(sp["H"]), n_H=len(sp["H"]),
            screen=sp["A"][:N_SCREEN] if t in ADAPTED else [],
            c_used=sp["C"] if t in ADAPTED else sp["C"][:N_C_REST],
            pos={k: sp["pos"][k] for k in sp["A"] + sp["C"]},
            init_hash={k: sp["hashes"][k] for k in sp["A"] + sp["C"]})
    return out


def splits_problems(S):
    """Уникальность, отсутствие пересечений A/C/H, объёмы."""
    p = []
    for t in range(10):
        d = S.get(str(t))
        if d is None:
            p.append(f"t{t}: нет разбиения")
            continue
        A, C = d["A"], d["C"]
        if len(A) != N_A or len(C) != N_C or d["n_H"] != N_H:
            p.append(f"t{t}: объёмы A/C/H")
        if len(set(A)) != len(A) or len(set(C)) != len(C) or \
                set(A) & set(C):
            p.append(f"t{t}: повтор или пересечение A/C")
        want_screen = A[:N_SCREEN] if t in ADAPTED else []
        want_c = C if t in ADAPTED else C[:N_C_REST]
        if d["screen"] != want_screen or d["c_used"] != want_c:
            p.append(f"t{t}: B-screen / C-список ≠ правилу")
        if set(d["pos"]) != set(A) | set(C):
            p.append(f"t{t}: позиции не для всех случаев")
    return p


def dev_cases(task, n=2, bank_dir=None):
    """Первые n dev-сцен пространства 0 (порядок idx) -> [(case_id, pos,
    init_hash, init_hash_full)]."""
    import k33_bank as kb
    cases, _m = kb.load_task(SPACE_DEV, task, bank_dir or kb.BANK_DIR)
    order = sorted(range(len(cases)), key=lambda i: int(cases[i]["idx"]))
    return [(cases[i]["case_id"], i, cases[i]["init_hash"],
             cases[i]["init_hash_full"]) for i in order[:n]]


# --- среда ------------------------------------------------------------------------
def _worker(conn):
    env = None
    try:
        import k33_bank as kb
        K33Env = kb.make_env_class()
        while True:
            cmd, data = conn.recv()
            if cmd == "make":
                if env is not None:
                    env.close()
                    env = None
                task, mode, space, bank_dir = data
                env = K33Env(task_id=int(task), image_size=224,
                             k33_mode=mode, k33_space=int(space),
                             k33_bank_dir=bank_dir)
                conn.send(("ok", env.observation_space))
            elif cmd == "reset":
                conn.send(("ok", env.reset(init_state_id=int(data))))
            elif cmd == "step":
                obs, r, d, _info = env.step(data)
                s = bool(env.check_success())
                conn.send(("ok", (obs, float(r), bool(d), s)))
            elif cmd == "close":
                if env is not None:
                    env.close()
                conn.send(("ok", None))
                break
            else:
                raise RuntimeError(f"команда {cmd!r}")
    except Exception as e:                                   # noqa: BLE001
        import traceback
        conn.send(("err", f"{e}\n{traceback.format_exc()}"))


class BankSim:
    """API k18_collector.Sim (n = 1): reset(task, [i]) — новый объект
    среды и i = позиция случая банка (bank) либо init_state_id (legacy)."""

    def __init__(self, mode, space=SPACE_CONF, bank_dir=None):
        import multiprocessing as mp
        import k33_bank as kb
        if mode not in ("bank", "legacy"):
            raise ValueError(mode)
        self.mode, self.space = mode, int(space)
        self.bank_dir = os.path.abspath(bank_dir or kb.BANK_DIR)
        self.ctx = mp.get_context("forkserver")
        a, b = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_worker, args=(b,), daemon=True)
        self.proc.start()
        b.close()
        self.conn = a
        self.task = None

    def _call(self, cmd, data=None):
        self.conn.send((cmd, data))
        st, v = self.conn.recv()
        if st != "ok":
            raise SystemExit(f"воркер среды K-36: {v}")
        return v

    def close(self):
        try:
            self._call("close")
        except Exception:                                    # noqa: BLE001
            pass
        self.proc.join(timeout=30)
        if self.proc.is_alive():
            self.proc.terminate()

    def reset(self, task, starts):
        from gym.vector.utils import concatenate, create_empty_array
        import copy
        if len(starts) != 1:
            raise SystemExit("K-36: только B = 1")
        self.space_obs = self._call("make", (int(task), self.mode,
                                             self.space, self.bank_dir))
        self.task = int(task)
        self.buf = create_empty_array(self.space_obs, n=1, fn=np.zeros)
        obs = self._call("reset", int(starts[0]))
        self.buf = concatenate(self.space_obs, [obs], self.buf)
        return copy.deepcopy(self.buf)

    def step(self, actions):
        from gym.vector.utils import concatenate
        import copy
        obs, r, d, s = self._call("step", actions[0])
        self.buf = concatenate(self.space_obs, [obs], self.buf)
        return (copy.deepcopy(self.buf), np.array([r]), np.array([d],
                                                                 np.bool_),
                np.array([s], np.bool_))


def selftest():
    import tempfile
    # разбиение: детерминизм, объёмы, без пересечений; другой порядок
    # входа (не по idx) — другой результат, поэтому вход сортируется
    ids = [f"k33s1_t3_c{i:05d}" for i in range(0, 340) if i % 9 != 4][:300]
    a1, a2 = split_ids(ids, 3), split_ids(list(ids), 3)
    assert a1 == a2 and len(a1["A"]) == 100 and len(a1["C"]) == 100 and \
        len(a1["H"]) == 100
    assert not (set(a1["A"]) & set(a1["C"])) and not (set(a1["C"]) &
                                                       set(a1["H"]))
    assert set(a1["A"]) | set(a1["C"]) | set(a1["H"]) == set(ids)
    assert split_ids(ids, 4) != a1                    # задача в ключе
    # непоследовательные idx: ID берутся как есть (c00004 отсутствует)
    assert "k33s1_t3_c00004" not in a1["A"] + a1["C"] + a1["H"]
    for bad in (ids[:299], ids[:299] + ids[:1]):
        try:
            split_ids(bad, 3)
            raise AssertionError("неполный/повторный список принят")
        except SystemExit:
            pass
    # проверка разбиения: правильное — без проблем; подмены — отказ
    S = {}
    for t in range(10):
        ii = [f"k33s1_t{t}_c{i:05d}" for i in range(300)]
        sp = split_ids(ii, t)
        S[str(t)] = dict(ids_sha1=ids_hash(ii), A=sp["A"], C=sp["C"],
                         H_sha1=ids_hash(sp["H"]), n_H=100,
                         screen=sp["A"][:30] if t in ADAPTED else [],
                         c_used=sp["C"] if t in ADAPTED else sp["C"][:20],
                         pos={k: int(k[-5:]) for k in sp["A"] + sp["C"]},
                         init_hash={})
    assert splits_problems(S) == []
    S2 = json.loads(json.dumps(S))
    S2["6"]["C"][0] = S2["6"]["A"][0]
    assert splits_problems(S2)
    S3 = json.loads(json.dumps(S))
    S3["0"]["screen"] = S3["0"]["A"][1:31]
    assert splits_problems(S3)
    S4 = json.loads(json.dumps(S))
    S4["2"]["c_used"] = S4["2"]["C"][:21]
    assert splits_problems(S4)
    # глобальные RNG не сдвигаются
    st = np.random.get_state()[1].copy()
    split_ids(ids, 3)
    assert np.array_equal(st, np.random.get_state()[1])
    with tempfile.TemporaryDirectory():
        pass
    print("самопроверка k36_bank пройдена: разбиение A/C/H детерминировано, "
          "без пересечений и повторов, ID из списка (не 0…299), задача в "
          "ключе, неполный/повторный список — отказ, B-screen и C-список — "
          "по правилу, глобальные RNG не сдвигаются")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
