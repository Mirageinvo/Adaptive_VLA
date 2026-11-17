#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-36: prepare (неизменяем, идемпотентен), очередь этапа A, бюджеты.

prepare — reports/k36/prepare.json: таблица размещения входов (SHA1 и
  SHA256), аудит K-32/K-25 (k36_inputs.k32_audit), аудит банка, разбиение
  A/C/H по задачам, бюджеты (§11), статистика (§7.2, §9.3), очередь A,
  конфигурация исполнения, SHA кода K-36 и зависимостей K-32, версии
  библиотек. Сравнение — в JSON-форме, без поля created; любое отличие —
  отказ до записи. Конфликт входов — отказ до записи и до резерва.
Очередь A (§7): задачи 0/6/8 в перестановке seed 36020; для каждой задачи
  20 блоков по 5 закреплённых A-случаев; метки [R_s0, T32_s0, P32_s0,
  R_s1, T32_s1, P32_s1] циклически сдвигаются на номер блока: 360 блоков,
  1800 эпизодов.
Журнал data/k36/ledger.json (k31_collect.Ledger): все scopes плана с
  неизменяемыми лимитами; до решения A исполняются только checks и
  confirm_a (проверка в k36_harness).
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k36_bank as kb36  # noqa: E402
import k36_inputs as kin  # noqa: E402

OUT = "reports/k36"
PREPARE = "reports/k36/prepare.json"
LEDGER = "data/k36/ledger.json"
EP = 610
CAPS = dict(checks=61_000, confirm_a=1_128_500, train_tplus_s0=200_000,
            train_tplus_s1=200_000, val_tplus_s0=27_450, val_tplus_s1=27_450,
            screen_b=122_000, confirm_c=1_640_900, bar_c=280_600)
TOTAL_CAP = 3_687_900
OPEN_BEFORE_A = ("checks", "confirm_a")
ARMS_A = ("R_s0", "T32_s0", "P32_s0", "R_s1", "T32_s1", "P32_s1")
KIND = dict(R="R", T32="T", P32="P", Tplus="T")
TASK_SEED_A = 36020
BLOCK = 5
STATS = dict(alpha_A=0.025, alpha_C=0.025, level_decision=0.975,
             level_report=0.95, boot_n=100_000, boot_seed_A=36011,
             boot_seed_C=36012, family_A=["T32-P32", "T32-R"],
             family_C=["Tplus-T32", "Tplus-R", "Tplus-BAR"])
RULE_A = dict(joint_T32_P32=0.03, joint_T32_R=0.03, per_seed_floor=-0.02)
RULE_B = dict(joint_Tplus_T32=0.03, per_seed_lag_T32=2, per_seed_lag_R=2)
CODE = ("k36_bank.py", "k36_inputs.py", "k36_protocol.py", "k36_harness.py",
        "k36_analyze.py", "k36_run.sh")


def sha(p):
    return kin.sha1(p)


def jnorm(d):
    return json.loads(json.dumps(d, ensure_ascii=False, default=str))


def save_json(path, d):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(path + ".tmp", path)


def load_json(p):
    return json.load(open(p, encoding="utf-8"))


def ledger():
    import k31_collect as kcol
    return kcol.Ledger(LEDGER, CAPS)


def libs():
    out = {}
    for m in ("torch", "numpy", "gym", "robosuite", "mujoco"):
        try:
            out[m] = str(__import__(m).__version__)
        except Exception as e:                               # noqa: BLE001
            out[m] = f"нет ({type(e).__name__})"
    return out


def code_shas():
    import k32_protocol as pc32
    out = {f: sha(os.path.join(HERE, f)) for f in CODE
           if os.path.exists(os.path.join(HERE, f))}
    deps = tuple(pc32.MODEL_CODE) + ("k33_bank.py", "k32_harness.py")
    out.update({f: sha(os.path.join(HERE, f)) for f in deps})
    return out


# --- очередь A -------------------------------------------------------------------
def task_order_a():
    g = np.random.default_rng(TASK_SEED_A)
    return [int(kb36.ADAPTED[i]) for i in g.permutation(len(kb36.ADAPTED))]


def cells_a(splits):
    """[(метка, задача, блок, [5 case ID])] в порядке исполнения."""
    out, k = [], 0
    for t in task_order_a():
        A = splits[str(t)]["A"]
        for g in range(len(A) // BLOCK):
            rot = list(ARMS_A[k % len(ARMS_A):]) + list(
                ARMS_A[:k % len(ARMS_A)])
            for lab in rot:
                out.append((lab, t, g, list(A[g * BLOCK:(g + 1) * BLOCK])))
            k += 1
    return out


def parse_seeds():
    return (0, 1)


def parse_arm(lab):
    """'T32_s1' -> ('T32', 'T', 1)."""
    name, s = lab.rsplit("_s", 1)
    return name, KIND[name], int(s)


# --- prepare ----------------------------------------------------------------------
def make_prepare():
    import torch
    import k32_protocol as pc32
    P = kin.logical_paths()
    pl = kin.placement(P)
    info, p = kin.k32_audit(torch, P)
    binfo, bp = kin.bank_audit(P)
    p += bp
    if p:
        raise SystemExit("входы K-36: " + "; ".join(p[:6]))
    splits = kb36.make_splits()
    sp = kb36.splits_problems(splits)
    if sp:
        raise SystemExit("разбиение банка: " + "; ".join(sp[:4]))
    b = pc32.load_base()
    ca = cells_a(splits)
    return jnorm(dict(
        kind="k36_prepare", placement=pl, k32=info, bank=binfo,
        splits=splits, caps=CAPS, total_cap=TOTAL_CAP,
        open_before_A=list(OPEN_BEFORE_A), stats=STATS, rule_A=RULE_A,
        rule_B=RULE_B, arms_A=list(ARMS_A), task_order_A=task_order_a(),
        cells_A_sha1=hashlib.sha1(json.dumps(ca).encode()).hexdigest(),
        n_cells_A=len(ca),
        k32_plan=dict(tasks=b["tasks"], init_ref=b["init_ref"],
                      bounds=b["bounds"], scale_q=b["scale_q"],
                      operator=b["operator"], dist=b["dist"], ppo=b["ppo"],
                      env=b["env"], banks=b["banks"],
                      bar_revision=b["bar_revision"], prep_id=b["prep_id"]),
        env=dict(seed=101, waiting_steps=10, max_steps=600, horizon=8, B=1,
                 fresh_env_per_episode=True, sequential=True),
        threads=dict(torch_intra=4, torch_inter=1, omp=4, mkl=4),
        libraries=libs(), code=code_shas()))


def prepare(a):
    new = make_prepare()
    if os.path.exists(PREPARE):
        old = load_json(PREPARE)
        diff = sorted(k for k in set(old) | set(new)
                      if k != "created" and old.get(k) != new.get(k))
        if diff:
            raise SystemExit(f"{PREPARE} закреплён иначе: {diff}")
        print(f"  {PREPARE}: закреплён, входы и разбиение совпадают")
    else:
        save_json(PREPARE, dict(new, created=datetime.datetime.now()
                                .isoformat(timespec="seconds")))
        print(f"  {PREPARE}: записан")
    for s, v in new["k32"]["seeds"].items():
        ts = v.get("T_state", {})
        print(f"    s{s}: T32 {v['T']['actor_hash'][:12]}, P32 "
              f"{v['P']['actor_hash'][:12]}; state: волн {ts.get('wave')}, "
              f"обновлений {ts.get('update')} (принято {ts.get('accepted')})"
              f", расход K-32 {ts.get('spent_imported')}")
    print(f"    банк {new['bank']['bank_id'][:12]}; очередь A: "
          f"{new['n_cells_A']} блоков, задачи {new['task_order_A']}")
    return 0


def prepare_problems():
    """prepare есть, входы не изменены, код K-36 и K-32 тот же."""
    if not os.path.exists(PREPARE):
        return ["нет prepare: k36_protocol.py prepare"]
    d = load_json(PREPARE)
    p = kin.placement_problems(d["placement"])
    cur = code_shas()
    bad = [f for f, s in d["code"].items() if cur.get(f) != s]
    if bad:
        p.append(f"код изменён после prepare: {bad[:4]}")
    return p


def load_prepare():
    p = prepare_problems()
    if p:
        raise SystemExit("prepare: " + "; ".join(p[:4]))
    d = load_json(PREPARE)
    d["sha1"] = sha(PREPARE)
    return d


def selftest():
    # очередь A: объём, сдвиг меток, все клетки различны, случаи без повторов
    S = {}
    for t in range(10):
        ids = [f"k33s1_t{t}_c{i:05d}" for i in range(300)]
        sp = kb36.split_ids(ids, t)
        S[str(t)] = dict(A=sp["A"], C=sp["C"])
    c = cells_a(S)
    assert len(c) == 360 and len({(x[0], x[1], x[2]) for x in c}) == 360
    assert sum(len(x[3]) for x in c) == 1800
    first = [x[0] for x in c[:6]]
    second = [x[0] for x in c[6:12]]
    assert first == list(ARMS_A) and second == first[1:] + first[:1]
    for lab in ARMS_A:
        for t in kb36.ADAPTED:
            got = [i for x in c if x[0] == lab and x[1] == t for i in x[3]]
            assert sorted(got) == sorted(S[str(t)]["A"]) and \
                len(set(got)) == 100
    assert sorted(task_order_a()) == [0, 6, 8]
    assert parse_arm("T32_s1") == ("T32", "T", 1) and \
        parse_arm("R_s0") == ("R", "R", 0)
    # бюджеты: сумма = полному пределу; эпизодные caps
    assert sum(CAPS.values()) == TOTAL_CAP
    assert CAPS["checks"] == 100 * EP and \
        CAPS["confirm_a"] == (1800 + 50) * EP and \
        CAPS["screen_b"] == 200 * EP and \
        CAPS["confirm_c"] == (2640 + 50) * EP and \
        CAPS["bar_c"] == (440 + 20) * EP and \
        CAPS["val_tplus_s0"] == 45 * EP
    assert CAPS["checks"] + CAPS["confirm_a"] == 1_189_500
    # JSON-нормализация: кортежи не дают ложной разницы
    assert jnorm(dict(a=(1, 2))) == json.loads(json.dumps(dict(a=[1, 2])))
    print("самопроверка k36_protocol пройдена: очередь A 360 блоков / 1800 "
          "эпизодов, сдвиг меток по блокам, все A-случаи каждой метки ровно "
          "по разу; бюджеты = плану (сумма 3 687 900, до A 1 189 500); "
          "JSON-нормализация")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-36: протокол")
    ap.add_argument("cmd", choices=("prepare", "verify"))
    a = ap.parse_args()
    try:
        if a.cmd == "verify":
            p = prepare_problems()
            print("  prepare: " + ("свеж" if not p else "; ".join(p[:4])))
            return 0 if not p else 3
        return prepare(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
