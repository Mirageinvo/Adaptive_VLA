#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-13h: побитовая регрессия позиционного пути после врезки шва в K-12d.

ЗАЧЕМ. В `k12d_rollout` добавлен шов по виду головы. Позиционная ветвь при
этом обязана остаться прежней НЕ ПРИБЛИЗИТЕЛЬНО, а побитово. Совпадения
эпизодических исходов для этого недостаточно: пять эпизодов легко дают те же
бинарные результаты при разных действиях — пороги успеха в LIBERO грубые, а
траектория до порога может отличаться.

ЧТО СВЕРЯЕТСЯ. Весь буфер (`h`, `q0`, `u`, `mu`, `logp`, `task`, `state`,
`call`) тензор к тензору, полный список `eps_sha1_by_call` вместе с длиной,
`eps_sha1_all`, `policy_sha1`, `init_hash_full` каждого эпизода, число вызовов
политики и шагов среды, исходы.

ДОПУСКА НЕТ. Позиционный путь не менялся, поэтому любое расхождение — это
изменение поведения, а не численный шум. Допуск здесь пришлось бы обосновывать
отдельно, и обосновать его нечем: те же веса, тот же поток шума, тот же
порядок операций.

ЧЕГО ЭТА ПРОВЕРКА НЕ ДАЁТ. Она сверяет ОДНУ ячейку. Пути, которые эта ячейка не
исполняет (продолжение от обученной головы, этап final, другая сюита), остаются
непроверенными, и это сказано прямо, а не подразумевается.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

# Поля буфера. Дублировать список нельзя — он берётся из K-12d, чтобы при
# добавлении поля регрессия не продолжила молча сверять старый набор.
def buffer_fields():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import k12d_rollout as k12d
    return tuple(k12d.Store.FIELDS)


def tensor_sha(t):
    import torch
    a = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def compare_cells(a, b, fields, log=print):
    """Побитовая сверка двух ячеек. Возвращает список расхождений."""
    import torch
    bad = []
    da, db = a.get("data") or {}, b.get("data") or {}
    if set(da) != set(db):
        bad.append(f"разный состав буфера: только слева {sorted(set(da) - set(db))}, "
                   f"только справа {sorted(set(db) - set(da))}")
    missing = [f for f in fields if f not in da or f not in db]
    if missing:
        bad.append(f"в буфере нет полей {missing}: сверять нечем")
    for f in fields:
        if f in da and f in db:
            ta, tb = da[f], db[f]
            if tuple(ta.shape) != tuple(tb.shape):
                bad.append(f"{f}: форма {tuple(ta.shape)} против "
                           f"{tuple(tb.shape)}")
            elif not torch.equal(ta, tb):
                d = float((ta.double() - tb.double()).abs().max())
                bad.append(f"{f}: не совпал побитово, max|Δ| = {d:.3e}")
            else:
                log(f"    {f}: совпало побитово {tuple(ta.shape)} "
                    f"sha {tensor_sha(ta)}")
    ma, mb = a.get("meta") or {}, b.get("meta") or {}
    for k in ("policy_sha1", "head_sha1", "eps_sha1_all", "eps_sha1_first",
              "eps_mode", "eps_salt", "sigma", "stage", "arm", "suite",
              "task_ids", "init_start", "n_envs", "max_steps", "horizon",
              "rollout_seed", "rollout_seed_mode", "joint_sha1",
              "codebooks_sha1", "pos_offset", "d_hidden", "rank"):
        if str(ma.get(k)) != str(mb.get(k)):
            bad.append(f"мета {k}: {ma.get(k)} против {mb.get(k)}")
    # ПОВЫЗОВНЫЕ ХЭШИ — СПИСКОМ И ПО ДЛИНЕ. Совпадение только первого прошло бы
    # и при расхождении траекторий со второго вызова.
    ea = list(ma.get("eps_sha1_by_call") or [])
    eb = list(mb.get("eps_sha1_by_call") or [])
    if len(ea) != len(eb):
        bad.append(f"вызовов политики {len(ea)} против {len(eb)}")
    elif ea != eb:
        i = next(i for i in range(len(ea)) if ea[i] != eb[i])
        bad.append(f"поток шума разошёлся с вызова {i}: {ea[i]} против {eb[i]}")
    elif not ea:
        bad.append("eps_sha1_by_call пуст: поток шума не записан, и сверять "
                   "его нечем")
    else:
        log(f"    поток шума: {len(ea)} вызовов, все хэши совпали")
    # ЭПИЗОДЫ: исход, начальное состояние, шаги
    pa = ma.get("episodes") or []
    pb = mb.get("episodes") or []
    if len(pa) != len(pb):
        bad.append(f"эпизодов {len(pa)} против {len(pb)}")
    else:
        for i, (x, y) in enumerate(zip(pa, pb)):
            for k in ("state_id", "task_id", "success", "init_hash_full",
                      "env_steps", "policy_calls"):
                if str(x.get(k)) != str(y.get(k)):
                    bad.append(f"эпизод {i}, {k}: {x.get(k)} против {y.get(k)}")
    if not bad:
        log(f"    эпизоды: {len(pa)}, исходы и начальные состояния совпали")
    return bad


def selftest():
    import torch

    def cell(seed=0, n=4):
        g = torch.Generator().manual_seed(seed)
        data = dict(h=torch.randn(n, 3, generator=g),
                    q0=torch.randint(0, 5, (n, 2), generator=g),
                    u=torch.randn(n, 6, generator=g),
                    mu=torch.randn(n, 6, generator=g),
                    logp=torch.randn(n, generator=g),
                    task=torch.zeros(n, dtype=torch.int16),
                    state=torch.zeros(n, dtype=torch.int16),
                    call=torch.arange(n, dtype=torch.int16))
        meta = dict(policy_sha1="p", head_sha1="h", eps_sha1_all="e",
                    eps_sha1_first="f", eps_mode="train", eps_salt="s",
                    sigma=0.1, stage="diag", arm="policy", suite="10",
                    task_ids=[3], init_start=0, n_envs=5, max_steps=600,
                    horizon=8, rollout_seed=1, rollout_seed_mode="m",
                    joint_sha1="j", codebooks_sha1="c", pos_offset=3,
                    d_hidden=768, rank=32,
                    eps_sha1_by_call=["a", "b", "c"],
                    episodes=[dict(state_id=i, task_id=3, success=True,
                                   init_hash_full=f"h{i}", env_steps=10,
                                   policy_calls=3) for i in range(5)])
        return dict(meta=meta, data=data)

    F = ("h", "q0", "u", "mu", "logp", "task", "state", "call")
    a = cell()
    assert compare_cells(a, cell(), F, log=lambda *_: None) == []

    # --- расхождение в ЛЮБОМ поле буфера обязано быть замечено ------------
    for f in F:
        b = cell()
        b["data"][f] = b["data"][f] + 1
        bad = compare_cells(a, b, F, log=lambda *_: None)
        assert any(x.startswith(f + ":") for x in bad), (f, bad)

    # --- ОДИНАКОВЫЕ ИСХОДЫ ПРИ РАЗНЫХ ДЕЙСТВИЯХ: главный сценарий ---------
    # Ровно то, ради чего эта регрессия и написана: успехи те же, буфер другой.
    b = cell()
    b["data"]["u"] = b["data"]["u"] * 1.0001
    bad = compare_cells(a, b, F, log=lambda *_: None)
    assert bad and all("эпизод" not in x for x in bad), bad

    # --- поток шума: длина и содержимое ----------------------------------
    b = cell()
    b["meta"]["eps_sha1_by_call"] = ["a", "b"]
    assert any("вызовов политики" in x
               for x in compare_cells(a, b, F, log=lambda *_: None))
    b = cell()
    b["meta"]["eps_sha1_by_call"] = ["a", "X", "c"]
    assert any("разошёлся с вызова 1" in x
               for x in compare_cells(a, b, F, log=lambda *_: None))
    b = cell()
    b["meta"]["eps_sha1_by_call"] = []
    a2 = cell()
    a2["meta"]["eps_sha1_by_call"] = []
    assert any("не записан" in x
               for x in compare_cells(a2, b, F, log=lambda *_: None))

    # --- эпизоды ----------------------------------------------------------
    b = cell()
    b["meta"]["episodes"][2]["success"] = False
    assert any("success" in x
               for x in compare_cells(a, b, F, log=lambda *_: None))
    b = cell()
    b["meta"]["episodes"][0]["init_hash_full"] = "ДРУГОЕ"
    assert any("init_hash_full" in x
               for x in compare_cells(a, b, F, log=lambda *_: None))

    # --- отсутствующее поле не считается совпадением -----------------------
    b = cell()
    del b["data"]["logp"]
    assert any("нет полей" in x or "состав буфера" in x
               for x in compare_cells(a, b, F, log=lambda *_: None))
    print("самопроверка k13h_seam_regression пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument("--out", default="data/k13h_seam_regression.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not a.before or not a.after:
        ap.error("нужны --before и --after")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    fields = buffer_fields()
    A = torch.load(a.before, map_location="cpu", weights_only=False)
    B = torch.load(a.after, map_location="cpu", weights_only=False)
    print(f"\n  СВЕРКА ПОЗИЦИОННОГО ПУТИ ДО И ПОСЛЕ ШВА")
    print(f"    до:    {a.before}")
    print(f"    после: {a.after}")
    print(f"    поля буфера из K-12d: {', '.join(fields)}")
    bad = compare_cells(A, B, fields)
    out = dict(before=a.before, after=a.after, fields=list(fields),
               n_records=int(len(A["data"][fields[0]])),
               n_calls=len((A["meta"] or {}).get("eps_sha1_by_call") or []),
               mismatches=bad, identical=not bad,
               script_sha1=hashlib.sha1(
                   open(os.path.abspath(__file__), "rb").read()
               ).hexdigest()[:12])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = f"{a.out}.tmp.{os.getpid()}"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, a.out)
    if bad:
        print("\n  РАСХОЖДЕНИЯ:")
        for x in bad:
            print(f"    {x}")
        raise SystemExit(
            f"позиционный путь изменился после врезки шва ({len(bad)} "
            f"расхождений). Шов не принимается: ветвь, которая не должна была "
            f"меняться, изменилась")
    print(f"\n  позиционный путь совпал ПОБИТОВО по всем полям, "
          f"{out['n_records']} записей, {out['n_calls']} вызовов политики")
    print(f"  сохранено: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
