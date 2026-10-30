#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, этап 2: официальная оценка обученной модели.

serve — сервер политики: официальный XPolicyLab PolicyServer (ws) с
  моделью rt2_policy.Model (окружение vla, устройство — --device; дерево
  XPolicyLab не меняется, CUDA_VISIBLE_DEVICES не используется). Клиент —
  официальный RoboTwin scripts/eval_policy_xpolicylab.py (окружение
  RoboTwin, см. rt2_eval_run.sh): он сам ведёт эпизод, проверяет состояния
  экспертом (seed 0 -> seed среды от 100000, неустойчивые и непройденные
  экспертом пропускаются) и считает успех; каждое состояние — первый
  проверенный экспертом seed после предыдущего.
summarize — сводка по задачам из журналов модели (episodes.jsonl на
  задачу): ровно N эпизодов, без расхождений prepare_case/trial_end, seed
  уникальны и ≥ 100000, одна модель; SR по задачам и среднее с равными
  весами задач; банк состояний (seed по задачам) закрепляется в отчёте.
"""
import argparse
import asyncio
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rt1_common as rc  # noqa: E402

PILOT_N = 10
SMOKE_TASKS = ("lift_pot", "open_laptop")
SEED_START = 100000


def serve(a):
    import rt2_policy as rp
    import rt2_train as r2
    root = os.environ.get("ROBOTWIN_ROOT")
    if not root:
        raise SystemExit("нужен ROBOTWIN_ROOT")
    xpl = os.path.join(root, "XPolicyLab")
    for p in (xpl, root):
        if p not in sys.path:
            sys.path.insert(0, p)
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    ck = a.ckpt or os.path.join(r2.outdir(a.seed, a.arch), "best.pt")
    rep = a.train_report or r2.report_path(a.seed, a.arch)
    model = rp.Model(dict(ckpt=ck, train_report=rep, device=a.device,
                          log_dir=a.log_dir, n_exec=a.n_exec))
    json.dump(dict(model.meta, n_exec=a.n_exec, port=a.port),
              open(os.path.join(a.log_dir, "server.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    server = PolicyServer(model, PolicyServerConfig(
        host=a.host, port=int(a.port), ws_ping_interval_s=20.0,
        ws_ping_timeout_s=60.0))
    print(f"[rt2_eval] сервер политики ws://{a.host}:{a.port}, журнал "
          f"{a.log_dir}", flush=True)
    asyncio.run(server.serve_forever())
    return 0


def read_task(log_dir, task):
    p = os.path.join(log_dir, task, "episodes.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def task_problems(eps, task, n):
    p = []
    if len(eps) != n:
        p.append(f"{task}: эпизодов {len(eps)}, нужно {n}")
    seeds = [e.get("seed") for e in eps]
    if any(e.get("task") != task for e in eps):
        p.append(f"{task}: эпизоды другой задачи")
    if any(e.get("mismatch") for e in eps):
        p.append(f"{task}: prepare_case ≠ trial_end")
    if len(set(seeds)) != len(seeds) or any(
            not isinstance(s, int) or s < SEED_START for s in seeds):
        p.append(f"{task}: seed {seeds}")
    if seeds != sorted(seeds):
        p.append(f"{task}: seed не возрастают")
    if len({json.dumps(e.get("model"), sort_keys=True) for e in eps}) > 1:
        p.append(f"{task}: разные модели в одном наборе")
    if any(e.get("calls", 0) < 1 for e in eps):
        p.append(f"{task}: эпизод без вызова политики")
    return p


def summarize(log_dir, tasks, n, out=None):
    rows, prob, bank, model = {}, [], {}, None
    for t in tasks:
        eps = read_task(log_dir, t)
        prob += task_problems(eps, t, n)
        if eps:
            model = eps[0].get("model")
        succ = sum(bool(e["success"]) for e in eps)
        rows[t] = dict(success=succ, n=len(eps),
                       sr=succ / len(eps) if eps else None,
                       seeds=[e["seed"] for e in eps],
                       calls_median=float(np.median([e["calls"] for e in eps]))
                       if eps else None,
                       ms_median=float(np.median([e["ms_median"] for e in eps
                                                  if e.get("ms_median")]))
                       if eps else None)
        bank[t] = rows[t]["seeds"]
    srs = [r["sr"] for r in rows.values() if r["sr"] is not None]
    rep = dict(tasks=rows, n_per_task=n, complete=not prob, problems=prob,
               sr_equal_weight=float(np.mean(srs)) if len(srs) == len(tasks)
               else None, bank=bank, model=model, log_dir=os.path.abspath(
                   log_dir),
               note="официальный клиент RoboTwin eval_policy_xpolicylab.py, "
                    "seed 0 (seed среды от 100000, экспертная проверка), "
                    "task_config demo_clean; пилот — не основное сравнение")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(rep, open(out, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=1)
    for t, r in rows.items():
        print(f"  {t:20}: {r['success']}/{r['n']}" + (
            f" ({100 * r['sr']:.0f}%)" if r["sr"] is not None else "") +
            f"; seed {r['seeds'][:3]}…; вызовов (медиана) "
            f"{r['calls_median']}; мс/вызов {r['ms_median']}")
    if rep["sr_equal_weight"] is not None:
        print(f"  SR (равные веса задач): {100 * rep['sr_equal_weight']:.1f}%")
    print("  " + ("набор полный" if not prob else "НЕПОЛНЫЙ: " +
                  "; ".join(prob[:6])) + (f"; {out}" if out else ""))
    return 0 if not prob else 3


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        model = dict(ckpt_sha1="c", weights_hash="w")
        for t, k in (("lift_pot", 7), ("open_laptop", 3)):
            os.makedirs(os.path.join(td, t))
            with open(os.path.join(td, t, "episodes.jsonl"), "w") as fh:
                for i in range(10):
                    fh.write(json.dumps(dict(
                        task=t, seed=SEED_START + 2 * i, success=i < k,
                        calls=5, ms_median=100.0, model=model)) + "\n")
        out = os.path.join(td, "r.json")
        assert summarize(td, ["lift_pot", "open_laptop"], 10, out) == 0
        r = json.load(open(out))
        assert r["tasks"]["lift_pot"]["success"] == 7 and \
            abs(r["sr_equal_weight"] - 0.5) < 1e-12
        assert r["bank"]["open_laptop"][0] == SEED_START
        # неполный набор, повтор seed, чужая задача, разные модели
        assert summarize(td, ["lift_pot"], 11) == 3
        eps = read_task(td, "lift_pot")
        bad = [dict(e) for e in eps]
        bad[1]["seed"] = bad[0]["seed"]
        assert task_problems(bad, "lift_pot", 10)
        bad = [dict(e) for e in eps]
        bad[2]["model"] = dict(ckpt_sha1="x")
        assert task_problems(bad, "lift_pot", 10)
        bad = [dict(e) for e in eps]
        bad[3]["task"] = "open_laptop"
        assert task_problems(bad, "lift_pot", 10)
        bad = [dict(e) for e in eps]
        bad[4]["mismatch"] = "x"
        assert task_problems(bad, "lift_pot", 10)
    assert set(SMOKE_TASKS) <= set(rc.TASKS) and len(rc.TASKS) == 8
    print("самопроверка rt2_eval пройдена: сводка по задачам (SR, равные "
          "веса, банк seed), отказ при неполном наборе, повторе seed, чужой "
          "задаче, разных моделях и расхождении prepare_case/trial_end")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="RoboTwin: официальная оценка")
    ap.add_argument("cmd", choices=("serve", "summarize", "tasks"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="flat24")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--train-report", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--n-exec", type=int, default=rc.N_EXEC)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=PILOT_N)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    try:
        if a.cmd == "tasks":
            print(" ".join(rc.TASKS))
            return 0
        if a.cmd == "serve":
            if a.port is None or not a.log_dir:
                raise SystemExit("serve: нужны --port и --log-dir")
            os.makedirs(a.log_dir, exist_ok=True)
            return serve(a)
        return summarize(a.log_dir, a.tasks or list(rc.TASKS), a.n, a.out)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
