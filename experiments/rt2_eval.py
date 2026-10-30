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
EXPECTED = "expected.json"
ROLLOUT_ERRORS = ("Policy rollout error", "Aborting task")
SMOKE_TASKS = ("lift_pot", "open_laptop")
SEED_START = 100000


def model_paths(a):
    import rt2_train as r2
    return (a.ckpt or os.path.join(r2.outdir(a.seed, a.arch), "best.pt"),
            a.train_report or r2.report_path(a.seed, a.arch))


def manifest_of(meta, n_exec):
    import rt2_policy as rp
    return dict({k: meta[k] for k in rp.MODEL_KEYS}, n_exec=int(n_exec))


def expect(a):
    """Манифест ожидаемой модели прогона (без GPU) -> <log_dir>/expected.json.
    Существующий манифест сверяется ДО записи: другая модель — отказ (новый
    каталог журнала), та же — продолжение."""
    import torch
    import rt1_data as rd
    import rt2_policy as rp
    ck, rep = model_paths(a)
    D = rd.Data(check=True, check_files=False)
    man = manifest_of(rp.load_spec(torch, ck, rep, D.prep["prep_id"])["meta"],
                      a.n_exec)
    p = os.path.join(a.log_dir, EXPECTED)
    if os.path.exists(p):
        old = json.load(open(p, encoding="utf-8"))
        if old != man:
            diff = sorted(k for k in set(old) | set(man)
                          if old.get(k) != man.get(k))
            raise SystemExit(f"журнал {a.log_dir} принадлежит другой модели "
                             f"(расхождения {diff}); нужен новый каталог")
        print(f"  ожидаемая модель совпадает с журналом {a.log_dir}")
        return 0
    os.makedirs(a.log_dir, exist_ok=True)
    json.dump(man, open(p, "w", encoding="utf-8"), indent=1)
    print(f"  ожидаемая модель закреплена: {p} ({man['ckpt_sha1'][:12]}, "
          f"{man['weights_hash'][:12]})")
    return 0


def serve(a):
    import rt2_policy as rp
    root = os.environ.get("ROBOTWIN_ROOT")
    if not root:
        raise SystemExit("нужен ROBOTWIN_ROOT")
    xpl = os.path.join(root, "XPolicyLab")
    for p in (xpl, root):
        if p not in sys.path:
            sys.path.insert(0, p)
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    ck, rep = model_paths(a)
    p = os.path.join(a.log_dir, EXPECTED)
    if not os.path.exists(p):
        raise SystemExit(f"нет {p}: сначала rt2_eval.py expect")
    want = json.load(open(p, encoding="utf-8"))
    model = rp.Model(dict(ckpt=ck, train_report=rep, device=a.device,
                          log_dir=a.log_dir, n_exec=a.n_exec))
    got = manifest_of(model.meta, a.n_exec)
    if got != want:
        raise SystemExit(f"загруженная модель ≠ ожидаемой: {got} vs {want}")
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


def task_problems(eps, task, n, expected=None):
    p = []
    if expected is not None:
        import rt2_policy as rp
        bad = [e.get("seed") for e in eps
               if {k: (e.get("model") or {}).get(k) for k in rp.MODEL_KEYS}
               != {k: expected[k] for k in rp.MODEL_KEYS}
               or e.get("n_exec") != expected["n_exec"]]
        if bad:
            p.append(f"{task}: эпизоды другой модели (seed {bad[:3]})")
    err = [e.get("seed") for e in eps if e.get("errors")]
    if err:
        p.append(f"{task}: отказ модели в эпизодах {err[:3]}")
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


def client_problems(path):
    """Лог официального клиента: техническая ошибка раскатки (исключение,
    перехваченное клиентом; аварийная остановка задачи) — задача
    непригодна. Обычный неуспех и отбраковка состояний экспертом —
    допустимы."""
    if not os.path.exists(path):
        return [f"нет лога клиента {path}"]
    txt = open(path, encoding="utf-8", errors="replace").read()
    return [f"в логе клиента «{m}» ({txt.count(m)} раз)"
            for m in ROLLOUT_ERRORS if m in txt]


DONE = "done.json"


def _sha(path):
    import hashlib
    return hashlib.sha1(open(path, "rb").read()).hexdigest()


def task_done_problems(log_dir, task, n):
    """Задача пропускается при возобновлении, только если есть отметка
    завершения (после кода клиента 0, чистого лога клиента и полной сводки),
    она привязана к текущему манифесту, задаче, n и SHA журнала, а журнал
    по-прежнему проходит проверку."""
    p = []
    mp = os.path.join(log_dir, EXPECTED)
    dp = os.path.join(log_dir, task, DONE)
    ep = os.path.join(log_dir, task, "episodes.jsonl")
    if not os.path.exists(dp):
        return [f"{task}: нет отметки завершения"]
    d = json.load(open(dp, encoding="utf-8"))
    exp = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else None
    if d.get("expected") != exp or exp is None:
        p.append(f"{task}: отметка другой модели")
    if d.get("task") != task or d.get("n") != n:
        p.append(f"{task}: отметка для {d.get('task')}/{d.get('n')}")
    if not os.path.exists(ep) or _sha(ep) != d.get("episodes_sha1"):
        p.append(f"{task}: журнал изменён после отметки")
    if not p:
        p += task_problems(read_task(log_dir, task), task, n, exp)
    return p


def mark_done(log_dir, task, n, client_log):
    """Отметка завершения задачи — только после чистого лога клиента и
    полной проверки журнала против манифеста."""
    mp = os.path.join(log_dir, EXPECTED)
    if not os.path.exists(mp):
        raise SystemExit(f"нет {mp}")
    exp = json.load(open(mp, encoding="utf-8"))
    prob = client_problems(client_log) + task_problems(
        read_task(log_dir, task), task, n, exp)
    if prob:
        raise SystemExit("отметка завершения не ставится: " +
                         "; ".join(prob[:4]))
    ep = os.path.join(log_dir, task, "episodes.jsonl")
    d = dict(task=task, n=n, expected=exp, episodes_sha1=_sha(ep),
             client_log=os.path.abspath(client_log),
             client_log_sha1=_sha(client_log))
    json.dump(d, open(os.path.join(log_dir, task, DONE), "w",
                      encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  {task}: отметка завершения поставлена")
    return 0


def summarize(log_dir, tasks, n, out=None):
    rows, prob, bank, model = {}, [], {}, None
    p = os.path.join(log_dir, EXPECTED)
    expected = json.load(open(p, encoding="utf-8")) if os.path.exists(p) \
        else None
    if expected is None:
        prob.append(f"нет манифеста ожидаемой модели {p}")
    for t in tasks:
        eps = read_task(log_dir, t)
        prob += task_problems(eps, t, n, expected)
        if not os.path.exists(os.path.join(log_dir, t, DONE)):
            prob.append(f"{t}: нет отметки завершения")
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
               expected_model=expected,
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
    import rt2_policy as rp
    with tempfile.TemporaryDirectory() as td:
        model = {k: f"v_{k}" for k in rp.MODEL_KEYS}
        json.dump(dict(model, n_exec=8), open(os.path.join(td, EXPECTED),
                                              "w"))
        for t, k in (("lift_pot", 7), ("open_laptop", 3)):
            os.makedirs(os.path.join(td, t))
            with open(os.path.join(td, t, "episodes.jsonl"), "w") as fh:
                for i in range(10):
                    fh.write(json.dumps(dict(
                        task=t, seed=SEED_START + 2 * i, success=i < k,
                        calls=5, ms_median=100.0, model=model, n_exec=8))
                        + "\n")
        clean = os.path.join(td, "clean.log")
        open(clean, "w").write("Success rate: 7/10\nFail!\n")
        out = os.path.join(td, "r.json")
        assert summarize(td, ["lift_pot", "open_laptop"], 10, out) == 3
        # отметка: полный журнал + лог с ошибкой клиента — не ставится и
        # не переиспользуется; обычный неуспех без ошибок — принимается
        dirty = os.path.join(td, "dirty.log")
        open(dirty, "w").write("Policy rollout error:\nTraceback\nFail!\n")
        try:
            mark_done(td, "lift_pot", 10, dirty)
            raise AssertionError("отметка при ошибке раскатки")
        except SystemExit:
            pass
        assert task_done_problems(td, "lift_pot", 10)
        for t in ("lift_pot", "open_laptop"):
            assert mark_done(td, t, 10, clean) == 0
            assert task_done_problems(td, t, 10) == []
        assert summarize(td, ["lift_pot", "open_laptop"], 10, out) == 0
        assert task_done_problems(td, "lift_pot", 9)          # другое n
        ep_ = os.path.join(td, "open_laptop", "episodes.jsonl")
        open(ep_, "a").write("\n")                            # журнал изменён
        assert any("изменён" in x for x in
                   task_done_problems(td, "open_laptop", 10))
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
        exp = json.load(open(os.path.join(td, EXPECTED)))
        assert task_problems(eps, "lift_pot", 10, exp) == []
        # другая задача целиком другой моделью — отказ по манифесту прогона
        bad = [dict(e, model=dict(model, weights_hash="другая"))
               for e in eps]
        assert any("другой модели" in x for x in
                   task_problems(bad, "lift_pot", 10, exp))
        bad = [dict(e, n_exec=4) for e in eps]
        assert task_problems(bad, "lift_pot", 10, exp)
        bad = [dict(e) for e in eps]
        bad[5]["errors"] = ["SystemExit: выход не конечен"]
        assert any("отказ модели" in x for x in
                   task_problems(bad, "lift_pot", 10, exp))
        with open(os.path.join(td, "open_laptop", "episodes.jsonl"), "w") as fh:
            for i in range(10):
                fh.write(json.dumps(dict(
                    task="open_laptop", seed=SEED_START + i, success=False,
                    calls=5, model=dict(model, ckpt_sha1="B"), n_exec=8))
                    + "\n")
        assert summarize(td, ["lift_pot", "open_laptop"], 10) == 3
        os.remove(os.path.join(td, EXPECTED))
        assert summarize(td, ["lift_pot"], 10) == 3     # без манифеста
        # лог клиента: перехваченная ошибка раскатки и аварийная остановка
        lg = os.path.join(td, "client.log")
        open(lg, "w").write("Success rate: 0/1\nFail!\n")
        assert client_problems(lg) == []
        open(lg, "w").write("\x1b[91mPolicy rollout error:\x1b[0m\nTraceback\n")
        assert client_problems(lg)
        open(lg, "w").write("RuntimeError: Aborting task: 5 consecutive")
        assert client_problems(lg)
        assert client_problems(os.path.join(td, "нет.log"))
    # возобновление: манифест сверяется ДО записи; другая модель — отказ
    import types
    import rt1_data as rd
    saved = (rp.load_spec, rd.Data)
    metas = [dict({k: f"a_{k}" for k in rp.MODEL_KEYS}),
             dict({k: f"b_{k}" for k in rp.MODEL_KEYS})]
    try:
        rd.Data = lambda **kw: types.SimpleNamespace(prep=dict(prep_id="P"))
        with tempfile.TemporaryDirectory() as td:
            args = types.SimpleNamespace(ckpt="c", train_report="r",
                                         log_dir=td, n_exec=8, seed=0,
                                         arch="flat24")
            rp.load_spec = lambda *x: dict(meta=metas[0])
            assert expect(args) == 0
            first = open(os.path.join(td, EXPECTED)).read()
            assert expect(args) == 0                    # та же модель
            rp.load_spec = lambda *x: dict(meta=metas[1])
            try:
                expect(args)
                raise AssertionError("журнал другой модели принят")
            except SystemExit as e:
                assert "другой модели" in str(e.code)
            assert open(os.path.join(td, EXPECTED)).read() == first
    finally:
        rp.load_spec, rd.Data = saved
    assert set(SMOKE_TASKS) <= set(rc.TASKS) and len(rc.TASKS) == 8
    print("самопроверка rt2_eval пройдена: сводка по задачам (SR, равные "
          "веса, банк seed), отказ при неполном наборе, повторе seed, чужой "
          "задаче, разных моделях (в задаче и между задачами по манифесту "
          "прогона), другом n_exec, без манифеста, отказе модели, "
          "расхождении prepare_case/trial_end; лог клиента — ошибка "
          "раскатки и аварийная остановка задачи; манифест при возобновлении "
          "сверяется до записи (другая модель — отказ, файл не меняется); "
          "отметка завершения задачи — только после чистого лога клиента и "
          "полной сводки, привязана к манифесту, n и SHA журнала; полный "
          "журнал с ошибкой клиента без отметки не переиспользуется")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="RoboTwin: официальная оценка")
    ap.add_argument("cmd", choices=("serve", "summarize", "tasks", "expect",
                                    "client-check", "mark-done", "is-done"))
    ap.add_argument("--log", default=None)
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
        if a.cmd == "expect":
            if not a.log_dir:
                raise SystemExit("expect: нужен --log-dir")
            return expect(a)
        if a.cmd in ("mark-done", "is-done"):
            if not a.log_dir or not a.tasks or len(a.tasks) != 1:
                raise SystemExit(f"{a.cmd}: нужны --log-dir и одна --tasks")
            if a.cmd == "mark-done":
                return mark_done(a.log_dir, a.tasks[0], a.n, a.log)
            prob = task_done_problems(a.log_dir, a.tasks[0], a.n)
            print("  " + ("; ".join(prob) if prob else
                          f"{a.tasks[0]}: завершена и проверена"))
            return 3 if prob else 0
        if a.cmd == "client-check":
            prob = client_problems(a.log)
            print("  " + ("; ".join(prob) if prob else
                          "лог клиента: технических ошибок раскатки нет"))
            return 3 if prob else 0
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
