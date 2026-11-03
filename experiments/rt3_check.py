#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, этап 3: привязка журналов оценки к банку случаев.

Окружение vla (без RoboTwin). Поверх проверок RT2 (rt2_eval: манифест
модели, журнал эпизодов, отметка done.json) — проверки банка:
bind      — <log_dir>/rt3_bank.json: путь и отпечаток банка; существующая
            привязка к другому банку — отказ (нужен новый каталог).
is-done / mark-done — задача пропускается при возобновлении и берётся в
            анализ, только если есть отметка rt3_done.json, привязанная к
            текущему банку (отпечаток всего банка и SHA записи задачи:
            случаи, инструкции, отпечатки, конфигурации), к манифесту модели,
            n и SHA журналов, и журналы по-прежнему проходят проверку:
            — rt3_cases.jsonl (пишет клиент после сверки начального
              состояния): ровно первые n случаев банка по порядку; seed,
              ep_num, инструкция и подтверждённый отпечаток (state, poses,
              pose_names, vision) = банку; отпечаток банка = текущему;
            — episodes.jsonl (пишет сервер): seed по порядку = банку;
              инструкция, которую видела модель (instruction_obs), и
              переданная в prepare_case (если записана) = банку;
            — штатные проверки RT2 (rt2_eval.task_problems с манифестом).
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BANK_KIND = "rt3_bank"
BIND = "rt3_bank.json"
CASES = "rt3_cases.jsonl"
DONE = "rt3_done.json"
FP_KEYS = ("state", "poses", "pose_names", "vision")
CASE_KEYS = ("seed", "ep_num", "instruction")


def sha_file(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def bank_fp(bank):
    return hashlib.sha1(json.dumps(bank["tasks"], sort_keys=True).encode()
                        ).hexdigest()


def load_bank(path):
    bank = json.load(open(path, encoding="utf-8"))
    if bank.get("kind") != BANK_KIND or bank_fp(bank) != bank.get(
            "fingerprint"):
        raise SystemExit(f"банк {path} повреждён")
    return bank


def task_entry_sha(bank, task, n):
    tb = bank["tasks"][task]
    return hashlib.sha1(json.dumps(dict(cases=tb["cases"][:n],
                                        config=tb["config"]),
                                   sort_keys=True).encode()).hexdigest()


def bind(log_dir, bank_path):
    bank = load_bank(bank_path)
    want = dict(bank=os.path.abspath(bank_path), fingerprint=bank["fingerprint"])
    p = os.path.join(log_dir, BIND)
    if os.path.exists(p):
        old = json.load(open(p, encoding="utf-8"))
        if old.get("fingerprint") != want["fingerprint"]:
            raise SystemExit(f"журнал {log_dir} привязан к другому банку "
                             f"({str(old.get('fingerprint'))[:12]}); нужен "
                             f"новый каталог")
        print(f"  банк совпадает с привязкой {log_dir}")
        return 0
    os.makedirs(log_dir, exist_ok=True)
    json.dump(want, open(p, "w", encoding="utf-8"), indent=1)
    print(f"  журнал привязан к банку {want['fingerprint'][:12]}")
    return 0


def read_jsonl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def expected_of(log_dir):
    import rt2_eval as re_
    p = os.path.join(log_dir, re_.EXPECTED)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def task_problems(log_dir, bank, task, n):
    import rt2_eval as re_
    p = []
    bp = os.path.join(log_dir, BIND)
    b = json.load(open(bp, encoding="utf-8")) if os.path.exists(bp) else {}
    if b.get("fingerprint") != bank["fingerprint"]:
        p.append(f"{task}: журнал не привязан к текущему банку")
    if task not in bank["tasks"]:
        return p + [f"{task}: задачи нет в банке"]
    want = bank["tasks"][task]["cases"][:n]
    if len(want) != n:
        return p + [f"{task}: в банке {len(want)} < {n} случаев"]
    cl = read_jsonl(os.path.join(log_dir, task, CASES))
    if len(cl) != n:
        p.append(f"{task}: подтверждённых случаев {len(cl)}, нужно {n}")
    for i, (c, w) in enumerate(zip(cl, want)):
        bad = [k for k in CASE_KEYS if c.get(k) != w.get(k)]
        bad += [f"fp.{k}" for k in FP_KEYS
                if (c.get("fingerprint") or {}).get(k) !=
                w["fingerprint"].get(k)]
        if c.get("task") != task:
            bad.append("task")
        if c.get("bank_fingerprint") != bank["fingerprint"]:
            bad.append("bank_fingerprint")
        if bad:
            p.append(f"{task}: случай {i} (seed {w['seed']}) ≠ банку: {bad}")
    eps = re_.read_task(log_dir, task)
    seeds = [e.get("seed") for e in eps]
    if seeds != [w["seed"] for w in want]:
        p.append(f"{task}: seed эпизодов {seeds} ≠ банку "
                 f"{[w['seed'] for w in want]}")
    for e, w in zip(eps, want):
        if e.get("instruction_obs") != w["instruction"]:
            p.append(f"{task}: seed {w['seed']}: модель видела другую "
                     f"инструкцию")
        if e.get("instruction") is not None and \
                e.get("instruction") != w["instruction"]:
            p.append(f"{task}: seed {w['seed']}: инструкция prepare_case ≠ "
                     f"банку")
    p += re_.task_problems(eps, task, n, expected_of(log_dir))
    return p


def _done_record(log_dir, bank, task, n):
    d = os.path.join(log_dir, task)
    return dict(task=task, n=n, bank_fingerprint=bank["fingerprint"],
                task_entry_sha1=task_entry_sha(bank, task, n),
                expected=expected_of(log_dir),
                episodes_sha1=sha_file(os.path.join(d, "episodes.jsonl")),
                cases_sha1=sha_file(os.path.join(d, CASES)))


def done_problems(log_dir, bank, task, n):
    dp = os.path.join(log_dir, task, DONE)
    if not os.path.exists(dp):
        return [f"{task}: нет отметки rt3"]
    d = json.load(open(dp, encoding="utf-8"))
    p = []
    try:
        now = _done_record(log_dir, bank, task, n)
    except FileNotFoundError as e:
        return [f"{task}: нет журнала ({e.filename})"]
    if now["expected"] is None:
        p.append(f"{task}: нет манифеста модели")
    bad = [k for k in now if d.get(k) != now[k]]
    if bad:
        p.append(f"{task}: отметка rt3 не совпадает с текущим ({bad})")
    if not p:
        p += task_problems(log_dir, bank, task, n)
    return p


def mark_done(log_dir, bank, task, n, client_log):
    import rt2_eval as re_
    prob = re_.client_problems(client_log) + task_problems(log_dir, bank,
                                                           task, n)
    if expected_of(log_dir) is None:
        prob.append("нет манифеста модели")
    if prob:
        raise SystemExit("отметка rt3 не ставится: " + "; ".join(prob[:4]))
    rec = dict(_done_record(log_dir, bank, task, n),
               client_log=os.path.abspath(client_log),
               client_log_sha1=sha_file(client_log))
    json.dump(rec, open(os.path.join(log_dir, task, DONE), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  {task}: отметка rt3 поставлена (банк "
          f"{bank['fingerprint'][:12]})")
    return 0


def load_outcomes(log_dir, bank, n=None):
    """{(задача, seed): успех} по всем задачам банка — только при
    действительных отметках rt3 и RT2; иначе (None, проблемы)."""
    import rt2_eval as re_
    out, prob = {}, []
    for t in sorted(bank["tasks"]):
        k = n or len(bank["tasks"][t]["cases"])
        p = done_problems(log_dir, bank, t, k) + re_.task_done_problems(
            log_dir, t, k)
        if p:
            prob += p
            continue
        for e in re_.read_task(log_dir, t):
            out[(t, int(e["seed"]))] = bool(e["success"])
    return (out if not prob else None), prob


def selftest():
    import tempfile
    import rt2_eval as re_
    import rt2_policy as rp
    tmp = tempfile.mkdtemp()
    fp = dict(state="s", poses="p", pose_names=["pot"], vision="v")
    tasks = dict(lift_pot=dict(cases=[dict(task="lift_pot", seed=100000 + i,
                                           ep_num=i, instruction=f"lift {i}",
                                           instruction_type="unseen",
                                           fingerprint=dict(fp, state=f"s{i}"))
                                      for i in range(2)],
                               skipped=[], config=dict(task_config="c")))
    bank = dict(kind=BANK_KIND, tasks=tasks)
    bank["fingerprint"] = bank_fp(bank)
    bpath = os.path.join(tmp, "bank.json")
    json.dump(bank, open(bpath, "w"))
    model = {k: f"m_{k}" for k in rp.MODEL_KEYS}
    exp = dict(model, n_exec=8)

    def make(log, bank_=bank, cases_mut=None, eps_mut=None):
        os.makedirs(os.path.join(log, "lift_pot"), exist_ok=True)
        json.dump(exp, open(os.path.join(log, re_.EXPECTED), "w"))
        bind(log, bpath)
        cs = [dict(task="lift_pot", seed=c["seed"], ep_num=c["ep_num"],
                   instruction=c["instruction"],
                   bank_fingerprint=bank_["fingerprint"],
                   fingerprint=dict(c["fingerprint"]))
              for c in bank_["tasks"]["lift_pot"]["cases"]]
        es = [dict(task="lift_pot", seed=c["seed"], instruction=None,
                   instruction_obs=c["instruction"], calls=3, success=i == 0,
                   model=model, n_exec=8)
              for i, c in enumerate(bank_["tasks"]["lift_pot"]["cases"])]
        if cases_mut:
            cases_mut(cs)
        if eps_mut:
            eps_mut(es)
        for nm, rows in ((CASES, cs), ("episodes.jsonl", es)):
            with open(os.path.join(log, "lift_pot", nm), "w") as fh:
                fh.write("".join(json.dumps(r) + "\n" for r in rows))
        cl = os.path.join(log, "client.log")
        open(cl, "w").write("Success rate: 1/2\n")
        return cl

    log = os.path.join(tmp, "ok")
    cl = make(log)
    assert task_problems(log, bank, "lift_pot", 2) == []
    mark_done(log, bank, "lift_pot", 2, cl)
    assert done_problems(log, bank, "lift_pot", 2) == []
    # другой банк с теми же seed: привязка и отметка не принимаются
    b2 = json.loads(json.dumps(bank))
    b2["tasks"]["lift_pot"]["cases"][1]["fingerprint"]["poses"] = "p2"
    b2["fingerprint"] = bank_fp(b2)
    assert done_problems(log, b2, "lift_pot", 2)
    assert any("не привязан" in x for x in task_problems(log, b2,
                                                         "lift_pot", 2))
    b2p = os.path.join(tmp, "bank2.json")
    json.dump(b2, open(b2p, "w"))
    try:
        bind(log, b2p)
        raise AssertionError("привязка к другому банку принята")
    except SystemExit:
        pass
    # другой отпечаток случая, другая инструкция, порядок seed
    for nm, cm, em in (
            ("fp", lambda c: c[1]["fingerprint"].update(vision="x"), None),
            ("instr", None, lambda e: e[0].update(instruction_obs="other")),
            ("order", None, lambda e: e.reverse()),
            ("missing", lambda c: c.pop(), None)):
        lg = os.path.join(tmp, nm)
        make(lg, cases_mut=cm, eps_mut=em)
        assert task_problems(lg, bank, "lift_pot", 2), nm
    # журнал изменён после отметки
    with open(os.path.join(log, "lift_pot", CASES), "a") as fh:
        fh.write("\n")
    assert done_problems(log, bank, "lift_pot", 2)
    print("самопроверка rt3_check пройдена: привязка к банку, отметка rt3 "
          "(банк, запись задачи, манифест, SHA журналов); другой банк с теми "
          "же seed, отпечаток, инструкция, порядок, нехватка случаев, "
          "изменение журнала — отказ")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="RT3: привязка к банку")
    ap.add_argument("cmd", choices=("bind", "is-done", "mark-done"))
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--task")
    ap.add_argument("--n", type=int)
    ap.add_argument("--log")
    a = ap.parse_args()
    try:
        if a.cmd == "bind":
            return bind(a.log_dir, a.bank)
        bank = load_bank(a.bank)
        if a.cmd == "is-done":
            p = done_problems(a.log_dir, bank, a.task, a.n)
            if p:
                print("  " + "; ".join(p[:3]))
            return 0 if not p else 1
        return mark_done(a.log_dir, bank, a.task, a.n, a.log)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
