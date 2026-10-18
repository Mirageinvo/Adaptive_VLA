#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-18, разведочная проверка: изменение качества full48 во время обучения.

Сохранённый чекпойнт full48_rl@100k оценивается на итоговых стартах 30–44
(30 волн, без шума, без обновлений и записи в replay) тем же сборщиком и той
же сборкой q0; затем парно по состояниям сравнивается с q0 и уже
измеренным full48_rl@300k (Мак-Немар, бутстрэп внутри задач).
Основное состояние прогона не меняется: используется его копия
(data/k18/run/probe_state.pt), результаты — в reports/k18/probe/.
Это НЕ замена отрицательного основного итога K-18.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

LABEL, MARK = "full48_rl", "100k"
PROBE_DIR = "reports/k18/probe"


def compare(probe, q0, late):
    import k15h_analyze as ha
    import k18_analyze as an
    o = {k: an.outcomes(v) for k, v in (("full48_100k", probe), ("q0", q0),
                                        ("full48_300k", late))}
    states = sorted(o["q0"])
    for k, v in o.items():
        if sorted(v) != states or len(states) != 150:
            raise SystemExit(f"{k}: набор состояний не равен итоговому банку")
    res = dict(states=len(states),
               sr={k: ha.equal_weight(v, states) for k, v in o.items()},
               per_task={k: {str(t): sum(v[s] for s in states if s[0] == t)
                             for t in range(10)} for k, v in o.items()})
    for a, b in (("full48_100k", "q0"), ("full48_100k", "full48_300k"),
                 ("full48_300k", "q0")):
        res[f"{a} − {b}"] = ha.compare(o[a], o[b], states, 10000, 20261018)
    return res


def run(a):
    import torch
    import k18_analyze as an
    import k18_collector as kc
    import k18_train as kt
    cfg = json.load(open(a.config))
    prob = kt.config_problems(cfg)
    if prob:
        raise SystemExit("; ".join(prob[:4]))
    if not os.path.exists(a.probe_state):
        shutil.copyfile(a.state, a.probe_state)
    kt.EVAL_DIR = PROBE_DIR
    ns = argparse.Namespace(config=a.config, state=a.probe_state,
                            vla_device=a.vla_device, rl_device=a.rl_device,
                            ckpt_dir=a.ckpt_dir)
    r = kt.Runner(ns, cfg, torch)
    r.sim = kc.Sim(cfg["n_envs"], "persistent")          # среды до CUDA
    r.rl_dev = torch.device(a.rl_device)
    r.load_or_new()
    if r.state["stage"] != "done":
        raise SystemExit("основной прогон K-18 не завершён")
    if r.state["worker_mode"] != "persistent":
        r.sim.close()
        r.sim = kc.Sim(cfg["n_envs"], r.state["worker_mode"])
    r.vla = kc.VLA(a.vla_device)
    r.enc = kc.Encoder(cfg["enc"]["path"], r.rl_dev, cfg["enc"]["state_sha1"])
    r.check_build()
    pin = r.state["ckpts"][f"{LABEL}@{MARK}"]
    if kt.sha(pin["path"]) != pin["sha1"]:
        raise SystemExit("чекпойнт full48_rl@100k изменён")
    ck = torch.load(pin["path"], map_location="cpu", weights_only=False)
    want = dict(variant=LABEL, mark=MARK, config_sha1=kt.sha(a.config),
                vla_build=r.vla.build_id, actor_sha1=pin["actor_sha1"])
    bad = [k for k, v in want.items() if ck.get(k) != v]
    if bad:
        raise SystemExit(f"чекпойнт не совпадает: {bad}")
    r.agents[LABEL].actor.load_state_dict(ck["actor"])
    r.agents[LABEL].version = int(ck["version"])        # версия чекпойнта
    if r.actor_sha(LABEL) != pin["actor_sha1"]:
        raise SystemExit("загруженные веса ≠ закреплённым")
    r.eval_bank(LABEL, MARK, "final")
    r.sim.close()
    pr = an.load_eval(PROBE_DIR, LABEL, MARK, "final")
    cs = kt.sha(a.config)
    p = an.eval_problems(pr, LABEL, MARK, "final", cs)
    q0 = an.load_eval(a.eval_dir, "q0", "q0", "final")
    late = an.load_eval(a.eval_dir, LABEL, "300k", "final")
    p += an.eval_problems(q0, "q0", "q0", "final", cs)
    p += an.eval_problems(late, LABEL, "300k", "final", cs)
    if len({pr["vla_build"], q0["vla_build"], late["vla_build"]}) != 1:
        p.append("разные сборки q0")
    if p:
        raise SystemExit("; ".join(p[:4]))
    res = compare(pr, q0, late)
    res["note"] = ("разведочная проверка изменения качества во время "
                   "обучения; не заменяет основной итог K-18")
    return res


def report(res):
    print("  SR на стартах 30–44: " + ", ".join(
        f"{k} {100 * v:.1f}%" for k, v in res["sr"].items()))
    for k, v in res["per_task"].items():
        print(f"    {k:12} " + " ".join(f"t{t} {n}/15" for t, n in v.items()))
    for k in ("full48_100k − q0", "full48_100k − full48_300k",
              "full48_300k − q0"):
        c = res[k]
        print(f"  {k}: ΔSR {100 * c['d_sr_equal_weight']:+.1f} п.п. "
              f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
              f"+{c['gain']}/−{c['loss']}, Мак-Немар p {c['mcnemar_p']:.4f}")


def selftest():
    rng = np.random.default_rng(0)
    eps = lambda f: dict(episodes=[dict(task=t, init_state_id=s,     # noqa
                                        success=bool(f(t, s)))
                                   for t in range(10) for s in range(30, 45)])
    base = {(t, s): rng.random() < .8 for t in range(10) for s in range(30, 45)}
    r = compare(eps(lambda t, s: base[(t, s)] or t == 8),
                eps(lambda t, s: base[(t, s)]),
                eps(lambda t, s: base[(t, s)]))
    assert r["full48_100k − q0"]["loss"] == 0
    assert r["full48_100k − q0"]["gain"] == sum(not base[(8, s)]
                                                for s in range(30, 45))
    try:
        compare(eps(lambda t, s: True), dict(episodes=[]), eps(lambda t, s:
                                                               True))
        raise AssertionError("принят неполный банк")
    except SystemExit:
        pass
    print("самопроверка k18_probe пройдена")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-18: проверка full48@100k")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--config", default="data/k18/config.json")
    ap.add_argument("--state", default="data/k18/run/state.pt")
    ap.add_argument("--probe-state", default="data/k18/run/probe_state.pt")
    ap.add_argument("--eval-dir", default="reports/k18/eval")
    ap.add_argument("--ckpt-dir", default="data/k18/ckpt")
    ap.add_argument("--vla-device", default="cuda:1")
    ap.add_argument("--rl-device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(PROBE_DIR,
                                                  "full48_100k_vs.json"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        res = run(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise
    report(res)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    print(f"  сводка: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
