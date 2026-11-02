#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-29: детерминированная оценка (шум выключен) на 30…44.

eval --seed S --which init|final:
  init  — исходный контроль K-29: замороженный reference FP32 (= actor до
          RL), снимается ДО обучения;
  final — последний принятый actor (data/k29/runs/sS/final.pt, SHA из отчёта
          обучения), схват — reference.
Пять задач 0/1/6/7/8 × блоки 30/35/40 × 5 сред = 75 эпизодов на политику.
Волна — k29_collect.run_wave (свежая среда на эпизод, холодный старт) без
шума; блок пишется только целиком: JSON (эпизоды, счётчики, модель) + npz
(исполненные действия, done_step). Проверка блока: задача/старты, начальные
хеши каждого эпизода = закреплённым хешам K-25 (холодный старт), action_sha1
пересчитывается из npz, отпечаток модели = ожидаемому. Готовый проверенный
блок пропускается при повторе; повреждённый уходит в карантин.
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k29_collect as kc  # noqa: E402
import k29_protocol as pc  # noqa: E402

WHICH = ("init", "final")


def eval_dir(seed, which):
    return f"reports/k29/eval/s{seed}_{which}"


def block_path(seed, which, task, block):
    return os.path.join(eval_dir(seed, which), f"t{task}_i{block}.json")


def expected_model(p, seed, which):
    """Отпечаток оцениваемой политики (до загрузки модели)."""
    import k29_ppo as kpo
    if which == "init":
        return dict(which="init", mode="reference",
                    source=p["sources"][str(seed)]["weights_hash"])
    rep = json.load(open(kpo.report_path(seed), encoding="utf-8"))
    f = pc.final_path(seed)
    if pc.sha(f) != rep["final_sha1"]:
        raise SystemExit(f"{f} ≠ отчёту обучения")
    return dict(which="final", mode="actor",
                source=p["sources"][str(seed)]["weights_hash"],
                actor_hash=rep["actor_hash"], final_sha1=rep["final_sha1"])


def block_problems(path, p, seed, which, task, block, want_model):
    import k18_collector as k18c
    if not os.path.exists(path):
        return ["нет блока"]
    try:
        d = json.load(open(path, encoding="utf-8"))
        z = np.load(os.path.join(os.path.dirname(path), d["npz"]))
        A, ds = z["actions"], z["done_step"]
    except Exception as e:                                   # noqa: BLE001
        return [f"не читается: {type(e).__name__}"]
    pr = []
    if pc.sha(os.path.join(os.path.dirname(path), d["npz"])) != d["npz_sha1"]:
        pr.append("npz изменён")
    if d.get("model") != want_model or d.get("seed") != seed or \
            d.get("which") != which:
        pr.append("модель/сид/вид блока ≠ ожидаемым")
    if int(d.get("task", -1)) != task or d.get("starts") != list(
            range(block, block + 5)):
        pr.append("задача/старты")
    if d.get("contract") != pc.contract(p)["fingerprint"]:
        pr.append("контракт подготовки")
    for i, e in enumerate(d.get("episodes") or []):
        key = f"{task}:{e['init_state_id']}"
        if e["init_hash"] != p["init_ref"].get(key):
            pr.append(f"старт {key}: начальный хеш ≠ холодному эталону K-25")
        if k18c.action_sha(A, i, ds[i]) != e["action_sha1"] or \
                int(ds[i]) != e["done_step"]:
            pr.append(f"эпизод {i}: действия/done_step ≠ npz")
    if len(d.get("episodes") or []) != 5:
        pr.append("эпизодов не 5")
    return pr


def run_eval(a):
    import k18_collector as k18c
    sim = k18c.Sim(5, "persistent")           # среды до CUDA
    import torch
    import k29_model as km
    import k29_policy as kp
    p = pc.load_prepare()
    want = expected_model(p, a.seed, a.which)
    odir = eval_dir(a.seed, a.which)
    os.makedirs(os.path.join(odir, "quarantine"), exist_ok=True)
    todo = []
    for t in pc.EVAL_TASKS:
        for b in pc.EVAL_BLOCKS:
            path = block_path(a.seed, a.which, t, b)
            if os.path.exists(path):
                pr = block_problems(path, p, a.seed, a.which, t, b, want)
                if not pr:
                    continue
                q = os.path.join(odir, "quarantine",
                                 time.strftime("%Y%m%dT%H%M%S"))
                os.makedirs(q, exist_ok=True)
                for f in (path, path[:-5] + ".npz"):
                    if os.path.exists(f):
                        shutil.move(f, q)
                print(f"  t{t} i{b}: в карантин ({pr[:2]})", flush=True)
            todo.append((t, b))
    if not todo:
        print(f"  {odir}: все 15 блоков готовы и проверены")
        sim.close()
        return 0
    try:
        mod, bar, M, actor, ref, _c = pc.load_all(torch, p, a.seed, a.device)
        if a.which == "final":
            f = torch.load(pc.final_path(a.seed), map_location="cpu",
                           weights_only=False)
            actor.load_state_dict(f["actor"])
            if km.tensor_hash(actor.state_dict()) != want["actor_hash"]:
                raise SystemExit("final actor ≠ отчёту обучения")
        actor.eval()
        inputs = kp.Inputs(torch, bar["proc"], a.device)
        pol = kp.Policy(torch, mod, actor, ref, M, want["mode"])
        for t, b in todo:
            starts = list(range(b, b + 5))
            eps, _b, log, c = kc.run_wave(sim, inputs, pol, t, starts,
                                          record=False)
            pc.add_resource(f"eval_{a.which}_s{a.seed}", c["physical"])
            path = block_path(a.seed, a.which, t, b)
            npz = os.path.basename(path)[:-5] + ".npz"
            npath = os.path.join(odir, npz)
            with open(npath + ".tmp", "wb") as fh:
                np.savez_compressed(fh, actions=log["actions"],
                                    done_step=log["done_step"])
            os.replace(npath + ".tmp", npath)
            d = dict(seed=a.seed, which=a.which, task=t, starts=starts,
                     model=want, contract=pc.contract(p)["fingerprint"],
                     episodes=eps, counts=c, npz=npz, npz_sha1=pc.sha(npath),
                     env=kc.ENV, policy_sha1=pc.sha(os.path.join(
                         HERE, "k29_policy.py")))
            with open(path + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(d, fh, ensure_ascii=False, indent=1, default=float)
            os.replace(path + ".tmp", path)
            pr = block_problems(path, p, a.seed, a.which, t, b, want)
            if pr:
                raise SystemExit(f"t{t} i{b}: новый блок не прошёл проверку: "
                                 f"{pr[:3]}")
            print(f"  s{a.seed} {a.which} t{t} i{b}: успех "
                  f"{sum(e['success'] for e in eps)}/5, шагов {c['steps']}, "
                  f"{c['seconds'] / 60:.1f} мин", flush=True)
        return 0
    finally:
        sim.close()


def outcomes(seed, which, p=None):
    """{(задача, init): успех} проверенных блоков или отказ при неполноте."""
    p = p or pc.load_prepare()
    want = expected_model(p, seed, which)
    out = {}
    for t in pc.EVAL_TASKS:
        for b in pc.EVAL_BLOCKS:
            path = block_path(seed, which, t, b)
            pr = block_problems(path, p, seed, which, t, b, want)
            if pr:
                raise SystemExit(f"s{seed} {which} t{t} i{b}: {pr[:2]}")
            for e in json.load(open(path, encoding="utf-8"))["episodes"]:
                out[(t, int(e["init_state_id"]))] = bool(e["success"])
    return out


def check(a):
    try:
        o = outcomes(a.seed, a.which)
    except SystemExit as e:
        print(f"  {e.code}")
        return 3
    print(f"  s{a.seed} {a.which}: 75/75 эпизодов проверены; SR "
          f"{np.mean(list(o.values())):.3f}")
    return 0


def selftest():
    import tempfile
    # блок с неверным стартом/хешем и порченым npz отвергается
    import k18_collector as k18c
    with tempfile.TemporaryDirectory() as td:
        A = np.random.default_rng(0).normal(size=(30, 5, 7)).astype(np.float32)
        ds = np.array([10, -1, 5, 29, 3])
        np.savez_compressed(os.path.join(td, "t0_i30.npz"), actions=A,
                            done_step=ds)
        p = dict(init_ref={f"0:{i}": f"h{i}" for i in range(30, 35)},
                 sources={"0": dict(weights_hash="w")})
        orig = pc.contract
        pc.contract = lambda p_, seed=None: dict(fingerprint="F")
        try:
            eps = [dict(init_state_id=30 + i, init_hash=f"h{30 + i}",
                        action_sha1=k18c.action_sha(A, i, ds[i]),
                        done_step=int(ds[i]), success=bool(i % 2))
                   for i in range(5)]
            want = dict(which="init", mode="reference", source="w")
            d = dict(seed=0, which="init", task=0, starts=list(range(30, 35)),
                     model=want, contract="F", episodes=eps, npz="t0_i30.npz",
                     npz_sha1=pc.sha(os.path.join(td, "t0_i30.npz")))
            path = os.path.join(td, "t0_i30.json")
            json.dump(d, open(path, "w"))
            assert block_problems(path, p, 0, "init", 0, 30, want) == []
            for mut, word in ((dict(starts=list(range(31, 36))), "старты"),
                              (dict(model=dict(want, source="x")), "модель"),
                              (dict(contract="G"), "контракт")):
                json.dump(dict(d, **mut), open(path, "w"))
                assert any(word in x for x in block_problems(
                    path, p, 0, "init", 0, 30, want)), word
            e2 = [dict(e) for e in eps]
            e2[2]["init_hash"] = "zzz"
            json.dump(dict(d, episodes=e2), open(path, "w"))
            assert any("начальный хеш" in x for x in block_problems(
                path, p, 0, "init", 0, 30, want))
            e3 = [dict(e) for e in eps]
            e3[1]["action_sha1"] = "0" * 16
            json.dump(dict(d, episodes=e3), open(path, "w"))
            assert any("действия" in x for x in block_problems(
                path, p, 0, "init", 0, 30, want))
        finally:
            pc.contract = orig
    print("самопроверка k29_harness пройдена: блок проходит; старты, модель, "
          "контракт, начальный хеш (холодный эталон K-25) и действия/npz — "
          "отказ")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-29: оценка")
    ap.add_argument("cmd", choices=("eval", "check"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--which", choices=WHICH, required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    try:
        return run_eval(a) if a.cmd == "eval" else check(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
