#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: план пилота и проверка блоков.

ПЛАН: fullbar (штатный харнесс), flat24, tree_adapt24 (k21_policy);
LIBERO-10, задачи 0–9, блок 25 (старты 25–29), 5 сред, сид 101; smoke —
задачи 0 и 8 (6 процессов, входят в 30). Закреплены: чекпойнты best.pt и
отчёт обучения, паспорт подготовки (prep_id) и снимок BAR, файлы стартов
LIBERO, код раскатки и модели.
БЛОК: строгая validate_block и контракт раскатки; для k21-рук —
метаданные руки (метка, план, чекпойнт, отпечаток весов, prep_id,
ревизия BAR, код), журнал (нумерация вызовов, конечность, пределы руки,
схват ±1) и исполненные actions = журнал.
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

KIND = "k21_pilot_plan"
PLAN = "reports/k21/plan_pilot.json"
LABELS = ("fullbar", "flat24", "tree_adapt24")
POLICY = dict(fullbar="fullbar", flat24="k15f", tree_adapt24="k15f")
BLOCKS = (25,)
SMOKE = [(0, 25), (8, 25)]
CODE = ("k21_policy.py", "k21_harness.py", "k21_protocol.py",
        "k21_analyze.py", "k21_rollout.sh", "k21_model.py", "k21_tree.py",
        "k21_data.py", "k21_train.py", "k20_data.py", "k16_models.py",
        "k16_harness.py", "k16_policy.py", "k9h_multiarm_gate.py",
        "k15g_local_probe.py", "k15h_protocol.py", "k15f_policy.py",
        "k21_fastload.py", "k21_train_fast.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]


def full_sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def make_plan(device):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import k21_data as d21
    import torch
    tr = json.load(open("reports/k21/train.json"))
    prep = json.load(open(d21.REPORT))
    # эталон full BAR и новые руки — из ОДНОГО закреплённого локального
    # снимка (путь содержит ревизию); содержимое файлов сверено с адресами
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    if os.path.basename(os.path.normpath(snap)) != \
            prep["bar_pin"]["revision"]:
        raise SystemExit("путь снимка не соответствует ревизии")
    inputs = dict(train_report="reports/k21/train.json", prepare=d21.REPORT,
                  teacher=d21.TEACHER_REPORT, harness_utils=hpc.HARNESS_UTILS,
                  inventory="reports/k15h/inventory.json")
    for lab in ("flat24", "tree_adapt24"):
        p = f"data/k21/runs/{lab}/best.pt"
        if full_sha(p) != tr[lab]["best_sha1"]:
            raise SystemExit(f"{p} не совпадает с отчётом обучения")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if ck["config"]["prep_id"] != prep["prep_id"]:
            raise SystemExit(f"{lab}: чекпойнт от другой подготовки")
        inputs[f"ckpt_{lab}"] = p
    inv = json.load(open("reports/k15h/inventory.json"))
    refs = {}
    for t, v in inv["libero_bank"].items():
        f = v["path"]
        if not os.path.exists(f) or full_sha(f) != v["sha1"]:
            raise SystemExit(f"файл стартов задачи {t} изменён: {f}")
        refs[f"init_t{t}"] = f
    used = {t: sorted(int(x) for x in v if 45 <= int(x) <= 49)
            for t, v in inv["used"].items()}
    if any(used.values()):
        raise SystemExit(f"резерв 45–49 использован: {used}")
    return dict(
        kind=KIND, created=datetime.datetime.now().isoformat(
            timespec="seconds"),
        labels=list(LABELS), policy=POLICY, tasks=list(range(10)),
        blocks=list(BLOCKS), n_envs=5, smoke=SMOKE, device=str(device),
        setup=dict(pr.expected_setup(), ckpt=snap),
        snapshot_path=snap, prep_id=prep["prep_id"],
        training_execution=dict(
            loader=dict(k21_fastload_sha1=full_sha(os.path.join(
                HERE, "k21_fastload.py")),
                k21_train_fast_sha1=full_sha(os.path.join(
                    HERE, "k21_train_fast.py"))),
            prefetch=dict(ahead=3, workers=6, bitwise_check_batches=4,
                          read="pread строки целиком"),
            used=dict(flat24="основная фаза с обновления 2500 (до этого — "
                             "штатная загрузка k21_data.make_batch)",
                      tree_adapt24="весь прогон (warmup и основная фаза)"),
            equivalence=("первые 4 батча каждого запуска совпали с "
                         "k21_data.make_batch побитово"),
            note=("загрузчик вне закреплённого контракта обучения; "
                  "согласовано агентом")),
        bar_revision=prep["bar_pin"]["revision"],
        bounds=prep["bounds_codec"],
        best=dict(flat24=tr["flat24"]["best"],
                  tree_adapt24=tr["tree_adapt24"]["best"]),
        inputs={k: os.path.abspath(v) for k, v in inputs.items()},
        inputs_sha1={k: sha(v) for k, v in inputs.items()},
        refs={k: os.path.abspath(v) for k, v in refs.items()},
        refs_sha1={k: sha(v) for k, v in refs.items()},
        code={f: sha(os.path.join(HERE, f)) for f in CODE},
        note=("старты 25–29 использовались в проекте раньше (разработочный "
              "банк); резерв 45–49 закрыт"))


def check_plan_inputs(plan):
    p = []
    if plan.get("kind") != KIND:
        return ["не план пилота K-21"]
    for grp in ("inputs", "refs"):
        for k, f in plan[grp].items():
            if not os.path.exists(f) or sha(f) != plan[f"{grp}_sha1"][k]:
                p.append(f"{grp} {k} изменён или отсутствует ({f})")
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    return p


def journal_problems(label, z, bounds):
    p = []
    need = (f"k15d_level_{label}", "k15g_chunk", "actions", "done_step")
    miss = [k for k in need if k not in z]
    if miss:
        return [f"{label}: в журнале нет {miss}"]
    lv = np.asarray(z[f"k15d_level_{label}"], np.float64)
    C = len(z["k15g_chunk"])
    if z["k15g_chunk"].tolist() != list(range(1, C + 1)):
        p.append(f"{label}: нумерация вызовов нарушена")
    if not np.isfinite(lv).all():
        p.append(f"{label}: журнал не конечен")
    L, U = np.asarray(bounds["L"]), np.asarray(bounds["U"])
    if (lv[..., :6] < L - 1e-4).any() or (lv[..., :6] > U + 1e-4).any():
        p.append(f"{label}: рука вне пределов кодека")
    if not np.isin(lv[..., 6], (-1.0, 1.0)).all():
        p.append(f"{label}: схват не ±1")
    return p


def check_block(plan, plan_path, path, label, task, block):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import k16_protocol as p16
    p = pr.validate_block(path, label, int(task), int(block),
                          require_single=True, setup=plan["setup"])
    if p:
        return p
    d, z = p16.load(path)
    if os.path.basename(os.path.normpath(str(d.get("ckpt")))) != \
            plan["bar_revision"]:
        p.append(f"{label}: снимок BAR блока не закреплённой ревизии")
    if label == "fullbar":
        return p
    j = d.get("joint") or {}
    want = dict(label=label, plan_sha1=sha(plan_path),
                prep_id=plan["prep_id"], bar_revision=plan["bar_revision"],
                ckpt_sha1=full_sha(plan["inputs"][f"ckpt_{label}"]),
                k21_policy_sha1=full_sha(os.path.join(HERE, "k21_policy.py")),
                harness_adapter_sha1=sha(os.path.join(HERE,
                                                      "k21_harness.py")))
    for k, v in want.items():
        if j.get(k) != v:
            p.append(f"{label}: {k} = {j.get(k)!r}, ожидалось {v!r}")
    p += journal_problems(label, z, plan["bounds"])
    if not p:
        p += hpc.actions_problems(label, z, hpc.harness_scale())
    return p


def check_run(plan, plan_path, run_dir, pairs=None):
    ok, gaps = {}, []
    pairs = pairs or [(t, b) for t in plan["tasks"] for b in plan["blocks"]]
    for t, b in pairs:
        for lab in plan["labels"]:
            pth = os.path.join(run_dir, f"{lab}_t{t}_i{b}.json")
            if not os.path.exists(pth):
                gaps.append(dict(label=lab, task=t, block=b, why="не готов"))
                continue
            prob = check_block(plan, plan_path, pth, lab, t, b)
            if prob:
                gaps.append(dict(label=lab, task=t, block=b,
                                 why="; ".join(prob[:2])))
            else:
                ok[(lab, t, b)] = pth
    return ok, gaps


def selftest():
    b = dict(L=[-1.0] * 6, U=[1.0] * 6)
    C = 3
    lv = np.zeros((C, 5, 8, 7), np.float32)
    lv[..., 6] = 1.0
    z = {"k15d_level_x": lv, "k15g_chunk": np.arange(1, C + 1),
         "actions": np.zeros((24, 5, 7), np.float32),
         "done_step": np.full(5, -1)}
    assert journal_problems("x", z, b) == []
    bad = dict(z, **{"k15d_level_x": lv * 0 + 2})
    assert journal_problems("x", bad, b)
    lv2 = lv.copy()
    lv2[..., 6] = 0.3
    assert journal_problems("x", dict(z, **{"k15d_level_x": lv2}), b)
    assert journal_problems("x", dict(z, k15g_chunk=np.array([1, 3, 2])), b)
    assert set(POLICY) == set(LABELS)
    print("самопроверка k21_protocol пройдена: журнал руки (пределы, схват "
          "±1, нумерация), метки и политики")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-21: протокол пилота")
    ap.add_argument("mode", choices=("plan", "check-plan", "check-block",
                                     "smoke-check", "selftest"))
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--block", nargs=4, default=None)
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    try:
        if a.mode == "plan":
            if os.path.exists(a.plan):
                raise SystemExit(f"{a.plan} уже зафиксирован")
            plan = make_plan(a.device)
            with open(a.plan + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=1)
            os.replace(a.plan + ".tmp", a.plan)
            print(f"  план {a.plan}: политик {len(plan['labels'])}, задач 10, "
                  f"блок {plan['blocks']}, процессов "
                  f"{len(plan['labels']) * 10 * len(plan['blocks'])}")
            return 0
        plan = json.load(open(a.plan))
        if a.mode == "check-plan":
            prob = check_plan_inputs(plan)
            print("\n".join(prob) if prob else "  входы плана сверены")
            return 3 if prob else 0
        if a.mode == "check-block":
            path, lab, t, b = a.block
            prob = check_block(plan, a.plan, path, lab, t, b)
            if prob:
                print("; ".join(prob[:6]))
            return 2 if prob else 0
        ok, gaps = check_run(plan, a.plan, a.run_dir,
                             [tuple(x) for x in plan["smoke"]])
        n = len(plan["smoke"]) * len(plan["labels"])
        print(f"  smoke: проверено блоков {len(ok)} из {n}; проблем "
              f"{len(gaps)}" + (f": {gaps[:4]}" if gaps else ""))
        return 3 if gaps else 0
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
