#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-36, этап C (§9): независимая окончательная проверка.

Только после passed_B_candidate (k36b_screen.gate_b_problems) и пройденных
предварительных проверок BAR (k36c_bar). До первого C-эпизода
закрепляется план reports/k36/plan_c.json: SHA обоих Tplus (final.pt и
отчёт продолжения), решения B и всех его файлов, очередь.
Объём: задачи 0/6/8 — 100 C-случаев, остальные семь — 20; руки R, T32,
Tplus × сиды 0/1 и общий BAR: 1800 + 840 + 440 = 3080 эпизодов. Очередь —
задачи в перестановке seed 36021; для каждой задачи блоки по 5 случаев
списка C; метки [R_s0, T32_s0, Tplus_s0, R_s1, T32_s1, Tplus_s1, BAR]
циклически сдвигаются на номер блока: 528 блоков R/T + 88 блоков BAR.
Исполнение — в порядке очереди кусками одного источника (сид или BAR):
один источник на GPU за раз, наружу — CPU-результаты. scopes confirm_c
(R/T32/Tplus) и bar_c (BAR). Промежуточные SR не печатаются.
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import k36_bank as kb36  # noqa: E402
import k36_harness as kh  # noqa: E402
import k36_protocol as pc  # noqa: E402
import k36b_gate as kg  # noqa: E402
import k36b_screen as kbs  # noqa: E402
import k36b_train as kbt  # noqa: E402
import k36c_bar as kbar  # noqa: E402

PLAN_C = "reports/k36/plan_c.json"
STAGE = "confirm_c"
kh.STAGE_SCOPE.setdefault(STAGE, "confirm_c")
ARMS_C = ("R_s0", "T32_s0", "Tplus_s0", "R_s1", "T32_s1", "Tplus_s1", "BAR")
TASK_SEED_C = 36021


def task_order_c():
    g = np.random.default_rng(TASK_SEED_C)
    return [int(t) for t in g.permutation(10)]


def cells_c(splits):
    out, k = [], 0
    for t in task_order_c():
        ids = splits[str(t)]["c_used"]
        for g in range(len(ids) // kh.pc.BLOCK):
            rot = list(ARMS_C[k % 7:]) + list(ARMS_C[:k % 7])
            for lab in rot:
                out.append((lab, t, g, list(ids[g * 5:(g + 1) * 5])))
            k += 1
    return out


def source_of(lab):
    return "BAR" if lab == "BAR" else pc.parse_arm(lab)[2]


def chunks(cells):
    out = []
    for c in cells:
        s = source_of(c[0])
        if out and out[-1][0] == s:
            out[-1][1].append(c)
        else:
            out.append((s, [c]))
    return out


CODE_BC = ("k36b_gate.py", "k36b_train.py", "k36b_screen.py", "k36c_bar.py",
           "k36c_confirm.py", "k36c_latency.py", "k36c_analyze.py",
           "k36bc_run.sh")


def bar_revision(prep):
    """Ревизия original BAR из prepare K-36; текущий снимок K-21 обязан
    совпадать."""
    import k21_data as d21
    want = prep["k32_plan"]["bar_revision"]
    cur = json.load(open(d21.REPORT, encoding="utf-8"))["bar_pin"]["revision"]
    if cur != want:
        raise SystemExit(f"ревизия BAR K-21 {cur} ≠ prepare K-36 {want}")
    return want


def make_plan_c(prep):
    gb = kbs.gate_b_problems()
    if gb:
        raise SystemExit("этап C не разрешён: " + "; ".join(gb[:3]))
    cells = cells_c(prep["splits"])
    tplus = {}
    for s in pc.parse_seeds():
        r = json.load(open(kbt.report_path(s), encoding="utf-8"))
        tplus[str(s)] = dict(actor_hash=r["actor_hash"],
                             final_sha1=pc.sha(kbt.final_path(s)),
                             report_sha1=pc.sha(kbt.report_path(s)))
    return json.loads(json.dumps(dict(
        kind="k36_plan_c", prepare_sha1=prep["sha1"],
        gate_b_sha1=pc.sha(kbs.GATE_B),
        gate_b_files=json.load(open(kbs.GATE_B))["files"],
        tplus=tplus, bar_revision=bar_revision(prep),
        code={f: pc.sha(os.path.join(HERE, f)) for f in CODE_BC},
        task_order=task_order_c(),
        cells_sha1=hashlib.sha1(json.dumps(cells).encode()).hexdigest(),
        n_cells=len(cells),
        n_blocks=dict(policy=sum(c[0] != "BAR" for c in cells),
                      bar=sum(c[0] == "BAR" for c in cells)),
        n_episodes=sum(len(c[3]) for c in cells))))


def plan_c_problems(prep):
    """Закреплённый план C = полному ожидаемому плану, построенному заново
    из текущих артефактов (оба Tplus, решение B и его файлы, ревизия BAR,
    SHA модулей B/C, очередь и объёмы)."""
    if not os.path.exists(PLAN_C):
        return ["нет плана C"]
    pl = json.load(open(PLAN_C, encoding="utf-8"))
    try:
        want = make_plan_c(prep)
    except SystemExit as e:
        return [f"ожидаемый план C не строится: {e.code}"]
    diff = sorted(k for k in set(pl) | set(want) if pl.get(k) != want.get(k))
    return [f"план C ≠ ожидаемому: {diff}"] if diff else []


def c_want(prep, plan, lab, t, cid, revision=None):
    sp = prep["splits"][str(t)]
    h = sp["init_hash"][cid]
    if lab == "BAR":
        return kbar.want_bar(prep, STAGE, t, cid, sp["pos"][cid], "bank",
                             revision, space=kb36.SPACE_CONF, case_id=cid,
                             want_hash=tuple(h))
    ah = None
    if lab.startswith("Tplus"):
        ah = plan["tplus"][str(pc.parse_arm(lab)[2])]["actor_hash"]
    return kh.want_of(prep, STAGE, lab, t, cid, sp["pos"][cid], "bank",
                      space=kb36.SPACE_CONF, case_id=cid, want_hash=tuple(h),
                      actor_hash=ah)


def block_path(lab, t, g):
    return os.path.join(pc.OUT, STAGE, lab, f"t{t}_b{g:02d}.json")


def block_files_problems(prep, plan, cell, revision):
    """Готовый блок: план C, точный набор файлов эпизодов и их SHA."""
    lab, t, g, cases = cell
    d = json.load(open(block_path(lab, t, g), encoding="utf-8"))
    if d.get("cases") != cases or d.get("plan_c_sha1") != pc.sha(PLAN_C):
        return ["блок другого плана"]
    want = set()
    for cid in cases:
        w = c_want(prep, plan, lab, t, cid, revision)
        want.update(kh.ep_paths(STAGE, w["label"], t, cid))
    if set(d.get("files") or {}) != want:
        return ["набор файлов блока ≠ ожидаемому"]
    for f, sh in d["files"].items():
        if not os.path.exists(f) or pc.sha(f) != sh:
            return [f"файл {f} изменён"]
    return []


def _chunk(torch, prep, plan, C, L, sim, src_key, chunk, device, revision,
           loader=None, bar_loader=None):
    """Область жизни одного источника (сид или BAR) -> число блоков."""
    n = 0
    qb = kbar.consts_bar(prep)

    def woe(eid):
        _st, lab, t, key = kg.parse_eid(eid)
        return c_want(prep, plan, lab, t, key, revision)

    vb = int(json.load(open(kbar.GATES["bar_smoke"], encoding="utf-8"))[
        "vocab"])                      # словарь BAR из пройденного smoke

    def validate(rec, z, w2):
        if w2["label"] == "BAR":
            return kbar.bar_episode_problems(rec, z, w2, qb, vb)
        return kh.episode_problems(rec, z, w2, C)
    if src_key == "BAR":
        b = (bar_loader or kbar.load_bar)(torch, device, prep)
        arm = kbar.BarArm(torch, b["bar"])
        q = kbar.consts_bar(prep)
        vocab = int(b["bar"]["codec"].vocab_size)
        for lab, t, g, cases in chunk:
            files = {}
            for cid in cases:
                w = c_want(prep, plan, lab, t, cid, b["revision"])
                kbar.run_bar_ep(w, sim, b["io"], arm, L, q, vocab,
                                allowed=kg.C_PENDING, woe=woe,
                                validate=validate)
                files.update(kh.ep_files(w))
            pc.save_json(block_path(lab, t, g), dict(
                label=lab, task=t, block=g, cases=cases, files=files,
                plan_c_sha1=pc.sha(PLAN_C)))
            n += 1
        return n
    v = kh.load_source(torch, src_key, device, loader)
    arms = {}
    for lab, t, g, cases in chunk:
        if lab not in arms:
            if lab.startswith("Tplus"):
                act = kbs.load_tplus(torch, prep, src_key, v["mod"].H) \
                    if "actors" not in v else v["actors"]["Tplus"]
                arms[lab] = kh.make_arm(torch, prep, f"T32_s{src_key}", v,
                                        actor=act)
            else:
                arms[lab] = kh.make_arm(torch, prep, lab, v)
        files = {}
        for cid in cases:
            w = c_want(prep, plan, lab, t, cid)
            if kh.load_ep(w) is None:          # дальше будет новый reserve
                kg.before_reserve(L, w, kg.C_PENDING, woe, validate)
            kh.run_ep(w, sim, v["io"], arms[lab], L, C)
            files.update(kh.ep_files(w))
        pc.save_json(block_path(lab, t, g), dict(
            label=lab, task=t, block=g, cases=cases, files=files,
            plan_c_sha1=pc.sha(PLAN_C)))
        n += 1
    return n


def confirm_c(a, sim=None):
    sim = sim or kb36.BankSim("bank", kb36.SPACE_CONF)
    import torch
    try:
        prep = pc.load_prepare()
        if not os.path.exists(PLAN_C):
            pc.save_json(PLAN_C, make_plan_c(prep))
        p = plan_c_problems(prep) + kbs.gate_b_problems()
        b = kbar.load_bar(torch, a.device, prep)
        rev, vocab = b["revision"], int(b["bar"]["codec"].vocab_size)
        del b
        kh.free(torch)
        p += kbar.preflight_problems(prep, rev, vocab)
        if p:
            raise SystemExit("этап C: " + "; ".join(p[:4]))
        pr = kh.resources_problems(torch, a.device)
        if pr:
            raise SystemExit("; ".join(pr))
        plan = json.load(open(PLAN_C, encoding="utf-8"))
        C = kh.consts(prep)
        L = pc.ledger()
        cells = cells_c(prep["splits"])
        todo, done0 = [], 0
        for c in cells:
            if os.path.exists(block_path(*c[:3])):
                pr = block_files_problems(prep, plan, c, rev)
                if pr:
                    raise SystemExit(f"{c[0]} t{c[1]} b{c[2]}: готовый блок "
                                     f"не сходится: {pr[:2]}")
                done0 += 1
            else:
                todo.append(c)
        t0, n_new = time.time(), 0
        for key, ch in chunks(todo):
            n_new += _chunk(torch, prep, plan, C, L, sim, key, ch, a.device,
                            rev)
            kh.free(torch)
            el = (time.time() - t0) / 3600
            left = len(cells) - done0 - n_new
            print(f"  C: готово блоков {done0 + n_new}/{len(cells)}; новых "
                  f"{n_new} за {el:.1f} ч, осталось {left} (~"
                  f"{el / max(n_new, 1) * left:.1f} ч); confirm_c "
                  f"{L.spent('confirm_c')}/{pc.CAPS['confirm_c']}, bar_c "
                  f"{L.spent('bar_c')}/{pc.CAPS['bar_c']}", flush=True)
        print(f"  этап C: все {len(cells)} блоков готовы")
        return 0
    finally:
        sim.close()


def selftest():
    S = {}
    for t in range(10):
        ids = [f"k33s1_t{t}_c{i:05d}" for i in range(300)]
        sp = kb36.split_ids(ids, t)
        S[str(t)] = dict(c_used=sp["C"] if t in kb36.ADAPTED else
                         sp["C"][:20])
    c = cells_c(S)
    rt = [x for x in c if x[0] != "BAR"]
    br = [x for x in c if x[0] == "BAR"]
    assert len(rt) == 528 and len(br) == 88 and len(c) == 616
    assert sum(len(x[3]) for x in rt) == 2640 and \
        sum(len(x[3]) for x in br) == 440
    assert [x[0] for x in c[:7]] == list(ARMS_C) and \
        [x[0] for x in c[7:14]] == list(ARMS_C[1:]) + list(ARMS_C[:1])
    for lab in ARMS_C:
        for t in range(10):
            got = [i for x in c if x[0] == lab and x[1] == t for i in x[3]]
            assert sorted(got) == sorted(S[str(t)]["c_used"])
    assert sorted(task_order_c()) == list(range(10))
    ch = chunks(c[:14])
    assert [k for k, _ in ch] == [0, 1, "BAR", 0, 1, "BAR", 0] and \
        sum(len(v) for _k, v in ch) == 14
    assert 1800 + 840 + 440 == 3080
    # план C: неполный (без Tplus, файлов B, объёмов) — отказ; полный — ок
    import tempfile
    g = globals()
    keep = dict(PLAN_C=g["PLAN_C"], make_plan_c=g["make_plan_c"])
    want = dict(kind="k36_plan_c", prepare_sha1="P", gate_b_sha1="G",
                gate_b_files={"f": "1"}, tplus={"0": dict(actor_hash="a"),
                                                "1": dict(actor_hash="b")},
                bar_revision="R", code={"k36c_bar.py": "x"},
                task_order=task_order_c(), cells_sha1="C", n_cells=616,
                n_blocks=dict(policy=528, bar=88), n_episodes=3080)
    with tempfile.TemporaryDirectory() as td:
        try:
            g["PLAN_C"] = os.path.join(td, "plan_c.json")
            g["make_plan_c"] = lambda prep: json.loads(json.dumps(want))
            pc.save_json(g["PLAN_C"], want)
            assert plan_c_problems(dict(sha1="P")) == []
            for k in ("tplus", "gate_b_files", "n_episodes", "n_cells",
                      "bar_revision", "code"):
                bad = {kk: v for kk, v in want.items() if kk != k}
                pc.save_json(g["PLAN_C"], bad)
                assert plan_c_problems(dict(sha1="P")), k
            pc.save_json(g["PLAN_C"], dict(want, tplus={"0": dict(
                actor_hash="a")}))
            assert plan_c_problems(dict(sha1="P"))
        finally:
            g.update(keep)
    print("самопроверка k36c_confirm пройдена: очередь C — 528 блоков R/T "
          "(2640 эпизодов) + 88 BAR (440), сдвиг 7 меток по блокам, все "
          "C-случаи каждой метки ровно по разу; куски по источнику; план C "
          "без Tplus/файлов B/объёмов/ревизии BAR/кода — отказ")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-36 C: подтверждение")
    ap.add_argument("cmd", choices=("confirm-c",))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    try:
        return confirm_c(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
