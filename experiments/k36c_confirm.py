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
    return dict(kind="k36_plan_c", prepare_sha1=prep["sha1"],
                gate_b_sha1=pc.sha(kbs.GATE_B),
                gate_b_files=json.load(open(kbs.GATE_B))["files"],
                tplus=tplus, task_order=task_order_c(),
                cells_sha1=hashlib.sha1(json.dumps(cells).encode())
                .hexdigest(), n_cells=len(cells),
                n_episodes=sum(len(c[3]) for c in cells))


def plan_c_problems(prep):
    if not os.path.exists(PLAN_C):
        return ["нет плана C"]
    pl = json.load(open(PLAN_C, encoding="utf-8"))
    p = []
    if pl.get("prepare_sha1") != prep["sha1"]:
        p.append("план C другого prepare")
    if not os.path.exists(kbs.GATE_B) or pc.sha(kbs.GATE_B) != \
            pl.get("gate_b_sha1"):
        p.append("решение B изменено после плана C")
    for f, s in (pl.get("gate_b_files") or {}).items():
        if not os.path.exists(f) or pc.sha(f) != s:
            p.append("файлы B изменены после плана C")
            break
    for s, v in (pl.get("tplus") or {}).items():
        if pc.sha(kbt.final_path(int(s))) != v["final_sha1"]:
            p.append(f"Tplus_s{s} изменён после плана C")
    cells = cells_c(prep["splits"])
    if hashlib.sha1(json.dumps(cells).encode()).hexdigest() != \
            pl.get("cells_sha1"):
        p.append("очередь C ≠ плану")
    return p


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
    if src_key == "BAR":
        b = (bar_loader or kbar.load_bar)(torch, device)
        arm = kbar.BarArm(torch, b["bar"])
        q = kbar.consts_bar(prep)
        vocab = int(b["bar"]["codec"].vocab_size)
        for lab, t, g, cases in chunk:
            files = {}
            for cid in cases:
                w = c_want(prep, plan, lab, t, cid, b["revision"])
                kbar.run_bar_ep(w, sim, b["io"], arm, L, q, vocab)
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
        b = kbar.load_bar(torch, a.device)
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
    print("самопроверка k36c_confirm пройдена: очередь C — 528 блоков R/T "
          "(2640 эпизодов) + 88 BAR (440), сдвиг 7 меток по блокам, все "
          "C-случаи каждой метки ровно по разу; куски по источнику")
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
