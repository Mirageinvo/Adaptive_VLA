#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-28, пакет A: одна сводка существующих провалов (без новых раскаток).

Исходы читаются только из проверенных блоков штатными проверками планов:
обязательные — parents24_s0/s1 и flat24_s0/s1 (план A K-21, K-22, повтор
K-23); необязательные — fullbar, fullbar_cached, parents_mr24 (K-24),
parents_cond24 (K-25), parents_attn24/flat_attn24 (K-26), parents_mem24
(K-27): при отсутствии полного проверенного набора — явная пометка.
Состояние — (набор, task_id, init_state_id, env seed 101); формулировка
задачи — из блоков fullbar плана A.

Выход reports/k28/failures/: outcomes.csv (150 строк, 0/1, пусто — нет
данных), summary.json, summary.md: провалы по задачам, общие провалы двух
сидов одной архитектуры, провалы всех доступных моделей, спасения/потери
дерева против flat и выбранные пары (общий провал / спасение первой /
спасение второй / общий успех). BAR — один фиксированный контроль.
Видео в блоках K-9h не сохраняются — причины провалов не размечаются.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT = "reports/k28/failures"
REQUIRED = ("parents24_s0", "parents24_s1", "flat24_s0", "flat24_s1")
PAIRS = [("parents24_s{s}", "flat24_s{s}"),
         ("parents24_s{s}", "parents_mr24_s{s}"),
         ("parents_attn24_s{s}", "flat_attn24_s{s}"),
         ("parents_mem24_s{s}", "parents24_s{s}"),
         ("parents24_s{s}", "fullbar")]


def _verified(name, fn):
    try:
        return fn(), None
    except (SystemExit, OSError, KeyError, ValueError) as e:
        return None, (str(e.code) if isinstance(e, SystemExit) else repr(e))


def _plan_blocks(proto, path):
    plan = json.load(open(path, encoding="utf-8"))
    prob = proto.check_plan_inputs(plan)
    if prob:
        raise SystemExit(f"{path}: " + "; ".join(prob[:2]))
    ok, gaps = proto.check_run(plan, path, proto.run_dir(path))
    want = len(plan["cells"]) * len(plan["labels"])
    if len(ok) != want or gaps:
        raise SystemExit(f"{path}: проверено {len(ok)}/{want}")
    return ok


def sources():
    """-> {имя источника: (блоки | None, причина)}."""
    import k23_protocol as p23
    import k24_protocol as p24
    import k25_protocol as p25
    import k26_protocol as p26
    out = {}
    out["k21_main_A"] = _verified("k21", lambda: p23.ref_k21_main()["blocks"])
    out["k22"] = _verified("k22", lambda: p23.ref_k22()["blocks"])
    out["k23_repeat_s1"] = _verified("k23", lambda: p24.ref_k23_repeat(1)
                                     ["blocks"])
    out["fullbar_cached"] = _verified("c", lambda: p23.ref_cached()["blocks"])
    for s in (0, 1):
        out[f"k24_s{s}"] = _verified("k24", lambda s=s: p25.ref_k24(s)
                                     ["blocks"])
        out[f"k25_s{s}"] = _verified("k25", lambda s=s: p26.ref_k25(s)
                                     ["blocks"])
        for mode in p26.MODES:
            out[f"k26_s{s}_{mode}"] = _verified("k26", lambda s=s, m=mode:
                                                _plan_blocks(
                                                    p26, p26.plan_path(s, m)))
        import k27_protocol as p27
        out[f"k27_s{s}"] = _verified("k27", lambda s=s: _plan_blocks(
            p27, p27.plan_path(s)))
    return out


def task_names(blocks):
    desc = {}
    for (lab, s, t, b), f in blocks.items():
        if lab == "fullbar":
            d = json.load(open(f, encoding="utf-8")).get("task_description")
            if desc.setdefault(int(t), d) != d:
                raise SystemExit(f"разные формулировки задачи {t}")
    return desc


def collect(src):
    import k23_analyze as a23
    outc, status = {}, {}
    for name, (blocks, why) in src.items():
        if blocks is None:
            status[name] = dict(status="не выполнено", why=why)
            continue
        o = a23.outcomes(blocks)
        for lab, v in o.items():
            if lab in outc and outc[lab] != v:
                raise SystemExit(f"{lab}: разные исходы в источниках")
            outc[lab] = v
        status[name] = dict(status="выполнено", labels=sorted(o))
    return outc, status


def arch_of(lab):
    return lab.rsplit("_s", 1)[0] if lab[-3:-1] == "_s" else lab


def analyze(outc, desc):
    import k23_analyze as a23
    miss = [x for x in REQUIRED if x not in outc]
    if miss:
        raise SystemExit(f"нет обязательных исходов: {miss}")
    states = a23.same_states(outc, list(REQUIRED), 150)
    labs = sorted(outc)
    for lab in labs:
        if sorted(outc[lab]) != states:
            raise SystemExit(f"состояния {lab} ≠ обязательным")
    fails = {lab: [k for k in states if not outc[lab][k]] for lab in labs}
    per_task = {lab: {} for lab in labs}
    for lab in labs:
        for k in fails[lab]:
            per_task[lab][k[0]] = per_task[lab].get(k[0], 0) + 1
    archs = {}
    for lab in labs:
        archs.setdefault(arch_of(lab), []).append(lab)
    both_seeds = {a: [list(k) for k in states if all(not outc[x][k]
                                                     for x in ls)]
                  for a, ls in archs.items() if len(ls) == 2}
    all_fail = [list(k) for k in states if all(not outc[x][k] for x in labs)]
    pairs = {}
    for a_, b_ in PAIRS:
        for s in (0, 1):
            A, B = a_.format(s=s), b_.format(s=s)
            if A not in outc or B not in outc or (A, B) in [
                    tuple(p.split(" vs ")) for p in pairs]:
                continue
            c = dict(both_fail=0, first_only=0, second_only=0, both_success=0)
            for k in states:
                x, y = outc[A][k], outc[B][k]
                key = "both_success" if x and y else "both_fail" if not (
                    x or y) else "first_only" if x else "second_only"
                c[key] += 1
            pairs[f"{A} vs {B}"] = c
    return dict(states=len(states), labels=labs,
                n_fail={lab: len(fails[lab]) for lab in labs},
                fail_per_task=per_task, task_names={str(k): v for k, v in
                                                    desc.items()},
                both_seeds_fail=both_seeds,
                all_models_fail=all_fail, pairs=pairs,
                note="частота провала — описательная характеристика этой "
                     "коллекции политик; BAR — один фиксированный контроль; "
                     "видео нет — причины не размечаются"), states, fails


def write(res, outc, states, desc, status, out=OUT):
    os.makedirs(out, exist_ok=True)
    labs = res["labels"]
    with open(os.path.join(out, "outcomes.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["suite", "task_id", "task", "init_state_id", "env_seed"] +
                   labs)
        for k in states:
            suite, t = k[0].split("/")
            w.writerow([suite, t, desc.get(int(t), ""), k[1], 101] +
                       [int(outc[lab][k]) if k in outc[lab] else ""
                        for lab in labs])
    json.dump(dict(res, sources=status), open(os.path.join(
        out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False,
        indent=1)
    md = ["# K-28: сводка существующих провалов", "",
          f"Состояний {res['states']} (LIBERO-10, старты 30–44, env seed "
          f"101). Источники: " + ", ".join(
              f"{k} — {v['status']}" for k, v in status.items()), "",
          "## Провалы по моделям", "", "| модель | провалов | по задачам |",
          "|---|---|---|"]
    for lab in labs:
        pt = res["fail_per_task"][lab]
        md.append(f"| {lab} | {res['n_fail'][lab]} | " + ", ".join(
            f"t{k.split('/')[1]}: {v}" for k, v in sorted(
                pt.items(), key=lambda kv: -kv[1])) + " |")
    md += ["", "## Общие провалы обоих сидов архитектуры", ""]
    for a, ks in res["both_seeds_fail"].items():
        md.append(f"- {a}: {len(ks)} — " + ", ".join(
            f"t{k[0].split('/')[1]}/{k[1]}" for k in ks))
    md += ["", f"## Провалены всеми доступными моделями ({len(labs)})", "",
           ", ".join(f"t{k[0].split('/')[1]}/{k[1]}"
                     for k in res["all_models_fail"]) or "нет", "",
           "## Пары (общий провал / только первая / только вторая / общий "
           "успех)", ""]
    for p, c in res["pairs"].items():
        md.append(f"- {p}: {c['both_fail']} / {c['first_only']} / "
                  f"{c['second_only']} / {c['both_success']}")
    md += ["", res["note"]]
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write(
        "\n".join(md) + "\n")
    return md


def selftest():
    import random
    rnd = random.Random(0)
    states = [(f"10/{t}", i) for t in range(10) for i in range(30, 45)]
    outc = {lab: {k: rnd.random() < 0.9 for k in states}
            for lab in REQUIRED + ("fullbar", "parents_mr24_s0")}
    # проверка здравого смысла как в K-24: 14 и 14 провалов, +8/−8 -> общих 6
    fp = states[:14]
    fm = states[:6] + states[14:22]
    outc["parents24_s0"] = {k: k not in fp for k in states}
    outc["parents_mr24_s0"] = {k: k not in fm for k in states}
    res, st, _f = analyze(outc, {t: f"задача {t}" for t in range(10)})
    c = res["pairs"]["parents24_s0 vs parents_mr24_s0"]
    assert c == dict(both_fail=6, first_only=8, second_only=8,
                     both_success=128), c
    assert res["n_fail"]["parents24_s0"] == 14
    assert "parents24" in res["both_seeds_fail"]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        write(res, outc, st, {t: f"задача {t}" for t in range(10)},
              dict(x=dict(status="выполнено")), out=td)
        rows = list(csv.reader(open(os.path.join(td, "outcomes.csv"))))
        assert len(rows) == 151 and rows[0][:5] == [
            "suite", "task_id", "task", "init_state_id", "env_seed"]
    try:
        analyze({k: v for k, v in outc.items() if k != "flat24_s1"}, {})
        raise AssertionError("неполный обязательный набор принят")
    except SystemExit:
        pass
    assert arch_of("parents_attn24_s1") == "parents_attn24" and \
        arch_of("fullbar") == "fullbar"
    print("самопроверка k28_failures пройдена: пары (6/8/8/128 как в K-24), "
          "провалы по задачам, общие провалы сидов, CSV 150 строк, отказ "
          "без обязательных моделей")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    try:
        src = sources()
        A = src["k21_main_A"][0]
        if A is None:
            raise SystemExit(f"план A K-21 не проверен: {src['k21_main_A'][1]}")
        desc = task_names(A)
        outc, status = collect(src)
        res, states, _f = analyze(outc, desc)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    md = write(res, outc, states, desc, status)
    print("\n".join(md))
    c = res["pairs"].get("parents24_s0 vs parents_mr24_s0")
    if c:
        print(f"\n  сверка с K-24: parents_s0 / MR_s0 — общих провалов "
              f"{c['both_fail']}, только MR успешен {c['second_only']}, только "
              f"parents успешен {c['first_only']} (в отчёте K-24: 6, +8/−8)")
    print(f"  {OUT}/outcomes.csv, summary.json, summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
