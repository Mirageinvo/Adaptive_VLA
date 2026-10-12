#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15g: SR всех фиксированных кандидатов на ВСЕХ состояниях плана.

ТОЛЬКО ЧИТАЕТ: снимок M1-cold, план K-15g, каталог раскаток (только
проверенные блоки — загрузчик k15g_compare_m1). Состояния — все стартовые
состояния блоков плана (задачи 2, 3, 8, 9 × блоки 0 и 5 = 40), а не только
14 провалов и 8 диагностических успехов.

ДЛЯ КАЖДОГО КАНДИДАТА (32 импульса K-15g и 16 постоянных рук M1-cold):
  SR по задачам и в целом, парная ΔSR против q0 того же состояния,
  выигрыши (q0 провал -> успех) и потери (q0 успех -> провал), число
  состояний, где импульс не был исполнен (q0ref завершился раньше чанка).
  Точный двусторонний тест Мак-Немара по парам — РАЗВЕДОЧНО: кандидатов 48,
  поправки на множественность нет; кандидат, выделенный после просмотра,
  требует проверки на новых стартах.

РАВНОМЕРНЫЙ ВЫБОР: средний SR, если направление выбирается равномерно и
независимо от исхода — по моменту и семейству, по семейству, по всем 32.

ПРОВЕРКА СУММ: по задачам сходится с итогом; спасения на 14 провалах и
повреждения на 8 диагностических пересчитываются из тех же данных и
сверяются с compare_m1.json, если он есть.

ПЕРЕСТАНОВОЧНЫЕ ТЕСТЫ (согласованность исходов РАЗНЫХ вмешательств в одном
состоянии; перестановка внутри состояния сохраняет его долю успеха). Это
не проверка воспроизводимости одного вмешательства: между чанками меняются
состояние и вектор поправки, у знаков ±e_j направленная коррекция может
давать противоположные исходы, learned и контроль — разные направления.
Отсутствие значимости НЕ означает, что направление неинформативно.
"""
import argparse
import collections
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

H = 8
MOMENTS = (2, 4)
DIRS = [f"{j}{s}" for j in range(4) for s in ("p", "m")]
FAMS = dict(l="learned", r="control")


def reached(done_step, moment):
    return bool(done_step < 0 or done_step >= H * (moment - 1))


def mcnemar_p(gain, loss):
    """Точный двусторонний тест Мак-Немара (биномиальный по расхождениям)."""
    n = gain + loss
    if n == 0:
        return 1.0
    k = min(gain, loss)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def candidate_row(outc, q0, states, moment=None, ref_done=None):
    """outc {(t, s): успех}, q0 {(t, s): успех}. Строка таблицы."""
    per_task = collections.defaultdict(lambda: [0, 0])
    gain = loss = missing = unreached = 0
    succ = 0
    for k in states:
        if k not in outc:
            missing += 1
            continue
        o, b = bool(outc[k]), bool(q0[k])
        per_task[k[0]][0] += o
        per_task[k[0]][1] += 1
        succ += o
        gain += (o and not b)
        loss += (b and not o)
        if moment is not None and ref_done is not None and \
                not reached(int(ref_done[k]), moment):
            unreached += 1
    n = len(states) - missing
    base = sum(bool(q0[k]) for k in states if k in outc)
    return dict(n=n, successes=succ, sr=succ / max(n, 1),
                q0_sr=base / max(n, 1), d_sr=(succ - base) / max(n, 1),
                gain=gain, loss=loss, mcnemar_p=mcnemar_p(gain, loss),
                per_task={str(t): v for t, v in sorted(per_task.items())},
                missing=missing, pulse_unreached=unreached)


def build(pulse, const, q0, ref_done, states, fails, diag):
    """pulse {метка K-15g: {(t, s): успех}}, const {метка M1: ...}."""
    rows = {}
    for lab, o in pulse.items():
        rows[lab] = dict(kind="impulse", moment=int(lab[1]),
                         family=FAMS[lab[2]], direction=lab[3:],
                         **candidate_row(o, q0, states, int(lab[1]),
                                         ref_done))
    for lab, o in const.items():
        rows[lab] = dict(kind="constant", moment=None, family=FAMS[lab[0]],
                         direction=lab[1:], **candidate_row(o, q0, states))
    q0row = candidate_row(q0, q0, states)

    def uniform(labs):
        rs = [rows[x] for x in labs if x in rows]
        if not rs:
            return None
        return dict(arms=len(rs), mean_sr=float(np.mean([r["sr"]
                                                         for r in rs])),
                    mean_d_sr=float(np.mean([r["d_sr"] for r in rs])),
                    min_sr=min(r["sr"] for r in rs),
                    max_sr=max(r["sr"] for r in rs),
                    q0_sr=q0row["sr"])
    uni = {}
    for m in MOMENTS:
        for f, name in FAMS.items():
            uni[f"impulse ч{m} {name}"] = uniform(
                [f"p{m}{f}{d}" for d in DIRS])
    for f, name in FAMS.items():
        uni[f"impulse ч2+ч4 {name}"] = uniform(
            [f"p{m}{f}{d}" for m in MOMENTS for d in DIRS])
        uni[f"constant {name}"] = uniform([f"{f}{d}" for d in DIRS])
    uni["impulse все 32"] = uniform(list(pulse))
    # сверка сумм: спасения на провалах и повреждения на диагностике
    fs, ds = set(fails), set(diag)
    cross = {}
    for kind, labs in (("impulse", list(pulse)), ("constant", list(const))):
        src = pulse if kind == "impulse" else const
        for f, name in FAMS.items():
            for m in (MOMENTS if kind == "impulse" else (None,)):
                ls = [x for x in labs if FAMS[x[2] if kind == "impulse"
                                              else x[0]] == name
                      and (m is None or int(x[1]) == m)]
                cross[f"{kind}{'' if m is None else f' ч{m}'} {name}"] = dict(
                    rescues=sum(bool(src[x].get(k)) for x in ls for k in fs
                                if k in src[x]),
                    harms=sum(not src[x][k] for x in ls for k in ds
                              if k in src[x]))
    # проверка: по задачам сходится с итогом
    bad = []
    for lab, r in list(rows.items()) + [("q0", q0row)]:
        if sum(v[0] for v in r["per_task"].values()) != r["successes"] or \
                sum(v[1] for v in r["per_task"].values()) != r["n"]:
            bad.append(lab)
    return dict(states=len(states), q0=q0row, rows=rows, uniform=uni,
                fail_diag_cross=cross, sum_check_failed=bad)


# --- перестановочные тесты ---------------------------------------------------
def perm_agreement(pairs_by_state, n_perm=20000, seed=0):
    """pairs_by_state: [(a [n], b [n])] — исходы двух вариантов по
    направлениям в каждом состоянии. Совпадения, ожидание при
    независимости (перестановка b внутри состояния), p(≥)."""
    rng = np.random.default_rng(seed)
    obs = 0
    perm = np.zeros(n_perm)
    for a, b in pairs_by_state:
        a, b = np.asarray(a), np.asarray(b)
        obs += int((a == b).sum())
        idx = np.argsort(rng.random((n_perm, len(b))), axis=1)
        perm += (a[None] == b[idx]).sum(1)
    return dict(agree=obs, expected=float(perm.mean()),
                p_ge=float((perm >= obs).mean()))


def permutation_tests(pulse, states_sel, n_perm=20000):
    out = {}
    fam_dirs = [(f, d) for f in FAMS for d in DIRS]
    specs = {
        "чанк 2 ↔ чанк 4 (та же метка направления)":
            lambda k: ([pulse[f"p2{f}{d}"][k] for f, d in fam_dirs],
                       [pulse[f"p4{f}{d}"][k] for f, d in fam_dirs]),
        "+e_j ↔ −e_j":
            lambda k: ([pulse[f"p{m}{f}{j}p"][k] for m in MOMENTS
                        for f in FAMS for j in range(4)],
                       [pulse[f"p{m}{f}{j}m"][k] for m in MOMENTS
                        for f in FAMS for j in range(4)]),
        "learned ↔ control (та же метка)":
            lambda k: ([pulse[f"p{m}l{d}"][k] for m in MOMENTS
                        for d in DIRS],
                       [pulse[f"p{m}r{d}"][k] for m in MOMENTS
                        for d in DIRS])}
    for name, sel in states_sel.items():
        for tname, fn in specs.items():
            try:
                pairs = [fn(k) for k in sel]
            except KeyError:
                out[f"{name}: {tname}"] = None
                continue
            out[f"{name}: {tname}"] = perm_agreement(pairs, n_perm)
    return out


# --- загрузка и отчёт --------------------------------------------------------
def run(snapshot, plan_path, run_dir):
    import k15f_m1cold_snapshot as msn
    import k15g_compare_m1 as cm
    import k15g_local_probe as pr
    snap = msn.verify(snapshot)
    plan = json.load(open(plan_path))
    if plan.get("kind") != pr.KIND or plan.get("smoke"):
        raise SystemExit("нужен основной план K-15g")
    prob = pr.check_plan_inputs(plan)
    if prob:
        raise SystemExit("входы плана K-15g не совпадают: " + "; ".join(
            prob[:4]))
    pulse_files, ref_files, gaps = cm.load_pulse(plan, run_dir)
    states = sorted((int(t), s) for t, bl in plan["blocks"].items()
                    for b in bl for s in range(int(b), int(b) + 5))
    ref = cm.episodes_of(ref_files)
    q0 = {k: v["success"] for k, v in ref.items()}
    ref_done = {k: v["done_step"] for k, v in ref.items()}
    missing_q0 = [k for k in states if k not in q0]
    if missing_q0:
        raise SystemExit(f"нет проверенного q0ref для {missing_q0[:5]}")
    pulse = {lab: {k: v["success"] for k, v in cm.episodes_of(fl).items()}
             for lab, fl in pulse_files.items()}
    tasks = set(plan["tasks"])
    const = {}
    for lab, fl in cm.load_const(snap).items():
        ep = cm.episodes_of(fl)
        const[lab] = {k: v["success"] for k, v in ep.items()
                      if k[0] in tasks and k in set(states)}
    fails = [tuple(x) for x in plan["fails"]]
    diag = [tuple(x) for x in plan["diag"]]
    res = build(pulse, const, q0, ref_done, states, fails, diag)
    q0_fail = [k for k in states if not q0[k]]
    q0_succ = [k for k in states if q0[k]]
    res["permutation"] = permutation_tests(pulse, {
        "провалы q0": q0_fail, "успехи q0 (все)": q0_succ})
    res["gaps"] = gaps
    res["note"] = ("кандидат, выделенный после просмотра 32+16, требует "
                   "проверки на новых стартах; тесты Мак-Немара разведочные, "
                   "без поправки на множественность")
    cmp_path = os.path.join(os.path.dirname(plan_path), "compare_m1.json")
    if os.path.exists(cmp_path):
        res["cross_check"] = cross_check(res, json.load(open(cmp_path)))
    return res


def cross_check(res, cmp_):
    """Спасения/повреждения из этих данных против compare_m1.json."""
    out, bad = {}, []
    for m in MOMENTS:
        for name in FAMS.values():
            t = cmp_["moments"][str(m)][name]
            pr_ = t["rescue"]["both"] + t["rescue"]["pulse_only"]
            ph = t["harm_all"]["both"] + t["harm_all"]["pulse_only"]
            mine = res["fail_diag_cross"][f"impulse ч{m} {name}"]
            out[f"ч{m} {name}"] = dict(compare=[pr_, ph],
                                       here=[mine["rescues"],
                                             mine["harms"]])
            if [pr_, ph] != [mine["rescues"], mine["harms"]]:
                bad.append(f"ч{m} {name}")
    t2 = cmp_["moments"]["2"]
    for name in FAMS.values():
        cr = t2[name]["rescue"]["both"] + t2[name]["rescue"]["const_only"]
        ch = t2[name]["harm_all"]["both"] + t2[name]["harm_all"][
            "const_only"]
        mine = res["fail_diag_cross"][f"constant {name}"]
        out[f"constant {name}"] = dict(compare=[cr, ch],
                                       here=[mine["rescues"],
                                             mine["harms"]])
        if [cr, ch] != [mine["rescues"], mine["harms"]]:
            bad.append(f"constant {name}")
    return dict(table=out, mismatches=bad)


def report(res):
    q = res["q0"]
    print(f"  состояний {res['states']}; q0: {q['successes']}/{q['n']} = "
          f"{100 * q['sr']:.1f}%; по задачам " + ", ".join(
              f"{t}: {v[0]}/{v[1]}" for t, v in q["per_task"].items()))
    if res["gaps"]:
        print(f"  ВНИМАНИЕ: непроверенных блоков {len(res['gaps'])}")
    print(f"  проверка сумм по задачам: "
          f"{'сходится' if not res['sum_check_failed'] else res['sum_check_failed']}")
    cc = res.get("cross_check")
    if cc:
        print(f"  сверка с compare_m1.json (спасения на 14 / вред на 8): "
              f"{'совпадает' if not cc['mismatches'] else cc['mismatches']}; "
              + ", ".join(f"{k} {v['here'][0]}/{v['here'][1]}"
                          for k, v in cc["table"].items()))
    print("  кандидаты по ΔSR (SR, ΔSR, выигрыш/потеря, Мак-Немар p "
          "разведочно, по задачам 2/3/8/9):")
    rows = sorted(res["rows"].items(), key=lambda kv: -kv[1]["d_sr"])
    for lab, r in rows:
        pt = " ".join(f"{v[0]}/{v[1]}" for v in r["per_task"].values())
        extra = (f", импульс не исполнен в {r['pulse_unreached']}"
                 if r["pulse_unreached"] else "")
        print(f"    {lab:6} {r['kind']:8} SR {100 * r['sr']:5.1f}% "
              f"ΔSR {100 * r['d_sr']:+5.1f} п.п., +{r['gain']}/−{r['loss']}, "
              f"p {r['mcnemar_p']:.3f}; {pt}{extra}")
    print("  равномерный выбор направления (независимо от исхода):")
    for k, v in res["uniform"].items():
        if v:
            print(f"    {k}: средний SR {100 * v['mean_sr']:.1f}% (ΔSR "
                  f"{100 * v['mean_d_sr']:+.1f} п.п.; диапазон "
                  f"{100 * v['min_sr']:.1f}–{100 * v['max_sr']:.1f}%), q0 "
                  f"{100 * v['q0_sr']:.1f}%")
    print("  перестановочные тесты (согласованность исходов РАЗНЫХ "
          "вмешательств; незначимость не означает неинформативность):")
    for k, v in res["permutation"].items():
        if v:
            print(f"    {k}: совпадений {v['agree']}, при независимости "
                  f"{v['expected']:.1f}, p(≥) = {v['p_ge']:.3f}")
    print(f"  {res['note']}")


def selftest():
    states = [(8, s) for s in range(10)]
    q0 = {k: k[1] in (5, 9) for k in states}
    ref_done = {k: (30 if k[1] == 9 else -1) for k in states}
    pulse = {f"p{m}{f}{d}": dict(q0) for m in MOMENTS for f in "lr"
             for d in DIRS}
    pulse["p4r3p"].update({(8, 0): True, (8, 1): True})   # +2
    pulse["p2l0p"].update({(8, 5): False})                 # −1
    const = {f"{f}{d}": dict(q0) for f in "lr" for d in DIRS}
    const["l0p"].pop((8, 3))                               # пропуск
    fails = [(8, s) for s in range(10) if s not in (5, 9)]
    diag = [(8, 5), (8, 9)]
    res = build(pulse, const, q0, ref_done, states, fails, diag)
    r = res["rows"]["p4r3p"]
    assert (r["successes"], r["gain"], r["loss"]) == (4, 2, 0)
    assert abs(r["d_sr"] - 0.2) < 1e-12 and r["per_task"]["8"] == [4, 10]
    # q0ref (8/9) завершился на шаге 30 — до чанка 4 (шаг 24 — нет: 30 >=
    # 24, дошёл); до чанка 2 — тоже дошёл
    assert r["pulse_unreached"] == 0
    r2 = res["rows"]["p2l0p"]
    assert (r2["gain"], r2["loss"]) == (0, 1)
    rl = res["rows"]["l0p"]
    assert rl["missing"] == 1 and rl["n"] == 9
    assert res["sum_check_failed"] == []
    u = res["uniform"]["impulse ч4 control"]
    assert abs(u["mean_sr"] - (0.2 * 7 + 0.4) / 8) < 1e-12
    cr = res["fail_diag_cross"]
    assert cr["impulse ч4 control"]["rescues"] == 2
    assert cr["impulse ч2 learned"]["harms"] == 1
    assert cr["constant learned"]["rescues"] == 0
    # недостигнутый импульс считается
    rd = {**ref_done, (8, 9): 10}
    assert build(pulse, const, q0, rd, states, fails, diag)["rows"][
        "p4l0p"]["pulse_unreached"] == 1
    assert mcnemar_p(0, 0) == 1.0 and abs(mcnemar_p(5, 0) - 2 / 32) < 1e-12
    assert abs(mcnemar_p(3, 3) - 1.0) < 1e-12
    # перестановочный тест: полная согласованность — малое p; одинаковые
    # исходы без вариации — p = 1
    pairs = [([1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0])] * 6
    t = perm_agreement(pairs, n_perm=4000)
    assert t["agree"] == 48 and t["p_ge"] < 0.01
    t = perm_agreement([([1] * 8, [1] * 8)], n_perm=500)
    assert t["p_ge"] == 1.0
    print("самопроверка k15g_full_table пройдена: SR, ΔSR, выигрыш/потеря, "
          "пропуск, недостигнутый импульс, равномерный выбор, суммы, "
          "Мак-Немар, перестановки")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-15g: полная таблица SR")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--snapshot", default="reports/k15f/m1cold_snapshot.json")
    ap.add_argument("--plan", default="reports/k15g/local_plan.json")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default="reports/k15g/full_table.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    rd = a.run_dir
    if rd is None:
        import glob
        cand = sorted(glob.glob("reports/k15g/local/run/p*/"))
        if len(cand) != 1:
            raise SystemExit(f"укажите --run-dir: найдено {cand}")
        rd = cand[0]
    res = run(a.snapshot, a.plan, rd)
    report(res)
    with open(a.out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1,
                  default=lambda x: [list(k) for k in x] if isinstance(
                      x, (set, tuple)) else str(x))
    os.replace(a.out + ".tmp", a.out)
    print(f"  сводка: {a.out}")
    cc = res.get("cross_check") or {}
    return 3 if (res["sum_check_failed"] or cc.get("mismatches")) else 0


if __name__ == "__main__":
    sys.exit(main())
