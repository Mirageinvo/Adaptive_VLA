#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-12i: отчёт по RL-smoke. Главное число — парная разность g_rl минус g0.

ПОЧЕМУ ИМЕННО ЭТА РАЗНОСТЬ. Сравнение с детерминированной D1 отвечает на
прикладной вопрос «лучше ли конечная политика исходной», но выигрыш над ней
может объясняться одним шумом: K-11g измерил, что шум сам по себе восстанавливает
часть провалов. Эффект ОБУЧЕНИЯ виден только в сравнении обученной гауссовой
головы с той же головой до обучения, при одном и том же потоке шума.

ЧТО ПРОВЕРЯЕТСЯ ДО СЧЁТА (и почему это отказы, а не замечания):
  * у g0 и g_rl совпадают соль шума и хэш первого сэмпла — иначе менялись и
    веса, и случайные числа, и разницу нельзя отнести к обучению;
  * пары сходятся по (задача, состояние) И по init_hash_full — иначе это разные
    начальные состояния;
  * оценочные состояния не пересекаются с обучающими — иначе улучшение
    заявлялось бы на тех же состояниях, на которых собран градиент;
  * обе руки сняты одной и той же sigma.
"""
import argparse
import glob
import json
import os
import sys


def load_cells(root, sub):
    out = []
    for p in sorted(glob.glob(os.path.join(root, sub, "*.json"))):
        c = json.load(open(p))
        c["_path"] = p
        out.append(c)
    return out


def episodes(cells):
    """{(задача, состояние): (успех, хэш)} с отказом на дублях."""
    out = {}
    for c in cells:
        for t in c["task_ids"]:
            for e in c["episodes"]:
                k = (int(t), int(e["state_id"]))
                if k in out:
                    raise SystemExit(f"{c['_path']}: эпизод {k} встречается "
                                     f"дважды")
                out[k] = (bool(e["success"]), str(e.get("init_hash_full") or ""))
    return out


def noise_key(cells):
    """Соль и хэш первого сэмпла — один на набор ячеек, иначе отказ."""
    keys = {(c.get("eps_mode"), c.get("eval_eps_seed"), c.get("eps_salt"))
            for c in cells}
    if len(keys) != 1:
        raise SystemExit(f"в наборе разные потоки шума: {sorted(keys)}")
    return keys.pop()


def paired(a, b, *, label_a, label_b, check_hash=True):
    """Парная разность a минус b по общим ключам. Непарность — отказ."""
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    if only_a or only_b:
        raise SystemExit(
            f"непарные эпизоды: только в {label_a} {only_a[:6]} "
            f"({len(only_a)}), только в {label_b} {only_b[:6]} "
            f"({len(only_b)}) — разность считалась бы по разным наборам")
    rec = los = n = 0
    bad_hash = []
    for k in sorted(a):
        sa, ha = a[k]
        sb, hb = b[k]
        if check_hash and (not ha or ha != hb):
            bad_hash.append((k, ha, hb))
            continue
        n += 1
        if sa and not sb:
            rec += 1
        elif sb and not sa:
            los += 1
    if bad_hash:
        raise SystemExit(
            f"хэши начального состояния различаются у {len(bad_hash)} пар, "
            f"например {bad_hash[:3]} — это разные состояния, а не пара")
    return dict(n=n, recovered=rec, lost=los,
                effect=((rec - los) / n) if n else None,
                discord=((rec + los) / n) if n else None,
                success_a=(sum(1 for k in a if a[k][0]) / len(a)) if a else None,
                success_b=(sum(1 for k in b if b[k][0]) / len(b)) if b else None)


def collect(root, ladder=None):
    det = load_cells(root, "eval_d1_det")
    g0 = load_cells(root, "eval_g0")
    if not det or not g0:
        raise SystemExit(f"в {root} нет оценочных ячеек d1_det или g0")
    ndet, ng0 = noise_key(det), noise_key(g0)
    if ng0[0] != "eval":
        raise SystemExit(f"g0 снята в режиме шума {ng0[0]!r}, а не 'eval': "
                         f"сравнивать с g_rl нельзя")
    sig = {float(c["sigma"]) for c in g0}
    e_det, e_g0 = episodes(det), episodes(g0)
    steps = {}
    pat = os.path.join(root, "eval_g_rl_step*")
    for d in sorted(glob.glob(pat)):
        k = int(os.path.basename(d).replace("eval_g_rl_step", ""))
        if ladder and k not in ladder:
            continue
        cells = load_cells(root, os.path.basename(d))
        if not cells:
            continue
        nk = noise_key(cells)
        if nk != ng0:
            raise SystemExit(
                f"шаг {k}: поток шума {nk} против {ng0} у g0 — вместе с весами "
                f"сменились случайные числа, и эффект обучения неотделим от "
                f"другого сэмпла")
        if {float(c["sigma"]) for c in cells} != sig:
            raise SystemExit(f"шаг {k}: другая sigma")
        steps[k] = episodes(cells)
    if not steps:
        raise SystemExit(f"в {root} нет ни одной оценки g_rl")
    # обучающие состояния не должны попасть в оценку
    tr = set()
    for d in sorted(glob.glob(os.path.join(root, "train_step*"))):
        for p in glob.glob(os.path.join(d, "*.pt")):
            b = os.path.basename(p)
            try:
                t = int(b.split("_")[0][1:])
                s0_ = int(b.split("_")[1][1:].split(".")[0])
            except (IndexError, ValueError):
                continue
            tr |= {(t, s0_ + i) for i in range(5)}
    leak = sorted(set(e_g0) & tr)
    if leak:
        raise SystemExit(
            f"оценочные состояния {leak[:8]} ({len(leak)}) встречаются среди "
            f"обучающих: улучшение заявлялось бы на тех же состояниях, на "
            f"которых собран градиент")
    return dict(det=e_det, g0=e_g0, steps=steps, sigma=sorted(sig),
                noise=dict(mode=ng0[0], eval_eps_seed=ng0[1], salt=ng0[2]),
                n_eval=len(e_g0), train_states=len(tr))


def report(data, head_tag=""):
    print(f"\n  RL-SMOKE{' ' + head_tag if head_tag else ''}: оценка на "
          f"{data['n_eval']} отложенных эпизодах, sigma {data['sigma']}, "
          f"поток шума {data['noise']}")
    print(f"    успех d1_det "
          f"{100 * sum(1 for k in data['det'] if data['det'][k][0]) / len(data['det']):.2f}%"
          f", успех g0 "
          f"{100 * sum(1 for k in data['g0'] if data['g0'][k][0]) / len(data['g0']):.2f}%")
    print(f"\n    {'шаг':>4}{'успех g_rl':>12}{'g_rl-g0':>10}{'восст':>7}"
          f"{'потер':>7}{'дискорд':>9}{'g_rl-d1_det':>13}")
    rows = {}
    for k in sorted(data["steps"]):
        e = data["steps"][k]
        vs_g0 = paired(e, data["g0"], label_a=f"g_rl шаг {k}", label_b="g0")
        vs_det = paired(e, data["det"], label_a=f"g_rl шаг {k}",
                        label_b="d1_det")
        rows[k] = dict(vs_g0=vs_g0, vs_det=vs_det)
        print(f"    {k:>4}{100 * vs_g0['success_a']:>11.2f}%"
              f"{100 * vs_g0['effect']:>+9.2f}{vs_g0['recovered']:>7}"
              f"{vs_g0['lost']:>7}{100 * vs_g0['discord']:>8.2f}%"
              f"{100 * vs_det['effect']:>+12.2f}")
    best = max(rows, key=lambda k: rows[k]["vs_g0"]["effect"])
    ok = rows[best]["vs_g0"]["effect"] > 0
    print(f"\n    лучший шаг по g_rl-g0: {best}, эффект "
          f"{100 * rows[best]['vs_g0']['effect']:+.2f} пп "
          f"({'сигнал есть' if ok else 'сигнала нет'})")
    if all(r["vs_g0"]["discord"] == 0 for r in rows.values()):
        print("    ВНИМАНИЕ: дискордантность ноль на всех шагах — действия на "
              "отложенных состояниях не изменились ни в одном эпизоде. Это "
              "«параметры поехали, поведение нет», а не отсутствие эффекта "
              "обучения.")
    return dict(rows=rows, best_step=best, signal=bool(ok))


def selftest():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="k12i_")

    def write(sub, arm, succ_fn, seed=777, sigma=0.10, hash_fn=None):
        d = os.path.join(tmp, sub)
        os.makedirs(d, exist_ok=True)
        for t in (0, 1):
            eps = []
            for i in range(30, 35):
                hs = (hash_fn or (lambda tt, ii: f"h{tt}_{ii}"))(t, i)
                eps.append(dict(state_id=i, init_hash_full=hs,
                                success=bool(succ_fn(t, i))))
            json.dump(dict(stage="diag", arm=arm, sigma=sigma,
                           eps_mode=("eval" if arm in ("g0", "g_rl")
                                     else "train"),
                           eval_eps_seed=(seed if arm in ("g0", "g_rl")
                                          else None),
                           eps_salt=(seed * 7 if arm in ("g0", "g_rl")
                                     else 11),
                           task_ids=[t], state_ids=list(range(30, 35)),
                           episodes=eps),
                      open(os.path.join(d, f"t{t}_s30.json"), "w"))

    write("eval_d1_det", "baseline", lambda t, i: (t + i) % 4 != 0)
    write("eval_g0", "g0", lambda t, i: (t + i) % 5 != 0)
    # шаг 1 восстанавливает один провал g0 и ничего не теряет
    write("eval_g_rl_step1", "g_rl", lambda t, i: (t + i) % 5 != 0 or i == 30)
    data = collect(tmp)
    res = report(data, "(тест)")
    assert res["signal"] is True, res
    r1 = res["rows"][1]["vs_g0"]
    assert r1["recovered"] >= 1 and r1["lost"] == 0, r1
    assert r1["n"] == 10, r1

    def _expect(fn, needle):
        try:
            fn()
        except SystemExit as e:
            assert needle in str(e), f"ожидал «{needle}», получил: {e}"
            return
        raise AssertionError(f"отказа «{needle}» не было")

    # ДРУГОЙ ПОТОК ШУМА — отказ
    write("eval_g_rl_step2", "g_rl", lambda t, i: True, seed=999)
    _expect(lambda: collect(tmp), "сменились случайные числа")
    import shutil
    shutil.rmtree(os.path.join(tmp, "eval_g_rl_step2"))

    # ДРУГИЕ ХЭШИ — не пара
    write("eval_g_rl_step2", "g_rl", lambda t, i: True,
          hash_fn=lambda t, i: "ДРУГОЙ")
    _expect(lambda: report(collect(tmp)), "это разные состояния")
    shutil.rmtree(os.path.join(tmp, "eval_g_rl_step2"))

    # ОБУЧАЮЩЕЕ СОСТОЯНИЕ В ОЦЕНКЕ — отказ
    d = os.path.join(tmp, "train_step0")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "t0_s30.pt"), "wb").write(b"x")
    _expect(lambda: collect(tmp), "собран градиент")
    os.remove(os.path.join(d, "t0_s30.pt"))
    open(os.path.join(d, "t0_s0.pt"), "wb").write(b"x")
    assert collect(tmp)["train_states"] == 5

    # НЕПАРНЫЕ ЭПИЗОДЫ — отказ
    a = {(0, 30): (True, "h"), (0, 31): (False, "h")}
    b = {(0, 30): (False, "h")}
    _expect(lambda: paired(a, b, label_a="a", label_b="b"), "непарные эпизоды")

    # g0 в режиме train — сравнивать нельзя
    shutil.rmtree(os.path.join(tmp, "eval_g0"))
    write("eval_g0", "baseline", lambda t, i: True)
    _expect(lambda: collect(tmp), "а не 'eval'")
    print("\nсамопроверка k12i_smoke_report пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--ladder", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not a.root:
        ap.error("нужен --root, например data/k12i/s0")
    lad = [int(x) for x in a.ladder.split(",") if x.strip()] or None
    data = collect(a.root, lad)
    res = report(data, os.path.basename(a.root.rstrip("/")))
    out = a.out or os.path.join(a.root, "smoke_report.json")
    json.dump(dict(root=a.root, sigma=data["sigma"], noise=data["noise"],
                   n_eval=data["n_eval"], best_step=res["best_step"],
                   signal=res["signal"],
                   rows={str(k): v for k, v in res["rows"].items()}),
              open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  сохранено: {out}")
    return 0 if res["signal"] else 2


if __name__ == "__main__":
    sys.exit(main() or 0)
