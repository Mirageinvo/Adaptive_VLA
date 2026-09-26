#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14j: парное сравнение АРХИТЕКТУР §49 по построчным дампам val_sel.

ЗАЧЕМ ОТДЕЛЬНЫЙ СКРИПТ, А НЕ РЕЖИМ K-14g. K-14g сравнивает ГОЛОВЫ поверх
ФИКСИРОВАННОГО кэша h18: все сравниваемые варианты читают одни и те же
скрытые состояния. Архитектуры §49 меняют вычисление МЕЖДУ слоем 12 и слоем
18, то есть у каждой свой h18, и кэш для них недействителен по построению.
Сравнивать их можно только по тому, что каждая получила на живом проходе, —
по построчным дампам, которые K-14c снимает на весах выбранной эпохи.

ЧТО СЧИТАЕТСЯ. Для набора строк S

    RMS(S) = sqrt( sum_S se / sum_S n )

то есть корень из среднего по ЭЛЕМЕНТАМ, а не среднее построчных корней.
Разность

    Δ = RMS(baseline) − RMS(кандидат),     положительная = КАНДИДАТ ЛУЧШЕ

и её интервал строятся парным кластерным бутстрапом ПО ЭПИЗОДАМ: в каждой
реплике пересэмплируются одни и те же эпизоды для всех архитектур, и разность
считается ВНУТРИ реплики. Непересечение двух отдельных интервалов проверкой
не является: ошибки сильно коррелированы, потому что считаются на одних
строках.

Сама процедура берётся из K-14c (`cluster_boot`) без переписывания: интервалы
§48 посчитаны ею же, и вторая реализация той же формулы означала бы два
немного разных ответа на один вопрос.

ПРАВИЛО §49, ЗАПИСАННОЕ ДО ДАННЫХ: кандидат допускается к подтверждающей
половине, только если НИЖНЯЯ граница 90% интервала Δ СТРОГО выше нуля. Не
прошёл ни один — не выбираем никого, и val_confirm не открывается.
"""
import argparse
import json
import os
import sys

import numpy as np


def load_dump(path):
    """Один дамп K-14c: массивы плюс разобранный meta."""
    if not os.path.exists(path):
        raise SystemExit(f"нет {path}")
    with np.load(path, allow_pickle=True) as z:
        need = ("rows", "episode", "se", "n", "meta")
        miss = [k for k in need if k not in z.files]
        if miss:
            raise SystemExit(f"{path}: нет массивов {miss}")
        d = dict(rows=np.asarray(z["rows"], np.int64),
                 episode=np.asarray(z["episode"], np.int64),
                 se=np.asarray(z["se"], float),
                 n=np.asarray(z["n"], np.int64),
                 meta=json.loads(str(z["meta"])))
    m = d["meta"]
    if m.get("kind") != "k14c_rows_val_sel":
        raise SystemExit(f"{path}: это {m.get('kind')}, а не построчный дамп")
    if m.get("part") != "val_sel":
        raise SystemExit(f"{path}: часть {m.get('part')}, ожидалась val_sel")
    for k in ("architecture", "seed", "selected_epoch", "selected_state_sha1",
              "plan_sha1", "q1_cache_sha1"):
        if m.get(k) is None:
            raise SystemExit(f"{path}: в meta нет {k}")
    if len(d["rows"]) != len(d["se"]) or len(d["rows"]) != len(d["n"]) \
            or len(d["rows"]) != len(d["episode"]):
        raise SystemExit(f"{path}: длины массивов не совпадают")
    if len(np.unique(d["rows"])) != len(d["rows"]):
        raise SystemExit(f"{path}: номера строк повторяются")
    if not np.isfinite(d["se"]).all():
        raise SystemExit(f"{path}: в ошибках есть nan или inf")
    if (d["se"] < 0).any():
        raise SystemExit(f"{path}: отрицательная сумма квадратов")
    # ПОРЯДОК СТРОК ПРИВОДИТСЯ К ОДНОМУ. Дампы снимаются по плану батчей, и
    # порядок в них не обязан совпадать; сопоставление идёт по номеру строки.
    o = np.argsort(d["rows"])
    for k in ("rows", "episode", "se", "n"):
        d[k] = d[k][o]
    d["path"] = path
    return d


def check_comparable(dumps, *, allow_smoke=False):
    """Дампы описывают ОДНУ И ТУ ЖЕ задачу на ОДНИХ И ТЕХ ЖЕ строках.

    Каждая проверка ловит свой вид несравнимости, и ни одна не выводится из
    остальных: совпадение состава строк не гарантирует, что у строк те же
    эпизоды; совпадение эпизодов не гарантирует того же плана; совпадение
    плана не гарантирует, что дамп снят не со смоука.
    """
    base = dumps[0]
    for d in dumps:
        m = d["meta"]
        if not allow_smoke:
            if m.get("smoke"):
                raise SystemExit(
                    f"{d['path']}: дамп снят в режиме smoke, данные урезаны; "
                    f"для выбора архитектуры он непригоден")
            if not m.get("selection_only"):
                raise SystemExit(
                    f"{d['path']}: дамп снят не в режиме отбора "
                    f"(selection_only={m.get('selection_only')!r})")
        if not np.array_equal(d["rows"], base["rows"]):
            raise SystemExit(
                f"{d['path']}: состав строк отличается от {base['path']} "
                f"({len(d['rows'])} против {len(base['rows'])})")
        if not np.array_equal(d["episode"], base["episode"]):
            raise SystemExit(
                f"{d['path']}: у строк другие эпизоды, чем в {base['path']}: "
                f"бутстрап кластеризовал бы разные разбиения")
        if not np.array_equal(d["n"], base["n"]):
            raise SystemExit(f"{d['path']}: другое число элементов в строке")
        for k in ("plan_sha1", "q1_cache_sha1"):
            if str(m.get(k)) != str(base["meta"].get(k)):
                raise SystemExit(
                    f"{d['path']}: {k} = {m.get(k)}, у {base['path']} "
                    f"{base['meta'].get(k)}: это разные задачи")
        q0a, q0b = (m.get("q0_prov") or {}), (base["meta"].get("q0_prov") or {})
        for k in ("q0_npz_sha1", "q0_manifest_sha1", "gate_r_sha1"):
            if str(q0a.get(k)) != str(q0b.get(k)):
                raise SystemExit(
                    f"{d['path']}: черновик отличается по {k}")
    nn = np.unique(base["n"])
    if len(nn) != 1:
        raise SystemExit(f"число элементов в строке не постоянно: {nn[:5]}")
    return int(nn[0])


def average_seeds(dumps):
    """Усреднение построчных ошибок между сидами, по правилу §49.1.

    Усредняются СУММЫ КВАДРАТОВ по строке, а не построчные RMS: величина,
    которую потом складывает бутстрап, — сумма, и среднее корней ей не
    соответствует.

    ТРЕБУЕТСЯ ОДИН И ТОТ ЖЕ НАБОР СИДОВ У ВСЕХ АРХИТЕКТУР. Иначе одна
    сравнивалась бы усреднённой по двум порядкам данных, другая — одним, и
    разность включала бы разницу в объёме усреднения.
    """
    by_arch = {}
    for d in dumps:
        by_arch.setdefault(str(d["meta"]["architecture"]), []).append(d)
    seed_sets = {a: tuple(sorted(int(x["meta"]["seed"]) for x in v))
                 for a, v in by_arch.items()}
    uniq = set(seed_sets.values())
    if len(uniq) != 1:
        raise SystemExit(f"наборы сидов различаются по архитектурам: "
                         f"{seed_sets}")
    for a, v in by_arch.items():
        if len(set(int(x["meta"]["seed"]) for x in v)) != len(v):
            raise SystemExit(f"{a}: сид повторяется")
    out = []
    for a, v in sorted(by_arch.items()):
        se = np.mean(np.stack([x["se"] for x in v]), axis=0)
        m = dict(v[0]["meta"])
        m["seed"] = list(seed_sets[a])
        m["averaged_over_seeds"] = len(v)
        m["selected_epoch"] = [int(x["meta"]["selected_epoch"]) for x in v]
        m["selected_state_sha1"] = [x["meta"]["selected_state_sha1"]
                                    for x in v]
        out.append(dict(rows=v[0]["rows"], episode=v[0]["episode"], se=se,
                        n=v[0]["n"], meta=m,
                        path=" + ".join(x["path"] for x in v)))
    return out


def rms(se, n):
    return float(np.sqrt(np.sum(se) / max(float(np.sum(n)), 1.0)))


def main():
    ap = argparse.ArgumentParser(
        description="Парное сравнение архитектур §49 по дампам val_sel")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rows", nargs="*", default=[],
                    help="дампы K-14c (--dump-rows), по одному на прогон")
    ap.add_argument("--baseline", default="baseline",
                    help="имя архитектуры, играющей роль опоры")
    ap.add_argument("--average-seeds", action="store_true",
                    help="усреднить построчные ошибки между сидами внутри "
                         "каждой архитектуры (правило §49.1)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=7)
    ap.add_argument("--eq", type=float, default=None,
                    help="зона практической эквивалентности; по умолчанию "
                         "берётся из K-14g")
    ap.add_argument("--allow-smoke", action="store_true",
                    help="принять дампы смоука; только для отладки самого "
                         "сравнения, решение по ним принимать нельзя")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="reports/k14j/arch_compare.json")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, os.path.abspath(a.root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    # ОДНА РЕАЛИЗАЦИЯ БУТСТРАПА И ОДИН ВЕРДИКТ НА ВЕСЬ ПРОЕКТ. Вторая копия
    # той же формулы означала бы два слегка разных ответа на один вопрос.
    import k14c_train_q1 as kt
    import k14g_compare as kg

    if a.selftest:
        selftest(kt, kg)
        return 0
    if not a.rows:
        raise SystemExit("нужен хотя бы --rows с двумя дампами")

    dumps = [load_dump(p) for p in a.rows]
    n_el = check_comparable(dumps, allow_smoke=a.allow_smoke)
    if a.average_seeds:
        dumps = average_seeds(dumps)
    else:
        seeds = sorted({int(d["meta"]["seed"]) for d in dumps})
        if len(seeds) != 1:
            raise SystemExit(
                f"в подборке сиды {seeds}: без --average-seeds сравнивать "
                f"прогоны разных сидов нельзя — разность включала бы "
                f"разницу порядков данных")
    names = [str(d["meta"]["architecture"]) for d in dumps]
    if len(set(names)) != len(names):
        raise SystemExit(f"архитектура повторяется: {names}")
    if a.baseline not in names:
        raise SystemExit(f"нет опоры {a.baseline}; есть {sorted(names)}")
    if len(names) < 2:
        raise SystemExit("сравнивать нечего: нужна опора и хотя бы один "
                         "кандидат")

    by = {str(d["meta"]["architecture"]): d for d in dumps}
    eq = float(a.eq) if a.eq is not None else float(kg.EQ)
    cands = [n_ for n_ in sorted(by) if n_ != a.baseline]
    # ПОРЯДОК В ПАРЕ ЗАДАЁТ ЗНАК: cluster_boot считает out[a] − out[b], то
    # есть RMS(опора) − RMS(кандидат). Положительное = КАНДИДАТ ЛУЧШЕ.
    deltas = [(a.baseline, c) for c in cands]
    if "reattn_draft" in by and "reattn_state" in by:
        deltas.append(("reattn_state", "reattn_draft"))
    res = kt.cluster_boot({k: v["se"] for k, v in by.items()},
                          by[a.baseline]["episode"], n_el,
                          n=int(a.boot), seed=int(a.boot_seed), qs=(5, 95),
                          deltas=deltas)

    point = {k: rms(v["se"], v["n"]) for k, v in by.items()}
    n_ep = int(len(np.unique(by[a.baseline]["episode"])))
    print(f"\n  строк {len(by[a.baseline]['rows'])}, эпизодов {n_ep}, "
          f"реплик {a.boot}, зона эквивалентности ±{eq}")
    print(f"  {'архитектура':16s} {'RMS-8':>10s}  {'эпоха':>6s}  сид")
    for k in sorted(by):
        m = by[k]["meta"]
        print(f"  {k:16s} {point[k]:10.6f}  {str(m['selected_epoch']):>6s}  "
              f"{m['seed']}")

    print("\n  ПАРНЫЕ РАЗНОСТИ, 90% интервал. Положительное = ВТОРАЯ лучше:")
    verdicts = {}
    for x_, y_ in deltas:
        ci = res[f"d_rms:{x_}-{y_}"]
        d_, e_ = kg.verdict(ci, eq=eq)
        verdicts[f"{y_} против {x_}"] = dict(
            ci=[float(ci[0]), float(ci[1])], direction=d_, magnitude=e_,
            point=float(point[x_] - point[y_]))
        print(f"    {y_:14s} против {x_:14s} "
              f"[{ci[0]:+.6f}, {ci[1]:+.6f}]  точечно "
              f"{point[x_] - point[y_]:+.6f}  {d_}, {e_}")

    # --- ПРАВИЛО §49 -----------------------------------------------------
    admitted = [c for c in cands
                if float(res[f"d_rms:{a.baseline}-{c}"][0]) > 0]
    print("\n  ПРАВИЛО §49: нижняя граница интервала против опоры строго "
          "выше нуля")
    if not admitted:
        choice = None
        print("    не прошёл НИ ОДИН кандидат -> не выбираем никого, "
              "val_confirm не открывается")
    elif len(admitted) == 1:
        choice = admitted[0]
        print(f"    прошёл один: {choice}")
    else:
        best = sorted(admitted, key=lambda c: point[c])
        if abs(point[best[0]] - point[best[1]]) <= 1e-9:
            choice = "reattn_draft" if "reattn_draft" in admitted else best[0]
            print(f"    прошли {admitted}, точечно равны -> {choice} "
                  f"(основная архитектурная гипотеза)")
        else:
            choice = best[0]
            print(f"    прошли {admitted} -> меньший точечный RMS: {choice}")

    out = dict(kind="k14j_arch_compare", baseline=a.baseline,
               architectures=sorted(by), rows=int(len(by[a.baseline]["rows"])),
               n_episodes=n_ep, n_elements_per_row=n_el,
               boot=int(a.boot), boot_seed=int(a.boot_seed), eq=eq,
               averaged_over_seeds=bool(a.average_seeds),
               allow_smoke=bool(a.allow_smoke),
               point_rms=point, intervals={k: [float(v[0]), float(v[1])]
                                           for k, v in res.items()},
               verdicts=verdicts, admitted=admitted, choice=choice,
               rule=("нижняя граница 90% парного интервала против опоры "
                     "строго выше нуля; иначе не выбираем никого"),
               sources={str(d["meta"]["architecture"]): dict(
                   path=d["path"], seed=d["meta"]["seed"],
                   selected_epoch=d["meta"]["selected_epoch"],
                   selected_state_sha1=d["meta"]["selected_state_sha1"],
                   val_sel=d["meta"].get("val_sel"),
                   git_head=d["meta"].get("git_head"),
                   identity_gate_run_id=d["meta"].get("identity_gate_run_id"))
                   for d in dumps})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    t_ = a.out + f".tmp.{os.getpid()}"
    json.dump(out, open(t_, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(t_, a.out)
    print(f"\n  сохранено: {a.out}")
    return 0


def _mk(path, arch, seed, se, eps, rows=None, n_el=56, **meta):
    rows = np.arange(len(se), dtype=np.int64) if rows is None else rows
    m = dict(kind="k14c_rows_val_sel", part="val_sel", variant="main",
             architecture=arch, seed=int(seed), additive_feedback="on",
             selected_epoch=3, selected_state_sha1=f"sha_{arch}_{seed}",
             plan_sha1="PLAN", q1_cache_sha1="CACHE", smoke=False,
             selection_only=True, val_sel=0.0,
             q0_prov=dict(q0_npz_sha1="Q0", q0_manifest_sha1="QM",
                          gate_r_sha1="GR"))
    m.update(meta)
    with open(path, "wb") as fh:
        np.savez_compressed(
            fh, rows=rows, episode=np.asarray(eps, np.int64),
            se=np.asarray(se, float),
            n=np.full(len(se), n_el, np.int64),
            meta=json.dumps(m, ensure_ascii=False))
    return path


def selftest(kt, kg):
    import tempfile
    rg = np.random.default_rng(0)
    N, NE = 400, 40
    eps = np.repeat(np.arange(NE), N // NE)
    base = rg.gamma(2.0, 0.5, size=N)

    with tempfile.TemporaryDirectory() as td:
        j = lambda x: os.path.join(td, x)

        # --- ЗНАК: кандидат СТРОГО лучше -> интервал строго положительный --
        pb = _mk(j("b.npz"), "baseline", 0, base, eps)
        pc = _mk(j("c.npz"), "reattn_draft", 0, base * 0.5, eps)
        db, dc = load_dump(pb), load_dump(pc)
        n_el = check_comparable([db, dc])
        r = kt.cluster_boot({"baseline": db["se"], "reattn_draft": dc["se"]},
                            db["episode"], n_el, n=400, seed=1,
                            deltas=[("baseline", "reattn_draft")])
        ci = r["d_rms:baseline-reattn_draft"]
        assert ci[0] > 0, ("кандидат лучше, а интервал не положителен", ci)
        assert kg.verdict(ci, eq=1e-9)[0] == "лучше"
        # и обратно: если поменять роли, знак обязан перевернуться
        r2 = kt.cluster_boot({"baseline": dc["se"], "reattn_draft": db["se"]},
                             db["episode"], n_el, n=400, seed=1,
                             deltas=[("baseline", "reattn_draft")])
        assert r2["d_rms:baseline-reattn_draft"][1] < 0, \
            "знак разности не зависит от того, кто опора"

        # --- РАВНЫЕ ДАННЫЕ -> интервал содержит ноль ----------------------
        pe = _mk(j("e.npz"), "reattn_state", 0, base.copy(), eps)
        de = load_dump(pe)
        r3 = kt.cluster_boot({"baseline": db["se"], "reattn_state": de["se"]},
                             db["episode"], n_el, n=400, seed=1,
                             deltas=[("baseline", "reattn_state")])
        c3 = r3["d_rms:baseline-reattn_state"]
        assert c3[0] <= 0 <= c3[1], ("одинаковые данные, а ноль вне "
                                     "интервала", c3)

        # --- ОТКАЗЫ -------------------------------------------------------
        bad = [
            (_mk(j("x1.npz"), "reattn_draft", 0, base[:-8], eps[:-8]),
             "состав строк"),
            (_mk(j("x2.npz"), "reattn_draft", 0, base, eps[::-1]),
             "другие эпизоды"),
            (_mk(j("x3.npz"), "reattn_draft", 0, base, eps, smoke=True),
             "smoke"),
            (_mk(j("x4.npz"), "reattn_draft", 0, base, eps,
                 selection_only=False), "не в режиме отбора"),
            (_mk(j("x5.npz"), "reattn_draft", 0, base, eps,
                 plan_sha1="ДРУГОЙ"), "plan_sha1"),
            (_mk(j("x6.npz"), "reattn_draft", 0, base, eps,
                 q0_prov=dict(q0_npz_sha1="ИНОЙ", q0_manifest_sha1="QM",
                              gate_r_sha1="GR")), "черновик отличается"),
            (_mk(j("x7.npz"), "reattn_draft", 0, base, eps, n_el=55),
             "число элементов"),
        ]
        for path, why in bad:
            try:
                check_comparable([db, load_dump(path)])
            except SystemExit as e:
                assert why in str(e), (why, str(e))
            else:
                raise AssertionError(f"принят несравнимый дамп: {why}")
        # дамп не того вида
        px = _mk(j("x8.npz"), "reattn_draft", 0, base, eps, part="train")
        try:
            load_dump(px)
        except SystemExit as e:
            assert "часть" in str(e), e
        else:
            raise AssertionError("принят дамп другой части")

        # --- ПОРЯДОК СТРОК НЕ ВЛИЯЕТ -------------------------------------
        perm = rg.permutation(N)
        pp = _mk(j("p.npz"), "reattn_draft", 0, base[perm] * 0.5, eps[perm],
                 rows=np.arange(N, dtype=np.int64)[perm])
        dp = load_dump(pp)
        assert np.array_equal(dp["rows"], dc["rows"])
        assert np.allclose(dp["se"], dc["se"]), \
            "приведение порядка строк не работает"

        # --- УСРЕДНЕНИЕ ПО СИДАМ -----------------------------------------
        s1b = _mk(j("b1.npz"), "baseline", 1, base * 1.2, eps)
        s1c = _mk(j("c1.npz"), "reattn_draft", 1, base * 0.6, eps)
        av = average_seeds([db, dc, load_dump(s1b), load_dump(s1c)])
        assert len(av) == 2
        m_ = {x["meta"]["architecture"]: x for x in av}
        assert np.allclose(m_["baseline"]["se"], (base + base * 1.2) / 2), \
            "усредняются не суммы квадратов"
        assert m_["baseline"]["meta"]["averaged_over_seeds"] == 2
        try:
            average_seeds([db, dc, load_dump(s1c)])
        except SystemExit as e:
            assert "наборы сидов" in str(e), e
        else:
            raise AssertionError("принят разный набор сидов по архитектурам")

        # --- СКВОЗНОЙ ПРОГОН: ПРАВИЛО §49 ЦЕЛИКОМ ------------------------
        # Помощники проверены выше по отдельности; здесь проверяется сам
        # разбор аргументов, печать и ВЫВОД РЕШЕНИЯ. Решающая ветка, не
        # покрытая тестом, — это ветка, про которую мы узнаем в день, когда
        # она сработает.
        import subprocess
        me = os.path.abspath(__file__)
        for tag, se_c, want in (("кандидат лучше", base * 0.5,
                                 "reattn_draft"),
                                ("кандидат хуже", base * 1.5, None),
                                ("кандидат равен", base.copy(), None)):
            _mk(j("e_b.npz"), "baseline", 0, base, eps)
            _mk(j("e_c.npz"), "reattn_draft", 0, se_c, eps)
            o = j("out.json")
            r = subprocess.run(
                [sys.executable, me, "--rows", j("e_b.npz"), j("e_c.npz"),
                 "--boot", "400", "--out", o], capture_output=True, text=True)
            assert r.returncode == 0, (tag, r.stdout[-800:], r.stderr[-800:])
            got = json.load(open(o))
            assert got["choice"] == want, (tag, got["choice"], want)
            assert got["kind"] == "k14j_arch_compare"
            assert "reattn_draft против baseline" in got["verdicts"]
            if want is None:
                assert not got["admitted"], (tag, got["admitted"])
            else:
                assert got["admitted"] == ["reattn_draft"], tag
        # ОТКАЗЫ СКВОЗНОГО ПУТИ: нет опоры / нечего сравнивать
        for rows_, why in ((["e_c.npz"], "нет опоры"),
                           (["e_b.npz"], "сравнивать нечего")):
            r = subprocess.run(
                [sys.executable, me, "--rows"] + [j(x) for x in rows_]
                + ["--out", j("out2.json")], capture_output=True, text=True)
            assert r.returncode != 0, (why, r.stdout[-400:])
            assert why in r.stdout + r.stderr, (why, r.stderr[-400:])

    print("самопроверка k14j_compare_arch пройдена")


if __name__ == "__main__":
    sys.exit(main())
