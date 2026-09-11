"""K-11i: сводная таблица перебора численного режима по сохранённым JSON.

ЗАЧЕМ ОТДЕЛЬНЫЙ СКРИПТ. Прежняя таблица считалась одноразовой командой, и
аудируемого артефакта для неё не было. Здесь агрегация по сидам — код, который
можно перечитать и прогнать заново на тех же файлах.

ЧТО АГРЕГИРУЕТСЯ. Из каждого прогона берётся ПОСЛЕДНИЙ замер ряда `post`, то
есть состояние политики ПОСЛЕ всех шагов, измеренное на ВСЁМ буфере. По сидам
печатается медиана и размах: выбирать предохранители по одному сиду нельзя.

Запуск:
    python3 experiments/k11i_table.py --selftest
    python3 experiments/k11i_table.py --glob 'data/k11i/m_s*.json'
"""

import argparse
import glob
import json
import os
import re

import numpy as np

def run_key(d, path):
    """Параметры режима из САМОГО JSON, а не из имени файла.

    Разбор имени был хрупким: стоило назвать файлы иначе, и скрипт отказывал,
    хотя все параметры лежат внутри. Имя теперь ни на что не влияет, а
    отсутствие поля — отказ, а не догадка.
    """
    miss = [k for k in ("epochs", "lr", "minibatch", "batch", "seed")
            if d.get(k) is None]
    if miss:
        raise SystemExit(f"{path}: в JSON нет полей {miss}")
    # BATCH В КЛЮЧЕ. Прежде он требовался в JSON, но в группировку не входил:
    # прогоны с батчем 128 и 256 попали бы в одну ячейку и молча перезаписали
    # бы друг друга.
    return (int(d["epochs"]), int(d["minibatch"]), int(d["batch"]),
            float(d["lr"]), int(d["seed"]))


# ПОЛЯ, ОБЯЗАННЫЕ СОВПАСТЬ У ВСЕХ ПРОГОНОВ. Прежде не сверялось ничего:
# набор из s0 с одним сидом и s1 с другим, при чужом script_sha1, принимался
# как прошедший режим.
SHARED = ("sigma", "clip_eps", "device", "dtype", "torch_version",
          "res_norm_sha1", "basis_sha1", "rho_sha1", "rank", "target",
          "cache", "batch", "script_sha1", "hicora_g_sha1",
          "hicora_vla_sha1")
# eps и преимущества зависят ТОЛЬКО от seed: внутри одного сида обязаны
# совпадать у всех режимов и обеих голов, между сидами — различаться.
PER_SEED = ("eps_sha1", "adv_sha1")


def check_provenance(meta, expect_script=None):
    """Все прогоны обязаны быть сопоставимы. Отсутствие поля — отказ."""
    bad = []
    for f in SHARED:
        vals = {str(m.get(f)) for m in meta.values()}
        if "None" in vals:
            bad.append(f"{f}: не записан в части прогонов")
        elif len(vals) > 1:
            bad.append(f"{f}: разные значения {sorted(vals)[:4]}")
    hs = {json.dumps(m.get("head_sha1"), sort_keys=True)
          for m in meta.values()}
    if len(hs) != 1 or "null" in hs:
        bad.append(f"head_sha1: разные или отсутствуют ({len(hs)} вариантов)")
    if expect_script:
        got = {str(m.get("script_sha1")) for m in meta.values()}
        if got != {expect_script}:
            bad.append(f"script_sha1 {sorted(got)} вместо {expect_script}: "
                       f"прогоны сделаны другой версией стенда")
    by_seed = {}
    for k, m in meta.items():
        by_seed.setdefault(k[-1], []).append(
            tuple(str(m.get(f)) for f in PER_SEED))
    for sd, vs in sorted(by_seed.items()):
        if len(set(vs)) > 1:
            bad.append(f"сид {sd}: eps/преимущества различаются между "
                       f"режимами, хотя зависят только от сида")
        if any("None" in v for v in vs):
            bad.append(f"сид {sd}: eps_sha1 или adv_sha1 не записан")
    uniq = {vs[0] for vs in by_seed.values()}
    if len(uniq) != len(by_seed):
        bad.append("разные сиды дали ОДИНАКОВЫЙ eps: буферы не зависят от "
                   "сида")
    if bad:
        raise SystemExit("ПРОГОНЫ НЕ СОПОСТАВИМЫ:\n    " + "\n    ".join(bad))
    return True


def check_complete(rows, expect_seeds=None, heads=("s0", "s1"),
                  manifest=None):
    """Одинаковый набор сидов у ВСЕХ режимов и ОБЕИХ голов; без дублей.

    MANIFEST «режим -> обязательные сиды» нужен, когда режимы намеренно шли с
    разным числом сидов. Без него один глобальный `--expect-seeds` либо
    отвергает законный неравномерный набор, либо (если не задан) принимает
    набор, где lr=1e-6 считан на сиде 0, а lr=3e-6 — на сиде 4. Второе я
    воспроизвёл: оно проходило.
    """
    bad = []
    regimes = sorted({k[:4] for k in rows})
    seeds_by = {}
    for key, per in rows.items():
        seeds_by[key] = set(per)
    for r in regimes:
        for hd in heads:
            k = r + (hd,)
            if k not in rows:
                bad.append(f"{r}: нет головы {hd}")
        got = [seeds_by.get(r + (hd,), set()) for hd in heads
               if r + (hd,) in rows]
        if got and len(set(map(frozenset, got))) > 1:
            bad.append(f"{r}: головы измерены на РАЗНЫХ сидах {got}")
        want = None
        if manifest is not None:
            key = f"e{r[0]}|mb{r[1]}|b{r[2]}|lr{r[3]:.0e}"
            if key not in manifest:
                bad.append(f"{key}: режима нет в манифесте")
            else:
                want = set(manifest[key])
        elif expect_seeds is not None:
            want = set(expect_seeds)
        if want is not None:
            for hd in heads:
                k = r + (hd,)
                if k in rows and seeds_by[k] != want:
                    bad.append(f"{r}, {hd}: сиды {sorted(seeds_by[k])} "
                               f"вместо {sorted(want)}")
    if manifest is not None:
        have = {f"e{r[0]}|mb{r[1]}|b{r[2]}|lr{r[3]:.0e}" for r in regimes}
        missing = sorted(set(manifest) - have)
        if missing:
            bad.append(f"режимы из манифеста не посчитаны: {missing}")
    if True:
        pass
    for key, per in rows.items():
        exp = ((key[2] + key[1] - 1) // key[1]) * key[0]
        for sd, v in per.items():
            if v["steps"] != exp:
                bad.append(f"{key}, сид {sd}: шагов {v['steps']} вместо "
                           f"{exp}")
    if bad:
        raise SystemExit("НАБОР НЕПОЛОН ИЛИ НЕСОГЛАСОВАН:\n    "
                         + "\n    ".join(bad[:10])
                         + ("\n    ..." if len(bad) > 10 else ""))
    return dict(regimes=len(regimes),
                seeds=sorted(next(iter(seeds_by.values()))))


def agg(vals):
    """Медиана и размах по сидам. Одно число по одному сиду — не вывод."""
    a = np.asarray(vals, float)
    return dict(median=float(np.median(a)), lo=float(a.min()),
                hi=float(a.max()), n=int(a.size))


def collect(files):
    """Последний замер ряда post по каждому (эпох, lr, голова, сид)."""
    rows, meta = {}, {}
    for f in sorted(files):
        d = json.load(open(f))
        ep, mb, bt, lr, sd = run_key(d, f)
        if d.get("train_log_std"):
            raise SystemExit(f"{f}: sigma обучалась, это другая абляция")
        for hd, v in d["heads"].items():
            post = v.get("post")
            if not post:
                raise SystemExit(f"{f}, {hd}: нет ряда post — файл от старой "
                                 f"версии, где замер шёл ДО обновления")
            key = (ep, mb, bt, lr, hd)
            if sd in rows.get(key, {}):
                raise SystemExit(f"ДУБЛЬ: {key}, сид {sd} встречается дважды "
                                 f"(последний файл {f})")
            meta[(ep, mb, bt, lr, sd)] = {
                k: d.get(k) for k in SHARED + PER_SEED + ("head_sha1",)}
            rows.setdefault(key, {})[sd] = dict(
                fin=post[-1],
                # ГЕЙТ СЧИТАЕТСЯ ПО ХУДШЕМУ ЗА ВСЮ ПОСЛЕДОВАТЕЛЬНОСТЬ, а не
                # только по финалу: промежуточно негодный шаг не становится
                # допустимым оттого, что последний вернулся назад.
                worst_clip=max(m["clip_frac"] for m in post),
                worst_q99=max(m.get("log_ratio_q99", float("inf"))
                              for m in post),
                worst_kl=max(m["kl_exact"] for m in post),
                traj=[m["clip_frac"] for m in post],
                identity=v["identity"], steps=len(post))
    return rows, meta


def table(rows, out=print):
    out("  столбцы «обрезано», «q99» и «KL» — ХУДШЕЕ за всю "
        "последовательность шагов")
    out(f"  {'эпох':>5}{'мб':>5}{'батч':>6}{'lr':>8}{'гол':>4}{'сид':>4}"
        f"{'обрезано, медиана [размах]':>27}{'q99|log_r|':>11}"
        f"{'KL':>10}{'выравн.':>9}{'хвост':>8}")
    best = []
    for key in sorted(rows):
        ep, mb, bt, lr, hd = key
        per = rows[key]
        cf = agg([p["worst_clip"] for p in per.values()])
        q99 = agg([p["worst_q99"] for p in per.values()])
        kl = agg([p["worst_kl"] for p in per.values()])
        al = agg([p["fin"]["align_mean"] for p in per.values()])
        # ПОЛЕ МОЖЕТ ОТСУТСТВОВАТЬ у прогонов, сделанных до его введения.
        # nan в таблице читался бы как измеренная величина; печатаем прочерк.
        tps = [p["fin"]["align_top1pct"] for p in per.values()
               if p["fin"].get("align_top1pct") is not None]
        tp = agg(tps) if tps else None
        tp_s = "—" if tp is None else f"{tp['median']:.2f}"
        out(f"  {ep:>5}{mb:>5}{bt:>6}{lr:>8.0e}{hd:>4}{cf['n']:>4}"
            f"{100 * cf['median']:>18.1f}% "
            f"[{100 * cf['lo']:.0f}-{100 * cf['hi']:.0f}%]"
            f"{q99['median']:>11.3f}{kl['median']:>10.4g}"
            f"{al['median']:>9.3f}{tp_s:>8}")
        best.append((key, cf, q99))
    return best


def read_guard(best, max_clip=0.10, max_q99=0.5):
    """Какие режимы уложились в КАНДИДАТНЫЕ предохранители на ВСЕХ сидах.

    Пороги здесь НЕ зарегистрированы: они выбираются по этому же перебору,
    поэтому называются кандидатными. Регистрировать их можно только отдельным
    объявлением до следующего измерения.
    """
    ok = {}
    for key, cf, q99 in best:
        ok[key] = bool(cf["hi"] <= max_clip and q99["hi"] <= max_q99)
    regimes = sorted({k[:4] for k in ok})
    passing = [r for r in regimes
               if all(ok.get(r + (hd,), False) for hd in ("s0", "s1"))]
    # ОДИН РЕЖИМ НЕ ВЫБИРАЕТСЯ. На поддельных преимуществах нет критерия
    # качества, позволяющего предпочесть больший шаг меньшему; возвращается
    # ОБЛАСТЬ. Прежде брался минимальный прошедший, и «кандидат 3e-6» из
    # отчёта правилом не был записан вовсе.
    return dict(per_arm=ok, passing=passing,
                region_lr=sorted({r[3] for r in passing}),
                thresholds=dict(max_clip=max_clip, max_q99=max_q99))


def selftest():
    # ПАРАМЕТРЫ ИЗ JSON, имя файла ни на что не влияет; batch входит в ключ.
    d0 = dict(epochs=2, lr=1e-5, minibatch=64, batch=256, seed=3)
    assert run_key(d0, "любое_имя.json") == (2, 64, 256, 1e-5, 3)
    for gone in ("epochs", "lr", "minibatch", "batch", "seed"):
        try:
            run_key({k: v for k, v in d0.items() if k != gone}, "f")
        except SystemExit:
            pass
        else:
            raise AssertionError(f"отсутствие {gone} принято")
    a = agg([0.1, 0.2, 0.3])
    assert a == dict(median=0.2, lo=0.1, hi=0.3, n=3), a

    def m_(sd, **kw):
        base = dict(sigma=0.1, clip_eps=0.2, device="cuda:0", dtype="float16",
                    torch_version="2.4.1+cu124",
                    res_norm_sha1="rn", basis_sha1="bs", rho_sha1="rh",
                    rank=32, target="coef", cache="data/c", batch=256,
                    script_sha1="SC", hicora_g_sha1="HG",
                    hicora_vla_sha1="HV", head_sha1={"s0": "h0", "s1": "h1"},
                    eps_sha1=f"E{sd}", adv_sha1=f"A{sd}")
        base.update(kw)
        return base

    meta = {(1, 256, 256, 3e-6, sd): m_(sd) for sd in (0, 1, 2)}
    assert check_provenance(meta, "SC")
    # ВОСПРОИЗВЕДЁННЫЙ ОБХОД: чужая версия стенда.
    try:
        check_provenance(meta, "ДРУГАЯ")
    except SystemExit as e:
        assert "script_sha1" in str(e), str(e)
    else:
        raise AssertionError("чужая версия стенда принята")
    for fld, val, why in (("res_norm_sha1", "ИНАЯ", "чужая норма"),
                          ("basis_sha1", "ИНОЙ", "чужой базис"),
                          ("sigma", 0.3, "иная sigma"),
                          ("clip_eps", 0.1, "иной clip_eps"),
                          ("batch", 128, "иной батч"),
                          ("device", "cpu", "иная карта"),
                          ("head_sha1", {"s0": "X"}, "иные головы")):
        bad = dict(meta)
        k0 = (1, 256, 256, 3e-6, 1)
        bad[k0] = m_(1, **{fld: val})
        try:
            check_provenance(bad, "SC")
        except SystemExit:
            pass
        else:
            raise AssertionError(f"принято: {why}")
    for fld in ("res_norm_sha1", "eps_sha1", "adv_sha1"):
        bad = dict(meta)
        bad[(1, 256, 256, 3e-6, 1)] = m_(1, **{fld: None})
        try:
            check_provenance(bad, "SC")
        except SystemExit:
            pass
        else:
            raise AssertionError(f"отсутствие {fld} принято")
    # eps ОБЯЗАН ЗАВИСЕТЬ ТОЛЬКО ОТ СИДА: один и тот же у разных сидов — отказ
    same = {(1, 256, 256, 3e-6, sd): m_(sd, eps_sha1="E", adv_sha1="A")
            for sd in (0, 1)}
    try:
        check_provenance(same, "SC")
    except SystemExit as e:
        assert "ОДИНАКОВЫЙ eps" in str(e), str(e)
    else:
        raise AssertionError("одинаковый eps у разных сидов принят")
    # ...и РАЗНЫЙ у одного сида между режимами — тоже отказ
    mix = {(1, 256, 256, 3e-6, 0): m_(0),
           (1, 256, 256, 1e-6, 0): m_(0, eps_sha1="ДРУГОЙ")}
    try:
        check_provenance(mix, "SC")
    except SystemExit:
        pass
    else:
        raise AssertionError("разный eps у одного сида принят")

    def cell(cf, q99, steps=1, seeds=(0, 1, 2)):
        return {sd: dict(fin=dict(clip_frac=cf, log_ratio_q99=q99,
                                  kl_exact=0.01, align_mean=0.05,
                                  align_top1pct=-0.5),
                         worst_clip=cf, worst_q99=q99, worst_kl=0.01,
                         traj=[cf] * steps, identity={}, steps=steps)
                for sd in seeds}

    rows = {(1, 256, 256, 3e-6, hd): cell(0.06, 0.35) for hd in ("s0", "s1")}
    c = check_complete(rows, [0, 1, 2])
    assert c["regimes"] == 1 and c["seeds"] == [0, 1, 2]
    # ВОСПРОИЗВЕДЁННЫЙ ОБХОД: s0 только сид 0, s1 только сид 4
    skew = {(1, 256, 256, 3e-6, "s0"): cell(0.06, 0.35, seeds=(0,)),
            (1, 256, 256, 3e-6, "s1"): cell(0.06, 0.35, seeds=(4,))}
    try:
        check_complete(skew, [0, 1, 2])
    except SystemExit as e:
        assert "РАЗНЫХ сидах" in str(e) or "вместо" in str(e), str(e)
    else:
        raise AssertionError("головы на разных сидах приняты")
    try:
        check_complete({(1, 256, 256, 3e-6, "s0"): cell(0.06, 0.35)}, [0, 1, 2])
    except SystemExit as e:
        assert "нет головы s1" in str(e), str(e)
    else:
        raise AssertionError("отсутствие головы принято")
    try:
        check_complete(rows, [0, 1, 2, 3, 4])
    except SystemExit:
        pass
    else:
        raise AssertionError("неполный набор сидов принят")
    # ЧИСЛО ШАГОВ ДОЛЖНО СООТВЕТСТВОВАТЬ (батч/минибатч)*эпох
    wrong = {(1, 64, 256, 1e-5, hd): cell(0.06, 0.35, steps=3)
             for hd in ("s0", "s1")}
    try:
        check_complete(wrong, [0, 1, 2])
    except SystemExit as e:
        assert "шагов 3 вместо 4" in str(e), str(e)
    else:
        raise AssertionError("неверное число шагов принято")

    # --- гейт по ХУДШЕМУ за последовательность ----------------------------
    ok_cells = {(1, 256, 256, 3e-6, hd): cell(0.06, 0.35)
                for hd in ("s0", "s1")}
    best = [(k, agg([v["worst_clip"] for v in per.values()]),
             agg([v["worst_q99"] for v in per.values()]))
            for k, per in ok_cells.items()]
    g = read_guard(best)
    assert g["passing"] == [(1, 256, 256, 3e-6)], g["passing"]
    assert g["region_lr"] == [3e-6]
    assert "chosen" not in g, "один режим выбираться не должен"
    # ПРОМЕЖУТОЧНО НЕГОДНЫЙ ШАГ: финал хороший, худшее плохое
    bad_mid = {}
    for hd in ("s0", "s1"):
        per = cell(0.06, 0.35)
        for sd in per:
            per[sd]["worst_clip"] = 0.8
        bad_mid[(1, 256, 256, 3e-6, hd)] = per
    best2 = [(k, agg([v["worst_clip"] for v in per.values()]),
              agg([v["worst_q99"] for v in per.values()]))
             for k, per in bad_mid.items()]
    assert read_guard(best2)["passing"] == [], "худшее по ряду не учтено"
    # ОДИН ПЛОХОЙ СИД ЛОМАЕТ РЕЖИМ
    g2 = read_guard([((1, 256, 256, 3e-6, "s0"), agg([0.04, 0.40]),
                      agg([0.2, 0.2])),
                     ((1, 256, 256, 3e-6, "s1"), agg([0.04, 0.04]),
                      agg([0.2, 0.2]))])
    assert g2["passing"] == []
    # ХВОСТ ЛОМАЕТ НАРАВНЕ С ДОЛЕЙ
    g3 = read_guard([((1, 256, 256, 3e-6, "s0"), agg([0.02]), agg([3.0])),
                     ((1, 256, 256, 3e-6, "s1"), agg([0.02]), agg([0.1]))])
    assert g3["passing"] == []
    # ОТСУТСТВИЕ ГОЛОВЫ
    g4 = read_guard([((1, 256, 256, 3e-6, "s0"), agg([0.02]), agg([0.1]))])
    assert g4["passing"] == []

    # --- файлы от старой версии и с обученной sigma ------------------------
    import tempfile
    d = tempfile.mkdtemp()
    f = os.path.join(d, "любое.json")
    base = dict(epochs=1, lr=1e-5, minibatch=64, batch=256, seed=0)
    json.dump(dict(base, heads=dict(s0=dict(identity={}, steps=[]))),
              open(f, "w"))
    try:
        collect([f])
    except SystemExit as e:
        assert "нет ряда post" in str(e), str(e)
    else:
        raise AssertionError("файл без post принят")
    json.dump(dict(base, train_log_std=True,
                   heads=dict(s0=dict(identity={},
                                      post=[dict(clip_frac=0.0,
                                                 log_ratio_q99=0.0,
                                                 kl_exact=0.0)]))),
              open(f, "w"))
    try:
        collect([f])
    except SystemExit as e:
        assert "другая абляция" in str(e), str(e)
    else:
        raise AssertionError("прогон с обученной sigma принят")
    # ДУБЛЬ ЯЧЕЙКИ
    post1 = [dict(clip_frac=0.0, log_ratio_q99=0.0, kl_exact=0.0)]
    g1 = os.path.join(d, "a.json")
    g2f = os.path.join(d, "b.json")
    for ff in (g1, g2f):
        json.dump(dict(base, heads=dict(s0=dict(identity={}, post=post1))),
                  open(ff, "w"))
    try:
        collect([g1, g2f])
    except SystemExit as e:
        assert "ДУБЛЬ" in str(e), str(e)
    else:
        raise AssertionError("дубль принят")

    # --- ИНТЕГРАЦИОННЫЙ ТЕСТ: НАСТОЯЩИЙ ВЫХОД ВОРКЕРА -------------------
    # Обе самопроверки проходили на РУЧНЫХ фикстурах, и несовпадение полей
    # (воркер не писал device и dtype, агрегатор их требовал) не обнаружилось.
    # Здесь набор полей берётся ИЗ КОДА ВОРКЕРА, а не переписывается руками.
    import re as _re
    wp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "k11i_ppo_smoke.py")
    if os.path.exists(wp):
        src = open(wp).read()
        i = src.index("    out = dict(sigma=a.sigma")
        j = src.index("heads={})", i)
        blk = src[i:j]
        written = set(_re.findall(r"(\w+)\s*=", blk))
        need = set(SHARED) | set(PER_SEED) | {"epochs", "lr", "minibatch",
                                              "batch", "seed", "head_sha1"}
        # eps_sha1 и adv_sha1 воркер пишет отдельным out.update — учитываем.
        for extra in _re.findall(r"out\.update\(([^)]*)\)", src):
            written |= set(_re.findall(r"(\w+)\s*=", extra))
        miss = sorted(need - written)
        if miss:
            raise SystemExit(
                "ВОРКЕР И АГРЕГАТОР НЕСОВМЕСТИМЫ: k11i_ppo_smoke.py не "
                f"записывает поля {miss},\n  а k11i_table.py их требует. "
                "Файлы, созданные воркером, будут отвергнуты.")
        print(f"  интеграция: воркер пишет все {len(need)} полей, которых "
              f"требует агрегатор")

    print("самопроверка k11i_table пройдена: провенанс сверяется по "
          "четырнадцати полям,\n  eps зависит только от сида, набор сидов "
          "обязан совпадать у обеих голов,\n  число шагов сверяется, дубли "
          "отвергаются, гейт берёт ХУДШЕЕ за ряд,\n  один режим не "
          "выбирается — возвращается область")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--glob", default="data/k11i/m_s*.json")
    ap.add_argument("--max-clip", type=float, default=0.10)
    ap.add_argument("--max-q99", type=float, default=0.5)
    ap.add_argument("--expect-seeds", default=None,
                    type=lambda v: [int(x) for x in v.split(",")],
                    help="обязательный набор сидов, например 0,1,2. Без него "
                         "проверяется только СОГЛАСОВАННОСТЬ сидов между "
                         "головами, но не их число")
    ap.add_argument("--manifest", default=None,
                    help="JSON «режим -> список сидов», например "
                         "'{\"e1|mb256|b256|lr3e-06\": [0,1,2]}'. Нужен, "
                         "когда режимы шли с РАЗНЫМ числом сидов: иначе "
                         "неравномерный набор либо отвергается, либо "
                         "принимается без проверки")
    ap.add_argument("--no-script-check", action="store_true",
                    help="не требовать, чтобы прогоны были сделаны текущей "
                         "версией k11i_ppo_smoke.py")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit(f"нет файлов по {a.glob}")
    rows, meta = collect(files)
    me = None
    if not a.no_script_check:
        sp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "k11i_ppo_smoke.py")
        if os.path.exists(sp):
            import hashlib
            h = hashlib.sha1()
            with open(sp, "rb") as fh:
                for c in iter(lambda: fh.read(1 << 22), b""):
                    h.update(c)
            me = h.hexdigest()[:12]
    check_provenance(meta, me)
    man = json.loads(a.manifest) if a.manifest else None
    comp = check_complete(rows, a.expect_seeds, manifest=man)
    print(f"  файлов {len(files)}, ячеек {len(rows)}, режимов "
          f"{comp['regimes']}, сиды {comp['seeds']}")
    print(f"  провенанс сверен: одна версия стенда"
          + (f" ({me})" if me else "")
          + ", одни веса, eps зависит только от сида")
    print("\n  ПОСЛЕ всех шагов, замер на ВСЁМ буфере, агрегация по сидам")
    best = table(rows)
    g = read_guard(best, a.max_clip, a.max_q99)
    print(f"\n  КАНДИДАТНЫЕ предохранители: обрезано <= {100*a.max_clip:.0f}% "
          f"и q99|log_ratio| <= {a.max_q99} НА ВСЕХ сидах и обеих головах")
    for r in sorted({k[:4] for k in g["per_arm"]}):
        mark = "годен" if r in g["passing"] else "-"
        print(f"    эпох {r[0]}, минибатч {r[1]}, батч {r[2]}, "
              f"lr {r[3]:.0e}  {mark}")
    print("\n  ТРАЕКТОРИЯ доли обрезанных по шагам (сид 0, голова s0):")
    for key, per in sorted(rows.items()):
        ep, mb, bt, lr, hd = key
        if hd != "s0" or 0 not in per:
            continue
        print(f"    эпох {ep}, мб {mb}, батч {bt}, lr {lr:.0e}: "
              + " ".join(f"{100*c:.0f}%" for c in per[0]["traj"]))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        # КЛЮЧИ-КОРТЕЖИ JSON НЕ ПРИНИМАЕТ: приводим к строкам. Таблица до
        # этого уже напечатана, поэтому падение теряло только файл.
        g_out = dict(g)
        g_out["per_arm"] = {f"e{k[0]}|mb{k[1]}|b{k[2]}|lr{k[3]:.0e}|{k[4]}": v
                            for k, v in g["per_arm"].items()}
        g_out["passing"] = [f"e{r[0]}|mb{r[1]}|b{r[2]}|lr{r[3]:.0e}"
                            for r in g["passing"]]
        g_out["region_lr"] = g["region_lr"]
        json.dump(dict(guard=g_out, n_files=len(files),
                       cells={f"e{k[0]}|mb{k[1]}|b{k[2]}|lr{k[3]:.0e}|{k[4]}":
                              {str(s): v["fin"] for s, v in per.items()}
                              for k, per in rows.items()}),
                  open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {a.out}")
    if not g["passing"]:
        print("\n  НИ ОДИН РЕЖИМ не уложился в кандидатные предохранители на "
              "всех сидах.\n  Подбирать порог под измеренное нельзя: нужен "
              "либо меньший шаг, либо другая\n  конструкция обновления.")
        raise SystemExit(1)
    print(f"\n  ДОПУСТИМАЯ ОБЛАСТЬ: lr в "
          f"{[f'{x:.0e}' for x in g['region_lr']]} при режимах "
          f"{[f'эпох {r[0]}, мб {r[1]}, батч {r[2]}' for r in g['passing']]}")
    print("  ОДИН РЕЖИМ НЕ ВЫБИРАЕТСЯ: на поддельных преимуществах нет "
          "критерия качества,\n  позволяющего предпочесть больший шаг "
          "меньшему. Пороги выбраны ПО ЭТОМУ ЖЕ\n  перебору, поэтому "
          "кандидатные: регистрировать их можно только отдельным\n  "
          "объявлением до следующего измерения.")


if __name__ == "__main__":
    main()
