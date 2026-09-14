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
    """{(сюита, задача, состояние): (успех, хэш)} с отказом на дублях.

    Сюита входит в ключ: object/0 и goal/0 — разные задачи, и без неё их
    эпизоды слились бы в одну пару.
    """
    out = {}
    for c in cells:
        su = str(c.get("suite"))
        for t in c["task_ids"]:
            for e in c["episodes"]:
                k = (su, int(t), int(e["state_id"]))
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


def by_block(cells):
    """Ячейки по (задача, init_start). Дубликаты блока — отказ."""
    out = {}
    for c in cells:
        k = (str(c.get("suite")), int(c["task_ids"][0]), int(c["init_start"]))
        if k in out:
            raise SystemExit(f"{c['_path']}: блок {k} уже есть в "
                             f"{out[k]['_path']}")
        out[k] = c
    return out


def check_eps_streams(a_cells, b_cells, *, label_a, label_b):
    """ФАКТИЧЕСКИЕ реализации шума, а не совпадение метаданных о сиде.

    Сверяется общий префикс повызовных хэшей: после расхождения траекторий
    число вызовов у двух политик разное, и это нормально — а вот разные
    реализации при одинаковых сиде и соли означают, что вместе с весами
    сменился и случайный поток, и разницу исходов нельзя отнести к обучению.
    """
    A, B = by_block(a_cells), by_block(b_cells)
    common = sorted(set(A) & set(B))
    if not common:
        raise SystemExit(f"у {label_a} и {label_b} нет общих блоков")
    miss = sorted((set(A) | set(B)) - (set(A) & set(B)))
    if miss:
        raise SystemExit(f"блоки {miss[:6]} есть только у одной из рук "
                         f"{label_a}/{label_b}")
    n_cmp = 0
    for k in common:
        ea = A[k].get("eps_sha1_by_call")
        eb = B[k].get("eps_sha1_by_call")
        if not ea or not eb:
            raise SystemExit(
                f"блок {k}: нет повызовных хэшей шума (eps_sha1_by_call) — "
                f"проверить общий поток нечем, а совпадение сида его не "
                f"доказывает")
        n = min(len(ea), len(eb))
        if ea[:n] != eb[:n]:
            first = next(i for i in range(n) if ea[i] != eb[i])
            raise SystemExit(
                f"блок {k}: реализации шума расходятся с вызова {first} "
                f"({ea[first]} против {eb[first]}) при одинаковых сиде и соли "
                f"— это разные случайные числа, а не разные веса")
        n_cmp += n
    return dict(blocks=len(common), calls_compared=n_cmp)


def check_provenance(cells, *, arm, d1_seed=None, rl_seed=None, sigma=None,
                     step_index=None, head_sha1=None, policy_sha1=None):
    """Состав руки: та ли рука, те ли сиды, головы, шаг и sigma."""
    bad = []
    for c in cells:
        tag = os.path.basename(c["_path"])
        if c.get("arm") != arm:
            bad.append(f"{tag}: рука {c.get('arm')} вместо {arm}")
        for nm, want, got in (("d1_seed", d1_seed, c.get("d1_seed")),
                              ("rl_seed", rl_seed, c.get("rl_seed")),
                              ("step_index", step_index, c.get("step_index")),
                              ("head_sha1", head_sha1, c.get("head_sha1")),
                              ("policy_sha1", policy_sha1,
                               c.get("policy_sha1"))):
            if want is not None and got != want:
                bad.append(f"{tag}: {nm}={got}, ожидалось {want}")
        if sigma is not None and abs(float(c.get("sigma", -1))
                                     - float(sigma)) > 1e-9:
            bad.append(f"{tag}: sigma={c.get('sigma')} вместо {sigma}")
        par = c.get("parity") or {}
        if par and not par.get("ok", True):
            bad.append(f"{tag}: паритет не сошёлся: {par}")
    if bad:
        raise SystemExit("состав руки не сходится:\n  - " + "\n  - ".join(bad))
    # единство происхождения внутри руки
    for nm in ("policy_sha1", "step_index", "sigma", "d1_seed", "rl_seed"):
        vals = {str(c.get(nm)) for c in cells}
        if len(vals) > 1:
            raise SystemExit(f"внутри руки {arm} разные {nm}: {sorted(vals)}")
    return True


def train_states(root):
    """Обучающие состояния — ИЗ МЕТАДАННЫХ раскаток, а не из имён файлов.

    Имя не несёт ни числа сред, ни списка состояний; прежняя версия разбирала
    имя и достраивала `range(5)` по захардкоженной пятёрке, то есть при другом
    --n-envs молча теряла часть состояний и пропускала пересечение с оценкой.
    """
    out = set()
    for side in sorted(glob.glob(os.path.join(root, "train_step*",
                                              "*.pt.meta.json"))):
        m = json.load(open(side))
        ids = m.get("state_ids")
        if not ids:
            ids = [e["state_id"] for e in m.get("episodes") or []]
        if not ids:
            raise SystemExit(f"{side}: в мете нет состояний, пересечение с "
                             f"оценкой не проверить")
        for t in m.get("task_ids") or [m.get("task_id")]:
            for i in ids:
                out.add((str(m.get("suite")), int(t), int(i)))
    return out


def mcnemar_p(rec, los):
    """Точный односторонний McNemar: P(X >= rec) при X ~ Binom(rec+los, 1/2).

    Единственная информация о знаке эффекта — в ДИСКОРДАНТНЫХ парах: согласные
    пары про разность не говорят ничего. Поэтому 9 против 7 — это не «+4.4 пп»,
    а подбрасывание монеты шестнадцать раз, и объявлять по такому сигнал
    нельзя.
    """
    import math
    n = int(rec) + int(los)
    if n == 0:
        return 1.0
    tot = sum(math.comb(n, k) for k in range(int(rec), n + 1))
    return tot / (2.0 ** n)


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
    import math
    return dict(n=n, recovered=rec, lost=los,
                p_one_sided=mcnemar_p(rec, los),
                # стандартная ошибка парной разности: вся она из дискордантных
                # пар, поэтому se = sqrt(rec + los) / n
                se=(math.sqrt(rec + los) / n) if n else None,
                effect=((rec - los) / n) if n else None,
                discord=((rec + los) / n) if n else None,
                success_a=(sum(1 for k in a if a[k][0]) / len(a)) if a else None,
                success_b=(sum(1 for k in b if b[k][0]) / len(b)) if b else None)


def check_versions(cells, allow_mixed=False):
    """Одна версия кода на весь отчёт, либо ЯВНОЕ разрешение смешать.

    Молчаливое смешение результатов до и после исправления ошибки — то, от чего
    журнал версий не защищает. Но и механический отказ не всегда верен: правка
    может менять только записываемые поля, не трогая исполнение. Поэтому
    исключение возможно, но лишь явным флагом и с перечислением расхождений.
    """
    import k12b_protocol as kb
    try:
        return kb.check_code_version(cells, tag="отчёт")
    except kb.ProtocolError as e:
        if not allow_mixed:
            raise SystemExit(
                str(e) + "\n  Если расхождение заведомо не влияет на "
                "траектории (изменились только записываемые поля), повторите "
                "с --allow-mixed-versions и укажите это в отчёте.")
        vers = sorted({json.dumps(c.get("code_version"), sort_keys=True)
                       for c in cells if c.get("code_version")})
        n_no = sum(1 for c in cells if not c.get("code_version"))
        print(f"\n  ВНИМАНИЕ: отчёт смешивает {len(vers)} версии кода"
              + (f" и {n_no} ячеек без записи версии" if n_no else "")
              + ".\n  Это разрешено явным флагом; расхождение обязано быть "
                "описано в тексте отчёта.")
        return None


def collect(root, ladder=None, allow_mixed=False):
    det = load_cells(root, "eval_d1_det")
    g0 = load_cells(root, "eval_g0")
    if not det or not g0:
        raise SystemExit(f"в {root} нет оценочных ячеек d1_det или g0")
    ng0 = noise_key(g0)
    if ng0[0] != "eval":
        raise SystemExit(f"g0 снята в режиме шума {ng0[0]!r}, а не 'eval': "
                         f"сравнивать с g_rl нельзя")
    d1_seed = det[0].get("d1_seed")
    sig = {float(c["sigma"]) for c in g0}
    if len(sig) != 1:
        raise SystemExit(f"у g0 разные sigma: {sorted(sig)}")
    sigma = sig.pop()
    check_provenance(det, arm="baseline", d1_seed=d1_seed, sigma=0.0,
                     step_index=0)
    check_provenance(g0, arm="g0", d1_seed=d1_seed, sigma=sigma, step_index=0)
    e_det, e_g0 = episodes(det), episodes(g0)

    steps, means, noise_ok = {}, {}, {}
    for d in sorted(glob.glob(os.path.join(root, "eval_g_rl_step*"))):
        k = int(os.path.basename(d).replace("eval_g_rl_step", ""))
        if ladder and k not in ladder:
            continue
        cells = load_cells(root, os.path.basename(d))
        if not cells:
            continue
        if noise_key(cells) != ng0:
            raise SystemExit(
                f"шаг {k}: метаданные потока шума {noise_key(cells)} против "
                f"{ng0} у g0")
        check_provenance(cells, arm="g_rl", d1_seed=d1_seed, sigma=sigma,
                         step_index=k)
        # ФАКТИЧЕСКИЕ реализации, а не только совпадение сида
        noise_ok[k] = check_eps_streams(cells, g0, label_a=f"g_rl шаг {k}",
                                        label_b="g0")
        steps[k] = episodes(cells)
    for d in sorted(glob.glob(os.path.join(root, "eval_g_rl_mean_step*"))):
        k = int(os.path.basename(d).replace("eval_g_rl_mean_step", ""))
        if ladder and k not in ladder:
            continue
        cells = load_cells(root, os.path.basename(d))
        if not cells:
            continue
        check_provenance(cells, arm="g_rl_mean", d1_seed=d1_seed, sigma=0.0,
                         step_index=k)
        means[k] = episodes(cells)
    if not steps:
        raise SystemExit(f"в {root} нет ни одной оценки g_rl")

    all_cells = list(det) + list(g0)
    for d in sorted(glob.glob(os.path.join(root, "eval_g_rl*step*"))):
        all_cells += load_cells(root, os.path.basename(d))
    check_versions(all_cells, allow_mixed)

    tr = train_states(root)
    leak = sorted(set(e_g0) & tr)
    if leak:
        raise SystemExit(
            f"оценочные состояния {leak[:8]} ({len(leak)}) встречаются среди "
            f"обучающих: улучшение заявлялось бы на тех же состояниях, на "
            f"которых собран градиент")
    return dict(det=e_det, g0=e_g0, steps=steps, means=means, sigma=[sigma],
                noise=dict(mode=ng0[0], eval_eps_seed=ng0[1], salt=ng0[2]),
                noise_checked=noise_ok, d1_seed=d1_seed,
                rl_seed=g0[0].get("rl_seed"), n_eval=len(e_g0),
                train_states=len(tr))


def report(data, head_tag=""):
    print(f"\n  RL-SMOKE{' ' + head_tag if head_tag else ''}: оценка на "
          f"{data['n_eval']} отложенных эпизодах, sigma {data['sigma']}, "
          f"поток шума {data['noise']}")
    print(f"    успех d1_det "
          f"{100 * sum(1 for k in data['det'] if data['det'][k][0]) / len(data['det']):.2f}%"
          f", успех g0 "
          f"{100 * sum(1 for k in data['g0'] if data['g0'][k][0]) / len(data['g0']):.2f}%")
    # ПЕРВИЧНАЯ МЕТРИКА — СРЕДНЕЕ ОБУЧЕННОЙ ПОЛИТИКИ ПРОТИВ D1. Именно
    # детерминированное среднее применяется при выводе; шумная рука отвечает на
    # вопрос про политику, которую мы не собираемся применять.
    if data.get("means"):
        print(f"\n  ПЕРВИЧНО: g_rl_mean - d1_det (исполняется среднее)")
        print(f"    {'шаг':>4}{'успех':>9}{'эффект':>9}{'± se':>8}{'p':>7}"
              f"{'восст':>7}{'потер':>7}{'дискорд':>9}")
        for k in sorted(data["means"]):
            vm = paired(data["means"][k], data["det"],
                        label_a=f"g_rl_mean шаг {k}", label_b="d1_det")
            print(f"    {k:>4}{100 * vm['success_a']:>8.2f}%"
                  f"{100 * vm['effect']:>+8.2f}{100 * vm['se']:>8.2f}"
                  f"{vm['p_one_sided']:>7.3f}{vm['recovered']:>7}"
                  f"{vm['lost']:>7}{100 * vm['discord']:>8.2f}%")
    print(f"\n  ВТОРИЧНО (диагностика обучения шумной политики):")
    print(f"    {'шаг':>4}{'успех g_rl':>12}{'g_rl-g0':>10}{'± se':>8}"
          f"{'p':>7}{'восст':>7}{'потер':>7}{'дискорд':>9}{'g_rl-d1_det':>13}")
    rows = {}
    for k in sorted(data["steps"]):
        e = data["steps"][k]
        vs_g0 = paired(e, data["g0"], label_a=f"g_rl шаг {k}", label_b="g0")
        vs_det = paired(e, data["det"], label_a=f"g_rl шаг {k}",
                        label_b="d1_det")
        rows[k] = dict(vs_g0=vs_g0, vs_det=vs_det)
        print(f"    {k:>4}{100 * vs_g0['success_a']:>11.2f}%"
              f"{100 * vs_g0['effect']:>+9.2f}{100 * vs_g0['se']:>8.2f}"
              f"{vs_g0['p_one_sided']:>7.3f}{vs_g0['recovered']:>7}"
              f"{vs_g0['lost']:>7}{100 * vs_g0['discord']:>8.2f}%"
              f"{100 * vs_det['effect']:>+12.2f}")
    # ИТОГ — ПО ПЕРВИЧНОЙ МЕТРИКЕ, если она есть
    if data.get("means"):
        mrows = {k: paired(data["means"][k], data["det"],
                           label_a=f"g_rl_mean шаг {k}", label_b="d1_det")
                 for k in sorted(data["means"])}
        kb_ = max(mrows, key=lambda k: mrows[k]["effect"])
        mb = mrows[kb_]
        ok_m = bool(mb["effect"] > 0 and mb["p_one_sided"] < 0.05)
        print(f"\n    ПЕРВИЧНЫЙ ИТОГ: лучший шаг {kb_}, "
              f"g_rl_mean - d1_det {100 * mb['effect']:+.2f} пп ± "
              f"{100 * mb['se']:.2f} (p={mb['p_one_sided']:.3f}) — "
              f"{'сигнал' if ok_m else 'НЕОТЛИЧИМО ОТ НУЛЯ'}")
        print(f"    динамика: " + ", ".join(
            f"шаг {k} {100 * mrows[k]['effect']:+.2f}" for k in sorted(mrows)))
    best = max(rows, key=lambda k: rows[k]["vs_g0"]["effect"])
    b = rows[best]["vs_g0"]
    # ЗНАКА НЕДОСТАТОЧНО. Порог по одному знаку объявлял бы сигнал в половине
    # случаев при полном его отсутствии: дискордантные пары при нулевом эффекте
    # делятся пополам, и перевес 9 против 7 — обычное подбрасывание монеты.
    ok = bool(b["effect"] > 0 and b["p_one_sided"] < 0.05)
    print(f"\n    лучший шаг по g_rl-g0: {best}, эффект "
          f"{100 * b['effect']:+.2f} пп ± {100 * b['se']:.2f} "
          f"(односторонний p={b['p_one_sided']:.3f}) — "
          f"{'сигнал' if ok else 'НЕОТЛИЧИМО ОТ НУЛЯ'}")
    print(f"    дискордантность {100 * b['discord']:.1f}%: при таком разбросе "
          f"на {b['n']} парах\n    различимым был бы эффект примерно от "
          f"{100 * 1.645 * b['se']:.1f} пп")
    if all(r["vs_g0"]["discord"] == 0 for r in rows.values()):
        print("    ВНИМАНИЕ: дискордантность ноль на всех шагах — действия на "
              "отложенных состояниях не изменились ни в одном эпизоде. Это "
              "«параметры поехали, поведение нет», а не отсутствие эффекта "
              "обучения.")
    return dict(rows=rows, best_step=best, signal=bool(ok))


def selftest():
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="k12i_")
    SEED, SIG, D1, RL = 777, 0.10, 0, 0

    def write(sub, arm, succ_fn, *, step=0, seed=SEED, sigma=None,
              eps_tag="e", policy="pol0", head="hd0", states=range(30, 35),
              tasks=(0, 1)):
        d = os.path.join(tmp, sub)
        os.makedirs(d, exist_ok=True)
        sigma = (0.0 if arm in ("baseline", "g_rl_mean")
                 else (SIG if sigma is None else sigma))
        for t in tasks:
            eps = [dict(state_id=i, init_hash_full=f"h{t}_{i}",
                        success=bool(succ_fn(t, i))) for i in states]
            json.dump(dict(
                stage="diag", arm=arm, sigma=sigma, d1_seed=D1, rl_seed=RL,
                step_index=step, head_sha1=head, policy_sha1=policy,
                eps_mode=("eval" if arm in ("g0", "g_rl") else "train"),
                eval_eps_seed=(seed if arm in ("g0", "g_rl") else None),
                eps_salt=(seed * 7 if arm in ("g0", "g_rl") else 11),
                eps_sha1_by_call=[f"{eps_tag}{t}_{c}" for c in range(6)],
                init_start=list(states)[0], task_ids=[t],
                state_ids=list(states), parity=dict(ok=True), episodes=eps),
                open(os.path.join(d, f"t{t}_s{list(states)[0]}.json"), "w"))

    def write_train(step, states=range(0, 5), tasks=(0, 1), n_envs=5):
        d = os.path.join(tmp, f"train_step{step}")
        os.makedirs(d, exist_ok=True)
        for t in tasks:
            json.dump(dict(task_ids=[t], state_ids=list(states),
                           n_envs=n_envs, step_index=step),
                      open(os.path.join(d,
                                        f"t{t}_s{list(states)[0]}.pt.meta.json"),
                           "w"))

    write("eval_d1_det", "baseline", lambda t, i: (t + i) % 4 != 0)
    write("eval_g0", "g0", lambda t, i: (t + i) % 5 != 0)
    write("eval_g_rl_step1", "g_rl", lambda t, i: (t + i) % 5 != 0 or i == 30,
          step=1, policy="pol1")
    write("eval_g_rl_mean_step1", "g_rl_mean", lambda t, i: (t + i) % 4 != 0,
          step=1, policy="pol1")
    write_train(0)
    data = collect(tmp, allow_mixed=True)
    res = report(data, "(тест)")
    r1 = res["rows"][1]["vs_g0"]
    assert r1["recovered"] >= 1 and r1["lost"] == 0 and r1["n"] == 10, r1
    assert data["noise_checked"][1]["blocks"] == 2, data["noise_checked"]
    assert isinstance(res["signal"], bool)
    # знака мало: 1 против 0 даёт p = 0.5
    assert res["signal"] is False, res

    def _expect(fn, needle):
        try:
            fn()
        except SystemExit as e:
            assert needle in str(e), f"ожидал «{needle}», получил: {e}"
            return
        raise AssertionError(f"отказа «{needle}» не было")

    # --- ОТРИЦАТЕЛЬНЫЕ ТЕСТЫ (пункт 4.6) ---------------------------------
    def _redo(sub, **kw):
        shutil.rmtree(os.path.join(tmp, sub), ignore_errors=True)
        write(sub, **kw)

    # 1. результат ДРУГОЙ sigma в том же каталоге
    _redo("eval_g_rl_step2", arm="g_rl", succ_fn=lambda t, i: True, step=2,
          sigma=0.03, policy="pol2")
    _expect(lambda: collect(tmp, allow_mixed=True), "sigma=0.03 вместо 0.1")
    shutil.rmtree(os.path.join(tmp, "eval_g_rl_step2"))

    # 2. другие ФАКТИЧЕСКИЕ реализации шума при тех же метаданных
    _redo("eval_g_rl_step2", arm="g_rl", succ_fn=lambda t, i: True, step=2,
          eps_tag="ДРУГОЙ", policy="pol2")
    _expect(lambda: collect(tmp, allow_mixed=True), "разные случайные числа, а не разные веса")
    shutil.rmtree(os.path.join(tmp, "eval_g_rl_step2"))

    # 3. разные policy_sha1 внутри одной руки
    write("eval_g_rl_step2", "g_rl", lambda t, i: True, step=2,
          policy="pol2", tasks=(0,))
    write("eval_g_rl_step2", "g_rl", lambda t, i: True, step=2,
          policy="ДРУГАЯ", tasks=(1,))
    _expect(lambda: collect(tmp, allow_mixed=True), "разные policy_sha1")
    shutil.rmtree(os.path.join(tmp, "eval_g_rl_step2"))

    # 4. неполный набор: у шага нет одного блока
    _redo("eval_g_rl_step2", arm="g_rl", succ_fn=lambda t, i: True, step=2,
          policy="pol2", tasks=(0,))
    _expect(lambda: collect(tmp, allow_mixed=True), "только у одной из рук")
    shutil.rmtree(os.path.join(tmp, "eval_g_rl_step2"))

    # 5. дубликат блока
    _redo("eval_g_rl_step2", arm="g_rl", succ_fn=lambda t, i: True, step=2,
          policy="pol2")
    d2 = os.path.join(tmp, "eval_g_rl_step2")
    shutil.copy(os.path.join(d2, "t0_s30.json"),
                os.path.join(d2, "t0_s30_копия.json"))
    _expect(lambda: collect(tmp, allow_mixed=True), "уже есть в")
    shutil.rmtree(d2)

    # 6. пересечение train и eval — по МЕТАДАННЫМ, а не по имени файла
    write_train(1, states=range(30, 35))
    _expect(lambda: collect(tmp, allow_mixed=True), "собран градиент")
    shutil.rmtree(os.path.join(tmp, "train_step1"))
    # и число сред берётся из меты: при n_envs=10 пересечение всё равно видно
    write_train(1, states=range(28, 38), n_envs=10)
    _expect(lambda: collect(tmp, allow_mixed=True), "собран градиент")
    shutil.rmtree(os.path.join(tmp, "train_step1"))

    # 7. номер шага не тот
    _redo("eval_g_rl_step2", arm="g_rl", succ_fn=lambda t, i: True, step=9,
          policy="pol2")
    _expect(lambda: collect(tmp, allow_mixed=True), "step_index=9, ожидалось 2")
    shutil.rmtree(os.path.join(tmp, "eval_g_rl_step2"))

    # 8. паритет не сошёлся
    d = os.path.join(tmp, "eval_g_rl_step2")
    write("eval_g_rl_step2", "g_rl", lambda t, i: True, step=2, policy="pol2")
    for f in glob.glob(os.path.join(d, "*.json")):
        o = json.load(open(f))
        o["parity"] = dict(ok=False, drift_from_d1=0.0)
        json.dump(o, open(f, "w"))
    _expect(lambda: collect(tmp, allow_mixed=True), "паритет не сошёлся")
    shutil.rmtree(d)

    # 9. чужая рука в каталоге шага
    _redo("eval_g_rl_step2", arm="g0", succ_fn=lambda t, i: True, step=2)
    _expect(lambda: collect(tmp, allow_mixed=True), "рука g0 вместо g_rl")
    shutil.rmtree(os.path.join(tmp, "eval_g_rl_step2"))

    # 10. непарные эпизоды в самой разности
    a = {(0, 30): (True, "h"), (0, 31): (False, "h")}
    b = {(0, 30): (False, "h")}
    _expect(lambda: paired(a, b, label_a="a", label_b="b"), "непарные эпизоды")
    # 11. хэши начального состояния различаются
    _expect(lambda: paired({(0, 30): (True, "h1")}, {(0, 30): (True, "h2")},
                           label_a="a", label_b="b"), "это разные состояния")

    assert collect(tmp, allow_mixed=True)["n_eval"] == 10
    print("\nсамопроверка k12i_smoke_report пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--ladder", default="")
    ap.add_argument("--allow-mixed-versions", action="store_true",
                    help="разрешить отчёт по ячейкам разных версий кода; "
                         "расхождение печатается и должно быть описано")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not a.root:
        ap.error("нужен --root, например data/k12i/s0")
    lad = [int(x) for x in a.ladder.split(",") if x.strip()] or None
    data = collect(a.root, lad, a.allow_mixed_versions)
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
