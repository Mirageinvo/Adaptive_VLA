#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-12f: сборка финальной оценки и гейт. Ничего не выбирает, только считает.

ЧТО ДЕЛАЕТ. Читает ячейки финальной оценки (обе руки), проверяет ПОЛНОТУ и
условия исполнения, собирает пары по `init_hash_full`, считает
зарегистрированный гейт и описательные числа. Ни одного решения здесь не
принимается: sigma и чекпойнты уже запечатаны, набор состояний и правило гейта
зарегистрированы. Это важно буквально — скрипт, который умеет выбирать, умеет
выбрать удачное.

ПОЧЕМУ ВСЕ ПРОВЕРКИ ЗДЕСЬ, А НЕ В ГЛАЗАХ ЧИТАЮЩЕГО. Полнота (все реплики, все
задачи, все состояния, обе руки), условия исполнения каждой ячейки, печать
решений и запись об открытии final — всё это отказы, а не замечания. Результат
финальной оценки нельзя пересчитать, поэтому обнаруживать неполноту надо до
того, как число станет выводом статьи.

ДИСКОРДАНТНОСТЬ СЧИТАЕТСЯ И ПЕЧАТАЕТСЯ РЯДОМ С ЭФФЕКТОМ. По расчёту мощности
выполнимая область — «дискордантность не более чем на процентный пункт выше
эффекта», и эти два числа имеют смысл только вместе: эффект +5 пп при
дискордантности 20% означает, что политика меняет исходы вчетверо чаще, чем
улучшает, и такой результат не воспроизведётся.
"""
import argparse
import glob
import hashlib
import json
import os
import sys


def collect(patterns):
    """Ячейки по маскам путей. Дубликаты путей отбрасываются, содержимое — нет:
    повтор одной ячейки дважды обязан поймать `check_final_complete`."""
    paths = []
    for pat in patterns:
        paths += sorted(glob.glob(pat))
    seen, cells = set(), []
    for p in paths:
        rp = os.path.realpath(p)
        if rp in seen:
            continue
        seen.add(rp)
        obj = json.load(open(p))
        obj["_path"] = p
        obj["_sha1"] = hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]
        cells.append(obj)
    if not cells:
        raise SystemExit(f"по маскам {patterns} не найдено ни одной ячейки")
    return cells


def pair_table(proto, cells):
    """Описательные числа по парам: успех рук, восстановления, потери.

    Отдельно от гейта, потому что гейт даёт одну границу, а разобраться, откуда
    она взялась, можно только по восстановлениям и потерям. Повторяет логику
    сборки пар из `diffs_from_final`, но считает не разности, а исходы — если
    они разойдутся, это видно: effect из этой таблицы обязан совпасть с
    точечной оценкой гейта.
    """
    import k12b_protocol as kb
    want = list(proto["splits"]["final"])
    pol, base = {}, {}
    for c in cells:
        if c.get("stage") != "final":
            raise SystemExit(f"{c.get('_path')}: этап {c.get('stage')}, а не "
                             f"final — диагностика в гейт не входит")
        tgt = base if c.get("arm") == "baseline" else pol
        who = (int(c["d1_seed"]) if c.get("arm") == "baseline"
               else c["replica"])
        for t in c["task_ids"]:
            for e in c["episodes"]:
                tgt[(who, int(t), int(e["state_id"]))] = bool(e["success"])
    rows = {}
    for r in proto["replicas"]:
        rk, sd = kb.replica_key(r), int(r["d1_seed"])
        rec = los = n = np_ok = nb_ok = 0
        per_task = {}
        for t in proto["tasks"]:
            tr = tl = tn = 0
            for i in want:
                a, b = pol.get((rk, int(t), int(i))), base.get(
                    (sd, int(t), int(i)))
                if a is None or b is None:
                    continue
                tn += 1
                if a and not b:
                    tr += 1
                elif b and not a:
                    tl += 1
                np_ok += int(a)
                nb_ok += int(b)
            per_task[int(t)] = dict(n=tn, recovered=tr, lost=tl,
                                    effect=((tr - tl) / tn) if tn else None,
                                    discord=((tr + tl) / tn) if tn else None)
            rec += tr
            los += tl
            n += tn
        rows[rk] = dict(
            n_pairs=n, recovered=rec, lost=los,
            success_policy=(np_ok / n) if n else None,
            success_baseline=(nb_ok / n) if n else None,
            effect=((rec - los) / n) if n else None,
            discord=((rec + los) / n) if n else None, by_task=per_task)
    return rows


def diag_summary(cells):
    """Сводка ДИАГНОСТИЧЕСКИХ ячеек: доля провалов исходной D1 по задачам.

    Считает только то, ради чего диагностика и делается: потолок эффекта равен
    доле провалов, и по нему видно, есть ли на сюите что восстанавливать. Гейта
    здесь нет и быть не может — у этих ячеек нет ни протокола, ни печати
    решений.
    """
    bad = [c["_path"] for c in cells if c.get("stage") != "diag"]
    if bad:
        raise SystemExit(f"не диагностические ячейки в сводке диагностики: "
                         f"{bad[:5]}")
    by = {}
    for c in cells:
        key = (str(c.get("suite")), int(c["task_ids"][0]))
        r = by.setdefault(key, dict(n=0, ok=0, states=set(),
                                    d1_seed=c.get("d1_seed")))
        for e in c["episodes"]:
            r["n"] += 1
            r["ok"] += int(bool(e["success"]))
            r["states"].add(int(e["state_id"]))
    rows, by_suite = [], {}
    for (su, t), r in sorted(by.items()):
        p = r["ok"] / r["n"]
        rows.append(dict(suite=su, task_id=t, n=r["n"], success=p,
                         p_fail=1.0 - p, n_states=len(r["states"])))
        s_ = by_suite.setdefault(su, dict(n=0, ok=0, tasks=0))
        s_["n"] += r["n"]
        s_["ok"] += r["ok"]
        s_["tasks"] += 1
    suites = {}
    for su, s_ in sorted(by_suite.items()):
        p = s_["ok"] / s_["n"]
        # ПОТОЛОК ЭФФЕКТА РАВЕН ДОЛЕ ПРОВАЛОВ: восстановить больше, чем
        # провалено, нельзя, поэтому доля провалов ниже целевого эффекта сразу
        # закрывает сюиту, сколько бы эпизодов ни набирать
        suites[su] = dict(tasks=s_["tasks"], episodes=s_["n"], success=p,
                          p_fail=1.0 - p, delta_ceiling=1.0 - p,
                          usable_for_5pp=bool(1.0 - p > 0.05))
    return dict(by_task=rows, by_suite=suites)


def report_diag(res):
    print(f"\n  ДИАГНОСТИКА: исходная детерминированная D1 на нетронутых "
          f"сюитах.\n  Потолок эффекта равен доле провалов — это и есть "
          f"критерий пригодности сюиты.")
    print(f"    {'сюита':<10}{'задач':>6}{'эпиз':>6}{'успех':>9}{'провалов':>10}"
          f"{'потолок':>9}  пригодна для +5 пп")
    for su, r in sorted(res["by_suite"].items()):
        print(f"    {su:<10}{r['tasks']:>6}{r['episodes']:>6}"
              f"{100 * r['success']:>8.2f}%{100 * r['p_fail']:>9.2f}%"
              f"{100 * r['delta_ceiling']:>8.2f}%"
              f"{'   да' if r['usable_for_5pp'] else '   НЕТ'}")
    print(f"\n    {'сюита':<10}{'задача':>7}{'эпиз':>6}{'успех':>9}"
          f"{'провалов':>10}")
    for r in res["by_task"]:
        print(f"    {r['suite']:<10}{r['task_id']:>7}{r['n']:>6}"
              f"{100 * r['success']:>8.2f}%{100 * r['p_fail']:>9.2f}%")
    hard = [r for r in res["by_task"] if r["p_fail"] <= 0.0]
    easy = [r for r in res["by_task"] if r["success"] <= 0.2]
    if hard:
        print(f"\n    задач без провалов: {len(hard)} "
              f"{[(r['suite'], r['task_id']) for r in hard][:8]} — на них "
              f"восстанавливать нечего")
    if easy:
        print(f"    задач с успехом <=20%: {len(easy)} "
              f"{[(r['suite'], r['task_id']) for r in easy][:8]} — там речь "
              f"уже не о поправке к работающей политике")


def run(proto, cells, *, decisions_path, final_open_path):
    import k12b_protocol as kb
    probs = []
    for c in cells:
        probs += kb.check_execution(proto, c, os.path.basename(c["_path"]))
    if probs:
        raise SystemExit("условия исполнения ячеек не совпали с протоколом:\n"
                         "  - " + "\n  - ".join(probs[:15]))
    info = kb.check_final_complete(proto, cells,
                                  decisions_path=decisions_path,
                                  final_open_path=final_open_path)
    diffs = kb.diffs_from_final(proto, cells)
    g = kb.gate(proto, diffs)
    tab = pair_table(proto, cells)
    # СХОДИМОСТЬ ДВУХ ПУТЕЙ СЧЁТА: точечная оценка гейта обязана совпасть со
    # средним эффектом из таблицы пар. Расхождение означало бы, что пары
    # собраны по-разному, и какое из чисел верно, было бы неизвестно.
    for i, rk in enumerate([kb.replica_key(r) for r in proto["replicas"]]):
        a, b = g["per_replica_point"][i], tab[rk]["effect"]
        if b is None or abs(a - b) > 1e-9:
            raise SystemExit(f"реплика {rk}: гейт видит эффект {a}, таблица пар "
                             f"{b} — пары собраны по-разному")
    dec = kb.load_decisions(decisions_path, proto)
    # ДИСКОРДАНТНОСТЬ СВЕРХ ЗАРЕГИСТРИРОВАННОГО ПРЕДЕЛА — не отказ гейта, но
    # и не мелочь: расчёт мощности выполним только при «дискордантность ≈
    # эффект», и превышение означает, что политика меняет исходы заметно чаще,
    # чем улучшает, так что результат вряд ли воспроизведётся
    over = {rk: r["discord"] for rk, r in tab.items()
            if r["discord"] is not None
            and r["discord"] > float(proto["discord_max"]) + 1e-12}
    return dict(
        discord_over_registered=over,
        protocol_sha1=proto["sha1"], decisions_sha1=dec["sha1"],
        sigma=dec["sigma"], gate=g, completeness=info, pairs=tab,
        delta_target=proto["delta_target"],
        discord_max=proto["discord_max"],
        cells=[dict(path=c["_path"], sha1=c["_sha1"], arm=c.get("arm"),
                    replica=c.get("replica"), d1_seed=c.get("d1_seed"),
                    tasks=c["task_ids"], n_states=len(c["state_ids"]))
               for c in cells],
        script_sha1=hashlib.sha1(
            open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12],
        protocol_script_sha1=hashlib.sha1(
            open(kb.__file__, "rb").read()).hexdigest()[:12])


def report(res):
    g = res["gate"]
    print(f"\n  ГЕЙТ. Зарегистрировано правило «{g['registered']}», "
          f"n_boot={g['n_boot']}, alpha={g['alpha']}, сид {g['seed']}.")
    print(f"    {'правило':<10}{'нижняя граница':>16}{'проходит':>10}")
    for rule in ("mean", "mean_rep", "all"):
        mark = " <- зарегистрировано" if rule == g["registered"] else ""
        print(f"    {rule:<10}{100 * g['lower'][rule]:>15.2f} пп"
              f"{str(g['values'][rule]):>10}{mark}")
    print(f"    наивное среднее границ {100 * g['naive_mean_of_lowers']:.2f} пп "
          f"— НЕ граница, приведено только для сверки с прежним расчётом")
    print(f"\n  ПО РЕПЛИКАМ. Эффект и дискордантность имеют смысл только "
          f"вместе:\n  зарегистрированный эффект "
          f"{100 * res['delta_target']:.0f} пп, предел дискордантности "
          f"{100 * res['discord_max']:.0f}%.")
    print(f"    {'реплика':<10}{'пар':>6}{'успех RL':>10}{'успех D1':>10}"
          f"{'восст':>7}{'потер':>7}{'эффект':>9}{'дискорд':>9}")
    for rk, r in sorted(res["pairs"].items()):
        print(f"    {rk:<10}{r['n_pairs']:>6}"
              f"{100 * r['success_policy']:>9.2f}%"
              f"{100 * r['success_baseline']:>9.2f}%"
              f"{r['recovered']:>7}{r['lost']:>7}"
              f"{100 * r['effect']:>+8.2f}{100 * r['discord']:>8.2f}%")
    over = res.get("discord_over_registered") or {}
    if over:
        print(f"\n  ВНИМАНИЕ: дискордантность выше зарегистрированного предела "
              f"{100 * res['discord_max']:.0f}% у реплик "
              + ", ".join(f"{k} ({100 * v:.2f}%)"
                          for k, v in sorted(over.items()))
              + ".\n  Гейт это не отменяет, но выполнимая область расчёта "
                "мощности — «дискордантность\n  не более чем на пункт выше "
                "эффекта»; вне неё результат вряд ли воспроизведётся.")
    print(f"\n  ИТОГ: {'ПРОЙДЕН' if g['passed'] else 'НЕ ПРОЙДЕН'} по "
          f"зарегистрированному правилу «{g['registered']}»: нижняя граница "
          f"{100 * g['lower'][g['registered']]:+.2f} пп.")


def selftest():
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import k12b_protocol as kb

    p = kb._proto_ok()
    tmp = tempfile.mkdtemp(prefix="k12f_")
    cks = {}
    for r in p["replicas"]:
        rk = kb.replica_key(r)
        fp = os.path.join(tmp, f"ck_{rk}.pt")
        open(fp, "wb").write(f"веса {rk}".encode())
        cks[rk] = dict(path=fp, sha1=kb._sha12(fp))
    evp = os.path.join(tmp, "dev.json")
    json.dump(dict(sigma=0.03), open(evp, "w"))
    dpath = os.path.join(tmp, "dec.json")
    dec = kb.seal_decisions(dpath, p, sigma=0.03, checkpoints=cks,
                            dev_evidence=dict(source=evp, sha1=kb._sha12(evp)))
    fpath = os.path.join(tmp, "fo.json")
    kb.open_final(fpath, p, dec)

    fin = list(p["splits"]["final"])
    reps = [kb.replica_key(r) for r in p["replicas"]]

    def base_ok(t, i):
        return (t + i) % 5 != 0

    def pol_ok(rk, t, i):
        sd = reps.index(rk)
        if not base_ok(t, i):
            return (i + t + sd) % 3 != 0
        return (i + 2 * t + sd) % 29 != 0

    def write(arm, who, t):
        eps = [dict(state_id=int(i), init_hash_full=f"h{t}_{i}",
                    success=bool(base_ok(t, i) if arm == "baseline"
                                 else pol_ok(who, t, i)))
               for i in fin]
        cell = dict(p["execution"])
        cell.update(protocol_sha1=p["sha1"], arm=arm, stage="final",
                    task_ids=[t], state_ids=fin, episodes=eps,
                    n_episodes_per_task=len(fin))
        if arm == "baseline":
            cell.update(replica=None, d1_seed=int(who), sigma=0.0,
                        ckpt_sha1=p["d1_checkpoints"][str(int(who))]["sha1"])
        else:
            sd = kb.replica_seeds(p, who)[0]
            cell.update(replica=who, d1_seed=sd, sigma=0.03,
                        ckpt_sha1=cks[who]["sha1"])
        fp = os.path.join(tmp, f"cell_{arm}_{who}_{t}.json")
        json.dump(cell, open(fp, "w"), ensure_ascii=False)
        return fp

    for rk in reps:
        for t in p["tasks"]:
            write("policy", rk, t)
    for sd in (0, 1):
        for t in p["tasks"]:
            write("baseline", sd, t)

    cells = collect([os.path.join(tmp, "cell_*.json")])
    assert len(cells) == 6 * len(p["tasks"]), len(cells)
    res = run(p, cells, decisions_path=dpath, final_open_path=fpath)
    report(res)
    assert set(res["pairs"]) == set(reps)
    for rk, r in res["pairs"].items():
        assert r["n_pairs"] == len(p["tasks"]) * len(fin), r
        # эффект и дискордантность согласованы по определению
        assert abs(r["effect"] - (r["recovered"] - r["lost"]) / r["n_pairs"]) < 1e-12
        assert abs(r["discord"] - (r["recovered"] + r["lost"]) / r["n_pairs"]) < 1e-12
        # успех политики минус успех опоры — это и есть эффект
        assert abs((r["success_policy"] - r["success_baseline"])
                   - r["effect"]) < 1e-12, r

    def _expect(fn, needle):
        try:
            fn()
        except (SystemExit, kb.ProtocolError) as e:
            assert needle in str(e), f"ожидал «{needle}», получил: {e}"
            return
        raise AssertionError(f"отказа «{needle}» не было")

    # НЕПОЛНОТА: нет одной задачи у одной реплики
    part = [c for c in cells
            if not (c.get("replica") == reps[0] and c["task_ids"] == [
                p["tasks"][0]])]
    _expect(lambda: run(p, part, decisions_path=dpath, final_open_path=fpath),
            "состояний 0 из 80")
    # нет опорной руки
    _expect(lambda: run(p, [c for c in cells if c.get("arm") != "baseline"],
                        decisions_path=dpath, final_open_path=fpath),
            "без опорной руки")
    # ячейка с другими условиями исполнения
    bad = [dict(c) for c in cells]
    bad[0] = dict(bad[0], horizon=3)
    _expect(lambda: run(p, bad, decisions_path=dpath, final_open_path=fpath),
            "horizon=3")
    # правленая печать решений
    tamp = json.load(open(dpath))
    tamp["sigma"] = 0.10
    tp = os.path.join(tmp, "dec_bad.json")
    json.dump(tamp, open(tp, "w"))
    _expect(lambda: run(p, cells, decisions_path=tp, final_open_path=fpath),
            "файл правили после печати")
    # пустая маска
    _expect(lambda: collect([os.path.join(tmp, "нет_*.json")]),
            "не найдено ни одной ячейки")

    # ДИАГНОСТИЧЕСКАЯ ЯЧЕЙКА В ГЕЙТ НЕ ВХОДИТ, хотя у неё arm=baseline
    dg = dict(cells[-1])
    dg.update(stage="diag", protocol_sha1=None, _path="diag.json")
    # отказ приходит раньше — на проверке этапа в check_run, и это нормально:
    # важно, что диагностика не доходит до сборки пар ни одним путём
    _expect(lambda: run(p, cells + [dg], decisions_path=dpath,
                        final_open_path=fpath), "этап 'diag' не из")
    _expect(lambda: kb.diffs_from_final(p, [dg]), "на этапе diag, а не final")
    # а сводка диагностики считает долю провалов и отвергает ячейки final
    dgs = []
    for t in (0, 1):
        eps = [dict(state_id=int(i), init_hash_full=f"d{t}_{i}",
                    success=bool((t + i) % 4)) for i in range(10)]
        dgs.append(dict(stage="diag", arm="baseline", suite="object",
                        task_ids=[t], state_ids=list(range(10)), sigma=0.0,
                        d1_seed=0, episodes=eps, _path=f"d{t}.json"))
    ds = diag_summary(dgs)
    assert ds["by_suite"]["object"]["tasks"] == 2, ds
    assert abs(ds["by_suite"]["object"]["p_fail"]
               - sum(1 for c in dgs for e in c["episodes"]
                     if not e["success"]) / 20) < 1e-12, ds
    assert ds["by_suite"]["object"]["usable_for_5pp"] is True
    report_diag(ds)
    _expect(lambda: diag_summary(cells), "не диагностические ячейки")

    # НУЛЕВОЙ ЭФФЕКТ: та же политика, что опора -> гейт не проходит
    for f in glob.glob(os.path.join(tmp, "cell_policy_*.json")):
        os.remove(f)
    for rk in reps:
        for t in p["tasks"]:
            eps = [dict(state_id=int(i), init_hash_full=f"h{t}_{i}",
                        success=bool(base_ok(t, i))) for i in fin]
            cell = dict(p["execution"])
            cell.update(protocol_sha1=p["sha1"], arm="policy", stage="final",
                        task_ids=[t], state_ids=fin, episodes=eps,
                        n_episodes_per_task=len(fin), replica=rk,
                        d1_seed=kb.replica_seeds(p, rk)[0], sigma=0.03,
                        ckpt_sha1=cks[rk]["sha1"])
            json.dump(cell, open(os.path.join(
                tmp, f"cell_policy_{rk}_{t}.json"), "w"), ensure_ascii=False)
    zero = run(p, collect([os.path.join(tmp, "cell_*.json")]),
               decisions_path=dpath, final_open_path=fpath)
    assert zero["gate"]["passed"] is False, zero["gate"]["lower"]
    assert all(r["discord"] == 0.0 for r in zero["pairs"].values())
    assert zero["discord_over_registered"] == {}, zero[
        "discord_over_registered"]
    # а в первом прогоне дискордантность была выше предела и это отмечено
    assert set(res["discord_over_registered"]) == set(reps), res[
        "discord_over_registered"]
    print("\nсамопроверка k12f_final_gate пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--protocol", default="data/k12b/protocol.json")
    ap.add_argument("--decisions", default="data/k12b/decisions.json")
    ap.add_argument("--final-open", default="data/k12b/final_open.json")
    ap.add_argument("--cells", action="append", default=[],
                    help="маска путей к ячейкам final; можно несколько раз")
    ap.add_argument("--diag", action="store_true",
                    help="сводка диагностических ячеек вместо гейта")
    ap.add_argument("--out", default="data/k12b/final_gate.json")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.diag:
        cells = collect(args.cells or ["data/k12g/diag/*.json"])
        res = diag_summary(cells)
        report_diag(res)
        out = args.out if args.out != "data/k12b/final_gate.json" \
            else "data/k12g/diag_summary.json"
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".",
                    exist_ok=True)
        json.dump(res, open(out + ".tmp", "w"), ensure_ascii=False, indent=1,
                  default=str)
        os.replace(out + ".tmp", out)
        print(f"\n  сохранено: {out}")
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.abspath("experiments"))
    import k12b_protocol as kb

    proto = kb.load_protocol(args.protocol)
    cells = collect(args.cells or ["data/k12b/final/*.json"])
    print(f"  ячеек {len(cells)}: политика "
          f"{sum(1 for c in cells if c.get('arm') != 'baseline')}, опора "
          f"{sum(1 for c in cells if c.get('arm') == 'baseline')}")
    res = run(proto, cells, decisions_path=args.decisions,
              final_open_path=args.final_open)
    report(res)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    tmp = args.out + ".tmp"
    json.dump(res, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"\n  сохранено: {args.out}")
    return 0 if res["gate"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main() or 0)
