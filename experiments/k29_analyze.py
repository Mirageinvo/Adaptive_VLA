#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-29: парный анализ final_RL − init_control.

Только проверенные блоки (k29_harness.outcomes; неполный набор — код 3).
Основное (по сиду): задачи 0/6/8, старты 30…44 — 45 пар; SR с равными
весами задач, rescue (init провал -> final успех) / harm, точный Мак-Немар;
Holm по двум p; совместный парный бутстрэп: одни и те же индексы состояний
внутри задачи для обеих политик и обоих сидов (2×45 не объединяются в 90).
Сохранение навыков: задачи 1/7 (30 пар на сид) — отдельно, описательно.
Все пять задач — описательно. Исторический K-25 на тех же состояниях —
описательно (не основной контроль). Валидация 20…24 и бюджет — из отчётов
обучения. SR и p не влияют на код завершения.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k15h_analyze as ha  # noqa: E402

OUT = "reports/k29/analysis.json"
LATENCY = "reports/k29/latency.json"
PRIMARY = (0, 6, 8)
KEEP = (1, 7)
BOOT = dict(n=10000, seed=20261102)


def states_of(outc, tasks):
    return sorted(k for k in outc if k[0] in tasks)


def paired(a, b, states, boot=BOOT):
    """a, b: {(t, i): bool}. -> SR, Δ, rescue/harm, Мак-Немар, интервал."""
    gain = sum(1 for k in states if a[k] and not b[k])
    loss = sum(1 for k in states if b[k] and not a[k])
    sa, sb = ha.equal_weight(a, states), ha.equal_weight(b, states)
    by = {}
    for k in states:
        by.setdefault(k[0], []).append(k)
    rng = np.random.default_rng(boot["seed"])
    D = {t: np.array([float(a[k]) - float(b[k]) for k in v])
         for t, v in by.items()}
    bs = np.empty(boot["n"])
    for i in range(boot["n"]):
        bs[i] = np.mean([d[rng.integers(0, len(d), len(d))].mean()
                         for d in D.values()])
    return dict(sr_final=sa, sr_init=sb, d=sa - sb, rescue=gain, harm=loss,
                mcnemar_p=ha.mcnemar_p(gain, loss), n=len(states),
                ci95=[float(np.percentile(bs, 2.5)),
                      float(np.percentile(bs, 97.5))],
                per_task={str(t): [int(sum(a[k] for k in v)),
                                   int(sum(b[k] for k in v)), len(v)]
                          for t, v in by.items()})


def joint_boot(pairs, tasks, boot=BOOT):
    """pairs {seed: (a, b)}; одни индексы внутри задачи для всех сидов."""
    seeds = sorted(pairs)
    st = states_of(pairs[seeds[0]][0], tasks)
    by = {}
    for k in st:
        by.setdefault(k[0], []).append(k)
    D = {s: {t: np.array([float(pairs[s][0][k]) - float(pairs[s][1][k])
                          for k in v]) for t, v in by.items()} for s in seeds}
    est = float(np.mean([np.mean([D[s][t].mean() for t in by])
                         for s in seeds]))
    rng = np.random.default_rng(boot["seed"] + 1)
    bs = np.empty(boot["n"])
    for i in range(boot["n"]):
        idx = {t: rng.integers(0, len(v), len(v)) for t, v in by.items()}
        bs[i] = np.mean([np.mean([D[s][t][idx[t]].mean() for t in by])
                         for s in seeds])
    return dict(d=est, ci95=[float(np.percentile(bs, 2.5)),
                             float(np.percentile(bs, 97.5))])


def historical(p, seed):
    out = {}
    for k, v in (p.get("k25_outcomes") or {}).items():
        s, key = k.split("|")
        if int(s) == seed:
            t, i = key.split(":")
            out[(int(t), int(i))] = bool(v)
    return out


def read_latency(path=LATENCY):
    if not os.path.exists(path):
        return dict(status="не выполнено", why="латентность ожидается")
    return dict(status="выполнено", **json.load(open(path, encoding="utf-8")))


def per_seed(seed, init, final, p):
    st_p, st_k = states_of(init, PRIMARY), states_of(init, KEEP)
    st_all = states_of(init, PRIMARY + KEEP)
    if len(st_p) != 45 or len(st_k) != 30:
        raise SystemExit(f"s{seed}: состояний {len(st_p)}/{len(st_k)}")
    o = dict(seed=seed, primary=paired(final, init, st_p),
             keep=paired(final, init, st_k),
             all5=dict(sr_final=ha.equal_weight(final, st_all),
                       sr_init=ha.equal_weight(init, st_all)))
    h = historical(p, seed)
    if h and all(k in h for k in st_all):
        o["k25_historical"] = dict(
            sr_primary=ha.equal_weight(h, st_p),
            sr_keep=ha.equal_weight(h, st_k),
            init_vs_hist=paired(init, h, st_all),
            note="описательно: исторический путь FP16, не основной контроль")
    tr = f"reports/k29/train_s{seed}.json"
    if os.path.exists(tr):
        r = json.load(open(tr, encoding="utf-8"))
        o["training"] = dict(
            validation={k: v["sr"] for k, v in r["validation"].items()},
            updates=r["updates"], accepted=r["accepted_updates"],
            actor_changed=r["changed"],
            train_steps=r["ledger"]["train_spent"],
            crashes=len(r["ledger"]["crashes"]),
            rollbacks=sum(1 for u in r["update_stats"]
                          if "KL" in str(u.get("stop"))),
            actor_steps=sum(int(u.get("accepted_steps", 0))
                            for u in r["update_stats"]),
            skipped=sum(1 for u in r["update_stats"]
                        if u.get("actor_skipped")),
            kl_ref_last=(r["update_stats"][-1].get("kl_ref_final")
                         if r["update_stats"] else None))
    return o


def summary(per, pairs):
    seeds = sorted(per)
    holm = ha.holm([per[s]["primary"]["mcnemar_p"] for s in seeds])
    d = {s: per[s]["primary"]["d"] for s in seeds}
    jb = joint_boot(pairs, PRIMARY)
    jk = joint_boot(pairs, KEEP)
    if all(v > 0 for v in d.values()):
        sign = "final выше init в обоих сидах"
    elif all(v < 0 for v in d.values()):
        sign = "final ниже init в обоих сидах"
    elif all(v == 0 for v in d.values()):
        sign = "равны в обоих сидах"
    else:
        sign = "знак различается между сидами"
    est = all(h < 0.05 for h in holm) and (jb["ci95"][0] > 0 or
                                          jb["ci95"][1] < 0)
    resc = sum(per[s]["primary"]["rescue"] for s in seeds)
    harm = sum(per[s]["primary"]["harm"] for s in seeds)
    return dict(seeds=seeds, holm=dict(zip(seeds, holm)), d_per_seed=d,
                d_mean=float(np.mean(list(d.values()))), joint_primary=jb,
                joint_keep=jk, sign=sign, established=est,
                rescue_total=resc, harm_total=harm,
                text=f"{sign}; " + ("выигрыш статистически установлен" if est
                                     else "выигрыш не установлен"),
                note="разведочный пилот: задачи выбраны по прошлым "
                     "результатам; интервал условен на двух обученных "
                     "реализациях; нулевая разность не доказывает отсутствие "
                     "эффекта RL")


def report(per, summ, lat):
    f = lambda c: (f"SR final {100 * c['sr_final']:.1f}% / init "  # noqa: E731
                   f"{100 * c['sr_init']:.1f}%, Δ {100 * c['d']:+.1f} п.п. "
                   f"[{100 * c['ci95'][0]:+.1f}; {100 * c['ci95'][1]:+.1f}], "
                   f"rescue {c['rescue']}/harm {c['harm']}, Мак-Немар p "
                   f"{c['mcnemar_p']:.4f}")
    for s, o in sorted(per.items()):
        print(f"\n  SEED {s}")
        print(f"    основное (задачи 0/6/8, 45 пар): {f(o['primary'])}")
        print("      по задачам final/init: " + ", ".join(
            f"t{t} {v[0]}/{v[1]} из {v[2]}"
            for t, v in o["primary"]["per_task"].items()))
        print(f"    сохранение (задачи 1/7, 30 пар, описательно): "
              f"{f(o['keep'])}")
        print(f"    все пять задач (описательно): final "
              f"{100 * o['all5']['sr_final']:.1f}%, init "
              f"{100 * o['all5']['sr_init']:.1f}%")
        if "k25_historical" in o:
            h = o["k25_historical"]
            print(f"    исторический K-25 (FP16, описательно): 0/6/8 "
                  f"{100 * h['sr_primary']:.1f}%, 1/7 "
                  f"{100 * h['sr_keep']:.1f}%; init − история: "
                  f"{100 * h['init_vs_hist']['d']:+.1f} п.п. (+"
                  f"{h['init_vs_hist']['rescue']}/−"
                  f"{h['init_vs_hist']['harm']})")
        if "training" in o:
            t = o["training"]
            print(f"    обучение: обновлений {t['updates']} (принято "
                  f"{t['accepted']}, шагов actor {t['actor_steps']}, KL-остановок "
                  f"{t['rollbacks']}, пропусков "
                  f"{t['skipped']}), шагов {t['train_steps']}, аварий "
                  f"{t['crashes']}; actor "
                  + ("изменён" if t["actor_changed"] else "НЕ изменён")
                  + f"; валидация 20–24 {t['validation']}")
    if summ:
        print(f"\n  ДВА СИДА: Δ {', '.join(f's{s} {100 * v:+.1f}' for s, v in summ['d_per_seed'].items())}"
              f"; среднее {100 * summ['d_mean']:+.1f} п.п., совместный "
              f"бутстрэп [{100 * summ['joint_primary']['ci95'][0]:+.1f}; "
              f"{100 * summ['joint_primary']['ci95'][1]:+.1f}]; Holm "
              f"{summ['holm']}; rescue {summ['rescue_total']}/harm "
              f"{summ['harm_total']}")
        print(f"    сохранение 1/7: среднее "
              f"{100 * summ['joint_keep']['d']:+.1f} п.п. "
              f"[{100 * summ['joint_keep']['ci95'][0]:+.1f}; "
              f"{100 * summ['joint_keep']['ci95'][1]:+.1f}]")
        print(f"  ВЫВОД: {summ['text']}")
    print(f"  задержка: {lat.get('status')}")


def selftest():
    def mk(n_succ_per_task, tasks=(0, 1, 6, 7, 8)):
        o = {}
        for t in tasks:
            for j, i in enumerate(range(30, 45)):
                o[(t, i)] = j < n_succ_per_task.get(t, 10)
        return o
    init = mk({0: 10, 6: 12, 8: 5, 1: 14, 7: 15})
    fin = mk({0: 12, 6: 13, 8: 8, 1: 13, 7: 15})
    p = dict(k25_outcomes={})
    per = {s: per_seed(s, init, fin, p) for s in (0, 1)}
    pr = per[0]["primary"]
    assert pr["rescue"] == 6 and pr["harm"] == 0 and pr["n"] == 45
    assert abs(pr["d"] - (6 / 15) / 3) < 1e-12
    summ = summary(per, {s: (fin, init) for s in (0, 1)})
    assert abs(summ["joint_primary"]["d"] - pr["d"]) < 1e-12
    assert "выше" in summ["sign"]
    assert per[0]["keep"]["harm"] == 1
    print("самопроверка k29_analyze пройдена: rescue/harm, равные веса задач, "
          "совместный бутстрэп = средней Δ, сохранение 1/7 отдельно")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-29: анализ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import k29_harness as kh
    import k29_protocol as pc
    try:
        p = pc.load_prepare()
        per, pairs = {}, {}
        for s in a.seeds:
            init = kh.outcomes(s, "init", p)
            final = kh.outcomes(s, "final", p)
            per[s] = per_seed(s, init, final, p)
            pairs[s] = (final, init)
        summ = summary(per, pairs) if len(per) > 1 else None
    except SystemExit as e:
        print(f"  ОТКАЗ/НЕПОЛНО: {e.code}")
        return 3
    lat = read_latency()
    report(per, summ, lat)
    out = OUT if summ else f"reports/k29/analysis_s{'_'.join(map(str, a.seeds))}.json"
    json.dump(dict(per_seed=per, summary=summ, latency=lat,
                   resources=json.load(open(pc.RESOURCES)) if os.path.exists(
                       pc.RESOURCES) else None),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=str)
    print(f"  сводка: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
