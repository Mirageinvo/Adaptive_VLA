#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15c: парный поведенческий анализ q0 против селектора ранга.

ФУНКЦИИ K-14q ИМПОРТИРУЮТСЯ, А НЕ КОПИРУЮТСЯ: загрузка артефактов k9h,
парность по (задача, состояние, сид) с поэпизодной сверкой хешей старта и
сида раскатки, кластерные средние, rescue/harm и стратифицированный по
задачам кластерный бутстрап — всё то, что уже проверено на §53.

ЕДИНИЦА — КЛАСТЕР (задача, init_state_id); один сид, раскатка
детерминирована. Все величины — с равным весом задач, как в K-14q.

ДВА РЕЖИМА И ДВА РАЗНЫХ РЕШЕНИЯ, ОБА ОБЪЯВЛЕНЫ ДО ДАННЫХ (03.10.2026):

    safety  ТЕХНИЧЕСКИЙ пилот на задачах 8-9. Проходит, если набор полон
            и парен, артефакты сняты одной версией харнесса, у каждой руки
            одна модель, действия конечны — и нет катастрофы: разность
            успеха не ниже CATASTROPHE (-0.15). Это НЕ вывод об
            эффективности: 50 кластеров для него мало.
    dev     250 кластеров. Зарегистрированный критерий селектора
            harm <= 0.3 * rescue применяется без изменений, при rescue > 0
            (иначе 0 <= 0 проходил бы тривиально). Разность успеха
            печатается с двусторонним 90 % интервалом.

КОДЫ: 0 — пройдено; 4 — не пройдено (катастрофа или критерий); 3 —
технический отказ набора. Отказ провенанса — с текстом.
"""
import argparse
import json
import os
import sys

import numpy as np

CATASTROPHE = -0.15
HARM_RATIO = 0.3
MODES = dict(safety=dict(tasks=[8, 9], states=list(range(25))),
             dev=dict(tasks=list(range(10)), states=list(range(25))))
SEEDS = [101]


def expected_clusters(mode):
    m = MODES[mode]
    return sorted((t, s) for t in m["tasks"] for s in m["states"])


def safety_verdict(technical, delta_sr, catastrophe=CATASTROPHE):
    if technical:
        return dict(code=3, outcome=f"технический отказ: {technical}")
    if delta_sr < catastrophe:
        return dict(code=4, outcome=(
            f"катастрофическое падение: разность успеха {delta_sr:+.3f} "
            f"ниже {catastrophe}"))
    return dict(code=0, outcome=(
        "технический пилот пройден; это НЕ вывод об эффективности"))


def dev_verdict(technical, rescue, harm, ratio=HARM_RATIO):
    if technical:
        return dict(code=3, outcome=f"технический отказ: {technical}")
    if rescue <= 0.0:
        return dict(code=4, outcome="rescue = 0: селектор не спас ни одного "
                                    "кластера, критерий не определён")
    ok = harm <= ratio * rescue + 1e-12
    return dict(code=0 if ok else 4, outcome=(
        f"harm {harm:.4f} {'<=' if ok else '>'} {ratio} * rescue "
        f"{rescue:.4f} = {ratio * rescue:.4f}"))


def pick_histogram(paths, arm):
    """Распределение выбранных рангов по артефактам руки k15c."""
    hist = np.zeros(8, np.int64)
    calls = 0
    for p in paths:
        d = json.load(open(p))
        if str(d.get("arm_label")) != arm:
            continue
        kp = d.get("k15c_picks")
        if kp is None:
            raise SystemExit(f"{p}: у руки {arm} нет k15c_picks")
        for k, v in kp["histogram"].items():
            hist[int(k)] += int(v)
        calls += int(kp["calls"])
    n = int(hist.sum())
    return dict(calls=calls, choices=n,
                histogram={int(i): int(c) for i, c in enumerate(hist)},
                share_rank0=(float(hist[0] / n) if n else None))


def finite_actions(paths):
    """Все сохранённые действия конечны; максимум |a| по каналам."""
    worst, bad = None, []
    for p in paths:
        d = json.load(open(p))
        npz = d.get("actions_npz")
        if not npz:
            bad.append(f"{os.path.basename(p)}: действия не сохранены")
            continue
        with np.load(os.path.join(os.path.dirname(p), npz),
                     allow_pickle=True) as z:
            a = np.asarray(z["actions"], np.float64)
        if not np.isfinite(a).all():
            bad.append(f"{os.path.basename(p)}: nan/inf в действиях")
            continue
        m = np.abs(a.reshape(-1, a.shape[-1])).max(0)
        worst = m if worst is None else np.maximum(worst, m)
    return bad, (None if worst is None else [float(x) for x in worst])


def selftest():
    assert len(expected_clusters("safety")) == 50
    assert len(expected_clusters("dev")) == 250
    assert safety_verdict(["x"], 0.0)["code"] == 3
    assert safety_verdict([], -0.2)["code"] == 4
    assert safety_verdict([], -0.15)["code"] == 0      # ровно на границе
    assert safety_verdict([], 0.05)["code"] == 0
    assert dev_verdict([], 0.0, 0.0)["code"] == 4       # тривиальный 0 <= 0
    assert dev_verdict([], 0.10, 0.03)["code"] == 0
    assert dev_verdict([], 0.10, 0.031)["code"] == 4
    assert dev_verdict(["x"], 0.10, 0.0)["code"] == 3
    print("самопроверка k15c_behavior пройдена")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import k14q_behavior as kb
    ap = argparse.ArgumentParser(
        description="K-15c: q0 против селектора, парно")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=tuple(MODES), default=None)
    ap.add_argument("--arts", nargs="*", default=[])
    ap.add_argument("--base", default="q0")
    ap.add_argument("--cand", default="k15c")
    ap.add_argument("--allow-partial", action="store_true",
                    help="только для смоука: неполный набор не является "
                         "результатом")
    ap.add_argument("--allow-preflight", action="store_true",
                    help="принять артефакты предполётной проверки механики. "
                         "Только вместе с --allow-partial: это не результат")
    ap.add_argument("--boot", type=int, default=kb.N_BOOT)
    ap.add_argument("--out", default="")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not a.mode or not a.arts or not a.out:
        raise SystemExit("нужны --mode, --arts и --out")
    if os.path.exists(a.out) and not a.overwrite:
        raise SystemExit(f"{a.out} уже существует")

    if a.allow_preflight and not a.allow_partial:
        raise SystemExit("--allow-preflight только вместе с --allow-partial")
    # АРТЕФАКТЫ ПРЕДПОЛЁТНОЙ ПРОВЕРКИ НЕ ЯВЛЯЮТСЯ РЕЗУЛЬТАТОМ: без явного
    # разрешения они отвергаются, чтобы не попасть в настоящий анализ.
    for p_ in a.arts:
        if ((json.load(open(p_)).get("joint") or {}).get("preflight")
                and not a.allow_preflight):
            raise SystemExit(f"{p_}: артефакт предполётной проверки")
    obs, meta = kb.load(a.arts)
    arms = [a.base, a.cand]
    for nm in arms:
        if nm not in meta:
            raise SystemExit(f"нет руки {nm}; есть {sorted(meta)}")
    clusters, seeds = kb.align(obs, arms)
    technical = []
    if not a.allow_partial:
        if clusters != expected_clusters(a.mode):
            technical.append(f"кластеров {len(clusters)} из "
                             f"{len(expected_clusters(a.mode))}")
        if seeds != SEEDS:
            technical.append(f"сиды {seeds}, зарегистрирован {SEEDS}")
    shas = set()
    for nm in arms:
        shas |= set(meta[nm]["script_shas"])
    if len(shas) > 1:
        technical.append(f"артефакты сняты разными версиями k9h: "
                         f"{sorted(shas)}")
    bad, act_max = finite_actions(meta[a.cand]["files"])
    technical += bad

    sr_b = kb.cluster_means(obs, a.base, clusters, seeds)
    sr_c = kb.cluster_means(obs, a.cand, clusters, seeds)
    res, hrm = kb.pair_rates(obs, a.base, a.cand, clusters, seeds)
    d = sr_c - sr_b
    ci, _reps = kb.boot(dict(delta=d), clusters, n=int(a.boot),
                        qs=(kb.Q_LO, kb.Q_HI))
    point = dict(sr_base=kb.point(sr_b, clusters),
                 sr_cand=kb.point(sr_c, clusters),
                 rescue=kb.point(res, clusters), harm=kb.point(hrm, clusters),
                 delta=kb.point(d, clusters),
                 selector_ceiling=kb.point(kb.selector_ceiling(sr_b, sr_c),
                                           clusters))
    point["discord"] = point["rescue"] + point["harm"]
    per_task = {}
    for t, idx in kb.strata(clusters).items():
        per_task[int(t)] = dict(
            clusters=int(len(idx)), sr_base=float(sr_b[idx].mean()),
            sr_cand=float(sr_c[idx].mean()), rescue=float(res[idx].mean()),
            harm=float(hrm[idx].mean()))
    picks = pick_histogram(meta[a.cand]["files"], a.cand)
    verdict = (safety_verdict(technical, point["delta"]) if a.mode == "safety"
               else dev_verdict(technical, point["rescue"], point["harm"]))

    print(f"  кластеров {len(clusters)}, сиды {seeds}, версия k9h "
          f"{sorted(shas)}")
    print(f"  успех: {a.base} {point['sr_base']:.4f}, {a.cand} "
          f"{point['sr_cand']:.4f}; разность {point['delta']:+.4f}, 90 % "
          f"интервал [{ci['delta'][0]:+.4f}, {ci['delta'][1]:+.4f}]")
    print(f"  rescue {point['rescue']:.4f}, harm {point['harm']:.4f}, "
          f"discord {point['discord']:.4f}; потолок селектора "
          f"{point['selector_ceiling']:.4f}")
    print("  по задачам: " + "; ".join(
        f"{t}: {v['sr_base']:.2f}->{v['sr_cand']:.2f}"
        for t, v in per_task.items()))
    print(f"  выбранные ранги: {picks['histogram']}, ранг 0 у "
          f"{100 * (picks['share_rank0'] or 0):.1f} %")
    print(f"  максимум |действия| по каналам у {a.cand}: {act_max}")
    print(f"  ИСХОД ({a.mode}): {verdict['outcome']} (код {verdict['code']})")
    out = dict(kind="k15c_behavior", mode=a.mode, verdict=verdict,
               technical=technical, point=point,
               ci90_delta=ci["delta"], per_task=per_task, picks=picks,
               action_absmax_cand=act_max, clusters=len(clusters),
               seeds=seeds, script_sha1=sorted(shas),
               arms=dict(base=a.base, cand=a.cand),
               fingerprints={nm: sorted(meta[nm]["fingerprints"])
                             for nm in arms},
               partial=bool(a.allow_partial),
               thresholds=dict(catastrophe=CATASTROPHE,
                               harm_ratio=HARM_RATIO,
                               declared="до данных, 03.10.2026"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                exist_ok=True)
    tmp = a.out + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, allow_nan=False)
    os.replace(tmp, a.out)
    print(f"  сводка: {a.out}")
    return int(verdict["code"])


if __name__ == "__main__":
    sys.exit(main())
