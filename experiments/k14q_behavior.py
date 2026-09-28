#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14q: анализ §53 — success, rescue, harm, не-хужесть.

ВСЕ ПРАВИЛА ЗАРЕГИСТРИРОВАНЫ ДО ДАННЫХ (§53.1, §53.2, §53.4, §53.5, §53.6).
Здесь они только применяются.

ЕДИНИЦА КЛАСТЕРА — (задача, init_state_id). Execution seeds НЕ являются
независимыми наблюдениями: они дают повторные измерения одного начального
состояния и остаются ВНУТРИ кластера. Бутстрап пересэмплирует кластеры, иначе
интервал занижается ровно во столько раз, сколько повторов сделано.

БУТСТРАП СТРАТИФИЦИРОВАН ПО ЗАДАЧАМ. Сюита фиксирована, десять задач обязаны
сохранять равный вес; простая выборка из всех кластеров меняла бы между
репликами состав задач и добавляла дисперсию, которой в вопросе нет.

ДВА КРИТЕРИЯ, И ОНИ РАЗНЫЕ:
    УЛУЧШЕНИЕ    двусторонний 90% интервал разности, требуется q05 > 0;
                 численно равно P(rescue) - P(harm);
    НЕ-ХУЖЕСТЬ   односторонняя 95% нижняя граница, требуется q05 > -m,
                 m = 0.03. Проверяется у ВЫБРАННОЙ однопроходной системы, а
                 не у q0: иначе возможен исход «q1 лучше q0, q0 не хуже BAR,
                 а q0+q1 хуже BAR на 5 п.п.», при котором все формальные
                 критерии пройдены, а заявление не доказано ни для одной
                 предъявляемой системы.

ТРИ ИСХОДА НЕ-ХУЖЕСТИ, И ВСЕ ТРИ ЗАКОННЫ: доказана, опровергнута,
НЕОПРЕДЕЛЁННО. Третий объявлен заранее: при 200 кластерах и дискордантности
около 20% полуширина около 5.2 п.п., и даже действительно равные системы часто
не смогут доказать не-хужесть с маржой 3 п.п.

ПРЕДЕЛ ВЫПОЛНИМОСТИ §37 печатается как СПРАВКА: discord <= 2*p_fail - delta,
где discord = P(rescue) + P(harm). Пошаговое расхождение действий — ДРУГАЯ
величина и в это неравенство не подставляется.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

MARGIN = 0.03          # маржа не-хужести, п.п./100; §53.6, как в K-6h
Q_LO = 5               # квантиль нижней границы для обоих критериев
Q_HI = 95
N_BOOT = 10000
BOOT_SEED = 53
SETUP_SAME = ("suite", "horizon", "max_steps", "waiting_steps", "ensemble",
              "rollout_seed_mode", "ckpt")


def file_sha(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:12]


def load(paths):
    """Артефакты k9h. Ключ наблюдения — (рука, задача, состояние, сид)."""
    obs, meta, base = {}, {}, None
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"нет {p}")
        try:
            d = json.load(open(p))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{p}: не разбирается как JSON ({e})")
        need = ("episodes", "arm_label", "policy", "task_id", "init_start",
                "seed", "n_envs") + SETUP_SAME
        miss = [k for k in need if d.get(k) is None]
        if miss:
            raise SystemExit(f"{p}: нет полей {miss}")
        if base is None:
            base = d
        bad = [k for k in SETUP_SAME if str(d[k]) != str(base[k])]
        if bad:
            raise SystemExit(
                f"{p}: условия прогона отличаются от {base.get('arm_label')}: "
                + "; ".join(f"{k}: {d[k]} против {base[k]}" for k in bad))
        arm = str(d["arm_label"])
        m = meta.setdefault(arm, dict(policy=str(d["policy"]), files=[],
                                      fingerprints=set(), shas={}))
        m["files"].append(p)
        m["shas"][p] = file_sha(p)
        m["fingerprints"].add(str((d.get("joint") or {}).get(
            "model_fingerprint")))
        for j, e in enumerate(d["episodes"]):
            for k in ("success", "init_state_id", "env_index"):
                if e.get(k) is None:
                    raise SystemExit(f"{p}: в эпизоде {j} нет {k}")
            if not isinstance(e["success"], bool):
                raise SystemExit(f"{p}: success в эпизоде {j} не bool")
            if int(e["init_state_id"]) != int(d["init_start"]) + j:
                raise SystemExit(
                    f"{p}: init_state_id {e['init_state_id']} не отвечает "
                    f"init_start {d['init_start']} и позиции {j}")
            key = (arm, int(d["task_id"]), int(e["init_state_id"]),
                   int(d["seed"]))
            if key in obs:
                raise SystemExit(
                    f"{p}: наблюдение {key} уже есть в {obs[key]['path']} — "
                    f"один и тот же прогон учтён бы дважды")
            obs[key] = dict(success=bool(e["success"]), path=p,
                            done_step=e.get("done_step"),
                            success_step=e.get("success_step"),
                            grip_positive_step=e.get("grip_positive_step"),
                            action_sha1=e.get("action_sha1"))
    for arm, m in meta.items():
        if len(m["fingerprints"]) > 1:
            raise SystemExit(
                f"рука {arm} посчитана разными моделями: "
                f"{sorted(m['fingerprints'])}")
    return obs, meta


def align(obs, arms):
    """Одни и те же (задача, состояние, сид) у всех рук. Иначе не парно."""
    sets = {}
    for a in arms:
        sets[a] = {(t, s, sd) for (arm, t, s, sd) in obs if arm == a}
    base = sets[arms[0]]
    for a in arms[1:]:
        miss, extra = base - sets[a], sets[a] - base
        if miss or extra:
            raise SystemExit(
                f"рука {a} покрывает другие наблюдения: нет {len(miss)} "
                f"(например {sorted(miss)[:3]}), лишних {len(extra)} "
                f"(например {sorted(extra)[:3]}). Сравнение парное, и "
                f"несовпадение состава делает его недействительным")
    clusters = sorted({(t, s) for (t, s, _sd) in base})
    seeds = sorted({sd for (_t, _s, sd) in base})
    for (t, s) in clusters:
        got = sorted(sd for (tt, ss, sd) in base if (tt, ss) == (t, s))
        if got != seeds:
            raise SystemExit(
                f"кластер (задача {t}, состояние {s}) покрыт сидами {got}, а "
                f"остальные — {seeds}: повторы обязаны быть у всех кластеров, "
                f"иначе вес кластеров разный")
    return clusters, seeds


def cluster_means(obs, arm, clusters, seeds):
    """Доля успеха в кластере: среднее по сидам. Повторы — внутри кластера."""
    out = np.zeros(len(clusters))
    for i, (t, s) in enumerate(clusters):
        v = [obs[(arm, t, s, sd)]["success"] for sd in seeds]
        out[i] = float(np.mean(v))
    return out


def pair_rates(obs, base_arm, cand_arm, clusters, seeds):
    """rescue и harm по кластерам: средние по сидам внутри кластера."""
    res = np.zeros(len(clusters))
    hrm = np.zeros(len(clusters))
    for i, (t, s) in enumerate(clusters):
        r = h = 0
        for sd in seeds:
            b = obs[(base_arm, t, s, sd)]["success"]
            c = obs[(cand_arm, t, s, sd)]["success"]
            r += int((not b) and c)
            h += int(b and (not c))
        res[i] = r / float(len(seeds))
        hrm[i] = h / float(len(seeds))
    return res, hrm


def strata(clusters):
    """Индексы кластеров по задачам: бутстрап стратифицирован по задачам."""
    by = {}
    for i, (t, _s) in enumerate(clusters):
        by.setdefault(int(t), []).append(i)
    return {t: np.asarray(v, np.int64) for t, v in sorted(by.items())}


def boot(values, clusters, n=N_BOOT, seed=BOOT_SEED, qs=(Q_LO, Q_HI)):
    """Стратифицированный кластерный бутстрап нескольких величин РАЗОМ.

    `values` — словарь {имя: массив по кластерам}. В каждой реплике
    пересэмплируются ОДНИ И ТЕ ЖЕ кластеры для всех величин: разности
    считаются внутри реплики, потому что ошибки величин сильно
    коррелированы — они посчитаны на одних и тех же состояниях.
    """
    st = strata(clusters)
    rg = np.random.default_rng(seed)
    names = sorted(values)
    for k in names:
        if len(values[k]) != len(clusters):
            raise SystemExit(f"{k}: {len(values[k])} значений на "
                             f"{len(clusters)} кластеров")
    picks = np.empty((n, len(clusters)), np.int64)
    at = 0
    for t, idx in st.items():
        m = len(idx)
        picks[:, at:at + m] = idx[rg.integers(0, m, size=(n, m))]
        at += m
    if at != len(clusters):
        raise SystemExit("страты не покрыли все кластеры")
    # РАВНЫЙ ВЕС ЗАДАЧ: среднее берётся по задачам, а не по кластерам, чтобы
    # задача с большим числом состояний не весила больше.
    out = {}
    starts, sizes = [], []
    at = 0
    for t, idx in st.items():
        starts.append(at); sizes.append(len(idx)); at += len(idx)
    for k in names:
        v = np.asarray(values[k], float)[picks]          # (n, C)
        per_task = np.stack([v[:, s0:s0 + sz].mean(axis=1)
                             for s0, sz in zip(starts, sizes)], axis=1)
        out[k] = per_task.mean(axis=1)
    res = {k: [float(np.percentile(out[k], q)) for q in qs] for k in names}
    return res, out


def point(values, clusters):
    """Точечная оценка с тем же равным весом задач, что в бутстрапе."""
    st = strata(clusters)
    v = np.asarray(values, float)
    return float(np.mean([v[idx].mean() for idx in st.values()]))


SECONDARY_MIN_PFAIL = 0.2      # §53.7: порог доли провалов q0 для задачи


def secondary_tasks(obs, base_arm, clusters, seeds,
                    min_p_fail=SECONDARY_MIN_PFAIL):
    """Задачи с ненулевым запасом: доля провалов q0 не ниже порога.

    ОПРЕДЕЛЯЕТСЯ НА dev, ПРИМЕНЯЕТСЯ НА final (§53.7). Определять набор и
    считать по нему результат на ОДНИХ данных — это отбор и вывод по одной
    выборке; разнесение по банкам снимает проблему.
    """
    by = {}
    for (t, s) in clusters:
        v = [obs[(base_arm, t, s, sd)]["success"] for sd in seeds]
        by.setdefault(int(t), []).extend(v)
    out = {}
    for t, v in sorted(by.items()):
        pf = 1.0 - float(np.mean(v))
        out[t] = dict(p_fail=pf, n=len(v),
                      included=bool(pf >= float(min_p_fail)))
    return out


def verdict_noninf(lo, hi, margin=MARGIN):
    """Три исхода не-хужести, и все три объявлены заранее."""
    if lo > -margin:
        return "не-хужесть доказана"
    if hi < -margin:
        return "не-хужесть опровергнута"
    return "НЕОПРЕДЕЛЁННО: набора не хватает"


def choose_for_final(improve_lo):
    """Правило единственного открытия final, §53.5."""
    return "q0+q1" if improve_lo > 0 else "q0"


def main():
    ap = argparse.ArgumentParser(description="Анализ §53: success/rescue/harm")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--arts", nargs="*", default=[])
    ap.add_argument("--base", default=None, help="метка руки q0")
    ap.add_argument("--cand", default=None, help="метка руки q0+q1")
    ap.add_argument("--ref", default=None, help="метка руки полной BAR")
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--delta", type=float, default=0.05,
                    help="целевой прирост для справки по пределу §37")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    ap.add_argument("--bank", default="", choices=("", "dev", "final"),
                    help="какой банк анализируется")
    ap.add_argument("--secondary-tasks", default="",
                    help="через запятую: вторичный набор задач, ОПРЕДЕЛЁННЫЙ "
                         "НА dev. Задаётся только при анализе final; на dev "
                         "набор вычисляется и записывается, но вторичный "
                         "результат по нему НЕ считается — иначе отбор и "
                         "вывод шли бы по одной выборке")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", default="reports/k14q/behavior.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    for nm, v in (("--base", a.base), ("--cand", a.cand), ("--ref", a.ref)):
        if not v:
            raise SystemExit(f"нужен {nm}")
    if not a.arts:
        raise SystemExit("нужны --arts")
    if os.path.exists(a.out) and not a.overwrite:
        raise SystemExit(f"{a.out} уже существует (--overwrite осознанно)")

    obs, meta = load(a.arts)
    arms = [a.base, a.cand, a.ref]
    for nm in arms:
        if nm not in meta:
            raise SystemExit(f"нет руки {nm}; есть {sorted(meta)}")
    clusters, seeds = align(obs, arms)
    n_task = len(strata(clusters))
    print(f"\n  кластеров {len(clusters)} в {n_task} задачах, повторов "
          f"{len(seeds)} (сиды {seeds}), реплик {a.boot}")

    sr = {nm: cluster_means(obs, nm, clusters, seeds) for nm in arms}
    resc, harm = pair_rates(obs, a.base, a.cand, clusters, seeds)
    vals = dict(sr_base=sr[a.base], sr_cand=sr[a.cand], sr_ref=sr[a.ref],
                rescue=resc, harm=harm,
                d_cand_base=sr[a.cand] - sr[a.base],
                d_cand_ref=sr[a.cand] - sr[a.ref],
                d_base_ref=sr[a.base] - sr[a.ref],
                discord=resc + harm)
    ci, _draws = boot(vals, clusters, n=int(a.boot), seed=int(a.boot_seed))
    pt = {k: point(v, clusters) for k, v in vals.items()}

    print(f"\n  {'величина':16s} {'точечно':>9s}  90% интервал")
    for k in ("sr_base", "sr_cand", "sr_ref", "rescue", "harm", "discord"):
        print(f"  {k:16s} {pt[k]:+9.4f}  [{ci[k][0]:+.4f}, {ci[k][1]:+.4f}]")

    # --- УЛУЧШЕНИЕ: двусторонний 90%, требуется q05 > 0 -------------------
    lo_imp, hi_imp = ci["d_cand_base"]
    improved = bool(lo_imp > 0)
    print(f"\n  УЛУЧШЕНИЕ {a.cand} против {a.base}: точечно "
          f"{pt['d_cand_base']:+.4f}, двусторонний 90% "
          f"[{lo_imp:+.4f}, {hi_imp:+.4f}] -> "
          f"{'есть' if improved else 'не доказано'}")
    print(f"    сверка тождества: P(rescue) - P(harm) = "
          f"{pt['rescue'] - pt['harm']:+.6f}, разность SR "
          f"{pt['d_cand_base']:+.6f}")
    if abs((pt["rescue"] - pt["harm"]) - pt["d_cand_base"]) > 1e-9:
        raise SystemExit(
            "P(rescue) - P(harm) не равно разности SR: одна из величин "
            "считается не по тем наблюдениям")

    # --- ВЫБОР СИСТЕМЫ И НЕ-ХУЖЕСТЬ У НЕЁ --------------------------------
    chosen = choose_for_final(lo_imp)
    key = "d_cand_ref" if chosen == "q0+q1" else "d_base_ref"
    lo_ni, hi_ni = ci[key]
    vn = verdict_noninf(lo_ni, hi_ni, margin=float(a.margin))
    print(f"\n  ВЫБРАНА ДЛЯ final: {chosen}")
    print(f"  НЕ-ХУЖЕСТЬ выбранной против {a.ref}: точечно {pt[key]:+.4f}, "
          f"односторонняя 95% нижняя граница {lo_ni:+.4f} против порога "
          f"{-float(a.margin):+.4f}")
    print(f"    -> {vn}")

    # --- СПРАВКА §37 ------------------------------------------------------
    p_fail = 1.0 - pt["sr_base"]
    lim = 2.0 * p_fail - float(a.delta)
    print(f"\n  СПРАВКА §37: p_fail(q0) = {p_fail:.3f}, дискордантность "
          f"{pt['discord']:.3f}, допустимая при delta={a.delta:.2f} — не выше "
          f"{lim:+.3f} -> {'в пределе' if pt['discord'] <= lim else 'ВНЕ предела'}")

    # --- ВТОРИЧНЫЙ НАБОР: НА dev ОПРЕДЕЛЯЕМ, НА final ПРИМЕНЯЕМ ----------
    sec = secondary_tasks(obs, a.base, clusters, seeds)
    sec_inc = sorted(t for t, v in sec.items() if v["included"])
    print(f"\n  доля провалов {a.base} по задачам: "
          + ", ".join(f"{t}:{sec[t]['p_fail']:.2f}" for t in sorted(sec)))
    secondary = None
    if a.bank == "final" and a.secondary_tasks:
        want = sorted(int(x) for x in a.secondary_tasks.split(",") if x != "")
        miss = [t for t in want if t not in {t2 for (t2, _s) in clusters}]
        if miss:
            raise SystemExit(f"вторичный набор называет задачи {miss}, "
                             f"которых в данных нет")
        sub = [i for i, (t, _s) in enumerate(clusters) if int(t) in want]
        cl_s = [clusters[i] for i in sub]
        vals_s = {k: np.asarray(v)[sub] for k, v in vals.items()}
        ci_s, _ = boot(vals_s, cl_s, n=int(a.boot), seed=int(a.boot_seed))
        pt_s = {k: point(v, cl_s) for k, v in vals_s.items()}
        key_s = "d_cand_ref" if chosen == "q0+q1" else "d_base_ref"
        vn_s = verdict_noninf(ci_s[key_s][0], ci_s[key_s][1],
                              margin=float(a.margin))
        secondary = dict(tasks=want, n_clusters=len(cl_s), point=pt_s,
                         ci=ci_s, non_inferiority=dict(
                             pair=key_s, lo=ci_s[key_s][0],
                             hi=ci_s[key_s][1], verdict=vn_s),
                         improvement=dict(lo=ci_s["d_cand_base"][0],
                                          hi=ci_s["d_cand_base"][1]))
        print(f"\n  ВТОРИЧНЫЙ НАБОР (задачи {want}, {len(cl_s)} кластеров, "
              f"определён на dev):")
        print(f"    улучшение [{ci_s['d_cand_base'][0]:+.4f}, "
              f"{ci_s['d_cand_base'][1]:+.4f}]; не-хужесть "
              f"{ci_s[key_s][0]:+.4f} -> {vn_s}")
    elif a.bank == "dev":
        print(f"  вторичный набор по правилу p_fail >= "
              f"{SECONDARY_MIN_PFAIL}: {sec_inc}. На dev он только "
              f"ОПРЕДЕЛЯЕТСЯ; результат по нему считается на final")

    # --- ДИАГНОСТИКА ТИПА ПРОВАЛА И РАСХОЖДЕНИЯ ДЕЙСТВИЙ -----------------
    diag = fail_diag(obs, arms, clusters, seeds)
    print(f"\n  расхождение действий {a.base}/{a.cand}: "
          f"{diag['action_disagree_frac']:.3f} наблюдений"
          + ("" if diag["action_sha_available"]
             else "  (отпечатков действий в артефактах нет)"))
    for nm in arms:
        d_ = diag["by_arm"][nm]
        if d_["n_fail"]:
            print(f"    {nm:16s} провалов {d_['n_fail']:4d}, из них дошли до "
                  f"схвата {d_['fail_with_grip']:4d}, медиана шага "
                  f"завершения {d_['median_done_step']}")

    out = dict(
        kind="k14q_behavior", bank=a.bank, arms=dict(base=a.base,
                                                     cand=a.cand, ref=a.ref),
        n_clusters=len(clusters), n_tasks=n_task, seeds=seeds,
        boot=int(a.boot), boot_seed=int(a.boot_seed),
        quantiles=dict(improvement="двусторонний 90%, требуется q05 > 0",
                       non_inferiority=("односторонняя 95% нижняя граница, "
                                        "требуется q05 > -margin"),
                       q_lo=Q_LO, q_hi=Q_HI),
        margin=float(a.margin), point=pt, ci=ci,
        improved=improved, chosen_for_final=chosen,
        non_inferiority=dict(pair=key, lo=lo_ni, hi=hi_ni, verdict=vn),
        feasibility_37=dict(p_fail=p_fail, discord=pt["discord"],
                            delta=float(a.delta), limit=lim,
                            within=bool(pt["discord"] <= lim)),
        diagnostics=diag,
        secondary_rule=dict(min_p_fail=SECONDARY_MIN_PFAIL,
                            defined_on="dev", applied_on="final"),
        secondary_per_task=sec, secondary_included=sec_inc,
        secondary=secondary,
        sources={nm: dict(files=meta[nm]["files"], sha1=meta[nm]["shas"],
                          policy=meta[nm]["policy"],
                          model_fingerprint=sorted(meta[nm]["fingerprints"]))
                 for nm in arms},
        note=("единица кластера (задача, init_state_id); сиды — повторные "
              "измерения внутри кластера; бутстрап стратифицирован по "
              "задачам с равным весом задач"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    t_ = a.out + f".tmp.{os.getpid()}"
    json.dump(out, open(t_, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(t_, a.out)
    print(f"\n  сохранено: {a.out}")
    return 0


def fail_diag(obs, arms, clusters, seeds):
    """Описательная диагностика: тип провала и расхождение действий.

    ЭТО НЕ КРИТЕРИЙ. Пошаговое расхождение действий в предел §37 не
    подставляется: там discord = P(rescue) + P(harm), другая величина.
    """
    by = {}
    for nm in arms:
        n_fail = grip = 0
        dones = []
        for (t, s) in clusters:
            for sd in seeds:
                e = obs[(nm, t, s, sd)]
                if not e["success"]:
                    n_fail += 1
                    if e.get("grip_positive_step") is not None \
                            and int(e["grip_positive_step"]) >= 0:
                        grip += 1
                if e.get("done_step") is not None:
                    dones.append(int(e["done_step"]))
        by[nm] = dict(n_fail=n_fail, fail_with_grip=grip,
                      median_done_step=(int(np.median(dones)) if dones
                                        else None))
    base, cand = arms[0], arms[1]
    have = diff = tot = 0
    for (t, s) in clusters:
        for sd in seeds:
            x = obs[(base, t, s, sd)].get("action_sha1")
            y = obs[(cand, t, s, sd)].get("action_sha1")
            tot += 1
            if x is not None and y is not None:
                have += 1
                diff += int(x != y)
    return dict(by_arm=by, action_sha_available=bool(have == tot and tot > 0),
                action_disagree_frac=(diff / have if have else float("nan")),
                n_observations=tot)


def _art(arm, policy, task, i0, succ, seed, **kw):
    d = dict(arm_label=arm, policy=policy, task_id=task, init_start=i0,
             n_envs=len(succ), seed=seed, suite="10", horizon=8,
             max_steps=600, waiting_steps=10, ensemble="off",
             rollout_seed_mode="fixed", ckpt="CK",
             joint=dict(model_fingerprint=f"F_{arm}"),
             episodes=[dict(env_index=j, success=bool(s), init_state_id=i0 + j,
                            done_step=100 + j, success_step=(50 if s else -1),
                            grip_positive_step=(20 if s else -1),
                            action_sha1=f"{arm}{task}{i0 + j}")
                       for j, s in enumerate(succ)])
    d.update(kw)
    return d


def selftest():
    import tempfile
    # --- РУЧНОЙ ПРИМЕР: rescue и harm считаются на бумаге ------------------
    # две задачи по два состояния, один сид.
    # q0:   T F | T T      q1:  T T | F T      BAR: T T | T T
    # rescue: (0,1) ; harm: (1,0)
    with tempfile.TemporaryDirectory() as td:
        def w(d, nm):
            q = os.path.join(td, nm)
            json.dump(d, open(q, "w"))
            return q
        ps = [w(_art("q0", "fast", 0, 0, [1, 0], 101), "a0.json"),
              w(_art("q0", "fast", 1, 0, [1, 1], 101), "a1.json"),
              w(_art("q1", "depthrvq", 0, 0, [1, 1], 101), "b0.json"),
              w(_art("q1", "depthrvq", 1, 0, [0, 1], 101), "b1.json"),
              w(_art("bar", "fullbar", 0, 0, [1, 1], 101), "c0.json"),
              w(_art("bar", "fullbar", 1, 0, [1, 1], 101), "c1.json")]
        obs, meta = load(ps)
        arms = ["q0", "q1", "bar"]
        cl, sd = align(obs, arms)
        assert cl == [(0, 0), (0, 1), (1, 0), (1, 1)] and sd == [101]
        r, h = pair_rates(obs, "q0", "q1", cl, sd)
        assert r.tolist() == [0, 1, 0, 0], r
        assert h.tolist() == [0, 0, 1, 0], h
        srb = cluster_means(obs, "q0", cl, sd)
        src = cluster_means(obs, "q1", cl, sd)
        assert srb.tolist() == [1, 0, 1, 1] and src.tolist() == [1, 1, 0, 1]
        # ТОЖДЕСТВО: разность SR равна P(rescue) - P(harm)
        d_sr = point(src - srb, cl)
        assert abs(d_sr - (point(r, cl) - point(h, cl))) < 1e-12
        assert abs(d_sr) < 1e-12, "здесь rescue и harm компенсируются"
        # РАВНЫЙ ВЕС ЗАДАЧ: у задачи 0 успех 1/2, у задачи 1 — 1
        assert abs(point(srb, cl) - 0.75) < 1e-12, point(srb, cl)

        # --- НЕПАРНЫЙ СОСТАВ ОТВЕРГАЕТСЯ ---------------------------------
        bad = ps[:-1] + [w(_art("bar", "fullbar", 1, 0, [1, 1, 1], 101),
                           "c1b.json")]
        try:
            align(*load(bad)[:1], arms)
        except SystemExit as e:
            assert "другие наблюдения" in str(e), e
        else:
            raise AssertionError("принят непарный состав")
        # --- РАЗНЫЕ УСЛОВИЯ ----------------------------------------------
        try:
            load([ps[0], w(_art("q0", "fast", 1, 0, [1, 1], 101,
                                max_steps=300), "x.json")])
        except SystemExit as e:
            assert "условия прогона" in str(e), e
        else:
            raise AssertionError("приняты разные условия")
        # --- ДВЕ МОДЕЛИ ПОД ОДНОЙ МЕТКОЙ ---------------------------------
        try:
            load([ps[0], w(_art("q0", "fast", 1, 0, [1, 1], 101,
                                joint=dict(model_fingerprint="ДРУГОЙ")),
                           "y.json")])
        except SystemExit as e:
            assert "разными моделями" in str(e), e
        else:
            raise AssertionError("принята рука из двух моделей")
        # --- ПОВТОР НЕ У ВСЕХ КЛАСТЕРОВ ----------------------------------
        ps2 = ps + [w(_art("q0", "fast", 0, 0, [1, 0], 102), "d0.json"),
                    w(_art("q1", "depthrvq", 0, 0, [1, 1], 102), "d1.json"),
                    w(_art("bar", "fullbar", 0, 0, [1, 1], 102), "d2.json")]
        try:
            align(load(ps2)[0], arms)
        except SystemExit as e:
            assert "покрыт сидами" in str(e), e
        else:
            raise AssertionError("принят повтор не у всех кластеров")

    # --- БУТСТРАП: ЗНАК И СТРАТИФИКАЦИЯ ----------------------------------
    rg = np.random.default_rng(0)
    T, S = 10, 20
    clusters = [(t, 10 + s) for t in range(T) for s in range(S)]
    base = (rg.random(len(clusters)) < 0.8).astype(float)
    cand = np.clip(base + 0.1, 0, 1)                 # кандидат строго лучше
    ci, _ = boot(dict(d=cand - base), clusters, n=500, seed=1)
    assert ci["d"][0] > 0, ci["d"]
    ci2, _ = boot(dict(d=base - cand), clusters, n=500, seed=1)
    assert ci2["d"][1] < 0, "знак разности зависит от порядка — ошибка"
    # одинаковые величины -> интервал содержит ноль
    ci3, _ = boot(dict(d=base - base.copy()), clusters, n=200, seed=2)
    assert ci3["d"][0] <= 0 <= ci3["d"][1]
    # стратификация: все кластеры покрыты стратами
    st = strata(clusters)
    assert len(st) == T and all(len(v) == S for v in st.values())
    # длина не та -> отказ
    try:
        boot(dict(d=base[:-1]), clusters, n=10)
    except SystemExit as e:
        assert "значений на" in str(e), e
    else:
        raise AssertionError("принята величина не по кластерам")

    # --- ВТОРИЧНЫЙ НАБОР: ПОРОГ ПО ДОЛЕ ПРОВАЛОВ --------------------------
    with tempfile.TemporaryDirectory() as td3:
        def w3(d, nm):
            q = os.path.join(td3, nm)
            json.dump(d, open(q, "w"))
            return q
        # задача 0: успех 3/5 -> p_fail 0.4, включена
        # задача 1: успех 5/5 -> p_fail 0.0, исключена
        pp = [w3(_art("q0", "fast", 0, 0, [1, 0, 1, 0, 1], 101), "s0.json"),
              w3(_art("q0", "fast", 1, 0, [1, 1, 1, 1, 1], 101), "s1.json")]
        o3, _ = load(pp)
        cl3 = sorted({(t, s) for (_a, t, s, _sd) in o3})
        sec = secondary_tasks(o3, "q0", cl3, [101])
        assert abs(sec[0]["p_fail"] - 0.4) < 1e-12 and sec[0]["included"]
        assert sec[1]["p_fail"] == 0.0 and not sec[1]["included"]
        # порог ровно на границе -> включена
        sec2 = secondary_tasks(o3, "q0", cl3, [101], min_p_fail=0.4)
        assert sec2[0]["included"]
        sec3 = secondary_tasks(o3, "q0", cl3, [101], min_p_fail=0.41)
        assert not sec3[0]["included"]

    # --- ТРИ ИСХОДА НЕ-ХУЖЕСТИ -------------------------------------------
    assert verdict_noninf(-0.01, +0.02) == "не-хужесть доказана"
    assert verdict_noninf(-0.09, -0.05) == "не-хужесть опровергнута"
    assert "НЕОПРЕДЕЛЁННО" in verdict_noninf(-0.06, +0.01)
    # ровно на границе -> не доказана
    assert "НЕОПРЕДЕЛЁННО" in verdict_noninf(-MARGIN, +0.01)
    # --- ПРАВИЛО ВЫБОРА ---------------------------------------------------
    assert choose_for_final(+0.001) == "q0+q1"
    assert choose_for_final(0.0) == "q0" and choose_for_final(-0.01) == "q0"
    # --- СКВОЗНОЙ ПРОГОН: ОБЕ ВЕТКИ ПРАВИЛА ВЫБОРА -----------------------
    # Ветка, принимающая решение и не покрытая тестом, — это ветка, про
    # которую узнаёшь в день, когда она сработает.
    import subprocess
    import tempfile as tf2
    me = os.path.abspath(__file__)
    with tf2.TemporaryDirectory() as td:
        def w2(d, nm):
            q = os.path.join(td, nm)
            json.dump(d, open(q, "w"))
            return q
        for ci_, (tag, q1succ, want) in enumerate((
                # кандидат спасает по одному состоянию в каждой задаче
                ("кандидат лучше", [1, 1], "q0+q1"),
                # кандидат ломает то, что работало
                ("кандидат хуже", [0, 0], "q0"))):
            arts = []
            for t in range(10):
                arts += [
                    w2(_art("q0", "fast", t, 10, [1, 0], 101),
                       f"c{ci_}_q0_{t}.json"),
                    w2(_art("q1", "depthrvq", t, 10, q1succ, 101),
                       f"c{ci_}_q1_{t}.json"),
                    w2(_art("bar", "fullbar", t, 10, [1, 1], 101),
                       f"c{ci_}_bar_{t}.json")]
            o = os.path.join(td, f"out_c{ci_}.json")
            r = subprocess.run(
                [sys.executable, me, "--arts"] + arts
                + ["--base", "q0", "--cand", "q1", "--ref", "bar",
                   "--boot", "300", "--out", o],
                capture_output=True, text=True)
            assert r.returncode == 0, (tag, r.stdout[-600:], r.stderr[-600:])
            got = json.load(open(o))
            assert got["chosen_for_final"] == want, (tag, got[
                "chosen_for_final"])
            assert got["kind"] == "k14q_behavior"
            # не-хужесть проверяется у ВЫБРАННОЙ системы
            assert got["non_inferiority"]["pair"] == (
                "d_cand_ref" if want == "q0+q1" else "d_base_ref"), tag
            if want == "q0+q1":
                assert got["improved"] and got["point"]["rescue"] > 0, tag
            else:
                assert not got["improved"] and got["point"]["harm"] > 0, tag
    print("самопроверка k14q_behavior пройдена")


if __name__ == "__main__":
    sys.exit(main())
