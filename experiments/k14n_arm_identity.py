#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14n: смоук тождественности рук перед парным поведенческим прогоном (§53).

ЗАЧЕМ. Оснастка k9h существует и проверена, но РУКА depthrvq в ней новая.
Наследовать доверие от прежних прогонов нельзя: новый архитектурный путь
обязан пройти собственную проверку, как это было с §49.

ЧТО ПРОВЕРЯЕТСЯ, И ПОЧЕМУ ИМЕННО ЭТО

  1. ОДНИ И ТЕ ЖЕ НАЧАЛЬНЫЕ СОСТОЯНИЯ. Вся статистика §53 парная; если руки
     стартовали из разных состояний, сравнивать нечего. Проверяется по
     init_hash и init_hash_full, которые k9h пишет на каждый эпизод.

  2. СОВПАДЕНИЕ ИСХОДОВ — И ТОЛЬКО ОНО. Первая редакция этого файла
     утверждала, что равенство success, env_steps и policy_calls по эпизодам
     доказывает равенство ДЕЙСТВИЙ. Это неверно: в k9h весь векторный батч
     шагает вместе, поэтому env_steps и policy_calls ОДИНАКОВЫ у всех
     эпизодов прогона и различаются только если батч целиком завершился на
     другом шаге. Фактически по эпизодам сравнивается один бит — success, —
     и две разные политики могут его совпасть.

     РАВЕНСТВО ДЕЙСТВИЙ ДОКАЗЫВАЕТСЯ НЕ ЗДЕСЬ, а прямым сравнением на ОДНОМ
     И ТОМ ЖЕ входном батче до шага среды (k14o_fast_identity.py). Этот файл
     отвечает на более слабый вопрос: сравнимы ли руки по условиям и
     стартовали ли они из одних состояний, и расходятся ли исходы.

  3. ПОВТОР ОДНОЙ РУКИ — ДИАГНОСТИКА, А НЕ ДОКАЗАТЕЛЬСТВО. Совпадение
     исходов двух повторов НЕ доказывает детерминизм: четыре бита success
     легко совпадают и у двух стохастических раскаток. Поэтому дизайн §53
     НЕ ставится в зависимость от этого сравнения — разные execution seeds
     для повторов фиксируются БЕЗУСЛОВНО (§53.2).

     Зато сравнение обязано убедиться, что два повтора — это одна и та же
     МОДЕЛЬ: иначе различие двух разных моделей будет названо шумом
     исполнения. Сверяется model_fingerprint, который k9h считает из sha
     головы, sha Joint12, режима, варианта и сида и который, в отличие от
     arm_fingerprint, НЕ включает метку руки.

ЧЕГО ЗДЕСЬ НЕТ. Анализа rescue/harm: это результат §53, а не смоук. Смоук
отвечает только на вопрос «сравнимы ли руки вообще».
"""
import argparse
import json
import os
import sys

SETUP_SAME = ("suite", "task_id", "init_start", "n_envs", "seed",
              "ensemble", "horizon", "max_steps", "waiting_steps",
              "pos_offset", "rollout_seed_mode", "ckpt")
EP_SAME = ("success", "env_steps", "policy_calls")


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"нет {path}")
    d = json.load(open(path))
    for k in ("episodes", "arm_label", "policy"):
        if d.get(k) is None:
            raise SystemExit(f"{path}: нет поля {k}")
    eps = d["episodes"]
    idx = [e.get("env_index") for e in eps]
    if sorted(idx) != list(range(len(eps))):
        raise SystemExit(f"{path}: env_index не покрывает 0..{len(eps) - 1}")
    d["_eps"] = {int(e["env_index"]): e for e in eps}
    d["_path"] = path
    return d


def check_setup(arts):
    """Руки сравнимы только при совпадении условий прогона."""
    base = arts[0]
    for d in arts[1:]:
        bad = [k for k in SETUP_SAME if str(d.get(k)) != str(base.get(k))]
        if bad:
            raise SystemExit(
                f"{d['_path']} и {base['_path']} посчитаны в разных "
                f"условиях: " + "; ".join(
                    f"{k}: {d.get(k)} против {base.get(k)}" for k in bad))
        if len(d["_eps"]) != len(base["_eps"]):
            raise SystemExit(f"{d['_path']}: {len(d['_eps'])} эпизодов "
                             f"против {len(base['_eps'])}")


def check_states(arts):
    """Начальные состояния совпадают поэпизодно."""
    base = arts[0]
    for d in arts[1:]:
        for i, e in sorted(base["_eps"].items()):
            o = d["_eps"][i]
            for k in ("init_hash", "init_hash_full"):
                if e.get(k) is None or o.get(k) is None:
                    raise SystemExit(f"эпизод {i}: нет {k}")
                if e[k] != o[k]:
                    raise SystemExit(
                        f"эпизод {i}: {k} у {d['arm_label']} {o[k]}, у "
                        f"{base['arm_label']} {e[k]} — руки стартовали из "
                        f"РАЗНЫХ состояний, парное сравнение недействительно")
    return True


def check_same_model(a, b):
    """Два повтора обязаны быть ОДНОЙ моделью.

    Иначе расхождение двух РАЗНЫХ моделей было бы названо шумом исполнения.
    Сверяется единый отпечаток руки, который k9h считает из sha головы, sha
    Joint12, режима, варианта и сида — одно поле вместо перечисления.
    """
    # ИМЕННО model_fingerprint, А НЕ arm_fingerprint. Второй включает МЕТКУ
    # руки, а у повторов метки по необходимости разные — проверка на нём не
    # могла пройти никогда.
    fa = (a.get("joint") or {}).get("model_fingerprint")
    fb = (b.get("joint") or {}).get("model_fingerprint")
    if fa is None or fb is None:
        raise SystemExit(
            f"у повторов {a.get('arm_label')}/{b.get('arm_label')} нет "
            f"model_fingerprint: доказать, что это одна модель, нечем. "
            f"arm_fingerprint для этого не годится — он включает метку")
    if fa != fb:
        raise SystemExit(
            f"повторы {a.get('arm_label')} и {b.get('arm_label')} — РАЗНЫЕ "
            f"модели: отпечатки {fa} и {fb}. Их расхождение нельзя называть "
            f"шумом исполнения")
    return fa


def diff_episodes(a, b):
    """Список эпизодов, где руки разошлись, и по каким полям."""
    out = []
    for i, e in sorted(a["_eps"].items()):
        o = b["_eps"][i]
        bad = [k for k in EP_SAME if e.get(k) != o.get(k)]
        if bad:
            out.append((i, {k: (e.get(k), o.get(k)) for k in bad}))
    return out


def main():
    ap = argparse.ArgumentParser(description="Смоук тождественности рук §53")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--arts", nargs="*", default=[],
                    help="артефакты k9h; условия прогона обязаны совпасть")
    ap.add_argument("--equal", nargs=2, action="append", default=[],
                    metavar=("A", "B"),
                    help="пара меток, ТРАЕКТОРИИ которых обязаны совпасть "
                         "(например fast и depthrvq_fast)")
    ap.add_argument("--repeat", nargs=2, action="append", default=[],
                    metavar=("A", "B"),
                    help="пара повторов ОДНОЙ руки: если совпали побитово, "
                         "повтор не является новой выборкой")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if len(a.arts) < 2:
        raise SystemExit("нужно хотя бы два артефакта")

    arts = [load(p) for p in a.arts]
    by = {}
    for d in arts:
        if d["arm_label"] in by:
            raise SystemExit(f"метка {d['arm_label']} встречается дважды: "
                             f"смоук сравнивает РАЗНЫЕ руки")
        by[d["arm_label"]] = d
    check_setup(arts)
    check_states(arts)
    print(f"  условия совпали, начальные состояния совпали: "
          f"{len(arts)} рук, {len(arts[0]['_eps'])} эпизодов")

    res = dict(kind="k14n_arm_identity", arms=sorted(by),
               n_episodes=len(arts[0]["_eps"]), equal=[], repeat=[])
    ok = True
    for x, y in a.equal:
        for nm in (x, y):
            if nm not in by:
                raise SystemExit(f"нет метки {nm}; есть {sorted(by)}")
        d_ = diff_episodes(by[x], by[y])
        res["equal"].append(dict(a=x, b=y, n_diff=len(d_),
                                 passed=not d_, diff=d_[:5]))
        if d_:
            ok = False
            print(f"  ТОЖДЕСТВО НАРУШЕНО: {x} против {y} — расхождение в "
                  f"{len(d_)} эпизодах из {len(arts[0]['_eps'])} "
                  f"(исходы, не действия)")
            for i, w in d_[:5]:
                print(f"    эпизод {i}: " + ", ".join(
                    f"{k} {v[0]} против {v[1]}" for k, v in w.items()))
        else:
            print(f"  исходы совпали: {x} и {y} во всех эпизодах по "
                  f"{', '.join(EP_SAME)}. ЭТО НЕ доказательство равенства "
                  f"действий — его даёт k14o на фиксированном входе")
    for x, y in a.repeat:
        for nm in (x, y):
            if nm not in by:
                raise SystemExit(f"нет метки {nm}; есть {sorted(by)}")
        check_same_model(by[x], by[y])
        d_ = diff_episodes(by[x], by[y])
        deterministic = not d_
        res["repeat"].append(dict(a=x, b=y, deterministic=deterministic,
                                  n_diff=len(d_)))
        if deterministic:
            print(f"  исходы повторов {x} и {y} совпали. ЭТО НЕ "
                  f"доказательство детерминизма: сравниваются исходы "
                  f"эпизодов, и они совпадают и у стохастических раскаток. "
                  f"Разные execution seeds фиксируются безусловно (§53.2)")
        else:
            print(f"  исходы повторов {x} и {y} разошлись в {len(d_)} "
                  f"эпизодах: разброс исполнения есть и наблюдаем")
    res["passed"] = bool(ok)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        t_ = a.out + f".tmp.{os.getpid()}"
        json.dump(res, open(t_, "w"), ensure_ascii=False, indent=1)
        os.replace(t_, a.out)
        print(f"  сохранено: {a.out}")
    if not ok:
        raise SystemExit("смоук не пройден: к §53 переходить нельзя")
    print("  СМОУК ПРОЙДЕН")
    return 0


def _art(label, policy, succ, steps=None, calls=None, ih=None, **kw):
    n = len(succ)
    steps = steps or [50] * n
    calls = calls or [5] * n
    ih = ih or [f"h{i}" for i in range(n)]
    d = dict(arm_label=label, policy=policy, suite="s", task_id=0,
             init_start=0, n_envs=n, seed=1, ensemble="off", horizon=8,
             max_steps=300, waiting_steps=10, pos_offset=0,
             rollout_seed_mode="m", ckpt="C",
             episodes=[dict(env_index=i, success=bool(succ[i]),
                            env_steps=steps[i], policy_calls=calls[i],
                            init_hash=ih[i], init_hash_full=ih[i] + "f")
                       for i in range(n)])
    d.update(kw)
    return d


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        def w(d, nm):
            q = os.path.join(td, nm)
            json.dump(d, open(q, "w"))
            return q

        # ПОВТОР: одна модель -> сравнение допускается, разные -> отказ
        r1 = load(w(_art("m_s0", "depthrvq", [1, 0, 1, 1],
                         joint=dict(model_fingerprint="F1")), "r1.json"))
        r2 = load(w(_art("m_rep_s0", "depthrvq", [1, 0, 1, 1],
                         joint=dict(model_fingerprint="F1")), "r2.json"))
        r3 = load(w(_art("m_oth_s0", "depthrvq", [1, 0, 1, 1],
                         joint=dict(model_fingerprint="F2")), "r3.json"))
        assert check_same_model(r1, r2) == "F1"
        r4 = load(w(_art("m_nof_s0", "depthrvq", [1, 0, 1, 1]), "r4.json"))
        for pair, why in (((r1, r3), "РАЗНЫЕ"), ((r1, r4), "нет"),
                          ((r4, r1), "нет")):
            try:
                check_same_model(*pair)
            except SystemExit as ex:
                assert why in str(ex), (why, ex)
            else:
                raise AssertionError(f"приняты повторы: {why}")

        a = load(w(_art("fast_s0", "fast", [1, 0, 1, 1]), "a.json"))
        b = load(w(_art("drvq_fast_s0", "depthrvq", [1, 0, 1, 1]), "b.json"))
        check_setup([a, b])
        assert check_states([a, b])
        assert diff_episodes(a, b) == [], "одинаковые руки объявлены разными"

        # расхождение хотя бы в одном поле обязано быть замечено
        c = load(w(_art("drvq_med_s0", "depthrvq", [1, 0, 1, 1],
                        steps=[50, 50, 51, 50]), "c.json"))
        d_ = diff_episodes(a, c)
        assert len(d_) == 1 and d_[0][0] == 2, d_
        e = load(w(_art("x_s0", "depthrvq", [1, 1, 1, 1]), "e.json"))
        assert len(diff_episodes(a, e)) == 1

        # разные начальные состояния -> отказ
        f = load(w(_art("y_s0", "fast", [1, 0, 1, 1],
                        ih=["h0", "ZZ", "h2", "h3"]), "f.json"))
        try:
            check_states([a, f])
        except SystemExit as ex:
            assert "РАЗНЫХ состояний" in str(ex), ex
        else:
            raise AssertionError("приняты разные начальные состояния")

        # разные условия прогона -> отказ, и сообщение называет поле
        g = load(w(_art("z_s0", "fast", [1, 0, 1, 1], seed=2), "g.json"))
        try:
            check_setup([a, g])
        except SystemExit as ex:
            assert "seed" in str(ex), ex
        else:
            raise AssertionError("приняты разные условия прогона")

        # битый артефакт: env_index не покрывает диапазон
        bad = _art("q_s0", "fast", [1, 1])
        bad["episodes"][1]["env_index"] = 5
        try:
            load(w(bad, "q.json"))
        except SystemExit as ex:
            assert "env_index" in str(ex), ex
        else:
            raise AssertionError("принят артефакт с дырой в env_index")
    print("самопроверка k14n_arm_identity пройдена")


if __name__ == "__main__":
    sys.exit(main())
