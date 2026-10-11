#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15g: постоянные поправки M1-cold против кратких импульсов K-15g.

ТОЛЬКО ЧИТАЕТ артефакты: снимок M1-cold (проверяется целиком), план K-15g
(его входы сверяются штатной check_plan_inputs) и каталог раскаток K-15g.

ПАРЫ. Постоянная рука M1 <f><j><s> сравнивается с импульсом той же метки
направления в каждом моменте: l0p ↔ p2l0p и p4l0p, r3p ↔ p2r3p и p4r3p и
т. д. Единица подсчёта — пара «состояние × направление», как в M1
(14 провалов × 8 направлений = 112 проверок на семейство и момент);
отдельно — покрытие состояний объединением 8 направлений семейства.

ТАБЛИЦЫ на тех же 14 провалах и 8 диагностических успехах плана:
  спасения:    оба / только постоянная / только импульс / ни один;
  повреждения: оба / только постоянная / только импульс / ни один —
    на всей диагностике и отдельно среди состояний, где q0ref дошёл до
    чанка импульса (среда, завершившаяся раньше, безопасность импульса
    не проверяет).
Каждый момент — отдельно; объединение моментов 2 и 4 — отдельной строкой
с пометкой «оракул» (лучший момент выбран по исходу).

СХВАТ: поэпизодный список переключений знака схвата относительно
собственного a0 на ИСПОЛНЕННЫХ шагах — источник, метка, задача,
состояние, чанк (с 1) и шаг (с 0, в нумерации вызовов политики).

ПРОМЕЖУТОЧНЫЕ ЗАПУСКИ: в анализ входят только готовые и ПРОВЕРЕННЫЕ блоки
K-15g (строгая validate_block, протокол импульса против q0ref того же
блока, старт каждого эпизода против q0 переписи). Не готовые и не
прошедшие проверку блоки перечисляются явно; недостающие пары НЕ
считаются ни спасением, ни провалом. Нового гейта «успеха» нет.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

H = 8
MOMENTS = (2, 4)
DIRS = [f"{j}{s}" for j in range(4) for s in ("p", "m")]
FAMS = ("l", "r")
FAM_NAME = dict(l="learned", r="control")


# --- ядро сравнения (чистые функции) -----------------------------------------
def reached(done_step, moment):
    """Достигнут ли чанк moment (с 1) средой с данным done_step."""
    return bool(done_step < 0 or done_step >= H * (moment - 1))


def categorize(const_bad, pulse_bad):
    """Оба / только постоянная / только импульс / ни один."""
    if const_bad and pulse_bad:
        return "both"
    if const_bad:
        return "const_only"
    if pulse_bad:
        return "pulse_only"
    return "neither"


def empty_counts():
    return dict(both=0, const_only=0, pulse_only=0, neither=0, missing=0)


def compare(const, pulse, ref, fails, diag):
    """const {метка M1: {(t, s): эпизод}}, pulse {метка K-15g: {(t, s):
    эпизод}}, ref {(t, s): эпизод q0ref}; эпизод — dict(success,
    done_step). Возвращает таблицы по моментам, семействам и направлениям."""
    out = dict(moments={}, pairs=[])
    for m in MOMENTS:
        mo = {}
        for f in FAMS:
            tab = dict(rescue=empty_counts(), harm_all=empty_counts(),
                       harm_reached=empty_counts(), harm_unreached_states=0,
                       per_dir={}, missing=[])
            st_c, st_p = set(), set()          # покрытие состояний
            st_known = set()
            for d in DIRS:
                cl, pl = f"{f}{d}", f"p{m}{f}{d}"
                pd = dict(rescue=empty_counts(), harm_all=empty_counts(),
                          harm_reached=empty_counts())
                for k in fails:
                    c, p = const.get(cl, {}).get(k), pulse.get(pl, {}).get(k)
                    if c is None or p is None:
                        tab["rescue"]["missing"] += 1
                        pd["rescue"]["missing"] += 1
                        tab["missing"].append([pl, list(k)])
                        continue
                    cat = categorize(c["success"], p["success"])
                    tab["rescue"][cat] += 1
                    pd["rescue"][cat] += 1
                    st_known.add(k)
                    if c["success"]:
                        st_c.add(k)
                    if p["success"]:
                        st_p.add(k)
                    out["pairs"].append(dict(
                        moment=m, family=FAM_NAME[f], direction=d,
                        task=k[0], state=k[1], kind="fail",
                        const=bool(c["success"]), pulse=bool(p["success"]),
                        category=cat))
                for k in diag:
                    c, p = const.get(cl, {}).get(k), pulse.get(pl, {}).get(k)
                    r = ref.get(k)
                    if c is None or p is None or r is None:
                        for t_ in ("harm_all", "harm_reached"):
                            tab[t_]["missing"] += 1
                            pd[t_]["missing"] += 1
                        tab["missing"].append([pl, list(k)])
                        continue
                    cat = categorize(not c["success"], not p["success"])
                    tab["harm_all"][cat] += 1
                    pd["harm_all"][cat] += 1
                    hit = reached(int(r["done_step"]), m)
                    if hit:
                        tab["harm_reached"][cat] += 1
                        pd["harm_reached"][cat] += 1
                    out["pairs"].append(dict(
                        moment=m, family=FAM_NAME[f], direction=d,
                        task=k[0], state=k[1], kind="diag",
                        const=not c["success"], pulse=not p["success"],
                        category=cat, pulse_reached=hit))
                tab["per_dir"][d] = pd
            tab["harm_unreached_states"] = sum(
                1 for k in diag if k in ref and
                not reached(int(ref[k]["done_step"]), m))
            tab["states"] = dict(known=len(st_known), const=len(st_c),
                                 pulse=len(st_p), both=len(st_c & st_p),
                                 const_only=len(st_c - st_p),
                                 pulse_only=len(st_p - st_c))
            tab["state_sets"] = dict(const=sorted(map(list, st_c)),
                                     pulse=sorted(map(list, st_p)))
            mo[FAM_NAME[f]] = tab
        out["moments"][str(m)] = mo
    # ОРАКУЛ: объединение моментов (лучший момент выбран по исходу)
    orc = {}
    for f in FAMS:
        st = [set(map(tuple, out["moments"][str(m)][FAM_NAME[f]]
                      ["state_sets"]["pulse"])) for m in MOMENTS]
        const_ = set(map(tuple, out["moments"][str(MOMENTS[0])][FAM_NAME[f]]
                         ["state_sets"]["const"]))
        u = set().union(*st)
        orc[FAM_NAME[f]] = dict(
            note="оракул: момент выбран по исходу", pulse_union=len(u),
            const=len(const_), both=len(u & const_),
            pulse_only=len(u - const_), const_only=len(const_ - u))
    out["oracle_moments_union"] = orc
    return out


# --- схват --------------------------------------------------------------------
def executed_mask(n_calls, done_step, total_steps):
    """[C, B, 8]: исполнен ли шаг t = 8c + k средой b."""
    t = np.arange(n_calls)[:, None] * H + np.arange(H)[None, :]
    end = np.where(done_step >= 0, done_step, total_steps - 1)
    return t[:, None, :] <= end[None, :, None]


def grip_switches(d, z, label, source):
    """Переключения знака схвата относительно собственного a0 на
    исполненных шагах: список событий."""
    lvl, a0 = z.get(f"k15d_level_{label}"), z.get("k15d_a0")
    if lvl is None or a0 is None:
        return None
    C = lvl.shape[0]
    m = executed_mask(C, np.asarray(z["done_step"]), len(z["actions"]))
    sw = ((lvl[..., 6] > 0) != (a0[..., 6] > 0)) & m
    ev = []
    for c, b, k in zip(*np.nonzero(sw)):
        e = d["episodes"][int(b)]
        ev.append(dict(source=source, label=label, task=int(d["task_id"]),
                       state=int(e["init_state_id"]), chunk=int(c) + 1,
                       step=int(c) * H + int(k)))
    return ev


# --- загрузка ----------------------------------------------------------------
def episodes_of(paths):
    out = {}
    for p in paths:
        d = json.load(open(p))
        for e in d["episodes"]:
            out[(int(d["task_id"]), int(e["init_state_id"]))] = dict(
                success=bool(e["success"]), done_step=int(e["done_step"]))
    return out


def load_const(snap):
    """Постоянные руки M1-cold из ПРОВЕРЕННОГО снимка."""
    by = {}
    for e in snap["pairs"]:
        if e["label"] != "q0":
            by.setdefault(e["label"], []).append(e["json"])
    return by


def load_pulse(plan, run_dir):
    """Готовые и проверенные блоки K-15g. Возвращает ({метка: [пути]},
    q0ref-пути, список непригодных/отсутствующих блоков с причиной)."""
    import k15g_local_probe as pr
    cen_q0 = pr.episodes(sorted(glob.glob(os.path.join(
        plan["census_q0_dir"], "q0_t*_i*.json"))))
    setup = pr.expected_setup()
    ok, gaps, refs = {}, [], {}
    blocks = [(int(t), int(b)) for t, bl in plan["blocks"].items()
              for b in bl]

    def start_ok(p):
        d = json.load(open(p))
        for e in d["episodes"]:
            k = (int(d["task_id"]), int(e["init_state_id"]))
            c = cen_q0.get(k)
            if c is None or any(str(e.get(f)) != str(c[0].get(f)) for f in
                                ("init_hash", "init_hash_full",
                                 "rollout_seed")):
                return False
        return True
    for t, b in blocks:
        rp = os.path.join(run_dir, f"q0ref_t{t}_i{b}.json")
        if not os.path.exists(rp):
            gaps.append(dict(label="q0ref", task=t, block=b,
                             why="нет блока q0ref"))
            continue
        prob = pr.validate_block(rp, "q0ref", t, b, require_single=True,
                                 setup=setup)
        if not prob and not start_ok(rp):
            prob = ["старт не равен q0 переписи"]
        if not prob:
            rd = json.load(open(rp))
            for e in rd["episodes"]:
                c = cen_q0[(t, int(e["init_state_id"]))][0]
                if any(str(e.get(f)) != str(c.get(f)) for f in
                       ("action_sha1", "success", "done_step")):
                    prob = ["q0ref не воспроизводит q0 переписи"]
                    break
        if prob:
            gaps.append(dict(label="q0ref", task=t, block=b,
                             why="; ".join(prob[:2])))
            continue
        refs[(t, b)] = rp
    for lab in plan["candidates"]:
        if lab == "q0ref":
            continue
        for t, b in blocks:
            p = os.path.join(run_dir, f"{lab}_t{t}_i{b}.json")
            if not os.path.exists(p):
                gaps.append(dict(label=lab, task=t, block=b, why="не готов"))
                continue
            if (t, b) not in refs:
                gaps.append(dict(label=lab, task=t, block=b,
                                 why="нет проверенного q0ref блока"))
                continue
            prob = pr.validate_block(p, lab, t, b, require_single=True,
                                     setup=setup)
            if not prob:
                d, z = pr.load(p)
                rd, rz = pr.load(refs[(t, b)])
                prob = pr.check_arm_block(d, z, lab, rz, rd)
            if not prob and not start_ok(p):
                prob = ["старт не равен q0 переписи"]
            if prob:
                gaps.append(dict(label=lab, task=t, block=b,
                                 why="; ".join(prob[:2])))
                continue
            ok.setdefault(lab, []).append(p)
    return ok, list(refs.values()), gaps


def run(snapshot, plan_path, run_dir):
    import k15f_m1cold_snapshot as msn
    import k15g_local_probe as pr
    snap = msn.verify(snapshot)
    plan = json.load(open(plan_path))
    if plan.get("kind") != pr.KIND or plan.get("smoke"):
        raise SystemExit("нужен основной план K-15g")
    prob = pr.check_plan_inputs(plan)
    if prob:
        raise SystemExit("входы плана K-15g не совпадают: " + "; ".join(
            prob[:4]))
    const_files = load_const(snap)
    pulse_files, ref_files, gaps = load_pulse(plan, run_dir)
    const = {lab: episodes_of(fl) for lab, fl in const_files.items()}
    pulse = {lab: episodes_of(fl) for lab, fl in pulse_files.items()}
    ref = episodes_of(ref_files)
    fails = [tuple(x) for x in plan["fails"]]
    diag = [tuple(x) for x in plan["diag"]]
    res = compare(const, pulse, ref, fails, diag)
    # схват: импульсы и постоянные руки на задачах плана
    events, no_log = [], []
    for lab, fl in pulse_files.items():
        for p in fl:
            d, z = pr.load(p)
            ev = grip_switches(d, z, lab, "K-15g")
            (no_log.append(p) if ev is None else events.extend(ev))
    tasks = set(plan["tasks"])
    for lab, fl in const_files.items():
        for p in fl:
            d, z = pr.load(p)
            if int(d["task_id"]) not in tasks:
                continue
            ev = grip_switches(d, z, lab, "M1-cold")
            (no_log.append(p) if ev is None else events.extend(ev))
    planned = sum(len(v) for v in plan["blocks"].values()) * (
        len(plan["candidates"]))
    res.update(
        snapshot=dict(path=snapshot, verdict=snap["verdict"]),
        plan=plan_path, run_dir=run_dir,
        coverage=dict(planned_blocks=planned,
                      verified_blocks=sum(len(v) for v in
                                          pulse_files.values())
                      + len(ref_files),
                      gaps=gaps),
        grip_switch_events=events, grip_no_log=no_log,
        note=("единица — пара «состояние × направление»; недостающие пары "
              "не считаются; объединение моментов — оракул"))
    return res


def report(res):
    cov = res["coverage"]
    print(f"  проверенных блоков K-15g: {cov['verified_blocks']}/"
          f"{cov['planned_blocks']}; не готовы или не прошли проверку: "
          f"{len(cov['gaps'])}")
    bad = [g for g in cov["gaps"] if g["why"] != "не готов"]
    for g in bad[:10]:
        print(f"    ВНИМАНИЕ {g['label']} t{g['task']} i{g['block']}: "
              f"{g['why']}")

    def fmt(c):
        return (f"оба {c['both']}, только пост. {c['const_only']}, только "
                f"имп. {c['pulse_only']}, ни один {c['neither']}"
                + (f", нет пары {c['missing']}" if c["missing"] else ""))
    for m, mo in res["moments"].items():
        print(f"  --- чанк {m}")
        for fam, t in mo.items():
            s = t["states"]
            print(f"    {fam}: спасения (пары) {fmt(t['rescue'])}")
            print(f"      состояния (объединение 8 направлений): пост. "
                  f"{s['const']}, имп. {s['pulse']}, оба {s['both']}, только "
                  f"пост. {s['const_only']}, только имп. {s['pulse_only']} "
                  f"из известных {s['known']}")
            print(f"      повреждения, вся диагностика: {fmt(t['harm_all'])}")
            print(f"      повреждения, импульс дошёл: "
                  f"{fmt(t['harm_reached'])}; состояний без импульса "
                  f"{t['harm_unreached_states']}")
            print("      по направлениям (спасения пост./имп., вред "
                  "пост./имп. где дошёл): " + ", ".join(
                      f"{d} {v['rescue']['both'] + v['rescue']['const_only']}"
                      f"/{v['rescue']['both'] + v['rescue']['pulse_only']},"
                      f"{v['harm_reached']['both'] + v['harm_reached']['const_only']}"
                      f"/{v['harm_reached']['both'] + v['harm_reached']['pulse_only']}"
                      for d, v in t["per_dir"].items()))
    for fam, o in res["oracle_moments_union"].items():
        print(f"  ОРАКУЛ (объединение чанков 2 и 4, момент выбран по исходу)"
              f" {fam}: имп. {o['pulse_union']}, пост. {o['const']}, только "
              f"имп. {o['pulse_only']}, только пост. {o['const_only']}")
    ev = res["grip_switch_events"]
    print(f"  переключения схвата относительно a0 на исполненных шагах: "
          f"{len(ev)}")
    for e in ev[:20]:
        print(f"    {e['source']} {e['label']} {e['task']}/{e['state']}: "
              f"чанк {e['chunk']}, шаг {e['step']}")
    if res["grip_no_log"]:
        print(f"  без журнала a0: {len(res['grip_no_log'])} блоков")


# --- самопроверка ------------------------------------------------------------
def selftest():
    fails = [(8, 0), (8, 1)]
    diag = [(8, 5), (9, 0)]
    E = lambda s, ds=-1: dict(success=s, done_step=ds)  # noqa: E731
    const = {f"{f}{d}": {(8, 0): E(False), (8, 1): E(True),
                         (8, 5): E(False), (9, 0): E(True, 100)}
             for f in FAMS for d in DIRS}
    # q0ref: 9/0 завершился на шаге 20 — до чанка 4 (шаг 24), но после
    # начала чанка 2 (шаг 8); 8/5 успешен на шаге 300
    ref = {(8, 0): E(False), (8, 1): E(False), (8, 5): E(True, 300),
           (9, 0): E(True, 20)}
    pulse = {}
    for m in MOMENTS:
        for f in FAMS:
            for d in DIRS:
                pulse[f"p{m}{f}{d}"] = {(8, 0): E(False), (8, 1): E(True),
                                        (8, 5): E(True, 300),
                                        (9, 0): E(True, 20)}
    # заранее известное спасение ТОЛЬКО импульсом: p4l0p на 8/0
    pulse["p4l0p"][(8, 0)] = E(True, 400)
    # отсутствующая пара: p2r1m без 8/1
    del pulse["p2r1m"][(8, 1)]
    res = compare(const, pulse, ref, fails, diag)
    L4 = res["moments"]["4"]["learned"]
    assert L4["rescue"]["pulse_only"] == 1 and L4["rescue"]["both"] == 8
    assert L4["per_dir"]["0p"]["rescue"]["pulse_only"] == 1
    assert L4["states"]["pulse_only"] == 1 and L4["states"]["pulse"] == 2
    L2 = res["moments"]["2"]["learned"]
    assert L2["rescue"]["pulse_only"] == 0
    C2 = res["moments"]["2"]["control"]
    assert C2["rescue"]["missing"] == 1 and ["p2r1m", [8, 1]] in C2["missing"]
    assert C2["rescue"]["both"] + C2["rescue"]["neither"] == 15
    # 8/5 повреждён постоянной, импульс сохранил; 9/0 — постоянная сохранила
    assert L4["harm_all"]["const_only"] == 8
    # недостигнутый импульс: 9/0 при чанке 4 не входит в «дошёл», при
    # чанке 2 входит
    assert L4["harm_reached"]["neither"] == 0 and L4["harm_all"][
        "neither"] == 8
    assert L4["harm_unreached_states"] == 1
    assert L2["harm_reached"]["neither"] == 8
    assert res["moments"]["2"]["learned"]["harm_unreached_states"] == 0
    o = res["oracle_moments_union"]["learned"]
    assert o["pulse_union"] == 2 and o["pulse_only"] == 1
    assert "оракул" in o["note"]
    # схват: завершение внутри чанка — шаги после done_step не считаются
    a0 = np.full((3, 1, H, 7), 0.5, np.float32)
    lv = a0.copy()
    lv[1, 0, :, 6] = -0.5                       # весь чанк 2 переключён
    z = dict(k15d_a0=a0, k15d_level_p2l0p=lv, done_step=np.array([10]),
             actions=np.zeros((24, 1, 7), np.float32))
    d = dict(task_id=8, episodes=[dict(init_state_id=5)])
    ev = grip_switches(d, z, "p2l0p", "K-15g")
    assert [e["step"] for e in ev] == [8, 9, 10], ev
    assert all(e["chunk"] == 2 and e["state"] == 5 for e in ev)
    z2 = dict(z, done_step=np.array([-1]))
    assert len(grip_switches(d, z2, "p2l0p", "K-15g")) == 8
    assert grip_switches(d, dict(z, k15d_a0=None), "p2l0p", "x") is None
    assert reached(-1, 4) and reached(24, 4) and not reached(23, 4)
    _loader_test()
    print("самопроверка k15g_compare_m1 пройдена: отсутствующая пара, "
          "недостигнутый импульс, завершение внутри чанка, спасение только "
          "одним вариантом, оракул, непроверенный блок исключён")
    return 0


def _loader_test():
    """Блок, не прошедший проверку, и блок без проверенного q0ref
    исключаются с причиной, а не считаются."""
    import tempfile
    import k15g_local_probe as pr
    keep = {k: getattr(pr, k) for k in ("validate_block", "check_arm_block",
                                        "load", "expected_setup")}
    with tempfile.TemporaryDirectory() as td:
        q0d = os.path.join(td, "q0")
        run_d = os.path.join(td, "run")
        os.makedirs(q0d)
        os.makedirs(run_d)

        def put(dirn, lab, t, b, succ=False):
            eps = [dict(init_state_id=s, init_hash=f"h{t}{s}",
                        init_hash_full=f"H{t}{s}", rollout_seed=101,
                        action_sha1=f"a{t}{s}", success=succ, done_step=-1)
                   for s in range(b, b + 5)]
            json.dump(dict(task_id=t, init_start=b, episodes=eps),
                      open(os.path.join(dirn, f"{lab}_t{t}_i{b}.json"), "w"))
        put(q0d, "q0", 8, 0)
        put(q0d, "q0", 9, 0)
        put(run_d, "q0ref", 8, 0)
        put(run_d, "p2l0p", 8, 0, succ=True)
        put(run_d, "p2l0m", 8, 0)
        put(run_d, "p2l0p", 9, 0)              # нет q0ref блока 9/0
        plan = dict(census_q0_dir=q0d, blocks={"8": [0], "9": [0]},
                    candidates=["q0ref", "p2l0p", "p2l0m", "p4l0p"])
        pr.expected_setup = lambda: {}
        pr.validate_block = lambda p, *a, **k: (
            ["повреждён"] if "p2l0m" in p else [])
        pr.load = lambda p: (json.load(open(p)), {})
        pr.check_arm_block = lambda *a: []
        try:
            ok, refs, gaps = load_pulse(plan, run_d)
        finally:
            for k, v in keep.items():
                setattr(pr, k, v)
        why = {(g["label"], g["task"]): g["why"] for g in gaps}
        assert list(ok) == ["p2l0p"] and len(ok["p2l0p"]) == 1
        assert why[("p2l0m", 8)] == "повреждён"
        assert why[("p2l0p", 9)] == "нет проверенного q0ref блока"
        assert why[("q0ref", 9)] == "нет блока q0ref"
        assert why[("p4l0p", 8)] == "не готов"
        assert len(refs) == 1


def main():
    ap = argparse.ArgumentParser(description="K-15g: M1 против импульсов")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--snapshot", default="reports/k15f/m1cold_snapshot.json")
    ap.add_argument("--plan", default="reports/k15g/local_plan.json")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default="reports/k15g/compare_m1.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    rd = a.run_dir
    if rd is None:
        cand = sorted(glob.glob("reports/k15g/local/run/p*/"))
        if len(cand) != 1:
            raise SystemExit(f"укажите --run-dir: найдено {cand}")
        rd = cand[0]
    res = run(a.snapshot, a.plan, rd)
    report(res)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print(f"  сводка: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
