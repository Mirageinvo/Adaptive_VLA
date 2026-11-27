#!/usr/bin/env python3
"""K-43: второй перенос из состояния T32 после первой постановки (t8).

Этап 1 (этот файл, v0): CPU-аудит точки передачи на готовых трассах K-42
(reports/k42/audit/v1/traces, T1 = T32 c2 задачи 8) — без новых раскаток.

Точка передачи tau — граница чанка (tau % 8 == 0), tau — число исполненных
команд после ожидания; state[tau] — текущее наблюдение. Решение на tau
читает только наблюдения 0…tau и команды A[:tau]:
  progress  одна кофеварка On в наблюдениях tau−7…tau (stable = 8), вторая
            в tau не On;
  release   три последние команды схвата A[tau−3:tau] — open, апертура
            пальцев state[tau] ≥ open_min, контакта удержания с кофеварками
            нет (grasp == 0; неизвестный −1 — отдельная причина unknown);
  time      MAX_STEPS − tau ≥ budget (128);
  active    tau < числа исполненных команд (граница на terminal исключена).
Первая граница со всеми условиями — развилка (одна на эпизод); геометрия
места на плите в аудите не проверяется — охват оптимистичный.
Причина недоступности — самый дальний невыполненный этап: progress →
release/unknown → time. Порог апертуры open_min — --open-min или авто:
0,8 × медианы апертуры после ≥ 16 подряд open-команд по всем трассам.

Конец эпизода используется только для разметки исхода после развилки:
поднимал ли T32 вторую кофеварку (подъём ≥ pot_lift или удержание), осталась
ли первая On до конца, длина успешного суффикса.

Команды:
  python3 experiments/k43_handoff.py --selftest
  python3 experiments/k43_handoff.py audit [--traces DIR] [--archive DIR]
                                           [--open-min M] [--partial]
  python3 experiments/k43_handoff.py fit-templates [--n-src 24] [--workers 4]
  python3 experiments/k43_handoff.py smoke --device cuda:0 [--n-dev 4]
Этап 2 (шаблоны, опция второго переноса, smoke) — k43_transfer.py.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

TASK = 8
H8 = 8
MAX_STEPS = 600
OPEN, CLOSE = -1.0, 1.0
TRACES = Path("reports/k42/audit/v1/traces")
ARCHIVE = Path("reports/k37c/runs/c2")
OUT = Path("reports/k43/handoff/v1")
CFG = dict(stable=8, n_open=3, budget=128, pot_lift=0.015, open_run=16,
           open_frac=0.8)
REASONS = ("progress", "release", "unknown", "time")


def aperture(state):
    s = np.asarray(state, np.float64)
    return np.abs(s[..., 0]) + np.abs(s[..., 1])


def auto_open_min(trs, cfg=CFG):
    """0,8 × медиана апертуры в наблюдениях после ≥ open_run подряд open."""
    vals = []
    for tr in trs:
        g = np.asarray(tr["A"], np.float64)[:, 6] == OPEN
        ap = aperture(tr["state"])
        run = 0
        for t, o in enumerate(g):
            run = run + 1 if o else 0
            if run >= cfg["open_run"] and np.isfinite(ap[t + 1]):
                vals.append(ap[t + 1])
    if not vals:
        raise SystemExit("нет участков с длинным open — задайте --open-min")
    med = float(np.median(vals))
    if not np.isfinite(med) or med <= 0:
        raise SystemExit(f"авто-порог апертуры {med} — задайте --open-min")
    return cfg["open_frac"] * med, dict(n=len(vals), median=med,
                                        p5=float(np.quantile(vals, 0.05)))


def check_trace(tr):
    """Согласованность форм трассы -> список проблем."""
    n = len(tr["A"])
    p = []
    if tr["A"].shape != (n, 7):
        p.append(f"A {tr['A'].shape}")
    for k, sh in (("state", (n + 1, 9)), ("pot", (n + 1, 2, 3)),
                  ("grasp", (n + 1, 2)), ("on", (n + 1, 2))):
        if tuple(tr[k].shape) != sh:
            p.append(f"{k} {tuple(tr[k].shape)} ≠ {sh}")
    if n > MAX_STEPS:
        p.append(f"шагов {n} > {MAX_STEPS}")
    for k in ("A", "state", "pot"):
        if not np.isfinite(np.asarray(tr[k], np.float64)).all():
            p.append(f"{k}: нечисловые значения")
    if tr["success"] and not bool(tr["on"][-1].all()):
        p.append("успех без двух On в конце")
    return p


def boundary_flags(tr, tau, open_min, cfg=CFG):
    """Условия передачи на границе tau по данным ≤ tau -> dict."""
    A, on, gr = tr["A"], tr["on"], tr["grasp"]
    n = len(A)
    k0 = tau - cfg["stable"] + 1
    first = None
    if k0 >= 0:
        for k in range(on.shape[1]):
            others = [j for j in range(on.shape[1]) if j != k]
            if on[k0:tau + 1, k].all() and not any(on[tau, j] for j in others):
                first = k
                break
    cmd_open = tau >= cfg["n_open"] and \
        bool((np.asarray(A[tau - cfg["n_open"]:tau, 6]) == OPEN).all())
    ap_ok = bool(aperture(tr["state"][tau]) >= open_min)
    unknown = bool((gr[tau] == -1).any())
    free = bool((gr[tau] == 0).all())
    return dict(progress=first is not None, first=first,
                release=cmd_open and ap_ok and free, unknown=unknown,
                time=MAX_STEPS - tau >= cfg["budget"], active=tau < n)


def handoff(tr, open_min, cfg=CFG):
    """Первая развилка эпизода -> dict(tau, first, second) | dict(reason)."""
    n = len(tr["A"])
    stage = 0
    for tau in range(H8, n, H8):           # tau < n: граница не terminal
        f = boundary_flags(tr, tau, open_min, cfg)
        if not f["progress"]:
            continue
        stage = max(stage, 1)
        if not f["release"]:
            if f["unknown"]:
                stage = max(stage, 1.5)
            continue
        stage = max(stage, 2)
        if not f["time"]:
            continue
        k = f["first"]
        return dict(tau=tau, first=k, second=1 - k)
    reason = {0: "progress", 1: "release", 1.5: "unknown", 2: "time"}[stage]
    return dict(tau=None, reason=reason)


def outcome(tr, h, cfg=CFG):
    """Разметка после развилки (только для отчёта)."""
    tau, k, j = h["tau"], h["first"], h["second"]
    pot, gr, on = tr["pot"], tr["grasp"], tr["on"]
    rise = float(pot[tau:, j, 2].max() - pot[tau, j, 2])
    kept1 = bool(on[tau:, k].all())
    n = len(tr["A"])
    return dict(contact_second=bool((gr[tau:, j] == 1).any()),
                lifted_second=rise >= cfg["pot_lift"],
                second_on_ever=bool(on[tau:, j].any()),
                first_kept=kept1,
                first_left_at=None if kept1 else
                int(tau + np.argmin(on[tau:, k])),
                first_on_end=bool(on[-1, k]),
                suffix=n - tau, both_on_end=bool(on[-1].all()))


def episode_row(tr, open_min, cfg=CFG):
    h = handoff(tr, open_min, cfg)
    on_end = int(tr["on"][-1].sum())
    max_on = int(tr["on"].sum(1).max())
    r = dict(eid=tr["eid"], seed=tr["seed"], case_id=tr["case_id"],
             success=bool(tr["success"]), steps=len(tr["A"]), handoff=h,
             n_on_end=on_end, max_on=max_on)
    if h["tau"] is not None:
        r["after"] = outcome(tr, h, cfg)
    return r


def need_fe(N, S_e, r, l, wins=5, n_pilot=96):
    """Минимум доступных провалов F_e: 96 (F_e r − S_e l) / N ≥ wins."""
    return int(np.ceil((wins * N / n_pilot + S_e * l) / r - 1e-9))


def summarize(rows):
    out = {}
    for sd in ["all", *sorted({r["seed"] for r in rows})]:
        g = rows if sd == "all" else [r for r in rows if r["seed"] == sd]
        N = len(g)
        fail = [r for r in g if not r["success"]]
        succ = [r for r in g if r["success"]]
        fe = [r for r in fail if r["handoff"]["tau"] is not None]
        se = [r for r in succ if r["handoff"]["tau"] is not None]
        F_e, S_e = len(fe), len(se)
        d = dict(N=N, fail=len(fail), success=len(succ), F_e=F_e, S_e=S_e,
                 ceiling_pp=100 * F_e / N if N else 0.0)
        d["why_fail"] = dict(Counter(r["handoff"].get("reason") for r in fail
                                     if r["handoff"]["tau"] is None))
        d["why_success"] = dict(Counter(r["handoff"].get("reason") for r in succ
                                        if r["handoff"]["tau"] is None))
        d["fail_stage"] = dict(Counter(f"on_end{r['n_on_end']}/max{r['max_on']}"
                                       for r in fail))
        d["fe_after"] = dict(
            contact_second=sum(r["after"]["contact_second"] for r in fe),
            lifted_second=sum(r["after"]["lifted_second"] for r in fe),
            second_on_ever=sum(r["after"]["second_on_ever"] for r in fe),
            first_kept=sum(r["after"]["first_kept"] for r in fe),
            first_left_back=sum(not r["after"]["first_kept"] and
                                r["after"]["first_on_end"] for r in fe),
            first_lost_end=sum(not r["after"]["first_on_end"] for r in fe),
            tau_q=np.quantile([r["handoff"]["tau"] for r in fe],
                              [0, .5, 1]).tolist() if fe else None)
        d["se_suffix_q"] = np.quantile([r["after"]["suffix"] for r in se],
                                       [.5, .9, 1]).tolist() if se else None
        d["expected_wins"] = {f"r={r_},l={l_}": 96 * (F_e * r_ - S_e * l_) / N
                              if N else 0.0
                              for r_, l_ in ((1, 0), (0.4, 0), (0.4, 0.01))}
        d["need_F_e"] = {f"r={r_},l={l_}": need_fe(N, S_e, r_, l_) if N else None
                         for r_, l_ in ((1, 0), (0.4, 0), (0.4, 0.01))}
        out[str(sd)] = d
    return out


def print_summary(s, open_min, info):
    print(f"  open_min {open_min * 1e3:.1f} мм ({info})")
    for sd, d in s.items():
        print(f"\n  сид {sd}: N {d['N']}, провалов {d['fail']}, успехов {d['success']}")
        print(f"    F_e (провалы с точкой передачи) {d['F_e']}/{d['fail']}, "
              f"S_e (успехи) {d['S_e']}/{d['success']}; потолок F_e/N "
              f"{d['ceiling_pp']:.1f} п.п.")
        print(f"    недоступно на провалах: {d['why_fail']}; на успехах: "
              f"{d['why_success']}")
        print(f"    стадия провалов (On в конце / max On): {d['fail_stage']}")
        fa = d["fe_after"]
        print(f"    среди F_e: контакт со второй {fa['contact_second']}, подъём "
              f"второй {fa['lifted_second']}, вторая хоть раз On "
              f"{fa['second_on_ever']}; первая On непрерывно "
              f"{fa['first_kept']}, выходила и вернулась {fa['first_left_back']}, "
              f"не On в конце {fa['first_lost_end']}; tau min/med/max {fa['tau_q']}")
        print(f"    успешный суффикс после tau (p50/p90/max): {d['se_suffix_q']}")
        print("    ожидаемые чистые выигрыши на 96: " + ", ".join(
            f"{k} {v:+.1f}" for k, v in d["expected_wins"].items()))
        print("    нужно F_e для 5 выигрышей: " + ", ".join(
            f"{k} ≥{v}" for k, v in d["need_F_e"].items()) +
            " — выполнено: " + ", ".join(
                f"{k} {'да' if d['F_e'] >= v else 'нет'}"
                for k, v in d["need_F_e"].items()))


def audit(a):
    import k36_protocol as pc
    import k42_audit as ka
    paths = sorted(p for p in Path(a.traces).glob("*.npz")
                   if not p.name.endswith(".tmp.npz"))
    if not paths:
        raise SystemExit(f"нет трасс в {a.traces}")
    trs = [ka.load_trace(p) for p in paths]
    prob = []
    for pth, t in zip(paths, trs):
        if ka.trace_name(dict(seed=t["seed"], case_id=t["case_id"])) != pth.stem:
            prob.append(f"{t['eid']}: имя файла ≠ сид/сцене")
        prob += [f"{t['eid']}: {x}" for x in check_trace(t)]
    # сверка с архивом и для --partial (по наличным трассам)
    jobs = ka.archive_jobs(Path(a.archive))
    want = {j["want"]["eid"]: j for j in jobs}
    got = {t["eid"]: t for t in trs}
    if len(got) != len(trs):
        dup = sorted(e for e, c in Counter(t["eid"] for t in trs).items() if c > 1)
        raise SystemExit(f"дубликаты eid в трассах: {dup[:3]}")
    if not set(got) <= set(want):
        raise SystemExit("есть трассы вне архива c2")
    if not a.partial and set(want) != set(got):
        raise SystemExit(f"трассы ≠ архиву: есть {len(got)}, нужно {len(want)} "
                         "(или --partial)")
    import k31_collect as kcol
    from k39_failure_probe import reference
    for e, j in ((e, want[e]) for e in sorted(got)):
        t = got[e]
        w = j["want"]
        if (t["npz_sha1"], t["success"], t["seed"], t["case_id"]) != (
                j["npz_sha1"], bool(j["success"]), int(w["seed"]),
                w["case_id"]):
            prob.append(f"{e}: хеш шарда/итог/сид/сцена ≠ архиву")
            continue
        rec, z = reference(Path(a.archive), j["part"], w)
        za = np.asarray(z["actions"], np.float32)
        q = rec["summary"]
        if len(t["A"]) != len(za) or len(t["A"]) != int(q["steps"]) or \
                kcol.actions_sha(t["A"]) != kcol.actions_sha(za) or \
                ("actions_sha1" in q and
                 kcol.actions_sha(t["A"]) != q["actions_sha1"]):
            prob.append(f"{e}: команды/число шагов трассы ≠ архиву")
    if prob:
        raise SystemExit("трассы: " + "; ".join(prob[:4]))
    if a.open_min is not None:
        open_min, info = float(a.open_min), "задан"
    else:
        open_min, info = auto_open_min(trs)
    cfg = dict(CFG, budget=int(a.budget)) if a.budget is not None else CFG
    rows = [episode_row(t, open_min, cfg) for t in trs]
    s = summarize(rows)
    print(f"  K-43 аудит точки передачи: трасс {len(trs)}, сверены с архивом"
          f"{' (неполный набор)' if a.partial else ''}; "
          f"условия {cfg}")
    print_summary(s, open_min, info)
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / ("audit_partial.json" if a.partial else "audit.json")
    if a.budget is not None and cfg["budget"] != CFG["budget"]:
        dst = dst.with_name(dst.stem + f"_budget{cfg['budget']}.json")
    pc.save_json(str(dst), dict(cfg=cfg, open_min=open_min,
                                               open_info=info, summary=s,
                                               rows=rows, partial=bool(a.partial)))
    print(f"\n  записано {dst}")


# --- самопроверка ---------------------------------------------------------------------
def _toy(n=400, place1=None, release_at=None, place2=None, lose1=None,
         lift2=None, unknown_at=None, success=False):
    """place1 — шаг, с которого первая (0) On; release_at — шаг open-команд
    и раскрытия; place2 — вторая On; lose1 — первая теряет On."""
    A = np.zeros((n, 7))
    A[:, 6] = CLOSE
    state = np.zeros((n + 1, 9))
    state[:, 0], state[:, 1] = 0.01, -0.01
    pot = np.zeros((n + 1, 2, 3))
    pot[:, :, 2] = 0.9
    grasp = np.zeros((n + 1, 2), np.int8)
    on = np.zeros((n + 1, 2), bool)
    if place1 is not None:
        on[place1:, 0] = True
        grasp[:place1, 0] = 1
    if release_at is not None:
        A[release_at:, 6] = OPEN
        state[release_at + 1:, 0], state[release_at + 1:, 1] = 0.04, -0.04
    if place2 is not None:
        on[place2:, 1] = True
    if lose1 is not None:
        on[lose1:, 0] = False
    if lift2 is not None:
        pot[lift2:, 1, 2] = 0.95
    if unknown_at is not None:
        grasp[unknown_at:, :] = -1
    return dict(A=A, state=state, pot=pot, grasp=grasp, on=on, eid="e",
                seed=0, case_id="c", success=success)


def selftest():
    om = 0.06
    # нормальная передача: первая On с 100, раскрытие с 90 -> первая граница
    # с 8 On подряд — 112 (наблюдения 105…112)
    t = _toy(place1=105, release_at=90, lift2=200)
    h = handoff(t, om)
    assert h == dict(tau=112, first=0, second=1), h
    assert check_trace(t) == []
    o = outcome(t, h)
    assert o["lifted_second"] and o["first_kept"] and o["suffix"] == 400 - 112
    assert not o["contact_second"] and not o["second_on_ever"]
    # контакт со второй без подъёма — не подъём
    tc = _toy(place1=105, release_at=90)
    tc["grasp"][200:, 1] = 1
    oc = outcome(tc, handoff(tc, om))
    assert oc["contact_second"] and not oc["lifted_second"]
    # нечисловые значения отвергаются, авто-порог их пропускает
    tn = _toy(place1=105, release_at=90)
    tn["state"][300, 0] = np.nan
    assert any("нечисловые" in x for x in check_trace(tn))
    assert np.isfinite(auto_open_min([tn])[0])
    # устойчивость: On только 7 наблюдений к границе -> следующая граница
    assert handoff(_toy(place1=106, release_at=90), om)["tau"] == 120
    # нет первой постановки
    assert handoff(_toy(), om) == dict(tau=None, reason="progress")
    # схват не раскрыт
    assert handoff(_toy(place1=105), om)["reason"] == "release"
    # команда open, но пальцы закрыты (апертура ниже порога)
    t2 = _toy(place1=105, release_at=90)
    t2["state"][:, 0], t2["state"][:, 1] = 0.01, -0.01
    assert handoff(t2, om)["reason"] == "release"
    # контакт неизвестен
    assert handoff(_toy(place1=105, release_at=90, unknown_at=0), om)["reason"] \
        == "unknown"
    # мало времени: первая On с 480
    assert handoff(_toy(n=600, place1=480, release_at=470), om)["reason"] == "time"
    # граница на terminal исключена: эпизод из 112 команд
    assert handoff(_toy(n=112, place1=105, release_at=90), om)["tau"] is None
    assert handoff(_toy(n=113, place1=105, release_at=90), om)["tau"] == 112
    # вторая уже On к границе -> не передача (обе стоят)
    assert handoff(_toy(place1=105, release_at=90, place2=106), om)["tau"] is None
    # смена идентичности: первой стала кофеварка 1
    t3 = _toy(release_at=90)
    t3["on"][105:, 1] = True
    assert handoff(t3, om) == dict(tau=112, first=1, second=0)
    # потеря первой после развилки
    t4 = _toy(place1=105, release_at=90, lose1=150)
    o4 = outcome(t4, handoff(t4, om))
    assert not o4["first_kept"] and o4["first_left_at"] == 150
    assert not o4["first_on_end"]
    # кратковременный выход первой из On с возвратом
    t6 = _toy(place1=105, release_at=90)
    t6["on"][150:160, 0] = False
    o6 = outcome(t6, handoff(t6, om))
    assert not o6["first_kept"] and o6["first_on_end"]
    # удержание (grasp == 1) первой на границе -> не свободен
    t5 = _toy(place1=105, release_at=90)
    t5["grasp"][100:120, 0] = 1
    assert handoff(t5, om)["tau"] == 120
    # авто-порог апертуры
    om_, info = auto_open_min([_toy(place1=105, release_at=90)])
    assert abs(om_ - 0.8 * 0.08) < 1e-9 and info["n"] > 0
    # сводка и пороги F_e
    rows = [episode_row(dict(_toy(place1=105, release_at=90), eid=f"f{i}"), om)
            for i in range(9)]
    rows += [episode_row(dict(_toy(n=300, place1=105, release_at=90,
                                   place2=250, success=True), eid=f"s{i}"), om)
             for i in range(3)]
    rows += [episode_row(dict(_toy(), eid=f"p{i}"), om) for i in range(4)]
    s = summarize(rows)["0"]
    assert (s["N"], s["F_e"], s["S_e"], s["fail"]) == (16, 9, 3, 13), s
    assert s["why_fail"] == {"progress": 4}
    assert need_fe(160, 123, 1, 0) == 9 and need_fe(160, 123, 0.4, 0) == 21
    assert need_fe(160, 123, 0.4, 0.01) == 24
    print("k43_handoff selftest ok")
    import k43_transfer as kt
    kt.selftest()


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    ap = argparse.ArgumentParser(description="K-43: второй перенос t8")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--traces", type=Path, default=TRACES)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--open-min", type=float, default=None)
    p.add_argument("--partial", action="store_true")
    p.add_argument("--budget", type=int, default=None,
                   help="минимальный остаток шагов (иначе 128); отчёт — audit_budgetN.json")
    p = sub.add_parser("fit-templates")
    p.add_argument("--traces", type=Path, default=TRACES)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--n-src", type=int, default=24)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("smoke")
    p.add_argument("--device", required=True)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--n-dev", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "audit":
        audit(a)
        return
    import k43_transfer as kt
    sys.exit(kt.fit_templates(a) if a.cmd == "fit-templates" else kt.smoke(a))


if __name__ == "__main__":
    main()
