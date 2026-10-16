#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-17 (шаг 1): окна сохраняемой цели на 24 команды и паспорт данных.

ЦЕЛЬ. Для наблюдения перед командой t — поза руки в state[t + 24] (после
исполнения команд t..t+23): Δp = p[t+24] − p[t] и R_t^T R[t+24] в 6D.
Схват не предсказывается. Горизонт — в командах (24 = три вызова по 8).

ОКНО — позиции нижнего уровня t, t+8, t+16 (три вызова). Для каждой нужны
h18 и a0 кэша K-15f по ТОЧНОМУ ключу (эпизод, шаг) — соседние строки
прореженного кэша соседними вызовами не считаются — и экспертная поправка
чанка r* = act − a0 (8 × 6, единицы кодека). Если state[t+24] нет
(хвост) — окно исключается, конец демонстрации не подставляется; если в
кэше нет t+8 или t+16 — окно исключается для всех моделей. Причины
исключения считаются отдельно.

РАЗДЕЛЕНИЕ — прежнее K-16 (половины A и B обучающих эпизодов, val_sel);
val_confirm не открывается. Полные эпизоды — из parquet (только состояния
и действия), сверка с кэшем на строках окон.

ВЫХОД: reports/k17/windows.npz (окна, индексы строк кэша, цель, позиции,
история, r*) и reports/k17/passport.json.
"""
import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k16_prepare as kp  # noqa: E402

GOAL = 24
POS = (0, 8, 16)
HIST = (8, 16, 24)
OUT = "reports/k17/windows.npz"


def episode_windows(S, A, step_to_row, L):
    """S [L, 8], A [L, 7]; step_to_row — {шаг: строка кэша} эпизода.
    Возвращает (окна, причины исключения по якорным строкам)."""
    sw = set(kp.stable_switches(A[:, 6]))
    wins, why = [], collections.Counter()
    for t in sorted(step_to_row):
        tail = t + GOAL > L - 1
        gap = any((t + d) not in step_to_row for d in POS[1:])
        if tail or gap:
            why["tail" if tail and not gap else "cache_gap" if gap and
                not tail else "tail_and_gap"] += 1
            continue
        Rm = kp.rel_mat(S[t, 3:6], S[t + GOAL, 3:6])
        hist, mask = [], []
        for d in POS:
            h_, m_ = [], []
            for o in HIST:
                if t + d - o >= 0:
                    h_.append(S[t + d - o])
                    m_.append(True)
                else:
                    h_.append(np.zeros(S.shape[1]))
                    m_.append(False)
            hist.append(np.stack(h_))
            mask.append(m_)
        wins.append(dict(
            t=t, rows=[step_to_row[t + d] for d in POS],
            dpos=S[t + GOAL, :3] - S[t, :3], drot6d=kp.mat_to_6d(Rm),
            angle_deg=float(np.degrees(np.arccos(np.clip(
                (np.trace(Rm) - 1) / 2, -1, 1)))),
            dist_m=float(np.linalg.norm(S[t + GOAL, :3] - S[t, :3])),
            grip_switch=bool(any(t <= j < t + GOAL for j in sw)),
            states=np.stack([S[t + d] for d in POS]),
            hist=np.stack(hist), hist_mask=np.asarray(mask),
            age=np.asarray([t + d for d in POS])))
    return wins, why


SPLITS = ("A", "B", "V")


def k16_link_problems(rep, L16, labels_sha, man_sha, cache_rows, train_eps,
                      val_eps):
    """Связь разметки K-16 с текущим кэшем: kind, проблемы, SHA разметки и
    манифеста h18, строки частей, покрытие обучающих эпизодов A/B."""
    p = []
    if rep.get("kind") != "k16_labels":
        p.append(f"labels.json: kind {rep.get('kind')!r}")
    if rep.get("problems"):
        p.append(f"labels.json: проблемы разметки {rep['problems'][:3]}")
    if (rep.get("out") or {}).get("sha1") != labels_sha:
        p.append("labels.npz не совпадает с SHA из labels.json")
    if rep.get("h18_manifest_sha1") != man_sha:
        p.append("манифест h18 не совпадает с сохранённым в разметке K-16")
    for part, rows in cache_rows.items():
        k = f"{part}__rows"
        if k not in L16 or not np.array_equal(np.asarray(L16[k]),
                                              np.asarray(rows)):
            p.append(f"{k} разметки K-16 не равны строкам текущего кэша")
    eA = set(np.asarray(L16.get("epA", [])).tolist())
    eB = set(np.asarray(L16.get("epB", [])).tolist())
    if eA & eB:
        p.append("половины A и B пересекаются")
    if (eA | eB) != set(train_eps):
        p.append(f"A ∪ B ≠ обучающие эпизоды кэша "
                 f"({len(eA | eB)} против {len(set(train_eps))})")
    if (eA | eB) & set(val_eps):
        p.append("A/B пересекаются с val_sel")
    return p


def main_prepare(a):
    import k15f_build_cache as kb
    import k15h_protocol as pc
    meta = json.load(open(f"{a.cache}.meta.json"))
    src = meta["cache"]
    d = np.load(src, allow_pickle=True)
    N = int(meta["n_obs"])
    epi = np.asarray(d["episode"])[:N].astype(np.int64)
    stp = np.asarray(d["step"])[:N].astype(np.int64)
    task = np.asarray(d["task"])[:N].astype(str)
    st_cache = np.load(src + ".state.npy", mmap_mode="r")
    man = meta.get("manifest") or {}
    repo, rev = man["dataset_repo"], man["dataset_revision"]
    scale = pc.harness_scale()
    L16 = dict(np.load(a.k16_labels, allow_pickle=False))
    epA, epB = set(L16["epA"].tolist()), set(L16["epB"].tolist())
    c = kb.load_cache(a.h18_cache, verify_h18=False)
    split_of, parts = {}, {}
    for p in kb.PARTS:
        rows = c[p]["rows"]
        parts[p] = dict(epi=epi[rows], step=stp[rows], task=task[rows])
    lj = a.k16_labels[:-4] + ".json"
    link = k16_link_problems(
        json.load(open(lj)), L16,
        hashlib.sha1(open(a.k16_labels, "rb").read()).hexdigest(),
        hashlib.sha1(open(os.path.join(a.h18_cache, "manifest.json"),
                          "rb").read()).hexdigest(),
        {p: c[p]["rows"] for p in kb.PARTS},
        parts["train"]["epi"].tolist(), parts["val_sel"]["epi"].tolist())
    if link:
        raise SystemExit("; ".join(link))
    for e in set(parts["train"]["epi"].tolist()):
        split_of[e] = "A" if e in epA else "B" if e in epB else None
    for e in set(parts["val_sel"]["epi"].tolist()):
        if e in split_of:
            raise SystemExit("эпизод и в train, и в val_sel")
        split_of[e] = "V"
    if any(v is None for v in split_of.values()):
        raise SystemExit("эпизод train вне половин A/B K-16")
    if epA & epB:
        raise SystemExit("половины A и B пересекаются")
    W = collections.defaultdict(list)
    why = {s: collections.Counter() for s in ("A", "B", "V")}
    anchors = collections.Counter()
    problems = []
    eps_all = sorted(split_of)
    print(f"  эпизодов к чтению: {len(eps_all)}")
    for j, e in enumerate(eps_all):
        sp = split_of[e]
        part = "val_sel" if sp == "V" else "train"
        m = np.nonzero(parts[part]["epi"] == e)[0]
        ep = kp.read_episode(repo, rev, e)
        pe = kp.episode_problems(ep, parts[part]["step"][m], e)
        if pe:
            problems += pe
            continue
        L = len(ep["S"])
        s2r = {int(s_): int(r_) for s_, r_ in zip(parts[part]["step"][m], m)}
        anchors[sp] += len(s2r)
        wins, wy = episode_windows(ep["S"], ep["A"], s2r, L)
        why[sp].update(wy)
        for w in wins:
            for k_, r_ in enumerate(w["rows"]):
                t_ = w["t"] + POS[k_]
                gr = c[part]["rows"][r_]
                if not np.allclose(ep["S"][t_], np.asarray(
                        st_cache[gr], np.float64), atol=1e-5) or \
                        not np.allclose(kp.codec_norm(ep["A"][t_:t_ + 8],
                                                      scale),
                                        c[part]["act"][r_], atol=1e-5):
                    problems.append(f"эпизод {e} шаг {t_}: кэш ≠ parquet")
                    break
            w.update(episode=e, task=str(parts[part]["task"][m[0]]),
                     part=part,
                     cache_rows=[int(c[part]["rows"][r_]) for r_ in w["rows"]],
                     rstar=np.stack([c[part]["act"][r_][:8, :6].astype(
                         np.float64) - c[part]["a0"][r_][:8, :6].astype(
                         np.float64) for r_ in w["rows"]]))
            W[sp].append(w)
        if j % 200 == 0:
            print(f"  эпизодов {j}/{len(eps_all)}", flush=True)
    if problems:
        print("  ОТКАЗ: " + "; ".join(problems[:6]))
        return 3
    arrays, pp = {}, {}
    try:
        pilot = kp.pilot_tasks()
    except Exception as ex:                   # noqa: BLE001
        print(f"  ОТКАЗ: нет списка задач LIBERO-10 ({ex}); нужен "
              f"PYTHONPATH=$HOME/LIBERO")
        return 3
    all_tasks = sorted(set(task.tolist()))
    for sp in SPLITS:
        ws = W.get(sp, [])
        for k in ("t", "rows", "dpos", "drot6d", "angle_deg", "dist_m",
                  "grip_switch", "states", "hist", "hist_mask", "age",
                  "episode", "task", "rstar", "cache_rows"):
            arrays[f"{sp}__{k}"] = np.asarray([w[k] for w in ws])
        tc = collections.Counter(w["task"] for w in ws)
        dist = np.asarray([w["dist_m"] for w in ws])
        ang = np.asarray([w["angle_deg"] for w in ws])
        pp[sp] = dict(
            windows=len(ws), episodes=len({w["episode"] for w in ws}),
            corrector_examples=3 * len(ws),
            anchors=int(anchors[sp]), excluded=dict(why[sp]),
            tasks=len(tc), missing_tasks=[t for t in all_tasks if not tc[t]],
            pilot_missing=[t for t in pilot if not tc[t]],
            pilot_windows={t: int(tc[t]) for t in pilot},
            task_windows={t: int(tc[t]) for t in all_tasks},
            task_episodes={t: len({w["episode"] for w in ws
                                   if w["task"] == t}) for t in all_tasks},
            dist_m=dict(p10=float(np.percentile(dist, 10)),
                        median=float(np.median(dist)),
                        p90=float(np.percentile(dist, 90))) if len(dist)
            else None,
            angle_deg=dict(p10=float(np.percentile(ang, 10)),
                           median=float(np.median(ang)),
                           p90=float(np.percentile(ang, 90))) if len(ang)
            else None,
            grip_switch_share=float(np.mean([w["grip_switch"]
                                             for w in ws])) if ws else None)
    empty = [sp for sp in SPLITS if not pp[sp]["windows"]]
    eps = {sp: {w["episode"] for w in W.get(sp, [])} for sp in SPLITS}
    disjoint = not (eps.get("A", set()) & eps.get("B", set())
                    or eps.get("A", set()) & eps.get("V", set())
                    or eps.get("B", set()) & eps.get("V", set()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **arrays)
    rep = dict(kind="k17_windows", goal_horizon_commands=GOAL,
               positions=list(POS), history_offsets=list(HIST),
               splits=pp, episodes_disjoint=disjoint, empty_splits=empty,
               ready=not empty and disjoint,
               conventions=dict(
                   rows=("<split>__rows — строки части кэша K-15f (train для "
                         "A/B, val_sel для V) на позициях t, t+8, t+16; "
                         "<split>__cache_rows — строки исходного кэша K-9a"),
                   goal="state[t+24] после команд t..t+23; Δp и R_t^T R в 6D",
                   positions="t, t+8, t+16 — по точному ключу (эпизод, шаг)",
                   exclusion=("tail — нет state[t+24]; cache_gap — нет "
                              "строки кэша t+8 или t+16; tail_and_gap — оба")),
               inputs=dict(k16_labels=os.path.abspath(a.k16_labels),
                           k16_labels_sha1=hashlib.sha1(open(
                               a.k16_labels, "rb").read()).hexdigest(),
                           h18_manifest_sha1=hashlib.sha1(open(os.path.join(
                               a.h18_cache, "manifest.json"), "rb").read())
                           .hexdigest()),
               out=dict(path=os.path.abspath(a.out), sha1=hashlib.sha1(
                   open(a.out, "rb").read()).hexdigest()),
               code=hashlib.sha1(open(os.path.abspath(__file__), "rb")
                                 .read()).hexdigest())
    with open(a.out[:-4] + ".json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    print(json.dumps(dict(splits={k: {kk: vv for kk, vv in v.items()
                                      if kk not in ("pilot_windows", "task_windows",
                                                    "task_episodes")}
                                  for k, v in pp.items()},
                          episodes_disjoint=disjoint, empty_splits=empty),
                     ensure_ascii=False, indent=1))
    if empty or not disjoint:
        print(f"  НЕ ГОТОВО: пустые части {empty}, эпизоды "
              f"{'не пересекаются' if disjoint else 'ПЕРЕСЕКАЮТСЯ'}")
        return 3
    return 0


def selftest():
    rng = np.random.default_rng(0)
    L = 60
    S = np.zeros((L, 8))
    S[:, 0] = np.arange(L) * 0.01
    S[:, 3:6] = rng.standard_normal(3) * 0.1
    A = np.zeros((L, 7))
    A[:, 6] = -1
    A[30:, 6] = 1                                  # переключение на 30
    # кэш: шаги 0, 8, 16, 10, 18, 26(нет 26 -> пробел для 10), 30, 38, 46
    steps = [0, 8, 16, 10, 18, 30, 38, 46, 40]
    s2r = {s: i for i, s in enumerate(steps)}
    wins, why = episode_windows(S, A, s2r, L)
    ts = sorted(w["t"] for w in wins)
    assert ts == [0, 30], ts                       # 10: нет 26; 46: хвост
    assert why["cache_gap"] >= 1 and why["tail_and_gap"] + why["tail"] >= 1
    w0 = [w for w in wins if w["t"] == 0][0]
    assert np.isclose(w0["dpos"][0], 0.24) and w0["rows"] == [0, 1, 2]
    assert not w0["grip_switch"]
    w30 = [w for w in wins if w["t"] == 30][0]
    assert w30["grip_switch"]                      # переключение на 30
    assert w0["hist_mask"].tolist() == [[False] * 3, [True, False, False],
                                        [True, True, False]]
    assert np.allclose(w0["hist"][2, 0], S[8])
    # окно с переключением внутри [t, t+24)
    s2 = {s: i for i, s in enumerate([10, 18, 26])}
    w2, _ = episode_windows(S, A, s2, L)
    assert w2[0]["t"] == 10 and w2[0]["grip_switch"]
    # хвост: t + 24 = L - 1 допустимо, L — нет
    s3 = {s: i for i, s in enumerate([35, 43, 51, 36, 44, 52])}
    w3, why3 = episode_windows(S, A, s3, L)
    assert [w["t"] for w in w3] == [35] and why3["tail"] == 1
    # связь с разметкой K-16
    L16 = dict(epA=np.array([1, 2]), epB=np.array([3]),
               train__rows=np.array([0, 1, 2]), val_sel__rows=np.array([5]))
    rep = dict(kind="k16_labels", problems=[], out=dict(sha1="x"),
               h18_manifest_sha1="m")
    cr = dict(train=np.array([0, 1, 2]), val_sel=np.array([5]))
    ok = k16_link_problems(rep, L16, "x", "m", cr, [1, 2, 3], [9])
    assert ok == [], ok
    assert k16_link_problems(dict(rep, kind="z"), L16, "x", "m", cr,
                             [1, 2, 3], [9])
    assert k16_link_problems(dict(rep, problems=["p"]), L16, "x", "m", cr,
                             [1, 2, 3], [9])
    assert k16_link_problems(rep, L16, "y", "m", cr, [1, 2, 3], [9])
    assert k16_link_problems(rep, L16, "x", "n", cr, [1, 2, 3], [9])
    assert k16_link_problems(rep, L16, "x", "m",
                             dict(cr, train=np.array([0, 1, 3])),
                             [1, 2, 3], [9])
    assert k16_link_problems(rep, L16, "x", "m", cr, [1, 2, 3, 4], [9])
    assert k16_link_problems(rep, L16, "x", "m", cr, [1, 2, 3], [3])
    print("самопроверка k17_prepare пройдена: точные позиции t/t+8/t+16, "
          "пропуск кэша, хвост без подстановки, цель t+24, история, "
          "переключение схвата в окне, связь с разметкой K-16")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-17: окна и паспорт")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--h18-cache", default="data/k15f/h18_cache")
    ap.add_argument("--k16-labels", default="reports/k16/labels.npz")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        return main_prepare(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
