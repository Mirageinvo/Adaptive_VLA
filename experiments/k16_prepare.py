#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16 (шаг 2): разметка ближайших целей и поправок по полным эпизодам.

ДАННЫЕ. Наблюдения — строки кэша h18 K-15f (h18, a0, экспертный чанк уже
посчитаны, новые проходы VLM не нужны). Полные эпизоды читаются из parquet
датасета (`state`, `actions`, `frame_index`, `timestamp`; изображения не
читаются) — ТОЛЬКО для эпизодов частей train и val_sel кэша h18;
val_confirm не открывается.

РАЗДЕЛЕНИЕ. Обучающие эпизоды делятся на две половины ВНУТРИ КАЖДОЙ задачи
(фиксированный сид): A — предсказатели цели, B — корректирующие головы (на
ПРЕДСКАЗАННЫХ целях). val_sel — отдельно. Выборка из A и из B — не больше
30 000 наблюдений, с балансом задач и пределом на эпизод; одни и те же
выборки для всех моделей.

СВЕРКА полного эпизода с кэшем на строках: состояние равно сохранённому;
нормированный экспертный чанк (рука / max(|Q99|,|Q01|), знак схвата
инвертирован, обрезка [-1, 1]) воспроизводит act кэша h18.

РАЗМЕТКА строки (эпизод e, индекс полного эпизода t):
  событие   наблюдение t записано ДО исполнения команды t; ближайшее
            УСТОЙЧИВОЕ переключение команды схвата j >= t (новый знак
            держится ≥ 2 последовательных команд; первый ненулевой знак —
            начальный режим; ноль сохраняет знак); состояние цели —
            state[j + 1]; без переключений — последнее записанное
            состояние; индексы команды и состояния хранятся отдельно;
  цель      Δpos = pos[цель] − pos[t] (мировые оси), относительное
            вращение R_t^T R_цель в непрерывном 6D (два первых столбца;
            axis-angle — только диагностика), команда схвата после
            события, тип: закрыть / открыть / конец; команд до события;
  поправка  r* = act[:8, :6] − a0[:8, :6] (единицы кодека), только если у
            строки 8 настоящих будущих действий (t + 8 <= длина);
  история   состояния в t, t−8, t−16, t−24 (шаг — 8 команд = один вызов
            политики при H = 8) с маской доступности; возраст эпизода t
            в шагах.
Переключение команды — метка этапа, а не успешный захват.

Выход: reports/k16/labels.npz и labels.json.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

H = 8
HIST = (0, 8, 16, 24)               # текущее и три предыдущих вызова
STABLE = 2
SPLIT_SEED = 20261014
CAP_ROWS = 30000
EVENT_CLOSE, EVENT_OPEN, EVENT_END = 0, 1, 2
OUT = "reports/k16/labels.npz"


# --- вращения (axis-angle <-> матрица) -----------------------------------------
def aa_to_mat(v):
    """axis-angle [..., 3] -> R [..., 3, 3] (Родриг)."""
    v = np.asarray(v, np.float64)
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    k = np.where(th > 1e-12, v / np.maximum(th, 1e-300), 0.0)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    th = th[..., None]
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def mat_to_aa(R):
    """R [..., 3, 3] -> axis-angle [..., 3], угол в [0, π]."""
    R = np.asarray(R, np.float64)
    shp = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)
    tr = np.clip((np.trace(Rf, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    th = np.arccos(tr)
    w = np.stack([Rf[:, 2, 1] - Rf[:, 1, 2], Rf[:, 0, 2] - Rf[:, 2, 0],
                  Rf[:, 1, 0] - Rf[:, 0, 1]], -1)
    out = np.zeros((len(Rf), 3))
    for n_ in range(len(Rf)):
        t_ = th[n_]
        if t_ < 1e-6:
            out[n_] = w[n_] / 2
        elif np.pi - t_ > 1e-4:
            out[n_] = w[n_] / (2 * np.sin(t_)) * t_
        else:                             # около π: ось из диагонали
            dg = np.diagonal(Rf[n_])
            ax = np.sqrt(np.clip((dg + 1) / 2, 0, None))
            i0 = int(np.argmax(ax))
            a = np.zeros(3)
            a[i0] = ax[i0]
            for j_ in range(3):
                if j_ != i0:
                    a[j_] = (Rf[n_, i0, j_] + Rf[n_, j_, i0]) / (4 * a[i0])
            a = a / np.linalg.norm(a)
            if a @ w[n_] < 0:             # знак оси — по антисимметричной части
                a = -a
            out[n_] = a * t_
    return out.reshape(shp + (3,))


def rel_mat(aa_from, aa_to):
    """Относительное вращение R_from^T R_to (матрица)."""
    Rf, Rt = aa_to_mat(aa_from), aa_to_mat(aa_to)
    return np.swapaxes(Rf, -1, -2) @ Rt


def rel_rot(aa_from, aa_to):
    """Относительное вращение в axis-angle (только диагностика)."""
    return mat_to_aa(rel_mat(aa_from, aa_to))


def mat_to_6d(R):
    """Непрерывное 6D: первые два СТОЛБЦА матрицы, [c0 (3), c1 (3)]."""
    R = np.asarray(R, np.float64)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def sixd_to_mat(x):
    """6D -> матрица (Грам — Шмидт)."""
    x = np.asarray(x, np.float64)
    a, b = x[..., :3], x[..., 3:]
    c0 = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (c0 * b).sum(-1, keepdims=True) * c0
    c1 = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([c0, c1, np.cross(c0, c1)], -1)


def rot_angle(Ra, Rb):
    """Угол между вращениями (рад)."""
    M = np.swapaxes(Ra, -1, -2) @ Rb
    return np.arccos(np.clip((np.trace(M, axis1=-2, axis2=-1) - 1) / 2,
                             -1, 1))


# --- переключения и события ------------------------------------------------------
def sign_seq(x, prev=0):
    out = []
    for v in np.asarray(x, np.float64):
        s = 1 if v > 0 else -1 if v < 0 else prev
        out.append(s)
        prev = s
    return np.asarray(out)


def stable_switches(grip, k=STABLE):
    """Индексы устойчивых переключений по ПОСЛЕДОВАТЕЛЬНЫМ командам."""
    s = sign_seq(grip)
    nz = np.nonzero(s)[0]
    if not len(nz):
        return []
    idx, cur = [], s[nz[0]]
    for t in range(nz[0] + 1, len(s)):
        if s[t] != cur and len(s) - t >= k and (s[t:t + k] == s[t]).all():
            idx.append(t)
            cur = s[t]
    return idx


def next_event(switches, grip, t, L):
    """Для наблюдения t (записано ДО исполнения команды t): (индекс команды
    события j, индекс состояния цели, тип, команда схвата после).
    Ближайшее переключение j >= t; цель — state[j + 1] (после команды
    события; устойчивость гарантирует j + 1 < L). Нет переключений —
    последнее записанное состояние, команда — последняя."""
    s = sign_seq(grip)
    for j in switches:
        if j >= t:
            return (j, j + 1, (EVENT_CLOSE if s[j] > 0 else EVENT_OPEN),
                    int(s[j]))
    return L - 1, L - 1, EVENT_END, int(s[L - 1])


def codec_norm(actions, scale):
    """Нормировка K-9a: рука / scale, знак схвата инвертирован, обрезка."""
    a = np.asarray(actions, np.float64).copy()
    a[..., :-1] /= scale[:-1]
    a[..., -1] *= -1
    return np.clip(a, -1.0, 1.0)


def label_episode(S, A, rows_t, scale):
    """S [L, 8] состояния, A [L, 7] действия (единицы среды), rows_t —
    индексы строк кэша в полном эпизоде. Возвращает разметку по строкам."""
    L = len(S)
    sw = stable_switches(A[:, 6])
    out = collections.defaultdict(list)
    for t in rows_t:
        j, s, typ, g = next_event(sw, A[:, 6], int(t), L)
        out["event_cmd_idx"].append(j)
        out["event_state_idx"].append(s)
        out["event_type"].append(typ)
        out["event_grip"].append(g)
        out["steps_to_event"].append(j - int(t))
        out["dpos"].append(S[s, :3] - S[t, :3])
        Rm = rel_mat(S[t, 3:6], S[s, 3:6])
        out["drot6d"].append(mat_to_6d(Rm))
        out["drot_aa"].append(mat_to_aa(Rm))
        hist, mask = [], []
        for d in HIST:
            if t - d >= 0:
                hist.append(S[t - d])
                mask.append(True)
            else:
                hist.append(np.zeros(S.shape[1]))
                mask.append(False)
        out["hist"].append(np.stack(hist))
        out["hist_mask"].append(mask)
        out["age"].append(int(t))
        out["full8"].append(int(t) + H <= L)
        out["ep_len"].append(L)
    return {k: np.asarray(v) for k, v in out.items()}, sw


def split_halves(ep_task, seed=SPLIT_SEED):
    """Эпизоды обучения -> половины A и B ВНУТРИ КАЖДОЙ задачи."""
    rng = np.random.default_rng(seed)
    A_, B_ = [], []
    for t in sorted(set(ep_task.values())):
        eps = sorted(e for e, tt in ep_task.items() if tt == t)
        rng.shuffle(eps)
        h = (len(eps) + 1) // 2
        A_ += eps[:h]
        B_ += eps[h:]
    return sorted(A_), sorted(B_)


def sample_rows(idx, epi, task, cap=CAP_ROWS, seed=SPLIT_SEED):
    """Не больше cap строк: поровну по задачам, предел на эпизод."""
    rng = np.random.default_rng(seed)
    tasks = sorted(set(task[idx].tolist()))
    per_task = cap // max(len(tasks), 1)
    keep = []
    for t in tasks:
        it = idx[task[idx] == t]
        eps = sorted(set(epi[it].tolist()))
        per_ep = max(1, math.ceil(per_task / max(len(eps), 1)))
        cand = []
        for e in eps:
            ie = it[epi[it] == e]
            cand += rng.permutation(ie)[:per_ep].tolist()
        cand = rng.permutation(np.asarray(cand, np.int64))[:per_task]
        keep += cand.tolist()
    if len(idx) <= cap and not keep:
        return np.sort(idx)
    return np.sort(np.asarray(keep, np.int64))


def read_episode(repo, rev, e):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    f = hf_hub_download(repo, f"data/chunk-{int(e) // 1000:03d}/"
                        f"episode_{int(e):06d}.parquet", repo_type="dataset",
                        revision=rev)
    cols = [c for c in ("state", "actions", "frame_index", "timestamp")]
    t = pq.read_table(f, columns=cols)
    return dict(S=np.asarray(t.column("state").to_pylist(), np.float64),
                A=np.asarray(t.column("actions").to_pylist(), np.float64),
                frame=np.asarray(t.column("frame_index").to_pylist()),
                ts=np.asarray(t.column("timestamp").to_pylist(), np.float64))


def episode_problems(ep, need_idx, e):
    """Полный эпизод пригоден: формы, индексы кадров 0..L-1, конечность,
    возрастающее время, запрошенные индексы внутри эпизода."""
    p = []
    S, A_ = ep["S"], ep["A"]
    frame = np.asarray(ep["frame"]).reshape(-1)
    ts = np.asarray(ep["ts"], np.float64).reshape(-1)
    L = len(S)
    if S.ndim != 2 or S.shape[1] != 8:
        p.append(f"эпизод {e}: состояния формы {S.shape}")
    if A_.shape != (L, 7):
        p.append(f"эпизод {e}: действия формы {A_.shape}, ожидалось "
                 f"({L}, 7)")
    if not np.issubdtype(frame.dtype, np.integer) or \
            not np.array_equal(frame, np.arange(L)):
        p.append(f"эпизод {e}: frame_index не равен 0..{L - 1}")
    if len(ts) != L or not (np.isfinite(S).all() and np.isfinite(A_).all()
                            and np.isfinite(ts).all()):
        p.append(f"эпизод {e}: нет конечных состояний/действий/времени")
    elif L > 1 and not (np.diff(ts) > 0).all():
        p.append(f"эпизод {e}: время не возрастает")
    need = np.asarray(need_idx).reshape(-1)
    if len(need) and (need.min() < 0 or need.max() >= L):
        p.append(f"эпизод {e}: запрошенные индексы вне [0, {L - 1}]")
    return p


def pilot_tasks():
    """Формулировки 10 задач LIBERO-10 (пилот)."""
    from libero.libero import benchmark
    b = benchmark.get_benchmark_dict()["libero_10"]()
    return [b.get_task(i).language for i in range(b.n_tasks)]


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
    c = kb.load_cache(a.h18_cache, verify_h18=False)
    parts = {}
    for p in kb.PARTS:
        rows = c[p]["rows"]
        parts[p] = dict(rows=rows, epi=epi[rows], step=stp[rows],
                        task=task[rows])
    tr_eps = {int(e): str(t) for e, t in zip(parts["train"]["epi"],
                                             parts["train"]["task"])}
    A_eps, B_eps = split_halves(tr_eps)
    if set(A_eps) & set(B_eps) or set(A_eps + B_eps) & set(
            parts["val_sel"]["epi"].tolist()):
        raise SystemExit("половины или val_sel пересекаются по эпизодам")
    problems, labels = [], {}
    eps_all = sorted(set(parts["train"]["epi"].tolist())
                     | set(parts["val_sel"]["epi"].tolist()))
    print(f"  эпизодов к чтению: {len(eps_all)} (только train и val_sel)")
    ep_info = {}
    for j, e in enumerate(eps_all):
        ep = read_episode(repo, rev, e)
        L = len(ep["S"])
        need = np.concatenate([parts[p]["step"][parts[p]["epi"] == e]
                               for p in kb.PARTS])
        pe = episode_problems(ep, need, e)
        if pe:
            problems += pe
            continue
        for p in kb.PARTS:
            m = np.nonzero(parts[p]["epi"] == e)[0]
            if not len(m):
                continue
            ts_ = parts[p]["step"][m]
            # сверка с кэшем: состояния и нормированный экспертный чанк
            if not np.allclose(ep["S"][ts_], np.asarray(
                    st_cache[parts[p]["rows"][m]], np.float64), atol=1e-5):
                problems.append(f"эпизод {e}: состояние не равно кэшу")
            for k_, t_ in zip(m, ts_):
                n8 = min(H, L - int(t_))
                if not np.allclose(codec_norm(ep["A"][t_:t_ + n8], scale),
                                   c[p]["act"][k_, :n8], atol=1e-5):
                    problems.append(f"эпизод {e} шаг {t_}: экспертный чанк "
                                    f"не равен кэшу")
                    break
            lab, sw = label_episode(ep["S"], ep["A"], ts_, scale)
            labels.setdefault(p, []).append((m, lab))
        dts = np.diff(np.asarray(ep["ts"]).reshape(-1))
        ep_info[e] = dict(len=L, dt=float(np.median(dts)) if L > 1 else None,
                          dt_min=float(dts.min()) if L > 1 else None,
                          dt_max=float(dts.max()) if L > 1 else None)
        if j % 200 == 0:
            print(f"  эпизодов {j}/{len(eps_all)}", flush=True)
    if problems:
        raise SystemExit("сверка с кэшем не прошла: " + "; ".join(
            problems[:6]))
    out = {}
    for p in kb.PARTS:
        n = len(parts[p]["rows"])
        merged = {}
        for m, lab in labels[p]:
            for k, v in lab.items():
                if k not in merged:
                    merged[k] = np.zeros((n,) + v.shape[1:], v.dtype)
                merged[k][m] = v
        merged["rstar"] = (c[p]["act"][:, :H, :6].astype(np.float64)
                           - c[p]["a0"][:, :H, :6].astype(np.float64))
        out[p] = merged
    # половины и выборки
    tr = parts["train"]
    ok = out["train"]["full8"].astype(bool)
    idxA = np.nonzero(np.isin(tr["epi"], A_eps) & ok)[0]
    idxB = np.nonzero(np.isin(tr["epi"], B_eps) & ok)[0]
    selA = sample_rows(idxA, tr["epi"], tr["task"])
    selB = sample_rows(idxB, tr["epi"], tr["task"], seed=SPLIT_SEED + 1)
    selV = np.nonzero(out["val_sel"]["full8"].astype(bool))[0]
    try:
        pilot = pilot_tasks()
    except Exception as e_:                  # noqa: BLE001
        pilot = []
        problems.append(f"список задач LIBERO-10 недоступен: {e_}")

    def cover(sel, part):
        cnt = collections.Counter(parts[part]["task"][sel].tolist())
        return dict(rows=int(len(sel)), tasks=len(cnt),
                    episodes=int(len(set(parts[part]["epi"][sel].tolist()))),
                    pilot_rows={t: int(cnt.get(t, 0)) for t in pilot},
                    pilot_missing=[t for t in pilot if not cnt.get(t)])
    arrays = dict(selA=selA, selB=selB, selV=selV,
                  epA=np.asarray(A_eps), epB=np.asarray(B_eps))
    for p in kb.PARTS:
        for k, v in out[p].items():
            arrays[f"{p}__{k}"] = v
        arrays[f"{p}__rows"] = parts[p]["rows"]
        arrays[f"{p}__epi"] = parts[p]["epi"]
        arrays[f"{p}__step"] = parts[p]["step"]
        arrays[f"{p}__task"] = parts[p]["task"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **arrays)
    ev = collections.Counter(out["train"]["event_type"][selA].tolist())
    rep = dict(
        kind="k16_labels", split_seed=SPLIT_SEED, cap_rows=CAP_ROWS,
        history_offsets=list(HIST), stable=STABLE,
        h18_cache=os.path.abspath(a.h18_cache),
        h18_manifest_sha1=hashlib.sha1(open(os.path.join(
            a.h18_cache, "manifest.json"), "rb").read()).hexdigest(),
        dataset=dict(repo=repo, revision=rev),
        episodes=dict(A=len(A_eps), B=len(B_eps),
                      val_sel=int(len(set(parts["val_sel"]["epi"])))),
        coverage=dict(A=cover(selA, "train"), B=cover(selB, "train"),
                      val_sel=cover(selV, "val_sel")),
        dropped_not_full8=dict(
            train=int((~ok).sum()),
            val_sel=int((~out["val_sel"]["full8"].astype(bool)).sum())),
        events_A={["закрыть", "открыть", "конец"][k]: int(v)
                  for k, v in sorted(ev.items())},
        timestamp_step=dict(
            median_counts=dict(collections.Counter(
                round(v["dt"], 4) for v in ep_info.values() if v["dt"])),
            min=min(v["dt_min"] for v in ep_info.values() if v["dt_min"]),
            max=max(v["dt_max"] for v in ep_info.values() if v["dt_max"])),
        conventions=dict(
            observation="наблюдение t записано ДО исполнения команды t",
            event="ближайшее устойчивое переключение команды j >= t",
            goal_state="state[j + 1] (после команды события); без "
                       "переключений — последнее записанное состояние",
            rotation="относительное R_t^T R_goal в 6D (два первых "
                     "столбца); axis-angle — только диагностика; ошибка "
                     "ориентации — угол между матрицами"),
        inputs=["h18", "a0", "hist", "hist_mask", "age"],
        labels_not_inputs=["event_cmd_idx", "event_state_idx",
                           "steps_to_event", "ep_len", "dpos", "drot6d",
                           "drot_aa", "event_type", "event_grip"],
        problems=problems,
        out=dict(path=os.path.abspath(a.out),
                 sha1=hashlib.sha1(open(a.out, "rb").read()).hexdigest()),
        code=hashlib.sha1(open(os.path.abspath(__file__), "rb").read())
        .hexdigest()[:12],
        note="переключение команды схвата — метка этапа, не захват")
    with open(a.out[:-4] + ".json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: rep[k] for k in ("episodes", "coverage",
                                          "dropped_not_full8", "events_A",
                                          "timestamp_step", "problems")},
                     ensure_ascii=False, indent=1))
    return 3 if problems else 0


# --- самопроверка ------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(0)
    # вращения: туда-обратно, относительное, около π
    for _ in range(200):
        v = rng.standard_normal(3)
        v = v / np.linalg.norm(v) * rng.uniform(0, np.pi - 1e-3)
        assert np.allclose(mat_to_aa(aa_to_mat(v)), v, atol=1e-6)
    v = np.array([0.0, 0.0, np.pi - 1e-6])
    assert np.allclose(aa_to_mat(mat_to_aa(aa_to_mat(v))), aa_to_mat(v),
                       atol=1e-5)
    vb = rng.standard_normal((4, 5, 3))           # пакетная форма
    assert np.allclose(aa_to_mat(mat_to_aa(aa_to_mat(vb))), aa_to_mat(vb),
                       atol=1e-6)
    a1, a2 = rng.standard_normal(3), rng.standard_normal(3)
    assert np.allclose(rel_rot(a1, a1), 0, atol=1e-6)
    r = rel_rot(a1, a2)
    assert np.allclose(aa_to_mat(a1) @ aa_to_mat(r), aa_to_mat(a2),
                       atol=1e-6)
    # около π с отрицательной осью: восстановление матрицы точное
    vn = np.array([0.0, -1.0, 0.0]) * (np.pi - 9e-5)
    assert np.abs(aa_to_mat(mat_to_aa(aa_to_mat(vn))) - aa_to_mat(vn)
                  ).max() < 1e-7
    # 6D: туда-обратно и угол между матрицами
    Rr = aa_to_mat(rng.standard_normal((10, 3)))
    assert np.allclose(sixd_to_mat(mat_to_6d(Rr)), Rr, atol=1e-9)
    assert np.allclose(rot_angle(Rr, Rr), 0, atol=1e-6)
    # углы, разные по знаку около π, — близкие вращения
    p1, p2 = np.array([3.1, 0, 0]), np.array([-3.1, 0, 0])
    assert np.linalg.norm(rel_rot(p1, p2)) < 0.1
    assert np.linalg.norm(p1 - p2) > 6           # простое вычитание неверно
    # переключения и события
    g = np.array([0, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1])
    sw = stable_switches(g)
    assert sw == [4, 10], sw                     # 7 — одиночный всплеск
    # перед переключением, на нём, после
    assert next_event(sw, g, 3, 12) == (4, 5, EVENT_CLOSE, 1)
    assert next_event(sw, g, 4, 12) == (4, 5, EVENT_CLOSE, 1)
    assert next_event(sw, g, 5, 12) == (10, 11, EVENT_OPEN, -1)
    assert next_event(sw, g, 11, 12) == (11, 11, EVENT_END, -1)
    # разметка эпизода: цель, история, хвост
    L = 30
    S = np.zeros((L, 8))
    S[:, 0] = np.arange(L) * 0.01
    S[:, 3] = 3.0
    A = np.zeros((L, 7))
    A[:, 6] = -1
    A[12:, 6] = 1
    lab, _sw = label_episode(S, A, np.array([5, 20, 25]), np.ones(7))
    assert lab["event_cmd_idx"].tolist() == [12, 29, 29]
    assert lab["event_state_idx"].tolist() == [13, 29, 29]
    assert np.isclose(lab["dpos"][0, 0], 0.08)
    assert lab["drot6d"].shape == (3, 6)
    assert lab["hist_mask"].tolist() == [[True, False, False, False],
                                         [True, True, True, False],
                                         [True, True, True, True]]
    assert np.allclose(lab["hist"][1, 1], S[12])
    assert lab["full8"].tolist() == [True, True, False]
    # проверка полного эпизода
    good = dict(S=np.zeros((5, 8)), A=np.zeros((5, 7)),
                frame=np.arange(5).reshape(5, 1),
                ts=np.arange(5) * 0.1)
    assert episode_problems(good, [0, 4], 0) == []
    for mut, word in ((dict(frame=np.array([[0], [2], [4], [6], [8]])),
                       "frame_index"),
                      (dict(frame=np.arange(5) * 1.0), "frame_index"),
                      (dict(A=np.zeros((5, 6))), "действия"),
                      (dict(ts=np.array([0, .1, .1, .3, .4])), "время"),
                      (dict(S=np.full((5, 8), np.nan)), "конечных")):
        pr_ = episode_problems(dict(good, **mut), [0], 0)
        assert any(word in x for x in pr_), (mut, pr_)
    assert episode_problems(good, [5], 0)          # индекс вне эпизода
    # нормировка кодека
    sc = np.array([0.5] * 6 + [1.0])
    x = codec_norm(np.array([[0.25, 0, 0, 0, 0, 1.0, -1.0]]), sc)
    assert np.allclose(x, [[0.5, 0, 0, 0, 0, 1.0, 1.0]])
    # половины внутри задач и без пересечения
    ep_task = {e: f"t{e % 3}" for e in range(30)}
    A_, B_ = split_halves(ep_task)
    assert not set(A_) & set(B_) and len(A_) + len(B_) == 30
    for t in ("t0", "t1", "t2"):
        na = sum(ep_task[e] == t for e in A_)
        nb = sum(ep_task[e] == t for e in B_)
        assert abs(na - nb) <= 1
    assert (A_, B_) == split_halves(ep_task)     # детерминирован
    # выборка: предел, баланс задач, предел на эпизод
    n = 5000
    epi = rng.integers(0, 50, n)
    task = np.asarray([f"t{e % 5}" for e in epi])
    sel = sample_rows(np.arange(n), epi, task, cap=1000)
    assert len(sel) <= 1000
    cnt = collections.Counter(task[sel].tolist())
    assert max(cnt.values()) - min(cnt.values()) <= 1
    per_ep = collections.Counter(epi[sel].tolist())
    assert max(per_ep.values()) <= math.ceil(200 / 10)
    print("самопроверка k16_prepare пройдена: вращения (в том числе около "
          "π), устойчивые переключения, ближайшее событие и конец, "
          "история с маской, хвост без 8 действий, нормировка кодека, "
          "половины внутри задач, выборка с балансом")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-16: разметка")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--h18-cache", default="data/k15f/h18_cache")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        return main_prepare(a)
    except SystemExit as e:               # технический отказ — код 3
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
