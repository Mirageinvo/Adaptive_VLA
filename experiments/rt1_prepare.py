#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, этап 1: подготовка восьми задач (rt1_common.TASKS).

  python3 experiments/rt1_prepare.py download-cmd      # что скачать
  python3 experiments/rt1_prepare.py prepare --dataset-sha SHA

Для каждой задачи demo_clean/aloha_agilex:
  - шаговый предел из закреплённого env_cfg/task_config/_eval_step_limit.yml
    = таблице плана (иначе отказ);
  - каждый эпизод: 14 каналов action и state, конечность,
    action[t] = state[t+1] (|Δ| ≤ 1e-6), схваты в [0, 1], три камеры той же
    длины, инструкции (список строк); SHA1 файла;
  - разбиение по ЭПИЗОДАМ (не по кадрам): VAL_PER_TASK эпизодов на задачу в
    проверку, выбор фиксированным генератором (SPLIT_SEED) по отсортированным
    именам; остальные — обучение;
  - масштабы — только по обучающим эпизодам: q01/q99, min/max, mean/std
    состояния и действия по 14 каналам;
  - технические пределы суставов — из URDF тела; доля обучающих действий за
    пределами записывается (сверх допуска 1e-3 рад — отказ).
Выход: data/rt1/prep.json (с prep_id) и data/rt1/index.npz (строки обучения
и проверки: задача, эпизод, шаг, длина эпизода). Код 3 при любой проблеме.
"""
import argparse
import datetime
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rt1_common as rc  # noqa: E402

OUT_DIR = "data/rt1"
PREP = os.path.join(OUT_DIR, "prep.json")
INDEX = os.path.join(OUT_DIR, "index.npz")
VAL_PER_TASK = 5
SPLIT_SEED = 20261025
STEP_TOL = 1e-6
LIMIT_TOL = 1e-3


def step_limits(robotwin_root):
    import yaml
    p = os.path.join(robotwin_root, "env_cfg", "task_config",
                     "_eval_step_limit.yml")
    return yaml.safe_load(open(p, encoding="utf-8")), rc.sha(p)


def instructions(h):
    if "instructions" not in h:
        return []
    v = h["instructions"][()]
    v = np.asarray(v).reshape(-1)
    return [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in v]


def check_episode(h, name):
    """-> (сводка, проблемы, act[T,14], st[T,14])."""
    p = []
    act = rc.channels(h, "action")
    st = rc.channels(h, "state")
    T = len(act)
    if len(st) != T:
        p.append(f"{name}: длины action {T} и state {len(st)}")
    if not (np.isfinite(act).all() and np.isfinite(st).all()):
        p.append(f"{name}: нечисловые значения")
    d = float(np.abs(act[:-1] - st[1:]).max()) if T > 1 else 0.0
    if d > STEP_TOL:
        p.append(f"{name}: action[t] ≠ state[t+1] (max |Δ| {d:.3g})")
    for g in rc.GRIP_IDX:
        for arr, nm in ((act, "action"), (st, "state")):
            if arr[:, g].min() < -1e-6 or arr[:, g].max() > 1 + 1e-6:
                p.append(f"{name}: схват {nm}[{g}] вне [0, 1]")
    for c in rc.CAMS:
        k = f"vision/{c}/colors"
        if k not in h:
            p.append(f"{name}: нет {k}")
        elif len(h[k]) != T:
            p.append(f"{name}: {k} длины {len(h[k])} ≠ {T}")
    ins = instructions(h)
    if not ins:
        p.append(f"{name}: нет инструкций")
    return dict(T=T, step_max_abs=d, n_instructions=len(ins)), p, act, st


def stats(x):
    x = np.asarray(x, np.float64)
    return dict(q01=np.percentile(x, 1, 0).tolist(),
                q99=np.percentile(x, 99, 0).tolist(),
                min=x.min(0).tolist(), max=x.max(0).tolist(),
                mean=x.mean(0).tolist(), std=x.std(0).tolist())


def split(names, task):
    rng = np.random.default_rng([SPLIT_SEED, sum(map(ord, task))])
    val = sorted(rng.choice(len(names), VAL_PER_TASK, replace=False).tolist())
    return val


def prepare(a):
    import h5py
    root = os.path.abspath(a.robotwin_root)
    data_root = os.path.abspath(a.data_root)
    prob, rep = [], dict(created=datetime.datetime.now().isoformat(
        timespec="seconds"), robotwin_root=root, data_root=data_root,
        dataset_sha=a.dataset_sha, tasks={}, decisions=dict(
            channels=[f"{k}[{d}]" for k, d in rc.CH_KEYS], cams=rc.CAMS,
            chunk=rc.T_CHUNK, n_exec=rc.N_EXEC, tile=rc.TILE,
            pos_offset=rc.POS_OFFSET, grip_w=rc.GRIP_W,
            val_per_task=VAL_PER_TASK, split_seed=SPLIT_SEED))
    lim, lim_sha = step_limits(root)
    rep["step_limit_file_sha1"] = lim_sha
    L, U, rinfo = rc.robot_limits(root)
    rep["robot"] = dict(rinfo, L=L.tolist(), U=U.tolist())
    rows = dict(task=[], ep=[], t=[], T=[], split=[])
    files, tr_act, tr_st = [], [], []
    for ti, task in enumerate(rc.TASKS):
        if int(lim.get(task, -1)) != rc.TASKS[task]:
            prob.append(f"{task}: предел шагов {lim.get(task)} ≠ плану "
                        f"{rc.TASKS[task]}")
        d = os.path.join(data_root, rc.DATA_REL.format(task=task))
        eps = sorted(glob.glob(os.path.join(d, "*.hdf5")))
        if not eps:
            prob.append(f"{task}: нет данных в {d}")
            continue
        val = set(split(eps, task))
        tr = dict(episodes=len(eps), val=sorted(val), lengths=[],
                  step_max_abs=0.0)
        for ei, f in enumerate(eps):
            with h5py.File(f, "r") as h:
                s, p, act, st = check_episode(h, f"{task}/{os.path.basename(f)}")
            prob += p
            fid = len(files)
            files.append(dict(task=task, path=os.path.relpath(f, data_root),
                              sha1=rc.sha(f), T=s["T"],
                              split="val" if ei in val else "train"))
            tr["lengths"].append(s["T"])
            tr["step_max_abs"] = max(tr["step_max_abs"], s["step_max_abs"])
            for t in range(s["T"]):
                rows["task"].append(ti)
                rows["ep"].append(fid)
                rows["t"].append(t)
                rows["T"].append(s["T"])
                rows["split"].append(1 if ei in val else 0)
            if ei not in val:
                tr_act.append(act)
                tr_st.append(st)
        Ls = np.asarray(tr["lengths"])
        tr["length"] = dict(min=int(Ls.min()), median=float(np.median(Ls)),
                            max=int(Ls.max()))
        del tr["lengths"]
        rep["tasks"][task] = tr
        print(f"  {task}: эпизодов {len(eps)} (проверка {sorted(val)}), "
              f"длина {tr['length']}, max|a[t]−s[t+1]| "
              f"{tr['step_max_abs']:.2g}", flush=True)
    if tr_act:
        A, S = np.concatenate(tr_act), np.concatenate(tr_st)
        rep["stats_train"] = dict(action=stats(A), state=stats(S),
                                  frames=int(len(A)))
        out_lo = (A[:, rc.ARM_IDX] < L[list(rc.ARM_IDX)] - LIMIT_TOL)
        out_hi = (A[:, rc.ARM_IDX] > U[list(rc.ARM_IDX)] + LIMIT_TOL)
        frac = float((out_lo | out_hi).mean())
        rep["train_actions_outside_robot_limits"] = frac
        if frac > 0:
            prob.append(f"обучающие действия за пределами URDF: доля "
                        f"{frac:.2e}")
    rep["files"] = files
    rep["problems"] = prob
    core = json.dumps({k: rep[k] for k in ("tasks", "files", "robot",
                                           "decisions")}, sort_keys=True,
                      default=str)
    import hashlib
    rep["prep_id"] = hashlib.sha1(core.encode()).hexdigest()[:16]
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(INDEX, **{k: np.asarray(v, np.int64) for k, v in rows.items()})
    rep["index_sha1"] = rc.sha(INDEX)
    json.dump(rep, open(PREP, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=str)
    n_tr = int(np.sum(np.asarray(rows["split"]) == 0))
    print(f"  строк обучения {n_tr}, проверки {len(rows['split']) - n_tr}; "
          f"prep_id {rep['prep_id']}; {PREP}")
    print("  " + ("ОТКАЗ: " + "; ".join(prob[:8]) if prob else
                  "подготовка пройдена"))
    return 3 if prob else 0


def download_cmd(a):
    missing = [t for t in rc.TASKS if not glob.glob(os.path.join(
        a.data_root, rc.DATA_REL.format(task=t), "*.hdf5"))]
    print("отсутствуют:", missing or "нет")
    if missing:
        print(f'cd "$ROBOTWIN_ROOT" && HF_REVISION={a.dataset_sha or "<SHA>"} '
              f'HF_KEEP_ARCHIVES=0 bash scripts/download_xpolicylab_data.sh '
              + " ".join(missing))
    return 0


def selftest():
    class H(dict):
        pass
    T = 6
    st = np.cumsum(np.full((T + 1, 14), 0.01), 0)
    st[:, 6] = np.linspace(0, 1, T + 1)
    st[:, 13] = 1.0
    h = H()
    for g, arr in (("action", st[1:]), ("state", st[:-1])):
        o = 0
        for k, d in rc.CH_KEYS:
            h[f"{g}/{k}"] = arr[:, o:o + d]
            o += d
    for c in rc.CAMS:
        h[f"vision/{c}/colors"] = [b"x"] * T
    h["instructions"] = np.array([b"lift the pot"])
    s, p, act, _ = check_episode(h, "ok")
    assert p == [] and s["T"] == T and s["n_instructions"] == 1, p
    h2 = H(h)
    h2["action/left_arm_joint_states"] = h["action/left_arm_joint_states"] + 1
    assert any("state[t+1]" in x for x in check_episode(h2, "bad")[1])
    h3 = H(h)
    h3["vision/cam_left_wrist/colors"] = [b"x"] * (T - 1)
    assert any("cam_left_wrist" in x for x in check_episode(h3, "cam")[1])
    names = [f"e{i}" for i in range(50)]
    v1, v2 = split(names, "lift_pot"), split(names, "lift_pot")
    assert v1 == v2 and len(v1) == VAL_PER_TASK and \
        v1 != split(names, "open_laptop")
    print("самопроверка rt1_prepare пройдена: проверки эпизода (шаг, "
          "камеры, инструкции), воспроизводимое разбиение по эпизодам")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="RoboTwin этап 1: подготовка")
    ap.add_argument("mode", choices=("prepare", "download-cmd"))
    ap.add_argument("--robotwin-root", default=os.environ.get(
        "ROBOTWIN_ROOT", ""))
    ap.add_argument("--data-root", default=os.environ.get("RT_DATA_ROOT",
                                                          ""))
    ap.add_argument("--dataset-sha", default=None)
    a = ap.parse_args()
    if not a.robotwin_root or not a.data_root:
        raise SystemExit("нужны ROBOTWIN_ROOT и RT_DATA_ROOT")
    try:
        return prepare(a) if a.mode == "prepare" else download_cmd(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
