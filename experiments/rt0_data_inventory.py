#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, этап 0: опись скачанных демонстраций (HDF5 любой структуры).

  python3 experiments/rt0_data_inventory.py DATA_ROOT [--max-episodes 50]

Ищет *.hdf5/*.h5 под DATA_ROOT; задача — первый каталог пути после
DATA_ROOT (например demo_clean/<задача>/...), конфигурация — каталог
demo_* в пути. Для каждого файла обходит дерево и записывает все наборы
(путь, форма, тип). Затем по задаче:
  - число эпизодов, длины (по первой оси наборов действий), объём на диске;
  - наборы действий/состояния (имена с action, joint, qpos, endpose, state):
    размерность, по измерениям min/max/среднее/ско, 1%/99%, ско шага
    (разность соседних шагов) и число различных значений — измерения с ≤ 3
    значениями или с двумя кластерами отмечаются как вероятные схваты;
  - изображения (uint8 с последней осью 3/4 или байтовые строки — JPEG/PNG):
    камеры, форма кадра, кодирование.
Ничего не решает: это вход для выбора кодека, длины чанка и задач.
Отчёт: reports/rt0/inventory.json и сводка в терминале.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

ACTION_KEYS = ("action", "joint", "qpos", "endpose", "state")


# --- доступ к дереву (h5py или подделка для самопроверки) --------------------
def walk_h5(f):
    """[(путь, объект-набор)] для всех наборов файла h5py."""
    out = []

    def visit(name, obj):
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            out.append((name, obj))
    f.visititems(visit)
    return out


def describe(name, ds):
    d = dict(path=name, shape=list(ds.shape), dtype=str(ds.dtype))
    low = name.lower()
    kind = "other"
    if ds.dtype.kind in ("S", "O", "V") or (ds.dtype == np.uint8 and
                                            len(ds.shape) == 1):
        if any(k in low for k in ("rgb", "image", "cam", "color")):
            kind = "image_encoded"
    elif ds.dtype == np.uint8 and len(ds.shape) >= 3 and \
            ds.shape[-1] in (3, 4):
        kind = "image_raw"
    elif ds.dtype.kind == "f" and len(ds.shape) in (1, 2) and \
            any(k in low for k in ACTION_KEYS):
        kind = "vector"
    d["kind"] = kind
    return d


def image_codec(sample):
    b = bytes(sample[:8]) if not isinstance(sample, (bytes, bytearray)) \
        else bytes(sample[:8])
    if b.startswith(b"\xff\xd8"):
        return "jpeg"
    if b.startswith(b"\x89PNG"):
        return "png"
    return "unknown"


def vector_stats(arrs):
    """arrs — список [T, D] по эпизодам."""
    X = np.concatenate([a.reshape(len(a), -1) for a in arrs]).astype(
        np.float64)
    dX = np.concatenate([np.diff(a.reshape(len(a), -1), axis=0)
                         for a in arrs if len(a) > 1]).astype(np.float64)
    per = []
    for j in range(X.shape[1]):
        x = X[:, j]
        u = np.unique(np.round(x, 4))
        lo, hi = np.percentile(x, [1, 99])
        mid = (x.min() + x.max()) / 2
        two = (len(u) > 3 and x.max() - x.min() > 1e-6 and
               np.mean(np.abs(x - mid) > 0.4 * (x.max() - x.min())) > 0.9)
        per.append(dict(
            min=float(x.min()), max=float(x.max()), mean=float(x.mean()),
            std=float(x.std()), q01=float(lo), q99=float(hi),
            step_std=float(dX[:, j].std()) if len(dX) else None,
            n_unique=int(min(len(u), 1000)),
            likely_gripper=bool(len(u) <= 3 or two)))
    return dict(dim=int(X.shape[1]), frames=int(X.shape[0]), per_dim=per,
                likely_gripper_dims=[j for j, p in enumerate(per)
                                     if p["likely_gripper"]])


def task_of(root, path):
    rel = os.path.relpath(path, root).split(os.sep)
    cfg = next((p for p in rel if p.startswith("demo_")), None)
    task = rel[rel.index(cfg) + 1] if cfg and rel.index(cfg) + 1 < len(rel) \
        else rel[0]
    return task, cfg


def inventory(root, files, opener, max_episodes):
    """files — пути; opener(path) -> контекст с .visititems (h5py.File)."""
    tasks = collections.OrderedDict()
    for fp in sorted(files):
        task, cfg = task_of(root, fp)
        t = tasks.setdefault((task, cfg), dict(
            task=task, config=cfg, files=0, bytes=0, episode_lengths=[],
            schema=None, vectors=collections.defaultdict(list), images={}))
        t["files"] += 1
        t["bytes"] += os.path.getsize(fp) if os.path.exists(fp) else 0
        if t["files"] > max_episodes:
            continue
        with opener(fp) as f:
            items = walk_h5(f)
            desc = [describe(n, d) for n, d in items]
            if t["schema"] is None:
                t["schema"] = desc
            lens = [d["shape"][0] for d in desc if d["kind"] == "vector"]
            if lens:
                t["episode_lengths"].append(int(max(set(lens),
                                                    key=lens.count)))
            for (n, ds), d in zip(items, desc):
                if d["kind"] == "vector":
                    a = np.asarray(ds[()], np.float64)
                    t["vectors"][n].append(a if a.ndim == 2 else a[:, None])
                elif d["kind"].startswith("image") and n not in t["images"]:
                    im = dict(shape=d["shape"], dtype=d["dtype"],
                              kind=d["kind"])
                    if d["kind"] == "image_encoded" and d["shape"][0] > 0:
                        im["codec"] = image_codec(ds[0])
                    t["images"][n] = im
    out = []
    for t in tasks.values():
        L = np.asarray(t["episode_lengths"])
        out.append(dict(
            task=t["task"], config=t["config"], episodes=t["files"],
            gb_on_disk=t["bytes"] / 2 ** 30,
            gb_per_episode=t["bytes"] / 2 ** 30 / max(t["files"], 1),
            episodes_read=min(t["files"], max_episodes),
            episode_length=(dict(min=int(L.min()), median=float(
                np.median(L)), max=int(L.max())) if len(L) else None),
            vectors={n: vector_stats(v) for n, v in t["vectors"].items()},
            images=t["images"], schema=t["schema"]))
    return out


def report(res):
    tot = sum(t["gb_on_disk"] for t in res)
    print(f"  задач/конфигураций {len(res)}, всего {tot:.1f} ГБ")
    for t in res:
        el = t["episode_length"]
        print(f"\n  {t['task']} [{t['config']}]: эпизодов {t['episodes']}, "
              f"{t['gb_on_disk']:.2f} ГБ ({t['gb_per_episode'] * 1024:.0f} "
              f"МБ/эп.); длина " + (f"{el['min']}–{el['max']} (медиана "
                                     f"{el['median']:.0f})" if el else "?"))
        for n, v in t["vectors"].items():
            print(f"    {n}: dim {v['dim']}, вероятные схваты "
                  f"{v['likely_gripper_dims']}")
        for n, im in t["images"].items():
            print(f"    камера {n}: {im['shape']} {im['dtype']} "
                  f"{im.get('codec', im['kind'])}")


# --- самопроверка без h5py ----------------------------------------------------
class _DS:
    def __init__(self, arr):
        self.a = np.asarray(arr)
        self.shape, self.dtype = self.a.shape, self.a.dtype

    def __getitem__(self, k):
        return self.a[k]


class _File:
    def __init__(self, tree):
        self.tree = tree

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False

    def visititems(self, fn):
        def rec(prefix, node):
            for k, v in node.items():
                p = f"{prefix}/{k}" if prefix else k
                if isinstance(v, dict):
                    rec(p, v)
                else:
                    fn(p, _DS(v))
        rec("", self.tree)


def selftest():
    rng = np.random.default_rng(0)
    trees = {}
    for e in range(3):
        T = 100 + 10 * e
        arm = np.cumsum(rng.normal(0, 0.01, (T, 14)), 0)
        arm[:, 6] = np.where(rng.random(T) < .5, 0.0, 1.0)
        arm[:, 13] = np.where(rng.random(T) < .5, 0.0, 1.0)
        jpg = np.array([b"\xff\xd8\xff" + b"x" * 10] * T, dtype=object)
        trees[f"/d/demo_clean/lift_pot/aloha/data/episode{e}.hdf5"] = {
            "joint_action": {"vector": arm},
            "observation": {"head_camera": {"rgb": jpg},
                            "left_camera": {"rgb": np.zeros((T, 8, 8, 3),
                                                            np.uint8)}}}
    res = inventory("/d", list(trees), lambda p: _File(trees[p]), 50)
    assert len(res) == 1
    t = res[0]
    assert t["task"] == "lift_pot" and t["config"] == "demo_clean"
    assert t["episodes"] == 3 and t["episode_length"]["max"] == 120
    v = t["vectors"]["joint_action/vector"]
    assert v["dim"] == 14 and v["likely_gripper_dims"] == [6, 13], v[
        "likely_gripper_dims"]
    im = t["images"]
    assert im["observation/head_camera/rgb"]["codec"] == "jpeg"
    assert im["observation/left_camera/rgb"]["kind"] == "image_raw"
    print("самопроверка rt0_data_inventory пройдена: задачи и конфигурации, "
          "длины, размерность действий, схваты, камеры и кодирование")
    return 0


def main():
    ap = argparse.ArgumentParser(description="RoboTwin этап 0: опись данных")
    ap.add_argument("root", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=50,
                    help="сколько эпизодов задачи читать для статистики")
    ap.add_argument("--out", default="reports/rt0/inventory.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.root:
        raise SystemExit("нужен DATA_ROOT")
    import h5py
    files = [os.path.join(d, f) for d, _s, fs in os.walk(a.root) for f in fs
             if f.endswith((".hdf5", ".h5"))]
    if not files:
        print(f"  ОТКАЗ: под {a.root} нет файлов .hdf5/.h5")
        return 3
    res = inventory(a.root, files, lambda p: h5py.File(p, "r"),
                    a.max_episodes)
    report(res)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(root=os.path.abspath(a.root), tasks=res), open(a.out, "w"),
              ensure_ascii=False, indent=1)
    print(f"\n  отчёт: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
