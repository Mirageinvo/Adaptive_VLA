#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, этап 0: опись скачанных демонстраций RoboTwin в HDF5.

  python3 experiments/rt0_data_inventory.py DATA_ROOT [--max-episodes 50]

Ищет *.hdf5/*.h5 под DATA_ROOT. Поддерживаемые расположения (оба описаны
в документации RoboTwin 2.0):
  demo_<конф>/<задача>/<тело>/data/…   (современное, XPolicyLab);
  <задача>/demo_<конф>/data/…          (старое);
файл вне этих схем — отказ с перечнем таких путей (не угадывается).
Для каждого файла обходит дерево и записывает все наборы (путь, форма,
тип). Затем по задаче:
  - число эпизодов, длины (по первой оси наборов действий), объём на диске;
  - наборы действий/состояния (имена с action, joint, qpos, endpose, state):
    размерность, по измерениям min/max/среднее/ско, 1%/99%, ско шага
    (разность соседних шагов) и число различных значений — измерения с ≤ 3
    значениями или с двумя кластерами отмечаются как вероятные схваты;
  - изображения: сырые uint8[..., 3/4]; сжатые — байтовые строки или
    uint8[T, N] / uint8[N] с сигнатурой JPEG/PNG в первом кадре (по
    сигнатуре, не по имени); для сжатых один кадр декодируется (cv2, PIL
    или imageio — что есть) и записывается разрешение.
Однокадровые эпизоды допустимы (ско шага тогда не считается).
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


def image_codec(sample):
    """jpeg/png по сигнатуре; sample — bytes или массив uint8."""
    try:
        b = bytes(sample[:8]) if isinstance(sample, (bytes, bytearray)) \
            else np.asarray(sample).reshape(-1)[:8].astype(np.uint8).tobytes()
    except Exception:  # noqa: BLE001
        return None
    if b.startswith(b"\xff\xd8"):
        return "jpeg"
    if b.startswith(b"\x89PNG"):
        return "png"
    return None


def first_frame(ds):
    """Первый элемент набора как bytes (для сжатых кадров) или None."""
    try:
        x = ds[0]
    except Exception:  # noqa: BLE001
        return None
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    a = np.asarray(x)
    if a.dtype == object and a.size:
        v = a.reshape(-1)[0]
        return bytes(v) if isinstance(v, (bytes, bytearray)) else None
    if a.dtype.kind == "S":
        return a.tobytes()
    if a.dtype == np.uint8:
        return a.reshape(-1).tobytes()
    return None


def decode_size(buf):
    """(высота, ширина, каналы) сжатого кадра или None (нет декодера)."""
    try:
        import cv2
        im = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
        if im is not None:
            return list(im.shape) + ([1] if im.ndim == 2 else [])
    except Exception:  # noqa: BLE001
        pass
    try:
        import io
        from PIL import Image
        im = Image.open(io.BytesIO(buf), encoding="utf-8")
        return [im.size[1], im.size[0], len(im.getbands())]
    except Exception:  # noqa: BLE001
        pass
    try:
        import imageio.v2 as iio
        im = iio.imread(buf)
        return list(im.shape) + ([1] if im.ndim == 2 else [])
    except Exception:  # noqa: BLE001
        return None


def describe(name, ds):
    d = dict(path=name, shape=list(ds.shape), dtype=str(ds.dtype))
    low = name.lower()
    kind = "other"
    shp = tuple(ds.shape)
    if ds.dtype.kind in ("S", "O", "V") or (ds.dtype == np.uint8 and
                                            len(shp) in (1, 2)):
        buf = first_frame(ds) if shp and shp[0] > 0 else None
        codec = image_codec(buf) if buf is not None else None
        if codec:
            kind = "image_encoded"
            d["codec"] = codec
            d["frame_bytes"] = len(buf)
            d["decoded_shape"] = decode_size(buf)
    elif ds.dtype == np.uint8 and len(shp) >= 3 and shp[-1] in (3, 4):
        kind = "image_raw"
    if kind == "other" and ds.dtype.kind == "f" and len(shp) in (1, 2) and \
            any(k in low for k in ACTION_KEYS):
        kind = "vector"
    d["kind"] = kind
    return d


def vector_stats(arrs):
    """arrs — список [T, D] по эпизодам."""
    X = np.concatenate([a.reshape(len(a), -1) for a in arrs]).astype(
        np.float64)
    dl = [np.diff(a.reshape(len(a), -1), axis=0) for a in arrs if len(a) > 1]
    dX = (np.concatenate(dl).astype(np.float64) if dl
          else np.zeros((0, X.shape[1])))
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
    """(задача, конфигурация, тело) по двум поддерживаемым расположениям;
    None — путь вне схем."""
    rel = os.path.relpath(path, root).split(os.sep)
    dirs = rel[:-1]
    ci = [i for i, p in enumerate(dirs) if p.startswith("demo_")]
    if len(ci) != 1:
        return None
    i = ci[0]
    rest = dirs[i + 1:]
    if rest and rest[0] != "data":
        # demo_<конф>/<задача>/<тело>/data/…
        emb = rest[1] if len(rest) > 1 and rest[1] != "data" else None
        return rest[0], dirs[i], emb
    if i > 0:
        # <задача>/demo_<конф>/data/…
        return dirs[i - 1], dirs[i], None
    return None


def inventory(root, files, opener, max_episodes):
    """files — пути; opener(path) -> контекст с .visititems (h5py.File)."""
    tasks = collections.OrderedDict()
    unknown = [fp for fp in files if task_of(root, fp) is None]
    if unknown:
        raise SystemExit(f"пути вне поддерживаемых расположений "
                         f"({len(unknown)}): {sorted(unknown)[:5]}")
    for fp in sorted(files):
        task, cfg, emb = task_of(root, fp)
        t = tasks.setdefault((task, cfg, emb), dict(
            task=task, config=cfg, embodiment=emb, files=0, bytes=0,
            episode_lengths=[],
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
                    t["images"][n] = {k: d[k] for k in (
                        "shape", "dtype", "kind", "codec", "frame_bytes",
                        "decoded_shape") if k in d}
    out = []
    for t in tasks.values():
        L = np.asarray(t["episode_lengths"])
        out.append(dict(
            task=t["task"], config=t["config"], embodiment=t["embodiment"],
            episodes=t["files"],
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
        print(f"\n  {t['task']} [{t['config']}, {t['embodiment']}]: "
              f"эпизодов {t['episodes']}, "
              f"{t['gb_on_disk']:.2f} ГБ ({t['gb_per_episode'] * 1024:.0f} "
              f"МБ/эп.); длина " + (f"{el['min']}–{el['max']} (медиана "
                                     f"{el['median']:.0f})" if el else "?"))
        for n, v in t["vectors"].items():
            print(f"    {n}: dim {v['dim']}, вероятные схваты "
                  f"{v['likely_gripper_dims']}")
        for n, im in t["images"].items():
            print(f"    камера {n}: хранение {im['shape']} {im['dtype']} "
                  f"{im.get('codec', im['kind'])}"
                  + (f", кадр {im['decoded_shape']}" if im.get("codec")
                     else ""))


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
    # JPEG в uint8[T, N] (по сигнатуре, без «image» в имени) и старое
    # расположение <задача>/demo_clean/data/…; однокадровый эпизод
    jb = np.frombuffer(b"\xff\xd8\xff" + b"y" * 13, np.uint8)
    trees["/d/beat_block/demo_clean/data/episode0.hdf5"] = {
        "joint_action": {"vector": rng.normal(size=(1, 14))},
        "observation": {"front": {"rgb": np.stack([jb] * 1)}}}
    res = inventory("/d", list(trees), lambda p: _File(trees[p]), 50)
    assert len(res) == 2, [(r["task"], r["config"]) for r in res]
    old_ = [r for r in res if r["task"] == "beat_block"][0]
    assert old_["config"] == "demo_clean" and old_["embodiment"] is None
    assert old_["vectors"]["joint_action/vector"]["per_dim"][0][
        "step_std"] is None
    fr = old_["images"]["observation/front/rgb"]
    assert fr["kind"] == "image_encoded" and fr["codec"] == "jpeg"
    t = [r for r in res if r["task"] == "lift_pot"][0]
    assert t["config"] == "demo_clean" and t["embodiment"] == "aloha"
    assert task_of("/d", "/d/x/y/e.hdf5") is None
    try:
        inventory("/d", ["/d/x/y/e.hdf5"], None, 50)
        raise AssertionError("путь вне схем принят")
    except SystemExit:
        pass
    assert t["episodes"] == 3 and t["episode_length"]["max"] == 120
    v = t["vectors"]["joint_action/vector"]
    assert v["dim"] == 14 and v["likely_gripper_dims"] == [6, 13], v[
        "likely_gripper_dims"]
    im = t["images"]
    assert im["observation/head_camera/rgb"]["codec"] == "jpeg"
    assert im["observation/left_camera/rgb"]["kind"] == "image_raw"
    print("самопроверка rt0_data_inventory пройдена: оба расположения, отказ "
          "вне схем, длины, однокадровый эпизод, размерность, схваты, "
          "камеры (сырые, байтовые строки, JPEG в uint8[T, N])")
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
    try:
        res = inventory(a.root, files, lambda p: h5py.File(p, "r"),
                        a.max_episodes)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise
    report(res)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(root=os.path.abspath(a.root), tasks=res), open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n  отчёт: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
