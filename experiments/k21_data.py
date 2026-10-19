#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21 (prepare): индексы, маски хвостов, состояние 6D, учитель, паспорт.

Строки и проверки содержимого — k20_data.make_index (кэш K-9a против K-15f
и разметки K-16: эпизоды, первые 8 экспертных действий, train ∩ val_sel = ∅);
train — все строки эпизодов A ∪ B (train_fit ∪ train_error K-20 —
обучающие), val_sel — прежние 62 эпизода; val_confirm не открывается.
МАСКА ХВОСТА: команда k строки (эпизод, t) настоящая, если t + k < L_actions
(длина эпизода — из метаданных parquet, без изображений).
СОСТОЯНИЕ: 11 чисел (позиция, вращение 6D, пальцы); нормировка позиции и
пальцев — по train.
УЧИТЕЛЬ: K-20 (5120 строк полного BAR), если паспорт prepare K-20 завершён
и отпечатки совпадают; иначе — тот же сбор (k20_data.run_teacher).
ВЫХОД: data/k21/prep.npz, reports/k21/prepare.json.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT = "data/k21/prep.npz"
REPORT = "reports/k21/prepare.json"
T = 20


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def episode_lengths(meta, episodes):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    man = meta.get("manifest") or {}
    repo, rev = man["dataset_repo"], man["dataset_revision"]
    out = {}
    for j, e in enumerate(sorted(set(int(x) for x in episodes))):
        f = hf_hub_download(repo, f"data/chunk-{e // 1000:03d}/"
                            f"episode_{e:06d}.parquet", repo_type="dataset",
                            revision=rev)
        out[e] = int(pq.read_metadata(f).num_rows)
        if j % 300 == 0:
            print(f"  длины эпизодов {j}", flush=True)
    return out


def tail_mask(step, ep_len):
    k = np.arange(T)[None]
    return (np.asarray(step)[:, None] + k) < np.asarray(ep_len)[:, None]


def device_passport():
    rep = {}
    try:
        import peft
        rep["peft"] = peft.__version__
    except ImportError:
        rep["peft"] = None
    import torch
    rep["torch"] = torch.__version__
    try:
        import transformers
        rep["transformers"] = transformers.__version__
    except ImportError:
        rep["transformers"] = None
    gp = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            with torch.cuda.device(i):
                bf = bool(torch.cuda.is_bf16_supported())
            gp.append(dict(index=i, name=pr.name,
                           memory_gb=round(pr.total_memory / 2 ** 30, 1),
                           capability=f"{pr.major}.{pr.minor}",
                           bf16_reported=bf, bf16_native=pr.major >= 8))
    rep["gpus"] = gp
    return rep


def teacher_k20(ix, k20_prep="reports/k20/prepare.json",
                k20_teacher="data/k20/teacher.npz"):
    """Учитель K-20 с проверкой происхождения или None."""
    if not (os.path.exists(k20_prep) and os.path.exists(k20_teacher)):
        return None, "нет готового учителя K-20"
    r = json.load(open(k20_prep))
    if not r.get("complete") or sha(k20_teacher) != r.get("teacher_sha1"):
        return None, "учитель K-20 не завершён или изменён"
    t = dict(np.load(k20_teacher))
    if not np.array_equal(t["rows"], ix["teacher"]):
        return None, "строки учителя K-20 ≠ текущему выбору"
    return t, dict(source="K-20", revision=r.get("bar", {}).get("revision"),
                   sha1=r["teacher_sha1"])


def check_teacher(t, rows_train, L, U):
    n = len(t["rows"])
    p = []
    if len(np.unique(t["rows"])) != n:
        p.append("повтор строк учителя")
    if not np.isin(t["rows"], rows_train).all():
        p.append("строки учителя вне train")
    for k in ("raw", "target"):
        if t[k].shape != (n, T, 7) or not np.isfinite(t[k]).all():
            p.append(f"учитель {k}: форма или конечность")
    want = t["raw"].copy()
    want[..., :6] = np.clip(t["raw"][..., :6], L, U)
    if not np.array_equal(want, t["target"]):
        p.append("цель учителя ≠ clip(raw, L, U)")
    return p


def main_prepare(a):
    import k20_data as kd
    import k15h_protocol as hpc
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    rep = dict(kind="k21_prepare", device=device_passport())
    print(json.dumps(rep["device"], ensure_ascii=False), flush=True)
    ix = kd.make_index(argparse.Namespace(cache=a.cache, h18_cache=a.h18_cache,
                                          k16_labels=a.k16_labels))
    d, N = ix["d"], ix["N"]
    tr, va = ix["rows"]["train"], ix["rows"]["val_sel"]
    keys = set(zip(ix["epi"][np.r_[tr, va]].tolist(),
                   ix["step"][np.r_[tr, va]].tolist()))
    if len(keys) != len(tr) + len(va):
        raise SystemExit("повтор ключа (эпизод, шаг)")
    k18 = json.load(open(a.k18_config))
    lim = k18["env_limits"]
    scale = hpc.harness_scale()
    L, U = kd.bounds_codec(lim["input_min"], lim["input_max"], scale)
    act = ix["act"]
    st8 = np.asarray(np.load(ix["src"] + ".state.npy", mmap_mode="r")[:N],
                     np.float64)
    if st8.shape[1] != 8 or not np.isfinite(st8).all():
        raise SystemExit(f"состояние формы {st8.shape} или не конечно")
    import k21_model as km21
    s11 = km21.state11(st8)
    idx = [0, 1, 2, 9, 10]
    mean = s11[tr][:, idx].mean(0)
    std = np.maximum(s11[tr][:, idx].std(0), 1e-4)
    used = np.r_[tr, va]
    lens = episode_lengths(ix["meta"], ix["epi"][used])
    ep_len = np.array([lens.get(int(e), -1) for e in ix["epi"]])
    valid = np.zeros((N, T), bool)
    valid[used] = tail_mask(ix["step"][used], ep_len[used])
    if (valid[used].sum(1) == 0).any():
        raise SystemExit("строки без действительных позиций")
    gz = int((act[used][..., 6] == 0).sum())
    if not np.isfinite(act[used]).all():
        raise SystemExit("экспертные действия не конечны")
    t, src = teacher_k20(ix)
    if t is None:
        print(f"  учитель: {src} — сбор заново", flush=True)
        bar = kd.load_bar(a.device)
        raw, feas, codes, rate, spd = kd.run_teacher(bar, ix, L, U)
        t = dict(rows=ix["teacher"], raw=raw, target=feas, codes=codes)
        src = dict(source="K-21", revision=bar["revision"],
                   out_of_bounds_rate=rate, seconds_per_obs=spd)
    prob = check_teacher(t, tr, L, U)
    if prob:
        raise SystemExit("; ".join(prob))
    try:
        import k16_prepare as kp16
        pilot = kp16.pilot_tasks()
    except Exception:                                        # noqa: BLE001
        pilot = None
    vt = set(ix["task"][va].tolist())
    np.savez(OUT, train_rows=tr, val_rows=va, episode=ix["epi"],
             step=ix["step"], task=ix["task"],
             pos_offset=np.asarray(d["pos_offset"])[:N], state11=s11,
             valid=valid, L=L, U=U, state_mean=mean, state_std=std,
             scale=scale, teacher_rows=t["rows"], teacher_raw=t["raw"],
             teacher_target=t["target"],
             cache=np.array(ix["src"]), images=np.array(ix["img"]))
    rep.update(
        rows=dict(train=int(len(tr)), val_sel=int(len(va))),
        episodes=dict(train=int(len(set(ix["epi"][tr].tolist()))),
                      val_sel=int(len(set(ix["epi"][va].tolist())))),
        tasks=dict(train=int(len(set(ix["task"][tr].tolist()))),
                   val_sel=int(len(vt))),
        val_sel_missing_tasks=sorted(set(ix["task"][tr].tolist()) - vt),
        pilot_missing_in_val=[x for x in (pilot or []) if x not in vt],
        tail=dict(rows_with_padding=int((~valid[used]).any(1).sum()),
                  padded_positions=int((~valid[used]).sum())),
        grip_zero_targets=gz,
        command_range=dict(min=act[used][..., :6].reshape(-1, 6).min(0)
                           .tolist(), max=act[used][..., :6].reshape(-1, 6)
                           .max(0).tolist()),
        bounds_codec=dict(L=L.tolist(), U=U.tolist()),
        state_norm=dict(mean=mean.tolist(), std=std.tolist()),
        teacher=dict(n=int(len(t["rows"])), **(src if isinstance(src, dict)
                                               else {})),
        volume=dict(updates=1000 + 8000, max_updates=1000 + 16000,
                    effective_batch=32,
                    note="наблюдений за 9000 обновлений: 288 тыс."),
        out=dict(path=os.path.abspath(OUT), sha1=sha(OUT)))
    json.dump(rep, open(REPORT, "w"), ensure_ascii=False, indent=1,
              default=float)
    print(json.dumps({k: rep[k] for k in ("rows", "episodes", "tasks",
                                          "val_sel_missing_tasks",
                                          "pilot_missing_in_val", "tail",
                                          "grip_zero_targets", "teacher",
                                          "bounds_codec")},
                     ensure_ascii=False, indent=1, default=float))
    print(f"  паспорт: {REPORT}")
    return 0


# --- загрузка и батчи ------------------------------------------------------------
def load_prepared(path=OUT, report=REPORT):
    if not os.path.exists(path) or not os.path.exists(report):
        raise SystemExit("prepare K-21 не выполнен")
    r = json.load(open(report))
    if r["out"]["sha1"] != sha(path):
        raise SystemExit("prep.npz изменён после prepare")
    z = dict(np.load(path, allow_pickle=False))
    meta_cache = str(z["cache"])
    D = dict(z, train_rows=z["train_rows"], cache_path=meta_cache)
    D["act"] = np.asarray(np.load(meta_cache, allow_pickle=True)["action"],
                          np.float32)
    D["IMG"] = np.load(str(z["images"]), mmap_mode="r")
    D["state8"] = np.load(meta_cache + ".state.npy", mmap_mode="r")
    return D


def make_batch(D, rows, bar, dev, teacher_idx=None):
    """rows — строки кэша; teacher_idx — индексы в учительском наборе (тогда
    цели учителя и все 20 позиций действительны)."""
    import torch
    import k20_data as kd
    rows = np.asarray(rows)
    batch = kd.prompt_batch(bar, [np.asarray(D["IMG"][i]) for i in rows],
                            kd.norm_states(np.asarray(D["state8"][rows])),
                            D["task"][rows].tolist())
    po = torch.as_tensor(D["pos_offset"][rows], device=dev)
    st = torch.as_tensor(D["state11"][rows], device=dev)
    if teacher_idx is None:
        tgt = torch.as_tensor(D["act"][rows], device=dev)
        val = torch.as_tensor(D["valid"][rows], device=dev)
    else:
        tgt = torch.as_tensor(D["teacher_target"][teacher_idx], device=dev)
        val = torch.ones(tgt.shape[:2], device=dev)
    return batch, po, st, tgt, val


def selftest():
    m = tail_mask([0, 5, 18], [30, 20, 25])
    assert m[0].all() and m[1].sum() == 15 and m[2].sum() == 7
    rng = np.random.default_rng(0)
    raw = rng.uniform(-2, 2, (4, T, 7)).astype(np.float32)
    L, U = -np.ones(6) * 1.1, np.ones(6) * 1.5
    tgt = raw.copy()
    tgt[..., :6] = np.clip(raw[..., :6], L, U)
    t = dict(rows=np.array([1, 2, 3, 4]), raw=raw, target=tgt)
    assert check_teacher(t, np.arange(10), L, U) == []
    assert check_teacher(dict(t, rows=np.array([1, 1, 3, 4])), np.arange(10),
                         L, U)
    assert check_teacher(dict(t, target=raw), np.arange(10), L, U)
    assert check_teacher(dict(t, rows=np.array([1, 2, 3, 40])),
                         np.arange(10), L, U)
    bad = raw.copy()
    bad[0, 0, 0] = np.nan
    assert check_teacher(dict(t, raw=bad), np.arange(10), L, U)
    import k21_model as km21
    s = np.zeros((2, 8))
    s[:, 6:] = 0.02
    x = km21.state11(s)
    assert x.shape == (2, 11) and np.allclose(x[:, 3:9], [1, 0, 0, 0, 1, 0])
    print("самопроверка k21_data пройдена: маска хвоста, проверки учителя "
          "(повтор, clip, вне train, NaN), состояние 6D")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-21: подготовка")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--h18-cache", default="data/k15f/h18_cache")
    ap.add_argument("--k16-labels", default="reports/k16/labels.npz")
    ap.add_argument("--k18-config", default="data/k18/config.json")
    ap.add_argument("--device", default="cuda:0")
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
