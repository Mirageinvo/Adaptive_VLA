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
ЗАКРЕПЛЕНИЕ ВХОДОВ: действия, состояния и выбор строк копируются в
prep.npz (SHA в паспорте); кадры — манифест (размер, mtime, SHA файла,
SHA фиксированной выборки 512 строк — сверяется при каждой загрузке);
исходный кэш — SHA. Чекпойнт BAR — ревизия (снимок HF) и отпечатки файлов
(имена LFS-blob = их SHA256); модель и процессор грузятся из этого снимка.
УЧИТЕЛЬ (отдельный этап после real-check): K-20, если его паспорт завершён,
ревизия BAR, источник кадров/кэша и разметка K-16 те же и строки равны
выбору; иначе — сбор заново (k20_data.run_teacher) из закреплённого снимка.
ВЫХОД: data/k21/prep.npz, reports/k21/prepare.json, data/k21/teacher.npz,
reports/k21/teacher.json.
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


CKPT = "ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO"
TEACHER = "data/k21/teacher.npz"
TEACHER_REPORT = "reports/k21/teacher.json"
SAMPLE_ROWS, SAMPLE_SEED = 512, 20261019


def sha_stream(path, chunk=1 << 24, log=None):
    h = hashlib.sha1()
    n = os.path.getsize(path)
    done = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            done += len(b)
            if log and done % (chunk * 256) < chunk:
                log(f"  хеш {os.path.basename(path)}: {done / n:.0%}")
    return h.hexdigest()


def image_sample_sha(IMG, rows):
    h = hashlib.sha1()
    for r in rows:
        h.update(np.ascontiguousarray(IMG[int(r)]).tobytes())
    return h.hexdigest()


def image_manifest(path, used, full=True, log=print):
    st = os.stat(path)
    IMG = np.load(path, mmap_mode="r")
    rng = np.random.default_rng(SAMPLE_SEED)
    rows = np.sort(rng.choice(used, min(SAMPLE_ROWS, len(used)),
                              replace=False))
    return dict(path=os.path.abspath(path), size=st.st_size,
                mtime_ns=st.st_mtime_ns, shape=list(IMG.shape),
                dtype=str(IMG.dtype), sample_rows=rows.tolist(),
                sample_sha1=image_sample_sha(IMG, rows),
                sha1=sha_stream(path, log=log) if full else None)


def check_images(man, full=False):
    p = man["path"]
    if not os.path.exists(p):
        return ["кадры отсутствуют"]
    st = os.stat(p)
    pr = []
    if st.st_size != man["size"] or st.st_mtime_ns != man["mtime_ns"]:
        pr.append("кадры: размер или время изменения ≠ манифесту")
    IMG = np.load(p, mmap_mode="r")
    if list(IMG.shape) != man["shape"] or str(IMG.dtype) != man["dtype"]:
        pr.append("кадры: форма или тип ≠ манифесту")
    elif image_sample_sha(IMG, man["sample_rows"]) != man["sample_sha1"]:
        pr.append("кадры: выборка строк ≠ манифесту")
    if full and sha_stream(p) != man["sha1"]:
        pr.append("кадры: SHA файла ≠ манифесту")
    return pr


def snapshot_pin(revision=None):
    """Снимок HF чекпойнта: ревизия и отпечатки файлов (blob = SHA256)."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(CKPT, revision=revision, local_files_only=True)
    rev = os.path.basename(os.path.normpath(path))
    files = {}
    for root, _d, fs in os.walk(path):
        for f in sorted(fs):
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, path)
            real = os.path.realpath(fp)
            files[rel] = dict(blob=os.path.basename(real),
                              size=os.path.getsize(real))
    return dict(checkpoint=CKPT, revision=rev, path=path, files=files)


def check_pin(pin):
    cur = snapshot_pin(pin["revision"])
    if cur["revision"] != pin["revision"] or cur["files"] != pin["files"]:
        raise SystemExit("снимок BAR не совпадает с закреплённым")
    return cur["path"]


def load_bar(device, pin, root="third_party/actioncodec",
             cfg_path="config/eval/bar.yaml"):
    """Исходный полный BAR и процессор из закреплённого снимка."""
    import torch
    path = check_pin(pin)
    r = os.path.abspath(root)
    if r not in sys.path:
        sys.path.insert(0, r)
    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import VisionLanguageActionProcessor, get_cfg
    cfg = get_cfg(os.path.join(r, cfg_path))
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = path
    model = SmolVLABlockwiseAR.from_pretrained(**cfg.MODEL.vlm.kwargs).to(
        device, torch.float16).eval()
    if type(model) is not SmolVLABlockwiseAR:
        raise SystemExit(f"модель класса {type(model)}")
    for p in model.parameters():
        p.requires_grad_(False)
    proc = VisionLanguageActionProcessor.from_pretrained(
        path, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None))
    codec = codec.to(device).eval()
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size), device=device).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(device)
    return dict(model=model, proc=proc, codec=codec, E=E, cfg=cfg,
                revision=pin["revision"])


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


def teacher_k20(sel_rows, prep_rep, k20_prep="reports/k20/prepare.json",
                k20_teacher="data/k20/teacher.npz"):
    """Учитель K-20 при строгом совпадении происхождения или (None, причина)."""
    if not (os.path.exists(k20_prep) and os.path.exists(k20_teacher)):
        return None, "нет готового учителя K-20"
    r = json.load(open(k20_prep))
    why = []
    if not r.get("complete") or sha(k20_teacher) != r.get("teacher_sha1"):
        why.append("не завершён или файл изменён")
    bar = r.get("bar") or {}
    if bar.get("class_name") != "SmolVLABlockwiseAR" or \
            bar.get("checkpoint") != CKPT or \
            bar.get("revision") != prep_rep["bar_pin"]["revision"]:
        why.append("другая модель или ревизия BAR")
    src = r.get("source") or {}
    if os.path.abspath(src.get("cache", "")) != os.path.abspath(
            prep_rep["inputs"]["cache"]["path"]) or \
            os.path.abspath(src.get("images", "")) != \
            prep_rep["inputs"]["images"]["path"] or \
            src.get("k16_labels_sha1") != prep_rep["inputs"]["k16_labels_sha1"]:
        why.append("другой источник кадров, кэша или разметки")
    t = dict(np.load(k20_teacher))
    if not np.array_equal(t["rows"], sel_rows):
        why.append("строки ≠ текущему выбору")
    if why:
        return None, "учитель K-20 не принят: " + "; ".join(why)
    return t, dict(source="K-20", revision=bar["revision"],
                   k20_teacher_sha1=r["teacher_sha1"])


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
    """Индексы, маски, состояния, закрепление входов (без учителя)."""
    import k20_data as kd
    import k15h_protocol as hpc
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    rep = dict(kind="k21_prepare", device=device_passport())
    print(json.dumps(rep["device"], ensure_ascii=False), flush=True)
    pin = snapshot_pin()
    print(f"  снимок BAR: ревизия {pin['revision']}, файлов "
          f"{len(pin['files'])}", flush=True)
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
    act = np.asarray(ix["act"], np.float32)[:N]
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
    if not np.isfinite(act[used]).all():
        raise SystemExit("экспертные действия не конечны")
    gz = int((act[used][..., 6] == 0).sum())
    try:
        import k16_prepare as kp16
        pilot = kp16.pilot_tasks()
    except Exception:                                        # noqa: BLE001
        pilot = None
    vt = set(ix["task"][va].tolist())
    print("  манифест кадров (полный SHA читает весь файл)...", flush=True)
    img_man = image_manifest(ix["img"], used)
    np.savez(OUT, train_rows=tr, val_rows=va, episode=ix["epi"],
             step=ix["step"], task=ix["task"],
             pos_offset=np.asarray(d["pos_offset"])[:N], act=act,
             state8=st8, state11=s11, valid=valid, L=L, U=U,
             state_mean=mean, state_std=std, scale=scale,
             teacher_rows=ix["teacher"])
    rep.update(
        bar_pin=pin,
        inputs=dict(cache=dict(path=os.path.abspath(ix["src"]),
                               sha1=sha_stream(ix["src"])),
                    state=dict(path=os.path.abspath(ix["src"] + ".state.npy"),
                               sha1=sha_stream(ix["src"] + ".state.npy")),
                    images=img_man,
                    k16_labels_sha1=sha(a.k16_labels),
                    k18_config_sha1=sha(a.k18_config)),
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
        teacher_selection=dict(n=int(len(ix["teacher"])),
                               short=ix["teacher_short"]),
        volume=dict(updates=1000 + 8000, max_updates=1000 + 16000,
                    effective_batch=32),
        out=dict(path=os.path.abspath(OUT), sha1=sha(OUT)))
    json.dump(rep, open(REPORT, "w"), ensure_ascii=False, indent=1,
              default=float)
    print(json.dumps({k: rep[k] for k in ("rows", "episodes", "tasks",
                                          "val_sel_missing_tasks",
                                          "pilot_missing_in_val", "tail",
                                          "grip_zero_targets",
                                          "teacher_selection",
                                          "bounds_codec")},
                     ensure_ascii=False, indent=1, default=float))
    print(f"  паспорт: {REPORT}")
    return 0


def main_teacher(a):
    """Учитель: K-20 при строгом совпадении происхождения или сбор заново."""
    import k20_data as kd
    D = load_prepared(require_teacher=False)
    prep = D["prep_report"]
    t, src = teacher_k20(D["teacher_rows"], prep)
    if t is None:
        print(f"  {src} — сбор заново из закреплённого снимка", flush=True)
        bar = load_bar(a.device, prep["bar_pin"])
        ix = dict(d=dict(pos_offset=D["pos_offset"]), src=D["cache_path"],
                  img=prep["inputs"]["images"]["path"], task=D["task"],
                  teacher=D["teacher_rows"])
        raw, feas, codes, rate, spd = kd.run_teacher(bar, ix, D["L"], D["U"],
                                                     batch=a.batch)
        t = dict(rows=D["teacher_rows"], raw=raw, target=feas, codes=codes)
        src = dict(source="K-21", revision=bar["revision"],
                   out_of_bounds_rate=rate, seconds_per_obs=spd)
    prob = check_teacher(t, D["train_rows"], D["L"], D["U"])
    if prob:
        raise SystemExit("; ".join(prob))
    np.savez(TEACHER, rows=t["rows"], raw=t["raw"], target=t["target"])
    rep = dict(kind="k21_teacher", provenance=src,
               prep_sha1=prep["out"]["sha1"], n=int(len(t["rows"])),
               out=dict(path=os.path.abspath(TEACHER), sha1=sha(TEACHER)))
    json.dump(rep, open(TEACHER_REPORT, "w"), ensure_ascii=False, indent=1,
              default=float)
    print(json.dumps(rep, ensure_ascii=False, default=float))
    return 0


# --- загрузка и батчи ------------------------------------------------------------
def load_prepared(path=OUT, report=REPORT, require_teacher=True,
                  full_images=False):
    """Сверка prep.npz, исходного кэша/состояний (SHA при подготовке не
    перечитывается), манифеста кадров и, при необходимости, учителя."""
    if not os.path.exists(path) or not os.path.exists(report):
        raise SystemExit("prepare K-21 не выполнен")
    r = json.load(open(report))
    if r.get("kind") != "k21_prepare" or r["out"]["sha1"] != sha(path):
        raise SystemExit("prep.npz изменён после prepare")
    pr = check_images(r["inputs"]["images"], full=full_images)
    if pr:
        raise SystemExit("; ".join(pr))
    z = dict(np.load(path, allow_pickle=False))
    D = dict(z, prep_report=r, cache_path=r["inputs"]["cache"]["path"])
    D["IMG"] = np.load(r["inputs"]["images"]["path"], mmap_mode="r")
    if require_teacher:
        if not os.path.exists(TEACHER) or not os.path.exists(TEACHER_REPORT):
            raise SystemExit("учитель K-21 не подготовлен")
        tr = json.load(open(TEACHER_REPORT))
        if tr["out"]["sha1"] != sha(TEACHER) or \
                tr["prep_sha1"] != r["out"]["sha1"]:
            raise SystemExit("учитель изменён или от другой подготовки")
        t = dict(np.load(TEACHER))
        if not np.array_equal(t["rows"], z["teacher_rows"]):
            raise SystemExit("строки учителя ≠ выбору подготовки")
        D["teacher_target"] = t["target"]
    return D


def make_batch(D, rows, bar, dev, teacher_idx=None):
    """rows — строки кэша; teacher_idx — индексы учительского набора (тогда
    rows обязаны совпадать с его строками; цели учителя, все 20 позиций)."""
    import torch
    import k20_data as kd
    rows = np.asarray(rows)
    if teacher_idx is not None and not np.array_equal(
            D["teacher_rows"][np.asarray(teacher_idx)], rows):
        raise SystemExit("строки батча ≠ строкам учителя")
    batch = kd.prompt_batch(bar, [np.asarray(D["IMG"][i]) for i in rows],
                            kd.norm_states(np.asarray(D["state8"][rows])),
                            D["task"][rows].tolist())
    po = torch.as_tensor(D["pos_offset"][rows], device=dev)
    st = torch.as_tensor(D["state11"][rows], device=dev)
    if teacher_idx is None:
        tgt = torch.as_tensor(D["act"][rows], device=dev)
        val = torch.as_tensor(D["valid"][rows], device=dev)
    else:
        tgt = torch.as_tensor(D["teacher_target"][np.asarray(teacher_idx)],
                              device=dev)
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
    import tempfile
    import time as _t
    with tempfile.TemporaryDirectory() as td:
        f = td + "/img.npy"
        np.save(f, rng.integers(0, 255, (40, 3, 4, 4), dtype=np.uint8))
        man = image_manifest(f, np.arange(40), log=None)
        assert check_images(man) == [] and check_images(man, full=True) == []
        X = np.load(f)
        X[7, 0, 0, 0] ^= 1
        _t.sleep(0.01)
        np.save(f, X)
        assert check_images(man)
    print("самопроверка k21_data пройдена: маска хвоста, проверки учителя "
          "(повтор, clip, вне train, NaN), состояние 6D, манифест кадров")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-21: подготовка")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--h18-cache", default="data/k15f/h18_cache")
    ap.add_argument("--k16-labels", default="reports/k16/labels.npz")
    ap.add_argument("--k18-config", default="data/k18/config.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--teacher", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        if a.teacher:
            return main_teacher(a)
        return main_prepare(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
