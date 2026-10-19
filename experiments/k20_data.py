#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-20 (prepare): индексы, внутреннее разбиение train, пределы команд,
кэш учителя полного BAR, реконструкции исходного ActionCodec.

ИСТОЧНИКИ (без новых больших кэшей): кэш K-9a (нормированный экспертный
чанк 20 × 7 `action` — рука / max|Q| харнесса, знак схвата инвертирован,
обрезка [−1, 1]; коды исходного токенизатора `K_true`; позиционные
смещения; кадры 224 × 448 `.images.npy`, подготовленные как в среде);
строки train / val_sel кэша K-15f (= разбиение K-16, объединение A и B).
val_confirm не открывается. Ключ строки — (эпизод, шаг).

ВНУТРЕННЕЕ РАЗБИЕНИЕ TRAIN: в каждой задаче 10% эпизодов (минимум два при
≥ 10 эпизодах, иначе один) -> train_error, остальное -> train_fit.

ПРЕДЕЛЫ КОМАНД в координатах кодека: L = exec_low / scale_arm,
U = exec_high / scale_arm (exec — измеренные входные пределы контроллера
K-18, scale — max|Q| харнесса > 0).

УЧИТЕЛЬ: полный BAR (исходный checkpoint, без Joint12/K-14 и прочих
надстроек), generate как в харнессе (fullbar: три уровня, greedy), сборка
трёх уровней -> чанк 20 × 7 в координатах кодека (до scale_arm и до знака
схвата среды). 5120 наблюдений train_fit — по 128 на задачу, равномерно по
эпизодам и времени. Сырой выход и допустимый (рука в [L, U]) хранятся
отдельно; целью служит допустимый; частота выхода за пределы — в паспорт.

ИСХОДНЫЙ ActionCodec на val_sel: декод истинных кодов трёх уровней и
только первого уровня — для сравнения с новым кодеком на тех же строках.

ВЫХОД: data/k20/index.npz, data/k20/teacher.npz, data/k20/orig_codec_val.npz,
reports/k20/prepare.json.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT = "data/k20"
REPORT = "reports/k20/prepare.json"
T_CHUNK, H_EXEC, N_POS, N_LEVEL = 20, 8, 16, 3
TEACHER_PER_TASK = 128
TIME_BINS = 8
SEED = 20261019
CKPT = "ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO"


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def split_error(epi_task, seed=SEED, frac=0.10):
    """{эпизод: задача} -> (train_fit эпизоды, train_error эпизоды)."""
    rng = np.random.default_rng(seed)
    by = {}
    for e, t in epi_task.items():
        by.setdefault(t, []).append(int(e))
    fit, err = [], []
    for t in sorted(by):
        eps = sorted(by[t])
        n = len(eps)
        k = max(2, int(round(frac * n))) if n >= 10 else max(1, int(round(
            frac * n))) if n >= 2 else 0
        pick = set(rng.choice(eps, k, replace=False).tolist()) if k else set()
        err += sorted(pick)
        fit += [e for e in eps if e not in pick]
    return sorted(fit), sorted(err)


def teacher_rows(rows, epi, step, task, per_task=TEACHER_PER_TASK,
                 seed=SEED):
    """Равномерно по эпизодам и времени внутри задачи; возвращает индексы
    в rows (отсортированы) и задачи, где строк не хватило."""
    rng = np.random.default_rng(seed + 1)
    out, short = [], []
    for t in sorted(set(task[rows].tolist())):
        r = rows[task[rows] == t]
        eps = np.unique(epi[r])
        mx = {e: step[r][epi[r] == e].max() for e in eps}
        tb = np.array([min(TIME_BINS - 1, int(TIME_BINS * step[i]
                                              / (mx[epi[i]] + 1)))
                       for i in r])
        if len(r) < per_task:
            short.append(str(t))
        cnt_e = {e: 0 for e in eps}
        cnt_b = np.zeros(TIME_BINS)
        avail = np.ones(len(r), bool)
        noise = rng.random(len(r))
        for _ in range(min(per_task, len(r))):
            ce = np.array([cnt_e[epi[i]] for i in r])
            key = ce * 1e6 + cnt_b[tb] * 1e3 + noise
            key[~avail] = np.inf
            j = int(np.argmin(key))
            avail[j] = False
            cnt_e[epi[r[j]]] += 1
            cnt_b[tb[j]] += 1
            out.append(int(r[j]))
    return np.sort(np.asarray(out, np.int64)), short


def bounds_codec(exec_lo, exec_hi, scale):
    s = np.asarray(scale, np.float64)[:6]
    if not (s > 0).all():
        raise SystemExit("scale_arm не положителен")
    return np.asarray(exec_lo, np.float64) / s, \
        np.asarray(exec_hi, np.float64) / s


def make_index(a):
    import k15f_build_cache as kb
    meta = json.load(open(f"{a.cache}.meta.json"))
    src = meta["cache"]
    d = np.load(src, allow_pickle=True)
    N = int(meta["n_obs"])
    epi = np.asarray(d["episode"])[:N].astype(np.int64)
    stp = np.asarray(d["step"])[:N].astype(np.int64)
    task = np.asarray(d["task"])[:N].astype(str)
    c = kb.load_cache(a.h18_cache, verify_h18=False)
    L16 = dict(np.load(a.k16_labels, allow_pickle=False))
    rows = {p: np.asarray(c[p]["rows"], np.int64) for p in kb.PARTS}
    for p in kb.PARTS:
        if not np.array_equal(L16[f"{p}__rows"], c[p]["rows"]):
            raise SystemExit(f"строки {p} кэша K-15f ≠ разметке K-16")
    tr_eps = set(epi[rows["train"]].tolist())
    if tr_eps != set(L16["epA"].tolist()) | set(L16["epB"].tolist()):
        raise SystemExit("train ≠ A ∪ B K-16")
    va_eps = set(epi[rows["val_sel"]].tolist())
    if tr_eps & va_eps:
        raise SystemExit("train и val_sel пересекаются по эпизодам")
    fit_e, err_e = split_error({int(e): str(task[rows["train"]][
        epi[rows["train"]] == e][0]) for e in tr_eps})
    fit = rows["train"][np.isin(epi[rows["train"]], fit_e)]
    err = rows["train"][np.isin(epi[rows["train"]], err_e)]
    trow, short = teacher_rows(fit, epi, stp, task)
    img = src + ".images.npy"
    if not os.path.exists(img):
        raise SystemExit(f"нет кадров {img}")
    IMG = np.load(img, mmap_mode="r")
    if IMG.shape[1:] != (3, 224, 448) or IMG.dtype != np.uint8 or \
            len(IMG) < N:
        raise SystemExit(f"кадры формы {IMG.shape} {IMG.dtype}")
    act = np.asarray(d["action"])
    if act.shape[1:] != (T_CHUNK, 7):
        raise SystemExit(f"экспертный чанк формы {act.shape}")
    return dict(meta=meta, src=src, d=d, N=N, epi=epi, step=stp, task=task,
                rows=rows, fit=fit, err=err, fit_e=fit_e, err_e=err_e,
                teacher=trow, teacher_short=short, img=img, act=act)


def coverage(ix, r):
    t = ix["task"][r]
    return dict(rows=int(len(r)), episodes=int(len(set(ix["epi"][r]
                                                       .tolist()))),
                tasks=int(len(set(t.tolist()))))


def load_bar(device, root="third_party/actioncodec",
             cfg_path="config/eval/bar.yaml"):
    """Исходный полный BAR без надстроек + процессор и кодек."""
    import torch
    r = os.path.abspath(root)
    if r not in sys.path:
        sys.path.insert(0, r)
    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import VisionLanguageActionProcessor, get_cfg
    cfg = get_cfg(os.path.join(r, cfg_path))
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = CKPT
    model = SmolVLABlockwiseAR.from_pretrained(**cfg.MODEL.vlm.kwargs).to(
        device, torch.float16).eval()
    if type(model) is not SmolVLABlockwiseAR:
        raise SystemExit(f"модель класса {type(model)}, ожидался исходный BAR")
    for p in model.parameters():
        p.requires_grad_(False)
    proc = VisionLanguageActionProcessor.from_pretrained(
        CKPT, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None))
    codec = codec.to(device).eval()
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size), device=device).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(device)
    try:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(CKPT, local_files_only=True)
        rev = os.path.basename(os.path.normpath(snap))
    except Exception as e:                                   # noqa: BLE001
        rev = f"не определена: {e}"
    return dict(model=model, proc=proc, codec=codec, E=E, cfg=cfg,
                revision=rev)


def decode_codes(bar, K, levels=3):
    """K [B, 3, 16] -> чанк [B, T, 7] (сборка уровней как в харнессе)."""
    import torch
    E, codec = bar["E"], bar["codec"]
    dev = E.device
    K = torch.as_tensor(np.asarray(K)).long().to(dev)
    with torch.no_grad():
        z = sum(E[j][K[:, j, :]] for j in range(levels))
        x, _ = codec._decode(z, embodiment_ids=0)
    return x[..., :7].float().cpu().numpy()


def prompt_batch(bar, imgs, states_n, tasks):
    import torch
    import utils as U
    cfg, proc = bar["cfg"], bar["proc"]
    msgs = []
    for s, t in zip(states_n, tasks):
        m = U.prompt_template(
            s, None, t, mode=cfg.MODEL.vla_processor.kwargs.mode,
            action_vocab_size=cfg.MODEL.action_processor.vocab_size,
            action_token_len=cfg.MODEL.action_processor.token_len)
        m[1]["content"] = m[1]["content"][1:]
        msgs.append(m)
    texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
    b = proc(text=texts, images=[[im] for im in imgs], return_tensors="pt",
             padding=True, padding_side="left",
             action_processor_kwargs={"embodiment_ids": 0})
    dev = bar["E"].device
    return U.dict_apply(lambda x: x.to(dev, torch.float16), b)


def norm_states(states):
    import utils as U
    st = np.asarray(states, np.float64)
    if st.shape[1] == len(U.STATE_Q01) + 1:
        st = U.process_state(st)
    return (st - U.STATE_Q01) / (U.STATE_Q99 - U.STATE_Q01) * 2.0 - 1.0


def run_teacher(bar, ix, lo, hi, batch=8, log=print):
    """Полный BAR на выбранных строках -> сырой и допустимый чанк."""
    import torch
    d = ix["d"]
    st_all = np.load(ix["src"] + ".state.npy", mmap_mode="r")
    IMG = np.load(ix["img"], mmap_mode="r")
    offs = np.asarray(d["pos_offset"])
    rows = ix["teacher"]
    raw = np.zeros((len(rows), T_CHUNK, 7), np.float32)
    codes = np.zeros((len(rows), N_LEVEL * N_POS), np.int64)
    order = np.lexsort((rows, offs[rows]))
    t0 = time.time()
    done = 0
    for po in sorted(set(offs[rows].tolist())):
        sel = order[offs[rows][order] == po]
        for s in range(0, len(sel), batch):
            j = sel[s:s + batch]
            r = rows[j]
            b = prompt_batch(bar, [np.asarray(IMG[i]) for i in r],
                             norm_states(np.asarray(st_all[r])),
                             ix["task"][r].tolist())
            with torch.no_grad():
                tk = bar["model"].generate(**b, position_offset=int(po),
                                           do_sample=False,
                                           initial_position_shift=1)
            k = tk.cpu().numpy().reshape(len(r), N_LEVEL, N_POS)
            codes[j] = k.reshape(len(r), -1)
            raw[j] = decode_codes(bar, k)[:, :T_CHUNK]
            done += len(r)
            if done == batch or done % (batch * 64) < batch:
                el = time.time() - t0
                log(f"  учитель: {done}/{len(rows)}, {el / done:.3f} с на "
                    f"наблюдение, осталось ~{el / done * (len(rows) - done) / 60:.1f} мин",
                    flush=True)
    if not np.isfinite(raw).all():
        raise SystemExit("выход учителя не конечен")
    feas = raw.copy()
    feas[..., :6] = np.clip(raw[..., :6], lo, hi)
    out_rate = float((np.abs(feas - raw) > 0).any(axis=(1, 2)).mean())
    return raw, feas, codes, out_rate, (time.time() - t0) / max(done, 1)


def main_prepare(a):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    import k15h_protocol as hpc
    ix = make_index(a)
    k18 = json.load(open(a.k18_config))
    lim = k18.get("env_limits") or {}
    if not lim.get("input_min"):
        raise SystemExit("нет измеренных пределов контроллера (K-18)")
    scale = hpc.harness_scale()
    lo, hi = bounds_codec(lim["input_min"], lim["input_max"], scale)
    act = ix["act"]
    fit = ix["fit"]
    std = np.maximum(act[fit][..., :6].reshape(-1, 6).std(0), 1e-3)
    if ix["teacher_short"]:
        print(f"  ВНИМАНИЕ: задачи с < {TEACHER_PER_TASK} строк train_fit: "
              f"{ix['teacher_short']}")
    rep = dict(
        kind="k20_prepare", T_chunk=T_CHUNK, H_exec=H_EXEC,
        source=dict(cache=ix["src"], images=ix["img"],
                    h18_cache_rows=os.path.abspath(a.h18_cache),
                    k16_labels=os.path.abspath(a.k16_labels),
                    k16_labels_sha1=sha(a.k16_labels)),
        coverage=dict(train=coverage(ix, ix["rows"]["train"]),
                      train_fit=coverage(ix, fit),
                      train_error=coverage(ix, ix["err"]),
                      val_sel=coverage(ix, ix["rows"]["val_sel"]),
                      teacher=coverage(ix, ix["teacher"])),
        val_sel_missing_tasks=sorted(set(ix["task"][ix["rows"]["train"]]
                                         .tolist())
                                     - set(ix["task"][ix["rows"]["val_sel"]]
                                           .tolist())),
        teacher_short_tasks=ix["teacher_short"],
        bounds_codec=dict(L=lo.tolist(), U=hi.tolist(),
                          exec=dict(lo=lim["input_min"], hi=lim["input_max"]),
                          scale_arm=scale[:6].tolist()),
        expert_range=dict(min=act[fit][..., :6].reshape(-1, 6).min(0)
                          .tolist(), max=act[fit][..., :6].reshape(-1, 6)
                          .max(0).tolist()),
        norm_std_arm=std.tolist(),
        disjoint=dict(fit_error=not set(ix["fit_e"]) & set(ix["err_e"])))
    arrays = dict(train=ix["rows"]["train"], val_sel=ix["rows"]["val_sel"],
                  train_fit=fit, train_error=ix["err"],
                  teacher=ix["teacher"], episode=ix["epi"], step=ix["step"],
                  task=ix["task"], L=lo, U=hi, std_arm=std, scale=scale)
    try:
        import peft
        rep["peft"] = peft.__version__
    except ImportError:
        rep["peft"] = None
    import torch
    if torch.cuda.is_available():
        rep["gpus"] = [dict(name=torch.cuda.get_device_name(i),
                            memory_gb=round(torch.cuda.get_device_properties(i)
                                            .total_memory / 2 ** 30, 1),
                            bf16=bool(torch.cuda.is_bf16_supported())
                            if i == 0 else None)
                       for i in range(torch.cuda.device_count())]
    if a.no_gpu:
        rep["teacher"] = "пропущен (--no-gpu)"
    else:
        bar = load_bar(a.device)
        rep["bar"] = dict(class_name=type(bar["model"]).__name__,
                          checkpoint=CKPT, revision=bar["revision"],
                          overlays="нет (исходный SmolVLABlockwiseAR)")
        raw, feas, codes, rate, spd = run_teacher(bar, ix, lo, hi,
                                                  batch=a.batch)
        np.savez(os.path.join(OUT, "teacher.npz"), rows=ix["teacher"],
                 raw=raw, target=feas, codes=codes)
        rep["teacher"] = dict(n=int(len(raw)), out_of_bounds_rate=rate,
                              seconds_per_obs=spd,
                              raw_range=dict(min=raw[..., :6].reshape(-1, 6)
                                             .min(0).tolist(),
                                             max=raw[..., :6].reshape(-1, 6)
                                             .max(0).tolist()),
                              note="цель — допустимый выход; не обрезан до "
                                   "экспертного диапазона [−1, 1]")
        vrows = ix["rows"]["val_sel"]
        K = np.asarray(ix["d"]["K_true"])[vrows]
        r3 = np.concatenate([decode_codes(bar, K[s:s + 256])
                             for s in range(0, len(K), 256)])[:, :T_CHUNK]
        r1 = np.concatenate([decode_codes(bar, K[s:s + 256], 1)
                             for s in range(0, len(K), 256)])[:, :T_CHUNK]
        np.savez(os.path.join(OUT, "orig_codec_val.npz"), rows=vrows,
                 recon3=r3, recon1=r1)
        e3 = ((r3[..., :6] - act[vrows][..., :6]) / std) ** 2
        e1 = ((r1[..., :6] - act[vrows][..., :6]) / std) ** 2
        rep["orig_codec_val"] = dict(mse_norm_3lvl_20=float(e3.mean()),
                                     mse_norm_3lvl_8=float(e3[:, :8].mean()),
                                     mse_norm_1lvl_20=float(e1.mean()),
                                     mse_norm_1lvl_8=float(e1[:, :8].mean()))
    np.savez(os.path.join(OUT, "index.npz"), **arrays)
    rep["index_sha1"] = sha(os.path.join(OUT, "index.npz"))
    json.dump(rep, open(REPORT, "w"), ensure_ascii=False, indent=1,
              default=float)
    print(json.dumps({k: rep[k] for k in ("coverage", "val_sel_missing_tasks",
                                          "teacher_short_tasks",
                                          "bounds_codec", "teacher", "peft",
                                          "gpus", "bar")
                      if k in rep}, ensure_ascii=False, indent=1,
                     default=float))
    if "orig_codec_val" in rep:
        print("  исходный ActionCodec на val_sel:", rep["orig_codec_val"])
    print(f"  паспорт: {REPORT}")
    return 0


def selftest():
    rng = np.random.default_rng(0)
    et = {e: f"t{e % 4}" for e in range(120)}
    fit, err = split_error(et)
    assert not set(fit) & set(err) and set(fit) | set(err) == set(et)
    from collections import Counter
    c = Counter(et[e] for e in err)
    assert all(v == 3 for v in c.values()), c           # 30 эпизодов -> 3
    epi = np.repeat(np.arange(20), 50)
    step = np.tile(np.arange(50), 20)
    task = np.array([f"t{e % 2}" for e in epi])
    rows = np.arange(len(epi))
    tr, short = teacher_rows(rows, epi, step, task, per_task=40)
    assert len(tr) == 80 and not short
    ce = Counter(epi[tr].tolist())
    assert max(ce.values()) - min(ce.values()) <= 1
    cb = Counter((step[tr] * TIME_BINS // 50).tolist())
    assert max(cb.values()) - min(cb.values()) <= 2
    tr2, short2 = teacher_rows(rows[:30], epi, step, task, per_task=40)
    assert short2 == ["t0"]
    lo, hi = bounds_codec(-np.ones(6), np.ones(6), [.5, .5, .5, .2, .25, .4,
                                                    1])
    assert np.allclose(hi, [2, 2, 2, 5, 4, 2.5]) and np.allclose(lo, -hi)
    try:
        bounds_codec(-np.ones(6), np.ones(6), [0, 1, 1, 1, 1, 1, 1])
        raise AssertionError
    except SystemExit:
        pass
    _ = rng
    print("самопроверка k20_data пройдена: внутреннее разбиение train, "
          "выбор учительских строк (эпизоды и время), пределы в координатах "
          "кодека")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-20: подготовка данных")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--h18-cache", default="data/k15f/h18_cache")
    ap.add_argument("--k16-labels", default="reports/k16/labels.npz")
    ap.add_argument("--k18-config", default="data/k18/config.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--no-gpu", action="store_true")
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
