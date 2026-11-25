#!/usr/bin/env python3
"""K-40 B: PCA64 визуальных признаков и нормировка state11 — только K-21 train.

Строки: N_ROWS строк K-21 train_rows, порядок по хешу (детерминированно).
Кадр строки — уже склеенное изображение двух камер (как в k29_policy
Inputs после tf); форма сверяется с путём симулятора. Вход процессора T32
(источник T32_s0; визуальная часть одинакова у сидов), get_image_features
(энкодер + коннектор), токены каждого наблюдения в порядке подачи → 4
последовательные группы → средние (как k40b_collect.group_vis) → 4·D.
PCA: центрирование, SVD, 64 компоненты; масштаб — std проекций.
state11: среднее и std по всем train_rows K-21 (state11 из prep).
Выход: data/k40b/pca.npz + reports/k40b/pca.json (SHA, доля дисперсии).
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

N_ROWS = 4096
N_COMP = 64
BATCH = 16
OUT = Path("data/k40b/pca.npz")
REP = Path("reports/k40b/pca.json")


def rank(i):
    return hashlib.sha256(f"k40b-pca-v1/{int(i)}".encode()).hexdigest()


def fit_pca(X, k=N_COMP):
    X = np.asarray(X, np.float64)
    mu = X.mean(0)
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    comps = Vt[:k]
    proj = (X - mu) @ comps.T
    sd = proj.std(0)
    if (sd <= 0).any():
        raise SystemExit("PCA: нулевая дисперсия компоненты")
    ev = (s ** 2) / (s ** 2).sum()
    return dict(mean=mu.astype(np.float32), comps=comps.astype(np.float32),
                proj_std=sd.astype(np.float32),
                explained=ev[:k].astype(np.float32))


def k21_batch(inp, imgs, states8, descs):
    """Вход процессора как Inputs.__call__, но с готовыми кадрами K-21."""
    torch, U, cfg = inp.torch, inp.U, inp.cfg
    st = (np.asarray(states8, np.float64) - U.STATE_Q01) / \
        (U.STATE_Q99 - U.STATE_Q01) * 2.0 - 1.0
    msgs = []
    for s, d in zip(st, descs):
        m = U.prompt_template(
            s, None, d, mode=cfg.MODEL.vla_processor.kwargs.mode,
            action_vocab_size=cfg.MODEL.action_processor.vocab_size,
            action_token_len=cfg.MODEL.action_processor.token_len)
        m[1]["content"] = m[1]["content"][1:]
        msgs.append(m)
    texts = inp.proc.apply_chat_template(msgs, add_generation_prompt=True)
    b = inp.proc(text=texts, images=[[np.asarray(im)] for im in imgs],
                 return_tensors="pt", padding=True, padding_side="left",
                 action_processor_kwargs={"embodiment_ids": 0})
    return U.dict_apply(lambda x: x.to(inp.dev, torch.float16), b)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import torch
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    import k21_data as k21
    import k40b_collect as kc
    from k38_grip_run import threads_problems
    if pr := threads_problems(torch):
        raise SystemExit(f"потоки: {pr}")
    if OUT.exists() or REP.exists():
        raise SystemExit(f"{OUT} уже есть — PCA закреплён, не пересчитываю")
    k36 = pc.load_prepare()
    if not k36["k32"]["seeds"]["0"]["source"].get("weights_hash"):
        raise SystemExit("нет отпечатка весов источника T32_s0")
    D = k21.load_prepared(require_teacher=False)
    rows = sorted(np.asarray(D["train_rows"]).tolist(), key=rank)[:N_ROWS]
    s11 = np.asarray(D["state11"])[np.asarray(D["train_rows"])]
    src = kh.load_source(torch, 0, a.device)
    inp, bar = src["io"].inp, src["mod"].bar
    # форма кадра K-21 = форме склеенного кадра симулятора
    z = np.zeros((1, 224, 224, 3), np.uint8)
    t1 = inp.tf(torch.tensor(z[:, :, ::-1].copy()).permute(0, 3, 1, 2))
    sim_shape = tuple(torch.cat([t1, t1], dim=-1)[0].shape)
    if tuple(D["IMG"].shape[1:]) != sim_shape:
        raise SystemExit(f"кадр K-21 {D['IMG'].shape[1:]} ≠ симулятору "
                         f"{sim_shape}")
    tid = getattr(bar, "image_token_id", None)
    if tid is None:
        tid = bar.config.vlm_config.image_token_id
    feats, ntok = [], set()
    for i in range(0, len(rows), a.batch):
        r = rows[i:i + a.batch]
        b = k21_batch(inp, [D["IMG"][j] for j in r],
                      np.asarray(D["state8"])[r],
                      [str(x) for x in np.asarray(D["task"])[r]])
        with torch.no_grad(), torch.autocast(inp.dev.type,
                                             dtype=torch.float16):
            img = bar.vlm.get_image_features(b["pixel_values"],
                                             b.get("pixel_attention_mask"))
        img = img.detach().float().cpu().numpy()
        n = len(r)
        img = img.reshape(n, -1, img.shape[-1])
        pos = (b["input_ids"] == tid).sum(1).cpu().numpy()
        if not (pos == img.shape[1]).all():
            raise SystemExit(f"токенов на наблюдение {img.shape[1]} ≠ "
                             f"позиций {pos.tolist()}")
        ntok.add(int(img.shape[1]))
        feats += [kc.group_vis(x).ravel() for x in img]
    X = np.stack(feats)
    if not np.isfinite(X).all():
        raise SystemExit("визуальные признаки K-21 не конечны")
    p = fit_pca(X)
    mu11, sd11 = s11.mean(0), s11.std(0)
    sd11 = np.where(sd11 > 1e-8, sd11, 1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **p, s11_mean=mu11.astype(np.float32),
             s11_std=sd11.astype(np.float32),
             rows=np.asarray(rows, np.int64))
    rep = dict(kind="k40b_pca", n_rows=len(rows), vis_dim=int(X.shape[1] //
                                                             4),
               tokens_per_obs=sorted(ntok), n_comp=N_COMP,
               explained_total=float(p["explained"].sum()),
               k21_prep_sha1=D["prep_report"]["out"]["sha1"],
               source_seed=0, k36_prepare_sha1=k36["sha1"],
               source_weights=k36["k32"]["seeds"]["0"]["source"][
                   "weights_hash"],
               shapes={k: list(v.shape) for k, v in p.items()},
               out_sha256=kcf.sha256(OUT))
    pc.save_json(str(REP), rep)
    print(f"  PCA: строк {len(rows)}, D {rep['vis_dim']}, токенов на "
          f"наблюдение {sorted(ntok)}; 64 компоненты объясняют "
          f"{100 * rep['explained_total']:.1f}% дисперсии; SHA "
          f"{rep['out_sha256'][:16]}")
    return 0


def selftest():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(500, 8)) @ rng.normal(size=(8, 200))
    p = fit_pca(A, k=8)
    rec = ((A - p["mean"]) @ p["comps"].T) @ p["comps"] + p["mean"]
    assert np.allclose(rec, A, atol=1e-3)
    assert abs(float(p["explained"].sum()) - 1.0) < 1e-5
    assert len({rank(i) for i in range(100)}) == 100
    print("самопроверка k40b_pca пройдена: PCA восстанавливает ранг-8 "
          "данные, доля дисперсии, детерминированный отбор строк")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
