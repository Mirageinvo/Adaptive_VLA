"""K-6b: нужна ли BAR последовательность блоков, или уровни предсказуемы сразу.

ВОПРОС. BAR тратит три полных прохода по башне (147 из 195 мс, 75% вызова),
потому что RVQ последовательна: код уровня 2 квантует остаток уровня 1. Но это
модельное решение BAR, а не математическая необходимость — коды суть функция
действий, значит в принципе предсказуемы из наблюдения напрямую. Меряем, во
сколько обходится отказ от обусловливания:

    dH_g = CE( k_g | h, ПЕРЕМЕШАННЫЕ k_<g ) - CE( k_g | h, настоящие k_<g )

где h — скрытое состояние ПЕРВОГО блока, то есть всё, чем располагала бы
однопроходная архитектура.

ПОЧЕМУ ПЕРЕМЕШИВАНИЕ, А НЕ ОТБРАСЫВАНИЕ ВХОДА. Голова с дополнительным входом
имеет больше параметров, и «выигрыш» обусловленной версии оказался бы отчасти
разницей ёмкости. Поэтому обе головы одинаковы во всём, включая размер входа,
и различаются ТОЛЬКО тем, несёт ли k_<g информацию: в контроле он перемешан
между примерами. Тот же приём, что с перестановочным нулём в K-5c.

ГЛАВНАЯ МЕТРИКА — НЕ ТОКЕННАЯ. Декодер принимает СУММУ латентов (§1), поэтому
промах на соседний по эмбеддингу код почти ничего не стоит. Кросс-энтропия и
точность по токенам вспомогательны; решает ошибка ДЕКОДИРОВАННОГО действия.

ЧТО ЖДЁМ ПО УЖЕ ИЗМЕРЕННОМУ. FINDINGS §A0: правка кода грубого уровня меняет
тонкие уровни в среднем в 4.79 позиции из 16 (медиана 5, квартили 3/6). Значит
зависимость РАЗРЕЖЕНА — примерно в одиннадцати позициях из шестнадцати её нет
вовсе. Поэтому dH считается ПО ПОЗИЦИЯМ: среднее по всем шестнадцати размажет
эффект, сосредоточенный в пяти.

ПРАВИЛО ЧТЕНИЯ, записано до запуска:
  ошибка действия при параллельном предсказании близка к обусловленному
      (< +10%) -> последовательность блоков не нужна, хватает голов;
  ошибка заметно хуже, но dH сосредоточена в немногих позициях -> нужна
      дешёвая голова, обусловленная на мягком coarse-эмбеддинге, а не
      повторный проход башни;
  ошибка хуже и dH размазана по всем позициям -> зависимость существенна
      всюду, и оправдан новый токенизатор с параллельными книгами.

Запуск:
    python3 experiments/k6b_delta_h.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k6b_delta_h.py \\
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \\
        --n-obs 3000 --n-ep 300 --out data/k6b_delta_h.json
"""

import argparse
import io
import json
import os
import sys

import numpy as np

N_POS, N_LEVEL = 16, 3


def train_head(X, prev, y, n_codes, n_pos, seed=0, epochs=30, hid=512,
               val_frac=0.2, device="cpu", verbose=False):
    """Обучить одну голову p(k | X, prev) и вернуть КРОСС-ЭНТРОПИЮ ПО ПОЗИЦИЯМ.

    X:    (N, n_pos, d)      скрытые состояния первого блока
    prev: (N, n_pos, p)      коды предыдущих уровней (или их перемешка)
    y:    (N, n_pos)         целевые коды
    Голова общая для всех позиций плюс обучаемый эмбеддинг позиции — так на
    позицию приходится больше примеров, а разбор по позициям всё равно
    возможен, потому что кросс-энтропия считается поэлементно.
    """
    import torch
    import torch.nn as nn

    g = torch.Generator().manual_seed(seed)
    N = X.shape[0]
    perm = torch.randperm(N, generator=g)
    n_val = max(1, int(N * val_frac))
    idx_val, idx_tr = perm[:n_val], perm[n_val:]

    dev = torch.device(device)
    X = torch.as_tensor(X, dtype=torch.float32)
    prev = torch.as_tensor(prev, dtype=torch.long)
    y = torch.as_tensor(y, dtype=torch.long)

    d, n_prev = X.shape[-1], prev.shape[-1]
    emb_dim = 64

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.code_emb = nn.Embedding(n_codes, emb_dim) if n_prev else None
            self.pos_emb = nn.Embedding(n_pos, emb_dim)
            inp = d + emb_dim * n_prev + emb_dim
            self.net = nn.Sequential(nn.Linear(inp, hid), nn.GELU(),
                                     nn.Linear(hid, hid), nn.GELU(),
                                     nn.Linear(hid, n_codes))

        def forward(self, x, pr):
            b, p, _ = x.shape
            pos = self.pos_emb(torch.arange(p, device=x.device))
            pos = pos.unsqueeze(0).expand(b, -1, -1)
            parts = [x, pos]
            if n_prev:
                parts.append(self.code_emb(pr).flatten(-2))
            return self.net(torch.cat(parts, dim=-1))

    m = Head().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(reduction="none")
    Xtr, Ptr, Ytr = X[idx_tr].to(dev), prev[idx_tr].to(dev), y[idx_tr].to(dev)
    Xva, Pva, Yva = X[idx_val].to(dev), prev[idx_val].to(dev), y[idx_val].to(dev)
    bs = 256
    best = None
    for ep in range(epochs):
        m.train()
        order = torch.randperm(Xtr.shape[0], generator=g)
        for i in range(0, len(order), bs):
            j = order[i:i + bs]
            opt.zero_grad()
            lg = m(Xtr[j], Ptr[j])
            loss = lossf(lg.reshape(-1, n_codes), Ytr[j].reshape(-1)).mean()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            lg = m(Xva, Pva)
            ce = lossf(lg.reshape(-1, n_codes),
                       Yva.reshape(-1)).reshape(Yva.shape)
            ce_pos = ce.mean(dim=0).cpu().numpy()
            acc = (lg.argmax(-1) == Yva).float().mean(dim=0).cpu().numpy()
        # ОТБОР ПО ВАЛИДАЦИИ, а не последняя эпоха: иначе переобучение одной
        # из двух голов создаст ложную разницу.
        if best is None or ce_pos.mean() < best[0].mean():
            best = (ce_pos, acc, lg.argmax(-1).cpu().numpy(), idx_val.numpy())
        if verbose and ep % 10 == 0:
            print(f"    эпоха {ep}: CE {ce_pos.mean():.4f}", flush=True)
    return best


def selftest():
    import torch  # noqa: F401
    rng = np.random.default_rng(0)
    N, d, C = 1500, 32, 16

    h = rng.normal(size=(N, N_POS, d)).astype(np.float32)
    k1 = rng.integers(0, C, size=(N, N_POS))

    def shuffled(a, seed):
        r = np.random.default_rng(seed)
        return a[r.permutation(a.shape[0])]

    # 1. ЗАВИСИМОСТИ НЕТ: цель определяется только h. dH обязана быть ~0.
    W = rng.normal(size=(d, C))
    y_indep = (h @ W).argmax(-1)
    ce_c, *_ = train_head(h, k1[..., None], y_indep, C, N_POS, epochs=25)
    ce_p, *_ = train_head(h, shuffled(k1, 1)[..., None], y_indep, C, N_POS,
                          epochs=25)
    d0 = float(ce_p.mean() - ce_c.mean())
    assert abs(d0) < 0.15, \
        f"без зависимости dH обязана быть около нуля, получено {d0:+.3f}"

    # 2. ЗАВИСИМОСТЬ ЕСТЬ И СОСРЕДОТОЧЕНА: цель зависит от k1 только в пяти
    #    позициях из шестнадцати — ровно та разреженность, которую предсказывает
    #    FINDINGS §A0. Тест обязан её увидеть И ЛОКАЛИЗОВАТЬ.
    dep_pos = np.array([2, 5, 7, 11, 14])
    y_dep = y_indep.copy()
    y_dep[:, dep_pos] = k1[:, dep_pos]
    ce_c2, *_ = train_head(h, k1[..., None], y_dep, C, N_POS, epochs=25)
    ce_p2, *_ = train_head(h, shuffled(k1, 2)[..., None], y_dep, C, N_POS,
                           epochs=25)
    gap = ce_p2 - ce_c2
    assert gap[dep_pos].mean() > 1.0, \
        f"зависимость в пяти позициях обязана быть видна: {gap[dep_pos]}"
    other = np.setdiff1d(np.arange(N_POS), dep_pos)
    assert abs(gap[other].mean()) < 0.2, \
        f"в остальных позициях разрыва быть не должно: {gap[other].mean():+.3f}"
    assert gap[dep_pos].mean() > 5 * abs(gap[other].mean()), \
        "разрыв обязан ЛОКАЛИЗОВАТЬСЯ, иначе поза позиционного разбора не нужна"

    print("самопроверка пройдена:")
    print(f"  без зависимости dH = {d0:+.3f} (около нуля)")
    print(f"  при зависимости в 5 позициях из 16: dH там {gap[dep_pos].mean():.2f}, "
          f"в остальных {gap[other].mean():+.3f}")
    print("  контроль — ПЕРЕМЕШИВАНИЕ входа, а не его отбрасывание, поэтому")
    print("  ёмкость голов одинакова и разница означает только информацию")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n-obs", type=int, default=3000)
    ap.add_argument("--n-ep", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pos-offset", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from torchvision.transforms.v2 import CenterCrop, Compose, Resize

    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01,  # noqa: E402
                       STATE_Q99, VisionLanguageActionProcessor, dict_apply,
                       get_cfg, process_state, prompt_template, seed_everything)

    seed_everything(args.seed)
    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt
    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))
    tf = Compose([CenterCrop(int(224 * 0.875)), Resize(224)])

    model = SmolVLABlockwiseAR.from_pretrained(
        **cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")

    # ХУК НА ВХОД action_lm_head. bar.py:1247-1248 нормирует скрытое состояние
    # экспертной башни и подаёт его в этот Linear, то есть на его входе ровно
    # то, чем располагает предсказатель кодов. За вызов generate голова
    # срабатывает ТРИЖДЫ (по разу на блок); берём ПЕРВЫЙ — только он доступен
    # однопроходной архитектуре.
    grabbed = []
    model.action_lm_head.register_forward_hook(
        lambda m, i, o: grabbed.append(i[0].detach().float().cpu()))

    # --- данные ---------------------------------------------------------------
    rid, rev = "physical-intelligence/libero", "v2.0"
    tasks_map = {}
    for line in open(hf_hub_download(rid, "meta/tasks.jsonl", repo_type="dataset",
                                     revision=rev)):
        r = json.loads(line)
        tasks_map[r["task_index"]] = r["task"]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(1693)
    per_ep = max(1, args.n_obs // max(args.n_ep, 1))

    def png(cell):
        return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))

    im1, im2, st, act, tsk = [], [], [], [], []
    for e in order:
        if len(tsk) >= args.n_obs:
            break
        f = hf_hub_download(rid, f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet",
                            repo_type="dataset", revision=rev)
        t = pq.read_table(f)
        n = t.num_rows
        if n < 20 + 1:
            continue
        A_ = np.asarray(t.column("actions").to_pylist(), np.float32)
        S_ = np.asarray(t.column("state").to_pylist(), np.float32)
        ti = t.column("task_index").to_pylist()
        c1, c2 = t.column("image").to_pylist(), t.column("wrist_image").to_pylist()
        for s0 in rng.choice(n - 20 + 1, size=min(per_ep, n - 20 + 1),
                             replace=False):
            im1.append(png(c1[int(s0)])); im2.append(png(c2[int(s0)]))
            st.append(S_[int(s0)]); act.append(A_[int(s0):int(s0) + 20])
            tsk.append(tasks_map[ti[int(s0)]])
        if len(tsk) % 500 < per_ep:
            print(f"  наблюдений {len(tsk)}", flush=True)
    N = len(tsk)
    print(f"собрано {N} наблюдений")

    # --- истинные коды из ДЕЙСТВИЙ -------------------------------------------
    a_codec = np.asarray(act, np.float64).copy()
    a_codec[..., :-1] /= max_act_q[..., :-1]
    a_codec[..., -1] *= -1
    toks = np.asarray(proc.action_processor.encode(a_codec), np.int64)
    n_codes = int(cfg.MODEL.action_processor.vocab_size)
    assert toks.shape[1] == N_POS * N_LEVEL, \
        f"ожидалось {N_POS * N_LEVEL} токенов, получено {toks.shape[1]}"
    # РАСКЛАДКА ПОУРОВНЕВАЯ, не перемежение по времени: BAR берёт блоками по
    # block_size подряд (bar.py:1500-1503), значит первые 16 — уровень 0.
    K = toks.reshape(N, N_LEVEL, N_POS)

    # --- скрытые состояния первого блока -------------------------------------
    H = []
    st_n = ((process_state(np.asarray(st)) - STATE_Q01)
            / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0)
    for i0 in range(0, N, args.batch):
        sl = slice(i0, min(i0 + args.batch, N))
        b = sl.stop - sl.start
        i1 = tf(torch.tensor(np.stack(im1[sl])[:, :, :, ::-1].copy()).permute(0, 3, 1, 2))
        i2 = tf(torch.tensor(np.stack(im2[sl])[:, :, :, ::-1].copy()).permute(0, 3, 1, 2))
        image = torch.cat([i1, i2], dim=-1)
        msgs = []
        for j in range(b):
            m = prompt_template(
                st_n[sl][j], None, tsk[sl.start + j],
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=n_codes,
                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        batch = proc(text=texts, images=[[image[j].numpy()] for j in range(b)],
                     return_tensors="pt", padding=True, padding_side="left",
                     action_processor_kwargs={"embodiment_ids": 0})
        batch = dict_apply(lambda x: x.to(dev, dtype), batch)
        grabbed.clear()
        with torch.no_grad():
            model.generate(**batch, position_offset=args.pos_offset,
                           do_sample=False, initial_position_shift=1)
        assert len(grabbed) == N_LEVEL, \
            f"голова сработала {len(grabbed)} раз, ждали {N_LEVEL} (по блоку)"
        # ФОРМУ ПРОВЕРЯЕМ, А НЕ ПРЕДПОЛАГАЕМ. На первом блоке истории нет,
        # запросов ровно block_size, поэтому вход головы обязан иметь ровно
        # N_POS позиций. Если их больше, значит голова видит и историю, и
        # молчаливый срез [-N_POS:] взял бы не то.
        g0 = grabbed[0]
        assert g0.shape[1] == N_POS, (
            f"на первом блоке ждали {N_POS} позиций на входе action_lm_head, "
            f"получено {g0.shape[1]} — разберитесь, где в этом тензоре запросы "
            f"блока, прежде чем срезать")
        H.append(g0.numpy())                            # ПЕРВЫЙ блок
        if i0 % (args.batch * 50) == 0:
            print(f"  скрытых состояний {i0 + b}/{N}", flush=True)
    H = np.concatenate(H).astype(np.float32)
    print(f"скрытые состояния: {H.shape}")

    # --- dH по уровням --------------------------------------------------------
    def shuf(a, s):
        return a[np.random.default_rng(s).permutation(a.shape[0])]

    res = {}
    for lvl in (1, 2):
        prev = np.transpose(K[:, :lvl, :], (0, 2, 1))      # (N, POS, lvl)
        y = K[:, lvl, :]
        ce_c, acc_c, pred_c, idx = train_head(H, prev, y, n_codes, N_POS,
                                              epochs=args.epochs, device=args.device)
        ce_p, acc_p, pred_p, _ = train_head(H, shuf(prev, 100 + lvl), y, n_codes,
                                            N_POS, epochs=args.epochs,
                                            device=args.device)
        gap = ce_p - ce_c
        res[f"level{lvl}"] = dict(
            ce_conditional=ce_c.tolist(), ce_parallel=ce_p.tolist(),
            gap=gap.tolist(), acc_conditional=acc_c.tolist(),
            acc_parallel=acc_p.tolist())
        print(f"\n=== уровень {lvl}: dH по позициям")
        print("  " + " ".join(f"{g:5.2f}" for g in gap))
        srt = np.sort(gap)[::-1]
        print(f"  среднее {gap.mean():.3f}; пять худших позиций дают "
              f"{srt[:5].sum() / max(gap.sum(), 1e-9):.0%} всего разрыва")
        print(f"  точность: обусловленно {acc_c.mean():.1%}, "
              f"параллельно {acc_p.mean():.1%}")

        # --- ОШИБКА ДЕЙСТВИЯ, главная метрика --------------------------------
        def decode(codes):
            fl = codes.reshape(codes.shape[0], -1).tolist()
            d = proc.action_processor.decode(fl)
            return np.asarray(d if isinstance(d, np.ndarray) else d[0], np.float64)

        Kv = K[idx].copy()
        ref = decode(Kv)
        for name, pred in (("обусловленно", pred_c), ("параллельно", pred_p)):
            Kx = Kv.copy()
            Kx[:, lvl, :] = pred
            err = np.linalg.norm(decode(Kx)[..., :6] - ref[..., :6], axis=-1).mean()
            res[f"level{lvl}"][f"action_err_{name}"] = float(err)
            print(f"  ошибка действия при подмене уровня {lvl}, {name}: {err:.4f}")

    print("\n  ЧИТАТЬ ТАК, правило записано до запуска.")
    print("  Ошибка действия параллельно ≈ обусловленно (< +10%) — блоки не")
    print("  нужны, хватает голов на общем скрытом состоянии.")
    print("  Ошибка хуже, но dH сосредоточена в немногих позициях — нужна")
    print("  дешёвая голова на мягком coarse-эмбеддинге, а не проход башни.")
    print("  Ошибка хуже и dH размазана — оправдан новый токенизатор.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        res["meta"] = dict(ckpt=args.ckpt, n_obs=N, n_codes=n_codes,
                           pos_offset=args.pos_offset, epochs=args.epochs,
                           hidden_dim=int(H.shape[-1]))
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
