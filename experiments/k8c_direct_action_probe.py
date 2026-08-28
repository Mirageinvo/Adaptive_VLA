"""K-8c: есть ли на ранней глубине информация о ДЕЙСТВИИ, а не о коде.

ВОПРОС. K-8b показал, что четыре конфигурации дешёвой адаптации сходятся к
q0 = 25.4–26.5% против 87.3% у полной глубины, и Fast застревает на 0.1386 при
цели 0.0500. Но обучение целилось в КЛАССИФИКАЦИЮ одного из 2048 кодов
ActionCodec. Отсюда две несовместимые причины, которые те прогоны не различают:

  A. на слое 12 нет достаточной информации о движении;
  B. информация есть, но выбранная цель — угадать произвольный индекс из 2048 —
     плохо к ней приспособлена.

Здесь цель другая: предсказывать действие НАПРЯМУЮ, минуя коды.

ЧЕГО ЭТОТ ЗОНД НЕ ОБОСНОВЫВАЕТ. Его не следует мотивировать тем, что
непрерывная параметризация «легче» дискретной: в K-6e пять параметризаций
(free, soft, tied, cont, direct) уложились в разброс 0.0000–0.0006, то есть
никакой разницы там не было. Основание другое — просто разделить A и B, которые
сейчас неразличимы.

ВСЁ НА ОДНОМ РАЗБИЕНИИ ИЗ ОДНОГО ФАЙЛА. Опоры BAR считаются здесь же из
сохранённого K_bar, а не переносятся из другого прогона с другой выборкой:
смешанное сравнение публиковать нельзя. Модель и изображения заново не нужны.

ОБЯЗАТЕЛЬНЫЕ СТРОКИ ТАБЛИЦЫ:
  BAR coarse-only   практическая цель — то, что уже есть даром на 24 слоях;
  BAR полная        нижняя практическая граница;
  потолок fast      D(E0[q0*]), лучшее из одного уровня;
  случайный выход   пол;
  зонд на h24       контроль инструмента: на полной глубине обязан быть хорош,
                    иначе плох зонд, а не глубина.

ПРАВИЛО ЧТЕНИЯ, записано до запуска:
  зонд на h12 подходит к 0.05 -> дело было в дискретной цели, а не в глубине,
      и осмысленно проектировать токенизатор под ранние выходы;
  зонд на h12 остаётся около 0.13 -> на слое 12 действительно нет достаточной
      информации о действии, и дешёвая линия закрывается надёжнее прежнего;
  train сильно лучше val -> не хватает данных, а не глубины, и вывод об
      отсутствии информации делать нельзя.

Запуск:
    python3 experiments/k8c_direct_action_probe.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k8c_direct_action_probe.py \\
        --feats data/k7b_depth_4k_n.npz --ckpt <ckpt> \\
        --depths 12,24 --heads direct,latent --out data/k8c.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np

N_POS, N_LEVEL, T_CHUNK, H_EXEC = 16, 3, 20, 8
Z_DIM = 512


def split_by_episode(epi, task, seed=0, frac=(0.8, 0.1)):
    """По эпизодам, стратифицировано по задачам — как в K-7c и K-8b."""
    rng = np.random.default_rng(seed)
    masks = [np.zeros(len(epi), bool) for _ in range(3)]
    for t in np.unique(task):
        g = np.where(task == t)[0]
        ep = rng.permutation(np.unique(epi[g]))
        n = len(ep)
        if n < 3:
            raise SystemExit(f"задача «{t}»: {n} эпизодов, не делится на три")
        n_va = max(1, int(round(n * frac[1])))
        n_te = max(1, int(round(n * (1.0 - frac[0] - frac[1]))))
        if n - n_va - n_te < 1:
            n_va, n_te = 1, 1
        parts = (ep[:n - n_va - n_te], ep[n - n_va - n_te:n - n_te], ep[n - n_te:])
        for m, p in zip(masks, parts):
            m[g] = np.isin(epi[g], p)
    return masks


def selftest():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        raise SystemExit("нет torch: самопроверки k8c проверяют потерю и "
                         "разбиение и без него бессмысленны. Запускать на кластере.")

    epi = np.repeat(np.arange(60), 4)
    tsk = np.array([f"t{e % 6}" for e in epi])
    tr, va, te = split_by_episode(epi, tsk, seed=0)
    assert (tr | va | te).all() and not (tr & va).any() and not (va & te).any()
    for m in (tr, va, te):
        assert set(tsk[m]) == set(tsk), "задачи потеряны"

    # Потеря: исполняемая часть весит больше хвоста, знак схвата отдельно.
    a = torch.zeros(4, T_CHUNK, 7)
    b_t = a.clone(); b_t[:, H_EXEC:, :6] += 0.5
    b_h = a.clone(); b_h[:, :H_EXEC, :6] += 0.5
    assert head_loss(b_h, a) > head_loss(b_t, a)
    tgt = a.clone(); tgt[:, :, 6] = 0.5
    flip = a.clone(); flip[:, :, 6] = -0.5
    same = a.clone(); same[:, :, 6] = 0.5
    assert head_loss(flip, tgt) > head_loss(same, tgt)

    # ГЛАВНОЕ СВОЙСТВО ЗОНДА: он обязан уметь выразить тождество, иначе его
    # верхняя строка (на полной глубине) ничего не будет означать. Проверяем на
    # линейной голове с нулевым скрытым слоем: она обучаема до нуля потерь.
    torch.manual_seed(0)
    x = torch.randn(64, N_POS, 8)
    w = torch.randn(N_POS * 8, T_CHUNK * 7) * 0.1
    y = (x.reshape(64, -1) @ w).reshape(64, T_CHUNK, 7)
    lin = torch.nn.Linear(N_POS * 8, T_CHUNK * 7)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    for _ in range(400):
        p = lin(x.reshape(64, -1)).reshape(64, T_CHUNK, 7)
        loss = F.mse_loss(p, y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < 1e-3, (
        f"линейная голова не выучила линейную зависимость ({float(loss):.2e}) — "
        f"инструмент негоден, и его верхняя строка не будет опорой")

    print("самопроверка k8c пройдена: разбиение по эпизодам со стратификацией, "
          "исполняемая часть чанка весит больше хвоста, знак схвата отдельно, "
          "линейная голова выучивает линейную зависимость до 1e-3")


def head_loss(a_hat, a_star, mu=1.0, eta=0.25, grip_scale=4.0):
    """Та же форма, что в K-8b, чтобы числа были сопоставимы."""
    import torch.nn.functional as F
    h = H_EXEC
    pose8 = F.mse_loss(a_hat[:, :h, :6], a_star[:, :h, :6])
    grip8 = F.mse_loss(a_hat[:, :h, 6], a_star[:, :h, 6])
    sign = F.binary_cross_entropy_with_logits(
        a_hat[:, :h, 6] * grip_scale, (a_star[:, :h, 6] > 0).to(a_hat.dtype))
    full = F.mse_loss(a_hat[..., :6], a_star[..., :6])
    return pose8 + grip8 / 6.0 + mu * sign + eta * full


def build_head(kind, d_in, hidden, layers, n_heads, codec_decode):
    """direct: h -> 20x7 напрямую. latent: h -> 16x512 -> замороженный декодер
    кодека. Второй вариант проверяет, доступна ли ранней глубине та же латента,
    из которой декодер и так строит действие."""
    import torch
    import torch.nn as nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(d_in)
            self.inp = nn.Linear(d_in, hidden)
            if n_heads > 0:
                enc = nn.TransformerEncoderLayer(
                    hidden, n_heads, hidden * 2, batch_first=True,
                    norm_first=True, dropout=0.0)
                self.trunk = nn.TransformerEncoder(enc, layers)
            else:
                mods = []
                for _ in range(layers):
                    mods += [nn.Linear(hidden, hidden), nn.GELU()]
                self.trunk = nn.Sequential(*mods)
            self.kind = kind
            if kind == "direct":
                self.out = nn.Linear(hidden * N_POS, T_CHUNK * 7)
            else:
                self.out = nn.Linear(hidden, Z_DIM)

        def forward(self, h):
            z = self.trunk(self.inp(self.norm(h)))
            if self.kind == "direct":
                return self.out(z.reshape(z.shape[0], -1)).reshape(
                    -1, T_CHUNK, 7)
            return codec_decode(self.out(z))

    return Head()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--feats", default="data/k7b_depth_4k_n.npz")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--depths", default="12,24")
    ap.add_argument("--heads", default="direct,latent")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=8,
                    help="0 = MLP вместо трансформера")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--normed", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/k8c_direct.json")
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    sys.path.insert(0, os.path.abspath(args.root))
    import torch
    import torch.nn as nn

    import actioncodec  # noqa: F401
    from utils import VisionLanguageActionProcessor  # noqa: E402

    dev = torch.device(args.device)
    z = np.load(args.feats, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    K_true, K_bar = z["K_true"], z["K_bar"]
    epi, tsk = z["episode"], z["task"]
    N = len(epi)
    have_n = f"hn_after_{meta['depths'][0]}" in z.files
    use_n = have_n if args.normed == "auto" else (args.normed == "yes")
    print(f"{N} наблюдений, {len(np.unique(epi))} эпизодов, "
          f"состояния {'после нормы' if use_n else 'сырые'}")

    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None)).to(dev).eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    def dec(zq):
        x, _ = codec._decode(zq.float(), embodiment_ids=0)
        return x[..., :7].float()

    def dec_codes(K, n_lv, idxs):
        out = []
        Kt = torch.as_tensor(K[idxs]).long().to(dev)
        for i0 in range(0, len(idxs), 256):
            k = Kt[i0:i0 + 256]
            with torch.no_grad():
                out.append(dec(sum(E[j][k[:, j, :]] for j in range(n_lv))).cpu())
        return torch.cat(out)

    tr, va, te = split_by_episode(epi, tsk, seed=args.seed)
    ite = np.where(te)[0]
    print(f"train {tr.sum()}, val {va.sum()}, test {te.sum()}; "
          f"задач {len(np.unique(tsk))}")

    A_star = dec_codes(K_true, N_LEVEL, np.arange(N))       # цель

    def m_pose8(a_hat, idxs):
        d = a_hat - A_star[idxs]
        return float(torch.sqrt((d[:, :H_EXEC, :6] ** 2).mean()))

    def m_flip(a_hat, idxs):
        return float((torch.sign(a_hat[:, :H_EXEC, 6])
                      != torch.sign(A_star[idxs][:, :H_EXEC, 6])).float().mean())

    # --- опоры, всё на ОДНОМ разбиении --------------------------------------
    ref = {}
    for nm, K, n_lv in (("BAR coarse-only", K_bar, 1), ("BAR полная", K_bar, 3),
                        ("потолок fast", K_true, 1)):
        a = dec_codes(K, n_lv, ite)
        ref[nm] = dict(pose8=m_pose8(a, ite), flip=m_flip(a, ite))
    rnd = np.random.default_rng(0).integers(0, int(codec.vocab_size),
                                            size=K_true.shape)
    a = dec_codes(rnd, 1, ite)
    ref["случайный код"] = dict(pose8=m_pose8(a, ite), flip=m_flip(a, ite))
    print("\nопоры на тестовой части:")
    for nm, r in ref.items():
        print(f"    {nm:<18} поза8 {r['pose8']:.4f}  знак {r['flip']:.2%}")

    # --- обучение зонда ------------------------------------------------------
    def run(depth, kind, seed):
        torch.manual_seed(seed)
        key = f"{'hn' if use_n else 'h'}_after_{depth}"
        if key not in z.files:
            raise SystemExit(f"в файле нет {key}; доступно: {list(z.files)[:8]}")
        X = torch.as_tensor(z[key].astype(np.float32))
        head = build_head(kind, X.shape[-1], args.hidden, args.layers,
                          args.n_heads, dec).to(dev)
        ps = [p for p in head.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.01)
        itr = np.where(tr)[0]
        spe = math.ceil(len(itr) / args.batch)
        total = args.epochs * spe
        warm = max(1, int(total * args.warmup_frac))
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, s / warm)
            * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
        rg = np.random.default_rng(seed)

        def predict(idxs):
            out = []
            head.eval()
            with torch.no_grad():
                for i0 in range(0, len(idxs), 256):
                    out.append(head(X[idxs[i0:i0 + 256]].to(dev)).cpu())
            return torch.cat(out)

        best, best_state = None, None
        iva = np.where(va)[0]
        for ep in range(args.epochs):
            head.train()
            perm = rg.permutation(itr)          # НА КАЖДУЮ эпоху целиком
            for i0 in range(0, len(perm), args.batch):
                sel = perm[i0:i0 + args.batch]
                a_hat = head(X[sel].to(dev))
                loss = head_loss(a_hat, A_star[sel].to(dev))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ps, 1.0)
                opt.step(); sch.step()
            v = m_pose8(predict(iva), iva)
            if best is None or v < best:
                best = v
                best_state = {k: t.detach().clone()
                              for k, t in head.state_dict().items()}
        head.load_state_dict(best_state)
        # ОБЕ ВЫБОРКИ: без train-метрики нельзя отличить «информации нет» от
        # «данных мало». Подвыборка train размером с тест.
        sub = rg.choice(itr, size=min(len(ite), len(itr)), replace=False)
        return (m_pose8(predict(ite), ite), m_flip(predict(ite), ite),
                m_pose8(predict(sub), sub))

    res = {}
    print(f"\n{'глубина':>8}{'голова':>9}{'тест поза8':>13}{'train поза8':>13}"
          f"{'знак':>9}")
    for depth in [int(v) for v in args.depths.split(",")]:
        for kind in args.heads.split(","):
            vals = [run(depth, kind, s) for s in range(args.seeds)]
            t8 = float(np.mean([v[0] for v in vals]))
            fl = float(np.mean([v[1] for v in vals]))
            tr8 = float(np.mean([v[2] for v in vals]))
            res[f"{depth}/{kind}"] = dict(test_pose8=t8, train_pose8=tr8,
                                          flip=fl, seeds=[v[0] for v in vals])
            print(f"{depth:>8}{kind:>9}{t8:>13.4f}{tr8:>13.4f}{fl:>9.2%}")

    print("\n  ЧИТАТЬ ТАК, правило записано до запуска.")
    goal = ref["BAR coarse-only"]["pose8"]
    d12 = [v for k, v in res.items() if k.startswith("12/")]
    d24 = [v for k, v in res.items() if k.startswith("24/")]
    if d24 and min(v["test_pose8"] for v in d24) > goal * 1.5:
        print(f"  ЗОНД НЕГОДЕН: на полной глубине он даёт "
              f"{min(v['test_pose8'] for v in d24):.4f} против цели {goal:.4f}. "
              f"Значит\n  плох инструмент, а не глубина, и строку 12 читать нельзя.")
    elif d12:
        b = min(v["test_pose8"] for v in d12)
        gap = max(v["train_pose8"] - v["test_pose8"] for v in d12)
        print(f"  лучший на слое 12: {b:.4f} при цели {goal:.4f}")
        if b <= goal * 1.3:
            print("  Дело было в ДИСКРЕТНОЙ ЦЕЛИ, а не в глубине: информация о "
                  "действии\n  на слое 12 есть. Токенизатор под ранние выходы "
                  "осмыслен.")
        elif gap < -0.02:
            print(f"  train лучше теста на {-gap:.4f}: не хватает ДАННЫХ, а не "
                  f"глубины.\n  Вывод об отсутствии информации делать нельзя.")
        else:
            print("  Информации о действии на слое 12 недостаточно, и это уже "
                  "не\n  объясняется дискретной параметризацией. Дешёвая линия "
                  "закрыта надёжно.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(dict(refs=ref, probes=res, feats=args.feats, normed=bool(use_n),
                   epochs=args.epochs, seeds=args.seeds,
                   note="всё на одном разбиении из одного файла признаков"),
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
