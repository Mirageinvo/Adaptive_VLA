"""K-7c: с какой глубины эксперта грубые коды уже определены.

ВОПРОС. Если голова на состоянии слоя 12 воспроизводит то, что полная башня
выдаёт на слое 24, то башню можно останавливать на двенадцатом — это ещё
примерно вдвое поверх уже измеренного отказа от двух проходов.

ЦЕЛЬ — K_bar[:, 0, :], грубые коды САМОЙ BAR на полной глубине. Вопрос ранней
остановки формулируется как «воспроизвести раньше то, что даёт полная башня», а
не «угадать токенизатор». Совпадение с K_true считается вторичной метрикой.

ГЛАВНАЯ МЕТРИКА — ОШИБКА ДЕЙСТВИЯ, НЕ ТОЧНОСТЬ КОДОВ. Урок K-6e: декодер
принимает СУММУ уровней, поэтому промахнуться в соседний код почти бесплатно, а
кросс-энтропия по кодам ранжирует модели не так, как ошибка декодированного
действия. Точность кодов печатается, но решение принимается по ошибке позы.

ОБЯЗАТЕЛЬНЫЕ ОПОРЫ В ТАБЛИЦЕ, без них цифры не читаются:
  `final`   — вход action_lm_head, то есть ровно то, из чего BAR берёт коды.
              Голова на нём обязана давать почти 100%. Если нет — сломан зонд,
              а не модель, и остальные строки бессмысленны.
  случайная — коды из равномерного распределения: пол ошибки.
  BAR       — сами коды BAR, то есть нулевая ошибка по построению цели.

РАЗБИЕНИЕ ПО ЭПИЗОДАМ, не по наблюдениям: соседние наблюдения одного эпизода
почти одинаковы, и разбиение по строкам дало бы утечку и оптимистичный зонд.

ПРАВИЛО ЧТЕНИЯ, записано до запуска:
  ошибка действия на глубине d в пределах ~10% от ошибки на `final` -> глубину
      d можно считать достаточной, ранняя остановка осмысленна;
  ошибка монотонно падает вплоть до последнего слоя -> ранняя остановка не
      работает, многоглубинная архитектура лишается опоры.

Запуск:
    python3 experiments/k7c_depth_probe.py --selftest
    python3 experiments/k7c_depth_probe.py --feats data/k7b_depth_4k.npz \\
        --ckpt <ckpt> --epochs 60 --out data/k7c_depth_probe.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np

N_POS, N_LEVEL = 16, 3


def split_by_episode(epi, seed=0, frac=(0.7, 0.15)):
    """Разбиение ПО ЭПИЗОДАМ. Возвращает три булевы маски."""
    ep = np.unique(epi)
    rng = np.random.default_rng(seed)
    ep = rng.permutation(ep)
    n1 = int(len(ep) * frac[0])
    n2 = int(len(ep) * (frac[0] + frac[1]))
    sets = (set(ep[:n1].tolist()), set(ep[n1:n2].tolist()), set(ep[n2:].tolist()))
    return [np.array([e in s for e in epi]) for s in sets]


def selftest():
    # 1. Разбиение по эпизодам: ни один эпизод не попадает в две части.
    epi = np.repeat(np.arange(50), 8)
    tr, va, te = split_by_episode(epi, seed=0)
    assert (tr | va | te).all() and not (tr & va).any() and not (tr & te).any()
    assert not (va & te).any()
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not (set(epi[a]) & set(epi[b])), "эпизод попал в две части"
    assert 0.6 < tr.mean() < 0.8, tr.mean()

    # 2. Разбиение по СТРОКАМ дало бы утечку — показываем, что это другое.
    rng = np.random.default_rng(0)
    rows = rng.random(len(epi)) < 0.7
    leak = len(set(epi[rows]) & set(epi[~rows])) / len(np.unique(epi))
    assert leak > 0.9, f"утечка при построчном разбиении {leak:.2f}"

    # 3. Точность кодов и ошибка действия — РАЗНЫЕ величины, и ранжируют
    #    по-разному. Урок K-6e: декодер берёт сумму, соседний код почти
    #    бесплатен. Модель A угадывает коды чаще, но промахивается далеко;
    #    модель B ошибается в кодах чаще, но всегда рядом.
    book = np.arange(8).astype(float)[:, None] * 10.0       # далеко разнесены
    true = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    A = np.array([0, 1, 2, 3, 4, 5, 7, 6])                  # 6 из 8 верно
    B = np.array([1, 0, 3, 2, 5, 4, 7, 6])                  # 0 из 8 верно
    accA = (A == true).mean(); accB = (B == true).mean()
    errA = np.abs(book[A] - book[true]).mean()
    errB = np.abs(book[B] - book[true]).mean()
    assert accA > accB and errA < errB
    A2 = np.array([0, 1, 2, 3, 4, 5, 6, 0])                 # 7 из 8, но далеко
    err2 = np.abs(book[A2] - book[true]).mean()
    assert (A2 == true).mean() > accA and err2 > errA, (
        "точность выросла, а ошибка тоже — именно поэтому решаем по ошибке")

    print("самопроверка пройдена: разбиение по эпизодам без утечки, "
          "точность кодов и ошибка действия ранжируют по-разному")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--feats", default="data/k7b_depth_4k.npz")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="data/k7c_depth_probe.json")
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

    z = np.load(args.feats, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    keys = meta["keys"]
    K_bar, K_true, epi = z["K_bar"], z["K_true"], z["episode"]
    act = z["act"]
    N, n_codes = len(epi), int(meta["n_codes"])
    dev = torch.device(args.device)
    print(f"  {N} наблюдений, {len(np.unique(epi))} эпизодов, глубины {keys}")

    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None)).to(dev).eval()
    with torch.no_grad():
        idx = torch.arange(n_codes, device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    def decode_coarse(codes):
        """Действие ТОЛЬКО из грубого уровня — тот режим, ради которого всё."""
        outs = []
        for i0 in range(0, len(codes), 256):
            k = torch.as_tensor(codes[i0:i0 + 256]).long().to(dev)
            with torch.no_grad():
                x, _ = codec._decode(E[0][k], embodiment_ids=0)
            outs.append(x[..., :7].float().cpu().numpy())
        return np.concatenate(outs)

    tr, va, te = split_by_episode(epi, seed=0)
    print(f"  train {tr.sum()}, val {va.sum()}, test {te.sum()}")
    tgt = K_bar[:, 0, :]                       # цель: грубые коды самой BAR
    a_ref = decode_coarse(tgt)                 # эталон: coarse-only от BAR
    rng_pose = float(act[..., :6].max() - act[..., :6].min())

    def pose_rms(codes, mask):
        d = decode_coarse(codes[mask]) - a_ref[mask]
        return float(np.sqrt((d[..., :6] ** 2).mean())) / rng_pose

    rnd = np.random.default_rng(0).integers(0, n_codes, size=tgt.shape)
    floor = pose_rms(rnd, te)
    print(f"\n  пол (случайные коды):     {floor:.4f}")
    print(f"  цель (сами коды BAR):     0.0000 по построению")

    def train_probe(X, seed):
        torch.manual_seed(seed)
        d_in = X.shape[-1]
        mods, d = [], d_in
        mods.append(nn.LayerNorm(d_in))        # масштабы глубин различаются
        for _ in range(args.layers):
            mods += [nn.Linear(d, args.hidden), nn.GELU()]
            d = args.hidden
        trunk = nn.Sequential(*mods).to(dev)
        head = nn.Linear(d, n_codes).to(dev)
        opt = torch.optim.AdamW(list(trunk.parameters()) + list(head.parameters()),
                                lr=args.lr, weight_decay=0.01)
        Xt = torch.as_tensor(X, dtype=torch.float32)
        Yt = torch.as_tensor(tgt, dtype=torch.long)
        itr = np.where(tr)[0]
        steps = args.epochs * max(1, len(itr) // args.batch)
        # ПРОГРЕВ ОБЯЗАТЕЛЕН: без него в K-6e глубокий вариант расходился, и
        # глубина ошибочно выглядела вредной.
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, s / max(1, int(steps * args.warmup_frac)))
            * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
        best, best_state = None, None
        rg = np.random.default_rng(seed)
        for ep in range(args.epochs):
            trunk.train(); head.train()
            perm = rg.permutation(itr)          # перемешивание НА КАЖДУЮ эпоху
            for i0 in range(0, len(perm), args.batch):
                sel = perm[i0:i0 + args.batch]
                xb = Xt[sel].to(dev); yb = Yt[sel].to(dev)
                lg = head(trunk(xb))
                loss = nn.functional.cross_entropy(
                    lg.reshape(-1, n_codes), yb.reshape(-1))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    list(trunk.parameters()) + list(head.parameters()), 1.0)
                opt.step(); sched.step()
            trunk.eval(); head.eval()
            pv = predict(trunk, head, Xt, va)
            # ОТБОР ПО ТОЙ ЖЕ ВЕЛИЧИНЕ, ПО КОТОРОЙ ОТЧИТЫВАЕМСЯ. В K-6e отбор
            # шёл по CE, а отчёт по ошибке действия — разные чекпойнты.
            m = pose_rms_pred(pv, va)
            if best is None or m < best:
                best = m
                best_state = ({k: v.detach().clone() for k, v in trunk.state_dict().items()},
                              {k: v.detach().clone() for k, v in head.state_dict().items()})
        trunk.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
        return trunk, head

    def predict(trunk, head, Xt, mask):
        out = []
        ii = np.where(mask)[0]
        with torch.no_grad():
            for i0 in range(0, len(ii), 512):
                xb = Xt[ii[i0:i0 + 512]].to(dev)
                out.append(head(trunk(xb)).argmax(-1).cpu().numpy())
        return np.concatenate(out)

    def pose_rms_pred(pred, mask):
        d = decode_coarse(pred) - a_ref[mask]
        return float(np.sqrt((d[..., :6] ** 2).mean())) / rng_pose

    res = {}
    print(f"\n{'глубина':>8}{'ошибка позы':>14}{'доля от пола':>14}"
          f"{'точность кодов':>16}{'vs токенизатор':>16}")
    for k_ in keys:
        X = z[f"h_{k_}"].astype(np.float32)
        errs, accs, accs_t = [], [], []
        for s in range(args.seeds):
            trunk, head = train_probe(X, s)
            p = predict(trunk, head, torch.as_tensor(X, dtype=torch.float32), te)
            errs.append(pose_rms_pred(p, te))
            accs.append(float((p == tgt[te]).mean()))
            accs_t.append(float((p == K_true[te][:, 0, :]).mean()))
        e = float(np.mean(errs))
        res[str(k_)] = dict(pose_rms=e, pose_rms_seeds=errs,
                            code_acc=float(np.mean(accs)),
                            code_acc_vs_true=float(np.mean(accs_t)),
                            frac_of_floor=e / floor)
        print(f"{str(k_):>8}{e:>14.4f}{e / floor:>14.2f}"
              f"{np.mean(accs):>15.1%}{np.mean(accs_t):>16.1%}")

    fin = res[str(keys[-1])]["pose_rms"]
    if fin > 0.25 * floor:
        print(f"\n  ВНИМАНИЕ: на `final` ошибка {fin:.4f} при поле {floor:.4f}.")
        print("  Голова на входе action_lm_head обязана почти точно "
              "воспроизводить коды BAR.\n  Если не воспроизводит — сломан зонд, "
              "и остальные строки читать нельзя.")
    else:
        ok = [k_ for k_ in keys if k_ != "final"
              and res[str(k_)]["pose_rms"] <= 1.1 * fin]
        print("\n  ЧИТАТЬ ТАК, правило записано до запуска.")
        if ok:
            print(f"  Достаточные глубины (в пределах 10% от `final`): {ok}")
            print(f"  Самая ранняя: {ok[0]} — ранняя остановка осмысленна.")
        else:
            print("  Ни одна промежуточная глубина не приблизилась к `final`:")
            print("  ошибка падает до последнего слоя, ранняя остановка не "
                  "работает,\n  и многоглубинная архитектура лишается опоры.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(dict(depths=res, floor_random=floor, n_obs=int(N),
                   feats=args.feats, epochs=args.epochs, seeds=args.seeds,
                   split="по эпизодам 70/15/15",
                   target="K_bar[:,0,:], грубые коды BAR на полной глубине"),
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
