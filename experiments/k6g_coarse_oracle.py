"""K-6g: можно ли выбрать грубые коды лучше, ничего больше не вычисляя.

ЗАЧЕМ. K-6e закрыл линию уточнения ПО h: пять параметризаций, сходимость,
ничтожный разброс — ни одна не обошла «только грубый уровень» (0.0277 против
BAR 0.0234). Но грубый код рождается в том проходе, который мы делаем в любом
случае, и BAR попадает в него лишь в 87.1% случаев.

Причина полагать, что запас есть: BAR выбирает код по argmax логитов,
обученных КРОСС-ЭНТРОПИЕЙ, а не по тому, какой код даёт лучшее ДЕЙСТВИЕ. Мы
измерили, что эти критерии систематически расходятся: декодер принимает сумму
латентов (§1), поэтому промах на соседний по эмбеддингу код почти бесплатен
для действия, но полностью штрафуется кросс-энтропией.

ГЛАВНЫЙ ОРАКУЛ — БЕЗ ТОНКИХ УРОВНЕЙ ВООБЩЕ:

    a = D( E0[k0] )

Это ровно та архитектура, которую хотим запускать за один проход. Варианты с
тонкими кодами от BAR или истинными считаются только как ДИАГНОСТИКА: те коды
вычислены как остаток от ДРУГОГО грубого кода и при его замене внутренне
несогласованы, поэтому верхней границей однопроходного метода не являются.

ПОЧЕМУ ЖАДНО, А НЕ НЕЗАВИСИМО ПО ПОЗИЦИЯМ. Декодер связывает шестнадцать
позиций: замена кода в одной меняет всё действие. Независимая оптимизация дала
бы величину, недостижимую совместно. Полный перебор 2048^16 невозможен, поэтому
жадно: по одной позиции за шаг, с пересчётом выигрышей.

ПОЧЕМУ ТОЛЬКО TOP-L КАНДИДАТОВ. Перебор всех 2048 кодов в каждой из 16 позиций
это 32768 прогонов декодера на наблюдение за круг. Ограничение верхушкой
логитов делает задачу выполнимой И отвечает на практический вопрос: если весь
выигрыш лежит в top-32, его берёт дешёвый переранжировщик БЕЗ изменения VLA.
При L = 2048 получается полный жадный оракул.

ПРАВИЛО ЧТЕНИЯ, записано до запуска:
  оракул заметно ниже 0.0277 -> грубый код выбирается неоптимально, и есть
      что отыграть ТЕМ ЖЕ проходом, нулевой ценой;
  оракул около 0.0277 -> грубый код уже почти оптимален по действию,
      направление закрыто;
  выигрыш есть, но только при большом L -> дешёвым переранжировщиком не
      возьмёшь, нужна другая голова.

Запуск:
    python3 experiments/k6g_coarse_oracle.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k6g_coarse_oracle.py \\
        --feats data/k6d_h30k.npz --ckpt <ckpt> --n 500 --out data/k6g.json
"""

import argparse
import json
import os
import sys

import numpy as np

N_POS, N_LEVEL = 16, 3


def greedy_improve(score_fn, codes, cand, rounds=3):
    """Жадная замена кодов по позициям. score_fn(codes) -> ошибка (меньше лучше).

    codes: (P,) текущие коды; cand: (P, L) кандидаты на позицию.
    Возвращает (коды, история ошибок, сколько позиций сменилось).

    Жадность НЕ ОПТИМАЛЬНА: на каждом шаге берётся лучшая одиночная замена,
    что вообще говоря не даёт наилучшего совместного набора. Это НИЖНЯЯ оценка
    оракула, и так её и надо называть.
    """
    cur = codes.copy()
    hist = [float(score_fn(cur))]
    changed = 0
    for _ in range(rounds):
        best = (hist[-1], None)
        for p in range(len(cur)):
            for c in cand[p]:
                if c == cur[p]:
                    continue
                trial = cur.copy()
                trial[p] = c
                s = float(score_fn(trial))
                if s < best[0] - 1e-12:
                    best = (s, (p, c))
        if best[1] is None:
            break                       # улучшений больше нет
        p, c = best[1]
        cur[p] = c
        changed += 1
        hist.append(best[0])
    return cur, hist, changed


def selftest():
    rng = np.random.default_rng(0)
    P, L = 6, 4
    target = rng.integers(0, 10, size=P)
    cand = np.stack([rng.permutation(10)[:L] for _ in range(P)])
    # подложим правильный ответ в кандидаты половины позиций
    for p in range(0, P, 2):
        cand[p, 0] = target[p]
    start = np.array([cand[p, -1] for p in range(P)])

    def score(c):
        return float(np.abs(c - target).sum())

    out, hist, ch = greedy_improve(score, start, cand, rounds=20)

    # 1. МОНОТОННОСТЬ. Жадность обязана только улучшать; рост означает ошибку
    #    в сравнении или в применении замены.
    assert all(hist[i + 1] <= hist[i] + 1e-12 for i in range(len(hist) - 1)), hist
    assert score(out) <= score(start), "итог не лучше старта"

    # 2. ДОСТИЖИМЫЕ ПОЗИЦИИ ИСПРАВЛЯЮТСЯ. Там, где верный код лежит среди
    #    кандидатов, жадность обязана его найти.
    for p in range(0, P, 2):
        assert out[p] == target[p], f"позиция {p}: {out[p]} вместо {target[p]}"

    # 3. L = 1 (только текущий код) обязан дать ТОЧНО старт: оракул при
    #    отсутствии выбора не может ничего улучшить.
    only = np.stack([[start[p]] for p in range(P)])
    out1, hist1, ch1 = greedy_improve(score, start, only, rounds=5)
    assert (out1 == start).all() and ch1 == 0, "при L=1 оракул обязан стоять"
    assert len(hist1) == 1

    print("самопроверка пройдена: жадность монотонна, достижимые позиции "
          f"исправляются ({ch} замен), при L=1 оракул не двигается")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--feats", default="data/k6d_h30k.npz")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=500, help="наблюдений из test")
    ap.add_argument("--tops", default="8,32,128",
                    help="сколько верхних кандидатов логитов перебирать")
    ap.add_argument("--rounds", type=int, default=4, help="жадных кругов")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    sys.path.insert(0, os.path.abspath(args.root))
    import torch
    import torch.nn as nn
    from huggingface_hub import snapshot_download

    import actioncodec  # noqa: F401
    from utils import VisionLanguageActionProcessor  # noqa: E402

    dev = torch.device(args.device)
    z = np.load(args.feats, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    h_all, K_bar, K_true, act_all = z["h"], z["K_bar"], z["K_true"], z["act"]
    epi = z["episode"]
    n_codes = int(meta["n_codes"])

    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = ac if hasattr(ac, "vq") else getattr(ac, "codec", None)
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь в action_processor")
    codec = codec.to(dev).eval()
    for prm in codec.parameters():
        prm.requires_grad_(False)
    with torch.no_grad():
        # ИНДЕКСЫ НА ТОМ ЖЕ УСТРОЙСТВЕ, ЧТО КНИГИ: codec уже перенесён строкой
        # выше, а arange по умолчанию создаётся на CPU, и F.embedding падает.
        idx = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(dev)
    print(f"  кодбуки: {tuple(E.shape)}")

    # ЛОГИТЫ ПЕРВОГО БЛОКА ВОССТАНАВЛИВАЮТСЯ ИЗ h. bar.py:1247-1248 нормирует
    # состояние экспертной башни и подаёт в action_lm_head; наш h — это его
    # ВХОД, то есть уже нормированный. Значит достаточно самой головы, а не
    # всей модели: веса лежат в action_components.bin.
    local = snapshot_download(args.ckpt) if not os.path.isdir(args.ckpt) else args.ckpt
    comp = torch.load(os.path.join(local, "action_components.bin"),
                      map_location="cpu")
    if "action_lm_head" not in comp:
        raise SystemExit(f"в action_components.bin нет action_lm_head: "
                         f"{list(comp)}")
    sd = comp["action_lm_head"]
    W = sd["weight"]
    lm = nn.Linear(W.shape[1], W.shape[0], bias="bias" in sd).to(dev)
    lm.load_state_dict(sd)
    lm.eval()
    print(f"  голова кодов: {tuple(W.shape)}")

    # --- выбираем наблюдения из ТОЙ ЖЕ test-части --------------------------
    def split_by_episode(ep, seed=0, fr=(0.7, 0.15)):
        u = np.unique(ep)
        r = np.random.default_rng(seed).permutation(len(u))
        n1, n2 = int(len(u) * fr[0]), int(len(u) * (fr[0] + fr[1]))
        return np.where(np.isin(ep, list(set(u[r[n2:]]))))[0]

    ite = split_by_episode(epi, seed=args.seed)
    sel = np.random.default_rng(args.seed).choice(
        ite, size=min(args.n, len(ite)), replace=False)
    print(f"  наблюдений: {len(sel)} из {len(ite)} в test")

    h = torch.as_tensor(h_all[sel], dtype=torch.float32).to(dev)
    act = act_all[sel]
    rng_pose = float(act_all[..., :6].max() - act_all[..., :6].min())
    with torch.no_grad():
        logits = lm(h)                                   # (n, 16, 2048)
    order = logits.argsort(dim=-1, descending=True).cpu().numpy()

    # СВЕРКА: argmax логитов обязан совпасть с сохранёнными кодами BAR.
    # Если нет — h снят не оттуда или голова не та, и всё дальнейшее неверно.
    match = float((order[:, :, 0] == K_bar[sel][:, 0, :]).mean())
    print(f"  argmax логитов против сохранённых кодов BAR: {match:.1%}")
    if match < 0.99:
        raise SystemExit(
            "argmax восстановленных логитов НЕ совпадает с кодами BAR — "
            "значит h или голова не те, и оракул считать нельзя")

    def decode_codes(k0, fine=None):
        """k0: (n, 16). fine: None (только грубый) или (n, 2, 16)."""
        with torch.no_grad():
            zq = E[0][torch.as_tensor(k0).long().to(dev)]
            if fine is not None:
                f = torch.as_tensor(fine).long().to(dev)
                for j in range(f.shape[1]):
                    zq = zq + E[j + 1][f[:, j, :]]
            x, _ = codec._decode(zq, embodiment_ids=0)
            return x[..., :7].cpu().numpy()

    def err(a):
        return float(np.sqrt(((a[..., :6] - act[..., :6]) ** 2).mean())) / rng_pose

    base_coarse = err(decode_codes(K_bar[sel][:, 0, :]))
    base_bar = err(decode_codes(K_bar[sel][:, 0, :], K_bar[sel][:, 1:, :]))
    base_exp = err(decode_codes(K_true[sel][:, 0, :], K_true[sel][:, 1:, :]))
    print(f"\n  только грубый (BAR):  {base_coarse:.4f}")
    print(f"  BAR, все три уровня:  {base_bar:.4f}")
    print(f"  эксперт:              {base_exp:.4f}")

    # --- жадный оракул по каждому наблюдению отдельно ----------------------
    res = {"base_coarse": base_coarse, "base_bar": base_bar,
           "base_expert": base_exp, "n": int(len(sel))}
    print(f"\n  {'L':>6}{'оракул':>10}{'к грубому':>12}{'замен/16':>11}"
          f"{'доля к BAR':>13}")
    for L in [int(v) for v in args.tops.split(",")]:
        outs, chs = [], []
        for i in range(len(sel)):
            cand = order[i, :, :L]
            a_t = act[i:i + 1]

            def score(c, _i=i, _a=a_t):
                a = decode_codes(c[None, :])
                return float(np.sqrt(((a[..., :6] - _a[..., :6]) ** 2).mean()))

            c0 = K_bar[sel[i], 0, :].copy()
            c, hist, ch = greedy_improve(score, c0, cand, rounds=args.rounds)
            outs.append(c); chs.append(ch)
            if i % 100 == 0:
                print(f"    {i}/{len(sel)}", flush=True)
        e = err(decode_codes(np.stack(outs)))
        gain_frac = ((base_coarse - e) / max(base_coarse - base_bar, 1e-12))
        res[f"top{L}"] = dict(err=e, changed=float(np.mean(chs)),
                              gain_fraction_of_bar=gain_frac)
        print(f"  {L:>6}{e:>10.4f}{e / base_coarse - 1:>+12.1%}"
              f"{np.mean(chs):>11.2f}{gain_frac:>13.1%}")

    print("\n  ЧИТАТЬ ТАК, правило записано до запуска.")
    print("  Оракул заметно ниже «только грубый» — код выбирается неоптимально,")
    print("  и есть что отыграть ТЕМ ЖЕ проходом, нулевой ценой.")
    print("  Оракул около него — грубый код уже почти оптимален по действию.")
    print("  Выигрыш только при большом L — дешёвым переранжировщиком не взять.")
    print("\n  ЖАДНОСТЬ НЕ ОПТИМАЛЬНА: это НИЖНЯЯ оценка оракула, совместный")
    print("  перебор 2048^16 невозможен.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
