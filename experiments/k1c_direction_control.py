"""K-1c: величина смещения латенты или его направление.

ПОВОД. K-1 показал, что при равной ошибке латенты вариант со старым суффиксом
декодируется хуже — 1.39 на грубом уровне, ДИ [1.29, 1.49]. Отсюда напрашивался
вывод «устаревший суффикс имеет собственную цену».

Но у A и B смещения не только разной ВЕЛИЧИНЫ, но и разного НАПРАВЛЕНИЯ:

    δ_A = Δe_0                        разность векторов грубого словаря
    δ_B = Δe_0 + поправки уровней 1,2 остаток после жадной проекции

Декодер вполне может быть просто анизотропен — чувствительнее к смещениям
вдоль «грубых» осей, чем вдоль тонких остаточных. Тогда 1.39 объясняется
геометрией декодера, и никакого «устаревания» за этим нет.

Плюс методическая тонкость: корзины уравнивают величину лишь ГРУБО. Внутри
корзины ошибки латенты всё ещё различаются, и часть эффекта может быть
остатком этого различия. Здесь нормы уравниваются ТОЧНО, масштабированием.

ЧАСТЬ 1 — прямой контроль для K-1. Берём настоящие смещения δ_A и δ_B, обоим
задаём РОВНО одну норму, декодируем, сравниваем. Это заодно более строгая
версия самого K-1: величина уравнена не корзиной, а тождественно.

ЧАСТЬ 2 — общая анизотропия. Смещения равной нормы вдоль направлений словаря
каждого уровня и вдоль изотропного случайного. Никакой логики суффикса.

Возможно, потому что `_decode` (modeling_actioncodec.py:279) принимает латенту
напрямую, а не токены: подавать можно произвольные точки, не только узлы.

ПРАВИЛО ЧТЕНИЯ, зафиксировано до запуска:
  Часть 1. Если при ТОЧНО равной норме B по-прежнему заметно лучше (скажем,
      A/B > 1.15) — направление смещения существенно, эффект K-1 не сводится
      к величине. Если отношение около 1.0 — эффект K-1 был остатком
      неполного выравнивания величины внутри корзин.
  Часть 2. Если смещения вдоль грубого словаря вредят сильнее тонких при
      равной норме — декодер анизотропен, и это объясняет часть 1 без всякого
      «устаревания суффикса». Если одинаково — анизотропии нет.

Запуск:
    python3 experiments/k1c_direction_control.py --zarr <путь>/libero10_N500.zarr
"""

import argparse
import os
import sys

import einops
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1_residual_cost import (  # noqa: E402
    latent_from_codes,
    load_codec,
    pick_candidates,
    projected_codebooks,
    requantize_at,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--model", default="ZibinDong/ActionCodec-Base-RVQft")
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--n-chunks", type=int, default=96)
    ap.add_argument("--n-trials", type=int, default=24, help="позиций/направлений")
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import zarr

    model = load_codec(os.path.abspath(args.root), args.model, args.device)
    V, L, P = model.vocab_size, model.num_quantizers, model.n_tokens_per_quantizer
    name = list(model.config.embodiment_config.keys())[args.embodiment]
    cfg = model.config.embodiment_config[name]
    T, D_act = int(cfg["freq"] * cfg["duration"]), cfg["action_dim"]

    z = zarr.open(os.path.abspath(args.zarr), mode="r")
    acts = np.asarray(z["data"]["action"])
    ends = np.asarray(z["meta"]["episode_ends"])
    chunks, start = [], 0
    for e in ends:
        ep = acts[start:e]
        chunks += [ep[i * T:(i + 1) * T] for i in range(len(ep) // T)]
        start = e
    A = np.stack(chunks).astype(np.float32)
    if A[:, :, -1].min() < -0.5:
        A[:, :, -1] = (1.0 - A[:, :, -1]) / 2.0
    idx = np.random.default_rng(0).choice(len(A), size=min(args.n_chunks, len(A)),
                                          replace=False)
    a = torch.from_numpy(A[idx]).to(args.device)
    B = len(a)
    scale = float(a.max() - a.min())

    E = projected_codebooks(model, args.device)
    Dz = E.shape[-1]
    gen = torch.Generator(device=args.device).manual_seed(1)
    print(f"чанков {B}, размах {scale:.2f}, латента {Dz}-мерная\n")

    with torch.no_grad():
        flat = torch.as_tensor(np.asarray(model.encode(a, embodiment_ids=args.embodiment)),
                               device=args.device, dtype=torch.long)
        codes = einops.rearrange(flat, "b (n m) -> b m n", m=P)
        h0 = latent_from_codes(E, codes)

        def dec_latent(h):
            """Декодирование ПРОИЗВОЛЬНОЙ латенты, минуя токены."""
            rec, _ = model._decode(h, args.embodiment, None)
            return rec[..., :D_act]

        base = dec_latent(h0)
        # сверка: путь через латенту должен совпасть с путём через токены
        rec_tok, _ = model.decode(flat.cpu().numpy().astype(np.int64),
                                  embodiment_ids=args.embodiment)
        gap = (base - torch.as_tensor(np.asarray(rec_tok)[..., :D_act],
                                      device=args.device)).abs().max().item()
        print(f"сверка _decode против decode: {gap:.2e}")
        assert gap < 1e-4, "прямое декодирование латенты расходится с токенным"

        def act_err(h):
            return (dec_latent(h) - base).abs().flatten(1).amax(-1) / scale

        def apply_at(p, delta):
            h = h0.clone()
            h[:, p] = h[:, p] + delta
            return h

        def rescale(d, tau):
            """Задать вектору норму так, чтобы ‖d‖/√Dz == tau."""
            n = d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return d / n * (tau * np.sqrt(Dz))

        taus = [0.004, 0.008, 0.015, 0.030, 0.060]

        # ---------------- ЧАСТЬ 1 ----------------
        print("\n" + "=" * 74)
        print("ЧАСТЬ 1. НАСТОЯЩИЕ СМЕЩЕНИЯ A И B ПРИ ТОЧНО РАВНОЙ НОРМЕ")
        print("=" * 74)
        print("Уровень 0, кандидаты local. Величина уравнена тождественно,\n"
              "различается только направление.\n")
        print(f"{'норма':>8}{'A':>10}{'B':>10}{'A/B':>7}")
        for tau in taus:
            eA, eB = [], []
            for _ in range(args.n_trials):
                p = int(torch.randint(P, (1,), generator=gen, device=args.device))
                v = pick_candidates(E, codes[:, p, 0], 0, "local", args.knn, gen)
                cA = codes.clone()
                cA[:, p, 0] = v
                cB = requantize_at(E, cA, h0, p, 0)
                dA = latent_from_codes(E, cA)[:, p] - h0[:, p]
                dB = latent_from_codes(E, cB)[:, p] - h0[:, p]
                eA.append(act_err(apply_at(p, rescale(dA, tau))).cpu().numpy())
                eB.append(act_err(apply_at(p, rescale(dB, tau))).cpu().numpy())
            mA, mB = np.median(np.concatenate(eA)), np.median(np.concatenate(eB))
            print(f"{tau:>8.3f}{mA:>10.4f}{mB:>10.4f}{mA/max(mB,1e-9):>7.2f}")

        # ---------------- ЧАСТЬ 2 ----------------
        print("\n" + "=" * 74)
        print("ЧАСТЬ 2. АНИЗОТРОПИЯ: НАПРАВЛЕНИЯ СЛОВАРЕЙ ПРИ РАВНОЙ НОРМЕ")
        print("=" * 74)
        print("Никакой логики суффикса: просто смещения заданной нормы вдоль\n"
              "разностей векторов словаря каждого уровня и вдоль случайного.\n")
        kinds = [f"словарь ур.{j}" for j in range(L)] + ["изотропное"]
        print(f"{'норма':>8}" + "".join(f"{k:>15}" for k in kinds))
        for tau in taus:
            meds = []
            for j in range(L + 1):
                acc = []
                for _ in range(args.n_trials):
                    p = int(torch.randint(P, (1,), generator=gen, device=args.device))
                    if j < L:
                        u = torch.randint(V, (B,), generator=gen, device=args.device)
                        w = torch.randint(V, (B,), generator=gen, device=args.device)
                        d = E[j][u] - E[j][w]
                    else:
                        d = torch.randn(B, Dz, generator=gen, device=args.device)
                    acc.append(act_err(apply_at(p, rescale(d, tau))).cpu().numpy())
                meds.append(np.median(np.concatenate(acc)))
            print(f"{tau:>8.3f}" + "".join(f"{m:>15.4f}" for m in meds))

    print("""
КАК ЧИТАТЬ.

Часть 1. Отношение A/B здесь получено при ТОЖДЕСТВЕННО равной норме, тогда как
в K-1 величина уравнивалась лишь корзинами. Если отношение осталось около
значения из K-1 (~1.4) — эффект настоящий и он про направление. Если упало к
1.0 — эффект K-1 был остатком неполного выравнивания величины.

Часть 2. Если строки для разных уровней различаются при одной норме, декодер
анизотропен: смещения вдоль одних осей латенты вредят сильнее других. Тогда
часть 1 объясняется этим, а не «устареванием суффикса», и формулировать надо
как свойство декодера. Изотропный столбец — опорный уровень.""")


if __name__ == "__main__":
    main()
