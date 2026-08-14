"""K-1d: профиль чувствительности декодера в латентном пространстве.

ПОВОД. K-1c установил два факта:
  1. при ТОЧНО равной норме смещение варианта A вредит в 1.27 раза сильнее,
     чем смещение варианта B (переквантованный суффикс);
  2. смещения вдоль разностей векторов словаря вредят в 1.3-1.5 раза сильнее
     изотропных случайных той же длины.

Объяснение напрашивалось такое: δ_B направлен «наружу» от кодового
подпространства, а наружу декодер реагирует слабее.

ЭТА ФОРМУЛИРОВКА НЕТОЧНА. 6144 вектора словаря (3 уровня по 2048) в
512-мерном пространстве почти наверняка натягивают его целиком, и никакого
«наружу» нет. Дело в РАСПРЕДЕЛЕНИИ ЭНЕРГИИ: у векторов словаря она собрана в
немногих главных направлениях, а изотропное направление размазано по всем 512
и потому слабо пересекается с чувствительными осями.

ЧТО МЕРЯЕМ.

  A. Спектр словаря: сколько главных компонент несут его энергию. Заодно
     проверка, что «подпространства» действительно нет — ранг полный.
  B. Профиль чувствительности: смещения РАВНОЙ НОРМЫ вдоль полос главных
     компонент (1-8, 9-32, ...) и вдоль изотропного. Если чувствительность
     падает с номером полосы — анизотропия измерена количественно.
  C. Где лежит энергия настоящих смещений δ_A и δ_B по этим же полосам.
     Замыкает объяснение: B должен иметь меньшую долю в чувствительных
     полосах, и именно этим отличаться.

ПРАВИЛО ЧТЕНИЯ, зафиксировано до запуска:
  Если чувствительность заметно падает с номером полосы И энергия δ_B
      смещена в менее чувствительные полосы против δ_A — механизм объяснён:
      переквантизация убирает из смещения долю, лежащую в чувствительных
      направлениях.
  Если чувствительность по полосам ровная — анизотропии по главным
      компонентам нет, и 1.27 из K-1c объясняется чем-то другим; тогда
      объяснение писать нельзя, а факт (1.27) остаётся.
  Если энергия δ_A и δ_B распределена одинаково — механизм не в этом.

Запуск:
    python3 experiments/k1d_sensitivity_profile.py --zarr <путь>/libero10_N500.zarr
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

BANDS = [(0, 8), (8, 32), (32, 128), (128, 512)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--model", default="ZibinDong/ActionCodec-Base-RVQft")
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--n-chunks", type=int, default=96)
    ap.add_argument("--n-trials", type=int, default=24)
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
    Bn = len(a)
    scale = float(a.max() - a.min())

    E = projected_codebooks(model, args.device)
    Dz = E.shape[-1]
    gen = torch.Generator(device=args.device).manual_seed(1)
    print(f"чанков {Bn}, размах {scale:.2f}, латента {Dz}-мерная\n")

    # ---------- A. спектр словаря ----------
    flat_cb = E.reshape(-1, Dz).double()
    flat_cb = flat_cb - flat_cb.mean(0, keepdim=True)   # смещение есть разность,
    U, S, Vt = torch.linalg.svd(flat_cb, full_matrices=False)   # среднее сокращается
    energy = (S ** 2)
    cum = torch.cumsum(energy, 0) / energy.sum()
    rank = int((S > S[0] * 1e-6).sum())
    print("=" * 74)
    print("A. СПЕКТР СЛОВАРЯ")
    print("=" * 74)
    print(f"векторов {flat_cb.shape[0]}, размерность {Dz}, численный ранг {rank}")
    for k in (8, 32, 128, 256, 512):
        if k <= Dz:
            print(f"  первые {k:>3} компонент несут {float(cum[k-1]):.1%} энергии")
    if rank >= Dz - 1:
        print("\nРанг полный: «подпространства словаря» не существует, и моя\n"
              "прежняя формулировка про «наружу» была неверна. Значит дело в\n"
              "распределении энергии, а не в наличии ортогонального дополнения.")
    PC = Vt.float()                                     # (Dz, Dz), строки — компоненты

    with torch.no_grad():
        flat = torch.as_tensor(np.asarray(model.encode(a, embodiment_ids=args.embodiment)),
                               device=args.device, dtype=torch.long)
        codes = einops.rearrange(flat, "b (n m) -> b m n", m=P)
        h0 = latent_from_codes(E, codes)

        rec0, _ = model._decode(h0, args.embodiment, None)
        base = rec0[..., :D_act]

        def act_err(h):
            rec, _ = model._decode(h, args.embodiment, None)
            return ((rec[..., :D_act] - base).abs().flatten(1).amax(-1) / scale)

        def apply_at(p, delta):
            h = h0.clone()
            h[:, p] = h[:, p] + delta
            return h

        def rescale(d, tau):
            n = d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return d / n * (tau * np.sqrt(Dz))

        taus = [0.008, 0.015, 0.030]

        # ---------- B. профиль чувствительности ----------
        print("\n" + "=" * 74)
        print("B. ЧУВСТВИТЕЛЬНОСТЬ ПО ПОЛОСАМ ГЛАВНЫХ КОМПОНЕНТ")
        print("=" * 74)
        print("Смещения РАВНОЙ нормы, направленные в разные полосы спектра.\n")
        names = [f"ГК {lo+1}-{hi}" for lo, hi in BANDS] + ["изотропное"]
        print(f"{'норма':>8}" + "".join(f"{n:>13}" for n in names))
        prof = {}
        for tau in taus:
            meds = []
            for lo, hi in BANDS + [(None, None)]:
                acc = []
                for _ in range(args.n_trials):
                    p = int(torch.randint(P, (1,), generator=gen, device=args.device))
                    if lo is None:
                        d = torch.randn(Bn, Dz, generator=gen, device=args.device)
                    else:
                        c = torch.randn(Bn, hi - lo, generator=gen, device=args.device)
                        d = c @ PC[lo:hi]
                    acc.append(act_err(apply_at(p, rescale(d, tau))).cpu().numpy())
                meds.append(float(np.median(np.concatenate(acc))))
            prof[tau] = meds
            print(f"{tau:>8.3f}" + "".join(f"{m:>13.4f}" for m in meds))

        # ---------- C. где лежит энергия настоящих смещений ----------
        print("\n" + "=" * 74)
        print("C. РАСПРЕДЕЛЕНИЕ ЭНЕРГИИ НАСТОЯЩИХ СМЕЩЕНИЙ")
        print("=" * 74)
        eA, eB = [], []
        for _ in range(args.n_trials * 2):
            p = int(torch.randint(P, (1,), generator=gen, device=args.device))
            v = pick_candidates(E, codes[:, p, 0], 0, "local", args.knn, gen)
            cA = codes.clone()
            cA[:, p, 0] = v
            cB = requantize_at(E, cA, h0, p, 0)
            dA = latent_from_codes(E, cA)[:, p] - h0[:, p]
            dB = latent_from_codes(E, cB)[:, p] - h0[:, p]
            for d, acc in ((dA, eA), (dB, eB)):
                co = (d @ PC.T) ** 2                       # (Bn, Dz)
                tot = co.sum(-1, keepdim=True).clamp_min(1e-30)
                acc.append((co / tot).cpu().numpy())
        eA, eB = np.concatenate(eA), np.concatenate(eB)

        print(f"{'полоса':>14}{'доля энергии δ_A':>20}{'δ_B':>10}{'A/B':>8}")
        for lo, hi in BANDS:
            fA = float(np.median(eA[:, lo:hi].sum(1)))
            fB = float(np.median(eB[:, lo:hi].sum(1)))
            print(f"{f'ГК {lo+1}-{hi}':>14}{fA:>20.3f}{fB:>10.3f}"
                  f"{fA/max(fB,1e-9):>8.2f}")

    print("""
КАК ЧИТАТЬ.

B. Если чувствительность падает от первых полос к последним, декодер
   анизотропен по главным компонентам словаря, и величина падения — мера
   анизотропии. Изотропный столбец должен лечь ближе к последним полосам:
   случайное направление в 512 измерениях почти целиком попадает в хвост
   спектра.

C. Если у δ_B доля энергии в ЧУВСТВИТЕЛЬНЫХ полосах меньше, чем у δ_A, —
   механизм объяснён: жадная переквантизация снимает с смещения ту часть,
   которая лежит вдоль направлений, к которым декодер чувствителен.
   Если распределения совпадают, объяснение неверно, а факт 1.27 остаётся
   необъяснённым — так и писать.""")


if __name__ == "__main__":
    main()
