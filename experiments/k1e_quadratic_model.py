"""K-1e: объясняет ли фиксированная анизотропная метрика отношение 1.27.

ПОВОД. K-1d показал, что отклик декодера АДДИТИВЕН по полосам спектра:
предсказание изотропного отклика из полосовых совпало с измеренным в обеих
нормах (0.0112 против 0.0111 в max, 0.0024 против 0.0025 в rms).

Раз аддитивность держится, отклик описывается квадратичной формой

    отклик(δ)^2  ≈  Σ_k s_k^2 · c_k^2,

где c_k — координаты смещения по главным компонентам словаря, s_k —
чувствительность по k-й компоненте. А если так, то отношение 1.27 из K-1c
ОБЯЗАНО вычисляться из координат δ_A и δ_B. Это и проверяем.

  1. меряем чувствительность по УЗКИМ полосам (логарифмическая сетка);
  2. по фактическим координатам δ_A и δ_B предсказываем их отклики;
  3. сравниваем предсказанное отношение с измеренным на тех же смещениях.

ПРАВИЛО ЧТЕНИЯ, зафиксировано до запуска:
  предсказанное отношение близко к измеренному (в пределах ~0.05) —
      механизм объяснён полностью: всё дело в том, по каким направлениям
      спектра распределено смещение, и мы можем назвать эти направления;
  предсказанное около 1.0, а измеренное около 1.27 — фиксированной
      анизотропной метрики НЕ хватает. Значит отклик зависит от точки h0, и
      дальше только локальный якобиан. Записать как неустановленный механизм
      и не продолжать.

Проверка однородности встроена: отклик должен расти линейно по норме
смещения, иначе квадратичная модель неприменима с самого начала.

Запуск:
    python3 experiments/k1e_quadratic_model.py --zarr <путь>/libero10_N500.zarr
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

EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--model", default="ZibinDong/ActionCodec-Base-RVQft")
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--n-chunks", type=int, default=96)
    ap.add_argument("--n-trials", type=int, default=32)
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.015)
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
    Bn, scale = len(a), float(a.max() - a.min())

    E = projected_codebooks(model, args.device)
    Dz = E.shape[-1]
    gen = torch.Generator(device=args.device).manual_seed(1)

    cb = E.reshape(-1, Dz).double()
    PC = torch.linalg.svd(cb - cb.mean(0, keepdim=True),
                          full_matrices=False)[2].float()
    print(f"чанков {Bn}, латента {Dz}-мерная, tau={args.tau}\n")

    with torch.no_grad():
        flat = torch.as_tensor(np.asarray(model.encode(a, embodiment_ids=args.embodiment)),
                               device=args.device, dtype=torch.long)
        codes = einops.rearrange(flat, "b (n m) -> b m n", m=P)
        h0 = latent_from_codes(E, codes)
        base = model._decode(h0, args.embodiment, None)[0][..., :D_act]

        def resp(h):
            dv = (model._decode(h, args.embodiment, None)[0][..., :D_act]
                  - base).abs().flatten(1)
            return dv.amax(-1) / scale

        def probe(d, tau):
            p = int(torch.randint(P, (1,), generator=gen, device=args.device))
            n = d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            h = h0.clone()
            h[:, p] = h[:, p] + d / n * (tau * np.sqrt(Dz))
            return resp(h)

        # ---------- однородность ----------
        print("=" * 70)
        print("0. ОДНОРОДНОСТЬ: отклик должен расти линейно по норме")
        print("=" * 70)
        prev = None
        for tau in (0.005, 0.010, 0.020, 0.040):
            acc = [probe(torch.randn(Bn, Dz, generator=gen, device=args.device),
                         tau).cpu().numpy() for _ in range(args.n_trials // 2)]
            m = float(np.median(np.concatenate(acc)))
            r = "" if prev is None else f"  x{m/prev:.2f} при удвоении нормы"
            print(f"  tau={tau:.3f}: отклик {m:.5f}{r}")
            prev = m
        print("  (ожидаем множители около 2.00; иначе квадратичная модель\n"
              "   неприменима и дальше читать нельзя)")

        # ---------- чувствительность по узким полосам ----------
        print("\n" + "=" * 70)
        print("1. ЧУВСТВИТЕЛЬНОСТЬ ПО УЗКИМ ПОЛОСАМ")
        print("=" * 70)
        s = np.zeros(len(EDGES) - 1)
        print(f"{'полоса ГК':>14}{'ширина':>8}{'отклик':>10}{'на ед. энергии':>16}")
        for b in range(len(EDGES) - 1):
            lo, hi = EDGES[b], EDGES[b + 1]
            acc = []
            for _ in range(args.n_trials):
                c = torch.randn(Bn, hi - lo, generator=gen, device=args.device)
                acc.append(probe(c @ PC[lo:hi], args.tau).cpu().numpy())
            s[b] = float(np.median(np.concatenate(acc)))
            print(f"{f'{lo+1}-{hi}':>14}{hi-lo:>8}{s[b]:>10.5f}"
                  f"{s[b]/args.tau:>16.3f}")

        # ---------- предсказание против измерения ----------
        print("\n" + "=" * 70)
        print("2. ПРЕДСКАЗАНИЕ ПРОТИВ ИЗМЕРЕНИЯ НА НАСТОЯЩИХ СМЕЩЕНИЯХ")
        print("=" * 70)
        pA, pB, mA, mB = [], [], [], []
        for _ in range(args.n_trials * 2):
            p = int(torch.randint(P, (1,), generator=gen, device=args.device))
            v = pick_candidates(E, codes[:, p, 0], 0, "local", args.knn, gen)
            cA = codes.clone()
            cA[:, p, 0] = v
            cB = requantize_at(E, cA, h0, p, 0)
            dA = latent_from_codes(E, cA)[:, p] - h0[:, p]
            dB = latent_from_codes(E, cB)[:, p] - h0[:, p]
            for d, pred, meas in ((dA, pA, mA), (dB, pB, mB)):
                co = (d @ PC.T) ** 2
                frac = co / co.sum(-1, keepdim=True).clamp_min(1e-30)
                e = torch.stack([frac[:, EDGES[b]:EDGES[b + 1]].sum(-1)
                                 for b in range(len(EDGES) - 1)], -1)  # (Bn, nb)
                sv = torch.as_tensor(s, device=args.device, dtype=torch.float32)
                pred.append(((e * sv ** 2).sum(-1)).sqrt().cpu().numpy())
                # измерение — при той же норме tau, как и калибровка
                n = d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                h = h0.clone()
                h[:, p] = h[:, p] + d / n * (args.tau * np.sqrt(Dz))
                meas.append(resp(h).cpu().numpy())

        pA, pB = np.concatenate(pA), np.concatenate(pB)
        mA, mB = np.concatenate(mA), np.concatenate(mB)
        r_pred = float(np.median(pA) / np.median(pB))
        r_meas = float(np.median(mA) / np.median(mB))
        print(f"{'':>14}{'A':>12}{'B':>12}{'A/B':>9}")
        print(f"{'предсказано':>14}{np.median(pA):>12.5f}{np.median(pB):>12.5f}"
              f"{r_pred:>9.3f}")
        print(f"{'измерено':>14}{np.median(mA):>12.5f}{np.median(mB):>12.5f}"
              f"{r_meas:>9.3f}")
        print(f"\nрасхождение отношений: {abs(r_pred - r_meas):.3f}")

        if abs(r_pred - r_meas) < 0.05:
            print("""
ВЫВОД: механизм объяснён. Отклик задаётся фиксированной анизотропной метрикой
в латентном пространстве, и разница между A и B целиком объясняется тем, по
каким направлениям спектра распределены их смещения. Направления известны —
см. таблицу чувствительностей выше.""")
        elif abs(r_pred - 1.0) < 0.1:
            print("""
ВЫВОД: фиксированной анизотропной метрики НЕ ХВАТАЕТ. Координаты смещений
предсказывают отношение около 1.0, а измеряется заметно больше. Значит отклик
зависит от точки h0, а не только от направления смещения, и объяснение
требует локального якобиана декодера. Записать как НЕУСТАНОВЛЕННЫЙ механизм;
факт (отношение из K-1c) при этом остаётся в силе.""")
        else:
            print("""
ВЫВОД: предсказание и измерение не сошлись, но и не разошлись начисто.
Квадратичная модель улавливает часть эффекта. Указать в записи обе величины и
не заявлять объяснение как полное.""")


if __name__ == "__main__":
    main()
