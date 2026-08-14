"""C-3: та ли это геометрия. Совпадает ли близость по решётке FSQ с близостью
в пространстве действий.

ПОЧЕМУ ЭТО РЕШАЮЩЕЕ. Голова из head_geometric.py сглаживает поправку по
решётке: коды, близкие в [-1,1]^4, получают близкие вероятности, и шаг
градиента двигает их заодно. Замер показал, что механизм работает — масса
уходит вдвое дальше, чем при пропорциональном перераспределении.

Но полезно это ровно в той мере, в какой близость ПО РЕШЁТКЕ означает
близость ПО ДЕЙСТВИЮ. Решётка — это вход декодера, а декодер нелинеен и
обучен. Ничто не обязывает его сохранять метрику. Если сохраняет плохо, то
голова обобщает вдоль неправильной оси: наказывает коды, которые кодируют
совсем другое движение, и не трогает те, что кодируют почти то же самое.
Тогда конструкция бессмысленна, как бы хорошо ни работал сам механизм.

ЧТО СЧИТАЕМ, для каждого регистра:
  A. ранговая связь между расстоянием по решётке и сдвигом действия — по
     ВСЕМ 1000 кодам и ОТДЕЛЬНО по ближней окрестности. Глобальная связь
     может быть высокой при полностью перепутанном локальном порядке, а
     голова работает именно локально;
  B. во сколько раз соседи по решётке ближе по действию, чем случайные коды
     — это и есть ответ на вопрос «покупает ли близость по решётке близость
     по действию» в понятных единицах;
  C. доля кодов, которые близки по решётке, но далеки по действию (ложные
     соседи) — именно их голова будет наказывать зря.

ЗАРАНЕЕ О ЧТЕНИИ. Ключевая величина — B на ближней окрестности. Если соседи
по решётке ближе по действию хотя бы вдвое-втрое, геометрия та. Если около
единицы, решётка про действия ничего не знает, и направление закрывается
здесь, до всякого обучения.

Запуск:
    python3 experiments/c3_grid_vs_action.py \
        --oat  /workspace/oat/oat \
        --ckpt /workspace/oat/oat/my_models/tokenizer_ep-0950_mse-0.002.ckpt \
        --zarr <путь>/libero10_N500.zarr
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oat0_resolution import load_tokenizer  # noqa: E402


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float(rx @ ry / (rx.norm() * ry.norm() + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oat", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--n-chunks", type=int, default=48)
    ap.add_argument("--near", type=int, default=20,
                    help="размер ближней окрестности по решётке")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import zarr

    tok, cfg = load_tokenizer(os.path.abspath(args.oat), os.path.abspath(args.ckpt),
                              args.device)
    Q = tok.quantizer
    K, R = Q.codebook_size, tok.latent_horizon
    T, D = cfg.horizon, cfg.action_dim
    print(f"\nрегистров {R}, словарь {K}, чанк {T}x{D}\n")

    z = zarr.open(os.path.abspath(args.zarr), mode="r")
    acts = np.asarray(z["data"]["action"])
    ends = np.asarray(z["meta"]["episode_ends"])
    out, start = [], 0
    for e in ends:
        ep = acts[start:e]
        for i in range(len(ep) // T):
            out.append(ep[i * T:(i + 1) * T])
        start = e
    A = np.stack(out).astype(np.float32)
    idx = np.random.default_rng(0).choice(len(A), size=min(args.n_chunks, len(A)),
                                          replace=False)
    a = torch.from_numpy(A[idx]).to(args.device)
    scale = float(a.max() - a.min())
    print(f"чанков {len(a)}, размах действий {scale:.2f}")

    with torch.no_grad():
        lat, tokens = tok.encode(a)
        base = tok.decode(lat)
        rec = ((base - a).abs().median() / scale).item()
        print(f"ошибка реконструкции: {rec:.4f} размаха")
        assert rec < 0.05, "данные вне распределения токенизатора"

        emb = Q.indices_to_embedding(torch.arange(K, device=args.device))   # (K, d)
        gdist_all = torch.cdist(emb, emb)                                   # (K, K)

        print("\n" + "=" * 86)
        print("РЕШЁТКА FSQ ПРОТИВ ПРОСТРАНСТВА ДЕЙСТВИЙ")
        print("=" * 86)
        print(f"{'рег.':>5}{'ранг. связь':>14}{'она же вблизи':>16}"
              f"{'соседи ближе в':>17}{'ложных соседей':>17}")

        rows = []
        for r in range(R):
            rho_all, rho_near, gain, false_nb = [], [], [], []
            for i in range(len(a)):
                alt = lat[i:i + 1].repeat(K, 1, 1)
                alt[:, r, :] = emb
                d_ = tok.decode(alt)
                dev = (d_ - base[i:i + 1]).abs().flatten(1).amax(1) / scale  # (K,)

                c0 = int(tokens[i, r])
                g = gdist_all[c0].clone()
                keep = torch.arange(K, device=args.device) != c0
                gv, dv = g[keep], dev[keep]

                rho_all.append(spearman(gv, dv))

                near = gv.argsort()[:args.near]
                rho_near.append(spearman(gv[near], dv[near]))

                # B: во сколько раз ближние по решётке ближе по действию
                gain.append(float(dv.median() / dv[near].median().clamp_min(1e-9)))

                # C: ложные соседи — близкие по решётке, но по действию
                # дальше медианы по всем кодам
                false_nb.append(float((dv[near] > dv.median()).float().mean()))

            row = (float(np.median(rho_all)), float(np.median(rho_near)),
                   float(np.median(gain)), float(np.median(false_nb)))
            rows.append(row)
            print(f"{r:>5}{row[0]:>14.3f}{row[1]:>16.3f}{row[2]:>16.2f}x"
                  f"{row[3]:>16.0%}")

        m = np.array(rows)
        print(f"{'медиана':>5}{np.median(m[:, 0]):>14.3f}{np.median(m[:, 1]):>16.3f}"
              f"{np.median(m[:, 2]):>16.2f}x{np.median(m[:, 3]):>16.0%}")

    print(f"""
ранг. связь     — связь расстояния по решётке со сдвигом действия по всем
                  {K} кодам. Общая картина.
она же вблизи   — то же, но только внутри {args.near} ближайших по решётке.
                  Голова работает ИМЕННО ЗДЕСЬ, поэтому эта колонка важнее
                  первой: глобальная связь может быть высокой при полностью
                  перепутанном ближнем порядке.
соседи ближе в  — во сколько раз медианный сдвиг действия у ближних по
                  решётке меньше, чем у произвольного кода. ГЛАВНОЕ ЧИСЛО.
ложных соседей  — доля ближних по решётке, которые по действию дальше
                  медианы. Их голова наказывала бы напрасно.

ЧТЕНИЕ, зафиксировано до запуска:
  «соседи ближе» заметно больше 1 (хотя бы вдвое) -> геометрия та, решётка
      несёт информацию о действиях, сглаживание по ней осмысленно;
  около 1 -> решётка про действия не знает, голова обобщает вдоль
      неправильной оси, и направление закрывается здесь.
Доля ложных соседей при этом показывает цену: даже при верной в среднем
геометрии часть окрестности будет наказана зря, и число компонент M нужно
выбирать так, чтобы моды не сливались.""")


if __name__ == "__main__":
    main()
