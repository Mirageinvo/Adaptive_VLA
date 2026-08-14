"""K-2a: есть ли ЧТО распределять — оракульная глубина на токенизаторе.

ЗАМЫСЕЛ K-2. Прежде чем строить планировщик вычислений, надо выяснить, даёт
ли вообще переменный бюджет выигрыш против фиксированного и против случайного
той же средней стоимости. Оракул подсматривает ответ задним числом, то есть
задаёт ВЕРХНЮЮ ГРАНИЦУ любой адаптивности: если он не обгоняет случайную
раскладку, ни один обучаемый планировщик не обгонит.

ПОЧЕМУ ЗДЕСЬ, А НЕ НА ПОЛИТИКЕ. Полный K-2 требует итеративной модели и
роллаутов в симуляторе. Но у RVQ уже есть своя ось бюджета — число уровней, и
декодирование по префиксу обучено (quantizer_dropout=0.25, rvq.py:301). Значит
самый дешёвый вариант вопроса задаётся на одном токенизаторе.

ЧЕСТНАЯ ОГОВОРКА, ОБЯЗАТЕЛЬНАЯ ПРИ ЧТЕНИИ. Здесь меряется ОШИБКА
РЕКОНСТРУКЦИИ, а не успех задачи. Связь между ними слабая: в предыдущей
работе на той же линейке корреляция реконструкционного зазора с ценностью в
замкнутом цикле составила 0.08. Поэтому:

    отрицательный результат здесь БЛИЗОК К РЕШАЮЩЕМУ: если даже при полном
        знании чанка и прямом доступе к ошибке распределять нечего, то в
        замкнутом цикле, где сигнал слабее, тем более нечего;
    положительный результат — лишь НЕОБХОДИМОЕ условие, не достаточное, и
        требует подтверждения на успехе задачи.

ЧТО МЕРЯЕМ.
  A. Кривая «глубина против ошибки» — есть ли у неё вообще наклон.
  B. Оракульная глубина K*(чанк) при заданном допуске, её разброс.
  C. ГЛАВНОЕ: оракул против ФИКСИРОВАННОЙ и против СЛУЧАЙНОЙ раскладки той же
     средней глубины. Случайная — обязательный контроль: без неё «выигрыш»
     оракула может оказаться просто следствием меньшего среднего бюджета.
  D. Предсказуема ли K* из простых признаков чанка. Планировщик обязан её
     угадывать, не подсматривая; если она непредсказуема, адаптивность не
     обучится даже при наличии разброса.

ПРАВИЛО ЧТЕНИЯ, зафиксировано до запуска:
  оракул обгоняет случайную раскладку той же средней глубины заметно
      (скажем, ошибка ниже на 15% и более) И K* предсказуема лучше
      большинства -> распределять есть что, идти к K-2 на политике;
  оракул совпадает со случайной -> ось бюджета не несёт полезной
      state-зависимости, адаптивность закрывается здесь;
  K* почти постоянна -> распределять нечего по построению.

Запуск:
    python3 experiments/k2a_oracle_depth.py --zarr <путь>/libero10_N500.zarr
"""

import argparse
import os
import sys

import einops
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1_residual_cost import latent_from_codes, load_codec, projected_codebooks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--model", default="ZibinDong/ActionCodec-Base-RVQft")
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--n-chunks", type=int, default=2048)
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--metric", choices=["max", "rms"], default="max",
                    help="норма ошибки по непрерывным каналам")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import zarr

    model = load_codec(os.path.abspath(args.root), args.model, args.device)
    L, P = model.num_quantizers, model.n_tokens_per_quantizer
    name = list(model.config.embodiment_config.keys())[args.embodiment]
    cfg = model.config.embodiment_config[name]
    T, D_act = int(cfg["freq"] * cfg["duration"]), cfg["action_dim"]
    print(f"quantizer_dropout при обучении: "
          f"{getattr(model.config, 'vq_quantizer_dropout', '?')}")

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
    N, scale = len(a), float(a.max() - a.min())
    print(f"чанков {N}, размах {scale:.2f}\n")
    err = err_rms = grip = None

    E = projected_codebooks(model, args.device)

    with torch.no_grad():
        flat = torch.as_tensor(np.asarray(model.encode(a, embodiment_ids=args.embodiment)),
                               device=args.device, dtype=torch.long)
        codes = einops.rearrange(flat, "b (n m) -> b m n", m=P)

        # Ошибка при декодировании из первых k уровней.
        #
        # КАНАЛ ЗАХВАТА СЧИТАЕТСЯ ОТДЕЛЬНО. Он бинарный, и в момент
        # переключения декодер промахивается примерно на единицу — при размахе
        # 1.94 это ~0.51 размаха. Максимум-норма по всему чанку тогда упирается
        # в него у всех чанков с переключением (а их больше половины), кривая
        # «глубина -> ошибка» выглядит плоской, и меряется наличие
        # переключения, а не потребность в глубине. Проверено: при такой норме
        # медиана держится на 0.5145 при всех k, тогда как 25-й процентиль
        # (чанки без переключения) честно падает 0.0664 -> 0.0425.
        cont = slice(0, D_act - 1)
        err = np.zeros((N, L))          # непрерывные каналы, max-норма
        err_rms = np.zeros((N, L))      # они же, среднеквадратичная
        grip = np.zeros((N, L))         # доля неверных шагов захвата
        for k in range(1, L + 1):
            h = sum(E[j][codes[:, :, j]] for j in range(k))
            rec = model._decode(h, args.embodiment, None)[0][..., :D_act]
            d = (rec - a).abs()
            err[:, k - 1] = (d[..., cont].flatten(1).amax(-1) / scale).cpu().numpy()
            err_rms[:, k - 1] = (d[..., cont].flatten(1).pow(2).mean(-1).sqrt()
                                 / scale).cpu().numpy()
            grip[:, k - 1] = ((rec[..., -1] > 0.5) != (a[..., -1] > 0.5)
                              ).float().mean(-1).cpu().numpy()

        print("захват отдельно, доля неверных шагов по глубине: "
              + " ".join(f"{grip[:, k].mean():.3f}" for k in range(L)))
        print("непрерывные каналы, медиана по элементам (сверка с K-1): "
              f"{float(np.median(err_rms[:, -1])):.4f} rms\n")

    if args.metric == "rms":
        err = err_rms
    print("=" * 74)
    print("A. КРИВАЯ ГЛУБИНА -> ОШИБКА (только непрерывные каналы)")
    print("=" * 74)
    print(f"{'уровней':>9}{'медиана':>11}{'25%':>10}{'75%':>10}{'90%':>10}"
          f"{'прирост':>11}")
    prev = None
    for k in range(L):
        q = np.percentile(err[:, k], [25, 50, 75, 90])
        inc = "" if prev is None else f"{prev / q[1]:.2f}x"
        print(f"{k+1:>9}{q[1]:>11.4f}{q[0]:>10.4f}{q[2]:>10.4f}{q[3]:>10.4f}"
              f"{inc:>11}")
        prev = q[1]
    if prev is not None and np.median(err[:, 0]) / np.median(err[:, -1]) < 1.2:
        print("\nВНИМАНИЕ: кривая почти плоская — распределять нечего "
              "независимо от\nвсего остального.")

    # ---------- B. оракульная глубина ----------
    print("\n" + "=" * 74)
    print("B. ОРАКУЛЬНАЯ ГЛУБИНА ПРИ РАЗНЫХ ДОПУСКАХ")
    print("=" * 74)
    print("K*(чанк) = наименьшее k, при котором ошибка не превышает допуск.\n")
    print(f"{'допуск':>9}{'сред. K*':>11}{'доля K*=1':>12}{'=2':>8}{'=3':>8}"
          f"{'разброс':>10}")
    tols = np.percentile(err[:, -1], [50, 70, 85, 95])
    rng = np.random.default_rng(1)
    scenarios = []
    for tol in tols:
        Ks = np.full(N, L)
        for k in range(L):
            hit = (err[:, k] <= tol) & (Ks == L)
            Ks[hit] = k + 1
        Ks = np.minimum(Ks, L)
        share = [float((Ks == k + 1).mean()) for k in range(L)]
        print(f"{tol:>9.4f}{Ks.mean():>11.2f}" + "".join(f"{s:>12.0%}" if i == 0
              else f"{s:>8.0%}" for i, s in enumerate(share))
              + f"{Ks.std():>10.2f}")
        scenarios.append((tol, Ks))

    # ---------- C. оракул против фиксированного и случайного ----------
    print("\n" + "=" * 74)
    print("C. ОРАКУЛ ПРОТИВ ФИКСИРОВАННОГО И СЛУЧАЙНОГО ТОЙ ЖЕ СРЕДНЕЙ ГЛУБИНЫ")
    print("=" * 74)

    def mean_err(Ks):
        return float(np.mean(err[np.arange(N), Ks - 1]))

    print(f"{'сред. K':>9}{'оракул':>11}{'случайно':>11}{'смесь фикс.':>13}"
          f"{'оракул/случ.':>14}{'95% ДИ':>18}")
    for tol, Ks in scenarios:
        dbar = Ks.mean()
        e_or = mean_err(Ks)
        # СЛУЧАЙНАЯ РАСКЛАДКА: та же гистограмма глубин, но без связи с чанком
        e_rnd = np.mean([mean_err(rng.permutation(Ks)) for _ in range(64)])
        # СМЕСЬ ФИКСИРОВАННЫХ: два соседних целых уровня в пропорции, дающей
        # ту же среднюю глубину. Это честная «неадаптивная» точка на кривой.
        lo = int(np.floor(dbar))
        w = dbar - lo
        e_fix = ((1 - w) * np.mean(err[:, lo - 1])
                 + w * np.mean(err[:, min(lo, L - 1)]))
        boots = []
        for _ in range(args.n_boot):
            s = rng.integers(0, N, N)
            eo = float(np.mean(err[s, Ks[s] - 1]))
            er = float(np.mean(err[s, rng.permutation(Ks)[s] - 1]))
            boots.append(eo / max(er, 1e-12))
        ci = np.percentile(boots, [2.5, 97.5])
        print(f"{dbar:>9.2f}{e_or:>11.4f}{e_rnd:>11.4f}{e_fix:>13.4f}"
              f"{e_or/max(e_rnd,1e-12):>14.2f}"
              f"{f'[{ci[0]:.2f}, {ci[1]:.2f}]':>18}")

    print("""
оракул/случ. — во сколько раз ошибка оракула МЕНЬШЕ случайной той же средней
глубины. Меньше 1 — оракул выигрывает; 1.00 — распределение по чанкам не
несёт пользы, и адаптивность закрывается. Смесь фиксированных нужна отдельно:
если она не хуже оракула, то и переменный бюджет не нужен — достаточно
подобрать одну глубину.""")

    # ---------- D. предсказуемость ----------
    print("\n" + "=" * 74)
    print("D. ПРЕДСКАЗУЕМА ЛИ K* ИЗ ПРИЗНАКОВ ЧАНКА")
    print("=" * 74)
    Aa = A[idx]
    feats = {
        "размах движения": np.abs(Aa).max(axis=(1, 2)),
        "скорость": np.abs(np.diff(Aa, axis=1)).mean(axis=(1, 2)),
        "рывок": np.abs(np.diff(Aa, n=2, axis=1)).max(axis=(1, 2)),
        "переключений захвата": (np.abs(np.diff(Aa[:, :, -1], axis=1)) > 0.5).sum(1),
    }
    tol, Ks = scenarios[1]
    print(f"допуск {tol:.4f}, средняя K* = {Ks.mean():.2f}\n")
    print(f"{'признак':>24}{'ранг. связь с K*':>20}")
    for nm, f in feats.items():
        rx = np.argsort(np.argsort(f)).astype(float)
        ry = np.argsort(np.argsort(Ks)).astype(float)
        rx, ry = rx - rx.mean(), ry - ry.mean()
        rho = float(rx @ ry / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))
        print(f"{nm:>24}{rho:>20.3f}")
    maj = float(max((Ks == k + 1).mean() for k in range(L)))
    print(f"\nдоля большинства (что даёт предсказание константой): {maj:.0%}")
    print("Планировщик обязан бить эту величину, не подсматривая ответ.\n"
          "Слабая связь у всех признаков означает, что даже при наличии\n"
          "разброса K* адаптивность не обучится.")


if __name__ == "__main__":
    main()
