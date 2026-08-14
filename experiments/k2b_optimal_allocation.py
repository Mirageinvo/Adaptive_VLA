"""K-2b: ТОЧНОЕ оптимальное распределение бюджета глубины RVQ.

ПОЧЕМУ ЗАНОВО. В K-2a оракул решал ПОРОГОВУЮ задачу: «наименьшее k, при
котором чанк стал достаточно хорош». А заявлялся он как верхняя граница
распределения бюджета, то есть как решение совсем другой задачи:

    min  (1/N) Σ e_i(K_i)   при   (1/N) Σ K_i ≤ B,   K_i ∈ {1..L}.

Пороговое правило её не решает и вообще не обязано быть оптимальным:
  - чанк, едва перешедший порог, останавливается, хотя следующий уровень дал
    бы крупный абсолютный выигрыш;
  - другому уровень выдаётся лишь потому, что порог не пройден, при мизерной
    предельной отдаче;
  - если ни одна глубина порог не прошла, назначается L, даже когда минимум
    ошибки достигался раньше.
Поэтому полученные там 5-6% — НИЖНЯЯ оценка выигрыша оракула, а не верхняя, и
вывод «распределять нечего» на ней держаться не мог.

КАК РЕШАЕТСЯ ТОЧНО. Лагранжева релаксация: при штрафе λ за уровень
    K_i(λ) = argmin_k [ e_i(k) + λ·k ]
даёт оптимальное назначение для того среднего бюджета, который при этом λ
получается. Пробегая λ, получаем всю нижнюю выпуклую оболочку компромисса
«бюджет против ошибки». Зазор двойственности не превышает вклада одного чанка,
то есть при N=2048 пренебрежим. Это стандартный и точный приём для таких
задач с разделяющейся структурой.

ЧТО ЕЩЁ ИСПРАВЛЕНО ПРОТИВ K-2a:
  1. перестановка глубин для matched-random делается ВНУТРИ бутстрап-выборки,
     а не по исходной популяции;
  2. бутстрап КЛАСТЕРНЫЙ ПО ЭПИЗОДАМ: чанки одной демонстрации коррелированы,
     и бутстрап по отдельным чанкам занижает интервал;
  3. ранговая связь считается со СРЕДНИМИ РАНГАМИ: у K* всего L значений,
     ties огромны, а argsort(argsort()) раздаёт им произвольные разные ранги;
  4. считаются обе нормы — max и rms;
  5. пороговый оракул из K-2a оставлен как отдельная строка, чтобы видеть,
     насколько он был неоптимален.

ОТДЕЛЬНО ПРО ПРИЗНАКИ. Размах, скорость и рывок вычисляются по ИСТИННОМУ
будущему чанку, которого на инференсе нет. Это привилегированная информация,
и предсказуемость по ней — верхняя оценка для настоящего планировщика, а не
достижимая величина. Помечено в выводе.

ПРАВИЛО ЧТЕНИЯ, зафиксировано до запуска:
  оптимальный оракул обгоняет matched-random той же средней глубины на 15% и
      более -> распределять есть что, идти к проверке на успехе задачи;
  меньше 15% -> при полном знании ошибок и точном решении задачи выигрыш мал,
      и ось глубины RVQ закрывается.

Запуск:
    python3 experiments/k2b_optimal_allocation.py --zarr <путь>/libero10_N500.zarr
"""

import argparse
import os
import sys

import einops
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1_residual_cost import load_codec, projected_codebooks  # noqa: E402


def avg_rank(x: np.ndarray) -> np.ndarray:
    """Средние ранги: при ties argsort(argsort()) раздаёт произвольные разные."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = avg_rank(a), avg_rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d > 0 else float("nan")


def optimal_assign(err: np.ndarray, lam: float) -> np.ndarray:
    """argmin_k [e_i(k) + lam*(k+1)] — оптимум при штрафе lam за уровень."""
    k = np.arange(1, err.shape[1] + 1)
    return (err + lam * k[None, :]).argmin(1) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--model", default="ZibinDong/ActionCodec-Base-RVQft")
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--n-chunks", type=int, default=4096)
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import zarr

    model = load_codec(os.path.abspath(args.root), args.model, args.device)
    L, P = model.num_quantizers, model.n_tokens_per_quantizer
    name = list(model.config.embodiment_config.keys())[args.embodiment]
    cfg = model.config.embodiment_config[name]
    T, D_act = int(cfg["freq"] * cfg["duration"]), cfg["action_dim"]

    z = zarr.open(os.path.abspath(args.zarr), mode="r")
    acts = np.asarray(z["data"]["action"])
    ends = np.asarray(z["meta"]["episode_ends"])
    chunks, epi, start = [], [], 0
    for e_id, e in enumerate(ends):
        ep = acts[start:e]
        for i in range(len(ep) // T):
            chunks.append(ep[i * T:(i + 1) * T])
            epi.append(e_id)                    # для кластерного бутстрапа
        start = e
    A = np.stack(chunks).astype(np.float32)
    epi = np.array(epi)
    if A[:, :, -1].min() < -0.5:
        A[:, :, -1] = (1.0 - A[:, :, -1]) / 2.0
    idx = np.random.default_rng(0).choice(len(A), size=min(args.n_chunks, len(A)),
                                          replace=False)
    A, epi = A[idx], epi[idx]
    a = torch.from_numpy(A).to(args.device)
    N, scale = len(a), float(a.max() - a.min())
    print(f"чанков {N}, эпизодов {len(np.unique(epi))}, размах {scale:.2f}\n")

    E = projected_codebooks(model, args.device)
    with torch.no_grad():
        flat = torch.as_tensor(np.asarray(model.encode(a, embodiment_ids=args.embodiment)),
                               device=args.device, dtype=torch.long)
        codes = einops.rearrange(flat, "b (n m) -> b m n", m=P)
        cont = slice(0, D_act - 1)          # захват отдельно: он бинарный и
        e_max = np.zeros((N, L))            # в max-норме забивает всё остальное
        e_rms = np.zeros((N, L))
        grip = np.zeros((N, L))
        for k in range(1, L + 1):
            h = sum(E[j][codes[:, :, j]] for j in range(k))
            rec = model._decode(h, args.embodiment, None)[0][..., :D_act]
            d = (rec - a).abs()
            e_max[:, k - 1] = (d[..., cont].flatten(1).amax(-1) / scale).cpu().numpy()
            e_rms[:, k - 1] = (d[..., cont].flatten(1).pow(2).mean(-1).sqrt()
                               / scale).cpu().numpy()
            grip[:, k - 1] = ((rec[..., -1] > 0.5) != (a[..., -1] > 0.5)
                              ).float().mean(-1).cpu().numpy()

    print("захват (доля неверных шагов) по глубине: "
          + " ".join(f"{grip[:, k].mean():.3f}" for k in range(L)) + "\n")

    for metric, err in (("max", e_max), ("rms", e_rms)):
        print("=" * 78)
        print(f"НОРМА {metric.upper()}: средняя ошибка по глубине "
              + " ".join(f"{err[:, k].mean():.4f}" for k in range(L)))
        print("=" * 78)

        # ---- пороговый оракул из K-2a, для сравнения ----
        tol = float(np.percentile(err[:, -1], 70))
        Kthr = np.full(N, L)
        for k in range(L):
            Kthr[(err[:, k] <= tol) & (Kthr == L)] = k + 1

        rng = np.random.default_rng(1)
        eps = np.unique(epi)
        by_ep = {e: np.where(epi == e)[0] for e in eps}

        def matched_random(err_s, K_s, reps=32):
            """Перестановка глубин ВНУТРИ выборки: та же гистограмма, но связь
            с чанком разорвана."""
            return float(np.mean([
                np.mean(err_s[np.arange(len(K_s)), rng.permutation(K_s) - 1])
                for _ in range(reps)]))

        def cluster_boot(K):
            """Бутстрап ПО ЭПИЗОДАМ: чанки одной демонстрации коррелированы."""
            out = []
            for _ in range(args.n_boot):
                pick = np.concatenate([by_ep[e] for e in
                                       rng.choice(eps, len(eps), replace=True)])
                es, Ks = err[pick], K[pick]
                eo = float(np.mean(es[np.arange(len(Ks)), Ks - 1]))
                out.append(eo / max(matched_random(es, Ks, reps=4), 1e-12))
            return np.percentile(out, [2.5, 97.5])

        print(f"{'вариант':>22}{'сред. K':>9}{'ошибка':>10}{'случайно':>10}"
              f"{'смесь фикс.':>13}{'опт/случ.':>11}{'95% ДИ':>16}")

        rows = [("пороговый (K-2a)", Kthr)]
        for lam in (0.0005, 0.001, 0.002, 0.004, 0.008, 0.016):
            rows.append((f"оптимальный λ={lam:g}", optimal_assign(err, lam)))

        for tag, K in rows:
            dbar = K.mean()
            eo = float(np.mean(err[np.arange(N), K - 1]))
            er = matched_random(err, K)
            lo = int(np.floor(dbar))
            w = dbar - lo
            ef = ((1 - w) * err[:, lo - 1].mean()
                  + w * err[:, min(lo, L - 1)].mean())
            ci = cluster_boot(K)
            print(f"{tag:>22}{dbar:>9.2f}{eo:>10.4f}{er:>10.4f}{ef:>13.4f}"
                  f"{eo/max(er,1e-12):>11.3f}"
                  f"{f'[{ci[0]:.2f}, {ci[1]:.2f}]':>16}")

    # ---------- предсказуемость ----------
    print("\n" + "=" * 78)
    print("ПРЕДСКАЗУЕМОСТЬ ОПТИМАЛЬНОЙ ГЛУБИНЫ (норма max)")
    print("=" * 78)
    K = optimal_assign(e_max, 0.002)
    print(f"средняя K = {K.mean():.2f}, доля большинства = "
          f"{max((K == k + 1).mean() for k in range(L)):.0%}\n")
    feats = {
        "размах движения": np.abs(A).max(axis=(1, 2)),
        "скорость": np.abs(np.diff(A, axis=1)).mean(axis=(1, 2)),
        "рывок": np.abs(np.diff(A, n=2, axis=1)).max(axis=(1, 2)),
        "переключений захвата": (np.abs(np.diff(A[:, :, -1], axis=1)) > 0.5).sum(1),
        "ошибка при k=1": e_max[:, 0],
    }
    print(f"{'признак':>24}{'ранг. связь (средние ранги)':>30}")
    for nm, f in feats.items():
        print(f"{nm:>24}{spearman(f, K.astype(float)):>30.3f}")
    print("""
ВНИМАНИЕ. Размах, скорость и рывок считаются по ИСТИННОМУ будущему чанку,
которого на инференсе нет. Это привилегированная информация, поэтому связь по
ним — ВЕРХНЯЯ оценка для настоящего планировщика, а не достижимая. Слабая
связь при подглядывании тем более означает слабую предсказуемость.
Строка «ошибка при k=1» — тоже подглядывание, но она показывает потолок:
столько даёт знание собственной ошибки на первом уровне.""")


if __name__ == "__main__":
    main()
