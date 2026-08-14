"""K-1: есть ли у устаревшего суффикса RVQ собственная цена.

АРХИТЕКТУРНЫЙ ФАКТ (из кода, не из замера). Декодер ActionCodec получает
ТОЛЬКО сумму векторов уровней:

    modeling_actioncodec.py:271  z_q = self.vq.from_codes(indices)[0]   # СУММА
    modeling_actioncodec.py:303  x_recon, _ = self.decoder(z_q, ...)

Значит h_A = h_B  =>  Dec(h_A) = Dec(h_B) тождественно: состав кортежа не
является входом декодера. Любое различие обязано проходить через смещение
суммы. Это не гипотеза, это следует из устройства.

ЧТО ТОГДА ЕЩЁ МОЖНО ИЗМЕРИТЬ. Смещение суммы у варианта со старым суффиксом
может уводить её в область, где декодер обучался хуже. Тогда при ОДИНАКОВОМ
смещении он всё равно декодировал бы хуже. Это и проверяем.

  A — подмена кода, суффикс оставлен старым;
  B — та же подмена, суффикс переквантован ЛОКАЛЬНО (см. ниже).

Наивное «B лучше A» не значит ничего: B строится ближе к цели, это арифметика.
Поэтому сравнение ведётся ПРИ РАВНОЙ ОШИБКЕ ЛАТЕНТЫ, по корзинам.

ЧТО ИСПРАВЛЕНО ПРОТИВ ПЕРВОЙ ВЕРСИИ (её вывод был получен с ошибками):

  1. ЛОКАЛЬНОСТЬ. Раньше суффикс переквантовывался во ВСЕХ 16 позициях, хотя
     подмена делалась в одной. И в нетронутых позициях он тоже менялся:
     кодировщик выбирал код против остатка НЕПРОКВАНТОВАННОЙ латенты, а мы
     квантуем против суммы уже проквантованных — цели разные. То есть
     сравнивались два разных вмешательства. Теперь трогается только изменённая
     позиция, и это проверяется assert'ом.
  2. РАЗБОР ПО УРОВНЯМ. Раньше все уровни сваливались в общие корзины. У
     ПОСЛЕДНЕГО уровня суффикса нет, A и B тождественны, отношение ровно 1.00 —
     и эта треть замеров тянула медиану к единице. Теперь последний уровень
     идёт только как отрицательный юнит-тест и в основную статистику не входит.
  3. РЕЖИМ ПОДСТАНОВОК. Раньше код брался равномерно из 2048. Поток предлагает
     ПРАВДОПОДОБНЫЕ коды, рядом с текущим. Теперь есть режим local (из k
     ближайших) и uniform, и они разбираются отдельно.
  4. ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ. Замеры сильно коррелированы внутри чанка (64
     чанка на тысячи замеров), поэтому кластерный бутстрап ПО ЧАНКАМ.
  5. Формулировка: жадная переквантизация НЕ является оптимальной проекцией,
     она жадная. Раньше было написано «наилучшее приближение» — неверно.

ПРАВИЛО ЧТЕНИЯ, зафиксировано до запуска. Смотрим на режим local (он
соответствует работе генератора), уровни отдельно, доверительный интервал:
  нижняя граница интервала выше 1.15 хотя бы на одном уровне -> есть
      дополнительный эффект сверх смещения суммы;
  интервал накрывает 1.0 -> дополнительного эффекта не обнаружено.

Запуск:
    python3 experiments/k1_residual_cost.py --zarr <путь>/libero10_N500.zarr
"""

import argparse
import os
import sys

import einops
import numpy as np
import torch


def load_codec(root: str, model_id: str, device: str):
    sys.path.insert(0, root)
    from actioncodec.modeling_actioncodec import ActionCodec

    m = ActionCodec.from_pretrained(model_id).to(device).eval()
    print(f"словарь {m.vocab_size}, уровней {m.num_quantizers}, "
          f"позиций {m.n_tokens_per_quantizer}")
    return m


def projected_codebooks(model, device: str) -> torch.Tensor:
    """(L, V, D) — вектор, который код ДОБАВЛЯЕТ к сумме. Остаток живёт именно
    здесь: from_codes складывает out_project(decode_code(c))."""
    V = model.vocab_size
    idx = torch.arange(V, device=device).unsqueeze(0)
    with torch.no_grad():
        out = [q.out_project(q.decode_code(idx))[0] for q in model.vq.quantizers]
    E = torch.stack(out)
    assert E.shape[0] == model.num_quantizers, E.shape
    return E


def latent_from_codes(E: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    return sum(E[j][codes[:, :, j]] for j in range(E.shape[0]))


def requantize_at(E: torch.Tensor, codes: torch.Tensor, target: torch.Tensor,
                  p: int, level: int) -> torch.Tensor:
    """Жадно переквантовать уровни > level ТОЛЬКО в позиции p.

    Жадно — не оптимально: на каждом шаге берётся ближайший код к текущему
    остатку, что вообще говоря не даёт наилучшего приближения суммой."""
    out = codes.clone()
    L = E.shape[0]
    r = target[:, p] - sum(E[j][out[:, p, j]] for j in range(level + 1))
    for j in range(level + 1, L):
        c = torch.cdist(r.unsqueeze(1), E[j]).squeeze(1).argmin(-1)
        out[:, p, j] = c
        r = r - E[j][c]
    return out


def pick_candidates(E: torch.Tensor, cur: torch.Tensor, level: int,
                    regime: str, knn: int, gen) -> torch.Tensor:
    """cur (B,) — текущие коды. Возвращает кандидатов (B,), не равных текущему.

    local  — из knn ближайших по словарю: так ведёт себя генератор, он не
             прыгает в произвольный код;
    uniform — равномерно по всему словарю: широкий диапазон смещений, нужен
             для контраста, но это режим, которого в работе не бывает."""
    V = E.shape[1]
    if regime == "uniform":
        v = torch.randint(V, cur.shape, generator=gen, device=cur.device)
        bad = v == cur
        v[bad] = (v[bad] + 1) % V
        return v
    d = torch.cdist(E[level][cur], E[level])            # (B, V)
    nb = d.topk(knn + 1, largest=False).indices[:, 1:]  # (B, knn), без себя
    pick = torch.randint(knn, (len(cur),), generator=gen, device=cur.device)
    return nb[torch.arange(len(cur), device=cur.device), pick]


def binned_ratio(rows: np.ndarray, n_bins: int, skew_max: float = 0.15,
                 min_n: int = 20, verbose: bool = False):
    """rows: (chunk, lat_err, act_err, variant). Медиана A/B по выровненным
    корзинам. Возвращает (медиана, число корзин, строки для печати)."""
    q = np.quantile(rows[:, 1], np.linspace(0, 1, n_bins + 1))
    ratios, lines = [], []
    for k in range(n_bins):
        m = (rows[:, 1] >= q[k]) & (rows[:, 1] < q[k + 1] if k < n_bins - 1
                                    else rows[:, 1] <= q[k + 1])
        mA, mB = m & (rows[:, 3] == 0), m & (rows[:, 3] == 1)
        nA, nB = int(mA.sum()), int(mB.sum())
        if nA < min_n or nB < min_n:
            continue
        lA, lB = rows[mA, 1].mean(), rows[mB, 1].mean()
        skew = (lA - lB) / max(q[k + 1] - q[k], 1e-12)
        eA, eB = np.median(rows[mA, 2]), np.median(rows[mB, 2])
        r = eA / max(eB, 1e-9)
        ok = abs(skew) < skew_max
        if ok:
            ratios.append(r)
        if verbose:
            lines.append(f"{f'[{q[k]:.3f}, {q[k+1]:.3f})':>22}{nA:>6}{nB:>6}"
                         f"{skew:>+7.2f}{eA:>11.4f}{eB:>11.4f}{r:>7.2f}"
                         f"{'' if ok else '  (перекос)'}")
    return (float(np.median(ratios)) if ratios else float("nan"),
            len(ratios), lines)


def cluster_bootstrap(rows: np.ndarray, n_bins: int, n_boot: int, seed: int = 0):
    """Кластерный бутстрап ПО ЧАНКАМ: замеры внутри чанка сильно коррелированы,
    и обычный бутстрап по строкам дал бы неправдоподобно узкий интервал."""
    rng = np.random.default_rng(seed)
    chunks = np.unique(rows[:, 0]).astype(int)
    by = {c: rows[rows[:, 0] == c] for c in chunks}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(chunks, size=len(chunks), replace=True)
        r, n, _ = binned_ratio(np.concatenate([by[c] for c in pick]), n_bins)
        if r == r and n >= 3:
            out.append(r)
    if len(out) < n_boot // 4:
        return None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--model", default="ZibinDong/ActionCodec-Base-RVQft")
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--n-chunks", type=int, default=96)
    ap.add_argument("--n-cand", type=int, default=48, help="подстановок на уровень")
    ap.add_argument("--knn", type=int, default=16, help="соседей в режиме local")
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--no-gripper", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import zarr

    model = load_codec(os.path.abspath(args.root), args.model, args.device)
    V, L, P = model.vocab_size, model.num_quantizers, model.n_tokens_per_quantizer
    name = list(model.config.embodiment_config.keys())[args.embodiment]
    cfg = model.config.embodiment_config[name]
    T, D_act = int(cfg["freq"] * cfg["duration"]), cfg["action_dim"]
    print(f"эмбодимент {args.embodiment} = {name}: чанк {T}x{D_act}\n")

    # ---------- данные ----------
    z = zarr.open(os.path.abspath(args.zarr), mode="r")
    acts = np.asarray(z["data"]["action"])
    ends = np.asarray(z["meta"]["episode_ends"])
    assert acts.shape[1] == D_act
    chunks, start = [], 0
    for e in ends:
        ep = acts[start:e]
        chunks += [ep[i * T:(i + 1) * T] for i in range(len(ep) // T)]
        start = e
    A = np.stack(chunks).astype(np.float32)

    # Соглашение о захвате у LIBERO ПЕРЕВЁРНУТОЕ относительно описания кодека
    # («1 open / 0 close»). Измерено: (x+1)/2 даёт пол 0.5120 размаха на этом
    # канале, (1-x)/2 даёт 0.0023.
    if A[:, :, -1].min() < -0.5:
        A[:, :, -1] = (1.0 - A[:, :, -1]) / 2.0
        print("захват: -1/+1 -> (1-x)/2")

    idx = np.random.default_rng(0).choice(len(A), size=min(args.n_chunks, len(A)),
                                          replace=False)
    a = torch.from_numpy(A[idx]).to(args.device)
    B = len(a)
    scale = float(a.max() - a.min())
    print(f"чанков {B}, размах {scale:.2f}")

    E = projected_codebooks(model, args.device)
    Dz = E.shape[-1]
    gen = torch.Generator(device=args.device).manual_seed(1)

    with torch.no_grad():
        flat = torch.as_tensor(np.asarray(model.encode(a, embodiment_ids=args.embodiment)),
                               device=args.device, dtype=torch.long)
        # Докстринг decode обещает перемежение по времени, но код делает
        # rearrange("b (n m) -> b m n", m=n_tokens_per_quantizer), то есть
        # раскладка ПОУРОВНЕВАЯ. Пользуемся их же вызовом.
        codes = einops.rearrange(flat, "b (n m) -> b m n", m=P)

        def unflat(c):
            return einops.rearrange(c, "b m n -> b (n m)")

        assert torch.equal(unflat(codes), flat)

        h0 = latent_from_codes(E, codes)
        gap = (h0 - model.vq.from_codes(codes)[0]).abs().max().item()
        print(f"сверка суммы с from_codes: {gap:.2e}")
        assert gap < 1e-4, "считаем в неверном пространстве"

        def decode(c):
            # numpy, а не torch: их torch-ветка зовёт tokens.dtype.is_integer,
            # чего у torch.dtype нет.
            rec, _ = model.decode(unflat(c).cpu().numpy().astype(np.int64),
                                  embodiment_ids=args.embodiment)
            return torch.as_tensor(np.asarray(rec)[..., :D_act],
                                   device=args.device, dtype=torch.float32)

        base = decode(codes)
        floor = ((base - a).abs().median() / scale).item()
        print(f"пол кодека: {floor:.4f} размаха, по каналам "
              + " ".join(f"{((base-a)[:,:,d].abs().median()/scale).item():.4f}"
                         for d in range(D_act)))
        assert floor < 0.05, "формат действий не сошёлся"

        # ---------- перебор ----------
        rows = {}                       # (level, regime) -> список строк
        unit_ok = True
        for lev in range(L):
            for regime in ("local", "uniform"):
                acc = []
                for _ in range(args.n_cand):
                    p = int(torch.randint(P, (1,), generator=gen,
                                          device=args.device))
                    v = pick_candidates(E, codes[:, p, lev], lev, regime,
                                        args.knn, gen)
                    cA = codes.clone()
                    cA[:, p, lev] = v
                    cB = requantize_at(E, cA, h0, p, lev)

                    # ЛОКАЛЬНОСТЬ: тронута только позиция p
                    other = [q for q in range(P) if q != p]
                    assert torch.equal(cB[:, other, :], cA[:, other, :]), \
                        "переквантизация задела чужие позиции"
                    if lev == L - 1:
                        unit_ok &= torch.equal(cB, cA)

                    for tag, c in ((0, cA), (1, cB)):
                        h = latent_from_codes(E, c)
                        le = (h - h0).norm(dim=-1).amax(-1) / np.sqrt(Dz)
                        dv = (decode(c) - base).abs()
                        if args.no_gripper:
                            dv = dv[..., :-1]
                        ae = dv.flatten(1).amax(-1) / scale
                        for i in range(B):
                            acc.append((i, float(le[i]), float(ae[i]), tag))
                rows[(lev, regime)] = np.array(acc)

    print(f"\nюнит-тест последнего уровня (A должно быть тождественно B): "
          f"{'ПРОЙДЕН' if unit_ok else 'ПРОВАЛЕН'}")
    assert unit_ok, "на последнем уровне суффикса нет, A и B обязаны совпадать"

    # ---------- анализ ----------
    print("\n" + "=" * 78)
    print("ПРИ РАВНОЙ ОШИБКЕ ЛАТЕНТЫ, ПО УРОВНЯМ И РЕЖИМАМ")
    print("=" * 78)
    print("Последний уровень исключён: там A ≡ B по построению, и его включение\n"
          "в общую статистику тянуло бы медиану к 1.0 механически.\n")

    verdict = []
    for lev in range(L - 1):
        for regime in ("local", "uniform"):
            R = rows[(lev, regime)]
            med, nb, lines = binned_ratio(R, args.n_bins, verbose=True)
            ci = cluster_bootstrap(R, args.n_bins, args.n_boot)
            print(f"--- уровень {lev}, режим {regime} ---")
            print(f"{'корзина':>22}{'n(A)':>6}{'n(B)':>6}{'пере':>7}"
                  f"{'действие A':>11}{'действие B':>11}{'A/B':>7}")
            for ln in lines:
                print(ln)
            ci_s = (f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "не сошёлся")
            print(f"  медиана A/B = {med:.2f} по {nb} выровненным корзинам, "
                  f"95% ДИ {ci_s}\n")
            if regime == "local" and ci:
                verdict.append((lev, med, ci))

    print("=" * 78)
    print("ВЫВОД (по режиму local — он соответствует работе генератора)")
    print("=" * 78)
    strong = [(l, m, c) for l, m, c in verdict if c[0] > 1.15]
    if strong:
        for l, m, c in strong:
            print(f"  уровень {l}: A/B = {m:.2f}, ДИ [{c[0]:.2f}, {c[1]:.2f}] — "
                  f"нижняя граница выше 1.15")
        print("\nЕсть эффект сверх смещения суммы: при равной ошибке латенты\n"
              "устаревший суффикс декодируется хуже. Механизм осмыслен.")
    else:
        for l, m, c in verdict:
            print(f"  уровень {l}: A/B = {m:.2f}, ДИ [{c[0]:.2f}, {c[1]:.2f}]")
        print("\nНи на одном уровне нижняя граница не превысила 1.15.\n"
              "Дополнительного эффекта сверх смещения суммы НЕ ОБНАРУЖЕНО при\n"
              "испытанных вмешательствах. Формулировать именно так: это не\n"
              "доказательство отсутствия, а отсутствие обнаружения.")
    print(f"\nПол кодека {floor:.4f} размаха — числа того же порядка означают,\n"
          f"что эффект тонет в собственной погрешности кодека.")


if __name__ == "__main__":
    main()
