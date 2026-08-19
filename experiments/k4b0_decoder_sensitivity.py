"""K-4b0.2: профиль чувствительности ДЕКОДЕРА ActionCodec по временным позициям.

ВОПРОС. Замер K-4b0.1 показал, что фиксированный набор позиций [0, 9, 10, 11]
закрывает 0.629 разрыва при одиночном оракуле 0.701, причём знание позиции
вмешательства p добавляет статистически неотличимый от нуля вклад. Профиль
двугорбый и не объясняется ни частотой попадания в changed-support (она почти
равномерна), ни разметкой position ids (пик совпал при офсетах 3 и 4).

ГИПОТЕЗА, КОТОРУЮ ЗДЕСЬ ПРОВЕРЯЕМ. Профиль — свойство ДЕКОДЕРА, а не модели, не
состояния и не процедуры refinement. Метрика ошибки считается по ПЕРВЫМ
`window` декодированным шагам действия (k4b0_build_router_dataset.py:739), а
латента имеет 16 временных позиций, разворачиваемых в T = freq*duration шагов.
Тогда «важные» позиции — просто те, что влияют на начало чанка, и это чистая
геометрия декодера.

ЧЕМ ЭТО ВАЖНО. Если гипотеза верна:
  - фиксированная маска оказывается свойством ТОКЕНИЗАТОРА, переносимым на
    любую модель поверх этого ActionCodec, а не результатом про наш VLM;
  - профиль обязан быть тем же на настоящих траекториях потока, что меняет
    априорную оценку фазы D0;
  - обусловленность состоянием надо оценивать УСЛОВНО на этой геометрии.
Если профиль чувствительности равномерен — гипотеза отпадает, и объяснение
надо искать в модели.

ЧТО СЧИТАЕТСЯ. Берутся НАСТОЯЩИЕ последовательности кодов из features.npz
(`cand_old_tokens`) и настоящие проекции кодов (`codebook_proj`) — VLM не
запускается, LIBERO не загружается, нужен только декодер из checkpoint.
Для каждой позиции q измеряется, насколько меняется декодированное действие при
возмущении ТОЛЬКО этой позиции, двумя способами:

  code — грубый код позиции q заменяется на равномерно случайный другой
         (это ровно то возмущение, из которого построен датасет);
  norm — к латенте позиции q добавляется гауссов шум фиксированной нормы
         (чистая якобианова чувствительность, вне зависимости от кодовой книги).

Каждый способ считается в ДВУХ окнах:
  первые `window` шагов — как в метрике датасета;
  весь чанк           — контроль, отделяющий геометрию декодера от выбора окна.

Если профиль пиковый на первых шагах и плоский на всём чанке, эффект создан
сочетанием «декодер + короткое окно», а не важностью позиций как таковой.

Запуск:
    python3 experiments/k4b0_decoder_sensitivity.py \
        --dir data/k4b0_v2 --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO
    python3 experiments/k4b0_decoder_sensitivity.py --selftest
"""

import argparse
import json
import os
import sys

import numpy as np


def selftest():
    """Синтетика с ИЗВЕСТНЫМ ОТВЕТОМ для машинки профиля.

    Строим поддельный «декодер», у которого влияние позиции q на выход задано
    руками: линейное отображение с известными весами. Профиль обязан
    воспроизвести эти веса с точностью до масштаба, а на равномерных весах
    обязан выйти плоским.
    """
    P, D, T = 6, 8, 12

    def profile(W, seed=0):
        """Профиль отклика для поддельного линейного «декодера» с весами W.

        Вклад позиции p в выход равен W[p] * (среднее латенты по каналам),
        поэтому ожидаемый ответ известен точно: отклик пропорционален |W[p]|.
        """
        r = np.random.default_rng(seed)
        lat0 = r.normal(size=(256, P, D))
        a0 = np.einsum("bp,pt->bt", lat0.mean(-1), W)
        out = np.zeros(P)
        for q in range(P):
            lat = lat0.copy()
            lat[:, q] += r.normal(size=(256, D))
            a = np.einsum("bp,pt->bt", lat.mean(-1), W)
            out[q] = np.sqrt(((a - a0) ** 2).mean())
        return out

    W = np.zeros((P, T))
    W[1], W[4] = 3.0, 1.0            # позиция 4 ровно втрое слабее первой
    prof = profile(W)
    assert prof.argmax() == 1, f"пик не там: {prof}"
    assert abs(prof[1] / prof[4] - 3.0) < 0.3, \
        f"отношение весов не воспроизведено: {prof[1] / prof[4]:.2f}"
    assert prof[[0, 2, 3, 5]].max() < 1e-12, "нулевые позиции дали отклик"

    flat = profile(np.ones((P, T)))  # равномерные веса -> плоский профиль
    assert flat.std() / flat.mean() < 0.25, f"плоский профиль не плоский: {flat}"
    print("самопроверка пройдена: профиль воспроизводит известные веса "
          "и остаётся плоским при равномерных")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="каталог датасета K-4b0")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--n-rows", type=int, default=512,
                    help="сколько последовательностей кодов взять")
    ap.add_argument("--n-rep", type=int, default=8,
                    help="повторов возмущения на позицию")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if not (args.dir and args.ckpt):
        raise SystemExit("нужны --dir и --ckpt, либо --selftest")

    import copy
    import importlib.util

    import torch
    import actioncodec  # noqa: F401

    meta = json.load(open(os.path.join(args.dir, "metadata.json")))
    _ft = np.load(os.path.join(args.dir, "features.npz"), allow_pickle=True)
    proj = _ft["codebook_proj"]                      # (L, V, D)
    old = _ft["cand_old_tokens"]                     # (n, P, L)
    window, scale = int(meta["window"]), float(meta["scale"])
    n_cont = int(meta["continuous_channels"])
    P, L = old.shape[1], old.shape[2]
    print(f"коды {old.shape}, проекции {proj.shape}, окно метрики {window}, "
          f"непрерывных каналов {n_cont}")

    sys.path.insert(0, os.path.abspath(args.root))
    sp = importlib.util.spec_from_file_location(
        "ac_vla_tok", os.path.join(os.path.abspath(args.root), "utils",
                                   "vla_tokenizer.py"))
    mm = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mm)
    proc = mm.VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    tok = copy.deepcopy(proc.action_processor).to(args.device).float().eval()
    cfg = tok.config.embodiment_config[
        list(tok.config.embodiment_config.keys())[args.embodiment]]
    T, D_act = int(cfg["freq"] * cfg["duration"]), cfg["action_dim"]
    assert P == tok.n_tokens_per_quantizer and L == tok.num_quantizers, \
        f"формы не сходятся: {P},{L} против {tok.n_tokens_per_quantizer}," \
        f"{tok.num_quantizers}"
    V = proj.shape[1]
    print(f"декодер: {P} позиций -> {T} шагов действия, {D_act} каналов")

    E = torch.as_tensor(proj, device=args.device).float()
    rng = np.random.default_rng(0)
    sel = rng.choice(len(old), size=min(args.n_rows, len(old)), replace=False)
    codes0 = torch.as_tensor(np.asarray(old[sel], np.int64), device=args.device)

    def latent(c):
        return sum(E[j][c[:, :, j]] for j in range(L))

    @torch.no_grad()
    def dec(h):
        out = []
        for i in range(0, len(h), args.chunk):
            out.append(tok._decode(h[i:i + args.chunk],
                                   args.embodiment, None)[0][..., :D_act])
        return torch.cat(out)

    def err(a, ref, w):
        """Та же величина, что в датасете: RMS по непрерывным каналам в окне w,
        нормированная на общий размах действий."""
        d = (a[:, :w] - ref[:, :w]).abs()[..., :n_cont]
        return (d.flatten(1).pow(2).mean(-1).sqrt() / scale).cpu().numpy()

    with torch.no_grad():
        a0 = dec(latent(codes0))
    # НОРМА ШУМА привязана к типичному расстоянию между кодами, иначе величина
    # отклика зависела бы от произвольно выбранного эпсилона.
    step = float(latent(codes0).std().item())
    print(f"типичное std латенты {step:.4f}; шум берётся той же нормы")

    prof = {k: np.zeros(P) for k in
            ("code/окно", "code/весь", "norm/окно", "norm/весь")}
    for q in range(P):
        acc = {k: [] for k in prof}
        for rep in range(args.n_rep):
            g = torch.Generator(device=args.device).manual_seed(1000 * q + rep)
            c = codes0.clone()
            # РАВНОМЕРНО ДРУГОЙ код: сдвиг на случайное ненулевое смещение по
            # модулю V гарантирует, что новый код не совпал со старым.
            shift = torch.randint(1, V, (len(c),), device=args.device,
                                  generator=g)
            c[:, q, 0] = (c[:, q, 0] + shift) % V
            with torch.no_grad():
                a = dec(latent(c))
            acc["code/окно"].append(err(a, a0, window))
            acc["code/весь"].append(err(a, a0, T))

            h = latent(codes0).clone()
            nz = torch.randn(h.shape[0], h.shape[2], device=args.device,
                             generator=g)
            h[:, q] += nz / nz.norm(dim=-1, keepdim=True) * step * np.sqrt(
                h.shape[2])
            with torch.no_grad():
                a = dec(h)
            acc["norm/окно"].append(err(a, a0, window))
            acc["norm/весь"].append(err(a, a0, T))
        for k in prof:
            prof[k][q] = float(np.mean(acc[k]))
        print(f"  позиция {q:>2}: code/окно {prof['code/окно'][q]:.5f}  "
              f"code/весь {prof['code/весь'][q]:.5f}  "
              f"norm/окно {prof['norm/окно'][q]:.5f}", flush=True)

    print("\n" + "=" * 74)
    print("ПРОФИЛЬ ЧУВСТВИТЕЛЬНОСТИ ДЕКОДЕРА (нормирован на максимум)")
    print("=" * 74)
    print(f"  {'поз':>4}" + "".join(f"{k:>14}" for k in prof))
    for q in range(P):
        print(f"  {q:>4}" + "".join(
            f"{prof[k][q] / prof[k].max():>14.3f}" for k in prof))

    fixed = sorted(np.argsort(-prof["code/окно"])[:4].tolist())
    print(f"\n  четыре самые чувствительные позиции по code/окно: {fixed}")
    print("  для сравнения, маска из K-4b0.1 по одиночным выигрышам: [0, 9, 10, 11]")
    for k in prof:
        cv = prof[k].std() / prof[k].mean()
        print(f"  {k:<12} коэффициент вариации {cv:.3f}"
              + ("  — профиль ПИКОВЫЙ" if cv > 0.3 else "  — профиль плоский"))
    print("\n  ЧИТАТЬ ТАК. Пиковый code/окно при плоском code/весь означает, что\n"
          "  эффект создан сочетанием геометрии декодера и КОРОТКОГО окна\n"
          "  метрики, а не важностью позиций самих по себе. Совпадение четырёх\n"
          "  верхних позиций с [0, 9, 10, 11] означает, что маска K-4b0.1 —\n"
          "  свойство токенизатора, а не обученного VLM.")

    if args.out:
        json.dump({k: v.tolist() for k, v in prof.items()}
                  | dict(fixed=fixed, window=window, T=T, n_rows=len(sel),
                         n_rep=args.n_rep, dataset_commit=meta.get("commit")),
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
