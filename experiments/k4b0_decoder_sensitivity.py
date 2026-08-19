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
возмущении ТОЛЬКО этой позиции, несколькими способами:

  код L1, L2, L1+L2 — код позиции q на данном уровне RVQ заменяется на
         равномерно случайный другой. ИМЕННО FINE-УРОВНИ, потому что в K-4b0
         повреждение живёт в них: грубый код возвращается к опорному
         (k4b0_build_router_dataset.py:800), а испорчены уровни 1 и 2, которые
         модель сгенерировала под неверный coarse. Режим по уровню 0 оставлен
         только для сравнения силы и помечен как НЕ соответствующий датасету;
  шум по норме уровня — к латенте позиции q добавляется гауссов шум с нормой,
         равной среднему расстоянию между двумя случайными кодами ЭТОГО уровня.
         Уровни RVQ имеют разный масштаб по построению, поэтому шум одной и той
         же абсолютной нормы означал бы на разных уровнях разное по силе
         возмущение. Это чистая якобианова чувствительность, не зависящая от
         кодовой книги.

ЧЕГО ЗОНД НЕ ДЕЛАЕТ. Он НЕ воспроизводит вмешательство K-4b0 буквально: там
грубый код берётся из рангов 2-5 собственного распределения модели, после чего
fine-уровни ПЕРЕГЕНЕРИРУЮТСЯ моделью, и повреждение есть разность между двумя
её собственными выходами. Равномерная замена кода — более грубое и более
сильное возмущение. Поэтому зонд отвечает на вопрос о ГЕОМЕТРИИ декодера, а
величины откликов с одиночными выигрышами датасета сравнивать по абсолютной
шкале нельзя — только по рангам позиций.

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
import hashlib
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
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="сидов выборки наблюдений")
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

    # ПУТЬ ДО ИМПОРТА. Пакет actioncodec живёт в вендоренном дереве и не
    # установлен в окружение, поэтому sys.path обязан быть дополнен раньше
    # импорта — как в k4b0_build_router_dataset.py.
    sys.path.insert(0, os.path.abspath(args.root))
    import actioncodec  # noqa: F401,E402

    meta = json.load(open(os.path.join(args.dir, "metadata.json")))
    _ft = np.load(os.path.join(args.dir, "features.npz"), allow_pickle=True)
    proj = _ft["codebook_proj"]                      # (L, V, D)
    old = _ft["cand_old_tokens"]                     # (n, P, L)
    window, scale = int(meta["window"]), float(meta["scale"])
    n_cont = int(meta["continuous_channels"])
    P, L = old.shape[1], old.shape[2]
    print(f"коды {old.shape}, проекции {proj.shape}, окно метрики {window}, "
          f"непрерывных каналов {n_cont}")

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

    # ЦЕЛОСТНОСТЬ. Профиль сравнивается с числами конкретного датасета, поэтому
    # молчаливое расхождение checkpoint или кодовых книг обесценило бы вывод.
    if meta.get("ckpt") and meta["ckpt"] != args.ckpt:
        raise SystemExit(f"checkpoint не тот: датасет собран на "
                         f"{meta['ckpt']}, передан {args.ckpt}")
    E = torch.as_tensor(proj, device=args.device).float()
    from k1_residual_cost import projected_codebooks           # noqa: E402
    E_live = projected_codebooks(tok, args.device).float()
    dev = (E_live - E).abs().max().item()
    assert dev < 5e-3, (f"codebook_proj из датасета разошёлся с загруженным "
                        f"токенизатором: макс {dev:.2e}")
    print(f"  кодовые книги сходятся с датасетом: макс расхождение {dev:.2e} "
          f"(хранятся во float16)")

    # ТИПИЧНОЕ РАССТОЯНИЕ МЕЖДУ КОДАМИ ПО УРОВНЯМ. Уровни RVQ имеют разный
    # масштаб по построению — каждый следующий кодирует остаток, — поэтому
    # шум одной и той же нормы означал бы на разных уровнях разное по силе
    # возмущение. Нормируем на средний шаг внутри уровня.
    g0 = torch.Generator(device=args.device).manual_seed(0)
    step_l = []
    for j in range(L):
        i1 = torch.randint(0, V, (4096,), device=args.device, generator=g0)
        i2 = torch.randint(0, V, (4096,), device=args.device, generator=g0)
        step_l.append(float((E[j][i1] - E[j][i2]).norm(dim=-1).mean()))
    print("  средний шаг между кодами по уровням: "
          + ", ".join(f"L{j}={v:.4f}" for j, v in enumerate(step_l)))

    obs = np.asarray(_ft["int_obs_idx"])
    sing = np.asarray(np.load(os.path.join(args.dir, "labels.npz"),
                              allow_pickle=True)["sing_gain_rms"])

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

    # РЕЖИМЫ ВОЗМУЩЕНИЯ. Повреждение в K-4b0 живёт В FINE-УРОВНЯХ: грубый код
    # возвращается к опорному (k4b0_build_router_dataset.py:800), а испорчены
    # уровни 1 и 2, которые модель сгенерировала под неверный coarse. Поэтому
    # режим по уровню 0 помечен как НЕ соответствующий датасету и оставлен
    # только для сравнения силы.
    MODES = [("код L0 (НЕ как в датасете)", "code", 0),
             ("код L1", "code", 1),
             ("код L2", "code", 2),
             ("коды L1+L2 (как в датасете)", "code", 12),
             ("шум по норме L0", "norm", 0),
             ("шум по норме L1", "norm", 1),
             ("шум по норме L2", "norm", 2)]

    def perturb(codes, q, mode, lev, g):
        """Возмущение ТОЛЬКО позиции q. Возвращает латенту."""
        c = codes.clone()
        if mode == "code":
            for j in ([1, 2] if lev == 12 else [lev]):
                # сдвиг на случайное НЕНУЛЕВОЕ смещение по модулю V: новый код
                # гарантированно отличается от старого
                sh = torch.randint(1, V, (len(c),), device=args.device,
                                   generator=g)
                c[:, q, j] = (c[:, q, j] + sh) % V
            return latent(c)
        h = latent(c)
        nz = torch.randn(h.shape[0], h.shape[2], device=args.device,
                         generator=g)
        h[:, q] += nz / nz.norm(dim=-1, keepdim=True) * step_l[lev]
        return h

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    all_prof, top4_seen = {}, {}
    for seed in range(args.n_seeds):
        # УНИКАЛЬНЫЕ НАБЛЮДЕНИЯ, а не строки: 16 строк одного наблюдения
        # отличаются лишь позицией правки и сильно коррелированы, поэтому
        # выборка по строкам завысила бы эффективный размер.
        r = np.random.default_rng(seed)
        uo = np.unique(obs)
        pick_obs = r.choice(uo, size=min(args.n_rows, len(uo)), replace=False)
        sel = np.array([r.choice(np.where(obs == o)[0]) for o in pick_obs])
        codes0 = torch.as_tensor(np.asarray(old[sel], np.int64),
                                 device=args.device)
        with torch.no_grad():
            a0 = dec(latent(codes0))
        print(f"\n  сид выборки {seed}: наблюдений {len(sel)}")

        prof = {f"{nm}/{w}": np.zeros((P, len(sel)))
                for nm, _, _ in MODES for w in ("окно", "весь")}
        for q in range(P):
            acc = {k: [] for k in prof}
            for nm, mode, lev in MODES:
                for rep in range(args.n_rep):
                    g = torch.Generator(device=args.device).manual_seed(
                        100000 * seed + 1000 * q + 10 * rep + lev)
                    with torch.no_grad():
                        a = dec(perturb(codes0, q, mode, lev, g))
                    acc[f"{nm}/окно"].append(err(a, a0, window))
                    acc[f"{nm}/весь"].append(err(a, a0, T))
            for k in prof:
                prof[k][q] = np.mean(acc[k], 0)
        for k, v in prof.items():
            all_prof.setdefault(k, []).append(v)
            top4_seen.setdefault(k, []).append(
                tuple(sorted(np.argsort(-v.mean(1))[:4].tolist())))
        print(f"    top-4 по «коды L1+L2/окно»: "
              f"{top4_seen['коды L1+L2 (как в датасете)/окно'][-1]}")

    print("\n" + "=" * 74)
    print("ПРОФИЛЬ ЧУВСТВИТЕЛЬНОСТИ ДЕКОДЕРА")
    print("=" * 74)
    sing_prof = sing.mean(0)
    res = {}
    for k, mats in all_prof.items():
        m = np.concatenate(mats, 1)                    # (P, наблюдений*сидов)
        mu = m.mean(1)
        # ИНТЕРВАЛ ПО НАБЛЮДЕНИЯМ: строки независимы по построению выборки.
        rb = np.random.default_rng(0).integers(0, m.shape[1],
                                               size=(1000, m.shape[1]))
        bs = m[:, rb].mean(2)                          # (P, 1000)
        cv = mu.std() / mu.mean()
        top4 = sorted(np.argsort(-mu)[:4].tolist())
        stab = len(set(top4_seen[k])) == 1
        rho = spearman(mu, sing_prof)
        res[k] = dict(profile=mu.tolist(), cv=float(cv), top4=top4,
                      spearman_with_singleton=rho, stable_top4=bool(stab),
                      ci=[np.percentile(bs, 2.5, axis=1).tolist(),
                          np.percentile(bs, 97.5, axis=1).tolist()])
        print(f"\n  {k}")
        print(f"    top-4 {top4}   коэф. вариации {cv:.3f}"
              + ("  ПИКОВЫЙ" if cv > 0.3 else "  плоский"))
        print(f"    ранговая корреляция с профилем одиночных выигрышей "
              f"{rho:+.3f}")
        print(f"    top-4 устойчив по {args.n_seeds} сидам: "
              f"{'да' if stab else 'НЕТ, ' + str(set(top4_seen[k]))}")
        print("    " + " ".join(f"{v / mu.max():.2f}" for v in mu))

    print("\n  ЧИТАТЬ ТАК. Пиковый профиль в окне при плоском на всём чанке\n"
          "  означает, что эффект создан сочетанием геометрии декодера и\n"
          "  КОРОТКОГО окна метрики. Высокая ранговая корреляция с профилем\n"
          "  одиночных выигрышей означает, что маска K-4b0.1 — свойство\n"
          "  токенизатора, а не обученного VLM. Ни то, ни другое НЕ объясняет\n"
          "  разрыв между точным оракулом и фиксированной маской.")

    if args.out:
        try:
            import subprocess
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                             text=True).strip()
        except Exception:
            commit = "unknown"
        res["meta"] = dict(
            analysis_commit=commit, dataset_commit=meta.get("commit"),
            ckpt=args.ckpt, window=window, T=int(T), P=int(P), L=int(L),
            n_rows=int(args.n_rows), n_rep=int(args.n_rep),
            n_seeds=int(args.n_seeds), step_by_level=step_l,
            singleton_profile=sing_prof.tolist(),
            sha256={f: hashlib.sha256(
                open(os.path.join(args.dir, f), "rb").read()).hexdigest()
                for f in ("features.npz", "labels.npz")})
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
