"""K-10b: что предсказуемее из h12 — ЦЕЛЬ чанка или КОД траектории.

ВОПРОС. Согласие с кодами учителя на глубине 12 составляет 33%, на полной
глубине 87%. Гипотеза предлагаемой архитектуры: модель хорошо знает, КУДА
тянуться, и плохо — точную форму пути, а нынешний токенизатор перемешивает то
и другое внутри одного кода и наследует предсказуемость худшего.

Здесь это проверяется прямо, на уже собранных данных: обе величины
предсказываются из ОДНОГО И ТОГО ЖЕ h12, одной и той же ёмкостью, на одном
разбиении.

СРАВНЕНИЕ ЧЕСТНОЕ, А НЕ В ПОЛЬЗУ ЦЕЛИ. Соблазн — сравнить дискретный код с
непрерывной регрессией цели; тогда цель выиграет просто потому, что её не
квантуют. Поэтому цель тоже ДИСКРЕТИЗУЕТСЯ: приходы кластеризуются на K
центроидов по обучающей части, и голова решает такую же задачу
классификации — выбрать один из K. Сравниваются:

    путь кода:  h12 -> 16 кодов из 2048  -> декодер -> приход
    путь цели:  h12 -> 1 код из K        -> центроид -> приход

и обе линии меряются ОДНОЙ величиной — ошибкой прихода, то есть куда рука
на самом деле придёт за исполняемое окно.

ЦЕЛЬ — ЭТО ИНТЕГРАЛ. Действия LIBERO суть приращения позы, поэтому «куда
придём» это их сумма по окну. Считается по первым 8 шагам (наш горизонт
исполнения) и отдельно по всем 20.

ЭТАЛОН. За истинное действие берётся decode(K_true) по трём уровням — лучшее,
что представимо этим токенизатором. Это же определение использовалось как
«эксперт» в K-9c, и обе линии меряются относительно него, поэтому выбор
эталона на сравнение не влияет.

ПРЕ-РЕГИСТРИРОВАННОЕ ЧТЕНИЕ, записано ДО запуска. Сравнивается ошибка прихода
на нетронутой части test:
  * путь цели ниже пути кода на >= 25% -> цель предсказуемее, посылка
    подтверждена;
  * разница <= 10% -> преимущества нет, направление закрывается;
  * между -> не доказано ничего.
Точность классификации печатается, но вердикт по ней НЕ выносится: 64-путевой
выбор и 2048-путевой несравнимы по точности напрямую.

Запуск:
    python3 experiments/k10b_goal_predictability.py --selftest

    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k10b_goal_predictability.py \\
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \\
        --cache data/k9_teacher_150k.npz --h12 data/k9e_orig \\
        --rstar data/k9f/head_only.pt --out data/k10b
"""

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

N_POS, N_LEVEL, T_CHUNK, H_EXEC = 16, 3, 20, 8


def displacement(a, horizon):
    """Приход: сумма приращений позы по первым `horizon` шагам."""
    return a[:, :horizon, :6].sum(axis=1)


def rel_gain(err_goal, err_code):
    """На сколько путь цели лучше пути кода, в долях ошибки кода."""
    if err_code <= 0:
        return None
    return (err_code - err_goal) / err_code


def read_rule(gain, good=0.25, bad=0.10):
    if gain is None:
        return "недействительно"
    if gain >= good:
        return "ЦЕЛЬ ПРЕДСКАЗУЕМЕЕ: посылка подтверждена"
    if gain <= bad:
        return "ПРЕИМУЩЕСТВА НЕТ: направление закрывается"
    return "НЕ ДОКАЗАНО НИЧЕГО"


def selftest():
    # Приход — это ИНТЕГРАЛ, и он обязан гасить знакопеременные приращения.
    a = np.zeros((1, T_CHUNK, 7))
    a[0, :, 0] = [(-1.0) ** i for i in range(T_CHUNK)]
    assert abs(displacement(a, 20)[0, 0]) < 1e-12, "чередование обязано гаснуть"
    a2 = np.zeros((1, T_CHUNK, 7)); a2[0, :, 1] = 0.1
    assert abs(displacement(a2, 8)[0, 1] - 0.8) < 1e-12
    # Схват в приход не входит: канал 6 игнорируется.
    a3 = np.zeros((1, T_CHUNK, 7)); a3[0, :, 6] = 5.0
    assert np.abs(displacement(a3, 8)).max() < 1e-12

    assert abs(rel_gain(0.75, 1.0) - 0.25) < 1e-12
    assert rel_gain(1.0, 0.0) is None
    assert "ПРЕДСКАЗУЕМЕЕ" in read_rule(0.30)
    assert "ПРЕИМУЩЕСТВА НЕТ" in read_rule(0.05)
    assert "НЕ ДОКАЗАНО" in read_rule(0.17)
    # Отрицательный выигрыш читается как отсутствие преимущества, а не как
    # промежуточный случай.
    assert "ПРЕИМУЩЕСТВА НЕТ" in read_rule(-0.4)

    # Эпоха 0 обязана участвовать в отборе: если обучение только портит,
    # побеждает исходное состояние.
    cands = [(0, 1.0), (1, 1.2), (2, 1.5)]
    assert min(cands, key=lambda t: t[1])[0] == 0

    # Взвешенное усреднение по батчам разного размера равно общему.
    rng = np.random.default_rng(0)
    x = rng.random(1000)
    w = sum(float(p.mean()) * len(p) for p in (x[:300], x[300:700], x[700:]))
    assert abs(w / len(x) - float(x.mean())) < 1e-12
    print("самопроверка k10b пройдена (версия «приход как интеграл»): "
          "чередование гаснет, схват не входит, правило чтения на четырёх "
          "исходах, эпоха 0 в кандидатах")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", default="data/k9_teacher_150k.npz")
    ap.add_argument("--h12", help="префикс от k9e (исходный ствол)")
    ap.add_argument("--rstar", default=None,
                    help="head_only.pt от k9f: лучший предсказатель кодов на "
                         "глубине 12. Без него путь кода берётся по кодам "
                         "учителя с полной глубины и будет ЗАВЫШЕН.")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k-goals", default="64,256")
    ap.add_argument("--horizons", default="8,20")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lrs", default="3e-4,1e-3")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--pool", choices=["flat", "mean"], default="flat")
    ap.add_argument("--gain-good", type=float, default=0.25)
    ap.add_argument("--gain-bad", type=float, default=0.10)
    ap.add_argument("--out", default="data/k10b")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    for need in ("ckpt", "h12"):
        if not getattr(args, need):
            raise SystemExit(f"нужен --{need} (или --selftest)")

    sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k10b sha1 {sha}")
    os.makedirs(args.out, exist_ok=True)

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import actioncodec  # noqa: F401
    from utils import VisionLanguageActionProcessor

    dev = torch.device(args.device)

    # --- кэши -----------------------------------------------------------------
    z = np.load(args.cache, allow_pickle=True)
    q_teach = z["teacher_codes_q0"].astype(np.int64)
    split = z["split"]
    N = len(q_teach)
    if "K_true" not in z.files:
        raise SystemExit("в кэше нет K_true — эталон не построить")
    Kt3 = z["K_true"].astype(np.int64)

    md = json.load(open(args.h12 + ".json"))
    if md["trunk"] != "original":
        raise SystemExit(f"{args.h12}: ствол «{md['trunk']}», нужен original")
    if md["n"] != N:
        raise SystemExit(f"{args.h12}: {md['n']} строк против {N}")
    if md.get("cache_vs_live_token_mismatch") is None:
        raise SystemExit("шум хранения h12 не измерен — кэш снят старой k9e")
    H = np.load(md["h12_file"], mmap_mode="r")
    D = md["dim"]
    if H.shape != (N, N_POS, D) or H.dtype != np.float16:
        raise SystemExit(f"h12 формы {H.shape} типа {H.dtype}")
    itr = np.where(split == "train")[0]
    iva = np.where(split == "val")[0]
    ite = np.where(split == "test")[0]
    print(f"кэш: {N}; обучение {len(itr)}, валидация {len(iva)}, test {len(ite)}")
    if len(ite) == 0:
        raise SystemExit("нет части test — итог считать не на чем")

    # --- кодек и эталон -------------------------------------------------------
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None))
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь")
    codec = codec.to(dev).eval()
    with torch.no_grad():
        ii = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(ii))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    def dec_levels(codes, n_lv):
        """Действие из первых n_lv уровней; codes формы (n, n_lv, 16)."""
        out = []
        for i0 in range(0, len(codes), 256):
            k = torch.as_tensor(codes[i0:i0 + 256]).long().to(dev)
            with torch.no_grad():
                zq = E[0][k[:, 0, :]]
                for j in range(1, n_lv):
                    zq = zq + E[j][k[:, j, :]]
                x, _ = codec._decode(zq, embodiment_ids=0)
            out.append(x[..., :7].float().cpu().numpy())
        return np.concatenate(out)

    print("эталон: decode(K_true) по трём уровням", flush=True)
    A_star = dec_levels(Kt3, N_LEVEL)

    # --- путь кода: предсказанные коды на глубине 12 ---------------------------
    if args.rstar:
        import copy
        rd = torch.load(md["readout_file"], map_location="cpu",
                        weights_only=False)
        norm = copy.deepcopy(rd["norm"]).float().to(dev)
        head = copy.deepcopy(rd["head"]).float().to(dev)
        st = torch.load(args.rstar, map_location="cpu", weights_only=False)
        norm.load_state_dict({k[len("norm."):]: v for k, v in st.items()
                              if k.startswith("norm.")})
        head.load_state_dict({k[len("head."):]: v for k, v in st.items()
                              if k.startswith("head.")})
        pc = np.zeros((N, N_POS), np.int64)
        with torch.no_grad():
            for i0 in range(0, N, args.batch):
                h = torch.from_numpy(np.asarray(H[i0:i0 + args.batch])
                                     ).to(dev).float()
                pc[i0:i0 + args.batch] = head(norm(h)).argmax(-1).cpu().numpy()
        code_src = f"R* с глубины 12 ({args.rstar})"
    else:
        pc = q_teach
        code_src = "коды учителя с ПОЛНОЙ глубины (путь кода завышен)"
    print(f"путь кода: {code_src}")
    A_code = dec_levels(pc[:, None, :], 1)
    acc_code = float((pc[ite] == q_teach[ite]).mean())

    # --- цели ------------------------------------------------------------------
    def kmeans(x, k, iters=25, seed=0):
        """Простой k-means. Своя реализация, чтобы не тянуть sklearn."""
        rng = np.random.default_rng(seed)
        c = x[rng.choice(len(x), k, replace=False)].copy()
        for _ in range(iters):
            d = ((x[:, None, :] - c[None]) ** 2).sum(-1) if len(x) < 20000 \
                else None
            if d is None:
                lab = np.empty(len(x), np.int64)
                for i0 in range(0, len(x), 20000):
                    xb = x[i0:i0 + 20000]
                    lab[i0:i0 + 20000] = ((xb[:, None, :] - c[None]) ** 2
                                          ).sum(-1).argmin(1)
            else:
                lab = d.argmin(1)
            for j in range(k):
                m = lab == j
                if m.any():
                    c[j] = x[m].mean(0)
        return c

    results = {}
    for hz in [int(x) for x in args.horizons.split(",")]:
        g_true = displacement(A_star, hz)
        g_code = displacement(A_code, hz)
        err_code = float(np.linalg.norm(g_code[ite] - g_true[ite], axis=1).mean())
        scale = float(np.linalg.norm(g_true[ite], axis=1).mean())
        print(f"\n=== горизонт {hz} шагов ===")
        print(f"  средняя величина прихода: {scale:.4f}")
        print(f"  путь КОДА: ошибка прихода {err_code:.4f} "
              f"({err_code / scale:.1%} от величины), "
              f"согласие с кодами {acc_code:.1%}")

        row = dict(scale=scale, err_code=err_code, acc_code=acc_code,
                   code_source=code_src, goals={})
        for K in [int(x) for x in args.k_goals.split(",")]:
            cent = kmeans(g_true[itr], K, seed=0)
            lab = np.empty(N, np.int64)
            for i0 in range(0, N, 20000):
                gb = g_true[i0:i0 + 20000]
                lab[i0:i0 + 20000] = ((gb[:, None, :] - cent[None]) ** 2
                                      ).sum(-1).argmin(1)
            # ПОЛ КВАНТОВАНИЯ ЦЕЛИ: ошибка при ИДЕАЛЬНОМ предсказании метки.
            floor = float(np.linalg.norm(cent[lab[ite]] - g_true[ite],
                                         axis=1).mean())
            # БАЗА: ЦЕЛЬ, ОБУСЛОВЛЕННАЯ ТОЛЬКО ЗАДАЧЕЙ. Без неё непонятно,
            # добавляет ли h12 хоть что-то сверх знания, какая это задача:
            # голова могла бы выучить «в этой задаче обычно тянутся вот
            # сюда» и показать низкую ошибку, ничего не увидев в картинке.
            tsk = z["task"]
            base_lab = {}
            for t_ in np.unique(tsk[itr]):
                m = itr[tsk[itr] == t_]
                gm = g_true[m].mean(0)
                base_lab[str(t_)] = int(((cent - gm) ** 2).sum(-1).argmin())
            bl = np.array([base_lab.get(str(t_), 0) for t_ in tsk[ite]])
            err_base = float(np.linalg.norm(cent[bl] - g_true[ite],
                                            axis=1).mean())
            acc_base = float((bl == lab[ite]).mean())

            best = None
            for lr in [float(x) for x in args.lrs.split(",")]:
                for sd in [int(x) for x in args.seeds.split(",")]:
                    torch.manual_seed(sd)
                    din = N_POS * D if args.pool == "flat" else D
                    net = nn.Linear(din, K).to(dev)
                    opt = torch.optim.AdamW(net.parameters(), lr=lr)
                    rng = np.random.default_rng(sd)

                    def fwd(idx):
                        h = torch.from_numpy(np.asarray(H[idx])).to(dev).float()
                        h = h.reshape(len(idx), -1) if args.pool == "flat" \
                            else h.mean(1)
                        return net(h)

                    @torch.no_grad()
                    def ev(idx):
                        net.eval()
                        pl, acc, n = [], 0.0, 0
                        for i0 in range(0, len(idx), args.batch):
                            s = idx[i0:i0 + args.batch]
                            p = fwd(s).argmax(-1).cpu().numpy()
                            pl.append(p)
                            acc += float((p == lab[s]).mean()) * len(s)
                            n += len(s)
                        p = np.concatenate(pl)
                        e = float(np.linalg.norm(cent[p] - g_true[idx],
                                                 axis=1).mean())
                        return dict(err=e, acc=acc / n)

                    # ЭПОХА 0 — ПОЛНОПРАВНЫЙ КАНДИДАТ.
                    cur = dict(ev(iva), epoch=0, lr=lr, seed=sd)
                    st_best = {k: v.detach().clone()
                               for k, v in net.state_dict().items()}
                    for ep in range(1, args.epochs + 1):
                        net.train()
                        order = rng.permutation(len(itr))
                        for i0 in range(0, len(order), args.batch):
                            s = itr[np.sort(order[i0:i0 + args.batch])]
                            loss = F.cross_entropy(
                                fwd(s), torch.from_numpy(lab[s]).to(dev))
                            opt.zero_grad(set_to_none=True)
                            loss.backward(); opt.step()
                        e = ev(iva)
                        if e["err"] < cur["err"]:
                            cur = dict(e, epoch=ep, lr=lr, seed=sd)
                            st_best = {k: v.detach().clone()
                                       for k, v in net.state_dict().items()}
                    if best is None or cur["err"] < best[0]["err"]:
                        net.load_state_dict(st_best)
                        best = (cur, ev(ite))
                    print(f"    K={K} lr={lr:g} сид={sd}: валидация "
                          f"{cur['err']:.4f} на эпохе {cur['epoch']}",
                          flush=True)

            cfg, te = best
            gain = rel_gain(te["err"], err_code)
            over_base = rel_gain(te["err"], err_base)
            row["goals"][str(K)] = dict(
                floor=floor, err_task_baseline=err_base, acc_task_baseline=acc_base,
                gain_over_baseline=over_base, val=cfg, test=te, gain=gain,
                verdict=read_rule(gain, args.gain_good, args.gain_bad))
            print(f"  путь ЦЕЛИ K={K}: ошибка прихода {te['err']:.4f} "
                  f"({te['err'] / scale:.1%}), точность {te['acc']:.1%}, "
                  f"пол квантования {floor:.4f}")
            print(f"    база «только задача»: {err_base:.4f}, точность "
                  f"{acc_base:.1%}; h12 сверх базы "
                  + (f"{over_base:+.1%}" if over_base is not None else "н/д"))
            print(f"    выигрыш против пути кода: "
                  + (f"{gain:+.1%}" if gain is not None else "н/д")
                  + f" — {row['goals'][str(K)]['verdict']}")
            if over_base is not None and over_base < 0.05:
                print("    ВНИМАНИЕ: h12 почти не улучшает базу «только "
                      "задача» — голова выучила среднее по задаче, а не "
                      "увидела сцену")
        results[str(hz)] = row

    print(f"\n  ЧИТАТЬ по горизонту 8 — это наше исполняемое окно. Вердикт "
          f"выносится по\n  ОШИБКЕ ПРИХОДА, а не по точности: 64-путевой "
          f"выбор и 2048-путевой\n  несравнимы по точности напрямую.")
    if not args.rstar:
        print("  ВНИМАНИЕ: путь кода взят по кодам ПОЛНОЙ глубины и завышен; "
              "для честного\n  сравнения нужен --rstar.")
    p = os.path.join(args.out, "table.json")
    json.dump(dict(results=results, script_sha1=sha, h12=md, ckpt=args.ckpt,
                   argv=vars(args)), open(p, "w"), ensure_ascii=False, indent=1)
    print(f"\n  сохранено: {p}")


if __name__ == "__main__":
    main()
