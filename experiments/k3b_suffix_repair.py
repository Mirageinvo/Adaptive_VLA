"""K-3b (Gate B0, исправленный): ремонт fine-суффикса после ПРАВКИ coarse.

ЧТО БЫЛО НЕ ТАК В K-3. Три независимых дефекта делали главное число
(R_BAR = 0.36) непригодным:

  1. РАЗНЫЕ ЦЕЛИ В ОДНОМ ОТНОШЕНИИ. Жадный вариант строился к латенте
     ИСТИННОГО датасетного действия, а мерилось всё расстоянием до
     НЕВОЗМУЩЁННОГО выхода BAR. Числитель и знаменатель относились к разным
     величинам.
  2. НЕВЕРНЫЙ КОНТЕКСТ У ЛОКАЛЬНОГО РЕМОНТА. Уровень 2 предсказывался при
     ГЛОБАЛЬНО пересчитанном уровне 1, а в кортеж вставлялась одна позиция.
     То есть код уровня 2 приходил из контекста, которого в итоге нет.
  3. ОБРАТНОЕ НАПРАВЛЕНИЕ. Бралcя жадный top-1 код и заменялся альтернативой
     ранга 2-5 — это ПОРЧА хорошего решения. Поток же исправляет ПЛОХОЕ
     решение на лучшее.

ПОСТАНОВКА K-3b. Воспроизводим то, что делает итеративный генератор:

  z_ref   — жадная генерация BAR целиком. Это «правильное» решение: в
              позиции p стоит top-1 код v, и суффикс согласован с ним.
  u         — код рангов 2..k в позиции p: РАННЕЕ, ХУДШЕЕ решение.
  z_old     — полная генерация при u: согласованное старое состояние.
  правка    — поток меняет u на v.

  stale     — новый coarse v, суффикс от z_old (устаревший);
  BAR лок.  — пересчёт ТОЛЬКО позиции p, уровень 2 при ЛОКАЛЬНО обновлённом
              уровне 1 (исправление дефекта 2);
  BAR глоб. — полная перегенерация обоих тонких блоков;
  проекция  — жадная переквантизация суффикса в позиции p К ЦЕЛИ.

ОДНА ЦЕЛЬ ВЕЗДЕ. Целью служит z_ref — то, что модель выдала бы, приняв
верное решение с самого начала. К ней строится проекция, ею же меряется
качество. Отдельно, НЕ смешивая внутри отношения, приводится качество против
датасетного действия на исполняемом окне.

Заметим: BAR глоб. при верном coarse воспроизводит z_ref тождественно, то
есть даёт нулевую ошибку ценой полной перегенерации. Поэтому вопрос ставится
так: насколько ЛОКАЛЬНЫЙ ремонт приближается к тому, что даёт полная
перегенерация, и обходит ли явная проекция собственный пересчёт модели.

    доля закрытого разрыва = (e_stale - e_вариант) / e_stale

ПРАВИЛО ЧТЕНИЯ, зафиксировано до запуска:
  проекция закрывает существенно больше, чем BAR лок. -> явное связывание
      оправдано, есть что добавлять к модели;
  BAR лок. закрывает столько же или больше -> модель чинит сама, механизм
      беспредметен;
  обе доли близки к нулю -> суффикс не компенсирует смену coarse вообще, и
      направление закрывается.

ЧТО ЕЩЁ ИСПРАВЛЕНО: парные разности вместо разности медиан; кластерный
бутстрап по эпизодам с сохранением пар; штатные position_offset; окно
исполнения 4 (умолчание их eval_libero) с кривой по 1,2,4,8,20; JS
агрегируется в фактической позиции p, а не максимумом по всему; матрица
влияния нормируется на число выпадений каждой позиции, отдельно по режимам.

Запуск:
    python3 experiments/k3b_suffix_repair.py \
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1_residual_cost import latent_from_codes, projected_codebooks  # noqa: E402
from k3_bar_suffix_repair import (  # noqa: E402
    MAX_ACTION_Q,
    STATE_Q01,
    STATE_Q99,
    build_batch,
    greedy_suffix,
    js_div,
    load_lerobot,
)

WINDOWS = (1, 2, 4, 8, 20)


def paired_ci(delta: np.ndarray, base: np.ndarray, epi: np.ndarray,
              n_boot: int = 400, seed: int = 0):
    """Доля закрытого разрыва по ПАРНЫМ разностям, с кластерным бутстрапом.

    delta[i] = e_stale[i] - e_вариант[i], base[i] = e_stale[i]. Отношение
    берётся как сумма/сумма внутри реплики: индивидуальные отношения неустойчивы
    при знаменателе около нуля."""
    rng = np.random.default_rng(seed)
    eps = np.unique(epi)
    idx = {e: np.where(epi == e)[0] for e in eps}
    point = delta.sum() / max(base.sum(), 1e-12)
    out = []
    for _ in range(n_boot):
        s = np.concatenate([idx[e] for e in rng.choice(eps, len(eps), replace=True)])
        out.append(delta[s].sum() / max(base[s].sum(), 1e-12))
    return point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-obs", type=int, default=96)
    ap.add_argument("--all-pos", action="store_true", default=True,
                    help="проходить ВСЕ позиции, а не случайные с возвращением")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="граница практически значимого преимущества проекции, "
                         "в долях закрытого разрыва")
    ap.add_argument("--rank-lo", type=int, default=1, help="нижний ранг худшего кода")
    ap.add_argument("--rank-hi", type=int, default=5, help="верхний ранг")
    ap.add_argument("--seeds", default="0", help="сиды выборки данных")
    ap.add_argument("--pos-offsets", default="0,3,4",
                    help="штатные смещения позиций; обучение шло со случайными, "
                         "валидация с 3, симулятор с 4")
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--center-crop", action="store_true", default=True)
    ap.add_argument("--tiled", action="store_true", default=True)
    ap.add_argument("--source", default="lerobot")
    ap.add_argument("--flip", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.root))
    import actioncodec  # noqa: F401  регистрирует action_codec в AutoModel
    import importlib.util

    sp = importlib.util.spec_from_file_location(
        "ac_vla_tok", os.path.join(os.path.abspath(args.root), "utils",
                                   "vla_tokenizer.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    proc = m.VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    tok = proc.action_processor.to(args.device).eval()
    L, P = tok.num_quantizers, tok.n_tokens_per_quantizer
    cfg = tok.config.embodiment_config[
        list(tok.config.embodiment_config.keys())[args.embodiment]]
    T, D_act = int(cfg["freq"] * cfg["duration"]), cfg["action_dim"]
    E = projected_codebooks(tok, args.device)
    print(f"кодек: словарь {tok.vocab_size}, уровней {L}, позиций {P}")

    IM1, IM2, ST_RAW, A, PREV, tasks, EPI = load_lerobot(args.n_obs, T)
    A = np.asarray(A, np.float32).copy()
    A[..., :-1] = A[..., :-1] / MAX_ACTION_Q[:-1]
    A[..., -1] = -A[..., -1]
    a_true = torch.from_numpy(np.clip(A, -1.0, 1.0)).to(args.device)
    scale = float(a_true.max() - a_true.min())
    B = len(A)
    st = ((ST_RAW - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0).astype(np.float32)

    from smolvla.bar import SmolVLABlockwiseAR

    model = SmolVLABlockwiseAR.from_pretrained(
        args.ckpt, trust_remote_code=True, dtype=torch.bfloat16,
        token_budget=P * L, num_blocks=L,
        action_vocab_size=tok.vocab_size).to(args.device).eval()
    bs, nb = model.block_size, model.num_blocks
    assert bs * nb == P * L
    bpl = P // bs

    batch = build_batch(IM1, IM2, tasks, st, proc, args, args.device)
    with torch.no_grad():
        _, vlm_len, VLM, _ = model._build_vlm_inputs_embeds(
            input_ids=batch["input_ids"], inputs_embeds=None,
            pixel_values=batch.get("pixel_values"),
            pixel_attention_mask=batch.get("pixel_attention_mask"),
            image_hidden_states=None)
    print(f"наблюдений {B}, эпизодов {len(np.unique(EPI))}, "
          f"VLM {tuple(VLM.shape)}\n")

    def dec(codes):
        return tok._decode(latent_from_codes(E, codes), args.embodiment,
                           None)[0][..., :D_act]

    def metrics(a, ref, w):
        """Три величины. Захват СЧИТАЕТСЯ ОТДЕЛЬНО: он бинарный, в общей норме
        теряется, а момент его переключения может определять успех задачи."""
        d = (a[:, :w] - ref[:, :w]).abs()
        cont = d[..., :D_act - 1]
        return (cont.flatten(1).amax(-1) / scale,
                cont.flatten(1).pow(2).mean(-1).sqrt() / scale,
                ((a[:, :w, -1] > 0) != (ref[:, :w, -1] > 0)).float().mean(-1))

    def to_levels(flat):
        return flat.reshape(-1, L, P).transpose(1, 2)

    for off in [int(x) for x in args.pos_offsets.split(",")]:
        print("=" * 80)
        print(f"POSITION_OFFSET = {off}")
        print("=" * 80)

        def blk(hist):
            """Логиты следующего блока со ШТАТНЫМИ позиционными id."""
            alen = bs + (0 if hist is None else hist.shape[1])
            apos = model._build_action_pos_ids_strided(
                batch_size=B, base_pos=vlm_len, action_seq_len=alen,
                device=VLM.device, position_offset=off)
            pids = model._build_joint_position_ids(
                batch_size=B, vlm_seq_len=vlm_len, action_pos_ids=apos,
                device=VLM.device)
            return model._predict_next_block_logits(
                vlm_inputs_embeds=VLM, attention_mask=batch.get("attention_mask"),
                history_tokens=hist, position_ids=pids).float()

        def gen(hist, n):
            for _ in range(n):
                c = blk(hist).argmax(-1)
                hist = c if hist is None else torch.cat([hist, c], 1)
            return hist

        with torch.no_grad():
            z_ref = to_levels(gen(None, nb))          # жадный top-1 выход BAR
            a_ref = dec(z_ref)
            lg0 = blk(None)
            rng = torch.Generator(device=args.device).manual_seed(1)

            VAR = ("old", "stale", "loc", "glob", "proj")
            acc = {k: {w: {m: [] for m in ("max", "rms", "grip")}
                       for w in WINDOWS} for k in VAR}
            acc_gt = {k: {w: {m: [] for m in ("max", "rms", "grip")}
                          for w in WINDOWS} for k in VAR + ("ref",)}
            js_p, chg_all = [], []
            infl = torch.zeros(P, P)
            cnt = torch.zeros(P)
            ar = torch.arange(B, device=args.device)
            plist = list(range(P)) if args.all_pos else [
                int(torch.randint(P, (1,), generator=rng, device=args.device))
                for _ in range(P)]

            for p_ in plist:
                v = z_ref[:, p_, 0]
                rk = lg0[:, p_].topk(args.rank_hi, -1).indices
                u = rk[ar, torch.randint(args.rank_lo, args.rank_hi, (B,),
                                         generator=rng, device=args.device)]

                c0_old = z_ref[:, :, 0].clone()
                c0_old[:, p_] = u
                lg1_old = blk(c0_old)
                c1_old = lg1_old.argmax(-1)
                lg2_old = blk(torch.cat([c0_old, c1_old], 1))
                z_old = torch.stack([c0_old, c1_old, lg2_old.argmax(-1)], -1)

                c0_new = z_ref[:, :, 0]
                stale = z_old.clone()
                stale[:, :, 0] = c0_new

                lg1 = blk(c0_new)
                c1 = z_old[:, :, 1].clone()
                c1[:, p_] = lg1.argmax(-1)[:, p_]
                lg2 = blk(torch.cat([c0_new, c1], 1))
                c2 = z_old[:, :, 2].clone()
                c2[:, p_] = lg2.argmax(-1)[:, p_]
                loc = torch.stack([c0_new, c1, c2], -1)

                glob = to_levels(gen(c0_new, nb - bpl))
                assert torch.equal(glob, z_ref), (
                    "перегенерация при верном coarse не дала z_ref — "
                    "генерация недетерминирована или контекст сбит")
                proj = greedy_suffix(E, stale, latent_from_codes(E, z_ref), p_, 0)

                js_p.append(float(js_div(lg1_old.softmax(-1)[:, p_],
                                         lg1.softmax(-1)[:, p_]).median()))
                ch = (c1_old != lg1.argmax(-1)).float().mean(0)
                infl[p_] += ch.cpu()
                cnt[p_] += 1
                chg_all.append(float(ch.mean()))

                dd = {k: dec(cc) for k, cc in
                      (("old", z_old), ("stale", stale), ("loc", loc),
                       ("glob", glob), ("proj", proj))}
                for w in WINDOWS:
                    for k, a_ in dd.items():
                        for nm, val in zip(("max", "rms", "grip"),
                                           metrics(a_, a_ref, w)):
                            acc[k][w][nm].append(val.cpu().numpy())
                    for k, a_ in list(dd.items()) + [("ref", a_ref)]:
                        for nm, val in zip(("max", "rms", "grip"),
                                           metrics(a_, a_true, w)):
                            acc_gt[k][w][nm].append(val.cpu().numpy())

        epi_rep = np.tile(EPI, len(plist))

        print("ЦЕЛЬ — z_ref (жадный top-1 выход BAR; НЕ истина)")
        print(f"{'окно':>5}{'норма':>6}{'z_old':>8}{'stale':>8}{'BAR лок.':>9}"
              f"{'глоб.':>7}{'проекц.':>8}{'закрыто BAR':>24}"
              f"{'закрыто проекц.':>24}{'преим. проекции':>24}")
        for w in WINDOWS:
            for nm in ("max", "rms"):
                e = {k: np.concatenate(acc[k][w][nm]) for k in VAR}
                rb = paired_ci(e["stale"] - e["loc"], e["stale"], epi_rep)
                rp = paired_ci(e["stale"] - e["proj"], e["stale"], epi_rep)
                # ПРЯМОЙ ПАРНЫЙ ТЕСТ ЭКВИВАЛЕНТНОСТИ. Перекрытие интервалов rb и
                # rp равенства НЕ доказывает; бутстрапим саму разность.
                ad = paired_ci(e["loc"] - e["proj"], e["stale"], epi_rep)
                print(f"{w:>5}{nm:>6}{e['old'].mean():>8.4f}{e['stale'].mean():>8.4f}"
                      f"{e['loc'].mean():>9.4f}{e['glob'].mean():>7.4f}"
                      f"{e['proj'].mean():>8.4f}"
                      f"{f'{rb[0]:+.2f} [{rb[1]:+.2f},{rb[2]:+.2f}]':>24}"
                      f"{f'{rp[0]:+.2f} [{rp[1]:+.2f},{rp[2]:+.2f}]':>24}"
                      f"{f'{ad[0]:+.3f} [{ad[1]:+.3f},{ad[2]:+.3f}]':>24}")

        print("\nЦЕЛЬ — датасетное действие (качество), с интервалами")
        print(f"{'окно':>5}{'норма':>6}{'z_ref':>8}{'z_old':>8}{'stale':>8}"
              f"{'BAR лок.':>9}{'проекц.':>8}{'z_old -> z_ref улучшает?':>30}")
        for w in WINDOWS:
            for nm in ("max", "rms"):
                e = {k: np.concatenate(acc_gt[k][w][nm]) for k in acc_gt}
                # ПРОВЕРКА ПРЕДПОСЫЛКИ: правка coarse должна улучшать качество,
                # иначе замер меряет согласованность, а не исправление.
                imp = paired_ci(e["old"] - e["ref"], e["old"], epi_rep)
                print(f"{w:>5}{nm:>6}{e['ref'].mean():>8.4f}{e['old'].mean():>8.4f}"
                      f"{e['stale'].mean():>8.4f}{e['loc'].mean():>9.4f}"
                      f"{e['proj'].mean():>8.4f}"
                      f"{f'{imp[0]:+.3f} [{imp[1]:+.3f},{imp[2]:+.3f}]':>30}")

        print("\nЗАХВАТ отдельно (доля неверных шагов), окно 4")
        eg = {k: np.concatenate(acc_gt[k][4]["grip"]).mean() for k in acc_gt}
        print("  " + "  ".join(f"{k}={eg[k]:.3f}" for k in
                               ("ref", "old", "stale", "loc", "proj")))

        infl = infl / cnt.clamp_min(1).unsqueeze(1)
        d_ = float(np.mean([infl[i, i] for i in range(P) if cnt[i] > 0]))
        o_ = float(np.mean([infl[i, q] for i in range(P) if cnt[i] > 0
                            for q in range(P) if q != i]))
        print(f"\nJS в изменённой позиции: медиана {np.median(js_p):.4f} бит, "
              f"смен top-1 {np.mean(chg_all):.1%}")
        print(f"влияние: диагональ {d_:.3f}, вне диагонали {o_:.3f}, "
              f"отношение {d_ / max(o_, 1e-9):.1f}\n")

    print(f"""
КАК ЧИТАТЬ.

ГЛАВНОЕ — столбец «преим. проекции» = (e_BAR_лок - e_проекц) / e_stale по
парным разностям. Перекрытие интервалов двух отдельных долей равенства НЕ
доказывает, поэтому бутстрапится сама разность. Граница практической
значимости зафиксирована до запуска: {args.margin:.2f} закрытого разрыва.
  верхняя граница интервала НИЖЕ {args.margin:.2f} -> жадная проекция практически
      эквивалентна обычному условному пересчёту, отдельный механизм не нужен;
  интервал уходит выше -> преимущество есть, механизм имеет смысл.

ПРОВЕРКА ПРЕДПОСЫЛКИ — столбец «z_old -> z_ref улучшает?». z_ref это лишь
жадный top-1 выход BAR, а не истина, и код ранга 2-5 не обязан быть ошибкой.
Если переход не улучшает качество относительно датасета, замер меряет
восстановление внутренней согласованности, а не исправление решения.

ВНИМАНИЕ ПРО ВЫЧИСЛЕНИЯ. BAR лок. и BAR глоб. зовут модель ОДИНАКОВОЕ число
раз (по разу на уровень). Трансформер считает логиты для всего блока, и
вставка одной позиции не экономит ни проходов, ни операций. Поэтому локальный
ремонт при равной цене даёт лишь часть того, что даёт полная перегенерация, и
компромисса «вычисления против качества» здесь НЕТ. Он появился бы только при
настоящей разреженной архитектуре. «закрыто» = (e_stale - e_вариант) / e_stale по ПАРНЫМ разностям,
отношение сумм внутри бутстрап-реплики, интервалы кластерные по эпизодам.
  проекция закрывает существенно больше BAR -> явное связывание оправдано;
  BAR не меньше проекции -> модель чинит сама, механизм беспредметен;
  обе доли около нуля -> суффикс не компенсирует смену coarse вообще.
BAR глоб. служит опорой и встроенной проверкой: при верном coarse он обязан
воспроизвести z_ref тождественно.""")


if __name__ == "__main__":
    main()
