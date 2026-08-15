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

  z_ideal   — жадная генерация BAR целиком. Это «правильное» решение: в
              позиции p стоит top-1 код v, и суффикс согласован с ним.
  u         — код рангов 2..k в позиции p: РАННЕЕ, ХУДШЕЕ решение.
  z_old     — полная генерация при u: согласованное старое состояние.
  правка    — поток меняет u на v.

  stale     — новый coarse v, суффикс от z_old (устаревший);
  BAR лок.  — пересчёт ТОЛЬКО позиции p, уровень 2 при ЛОКАЛЬНО обновлённом
              уровне 1 (исправление дефекта 2);
  BAR глоб. — полная перегенерация обоих тонких блоков;
  проекция  — жадная переквантизация суффикса в позиции p К ЦЕЛИ.

ОДНА ЦЕЛЬ ВЕЗДЕ. Целью служит z_ideal — то, что модель выдала бы, приняв
верное решение с самого начала. К ней строится проекция, ею же меряется
качество. Отдельно, НЕ смешивая внутри отношения, приводится качество против
датасетного действия на исполняемом окне.

Заметим: BAR глоб. при верном coarse воспроизводит z_ideal тождественно, то
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
    ap.add_argument("--n-pos", type=int, default=8)
    ap.add_argument("--rank-lo", type=int, default=1, help="нижний ранг худшего кода")
    ap.add_argument("--rank-hi", type=int, default=5, help="верхний ранг")
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
            z_ideal = to_levels(gen(None, nb))           # верное решение
            a_ideal = dec(z_ideal)
            lg0 = blk(None)
            rng = torch.Generator(device=args.device).manual_seed(1)

            acc = {k: {w: [] for w in WINDOWS} for k in
                   ("stale", "loc", "glob", "proj")}
            acc_gt = {k: {w: [] for w in WINDOWS} for k in
                      ("ideal", "stale", "loc", "glob", "proj")}
            js_p, chg_all = [], []
            infl = torch.zeros(P, P)
            cnt = torch.zeros(P)
            ar = torch.arange(B, device=args.device)

            for _ in range(args.n_pos):
                p = int(torch.randint(P, (1,), generator=rng, device=args.device))
                v = z_ideal[:, p, 0]                     # top-1: верный код
                # индекс 0 — сам top-1, поэтому худший код берём из индексов
                # [rank_lo, rank_hi), что соответствует рангам rank_lo+1..rank_hi
                rk = lg0[:, p].topk(args.rank_hi, -1).indices
                u = rk[ar, torch.randint(args.rank_lo, args.rank_hi, (B,),
                                         generator=rng, device=args.device)]

                # согласованное СТАРОЕ состояние при худшем коде u
                c0_old = z_ideal[:, :, 0].clone()
                c0_old[:, p] = u
                lg1_old = blk(c0_old)                    # кэш: нужен и для JS
                c1_old = lg1_old.argmax(-1)
                lg2_old = blk(torch.cat([c0_old, c1_old], 1))
                z_old = torch.stack([c0_old, c1_old, lg2_old.argmax(-1)], -1)

                # поток исправляет u -> v
                c0_new = z_ideal[:, :, 0]
                stale = z_old.clone()
                stale[:, :, 0] = c0_new

                # ЛОКАЛЬНЫЙ ремонт с ПОСЛЕДОВАТЕЛЬНЫМ контекстом
                lg1 = blk(c0_new)
                c1 = z_old[:, :, 1].clone()
                c1[:, p] = lg1.argmax(-1)[:, p]
                lg2 = blk(torch.cat([c0_new, c1], 1))
                c2 = z_old[:, :, 2].clone()
                c2[:, p] = lg2.argmax(-1)[:, p]
                loc = torch.stack([c0_new, c1, c2], -1)

                glob = to_levels(gen(c0_new, nb - bpl))
                # ВСТРОЕННАЯ ПРОВЕРКА: жадная генерация детерминирована, и при
                # верном coarse полная перегенерация обязана дать ровно цель.
                assert torch.equal(glob, z_ideal), (
                    "полная перегенерация при верном coarse не воспроизвела "
                    "z_ideal — генерация недетерминирована или контекст сбит")
                proj = greedy_suffix(E, stale, latent_from_codes(E, z_ideal), p, 0)

                # чувствительность ИМЕННО в изменённой позиции
                js_p.append(float(js_div(lg1_old.softmax(-1)[:, p],
                                         lg1.softmax(-1)[:, p]).median()))
                ch = (c1_old != lg1.argmax(-1)).float().mean(0)
                infl[p] += ch.cpu()
                cnt[p] += 1
                chg_all.append(float(ch.mean()))

                for k, cc in (("stale", stale), ("loc", loc), ("glob", glob),
                              ("proj", proj)):
                    d = (dec(cc) - a_ideal).abs()[..., :D_act - 1]
                    for w in WINDOWS:
                        acc[k][w].append((d[:, :w].flatten(1).amax(-1)
                                          / scale).cpu().numpy())
                for k, cc in (("ideal", z_ideal), ("stale", stale), ("loc", loc),
                              ("glob", glob), ("proj", proj)):
                    d = (dec(cc) - a_true).abs()[..., :D_act - 1]
                    for w in WINDOWS:
                        acc_gt[k][w].append((d[:, :w].flatten(1).amax(-1)
                                             / scale).cpu().numpy())

        epi_rep = np.tile(EPI, args.n_pos)
        print("ЦЕЛЬ — z_ideal (что модель выдала бы, приняв верное решение сразу)")
        print(f"{'окно':>6}{'stale':>9}{'BAR лок.':>10}{'BAR глоб.':>11}"
              f"{'проекция':>11}{'закрыто BAR':>26}{'закрыто проекцией':>26}")
        for w in WINDOWS:
            e = {k: np.concatenate(acc[k][w]) for k in acc}
            rb = paired_ci(e["stale"] - e["loc"], e["stale"], epi_rep)
            rp = paired_ci(e["stale"] - e["proj"], e["stale"], epi_rep)
            print(f"{w:>6}{e['stale'].mean():>9.4f}{e['loc'].mean():>10.4f}"
                  f"{e['glob'].mean():>11.4f}{e['proj'].mean():>11.4f}"
                  f"{f'{rb[0]:+.2f} [{rb[1]:+.2f}, {rb[2]:+.2f}]':>26}"
                  f"{f'{rp[0]:+.2f} [{rp[1]:+.2f}, {rp[2]:+.2f}]':>26}")

        print("\nЦЕЛЬ — датасетное действие (качество, отдельно от отношения выше)")
        print(f"{'окно':>6}{'ideal':>9}{'stale':>9}{'BAR лок.':>10}"
              f"{'BAR глоб.':>11}{'проекция':>11}")
        for w in WINDOWS:
            e = {k: np.concatenate(acc_gt[k][w]).mean() for k in acc_gt}
            print(f"{w:>6}{e['ideal']:>9.4f}{e['stale']:>9.4f}{e['loc']:>10.4f}"
                  f"{e['glob']:>11.4f}{e['proj']:>11.4f}")

        infl = infl / cnt.clamp_min(1).unsqueeze(1)
        d_ = float(np.mean([infl[i, i] for i in range(P) if cnt[i] > 0]))
        o_ = float(np.mean([infl[i, q] for i in range(P) if cnt[i] > 0
                            for q in range(P) if q != i]))
        print(f"\nJS в изменённой позиции p: медиана {np.median(js_p):.4f} бит, "
              f"смен top-1 всего {np.mean(chg_all):.1%}")
        print(f"влияние: диагональ {d_:.3f}, вне диагонали {o_:.3f}, "
              f"отношение {d_ / max(o_, 1e-9):.1f}")
        print("(нормировка на число выпадений каждой позиции; внедиагональ по "
              "ВСЕМ q != p)\n")

    print("""
КАК ЧИТАТЬ. «закрыто» = (e_stale - e_вариант) / e_stale по ПАРНЫМ разностям,
отношение сумм внутри бутстрап-реплики, интервалы кластерные по эпизодам.
  проекция закрывает существенно больше BAR -> явное связывание оправдано;
  BAR не меньше проекции -> модель чинит сама, механизм беспредметен;
  обе доли около нуля -> суффикс не компенсирует смену coarse вообще.
BAR глоб. служит опорой: при верном coarse он воспроизводит цель тождественно,
то есть показывает, чего стоит добиться ценой полной перегенерации.""")


if __name__ == "__main__":
    main()
