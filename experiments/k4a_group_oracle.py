"""K-4a: групповой оракул — какую группу позиций надо пересчитать.

ПЕРЕФОРМУЛИРОВКА. Прежняя мотивация (восстановление остаточной семантики)
опровергнута: декодер видит только сумму, отдельного штрафа за состав кортежа
нет, явная проекция не превосходит условный пересчёт. Новый вопрос:

    можно ли после пересмотра части плана определить НЕБОЛЬШУЮ ГРУППУ
    действительно затронутых временных позиций и пересчитать только её?

Основание. Пересчёт ТОЛЬКО изменённой позиции закрывает ~14% разрыва, а лучшая
ОДНА позиция — ~50%. Значит важнейшая позиция часто лежит в другом месте
последовательности. Отбор по JS при четырёх позициях даёт ~51% против ~16% у
случайного: зависимость нелокальна, но частично предсказуема.

ЗАЧЕМ ГРУППОВОЙ ОРАКУЛ, А НЕ ПРЕЖНИЙ. Прежний ранжировал позиции ОДИН раз по
их одиночному вкладу. Кривая вышла с плато: 1 -> 0.50, 2 -> 0.67, 4 -> 0.77,
8 -> 0.79, 16 -> 1.00. Такое плато означает НЕАДДИТИВНОСТЬ: часть позиций
почти бесполезна поодиночке, но даёт эффект совместно. Одиночный оракул этого
не видит.

ЧТО СЧИТАЕМ:
  A. последовательный жадный оракул — предельный выигрыш пересчитывается
     ЗАНОВО после каждого выбора, поэтому часть взаимодействий учитывается;
  B. ТОЧНЫЙ перебор всех подмножеств размера <= 4 на подвыборке (2517
     комбинаций) — насколько жадный близок к настоящему оптимуму;
  C. матрица парных взаимодействий I(q,r) = G({q,r}) - G({q}) - G({r});
  D. непрерывные временные окна против произвольных наборов, ПРИ РАВНОМ ЧИСЛЕ
     позиций (иначе крупный блок получил бы больший бюджет и выглядел одним
     выбором).

ВЫЗОВЫ VLM НЕ НУЖНЫ. Оракул копирует коды из z_ref, поэтому после построения
stale вся работа — декодирование вариантов кодеком, который на три порядка
дешевле VLM.

ПРАВИЛО ЧТЕНИЯ, зафиксировано до запуска:
  точный оракул при 4 позициях закрывает >= 0.80 -> групповая разреженная
      ветка имеет запас, идти к router;
  жадный сохраняет >= 95% выигрыша точного -> дальше можно пользоваться
      жадным как дешёвым приближением;
  даже точный оракул требует 12+ позиций -> selective recomputation закрыть
      до обучения модели;
  сильные взаимодействия локальны по времени -> хватит фиксированных блоков;
      устойчивые дальние пары -> нужен обучаемый граф; диффузны -> не окупится.

Запуск:
    python3 experiments/k4a_group_oracle.py \
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO
"""

import argparse
import itertools
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
    js_div,
    load_lerobot,
)
from k3b_suffix_repair import paired_ci  # noqa: E402

BUDGETS = (1, 2, 4, 8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-obs", type=int, default=96)
    ap.add_argument("--exact-obs", type=int, default=16,
                    help="подвыборка для точного перебора")
    ap.add_argument("--exact-pos", type=int, default=4,
                    help="сколько позиций p брать для точного перебора")
    ap.add_argument("--rank-lo", type=int, default=1)
    ap.add_argument("--rank-hi", type=int, default=5)
    ap.add_argument("--pos-offset", type=int, default=3)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=4096, help="вариантов за раз")
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--center-crop", action="store_true", default=True)
    ap.add_argument("--tiled", action="store_true", default=True)
    ap.add_argument("--source", default="lerobot")
    ap.add_argument("--flip", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.root))
    import importlib.util

    import actioncodec  # noqa: F401

    sp = importlib.util.spec_from_file_location(
        "ac_vla_tok", os.path.join(os.path.abspath(args.root), "utils",
                                   "vla_tokenizer.py"))
    mm = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mm)
    proc = mm.VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    tok = proc.action_processor.to(args.device).eval()
    L, P = tok.num_quantizers, tok.n_tokens_per_quantizer
    cfg = tok.config.embodiment_config[
        list(tok.config.embodiment_config.keys())[args.embodiment]]
    T, D_act = int(cfg["freq"] * cfg["duration"]), cfg["action_dim"]
    E = projected_codebooks(tok, args.device)

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
    bpl = P // bs

    batch = build_batch(IM1, IM2, tasks, st, proc, args, args.device)
    with torch.no_grad():
        _, vlen, VLM, _ = model._build_vlm_inputs_embeds(
            input_ids=batch["input_ids"], inputs_embeds=None,
            pixel_values=batch.get("pixel_values"),
            pixel_attention_mask=batch.get("pixel_attention_mask"),
            image_hidden_states=None)

    def blk(hist):
        alen = bs + (0 if hist is None else hist.shape[1])
        apos = model._build_action_pos_ids_strided(
            batch_size=B, base_pos=vlen, action_seq_len=alen,
            device=VLM.device, position_offset=args.pos_offset)
        pids = model._build_joint_position_ids(
            batch_size=B, vlm_seq_len=vlen, action_pos_ids=apos, device=VLM.device)
        return model._predict_next_block_logits(
            vlm_inputs_embeds=VLM, attention_mask=batch.get("attention_mask"),
            history_tokens=hist, position_ids=pids).float()

    def gen(hist, n):
        for _ in range(n):
            c = blk(hist).argmax(-1)
            hist = c if hist is None else torch.cat([hist, c], 1)
        return hist

    def to_levels(f):
        return f.reshape(-1, L, P).transpose(1, 2)

    def dec_lat(h):
        """Декодирование латенты. Дробим на куски: вариантов бывают тысячи."""
        out = []
        for i in range(0, len(h), args.chunk):
            out.append(tok._decode(h[i:i + args.chunk], args.embodiment,
                                   None)[0][..., :D_act])
        return torch.cat(out)

    def err_lat(h, ref):
        d = (dec_lat(h)[:, :args.window] - ref[:, :args.window]).abs()[..., :D_act - 1]
        return d.flatten(1).pow(2).mean(-1).sqrt() / scale

    rng = torch.Generator(device=args.device).manual_seed(1)
    ar = torch.arange(B, device=args.device)
    print(f"наблюдений {B}, эпизодов {len(np.unique(EPI))}, "
          f"смещение {args.pos_offset}, окно {args.window}\n")

    with torch.no_grad():
        z_ref = to_levels(gen(None, nb))
        a_ref = dec_lat(latent_from_codes(E, z_ref))
        lg0 = blk(None)

        res = {k: {K: [] for K in BUDGETS} for k in
               ("singleton", "greedy", "same", "js", "window", "random")}
        es_all, inter_sum, inter_cnt = [], torch.zeros(P, P), 0
        exact_g, exact_greedy = [], []
        ex_pos = list(range(0, P, max(1, P // args.exact_pos)))[:args.exact_pos]

        for p_ in range(P):
            u = lg0[:, p_].topk(args.rank_hi, -1).indices[
                ar, torch.randint(args.rank_lo, args.rank_hi, (B,),
                                  generator=rng, device=args.device)]
            c0_old = z_ref[:, :, 0].clone()
            c0_old[:, p_] = u
            c1_old = blk(c0_old).argmax(-1)
            z_old = torch.stack([c0_old, c1_old,
                                 blk(torch.cat([c0_old, c1_old], 1)).argmax(-1)], -1)
            stale = z_old.clone()
            stale[:, :, 0] = z_ref[:, :, 0]

            h_st = latent_from_codes(E, stale)          # (B, P, D)
            h_rf = latent_from_codes(E, z_ref)
            e_st = err_lat(h_st, a_ref)
            es_all.append(e_st.cpu().numpy())

            def g_of(sets):
                """Выигрыш для списка наборов позиций: (n_sets, B)."""
                out = []
                for S in sets:
                    h = h_st.clone()
                    if S:
                        h[:, list(S)] = h_rf[:, list(S)]
                    out.append(e_st - err_lat(h, a_ref))
                return torch.stack(out)

            singles = g_of([(q,) for q in range(P)])     # (P, B)

            # ---------- A. последовательный жадный ----------
            chosen = [[] for _ in range(B)]
            cur = h_st.clone()
            e_cur = e_st.clone()
            greedy_gain = {}
            for step in range(max(BUDGETS)):
                best_g = torch.full((B,), -1e9, device=args.device)
                best_q = torch.zeros(B, dtype=torch.long, device=args.device)
                for q in range(P):
                    h = cur.clone()
                    h[:, q] = h_rf[:, q]
                    g = e_cur - err_lat(h, a_ref)
                    taken = torch.tensor([q in c for c in chosen],
                                         device=args.device)
                    g = torch.where(taken, torch.full_like(g, -1e9), g)
                    upd = g > best_g
                    best_g = torch.where(upd, g, best_g)
                    best_q = torch.where(upd, torch.full_like(best_q, q), best_q)
                cur[ar, best_q] = h_rf[ar, best_q]
                e_cur = e_cur - best_g
                for i in range(B):
                    chosen[i].append(int(best_q[i]))
                if step + 1 in BUDGETS:
                    greedy_gain[step + 1] = (e_st - e_cur).cpu().numpy()

            # ---------- прочие способы отбора ----------
            lg1 = blk(z_ref[:, :, 0])
            lg1_old = blk(c0_old)
            rank_js = js_div(lg1_old.softmax(-1), lg1.softmax(-1))
            rnd = torch.rand(B, P, generator=rng, device=args.device)
            for K in BUDGETS:
                res["greedy"][K].append(greedy_gain[K])
                res["singleton"][K].append(
                    _topk_gain(singles.T, K, h_st, h_rf, e_st, err_lat, a_ref, ar))
                res["js"][K].append(
                    _topk_gain(rank_js, K, h_st, h_rf, e_st, err_lat, a_ref, ar))
                res["random"][K].append(
                    _topk_gain(rnd, K, h_st, h_rf, e_st, err_lat, a_ref, ar))
                # только исходная позиция, добитая до K соседями по времени
                sm = torch.full((B, P), -1e9, device=args.device)
                sm[:, p_] = 1.0
                res["same"][K].append(
                    _topk_gain(sm, K, h_st, h_rf, e_st, err_lat, a_ref, ar)
                    if K == 1 else res["same"][1][-1])
                # лучшее НЕПРЕРЫВНОЕ окно длины K
                wins = [tuple(range(s, s + K)) for s in range(P - K + 1)]
                res["window"][K].append(g_of(wins).max(0).values.cpu().numpy())

            # ---------- C. парные взаимодействия ----------
            pairs = list(itertools.combinations(range(P), 2))
            gp = g_of(pairs)                              # (n_pairs, B)
            for idx, (q, r) in enumerate(pairs):
                v = float((gp[idx] - singles[q] - singles[r]).mean())
                inter_sum[q, r] += v
                inter_sum[r, q] += v
            inter_cnt += 1

            # ---------- B. точный перебор на подвыборке ----------
            if p_ in ex_pos:
                sub = slice(0, min(args.exact_obs, B))
                subsets = [S for k in range(1, 5)
                           for S in itertools.combinations(range(P), k)]
                hs, hr = h_st[sub], h_rf[sub]
                est = e_st[sub]
                best = torch.zeros(hs.shape[0], device=args.device)
                for i in range(0, len(subsets), 64):
                    blockS = subsets[i:i + 64]
                    hh = hs.unsqueeze(0).repeat(len(blockS), 1, 1, 1)
                    for j, S in enumerate(blockS):
                        hh[j][:, list(S)] = hr[:, list(S)]
                    ee = err_lat(hh.reshape(-1, P, hs.shape[-1]),
                                 a_ref[sub].repeat(len(blockS), 1, 1))
                    best = torch.maximum(best,
                                         (est.repeat(len(blockS))
                                          - ee).reshape(len(blockS), -1).max(0).values)
                exact_g.append(best.cpu().numpy())
                exact_greedy.append(greedy_gain[4][sub])

    es = np.concatenate(es_all)
    epi_rep = np.tile(EPI, P)

    print("=" * 78)
    print("ДОЛЯ РАЗРЫВА, ЗАКРЫТАЯ ПЕРЕСЧЁТОМ K ПОЗИЦИЙ (при РАВНОМ числе позиций)")
    print("=" * 78)
    names = {"greedy": "жадный послед.", "singleton": "одиночный оракул",
             "same": "только p", "js": "по JS", "window": "лучшее окно",
             "random": "случайно"}
    print(f"{'K':>3}" + "".join(f"{names[k]:>24}" for k in names))
    for K in BUDGETS:
        row = f"{K:>3}"
        for k in names:
            g = np.concatenate(res[k][K])
            pt, lo, hi = paired_ci(g, es, epi_rep)
            row += f"{f'{pt:.2f} [{lo:.2f},{hi:.2f}]':>24}"
        print(row)

    if exact_g:
        eg, gg = np.concatenate(exact_g), np.concatenate(exact_greedy)
        print("\n" + "=" * 78)
        print(f"ТОЧНЫЙ ПЕРЕБОР ПОДМНОЖЕСТВ <= 4 (позиций {len(ex_pos)}, "
              f"наблюдений {min(args.exact_obs, B)})")
        print("=" * 78)
        print(f"  точный оракул закрывает   {eg.sum() / es[:len(eg)].sum():.3f}")
        print(f"  жадный послед. закрывает  {gg.sum() / es[:len(gg)].sum():.3f}")
        print(f"  жадный сохраняет от точного {gg.sum() / max(eg.sum(), 1e-9):.1%}")

    M = (inter_sum / max(inter_cnt, 1)).numpy()
    off = M[~np.eye(P, dtype=bool)]
    dist = np.abs(np.subtract.outer(np.arange(P), np.arange(P)))
    near = M[(dist == 1)]
    far = M[(dist >= 4) & ~np.eye(P, dtype=bool)]
    print("\n" + "=" * 78)
    print("ПАРНЫЕ ВЗАИМОДЕЙСТВИЯ I(q,r) = G({q,r}) - G({q}) - G({r})")
    print("=" * 78)
    print(f"  среднее по всем парам      {off.mean():+.5f}")
    print(f"  соседние по времени (|q-r|=1) {near.mean():+.5f}")
    print(f"  дальние (|q-r|>=4)         {far.mean():+.5f}")
    print(f"  доля положительных          {(off > 0).mean():.1%}")
    print(f"  верхний дециль / медиана    {np.percentile(off, 90):+.5f} / "
          f"{np.median(off):+.5f}")
    print("""
Положительное взаимодействие означает, что позиции полезнее обновлять вместе.
  сильные взаимодействия у СОСЕДНИХ по времени -> хватит фиксированных блоков;
  сильные у ДАЛЬНИХ и устойчивые -> нужен обучаемый граф зависимостей;
  всё около нуля и размазано -> разреженная архитектура не окупится.""")

    print("""
ВОРОТА, зафиксированы до запуска.
  точный оракул при 4 позициях >= 0.80 -> запас есть, идти к router;
  жадный сохраняет >= 95% точного -> пользоваться жадным как приближением;
  даже точный требует 12+ позиций -> selective recomputation закрыть.
СРАВНЕНИЕ ЧЕСТНОЕ: все способы получают ОДИНАКОВОЕ число позиций, поэтому
непрерывное окно не получает большего бюджета под видом одного выбора.""")


def _topk_gain(score, K, h_st, h_rf, e_st, err_lat, a_ref, ar):
    """Выигрыш при пересчёте K позиций с наибольшим баллом."""
    sel = score.argsort(-1, descending=True)[:, :K]
    h = h_st.clone()
    idx = ar.unsqueeze(1).expand_as(sel)
    h[idx, sel] = h_rf[idx, sel]
    return (e_st - err_lat(h, a_ref)).cpu().numpy()


if __name__ == "__main__":
    main()
