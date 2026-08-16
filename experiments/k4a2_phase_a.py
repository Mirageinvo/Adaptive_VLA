"""K-4a, фаза A: устойчив ли групповой оракул и ЧЕМ объясняется его выигрыш.

K-4a дал 0.90 закрытого разрыва при четырёх позициях из шестнадцати против
0.77 у одиночного ранжирования. Прежде чем обучать router, надо закрыть три
вопроса, два из которых способны обнулить сам результат.

A0. СКОЛЬКО ПОЗИЦИЙ ВООБЩЕ ОТЛИЧАЮТСЯ. stale и z_ref расходятся только там,
    где правка coarse изменила тонкие уровни. Если расходятся всего пять
    позиций, то «четыре из шестнадцати» — это четыре из пяти, никакой
    разреженности нет, а случайный baseline занижен разбавлением по шестнадцати
    ячейкам. Этого замера в K-4a не было, и он проверяется первым.

A2. СИНЕРГИЯ ИЛИ ИЗБЫТОЧНОСТЬ. Прежняя матрица парных взаимодействий
    усреднялась по ВСЕМ 120 парам, а в большинстве пар хотя бы одна позиция
    почти ничего не даёт, и взаимодействие там нулевое по построению. Поэтому
    «0.7% одиночного выигрыша» почти ничего не значило. Правильная величина —
    неаддитивность ВЫБРАННОГО набора:

        Delta(S) = G(S) - sum_q G({q}),   q in S.

    Delta(одиночный top-4) сильно отрицательна, Delta(жадный) около нуля ->
        механизм ИЗБЫТОЧНОСТЬ: одиночный отбор берёт взаимозаменяемые позиции.
        Тогда независимый scorer со штрафом за похожесть должен почти догнать
        жадный, и вклад «условного последовательного выбора» слабый.
    Delta(жадный) заметно положительна -> механизм СИНЕРГИЯ: есть позиции,
        бесполезные поодиночке и полезные вместе. Это и есть содержательный
        случай для dependency-aware router.

A1. ТОЧНЫЙ ПЕРЕБОР НА ПОЛНОЙ ВЫБОРКЕ. Вместо нескольких стратифицированных
    подвыборок перебираем ВСЕ 2516 подмножеств размера <= 4 на всех
    наблюдениях и всех шестнадцати позициях вмешательства. Стоимость того же
    порядка, а вопрос о выборе подвыборки исчезает целиком.

A3. СТРУКТУРИРОВАННЫЕ ВРЕМЕННЫЕ НАБОРЫ. Один отрезок длины 4, два отрезка,
    отрезок плюс одиночки, произвольный набор — при ОДИНАКОВОМ числе позиций.
    Отдельно deployable-варианты, которым доступны только величины ДО нового
    плотного прохода.

A4. РАЗБИВКА по задаче, скорости, переключению схвата, амплитуде правки,
    исходной ошибке и позиции внутри горизонта.

A5. АБСОЛЮТНЫЙ МАСШТАБ. Всё меряется долей разрыва между stale и полным
    пересчётом. Если сам разрыв мал по сравнению с ошибкой модели относительно
    датасета, то закрывать его нечем и незачем. В плане этого замера нет.

ЧТО ДОСТУПНО ROUTER'У. Логиты lg_old = blk(z_ref[:,:,0]) получены проходом,
который УЖЕ состоялся до правки, поэтому их энтропия и запас top1-top2
допустимы. Логиты blk(c0_old) — это новый плотный проход, который мы и хотим
сэкономить; они и JS по ним разрешены только как оракульная верхняя граница.

Запуск:
    python3 experiments/k4a2_phase_a.py \
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


def seg_family(P: int, lens):
    """Наборы из непрерывных отрезков заданных длин, попарно неслипающихся.

    Между отрезками требуется хотя бы один пропуск, иначе два соседних отрезка
    склеиваются в один и семейство перестаёт отличаться от более простого.
    Перебираем все перестановки длин: иначе потерялись бы конфигурации, где
    короткий отрезок стоит раньше длинного."""
    res = set()

    def rec(perm, i, start, cur):
        if i == len(perm):
            res.add(tuple(sorted(cur)))
            return
        ln = perm[i]
        for s in range(start, P - ln + 1):
            rec(perm, i + 1, s + ln + 1, cur + list(range(s, s + ln)))

    for perm in set(itertools.permutations(lens)):
        rec(perm, 0, 0, [])
    return sorted(res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-obs", type=int, default=96)
    ap.add_argument("--n-ep", type=int, default=48,
                    help="эпизодов: кластеров для бутстрапа и для разбивки")
    ap.add_argument("--exact", type=int, default=1,
                    help="1 — полный перебор подмножеств <= 4 на ВСЕЙ выборке")
    ap.add_argument("--exact-obs", type=int, default=0,
                    help="0 — все наблюдения; иначе подвыборка (быстрая проверка)")
    ap.add_argument("--exact-block", type=int, default=32,
                    help="подмножеств за один вызов декодера")
    ap.add_argument("--max-pos", type=int, default=0,
                    help="0 — все 16 позиций вмешательства; иначе первые N "
                         "(только для быстрой проверки, не для отчёта)")
    ap.add_argument("--rank-lo", type=int, default=1)
    ap.add_argument("--rank-hi", type=int, default=5)
    ap.add_argument("--pos-offset", type=int, default=3)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--center-crop", action="store_true", default=True)
    ap.add_argument("--tiled", action="store_true", default=True)
    ap.add_argument("--source", default="lerobot")
    ap.add_argument("--flip", default="")
    ap.add_argument("--seed", type=int, default=0)
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

    IM1, IM2, ST_RAW, A, PREV, tasks, EPI = load_lerobot(
        args.n_obs, T, n_ep=args.n_ep, seed=args.seed)
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
        out = []
        for i in range(0, len(h), args.chunk):
            out.append(tok._decode(h[i:i + args.chunk], args.embodiment,
                                   None)[0][..., :D_act])
        return torch.cat(out)

    def err_of(dec, ref):
        """RMS по НЕПРЕРЫВНЫМ каналам на исполняемом окне. Схват выведен
        отдельно: бинарный канал ошибается примерно на единицу в моменты
        переключения и подминает максимум по чанку."""
        d = (dec[:, :args.window] - ref[:, :args.window]).abs()[..., :D_act - 1]
        return d.flatten(1).pow(2).mean(-1).sqrt() / scale

    def err_lat(h, ref):
        return err_of(dec_lat(h), ref)

    def grip_of(dec, ref):
        """Доля позиций окна, где знак схвата разошёлся с опорой."""
        a = dec[:, :args.window, -1] > 0
        b = ref[:, :args.window, -1] > 0
        return (a != b).float().mean(-1)

    rng = torch.Generator(device=args.device).manual_seed(1)
    ar = torch.arange(B, device=args.device)
    print(f"наблюдений {B}, эпизодов {len(np.unique(EPI))}, "
          f"смещение {args.pos_offset}, окно {args.window}, уровней {L}, "
          f"позиций {P}\n")

    # семейства структурированных наборов при K = 4
    fam = {
        "1 отрезок 4": seg_family(P, (4,)),
        "2 отрезка 2+2": seg_family(P, (2, 2)),
        "отрезок 2 + 2 одиночки": seg_family(P, (2, 1, 1)),
        "<= 2 отрезков, всего 4": sorted(set(seg_family(P, (4,)))
                                         | set(seg_family(P, (3, 1)))
                                         | set(seg_family(P, (2, 2)))),
    }
    print("размеры семейств: " + ", ".join(f"{k} — {len(v)}" for k, v in fam.items()))

    all_sub = [S for k in range(1, 5) for S in itertools.combinations(range(P), k)]
    print(f"подмножеств <= 4 для точного перебора: {len(all_sub)}\n")

    with torch.no_grad():
        z_ref = to_levels(gen(None, nb))
        a_ref = dec_lat(latent_from_codes(E, z_ref))
        lg0 = blk(None)
        lg_old = blk(z_ref[:, :, 0])            # ДОПУСТИМЫЙ признак: старый проход
        ent_old = -(lg_old.softmax(-1) * lg_old.log_softmax(-1)).sum(-1)
        t2 = lg_old.topk(2, -1).values
        marg_old = (t2[..., 0] - t2[..., 1])

        # ошибка самой модели относительно датасета — масштаб для A5
        e_model_true = err_of(a_ref, a_true).cpu().numpy()
        g_model_true = grip_of(a_ref, a_true).cpu().numpy()

        keys = ("greedy", "singleton", "same", "js", "window", "random",
                "random_chg", "exact", "ent_win", "ent_topk")
        res = {k: {K: [] for K in BUDGETS} for k in keys}
        fam_g = {k: [] for k in fam}
        div_g = {}                       # (вид похожести, lambda) -> список
        es_all, gr_st, gr_gr = [], [], []
        ndiff_all, feat = [], {k: [] for k in
                               ("task", "speed", "gswitch", "edit", "p")}
        delta_sing, delta_greedy, delta_exact = [], [], []
        step_gain = [[] for _ in range(max(BUDGETS))]
        jac, chg2 = [], []
        # у точного перебора СВОЙ знаменатель и свои кластеры: при --exact-obs
        # он идёт на подвыборке, и брать общий es было бы подменой знаменателя
        exact_size, exact_span, exact_contig = [], [], []
        exact_base, exact_greedy, exact_epi = [], [], []

        n_pos = args.max_pos or P
        for p_ in range(n_pos):
            u = lg0[:, p_].topk(args.rank_hi, -1).indices[
                ar, torch.randint(args.rank_lo, args.rank_hi, (B,),
                                  generator=rng, device=args.device)]
            v = z_ref[:, p_, 0]
            c0_old = z_ref[:, :, 0].clone()
            c0_old[:, p_] = u
            c1_old = blk(c0_old).argmax(-1)
            z_old = torch.stack([c0_old, c1_old,
                                 blk(torch.cat([c0_old, c1_old], 1)).argmax(-1)], -1)
            stale = z_old.clone()
            stale[:, :, 0] = z_ref[:, :, 0]

            h_st = latent_from_codes(E, stale)
            h_rf = latent_from_codes(E, z_ref)
            dec_st = dec_lat(h_st)
            e_st = err_of(dec_st, a_ref)
            es_all.append(e_st.cpu().numpy())
            gr_st.append(grip_of(dec_st, a_ref).cpu().numpy())

            # ---------- A0: сколько позиций реально изменились ----------
            diff = (stale != z_ref).any(-1)            # (B, P)
            ndiff_all.append(diff.sum(-1).cpu().numpy())

            # ---------- признаки для A4 ----------
            spd = (a_true[:, 1:args.window, :D_act - 1]
                   - a_true[:, :args.window - 1, :D_act - 1]).abs().mean((1, 2))
            gsw = ((a_true[:, :args.window, -1] > 0).float().std(1) > 0).float()
            edit = (E[0][v] - E[0][u]).float().norm(dim=-1)
            feat["task"].append(np.array(tasks))
            feat["speed"].append(spd.cpu().numpy())
            feat["gswitch"].append(gsw.cpu().numpy())
            feat["edit"].append(edit.cpu().numpy())
            feat["p"].append(np.full(B, p_))

            def g_of(sets):
                out = []
                for S in sets:
                    h = h_st.clone()
                    if S:
                        h[:, list(S)] = h_rf[:, list(S)]
                    out.append(e_st - err_lat(h, a_ref))
                return torch.stack(out)

            def g_of_sel(sel):
                """Выигрыш для ПОЭКЗЕМПЛЯРНОГО набора sel (B, K)."""
                h = h_st.clone()
                idx = ar.unsqueeze(1).expand_as(sel)
                h[idx, sel] = h_rf[idx, sel]
                return e_st - err_lat(h, a_ref)

            singles = g_of([(q,) for q in range(P)])        # (P, B)
            sing_T = singles.T                              # (B, P)

            # ---------- последовательный жадный ----------
            cur, e_cur = h_st.clone(), e_st.clone()
            taken = torch.zeros(B, P, dtype=torch.bool, device=args.device)
            gsel = torch.zeros(B, max(BUDGETS), dtype=torch.long, device=args.device)
            greedy_gain = {}
            for step in range(max(BUDGETS)):
                best_g = torch.full((B,), -1e9, device=args.device)
                best_q = torch.zeros(B, dtype=torch.long, device=args.device)
                for q in range(P):
                    h = cur.clone()
                    h[:, q] = h_rf[:, q]
                    g = e_cur - err_lat(h, a_ref)
                    g = torch.where(taken[:, q], torch.full_like(g, -1e9), g)
                    upd = g > best_g
                    best_g = torch.where(upd, g, best_g)
                    best_q = torch.where(upd, torch.full_like(best_q, q), best_q)
                cur[ar, best_q] = h_rf[ar, best_q]
                e_cur = e_cur - best_g
                taken[ar, best_q] = True
                gsel[:, step] = best_q
                step_gain[step].append(best_g.cpu().numpy())
                if step + 1 in BUDGETS:
                    greedy_gain[step + 1] = (e_st - e_cur).cpu().numpy()

            # ---------- A2: неаддитивность ВЫБРАННЫХ наборов ----------
            ssel = sing_T.argsort(-1, descending=True)[:, :4]
            g_sing4 = g_of_sel(ssel)
            g_gre4 = g_of_sel(gsel[:, :4])
            sum_sing = sing_T.gather(1, ssel).sum(-1)
            sum_gre = sing_T.gather(1, gsel[:, :4]).sum(-1)
            delta_sing.append((g_sing4 - sum_sing).cpu().numpy())
            delta_greedy.append((g_gre4 - sum_gre).cpu().numpy())

            a_set = [set(x.tolist()) for x in ssel]
            b_set = [set(x.tolist()) for x in gsel[:, :4]]
            jac.append(np.array([len(x & y) / len(x | y)
                                 for x, y in zip(a_set, b_set)]))
            second_sing = sing_T.argsort(-1, descending=True)[:, 1]
            chg2.append((gsel[:, 1] != second_sing).float().cpu().numpy())

            # ---------- A2: штраф за похожесть без router ----------
            dh = (h_rf - h_st).float()
            dn = torch.nn.functional.normalize(dh, dim=-1)
            sim_d = torch.bmm(dn, dn.transpose(1, 2)).abs()
            dist = torch.abs(torch.arange(P, device=args.device).view(-1, 1)
                             - torch.arange(P, device=args.device).view(1, -1))
            sim_t = torch.exp(-dist.float() / 2.0).expand(B, P, P)
            sc_scale = sing_T.abs().max(-1, keepdim=True).values.clamp_min(1e-12)
            for nm, sim in (("латентная", sim_d), ("временная", sim_t)):
                for lam in (0.0, 0.25, 0.5, 1.0, 2.0):
                    sel = torch.zeros(B, 4, dtype=torch.long, device=args.device)
                    tk = torch.zeros(B, P, dtype=torch.bool, device=args.device)
                    pen = torch.zeros(B, P, device=args.device)
                    for k in range(4):
                        s = sing_T / sc_scale - lam * pen
                        s = s.masked_fill(tk, -1e9)
                        q = s.argmax(-1)
                        sel[:, k] = q
                        tk[ar, q] = True
                        pen = torch.maximum(pen, sim[ar, q])
                    div_g.setdefault((nm, lam), []).append(
                        g_of_sel(sel).cpu().numpy())

            # ---------- прочие способы отбора ----------
            lg_new = blk(c0_old)                       # ОРАКУЛ: новый плотный проход
            rank_js = js_div(lg_new.softmax(-1), lg_old.softmax(-1))
            rnd = torch.rand(B, P, generator=rng, device=args.device)
            rnd_chg = torch.rand(B, P, generator=rng, device=args.device) + diff.float()
            for K in BUDGETS:
                res["greedy"][K].append(greedy_gain[K])
                for nm, sc in (("singleton", sing_T), ("js", rank_js),
                               ("random", rnd), ("random_chg", rnd_chg),
                               ("ent_topk", ent_old)):
                    res[nm][K].append(
                        g_of_sel(sc.argsort(-1, descending=True)[:, :K]).cpu().numpy())
                sm = torch.full((B, P), -1e9, device=args.device)
                sm[:, p_] = 1.0
                res["same"][K].append(
                    g_of_sel(sm.argsort(-1, descending=True)[:, :1]).cpu().numpy()
                    if K == 1 else res["same"][1][-1])
                wins = [tuple(range(s, s + K)) for s in range(P - K + 1)]
                res["window"][K].append(g_of(wins).max(0).values.cpu().numpy())
                # DEPLOYABLE окно: центр в позиции максимальной СТАРОЙ энтропии
                ctr = ent_old.argmax(-1)
                lo = (ctr - K // 2).clamp(0, P - K)
                sel = lo.unsqueeze(1) + torch.arange(K, device=args.device)
                res["ent_win"][K].append(g_of_sel(sel).cpu().numpy())

            gr_gr.append(grip_of(dec_lat(cur), a_ref).cpu().numpy())

            # ---------- A3: структурированные семейства при K = 4 ----------
            for nm, sets in fam.items():
                fam_g[nm].append(g_of(sets).max(0).values.cpu().numpy())

            # ---------- A1: точный перебор подмножеств <= 4 ----------
            if args.exact:
                sub = slice(0, args.exact_obs) if args.exact_obs else slice(None)
                hs, hr, est = h_st[sub], h_rf[sub], e_st[sub]
                nsub = hs.shape[0]
                aref_s = a_ref[sub]
                best = torch.full((nsub,), -1e9, device=args.device)
                best_i = torch.zeros(nsub, dtype=torch.long, device=args.device)
                for i in range(0, len(all_sub), args.exact_block):
                    blockS = all_sub[i:i + args.exact_block]
                    hh = hs.unsqueeze(0).repeat(len(blockS), 1, 1, 1)
                    for j, S in enumerate(blockS):
                        hh[j][:, list(S)] = hr[:, list(S)]
                    ee = err_lat(hh.reshape(-1, P, hs.shape[-1]),
                                 aref_s.repeat(len(blockS), 1, 1))
                    gg = (est.repeat(len(blockS)) - ee).reshape(len(blockS), -1)
                    mx, am = gg.max(0)
                    upd = mx > best
                    best = torch.where(upd, mx, best)
                    best_i = torch.where(upd, am + i, best_i)
                res["exact"][4].append(best.cpu().numpy())
                exact_base.append(est.cpu().numpy())
                exact_greedy.append(greedy_gain[4][sub])
                exact_epi.append(EPI[sub])
                won = [all_sub[int(k)] for k in best_i.cpu().numpy()]
                exact_size.append(np.array([len(S) for S in won]))
                exact_span.append(np.array([max(S) - min(S) + 1 for S in won]))
                exact_contig.append(np.array(
                    [float(max(S) - min(S) + 1 == len(S)) for S in won]))
                ssum = torch.stack([sing_T[k, list(S)].sum()
                                    for k, S in enumerate(won)])
                delta_exact.append((best - ssum).cpu().numpy())
            print(f"  позиция {p_ + 1}/{n_pos} готова", flush=True)

    es = np.concatenate(es_all)
    epi_rep = np.tile(EPI, n_pos)
    if args.max_pos:
        print(f"\n!!! БЫСТРАЯ ПРОВЕРКА: только {n_pos} позиций из {P}, "
              f"числа в отчёт не годятся\n")
    ndiff = np.concatenate(ndiff_all)

    ex_base = np.concatenate(exact_base) if exact_base else None
    ex_epi = np.concatenate(exact_epi) if exact_epi else None
    ex_gre = np.concatenate(exact_greedy) if exact_greedy else None

    def ci(vals, base=None, epi=None):
        return paired_ci(np.asarray(vals),
                         es if base is None else base,
                         epi_rep if epi is None else epi)

    def fmt(vals, base=None, epi=None):
        pt, lo, hi = ci(vals, base, epi)
        return f"{pt:.2f} [{lo:.2f},{hi:.2f}]"

    print("\n" + "=" * 80)
    print("A0. СКОЛЬКО ПОЗИЦИЙ ВООБЩЕ РАСХОДЯТСЯ МЕЖДУ stale И z_ref (из 16)")
    print("=" * 80)
    print(f"  среднее {ndiff.mean():.2f}, медиана {np.median(ndiff):.0f}, "
          f"квартили {np.percentile(ndiff, 25):.0f}/{np.percentile(ndiff, 75):.0f}, "
          f"мин {ndiff.min()}, макс {ndiff.max()}")
    hist = np.bincount(ndiff, minlength=P + 1)
    print("  распределение: " + " ".join(
        f"{i}:{c}" for i, c in enumerate(hist) if c))
    print(f"  доля случаев, где расходится <= 4 позиций: {(ndiff <= 4).mean():.1%}")
    print("""
ЧИТАТЬ ТАК. Если расходится в среднем сильно больше четырёх — «4 из 16»
действительно разреженный выбор. Если около четырёх-пяти — заявлять
разреженность нельзя: выбор идёт почти из всех кандидатов, а случайный
baseline занижен разбавлением по шестнадцати ячейкам.
Строка «случ. средь измен.(орк)» ниже даёт честное сравнение по СОДЕРЖАНИЮ,
но deployable baseline'ом НЕ является: чтобы узнать, какие позиции изменились,
нужен тот самый полный пересчёт. Это оракульная диагностика.
ДВА ЗНАМЕНАТЕЛЯ, не путать: для ЭКОНОМИИ вычислений считается K/16, потому что
плотный проход всё равно пересчитывает все шестнадцать; для заявки «полезный
ремонт сосредоточен в малом наборе» знаменатель — число реально затронутых
позиций, и там K=4 это уже почти всё.""")

    print("\n" + "=" * 80)
    print("A5. АБСОЛЮТНЫЙ МАСШТАБ: велик ли вообще закрываемый разрыв")
    print("=" * 80)
    print(f"  ошибка модели относительно ДАТАСЕТА   {e_model_true.mean():.5f}")
    print(f"  разрыв stale — полный пересчёт        {es.mean():.5f}")
    print(f"  отношение разрыв / ошибка модели      "
          f"{es.mean() / max(e_model_true.mean(), 1e-12):.2f}")
    print(f"  схват: расхождение stale c опорой     "
          f"{np.concatenate(gr_st).mean():.4f}")
    print(f"  схват: после жадного пересчёта 8 поз. "
          f"{np.concatenate(gr_gr).mean():.4f}")
    print(f"  схват: модель против датасета         {g_model_true.mean():.4f}")
    print("""
Разрыв заметно меньше собственной ошибки модели -> даже идеальное закрытие
разрыва почти не меняет действие, и вся линия имеет низкий потолок.""")

    print("\n" + "=" * 80)
    print("A1/A3. ДОЛЯ РАЗРЫВА, ЗАКРЫТАЯ ПЕРЕСЧЁТОМ K ПОЗИЦИЙ (равное число позиций)")
    print("=" * 80)
    names = {"exact": "точный <=4", "greedy": "жадный послед.",
             "singleton": "одиночный оракул", "window": "лучшее окно (оракул)",
             "js": "по JS (оракул)", "ent_win": "окно по стар. энтр.",
             "ent_topk": "top-K стар. энтр.", "same": "только p",
             "random": "случайно из 16",
             "random_chg": "случ. средь измен.(орк)"}
    print(f"{'K':>3}" + "".join(f"{n:>24}" for n in names.values()))
    for K in BUDGETS:
        row = f"{K:>3}"
        for k in names:
            if not res[k][K]:
                row += f"{'—':>24}"
                continue
            if k == "exact":
                row += f"{fmt(np.concatenate(res[k][K]), ex_base, ex_epi):>24}"
            else:
                row += f"{fmt(np.concatenate(res[k][K])):>24}"
        print(row)

    if res["exact"][4]:
        ex = np.concatenate(res["exact"][4])
        if args.exact_obs:
            print(f"\n  точный перебор шёл на подвыборке {args.exact_obs} "
                  f"наблюдений; жадный на ней же для сопоставимости")
        print(f"  жадный сохраняет от точного: {ex_gre.sum() / max(ex.sum(), 1e-9):.1%}")
        sz = np.concatenate(exact_size)
        print(f"  размер победившего набора: среднее {sz.mean():.2f}, "
              + " ".join(f"{i}:{(sz == i).mean():.0%}" for i in (1, 2, 3, 4)))
        sp_ = np.concatenate(exact_span)
        print(f"  протяжённость победившего набора во времени: "
              f"среднее {sp_.mean():.2f} из {P}")
        print(f"  доля НЕПРЕРЫВНЫХ победивших наборов: "
              f"{np.concatenate(exact_contig).mean():.1%}")

    print("\n" + "=" * 80)
    print("A3. СТРУКТУРИРОВАННЫЕ ВРЕМЕННЫЕ СЕМЕЙСТВА, K = 4, ОРАКУЛ ВНУТРИ СЕМЕЙСТВА")
    print("=" * 80)
    for nm in fam:
        print(f"{nm:>26}  {fmt(np.concatenate(fam_g[nm]))}")
    print(f"{'произвольный (жадный)':>26}  {fmt(np.concatenate(res['greedy'][4]))}")
    if res["exact"][4]:
        print(f"{'произвольный (точный)':>26}  "
              f"{fmt(np.concatenate(res['exact'][4]), ex_base, ex_epi)}")
    print("""
РЕШЕНИЕ ПО АРХИТЕКТУРЕ, зафиксировано до запуска:
  два коротких отрезка отстают от произвольного не более чем на 0.03 ->
      предпочесть multi-segment router, он проще и аппаратно дружелюбнее;
  отставание больше 0.08 -> нужен произвольный set-router;
  между — решать по реальной эффективности на GPU.
Все семейства здесь ОРАКУЛЬНЫЕ: они подсматривают результат и baseline'ами
не являются.""")

    print("\n" + "=" * 80)
    print("A2. СИНЕРГИЯ ИЛИ ИЗБЫТОЧНОСТЬ: неаддитивность ВЫБРАННОГО набора")
    print("=" * 80)
    g1 = np.concatenate(res["singleton"][1]).mean()
    print(f"типичный одиночный выигрыш для масштаба: {g1:.5f}\n")
    rows = [("Delta(одиночный top-4)", np.concatenate(delta_sing)),
            ("Delta(жадный 4)", np.concatenate(delta_greedy))]
    if delta_exact:
        rows.append(("Delta(точный <=4)", np.concatenate(delta_exact)))
    print(f"{'набор':>26}{'Delta':>12}{'в долях выигрыша':>20}{'доля Delta>0':>14}")
    for nm, d in rows:
        print(f"{nm:>26}{d.mean():>12.5f}{d.mean() / max(abs(g1), 1e-12):>19.1%}"
              f"{(d > 0).mean():>14.1%}")
    print("\nпредельный выигрыш по шагам жадного отбора (в долях первого шага):")
    s1 = np.concatenate(step_gain[0]).mean()
    for i, sg in enumerate(step_gain):
        m = np.concatenate(sg).mean()
        print(f"  шаг {i + 1}: {m:.5f}  ({m / max(abs(s1), 1e-12):.1%})")
    print(f"\n  Jaccard(одиночный top-4, жадный 4): {np.concatenate(jac).mean():.3f}")
    print(f"  доля примеров, где второй выбор жадного отличается от второго "
          f"по одиночному баллу: {np.concatenate(chg2).mean():.1%}")

    print("\n  ШТРАФ ЗА ПОХОЖЕСТЬ поверх одиночных баллов (K=4, без router):")
    print(f"{'похожесть':>14}{'lambda':>9}{'закрытая доля':>26}")
    for (nm, lam), v in sorted(div_g.items()):
        print(f"{nm:>14}{lam:>9}{fmt(np.concatenate(v)):>26}")
    print("""
ЧИТАТЬ ТАК.
  Delta(одиночный) заметно отрицательна, Delta(жадный) около нуля, и штраф за
      похожесть почти догоняет жадный -> механизм ИЗБЫТОЧНОСТЬ. Тогда
      достаточно независимого scorer с diversity-членом, и заявка про
      «условный последовательный выбор» слабая.
  Delta(жадный) положительна и штраф не догоняет -> механизм СИНЕРГИЯ,
      dependency-aware router содержателен.""")

    print("\n" + "=" * 80)
    print("A4. РАЗБИВКА ПО УСЛОВИЯМ (закрытая доля при K = 4)")
    print("=" * 80)
    F = {k: np.concatenate(v) for k, v in feat.items()}
    gg4 = np.concatenate(res["greedy"][4])
    gs4 = np.concatenate(res["singleton"][4])

    def bucket_report(title, key, edges=None, labels=None):
        print(f"\n  по {title}:")
        x = F[key]
        if edges is None:
            groups = [(str(u), x == u) for u in np.unique(x)]
        else:
            qs = np.quantile(x.astype(float), edges)
            groups = []
            for i in range(len(qs) - 1):
                last = i == len(qs) - 2
                hi_ok = x <= qs[i + 1] if last else x < qs[i + 1]
                m = (x >= qs[i]) & hi_ok
                groups.append((f"{labels[i]} [{qs[i]:.3g},{qs[i + 1]:.3g}]", m))
        print(f"{'группа':>44}{'n':>7}{'жадный':>10}{'одиночный':>12}")
        for nm, m in groups:
            if m.sum() < 20:
                continue
            a = gg4[m].sum() / max(es[m].sum(), 1e-12)
            b = gs4[m].sum() / max(es[m].sum(), 1e-12)
            print(f"{nm[:44]:>44}{int(m.sum()):>7}{a:>10.2f}{b:>12.2f}")

    bucket_report("позиции вмешательства p", "p")
    bucket_report("переключению схвата в окне", "gswitch")
    bucket_report("скорости движения", "speed", [0, .25, .5, .75, 1.],
                  ["Q1 медл", "Q2", "Q3", "Q4 быстр"])
    bucket_report("амплитуде правки coarse", "edit", [0, .25, .5, .75, 1.],
                  ["Q1 мал", "Q2", "Q3", "Q4 крупн"])
    ts = F["task"]
    cnt = {t: int((ts == t).sum()) for t in set(ts.tolist())}
    top = sorted(cnt, key=cnt.get, reverse=True)[:12]
    if len(cnt) > 1:
        print(f"\n  по задаче (12 самых частых из {len(cnt)}):")
        print(f"{'задача':>60}{'n':>7}{'жадный':>10}")
        for t in top:
            m = ts == t
            if m.sum() < 20:
                continue
            print(f"{t[:60]:>60}{int(m.sum()):>7}"
                  f"{gg4[m].sum() / max(es[m].sum(), 1e-12):>10.2f}")

    print("\n" + "=" * 80)
    print("ВОРОТА ФАЗЫ A, зафиксированы до запуска")
    print("=" * 80)
    pt, lo, hi = ci(gg4)
    print(f"  A1: нижняя граница ДИ жадного при K=4 >= 0.80 -> {lo:.2f} "
          f"{'ПРОЙДЕНО' if lo >= 0.80 else 'НЕ ПРОЙДЕНО'}")
    if res["exact"][4]:
        r = ex_gre.sum() / max(np.concatenate(res["exact"][4]).sum(), 1e-9)
        print(f"  A1: жадный сохраняет >= 95% точного -> {r:.1%} "
              f"{'ПРОЙДЕНО' if r >= 0.95 else 'НЕ ПРОЙДЕНО'}")
    print("  A0 и A2 ворот не имеют: они определяют, ЧТО именно должен учить "
          "router, и корректна ли сама постановка про разреженность.")


if __name__ == "__main__":
    main()
