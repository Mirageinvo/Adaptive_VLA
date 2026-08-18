"""K-4b0: совпадает ли режим паддинга датасета с реальным инференсом.

ЗАЧЕМ. Общая длина паддинга по всей выборке убрала зависимость датасета от
builder-батча. Но она НЕ совпадает с тем, как модель работает в LIBERO: там
батч равен единице, и каждое наблюдение идёт со своей естественной длиной.
Длина входит в base_pos позиционных идентификаторов токенов действия, поэтому
режим паддинга — часть распределения входов. Если режимы расходятся, router
обучится не на том, на чём будет применяться.

Решение принимается ДО обучения B1.

ТРИ ОСИ РАСХОЖДЕНИЯ с официальным scripts/eval_libero.py, а не одна:
  ДЛИНА паддинга — там padding=True по батчу из n_envs окружений;
  СТОРОНА паддинга — там padding_side="left", у нас умолчание процессора;
  pos_offset — там ПО УМОЛЧАНИЮ 4, а все наши замеры идут на 3.

Утверждение «в LIBERO батч равен единице» НЕВЕРНО: n_envs по умолчанию 10.
Правильно — семантика n_envs=1 достигается естественной длиной, а официальный
режим это дополненный слева батч из десяти окружений одной задачи.

РЕЖИМЫ на ОДНИХ И ТЕХ ЖЕ наблюдениях (имя = длина_сторона_offset):
  nat_r_3     естественная длина, без паддинга — новая сборка B0;
  fix181_r_3  фиксированная 181 — прежняя сборка B0;
  nat_l_4     естественная длина, слева, offset 4;
  dyn10_l_4   дополнение по батчу из 10 слева, offset 4 — ближе всего к eval;
  dyn10_l_3   то же при offset 3 — выделяет вклад ОДНОГО offset.

ЧТО СРАВНИВАЕМ:
  доля наблюдений с побитово совпавшим z_ref;
  Jaccard changed-support по вмешательствам;
  ранговую корреляцию Спирмена одиночных выигрышей внутри вмешательства;
  оракульные числа при K=4 (точный, жадный, одиночный);
  лучшие причинные baseline (энтропия, малый запас).

ЧТЕНИЕ, зафиксировано до запуска:
  структура и ранжирование практически совпадают (Спирмен >= 0.9, оракульные
      числа в пределах 0.02) -> оставить B0 как есть, но ЗАФИКСИРОВАТЬ тот же
      паддинг в будущем refiner и симуляторе;
  расходятся -> выбрать режим будущего LIBERO evaluation, то есть batch1, и
      пересобрать B0 на нём.

Запуск:
    python3 experiments/k4b0_padding_probe.py \
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO --n-obs 96
"""

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k4b0_build_router_dataset import greedy_paths, subsets_of  # noqa: E402


def rankdata_avg(x):
    """Ранги СО СРЕДНИМ по ничьим.

    Наивное argsort(argsort(x)) даёт порядковые ранги и разрывает ничьи по
    индексу. В одиночных выигрышах около десяти точных нулей на строку
    (позиции вне changed-support), и порядковые ранги присваивают им
    одинаковый порядок в обоих режимах — то есть создают искусственную
    согласованность и ЗАВЫШАЮТ корреляцию."""
    x = np.asarray(x, np.float64)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), np.float64)
    r[o] = np.arange(len(x), dtype=np.float64)
    xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    """Ранговая корреляция Спирмена с корректной обработкой ничьих."""
    ra, rb = rankdata_avg(a), rankdata_avg(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-obs", type=int, default=96)
    ap.add_argument("--n-ep", type=int, default=48)
    ap.add_argument("--fixed-pad", type=int, default=181,
                    help="длина фиксированного паддинга ПРЕЖНЕГО B0; берётся "
                         "из его metadata, а не пересчитывается по выборке")
    ap.add_argument("--dyn-batch", type=int, default=32,
                    help="батч для режима dynamic; должен быть меньше n-obs, "
                         "иначе dynamic совпадёт с fixed")
    ap.add_argument("--kmax", type=int, default=4)
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

    import torch

    from k1_residual_cost import latent_from_codes, projected_codebooks
    from k3_bar_suffix_repair import (MAX_ACTION_Q, STATE_Q01, STATE_Q99,
                                      build_batch, load_lerobot)

    sys.path.insert(0, os.path.abspath(args.root))
    import copy
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
    tok32 = copy.deepcopy(tok).float().eval()
    E = projected_codebooks(tok32, args.device)

    IM1, IM2, ST_RAW, A_, PREV, TASKS, EPI = load_lerobot(
        args.n_obs, T, n_ep=args.n_ep, seed=args.seed)
    N = len(TASKS)
    A_ = np.asarray(A_, np.float32).copy()
    A_[..., :-1] = A_[..., :-1] / MAX_ACTION_Q[:-1]
    A_[..., -1] = -A_[..., -1]
    scale = float(np.clip(A_, -1, 1).max() - np.clip(A_, -1, 1).min())
    st_all = ((ST_RAW - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0
              - 1.0).astype(np.float32)

    from smolvla.bar import SmolVLABlockwiseAR

    model = SmolVLABlockwiseAR.from_pretrained(
        args.ckpt, trust_remote_code=True, dtype=torch.bfloat16,
        token_budget=P * L, num_blocks=L,
        action_vocab_size=tok.vocab_size).to(args.device).eval()
    bs, nb = model.block_size, model.num_blocks

    # общая длина по всей выборке — та, на которой собран B0
    pad_fixed = 0
    for lo in range(0, N, args.dyn_batch):
        b = build_batch(IM1[lo:lo + args.dyn_batch], IM2[lo:lo + args.dyn_batch],
                        TASKS[lo:lo + args.dyn_batch], st_all[lo:lo + args.dyn_batch],
                        proc, args, "cpu")
        pad_fixed = max(pad_fixed, int(b["input_ids"].shape[1]))
    print(f"общая длина паддинга (как в B0): {pad_fixed}\n")

    rank_table = np.random.default_rng(10_000 + args.seed).integers(
        args.rank_lo, args.rank_hi, size=(N, P))

    def run(lo, hi, pad_to, pad_side, pos_off):
        """Один блок наблюдений в заданном режиме."""
        B = hi - lo
        batch = build_batch(IM1[lo:hi], IM2[lo:hi], TASKS[lo:hi], st_all[lo:hi],
                            proc, args, args.device, pad_to=pad_to,
                            pad_side=pad_side)
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
                    device=VLM.device, position_offset=pos_off)
                pids = model._build_joint_position_ids(
                    batch_size=B, vlm_seq_len=vlen, action_pos_ids=apos,
                    device=VLM.device)
                return model._predict_next_block_logits(
                    vlm_inputs_embeds=VLM,
                    attention_mask=batch.get("attention_mask"),
                    history_tokens=hist, position_ids=pids).float()

            def dec(h):
                out = []
                for i in range(0, len(h), args.chunk):
                    out.append(tok32._decode(h[i:i + args.chunk],
                                             args.embodiment, None)[0][..., :D_act])
                return torch.cat(out)

            def sq(h, ref):
                d = (dec(h)[:, :args.window]
                     - ref[:, :args.window]).abs()[..., :D_act - 1]
                return d.flatten(1).pow(2).mean(-1) / scale ** 2

            hist = None
            for _ in range(nb):
                hist = (blk(hist).argmax(-1) if hist is None
                        else torch.cat([hist, blk(hist).argmax(-1)], 1))
            z_ref = hist.reshape(-1, L, P).transpose(1, 2)
            a_ref = dec(latent_from_codes(E, z_ref))
            lg0 = blk(None)
            ar = torch.arange(B, device=args.device)
            h_rf = latent_from_codes(E, z_ref)

            out = dict(z_ref=z_ref.cpu().numpy(), vlen=int(vlen),
                       supp=np.zeros((P, B), np.int64),
                       sing=np.zeros((P, B, P), np.float32),
                       e0=np.zeros((P, B), np.float64),
                       ent=np.zeros((P, B, P), np.float32),
                       mrg=np.zeros((P, B, P), np.float32),
                       gmaps=[[None] * B for _ in range(P)])
            for p_ in range(P):
                u = lg0[:, p_].topk(args.rank_hi, -1).indices[
                    ar, torch.as_tensor(rank_table[lo:hi, p_], device=args.device)]
                c0 = z_ref[:, :, 0].clone()
                c0[:, p_] = u
                lgb = blk(c0)
                pb = lgb.softmax(-1)
                out["ent"][p_] = (-(pb * lgb.log_softmax(-1)).sum(-1)).cpu().numpy()
                t2 = lgb.topk(2, -1).values
                out["mrg"][p_] = (t2[..., 0] - t2[..., 1]).cpu().numpy()
                c1 = lgb.argmax(-1)
                z_old = torch.stack(
                    [c0, c1, blk(torch.cat([c0, c1], 1)).argmax(-1)], -1)
                stale = z_old.clone()
                stale[:, :, 0] = z_ref[:, :, 0]
                h_st = latent_from_codes(E, stale)
                e0 = sq(h_st, a_ref)
                out["e0"][p_] = e0.cpu().numpy()
                diff = (stale != z_ref).any(-1)
                out["supp"][p_] = (diff.int()
                                   * (1 << torch.arange(P, device=args.device))
                                   ).sum(-1).cpu().numpy()
                for b in range(B):
                    C = torch.nonzero(diff[b]).flatten().tolist()
                    subs = subsets_of(C, args.kmax)
                    hh = h_st[b].unsqueeze(0).repeat(len(subs), 1, 1)
                    for j, S in enumerate(subs):
                        if S:
                            hh[j, list(S)] = h_rf[b, list(S)]
                    gg = (e0[b] - sq(hh, a_ref[b].unsqueeze(0)
                                     .repeat(len(subs), 1, 1))).cpu().numpy()
                    gg[0] = 0.0
                    gm = {tuple(sorted(S)): float(gg[j])
                          for j, S in enumerate(subs)}
                    out["gmaps"][p_][b] = gm
                    for q in range(P):
                        out["sing"][p_, b, q] = gm.get((q,), 0.0)
        return out

    # опорный режим — ближайший к официальному eval
    specs = [
        ("nat_r_3", None, None, 3, 1),
        ("fix181_r_3", args.fixed_pad, None, 3, args.dyn_batch),
        ("nat_l_4", None, "left", 4, 1),
        ("dyn10_l_3", None, "left", 3, 10),
        ("dyn10_l_4", None, "left", 4, 10),
    ]
    regimes, vl = {}, {}
    for nm, pad, side, off, bsz in specs:
        print(f"режим {nm}: pad={pad or 'естественная'}, "
              f"side={side or 'умолчание'}, offset={off}, batch={bsz}")
        ps = [run(lo, min(lo + bsz, N), pad, side, off)
              for lo in range(0, N, bsz)]
        regimes[nm] = _merge(ps, P, N)
        vl[nm] = sorted({p["vlen"] for p in ps})
        print(f"  vlen: {vl[nm] if len(vl[nm]) < 6 else [vl[nm][0], '...', vl[nm][-1]]}")

    _report(regimes, P, N, args, ref="dyn10_l_4")


def _merge(parts, P, N):
    out = dict(z_ref=np.concatenate([p["z_ref"] for p in parts]),
               supp=np.concatenate([p["supp"] for p in parts], 1),
               sing=np.concatenate([p["sing"] for p in parts], 1),
               e0=np.concatenate([p["e0"] for p in parts], 1),
               ent=np.concatenate([p["ent"] for p in parts], 1),
               mrg=np.concatenate([p["mrg"] for p in parts], 1),
               gmaps=[[g for p in parts for g in p["gmaps"][q]]
                      for q in range(P)])
    return out


def _report(R, P, N, args, ref):
    def to_rms(e0, g):
        return np.sqrt(e0) - np.sqrt(max(e0 - g, 0.0))

    def g_of(gm, C, S):
        return gm[tuple(sorted(set(S) & C))]

    K = args.kmax
    print("\n" + "=" * 78)
    print(f"РАСХОЖДЕНИЕ С ОПОРНЫМ РЕЖИМОМ {ref} (ближайший к eval_libero)")
    print("=" * 78)
    print(f"  {'режим':>12}{'z_ref':>8}{'Jac supp':>10}{'Спирмен':>10}"
          f"{'Jac top4':>10}{'лучшая':>9}{'перенос':>9}")
    a_ = R[ref]
    for nm in R:
        if nm == ref:
            continue
        b_ = R[nm]
        zeq = (a_["z_ref"] == b_["z_ref"]).all(axis=(1, 2)).mean()
        jac, spr, jt, best1, num_x, den_x = [], [], [], [], 0.0, 0.0
        for p_ in range(P):
            for i in range(N):
                sa, sb = int(a_["supp"][p_, i]), int(b_["supp"][p_, i])
                uni = bin(sa | sb).count("1")
                jac.append(1.0 if uni == 0 else bin(sa & sb).count("1") / uni)
                spr.append(spearman(a_["sing"][p_, i], b_["sing"][p_, i]))
                ta = set(np.argsort(-a_["sing"][p_, i])[:K].tolist())
                tb = set(np.argsort(-b_["sing"][p_, i])[:K].tolist())
                jt.append(len(ta & tb) / len(ta | tb))
                best1.append(int(np.argmax(a_["sing"][p_, i])
                                 == np.argmax(b_["sing"][p_, i])))
                # ПЕРЕНОС: набор, выбранный в режиме nm, оцениваем метками
                # ОПОРНОГО режима. Это и есть вопрос «сработает ли router,
                # обученный на чужом протоколе».
                gm = a_["gmaps"][p_][i]
                C = set(q for q in range(P) if a_["supp"][p_, i] >> q & 1)
                e0 = float(a_["e0"][p_, i])
                num_x += to_rms(e0, g_of(gm, C, list(tb)))
                den_x += to_rms(e0, g_of(gm, C, list(ta)))
        print(f"  {nm:>12}{zeq:>8.1%}{np.mean(jac):>10.3f}{np.mean(spr):>10.3f}"
              f"{np.mean(jt):>10.3f}{np.mean(best1):>9.1%}"
              f"{num_x / max(den_x, 1e-30):>9.1%}")
    print("""  z_ref   — доля наблюдений с побитово совпавшим планом;
  Jac supp— Jaccard changed-support; Спирмен — по одиночным выигрышам (ничьи
            усреднены); Jac top4 — совпадение четвёрок; лучшая — совпала ли
            сильнейшая позиция;
  ПЕРЕНОС — какую долю ОПОРНОГО одиночного оракула даёт top-4, выбранная в
            чужом режиме. Это прямой ответ на вопрос, переносится ли router.""")

    print("\n  ОРАКУЛЫ И ПРИЧИННЫЕ BASELINE ВНУТРИ КАЖДОГО РЕЖИМА, K=4")
    print(f"  {'режим':>12}{'точный':>9}{'жадный':>9}{'одиночн.':>10}"
          f"{'энтроп.':>9}{'запас':>8}")
    for nm, r in R.items():
        num = {k: 0.0 for k in ("ex", "gr", "sg", "en", "mg")}
        den = 0.0
        for p_ in range(P):
            for i in range(N):
                gm = r["gmaps"][p_][i]
                C = set(q for q in range(P) if r["supp"][p_, i] >> q & 1)
                e0 = float(r["e0"][p_, i])
                den += np.sqrt(e0)
                num["ex"] += to_rms(e0, max(gm.values()))
                add = greedy_paths(
                    {S: to_rms(e0, g) for S, g in gm.items()},
                    sorted(C), 0.0, K)[0]
                num["gr"] += to_rms(e0, g_of(gm, C, add))
                num["sg"] += to_rms(e0, g_of(
                    gm, C, np.argsort(-r["sing"][p_, i])[:K]))
                num["en"] += to_rms(e0, g_of(
                    gm, C, np.argsort(-r["ent"][p_, i])[:K]))
                num["mg"] += to_rms(e0, g_of(
                    gm, C, np.argsort(r["mrg"][p_, i])[:K]))
        print(f"  {nm:>12}{num['ex'] / den:>9.3f}{num['gr'] / den:>9.3f}"
              f"{num['sg'] / den:>10.3f}{num['en'] / den:>9.3f}"
              f"{num['mg'] / den:>8.3f}")

    print("""
ЧИТАТЬ ТАК, зафиксировано до запуска.
  Сравнение dyn10_l_3 против dyn10_l_4 выделяет вклад ОДНОГО pos_offset.
  Сравнение nat_l_4 против dyn10_l_4 выделяет вклад ОДНОЙ длины паддинга.
  Сравнение nat_r_3 против nat_l_4 выделяет вклад СТОРОНЫ и offset вместе.

  ПЕРЕНОС близок к 100% -> протокол сборки можно выбирать по удобству, router
      перенесётся; достаточно зафиксировать один режим везде.
  ПЕРЕНОС заметно ниже 100% -> собирать B0 надо в том режиме, в котором будет
      идти evaluation, и объявить его частью метода.
  Агрегаты внутри режимов при этом могут совпадать: устойчивость статистики не
      означает переносимости поэкземплярных решений.""")


if __name__ == "__main__":
    main()
