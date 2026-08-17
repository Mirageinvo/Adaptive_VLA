"""K-4b0: совпадает ли режим паддинга датасета с реальным инференсом.

ЗАЧЕМ. Общая длина паддинга по всей выборке убрала зависимость датасета от
builder-батча. Но она НЕ совпадает с тем, как модель работает в LIBERO: там
батч равен единице, и каждое наблюдение идёт со своей естественной длиной.
Длина входит в base_pos позиционных идентификаторов токенов действия, поэтому
режим паддинга — часть распределения входов. Если режимы расходятся, router
обучится не на том, на чём будет применяться.

Решение принимается ДО обучения B1.

ТРИ РЕЖИМА на ОДНИХ И ТЕХ ЖЕ наблюдениях:
  batch1   — по одному наблюдению, естественная длина. Это инференс.
  dynamic  — padding до максимума В БАТЧЕ, как было до правки.
  fixed    — padding до общей длины по всей выборке. Так собран B0.

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


def spearman(a, b):
    """Ранговая корреляция без scipy: Пирсон по рангам."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-obs", type=int, default=96)
    ap.add_argument("--n-ep", type=int, default=48)
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

    def run(lo, hi, pad_to):
        """Один блок наблюдений в заданном режиме паддинга."""
        B = hi - lo
        batch = build_batch(IM1[lo:hi], IM2[lo:hi], TASKS[lo:hi], st_all[lo:hi],
                            proc, args, args.device, pad_to=pad_to)
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

    regimes = {}
    print("режим batch1 (инференс): по одному наблюдению...")
    parts = [run(i, i + 1, None) for i in range(N)]
    regimes["batch1"] = _merge(parts, P, N)
    print(f"  vlen: мин {min(p['vlen'] for p in parts)}, "
          f"макс {max(p['vlen'] for p in parts)}")
    for nm, pad in (("dynamic", None), ("fixed", pad_fixed)):
        print(f"режим {nm}...")
        ps = [run(lo, min(lo + args.dyn_batch, N), pad)
              for lo in range(0, N, args.dyn_batch)]
        regimes[nm] = _merge(ps, P, N)
        print(f"  vlen по батчам: {[p['vlen'] for p in ps]}")

    _report(regimes, P, N, args)


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


def _report(R, P, N, args):
    def to_rms(e0, g):
        return np.sqrt(e0) - np.sqrt(max(e0 - g, 0.0))

    def g_of(gm, C, S):
        return gm[tuple(sorted(set(S) & C))]

    print("\n" + "=" * 74)
    print("СРАВНЕНИЕ РЕЖИМОВ ПАДДИНГА")
    print("=" * 74)
    ref = "batch1"
    for nm in ("dynamic", "fixed"):
        a, b = R[ref], R[nm]
        zeq = (a["z_ref"] == b["z_ref"]).all(axis=(1, 2)).mean()
        jac, spr = [], []
        for p_ in range(P):
            for i in range(N):
                sa, sb = int(a["supp"][p_, i]), int(b["supp"][p_, i])
                inter = bin(sa & sb).count("1")
                uni = bin(sa | sb).count("1")
                jac.append(1.0 if uni == 0 else inter / uni)
                spr.append(spearman(a["sing"][p_, i], b["sing"][p_, i]))
        print(f"\n  {ref} против {nm}:")
        print(f"    z_ref совпал побитово: {zeq:.1%} наблюдений")
        print(f"    Jaccard changed-support: среднее {np.mean(jac):.3f}, "
              f"доля полного совпадения {np.mean(np.array(jac) == 1.0):.1%}")
        print(f"    Спирмен одиночных выигрышей: среднее {np.mean(spr):.3f}, "
              f"10-й проц. {np.percentile(spr, 10):.3f}")

    print("\n  ОРАКУЛЬНЫЕ ЧИСЛА И BASELINE ПРИ K=4, доля закрытого разрыва")
    print(f"  {'режим':>10}{'точный':>10}{'жадный':>10}{'одиночн.':>10}"
          f"{'энтропия':>10}{'запас':>9}{'vlen':>8}")
    for nm in ("batch1", "dynamic", "fixed"):
        r = R[nm]
        num = {k: 0.0 for k in ("ex", "gr", "sg", "en", "mg")}
        den = 0.0
        for p_ in range(P):
            for i in range(N):
                gm = r["gmaps"][p_][i]
                C = set(q for q in range(P) if r["supp"][p_, i] >> q & 1)
                e0 = float(r["e0"][p_, i])
                den += np.sqrt(e0)
                num["ex"] += to_rms(e0, max(gm.values()))
                add, _, _, _, _, _ = greedy_paths(
                    {S: to_rms(e0, g) for S, g in gm.items()},
                    sorted(C), 0.0, args.kmax)
                num["gr"] += to_rms(e0, g_of(gm, C, add))
                num["sg"] += to_rms(e0, g_of(
                    gm, C, np.argsort(-r["sing"][p_, i])[:args.kmax]))
                num["en"] += to_rms(e0, g_of(
                    gm, C, np.argsort(-r["ent"][p_, i])[:args.kmax]))
                num["mg"] += to_rms(e0, g_of(
                    gm, C, np.argsort(r["mrg"][p_, i])[:args.kmax]))
        print(f"  {nm:>10}{num['ex'] / den:>10.3f}{num['gr'] / den:>10.3f}"
              f"{num['sg'] / den:>10.3f}{num['en'] / den:>10.3f}"
              f"{num['mg'] / den:>9.3f}")

    print("""
ЧИТАТЬ ТАК, зафиксировано до запуска.
  Спирмен >= 0.9 и оракульные числа в пределах 0.02 -> структура и ранжирование
      сохраняются; оставить B0, но ЗАФИКСИРОВАТЬ тот же паддинг в будущем
      refiner и симуляторе, иначе распределение входов разъедется позже.
  Расходятся -> пересобрать B0 в режиме batch1, потому что именно он
      соответствует инференсу LIBERO.
Отдельно: если dynamic и fixed близки друг к другу, но оба далеки от batch1,
      значит дело не в разбросе длины, а в самом факте паддинга.""")


if __name__ == "__main__":
    main()
