"""K-3c: разреженность зависимостей и устойчивость проекции к шуму в цели.

ДВА ВОПРОСА, оба дешёвые и оба решают судьбу направлений.

ТЕСТ G — РАЗРЕЖЕННОСТЬ. Достаточно ли пересчитать малое подмножество позиций,
чтобы приблизиться к полной перегенерации? K-3b показал, что влияние правки
нелокально, но разрежено: вне диагонали 12-20% при диагонали 0.77-0.84.

Ворота (зафиксированы до запуска, §11.4 плана): проектировать разреженную
архитектуру имеет смысл, только если оракул при K<=4 закрывает не менее 80%
разрыва И доступный без подглядывания отбор обгоняет случайный при том же K.

ПРОВЕРКА H5 — ЦЕНА ПРОТИВ ТОЧНОСТИ. K-3b мерил качество и показал, что
проекция не лучше условного пересчёта. Но цены разные: BAR-local стоит ДВУХ
вызовов модели, проекция — НИ ОДНОГО, это арифметика по словарям. В настоящем
потоке целевая латента приходит из того же прохода, что принял решение о
правке. Значит ценность проекции может лежать в эффективности, а не в
качестве.

Однако во всех замерах проекция получала ТОЧНУЮ опорную латенту, а обучаемая
голова давала бы зашумлённую. Поэтому здесь цель портится шумом заданной
величины, и смотрится, при каком уровне проекция перестаёт догонять
условный пересчёт. Если рассыпается уже при малом шуме — H5 закрывается без
обучения refiner.

Шум задаётся В ДОЛЯХ САМОЙ ПОПРАВКИ: sigma=1 означает, что ошибка предсказания
цели сравнима с величиной исправляемого смещения.

ОСНОВНАЯ КОНФИГУРАЦИЯ (§18 плана): position_offset=3, окно 4, RMS по
непрерывным каналам, все 16 позиций, кластерный бутстрап по эпизодам.

Запуск:
    python3 experiments/k3c_sparse_and_noise.py \
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
from k3b_suffix_repair import paired_ci  # noqa: E402

BUDGETS = (0, 1, 2, 4, 8, 16)
SIGMAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-obs", type=int, default=96)
    ap.add_argument("--rank-lo", type=int, default=1)
    ap.add_argument("--rank-hi", type=int, default=5)
    ap.add_argument("--pos-offset", type=int, default=3)
    ap.add_argument("--window", type=int, default=4)
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
    print(f"наблюдений {B}, эпизодов {len(np.unique(EPI))}, "
          f"смещение {args.pos_offset}, окно {args.window}\n")

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

    def dec(c):
        return tok._decode(latent_from_codes(E, c), args.embodiment,
                           None)[0][..., :D_act]

    def err(c, ref):
        d = (dec(c)[:, :args.window] - ref[:, :args.window]).abs()[..., :D_act - 1]
        return d.flatten(1).pow(2).mean(-1).sqrt() / scale        # RMS

    rng = torch.Generator(device=args.device).manual_seed(1)
    nrng = np.random.default_rng(0)
    ar = torch.arange(B, device=args.device)

    with torch.no_grad():
        z_ref = to_levels(gen(None, nb))
        a_ref = dec(z_ref)
        lg0 = blk(None)
        h_ref = latent_from_codes(E, z_ref)

        curves = {k: {K: [] for K in BUDGETS} for k in
                  ("oracle", "same", "js", "entropy", "random")}
        noise = {s: [] for s in SIGMAS}
        base_stale, base_loc = [], []

        for p_ in range(P):
            u = lg0[:, p_].topk(args.rank_hi, -1).indices[
                ar, torch.randint(args.rank_lo, args.rank_hi, (B,),
                                  generator=rng, device=args.device)]
            c0_old = z_ref[:, :, 0].clone()
            c0_old[:, p_] = u
            lg1_old = blk(c0_old)
            c1_old = lg1_old.argmax(-1)
            z_old = torch.stack([c0_old, c1_old,
                                 blk(torch.cat([c0_old, c1_old], 1)).argmax(-1)], -1)

            c0_new = z_ref[:, :, 0]
            stale = z_old.clone()
            stale[:, :, 0] = c0_new
            e_stale = err(stale, a_ref)
            base_stale.append(e_stale.cpu().numpy())

            lg1 = blk(c0_new)
            c1 = z_old[:, :, 1].clone()
            c1[:, p_] = lg1.argmax(-1)[:, p_]
            c2 = z_old[:, :, 2].clone()
            c2[:, p_] = blk(torch.cat([c0_new, c1], 1)).argmax(-1)[:, p_]
            base_loc.append(err(torch.stack([c0_new, c1, c2], -1), a_ref).cpu().numpy())

            # ---------- G: выигрыш от замены тонких кодов позиции q на эталон ----
            gain = torch.zeros(B, P, device=args.device)
            for q in range(P):
                cc = stale.clone()
                cc[:, q, 1:] = z_ref[:, q, 1:]
                gain[:, q] = e_stale - err(cc, a_ref)

            # ранжирования: оракул (подглядывает), и доступные без ответа
            rank_js = js_div(lg1_old.softmax(-1), lg1.softmax(-1))          # (B,P)
            pl = lg1.softmax(-1)
            rank_ent = -(pl * pl.clamp_min(1e-30).log()).sum(-1)            # (B,P)

            for name, sc in (("oracle", gain), ("js", rank_js),
                             ("entropy", rank_ent),
                             ("random", torch.rand(B, P, generator=rng,
                                                   device=args.device))):
                order = sc.argsort(-1, descending=True)
                for K in BUDGETS:
                    cc = stale.clone()
                    if K:
                        sel = order[:, :K]
                        idx = ar.unsqueeze(1).expand_as(sel)
                        cc[idx, sel, 1:] = z_ref[idx, sel, 1:]
                    curves[name][K].append(err(cc, a_ref).cpu().numpy())
            for K in BUDGETS:                     # только изменённая позиция
                cc = stale.clone()
                if K:
                    cc[:, p_, 1:] = z_ref[:, p_, 1:]
                curves["same"][K].append(err(cc, a_ref).cpu().numpy())

            # ---------- H5: проекция при зашумлённой цели ----------
            dh = (h_ref - latent_from_codes(E, stale))[:, p_]      # поправка
            unit = dh.norm(dim=-1, keepdim=True)
            for s in SIGMAS:
                tgt = h_ref.clone()
                if s:
                    n = torch.randn(B, h_ref.shape[-1], generator=rng,
                                    device=args.device)
                    tgt[:, p_] = tgt[:, p_] + n / n.norm(dim=-1, keepdim=True) * unit * s
                noise[s].append(err(greedy_suffix(E, stale, tgt, p_, 0),
                                    a_ref).cpu().numpy())

    epi_rep = np.tile(EPI, P)
    es = np.concatenate(base_stale)
    el = np.concatenate(base_loc)

    print("=" * 78)
    print("ТЕСТ G. ДОЛЯ РАЗРЫВА, ЗАКРЫТАЯ ПЕРЕСЧЁТОМ K ПОЗИЦИЙ")
    print("=" * 78)
    print("Полная перегенерация (K=16) даёт эталон, то есть закрывает 100%.\n")
    print(f"{'K':>4}" + "".join(f"{n:>22}" for n in
                                ("оракул", "только p", "по JS", "по энтропии",
                                 "случайно")))
    for K in BUDGETS:
        row = f"{K:>4}"
        for n in ("oracle", "same", "js", "entropy", "random"):
            e = np.concatenate(curves[n][K])
            pt, lo, hi = paired_ci(es - e, es, epi_rep)
            row += f"{f'{pt:.2f} [{lo:.2f},{hi:.2f}]':>22}"
        print(row)

    o4 = paired_ci(es - np.concatenate(curves["oracle"][4]), es, epi_rep)
    j4 = paired_ci(es - np.concatenate(curves["js"][4]), es, epi_rep)
    r4 = paired_ci(es - np.concatenate(curves["random"][4]), es, epi_rep)
    print(f"""
ВОРОТА (зафиксированы до запуска): разреженную архитектуру строить, только если
оракул при K<=4 закрывает не менее 80% И доступный отбор бьёт случайный.
  оракул K=4:   {o4[0]:.2f} [{o4[1]:.2f}, {o4[2]:.2f}]  {'ПРОШЁЛ' if o4[1] >= 0.80 else 'НЕ ПРОШЁЛ'}
  по JS K=4:    {j4[0]:.2f} против случайного {r4[0]:.2f}""")

    print("\n" + "=" * 78)
    print("H5. УСТОЙЧИВОСТЬ ПРОЕКЦИИ К ШУМУ В ЦЕЛЕВОЙ ЛАТЕНТЕ")
    print("=" * 78)
    rl = paired_ci(es - el, es, epi_rep)
    print(f"условный пересчёт BAR (цена: 2 вызова модели): "
          f"закрыто {rl[0]:.3f} [{rl[1]:.3f}, {rl[2]:.3f}]\n")
    print(f"{'шум':>6}{'закрыто проекцией':>26}{'преим. над BAR':>26}")
    for s in SIGMAS:
        e = np.concatenate(noise[s])
        rp = paired_ci(es - e, es, epi_rep)
        ad = paired_ci(el - e, es, epi_rep)
        print(f"{s:>6.2f}{f'{rp[0]:.3f} [{rp[1]:.3f},{rp[2]:.3f}]':>26}"
              f"{f'{ad[0]:+.3f} [{ad[1]:+.3f},{ad[2]:+.3f}]':>26}")
    print("""
Шум задан В ДОЛЯХ САМОЙ ПОПРАВКИ: 1.00 значит, что ошибка предсказания цели
сравнима с величиной исправляемого смещения. Проекция стоит НОЛЬ вызовов
модели против двух у условного пересчёта, поэтому её ценность — в цене, а не в
качестве. Но если преимущество уходит в минус уже при малом шуме, обучаемая
голова цели этого не выдержит, и заявка об эффективности закрывается без
обучения refiner.""")


if __name__ == "__main__":
    main()
