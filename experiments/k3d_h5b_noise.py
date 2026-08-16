"""H5b: устойчивость проекции к шуму в цели — ПО НАПРАВЛЕНИЯМ, а не сферически.

ЧТО БЫЛО НЕ ТАК В H5 (K-3c). Шум добавлялся как случайный 512-мерный вектор
заданной L2-нормы. В высокой размерности случайное направление почти
ортогонально любой конкретной границе Вороного между кодами, поэтому выбор
кода почти не менялся, и «устойчивость» вышла артефактом размерности.

Признак был налицо: результат оказался ПЛОСКИМ — 0.133 / 0.134 / 0.134 при
росте шума в десять раз. Отсутствие деградации само по себе подозрительно.

ЧТО ИСПРАВЛЕНО:
  1. одно базовое направление на пример, меняется только масштаб;
  2. семейства направлений: сферическое, вдоль поправки, против поправки,
     К БЛИЖАЙШЕЙ ГРАНИЦЕ смены кода, в подпространстве словаря;
  3. печатается ДОЛЯ СМЕНЫ КОДА — без неё заявление об устойчивости пусто;
  4. печатается запас до ближайшего конкурирующего кода, в тех же единицах,
     что и шум, — видно, какой масштаб вообще способен что-то изменить;
  5. sigma явно есть ОТНОСИТЕЛЬНАЯ L2-норма поправки, а не покоординатное СКО.

ГРАНИЧНОЕ НАПРАВЛЕНИЕ. Жадная переквантизация на уровне 1 выбирает
c = argmin_u ||r - E1[u]||. Ближайшая граница отделяет его от занявшего второе
место c'. Наиболее «экономный» способ сменить выбор — идти вдоль
E1[c'] - E1[c]; расстояние до границы есть
    (||r - E1[c']||^2 - ||r - E1[c]||^2) / (2 ||E1[c'] - E1[c]||).

ВОРОТА (§8.5 плана): проекцию считать геометрически устойчивой, только если
полезный ремонт сохраняется НЕ ТОЛЬКО на сферическом шуме, но и на граничном и
подпространственном, при масштабе, сопоставимом с ожидаемой ошибкой головы.

Запуск:
    python3 experiments/k3d_h5b_noise.py \
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
    load_lerobot,
)
from k3b_suffix_repair import paired_ci  # noqa: E402

SIGMAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
FAMILIES = ("сферич.", "вдоль", "против", "к границе", "подпр. словаря")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-obs", type=int, default=96)
    ap.add_argument("--rank-lo", type=int, default=1)
    ap.add_argument("--rank-hi", type=int, default=5)
    ap.add_argument("--pos-offset", type=int, default=3)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--pca-dim", type=int, default=32,
                    help="подпространство словаря; 32 компоненты несут 99% энергии")
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
    Dz = E.shape[-1]

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

    def dec(c):
        return tok._decode(latent_from_codes(E, c), args.embodiment,
                           None)[0][..., :D_act]

    def err(c, ref):
        d = (dec(c)[:, :args.window] - ref[:, :args.window]).abs()[..., :D_act - 1]
        return d.flatten(1).pow(2).mean(-1).sqrt() / scale

    # подпространство словаря: 32 главные компоненты несут 99% энергии (K-1d)
    cb = E.reshape(-1, Dz).double()
    PC = torch.linalg.svd(cb - cb.mean(0, keepdim=True),
                          full_matrices=False)[2].float()[:args.pca_dim]

    rng = torch.Generator(device=args.device).manual_seed(1)
    ar = torch.arange(B, device=args.device)
    print(f"наблюдений {B}, эпизодов {len(np.unique(EPI))}, смещение "
          f"{args.pos_offset}, окно {args.window}\n")

    with torch.no_grad():
        z_ref = to_levels(gen(None, nb))
        a_ref = dec(z_ref)
        lg0 = blk(None)
        h_ref = latent_from_codes(E, z_ref)

        es, el = [], []
        rep = {f: {s: [] for s in SIGMAS} for f in FAMILIES}
        sw = {f: {s: [] for s in SIGMAS} for f in FAMILIES}
        margins = []

        for p_ in range(P):
            u = lg0[:, p_].topk(args.rank_hi, -1).indices[
                ar, torch.randint(args.rank_lo, args.rank_hi, (B,),
                                  generator=rng, device=args.device)]
            c0_old = z_ref[:, :, 0].clone()
            c0_old[:, p_] = u
            c1_old = blk(c0_old).argmax(-1)
            z_old = torch.stack([c0_old, c1_old,
                                 blk(torch.cat([c0_old, c1_old], 1)).argmax(-1)], -1)

            c0_new = z_ref[:, :, 0]
            stale = z_old.clone()
            stale[:, :, 0] = c0_new
            e_st = err(stale, a_ref)
            es.append(e_st.cpu().numpy())

            lg1 = blk(c0_new)
            c1 = z_old[:, :, 1].clone()
            c1[:, p_] = lg1.argmax(-1)[:, p_]
            c2 = z_old[:, :, 2].clone()
            c2[:, p_] = blk(torch.cat([c0_new, c1], 1)).argmax(-1)[:, p_]
            el.append(err(torch.stack([c0_new, c1, c2], -1), a_ref).cpu().numpy())

            # остаток, который квантует уровень 1, и геометрия его выбора
            r = h_ref[:, p_] - E[0][c0_new[:, p_]]
            d1 = torch.cdist(r.unsqueeze(1), E[1]).squeeze(1)      # (B, V)
            top2 = d1.topk(2, largest=False)
            c_win, c_run = top2.indices[:, 0], top2.indices[:, 1]
            diff = E[1][c_run] - E[1][c_win]
            nd = diff.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            # расстояние до биссектрисы между победителем и вторым
            marg = ((d1.gather(1, c_run[:, None]) ** 2
                     - d1.gather(1, c_win[:, None]) ** 2) / (2 * nd)).squeeze(1)
            dh = h_ref[:, p_] - latent_from_codes(E, stale)[:, p_]
            unit = dh.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            margins.append((marg / unit.squeeze(1)).cpu().numpy())

            g = torch.randn(B, Dz, generator=rng, device=args.device)
            dirs = {
                "сферич.": g / g.norm(dim=-1, keepdim=True),
                "вдоль": dh / unit,
                "против": -dh / unit,
                "к границе": diff / nd,
                "подпр. словаря": None,
            }
            gp = torch.randn(B, args.pca_dim, generator=rng, device=args.device) @ PC
            dirs["подпр. словаря"] = gp / gp.norm(dim=-1, keepdim=True)

            base_codes = None
            for f, dv in dirs.items():
                for s in SIGMAS:
                    tgt = h_ref.clone()
                    tgt[:, p_] = tgt[:, p_] + dv * unit * s
                    pr = greedy_suffix(E, stale, tgt, p_, 0)
                    rep[f][s].append(err(pr, a_ref).cpu().numpy())
                    if s == 0.0:
                        base_codes = pr[:, p_, 1:].clone()
                    sw[f][s].append((pr[:, p_, 1:] != base_codes).any(-1)
                                    .float().cpu().numpy())

    epi_rep = np.tile(EPI, P)
    es_a, el_a = np.concatenate(es), np.concatenate(el)
    mg = np.concatenate(margins)
    rl = paired_ci(es_a - el_a, es_a, epi_rep)

    print("=" * 78)
    print("ЗАПАС ДО БЛИЖАЙШЕЙ ГРАНИЦЫ СМЕНЫ КОДА (в долях поправки)")
    print("=" * 78)
    print(f"медиана {np.median(mg):.3f}, 25-й {np.percentile(mg, 25):.3f}, "
          f"75-й {np.percentile(mg, 75):.3f}")
    print("Шум меньше этой величины выбор кода изменить не может в принципе.\n")

    print("=" * 78)
    print("РЕМОНТ И ДОЛЯ СМЕНЫ КОДА ПО НАПРАВЛЕНИЯМ ШУМА")
    print("=" * 78)
    print(f"условный пересчёт BAR (2 вызова модели): "
          f"{rl[0]:.3f} [{rl[1]:.3f}, {rl[2]:.3f}]\n")
    print(f"{'шум':>6}" + "".join(f"{f:>24}" for f in FAMILIES))
    for s in SIGMAS:
        row = f"{s:>6.2f}"
        for f in FAMILIES:
            e = np.concatenate(rep[f][s])
            pt = paired_ci(es_a - e, es_a, epi_rep)[0]
            k = float(np.concatenate(sw[f][s]).mean())
            row += f"{f'{pt:.3f} / смен {k:.0%}':>24}"
        print(row)

    print("\n" + "=" * 78)
    print("ПРЕИМУЩЕСТВО НАД BAR-LOCAL (граница практической значимости 0.05)")
    print("=" * 78)
    print(f"{'шум':>6}" + "".join(f"{f:>24}" for f in FAMILIES))
    for s in SIGMAS:
        row = f"{s:>6.2f}"
        for f in FAMILIES:
            e = np.concatenate(rep[f][s])
            a = paired_ci(el_a - e, es_a, epi_rep)
            row += f"{f'{a[0]:+.3f} [{a[1]:+.3f},{a[2]:+.3f}]':>24}"
        print(row)

    print("""
КАК ЧИТАТЬ. Столбец «смен» — доля случаев, когда шум изменил выбранные коды.
Если она около нуля, шум ничего не сделал, и «устойчивость» в этой строке
ПУСТА: она говорит лишь о том, что случайное направление в 512 измерениях почти
ортогонально границам Вороного.

Осмысленны прежде всего направления «к границе» и «подпр. словаря»: первое —
наиболее экономный способ сменить код, второе — там, где реально живёт энергия
словаря и, вероятно, ошибка обучаемой головы.

Ворота: проекция геометрически устойчива, только если полезный ремонт
сохраняется и на этих двух направлениях при масштабе порядка запаса до
границы, а не только на сферическом шуме.""")


if __name__ == "__main__":
    main()
