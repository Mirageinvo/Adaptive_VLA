"""K-4a4: прямой тест субмодулярности функции группового ремонта.

ЗАЧЕМ. В фазе A измерено, что СРЕДНЕЕ Delta для трёх способов выбора
отрицательно. Это утверждение про три конкретных набора, а не про функцию:
Delta положительна в 10.4-44.9% отдельных случаев. Субмодулярность — условие на
ВСЕ приращения, и его надо проверять прямо, иначе заявка «функция ремонта
субмодулярна» ничем не обеспечена.

ЧТО СЧИТАЕМ. В позициях набора S устаревшая латента заменяется опорной:

    e(S) = MSE(Dec(h_stale->S), a_ref),   G(S) = e(пусто) - e(S).

Всё на КВАДРАТЕ ошибки, без корня: корень вогнут и сам порождает мнимую
супераддитивность (см. LESSONS.md и самопроверку в k4a2_phase_a.py).

Локальная убывающая отдача:

    Omega(A; q, r) = G(A+q) + G(A+r) - G(A) - G(A+q+r) >= 0.

Достаточно перебрать все A размера <= 2 и все пары q, r вне A: итоговый набор
имеет размер <= 4. Это 12720 троек на пример.

Монотонность:

    M(A, q) = G(A+q) - G(A).

Отрицательная M означает, что добавление позиции ухудшает результат — оракулу
тогда нужно право отказа, а router нельзя учить «чем больше, тем лучше».

ТОЧНОСТЬ ДЕКОДИРОВАНИЯ. Здесь она может решить исход, потому что решение
принимается ПО КАЖДОЙ ТРОЙКЕ, а не по среднему: случайная ошибка округления в
среднем гасится числом примеров, в пороговом счётчике — нет.

Важное уточнение, найденное при первом запуске: кодек `proc.action_processor`
грузится БЕЗ указания dtype, то есть во float32. Аргумент `dtype=torch.bfloat16`
задаётся только BAR-модели и на путь декодирования не влияет. Оценка «шум
округления ~4% одиночного выигрыша» относилась бы к bf16-кодеку и здесь
неприменима; прежние замеры этим не затронуты.

Порог всё равно берётся с измеренным полом, а не только по формуле из плана:

    tau = max(1e-8, 1e-3 * median(g1), 3 * измеренный численный пол),

где пол — максимальное расхождение float32 с float64 на одиночных наборах.
Справочное плечо bfloat16 печатается, если получится: их декодер не переводится
в половинную точность целиком (часть нормировок остаётся во float32), а чужой
код мы не правим. На порог это не влияет — он считается по float64.

Нарушением считается Omega < -tau.

ПРАВИЛО ЧТЕНИЯ, зафиксировано до запуска:
  доля массы нарушений <= 5% доступного выигрыша И крупные нарушения редки ->
      разрешена формулировка «приближённая субмодулярность»;
  иначе — только «наборы в среднем избыточны, но функция не субмодулярна».
Провал НЕ закрывает active-set направление: он лишь запрещает строить новизну
на субмодулярности.

ПРО ОБЪЁМ. Полная таблица G(S) для всех вмешательств — 1536 x 2517 x 4 байта =
15.5 ГБ. Поэтому статистика считается по каждому вмешательству на лету, а в NPZ
кладётся подвыборка таблиц (--keep-tables) и поэкземплярные агрегаты, которых
достаточно для кластерного бутстрапа.

Запуск:
    python3 experiments/k4a4_submodularity_audit.py \
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO
"""

import argparse
import itertools
import os
import subprocess
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
    load_lerobot,
)


def build_sets(P: int, kmax: int = 4):
    """Все наборы размера 0..kmax. Пустой ИДЁТ ПЕРВЫМ и имеет G = 0."""
    sets = [()] + [S for k in range(1, kmax + 1)
                   for S in itertools.combinations(range(P), k)]
    return sets, {S: i for i, S in enumerate(sets)}


def build_triples(P: int, idx: dict, max_a: int = 2):
    """Тройки (A, q, r): |A| <= max_a, q < r, обе вне A."""
    iA, iAq, iAr, iAqr, um, qm, na = [], [], [], [], [], [], []
    for ka in range(max_a + 1):
        for A in itertools.combinations(range(P), ka):
            rest = [x for x in range(P) if x not in A]
            for q, r in itertools.combinations(rest, 2):
                iA.append(idx[A])
                iAq.append(idx[tuple(sorted(A + (q,)))])
                iAr.append(idx[tuple(sorted(A + (r,)))])
                iAqr.append(idx[tuple(sorted(A + (q, r)))])
                m = 0
                for x in A + (q, r):
                    m |= 1 << x
                um.append(m)
                qm.append((1 << q) | (1 << r))   # маска ТОЛЬКО пары
                na.append(ka)
    return tuple(np.asarray(v) for v in (iA, iAq, iAr, iAqr, um, qm, na))


def build_mono(P: int, idx: dict, max_a: int = 3):
    """Пары (A, q): |A| <= max_a, q вне A."""
    iA, iAq, um, qm = [], [], [], []
    for ka in range(max_a + 1):
        for A in itertools.combinations(range(P), ka):
            for q in range(P):
                if q in A:
                    continue
                iA.append(idx[A])
                iAq.append(idx[tuple(sorted(A + (q,)))])
                m = 0
                for x in A + (q,):
                    m |= 1 << x
                um.append(m)
                qm.append(1 << q)                # маска ТОЛЬКО добавляемой
    return tuple(np.asarray(v) for v in (iA, iAq, um, qm))


def selftest(P: int = 8) -> None:
    """Проверка счётчика нарушений на функциях С ИЗВЕСТНЫМ ОТВЕТОМ.

    1. Модулярная (аддитивная) -> Omega тождественно ноль, нарушений нет.
    2. Покрытие множеств -> субмодулярна по построению, нарушений нет.
    3. Покрытие плюс супермодулярная надбавка на ОДНОЙ паре -> нарушения
       обязаны найтись, и ровно на тройках, содержащих эту пару."""
    sets, idx = build_sets(P)
    iA, iAq, iAr, iAqr, _, _, _ = build_triples(P, idx)
    rng = np.random.default_rng(0)

    w = rng.gamma(1.0, 1.0, size=P)
    g_mod = np.array([w[list(S)].sum() if S else 0.0 for S in sets])

    U = 40
    cov = rng.random((P, U)) < 0.25
    g_cov = np.array([cov[list(S)].any(0).sum() if S else 0 for S in sets],
                     float)

    bonus, pair = 3.0, (2, 5)
    g_sup = g_cov + np.array([bonus if set(pair) <= set(S) else 0.0
                              for S in sets])

    print("САМОПРОВЕРКА счётчика нарушений:")
    for nm, g, expect in (("модулярная", g_mod, 0),
                          ("покрытие (субмодулярна)", g_cov, 0),
                          ("покрытие + бонус на паре", g_sup, None)):
        om = g[iAq] + g[iAr] - g[iA] - g[iAqr]
        n = int((om < -1e-9).sum())
        print(f"  {nm:>28}: нарушений {n}")
        if expect == 0 and n:
            raise SystemExit(f"самопроверка провалена: {nm} дала {n} нарушений")
        if expect is None and n == 0:
            raise SystemExit("самопроверка провалена: супермодулярную надбавку "
                             "не заметили")
    print("  счётчик отличает субмодулярную функцию от несубмодулярной\n")


def cluster_ci(num, den, epi, n_boot: int = 2000, seed: int = 0):
    """Отношение сумм с кластерным бутстрапом по эпизодам."""
    rng = np.random.default_rng(seed)
    eps = np.unique(epi)
    ix = {e: np.where(epi == e)[0] for e in eps}
    pt = num.sum() / max(den.sum(), 1e-30)
    out = []
    for _ in range(n_boot):
        s = np.concatenate([ix[e] for e in rng.choice(eps, len(eps), replace=True)])
        out.append(num[s].sum() / max(den[s].sum(), 1e-30))
    return pt, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-obs", type=int, default=96)
    ap.add_argument("--n-ep", type=int, default=48)
    ap.add_argument("--max-pos", type=int, default=0)
    ap.add_argument("--set-block", type=int, default=32)
    ap.add_argument("--keep-neg", type=int, default=100_000,
                    help="сколько величин нарушения хранить на позицию "
                         "для процентилей")
    ap.add_argument("--keep-tables", type=int, default=64,
                    help="сколько таблиц G(S) сложить в NPZ целиком")
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
    ap.add_argument("--dump", default="logs/k4a4_submodularity.npz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    selftest()

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = "unknown"
    print(f"commit {commit}, seed {args.seed}\n")

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
    # Кодек грузится БЕЗ указания dtype, то есть уже во float32: аргумент
    # dtype=bfloat16 задаётся только BAR-модели и пути декодирования не
    # касается. .float() ниже — страховка на случай смены умолчания, а не
    # понижение точности откуда-то сверху.
    import copy

    print(f"dtype кодека при загрузке: {next(tok.parameters()).dtype}")
    tok32 = copy.deepcopy(tok).float().eval()
    E = projected_codebooks(tok32, args.device)

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

    def dec_with(m, h):
        out = []
        for i in range(0, len(h), args.chunk):
            out.append(m._decode(h[i:i + args.chunk], args.embodiment,
                                 None)[0][..., :D_act])
        return torch.cat(out)

    def dec_lat(h):
        return dec_with(tok32, h)

    def sq_of(dec, ref):
        """СРЕДНИЙ КВАДРАТ по непрерывным каналам на исполняемом окне."""
        d = (dec[:, :args.window] - ref[:, :args.window]).abs()[..., :D_act - 1]
        return d.flatten(1).pow(2).mean(-1) / scale ** 2

    sets, idx = build_sets(P)
    tA, tAq, tAr, tAqr, t_um, qr_um, t_na = build_triples(P, idx)
    mA, mAq, m_um, m_qm = build_mono(P, idx)
    print(f"наборов размера <=4 (с пустым): {len(sets)}")
    print(f"троек (A,q,r), |A|<=2: {len(tA)}")
    print(f"пар (A,q), |A|<=3: {len(mA)}\n")

    rng = torch.Generator(device=args.device).manual_seed(1)
    ar = torch.arange(B, device=args.device)
    n_pos = args.max_pos or P

    # поэкземплярные агрегаты для бутстрапа
    agg = {k: [] for k in ("n_viol", "sum_neg", "max_neg", "n_tri",
                           "n_viol_nz", "sum_neg_nz", "n_tri_nz",
                           "n_viol_ch", "sum_neg_ch", "n_tri_ch",
                           "n_mono_viol", "sum_mono_neg", "n_mono",
                           "n_mono_viol_nz", "n_mono_nz",
                           "om_all", "om_hi", "om_lo", "n_viol_hi", "n_tri_hi",
                           "g_best4", "g1", "e0", "ndiff")}
    neg_keep, gam_keep, tables, epi_all, pos_all = [], [], [], [], []
    n_kept = 0
    # разбивка нарушений по размеру A: растёт ли отклонение с глубиной набора
    viol_by_a = np.zeros(3)
    tri_by_a = np.zeros(3)

    with torch.no_grad():
        # раскладка плоских токенов ПОУРОВНЕВАЯ, см. FINDINGS §1
        z_ref = gen(None, nb).reshape(-1, L, P).transpose(1, 2)
        a_ref = dec_lat(latent_from_codes(E, z_ref))
        lg0 = blk(None)

        # ---------- ЧИСЛЕННЫЙ ПОЛ ----------
        # Сравниваем одиночные выигрыши, посчитанные тремя точностями. float64
        # принимаем за истину; расхождение float32 задаёт пол порога, а
        # расхождение bf16 показывает, что было бы без этой правки.
        tok64 = copy.deepcopy(tok).double().eval()
        E64 = projected_codebooks(tok64, args.device)
        u0 = lg0[:, 0].topk(args.rank_hi, -1).indices[
            ar, torch.randint(args.rank_lo, args.rank_hi, (B,),
                              generator=torch.Generator(device=args.device)
                              .manual_seed(99), device=args.device)]
        c0p = z_ref[:, :, 0].clone()
        c0p[:, 0] = u0
        c1p = blk(c0p).argmax(-1)
        zp = torch.stack([c0p, c1p,
                          blk(torch.cat([c0p, c1p], 1)).argmax(-1)], -1)
        stp = zp.clone()
        stp[:, :, 0] = z_ref[:, :, 0]

        def singles_at(m, EE, dt):
            hs = latent_from_codes(EE, stp).to(dt)
            hr = latent_from_codes(EE, z_ref).to(dt)
            ref = dec_with(m, hr)
            e0_ = sq_of(dec_with(m, hs), ref)
            out = []
            for q in range(P):
                h = hs.clone()
                h[:, q] = hr[:, q]
                out.append(e0_ - sq_of(dec_with(m, h), ref))
            return torch.stack(out).double()

        g64 = singles_at(tok64, E64, torch.float64)
        g32 = singles_at(tok32, E, torch.float32)
        gsc = float(g64.max(0).values.median())
        d32 = float((g32 - g64).abs().max())
        floor32 = 3.0 * d32
        print(f"\nЧИСЛЕННЫЙ ПОЛ (масштаб: медианный лучший одиночный "
              f"{gsc:.3e})")
        print(f"  float32 против float64: макс. расхождение {d32:.3e} "
              f"({d32 / max(gsc, 1e-30):.3%} одиночного выигрыша)")
        print(f"  пол порога 3*float32 = {floor32:.3e}")
        print(f"  порог из плана 1e-3*g1 = {1e-3 * gsc:.3e} -> "
              f"{'его и берём' if 1e-3 * gsc >= floor32 else 'ниже пола, берём пол'}")
        # Плечо bfloat16 — справочное. Их декодер не переводится в половинную
        # точность целиком (часть нормировок остаётся во float32 и слой падает),
        # а чужой код мы не трогаем. Рабочий путь float32, и нужное для порога
        # число даёт сравнение с float64 выше, поэтому неудача здесь не мешает.
        try:
            tokbf = copy.deepcopy(tok).to(torch.bfloat16).eval()
            gbf = singles_at(tokbf, projected_codebooks(tokbf, args.device),
                             torch.bfloat16)
            dbf = float((gbf - g64).abs().max())
            print(f"  справочно, bfloat16 против float64: {dbf:.3e} "
                  f"({dbf / max(gsc, 1e-30):.3%})")
            del tokbf, gbf
        except RuntimeError as e:
            print(f"  справочное плечо bfloat16 не считается: "
                  f"{str(e).splitlines()[0]}")
        print()
        del tok64, E64, g64, g32
        torch.cuda.empty_cache()

        for p_ in range(n_pos):
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

            h_st = latent_from_codes(E, stale)
            h_rf = latent_from_codes(E, z_ref)
            e0 = sq_of(dec_lat(h_st), a_ref)                 # e(пусто)
            diff = (stale != z_ref).any(-1)                  # (B, P)
            chg_mask = (diff.int()
                        * (1 << torch.arange(P, device=args.device))).sum(-1)

            # ---------- таблица G(S) для всех 2517 наборов ----------
            G = torch.zeros(B, len(sets), device=args.device)
            for i in range(1, len(sets), args.set_block):
                blockS = sets[i:i + args.set_block]
                hh = h_st.unsqueeze(0).repeat(len(blockS), 1, 1, 1)
                for j, S in enumerate(blockS):
                    hh[j][:, list(S)] = h_rf[:, list(S)]
                ee = sq_of(dec_lat(hh.reshape(-1, P, h_st.shape[-1])),
                           a_ref.repeat(len(blockS), 1, 1))
                G[:, i:i + len(blockS)] = (e0.repeat(len(blockS)) - ee).reshape(
                    len(blockS), -1).T
            Gn = G.cpu().numpy()

            # ---------- Omega и монотонность ----------
            om = Gn[:, tAq] + Gn[:, tAr] - Gn[:, tA] - Gn[:, tAqr]
            mo = Gn[:, mAq] - Gn[:, mA]

            g1 = Gn[:, 1:1 + P].max(1)                       # лучший одиночный
            # порог из плана, но не ниже измеренного численного пола
            tau = max(1e-8, 1e-3 * float(np.median(g1)), floor32)
            neg = np.maximum(0.0, -om)
            viol = om < -tau
            mneg = np.maximum(0.0, -mo)
            mviol = mo < -tau

            cm = chg_mask.cpu().numpy()[:, None]
            # СОДЕРЖАТЕЛЬНЫЕ тройки. Если позиция не изменилась, её латента
            # совпадает с опорной ПОБИТОВО, поэтому G(A+q) = G(A) точно и
            # Omega = 0 тождественно. Значит Omega может быть ненулевой только
            # когда ОБЕ позиции q, r лежат в changed support. Без этого условия
            # знаменатель разбавлен вырожденными тройками примерно в 12 раз, и
            # доля нарушений выглядит вчетверо меньше настоящей.
            nz = ((qr_um[None, :] & ~cm) == 0)                # (B, n_tri)
            inside = (t_um[None, :] & ~cm) == 0               # ещё и A внутри

            # ПРИМИРЕНИЕ С A2. Там Delta ВЫБРАННЫХ наборов систематически
            # отрицательна, здесь знак Omega по всем содержательным тройкам
            # симметричен. Совместимо это лишь если избыточность сосредоточена
            # среди ВЫСОКОВЫИГРЫШНЫХ пар — тех, что и попадают в набор.
            # Проверяем маской «обе позиции входят в top-4 по одиночному
            # выигрышу». Знак: Omega > 0 означает убывающую отдачу, то есть
            # избыточность, и отвечает отрицательной Delta.
            t4 = np.argsort(-Gn[:, 1:1 + P], 1)[:, :4]
            top_m = np.zeros(B, np.int64)
            for j in range(4):
                top_m |= (1 << t4[:, j])
            hi = (qr_um[None, :] & ~top_m[:, None]) == 0
            agg["om_all"].append((om * nz).sum(1) / np.maximum(nz.sum(1), 1))
            agg["om_hi"].append((om * hi).sum(1) / np.maximum(hi.sum(1), 1))
            agg["om_lo"].append((om * (nz & ~hi)).sum(1)
                                / np.maximum((nz & ~hi).sum(1), 1))
            agg["n_viol_hi"].append((viol & hi).sum(1).astype(np.float64))
            agg["n_tri_hi"].append(hi.sum(1).astype(np.float64))

            agg["n_viol"].append(viol.sum(1).astype(np.float64))
            agg["sum_neg"].append((neg * viol).sum(1))
            agg["max_neg"].append(neg.max(1))
            agg["n_tri"].append(np.full(B, om.shape[1], np.float64))
            agg["n_viol_nz"].append((viol & nz).sum(1).astype(np.float64))
            agg["sum_neg_nz"].append((neg * viol * nz).sum(1))
            agg["n_tri_nz"].append(nz.sum(1).astype(np.float64))
            agg["n_viol_ch"].append((viol & inside).sum(1).astype(np.float64))
            agg["sum_neg_ch"].append((neg * viol * inside).sum(1))
            agg["n_tri_ch"].append(inside.sum(1).astype(np.float64))

            # ОТНОШЕНИЕ СУБМОДУЛЯРНОСТИ (Das & Kempe). Для пары {q,r} при
            # префиксе A:  gamma = [m(q|A) + m(r|A)] / m({q,r}|A) = 1 + Om/den.
            # Для субмодулярной функции gamma >= 1; жадный отбор получает
            # гарантию (1 - e^{-gamma}). Величина безразмерная и не зависит от
            # произвольной нормировки, в отличие от «массы нарушений».
            den = Gn[:, tAqr] - Gn[:, tA]
            okd = (den > tau) & nz
            gam = np.where(okd, 1.0 + om / np.where(okd, den, 1.0), np.nan)
            gam_keep.append(gam[okd].astype(np.float32))
            # у монотонности вырожденность та же: если q не изменилась,
            # M(A,q) = 0 тождественно
            mnz = (m_qm[None, :] & ~cm) == 0
            agg["n_mono_viol"].append(mviol.sum(1).astype(np.float64))
            agg["sum_mono_neg"].append((mneg * mviol).sum(1))
            agg["n_mono"].append(np.full(B, mo.shape[1], np.float64))
            agg["n_mono_viol_nz"].append((mviol & mnz).sum(1).astype(np.float64))
            agg["n_mono_nz"].append(mnz.sum(1).astype(np.float64))
            agg["g_best4"].append(Gn.max(1))
            agg["g1"].append(g1)
            agg["e0"].append(e0.cpu().numpy())
            agg["ndiff"].append(diff.sum(-1).cpu().numpy().astype(np.float64))
            epi_all.append(EPI)
            pos_all.append(np.full(B, p_))

            for ka in range(3):
                sl = t_na == ka
                viol_by_a[ka] += viol[:, sl].sum()
                tri_by_a[ka] += viol[:, sl].size

            v = neg[viol]
            if v.size:
                if v.size > args.keep_neg:
                    v = np.random.default_rng(p_).choice(v, args.keep_neg,
                                                         replace=False)
                neg_keep.append(v.astype(np.float32))
            if n_kept < args.keep_tables:
                take = min(B, args.keep_tables - n_kept)
                tables.append(Gn[:take].astype(np.float32))
                n_kept += take
            print(f"  позиция {p_ + 1}/{n_pos}: нарушений "
                  f"{viol.mean():.4%}, tau {tau:.3e}", flush=True)

    Ag = {k: np.concatenate(v) for k, v in agg.items()}
    epi = np.concatenate(epi_all)
    negs = np.concatenate(neg_keep) if neg_keep else np.zeros(1)
    g1m = Ag["g1"].mean()

    print("\n" + "=" * 78)
    print("K-4a4. ПРЯМОЙ ТЕСТ СУБМОДУЛЯРНОСТИ (всё на КВАДРАТЕ ошибки)")
    print("=" * 78)
    print(f"вмешательств {len(epi)}, эпизодов {len(np.unique(epi))}, "
          f"троек на вмешательство {int(Ag['n_tri'][0])}")
    print(f"масштаб: лучший одиночный выигрыш {g1m:.3e}, "
          f"полный доступный выигрыш {Ag['g_best4'].mean():.3e}\n")

    print("""ВЫРОЖДЕННЫЕ ТРОЙКИ. Если позиция не изменилась, её латента совпадает
с опорной побитово, поэтому G(A+q) = G(A) точно и Omega = 0 тождественно.
Omega может быть ненулевой ТОЛЬКО когда обе позиции q, r лежат в changed
support. Доля по всем тройкам разбавлена этими нулями примерно в 12 раз и
самостоятельного смысла не имеет — читать строку «среди содержательных».
""")
    r, lo, hi = cluster_ci(Ag["n_viol"], Ag["n_tri"], epi)
    print(f"  доля нарушений, все тройки           {r:.4%} [{lo:.4%}, {hi:.4%}]"
          f"   <- разбавлена")
    rn, lon, hin = cluster_ci(Ag["n_viol_nz"], Ag["n_tri_nz"], epi)
    print(f"  СРЕДИ СОДЕРЖАТЕЛЬНЫХ (q,r изменены)  {rn:.4%} "
          f"[{lon:.4%}, {hin:.4%}]   <- главное число")
    r2, lo2, hi2 = cluster_ci(Ag["n_viol_ch"], Ag["n_tri_ch"], epi)
    print(f"  строже: ещё и A внутри support       {r2:.4%} [{lo2:.4%}, {hi2:.4%}]")
    print(f"  доля содержательных троек от всех    "
          f"{Ag['n_tri_nz'].sum() / Ag['n_tri'].sum():.2%}")
    r3, lo3, hi3 = cluster_ci(Ag["n_mono_viol"], Ag["n_mono"], epi)
    r4, lo4, hi4 = cluster_ci(Ag["n_mono_viol_nz"], Ag["n_mono_nz"], epi)
    print(f"\n  нарушения монотонности, все пары     {r3:.4%} [{lo3:.4%}, {hi3:.4%}]")
    print(f"  СРЕДИ СОДЕРЖАТЕЛЬНЫХ (q изменена)    {r4:.4%} "
          f"[{lo4:.4%}, {hi4:.4%}]")

    print(f"\n  средняя отрицательная часть Omega на тройку: "
          f"{Ag['sum_neg'].sum() / Ag['n_tri'].sum():.3e} "
          f"({Ag['sum_neg'].sum() / Ag['n_tri'].sum() / max(g1m, 1e-30):.3%} "
          f"одиночного выигрыша)")
    if negs.size > 1:
        print("  величина нарушения (только нарушения), в долях одиночного "
              "выигрыша:")
        for q in (50, 90, 95, 99):
            print(f"    {q}-й процентиль {np.percentile(negs, q) / max(g1m, 1e-30):>10.2%}")
    print(f"    максимум         {Ag['max_neg'].max() / max(g1m, 1e-30):>10.2%}")

    # ОТНОШЕНИЕ СУБМОДУЛЯРНОСТИ. Безразмерно, от нормировки не зависит и прямо
    # даёт гарантию жадного отбора (1 - e^{-gamma}). Для субмодулярной функции
    # gamma >= 1 всюду.
    gam = np.concatenate(gam_keep) if gam_keep else np.zeros(1)
    print(f"\n  ОТНОШЕНИЕ СУБМОДУЛЯРНОСТИ gamma = [m(q|A)+m(r|A)] / m(qr|A)")
    print(f"  (>= 1 для субмодулярной; считается по {len(gam)} тройкам "
          f"со знаменателем выше tau)")
    for q in (1, 5, 25, 50, 75):
        print(f"    {q:>2}-й процентиль {np.percentile(gam, q):>8.3f}")
    print(f"    минимум          {gam.min():>8.3f}")
    print(f"    доля gamma < 1   {(gam < 1).mean():>8.1%}")
    gmin = float(np.percentile(gam, 5))
    print(f"    гарантия жадного при gamma = 5-й проц.: "
          f"1 - exp(-gamma) = {1 - np.exp(-max(gmin, 0)):.3f}")
    print("    ВНИМАНИЕ: минимум gamma неустойчив — при знаменателе чуть выше "
          "tau\n    отношение разлетается. Читать процентили, не минимум. "
          "И сама\n    гарантия здесь пессимистична до бесполезности: жадный "
          "эмпирически\n    сохраняет 97% точного (K-4a2), так что предсказательной "
          "силы у неё нет.")

    # Метрика «масса нарушений» из плана НЕ приводится: сумма max(0,-Omega) по
    # 12720 тройкам, делённая на один скалярный выигрыш, имеет произвольный
    # масштаб, и порог 5% к ней неприменим. Вместо неё — доля нарушений и
    # gamma, обе безразмерные.

    print("\n  ЗНАК Omega ПО СЛОЯМ (Omega > 0 = убывающая отдача = избыточность;"
          "\n  в долях одиночного выигрыша). Примиряет A2 с настоящим замером:")
    rh, loh, hih = cluster_ci(Ag["n_viol_hi"], Ag["n_tri_hi"], epi)
    for nm, key in (("все содержательные тройки", "om_all"),
                    ("обе позиции в top-4 по одиночному", "om_hi"),
                    ("остальные содержательные", "om_lo")):
        v = Ag[key].mean() / max(g1m, 1e-30)
        print(f"    {nm:>36}: среднее Omega {v:>+8.2%}")
    print(f"    доля нарушений среди пар из top-4: {rh:.2%} "
          f"[{loh:.2%}, {hih:.2%}]")
    print("""    Читать: если среднее Omega у пар из top-4 заметно ПОЛОЖИТЕЛЬНО, а по
    всем содержательным около нуля, то избыточность есть, но сосредоточена
    среди высоковыигрышных позиций — именно тех, что берёт независимый отбор.
    Тогда A2 и настоящий замер согласованы, и учить router надо распознавать
    перекрытие сильных позиций, а не общую убывающую отдачу.""")

    print("\n  доля нарушений по размеру A (растёт ли отклонение с глубиной):")
    for ka in range(3):
        print(f"    |A| = {ka}: {viol_by_a[ka] / max(tri_by_a[ka], 1):.4%}"
              f"   троек {int(tri_by_a[ka])}")

    print("\n  макро-среднее по вмешательствам (не по тройкам):")
    mac = Ag["n_viol"] / np.maximum(Ag["n_tri"], 1)
    print(f"    доля нарушений: среднее {mac.mean():.4%}, "
          f"медиана {np.median(mac):.4%}, "
          f"90-й проц. {np.percentile(mac, 90):.4%}, макс {mac.max():.4%}")
    print(f"    доля вмешательств хотя бы с одним нарушением: "
          f"{(Ag['n_viol'] > 0).mean():.1%}")

    print("\n" + "=" * 78)
    print("ВЫВОД")
    print("=" * 78)
    print("""МЕТРИКА ВОРОТ ИЗ ПЛАНА НЕПРИГОДНА и здесь не приводится: «масса
нарушений» — это сумма max(0,-Omega) по 12720 тройкам, делённая на один
скалярный выигрыш; её масштаб произвольный, и порог 5% к ней неприменим. Это
дефект спецификации метрики, а не результат.

Вывод при этом порога не требует и от нормировки не зависит:""")
    if rn < 0.05:
        print(f"  доля нарушений среди содержательных троек {rn:.2%} мала и "
              f"gamma в основном >= 1 -> приближённая субмодулярность,\n"
              f"  жадный отбор получает теоретическое основание.")
    elif rn < 0.20:
        print(f"  доля нарушений {rn:.2%}: отклонения заметны, но редки. "
              f"Формулировать как «преимущественно убывающая отдача\n"
              f"  с заметным хвостом», гарантию жадного брать по gamma, а не "
              f"по классической 1-1/e.")
    else:
        print(f"  доля нарушений среди содержательных троек {rn:.2%} — знак "
              f"взаимодействия близок к подбрасыванию монеты.\n"
              f"  ФУНКЦИЯ НЕ СУБМОДУЛЯРНА. Формулировка «приближённая "
              f"субмодулярность» недоступна; писать\n"
              f"  «наборы в среднем избыточны, но знак локального "
              f"взаимодействия непостоянен».")
    print("""
Направление это НЕ закрывает: оракульная разреженность и провал независимого
ранжирования от субмодулярности не зависят. Закрывается лишь возможность
опереть новизну на субмодулярность и на классическую гарантию жадного отбора.
ОТДЕЛЬНО. Нарушения МОНОТОННОСТИ означают, что добавление позиции способно
ухудшить результат. Тогда оракулу необходимо право отказа (пустой набор в
переборе), а router нельзя учить правилу «чем больше позиций, тем лучше».""")

    if args.dump:
        os.makedirs(os.path.dirname(args.dump) or ".", exist_ok=True)
        np.savez_compressed(
            args.dump, commit=commit, seed=args.seed, epi=epi,
            pos=np.concatenate(pos_all), neg_sample=negs, gamma=gam,
            tables=np.concatenate(tables) if tables else np.zeros(1),
            **{k: v for k, v in Ag.items()})
        print(f"\nсырые величины сохранены: {args.dump}")


if __name__ == "__main__":
    main()
