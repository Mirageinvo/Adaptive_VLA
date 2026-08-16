"""K-4a, фаза A: устойчив ли групповой оракул и ЧЕМ объясняется его выигрыш.

K-4a дал 0.90 закрытого разрыва при четырёх позициях из шестнадцати против
0.77 у одиночного ранжирования. Прежде чем обучать router, надо закрыть три
вопроса, два из которых способны обнулить сам результат.

A0. СКОЛЬКО ПОЗИЦИЙ ВООБЩЕ ОТЛИЧАЮТСЯ. stale и z_ref расходятся только там,
    где правка coarse изменила тонкие уровни. Если расходятся всего пять
    позиций, то «четыре из шестнадцати» — это четыре из пяти, никакой
    разреженности нет, а случайный baseline занижен разбавлением по шестнадцати
    ячейкам. Этого замера в K-4a не было, и он проверяется первым.

ИСПРАВЛЕНО ПОСЛЕ АУДИТА (вторая версия).
  1. НАПРАВЛЕНИЕ ВРЕМЕНИ У ЛОГИТОВ. c0_old содержит худший код u, то есть
     состояние ДО правки. Значит blk(c0_old) — проход, который уже состоялся,
     а blk(z_ref[:,:,0]) — новый плотный проход ПОСЛЕ правки. В первой версии
     подписи стояли наоборот, и «энтропия старого прохода» бралась из
     будущего. Причинные baseline'ы отозваны и пересчитаны.
  2. КОРЕНЬ ПОРОЖДАЕТ СУПЕРАДДИТИВНОСТЬ САМ ПО СЕБЕ. e(S) = sqrt(sum d_q^2),
     корень вогнут, поэтому G({q,r}) >= G({q}) + G({r}) даже при полностью
     независимых ошибках позиций; при пяти равных изменившихся позициях
     артефакт даёт +124% одиночного выигрыша. Delta теперь считается и на
     КВАДРАТЕ ошибки, где нулевое значение означает ровно аддитивность.
     Заметим: разницу между жадным и одиночным этот артефакт породить НЕ
     может — в разложимой модели оба выбирают один и тот же набор.
  3. ТЕСТ ЭКВИВАЛЕНТНОСТИ для multi-segment вместо сравнения округлённых
     точек: односторонняя верхняя граница отставания против порога.
  4. Пустой набор в точном переборе — оракул вправе отказаться от ремонта.
  5. Случайные отборы усредняются по многим сидам.
  6. В разбивках печатается число уникальных наблюдений и эпизодов, интервалы
     кластерные.

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

ЧТО ДОСТУПНО ROUTER'У. Логиты lg_before = blk(c0_old) получены проходом,
который УЖЕ состоялся до правки: его argmax и лежит в stale. Поэтому их
энтропия и запас top1-top2 допустимы как признаки. Логиты
lg_after = blk(z_ref[:,:,0]) — это новый плотный проход после правки u -> v,
который мы и хотим сэкономить; они и JS по ним разрешены только как оракульная
верхняя граница.

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


def paired_diff_ci(g_a, g_b, base, epi, n_boot: int = 2000, seed: int = 0):
    """Разность ДВУХ долей закрытого разрыва, кластерный бутстрап по эпизодам.

    D = sum(g_a)/sum(base) - sum(g_b)/sum(base), пересчитывается внутри каждой
    реплики. Возвращает точку, двусторонний 95% интервал и ОДНОСТОРОННЮЮ
    верхнюю границу 95% — она и нужна для теста эквивалентности «отставание не
    превышает порога»."""
    rng = np.random.default_rng(seed)
    eps = np.unique(epi)
    idx = {e: np.where(epi == e)[0] for e in eps}
    point = (g_a.sum() - g_b.sum()) / max(base.sum(), 1e-12)
    out = []
    for _ in range(n_boot):
        s = np.concatenate([idx[e] for e in rng.choice(eps, len(eps), replace=True)])
        out.append((g_a[s].sum() - g_b[s].sum()) / max(base[s].sum(), 1e-12))
    out = np.asarray(out)
    return (point, float(np.percentile(out, 2.5)),
            float(np.percentile(out, 97.5)), float(np.percentile(out, 95)))


def selftest_metric_artifact(P: int = 16, B: int = 4000, seed: int = 0) -> None:
    """Проверка оценки Delta на синтетике С ИЗВЕСТНЫМ ОТВЕТОМ.

    Строим РАЗЛОЖИМУЮ модель: ошибка чанка — сумма независимых вкладов позиций,
    взаимодействий нет по построению. Правильная оценка обязана дать ноль.
    Проверяем, что на квадрате ноль и получается, а на RMS возникает крупное
    положительное Delta из ничего."""
    rng = np.random.default_rng(seed)
    d2 = rng.gamma(0.6, 1.0, size=(B, P)) * (rng.random((B, P)) < 0.3)
    z = np.zeros((B, P), bool)

    def er(m):
        return np.sqrt((d2 * ~m).sum(1) / P)

    def em(m):
        return (d2 * ~m).sum(1) / P

    def gains(sel):
        m = np.zeros((B, P), bool)
        np.put_along_axis(m, sel, True, 1)
        return er(z) - er(m), em(z) - em(m)

    sr = np.stack([gains(np.full((B, 1), q))[0] for q in range(P)], 1)
    sm = np.stack([gains(np.full((B, 1), q))[1] for q in range(P)], 1)
    top = np.argsort(-sr, 1)[:, :4]
    gr, gm = gains(top)
    dr = (gr - np.take_along_axis(sr, top, 1).sum(1)).mean() / sr.max(1).mean()
    dm = (gm - np.take_along_axis(sm, top, 1).sum(1)).mean() / sm.max(1).mean()
    same = (np.sort(top, 1) == np.sort(np.argsort(-sm, 1)[:, :4], 1)).all()
    print("САМОПРОВЕРКА на разложимой модели (взаимодействий НЕТ):")
    print(f"  Delta на RMS      {dr:+.1%} одиночного выигрыша  <- АРТЕФАКТ")
    print(f"  Delta на КВАДРАТЕ {dm:+.1%}  <- верно, ноль")
    print(f"  top-4 по RMS и по квадрату совпадают: {same}")
    if abs(dm) > 1e-6:
        raise SystemExit("самопроверка провалена: Delta на квадрате не ноль")
    print("  вывод: разницу ЖАДНЫЙ/ОДИНОЧНЫЙ артефакт корня породить не может,"
          "\n  а величину Delta — вполне; про синергию судить только по"
          " квадрату.\n")


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
    ap.add_argument("--rand-seeds", type=int, default=20,
                    help="сидов для случайных отборов; одна реализация шума "
                         "неотличима от систематического эффекта")
    ap.add_argument("--equiv-margin", type=float, default=0.03,
                    help="допустимое отставание multi-segment от произвольного")
    ap.add_argument("--dump", default="logs/k4a2_phase_a.npz",
                    help="куда сложить сырые величины; пусто — не сохранять")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    selftest_metric_artifact()

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

    def sq_of(dec, ref):
        """СРЕДНИЙ КВАДРАТ ошибки по непрерывным каналам на исполняемом окне.

        Нужен отдельно от RMS. Корень ВОГНУТ, поэтому на нём неаддитивность
        возникает механически даже когда ошибки позиций независимы:
        e(S) = sqrt(sum_{q не в S} d_q^2), и для вогнутой f выполняется
        f(a) + f(b) >= f(0) + f(a+b), то есть G({q,r}) >= G({q}) + G({r})
        БЕЗ всякого взаимодействия. При пяти равных изменившихся позициях
        артефакт даёт Delta(4-набор) около +124% одиночного выигрыша.
        Поэтому доля закрытого разрыва считается на RMS (так предписано
        планом), а всё про синергию — ТОЛЬКО на квадрате, где «нет
        взаимодействия» означает ровно аддитивность."""
        d = (dec[:, :args.window] - ref[:, :args.window]).abs()[..., :D_act - 1]
        return d.flatten(1).pow(2).mean(-1)

    def err_of(dec, ref):
        """RMS. Схват выведен отдельно: бинарный канал ошибается примерно на
        единицу в моменты переключения и подминает метрику."""
        return sq_of(dec, ref).sqrt() / scale

    def err_lat(h, ref):
        return err_of(dec_lat(h), ref)

    def errs_lat(h, ref):
        """RMS и квадрат за ОДНО декодирование."""
        s = sq_of(dec_lat(h), ref)
        return s.sqrt() / scale, s / scale ** 2

    def grip_of(dec, ref):
        """Доля позиций окна, где знак схвата разошёлся с опорой."""
        a = dec[:, :args.window, -1] > 0
        b = ref[:, :args.window, -1] > 0
        return (a != b).float().mean(-1)

    rng = torch.Generator(device=args.device).manual_seed(1)
    # ОТДЕЛЬНЫЙ поток для случайных отборов: иначе двадцать сидов на каждом шаге
    # сдвигают общий генератор, и выборка возмущений u перестаёт совпадать с
    # прежними прогонами — числа стали бы несравнимы по чужой причине
    rng_sel = torch.Generator(device=args.device).manual_seed(1234)
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
        # НАПРАВЛЕНИЕ ВРЕМЕНИ, исправлено после аудита. c0_old содержит худший
        # код u, то есть состояние ДО правки; поэтому blk(c0_old) — проход,
        # который уже состоялся, а blk(z_ref[:,:,0]) — тот самый НОВЫЙ плотный
        # проход после правки u -> v, который router и должен сэкономить.
        # Раньше подписи стояли наоборот, и «старая энтропия» бралась из
        # будущего. lg_after от p не зависит и считается один раз.
        lg_after = blk(z_ref[:, :, 0])          # ОРАКУЛ: новый плотный проход

        # ошибка самой модели относительно датасета — масштаб для A5
        e_model_true = err_of(a_ref, a_true).cpu().numpy()
        g_model_true = grip_of(a_ref, a_true).cpu().numpy()

        keys = ("greedy", "greedy_stop", "singleton", "same", "js", "window",
                "random", "random_chg", "exact", "ent_win", "ent_topk",
                "marg_topk")
        res = {k: {K: [] for K in BUDGETS} for k in keys}
        fam_g = {k: [] for k in fam}
        div_g = {}                       # (вид похожести, lambda) -> список
        es_all, gr_st, gr_gr = [], [], []
        ndiff_all, feat = [], {k: [] for k in
                               ("task", "speed", "gswitch", "edit", "p")}
        # Delta считаем И на RMS (для сопоставимости с прежним отчётом), И на
        # КВАДРАТЕ (единственная величина, где нулевое Delta = аддитивность)
        dlt = {f"{a}_{b}": [] for a in ("sing", "greedy", "exact")
               for b in ("rms", "mse")}
        g1_mse = []                      # масштаб: типичный одиночный на квадрате
        rand_spread = []                 # разброс закрытой доли по сидам, K=4
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
            # ЭТОТ проход состоялся ДО правки: его argmax и лежит в stale.
            # Только его величины допустимы как признаки для router.
            lg_before = blk(c0_old)
            c1_old = lg_before.argmax(-1)
            ent_before = -(lg_before.softmax(-1)
                           * lg_before.log_softmax(-1)).sum(-1)
            t2 = lg_before.topk(2, -1).values
            marg_before = t2[..., 0] - t2[..., 1]
            z_old = torch.stack([c0_old, c1_old,
                                 blk(torch.cat([c0_old, c1_old], 1)).argmax(-1)], -1)
            stale = z_old.clone()
            stale[:, :, 0] = z_ref[:, :, 0]

            h_st = latent_from_codes(E, stale)
            h_rf = latent_from_codes(E, z_ref)
            dec_st = dec_lat(h_st)
            e_st = err_of(dec_st, a_ref)
            e2_st = sq_of(dec_st, a_ref) / scale ** 2
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

            def sel_to_h(sel):
                h = h_st.clone()
                idx = ar.unsqueeze(1).expand_as(sel)
                h[idx, sel] = h_rf[idx, sel]
                return h

            def g_of_sel(sel):
                """Выигрыш для ПОЭКЗЕМПЛЯРНОГО набора sel (B, K)."""
                return e_st - err_lat(sel_to_h(sel), a_ref)

            def both_of_sel(sel):
                """Выигрыш на RMS и на КВАДРАТЕ за одно декодирование."""
                r, m = errs_lat(sel_to_h(sel), a_ref)
                return e_st - r, e2_st - m

            sr, sm = [], []
            for q in range(P):
                h = h_st.clone()
                h[:, q] = h_rf[:, q]
                r, m = errs_lat(h, a_ref)
                sr.append(e_st - r)
                sm.append(e2_st - m)
            singles = torch.stack(sr)                       # (P, B) RMS
            singles2 = torch.stack(sm)                      # (P, B) квадрат
            sing_T, sing2_T = singles.T, singles2.T

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
                greedy_gain[step + 1] = (e_st - e_cur).cpu().numpy()

            # жадный с ПРАВОМ ОТКАЗА: лучший префикс длины <= K. Нужен для
            # честного сравнения с точным перебором, куда добавлен пустой набор
            cum = np.stack([greedy_gain[k + 1] for k in range(max(BUDGETS))], 1)
            cum = np.concatenate([np.zeros((B, 1), np.float32), cum], 1)

            # ---------- A2: неаддитивность ВЫБРАННЫХ наборов ----------
            # На RMS — для сопоставимости с прежним отчётом; на КВАДРАТЕ — для
            # вывода о механизме, потому что вогнутость корня даёт Delta > 0
            # даже при полном отсутствии взаимодействия (см. sq_of).
            ssel = sing_T.argsort(-1, descending=True)[:, :4]
            gs_r, gs_m = both_of_sel(ssel)
            gg_r, gg_m = both_of_sel(gsel[:, :4])
            dlt["sing_rms"].append((gs_r - sing_T.gather(1, ssel).sum(-1)).cpu().numpy())
            dlt["sing_mse"].append((gs_m - sing2_T.gather(1, ssel).sum(-1)).cpu().numpy())
            g4 = gsel[:, :4]
            dlt["greedy_rms"].append((gg_r - sing_T.gather(1, g4).sum(-1)).cpu().numpy())
            dlt["greedy_mse"].append((gg_m - sing2_T.gather(1, g4).sum(-1)).cpu().numpy())
            g1_mse.append(sing2_T.max(-1).values.cpu().numpy())

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
            # JS симметрична, поэтому порядок аргументов роли не играет; но
            # величина ОРАКУЛЬНАЯ: её вычисление требует нового плотного прохода
            rank_js = js_div(lg_before.softmax(-1), lg_after.softmax(-1))
            # ПРИЧИННЫЕ признаки: только из прохода ДО правки. Малый запас
            # top1-top2 = неуверенность, поэтому берём его со знаком минус
            for K in BUDGETS:
                res["greedy"][K].append(greedy_gain[K])
                res["greedy_stop"][K].append(cum[:, :K + 1].max(1))
                for nm, sc in (("singleton", sing_T), ("js", rank_js),
                               ("ent_topk", ent_before),
                               ("marg_topk", -marg_before)):
                    res[nm][K].append(
                        g_of_sel(sc.argsort(-1, descending=True)[:, :K]).cpu().numpy())
                sm = torch.full((B, P), -1e9, device=args.device)
                sm[:, p_] = 1.0
                res["same"][K].append(
                    g_of_sel(sm.argsort(-1, descending=True)[:, :1]).cpu().numpy()
                    if K == 1 else res["same"][1][-1])
                wins = [tuple(range(s, s + K)) for s in range(P - K + 1)]
                res["window"][K].append(g_of(wins).max(0).values.cpu().numpy())
                # DEPLOYABLE окно: центр в позиции максимальной энтропии ДО правки
                ctr = ent_before.argmax(-1)
                lo = (ctr - K // 2).clamp(0, P - K)
                sel = lo.unsqueeze(1) + torch.arange(K, device=args.device)
                res["ent_win"][K].append(g_of_sel(sel).cpu().numpy())
                # случайные отборы: усреднение по многим сидам, иначе одна
                # реализация шума неотличима от систематического эффекта
                acc = {"random": np.zeros(B, np.float32),
                       "random_chg": np.zeros(B, np.float32)}
                per_seed = []
                for _ in range(args.rand_seeds):
                    r0 = torch.rand(B, P, generator=rng_sel, device=args.device)
                    gr_ = g_of_sel(r0.argsort(-1, descending=True)[:, :K]).cpu().numpy()
                    acc["random"] += gr_ / args.rand_seeds
                    r1 = torch.rand(B, P, generator=rng_sel,
                                    device=args.device) + diff.float()
                    acc["random_chg"] += g_of_sel(
                        r1.argsort(-1, descending=True)[:, :K]).cpu().numpy() / args.rand_seeds
                    if K == 4:
                        per_seed.append(gr_.sum() / max(e_st.sum().item(), 1e-12))
                res["random"][K].append(acc["random"])
                res["random_chg"][K].append(acc["random_chg"])
                if K == 4:
                    rand_spread.append(np.array(per_seed))

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
                # ПУСТОЙ НАБОР включён: G(пусто) = 0, поэтому нулевая начальная
                # величина ровно и означает право оракула отказаться от ремонта.
                # Без этого оракул был бы обязан испортить там, где всякая
                # правка вредна. Индекс -1 обозначает пустой набор.
                best = torch.zeros(nsub, device=args.device)
                best_i = torch.full((nsub,), -1, dtype=torch.long,
                                    device=args.device)
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
                exact_greedy.append(cum[sub, :5].max(1))   # жадный с отказом
                exact_epi.append(EPI[sub])
                won = [() if int(k) < 0 else all_sub[int(k)]
                       for k in best_i.cpu().numpy()]
                exact_size.append(np.array([len(S) for S in won]))
                exact_span.append(np.array(
                    [0 if not S else max(S) - min(S) + 1 for S in won]))
                exact_contig.append(np.array(
                    [1.0 if len(S) < 2 else float(max(S) - min(S) + 1 == len(S))
                     for S in won]))
                # Delta победившего набора: на RMS и на КВАДРАТЕ. Второй строим
                # отдельным декодированием, иначе квадрат для этого набора
                # неизвестен.
                hw = h_st[sub].clone()
                for k, S in enumerate(won):
                    if S:
                        hw[k, list(S)] = h_rf[sub][k, list(S)]
                _, mw = errs_lat(hw, aref_s)
                gw_m = e2_st[sub] - mw
                sr_ = torch.stack([sing_T[sub][k, list(S)].sum() if S
                                   else torch.zeros((), device=args.device)
                                   for k, S in enumerate(won)])
                sm_ = torch.stack([sing2_T[sub][k, list(S)].sum() if S
                                   else torch.zeros((), device=args.device)
                                   for k, S in enumerate(won)])
                dlt["exact_rms"].append((best - sr_).cpu().numpy())
                dlt["exact_mse"].append((gw_m - sm_).cpu().numpy())
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
ЧИТАТЬ ТАК: это ДВЕ РАЗНЫЕ разреженности, и путать их нельзя.
  1) РАЗРЕЖЕН САМ SUPPORT. Правка coarse меняет тонкие уровни лишь в части
     позиций, остальные полному пересчёту вообще не подлежат.
  2) ВНУТРИ support полезность неоднородна: случайные K изменённых закрывают
     заметно меньше, чем оптимально выбранные K (строки ниже).
Формулировка: полный пересчёт затрагивает небольшое подмножество позиций, а
основная полезность внутри него сосредоточена в структурированной группе.
Строка «случ. средь измен.(орк)» — ОРАКУЛЬНАЯ диагностика, а не baseline:
узнать состав support без полного прохода нельзя.
ДВА ЗНАМЕНАТЕЛЯ: для ЭКОНОМИИ вычислений — K/16, потому что плотный проход всё
равно пересчитывает все шестнадцать; для заявки о структуре — число реально
затронутых позиций.""")

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
Отношение много меньше единицы -> даже идеальное закрытие разрыва почти не
меняет действие, и вся линия имеет низкий потолок. Порядка единицы и выше ->
закрывать есть что.""")

    print("\n" + "=" * 80)
    print("A1/A3. ДОЛЯ РАЗРЫВА, ЗАКРЫТАЯ ПЕРЕСЧЁТОМ K ПОЗИЦИЙ (равное число позиций)")
    print("=" * 80)
    print("ОРАКУЛЫ подсматривают результат; ПРИЧИННЫЕ пользуются только "
          "проходом ДО правки")
    names = {"exact": "точный <=4 (орк)", "greedy": "жадный послед. (орк)",
             "greedy_stop": "жадный с отказом (орк)",
             "singleton": "одиночный (орк)", "window": "лучшее окно (орк)",
             "js": "по JS (орк)", "ent_win": "окно по энтр. ДО (прич.)",
             "ent_topk": "top-K энтр. ДО (прич.)",
             "marg_topk": "top-K запас ДО (прич.)", "same": "только p (прич.)",
             "random": f"случайно из 16, {args.rand_seeds} сид.",
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
        print(f"  жадный (с отказом) сохраняет от точного: "
              f"{ex_gre.sum() / max(ex.sum(), 1e-9):.1%}")
        sz = np.concatenate(exact_size)
        print(f"  размер победившего набора: среднее {sz.mean():.2f}, "
              + " ".join(f"{i}:{(sz == i).mean():.0%}" for i in (0, 1, 2, 3, 4)))
        sp_ = np.concatenate(exact_span)
        print(f"  протяжённость победившего набора во времени: "
              f"среднее {sp_.mean():.2f} из {P}")
        print(f"  доля НЕПРЕРЫВНЫХ победивших наборов: "
              f"{np.concatenate(exact_contig).mean():.1%}")
        if rand_spread:
            rs = np.stack(rand_spread)      # (n_pos, n_seeds)
            print(f"  разброс случайного отбора при K=4 по "
                  f"{args.rand_seeds} сидам: ст.откл. {rs.std(1).mean():.4f} "
                  f"доли разрыва")

    print("\n" + "=" * 80)
    print("A3. СТРУКТУРИРОВАННЫЕ ВРЕМЕННЫЕ СЕМЕЙСТВА, K = 4, ОРАКУЛ ВНУТРИ СЕМЕЙСТВА")
    print("=" * 80)
    for nm in fam:
        print(f"{nm:>26}  {fmt(np.concatenate(fam_g[nm]))}")
    print(f"{'произвольный (жадный)':>26}  {fmt(np.concatenate(res['greedy'][4]))}")
    if res["exact"][4]:
        print(f"{'произвольный (точный)':>26}  "
              f"{fmt(np.concatenate(res['exact'][4]), ex_base, ex_epi)}")
    # ТЕСТ ЭКВИВАЛЕНТНОСТИ. По округлённым точкам ворота объявлять нельзя:
    # 0.90 против 0.93 при пороге 0.03 — ровно на границе. Нужна верхняя
    # граница одностороннего 95% интервала для отставания.
    ms = np.concatenate(fam_g["<= 2 отрезков, всего 4"])
    if res["exact"][4] and len(np.concatenate(res["exact"][4])) == len(ms):
        ref_g, ref_nm, bse, epe = (np.concatenate(res["exact"][4]), "точного",
                                   ex_base, ex_epi)
    else:
        # точный шёл на подвыборке — сравниваем с жадным на общих строках
        ref_g, ref_nm, bse, epe = (np.concatenate(res["greedy"][4]), "жадного",
                                   es, epi_rep)
    d_pt, d_lo, d_hi, d_up = paired_diff_ci(ref_g, ms, bse, epe)
    print(f"\n  ОТСТАВАНИЕ multi-segment от {ref_nm} произвольного:")
    print(f"    точка {d_pt:.4f}, 95% ДИ [{d_lo:.4f}, {d_hi:.4f}], "
          f"верхняя односторонняя 95% граница {d_up:.4f}")
    ok = d_up <= args.equiv_margin
    print(f"    порог эквивалентности {args.equiv_margin:.2f} -> "
          f"{'ЭКВИВАЛЕНТНОСТЬ ПОДТВЕРЖДЕНА' if ok else 'НЕ ПОДТВЕРЖДЕНА'}")
    print("""
РЕШЕНИЕ ПО АРХИТЕКТУРЕ, зафиксировано до запуска:
  верхняя односторонняя граница отставания <= 0.03 -> multi-segment router,
      он проще и аппаратно дружелюбнее;
  граница выше 0.03 -> оставить произвольный set-router как основной.
По одним лишь округлённым точкам ворота не объявляются.
Все семейства здесь ОРАКУЛЬНЫЕ: они подсматривают результат и baseline'ами
не являются.""")

    print("\n" + "=" * 80)
    print("A2. СИНЕРГИЯ ИЛИ ИЗБЫТОЧНОСТЬ: неаддитивность ВЫБРАННОГО набора")
    print("=" * 80)
    g1 = np.concatenate(res["singleton"][1]).mean()
    g1m = np.concatenate(g1_mse).mean()
    print(f"масштаб: типичный одиночный выигрыш, RMS {g1:.5f}, "
          f"квадрат {g1m:.6f}\n")
    print("""ВНИМАНИЕ. На RMS положительное Delta возникает МЕХАНИЧЕСКИ: корень
вогнут, и G({q,r}) >= G({q}) + G({r}) даже при полностью независимых ошибках
позиций. Единственная величина, где нулевое Delta означает ровно аддитивность,
— КВАДРАТ ошибки. Столбец RMS оставлен только для сопоставимости с прежним
отчётом; выводы делать по столбцу «квадрат».""")
    print(f"\n{'набор':>24}{'Delta RMS':>12}{'дол.':>8}"
          f"{'Delta квадрат':>15}{'дол.':>8}{'доля >0 (кв.)':>15}")
    for nm, key in (("одиночный top-4", "sing"), ("жадный 4", "greedy"),
                    ("точный <=4", "exact")):
        if not dlt[f"{key}_rms"]:
            continue
        dr = np.concatenate(dlt[f"{key}_rms"])
        dm = np.concatenate(dlt[f"{key}_mse"])
        print(f"{nm:>24}{dr.mean():>12.5f}{dr.mean() / max(abs(g1), 1e-12):>7.0%}"
              f"{dm.mean():>15.6f}{dm.mean() / max(abs(g1m), 1e-12):>7.0%}"
              f"{(dm > 0).mean():>15.1%}")

    # РАЗЛОЖЕНИЕ перехода «одиночный top-4 -> лучший набор», чтобы не считать
    # его руками в отчёте. Всё в долях одиночного выигрыша на КВАДРАТЕ.
    if dlt["exact_mse"] and len(np.concatenate(dlt["exact_mse"])) == len(es):
        best_key, best_nm = "exact", "точный"
    else:
        best_key, best_nm = "greedy", "жадный"
    d_s = np.concatenate(dlt["sing_mse"])
    d_b = np.concatenate(dlt[f"{best_key}_mse"])
    if len(d_b) == len(d_s):
        print(f"\nРАЗЛОЖЕНИЕ перехода «одиночный top-4 -> {best_nm}», "
              f"в долях одиночного выигрыша (квадрат):")
        di = (d_b - d_s).mean() / max(abs(g1m), 1e-12)
        print(f"  прирост взаимодействия      {di:>+8.1%}")
        print(f"  (остаток — отданный индивидуальный выигрыш; он отрицателен, "
              f"иначе набор совпал бы с одиночным)")
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
ЧИТАТЬ ТАК, по столбцу КВАДРАТ.
  Delta(жадный/точный) около нуля -> супераддитивности нет, прежний вывод о
      синергии был артефактом корня; преимущество жадного объясняется тем, что
      он избегает взаимозаменяемых позиций.
  Delta(жадный/точный) заметно положительна, а Delta(одиночного) около нуля ->
      СУПЕРАДДИТИВНОСТЬ ГРУППОВОГО РЕМОНТА подтверждена.
ОГРАНИЧЕНИЕ ФОРМУЛИРОВКИ. Измеряется супераддитивность в пространстве
ДЕКОДИРОВАННЫХ ДЕЙСТВИЙ. Это не то же самое, что зависимость токенов внутри
потока: вклад может давать нелинейность декодера ActionCodec и его временное
поле восприятия. Писать «супераддитивность группового action-repair».
ПРО ШТРАФ ЗА ПОХОЖЕСТЬ. Он показывает лишь то, что ДВА ЗАДАННЫХ ВРУЧНУЮ
штрафа не объясняют выигрыш жадного. Это НЕ доказывает, что любой независимый
или однопроходный router проиграет последовательному: одношаговый предсказатель
набора может выучить взаимодействия и без последовательного выбора. Сравнение
independent / one-shot set / one-shot multi-segment / sequential обязательно
в K-4b.""")

    print("\n" + "=" * 80)
    print("A4. РАЗБИВКА ПО УСЛОВИЯМ (закрытая доля при K = 4)")
    print("=" * 80)
    F = {k: np.concatenate(v) for k, v in feat.items()}
    gg4 = np.concatenate(res["greedy"][4])
    gs4 = np.concatenate(res["singleton"][4])
    obs_rep = np.tile(np.arange(B), n_pos)

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
        # n вмешательств вводит в заблуждение: строки размножены по 16 позициям
        # p, поэтому n=64 может отвечать всего четырём исходным наблюдениям.
        # Печатаем ещё число уникальных наблюдений и эпизодов, а интервалы
        # берём кластерными по эпизодам.
        print(f"{'группа':>40}{'n':>6}{'набл':>6}{'эпиз':>6}"
              f"{'жадный':>20}{'одиночн.':>10}")
        for nm, m in groups:
            if m.sum() < 20:
                continue
            n_ob, n_ep_ = len(np.unique(obs_rep[m])), len(np.unique(epi_rep[m]))
            a = fmt(gg4[m], es[m], epi_rep[m]) if n_ep_ >= 4 else \
                f"{gg4[m].sum() / max(es[m].sum(), 1e-12):.2f} (мало кл.)"
            b = gs4[m].sum() / max(es[m].sum(), 1e-12)
            print(f"{nm[:40]:>40}{int(m.sum()):>6}{n_ob:>6}{n_ep_:>6}"
                  f"{a:>20}{b:>10.2f}")

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
        print(f"{'задача':>52}{'n':>6}{'набл':>6}{'эпиз':>6}{'жадный':>20}")
        for t in top:
            m = ts == t
            if m.sum() < 20:
                continue
            n_ob, n_ep_ = len(np.unique(obs_rep[m])), len(np.unique(epi_rep[m]))
            a = fmt(gg4[m], es[m], epi_rep[m]) if n_ep_ >= 4 else \
                f"{gg4[m].sum() / max(es[m].sum(), 1e-12):.2f} (мало кл.)"
            print(f"{t[:52]:>52}{int(m.sum()):>6}{n_ob:>6}{n_ep_:>6}{a:>20}")

    if args.dump:
        os.makedirs(os.path.dirname(args.dump) or ".", exist_ok=True)
        out = {"e_st": es, "epi": epi_rep, "obs": obs_rep,
               "ndiff": ndiff, "e_model_true": e_model_true}
        for k in res:
            for K in BUDGETS:
                if res[k][K]:
                    out[f"g_{k}_{K}"] = np.concatenate(res[k][K])
        for k, v in dlt.items():
            if v:
                out[f"delta_{k}"] = np.concatenate(v)
        for k, v in feat.items():
            out[f"feat_{k}"] = np.concatenate(v)
        for i, nm in enumerate(fam):
            out[f"fam_{i}"] = np.concatenate(fam_g[nm])
        if exact_base:
            out["exact_base"] = ex_base
            out["exact_epi"] = ex_epi
            out["exact_size"] = np.concatenate(exact_size)
        np.savez_compressed(args.dump, **out)
        print(f"\nсырые величины сохранены: {args.dump}")

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
    print(f"  A3: верхняя граница отставания multi-segment "
          f"<= {args.equiv_margin:.2f} -> {d_up:.4f} "
          f"{'ПРОЙДЕНО' if d_up <= args.equiv_margin else 'НЕ ПРОЙДЕНО'}")
    best_dep = max((np.concatenate(res[k][4]).sum() / max(es.sum(), 1e-12), k)
                   for k in ("ent_topk", "ent_win", "marg_topk", "same",
                             "random"))
    print(f"  планка для K-4b: лучший ПРИЧИННЫЙ отбор при K=4 — "
          f"{best_dep[0]:.2f} ({names[best_dep[1]]}), оракул {pt:.2f}")
    print("  A0 и A2 ворот не имеют: они определяют, ЧТО именно должен учить "
          "router, и корректна ли сама постановка про разреженность.")


if __name__ == "__main__":
    main()
