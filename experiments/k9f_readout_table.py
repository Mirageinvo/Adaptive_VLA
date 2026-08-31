"""K-9f: факториальная таблица «ствол x голова» и контроль head-only.

ВОПРОС. Прирост согласия 23.1 -> 33.0 получен обучением ствола и головы
вместе. Отсюда не следует, что информация переехала на слой 12: голова могла
просто научиться лучше читать то же самое представление. Пока это не
разделено, фразу «первые 12 слоёв научились нести грубый код» писать нельзя.

ЧЕТЫРЕ КЛЕТКИ.

    ствол      голова      что это
    исходный   исходная    опора, ожидается 23.1%
    исходный   ОБУЧЕННАЯ   head-only: сколько даёт одна голова
    ОБУЧЕННЫЙ  исходная    вклад изменения представления
    ОБУЧЕННЫЙ  ОБУЧЕННАЯ   Joint-12, ожидается 33.0%

ЧИТАЕТСЯ ТАК, и это записано ДО запуска:
  * head-only около 33% -> весь прирост принадлежит голове, переноса не было,
    и главное утверждение K-9 закрывается отрицательно;
  * head-only около 24-26% -> прирост принадлежит стволу;
  * клетка «обученный ствол + исходная голова» доказательна АСИММЕТРИЧНО:
    высокое значение — сильное свидетельство переноса, низкое не значит
    ничего, потому что ствол и голова со-адаптировались и исходная голова
    просто не умеет читать новое кодирование. Отрицательный вывод по этой
    клетке делать нельзя.

ДВЕ КЛЕТКИ ИЗ ЧЕТЫРЁХ ИМЕЮТ ИЗВЕСТНЫЙ ОТВЕТ, и это главная защита от ошибки в
самом конвейере: первая обязана воспроизвести согласие эпохи 0, четвёртая —
эпохи 3. Если они не сходятся, таблица недействительна целиком, и остальные
две клетки читать нельзя. Числа задаются --expect-* и сверяются, а не
подразумеваются.

ЧЕСТНОСТЬ КОНТРОЛЯ. head-only должен быть НЕ СЛАБЕЕ, ЧЕМ МОЖЕТ БЫТЬ: он здесь
опровергает нашу же гипотезу, и поддавки ему в нашу пользу обесценили бы
вывод. Поэтому те же данные, цели, разбиение и потеря, что у Joint-12, но
перебор шага обучения и выбор лучшей эпохи по валидации — на кэше h12 это
минуты. Прежнее сравнение 33.4% против 26.5% было нечестным: там разом
менялись объём данных, цель, разбиение и потеря.

Запуск:
    python3 experiments/k9f_readout_table.py --selftest

    python3 experiments/k9f_readout_table.py --ckpt <base> \\
        --cache data/k9_teacher_150k.npz \\
        --orig data/k9e_orig --trained data/k9e_ep3 \\
        --expect-cell1 0.231 --expect-cell4 0.330 --out data/k9f
"""

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

N_POS, N_LEVEL, T_CHUNK, H_EXEC = 16, 3, 20, 8


def read_rule(head_only, joint, base):
    """Пре-регистрированное чтение. Отдельной функцией ради самопроверки."""
    span = joint - base
    if span <= 0:
        return "прирост не воспроизведён, таблица недействительна"
    frac = (head_only - base) / span
    if frac >= 0.85:
        return "прирост принадлежит ГОЛОВЕ: переноса на слой 12 не было"
    if frac <= 0.35:
        return "прирост принадлежит СТВОЛУ: перенос состоялся"
    return "вклад разделён, однозначного вывода нет"


def selftest():
    assert "ГОЛОВЕ" in read_rule(0.330, 0.330, 0.231)
    assert "СТВОЛУ" in read_rule(0.255, 0.330, 0.231)
    assert "разделён" in read_rule(0.290, 0.330, 0.231)
    assert "недействительна" in read_rule(0.25, 0.20, 0.231)

    # Доля считается от ПРИРОСТА, а не от абсолютного значения: head-only 26%
    # при опоре 23% и совместном 33% — это 29% прироста, а не 79% результата.
    f = (0.26 - 0.231) / (0.330 - 0.231)
    assert 0.25 < f < 0.35, f

    # Взвешенное усреднение метрик по батчам разного размера обязано совпасть
    # с усреднением по всем строкам сразу.
    rng = np.random.default_rng(0)
    x = rng.random(1000)
    parts = [x[:300], x[300:700], x[700:]]
    w = sum(float(p.mean()) * len(p) for p in parts) / len(x)
    assert abs(w - float(x.mean())) < 1e-12

    # RMS складывается по КВАДРАТАМ, а не по значениям: частая ошибка при
    # усреднении поза8 между батчами.
    a, b = rng.random(400), rng.random(600)
    rms = math.sqrt((float((a ** 2).mean()) * 400
                     + float((b ** 2).mean()) * 600) / 1000)
    assert abs(rms - float(np.sqrt((np.concatenate([a, b]) ** 2).mean()))) < 1e-12
    print("самопроверка k9f пройдена (версия «доля прироста»): правило чтения "
          "на четырёх исходах, доля считается от прироста, взвешенные "
          "средние и RMS складываются верно")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", default="data/k9_teacher_150k.npz")
    ap.add_argument("--orig", help="префикс от k9e без --joint-ckpt")
    ap.add_argument("--trained", help="префикс от k9e с --joint-ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lrs", default="3e-4,1e-3,3e-3",
                    help="перебор шага для head-only; контроль обязан быть "
                         "не слабее, чем может быть")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--hard-weight", type=float, default=0.25)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    # ЛОГИТЫ УЧИТЕЛЯ В ПАМЯТЬ. Перебор из трёх шагов по восемь эпох — это 24
    # полных прохода по файлу логитов; при 9.8 ГиБ выходит около 235 ГиБ
    # случайных чтений с диска, заполненного на 95%. Обучающая часть занимает
    # порядка 8 ГиБ и помещается в оперативную память целиком.
    ap.add_argument("--preload-logits", choices=["auto", "on", "off"],
                    default="auto")
    ap.add_argument("--preload-limit-gib", type=float, default=16.0)
    ap.add_argument("--fit-trained", action="store_true",
                    help="дополнительно дообучить голову на ОБУЧЕННОМ стволе")
    ap.add_argument("--expect-cell1", type=float, default=None)
    ap.add_argument("--expect-cell4", type=float, default=None)
    ap.add_argument("--tol-warn", type=float, default=0.01)
    ap.add_argument("--tol-fail", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/k9f")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    for need in ("ckpt", "orig", "trained"):
        if not getattr(args, need):
            raise SystemExit(f"нужен --{need} (или --selftest)")

    sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k9f sha1 {sha}")
    os.makedirs(args.out, exist_ok=True)

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import actioncodec  # noqa: F401
    from joint12_vla import kd_loss
    from utils import VisionLanguageActionProcessor

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)

    # --- кэш учителя ----------------------------------------------------------
    z = np.load(args.cache, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    q_teach = z["teacher_codes_q0"].astype(np.int64)
    K_true0 = z["K_true_q0"].astype(np.int64)
    split = z["split"]
    N, V = len(q_teach), int(meta["vocab"])
    T_LOG = np.load(args.cache + ".logits.npy", mmap_mode="r")
    assert T_LOG.shape == (N, N_POS, V), T_LOG.shape
    itr, iva = np.where(split == "train")[0], np.where(split == "val")[0]
    print(f"кэш: {N} наблюдений, обучение {len(itr)}, валидация {len(iva)}")

    # itr отсортирован по возрастанию (np.where), поэтому сортировка позиций
    # внутри itr совпадает с сортировкой глобальных индексов — и обращение к
    # memmap остаётся монотонным, когда предзагрузки нет.
    tl_gib = len(itr) * N_POS * V * T_LOG.dtype.itemsize / 2 ** 30
    pre = (args.preload_logits == "on" or
           (args.preload_logits == "auto" and tl_gib <= args.preload_limit_gib))
    TL_tr = None
    if pre:
        print(f"логиты учителя обучающей части в память: {tl_gib:.1f} ГиБ "
              f"(иначе {args.epochs * len(args.lrs.split(','))} проходов по "
              f"диску)", flush=True)
        TL_tr = np.asarray(T_LOG[itr])
    else:
        print(f"логиты учителя остаются на диске: {tl_gib:.1f} ГиБ больше "
              f"порога {args.preload_limit_gib:.0f} ГиБ")

    # --- два кэша h12 ---------------------------------------------------------
    def load_side(prefix, want_trunk):
        md = json.load(open(prefix + ".json"))
        if md["trunk"] != want_trunk:
            raise SystemExit(f"{prefix}: ствол «{md['trunk']}», ожидался "
                             f"«{want_trunk}»")
        if md["n"] != N:
            raise SystemExit(f"{prefix}: {md['n']} строк против {N} в кэше")
        if os.path.abspath(md["cache"]) != os.path.abspath(args.cache):
            raise SystemExit(f"{prefix} снят с другого кэша: {md['cache']}")
        h = np.load(md["h12_file"], mmap_mode="r")
        rd = torch.load(md["readout_file"], map_location="cpu",
                        weights_only=False)
        mm = md.get("cache_vs_live_token_mismatch")
        print(f"  {want_trunk}: {md['h12_file']}, шум хранения "
              + (f"{mm:.4%}" if mm is not None else "не мерился")
              + f", голова из {md['readout_file']}")
        return md, h, rd

    print("кэши h12:")
    md_o, H_o, rd_o = load_side(args.orig, "original")
    md_t, H_t, rd_t = load_side(args.trained, "trained")
    if md_o["joint12_vla_sha1"] != md_t["joint12_vla_sha1"]:
        raise SystemExit(
            f"кэши сняты разными версиями joint12_vla.py: "
            f"{md_o['joint12_vla_sha1']} и {md_t['joint12_vla_sha1']}")
    D = md_o["dim"]

    # --- кодек и опорные действия --------------------------------------------
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None))
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь в action_processor")
    codec = codec.to(dev).eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    def dec0(codes):
        """Действие из грубого уровня — ровно то, что исполняет симулятор."""
        out = []
        for i0 in range(0, len(codes), 256):
            k = torch.as_tensor(codes[i0:i0 + 256]).long().to(dev)
            with torch.no_grad():
                x, _ = codec._decode(E[0][k], embodiment_ids=0)
            out.append(x[..., :7].float().cpu())
        return torch.cat(out)

    A_teach = dec0(q_teach)
    A_star = None
    if "K_true" in z.files:
        Kt3 = z["K_true"].astype(np.int64)
        outs = []
        for i0 in range(0, N, 256):
            k = torch.as_tensor(Kt3[i0:i0 + 256]).long().to(dev)
            with torch.no_grad():
                zz = sum(E[j][k[:, j, :]] for j in range(N_LEVEL))
                x, _ = codec._decode(zz, embodiment_ids=0)
            outs.append(x[..., :7].float().cpu())
        A_star = torch.cat(outs)

    # --- голова как отдельный модуль -----------------------------------------
    class Readout(nn.Module):
        """Норма плюс линейная голова. ВСЁ В FP32.

        Живой путь шёл под autocast fp16, здесь считается точнее. Разница
        обязана быть мала, и это не предположение: клетки 1 и 4 сверяются с
        известными числами, и расхождение больше --tol-fail останавливает
        разбор.
        """

        def __init__(self, norm, head):
            super().__init__()
            self.norm, self.head = norm, head

        def forward(self, h):
            return self.head(self.norm(h))

    def fresh(rd):
        import copy
        return Readout(copy.deepcopy(rd["norm"]),
                       copy.deepcopy(rd["head"])).to(dev).float()

    # --- оценка ---------------------------------------------------------------
    @torch.no_grad()
    def evaluate(Hc, ro, idxs, tag):
        ro.eval()
        acc_t = acc_k = 0.0
        se_i = sg_i = se_e = fl4 = fl8 = 0.0
        wsum = 0
        for i0 in range(0, len(idxs), args.batch):
            sel = idxs[i0:i0 + args.batch]
            h = torch.from_numpy(np.asarray(Hc[sel])).to(dev).float()
            pc = ro(h).argmax(-1).cpu().numpy()
            w = len(sel)
            acc_t += float((pc == q_teach[sel]).mean()) * w
            acc_k += float((pc == K_true0[sel]).mean()) * w
            a = dec0(pc)
            d_i = a - A_teach[sel]
            se_i += float((d_i[:, :H_EXEC, :6] ** 2).mean()) * w
            sg_i += float((d_i[:, :H_EXEC, 6] ** 2).mean()) * w
            fl4 += float((torch.sign(a[:, :4, 6])
                          != torch.sign(A_teach[sel][:, :4, 6])).float().mean()) * w
            fl8 += float((torch.sign(a[:, :H_EXEC, 6])
                          != torch.sign(A_teach[sel][:, :H_EXEC, 6])).float().mean()) * w
            if A_star is not None:
                se_e += float(((a - A_star[sel])[:, :H_EXEC, :6] ** 2).mean()) * w
            wsum += w
        r = dict(acc_teacher=acc_t / wsum, acc_ktrue=acc_k / wsum,
                 imit_pose8=math.sqrt(se_i / wsum),
                 imit_grip8=math.sqrt(sg_i / wsum),
                 grip_flip4=fl4 / wsum, grip_flip8=fl8 / wsum,
                 expert_pose8=(math.sqrt(se_e / wsum)
                               if A_star is not None else None), n=wsum)
        print(f"  [{tag}] согласие {r['acc_teacher']:.1%} "
              f"(с токенизатором {r['acc_ktrue']:.1%}); поза8 "
              f"{r['imit_pose8']:.4f}, знак8 {r['grip_flip8']:.2%}"
              + (f"; до эксперта {r['expert_pose8']:.4f}"
                 if r['expert_pose8'] is not None else ""))
        return r

    # --- обучение головы ------------------------------------------------------
    def fit(Hc, rd, lr, tag):
        """Одна голова, один шаг обучения. Возвращает лучшее по валидации."""
        ro = fresh(rd)
        opt = torch.optim.AdamW(ro.parameters(), lr=lr,
                                weight_decay=args.weight_decay)
        rng = np.random.default_rng(args.seed)
        best, best_state, hist = None, None, []
        for ep in range(1, args.epochs + 1):
            ro.train()
            # ПЕРЕМЕШИВАЮТСЯ ПОЗИЦИИ ВНУТРИ itr, а не глобальные индексы:
            # тогда одна и та же выборка адресует и предзагруженный массив, и
            # memmap, и порядок обращения остаётся монотонным.
            order = rng.permutation(len(itr))
            tot, nb = 0.0, 0
            for i0 in range(0, len(order), args.batch):
                pos = np.sort(order[i0:i0 + args.batch])
                sel = itr[pos]
                h = torch.from_numpy(np.asarray(Hc[sel])).to(dev).float()
                tl = torch.from_numpy(
                    TL_tr[pos] if TL_tr is not None
                    else np.asarray(T_LOG[sel])).to(dev).float()
                y = torch.from_numpy(q_teach[sel]).to(dev)
                lg = ro(h)
                # ТА ЖЕ ПОТЕРЯ, ЧТО У JOINT-12: KD при T=2 плюс 0.25 жёсткой.
                loss = kd_loss(lg, tl, args.temperature) + args.hard_weight * \
                    F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss); nb += 1
            ev = evaluate(Hc, ro, iva, f"{tag} lr={lr:g} эпоха {ep}")
            hist.append(dict(epoch=ep, loss=tot / max(nb, 1), **{
                k: v for k, v in ev.items() if k != "n"}))
            if best is None or ev["acc_teacher"] > best["acc_teacher"]:
                best = ev
                best_state = {k: v.detach().clone()
                              for k, v in ro.state_dict().items()}
        ro.load_state_dict(best_state)
        return ro, best, hist

    results, all_hist = {}, {}

    # Клетка 1 и клетка 4 — известные ответы, считаются ПЕРВЫМИ. Если они не
    # сходятся, дальше идти незачем.
    print("\n=== клетки с известным ответом ===")
    c1 = evaluate(H_o, fresh(rd_o), iva, "1: ствол исходный, голова исходная")
    c4 = evaluate(H_t, fresh(rd_t), iva, "4: ствол обученный, голова обученная")
    results["cell1_orig_orig"], results["cell4_trained_trained"] = c1, c4

    bad = []
    for nm, got, want in (("клетка 1", c1["acc_teacher"], args.expect_cell1),
                          ("клетка 4", c4["acc_teacher"], args.expect_cell4)):
        if want is None:
            print(f"  {nm}: ожидание не задано, сверка не выполнена")
            continue
        d = abs(got - want)
        print(f"  {nm}: получено {got:.1%}, ожидалось {want:.1%}, "
              f"расхождение {d * 100:.2f} пп")
        if d > args.tol_fail:
            bad.append(f"{nm}: {d * 100:.2f} пп")
        elif d > args.tol_warn:
            print(f"    ВНИМАНИЕ: больше {args.tol_warn * 100:.0f} пп; "
                  f"вероятно, разница fp32 против autocast fp16")
    if bad:
        raise SystemExit(
            "клетки с известным ответом не воспроизведены (" + "; ".join(bad)
            + f") при пороге {args.tol_fail * 100:.0f} пп.\nЭто отказ "
            f"конвейера, а не результат: остальные две клетки читать нельзя.")

    # Клетка 3 — бесплатная, без обучения.
    print("\n=== клетка 3: обученный ствол, ИСХОДНАЯ голова ===")
    print("  напоминание: высокое значение доказательно, низкое — нет "
          "(со-адаптация)")
    c3 = evaluate(H_t, fresh(rd_o), iva, "3: ствол обученный, голова исходная")
    results["cell3_trained_orig"] = c3

    # Клетка 2 — head-only, единственная требующая обучения.
    print("\n=== клетка 2: head-only на ИСХОДНОМ стволе ===")
    best2, best2_lr = None, None
    for lr in [float(x) for x in args.lrs.split(",")]:
        ro, ev, hist = fit(H_o, rd_o, lr, "head-only")
        all_hist[f"head_only_lr{lr:g}"] = hist
        print(f"  lr={lr:g}: лучшее согласие {ev['acc_teacher']:.1%}")
        if best2 is None or ev["acc_teacher"] > best2["acc_teacher"]:
            best2, best2_lr = ev, lr
            torch.save(ro.state_dict(),
                       os.path.join(args.out, "head_only.pt"))
    results["cell2_orig_headonly"] = dict(lr=best2_lr, **best2)

    if args.fit_trained:
        print("\n=== дополнительно: голова заново на ОБУЧЕННОМ стволе ===")
        bt, bt_lr = None, None
        for lr in [float(x) for x in args.lrs.split(",")]:
            ro, ev, hist = fit(H_t, rd_o, lr, "refit")
            all_hist[f"refit_lr{lr:g}"] = hist
            if bt is None or ev["acc_teacher"] > bt["acc_teacher"]:
                bt, bt_lr = ev, lr
        results["extra_trained_refit"] = dict(lr=bt_lr, **bt)

    # --- таблица --------------------------------------------------------------
    base, joint, ho = c1["acc_teacher"], c4["acc_teacher"], best2["acc_teacher"]
    print(f"\n{'=' * 70}")
    print(f"  {'ствол':<11}{'голова':<11}{'согласие':>10}{'поза8':>9}"
          f"{'знак8':>8}")
    for lab_s, lab_h, r in (("исходный", "исходная", c1),
                            ("исходный", "head-only", best2),
                            ("обученный", "исходная", c3),
                            ("обученный", "обученная", c4)):
        print(f"  {lab_s:<11}{lab_h:<11}{r['acc_teacher']:>9.1%}"
              f"{r['imit_pose8']:>9.4f}{r['grip_flip8']:>8.2%}")
    span = joint - base
    frac = (ho - base) / span if span > 0 else float("nan")
    print(f"\n  прирост всего {span * 100:+.1f} пп; головой одной "
          f"{(ho - base) * 100:+.1f} пп = {frac:.0%} прироста")
    print(f"  ВЫВОД: {read_rule(ho, joint, base)}")
    print("  Клетка «обученный ствол + исходная голова» "
          f"{c3['acc_teacher']:.1%}: доказательна только если высока.")

    md = dict(script_sha1=sha, cells=results, history=all_hist,
              base=base, joint=joint, head_only=ho, head_only_lr=best2_lr,
              gain_fraction_head=frac, verdict=read_rule(ho, joint, base),
              orig=md_o, trained=md_t, argv=vars(args))
    p = os.path.join(args.out, "table.json")
    json.dump(md, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"\n  сохранено: {p}")


if __name__ == "__main__":
    main()
