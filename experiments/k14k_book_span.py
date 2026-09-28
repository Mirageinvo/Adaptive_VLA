#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14k: чего не хватает книге первого уровня — размаха, направлений или выбора.

ВОПРОС. Четыре попытки улучшить ЧИТАТЕЛЯ (§45.1, §46.1, §47, §48.4) упёрлись
в захват 0.05...0.09 при пороге 0.20. Все четыре были классификаторами над
логитами. Прежде чем пробовать пятого, надо узнать, есть ли вообще в книге
`E1` то, чем нужную поправку можно выразить: RVQ обучала её на остатках при
ПРАВИЛЬНОМ k0, а при ошибочном q0 требуемая поправка — величина порядка кода
нулевого уровня, а не мелкий остаток.

ТРИ РАЗНЫХ ДИАГНОЗА, КОТОРЫЕ НЕЛЬЗЯ СМЕШИВАТЬ. Для остатка
r = z_e − D0[q0hat] считаются две величины:

    rho_fixed = min_k  ||r − D1[k]|| / ||r||           книга как она есть
    rho_ray   = min_k,a>=0 ||r − a·D1[k]|| / ||r||     направление со свободным
                                                       масштабом

и выбранный масштаб a* = max(0, <r, D1[k]> / ||D1[k]||^2). Тогда:

    rho_fixed велика, rho_ray мала, a* > 1  -> направление есть, НЕТ МАСШТАБА
    обе велики                              -> НЕТ НАПРАВЛЕНИЙ (покрытие)
    обе малы                                -> книга достаточна, дело в ВЫБОРЕ

Одна rho_fixed этих случаев не различает: код может иметь верную длину при
неверном направлении, и наоборот. Решение «нужна книга большего масштаба»
принимается ТОЛЬКО по второму признаку, иначе новая книга лечит не тот
диагноз.

В КАКОМ ПРОСТРАНСТВЕ СЧИТАТЬ. Квантователь ActionCodec выбирает код по
расстоянию ПОСЛЕ `in_project`, а вклад уровня попадает в латент ПОСЛЕ
`out_project`. Это разные пространства, и они не обязаны быть изометричны.
Ошибка, которая важна декодеру, живёт в ЛАТЕНТЕ, поэтому геометрия считается
там, по эффективному словарю D1[k] = out_project(codebook_1[k]).

ПОБОЧНО ЭТО ИЗМЕРЯЕТ ЗНАМЕНАТЕЛЬ C. Оракульный q1* в K-14a выбран
`nearest_code`, то есть argmin'ом в ПРОЕЦИРОВАННОМ пространстве. Лучший код в
латенте может быть другим, и тогда A01 = 0.0869 — потолок жадной процедуры
кодека, а не потолок «лучшего кода». Доля расхождения двух argmin'ов
считается здесь же.

РАЗМЕР СЛОВАРЯ НЕ ЗАШИТ. V и d читаются из формы книги; ожидаемые 2048x512
проверяются fail-closed. Предыдущая версия этого отчёта называла 512 «числом
кодов», спутав размерность эмбеддинга с размером словаря.

ЧЕГО ЗДЕСЬ НЕТ. Декодирования: эта часть отвечает на вопрос о ЛАТЕНТНОЙ
геометрии и ничего не говорит про ошибку действия. Контроль масштабированных
книг и beam-оракул требуют декодирования и идут отдельным этапом.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

EXPECT_VOCAB = 2048
EXPECT_DIM = 512
RHO_HI = 0.5        # порог «книга не покрывает больше половины»
P_LIMIT = 0.25      # порог существенности ограничения представления
ALPHA_HI = 1.0      # масштаб выше единицы = книге не хватает длины


def sha12(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:12]


def geometry(r, D, torch, chunk=4096):
    """Две ошибки аппроксимации и выбранный масштаб, по эффективному словарю.

    `r`: (M, d) остатки, `D`: (V, d) словарь В ЛАТЕНТЕ. Возвращает словарь
    массивов длины M. Считается через ||r−d||^2 = ||r||^2 − 2<r,d> + ||d||^2,
    то есть одним матричным произведением на кусок: тензор (M, V) целиком
    при V=2048 и M в сотнях тысяч не нужен и не строится.
    """
    if r.ndim != 2 or D.ndim != 2 or r.shape[1] != D.shape[1]:
        raise ValueError(f"формы {tuple(r.shape)} и {tuple(D.shape)}")
    d2 = (D * D).sum(1)                       # (V,)
    safe_d2 = torch.clamp(d2, min=1e-24)
    out = {k: [] for k in ("rho_fixed", "rho_ray", "alpha", "k_fixed",
                           "k_ray", "r_norm", "d_norm_fixed", "cos_ray")}
    for i in range(0, r.shape[0], chunk):
        x = r[i:i + chunk]
        x2 = (x * x).sum(1)                   # (m,)
        ip = x @ D.T                          # (m, V)
        # --- КНИГА КАК ОНА ЕСТЬ -----------------------------------------
        dist2 = x2[:, None] - 2.0 * ip + d2[None, :]
        kf = dist2.argmin(1)
        best = torch.clamp(dist2.gather(1, kf[:, None]).squeeze(1), min=0.0)
        rn = torch.sqrt(torch.clamp(x2, min=0.0))
        safe_rn = torch.clamp(rn, min=1e-24)
        out["rho_fixed"].append(torch.sqrt(best) / safe_rn)
        out["k_fixed"].append(kf)
        out["r_norm"].append(rn)
        out["d_norm_fixed"].append(torch.sqrt(safe_d2)[kf])
        # --- СВОБОДНЫЙ МАСШТАБ ВДОЛЬ ЛУЧА -------------------------------
        # При a >= 0 минимум ||r − a·d|| достигается на a* = <r,d>/||d||^2,
        # и остаточная норма равна ||r||·sqrt(1 − cos^2) при <r,d> > 0,
        # либо ||r|| при <r,d> <= 0: луч смотрит в другую сторону.
        cos = ip / (safe_rn[:, None] * torch.sqrt(safe_d2)[None, :])
        cos_pos = torch.clamp(cos, min=0.0)
        kr = cos_pos.argmax(1)
        c = cos_pos.gather(1, kr[:, None]).squeeze(1)
        out["cos_ray"].append(c)
        out["rho_ray"].append(torch.sqrt(torch.clamp(1.0 - c * c, min=0.0)))
        out["k_ray"].append(kr)
        a_star = ip.gather(1, kr[:, None]).squeeze(1) / safe_d2[kr]
        out["alpha"].append(torch.clamp(a_star, min=0.0))
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


def boot_prop(flags, eps, groups, n=10000, seed=11, qs=(5, 95)):
    """Кластерный бутстрап по эпизодам для ДОЛЕЙ и их разности.

    `flags` — булев признак по позициям, `eps` — эпизод позиции, `groups` —
    словарь {имя: булева маска подмножества}. В каждой реплике
    пересэмплируются ОДНИ И ТЕ ЖЕ эпизоды для всех подмножеств, и разность
    считается ВНУТРИ реплики: два отдельных интервала не отвечают на вопрос,
    больше ли одна доля другой, потому что их ошибки коррелированы.

    Пустое подмножество в реплике даёт nan и выбрасывается перцентилем: доля
    по нулю позиций не определена, и подставлять ноль значило бы утверждать,
    что признака там нет.
    """
    eps = np.asarray(eps)
    uniq, inv = np.unique(eps, return_inverse=True)
    f = np.asarray(flags, bool)
    num, den = {}, {}
    for g, m in groups.items():
        m = np.asarray(m, bool)
        num[g] = np.bincount(inv, weights=(f & m).astype(float),
                             minlength=len(uniq))
        den[g] = np.bincount(inv, weights=m.astype(float),
                             minlength=len(uniq))
    rg = np.random.default_rng(seed)
    pick = rg.integers(0, len(uniq), size=(n, len(uniq)))
    p = {}
    for g in groups:
        nn_, dd_ = num[g][pick].sum(1), den[g][pick].sum(1)
        p[g] = np.where(dd_ > 0, nn_ / np.where(dd_ > 0, dd_, 1), np.nan)
    res = {g: [float(np.nanpercentile(p[g], q)) for q in qs] for g in p}
    # ОБЕ СТОРОНЫ РАЗНОСТИ. Один ключ на пару означал бы, что читатель обязан
    # помнить порядок сортировки имён, а перепутанный знак интервала — это
    # перевёрнутый вывод. Пусть лучше будет лишний ключ.
    gs = sorted(groups)
    for i in range(len(gs)):
        for j in range(len(gs)):
            if i == j:
                continue
            res[f"d:{gs[i]}-{gs[j]}"] = [
                float(np.nanpercentile(p[gs[i]] - p[gs[j]], q)) for q in qs]
    res["_point"] = {g: float(np.nansum(num[g]) / max(np.nansum(den[g]), 1))
                     for g in groups}
    return res


def selftest():
    import torch
    torch.manual_seed(0)

    # --- ГЕОМЕТРИЯ НА СЛУЧАЯХ С ИЗВЕСТНЫМ ОТВЕТОМ ------------------------
    # Словарь из ОРТОГОНАЛЬНЫХ векторов разной длины: тогда обе величины
    # считаются на бумаге, и тест проверяет формулу, а не сам себя.
    d = 6
    D = torch.zeros(3, d)
    D[0, 0] = 1.0
    D[1, 1] = 2.0
    D[2, 2] = 0.5
    r = torch.stack([
        D[0].clone(),                    # ровно элемент книги
        3.0 * D[1].clone(),              # верное направление, масштаб 3
        torch.zeros(d).index_fill_(0, torch.tensor([3]), 1.0),  # ортогонален
        -2.0 * D[2].clone(),             # противоположное направление
    ])
    g = geometry(r, D, torch)
    assert abs(g["rho_fixed"][0]) < 1e-6 and abs(g["rho_ray"][0]) < 1e-6
    assert abs(g["alpha"][0] - 1.0) < 1e-6, g["alpha"][0]
    # верное направление, не хватает длины: rho_ray = 0, rho_fixed = 2/3
    assert abs(g["rho_ray"][1]) < 1e-6, g["rho_ray"][1]
    assert abs(g["rho_fixed"][1] - 2.0 / 3.0) < 1e-6, g["rho_fixed"][1]
    assert abs(g["alpha"][1] - 3.0) < 1e-6, g["alpha"][1]
    assert g["k_ray"][1] == 1
    # ортогонален всему: свободный масштаб не помогает
    assert abs(g["rho_ray"][2] - 1.0) < 1e-6, g["rho_ray"][2]
    assert g["rho_fixed"][2] >= 1.0 - 1e-6
    assert abs(g["alpha"][2]) < 1e-12, "масштаб обязан быть неотрицательным"
    # противоположное направление: луч с a>=0 не достаёт, rho_ray = 1
    assert abs(g["rho_ray"][3] - 1.0) < 1e-6, g["rho_ray"][3]
    assert abs(g["cos_ray"][3]) < 1e-6
    # куски не влияют на результат
    g2 = geometry(r, D, torch, chunk=1)
    for k in g:
        assert np.allclose(g[k], g2[k]), f"разбиение на куски меняет {k}"
    # rho_ray НЕ БОЛЬШЕ rho_fixed никогда: свободный масштаб включает a=1
    rr = torch.randn(200, d)
    gr = geometry(rr, torch.randn(17, d), torch)
    assert (gr["rho_ray"] <= gr["rho_fixed"] + 1e-5).all(), \
        "свободный масштаб оказался хуже фиксированного"
    assert (gr["alpha"] >= 0).all()

    # --- БУТСТРАП ДОЛЕЙ ---------------------------------------------------
    rg = np.random.default_rng(1)
    NE, PER = 40, 25
    eps = np.repeat(np.arange(NE), PER)
    wrong = np.zeros(NE * PER, bool)
    wrong[: NE * PER // 2] = True
    # одинаковые доли -> разность содержит ноль
    fl = rg.random(NE * PER) < 0.3
    b = boot_prop(fl, eps, {"wrong": wrong, "ok": ~wrong}, n=500, seed=2)
    assert b["d:ok-wrong"][0] <= 0 <= b["d:ok-wrong"][1], b["d:ok-wrong"]
    # доля явно выше у wrong -> интервал разности не содержит ноль
    fl2 = np.where(wrong, rg.random(NE * PER) < 0.9,
                   rg.random(NE * PER) < 0.1)
    b2 = boot_prop(fl2, eps, {"wrong": wrong, "ok": ~wrong}, n=500, seed=2)
    assert b2["d:wrong-ok"][0] > 0, b2["d:wrong-ok"]
    assert b2["_point"]["wrong"] > b2["_point"]["ok"]
    # пустое подмножество не превращается в нулевую долю
    # ПУСТОЕ ПОДМНОЖЕСТВО ДАЁТ nan, А НЕ НОЛЬ. Доля по нулю позиций не
    # определена; ноль означал бы «признака там нет», то есть утверждение о
    # данных, которых не было.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        b3 = boot_prop(fl2, eps, {"none": np.zeros(NE * PER, bool),
                                  "all": np.ones(NE * PER, bool)},
                       n=200, seed=3)
    assert np.isnan(b3["none"][0]) and np.isnan(b3["none"][1]), b3["none"]
    assert 0.0 <= b3["_point"]["all"] <= 1.0
    assert abs(b3["_point"]["all"] - fl2.mean()) < 1e-12
    # --- МАСКИ ПОЗИЦИЙ: ДВЕ ФОРМЫ ПРИЗНАКА ОПРЕДЕЛЁННОСТИ ----------------
    # Именно здесь прогон упал: построчный признак (n,) подставлялся туда,
    # где нужна позиционная маска (n*P,). Тест проверяет обе формы, порядок
    # row-major и отказ на чужих формах.
    nR, nP = 4, 3
    q0t = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 1]], np.int64)
    k0t = q0t.copy()
    k0t[1, 2] = 99                       # ровно одна ошибка: строка 1, поз 2
    dr = np.array([True, True, False, True])
    md, mw, mo = position_masks(q0t, k0t, dr, nP)
    assert md.shape == (nR * nP,) and mw.shape == md.shape
    assert md.tolist() == [True] * 6 + [False] * 3 + [True] * 3, md.tolist()
    assert mw.sum() == 1 and mw[1 * nP + 2], np.where(mw)[0]
    assert (mw | mo == md).all() and not (mw & mo).any()
    # позиционная форма даёт то же при согласованном входе
    md2, mw2, mo2 = position_masks(q0t, k0t,
                                   np.repeat(dr, nP).reshape(nR, nP), nP)
    assert (md2 == md).all() and (mw2 == mw).all() and (mo2 == mo).all()
    # ошибка в непосчитанной строке в mw не попадает
    k0t3 = q0t.copy(); k0t3[2, 0] = 55
    _, mw3, _ = position_masks(q0t, k0t3, dr, nP)
    assert not mw3[2 * nP + 0], "ошибка в непосчитанной строке учтена"
    for bad, why in (((q0t[:, :2], k0t, dr), "q0 формы"),
                     ((q0t, k0t[:, :2], dr), "k0 формы"),
                     ((q0t, k0t, np.zeros((nR, nP + 1), bool)),
                      "определённости формы"),
                     ((q0t, k0t, np.zeros(nR + 1, bool)),
                      "определённости формы")):
        try:
            position_masks(*bad, nP)
        except SystemExit as e:
            assert why in str(e), (why, e)
        else:
            raise AssertionError(f"принята чужая форма: {why}")

    print("самопроверка k14k_book_span пройдена")


def load_codec(ckpt, dev, torch, VisionLanguageActionProcessor):
    """Кодек и три квантователя — тем же путём, что K-14a."""
    proc = VisionLanguageActionProcessor.from_pretrained(
        ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None))
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь")
    codec = codec.to(dev).eval()
    qs = list(codec.vq.quantizers)
    if len(qs) != 3:
        raise SystemExit(f"уровней {len(qs)}, ожидалось 3")
    return codec, qs


def position_masks(q0_rows, k0_rows, defined_rows, n_pos):
    """Позиционные маски из построчных и построчно-позиционных величин.

    ВЫНЕСЕНО ПОСЛЕ ПАДЕНИЯ. `load_canonical_q0` отдаёт `defined` как
    ПОСТРОЧНЫЙ признак `q0.min(axis=1) >= 0` формы (n,), а q0 и k0 —
    построчно-позиционные, формы (n, P). Смешение этих двух форм и было
    ошибкой: маски длиной n и n*P молча не сходятся только потому, что numpy
    отказался их транслировать. Функция чистая, чтобы это проверялось тестом,
    а не прогоном на кластере.

    Возвращает три плоские маски длины n*P в порядке row-major — том же, в
    каком разворачивается латент (n, P, d) -> (n*P, d).
    """
    q0_rows = np.asarray(q0_rows, np.int64)
    k0_rows = np.asarray(k0_rows, np.int64)
    n = q0_rows.shape[0]
    if q0_rows.shape != (n, n_pos):
        raise SystemExit(f"q0 формы {q0_rows.shape}, ожидалась {(n, n_pos)}")
    if k0_rows.shape != (n, n_pos):
        raise SystemExit(f"k0 формы {k0_rows.shape}, ожидалась {(n, n_pos)}")
    d = np.asarray(defined_rows, bool)
    if d.shape == (n,):
        # ПОСТРОЧНЫЙ ПРИЗНАК РАЗМНОЖАЕТСЯ НА ПОЗИЦИИ, а не транслируется
        # молча: строка либо посчитана планом целиком, либо не посчитана.
        d = np.repeat(d, n_pos)
    elif d.shape == (n, n_pos):
        d = d.reshape(-1)
    else:
        raise SystemExit(f"признак определённости формы {d.shape}: ожидалась "
                         f"{(n,)} или {(n, n_pos)}")
    wrong = (q0_rows != k0_rows).reshape(-1)
    return d, wrong & d, (~wrong) & d


def measure_part(name, rows, ACT, epi, ktrue, q0, q0_def, books, codec, qs,
                 dev, torch, nearest_code, boot=10000, enc_batch=256):
    """Геометрия остатков одной части. Без декодирования и без обучения."""
    n = len(rows)
    act = torch.from_numpy(np.asarray(ACT[rows], np.float32))
    ze = []
    with torch.no_grad():
        for i in range(0, n, enc_batch):
            ze.append(codec._encode(act[i:i + enc_batch].to(dev),
                                    embodiment_ids=0).float())
    ze = torch.cat(ze)                                   # (n, P, d)
    P = int(ze.shape[1])
    q0r = np.asarray(q0[rows], np.int64)
    k0 = np.asarray(ktrue[rows])[:, 0, :].astype(np.int64)
    defined = (np.ones(n, bool) if q0_def is None
               else np.asarray(q0_def)[rows])
    m_def, m_wrong, m_ok = position_masks(q0r, k0, defined, P)
    # НЕПОСЧИТАННЫЕ СТРОКИ ОТВЕРГАЮТСЯ, А НЕ МАСКИРУЮТСЯ. В q0 они помечены
    # -1, а индексация книги значением -1 молча берёт ПОСЛЕДНИЙ элемент и
    # даёт правдоподобный мусор. Каноническая часть обязана быть покрыта
    # планом целиком, поэтому это отказ, а не фильтрация.
    if not m_def.all():
        raise SystemExit(
            f"{name}: {int((~m_def).sum())} позиций из {len(m_def)} не "
            f"посчитаны планом (q0 = -1). Часть покрыта не целиком")
    if int(q0r.min()) < 0 or int(k0.min()) < 0:
        raise SystemExit(f"{name}: отрицательный код в q0 или k0")
    # ОСТАТОК СЧИТАЕТСЯ В ЛАТЕНТЕ: вклад уровня берётся через out_project,
    # а не прямым lookup во внутренней книге. Индекс обязан быть long:
    # int32-тензор в индексации torch не принимается.
    D0 = books[0][torch.from_numpy(q0r.reshape(-1)).long().to(dev)]
    r = (ze.reshape(-1, ze.shape[-1]) - D0)
    if r.shape[0] != n * P:
        raise SystemExit(f"{name}: остатков {r.shape[0]}, позиций {n * P}")
    g = geometry(r, books[1], torch)
    if len(g["rho_fixed"]) != n * P:
        raise SystemExit(f"{name}: геометрия дала {len(g['rho_fixed'])} "
                         f"значений на {n * P} позиций")

    # --- ARGMIN В ЛАТЕНТЕ ПРОТИВ ВЫБОРА КОДЕКА ---------------------------
    # Кодек выбирает код по расстоянию ПОСЛЕ in_project, а ошибка, которая
    # важна декодеру, живёт в латенте. Если выборы расходятся, оракульный q1*
    # из K-14a — потолок ЖАДНОЙ ПРОЦЕДУРЫ КОДЕКА, а не лучшего кода, и это
    # касается знаменателя C.
    with torch.no_grad():
        k_codec = nearest_code(r.reshape(n, P, -1), qs[1]).reshape(-1)
    k_codec = k_codec.cpu().numpy()
    agree = float((k_codec == g["k_fixed"]).mean())

    hi = g["rho_fixed"] > RHO_HI
    eps_pos = np.repeat(np.asarray(epi[rows], np.int64), P)
    bp = boot_prop(hi, eps_pos, {"wrong": m_wrong, "ok": m_ok,
                                 "all": m_def}, n=int(boot))

    def q(x, m):
        v = np.asarray(x)[m]
        if not len(v):
            return None
        return {f"p{p_}": float(np.percentile(v, p_))
                for p_ in (5, 25, 50, 75, 95)}

    # --- ВЕРДИКТЫ ПО ЗАРЕГИСТРИРОВАННЫМ ПРАВИЛАМ -------------------------
    lo_wrong = bp["wrong"][0]
    lo_diff = bp["d:wrong-ok"][0]
    limited = bool(lo_wrong > P_LIMIT and lo_diff > 0)
    # МАСШТАБНОЕ ограничение — отдельное утверждение: свободный масштаб
    # заметно закрывает ошибку И выбранный коэффициент систематически больше
    # единицы. Без обоих признаков увеличенная книга лечит не тот диагноз.
    med_fixed = float(np.median(g["rho_fixed"][m_wrong])) if m_wrong.any() \
        else float("nan")
    med_ray = float(np.median(g["rho_ray"][m_wrong])) if m_wrong.any() \
        else float("nan")
    med_alpha = float(np.median(g["alpha"][m_wrong])) if m_wrong.any() \
        else float("nan")
    scale_limited = bool(med_fixed > RHO_HI and med_ray < RHO_HI / 2
                         and med_alpha > ALPHA_HI)
    out = dict(
        rows=int(n), positions=int(P), defined=int(m_def.sum()),
        frac_wrong_q0=float(m_wrong.sum() / max(m_def.sum(), 1)),
        n_episodes=int(len(np.unique(epi[rows]))),
        latent_vs_codec_argmin_agree=agree,
        rho_fixed=dict(all=q(g["rho_fixed"], m_def),
                       wrong_q0=q(g["rho_fixed"], m_wrong),
                       ok_q0=q(g["rho_fixed"], m_ok)),
        rho_ray=dict(all=q(g["rho_ray"], m_def),
                     wrong_q0=q(g["rho_ray"], m_wrong),
                     ok_q0=q(g["rho_ray"], m_ok)),
        alpha=dict(all=q(g["alpha"], m_def), wrong_q0=q(g["alpha"], m_wrong),
                   ok_q0=q(g["alpha"], m_ok)),
        r_norm=dict(all=q(g["r_norm"], m_def),
                    wrong_q0=q(g["r_norm"], m_wrong),
                    ok_q0=q(g["r_norm"], m_ok)),
        d_norm_at_best=q(g["d_norm_fixed"], m_def),
        boot=dict({k: v for k, v in bp.items() if k != "_point"},
                  point=bp["_point"]),
        representation_limited=limited, scale_limited=scale_limited,
        medians_on_wrong_q0=dict(rho_fixed=med_fixed, rho_ray=med_ray,
                                 alpha=med_alpha))
    print(f"\n  === {name}: {n} строк x {P} позиций, "
          f"{100 * out['frac_wrong_q0']:.1f}% позиций с ошибочным q0 ===")
    print(f"    argmin в латенте совпал с выбором кодека: "
          f"{100 * agree:.2f}% позиций")
    for tag in ("all", "wrong_q0", "ok_q0"):
        rf, rr, al = (out["rho_fixed"][tag], out["rho_ray"][tag],
                      out["alpha"][tag])
        if rf is None:
            continue
        print(f"    {tag:9s} rho_fixed мед {rf['p50']:.3f} "
              f"[{rf['p5']:.3f}, {rf['p95']:.3f}]   "
              f"rho_ray мед {rr['p50']:.3f}   alpha мед {al['p50']:.3f}")
    print(f"    P(rho_fixed>{RHO_HI}): ошибочный q0 "
          f"{bp['_point']['wrong']:.3f} [{bp['wrong'][0]:.3f}, "
          f"{bp['wrong'][1]:.3f}], верный q0 {bp['_point']['ok']:.3f} "
          f"[{bp['ok'][0]:.3f}, {bp['ok'][1]:.3f}]")
    print(f"    разность (ошибочный − верный): "
          f"[{bp['d:wrong-ok'][0]:+.3f}, {bp['d:wrong-ok'][1]:+.3f}]")
    # КАЖДОЕ УСЛОВИЕ ПЕЧАТАЕТСЯ СО СВОИМ ИСХОДОМ. Строка вида «вердикт False
    # (условие A и условие B)» читается как утверждение обоих, и по ней не
    # видно, какое именно не выполнилось.
    def mark(ok):
        return "выполнено" if ok else "НЕ выполнено"
    print(f"    ОГРАНИЧЕНИЕ ПРЕДСТАВЛЕНИЯ существенно: {limited}")
    print(f"      нижняя граница P(rho_fixed>{RHO_HI}|q0 неверен) "
          f"{lo_wrong:.3f} > {P_LIMIT}: {mark(lo_wrong > P_LIMIT)}")
    print(f"      нижняя граница разности {lo_diff:+.3f} > 0: "
          f"{mark(lo_diff > 0)}")
    print(f"    ограничение именно МАСШТАБА: {scale_limited}")
    print(f"      мед rho_fixed {med_fixed:.3f} > {RHO_HI}: "
          f"{mark(med_fixed > RHO_HI)};  мед rho_ray {med_ray:.3f} < "
          f"{RHO_HI / 2}: {mark(med_ray < RHO_HI / 2)};  мед alpha "
          f"{med_alpha:.3f} > {ALPHA_HI}: {mark(med_alpha > ALPHA_HI)}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Размах и покрытие книги E1 в латентном пространстве")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--q0", default="data/k14d/q0_b8_e0.npz")
    ap.add_argument("--gate-r", default="reports/k14d/gate_r.json")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--root", default=".")
    ap.add_argument("--parts", nargs="*", default=["val_sel"],
                    help="части для замера. val_confirm ЗДЕСЬ НЕ НУЖНА: "
                         "вопрос о геометрии книги решается на отборочной "
                         "половине, и тратить подтверждающую на него нельзя")
    ap.add_argument("--expect-rows-sha1", default="",
                    help="ожидаемый отпечаток строк части (из K-14a). "
                         "Замер на ДРУГИХ строках сравнивать с оракулом "
                         "нельзя, а молча он этого не заметит")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out", default="reports/k14k/book_span.json")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(a.root)
    # `experiments` РЕПОЗИТОРИЯ ДОБАВЛЯЕТСЯ ЯВНО. Тогда сам файл может лежать
    # ВНЕ дерева — а это единственный способ запустить замер, пока идёт
    # длинный прогон: новый .py внутри репозитория сделал бы дерево грязным,
    # и следующий прогон цепочки отказался бы стартовать.
    # КОРЕНЬ ВЕНДОРЕННОГО ПАКЕТА, А НЕ ЕГО `scripts`. Пакеты `utils`,
    # `smolvla` и `actioncodec` лежат в third_party/actioncodec, и на
    # кластере он попадает на путь как символическая ссылка `src`, которой в
    # чистом клоне нет. Добавлять `scripts` НЕЛЬЗЯ: тамошний модуль utils.py
    # заслоняет одноимённый ПАКЕТ и ломает импорт изнутри самого пакета.
    for p in (root, os.path.join(root, "src"),
              os.path.join(root, "experiments"),
              os.path.join(root, "third_party", "actioncodec"),
              here):
        if p not in sys.path:
            sys.path.insert(0, p)
    if a.selftest:
        selftest()
        return 0
    if a.limit and a.expect_rows_sha1:
        raise SystemExit(
            "--limit вместе с --expect-rows-sha1 бессмысленны: отпечаток "
            "относится к ПОЛНОЙ части, а мерить будем урезанную, и сверка "
            "утверждала бы неверное")
    if "val_confirm" in a.parts:
        raise SystemExit(
            "val_confirm в этом замере не участвует: геометрия книги не "
            "требует подтверждающей половины, а открыть её можно один раз")

    import torch
    import k14_common as kc
    import k11a_build_hicora_cache as k11a
    import k13a_build_trajectory_basis as k13a
    from k11c_train_d1 import split_episodes
    from k14a_oracle_cache import build_parts
    from depth_rvq_joint12 import code_contribution, nearest_code
    from utils import VisionLanguageActionProcessor
    # ИМПОРТ НУЖЕН РАДИ ПОБОЧНОГО ДЕЙСТВИЯ, а не ради имени: при загрузке
    # пакета исполняются AutoConfig.register("action_codec", ...) и
    # AutoModel.register(...). Без него from_pretrained не узнаёт тип модели
    # и падает с «Transformers does not recognize this architecture».
    # pyflakes считает такой импорт неиспользуемым — удалять его нельзя.
    import actioncodec  # noqa: F401

    head, dirty, _ = kc.check_code_clean(a.allow_dirty)
    print(f"  код: коммит {head}" + ("  (--allow-dirty)" if dirty else ""))
    dev = torch.device(a.device)

    meta = json.load(open(f"{a.cache}.meta.json"))
    src = meta["cache"]
    d = np.load(src, allow_pickle=True)
    N = int(meta["n_obs"])
    epi = np.asarray(d["episode"])[:N].astype(np.int64)
    stp = np.asarray(d["step"])[:N]
    keys_sha = hashlib.sha1(np.ascontiguousarray(
        np.stack([epi, stp])).tobytes()).hexdigest()[:12]
    if keys_sha != meta.get("keys_sha1"):
        raise SystemExit(f"ключи наблюдений {keys_sha} против "
                         f"{meta.get('keys_sha1')}")
    ACT = np.asarray(d["action"])[:N]
    ktrue = np.load(f"{a.cache}.ktrue.npy", mmap_mode="r")

    q0_can, q0_def, q0_man, q0_prov = kc.load_canonical_q0(
        a.q0, gate_r_path=a.gate_r, n_obs=N, keys_sha=keys_sha,
        cache_meta_sha1=k11a.file_sha1(f"{a.cache}.meta.json"))
    idx, _ = k13a.load_split(f"{a.cache}.split.npy", N)
    parts, pmeta = build_parts(idx, epi, kc.SEL_FRAC, kc.SPLIT_SEED, 0,
                               np.random.default_rng(0), split_episodes)
    # РАЗБИЕНИЕ СВЕРЯЕТСЯ С ТЕМ, НА КОТОРОМ СЧИТАН ОРАКУЛ. Иначе числа
    # отсюда нельзя ставить рядом с A0/A01 — это были бы другие строки.
    if a.expect_rows_sha1:
        got = pmeta[a.parts[0]]["rows_sha1"]
        if got != a.expect_rows_sha1:
            raise SystemExit(
                f"строки части {a.parts[0]}: {got}, ожидалось "
                f"{a.expect_rows_sha1}: замер и оракул на разных строках")
        print(f"  строки {a.parts[0]} сверены с оракулом: {got}")

    codec, qs = load_codec(a.ckpt or meta.get("ckpt") or "", dev, torch,
                           VisionLanguageActionProcessor)
    books = []
    with torch.no_grad():
        for lev, qz in enumerate(qs):
            Vl = int(qz.codebook.shape[0])
            codes = torch.arange(Vl, device=dev, dtype=torch.long)[None, :]
            Dl = code_contribution(qz, codes)[0].float()
            books.append(Dl)
            nm_ = Dl.norm(dim=1)
            print(f"  уровень {lev}: словарь {Vl} x {int(Dl.shape[1])} в "
                  f"латенте, нормы {float(nm_.min()):.4f}..."
                  f"{float(nm_.max()):.4f}, медиана "
                  f"{float(nm_.median()):.4f}")
    # --- КОДЕК ЗАВЕРЯЕТСЯ, А НЕ ПРИНИМАЕТСЯ НА ВЕРУ ----------------------
    # Другой кодек тех же форм 2048x512 дал бы правдоподобные числа и прошёл
    # молча, а сравнивать их с A0/A01 из K-14a было бы нельзя: там другой
    # декодер. Те же три отпечатка, что и в K-14a.
    E_all = torch.stack(books).float()
    k11a.check_fingerprints(meta, dict(
        codebooks_sha1=hashlib.sha1(np.ascontiguousarray(
            E_all.cpu().numpy().astype(np.float32)).tobytes()).hexdigest()[:12],
        decoder_probe=k11a.decoder_probe(codec, E_all.to(dev), dev),
        codec_state_sha1=k11a.state_sha1(codec)))
    print("  книги, проба декодера и веса кодека сверены с метой кэша K-11a")
    # ТИПЫ ПРОЕКЦИЙ ПЕЧАТАЮТСЯ. Если обе окажутся Identity, совпадение
    # argmin'ов в двух пространствах тривиально гарантировано, и говорить о
    # нём как о находке нельзя.
    for lev, qz in enumerate(qs):
        ip, op = getattr(qz, "in_project", None), getattr(qz, "out_project",
                                                          None)
        print(f"    уровень {lev}: in_project {type(ip).__name__}, "
              f"out_project {type(op).__name__}")

    V, DIM = int(books[1].shape[0]), int(books[1].shape[1])
    if (V, DIM) != (EXPECT_VOCAB, EXPECT_DIM):
        raise SystemExit(
            f"словарь {V}x{DIM}, ожидался {EXPECT_VOCAB}x{EXPECT_DIM}: "
            f"размеры в вычислении не зашиты, но расхождение с ожиданием "
            f"означает другой кодек, и пороги к нему не относятся")

    res = {}
    for part in a.parts:
        rows = np.asarray(parts[part], np.int64)
        if a.limit:
            rows = rows[:int(a.limit)]
        res[part] = measure_part(part, rows, ACT, epi, ktrue, q0_can, q0_def,
                                 books, codec, qs, dev, torch, nearest_code,
                                 boot=int(a.boot))
        res[part]["split_meta"] = pmeta[part]
    # РЕЖИМ ВЫЧИСЛЕНИЙ ЗАПИСЫВАЕТСЯ ЦЕЛИКОМ. z_e пересчитывается на карте,
    # поэтому результат от неё зависит, и одного слова «cuda:1» мало.
    rt = dict(device=str(dev),
              gpu_uuid=(kc.gpu_uuid(dev, torch) if dev.type == "cuda"
                        else None),
              torch_version=str(torch.__version__),
              cuda_version=str(getattr(torch.version, "cuda", None)),
              tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
              tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
              cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
              cudnn_benchmark=bool(torch.backends.cudnn.benchmark))
    out = dict(kind="k14k_book_span", git_head=head, git_dirty=bool(dirty),
               device=str(dev), runtime=rt, ckpt=a.ckpt,
               codec_fingerprints=dict(
                   codebooks_sha1=meta.get("codebooks_sha1"),
                   decoder_probe=meta.get("decoder_probe"),
                   codec_state_sha1=meta.get("codec_state_sha1")),
               projections={str(l): dict(
                   in_project=type(getattr(q_, "in_project", None)).__name__,
                   out_project=type(getattr(q_, "out_project", None)).__name__)
                   for l, q_ in enumerate(qs)},
               vocab=V, latent_dim=DIM,
               rho_hi=RHO_HI, p_limit=P_LIMIT, alpha_hi=ALPHA_HI,
               boot=int(a.boot), limit=int(a.limit),
               level_norms={str(l): dict(
                   min=float(b.norm(dim=1).min()),
                   median=float(b.norm(dim=1).median()),
                   max=float(b.norm(dim=1).max())) for l, b in
                   enumerate(books)},
               q0_prov=q0_prov, cache=a.cache, source_cache_sha1=sha12(src),
               script_sha1=sha12(os.path.abspath(__file__)), parts=res)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    t_ = a.out + f".tmp.{os.getpid()}"
    json.dump(out, open(t_, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(t_, a.out)
    print(f"\n  сохранено: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
