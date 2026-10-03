#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15c: головы выбора ранга. Чистые модули, без модели VLA и без данных.

ЗАДАЧА. Читатель q1 на h18 упорядочивает коды замороженной книги C1. Из
первых восьми кодов порядка строятся восемь СОГЛАСОВАННЫХ путей: путь j —
это j-й код порядка во всех 16 кодовых позициях. Голова по состоянию
строки выбирает ОДИН из восьми путей; исполняется только он.

ТРИ ГОЛОВЫ, ОДИНАКОВЫЙ ВЫХОД [B, 8]:

    h18_linear      LayerNorm(h18) -> mean по 16 позициям -> Linear(D, 8)
    h24_linear      то же на h24
    h24_candidate   общий scorer, одинаковый для всех восьми кандидатов:
                    видит контекст h24, содержимое пути (средний эмбеддинг
                    его кодов) и признаки его логитов

Первые две нужны для сравнения глубин: польза h24 против h18 — прямой
ответ на вопрос, добавляют ли слои 19-24 селективную информацию.

ПУЛИНГ ДЕЛАЕТСЯ ДО ОБУЧЕНИЯ, И ЭТО ТОЧНО, А НЕ ПРИБЛИЖЁННО. LayerNorm с
обучаемым аффинным преобразованием коммутирует со средним по позициям:

    mean_t(gamma * norm(h_t) + beta) = gamma * mean_t(norm(h_t)) + beta,

поэтому среднее БЕЗАФФИННОЙ нормы считается один раз, а gamma и beta
обучаются уже на нём. Тренеру не нужно держать в памяти [N, 16, D].
Самопроверка сверяет это равенство численно.

ПРИЗНАКИ КАНДИДАТА НЕ ЗАВИСЯТ ОТ ЕГО МЕСТА В МАССИВЕ. Scorer обязан быть
перестановочно-эквивариантным: переставили кандидатов вместе с их кодами и
логитами — переставились оценки. Признаки «отрыв от ранга 0» и «отрыв от
следующего ранга» по номеру в массиве этому противоречили бы, поэтому они
считаются по ЗНАЧЕНИЯМ внутри позиции:

    отрыв от лучшего      lp_j - max_l lp_l           (<= 0)
    отрыв от следующего   lp_j - max{lp_l : lp_l < lp_j}, ноль у младшего

При исходном порядке это ровно «от ранга 0» и «от следующего ранга», но
определение от порядка массива не зависит. Номер ранга голова видит только
через эти значения — как и задумано: судить по содержимому, а не по номеру.
"""
import argparse
import sys

import numpy as np

N_CAND = 8
LN_EPS = 1e-5                  # безаффинная норма до пулинга, фиксирована
N_SCORE_FEATURES = 5
HEADS = ("h18_linear", "h24_linear", "h24_candidate", "h24_candidate_attn")


def ln_mean_pool(h, torch, eps=LN_EPS):
    """mean_t(LayerNorm_без_аффинного(h_t)): [B, T, D] -> [B, D] в fp32."""
    import torch.nn.functional as F
    x = h.float()
    return F.layer_norm(x, (int(x.shape[-1]),), eps=float(eps)).mean(dim=1)


def candidate_score_features(top_lp, torch):
    """Признаки кандидатов по лог-вероятностям: [B, T, 8] -> [B, 8, 5].

    Пять признаков, усреднённых или агрегированных по T кодовым позициям:
    среднее lp, минимум, максимум, средний отрыв от лучшего, средний отрыв
    от следующего. Все определены по ЗНАЧЕНИЯМ, поэтому эквивариантны.
    """
    lp = top_lp.float()
    if lp.dim() != 3:
        raise ValueError(f"ожидалось [B, T, K], получено {tuple(lp.shape)}")
    best = lp.max(dim=-1, keepdim=True).values
    gap_best = lp - best
    # Следующий по значению: максимум среди строго меньших. У младшего
    # такого нет — отрыв ноль.
    less = lp.unsqueeze(-1) > lp.unsqueeze(-2)            # [B,T,K,K]: l < j
    neg_inf = torch.full_like(lp.unsqueeze(-2).expand_as(less.float()),
                              float("-inf"))
    cand = torch.where(less, lp.unsqueeze(-2).expand_as(neg_inf), neg_inf)
    nxt = cand.max(dim=-1).values
    gap_next = torch.where(torch.isfinite(nxt), lp - nxt,
                           torch.zeros_like(lp))
    feats = torch.stack([lp.mean(1), lp.min(1).values, lp.max(1).values,
                         gap_best.mean(1), gap_next.mean(1)], dim=-1)
    return feats


def candidate_embeddings(top_codes, book, torch):
    """Средний по позициям эмбеддинг пути: [B, T, 8] коды -> [B, 8, E]."""
    if top_codes.dim() != 3:
        raise ValueError(f"ожидалось [B, T, K], получено "
                         f"{tuple(top_codes.shape)}")
    return book[top_codes.long()].float().mean(dim=1)


def build_head(name, d_model, e_dim, torch, proj=128, feat_mean=None,
               feat_std=None):
    """Фабрика головы по имени. Все головы возвращают [B, 8]."""
    nn = torch.nn

    class LinearHead(nn.Module):
        """Аффинная часть LayerNorm + Linear(D, 8) над готовым пулингом."""

        def __init__(self):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(d_model))
            self.beta = nn.Parameter(torch.zeros(d_model))
            self.out = nn.Linear(d_model, N_CAND)

        def forward(self, ctx, **_kw):
            return self.out(ctx * self.gamma + self.beta)

    class CandidateHead(nn.Module):
        """Общий scorer: одинаковые веса для всех восьми кандидатов."""

        def __init__(self, attn=False):
            super().__init__()
            self.attn = bool(attn)
            self.gamma = nn.Parameter(torch.ones(d_model))
            self.beta = nn.Parameter(torch.zeros(d_model))
            if self.attn:
                self.attn_q = nn.Linear(d_model, 1)
            self.ctx_proj = nn.Linear(d_model, proj)
            self.cand_norm = nn.LayerNorm(e_dim)
            self.cand_proj = nn.Linear(e_dim, proj)
            fm = (torch.zeros(N_SCORE_FEATURES) if feat_mean is None
                  else torch.as_tensor(feat_mean, dtype=torch.float32))
            fs = (torch.ones(N_SCORE_FEATURES) if feat_std is None
                  else torch.as_tensor(feat_std, dtype=torch.float32))
            # СТАНДАРТИЗАЦИЯ ПРИЗНАКОВ — БУФЕРЫ, А НЕ ПАРАМЕТРЫ: считается по
            # train один раз и сохраняется вместе с состоянием, иначе
            # загруженная голова видела бы признаки в другом масштабе.
            self.register_buffer("feat_mean", fm.clone())
            self.register_buffer("feat_std", fs.clamp_min(1e-6).clone())
            self.feat_proj = nn.Linear(N_SCORE_FEATURES, proj)
            self.mlp = nn.Sequential(nn.GELU(), nn.Linear(3 * proj, proj),
                                     nn.GELU(), nn.Linear(proj, 1))

        def pooled(self, ctx=None, h_full=None):
            if self.attn:
                import torch.nn.functional as F
                if h_full is None:
                    raise ValueError("пулинг вниманием требует полного h24")
                x = F.layer_norm(h_full.float(), (d_model,), eps=LN_EPS)
                x = x * self.gamma + self.beta
                w = torch.softmax(self.attn_q(x).squeeze(-1), dim=-1)
                return (w.unsqueeze(-1) * x).sum(1)
            if ctx is None:
                raise ValueError("нужен готовый пулинг контекста")
            return ctx * self.gamma + self.beta

        def forward(self, ctx=None, cand_emb=None, cand_feat=None,
                    h_full=None):
            c = self.ctx_proj(self.pooled(ctx, h_full))           # [B, P]
            e = self.cand_proj(self.cand_norm(cand_emb.float()))   # [B,8,P]
            f = self.feat_proj((cand_feat.float() - self.feat_mean)
                               / self.feat_std)                    # [B,8,P]
            z = torch.cat([c.unsqueeze(1).expand_as(e), e, f], dim=-1)
            return self.mlp(z).squeeze(-1)                          # [B, 8]

    if name in ("h18_linear", "h24_linear"):
        return LinearHead()
    if name == "h24_candidate":
        return CandidateHead(attn=False)
    if name == "h24_candidate_attn":
        return CandidateHead(attn=True)
    raise ValueError(f"голова {name!r} не бывает: {HEADS}")


def expected_regret(scores, costs, torch):
    """sum_j softmax(s)_j * (e_j - min_l e_l), среднее по строкам.

    Затраты — ИСХОДНЫЕ построчные MSE. Построчная нормировка на ошибку
    черновика изменила бы веса строк относительно глобального action RMS,
    по которому идёт отбор.
    """
    if scores.shape != costs.shape:
        raise ValueError(f"формы {tuple(scores.shape)} и "
                         f"{tuple(costs.shape)}")
    r = costs - costs.min(dim=-1, keepdim=True).values
    return (torch.softmax(scores.float(), dim=-1) * r).sum(-1).mean()


def hard_selected_cost(scores, costs, torch):
    """Исполняемый путь: argmax оценок, его построчная MSE."""
    pick = scores.argmax(dim=-1)
    return costs.gather(-1, pick.unsqueeze(-1)).squeeze(-1), pick


def selftest():
    import torch
    torch.manual_seed(0)

    # --- ПУЛИНГ: НОРМА С АФФИННЫМ КОММУТИРУЕТ СО СРЕДНИМ -----------------
    B, T, D, E = 5, 16, 12, 6
    h = torch.randn(B, T, D) * 3.0 + 1.0
    gamma, beta = torch.randn(D), torch.randn(D)
    ln = torch.nn.LayerNorm(D, eps=LN_EPS)
    with torch.no_grad():
        ln.weight.copy_(gamma)
        ln.bias.copy_(beta)
    direct = ln(h).mean(1)
    pooled = ln_mean_pool(h, torch) * gamma + beta
    assert torch.allclose(direct, pooled, atol=1e-5), \
        float((direct - pooled).abs().max())

    # --- ПРИЗНАКИ КАНДИДАТОВ --------------------------------------------
    lp = torch.tensor([[[-1.0, -2.0, -2.5, -4.0, -5.0, -6.0, -7.0, -9.0]]])
    f = candidate_score_features(lp, torch)
    assert f.shape == (1, 8, 5), f.shape
    # в исходном порядке «от лучшего» = от ранга 0, «от следующего» = от j+1
    assert torch.allclose(f[0, :, 3], lp[0, 0] - (-1.0))
    want_next = torch.tensor([1.0, 0.5, 1.5, 1.0, 1.0, 1.0, 2.0, 0.0])
    assert torch.allclose(f[0, :, 4], want_next), f[0, :, 4]
    # НИЧЬИ: следующий — строго меньший, равные не считаются
    lpt = torch.tensor([[[-1.0, -1.0, -3.0, -3.0, -4.0, -5.0, -6.0, -6.0]]])
    ft = candidate_score_features(lpt, torch)
    assert torch.allclose(ft[0, :, 4], torch.tensor(
        [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])), ft[0, :, 4]
    assert torch.isfinite(ft).all()
    # ЭКВИВАРИАНТНОСТЬ ПРИЗНАКОВ
    lpr = torch.randn(3, T, 8)
    perm = torch.randperm(8)
    assert torch.allclose(candidate_score_features(lpr[..., perm], torch),
                          candidate_score_features(lpr, torch)[:, perm],
                          atol=1e-6)

    # --- ЭМБЕДДИНГИ ПУТЕЙ -----------------------------------------------
    book = torch.randn(30, E)
    codes = torch.randint(0, 30, (B, T, 8))
    ce = candidate_embeddings(codes, book, torch)
    assert ce.shape == (B, 8, E)
    assert torch.allclose(ce[2, 3], book[codes[2, :, 3]].mean(0), atol=1e-6)

    # --- ГОЛОВЫ: ФОРМА, ЭКВИВАРИАНТНОСТЬ, НЕСМЕШИВАНИЕ БАТЧА -------------
    ctx = torch.randn(B, D)
    cf = candidate_score_features(torch.randn(B, T, 8), torch)
    for name in HEADS:
        head = build_head(name, D, E, torch, proj=16).eval()
        kw = dict(ctx=ctx, cand_emb=ce, cand_feat=cf, h_full=h)
        s = head(**kw)
        assert s.shape == (B, 8) and torch.isfinite(s).all(), (name, s.shape)
        # СТРОКИ БАТЧА НЕ СМЕШИВАЮТСЯ: оценка строки не зависит от соседей
        s1 = head(ctx=ctx[1:2], cand_emb=ce[1:2], cand_feat=cf[1:2],
                  h_full=h[1:2])
        assert torch.allclose(s1, s[1:2], atol=1e-5), name
        if "candidate" in name:
            p = torch.randperm(8)
            sp = head(ctx=ctx, cand_emb=ce[:, p], cand_feat=cf[:, p],
                      h_full=h)
            assert torch.allclose(sp, s[:, p], atol=1e-5), name
    # ЛИНЕЙНАЯ ГОЛОВА ЭКВИВАРИАНТНОЙ НЕ ЯВЛЯЕТСЯ — и не должна: она выдаёт
    # оценку НОМЕРУ ранга, это и есть её отличие от candidate
    try:
        build_head("нет", D, E, torch)
    except ValueError as e:
        assert "не бывает" in str(e), e
    else:
        raise AssertionError("принята несуществующая голова")

    # --- СОХРАНЕНИЕ И ЗАГРУЗКА ВОСПРОИЗВОДЯТ ВЫХОД ----------------------
    head = build_head("h24_candidate", D, E, torch, proj=16,
                      feat_mean=cf.mean((0, 1)), feat_std=cf.std((0, 1)))
    s0 = head(ctx=ctx, cand_emb=ce, cand_feat=cf)
    st = {k: v.clone() for k, v in head.state_dict().items()}
    assert "feat_mean" in st and "feat_std" in st
    other = build_head("h24_candidate", D, E, torch, proj=16)
    other.load_state_dict(st)
    assert torch.equal(other(ctx=ctx, cand_emb=ce, cand_feat=cf), s0)

    # --- ПОТЕРИ ----------------------------------------------------------
    costs = torch.tensor([[1.0, 0.5, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                          [0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]])
    big = torch.full((2, 8), -50.0)
    big[0, 1] = 50.0
    big[1, 0] = 50.0
    assert float(expected_regret(big, costs, torch)) < 1e-6
    uniform = torch.zeros(2, 8)
    want = ((costs - costs.min(1, keepdim=True).values).mean(1)).mean()
    assert abs(float(expected_regret(uniform, costs, torch))
               - float(want)) < 1e-6
    sel, pick = hard_selected_cost(big, costs, torch)
    assert pick.tolist() == [1, 0], pick
    assert torch.allclose(sel, torch.tensor([0.5, 0.2])), sel
    try:
        expected_regret(torch.zeros(2, 7), costs, torch)
    except ValueError as e:
        assert "формы" in str(e), e
    else:
        raise AssertionError("приняты формы разной длины")
    # РЕГРЕТ ДИФФЕРЕНЦИРУЕМ И ТЯНЕТ К ЛУЧШЕМУ
    s_ = torch.zeros(2, 8, requires_grad=True)
    expected_regret(s_, costs, torch).backward()
    assert int(s_.grad[0].argmin()) == 1 and int(s_.grad[1].argmin()) == 0
    print(f"самопроверка k15c_rank_selector пройдена: {len(HEADS)} головы, "
          f"{N_SCORE_FEATURES} признаков кандидата, эквивариантность и "
          f"точность пулинга сверены")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="K-15c: головы выбора ранга")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    raise SystemExit("это модуль; отдельно запускать нечего")
