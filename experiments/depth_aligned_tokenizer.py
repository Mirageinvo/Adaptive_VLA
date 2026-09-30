#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15: примитивы depth-aligned остаточного токенизатора.

Файл НЕ загружает VLA и датасет: только проверяемые строительные блоки, чтобы
их можно было тестировать без кластера.

ДВА РАСПРЕДЕЛЕНИЯ НА ОДНУ КНИГУ. Q_i(k | r_i) — токенизатор, видит целевое
действие через остаток; P_i(k | h, q_<i) — политика, видит скрытое состояние
назначенной глубины. Q говорит, какие поправки полезны; P — реальный путь
вывода. Двустороннее согласование позволяет книге сдвигаться к различиям,
доступным этой глубине; одностороннее Q->P дало бы лишь ещё одного читателя
для reconstruction-книги, то есть повторение K-14.

ТОЧНОСТЬ ЗНАЧЕНИЯ ST — ЧАСТЬ КОНТРАКТА, А НЕ ПОЖЕЛАНИЕ. Гейт §15a сравнивает
эмбеддинги, накопленный латент и декодированное действие ПОБИТОВО, и это
законно только если прямое значение ST равно строке книги в точности. Прежняя
формула `soft + (hard - soft).detach()` (та же, что в K-14) этого НЕ даёт:
`.detach()` меняет градиент, а не значение, и `soft + (hard - soft)` во float
округляется. Измерено на рабочих размерностях V=2048, D=512: расходятся 4124
элемента из 65536, максимум 2.4e-07. Здесь используется форма, точная по
построению.
"""
import argparse
import sys

import torch
import torch.nn.functional as F


def _check_temperature(value: float, name: str) -> float:
    value = float(value)
    if not value > 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def mean_squared_distances(residual: torch.Tensor,
                           book: torch.Tensor) -> torch.Tensor:
    """||r - e_k||^2 / D для всех k, без тензора (..., V, D).

    Раскрытие ||r||^2 - 2<r,e> + ||e||^2 нужно именно затем, чтобы не
    материализовать (..., V, D): при V=2048, D=512 и батче 8x16 это 134 млн
    элементов на уровень.
    """
    if residual.ndim < 2:
        raise ValueError(
            f"residual must have at least two dims, got {residual.shape}")
    if book.ndim != 2:
        raise ValueError(f"book must be (V,D), got {book.shape}")
    if residual.shape[-1] != book.shape[-1]:
        raise ValueError(
            f"width differs: residual {residual.shape[-1]}, "
            f"book {book.shape[-1]}")
    if not (torch.isfinite(residual).all() and torch.isfinite(book).all()):
        raise ValueError("residual and book must be finite")
    r = residual.float()
    e = book.float()
    r2 = (r * r).sum(-1, keepdim=True)
    e2 = (e * e).sum(-1)
    inner = r @ e.T
    dim = float(e.shape[-1])
    return (r2 - 2.0 * inner + e2) / dim


def tokenizer_logits(residual: torch.Tensor, book: torch.Tensor,
                     temperature: float = 1.0) -> torch.Tensor:
    """Логиты мягкого остаточного квантователя."""
    temperature = _check_temperature(temperature, "tokenizer temperature")
    return -mean_squared_distances(residual, book) / temperature


def hard_straight_through(logits: torch.Tensor, book: torch.Tensor,
                          temperature: float = 1.0):
    """Жёсткий lookup вперёд, мягкое ожидание назад. Значение ТОЧНОЕ.

    Возвращает (embedding, indices, probabilities).

    ФОРМУЛА ВЫБРАНА ИЗ ДВУХ, И РАЗЛИЧИЕ НЕ КОСМЕТИЧЕСКОЕ:

        hard.detach() + (soft - soft.detach())   <- здесь
            значение = hard побитово; градиент по книге идёт ТОЛЬКО через
            soft, то есть строка получает вес своей вероятности. Это
            сохраняет семантику градиента, записанную в плане.

        hard + (soft - soft.detach())
            значение тоже точное, но F.embedding дифференцируема по книге, и
            выбранная строка получает ПОЛНЫЙ градиент потери плюс мягкий, как
            в VQ-VAE. При пиковом softmax формы почти совпадают, при
            размазанном — нет.

    Вторая форма — осознанная следующая проба, если книги окажутся
    неподвижными, а не молчаливая замена.
    """
    temperature = _check_temperature(temperature, "policy temperature")
    if logits.ndim < 2 or book.ndim != 2:
        raise ValueError(
            f"bad shapes: logits {logits.shape}, book {book.shape}")
    if logits.shape[-1] != book.shape[0]:
        raise ValueError(
            f"vocabulary differs: logits {logits.shape[-1]}, "
            f"book {book.shape[0]}")
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    soft = probs @ book.to(probs.dtype)
    indices = logits.argmax(dim=-1)
    hard = F.embedding(indices, book.to(probs.dtype))
    # soft - soft.detach() равно РОВНО нулю по значению: это один и тот же
    # тензор, вычитание даёт 0.0 в каждом элементе без округления.
    embedding = hard.detach() + (soft - soft.detach())
    return embedding, indices, probs


def bidirectional_alignment(tokenizer_logits_: torch.Tensor,
                            policy_logits: torch.Tensor) -> dict:
    """Две кросс-энтропии со stop-gradient между токенизатором и политикой.

    `tokenizer_to_policy` учит политику читать разбиение со стороны действия.
    `policy_to_tokenizer` — то, из-за чего это depth alignment, а не ещё один
    читатель: оно позволяет самому разбиению сдвигаться к различиям,
    представленным на назначенной глубине.
    """
    if tokenizer_logits_.shape != policy_logits.shape:
        raise ValueError(
            f"posterior shapes differ: {tokenizer_logits_.shape} and "
            f"{policy_logits.shape}")
    q = torch.softmax(tokenizer_logits_.float(), dim=-1)
    p = torch.softmax(policy_logits.float(), dim=-1)
    q_to_p = -(q.detach() * torch.log_softmax(
        policy_logits.float(), dim=-1)).sum(-1).mean()
    p_to_q = -(p.detach() * torch.log_softmax(
        tokenizer_logits_.float(), dim=-1)).sum(-1).mean()
    return {"tokenizer_to_policy": q_to_p,
            "policy_to_tokenizer": p_to_q,
            "total": 0.5 * (q_to_p + p_to_q)}


def usage_kl_to_uniform(probabilities: torch.Tensor) -> torch.Tensor:
    """KL среднего использования кодов до равномерного.

    НЕ УТВЕРЖДЕНИЕ, ЧТО ИСТИННОЕ РАСПРЕДЕЛЕНИЕ РАВНОМЕРНО. Член нужен только
    против полного схлопывания и обязан иметь малый вес; perplexity, мёртвые
    коды и максимальная доля кода печатаются отдельно.
    """
    if probabilities.ndim < 2:
        raise ValueError(
            f"probabilities need a vocabulary dim: {probabilities.shape}")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    flat = probabilities.reshape(-1, probabilities.shape[-1]).float()
    mean_p = flat.mean(0)
    mean_p = mean_p / mean_p.sum().clamp_min(1e-12)
    vocab = mean_p.numel()
    return (mean_p * (mean_p.clamp_min(1e-12).log() + torch.log(
        torch.tensor(float(vocab), device=mean_p.device)))).sum()


def monotonic_hinge(previous_error: torch.Tensor,
                    refined_error: torch.Tensor,
                    margin: float = 0.0) -> torch.Tensor:
    """Штраф за построчное ухудшение относительно предыдущего черновика.

    МЯГКИЙ, А НЕ ЗАПРЕТ: изменение с большей imitation error иногда
    поведенчески полезно. И сама imitation error — слабый прокси
    (§48.4, §50.4), поэтому вес этого члена держится малым.
    """
    if previous_error.shape != refined_error.shape:
        raise ValueError(
            f"error shapes differ: {previous_error.shape} and "
            f"{refined_error.shape}")
    return F.relu(refined_error - previous_error + float(margin)).mean()


def code_usage_stats(indices: torch.Tensor, vocab: int) -> dict:
    """Perplexity, мёртвые коды и максимальная доля — печатаются всегда.

    Схлопывание книги обнаруживается этими числами, а не значением
    регуляризатора: он мал по построению и на схлопывание почти не реагирует.
    """
    flat = indices.reshape(-1)
    counts = torch.bincount(flat, minlength=int(vocab)).float()
    total = counts.sum().clamp_min(1.0)
    p = counts / total
    nz = p[p > 0]
    entropy = float(-(nz * nz.log()).sum())
    return {"perplexity": float(torch.exp(torch.tensor(entropy))),
            "dead_codes": int((counts == 0).sum()),
            "max_code_share": float(p.max()),
            "used_codes": int((counts > 0).sum()),
            "vocab": int(vocab)}


def selftest() -> None:
    torch.manual_seed(0)

    # --- расстояния и логиты ---------------------------------------------
    residual = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    book = torch.tensor([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    dist = mean_squared_distances(residual, book)
    assert dist.shape == (1, 2, 3)
    assert dist.argmin(-1).tolist() == [[1, 2]]
    assert tokenizer_logits(residual, book).argmax(-1).tolist() == [[1, 2]]

    # --- ТОЧНОСТЬ ЗНАЧЕНИЯ НА РАБОЧИХ РАЗМЕРНОСТЯХ -----------------------
    # Игрушечные размеры этого не проверяют: суммы там случайно не
    # округляются, и тест проходит у ЛЮБОЙ формулы. Именно так прежняя
    # неточная версия и прошла свой тест.
    V, D = 2048, 512
    big_book = torch.randn(V, D)
    big_logits = torch.randn(8, 16, V)
    for temperature in (1.0, 0.25, 4.0):
        emb, idx, probs = hard_straight_through(
            big_logits, big_book, temperature=temperature)
        exact = big_book[idx]
        assert torch.equal(emb.detach(), exact), (
            f"tau={temperature}: значение ST не равно строке книги; "
            f"расходятся {int((emb.detach() != exact).sum())} элементов")
        assert torch.equal(idx, big_logits.argmax(-1))
        assert probs.shape == big_logits.shape
        assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)))
    # прежняя формула на тех же данных ТОЧНОЙ НЕ БЫЛА — фиксируем разницу,
    # чтобы правку нельзя было откатить незаметно
    probs = torch.softmax(big_logits.float(), dim=-1)
    soft = probs @ big_book
    idx = big_logits.argmax(-1)
    legacy = soft + (F.embedding(idx, big_book) - soft)
    assert not torch.equal(legacy, big_book[idx]), \
        "прежняя формула внезапно стала точной — тест потерял смысл"

    # --- ГРАДИЕНТЫ ДОХОДЯТ ------------------------------------------------
    lg = torch.randn(4, 6, 8, requires_grad=True)
    bk = torch.randn(8, 5, requires_grad=True)
    emb, _idx, _p = hard_straight_through(lg, bk)
    emb.sum().backward()
    assert lg.grad is not None and float(lg.grad.abs().sum()) > 0
    assert bk.grad is not None and float(bk.grad.abs().sum()) > 0

    # --- ДВУСТОРОННЕЕ СОГЛАСОВАНИЕ ----------------------------------------
    tq = torch.randn(3, 4, 7, requires_grad=True)
    pl = torch.randn(3, 4, 7, requires_grad=True)
    al = bidirectional_alignment(tq, pl)
    assert set(al) == {"tokenizer_to_policy", "policy_to_tokenizer", "total"}
    al["total"].backward()
    assert tq.grad is not None and float(tq.grad.abs().sum()) > 0, \
        "направление P->Q не даёт градиента книге: это снова K-14"
    assert pl.grad is not None and float(pl.grad.abs().sum()) > 0
    # совпадающие распределения -> оба члена равны энтропии, разность нулевая
    same = torch.randn(2, 3, 5)
    al2 = bidirectional_alignment(same, same.clone())
    assert abs(float(al2["tokenizer_to_policy"])
               - float(al2["policy_to_tokenizer"])) < 1e-5
    try:
        bidirectional_alignment(torch.randn(2, 3), torch.randn(2, 4))
    except ValueError as e:
        assert "shapes differ" in str(e)
    else:
        raise AssertionError("приняты разные формы распределений")

    # --- АНТИ-СХЛОПЫВАНИЕ --------------------------------------------------
    uniform = torch.full((4, 6), 1.0 / 6)
    assert abs(float(usage_kl_to_uniform(uniform))) < 1e-6
    collapsed = torch.zeros(4, 6)
    collapsed[:, 0] = 1.0
    assert float(usage_kl_to_uniform(collapsed)) > 1.0

    # --- МОНОТОННЫЙ ШТРАФ --------------------------------------------------
    prev = torch.tensor([1.0, 1.0, 1.0])
    new = torch.tensor([0.5, 1.0, 1.2])
    assert torch.allclose(monotonic_hinge(prev, new),
                          torch.tensor(0.2 / 3.0))
    assert float(monotonic_hinge(prev, prev)) == 0.0
    try:
        monotonic_hinge(prev, new[:2])
    except ValueError as e:
        assert "shapes differ" in str(e)
    else:
        raise AssertionError("приняты разные формы ошибок")

    # --- СТАТИСТИКА ИСПОЛЬЗОВАНИЯ КОДОВ ------------------------------------
    st = code_usage_stats(torch.zeros(100, dtype=torch.long), 2048)
    assert st["dead_codes"] == 2047 and abs(st["max_code_share"] - 1.0) < 1e-9
    assert st["perplexity"] < 1.01, st["perplexity"]
    st2 = code_usage_stats(torch.arange(2048), 2048)
    assert st2["dead_codes"] == 0 and st2["perplexity"] > 2000

    # --- ОТКАЗЫ НА НЕВЕРНЫХ ВХОДАХ ----------------------------------------
    for fn, args, why in (
            (mean_squared_distances, (torch.randn(4), book), "at least two"),
            (mean_squared_distances, (residual, torch.randn(3)), "(V,D)"),
            (mean_squared_distances, (residual, torch.randn(3, 5)), "width"),
            (tokenizer_logits, (residual, book, 0.0), "must be positive"),
            (hard_straight_through, (torch.randn(2, 3), torch.randn(4, 5)),
             "vocabulary differs")):
        try:
            fn(*args)
        except ValueError as e:
            assert why in str(e), (why, e)
        else:
            raise AssertionError(f"принят неверный вход: {why}")
    nan_book = book.clone()
    nan_book[0, 0] = float("nan")
    try:
        mean_squared_distances(residual, nan_book)
    except ValueError as e:
        assert "finite" in str(e)
    else:
        raise AssertionError("принята книга с nan")

    print("самопроверка depth_aligned_tokenizer пройдена")


def main() -> int:
    ap = argparse.ArgumentParser(description="K-15 примитивы токенизатора")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not a.selftest:
        raise SystemExit("нужен --selftest: модуль не имеет другого действия")
    selftest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
