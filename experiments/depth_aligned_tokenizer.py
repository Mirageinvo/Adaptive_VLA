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

import math

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
    # ОТСЕЧЕНИЕ НУЛЁМ ОБЯЗАТЕЛЬНО. Раскрытая формула даёт слегка
    # отрицательный квадрат расстояния, когда остаток почти совпадает со
    # строкой книги: r2 и e2 считаются независимо и сокращаются не точно.
    # Для логитов знак несущественен, но эта же функция печатается как
    # расстояние, а отрицательное расстояние — ложь в отчёте. Отсечение
    # может изменить argmin только если ДВЕ строки одновременно ушли ниже
    # нуля, то есть обе лежат в пределах округления float32 от остатка; в
    # этом случае выбор произволен и в точной арифметике тоже.
    return ((r2 - 2.0 * inner + e2) / dim).clamp_min(0.0)


def soft_perplexity(probabilities: torch.Tensor) -> torch.Tensor:
    """Построчная perplexity МЯГКОГО апостериора: exp(энтропия).

    Нужна затем, что hard-perplexity по argmin и мягкая — разные величины.
    Первая может быть здоровой, пока вторая равна размеру словаря, то есть
    апостериор равномерен и не несёт ничего. Ровно это и случилось.
    """
    p = probabilities.float()
    entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)
    return entropy.exp()


def calibrate_temperature(distances: torch.Tensor, target_perplexity: float,
                          *, lo: float = 1e-10, hi: float = 1e10,
                          iterations: int = 80) -> float:
    """tau, при которой МЕДИАННАЯ мягкая perplexity равна целевой.

    ЗАЧЕМ. `mean_squared_distances` делит квадрат расстояния на D, поэтому
    при tau = 1 разброс логитов порядка 1/D, и softmax практически
    равномерен: измерено расхождение с равномерным в 0.15 % по кросс-
    энтропии. При равномерном Q член выравнивания превращается в давление
    «сделай P равномерным», градиент P->Q в книги почти нулевой, а член
    использования не имеет сигнала. Жёсткий выбор (argmin) от tau не
    зависит вовсе, поэтому поломка была не видна по путям a1_tok/a2_tok.

    Perplexity монотонно растёт по tau, поэтому деление отрезка корректно.
    """
    target = float(target_perplexity)
    vocab = int(distances.shape[-1])
    if not 1.0 < target < vocab:
        raise ValueError(
            f"целевая perplexity {target} вне (1, {vocab})")

    def ppl(tau: float) -> float:
        probabilities = torch.softmax(-distances.float() / tau, dim=-1)
        return float(soft_perplexity(probabilities).median())

    if ppl(lo) > target or ppl(hi) < target:
        raise ValueError(
            f"целевая perplexity {target} недостижима: при tau={lo} "
            f"получается {ppl(lo):.3f}, при tau={hi} — {ppl(hi):.3f}")
    left, right = lo, hi
    for _ in range(int(iterations)):
        middle = math.sqrt(left * right)
        if ppl(middle) < target:
            left = middle
        else:
            right = middle
    return math.sqrt(left * right)


def tokenizer_logits(residual: torch.Tensor, book: torch.Tensor,
                     temperature: float = 1.0) -> torch.Tensor:
    """Логиты мягкого остаточного квантователя."""
    temperature = _check_temperature(temperature, "tokenizer temperature")
    return -mean_squared_distances(residual, book) / temperature


def quantize_residual(residual: torch.Tensor, book: torch.Tensor,
                      temperature: float = 1.0):
    """Логиты и жёсткий выбор для остатка: температура применяется ОДИН раз.

    Возвращает (logits, embedding, indices, probabilities).

    ЗАЧЕМ ОТДЕЛЬНАЯ ФУНКЦИЯ. Композиция `tokenizer_logits(r, C, tau)` и
    `hard_straight_through(logits, C, tau)` применяет температуру ДВАЖДЫ:
    softmax(-d / tau^2). При tau = 1 это незаметно, а при любом другом —
    молча другой эксперимент, причём `probabilities` разошлись бы с теми,
    по которым считается выравнивание. Здесь единственный правильный
    порядок зафиксирован в одном месте.
    """
    logits = tokenizer_logits(residual, book, temperature=temperature)
    embedding, indices, probabilities = hard_straight_through(
        logits, book, temperature=1.0)
    return logits, embedding, indices, probabilities


def hard_straight_through(logits: torch.Tensor, book: torch.Tensor,
                          temperature: float = 1.0):
    """Жёсткий lookup вперёд, мягкое ожидание назад. Значение ТОЧНОЕ.

    Возвращает (embedding, indices, probabilities).

    ВНИМАНИЕ: `temperature` применяется к УЖЕ ГОТОВЫМ логитам. Если они
    получены из `tokenizer_logits`, температура там уже применена, и здесь
    нужна 1.0 — либо, лучше, вызов `quantize_residual`.

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


# СГЛАЖИВАНИЕ ПО УМОЛЧАНИЮ ВЫКЛЮЧЕНО. Его ввели по неверному доводу — см.
# docstring ниже — и обратно не включают без замера норм градиентов.
ALIGNMENT_SMOOTHING = 0.0


def bidirectional_alignment(tokenizer_logits_: torch.Tensor,
                            policy_logits: torch.Tensor,
                            smoothing: float = ALIGNMENT_SMOOTHING) -> dict:
    """Две кросс-энтропии со stop-gradient между токенизатором и политикой.

    `tokenizer_to_policy` учит политику читать разбиение со стороны действия.
    `policy_to_tokenizer` — то, из-за чего это depth alignment, а не ещё один
    читатель: оно позволяет самому разбиению сдвигаться к различиям,
    представленным на назначенной глубине.

    О СГЛАЖИВАНИИ, И ПОЧЕМУ ОНО ВЫКЛЮЧЕНО ПО УМОЛЧАНИЮ.

    Измеренный факт: после калибровки температуры `policy_to_tokenizer`
    уровня 2 равнялась 225.6 при log V = 7.62, то есть ЗНАЧЕНИЕ члена
    превышало величину, на которую он нормируется, в 29.6 раза.

    Мой прежний вывод из этого — «такой градиент разрушит книги» — НЕВЕРЕН.
    У кросс-энтропии через `log_softmax` градиент по логитам равен
    (softmax - цель) и ограничен единицей по модулю, каким бы большим ни
    было само значение. Большое значение меняет вклад члена в СУММУ, но не
    величину градиента.

    Настоящий усилитель здесь другой: логиты токенизатора равны -d / tau
    при tau порядка 1e-05, поэтому градиент по расстоянию, а значит и по
    книге, получает множитель 1 / tau. Ограничение значения потери этого
    не контролирует никак.

    Сглаживание предсказываемой стороны внутри логарифма ограничивает
    каждый член величиной log(V / eps), но делает и кое-что ещё, о чём
    умалчивал прежний комментарий: при p_j -> 0 производная по
    соответствующему логиту стремится к НУЛЮ, то есть подавляется
    исправляющий градиент именно на самых тяжёлых несовпадениях. Для
    случайной головы это может работать как неявный прогрев, но это другой
    механизм, и он не проверен. Поэтому eps = 0 по умолчанию, а решение
    принимается по замеренным нормам градиентов по компонентам.
    """
    if tokenizer_logits_.shape != policy_logits.shape:
        raise ValueError(
            f"posterior shapes differ: {tokenizer_logits_.shape} and "
            f"{policy_logits.shape}")
    eps = float(smoothing)
    if not 0.0 <= eps < 1.0:
        raise ValueError(f"smoothing must be in [0, 1): {eps}")
    vocab = int(tokenizer_logits_.shape[-1])

    def smoothed_log(logits: torch.Tensor) -> torch.Tensor:
        # ПРИ eps = 0 — РОВНО `log_softmax`, БЕЗ clamp_min. Прежняя версия
        # брала здесь `softmax(...).clamp_min(1e-30).log()`, и это НЕ
        # несглаженная кросс-энтропия: значение упиралось в -log(1e-30) =
        # 69.08 (измеренные 225.6 такая реализация выдать не могла), а
        # градиент по коду с вероятностью ниже 1e-30 обнулялся — то есть
        # подавление тяжёлых несовпадений оставалось и при выключенном
        # сглаживании. Тогда замер «P->Q почти нулевой» в смоуке говорил бы
        # не о геометрии и не о tau, а об этом clamp.
        if eps == 0.0:
            return torch.log_softmax(logits.float(), dim=-1)
        return ((1.0 - eps) * torch.softmax(logits.float(), dim=-1)
                + eps / vocab).log()

    q = torch.softmax(tokenizer_logits_.float(), dim=-1)
    p = torch.softmax(policy_logits.float(), dim=-1)
    q_to_p = -(q.detach() * smoothed_log(policy_logits)).sum(-1).mean()
    p_to_q = -(p.detach() * smoothed_log(tokenizer_logits_)).sum(-1).mean()
    bound = math.log(vocab / eps) if eps > 0 else float("inf")
    return {"tokenizer_to_policy": q_to_p,
            "policy_to_tokenizer": p_to_q,
            "total": 0.5 * (q_to_p + p_to_q),
            "bound": bound,
            "smoothing": eps}


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
    assert set(al) == {"tokenizer_to_policy", "policy_to_tokenizer", "total",
                       "bound", "smoothing"}
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

    # --- ТЕМПЕРАТУРА ПРИМЕНЯЕТСЯ РОВНО ОДИН РАЗ ---------------------------
    # Композиция двух функций применяла её дважды, и на этом можно было
    # потерять эксперимент: probabilities разошлись бы с логитами, по
    # которым считается выравнивание.
    r_t = torch.randn(3, 5, 16)
    b_t = torch.randn(32, 16)
    for tau_t in (0.25, 1.0, 3.0):
        lg_t, emb_t, idx_t, pr_t = quantize_residual(r_t, b_t,
                                                     temperature=tau_t)
        want_lg = -mean_squared_distances(r_t, b_t) / tau_t
        assert torch.equal(lg_t, want_lg), tau_t
        assert torch.equal(pr_t, torch.softmax(want_lg, dim=-1)), tau_t
        assert torch.equal(emb_t, F.embedding(idx_t, b_t.float())), tau_t
        double = hard_straight_through(lg_t, b_t, temperature=tau_t)[2]
        if tau_t != 1.0:
            assert not torch.equal(double, pr_t), (
                "двойное применение температуры перестало отличаться: "
                "регрессионная проверка больше ничего не ловит")

    # --- РАССТОЯНИЕ НЕ БЫВАЕТ ОТРИЦАТЕЛЬНЫМ -------------------------------
    b_big = torch.randn(64, 128) * 7.0
    r_on = b_big[[5, 9, 9, 40]].clone().reshape(2, 2, 128)
    d_on = mean_squared_distances(r_on, b_big)
    assert float(d_on.min()) >= 0.0, float(d_on.min())
    assert d_on.argmin(-1).reshape(-1).tolist() == [5, 9, 9, 40]

    # --- КАЛИБРОВКА ТЕМПЕРАТУРЫ -------------------------------------------
    # ВОСПРОИЗВЕДЕНИЕ ПОЛОМКИ: при делении на D апостериор равномерен.
    torch.manual_seed(7)
    D_real, V_real = 512, 2048
    book_r = torch.randn(V_real, D_real) * 0.05
    resid_r = torch.randn(4, 8, D_real) * 0.05
    d_r = mean_squared_distances(resid_r, book_r)
    flat = torch.softmax(-d_r, dim=-1)
    ppl_flat = float(soft_perplexity(flat).median())
    assert ppl_flat > 0.99 * V_real, (
        f"мягкая perplexity при tau=1 равна {ppl_flat:.1f}, а поломка "
        f"состояла в том, что она почти равна {V_real}: проверка больше "
        f"не воспроизводит то, из-за чего введена калибровка")

    for target in (5.0, 20.0, 100.0):
        tau_c = calibrate_temperature(d_r, target)
        got = float(soft_perplexity(
            torch.softmax(-d_r / tau_c, dim=-1)).median())
        assert abs(got - target) / target < 0.01, (target, got, tau_c)
        # ЖЁСТКИЙ ВЫБОР ОТ ТЕМПЕРАТУРЫ НЕ ЗАВИСИТ — поэтому поломка и была
        # невидима по путям a1_tok/a2_tok
        assert torch.equal((-d_r).argmax(-1), (-d_r / tau_c).argmax(-1))
    # МОНОТОННОСТЬ: большая tau — более равномерно
    t_small = calibrate_temperature(d_r, 5.0)
    t_big = calibrate_temperature(d_r, 100.0)
    assert t_small < t_big, (t_small, t_big)
    for bad_target, why in ((1.0, "вне"), (float(V_real), "вне"),
                            (0.5, "вне")):
        try:
            calibrate_temperature(d_r, bad_target)
        except ValueError as e:
            assert why in str(e), (bad_target, e)
        else:
            raise AssertionError(f"принята цель {bad_target}")

    # --- ВЫРАВНИВАНИЕ ОГРАНИЧЕНО СВЕРХУ ----------------------------------
    # ВОСПРОИЗВЕДЕНИЕ ПОЛОМКИ: без сглаживания член неограничен. Резкое q и
    # размазанное p по кодам вне его носителя — ровно то, что дало 225.6 на
    # настоящих данных.
    V_a = 256
    sharp = torch.full((2, 3, V_a), -40.0)
    sharp[..., 0] = 40.0                       # почти one-hot на коде 0
    spread = torch.zeros(2, 3, V_a)            # равномерное
    spread[..., 0] = -60.0                     # и нулевая масса на коде 0
    raw = bidirectional_alignment(sharp, spread, smoothing=0.0)
    # ПРИ eps = 0 ЭТО РОВНО log_softmax, А НЕ clamp_min(1e-30).
    # Проверяется тремя признаками: совпадение с log_softmax, рост потери
    # при удвоении разрыва логитов (clamp упёрся бы в 69.08) и КОНЕЧНЫЙ
    # НЕНУЛЕВОЙ градиент по тяжело ошибочному коду.
    q_ref = torch.softmax(sharp.float(), dim=-1)
    want_q2p = -(q_ref * torch.log_softmax(spread.float(),
                                           dim=-1)).sum(-1).mean()
    assert torch.allclose(raw["tokenizer_to_policy"], want_q2p, atol=0,
                          rtol=1e-6), (float(raw["tokenizer_to_policy"]),
                                       float(want_q2p))
    gap1 = torch.zeros(1, 1, V_a)
    gap1[..., 0] = -200.0
    gap2 = torch.zeros(1, 1, V_a)
    gap2[..., 0] = -400.0
    tgt0 = torch.full((1, 1, V_a), -60.0)
    tgt0[..., 0] = 60.0
    l1 = float(bidirectional_alignment(
        tgt0, gap1, smoothing=0.0)["tokenizer_to_policy"])
    l2 = float(bidirectional_alignment(
        tgt0, gap2, smoothing=0.0)["tokenizer_to_policy"])
    assert l2 > 1.8 * l1, (
        f"удвоение разрыва логитов дало {l1:.1f} -> {l2:.1f}: потеря во "
        f"что-то упирается, то есть это не log_softmax")
    assert l1 > -math.log(1e-30), (
        f"{l1:.2f} не превышает прежний предел clamp_min, проверка слепа")
    z_bad = gap1.clone().requires_grad_(True)
    g_bad = torch.autograd.grad(bidirectional_alignment(
        tgt0, z_bad, smoothing=0.0)["tokenizer_to_policy"], z_bad)[0]
    assert torch.isfinite(g_bad).all()
    assert abs(float(g_bad[0, 0, 0])) > 1e-6, (
        f"градиент по тяжело ошибочному коду равен {float(g_bad[0, 0, 0])}: "
        f"исправляющий сигнал обнулён")
    assert float(raw["policy_to_tokenizer"]) > 5.0 * math.log(V_a), (
        f"неограниченный член равен {float(raw['policy_to_tokenizer']):.1f}, "
        f"а проверка нужна ровно про его неограниченность")
    assert raw["bound"] == float("inf")
    # ПО УМОЛЧАНИЮ СГЛАЖИВАНИЯ НЕТ, и значение совпадает с неограниченным
    assert ALIGNMENT_SMOOTHING == 0.0
    default = bidirectional_alignment(sharp, spread)
    assert float(default["policy_to_tokenizer"]) == float(
        raw["policy_to_tokenizer"])
    # ЯВНОЕ СГЛАЖИВАНИЕ ОГРАНИЧИВАЕТ ЧЛЕН, И ЭТО ТОЖЕ ПРОВЕРЯЕТСЯ: опция
    # остаётся доступной, просто не включена.
    eps_t = 0.01
    bounded = bidirectional_alignment(sharp, spread, smoothing=eps_t)
    limit = math.log(V_a / eps_t)
    for key in ("tokenizer_to_policy", "policy_to_tokenizer"):
        assert float(bounded[key]) <= limit + 1e-4, (key, float(bounded[key]))
    assert abs(bounded["bound"] - limit) < 1e-9
    # И ПОДАВЛЯЕТ ГРАДИЕНТ НА САМЫХ ТЯЖЁЛЫХ НЕСОВПАДЕНИЯХ — ровно тот
    # побочный эффект, из-за которого оно выключено по умолчанию.
    z_hard = torch.full((1, 1, V_a), 0.0, requires_grad=True)
    tgt = torch.full((1, 1, V_a), -60.0)
    tgt[..., 1] = 60.0
    with torch.no_grad():
        pass
    g_plain = torch.autograd.grad(
        bidirectional_alignment(tgt, z_hard,
                                smoothing=0.0)["tokenizer_to_policy"],
        z_hard, retain_graph=False)[0].abs().max()
    z_hard2 = torch.zeros(1, 1, V_a, requires_grad=True)
    g_smooth = torch.autograd.grad(
        bidirectional_alignment(tgt, z_hard2,
                                smoothing=0.5)["tokenizer_to_policy"],
        z_hard2)[0].abs().max()
    assert float(g_smooth) < float(g_plain), (float(g_smooth),
                                              float(g_plain))
    # ГРАДИЕНТ ВСЁ ЕЩЁ ИДЁТ В ОБЕ СТОРОНЫ
    tl = torch.randn(2, 3, V_a, requires_grad=True)
    pl = torch.randn(2, 3, V_a, requires_grad=True)
    bidirectional_alignment(tl, pl)["total"].backward()
    assert tl.grad is not None and float(tl.grad.abs().sum()) > 0
    assert pl.grad is not None and float(pl.grad.abs().sum()) > 0
    for bad in (-0.1, 1.0, 1.5):
        try:
            bidirectional_alignment(sharp, spread, smoothing=bad)
        except ValueError as e:
            assert "smoothing" in str(e), e
        else:
            raise AssertionError(f"принято сглаживание {bad}")

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
