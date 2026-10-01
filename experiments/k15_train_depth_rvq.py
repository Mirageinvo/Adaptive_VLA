#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15: обучение depth-aligned остаточного токенизатора q0 -> q1 -> q2.

ОДИН ОСНОВНОЙ ЗАПУСК, БЕЗ СЕТКИ. Вопрос один: даёт ли совместно обучаемое
остаточное разбиение полезное последовательное уточнение за ОДИН проход VLA.
K-14 проверил другое — может ли h18 выбрать полезный элемент СТАРОЙ книги,
обученной на реконструкцию остатка после ИСТИННОГО k0.

ЧТО ЗАМОРОЖЕНО: backbone, все 24 слоя, голова q0 (`fast_head` и
`action_expert.norm`), книга C0, энкодер и декодер ActionCodec.
ЧТО ОБУЧАЕТСЯ: C1, C2, нормы и головы уровней 1-2, оба feedback-модуля.

ПЯТЬ ПУТЕЙ ДЕЙСТВИЯ, И КАЖДЫЙ НУЖЕН ДЛЯ РАЗНОГО ДИАГНОЗА:

    a0      = D(z0)                              черновик, опора
    a1_tok  = D(z0 + c1_Q)                       книга со стороны действия
    a1_pol  = D(z0 + c1_P)                       политика на h18
    a2_tok  = D(z0 + hard(C1[q1]) + c2_Q)        книга второго уровня
    a2_pol  = D(z0 + c1_P + c2_P)                ФАКТИЧЕСКИЙ путь вывода

    токенизатор улучшает, политика нет -> полезная книга есть, h18 её не
        различает;
    ни один не улучшает -> дело в геометрии остатка или декодере;
    q1 улучшает, q2 хуже -> второй уровень или feedback1 нестабилен.

ВЕСА КАНАЛОВ: как в МЕТРИКЕ K-14 (max_act_q, схват 1.0), а НЕ как в её
обучающей потере, где все каналы равны. Это осознанное отличие: K-14
измерил, что на вращательных каналах веса потери и метрики расходятся в
6-21 раз, то есть оптимизировалась не та величина, по которой судили. Здесь
оптимизируется та. Для сравнимости с историей K-14 равновесная величина всё
равно считается и печатается.

ДЕКОДИРОВАНИЕ В fp32 ВНЕ AUTOCAST — тот же контекст, который заверил гейт
K-15a. Иначе тренер считал бы действие, которого гейт не проверял; на этом
классе ошибки мы уже теряли день в K-14e.

КНИГИ — ОТДЕЛЬНАЯ ГРУППА ОПТИМИЗАТОРА С weight_decay = 0. AdamW с ненулевым
wd двигает параметр при НУЛЕВОМ градиенте, и книги поехали бы сами.
"""
import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import sys
import time

import numpy as np

H_EXEC = 8            # исполняемых позиций чанка, как в K-14
PATH_NAMES = ("a0", "a1_tok", "a1_pol", "a2_tok", "a2_pol")
# TOP-K ОРАКУЛ: сколько выигрыша книги достаётся, если разрешить выбрать
# лучший из k верхних кодов политики. Прямо отвечает на вопрос, который
# поставил smoke: политика ставит верный код на медианное место 3, а
# выигрыш при этом теряется почти весь. Диагностика, не гейт.
TOPK_ORACLE = (1, 2, 3, 4, 5, 10)
# ГЕЙТ K-15a: СКОЛЬКО БАТЧЕЙ И КАКИЕ ПРИЧИННЫЕ ПРОВЕРКИ ТРЕБУЮТСЯ.
# Требование живёт в тренере, а не берётся из проверяемого артефакта.
GATE_MIN_BATCHES = 3
# ГЕЙТЫ ПРИЁМКИ: ИМЯ -> КАТЕГОРИЯ, ЗАФИКСИРОВАННОЕ ОТОБРАЖЕНИЕ.
# `rollout_blocker` — кандидата нельзя выпускать на робота, а для
# decoder_support и align_not_broken под вопросом и само офлайновое число.
# `candidate` — артефакт исправен, а ответ эксперимента отрицательный.
# Провенанс, инварианты и нечисловые величины сюда НЕ входят: они
# останавливают прогон раньше и означают «выводов делать нельзя».
# ГЕЙТЫ ЗАДАНЫ ПО УРОВНЯМ: имя -> (категория, уровень). Уровень 1
# исполняется за 18 слоёв и не зависит от уровня 2 вообще, поэтому его
# пригодность считается по его собственным гейтам.
LEVEL_LAYERS = {1: 18, 2: 24}
EXPECTED_GATES = {}
for _lv in (1, 2):
    for _name in (f"collapse_q{_lv}_pol", f"collapse_q{_lv}_tok",
                  f"decoder_support_q{_lv}", f"action_range_q{_lv}",
                  f"align_not_broken_q{_lv}"):
        EXPECTED_GATES[_name] = ("rollout_blocker", _lv)
    EXPECTED_GATES[f"improves_q{_lv}"] = ("candidate", _lv)
del _lv, _name
GATE_CAUSAL_KINDS = ("feedback0_changes_q1", "feedback1_changes_q2",
                     "probe_old_vs_new_bounded")
EPS_NORM = 1e-6       # нижний предел нормировки на q0 MSE
# Веса функции потерь из плана K-15. Изменение любого — новый эксперимент.
W_A1_POL, W_A1_TOK, W_A2_POL, W_A2_TOK = 0.5, 0.5, 1.0, 0.5
W_ALIGN, W_MONO, W_USAGE = 0.25, 0.05, 0.01


def sha12(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:12]


def check_init_gate(path, *, expect, file_sha, code_version,
                    decoder_context, codec_fingerprints):
    """Гейт тождественности K-15a обязателен, как Gate R в K-14.

    Обучать архитектуру, про которую не доказано, что до обучения она
    побитово совпадает с прежней, значит сравнивать её с неизвестно чем.
    """
    if not path:
        raise SystemExit(
            "не указан артефакт гейта K-15a. Без доказательства, что при "
            "инициализации новый путь побитово равен прежнему, обучение "
            "сравнивать не с чем")
    if not os.path.exists(path):
        raise SystemExit(f"нет {path}: гейт K-15a не снят")
    g = json.load(open(path))
    if g.get("kind") != "k15_init_identity":
        raise SystemExit(f"{path} описывает {g.get('kind')}")
    if g.get("passed") is not True:
        raise SystemExit(f"гейт K-15a не пройден: {g.get('passed')!r}")
    if g.get("git_dirty"):
        raise SystemExit("гейт K-15a снят при незакоммиченном коде")
    # ПРИЧИННЫЕ ПРОВЕРКИ ОБЯЗАТЕЛЬНЫ. Гейт без них не отличает подключённый
    # feedback от неподключённого: при нулевой инициализации проверка
    # «выключение ничего не меняет» тавтологична.
    causal = g.get("causal_checks") or {}
    # СТРОГО `is not True`, А НЕ `bool(v)`. Строка "false" в Python истинна,
    # и артефакт с девятью значениями "false" проходил как пройденный.
    bad_values = sorted(k for k, v in causal.items() if v is not True)
    if not causal or bad_values:
        raise SystemExit(
            f"в гейте K-15a причинные проверки отсутствуют или их значения "
            f"не True: {bad_values[:4]}. Тождественность при нулевых весах "
            f"ничего не доказывает про подключение путей")
    # ЧИСЛО ПРИЧИННЫХ ПРОВЕРОК СВЕРЯЕТСЯ С ЧИСЛОМ БАТЧЕЙ. «Хотя бы одна»
    # проходило и в том случае, когда проверка отработала на первом батче и
    # молча пропала на остальных.
    n_batches = int(g.get("n_batches") or 0)
    if n_batches < GATE_MIN_BATCHES:
        raise SystemExit(
            f"в гейте K-15a батчей {n_batches}, минимум {GATE_MIN_BATCHES}. "
            f"Артефакт не должен сам назначать себе планку: сверка «по "
            f"числу батчей из артефакта» проходила и при одном батче")
    # ТОЧНОЕ МНОЖЕСТВО КЛЮЧЕЙ, А НЕ ИХ КОЛИЧЕСТВО. Подсчёт по суффиксам
    # проходил при постороннем ключе и при пропуске батча с номером в
    # середине.
    expected_causal = {f"batch{i}.{k}"
                       for i in range(n_batches) for k in GATE_CAUSAL_KINDS}
    if set(causal) != expected_causal:
        raise SystemExit(
            f"набор причинных проверок не тот: нет "
            f"{sorted(expected_causal - set(causal))[:4]}, посторонние "
            f"{sorted(set(causal) - expected_causal)[:4]}")
    # АРХИТЕКТУРНЫЕ ФАЙЛЫ СВЕРЯЮТСЯ ПОСОДЕРЖИМОМУ, А НЕ ПО git HEAD.
    # Правка тренера не обязана обесценивать гейт; правка архитектуры
    # обязана, и молча пройти не должна.
    if not isinstance(g.get("code_version"), dict):
        raise SystemExit("в гейте K-15a нет code_version")
    drift = [f"{k}: гейт {g['code_version'].get(k)}, сейчас "
             f"{code_version.get(k)}"
             for k in sorted(set(code_version) | set(g["code_version"]))
             if g["code_version"].get(k) != code_version.get(k)]
    if drift:
        raise SystemExit(
            "архитектура изменилась после гейта K-15a, его надо переснять: "
            + "; ".join(drift))
    # КОНТЕКСТ ДЕКОДИРОВАНИЯ. Гейт уже заверял действие в fp16 autocast,
    # пока тренер считал его в fp32 вне autocast, и обнаружить это было
    # нечем, кроме чтения обоих файлов.
    if g.get("decoder_context") != decoder_context:
        raise SystemExit(
            f"контекст декодера в гейте {g.get('decoder_context')!r} против "
            f"{decoder_context!r} в тренере: заверено не то действие")
    if g.get("codec") != codec_fingerprints:
        raise SystemExit(
            f"отпечатки кодека в гейте {g.get('codec')!r} против "
            f"{codec_fingerprints!r} сейчас: декодер не тот")
    bad = [f"{k}: гейт {g.get(k)}, сейчас {expect.get(k)}"
           for k in sorted(expect) if str(g.get(k)) != str(expect.get(k))]
    if bad:
        raise SystemExit("гейт K-15a снят в другой обстановке: "
                         + "; ".join(bad))
    return dict(init_gate=path, init_gate_sha1=file_sha(path),
                init_gate_run_id=g.get("run_id"),
                init_gate_n_batches=n_batches,
                init_gate_causal=len(causal),
                init_gate_code_version=dict(code_version),
                decoder_context=dict(decoder_context))


def classify_gates(gates):
    """Разбор гейтов по уровням. Чистая функция.

    Возвращает (failed, by_level), где by_level[уровень] — словарь с
    ключами failed, blocker, candidate.

    ЗАЧЕМ ОТДЕЛЬНО. Прежде категории фильтровались сравнением со строкой,
    и опечатка в `category` давала отказ, не попавший НИ В ОДИН список:
    `accepted=false`, оба списка пустые, код возврата 0. Здесь отображение
    имя->(категория, уровень) фиксировано, и любое расхождение с ним —
    отказ.
    """
    actual = {name: gate.get("category") for name, gate in gates.items()}
    want = {name: cat for name, (cat, _lv) in EXPECTED_GATES.items()}
    if actual != want:
        diff = sorted(set(actual) ^ set(want)) or [
            f"{k}: {actual[k]} вместо {want[k]}"
            for k in sorted(actual) if actual[k] != want[k]]
        raise SystemExit(
            f"набор гейтов приёмки или их категории не те: {diff[:5]}")
    for name, gate in gates.items():
        if not isinstance(gate.get("passed"), bool):
            raise SystemExit(
                f"гейт {name}: passed = {gate.get('passed')!r}, а обязан "
                f"быть bool")
    failed = sorted(k for k, v in gates.items() if not v["passed"])
    by_level = {}
    for lv in sorted({lv for _c, lv in EXPECTED_GATES.values()}):
        own = [k for k in failed if EXPECTED_GATES[k][1] == lv]
        by_level[lv] = dict(
            failed=own,
            blocker=[k for k in own
                     if EXPECTED_GATES[k][0] == "rollout_blocker"],
            candidate=[k for k in own
                       if EXPECTED_GATES[k][0] == "candidate"])
    classified = sorted(k for d in by_level.values() for k in d["failed"])
    if classified != failed:
        raise SystemExit("не все отказы отнесены к уровню")
    for lv, d in by_level.items():
        if sorted(d["blocker"] + d["candidate"]) != sorted(d["failed"]):
            raise SystemExit(f"уровень {lv}: отказы не классифицированы")
    return failed, by_level


def frozen_tensors(model, trainable_names):
    """Пары (имя, тензор) для всего ЗАМОРОЖЕННОГО: параметры И буферы.

    Буферы раньше не проверялись вовсе, а книга C0 — именно буфер.
    """
    for name, p in model.named_parameters():
        if name not in trainable_names:
            yield f"param:{name}", p
    for name, b in model.named_buffers():
        if b is not None:
            yield f"buffer:{name}", b


def frozen_invariant(model, torch, trainable_names):
    """ДЕШЁВЫЙ инвариант замороженного: версии, хранилища, формы.

    ПОЧЕМУ НЕ МОМЕНТЫ. Отпечаток по (сумма, сумма квадратов) инвариантен к
    ПЕРЕСТАНОВКЕ значений внутри тензора: [1,2,3,4] и [4,3,2,1] давали один
    и тот же отпечаток. Здесь вместо моментов берутся:

    * `_version` — счётчик изменений на месте: любая in-place запись его
      увеличивает, а именно так и двигал бы веса оптимизатор;
    * идентификатор хранилища — ловит ПОДМЕНУ тензора, при которой версия
      осталась бы нулевой;
    * форма и тип.

    Побитовое доказательство даёт `frozen_content_sha`; оно дорогое и
    считается дважды за прогон, а этот инвариант — на каждой эпохе.
    """
    import hashlib as _h
    acc = _h.sha1()
    n = 0
    for name, t_ in frozen_tensors(model, trainable_names):
        n += 1
        try:
            storage = t_.untyped_storage().data_ptr()
        except AttributeError:                    # torch < 2.0
            storage = t_.storage().data_ptr()
        acc.update(f"{name}|{tuple(t_.shape)}|{t_.dtype}|"
                   f"{t_._version}|{storage}".encode())
    return acc.hexdigest()[:12], n


def frozen_content_sha(model, torch, trainable_names):
    """ПОБИТОВЫЙ отпечаток содержимого всего замороженного.

    Единственная проверка, которая ловит любое изменение, включая
    перестановку. Читает все веса на хост, поэтому вызывается дважды за
    прогон: до обучения и после восстановления выбранной эпохи.
    """
    import hashlib as _h
    acc = _h.sha1()
    n, elems = 0, 0
    with torch.no_grad():
        for name, t_ in frozen_tensors(model, trainable_names):
            n += 1
            elems += int(t_.numel())
            acc.update(f"{name}|{tuple(t_.shape)}|{t_.dtype}".encode())
            # ЧЕРЕЗ view(uint8), А НЕ numpy(): на bfloat16 `numpy()`
            # падает с TypeError, и отпечаток замороженного нельзя было бы
            # снять вообще — проверено, падает.
            # reshape(-1) ОБЯЗАТЕЛЕН: `view(dtype)` при другом размере
            # элемента требует dim() > 0, а нульмерные тензоры в модели
            # есть — вентиль `alpha` у CodeFeedback именно такой.
            acc.update(t_.detach().cpu().contiguous().reshape(-1)
                       .view(torch.uint8).numpy().tobytes())
    return acc.hexdigest()[:12], n, elems


def optimizer_covers_exactly(opt, model, trainable_names):
    """Множество тензоров в оптимизаторе РОВНО равно белому списку.

    Сверка по id: имя в белом списке ничего не гарантирует, если в группу
    попал другой тензор с тем же содержимым.
    """
    in_opt = {id(p) for group in opt.param_groups for p in group["params"]}
    want = {id(p) for name, p in model.named_parameters()
            if name in trainable_names}
    extra = len(in_opt - want)
    missing = sorted(name for name, p in model.named_parameters()
                     if name in trainable_names and id(p) not in in_opt)
    return extra, missing


def no_alias_between(model, trainable_names):
    """Ни один обучаемый тензор не делит хранилище с замороженным."""
    def sid(t_):
        try:
            return t_.untyped_storage().data_ptr()
        except AttributeError:
            return t_.storage().data_ptr()

    frozen_ids = {}
    for name, t_ in frozen_tensors(model, trainable_names):
        frozen_ids.setdefault(sid(t_), name)
    clashes = []
    for name, p in model.named_parameters():
        if name in trainable_names and sid(p) in frozen_ids:
            clashes.append((name, frozen_ids[sid(p)]))
    return clashes


def weighted_row_error(predicted, target, weights, torch):
    """Построчная взвешенная MSE по первым H_EXEC позициям и семи каналам.

    Возвращает (по_строкам, среднее). Построчная нужна и для монотонного
    штрафа, и для сохранения метрик без повторного прогона.
    """
    if predicted.shape != target.shape:
        raise ValueError(f"формы {tuple(predicted.shape)} и "
                         f"{tuple(target.shape)}")
    if predicted.shape[1] < H_EXEC:
        raise ValueError(f"позиций {predicted.shape[1]} меньше {H_EXEC}")
    d = (predicted[:, :H_EXEC, :7] - target[:, :H_EXEC, :7]) * weights
    per_row = (d ** 2).mean(dim=(1, 2))
    return per_row, per_row.mean()


def build_losses(paths, target, weights, torch, tok, *,
                 w_align=W_ALIGN, w_mono=W_MONO, w_usage=W_USAGE,
                 vocab=2048):
    """Функция потерь K-15 целиком. Чистая: ни модели, ни данных внутри.

    `paths` — словарь пяти действий, четырёх наборов логитов и четырёх
    наборов вероятностей:
        a0, a1_tok, a1_pol, a2_tok, a2_pol,
        q1_tok_logits, q1_pol_logits, q2_tok_logits, q2_pol_logits,
        q1_tok_probs, q2_tok_probs, q1_pol_probs, q2_pol_probs.

    РЕГУЛЯРИЗАТОР ИСПОЛЬЗОВАНИЯ ПРИМЕНЯЕТСЯ К Q, как записано в плане:
    KL(mean Q1) + KL(mean Q2). Он защищает от схлопывания разбиения, а не
    читателя. Та же величина для P считается и возвращается, но в потерю
    НЕ входит: схлопывание P — отдельный диагноз, и лечится он не
    регуляризацией P, а тем, что читать стало нечего.

    НОРМИРОВКА НА ОШИБКУ ЧЕРНОВИКА. Все action- и monotonic-члены делятся на
    detached MSE пути a0 с нижним пределом: иначе кросс-энтропии масштаба
    log(V) ~ 7.6 поглотили бы реконструкцию, и коэффициенты перестали бы
    переноситься между каналами.
    """
    rows, means = {}, {}
    for name in ("a0", "a1_tok", "a1_pol", "a2_tok", "a2_pol"):
        rows[name], means[name] = weighted_row_error(
            paths[name], target, weights, torch)
    norm = means["a0"].detach().clamp_min(EPS_NORM)

    l_action = (W_A1_POL * means["a1_pol"] + W_A1_TOK * means["a1_tok"]
                + W_A2_POL * means["a2_pol"] + W_A2_TOK * means["a2_tok"]
                ) / norm

    align1 = tok.bidirectional_alignment(paths["q1_tok_logits"],
                                         paths["q1_pol_logits"])
    align2 = tok.bidirectional_alignment(paths["q2_tok_logits"],
                                         paths["q2_pol_logits"])
    log_v = math.log(float(vocab))
    # ДВА НАПРАВЛЕНИЯ — ДВА ОТДЕЛЬНЫХ СЛАГАЕМЫХ. Сумма та же, что прежняя
    # `w * (align1.total + align2.total) / log V`, потому что total =
    # 0.5 (q->p + p->q). Разделение нужно, чтобы норму градиента КАЖДОГО
    # направления можно было замерить ровно по той величине, которая стоит
    # в сумме, а не по пересобранной заново.
    l_align_q2p = w_align * 0.5 * (align1["tokenizer_to_policy"]
                                   + align2["tokenizer_to_policy"]) / log_v
    l_align_p2q = w_align * 0.5 * (align1["policy_to_tokenizer"]
                                   + align2["policy_to_tokenizer"]) / log_v
    l_align = l_align_q2p + l_align_p2q

    # МОНОТОННОСТЬ МЯГКАЯ И С МАЛЫМ ВЕСОМ. Imitation error — слабый прокси
    # поведения (§48.4, §50.4); большой вес толкал бы модель держаться ближе
    # к q0, то есть к измеренному нулю.
    mono1 = tok.monotonic_hinge(rows["a0"].detach(), rows["a1_pol"])
    mono2 = tok.monotonic_hinge(rows["a1_pol"].detach(), rows["a2_pol"])
    l_mono = w_mono * (mono1 + mono2) / norm

    l_usage = w_usage * (tok.usage_kl_to_uniform(paths["q1_tok_probs"])
                         + tok.usage_kl_to_uniform(paths["q2_tok_probs"])
                         ) / log_v
    with torch.no_grad():
        usage_pol = (tok.usage_kl_to_uniform(paths["q1_pol_probs"])
                     + tok.usage_kl_to_uniform(paths["q2_pol_probs"])) / log_v

    total = l_action + l_align + l_mono + l_usage
    parts = dict(
        total=total, action=l_action, align=l_align, mono=l_mono,
        usage=l_usage, usage_pol_diagnostic=usage_pol, norm=norm,
        align_q_to_p_term=l_align_q2p, align_p_to_q_term=l_align_p2q,
        align_bound=float(align1["bound"]),
        align_smoothing=float(align1["smoothing"]),
        align1_q_to_p=align1["tokenizer_to_policy"],
        align1_p_to_q=align1["policy_to_tokenizer"],
        align2_q_to_p=align2["tokenizer_to_policy"],
        align2_p_to_q=align2["policy_to_tokenizer"],
        mono1=mono1, mono2=mono2)
    return total, parts, rows, means


def decoder_support(latent, codec, torch, quantizers, nearest_code,
                    code_contribution):
    """Насколько полученный латент ушёл с многообразия сумм книг кодека.

    ЗАЧЕМ. C1 и C2 обучаются, декодер заморожен и обучался на суммах ИСХОДНЫХ
    книг. Успех по action RMS на латенте вне этого многообразия может быть
    экстраполяцией декодера и развалиться на роботе. Мера: переквантовать
    латент собственной жадной процедурой кодека и измерить остаток.
    """
    with torch.no_grad():
        residual = latent.float()
        total = torch.zeros_like(residual)
        for q in quantizers:
            codes = nearest_code(residual, q)
            contribution = code_contribution(q, codes)
            total = total + contribution
            residual = residual - contribution
        num = residual.norm(dim=-1)
        den = latent.float().norm(dim=-1).clamp_min(1e-12)
        rel = (num / den).flatten()
        # ПОСТРОЧНЫЕ ОТНОШЕНИЯ ВОЗВРАЩАЮТСЯ ЦЕЛИКОМ. Прежде каждый батч
        # отдавал свой p95, а вызывающий брал их среднее — это НЕ p95
        # выборки, и на длинном хвосте расходится с ним в разы.
        return dict(abs_residual=float(num.mean()),
                    rel_residual=float(rel.mean()),
                    n=int(rel.numel())), rel.cpu()


def forecast_runtime(elapsed_seconds, done_batches, total_batches, epochs):
    """Прогноз полного времени по первым батчам. Решение «три эпохи» должно
    приниматься с цифрой, а не на глаз."""
    if done_batches <= 0:
        raise ValueError("нет посчитанных батчей")
    per_batch = float(elapsed_seconds) / float(done_batches)
    per_epoch = per_batch * float(total_batches)
    return dict(per_batch_s=per_batch, per_epoch_h=per_epoch / 3600.0,
                total_h=per_epoch * float(epochs) / 3600.0,
                batches_per_epoch=int(total_batches), epochs=int(epochs))


def select_epoch(history, level=2):
    """Выбор эпохи ДЛЯ УКАЗАННОГО УРОВНЯ: минимум его action RMS на val_sel.

    Tie-break — меньшая доля строк, где этот уровень хуже q0. Эпоха 0 (без
    обучения) участвует, как в K-14.

    УРОВЕНЬ — ПАРАМЕТР, А НЕ 2. Прежде эпоха выбиралась только по
    `val_a2_pol_rms`, и при этом вердикт мог объявить кандидатом q1: тогда
    сохранялась эпоха, лучшая для НЕ ТОГО уровня, а лучшая эпоха q1 просто
    терялась. Уровень 1 — отдельная точка (18 слоёв вместо 24), и для
    статьи потенциально более интересная.
    """
    if not history:
        raise SystemExit("история пуста")
    if not isinstance(level, int) or level not in (1, 2):
        raise SystemExit(f"уровень {level!r} не бывает")
    key = f"val_a{int(level)}_pol_rms"
    tie = f"val_frac_worse_q{int(level)}"
    missing = [r["epoch"] for r in history if key not in r or tie not in r]
    if missing:
        raise SystemExit(
            f"в истории эпох {missing[:3]} нет полей {key}/{tie}")
    best = min(history, key=lambda r: (round(float(r[key]), 12),
                                       round(float(r[tie]), 12),
                                       int(r["epoch"])))
    return int(best["epoch"]), best


def selftest():
    import torch
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import depth_aligned_tokenizer as tok

    torch.manual_seed(0)
    B, P, C, V = 4, 16, 7, 32
    weights = torch.ones(C)
    target = torch.randn(B, P, C)

    # --- ПОСТРОЧНАЯ ОШИБКА: ИЗВЕСТНЫЙ ОТВЕТ ------------------------------
    pred = target.clone()
    pred[1, :H_EXEC, :] += 1.0
    rows, mean = weighted_row_error(pred, target, weights, torch)
    assert rows.shape == (B,)
    assert abs(float(rows[0])) < 1e-12 and abs(float(rows[1]) - 1.0) < 1e-6
    assert abs(float(mean) - 0.25) < 1e-6, float(mean)
    # ПОЗИЦИИ ПОСЛЕ H_EXEC НЕ УЧАСТВУЮТ: их робот не исполняет
    pred2 = target.clone()
    pred2[:, H_EXEC:, :] += 10.0
    _r, m2 = weighted_row_error(pred2, target, weights, torch)
    assert float(m2) == 0.0, float(m2)
    # ВЕСА КАНАЛОВ ДЕЙСТВУЮТ
    w2 = torch.ones(C)
    w2[0] = 2.0
    pred3 = target.clone()
    pred3[:, :H_EXEC, 0] += 1.0
    _r, m3 = weighted_row_error(pred3, target, w2, torch)
    assert abs(float(m3) - 4.0 / C) < 1e-6, float(m3)
    for bad, why in (((torch.randn(B, P, C), torch.randn(B, P, C + 1)),
                      "формы"),
                     ((torch.randn(B, 4, C), torch.randn(B, 4, C)),
                      "меньше")):
        try:
            weighted_row_error(bad[0], bad[1], weights, torch)
        except ValueError as e:
            assert why in str(e), (why, e)
        else:
            raise AssertionError(f"принят неверный вход: {why}")

    # --- ФУНКЦИЯ ПОТЕРЬ ---------------------------------------------------
    def make_paths(scale_a1=1.0, scale_a2=1.0):
        p = {}
        p["a0"] = target + 0.5
        p["a1_tok"] = target + 0.5 * scale_a1
        p["a1_pol"] = target + 0.5 * scale_a1
        p["a2_tok"] = target + 0.5 * scale_a2
        p["a2_pol"] = target + 0.5 * scale_a2
        for nm in ("q1_tok_logits", "q1_pol_logits",
                   "q2_tok_logits", "q2_pol_logits"):
            p[nm] = torch.randn(B, P, V, requires_grad=nm.endswith("logits"))
        for lv in (1, 2):
            p[f"q{lv}_tok_probs"] = torch.softmax(
                p[f"q{lv}_tok_logits"], dim=-1)
            p[f"q{lv}_pol_probs"] = torch.softmax(
                p[f"q{lv}_pol_logits"], dim=-1)
        return p

    paths = make_paths()
    total, parts, rows_, means_ = build_losses(
        paths, target, weights, torch, tok, vocab=V)
    assert set(rows_) == {"a0", "a1_tok", "a1_pol", "a2_tok", "a2_pol"}
    assert bool(torch.isfinite(total))
    # НОРМИРОВКА: при равных путям ошибках action-член равен сумме весов
    expected = W_A1_POL + W_A1_TOK + W_A2_POL + W_A2_TOK
    assert abs(float(parts["action"]) - expected) < 1e-5, float(parts["action"])
    # улучшение путей уменьшает action-член
    better = build_losses(make_paths(scale_a1=0.5, scale_a2=0.25),
                          target, weights, torch, tok, vocab=V)[1]
    assert float(better["action"]) < float(parts["action"])
    # МОНОТОННЫЙ ШТРАФ НУЛЕВОЙ, КОГДА УТОЧНЕНИЕ НЕ ХУЖЕ
    assert float(better["mono"]) == 0.0, float(better["mono"])
    worse = build_losses(make_paths(scale_a1=2.0, scale_a2=3.0),
                         target, weights, torch, tok, vocab=V)[1]
    assert float(worse["mono"]) > 0.0
    # ОБА НАПРАВЛЕНИЯ СОГЛАСОВАНИЯ ПРИСУТСТВУЮТ И НЕНУЛЕВЫ
    assert float(parts["align1_p_to_q"].detach()) > 0 \
        and float(parts["align1_q_to_p"].detach()) > 0
    # ГРАДИЕНТ ДОХОДИТ ДО ЛОГИТОВ ТОКЕНИЗАТОРА — иначе это снова K-14
    total.backward()
    assert paths["q1_tok_logits"].grad is not None \
        and float(paths["q1_tok_logits"].grad.abs().sum()) > 0, \
        "нет градиента по логитам токенизатора: P->Q не работает"
    assert paths["q2_pol_logits"].grad is not None \
        and float(paths["q2_pol_logits"].grad.abs().sum()) > 0
    # РЕГУЛЯРИЗАТОР ИСПОЛЬЗОВАНИЯ СИДИТ НА Q, А НЕ НА P. Проверяется
    # причинно: подмена Q-вероятностей на схлопнутые обязана изменить член,
    # подмена P-вероятностей — не обязана его менять вовсе.
    # ОДИН И ТОТ ЖЕ БАЗОВЫЙ СЛОВАРЬ: make_paths() каждый раз рисует новые
    # логиты, и сравнение двух его вызовов сравнивало бы разные входы.
    base_paths = make_paths()
    one_hot = torch.zeros(B, P, V)
    one_hot[..., 3] = 1.0
    base_usage = float(build_losses(dict(base_paths), target, weights, torch,
                                    tok, vocab=V)[1]["usage"].detach())
    collapsed = dict(base_paths)
    collapsed["q1_tok_probs"] = one_hot
    collapsed["q2_tok_probs"] = one_hot
    u_q = float(build_losses(collapsed, target, weights, torch, tok,
                             vocab=V)[1]["usage"].detach())
    assert u_q > base_usage * 1.5, (u_q, base_usage)
    only_p = dict(base_paths)
    only_p["q1_pol_probs"] = one_hot
    only_p["q2_pol_probs"] = one_hot
    out_p = build_losses(only_p, target, weights, torch, tok, vocab=V)[1]
    assert abs(float(out_p["usage"].detach()) - base_usage) < 1e-6, (
        "член использования реагирует на P: он снова не на разбиении")
    assert float(out_p["usage_pol_diagnostic"]) > 0
    # НИЖНИЙ ПРЕДЕЛ НОРМИРОВКИ: при нулевой ошибке черновика не делим на ноль
    zero = make_paths()
    for nm in ("a0", "a1_tok", "a1_pol", "a2_tok", "a2_pol"):
        zero[nm] = target.clone()
    t0, p0, _r, _m = build_losses(zero, target, weights, torch, tok, vocab=V)
    # НИЖНИЙ ПРЕДЕЛ СРАВНИВАЕТСЯ С ДОПУСКОМ: clamp_min идёт во float32, и
    # 1e-6 там представляется как 1.0000000117e-06. Требовать побитового
    # равенства с питоновским float64 значило бы требовать невозможного.
    assert bool(torch.isfinite(t0))
    assert abs(float(p0["norm"]) - EPS_NORM) < 1e-12, float(p0["norm"])

    # --- DECODER SUPPORT --------------------------------------------------
    class FakeQ:
        def __init__(self, book):
            self.book = book

    def fake_nearest(residual, q):
        d = ((residual.unsqueeze(-2) - q.book) ** 2).sum(-1)
        return d.argmin(-1)

    def fake_contribution(q, codes):
        return q.book[codes]

    book = torch.eye(4)[:, :3] * 1.0
    qs = [FakeQ(book)]
    # латент РОВНО из книги -> остаток нулевой
    exact = book[torch.tensor([[0, 1, 2]])]
    st, rows_st = decoder_support(exact, None, torch, qs, fake_nearest,
                                  fake_contribution)
    assert st["abs_residual"] < 1e-6 and st["rel_residual"] < 1e-6
    assert rows_st.numel() == st["n"] == 3
    # латент ВНЕ книги -> остаток заметный
    off = exact + 0.7
    st2, rows2 = decoder_support(off, None, torch, qs, fake_nearest,
                                 fake_contribution)
    assert st2["rel_residual"] > 0.1, st2
    # ПОСТРОЧНЫЕ ЗНАЧЕНИЯ СОГЛАСОВАНЫ СО СРЕДНИМ: иначе p95 считался бы не
    # по тем же числам, что среднее.
    assert abs(float(rows2.mean()) - st2["rel_residual"]) < 1e-6

    # --- ИНВАРИАНТ И ОТПЕЧАТОК ЗАМОРОЖЕННОГО ------------------------------
    import torch.nn as nn_
    toy = nn_.Module()
    toy.free = nn_.Linear(3, 3)
    toy.frozen = nn_.Linear(3, 3)
    toy.register_buffer("book", torch.arange(4.0))
    train_names = {"free.weight", "free.bias"}
    names_f = [n for n, _ in frozen_tensors(toy, train_names)]
    # БУФЕРЫ ВХОДЯТ В ПРОВЕРКУ: книга C0 — именно буфер, и раньше она в
    # отпечаток не попадала вовсе.
    assert names_f == ["param:frozen.weight", "param:frozen.bias",
                       "buffer:book"], names_f
    inv_a, n_a = frozen_invariant(toy, torch, train_names)
    sha_a, n_s, n_el = frozen_content_sha(toy, torch, train_names)
    # ОТПЕЧАТОК СНИМАЕТСЯ И С bfloat16: через numpy() он падал бы
    bf = nn_.Module()
    bf.w = nn_.Parameter(torch.ones(3, 4, dtype=torch.bfloat16))
    # НУЛЬМЕРНЫЙ ТЕНЗОР: `view(uint8)` на нём падает без reshape(-1)
    bf.register_buffer("gate", torch.tensor(1.0))
    bf.register_buffer("gate16", torch.tensor(1.0, dtype=torch.float16))
    sha_bf, n_bf, _e_bf = frozen_content_sha(bf, torch, set())
    assert n_bf == 3, n_bf
    with torch.no_grad():
        bf.w[0, 0] = 2.0
    assert frozen_content_sha(bf, torch, set())[0] != sha_bf
    with torch.no_grad():
        bf.w[0, 0] = 1.0
        bf.gate.fill_(2.0)                  # изменение СКАЛЯРА
    assert frozen_content_sha(bf, torch, set())[0] != sha_bf, \
        "изменение нульмерного буфера не замечено"
    assert frozen_invariant(bf, torch, set())[1] == 3
    assert n_a == n_s == 3 and n_el == 9 + 3 + 4, (n_a, n_s, n_el)
    with torch.no_grad():
        toy.free.weight += 1.0            # обучаемое: ничего не меняется
    assert frozen_invariant(toy, torch, train_names)[0] == inv_a
    assert frozen_content_sha(toy, torch, train_names)[0] == sha_a
    # ПЕРЕСТАНОВКА ЗНАЧЕНИЙ — то, что моменты не ловили вовсе: сумма и
    # сумма квадратов у [0,1,2,3] и [3,2,1,0] совпадают.
    with torch.no_grad():
        toy.book.copy_(torch.tensor([3.0, 2.0, 1.0, 0.0]))
    assert frozen_content_sha(toy, torch, train_names)[0] != sha_a, \
        "перестановка значений замороженного буфера не замечена"
    assert frozen_invariant(toy, torch, train_names)[0] != inv_a, \
        "запись на месте не увеличила _version"
    with torch.no_grad():
        toy.book.copy_(torch.arange(4.0))
    assert frozen_content_sha(toy, torch, train_names)[0] == sha_a
    # ПОДМЕНА ТЕНЗОРА: версия у нового нулевая, ловится по хранилищу
    inv_now = frozen_invariant(toy, torch, train_names)[0]
    toy.frozen.bias = nn_.Parameter(toy.frozen.bias.detach().clone())
    assert frozen_invariant(toy, torch, train_names)[0] != inv_now, \
        "подмена замороженного тензора не замечена"

    # --- ОПТИМИЗАТОР НАКРЫВАЕТ РОВНО БЕЛЫЙ СПИСОК -------------------------
    good_opt = torch.optim.SGD([toy.free.weight, toy.free.bias], lr=0.1)
    assert optimizer_covers_exactly(good_opt, toy, train_names) == (0, [])
    wide = torch.optim.SGD([toy.free.weight, toy.free.bias,
                            toy.frozen.weight], lr=0.1)
    assert optimizer_covers_exactly(wide, toy, train_names)[0] == 1
    narrow = torch.optim.SGD([toy.free.weight], lr=0.1)
    assert optimizer_covers_exactly(narrow, toy, train_names)[1] == \
        ["free.bias"]

    # --- НЕТ ОБЩЕГО ХРАНИЛИЩА С ЗАМОРОЖЕННЫМ ------------------------------
    assert no_alias_between(toy, train_names) == []
    shared = nn_.Module()
    base = torch.arange(6.0).reshape(2, 3)
    shared.frozen = nn_.Parameter(base)
    shared.free = nn_.Parameter(base)          # то же хранилище
    assert no_alias_between(shared, {"free"}) == [("free", "param:frozen")]

    # --- КЛАССИФИКАЦИЯ ГЕЙТОВ ПРИЁМКИ -------------------------------------
    def mk_gates(**fails):
        return {name: dict(category=cat, passed=name not in fails)
                for name, (cat, _lv) in EXPECTED_GATES.items()}

    f_all, by_lv = classify_gates(mk_gates())
    assert f_all == [] and all(not by_lv[lv]["failed"] for lv in (1, 2))
    # ОТКАЗ УРОВНЯ 2 НЕ КАСАЕТСЯ УРОВНЯ 1 — ровно то, из-за чего случайная
    # голова q2 блокировала исправный q1.
    f_all, by_lv = classify_gates(mk_gates(decoder_support_q2=1,
                                           collapse_q2_pol=1))
    assert by_lv[1]["failed"] == [] and by_lv[1]["blocker"] == []
    assert sorted(by_lv[2]["blocker"]) == ["collapse_q2_pol",
                                           "decoder_support_q2"]
    f_all, by_lv = classify_gates(mk_gates(improves_q1=1))
    assert by_lv[1]["candidate"] == ["improves_q1"]
    assert by_lv[1]["blocker"] == [] and by_lv[2]["failed"] == []
    f_all, by_lv = classify_gates(mk_gates(action_range_q1=1, improves_q2=1))
    assert by_lv[1]["blocker"] == ["action_range_q1"]
    assert by_lv[2]["candidate"] == ["improves_q2"]
    assert sorted(f_all) == ["action_range_q1", "improves_q2"]
    # ОПЕЧАТКА В КАТЕГОРИИ: раньше отказ не попадал ни в один список и
    # прогон завершался кодом 0
    typo = mk_gates(decoder_support_q2=1)
    typo["decoder_support_q2"]["category"] = "techncial"
    try:
        classify_gates(typo)
    except SystemExit as e:
        assert "категории не те" in str(e), e
    else:
        raise AssertionError("гейт с опечаткой в категории классифицирован")
    for broken, why in (
            ({k: v for k, v in mk_gates().items() if k != "improves_q1"},
             "категории не те"),
            (dict(mk_gates(), лишний=dict(category="candidate", passed=True)),
             "категории не те")):
        try:
            classify_gates(broken)
        except SystemExit as e:
            assert why in str(e), (why, e)
        else:
            raise AssertionError(f"принят неверный набор гейтов: {why}")
    non_bool = mk_gates()
    non_bool["align_not_broken_q1"]["passed"] = "false"
    try:
        classify_gates(non_bool)
    except SystemExit as e:
        assert "обязан быть bool" in str(e), e
    else:
        raise AssertionError('passed="false" принят')
    assert sum(1 for c, _lv in EXPECTED_GATES.values()
               if c == "candidate") == 2, EXPECTED_GATES
    assert len(EXPECTED_GATES) == 12, len(EXPECTED_GATES)
    assert set(LEVEL_LAYERS) == {1, 2} and LEVEL_LAYERS[1] == 18

    # --- ПРОГНОЗ ВРЕМЕНИ --------------------------------------------------
    f = forecast_runtime(100.0, 100, 16951, 3)
    assert abs(f["per_batch_s"] - 1.0) < 1e-9
    assert abs(f["per_epoch_h"] - 16951 / 3600.0) < 1e-9
    assert abs(f["total_h"] - 3 * 16951 / 3600.0) < 1e-9
    try:
        forecast_runtime(1.0, 0, 10, 1)
    except ValueError as e:
        assert "нет посчитанных" in str(e)
    else:
        raise AssertionError("принят нулевой счёт батчей")

    # --- ВЫБОР ЭПОХИ ------------------------------------------------------
    def row(ep_, r1, r2, w1=0.5, w2=0.5):
        return dict(epoch=ep_, val_a1_pol_rms=r1, val_a2_pol_rms=r2,
                    val_frac_worse_q1=w1, val_frac_worse_q2=w2)

    hist = [row(0, 0.30, 0.20, 0.9, 0.5), row(1, 0.25, 0.15, 0.8, 0.4),
            row(2, 0.28, 0.15, 0.7, 0.3), row(3, 0.22, 0.18, 0.6, 0.1)]
    ep, best = select_epoch(hist, level=2)
    assert ep == 2, ep                     # tie-break по доле ухудшений
    assert best["val_frac_worse_q2"] == 0.3
    # УРОВЕНЬ 1 ВЫБИРАЕТ ДРУГУЮ ЭПОХУ — ровно та потеря, из-за которой
    # лучшая эпоха q1 раньше пропадала.
    ep1, best1 = select_epoch(hist, level=1)
    assert ep1 == 3 and best1["val_a1_pol_rms"] == 0.22, (ep1, best1)
    assert select_epoch([row(0, 0.1, 0.1)], level=1)[0] == 0
    assert select_epoch([row(0, 0.1, 0.1)], level=2)[0] == 0
    for bad_level in (0, 3, "q1"):
        try:
            select_epoch(hist, level=bad_level)
        except SystemExit as e:
            assert "не бывает" in str(e), e
        else:
            raise AssertionError(f"принят уровень {bad_level}")
    try:
        select_epoch([dict(epoch=0, val_a2_pol_rms=0.1)], level=2)
    except SystemExit as e:
        assert "нет полей" in str(e), e
    else:
        raise AssertionError("принята история без полей уровня")

    # --- ГЕЙТ K-15a: КАЖДАЯ МУТАЦИЯ ОТВЕРГАЕТСЯ ---------------------------
    import tempfile
    expect = {"joint_sha1": "J", "q1_sha1": "Q", "device": "cuda:1"}
    cv = {"depth_aligned_joint12.py": "aaa", "bar.py": "bbb"}
    dctx = {"autocast": "disabled", "channels": 7}
    cfp = {"codebooks_sha1": "c1", "codec_state_sha1": "c2",
           "decoder_probe": "c3"}
    kinds = GATE_CAUSAL_KINDS
    NB = GATE_MIN_BATCHES
    good = dict(kind="k15_init_identity", passed=True, git_dirty=False,
                run_id="R", n_batches=NB,
                causal_checks={f"batch{i}.{k}": True
                               for i in range(NB) for k in kinds},
                code_version=cv, decoder_context=dctx, codec=cfp,
                **expect)
    fixed = dict(code_version=cv, decoder_context=dctx,
                 codec_fingerprints=cfp, file_sha=lambda _p: "SH")
    with tempfile.TemporaryDirectory() as td:
        def w(obj, nm="g.json"):
            q = os.path.join(td, nm)
            json.dump(obj, open(q, "w"))
            return q
        info = check_init_gate(w(good), expect=expect, **fixed)
        assert info["init_gate_causal"] == NB * len(kinds)
        assert info["init_gate_run_id"] == "R"
        assert info["init_gate_n_batches"] == NB
        assert info["decoder_context"] == dctx
        half = {f"batch0.{k}": True for k in kinds}
        # АРТЕФАКТ НЕ НАЗНАЧАЕТ СЕБЕ ПЛАНКУ САМ: один батч с полным набором
        # своих ключей раньше проходил.
        one = dict(good, n_batches=1,
                   causal_checks={f"batch0.{k}": True for k in kinds})
        try:
            check_init_gate(w(one, "one.json"), expect=expect, **fixed)
        except SystemExit as e:
            assert "минимум" in str(e), e
        else:
            raise AssertionError("гейт из одного батча принят")
        # ПОСТОРОННИЙ КЛЮЧ
        extra = dict(good, causal_checks=dict(good["causal_checks"],
                                              **{"batch9.чужое": True}))
        try:
            check_init_gate(w(extra, "ex.json"), expect=expect, **fixed)
        except SystemExit as e:
            assert "посторонние" in str(e), e
        else:
            raise AssertionError("гейт с посторонним ключом принят")
        # ПРОПУЩЕН БАТЧ В СЕРЕДИНЕ, А ОБЩЕЕ ЧИСЛО КЛЮЧЕЙ СОВПАДАЕТ
        gap = {f"batch{i}.{k}": True for i in (0, 2) for k in kinds}
        gap[f"batch7.{kinds[0]}"] = True
        gap[f"batch7.{kinds[1]}"] = True
        gap[f"batch7.{kinds[2]}"] = True
        try:
            check_init_gate(w(dict(good, causal_checks=gap), "gap.json"),
                            expect=expect, **fixed)
        except SystemExit as e:
            assert "не тот" in str(e), e
        else:
            raise AssertionError("гейт с пропущенным батчем принят")
        for mut, why in (
                ({"passed": False}, "не пройден"),
                ({"kind": "x"}, "описывает"),
                ({"git_dirty": True}, "незакоммиченном"),
                ({"causal_checks": {}}, "отсутствуют"),
                ({"causal_checks": dict(good["causal_checks"],
                                        **{"batch0.feedback0_changes_q1":
                                           False})}, "не True"),
                # СТРОКА "false" В PYTHON ИСТИННА: bool(v) её принимал
                ({"causal_checks": {k: "false" for k in
                                    good["causal_checks"]}}, "не True"),
                ({"causal_checks": {k: 1 for k in
                                    good["causal_checks"]}}, "не True"),
                ({"causal_checks": {k: None for k in
                                    good["causal_checks"]}}, "не True"),
                # проверка отработала на первом батче и пропала дальше
                ({"causal_checks": half}, "не тот"),
                # новый вид причинной проверки отсутствует целиком
                ({"causal_checks": {f"batch{i}.{k}": True for i in range(NB)
                                    for k in kinds[:2]}}, "не тот"),
                ({"n_batches": 0}, "минимум"),
                ({"code_version": dict(cv, **{"bar.py": "ДРУГОЙ"})},
                 "архитектура изменилась"),
                ({"code_version": {"bar.py": "bbb"}},
                 "архитектура изменилась"),
                ({"code_version": None}, "нет code_version"),
                ({"decoder_context": {"autocast": "enabled", "channels": 7}},
                 "контекст декодера"),
                ({"decoder_context": None}, "контекст декодера"),
                ({"codec": dict(cfp, decoder_probe="ИНОЙ")},
                 "отпечатки кодека"),
                ({"joint_sha1": "ДРУГОЙ"}, "другой обстановке")):
            try:
                check_init_gate(w(dict(good, **mut), "m.json"),
                                expect=expect, **fixed)
            except SystemExit as e:
                assert why in str(e), (why, e)
            else:
                raise AssertionError(f"гейт принят при {mut}")
        try:
            check_init_gate("", expect=expect, **fixed)
        except SystemExit as e:
            assert "не указан" in str(e)
        else:
            raise AssertionError("принят пустой путь к гейту")

    print("самопроверка k15_train_depth_rvq пройдена")


def main():
    ap = argparse.ArgumentParser(
        description="K-15: обучение depth-aligned RVQ q0->q1->q2")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--q0", default="data/k14d/q0_b8_e0.npz")
    ap.add_argument("--gate-r", default="reports/k14d/gate_r.json")
    ap.add_argument("--joint-ckpt", default="data/k9d_ep3.pt")
    ap.add_argument("--q1-init", default="data/k14c/q1_main_s0.pt",
                    help="чекпойнт K-14: инициализация G1 и Feedback0. Он же "
                         "создаёт осмысленный h18 с первого шага")
    ap.add_argument("--init-gate", default="reports/k15a/init_identity.json",
                    help="артефакт гейта K-15a; без него обучать нельзя")
    ap.add_argument("--ckpt",
                    default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--variant", default="main")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8,
                    help="ТОЛЬКО 8. Состав батча входит в определение "
                         "канонического q0: левый паддинг делает логиты "
                         "зависимыми от того, кто рядом")
    ap.add_argument("--accum", type=int, default=1,
                    help="накопление градиента поверх ЦЕЛЫХ канонических "
                         "микробатчей по 8 строк; границы батчей не меняются")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--grip-weight", type=float, default=1.0)
    ap.add_argument("--tau-tokenizer", default="auto",
                    help="температура мягкого апостериора токенизатора. "
                         "'auto' — калибровка по целевой медианной мягкой "
                         "perplexity; число — фиксированное значение. При "
                         "tau=1 апостериор почти равномерен, потому что "
                         "квадрат расстояния делится на D")
    ap.add_argument("--tau-target-perplexity", type=float, default=20.0,
                    help="цель калибровки. Выбрана ПОСЛЕ смоука, который показал почти равномерный апостериор, и зафиксирована до полного прогона — не «до данных»")
    ap.add_argument("--topk-oracle-batches", type=int, default=32,
                    help="на скольких батчах итоговой переоценки считать "
                         "оракул верхних кодов; 0 — не считать")
    ap.add_argument("--calib-batches", type=int, default=4,
                    help="канонических батчей обучения на калибровку tau")
    ap.add_argument("--tau-policy", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--forecast-batches", type=int, default=100)
    ap.add_argument("--smoke", action="store_true",
                    help="проверка связности: train и val_sel урезаются, "
                         "подтверждающая половина НЕ читается, решения не "
                         "принимаются")
    ap.add_argument("--limit", type=int, default=0,
                    help="целых канонических батчей на часть, только со "
                         "--smoke")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--summary", default="")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if int(a.batch) != 8:
        raise SystemExit(
            f"--batch {a.batch}: канонический батч равен 8, и уменьшать его "
            f"нельзя — состав батча входит в определение q0. Для экономии "
            f"памяти используйте --accum поверх целых микробатчей")
    if a.limit and not a.smoke:
        raise SystemExit("--limit допустим только со --smoke")
    if a.smoke and not a.limit:
        # SMOKE БЕЗ --limit НИЧЕГО НЕ УРЕЗАЛ, то есть был полным прогоном
        # под именем проверки связности.
        a.limit = 2
        print("  --smoke без --limit: беру 2 канонических батча на часть")
    if a.seed is None:
        raise SystemExit("--seed задаётся явно, умолчания у него нет")
    if float(a.tau_policy) != 1.0:
        # ПРИ tau_policy != 1 ДВА МЕСТА РАСХОДЯТСЯ: жёсткий выбор политики
        # использует температуру, а выравнивание получает СЫРЫЕ логиты
        # политики, то есть работает при tau = 1. Это было бы молча другим
        # экспериментом, поэтому запрещено до согласования распределений.
        raise SystemExit(
            f"--tau-policy {a.tau_policy}: выравнивание считается по сырым "
            f"логитам политики, то есть при tau=1, а её жёсткий выбор — по "
            f"масштабированным. Пока эти два места не согласованы, "
            f"допустимо только 1.0")
    out_path = a.out or (f"data/k15/{'smoke' if a.smoke else 'depth_rvq'}"
                         f"_s{a.seed}.pt")
    if os.path.exists(out_path) and not a.overwrite:
        raise SystemExit(f"{out_path} уже существует: чекпойнт не "
                         f"перезаписывается молча")

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(a.root)
    for p in (root, os.path.join(root, "src"),
              os.path.join(root, "experiments"),
              os.path.abspath("third_party/actioncodec"), here):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    import k14_common as kc
    import k11a_build_hicora_cache as k11a
    import k11b_hicora_identity as k11b
    import k14c_train_q1 as k14c
    import k9h_multiarm_gate as k9h
    import depth_aligned_tokenizer as tok
    from depth_aligned_joint12 import (architecture_code_version,
                                       make_action_decoder,
                                       make_depth_aligned_joint12_class)
    from depth_rvq_joint12 import code_contribution, nearest_code
    from joint12_vla import make_joint12_class
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       VisionLanguageActionProcessor, dict_apply, get_cfg,
                       prompt_template)
    import actioncodec  # noqa: F401  регистрация типа модели в AutoConfig

    git_head, dirty, _arte = kc.check_code_clean(a.allow_dirty)
    print(f"  код: коммит {git_head}"
          + ("  (--allow-dirty)" if dirty else ""))
    dev = torch.device(a.device)
    dt = getattr(torch, a.dtype)
    torch.manual_seed(int(a.seed))
    np.random.seed(int(a.seed))

    # --- ДАННЫЕ, ПУТЬ K-14c БЕЗ ИЗМЕНЕНИЙ --------------------------------
    meta = json.load(open(f"{a.cache}.meta.json"))
    src = meta["cache"]
    d = np.load(src, allow_pickle=True)
    cmeta = json.loads(str(d["meta"]))
    N = int(meta["n_obs"])
    epi = np.asarray(d["episode"])[:N].astype(np.int64)
    stp = np.asarray(d["step"])[:N]
    keys_sha = hashlib.sha1(np.ascontiguousarray(
        np.stack([epi, stp])).tobytes()).hexdigest()[:12]
    if keys_sha != meta.get("keys_sha1"):
        raise SystemExit(f"ключи наблюдений {keys_sha} против "
                         f"{meta.get('keys_sha1')}")
    ACT = np.asarray(d["action"])[:N]
    offs = np.asarray(d["pos_offset"])[:N].astype(np.int64)
    tsk = np.asarray(d["task"])[:N]
    E = np.load(f"{a.cache}.codebooks.npy")
    IMG = np.load(os.path.join(os.path.dirname(src),
                               cmeta["images_file"]), mmap_mode="r")
    ds_repo, ds_rev = k11b.dataset_source(meta)
    st_n, _sm, _sh = kc.load_states(src, N, ds_repo, ds_rev, keys_sha,
                                    STATE_Q01, STATE_Q99)
    max_act_q = np.maximum(np.abs(ACTION_Q01), np.abs(ACTION_Q99))
    # ЭТАЛОН ДИАПАЗОНА ДЕЙСТВИЙ — В ТЕХ ЖЕ ЕДИНИЦАХ, ЧТО ВЫХОД ДЕКОДЕРА.
    # Кэш хранит действие НОРМИРОВАННЫМ: в k9a оно поделено на max_act_q,
    # у схвата перевёрнут знак, и всё обрезано в [-1, 1]. Декодер
    # возвращает его же. Поэтому эталоном служит p99 |действия| ПО ВСЕМУ
    # НАБОРУ в этих же единицах и на тех же исполняемых позициях, а не
    # физический max_act_q и не максимум по оценочным строкам.
    act_p99_dataset = np.percentile(
        np.abs(np.asarray(ACT[:, :H_EXEC, :7], np.float64)).reshape(-1, 7),
        99.0, axis=0)
    print("  эталон диапазона (p99 |действия| по набору, нормированные "
          "единицы): "
          + ", ".join(f"{x:.3f}" for x in act_p99_dataset))

    q0_can, _q0_def, q0_man, q0_prov = kc.load_canonical_q0(
        a.q0, gate_r_path=a.gate_r, n_obs=N, keys_sha=keys_sha,
        cache_meta_sha1=k11a.file_sha1(f"{a.cache}.meta.json"))
    plan = kc.load_plan(a.q0, q0_man)
    parts = {}
    for name, po, sel in plan:
        parts.setdefault(name, []).append((po, sel))
    if a.limit:
        parts = {k: v[:int(a.limit)] for k, v in parts.items()}
    # СМЕЩЕНИЯ ПЛАНА СВЕРЯЮТСЯ СО СТРОКАМИ. Батч, заявленный с одним
    # смещением, а собранный из строк с другим, дал бы другой q0 — и это
    # обнаружилось бы только побитовой сверкой, уже потратив проход.
    for name, po, sel in plan:
        if not bool((offs[sel] == po).all()):
            raise SystemExit(f"батч части {name} заявлен со смещением {po}, "
                             f"а строки имеют другое")
    if "train" not in parts or "val_sel" not in parts:
        raise SystemExit(f"в плане нет нужных частей: {sorted(parts)}")
    # ПОДТВЕРЖДАЮЩАЯ ПОЛОВИНА НЕ ФОРМИРУЕТСЯ ВОВСЕ. В K-14 она уже
    # прочитана, поэтому по §53 любое её использование здесь было бы
    # retrospective; открывать её тренером нельзя ни при каком исходе.
    parts.pop("val_confirm", None)
    n_train, n_val = len(parts["train"]), len(parts["val_sel"])
    print(f"  план {q0_prov['plan_sha1']}: train {n_train} батчей, "
          f"val_sel {n_val} батчей, батч {a.batch}")

    # --- МОДЕЛЬ -----------------------------------------------------------
    cfg = get_cfg(os.path.join(root, a.cfg_path))
    cfg.TRAINING.ckpt_dir = a.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = a.ckpt
    Base = make_depth_aligned_joint12_class(
        make_joint12_class(SmolVLABlockwiseAR))
    model = Base.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    model.init_joint_fast(depth=int(meta["depth"]), head_dtype=torch.float32)
    refine_norm = copy.deepcopy(model.action_expert.norm)
    joint_sha = k11a.file_sha1(a.joint_ckpt)
    kc.load_joint12_strict(model, a.joint_ckpt, int(meta["depth"]),
                           (meta.get("source") or {}).get("weights_sha1"),
                           torch, dev)
    model.init_depth_aligned_rvq(refine_norm=refine_norm,
                                 books=torch.from_numpy(E),
                                 head_dtype=torch.float32,
                                 verbose_init=False)

    # ИНИЦИАЛИЗАЦИЯ G1 И FEEDBACK0 ИЗ K-14. Он уже создаёт осмысленный h18,
    # и стартовать с него дешевле, чем учить с нуля. Загрузка строгая:
    # белый список и отпечаток весов сверяются, иначе обучение шло бы от
    # неизвестного состояния.
    legacy_info = model.configure_joint_depth_rvq(
        stage="q1", variant=a.variant, verbose=False)
    q1_obj = torch.load(a.q1_init, map_location="cpu", weights_only=False)
    q1_prov = k9h.check_depthrvq_q1_ckpt(
        q1_obj, joint_sha1=joint_sha, expect_variant=a.variant,
        expect_q0_manifest_sha1=q0_prov["q0_manifest_sha1"])
    want = set(legacy_info["names"])
    st = q1_obj["state"]
    if set(q1_obj["trainable_names"]) != want or set(st) != want:
        raise SystemExit(
            f"белый список K-14 не совпал: нет {sorted(want - set(st))[:5]}, "
            f"лишние {sorted(set(st) - want)[:5]}")
    own = dict(model.named_parameters())
    with torch.no_grad():
        for k_, v_ in st.items():
            if tuple(own[k_].shape) != tuple(v_.shape):
                raise SystemExit(f"форма {k_}")
            if not torch.isfinite(v_).all():
                raise SystemExit(f"в {k_} есть nan или inf")
            own[k_].data.copy_(v_.to(own[k_].device, own[k_].dtype))
    loaded_sha = k14c.state_sha({k_: own[k_].detach().float().cpu().numpy()
                                 for k_ in want})
    if loaded_sha != str(q1_obj["selected_state_sha1"]):
        raise SystemExit(f"после загрузки K-14 отпечаток {loaded_sha}, в "
                         f"чекпойнте {q1_obj['selected_state_sha1']}")
    info = model.configure_depth_aligned_rvq(verbose=True)
    status = model.initialization_status()
    for key in ("c1_exact", "c2_exact", "books_not_aliased"):
        if not status.get(key):
            raise SystemExit(f"инициализация нарушена: {key} = "
                             f"{status.get(key)}")

    ac_ = proc.action_processor
    codec = ac_ if hasattr(ac_, "vq") else getattr(ac_, "codec", None)
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь")
    codec = codec.to(dev).eval()
    for p_ in codec.parameters():
        p_.requires_grad_(False)
    quantizers = list(codec.vq.quantizers)
    vocab = int(model.depth_aligned_book(1).shape[0])

    # --- КОДЕК СВЕРЯЕТСЯ ТАК ЖЕ, КАК В K-14c -----------------------------
    # Тренер декодирует действия этим кодеком, и все цели K-14 построены
    # его книгами. Другой кодек дал бы другие числа при полном совпадении
    # всего остального; в K-14c эта сверка уже есть, здесь её не было.
    with torch.no_grad():
        idx_all = torch.arange(int(codec.vocab_size), device=dev)[None, :]
        codec_books = torch.stack([
            q_.out_project(q_.decode_code(idx_all))[0]
            for q_ in quantizers]).float()
    if tuple(codec_books.shape) != tuple(np.asarray(E).shape):
        raise SystemExit(f"книги кодека {tuple(codec_books.shape)} против "
                         f"кэша {tuple(np.asarray(E).shape)}")
    book_gap = float((codec_books.cpu()
                      - torch.from_numpy(np.asarray(E, np.float32))
                      ).abs().max())
    if book_gap > 1e-5:
        raise SystemExit(f"книги кодека разошлись с кэшем на {book_gap:.3e}")
    codec_fp = {
        "codebooks_sha1": hashlib.sha1(np.ascontiguousarray(
            np.asarray(E, np.float32)).tobytes()).hexdigest()[:12],
        "codec_state_sha1": k11a.state_sha1(codec),
        "decoder_probe": k11a.decoder_probe(codec, codec_books, dev),
    }
    k11a.check_fingerprints(meta, codec_fp)
    print(f"  кодек сверен: книги {codec_fp['codebooks_sha1']}, веса "
          f"{codec_fp['codec_state_sha1']}, проба "
          f"{codec_fp['decoder_probe']}")

    # ДЕКОДЕР ОБЩИЙ С ГЕЙТОМ: одна функция, один контекст.
    decode_fp32, decoder_context = make_action_decoder(codec, dev.type)
    code_version = architecture_code_version(
        here, inspect.getfile(SmolVLABlockwiseAR), sha12)

    # --- ГЕЙТ K-15a ОБЯЗАТЕЛЕН -------------------------------------------
    gate_info = check_init_gate(
        a.init_gate,
        expect=dict(joint_sha1=joint_sha,
                    q1_sha1=k11a.file_sha1(a.q1_init),
                    plan_sha1=q0_prov["plan_sha1"],
                    compute_dtype=a.dtype,
                    device=str(dev)),
        file_sha=k11a.file_sha1,
        code_version=code_version,
        decoder_context=decoder_context,
        codec_fingerprints=codec_fp)
    print(f"  гейт K-15a: {gate_info['init_gate']}, запуск "
          f"{gate_info['init_gate_run_id']}, причинных проверок "
          f"{gate_info['init_gate_causal']}")

    # --- ОПТИМИЗАТОР: КНИГИ ОТДЕЛЬНОЙ ГРУППОЙ С wd = 0 -------------------
    book_names = {"depth_aligned_c1", "depth_aligned_c2"}
    books_group, rest_group = [], []
    for name, p_ in model.named_parameters():
        if not p_.requires_grad:
            continue
        (books_group if name in book_names else rest_group).append(p_)
    if len(books_group) != 2:
        raise SystemExit(f"книг в оптимизаторе {len(books_group)}, ожидалось 2")
    opt = torch.optim.AdamW(
        [dict(params=books_group, weight_decay=0.0),
         dict(params=rest_group, weight_decay=float(a.wd))],
        lr=float(a.lr))
    print(f"  оптимизатор: книги {sum(p_.numel() for p_ in books_group)} "
          f"параметров с wd=0, остальное "
          f"{sum(p_.numel() for p_ in rest_group)} с wd={a.wd}")

    weights = torch.ones(7, device=dev, dtype=torch.float32)
    weights_gate = torch.as_tensor(max_act_q[:7], device=dev,
                                   dtype=torch.float32).clone()
    weights_gate[-1] = float(a.grip_weight)

    def build_batch(po, sel):
        """Дословно как в K-14h/K-14o: свой вариант уже оказывался неверным."""
        image = torch.from_numpy(np.asarray(IMG[sel]))
        msgs = []
        for gi in sel:
            mm = prompt_template(
                st_n[gi], None, str(tsk[gi]),
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                action_token_len=cfg.MODEL.action_processor.token_len)
            mm[1]["content"] = mm[1]["content"][1:]
            msgs.append(mm)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        b = proc(text=texts, images=[[image[k].numpy()]
                                     for k in range(len(sel))],
                 return_tensors="pt", padding=True, padding_side="left",
                 action_processor_kwargs={"embodiment_ids": 0})
        return dict_apply(lambda x: x.to(dev, dt), b)

    ac16 = torch.autocast(device_type=dev.type, dtype=dt)
    run_batch_checked = [False]
    decode_batched = [None, 0.0]
    # ТЕМПЕРАТУРА ТОКЕНИЗАТОРА ПО УРОВНЯМ. Заполняется калибровкой ниже, до
    # первой эпохи; run_batch читает её по ссылке.
    tau_tok = {1: 1.0, 2: 1.0}
    tau_report = {}
    collect_dist = [False]
    collect_topk = [0]
    collect_components = [False]
    q0_can_dev = torch.as_tensor(np.asarray(q0_can), device=dev)
    pending_q0 = [None]
    pending_bad_action = [None]
    action_shape_checked = [False]

    def check_pending():
        """Чтение отложенных счётчиков: q0 и конечность пяти действий."""
        if pending_q0[0] is not None:
            bad = int(pending_q0[0])
            pending_q0[0] = None
            if bad:
                raise SystemExit(
                    f"q0 разошёлся с каноническим в {bad} позициях: "
                    f"обучение относилось бы к другому черновику")
        if pending_bad_action[0] is not None:
            bad = int(pending_bad_action[0])
            pending_bad_action[0] = None
            if bad:
                raise SystemExit(
                    f"в декодированных действиях {bad} нечисловых значений: "
                    f"дальше считалась бы потеря по nan")

    def run_batch(po, sel, train):
        # флаг однократной проверки; список, чтобы не объявлять nonlocal
        """Один канонический микробатч: пять путей и все метрики."""
        b = build_batch(po, sel)
        am = b.get("attention_mask")
        with ac16:
            v, p_ids = model.build_inputs(position_offset=po, **b)
            out = model.forward_depth_aligned_rvq(
                vlm_inputs_embeds=v, attention_mask=am, position_ids=p_ids,
                mode="full", tau=float(a.tau_policy))
        # q0 СВЕРЯЕТСЯ НА КАЖДОМ БАТЧЕ, НО НА УСТРОЙСТВЕ. Сравнение идёт
        # каждый батч; на хост счётчик читается в оценке сразу, а в
        # обучении — раз в несколько сотен батчей, потому что чтение
        # одного int с GPU останавливает конвейер.
        q0_bad = (out["pred_codes"][0].detach()
                  != q0_can_dev[torch.as_tensor(sel, device=dev)]).sum()
        pending_q0[0] = q0_bad if pending_q0[0] is None \
            else pending_q0[0] + q0_bad
        c1 = model.depth_aligned_book(1)
        c2 = model.depth_aligned_book(2)
        z0 = out["policy_embeddings"][0]
        c1_pol = out["policy_embeddings"][1]
        c2_pol = out["policy_embeddings"][2]
        action = torch.from_numpy(
            np.asarray(ACT[sel], np.float32)).to(dev)[..., :7]
        with torch.no_grad():
            z_e = codec._encode(action.float(), embodiment_ids=0).float()
        r1 = z_e - z0.detach()
        # ТЕМПЕРАТУРА ПРИМЕНЯЕТСЯ ОДИН РАЗ. Композиция tokenizer_logits и
        # hard_straight_through применяла её дважды, и probabilities
        # расходились с логитами, по которым считается выравнивание.
        q1_tok_logits, c1_tok, i1_tok, p1_tok = tok.quantize_residual(
            r1, c1, temperature=tau_tok[1])
        # ПРЕФИКС ДЛЯ ВТОРОГО УРОВНЯ — ФАКТИЧЕСКИЙ q1 МОДЕЛИ, И ОН
        # DETACH-НУТ: иначе токенизатор q2 уменьшал бы собственную задачу,
        # двигая C1. C1 всё равно получает градиент от q1-путей.
        hard_c1 = c1[out["pred_codes"][1]].detach()
        r2 = z_e - z0.detach() - hard_c1
        q2_tok_logits, c2_tok, i2_tok, p2_tok = tok.quantize_residual(
            r2, c2, temperature=tau_tok[2])
        # ФАКТИЧЕСКИЙ ПУТЬ ВЫВОДА БЕРЁТСЯ ИЗ МОДЕЛИ, А НЕ ПЕРЕСОБИРАЕТСЯ:
        # пересборка была бы второй реализацией той же суммы. Один раз
        # проверяем, что она совпадает с z0 + c1_pol + c2_pol.
        if not run_batch_checked[0]:
            run_batch_checked[0] = True
            rebuilt = z0 + c1_pol + c2_pol
            if not torch.equal(rebuilt, out["cumulative_latents"][2]):
                raise SystemExit(
                    "накопленный латент модели не равен сумме её же "
                    "эмбеддингов: путь вывода собран по-разному")
        # ОДИН ВЫЗОВ ДЕКОДЕРА НА ПЯТЬ ПУТЕЙ. Пять отдельных вызовов на
        # батч из восьми строк — это пять запусков ядер ради 40 строк.
        # Склейка по батчу законна только если декодер построчный, и это
        # ПРОВЕРЯЕТСЯ один раз, а не предполагается: подбор ядра cuBLAS
        # зависит от размера батча, поэтому побитового равенства может и не
        # быть. Если его нет — работаем по одному и пишем это в артефакт.
        lat5 = [z0, z0 + c1_tok, z0 + c1_pol, z0 + hard_c1 + c2_tok,
                out["cumulative_latents"][2]]
        names5 = ("a0", "a1_tok", "a1_pol", "a2_tok", "a2_pol")
        if decode_batched[0] is None:
            with torch.no_grad():
                joined = decode_fp32(torch.cat([x.detach() for x in lat5], 0))
                apart = torch.cat([decode_fp32(x.detach()) for x in lat5], 0)
                same = bool(torch.equal(joined, apart))
                gap = 0.0 if same else float((joined - apart).abs().max())
            decode_batched[0] = same
            decode_batched[1] = gap
            print(f"  декодер построчный: склейка по батчу "
                  f"{'побитово совпала' if same else 'РАСХОДИТСЯ'}"
                  + ("" if same else f" на {gap:.3e}, считаем по одному"))
        if decode_batched[0]:
            dec = decode_fp32(torch.cat(lat5, 0))
            acts = dict(zip(names5, dec.chunk(len(lat5), dim=0)))
        else:
            acts = {n_: decode_fp32(x) for n_, x in zip(names5, lat5)}
        # ФОРМА И ТИП ПЯТИ ДЕЙСТВИЙ ПРОВЕРЯЮТСЯ ОДИН РАЗ, КОНЕЧНОСТЬ —
        # КАЖДЫЙ БАТЧ. Разошедшаяся форма — ошибка сборки и появится на
        # первом же батче; nan появляется в середине обучения.
        if not action_shape_checked[0]:
            action_shape_checked[0] = True
            want = (int(len(sel)), int(action.shape[1]), 7)
            for n_, x_ in acts.items():
                if tuple(x_.shape) != want:
                    raise SystemExit(
                        f"путь {n_} имеет форму {tuple(x_.shape)}, "
                        f"ожидалось {want}")
                if x_.dtype != torch.float32:
                    raise SystemExit(f"путь {n_} имеет тип {x_.dtype}, "
                                     f"ожидался float32")
            print(f"  пять путей: форма {want}, тип float32")
        bad_act = sum((~torch.isfinite(x_)).sum() for x_ in acts.values())
        pending_bad_action[0] = bad_act if pending_bad_action[0] is None \
            else pending_bad_action[0] + bad_act
        paths = dict(
            q1_tok_logits=q1_tok_logits,
            q1_pol_logits=out["logits"][1].float(),
            q2_tok_logits=q2_tok_logits,
            q2_pol_logits=out["logits"][2].float(),
            # РЕГУЛЯРИЗАТОР ИСПОЛЬЗОВАНИЯ ИДЁТ НА Q, КАК В ПЛАНЕ. Он нужен
            # против схлопывания САМОГО РАЗБИЕНИЯ; на P он удерживал бы от
            # схлопывания только читателя, а книга могла бы выродиться.
            # Схлопывание P наблюдается отдельно, по usage_q1_pol.
            q1_tok_probs=p1_tok, q2_tok_probs=p2_tok,
            q1_pol_probs=out["policy_probabilities"][1],
            q2_pol_probs=out["policy_probabilities"][2],
            **acts)
        total, lparts, rows, means = build_losses(
            paths, action, weights_gate, torch, tok, vocab=vocab)
        with torch.no_grad():
            # В ОБУЧЕНИИ НИ ОДНОГО float(): каждый такой вызов — ожидание
            # GPU. Числа отдаются тензорами, вызывающий складывает их на
            # устройстве и читает раз в несколько сотен батчей.
            track = dict(loss=total.detach(), action=lparts["action"].detach(),
                         align=lparts["align"].detach(),
                         mono=lparts["mono"].detach(),
                         usage=lparts["usage"].detach())
            for k_ in PATH_NAMES:
                track[k_] = means[k_].detach()
            stat = dict(rows=int(len(sel)), track=track)
        if collect_components[0]:
            # ТЕНЗОРЫ С ГРАФОМ, не detach: по ним берутся нормы градиентов
            # по компонентам в смоуке.
            stat["components"] = {
                "action": lparts["action"],
                "align_q_to_p": lparts["align_q_to_p_term"],
                "align_p_to_q": lparts["align_p_to_q_term"],
                "mono": lparts["mono"], "usage": lparts["usage"]}
        with torch.no_grad():
            if not train:
                stat.update(
                    {k_: float(means[k_]) for k_ in PATH_NAMES},
                    loss=float(total), loss_action=float(lparts["action"]),
                    loss_align=float(lparts["align"]),
                    loss_mono=float(lparts["mono"]),
                    loss_usage=float(lparts["usage"]))
                # ПОСТРОЧНЫЕ МАССИВЫ И КОПИИ НА ХОСТ — ТОЛЬКО В ОЦЕНКЕ.
                # В обучении они считались каждый батч и выбрасывались:
                # синхронизации с GPU ради чисел, которые никто не читал.
                _r_eq, eq = weighted_row_error(paths["a2_pol"], action,
                                               weights, torch)
                stat.update(
                    a2_pol_equal_weights=float(eq),
                    frac_worse_q1=float(
                        (rows["a1_pol"] > rows["a0"]).float().mean()),
                    frac_worse_q2=float(
                        (rows["a2_pol"] > rows["a0"]).float().mean()),
                    # ВЕЛИЧИНА ПОПРАВКИ, А НЕ «НЕНУЛЕВАЯ ЛИ СТРОКА КНИГИ».
                    # Прежняя доля строк с ненулевой нормой c1 равнялась
                    # единице по построению: почти все строки книги
                    # ненулевые, и число ничего не сообщало.
                    d_latent_q1=float((c1_pol.float().norm(dim=-1)
                                       / z0.float().norm(dim=-1)
                                       .clamp_min(1e-12)).mean()),
                    d_latent_q2=float((c2_pol.float().norm(dim=-1)
                                       / z0.float().norm(dim=-1)
                                       .clamp_min(1e-12)).mean()),
                    d_action_q1=float((acts["a1_pol"] - acts["a0"])
                                      .abs().max()),
                    d_action_q2=float((acts["a2_pol"] - acts["a1_pol"])
                                      .abs().max()),
                    # ПОКАНАЛЬНЫЕ МАКСИМУМЫ ПО ИСПОЛНЯЕМЫМ ПОЗИЦИЯМ: их
                    # сравнение с истинными действиями и есть проверка
                    # разумности диапазона.
                    act_absmax=acts["a2_pol"][:, :H_EXEC].abs()
                    .amax(dim=(0, 1)).cpu().numpy(),
                    true_absmax=action[:, :H_EXEC].abs()
                    .amax(dim=(0, 1)).cpu().numpy(),
                    align1_q_to_p=float(lparts["align1_q_to_p"]),
                    align2_q_to_p=float(lparts["align2_q_to_p"]),
                    align1_p_to_q=float(lparts["align1_p_to_q"]),
                    align2_p_to_q=float(lparts["align2_p_to_q"]),
                    **align_exec_stats(q1_tok_logits, q2_tok_logits, paths,
                                       tok),
                    **agreement_stats(i1_tok, i2_tok, paths, torch),
                    # ПОСТРОЧНЫЕ ЗНАЧЕНИЯ, А НЕ МЕДИАНЫ ПО БАТЧУ. Среднее
                    # медиан батчей — не медиана выборки; складывать будем
                    # сами значения и брать одну медиану в evaluate.
                    **{f"soft_ppl_{s}{lv}_rows": tok.soft_perplexity(
                        pr[:, :H_EXEC]).reshape(-1).cpu().numpy()
                       for s, lv, pr in (("q", 1, p1_tok), ("q", 2, p2_tok),
                                         ("p", 1, paths["q1_pol_probs"]),
                                         ("p", 2, paths["q2_pol_probs"]))},
                    # ПОКАНАЛЬНЫЕ АБСОЛЮТНЫЕ ЗНАЧЕНИЯ ДЛЯ p99 — в тех же
                    # НОРМИРОВАННЫХ единицах, в которых лежит кэш.
                    **{f"abs_{nm}": acts[nm][:, :H_EXEC].abs()
                       .reshape(-1, 7).cpu().numpy()
                       for nm in ("a1_pol", "a2_pol")},
                    latent1=(z0 + c1_pol).detach(),
                    q1_codes=out["pred_codes"][1].detach().cpu().numpy(),
                    q2_codes=out["pred_codes"][2].detach().cpu().numpy(),
                    q1_codes_tok=i1_tok.detach().cpu().numpy(),
                    q2_codes_tok=i2_tok.detach().cpu().numpy(),
                    row_a0=rows["a0"].detach().cpu().numpy(),
                    row_a1_pol=rows["a1_pol"].detach().cpu().numpy(),
                    row_a2_pol=rows["a2_pol"].detach().cpu().numpy(),
                    row_a1_tok=rows["a1_tok"].detach().cpu().numpy(),
                    row_a2_tok=rows["a2_tok"].detach().cpu().numpy(),
                    latent=out["cumulative_latents"][2].detach(),
                    latent_true=z_e.detach(),
                    q0_logits_sha=hashlib.sha1(np.ascontiguousarray(
                        out["logits"][0].detach().float().cpu().numpy()
                    ).tobytes()).hexdigest()[:12])
        if not train and collect_topk[0] > 0:
            collect_topk[0] -= 1
            oracle = {}
            for lv, base_, book_, probs_ in (
                    (1, z0, c1, paths["q1_pol_probs"]),
                    (2, z0 + hard_c1, c2, paths["q2_pol_probs"])):
                idx_k = probs_.topk(max(TOPK_ORACLE), dim=-1).indices
                errs = []
                for j in range(max(TOPK_ORACLE)):
                    rows_j, _m_j = weighted_row_error(
                        decode_fp32(base_ + book_[idx_k[..., j]]),
                        action, weights_gate, torch)
                    errs.append(rows_j)
                stacked = torch.stack(errs, 0)
                for k_ in TOPK_ORACLE:
                    oracle[f"topk{k_}_level{lv}"] = float(
                        stacked[:k_].min(0).values.mean())
            stat["topk_oracle"] = oracle
        if not train and collect_dist[0]:
            # РАССТОЯНИЯ ДЛЯ КАЛИБРОВКИ ВОССТАНАВЛИВАЮТСЯ ИЗ ЛОГИТОВ:
            # logits = -d / tau, tau > 0, поэтому второй раз считать
            # расстояния не нужно.
            for lv, lg in ((1, q1_tok_logits), (2, q2_tok_logits)):
                stat[f"d{lv}_exec"] = (
                    -lg[:, :H_EXEC].detach().float() * tau_tok[lv]).cpu()
        if not train:
            # В ОЦЕНКЕ ОТЛОЖЕННЫЕ СЧЁТЧИКИ ЧИТАЮТСЯ СРАЗУ: она и так
            # синхронизируется, а решение принимается по её числам.
            check_pending()
        return (total if train else None), stat

    PATHS5 = PATH_NAMES
    q0_logits_ref = {}

    def agreement_stats(i1, i2, paths_, torch_):
        """Согласие argmax политики с argmin токенизатора и ранг.

        ВЕЛИЧИНА, НЕ ЗАВИСЯЩАЯ ОТ ТЕМПЕРАТУРЫ. Кросс-энтропия Q->P
        измеряет в основном резкость Q: при почти равномерном Q она равна
        среднему -log p_k и велика для любой уверенной политики, читает та
        токенизатор или нет. Согласие по верхнему коду и ранг выбора
        токенизатора в порядке политики от масштаба логитов не зависят.
        """
        res = {}
        for lv, idx, probs in ((1, i1, paths_["q1_pol_probs"]),
                               (2, i2, paths_["q2_pol_probs"])):
            p_exec = probs[:, :H_EXEC].float()
            want = idx[:, :H_EXEC]
            res[f"agree{lv}_top1"] = float(
                (p_exec.argmax(-1) == want).float().mean())
            chosen = p_exec.gather(-1, want.unsqueeze(-1))
            # ЧИСЛО КОДОВ ВЫШЕ, А НЕ «РАНГ». Величина нуль-базированная:
            # 3 означает четвёртое место. Прежнее имя rank_median читалось
            # как «третье место», и я так его и прочитал в отчёте.
            above = (p_exec > chosen).sum(-1).reshape(-1)
            res[f"n_above{lv}_rows"] = above.cpu().numpy()
            # RECALL@K: попал ли код токенизатора в top-k политики. Это
            # ПРИСУТСТВИЕ кода, в отличие от оракула, который выбирает
            # лучший по истинной ошибке и потому мерит «есть ли среди
            # top-k хоть какой-нибудь полезный код».
            for k_ in TOPK_ORACLE:
                res[f"recall{lv}_at{k_}"] = float(
                    (above < k_).float().mean())
        return res

    def align_exec_stats(q1_tok_lg, q2_tok_lg, paths_, tok_):
        """Выравнивание ТОЛЬКО на исполняемых позициях, оба направления.

        Среднее по всем 16 позициям чанка может скрыть случайного читателя
        на первых восьми, а исполняются именно они.
        """
        res = {}
        for lv, tok_lg, pol_lg in ((1, q1_tok_lg, paths_["q1_pol_logits"]),
                                   (2, q2_tok_lg, paths_["q2_pol_logits"])):
            pair = tok_.bidirectional_alignment(tok_lg[:, :H_EXEC],
                                                pol_lg[:, :H_EXEC])
            res[f"align{lv}_q_to_p_exec"] = float(
                pair["tokenizer_to_policy"])
            res[f"align{lv}_p_to_q_exec"] = float(
                pair["policy_to_tokenizer"])
        return res

    def evaluate(batch_list, tag, keep_rows=False):
        """Жёсткий вывод на части: та же величина, что и в отборе эпохи.

        ВСЕ ПЯТЬ ПУТЕЙ АГРЕГИРУЮТСЯ. Прежде считались все пять, а в
        результат попадали три, и таксономия отказа — ради которой пять
        путей и заведены — по артефакту была недоказуема.
        """
        model.eval()
        acc = {k: 0.0 for k in PATHS5}
        acc.update(a2_pol_eq=0.0, frac_worse_q1=0.0, frac_worse_q2=0.0,
                   d_latent_q1=0.0, d_latent_q2=0.0)
        n_rows = 0
        dmax = {"d_action_q1": 0.0, "d_action_q2": 0.0}
        absmax = {"act_absmax": None, "true_absmax": None}
        align_acc = {f"align{lv}_{d}{suf}": 0.0
                     for lv in (1, 2) for d in ("q_to_p", "p_to_q")
                     for suf in ("", "_exec")}
        align_acc.update({f"agree{lv}_top1": 0.0 for lv in (1, 2)})
        align_acc.update({f"recall{lv}_at{k_}": 0.0
                          for lv in (1, 2) for k_ in TOPK_ORACLE})
        pooled = {f"soft_ppl_{s}{lv}": [] for s in ("q", "p")
                  for lv in (1, 2)}
        pooled.update({f"n_above{lv}": [] for lv in (1, 2)})
        abs_pool = {"a1_pol": [], "a2_pol": []}
        codes = {k: [] for k in ("q1_pol", "q2_pol", "q1_tok", "q2_tok")}
        sup_rows, sup1_rows, ref_rows, rows_keep = [], [], [], []
        oracle_acc, oracle_n = {}, 0
        with torch.no_grad():
            for po, sel in batch_list:
                _l, stat = run_batch(po, sel, False)
                w = float(stat["rows"])
                n_rows += stat["rows"]
                for k in PATHS5:
                    acc[k] += stat[k] * w
                acc["a2_pol_eq"] += stat["a2_pol_equal_weights"] * w
                acc["frac_worse_q1"] += stat["frac_worse_q1"] * w
                acc["frac_worse_q2"] += stat["frac_worse_q2"] * w
                acc["d_latent_q1"] += stat["d_latent_q1"] * w
                acc["d_latent_q2"] += stat["d_latent_q2"] * w
                # МАКСИМУМЫ АГРЕГИРУЮТСЯ МАКСИМУМОМ, А НЕ СРЕДНИМ. Они
                # считались в run_batch и терялись в evaluate — ровно та же
                # ошибка, из-за которой пропадали два из пяти путей.
                for k_ in ("d_action_q1", "d_action_q2"):
                    dmax[k_] = max(dmax[k_], float(stat[k_]))
                for k_ in absmax:
                    absmax[k_] = np.asarray(stat[k_], np.float64) \
                        if absmax[k_] is None \
                        else np.maximum(absmax[k_],
                                        np.asarray(stat[k_], np.float64))
                for k_ in align_acc:
                    align_acc[k_] += float(stat[k_]) * w
                # КОДЫ ХРАНЯТСЯ С ПОЗИЦИЯМИ, А НЕ СПЛЮЩЕННЫМИ: схлопывание
                # бывает позиционным — книга жива в среднем и мертва на
                # первой позиции чанка, которая и исполняется.
                for key_, src_ in (("q1_pol", "q1_codes"),
                                   ("q2_pol", "q2_codes"),
                                   ("q1_tok", "q1_codes_tok"),
                                   ("q2_tok", "q2_codes_tok")):
                    codes[key_].append(
                        np.asarray(stat[src_]).reshape(stat["rows"], -1))
                # ЛОГИТЫ q0 СВЕРЯЮТСЯ ПО ОТПЕЧАТКУ, А НЕ ТОЛЬКО КОДЫ.
                # Коды — это argmax; замороженная часть могла бы поехать,
                # не сдвинув argmax ни в одной позиции.
                key = (po, int(sel[0]), int(sel[-1]))
                if key in q0_logits_ref:
                    if q0_logits_ref[key] != stat["q0_logits_sha"]:
                        raise SystemExit(
                            f"логиты q0 на батче {key} изменились "
                            f"({q0_logits_ref[key]} -> "
                            f"{stat['q0_logits_sha']}): замороженная часть "
                            f"модели поехала, коды это скрыли")
                else:
                    q0_logits_ref[key] = stat["q0_logits_sha"]
                # ОПОРА ДЕКОДЕРА — НА ВСЕЙ ЧАСТИ, А НЕ НА ПЕРВЫХ ВОСЬМИ
                # БАТЧАХ, И С ЭТАЛОНОМ. Эталон — собственная ошибка
                # квантования кодека на ИСТИННОМ латенте: это тот уровень
                # непредставимости, с которым декодер обучался работать.
                for k_ in pooled:
                    pooled[k_].append(np.asarray(stat[f"{k_}_rows"]))
                for k_ in abs_pool:
                    abs_pool[k_].append(np.asarray(stat[f"abs_{k_}"]))
                _s, rel = decoder_support(
                    stat["latent"], codec, torch, quantizers,
                    nearest_code, code_contribution)
                sup_rows.append(rel)
                # ОПОРА ДЕКОДЕРА ДЛЯ УРОВНЯ 1 ТОЖЕ. Кандидатом может
                # оказаться q1, и тогда гейт обязан относиться к ЕГО
                # латенту, а не к латенту пути, который не исполняется.
                _s1, rel1 = decoder_support(
                    stat["latent1"], codec, torch, quantizers,
                    nearest_code, code_contribution)
                sup1_rows.append(rel1)
                _s2, rel2 = decoder_support(
                    stat["latent_true"], codec, torch, quantizers,
                    nearest_code, code_contribution)
                ref_rows.append(rel2)
                if "topk_oracle" in stat:
                    oracle_n += 1
                    for k_, v_ in stat["topk_oracle"].items():
                        oracle_acc[k_] = oracle_acc.get(k_, 0.0) + v_
                if keep_rows:
                    rows_keep.append({k: stat[f"row_{k}"] for k in PATHS5})
        n = max(n_rows, 1)
        res = {k: v / n for k, v in acc.items()}
        res.update(dmax)
        res.update({k: v / n for k, v in align_acc.items()})
        res["act_absmax"] = [float(x) for x in absmax["act_absmax"]]
        res["true_absmax"] = [float(x) for x in absmax["true_absmax"]]
        res["log_vocab"] = float(np.log(float(vocab)))
        # ГЛОБАЛЬНЫЕ МЕДИАНЫ, А НЕ СРЕДНЕЕ МЕДИАН ПО БАТЧАМ.
        for k_, chunks in pooled.items():
            flat = np.concatenate(chunks)
            res[f"{k_}_median"] = float(np.median(flat))
            res[f"{k_}_mean"] = float(np.mean(flat))
        # ПОКАНАЛЬНЫЙ p99 КАНДИДАТА — ТА ЖЕ СТАТИСТИКА, ЧТО У ЭТАЛОНА, И В
        # ТЕХ ЖЕ НОРМИРОВАННЫХ ЕДИНИЦАХ.
        for k_, chunks in abs_pool.items():
            flat = np.concatenate(chunks, axis=0)
            res[f"p99_{k_}"] = [float(x) for x in
                                np.percentile(flat, 99.0, axis=0)]
            res[f"absmax_{k_}"] = [float(x) for x in flat.max(axis=0)]
        for k in PATHS5:
            res[f"rms_{k}"] = float(np.sqrt(res[k]))
        for name, key in (("usage_q1", "q1_pol"), ("usage_q2", "q2_pol"),
                          ("usage_q1_tok", "q1_tok"),
                          ("usage_q2_tok", "q2_tok")):
            arr = np.concatenate(codes[key], axis=0)
            if arr.ndim != 2 or arr.shape[1] < H_EXEC:
                raise SystemExit(
                    f"{name}: коды имеют форму {arr.shape}, а исполняемых "
                    f"позиций {H_EXEC}. Разбиение по позициям считало бы "
                    f"не то")
            res[name] = tok.code_usage_stats(
                torch.from_numpy(arr.reshape(-1)), vocab)
            per_pos = [tok.code_usage_stats(
                torch.from_numpy(np.ascontiguousarray(arr[:, t])), vocab)
                for t in range(arr.shape[1])]
            worst = int(np.argmax([s["max_code_share"] for s in per_pos]))
            res[name]["by_position"] = dict(
                n_positions=len(per_pos),
                worst_position=worst,
                worst_max_code_share=float(per_pos[worst]["max_code_share"]),
                worst_perplexity=float(min(s["perplexity"]
                                           for s in per_pos)),
                executed_max_code_share=float(max(
                    s["max_code_share"] for s in per_pos[:H_EXEC])),
                executed_min_perplexity=float(min(
                    s["perplexity"] for s in per_pos[:H_EXEC])))
        sup_all = torch.cat(sup_rows) if sup_rows else None
        sup1_all = torch.cat(sup1_rows) if sup1_rows else None
        ref_all = torch.cat(ref_rows) if ref_rows else None

        def support_block(rows_, scope_):
            if rows_ is None:
                return None
            return dict(
                scope=scope_,
                rel_residual=float(rows_.mean()),
                rel_residual_median=float(rows_.median()),
                rel_residual_p95=float(torch.quantile(rows_, 0.95)),
                rel_residual_p99=float(torch.quantile(rows_, 0.99)),
                rel_residual_max=float(rows_.max()),
                reference_rel_residual=float(ref_all.mean()),
                reference_rel_residual_median=float(ref_all.median()),
                reference_rel_residual_p95=float(
                    torch.quantile(ref_all, 0.95)),
                reference_rel_residual_p99=float(
                    torch.quantile(ref_all, 0.99)),
                share_above_reference_p99=float(
                    (rows_ > torch.quantile(ref_all, 0.99)).float().mean()),
                n=int(rows_.numel()),
                reference="собственная ошибка квантования кодека на "
                          "истинном латенте того же батча")

        res["decoder_support_q1"] = support_block(sup1_all, "path a1_pol")
        res["decoder_support_q2"] = support_block(sup_all, "path a2_pol")
        # ОПОРА ДЕКОДЕРА СЧИТАЕТСЯ ДЛЯ ОБОИХ ИСПОЛНЯЕМЫХ ПУТЕЙ: кандидатом
        # может оказаться уровень 1, и тогда гейт обязан относиться к его
        # латенту. Пять путей, цикл decode->encode и суммы исходных книг
        # как отдельная опора остаются работой перед пилотом.
        res["decoder_support"] = res["decoder_support_q2"]
        res["rows"] = n_rows
        if oracle_n:
            # RMS, А НЕ MSE: сравнимо с остальными путями.
            res["topk_oracle"] = dict(
                {k_: float(np.sqrt(v_ / oracle_n))
                 for k_, v_ in oracle_acc.items()},
                batches=oracle_n,
                scope=f"первые {oracle_n} батчей части, диагностика")
            line = "; ".join(
                f"уровень {lv}: " + ", ".join(
                    f"top{k_} {res['topk_oracle'][f'topk{k_}_level{lv}']:.6f}"
                    for k_ in TOPK_ORACLE)
                for lv in (1, 2))
            print(f"      оракул верхних кодов ({oracle_n} батчей) — {line}")
        if keep_rows:
            res["rows_by_path"] = {
                k: np.concatenate([r[k] for r in rows_keep]) for k in PATHS5}
        print(f"    {tag}: RMS a0 {res['rms_a0']:.6f}; книга a1_tok "
              f"{res['rms_a1_tok']:.6f}, a2_tok {res['rms_a2_tok']:.6f}; "
              f"политика a1_pol {res['rms_a1_pol']:.6f}, a2_pol "
              f"{res['rms_a2_pol']:.6f}; хуже q0: q1 "
              f"{100 * res['frac_worse_q1']:.1f}%, q2 "
              f"{100 * res['frac_worse_q2']:.1f}% строк")
        print(f"      апостериор: мягкая perplexity (медиана) Q "
              f"{res['soft_ppl_q1_median']:.1f}/"
              f"{res['soft_ppl_q2_median']:.1f}, P "
              f"{res['soft_ppl_p1_median']:.1f}/"
              f"{res['soft_ppl_p2_median']:.1f}")
        print(f"      чтение книги: согласие top-1 "
              f"{100 * res['agree1_top1']:.2f}%/"
              f"{100 * res['agree2_top1']:.2f}% при случайном "
              f"{100 / vocab:.3f}%; кодов выше выбора Q (медиана) "
              f"{res['n_above1_median']:.0f}/{res['n_above2_median']:.0f}; "
              f"recall@3 {100 * res['recall1_at3']:.1f}%/"
              f"{100 * res['recall2_at3']:.1f}%, recall@10 "
              f"{100 * res['recall1_at10']:.1f}%/"
              f"{100 * res['recall2_at10']:.1f}%")
        print(f"      поправка: |c1|/|z0| {res['d_latent_q1']:.4f}, "
              f"|c2|/|z0| {res['d_latent_q2']:.4f}; max|a1-a0| "
              f"{res['d_action_q1']:.5f}, max|a2-a1| "
              f"{res['d_action_q2']:.5f}")
        for lv in (1, 2):
            up, uq = res[f"usage_q{lv}"], res[f"usage_q{lv}_tok"]
            print(f"      книга {lv}: P perplexity {up['perplexity']:.1f}, "
                  f"мёртвых {up['dead_codes']}, макс доля "
                  f"{up['max_code_share']:.3f} | Q perplexity "
                  f"{uq['perplexity']:.1f}, мёртвых {uq['dead_codes']}, "
                  f"макс доля {uq['max_code_share']:.3f}")
        for lv_ in (1, 2):
            ds = res[f"decoder_support_q{lv_}"]
            if not ds:
                continue
            print(f"      опора декодера a{lv_}_pol: остаток "
                  f"{ds['rel_residual']:.4f}, медиана "
                  f"{ds['rel_residual_median']:.4f}, p95 "
                  f"{ds['rel_residual_p95']:.4f}, p99 "
                  f"{ds['rel_residual_p99']:.4f}; эталон "
                  f"{ds['reference_rel_residual']:.4f}, p95 "
                  f"{ds['reference_rel_residual_p95']:.4f}, p99 "
                  f"{ds['reference_rel_residual_p99']:.4f}; выше эталонного "
                  f"p99 {100 * ds['share_above_reference_p99']:.1f}% строк")
        return res

    # --- ОБУЧЕНИЕ ---------------------------------------------------------
    # --- НЕТ ЛИ В МОДЕЛИ ЧЕГО-ТО, ЗАВИСЯЩЕГО ОТ РЕЖИМА -------------------
    mode_dependent = []
    for name_, mod_ in model.named_modules():
        if isinstance(mod_, torch.nn.modules.dropout._DropoutNd):
            if float(getattr(mod_, "p", 0.0)) > 0.0:
                mode_dependent.append(f"{name_}: dropout p={mod_.p}")
        elif isinstance(mod_, torch.nn.modules.batchnorm._BatchNorm):
            mode_dependent.append(f"{name_}: batchnorm")
    if mode_dependent:
        raise SystemExit(
            "в модели есть операции, зависящие от режима: "
            + "; ".join(mode_dependent[:5])
            + ". Обучение идёт в режиме eval именно потому, что train() "
              "менял бы h18 и h24 при неизменных кодах q0; с такими "
              "модулями выбор режима надо решать явно, а не молча")
    print(f"  операций, зависящих от режима: нет "
          f"(проверено {sum(1 for _ in model.named_modules())} модулей)")

    # --- ЗАМОРОЖЕННОЕ: ЧТО ИМЕННО ДОКАЗЫВАЕТСЯ И ЧЕМ ---------------------
    trainable_set = set(info["names"])
    extra_opt, missing_opt = optimizer_covers_exactly(opt, model,
                                                      trainable_set)
    if extra_opt or missing_opt:
        raise SystemExit(
            f"оптимизатор накрывает не белый список: лишних тензоров "
            f"{extra_opt}, не попало {missing_opt[:4]}")
    clashes = no_alias_between(model, trainable_set)
    if clashes:
        raise SystemExit(
            f"обучаемый тензор делит хранилище с замороженным: "
            f"{clashes[:3]}. Шаг оптимизатора двигал бы замороженное")
    frozen_inv0, n_frozen = frozen_invariant(model, torch, trainable_set)
    t_hash = time.time()
    frozen_sha0, n_sha, n_elem_frozen = frozen_content_sha(
        model, torch, trainable_set)
    print(f"  замороженных тензоров {n_frozen} ({n_elem_frozen} значений): "
          f"инвариант {frozen_inv0}, побитовый отпечаток {frozen_sha0} "
          f"({time.time() - t_hash:.1f} с)")
    if n_sha != n_frozen:
        raise SystemExit(f"обход замороженного дал {n_sha} и {n_frozen}")

    def check_frozen(tag, content=False):
        """Дешёвый инвариант всегда, побитовый отпечаток — по требованию.

        Инвариант ловит запись на месте (по `_version`) и подмену тензора
        (по хранилищу); побитовый отпечаток ловит вообще всё, но читает
        все веса на хост, поэтому берётся дважды за прогон.
        """
        inv_, n_ = frozen_invariant(model, torch, trainable_set)
        if inv_ != frozen_inv0 or n_ != n_frozen:
            raise SystemExit(
                f"{tag}: инвариант замороженного {inv_} ({n_} тензоров) "
                f"против {frozen_inv0} ({n_frozen}) до обучения: что-то "
                f"вне белого списка изменено на месте или подменено")
        if content:
            sha_, _n, _e = frozen_content_sha(model, torch, trainable_set)
            if sha_ != frozen_sha0:
                raise SystemExit(
                    f"{tag}: побитовый отпечаток замороженного {sha_} "
                    f"против {frozen_sha0} до обучения")
            return sha_
        return inv_

    # --- КАЛИБРОВКА ТЕМПЕРАТУРЫ ТОКЕНИЗАТОРА -----------------------------
    # ПРАВИЛО ВЫБРАНО ПОСЛЕ СМОУКА И ЗАФИКСИРОВАНО ДО ПОЛНОГО ПРОГОНА:
    # медианная мягкая perplexity апостериора
    # Q на исполняемых позициях равна --tau-target-perplexity. Без этого
    # апостериор почти равномерен (квадрат расстояния делится на D), и
    # тогда член выравнивания сводится к давлению «сделай P равномерным»,
    # градиент P->Q в книги почти нулевой, а член использования не имеет
    # сигнала. Жёсткий выбор от температуры не зависит.
    if str(a.tau_tokenizer).strip().lower() == "auto":
        collect_dist[0] = True
        # ЧИСЛО КАЛИБРОВОЧНЫХ БАТЧЕЙ — СКОЛЬКО ЕСТЬ. При --smoke --limit 2
        # части уже урезаны, поэтому их два, а не четыре; в артефакт идёт
        # фактическое число.
        calib = parts["train"][:max(int(a.calib_batches), 1)]
        dist = {1: [], 2: []}
        with torch.no_grad():
            for po, sel in calib:
                _l, st_c = run_batch(po, sel, False)
                for lv in (1, 2):
                    dist[lv].append(st_c[f"d{lv}_exec"])
        collect_dist[0] = False
        for lv in (1, 2):
            d_all = torch.cat(dist[lv], 0)
            before = float(tok.soft_perplexity(
                torch.softmax(-d_all, dim=-1)).median())
            tau_tok[lv] = tok.calibrate_temperature(
                d_all, float(a.tau_target_perplexity))
            after = float(tok.soft_perplexity(
                torch.softmax(-d_all / tau_tok[lv], dim=-1)).median())
            tau_report[f"level{lv}"] = dict(
                tau=float(tau_tok[lv]),
                median_soft_perplexity_before=before,
                median_soft_perplexity_after=after,
                target=float(a.tau_target_perplexity),
                rows=int(d_all.shape[0]), batches=len(calib),
                vocab=int(d_all.shape[-1]))
            print(f"  калибровка tau уровня {lv}: {tau_tok[lv]:.4g}; "
                  f"медианная мягкая perplexity {before:.1f} -> "
                  f"{after:.1f} при цели {a.tau_target_perplexity:.1f}")
    else:
        fixed = float(a.tau_tokenizer)
        tau_tok[1] = tau_tok[2] = fixed
        tau_report = dict(mode="fixed", tau=fixed,
                          note="калибровка не проводилась")
        print(f"  tau токенизатора задана вручную: {fixed}")

    history = []
    val0 = evaluate(parts["val_sel"], "эпоха 0, без обучения")
    history.append(dict(epoch=0, train_loss=None,
                        val_a1_pol_rms=val0["rms_a1_pol"],
                        val_a2_pol_rms=val0["rms_a2_pol"],
                        val_frac_worse_q1=val0["frac_worse_q1"],
                        val_frac_worse_q2=val0["frac_worse_q2"], val=val0))
    snapshots = {0: {k: v.detach().clone()
                     for k, v in model.state_dict().items()
                     if k in set(info["names"])}}
    if a.smoke:
        # SMOKE ПРОВЕРЯЕТ, ЧТО ОПТИМИЗАТОР ДЕЙСТВИТЕЛЬНО УЧИТ. Три шага по
        # ОДНОМУ И ТОМУ ЖЕ микробатчу обязаны уменьшить на нём потерю: иначе
        # связность собрана, а обучения нет, и это выяснилось бы только
        # через часы полного прогона.
        po0, sel0 = parts["train"][0]
        probe_losses = []
        collect_components[0] = True
        # ГРУППЫ ДЛЯ НОРМ ГРАДИЕНТОВ. «Градиент ненулевой» при tau порядка
        # 1e-05 почти ничего не говорит: логиты токенизатора равны -d/tau,
        # поэтому градиент по книге получает множитель 1/tau. Нужны сами
        # нормы по компонентам и относительный шаг книг.
        groups = {"books": ["depth_aligned_c1", "depth_aligned_c2"],
                  "heads": [n for n in info["names"]
                            if n.startswith("depth_rvq_heads")],
                  "norms": [n for n in info["names"]
                            if n.startswith("depth_rvq_norms")],
                  "feedback": [n for n in info["names"]
                               if n.startswith("depth_rvq_feedback")]}
        named = dict(model.named_parameters())
        group_params = {g: [named[n] for n in ns] for g, ns in groups.items()}
        books0 = {n: named[n].detach().clone() for n in groups["books"]}
        probe_report = []
        # И ГРАДИЕНТ КАЖДОГО РАЗРЕШЁННОГО ТЕНЗОРА ОБЯЗАН БЫТЬ НЕНУЛЕВЫМ.
        # Общая потеря может падать при мёртвой подгруппе: головы учатся,
        # книги стоят, и по одному числу этого не видно.
        grad_mass = {n_: 0.0 for n_ in info["names"]}
        for probe_step in range(3):
            opt.zero_grad(set_to_none=True)
            l_probe, s_probe = run_batch(po0, sel0, True)
            probe_losses.append(float(l_probe.detach()))
            # НОРМЫ ГРАДИЕНТА ПО КОМПОНЕНТАМ ПОТЕРИ И ПО ГРУППАМ ВЕСОВ.
            step_report = {"step": probe_step + 1,
                           "loss": float(l_probe.detach()), "grad": {}}
            for cname, ctensor in s_probe["components"].items():
                row_ = {}
                for gname, params_ in group_params.items():
                    gs = torch.autograd.grad(
                        ctensor, params_, retain_graph=True,
                        allow_unused=True)
                    row_[gname] = float(sum(
                        float(g.norm()) ** 2 for g in gs
                        if g is not None) ** 0.5)
                step_report["grad"][cname] = row_
            l_probe.backward()
            # КОНЕЧНОСТЬ ПРОВЕРЯЕТСЯ ДО opt.step(), А НЕ ПОСЛЕ НЕГО. Шаг по
            # nan-градиенту портит веса, и после него проверять уже поздно;
            # а накопленная масса с nan не равна нулю, поэтому проверка на
            # «мёртвый тензор» такой градиент пропускала.
            for n_, p_ in model.named_parameters():
                if n_ not in grad_mass:
                    continue
                if p_.grad is None:
                    raise SystemExit(
                        f"проба, шаг {probe_step + 1}: у {n_} нет "
                        f"градиента, хотя он в белом списке")
                if not torch.isfinite(p_.grad).all():
                    raise SystemExit(
                        f"проба, шаг {probe_step + 1}: градиент {n_} "
                        f"нечисловой. Шаг по нему испортил бы веса")
                grad_mass[n_] += float(p_.grad.abs().sum())
            opt.step()
            with torch.no_grad():
                step_report["book_relative_step"] = {
                    n: float((named[n].detach() - books0[n]).norm()
                             / books0[n].norm().clamp_min(1e-12))
                    for n in groups["books"]}
            _l_ev, s_ev = run_batch(po0, sel0, False)
            step_report["after_step"] = {
                k_: float(s_ev[k_]) for k_ in ("agree1_top1", "agree2_top1")}
            for s_, lv_ in (("q", 1), ("q", 2)):
                step_report["after_step"][f"soft_ppl_{s_}{lv_}"] = float(
                    np.median(s_ev[f"soft_ppl_{s_}{lv_}_rows"]))
            probe_report.append(step_report)
            print(f"    шаг {probe_step + 1}: потеря "
                  f"{step_report['loss']:.5f}; нормы градиента по книгам "
                  + ", ".join(
                      f"{c}={step_report['grad'][c]['books']:.3e}"
                      for c in ("action", "align_q_to_p", "align_p_to_q",
                                "mono", "usage"))
                  + "; по головам "
                  + ", ".join(
                      f"{c}={step_report['grad'][c]['heads']:.3e}"
                      for c in ("action", "align_q_to_p", "align_p_to_q"))
                  + "; шаг книг "
                  + ", ".join(
                      f"{n.split('_')[-1]}="
                      f"{step_report['book_relative_step'][n]:.3e}"
                      for n in groups["books"])
                  + "; согласие "
                  f"{100 * step_report['after_step']['agree1_top1']:.2f}%/"
                  f"{100 * step_report['after_step']['agree2_top1']:.2f}%, "
                  f"мягкая ppl Q "
                  f"{step_report['after_step']['soft_ppl_q1']:.1f}/"
                  f"{step_report['after_step']['soft_ppl_q2']:.1f}")
        collect_components[0] = False
        bad_mass = sorted(n_ for n_, v_ in grad_mass.items()
                          if not (np.isfinite(v_) and v_ > 0.0))
        if bad_mass:
            raise SystemExit(
                f"масса градиента за три шага не конечна и положительна у "
                f"{bad_mass}: эти тензоры в белом списке, но не обучаются, "
                f"и падение общей потери их не касается")
        print(f"  градиент дошёл до всех {len(grad_mass)} разрешённых "
              f"тензоров, минимальная масса "
              f"{min(grad_mass.values()):.3e}")
        with torch.no_grad():
            own_p = dict(model.state_dict())
            for k_, v_ in snapshots[0].items():
                own_p[k_].copy_(v_)
        opt.state.clear()
        opt.zero_grad(set_to_none=True)
        print("  проба обучаемости на одном микробатче: "
              + " -> ".join(f"{x:.5f}" for x in probe_losses))
        if not probe_losses[-1] < probe_losses[0]:
            raise SystemExit(
                f"три шага по одному микробатчу не уменьшили потерю "
                f"({probe_losses}): оптимизатор собран, но не учит")
        sha_back = check_frozen("после пробы", content=True)
        st_back = k14c.state_sha({k_: model.state_dict()[k_].detach().float()
                                  .cpu().numpy() for k_ in info["names"]})
        st_zero = k14c.state_sha({k_: v_.detach().float().cpu().numpy()
                                  for k_, v_ in snapshots[0].items()})
        if st_back != st_zero:
            raise SystemExit(
                f"после пробы веса не восстановились: {st_back} против "
                f"{st_zero}")
        print(f"  веса после пробы восстановлены побитово ({st_back}), "
              f"замороженное {sha_back}")


    probe_diagnostics = probe_report if a.smoke else None
    forecast = None
    t_start = time.time()
    order = list(parts["train"])
    for epoch in range(1, int(a.epochs) + 1):
        # РЕЖИМ eval ДЕРЖИТСЯ И ВО ВРЕМЯ ОПТИМИЗАЦИИ. model.train()
        # переключает всю VLA, включая замороженные 24 слоя: autograd от
        # режима не зависит, а dropout зависит, и h18/h24 стали бы другими
        # при том же q0 — побитовая сверка КОДОВ q0 этого не заметила бы.
        # Отсутствие активного dropout проверено выше, отказом.
        model.eval()
        rng = np.random.default_rng(int(a.seed) + epoch)
        idx = rng.permutation(len(order))
        run_sum, nb, t_ep = None, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for step, j in enumerate(idx, start=1):
            po, sel = order[j]
            loss, stat = run_batch(po, sel, True)
            (loss / float(a.accum)).backward()
            # НАКОПЛЕНИЕ НА УСТРОЙСТВЕ. float(loss) на каждом батче — это
            # синхронизация с GPU ради числа, которое печатается раз в 250
            # батчей.
            if run_sum is None:
                run_sum = {k_: v_.clone() for k_, v_ in stat["track"].items()}
            else:
                for k_, v_ in stat["track"].items():
                    run_sum[k_] += v_
            nb += 1
            if step % int(a.accum) == 0 or step == len(idx):
                # ПОЛНАЯ ПРОВЕРКА ГРАДИЕНТОВ — НА ПЕРВЫХ ДЕСЯТИ ШАГАХ И
                # ДАЛЬШЕ РЕДКО. `isfinite(...).all()` по каждому обучаемому
                # тензору — это синхронизация с GPU на каждый шаг; список
                # обучаемого фиксирован, поэтому «нет градиента» — ошибка
                # сборки и видна сразу, а nan появляется не бесшумно.
                if step <= 10 or step % 250 == 0 or step == len(idx):
                    check_pending()
                    nog = [n_ for n_, p_ in model.named_parameters()
                           if p_.requires_grad and p_.grad is None]
                    nf = [n_ for n_, p_ in model.named_parameters()
                          if p_.requires_grad and p_.grad is not None
                          and not torch.isfinite(p_.grad).all()]
                    if nog or nf:
                        raise SystemExit(f"градиенты: нет у {nog[:3]}, "
                                         f"нечисловые у {nf[:3]}")
                opt.step()
                opt.zero_grad(set_to_none=True)
            if epoch == 1 and step == int(a.forecast_batches):
                forecast = forecast_runtime(time.time() - t_start, step,
                                            len(order), int(a.epochs))
                print(f"    ПРОГНОЗ по {step} батчам: "
                      f"{forecast['per_batch_s']:.2f} с/батч, "
                      f"{forecast['per_epoch_h']:.1f} ч/эпоха, "
                      f"{forecast['total_h']:.1f} ч на {a.epochs} эпох",
                      flush=True)
            if step % 250 == 0:
                el = (time.time() - t_ep) / 60
                mean_ = {k_: float(v_) / nb for k_, v_ in run_sum.items()}
                print(f"    эпоха {epoch}: батч {step}/{len(idx)}, потеря "
                      f"{mean_['loss']:.5f} (action {mean_['action']:.4f}, "
                      f"align {mean_['align']:.4f}, mono {mean_['mono']:.4f}, "
                      f"usage {mean_['usage']:.4f}), a2_pol/a0 "
                      f"{mean_['a2_pol'] / max(mean_['a0'], 1e-12):.4f}, "
                      f"{el:.1f} мин, осталось "
                      f"{el * (len(idx) - step) / max(step, 1):.0f} мин",
                      flush=True)
        train_mean = {k_: float(v_) / max(nb, 1)
                      for k_, v_ in (run_sum or {}).items()}
        val = evaluate(parts["val_sel"], f"эпоха {epoch}")
        history.append(dict(epoch=epoch, train_loss=train_mean.get("loss"),
                            train_parts=train_mean,
                            val_a1_pol_rms=val["rms_a1_pol"],
                            val_a2_pol_rms=val["rms_a2_pol"],
                            val_frac_worse_q1=val["frac_worse_q1"],
                            val_frac_worse_q2=val["frac_worse_q2"],
                            val=val))
        snapshots[epoch] = {k: v.detach().clone()
                            for k, v in model.state_dict().items()
                            if k in set(info["names"])}
        check_frozen(f"эпоха {epoch}")
        print(f"  эпоха {epoch}: потеря {train_mean.get('loss', 0.0):.5f}, "
              f"val_sel RMS a2_pol {val['rms_a2_pol']:.6f}, замороженное "
              f"не двигалось")

    # --- ДВА УРОВНЯ — ДВА КАНДИДАТА, ДВА СНАПШОТА ------------------------
    # Уровень 1 исполняется за 18 слоёв, уровень 2 за 24. Это РАЗНЫЕ
    # рабочие точки, и выбирать эпоху по одной, а объявлять кандидатом
    # другую, нельзя: лучшая эпоха для q1 просто терялась.
    best_level = {}
    for lv in (1, 2):
        ep_, row_ = select_epoch(history, level=lv)
        best_level[lv] = (ep_, row_)
        print(f"\n  уровень q{lv} ({LEVEL_LAYERS[lv]} слоёв): выбрана "
              f"эпоха {ep_} по val_sel RMS a{lv}_pol "
              f"({row_[f'val_a{lv}_pol_rms']:.6f}), доля ухудшений "
              f"{row_[f'val_frac_worse_q{lv}']:.3f}")

    confirm_level, sel_sha_level, confirm_cache = {}, {}, {}
    for lv in (1, 2):
        ep_, row_ = best_level[lv]
        with torch.no_grad():
            own = dict(model.state_dict())
            for k_, v_ in snapshots[ep_].items():
                own[k_].copy_(v_)
        sel_sha_level[lv] = k14c.state_sha(
            {k_: model.state_dict()[k_].detach().float().cpu().numpy()
             for k_ in info["names"]})
        if ep_ in confirm_cache:
            # ОДНА И ТА ЖЕ ЭПОХА — один проход, а не два одинаковых.
            confirm_level[lv] = confirm_cache[ep_]
            print(f"  переоценка q{lv}: эпоха {ep_} уже переоценена")
        else:
            check_frozen(f"после восстановления q{lv}")
            collect_topk[0] = int(a.topk_oracle_batches)
            conf = evaluate(parts["val_sel"],
                            f"переоценка q{lv}, эпоха {ep_}", keep_rows=True)
            collect_topk[0] = 0
            # ПОБИТОВЫЙ ОТПЕЧАТОК — ПОСЛЕ оценки: мутация замороженного
            # внутри прохода, по которому принимается решение, иначе
            # осталась бы непроверенной.
            check_frozen(f"после итоговой оценки q{lv}", content=True)
            confirm_cache[ep_] = conf
            confirm_level[lv] = conf
        conf = confirm_level[lv]
        for name_, got_, want_ in (
                (f"RMS a{lv}_pol", conf[f"rms_a{lv}_pol"],
                 row_[f"val_a{lv}_pol_rms"]),
                ("доля ухудшений", conf[f"frac_worse_q{lv}"],
                 row_[f"val_frac_worse_q{lv}"])):
            scale_ = max(abs(float(want_)), 1e-12)
            if abs(float(got_) - float(want_)) / scale_ > 1e-6:
                raise SystemExit(
                    f"переоценка q{lv}, эпоха {ep_}, не воспроизвела "
                    f"историю: {name_} {got_!r} против {want_!r}. "
                    f"Восстановленный чекпойнт — не тот, по которому "
                    f"принято решение")
        print(f"  переоценка q{lv} воспроизвела строку эпохи {ep_}")
    # Совместимость печати и артефакта: «основной» — исполняемый путь q2.
    best_epoch, best = best_level[2]
    confirm = confirm_level[2]
    sel_sha = sel_sha_level[2]

    # --- ГЕЙТЫ ПРИЁМКИ: ПО КАЖДОМУ УРОВНЮ НЕЗАВИСИМО --------------------
    # Уровень 1 вообще не зависит от уровня 2: его логиты считаются до него.
    # Поэтому схлопнувшаяся или случайная книга 2 НЕ ИМЕЕТ ПРАВА блокировать
    # исправный q1, который к тому же дешевле на шесть слоёв. Пороги
    # зафиксированы до полного прогона (часть из них выбрана после смоука —
    # это отмечено в `thresholds_origin`).
    COLLAPSE_MAX_SHARE, COLLAPSE_MIN_PPL = 0.98, 2.0
    ACTION_RANGE_FACTOR = 1.5
    ACTION_CLIP_BOUND = 1.5
    ALIGN_AGREE_FACTOR = 5.0
    chance = 1.0 / float(vocab)
    gates = {}
    for lv in (1, 2):
        conf = confirm_level[lv]
        # --- схлопывание книги этого уровня, P и Q
        for side, tag_ in (("", "pol"), ("_tok", "tok")):
            u = conf[f"usage_q{lv}{side}"]
            bp = u["by_position"]
            gates[f"collapse_q{lv}_{tag_}"] = dict(
                max_code_share=float(u["max_code_share"]),
                perplexity=float(u["perplexity"]),
                used_codes=int(u["used_codes"]),
                executed_max_code_share=bp["executed_max_code_share"],
                executed_min_perplexity=bp["executed_min_perplexity"],
                limit_max_share=COLLAPSE_MAX_SHARE,
                limit_min_perplexity=COLLAPSE_MIN_PPL,
                category="rollout_blocker",
                passed=bool(
                    u["max_code_share"] <= COLLAPSE_MAX_SHARE
                    and u["perplexity"] >= COLLAPSE_MIN_PPL
                    and bp["executed_max_code_share"] <= COLLAPSE_MAX_SHARE
                    and bp["executed_min_perplexity"] >= COLLAPSE_MIN_PPL))
        # --- опора декодера для латента ЭТОГО уровня
        ds = conf[f"decoder_support_q{lv}"]
        gates[f"decoder_support_q{lv}"] = dict(
            scope=ds["scope"],
            rel_residual_p95=float(ds["rel_residual_p95"]),
            reference_p95=float(ds["reference_rel_residual_p95"]),
            rel_residual_mean=float(ds["rel_residual"]),
            reference_mean=float(ds["reference_rel_residual"]),
            share_above_reference_p99=float(ds["share_above_reference_p99"]),
            rule="p95 остатка модели <= p95 остатка кодека на истинном "
                 "латенте",
            category="rollout_blocker",
            passed=bool(ds["rel_residual_p95"]
                        <= ds["reference_rel_residual_p95"]))
        # --- диапазон действий: ОДНИ И ТЕ ЖЕ ЕДИНИЦЫ И ОДНА И ТА ЖЕ
        # СТАТИСТИКА. Кэш хранит действие НОРМИРОВАННЫМ (поделено на
        # max_act_q, обрезано в [-1,1]), декодер возвращает его же. Прежний
        # гейт делил нормированный максимум на ФИЗИЧЕСКИЙ max_act_q и
        # поэтому отказал на смоуке ложно: 0.404/0.204 = 1.99 сравнивало
        # разные единицы. Теперь p99 кандидата против p99 набора в тех же
        # единицах, плюс отдельно абсолютный предел нормированной шкалы.
        p99_cand = np.asarray(conf[f"p99_a{lv}_pol"], np.float64)
        absmax_cand = np.asarray(conf[f"absmax_a{lv}_pol"], np.float64)
        ratios = p99_cand / np.maximum(act_p99_dataset, 1e-12)
        worst_ch = int(np.argmax(ratios))
        gates[f"action_range_q{lv}"] = dict(
            units="normalized codec units, as stored in the cache",
            p99_candidate=[float(x) for x in p99_cand],
            p99_dataset=[float(x) for x in act_p99_dataset],
            absmax_candidate=[float(x) for x in absmax_cand],
            clip_bound=ACTION_CLIP_BOUND,
            worst_channel=worst_ch,
            worst_ratio=float(ratios[worst_ch]),
            rule=(f"p99|a{lv}_pol| <= {ACTION_RANGE_FACTOR} x p99|действия| "
                  f"по набору, поканально, И max|a{lv}_pol| <= "
                  f"{ACTION_CLIP_BOUND} в нормированной шкале"),
            category="rollout_blocker",
            passed=bool(bool((ratios <= ACTION_RANGE_FACTOR).all())
                        and bool((absmax_cand <= ACTION_CLIP_BOUND).all())))
        # --- связь книги и политики НЕ РАЗОРВАНА, величина не зависит от
        # температуры
        agree = float(conf[f"agree{lv}_top1"])
        gates[f"align_not_broken_q{lv}"] = dict(
            agree_top1=agree, chance=chance, factor=ALIGN_AGREE_FACTOR,
            limit=ALIGN_AGREE_FACTOR * chance,
            n_above_median=float(conf[f"n_above{lv}_median"]),
            recall_at3=float(conf[f"recall{lv}_at3"]),
            recall_at10=float(conf[f"recall{lv}_at10"]),
            soft_ppl_q=float(conf[f"soft_ppl_q{lv}_median"]),
            soft_ppl_p=float(conf[f"soft_ppl_p{lv}_median"]),
            q_to_p_exec=float(conf[f"align{lv}_q_to_p_exec"]),
            p_to_q_exec=float(conf[f"align{lv}_p_to_q_exec"]),
            rule=(f"согласие argmax политики с argmin токенизатора на "
                  f"исполняемых позициях >= {ALIGN_AGREE_FACTOR} x 1/V"),
            category="rollout_blocker",
            passed=bool(np.isfinite(agree)
                        and agree >= ALIGN_AGREE_FACTOR * chance))
        # --- уровень вообще лучше черновика. Это КАНДИДАТСКИЙ гейт:
        # отрицательный ответ — результат, а не поломка. Он включает в себя
        # прежний no_regression того же уровня: из a < a0 следует
        # a <= 1.01 a0.
        r_lv = float(conf[f"rms_a{lv}_pol"])
        r0_lv = float(conf["rms_a0"])
        gates[f"improves_q{lv}"] = dict(
            rms=r_lv, rms_a0=r0_lv,
            relative=float(r_lv / max(r0_lv, 1e-12)),
            frac_worse=float(conf[f"frac_worse_q{lv}"]),
            rule=f"RMS(a{lv}_pol) < RMS(a0)",
            category="candidate",
            passed=bool(r_lv < r0_lv))
    failed, by_level = classify_gates(gates)
    failed_level = {lv: by_level[lv]["failed"] for lv in (1, 2)}
    blocked_level = {lv: by_level[lv]["blocker"] for lv in (1, 2)}
    candidate_failed_level = {lv: by_level[lv]["candidate"] for lv in (1, 2)}
    failed_tech = sorted(k for lv in (1, 2) for k in blocked_level[lv])
    failed_cand = sorted(k for lv in (1, 2)
                         for k in candidate_failed_level[lv])
    # ДВУХУРОВНЕВЫЙ ИСХОД: ПРИГОДНОСТЬ КАЖДОГО УРОВНЯ СЧИТАЕТСЯ ПО ЕГО
    # СОБСТВЕННЫМ ГЕЙТАМ. Прежде вердикт мог объявить кандидатом q1, а
    # гейты при этом относились к q2 целиком — то есть случайная голова
    # уровня 2 блокировала исправный и более дешёвый уровень 1.
    two_level = {}
    for lv in (1, 2):
        conf = confirm_level[lv]
        ep_, _row = best_level[lv]
        two_level[f"q{lv}"] = dict(
            layers=LEVEL_LAYERS[lv],
            epoch=ep_,
            rms=float(conf[f"rms_a{lv}_pol"]),
            rms_a0=float(conf["rms_a0"]),
            rms_tokenizer=float(conf[f"rms_a{lv}_tok"]),
            improves=bool(conf[f"rms_a{lv}_pol"] < conf["rms_a0"]),
            failed=sorted(failed_level[lv]),
            failed_rollout_blocker=sorted(blocked_level[lv]),
            failed_candidate=sorted(candidate_failed_level[lv]),
            eligible=bool(not failed_level[lv]),
            state_sha1=sel_sha_level[lv])
    if two_level["q2"]["eligible"]:
        candidate_level = "q2"
    elif two_level["q1"]["eligible"]:
        candidate_level = "q1"
    else:
        candidate_level = "none"
    two_level["candidate_level"] = candidate_level
    two_level["note"] = (
        "уровень 1 исполняется за 18 слоёв, уровень 2 за 24; гейты каждого "
        "уровня считаны по ЕГО путям и ЕГО латенту, и уровень 1 не зависит "
        "от состояния книги 2")
    for lv in (1, 2):
        d_ = two_level[f"q{lv}"]
        print(f"  уровень q{lv} ({d_['layers']} слоёв, эпоха {d_['epoch']}): "
              f"RMS {d_['rms']:.6f} против черновика {d_['rms_a0']:.6f}, "
              f"книга {d_['rms_tokenizer']:.6f}; "
              + ("ПРИГОДЕН" if d_["eligible"]
                 else "не пригоден: " + ", ".join(d_["failed"])))
    print(f"  кандидат: {candidate_level}")
    accepted = None if a.smoke else candidate_level != "none"
    print("  гейты приёмки: " + ("все пройдены" if not failed
                                 else "НЕ ПРОЙДЕНЫ " + ", ".join(failed)))
    for k, v in sorted(gates.items()):
        print(f"    [{v['category'][:4]}] {k:22s} "
              f"{'ok' if v['passed'] else 'ОТКАЗ'}  "
              + ", ".join(f"{kk}={vv}" for kk, vv in v.items()
                          if kk not in ("passed", "rule", "category")))

    payload = dict(
        kind=("k15_smoke" if a.smoke else "k15_depth_rvq"), stage="q1q2",
        variant="depth_aligned", seed=int(a.seed), epochs=int(a.epochs),
        batch=int(a.batch), accum=int(a.accum), lr=float(a.lr),
        wd=float(a.wd), grip_weight=float(a.grip_weight),
        tau_tokenizer=dict(tau_tok), tau_tokenizer_mode=str(a.tau_tokenizer),
        tau_target_perplexity=float(a.tau_target_perplexity),
        tau_calibration=tau_report, tau_policy=float(a.tau_policy),
        align_smoothing=float(tok.ALIGNMENT_SMOOTHING),
        topk_oracle_k=list(TOPK_ORACLE),
        loss_weights=dict(a1_pol=W_A1_POL, a1_tok=W_A1_TOK, a2_pol=W_A2_POL,
                          a2_tok=W_A2_TOK, align=W_ALIGN, mono=W_MONO,
                          usage=W_USAGE, eps_norm=EPS_NORM),
        channel_weights="metric (max_act_q, grip=grip_weight)",
        decode_context="fp32, autocast disabled",
        # ВЕСА СОХРАНЯЮТСЯ И В SMOKE: иначе сохранение и загрузку в нём
        # проверить нечем, а это ровно то, что smoke должен покрывать.
        state={k_: v_.detach().cpu() for k_, v_ in model.state_dict().items()
               if k_ in set(info["names"])},
        accepted=accepted, gates=gates,
        failed_rollout_blocker=sorted(failed_tech),
        failed_candidate=sorted(failed_cand),
        two_level_verdict=two_level,
        candidate_level=candidate_level,
        accepted_scope="ОФЛАЙНОВЫЕ ГЕЙТЫ ДВУХ КАТЕГОРИЙ. rollout_blocker: "
                       "схлопывание книг (P и Q, включая исполняемые "
                       "позиции), опора декодера на финальном пути, "
                       "диапазон действий, неразорванное выравнивание на "
                       "исполняемых позициях. candidate: отсутствие "
                       "регрессии на a2_pol и хотя бы одно улучшение. "
                       "Отказ любого из них — результат обучения, а не "
                       "испорченный артефакт: провенанс, инварианты и "
                       "конечность останавливают прогон раньше. НЕ "
                       "доказывает поведенческой пользы: её мерит роллаут",
        gate_categories={k: dict(category=c, level=lv)
                         for k, (c, lv) in EXPECTED_GATES.items()},
        acceptance_thresholds=dict(
            collapse_max_code_share=COLLAPSE_MAX_SHARE,
            collapse_min_perplexity=COLLAPSE_MIN_PPL,
            action_range_factor=ACTION_RANGE_FACTOR,
            action_clip_bound=ACTION_CLIP_BOUND,
            align_agreement_factor=ALIGN_AGREE_FACTOR),
        thresholds_origin=(
            "до смоука: collapse (0.98 / 2.0), сравнительное правило опоры "
            "декодера, правило выбора эпохи. ПОСЛЕ СМОУКА и зафиксировано "
            "до полного прогона: целевая мягкая perplexity 20, множитель "
            "согласия 5x, множитель диапазона 1.5 и предел нормированной "
            "шкалы 1.5, замена кросс-энтропийного гейта на согласие по "
            "верхнему коду"),
        confirm=confirm,
        confirm_q1=confirm_level[1], confirm_q2=confirm_level[2],
        selected_epoch_q1=best_level[1][0],
        selected_epoch_q2=best_level[2][0],
        selected_state_sha1_q1=sel_sha_level[1],
        selected_state_sha1_q2=sel_sha_level[2],
        states={f"q{lv}": {k_: snapshots[best_level[lv][0]][k_]
                           .detach().cpu()
                           for k_ in info["names"]} for lv in (1, 2)},
        level_layers=dict(LEVEL_LAYERS),
        probe_diagnostics=probe_diagnostics,
        decode_batched=bool(decode_batched[0]),
        decode_batched_gap=float(decode_batched[1]),
        codec=codec_fp, code_version=code_version,
        collapse_thresholds=dict(max_code_share=COLLAPSE_MAX_SHARE,
                                 min_perplexity=COLLAPSE_MIN_PPL),
        trainable_names=info["names"], selected_epoch=best_epoch,
        selected_state_sha1=sel_sha, history=history, forecast=forecast,
        q0_prov=q0_prov, joint_sha1=joint_sha,
        q1_init=os.path.abspath(a.q1_init),
        q1_init_sha1=k11a.file_sha1(a.q1_init), **q1_prov, **gate_info,
        git_head=git_head, git_dirty=bool(dirty),
        val_confirm="НЕ ОТКРЫВАЛАСЬ; в K-14 уже прочитана, поэтому любое её "
                    "использование будет retrospective",
        # МАШИНОЧИТАЕМЫЕ ПОЛЯ, А НЕ ТОЛЬКО ФРАЗА. Строку следующий скрипт
        # прочитать не может, а решение по этой половине принимать нельзя.
        val_confirm_role="retrospective_reused",
        val_confirm_used_for_selection=False,
        val_confirm_evaluated=False,
        frozen_content_sha=frozen_sha0,
        frozen_invariant=frozen_inv0,
        frozen_invariant_note="ЗНАЧЕНИЕ ЛОКАЛЬНО ДЛЯ ПРОЦЕССА: в него входят "
                              "идентификаторы хранилищ, то есть адреса. "
                              "Сравнивать его между прогонами нельзя — для "
                              "этого есть frozen_content_sha, который между "
                              "прогонами совпадает",
        frozen_elements=n_elem_frozen,
        frozen_tensors=n_frozen,
        save_load_check="state serialization round-trip: побитовое равенство "
                        "сохранённых тензоров и state_sha. Воспроизведение "
                        "логитов, кодов и действия после загрузки —"
                        " в k15_rollout",
        device=str(dev), compute_dtype=a.dtype,
        torch_version=str(torch.__version__),
        note=("подтверждающая половина не формировалась; отбор эпохи по "
              "val_sel RMS a2_pol, tie-break по доле строк хуже q0"))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    tmp = out_path + f".tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, out_path)
    # ПРОВЕРКА ОБРАТНОГО ЧТЕНИЯ. Файл читается тем же способом, которым его
    # прочтёт роллаут, и веса сверяются побитово: молчаливо испорченный
    # чекпойнт иначе обнаружился бы только в следующем эксперименте.
    back = torch.load(out_path, map_location="cpu", weights_only=False)
    checked = []
    for tag, saved, want_sha in (
            ("state", payload["state"], sel_sha),
            ("states.q1", payload["states"]["q1"], sel_sha_level[1]),
            ("states.q2", payload["states"]["q2"], sel_sha_level[2])):
        got = back["state"] if tag == "state" else back["states"][
            tag.split(".")[1]]
        if set(got) != set(saved):
            raise SystemExit(f"после чтения набор весов {tag} другой")
        for k_, v_ in saved.items():
            if not torch.equal(got[k_], v_):
                raise SystemExit(f"после чтения {tag}.{k_} изменился")
        back_sha = k14c.state_sha({k_: v_.float().numpy()
                                   for k_, v_ in got.items()})
        if back_sha != want_sha:
            raise SystemExit(f"отпечаток {tag} после чтения {back_sha} "
                             f"против {want_sha}")
        checked.append(f"{tag}={back_sha}")
    print(f"  сохранено: {out_path} (обратное чтение сошлось: "
          + ", ".join(checked) + ")")
    if a.summary:
        light = {k: v for k, v in payload.items()
                 if k not in ("state", "confirm")}
        light["confirm"] = {k: v for k, v in confirm.items()
                            if k != "rows_by_path"}
        os.makedirs(os.path.dirname(os.path.abspath(a.summary)) or ".",
                    exist_ok=True)
        t_ = a.summary + f".tmp.{os.getpid()}"
        json.dump(light, open(t_, "w"), ensure_ascii=False, indent=1,
                  default=str)
        os.replace(t_, a.summary)
        print(f"  сводка: {a.summary}")
    if a.smoke:
        print("  РЕЖИМ SMOKE: данные урезаны, решения не принимаются, "
              "голова непригодна")
        return 0
    # КОД ВОЗВРАТА ПО ПРИГОДНОСТИ УРОВНЕЙ, А НЕ ПО ОБЩЕМУ СПИСКУ ОТКАЗОВ.
    # Пригодный q1 при сломанном q2 — успех, а не отказ: это рабочая точка
    # на 18 слоях.
    if candidate_level != "none":
        if failed:
            print(f"  КАНДИДАТ {candidate_level} ПРИГОДЕН; у другого уровня "
                  f"отказы: {', '.join(failed)}")
        return 0
    if all(blocked_level[lv] for lv in (1, 2)):
        suspect = [k for k in failed_tech
                   if k.startswith(("decoder_support", "align_not_broken"))]
        print("  РОЛЛАУТ ЗАБЛОКИРОВАН НА ОБОИХ УРОВНЯХ: "
              + ", ".join(failed_tech)
              + ". Чекпойнт сохранён с accepted=false. Это результат "
                "обучения, а не испорченный артефакт: провенанс, инварианты "
                "и конечность проверены и прошли."
              + (f" Но по {', '.join(suspect)} под вопросом и само "
                 f"офлайновое число: улучшение RMS могло быть "
                 f"экстраполяцией декодера или чтением наугад."
                 if suspect else ""))
        return 3
    print("  ТЕХНИЧЕСКИ ИСПРАВНО, НО НИ ОДИН УРОВЕНЬ НЕ ЛУЧШЕ ЧЕРНОВИКА: "
          + ", ".join(failed_cand or failed)
          + ". Это ОТРИЦАТЕЛЬНЫЙ РЕЗУЛЬТАТ эксперимента")
    return 4


if __name__ == "__main__":
    sys.exit(main())
