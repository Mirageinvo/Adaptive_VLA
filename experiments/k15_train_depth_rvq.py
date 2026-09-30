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
import json
import math
import os
import sys
import time

import numpy as np

H_EXEC = 8            # исполняемых позиций чанка, как в K-14
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


def check_init_gate(path, *, expect, file_sha):
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
    if not causal or not all(bool(v) for v in causal.values()):
        raise SystemExit(
            "в гейте K-15a нет причинных проверок feedback или они не "
            "пройдены: тождественность при нулевых весах ничего не "
            "доказывает про подключение путей")
    n0 = sum(1 for k in causal if k.endswith("feedback0_changes_q1"))
    n1 = sum(1 for k in causal if k.endswith("feedback1_changes_q2"))
    if n0 < 1 or n1 < 1:
        raise SystemExit(
            f"причинные проверки неполны: feedback0 {n0}, feedback1 {n1}")
    bad = [f"{k}: гейт {g.get(k)}, сейчас {expect.get(k)}"
           for k in sorted(expect) if str(g.get(k)) != str(expect.get(k))]
    if bad:
        raise SystemExit("гейт K-15a снят в другой обстановке: "
                         + "; ".join(bad))
    return dict(init_gate=path, init_gate_sha1=file_sha(path),
                init_gate_run_id=g.get("run_id"),
                init_gate_causal=len(causal))


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

    `paths` — словарь пяти действий и четырёх наборов логитов:
        a0, a1_tok, a1_pol, a2_tok, a2_pol,
        q1_tok_logits, q1_pol_logits, q2_tok_logits, q2_pol_logits,
        q1_pol_probs, q2_pol_probs.

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
    l_align = w_align * (align1["total"] + align2["total"]) / log_v

    # МОНОТОННОСТЬ МЯГКАЯ И С МАЛЫМ ВЕСОМ. Imitation error — слабый прокси
    # поведения (§48.4, §50.4); большой вес толкал бы модель держаться ближе
    # к q0, то есть к измеренному нулю.
    mono1 = tok.monotonic_hinge(rows["a0"].detach(), rows["a1_pol"])
    mono2 = tok.monotonic_hinge(rows["a1_pol"].detach(), rows["a2_pol"])
    l_mono = w_mono * (mono1 + mono2) / norm

    l_usage = w_usage * (tok.usage_kl_to_uniform(paths["q1_pol_probs"])
                         + tok.usage_kl_to_uniform(paths["q2_pol_probs"])
                         ) / log_v

    total = l_action + l_align + l_mono + l_usage
    parts = dict(
        total=total, action=l_action, align=l_align, mono=l_mono,
        usage=l_usage, norm=norm,
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
        return dict(abs_residual=float(num.mean()),
                    rel_residual=float((num / den).mean()),
                    rel_residual_p95=float(
                        torch.quantile((num / den).flatten(), 0.95)))


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


def select_epoch(history):
    """Выбор эпохи: минимум q2 policy action RMS на val_sel.

    Tie-break — меньшая доля строк, где q2 хуже q0. Правило записано до
    данных; эпоха 0 (без обучения) участвует, как в K-14.
    """
    if not history:
        raise SystemExit("история пуста")
    best = min(history, key=lambda r: (round(float(r["val_a2_pol_rms"]), 12),
                                       round(float(r["val_frac_worse"]), 12),
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
        p["q1_pol_probs"] = torch.softmax(p["q1_pol_logits"], dim=-1)
        p["q2_pol_probs"] = torch.softmax(p["q2_pol_logits"], dim=-1)
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
    st = decoder_support(exact, None, torch, qs, fake_nearest,
                         fake_contribution)
    assert st["abs_residual"] < 1e-6 and st["rel_residual"] < 1e-6
    # латент ВНЕ книги -> остаток заметный
    off = exact + 0.7
    st2 = decoder_support(off, None, torch, qs, fake_nearest,
                          fake_contribution)
    assert st2["rel_residual"] > 0.1, st2
    assert st2["rel_residual_p95"] >= st2["rel_residual"] * 0.5

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
    hist = [dict(epoch=0, val_a2_pol_rms=0.20, val_frac_worse=0.5),
            dict(epoch=1, val_a2_pol_rms=0.15, val_frac_worse=0.4),
            dict(epoch=2, val_a2_pol_rms=0.15, val_frac_worse=0.3),
            dict(epoch=3, val_a2_pol_rms=0.18, val_frac_worse=0.1)]
    ep, best = select_epoch(hist)
    assert ep == 2, ep                     # tie-break по доле ухудшений
    assert best["val_frac_worse"] == 0.3
    hist2 = [dict(epoch=0, val_a2_pol_rms=0.1, val_frac_worse=0.2)]
    assert select_epoch(hist2)[0] == 0     # эпоха 0 участвует

    # --- ГЕЙТ K-15a: КАЖДАЯ МУТАЦИЯ ОТВЕРГАЕТСЯ ---------------------------
    import tempfile
    expect = {"joint_sha1": "J", "q1_sha1": "Q", "device": "cuda:1"}
    good = dict(kind="k15_init_identity", passed=True, git_dirty=False,
                run_id="R", causal_checks={"batch0.feedback0_changes_q1": True,
                                           "batch0.feedback1_changes_q2": True},
                **expect)
    with tempfile.TemporaryDirectory() as td:
        def w(obj, nm="g.json"):
            q = os.path.join(td, nm)
            json.dump(obj, open(q, "w"))
            return q
        info = check_init_gate(w(good), expect=expect,
                               file_sha=lambda _p: "SH")
        assert info["init_gate_causal"] == 2 and info["init_gate_run_id"] == "R"
        for patch, why in (
                ({"passed": False}, "не пройден"),
                ({"kind": "x"}, "описывает"),
                ({"git_dirty": True}, "незакоммиченном"),
                ({"causal_checks": {}}, "причинных проверок"),
                ({"causal_checks": {"batch0.feedback0_changes_q1": False,
                                    "batch0.feedback1_changes_q2": True}},
                 "причинных проверок"),
                ({"causal_checks": {"batch0.feedback0_changes_q1": True}},
                 "неполны"),
                ({"joint_sha1": "ДРУГОЙ"}, "другой обстановке")):
            try:
                check_init_gate(w(dict(good, **patch), "m.json"),
                                expect=expect, file_sha=lambda _p: "SH")
            except SystemExit as e:
                assert why in str(e), (why, e)
            else:
                raise AssertionError(f"гейт принят при {patch}")
        try:
            check_init_gate("", expect=expect, file_sha=lambda _p: "SH")
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
    ap.add_argument("--tau-tokenizer", type=float, default=1.0)
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
    if a.seed is None:
        raise SystemExit("--seed задаётся явно, умолчания у него нет")
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
    from depth_aligned_joint12 import make_depth_aligned_joint12_class
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

    q0_can, _q0_def, q0_man, q0_prov = kc.load_canonical_q0(
        a.q0, gate_r_path=a.gate_r, n_obs=N, keys_sha=keys_sha,
        cache_meta_sha1=k11a.file_sha1(f"{a.cache}.meta.json"))
    plan = kc.load_plan(a.q0, q0_man)
    parts = {}
    for name, po, sel in plan:
        parts.setdefault(name, []).append((po, sel))
    if a.smoke and a.limit:
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

    # --- ГЕЙТ K-15a ОБЯЗАТЕЛЕН -------------------------------------------
    gate_info = check_init_gate(
        a.init_gate,
        expect=dict(joint_sha1=joint_sha,
                    q1_sha1=k11a.file_sha1(a.q1_init),
                    plan_sha1=q0_prov["plan_sha1"],
                    compute_dtype=a.dtype,
                    device=str(dev)),
        file_sha=k11a.file_sha1)
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

    def decode_fp32(latent):
        """ЕДИНЫЙ КОНТЕКСТ: fp32, autocast выключен явно.

        Тот же контекст заверил гейт K-15a. Градиент через декодер нужен —
        он замороженный, но по нему течёт градиент к книгам и головам, —
        поэтому no_grad здесь НЕ ставится.
        """
        with torch.autocast(device_type=dev.type, enabled=False):
            x, _ = codec._decode(latent.float(), embodiment_ids=0)
            return x[..., :7].float()

    ac16 = torch.autocast(device_type=dev.type, dtype=dt)
    run_batch_checked = [False]

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
        # q0 СВЕРЯЕТСЯ ПОБИТОВО НА КАЖДОМ БАТЧЕ. Расхождение означает, что
        # черновик поехал, и всё обучение относится не к тому q0.
        q0_now = out["pred_codes"][0].detach().cpu().numpy()
        bad = int((q0_now != q0_can[sel]).sum())
        if bad:
            raise SystemExit(
                f"q0 разошёлся с каноническим в {bad} позициях: обучение "
                f"относилось бы к другому черновику")
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
        q1_tok_logits = tok.tokenizer_logits(r1, c1,
                                             temperature=float(a.tau_tokenizer))
        c1_tok, _i1, _p1 = tok.hard_straight_through(
            q1_tok_logits, c1, temperature=float(a.tau_tokenizer))
        # ПРЕФИКС ДЛЯ ВТОРОГО УРОВНЯ — ФАКТИЧЕСКИЙ q1 МОДЕЛИ, И ОН
        # DETACH-НУТ: иначе токенизатор q2 уменьшал бы собственную задачу,
        # двигая C1. C1 всё равно получает градиент от q1-путей.
        hard_c1 = c1[out["pred_codes"][1]].detach()
        r2 = z_e - z0.detach() - hard_c1
        q2_tok_logits = tok.tokenizer_logits(r2, c2,
                                             temperature=float(a.tau_tokenizer))
        c2_tok, _i2, _p2 = tok.hard_straight_through(
            q2_tok_logits, c2, temperature=float(a.tau_tokenizer))
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
        paths = dict(
            a0=decode_fp32(z0),
            a1_tok=decode_fp32(z0 + c1_tok),
            a1_pol=decode_fp32(z0 + c1_pol),
            a2_tok=decode_fp32(z0 + hard_c1 + c2_tok),
            a2_pol=decode_fp32(out["cumulative_latents"][2]),
            q1_tok_logits=q1_tok_logits,
            q1_pol_logits=out["logits"][1].float(),
            q2_tok_logits=q2_tok_logits,
            q2_pol_logits=out["logits"][2].float(),
            q1_pol_probs=out["policy_probabilities"][1],
            q2_pol_probs=out["policy_probabilities"][2])
        total, lparts, rows, means = build_losses(
            paths, action, weights_gate, torch, tok, vocab=vocab)
        with torch.no_grad():
            # ТА ЖЕ ВЕЛИЧИНА В ВЕСАХ K-14 — для сравнимости с историей, где
            # обучающая потеря считалась равновесной.
            _r_eq, eq = weighted_row_error(paths["a2_pol"], action,
                                           weights, torch)
            stat = dict(
                rows=int(len(sel)),
                a0=float(means["a0"]), a1_tok=float(means["a1_tok"]),
                a1_pol=float(means["a1_pol"]), a2_tok=float(means["a2_tok"]),
                a2_pol=float(means["a2_pol"]), a2_pol_equal_weights=float(eq),
                loss=float(total), loss_action=float(lparts["action"]),
                loss_align=float(lparts["align"]),
                loss_mono=float(lparts["mono"]),
                loss_usage=float(lparts["usage"]),
                frac_worse_than_q0=float(
                    (rows["a2_pol"] > rows["a0"]).float().mean()),
                frac_nonzero_q1=float(
                    (c1_pol.abs().sum(-1) > 0).float().mean()),
                q1_codes=out["pred_codes"][1].detach().cpu().numpy(),
                q2_codes=out["pred_codes"][2].detach().cpu().numpy(),
                row_a0=rows["a0"].detach().cpu().numpy(),
                row_a1_pol=rows["a1_pol"].detach().cpu().numpy(),
                row_a2_pol=rows["a2_pol"].detach().cpu().numpy(),
                row_a1_tok=rows["a1_tok"].detach().cpu().numpy(),
                row_a2_tok=rows["a2_tok"].detach().cpu().numpy(),
                latent=out["cumulative_latents"][2].detach())
        return (total if train else None), stat

    def evaluate(batch_list, tag):
        """Жёсткий вывод на части: та же величина, что и в отборе эпохи."""
        model.eval()
        acc = {k: 0.0 for k in ("a0", "a1_pol", "a2_pol", "a2_pol_eq",
                                "frac_worse")}
        n_rows = 0
        q1_all, q2_all, sup = [], [], []
        with torch.no_grad():
            for po, sel in batch_list:
                _l, stat = run_batch(po, sel, False)
                w = float(stat["rows"])
                n_rows += stat["rows"]
                acc["a0"] += stat["a0"] * w
                acc["a1_pol"] += stat["a1_pol"] * w
                acc["a2_pol"] += stat["a2_pol"] * w
                acc["a2_pol_eq"] += stat["a2_pol_equal_weights"] * w
                acc["frac_worse"] += stat["frac_worse_than_q0"] * w
                q1_all.append(stat["q1_codes"].reshape(-1))
                q2_all.append(stat["q2_codes"].reshape(-1))
                if len(sup) < 8:
                    sup.append(decoder_support(
                        stat["latent"], codec, torch, quantizers,
                        nearest_code, code_contribution))
        n = max(n_rows, 1)
        res = {k: v / n for k, v in acc.items()}
        res["rms_a2_pol"] = float(np.sqrt(res["a2_pol"]))
        res["rms_a0"] = float(np.sqrt(res["a0"]))
        res["rms_a1_pol"] = float(np.sqrt(res["a1_pol"]))
        res["usage_q1"] = tok.code_usage_stats(
            torch.from_numpy(np.concatenate(q1_all)), vocab)
        res["usage_q2"] = tok.code_usage_stats(
            torch.from_numpy(np.concatenate(q2_all)), vocab)
        res["decoder_support"] = {
            k: float(np.mean([s[k] for s in sup])) for k in sup[0]} if sup \
            else None
        res["rows"] = n_rows
        print(f"    {tag}: RMS a0 {res['rms_a0']:.6f} -> a1_pol "
              f"{res['rms_a1_pol']:.6f} -> a2_pol {res['rms_a2_pol']:.6f}; "
              f"хуже q0 {100 * res['frac_worse']:.1f}% строк")
        print(f"      книги: q1 perplexity {res['usage_q1']['perplexity']:.1f}, "
              f"мёртвых {res['usage_q1']['dead_codes']}, макс доля "
              f"{res['usage_q1']['max_code_share']:.3f}; q2 perplexity "
              f"{res['usage_q2']['perplexity']:.1f}, мёртвых "
              f"{res['usage_q2']['dead_codes']}")
        if res["decoder_support"]:
            print(f"      опора декодера: относительный остаток "
                  f"{res['decoder_support']['rel_residual']:.4f} "
                  f"(p95 {res['decoder_support']['rel_residual_p95']:.4f})")
        return res

    # --- ОБУЧЕНИЕ ---------------------------------------------------------
    history = []
    val0 = evaluate(parts["val_sel"], "эпоха 0, без обучения")
    history.append(dict(epoch=0, train_loss=None,
                        val_a2_pol_rms=val0["rms_a2_pol"],
                        val_frac_worse=val0["frac_worse"], val=val0))
    snapshots = {0: {k: v.detach().clone()
                     for k, v in model.state_dict().items()
                     if k in set(info["names"])}}
    forecast = None
    t_start = time.time()
    order = list(parts["train"])
    for epoch in range(1, int(a.epochs) + 1):
        model.train()
        rng = np.random.default_rng(int(a.seed) + epoch)
        idx = rng.permutation(len(order))
        run_loss, nb, t_ep = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for step, j in enumerate(idx, start=1):
            po, sel = order[j]
            loss, stat = run_batch(po, sel, True)
            (loss / float(a.accum)).backward()
            run_loss += float(loss.detach())
            nb += 1
            if step % int(a.accum) == 0 or step == len(idx):
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
                print(f"    эпоха {epoch}: батч {step}/{len(idx)}, потеря "
                      f"{run_loss / nb:.5f}, {el:.1f} мин, осталось "
                      f"{el * (len(idx) - step) / max(step, 1):.0f} мин",
                      flush=True)
        val = evaluate(parts["val_sel"], f"эпоха {epoch}")
        history.append(dict(epoch=epoch, train_loss=run_loss / max(nb, 1),
                            val_a2_pol_rms=val["rms_a2_pol"],
                            val_frac_worse=val["frac_worse"], val=val))
        snapshots[epoch] = {k: v.detach().clone()
                            for k, v in model.state_dict().items()
                            if k in set(info["names"])}
        print(f"  эпоха {epoch}: потеря {run_loss / max(nb, 1):.5f}, "
              f"val_sel RMS a2_pol {val['rms_a2_pol']:.6f}")

    best_epoch, best = select_epoch(history)
    with torch.no_grad():
        own = dict(model.state_dict())
        for k_, v_ in snapshots[best_epoch].items():
            own[k_].copy_(v_)
    print(f"\n  выбрана эпоха {best_epoch} по val_sel RMS a2_pol "
          f"({best['val_a2_pol_rms']:.6f}), доля ухудшений "
          f"{best['val_frac_worse']:.3f}")
    sel_sha = k14c.state_sha({k_: model.state_dict()[k_].detach().float()
                              .cpu().numpy() for k_ in info["names"]})

    payload = dict(
        kind=("k15_smoke" if a.smoke else "k15_depth_rvq"), stage="q1q2",
        variant="depth_aligned", seed=int(a.seed), epochs=int(a.epochs),
        batch=int(a.batch), accum=int(a.accum), lr=float(a.lr),
        wd=float(a.wd), grip_weight=float(a.grip_weight),
        tau_tokenizer=float(a.tau_tokenizer), tau_policy=float(a.tau_policy),
        loss_weights=dict(a1_pol=W_A1_POL, a1_tok=W_A1_TOK, a2_pol=W_A2_POL,
                          a2_tok=W_A2_TOK, align=W_ALIGN, mono=W_MONO,
                          usage=W_USAGE, eps_norm=EPS_NORM),
        channel_weights="metric (max_act_q, grip=grip_weight)",
        decode_context="fp32, autocast disabled",
        state=({k_: v_.detach().cpu() for k_, v_ in model.state_dict().items()
                if k_ in set(info["names"])} if not a.smoke else None),
        trainable_names=info["names"], selected_epoch=best_epoch,
        selected_state_sha1=sel_sha, history=history, forecast=forecast,
        q0_prov=q0_prov, joint_sha1=joint_sha,
        q1_init=os.path.abspath(a.q1_init),
        q1_init_sha1=k11a.file_sha1(a.q1_init), **q1_prov, **gate_info,
        git_head=git_head, git_dirty=bool(dirty),
        val_confirm="НЕ ОТКРЫВАЛАСЬ; в K-14 уже прочитана, поэтому любое её "
                    "использование будет retrospective",
        device=str(dev), compute_dtype=a.dtype,
        torch_version=str(torch.__version__),
        note=("подтверждающая половина не формировалась; отбор эпохи по "
              "val_sel RMS a2_pol, tie-break по доле строк хуже q0"))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    tmp = out_path + f".tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, out_path)
    print(f"  сохранено: {out_path}")
    if a.summary:
        light = {k: v for k, v in payload.items() if k != "state"}
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


if __name__ == "__main__":
    sys.exit(main())
