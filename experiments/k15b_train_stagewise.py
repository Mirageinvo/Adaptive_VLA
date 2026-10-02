#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15b: поэтапное обучение. Фаза `q1_reader` — научить h18 читать книгу.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ K-15. В K-15 книга и читатель обучались вместе, и
прогон пришёл к почти тождественному политическому пути: словарь стал
вдвое лучше (`a1_tok` 0.084 -> 0.055), а согласие читателя с ним упало с
13.7 % до 0.8 %. Здесь книга ЗАМОРОЖЕНА, обучается только читатель, и
задача его обучения — основная, а не побочная.

ЧИСЛО, КОТОРОЕ НАДО ПОДНЯТЬ. На полной `val_sel` при черновике 0.145469
учитель (rank-path по замороженной исходной книге) даёт 0.072892, а
нынешний читатель 0.142997. То есть он берёт

    capture = (RMS(a0) - RMS(a1_pol)) / (RMS(a0) - RMS(a1_rank)) = 3.4 %

доступного разрыва. Это и есть предмет фазы.

ПОРОГИ ОБЪЯВЛЕНЫ ДО ЗАПУСКА (02.10.2026), И ИХ ДВА, РАЗДЕЛЬНЫХ:

    технический гейт   RMS(a1_pol) <= 0.99 * RMS(a0)
    научный go/no-go   capture >= 0.20 И строго больше capture эпохи 0

Технический порог исходная модель проходит БЕЗ обучения (её улучшение
1.7 %), поэтому сам по себе он ничего не доказывает — отсюда второй.
`capture >= 0.20` при текущих числах означает RMS(a1_pol) <= 0.130954.

ДВЕ РАЗНЫЕ ОСИ, И ИХ НЕЛЬЗЯ СМЕШИВАТЬ:

    кросс-энтропия читателя -> ВСЕ 16 КОДОВЫХ позиций
    action-член и метрика   -> первые 8 ВРЕМЕННЫХ ШАГОВ действия

`PerceiverDecoder` смотрит на все латентные токены через cross-attention
без каузальной маски, поэтому коды поздних позиций влияют и на первые шаги
действия: малая ошибка траектории получена совместным действием всех
шестнадцати кодов, и учить только первым восьми значило бы выбросить часть
того, чем она получена.

ФАЗЫ. Реализована только `q1_reader`. `c2_book` требует
ДИФФЕРЕНЦИРУЕМОЙ опоры декодера — нынешняя `decoder_support` целиком под
`no_grad` и в потерю не годится, — а `q2_reader` имеет смысл только после
неё. Обе отвергаются явным отказом, а не заглушкой.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

# --- ПОРОГИ, ОБЪЯВЛЕННЫЕ ДО ДАННЫХ (02.10.2026) ----------------------
TECH_IMPROVE_FACTOR = 0.99      # RMS(a1_pol) <= 0.99 * RMS(a0)
CAPTURE_THRESHOLD = 0.20        # научный go/no-go
LAMBDA_TARGET_RATIO = 0.175     # норма градиента action / CE на головах
COLLAPSE_MAX_SHARE, COLLAPSE_MIN_PPL = 0.98, 2.0
ACTION_RANGE_FACTOR, ACTION_CLIP_BOUND = 1.5, 1.5

PHASES = ("q1_reader", "c2_book", "q2_reader")
PHASE_WHITELIST = {
    "q1_reader": ("depth_rvq_norms.0.", "depth_rvq_heads.0.",
                  "depth_rvq_feedback.0."),
    "c2_book": ("depth_aligned_c2",),
    "q2_reader": ("depth_rvq_norms.1.", "depth_rvq_heads.1.",
                  "depth_rvq_feedback.1."),
}
# Поля кэша, которые тренер ОБЯЗАН сверить. `builder_sha1` здесь нет
# намеренно: правка печати в построителе цель не меняет.
CACHE_CONTRACT = ("c1_sha1", "topk", "action_error_positions",
                  "rank_candidates_file_sha1", "plan_sha1",
                  "teacher_target", "part")


def capture(rms_draft, rms_teacher, rms_policy):
    """Доля разрыва до учителя, забранная читателем. Чистая функция.

    None, когда разрыв неположителен: тогда доля ничего не означает, и в
    K-15 такая формула выдавала +280 %.
    """
    gap = float(rms_draft) - float(rms_teacher)
    if gap <= 1e-12:
        return None
    return float((float(rms_draft) - float(rms_policy)) / gap)


def phase_names(all_names, phase):
    """Имена обучаемого для фазы — подмножество общего белого списка.

    Сужение делается ЗДЕСЬ, а не в архитектурном файле: он входит в
    `ARCHITECTURE_FILES`, и его правка обесценила бы гейт K-15a.
    """
    if phase not in PHASE_WHITELIST:
        raise SystemExit(f"фаза {phase!r} не бывает: {sorted(PHASE_WHITELIST)}")
    prefixes = PHASE_WHITELIST[phase]
    picked = sorted(n for n in all_names if n.startswith(prefixes))
    if not picked:
        raise SystemExit(
            f"фаза {phase}: ни одно имя из {all_names[:3]}... не начинается "
            f"с {prefixes}")
    missing = [p for p in prefixes
               if not any(n.startswith(p) for n in picked)]
    if missing:
        raise SystemExit(f"фаза {phase}: нет тензоров для {missing}")
    return picked


def check_cache_contract(meta, book, ctx_q0_prov, plan_sha1, topk):
    """Кэш осмыслен только для своей книги и своей обстановки. Fail-closed."""
    problems = []
    missing = [f for f in CACHE_CONTRACT if f not in meta]
    if missing:
        problems.append(f"в кэше нет обязательных полей {missing}")

    def same(name, got, want):
        if got != want:
            problems.append(f"{name}: в кэше {got!r}, сейчас {want!r}")

    same("kind", meta.get("kind"), "k15b_rankpath_target")
    same("part", meta.get("part"), "train")
    same("teacher_target", meta.get("teacher_target"),
         "action_best_rankpath")
    same("c1_sha1", meta.get("c1_sha1"), book.get("c1_sha1"))
    same("topk", meta.get("topk"), int(topk))
    same("plan_sha1", meta.get("plan_sha1"), plan_sha1)
    for key in ("plan_sha1", "q0_manifest_sha1"):
        same(f"q0_prov.{key}", (meta.get("q0_prov") or {}).get(key),
             ctx_q0_prov.get(key))
    # ЦЕЛЬ ЗАВИСИТ ОТ МЕТРИКИ, ПО КОТОРОЙ ВЫБРАН РАНГ.
    tm = meta.get("teacher_metric") or {}
    if not tm:
        problems.append("в кэше нет teacher_metric")
    return problems, tm


def select_epoch(history):
    """Эпоха по минимуму RMS исполняемого пути; tie-break по CE.

    `capture` монотонно убывает по RMS при фиксированных черновике и
    учителе на той же части, поэтому минимум RMS и максимум capture — это
    одна и та же эпоха. Эпоха 0 участвует, как в K-14 и K-15, и именно
    поэтому нужен второй, научный порог.
    """
    if not history:
        raise SystemExit("история пуста")
    best = min(history, key=lambda r: (round(float(r["val_rms_a1_pol"]), 12),
                                       round(float(r["val_ce"]), 12),
                                       int(r["epoch"])))
    return int(best["epoch"]), best


def lambda_from_norms(norm_ce, norm_action, target_ratio=LAMBDA_TARGET_RATIO):
    """lambda, при которой норма action-градиента составит target_ratio от CE.

    Калибровка одна, сетки нет. При нулевом action-градиенте — отказ: это
    значит, что action-член до головы не доходит вовсе.
    """
    if not (np.isfinite(norm_ce) and np.isfinite(norm_action)):
        raise ValueError(f"нормы не числа: {norm_ce}, {norm_action}")
    if norm_action <= 0.0:
        raise ValueError(
            "норма action-градиента нулевая: член не влияет на головы")
    if norm_ce <= 0.0:
        raise ValueError("норма CE-градиента нулевая: цель не доходит")
    return float(target_ratio * norm_ce / norm_action)


def collapse_gate(usage):
    """Схлопывание книги: общее и худшее по КОДОВЫМ позициям.

    Позиции здесь КОДОВЫЕ, их 16, и берутся ВСЕ. Брать первые H_EXEC
    нельзя: H_EXEC = 8 относится к шагам ДЕЙСТВИЯ, а поздние коды влияют
    на первые действия через глобальный cross-attention, поэтому
    схлопывание позиций 8-15 пропускать нельзя.
    """
    bp = usage["by_position"]
    return dict(
        max_code_share=float(usage["max_code_share"]),
        perplexity=float(usage["perplexity"]),
        used_codes=int(usage["used_codes"]),
        position_max_code_share=bp["position_max_code_share"],
        position_min_perplexity=bp["position_min_perplexity"],
        positions=bp["positions"],
        limit_max_share=COLLAPSE_MAX_SHARE,
        limit_min_perplexity=COLLAPSE_MIN_PPL,
        passed=bool(usage["max_code_share"] <= COLLAPSE_MAX_SHARE
                    and usage["perplexity"] >= COLLAPSE_MIN_PPL
                    and bp["position_max_code_share"] <= COLLAPSE_MAX_SHARE
                    and bp["position_min_perplexity"] >= COLLAPSE_MIN_PPL))


def range_gate(p99_candidate, p99_dataset, absmax_candidate):
    """Диапазон действий: поканальный p99 И предел нормированной шкалы."""
    by_p99 = all(float(c) <= ACTION_RANGE_FACTOR * float(d)
                 for c, d in zip(p99_candidate, p99_dataset))
    by_clip = all(float(m) <= ACTION_CLIP_BOUND for m in absmax_candidate)
    return dict(passed=bool(by_p99 and by_clip), by_p99=bool(by_p99),
                by_clip=bool(by_clip), p99_candidate=list(p99_candidate),
                p99_dataset=list(p99_dataset),
                absmax_candidate=list(absmax_candidate),
                factor=ACTION_RANGE_FACTOR, clip_bound=ACTION_CLIP_BOUND)


def verdict(gates, capture_value, capture_epoch0):
    """Исход фазы. Чистая функция, разделяющая три разных случая."""
    blockers = sorted(k for k, v in gates.items()
                      if v["category"] == "rollout_blocker"
                      and not v["passed"])
    technical = gates["improves_q1"]["passed"]
    if capture_value is None:
        return dict(code=3, outcome="разрыв до учителя неположителен",
                    blockers=blockers)
    cap_ok = (capture_value >= CAPTURE_THRESHOLD
              and (capture_epoch0 is None
                   or capture_value > capture_epoch0 + 1e-12))
    if blockers:
        return dict(code=3, outcome="роллаут заблокирован",
                    blockers=blockers, capture=float(capture_value),
                    capture_passed=bool(cap_ok))
    if not technical:
        return dict(code=4, outcome="технический порог не пройден",
                    blockers=[], capture=float(capture_value),
                    capture_passed=bool(cap_ok))
    if not cap_ok:
        return dict(
            code=4,
            outcome=("читатель не забрал объявленную долю разрыва: "
                     "отрицательный результат, реализацию закрываем"),
            blockers=[], capture=float(capture_value),
            capture_passed=False, threshold=CAPTURE_THRESHOLD,
            capture_epoch0=capture_epoch0)
    return dict(code=0, outcome="фаза пройдена", blockers=[],
                capture=float(capture_value), capture_passed=True,
                threshold=CAPTURE_THRESHOLD, capture_epoch0=capture_epoch0)


def selftest():
    # --- ДОЛЯ РАЗРЫВА ---------------------------------------------------
    assert abs(capture(0.145469, 0.072892, 0.142997) - 0.034061) < 1e-5
    assert capture(0.1, 0.1, 0.09) is None
    assert capture(0.1, 0.12, 0.09) is None        # учитель хуже черновика
    # ОБЪЯВЛЕННЫЙ ПОРОГ СООТВЕТСТВУЕТ ИМЕННО ЭТОМУ RMS
    target_rms = 0.145469 - CAPTURE_THRESHOLD * (0.145469 - 0.072892)
    assert abs(target_rms - 0.130954) < 1e-6, target_rms
    assert abs(capture(0.145469, 0.072892, target_rms)
               - CAPTURE_THRESHOLD) < 1e-9

    # --- БЕЛЫЙ СПИСОК ФАЗЫ ----------------------------------------------
    allnames = [
        "depth_aligned_c1", "depth_aligned_c2",
        "depth_rvq_norms.0.weight", "depth_rvq_norms.1.weight",
        "depth_rvq_heads.0.weight", "depth_rvq_heads.1.weight",
        "depth_rvq_feedback.0.proj.weight",
        "depth_rvq_feedback.1.proj.weight"]
    p1 = phase_names(allnames, "q1_reader")
    assert p1 == ["depth_rvq_feedback.0.proj.weight",
                  "depth_rvq_heads.0.weight",
                  "depth_rvq_norms.0.weight"], p1
    assert "depth_aligned_c1" not in p1 and "depth_aligned_c2" not in p1
    assert not any(".1." in n for n in p1), p1
    assert phase_names(allnames, "c2_book") == ["depth_aligned_c2"]
    assert phase_names(allnames, "q2_reader") == [
        "depth_rvq_feedback.1.proj.weight", "depth_rvq_heads.1.weight",
        "depth_rvq_norms.1.weight"]
    # ТРИ ФАЗЫ НЕ ПЕРЕСЕКАЮТСЯ И ВМЕСТЕ НЕ НАКРЫВАЮТ C1
    sets = [set(phase_names(allnames, p)) for p in PHASES]
    assert not (sets[0] & sets[1]) and not (sets[1] & sets[2])
    assert not (sets[0] & sets[2])
    assert "depth_aligned_c1" not in set().union(*sets)
    for bad in ("нет такой", "p1"):
        try:
            phase_names(allnames, bad)
        except SystemExit as e:
            assert "не бывает" in str(e), e
        else:
            raise AssertionError(f"принята фаза {bad}")
    try:
        phase_names(["depth_aligned_c1"], "q1_reader")
    except SystemExit as e:
        assert "не начинается" in str(e), e
    else:
        raise AssertionError("принят список без тензоров фазы")

    # --- КОНТРАКТ КЭША: ОТСУТСТВИЕ ПОЛЯ — ОТКАЗ -------------------------
    q0p = {"plan_sha1": "P", "q0_manifest_sha1": "M"}
    book = {"c1_sha1": "B"}
    meta = dict(kind="k15b_rankpath_target", part="train",
                teacher_target="action_best_rankpath", c1_sha1="B", topk=10,
                action_error_positions=8, rank_candidates_file_sha1="R",
                plan_sha1="P", q0_prov=dict(q0p),
                teacher_metric=dict(action_error_positions=8))
    probs, tm = check_cache_contract(meta, book, q0p, "P", 10)
    assert probs == [] and tm["action_error_positions"] == 8, probs
    for field in CACHE_CONTRACT:
        broken = {k: v for k, v in meta.items() if k != field}
        assert check_cache_contract(broken, book, q0p, "P", 10)[0], field
    for key, val in (("c1_sha1", "ДРУГАЯ"), ("topk", 5),
                     ("part", "val_sel"), ("plan_sha1", "ИНОЙ"),
                     ("teacher_target", "latent")):
        probs, _tm = check_cache_contract(dict(meta, **{key: val}), book,
                                          q0p, "P", 10)
        assert any(key in p for p in probs), (key, probs)
    probs, _tm = check_cache_contract(
        dict(meta, q0_prov={"plan_sha1": "P", "q0_manifest_sha1": "ИНОЙ"}),
        book, q0p, "P", 10)
    assert any("q0_manifest_sha1" in p for p in probs), probs
    probs, tm = check_cache_contract({k: v for k, v in meta.items()
                                      if k != "teacher_metric"},
                                     book, q0p, "P", 10)
    assert any("teacher_metric" in p for p in probs) and tm == {}

    # --- ВЫБОР ЭПОХИ ----------------------------------------------------
    hist = [dict(epoch=0, val_rms_a1_pol=0.143, val_ce=7.0),
            dict(epoch=1, val_rms_a1_pol=0.120, val_ce=5.0),
            dict(epoch=2, val_rms_a1_pol=0.120, val_ce=4.0)]
    ep, best = select_epoch(hist)
    assert ep == 2 and best["val_ce"] == 4.0, (ep, best)
    assert select_epoch(hist[:1])[0] == 0

    # --- КАЛИБРОВКА LAMBDA ----------------------------------------------
    lam = lambda_from_norms(10.0, 2.0, target_ratio=0.2)
    assert abs(lam - 1.0) < 1e-12, lam
    assert abs(lambda_from_norms(4.0, 4.0) - LAMBDA_TARGET_RATIO) < 1e-12
    for ce, act, why in ((1.0, 0.0, "нулевая"), (0.0, 1.0, "не доходит"),
                         (float("nan"), 1.0, "не числа")):
        try:
            lambda_from_norms(ce, act)
        except ValueError as e:
            assert why in str(e), (why, e)
        else:
            raise AssertionError(f"приняты нормы {ce}, {act}")

    # --- ГЕЙТЫ ----------------------------------------------------------
    usage_ok = dict(max_code_share=0.1, perplexity=50.0, used_codes=300,
                    by_position=dict(positions=16,
                                     position_max_code_share=0.2,
                                     position_min_perplexity=20.0))
    assert collapse_gate(usage_ok)["passed"]
    assert collapse_gate(usage_ok)["positions"] == 16
    # СХЛОПЫВАНИЕ НА ОДНОЙ КОДОВОЙ ПОЗИЦИИ ЛОВИТСЯ, даже если в среднем
    # книга выглядит здоровой
    usage_bad = dict(usage_ok, by_position=dict(
        positions=16, position_max_code_share=0.99,
        position_min_perplexity=20.0))
    assert not collapse_gate(usage_bad)["passed"]
    usage_ppl = dict(usage_ok, by_position=dict(
        positions=16, position_max_code_share=0.2,
        position_min_perplexity=1.5))
    assert not collapse_gate(usage_ppl)["passed"]
    assert range_gate([0.5], [1.0], [0.9])["passed"]
    assert not range_gate([0.5], [1.0], [1.6])["passed"]
    assert not range_gate([2.0], [1.0], [0.9])["passed"]

    # --- ИСХОД ----------------------------------------------------------
    def mk(passed_tech=True, blocker=True):
        return {
            "improves_q1": dict(category="candidate", passed=passed_tech),
            "support_q1": dict(category="rollout_blocker", passed=blocker),
        }

    v = verdict(mk(), 0.35, 0.034)
    assert v["code"] == 0 and v["capture_passed"], v
    v = verdict(mk(), 0.15, 0.034)
    assert v["code"] == 4 and "не забрал" in v["outcome"], v
    # ПОРОГ ПРОЙДЕН, НО НЕ ЛУЧШЕ ЭПОХИ 0 — тоже отрицательный результат
    v = verdict(mk(), 0.30, 0.30)
    assert v["code"] == 4 and not v["capture_passed"], v
    v = verdict(mk(blocker=False), 0.35, 0.034)
    assert v["code"] == 3 and v["blockers"] == ["support_q1"], v
    v = verdict(mk(passed_tech=False), 0.35, 0.034)
    assert v["code"] == 4 and "технический" in v["outcome"], v
    v = verdict(mk(), None, 0.034)
    assert v["code"] == 3, v

    print("самопроверка k15b_train_stagewise пройдена")


def main():
    ap = argparse.ArgumentParser(
        description="K-15b: поэтапное обучение, фаза q1_reader")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--phase", default="q1_reader", choices=PHASES)
    ap.add_argument("--c1", default="data/k15b/c1_selected.pt")
    ap.add_argument("--target", default="data/k15b/rankpath_target_train.npz")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lambda-action", default="auto",
                    help="'auto' — калибровка по отношению норм градиентов "
                         "на головах; число — фиксированное значение")
    ap.add_argument("--calib-batches", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-batches", type=int, default=100)
    ap.add_argument("--out", default="")
    ap.add_argument("--summary", default="")
    ap.add_argument("--report-every", type=int, default=250)
    ap.add_argument("--overwrite", action="store_true")
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import inspect
    import k15_context
    import k15b_probe_and_extract as probe
    import k15b_build_rankpath_cache as cachelib
    from k15_train_depth_rvq import H_EXEC
    k15_context.add_common_arguments(ap)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.phase != "q1_reader":
        raise SystemExit(
            f"фаза {a.phase} не реализована, и порядок после P1 такой: "
            f"(1) проверить ИСХОДНУЮ замороженную C2 относительно "
            f"ФАКТИЧЕСКИ предсказанного q1; (2) если она содержит полезное "
            f"уточнение и проходит опору — заморозить её и обучать "
            f"`q2_reader`; (3) только если исходная C2 непригодна — "
            f"рассматривать обучение книги с ЯВНЫМ ограничением опоры, для "
            f"которого нужна дифференцируемая версия `decoder_support` "
            f"(нынешняя целиком под no_grad). То есть `q2_reader` НЕ "
            f"требует предварительного обучения книги")
    if int(a.limit) != 0:
        raise SystemExit("--limit не применяется: урезание задаётся --smoke")
    out_path = a.out or (f"data/k15b/{a.phase}"
                         f"{'_smoke' if a.smoke else ''}_s{a.seed}.pt")
    summary = a.summary or (f"reports/k15b/{a.phase}"
                            f"{'_smoke' if a.smoke else ''}_s{a.seed}.json")
    for path in (out_path, summary):
        if os.path.exists(path) and not a.overwrite:
            raise SystemExit(f"{path} уже существует: без --overwrite не "
                             f"перезаписываю")
    archived = {p: probe.archive_existing(p) for p in (out_path, summary)}
    for path, dest in archived.items():
        if dest:
            print(f"  прежний {path} перенесён в {dest}")

    ctx = k15_context.build(a)
    torch = ctx.torch
    import torch.nn.functional as F
    dev, dt = ctx.dev, ctx.dt
    model, codec = ctx.model, ctx.codec
    k15t = k15_context.k15t
    ac16 = torch.autocast(device_type=dev.type, dtype=dt)
    vocab = int(ctx.vocab)

    # --- КНИГА: ЗАГРУЗИТЬ И ЗАМОРОЗИТЬ ----------------------------------
    book = torch.load(a.c1, map_location="cpu", weights_only=False)
    if book.get("kind") != "k15b_c1_selected" or book.get("accepted") is not True:
        raise SystemExit(
            f"{a.c1}: kind {book.get('kind')!r}, accepted "
            f"{book.get('accepted')!r}. Книга, не прошедшая гейты probe, "
            f"учителем быть не может")
    for key, want in (("codec", ctx.codec_fp),
                      ("code_version", ctx.code_version),
                      ("joint_sha1", ctx.joint_sha)):
        if book.get(key) != want:
            raise SystemExit(f"{a.c1}: {key} выбран в другой обстановке")

    def book_sha():
        return hashlib.sha1(np.ascontiguousarray(
            model.depth_aligned_book(1).detach().float().cpu().numpy()
        ).tobytes()).hexdigest()[:12]

    if book_sha() != book["c1_sha1"]:
        with torch.no_grad():
            model.depth_aligned_c1.copy_(book["c1"].to(
                model.depth_aligned_c1.device, model.depth_aligned_c1.dtype))
    if book_sha() != book["c1_sha1"]:
        raise SystemExit("после загрузки книга не та")
    print(f"  книга {book['c1_sha1']} из состояния "
          f"{book.get('source_state')!r}, эпоха {book.get('source_epoch')}, "
          f"цель {book.get('teacher_target')}")

    # --- СУЖЕНИЕ ОБУЧАЕМОГО ПОД ФАЗУ ------------------------------------
    all_names = list(ctx.info["names"])
    train_names = phase_names(all_names, a.phase)
    named = dict(model.named_parameters())
    for n_, p_ in named.items():
        p_.requires_grad_(n_ in set(train_names))
    live = sorted(n for n, p in named.items() if p.requires_grad)
    if live != sorted(train_names):
        raise SystemExit(f"обучаемыми оказались {live}, а не {train_names}")
    n_par = sum(named[n].numel() for n in train_names)
    print(f"  фаза {a.phase}: обучаемых {len(train_names)} тензоров, "
          f"{n_par / 1e6:.3f}M параметров; C1, C2 и уровень 2 заморожены")

    frozen_sha, n_frozen, n_elem = k15t.frozen_content_sha(
        model, torch, set(train_names))
    frozen_inv0, _nf = k15t.frozen_invariant(model, torch, set(train_names))
    print(f"  замороженного {n_frozen} тензоров ({n_elem} значений), "
          f"отпечаток {frozen_sha}")
    clashes = k15t.no_alias_between(model, set(train_names))
    if clashes:
        raise SystemExit(f"обучаемое делит хранилище с замороженным: "
                         f"{clashes[:3]}")

    # --- КЭШ ЦЕЛИ -------------------------------------------------------
    if not os.path.exists(a.target):
        raise SystemExit(f"нет {a.target}: сначала построитель кэша")
    cache = np.load(a.target, allow_pickle=True)
    meta = json.loads(str(cache["meta"]))
    # topk БЕРЁТСЯ ИЗ КНИГИ, А НЕ ИЗ САМОГО КЭША: сравнение кэша с собой
    # было бы тавтологией, а цель зависит от topk напрямую.
    if book.get("topk") is None:
        raise SystemExit(f"{a.c1}: нет topk, с чем сверять кэш — неизвестно")
    problems, tmetric = check_cache_contract(
        meta, book, ctx.q0_prov, ctx.q0_prov.get("plan_sha1"),
        int(book["topk"]))
    if problems:
        raise SystemExit("кэш цели не подходит: " + "; ".join(problems[:6]))
    want_weights = [float(x) for x in
                    ctx.weights_gate.detach().cpu().numpy()]
    if [float(x) for x in tmetric.get("channel_weights", [])] != want_weights:
        raise SystemExit(
            f"веса каналов цели {tmetric.get('channel_weights')} против "
            f"{want_weights}: цель выбрана по другой метрике")
    if tmetric.get("decoder_context") != ctx.decoder_context:
        raise SystemExit("контекст декодера цели не тот")
    if meta.get("rank_candidates_file_sha1") != k15_context.sha12(
            inspect.getfile(probe.rank_candidates)):
        raise SystemExit(
            "файл функции кандидатов изменился после построения кэша: "
            "множество кандидатов могло стать другим")
    if int(meta["action_error_positions"]) != int(H_EXEC):
        raise SystemExit(
            f"цель выбрана по {meta['action_error_positions']} шагам, "
            f"сейчас H_EXEC={H_EXEC}")
    # КЭШ ЗАВИСИТ ОТ ДЕКОДЕРА: цель выбиралась по декодированной ошибке.
    for key, want in (("codec", ctx.codec_fp),
                      ("code_version", ctx.code_version),
                      ("joint_sha1", ctx.joint_sha)):
        if meta.get(key) != want:
            raise SystemExit(
                f"кэш: {key} = {meta.get(key)!r}, сейчас {want!r}. Цель "
                f"выбиралась другим декодером")
    # СОДЕРЖИМОЕ МАССИВОВ ПРОВЕРЯЕТСЯ, А НЕ ТОЛЬКО МЕТАДАННЫЕ. Иначе
    # прошли бы переставленные строки при непереставленных кодах, дубли
    # строк (присваивание сработало бы по правилу «последняя победила») и
    # изменённые коды при прежнем content_sha1.
    rows_t = np.asarray(cache["rows"], np.int64)
    codes_t = np.asarray(cache["codes"], np.int64)
    ranks_t = np.asarray(cache["ranks"], np.int16)
    plan_rows, _slot = cachelib.build_row_index(ctx.parts_full["train"])
    if hashlib.sha1(np.ascontiguousarray(plan_rows).tobytes()
                    ).hexdigest()[:12] != meta["rows_sha1"]:
        raise SystemExit("кэш построен по другому набору строк train")
    if not np.array_equal(rows_t, plan_rows):
        raise SystemExit(
            "порядок строк в кэше не совпал с планом: коды относились бы "
            "не к тем строкам")
    if np.unique(rows_t).size != rows_t.size:
        raise SystemExit("в кэше есть повторяющиеся строки")
    if not (len(rows_t) == len(codes_t) == len(ranks_t)):
        raise SystemExit(
            f"длины массивов кэша расходятся: строк {len(rows_t)}, кодов "
            f"{len(codes_t)}, рангов {len(ranks_t)}")
    n_pos = int(codes_t.shape[1])
    if int(meta["code_target_positions"]) != n_pos:
        raise SystemExit("число кодовых позиций в кэше не согласовано")
    if codes_t.shape != (len(plan_rows), n_pos):
        raise SystemExit(f"форма кодов {codes_t.shape}, ожидалась "
                         f"{(len(plan_rows), n_pos)}")
    if int((codes_t < 0).sum()) or int((codes_t >= vocab).sum()):
        raise SystemExit(f"в кэше есть коды вне [0, {vocab})")
    got_content = cachelib.cache_fingerprint(rows_t, codes_t, ranks_t)
    if got_content != meta["content_sha1"]:
        raise SystemExit(
            f"отпечаток содержимого кэша {got_content}, в метаданных "
            f"{meta['content_sha1']}: массивы изменены")
    target_by_row = np.full((int(ctx.N), n_pos), -1, np.int64)
    target_by_row[rows_t] = codes_t
    print(f"  цель: {rows_t.size} строк, {n_pos} кодовых позиций, "
          f"отпечаток {meta['content_sha1']}; кросс-энтропия по ВСЕМ "
          f"{n_pos} позициям, ранг выбран по "
          f"{meta['action_error_positions']} шагам действия")
    print(f"  на train: черновик {meta['rms_draft']:.6f}, латентная цель "
          f"{meta['rms_latent']:.6f}, учитель {meta['rms_rankpath']:.6f}")

    # --- ОПТИМИЗАТОР ----------------------------------------------------
    opt = torch.optim.AdamW([named[n] for n in train_names],
                            lr=float(a.lr), weight_decay=float(a.wd))
    extra, missing = k15t.optimizer_covers_exactly(opt, model,
                                                   set(train_names))
    if extra or missing:
        raise SystemExit(f"оптимизатор не совпал с белым списком фазы: "
                         f"лишних {extra}, нет {missing}")

    mode_dep = []
    for nm_, mod_ in model.named_modules():
        if isinstance(mod_, torch.nn.modules.dropout._DropoutNd):
            if float(getattr(mod_, "p", 0.0)) > 0.0:
                mode_dep.append(f"{nm_}: dropout p={mod_.p}")
        elif isinstance(mod_, torch.nn.modules.batchnorm._BatchNorm):
            mode_dep.append(f"{nm_}: batchnorm")
    if mode_dep:
        raise SystemExit("есть операции, зависящие от режима: "
                         + "; ".join(mode_dep[:5]))
    model.eval()   # обучение в eval, как в K-15

    parts = {k: list(v) for k, v in ctx.parts.items()}
    if a.smoke:
        n_take = max(int(a.smoke_batches), 1)
        parts = {k: v[:min(n_take, len(v))] for k, v in parts.items()}
        print(f"  РЕЖИМ SMOKE: по {len(parts['train'])} батчей на часть, "
              f"решения не принимаются")
    q0_dev = torch.as_tensor(np.asarray(ctx.q0_can), device=dev)

    def forward(po, sel):
        b = ctx.build_batch(po, sel)
        am = b.get("attention_mask")
        with ac16:
            v, p_ids = model.build_inputs(position_offset=po, **b)
            out = model.forward_depth_aligned_rvq(
                vlm_inputs_embeds=v, attention_mask=am, position_ids=p_ids,
                mode="full", tau=1.0)
        bad = int((out["pred_codes"][0].detach()
                   != q0_dev[torch.as_tensor(sel, device=dev)]).sum())
        if bad:
            raise SystemExit(f"q0 разошёлся с каноническим в {bad} позициях")
        return out

    # НОРМИРОВКА ACTION-ЧЛЕНА — ОДИН ФИКСИРОВАННЫЙ СКАЛЯР, А НЕ ОШИБКА
    # ЧЕРНОВИКА НА БАТЧЕ. Батчевая нормировка перевзвешивала батчи — малая
    # ошибка q0 давала больший вес, — и глобальный оптимум расходился с
    # агрегированным RMS, по которому идёт отбор. Масштаб всё равно
    # поглощается калибровкой lambda.
    action_scale = float(meta["rms_draft"]) ** 2

    def losses(po, sel, out, table, table_name):
        """CE по всем КОДОВЫМ позициям + action-член по шагам действия."""
        action = torch.from_numpy(
            np.asarray(ctx.ACT[sel], np.float32)).to(dev)[..., :7]
        tgt = torch.from_numpy(table[np.asarray(sel, np.int64)]).to(dev)
        if int((tgt < 0).sum()):
            raise SystemExit(
                f"в батче есть строки без цели в таблице {table_name}")
        logits = out["logits"][1].float()
        ce = F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1))
        z0 = out["policy_embeddings"][0]
        c1_pol = out["policy_embeddings"][1]
        act_pol = ctx.decode_fp32(z0 + c1_pol)
        _r, m_pol = k15t.weighted_row_error(act_pol, action,
                                            ctx.weights_gate, torch)
        with torch.no_grad():
            _r0, m_draft = k15t.weighted_row_error(
                ctx.decode_fp32(z0), action, ctx.weights_gate, torch)
        # НОРМИРОВКА — ПОСТОЯННАЯ ПО ПАРАМЕТРАМ: она перевзвешивает батчи,
        # но положение минимума внутри батча не двигает.
        act_term = m_pol / action_scale
        return ce, act_term, dict(
            ce=ce, action=act_term, m_pol=m_pol, m_draft=m_draft,
            agree=float((out["pred_codes"][1] == tgt).float().mean()))

    # --- КАЛИБРОВКА LAMBDA ----------------------------------------------
    head_params = [named[n] for n in train_names]
    if str(a.lambda_action).strip().lower() == "auto":
        n_ce, n_act = 0.0, 0.0
        for po, sel in parts["train"][:max(int(a.calib_batches), 1)]:
            out = forward(po, sel)
            ce, act, _d = losses(po, sel, out, target_by_row, "train")
            g_ce = torch.autograd.grad(ce, head_params, retain_graph=True,
                                       allow_unused=True)
            g_act = torch.autograd.grad(act, head_params, retain_graph=False,
                                        allow_unused=True)
            n_ce += float(sum(float(g.norm()) ** 2 for g in g_ce
                              if g is not None) ** 0.5)
            n_act += float(sum(float(g.norm()) ** 2 for g in g_act
                               if g is not None) ** 0.5)
        nb = max(int(a.calib_batches), 1)
        lam = lambda_from_norms(n_ce / nb, n_act / nb)
        lam_report = dict(mode="auto", target_ratio=LAMBDA_TARGET_RATIO,
                          norm_ce=n_ce / nb, norm_action=n_act / nb,
                          batches=nb, lambda_action=lam)
        print(f"  калибровка lambda по ВСЕМУ белому списку фазы "
              f"({len(train_names)} тензоров: нормы, головы, feedback): "
              f"|g_CE| {n_ce / nb:.3e}, |g_action| {n_act / nb:.3e} -> "
              f"lambda {lam:.4g} при целевом НАЧАЛЬНОМ отношении "
              f"{LAMBDA_TARGET_RATIO}. На последующих шагах отношение не "
              f"поддерживается")
    else:
        lam = float(a.lambda_action)
        lam_report = dict(mode="fixed", lambda_action=lam)
        print(f"  lambda задана вручную: {lam}")

    # --- УЧИТЕЛЬ НА val_sel СЧИТАЕТСЯ ОДИН РАЗ --------------------------
    # Книга заморожена и q0 каноничен, поэтому цель на val фиксирована:
    # пересчитывать её каждую эпоху значило бы платить десять декодов на
    # батч за неизменный ответ.
    # ЗДЕСЬ ЖЕ СОБИРАЮТСЯ ЦЕЛЕВЫЕ КОДЫ ДЛЯ val_sel. Кэш покрывает только
    # train, а train и val_sel не пересекаются, поэтому без этого все
    # val-цели равнялись бы -1 и прогон останавливался бы на первой же
    # оценке. Дополнительных декодирований не нужно: `near`, `stack` и
    # `best_j` в этом проходе уже есть.
    print("  считаю учителя на val_sel (один раз) и собираю его цели...")
    t_teacher = time.time()
    acc_t = dict(draft=0.0, latent=0.0, rank=0.0)
    n_rows_val = 0
    val_target_by_row = np.full((int(ctx.N), n_pos), -1, np.int64)
    val_rows_seen = []
    with torch.no_grad():
        for po, sel in parts["val_sel"]:
            out = forward(po, sel)
            z0 = out["policy_embeddings"][0]
            c1 = model.depth_aligned_book(1)
            action = torch.from_numpy(
                np.asarray(ctx.ACT[sel], np.float32)).to(dev)[..., :7]
            z_e = codec._encode(action.float(), embodiment_ids=0).float()
            d1 = ctx.tok.mean_squared_distances(z_e - z0, c1)
            _lg, _emb, i1, _p = ctx.tok.quantize_residual(
                z_e - z0, c1, temperature=1.0)
            near = probe.rank_candidates(d1, i1, int(meta["topk"]), torch)
            errs = []
            for j in range(near.shape[-1]):
                rows_j, _m = k15t.weighted_row_error(
                    ctx.decode_fp32(z0 + c1[near[..., j]]), action,
                    ctx.weights_gate, torch)
                errs.append(rows_j)
            stack = torch.stack(errs, 0)
            best_j = stack.argmin(0)
            idx_ = best_j.view(-1, 1, 1).expand(-1, near.shape[1], 1)
            best_code = near.gather(-1, idx_).squeeze(-1)
            rr = np.asarray(sel, np.int64)
            val_target_by_row[rr] = best_code.cpu().numpy().astype(np.int64)
            val_rows_seen.append(rr)
            w = float(len(sel))
            n_rows_val += len(sel)
            acc_t["rank"] += float(stack.min(0).values.mean()) * w
            acc_t["latent"] += float(stack[0].mean()) * w
            _r0, m0 = k15t.weighted_row_error(
                ctx.decode_fp32(z0), action, ctx.weights_gate, torch)
            acc_t["draft"] += float(m0) * w
    nv = max(n_rows_val, 1)
    val_rows = np.concatenate(val_rows_seen)
    if np.unique(val_rows).size != val_rows.size:
        raise SystemExit("строки val_sel повторяются: цель зависела бы от "
                         "порядка обхода")
    if int((val_target_by_row[val_rows] < 0).sum()):
        raise SystemExit("часть строк val_sel осталась без цели")
    if int((val_target_by_row[val_rows] >= vocab).sum()):
        raise SystemExit("в целях val_sel есть коды вне словаря")
    print(f"  цели val_sel собраны: {val_rows.size} строк, "
          f"{n_pos} кодовых позиций")
    teacher = {k: float(np.sqrt(v / nv)) for k, v in acc_t.items()}
    print(f"  val_sel: черновик {teacher['draft']:.6f}, латентная цель "
          f"{teacher['latent']:.6f}, учитель {teacher['rank']:.6f}; "
          f"разрыв {teacher['draft'] - teacher['rank']:.6f} "
          f"({time.time() - t_teacher:.0f} с)")
    # ДЕШЁВЫЙ СИЛЬНЫЙ ГЕЙТ: в полном режиме учитель тут и учитель, которого
    # probe записал в книгу, считаются на одной и той же части одной и той
    # же замороженной книгой — значит обязаны совпасть.
    book_teacher = book.get("teacher_rms")
    if not a.smoke and book_teacher is not None:
        rel = abs(teacher["rank"] - float(book_teacher)) / max(
            float(book_teacher), 1e-12)
        if rel > 1e-4:
            raise SystemExit(
                f"учитель на val_sel {teacher['rank']!r} против записанного "
                f"probe {book_teacher!r} (относительно {rel:.2e}): считается "
                f"не то же самое")
        print(f"  учитель сошёлся с записанным probe ({book_teacher:.6f}, "
              f"относительно {rel:.1e})")
    elif a.smoke:
        print("  SMOKE: сверка учителя с probe пропущена — часть урезана, "
              "и первые батчи val_sel смещены по RMS")
    cap_target = teacher["draft"] - CAPTURE_THRESHOLD * (
        teacher["draft"] - teacher["rank"])
    print(f"  порог capture {CAPTURE_THRESHOLD} означает RMS(a1_pol) <= "
          f"{cap_target:.6f}")

    def evaluate(tag):
        model.eval()
        acc = dict(a0=0.0, a1_pol=0.0, ce=0.0, agree=0.0)
        n_rows = 0
        codes_p, codes_q, sup, ref, abs_pol = [], [], [], [], []
        with torch.no_grad():
            for po, sel in parts["val_sel"]:
                out = forward(po, sel)
                ce, act, d = losses(po, sel, out, val_target_by_row,
                                    "val_sel")
                w = float(len(sel))
                n_rows += len(sel)
                acc["a0"] += float(d["m_draft"]) * w
                acc["a1_pol"] += float(d["m_pol"]) * w
                acc["ce"] += float(ce) * w
                acc["agree"] += d["agree"] * w
                z0 = out["policy_embeddings"][0]
                c1 = model.depth_aligned_book(1)
                action = torch.from_numpy(
                    np.asarray(ctx.ACT[sel], np.float32)).to(dev)[..., :7]
                z_e = codec._encode(action.float(), embodiment_ids=0).float()
                _lg, _emb, i1, _p = ctx.tok.quantize_residual(
                    z_e - z0, c1, temperature=1.0)
                lat = z0 + out["policy_embeddings"][1]
                sup.append(k15t.decoder_support(
                    lat.detach(), codec, torch, ctx.quantizers,
                    ctx.nearest_code, ctx.code_contribution)[1])
                ref.append(k15t.decoder_support(
                    z_e.detach(), codec, torch, ctx.quantizers,
                    ctx.nearest_code, ctx.code_contribution)[1])
                codes_p.append(out["pred_codes"][1].reshape(
                    len(sel), -1).cpu().numpy())
                codes_q.append(i1.reshape(len(sel), -1).cpu().numpy())
                abs_pol.append(ctx.decode_fp32(lat)[:, :H_EXEC].abs()
                               .reshape(-1, 7).cpu().numpy())
        n = max(n_rows, 1)
        res = {k: v / n for k, v in acc.items()}
        res["rms_a0"] = float(np.sqrt(res["a0"]))
        res["rms_a1_pol"] = float(np.sqrt(res["a1_pol"]))
        res["capture"] = capture(teacher["draft"], teacher["rank"],
                                 res["rms_a1_pol"])
        for nm, arr in (("usage_p", codes_p), ("usage_q", codes_q)):
            flat = np.concatenate(arr, axis=0)
            res[nm] = ctx.tok.code_usage_stats(
                torch.from_numpy(flat.reshape(-1)), vocab)
            per_pos = [ctx.tok.code_usage_stats(
                torch.from_numpy(np.ascontiguousarray(flat[:, t])), vocab)
                for t in range(flat.shape[1])]
            # ВСЕ КОДОВЫЕ ПОЗИЦИИ, А НЕ ПЕРВЫЕ H_EXEC: это разные оси.
            res[nm]["by_position"] = dict(
                positions=len(per_pos),
                position_max_code_share=float(max(
                    s["max_code_share"] for s in per_pos)),
                position_min_perplexity=float(min(
                    s["perplexity"] for s in per_pos)))
        s_all, r_all = torch.cat(sup), torch.cat(ref)
        res["support"] = dict(
            mean=float(s_all.mean()),
            p95=float(torch.quantile(s_all, 0.95)),
            p99=float(torch.quantile(s_all, 0.99)),
            reference_p95=float(torch.quantile(r_all, 0.95)),
            passed=bool(float(torch.quantile(s_all, 0.95))
                        <= float(torch.quantile(r_all, 0.95))))
        flat = np.concatenate(abs_pol, axis=0)
        res["range"] = range_gate(
            [float(x) for x in np.percentile(flat, 99.0, axis=0)],
            [float(x) for x in ctx.act_p99_dataset],
            [float(x) for x in flat.max(axis=0)])
        res["rows"] = n_rows
        print(f"    {tag}: RMS a0 {res['rms_a0']:.6f} -> a1_pol "
              f"{res['rms_a1_pol']:.6f}; capture "
              f"{100 * (res['capture'] or 0):.1f} % (порог "
              f"{100 * CAPTURE_THRESHOLD:.0f} %); CE {res['ce']:.4f}; "
              f"согласие с целью {100 * res['agree']:.2f} %")
        print(f"      опора p95 {res['support']['p95']:.4f} при эталонном "
              f"{res['support']['reference_p95']:.4f}; диапазон "
              f"{'ok' if res['range']['passed'] else 'ОТКАЗ'}; книга P "
              f"perplexity {res['usage_p']['perplexity']:.1f}, мёртвых "
              f"{res['usage_p']['dead_codes']}, худшая кодовая позиция: "
              f"доля {res['usage_p']['by_position']['position_max_code_share']:.3f}, "
              f"perplexity "
              f"{res['usage_p']['by_position']['position_min_perplexity']:.1f}")
        return res

    history = []
    val0 = evaluate("эпоха 0, без обучения")
    history.append(dict(epoch=0, train_loss=None,
                        val_rms_a1_pol=val0["rms_a1_pol"],
                        val_ce=val0["ce"], val=val0))
    capture0 = val0["capture"]
    snapshots = {0: {k: named[k].detach().clone() for k in train_names}}

    order = list(parts["train"])
    forecast = None
    t_start = time.time()
    for epoch in range(1, int(a.epochs) + 1):
        model.eval()
        rng = np.random.default_rng(int(a.seed) + epoch)
        idx = rng.permutation(len(order))
        run = None
        nb = 0
        opt.zero_grad(set_to_none=True)
        for step, j in enumerate(idx, start=1):
            po, sel = order[j]
            out = forward(po, sel)
            ce, act, d = losses(po, sel, out, target_by_row, "train")
            total = ce + lam * act
            (total / float(a.accum)).backward()
            with torch.no_grad():
                cur = dict(total=total.detach(), ce=ce.detach(),
                           action=act.detach())
                run = {k: (v.clone() if run is None else run[k] + v)
                       for k, v in cur.items()}
            nb += 1
            if step % int(a.accum) == 0 or step == len(idx):
                if step <= 10 or step % int(a.report_every) == 0 \
                        or step == len(idx):
                    nog = [n for n in train_names if named[n].grad is None]
                    nf = [n for n in train_names
                          if named[n].grad is not None
                          and not torch.isfinite(named[n].grad).all()]
                    if nog or nf:
                        raise SystemExit(f"градиенты: нет у {nog[:3]}, "
                                         f"нечисловые у {nf[:3]}")
                opt.step()
                opt.zero_grad(set_to_none=True)
            if step == 100 and epoch == 1:
                el = time.time() - t_start
                forecast = dict(
                    per_batch_s=el / step,
                    total_h=el / step * len(order) * int(a.epochs) / 3600.0)
                print(f"    ПРОГНОЗ: {forecast['per_batch_s']:.2f} с/батч, "
                      f"{forecast['total_h']:.1f} ч на {a.epochs} эпох",
                      flush=True)
            if step % int(a.report_every) == 0:
                m = {k: float(v) / nb for k, v in run.items()}
                el = (time.time() - t_start) / 60
                print(f"    эпоха {epoch}: батч {step}/{len(idx)}, потеря "
                      f"{m['total']:.4f} (CE {m['ce']:.4f}, action "
                      f"{m['action']:.4f}), {el:.1f} мин", flush=True)
        train_mean = {k: float(v) / max(nb, 1) for k, v in (run or {}).items()}
        inv, _n = k15t.frozen_invariant(model, torch, set(train_names))
        if inv != frozen_inv0:
            raise SystemExit(
                f"эпоха {epoch}: инвариант замороженного {inv} против "
                f"{frozen_inv0} — сдвинулось что-то вне белого списка фазы")
        val = evaluate(f"эпоха {epoch}")
        history.append(dict(epoch=epoch, train_loss=train_mean.get("total"),
                            train_parts=train_mean,
                            val_rms_a1_pol=val["rms_a1_pol"],
                            val_ce=val["ce"], val=val))
        snapshots[epoch] = {k: named[k].detach().clone() for k in train_names}

    best_epoch, best = select_epoch(history)
    with torch.no_grad():
        for k_, v_ in snapshots[best_epoch].items():
            named[k_].copy_(v_)
    # ОТПЕЧАТОК ОБУЧЕННЫХ ВЕСОВ. `frozen_content_sha` с пустым белым
    # списком хешировала бы всю модель — 2.4e9 значений ради трёх тензоров.
    sel_sha = ctx.k14c.state_sha(
        {k: named[k].detach().float().cpu().numpy() for k in train_names})
    confirm = evaluate(f"переоценка эпохи {best_epoch}")
    for nm, got, want in (("RMS", confirm["rms_a1_pol"],
                           best["val_rms_a1_pol"]),
                          ("CE", confirm["ce"], best["val_ce"])):
        if abs(float(got) - float(want)) / max(abs(float(want)), 1e-12) > 1e-6:
            raise SystemExit(f"переоценка не воспроизвела {nm}: {got!r} "
                             f"против {want!r}")
    sha_after, _n, _e = k15t.frozen_content_sha(model, torch,
                                                set(train_names))
    if sha_after != frozen_sha:
        raise SystemExit(f"побитовый отпечаток замороженного {sha_after} "
                         f"против {frozen_sha}")
    print(f"  переоценка воспроизвела эпоху {best_epoch}; замороженное не "
          f"двигалось")

    gates = {
        "improves_q1": dict(
            category="candidate", rms=confirm["rms_a1_pol"],
            rms_a0=confirm["rms_a0"],
            limit=TECH_IMPROVE_FACTOR * confirm["rms_a0"],
            rule=f"RMS(a1_pol) <= {TECH_IMPROVE_FACTOR} * RMS(a0)",
            passed=bool(confirm["rms_a1_pol"]
                        <= TECH_IMPROVE_FACTOR * confirm["rms_a0"])),
        "support_q1": dict(category="rollout_blocker", **confirm["support"]),
        "range_q1": dict(category="rollout_blocker", **confirm["range"]),
        "collapse_q1_pol": dict(category="rollout_blocker",
                                **collapse_gate(confirm["usage_p"])),
    }
    out_verdict = verdict(gates, confirm["capture"], capture0)
    print(f"\n  capture: эпоха 0 {100 * (capture0 or 0):.1f} % -> эпоха "
          f"{best_epoch} {100 * (confirm['capture'] or 0):.1f} % при пороге "
          f"{100 * CAPTURE_THRESHOLD:.0f} %")
    for k, v in sorted(gates.items()):
        print(f"    [{v['category'][:4]}] {k:18s} "
              f"{'ok' if v['passed'] else 'ОТКАЗ'}")
    print(f"  ИСХОД: {out_verdict['outcome']} (код {out_verdict['code']})")

    payload = dict(
        kind=("k15b_q1_reader_smoke" if a.smoke else "k15b_q1_reader"),
        phase=a.phase, seed=int(a.seed), epochs=int(a.epochs),
        lr=float(a.lr), wd=float(a.wd), accum=int(a.accum),
        lambda_action=lam_report,
        thresholds=dict(tech_improve_factor=TECH_IMPROVE_FACTOR,
                        capture_threshold=CAPTURE_THRESHOLD,
                        lambda_target_ratio=LAMBDA_TARGET_RATIO,
                        declared="до запуска, 02.10.2026"),
        ce_positions="all code positions",
        action_error_positions=int(H_EXEC),
        state={k: named[k].detach().cpu() for k in train_names},
        trainable_names=train_names, selected_epoch=best_epoch,
        selected_state_sha1=sel_sha, history=history, confirm=confirm,
        teacher=teacher, capture_epoch0=capture0,
        capture_target_rms=float(cap_target),
        gates=gates, verdict=out_verdict, forecast=forecast,
        c1_file=os.path.abspath(a.c1), c1_sha1=book["c1_sha1"],
        target_file=os.path.abspath(a.target),
        target_content_sha1=meta["content_sha1"],
        target_meta={k: meta[k] for k in sorted(meta)
                     if k not in ("rank_histogram",)},
        frozen_content_sha=frozen_sha, frozen_invariant=frozen_inv0,
        codec=ctx.codec_fp, code_version=ctx.code_version,
        q0_prov=ctx.q0_prov, joint_sha1=ctx.joint_sha,
        git_head=ctx.git_head, git_dirty=bool(ctx.dirty),
        archived={k: v for k, v in archived.items() if v},
        **ctx.gate_info)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    tmp = out_path + f".tmp.{os.getpid()}"
    torch.save(payload, tmp)
    back = torch.load(tmp, map_location="cpu", weights_only=False)
    for k_, v_ in payload["state"].items():
        if not torch.equal(back["state"][k_], v_):
            os.unlink(tmp)
            raise SystemExit(f"после чтения {k_} изменился")
    os.replace(tmp, out_path)
    print(f"  сохранено: {out_path} (обратное чтение сошлось)")

    os.makedirs(os.path.dirname(os.path.abspath(summary)) or ".",
                exist_ok=True)
    light = {k: v for k, v in payload.items() if k != "state"}
    tmp = summary + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(light, fh, ensure_ascii=False, indent=1, allow_nan=False,
                  default=k15t.json_scalar)
    os.replace(tmp, summary)
    print(f"  сводка: {summary}")

    if a.smoke:
        print("  РЕЖИМ SMOKE: данные урезаны, решения не принимаются")
        return 0
    return int(out_verdict["code"])


if __name__ == "__main__":
    sys.exit(main())
