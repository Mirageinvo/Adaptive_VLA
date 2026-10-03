#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15b, замер M2: мягкий путь по top-k логитов с температурой.

ЗАЧЕМ. Из M1 известно три числа на одних и тех же 5795 строках val_sel при
черновике 0.145469 и учителе 0.072892:

    жёсткий argmax                        0.142997    3.4 %
    мягкий по всему распределению         0.140457    6.9 %
    мягкий внутри ИСТИННОГО кластера K=64 0.135866   13.2 %

Мягкое усреднение вдвое лучше жёсткого выбора, а сужение области
усреднения до правильного соседства удваивает его ещё раз. Последнее
получено с оракулом и само по себе ничего не обещает: оно лишь говорит,
что ОГРАНИЧЕНИЕ ОБЛАСТИ может помочь, и не говорит, что читатель сам
найдёт ту же область. Здесь это и проверяется — область задаётся ТОЛЬКО
логитами читателя, без истинного действия.

ЧТО СЧИТАЕТСЯ. Для каждой пары (k, tau):

    маска    = top-k кодов по логитам читателя
    вес      = softmax(логиты / tau), обнулённый вне маски и перенормированный
    поправка = вес @ C1
    действие = декод(z0 + поправка)

и для каждой точки — RMS, доля разрыва, опора декодера и диапазон. Всё
развёртываемо: на выводе это одно целое k, один скаляр tau и один проход.

СТРУКТУРНЫЕ ИНВАРИАНТЫ. При k = 1 вес вырождается в единицу независимо от
tau, поэтому все пять значений tau обязаны дать ПОБИТОВО одно действие, и
оно обязано совпасть с жёстким путём. При k = словарь и tau = 1 точка
обязана совпасть с нынешним мягким путём. recall@k обязан не убывать по k
и равняться единице при k = словарь. Каждый из этих инвариантов проверяет
именно ту арифметику, которой считаются все остальные точки.

ПОТОЛОК ДВУХСТУПЕНЧАТОГО ЧИТАТЕЛЯ. Для малых k дополнительно считается
лучший из k кандидатов ПО ОШИБКЕ ДЕЙСТВИЯ. Это ceiling схемы «h18 сужает
до k, h24 выбирает», и он НЕ развёртываем: выбор требует истинного
действия. Приводится как потолок, не как результат.

ПОРОГ ОБЪЯВЛЕН ДО ДАННЫХ, 03.10.2026. Лучшая точка обязана пройти опору и
диапазон И взять долю разрыва не меньше, чем мягкий путь плюс 0.05, то
есть около 11.9 % при нынешних числах. Иначе сужение области не работает
и остаётся обучение исполняемого пути по action-члену. Выбор делается на
`val_sel` и помечается как выбранный на этой части; `val_confirm` не
открывается ни при каком исходе.
"""
import argparse
import hashlib
import inspect
import json
import os
import sys
import time

import numpy as np

TOPK_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
TAU_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
ORACLE_TOPK = (2, 4, 8)
SOFT_GAIN_MIN = 0.05
K1_ABS_LIMIT = 1e-5            # допуск вырожденного инварианта при k=1


def topk_mask(logits, k, torch):
    """Маска top-k по логитам. Только логиты, никакого истинного действия."""
    vocab = int(logits.shape[-1])
    k = int(k)
    if not (1 <= k <= vocab):
        raise ValueError(f"k={k} вне [1, {vocab}]")
    if k == vocab:
        return torch.ones_like(logits, dtype=torch.bool)
    idx = logits.topk(k, dim=-1).indices
    m = torch.zeros_like(logits, dtype=torch.bool)
    return m.scatter_(-1, idx, True)


def soft_masked(weights, mask, book, fallback, torch, eps=1e-12):
    """Нормированное среднее книги по маске.

    Где масса под маской вырождена, берётся `fallback`: нормировать на ноль
    нельзя, а вернуть ноль значило бы подменить поправку её отсутствием и
    улучшить результат по ошибке. При единственном ненулевом весе результат
    равен соответствующей строке книги ПОБИТОВО, потому что w/w = 1.0.
    """
    w = weights * mask
    den = w.sum(-1, keepdim=True)
    deg = den.squeeze(-1) <= float(eps)
    emb = (w / den.clamp_min(float(eps))) @ book
    return torch.where(deg.unsqueeze(-1), fallback, emb), deg


def effective_support(weights, mask, torch, eps=1e-12):
    """Perplexity веса под маской: сколько кодов реально участвует."""
    w = weights * mask
    p = w / w.sum(-1, keepdim=True).clamp_min(float(eps))
    ent = -(p * p.clamp_min(1e-30).log()).sum(-1)
    return ent.exp()


def soft_decision(points, capture_base, gain_min=SOFT_GAIN_MIN):
    """Исход замера. Чистая функция.

    `points` — список словарей с полями k, tau, capture, support_passed,
    range_passed. Выбор только среди прошедших ОБА гейта: точка, которая
    не проходит опору, не является рабочей точкой ни при какой доле.
    """
    if capture_base is None:
        return dict(code=3, outcome="доля разрыва мягкого пути не "
                                    "определена: разрыв неположителен",
                    best=None)
    need = float(capture_base) + float(gain_min)
    ok = [p for p in points
          if p.get("support_passed") and p.get("range_passed")
          and p.get("capture") is not None]
    if not ok:
        return dict(code=3, outcome="ни одна точка не прошла опору и "
                                    "диапазон: выбирать не из чего",
                    best=None, required=need)
    best = max(ok, key=lambda p: (float(p["capture"]), -float(p["k"]),
                                  -float(p["tau"])))
    out = dict(best=dict(best), required=need, eligible=len(ok),
               total=len(points), base=float(capture_base),
               gain_min=float(gain_min))
    if float(best["capture"]) >= need - 1e-12:
        out.update(code=0, outcome=(
            f"точка k={best['k']}, tau={best['tau']} берёт "
            f"{100 * float(best['capture']):.1f} % разрыва против "
            f"{100 * float(capture_base):.1f} % у мягкого пути: сужение "
            f"области усреднения работает и развёртываемо без обучения"))
    else:
        out.update(code=4, outcome=(
            f"лучшая прошедшая точка k={best['k']}, tau={best['tau']} даёт "
            f"{100 * float(best['capture']):.1f} % при требуемых "
            f"{100 * need:.1f} %: сужение области само по себе не "
            f"вытягивает, остаётся обучение исполняемого пути"))
    return out


def selftest():
    import torch
    import k15b_measure_interface as mi

    # --- МАСКА TOP-K ----------------------------------------------------
    lg = torch.tensor([[[0.0, 5.0, 1.0, 2.0]]])
    assert topk_mask(lg, 1, torch).tolist() == [[[False, True, False,
                                                  False]]]
    assert topk_mask(lg, 2, torch).tolist() == [[[False, True, False,
                                                  True]]]
    assert bool(topk_mask(lg, 4, torch).all())
    assert int(topk_mask(lg, 3, torch).sum()) == 3
    for bad in (0, 5):
        try:
            topk_mask(lg, bad, torch)
        except ValueError as e:
            assert "вне" in str(e), e
        else:
            raise AssertionError(f"принято k={bad}")

    # --- МЯГКОЕ СРЕДНЕЕ ПО МАСКЕ ----------------------------------------
    bk = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 5.0]])
    p = torch.tensor([[[0.1, 0.4, 0.2, 0.3]]])
    fb = bk[torch.tensor([[1]])]
    # k=1: ПОБИТОВО строка книги, и так при любой температуре
    emb1, deg1 = soft_masked(p, topk_mask(p.log(), 1, torch), bk, fb, torch)
    assert torch.equal(emb1, bk[torch.tensor([[1]])]), emb1
    assert not bool(deg1.any())
    for tau in (0.25, 1.0, 4.0):
        pt = torch.softmax(p.log() / tau, dim=-1)
        e, _d = soft_masked(pt, topk_mask(p.log(), 1, torch), bk, fb, torch)
        assert torch.equal(e, emb1), (tau, e)
    # k=2: веса 0.4 и 0.3 -> (0.4*[3,0] + 0.3*[0,5]) / 0.7
    emb2, _d2 = soft_masked(p, topk_mask(p.log(), 2, torch), bk, fb, torch)
    assert abs(float(emb2[0, 0, 0]) - 1.2 / 0.7) < 1e-6, emb2
    assert abs(float(emb2[0, 0, 1]) - 1.5 / 0.7) < 1e-6, emb2
    # k=V: обычное среднее по всему распределению
    embV, _dV = soft_masked(p, topk_mask(p.log(), 4, torch), bk, fb, torch)
    assert torch.allclose(embV, p @ bk, atol=1e-6), embV
    # ВЫРОЖДЕННАЯ МАССА -> ПРЕДСТАВИТЕЛЬ, А НЕ НОЛЬ
    z = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
    embz, degz = soft_masked(z, topk_mask(p.log(), 2, torch), bk, fb, torch)
    assert bool(degz.all()) and torch.equal(embz, fb)
    # СОГЛАСОВАНО С КЛАСТЕРНОЙ ВЕРСИЕЙ ИЗ M1 НА ОБЩЕМ СЛУЧАЕ
    lab = torch.tensor([0, 0, 1, 1])
    cl = torch.tensor([[1]])
    e_cl, _ = mi.soft_inside(p, lab, cl, bk, fb, torch)
    e_mk, _ = soft_masked(p, (lab.view(1, 1, -1) == cl.unsqueeze(-1)),
                          bk, fb, torch)
    assert torch.equal(e_cl, e_mk), (e_cl, e_mk)

    # --- ЭФФЕКТИВНАЯ ОПОРА ВЕСА -----------------------------------------
    one = effective_support(p, topk_mask(p.log(), 1, torch), torch)
    assert abs(float(one) - 1.0) < 1e-5, one
    flat = torch.full((1, 1, 4), 0.25)
    assert abs(float(effective_support(
        flat, torch.ones_like(flat, dtype=torch.bool), torch)) - 4.0) < 1e-4

    # --- ИСХОД ----------------------------------------------------------
    pts = [dict(k=1, tau=1.0, capture=0.034, support_passed=True,
                range_passed=True),
           dict(k=16, tau=0.5, capture=0.14, support_passed=True,
                range_passed=True),
           dict(k=32, tau=0.5, capture=0.30, support_passed=False,
                range_passed=True)]
    d = soft_decision(pts, 0.069)
    assert d["code"] == 0 and d["best"]["k"] == 16, d
    assert abs(d["required"] - 0.119) < 1e-12, d
    # НЕ ПРОШЕДШАЯ ОПОРУ ТОЧКА НЕ ВЫБИРАЕТСЯ, ДАЖЕ ЕСЛИ ОНА ЛУЧШАЯ
    assert d["best"]["capture"] == 0.14 and d["eligible"] == 2, d
    low = [dict(k=8, tau=1.0, capture=0.10, support_passed=True,
                range_passed=True)]
    assert soft_decision(low, 0.069)["code"] == 4
    # РОВНО НА ПОРОГЕ — ПРОХОДИТ
    edge = [dict(k=8, tau=1.0, capture=0.119, support_passed=True,
                 range_passed=True)]
    assert soft_decision(edge, 0.069)["code"] == 0
    assert soft_decision(pts, None)["code"] == 3
    assert soft_decision([dict(k=8, tau=1.0, capture=0.5,
                               support_passed=False,
                               range_passed=True)], 0.069)["code"] == 3
    # ПРИ РАВНОЙ ДОЛЕ ПРЕДПОЧИТАЕТСЯ МЕНЬШЕЕ k — простейшая точка
    tie = [dict(k=64, tau=1.0, capture=0.2, support_passed=True,
                range_passed=True),
           dict(k=8, tau=1.0, capture=0.2, support_passed=True,
                range_passed=True)]
    assert soft_decision(tie, 0.069)["best"]["k"] == 8

    # КЛЮЧИ АРТЕФАКТА НЕ ДОЛЖНЫ СТОЛКНУТЬСЯ С gate_info: payload
    # собирается как dict(..., **ctx.gate_info), и дубль даёт TypeError
    # после всего прохода.
    import ast
    gate_keys = ("init_gate", "init_gate_sha1", "init_gate_run_id",
                 "init_gate_n_batches", "init_gate_causal",
                 "init_gate_code_version", "decoder_context")
    tree = ast.parse(open(os.path.abspath(__file__), encoding="utf-8")
                     .read())
    found = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "dict"
                and any(k.arg == "kind" for k in node.keywords
                        if isinstance(k, ast.keyword))):
            continue
        explicit = [k.arg for k in node.keywords if k.arg]
        assert len(explicit) == len(set(explicit)), explicit
        clash = sorted(set(explicit) & set(gate_keys))
        assert not clash, f"строка {node.lineno}: {clash} уже в gate_info"
        found += 1
    assert found >= 1, "не нашёлся сбор артефакта"

    assert TOPK_GRID[0] == 1 and TOPK_GRID[-1] == 2048
    assert 1.0 in TAU_GRID and len(TOPK_GRID) * len(TAU_GRID) == 60
    assert all(k in TOPK_GRID for k in ORACLE_TOPK)
    print(f"самопроверка k15b_measure_soft пройдена: {len(TOPK_GRID)} "
          f"значений k, {len(TAU_GRID)} температур, порог "
          f"мягкий+{SOFT_GAIN_MIN}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import k15_context
    import k15b_probe_and_extract as probe
    import k15b_train_stagewise as trainer
    import k15b_build_rankpath_cache as cachelib
    import k15b_measure_interface as mi
    from k15_train_depth_rvq import H_EXEC

    ap = argparse.ArgumentParser(
        description="K-15b: мягкий путь по top-k с температурой")
    k15_context.add_common_arguments(ap)
    ap.add_argument("--c1", default="data/k15b/c1_selected.pt")
    ap.add_argument("--target", default="data/k15b/rankpath_target_train.npz")
    ap.add_argument("--checkpoint", default="data/k15b/q1_reader_s0.pt")
    ap.add_argument("--point", default="")
    ap.add_argument("--batches", type=int, default=0)
    ap.add_argument("--support-every", type=int, default=8)
    ap.add_argument("--out", default="data/k15b/soft_operating_point.pt")
    ap.add_argument("--summary", default="reports/k15b/measure_soft.json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if int(a.limit) != 0:
        raise SystemExit("--limit не применяется: берите --batches")
    for path in (a.out, a.summary):
        if os.path.exists(path) and not a.overwrite:
            raise SystemExit(f"{path} уже существует: без --overwrite не "
                             f"перезаписываю")
    archived = {p: probe.archive_existing(p) for p in (a.out, a.summary)}
    for path, dest in archived.items():
        if dest:
            print(f"  прежний {path} перенесён в {dest}")

    ctx = k15_context.build(a)
    torch = ctx.torch
    dev = ctx.dev
    model, codec = ctx.model, ctx.codec
    k15t = k15_context.k15t
    vocab = int(ctx.vocab)
    ks = [int(k) for k in TOPK_GRID if int(k) <= vocab]
    if ks[-1] != vocab:
        ks.append(vocab)
    taus = [float(t) for t in TAU_GRID]
    print(f"  сетка: k {ks}, tau {taus} — {len(ks) * len(taus)} точек")

    # --- КНИГА, ЗАМОРОЖЕННОЕ, КЭШ, ЧИТАТЕЛЬ: КАК В M1 -------------------
    book = torch.load(a.c1, map_location="cpu", weights_only=False)
    if book.get("kind") != "k15b_c1_selected" \
            or book.get("accepted") is not True:
        raise SystemExit(f"{a.c1}: kind {book.get('kind')!r}, accepted "
                         f"{book.get('accepted')!r}")
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
    train_names = trainer.phase_names(list(ctx.info["names"]), "q1_reader")
    frozen_sha, n_frozen, n_elem = k15t.frozen_content_sha(
        model, torch, set(train_names))
    print(f"  книга {book['c1_sha1']}; замороженного {n_frozen} тензоров "
          f"({n_elem} значений), отпечаток {frozen_sha}")

    if not os.path.exists(a.target):
        raise SystemExit(f"нет {a.target}")
    cache = np.load(a.target, allow_pickle=True)
    meta = json.loads(str(cache["meta"]))
    n_pos_model = int(np.asarray(ctx.q0_can).shape[1])
    expect = dict(
        q0_prov=ctx.q0_prov, plan_sha1=ctx.q0_prov.get("plan_sha1"),
        topk=int(book["topk"]), codec=ctx.codec_fp,
        code_version=ctx.code_version, joint_sha1=ctx.joint_sha,
        channel_weights=[float(x) for x in
                         ctx.weights_gate.detach().cpu().numpy()],
        decoder_context=ctx.decoder_context,
        rank_candidates_file_sha1=k15_context.sha12(
            inspect.getfile(probe.rank_candidates)),
        action_error_positions=int(H_EXEC),
        code_target_positions=n_pos_model)
    problems, _tm = trainer.check_cache_contract(meta, book, expect)
    if problems:
        raise SystemExit("кэш цели не подходит: " + "; ".join(problems[:6]))
    rows_t = np.asarray(cache["rows"], np.int64)
    codes_t = np.asarray(cache["codes"], np.int64)
    ranks_t = np.asarray(cache["ranks"], np.int16)
    plan_rows, _slot = cachelib.build_row_index(ctx.parts_full["train"])
    if not np.array_equal(rows_t, plan_rows) \
            or cachelib.cache_fingerprint(rows_t, codes_t, ranks_t) \
            != meta["content_sha1"]:
        raise SystemExit("кэш не тот: строки или содержимое не совпали")
    n_pos = int(codes_t.shape[1])
    if n_pos != n_pos_model:
        raise SystemExit(f"кодовых позиций в кэше {n_pos}, у модели "
                         f"{n_pos_model}")
    topk_teacher = int(meta["topk"])

    if not os.path.exists(a.checkpoint):
        raise SystemExit(f"нет {a.checkpoint}")
    obj = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if str(obj.get("kind")) not in mi.READER_KINDS:
        raise SystemExit(f"{a.checkpoint} описывает {obj.get('kind')!r}")
    problems = mi.check_reader(obj, ctx, book, meta, frozen_sha,
                               train_names)
    if problems:
        raise SystemExit("чекпойнт читателя собран в другой обстановке: "
                         + "; ".join(problems[:6]))
    sel_tag, states, has_all = mi.reader_points(obj)
    point = str(a.point) or sel_tag
    if point not in states:
        raise SystemExit(f"точки {point!r} нет. Доступны {sorted(states)}")
    own = dict(model.state_dict())
    with torch.no_grad():
        for k_, v_ in states[point].items():
            if tuple(v_.shape) != tuple(own[k_].shape):
                raise SystemExit(f"{point}.{k_}: форма не та")
            if not torch.isfinite(v_).all():
                raise SystemExit(f"{point}.{k_}: нечисловые значения")
            own[k_].copy_(v_.to(own[k_].device, own[k_].dtype))
    point_sha = ctx.k14c.state_sha(
        {k_: model.state_dict()[k_].detach().float().cpu().numpy()
         for k_ in train_names})
    if point == sel_tag and point_sha != str(obj["selected_state_sha1"]):
        raise SystemExit(f"после загрузки отпечаток {point_sha}, в "
                         f"чекпойнте {obj['selected_state_sha1']}")
    again, _n, _e = k15t.frozen_content_sha(model, torch, set(train_names))
    if again != frozen_sha:
        raise SystemExit("подстановка тронула не только белый список")
    hist = {str(h.get("tag")): h for h in (obj.get("history") or [])}
    hp = hist.get(point) or {}
    print(f"  читатель: точка {point} ({point_sha}), в истории RMS "
          f"{hp.get('val_rms_a1_pol')}, CE {hp.get('val_ce')}")
    if point == trainer.ZERO_TAG:
        print("    ВНИМАНИЕ: это НЕОБУЧЕННАЯ точка — инициализация K-14")
    model.eval()

    # --- ПРОХОД ---------------------------------------------------------
    batch_list = ctx.parts["val_sel"]
    take = probe.strided(len(batch_list), int(a.batches))
    chosen = [batch_list[i] for i in take]
    full_val = len(chosen) == len(batch_list)
    rows_all = np.unique(np.concatenate(
        [np.asarray(sel, np.int64) for _po, sel in chosen]))
    if rows_all.size != sum(len(sel) for _po, sel in chosen):
        raise SystemExit("строки val_sel повторяются")
    rows_sha = hashlib.sha1(
        np.ascontiguousarray(rows_all).tobytes()).hexdigest()[:12]
    print(f"  проход по {len(chosen)} батчам val_sel из {len(batch_list)}, "
          f"{rows_all.size} строк, отпечаток {rows_sha}")

    BASE = ("a0", "a1_tok", "a1_rank", "a1_pol", "a1_soft")
    keyf = (lambda k_, t_: f"k{int(k_)}_t{float(t_):g}")
    acc = {nm: 0.0 for nm in BASE}
    acc.update({keyf(k_, t_): 0.0 for k_ in ks for t_ in taus})
    acc.update({f"oracle_top{k_}": 0.0 for k_ in ORACLE_TOPK})
    sup = {"reference": [], "reference_strided": [], "a1_pol": [],
           "a1_soft": []}
    rng_abs = {"a1_pol": [], "a1_soft": []}
    for k_ in ks:
        for t_ in taus:
            sup[keyf(k_, t_)] = []
            rng_abs[keyf(k_, t_)] = []
    recall = {int(k_): 0.0 for k_ in ks}
    eff = {keyf(k_, t_): 0.0 for k_ in ks for t_ in taus}
    degen = {keyf(k_, t_): 0.0 for k_ in ks for t_ in taus}
    prob_gap, n_rows, forecast = 0.0, 0, None
    gaps = dict(k1_vs_hard=0.0, k1_across_tau=0.0)
    q0_dev = torch.as_tensor(np.asarray(ctx.q0_can), device=dev)
    ac16 = torch.autocast(device_type=dev.type, dtype=ctx.dt)
    c1 = model.depth_aligned_book(1)
    every = max(int(a.support_every), 1)
    t0 = time.time()
    with torch.no_grad():
        for bi, (po, sel) in enumerate(chosen, start=1):
            b = ctx.build_batch(po, sel)
            am = b.get("attention_mask")
            with ac16:
                v, p_ids = model.build_inputs(position_offset=po, **b)
                out = model.forward_depth_aligned_rvq(
                    vlm_inputs_embeds=v, attention_mask=am,
                    position_ids=p_ids, mode="full", tau=1.0)
            bad = int((out["pred_codes"][0]
                       != q0_dev[torch.as_tensor(sel, device=dev)]).sum())
            if bad:
                raise SystemExit(f"q0 разошёлся с каноническим в {bad} "
                                 f"позициях")
            B = len(sel)
            n_rows += B
            w = float(B)
            z0 = out["policy_embeddings"][0]
            c1_pol = out["policy_embeddings"][1]
            lg = out["logits"][1].float()
            probs_model = out["policy_probabilities"][1].float()
            pred = out["pred_codes"][1]
            action = torch.from_numpy(
                np.asarray(ctx.ACT[sel], np.float32)).to(dev)[..., :7]
            z_e = codec._encode(action.float(), embodiment_ids=0).float()
            c1f = c1.float()

            def record(name, act, latent=None, keep=False):
                rows_, mean_ = k15t.weighted_row_error(
                    act, action, ctx.weights_gate, torch)
                acc[name] += float(mean_) * w
                if keep:
                    sup[name].append(k15t.decoder_support(
                        latent.detach(), codec, torch, ctx.quantizers,
                        ctx.nearest_code, ctx.code_contribution)[1])
                    rng_abs[name].append(act[:, :H_EXEC].abs()
                                         .reshape(-1, 7).cpu().numpy())
                return rows_

            # --- УЧИТЕЛЬ: ЦЕЛЬ И ЕГО RMS ------------------------------
            d1 = ctx.tok.mean_squared_distances(z_e - z0, c1)
            _l, _e2, i1, _p = ctx.tok.quantize_residual(
                z_e - z0, c1, temperature=1.0)
            near = probe.rank_candidates(d1, i1, topk_teacher, torch)
            errs, decoded = [], []
            for j in range(near.shape[-1]):
                dec_j = ctx.decode_fp32(z0 + c1[near[..., j]])
                rows_j, _m = k15t.weighted_row_error(
                    dec_j, action, ctx.weights_gate, torch)
                errs.append(rows_j)
                decoded.append(dec_j)
            stack = torch.stack(errs, 0)
            best_j = stack.argmin(0)
            idx_ = best_j.view(-1, 1, 1).expand(-1, near.shape[1], 1)
            target = near.gather(-1, idx_).squeeze(-1)
            rows_idx = torch.arange(B, device=best_j.device)
            record("a0", ctx.decode_fp32(z0))
            record("a1_tok", decoded[0])
            record("a1_rank", torch.stack(decoded, 0)[best_j, rows_idx])

            keep = (bi % every == 0)
            ref_rows = k15t.decoder_support(
                z_e.detach(), codec, torch, ctx.quantizers,
                ctx.nearest_code, ctx.code_contribution)[1]
            sup["reference"].append(ref_rows)
            if keep:
                sup["reference_strided"].append(ref_rows)
            lat_pol = z0 + c1_pol
            record("a1_pol", ctx.decode_fp32(lat_pol), lat_pol, True)
            lat_soft = z0.float() + probs_model @ c1f
            record("a1_soft", ctx.decode_fp32(lat_soft), lat_soft, True)
            prob_gap = max(prob_gap, float(
                (torch.softmax(lg, dim=-1) - probs_model).abs().max()))

            # --- РАЗВЁРТКА (k, tau) -----------------------------------
            masks = {int(k_): topk_mask(lg, int(k_), torch) for k_ in ks}
            if bi == 1:
                # МАСКИ ВЛОЖЕНЫ ПО ПОСТРОЕНИЮ, И ИМЕННО ЭТО ДЕЛАЕТ
                # recall@k неубывающим. Проверяется на первом батче.
                for ka, kb in zip(ks, ks[1:]):
                    if int((masks[ka] & ~masks[kb]).sum()):
                        raise SystemExit(
                            f"маска top-{ka} не вложена в top-{kb}")
                if not bool(masks[ks[-1]].all()):
                    raise SystemExit("маска при k=словарь не полная")
            for k_ in ks:
                recall[int(k_)] += float(masks[int(k_)].gather(
                    -1, target.unsqueeze(-1)).squeeze(-1)
                    .float().mean()) * w
            fallback = c1f[pred]
            emb_k1 = None
            for t_ in taus:
                p_t = torch.softmax(lg / float(t_), dim=-1)
                for k_ in ks:
                    key = keyf(k_, t_)
                    emb, deg = soft_masked(p_t, masks[int(k_)], c1f,
                                           fallback, torch)
                    if int(k_) == 1:
                        # ВЕС ВЫРОЖДАЕТСЯ В ЕДИНИЦУ: при любой температуре
                        # это строка книги под argmax. Сравнение не
                        # побитовое, а с допуском K1_ABS_LIMIT: в сумме
                        # 2047 слагаемых — ровно нули, и на fp32-gemm это
                        # точно, но tf32 на Ampere округлил бы и
                        # единственное произведение. Настоящая ошибка в
                        # маске или нормировке даёт расхождение порядка
                        # самой поправки, а не 1e-7.
                        g = float((emb - fallback).abs().max())
                        gaps["k1_vs_hard"] = max(gaps["k1_vs_hard"], g)
                        if g > K1_ABS_LIMIT:
                            raise SystemExit(
                                f"k=1, tau={t_}: мягкое среднее по одному "
                                f"коду разошлось со строкой книги на "
                                f"{g:.3e} при пределе {K1_ABS_LIMIT}")
                        if emb_k1 is None:
                            emb_k1 = emb
                        else:
                            g2 = float((emb - emb_k1).abs().max())
                            gaps["k1_across_tau"] = max(
                                gaps["k1_across_tau"], g2)
                            if g2 > K1_ABS_LIMIT:
                                raise SystemExit(
                                    f"k=1: tau={t_} дала другое среднее, "
                                    f"чем первая температура, на "
                                    f"{g2:.3e}")
                    degen[key] += float(deg.float().mean()) * w
                    eff[key] += float(effective_support(
                        p_t, masks[int(k_)], torch).mean()) * w
                    lat = z0.float() + emb
                    record(key, ctx.decode_fp32(lat), lat, keep)

            # --- ПОТОЛОК ДВУХСТУПЕНЧАТОЙ СХЕМЫ ------------------------
            for k_ in ORACLE_TOPK:
                idx = lg.topk(int(k_), dim=-1).indices
                errs_o = []
                for j in range(int(k_)):
                    dec_j = ctx.decode_fp32(z0 + c1[idx[..., j]])
                    rows_j, _m = k15t.weighted_row_error(
                        dec_j, action, ctx.weights_gate, torch)
                    errs_o.append(rows_j)
                acc[f"oracle_top{k_}"] += float(
                    torch.stack(errs_o, 0).min(0).values.mean()) * w

            if bi == min(20, len(chosen)):
                forecast = k15t.forecast_runtime(time.time() - t0, bi,
                                                 len(chosen), 1)
                print(f"    прогноз: {forecast['per_batch_s']:.2f} с/батч, "
                      f"весь проход {forecast['total_h']:.2f} ч", flush=True)
            if bi % 100 == 0:
                print(f"    {bi}/{len(chosen)} батчей, "
                      f"{time.time() - t0:.0f} с", flush=True)
    elapsed = time.time() - t0
    n = max(n_rows, 1)
    rms = {k_: float(np.sqrt(v / n)) for k_, v in acc.items()}
    print(f"  проход занял {elapsed / 60:.1f} мин, {n_rows} строк; "
          f"максимальное расхождение softmax(логиты) с вероятностями "
          f"модели {prob_gap:.3e}")
    print(f"  вырожденный инвариант k=1: расхождение со строкой книги "
          f"{gaps['k1_vs_hard']:.3e}, между температурами "
          f"{gaps['k1_across_tau']:.3e} при пределе {K1_ABS_LIMIT}")
    # --- СВЕРКИ ---------------------------------------------------------
    draft, teach = rms["a0"], rms["a1_rank"]
    book_teacher = book.get("teacher_rms")
    if full_val:
        if book_teacher is None:
            raise SystemExit(f"{a.c1}: нет teacher_rms")
        rel_t = abs(teach - float(book_teacher)) / max(float(book_teacher),
                                                       1e-12)
        if rel_t > mi.TEACHER_REL_LIMIT:
            raise SystemExit(f"учитель {teach!r} против записанного probe "
                             f"{book_teacher!r} ({rel_t:.2e})")
        print(f"  учитель сошёлся с probe ({book_teacher:.6f}, "
              f"относительно {rel_t:.1e})")
    else:
        rel_t = None
        print("  сверка учителя с probe пропущена: взята не вся val_sel")

    recall = {k_: float(v / n) for k_, v in recall.items()}
    if abs(recall[ks[-1]] - 1.0) > 1e-12:
        raise SystemExit(f"recall при k=словарь равен {recall[ks[-1]]!r}, "
                         f"а обязан быть единицей")
    for ka, kb in zip(ks, ks[1:]):
        if recall[kb] < recall[ka] - 1e-12:
            raise SystemExit(f"recall убыл: @{ka} {recall[ka]}, @{kb} "
                             f"{recall[kb]}")

    consistency = {}
    for name, other, why in (
            [(keyf(1, t_), "a1_pol",
              "мягкое среднее по одному коду — это жёсткий путь")
             for t_ in taus]
            + [(keyf(ks[-1], 1.0), "a1_soft",
                "вся маска при tau=1 — это нынешний мягкий путь")]):
        got = abs(rms[name] - rms[other]) / max(rms[other], 1e-12)
        consistency[name] = dict(against=other, rel=float(got), why=why,
                                 limit=mi.PATH_REL_LIMIT,
                                 passed=bool(got <= mi.PATH_REL_LIMIT))
    broken = sorted(k_ for k_, v in consistency.items() if not v["passed"])
    worst = max(v["rel"] for v in consistency.values())
    if broken:
        print(f"  СТРУКТУРНЫЕ ИНВАРИАНТЫ НЕ СОШЛИСЬ: {broken}")
        for k_ in broken:
            print(f"    {k_} против {consistency[k_]['against']}: "
                  f"относительно {consistency[k_]['rel']:.2e}")
    else:
        print(f"  структурные инварианты сошлись: {len(consistency)} "
              f"сверок, худшая относительная разность {worst:.2e}")

    # --- ОПОРА И ДИАПАЗОН -----------------------------------------------
    ref_all = torch.cat(sup["reference"])
    ref_p95 = float(torch.quantile(ref_all, 0.95))
    ref_str = torch.cat(sup["reference_strided"])
    ref_str_p95 = float(torch.quantile(ref_str, 0.95))

    def sup_stats(name, strided=False):
        if not sup.get(name):
            return None
        v = torch.cat(sup[name])
        r95 = ref_str_p95 if strided else ref_p95
        rn = int(ref_str.numel() if strided else ref_all.numel())
        if int(v.numel()) != rn:
            raise SystemExit(f"опора {name}: {int(v.numel())} значений "
                             f"против {rn} у эталона — гейт не парный")
        p95 = float(torch.quantile(v, 0.95))
        return dict(n=int(v.numel()), mean=float(v.mean()), p95=p95,
                    p99=float(torch.quantile(v, 0.99)),
                    reference_p95=r95, reference_n=rn,
                    reference_scope=("каждый n-й батч, те же строки"
                                     if strided else "все батчи"),
                    passed=bool(p95 <= r95),
                    rule="p95 остатка пути <= p95 остатка кодека на "
                         "истинном латенте, НА ТЕХ ЖЕ СТРОКАХ")

    def rng_stats(name):
        if not rng_abs.get(name):
            return None
        flat = np.concatenate(rng_abs[name], axis=0)
        p99 = [float(x) for x in np.percentile(flat, 99.0, axis=0)]
        amax = [float(x) for x in flat.max(axis=0)]
        return dict(values=int(flat.shape[0]),
                    rows=int(flat.shape[0] // int(H_EXEC)),
                    positions=int(H_EXEC), p99_candidate=p99,
                    absmax_candidate=amax,
                    p99_dataset=[float(x) for x in ctx.act_p99_dataset],
                    **probe.range_ok(p99, ctx.act_p99_dataset, amax))

    # --- ТОЧКИ ----------------------------------------------------------
    capture_hard = trainer.capture(draft, teach, rms["a1_pol"])
    capture_soft = trainer.capture(draft, teach, rms["a1_soft"])
    points = []
    for k_ in ks:
        for t_ in taus:
            key = keyf(k_, t_)
            s_ = sup_stats(key, strided=True)
            g_ = rng_stats(key)
            points.append(dict(
                key=key, k=int(k_), tau=float(t_), rms=float(rms[key]),
                capture=trainer.capture(draft, teach, rms[key]),
                support=s_, range=g_,
                support_passed=bool(s_ is not None and s_["passed"]),
                range_passed=bool(g_ is not None and g_["passed"]),
                effective_support=float(eff[key] / n),
                degenerate_share=float(degen[key] / n)))
    decision = soft_decision(points, capture_soft)
    if broken:
        decision = dict(code=3, outcome=(
            f"структурные инварианты не сошлись ({broken}): рабочая точка "
            f"не выбирается, числа сохранены для разбора"),
            best=None, broken=broken, would_have_been=decision)

    ceiling = {int(k_): dict(
        rms=float(rms[f"oracle_top{k_}"]),
        capture=trainer.capture(draft, teach, rms[f"oracle_top{k_}"]))
        for k_ in ORACLE_TOPK}

    # --- ОТЧЁТ ----------------------------------------------------------
    print(f"\n  ЧЕРНОВИК {draft:.6f}; латентная цель {rms['a1_tok']:.6f}; "
          f"УЧИТЕЛЬ {teach:.6f}")
    print(f"  жёсткий {rms['a1_pol']:.6f} "
          f"({100 * (capture_hard or 0):.1f} %); мягкий "
          f"{rms['a1_soft']:.6f} ({100 * (capture_soft or 0):.1f} %)")
    print("\n  ДОЛЯ РАЗРЫВА, % (строки — k, столбцы — tau); звёздочка — "
          "не прошла опору или диапазон:")
    print("      k " + "".join(f"{t_:>9g}" for t_ in taus)
          + "   recall@k")
    by_key = {p["key"]: p for p in points}
    for k_ in ks:
        cells = []
        for t_ in taus:
            p_ = by_key[keyf(k_, t_)]
            mark = "" if (p_["support_passed"] and p_["range_passed"]) \
                else "*"
            cells.append(f"{100 * (p_['capture'] or 0):8.1f}{mark:1s}")
        print(f"  {k_:>5d} " + "".join(cells)
              + f"   {100 * recall[k_]:7.2f} %")
    print("\n  эффективное число кодов под маской при tau=1: " + ", ".join(
        f"k={k_} {by_key[keyf(k_, 1.0)]['effective_support']:.1f}"
        for k_ in ks))
    print("\n  ПОТОЛОК ДВУХСТУПЕНЧАТОЙ СХЕМЫ (не развёртываем, выбор "
          "требует истинного действия):")
    for k_ in ORACLE_TOPK:
        print(f"    лучший из top-{k_} по ошибке действия: "
              f"{ceiling[k_]['rms']:.6f} "
              f"({100 * (ceiling[k_]['capture'] or 0):.1f} %)")
    if decision.get("best"):
        bp = by_key[decision["best"]["key"]]
        print(f"\n  ЛУЧШАЯ ПРОШЕДШАЯ ТОЧКА: k={bp['k']}, tau={bp['tau']}, "
              f"RMS {bp['rms']:.6f}, доля {100 * (bp['capture'] or 0):.1f} "
              f"% при требуемых {100 * decision['required']:.1f} %")
        print(f"    опора p95 {bp['support']['p95']:.4f} при эталоне "
              f"{bp['support']['reference_p95']:.4f}; диапазон "
              f"{'ok' if bp['range_passed'] else 'ОТКАЗ'}; эффективных "
              f"кодов {bp['effective_support']:.1f}")
        print("    ВЫБРАНА НА val_sel — это часть отбора, не независимая "
              "проверка; val_confirm не открывалась")
    print(f"\n  ИСХОД: {decision['outcome']} (код {decision['code']})")

    # --- СОХРАНЕНИЕ -----------------------------------------------------
    payload = dict(
        kind="k15b_soft_operating_point",
        accepted=bool(decision["code"] == 0),
        thresholds=dict(soft_gain_min=SOFT_GAIN_MIN,
                        declared="до данных, 03.10.2026"),
        selected_on="val_sel", val_confirm_used_for_selection=False,
        decision=decision, points=points, ceiling_oracle_topk=ceiling,
        recall={str(k_): v for k_, v in recall.items()},
        rms=rms, consistency=consistency,
        capture_hard=capture_hard, capture_soft=capture_soft,
        teacher=dict(draft=float(draft), latent=float(rms["a1_tok"]),
                     rank=float(teach),
                     probe_teacher_rms=(None if book_teacher is None
                                        else float(book_teacher)),
                     relative_to_probe=rel_t, full_val_sel=bool(full_val)),
        support=dict(a1_pol=sup_stats("a1_pol"),
                     a1_soft=sup_stats("a1_soft"),
                     reference_p95=ref_p95,
                     reference_strided_p95=ref_str_p95),
        range={nm: rng_stats(nm) for nm in ("a1_pol", "a1_soft")},
        probability_gap_softmax_vs_model=float(prob_gap),
        degenerate_invariant_gaps={k_: float(v) for k_, v in gaps.items()},
        reader=dict(checkpoint=os.path.abspath(a.checkpoint), point=point,
                    point_state_sha1=point_sha, selected_tag=sel_tag,
                    has_all_states=bool(has_all),
                    trained=bool(point != trainer.ZERO_TAG),
                    history_entry=hp),
        grid=dict(topk=[int(x) for x in ks], tau=[float(t) for t in taus],
                  oracle_topk=[int(x) for x in ORACLE_TOPK]),
        rows=int(n_rows), rows_sha1=rows_sha, batches=len(chosen),
        batches_in_part=len(batch_list), seconds=float(elapsed),
        forecast=forecast, support_every=int(every),
        code_target_positions=int(n_pos),
        action_error_positions=int(H_EXEC),
        c1_file=os.path.abspath(a.c1), c1_sha1=book["c1_sha1"],
        target_file=os.path.abspath(a.target),
        target_content_sha1=meta["content_sha1"],
        frozen_content_sha=frozen_sha, codec=ctx.codec_fp,
        code_version=ctx.code_version, q0_prov=ctx.q0_prov,
        joint_sha1=ctx.joint_sha, git_head=ctx.git_head,
        git_dirty=bool(ctx.dirty),
        archived={k_: v for k_, v in archived.items() if v},
        **ctx.gate_info)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                exist_ok=True)
    tmp = a.out + f".tmp.{os.getpid()}"
    torch.save(payload, tmp)
    back = torch.load(tmp, map_location="cpu", weights_only=False)
    if back["decision"]["code"] != payload["decision"]["code"] \
            or len(back["points"]) != len(payload["points"]):
        os.unlink(tmp)
        raise SystemExit("после чтения артефакт изменился")
    os.replace(tmp, a.out)
    print(f"  сохранено: {a.out} (обратное чтение сошлось)")
    os.makedirs(os.path.dirname(os.path.abspath(a.summary)) or ".",
                exist_ok=True)
    tmp = a.summary + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1,
                  allow_nan=False, default=k15t.json_scalar)
    os.replace(tmp, a.summary)
    print(f"  сводка: {a.summary}")
    return int(decision["code"])


if __name__ == "__main__":
    sys.exit(main())
