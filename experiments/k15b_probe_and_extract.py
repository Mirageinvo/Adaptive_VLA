#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15b, шаг 0: выбрать книгу C1 из K-15 и решить, какую цель дать читателю.

ОДИН ПРОХОД, ТРИ ВОПРОСА. Все три отвечаются на одних и тех же строках, и
это важно: доли и разности между ними сравнимы только при общей области.

ВОПРОС 1. КАКОЕ ИЗ СОХРАНЁННЫХ СОСТОЯНИЙ ДАЁТ ЛУЧШУЮ КНИГУ.
Выбор по `a1_tok` — качеству СЛОВАРЯ, а не по `a1_pol`: читатель в K-15
деградировал, и его качество к выбору книги отношения не имеет.

ВОПРОС 2. ЛЕЖИТ ЛИ ЭТА КНИГА НА МНОГООБРАЗИИ ДЕКОДЕРА.
В K-15 опора декодера считалась для `a1_pol`, `a2_pol` и `a1_soft`, но НЕ
для `a1_tok`. То есть посылка «книга хорошая, потому что `a1_tok` = 0.055»
не проверена в главном: если полезные строки книги сами вне многообразия,
то 0.055 — экстраполяция замороженного декодера, и учить читателя попадать
в такие коды значит учить его действию, которое на роботе может не
воспроизвестись. Здесь это измеряется.

Есть и конкретная гипотеза, которую проверяет тот же проход. В K-15 нормы
строк разошлись: выбираемые ТОКЕНИЗАТОРОМ не выросли (0.2754 -> 0.2685), а
выбираемые ПОЛИТИКОЙ выросли втрое (0.1966 -> 0.5932). Похоже, побег в
нуль-пространство декодера сосредоточен в подмножестве строк, которые
политика научилась выбирать именно потому, что они действенно нейтральны.
Если так, заморозка книги и цель по коду токенизатора эти строки обходят.
Проверяется разбиением множеств кодов на «только Q», «только P» и «оба» с
нормами и опорой по каждому.

ВОПРОС 3. КАКОЙ КОД ВООБЩЕ СТОИТ ПРЕДСКАЗЫВАТЬ.
Цель читателя в K-15 — argmin ЛАТЕНТНОГО расстояния. Лучший по ОШИБКЕ
ДЕЙСТВИЯ код может быть другим. Полный ответ (§50.2) требует 2048 декодов
на строку; здесь считается честно суженная версия — лучший по действию
среди `--topk` ближайших латентных, — и она так и называется:
`action probe внутри латентного соседства`. За пределами этого соседства
может лежать код лучше, и этот probe о нём ничего не говорит.

ПРАВИЛО ВЫБОРА ЦЕЛИ ЗАПИСАНО ДО ЗАПУСКА (см. TARGET_RULE):
    выигрыш < 2 %      -> латентная цель
    2-5 %              -> латентная цель (в первом быстром K-15b)
    > 5 %              -> action-best внутри соседства, если полный кэш
                          укладывается в объявленный бюджет
    кэш дороже бюджета -> латентная цель плюс малый action-член

Скрипт НИЧЕГО НЕ ОБУЧАЕТ и ничего не перезаписывает без --overwrite.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

# ПРАВИЛО ВЫБОРА ЦЕЛИ, ЗАФИКСИРОВАННОЕ ДО ДАННЫХ.
TARGET_RULE = (
    (0.02, "latent", "выигрыш ниже 2 %: латентная цель"),
    (0.05, "latent", "выигрыш 2-5 %: латентная цель в первом K-15b"),
    (float("inf"), "action_topk",
     "выигрыш выше 5 %: action-best внутри латентного соседства, "
     "если полный кэш укладывается в бюджет"),
)
CACHE_BUDGET_HOURS = 6.0   # дороже — остаёмся на латентной цели


def target_decision(relative_gain, projected_cache_hours):
    """Решение по правилу, записанному выше. Чистая функция.

    `relative_gain` — (RMS латентного argmin - RMS action-best) / RMS
    латентного argmin, то есть насколько action-цель лучше латентной.
    """
    if not np.isfinite(relative_gain):
        raise ValueError(f"выигрыш не число: {relative_gain}")
    for threshold, choice, why in TARGET_RULE:
        if relative_gain < threshold:
            break
    if choice == "action_topk" and projected_cache_hours > CACHE_BUDGET_HOURS:
        return dict(target="latent_plus_action_term",
                    rule_branch=why,
                    reason=(f"полный кэш оценён в {projected_cache_hours:.1f} ч "
                            f"при бюджете {CACHE_BUDGET_HOURS:.1f} ч, поэтому "
                            f"остаёмся на латентной цели и добавляем малый "
                            f"action-член"),
                    relative_gain=float(relative_gain))
    return dict(target=choice, rule_branch=why, reason=why,
                relative_gain=float(relative_gain),
                projected_cache_hours=float(projected_cache_hours))


def strided(n_total, n_take):
    """Детерминированно разреженные индексы по всей части, а не первые N.

    Первые N батчей плана — это первые эпизоды, а не выборка из части.
    """
    if n_take <= 0 or n_total <= 0:
        return []
    n_take = min(int(n_take), int(n_total))
    return sorted(set(int(x) for x in
                      np.linspace(0, n_total - 1, n_take).astype(int)))


def row_set_split(codes_q, codes_p, book_norms):
    """Разбиение множеств использованных кодов и нормы по каждой части.

    Проверяет гипотезу «побег сосредоточен в строках, которые выбирает
    политика и не выбирает токенизатор».
    """
    set_q = set(int(x) for x in np.unique(codes_q))
    set_p = set(int(x) for x in np.unique(codes_p))
    both = sorted(set_q & set_p)
    only_q = sorted(set_q - set_p)
    only_p = sorted(set_p - set_q)
    union = set_q | set_p

    def stats(ids):
        if not ids:
            return dict(count=0, median=None, p95=None, max=None)
        v = np.asarray([book_norms[i] for i in ids], np.float64)
        return dict(count=len(ids), median=float(np.median(v)),
                    p95=float(np.percentile(v, 95)), max=float(v.max()))

    return dict(
        used_by_q=len(set_q), used_by_p=len(set_p),
        jaccard=float(len(both) / max(len(union), 1)),
        both=stats(both), only_q=stats(only_q), only_p=stats(only_p),
        note=("нормы строк КНИГИ по тому, кто их выбирает; гипотеза проверена, "
              "если only_p заметно крупнее only_q и both"))


def selftest():
    # --- ПРАВИЛО ВЫБОРА ЦЕЛИ ---------------------------------------------
    d = target_decision(0.005, 1.0)
    assert d["target"] == "latent" and "ниже 2" in d["rule_branch"], d
    d = target_decision(0.03, 1.0)
    assert d["target"] == "latent" and "2-5" in d["rule_branch"], d
    d = target_decision(0.12, 1.0)
    assert d["target"] == "action_topk", d
    # ДОРОГОЙ КЭШ ПЕРЕВЕШИВАЕТ БОЛЬШОЙ ВЫИГРЫШ
    d = target_decision(0.12, CACHE_BUDGET_HOURS + 0.1)
    assert d["target"] == "latent_plus_action_term", d
    assert "бюджете" in d["reason"]
    # граница ровно на пороге относится к следующей ветке
    assert target_decision(0.02, 1.0)["target"] == "latent"
    assert target_decision(0.05, 1.0)["target"] == "action_topk"
    for bad in (float("nan"), float("inf")):
        try:
            target_decision(bad, 1.0)
        except ValueError as e:
            assert "не число" in str(e), e
        else:
            raise AssertionError(f"принят выигрыш {bad}")

    # --- РАЗРЕЖЕННЫЙ ВЫБОР БАТЧЕЙ ----------------------------------------
    assert strided(10, 0) == [] and strided(0, 5) == []
    assert strided(5, 10) == [0, 1, 2, 3, 4]        # больше, чем есть
    s = strided(100, 5)
    assert s[0] == 0 and s[-1] == 99 and len(s) == 5, s
    assert s == sorted(set(s))
    # РАЗРЕЖЕННО, А НЕ ПЕРВЫЕ ПОДРЯД — иначе это не выборка из части
    assert s != list(range(5)), s

    # --- РАЗБИЕНИЕ МНОЖЕСТВ КОДОВ ----------------------------------------
    norms = {0: 0.1, 1: 0.1, 2: 0.1, 7: 3.0, 8: 3.0}
    q = np.array([0, 1, 2, 2, 1])
    p = np.array([0, 7, 8, 7, 1])
    r = row_set_split(q, p, norms)
    assert r["used_by_q"] == 3 and r["used_by_p"] == 4
    assert r["both"]["count"] == 2 and r["only_q"]["count"] == 1
    assert r["only_p"]["count"] == 2
    assert abs(r["jaccard"] - 2 / 5) < 1e-12, r["jaccard"]
    # РОВНО ТА ГИПОТЕЗА: строки «только P» крупнее
    assert r["only_p"]["median"] == 3.0 and r["only_q"]["median"] == 0.1
    empty = row_set_split(np.array([0]), np.array([0]), norms)
    assert empty["only_p"]["count"] == 0 and empty["only_p"]["median"] is None

    print("самопроверка k15b_probe_and_extract пройдена")


def main():
    ap = argparse.ArgumentParser(
        description="K-15b шаг 0: выбор C1 и цели читателя")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--checkpoint", default="data/k15/depth_rvq_s0.pt",
                    help="чекпойнт K-15, из которого берутся состояния")
    ap.add_argument("--batches", type=int, default=100,
                    help="батчей val_sel на оценку, разреженно по всей части")
    ap.add_argument("--topk", type=int, default=10,
                    help="сколько ближайших латентных кодов проверять "
                         "декодированием")
    ap.add_argument("--tau-probe", type=float, default=1.0,
                    help="температура для апостериора Q в диагностике. "
                         "Жёсткий выбор от неё не зависит вовсе")
    ap.add_argument("--out", default="reports/k15b/probe_and_extract.json")
    ap.add_argument("--out-c1", default="data/k15b/c1_selected.pt")
    ap.add_argument("--overwrite", action="store_true")
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import k15_context
    from k15_train_depth_rvq import H_EXEC
    k15_context.add_common_arguments(ap)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    for path in (a.out, a.out_c1):
        if os.path.exists(path) and not a.overwrite:
            raise SystemExit(f"{path} уже существует: без --overwrite не "
                             f"перезаписываю")

    ctx = k15_context.build(a)
    torch = ctx.torch
    dev, dt = ctx.dev, ctx.dt
    model, codec = ctx.model, ctx.codec
    names = list(ctx.info["names"])
    ac16 = torch.autocast(device_type=dev.type, dtype=dt)

    # --- СОСТОЯНИЯ ИЗ ЧЕКПОЙНТА K-15 ------------------------------------
    if not os.path.exists(a.checkpoint):
        raise SystemExit(f"нет {a.checkpoint}")
    obj = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if obj.get("kind") not in ("k15_depth_rvq", "k15_smoke"):
        raise SystemExit(f"{a.checkpoint} описывает {obj.get('kind')!r}")
    if obj.get("kind") == "k15_smoke":
        print("  ВНИМАНИЕ: это smoke-чекпойнт, его числа ничего не решают")
    states = obj.get("states")
    if not isinstance(states, dict) or not states:
        raise SystemExit("в чекпойнте нет словаря states")
    if set(obj.get("trainable_names", [])) != set(names):
        raise SystemExit(
            "белый список чекпойнта не совпал с текущим: "
            f"нет {sorted(set(names) - set(obj.get('trainable_names', [])))[:4]}")
    recorded_sha = {
        k: obj.get(f"selected_state_sha1_{k}") for k in states}
    print(f"  чекпойнт {a.checkpoint}: состояний {sorted(states)}, "
          f"кандидат в нём {obj.get('candidate_level')!r}, "
          f"accepted {obj.get('accepted')!r}")

    batch_list = ctx.parts["val_sel"]
    take = strided(len(batch_list), a.batches)
    chosen_batches = [batch_list[i] for i in take]
    rows_all = np.unique(np.concatenate(
        [np.asarray(sel, np.int64) for _po, sel in chosen_batches]))
    rows_sha = hashlib.sha1(
        np.ascontiguousarray(rows_all).tobytes()).hexdigest()[:12]
    print(f"  оценка на {len(chosen_batches)} батчах val_sel из "
          f"{len(batch_list)}, {rows_all.size} строк, отпечаток {rows_sha}")

    def restore(state):
        with torch.no_grad():
            own = dict(model.state_dict())
            for k_, v_ in state.items():
                own[k_].copy_(v_.to(own[k_].device, own[k_].dtype))
        got = ctx.k14c.state_sha(
            {k_: model.state_dict()[k_].detach().float().cpu().numpy()
             for k_ in names})
        return got

    def forward(po, sel):
        """Минимальный проход: только то, что нужно диагностике."""
        b = ctx.build_batch(po, sel)
        am = b.get("attention_mask")
        with ac16:
            v, p_ids = model.build_inputs(position_offset=po, **b)
            out = model.forward_depth_aligned_rvq(
                vlm_inputs_embeds=v, attention_mask=am, position_ids=p_ids,
                mode="full", tau=1.0)
        q0_now = out["pred_codes"][0].detach().cpu().numpy()
        bad = int((q0_now != ctx.q0_can[sel]).sum())
        if bad:
            raise SystemExit(
                f"q0 разошёлся с каноническим в {bad} позициях: окружение "
                f"собрано не так, как в тренере")
        return out

    def support(latent):
        s, rel = k15_context.k15t.decoder_support(
            latent, codec, torch, ctx.quantizers, ctx.nearest_code,
            ctx.code_contribution)
        return s, rel

    def evaluate_state(tag):
        """Все числа одного состояния на выбранных батчах."""
        acc = {k: 0.0 for k in ("a0", "a1_tok", "a1_pol", "a1_soft",
                                "a2_tok", "a2_pol")}
        n_rows = 0
        sup = {k: [] for k in ("a1_tok", "a1_pol", "a1_soft", "reference")}
        codes_q, codes_p = [], []
        norm_q, norm_p, norm_z0 = [], [], []
        agree, recall3, recall10 = 0.0, 0.0, 0.0
        abs_tok = []
        model.eval()
        with torch.no_grad():
            for po, sel in chosen_batches:
                out = forward(po, sel)
                z0 = out["policy_embeddings"][0]
                c1_pol = out["policy_embeddings"][1]
                c1 = model.depth_aligned_book(1)
                c2 = model.depth_aligned_book(2)
                action = torch.from_numpy(
                    np.asarray(ctx.ACT[sel], np.float32)).to(dev)[..., :7]
                z_e = codec._encode(action.float(), embodiment_ids=0).float()
                r1 = z_e - z0.detach()
                lg1, c1_tok, i1, p1 = ctx.tok.quantize_residual(
                    r1, c1, temperature=float(a.tau_probe))
                hard_c1 = c1[out["pred_codes"][1]].detach()
                r2 = z_e - z0.detach() - hard_c1
                _lg2, c2_tok, _i2, _p2 = ctx.tok.quantize_residual(
                    r2, c2, temperature=float(a.tau_probe))
                soft1 = out["policy_probabilities"][1].float() @ c1.float()
                lat = dict(
                    a0=z0, a1_tok=z0 + c1_tok, a1_pol=z0 + c1_pol,
                    a1_soft=z0.float() + soft1,
                    a2_tok=z0 + hard_c1 + c2_tok,
                    a2_pol=out["cumulative_latents"][2])
                w = float(len(sel))
                n_rows += len(sel)
                for nm, z in lat.items():
                    _rows, mean = k15_context.k15t.weighted_row_error(
                        ctx.decode_fp32(z), action, ctx.weights_gate, torch)
                    acc[nm] += float(mean) * w
                # ОПОРА ДЛЯ ПУТИ ТОКЕНИЗАТОРА — ГЛАВНОЕ НОВОЕ ЧИСЛО
                for nm in ("a1_tok", "a1_pol", "a1_soft"):
                    sup[nm].append(support(lat[nm].detach())[1])
                sup["reference"].append(support(z_e.detach())[1])
                # кто какие строки выбирает
                codes_q.append(i1[:, :H_EXEC].reshape(-1).cpu().numpy())
                codes_p.append(out["pred_codes"][1][:, :H_EXEC]
                               .reshape(-1).cpu().numpy())
                norm_q.append(c1_tok[:, :H_EXEC].float().norm(dim=-1)
                              .reshape(-1).cpu().numpy())
                norm_p.append(c1_pol[:, :H_EXEC].float().norm(dim=-1)
                              .reshape(-1).cpu().numpy())
                norm_z0.append(z0[:, :H_EXEC].float().norm(dim=-1)
                               .reshape(-1).cpu().numpy())
                rd = k15_context.k15t.reading_stats(
                    out["policy_probabilities"][1][:, :H_EXEC],
                    i1[:, :H_EXEC], torch,
                    policy_indices=out["pred_codes"][1][:, :H_EXEC])
                agree += rd["agree_top1"] * w
                recall3 += rd["recall_at3"] * w
                recall10 += rd["recall_at10"] * w
                abs_tok.append(ctx.decode_fp32(lat["a1_tok"])[:, :H_EXEC]
                               .abs().reshape(-1, 7).cpu().numpy())
        n = max(n_rows, 1)
        res = {f"rms_{k}": float(np.sqrt(v / n)) for k, v in acc.items()}
        res["rows"] = n_rows
        ref = torch.cat(sup["reference"])
        for nm in ("a1_tok", "a1_pol", "a1_soft"):
            v = torch.cat(sup[nm])
            res[f"support_{nm}"] = dict(
                mean=float(v.mean()), median=float(v.median()),
                p95=float(torch.quantile(v, 0.95)),
                p99=float(torch.quantile(v, 0.99)),
                reference_mean=float(ref.mean()),
                reference_p95=float(torch.quantile(ref, 0.95)),
                reference_p99=float(torch.quantile(ref, 0.99)),
                share_above_reference_p99=float(
                    (v > torch.quantile(ref, 0.99)).float().mean()),
                passes_k15_rule=bool(float(torch.quantile(v, 0.95))
                                     <= float(torch.quantile(ref, 0.95))))
        cq = np.concatenate(codes_q)
        cp = np.concatenate(codes_p)
        book = model.depth_aligned_book(1).detach().float()
        book_norms = {int(i): float(x) for i, x in enumerate(
            book.norm(dim=-1).cpu().numpy())}
        res["row_sets"] = row_set_split(cq, cp, book_norms)
        for nm, arr in (("c1_tok", norm_q), ("c1_pol", norm_p),
                        ("z0", norm_z0)):
            flat = np.concatenate(arr)
            res[f"norm_{nm}"] = dict(median=float(np.median(flat)),
                                     p95=float(np.percentile(flat, 95)))
        res["agree_top1"] = agree / n
        res["recall_at3"] = recall3 / n
        res["recall_at10"] = recall10 / n
        flat_tok = np.concatenate(abs_tok, axis=0)
        res["p99_a1_tok"] = [float(x) for x in
                             np.percentile(flat_tok, 99.0, axis=0)]
        res["p99_dataset"] = [float(x) for x in ctx.act_p99_dataset]
        res["action_range_ok"] = bool(all(
            c <= 1.5 * d for c, d in zip(res["p99_a1_tok"],
                                         res["p99_dataset"])))
        print(f"    {tag}: a0 {res['rms_a0']:.6f}; книга a1_tok "
              f"{res['rms_a1_tok']:.6f}, a2_tok {res['rms_a2_tok']:.6f}; "
              f"политика a1_pol {res['rms_a1_pol']:.6f}, мягкая "
              f"{res['rms_a1_soft']:.6f}")
        st = res["support_a1_tok"]
        print(f"      ОПОРА ПУТИ ТОКЕНИЗАТОРА: остаток {st['mean']:.4f}, "
              f"p95 {st['p95']:.4f}, p99 {st['p99']:.4f} при эталонном p95 "
              f"{st['reference_p95']:.4f} — правило K-15 "
              f"{'пройдено' if st['passes_k15_rule'] else 'НЕ ПРОЙДЕНО'}")
        rs = res["row_sets"]
        print(f"      строки книги: Q использует {rs['used_by_q']}, P "
              f"{rs['used_by_p']}, Jaccard {rs['jaccard']:.3f}; нормы "
              f"медиана — оба {rs['both']['median']}, только Q "
              f"{rs['only_q']['median']}, только P {rs['only_p']['median']}")
        print(f"      чтение: согласие {100 * res['agree_top1']:.2f}%, "
              f"recall@3 {100 * res['recall_at3']:.1f}%, recall@10 "
              f"{100 * res['recall_at10']:.1f}%; диапазон a1_tok "
              f"{'ok' if res['action_range_ok'] else 'ОТКАЗ'}")
        return res

    # --- ОЦЕНКА ВСЕХ СОХРАНЁННЫХ СОСТОЯНИЙ -------------------------------
    per_state = {}
    for tag in sorted(states):
        got_sha = restore(states[tag])
        want = recorded_sha.get(tag)
        if want and got_sha != want:
            raise SystemExit(
                f"{tag}: после загрузки отпечаток {got_sha}, в чекпойнте "
                f"{want}")
        per_state[tag] = dict(state_sha1=got_sha, epoch=obj.get(
            f"selected_epoch_{tag}"), **evaluate_state(tag))

    best_tag = min(per_state, key=lambda t: (
        round(per_state[t]["rms_a1_tok"], 12), t))
    print(f"\n  ЛУЧШАЯ КНИГА: {best_tag} (a1_tok "
          f"{per_state[best_tag]['rms_a1_tok']:.6f}), выбрано по качеству "
          f"СЛОВАРЯ, а не читателя")

    # --- ACTION PROBE ВНУТРИ ЛАТЕНТНОГО СОСЕДСТВА ------------------------
    restore(states[best_tag])
    k_probe = int(a.topk)
    probe_acc = dict(latent=0.0, action_best=0.0, second=0.0)
    probe_rows, changed, margins, gains = 0, 0, [], []
    flips_probe = 0
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for po, sel in chosen_batches:
            out = forward(po, sel)
            z0 = out["policy_embeddings"][0]
            c1 = model.depth_aligned_book(1)
            action = torch.from_numpy(
                np.asarray(ctx.ACT[sel], np.float32)).to(dev)[..., :7]
            z_e = codec._encode(action.float(), embodiment_ids=0).float()
            r1 = z_e - z0.detach()
            d1 = ctx.tok.mean_squared_distances(r1, c1)
            near = (-d1).topk(k_probe, dim=-1).indices
            # СТОЛБЕЦ 0 — КОД ТОКЕНИЗАТОРА, А НЕ ВЕРХНИЙ ПО topk. В K-15
            # измерено, что argmax и topk расходятся в единичных позициях
            # (логиты в fp16), и тогда инвариант «ранг 1 = a1_tok» ломался
            # бы не по делу. Подстановка делает его точным по построению.
            _lg, _emb, i1_probe, _p = ctx.tok.quantize_residual(
                r1, c1, temperature=float(a.tau_probe))
            flips_probe += int((near[..., 0] != i1_probe).sum())
            near = torch.cat([i1_probe.unsqueeze(-1), near[..., 1:]], dim=-1)
            errs = []
            for j in range(k_probe):
                rows_j, _m = k15_context.k15t.weighted_row_error(
                    ctx.decode_fp32(z0 + c1[near[..., j]]), action,
                    ctx.weights_gate, torch)
                errs.append(rows_j)
            stack = torch.stack(errs, 0)                      # (k, rows)
            order = stack.argsort(dim=0)
            best = stack.min(0)
            second = stack.sort(dim=0).values[1]
            w = float(len(sel))
            probe_rows += len(sel)
            probe_acc["latent"] += float(stack[0].mean()) * w
            probe_acc["action_best"] += float(best.values.mean()) * w
            probe_acc["second"] += float(second.mean()) * w
            changed += int((order[0] != 0).sum())
            margins.append((second - best.values).cpu().numpy())
            gains.append((stack[0] - best.values).cpu().numpy())
    elapsed = time.time() - t0
    den = max(probe_rows, 1)
    rms_latent = float(np.sqrt(probe_acc["latent"] / den))
    rms_best = float(np.sqrt(probe_acc["action_best"] / den))
    rel_gain = (rms_latent - rms_best) / max(rms_latent, 1e-12)
    gains = np.concatenate(gains)
    margins = np.concatenate(margins)
    per_batch = elapsed / max(len(chosen_batches), 1)
    full_cache_hours = per_batch * len(ctx.parts_full["train"]) / 3600.0
    probe = dict(
        kind="action probe внутри латентного соседства",
        not_a_full_oracle=("лучший по действию код может лежать ВНЕ "
                           f"{k_probe} ближайших латентных строк; §50.2 "
                           f"этим не закрыт"),
        topk=k_probe, rows=probe_rows, batches=len(chosen_batches),
        rows_sha1=rows_sha,
        rms_latent_argmin=rms_latent, rms_action_best=rms_best,
        rms_second_best=float(np.sqrt(probe_acc["second"] / den)),
        rank1_flips_vs_topk=int(flips_probe),
        relative_gain=float(rel_gain),
        share_rows_code_changed=float(changed / den),
        gain_quantiles={q: float(np.percentile(gains, q))
                        for q in (50, 75, 90, 95, 99)},
        margin_quantiles={q: float(np.percentile(margins, q))
                          for q in (50, 75, 90, 95, 99)},
        seconds=float(elapsed), seconds_per_batch=float(per_batch),
        decodes_per_batch=k_probe,
        projected_full_cache_hours=float(full_cache_hours),
        train_batches_in_plan=len(ctx.parts_full["train"]))
    # ИНВАРИАНТ: путь ранга 1 — это и есть путь токенизатора, и обе
    # величины посчитаны на ОДНИХ строках. Расхождение означало бы, что
    # probe считается не на том.
    want_tok = per_state[best_tag]["rms_a1_tok"]
    if abs(rms_latent - want_tok) / max(want_tok, 1e-12) > 1e-6:
        raise SystemExit(
            f"латентный ранг 1 дал {rms_latent!r}, а путь a1_tok на тех же "
            f"строках {want_tok!r}: probe считается не на тех строках или "
            f"не по той величине")
    decision = target_decision(rel_gain, full_cache_hours)
    print(f"\n  ACTION PROBE (top-{k_probe}, {probe_rows} строк): латентный "
          f"argmin {rms_latent:.6f} -> лучший по действию {rms_best:.6f}, "
          f"выигрыш {100 * rel_gain:.1f} %")
    print(f"    код сменился на {100 * probe['share_rows_code_changed']:.1f} % "
          f"строк; медианный выигрыш {probe['gain_quantiles'][50]:.2e}, "
          f"p95 {probe['gain_quantiles'][95]:.2e}")
    print(f"    {per_batch:.2f} с/батч -> полный кэш train оценён в "
          f"{full_cache_hours:.1f} ч при бюджете {CACHE_BUDGET_HOURS:.1f} ч")
    print(f"    РЕШЕНИЕ ПО ПРАВИЛУ: цель читателя = {decision['target']} "
          f"({decision['reason']})")

    # --- ИЗВЛЕЧЕНИЕ КНИГИ ------------------------------------------------
    c1_sel = model.depth_aligned_book(1).detach().cpu().clone()
    c1_sha = hashlib.sha1(np.ascontiguousarray(
        c1_sel.float().numpy()).tobytes()).hexdigest()[:12]
    payload = dict(
        kind="k15b_c1_selected",
        c1=c1_sel, c1_sha1=c1_sha, c1_shape=list(c1_sel.shape),
        source_checkpoint=os.path.abspath(a.checkpoint),
        source_checkpoint_sha1=k15_context.sha12(a.checkpoint),
        source_state=best_tag,
        source_epoch=per_state[best_tag]["epoch"],
        source_state_sha1=per_state[best_tag]["state_sha1"],
        rms_a1_tok=per_state[best_tag]["rms_a1_tok"],
        support_a1_tok=per_state[best_tag]["support_a1_tok"],
        action_range_ok=per_state[best_tag]["action_range_ok"],
        target_decision=decision,
        rows_sha1=rows_sha, rows=int(rows_all.size),
        selection_rule="минимум rms_a1_tok среди сохранённых состояний",
        codec=ctx.codec_fp, code_version=ctx.code_version,
        q0_prov=ctx.q0_prov, joint_sha1=ctx.joint_sha,
        git_head=ctx.git_head, git_dirty=bool(ctx.dirty),
        **ctx.gate_info)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_c1)) or ".",
                exist_ok=True)
    tmp = a.out_c1 + f".tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, a.out_c1)
    back = torch.load(a.out_c1, map_location="cpu", weights_only=False)
    if not torch.equal(back["c1"], c1_sel):
        raise SystemExit("книга после обратного чтения отличается")
    if back["c1_sha1"] != c1_sha:
        raise SystemExit("отпечаток книги после чтения не совпал")
    print(f"  книга сохранена: {a.out_c1} ({c1_sha})")

    # СТОЛКНОВЕНИЕ ИМЁН ЛОВИТСЯ НАЗВАННЫМ ОТКАЗОМ, А НЕ TypeError: на
    # этом уже падал тренер (`decoder_context` приходит из gate_info).
    own_keys = {"kind", "checkpoint", "checkpoint_kind", "batches",
                "batches_in_part", "rows", "rows_sha1", "tau_probe",
                "per_state", "best_state", "probe", "target_decision",
                "target_rule", "cache_budget_hours", "c1_file", "c1_sha1",
                "codec", "code_version", "q0_prov", "joint_sha1",
                "git_head", "git_dirty"}
    clash = sorted(set(ctx.gate_info) & own_keys)
    if clash:
        raise SystemExit(f"ключи {clash} из gate_info сталкиваются с полями "
                         f"отчёта")
    report = dict(
        kind="k15b_probe_and_extract",
        checkpoint=os.path.abspath(a.checkpoint),
        checkpoint_kind=obj.get("kind"),
        batches=len(chosen_batches), batches_in_part=len(batch_list),
        rows=int(rows_all.size), rows_sha1=rows_sha,
        tau_probe=float(a.tau_probe),
        per_state=per_state, best_state=best_tag,
        probe=probe, target_decision=decision,
        target_rule=[dict(below=t if np.isfinite(t) else None,
                          choice=c, why=w) for t, c, w in TARGET_RULE],
        cache_budget_hours=CACHE_BUDGET_HOURS,
        c1_file=os.path.abspath(a.out_c1), c1_sha1=c1_sha,
        codec=ctx.codec_fp, code_version=ctx.code_version,
        q0_prov=ctx.q0_prov, joint_sha1=ctx.joint_sha,
        git_head=ctx.git_head, git_dirty=bool(ctx.dirty),
        **ctx.gate_info)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                exist_ok=True)
    tmp = a.out + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1,
                  default=k15_context.k15t.json_scalar)
    os.replace(tmp, a.out)
    print(f"  отчёт: {a.out}")

    st = per_state[best_tag]["support_a1_tok"]
    if not st["passes_k15_rule"]:
        print("\n  ПОСЫЛКА K-15b ПОД ВОПРОСОМ: путь токенизатора сам вне "
              "многообразия декодера (p95 остатка "
              f"{st['p95']:.4f} против эталонного {st['reference_p95']:.4f}). "
              "Значит a1_tok частично экстраполяция декодера, и учить "
              "читателя попадать в эти коды нельзя без оговорки. Разумнее "
              "взять книгу из более ранней эпохи с целой опорой.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
