#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15c: проверка вывода настоящим проходом модели по val_sel.

ЗАЧЕМ. Голова обучалась и оценивалась по КЭШУ: признаки сняты заранее, а
затраты восьми путей посчитаны построителем. Здесь тот же выбор делается так,
как он делался бы при развёртывании — один проход VLA, хук на h24, тот же
устойчивый порядок q1, голова, ОДИН декод выбранного пути — и сверяется с
кэшем. Это и есть полный inference round-trip, которого раньше не было.

ЧТО СВЕРЯЕТСЯ НА КАЖДОМ БАТЧЕ

    q0                         побитово с каноническим
    top-8 коды                 побитово с кэшем
    лог-вероятности top-8      с кэшем, допуск LOGPROB_ABS
    логиты головы              с кэшевыми, допуск SCORE_ABS
    выбранные ранги            ТОЧНО, ни одного расхождения
    построчная ошибка          с rank_costs[строка, выбранный ранг]
    черновик                   с draft_cost

и по итогам — hard RMS и доля разрыва против оценки по кэшу, опора декодера
(парная, на тех же строках) и диапазон действий, побитовый отпечаток
замороженного до и после.

ОШИБКА ДЕКОДА СВЕРЯЕТСЯ С ДОПУСКОМ, А НЕ ПОБИТОВО. В кэше каждый ранг
декодировался целым батчем, здесь в одном батче смешаны разные ранги, а
декодер не строго независим по строкам: в K-15 измерено расхождение
порядка 4e-6 при другом составе батча. Коды и выбор — побитово, ошибка
строки — по правилу |live - cached| <= COST_ATOL + COST_RTOL * |cached|,
с записью и абсолютного, и относительного максимума.

ПУЛИНГ — ПО h24, ПРИВЕДЁННОМУ К fp16, как в кэше. Голова обучалась на
сохранённом в fp16 состоянии, и путь вывода определён так же; построитель
сообщает число значений, которые fp16 меняет (при прямом проходе в fp16 их
ноль), так что это определение, а не поправка.

КОДЫ: 0 — вывод воспроизвёл кэшевую оценку и прошёл опору и диапазон;
3 — что-то не сошлось, научных выводов по голове не делаем.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

LOGPROB_ABS = 1e-5
SCORE_ABS = 1e-4
# ОШИБКА СТРОКИ СВЕРЯЕТСЯ КАК |live - cached| <= ATOL + RTOL * |cached|.
# Только относительная мера ломалась бы на строке с почти нулевой MSE:
# безобидное абсолютное расхождение декодера (порядка 4e-6 в действии, то
# есть порядка 1e-8..1e-6 в MSE строки) давало бы там огромную долю.
COST_RTOL = 1e-4
COST_ATOL = 1e-7
RMS_REL = 1e-5


def compare_rows(live, cached, rtol=COST_RTOL, atol=COST_ATOL):
    """(макс. абсолютное, макс. относительное, строк за пределом).

    Предел — atol + rtol * |cached|, как в numpy.isclose. Оба максимума
    записываются: по одному абсолютному не видно масштаба, по одному
    относительному — ложных тревог на почти нулевых строках.
    """
    lv = np.asarray(live, np.float64)
    cv = np.asarray(cached, np.float64)
    if lv.size == 0:
        return 0.0, 0.0, 0
    d = np.abs(lv - cv)
    rel = d / np.maximum(np.abs(cv), 1e-12)
    over = int((d > atol + rtol * np.abs(cv)).sum())
    return float(d.max()), float(rel.max()), over


def roundtrip_verdict(checks):
    """Все проверки обязательны; список провалов — в исход."""
    failed = sorted(k for k, v in checks.items() if not v)
    return dict(code=0 if not failed else 3, failed=failed,
                passed=not failed)


def selftest():
    a_, r_, n = compare_rows([1.0, 2.0, 3.0001], [1.0, 2.0, 3.0])
    assert n == 0 and r_ < 1e-4 and abs(a_ - 1e-4) < 1e-9, (a_, r_, n)
    a_, r_, n = compare_rows([1.0, 2.1], [1.0, 2.0])
    assert n == 1 and abs(r_ - 0.05) < 1e-12, (a_, r_, n)
    assert compare_rows([], []) == (0.0, 0.0, 0)
    # ПОЧТИ НУЛЕВАЯ СТРОКА: огромная относительная доля, но абсолютно
    # безобидно — не считается расхождением
    a_, r_, n = compare_rows([2e-8], [1e-9])
    assert n == 0 and r_ > 10.0 and a_ < 1e-7, (a_, r_, n)
    # а настоящее расхождение на той же строке ловится
    a_, r_, n = compare_rows([5e-7], [1e-9])
    assert n == 1, (a_, r_, n)
    v = roundtrip_verdict(dict(a=True, b=True))
    assert v["code"] == 0 and v["passed"]
    v = roundtrip_verdict(dict(a=True, b=False, c=False))
    assert v["code"] == 3 and v["failed"] == ["b", "c"]
    assert SCORE_ABS >= LOGPROB_ABS
    print("самопроверка k15c_check_inference пройдена")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import k15c_build_rank_cache as cb
    ap = argparse.ArgumentParser(
        description="K-15c: проверка вывода настоящим проходом")
    cb.add_stack_arguments(ap)
    # НЕ `--cache`: это имя уже занято общими аргументами k15_context (кэш
    # K-11a), и argparse отказывался бы собирать парсер.
    ap.add_argument("--rank-cache", default="data/k15c/rank_cache")
    ap.add_argument("--selector", required=False, default="")
    ap.add_argument("--summary", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not a.selector:
        raise SystemExit("--selector обязателен: путь к чекпойнту головы")
    if int(a.limit) != 0:
        raise SystemExit("--limit не применяется: проверяется вся val_sel")

    import k15b_measure_soft as ms
    import k15b_probe_and_extract as probe
    import k15c_rank_selector as rs
    import k15c_train_rank_selector as tr
    import torch as _torch
    import torch.nn.functional as F

    sel_obj = _torch.load(a.selector, map_location="cpu", weights_only=False)
    if sel_obj.get("kind") != "k15c_rank_selector":
        raise SystemExit(f"{a.selector}: kind {sel_obj.get('kind')!r}")
    if sel_obj.get("smoke"):
        raise SystemExit("голова обучена на smoke-кэше: проверять её "
                         "настоящим проходом бессмысленно")
    head_name = sel_obj["head"]
    summary = a.summary or f"reports/k15c/inference_{head_name}.json"
    if os.path.exists(summary) and not a.overwrite:
        raise SystemExit(f"{summary} уже существует: без --overwrite не "
                         f"перезаписываю")
    if head_name.startswith("h18"):
        raise SystemExit("голова на h18: для неё проверка вывода другая, "
                         "а развёртываемая архитектура — h24")

    man, problems = cb.validate_cache(a.rank_cache)
    if problems:
        raise SystemExit("кэш не принят: " + "; ".join(problems[:6]))
    if cb.sha_file(os.path.join(a.rank_cache, "manifest.json")) \
            != sel_obj.get("cache_manifest_sha1"):
        raise SystemExit("голова обучалась на другом кэше")
    # ЯВНО, А НЕ ТОЛЬКО ЧЕРЕЗ ОТПЕЧАТОК МАНИФЕСТА: книга и читатель головы
    # сверяются с кэшем напрямую, как и обещает отчёт.
    for key, ck_key in (("c1_sha1", "c1_sha1"),
                        ("reader_state_sha1", "reader_state_sha1")):
        if sel_obj.get(ck_key) != man.get(key):
            raise SystemExit(f"{key}: у головы {sel_obj.get(ck_key)!r}, в "
                             f"кэше {man.get(key)!r}")
    if str(a.selector).endswith(".technical_fail.pt"):
        raise SystemExit("это чекпойнт головы с техническим отказом")

    S = cb.load_stack(a)
    ctx, torch, model, k15t = S.ctx, S.torch, S.model, S.k15t
    for key, want, got in (("c1_sha1", man["c1_sha1"], S.book["c1_sha1"]),
                           ("reader_state_sha1", man["reader_state_sha1"],
                            S.point_sha),
                           ("frozen_content_sha", man["frozen_content_sha"],
                            S.frozen_sha),
                           ("plan_sha1", man["plan_sha1"],
                            ctx.q0_prov.get("plan_sha1")),
                           ("codec", man["codec"], ctx.codec_fp),
                           ("decoder_context", man["decoder_context"],
                            ctx.decoder_context)):
        if want != got:
            raise SystemExit(f"{key}: в кэше {want!r}, сейчас {got!r}")
    dev = ctx.dev
    c1 = model.depth_aligned_book(1)
    head = rs.build_head(head_name, int(sel_obj["d_model"]),
                         int(sel_obj["e_dim"]), torch,
                         proj=int(sel_obj["proj"]),
                         book=c1.detach().float()).to(dev)
    head.load_state_dict(sel_obj["state"])
    head.eval()

    def arr(name):
        return np.load(os.path.join(
            a.rank_cache, man["arrays"][f"val_sel_{name}"]["file"]),
            mmap_mode="r")

    c_rows = np.asarray(arr("rows"))
    slot = {int(r): i for i, r in enumerate(c_rows)}
    c_codes = np.asarray(arr("top8_codes"))
    c_lp = np.asarray(arr("top8_logprobs"))
    c_costs = np.asarray(arr("rank_costs"), np.float64)
    c_draft = np.asarray(arr("draft_cost"), np.float64)
    c_teacher = np.asarray(arr("teacher_cost"), np.float64)
    c_tasks = np.asarray(arr("task_ids"))

    # ОЦЕНКА ПО КЭШУ — ТА ЖЕ ФУНКЦИЯ, ЧТО У ТРЕНЕРА, НА ТЕХ ЖЕ ПРИЗНАКАХ.
    h24_c = arr("h24")
    ctx_c = torch.cat([rs.ln_mean_pool(torch.from_numpy(
        np.asarray(h24_c[s:s + 4096], np.float32)), torch)
        for s in range(0, h24_c.shape[0], 4096)], 0)
    emb_c = rs.candidate_embeddings(torch.from_numpy(
        c_codes.astype(np.int64)), c1.detach().float().cpu(), torch)
    feat_c = rs.candidate_score_features(torch.from_numpy(
        np.array(c_lp, np.float32)), torch)
    inp_c = tr.Inputs(
        torch, ctx_c, emb_c, feat_c,
        h_full=h24_c if head_name in rs.NEEDS_FULL_H24 else None,
        device=dev,
        cand_codes=(torch.from_numpy(c_codes.astype(np.int64))
                    if head_name in rs.NEEDS_BOOK else None))
    scores_c = tr.scores_for(head, inp_c, np.arange(len(c_rows)), torch)
    eval_c = tr.evaluate_scores(scores_c, c_costs, c_draft, c_teacher)
    pick_c = scores_c.argmax(1)
    print(f"  оценка по кэшу: hard RMS {eval_c['rms']:.6f}, доля "
          f"{100 * (eval_c['capture'] or 0):.1f} %")

    hooks = cb.StateHooks(model)
    q0_dev = torch.as_tensor(np.asarray(ctx.q0_can), device=dev)
    ac16 = torch.autocast(device_type=dev.type, dtype=ctx.dt)
    stats = dict(code_mismatch=0, lp_max=0.0, score_max=0.0,
                 pick_mismatch=0, cost_abs_max=0.0, cost_rel_max=0.0,
                 cost_over=0, draft_abs_max=0.0, draft_rel_max=0.0,
                 draft_over=0)
    live_sel = np.full(len(c_rows), np.nan)
    sup_sel, sup_ref, abs_act = [], [], []
    seen = np.zeros(len(c_rows), bool)
    t0 = time.time()
    with torch.no_grad():
        for po, sel in ctx.parts_full["val_sel"]:
            b = ctx.build_batch(po, sel)
            with ac16:
                v, p_ids = model.build_inputs(position_offset=po, **b)
                out = model.forward_depth_aligned_rvq(
                    vlm_inputs_embeds=v,
                    attention_mask=b.get("attention_mask"),
                    position_ids=p_ids, mode="full", tau=1.0)
            if int((out["pred_codes"][0]
                    != q0_dev[torch.as_tensor(sel, device=dev)]).sum()):
                raise SystemExit("q0 разошёлся с каноническим")
            _h18, h24 = hooks.take()
            ii = np.asarray([slot[int(r)] for r in sel], np.int64)
            if seen[ii].any():
                raise SystemExit("строка val_sel встречена дважды")
            seen[ii] = True
            lg = out["logits"][1].float()
            top8 = ms.reader_order(lg, torch)[..., :8]
            stats["code_mismatch"] += int(
                (top8.cpu().numpy() != c_codes[ii].astype(np.int64)).sum())
            lp8 = F.log_softmax(lg, dim=-1).gather(-1, top8)
            stats["lp_max"] = max(stats["lp_max"], float(
                np.abs(lp8.cpu().numpy() - c_lp[ii]).max()))
            h24s = h24.half().float()
            kw = dict(ctx=rs.ln_mean_pool(h24s, torch),
                      cand_emb=rs.candidate_embeddings(top8, c1, torch),
                      cand_feat=rs.candidate_score_features(lp8, torch),
                      h_full=h24s, cand_codes=top8)
            sc = head(**kw).float()
            stats["score_max"] = max(stats["score_max"], float(
                np.abs(sc.cpu().numpy() - scores_c[ii]).max()))
            pick = sc.argmax(-1)
            stats["pick_mismatch"] += int(
                (pick.cpu().numpy() != pick_c[ii]).sum())
            codes_sel = top8.gather(
                -1, pick.view(-1, 1, 1).expand(-1, top8.shape[1], 1)
            ).squeeze(-1)
            z0 = out["policy_embeddings"][0]
            z_sel = z0 + c1[codes_sel]
            action = torch.from_numpy(np.asarray(
                ctx.ACT[sel], np.float32)).to(dev)[..., :7]
            act_sel = ctx.decode_fp32(z_sel)              # ОДИН декод
            rows_sel, _m = k15t.weighted_row_error(act_sel, action,
                                                   ctx.weights_gate, torch)
            rows_d, _m0 = k15t.weighted_row_error(
                ctx.decode_fp32(z0), action, ctx.weights_gate, torch)
            rl = rows_sel.cpu().numpy()
            live_sel[ii] = rl
            ac_, rc_, o_c = compare_rows(rl, c_costs[ii,
                                                     pick.cpu().numpy()])
            stats["cost_abs_max"] = max(stats["cost_abs_max"], ac_)
            stats["cost_rel_max"] = max(stats["cost_rel_max"], rc_)
            stats["cost_over"] += o_c
            ad_, rd_, o_d = compare_rows(rows_d.cpu().numpy(), c_draft[ii])
            stats["draft_abs_max"] = max(stats["draft_abs_max"], ad_)
            stats["draft_rel_max"] = max(stats["draft_rel_max"], rd_)
            stats["draft_over"] += o_d
            z_e = cb.codec_encode(ctx, action)
            sup_sel.append(k15t.decoder_support(
                z_sel.detach(), ctx.codec, torch, ctx.quantizers,
                ctx.nearest_code, ctx.code_contribution)[1])
            sup_ref.append(k15t.decoder_support(
                z_e.detach(), ctx.codec, torch, ctx.quantizers,
                ctx.nearest_code, ctx.code_contribution)[1])
            abs_act.append(act_sel[:, :8].abs().reshape(-1, 7)
                           .cpu().numpy())
    hooks.remove()
    if not seen.all():
        raise SystemExit(f"не пройдено {int((~seen).sum())} строк val_sel")
    again, _n, _e = k15t.frozen_content_sha(model, torch,
                                            set(S.train_names))
    rms_live = tr.rms(live_sel)
    cap_live = tr.capture(tr.rms(c_draft), tr.rms(c_teacher), rms_live)
    # ПО ЗАДАЧАМ — ИЗ ЖИВЫХ ОШИБОК, черновик и учитель из кэша.
    vocab_t = man.get("task_vocab") or []
    per_task_live = {}
    for tid in np.unique(c_tasks):
        m = c_tasks == tid
        name = vocab_t[int(tid)] if int(tid) < len(vocab_t) else str(tid)
        per_task_live[name] = dict(
            rows=int(m.sum()), rms=tr.rms(live_sel[m]),
            capture=tr.capture(tr.rms(c_draft[m]), tr.rms(c_teacher[m]),
                               tr.rms(live_sel[m])))
    s_sel, s_ref = torch.cat(sup_sel), torch.cat(sup_ref)
    p95_sel = float(torch.quantile(s_sel, 0.95))
    p95_ref = float(torch.quantile(s_ref, 0.95))
    flat = np.concatenate(abs_act, 0)
    p99 = [float(x) for x in np.percentile(flat, 99.0, axis=0)]
    amax = [float(x) for x in flat.max(axis=0)]
    rng_ = probe.range_ok(p99, ctx.act_p99_dataset, amax)
    rel_rms = abs(rms_live - eval_c["rms"]) / max(eval_c["rms"], 1e-12)
    checks = dict(
        top8_codes_equal=stats["code_mismatch"] == 0,
        logprobs_within=stats["lp_max"] <= LOGPROB_ABS,
        scores_within=stats["score_max"] <= SCORE_ABS,
        picks_equal=stats["pick_mismatch"] == 0,
        costs_within=stats["cost_over"] == 0,
        draft_within=stats["draft_over"] == 0,
        rms_reproduced=rel_rms <= RMS_REL,
        support_passed=p95_sel <= p95_ref,
        range_passed=bool(rng_["passed"]),
        frozen_unchanged=again == S.frozen_sha)
    verdict = roundtrip_verdict(checks)
    print(f"  настоящий проход за {(time.time() - t0) / 60:.1f} мин: hard "
          f"RMS {rms_live:.6f} (по кэшу {eval_c['rms']:.6f}, относительно "
          f"{rel_rms:.1e}), доля {100 * (cap_live or 0):.1f} %")
    print(f"    коды top-8 расходятся в {stats['code_mismatch']}; "
          f"лог-вероятности до {stats['lp_max']:.1e}; логиты головы до "
          f"{stats['score_max']:.1e}; выбор расходится в "
          f"{stats['pick_mismatch']} строках")
    print(f"    ошибка выбранного пути против кэша: до "
          f"{stats['cost_abs_max']:.1e} абсолютно и "
          f"{stats['cost_rel_max']:.1e} относительно; за пределом "
          f"{COST_ATOL:g} + {COST_RTOL:g}*|кэш| — {stats['cost_over']} "
          f"строк; черновик до {stats['draft_abs_max']:.1e} абсолютно, "
          f"за пределом {stats['draft_over']}")
    frozen_s = ("не двигалось" if checks["frozen_unchanged"]
                else "ИЗМЕНИЛОСЬ")
    print(f"    опора p95 {p95_sel:.4f} при эталоне {p95_ref:.4f} на тех же "
          f"строках; диапазон {'ok' if rng_['passed'] else 'ОТКАЗ'}; "
          f"замороженное {frozen_s}")
    out_s = ("вывод воспроизвёл кэш" if verdict["passed"]
             else "НЕ СОШЛОСЬ: " + str(verdict["failed"]))
    print(f"  ИСХОД: {out_s} (код {verdict['code']})")
    out = dict(kind="k15c_inference_check", head=head_name,
               selector=os.path.abspath(a.selector), verdict=verdict,
               checks=checks, stats=stats, rms_live=rms_live,
               capture_live=cap_live, cache_eval=eval_c,
               live_per_task=per_task_live,
               support=dict(p95=p95_sel, reference_p95=p95_ref,
                            n=int(s_sel.numel())),
               range=dict(p99_candidate=p99, absmax_candidate=amax, **rng_),
               tolerances=dict(logprob_abs=LOGPROB_ABS, score_abs=SCORE_ABS,
                               cost_rtol=COST_RTOL, cost_atol=COST_ATOL,
                               rms_rel=RMS_REL),
               decodes_per_batch_at_inference=1,
               cache_manifest_sha1=cb.sha_file(
                   os.path.join(a.rank_cache, "manifest.json")),
               git_head=ctx.git_head, git_dirty=bool(ctx.dirty))
    os.makedirs(os.path.dirname(os.path.abspath(summary)) or ".",
                exist_ok=True)
    tmp = summary + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, allow_nan=False,
                  default=k15t.json_scalar)
    os.replace(tmp, summary)
    print(f"  сводка: {summary}")
    return int(verdict["code"])


if __name__ == "__main__":
    sys.exit(main())
