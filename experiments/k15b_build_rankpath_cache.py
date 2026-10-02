#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15b, шаг 1: предпосчёт учительской цели rank-path по всему train.

ЗАЧЕМ ПРЕДПОСЧЁТ. При ЗАМОРОЖЕННОЙ книге C1 и каноничном q0 целевой код
каждой строки — ФИКСИРОВАННАЯ величина, не зависящая от обучения. Считать
её заново на каждой эпохе значило бы платить десять декодов на батч за
неизменный ответ. Здесь она считается один раз (~2 ч по оценке probe) и
пишется в кэш с отпечатком.

ЧТО ТАКОЕ ЦЕЛЬ. Для строки берутся `--topk` РАЗЛИЧНЫХ ближайших по
латентному расстоянию кодов на каждой позиции чанка; траектория j
использует j-й код на ВСЕХ позициях; выбирается траектория с наименьшей
взвешенной ошибкой действия. Это НЕ action-оптимум: смеси рангов по
позициям не перебираются (их k^T), коды вне соседства не рассматриваются.
§50.2 этим не закрыт, и в метаданных кэша это записано.

ДВЕ РАЗНЫЕ ОСИ, КОТОРЫЕ НЕЛЬЗЯ СМЕШИВАТЬ. Действие имеет 20 временных
шагов, из которых в метрику входят первые H_EXEC = 8. Латент имеет 16
КОДОВЫХ позиций. Соответствия «первый код — первый шаг действия» НЕТ:
`PerceiverDecoder` смотрит на все 16 латентных токенов через
cross-attention без каузальной маски (`decoder_add_causal_mask = False`,
шесть слоёв), поэтому коды позиций 8-15 влияют и на первые 8 шагов.

Отсюда контракт:

    выбор ранга и action-член  -> первые 8 ШАГОВ ДЕЙСТВИЯ
    кросс-энтропия читателя    -> ВСЕ 16 КОДОВЫХ ПОЗИЦИЙ
    записанные коды            -> все 16, и это правильно

Траектория ранга j подставляет j-й код во ВСЕ 16 позиций, и её малая
ошибка на первых восьми шагах — результат совместного действия всех
шестнадцати кодов. Поэтому учить читателя только первым восьми кодам
значило бы выбросить часть того, чем эта ошибка получена. Поля
`action_error_positions`, `code_target_positions` и `ce_target_positions`
в метаданных говорят это прямо.

КНИГА СВЕРЯЕТСЯ, А НЕ ПОДРАЗУМЕВАЕТСЯ. Кэш осмыслен только для той C1,
на которой построен, поэтому пишется её отпечаток, а при несовпадении с
моделью книга ЗАГРУЖАЕТСЯ из артефакта probe и отпечаток проверяется
снова. Тренер обязан сверить `c1_sha1` кэша со своей книгой.
"""
import argparse
import hashlib
import inspect
import json
import os
import sys
import time

import numpy as np


def build_row_index(batches):
    """(строки, отображение строка->слот). Повтор строки — отказ.

    План делит строки между частями без пересечений, но проверяется это,
    а не предполагается: повтор означал бы, что одна строка получит две
    цели, и какая попадёт в кэш — вопрос порядка обхода.
    """
    seen, order = {}, []
    for po, sel in batches:
        for r in np.asarray(sel, np.int64).tolist():
            if r in seen:
                raise SystemExit(
                    f"строка {r} встречается в плане дважды: цель стала бы "
                    f"зависеть от порядка обхода")
            seen[r] = len(order)
            order.append(int(r))
    rows = np.asarray(order, np.int64)
    return rows, seen


def cache_fingerprint(rows, codes, ranks):
    """Отпечаток содержимого кэша: строки, коды и выбранные ранги."""
    h = hashlib.sha1()
    for arr in (np.ascontiguousarray(rows, np.int64),
                np.ascontiguousarray(codes, np.int32),
                np.ascontiguousarray(ranks, np.int16)):
        h.update(arr.tobytes())
    return h.hexdigest()[:12]


def save_npz(path, **arrays):
    """Запись через ФАЙЛОВЫЙ ДЕСКРИПТОР и обратное чтение.

    `np.savez` по ИМЕНИ дописывает `.npz`, из-за чего `os.replace` потом не
    находит файл. В этом проекте ловушка срабатывала трижды, поэтому здесь
    открытый дескриптор и сверка после чтения.
    """
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    back = np.load(tmp, allow_pickle=True)
    for k, v in arrays.items():
        got = back[k]
        if isinstance(v, np.ndarray):
            if not np.array_equal(got, v):
                os.unlink(tmp)
                raise SystemExit(f"{k}: после чтения не совпало")
        elif str(got) != str(v):
            os.unlink(tmp)
            raise SystemExit(f"{k}: после чтения {got!r} против {v!r}")
    back.close()
    os.replace(tmp, path)


def archive_existing(path):
    """Старый кэш уходит в версию со штампом, а не переписывается в конце."""
    if not os.path.exists(path):
        return None
    dest = f"{path}.{time.strftime('%Y%m%dT%H%M%S')}.bak"
    if os.path.exists(dest):
        dest = f"{dest}.{os.getpid()}"
    os.replace(path, dest)
    return dest


def forecast(elapsed, done, total):
    """Прогноз по сделанному. Чистая функция."""
    if done <= 0:
        raise ValueError("нет сделанных батчей, прогнозировать нечего")
    per = elapsed / done
    return dict(per_batch_s=float(per), total_h=float(per * total / 3600.0),
                remaining_min=float(per * (total - done) / 60.0))


def selftest():
    # --- ИНДЕКС СТРОК --------------------------------------------------
    rows, idx = build_row_index([(3, [5, 7]), (3, [9])])
    assert rows.tolist() == [5, 7, 9]
    assert idx == {5: 0, 7: 1, 9: 2}
    try:
        build_row_index([(3, [5, 7]), (3, [7])])
    except SystemExit as e:
        assert "дважды" in str(e), e
    else:
        raise AssertionError("повтор строки принят")

    # --- ОТПЕЧАТОК РЕАГИРУЕТ НА КАЖДУЮ ЧАСТЬ ----------------------------
    r = np.array([1, 2], np.int64)
    c = np.array([[1, 2], [3, 4]], np.int32)
    j = np.array([0, 1], np.int16)
    base = cache_fingerprint(r, c, j)
    assert base == cache_fingerprint(r.copy(), c.copy(), j.copy())
    assert base != cache_fingerprint(np.array([1, 3], np.int64), c, j)
    assert base != cache_fingerprint(r, np.array([[1, 2], [3, 5]], np.int32), j)
    assert base != cache_fingerprint(r, c, np.array([0, 2], np.int16))

    # --- ЗАПИСЬ И ОБРАТНОЕ ЧТЕНИЕ ---------------------------------------
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "cache.npz")
        save_npz(p, rows=r, codes=c, ranks=j, meta="x")
        assert os.path.exists(p), "файл не на своём имени: ловушка savez"
        back = np.load(p, allow_pickle=True)
        assert np.array_equal(back["codes"], c) and str(back["meta"]) == "x"
        back.close()
        # АРХИВИРОВАНИЕ
        dest = archive_existing(p)
        assert dest and os.path.exists(dest) and not os.path.exists(p)
        assert archive_existing(p) is None

    # --- ПРОГНОЗ --------------------------------------------------------
    f = forecast(100.0, 100, 1000)
    assert abs(f["per_batch_s"] - 1.0) < 1e-12
    assert abs(f["total_h"] - 1000 / 3600.0) < 1e-12
    assert abs(f["remaining_min"] - 900 / 60.0) < 1e-12
    try:
        forecast(1.0, 0, 10)
    except ValueError as e:
        assert "нет сделанных" in str(e), e
    else:
        raise AssertionError("принят нулевой счёт")

    print("самопроверка k15b_build_rankpath_cache пройдена")


def main():
    ap = argparse.ArgumentParser(
        description="K-15b шаг 1: кэш учительской цели rank-path")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--c1", default="data/k15b/c1_selected.pt",
                    help="артефакт probe с выбранной книгой")
    ap.add_argument("--topk", type=int, default=10,
                    help="сколько РАЗЛИЧНЫХ ближайших кодов проверять; "
                         "обязано совпадать с topk из probe")

    ap.add_argument("--out", default="data/k15b/rankpath_target_train.npz")
    ap.add_argument("--summary", default="reports/k15b/rankpath_cache.json")
    ap.add_argument("--report-every", type=int, default=500)
    ap.add_argument("--overwrite", action="store_true")
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import k15_context
    import k15b_probe_and_extract as probe
    from k15_train_depth_rvq import H_EXEC
    k15_context.add_common_arguments(ap)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    # ТОЛЬКО train, И ЭТО НЕ АРГУМЕНТ. `parts_full` в контексте собирается
    # ДО удаления подтверждающей половины, поэтому аргумент `--part`
    # позволял построить учительские цели по отложенной части. Назначение у
    # скрипта одно, поэтому выбора здесь нет.
    part = "train"
    # ЦЕЛЬ ЗАВИСИТ ОТ ВЕСОВ КАНАЛОВ. При другом grip_weight получился бы
    # другой учитель под тем же именем файла, поэтому выбора здесь нет.
    if float(a.grip_weight) != 1.0:
        raise SystemExit(
            f"--grip-weight {a.grip_weight}: вес схвата входит в метрику, по "
            f"которой выбирается ранг. Другое значение — другая цель, и её "
            f"нельзя писать под тем же именем")
    if int(a.limit) != 0:
        raise SystemExit(
            f"--limit {a.limit}: кэш обязан покрывать ВСЮ часть, иначе у "
            f"части строк не будет цели, и обучение молча пропустит их")
    for path in (a.out, a.summary):
        if os.path.exists(path) and not a.overwrite:
            raise SystemExit(f"{path} уже существует: без --overwrite не "
                             f"перезаписываю")
    archived = {p: archive_existing(p) for p in (a.out, a.summary)}
    for path, dest in archived.items():
        if dest:
            print(f"  прежний {path} перенесён в {dest}")

    ctx = k15_context.build(a)
    torch = ctx.torch
    dev, dt = ctx.dev, ctx.dt
    model, codec = ctx.model, ctx.codec
    names = list(ctx.info["names"])
    ac16 = torch.autocast(device_type=dev.type, dtype=dt)
    k15t = k15_context.k15t

    frozen_sha, n_frozen, _n_elem = k15t.frozen_content_sha(
        model, torch, set(names))
    print(f"  замороженного: {n_frozen} тензоров, отпечаток {frozen_sha}")

    # --- КНИГА: СВЕРКА, А ПРИ НЕСОВПАДЕНИИ ЗАГРУЗКА ---------------------
    if not os.path.exists(a.c1):
        raise SystemExit(f"нет {a.c1}: сначала probe")
    sel = torch.load(a.c1, map_location="cpu", weights_only=False)
    if sel.get("kind") != "k15b_c1_selected":
        raise SystemExit(f"{a.c1} описывает {sel.get('kind')!r}")
    if sel.get("accepted") is not True:
        raise SystemExit(
            f"{a.c1}: accepted = {sel.get('accepted')!r}. Книга, не прошедшая "
            f"гейты probe, учителем быть не может")
    for key, want in (("codec", ctx.codec_fp),
                      ("code_version", ctx.code_version),
                      ("frozen_content_sha", frozen_sha),
                      ("joint_sha1", ctx.joint_sha)):
        if sel.get(key) != want:
            raise SystemExit(
                f"{a.c1}: {key} = {sel.get(key)!r}, сейчас {want!r}. Книга "
                f"выбрана в другой обстановке")
    c1_file = sel["c1"]
    want_sha = sel["c1_sha1"]
    got_sha = hashlib.sha1(np.ascontiguousarray(
        c1_file.float().numpy()).tobytes()).hexdigest()[:12]
    if got_sha != want_sha:
        raise SystemExit(f"{a.c1}: книга внутри даёт {got_sha}, заявлено "
                         f"{want_sha}")
    # ПРОВЕНАНС ЧЕРНОВИКА У КНИГИ И СЕЙЧАС ОБЯЗАН СОВПАДАТЬ: цель строится
    # от q0, и книга выбиралась от того же.
    for key in ("plan_sha1", "q0_manifest_sha1"):
        want_q0 = (sel.get("q0_prov") or {}).get(key)
        if want_q0 != ctx.q0_prov.get(key):
            raise SystemExit(
                f"{a.c1}: q0_prov.{key} = {want_q0!r}, сейчас "
                f"{ctx.q0_prov.get(key)!r}. Книга выбрана от другого "
                f"черновика")
    teacher_target = sel.get("teacher_target")
    if teacher_target != "action_best_rankpath":
        raise SystemExit(
            f"{a.c1}: цель {teacher_target!r}, а этот скрипт строит кэш "
            f"только для action_best_rankpath")

    def book_sha():
        return hashlib.sha1(np.ascontiguousarray(
            model.depth_aligned_book(1).detach().float().cpu().numpy()
        ).tobytes()).hexdigest()[:12]

    if book_sha() != want_sha:
        print(f"  книга модели {book_sha()} != выбранной {want_sha}: "
              f"загружаю выбранную")
        with torch.no_grad():
            model.depth_aligned_c1.copy_(
                c1_file.to(model.depth_aligned_c1.device,
                           model.depth_aligned_c1.dtype))
        if book_sha() != want_sha:
            raise SystemExit("после загрузки книга всё ещё не та")
        again, _n, _e = k15t.frozen_content_sha(model, torch, set(names))
        if again != frozen_sha:
            raise SystemExit(
                f"загрузка книги сдвинула замороженное: {again} против "
                f"{frozen_sha}")
    else:
        print(f"  книга модели уже совпадает с выбранной ({want_sha})")
    if int(a.topk) != int(sel.get("topk", -1)):
        raise SystemExit(
            f"--topk {a.topk}, а probe выбирал цель при topk "
            f"{sel.get('topk')}: цель была бы другой")

    if part not in ctx.parts_full:
        raise SystemExit(f"в плане нет части {part!r}: "
                         f"{sorted(ctx.parts_full)}")
    batches = ctx.parts_full[part]
    rows, slot = build_row_index(batches)
    n_rows = int(rows.size)
    rows_sha = hashlib.sha1(
        np.ascontiguousarray(rows).tobytes()).hexdigest()[:12]
    print(f"  часть {part} (жёстко, без выбора): {len(batches)} батчей, "
          f"{n_rows} строк, "
          f"отпечаток {rows_sha}")

    n_pos = int(np.asarray(ctx.q0_can).shape[1])
    codes = np.full((n_rows, n_pos), -1, np.int32)
    ranks = np.full(n_rows, -1, np.int16)
    err_rank = np.full(n_rows, np.nan, np.float64)
    err_latent = np.full(n_rows, np.nan, np.float64)
    err_draft = np.full(n_rows, np.nan, np.float64)
    flips = 0

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    print(f"  строю цель: topk {a.topk}, ранг выбирается по ошибке на "
          f"первых {H_EXEC} позициях, коды пишутся для всех {n_pos}")
    model.eval()
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for po, sel_rows in batches:
            b = ctx.build_batch(po, sel_rows)
            am = b.get("attention_mask")
            with ac16:
                v, p_ids = model.build_inputs(position_offset=po, **b)
                out = model.forward_depth_aligned_rvq(
                    vlm_inputs_embeds=v, attention_mask=am,
                    position_ids=p_ids, mode="full", tau=1.0)
            q0_now = out["pred_codes"][0].detach().cpu().numpy()
            bad = int((q0_now != ctx.q0_can[sel_rows]).sum())
            if bad:
                raise SystemExit(
                    f"q0 разошёлся с каноническим в {bad} позициях: цель "
                    f"строилась бы от другого черновика")
            z0 = out["policy_embeddings"][0]
            c1 = model.depth_aligned_book(1)
            action = torch.from_numpy(
                np.asarray(ctx.ACT[sel_rows], np.float32)).to(dev)[..., :7]
            z_e = codec._encode(action.float(), embodiment_ids=0).float()
            d1 = ctx.tok.mean_squared_distances(z_e - z0.detach(), c1)
            _lg, _emb, i1, _p = ctx.tok.quantize_residual(
                z_e - z0.detach(), c1, temperature=1.0)
            near = probe.rank_candidates(d1, i1, int(a.topk), torch)
            flips += int(((-d1).topk(int(a.topk), dim=-1).indices[..., 0]
                          != i1).sum())
            errs = []
            for j in range(near.shape[-1]):
                rows_j, _m = k15t.weighted_row_error(
                    ctx.decode_fp32(z0 + c1[near[..., j]]), action,
                    ctx.weights_gate, torch)
                errs.append(rows_j)
            stack = torch.stack(errs, 0)
            best_j = stack.argmin(0)
            idx = best_j.view(-1, 1, 1).expand(-1, near.shape[1], 1)
            best_code = near.gather(-1, idx).squeeze(-1)
            rows_draft, _m = k15t.weighted_row_error(
                ctx.decode_fp32(z0), action, ctx.weights_gate, torch)

            bj = best_j.cpu().numpy()
            bc = best_code.cpu().numpy().astype(np.int32)
            e_best = stack.min(0).values.cpu().numpy()
            e_lat = stack[0].cpu().numpy()
            e_dr = rows_draft.cpu().numpy()
            for k_, r_ in enumerate(np.asarray(sel_rows, np.int64).tolist()):
                s = slot[r_]
                codes[s] = bc[k_]
                ranks[s] = bj[k_]
                err_rank[s] = e_best[k_]
                err_latent[s] = e_lat[k_]
                err_draft[s] = e_dr[k_]
            done += 1
            if done % int(a.report_every) == 0 or done == len(batches):
                sync()
                f = forecast(time.time() - t0, done, len(batches))
                print(f"    {done}/{len(batches)}: {f['per_batch_s']:.2f} "
                      f"с/батч, всего {f['total_h']:.1f} ч, осталось "
                      f"{f['remaining_min']:.0f} мин", flush=True)
    elapsed = time.time() - t0

    # --- ПОЛНОТА И СОГЛАСОВАННОСТЬ --------------------------------------
    if int((codes < 0).sum()) or int((ranks < 0).sum()):
        raise SystemExit(
            f"не заполнено: кодов {int((codes < 0).sum())}, ранга "
            f"{int((ranks < 0).sum())}. Часть строк осталась без цели")
    for nm, arr in (("err_rankpath", err_rank), ("err_latent", err_latent),
                    ("err_draft", err_draft)):
        if not np.isfinite(arr).all():
            raise SystemExit(
                f"{nm}: {int((~np.isfinite(arr)).sum())} нечисловых значений")
    vocab = int(ctx.vocab)
    if int((codes >= vocab).sum()):
        raise SystemExit(f"есть коды вне словаря {vocab}")
    if int((ranks >= int(a.topk)).sum()):
        raise SystemExit(f"есть ранги вне [0, {a.topk})")
    # РАНГ 0 ОБЯЗАН СОВПАДАТЬ С КОДОМ ТОКЕНИЗАТОРА ПО ОШИБКЕ
    worse = int((err_rank > err_latent + 1e-12).sum())
    if worse:
        raise SystemExit(
            f"у {worse} строк лучший ранг хуже ранга 0: выбор минимума "
            f"собран неверно")

    rms = {nm: float(np.sqrt(v.mean())) for nm, v in
           (("draft", err_draft), ("latent", err_latent),
            ("rankpath", err_rank))}
    gap = rms["draft"] - rms["rankpath"]
    content_sha = cache_fingerprint(rows, codes, ranks)
    hist = {int(k): int(c) for k, c in zip(*np.unique(ranks,
                                                      return_counts=True))}
    print(f"\n  на части {part}: черновик {rms['draft']:.6f}, латентная "
          f"цель {rms['latent']:.6f}, rank-path {rms['rankpath']:.6f} "
          f"(−{100 * (1 - rms['rankpath'] / rms['draft']):.1f} % от "
          f"черновика)")
    print(f"  выигрыш rank-path над латентной целью "
          f"{100 * (rms['latent'] - rms['rankpath']) / rms['latent']:.1f} %; "
          f"ранг не нулевой у {100 * float((ranks != 0).mean()):.1f} % строк")
    print(f"  гистограмма рангов: {hist}")
    print(f"  время {elapsed / 3600:.2f} ч, {elapsed / len(batches):.2f} "
          f"с/батч; отпечаток содержимого {content_sha}")

    meta = dict(
        kind="k15b_rankpath_target", part=part, topk=int(a.topk),
        rows=n_rows, positions=n_pos, batches=len(batches),
        rows_sha1=rows_sha, content_sha1=content_sha,
        action_error_positions=int(H_EXEC),
        code_target_positions=int(n_pos),
        ce_target_positions="all code positions",
        positions_note=(
            f"ДВЕ РАЗНЫЕ ОСИ. Ранг выбран по взвешенной ошибке действия на "
            f"первых {H_EXEC} ВРЕМЕННЫХ ШАГАХ — только они входят в "
            f"метрику. Коды записаны для всех {n_pos} КОДОВЫХ позиций, и "
            f"кросс-энтропию читателя надо брать по ВСЕМ {n_pos}: "
            f"PerceiverDecoder смотрит на все латентные токены через "
            f"cross-attention без каузальной маски, поэтому коды поздних "
            f"позиций влияют и на первые шаги действия, а малая ошибка "
            f"траектории получена совместным действием всех кодов"),
        teacher_metric=dict(
            action_error_positions=int(H_EXEC),
            channel_weights=[float(x) for x in
                             ctx.weights_gate.detach().cpu().numpy()],
            grip_weight=float(a.grip_weight),
            decoder_context=dict(ctx.decoder_context),
            # ЧЕРЕЗ getattr ОТ САМОГО codec: у другого квантователя
            # атрибута `config` может не быть, и падать на провенансе в
            # конце двухчасового прохода нельзя.
            decoder_add_causal_mask=bool(getattr(
                getattr(codec, "config", None),
                "decoder_add_causal_mask", False)),
            decoder_n_layers=int(getattr(
                getattr(codec, "config", None), "decoder_n_layers", -1)),
            note="всё, что определяет, какой ранг окажется лучшим"),
        builder_sha1=k15_context.sha12(os.path.abspath(__file__)),
        rank_candidates_file=os.path.basename(
            inspect.getfile(probe.rank_candidates)),
        rank_candidates_file_sha1=k15_context.sha12(
            inspect.getfile(probe.rank_candidates)),
        contract_for_trainer=(
            "тренер обязан сверить c1_sha1, topk, action_error_positions, "
            "teacher_metric.channel_weights, teacher_metric.decoder_context, "
            "rank_candidates_file_sha1, q0_prov и plan_sha1. Свой "
            "builder_sha1 сверять не нужно: правка печати в построителе "
            "цель не меняет"),
        histogram_note=(
            "гистограмма рангов показывает, как часто action-цель отличается "
            "от латентного argmin и хватает ли topk. Она НЕ показывает, "
            "насколько трудно обучить читателя: это покажут held-out "
            "top-1/recall, action RMS и capture на val_sel"),
        not_a_full_oracle=(
            "§50.2 не закрыт: смеси рангов по позициям (k^T вариантов) не "
            "перебираются, коды вне латентного соседства не "
            "рассматриваются"),
        rms_draft=rms["draft"], rms_latent=rms["latent"],
        rms_rankpath=rms["rankpath"],
        teacher_gap=float(gap),
        relative_gain_over_latent=float(
            (rms["latent"] - rms["rankpath"]) / max(rms["latent"], 1e-12)),
        rank_histogram=hist,
        share_rank_nonzero=float((ranks != 0).mean()),
        rank1_flips_vs_tokenizer_argmin=int(flips),
        c1_file=os.path.abspath(a.c1), c1_sha1=want_sha,
        c1_source_state=sel.get("source_state"),
        c1_source_epoch=sel.get("source_epoch"),
        teacher_target=teacher_target,
        seconds=float(elapsed), seconds_per_batch=float(elapsed / len(batches)),
        frozen_content_sha=frozen_sha,
        codec=ctx.codec_fp, code_version=ctx.code_version,
        q0_prov=ctx.q0_prov, joint_sha1=ctx.joint_sha,
        plan_sha1=ctx.q0_prov.get("plan_sha1"),
        git_head=ctx.git_head, git_dirty=bool(ctx.dirty),
        **ctx.gate_info)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    save_npz(a.out, rows=rows, codes=codes, ranks=ranks,
             err_rankpath=err_rank, err_latent=err_latent,
             err_draft=err_draft,
             meta=json.dumps(meta, ensure_ascii=False, allow_nan=False,
                             default=k15t.json_scalar))
    print(f"  кэш сохранён: {a.out} ({content_sha}), обратное чтение сошлось")

    os.makedirs(os.path.dirname(os.path.abspath(a.summary)) or ".",
                exist_ok=True)
    tmp = a.summary + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(dict(meta, cache_file=os.path.abspath(a.out),
                       archived={k: v for k, v in archived.items() if v}),
                  fh, ensure_ascii=False, indent=1, allow_nan=False,
                  default=k15t.json_scalar)
    os.replace(tmp, a.summary)
    print(f"  сводка: {a.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
