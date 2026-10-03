#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15c: обучение голов выбора ранга по готовому кэшу. Без модели VLA.

ЧТО ОБУЧАЕТСЯ. Голова получает состояние строки и выбирает один из восьми
согласованных путей; исполняется argmax. Признаки фиксированы (C1, читатель
q1 и backbone заморожены), поэтому прогон занимает минуты, а не часы.

ПЕРВАЯ ОЧЕРЕДЬ, seed 0, одинаковые оптимизатор, батч, эпохи и остановка:

    h18_linear     — для сравнения глубин
    h24_linear     — главный sanity baseline
    h24_candidate  — предлагаемая архитектура

Если ни одна не взяла порог, следующая дешёвая голова на том же кэше —
`--heads h24_positional`: совместимость h24[t] с кодом кандидата в позиции
t по каждой паре (t, j), а не усреднённого с усреднённым. Пулинг вниманием
(`h24_candidate_attn`) этого сопоставления не даёт и потому не основной
следующий вариант.

ФУНКЦИЯ ПОТЕРЬ — ОЖИДАЕМЫЙ REGRET ПО ИСХОДНЫМ ЗАТРАТАМ:

    L = mean_i sum_j softmax(s_i)_j * (e_ij - min_l e_il),

без построчной нормировки (она изменила бы веса строк относительно
глобального action RMS). CE на первом проходе не используется.

ОТБОР И КРИТЕРИЙ. Эпоха выбирается ТОЛЬКО по минимальному hard RMS на
val_sel: argmax оценок, его затрата, корень из среднего. Не по CE, не по
точности, не по мягкому regret и не по потере на train. Научный критерий
зафиксирован: доля разрыва на val_sel >= 0.20, причём голова обязана быть
лучше ранга 0 и лучше лучшего фиксированного ранга, а выбранное состояние —
воспроизводиться после сохранения и загрузки. Точность выбора лучшего ранга
приводится только как диагностика и по ней ничего не решается; переводить
её в долю разрыва аналитической формулой нельзя.

ПРОБА КОНВЕЙЕРА ПЕРЕД ПОЛНЫМ ОБУЧЕНИЕМ, на 64 фиксированных строках train.
Обязательно: всё конечно, градиент ненулевой, параметры изменились, снимок
восстанавливается побитово, regret заметно упал. Монотонность и улучшение
hard argmax — только диагностика: неумение запомнить неоднозначные строки
было бы научным результатом, а не поломкой. Отказ пробы снимает ТОЛЬКО эту
голову.

ДВА СТАТУСА ПОРОГА. Основной — зарегистрированные 0.20. Разведочный —
доля разрыва мягкого пути M2 плюс 0.05 (около 11.9 %): он не доказывает
архитектурный успех, а лишь оправдывает проверку вывода и короткий
разведочный роллаут. Остальные условия у обоих одинаковы.

РЕШАЮТ ТОЛЬКО h24-ГОЛОВЫ. h18_linear — абляция глубины: её исход
печатается и пишется, но ни успехом, ни отказом запуска не является, и
технический отказ одной головы не отменяет успех другой.

КОДЫ: 0 — h24-голова взяла основной критерий; 6 — только разведочный
(2 занят argparse, 5 в K-15 означал сбой схемы сводки); 4 — порогов нет;
3 — ни одной технически оценённой h24-головы; провенанс кэша — отказ с
текстом. Выбор делается на val_sel и помечается так; val_confirm не
открывается.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

CAPTURE_MIN = 0.20
# РАЗВЕДОЧНЫЙ ПОРОГ — не научный критерий. Число = доля разрыва мягкого
# пути M2 плюс 0.05: это порог сетки (k, tau) из M2, объявленный до данных
# 03.10.2026. Его ПРИМЕНЕНИЕ к головам K-15c зарегистрировано до их
# обучения. Он не доказывает архитектурный успех, а лишь оправдывает
# проверку вывода и короткий разведочный роллаут.
PILOT_GAIN = 0.05
OVERFIT_DROP = 0.67
OVERFIT_CHECK_EVERY = 50
DEFAULT_HEADS = ("h18_linear", "h24_linear", "h24_candidate")
# Развёртываемая архитектура читает h24. h18-голова — абляция глубины: её
# исход печатается, но ни успехом, ни отказом всего запуска не является.
DEPLOYABLE_PREFIX = "h24_"
# Коды исхода. 2 занят argparse (ошибка аргументов), 5 в K-15 означал сбой
# схемы сводки, поэтому разведочный исход — 6.
CODE_PRIMARY, CODE_PILOT, CODE_NONE, CODE_TECH = 0, 6, 4, 3


def rms(costs):
    return float(np.sqrt(np.mean(np.asarray(costs, np.float64))))


def capture(rms_draft, rms_teacher, rms_policy):
    """Доля разрыва; None при неположительном разрыве. Как в K-15b."""
    gap = float(rms_draft) - float(rms_teacher)
    if gap <= 1e-12:
        return None
    return float((float(rms_draft) - float(rms_policy)) / gap)


M2_KIND = "k15b_soft_operating_point"
M2_REL = 1e-6


def sorted_rows_sha(rows):
    """Отпечаток МНОЖЕСТВА строк в порядке M2: np.unique по int64.

    Кэш хранит строки в порядке плана, а M2 хешировал отсортированные, поэтому
    сравнивать их отпечатки напрямую нельзя — только после приведения.
    """
    u = np.unique(np.asarray(rows, np.int64))
    return hashlib.sha1(np.ascontiguousarray(u).tobytes()).hexdigest()[:12]


def check_m2(m2, man, rows_val, smoke=False):
    """M2 против кэша. Возвращает (разведочный порог, проблемы). Fail-closed.

    Разведочный порог берётся из M2, поэтому подменённый или устаревший M2
    мог бы перевести исход 4 <-> 6. Проверяется: вид артефакта, отбор на
    ПОЛНОЙ val_sel, та же книга, тот же читатель, тот же кодек и код, те же
    строки, те же черновик и учитель, что посчитаны в кэше, и внутренняя
    арифметика самой доли мягкого пути. На smoke-кэше часть val_sel урезана,
    поэтому сверка строк и агрегатов там пропускается, а провенанс — нет.
    """
    problems = []

    def need(cond, msg):
        if not cond:
            problems.append(msg)

    need(m2.get("kind") == M2_KIND,
         f"kind {m2.get('kind')!r}, ожидался {M2_KIND!r}")
    need(m2.get("selected_on") == "val_sel",
         f"selected_on {m2.get('selected_on')!r}: нужна полная val_sel")
    teacher = m2.get("teacher") or {}
    need(teacher.get("full_val_sel") is True, "M2 снят не на всей val_sel")
    need(m2.get("c1_sha1") == man.get("c1_sha1"), "M2 снят с другой книгой")
    need((m2.get("reader") or {}).get("point_state_sha1")
         == man.get("reader_state_sha1"), "M2 снят с другим читателем")
    for key in ("codec", "code_version", "joint_sha1"):
        need(m2.get(key) == man.get(key), f"{key} M2 не совпал с кэшем")
    draft, rank = teacher.get("draft"), teacher.get("rank")
    a1_soft = (m2.get("rms") or {}).get("a1_soft")
    cap_soft = m2.get("capture_soft")
    nums = (draft, rank, a1_soft, cap_soft)
    if any(x is None for x in nums) \
            or not all(np.isfinite(float(x)) for x in nums):
        problems.append("в M2 нет черновика, учителя, мягкого пути или его "
                        "доли разрыва")
        return None, problems
    want = capture(draft, rank, a1_soft)
    need(want is not None and abs(float(cap_soft) - want) <= 1e-9,
         f"capture_soft {cap_soft!r} не равна capture(черновик, учитель, "
         f"мягкий) = {want!r}")
    if not smoke:
        vs = (man.get("parts") or {}).get("val_sel") or {}
        need(int(m2.get("rows", -1)) == int(vs.get("rows", -2)),
             f"строк в M2 {m2.get('rows')}, в кэше {vs.get('rows')}")
        need(int(m2.get("batches", -1)) == int(vs.get("batches", -2)),
             f"батчей в M2 {m2.get('batches')}, в кэше {vs.get('batches')}")
        need(m2.get("rows_sha1") == sorted_rows_sha(rows_val),
             "множество строк val_sel в M2 не то, что в кэше")
        agg = vs.get("aggregates") or {}
        for key, val in (("draft", draft), ("teacher", rank)):
            ref = agg.get(key)
            rel = (abs(float(val) - float(ref)) / max(abs(float(ref)), 1e-12)
                   if ref is not None else float("inf"))
            need(rel <= M2_REL, f"{key} M2 {val!r} против кэша {ref!r}")
        ref = ((man.get("m2_check") or {}).get("reference") or {})
        need(ref.get("draft") == draft and ref.get("teacher") == rank,
             "M2 не тот, с которым сверялся построитель кэша")
    if problems:
        return None, problems
    return float(cap_soft) + PILOT_GAIN, []


def probe_rows(n_total, n_take):
    """Строки пробы — РАВНОМЕРНО по всей части, а не первые подряд.

    Первые строки плана — соседние кадры одних эпизодов: признаки у них
    почти совпадают, а лучший ранг от кадра к кадру скачет. Такие
    противоречивые почти-дубликаты не разделит никакая голова, и regret
    упирается в плато — на smoke-кэше это выглядело как технический отказ
    всех голов. Проба проверяет КОНВЕЙЕР, и эта помеха ей не нужна.
    """
    n_total, n_take = int(n_total), int(n_take)
    if n_total <= 0 or n_take <= 0:
        raise ValueError("нет строк для пробы")
    if n_take >= n_total:
        return np.arange(n_total)
    return np.unique(np.linspace(0, n_total - 1, n_take).round()
                     .astype(np.int64))


def archive_working(path):
    """Прежний чекпойнт с РАБОЧИМ именем — в архив. Возвращает новый путь.

    Вызывается при ЛЮБОМ техническом отказе головы: пробы, исключения при
    обучении, невоспроизводимости. Иначе файл прошлого прогона лежал бы под
    рабочим именем и выглядел бы действующим, хотя в этом прогоне голова
    отказала.
    """
    if not os.path.exists(path):
        return None
    dest = f"{path}.stale.{time.strftime('%Y%m%dT%H%M%S')}.bak"
    if os.path.exists(dest):
        dest = f"{dest}.{os.getpid()}"
    os.replace(path, dest)
    return dest


def evaluate_scores(scores, costs, draft, teacher, task_ids=None,
                    task_vocab=None):
    """Все метрики исполняемого выбора по оценкам [N, 8]. Чистая функция."""
    s = np.asarray(scores, np.float64)
    c = np.asarray(costs, np.float64)
    if s.shape != c.shape or s.ndim != 2:
        raise ValueError(f"формы {s.shape} и {c.shape}")
    if not np.isfinite(s).all():
        raise ValueError("оценки не конечны")
    pick = s.argmax(1)
    sel = c[np.arange(len(c)), pick]
    best = c.min(1)
    regret = sel - best
    z = s - s.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    soft_regret = float((p * (c - best[:, None])).sum(1).mean())
    r_d, r_t = rms(draft), rms(teacher)
    out = dict(
        rms=rms(sel), capture=capture(r_d, r_t, rms(sel)),
        rms_draft=r_d, rms_teacher=r_t,
        regret_mean=float(regret.mean()),
        regret_median=float(np.median(regret)),
        regret_p95=float(np.percentile(regret, 95)),
        soft_expected_regret=soft_regret,
        best_rank_accuracy=float((pick == c.argmin(1)).mean()),
        share_rank0=float((pick == 0).mean()),
        histogram={int(k): int(v) for k, v in
                   enumerate(np.bincount(pick, minlength=c.shape[1]))},
        rows=int(len(c)))
    if task_ids is not None:
        per = {}
        t = np.asarray(task_ids)
        for tid in np.unique(t):
            m = t == tid
            name = (task_vocab[int(tid)] if task_vocab is not None
                    else str(int(tid)))
            rd, rt, rs_ = rms(np.asarray(draft)[m]), \
                rms(np.asarray(teacher)[m]), rms(sel[m])
            per[name] = dict(rows=int(m.sum()), rms=rs_,
                             capture=capture(rd, rt, rs_))
        out["per_task"] = per
    return out


def baselines(costs_val, draft, teacher, costs_train):
    """Опорные числа из того же кэша, без какой-либо модели."""
    c = np.asarray(costs_val, np.float64)
    r_d, r_t = rms(draft), rms(teacher)
    out = {}
    for j in range(c.shape[1]):
        out[f"fixed_rank{j}"] = dict(rms=rms(c[:, j]),
                                     capture=capture(r_d, r_t, rms(c[:, j])))
    best_fixed = min(range(c.shape[1]), key=lambda j: rms(c[:, j]))
    maj = int(np.bincount(np.asarray(costs_train).argmin(1),
                          minlength=c.shape[1]).argmax())
    out["majority_rank"] = dict(rank=maj, rms=rms(c[:, maj]),
                                capture=capture(r_d, r_t, rms(c[:, maj])),
                                note="majority лучшего ранга на TRAIN")
    out["uniform_random"] = dict(
        rms=rms(c.mean(1)), capture=capture(r_d, r_t, rms(c.mean(1))),
        note="ожидаемая MSE равномерного выбора, без выборки")
    out["oracle_top8"] = dict(rms=rms(c.min(1)),
                              capture=capture(r_d, r_t, rms(c.min(1))))
    out["hard_q1"] = dict(out["fixed_rank0"])
    out["best_fixed_rank"] = dict(rank=int(best_fixed),
                                  **out[f"fixed_rank{best_fixed}"])
    return out


def probe_verdict(trace, hard_before, hard_after, changed, restored_equal,
                  finite=True, grad_nonzero=True, drop=OVERFIT_DROP):
    """Проба конвейера. ТЕХНИЧЕСКАЯ, а не мера способности модели.

    Обязательно: всё конечно, градиент ненулевой, параметры изменились,
    снимок восстанавливается побитово, regret заметно упал. Монотонность и
    улучшение hard argmax — диагностика: гладкий regret может падать при
    неизменном argmax, и неумение головы запомнить неоднозначные строки —
    научный результат, а не поломка.
    """
    first, last = float(trace[0]), float(trace[-1])
    monotone = all(float(b) <= float(a) * (1.0 + 1e-3) + 1e-12
                   for a, b in zip(trace, trace[1:]))
    checks = dict(
        finite=bool(finite) and all(np.isfinite(trace)),
        grad_nonzero=bool(grad_nonzero),
        params_changed=bool(changed),
        restore_exact=bool(restored_equal),
        regret_dropped=bool(last <= drop * first + 1e-12))
    diagnostics = dict(regret_monotone=bool(monotone),
                       hard_improved=bool(hard_after < hard_before))
    return dict(passed=all(checks.values()), checks=checks,
                diagnostics=diagnostics,
                first=first, last=last, ratio=float(last / max(first, 1e-12)),
                hard_before=float(hard_before), hard_after=float(hard_after),
                trace=[float(x) for x in trace], required_drop=float(drop))


def select_epoch(history):
    """Минимум hard RMS на val_sel; ничья — более ранняя эпоха."""
    if not history:
        raise ValueError("история пуста")
    return min(history, key=lambda h: (float(h["val"]["rms"]),
                                       int(h["epoch"])))


def head_verdict(val, base, finite, reproducible, cap_min=CAPTURE_MIN,
                 pilot_min=None):
    """Критерий головы: основной и разведочный. Точность не участвует.

    Общие условия у обоих: лучше ранга 0, лучше лучшего фиксированного
    ранга, всё конечно, выбранное состояние воспроизводится. Различаются
    только порогом доли разрыва.
    """
    cap = val.get("capture")
    common = dict(
        better_than_rank0=bool(val["rms"] < base["fixed_rank0"]["rms"]),
        better_than_best_fixed=bool(
            val["rms"] < base["best_fixed_rank"]["rms"]),
        finite=bool(finite), reproducible=bool(reproducible))
    primary_cap = bool(cap is not None and float(cap) >= cap_min - 1e-12)
    pilot_cap = (None if pilot_min is None else
                 bool(cap is not None and float(cap) >= pilot_min - 1e-12))
    primary = primary_cap and all(common.values())
    pilot = (None if pilot_cap is None else
             bool(pilot_cap and all(common.values())))
    return dict(passed=bool(primary), primary_passed=bool(primary),
                pilot_passed=pilot,
                checks=dict(common, capture_primary=primary_cap,
                            capture_pilot=pilot_cap),
                capture_min=float(cap_min),
                pilot_min=(None if pilot_min is None else float(pilot_min)))


def overall_code(results):
    """Общий исход по головам. Решают ТОЛЬКО развёртываемые h24-головы.

    0 — хотя бы одна h24-голова взяла основной критерий;
    6 — основной не взят, но хотя бы одна взяла разведочный;
    4 — хотя бы одна h24-голова технически оценена, порогов нет;
    3 — ни одной технически оценённой h24-головы.
    Технический отказ одной головы не отменяет успех другой, а h18 —
    абляция и на код не влияет вовсе.
    """
    dep = {k: v for k, v in results.items()
           if k.startswith(DEPLOYABLE_PREFIX)}
    ok = {k: v for k, v in dep.items()
          if v.get("trained") and not v.get("technical")}
    if not ok:
        return CODE_TECH
    if any(v["verdict"]["primary_passed"] for v in ok.values()):
        return CODE_PRIMARY
    if any(v["verdict"].get("pilot_passed") for v in ok.values()):
        return CODE_PILOT
    return CODE_NONE


class Inputs:
    """Входы головы по индексам строк. Держит тензоры на устройстве."""

    def __init__(self, torch, ctx, cand_emb, cand_feat, h_full=None,
                 device="cpu", cand_codes=None):
        self.torch = torch
        self.ctx = None if ctx is None else ctx.to(device)
        self.cand_emb = cand_emb.to(device)
        self.cand_feat = cand_feat.to(device)
        self.cand_codes = (None if cand_codes is None
                           else cand_codes.to(device))
        self.h_full = h_full           # memmap или тензор на CPU
        self.device = device

    def batch(self, idx):
        t = self.torch
        i = t.as_tensor(np.asarray(idx), device=self.device)
        kw = dict(ctx=None if self.ctx is None else self.ctx[i],
                  cand_emb=self.cand_emb[i], cand_feat=self.cand_feat[i])
        if self.cand_codes is not None:
            kw["cand_codes"] = self.cand_codes[i]
        if self.h_full is not None:
            srt = np.sort(np.asarray(idx))
            back = np.argsort(np.argsort(np.asarray(idx)))
            hf = t.from_numpy(np.asarray(self.h_full[srt],
                                         np.float32))[back]
            kw["h_full"] = hf.to(self.device)
        return kw


def scores_for(head, inp, idx_all, torch, chunk=4096):
    out = []
    head.eval()
    with torch.no_grad():
        for s in range(0, len(idx_all), chunk):
            out.append(head(**inp.batch(idx_all[s:s + chunk])).float()
                       .cpu().numpy())
    return np.concatenate(out, 0)


def train_head(head, inp_tr, costs_tr, inp_va, costs_va, draft_va,
               teacher_va, torch, *, epochs, batch, lr, wd, patience, seed,
               log=print, task_ids=None, task_vocab=None):
    """Полное обучение по regret, оценка каждой эпохи, все состояния."""
    import k15c_rank_selector as rs
    dev = inp_tr.device
    c_tr = torch.as_tensor(np.asarray(costs_tr, np.float32), device=dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n = len(costs_tr)
    idx_va = np.arange(len(costs_va))
    history, states = [], {}

    def evaluate(epoch, train_loss):
        sc = scores_for(head, inp_va, idx_va, torch)
        val = evaluate_scores(sc, costs_va, draft_va, teacher_va,
                              task_ids=task_ids, task_vocab=task_vocab)
        history.append(dict(epoch=int(epoch), train_regret=train_loss,
                            val=val))
        states[int(epoch)] = {k: v.detach().cpu().clone()
                              for k, v in head.state_dict().items()}
        log(f"      эпоха {epoch:3d}: train regret "
            + ("—" if train_loss is None else f"{train_loss:.4e}")
            + f"; val hard RMS {val['rms']:.6f}, доля "
            f"{100 * (val['capture'] or 0):.1f} %, ранг 0 у "
            f"{100 * val['share_rank0']:.1f} % строк")
        return val

    evaluate(0, None)
    best_rms, since = history[-1]["val"]["rms"], 0
    for epoch in range(1, int(epochs) + 1):
        head.train()
        rng = np.random.default_rng(int(seed) * 1000 + epoch)
        perm = rng.permutation(n)
        tot, cnt = 0.0, 0
        for s in range(0, n, int(batch)):
            ii = perm[s:s + int(batch)]
            sc = head(**inp_tr.batch(ii))
            loss = rs.expected_regret(sc, c_tr[torch.as_tensor(
                ii, device=dev)], torch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"эпоха {epoch}: regret не конечен")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(ii)
            cnt += len(ii)
        val = evaluate(epoch, tot / max(cnt, 1))
        if val["rms"] < best_rms - 1e-12:
            best_rms, since = val["rms"], 0
        else:
            since += 1
            if since >= int(patience):
                log(f"      остановка: {patience} эпох без улучшения hard "
                    f"RMS на val_sel")
                break
    return history, states


def overfit_probe(head, inp, idx, costs, torch, steps, lr, wd,
                  check_every=OVERFIT_CHECK_EVERY, drop=OVERFIT_DROP):
    """Проба конвейера на фиксированных строках, с восстановлением.

    REGRET ИЗМЕРЯЕТСЯ ТОЧНО, А НЕ СРЕДНИМ ПО ОКНУ: до первого шага и в
    контрольных точках, на том же батче, без градиента. Прежняя версия
    сравнивала средние по окнам, и первое окно уже содержало большую часть
    падения — на smoke-кэше это дало отношение 0.71-0.84 при hard-затрате,
    упавшей вдвое, то есть ложный технический отказ всех голов.

    Проба ОСТАНАВЛИВАЕТСЯ, как только regret опустился до `drop` от
    начального; `steps` — бюджет, а не обязательная длина. Argmax
    становится верным раньше, чем softmax сосредотачивается, поэтому
    фиксированная короткая проба меряла скорость сжатия распределения, а не
    исправность конвейера.
    """
    import k15c_rank_selector as rs
    dev = inp.device
    snap = {k: v.detach().clone() for k, v in head.state_dict().items()}
    c = torch.as_tensor(np.asarray(costs[idx], np.float32), device=dev)
    kw = inp.batch(idx)

    def exact():
        head.eval()
        with torch.no_grad():
            sc = head(**kw)
            return (float(rs.expected_regret(sc, c, torch)), sc.detach())

    r0, s0 = exact()
    s0 = s0.clone()
    hard_before = float(rs.hard_selected_cost(s0, c, torch)[0].mean())
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    trace = [r0]
    finite, grad_nonzero, used = bool(np.isfinite(r0)), False, 0
    for st in range(1, int(steps) + 1):
        head.train()
        loss = rs.expected_regret(head(**kw), c, torch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        finite = finite and bool(torch.isfinite(loss))
        if st == 1:
            gn = sum(float(p.grad.norm()) ** 2 for p in head.parameters()
                     if p.grad is not None) ** 0.5
            finite = finite and bool(np.isfinite(gn))
            grad_nonzero = bool(gn > 0.0)
        opt.step()
        used = st
        if st % int(check_every) == 0 or st == int(steps):
            r, _ = exact()
            trace.append(r)
            finite = finite and bool(np.isfinite(r))
            if r <= drop * r0 + 1e-12 or not finite:
                break
    _r, s1 = exact()
    hard_after = float(rs.hard_selected_cost(s1, c, torch)[0].mean())
    changed = any(not torch.equal(v, snap[k])
                  for k, v in head.state_dict().items())
    head.load_state_dict(snap)
    _r2, s2 = exact()
    out = probe_verdict(trace, hard_before, hard_after, changed,
                        bool(torch.equal(s2, s0)), finite=finite,
                        grad_nonzero=grad_nonzero, drop=drop)
    out.update(steps_used=int(used), steps_budget=int(steps),
               lr=float(lr), measured="exact, before training and at "
                                      "checkpoints, same batch, no grad")
    return out


def selftest():
    import torch
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import k15c_rank_selector as rs
    import k15b_train_stagewise as stagewise

    # --- ДОЛЯ РАЗРЫВА СОВПАДАЕТ С K-15b ---------------------------------
    for v in (0.142997, 0.111093, 0.130954):
        assert abs(capture(0.145469, 0.072892, v)
                   - stagewise.capture(0.145469, 0.072892, v)) < 1e-12
    assert capture(0.1, 0.1, 0.05) is None
    assert CAPTURE_MIN == stagewise.CAPTURE_THRESHOLD

    # --- МЕТРИКИ ИСПОЛНЯЕМОГО ВЫБОРА ------------------------------------
    costs = np.array([[1.0, 0.5, 2.0, 3, 3, 3, 3, 3],
                      [0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                      [0.4, 0.4, 0.1, 0.4, 0.4, 0.4, 0.4, 0.4]])
    draft = np.array([1.2, 1.0, 0.6])
    teacher = np.array([0.1, 0.1, 0.05])
    sc = np.zeros((3, 8))
    sc[0, 1] = sc[1, 0] = sc[2, 2] = 5.0          # оракульный выбор
    ev = evaluate_scores(sc, costs, draft, teacher, task_ids=[0, 1, 1],
                         task_vocab=["a", "b"])
    assert abs(ev["rms"] - np.sqrt((0.5 + 0.2 + 0.1) / 3)) < 1e-12
    assert ev["regret_mean"] == 0.0 and ev["best_rank_accuracy"] == 1.0
    assert ev["histogram"] == {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0,
                               7: 0}
    assert abs(ev["share_rank0"] - 1 / 3) < 1e-12
    assert set(ev["per_task"]) == {"a", "b"}
    assert ev["per_task"]["b"]["rows"] == 2
    ev0 = evaluate_scores(np.zeros((3, 8)), costs, draft, teacher)
    assert ev0["share_rank0"] == 1.0          # ничья в argmax -> ранг 0
    assert abs(ev0["rms"] - np.sqrt((1.0 + 0.2 + 0.4) / 3)) < 1e-12
    assert ev0["soft_expected_regret"] > 0
    for bad in (np.zeros((3, 7)), np.full((3, 8), np.nan)):
        try:
            evaluate_scores(bad, costs, draft, teacher)
        except ValueError:
            pass
        else:
            raise AssertionError("приняты негодные оценки")

    # --- ОПОРНЫЕ ЧИСЛА ---------------------------------------------------
    b = baselines(costs, draft, teacher, costs)
    assert abs(b["oracle_top8"]["rms"]
               - np.sqrt((0.5 + 0.2 + 0.1) / 3)) < 1e-12
    assert b["hard_q1"]["rms"] == b["fixed_rank0"]["rms"]
    assert b["best_fixed_rank"]["rank"] in range(8)
    assert abs(b["uniform_random"]["rms"]
               - np.sqrt(costs.mean(1).mean())) < 1e-12
    assert b["majority_rank"]["rank"] in (0, 1, 2)

    # --- ПРОБА: ОБЯЗАТЕЛЬНОЕ И ДИАГНОСТИКА ------------------------------
    pv = probe_verdict([1.0, 0.8, 0.6, 0.5, 0.4, 0.3], 0.9, 0.5, True, True)
    assert pv["passed"], pv
    # НЕМОНОТОННОСТЬ И НЕУЛУЧШЕННЫЙ argmax — ТОЛЬКО ДИАГНОСТИКА
    nm = probe_verdict([1.0, 0.9, 0.95, 0.5], 0.9, 0.5, True, True)
    assert nm["passed"] and not nm["diagnostics"]["regret_monotone"], nm
    nh = probe_verdict([1.0, 0.5], 0.9, 0.9, True, True)
    assert nh["passed"] and not nh["diagnostics"]["hard_improved"], nh
    # ОБЯЗАТЕЛЬНЫЕ — каждое по отдельности валит пробу
    assert not probe_verdict([1.0, 0.9], 0.9, 0.5, True, True)["passed"]
    assert not probe_verdict([1.0, 0.5], 0.9, 0.5, False, True)["passed"]
    assert not probe_verdict([1.0, 0.5], 0.9, 0.5, True, False)["passed"]
    assert not probe_verdict([1.0, 0.5], 0.9, 0.5, True, True,
                             finite=False)["passed"]
    assert not probe_verdict([1.0, 0.5], 0.9, 0.5, True, True,
                             grad_nonzero=False)["passed"]
    assert not probe_verdict([1.0, float("nan")], 0.9, 0.5, True,
                             True)["passed"]

    # --- ВЫБОР ЭПОХИ ----------------------------------------------------
    hist = [dict(epoch=0, val=dict(rms=0.14)),
            dict(epoch=1, val=dict(rms=0.12)),
            dict(epoch=2, val=dict(rms=0.12))]
    assert select_epoch(hist)["epoch"] == 1

    # --- КРИТЕРИЙ: ОСНОВНОЙ И РАЗВЕДОЧНЫЙ -------------------------------
    base = dict(fixed_rank0=dict(rms=0.143),
                best_fixed_rank=dict(rms=0.143))
    good = dict(rms=0.12, capture=0.30)
    v = head_verdict(good, base, True, True, pilot_min=0.119)
    assert v["primary_passed"] and v["pilot_passed"] and v["passed"], v
    mid = head_verdict(dict(rms=0.13, capture=0.15), base, True, True,
                       pilot_min=0.119)
    assert not mid["primary_passed"] and mid["pilot_passed"], mid
    low = head_verdict(dict(rms=0.14, capture=0.10), base, True, True,
                       pilot_min=0.119)
    assert not low["primary_passed"] and not low["pilot_passed"], low
    # ОБЩИЕ УСЛОВИЯ ДЕРЖАТ И РАЗВЕДОЧНЫЙ: невоспроизводимое — не пилот
    nr = head_verdict(dict(rms=0.13, capture=0.15), base, True, False,
                      pilot_min=0.119)
    assert not nr["pilot_passed"], nr
    assert not head_verdict(dict(rms=0.15, capture=0.30), base, True,
                            True)["checks"]["better_than_rank0"]
    # БЕЗ ЭТАЛОНА M2 РАЗВЕДОЧНЫЙ СТАТУС НЕ ОПРЕДЕЛЁН, А НЕ «ПРОЙДЕН»
    assert head_verdict(good, base, True, True)["pilot_passed"] is None

    # --- M2 ПРОТИВ КЭША: FAIL-CLOSED ------------------------------------
    rows_v = np.array([30, 10, 20], np.int64)        # порядок плана
    man_t = dict(c1_sha1="C", reader_state_sha1="R", codec={"c": 1},
                 code_version={"v": 1}, joint_sha1="J",
                 parts=dict(val_sel=dict(rows=3, batches=1, aggregates=dict(
                     draft=0.145469, teacher=0.072892))),
                 m2_check=dict(reference=dict(draft=0.145469,
                                              teacher=0.072892)))
    m2_t = dict(kind=M2_KIND, selected_on="val_sel", c1_sha1="C",
                reader=dict(point_state_sha1="R"), codec={"c": 1},
                code_version={"v": 1}, joint_sha1="J", rows=3, batches=1,
                rows_sha1=sorted_rows_sha([10, 20, 30]),
                teacher=dict(full_val_sel=True, draft=0.145469,
                             rank=0.072892),
                rms=dict(a1_soft=0.140457),
                capture_soft=capture(0.145469, 0.072892, 0.140457))
    pm, probs = check_m2(m2_t, man_t, rows_v)
    assert probs == [] and abs(pm - (m2_t["capture_soft"] + 0.05)) < 1e-12
    # ОТПЕЧАТОК — МНОЖЕСТВА СТРОК: порядок плана его не меняет
    assert sorted_rows_sha([30, 10, 20]) == sorted_rows_sha([10, 20, 30])
    for key, val in (("kind", "иное"), ("selected_on", "val_sel_subset"),
                     ("c1_sha1", "ИНАЯ"), ("codec", {"c": 2}),
                     ("code_version", {"v": 2}), ("joint_sha1", "И"),
                     ("rows", 4), ("batches", 2),
                     ("rows_sha1", sorted_rows_sha([10, 20, 31])),
                     ("reader", dict(point_state_sha1="ИНОЙ")),
                     ("capture_soft", 0.5)):
        pm_, probs_ = check_m2(dict(m2_t, **{key: val}), man_t, rows_v)
        assert pm_ is None and probs_, (key, probs_)
    for tkey, val in (("full_val_sel", False), ("draft", 0.15),
                      ("rank", 0.07)):
        bad_t = dict(m2_t, teacher=dict(m2_t["teacher"], **{tkey: val}))
        assert check_m2(bad_t, man_t, rows_v)[0] is None, tkey
    assert check_m2({k: v for k, v in m2_t.items() if k != "capture_soft"},
                    man_t, rows_v)[0] is None
    # SMOKE: строки и агрегаты не сверяются, провенанс — сверяется
    assert check_m2(dict(m2_t, rows=99), man_t, rows_v, smoke=True)[1] == []
    assert check_m2(dict(m2_t, c1_sha1="ИНАЯ"), man_t, rows_v,
                    smoke=True)[0] is None

    # --- СТРОКИ ПРОБЫ: РАВНОМЕРНО ПО ЧАСТИ -----------------------------
    pr = probe_rows(121100, 64)
    assert len(pr) == 64 and pr[0] == 0 and pr[-1] == 121099, pr[[0, -1]]
    assert np.all(np.diff(pr) > 1000), "строки пробы идут подряд"
    assert probe_rows(10, 64).tolist() == list(range(10))
    assert len(set(probe_rows(100, 64).tolist())) == 64
    for bad in ((0, 64), (10, 0)):
        try:
            probe_rows(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"принято {bad}")

    # --- АРХИВ ПРЕЖНЕГО РАБОЧЕГО ЧЕКПОЙНТА ------------------------------
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        wp = os.path.join(td, "h24_linear_s0.pt")
        assert archive_working(wp) is None            # нечего архивировать
        open(wp, "w").write("старый")
        dest = archive_working(wp)
        assert dest and ".stale." in dest and not os.path.exists(wp)
        assert open(dest).read() == "старый"

    # --- ОБЩИЙ ИСХОД: РЕШАЮТ ТОЛЬКО h24-ГОЛОВЫ --------------------------
    def r(primary=False, pilot=False, trained=True, technical=False):
        return dict(trained=trained, technical=technical,
                    verdict=dict(primary_passed=primary,
                                 pilot_passed=pilot))
    # Сценарий из ревью: h24_linear прошла, h18_linear не прошла пробу
    assert overall_code({"h24_linear": r(primary=True),
                         "h18_linear": r(trained=False,
                                         technical=True)}) == CODE_PRIMARY
    # Отказ одной h24-головы не отменяет успех другой
    assert overall_code({"h24_linear": r(primary=True),
                         "h24_candidate": r(trained=False,
                                            technical=True)}) == CODE_PRIMARY
    # h18 прошла, h24 — нет: это НЕ успех развёртываемой архитектуры
    assert overall_code({"h18_linear": r(primary=True),
                         "h24_linear": r()}) == CODE_NONE
    assert overall_code({"h24_linear": r(pilot=True)}) == CODE_PILOT
    assert overall_code({"h24_linear": r(primary=True),
                         "h24_candidate": r(pilot=True)}) == CODE_PRIMARY
    # ни одной технически оценённой h24 — технический исход
    assert overall_code({"h18_linear": r(primary=True),
                         "h24_linear": r(trained=False,
                                         technical=True)}) == CODE_TECH
    assert overall_code({"h18_linear": r(primary=True)}) == CODE_TECH
    # ОБУЧЕНА, НО НЕВОСПРОИЗВОДИМА — технический отказ, а не наука:
    # такая голова исключается, даже если её числа «прошли»
    assert overall_code({"h24_linear": r(primary=True,
                                         technical=True)}) == CODE_TECH
    assert overall_code({"h24_linear": r(primary=True, technical=True),
                         "h24_candidate": r(pilot=True)}) == CODE_PILOT
    assert overall_code({"h24_linear": r(technical=True),
                         "h24_candidate": r(primary=True)}) == CODE_PRIMARY
    assert len({CODE_PRIMARY, CODE_PILOT, CODE_NONE, CODE_TECH}) == 4
    assert 2 not in (CODE_PRIMARY, CODE_PILOT, CODE_NONE, CODE_TECH)

    # --- СКВОЗНОЕ ОБУЧЕНИЕ НА СИНТЕТИКЕ ---------------------------------
    # Лучший ранг строки линейно задан состоянием: голова, видящая его,
    # обязана выучить выбор и пройти пробу. Это проверяет весь конвейер:
    # входы, потерю, оптимизатор, оценку, снимки и восстановление.
    torch.manual_seed(0)
    gen = np.random.default_rng(0)
    N, D, E, T = 600, 16, 6, 16
    W = gen.normal(size=(D, 8))
    ctx_np = gen.normal(size=(N, D)).astype(np.float32)
    best = (ctx_np @ W).argmax(1)
    costs_s = gen.uniform(0.5, 1.0, size=(N, 8)).astype(np.float32)
    costs_s[np.arange(N), best] = 0.05
    tr, va = np.arange(0, 500), np.arange(500, N)
    book = torch.randn(30, E)
    codes = torch.randint(0, 30, (N, T, 8))
    feats = rs.candidate_score_features(torch.randn(N, T, 8), torch)
    emb = rs.candidate_embeddings(codes, book, torch)
    ctx_t = torch.from_numpy(ctx_np)
    inp_tr = Inputs(torch, ctx_t[tr], emb[tr], feats[tr])
    inp_va = Inputs(torch, ctx_t[va], emb[va], feats[va])
    d_va = np.full(len(va), 1.0)
    t_va = np.full(len(va), 0.05)
    for name in ("h24_linear", "h24_candidate"):
        head = rs.build_head(name, D, E, torch, proj=16,
                             feat_mean=feats.mean((0, 1)),
                             feat_std=feats.std((0, 1)))
        pv = overfit_probe(head, inp_tr, np.arange(128), costs_s[tr], torch,
                           steps=120, lr=3e-2, wd=0.0)
        assert pv["passed"], (name, pv)
        hist, states = train_head(
            head, inp_tr, costs_s[tr], inp_va, costs_s[va], d_va, t_va,
            torch, epochs=25, batch=64, lr=1e-2, wd=0.0, patience=25,
            seed=0, log=lambda *_: None)
        sel = select_epoch(hist)
        assert sel["val"]["rms"] < hist[0]["val"]["rms"], (name, sel)
        assert len(states) == len(hist)
        if name == "h24_linear":
            assert sel["val"]["capture"] > 0.5, sel["val"]["capture"]
        # ВОССТАНОВЛЕНИЕ ВЫБРАННОЙ ЭПОХИ ВОСПРОИЗВОДИТ ЕЁ ОЦЕНКУ
        head.load_state_dict(states[sel["epoch"]])
        again = evaluate_scores(scores_for(head, inp_va, np.arange(len(va)),
                                           torch),
                                costs_s[va], d_va, t_va)
        assert again["rms"] == sel["val"]["rms"], (again["rms"],
                                                   sel["val"]["rms"])
    print("самопроверка k15c_train_rank_selector пройдена: метрики, опорные "
          "числа, критерий и сквозное обучение на синтетике (проба, "
          "отбор эпохи, восстановление)")


def sha12_bytes(b):
    return hashlib.sha1(b).hexdigest()[:12]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    ap = argparse.ArgumentParser(
        description="K-15c: обучение голов выбора ранга по кэшу")
    ap.add_argument("--cache", default="data/k15c/rank_cache")
    ap.add_argument("--allow-smoke", action="store_true")
    ap.add_argument("--c1", default="data/k15b/c1_selected.pt")
    ap.add_argument("--m2", default="reports/k15b/measure_soft.json")
    ap.add_argument("--heads", default=",".join(DEFAULT_HEADS))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--proj", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    # ПРОБА МАЛЕНЬКАЯ НАМЕРЕННО: она проверяет конвейер, и 64 строки
    # запоминает любая исправная голова. Неумение запомнить сотни
    # неоднозначных строк было бы научным результатом, а не поломкой.
    ap.add_argument("--overfit-rows", type=int, default=64)
    # БЮДЖЕТ, А НЕ ДЛИНА: проба останавливается, как только regret упал до
    # OVERFIT_DROP от начального. На 64 строках 2000 шагов — секунды.
    ap.add_argument("--overfit-steps", type=int, default=2000)
    ap.add_argument("--overfit-lr", type=float, default=1e-2)
    ap.add_argument("--out", default="data/k15c/selectors")
    ap.add_argument("--summary", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    import torch
    import k15c_build_rank_cache as cb
    import k15c_rank_selector as rs
    import k15_train_depth_rvq as k15t
    heads = [h.strip() for h in a.heads.split(",") if h.strip()]
    for h in heads:
        if h not in rs.HEADS:
            raise SystemExit(f"голова {h!r} не бывает: {rs.HEADS}")
    tag = "_smoke" if a.allow_smoke else ""
    summary = a.summary or f"reports/k15c/selector{tag}_s{a.seed}.json"
    if os.path.exists(summary) and not a.overwrite:
        raise SystemExit(f"{summary} уже существует: без --overwrite не "
                         f"перезаписываю")

    # --- КЭШ: FAIL-CLOSED ------------------------------------------------
    t_chk = time.time()
    man, problems = cb.validate_cache(a.cache, allow_smoke=a.allow_smoke)
    if problems:
        raise SystemExit("кэш не принят: " + "; ".join(problems[:6]))
    kind_s = "canonical" if man["canonical"] else "SMOKE"
    print(f"  кэш {a.cache} принят: {kind_s}"
          f", отпечатки {len(man['arrays'])} массивов сверены за "
          f"{time.time() - t_chk:.0f} с")
    book = torch.load(a.c1, map_location="cpu", weights_only=False)
    if book.get("kind") != "k15b_c1_selected" \
            or book.get("accepted") is not True:
        raise SystemExit(f"{a.c1}: книга не принята probe")
    C1 = book["c1"].detach().float().cpu()
    got = hashlib.sha1(np.ascontiguousarray(
        C1.numpy().astype(np.float32)).tobytes()).hexdigest()[:12]
    if got != man["c1_sha1"] or got != book["c1_sha1"]:
        raise SystemExit(f"книга {got}, в кэше {man['c1_sha1']}, в файле "
                         f"{book['c1_sha1']}")
    if int(C1.shape[1]) != int(man["e_dim"]):
        raise SystemExit("размерность книги не та, что в кэше")

    def arr(part, name):
        return np.load(os.path.join(a.cache,
                                    man["arrays"][f"{part}_{name}"]["file"]),
                       mmap_mode="r")

    d_model = int(man["d_model"])
    # БЕЗ МОЛЧАЛИВОГО ОТКАТА НА CPU: запрошенная карта либо есть, либо
    # это отказ. Карта здесь любая — тренер не загружает модель и гейт
    # K-15a не проверяет, поэтому пин устройства к нему не относится.
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"запрошено {a.device}, а CUDA недоступна")
    dev = torch.device(a.device)

    def pooled(part, src):
        if src == "h24":
            h = arr(part, "h24")
        elif man["h18_mode"] == "full":
            h = arr(part, "h18")
        elif man["h18_mode"] == "mean":
            return torch.from_numpy(np.asarray(arr(part, "h18_lnmean"),
                                               np.float32))
        else:
            return None
        out = []
        for s in range(0, h.shape[0], 4096):
            out.append(rs.ln_mean_pool(torch.from_numpy(
                np.asarray(h[s:s + 4096], np.float32)), torch))
        return torch.cat(out, 0)

    data = {}
    t_prep = time.time()
    for part in ("train", "val_sel"):
        # КОПИИ, А НЕ ВИДЫ НА memmap: файл открыт только на чтение, и
        # тензор поверх него был бы незаписываемым видом.
        codes = torch.from_numpy(np.array(arr(part, "top8_codes"),
                                          np.int64))
        lp = torch.from_numpy(np.array(arr(part, "top8_logprobs"),
                                       np.float32))
        emb, feat = [], []
        for s in range(0, codes.shape[0], 8192):
            emb.append(rs.candidate_embeddings(codes[s:s + 8192], C1, torch))
            feat.append(rs.candidate_score_features(lp[s:s + 8192], torch))
        data[part] = dict(
            ctx24=pooled(part, "h24"), ctx18=pooled(part, "h18"),
            emb=torch.cat(emb, 0), feat=torch.cat(feat, 0), codes=codes,
            costs=np.asarray(arr(part, "rank_costs"), np.float64),
            draft=np.asarray(arr(part, "draft_cost"), np.float64),
            tasks=np.asarray(arr(part, "task_ids")),
            h24=arr(part, "h24"))
        for k_ in ("ctx24", "emb", "feat"):
            if not torch.isfinite(data[part][k_]).all():
                raise SystemExit(f"{part}.{k_}: нечисловые значения")
    teacher_va = np.asarray(arr("val_sel", "teacher_cost"), np.float64)
    print(f"  признаки подготовлены за {time.time() - t_prep:.0f} с: "
          f"train {len(data['train']['costs'])} строк, val_sel "
          f"{len(data['val_sel']['costs'])}; d_model {d_model}, книга "
          f"{int(C1.shape[1])}")

    va, tr = data["val_sel"], data["train"]
    base = baselines(va["costs"], va["draft"], teacher_va, tr["costs"])
    # M2 ОБЯЗАТЕЛЕН И ПРОВЕРЯЕТСЯ: из него берётся разведочный порог, и
    # подменённый или устаревший файл переводил бы исход 4 <-> 6.
    if not os.path.exists(a.m2):
        raise SystemExit(f"нет {a.m2}: разведочный порог взять неоткуда")
    with open(a.m2, encoding="utf-8") as fh:
        m2 = json.load(fh)
    rows_val = np.asarray(arr("val_sel", "rows"))
    pilot_min, m2_problems = check_m2(m2, man, rows_val,
                                      smoke=bool(a.allow_smoke))
    if m2_problems:
        raise SystemExit(f"{a.m2} не соответствует кэшу: "
                         + "; ".join(m2_problems[:6]))
    m2_sha = cb.sha_file(a.m2)
    pilot_origin = (
        "capture(a1_soft) из канонического M2 плюс 0.05: порог сетки "
        "(k, tau), объявленный до данных 03.10.2026; применение к головам "
        "K-15c зарегистрировано до их обучения. Разведочный статус, НЕ "
        "научный критерий")
    best_grid = (m2.get("decision") or {}).get("best") or {}
    m2_soft = dict(a1_soft=(m2.get("rms") or {}).get("a1_soft"),
                   capture_soft=m2.get("capture_soft"),
                   best_grid_rms=best_grid.get("rms"),
                   best_grid_k=best_grid.get("k"),
                   best_grid_tau=best_grid.get("tau"))
    print(f"  M2 {a.m2} ({m2_sha}) сверен с кэшем"
          + (" (smoke: без строк и агрегатов)" if a.allow_smoke else ""))
    r_d, r_t = rms(va["draft"]), rms(teacher_va)
    print(f"\n  ОПОРНЫЕ ЧИСЛА val_sel (черновик {r_d:.6f}, учитель "
          f"{r_t:.6f}):")
    for key in ("hard_q1", "best_fixed_rank", "majority_rank",
                "uniform_random", "oracle_top8"):
        v = base[key]
        extra = f" (ранг {v['rank']})" if "rank" in v else ""
        print(f"    {key:16s}{extra:10s} RMS {v['rms']:.6f}, доля "
              f"{100 * (v['capture'] or 0):6.1f} %")
    if m2_soft and m2_soft.get("a1_soft") is not None:
        print(f"    мягкий путь M2 (внешняя опора) RMS "
              f"{m2_soft['a1_soft']:.6f}; лучшая точка сетки "
              f"{m2_soft.get('best_grid_rms')}")
    print(f"  ПОРОГИ: основной {100 * CAPTURE_MIN:.0f} %"
          + ("" if pilot_min is None else
             f"; разведочный {100 * pilot_min:.1f} % (мягкий M2 + 5 п.п., "
             f"оправдывает только проверку вывода и короткий роллаут)"))

    feat_mean = tr["feat"].mean((0, 1))
    feat_std = tr["feat"].std((0, 1))
    task_vocab = man.get("task_vocab")
    results = {}
    os.makedirs(a.out, exist_ok=True)
    for name in heads:
        src = "h18" if name.startswith("h18") else "h24"
        ctx_tr = tr["ctx18"] if src == "h18" else tr["ctx24"]
        ctx_va = va["ctx18"] if src == "h18" else va["ctx24"]
        if ctx_tr is None:
            print(f"\n  {name}: h18 в кэше не сохранён — голова пропущена")
            results[name] = dict(skipped="h18 не сохранён")
            continue
        full = name in rs.NEEDS_FULL_H24
        need_codes = name in rs.NEEDS_BOOK
        inp_tr = Inputs(torch, ctx_tr, tr["emb"], tr["feat"],
                        h_full=tr["h24"] if full else None, device=dev,
                        cand_codes=tr["codes"] if need_codes else None)
        inp_va = Inputs(torch, ctx_va, va["emb"], va["feat"],
                        h_full=va["h24"] if full else None, device=dev,
                        cand_codes=va["codes"] if need_codes else None)
        torch.manual_seed(int(a.seed))
        head = rs.build_head(name, d_model, int(C1.shape[1]), torch,
                             proj=int(a.proj), feat_mean=feat_mean,
                             feat_std=feat_std, book=C1).to(dev)
        n_par = sum(p.numel() for p in head.parameters())
        print(f"\n  {name}: {n_par} параметров, вход {src}")
        path = os.path.join(a.out, f"{name}{tag}_s{a.seed}.pt")
        probe_idx = probe_rows(len(tr["costs"]), int(a.overfit_rows))
        pv = overfit_probe(head, inp_tr, probe_idx, tr["costs"], torch,
                           steps=int(a.overfit_steps), lr=float(a.overfit_lr),
                           wd=0.0)
        pv["rows"] = [int(x) for x in probe_idx]
        print(f"    проба обучаемости на {len(probe_idx)} строках, "
              f"{pv['steps_used']} шагов из {pv['steps_budget']}: regret "
              f"{pv['first']:.4e} -> {pv['last']:.4e} (отношение "
              f"{pv['ratio']:.3f}), hard {pv['hard_before']:.4e} -> "
              f"{pv['hard_after']:.4e}; "
              + ("ПРОЙДЕНА" if pv["passed"] else
                 "НЕ ПРОЙДЕНА "
                 f"{[k for k, v in pv['checks'].items() if not v]}")
              + f"; диагностика {pv['diagnostics']}")
        # ТЕХНИЧЕСКИЙ ОТКАЗ ОТНОСИТСЯ К ЭТОЙ ГОЛОВЕ, а не ко всему запуску:
        # чужая проба не может отменить успех другой головы.
        if not pv["passed"]:
            results[name] = dict(overfit_probe=pv, trained=False,
                                 technical="проба конвейера не пройдена",
                                 archived_stale=archive_working(path))
            continue
        try:
            hist, states = train_head(
                head, inp_tr, tr["costs"], inp_va, va["costs"], va["draft"],
                teacher_va, torch, epochs=int(a.epochs), batch=int(a.batch),
                lr=float(a.lr), wd=float(a.wd), patience=int(a.patience),
                seed=int(a.seed), task_ids=va["tasks"],
                task_vocab=task_vocab)
        except (FloatingPointError, ValueError) as e:
            results[name] = dict(overfit_probe=pv, trained=False,
                                 technical=str(e),
                                 archived_stale=archive_working(path))
            continue
        sel = select_epoch(hist)
        head.load_state_dict(states[sel["epoch"]])
        again = evaluate_scores(
            scores_for(head, inp_va, np.arange(len(va["costs"])), torch),
            va["costs"], va["draft"], teacher_va)
        reproduced = again["rms"] == sel["val"]["rms"]
        payload = dict(
            kind="k15c_rank_selector", head=name, smoke=bool(a.allow_smoke),
            d_model=d_model, e_dim=int(C1.shape[1]), proj=int(a.proj),
            selected_epoch=int(sel["epoch"]),
            state={k: v.clone() for k, v in states[sel["epoch"]].items()},
            all_states=states, history=hist,
            cache_manifest_sha1=cb.sha_file(os.path.join(a.cache,
                                                         "manifest.json")),
            cache=os.path.abspath(a.cache), c1_sha1=man["c1_sha1"],
            reader_state_sha1=man["reader_state_sha1"],
            hyper=dict(epochs=a.epochs, batch=a.batch, lr=a.lr, wd=a.wd,
                       patience=a.patience, seed=a.seed, loss="regret",
                       lambda_ce=0.0))
        tmp = path + f".tmp.{os.getpid()}"
        torch.save(payload, tmp)
        back = torch.load(tmp, map_location="cpu", weights_only=False)
        fresh = rs.build_head(name, d_model, int(C1.shape[1]), torch,
                              proj=int(a.proj), book=C1).to(dev)
        fresh.load_state_dict(back["state"])
        idx_chk = np.arange(min(2048, len(va["costs"])))
        loaded_equal = np.array_equal(scores_for(fresh, inp_va, idx_chk,
                                                 torch),
                                      scores_for(head, inp_va, idx_chk,
                                                 torch))
        finite = all(np.isfinite(h["val"]["rms"]) for h in hist)
        # НЕВОСПРОИЗВОДИМОЕ — ТЕХНИЧЕСКИЙ ОТКАЗ, А НЕ НАУЧНЫЙ ОТРИЦАТЕЛЬНЫЙ
        # РЕЗУЛЬТАТ. Голова исключается из исхода, а её чекпойнт не
        # публикуется под рабочим именем: раннер и проверка вывода не должны
        # его найти.
        tech = [why for ok_, why in (
            (finite, "в истории нечисловой RMS"),
            (reproduced, "выбранная эпоха не воспроизвелась"),
            (loaded_equal, "сохранение/загрузка изменили оценки"))
            if not ok_]
        if tech:
            archive_working(path)
            path = path[:-3] + ".technical_fail.pt"
        os.replace(tmp, path)
        verdict = head_verdict(sel["val"], base, finite,
                               bool(reproduced and loaded_equal),
                               pilot_min=pilot_min)
        results[name] = dict(
            overfit_probe=pv, trained=True, selected_epoch=int(sel["epoch"]),
            epochs_run=len(hist) - 1, val=sel["val"],
            reproduced=bool(reproduced), save_load_equal=bool(loaded_equal),
            verdict=verdict, checkpoint=os.path.abspath(path),
            technical=("; ".join(tech) if tech else None),
            trajectory=[dict(epoch=h["epoch"], rms=h["val"]["rms"],
                             capture=h["val"]["capture"],
                             train_regret=h["train_regret"]) for h in hist])
        v = sel["val"]
        print(f"    ВЫБРАНА эпоха {sel['epoch']} из {len(hist) - 1}: hard RMS "
              f"{v['rms']:.6f}, доля {100 * (v['capture'] or 0):.1f} % "
              f"(порог {100 * CAPTURE_MIN:.0f} %); regret медиана "
              f"{v['regret_median']:.3e}, p95 {v['regret_p95']:.3e}; "
              f"точность лучшего ранга {100 * v['best_rank_accuracy']:.1f} "
              f"% (диагностика); ранг 0 у {100 * v['share_rank0']:.1f} %")
        print(f"    гистограмма выбора: " + ", ".join(
            f"{k}:{c}" for k, c in sorted(v["histogram"].items())))
        if tech:
            print(f"    ТЕХНИЧЕСКИЙ ОТКАЗ: {'; '.join(tech)} — голова "
                  f"исключена из исхода, чекпойнт записан как {path}")
        failed = [k for k, x in verdict["checks"].items() if x is False]
        role = ("" if name.startswith(DEPLOYABLE_PREFIX)
                else " — АБЛЯЦИЯ ГЛУБИНЫ, на исход не влияет")
        print(f"    воспроизведение {'ok' if reproduced else 'НЕТ'}, "
              f"сохранение/загрузка {'ok' if loaded_equal else 'НЕТ'}; "
              f"основной {'ПРОЙДЕН' if verdict['primary_passed'] else 'нет'}"
              f", разведочный "
              + {True: "ПРОЙДЕН", False: "нет", None: "не определён"}[
                  verdict["pilot_passed"]]
              + f"{role} {failed or ''}")

    trained = {k: v for k, v in results.items() if v.get("trained")}
    technical = {k: v["technical"] for k, v in results.items()
                 if v.get("technical")}
    code = overall_code(results)
    depth = None
    if "h18_linear" in trained and "h24_linear" in trained:
        r18 = trained["h18_linear"]["val"]["rms"]
        r24 = trained["h24_linear"]["val"]["rms"]
        depth = dict(h18_rms=r18, h24_rms=r24,
                     h18_capture=trained["h18_linear"]["val"]["capture"],
                     h24_capture=trained["h24_linear"]["val"]["capture"],
                     relation=("h24 > h18" if r24 < r18 else
                               "h24 = h18" if r24 == r18 else "h24 < h18"))
        print(f"\n  ГЛУБИНА: h18_linear {r18:.6f} "
              f"({100 * (depth['h18_capture'] or 0):.1f} %) против "
              f"h24_linear {r24:.6f} "
              f"({100 * (depth['h24_capture'] or 0):.1f} %) — "
              f"{depth['relation']}")
    outcome = {
        CODE_PRIMARY: "h24-голова взяла ОСНОВНОЙ критерий: дальше проверка "
                      "вывода, safety-роллаут задач 8-9 и парный dev",
        CODE_PILOT: "основной критерий не взят, но h24-голова взяла "
                    "РАЗВЕДОЧНЫЙ порог: проверка вывода и только короткий "
                    "разведочный роллаут, архитектурного успеха это не "
                    "доказывает",
        CODE_NONE: "ни одна h24-голова не взяла порогов на замороженном "
                   "h24: кэш не пересобирать, следующая дешёвая голова — "
                   "h24_positional",
        CODE_TECH: f"ни одной технически оценённой h24-головы: {technical}",
    }[code]
    if technical:
        print(f"\n  технические отказы отдельных голов: {technical}")
    print(f"\n  ИСХОД: {outcome} (код {code}). Выбор на val_sel — часть "
          f"отбора, не независимая проверка; val_confirm не открывалась")
    out = dict(kind="k15c_selector_summary", code=code, outcome=outcome,
               heads=heads, results=results, baselines=base, depth=depth,
               m2_soft=m2_soft, m2_file=os.path.abspath(a.m2),
               m2_sha1=m2_sha, technical_failures=technical,
               capture_min=CAPTURE_MIN, pilot_min=pilot_min,
               pilot_origin=pilot_origin,
               deployable_prefix=DEPLOYABLE_PREFIX,
               codes=dict(primary=CODE_PRIMARY, pilot=CODE_PILOT,
                          none=CODE_NONE, technical=CODE_TECH),
               selected_on="val_sel",
               val_confirm_used_for_selection=False,
               cache=os.path.abspath(a.cache),
               cache_canonical=bool(man["canonical"]),
               cache_manifest_sha1=cb.sha_file(
                   os.path.join(a.cache, "manifest.json")),
               hyper=dict(epochs=a.epochs, batch=a.batch, lr=a.lr, wd=a.wd,
                          patience=a.patience, proj=a.proj, seed=a.seed,
                          overfit_rows=a.overfit_rows,
                          overfit_steps=a.overfit_steps,
                          overfit_lr=a.overfit_lr))
    os.makedirs(os.path.dirname(os.path.abspath(summary)) or ".",
                exist_ok=True)
    tmp = summary + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        # СТРОГАЯ СХЕМА: прежний `default=str` молча превращал бы любой
        # неописанный объект в строку, и ошибка артефакта не всплывала бы.
        json.dump(out, fh, ensure_ascii=False, indent=1, allow_nan=False,
                  default=k15t.json_scalar)
    os.replace(tmp, summary)
    print(f"  сводка: {summary}")
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
