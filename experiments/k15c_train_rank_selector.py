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

Одна дополнительная попытка разрешена только если ни одна не взяла порог:
`--heads h24_candidate_attn` (пулинг вниманием вместо среднего).

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

ПРОБА ОБУЧАЕМОСТИ ПЕРЕД ПОЛНЫМ ОБУЧЕНИЕМ. На фиксированных строках
train regret обязан падать, hard-затрата — улучшиться, параметры —
измениться, а логиты после восстановления снимка — совпасть побитово. Это
проверка конвейера, а не научная достижимость; если проба не прошла, полное
обучение не запускается и исход — код 3.

КОДЫ: 0 — хотя бы одна голова взяла критерий; 4 — ни одна; 3 — проба
обучаемости или нечисловые величины; провенанс кэша — отказ с текстом.
Выбор делается на val_sel и помечается так; val_confirm не открывается.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

CAPTURE_MIN = 0.20
OVERFIT_DROP = 0.67
OVERFIT_WINDOWS = 6
DEFAULT_HEADS = ("h18_linear", "h24_linear", "h24_candidate")


def rms(costs):
    return float(np.sqrt(np.mean(np.asarray(costs, np.float64))))


def capture(rms_draft, rms_teacher, rms_policy):
    """Доля разрыва; None при неположительном разрыве. Как в K-15b."""
    gap = float(rms_draft) - float(rms_teacher)
    if gap <= 1e-12:
        return None
    return float((float(rms_draft) - float(rms_policy)) / gap)


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
                  drop=OVERFIT_DROP):
    """Проба обучаемости: четыре условия, все обязательны."""
    first, last = float(trace[0]), float(trace[-1])
    monotone = all(float(b) <= float(a) * (1.0 + 1e-3) + 1e-12
                   for a, b in zip(trace, trace[1:]))
    checks = dict(
        regret_dropped=bool(last <= drop * first + 1e-12),
        regret_monotone=bool(monotone),
        hard_improved=bool(hard_after < hard_before),
        params_changed=bool(changed),
        restore_exact=bool(restored_equal))
    return dict(passed=all(checks.values()), checks=checks,
                first=first, last=last, ratio=float(last / max(first, 1e-12)),
                hard_before=float(hard_before), hard_after=float(hard_after),
                trace=[float(x) for x in trace], required_drop=float(drop))


def select_epoch(history):
    """Минимум hard RMS на val_sel; ничья — более ранняя эпоха."""
    if not history:
        raise ValueError("история пуста")
    return min(history, key=lambda h: (float(h["val"]["rms"]),
                                       int(h["epoch"])))


def head_verdict(val, base, finite, reproducible, cap_min=CAPTURE_MIN):
    """Критерий головы. Точность в нём не участвует."""
    checks = dict(
        capture_ok=bool(val.get("capture") is not None
                        and float(val["capture"]) >= cap_min - 1e-12),
        better_than_rank0=bool(val["rms"] < base["fixed_rank0"]["rms"]),
        better_than_best_fixed=bool(
            val["rms"] < base["best_fixed_rank"]["rms"]),
        finite=bool(finite), reproducible=bool(reproducible))
    return dict(passed=all(checks.values()), checks=checks,
                capture_min=float(cap_min))


def overall_code(verdicts, technical_failures):
    if technical_failures:
        return 3
    return 0 if any(v["passed"] for v in verdicts.values()) else 4


class Inputs:
    """Входы головы по индексам строк. Держит тензоры на устройстве."""

    def __init__(self, torch, ctx, cand_emb, cand_feat, h_full=None,
                 device="cpu"):
        self.torch = torch
        self.ctx = None if ctx is None else ctx.to(device)
        self.cand_emb = cand_emb.to(device)
        self.cand_feat = cand_feat.to(device)
        self.h_full = h_full           # memmap или тензор на CPU
        self.device = device

    def batch(self, idx):
        t = self.torch
        i = t.as_tensor(np.asarray(idx), device=self.device)
        kw = dict(ctx=None if self.ctx is None else self.ctx[i],
                  cand_emb=self.cand_emb[i], cand_feat=self.cand_feat[i])
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


def overfit_probe(head, inp, idx, costs, torch, steps, lr, wd):
    """Проба обучаемости на фиксированных строках, с восстановлением."""
    import k15c_rank_selector as rs
    dev = inp.device
    snap = {k: v.detach().clone() for k, v in head.state_dict().items()}
    c = torch.as_tensor(np.asarray(costs[idx], np.float32), device=dev)
    kw = inp.batch(idx)
    head.eval()
    with torch.no_grad():
        s0 = head(**kw).detach().clone()
    hard_before = float(rs.hard_selected_cost(s0, c, torch)[0].mean())
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    trace, win = [], []
    head.train()
    for st in range(1, int(steps) + 1):
        loss = rs.expected_regret(head(**kw), c, torch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        win.append(float(loss.detach()))
        if st % max(int(steps) // OVERFIT_WINDOWS, 1) == 0:
            trace.append(float(np.mean(win)))
            win = []
    head.eval()
    with torch.no_grad():
        s1 = head(**kw)
    hard_after = float(rs.hard_selected_cost(s1, c, torch)[0].mean())
    changed = any(not torch.equal(v, snap[k])
                  for k, v in head.state_dict().items())
    head.load_state_dict(snap)
    with torch.no_grad():
        s2 = head(**kw)
    return probe_verdict(trace, hard_before, hard_after, changed,
                         bool(torch.equal(s2, s0)))


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

    # --- ПРОБА, ВЫБОР ЭПОХИ, ИСХОД --------------------------------------
    pv = probe_verdict([1.0, 0.8, 0.6, 0.5, 0.4, 0.3], 0.9, 0.5, True, True)
    assert pv["passed"], pv
    assert not probe_verdict([1.0, 0.9, 0.95, 0.5], 0.9, 0.5, True,
                             True)["checks"]["regret_monotone"]
    assert not probe_verdict([1.0, 0.9], 0.9, 0.5, True,
                             True)["checks"]["regret_dropped"]
    assert not probe_verdict([1.0, 0.5], 0.9, 0.9, True,
                             True)["checks"]["hard_improved"]
    assert not probe_verdict([1.0, 0.5], 0.9, 0.5, False, True)["passed"]
    assert not probe_verdict([1.0, 0.5], 0.9, 0.5, True, False)["passed"]
    hist = [dict(epoch=0, val=dict(rms=0.14)),
            dict(epoch=1, val=dict(rms=0.12)),
            dict(epoch=2, val=dict(rms=0.12))]
    assert select_epoch(hist)["epoch"] == 1
    base = dict(fixed_rank0=dict(rms=0.143),
                best_fixed_rank=dict(rms=0.143))
    good = dict(rms=0.12, capture=0.30)
    assert head_verdict(good, base, True, True)["passed"]
    assert not head_verdict(dict(rms=0.12, capture=0.19), base, True,
                            True)["passed"]
    assert not head_verdict(good, base, True, False)["passed"]
    assert not head_verdict(dict(rms=0.15, capture=0.30), base, True,
                            True)["checks"]["better_than_rank0"]
    assert overall_code({"a": dict(passed=True)}, []) == 0
    assert overall_code({"a": dict(passed=False)}, []) == 4
    assert overall_code({"a": dict(passed=True)}, ["x"]) == 3

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
    ap.add_argument("--overfit-rows", type=int, default=512)
    ap.add_argument("--overfit-steps", type=int, default=300)
    ap.add_argument("--overfit-lr", type=float, default=3e-3)
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
            emb=torch.cat(emb, 0), feat=torch.cat(feat, 0),
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
    m2_soft = None
    if os.path.exists(a.m2):
        with open(a.m2, encoding="utf-8") as fh:
            m2 = json.load(fh)
        best_grid = (m2.get("decision") or {}).get("best") or {}
        m2_soft = dict(a1_soft=(m2.get("rms") or {}).get("a1_soft"),
                       best_grid_rms=best_grid.get("rms"),
                       best_grid_k=best_grid.get("k"),
                       best_grid_tau=best_grid.get("tau"))
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

    feat_mean = tr["feat"].mean((0, 1))
    feat_std = tr["feat"].std((0, 1))
    task_vocab = man.get("task_vocab")
    results, technical = {}, []
    os.makedirs(a.out, exist_ok=True)
    for name in heads:
        src = "h18" if name.startswith("h18") else "h24"
        ctx_tr = tr["ctx18"] if src == "h18" else tr["ctx24"]
        ctx_va = va["ctx18"] if src == "h18" else va["ctx24"]
        if ctx_tr is None:
            print(f"\n  {name}: h18 в кэше не сохранён — голова пропущена")
            results[name] = dict(skipped="h18 не сохранён")
            continue
        attn = name.endswith("_attn")
        inp_tr = Inputs(torch, ctx_tr, tr["emb"], tr["feat"],
                        h_full=tr["h24"] if attn else None, device=dev)
        inp_va = Inputs(torch, ctx_va, va["emb"], va["feat"],
                        h_full=va["h24"] if attn else None, device=dev)
        torch.manual_seed(int(a.seed))
        head = rs.build_head(name, d_model, int(C1.shape[1]), torch,
                             proj=int(a.proj), feat_mean=feat_mean,
                             feat_std=feat_std).to(dev)
        n_par = sum(p.numel() for p in head.parameters())
        print(f"\n  {name}: {n_par} параметров, вход {src}")
        probe_idx = np.arange(min(int(a.overfit_rows), len(tr["costs"])))
        pv = overfit_probe(head, inp_tr, probe_idx, tr["costs"], torch,
                           steps=int(a.overfit_steps), lr=float(a.overfit_lr),
                           wd=0.0)
        print(f"    проба обучаемости на {len(probe_idx)} строках: regret "
              f"{pv['first']:.4e} -> {pv['last']:.4e} (отношение "
              f"{pv['ratio']:.3f}), hard {pv['hard_before']:.4e} -> "
              f"{pv['hard_after']:.4e}; "
              + ("ПРОЙДЕНА" if pv["passed"] else
                 "НЕ ПРОЙДЕНА "
                 f"{[k for k, v in pv['checks'].items() if not v]}"))
        if not pv["passed"]:
            technical.append(f"{name}: проба обучаемости")
            results[name] = dict(overfit_probe=pv, trained=False)
            continue
        try:
            hist, states = train_head(
                head, inp_tr, tr["costs"], inp_va, va["costs"], va["draft"],
                teacher_va, torch, epochs=int(a.epochs), batch=int(a.batch),
                lr=float(a.lr), wd=float(a.wd), patience=int(a.patience),
                seed=int(a.seed), task_ids=va["tasks"],
                task_vocab=task_vocab)
        except (FloatingPointError, ValueError) as e:
            technical.append(f"{name}: {e}")
            results[name] = dict(overfit_probe=pv, trained=False,
                                 error=str(e))
            continue
        sel = select_epoch(hist)
        head.load_state_dict(states[sel["epoch"]])
        again = evaluate_scores(
            scores_for(head, inp_va, np.arange(len(va["costs"])), torch),
            va["costs"], va["draft"], teacher_va)
        reproduced = again["rms"] == sel["val"]["rms"]
        path = os.path.join(a.out, f"{name}{tag}_s{a.seed}.pt")
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
                              proj=int(a.proj)).to(dev)
        fresh.load_state_dict(back["state"])
        idx_chk = np.arange(min(2048, len(va["costs"])))
        loaded_equal = np.array_equal(scores_for(fresh, inp_va, idx_chk,
                                                 torch),
                                      scores_for(head, inp_va, idx_chk,
                                                 torch))
        os.replace(tmp, path)
        finite = all(np.isfinite(h["val"]["rms"]) for h in hist)
        verdict = head_verdict(sel["val"], base, finite,
                               bool(reproduced and loaded_equal))
        results[name] = dict(
            overfit_probe=pv, trained=True, selected_epoch=int(sel["epoch"]),
            epochs_run=len(hist) - 1, val=sel["val"],
            reproduced=bool(reproduced), save_load_equal=bool(loaded_equal),
            verdict=verdict, checkpoint=os.path.abspath(path),
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
        print(f"    воспроизведение {'ok' if reproduced else 'НЕТ'}, "
              f"сохранение/загрузка {'ok' if loaded_equal else 'НЕТ'}; "
              f"критерий {'ПРОЙДЕН' if verdict['passed'] else 'не пройден'}"
              f" {[k for k, x in verdict['checks'].items() if not x] or ''}")

    trained = {k: v for k, v in results.items() if v.get("trained")}
    verdicts = {k: v["verdict"] for k, v in trained.items()}
    code = overall_code(verdicts, technical)
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
    outcome = {0: "хотя бы одна голова взяла критерий: дальше проверка "
                  "вывода настоящим проходом",
               4: "ни одна голова не взяла критерий на замороженном h24",
               3: f"технический отказ: {technical}"}[code]
    print(f"\n  ИСХОД: {outcome} (код {code}). Выбор на val_sel — часть "
          f"отбора, не независимая проверка; val_confirm не открывалась")
    out = dict(kind="k15c_selector_summary", code=code, outcome=outcome,
               heads=heads, results=results, baselines=base, depth=depth,
               m2_soft=m2_soft, technical_failures=technical,
               capture_min=CAPTURE_MIN, selected_on="val_sel",
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
        json.dump(out, fh, ensure_ascii=False, indent=1, allow_nan=False,
                  default=lambda o: o.item() if hasattr(o, "item")
                  else str(o))
    os.replace(tmp, summary)
    print(f"  сводка: {summary}")
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
