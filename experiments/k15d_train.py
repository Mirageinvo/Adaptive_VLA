#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15d: обучение уточнения D0 и H1 (фазы h1p1, h1p2).

Фазы:
  d0    — a_d = a0 + Δ_d(h24, a0); LoRA слоёв 13–24, φ_d, голова.
  h1p1  — a1 = a0 + Δ1(h18, a0);   LoRA 13–18, φ1, голова 1. Проход
          останавливается на слое 18: слои 19–24 фазе не нужны.
  h1p2  — a2 = a1 + Δ2(h24, a1);   LoRA 19–24, φ2, голова 2. Лучшая точка
          h1p1 загружается из её чекпойнта и заморожена.

Режимы:
  overfit — 32 фиксированные строки train, 300 шагов: техническая
            обучаемость (потеря падает, выход уходит от q0, градиенты
            доходят до головы, φ и LoRA, восстановление возвращает q0);
  smoke   — 100 батчей train, 40 батчей val_sel: контракт данных, нулевая
            точка, градиенты, отпечаток замороженного, сохранение/загрузка,
            диапазон. Решений по величине улучшения не принимается;
  full    — одна эпоха train, снапшоты на 0, 5000, 10000 и в конце; отбор
            по RMS уровня фазы на всём val_sel; диагностика причинности по
            выбранной точке; фильтр допуска к роллауту (план §12).

Коды выхода: 0 — технически исправно (в full — и допущено к роллауту);
4 — технически исправно, но фильтр допуска не пройден; 3 — технический
отказ. val_confirm не открывается ни в каком режиме.
"""
import argparse
import datetime
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k15_context  # noqa: E402
import k15d_check_init_identity as kg  # noqa: E402
import k15d_depth_refine as dr  # noqa: E402
from k15_train_depth_rvq import H_EXEC  # noqa: E402

KIND = "k15d_phase"
# RMS черновика q0 на val_sel, измеренный в K-15b/K-15c тем же
# weighted_row_error. В полном режиме совпадение обязательно: иначе метрика
# K-15d считала бы не то же самое, с чем сравнивались прежние результаты.
REF_Q0_VAL_RMS = 0.145469
REF_Q0_REL_TOL = 1e-4
# Фильтр допуска к роллауту (план §12). Это защита от сломанной модели, а
# не научный критерий.
ADMIT = dict(global_factor=1.05, task_factor=1.10, range_factor=1.5,
             clip_bound=1.5, grip_acc_drop=0.02)
OVERFIT = dict(batches=4, steps=300, min_drop=0.5, min_move=1e-3)
SMOKE = dict(train_batches=100, val_batches=40, snap_every=50)
# В полном режиме промежуточный снапшот ближе этого к концу эпохи не
# делается: точки 15000 и 15138 почти совпадали бы, а стоят 5 минут.
SNAP_END_MARGIN = 1000
GROUPS = ("heads", "feedback", "lora")


def strided(n, k):
    return kg.strided(n, k)


def snap_points(total, every, margin):
    pts = [0]
    s = int(every)
    while s < total:
        if total - s >= margin:
            pts.append(s)
        s += int(every)
    pts.append(int(total))
    return sorted(set(pts))


def lr_lambda(warmup, total, min_frac):
    def f(step):
        if step < warmup:
            return float(step + 1) / float(max(warmup, 1))
        t = (step - warmup) / float(max(total - warmup, 1))
        t = min(max(t, 0.0), 1.0)
        return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * t))
    return f


def eval_levels(phase):
    """Какие уровни считаются в оценке фазы и какой из них отбирается."""
    lv = dr.PHASES[phase]
    if phase == "h1p2":
        return ["1", "2"], lv
    return [lv], lv


def admission(metrics, lv, act_p99, *, smoke):
    """Фильтр допуска уровня lv. Чистая по metrics."""
    m = metrics
    r_lv, r0 = m["rms"][lv], m["rms"]["a0"]
    gates = {}
    gates["finite"] = dict(passed=bool(m["finite"][lv]))
    gates["global_rms"] = dict(
        value=r_lv, q0=r0, limit=ADMIT["global_factor"] * r0,
        passed=bool(r_lv <= ADMIT["global_factor"] * r0))
    worst, worst_task = 0.0, None
    for t, d in m["per_task"].items():
        ratio = d[lv] / max(d["a0"], 1e-12)
        if ratio > worst:
            worst, worst_task = ratio, t
    gates["task_rms"] = dict(worst_ratio=worst, worst_task=worst_task,
                             limit=ADMIT["task_factor"],
                             passed=bool(worst <= ADMIT["task_factor"]))
    p99 = np.asarray(m["p99"][lv], np.float64)
    amax = np.asarray(m["absmax"][lv], np.float64)
    ratio = p99 / np.maximum(np.asarray(act_p99, np.float64), 1e-12)
    gates["range"] = dict(
        p99=p99.tolist(), absmax=amax.tolist(), ratio=ratio.tolist(),
        range_factor=ADMIT["range_factor"], clip_bound=ADMIT["clip_bound"],
        passed=bool((ratio <= ADMIT["range_factor"]).all()
                    and (amax <= ADMIT["clip_bound"]).all()))
    acc, acc0 = m["grip_acc"][lv], m["grip_acc"]["a0"]
    gates["gripper"] = dict(acc=acc, acc_q0=acc0,
                            limit=acc0 - ADMIT["grip_acc_drop"],
                            passed=bool(acc >= acc0 - ADMIT["grip_acc_drop"]))
    passed = all(g["passed"] for g in gates.values())
    return dict(level=lv, gates=gates, passed=bool(passed),
                decided=not smoke,
                admissible=bool(passed and not smoke),
                note=("SMOKE: допуск не решается" if smoke else
                      "фильтр безопасности, не критерий научного успеха"))


def select_snapshot(history, lv):
    """Минимум RMS уровня; при равенстве — более ранняя точка."""
    best = None
    for h in history:
        v = h["metrics"]["rms"][lv]
        if not np.isfinite(v):
            continue
        if best is None or v < best["metrics"]["rms"][lv]:
            best = h
    if best is None:
        raise SystemExit("ни одна точка не дала конечного RMS")
    return best["tag"]


def main():
    ap = argparse.ArgumentParser(description="K-15d: обучение уточнения")
    k15_context.add_common_arguments(ap)
    ap.add_argument("--phase", choices=sorted(dr.PHASES))
    ap.add_argument("--mode", choices=("full", "smoke", "overfit"),
                    default="full")
    ap.add_argument("--phase1", default=None,
                    help="чекпойнт h1p1 для фазы h1p2")
    ap.add_argument("--allow-smoke-phase1", action="store_true",
                    help="h1p2 в smoke/overfit может стартовать от "
                         "smoke-h1p1")
    ap.add_argument("--k15d-gate", default="reports/k15d/init_identity.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--overfit-lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--snap-every", type=int, default=5000)
    ap.add_argument("--report-every", type=int, default=250)
    ap.add_argument("--val-batches", type=int, default=0,
                    help="0 — весь val_sel (обязательно в full)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.phase is None:
        ap.error("--phase обязателен")
    if a.mode == "full" and int(a.val_batches) != 0:
        raise SystemExit("в full отбор идёт по ВСЕМУ val_sel")
    if a.phase == "h1p2" and not a.phase1:
        raise SystemExit("h1p2 требует --phase1")
    suffix = "" if a.mode == "full" else f"_{a.mode}"
    tag_run = f"{a.phase}_s{a.seed}{suffix}"
    out = a.out or f"data/k15d/{tag_run}.pt"
    report = a.report or f"reports/k15d/{tag_run}.json"
    rows_npz = os.path.splitext(report)[0] + "_rows.npz"
    partial, validating = out + ".partial", out + ".validating"
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + \
        f"-{os.getpid()}"
    t_start = time.time()

    ctx = k15_context.build(a)
    torch, model, dev = ctx.torch, ctx.model, ctx.dev
    k15t = k15_context.k15t
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    frozen0, n_frozen, _ne = k15t.frozen_content_sha(model, torch, set())
    stats = dr.compute_stats(ctx, H_EXEC)
    gate = kg.check_gate_report(a.k15d_gate, expect=dict(
        device=str(dev), compute_dtype=a.dtype, code=kg.code_shas(),
        architecture_code_version=ctx.code_version,
        joint_sha1=ctx.joint_sha, plan_sha1=ctx.q0_prov["plan_sha1"],
        frozen_sha1=frozen0, stats_sha1=stats["sha1"], hp=dict(dr.HP),
        seed=int(a.seed)))
    print(f"  гейт K-15d {gate['run_id']} сверен; статистика "
          f"{stats['sha1']}, замороженное {frozen0}")

    variant = dr.PHASE_VARIANT[a.phase]
    names_eval, lv = eval_levels(a.phase)
    ref, n_hooks = dr.make_refiner(ctx, variant, stats, a.seed)
    # Начальное состояние обязано быть тем, что аттестовал гейт: тот же
    # сид гейта ещё не гарантирует ту же инициализацию.
    init_sha = dr.state_sha(ref)
    want_init = ((gate.get("variants") or {}).get(variant) or {}).get(
        "init_state_sha1")
    if init_sha != want_init:
        raise SystemExit(f"начальное состояние {variant} {init_sha}, гейт "
                         f"аттестовал {want_init}")
    phase1_prov = None
    if a.phase == "h1p2":
        p1 = torch.load(a.phase1, map_location="cpu", weights_only=False)
        problems = []
        if p1.get("kind") != KIND or p1.get("phase") != "h1p1":
            problems.append(f"kind/phase {p1.get('kind')}/{p1.get('phase')}")
        if p1.get("mode") != "full" and not (
                a.allow_smoke_phase1 and a.mode in ("smoke", "overfit")):
            problems.append(f"режим фазы 1 {p1.get('mode')}")
        for key, want in (("stats_sha1", stats["sha1"]),
                          ("frozen_sha1", frozen0),
                          ("gate_run_id", gate["run_id"]),
                          ("seed", int(a.seed)),
                          ("status", "complete"),
                          ("final", True),
                          ("technical_ok", True)):
            if p1.get(key) != want:
                problems.append(f"{key}: {p1.get(key)!r} против {want!r}")
        if not p1.get("selected_tag") or not p1.get("selected_level_sha1"):
            problems.append("нет выбранной точки фазы 1")
        if problems:
            raise SystemExit("чекпойнт h1p1 не подходит: "
                             + "; ".join(problems))
        ref.detach_hooks()
        ref = dr.DepthRefiner.from_export(
            p1["refiner"], norm_src=model.action_expert.norm,
            layers=model.action_expert.layers, device=dev)
        n_hooks = ref.attach(model)
        lv1_names = {n for n, _ in ref.phase_named_parameters("h1p1")}
        if dr.state_sha(ref, lv1_names) != p1["selected_level_sha1"]:
            raise SystemExit("уровень 1 загрузился не тем")
        phase1_prov = dict(file=a.phase1, run_id=p1.get("run_id"),
                           file_sha1=ctx.k11a.file_sha1(a.phase1),
                           selected=p1["selected_tag"],
                           level_sha1=p1["selected_level_sha1"],
                           admission=p1.get("admission"))
        print(f"  фаза 1: {a.phase1}, точка {p1['selected_tag']}, "
              f"уровень {p1['selected_level_sha1']}, допуск "
              f"{(p1.get('admission') or {}).get('admissible')}")
    train_names = ref.set_phase(a.phase)
    named = dict(ref.named_parameters())
    params = [named[n] for n in train_names]
    print(f"  фаза {a.phase} ({variant}): {len(train_names)} тензоров, "
          f"{ref.count(a.phase) / 1e6:.3f} млн параметров; хуков LoRA "
          f"{n_hooks}; уровни оценки {names_eval}, отбор по {lv}")

    # --- данные ------------------------------------------------------------
    # Берутся только train и val_sel; val_confirm в parts_full есть, но
    # здесь не читается ни в каком режиме.
    train_all = list(ctx.parts_full["train"])
    val_all = list(ctx.parts_full["val_sel"])
    if a.mode == "overfit":
        train_b = [train_all[i] for i in strided(len(train_all),
                                                 OVERFIT["batches"])]
        val_b = list(train_b)
    elif a.mode == "smoke":
        train_b = [train_all[i] for i in strided(len(train_all),
                                                 SMOKE["train_batches"])]
        val_b = [val_all[i] for i in strided(len(val_all),
                                             SMOKE["val_batches"])]
    else:
        train_b = train_all
        val_b = val_all if not a.val_batches else [
            val_all[i] for i in strided(len(val_all), a.val_batches)]
    horizon = int(stats["horizon"])
    q0_dev = torch.as_tensor(np.asarray(ctx.q0_can), device=dev)
    ac16 = torch.autocast(device_type=dev.type, dtype=ctx.dt)
    tasks_all = np.asarray(ctx.tsk)
    epi = getattr(ctx, "epi", None)
    epi = None if epi is None else np.asarray(epi)
    act_p99 = np.asarray(ctx.act_p99_dataset, np.float64)

    def target(sel):
        x = np.asarray(ctx.ACT[sel], np.float32)[:, :horizon, :7]
        return torch.from_numpy(x).to(dev)

    def forward(po, sel, **kw):
        b = ctx.build_batch(po, sel)
        with ac16:
            v, p_ids = model.build_inputs(position_offset=po, **b)
            o = ref.run(model, vlm_inputs_embeds=v,
                        attention_mask=b.get("attention_mask"),
                        position_ids=p_ids, decode=ctx.decode_fp32, **kw)
        bad = int((o["q0"] != q0_dev[torch.as_tensor(sel, device=dev)])
                  .sum())
        if bad:
            raise SystemExit(f"q0 разошёлся с каноническим в {bad} позициях")
        return o

    def row_err(a_, t_):
        r, _m = k15t.weighted_row_error(a_, t_, ctx.weights_gate, torch)
        return r

    def evaluate(batches, *, diagnostics=False, keep_rows=False):
        """Метрики уровней на батчах. Диагностика — только по выбранной."""
        keys = ["a0"] + names_eval
        acc = {k: dict(sum=0.0, arm=0.0, grip=0.0, n=0, gacc=0.0,
                       gn=0) for k in keys}
        per_task, rows_out = {}, {k: [] for k in keys}
        vals = {k: [] for k in names_eval}
        delta_abs = {k: np.zeros(7) for k in names_eval}
        better = {k: 0 for k in names_eval}
        worse = {k: 0 for k in names_eval}
        better_prev = {k: 0 for k in names_eval}
        finite = {k: True for k in names_eval}
        diag = {}
        prev_plans = []
        n_rows = 0
        w = ctx.weights_gate
        with torch.no_grad():
            for po, sel in batches:
                o = forward(po, sel, stop_after=names_eval[-1])
                t_ = target(sel)
                acts = {"a0": o["a0"]}
                acts.update({k: o["actions"][k] for k in names_eval})
                errs = {}
                for k in keys:
                    a_ = acts[k]
                    d = (a_[:, :H_EXEC, :7] - t_[:, :H_EXEC, :7]) * w
                    r = (d ** 2).mean(dim=(1, 2))
                    errs[k] = r
                    acc[k]["sum"] += float(r.sum())
                    acc[k]["arm"] += float((d[..., :6] ** 2).mean(
                        dim=(1, 2)).sum())
                    acc[k]["grip"] += float((d[..., 6] ** 2).mean(
                        dim=1).sum())
                    acc[k]["n"] += len(sel)
                    gs = (a_[:, :H_EXEC, 6] > 0) == (t_[:, :H_EXEC, 6] > 0)
                    acc[k]["gacc"] += float(gs.float().sum())
                    acc[k]["gn"] += int(gs.numel())
                    if keep_rows:
                        rows_out[k].append(r.cpu().numpy())
                prev = o["a0"]
                for k in names_eval:
                    a_ = acts[k]
                    finite[k] &= bool(torch.isfinite(a_).all())
                    vals[k].append(a_[:, :H_EXEC].abs().reshape(-1, 7)
                                   .cpu().numpy())
                    delta_abs[k] += (a_[:, :H_EXEC] - prev[:, :H_EXEC]
                                     ).abs().reshape(-1, 7).sum(0).cpu() \
                        .numpy()
                    better[k] += int((errs[k] < errs["a0"]).sum())
                    worse[k] += int((errs[k] > errs["a0"]).sum())
                    pk = "a0" if k == names_eval[0] else names_eval[
                        names_eval.index(k) - 1]
                    better_prev[k] += int((errs[k] < errs[pk]).sum())
                    prev = a_
                tk = tasks_all[np.asarray(sel, np.int64)]
                for j, t in enumerate(tk.tolist()):
                    d_ = per_task.setdefault(str(t), {k: 0.0 for k in keys}
                                             | {"n": 0})
                    for k in keys:
                        d_[k] += float(errs[k][j])
                    d_["n"] += 1
                # предыдущий план уровня отбора — для «чужого» входа φ
                plan_in = o["a0"] if lv == names_eval[0] else \
                    o["actions"][names_eval[names_eval.index(lv) - 1]]
                prev_plans.append(plan_in.cpu())
                n_rows += len(sel)
            if diagnostics:
                nb = len(batches)
                same_ep = []
                for mode in ("zero", "far_batch") + (
                        ("ground_truth_prev",) if a.phase == "h1p2"
                        else ()):
                    s_, n_ = 0.0, 0
                    for i, (po, sel) in enumerate(batches):
                        kw = {}
                        if mode == "zero":
                            kw["fb_mode"] = {lv: "zero"}
                        elif mode == "far_batch":
                            # План батча через половину списка. Другой
                            # эпизод не гарантирован — доля строк с тем же
                            # эпизодом измеряется и пишется в отчёт.
                            j = (i + nb // 2) % nb
                            other = prev_plans[j]
                            o_sel = np.asarray(batches[j][1], np.int64)
                            idx = np.arange(len(sel)) % other.shape[0]
                            other = other[torch.as_tensor(idx)]
                            if epi is not None:
                                same_ep.append(epi[np.asarray(sel, np.int64)]
                                               == epi[o_sel[idx]])
                            kw["fb_mode"] = {lv: other.to(dev)}
                        else:
                            kw["replace_prev"] = {"2": target(sel)}
                        o = forward(po, sel, stop_after=lv, **kw)
                        r = row_err(o["actions"][lv], target(sel))
                        s_ += float(r.sum())
                        n_ += len(sel)
                    diag[mode] = float(np.sqrt(s_ / max(n_, 1)))
                diag["far_batch_same_episode_share"] = (
                    float(np.concatenate(same_ep).mean()) if same_ep
                    else None)
        n = max(n_rows, 1)
        rms = {k: float(np.sqrt(acc[k]["sum"] / n)) for k in keys}
        res = dict(
            rows=n_rows, rms=rms,
            rms_arm={k: float(np.sqrt(acc[k]["arm"] / n)) for k in keys},
            rms_grip={k: float(np.sqrt(acc[k]["grip"] / n)) for k in keys},
            rel_change={k: rms[k] / max(rms["a0"], 1e-12) - 1.0
                        for k in names_eval},
            grip_acc={k: acc[k]["gacc"] / max(acc[k]["gn"], 1)
                      for k in keys},
            better_share={k: better[k] / n for k in names_eval},
            worse_share={k: worse[k] / n for k in names_eval},
            better_than_prev_share={k: better_prev[k] / n
                                    for k in names_eval},
            mean_abs_delta={k: (delta_abs[k] / (n * H_EXEC)).tolist()
                            for k in names_eval},
            p99={k: np.percentile(np.concatenate(vals[k]), 99.0, axis=0)
                 .tolist() for k in names_eval},
            absmax={k: np.concatenate(vals[k]).max(0).tolist()
                    for k in names_eval},
            finite=finite,
            per_task={t: {k: float(np.sqrt(d_[k] / d_["n"]))
                          for k in keys} | {"n": d_["n"]}
                      for t, d_ in sorted(per_task.items())})
        if diagnostics:
            res["diagnostics"] = dict(
                level=lv, normal=rms[lv], **diag,
                note="zero/far_batch — вход φ уровня (обнулён / план "
                     "далёкого батча); ground_truth_prev — демонстрация "
                     "вместо a1 целиком (база, φ, признаки головы). Для "
                     "отбора не используются")
        if keep_rows:
            res["_rows"] = {k: np.concatenate(v) for k, v in
                            rows_out.items()}
        return res

    def print_metrics(tag, m):
        s = f"  [{tag}] val RMS a0 {m['rms']['a0']:.6f}"
        for k in names_eval:
            s += (f" | {k} {m['rms'][k]:.6f} ({100 * m['rel_change'][k]:+.2f}"
                  f"%, лучше {100 * m['better_share'][k]:.1f}% строк)")
        print(s)

    def phase_state():
        return {n: named[n].detach().cpu().clone() for n in train_names}

    def load_phase_state(st):
        with torch.no_grad():
            for n in train_names:
                named[n].copy_(st[n].to(named[n].device, named[n].dtype))

    def group_of(n):
        return n.split(".", 1)[0]

    # ПОВТОРНЫЙ ЗАПУСК АРХИВИРУЕТ ВСЕ ФАЙЛЫ ПРОГОНА СОГЛАСОВАННО. Иначе после
    # падения рядом лежали бы старый завершённый отчёт и новый чекпойнт.
    # Канонический .pt появляется только в самом конце и служит маркером
    # завершения; снапшоты пишутся в .partial.
    # АРХИВИРУЕТСЯ ПОСЛЕ ВСЕХ ПРОВЕРОК ПРЕДУСЛОВИЙ: отказавший запуск (чужой
    # сид, негодная фаза 1) не должен убирать прежний исправный результат.
    for f_ in (out, report, rows_npz, partial, validating):
        if os.path.exists(f_):
            os.replace(f_, f"{f_}.{run_id}.bak")
            print(f"  прежний {f_} -> .{run_id}.bak")

    # --- нулевая точка: уровень обязан совпасть со своим входом -------------
    technical = {}
    po0, sel0 = (val_b or train_b)[0]
    with torch.no_grad():
        o = forward(po0, sel0, stop_after=lv)
    prev_in = o["a0"] if lv == names_eval[0] else o["actions"]["1"]
    technical["start_identity"] = bool(torch.equal(o["actions"][lv],
                                                   prev_in))
    print(f"  нулевая точка: {lv} == вход уровня побитово: "
          f"{technical['start_identity']}")
    if not technical["start_identity"]:
        raise SystemExit("в нулевой точке уровень не равен входу")
    init_state = phase_state()

    lr = float(a.overfit_lr if a.mode == "overfit" else a.lr)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(a.wd))
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    if in_opt != {id(p) for p in params}:
        raise SystemExit("оптимизатор не совпал с белым списком")
    total = OVERFIT["steps"] if a.mode == "overfit" else len(train_b)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(
        0 if a.mode == "overfit" else int(a.warmup), total,
        1.0 if a.mode == "overfit" else float(a.min_lr_frac)))
    try:
        scaler = torch.amp.GradScaler(dev.type,
                                      enabled=(dev.type == "cuda"))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda"))
    rng = np.random.default_rng(int(a.seed))
    if a.mode == "overfit":
        order = [i % len(train_b) for i in range(total)]
    else:
        order = rng.permutation(len(train_b)).tolist()
    snaps = [] if a.mode == "overfit" else snap_points(
        total, SMOKE["snap_every"] if a.mode == "smoke" else a.snap_every,
        10 if a.mode == "smoke" else SNAP_END_MARGIN)
    print(f"  шагов {total}, lr {lr}, снапшоты {snaps}")

    history, states, losses = [], {}, []
    grad_seen = {g: 0.0 for g in GROUPS}
    last_grad = {g: 0.0 for g in GROUPS}
    skipped = 0
    run_stats = dict(loss=0.0, arm=0.0, grip=0.0, r_lv=0.0, r0=0.0, n=0)
    t0 = time.time()
    # Время оценок на val вычитается из скорости шага: иначе в smoke две
    # оценки на 100 шагов раздували «с/шаг» в 1.6 раза.
    eval_time = [0.0]

    def snapshot(step):
        tag = f"step{step:06d}"
        t_e = time.time()
        m = evaluate(val_b)
        print_metrics(tag, m)
        eval_time[0] += time.time() - t_e
        print(f"    оценка {time.time() - t_e:.0f} с")
        states[tag] = phase_state()
        history.append(dict(tag=tag, step=step, metrics=m,
                            state_sha1=dr.state_sha(ref, set(train_names))))
        if step == 0 and a.mode == "full":
            # СРАЗУ, а не после эпохи: другая метрика сделала бы все три
            # часа обучения несравнимыми с K-15b/c.
            rel0 = abs(m["rms"]["a0"] - REF_Q0_VAL_RMS) / REF_Q0_VAL_RMS
            if rel0 > REF_Q0_REL_TOL:
                raise SystemExit(
                    f"q0 на val_sel {m['rms']['a0']!r} против K-15b/c "
                    f"{REF_Q0_VAL_RMS} (отн. {rel0:.1e}): метрика не та")
        if step == 0:
            technical["step0_equals_q0"] = bool(
                all(abs(m["rms"][k] - m["rms"]["a0"]) == 0.0
                    for k in names_eval) if a.phase != "h1p2" else
                m["rms"]["2"] == m["rms"]["1"])
        save_checkpoint(partial, "partial")

    def save_checkpoint(path, status, selected=None, adm=None, extra=None):
        """status: partial (снапшоты), validating (до проверки загрузки),
        complete (канонический файл, пишется последним)."""
        assert status in ("partial", "validating", "complete")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        obj = dict(
            kind=KIND, phase=a.phase, variant=variant, mode=a.mode,
            seed=int(a.seed), status=status, run_id=run_id,
            final=(status == "complete"),
            refiner=ref.export(), states=states,
            history=[dict(h, metrics={k: v for k, v in h["metrics"].items()
                                      if not k.startswith("_")})
                     for h in history],
            selected_tag=selected, stats_sha1=stats["sha1"],
            frozen_sha1=frozen0, gate_run_id=gate["run_id"],
            train_names=train_names, phase1=phase1_prov,
            technical_ok=(extra or {}).get("technical_ok"),
            admission=adm)
        if selected is not None and a.phase in ("h1p1", "h1p2", "d0"):
            obj["selected_level_sha1"] = dr.state_sha(ref, set(train_names))
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    # --- обучение ----------------------------------------------------------
    report_every = int(a.report_every)
    if a.mode != "full":
        report_every = max(5, min(report_every, total // 10))
    for step in range(total + 1):
        if step in snaps:
            snapshot(step)
        if step == total:
            break
        po, sel = train_b[order[step]]
        o = forward(po, sel, train_level=lv)
        t_ = target(sel)
        loss, parts = dr.level_loss(o["actions"][lv], o["logits"][lv], t_,
                                    ref.loss_scale, H_EXEC)
        if not bool(torch.isfinite(loss)):
            raise SystemExit(f"потеря не конечна на шаге {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = {g: 0.0 for g in GROUPS}
        for n in train_names:
            gr = named[n].grad
            if gr is not None:
                gnorm[group_of(n)] += float(gr.float().pow(2).sum())
        gn = torch.nn.utils.clip_grad_norm_(params, float(a.clip))
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() < scale_before:
            skipped += 1
        opt.zero_grad(set_to_none=True)
        sched.step()
        for g in GROUPS:
            last_grad[g] = math.sqrt(gnorm[g])
            # На шаге, пропущенном GradScaler, норма бесконечна; считать её
            # «градиент дошёл» значило бы пройти проверку на мусоре.
            if math.isfinite(last_grad[g]):
                grad_seen[g] = max(grad_seen[g], last_grad[g])
        losses.append(float(loss))
        with torch.no_grad():
            run_stats["loss"] += float(loss)
            run_stats["arm"] += float(parts["arm"])
            run_stats["grip"] += float(parts["grip"])
            run_stats["r_lv"] += float(row_err(o["actions"][lv].detach(),
                                               t_).mean())
            run_stats["r0"] += float(row_err(o["a0"], t_).mean())
            run_stats["n"] += 1
        if (step + 1) % report_every == 0 or step + 1 == total:
            k_ = max(run_stats["n"], 1)
            el = time.time() - t0 - eval_time[0]
            n_left_snaps = sum(1 for s_ in snaps if s_ > step)
            per_eval = eval_time[0] / max(sum(1 for s_ in snaps
                                              if s_ <= step), 1)
            eta = el / (step + 1) * (total - step - 1) \
                + per_eval * n_left_snaps
            print(f"  шаг {step + 1}/{total}: потеря "
                  f"{run_stats['loss'] / k_:.4f} (рука "
                  f"{run_stats['arm'] / k_:.4f}, схват "
                  f"{run_stats['grip'] / k_:.4f}); батч RMS {lv}/a0 "
                  f"{math.sqrt(run_stats['r_lv'] / k_):.4f}/"
                  f"{math.sqrt(run_stats['r0'] / k_):.4f}; |g| "
                  + ", ".join(f"{g} {last_grad[g]:.2e}" for g in GROUPS)
                  + f"; clip {float(gn):.2e}; lr {sched.get_last_lr()[0]:.2e}"
                  f"; scale {scaler.get_scale():.0f}, пропусков {skipped}; "
                  f"{el / (step + 1):.2f} с/шаг без оценок (оценки "
                  f"{eval_time[0] / 60:.0f} мин), осталось "
                  f"~{eta / 60:.0f} мин",
                  flush=True)
            run_stats = {k: 0.0 for k in run_stats}
            run_stats["n"] = 0

    # --- итог --------------------------------------------------------------
    rep = dict(kind=KIND + "_report", phase=a.phase, variant=variant,
               mode=a.mode, seed=int(a.seed), out=out,
               args={k: (v if isinstance(v, (int, float, str, bool,
                                              type(None))) else str(v))
                     for k, v in vars(a).items()},
               gate_run_id=gate["run_id"], k15d_gate_code=gate["code"],
               k15a_gate=dict(ctx.gate_info), stats=stats,
               frozen_sha1=frozen0, frozen_tensors=n_frozen,
               git_head=ctx.git_head, dirty=bool(ctx.dirty),
               plan_sha1=ctx.q0_prov["plan_sha1"], hp=dict(dr.HP),
               admit_rules=dict(ADMIT), phase1=phase1_prov,
               params=ref.count(a.phase), params_all=ref.count(),
               n_hooks=n_hooks, steps=total, skipped_steps=skipped,
               grad_max_by_group=grad_seen, grad_last_by_group=last_grad,
               train_batches=len(train_b), val_batches=len(val_b))
    code = 0
    if a.mode == "overfit":
        first = float(np.mean(losses[:5]))
        last = float(np.mean(losses[-20:]))
        with torch.no_grad():
            moved = 0.0
            for po, sel in train_b:
                o = forward(po, sel, stop_after=lv)
                base = o["a0"] if lv == names_eval[0] else o["actions"]["1"]
                moved = max(moved, float((o["actions"][lv] - base).abs()
                                         .max()))
        m_end = evaluate(train_b)
        load_phase_state(init_state)
        with torch.no_grad():
            o = forward(po0, sel0, stop_after=lv)
        base = o["a0"] if lv == names_eval[0] else o["actions"]["1"]
        restored = bool(torch.equal(o["actions"][lv], base))
        frozen1, _n, _e = k15t.frozen_content_sha(model, torch, set())
        technical.update(
            loss_drop=bool(last <= OVERFIT["min_drop"] * first),
            moved=bool(moved > OVERFIT["min_move"]),
            grads_reach_all_groups=bool(all(grad_seen[g] > 0
                                            for g in GROUPS)),
            restored_identity=restored,
            frozen_unchanged=bool(frozen1 == frozen0))
        rep.update(overfit=dict(first_loss=first, last_loss=last,
                                ratio=last / max(first, 1e-12),
                                max_move=moved, rows=sum(len(s) for _p, s
                                                         in train_b),
                                rms_end=m_end["rms"],
                                rms_change=m_end["rel_change"]))
        print(f"  OVERFIT: потеря {first:.4f} -> {last:.4f} "
              f"({last / max(first, 1e-12):.2f}), сдвиг от входа "
              f"{moved:.2e}, RMS на этих строках {m_end['rms']}")
        adm = None
    else:
        sel_tag = select_snapshot(history, lv)
        load_phase_state(states[sel_tag])
        sel_sha = dr.state_sha(ref, set(train_names))
        want_sha = [h for h in history if h["tag"] == sel_tag][0][
            "state_sha1"]
        if sel_sha != want_sha:
            raise SystemExit("восстановленная точка не та")
        print(f"  выбрана точка {sel_tag} по RMS {lv}")
        t_d = time.time()
        final = evaluate(val_b, diagnostics=True, keep_rows=True)
        print_metrics(f"{sel_tag}, повтор", final)
        dg = final["diagnostics"]
        print(f"  причинность ({lv}): обычный {dg['normal']:.6f}, обнулённый"
              f" φ {dg['zero']:.6f}, план далёкого батча "
              f"{dg['far_batch']:.6f} (тот же эпизод у "
              f"{dg['far_batch_same_episode_share']} строк)"
              + (f", демонстрация вместо a1 {dg['ground_truth_prev']:.6f}"
                 if "ground_truth_prev" in dg else "")
              + f" ({time.time() - t_d:.0f} с)")
        rec = [h for h in history if h["tag"] == sel_tag][0]["metrics"]
        rel = abs(final["rms"][lv] - rec["rms"][lv]) / max(rec["rms"][lv],
                                                           1e-12)
        technical["selected_reproduces"] = bool(rel <= 1e-4)
        technical["finite"] = bool(all(final["finite"].values()))
        technical["grads_reach_all_groups"] = bool(all(
            grad_seen[g] > 0 for g in GROUPS))
        frozen1, _n, _e = k15t.frozen_content_sha(model, torch, set())
        technical["frozen_unchanged"] = bool(frozen1 == frozen0)
        if a.mode == "full":
            rel0 = abs(final["rms"]["a0"] - REF_Q0_VAL_RMS) / REF_Q0_VAL_RMS
            technical["q0_val_rms_matches_k15"] = bool(rel0 <= REF_Q0_REL_TOL)
            print(f"  q0 на val_sel {final['rms']['a0']:.6f} против "
                  f"K-15b/c {REF_Q0_VAL_RMS} (отн. {rel0:.1e})")
        if a.mode == "smoke":
            technical["range_smoke"] = bool(all(
                max(final["absmax"][k]) <= ADMIT["clip_bound"]
                for k in names_eval))
        adm = admission(final, lv, act_p99, smoke=(a.mode != "full"))
        os.makedirs(os.path.dirname(os.path.abspath(rows_npz)),
                    exist_ok=True)
        val_rows = np.concatenate([np.asarray(s, np.int64)
                                   for _p, s in val_b])
        np.savez(rows_npz, rows=val_rows,
                 tasks=tasks_all[val_rows].astype(str),
                 **{f"err_{k}": v for k, v in final.pop("_rows").items()})
        rep.update(history=[dict(tag=h["tag"], step=h["step"],
                                 metrics=h["metrics"],
                                 state_sha1=h["state_sha1"])
                            for h in history],
                   selected_tag=sel_tag, selected_state_sha1=sel_sha,
                   final=final, admission=adm, rows_file=rows_npz)
        # сохранение и загрузка: чекпойнт с диска воспроизводит выход
        # Проверочный файл — под отдельным именем со статусом validating:
        # канонический появится только после всех проверок.
        save_checkpoint(validating, "validating", selected=sel_tag, adm=adm,
                        extra=dict(technical_ok=None))
        obj = torch.load(validating, map_location="cpu", weights_only=False)
        with torch.no_grad():
            o_mem = forward(po0, sel0, stop_after=lv)
        ref.detach_hooks()
        ref2 = dr.DepthRefiner.from_export(
            obj["refiner"], norm_src=model.action_expert.norm,
            layers=model.action_expert.layers, device=dev)
        ref2.attach(model)
        ref = ref2
        with torch.no_grad():
            o_disk = forward(po0, sel0, stop_after=lv)
        technical["save_load"] = bool(all(
            torch.equal(o_mem["actions"][k], o_disk["actions"][k])
            for k in o_mem["actions"]))
        ref2.detach_hooks()
        print("  допуск к роллауту (" + adm["note"] + "): " + ", ".join(
            f"{k} {'OK' if g['passed'] else 'НЕТ'}"
            for k, g in adm["gates"].items())
            + f" -> {'ДОПУЩЕН' if adm['admissible'] else 'не допущен'}")
    technical_ok = bool(all(technical.values()))
    rep.update(technical=technical, technical_ok=technical_ok,
               seconds=round(time.time() - t_start, 1),
               finished=datetime.datetime.now().isoformat(
                   timespec="seconds"))
    if not technical_ok:
        code = 3
    elif a.mode == "full" and not adm["admissible"]:
        code = 4
    rep["exit_code"] = code
    rep["run_id"] = run_id
    os.makedirs(os.path.dirname(os.path.abspath(report)), exist_ok=True)
    tmp = report + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False,
                  default=k15t.json_scalar)
    os.replace(tmp, report)
    # КАНОНИЧЕСКИЙ ЧЕКПОЙНТ — ПОСЛЕДНИМ: его наличие и есть маркер
    # завершённого прогона, отчёт с тем же run_id уже лежит рядом.
    if a.mode == "overfit":
        save_checkpoint(out, "complete",
                        extra=dict(technical_ok=technical_ok))
    else:
        save_checkpoint(out, "complete", selected=rep["selected_tag"],
                        adm=adm, extra=dict(technical_ok=technical_ok))
    for f_ in (partial, validating):
        if os.path.exists(f_):
            os.remove(f_)
    bad = [k for k, v in technical.items() if not v]
    print(f"ИТОГ {a.phase}/{a.mode}: технически "
          f"{'исправно' if technical_ok else 'ОТКАЗ ' + str(bad)}; код "
          f"{code}; {report}")
    return code


def selftest():
    # снапшоты: 0, кратные, конец; близкий к концу выбрасывается
    assert snap_points(15138, 5000, 1000) == [0, 5000, 10000, 15138]
    assert snap_points(100, 50, 10) == [0, 50, 100]
    assert snap_points(10200, 5000, 1000) == [0, 5000, 10200]
    f = lr_lambda(10, 110, 0.1)
    assert abs(f(0) - 0.1) < 1e-9 and abs(f(9) - 1.0) < 1e-9
    assert abs(f(10) - 1.0) < 1e-9 and abs(f(110) - 0.1) < 1e-9
    assert eval_levels("h1p2") == (["1", "2"], "2")
    assert eval_levels("d0") == (["d"], "d")
    # отбор: минимум, ранняя при равенстве, NaN пропускается
    hist = [dict(tag=t, metrics=dict(rms={"d": v})) for t, v in
            (("s0", 0.2), ("s1", 0.1), ("s2", 0.1), ("s3", float("nan")))]
    assert select_snapshot(hist, "d") == "s1"
    # допуск: каждый гейт может провалить решение отдельно
    base = dict(rms={"a0": 0.10, "d": 0.104}, finite={"d": True},
                per_task={"t1": {"a0": 0.1, "d": 0.105, "n": 5}},
                p99={"d": [0.5] * 7}, absmax={"d": [0.9] * 7},
                grip_acc={"a0": 0.95, "d": 0.94})
    p99 = [0.5] * 7
    assert admission(base, "d", p99, smoke=False)["admissible"]
    assert not admission(base, "d", p99, smoke=True)["admissible"]
    import copy
    for mut in (lambda m: m["rms"].update(d=0.106),
                lambda m: m["finite"].update(d=False),
                lambda m: m["per_task"]["t1"].update(d=0.111),
                lambda m: m["p99"].update(d=[0.8] * 7),
                lambda m: m["absmax"].update(d=[1.6] * 7),
                lambda m: m["grip_acc"].update(d=0.92)):
        m = copy.deepcopy(base)
        mut(m)
        assert not admission(m, "d", p99, smoke=False)["admissible"]
    # тренер обязан сверять гейт по коду обоих файлов K-15d
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    for needed in ("check_gate_report", "frozen_content_sha",
                   "REF_Q0_VAL_RMS", "selected_level_sha1"):
        assert needed in src, needed
    assert "val_confirm" not in [p for p in dr.PHASES]
    print("самопроверка k15d_train пройдена")
    return 0


def _fake_ctx(n_rows=200, seed=0):
    """Игрушечная среда с интерфейсом k15_context.build для интеграции."""
    import types
    import torch
    torch.manual_seed(seed)
    m = dr._FakeModel(seed=seed)
    S, d = 7, 16
    g = torch.Generator().manual_seed(seed + 1)
    X = torch.randn(n_rows, S, d, generator=g)

    def prefix_q0(v):
        act = m.bos_embedding.expand(v.shape[0], m.block_size, -1)
        act = torch.cat([act, act[:, :0]], 1)
        vlm = v
        for i in range(dr.Q0_DEPTH):
            vlm, act = m._shared_attention_forward(
                vlm_hidden_states=vlm, action_hidden_states=act,
                layer_idx=i, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, cache_position=None)
        return m._joint_depth_logits(act, 0).argmax(-1)

    m.build_inputs = lambda position_offset, v, attention_mask: (v, None)
    m.forward_depth_aligned_rvq = lambda **kw: dict(
        pred_codes=[prefix_q0(kw["vlm_inputs_embeds"])])
    decode = dr._fake_decoder(m.book0.shape[1], 20, m.block_size)
    # Канонический q0 снимается в том же autocast, что проход (как в K-14d).
    with torch.no_grad(), torch.autocast(device_type="cpu",
                                         dtype=torch.bfloat16):
        q0 = prefix_q0(X).numpy()
        a0 = decode(m.book0[torch.as_tensor(q0)]).numpy()
    rng = np.random.default_rng(seed)
    act = np.clip(a0 + rng.normal(0, 0.15, a0.shape), -1, 1)
    act[..., 6] = np.where(rng.random(a0.shape[:2]) < 0.8,
                           np.sign(a0[..., 6] + 1e-9), -np.sign(a0[..., 6]))
    rows = np.arange(n_rows)
    tr = [(0, rows[i:i + 8]) for i in range(0, 160, 8)]
    va = [(0, rows[i:i + 8]) for i in range(160, n_rows, 8)]
    tasks = np.array([f"task{i % 3}" for i in range(n_rows)])
    epi = rows // 10

    def build_batch(po, sel):
        return dict(v=X[np.asarray(sel)], attention_mask=torch.ones(
            len(sel), S))

    ns = types.SimpleNamespace(
        torch=torch, model=m, dev=torch.device("cpu"), dt=torch.bfloat16,
        q0_can=q0, parts_full=dict(train=tr, val_sel=va,
                                   val_confirm=[(0, rows[:8])]),
        ACT=act.astype(np.float32), act_p99_dataset=np.percentile(
            np.abs(act[:, :8]).reshape(-1, 7), 99, axis=0),
        decode_fp32=decode, tsk=tasks, epi=epi,
        weights_gate=torch.ones(7),
        build_batch=build_batch, gate_info=dict(init_gate="fake"),
        git_head="fake", dirty=False, q0_prov=dict(plan_sha1="fakeplan"),
        code_version=dict(fake=1), joint_sha="fakejoint",
        codec_fp=dict(fake=1),
        k11a=types.SimpleNamespace(file_sha1=k15_context.sha12),
        proc=None, codec=None)
    return ns


def integration():
    """Реальные main() гейта и тренера на игрушечной среде, CPU."""
    import tempfile
    import torch
    global REF_Q0_VAL_RMS
    saved = (k15_context.build, REF_Q0_VAL_RMS, sys.argv)
    ctx = _fake_ctx()
    k15_context.build = lambda a: ctx
    codes = {}
    try:
        with tempfile.TemporaryDirectory() as td:
            gate = os.path.join(td, "gate.json")
            sys.argv = ["x", "--out", gate, "--device", "cpu",
                        "--gate-batches", "2"]
            codes["gate"] = kg.main()
            # q0 игрушки другой; сверка с K-15 подменяется ЕЁ ЖЕ значением
            m0 = None

            def run(phase, mode, extra=()):
                sys.argv = ["x", "--phase", phase, "--mode", mode,
                            "--device", "cpu", "--k15d-gate", gate,
                            "--out", os.path.join(td, f"{phase}_{mode}.pt"),
                            "--report", os.path.join(
                                td, f"{phase}_{mode}.json"),
                            "--report-every", "5"] + list(extra)
                return main()
            codes["overfit_d0"] = run("d0", "overfit")
            codes["overfit_h1p1"] = run("h1p1", "overfit")
            codes["smoke_d0"] = run("d0", "smoke")
            codes["smoke_h1p1"] = run("h1p1", "smoke")
            codes["smoke_h1p2"] = run("h1p2", "smoke", [
                "--phase1", os.path.join(td, "h1p1_smoke.pt"),
                "--allow-smoke-phase1"])
            codes["overfit_h1p2"] = run("h1p2", "overfit", [
                "--phase1", os.path.join(td, "h1p1_smoke.pt"),
                "--allow-smoke-phase1"])
            rep = json.load(open(os.path.join(td, "d0_smoke.json")))
            m0 = rep["final"]["rms"]["a0"]
            REF_Q0_VAL_RMS = m0
            codes["full_h1p1"] = run("h1p1", "full")
            codes["full_h1p2"] = run("h1p2", "full", [
                "--phase1", os.path.join(td, "h1p1_full.pt")])
            # h1p2 от smoke-фазы 1 без разрешения обязан отказать
            try:
                run("h1p2", "full", ["--phase1",
                                     os.path.join(td, "h1p1_smoke.pt")])
                codes["h1p2_from_smoke_refused"] = 1
            except SystemExit as e:
                codes["h1p2_from_smoke_refused"] = 0 if 'режим фазы 1' in str(e) else str(e)
            # чужой сид: гейт снят с seed 0
            try:
                run("d0", "smoke", ["--seed", "1"])
                codes["seed_mismatch_refused"] = 1
            except SystemExit as e:
                codes["seed_mismatch_refused"] = 0 if 'seed' in str(e) else str(e)
            # h1p2 от незавершённой фазы 1
            p1f = os.path.join(td, "h1p1_full.pt")
            ck = torch.load(p1f, map_location="cpu", weights_only=False)
            bad = os.path.join(td, "h1p1_partial.pt")
            torch.save(dict(ck, status="partial", final=False), bad)
            try:
                run("h1p2", "smoke", ["--phase1", bad,
                                      "--allow-smoke-phase1"])
                codes["partial_phase1_refused"] = 1
            except SystemExit as e:
                codes["partial_phase1_refused"] = 0 if 'status' in str(e) else str(e)
            # повтор архивирует прежние файлы прогона
            codes["rerun_d0"] = run("d0", "smoke")
            baks = [f for f in os.listdir(td) if f.startswith("d0_smoke")
                    and f.endswith(".bak")]
            codes["rerun_archived"] = 0 if len(baks) >= 3 else 1
            rep2 = json.load(open(os.path.join(td, "h1p2_full.json")))
            # --- проверка вывода и рука на чекпойнте h1p2 -----------------
            import k15d_check_inference as ki
            import k15d_policy as kp
            saved_gp = kp.gate_paths
            kp.gate_paths = lambda device: ("unused", gate)
            try:
                ck_path = os.path.join(td, "h1p2_full.pt")
                inf_rep = os.path.join(td, "inference.json")
                sys.argv = ["x", "--checkpoint", ck_path, "--device", "cpu",
                            "--train-report",
                            os.path.join(td, "h1p2_full.json"),
                            "--out", inf_rep]
                codes["inference"] = ki.main()
                ir = json.load(open(inf_rep))
                assert ir["verdict"]["passed"], ir["verdict"]
                assert ir["rms_rel"] <= ki.RMS_REL
                # рука собирается с этим отчётом и исполняет чанк
                arm = kp.build_arm("cpu", ck_path, inf_rep, torch,
                                   init_gate=gate, k15d_gate=gate)
                po, sel = ctx.parts_full["val_sel"][0]
                ac = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
                a_np, q0c = arm.act(ctx.build_batch(po, sel), po, ac, True)
                assert a_np.shape == (len(sel), 20, 7)
                assert np.array_equal(q0c, ctx.q0_can[np.asarray(sel)])
                summ, arrs = arm.log.take()
                assert summ["calls"] == 1 and "k15d_level_2" in arrs
                # отчёт другого чекпойнта рука не принимает
                bad_rep = os.path.join(td, "inference_bad.json")
                json.dump(dict(ir, checkpoint_sha1="000000000000"),
                          open(bad_rep, "w"))
                try:
                    kp.build_arm("cpu", ck_path, bad_rep, torch,
                                 init_gate=gate, k15d_gate=gate)
                    codes["arm_bad_report_refused"] = 1
                except SystemExit as e:
                    codes["arm_bad_report_refused"] = (
                        0 if "другого файла" in str(e) else str(e))
                # --- рука h18: чекпойнт h1p1, проход до 18-го слоя ------
                p1_path = os.path.join(td, "h1p1_full.pt")
                rep18 = os.path.join(td, "inference_h18.json")
                sys.argv = ["x", "--checkpoint", p1_path, "--device", "cpu",
                            "--train-report",
                            os.path.join(td, "h1p1_full.json"),
                            "--out", rep18]
                codes["inference_h18"] = ki.main()
                arm18 = kp.build_arm("cpu", p1_path, rep18, torch,
                                     init_gate=gate, k15d_gate=gate)
                assert arm18.meta["layers_per_call"] == 18
                assert arm18.meta["levels"] == ["1"]
                arm18.act(ctx.build_batch(po, sel), po, ac, True)
                s18, a18 = arm18.log.take()
                assert set(a18) == {"k15d_a0", "k15d_level_1"}, set(a18)
                # --- исключение из фильтра допуска ------------------------
                ck2 = torch.load(ck_path, map_location="cpu",
                                 weights_only=False)
                adm = dict(ck2["admission"], admissible=False, passed=False)
                adm["gates"] = dict(adm["gates"], task_rms=dict(
                    adm["gates"]["task_rms"], passed=False))
                nad = os.path.join(td, "h1p2_notadmitted.pt")
                torch.save(dict(ck2, admission=adm), nad)
                rep_n = os.path.join(td, "inference_notadmitted.json")
                base_argv = ["x", "--checkpoint", nad, "--device", "cpu",
                             "--train-report",
                             os.path.join(td, "h1p2_full.json"),
                             "--out", rep_n]
                sys.argv = list(base_argv)
                try:
                    ki.main()
                    codes["notadmitted_refused"] = 1
                except SystemExit as e:
                    codes["notadmitted_refused"] = (
                        0 if "причин" in str(e) else str(e))
                sys.argv = base_argv + ["--allow-failed-admission",
                                        "контроль"]
                codes["override_inference"] = ki.main()
                irn = json.load(open(rep_n))
                assert irn["admission_override"]["reason"] == "контроль"
                assert "task_rms" in irn["admission_override"]["failed"]
                armn = kp.build_arm("cpu", nad, rep_n, torch, init_gate=gate,
                                    k15d_gate=gate, override="контроль")
                assert armn.meta["admission_override"]["reason"] == \
                    "контроль"
                try:
                    kp.build_arm("cpu", nad, rep_n, torch, init_gate=gate,
                                 k15d_gate=gate)
                    codes["override_arm_needs_reason"] = 1
                except SystemExit as e:
                    codes["override_arm_needs_reason"] = (
                        0 if "исключение" in str(e) else str(e))
            finally:
                kp.gate_paths = saved_gp
            dg = rep2["final"]["diagnostics"]
            assert {"normal", "zero", "far_batch",
                    "ground_truth_prev"} <= set(dg)
            assert dg["far_batch_same_episode_share"] is not None
            ck = torch.load(os.path.join(td, "h1p2_full.pt"),
                            map_location="cpu", weights_only=False)
            assert ck["status"] == "complete" and ck["final"] is True
            assert ck["run_id"] == rep2["run_id"]
            assert not os.path.exists(os.path.join(td, "h1p2_full.pt")
                                      + ".partial")
            assert not os.path.exists(os.path.join(td, "h1p2_full.pt")
                                      + ".validating")
            assert rep2["phase1"]["selected"] is not None
            assert os.path.exists(rep2["rows_file"])
    finally:
        k15_context.build, REF_Q0_VAL_RMS, sys.argv = saved
    print("  коды:", codes)
    for k in ("gate", "overfit_d0", "overfit_h1p1", "overfit_h1p2",
              "smoke_d0", "smoke_h1p1", "smoke_h1p2",
              "h1p2_from_smoke_refused", "seed_mismatch_refused",
              "partial_phase1_refused", "rerun_d0", "rerun_archived",
              "inference", "arm_bad_report_refused", "inference_h18",
              "notadmitted_refused", "override_inference",
              "override_arm_needs_reason"):
        assert codes[k] == 0, (k, codes[k])
    for k in ("full_h1p1", "full_h1p2"):
        assert codes[k] in (0, 4), (k, codes[k])
    print("интеграция k15d пройдена: гейт, overfit, smoke и full всех фаз")
    return 0


if __name__ == "__main__":
    if "--integration" in sys.argv:
        sys.exit(integration())
    sys.exit(main())
