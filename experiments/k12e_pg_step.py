#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-12e: ОДИН полнобатчевый шаг градиента политики с дроблением и откатом.

ПОЧЕМУ ОДИН ШАГ И ПОЛНЫМ БАТЧЕМ. K-11i измерил, что минибатчевый PPO здесь
непригоден при любом lr: отношение правдоподобий на сохранённых сэмплах уходит
в нуль или в бесконечность уже внутри первой эпохи. Работает ровно один
полнобатчевый шаг при lr <= 3e-6. Поэтому здесь нет ни эпох, ни минибатчей:
минибатчи служат только накоплением градиента и математически дают тот же
полный батч.

ЧТО СЧИТАЕТСЯ ГРАДИЕНТОМ

    loss = -(1/E) * sum_по_вызовам A_эпизода * log pi(u | s)

A — преимущество эпизода с базой «среднее по ОСТАЛЬНЫМ эпизодам этой задачи»
(leave-one-out). База по задаче обязательна: доли успеха задач различаются в
разы, и общая база превратила бы «задача лёгкая» в «политика хорошая».
Leave-one-out, а не обычное среднее: собственный успех в своей базе даёт
смещение к нулю тем большее, чем меньше эпизодов.

ТРУНК НЕ УЧАСТВУЕТ. Обучается только ветвь mu головы поправки; log_std
заморожена, потому что sigma — зарегистрированная константа, выбранная на dev.
Вход головы (h24 и индексы черновика) сохранён раскаткой, поэтому шаг не
требует ни VLM, ни симулятора: это единственная причина, по которой его вообще
можно отлаживать отдельно.

ГОЛОВА СЧИТАЕТСЯ В fp32. Под autocast fp16 разрешение mu крупнее самого шага
при lr 3e-6: отношение правдоподобий осталось бы единицей из-за округления, и
«шаг без изменений» был бы неотличим от «шага, которого не было».

ДРОБЛЕНИЕ И ОТКАТ. После шага на ТОМ ЖЕ буфере считаются KL (аналитически, по
сдвигу mu при фиксированной sigma) и хвост отношения правдоподобий. Если хоть
одно вне области доверия — восстанавливаются параметры И состояние Adam, шаг
делится вдвое и повторяется. Состояние оптимизатора обязательно: моменты Adam
переживают возврат параметров, и следующий шаг пошёл бы по устаревшему
направлению, то есть «откат» откатил бы не всё.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np


# ----------------------------- преимущества -------------------------------

def loo_advantage(rewards, tasks):
    """Преимущество с базой leave-one-out внутри задачи.

    Возвращает (adv, info). Задача с единственным эпизодом даёт нулевое
    преимущество: базы из «остальных» там нет, и любое другое решение было бы
    молчаливым сравнением эпизода с самим собой.
    """
    r = np.asarray(rewards, dtype=np.float64)
    t = np.asarray(tasks)
    adv = np.zeros_like(r)
    info = dict(by_task={}, singleton_tasks=[], flat_tasks=[])
    for task in sorted(set(t.tolist())):
        m = t == task
        n = int(m.sum())
        rt = r[m]
        if n < 2:
            info["singleton_tasks"].append(task)
            continue
        base = (rt.sum() - rt) / (n - 1)
        adv[m] = rt - base
        if float(rt.std()) == 0.0:
            info["flat_tasks"].append(task)
        info["by_task"][str(task)] = dict(n=n, mean_reward=float(rt.mean()),
                                          adv_abs_mean=float(np.abs(
                                              rt - base).mean()))
    return adv, info


def standardize(adv):
    """Деление на СКО без повторного центрирования.

    Центрирование уже сделано базой; повторное сдвинуло бы преимущества
    успешных эпизодов задачи, где успехов больше половины. Деление нужно,
    чтобы предел lr = 3e-6, измеренный в K-11i при преимуществах единичного
    масштаба, переносился сюда.
    """
    a = np.asarray(adv, dtype=np.float64)
    sd = float(a.std())
    if sd == 0.0:
        return a * 0.0, dict(scale=0.0, degenerate=True)
    return a / sd, dict(scale=sd, degenerate=False)


# ------------------------- измерения после шага ---------------------------

def kl_mu(mu_old, mu_new, std):
    """KL(старая || новая) при одной и той же sigma — точно, без оценок.

    Для гауссиан с равными sigma KL = ||dmu||^2 / (2 sigma^2). Оценки k1/k3
    здесь не нужны и только добавили бы дисперсию.
    """
    import torch
    d = (mu_new.double() - mu_old.double()) / std.double()
    return 0.5 * (d * d).flatten(1).sum(-1)


def ratio_stats(lp_new, lp_old, ratio_max):
    """Хвост отношения правдоподобий — в fp64 И как его увидел бы fp32.

    В fp64 считается настоящая величина, а доля нулей в fp32 — то, что
    реально произошло бы в PPO: exp(-105) в fp32 равен ровно нулю, и эпизод
    молча выпадает из градиента.
    """
    import torch
    # форма сверяется явно: при (n,) против (1,) разность ТИХО размножилась бы
    # по батчу, и хвост считался бы относительно одного сэмпла
    if tuple(lp_new.shape) != tuple(lp_old.shape):
        raise ValueError(f"формы правдоподобий {tuple(lp_new.shape)} и "
                         f"{tuple(lp_old.shape)} не совпадают")
    d = (lp_new.double() - lp_old.double())
    r = torch.exp(d)
    r32 = torch.exp(d.float())
    fin = torch.isfinite(r)
    out = dict(
        n=int(r.numel()),
        logdiff_max=float(d.max()), logdiff_min=float(d.min()),
        logdiff_absmean=float(d.abs().mean()),
        ratio_max=float(r[fin].max()) if int(fin.sum()) else float("inf"),
        ratio_min=float(r[fin].min()) if int(fin.sum()) else 0.0,
        ratio_mean=float(r[fin].mean()) if int(fin.sum()) else float("nan"),
        nonfinite=int((~fin).sum()),
        zero_frac_f32=float((r32 == 0).double().mean()),
        inf_frac_f32=float((~torch.isfinite(r32)).double().mean()))
    out["over_frac"] = float((r > ratio_max).double().mean())
    out["under_frac"] = float((r < 1.0 / ratio_max).double().mean())
    out["ok"] = bool(out["nonfinite"] == 0 and out["zero_frac_f32"] == 0.0
                     and out["inf_frac_f32"] == 0.0
                     and out["ratio_max"] <= ratio_max
                     and out["ratio_min"] >= 1.0 / ratio_max)
    return out


def accept_step(meas, trust):
    """Принять ли шаг. Все причины отказа сразу — их полезно видеть в логе."""
    why = []
    if not meas["ratio"]["ok"]:
        r = meas["ratio"]
        why.append(f"хвост отношения: max {r['ratio_max']:.4g}, min "
                   f"{r['ratio_min']:.4g}, нулей в fp32 "
                   f"{100 * r['zero_frac_f32']:.2f}%, нечисел {r['nonfinite']}")
    if meas["kl_mean"] > float(trust["kl_max"]):
        why.append(f"KL среднее {meas['kl_mean']:.5f} > "
                   f"{float(trust['kl_max']):.5f}")
    if not meas["finite_params"]:
        why.append("в параметрах после шага нет конечности")
    return (len(why) == 0), why


# --------------------- цепочка шагов и состояние Adam ----------------------

def check_resume_chain(prev, *, protocol_sha1, replica, step_index, d1_sha,
                       d1_seed, sigma, require_optimizer):
    """Сверка головы предыдущего шага. ОДНА функция на раскатку и на шаг.

    Проверяется не только протокол: реплика, сид D1, исходный чекпойнт D1,
    sigma и НОМЕР ШАГА. Без номера шага можно было бы дважды обучить от одной
    головы и посчитать это двумя шагами; без реплики — подать голову одной
    реплики другой, и четыре «независимых» прогона оказались бы одним.

    `require_optimizer` — для шага: состояние Adam обязано переезжать между
    командами, иначе каждый шаг начинается с нулевых моментов, то есть
    алгоритм не тот, который зарегистрирован.
    """
    import k12b_protocol as kb
    bad = []
    if prev.get("protocol_sha1") != protocol_sha1:
        bad.append(f"голова под протоколом {prev.get('protocol_sha1')}, а "
                   f"прогон под {protocol_sha1}")
    if prev.get("replica") != replica:
        bad.append(f"голова реплики {prev.get('replica')}, а прогон реплики "
                   f"{replica}: это смешало бы реплики, которые обязаны быть "
                   f"независимыми")
    want_prev = int(step_index) - 1
    if int(prev.get("step_index", -10 ** 9)) != want_prev:
        bad.append(f"голова с шага {prev.get('step_index')}, а ожидался "
                   f"{want_prev}: цепочка шагов разорвана, и номер шага "
                   f"перестал означать число сделанных обновлений")
    if d1_sha is not None and prev.get("d1_head_sha1") != d1_sha:
        bad.append(f"голова выросла из D1 {prev.get('d1_head_sha1')}, а подан "
                   f"D1 {d1_sha}")
    if d1_seed is not None and prev.get("d1_seed") != d1_seed:
        bad.append(f"сид D1 головы {prev.get('d1_seed')} вместо {d1_seed}")
    if sigma is not None and abs(float(prev.get("sigma", -1))
                                 - float(sigma)) > 1e-9:
        bad.append(f"голова обучалась при sigma {prev.get('sigma')}, а раскатка"
                   f" при {sigma}")
    if require_optimizer and not prev.get("optimizer_state"):
        bad.append("в голове нет состояния оптимизатора: моменты Adam начались "
                   "бы с нуля, и это другой алгоритм, а не продолжение")
    if bad:
        raise kb.ProtocolError("продолжение не сходится с протоколом:\n  - "
                               + "\n  - ".join(bad))
    return True


def load_optimizer_state(opt, state, train_params):
    """Восстановление моментов Adam с проверкой, что они действительно легли.

    Пустая проверка была бы бесполезна: load_state_dict молча принимает
    состояние с другим числом параметров, если совпадает число групп, и тогда
    моменты относились бы к другим весам.
    """
    import torch
    got = state.get("state") or {}
    if len(got) != len(train_params):
        raise ValueError(f"в состоянии Adam {len(got)} параметров, а обучаемых "
                         f"{len(train_params)}: моменты относятся к другим "
                         f"весам")
    for i, p_ in enumerate(train_params):
        sd = got.get(i) if i in got else got.get(str(i))
        if sd is None:
            raise ValueError(f"в состоянии Adam нет параметра {i}")
        for k in ("exp_avg", "exp_avg_sq"):
            if tuple(sd[k].shape) != tuple(p_.shape):
                raise ValueError(f"момент {k} параметра {i} формы "
                                 f"{tuple(sd[k].shape)} против "
                                 f"{tuple(p_.shape)}")
    opt.load_state_dict(state)
    steps = [float(v["step"]) for v in opt.state.values()]
    if not steps or min(steps) < 1:
        raise ValueError(f"счётчики шагов Adam {steps}: состояние не "
                         f"восстановилось, продолжения нет")
    return dict(n_params=len(steps), step_min=min(steps), step_max=max(steps))


# ----------------------------- буфер раскаток ------------------------------

REC_KEYS = ("h", "q0", "u", "mu", "logp", "task", "state", "call")


def check_rollouts(files, proto, *, replica, stage, sigma=None):
    """Сверка набора файлов раскатки с протоколом ДО любых вычислений.

    Отдельная функция, потому что именно здесь прежние протоколы были
    fail-open: файл лежал рядом, его происхождение никто не сверял.
    """
    import k12b_protocol as kb
    bad, seen, metas = [], {}, []
    for f in files:
        m = f["meta"]
        metas.append(m)
        tag = os.path.basename(str(m.get("path", "?")))
        if m.get("protocol_sha1") != proto["sha1"]:
            bad.append(f"{tag}: протокол {m.get('protocol_sha1')} вместо "
                       f"{proto['sha1']}")
        if m.get("replica") != replica:
            bad.append(f"{tag}: реплика {m.get('replica')} вместо {replica}")
        if m.get("stage") != stage:
            bad.append(f"{tag}: этап {m.get('stage')} вместо {stage}")
        if not m.get("policy_sha1"):
            bad.append(f"{tag}: нет policy_sha1 — происхождение действующей "
                       f"политики не проверить")
        if m.get("head_precision") != "fp32":
            bad.append(f"{tag}: голова считалась в {m.get('head_precision')}, "
                       f"а не fp32: отношение правдоподобий нельзя было бы "
                       f"пересчитать тем же арифметическим путём")
        if sigma is not None and abs(float(m.get("sigma", -1))
                                     - float(sigma)) > 1e-9:
            bad.append(f"{tag}: sigma {m.get('sigma')} вместо {sigma}")
        for key, st in (("episodes", None),):
            if not m.get(key):
                bad.append(f"{tag}: нет {key}")
        for e in m.get("episodes") or []:
            k = (int(e["task_id"]), int(e["state_id"]))
            if k in seen:
                bad.append(f"{tag}: эпизод {k} уже есть в {seen[k]} — один "
                           f"эпизод, посчитанный дважды, удваивает его вес в "
                           f"градиенте")
            seen[k] = tag
            if e.get("success") is None:
                bad.append(f"{tag}: эпизод {k} без успеха")
            if not e.get("init_hash_full"):
                bad.append(f"{tag}: эпизод {k} без init_hash_full")
    # набор состояний и задач — против зарегистрированных
    allowed = set(proto["splits"].get(stage, []))
    leaked = sorted({s for (_t, s) in seen} - allowed)
    if leaked:
        bad.append(f"состояния {leaked[:8]} ({len(leaked)} шт.) не из набора "
                   f"'{stage}'")
    unknown = sorted({t for (t, _s) in seen} - set(proto["tasks"]))
    if unknown:
        bad.append(f"задачи {unknown} не зарегистрированы")
    # единая геометрия и единая голова
    for key, why in (("d_hidden", "разная размерность отвода"),
                     ("rank", "разный ранг"),
                     ("head_sha1", "разные исходные головы D1"),
                     # policy_sha1 — sha ДЕЙСТВУЮЩЕЙ политики, а не D1: после
                     # первого шага две разные обученные головы имеют один и
                     # тот же head_sha1, и без этого поля их нельзя различить
                     ("policy_sha1", "раскатки разными политиками"),
                     ("step_index", "раскатки с разных шагов обучения"),
                     ("codebooks_sha1", "разные кодовые книги"),
                     ("joint_sha1", "разные стволы")):
        vals = {str(m.get(key)) for m in metas}
        if len(vals) > 1:
            bad.append(f"{why}: {sorted(vals)}")
    if bad:
        raise kb.ProtocolError("раскатки не соответствуют протоколу:\n  - "
                               + "\n  - ".join(bad))
    return dict(n_files=len(files), n_episodes=len(seen),
                tasks=sorted({t for (t, _s) in seen}))


def concat_buffer(files, cb0, device):
    """Склейка файлов в один буфер в ФИКСИРОВАННОМ порядке.

    Порядок не перемешивается и хэшируется: полный батч от порядка не зависит
    математически, но его хэш — единственный способ доказать, что второй
    прогон считал тот же набор.
    """
    import torch
    order = sorted(range(len(files)),
                   key=lambda i: str(files[i]["meta"].get("path")))
    parts = {k: [] for k in REC_KEYS}
    ep_rows, h = [], hashlib.sha1()
    for i in order:
        d = files[i]["data"]
        for k in REC_KEYS:
            parts[k].append(d[k])
        for e in files[i]["meta"]["episodes"]:
            ep_rows.append(dict(e))
        h.update(str(files[i]["meta"].get("path")).encode())
    buf = {k: torch.cat(parts[k]) for k in REC_KEYS}
    n = buf["h"].shape[0]
    for k in REC_KEYS:
        if buf[k].shape[0] != n:
            raise ValueError(f"поле {k} имеет {buf[k].shape[0]} записей "
                             f"вместо {n}")
    key = torch.stack([buf["task"].long(), buf["state"].long(),
                       buf["call"].long()], dim=-1)
    if key.unique(dim=0).shape[0] != n:
        raise ValueError("повторяющиеся (задача, состояние, вызов): один вызов "
                         "политики попал в буфер дважды")
    h.update(np.ascontiguousarray(key.cpu().numpy().astype(np.int64)).tobytes())
    buf["order_sha1"] = h.hexdigest()[:12]
    buf["n"] = n
    buf["episodes"] = ep_rows
    buf["cb0"] = cb0.to(device)
    for k in REC_KEYS:
        buf[k] = buf[k].to(device)
    # СОБЫТИЯ БЕЗ ЭПИЗОДА И ЭПИЗОДЫ БЕЗ СОБЫТИЙ — отказ: и то и другое
    # означает, что часть градиента или часть награды потеряна
    have = {(int(r["task_id"]), int(r["state_id"])) for r in ep_rows}
    got = {(int(a), int(b)) for a, b in
           zip(buf["task"].cpu().tolist(), buf["state"].cpu().tolist())}
    if got - have:
        raise ValueError(f"вызовы без эпизода: {sorted(got - have)[:5]}")
    if have - got:
        raise ValueError(f"эпизоды без вызовов: {sorted(have - got)[:5]}")
    return buf


def episode_advantages(buf, adv_by_ep):
    import torch
    idx = {(int(r["task_id"]), int(r["state_id"])): i
           for i, r in enumerate(buf["episodes"])}
    a = torch.zeros(buf["n"], dtype=torch.float64, device=buf["h"].device)
    tl = buf["task"].cpu().tolist()
    sl = buf["state"].cpu().tolist()
    for j in range(buf["n"]):
        a[j] = float(adv_by_ep[idx[(int(tl[j]), int(sl[j]))]])
    return a


# ------------------------------ проходы -----------------------------------

def head_logp(head, buf, sl, std):
    """log pi для среза записей. Всё в fp32: см. шапку модуля."""
    import torch
    h = buf["h"][sl].float()
    z0 = buf["cb0"][buf["q0"][sl].long()].float()
    mu = head.mean_coeffs(h, z0)
    return mu, head.log_prob_u(buf["u"][sl].float(), mu, std)


def parity_check(head, buf, std, tol=1e-4):
    """log pi, пересчитанное ДО шага, обязано совпасть с записанным.

    Если не совпало — политика на обновлении не та, что действовала: отношение
    правдоподобий начиналось бы не с единицы, и весь шаг считался бы не по тем
    сэмплам. Это ровно тот дефект, из-за которого пришлось отозвать первые
    числа K-11i.
    """
    import torch
    with torch.no_grad():
        d_lp, d_mu = 0.0, 0.0
        for a, b in iter_slices(buf["n"], 256):
            mu, lp = head_logp(head, buf, slice(a, b), std)
            d_lp = max(d_lp, float((lp - buf["logp"][a:b].float()).abs().max()))
            d_mu = max(d_mu, float((mu - buf["mu"][a:b].float()).abs().max()))
    ok = bool(d_lp <= tol and d_mu <= tol)
    return dict(logp_max_abs_diff=d_lp, mu_max_abs_diff=d_mu, tol=tol, ok=ok)


def iter_slices(n, micro):
    a = 0
    while a < n:
        yield a, min(a + micro, n)
        a += micro


def grad_pass(head, buf, adv, std, n_episodes, micro):
    """Полный батч накоплением. Сумма частей равна полной функции потерь."""
    import torch
    head.zero_grad(set_to_none=True)
    total = 0.0
    for a, b in iter_slices(buf["n"], micro):
        _mu, lp = head_logp(head, buf, slice(a, b), std)
        part = -(adv[a:b].float() * lp).sum() / float(n_episodes)
        part.backward()
        total += float(part.detach())
    gn = 0.0
    for p in head.parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise FloatingPointError("в градиенте головы nan или inf")
            gn += float((p.grad.double() ** 2).sum())
    return total, float(np.sqrt(gn))


def measure(head, buf, std, micro):
    import torch
    kls, lps = [], []
    with torch.no_grad():
        for a, b in iter_slices(buf["n"], micro):
            mu, lp = head_logp(head, buf, slice(a, b), std)
            kls.append(kl_mu(buf["mu"][a:b].float(), mu, std))
            lps.append(lp)
        lp_new = torch.cat(lps)
        kl = torch.cat(kls)
        fin = all(torch.isfinite(p).all() for p in head.parameters())
    return lp_new, dict(kl_mean=float(kl.mean()), kl_max=float(kl.max()),
                        finite_params=bool(fin))


# --------------------------- снимок и откат --------------------------------

def snapshot(head, opt):
    """Снимок параметров И состояния оптимизатора, с отвязкой от живых
    тензоров: без clone откат вернул бы ссылки на уже изменённые данные."""
    return dict(
        params={k: v.detach().clone() for k, v in head.state_dict().items()},
        opt=copy.deepcopy(opt.state_dict()))


def restore(head, opt, snap, lr=None):
    """Откат и ПРОВЕРКА, что откат точный побитово."""
    import torch
    head.load_state_dict(snap["params"])
    opt.load_state_dict(copy.deepcopy(snap["opt"]))
    if lr is not None:
        for g in opt.param_groups:
            g["lr"] = float(lr)
    cur = head.state_dict()
    for k, v in snap["params"].items():
        if not torch.equal(cur[k].detach().cpu(), v.detach().cpu()):
            raise RuntimeError(f"откат неточен по {k}")
    st = opt.state_dict()["state"]
    for pid, sd in snap["opt"]["state"].items():
        for k, v in sd.items():
            if hasattr(v, "shape"):
                if not torch.equal(st[pid][k].detach().cpu(),
                                   v.detach().cpu()):
                    raise RuntimeError(f"откат неточен по моменту {pid}/{k}")
            elif st[pid][k] != v:
                raise RuntimeError(f"откат неточен по счётчику {pid}/{k}")
    return True


def one_step(head, opt, buf, adv, std, *, n_episodes, lr, trust,
             max_halvings, micro=256, log=print):
    """Один шаг с дроблением. Возвращает запись о том, что произошло.

    Если ни одно дробление не принято — параметры возвращаются в исходное
    состояние, и это ЧЕСТНЫЙ результат «шага не было», а не ошибка.
    """
    snap = snapshot(head, opt)
    loss, gnorm = grad_pass(head, buf, adv, std, n_episodes, micro)
    attempts = []
    cur_lr = float(lr)
    for k in range(int(max_halvings) + 1):
        for g in opt.param_groups:
            g["lr"] = cur_lr
        opt.step()
        lp_new, meas = measure(head, buf, std, micro)
        meas["ratio"] = ratio_stats(lp_new, buf["logp"].float(),
                                    float(trust["ratio_max"]))
        ok, why = accept_step(meas, trust)
        attempts.append(dict(halving=k, lr=cur_lr, accepted=bool(ok),
                             refused=why, kl_mean=meas["kl_mean"],
                             kl_max=meas["kl_max"],
                             ratio=meas["ratio"]))
        log(f"  дробление {k}: lr={cur_lr:.3g}, KL={meas['kl_mean']:.6f}, "
            f"отношение [{meas['ratio']['ratio_min']:.4f}, "
            f"{meas['ratio']['ratio_max']:.4f}] -> "
            f"{'принято' if ok else 'отказ: ' + '; '.join(why)}")
        if ok:
            return dict(status="stepped", loss=loss, grad_norm=gnorm,
                        lr_used=cur_lr, halvings=k, attempts=attempts,
                        order_sha1=buf["order_sha1"], n_records=buf["n"],
                        n_episodes=int(n_episodes))
        restore(head, opt, snap, lr=cur_lr)
        cur_lr /= 2.0
    restore(head, opt, snap, lr=float(lr))
    return dict(status="no_step", loss=loss, grad_norm=gnorm, lr_used=None,
                halvings=int(max_halvings), attempts=attempts,
                order_sha1=buf["order_sha1"], n_records=buf["n"],
                n_episodes=int(n_episodes))


# ------------------------------ самопроверка ------------------------------

def _stub_head(d_h, d_l, rank, sigma, seed=0):
    """Заглушка с тем же интерфейсом, что гауссова голова.

    Нужна, чтобы проверять САМ ШАГ без VLM и чекпойнтов: иначе эта логика
    тестировалась бы только на кластере, то есть не тестировалась бы.
    """
    import torch
    import torch.nn as nn
    g = torch.Generator().manual_seed(seed)

    class Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(d_l, 8)
            self.net = nn.Linear(d_h + 8, rank)
            with torch.no_grad():
                for p in self.parameters():
                    p.copy_(torch.randn(p.shape, generator=g) * 0.05)
            self.register_buffer("log_std",
                                 torch.full((rank,), float(np.log(sigma))))

        def std(self):
            return torch.exp(self.log_std)

        def mean_coeffs(self, h, z0):
            return self.net(torch.cat([h, self.proj(z0.detach())], dim=-1))

        def log_prob_u(self, u, mu, std):
            var = std * std
            lp = (-0.5 * ((u - mu) ** 2) / var - torch.log(std)
                  - 0.5 * float(np.log(2.0 * np.pi)))
            return lp.flatten(1).sum(-1)

    return Stub()


def _fake_rollout(head, *, tasks, states, calls, d_h, d_l, n_pos, rank, sigma,
                  proto, replica, stage, vocab=64, seed=1):
    """Синтетические раскатки, согласованные с головой: logp записан той же
    формулой, что будет пересчитана. Иначе проверка паритета проверяла бы
    только саму себя."""
    import torch
    g = torch.Generator().manual_seed(seed)
    cb0 = torch.randn(vocab, d_l, generator=g)
    H, Q, U, MU, LP, T, S, C = [], [], [], [], [], [], [], []
    eps = []
    for t in tasks:
        for si, s in enumerate(states):
            for c in range(calls):
                h = torch.randn(1, n_pos, d_h, generator=g)
                q0 = torch.randint(0, vocab, (1, n_pos), generator=g)
                with torch.no_grad():
                    mu = head.mean_coeffs(h, cb0[q0])
                    u = mu + sigma * torch.randn(mu.shape, generator=g)
                    lp = head.log_prob_u(u, mu, head.std())
                H.append(h); Q.append(q0.to(torch.int16)); U.append(u)
                MU.append(mu); LP.append(lp)
                T.append(torch.tensor([t])); S.append(torch.tensor([s]))
                C.append(torch.tensor([c]))
            eps.append(dict(task_id=int(t), state_id=int(s),
                            success=bool((t + si) % 2 == 0),
                            init_hash_full=f"h{t}_{s}", env_steps=calls * 8))
    data = dict(h=torch.cat(H), q0=torch.cat(Q), u=torch.cat(U),
                mu=torch.cat(MU), logp=torch.cat(LP), task=torch.cat(T),
                state=torch.cat(S), call=torch.cat(C))
    meta = dict(path="fake.pt", protocol_sha1=proto["sha1"], replica=replica,
                stage=stage, head_precision="fp32", sigma=sigma,
                d_hidden=d_h, rank=rank, head_sha1="h" * 12,
                policy_sha1="p" * 12, step_index=0,
                codebooks_sha1="c" * 12, joint_sha1="j" * 12, episodes=eps)
    return dict(meta=meta, data=data), cb0


def selftest():
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import k12b_protocol as kb

    # --- 1. преимущества: база по задаче, leave-one-out --------------------
    a, info = loo_advantage([1, 0, 1, 0], [0, 0, 1, 1])
    assert abs(a[0] - 1.0) < 1e-12 and abs(a[1] + 1.0) < 1e-12, a
    # одна задача, три успеха из четырёх: успех стоит меньше, провал дороже
    a2, _ = loo_advantage([1, 1, 1, 0], [0, 0, 0, 0])
    assert abs(a2[0] - (1 - 2 / 3)) < 1e-12, a2
    assert abs(a2[3] - (0 - 1.0)) < 1e-12, a2
    # преимущества leave-one-out центрированы ТОЖДЕСТВЕННО: сумма по задаче
    # равна нулю при любых наградах, так что общего сдвига в градиенте нет
    assert abs(a2.sum()) < 1e-12, a2.sum()
    assert abs(a.sum()) < 1e-12, a.sum()
    # задача, где все эпизоды одинаковы, не даёт градиента — и это видно
    a3, i3 = loo_advantage([1, 1, 1], [5, 5, 5])
    assert float(np.abs(a3).max()) == 0.0 and i3["flat_tasks"] == [5], i3
    a4, i4 = loo_advantage([1], [7])
    assert float(a4[0]) == 0.0 and i4["singleton_tasks"] == [7], i4
    # задачи разной трудности НЕ сравниваются между собой
    a5, _ = loo_advantage([1, 1, 0, 0], [0, 0, 1, 1])
    assert abs(a5[0] - 0.0) < 1e-12 and abs(a5[2] - 0.0) < 1e-12, a5

    st, si = standardize([2.0, -2.0, 0.0, 0.0])
    assert abs(float(np.std(st)) - 1.0) < 1e-12 and not si["degenerate"]
    st0, si0 = standardize([0.0, 0.0])
    assert si0["degenerate"] and float(np.abs(st0).max()) == 0.0

    # --- 2. KL против torch.distributions ---------------------------------
    mu_o = torch.randn(7, 4, 3)
    mu_n = mu_o + 0.01 * torch.randn(7, 4, 3)
    sd = torch.full((3,), 0.1)
    ref = torch.distributions.kl_divergence(
        torch.distributions.Normal(mu_o, sd.expand_as(mu_o)),
        torch.distributions.Normal(mu_n, sd.expand_as(mu_n))
    ).flatten(1).sum(-1)
    got = kl_mu(mu_o, mu_n, sd)
    assert torch.allclose(got, ref.double(), atol=1e-9), (got - ref).abs().max()
    assert float(kl_mu(mu_o, mu_o, sd).max()) == 0.0

    # --- 3. хвост отношения: нуль в fp32 и бесконечность ------------------
    lp_o = torch.zeros(4)
    rs = ratio_stats(torch.tensor([0.0, -105.0, 0.1, -0.1]), lp_o, 1.5)
    assert rs["zero_frac_f32"] == 0.25, rs     # exp(-105) в fp32 — ровно нуль
    assert not rs["ok"] and rs["ratio_min"] > 0.0, rs
    rs2 = ratio_stats(torch.tensor([0.0, 0.05, -0.05, 0.02]), lp_o, 1.5)
    assert rs2["ok"] and rs2["zero_frac_f32"] == 0.0, rs2
    rs3 = ratio_stats(torch.tensor([0.0, 110.0]), torch.zeros(2), 1.5)
    assert rs3["inf_frac_f32"] > 0 and not rs3["ok"], rs3
    # ровно на границе — принимается, за границей — нет
    import math
    assert ratio_stats(torch.tensor([math.log(1.5)]), torch.zeros(1),
                       1.5)["ok"]
    assert not ratio_stats(torch.tensor([math.log(1.5) + 1e-6]),
                           torch.zeros(1), 1.5)["ok"]
    try:
        ratio_stats(torch.zeros(3), torch.zeros(1), 1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("несовпадение форм правдоподобий принято")

    # --- 4. буфер, паритет и накопление градиента -------------------------
    proto = kb._proto_ok()
    tr = proto["splits"]["train"]
    d_h, d_l, n_pos, rank, sigma = 12, 16, 5, 3, 0.1
    head = _stub_head(d_h, d_l, rank, sigma)
    f1, cb0 = _fake_rollout(head, tasks=[0, 1], states=tr[:3], calls=4,
                            d_h=d_h, d_l=d_l, n_pos=n_pos, rank=rank,
                            sigma=sigma, proto=proto, replica="d10_rl0",
                            stage="train")
    info = check_rollouts([f1], proto, replica="d10_rl0", stage="train",
                          sigma=sigma)
    assert info["n_episodes"] == 6, info
    buf = concat_buffer([f1], cb0, torch.device("cpu"))
    assert buf["n"] == 2 * 3 * 4

    par = parity_check(head, buf, head.std())
    assert par["ok"], par
    # сдвинутая голова паритет НЕ проходит: это и есть защита от «не та
    # политика на обновлении»
    h2 = _stub_head(d_h, d_l, rank, sigma, seed=3)
    assert not parity_check(h2, buf, h2.std())["ok"]

    rew = [1.0 if e["success"] else 0.0 for e in buf["episodes"]]
    tsk = [e["task_id"] for e in buf["episodes"]]
    adv_ep, _ = loo_advantage(rew, tsk)
    adv_ep, _ = standardize(adv_ep)
    adv = episode_advantages(buf, adv_ep)
    assert adv.shape[0] == buf["n"]
    # преимущество приклеено к своему эпизоду, а не к порядку записей
    j = 7
    ek = (int(buf["task"][j]), int(buf["state"][j]))
    which = [i for i, e in enumerate(buf["episodes"])
             if (e["task_id"], e["state_id"]) == ek][0]
    assert abs(float(adv[j]) - float(adv_ep[which])) < 1e-12

    l_full, g_full = grad_pass(head, buf, adv, head.std(), 6, micro=10 ** 9)
    gr_full = {k: v.grad.clone() for k, v in head.named_parameters()}
    l_mb, g_mb = grad_pass(head, buf, adv, head.std(), 6, micro=5)
    # допуск — это порядок суммирования в fp32, а не другая величина:
    # накопление складывает частичные суммы, полный проход — одну
    assert abs(l_full - l_mb) < 1e-5 * max(1.0, abs(l_full)), (l_full, l_mb)
    for k, v in head.named_parameters():
        assert torch.allclose(gr_full[k], v.grad, atol=1e-6, rtol=1e-4), (
            k, float((gr_full[k] - v.grad).abs().max()))
    assert abs(g_full - g_mb) < 1e-5 * max(1.0, g_full)

    # --- 5. шаг, дробление, откат ------------------------------------------
    opt = torch.optim.Adam(head.parameters(), lr=3e-6)
    before = {k: v.detach().clone() for k, v in head.state_dict().items()}
    rec = one_step(head, opt, buf, adv, head.std(), n_episodes=6, lr=3e-6,
                   trust=dict(kl_max=0.02, ratio_max=1.5), max_halvings=4,
                   micro=10, log=lambda *_a, **_k: None)
    assert rec["status"] == "stepped" and rec["halvings"] == 0, rec
    assert any(not torch.equal(before[k], v)
               for k, v in head.state_dict().items()), "шаг ничего не изменил"
    # после шага отношение правдоподобий уже не единица — иначе шага не было
    lp_new, _m = measure(head, buf, head.std(), 10)
    assert float((lp_new - buf["logp"]).abs().max()) > 0

    # слишком большой lr: дробление доходит до приемлемого
    head2 = _stub_head(d_h, d_l, rank, sigma)
    opt2 = torch.optim.Adam(head2.parameters(), lr=1.0)
    rec2 = one_step(head2, opt2, buf, adv, head2.std(), n_episodes=6, lr=1.0,
                    trust=dict(kl_max=0.02, ratio_max=1.5), max_halvings=30,
                    micro=10, log=lambda *_a, **_k: None)
    assert rec2["status"] == "stepped" and rec2["halvings"] > 0, rec2
    assert rec2["lr_used"] < 1.0 and rec2["attempts"][0]["refused"], rec2

    # откат полный: и параметры, и моменты Adam
    head3 = _stub_head(d_h, d_l, rank, sigma)
    opt3 = torch.optim.Adam(head3.parameters(), lr=1.0)
    # сначала один настоящий шаг, чтобы моменты Adam стали ненулевыми:
    # откат с нулевыми моментами ничего бы не проверял
    grad_pass(head3, buf, adv, head3.std(), 6, micro=10)
    opt3.step()
    p_before = {k: v.detach().clone() for k, v in head3.state_dict().items()}
    o_before = copy.deepcopy(opt3.state_dict())
    mom = [s["exp_avg"].abs().sum() for s in opt3.state.values()]
    assert float(sum(mom)) > 0, "моменты Adam нулевые, откат не проверяется"
    rec3 = one_step(head3, opt3, buf, adv, head3.std(), n_episodes=6, lr=1.0,
                    trust=dict(kl_max=1e-12, ratio_max=1.0 + 1e-12),
                    max_halvings=2, micro=10, log=lambda *_a, **_k: None)
    assert rec3["status"] == "no_step", rec3
    for k, v in head3.state_dict().items():
        assert torch.equal(p_before[k], v), f"параметр {k} не восстановлен"
    st_after = opt3.state_dict()["state"]
    for pid, sd in o_before["state"].items():
        for k, v in sd.items():
            if hasattr(v, "shape"):
                assert torch.equal(st_after[pid][k], v), f"момент {pid}/{k}"
            else:
                assert st_after[pid][k] == v, f"счётчик {pid}/{k}"
    # шаг счётчика Adam тоже откатился: иначе поправка смещения поехала бы
    assert all(float(s["step"]) == 1.0 for s in opt3.state.values()), \
        [float(s["step"]) for s in opt3.state.values()]

    # --- 6. отказы сверки --------------------------------------------------
    def _expect(fn, needle):
        try:
            fn()
        except (kb.ProtocolError, ValueError) as e:
            assert needle in str(e), f"ожидал «{needle}», получил: {e}"
            return
        raise AssertionError(f"отказа «{needle}» не было")

    f_dev, _ = _fake_rollout(head, tasks=[0],
                             states=proto["splits"]["dev"][:2], calls=2,
                             d_h=d_h, d_l=d_l, n_pos=n_pos, rank=rank,
                             sigma=sigma, proto=proto, replica="d10_rl0",
                             stage="train", seed=5)
    _expect(lambda: check_rollouts([f_dev], proto, replica="d10_rl0",
                                   stage="train", sigma=sigma),
            "не из набора 'train'")
    _expect(lambda: check_rollouts([f1, f1], proto, replica="d10_rl0",
                                   stage="train", sigma=sigma),
            "посчитанный дважды")
    _expect(lambda: check_rollouts([f1], proto, replica="d11_rl1",
                                   stage="train", sigma=sigma), "реплика")
    _expect(lambda: check_rollouts([f1], proto, replica="d10_rl0",
                                   stage="train", sigma=0.2), "sigma")
    bad16 = dict(meta=dict(f1["meta"], head_precision="fp16"),
                 data=f1["data"])
    _expect(lambda: check_rollouts([bad16], proto, replica="d10_rl0",
                                   stage="train", sigma=sigma), "fp32")
    bad_sha = dict(meta=dict(f1["meta"], protocol_sha1="0" * 12),
                   data=f1["data"])
    _expect(lambda: check_rollouts([bad_sha], proto, replica="d10_rl0",
                                   stage="train", sigma=sigma), "протокол")
    bad_task = dict(meta=dict(f1["meta"], episodes=[
        dict(task_id=77, state_id=tr[0], success=True, init_hash_full="x")]),
        data=f1["data"])
    _expect(lambda: check_rollouts([bad_task], proto, replica="d10_rl0",
                                   stage="train", sigma=sigma),
            "не зарегистрированы")
    no_succ = dict(meta=dict(f1["meta"], episodes=[
        dict(task_id=0, state_id=tr[0], success=None, init_hash_full="x")]),
        data=f1["data"])
    _expect(lambda: check_rollouts([no_succ], proto, replica="d10_rl0",
                                   stage="train", sigma=sigma), "без успеха")

    # буфер: эпизод без вызовов и вызов без эпизода
    import torch as _t
    f_miss = dict(meta=dict(f1["meta"], episodes=f1["meta"]["episodes"][:-1]),
                  data=f1["data"])
    _expect(lambda: concat_buffer([f_miss], cb0, _t.device("cpu")),
            "вызовы без эпизода")
    f_extra = dict(meta=dict(f1["meta"], episodes=f1["meta"]["episodes"]
                             + [dict(task_id=1, state_id=tr[9], success=True,
                                     init_hash_full="q")]),
                   data=f1["data"])
    _expect(lambda: concat_buffer([f_extra], cb0, _t.device("cpu")),
            "эпизоды без вызовов")
    d_dup = {k: _t.cat([v[:1], v[:1], v[1:]])
             for k, v in f1["data"].items()}
    f_dup = dict(meta=f1["meta"], data=d_dup)
    _expect(lambda: concat_buffer([f_dup], cb0, _t.device("cpu")),
            "повторяющиеся (задача, состояние, вызов)")

    # --- 7. ЦЕПОЧКА ШАГОВ И НЕПРЕРЫВНОСТЬ Adam ----------------------------
    ok_prev = dict(protocol_sha1=proto["sha1"], replica="d10_rl0",
                   step_index=0, d1_head_sha1="d" * 12, d1_seed=7,
                   sigma=sigma, optimizer_state={"state": {}, "param_groups": []})
    assert check_resume_chain(ok_prev, protocol_sha1=proto["sha1"],
                              replica="d10_rl0", step_index=1,
                              d1_sha="d" * 12, d1_seed=7, sigma=sigma,
                              require_optimizer=True)
    for over, needle in ((dict(replica="d11_rl1"), "смешало бы реплики"),
                         (dict(step_index=3), "цепочка шагов разорвана"),
                         (dict(d1_head_sha1="z" * 12), "выросла из D1"),
                         (dict(d1_seed=9), "сид D1"),
                         (dict(sigma=0.2), "обучалась при sigma"),
                         (dict(protocol_sha1="0" * 12), "под протоколом"),
                         (dict(optimizer_state=None), "моменты Adam начались")):
        _expect(lambda o=over: check_resume_chain(
            dict(ok_prev, **o), protocol_sha1=proto["sha1"],
            replica="d10_rl0", step_index=1, d1_sha="d" * 12, d1_seed=7,
            sigma=sigma, require_optimizer=True), needle)

    # непрерывность НАСТОЯЩАЯ: шаг с восстановленными моментами отличается от
    # шага свежего Adam при том же градиенте — иначе проверка ничего не значит
    hA = _stub_head(d_h, d_l, rank, sigma)
    optA = torch.optim.Adam([p_ for n_, p_ in hA.named_parameters()], lr=1e-3)
    grad_pass(hA, buf, adv, hA.std(), 6, micro=10)
    optA.step()
    w_mid = {k: v.detach().clone() for k, v in hA.state_dict().items()}
    ost = copy.deepcopy(optA.state_dict())
    grad_pass(hA, buf, adv, hA.std(), 6, micro=10)
    optA.step()
    w_cont = {k: v.detach().clone() for k, v in hA.state_dict().items()}

    hB = _stub_head(d_h, d_l, rank, sigma)
    hB.load_state_dict(w_mid)
    tpB = [p_ for n_, p_ in hB.named_parameters()]
    optB = torch.optim.Adam(tpB, lr=1e-3)
    info = load_optimizer_state(optB, ost, tpB)
    assert info["step_min"] == 1.0, info
    grad_pass(hB, buf, adv, hB.std(), 6, micro=10)
    optB.step()
    for k in w_cont:
        assert torch.allclose(w_cont[k], hB.state_dict()[k], atol=1e-7), k

    hC = _stub_head(d_h, d_l, rank, sigma)
    hC.load_state_dict(w_mid)
    optC = torch.optim.Adam([p_ for n_, p_ in hC.named_parameters()], lr=1e-3)
    grad_pass(hC, buf, adv, hC.std(), 6, micro=10)
    optC.step()
    assert any(not torch.allclose(w_cont[k], hC.state_dict()[k], atol=1e-7)
               for k in w_cont), ("свежий Adam дал тот же шаг, что "
                                  "продолженный: тест непрерывности пустой")

    # состояние не от тех весов отвергается, а не применяется молча
    _expect(lambda: load_optimizer_state(
        torch.optim.Adam([p_ for p_ in hC.parameters()], lr=1e-3),
        {"state": {0: ost["state"][0]}, "param_groups": ost["param_groups"]},
        tpB), "относятся к другим весам")
    bad_shape = copy.deepcopy(ost)
    bad_shape["state"][0]["exp_avg"] = torch.zeros(3, 3)
    _expect(lambda: load_optimizer_state(
        torch.optim.Adam([p_ for p_ in hC.parameters()], lr=1e-3), bad_shape,
        tpB), "формы")

    # раскатка без policy_sha1 больше не принимается
    no_pol = dict(meta={k: v for k, v in f1["meta"].items()
                        if k != "policy_sha1"}, data=f1["data"])
    _expect(lambda: check_rollouts([no_pol], proto, replica="d10_rl0",
                                   stage="train", sigma=sigma),
            "нет policy_sha1")
    two_pol = dict(meta=dict(f1["meta"], policy_sha1="q" * 12),
                   data=f1["data"])
    _expect(lambda: check_rollouts([f1, two_pol], proto, replica="d10_rl0",
                                   stage="train", sigma=sigma),
            "раскатки разными политиками")
    two_step = dict(meta=dict(f1["meta"], step_index=5), data=f1["data"])
    _expect(lambda: check_rollouts([f1, two_step], proto, replica="d10_rl0",
                                   stage="train", sigma=sigma),
            "раскатки с разных шагов")

    print("самопроверка k12e_pg_step пройдена")


# -------------------------------- прогон ----------------------------------

def build_head(h_obj, d_h, d_l, basis, rho, dev, gaussian=True):
    import torch
    import hicora_g as hg
    import hicora_vla as hv
    cls = (hg.make_gaussian_residual_head() if gaussian
           else hv.make_residual_head())
    kw = dict(rank=int(h_obj["rank"]), hidden=int(h_obj.get("hidden", 512)),
              proj=int(h_obj.get("proj", 64)))
    h = cls(d_h, d_l, **kw).to(dev, torch.float32)
    h.set_basis(torch.as_tensor(basis))
    h.set_rho(torch.as_tensor(rho))
    st = {k[len("hicora_head."):]: v for k, v in h_obj["state"].items()}
    want = {k for k in h.state_dict() if k.startswith(("proj.", "net."))}
    if set(st) != want:
        raise SystemExit(f"набор весов головы не совпал: лишние "
                         f"{sorted(set(st) - want)[:4]}, нет "
                         f"{sorted(want - set(st))[:4]}")
    with torch.no_grad():
        for k, v in st.items():
            h.state_dict()[k].copy_(v.to(dev, torch.float32))
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--protocol", default="data/k12b/protocol.json")
    ap.add_argument("--replica", required=False)
    ap.add_argument("--stage", default="train", choices=["train"])
    ap.add_argument("--rollouts", default="",
                    help="файлы раскаток через запятую")
    ap.add_argument("--head-ckpt", required=False,
                    help="чекпойнт D1, с которого начинается шаг")
    ap.add_argument("--resume-head", default=None,
                    help="голова после предыдущего шага, если шаг не первый")
    ap.add_argument("--cb0", default="data/k12d/cb0.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro", type=int, default=256)
    ap.add_argument("--step-index", type=int, default=0)
    ap.add_argument("--out-head", required=False)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    for need in ("replica", "head_ckpt", "rollouts", "out", "out_head"):
        if not getattr(args, need):
            ap.error(f"нужен --{need.replace('_', '-')}")

    sys.path.insert(0, os.path.abspath("experiments"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    import k12b_protocol as kb
    import k9h_multiarm_gate as k9h

    proto = kb.load_protocol(args.protocol)
    sg = proto["step"]
    sigma = None          # sigma берётся из раскаток и сверяется с сеткой
    dev = torch.device(args.device)

    files = []
    for p in [x for x in args.rollouts.split(",") if x.strip()]:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        obj["meta"]["path"] = p
        files.append(obj)
    sig = {round(float(f["meta"]["sigma"]), 6) for f in files}
    if len(sig) != 1:
        raise SystemExit(f"в раскатках разные sigma: {sorted(sig)}")
    sigma = sig.pop()
    if sigma not in [round(float(s), 6) for s in proto["sigma_grid"]]:
        raise SystemExit(f"sigma={sigma} вне зарегистрированной сетки "
                         f"{proto['sigma_grid']}")
    info = check_rollouts(files, proto, replica=args.replica,
                          stage=args.stage, sigma=sigma)
    kb.check_run(proto, dict(
        stage=args.stage, protocol_sha1=proto["sha1"],
        state_ids=sorted({int(e["state_id"]) for f in files
                          for e in f["meta"]["episodes"]}),
        task_ids=info["tasks"], sigma=sigma, replica=args.replica))
    print(f"раскатки приняты: файлов {info['n_files']}, эпизодов "
          f"{info['n_episodes']}, задач {len(info['tasks'])}, sigma {sigma}")

    h_obj = torch.load(args.head_ckpt, map_location="cpu", weights_only=False)
    pref = h_obj["cache"]
    basis = np.load(pref + ".basis.npy").astype(np.float32)
    rho = np.load(pref + ".rho.npy").astype(np.float32)
    for f_, want_, lbl in ((pref + ".basis.npy", h_obj["basis_sha1"], "базис"),
                           (pref + ".rho.npy", h_obj["rho_sha1"], "предел")):
        got_ = k9h.file_sha12(f_)
        if got_ != want_:
            raise SystemExit(f"{lbl} sha {got_}, голова обучена на {want_}")
    meta0 = files[0]["meta"]
    head = build_head(h_obj, int(meta0["d_hidden"]), int(basis.shape[0]),
                      basis, rho, dev)
    d1_sha = k9h.file_sha12(args.head_ckpt)
    d1_seed = h_obj.get("seed")
    prev = None
    if args.resume_head:
        prev = torch.load(args.resume_head, map_location="cpu",
                          weights_only=False)
        check_resume_chain(prev, protocol_sha1=proto["sha1"],
                           replica=args.replica, step_index=args.step_index,
                           d1_sha=d1_sha, d1_seed=d1_seed, sigma=sigma,
                           require_optimizer=True)
        head.load_state_dict({k: v.to(dev, torch.float32)
                              for k, v in prev["state"].items()})
        policy_sha = k9h.file_sha12(args.resume_head)
    else:
        if int(args.step_index) != 0:
            raise SystemExit(f"--step-index {args.step_index} без "
                             f"--resume-head: шаг не первый, а голова взята "
                             f"исходная, то есть предыдущий шаг потерян")
        policy_sha = d1_sha
    # РАСКАТКИ ОБЯЗАНЫ БЫТЬ СОБРАНЫ ИМЕННО ЭТОЙ ПОЛИТИКОЙ. Иначе обновление
    # идёт по сэмплам другой политики без всякой поправки, и отношение
    # правдоподобий стартует не с единицы — это уже не тот алгоритм
    roll_pol = {str(f["meta"].get("policy_sha1")) for f in files}
    if roll_pol != {policy_sha}:
        raise SystemExit(f"раскатки собраны политикой {sorted(roll_pol)}, а "
                         f"шаг делается от {policy_sha}")
    roll_step = {int(f["meta"].get("step_index", -1)) for f in files}
    if roll_step != {int(args.step_index)}:
        raise SystemExit(f"раскатки с шага {sorted(roll_step)}, а шаг "
                         f"{args.step_index}")
    with torch.no_grad():
        head.log_std.fill_(float(np.log(sigma)))
    std = head.std().detach()
    if abs(float(std.max()) - sigma) > 1e-6 or \
            float(std.min()) != float(std.max()):
        raise SystemExit(f"std головы {float(std.max())} не равна sigma "
                         f"{sigma}")
    if sg.get("train_log_std"):
        raise SystemExit("протокол разрешает учить log_std, а этот шаг его "
                         "морозит: расхождение кода и регистрации")
    head.log_std.requires_grad_(False)
    train_params = [p for n, p in head.named_parameters()
                    if n.startswith(("proj.", "net."))]
    if not train_params:
        raise SystemExit("нет обучаемых параметров ветви mu")

    cb0 = torch.load(args.cb0, map_location="cpu", weights_only=False)
    cb0_t = cb0["codebook0"] if isinstance(cb0, dict) else cb0
    cb_sha = hashlib.sha1(np.ascontiguousarray(
        cb0_t.float().cpu().numpy()).tobytes()).hexdigest()[:12]
    if cb_sha != meta0["codebooks_sha1"]:
        raise SystemExit(f"кодовая книга sha {cb_sha}, раскатки писались под "
                         f"{meta0['codebooks_sha1']}: черновик z0 "
                         f"восстановился бы из другой книги")
    buf = concat_buffer(files, cb0_t, dev)

    par = parity_check(head, buf, std)
    print(f"паритет правдоподобия: |dlogp| {par['logp_max_abs_diff']:.3e}, "
          f"|dmu| {par['mu_max_abs_diff']:.3e}")
    if not par["ok"]:
        raise SystemExit(
            "ПАРИТЕТ НЕ СОШЁЛСЯ: пересчитанное log pi не совпало с записанным "
            "при раскатке.\n  Значит обновление шло бы по сэмплам другой "
            "политики, и отношение правдоподобий начиналось бы не с единицы")

    rew = [1.0 if e["success"] else 0.0 for e in buf["episodes"]]
    tsk = [int(e["task_id"]) for e in buf["episodes"]]
    adv_raw, adv_info = loo_advantage(rew, tsk)
    adv_std, sc = standardize(adv_raw)
    print(f"успех в буфере {np.mean(rew):.4f}; масштаб преимуществ "
          f"{sc['scale']:.4f}; задач без разброса "
          f"{adv_info['flat_tasks']}")
    if sc["degenerate"]:
        raise SystemExit(
            "все преимущества нулевые: в каждой задаче все эпизоды кончились "
            "одинаково.\n  Градиент был бы тождественно нулём — нужен буфер с "
            "обоими исходами, а не ещё один шаг")
    adv = episode_advantages(buf, adv_std)

    opt = torch.optim.Adam(train_params, lr=float(sg["lr"]))
    adam_info = None
    if prev is not None:
        adam_info = load_optimizer_state(opt, prev["optimizer_state"],
                                         train_params)
        print(f"состояние Adam восстановлено: параметров "
              f"{adam_info['n_params']}, счётчик шагов "
              f"{adam_info['step_min']:.0f}..{adam_info['step_max']:.0f}")
    t0 = time.time()
    rec = one_step(head, opt, buf, adv, std,
                   n_episodes=len(buf["episodes"]), lr=float(sg["lr"]),
                   trust=sg["trust"],
                   max_halvings=int(sg["backtrack"]["max_halvings"]),
                   micro=args.micro)
    rec.update(protocol_sha1=proto["sha1"], replica=args.replica,
               stage=args.stage, sigma=sigma, step_index=int(args.step_index),
               parity=par, advantage=dict(adv_info, scale=sc["scale"]),
               success_in_buffer=float(np.mean(rew)),
               rollouts=[f["meta"]["path"] for f in files],
               head_ckpt=args.head_ckpt, head_sha1=k9h.file_sha12(
                   args.head_ckpt),
               resume_head=args.resume_head,
               resume_head_sha1=(None if prev is None
                                 else k9h.file_sha12(args.resume_head)),
               policy_sha1=policy_sha, d1_head_sha1=d1_sha, d1_seed=d1_seed,
               adam_resumed=adam_info, cb0_sha1=cb_sha,
               device=str(dev), dtype="float32",
               torch_version=torch.__version__,
               script_sha1=k9h.file_sha12(os.path.abspath(__file__)),
               protocol_script_sha1=k9h.file_sha12(kb.__file__),
               minutes=(time.time() - t0) / 60.0)

    if rec["status"] == "stepped":
        os.makedirs(os.path.dirname(os.path.abspath(args.out_head)) or ".",
                    exist_ok=True)
        # СОСТОЯНИЕ ОПТИМИЗАТОРА СОХРАНЯЕТСЯ ВМЕСТЕ С ВЕСАМИ. Без него
        # следующая команда создала бы новый Adam с нулевыми моментами: точный
        # откат внутри одной попытки есть, а непрерывности между шагами не
        # было бы, и зарегистрированный алгоритм не выполнялся бы.
        torch.save(dict(state={k: v.detach().cpu()
                              for k, v in head.state_dict().items()},
                        optimizer_state=opt.state_dict(),
                        protocol_sha1=proto["sha1"], replica=args.replica,
                        step_index=int(args.step_index), sigma=sigma,
                        d1_head_sha1=d1_sha, d1_seed=d1_seed,
                        prev_head_sha1=(None if prev is None else
                                        k9h.file_sha12(args.resume_head)),
                        policy_sha1_in=policy_sha,
                        lr_used=rec["lr_used"], halvings=rec["halvings"],
                        from_head=args.resume_head or args.head_ckpt,
                        order_sha1=buf["order_sha1"]), args.out_head)
        print(f"голова сохранена: {args.out_head}")
    else:
        print("ШАГА НЕ БЫЛО: ни одно дробление не прошло область доверия; "
              "параметры возвращены побитово")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    tmp = args.out + ".tmp"
    json.dump(rec, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"запись шага: {args.out} ({rec['minutes']:.1f} мин), статус "
          f"{rec['status']}, lr {rec['lr_used']}, дроблений {rec['halvings']}")


if __name__ == "__main__":
    main()
