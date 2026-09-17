#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14a: условный оракул RVQ по кэшу K-11a. Без VLA, без раскатки, без GPU.

ВОПРОС. Умеют ли оставшиеся уровни RVQ исправить ОШИБОЧНЫЙ черновик q0? Это
главный вопрос K-14, и он решается до всякого обучения: если книг на это не
хватает, подбирать learning rate поздним головам бессмысленно.

ПОЧЕМУ ЭТО СЧИТАЕТСЯ ПО КЭШУ. В кэше K-11a уже лежат предсказанный q0hat,
истинные коды всех трёх уровней и разбиение. Ствол VLA в вопросе не участвует
вовсе: речь о латентном пространстве кодека. Значит ответ стоит десятки минут
на CPU и не мешает идущей лестнице K-13.

ЧТО СЧИТАЕТСЯ:

    z_q    = E0[k0] + E1[k1] + E2[k2]      что кодек вообще умеет представить
    A0     = decode(E0[q0hat])             нынешний ранний выход
    q1*    = Q1(z_q - E0[q0hat])           условная цель ОТ ПРЕДСКАЗАННОГО q0
    A01*   = decode(E0[q0hat] + E1[q1*])
    q2*    = Q2(z_q - E0[q0hat] - E1[q1*])
    A012*  = decode(... + E2[q2*])
    Acodec = decode(z_q)                   предел, достижимый тремя уровнями

ДВЕ МИШЕНИ, И ОСНОВНАЯ — z_e. Условные цели считаются и от непрерывного
z_e = codec._encode(action), и от трёхуровневой реконструкции z_q. Решение об
обучении принимается по z_e и по ошибке относительно НАСТОЯЩЕГО действия:
именно от z_e тренер будет строить мишени, и именно настоящее действие
исполняет робот. z_q оставлен как отдельный, более узкий вопрос о ЁМКОСТИ
книг: там знаменатель `recovery` — ровно разрыв от A0 до предела трёх уровней,
без той части непрерывного латента, которую RVQ не представляет ни при каком
префиксе. Пройденный вопрос о ёмкости при ухудшении настоящих действий
обучение не разрешает.

ЭТО ОРАКУЛ, А НЕ РЕЗУЛЬТАТ. q1* и q2* вычислены с доступом к истине. Мера
говорит, ЕСТЬ ЛИ что исправлять, и не говорит, сможет ли голова это
предсказать по h18. Между этими вопросами лежит разрыв, который проект уже
измерял: K-13b убрала 19.7% подтверждающей ошибки офлайн, а в раскатке это
дало статистически неопределённую разность.

ГОРИЗОНТ. Робот исполняет первые 8 позиций чанка из 16 (k12d_rollout,
k13c_cell). Улучшение, живущее в позициях 8..15, до него не доходит, поэтому
метрики считаются и по восьми, и по шестнадцати, а gate смотрит на восемь.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np


def sha12(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:12]


def finite(x):
    """NaN не больше и не меньше ничего.

    `float("nan") > 1e-5` равно False, поэтому любая проверка «не превышает
    порог» пропускает NaN молча. Гейт, который так устроен, при полном
    разрушении вычислений ответит «пройден».
    """
    try:
        return x is not None and bool(np.isfinite(np.asarray(x, np.float64)).all())
    except (TypeError, ValueError):
        return False


def check_finite(name, *arrays):
    for i, arr in enumerate(arrays):
        if not finite(arr):
            raise SystemExit(f"{name}[{i}]: есть nan или inf — дальнейшие "
                             f"сравнения были бы бессмысленны")
    return True


def err_blocks(a, b, max_act_q, horizon):
    """Ошибка в единицах робота: RMS, MAE и блоки каналов.

    ПОСЛЕДНИЙ КАНАЛ НЕ МАСШТАБИРУЕТСЯ — схват это команда +-1, а не
    физическая величина; в прогоне множитель применяется к `action[..., :-1]`.
    Блоки разделены, потому что одинаковый общий RMS может скрывать перенос
    ошибки из перемещения в поворот.
    """
    d = (np.asarray(a, np.float64) - np.asarray(b, np.float64))[:, :horizon]
    q = np.asarray(max_act_q, np.float64)[:d.shape[-1]].copy()
    q[-1] = 1.0
    d = d * q
    out = dict(rms=float(np.sqrt((d ** 2).mean())),
               mae=float(np.abs(d).mean()),
               rms_trans=float(np.sqrt((d[..., :3] ** 2).mean())),
               rms_rot=float(np.sqrt((d[..., 3:6] ** 2).mean())),
               rms_grip=float(np.sqrt((d[..., 6:7] ** 2).mean())),
               by_channel=[float(x) for x in np.sqrt((d ** 2).mean((0, 1)))],
               by_step=[float(x) for x in np.sqrt((d ** 2).mean((0, 2)))])
    # ПОСТРОЧНАЯ ошибка нужна для доли улучшившихся примеров: среднее по всем
    # строкам может падать за счёт немногих, при том что большинство хуже.
    # MISMATCH СХВАТА — ОТДЕЛЬНАЯ ВЕЛИЧИНА, А НЕ RMS. Схват это команда +-1;
    # средняя квадратичная ошибка по нему говорит не то, что доля позиций, где
    # знак команды разошёлся, а исполняется именно знак.
    aa = np.asarray(a, np.float64)[:, :horizon, 6]
    bb = np.asarray(b, np.float64)[:, :horizon, 6]
    out["grip_mismatch"] = float((np.sign(aa) != np.sign(bb)).mean())
    out["per_row"] = np.sqrt((d ** 2).mean((1, 2)))
    for k_, v_ in out.items():
        if not finite(v_):
            raise SystemExit(f"метрика {k_} нечисловая")
    return out


def recovery(e0, e1, e_lim, eps=1e-12):
    """Какую долю разрыва от A0 до предела трёх уровней закрывает поправка.

    ЗНАМЕНАТЕЛЬ ПРОВЕРЯЕТСЯ ЯВНО. При e0 <= e_lim разрыва нет, и отношение
    было бы делением на ноль или отрицательным числом с видом доли. Среднее
    построчных отношений тоже не годится: оно не равно отношению
    агрегированных ошибок и произвольно чувствительно к строкам с малым
    знаменателем.
    """
    gap = float(e0) - float(e_lim)
    if gap <= eps:
        return None
    return (float(e0) - float(e1)) / gap


def _blk_ok(part, names, blocks=("rms", "rms_trans", "rms_rot", "rms_grip")):
    return all(finite(part[n][b]) for n in names for b in blocks
               if n in part)


def gate(res, log=print):
    """ТРИ НЕЗАВИСИМЫХ РЕШЕНИЯ. Пороги из плана K-14, Gate 2.

    `latent_capacity_ok` — умеют ли книги восстановить ПОТОЛОК КОДЕКА после
    подмены q0. Мишени от z_q, опора decode(z_q). Узкий вопрос о ёмкости.

    `action_oracle_ok` — становится ли лучше НАСТОЯЩЕЕ действие. Мишени от
    непрерывного z_e, опора — действие из кэша K-9a. Именно это решает, имеет
    ли смысл писать тренер: первый вопрос может пройти при ухудшении
    настоящих действий, потому что там ошибка кодека тождественно нулевая и из
    анализа исчезает.

    `dynamic_q1_relabeling_supported` — даёт ли условная переразметка q1
    что-то сверх статической цели K-8. Названо с q1 в имени: статическое
    сравнение для q2 считается отдельно и в это решение не входит.

    Обучение разумно только при первых двух вместе.

    NaN ПРОВЕРЯЕТСЯ ДО СРАВНЕНИЙ: иначе каждое «больше порога» вернёт False, и
    полностью разрушенный счёт объявит гейт пройденным.
    """
    c, sel = res["val_confirm"], res["val_sel"]
    bad_cap, bad_act, note = [], [], []
    names = ("A0", "A01", "A012", "A01_static", "Acodec")
    if not (_blk_ok(c, names) and _blk_ok(sel, names)):
        bad_cap.append("среди метрик есть nan или inf: сравнивать нечего")
        return False, False, False, bad_cap, bad_act, note

    # --- РЕШЕНИЕ 1: ёмкость книг (z_q -> потолок кодека) ------------------
    if not finite(c["recovery_012"]):
        bad_cap.append(f"совокупное восстановление {c['recovery_012']}: не "
                       f"число или разрыва нет")
    if not c["A01"]["rms"] < c["A0"]["rms"]:
        bad_cap.append(f"A01* не улучшает RMS-8: {c['A01']['rms']:.5f} против "
                       f"{c['A0']['rms']:.5f}")
    if not sel["A01"]["rms"] < sel["A0"]["rms"]:
        bad_cap.append("A01* не улучшает RMS-8 на отборочной половине")
    for blk in ("rms_trans", "rms_rot", "rms_grip"):
        if c["A012"][blk] > c["A01"][blk] * 1.005:
            bad_cap.append(f"A012* хуже A01* по {blk} более чем на 0.5%: "
                           f"{c['A012'][blk]:.5f} против {c['A01'][blk]:.5f}")
    r = c["recovery_012"]
    if finite(r) and r < 0.25:
        bad_cap.append(f"совокупное восстановление {r:.3f}: меньше 25% "
                       f"разрыва")
    g8 = c["A0"]["rms"] - c["A01"]["rms"]
    g16 = c["A0_full16"]["rms"] - c["A01_full16"]["rms"]
    if g8 <= 0 < g16:
        bad_cap.append("улучшение есть только на всех 16 позициях, а на "
                       "исполняемых восьми его нет")

    # --- РЕШЕНИЕ 2: настоящее действие, мишени от z_e ---------------------
    av = lambda part, nm: part.get("vs_action." + nm)
    need2 = [av(c, n) for n in ("A0", "A01_ze", "A012_ze", "A01_static",
                                "Acodec")]
    need2 += [av(sel, n) for n in ("A0", "A01_ze")]
    if any(x is None for x in need2) or not all(
            finite(x["rms"]) and finite(x["rms_trans"])
            and finite(x["rms_rot"]) and finite(x["rms_grip"])
            and finite(x["grip_mismatch"]) for x in need2):
        bad_act.append("таблица против настоящего действия не посчитана или "
                       "нечисловая")
    else:
        if not av(c, "A01_ze")["rms"] < av(c, "A0")["rms"]:
            bad_act.append(
                f"мишени от z_e не улучшают настоящее действие: "
                f"{av(c, 'A01_ze')['rms']:.5f} против "
                f"{av(c, 'A0')['rms']:.5f}")
        if not av(sel, "A01_ze")["rms"] < av(sel, "A0")["rms"]:
            bad_act.append("на отборочной половине улучшения настоящего "
                           "действия нет")
        ra = c.get("recovery_012_ze_vs_action")
        if not finite(ra):
            bad_act.append(f"восстановление против действия {ra}: не число "
                           f"или разрыва нет")
        elif ra < 0.25:
            bad_act.append(f"восстановление против действия {ra:.3f}: меньше "
                           f"25% разрыва до предела кодека")
        # ТРЕТИЙ УРОВЕНЬ НЕ ДОЛЖЕН ПОРТИТЬ БЛОКИ. Совокупный RMS может
        # улучшиться при том, что поворот стал вчетверо хуже: агрегат
        # усредняет блоки с разными масштабами. Требование относится к
        # переходу A01 -> A012, а не к сравнению с A0.
        for blk in ("rms_trans", "rms_rot", "rms_grip"):
            if av(c, "A012_ze")[blk] > av(c, "A01_ze")[blk] * 1.005:
                bad_act.append(
                    f"A012 (z_e) хуже A01 (z_e) по {blk} более чем на 0.5%: "
                    f"{av(c, 'A012_ze')[blk]:.5f} против "
                    f"{av(c, 'A01_ze')[blk]:.5f}")
        if av(c, "A012_ze")["grip_mismatch"] > \
                av(c, "A01_ze")["grip_mismatch"] + 0.005:
            bad_act.append(
                f"расхождение схвата выросло на переходе A01 -> A012: "
                f"{av(c, 'A012_ze')['grip_mismatch']:.4f} против "
                f"{av(c, 'A01_ze')['grip_mismatch']:.4f}")

    for b in bad_cap:
        log(f"    ЁМКОСТЬ, ОТКАЗ: {b}")
    for b in bad_act:
        log(f"    ДЕЙСТВИЕ, ОТКАЗ: {b}")

    # --- РЕШЕНИЕ 3: динамическая цель q1 против статической ---------------
    # СРАВНИВАЕТСЯ ТА ВЕТВЬ, КОТОРУЮ БУДЕТ УЧИТЬ ТРЕНЕР: мишени от z_e и
    # ошибка относительно НАСТОЯЩЕГО действия. Сравнение в координатах z_q
    # против потолка кодека могло дать «динамика лучше» там, где в
    # исполняемом пространстве лучше статика.
    dyn = (av(c, "A01_ze") or {}).get("rms")
    sta = (av(c, "A01_static") or {}).get("rms")
    if not (finite(dyn) and finite(sta)):
        note.append("статическая цель не посчитана")
        dynamic_ok = False
    elif sta <= dyn * 1.005:
        note.append(f"статическая цель K-8 даёт то же ({sta:.5f} против "
                    f"{dyn:.5f}): условная переразметка q1 не подтверждена как "
                    f"механизм, приписывать ей эффект нельзя")
        dynamic_ok = False
    else:
        note.append(f"условная цель q1 лучше статической: {dyn:.5f} против "
                    f"{sta:.5f} ({100 * (sta - dyn) / sta:.1f}%)")
        dynamic_ok = True
    for x in note:
        log(f"    {x}")
    return (not bad_cap), (not bad_act), bool(dynamic_ok), bad_cap, bad_act, note


def build_parts(idx, epi, sel_frac, seed, n_rows, rng, split_episodes):
    """train / val_sel / val_confirm из разбиения кэша. ЧИСТАЯ ФУНКЦИЯ.

    ВЫНЕСЕНО ПОСЛЕ РЕАЛЬНОГО ПАДЕНИЯ. Здесь стояло `idx["dev"]`, а
    `load_split` отдаёт `train/val/test`: KeyError на настоящем кэше при
    зелёной самопроверке, потому что самопроверка это место не исполняла.
    Теперь исполняет, с настоящими именами ключей.

    ПОДВЫБОРКА ТОЛЬКО У train. K-13b считал на ЦЕЛЫХ val_sel и val_confirm;
    случайная часть сделала бы слова «та же подтверждающая половина» неверными.
    """
    miss = [k for k in ("train", "val") if k not in idx]
    if miss:
        raise SystemExit(f"в разбиении нет частей {miss}: ожидались "
                         f"train/val/test, получено {sorted(idx)}")
    val_idx = idx["val"]
    sel_eps, cnf_eps = split_episodes(np.asarray(epi[val_idx]), sel_frac,
                                      seed=int(seed))
    e_val = np.asarray(epi[val_idx])
    parts = {"train": np.asarray(idx["train"]),
             "val_sel": val_idx[np.isin(e_val, list(sel_eps))],
             "val_confirm": val_idx[np.isin(e_val, list(cnf_eps))]}
    avail = {k: int(len(v)) for k, v in parts.items()}
    if n_rows:
        parts["train"] = np.sort(rng.choice(
            parts["train"], size=min(int(n_rows), len(parts["train"])),
            replace=False))
    meta = {}
    for k, v in parts.items():
        if len(v) == 0:
            raise SystemExit(f"часть {k} пуста")
        meta[k] = dict(
            n_available=avail[k], n_used=int(len(v)),
            rows_sha1=hashlib.sha1(np.ascontiguousarray(
                np.asarray(v, np.int64)).tobytes()).hexdigest()[:12],
            episodes_sha1=hashlib.sha1(np.ascontiguousarray(
                np.unique(epi[v]).astype(np.int64)).tobytes()).hexdigest()[:12],
            n_episodes=int(len(np.unique(epi[v]))))
    return parts, meta


def selftest():
    a = np.zeros((4, 16, 7))
    b = np.zeros((4, 16, 7))
    b[..., 0] = 1.0
    q = np.ones(7)
    e = err_blocks(a, b, q, 8)
    assert abs(e["rms"] - np.sqrt(1.0 / 7)) < 1e-12
    assert abs(e["rms_trans"] - np.sqrt(1.0 / 3)) < 1e-12
    assert e["rms_rot"] == 0.0 and e["rms_grip"] == 0.0
    assert len(e["by_step"]) == 8 and len(e["per_row"]) == 4

    c = np.zeros((4, 16, 7))
    c[:, 8:, 0] = 5.0
    assert err_blocks(a, c, q, 8)["rms"] == 0.0
    assert err_blocks(a, c, q, 16)["rms"] > 0.0

    g = np.zeros((4, 16, 7))
    g[..., 6] = 1.0
    q2 = np.full(7, 10.0)
    assert abs(err_blocks(a, g, q2, 8)["rms"] - np.sqrt(1.0 / 7)) < 1e-12

    # --- NaN И Inf НЕ ПРОХОДЯТ -------------------------------------------
    assert not finite(float("nan")) and not finite(float("inf"))
    assert not finite(None) and finite(0.0)
    assert finite(np.zeros(3)) and not finite(np.array([1.0, np.nan]))
    bad_arr = np.zeros((4, 16, 7)); bad_arr[0, 0, 0] = np.nan
    for arr in (bad_arr, np.full((4, 16, 7), np.inf)):
        try:
            err_blocks(a, arr, q, 8)
        except SystemExit as e_:
            assert "нечисловая" in str(e_), e_
        else:
            raise AssertionError("нечисловые действия прошли в метрики")
    try:
        check_finite("проба", np.zeros(3), bad_arr)
    except SystemExit as e_:
        assert "nan" in str(e_), e_
    else:
        raise AssertionError("check_finite пропустил nan")

    assert abs(recovery(1.0, 0.5, 0.0) - 0.5) < 1e-12
    assert recovery(1.0, 0.5, 1.0) is None
    assert recovery(1.0, 0.5, 2.0) is None
    assert recovery(1.0, 1.5, 0.0) < 0

    def mk(a0, a01, a012, rec, a0f=None, a01f=None, static=None,
           act_a0=1.0, act_a01=0.5, act_a012=0.4, act_floor=0.0,
           act_rec=0.6, grip0=0.05, grip012=0.05, static_act=None):
        blk = lambda r, gm=0.0: dict(rms=r, rms_trans=r, rms_rot=r,
                                     rms_grip=r, grip_mismatch=gm)

        def part():
            d = dict(A0=blk(a0), A01=blk(a01), A012=blk(a012),
                     A01_static=blk(a01 * 1.5 if static is None else static),
                     Acodec=blk(0.0), recovery_012=rec,
                     A0_full16=blk(a0 if a0f is None else a0f),
                     A01_full16=blk(a01 if a01f is None else a01f),
                     recovery_012_ze_vs_action=act_rec)
            d["vs_action.A0"] = blk(act_a0, grip0)
            d["vs_action.A01_ze"] = blk(act_a01, grip0)
            d["vs_action.A012_ze"] = blk(act_a012, grip012)
            d["vs_action.Acodec"] = blk(act_floor, grip0)
            d["vs_action.A01_static"] = blk(
                act_a01 * 1.5 if static_act is None else static_act, grip0)
            return d
        return dict(val_sel=part(), val_confirm=part())

    cap, act, dyn, bc, ba, _ = gate(mk(1.0, 0.5, 0.4, 0.6),
                                    log=lambda *_: None)
    assert cap and act and dyn, (bc, ba)
    for args_, why in (((1.0, 1.2, 1.1, 0.6), "не улучшает"),
                       ((1.0, 0.5, 0.9, 0.6), "хуже A01"),
                       ((1.0, 0.5, 0.4, 0.1), "меньше 25%"),
                       ((1.0, 0.5, 0.4, None), "не число")):
        cap, act, _d, bc, ba, _ = gate(mk(*args_), log=lambda *_: None)
        assert not cap and any(why in x for x in bc), (why, bc)
    cap, act, _d, bc, _ba, _ = gate(mk(1.0, 1.0, 0.9, 0.6, a0f=1.0, a01f=0.5),
                                    log=lambda *_: None)
    assert not cap and any("исполняемых восьми" in x for x in bc), bc

    # --- РЕШЕНИЕ 2 НЕЗАВИСИМО: ёмкость есть, настоящее действие хуже ------
    # Главный случай из разбора: гейт по потолку кодека проходит, а действия
    # портятся. Раньше такая комбинация давала «гейт пройден».
    cap, act, _d, _bc, ba, _ = gate(mk(1.0, 0.5, 0.4, 0.6, act_a01=1.2),
                                    log=lambda *_: None)
    assert cap and not act and any("не улучшают настоящее" in x for x in ba), ba
    cap, act, _d, _bc, ba, _ = gate(mk(1.0, 0.5, 0.4, 0.6, act_rec=0.1),
                                    log=lambda *_: None)
    assert cap and not act and any("меньше" in x for x in ba), ba
    cap, act, _d, _bc, ba, _ = gate(mk(1.0, 0.5, 0.4, 0.6, grip012=0.20),
                                    log=lambda *_: None)
    assert cap and not act and any("схват" in x for x in ba), ba

    nan_res = mk(1.0, 0.5, float("nan"), float("nan"))
    cap, act, _d, bc, _ba, _ = gate(nan_res, log=lambda *_: None)
    assert not cap and any("nan" in x for x in bc), bc

    # --- КОНТРПРИМЕР 1: поворот у A012_ze вчетверо хуже A01_ze -----------
    # Совокупный RMS улучшается, блок портится. Раньше гейт это пропускал.
    res_rot = mk(1.0, 0.5, 0.4, 0.6)
    for part in res_rot.values():
        part["vs_action.A012_ze"] = dict(part["vs_action.A012_ze"])
        part["vs_action.A012_ze"]["rms_rot"] = \
            part["vs_action.A01_ze"]["rms_rot"] * 4.0
    cap, act, _d, _bc, ba, _ = gate(res_rot, log=lambda *_: None)
    assert cap and not act and any("rms_rot" in x for x in ba), ba

    # --- КОНТРПРИМЕР 2: статическая q1 лучше динамической от z_e ---------
    res_st = mk(1.0, 0.5, 0.4, 0.6, act_a01=0.8, static_act=0.7)
    cap, act, dyn, _bc, _ba, note = gate(res_st, log=lambda *_: None)
    assert not dyn, note
    assert any("статическая цель" in x for x in note), note

    # --- схват портится именно на переходе A01 -> A012 --------------------
    res_gr = mk(1.0, 0.5, 0.4, 0.6, grip0=0.05, grip012=0.20)
    for part in res_gr.values():
        part["vs_action.A01_ze"] = dict(part["vs_action.A01_ze"])
        part["vs_action.A01_ze"]["grip_mismatch"] = 0.05
    cap, act, _d, _bc, ba, _ = gate(res_gr, log=lambda *_: None)
    assert cap and not act and any("A01 -> A012" in x for x in ba), ba

    cap, act, dyn, _bc, _ba, note = gate(mk(1.0, 0.5, 0.4, 0.6, static=0.5),
                                         log=lambda *_: None)
    assert cap and act and dyn, note

    # --- ПОСТРОЕНИЕ ЧАСТЕЙ НА НАСТОЯЩИХ КЛЮЧАХ РАЗБИЕНИЯ ------------------
    # Регрессия на реальное падение: тут стояло idx["dev"], а load_split даёт
    # train/val/test. Самопроверка это место не исполняла и была зелёной.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from k11c_train_d1 import split_episodes as _se
    # эпизодов в val должно хватать на обе половины при min_each=8
    n_ep, per_ep = 60, 6
    epi_t = np.repeat(np.arange(n_ep), per_ep)
    sp = np.array(["train"] * (20 * per_ep) + ["val"] * (30 * per_ep)
                  + ["test"] * (10 * per_ep))
    idx_t = {k: np.flatnonzero(sp == k) for k in ("train", "val", "test")}
    rng_t = np.random.default_rng(0)
    parts_t, meta_t = build_parts(idx_t, epi_t, 0.4, 61, 0, rng_t, _se)
    assert set(parts_t) == {"train", "val_sel", "val_confirm"}
    # ЦЕЛЫЕ ПОЛОВИНЫ: без подвыборки объединение val_sel и val_confirm равно val
    assert (len(parts_t["val_sel"]) + len(parts_t["val_confirm"])
            == len(idx_t["val"]))
    e_sel = set(epi_t[parts_t["val_sel"]].tolist())
    e_cnf = set(epi_t[parts_t["val_confirm"]].tolist())
    assert not (e_sel & e_cnf), "эпизод попал в обе половины"
    # ПОДВЫБОРКА ТОЛЬКО У train
    parts_s, meta_s = build_parts(idx_t, epi_t, 0.4, 61, 30,
                                  np.random.default_rng(0), _se)
    assert len(parts_s["train"]) == 30
    assert len(parts_s["val_confirm"]) == len(parts_t["val_confirm"])
    assert meta_s["train"]["n_available"] == len(idx_t["train"])
    assert meta_s["val_confirm"]["rows_sha1"] == \
        meta_t["val_confirm"]["rows_sha1"]
    # ОТСУТСТВИЕ ЧАСТИ — ОТКАЗ, А НЕ KeyError
    try:
        build_parts({"train": idx_t["train"]}, epi_t, 0.4, 61, 0, rng_t, _se)
    except SystemExit as e_:
        assert "val" in str(e_), e_
    else:
        raise AssertionError("отсутствующая часть разбиения пропущена")

    print("самопроверка k14a_oracle_cache пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--ckpt",
                    default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--n-rows", type=int, default=4096,
                    help="сколько строк каждой части брать; 0 — все")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--sel-frac", type=float, default=0.4)
    ap.add_argument("--probe-rows", type=int, default=256,
                    help="сколько строк кодировать заново для сверки с K_true")
    ap.add_argument("--probe-tol", type=float, default=0.0,
                    help="допустимая доля расходящихся кодов; ноль означает "
                         "требование точного совпадения")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/k14a/oracle_cache.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(a.root)
    for p in (root, os.path.join(root, "src"), here,
              os.path.abspath("experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import k11a_build_hicora_cache as k11a
    import k12b_protocol as kb
    import k13a_build_trajectory_basis as k13a
    from k11c_train_d1 import split_episodes
    import actioncodec  # noqa: F401
    from utils import ACTION_Q01, ACTION_Q99, VisionLanguageActionProcessor

    dev = torch.device(a.device)
    max_act_q = np.maximum(np.abs(ACTION_Q01), np.abs(ACTION_Q99))

    # --- кэш и происхождение ------------------------------------------------
    meta = json.load(open(f"{a.cache}.meta.json"))
    if meta.get("ckpt") != a.ckpt:
        raise SystemExit(f"кэш собран чекпойнтом {meta.get('ckpt')}, а кодек "
                         f"берётся из {a.ckpt}")
    # --- ПРОВЕНАНС q0hat: от него зависит весь эксперимент ------------------
    # Проверки кодека недостаточно: другой q0hat подходящей формы, собранный
    # другой моделью или другой глубиной, прошёл бы их все. Здесь сверяется то,
    # ЧЕМ он получен.
    if meta.get("q0_source") != "joint12":
        raise SystemExit(f"q0 получен источником {meta.get('q0_source')}, а "
                         f"K-14 исследует уточнение черновика Joint12")
    if int(meta.get("depth", -1)) != 12:
        raise SystemExit(f"кэш собран на глубине {meta.get('depth')}, а "
                         f"ранний выход K-14 задан на 12")
    stamp_p = a.cache + ".artifacts.json"
    if not os.path.exists(stamp_p):
        raise SystemExit(
            f"нет {stamp_p}: кэш не заверен K-11b, и совпадение массивов с "
            f"теми, на которых проверено тождество, подтвердить нечем")
    stamp = json.load(open(stamp_p))
    if not stamp.get("identity_ok"):
        raise SystemExit("K-11b не подтвердила тождество для этого кэша")
    for nm in ("q0hat", "ktrue", "split", "codebooks"):
        p_ = f"{a.cache}.{nm}.npy"
        want_ = (stamp.get("arrays") or {}).get(nm)
        if want_ is None:
            raise SystemExit(f"в заверении K-11b нет отпечатка {nm}")
        got_ = k11a.file_sha1(p_)
        if got_ != want_:
            raise SystemExit(f"{nm}.npy имеет sha {got_}, K-11b заверила "
                             f"{want_}: массив подменён после проверки")
    if stamp.get("cache_meta_sha1") != k11a.file_sha1(f"{a.cache}.meta.json"):
        raise SystemExit("meta.json изменён после заверения K-11b")
    src_meta = meta.get("source") or {}
    jp = src_meta.get("path")
    if jp and os.path.exists(jp):
        got_w = k11a.file_sha1(jp)
        if got_w != src_meta.get("weights_sha1"):
            raise SystemExit(f"чекпойнт Joint12 {jp} имеет sha {got_w}, кэш "
                             f"собран на {src_meta.get('weights_sha1')}")
        print(f"  Joint12 сверен: {jp}, sha {got_w}, глубина "
              f"{src_meta.get('depth')}")
    else:
        print(f"  ВНИМАНИЕ: чекпойнт Joint12 {jp} недоступен, сверен только "
              f"его отпечаток в meta ({src_meta.get('weights_sha1')})")
    print(f"  заверение K-11b: тождество подтверждено, четыре массива и "
          f"meta совпали с заверенными")

    ktrue = np.load(f"{a.cache}.ktrue.npy", mmap_mode="r")
    q0hat = np.load(f"{a.cache}.q0hat.npy", mmap_mode="r")
    E = np.load(f"{a.cache}.codebooks.npy")
    idx, _ = k13a.load_split(f"{a.cache}.split.npy", q0hat.shape[0])
    # ЭПИЗОДЫ БЕРУТСЯ ИЗ ИСХОДНОГО КЭША K-9a, как это делает K-13b: отдельного
    # .episode.npy у K-11a нет, а делить val по наблюдениям нельзя — кадры
    # одного эпизода сильно зависимы, и «подтверждающая» половина перестала бы
    # быть независимой.
    src = meta.get("cache")
    if not src or not os.path.exists(src):
        raise SystemExit(f"исходный кэш {src} недоступен: без эпизодов val "
                         f"нельзя разделить так же, как в K-11c и K-13b")
    epi = np.asarray(np.load(src, allow_pickle=True)["episode"]).astype(
        np.int64)[:q0hat.shape[0]]

    N = int(meta["n_obs"])
    V = int(meta.get("vocab", 0)) or int(E.shape[1])
    want_shapes = {"q0hat": (N, 16), "ktrue": (N, 3, 16)}
    for nm, arr in (("q0hat", q0hat), ("ktrue", ktrue)):
        if tuple(arr.shape) != want_shapes[nm]:
            raise SystemExit(f"{nm} формы {tuple(arr.shape)}, ожидалась "
                             f"{want_shapes[nm]} при n_obs {N}")
    for nm, arr in (("q0hat", q0hat), ("ktrue", ktrue)):
        lo, hi = int(np.asarray(arr).min()), int(np.asarray(arr).max())
        if lo < 0 or hi >= V:
            raise SystemExit(f"{nm}: коды в диапазоне [{lo}, {hi}] при "
                             f"словаре {V}")
    print(f"  формы и диапазоны кодов проверены: n_obs {N}, словарь {V}")

    # --- кодек: только он и нужен -------------------------------------------
    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None))
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь")
    codec = codec.to(dev).eval()
    qs = list(codec.vq.quantizers)
    if len(qs) != 3:
        raise SystemExit(f"уровней {len(qs)}, ожидалось 3")

    with torch.no_grad():
        ii = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        Ecur = torch.stack([q.out_project(q.decode_code(ii))[0]
                            for q in qs]).float()
    dmax = float((Ecur.cpu() - torch.from_numpy(E)).abs().max())
    if dmax > 1e-5:
        raise SystemExit(f"книги разошлись с кэшем на {dmax:.3e}")
    k11a.check_fingerprints(meta, dict(
        codebooks_sha1=hashlib.sha1(np.ascontiguousarray(
            E.astype(np.float32)).tobytes()).hexdigest()[:12],
        decoder_probe=k11a.decoder_probe(codec, Ecur.to(dev), dev),
        codec_state_sha1=k11a.state_sha1(codec)))
    print(f"  кодек сверен с кэшем: книги max|Δ| = {dmax:.3e}, проба и веса "
          f"совпали; уровней {len(qs)}")

    # --- части: train и val, разделённый ПО ЭПИЗОДАМ ------------------------
    # ТА ЖЕ разметка, что в K-11c и K-13b (сид 61): подтверждающая половина
    # должна оставаться той же самой во всех работах, иначе «подтверждение»
    # каждый раз считается на новых данных.
    rng = np.random.default_rng(a.seed)
    parts, sample_meta = build_parts(idx, epi, a.sel_frac, 61, a.n_rows, rng,
                                     split_episodes)
    avail = {k: v["n_available"] for k, v in sample_meta.items()}
    print("  части: " + ", ".join(
        f"{k} {len(v)} из {avail[k]}" for k, v in parts.items())
        + ". Подвыборка только у train; финальная выборка не читается")

    # ВТОРАЯ ОПОРА: НАСТОЯЩЕЕ ДЕЙСТВИЕ. z_q -> Acodec отвечает, умеют ли книги
    # восстановить потолок кодека. Но ошибка самого кодека относительно
    # демонстрации при такой опоре тождественно нулевая и из анализа исчезает.
    # Поэтому вторая таблица считается против действия из кэша K-9a — того же,
    # на котором будут строиться мишени обучения.
    src_npz = np.load(src, allow_pickle=True)
    for nm in ("action", "episode", "step", "K_true", "split"):
        if nm not in src_npz:
            raise SystemExit(f"в {src} нет массива {nm}: замкнуть провенанс "
                             f"исходного кэша нечем")
    ACT = src_npz["action"]
    # ИСХОДНЫЙ КЭШ СВЕРЯЕТСЯ, А НЕ ПРОСТО ЗАПИСЫВАЕТСЯ. Раньше его sha попадал
    # в артефакт и ни с чем не сравнивался — то есть был отчётом, а не
    # проверкой: подменённый файл прошёл бы.
    if ACT.shape[0] != N:
        raise SystemExit(f"в исходном кэше {ACT.shape[0]} действий при "
                         f"n_obs {N}")
    keys_now = hashlib.sha1(np.ascontiguousarray(np.stack(
        [np.asarray(src_npz["episode"]),
         np.asarray(src_npz["step"])])).tobytes()).hexdigest()[:12]
    if not meta.get("keys_sha1"):
        raise SystemExit(
            "в meta кэша нет keys_sha1: сверить, что это тот же набор "
            "наблюдений, нечем. Условие «если поле есть» было fail-open — "
            "кэш без поля проходил бы")
    if keys_now != meta["keys_sha1"]:
        raise SystemExit(f"(episode, step) исходного кэша дают {keys_now}, а "
                         f"K-11a собран на {meta['keys_sha1']}: это другой "
                         f"набор наблюдений")
    kt_src = np.asarray(src_npz["K_true"])[:N].astype(np.int64)
    if kt_src.shape != tuple(ktrue.shape):
        raise SystemExit(f"K_true исходного кэша формы {kt_src.shape} против "
                         f"{tuple(ktrue.shape)}")
    if not np.array_equal(kt_src, np.asarray(ktrue).astype(np.int64)):
        n_d = int((kt_src != np.asarray(ktrue)).sum())
        raise SystemExit(f"K_true исходного кэша расходится с заверенным в "
                         f"{n_d} позициях: истинные коды не те")
    sp_src = np.asarray(src_npz["split"])[:N].astype(str)
    sp_cache = np.load(f"{a.cache}.split.npy", allow_pickle=True).astype(str)
    if not np.array_equal(sp_src, sp_cache):
        raise SystemExit("split исходного кэша расходится с заверенным: "
                         "части считались бы на других наблюдениях")
    print(f"  исходный кэш K-9a замкнут: ключи {keys_now}, K_true и split "
          f"совпали с заверенными, {N} действий")

    from depth_rvq_joint12 import code_contribution, nearest_code

    def decode(z, batch=256):
        out = []
        with torch.no_grad():
            for i in range(0, len(z), batch):
                x, _ = codec._decode(z[i:i + batch].float(), embodiment_ids=0)
                out.append(x[..., :7].float().cpu().numpy())
        return np.concatenate(out)

    def encode(x, batch=256):
        """Непрерывный латент действий — тот самый z_e из плана."""
        out = []
        with torch.no_grad():
            for i in range(0, len(x), batch):
                out.append(codec._encode(x[i:i + batch].float(),
                                         embodiment_ids=0).float())
        return torch.cat(out)

    # ДЕЙСТВИЯ СВЕРЯЮТСЯ С ЗАВЕРЕННЫМИ КОДАМИ. K_true, split и ключи уже
    # сверены, но сам массив `action` теперь определяет z_e, а значит и всю
    # вторую таблицу. Переставленные или изменённые действия оставили бы
    # остальные заверенные массивы нетронутыми и молча сдвинули бы оракул.
    probe_rows = np.sort(np.random.default_rng(20260917).choice(
        N, size=min(int(a.probe_rows), N), replace=False))
    with torch.no_grad():
        z_probe = codec._encode(
            torch.from_numpy(np.asarray(ACT[probe_rows], np.float32)).to(dev),
            embodiment_ids=0).float()
        k_probe = []
        resid = z_probe.clone()
        for l in range(3):
            c_ = nearest_code(resid, qs[l])
            k_probe.append(c_)
            resid = resid - code_contribution(qs[l], c_)
        k_probe = torch.stack(k_probe, 1).cpu().numpy()
    k_ref = np.asarray(ktrue[probe_rows]).astype(np.int64)
    dis = float((k_probe != k_ref).mean())
    if dis > float(a.probe_tol):
        raise SystemExit(
            f"проба кодирования: {100 * dis:.3f}% кодов расходятся с "
            f"заверенным K_true при допуске {100 * float(a.probe_tol):.3f}%. "
            f"Действия в исходном кэше не те, на которых собран K-11a")
    print(f"  проба кодирования на {len(probe_rows)} строках: расхождение "
          f"кодов {100 * dis:.4f}% при допуске "
          f"{100 * float(a.probe_tol):.3f}%")

    res, extra = {}, {}
    for name, rows in parts.items():
        k = torch.from_numpy(np.asarray(ktrue[rows]).astype(np.int64)).to(dev)
        q0 = torch.from_numpy(np.asarray(q0hat[rows]).astype(np.int64)).to(dev)
        act = torch.from_numpy(np.asarray(ACT[rows], np.float32)).to(dev)
        with torch.no_grad():
            z_q = sum(code_contribution(qs[l], k[:, l, :]) for l in range(3))
            # Z_E — ОСНОВНАЯ МИШЕНЬ ПЛАНА. z_q оставлен как отдельный вопрос о
            # ёмкости книг; решение об обучении принимается по z_e, потому что
            # именно от него будут строиться мишени тренера и именно он
            # сравним с настоящим действием.
            z_e = encode(act)
            if tuple(z_e.shape) != tuple(z_q.shape):
                raise SystemExit(f"z_e формы {tuple(z_e.shape)}, z_q "
                                 f"{tuple(z_q.shape)}")
            e0 = code_contribution(qs[0], q0)
            tgt = {}
            for tag, zt in (("zq", z_q), ("ze", z_e)):
                q1s = nearest_code(zt - e0, qs[1])
                e1 = code_contribution(qs[1], q1s)
                q2s = nearest_code(zt - e0 - e1, qs[2])
                e2 = code_contribution(qs[2], q2s)
                tgt[tag] = (q1s, e1, q2s, e2)
        q1s, e1, q2s, e2 = tgt["zq"]
        q1e, e1e, q2e, e2e = tgt["ze"]
        A = dict(A0=decode(e0), A01=decode(e0 + e1), A012=decode(e0 + e1 + e2),
                 Acodec=decode(z_q),
                 A01_ze=decode(e0 + e1e), A012_ze=decode(e0 + e1e + e2e))
        # СТАТИЧЕСКАЯ ЦЕЛЬ K-8 ДЛЯ СРАВНЕНИЯ: q1 берётся истинный, без учёта
        # того, что q0 предсказан с ошибкой. Если разницы нет, вся идея
        # условной переразметки не нужна, и это надо знать до обучения.
        with torch.no_grad():
            e1_static = code_contribution(qs[1], k[:, 1, :])
            e2_static = code_contribution(qs[2], k[:, 2, :])
        A["A01_static"] = decode(e0 + e1_static)
        A["A012_static"] = decode(e0 + e1_static + e2_static)
        check_finite("декодированные действия", *A.values())
        atrue = A["Acodec"]
        # НАСТОЯЩЕЕ ДЕЙСТВИЕ ИЗ КЭША K-9a — вторая опора. Первые семь каналов
        # и те же шестнадцать позиций, что у декодированных.
        a_real = np.asarray(ACT[rows], np.float64)[..., :7]
        if a_real.shape[1:] != atrue.shape[1:]:
            raise SystemExit(f"действие из кэша формы {a_real.shape[1:]}, "
                             f"декодированное {atrue.shape[1:]}")
        check_finite("действие из кэша", a_real)
        r = {}
        for nm, arr in A.items():
            r[nm] = err_blocks(arr, atrue, max_act_q, a.horizon)
            r[nm + "_full16"] = err_blocks(arr, atrue, max_act_q, 16)
            # ВТОРАЯ ТАБЛИЦА: та же поправка, другая опора
            r["vs_action." + nm] = err_blocks(arr, a_real, max_act_q,
                                              a.horizon)
        r["recovery_01"] = recovery(r["A0"]["rms"], r["A01"]["rms"],
                                    r["Acodec"]["rms"])
        r["recovery_012"] = recovery(r["A0"]["rms"], r["A012"]["rms"],
                                     r["Acodec"]["rms"])
        # ТО ЖЕ ОТНОСИТЕЛЬНО НАСТОЯЩЕГО ДЕЙСТВИЯ: знаменатель здесь — разрыв
        # от A0 до того, что даёт сам кодек, и он НЕ нулевой.
        r["recovery_01_vs_action"] = recovery(
            r["vs_action.A0"]["rms"], r["vs_action.A01"]["rms"],
            r["vs_action.Acodec"]["rms"])
        r["recovery_012_vs_action"] = recovery(
            r["vs_action.A0"]["rms"], r["vs_action.A012"]["rms"],
            r["vs_action.Acodec"]["rms"])
        r["codec_floor_vs_action"] = r["vs_action.Acodec"]["rms"]
        r["recovery_01_ze_vs_action"] = recovery(
            r["vs_action.A0"]["rms"], r["vs_action.A01_ze"]["rms"],
            r["vs_action.Acodec"]["rms"])
        r["recovery_012_ze_vs_action"] = recovery(
            r["vs_action.A0"]["rms"], r["vs_action.A012_ze"]["rms"],
            r["vs_action.Acodec"]["rms"])
        r["frac_A01_better"] = float(
            (r["A01"]["per_row"] < r["A0"]["per_row"]).mean())
        r["frac_A012_better"] = float(
            (r["A012"]["per_row"] < r["A01"]["per_row"]).mean())
        wrong = (q0 != k[:, 0, :]).any(-1).cpu().numpy()
        r["frac_rows_with_wrong_q0"] = float(wrong.mean())
        for tag, m in (("q0_wrong", wrong), ("q0_right", ~wrong)):
            if m.sum() >= 8:
                r[f"rms_A0_{tag}"] = float(r["A0"]["per_row"][m].mean())
                r[f"rms_A01_{tag}"] = float(r["A01"]["per_row"][m].mean())
        # РАСХОЖДЕНИЯ РАЗДЕЛЕНЫ ПО МИШЕНИ. Основная — от z_e: именно её
        # будет учить тренер. Цифра от z_q оставлена как диагностика.
        r["dynamic_zq_vs_static_q1_disagree"] = float(
            (q1s != k[:, 1, :]).float().mean())
        r["dynamic_ze_vs_static_q1_disagree"] = float(
            (q1e != k[:, 1, :]).float().mean())
        r["dynamic_zq_vs_static_q2_disagree"] = float(
            (q2s != k[:, 2, :]).float().mean())
        r["dynamic_ze_vs_static_q2_disagree"] = float(
            (q2e != k[:, 2, :]).float().mean())
        r["ze_vs_zq_q1_disagree"] = float((q1e != q1s).float().mean())
        r["n_rows"] = int(len(rows))
        for nm in list(r):
            if isinstance(r[nm], dict) and "per_row" in r[nm]:
                extra[f"{name}.{nm}"] = r[nm].pop("per_row")
        res[name] = r
        print(f"\n  === {name}: {len(rows)} строк, строк с ошибочным q0 "
              f"{100 * r['frac_rows_with_wrong_q0']:.1f}% ===")
        print("    опора 1 — потолок кодека decode(z_q):")
        for nm in ("A0", "A01", "A012", "A01_static", "Acodec"):
            print(f"      {nm:11s} RMS-8 {r[nm]['rms']:.5f}  "
                  f"(перемещение {r[nm]['rms_trans']:.5f}, поворот "
                  f"{r[nm]['rms_rot']:.5f}, схват {r[nm]['rms_grip']:.5f})")
        print("    опора 2 — настоящее действие из кэша K-9a:")
        for nm in ("A0", "A01", "A012", "A01_ze", "A012_ze", "A01_static",
                   "Acodec"):
            k_ = "vs_action." + nm
            print(f"      {nm:11s} RMS-8 {r[k_]['rms']:.5f}  "
                  f"схват расходится {100*r[k_]['grip_mismatch']:.2f}%")
        print(f"      восстановление против действия, мишени z_e: q0->q01 "
              f"{r['recovery_01_ze_vs_action']}, q0->q012 "
              f"{r['recovery_012_ze_vs_action']}; пол кодека "
              f"{r['codec_floor_vs_action']:.5f}")
        print(f"      мишени z_e и z_q расходятся по q1 у "
              f"{100*r['ze_vs_zq_q1_disagree']:.1f}% позиций")
        print(f"    восстановление: q0->q01 {r['recovery_01']}, "
              f"q0->q012 {r['recovery_012']}")
        print(f"    доля улучшившихся: A01 лучше A0 у "
              f"{100 * r['frac_A01_better']:.1f}%, A012 лучше A01 у "
              f"{100 * r['frac_A012_better']:.1f}%")
        print(f"    динамическая цель q1 отличается от истинной: от z_e у "
              f"{100 * r['dynamic_ze_vs_static_q1_disagree']:.1f}%, от z_q у "
              f"{100 * r['dynamic_zq_vs_static_q1_disagree']:.1f}% позиций")

    print("\n  ГЕЙТ 2 (на подтверждающей половине, ЦЕЛИКОМ):")
    cap_ok, act_ok, dyn_ok, bad_cap, bad_act, note = gate(res)
    print(f"\n  РЕШЕНИЕ 1, latent_capacity_ok:              "
          f"{'ДА' if cap_ok else 'НЕТ'}  (книги против потолка кодека)")
    print(f"  РЕШЕНИЕ 2, action_oracle_ok:                "
          f"{'ДА' if act_ok else 'НЕТ'}  (мишени z_e против настоящего "
          f"действия)")
    print(f"  РЕШЕНИЕ 3, dynamic_q1_relabeling_supported: "
          f"{'ДА' if dyn_ok else 'НЕТ'}")
    go = bool(cap_ok and act_ok)
    print(f"\n  ОБУЧАТЬ ГОЛОВЫ: {'ДА' if go else 'НЕТ'} — требуются оба "
          f"первых решения")
    if go and not dyn_ok:
        print("  Но приписывать эффект условной переразметке нельзя: "
              "механизм не подтверждён, статическая цель даёт то же")

    out = dict(parts=res, sampling=sample_meta,
               latent_capacity_ok=bool(cap_ok), action_oracle_ok=bool(act_ok),
               dynamic_q1_relabeling_supported=bool(dyn_ok),
               train_heads=bool(go),
               gate_failures_capacity=bad_cap, gate_failures_action=bad_act,
               gate_notes=note, sample_seed=int(a.seed),
               source_cache=src, source_cache_sha1=sha12(src),
               stamp_k11b=dict(script_sha1=stamp.get("script_sha1"),
                               identity_ok=stamp.get("identity_ok"),
                               arrays=stamp.get("arrays")),
               joint12=src_meta,
               horizon=int(a.horizon), cache=a.cache, ckpt=a.ckpt,
               split_seed=61, sel_frac=float(a.sel_frac),
               target_latents=dict(
                   primary="z_e = codec._encode(action) — мишени тренера",
                   secondary="z_q = sum E_l[k_l] — вопрос о ёмкости книг",
                   references=["decode(z_q) — потолок кодека",
                               "action из кэша K-9a — настоящее действие"]),
               device=str(dev), dtype="float32",
               torch_version=str(torch.__version__),
               cuda_version=str(getattr(torch.version, "cuda", None)),
               probe_rows=int(a.probe_rows), probe_tol=float(a.probe_tol),
               probe_code_disagree=float(dis),
               codebooks_sha1=meta.get("codebooks_sha1"),
               decoder_probe=meta.get("decoder_probe"),
               codec_state_sha1=meta.get("codec_state_sha1"),
               code_version=kb.code_version([
                   os.path.abspath(__file__),
                   os.path.join(here, "depth_rvq_joint12.py")]),
               script_sha1=sha12(os.path.abspath(__file__)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = f"{a.out}.tmp.{os.getpid()}"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, a.out)
    print(f"  сохранено: {a.out}")
    return 0 if go else 4


if __name__ == "__main__":
    sys.exit(main())
