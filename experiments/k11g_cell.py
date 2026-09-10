"""K-11g: одна ячейка окна исследования HiCoRA-G. Один процесс — одна ячейка.

ЯЧЕЙКА = (голова, sigma, задача). Пять эпизодов с общими начальными
состояниями. Полный пилот — 2 головы x 5 sigma x 10 задач = 100 ячеек,
500 эпизодов.

ПОЧЕМУ ОТДЕЛЬНЫЙ ПРОЦЕСС НА ЯЧЕЙКУ, А НЕ ОДИН ПРОГОН. Сто загрузок модели
дороже, но между руками не переносится ничего: ни состояние ГСЧ, ни log_std,
ни состояние среды, ни контекст CUDA. Для одноразового гейта это верный
обмен. Кроме того, ячейки становятся атомарными и возобновляемыми.

СРЕДЫ СОЗДАЮТСЯ ДО МОДЕЛИ. Fork после инициализации CUDA вешает процесс —
это установлено в K-6h и записано в K-9h как «менять нельзя». Порядок здесь
тот же, и от него зависит расход глобального ГСЧ, то есть сами начальные
состояния.

sigma = 0 — ОТДЕЛЬНЫЙ ДЕТЕРМИНИРОВАННЫЙ РЕЖИМ, а не гауссиана с нулевой
дисперсией. Никакого log(0) и никакого фиктивного правдоподобия: в ячейке
пишется mode="deterministic", а log_prob не пишется вовсе. Это опора, от
которой считается падение успеха у своей же головы.

sigma ЗАДАЁТСЯ ПОЛИТИКЕ, А НЕ МАСШТАБИРУЕТ ШУМ СНАРУЖИ. Прежняя версия
строила u = mu + sigma*eps, оставляя головной log_std прежним: действие
сэмплировалось под одним распределением, а log_prob считался под другим. Для
PPO это ядовито. Здесь перед раскаткой выполняется log_std.fill_(log sigma) и
проверяется, что std совпала с sigma.

ОДИН И ТОТ ЖЕ ПОТОК eps ДЛЯ РАЗНЫХ sigma: шум выводится из (задача,
начальное состояние, номер вызова). Иначе ячейки отличались бы и масштабом, и
самой случайностью.

ИЗМЕНЕНИЕ ДЕЙСТВИЯ СЧИТАЕТСЯ ПО КАЖДОМУ АКТИВНОМУ ЧАНКУ ОТДЕЛЬНО. Прежняя
версия брала максимум сразу по пяти средам, шестнадцати позициям и всем
каналам: одно изменение где угодно засчитывало весь батч, и почти любой шум
давал бы около 100%. Здесь: по одной среде, только исполняемые первые H
действий из чанка, разность переводится в реальные единицы до нормировки на
диапазон канала, схват учитывается по знаку, завершившиеся среды из
знаменателя исключаются.

Запуск:
    python3 experiments/k11g_cell.py --selftest
    python experiments/k11g_cell.py --ckpt <hf> --head s0 --sigma 0.10 \\
        --task-id 0 --out data/k11g_smoke/t0_s0_sig0.10.json
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# СЕТКА PILOT-INFORMED, и так и описывается. Smoke на задаче 0 дал при
# sigma=0.10 медианный RMS около 2% диапазона; при линейной зависимости 0.03
# даёт около 0.6% и порога исследования не достигает, а 0.10 уже уронила успех
# одной головы. Точка между ними нужна, поэтому добавлены 0.05 и 0.07.
SIGMAS = (0.0, 0.03, 0.05, 0.07, 0.10, 0.30, 0.50)
# ЗАРЕГИСТРИРОВАННЫЙ ПОРОГ ИССЛЕДОВАНИЯ. Ниже 1% диапазона канала изменение
# сопоставимо с дрожанием самого декодера, и «шум меняет поведение» было бы
# утверждением о численном шуме. Выбран ДО регистрации и из этого основания, а
# не из того, что дала какая-либо sigma.
MIN_RMS = 0.01
CHANGE_LADDER = (0.01, 0.05, 0.10, 0.20)   # диагностика, НЕ критерий
HEADS = ("s0", "s1")
CHANGE_THR = 0.01      # «существенно» — более 1% диапазона канала
SAT_THR = 0.99         # |tanh(u)| выше этого — насыщение
H_EXEC = 8             # исполняемый префикс чанка
PREPROCESS = "CenterCrop(196)->Resize(224)"   # как в K-9h при image_size=224


def eps_seed(task, init, call, salt=0):
    """Сид шума из (задача, начальное состояние, номер вызова).

    НЕ ИЗ ОБЩЕГО ГЕНЕРАТОРА: тогда при разных sigma поток eps совпадает, и
    между ячейками отличается только масштаб шума.
    """
    h = hashlib.sha1(f"k11g|{int(task)}|{int(init)}|{int(call)}|{int(salt)}"
                     .encode()).digest()
    return int.from_bytes(h[:8], "little")


def sigma_mode(sigma):
    """Режим ячейки по sigma. Ноль — ДЕТЕРМИНИРОВАННЫЙ, не вырожденная норма."""
    s = float(sigma)
    if s < 0:
        raise ValueError(f"sigma={s} отрицательна")
    if s == 0.0:
        return ("deterministic", None)
    return ("gaussian", math.log(s))


def chunk_changes(sampled, mean, active, max_act_q, act_range,
                  h_exec=H_EXEC, thr=CHANGE_THR):
    """Изменение действия ПО КАЖДОМУ АКТИВНОМУ ЧАНКУ, а не по батчу.

    `sampled`, `mean` — [n_env, T, 7] в нормированных единицах, как их
    отдаёт декодер. Сравниваются только первые `h_exec` шагов: остальные до
    среды не доходят, и включать их значило бы мерить то, что не исполнялось.

    Разность переводится в РЕАЛЬНЫЕ единицы (умножением на max_act_q, ровно
    как в раскатке) и нормируется на диапазон канала. Нормировать на саму
    величину нельзя: около нуля она взрывается.

    Схват бинарен по смыслу, поэтому в RMS не попадает и учитывается
    отдельно, по смене знака.

    Завершившиеся среды исключаются: их действия никуда не идут, и держать их
    в знаменателе значило бы занижать долю изменённых чанков.
    """
    s = np.asarray(sampled, float)
    m = np.asarray(mean, float)
    if s.shape != m.shape or s.ndim != 3 or s.shape[-1] != 7:
        raise ValueError(f"ожидались одинаковые [n_env, T, 7], получено "
                         f"{s.shape} и {m.shape}")
    if s.shape[1] < h_exec:
        raise ValueError(f"в чанке {s.shape[1]} шагов, а исполняется "
                         f"{h_exec}: мерить нечего")
    a = np.asarray(active, bool)
    if a.shape != (s.shape[0],):
        raise ValueError(f"маска активности {a.shape}, ожидалась "
                         f"({s.shape[0]},)")
    q = np.asarray(max_act_q, float)[:7]
    r = np.asarray(act_range, float)[:7]
    if q.shape != (7,) or r.shape != (7,) or not np.all(r > 0):
        raise ValueError("max_act_q и act_range — положительные векторы по 7 "
                         "каналам")
    out = []
    for i in range(s.shape[0]):
        if not a[i]:
            continue
        ds = s[i, :h_exec]
        dm = m[i, :h_exec]
        d_real = np.abs(ds[:, :-1] - dm[:, :-1]) * q[:-1]
        d_norm = d_real / r[:-1]
        flip = bool(np.any(np.sign(ds[:, -1]) != np.sign(dm[:, -1])))
        mx = float(d_norm.max())
        out.append(dict(env_index=int(i), rms=float(np.sqrt((d_norm ** 2).mean())),
                        max=mx, grip_flip=flip,
                        changed=bool(mx > thr or flip)))
    return out


def rms_median(rows, episodes):
    """RMS: по чанку -> медиана внутри эпизода -> медиана по эпизодам.

    ПОЧЕМУ ДВУХУРОВНЕВО, А НЕ ПЛОСКО. Неудачный эпизод идёт до max_steps и
    даёт втрое больше чанков, чем успешный. Плоская медиана по всем чанкам
    взвесила бы долгие провалы сильнее, и «насколько шум меняет действие»
    описывало бы в основном их.

    Эпизод без активных чанков в расчёт не входит: мерить в нём нечего.
    """
    per_ep = []
    for e in episodes:
        v = [r["rms"] for r in rows
             if r.get("env_index") == e.get("env_index")
             and r.get("rms") is not None]
        if v:
            per_ep.append(float(np.median(v)))
    return (float(np.median(per_ep)) if per_ep else 0.0), per_ep


def summarize(rows, episodes, mode):
    """Сводка ячейки. Считается ТОЛЬКО из сохранённых строк.

    Так сводку можно пересчитать независимо по тому же JSON — ради этого
    построчные данные и сохраняются.
    """
    n = len(rows)
    sat = [r["sat_frac"] for r in rows if r.get("sat_frac") is not None]
    dzf = [r["dz_frac"] for r in rows if r.get("dz_frac") is not None]
    lp = [r["log_prob_u"] for r in rows if r.get("log_prob_u") is not None]
    changed = [r for r in rows if r.get("changed")]
    grip = [r for r in rows if r.get("grip_flip")]
    rms = [r["rms"] for r in rows if r.get("rms") is not None]
    mx = [r["max"] for r in rows if r.get("max") is not None]
    lay = sorted({r["layers_run"] for r in rows})
    return dict(
        mode=mode, chunks=n,
        success=float(np.mean([e["success"] for e in episodes]))
        if episodes else 0.0,
        successes=int(sum(e["success"] for e in episodes)),
        episodes=len(episodes),
        changed_frac=(len(changed) / n) if n else 0.0,
        grip_flip_frac=(len(grip) / n) if n else 0.0,
        rms_median=rms_median(rows, episodes)[0],
        rms_per_episode=rms_median(rows, episodes)[1],
        rms_flat_median=float(np.median(rms)) if rms else 0.0,
        max_p95=float(np.percentile(mx, 95)) if mx else 0.0,
        # ЛЕСТНИЦА ПОРОГОВ — ДИАГНОСТИКА. Доля при 1% насыщается (smoke дал
        # 100% уже при sigma=0.10), поэтому в критерий она не входит.
        changed_frac_ladder={str(t): (float(np.mean(np.asarray(mx) > t))
                                      if mx else 0.0)
                             for t in CHANGE_LADDER},
        sat_frac_mean=float(np.mean(sat)) if sat else 0.0,
        dz_frac_max=float(np.max(dzf)) if dzf else 0.0,
        logp_median=float(np.median(lp)) if lp else None,
        layers_run=lay,
        invariants_ok=bool(lay == [24] and n > 0))


def check_cell_self(cell):
    """Внутренняя непротиворечивость ячейки. Отсутствие поля — отказ.

    Это НЕ сверка с протоколом (она появится отдельным модулем), а проверка
    того, что ячейка не противоречит себе: сводка пересчитывается из строк,
    режим согласован с sigma, правдоподобие есть ровно там, где должно быть.
    """
    bad = []
    for f in ("head", "sigma", "mode", "task_id", "init_start", "n_envs",
              "horizon", "episodes", "chunks", "summary", "parity",
              "rollout_seed", "pos_offset", "preprocess", "eps_salt"):
        if cell.get(f) is None:
            bad.append(f"нет поля {f}")
    if bad:
        raise SystemExit("ЯЧЕЙКА НЕПОЛНА:\n    " + "\n    ".join(bad))
    want_mode, want_log = sigma_mode(cell["sigma"])
    if cell["mode"] != want_mode:
        bad.append(f"mode={cell['mode']!r} при sigma={cell['sigma']}")
    if want_mode == "deterministic":
        if any(r.get("log_prob_u") is not None for r in cell["chunks"]):
            bad.append("в детерминированной ячейке записано правдоподобие")
        if cell.get("log_std_set") is not None:
            bad.append("в детерминированной ячейке задан log_std")
    else:
        if cell.get("log_std_set") is None:
            bad.append("в гауссовой ячейке не записан log_std")
        elif abs(cell["log_std_set"] - want_log) > 1e-9:
            bad.append(f"log_std={cell['log_std_set']} вместо {want_log}")
        if cell.get("std_check") is None or \
                abs(cell["std_check"] - cell["sigma"]) > 1e-6:
            bad.append(f"std политики {cell.get('std_check')} не равна sigma "
                       f"{cell['sigma']}")
        if not any(r.get("log_prob_u") is not None for r in cell["chunks"]):
            bad.append("в гауссовой ячейке нет ни одного правдоподобия")
        # ХЕШ eps ПЕРВОГО ВЫЗОВА. Агрегатор потребует его совпадения у всех
        # ненулевых sigma: это прямая проверка общего потока шума. Восстанав-
        # ливать eps через (u-mu)/sigma значило бы сравнивать с ошибками
        # округления.
        if not cell.get("eps_sha1"):
            bad.append("в гауссовой ячейке нет eps_sha1: общий поток шума "
                       "нечем подтвердить")
    if want_mode == "deterministic" and cell.get("eps_sha1"):
        bad.append("в детерминированной ячейке записан eps_sha1")
    if len(cell["episodes"]) != int(cell["n_envs"]):
        bad.append(f"эпизодов {len(cell['episodes'])} при n_envs="
                   f"{cell['n_envs']}")
    ids = [e.get("init_state_id") for e in cell["episodes"]]
    want_ids = [cell["init_start"] + i for i in range(int(cell["n_envs"]))]
    if ids != want_ids:
        bad.append(f"init_state_id {ids} вместо {want_ids}")
    if any(not e.get("init_hash_full") for e in cell["episodes"]):
        bad.append("нет init_hash_full: общие начальные состояния не "
                   "подтверждены")
    # КЛЮЧ ПАРЫ ПИШЕТ ВОРКЕР, дискордантность считает агрегатор: воркер видит
    # только свою руку и сравнивать ему не с чем.
    for e in cell["episodes"]:
        want_k = f"{cell.get('suite')}|{cell['task_id']}|{e['init_state_id']}"
        if e.get("pair_key") != want_k:
            bad.append(f"pair_key {e.get('pair_key')!r} вместо {want_k!r}")
            break
    rec = summarize(cell["chunks"], cell["episodes"], cell["mode"])
    for k in ("chunks", "successes", "changed_frac", "grip_flip_frac",
              "sat_frac_mean", "dz_frac_max", "invariants_ok"):
        a, b = rec[k], cell["summary"].get(k)
        same = (a == b) if isinstance(a, (bool, int)) else \
            (b is not None and abs(float(a) - float(b)) < 1e-9)
        if not same:
            bad.append(f"сводка не пересчитывается: {k} {b} против {a}")
    if not cell["parity"].get("ok"):
        bad.append("паритет с D1 не сошёлся")
    if bad:
        raise SystemExit("ЯЧЕЙКА ПРОТИВОРЕЧИТ СЕБЕ:\n    "
                         + "\n    ".join(bad))
    return True


def selftest():
    # --- поток шума ---------------------------------------------------------
    assert eps_seed(3, 10, 7) == eps_seed(3, 10, 7)
    for other in ((3, 10, 8), (3, 11, 7), (4, 10, 7)):
        assert eps_seed(3, 10, 7) != eps_seed(*other), other
    assert eps_seed(3, 10, 7) != eps_seed(3, 10, 7, salt=1)
    assert len({eps_seed(t, i, c) for t in range(10) for i in range(5)
                for c in range(80)}) == 10 * 5 * 80

    # --- режим по sigma -----------------------------------------------------
    assert sigma_mode(0.0) == ("deterministic", None)
    m, l = sigma_mode(0.10)
    assert m == "gaussian" and abs(l - math.log(0.10)) < 1e-12
    try:
        sigma_mode(-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("отрицательная sigma принята")

    # --- изменение по чанкам ------------------------------------------------
    Q = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0])
    R = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0])
    base = np.zeros((3, 20, 7))
    act = np.array([True, True, True])
    assert chunk_changes(base, base, act, Q, R) == [
        dict(env_index=i, rms=0.0, max=0.0, grip_flip=False, changed=False)
        for i in range(3)]
    # ПО КАЖДОЙ СРЕДЕ ОТДЕЛЬНО: изменение в одной не красит остальные.
    s1 = base.copy(); s1[1, 0, 0] = 0.10      # 0.10*1.0/2.0 = 5% диапазона
    rows = chunk_changes(s1, base, act, Q, R)
    assert [r["changed"] for r in rows] == [False, True, False], rows
    # ТОЛЬКО ПЕРВЫЕ H ШАГОВ: изменение в хвосте чанка не исполнялось.
    s2 = base.copy(); s2[0, H_EXEC, 0] = 1.0
    assert not chunk_changes(s2, base, act, Q, R)[0]["changed"]
    s3 = base.copy(); s3[0, H_EXEC - 1, 0] = 1.0
    assert chunk_changes(s3, base, act, Q, R)[0]["changed"]
    # РЕАЛЬНЫЕ ЕДИНИЦЫ ДО НОРМИРОВКИ: тот же сдвиг в канале с малым max_act_q
    # даёт меньшую долю диапазона.
    s4 = base.copy(); s4[0, 0, 3] = 0.10      # 0.10*0.5/1.0 = 5%
    s5 = base.copy(); s5[0, 0, 0] = 0.01      # 0.01*1.0/2.0 = 0.5%
    assert chunk_changes(s4, base, act, Q, R)[0]["changed"]
    assert not chunk_changes(s5, base, act, Q, R)[0]["changed"]
    # РАЗЛИЧАЮЩИЙ СЛУЧАЙ: канал с max_act_q=0.5 и диапазоном 1.0, сдвиг 0.015.
    # В реальных единицах это 0.75% и НЕ изменение; если множитель забыть —
    # получится 1.5% и ложное изменение. Без этого случая пропуск умножения
    # на max_act_q тестом не ловился.
    s6 = base.copy(); s6[0, 0, 3] = 0.015
    r6 = chunk_changes(s6, base, act, Q, R)[0]
    assert not r6["changed"], r6
    assert abs(r6["max"] - 0.0075) < 1e-12, r6["max"]
    # СХВАТ: смена знака — изменение поведения, но в RMS не попадает.
    g = base.copy(); g[:, :, -1] = 1.0
    gm = base.copy(); gm[:, :, -1] = -1.0
    rg = chunk_changes(g, gm, act, Q, R)
    assert all(r["grip_flip"] and r["changed"] and r["rms"] == 0.0 for r in rg)
    # ЗАВЕРШИВШИЕСЯ СРЕДЫ ВНЕ ЗНАМЕНАТЕЛЯ
    rows = chunk_changes(s1, base, np.array([False, True, False]), Q, R)
    assert len(rows) == 1 and rows[0]["env_index"] == 1
    assert chunk_changes(base, base, np.zeros(3, bool), Q, R) == []
    for bad, why in (((base, base[:, :, :6], act, Q, R), "каналов не 7"),
                     ((base[:, :4], base[:, :4], act, Q, R), "чанк короче H"),
                     ((base, base, np.array([True]), Q, R), "маска не та"),
                     ((base, base, act, Q, np.zeros(7)), "нулевой диапазон")):
        try:
            chunk_changes(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"принято: {why}")

    # --- сводка и самопроверка ячейки --------------------------------------
    def mk(mode="gaussian", sigma=0.10, n=5, nch=12, bad=None, suite="10"):
        eps_rows = [dict(env_index=i, init_state_id=i,
                         pair_key=f"{suite}|0|{i}", init_hash="a",
                         init_hash_full="b", success=(i < 4), env_steps=80,
                         policy_calls=nch) for i in range(n)]
        ch = []
        for c in range(nch):
            ch.append(dict(call=c, env_index=c % n, rms=0.02, max=0.05,
                           changed=(c % 3 == 0), grip_flip=False,
                           sat_frac=0.01, dz_frac=0.3, layers_run=24,
                           log_prob_u=(-12.0 if mode == "gaussian" else None)))
        cell = dict(head="s0", sigma=sigma, mode=mode, task_id=0, suite=suite,
                    init_start=0, n_envs=n, horizon=H_EXEC, eps_salt=0,
                    rollout_seed=7, pos_offset=4, preprocess=PREPROCESS,
                    log_std_set=(math.log(sigma) if mode == "gaussian"
                                 else None),
                    std_check=(sigma if mode == "gaussian" else None),
                    eps_sha1=("abc123" if mode == "gaussian" else None),
                    episodes=eps_rows, chunks=ch,
                    parity=dict(ok=True, gauss_mean_vs_d1=0.0,
                                d1_vs_forward_hicora=0.0, layers_run=24))
        cell["summary"] = summarize(ch, eps_rows, mode)
        if bad:
            cell.update(bad)
        return cell

    ok = mk()
    assert ok["summary"]["changed_frac"] == 4 / 12
    assert ok["summary"]["successes"] == 4 and ok["summary"]["success"] == 0.8
    assert ok["summary"]["invariants_ok"]
    assert check_cell_self(ok)
    det = mk(mode="deterministic", sigma=0.0)
    assert det["summary"]["logp_median"] is None
    assert check_cell_self(det)

    # СИНТЕТИЧЕСКИЕ ПОРЧИ: каждая обязана быть отвергнута.
    cases = {
        "нет eps_sha1 у гауссовой": dict(eps_sha1=None),
        "eps_sha1 у детерминированной": dict(mode="deterministic", sigma=0.0,
                                             eps_sha1="abc", log_std_set=None,
                                             std_check=None),
        "режим не тот": dict(mode="deterministic"),
        "log_std не тот": dict(log_std_set=0.0),
        "std не равна sigma": dict(std_check=0.5),
        "паритет не сошёлся": dict(parity=dict(ok=False)),
        "эпизодов не столько": dict(n_envs=7),
        "нет поля": dict(pos_offset=None),
    }
    for why, mut in cases.items():
        try:
            check_cell_self(mk(bad=mut))
        except SystemExit:
            pass
        else:
            raise AssertionError(f"принято: {why}")
    # СВОДКА, НЕ СОГЛАСОВАННАЯ СО СТРОКАМИ
    c = mk()
    c["summary"] = dict(c["summary"], changed_frac=0.99)
    try:
        check_cell_self(c)
    except SystemExit:
        pass
    else:
        raise AssertionError("подделанная сводка принята")
    # ПРАВДОПОДОБИЕ В ДЕТЕРМИНИРОВАННОЙ ЯЧЕЙКЕ
    c = mk(mode="deterministic", sigma=0.0)
    c["chunks"][0]["log_prob_u"] = -1.0
    c["summary"] = summarize(c["chunks"], c["episodes"], c["mode"])
    try:
        check_cell_self(c)
    except SystemExit:
        pass
    else:
        raise AssertionError("фиктивное правдоподобие принято")
    # ПЕРЕПУТАННЫЕ init_state_id
    c = mk()
    c["episodes"][2]["init_state_id"] = 99
    try:
        check_cell_self(c)
    except SystemExit:
        pass
    else:
        raise AssertionError("чужой init_state_id принят")
    # ОТСУТСТВУЮЩИЙ ХЕШ СОСТОЯНИЯ
    c = mk()
    c["episodes"][1]["init_hash_full"] = ""
    try:
        check_cell_self(c)
    except SystemExit:
        pass
    else:
        raise AssertionError("эпизод без init_hash_full принят")
    # ЧУЖОЙ КЛЮЧ ПАРЫ: дискордантность считается по нему, и подмена сломала
    # бы сопоставление у агрегатора молча.
    c = mk()
    c["episodes"][2]["pair_key"] = "10|0|99"
    try:
        check_cell_self(c)
    except SystemExit:
        pass
    else:
        raise AssertionError("чужой pair_key принят")

    # --- RMS: ДВУХУРОВНЕВАЯ АГРЕГАЦИЯ --------------------------------------
    # Долгий неудачный эпизод даёт больше чанков. Плоская медиана взвесила бы
    # его сильнее; двухуровневая — нет. Тест различает эти два способа.
    eps2 = [dict(env_index=0, init_state_id=0, pair_key="10|0|0",
                 init_hash="a", init_hash_full="b", success=True,
                 env_steps=10, policy_calls=1),
            dict(env_index=1, init_state_id=1, pair_key="10|0|1",
                 init_hash="a", init_hash_full="b", success=False,
                 env_steps=90, policy_calls=9)]
    rows2 = ([dict(env_index=0, rms=0.10, max=0.2, changed=True,
                   grip_flip=False, sat_frac=0.0, dz_frac=0.1, layers_run=24,
                   log_prob_u=-1.0)]
             + [dict(env_index=1, rms=0.02, max=0.05, changed=True,
                     grip_flip=False, sat_frac=0.0, dz_frac=0.1,
                     layers_run=24, log_prob_u=-1.0) for _ in range(9)])
    med, per = rms_median(rows2, eps2)
    assert per == [0.10, 0.02], per
    assert abs(med - 0.06) < 1e-12, med         # медиана двух медиан
    flat = float(np.median([r["rms"] for r in rows2]))
    assert abs(flat - 0.02) < 1e-12, flat       # плоская утонула бы в провале
    assert abs(med - flat) > 0.03, "тест не различает два способа"
    # ЭПИЗОД БЕЗ ЧАНКОВ НЕ УЧИТЫВАЕТСЯ
    eps3 = eps2 + [dict(env_index=2, init_state_id=2, pair_key="10|0|2",
                        init_hash="a", init_hash_full="b", success=True,
                        env_steps=0, policy_calls=0)]
    assert rms_median(rows2, eps3)[1] == [0.10, 0.02]
    assert rms_median([], eps2) == (0.0, [])

    # --- ЛЕСТНИЦА ПОРОГОВ — ДИАГНОСТИКА, И ОНА НАСЫЩАЕТСЯ ------------------
    # Smoke дал 100% при пороге 1% уже на sigma=0.10, поэтому доля в критерий
    # не входит. Здесь фиксируется, что лестница считается по всем порогам.
    sm = summarize(rows2, eps2, "gaussian")
    # СВОДКА ОБЯЗАНА БРАТЬ ДВУХУРОВНЕВУЮ МЕДИАНУ, а не плоскую. Без этой
    # проверки подмена внутри summarize проходила бы: сама rms_median
    # тестировалась отдельно и оставалась исправной.
    assert abs(sm["rms_median"] - 0.06) < 1e-12, sm["rms_median"]
    assert abs(sm["rms_flat_median"] - 0.02) < 1e-12, sm["rms_flat_median"]
    assert sm["rms_per_episode"] == [0.10, 0.02], sm["rms_per_episode"]
    lad = sm["changed_frac_ladder"]
    assert set(lad) == {str(t) for t in CHANGE_LADDER}, lad
    assert lad["0.01"] == 1.0 and lad["0.1"] == 0.1, lad
    assert MIN_RMS == 0.01

    # НЕ ТЕ СЛОИ
    c = mk()
    c["chunks"][3]["layers_run"] = 12
    c["summary"] = summarize(c["chunks"], c["episodes"], c["mode"])
    assert not c["summary"]["invariants_ok"]

    print("самопроверка k11g_cell пройдена: поток шума одинаков по sigma и "
          "различен\n  по вызову; sigma=0 — отдельный детерминированный режим "
          "без log(0) и без\n  фиктивного правдоподобия; изменение считается "
          "по каждому активному чанку,\n  только по первым H шагам, в "
          "реальных единицах, со схватом отдельно;\n  сводка пересчитывается "
          "из строк; RMS агрегируется чанк->эпизод->эпизоды, и тест "
          "различает это\n  от плоской медианы; лестница порогов — "
          "диагностика; восемь синтетических\n  порч ячейки отвергаются, "
          "включая отсутствующий eps_sha1 и чужой pair_key")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--policy-ckpt", default="data/k9d_ep3.pt")
    ap.add_argument("--hicora-s0",
                    default="data/k11d/d1_mlp_coef_0.001_wd0_s0.pt")
    ap.add_argument("--hicora-s1",
                    default="data/k11d/d1_mlp_coef_0.001_wd0_s1.pt")
    ap.add_argument("--head", choices=HEADS, required=False)
    ap.add_argument("--sigma", type=float, required=False)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--task-suite", default="10")
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--init-start", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=H_EXEC)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--waiting-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout-seed-mode", default="block",
                    choices=["block", "fixed"])
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--pos-offset", type=int, default=None)
    ap.add_argument("--expect-depth", type=int, default=12)
    ap.add_argument("--expect-hicora-target", default="coef")
    ap.add_argument("--eps-salt", type=int, default=0,
                    help="пространство шумовых сидов. Зарегистрированный "
                         "прогон обязан идти с другой солью, чем пилот: "
                         "иначе он переиспользовал бы тот же шум на тех же "
                         "номерах вызовов")
    ap.add_argument("--run-tag", default="k11g")
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    for need in ("ckpt", "head", "sigma", "out"):
        if getattr(args, need) in (None, ""):
            raise SystemExit(f"нужен --{need.replace('_', '-')}")

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"нет каталога ActionCodec: {root}")
    sys.path.insert(0, root)

    mode, log_sigma = sigma_mode(args.sigma)
    # СМЕЩЕНИЕ ПОЗИЦИЙ ИЗ ТАБЛИЦЫ, как в K-11e. Зашитое значение сделало бы
    # политику другой, и успех нельзя было бы сопоставлять с прежним гейтом.
    if args.pos_offset is not None:
        pos_off = int(args.pos_offset)
        off_sha = None
    else:
        if not os.path.exists(args.offset_table):
            raise SystemExit(f"нет {args.offset_table}; задайте --pos-offset")
        tb = json.load(open(args.offset_table))
        pos_off = int(tb["offsets_by_suite"][args.task_suite][args.task_id])
        off_sha = hashlib.sha1(
            open(args.offset_table, "rb").read()).hexdigest()[:12]

    # ЧЕКПОЙНТЫ ЧИТАЮТСЯ ДО СРЕД: torch.load с map_location="cpu" CUDA не
    # инициализирует, зато сообщение об отказе не тонет в предупреждениях
    # robosuite от пяти процессов сред.
    import torch
    import k9h_multiarm_gate as k9h
    import k11e_protocol as kp

    head_path = {"s0": args.hicora_s0, "s1": args.hicora_s1}[args.head]
    other_path = {"s0": args.hicora_s1, "s1": args.hicora_s0}[args.head]
    h_obj = torch.load(head_path, map_location="cpu", weights_only=False)
    k9h.check_hicora_ckpt(h_obj, f"hicora_{args.head}",
                          args.expect_hicora_target)
    o_obj = torch.load(other_path, map_location="cpu", weights_only=False)
    head_sha = k9h.file_sha12(head_path)
    if head_sha == k9h.file_sha12(other_path):
        raise SystemExit("обе головы — один файл: это не две руки")
    # РЕПЛИКАЦИЯ СВЕРЯЕТСЯ В КАЖДОЙ ЯЧЕЙКЕ, а не один раз в прогоне: ячейка
    # должна быть самодостаточной для аудита.
    kp.check_replication(kp.head_config(o_obj if args.head == "s1" else h_obj),
                         kp.head_config(h_obj if args.head == "s1" else o_obj))
    j_obj = torch.load(args.policy_ckpt, map_location="cpu",
                       weights_only=False)
    joint_sha = k9h.file_sha12(args.policy_ckpt)

    from torchvision.transforms.v2 import CenterCrop, Compose, Resize

    import actioncodec  # noqa: F401
    import joint12_vla as jv
    from joint12_vla import make_joint12_class
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       VisionLanguageActionProcessor, dict_apply, get_cfg,
                       get_envs, process_state, prompt_template,
                       seed_everything)
    import hicora_vla as hv
    import hicora_g as hg
    import k11a_build_hicora_cache as k11a

    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt

    # --- СРЕДЫ ДО МОДЕЛИ. Порядок и расход ГСЧ те же, что в K-9h ------------
    roll_seed = k9h.rollout_seed(args.seed, args.init_start,
                                 args.rollout_seed_mode)
    seed_everything(roll_seed)
    envs, task_desc = get_envs(args.task_suite,
                               {"task_id": args.task_id, "image_size": 224},
                               args.n_envs)
    print(f"  среды созданы до модели: задача {args.task_id}, сред "
          f"{args.n_envs}, сид раскатки {roll_seed}", flush=True)

    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    # ТО ЖЕ ПРЕОБРАЗОВАНИЕ, ЧТО В K-9h, при кадре среды 224. Вариант из K-9i
    # (Resize(512)->CenterCrop(512)) относится к кадрам ДАТАСЕТА и дал бы
    # другую политику.
    tf = Compose([CenterCrop(int(224 * 0.875)), Resize(224)])
    max_act_q = np.maximum(np.abs(ACTION_Q01), np.abs(ACTION_Q99))
    act_range = np.asarray(ACTION_Q99, float) - np.asarray(ACTION_Q01, float)
    if not np.all(act_range[:7] > 0):
        raise SystemExit("диапазон действия не положителен")

    import copy
    Cls = make_joint12_class(SmolVLABlockwiseAR)
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
    res_norm_orig = copy.deepcopy(model.action_expert.norm)
    model.init_joint_fast(depth=args.expect_depth, head_dtype=dtype)
    own = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in j_obj["state"].items():
            if k not in own:
                raise SystemExit(f"ключ вне модели: {k}")
            own[k].data = v.to(dev, torch.float32)
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None))
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь")
    codec = codec.to(dev).eval()
    with torch.no_grad():
        ii = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(ii))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    model.__class__ = hv.make_hicora_class(type(model))
    model.set_codebooks(E)
    model.set_res_norm(res_norm_orig.to(dev))
    model.taps, model.q0_depth = (12, 18, 24), args.expect_depth
    model.n_layers_total = len(model.action_expert.layers)
    if max(model.taps) != model.n_layers_total:
        raise SystemExit(f"последний отвод {max(model.taps)} против "
                         f"{model.n_layers_total} слоёв")
    rn = hashlib.sha1()
    for k_ in sorted(model.res_norm.state_dict()):
        v_ = model.res_norm.state_dict()[k_]
        rn.update(k_.encode())
        rn.update(np.ascontiguousarray(
            v_.detach().float().cpu().numpy()).tobytes())
    rn_sha = rn.hexdigest()[:12]
    if rn_sha != h_obj["res_norm_sha1"]:
        raise SystemExit(f"res_norm sha {rn_sha}, голова обучена на "
                         f"{h_obj['res_norm_sha1']}")
    pref = h_obj["cache"]
    bp, rp, mp = pref + ".basis.npy", pref + ".rho.npy", pref + ".meta.json"
    for f_ in (bp, rp, mp):
        if not os.path.exists(f_):
            raise SystemExit(f"нет {f_}")
    for f_, want_, lbl in ((bp, h_obj["basis_sha1"], "базис"),
                           (rp, h_obj["rho_sha1"], "предел")):
        got_ = k9h.file_sha12(f_)
        if got_ != want_:
            raise SystemExit(f"{lbl} sha {got_}, голова обучена на {want_}")
    meta = json.load(open(mp))
    k9h.check_hicora_meta(meta, args.ckpt, joint_sha,
                          k9h.file_sha12(hv.__file__),
                          k9h.file_sha12(jv.__file__))
    k11a.check_fingerprints(meta, dict(
        codebooks_sha1=hashlib.sha1(np.ascontiguousarray(
            E.cpu().numpy().astype(np.float32)).tobytes()).hexdigest()[:12],
        decoder_probe=k11a.decoder_probe(codec, E, dev),
        codec_state_sha1=k11a.state_sha1(codec)))
    B = np.load(bp).astype(np.float32)
    rho = np.load(rp).astype(np.float32)
    d_h = int(model.fast_head.in_features)
    kw = dict(rank=int(h_obj["rank"]), hidden=int(h_obj.get("hidden", 512)),
              proj=int(h_obj.get("proj", 64)))
    st_h = {k[len("hicora_head."):]: v for k, v in h_obj["state"].items()}
    stray = [k for k in h_obj["state"] if not k.startswith("hicora_head.")]
    if stray:
        raise SystemExit(f"ключи вне hicora_head.: {stray[:5]}")
    heads = {}
    for which, cls in (("det", hv.make_residual_head()),
                       ("gau", hg.make_gaussian_residual_head())):
        h_ = cls(d_h, int(E.shape[-1]), **kw).to(dev)
        h_.set_basis(torch.as_tensor(B))
        h_.set_rho(torch.as_tensor(rho))
        want = {k for k in h_.state_dict() if k.startswith(("proj.", "net."))}
        if set(st_h) != want:
            raise SystemExit(f"набор весов головы не совпал ({which})")
        with torch.no_grad():
            for k, v in st_h.items():
                h_.state_dict()[k].copy_(v.to(dev, torch.float32))
        h_.eval()
        heads[which] = h_
    det_h, gau_h = heads["det"], heads["gau"]
    rho_norm = float(np.linalg.norm(rho))

    # --- sigma ЗАДАЁТСЯ ПОЛИТИКЕ -------------------------------------------
    std_check = None
    if mode == "gaussian":
        with torch.no_grad():
            gau_h.log_std.fill_(float(log_sigma))
        std_check = float(gau_h.std().max())
        if abs(std_check - args.sigma) > 1e-6 or \
                float(gau_h.std().min()) != std_check:
            raise SystemExit(
                f"std политики {std_check} не равна sigma {args.sigma}: "
                f"действие сэмплировалось бы под одним распределением, а "
                f"правдоподобие считалось под другим")
        print(f"  режим gaussian: log_std={log_sigma:.6f}, std={std_check:.6f}",
              flush=True)
    else:
        print("  режим deterministic: шума нет, правдоподобие не пишется",
              flush=True)

    ac16 = torch.autocast("cuda", dtype=torch.float16)
    parity = {"ok": False}

    def decode_latent(z):
        x, _ = codec._decode(z.float(), embodiment_ids=0)
        return x[..., :7].detach().float().cpu().numpy()

    chunks = []
    eps_sha = [None]
    t0 = time.time()
    try:
        n = args.n_envs
        obs = envs.reset(options=[{"init_state_id": args.init_start + i}
                                  for i in range(n)])
        reward = np.zeros(n)
        done = np.zeros(n, bool)
        dummy = np.array([[0, 0, 0, 0, 0, 0, -1]] * n)
        for _ in range(args.waiting_steps):
            obs, r_, done, _ = envs.step(dummy)
            reward = np.clip(reward + r_, 0, 1)

        def _h(parts):
            return hashlib.sha1(np.ascontiguousarray(
                np.concatenate(parts).astype(np.float32)).tobytes()
            ).hexdigest()[:16]
        init_hash = [_h([obs["state"][i].ravel(),
                         obs["agentview_image"][i].ravel() / 255.0])
                     for i in range(n)]
        init_hash_full = [
            _h([obs["state"][i].ravel(),
                obs["agentview_image"][i].ravel() / 255.0,
                obs["robot0_eye_in_hand_image"][i].ravel() / 255.0])
            for i in range(n)]

        calls = steps = 0
        while not np.all(done) and steps < args.max_steps:
            state = ((process_state(obs["state"]) - STATE_Q01)
                     / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0)
            i1 = tf(torch.tensor(
                obs["agentview_image"][:, :, ::-1].copy()).permute(0, 3, 1, 2))
            i2 = tf(torch.tensor(
                obs["robot0_eye_in_hand_image"][:, :, ::-1].copy()
            ).permute(0, 3, 1, 2))
            image = torch.cat([i1, i2], dim=-1)
            msgs = []
            for i in range(n):
                m = prompt_template(
                    state[i], None, task_desc,
                    mode=cfg.MODEL.vla_processor.kwargs.mode,
                    action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                    action_token_len=cfg.MODEL.action_processor.token_len)
                m[1]["content"] = m[1]["content"][1:]
                msgs.append(m)
            texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
            batch = proc(text=texts,
                         images=[[image[i].numpy()] for i in range(n)],
                         return_tensors="pt", padding=True,
                         padding_side="left",
                         action_processor_kwargs={"embodiment_ids": 0})
            batch = dict_apply(lambda x: x.to(dev, dtype), batch)
            active = ~done.copy()
            with torch.no_grad(), ac16:
                v_, p_ = model.build_inputs(position_offset=pos_off, **batch)
                taps = model.forward_taps(
                    vlm_inputs_embeds=v_,
                    attention_mask=batch.get("attention_mask"),
                    position_ids=p_)
                n_lay = int(taps["layers_run"])
                _, q0 = model.q0_from(taps[model.q0_depth])
                z0 = model.codebooks[0][q0]
                h24 = model.res_norm(taps[max(model.taps)]).float()
                o_mean = gau_h(h24, z0, deterministic=True)
                if mode == "deterministic":
                    o_exec = o_mean
                else:
                    gen = torch.Generator(device=h24.device)
                    gen.manual_seed(eps_seed(args.task_id, args.init_start,
                                             calls, salt=args.eps_salt)
                                    % (2 ** 63))
                    eps = torch.empty_like(o_mean["mu"]).normal_(generator=gen)
                    if eps_sha[0] is None:
                        # ХЕШ САМОГО eps, а не восстановленного (u-mu)/sigma:
                        # восстановление внесло бы ошибки округления, и
                        # сравнение потоков между sigma стало бы приблизи-
                        # тельным там, где оно обязано быть точным.
                        eps_sha[0] = hashlib.sha1(np.ascontiguousarray(
                            eps.detach().float().cpu().numpy()
                        ).tobytes()).hexdigest()[:16]
                    o_exec = gau_h(h24, z0, u=o_mean["mu"]
                                   + float(args.sigma) * eps)
                # ПАРИТЕТ В КАЖДОЙ ЯЧЕЙКЕ, на НАСТОЯЩЕМ D1 и под autocast.
                if not parity["ok"]:
                    dz_d, _ = det_h(h24, z0)
                    model.hicora_head = det_h
                    o_fw = model.forward_hicora(
                        vlm_inputs_embeds=v_,
                        attention_mask=batch.get("attention_mask"),
                        position_ids=p_)
                    d_head = float((o_mean["dz"] - dz_d).abs().max())
                    d_full = float((dz_d - o_fw["dz"]).abs().max())
                    parity = dict(gauss_mean_vs_d1=d_head,
                                  d1_vs_forward_hicora=d_full,
                                  layers_run=n_lay,
                                  ok=bool(d_head <= 1e-4 and d_full <= 1e-4
                                          and n_lay == 24))
                    print(f"  паритет: гауссова(mean) против D1 {d_head:.2e}, "
                          f"D1 против forward_hicora {d_full:.2e}, слоёв "
                          f"{n_lay}", flush=True)
                    if not parity["ok"]:
                        raise SystemExit(
                            "ПАРИТЕТ НЕ СОШЁЛСЯ: в режиме среднего гауссова "
                            "голова обязана давать\n  ровно то же, что D1, "
                            "иначе сравнение с K-11e бессмысленно")
                if not torch.isfinite(o_exec["dz"]).all():
                    raise SystemExit("в поправке nan или inf")
                dzn = float(torch.linalg.norm(o_exec["dz"], dim=-1).max())
                if dzn > rho_norm + 1e-4:
                    raise SystemExit(f"||dz|| = {dzn:.4f} превысила предел "
                                     f"{rho_norm:.4f}")
                sat = float((o_exec["coeffs"].abs() > SAT_THR).float().mean())
                lp = (None if mode == "deterministic"
                      else float(o_exec["log_prob_u"].mean()))
                a_exec = decode_latent(z0 + o_exec["dz"])
                a_mean = (a_exec if mode == "deterministic"
                          else decode_latent(z0 + o_mean["dz"]))
            for row in chunk_changes(a_exec, a_mean, active, max_act_q,
                                     act_range, h_exec=args.horizon):
                row.update(call=calls, sat_frac=sat, dz_frac=dzn / rho_norm,
                           layers_run=n_lay, log_prob_u=lp)
                chunks.append(row)
            calls += 1
            action = np.copy(a_exec)
            action[..., :-1] = action[..., :-1] * max_act_q[..., :-1]
            action[..., -1] = -action[..., -1]
            for t in range(args.horizon):
                if np.all(done) or steps >= args.max_steps:
                    break
                obs, r_, done, _ = envs.step(action[:, t])
                reward = np.clip(reward + r_, 0, 1)
                steps += 1
        eps_rows = [dict(env_index=i, init_state_id=args.init_start + i,
                         pair_key=f"{args.task_suite}|{args.task_id}|"
                                  f"{args.init_start + i}",
                         init_hash=init_hash[i],
                         init_hash_full=init_hash_full[i],
                         success=bool(reward[i] >= 1.0), env_steps=steps,
                         policy_calls=calls, rollout_seed=roll_seed)
                    for i in range(n)]
    finally:
        envs.close()

    cell = dict(
        run_tag=args.run_tag, head=args.head, sigma=float(args.sigma),
        mode=mode, log_std_set=(None if log_sigma is None else float(log_sigma)),
        std_check=std_check, task_id=args.task_id,
        task_description=task_desc, suite=args.task_suite,
        init_start=args.init_start, n_envs=args.n_envs,
        horizon=args.horizon, max_steps=args.max_steps,
        waiting_steps=args.waiting_steps, ensemble="off",
        pos_offset=pos_off, offset_table_sha1=off_sha,
        preprocess=PREPROCESS, image_size=224,
        seed=args.seed, rollout_seed=roll_seed, eps_salt=args.eps_salt,
        eps_sha1=eps_sha[0], min_rms=MIN_RMS,
        rollout_seed_mode=args.rollout_seed_mode,
        ckpt=args.ckpt, joint_sha1=joint_sha, head_sha1=head_sha,
        res_norm_sha1=rn_sha, basis_sha1=h_obj["basis_sha1"],
        rho_sha1=h_obj["rho_sha1"], rho_norm=rho_norm,
        rank=int(h_obj["rank"]), hicora_seed=h_obj.get("seed"),
        selected_epoch=h_obj.get("selected_epoch"),
        script_sha1=k9h.file_sha12(os.path.abspath(__file__)),
        hicora_g_sha1=k9h.file_sha12(hg.__file__),
        hicora_vla_sha1=k9h.file_sha12(hv.__file__),
        joint12_vla_sha1=k9h.file_sha12(jv.__file__),
        k9h_sha1=k9h.file_sha12(k9h.__file__),
        device=str(dev), dtype=args.dtype, parity=parity,
        episodes=eps_rows, chunks=chunks, minutes=(time.time() - t0) / 60.0)
    cell["summary"] = summarize(chunks, eps_rows, mode)
    check_cell_self(cell)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    tmp = args.out + ".tmp"
    json.dump(cell, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)          # АТОМАРНО: оборванная запись не
    s = cell["summary"]                # оставит полуячейку
    lad = " ".join(f"{k}:{100 * v:.0f}%"
                   for k, v in sorted(s["changed_frac_ladder"].items(),
                                      key=lambda kv: float(kv[0])))
    print(f"\n  {args.head}, sigma={args.sigma}, задача {args.task_id}: "
          f"успех {s['successes']}/{s['episodes']}, чанков {s['chunks']}")
    print(f"    RMS медиана по эпизодам {s['rms_median']:.4f} "
          f"(порог исследования {MIN_RMS}), плоская {s['rms_flat_median']:.4f}")
    print(f"    лестница изменения {lad}  — ДИАГНОСТИКА, не критерий")
    print(f"    схват {100 * s['grip_flip_frac']:.1f}%, насыщение "
          f"{100 * s['sat_frac_mean']:.2f}%, ||dz||/||rho|| макс "
          f"{s['dz_frac_max']:.3f}, медиана log pi {s['logp_median']}")
    print(f"    eps первого вызова sha {cell['eps_sha1']}, соль "
          f"{args.eps_salt}")
    print(f"  сохранено: {args.out} ({cell['minutes']:.1f} мин)")


if __name__ == "__main__":
    main()
