#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14b: канонический кэш условных меток q1*. Один режим, один файл, один sha.

КАНОНИЧЕСКИЙ РЕЖИМ ОГРАНИЧИВАЕТ ПОСТРОЕНИЕ МЕТОК, А НЕ ОБУЧЕНИЕ. Метки
строятся один раз в одном режиме и кладутся целыми числами; дальше тренер
читает готовые числа, и на каком устройстве он учится — безразлично.

ЗАЧЕМ КАНОНИЧЕСКИЙ КЭШ, А НЕ ПЕРЕСЧЁТ В ТРЕНЕРЕ. Метка q1* выбирается как
ближайший код к остатку z_e - E0[q0hat], и у части позиций два кода почти
равноудалены: измеренный разрыв там порядка 5e-08 при величинах 1e-02, то есть
несколько единиц последнего разряда fp32 (§42). Пересчитывай тренер метки сам,
два training seed на cuda:0 и cuda:1 получили бы РАЗНЫЕ задачи, и разность
между ними нельзя было бы приписать порядку данных. Поэтому метки строятся
ОДИН РАЗ в одном режиме, кладутся целыми числами, и обе головы читают один
файл с одним отпечатком.

ЧТО ЗДЕСЬ ЕСТЬ И ЧЕГО НЕТ. Только q1*. Оракульных q2* здесь НЕТ намеренно:
обучать q2 надо относительно ФАКТИЧЕСКИ предсказанного замороженной головой
q1, а не относительно оракульного. Кэш q2 строится позже и отдельно для каждой
обученной головы — иначе вернётся teacher-forcing mismatch, ради устранения
которого всё и делается.

ЧАСТИ. train, val_sel, val_confirm — теми же строками и тем же разбиением по
эпизодам (сид 61, доля 0.4), что в K-11c, K-13b и K-14a. Финальная выборка не
читается вовсе.
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


def arr_sha(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def check_codes(q1, vocab, q0hat, ktrue):
    """Метки обязаны быть целыми, в диапазоне и той же формы, что черновик."""
    if q1.dtype.kind != "i":
        raise SystemExit(f"метки типа {q1.dtype}: кэш обязан быть целым, "
                         f"иначе тренер получит числа с плавающей точкой и "
                         f"молча приведёт их сам")
    if q1.shape != q0hat.shape:
        raise SystemExit(f"метки формы {q1.shape}, черновик {q0hat.shape}")
    lo, hi = int(q1.min()), int(q1.max())
    if lo < 0 or hi >= vocab:
        raise SystemExit(f"метки в диапазоне [{lo}, {hi}] при словаре {vocab}")
    # ДИАГНОСТИКА, А НЕ ПРОВЕРКА: доля совпадений условной метки с истинной.
    # Совпадать они не обязаны — в том и смысл условной переразметки.
    return dict(vocab=int(vocab), lo=lo, hi=hi,
                frac_equal_static=float((q1 == ktrue).mean()))


def check_manifest(man, want):
    """Сверка манифеста кэша с ожидаемым составом. Отсутствие поля — отказ."""
    miss = [k for k in want if man.get(k) is None]
    if miss:
        raise SystemExit(f"в манифесте нет полей {miss}")
    bad = [(k, man[k], v) for k, v in want.items() if str(man[k]) != str(v)]
    if bad:
        raise SystemExit("манифест не совпал: "
                         + "; ".join(f"{k}: в кэше {a}, ожидалось {b}"
                                     for k, a, b in bad))
    return True


def check_outputs_absent(paths, overwrite):
    """Канонический кэш неизменяем. ПРОВЕРЯЕТСЯ ДО СЧЁТА, а не после.

    Проверка в конце стоила бы получаса работы, чтобы затем отказать: все
    135 593 строки были бы закодированы впустую.
    """
    ex = [p for p in paths if os.path.exists(p)]
    if ex and not overwrite:
        raise SystemExit(
            f"уже существует: {ex}. Канонический кэш не перезаписывается — "
            f"обученные головы ссылаются на его sha, и молчаливая перезапись "
            f"поменяла бы задачу под ними. Укажите --overwrite явно или "
            f"другое имя --out")
    return True


def check_oracle(orc, *, device, ckpt, cache, split_seed, sel_frac,
                 q0_prov=None):
    """Артефакт пройденного Gate 2: состав, режим и ТОТ ЖЕ черновик.

    RUN_ID ОБЯЗАТЕЛЕН. Артефакт без него снят версией до введения номера
    запуска, и связать его с конкретным прогоном нельзя — а именно на эту
    связь опирается вся привязка кэша.

    ЧЕРНОВИК ОБЯЗАН БЫТЬ ТОТ ЖЕ. Gate 2 измеряет, сколько восстанавливает
    оракул ОТ ПРЕДСКАЗАННОГО q0; цели кэша считаются как Q1(z_e - E0[q0]).
    Если это разные q0, порог Gate 4 и мишень обучения относятся к разным
    задачам, и расхождение выглядело бы как свойство модели.
    """
    bad = []
    if q0_prov is not None:
        if orc.get("q0_canonical") is not True:
            bad.append(f"гейт пройден на черновике {orc.get('q0_source')}, "
                       f"а кэш строится на каноническом")
        for k in ("q0_manifest_sha1", "q0_npz_sha1", "q0_run_id", "plan_sha1",
                  "plan_batch", "gate_r_sha1"):
            if orc.get(k) is None:
                bad.append(f"в артефакте гейта нет {k}")
            elif str(orc[k]) != str(q0_prov.get(k)):
                bad.append(f"{k}: гейт {orc[k]}, черновик кэша "
                           f"{q0_prov.get(k)}")
    if not orc.get("run_id"):
        bad.append("нет run_id: артефакт снят версией до его введения и с "
                   "конкретным прогоном не связан")
    for k in ("latent_capacity_ok", "action_oracle_ok",
              "dynamic_q1_relabeling_supported"):
        if not orc.get(k):
            bad.append(f"{k} = {orc.get(k)}")
    if str(orc.get("device")) != str(device):
        bad.append(f"гейт пройден на {orc.get('device')}, кэш строится на "
                   f"{device}")
    if float(orc.get("probe_code_disagree", 1.0)) != 0.0:
        bad.append(f"проба кодирования {orc.get('probe_code_disagree')}")
    if not orc.get("decoder_probe_matches_cache", False):
        bad.append("поведение декодера не совпадало с кэшем")
    for k, want in (("ckpt", ckpt), ("cache", cache),
                    ("split_seed", split_seed), ("sel_frac", sel_frac)):
        if str(orc.get(k)) != str(want):
            bad.append(f"{k}: {orc.get(k)} против {want}")
    if bad:
        raise SystemExit("артефакт Gate 2 непригоден: " + "; ".join(bad))
    return True


INT_DTYPES = ("int16", "int32", "int64")


def check_oracle_schema(orc, parts=("train", "val_sel", "val_confirm")):
    """Поля артефакта гейта, без которых сверка меток становится видимостью.

    ОТСУТСТВИЕ ПОЛЯ — ОТКАЗ. Условия вида «сверить, если поле есть» уже трижды
    за проект превращали проверку в согласие: в сводке K-13c, в keys_sha1 и в
    отпечатках кодека. Здесь то же самое: без rows_sha1 контрольные строки не
    сверяются, без n_used они не восстанавливаются, без dtype метки приводятся
    неизвестно к чему.
    """
    bad = []
    if orc.get("sample_seed") is None:
        bad.append("sample_seed")
    for nm in parts:
        smp = (orc.get("sampling") or {}).get(nm) or {}
        po = (orc.get("parts") or {}).get(nm) or {}
        for k in ("n_used", "rows_sha1"):
            if smp.get(k) in (None, ""):
                bad.append(f"sampling.{nm}.{k}")
        for k in ("q1_ze_sha1", "q1_ze_dtype"):
            if po.get(k) in (None, ""):
                bad.append(f"parts.{nm}.{k}")
        dt = po.get("q1_ze_dtype")
        if dt is not None and str(dt) not in INT_DTYPES:
            bad.append(f"parts.{nm}.q1_ze_dtype = {dt}: метки обязаны быть "
                       f"целыми, допустимы {INT_DTYPES}")
    if bad:
        raise SystemExit(
            "в артефакте гейта нет обязательных полей или они неверны: "
            + ", ".join(bad) + ". Без них сверка меток была бы видимостью")
    return True


def check_stamp_match(stamp, orc, got_w, probe_now):
    """Те ли САМЫЕ массивы и тот ли кодек, на которых пройден гейт.

    Совпадения с текущим заверением K-11b мало: заверение могло быть
    переснято на других массивах. Сверять надо с отпечатками, записанными В
    АРТЕФАКТЕ ГЕЙТА. Иначе q0hat, изменённый вне контрольных строк, прошёл бы
    и побитовую сверку меток, и заверение, а полный train получил бы другие
    цели.
    """
    go = (orc.get("stamp_k11b") or {}).get("arrays") or {}
    if not go:
        raise SystemExit("в артефакте гейта нет отпечатков массивов K-11b")
    bad = []
    for nm in ("q0hat", "ktrue", "split", "codebooks"):
        cur, gat = (stamp.get("arrays") or {}).get(nm), go.get(nm)
        if not gat:
            bad.append(f"{nm}: в артефакте гейта нет отпечатка")
        elif cur != gat:
            bad.append(f"{nm}: сейчас {cur}, на гейте {gat}")
    # ОТСУТСТВИЕ ПОЛЯ — ОТКАЗ, А НЕ СОГЛАСИЕ. Условие «сверить, если поле
    # есть» проходит на артефакте, который этих отпечатков не несёт, и тогда
    # кодек не сверяется вовсе. Это тот же fail-open, что уже ловился в
    # сводке K-13c и в проверке keys_sha1.
    for nm, cur in (("codebooks_sha1", got_w.get("codebooks_sha1")),
                    ("codec_state_sha1", got_w.get("codec_state_sha1")),
                    ("decoder_probe_now", probe_now)):
        gat = orc.get(nm)
        if gat is None:
            bad.append(f"{nm}: в артефакте гейта нет поля, сверить нечем")
        elif str(cur) != str(gat):
            bad.append(f"{nm}: сейчас {cur}, на гейте {gat}")
    if bad:
        raise SystemExit("данные или кодек не те, на которых пройден Gate 2: "
                         + "; ".join(bad))
    return True


def selftest():
    q0 = np.zeros((5, 16), np.int64)
    kt = np.zeros((5, 16), np.int64)
    q1 = np.arange(80, dtype=np.int64).reshape(5, 16) % 10
    st = check_codes(q1, 2048, q0, kt)
    assert st["lo"] == 0 and st["hi"] == 9
    assert abs(st["frac_equal_static"] - (q1 == kt).mean()) < 1e-12
    for bad, why in ((q1.astype(np.float32), "целым"),
                     (q1[:, :8], "формы"),
                     (q1 + 5000, "диапазоне")):
        try:
            check_codes(bad, 2048, q0, kt)
        except SystemExit as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"пропущено: {why}")

    man = dict(a="1", b="2")
    check_manifest(man, dict(a="1"))
    try:
        check_manifest(man, dict(a="9"))
    except SystemExit as e:
        assert "не совпал" in str(e), e
    else:
        raise AssertionError("расхождение манифеста пропущено")
    try:
        check_manifest(man, dict(c="1"))
    except SystemExit as e:
        assert "нет полей" in str(e), e
    else:
        raise AssertionError("отсутствующее поле пропущено")
    # --- НЕИЗМЕНЯЕМОСТЬ ВЫХОДА: отказ ДО счёта ---------------------------
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        p1 = os.path.join(_td, "q1_cache.npz")
        p2 = os.path.join(_td, "q1_cache.manifest.json")
        check_outputs_absent((p1, p2), False)
        open(p1, "wb").close()
        try:
            check_outputs_absent((p1, p2), False)
        except SystemExit as e:
            assert "не перезаписывается" in str(e), e
        else:
            raise AssertionError("существующий кэш не отвергнут")
        check_outputs_absent((p1, p2), True)
        open(p2, "w").close()
        try:
            check_outputs_absent((p1, p2), False)
        except SystemExit as e:
            assert "q1_cache.npz" in str(e) and "manifest" in str(e), e
        else:
            raise AssertionError("существующий манифест не отвергнут")

    # --- АРТЕФАКТ ГЕЙТА: состав и режим ----------------------------------
    good = dict(run_id="R1", latent_capacity_ok=True, action_oracle_ok=True,
                dynamic_q1_relabeling_supported=True, device="cuda:0",
                probe_code_disagree=0.0, decoder_probe_matches_cache=True,
                ckpt="CK", cache="data/c", split_seed=61, sel_frac=0.4)
    kw = dict(device="cuda:0", ckpt="CK", cache="data/c", split_seed=61,
              sel_frac=0.4)
    check_oracle(good, **kw)
    for patch, why in (({"run_id": None}, "нет run_id"),
                       ({"action_oracle_ok": False}, "action_oracle_ok"),
                       ({"device": "cpu"}, "гейт пройден на cpu"),
                       ({"probe_code_disagree": 0.01}, "проба кодирования"),
                       ({"decoder_probe_matches_cache": False},
                        "не совпадало"),
                       ({"cache": "data/other"}, "cache"),
                       ({"sel_frac": 0.5}, "sel_frac")):
        try:
            check_oracle(dict(good, **patch), **kw)
        except SystemExit as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"артефакт гейта принят при: {why}")

    # ТОТ ЖЕ ЧЕРНОВИК. Гейт и кэш обязаны стоять на одном q0, иначе порог
    # Gate 4 и мишень обучения относятся к разным задачам.
    q0p = dict(q0_canonical=True, q0_manifest_sha1="QM", q0_npz_sha1="QN",
               q0_run_id="QR", plan_sha1="PL", plan_batch=8, gate_r_sha1="GR")
    good_q0 = dict(good, **q0p)
    check_oracle(good_q0, **kw, q0_prov=q0p)
    for patch, why in (({"q0_canonical": False}, "на каноническом"),
                       ({"q0_manifest_sha1": "ДРУГОЕ"}, "q0_manifest_sha1"),
                       ({"q0_npz_sha1": "ДРУГОЕ"}, "q0_npz_sha1"),
                       ({"q0_run_id": "ДРУГОЕ"}, "q0_run_id"),
                       ({"plan_sha1": "ДРУГОЕ"}, "plan_sha1"),
                       ({"plan_batch": 16}, "plan_batch"),
                       ({"gate_r_sha1": "ДРУГОЕ"}, "gate_r_sha1")):
        try:
            check_oracle(dict(good_q0, **patch), **kw, q0_prov=q0p)
        except SystemExit as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"гейт с чужим черновиком принят: {why}")
    # артефакт гейта старого образца, без полей черновика вовсе
    for k_ in ("q0_manifest_sha1", "plan_sha1", "gate_r_sha1"):
        try:
            check_oracle({k: v for k, v in good_q0.items() if k != k_},
                         **kw, q0_prov=q0p)
        except SystemExit as e:
            assert f"нет {k_}" in str(e), (k_, str(e))
        else:
            raise AssertionError(f"гейт без {k_} принят")

    # --- ТЕ ЖЕ МАССИВЫ И ТОТ ЖЕ КОДЕК, ЧТО НА ГЕЙТЕ ----------------------
    arrs = {"q0hat": "A", "ktrue": "B", "split": "C", "codebooks": "D"}
    stamp_ok = dict(arrays=dict(arrs))
    orc_ok = dict(stamp_k11b=dict(arrays=dict(arrs)),
                  codebooks_sha1="CB", codec_state_sha1="CS",
                  decoder_probe_now="PR")
    gw = dict(codebooks_sha1="CB", codec_state_sha1="CS")
    check_stamp_match(stamp_ok, orc_ok, gw, "PR")
    # q0hat изменён ВНЕ контрольных строк: побитовая сверка меток его не
    # поймает, а эта проверка обязана.
    try:
        check_stamp_match(dict(arrays=dict(arrs, q0hat="X")), orc_ok, gw, "PR")
    except SystemExit as e:
        assert "q0hat" in str(e), e
    else:
        raise AssertionError("подменённый q0hat принят")
    for gw_, pr_, why in ((dict(gw, codebooks_sha1="Z"), "PR", "codebooks"),
                          (dict(gw, codec_state_sha1="Z"), "PR", "codec_state"),
                          (gw, "ZZ", "decoder_probe_now")):
        try:
            check_stamp_match(stamp_ok, orc_ok, gw_, pr_)
        except SystemExit as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"принято расхождение по {why}")
    try:
        check_stamp_match(stamp_ok, {}, gw, "PR")
    except SystemExit as e:
        assert "нет отпечатков" in str(e), e
    else:
        raise AssertionError("артефакт без отпечатков массивов принят")
    # ОТСУТСТВИЕ КАЖДОГО ИЗ ТРЁХ ОТПЕЧАТКОВ КОДЕКА — ОТКАЗ
    for nm_ in ("codebooks_sha1", "codec_state_sha1", "decoder_probe_now"):
        orc_miss = {k_: v_ for k_, v_ in orc_ok.items() if k_ != nm_}
        try:
            check_stamp_match(stamp_ok, orc_miss, gw, "PR")
        except SystemExit as e:
            assert f"{nm_}: в артефакте гейта нет поля" in str(e), (nm_, str(e))
        else:
            raise AssertionError(f"артефакт без {nm_} принят")

    # --- СХЕМА АРТЕФАКТА ГЕЙТА: ОТСУТСТВИЕ ПОЛЯ — ОТКАЗ ------------------
    def _orc(**patch):
        base = dict(sample_seed=0,
                    sampling={p: dict(n_used=10, rows_sha1="R" + p)
                              for p in ("train", "val_sel", "val_confirm")},
                    parts={p: dict(q1_ze_sha1="S" + p, q1_ze_dtype="int32")
                           for p in ("train", "val_sel", "val_confirm")})
        base.update(patch)
        return base

    check_oracle_schema(_orc())
    for patch, why in (
            ({"sample_seed": None}, "sample_seed"),
            ({"sampling": {}}, "sampling.train.n_used"),
            ({"parts": {}}, "parts.train.q1_ze_sha1")):
        try:
            check_oracle_schema(_orc(**patch))
        except SystemExit as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"схема принята без {why}")
    # по одному отсутствующему полю на часть
    for part_ in ("train", "val_sel", "val_confirm"):
        for grp, key in (("sampling", "n_used"), ("sampling", "rows_sha1"),
                         ("parts", "q1_ze_sha1"), ("parts", "q1_ze_dtype")):
            o = _orc()
            o[grp][part_] = {k: v for k, v in o[grp][part_].items()
                             if k != key}
            try:
                check_oracle_schema(o)
            except SystemExit as e:
                assert f"{grp}.{part_}.{key}" in str(e), (part_, key, str(e))
            else:
                raise AssertionError(f"принято без {grp}.{part_}.{key}")
    # нецелый тип меток
    o = _orc()
    o["parts"]["train"]["q1_ze_dtype"] = "float32"
    try:
        check_oracle_schema(o)
    except SystemExit as e:
        assert "обязаны быть целыми" in str(e), e
    else:
        raise AssertionError("нецелый тип меток принят")

    print("самопроверка k14b_build_q1_cache пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--ckpt",
                    default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--device", default="cuda:0",
                    help="КАНОНИЧЕСКИЙ режим разметки; пишется в манифест")
    ap.add_argument("--sel-frac", type=float, default=0.4)
    ap.add_argument("--split-seed", type=int, default=61)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--oracle", default="reports/k14a/oracle_cache_cuda0.json",
                    help="артефакт пройденного Gate 2: кэш меток обязан "
                         "строиться на тех же данных и в том же режиме")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="строить канонический кэш при незакоммиченных "
                         "изменениях (по умолчанию отказ)")
    ap.add_argument("--overwrite", action="store_true",
                    help="перезаписать существующий канонический кэш")
    ap.add_argument("--q0", default="",
                    help="канонический черновик K-14d (.npz); ТОЛЬКО ОН "
                         "определяет остаток, от которого считаются цели")
    ap.add_argument("--gate-r", default="reports/k14d/gate_r.json",
                    help="доказательство, что план батчей воспроизводим")
    ap.add_argument("--legacy-q0hat", action="store_true",
                    help="взять q0 из сентябрьского кэша K-11a. Он сегодня "
                         "побитово не воспроизводится, поэтому такой кэш "
                         "целей помечается как неканонический")
    ap.add_argument("--out", default="data/k14b/q1_cache")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    npz_p, man_p = a.out + ".npz", a.out + ".manifest.json"
    check_outputs_absent((npz_p, man_p), a.overwrite)

    # ЧИСТОЕ ДЕРЕВО ДЛЯ КАНОНИЧЕСКОГО АРТЕФАКТА. Кэш будет жить дольше этой
    # рабочей копии, и «построен таким-то коммитом» должно означать ровно то,
    # что написано. Состояние записывается в манифест в любом случае.
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(a.root)
    for p in (root, os.path.join(root, "src"), here,
              os.path.abspath("experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import k14_common as kc
    git_head, dirty_code, new_arte = kc.check_code_clean(a.allow_dirty)
    dirty = "\n".join(dirty_code)
    if new_arte:
        print(f"  незакоммиченных результатов рядом: {len(new_arte)} "
              f"(на код не влияют, в манифест записаны)")
    import torch
    import k11a_build_hicora_cache as k11a
    import k12b_protocol as kb
    import k13a_build_trajectory_basis as k13a
    import k14a_oracle_cache as k14a
    from k11c_train_d1 import split_episodes
    from depth_rvq_joint12 import code_contribution, nearest_code
    import actioncodec  # noqa: F401
    from utils import VisionLanguageActionProcessor

    dev = torch.device(a.device)

    # --- ПРОИСХОЖДЕНИЕ: ТОТ ЖЕ НАБОР ПРОВЕРОК, ЧТО В K-14a ------------------
    # Дублировать их нельзя, поэтому переиспользуются функции оттуда; кэш
    # меток обязан быть построен на тех же данных, на которых пройден гейт.
    meta = json.load(open(f"{a.cache}.meta.json"))
    if meta.get("ckpt") != a.ckpt:
        raise SystemExit(f"кэш собран чекпойнтом {meta.get('ckpt')}, а кодек "
                         f"берётся из {a.ckpt}")
    if meta.get("q0_source") != "joint12" or int(meta.get("depth", -1)) != 12:
        raise SystemExit(f"кэш собран источником {meta.get('q0_source')} на "
                         f"глубине {meta.get('depth')}")
    stamp_p = a.cache + ".artifacts.json"
    if not os.path.exists(stamp_p):
        raise SystemExit(f"нет {stamp_p}: кэш не заверен K-11b")
    stamp = json.load(open(stamp_p))
    if not stamp.get("identity_ok"):
        raise SystemExit("K-11b не подтвердила тождество для этого кэша")
    for nm in ("q0hat", "ktrue", "split", "codebooks"):
        got = k11a.file_sha1(f"{a.cache}.{nm}.npy")
        if got != (stamp.get("arrays") or {}).get(nm):
            raise SystemExit(f"{nm}.npy имеет sha {got}, K-11b заверила "
                             f"{(stamp.get('arrays') or {}).get(nm)}")


    q0hat_legacy = np.load(f"{a.cache}.q0hat.npy", mmap_mode="r")
    q0hat = q0hat_legacy
    ktrue = np.load(f"{a.cache}.ktrue.npy", mmap_mode="r")
    E = np.load(f"{a.cache}.codebooks.npy")
    N = int(meta["n_obs"])
    idx, _ = k13a.load_split(f"{a.cache}.split.npy", N)
    if tuple(q0hat.shape) != (N, 16) or tuple(ktrue.shape) != (N, 3, 16):
        raise SystemExit(f"формы q0hat {tuple(q0hat.shape)} и ktrue "
                         f"{tuple(ktrue.shape)} не те при n_obs {N}")

    src = meta.get("cache")
    if not src or not os.path.exists(src):
        raise SystemExit(f"исходный кэш {src} недоступен")
    src_npz = np.load(src, allow_pickle=True)
    keys_now = hashlib.sha1(np.ascontiguousarray(np.stack(
        [np.asarray(src_npz["episode"]),
         np.asarray(src_npz["step"])])).tobytes()).hexdigest()[:12]
    if not meta.get("keys_sha1") or keys_now != meta["keys_sha1"]:
        raise SystemExit(f"(episode, step) дают {keys_now}, в кэше "
                         f"{meta.get('keys_sha1')}")
    # --- ИСТОЧНИК ЧЕРНОВИКА -------------------------------------------------
    # Цели q1* определяются остатком z_e - E0[q0]. Значит, q0 — часть
    # определения целей, а не деталь реализации. Сентябрьский q0hat сегодня не
    # воспроизводится, поэтому канонический путь — артефакт K-14d с
    # доказательством Gate R.
    if a.q0 and a.legacy_q0hat:
        raise SystemExit("--q0 и --legacy-q0hat одновременно: источник "
                         "черновика должен быть один")
    q0_prov = dict(q0_source="k11a_legacy", q0_canonical=False)
    if a.q0:
        q0_arr, q0_defined, q0_man, q0_prov = kc.load_canonical_q0(
            a.q0, gate_r_path=a.gate_r, n_obs=N, keys_sha=keys_now,
            cache_meta_sha1=k11a.file_sha1(f"{a.cache}.meta.json"))
        q0_prov["q0_canonical"] = True
        q0hat = q0_arr
        # ТОЛЬКО ПО ПОСЧИТАННЫМ СТРОКАМ: строки вне плана помечены -1, и
        # считать их расхождением значит выдавать непокрытие за различие.
        d_msk = np.asarray(q0_defined)
        d_leg = int((np.asarray(q0hat_legacy)[d_msk].astype(np.int64)
                     != q0_arr[d_msk]).sum())
        q0_prov["diff_vs_k11a_positions"] = d_leg
        q0_prov["diff_vs_k11a_compared"] = int(d_msk.sum()) * 16
        q0_prov["rows_not_in_plan"] = int((~d_msk).sum())
        print(f"  черновик: канонический K-14d, план "
              f"{q0_prov['plan_sha1']}, Gate R {q0_prov['gate_r_sha1']}, "
              f"расхождение с кэшем K-11a {d_leg} из "
              f"{q0_prov['diff_vs_k11a_compared']} позиций; вне плана "
              f"{q0_prov['rows_not_in_plan']} строк")
    elif a.legacy_q0hat:
        q0_defined = np.ones(N, bool)
        print("  черновик: сентябрьский q0hat K-11a. Кэш целей будет помечен "
              "НЕКАНОНИЧЕСКИМ: этот массив сегодня побитово не повторяется")
    else:
        raise SystemExit(
            "не указан источник черновика. Цели считаются от остатка "
            "z_e - E0[q0], поэтому q0 входит в определение целей: укажите "
            "--q0 <артефакт K-14d> или явно --legacy-q0hat")

    # --- ПРИВЯЗКА К ПРОЙДЕННОМУ GATE 2 --------------------------------------
    # Без неё кэш меток «канонический» только на словах: изменённый массив
    # действий с прежними ключами (episode, step) прошёл бы все проверки выше
    # и дал бы ДРУГИЕ q1*. K-14a этот случай закрывает пробой кодирования и
    # сверкой K_true; здесь тот же разрыв закрывается ссылкой на его артефакт.
    if not os.path.exists(a.oracle):
        raise SystemExit(
            f"нет {a.oracle}: кэш меток обязан ссылаться на артефакт "
            f"пройденного Gate 2, иначе он ни к чему не привязан")
    orc = json.load(open(a.oracle))
    check_oracle(orc, device=dev, ckpt=a.ckpt, cache=a.cache,
                 split_seed=a.split_seed, sel_frac=a.sel_frac,
                 q0_prov=(q0_prov if q0_prov.get("q0_canonical") else None))
    check_oracle_schema(orc)
    print(f"  привязка к Gate 2: {a.oracle}, запуск {orc.get('run_id')}, "
          f"режим {orc.get('device')}")

    ACT = src_npz["action"]
    if ACT.shape[0] != N:
        raise SystemExit(f"в исходном кэше {ACT.shape[0]} действий при {N}")
    src_sha = sha12(src)
    if str(orc.get("source_cache_sha1")) != src_sha:
        raise SystemExit(
            f"исходный кэш K-9a имеет sha {src_sha}, а Gate 2 пройден на "
            f"{orc.get('source_cache_sha1')}: массив действий определяет z_e "
            f"и, значит, сами метки")
    kt_src = np.asarray(src_npz["K_true"])[:N].astype(np.int64)
    if not np.array_equal(kt_src, np.asarray(ktrue).astype(np.int64)):
        raise SystemExit("K_true исходного кэша расходится с заверенным")
    epi = np.asarray(src_npz["episode"]).astype(np.int64)[:N]

    # --- кодек --------------------------------------------------------------
    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None))
    codec = codec.to(dev).eval()
    qs = list(codec.vq.quantizers)
    if len(qs) != 3:
        raise SystemExit(f"уровней RVQ {len(qs)}, ожидалось 3")
    with torch.no_grad():
        ii = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        Ecur = torch.stack([q.out_project(q.decode_code(ii))[0]
                            for q in qs]).float()
    if float((Ecur.cpu() - torch.from_numpy(E)).abs().max()) > 1e-5:
        raise SystemExit("книги разошлись с кэшем")
    got_w = dict(codebooks_sha1=arr_sha(E.astype(np.float32)),
                 codec_state_sha1=k11a.state_sha1(codec))
    bad_w = [k for k, v in got_w.items() if meta.get(k) != v]
    if bad_w:
        raise SystemExit(f"веса кодека не те, которыми собран кэш: {bad_w}")
    probe_now = k11a.decoder_probe(codec, Ecur.to(dev), dev)
    if meta.get("decoder_probe") != probe_now:
        raise SystemExit(
            f"поведение декодера отличается от записанного при сборке кэша "
            f"(в кэше {meta.get('decoder_probe')}, сейчас {probe_now}). Кэш "
            f"меток обязан строиться в ТОМ ЖЕ режиме, в каком пройден гейт "
            f"K-14a: устройство {dev}")
    check_stamp_match(stamp, orc, got_w, probe_now)
    print(f"  происхождение сверено с артефактом гейта: массивы K-11b, книги, "
          f"веса кодека и проба декодера совпали; режим {dev}")

    # --- части: те же, что в K-14a -----------------------------------------
    parts, sample_meta = k14a.build_parts(
        idx, epi, a.sel_frac, a.split_seed, 0, np.random.default_rng(0),
        split_episodes)
    print("  части: " + ", ".join(f"{k} {v['n_used']}"
                                  for k, v in sample_meta.items())
          + ". Подвыборки нет; финальная выборка не читается")

    # --- метки --------------------------------------------------------------
    out_rows, out_q1, out_part = [], [], []
    stats = {}
    for name in ("train", "val_sel", "val_confirm"):
        rows = parts[name]
        if not q0_defined[np.asarray(rows, np.int64)].all():
            raise SystemExit(
                f"часть {name}: план K-14d не покрывает все её строки, "
                f"черновик там не посчитан. Цели от непосчитанного q0 были бы "
                f"целями от кода -1")
        acc = []
        for i in range(0, len(rows), a.batch):
            r = rows[i:i + a.batch]
            with torch.no_grad():
                act = torch.from_numpy(
                    np.asarray(ACT[r], np.float32)).to(dev)
                z_e = codec._encode(act, embodiment_ids=0).float()
                e0 = code_contribution(
                    qs[0], torch.from_numpy(
                        np.asarray(q0hat[r]).astype(np.int64)).to(dev))
                acc.append(nearest_code(z_e - e0, qs[1]).cpu().numpy())
        q1 = np.concatenate(acc).astype(np.int32)
        stats[name] = check_codes(q1, int(codec.vocab_size),
                                  np.asarray(q0hat[rows]),
                                  np.asarray(ktrue[rows])[:, 1, :])
        stats[name].update(n_rows=int(len(rows)),
                           q1_sha1=arr_sha(q1),
                           rows_sha1=arr_sha(np.asarray(rows, np.int64)))
        out_rows.append(np.asarray(rows, np.int64))
        out_q1.append(q1)
        out_part.append(np.full(len(rows), name, dtype=object))
        print(f"    {name}: {len(rows)} строк, sha меток "
              f"{stats[name]['q1_sha1']}, совпадает с истинной q1 у "
              f"{100 * stats[name]['frac_equal_static']:.1f}% позиций")

    # --- ПОБИТОВАЯ СВЕРКА С МЕТКАМИ ОРАКУЛА ---------------------------------
    # Один и тот же вычислительный путь обязан дать те же метки. Это
    # одновременно проверяет данные, устройство, разбиение на партии и
    # реализацию поиска ближайшего кода. Для train у оракула была
    # КОНТРОЛЬНАЯ подвыборка, поэтому её строки восстанавливаются тем же
    # генератором, и сверка идёт на них.
    ctl_parts, ctl_meta = k14a.build_parts(
        idx, epi, a.sel_frac, a.split_seed,
        int(orc["sampling"]["train"]["n_used"]),
        np.random.default_rng(int(orc["sample_seed"])),
        split_episodes)
    q1_by_part = {nm: arr for nm, arr in zip(
        ("train", "val_sel", "val_confirm"), out_q1)}
    rows_by_part = {nm: np.asarray(r, np.int64) for nm, r in zip(
        ("train", "val_sel", "val_confirm"), out_rows)}
    checked = {}
    for name in ("train", "val_sel", "val_confirm"):
        po = (orc.get("parts") or {}).get(name) or {}
        want_sha, want_dt = po["q1_ze_sha1"], po["q1_ze_dtype"]
        ctl = np.asarray(ctl_parts[name], np.int64)
        smp = orc["sampling"][name]
        if len(ctl) != int(smp["n_used"]):
            raise SystemExit(
                f"{name}: восстановлено {len(ctl)} контрольных строк, в "
                f"артефакте заявлено {smp['n_used']}")
        got_rows_sha = arr_sha(ctl)
        if got_rows_sha != smp["rows_sha1"]:
            raise SystemExit(
                f"{name}: контрольные строки дают sha {got_rows_sha}, у "
                f"оракула {smp['rows_sha1']} — восстановлен другой набор")
        pos = np.searchsorted(rows_by_part[name], ctl)
        if pos.max() >= len(rows_by_part[name]) or \
                not np.array_equal(rows_by_part[name][pos], ctl):
            raise SystemExit(f"{name}: контрольные строки не лежат в кэше")
        sub = q1_by_part[name][pos].astype(np.dtype(want_dt))
        got = arr_sha(sub)
        if got != want_sha:
            n_d = "неизвестно"
            raise SystemExit(
                f"{name}: метки q1 не совпали с оракулом — sha {got} против "
                f"{want_sha} ({n_d} расхождений). Кэш строится не тем путём, "
                f"которым пройден Gate 2")
        checked[name] = dict(rows_sha1=got_rows_sha, q1_sha1=got,
                             dtype=str(want_dt), n=int(len(ctl)))
        print(f"    {name}: метки совпали с оракулом побитово на "
              f"{len(ctl)} строках (sha {got}, {want_dt})")

    rows_all = np.concatenate(out_rows)
    q1_all = np.concatenate(out_q1)
    part_all = np.concatenate(out_part).astype(str)
    if len(np.unique(rows_all)) != len(rows_all):
        raise SystemExit("строки повторяются между частями")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    npz = npz_p
    tmp = npz + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, rows=rows_all, q1=q1_all, part=part_all)
    os.replace(tmp, npz)
    with np.load(npz, allow_pickle=True) as z:
        if sorted(z.files) != ["part", "q1", "rows"]:
            raise SystemExit(f"{npz}: набор массивов {sorted(z.files)}")
        for nm_, ref_ in (("rows", rows_all), ("q1", q1_all),
                          ("part", part_all)):
            got_ = z[nm_]
            if got_.shape != ref_.shape or str(got_.dtype) != str(ref_.dtype):
                raise SystemExit(f"{npz}: {nm_} формы {got_.shape} "
                                 f"{got_.dtype}, записывалось {ref_.shape} "
                                 f"{ref_.dtype}")
            if not np.array_equal(got_, ref_):
                raise SystemExit(f"{npz}: {nm_} прочитался иначе, чем записан")

    man = dict(
        kind="canonical_q1_targets", target_latent="z_e = codec._encode(action)",
        device=str(dev), dtype="int32", n_rows=int(len(rows_all)),
        parts={k: dict(v) for k, v in stats.items()},
        sampling=sample_meta, split_seed=int(a.split_seed),
        sel_frac=float(a.sel_frac), cache=a.cache,
        cache_meta_sha1=k11a.file_sha1(f"{a.cache}.meta.json"),
        source_cache=src, source_cache_sha1=sha12(src), keys_sha1=keys_now,
        q0hat_sha1=k11a.file_sha1(f"{a.cache}.q0hat.npy"),
        **q0_prov,
        ktrue_sha1=k11a.file_sha1(f"{a.cache}.ktrue.npy"),
        split_sha1=k11a.file_sha1(f"{a.cache}.split.npy"),
        ckpt=a.ckpt, vocab=int(codec.vocab_size),
        codebooks_sha1=got_w["codebooks_sha1"],
        codec_state_sha1=got_w["codec_state_sha1"], decoder_probe=probe_now,
        labels_npz=os.path.basename(npz), labels_sha1=sha12(npz),
        rows_sha1=arr_sha(rows_all), q1_sha1=arr_sha(q1_all),
        note="только q1. Цели q2 строятся позже и отдельно для каждой "
             "обученной головы, от её ФАКТИЧЕСКОГО q1, а не от оракульного",
        oracle_artifact=a.oracle, oracle_sha1=sha12(a.oracle),
        oracle_run_id=orc.get("run_id"),
        checked_against_oracle=checked,
        torch_version=str(torch.__version__),
        cuda_version=str(getattr(torch.version, "cuda", None)),
        gpu=(torch.cuda.get_device_name(dev)
             if dev.type == "cuda" else None),
        tf32_matmul=bool(getattr(torch.backends.cuda, "matmul", None)
                         and torch.backends.cuda.matmul.allow_tf32),
        tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
        code_version=kb.code_version([
            os.path.abspath(__file__),
            os.path.join(here, "depth_rvq_joint12.py"),
            os.path.join(here, "k14a_oracle_cache.py"),
            os.path.join(here, "k11a_build_hicora_cache.py"),
            os.path.join(here, "k11c_train_d1.py"),
            os.path.join(here, "k12b_protocol.py"),
            os.path.join(here, "k13a_build_trajectory_basis.py")]),
        actioncodec_sha1=sha12(os.path.join(
            root, "actioncodec", "rvq.py")),
        git_head=git_head or None,
        git_dirty=bool(dirty),
        git_dirty_files=len(dirty.splitlines()) if dirty else 0,
        script_sha1=sha12(os.path.abspath(__file__)))
    mp = man_p
    tmpm = mp + f".tmp.{os.getpid()}"
    json.dump(man, open(tmpm, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmpm, mp)
    print(f"\n  кэш: {npz} (sha {man['labels_sha1']}), манифест: {mp}")
    print(f"  всего {len(rows_all)} строк, метки int32, канонический режим "
          f"{dev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
