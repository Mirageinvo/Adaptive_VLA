#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14c: обучение головы q1 на КАНОНИЧЕСКОМ кэше целей. Stage 4 плана K-14.

ЧТО ОБУЧАЕТСЯ. Ровно три группы весов: `depth_rvq_norms.0`,
`depth_rvq_heads.0` и — в вариантах с самообусловливанием — `feedback.0`.
Backbone, Joint12, q0, кодек и уровень q2 заморожены. Состав задаётся
`configure_joint_depth_rvq(stage="q1", variant=...)` и не имеет режима по
умолчанию: молча выбранный набор обучаемых весов — это молча другой
эксперимент.

ЦЕЛИ НЕ ПЕРЕСЧИТЫВАЮТСЯ. Метка q1* — ближайший код к остатку
z_e - E0[q0hat], и у части позиций два кода почти равноудалены (§42, разрыв
5e-08 при величинах 1e-02). Пересчёт в тренере дал бы двум прогонам РАЗНЫЕ
задачи, и разность между ними нельзя было бы приписать порядку данных. Здесь
читается готовый целочисленный кэш K-14b с проверкой отпечатков.

ВАРИАНТЫ (`--variant`):
    main         условные цели из кэша, feedback включён
    static       ИСТИННЫЕ коды q1 как цели, feedback включён — контроль на
                 расхождение меток K-8
    no_feedback  условные цели, feedback ОБХОДИТСЯ в forward, а не просто
                 заморожен

ДВА ПРОГОНА — ЭТО TRAINING/DATA-ORDER SEEDS, А НЕ ДВЕ ИНИЦИАЛИЗАЦИИ. Поздние
головы инициализируются детерминированно: копия `action_lm_head`, нулевая
проекция feedback, копия исходной финальной нормы. Различается только порядок
данных, и называть это независимыми инициализациями неверно.

ОТБОР ЭПОХИ ТОЛЬКО ПО val_sel. Подтверждающая половина открывается ОДИН РАЗ,
после выбора. Gate 4 считается на ней по заранее записанной формуле:

    C = (E_A0 - E_model) / (E_A0 - E_oracle) >= 0.20

где E_A0 и E_oracle берутся ИЗ АРТЕФАКТА ОРАКУЛА, а не из этого прогона.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np

GATE4_MIN_CAPTURE = 0.20


def sha12(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:12]


def arr_sha(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def capture_fraction(e_a0, e_model, e_oracle):
    """Какую долю оракульного выигрыша забрала обученная голова.

    ЗНАМЕНАТЕЛЬ ПРОВЕРЯЕТСЯ ЯВНО. При e_oracle >= e_a0 выигрыша нет вовсе, и
    отношение было бы делением на ноль или отрицательным числом с видом доли.
    Отрицательный результат при исправном знаменателе законен и означает, что
    модель хуже опоры — это надо видеть, а не прятать.
    """
    for nm, v in (("e_a0", e_a0), ("e_model", e_model),
                  ("e_oracle", e_oracle)):
        if v is None or not np.isfinite(float(v)):
            raise SystemExit(f"{nm} = {v}: не число")
    gap = float(e_a0) - float(e_oracle)
    if gap <= 1e-12:
        raise SystemExit(
            f"оракул не лучше опоры: E_A0 {e_a0}, E_oracle {e_oracle}. "
            f"Доля захвата не определена")
    return (float(e_a0) - float(e_model)) / gap


def check_gate4(e_a0, e_model, e_oracle, threshold=GATE4_MIN_CAPTURE):
    """Gate 4 плана K-14. Порог записан ДО обучения и здесь не подбирается."""
    c = capture_fraction(e_a0, e_model, e_oracle)
    limit = float(e_a0) - threshold * (float(e_a0) - float(e_oracle))
    ok = bool(c >= threshold)
    return dict(capture=c, threshold=float(threshold), passed=ok,
                rms_limit=limit, e_a0=float(e_a0), e_model=float(e_model),
                e_oracle=float(e_oracle))


def select_epoch(history):
    """Эпоха выбирается ТОЛЬКО по val_sel. Эпоха 0 — полноценный кандидат.

    Если начальное состояние головы окажется лучше всего обученного, это
    результат, а не повод исключить его из рассмотрения.
    """
    if not history:
        raise SystemExit("пустая история обучения")
    for h in history:
        if h.get("val_sel") is None or not np.isfinite(float(h["val_sel"])):
            raise SystemExit(f"эпоха {h.get('epoch')}: val_sel = "
                             f"{h.get('val_sel')}")
        if "val_confirm" in h:
            raise SystemExit(
                f"эпоха {h.get('epoch')}: в истории есть val_confirm. "
                f"Подтверждающая половина открывается один раз ПОСЛЕ выбора, "
                f"иначе отбор идёт по ней")
    best = min(history, key=lambda h: (float(h["val_sel"]), int(h["epoch"])))
    return int(best["epoch"]), float(best["val_sel"])


def check_resume(prev, *, stage, variant, q1_cache_sha1, trainable_names,
                 feedback_mask, joint_sha1):
    """Продолжение обязано совпасть по всему, что определяет задачу."""
    bad = []
    if prev.get("stage") != stage:
        bad.append(f"этап {prev.get('stage')} против {stage}")
    if prev.get("variant") != variant:
        bad.append(f"вариант {prev.get('variant')} против {variant}")
    if prev.get("q1_cache_sha1") != q1_cache_sha1:
        bad.append(f"кэш целей {prev.get('q1_cache_sha1')} против "
                   f"{q1_cache_sha1}")
    if prev.get("joint_sha1") != joint_sha1:
        bad.append(f"Joint12 {prev.get('joint_sha1')} против {joint_sha1}")
    if list(prev.get("feedback_mask") or []) != list(feedback_mask):
        bad.append(f"маска feedback {prev.get('feedback_mask')} против "
                   f"{list(feedback_mask)}")
    pn = sorted(prev.get("trainable_names") or [])
    if pn != sorted(trainable_names):
        bad.append(f"набор обучаемых весов: нет {sorted(set(trainable_names) - set(pn))[:3]}, "
                   f"лишние {sorted(set(pn) - set(trainable_names))[:3]}")
    if bad:
        raise SystemExit("продолжение не сходится: " + "; ".join(bad))
    return True


def check_cache_manifest(man, *, oracle_sha1, cache, ckpt, expect_sha1):
    """Кэш целей обязан быть тем, что построен на данных пройденного гейта."""
    need = ("kind", "labels_sha1", "q1_sha1", "rows_sha1", "device",
            "oracle_sha1", "cache", "ckpt", "n_rows", "parts")
    miss = [k for k in need if man.get(k) is None]
    if miss:
        raise SystemExit(f"в манифесте кэша нет полей {miss}")
    if man["kind"] != "canonical_q1_targets":
        raise SystemExit(f"манифест описывает {man['kind']}, а нужен "
                         f"canonical_q1_targets")
    bad = []
    if man["labels_sha1"] != expect_sha1:
        bad.append(f"labels_sha1 {man['labels_sha1']} против фактического "
                   f"{expect_sha1}")
    if man["oracle_sha1"] != oracle_sha1:
        bad.append(f"кэш построен по артефакту гейта {man['oracle_sha1']}, "
                   f"а подан {oracle_sha1}")
    for k, v in (("cache", cache), ("ckpt", ckpt)):
        if str(man[k]) != str(v):
            bad.append(f"{k}: {man[k]} против {v}")
    if bad:
        raise SystemExit("манифест кэша целей не сходится: " + "; ".join(bad))
    return True


def selftest():
    # --- ДОЛЯ ЗАХВАТА И ПОРОГ --------------------------------------------
    assert abs(capture_fraction(1.0, 0.8, 0.0) - 0.2) < 1e-12
    assert abs(capture_fraction(1.0, 1.2, 0.0) + 0.2) < 1e-12   # хуже опоры
    for a_, m_, o_, why in ((1.0, 0.5, 1.0, "не лучше опоры"),
                            (1.0, 0.5, 2.0, "не лучше опоры"),
                            (float("nan"), 0.5, 0.0, "e_a0"),
                            (1.0, float("inf"), 0.0, "e_model")):
        try:
            capture_fraction(a_, m_, o_)
        except SystemExit as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"пропущено: {why}")

    # Значения округлённые, поэтому и предел здесь округлённый. Настоящий
    # порог берётся из артефакта оракула с полной точностью: на его числах
    # (0.1443760…, 0.0826230…) он равен 0.1320250185.
    g = check_gate4(0.14438, 0.13, 0.08262)
    assert g["passed"] and abs(g["rms_limit"] - 0.132028) < 1e-6, g
    assert not check_gate4(0.14438, 0.14, 0.08262)["passed"]
    # ровно на пороге — проходит
    lim = check_gate4(0.14438, 0.0, 0.08262)["rms_limit"]
    assert check_gate4(0.14438, lim, 0.08262)["passed"]

    # --- ОТБОР ЭПОХИ ------------------------------------------------------
    hist = [dict(epoch=0, val_sel=0.20), dict(epoch=1, val_sel=0.15),
            dict(epoch=2, val_sel=0.17)]
    assert select_epoch(hist) == (1, 0.15)
    # эпоха 0 — полноценный кандидат
    assert select_epoch([dict(epoch=0, val_sel=0.1),
                         dict(epoch=1, val_sel=0.2)])[0] == 0
    # при равенстве берётся РАННЯЯ эпоха, а не случайная
    assert select_epoch([dict(epoch=0, val_sel=0.1),
                         dict(epoch=1, val_sel=0.1)])[0] == 0
    for h_, why in (([], "пустая"),
                    ([dict(epoch=0, val_sel=None)], "val_sel"),
                    ([dict(epoch=0, val_sel=0.1, val_confirm=0.2)],
                     "val_confirm")):
        try:
            select_epoch(h_)
        except SystemExit as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"отбор принял: {why}")

    # --- ПРОДОЛЖЕНИЕ ------------------------------------------------------
    base = dict(stage="q1", variant="main", q1_cache_sha1="C",
                joint_sha1="J", feedback_mask=[True, True],
                trainable_names=["a", "b"])
    kw = dict(stage="q1", variant="main", q1_cache_sha1="C", joint_sha1="J",
              feedback_mask=(True, True), trainable_names=["b", "a"])
    check_resume(dict(base), **kw)
    for patch, why in ((dict(variant="no_feedback"), "вариант"),
                       (dict(q1_cache_sha1="X"), "кэш целей"),
                       (dict(joint_sha1="X"), "Joint12"),
                       (dict(feedback_mask=[False, True]), "маска feedback"),
                       (dict(trainable_names=["a"]), "набор обучаемых")):
        try:
            check_resume(dict(base, **patch), **kw)
        except SystemExit as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"продолжение принято при: {why}")

    # --- МАНИФЕСТ КЭША ----------------------------------------------------
    man = dict(kind="canonical_q1_targets", labels_sha1="L", q1_sha1="Q",
               rows_sha1="R", device="cuda:0", oracle_sha1="O",
               cache="data/c", ckpt="CK", n_rows=10, parts={})
    mk = dict(oracle_sha1="O", cache="data/c", ckpt="CK", expect_sha1="L")
    check_cache_manifest(man, **mk)
    for patch, why in ((dict(kind="other"), "canonical_q1_targets"),
                       (dict(labels_sha1="Z"), "labels_sha1"),
                       (dict(oracle_sha1="Z"), "артефакту гейта"),
                       (dict(cache="d"), "cache")):
        try:
            check_cache_manifest(dict(man, **patch), **mk)
        except SystemExit as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"манифест принят при: {why}")
    try:
        check_cache_manifest({k: v for k, v in man.items()
                              if k != "labels_sha1"}, **mk)
    except SystemExit as e:
        assert "нет полей" in str(e), e
    else:
        raise AssertionError("манифест без labels_sha1 принят")
    print("самопроверка k14c_train_q1 пройдена")


def group_by_offset(rows, offs, batch):
    """Батчи из строк с ОДИНАКОВЫМ смещением позиций.

    `build_inputs` принимает одно `position_offset` на батч, поэтому смешивать
    строки разных задач в одном батче нельзя: часть получила бы чужое
    смещение и читала бы вход не с той позиции.
    """
    out = []
    for po in sorted(set(int(x) for x in offs[rows])):
        sel = rows[offs[rows] == po]
        for i in range(0, len(sel), batch):
            out.append((po, sel[i:i + batch]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--q1-cache", default="data/k14b/q1_cache")
    ap.add_argument("--oracle", default="reports/k14a/oracle_cache_cuda0.json")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--joint-ckpt", default="data/k9d_ep3.pt")
    ap.add_argument("--ckpt",
                    default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    # НЕ required: иначе `--selftest` требовал бы вариант и сид, которых у
    # проверки чистых функций нет. Обязательность проверяется ниже, после
    # ветки самопроверки.
    ap.add_argument("--variant", choices=("main", "static", "no_feedback"))
    ap.add_argument("--seed", type=int, default=None,
                    help="training/data-order seed: поздние головы "
                         "инициализируются детерминированно, различается "
                         "только порядок данных")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lambda-action", type=float, default=1.0)
    ap.add_argument("--lambda-fb", type=float, default=1e-4)
    ap.add_argument("--grip-weight", type=float, default=1.0,
                    help="вес канала схвата в потере действия; фиксируется "
                         "до первого запуска")
    ap.add_argument("--limit", type=int, default=0,
                    help="ограничить train для проверки связности")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    miss = [n for n, v in (("--variant", a.variant), ("--seed", a.seed))
            if v is None]
    if miss:
        raise SystemExit(f"нужны {miss}: вариант и training-сид задаются "
                         f"явно, умолчаний у них нет")
    out_p = a.out or f"data/k14c/q1_{a.variant}_s{a.seed}.pt"
    if os.path.exists(out_p) and not a.resume:
        raise SystemExit(f"{out_p} уже существует: обученная голова не "
                         f"перезаписывается молча")

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(a.root)
    for p in (root, os.path.join(root, "src"), here,
              os.path.abspath("experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import torch.nn.functional as F
    import k11a_build_hicora_cache as k11a
    import k12b_protocol as kb
    from depth_rvq_joint12 import (make_joint_depth_rvq_class,
                                   code_contribution)
    from depth_rvq_vla import straight_through
    from joint12_vla import make_joint12_class
    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       VisionLanguageActionProcessor, dict_apply, get_cfg,
                       prompt_template)

    dev = torch.device(a.device)
    dt = getattr(torch, a.dtype)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    max_act_q = np.maximum(np.abs(ACTION_Q01), np.abs(ACTION_Q99))
    H_EXEC = 8

    # --- кэш целей и его привязка -------------------------------------------
    npz_p = a.q1_cache + ".npz"
    man_p = a.q1_cache + ".manifest.json"
    man = json.load(open(man_p))
    check_cache_manifest(man, oracle_sha1=sha12(a.oracle), cache=a.cache,
                         ckpt=a.ckpt, expect_sha1=sha12(npz_p))
    with np.load(npz_p, allow_pickle=True) as z:
        rows_all = np.asarray(z["rows"], np.int64)
        q1_all = np.asarray(z["q1"], np.int64)
        part_all = np.asarray(z["part"]).astype(str)
    if arr_sha(q1_all.astype(np.int32)) != man["q1_sha1"]:
        raise SystemExit("массив целей не совпал с отпечатком манифеста")
    orc = json.load(open(a.oracle))
    print(f"  цели: {npz_p}, sha {man['labels_sha1']}, построены в режиме "
          f"{man['device']} по гейту {man['oracle_sha1']}")

    # --- исходные данные ----------------------------------------------------
    meta = json.load(open(f"{a.cache}.meta.json"))
    src = meta["cache"]
    d = np.load(src, allow_pickle=True)
    cmeta = json.loads(str(d["meta"]))
    N = int(meta["n_obs"])
    epi, stp = np.asarray(d["episode"])[:N], np.asarray(d["step"])[:N]
    keys_sha = hashlib.sha1(np.ascontiguousarray(
        np.stack([epi, stp])).tobytes()).hexdigest()[:12]
    if keys_sha != meta.get("keys_sha1"):
        raise SystemExit(f"ключи наблюдений {keys_sha} против "
                         f"{meta.get('keys_sha1')}")
    ACT = np.asarray(d["action"])[:N]
    offs = np.asarray(d["pos_offset"])[:N].astype(np.int64)
    tsk = np.asarray(d["task"])[:N]
    ktrue = np.load(f"{a.cache}.ktrue.npy", mmap_mode="r")
    q0hat_c = np.load(f"{a.cache}.q0hat.npy", mmap_mode="r")
    E = np.load(f"{a.cache}.codebooks.npy")

    img_p = os.path.join(os.path.dirname(src), cmeta["images_file"])
    IMG = np.load(img_p, mmap_mode="r")
    if IMG.shape[0] < N or IMG.dtype != np.uint8:
        raise SystemExit(f"кадры {IMG.shape} {IMG.dtype}: не те")
    st_p = a.cache + ".state.npy"
    stm_p = a.cache + ".state.json"
    if not (os.path.exists(st_p) and os.path.exists(stm_p)):
        raise SystemExit(f"нет {st_p}: состояния собираются K-11a, "
                         f"пересобирать их здесь нельзя — получился бы другой "
                         f"вход")
    sm = json.load(open(stm_p))
    if sm.get("keys_sha1") != keys_sha:
        raise SystemExit(f"состояния собраны для ключей {sm.get('keys_sha1')}")
    st_n = ((np.load(st_p)[:N] - STATE_Q01) / (STATE_Q99 - STATE_Q01)
            * 2.0 - 1.0)
    print(f"  данные: {N} наблюдений, кадры {IMG.shape[1:]}, состояния "
          f"{st_n.shape[1]}-мерные")

    # --- модель -------------------------------------------------------------
    cfg = get_cfg(os.path.join(root, a.cfg_path))
    Base = make_joint12_class(SmolVLABlockwiseAR)
    model = Base.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    model.init_joint_fast(depth=int(meta["depth"]), head_dtype=torch.float32)
    # ИСХОДНАЯ ФИНАЛЬНАЯ НОРМА СНИМАЕТСЯ ДО ЗАГРУЗКИ Joint12: он перезаписывает
    # action_expert.norm, обученную читать h12, а поздние выходы должны
    # стартовать с нормы, калиброванной на полной глубине.
    refine_norm = copy.deepcopy(model.action_expert.norm)
    j_sha = k11a.file_sha1(a.joint_ckpt)
    if j_sha != (meta.get("source") or {}).get("weights_sha1"):
        raise SystemExit(f"Joint12 {j_sha}, кэш собран на "
                         f"{(meta.get('source') or {}).get('weights_sha1')}")
    obj = torch.load(a.joint_ckpt, map_location="cpu", weights_only=False)
    state = obj["state"]
    own = dict(model.named_parameters())
    stray = [k for k in state if not any(k.startswith(p) or k == p.rstrip(".")
                                         for p in model.trainable_prefixes)]
    missing = [k for k in own if own[k].requires_grad and k not in state]
    if stray or missing:
        raise SystemExit(f"Joint12 загружается не строго: лишние {stray[:3]}, "
                         f"нет {missing[:3]}")
    with torch.no_grad():
        for k, v in state.items():
            own[k].data = v.to(dev, torch.float32)
    print(f"  Joint12 загружен строго: {len(state)} тензоров, sha {j_sha}")

    model.__class__ = make_joint_depth_rvq_class(type(model))
    model.init_joint_depth_rvq(refine_norm=refine_norm,
                               books=torch.from_numpy(E),
                               feedback=(a.variant != "no_feedback"),
                               verbose_init=False)
    info = model.configure_joint_depth_rvq(stage="q1", variant=a.variant)
    codec = proc.action_processor
    codec = (codec if hasattr(codec, "vq") else codec.codec).to(dev).eval()
    for p_ in codec.parameters():
        p_.requires_grad_(False)

    if a.resume:
        prev = torch.load(a.resume, map_location="cpu", weights_only=False)
        check_resume(prev, stage="q1", variant=a.variant,
                     q1_cache_sha1=man["labels_sha1"],
                     trainable_names=info["names"],
                     feedback_mask=info["feedback_mask"], joint_sha1=j_sha)
        with torch.no_grad():
            for k, v in prev["state"].items():
                dict(model.named_parameters())[k].data = v.to(dev,
                                                              torch.float32)
        print(f"  продолжение от {a.resume}, эпох пройдено "
              f"{prev.get('epochs_run')}")

    # --- батчи --------------------------------------------------------------
    sets = {nm: rows_all[part_all == nm] for nm in ("train", "val_sel",
                                                    "val_confirm")}
    # ЦЕЛИ РАСКЛАДЫВАЮТСЯ ПО НОМЕРАМ СТРОК КЭША K-11a: батчи формируются по
    # смещению позиций, а не по порядку в кэше целей, и брать цель по позиции
    # в массиве было бы сопоставлением не тех строк.
    q1_of_pos = np.full((N, q1_all.shape[1]), -1, np.int64)
    q1_of_pos[rows_all] = q1_all
    if a.limit:
        sets["train"] = sets["train"][:a.limit]
    print("  части: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))

    def build(po, sel):
        image = torch.from_numpy(np.asarray(IMG[sel]))
        msgs = []
        for gi in sel:
            m = prompt_template(
                st_n[gi], None, str(tsk[gi]),
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        b = proc(text=texts, images=[[image[k].numpy()]
                                     for k in range(len(sel))],
                 return_tensors="pt", padding=True, padding_side="left",
                 action_processor_kwargs={"embodiment_ids": 0})
        return dict_apply(lambda x: x.to(dev, dt), b)

    books = model.depth_rvq_books
    ac16 = torch.autocast(device_type=dev.type, dtype=dt)

    def targets(sel):
        if a.variant == "static":
            # КОНТРОЛЬ K-8: истинные коды q1 без учёта ошибки префикса.
            return torch.from_numpy(
                np.asarray(ktrue[sel])[:, 1, :].astype(np.int64)).to(dev)
        t = q1_of_pos[sel]
        if (t < 0).any():
            raise SystemExit("строка вне канонического кэша целей")
        return torch.from_numpy(t).to(dev)

    def decode_actions(z):
        x, _ = codec._decode(z.float(), embodiment_ids=0)
        return x[..., :7].float()

    def run_batch(po, sel, train):
        b = build(po, sel)
        with ac16:
            v_, p_ = model.build_inputs(position_offset=po, **b)
            out = model.forward_joint_depth_rvq(
                vlm_inputs_embeds=v_, attention_mask=b.get("attention_mask"),
                position_ids=p_, mode="medium")
        lg = out["logits"][1].float()
        q0 = out["pred_codes"][0]
        # Q0 ОБЯЗАН СОВПАСТЬ С КЭШЕМ: цели построены от q0hat кэша, и если
        # модель выдаёт другой черновик, они относятся к другому остатку.
        cq0 = torch.from_numpy(
            np.asarray(q0hat_c[sel]).astype(np.int64)).to(dev)
        d_q0 = int((q0 != cq0).sum())
        tg = targets(sel)
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1))
        e0 = books[0][cq0.long()].float()
        if train:
            emb, _, _ = straight_through(lg, books[1].float(), tau=1.0)
        else:
            emb = books[1][lg.argmax(-1)].float()
        a_hat = decode_actions(e0 + emb)
        a_true = torch.from_numpy(
            np.asarray(ACT[sel], np.float32)).to(dev)[..., :7]
        w = torch.ones(7, device=dev, dtype=torch.float32)
        w[6] = float(a.grip_weight)
        dd = (a_hat[:, :H_EXEC] - a_true[:, :H_EXEC]) * w
        act_loss = (dd ** 2).mean()
        fb_reg = torch.zeros((), device=dev)
        if a.variant != "no_feedback":
            for p_ in model.depth_rvq_feedback[0].parameters():
                fb_reg = fb_reg + (p_.float() ** 2).sum()
        loss = ce + a.lambda_action * act_loss + a.lambda_fb * fb_reg
        return loss, ce, act_loss, a_hat.detach(), a_true, d_q0

    def evaluate(rows):
        """RMS первых восьми действий в единицах робота, argmax без ST."""
        se, n = 0.0, 0
        model.eval()
        with torch.no_grad():
            for po, sel in group_by_offset(rows, offs, a.batch):
                _l, _c, _al, a_hat, a_true, _d = run_batch(po, sel, False)
                q = torch.as_tensor(max_act_q[:7], device=dev,
                                    dtype=torch.float32).clone()
                q[-1] = 1.0
                dd = (a_hat[:, :H_EXEC] - a_true[:, :H_EXEC]) * q
                se += float((dd ** 2).sum())
                n += dd.numel()
        return float(np.sqrt(se / max(n, 1)))

    params = [p for n_, p in model.named_parameters() if n_ in set(info["names"])]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    rng = np.random.default_rng(a.seed)
    hist = []
    t0 = time.time()
    e0_val = evaluate(sets["val_sel"])
    hist.append(dict(epoch=0, train=None, val_sel=e0_val))
    print(f"  эпоха 0 (без обучения): val_sel RMS-8 {e0_val:.6f}")
    for ep in range(1, int(a.epochs) + 1):
        model.train()
        order = group_by_offset(sets["train"], offs, a.batch)
        rng.shuffle(order)
        run, nb, dq = 0.0, 0, 0
        for po, sel in order:
            opt.zero_grad(set_to_none=True)
            loss, ce, al, _ah, _at, d_q0 = run_batch(po, sel, True)
            loss.backward()
            opt.step()
            run += float(loss.detach()); nb += 1; dq += d_q0
        if dq:
            raise SystemExit(f"эпоха {ep}: q0 модели разошёлся с кэшем в {dq} "
                             f"позициях — цели относятся к другому остатку")
        v = evaluate(sets["val_sel"])
        hist.append(dict(epoch=ep, train=run / max(nb, 1), val_sel=v))
        print(f"  эпоха {ep}: потеря {run / max(nb, 1):.5f}, val_sel RMS-8 "
              f"{v:.6f} ({(time.time() - t0) / 60:.1f} мин)")

    best_ep, best_val = select_epoch(hist)
    print(f"\n  выбрана эпоха {best_ep} по val_sel "
          f"({best_val:.6f})")
    e_conf = evaluate(sets["val_confirm"])
    po_ = (orc.get("parts") or {}).get("val_confirm") or {}
    e_a0 = (po_.get("vs_action.A0") or {}).get("rms")
    e_or = (po_.get("vs_action.A01_ze") or {}).get("rms")
    g4 = check_gate4(e_a0, e_conf, e_or)
    print(f"  ПОДТВЕРЖДЕНИЕ (открыто один раз): RMS-8 {e_conf:.6f}")
    print(f"  Gate 4: захват {g4['capture']:.4f} при пороге "
          f"{g4['threshold']}, предел RMS {g4['rms_limit']:.6f} -> "
          f"{'ПРОЙДЕН' if g4['passed'] else 'НЕ ПРОЙДЕН'}")

    os.makedirs(os.path.dirname(os.path.abspath(out_p)) or ".", exist_ok=True)
    ck = dict(
        state={k: v.detach().cpu() for k, v in model.state_dict().items()
               if k in set(info["names"])},
        stage="q1", variant=a.variant, seed=int(a.seed),
        seed_kind="training/data-order: поздние головы инициализируются "
                  "детерминированно",
        trainable_names=info["names"], n_tensors=info["n_tensors"],
        n_params=info["n_params"], feedback_mask=info["feedback_mask"],
        q1_cache=a.q1_cache, q1_cache_sha1=man["labels_sha1"],
        oracle=a.oracle, oracle_sha1=sha12(a.oracle),
        joint_ckpt=a.joint_ckpt, joint_sha1=j_sha, cache=a.cache, ckpt=a.ckpt,
        selected_epoch=best_ep, val_sel=best_val, val_confirm=e_conf,
        gate4=g4, history=hist, epochs_run=int(a.epochs),
        lr=a.lr, wd=a.wd, batch=int(a.batch),
        lambda_action=a.lambda_action, lambda_fb=a.lambda_fb,
        grip_weight=a.grip_weight, device=str(dev), dtype=a.dtype,
        code_version=kb.code_version([
            os.path.abspath(__file__),
            os.path.join(here, "depth_rvq_joint12.py"),
            os.path.join(here, "k14b_build_q1_cache.py")]),
        script_sha1=sha12(os.path.abspath(__file__)))
    tmp = out_p + f".tmp.{os.getpid()}"
    torch.save(ck, tmp)
    os.replace(tmp, out_p)
    print(f"  сохранено: {out_p}")
    return 0 if g4["passed"] else 4


if __name__ == "__main__":
    sys.exit(main())
