#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15c: один дорогой проход по train и val_sel — всё, что нужно головам.

ЗАЧЕМ. В K-15b каждая гипотеза стоила шести часов, и почти всё это время
уходило в проход VLA, а не в обучение трёх тензоров. Здесь C1, читатель q1 и
backbone заморожены, поэтому состояния h18 и h24 ФИКСИРОВАНЫ: их достаточно
снять один раз. После этого головы выбора ранга обучаются за минуты, а
абляции (глубина, пулинг, функция потерь) почти ничего не стоят.

ЧТО ДЕЛАЕТСЯ НА КАЖДОМ БАТЧЕ

  1. Один проход `mode="full"`, q0 сверяется с каноническим побитово.
  2. Pre-hook перед `depth_rvq_norms[0]` снимает h18, перед
     `depth_rvq_norms[1]` — h24. Каждый обязан сработать ровно один раз.
     Хуки, а не правка архитектуры: изменение `depth_aligned_joint12.py`
     инвалидировало бы гейт K-15a.
  3. Порядок кодов q1 — ТОТ ЖЕ устойчивый `sort`, что в каноническом M2,
     по fp32-логитам. Берутся первые восемь кодов в каждой из 16 позиций.
  4. Восемь согласованных путей: путь j — j-й код порядка во ВСЕХ
     позициях. Каждый декодируется, и его построчная взвешенная MSE по
     первым восьми шагам действия записывается в `rank_costs`.
  5. Ещё одна MSE — черновика (decode z0); на val_sel дополнительно
     учитель (лучший латентный rank-path из десяти), чтобы тренер мог
     считать долю разрыва по задачам, а не только в целом.

О ТОМ, ЧТО ВИДИТ h24. Буквально, как в манифесте:

    h24 is produced after injecting the rank-0 q1 embedding into layers
    19-24; the selected alternative rank is applied after h24 and is not
    fed back through these layers.

ПУБЛИКАЦИЯ АТОМАРНАЯ. Всё пишется во временный каталог; агрегаты на
val_sel сверяются с каноническим M2 (ранг 0 — 0.142997, оракул по восьми —
0.111093, черновик и учитель); при расхождении кэш НЕ публикуется. Только
после всех проверок пишутся `manifest.json` с отпечатками каждого массива и
`COMPLETE`, прежний канонический кэш архивируется переименованием, и
временный каталог становится каноническим. Smoke пишется в отдельный
каталог с `canonical=false`, и тренер его без явного разрешения не берёт.
"""
import argparse
import hashlib
import inspect
import json
import os
import shutil
import sys
import time

import numpy as np

KIND = "k15c_rank_cache"
TOPK = 8
CODE_POSITIONS = 16
PARTS = ("train", "val_sel")
M2_REL_LIMIT = 1e-6          # ранг 0, оракул, черновик и учитель против M2
H24_FEEDBACK_NOTE = (
    "h24 is produced after injecting the rank-0 q1 embedding into layers "
    "19-24; the selected alternative rank is applied after h24 and is not "
    "fed back through these layers.")
RANK_COSTS_DEFINITION = (
    "rank_costs[i, j] = weighted row MSE over the first 8 action steps and 7 "
    "channels (k15_train_depth_rvq.weighted_row_error with weights_gate) of "
    "decode(z0 + C1[top8_codes[i, :, j]]), i.e. the j-th code of the stable "
    "descending q1-logit order applied at ALL 16 code positions")
ORDER_DEFINITION = (
    "torch.sort(q1_logits.float(), dim=-1, descending=True, stable=True)"
    ".indices — the same function as k15b_measure_soft.reader_order")
H18_MODES = ("auto", "full", "mean", "none")
REQUIRED_MANIFEST = (
    "kind", "canonical", "parts", "plan_sha1", "q0_prov", "codec",
    "decoder_context", "c1_sha1", "reader_checkpoint_sha1", "reader_point",
    "reader_state_sha1", "code_version", "joint_sha1", "topk",
    "code_positions", "action_error_positions", "channel_weights", "arrays",
    "rank_costs_definition", "order_definition", "h24_feedback_note",
    "h18_mode", "d_model", "e_dim", "ln_eps", "task_vocab", "m2_check")


def part_arrays(part, h18_mode):
    """Состав массивов части. Учитель — только на val_sel."""
    names = ["rows", "task_ids", "h24", "top8_codes", "top8_logprobs",
             "rank_costs", "draft_cost", "q0_codes", "q1_top1_codes"]
    if h18_mode == "full":
        names.append("h18")
    elif h18_mode == "mean":
        names.append("h18_lnmean")
    if part == "val_sel":
        names.append("teacher_cost")
    return names


def array_spec(name, n, d_model):
    """(dtype, shape) массива. Единственное место, где это определено."""
    T, K = CODE_POSITIONS, TOPK
    spec = {
        "rows": ("int64", (n,)),
        "task_ids": ("int16", (n,)),
        "h24": ("float16", (n, T, d_model)),
        "h18": ("float16", (n, T, d_model)),
        "h18_lnmean": ("float32", (n, d_model)),
        "top8_codes": ("uint16", (n, T, K)),
        "top8_logprobs": ("float32", (n, T, K)),
        "rank_costs": ("float32", (n, K)),
        "draft_cost": ("float32", (n,)),
        "teacher_cost": ("float32", (n,)),
        "q0_codes": ("uint16", (n, T)),
        "q1_top1_codes": ("uint16", (n, T)),
    }
    if name not in spec:
        raise ValueError(f"массива {name!r} нет")
    return spec[name]


def bytes_needed(n_rows_by_part, d_model, h18_mode):
    total = 0
    for part, n in n_rows_by_part.items():
        for nm in part_arrays(part, h18_mode):
            dt, shp = array_spec(nm, n, d_model)
            total += int(np.prod(shp)) * np.dtype(dt).itemsize
    return int(total)


def choose_h18_mode(requested, free_bytes, need_by_mode, margin=1.15):
    """Режим хранения h18. Отсутствие полного h18 эксперимент не блокирует."""
    if requested not in H18_MODES:
        raise ValueError(f"режим h18 {requested!r} не бывает: {H18_MODES}")
    if requested != "auto":
        if free_bytes < need_by_mode[requested] * margin:
            raise ValueError(
                f"режим {requested}: нужно "
                f"{need_by_mode[requested] / 2 ** 30:.1f} ГиБ с запасом, "
                f"свободно {free_bytes / 2 ** 30:.1f}")
        return requested
    for mode in ("full", "mean", "none"):
        if free_bytes >= need_by_mode[mode] * margin:
            return mode
    raise ValueError(f"не хватает места даже без h18: нужно "
                     f"{need_by_mode['none'] / 2 ** 30:.1f} ГиБ, свободно "
                     f"{free_bytes / 2 ** 30:.1f}")


def sha_file(path, chunk=1 << 24):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:12]


def task_vocabulary(tasks):
    """Отображение задач в int16 по отсортированному словарю всех задач."""
    vocab = sorted({str(t) for t in np.asarray(tasks).tolist()})
    if len(vocab) >= 2 ** 15:
        raise ValueError(f"задач {len(vocab)}: в int16 не поместятся")
    return vocab, {t: i for i, t in enumerate(vocab)}


def compare_m2(got, ref, limit=M2_REL_LIMIT):
    """Агрегаты val_sel против канонического M2. Пустой список — сошлось."""
    problems = []
    for key in ("rank0", "oracle8", "draft", "teacher"):
        if key not in ref or ref[key] is None:
            problems.append(f"в эталоне M2 нет {key}")
            continue
        if key not in got:
            problems.append(f"не посчитано {key}")
            continue
        r = abs(float(got[key]) - float(ref[key])) / max(abs(float(ref[key])),
                                                         1e-12)
        if not np.isfinite(r) or r > limit:
            problems.append(f"{key}: {got[key]!r} против M2 {ref[key]!r} "
                            f"(относительно {r:.2e} при пределе {limit})")
    return problems


def m2_reference(summary):
    """Четыре эталонных числа из сводки канонического M2."""
    rms = summary.get("rms") or {}
    teacher = summary.get("teacher") or {}
    return dict(rank0=rms.get("fixed_rank0"),
                oracle8=rms.get("aligned_rankpath_top8"),
                draft=teacher.get("draft"), teacher=teacher.get("rank"))


def validate_cache(root, expect=None, verify_sha=True, allow_smoke=False):
    """Проверка опубликованного кэша. Возвращает (манифест, проблемы).

    Fail-closed: нет COMPLETE, COMPLETE не от этого манифеста, нет поля,
    не тот dtype или форма, не тот отпечаток файла, smoke без разрешения,
    поле обстановки не совпало с ожидаемым — всё это проблемы.
    """
    problems = []
    mpath = os.path.join(root, "manifest.json")
    cpath = os.path.join(root, "COMPLETE")
    if not os.path.exists(mpath):
        return None, [f"нет {mpath}"]
    if not os.path.exists(cpath):
        return None, [f"нет {cpath}: кэш не дописан или не прошёл проверки"]
    with open(cpath, encoding="utf-8") as fh:
        stamp = fh.read().strip()
    if stamp != sha_file(mpath):
        problems.append("COMPLETE выписан для другого манифеста")
    with open(mpath, encoding="utf-8") as fh:
        man = json.load(fh)
    missing = [f for f in REQUIRED_MANIFEST if f not in man]
    if missing:
        problems.append(f"в манифесте нет полей {missing}")
        return man, problems
    if man.get("kind") != KIND:
        problems.append(f"kind {man.get('kind')!r}, ожидался {KIND!r}")
    if man.get("canonical") is not True and not allow_smoke:
        problems.append("кэш не канонический (smoke): без явного разрешения "
                        "не принимается")
    if int(man.get("topk", -1)) != TOPK:
        problems.append(f"topk {man.get('topk')}, ожидалось {TOPK}")
    if int(man.get("code_positions", -1)) != CODE_POSITIONS:
        problems.append("число кодовых позиций не то")
    if int(man.get("action_error_positions", -1)) != 8:
        problems.append("число шагов действия в ошибке не 8")
    if set(man.get("parts", {})) != set(PARTS):
        problems.append(f"части {sorted(man.get('parts', {}))}, ожидались "
                        f"{sorted(PARTS)}")
    for key, want in (expect or {}).items():
        if man.get(key) != want:
            problems.append(f"{key}: в кэше {man.get(key)!r}, ожидалось "
                            f"{want!r}")
    d_model = int(man["d_model"])
    for part, info in man.get("parts", {}).items():
        n = int(info.get("rows", -1))
        for nm in part_arrays(part, man["h18_mode"]):
            key = f"{part}_{nm}"
            ent = man["arrays"].get(key)
            if ent is None:
                problems.append(f"в манифесте нет массива {key}")
                continue
            dt, shp = array_spec(nm, n, d_model)
            if ent.get("dtype") != dt or tuple(ent.get("shape", ())) != shp:
                problems.append(f"{key}: в манифесте {ent.get('dtype')} "
                                f"{ent.get('shape')}, ожидалось {dt} "
                                f"{list(shp)}")
            path = os.path.join(root, ent.get("file", ""))
            if not os.path.exists(path):
                problems.append(f"нет файла {path}")
                continue
            arr = np.load(path, mmap_mode="r")
            if str(arr.dtype) != dt or tuple(arr.shape) != shp:
                problems.append(f"{key}: в файле {arr.dtype} {arr.shape}")
            del arr
            if verify_sha and sha_file(path) != ent.get("sha1"):
                problems.append(f"{key}: отпечаток файла не совпал")
    return man, problems


def publish_dir(tmp, final, stamp=None):
    """Прежний канонический каталог — в архив, временный — на его место."""
    archived = None
    if os.path.exists(final):
        archived = f"{final}.{stamp or time.strftime('%Y%m%dT%H%M%S')}.bak"
        if os.path.exists(archived):
            archived = f"{archived}.{os.getpid()}"
        os.replace(final, archived)
    os.replace(tmp, final)
    return archived


def selftest():
    import tempfile

    # --- СПЕЦИФИКАЦИЯ И РАЗМЕРЫ -----------------------------------------
    assert "teacher_cost" in part_arrays("val_sel", "full")
    assert "teacher_cost" not in part_arrays("train", "full")
    assert "h18" in part_arrays("train", "full")
    assert "h18_lnmean" in part_arrays("train", "mean")
    assert not any(n.startswith("h18") for n in part_arrays("train", "none"))
    assert array_spec("h24", 10, 4) == ("float16", (10, 16, 4))
    assert array_spec("top8_codes", 3, 4) == ("uint16", (3, 16, 8))
    assert np.iinfo(np.uint16).max >= 2047, "коды словаря 2048 в uint16"
    try:
        array_spec("нет", 1, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("принят несуществующий массив")
    n_by = {"train": 100, "val_sel": 10}
    full = bytes_needed(n_by, 64, "full")
    mean = bytes_needed(n_by, 64, "mean")
    none = bytes_needed(n_by, 64, "none")
    assert full > mean > none > 0
    assert full - none == 110 * 16 * 64 * 2, (full, none)

    # --- ВЫБОР РЕЖИМА h18 -----------------------------------------------
    need = {"full": 1000, "mean": 500, "none": 300}
    assert choose_h18_mode("auto", 2000, need) == "full"
    assert choose_h18_mode("auto", 800, need) == "mean"
    assert choose_h18_mode("auto", 400, need) == "none"
    for bad_free in (100,):
        try:
            choose_h18_mode("auto", bad_free, need)
        except ValueError as e:
            assert "не хватает" in str(e), e
        else:
            raise AssertionError("принято при нехватке места")
    try:
        choose_h18_mode("full", 600, need)
    except ValueError as e:
        assert "свободно" in str(e), e
    else:
        raise AssertionError("явный режим принят без места")
    try:
        choose_h18_mode("иной", 10 ** 9, need)
    except ValueError as e:
        assert "не бывает" in str(e), e
    else:
        raise AssertionError("принят несуществующий режим")

    # --- СЛОВАРЬ ЗАДАЧ --------------------------------------------------
    vocab, idx = task_vocabulary(["b", "a", "b", "c"])
    assert vocab == ["a", "b", "c"] and idx["b"] == 1

    # --- СВЕРКА С M2 ----------------------------------------------------
    ref = dict(rank0=0.142997, oracle8=0.111093, draft=0.145469,
               teacher=0.072892)
    assert compare_m2(dict(ref), ref) == []
    bad = dict(ref, oracle8=0.111094)
    assert any("oracle8" in p for p in compare_m2(bad, ref))
    assert any("draft" in p for p in compare_m2(
        {k: v for k, v in ref.items() if k != "draft"}, ref))
    assert any("нет rank0" in p for p in compare_m2(ref, dict(ref,
                                                              rank0=None)))
    assert any("teacher" in p for p in compare_m2(
        dict(ref, teacher=float("nan")), ref))
    summ = dict(rms=dict(fixed_rank0=0.1, aligned_rankpath_top8=0.09),
                teacher=dict(draft=0.2, rank=0.05))
    assert m2_reference(summ) == dict(rank0=0.1, oracle8=0.09, draft=0.2,
                                      teacher=0.05)

    # --- ПОЛНЫЙ ЦИКЛ ПУБЛИКАЦИИ И ПРОВЕРКИ НА МИНИАТЮРНОМ КЭШЕ -----------
    with tempfile.TemporaryDirectory() as td:
        d_model = 4
        parts = {"train": 3, "val_sel": 2}
        tmp = os.path.join(td, "cache.tmp")
        os.makedirs(tmp)
        arrays = {}
        for part, n in parts.items():
            for nm in part_arrays(part, "full"):
                dt, shp = array_spec(nm, n, d_model)
                fn = f"{part}_{nm}.npy"
                np.save(os.path.join(tmp, fn), np.zeros(shp, dt))
                arrays[f"{part}_{nm}"] = dict(
                    file=fn, dtype=dt, shape=list(shp),
                    sha1=sha_file(os.path.join(tmp, fn)))
        man = {k: None for k in REQUIRED_MANIFEST}
        man.update(kind=KIND, canonical=True, topk=TOPK,
                   code_positions=CODE_POSITIONS, action_error_positions=8,
                   parts={p: dict(rows=n) for p, n in parts.items()},
                   arrays=arrays, h18_mode="full", d_model=d_model,
                   c1_sha1="C")

        def write(root, m, complete=True, stamp=None):
            mp = os.path.join(root, "manifest.json")
            with open(mp, "w", encoding="utf-8") as fh:
                json.dump(m, fh)
            if complete:
                with open(os.path.join(root, "COMPLETE"), "w") as fh:
                    fh.write(stamp or sha_file(mp))

        final = os.path.join(td, "cache")
        write(tmp, man)
        assert publish_dir(tmp, final) is None
        got, probs = validate_cache(final, expect={"c1_sha1": "C"})
        assert probs == [], probs
        # ЧУЖАЯ ОБСТАНОВКА
        _m, probs = validate_cache(final, expect={"c1_sha1": "ИНАЯ"})
        assert any("c1_sha1" in p for p in probs), probs
        # ИЗМЕНЁННЫЙ ФАЙЛ
        np.save(os.path.join(final, "train_rank_costs.npy"),
                np.ones((3, 8), "float32"))
        _m, probs = validate_cache(final)
        assert any("отпечаток файла" in p for p in probs), probs
        # ПРОПУЩЕННАЯ СТРОКА: форма файла разошлась с манифестом
        np.save(os.path.join(final, "train_rank_costs.npy"),
                np.zeros((2, 8), "float32"))
        _m, probs = validate_cache(final, verify_sha=False)
        assert any("в файле" in p for p in probs), probs
        # SMOKE НЕ ПРИНИМАЕТСЯ БЕЗ РАЗРЕШЕНИЯ
        tmp2 = os.path.join(td, "cache.tmp2")
        shutil.copytree(final, tmp2)
        np.save(os.path.join(tmp2, "train_rank_costs.npy"),
                np.zeros((3, 8), "float32"))
        write(tmp2, dict(man, canonical=False))
        archived = publish_dir(tmp2, final, stamp="S")
        assert archived and archived.endswith(".S.bak"), archived
        assert os.path.exists(archived)
        _m, probs = validate_cache(final)
        assert any("smoke" in p for p in probs), probs
        _m, probs = validate_cache(final, allow_smoke=True)
        assert probs == [], probs
        # COMPLETE НЕ ОТ ЭТОГО МАНИФЕСТА
        write(final, dict(man, canonical=False), stamp="0" * 12)
        _m, probs = validate_cache(final, allow_smoke=True)
        assert any("COMPLETE" in p for p in probs), probs
        # НЕТ COMPLETE ВОВСЕ
        os.unlink(os.path.join(final, "COMPLETE"))
        _m, probs = validate_cache(final, allow_smoke=True)
        assert probs and "COMPLETE" in probs[0], probs
        # НЕТ ПОЛЯ МАНИФЕСТА
        write(final, {k: v for k, v in man.items() if k != "plan_sha1"})
        _m, probs = validate_cache(final)
        assert any("plan_sha1" in p for p in probs), probs
    print("самопроверка k15c_build_rank_cache пройдена: спецификация, "
          "режимы h18, сверка с M2 и полный цикл публикации с семью "
          "мутациями")


def load_stack(a, need_reader=True):
    """Каноническая обстановка K-15b с книгой и читателем e0s0.

    Общая для построителя и проверки вывода: оба обязаны видеть одну и ту
    же модель. Проверки — те же, что в M1/M2: книга с `accepted=True`,
    побитовый отпечаток замороженного, совместимость чекпойнта читателя
    fail-closed, отпечаток состояния после подстановки.
    """
    from types import SimpleNamespace
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import k15_context
    import k15b_measure_interface as mi
    import k15b_train_stagewise as trainer

    ctx = k15_context.build(a)
    torch = ctx.torch
    model = ctx.model
    k15t = k15_context.k15t
    book = torch.load(a.c1, map_location="cpu", weights_only=False)
    if book.get("kind") != "k15b_c1_selected" \
            or book.get("accepted") is not True:
        raise SystemExit(f"{a.c1}: kind {book.get('kind')!r}, accepted "
                         f"{book.get('accepted')!r}")
    for key, want in (("codec", ctx.codec_fp),
                      ("code_version", ctx.code_version),
                      ("joint_sha1", ctx.joint_sha)):
        if book.get(key) != want:
            raise SystemExit(f"{a.c1}: {key} выбран в другой обстановке")

    def book_sha():
        return hashlib.sha1(np.ascontiguousarray(
            model.depth_aligned_book(1).detach().float().cpu().numpy()
        ).tobytes()).hexdigest()[:12]

    if book_sha() != book["c1_sha1"]:
        with torch.no_grad():
            model.depth_aligned_c1.copy_(book["c1"].to(
                model.depth_aligned_c1.device, model.depth_aligned_c1.dtype))
    if book_sha() != book["c1_sha1"]:
        raise SystemExit("после загрузки книга не та")
    train_names = trainer.phase_names(list(ctx.info["names"]), "q1_reader")
    frozen_sha, n_frozen, n_elem = k15t.frozen_content_sha(
        model, torch, set(train_names))
    print(f"  книга {book['c1_sha1']}; замороженного {n_frozen} тензоров "
          f"({n_elem} значений), отпечаток {frozen_sha}")

    point, point_sha, reader_file_sha = None, None, None
    if need_reader:
        # Кэш целей K-15b нужен только чтобы привязать чекпойнт читателя к
        # той цели, на которой он записан; его массивы здесь не читаются.
        tmeta = json.loads(str(np.load(a.k15b_target,
                                       allow_pickle=True)["meta"]))
        obj = torch.load(a.checkpoint, map_location="cpu",
                         weights_only=False)
        if str(obj.get("kind")) not in mi.READER_KINDS:
            raise SystemExit(f"{a.checkpoint} описывает {obj.get('kind')!r}")
        problems = mi.check_reader(obj, ctx, book, tmeta, frozen_sha,
                                   train_names)
        if problems:
            raise SystemExit("чекпойнт читателя собран в другой обстановке: "
                             + "; ".join(problems[:6]))
        sel_tag, states, _has_all = mi.reader_points(obj)
        point = trainer.ZERO_TAG
        if point not in states:
            raise SystemExit(f"в чекпойнте нет точки {point}")
        own = dict(model.state_dict())
        with torch.no_grad():
            for k_, v_ in states[point].items():
                if tuple(v_.shape) != tuple(own[k_].shape) \
                        or not torch.isfinite(v_).all():
                    raise SystemExit(f"{point}.{k_}: форма или значения")
                own[k_].copy_(v_.to(own[k_].device, own[k_].dtype))
        point_sha = ctx.k14c.state_sha(
            {k_: model.state_dict()[k_].detach().float().cpu().numpy()
             for k_ in train_names})
        if point == sel_tag and point_sha != str(obj["selected_state_sha1"]):
            raise SystemExit(f"после загрузки отпечаток {point_sha}, в "
                             f"чекпойнте {obj['selected_state_sha1']}")
        again, _n, _e = k15t.frozen_content_sha(model, torch,
                                                set(train_names))
        if again != frozen_sha:
            raise SystemExit("подстановка тронула не только белый список")
        reader_file_sha = ctx.k11a.file_sha1(a.checkpoint)
        print(f"  читатель: точка {point} ({point_sha}), файл "
              f"{reader_file_sha}")
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    return SimpleNamespace(ctx=ctx, torch=torch, model=model, k15t=k15t,
                           book=book, train_names=train_names,
                           frozen_sha=frozen_sha, point=point,
                           point_sha=point_sha,
                           reader_file_sha=reader_file_sha)


def add_stack_arguments(ap):
    import k15_context
    k15_context.add_common_arguments(ap)
    ap.add_argument("--c1", default="data/k15b/c1_selected.pt")
    ap.add_argument("--k15b-target",
                    default="data/k15b/rankpath_target_train.npz")
    ap.add_argument("--checkpoint", default="data/k15b/q1_reader_s0.pt")
    return ap


class StateHooks:
    """Pre-hooks на входы двух поздних норм. Считают срабатывания."""

    def __init__(self, model):
        self.grab, self.calls = {}, {"h18": 0, "h24": 0}
        self.handles = [
            model.depth_rvq_norms[0].register_forward_pre_hook(
                self._make("h18")),
            model.depth_rvq_norms[1].register_forward_pre_hook(
                self._make("h24"))]

    paused = False

    def _make(self, name):
        def hook(_m, inp):
            if self.paused:
                return
            self.grab[name] = inp[0].detach()
            self.calls[name] += 1
        return hook

    def take(self):
        if self.calls != {"h18": 1, "h24": 1}:
            raise SystemExit(f"хуки сработали {self.calls}, а должны ровно "
                             f"по одному разу за проход")
        out = self.grab
        self.grab, self.calls = {}, {"h18": 0, "h24": 0}
        return out["h18"], out["h24"]

    def remove(self):
        for h in self.handles:
            h.remove()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    ap = argparse.ArgumentParser(
        description="K-15c: кэш состояний h18/h24 и восьми ранговых ошибок")
    add_stack_arguments(ap)
    ap.add_argument("--m2", default="reports/k15b/measure_soft.json",
                    help="сводка канонического M2: эталон для сверки")
    ap.add_argument("--out", default="data/k15c/rank_cache")
    ap.add_argument("--h18", default="auto", choices=H18_MODES)
    ap.add_argument("--smoke-batches", type=int, default=0,
                    help="0 — канонический полный кэш; иначе первые N "
                         "батчей каждой части, canonical=false")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-failed", action="store_true",
                    help="не удалять временный каталог при отказе")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if int(a.limit) != 0:
        raise SystemExit("--limit не применяется: урезание — --smoke-batches")
    smoke = int(a.smoke_batches) > 0
    final = a.out + ("_smoke" if smoke else "")
    if os.path.exists(final) and not a.overwrite:
        raise SystemExit(f"{final} уже существует: без --overwrite не "
                         f"перезаписываю (прежний уйдёт в архив)")
    if not smoke:
        if not os.path.exists(a.m2):
            raise SystemExit(f"нет {a.m2}: сверять канонический кэш не с чем")
        with open(a.m2, encoding="utf-8") as fh:
            m2 = json.load(fh)
        m2_ref = m2_reference(m2)
        missing_ref = [k for k, v in m2_ref.items() if v is None]
        if missing_ref:
            raise SystemExit(f"в {a.m2} нет эталонов {missing_ref}")
    else:
        m2, m2_ref = None, None

    import k15b_measure_soft as ms
    import k15b_probe_and_extract as probe
    import k15c_rank_selector as rs
    S = load_stack(a)
    ctx, torch, model, k15t = S.ctx, S.torch, S.model, S.k15t
    if m2 is not None:
        for key, want in (("c1_sha1", S.book["c1_sha1"]),):
            if m2.get(key) != want:
                raise SystemExit(f"M2 снят с другой книгой: {key}")
        if (m2.get("reader") or {}).get("point_state_sha1") != S.point_sha:
            raise SystemExit("M2 снят с другим состоянием читателя")
    dev = ctx.dev
    import torch.nn.functional as F

    parts = {p: list(ctx.parts_full[p]) for p in PARTS}
    if "val_confirm" in parts:
        raise SystemExit("val_confirm не открывается ни при каком исходе")
    if smoke:
        # SMOKE БЕРЁТ БАТЧИ РАВНОМЕРНО ПО ЧАСТИ, А НЕ ПЕРВЫЕ ПОДРЯД. Первые
        # батчи плана — соседние кадры одних эпизодов: почти одинаковые
        # признаки при скачущем лучшем ранге. На таком наборе проба головы
        # упиралась в плато и давала ложный технический отказ всех голов.
        parts = {p: [v[i] for i in probe.strided(len(v), int(a.smoke_batches))]
                 for p, v in parts.items()}
    import k15b_build_rankpath_cache as cachelib
    index = {p: cachelib.build_row_index(v) for p, v in parts.items()}
    n_rows = {p: int(index[p][0].size) for p in PARTS}
    d_model = int(model.depth_rvq_norms[0].weight.shape[0])
    c1 = model.depth_aligned_book(1)
    e_dim = int(c1.shape[1])
    if int(model.block_size) != CODE_POSITIONS:
        raise SystemExit(f"кодовых позиций {model.block_size}")
    vocab = int(ctx.vocab)
    if vocab - 1 > np.iinfo(np.uint16).max:
        raise SystemExit("коды словаря не помещаются в uint16")

    os.makedirs(os.path.dirname(os.path.abspath(final)) or ".",
                exist_ok=True)
    free = shutil.disk_usage(
        os.path.dirname(os.path.abspath(final)) or ".").free
    need_by_mode = {m: bytes_needed(n_rows, d_model, m)
                    for m in ("full", "mean", "none")}
    try:
        h18_mode = choose_h18_mode(a.h18, free, need_by_mode)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"  строк: train {n_rows['train']}, val_sel {n_rows['val_sel']}; "
          f"d_model {d_model}, размерность книги {e_dim}")
    print(f"  размер кэша: с полным h18 "
          f"{need_by_mode['full'] / 2 ** 30:.1f} ГиБ, со средним "
          f"{need_by_mode['mean'] / 2 ** 30:.1f}, без h18 "
          f"{need_by_mode['none'] / 2 ** 30:.1f}; свободно "
          f"{free / 2 ** 30:.1f} ГиБ -> режим h18: {h18_mode}")

    vocab_t, task_idx = task_vocabulary(ctx.tsk)
    tmp = final + f".tmp.{os.getpid()}"
    os.makedirs(tmp)
    ok = False
    try:
        mm = {}
        for p in PARTS:
            for nm in part_arrays(p, h18_mode):
                dt, shp = array_spec(nm, n_rows[p], d_model)
                mm[(p, nm)] = np.lib.format.open_memmap(
                    os.path.join(tmp, f"{p}_{nm}.npy"), mode="w+",
                    dtype=np.dtype(dt), shape=shp)
        filled = {p: np.zeros(n_rows[p], bool) for p in PARTS}
        hooks = StateHooks(model)
        q0_dev = torch.as_tensor(np.asarray(ctx.q0_can), device=dev)
        ac16 = torch.autocast(device_type=dev.type, dtype=ctx.dt)
        topk_teacher = int(S.book["topk"])
        agg = {p: dict(rank0=0.0, oracle8=0.0, draft=0.0, teacher=0.0,
                       n=0) for p in PARTS}
        lossy = dict(h18=0, h24=0)
        total_batches = sum(len(v) for v in parts.values())
        done, t0, forecast = 0, time.time(), None
        with torch.no_grad():
            for p in PARTS:
                rows_p, slot = index[p]
                for po, sel in parts[p]:
                    b = ctx.build_batch(po, sel)
                    with ac16:
                        v, p_ids = model.build_inputs(position_offset=po, **b)
                        out = model.forward_depth_aligned_rvq(
                            vlm_inputs_embeds=v,
                            attention_mask=b.get("attention_mask"),
                            position_ids=p_ids, mode="full", tau=1.0)
                    sel_t = torch.as_tensor(sel, device=dev)
                    if int((out["pred_codes"][0] != q0_dev[sel_t]).sum()):
                        raise SystemExit(f"q0 разошёлся с каноническим на "
                                         f"части {p}")
                    h18, h24 = hooks.take()
                    B = len(sel)
                    for nm_, h_ in (("h18", h18), ("h24", h24)):
                        if tuple(h_.shape) != (B, CODE_POSITIONS, d_model):
                            raise SystemExit(f"{nm_}: форма "
                                             f"{tuple(h_.shape)}")
                        if not torch.isfinite(h_).all():
                            raise SystemExit(f"{nm_}: nan или inf")
                        n_l = int((h_.half().to(h_.dtype) != h_).sum())
                        lossy[nm_] += n_l
                        if n_l:
                            # ХРАНЕНИЕ В fp16 ДОКАЗЫВАЕТСЯ, А НЕ
                            # ПРЕДПОЛАГАЕТСЯ: коды, пересчитанные из
                            # сохранённого состояния в том же autocast,
                            # обязаны совпасть с живыми. Хуки на пересчёт
                            # приостановлены — иначе он засчитался бы как
                            # второе срабатывание.
                            lev = 1 if nm_ == "h18" else 2
                            hooks.paused = True
                            with ac16:
                                lg_s = model._joint_depth_logits(
                                    h_.half().to(h_.dtype), lev)
                            hooks.paused = False
                            flips = int((lg_s.argmax(-1)
                                         != out["pred_codes"][lev]).sum())
                            if flips:
                                raise SystemExit(
                                    f"хранение {nm_} в fp16 меняет {flips} "
                                    f"кодов уровня {lev}: кэш был бы не тем "
                                    f"состоянием")
                    lg = out["logits"][1].float()
                    order = ms.reader_order(lg, torch)
                    top8 = order[..., :TOPK]
                    pred = out["pred_codes"][1]
                    if not torch.equal(top8[..., 0], pred):
                        raise SystemExit(
                            f"первый код порядка разошёлся с исполняемым q1 "
                            f"на части {p}")
                    if done == 0:
                        srt = top8.sort(dim=-1).values
                        if int((srt[..., 1:] == srt[..., :-1]).sum()):
                            raise SystemExit("top-8 содержит повторы")
                    lp8 = F.log_softmax(lg, dim=-1).gather(-1, top8)
                    z0 = out["policy_embeddings"][0]
                    action = torch.from_numpy(np.asarray(
                        ctx.ACT[sel], np.float32)).to(dev)[..., :7]
                    costs = []
                    for j in range(TOPK):
                        rows_j, _m = k15t.weighted_row_error(
                            ctx.decode_fp32(z0 + c1[top8[..., j]]), action,
                            ctx.weights_gate, torch)
                        costs.append(rows_j)
                    cost = torch.stack(costs, 1)                # [B, 8]
                    draft_rows, m0 = k15t.weighted_row_error(
                        ctx.decode_fp32(z0), action, ctx.weights_gate, torch)
                    if not (torch.isfinite(cost).all()
                            and torch.isfinite(draft_rows).all()):
                        raise SystemExit("ошибки путей не конечны")
                    w = float(B)
                    # АГРЕГАТЫ НАКАПЛИВАЮТСЯ ТАК ЖЕ, КАК В M2: среднее по
                    # батчу, умноженное на его размер. Иначе сравнение с
                    # эталоном расходилось бы в последних разрядах из-за
                    # порядка суммирования, а не из-за содержания.
                    agg[p]["rank0"] += float(cost[:, 0].mean()) * w
                    agg[p]["oracle8"] += float(cost.min(1).values.mean()) * w
                    agg[p]["draft"] += float(m0) * w
                    agg[p]["n"] += B
                    if p == "val_sel":
                        z_e = codec_encode(ctx, action)
                        d1 = ctx.tok.mean_squared_distances(z_e - z0, c1)
                        _l, _e2, i1, _p = ctx.tok.quantize_residual(
                            z_e - z0, c1, temperature=1.0)
                        near = probe.rank_candidates(d1, i1, topk_teacher,
                                                     torch)
                        errs = []
                        for j in range(near.shape[-1]):
                            rows_j, _m = k15t.weighted_row_error(
                                ctx.decode_fp32(z0 + c1[near[..., j]]),
                                action, ctx.weights_gate, torch)
                            errs.append(rows_j)
                        teach_rows = torch.stack(errs, 0).min(0).values
                        agg[p]["teacher"] += float(teach_rows.mean()) * w
                    ii = np.asarray([slot[int(r)] for r in sel], np.int64)
                    if filled[p][ii].any():
                        raise SystemExit(f"строка части {p} пишется дважды")
                    mm[(p, "rows")][ii] = np.asarray(sel, np.int64)
                    mm[(p, "task_ids")][ii] = np.asarray(
                        [task_idx[str(ctx.tsk[r])] for r in sel], np.int16)
                    mm[(p, "h24")][ii] = h24.half().cpu().numpy()
                    if h18_mode == "full":
                        mm[(p, "h18")][ii] = h18.half().cpu().numpy()
                    elif h18_mode == "mean":
                        mm[(p, "h18_lnmean")][ii] = rs.ln_mean_pool(
                            h18, torch).cpu().numpy()
                    mm[(p, "top8_codes")][ii] = top8.cpu().numpy().astype(
                        np.uint16)
                    mm[(p, "top8_logprobs")][ii] = lp8.cpu().numpy()
                    mm[(p, "rank_costs")][ii] = cost.cpu().numpy()
                    mm[(p, "draft_cost")][ii] = draft_rows.cpu().numpy()
                    mm[(p, "q0_codes")][ii] = out["pred_codes"][0] \
                        .cpu().numpy().astype(np.uint16)
                    mm[(p, "q1_top1_codes")][ii] = pred.cpu().numpy() \
                        .astype(np.uint16)
                    if p == "val_sel":
                        mm[(p, "teacher_cost")][ii] = \
                            teach_rows.cpu().numpy()
                    filled[p][ii] = True
                    done += 1
                    if done == min(100, total_batches):
                        forecast = k15t.forecast_runtime(
                            time.time() - t0, done, total_batches, 1)
                        print(f"    прогноз: {forecast['per_batch_s']:.2f} "
                              f"с/батч, весь проход "
                              f"{forecast['total_h']:.2f} ч", flush=True)
                    if done % 1000 == 0:
                        print(f"    {done}/{total_batches} батчей, "
                              f"{(time.time() - t0) / 60:.1f} мин",
                              flush=True)
        hooks.remove()
        for p in PARTS:
            if not filled[p].all():
                raise SystemExit(f"часть {p}: не заполнено "
                                 f"{int((~filled[p]).sum())} строк")
            if not np.array_equal(np.asarray(mm[(p, "rows")]), index[p][0]):
                raise SystemExit(f"часть {p}: строки не в порядке плана")
        for arr in mm.values():
            arr.flush()

        # ХРАНЕНИЕ В fp16 ДОКАЗЫВАЕТСЯ: если округление что-то меняет,
        # проверяется, что пересчёт кодов из сохранённых состояний даёт те
        # же коды, иначе кэш был бы не тем состоянием.
        print(f"  хранение в fp16: изменённых значений h18 {lossy['h18']}, "
              f"h24 {lossy['h24']}")

        result = {p: {k: (float(np.sqrt(v / max(agg[p]["n"], 1)))
                          if k != "n" else int(v))
                      for k, v in agg[p].items()} for p in PARTS}
        # ТО ЖЕ ИЗ ЗАПИСАННЫХ МАССИВОВ: тренер будет считать именно так.
        for p in PARTS:
            rc = np.asarray(mm[(p, "rank_costs")], np.float64)
            from_arr = dict(rank0=float(np.sqrt(rc[:, 0].mean())),
                            oracle8=float(np.sqrt(rc.min(1).mean())))
            for k_, v_ in from_arr.items():
                rel = abs(v_ - result[p][k_]) / max(result[p][k_], 1e-12)
                if rel > 1e-6:
                    raise SystemExit(f"{p}.{k_}: из массивов {v_!r}, при "
                                     f"проходе {result[p][k_]!r}")
        print("  агрегаты: " + "; ".join(
            f"{p}: ранг 0 {result[p]['rank0']:.6f}, оракул восьми "
            f"{result[p]['oracle8']:.6f}, черновик {result[p]['draft']:.6f}"
            for p in PARTS)
            + f"; учитель val_sel {result['val_sel']['teacher']:.6f}")

        m2_check = dict(skipped=bool(smoke), limit=M2_REL_LIMIT,
                        reference=m2_ref, got=result["val_sel"])
        if not smoke:
            problems = compare_m2(result["val_sel"], m2_ref)
            m2_check["problems"] = problems
            if problems:
                raise SystemExit("агрегаты val_sel не воспроизвели M2, кэш "
                                 "не публикуется: " + "; ".join(problems))
            print(f"  val_sel воспроизвёл M2 (предел {M2_REL_LIMIT})")
        else:
            print("  SMOKE: части урезаны, сверка с M2 не выполняется")

        del mm
        arrays = {}
        for p in PARTS:
            for nm in part_arrays(p, h18_mode):
                fn = f"{p}_{nm}.npy"
                dt, shp = array_spec(nm, n_rows[p], d_model)
                arrays[f"{p}_{nm}"] = dict(file=fn, dtype=dt,
                                           shape=list(shp),
                                           sha1=sha_file(os.path.join(tmp,
                                                                      fn)))
        man = dict(
            kind=KIND, canonical=not smoke,
            smoke_batches=(int(a.smoke_batches) if smoke else None),
            parts={p: dict(rows=n_rows[p], batches=len(parts[p]),
                           rows_sha1=hashlib.sha1(np.ascontiguousarray(
                               index[p][0]).tobytes()).hexdigest()[:12],
                           aggregates=result[p])
                   for p in PARTS},
            plan_sha1=ctx.q0_prov.get("plan_sha1"), q0_prov=ctx.q0_prov,
            codec=ctx.codec_fp, decoder_context=ctx.decoder_context,
            c1_sha1=S.book["c1_sha1"],
            reader_checkpoint=os.path.abspath(a.checkpoint),
            reader_checkpoint_sha1=S.reader_file_sha,
            reader_point=S.point, reader_state_sha1=S.point_sha,
            reader_trainable_names=list(S.train_names),
            frozen_content_sha=S.frozen_sha,
            code_version=ctx.code_version, joint_sha1=ctx.joint_sha,
            gate=ctx.gate_info, topk=TOPK, code_positions=CODE_POSITIONS,
            action_error_positions=8,
            channel_weights=[float(x) for x in
                             ctx.weights_gate.detach().cpu().numpy()],
            arrays=arrays, rank_costs_definition=RANK_COSTS_DEFINITION,
            order_definition=ORDER_DEFINITION,
            order_function_sha1=k15_context_sha12(ms.reader_order),
            h24_feedback_note=H24_FEEDBACK_NOTE, h18_mode=h18_mode,
            h18_lnmean_definition=(
                "mean over 16 positions of F.layer_norm(h18.float(), (D,), "
                f"eps={rs.LN_EPS}) without affine" if h18_mode == "mean"
                else None),
            d_model=d_model, e_dim=e_dim, ln_eps=rs.LN_EPS,
            task_vocab=vocab_t, teacher_topk=topk_teacher,
            fp16_storage_changed_values=lossy, m2_check=m2_check,
            forecast=forecast, seconds=float(time.time() - t0),
            git_head=ctx.git_head, git_dirty=bool(ctx.dirty))
        mpath = os.path.join(tmp, "manifest.json")
        with open(mpath, "w", encoding="utf-8") as fh:
            json.dump(man, fh, ensure_ascii=False, indent=1, allow_nan=False,
                      default=k15t.json_scalar)
        with open(os.path.join(tmp, "COMPLETE"), "w") as fh:
            fh.write(sha_file(mpath))
        _m, probs = validate_cache(tmp, allow_smoke=smoke)
        if probs:
            raise SystemExit("готовый кэш не прошёл собственную проверку: "
                             + "; ".join(probs[:6]))
        archived = publish_dir(tmp, final)
        ok = True
        print(f"  опубликовано: {final}"
              + (f" (прежний — в {archived})" if archived else ""))
        print(f"  {'SMOKE, canonical=false' if smoke else 'канонический'}; "
              f"за {(time.time() - t0) / 60:.1f} мин")
    finally:
        if not ok and os.path.exists(tmp) and not a.keep_failed:
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"  временный каталог {tmp} удалён")
    return 0


def codec_encode(ctx, action):
    return ctx.codec._encode(action.float(), embodiment_ids=0).float()


def k15_context_sha12(fn):
    import k15_context
    return k15_context.sha12(inspect.getfile(fn))


if __name__ == "__main__":
    sys.exit(main())
