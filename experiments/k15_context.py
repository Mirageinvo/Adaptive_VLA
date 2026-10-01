#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Общая загрузка для K-15b: данные, модель, кодек, гейт K-15a, сборка батча.

ЭТО ДОСЛОВНАЯ ВЫДЕРЖКА проверенной последовательности из
`k15_train_depth_rvq.py`, а не импорт из него. Причина одна: сам K-15
остаётся неизменным как воспроизводимый эксперимент, а его загрузка живёт
внутри `main()` и переиспользованию не подлежит.

КОПИЯ НЕ МОЖЕТ РАСХОДИТЬСЯ С ОРИГИНАЛОМ МОЛЧА. Внутри стоят те же гейты,
что в тренере: побитовая сверка канонического q0 с планом, три отпечатка
кодека через `k11a.check_fingerprints`, гейт K-15a по СОДЕРЖИМОМУ
архитектурных файлов и отпечаток замороженного. Любое отличие в загрузке
сдвинуло бы q0 и остановило запуск на первом же батче.

ЧИСТЫЕ ФУНКЦИИ НЕ КОПИРУЮТСЯ, А ИМПОРТИРУЮТСЯ из `k15_train_depth_rvq`:
проверка гейта, построчная ошибка, опора декодера, статистика чтения,
отпечатки замороженного. Расходиться им нечем.
"""
import argparse
import copy
import hashlib
import inspect
import json
import os
import sys

import numpy as np

import k15_train_depth_rvq as k15t
from k15_train_depth_rvq import H_EXEC, check_init_gate, sha12

# Умолчания повторяют тренер. Самопроверка ниже сверяет их с его исходником,
# чтобы probe и обучение не смотрели на разные кэши и чекпойнты.
DEFAULTS = dict(
    cache="data/k11a_joint12", q0="data/k14d/q0_b8_e0.npz",
    gate_r="reports/k14d/gate_r.json", joint_ckpt="data/k9d_ep3.pt",
    q1_init="data/k14c/q1_main_s0.pt",
    init_gate="reports/k15a/init_identity.json",
    ckpt="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO",
    root="third_party/actioncodec", cfg_path="config/eval/bar.yaml",
    variant="main", batch=8, device="cuda:1", dtype="float16",
    grip_weight=1.0, limit=0, allow_dirty=False, seed=0)

# Поля, которые обязаны быть в возвращаемом пространстве имён. Проверяются
# после сборки: забытое поле иначе всплыло бы `AttributeError` посреди
# длинного прогона.
# `np` здесь НЕТ намеренно: он импортируется на уровне модуля и в
# `locals()` функции не попадает. Потребители берут numpy сами.
REQUIRED_FIELDS = (
    "torch", "model", "codec", "quantizers", "decode_fp32",
    "decoder_context", "proc", "cfg", "dev", "dt", "build_batch",
    "weights", "weights_gate", "parts", "parts_full", "q0_can", "q0_prov",
    "ACT", "IMG", "st_n", "tsk", "offs", "max_act_q", "act_p99_dataset",
    "act_ref_prov", "vocab", "info", "codec_fp", "code_version",
    "gate_info", "joint_sha", "git_head", "dirty", "meta", "E",
    "nearest_code", "code_contribution", "tok", "k11a", "k14c", "kc",
    "q1_prov", "keys_sha", "N",
)


def add_common_arguments(parser):
    """Общие аргументы загрузки. Имена совпадают с тренером K-15."""
    parser.add_argument("--cache", default=DEFAULTS["cache"])
    parser.add_argument("--q0", default=DEFAULTS["q0"])
    parser.add_argument("--gate-r", default=DEFAULTS["gate_r"])
    parser.add_argument("--joint-ckpt", default=DEFAULTS["joint_ckpt"])
    parser.add_argument("--q1-init", default=DEFAULTS["q1_init"])
    parser.add_argument("--init-gate", default=DEFAULTS["init_gate"])
    parser.add_argument("--ckpt", default=DEFAULTS["ckpt"])
    parser.add_argument("--root", default=DEFAULTS["root"])
    parser.add_argument("--cfg-path", default=DEFAULTS["cfg_path"])
    parser.add_argument("--variant", default=DEFAULTS["variant"])
    parser.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    parser.add_argument("--device", default=DEFAULTS["device"])
    parser.add_argument("--dtype", default=DEFAULTS["dtype"])
    parser.add_argument("--grip-weight", type=float,
                        default=DEFAULTS["grip_weight"])
    parser.add_argument("--limit", type=int, default=DEFAULTS["limit"],
                        help="батчей на часть; 0 — вся часть. Эталон "
                             "диапазона считается по ПОЛНОМУ плану всегда")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    return parser


def build(a):
    """Поднимает окружение K-15 и возвращает его одним пространством имён.

    `locals()` возвращается целиком намеренно: перечислять поля руками
    значило бы однажды забыть одно. Обязательный набор сверяется ниже.
    """
    from types import SimpleNamespace
    if int(a.batch) != 8:
        raise SystemExit(
            f"батч {a.batch}: состав батча входит в определение "
            f"канонического q0, менять его нельзя")
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(a.root)
    for p in (root, os.path.join(root, "src"),
              os.path.join(root, "experiments"),
              os.path.abspath("third_party/actioncodec"), here):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    import k14_common as kc
    import k11a_build_hicora_cache as k11a
    import k11b_hicora_identity as k11b
    import k14c_train_q1 as k14c
    import k9h_multiarm_gate as k9h
    import depth_aligned_tokenizer as tok
    from depth_aligned_joint12 import (architecture_code_version,
                                       make_action_decoder,
                                       make_depth_aligned_joint12_class)
    from depth_rvq_joint12 import code_contribution, nearest_code
    from joint12_vla import make_joint12_class
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       VisionLanguageActionProcessor, dict_apply, get_cfg,
                       prompt_template)
    import actioncodec  # noqa: F401  регистрация типа модели в AutoConfig

    git_head, dirty, _arte = kc.check_code_clean(a.allow_dirty)
    print(f"  код: коммит {git_head}"
          + ("  (--allow-dirty)" if dirty else ""))
    dev = torch.device(a.device)
    dt = getattr(torch, a.dtype)
    torch.manual_seed(int(a.seed))
    np.random.seed(int(a.seed))

    # --- ДАННЫЕ, ПУТЬ K-14c БЕЗ ИЗМЕНЕНИЙ --------------------------------
    meta = json.load(open(f"{a.cache}.meta.json"))
    src = meta["cache"]
    d = np.load(src, allow_pickle=True)
    cmeta = json.loads(str(d["meta"]))
    N = int(meta["n_obs"])
    epi = np.asarray(d["episode"])[:N].astype(np.int64)
    stp = np.asarray(d["step"])[:N]
    keys_sha = hashlib.sha1(np.ascontiguousarray(
        np.stack([epi, stp])).tobytes()).hexdigest()[:12]
    if keys_sha != meta.get("keys_sha1"):
        raise SystemExit(f"ключи наблюдений {keys_sha} против "
                         f"{meta.get('keys_sha1')}")
    ACT = np.asarray(d["action"])[:N]
    offs = np.asarray(d["pos_offset"])[:N].astype(np.int64)
    tsk = np.asarray(d["task"])[:N]
    E = np.load(f"{a.cache}.codebooks.npy")
    # НОРМЫ СТРОК КНИГ ПО УРОВНЯМ. В остаточном квантователе нормы должны
    # падать с уровнем. Если нет — отношение |c2|/|z0| = 3.5, увиденное в
    # смоуке, объясняется самой книгой, а не головой.
    _norms = np.linalg.norm(np.asarray(E, np.float64), axis=-1)
    book_norms = {f"level{i}": dict(
        mean=float(_norms[i].mean()), median=float(np.median(_norms[i])),
        p95=float(np.percentile(_norms[i], 95)),
        max=float(_norms[i].max())) for i in range(_norms.shape[0])}
    print("  нормы строк книг кодека: " + "; ".join(
        f"{k}: медиана {v['median']:.4f}, среднее {v['mean']:.4f}, "
        f"макс {v['max']:.4f}" for k, v in book_norms.items()))
    IMG = np.load(os.path.join(os.path.dirname(src),
                               cmeta["images_file"]), mmap_mode="r")
    ds_repo, ds_rev = k11b.dataset_source(meta)
    st_n, _sm, _sh = kc.load_states(src, N, ds_repo, ds_rev, keys_sha,
                                    STATE_Q01, STATE_Q99)
    max_act_q = np.maximum(np.abs(ACTION_Q01), np.abs(ACTION_Q99))


    q0_can, _q0_def, q0_man, q0_prov = kc.load_canonical_q0(
        a.q0, gate_r_path=a.gate_r, n_obs=N, keys_sha=keys_sha,
        cache_meta_sha1=k11a.file_sha1(f"{a.cache}.meta.json"))
    plan = kc.load_plan(a.q0, q0_man)
    parts = {}
    for name, po, sel in plan:
        parts.setdefault(name, []).append((po, sel))
    # ПОЛНЫЙ СОСТАВ ЧАСТЕЙ СОХРАНЯЕТСЯ ДО УРЕЗАНИЯ: эталон диапазона
    # считается по ВСЕМ строкам обучения из плана, иначе в смоуке он вышел
    # бы по 16 строкам и гейт диапазона там ничего не значил бы. Урезается
    # только то, что идёт через модель.
    parts_full = {k: list(v) for k, v in parts.items()}
    if a.limit:
        parts = {k: v[:int(a.limit)] for k, v in parts.items()}
    # СМЕЩЕНИЯ ПЛАНА СВЕРЯЮТСЯ СО СТРОКАМИ. Батч, заявленный с одним
    # смещением, а собранный из строк с другим, дал бы другой q0 — и это
    # обнаружилось бы только побитовой сверкой, уже потратив проход.
    for name, po, sel in plan:
        if not bool((offs[sel] == po).all()):
            raise SystemExit(f"батч части {name} заявлен со смещением {po}, "
                             f"а строки имеют другое")
    if "train" not in parts or "val_sel" not in parts:
        raise SystemExit(f"в плане нет нужных частей: {sorted(parts)}")
    # ПОДТВЕРЖДАЮЩАЯ ПОЛОВИНА НЕ ФОРМИРУЕТСЯ ВОВСЕ. В K-14 она уже
    # прочитана, поэтому по §53 любое её использование здесь было бы
    # retrospective; открывать её тренером нельзя ни при каком исходе.
    parts.pop("val_confirm", None)
    n_train, n_val = len(parts["train"]), len(parts["val_sel"])
    print(f"  план {q0_prov['plan_sha1']}: train {n_train} батчей, "
          f"val_sel {n_val} батчей, батч {a.batch}")

    # ЭТАЛОН ДИАПАЗОНА ДЕЙСТВИЙ — В ТЕХ ЖЕ ЕДИНИЦАХ, ЧТО ВЫХОД ДЕКОДЕРА, И
    # ТОЛЬКО ПО СТРОКАМ ОБУЧЕНИЯ. Кэш хранит действие НОРМИРОВАННЫМ: в k9a
    # оно поделено на max_act_q, у схвата перевёрнут знак, и всё обрезано в
    # [-1, 1]; декодер возвращает его же. Поэтому эталон — p99 |действия| в
    # этих же единицах на тех же исполняемых позициях, а не физический
    # max_act_q.
    # СЧИТАЕТСЯ ПОСЛЕ ПЛАНА И ПО `parts["train"]`: прежняя версия брала
    # p99 по ВСЕМУ ACT, то есть и по строкам подтверждающей половины, а
    # этот эталон входит в гейт диапазона, значит в `eligible` и в
    # `candidate_level`. Поля «val_confirm НЕ ОТКРЫВАЛАСЬ» и
    # `val_confirm_used_for_selection=False` были при этом ложью.
    train_rows = np.unique(np.concatenate(
        [np.asarray(sel_, np.int64) for _po, sel_ in parts_full["train"]]))
    act_train = np.abs(np.asarray(
        ACT[train_rows][:, :H_EXEC, :7], np.float64)).reshape(-1, 7)
    act_p99_dataset = np.percentile(act_train, 99.0, axis=0)
    act_ref_prov = dict(
        scope="train (полный план, не урезанный --limit)",
        rows=int(train_rows.size),
        batches_in_plan=len(parts_full["train"]),
        batches_through_model=len(parts["train"]),
        positions=int(H_EXEC), values=int(act_train.shape[0]),
        statistic="p99 |action| per channel, normalized codec units",
        rows_sha1=hashlib.sha1(
            np.ascontiguousarray(train_rows).tobytes()).hexdigest()[:12])
    print(f"  эталон диапазона: p99 |действия| по {act_ref_prov['rows']} "
          f"строкам ОБУЧЕНИЯ ({act_ref_prov['rows_sha1']}), нормированные "
          f"единицы: " + ", ".join(f"{x:.3f}" for x in act_p99_dataset))

    # --- МОДЕЛЬ -----------------------------------------------------------
    cfg = get_cfg(os.path.join(root, a.cfg_path))
    cfg.TRAINING.ckpt_dir = a.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = a.ckpt
    Base = make_depth_aligned_joint12_class(
        make_joint12_class(SmolVLABlockwiseAR))
    model = Base.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    model.init_joint_fast(depth=int(meta["depth"]), head_dtype=torch.float32)
    refine_norm = copy.deepcopy(model.action_expert.norm)
    joint_sha = k11a.file_sha1(a.joint_ckpt)
    kc.load_joint12_strict(model, a.joint_ckpt, int(meta["depth"]),
                           (meta.get("source") or {}).get("weights_sha1"),
                           torch, dev)
    model.init_depth_aligned_rvq(refine_norm=refine_norm,
                                 books=torch.from_numpy(E),
                                 head_dtype=torch.float32,
                                 verbose_init=False)

    # ИНИЦИАЛИЗАЦИЯ G1 И FEEDBACK0 ИЗ K-14. Он уже создаёт осмысленный h18,
    # и стартовать с него дешевле, чем учить с нуля. Загрузка строгая:
    # белый список и отпечаток весов сверяются, иначе обучение шло бы от
    # неизвестного состояния.
    legacy_info = model.configure_joint_depth_rvq(
        stage="q1", variant=a.variant, verbose=False)
    q1_obj = torch.load(a.q1_init, map_location="cpu", weights_only=False)
    q1_prov = k9h.check_depthrvq_q1_ckpt(
        q1_obj, joint_sha1=joint_sha, expect_variant=a.variant,
        expect_q0_manifest_sha1=q0_prov["q0_manifest_sha1"])
    want = set(legacy_info["names"])
    st = q1_obj["state"]
    if set(q1_obj["trainable_names"]) != want or set(st) != want:
        raise SystemExit(
            f"белый список K-14 не совпал: нет {sorted(want - set(st))[:5]}, "
            f"лишние {sorted(set(st) - want)[:5]}")
    own = dict(model.named_parameters())
    with torch.no_grad():
        for k_, v_ in st.items():
            if tuple(own[k_].shape) != tuple(v_.shape):
                raise SystemExit(f"форма {k_}")
            if not torch.isfinite(v_).all():
                raise SystemExit(f"в {k_} есть nan или inf")
            own[k_].data.copy_(v_.to(own[k_].device, own[k_].dtype))
    loaded_sha = k14c.state_sha({k_: own[k_].detach().float().cpu().numpy()
                                 for k_ in want})
    if loaded_sha != str(q1_obj["selected_state_sha1"]):
        raise SystemExit(f"после загрузки K-14 отпечаток {loaded_sha}, в "
                         f"чекпойнте {q1_obj['selected_state_sha1']}")
    info = model.configure_depth_aligned_rvq(verbose=True)
    status = model.initialization_status()
    for key in ("c1_exact", "c2_exact", "books_not_aliased"):
        if not status.get(key):
            raise SystemExit(f"инициализация нарушена: {key} = "
                             f"{status.get(key)}")

    ac_ = proc.action_processor
    codec = ac_ if hasattr(ac_, "vq") else getattr(ac_, "codec", None)
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь")
    codec = codec.to(dev).eval()
    for p_ in codec.parameters():
        p_.requires_grad_(False)
    quantizers = list(codec.vq.quantizers)
    vocab = int(model.depth_aligned_book(1).shape[0])

    # --- КОДЕК СВЕРЯЕТСЯ ТАК ЖЕ, КАК В K-14c -----------------------------
    # Тренер декодирует действия этим кодеком, и все цели K-14 построены
    # его книгами. Другой кодек дал бы другие числа при полном совпадении
    # всего остального; в K-14c эта сверка уже есть, здесь её не было.
    with torch.no_grad():
        idx_all = torch.arange(int(codec.vocab_size), device=dev)[None, :]
        codec_books = torch.stack([
            q_.out_project(q_.decode_code(idx_all))[0]
            for q_ in quantizers]).float()
    if tuple(codec_books.shape) != tuple(np.asarray(E).shape):
        raise SystemExit(f"книги кодека {tuple(codec_books.shape)} против "
                         f"кэша {tuple(np.asarray(E).shape)}")
    book_gap = float((codec_books.cpu()
                      - torch.from_numpy(np.asarray(E, np.float32))
                      ).abs().max())
    if book_gap > 1e-5:
        raise SystemExit(f"книги кодека разошлись с кэшем на {book_gap:.3e}")
    codec_fp = {
        "codebooks_sha1": hashlib.sha1(np.ascontiguousarray(
            np.asarray(E, np.float32)).tobytes()).hexdigest()[:12],
        "codec_state_sha1": k11a.state_sha1(codec),
        "decoder_probe": k11a.decoder_probe(codec, codec_books, dev),
    }
    k11a.check_fingerprints(meta, codec_fp)
    print(f"  кодек сверен: книги {codec_fp['codebooks_sha1']}, веса "
          f"{codec_fp['codec_state_sha1']}, проба "
          f"{codec_fp['decoder_probe']}")

    # ДЕКОДЕР ОБЩИЙ С ГЕЙТОМ: одна функция, один контекст.
    decode_fp32, decoder_context = make_action_decoder(codec, dev.type)
    code_version = architecture_code_version(
        here, inspect.getfile(SmolVLABlockwiseAR), sha12)

    # --- ГЕЙТ K-15a ОБЯЗАТЕЛЕН -------------------------------------------
    gate_info = check_init_gate(
        a.init_gate,
        expect=dict(joint_sha1=joint_sha,
                    q1_sha1=k11a.file_sha1(a.q1_init),
                    plan_sha1=q0_prov["plan_sha1"],
                    compute_dtype=a.dtype,
                    device=str(dev)),
        file_sha=k11a.file_sha1,
        code_version=code_version,
        decoder_context=decoder_context,
        codec_fingerprints=codec_fp)
    print(f"  гейт K-15a: {gate_info['init_gate']}, запуск "
          f"{gate_info['init_gate_run_id']}, причинных проверок "
          f"{gate_info['init_gate_causal']}")

    weights = torch.ones(7, device=dev, dtype=torch.float32)
    weights_gate = torch.as_tensor(max_act_q[:7], device=dev,
                                   dtype=torch.float32).clone()
    weights_gate[-1] = float(a.grip_weight)

    def build_batch(po, sel):
        """Дословно как в K-14h/K-14o: свой вариант уже оказывался неверным."""
        image = torch.from_numpy(np.asarray(IMG[sel]))
        msgs = []
        for gi in sel:
            mm = prompt_template(
                st_n[gi], None, str(tsk[gi]),
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                action_token_len=cfg.MODEL.action_processor.token_len)
            mm[1]["content"] = mm[1]["content"][1:]
            msgs.append(mm)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        b = proc(text=texts, images=[[image[k].numpy()]
                                     for k in range(len(sel))],
                 return_tensors="pt", padding=True, padding_side="left",
                 action_processor_kwargs={"embodiment_ids": 0})
        return dict_apply(lambda x: x.to(dev, dt), b)

    # Эти имена нужны ВЫЗЫВАЮЩЕМУ через пространство имён, а не здесь;
    # строка делает это видимым и для линтера.
    _exported = (tok, nearest_code, code_contribution, k11a, k11b, k14c,
                 kc, k9h, weights, info, vocab, q1_prov)
    assert all(x is not None for x in _exported)

    ns = SimpleNamespace(**{k: v for k, v in locals().items()
                            if not k.startswith("__")})
    missing = [f for f in REQUIRED_FIELDS if not hasattr(ns, f)]
    if missing:
        raise SystemExit(
            f"в окружении не собрались поля {missing}: выдержка из тренера "
            f"разошлась с оригиналом")
    return ns


def selftest():
    """Без GPU проверяется набор аргументов и совпадение умолчаний."""
    import re
    parser = add_common_arguments(argparse.ArgumentParser())
    got = {x.dest for x in parser._actions if x.dest != "help"}
    assert got == set(DEFAULTS), sorted(got ^ set(DEFAULTS))
    src = open(inspect.getfile(k15t), encoding="utf-8").read()
    checked = 0
    for name in ("cache", "q0", "gate_r", "joint_ckpt", "q1_init",
                 "init_gate", "ckpt", "root", "cfg_path", "variant"):
        flag = "--" + name.replace("_", "-")
        m = re.search(re.escape(flag) + r'",\s*\n?\s*default="([^"]*)"', src)
        if m is None:
            m = re.search(re.escape(flag) + r'", default="([^"]*)"', src)
        assert m is not None, f"в тренере не нашёлся default для {flag}"
        assert DEFAULTS[name] == m.group(1), (name, DEFAULTS[name],
                                              m.group(1))
        checked += 1
    assert H_EXEC == 8
    # Выдержка обязана ссылаться на те же проверки, что тренер.
    body = inspect.getsource(build)
    for needed in ("check_fingerprints", "check_init_gate",
                   "load_canonical_q0", "check_code_clean", "parts_full"):
        assert needed in body, needed
    print(f"самопроверка k15_context пройдена: {len(got)} аргументов, "
          f"{checked} умолчаний сверены с тренером, выдержка содержит "
          f"все обязательные проверки")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="K-15b: общая загрузка")
    p.add_argument("--selftest", action="store_true")
    add_common_arguments(p)
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        raise SystemExit("это модуль; отдельно запускать нечего")
