#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15a: bitwise initialization identity of K-14 and Depth-RVQ.

This gate runs before any K-15 optimization.  In one model with one set of
weights it compares:

* the unchanged K-14 ``forward_joint_depth_rvq``;
* the new ``forward_depth_aligned_rvq`` with C1/C2 copied from ActionCodec.

What is proven bitwise, stated exactly:

* q0->q1 logits and codes are bitwise equal between the two paths;
* the latent and action reconstructed from HARD CODE ROWS agree bitwise --
  that reference is rebuilt here as ``books[0][k0] + books[1][k1]``, so it is
  an idealized sum, NOT the value the legacy path would hand the decoder;
* the legacy path builds its level>=1 embedding as ``emb + (hard -
  emb).detach()``, whose value is not the book row.  That error is MEASURED
  and recorded as ``legacy_st_error`` instead of being assumed;
* with feedback1 zeroed (its projection is zero-initialized) q2 is bitwise
  equal; with a NON-ZERO feedback1 the legacy error reaches q2, so the two
  paths are compared there against a recorded relative limit, with the
  number of flipped q2 codes written down.

The output JSON is a required training provenance artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np


# Масштаб детерминированного узора для зонда feedback1. Мал настолько, чтобы
# не выводить скрытые состояния из рабочего диапазона, и велик настолько, чтобы
# изменение уровня 2 было заведомо больше шума fp16.
FEEDBACK_PROBE_SCALE = 0.05


def sha12(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:12]


def compare_exact(name, left, right, output):
    """Bitwise array comparison with no numerical tolerance."""
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise SystemExit(f"{name}: shapes {left.shape} versus {right.shape}")
    if str(left.dtype) != str(right.dtype):
        raise SystemExit(f"{name}: dtypes {left.dtype} versus {right.dtype}")
    equal = bool(np.array_equal(left, right))
    if equal:
        max_abs = 0.0
        n_differing = 0
    else:
        max_abs = float(np.max(np.abs(
            left.astype(np.float64) - right.astype(np.float64))))
        n_differing = int(np.count_nonzero(left != right))
    output[name] = {
        "equal": equal,
        "max_abs_diff": max_abs,
        "n_differing": n_differing,
        "size": int(left.size),
    }
    if not equal:
        raise SystemExit(
            f"{name}: not bitwise equal; {n_differing}/{left.size} values, "
            f"max abs difference {max_abs:.3e}")
    print(f"    {name:48s} bitwise equal ({left.size} values)")


def compare_bounded(name, left, right, output, *, rel_limit, note):
    """Сравнение с ЗАПИСАННЫМ расхождением и относительным пределом.

    Нужно ровно там, где точного равенства ждать НЕЛЬЗЯ по известной
    причине: legacy-путь собирает эмбеддинг как `emb + (hard - emb).detach()`,
    и его значение отличается от строки книги на величину округления. Пока
    feedback1 нулевой, это отличие никуда не распространяется; при
    НЕНУЛЕВОМ feedback1 оно доходит до логитов q2. Поэтому здесь не
    `compare_exact`, но и не молчание: расхождение печатается, пишется в
    артефакт и обязано укладываться в предел.
    """
    left = np.asarray(left).astype(np.float64)
    right = np.asarray(right).astype(np.float64)
    if left.shape != right.shape:
        raise SystemExit(f"{name}: shapes {left.shape} versus {right.shape}")
    # ПРОВЕРКА КОНЕЧНОСТИ ДО ВЫЧИТАНИЯ. `nan > limit` равно False, поэтому
    # без неё нечисловое расхождение объявлялось бы уложившимся в предел, и
    # обязательный гейт оказывался fail-open. Пара inf/inf даёт в разности
    # NaN и проходила так же.
    for side, arr in (("left", left), ("right", right)):
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        if n_bad:
            raise SystemExit(
                f"{name}: в {side} {n_bad}/{arr.size} нечисловых значений "
                f"(nan или inf). Сравнение с пределом на таких данных "
                f"ничего не проверяет")
    max_abs = float(np.max(np.abs(left - right))) if left.size else 0.0
    scale = float(np.max(np.abs(left))) if left.size else 0.0
    rel = max_abs / scale if scale > 0 else max_abs
    n_differing = int(np.count_nonzero(left != right))
    output[name] = {
        "equal": max_abs == 0.0,
        "max_abs_diff": max_abs,
        "scale": scale,
        "rel_diff": rel,
        "n_differing": n_differing,
        "size": int(left.size),
        "rel_limit": float(rel_limit),
        "expected_cause": note,
    }
    if not np.isfinite(rel):
        raise SystemExit(
            f"{name}: относительное расхождение получилось нечисловым "
            f"(max {max_abs}, масштаб {scale})")
    if rel > float(rel_limit):
        raise SystemExit(
            f"{name}: относительное расхождение {rel:.3e} больше предела "
            f"{rel_limit:.3e} ({n_differing}/{left.size} значений, max "
            f"{max_abs:.3e}). Ожидаемая причина: {note}. Такой масштаб "
            f"причиной не объясняется")
    print(f"    {name:48s} rel {rel:.2e} <= {rel_limit:.0e} "
          f"({n_differing}/{left.size})")


def require_initialization_status(status):
    required = (
        "c1_exact",
        "c2_exact",
        "feedback1_weight_zero",
        "feedback1_bias_zero",
        "feedback1_alpha_one",
        "books_not_aliased",
    )
    missing = [key for key in required if key not in status]
    failed = [key for key in required if key in status and not bool(status[key])]
    if missing or failed:
        raise SystemExit(
            f"bad K-15 initialization: missing {missing}, failed {failed}")
    return True


def selftest():
    out = {}
    values = np.arange(24, dtype=np.float32).reshape(3, 8)
    compare_exact("same", values, values.copy(), out)
    assert out["same"]["equal"] and out["same"]["n_differing"] == 0

    changed = values.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.float32(np.inf))
    try:
        compare_exact("ulp", values, changed, out)
    except SystemExit as error:
        assert "not bitwise equal" in str(error), error
    else:
        raise AssertionError("one-ULP difference was accepted")

    for bad, phrase in ((values[:, :-1], "shapes"),
                        (values.astype(np.float64), "dtypes")):
        try:
            compare_exact("bad", values, bad, out)
        except SystemExit as error:
            assert phrase in str(error), (phrase, error)
        else:
            raise AssertionError(f"mismatch accepted: {phrase}")

    good = dict(c1_exact=True, c2_exact=True,
                feedback1_weight_zero=True, feedback1_bias_zero=True,
                feedback1_alpha_one=True, books_not_aliased=True)
    assert require_initialization_status(good)
    for key in good:
        bad = dict(good)
        bad[key] = False
        try:
            require_initialization_status(bad)
        except SystemExit as error:
            assert key in str(error), error
        else:
            raise AssertionError(f"failed initialization field accepted: {key}")
    missing = dict(good)
    del missing["c2_exact"]
    try:
        require_initialization_status(missing)
    except SystemExit as error:
        assert "c2_exact" in str(error), error
    else:
        raise AssertionError("missing initialization field accepted")
    # --- СРАВНЕНИЕ С ПРЕДЕЛОМ --------------------------------------------
    base = np.full((4, 6), 100.0, np.float32)
    near = base.copy()
    near[1, 1] = 100.0 + 1e-4          # относительно 1e-6
    compare_bounded("near", base, near, out, rel_limit=1e-3, note="округление")
    assert out["near"]["equal"] is False
    assert out["near"]["n_differing"] == 1
    assert out["near"]["rel_diff"] < 1e-3 and out["near"]["scale"] == 100.0
    assert out["near"]["expected_cause"] == "округление"
    far = base.copy()
    far[0, 0] = 101.0                  # относительно 1e-2
    try:
        compare_bounded("far", base, far, out, rel_limit=1e-3, note="о")
    except SystemExit as error:
        assert "больше предела" in str(error), error
    else:
        raise AssertionError("расхождение выше предела принято")
    # NaN И Inf ОБЯЗАНЫ БЫТЬ ОТВЕРГНУТЫ, А НЕ УЛОЖИТЬСЯ В ПРЕДЕЛ
    for bad_value, why_ in ((np.nan, "nan"), (np.inf, "inf")):
        spoiled = base.copy()
        spoiled[2, 2] = bad_value
        for pair in ((base, spoiled), (spoiled, base), (spoiled, spoiled)):
            try:
                compare_bounded("nonfinite", pair[0], pair[1], out,
                                rel_limit=1e-3, note="о")
            except SystemExit as error:
                assert "нечисловых" in str(error), (why_, error)
            else:
                raise AssertionError(f"{why_} принят как уложившийся в предел")
    compare_bounded("identical", base, base.copy(), out, rel_limit=0.0,
                    note="точное совпадение")
    assert out["identical"]["equal"] is True
    try:
        compare_bounded("shape", base, base[:, :-1], out, rel_limit=1.0,
                        note="о")
    except SystemExit as error:
        assert "shapes" in str(error), error
    else:
        raise AssertionError("несовпадение форм принято")

    print("k15a_check_init_identity selftest passed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bitwise K-14 -> K-15 initialization identity gate")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--cache", default="data/k11a_joint12")
    parser.add_argument("--q0", default="data/k14d/q0_b8_e0.npz")
    parser.add_argument("--gate-r", default="reports/k14d/gate_r.json")
    parser.add_argument("--joint-ckpt", default="data/k9d_ep3.pt")
    parser.add_argument("--q1-ckpt", default="data/k14c/q1_main_s0.pt")
    parser.add_argument(
        "--ckpt",
        default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    parser.add_argument("--root", default="third_party/actioncodec")
    parser.add_argument("--cfg-path", default="config/eval/bar.yaml")
    parser.add_argument("--variant", default="main")
    parser.add_argument("--n-batches", type=int, default=3)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out", default="reports/k15a/init_identity.json")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return 0
    if args.n_batches <= 0:
        raise SystemExit("--n-batches must be positive")
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} already exists; use --overwrite explicitly")

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(args.root)
    for path in (root, os.path.join(root, "src"), here,
                 os.path.abspath("experiments")):
        if path not in sys.path:
            sys.path.insert(0, path)

    import copy
    import inspect
    import torch
    import actioncodec  # noqa: F401  register model types
    import k11a_build_hicora_cache as k11a
    import k11b_hicora_identity as k11b
    import k14_common as kc
    import k14c_train_q1 as k14c
    import k9h_multiarm_gate as k9h
    from depth_aligned_joint12 import (architecture_code_version,
                                       make_action_decoder,
                                       make_depth_aligned_joint12_class)
    from joint12_vla import make_joint12_class
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (STATE_Q01, STATE_Q99, VisionLanguageActionProcessor,
                       dict_apply, get_cfg, prompt_template)

    git_head, dirty_code, _artifacts = kc.check_code_clean(args.allow_dirty)
    device = torch.device(args.device)
    compute_dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    np.random.seed(0)

    meta = json.load(open(f"{args.cache}.meta.json"))
    source_cache = meta["cache"]
    source = np.load(source_cache, allow_pickle=True)
    source_meta = json.loads(str(source["meta"]))
    n_obs = int(meta["n_obs"])
    episode = np.asarray(source["episode"])[:n_obs].astype(np.int64)
    step = np.asarray(source["step"])[:n_obs].astype(np.int64)
    keys_sha = hashlib.sha1(np.ascontiguousarray(
        np.stack([episode, step])).tobytes()).hexdigest()[:12]
    if keys_sha != str(meta.get("keys_sha1")):
        raise SystemExit(f"observation keys {keys_sha} != {meta.get('keys_sha1')}")
    offsets = np.asarray(source["pos_offset"])[:n_obs].astype(np.int64)
    tasks = np.asarray(source["task"])[:n_obs]

    q0_canonical, _defined, q0_manifest, q0_provenance = kc.load_canonical_q0(
        args.q0,
        gate_r_path=args.gate_r,
        n_obs=n_obs,
        keys_sha=keys_sha,
        cache_meta_sha1=k11a.file_sha1(f"{args.cache}.meta.json"),
    )
    plan = kc.load_plan(args.q0, q0_manifest)[:args.n_batches]
    if len(plan) != args.n_batches:
        raise SystemExit(
            f"canonical plan has only {len(plan)} batches, requested "
            f"{args.n_batches}")
    for part, position_offset, rows in plan:
        if not bool((offsets[rows] == position_offset).all()):
            raise SystemExit(
                f"batch {part}/{position_offset} contains another offset")

    image_path = os.path.join(
        os.path.dirname(source_cache), source_meta["images_file"])
    images = np.load(image_path, mmap_mode="r")
    dataset_repo, dataset_revision = k11b.dataset_source(meta)
    states, _state_meta, _state_sha = kc.load_states(
        source_cache, n_obs, dataset_repo, dataset_revision, keys_sha,
        STATE_Q01, STATE_Q99)
    books_np = np.load(f"{args.cache}.codebooks.npy")

    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt
    processor = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")

    model_cls = make_depth_aligned_joint12_class(
        make_joint12_class(SmolVLABlockwiseAR))
    model = model_cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(
        device, compute_dtype).eval()
    model.init_joint_fast(depth=int(meta["depth"]), head_dtype=torch.float32)
    refine_norm = copy.deepcopy(model.action_expert.norm)
    joint_sha = k11a.file_sha1(args.joint_ckpt)
    kc.load_joint12_strict(
        model, args.joint_ckpt, int(meta["depth"]),
        (meta.get("source") or {}).get("weights_sha1"), torch, device)
    model.init_depth_aligned_rvq(
        refine_norm=refine_norm,
        books=torch.from_numpy(books_np),
        feedback=True,
        head_dtype=torch.float32,
        verbose_init=True,
    )

    # Load the exact selected K-14 q1 state into the shared modules.  The new
    # books are not part of that checkpoint and must remain exact copies.
    legacy_info = model.configure_joint_depth_rvq(
        stage="q1", variant=args.variant, verbose=False)
    q1_checkpoint = torch.load(
        args.q1_ckpt, map_location="cpu", weights_only=False)
    q1_provenance = k9h.check_depthrvq_q1_ckpt(
        q1_checkpoint,
        joint_sha1=joint_sha,
        expect_variant=args.variant,
        expect_q0_manifest_sha1=q0_provenance["q0_manifest_sha1"],
    )
    checkpoint_state = q1_checkpoint["state"]
    wanted = set(legacy_info["names"])
    if set(q1_checkpoint["trainable_names"]) != wanted \
            or set(checkpoint_state) != wanted:
        raise SystemExit(
            "q1 checkpoint whitelist mismatch: missing "
            f"{sorted(wanted - set(checkpoint_state))[:5]}, extra "
            f"{sorted(set(checkpoint_state) - wanted)[:5]}")
    own_parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in checkpoint_state.items():
            if name not in own_parameters:
                raise SystemExit(f"q1 checkpoint key absent in model: {name}")
            if tuple(value.shape) != tuple(own_parameters[name].shape):
                raise SystemExit(
                    f"q1 shape {name}: {tuple(value.shape)} versus "
                    f"{tuple(own_parameters[name].shape)}")
            if not torch.isfinite(value).all():
                raise SystemExit(f"q1 checkpoint has nonfinite values in {name}")
            own_parameters[name].data.copy_(
                value.to(own_parameters[name].device,
                         own_parameters[name].dtype))
    loaded_sha = k14c.state_sha({
        name: own_parameters[name].detach().float().cpu().numpy()
        for name in wanted
    })
    if loaded_sha != str(q1_checkpoint["selected_state_sha1"]):
        raise SystemExit(
            f"loaded q1 state hash {loaded_sha} != "
            f"{q1_checkpoint['selected_state_sha1']}")

    initialization = model.initialization_status()
    require_initialization_status(initialization)
    model.set_depth_aligned_feedback_mask((True, True))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    action_processor = processor.action_processor
    codec = (action_processor if hasattr(action_processor, "vq")
             else getattr(action_processor, "codec", None))
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("could not locate ActionCodec quantizer")
    codec = codec.to(device).eval()
    with torch.no_grad():
        indices = torch.arange(int(codec.vocab_size), device=device)[None, :]
        codec_books = torch.stack([
            quantizer.out_project(quantizer.decode_code(indices))[0]
            for quantizer in codec.vq.quantizers
        ]).float()
    if tuple(codec_books.shape) != tuple(books_np.shape):
        raise SystemExit(
            f"codec books {tuple(codec_books.shape)} versus cache "
            f"{tuple(books_np.shape)}")
    max_book_diff = float((codec_books.cpu()
                           - torch.from_numpy(books_np).float()).abs().max())
    if max_book_diff > 1e-5:
        raise SystemExit(f"codec books differ from cache by {max_book_diff:.3e}")
    codec_fingerprints = {
        "codebooks_sha1": hashlib.sha1(np.ascontiguousarray(
            np.asarray(books_np, np.float32)).tobytes()).hexdigest()[:12],
        "codec_state_sha1": k11a.state_sha1(codec),
        "decoder_probe": k11a.decoder_probe(codec, codec_books, device),
    }
    k11a.check_fingerprints(meta, codec_fingerprints)

    def build_batch(position_offset, rows):
        image = torch.from_numpy(np.asarray(images[rows]))
        messages = []
        for row in rows:
            message = prompt_template(
                states[row], None, str(tasks[row]),
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                action_token_len=cfg.MODEL.action_processor.token_len,
            )
            message[1]["content"] = message[1]["content"][1:]
            messages.append(message)
        texts = processor.apply_chat_template(
            messages, add_generation_prompt=True)
        batch = processor(
            text=texts,
            images=[[image[index].numpy()] for index in range(len(rows))],
            return_tensors="pt",
            padding=True,
            padding_side="left",
            action_processor_kwargs={"embodiment_ids": 0},
        )
        return dict_apply(lambda value: value.to(device, compute_dtype), batch)

    def as_numpy(tensor):
        return tensor.detach().cpu().numpy()

    # ДЕКОДЕР ОБЩИЙ С ТРЕНЕРОМ. Раньше здесь была своя копия, и она
    # работала внутри fp16 autocast, тогда как тренер декодирует в fp32 вне
    # autocast: гейт заверял не то действие, которое считает тренер.
    decode_latent, decoder_context = make_action_decoder(codec, device.type)

    comparisons = {}
    causal = {}
    rows_used = []
    autocast = torch.autocast(device_type=device.type, dtype=compute_dtype)

    for batch_index, (part, position_offset, rows) in enumerate(plan):
        batch = build_batch(position_offset, rows)
        attention_mask = batch.get("attention_mask")
        with torch.no_grad(), autocast:
            vlm_inputs, position_ids = model.build_inputs(
                position_offset=position_offset, **batch)

            model.set_depth_aligned_feedback_mask((True, True))
            old_medium = model.forward_joint_depth_rvq(
                vlm_inputs_embeds=vlm_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                mode="medium",
            )
            old_full = model.forward_joint_depth_rvq(
                vlm_inputs_embeds=vlm_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                mode="full",
            )
            new_medium = model.forward_depth_aligned_rvq(
                vlm_inputs_embeds=vlm_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                mode="medium",
            )
            new_full_on = model.forward_depth_aligned_rvq(
                vlm_inputs_embeds=vlm_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                mode="full",
            )
            model.set_depth_aligned_feedback_mask((True, False))
            new_full_off = model.forward_depth_aligned_rvq(
                vlm_inputs_embeds=vlm_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                mode="full",
            )
            model.set_depth_aligned_feedback_mask((True, True))

            # --- ПРИЧИННОСТЬ, А НЕ ТАВТОЛОГИЯ --------------------------
            # Сравнение выше показывает, что НУЛЕВОЙ feedback1 ничего не
            # меняет. Полностью отсоединённый модуль прошёл бы его так же.
            # Ниже каждый путь проверяется на НЕНУЛЕВЫХ весах: feedback0
            # ненулевой после загрузки чекпойнта K-14, feedback1 получает
            # детерминированный узор. Выключение обязано изменить
            # соответствующий уровень, иначе путь не подключён.
            model.set_depth_aligned_feedback_mask((False, True))
            fb0_off = model.forward_depth_aligned_rvq(
                vlm_inputs_embeds=vlm_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                mode="full",
            )
            model.set_depth_aligned_feedback_mask((True, True))

            probe_weight = model.depth_rvq_feedback[1].proj.weight
            probe_saved = probe_weight.detach().clone()
            probe_pattern = torch.linspace(
                -FEEDBACK_PROBE_SCALE, FEEDBACK_PROBE_SCALE,
                probe_weight.numel(), dtype=probe_weight.dtype,
                device=probe_weight.device).reshape(probe_weight.shape)
            probe_weight.copy_(probe_pattern)
            probe_on = model.forward_depth_aligned_rvq(
                vlm_inputs_embeds=vlm_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                mode="full",
            )
            # СТАРЫЙ ПУТЬ ПРИ ТОМ ЖЕ НЕНУЛЕВОМ feedback1. Сравнение выше
            # проведено при нулевом feedback1, когда обусловливание второго
            # уровня не доходит до логитов вообще: тождественность там
            # выполняется независимо от того, ЧТО подаётся в feedback1.
            # Именно здесь проверяется, что подаётся то же самое.
            old_full_probe = model.forward_joint_depth_rvq(
                vlm_inputs_embeds=vlm_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                mode="full",
            )
            model.set_depth_aligned_feedback_mask((True, False))
            probe_off = model.forward_depth_aligned_rvq(
                vlm_inputs_embeds=vlm_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                mode="full",
            )
            model.set_depth_aligned_feedback_mask((True, True))
            probe_weight.copy_(probe_saved)
            restored = model.forward_depth_aligned_rvq(
                vlm_inputs_embeds=vlm_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                mode="full",
            )

            old_z1 = (
                model.depth_rvq_books[0][old_medium["pred_codes"][0]]
                + model.depth_rvq_books[1][old_medium["pred_codes"][1]])
            old_action = decode_latent(old_z1)
            new_action = decode_latent(
                new_medium["cumulative_latents"][1])

        prefix = f"batch{batch_index}"
        if old_medium["layers_run"] != 18 or new_medium["layers_run"] != 18:
            raise SystemExit(f"{prefix}: medium path did not run 18 layers")
        if old_full["layers_run"] != 24 or new_full_on["layers_run"] != 24:
            raise SystemExit(f"{prefix}: full path did not run 24 layers")

        for level in (0, 1):
            compare_exact(
                f"{prefix}.medium.logits{level}",
                as_numpy(old_medium["logits"][level]),
                as_numpy(new_medium["logits"][level]), comparisons)
            compare_exact(
                f"{prefix}.medium.codes{level}",
                as_numpy(old_medium["pred_codes"][level]),
                as_numpy(new_medium["pred_codes"][level]), comparisons)
            compare_exact(
                f"{prefix}.old_medium_vs_full.logits{level}",
                as_numpy(old_medium["logits"][level]),
                as_numpy(old_full["logits"][level]), comparisons)
            compare_exact(
                f"{prefix}.new_medium_vs_full.logits{level}",
                as_numpy(new_medium["logits"][level]),
                as_numpy(new_full_on["logits"][level]), comparisons)

        for level in (0, 1, 2):
            compare_exact(
                f"{prefix}.full.logits{level}",
                as_numpy(old_full["logits"][level]),
                as_numpy(new_full_on["logits"][level]), comparisons)
            compare_exact(
                f"{prefix}.full.codes{level}",
                as_numpy(old_full["pred_codes"][level]),
                as_numpy(new_full_on["pred_codes"][level]), comparisons)
            compare_exact(
                f"{prefix}.feedback1_on_vs_off.logits{level}",
                as_numpy(new_full_on["logits"][level]),
                as_numpy(new_full_off["logits"][level]), comparisons)
            compare_exact(
                f"{prefix}.feedback1_on_vs_off.codes{level}",
                as_numpy(new_full_on["pred_codes"][level]),
                as_numpy(new_full_off["pred_codes"][level]), comparisons)

        # ВЫКЛЮЧЕНИЕ НЕНУЛЕВОГО feedback0 ОБЯЗАНО ИЗМЕНИТЬ q1.
        if np.array_equal(as_numpy(new_full_on["logits"][1]),
                          as_numpy(fb0_off["logits"][1])):
            raise SystemExit(
                f"{prefix}: выключение НЕНУЛЕВОГО feedback0 не изменило q1 — "
                f"путь черновика в слои 13-18 не подключён")
        causal[f"{prefix}.feedback0_changes_q1"] = True

        # ВЫКЛЮЧЕНИЕ НЕНУЛЕВОГО feedback1 ОБЯЗАНО ИЗМЕНИТЬ q2 И НЕ ИМЕЕТ
        # ПРАВА МЕНЯТЬ УРОВНИ 0 И 1, КОТОРЫЕ СТОЯТ ДО НЕГО.
        if np.array_equal(as_numpy(probe_on["logits"][2]),
                          as_numpy(probe_off["logits"][2])):
            raise SystemExit(
                f"{prefix}: выключение НЕНУЛЕВОГО feedback1 не изменило q2 — "
                f"путь черновика в слои 19-24 не подключён")
        causal[f"{prefix}.feedback1_changes_q2"] = True

        # --- СТАРЫЙ ПРОТИВ НОВОГО ПРИ НЕНУЛЕВОМ feedback1 ----------------
        # Уровни 0 и 1 стоят ДО feedback1 и обязаны совпадать побитово.
        for level in (0, 1):
            compare_exact(
                f"{prefix}.probe_old_vs_new.logits{level}",
                as_numpy(old_full_probe["logits"][level]),
                as_numpy(probe_on["logits"][level]), comparisons)
            compare_exact(
                f"{prefix}.probe_old_vs_new.codes{level}",
                as_numpy(old_full_probe["pred_codes"][level]),
                as_numpy(probe_on["pred_codes"][level]), comparisons)

        # ВЕЛИЧИНА СОБСТВЕННОЙ НЕТОЧНОСТИ LEGACY-ПУТИ, ИЗМЕРЕННАЯ, А НЕ
        # ЗАЯВЛЕННАЯ. Legacy собирает эмбеддинг как `emb + (hard -
        # emb).detach()`; его значение не равно строке книги. Новый путь
        # даёт строку побитово. Это и есть причина расхождения ниже.
        legacy_emb = old_full_probe["embeddings"][1]
        exact_row = model.depth_rvq_books[1][
            old_full_probe["pred_codes"][1]].to(legacy_emb.dtype)
        legacy_gap = (legacy_emb.float() - exact_row.float()).abs()
        legacy_st = {
            "max_abs": float(legacy_gap.max()),
            "n_differing": int((legacy_emb != exact_row).sum()),
            "size": int(legacy_emb.numel()),
            "note": "legacy emb + (hard - emb).detach() против строки книги",
        }
        comparisons[f"{prefix}.legacy_st_error"] = legacy_st
        print(f"    {prefix}.legacy_st_error: {legacy_st['n_differing']}/"
              f"{legacy_st['size']} элементов, max {legacy_st['max_abs']:.2e}")

        # Уровень 2 получает эту неточность через feedback1, поэтому здесь
        # предел, а не равенство. Коды при этом могут переворачиваться
        # только на границе, и их число записывается.
        compare_bounded(
            f"{prefix}.probe_old_vs_new.logits2",
            as_numpy(old_full_probe["logits"][2]),
            as_numpy(probe_on["logits"][2]), comparisons,
            # ПРЕДЕЛ ОСТАЁТСЯ 1e-3, КАКИМ ОБЪЯВЛЕН ДО ДАННЫХ. Я ослаблял
            # его до 1e-2 после того, как он прошёл (замерено 7.62e-04 на
            # батче 1, то есть 76 % предела при НУЛЕ перевёрнутых кодов) —
            # это неоправданно: ослаблять пройденный предел значит терять
            # проверку. Если он когда-нибудь сработает, разбираться тогда,
            # имея рядом долю перевёрнутых кодов как подтверждающую
            # величину.
            rel_limit=1e-3,
            note="неточность legacy-ST в эмбеддинге q1, поданном в "
                 "feedback1, усиленная шестью слоями")
        flips = int((as_numpy(old_full_probe["pred_codes"][2])
                     != as_numpy(probe_on["pred_codes"][2])).sum())
        total_codes = int(as_numpy(probe_on["pred_codes"][2]).size)
        comparisons[f"{prefix}.probe_old_vs_new.code2_flips"] = {
            "flips": flips, "size": total_codes,
            "share": flips / max(total_codes, 1)}
        print(f"    {prefix}.probe_old_vs_new.codes2: перевёрнуто {flips}/"
              f"{total_codes}")
        if flips / max(total_codes, 1) > 0.01:
            raise SystemExit(
                f"{prefix}: при ненулевом feedback1 старый и новый путь "
                f"расходятся по кодам q2 в {flips}/{total_codes} позициях. "
                f"Округление legacy-ST такой доли не объясняет")
        causal[f"{prefix}.probe_old_vs_new_bounded"] = True
        for level in (0, 1):
            compare_exact(
                f"{prefix}.probe.levels_before_feedback1.logits{level}",
                as_numpy(probe_on["logits"][level]),
                as_numpy(probe_off["logits"][level]), comparisons)
        # ПОСЛЕ ЗОНДА СОСТОЯНИЕ ОБЯЗАНО ВЕРНУТЬСЯ ПОБИТОВО: иначе гейт сам
        # оставил бы модель изменённой, и всё, что считается после него,
        # относилось бы к другой модели.
        for level in (0, 1, 2):
            compare_exact(
                f"{prefix}.probe_restored.logits{level}",
                as_numpy(new_full_on["logits"][level]),
                as_numpy(restored["logits"][level]), comparisons)

        for level in (0, 1, 2):
            expected_embedding = model.depth_aligned_book(level)[
                new_full_on["pred_codes"][level]]
            compare_exact(
                f"{prefix}.hard_embedding{level}",
                as_numpy(expected_embedding),
                as_numpy(new_full_on["policy_embeddings"][level]),
                comparisons)
        # ИМЯ ГОВОРИТ, ЧТО ЭТО ЭТАЛОН ИЗ ЖЁСТКИХ СТРОК КНИГИ, а не
        # фактический латент legacy-пути: тот отличается на legacy_st_error.
        compare_exact(
            f"{prefix}.hard_reconstructed_reference_latent",
            as_numpy(old_z1),
            as_numpy(new_medium["cumulative_latents"][1]), comparisons)
        compare_exact(
            f"{prefix}.hard_reconstructed_reference_action",
            as_numpy(old_action), as_numpy(new_action), comparisons)

        q0_now = as_numpy(new_medium["pred_codes"][0])
        q0_expected = np.asarray(q0_canonical[rows], np.int64)
        compare_exact(
            f"{prefix}.q0_canonical", q0_expected, q0_now, comparisons)
        rows_used.append(np.asarray(rows, np.int64))
        print(f"    {prefix}.feedback0_changes_q1                 causal: "
              f"off changes q1")
        print(f"    {prefix}.probe_old_vs_new_bounded             causal: "
              f"старый и новый путь сошлись при НЕНУЛЕВОМ feedback1")
        print(f"    {prefix}.feedback1_changes_q2                 causal: "
              f"probe off changes q2")
        print(f"  batch {batch_index} ({part}, offset {position_offset}) passed")

    rows_used = np.concatenate(rows_used)
    rows_sha = hashlib.sha1(
        np.ascontiguousarray(rows_used).tobytes()).hexdigest()[:12]
    code_files = architecture_code_version(
        here, inspect.getfile(SmolVLABlockwiseAR), sha12)
    # ПРИЧИННЫЕ ПРОВЕРКИ ИДУТ В АРТЕФАКТ. Они останавливают прогон при
    # отказе, но пока их не записать, артефакт не доказывает, что они были:
    # отличить "проверяли и сошлось" от "не проверяли" по нему нельзя.
    expected_causal = {f"batch{i}.{name}"
                       for i in range(len(plan))
                       for name in ("feedback0_changes_q1",
                                    "feedback1_changes_q2",
                                    "probe_old_vs_new_bounded")}
    if set(causal) != expected_causal or not all(causal.values()):
        raise SystemExit(
            f"причинные проверки неполны: нет "
            f"{sorted(expected_causal - set(causal))[:4]}, ложных "
            f"{sorted(k for k, v in causal.items() if not v)[:4]}")
    result = {
        "kind": "k15_init_identity",
        "passed": True,
        "causal_checks": causal,
        "feedback_probe_scale": FEEDBACK_PROBE_SCALE,
        "run_id": f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}",
        "git_head": git_head,
        "git_dirty": bool(dirty_code),
        "code_version": code_files,
        "initialization": initialization,
        "comparisons": comparisons,
        "n_batches": len(plan),
        "decoder_context": decoder_context,
        "n_rows": int(len(rows_used)),
        "rows_sha1": rows_sha,
        "plan_sha1": q0_provenance["plan_sha1"],
        "q0_prov": q0_provenance,
        "joint_ckpt": os.path.abspath(args.joint_ckpt),
        "joint_sha1": joint_sha,
        "q1_ckpt": os.path.abspath(args.q1_ckpt),
        "q1_sha1": k11a.file_sha1(args.q1_ckpt),
        "q1_state_sha1": loaded_sha,
        "q1_provenance": q1_provenance,
        "codec": codec_fingerprints,
        "device": str(device),
        "gpu_uuid": (kc.gpu_uuid(device, torch)
                     if device.type == "cuda" else None),
        "compute_dtype": args.dtype,
        "torch_version": str(torch.__version__),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    temporary = args.out + f".tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1, default=str)
    os.replace(temporary, args.out)
    st_worst = max(
        (v["max_abs"] for k, v in comparisons.items()
         if k.endswith("legacy_st_error")), default=0.0)
    q2_flips = sum(v["flips"] for k, v in comparisons.items()
                   if k.endswith("code2_flips"))
    print(
        f"\nK-15 INITIALIZATION IDENTITY PASSED on {len(plan)} canonical "
        f"batches / {len(rows_used)} rows.\n"
        f"  bitwise equal: q0->q1 logits and codes; latent and action "
        f"rebuilt from hard code rows; q2 with feedback1 zeroed.\n"
        f"  NOT bitwise equal, by a measured amount: the legacy path's own "
        f"level-1 embedding (max {st_worst:.2e}), which is why the "
        f"non-zero-feedback1 comparison of q2 uses a recorded limit "
        f"({q2_flips} q2 codes flipped in total)."
    )
    print(f"  saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
