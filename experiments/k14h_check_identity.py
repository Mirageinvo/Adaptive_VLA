#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14h: архитектурный гейт — новая модель при нулевой W_o тождественна старой.

ЗАЧЕМ ОТДЕЛЬНЫМ СКРИПТОМ. Это проверяется один раз и до всякого обучения; она
не должна тащить за собой аргументы тренера и не должна быть его частью,
иначе её легко пропустить, передав не тот флаг.

ЧТО ДОКАЗЫВАЕТСЯ. Блок re-attention добавлен так, что ДО обучения он не
меняет ничего: при обнулённой выходной проекции все три режима обязаны
совпасть со старой моделью ПОБИТОВО. Приблизительного совпадения
недостаточно: сравнение архитектур началось бы с неучтённого сдвига, и любой
результат обучения было бы нечем отделить от него.

ПОЧЕМУ ЭТО НЕ ФОРМАЛЬНОСТЬ. Проход в новом классе переписан целиком — блоку
нужны vlm_hidden, маска и состояние действий в один и тот же момент, а
базовый проход их наружу не отдаёт. Расхождение переписанного цикла с
оригиналом ловится ровно здесь.

    python experiments/k14h_check_identity.py --device cuda:1
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np


def sha12(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:12]


def compare(name, a, b, out):
    """Побитовое сравнение с записью результата. Допуска нет намеренно."""
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise SystemExit(f"{name}: длины {len(a)} и {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(f"{name}[{i}]", x, y, out)
        return
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        raise SystemExit(f"{name}: {tuple(a.shape)} {a.dtype} против "
                         f"{tuple(b.shape)} {b.dtype}")
    import torch
    same = bool(torch.equal(a, b))
    d = float((a.float() - b.float()).abs().max()) if not same else 0.0
    out[name] = dict(equal=same, max_abs_diff=d)
    if not same:
        raise SystemExit(
            f"{name}: НЕ совпало побитово, максимум расхождения {d:.3e}. "
            f"При обнулённой W_o новая модель обязана быть тождественна "
            f"старой — иначе сравнение архитектур начнётся со сдвига")
    print(f"    {name:34s} совпало побитово")


def selftest():
    import torch
    out = {}
    a = torch.randn(3, 4)
    compare("x", a, a.clone(), out)
    assert out["x"]["equal"]
    compare("l", [a, a], [a.clone(), a.clone()], out)
    b = a.clone(); b[0, 0] += 1e-7
    try:
        compare("y", a, b, out)
    except SystemExit as e:
        assert "НЕ совпало" in str(e), e
    else:
        raise AssertionError("допуск там, где его быть не должно")
    try:
        compare("z", a, a.double(), out)
    except SystemExit as e:
        assert "против" in str(e), e
    else:
        raise AssertionError("разные типы приняты")
    try:
        compare("w", [a], [a, a], out)
    except SystemExit:
        pass
    else:
        raise AssertionError("разные длины приняты")
    print("самопроверка k14h_check_identity пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--q0", default="data/k14d/q0_b8_e0.npz")
    ap.add_argument("--gate-r", default="reports/k14d/gate_r.json")
    ap.add_argument("--joint-ckpt", default="data/k9d_ep3.pt")
    ap.add_argument("--head-ckpt", default="data/k14c/q1_main_s0.pt")
    ap.add_argument("--ckpt",
                    default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--variant", default="main")
    ap.add_argument("--n-batches", type=int, default=3)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--allow-code-drift", default="")
    ap.add_argument("--out", default="reports/k14h/identity.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} уже существует")

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(a.root)
    for p in (root, os.path.join(root, "src"), here,
              os.path.abspath("experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import copy
    import k11a_build_hicora_cache as k11a
    import inspect
    import k11b_hicora_identity as k11b
    import k12b_protocol as kb
    import k14_common as kc
    from joint12_vla import make_joint12_class
    from depth_rvq_joint12 import make_joint_depth_rvq_class
    import k14h_reattn as kh
    from k14h_reattn import make_draft_reattn_class
    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (STATE_Q01, STATE_Q99, VisionLanguageActionProcessor,
                       dict_apply, get_cfg, prompt_template)

    git_head, dirty, _ = kc.check_code_clean(True)
    dev, dt = torch.device(a.device), getattr(torch, a.dtype)
    torch.manual_seed(0)
    np.random.seed(0)

    meta = json.load(open(f"{a.cache}.meta.json"))
    src = meta["cache"]
    d = np.load(src, allow_pickle=True)
    cmeta = json.loads(str(d["meta"]))
    N = int(meta["n_obs"])
    epi, stp = np.asarray(d["episode"])[:N], np.asarray(d["step"])[:N]
    keys_sha = hashlib.sha1(np.ascontiguousarray(
        np.stack([epi, stp])).tobytes()).hexdigest()[:12]
    offs = np.asarray(d["pos_offset"])[:N].astype(np.int64)
    tsk = np.asarray(d["task"])[:N]
    q0_for_check = None            # заполняется ниже, после загрузки плана
    q0_can, _defined, q0_man, q0_prov = kc.load_canonical_q0(
        a.q0, gate_r_path=a.gate_r, n_obs=N, keys_sha=keys_sha,
        cache_meta_sha1=k11a.file_sha1(f"{a.cache}.meta.json"))
    plan = kc.load_plan(a.q0, q0_man)[:max(int(a.n_batches), 1)]
    for nm_, po_, sel_ in plan:
        if not bool((offs[sel_] == po_).all()):
            raise SystemExit(f"батч части {nm_} заявлен со смещением {po_}, а "
                             f"строки имеют другое")
    q0_for_check = q0_can
    print(f"  батчей для сверки: {len(plan)}")

    img_p = os.path.join(os.path.dirname(src), cmeta["images_file"])
    IMG = np.load(img_p, mmap_mode="r")
    ds_repo, ds_rev = k11b.dataset_source(meta)
    st_n, _sm, _sh = kc.load_states(src, N, ds_repo, ds_rev, keys_sha,
                                    STATE_Q01, STATE_Q99)
    E = np.load(f"{a.cache}.codebooks.npy")

    cfg = get_cfg(os.path.join(root, a.cfg_path))
    cfg.TRAINING.ckpt_dir = a.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = a.ckpt
    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")

    def build_model(with_reattn):
        base = make_joint_depth_rvq_class(make_joint12_class(
            SmolVLABlockwiseAR))
        cls = make_draft_reattn_class(base) if with_reattn else base
        m = cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
        m.init_joint_fast(depth=int(meta["depth"]), head_dtype=torch.float32)
        rn = copy.deepcopy(m.action_expert.norm)
        kc.load_joint12_strict(m, a.joint_ckpt, int(meta["depth"]),
                               (meta.get("source") or {}).get("weights_sha1"),
                               torch, dev)
        m.init_joint_depth_rvq(refine_norm=rn, books=torch.from_numpy(E),
                               feedback=True, head_dtype=torch.float32,
                               verbose_init=False)
        ck = torch.load(a.head_ckpt, map_location="cpu", weights_only=False)
        if with_reattn:
            m.init_draft_reattn(n_heads=a.heads, use_draft=True,
                                head_dtype=torch.float32, verbose=True)
            info = m.configure_draft_reattn(stage="q1", variant=a.variant,
                                            verbose=False)
        else:
            info = m.configure_joint_depth_rvq(stage="q1", variant=a.variant,
                                               verbose=False)
        own = dict(m.named_parameters())
        st = ck["state"]
        # СТАРЫЙ state dict ДОПУСКАЕТ ОТСУТСТВИЕ ТОЛЬКО КЛЮЧЕЙ БЛОКА.
        extra = [k for k in st if k not in own]
        if extra:
            raise SystemExit(f"в модели нет ключей чекпойнта {extra[:5]}")
        missing = [k for k in info["names"] if k not in st]
        if with_reattn:
            bad = [k for k in missing if not k.startswith("draft_reattn.")]
            if bad:
                raise SystemExit(f"в чекпойнте нет весов {bad[:5]}")
        elif missing:
            raise SystemExit(f"в чекпойнте нет весов {missing[:5]}")
        with torch.no_grad():
            for k, v in st.items():
                own[k].data.copy_(v.to(own[k].device, own[k].dtype))
        for p_ in m.parameters():
            p_.requires_grad_(False)
        return m, info

    # ДВЕ МОДЕЛИ НА ВСЕ ШЕСТЬ СЛУЧАЕВ. Вариант и архитектура переключаются
    # на месте: блок обслуживает оба режима запроса одним набором весов, а
    # `draft_reattn_enabled` выключает его целиком. Пересобирать модель под
    # каждый случай значило бы сравнивать разные сборки, а не один блок в
    # разных режимах.
    m_old, _ = build_model(False)
    m_new, _ = build_model(True)

    blk = m_new.draft_reattn[0]
    if float(blk.w_o.weight.abs().max()) != 0.0:
        raise SystemExit("W_o не нулевая")
    if blk.w_o.bias is not None and float(blk.w_o.bias.abs().max()) != 0.0:
        raise SystemExit("смещение W_o не нулевое")
    if float(blk.alpha.item()) != 1.0:
        raise SystemExit(f"alpha = {float(blk.alpha.item())}, ожидалась 1")
    print("  W_o и её смещение нулевые, alpha = 1")

    def build_batch(po, sel):
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

    ac = torch.autocast(device_type=dev.type, dtype=dt)
    batches = [(nm_, po_, sel_, build_batch(po_, sel_))
               for nm_, po_, sel_ in plan]
    rows_used = np.concatenate([s_ for _n, _p, s_, _b in batches])
    rows_sha = hashlib.sha1(np.ascontiguousarray(
        rows_used.astype(np.int64)).tobytes()).hexdigest()[:12]

    res = {}
    cases, names_by_case = {}, {}
    for arch in ("baseline", "reattn_draft", "reattn_state"):
        for fb in ("on", "off"):
            case = kh.gate_case(arch, fb)
            var_new = "main" if fb == "on" else "no_additive_feedback"
            var_old = "main" if fb == "on" else "no_feedback"
            info_old = m_old.configure_joint_depth_rvq(
                stage="q1", variant=var_old, verbose=False)
            info_new = m_new.configure_draft_reattn(
                stage="q1", variant=var_new, verbose=False)
            m_new.draft_reattn_enabled = (arch != "baseline")
            blk.set_use_draft(arch == "reattn_draft")
            print(f"\n  === случай {case} ===")
            modes, calls, q0_ok = {}, {}, True
            for mode in ("fast", "medium", "full"):
                for bi, (nm_, po_, sel_, b) in enumerate(batches):
                    am = b.get("attention_mask")
                    outs = {}
                    for mdl, tag in ((m_old, "старая"), (m_new, "новая")):
                        with ac:
                            v_, p_ = mdl.build_inputs(position_offset=po_, **b)
                            if am is not None and \
                                    int(am.shape[1]) != int(v_.shape[1]):
                                raise SystemExit(
                                    f"маска длины {int(am.shape[1])} при "
                                    f"префиксе {int(v_.shape[1])}")
                            if am is not None and int(am.sum(1).min()) < 1:
                                raise SystemExit("есть пример без открытых "
                                                 "ключей префикса")
                            outs[tag] = mdl.forward_joint_depth_rvq(
                                vlm_inputs_embeds=v_, attention_mask=am,
                                position_ids=p_, mode=mode)
                    got = outs["новая"].get("reattn_calls")
                    want = 0 if (mode == "fast" or arch == "baseline") else 1
                    if got != want:
                        raise SystemExit(f"{case}/{mode}: блок вызван {got} "
                                         f"раз, ожидалось {want}")
                    calls[mode] = int(got)
                    pref = f"{case}.b{bi}.{mode}"
                    compare(f"{pref}.logits", outs["старая"]["logits"],
                            outs["новая"]["logits"], res)
                    compare(f"{pref}.pred_codes", outs["старая"]["pred_codes"],
                            outs["новая"]["pred_codes"], res)
                    for tag in ("старая", "новая"):
                        got_q0 = outs[tag]["pred_codes"][0].cpu().numpy()
                        bad = int((got_q0 != q0_for_check[sel_]).sum())
                        if bad:
                            q0_ok = False
                            raise SystemExit(
                                f"{case}, {tag} модель, {mode}: q0 разошёлся "
                                f"с каноническим в {bad} позициях")
                modes[mode] = dict(batches=len(batches))
            new_only = sorted(set(info_new["names"]) - set(info_old["names"]))
            if arch != "baseline" and not new_only:
                raise SystemExit(f"{case}: блок не попал в обучаемые")
            cases[case] = dict(
                passed=True, modes=modes, reattn_calls=calls,
                q0_matches_canonical=bool(q0_ok),
                trainable_old=info_old["names"],
                trainable_new=info_new["names"], added=new_only,
                n_params_old=int(info_old["n_params"]),
                n_params_new=int(info_new["n_params"]))
            names_by_case[case] = new_only
            print(f"    пройден; обучаемых {info_old['n_tensors']} -> "
                  f"{info_new['n_tensors']}, добавлено {len(new_only)}")

    out = dict(
        kind="k14h_identity_gate", passed=True,
        run_id=f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}",
        cases=cases, comparisons=res, n_batches=len(batches),
        rows_sha1=rows_sha, heads=int(a.heads), variant=a.variant,
        joint_ckpt=a.joint_ckpt, joint_sha1=k11a.file_sha1(a.joint_ckpt),
        ckpt=a.ckpt, head_ckpt=a.head_ckpt,
        q0_npz=q0_prov["q0_npz"], q0_npz_sha1=q0_prov["q0_npz_sha1"],
        q0_manifest_sha1=q0_prov["q0_manifest_sha1"],
        plan_sha1=q0_prov["plan_sha1"], plan_batch=q0_prov["plan_batch"],
        gate_r_sha1=q0_prov["gate_r_sha1"],
        device=str(dev), gpu_uuid=kc.gpu_uuid(dev, torch),
        compute_dtype=a.dtype, torch_version=str(torch.__version__),
        cuda_version=str(getattr(torch.version, "cuda", None)),
        tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
        tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        reattn_sha1=sha12(os.path.join(here, "k14h_reattn.py")),
        depth_rvq_joint12_sha1=sha12(os.path.join(here,
                                                  "depth_rvq_joint12.py")),
        depth_rvq_vla_sha1=sha12(os.path.join(here, "depth_rvq_vla.py")),
        joint12_vla_sha1=sha12(os.path.join(here, "joint12_vla.py")),
        bar_sha1=sha12(inspect.getfile(SmolVLABlockwiseAR)),
        git_head=git_head, git_dirty=bool(dirty),
        code_version=kb.code_version([
            os.path.abspath(__file__),
            os.path.join(here, "k14h_reattn.py"),
            os.path.join(here, "depth_rvq_joint12.py"),
            os.path.join(here, "k14_common.py")]),
        script_sha1=sha12(os.path.abspath(__file__)))
    if dirty:
        raise SystemExit("гейт снят при незакоммиченном коде: он быстрый, "
                         "переснимите после коммита")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = a.out + f".tmp.{os.getpid()}"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, a.out)
    print(f"\n  ГЕЙТ ПРОЙДЕН: {len(cases)} случаев x 3 режима x "
          f"{len(batches)} батчей, всё побитово")
    print(f"  строки сверки: {rows_sha}, запуск {out['run_id']}")
    print(f"  сохранено: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
