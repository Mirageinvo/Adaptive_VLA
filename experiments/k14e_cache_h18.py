#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14e: кэш состояний h18 для пробы читающих голов (§46).

ЗАЧЕМ. Перебор голов поверх замороженной магистрали стоит семь часов за
вариант только потому, что каждый раз заново считается прямой проход. Сам
проход от варианта головы не зависит: голова читает `action_hidden` после
18-го слоя. Посчитав его один раз, перебор сводят к минутам.

ЧТО ИМЕННО КЭШИРУЕТСЯ И ПОЧЕМУ ИМЕННО ЭТО. Берётся вход нормы уровня q1, то
есть состояние ДО `depth_rvq_norms[0]` и головы. Норма и голова остаются
обучаемыми в пробе; всё, что до них, заморожено и потому кэшируемо.

ЗАВИСИМОСТЬ ОТ ОБРАТНОЙ СВЯЗИ ЯВНАЯ. Проекция `depth_rvq_feedback[0]`
прибавляется к `action_hidden` сразу после выхода на 12-м слое, ДО слоёв
13-18. Значит кэш фиксирует нынешнюю обратную связь, и проба отвечает про
чтение ЭТОГО h18, а не про информацию вообще. Ветка с переобучаемой обратной
связью кэшем не покрывается — это записано в §46.

ПЛАН БАТЧЕЙ КАНОНИЧЕСКИЙ. Промпты дополняются слева до самого длинного в
батче, поэтому состав батча меняет h18 ровно так же, как менял q0. Считать
кэш другой нарезкой значило бы кэшировать состояния, которых модель при
каноническом прогоне не производила.

АРХИТЕКТУРНЫЙ ФАЙЛ НЕ ПРАВИТСЯ. Состояние забирается предварительным хуком на
`depth_rvq_norms[0]`: правка `depth_rvq_joint12.py` изменила бы его отпечаток
и поссорила бы кэш с чекпойнтом, который на нём обучен.

    python experiments/k14e_cache_h18.py --device cuda:1 \
        --parts train,val_sel,val_confirm --out data/k14e/h18
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


def arr_sha(a):
    return hashlib.sha1(
        np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def plan_of(plan, parts):
    """Батчи канонического плана, относящиеся к выбранным частям.

    ПОРЯДОК ИСПОЛНЕНИЯ ЗДЕСЬ БЕЗРАЗЛИЧЕН, а СОСТАВ — нет: h18 зависит от
    того, какие строки оказались в одном батче, и не зависит от того, в
    каком порядке батчи посчитаны.
    """
    out = [(nm, po, rows) for nm, po, rows in plan if nm in parts]
    if not out:
        raise SystemExit(f"в плане нет батчей частей {sorted(parts)}")
    return out


def selftest():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import k14_common as kc
    kc.selftest()

    offs = np.array([3, 3, 4, 4, 4, 3, 4, 3], np.int64)
    parts = dict(train=np.array([0, 1, 2, 5]), val_sel=np.array([3, 4]),
                 val_confirm=np.array([6, 7]))
    plan = kc.make_plan(parts, offs, 2)
    sub = plan_of(plan, {"val_sel"})
    assert all(n == "val_sel" for n, _o, _r in sub)
    assert sorted(int(x) for _n, _o, r in sub for x in r) == [3, 4]
    two = plan_of(plan, {"train", "val_confirm"})
    assert len(two) == len(plan_of(plan, {"train"})) \
        + len(plan_of(plan, {"val_confirm"}))
    try:
        plan_of(plan, {"нет такой"})
    except SystemExit:
        pass
    else:
        raise AssertionError("принята несуществующая часть")

    # ПОРЯДОК СТРОК В КЭШЕ — ГЛОБАЛЬНЫЙ ИНДЕКС, а не порядок батчей: иначе
    # обучающая проба не сможет сопоставить строку с её целью.
    rows = np.concatenate([r for _n, _o, r in plan_of(plan, {"train"})])
    order = np.argsort(rows)
    assert list(rows[order]) == sorted(rows)
    print("самопроверка k14e_cache_h18 пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--q1-cache", default="data/k14b/q1_canonical")
    ap.add_argument("--q0", default="data/k14d/q0_b8_e0.npz")
    ap.add_argument("--gate-r", default="reports/k14d/gate_r.json")
    ap.add_argument("--joint-ckpt", default="data/k9d_ep3.pt")
    ap.add_argument("--head-ckpt", default="data/k14c/q1_main_s0.pt",
                    help="чекпойнт, чью обратную связь кэш фиксирует; она "
                         "входит в h18 и потому обязана быть названа")
    ap.add_argument("--ckpt",
                    default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--variant", default="main")
    ap.add_argument("--parts", default="train,val_sel,val_confirm")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--allow-code-drift", default="")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out", default="data/k14e/h18")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    parts_want = tuple(x.strip() for x in a.parts.split(",") if x.strip())
    for suf in (".h18.npy", ".manifest.json", ".meta.npz"):
        if os.path.exists(a.out + suf):
            raise SystemExit(f"{a.out}{suf} уже существует")

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(a.root)
    for p in (root, os.path.join(root, "src"), here,
              os.path.abspath("experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import inspect
    import copy
    import k11a_build_hicora_cache as k11a
    import k11b_hicora_identity as k11b
    import k12b_protocol as kb
    import k14_common as kc
    from joint12_vla import make_joint12_class
    from depth_rvq_joint12 import make_joint_depth_rvq_class
    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (STATE_Q01, STATE_Q99, VisionLanguageActionProcessor,
                       dict_apply, get_cfg, prompt_template)

    git_head, dirty, arte = kc.check_code_clean(a.allow_dirty)
    if arte:
        print(f"  незакоммиченных результатов рядом: {len(arte)}")
    code_v = kb.code_version([
        os.path.abspath(__file__),
        os.path.join(here, "k14_common.py"),
        os.path.join(here, "depth_rvq_joint12.py"),
        os.path.join(here, "depth_rvq_vla.py"),
        os.path.join(here, "joint12_vla.py")])
    print(f"  код: коммит {git_head}, {len(code_v)} файлов в версии")

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
    if keys_sha != meta.get("keys_sha1"):
        raise SystemExit(f"ключи {keys_sha} против {meta.get('keys_sha1')}")
    ACT = np.asarray(d["action"])[:N]
    offs = np.asarray(d["pos_offset"])[:N].astype(np.int64)
    tsk = np.asarray(d["task"])[:N]

    q0_can, q0_defined, q0_man, q0_prov = kc.load_canonical_q0(
        a.q0, gate_r_path=a.gate_r, n_obs=N, keys_sha=keys_sha,
        cache_meta_sha1=k11a.file_sha1(f"{a.cache}.meta.json"))
    plan_all = kc.load_plan(a.q0, q0_man)
    plan = plan_of(plan_all, set(parts_want))
    rows_all = np.sort(np.concatenate([r for _n, _o, r in plan]))
    if len(np.unique(rows_all)) != len(rows_all):
        raise SystemExit("строки плана повторяются")
    print(f"  план: {len(plan)} батчей, {len(rows_all)} строк, части "
          f"{list(parts_want)}")
    for nm_, po_, sel_ in plan:
        if not bool((offs[sel_] == po_).all()):
            raise SystemExit(f"батч части {nm_} заявлен со смещением {po_}")

    img_p = os.path.join(os.path.dirname(src), cmeta["images_file"])
    IMG = np.load(img_p, mmap_mode="r")
    if IMG.shape[0] < N or IMG.dtype != np.uint8:
        raise SystemExit(f"кадры {IMG.shape} {IMG.dtype}")
    ds_repo, ds_rev = k11b.dataset_source(meta)
    st_n, sm, st_shas = kc.load_states(src, N, ds_repo, ds_rev, keys_sha,
                                       STATE_Q01, STATE_Q99)

    cfg = get_cfg(os.path.join(root, a.cfg_path))
    cfg.TRAINING.ckpt_dir = a.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = a.ckpt
    Base = make_joint_depth_rvq_class(make_joint12_class(SmolVLABlockwiseAR))
    model = Base.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    model.init_joint_fast(depth=int(meta["depth"]), head_dtype=torch.float32)
    refine_norm = copy.deepcopy(model.action_expert.norm)
    j_sha, n_t = kc.load_joint12_strict(
        model, a.joint_ckpt, int(meta["depth"]),
        (meta.get("source") or {}).get("weights_sha1"), torch, dev)
    E = np.load(f"{a.cache}.codebooks.npy")
    model.init_joint_depth_rvq(refine_norm=refine_norm,
                               books=torch.from_numpy(E), feedback=True,
                               head_dtype=torch.float32, verbose_init=False)
    info = model.configure_joint_depth_rvq(stage="q1", variant=a.variant,
                                           verbose=False)

    # --- ОБРАТНАЯ СВЯЗЬ ИЗ ЧЕКПОЙНТА: она входит в h18 -----------------------
    ck = torch.load(a.head_ckpt, map_location="cpu", weights_only=False)
    st = ck["state"]
    if set(st) != set(info["names"]):
        raise SystemExit("белый список чекпойнта не совпадает с этапом")
    own = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in st.items():
            own[k].data.copy_(v.to(own[k].device, own[k].dtype))
    bad_p = [k for k in ("q0_manifest_sha1", "q0_npz_sha1", "plan_sha1",
                         "gate_r_sha1")
             if str((ck.get("q0_prov") or {}).get(k)) != str(q0_prov.get(k))]
    if bad_p:
        raise SystemExit(f"чекпойнт обучен на другом черновике: {bad_p}")
    for p_ in model.parameters():
        p_.requires_grad_(False)
    print(f"  обратная связь взята из {a.head_ckpt} (эпоха "
          f"{ck.get('selected_epoch')}, sha {ck.get('selected_state_sha1')})")

    # --- ХУК: вход нормы уровня q1 ------------------------------------------
    grabbed = {}

    def pre_hook(_m, inp):
        grabbed["h"] = inp[0].detach()

    h_handle = model.depth_rvq_norms[0].register_forward_pre_hook(pre_hook)

    d_model = int(model.fast_head.in_features)
    n_pos = int(model.block_size)
    print(f"  d_model {d_model}, позиций {n_pos}; кэш будет "
          f"{len(rows_all) * n_pos * d_model * 2 / 2 ** 30:.1f} ГиБ")

    pos = {int(r): i for i, r in enumerate(rows_all)}
    tmp = a.out + f".h18.npy.tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    H = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                  shape=(len(rows_all), n_pos, d_model))
    filled = np.zeros(len(rows_all), bool)

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

    ac16 = torch.autocast(device_type=dev.type, dtype=dt)
    t0 = time.time()
    q0_bad = 0
    with torch.no_grad():
        for k_, (nm_, po_, sel_) in enumerate(plan):
            b = build(po_, sel_)
            with ac16:
                v_, p_ = model.build_inputs(position_offset=po_, **b)
                out = model.forward_joint_depth_rvq(
                    vlm_inputs_embeds=v_,
                    attention_mask=b.get("attention_mask"),
                    position_ids=p_, mode="medium")
            # Q0 СВЕРЯЕТСЯ С КАНОНИЧЕСКИМ ПОБИТОВО. Если он разошёлся, то и
            # h18 не тот, что был при обучении головы.
            q0b = out["pred_codes"][0].cpu().numpy().astype(np.int64)
            q0_bad += int((q0b != q0_can[sel_]).sum())
            if q0_bad:
                raise SystemExit(
                    f"q0 разошёлся с каноническим на батче {k_} части {nm_}: "
                    f"{q0_bad} позиций. Кэш h18 был бы от другого состояния")
            h = grabbed.pop("h")
            if tuple(h.shape) != (len(sel_), n_pos, d_model):
                raise SystemExit(f"h имеет форму {tuple(h.shape)}")
            ii = np.array([pos[int(r)] for r in sel_])
            H[ii] = h.to(torch.float16).cpu().numpy()
            filled[ii] = True
            if k_ % 1000 == 0:
                el = (time.time() - t0) / 60
                print(f"    батч {k_}/{len(plan)} ({el:.1f} мин, осталось "
                      f"{el * (len(plan) - k_) / max(k_, 1):.0f})", flush=True)
    h_handle.remove()
    if not filled.all():
        raise SystemExit(f"не заполнено {int((~filled).sum())} строк")
    if not np.isfinite(H[:64].astype(np.float32)).all():
        raise SystemExit("в начале кэша есть nan или inf")
    H.flush()
    del H
    os.replace(tmp, a.out + ".h18.npy")

    part_of = np.empty(len(rows_all), object)
    for nm_, _po, sel_ in plan:
        part_of[[pos[int(r)] for r in sel_]] = nm_
    mp = a.out + ".meta.npz"
    tmpm = mp + f".tmp.{os.getpid()}"
    with open(tmpm, "wb") as fh:
        np.savez_compressed(fh, rows=rows_all.astype(np.int64),
                            part=part_of.astype(str),
                            q0=q0_can[rows_all].astype(np.int32),
                            action=ACT[rows_all].astype(np.float32),
                            pos_offset=offs[rows_all].astype(np.int64),
                            episode=epi[rows_all].astype(np.int64))
    os.replace(tmpm, mp)

    man = dict(
        kind="k14_h18_cache", parts=list(parts_want),
        n_rows=int(len(rows_all)), n_pos=n_pos, d_model=d_model,
        dtype="float16", rows_sha1=arr_sha(rows_all),
        h18_sha1=sha12(a.out + ".h18.npy"), meta_sha1=sha12(mp),
        head_ckpt=a.head_ckpt, head_state_sha1=ck.get("selected_state_sha1"),
        head_epoch=ck.get("selected_epoch"), variant=a.variant,
        feedback_baked_in=True, trainable_names=list(info["names"]),
        cache=a.cache, cache_meta_sha1=k11a.file_sha1(f"{a.cache}.meta.json"),
        codebooks_sha1=arr_sha(np.asarray(E, np.float32)),
        source_cache=src, source_cache_sha1=sha12(src), keys_sha1=keys_sha,
        images_sha1=sha12(img_p), state_npy_sha1=st_shas["state_npy"],
        state_json_sha1=st_shas["state_json"], joint_ckpt=a.joint_ckpt,
        joint_sha1=j_sha, ckpt=a.ckpt, q1_cache=a.q1_cache,
        bar_sha1=sha12(inspect.getfile(SmolVLABlockwiseAR)),
        cfg_sha1=sha12(os.path.join(root, a.cfg_path)),
        device=str(dev), gpu_uuid=kc.gpu_uuid(dev, torch),
        compute_dtype=a.dtype, torch_version=str(torch.__version__),
        cuda_version=str(getattr(torch.version, "cuda", None)),
        tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
        tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        git_head=git_head, git_dirty=bool(dirty), code_version=code_v,
        script_sha1=sha12(os.path.abspath(__file__)),
        minutes=float((time.time() - t0) / 60), **q0_prov)
    tmpj = a.out + ".manifest.json" + f".tmp.{os.getpid()}"
    json.dump(man, open(tmpj, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmpj, a.out + ".manifest.json")
    print(f"\n  сохранено: {a.out}.h18.npy, {mp}, {a.out}.manifest.json")
    print(f"  q0 совпал с каноническим на всех {len(rows_all)} строках")
    print(f"  {(time.time() - t0) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    sys.exit(main())
