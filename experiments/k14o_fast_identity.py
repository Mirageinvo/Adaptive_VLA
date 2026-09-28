#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14o: рука depthrvq в режиме fast обязана давать ТОТ ЖЕ черновик (§53).

ЗАЧЕМ ОТДЕЛЬНО ОТ K-14n. K-14n сравнивает ИСХОДЫ эпизодов, а в k9h весь
векторный батч шагает вместе: env_steps и policy_calls одинаковы у всех
эпизодов прогона, и по эпизодам фактически сравнивается один бит — success.
Две разные политики могут его совпасть. Равенство ДЕЙСТВИЙ так не доказать.

ЧТО ДЕЛАЕТ ЭТОТ СКРИПТ. Сравнивает на ОДНОМ И ТОМ ЖЕ входном батче, до
всякого шага среды:

    forward_joint_fast                      <->  forward_joint_depth_rvq(fast)

в ОДНОЙ модели, где голова q1 и книги уже установлены. Если совпадает
побитово — установка головы и сборка depth-RVQ не трогают черновик, и рука
depthrvq в режиме fast исполняет ровно ту политику, что рука fast.

Дополнительно проверяется, что q0 НЕ ЗАВИСИТ ОТ РЕЖИМА: уровень 0 при
medium и full обязан совпасть с fast. Иначе «q0 + q1» сравнивалось бы с
другим q0, а не с тем, который исполняет быстрый выход.

И ДЕЙСТВИЯ, А НЕ ТОЛЬКО КОДЫ. Коды идут через декодер, и сравнение кодов
формально не покрывает сборку действия. Декодированные действия сверяются
побитово тем же способом, каким их собирает k9h: сумма вкладов уровней,
затем codec._decode.

БЕЗ СИМУЛЯТОРА. Батчи берутся из канонического плана K-14d, поэтому проверка
воспроизводима и не зависит от раскаток.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np


def compare_exact(name, a, b, out):
    """Побитовое равенство двух массивов. Допуска нет и быть не должно."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise SystemExit(f"{name}: формы {a.shape} против {b.shape}")
    if str(a.dtype) != str(b.dtype):
        raise SystemExit(f"{name}: типы {a.dtype} против {b.dtype}")
    eq = bool(np.array_equal(a, b))
    d = (0.0 if eq else float(np.abs(a.astype(np.float64)
                                     - b.astype(np.float64)).max()))
    n_bad = 0 if eq else int((a != b).sum())
    out[name] = dict(equal=eq, max_abs_diff=d, n_differing=n_bad,
                     size=int(a.size))
    if not eq:
        raise SystemExit(
            f"{name}: НЕ совпало побитово — {n_bad} из {a.size} элементов, "
            f"max|Δ| = {d:.3e}. Рука depthrvq исполняла бы другой черновик")
    print(f"    {name:38s} совпало побитово ({a.size} элементов)")
    return True


def selftest():
    out = {}
    a = np.arange(12, dtype=np.int64).reshape(3, 4)
    assert compare_exact("x", a, a.copy(), out) and out["x"]["equal"]
    assert out["x"]["size"] == 12 and out["x"]["n_differing"] == 0
    b = a.copy(); b[1, 1] += 1
    try:
        compare_exact("y", a, b, out)
    except SystemExit as e:
        assert "НЕ совпало" in str(e) and "1 из 12" in str(e), e
    else:
        raise AssertionError("расхождение в одном элементе не замечено")
    for bad, why in (((a, a[:, :3]), "формы"),
                     ((a, a.astype(np.int32)), "типы")):
        try:
            compare_exact("z", *bad, out=out)
        except SystemExit as e:
            assert why in str(e), (why, e)
        else:
            raise AssertionError(f"принято несовпадение: {why}")
    # ДОПУСКА НЕТ: различие в последнем разряде float обязано быть замечено
    f = np.array([1.0, 2.0], np.float32)
    g = f.copy(); g[0] = np.nextafter(g[0], np.float32(np.inf))
    assert not np.array_equal(f, g)
    try:
        compare_exact("w", f, g, out)
    except SystemExit:
        pass
    else:
        raise AssertionError("допуск там, где его быть не должно")
    print("самопроверка k14o_fast_identity пройдена")


def main():
    ap = argparse.ArgumentParser(
        description="Тождество черновика: fast против depthrvq/fast")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--q0", default="data/k14d/q0_b8_e0.npz")
    ap.add_argument("--gate-r", default="reports/k14d/gate_r.json")
    ap.add_argument("--joint-ckpt", default="data/k9d_ep3.pt")
    ap.add_argument("--q1-ckpt", default="data/k14c/q1_main_s0.pt")
    ap.add_argument("--ckpt",
                    default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--variant", default="main")
    ap.add_argument("--n-batches", type=int, default=3)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--allow-code-drift", default="")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", default="reports/k14o/fast_identity.json")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(a.root)
    for p in (root, os.path.join(root, "src"),
              os.path.join(root, "experiments"),
              os.path.abspath("third_party/actioncodec"), here):
        if p not in sys.path:
            sys.path.insert(0, p)
    if a.selftest:
        selftest()
        return 0
    if os.path.exists(a.out) and not a.overwrite:
        raise SystemExit(f"{a.out} уже существует (--overwrite осознанно)")

    import copy
    import torch
    import k14_common as kc
    import k11a_build_hicora_cache as k11a
    import k12b_protocol as kb
    from depth_rvq_joint12 import make_joint_depth_rvq_class
    from joint12_vla import make_joint12_class
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (STATE_Q01, STATE_Q99, VisionLanguageActionProcessor,
                       dict_apply, get_cfg)
    import actioncodec  # noqa: F401  регистрация типа модели в AutoConfig
    import k9h_multiarm_gate as k9h

    head, dirty, _ = kc.check_code_clean(a.allow_dirty)
    print(f"  код: коммит {head}" + ("  (--allow-dirty)" if dirty else ""))
    dev = torch.device(a.device)
    dt = getattr(torch, a.dtype)

    meta = json.load(open(f"{a.cache}.meta.json"))
    src = meta["cache"]
    d = np.load(src, allow_pickle=True)
    N = int(meta["n_obs"])
    epi = np.asarray(d["episode"])[:N].astype(np.int64)
    stp = np.asarray(d["step"])[:N]
    keys_sha = hashlib.sha1(np.ascontiguousarray(
        np.stack([epi, stp])).tobytes()).hexdigest()[:12]
    if keys_sha != meta.get("keys_sha1"):
        raise SystemExit(f"ключи наблюдений {keys_sha} против "
                         f"{meta.get('keys_sha1')}")
    offs = np.asarray(d["pos_offset"])[:N].astype(np.int64)
    tsk = np.asarray(d["task"])[:N]
    q0_can, _q0_def, q0_man, q0_prov = kc.load_canonical_q0(
        a.q0, gate_r_path=a.gate_r, n_obs=N, keys_sha=keys_sha,
        cache_meta_sha1=k11a.file_sha1(f"{a.cache}.meta.json"))
    plan = kc.load_plan(a.q0, q0_man)[:max(int(a.n_batches), 1)]
    for nm_, po_, sel_ in plan:
        if not bool((offs[sel_] == po_).all()):
            raise SystemExit(f"батч части {nm_} заявлен со смещением {po_}")
    print(f"  батчей для сверки: {len(plan)}, план {q0_prov['plan_sha1']}")

    img_p = os.path.join(os.path.dirname(src), json.loads(
        str(d["meta"]))["images_file"])
    IMG = np.load(img_p, mmap_mode="r")
    ds_repo, ds_rev = kb.dataset_source(meta)
    st_n, _sm, _sh = kc.load_states(src, N, ds_repo, ds_rev, keys_sha,
                                   STATE_Q01, STATE_Q99)
    E = np.load(f"{a.cache}.codebooks.npy")

    cfg = get_cfg(os.path.join(root, a.cfg_path))
    cfg.TRAINING.ckpt_dir = a.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = a.ckpt
    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")

    # --- ОДНА МОДЕЛЬ, В КОТОРОЙ ГОЛОВА q1 УЖЕ УСТАНОВЛЕНА -----------------
    # Сравнение двух проходов В ОДНОЙ модели и доказывает нужное: установка
    # головы и книг не трогает черновик. Две отдельные сборки различались бы
    # ещё и расходом аллокатора, а в K-9b именно он давал расхождение логитов
    # при совпадающих весах.
    Cls = make_joint_depth_rvq_class(make_joint12_class(SmolVLABlockwiseAR))
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    model.init_joint_fast(depth=int(meta["depth"]), head_dtype=torch.float32)
    rn = copy.deepcopy(model.action_expert.norm)
    j_sha = k11a.file_sha1(a.joint_ckpt)
    kc.load_joint12_strict(model, a.joint_ckpt, int(meta["depth"]),
                           (meta.get("source") or {}).get("weights_sha1"),
                           torch, dev)
    model.init_joint_depth_rvq(refine_norm=rn, books=torch.from_numpy(E),
                               feedback=(a.variant != "no_feedback"),
                               head_dtype=torch.float32, verbose_init=False)
    info = model.configure_joint_depth_rvq(stage="q1", variant=a.variant,
                                           verbose=False)
    q1_obj = torch.load(a.q1_ckpt, map_location="cpu", weights_only=False)
    prov_q1 = k9h.check_depthrvq_q1_ckpt(q1_obj, joint_sha1=j_sha,
                                         expect_variant=a.variant)
    st = q1_obj["state"]
    want = set(info["names"])
    if set(q1_obj["trainable_names"]) != want or set(st) != want:
        raise SystemExit(
            f"белый список головы не совпал: нет {sorted(want - set(st))[:5]}, "
            f"лишние {sorted(set(st) - want)[:5]}")
    own = dict(model.named_parameters())
    with torch.no_grad():
        for k_, v_ in st.items():
            if tuple(own[k_].shape) != tuple(v_.shape):
                raise SystemExit(f"форма {k_}")
            if not torch.isfinite(v_).all():
                raise SystemExit(f"в {k_} есть nan или inf")
            own[k_].data.copy_(v_.to(own[k_].device, own[k_].dtype))
    h_ = hashlib.sha1()
    for k_ in sorted(want):
        h_.update(k_.encode())
        h_.update(np.ascontiguousarray(
            own[k_].detach().float().cpu().numpy()).tobytes())
    if h_.hexdigest()[:12] != str(q1_obj["selected_state_sha1"]):
        raise SystemExit("после загрузки веса головы имеют другой отпечаток")
    model.eval()
    print(f"  голова q1 установлена и сверена: {len(st)} тензоров, вид "
          f"{prov_q1['q1_kind']}, сид {prov_q1['q1_seed']}")

    Ed = torch.from_numpy(E).float().to(dev)

    def build_batch(po, sel):
        b = proc(images=[np.asarray(IMG[i]) for i in sel],
                 prompts=[str(tsk[i]) for i in sel],
                 states=[st_n[i] for i in sel],
                 return_tensors="pt", padding=True, padding_side="left",
                 action_processor_kwargs={"embodiment_ids": 0})
        return dict_apply(lambda x: x.to(dev, dt), b)

    def act_of(levels):
        """Действие ровно так, как его собирает k9h: сумма вкладов, декодер."""
        with torch.no_grad():
            z = Ed[0][levels[0].long()]
            for j in range(1, len(levels)):
                z = z + Ed[j][levels[j].long()]
            x, _ = codec._decode(z, embodiment_ids=0)
            return x[..., :7].float().cpu().numpy()

    ac_ = proc.action_processor
    codec = ac_ if hasattr(ac_, "vq") else getattr(ac_, "codec", None)
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь")
    codec = codec.to(dev).eval()

    res = {}
    ac16 = torch.autocast(device_type=dev.type, dtype=dt)
    for bi, (nm_, po_, sel_) in enumerate(plan):
        b = build_batch(po_, sel_)
        am = b.get("attention_mask")
        with torch.no_grad(), ac16:
            v, p = model.build_inputs(position_offset=po_, **b)
            o_fast = model.forward_joint_fast(
                vlm_inputs_embeds=v, attention_mask=am, position_ids=p)
            o_drvq = {m_: model.forward_joint_depth_rvq(
                vlm_inputs_embeds=v, attention_mask=am, position_ids=p,
                mode=m_) for m_ in ("fast", "medium", "full")}
        c_fast = o_fast["pred_codes"].cpu().numpy()
        c_drvq = o_drvq["fast"]["pred_codes"][0].cpu().numpy()
        compare_exact(f"b{bi}.q0.fast_vs_depthrvq", c_fast, c_drvq, res)
        # q0 НЕ ЗАВИСИТ ОТ РЕЖИМА
        for m_ in ("medium", "full"):
            compare_exact(f"b{bi}.q0.fast_vs_{m_}", c_fast,
                          o_drvq[m_]["pred_codes"][0].cpu().numpy(), res)
        # ДЕЙСТВИЕ, А НЕ ТОЛЬКО КОДЫ
        compare_exact(
            f"b{bi}.action.fast_vs_depthrvq",
            act_of([o_fast["pred_codes"]]),
            act_of([o_drvq["fast"]["pred_codes"][0]]), res)
        # И СВЕРКА С КАНОНИЧЕСКИМ ЧЕРНОВИКОМ
        bad = int((c_fast != q0_can[sel_]).sum())
        if bad:
            raise SystemExit(f"батч {bi}: q0 расошёлся с каноническим в "
                             f"{bad} позициях")
        if int(o_drvq["medium"]["layers_run"]) != model.depth_rvq_exits[1]:
            raise SystemExit("medium исполнил не то число слоёв")
        print(f"    батч {bi} ({nm_}): q0 сверен с каноническим, medium "
              f"прошёл {o_drvq['medium']['layers_run']} слоёв")

    out = dict(kind="k14o_fast_identity", passed=True,
               run_id=f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}",
               git_head=head, git_dirty=bool(dirty),
               comparisons=res, n_batches=len(plan),
               plan_sha1=q0_prov["plan_sha1"], q0_prov=q0_prov,
               joint_ckpt=a.joint_ckpt, joint_sha1=j_sha,
               q1_ckpt=os.path.abspath(a.q1_ckpt),
               q1_sha1=k11a.file_sha1(a.q1_ckpt), **prov_q1,
               device=str(dev),
               gpu_uuid=(kc.gpu_uuid(dev, torch) if dev.type == "cuda"
                         else None),
               compute_dtype=a.dtype, torch_version=str(torch.__version__),
               depth_rvq_exits=list(model.depth_rvq_exits))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    t_ = a.out + f".tmp.{os.getpid()}"
    json.dump(out, open(t_, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(t_, a.out)
    print(f"\n  ТОЖДЕСТВО ЧЕРНОВИКА ПОДТВЕРЖДЕНО на {len(plan)} батчах: "
          f"depthrvq/fast исполняет ту же политику, что fast")
    print(f"  сохранено: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
