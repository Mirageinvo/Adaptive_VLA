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


def state_sha(named):
    """Отпечаток набора именованных тензоров, в фиксированном порядке.

    Та же функция, что в K-14c: отпечаток чекпойнта считается ею, и считать
    его здесь иначе значило бы сравнивать разные величины.
    """
    h = hashlib.sha1()
    for k in sorted(named):
        h.update(k.encode())
        h.update(np.ascontiguousarray(
            np.asarray(named[k], dtype=np.float64)).tobytes())
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
    import importlib.util
    sp = importlib.util.spec_from_file_location(
        "_k14c", os.path.join(here, "k14c_train_q1.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    ref = dict(a=np.arange(6, dtype=np.float32).reshape(2, 3),
               b=np.linspace(-1, 1, 4, dtype=np.float32))
    assert state_sha(ref) == m.state_sha(ref), \
        "отпечаток весов считается иначе, чем в K-14c"
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
    ap.add_argument("--dtype-store", default="float16",
                    choices=("float16", "float32"),
                    help="тип хранения h18; float16 допустим только если "
                         "он не переворачивает ни одного кода q1, и это "
                         "проверяется на каждом батче")
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
    # РЕЖИМ ВЫЧИСЛЕНИЙ СВЕРЯЕТСЯ С ТЕМ, В КОТОРОМ ПОСТРОЕН q0. Совпадения q0
    # недостаточно: он argmax и грубее скрытого состояния, ради которого всё
    # и затевается. Одинаковые коды при слегка разных h18 — ровно тот случай,
    # который кэш обязан исключить.
    rt_now = dict(device=str(dev), gpu_uuid=kc.gpu_uuid(dev, torch),
                  compute_dtype=a.dtype, torch_version=str(torch.__version__),
                  cuda_version=str(getattr(torch.version, "cuda", None)),
                  tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                  tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                  cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                  cudnn_benchmark=bool(torch.backends.cudnn.benchmark))
    drift = [f"{k}: сейчас {v}, у q0 {q0_man.get(k)}"
             for k, v in rt_now.items() if str(q0_man.get(k)) != str(v)]
    if drift:
        raise SystemExit("кэш h18 строился бы в другом режиме, чем q0: "
                         + "; ".join(drift))
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
    need_ck = ("stage", "variant", "seed", "state", "trainable_names",
               "selected_state_sha1", "q0_prov", "code_version", "bar_sha1",
               "q1_cache_sha1", "q1_manifest_sha1", "oracle_sha1",
               "joint_sha1", "codebooks_sha1", "codec_state_sha1",
               "decoder_probe", "cache", "ckpt")
    miss_ck = [k for k in need_ck if ck.get(k) is None]
    if miss_ck:
        raise SystemExit(f"в чекпойнте нет полей {miss_ck}")
    if str(ck["stage"]) != "q1":
        raise SystemExit(f"чекпойнт этапа {ck['stage']}")
    if str(ck["variant"]) != str(a.variant):
        raise SystemExit(f"чекпойнт варианта {ck['variant']}, запрошен "
                         f"{a.variant}")
    st = ck["state"]
    want_ = set(info["names"])
    if set(st) != want_ or set(ck["trainable_names"]) != want_:
        raise SystemExit(
            f"белый список чекпойнта не совпадает с этапом: лишние "
            f"{sorted(set(st) - want_)[:5]}, нет {sorted(want_ - set(st))[:5]}")
    own = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in st.items():
            if tuple(own[k].shape) != tuple(v.shape):
                raise SystemExit(f"форма {k}: {tuple(v.shape)} против "
                                 f"{tuple(own[k].shape)}")
            if not torch.isfinite(v).all():
                raise SystemExit(f"в {k} есть nan или inf")
            own[k].data.copy_(v.to(own[k].device, own[k].dtype))
    got_sha = state_sha({k: own[k].detach().float().cpu().numpy()
                              for k in want_})
    if got_sha != ck["selected_state_sha1"]:
        raise SystemExit(f"после загрузки веса имеют отпечаток {got_sha}, в "
                         f"чекпойнте {ck['selected_state_sha1']}")
    bad_p = [k for k in ("q0_manifest_sha1", "q0_npz_sha1", "plan_sha1",
                         "gate_r_sha1", "q0_run_id", "plan_batch")
             if str((ck.get("q0_prov") or {}).get(k)) != str(q0_prov.get(k))]
    if bad_p:
        raise SystemExit(f"чекпойнт обучен на другом черновике: {bad_p}")
    q1_man = json.load(open(a.q1_cache + ".manifest.json"))
    now_ = dict(q1_cache_sha1=q1_man["labels_sha1"],
                q1_manifest_sha1=sha12(a.q1_cache + ".manifest.json"),
                joint_sha1=j_sha,
                codebooks_sha1=arr_sha(np.asarray(E, np.float32)),
                cache=a.cache, ckpt=a.ckpt,
                bar_sha1=sha12(inspect.getfile(SmolVLABlockwiseAR)))
    bad_pr = [f"{k}: чекпойнт {ck[k]}, сейчас {v}"
              for k, v in now_.items() if str(ck[k]) != str(v)]
    if bad_pr:
        raise SystemExit("голова обучена в другой обстановке: "
                         + "; ".join(bad_pr))
    # ДРЕЙФ АРХИТЕКТУРНОГО КОДА. Аргумент был объявлен и не использовался,
    # то есть проверки дрейфа фактически не было вовсе.
    drift_ok = set(x.strip() for x in a.allow_code_drift.split(",")
                   if x.strip())
    arch_ = ("depth_rvq_joint12.py", "depth_rvq_vla.py", "joint12_vla.py",
             "k14_common.py")
    unknown_ = drift_ok - set(arch_)
    if unknown_:
        raise SystemExit(f"--allow-code-drift вне списка: {sorted(unknown_)}")
    cv_ck = ck["code_version"]
    miss_cv = [k for k in arch_ if cv_ck.get(k) is None]
    if miss_cv:
        raise SystemExit(f"в code_version чекпойнта нет {miss_cv}")
    hard_ = {k: (cv_ck[k], code_v.get(k)) for k in arch_
             if str(cv_ck[k]) != str(code_v.get(k)) and k not in drift_ok}
    if hard_:
        raise SystemExit(
            "архитектурный код изменился с момента обучения головы: "
            + "; ".join(f"{k}: чекпойнт {v[0]}, сейчас {v[1]}"
                        for k, v in hard_.items())
            + ". Назовите файл в --allow-code-drift, если изменение не "
              "влияет на вычисление")
    for k in arch_:
        if str(cv_ck[k]) != str(code_v.get(k)):
            print(f"  ДОПУЩЕНО РАСХОЖДЕНИЕ КОДА: {k} — чекпойнт {cv_ck[k]}, "
                  f"сейчас {code_v.get(k)}")
    for p_ in model.parameters():
        p_.requires_grad_(False)
    print(f"  обратная связь взята из {a.head_ckpt} (эпоха "
          f"{ck.get('selected_epoch')}, sha {ck.get('selected_state_sha1')})")

    # --- ХУК: вход нормы уровня q1 ------------------------------------------
    grabbed = {}

    def pre_hook(_m, inp):
        grabbed["h"] = inp[0].detach()

    h_handle = model.depth_rvq_norms[0].register_forward_pre_hook(pre_hook)

    # КЛАСС И eps НОРМЫ ЗАПИСЫВАЮТСЯ В МАНИФЕСТ. Проба обучает норму заново
    # и потому обязана воспроизвести её ровно, не загружая VLM: без eps
    # совпадение чисел было бы случайным.
    nrm0 = model.depth_rvq_norms[0]
    norm_eps = None
    for at in ("eps", "variance_epsilon"):
        if hasattr(nrm0, at):
            norm_eps = float(getattr(nrm0, at))
            break
    if norm_eps is None:
        raise SystemExit(f"у нормы {type(nrm0).__name__} не найдено eps: "
                         f"проба не сможет воспроизвести её точно")
    norm_class = type(nrm0).__name__
    norm_shapes = {k: list(v.shape)
                   for k, v in nrm0.state_dict().items()}
    print(f"  норма уровня q1: {norm_class}, eps {norm_eps}, "
          f"параметры {norm_shapes}")

    d_model = int(model.fast_head.in_features)
    n_pos = int(model.block_size)
    print(f"  d_model {d_model}, позиций {n_pos}; кэш будет "
          f"{len(rows_all) * n_pos * d_model * (2 if a.dtype_store == 'float16' else 4) / 2 ** 30:.1f}"
          f" ГиБ ({a.dtype_store})")

    pos = {int(r): i for i, r in enumerate(rows_all)}
    tmp = a.out + f".h18.npy.tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    need_b = len(rows_all) * n_pos * d_model * (
        2 if a.dtype_store == "float16" else 4)
    free_b = __import__("shutil").disk_usage(
        os.path.dirname(os.path.abspath(a.out)) or ".").free
    if free_b < need_b * 1.15:
        raise SystemExit(f"нужно {need_b / 2 ** 30:.1f} ГиБ, свободно "
                         f"{free_b / 2 ** 30:.1f}")
    store_np = np.float16 if a.dtype_store == "float16" else np.float32
    store_t = torch.float16 if a.dtype_store == "float16" else torch.float32
    H = np.lib.format.open_memmap(tmp, mode="w+", dtype=store_np,
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
    n_lossy = 0
    q1_sum = hashlib.sha1()
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
            if not torch.isfinite(h).all():
                raise SystemExit(f"в h18 батча {k_} есть nan или inf")
            # --- ХРАНЕНИЕ В fp16 ДОКАЗЫВАЕТСЯ, А НЕ ПРЕДПОЛАГАЕТСЯ ---------
            # Хук стоит ПОСЛЕ приведения к типу нормы, то есть h здесь fp32.
            # Прямой проход идёт в fp16, поэтому округление скорее всего
            # ничего не теряет — но «скорее всего» тут недостаточно: если
            # оно меняет хотя бы один код, кэш окажется не тем состоянием,
            # на котором обучалась голова. Проверяется побитово И по кодам.
            h16 = h.to(store_t)
            n_lossy += int((h16.to(h.dtype) != h).sum())
            lg16 = model.depth_rvq_heads[0](
                model.depth_rvq_norms[0](h16.float()))
            q1_live = out["logits"][1].argmax(-1)
            n_flip = int((lg16.argmax(-1) != q1_live).sum())
            if n_flip:
                raise SystemExit(
                    f"хранение в fp16 меняет {n_flip} кодов q1 на батче "
                    f"{k_} части {nm_}: кэш был бы не тем состоянием. "
                    f"Пересоберите с --dtype-store float32")
            q1_sum.update(q1_live.cpu().numpy().astype(np.int32).tobytes())
            ii = np.array([pos[int(r)] for r in sel_])
            H[ii] = h16.cpu().numpy()
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
    print(f"  {a.dtype_store}: неточных значений {n_lossy} из "
          f"{len(rows_all) * n_pos * d_model}, перевёрнутых кодов q1 — 0")
    print(f"  отпечаток предсказанных q1 живого прохода: "
          f"{q1_sum.hexdigest()[:12]}")

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
    man = dict(
        kind="k14_h18_cache", parts=list(parts_want),
        norm_class=norm_class, norm_eps=norm_eps, norm_shapes=norm_shapes,
        n_rows=int(len(rows_all)), n_pos=n_pos, d_model=d_model,
        dtype=a.dtype_store, rows_sha1=arr_sha(rows_all),
        h18_sha1=sha12(tmp), meta_sha1=sha12(tmpm),
        q1_live_sha1=q1_sum.hexdigest()[:12], fp16_lossy_values=int(n_lossy),
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
    # ВСЕ ТРИ ФАЙЛА ПУБЛИКУЮТСЯ ТОЛЬКО ЗДЕСЬ. Прежний порядок выкладывал
    # массив до манифеста: падение между ними оставляло неполную пару, а
    # повторный запуск упирался в проверку существования и требовал ручной
    # уборки после полутора часов счёта.
    for src_, dst_ in ((tmp, a.out + ".h18.npy"), (tmpm, mp),
                       (tmpj, a.out + ".manifest.json")):
        os.replace(src_, dst_)
    if sha12(a.out + ".h18.npy") != man["h18_sha1"] \
            or sha12(mp) != man["meta_sha1"]:
        raise SystemExit("после публикации отпечатки разошлись")
    print(f"\n  сохранено: {a.out}.h18.npy, {mp}, {a.out}.manifest.json")
    print(f"  q0 совпал с каноническим на всех {len(rows_all)} строках")
    print(f"  {(time.time() - t0) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    sys.exit(main())
