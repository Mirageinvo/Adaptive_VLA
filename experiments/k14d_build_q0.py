#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14d: план батчей, построение q0 по нему и Gate R — проверка повторяемости.

ЗАЧЕМ. Черновик q0 из кэша K-11a не воспроизводится сегодня побитово: три
пути вычисления (сегментированный проход, forward_joint_fast, forward_taps)
совпадают МЕЖДУ СОБОЙ и одинаково расходятся с кэшем. Значит различается не
метод, а состав батчей: K-11a гонял 150 тысяч строк своими группами.

ЧТО ЭТО НЕ ДОКАЗЫВАЕТ. Что q0 невоспроизводим в принципе. Зависимость от
состава батча не означает зависимости от ЗАПУСКА при одном и том же составе.
Это разные утверждения, и второе здесь измеряется, а не предполагается.

GATE R. План батчей фиксируется и сохраняется: часть, смещение позиций,
упорядоченный список строк каждого батча, размер батча, отпечаток плана.
Затем q0 строится этим планом дважды, в разных процессах, с разным ПОРЯДКОМ
ИСПОЛНЕНИЯ батчей, но неизменным их СОСТАВОМ. Совпадение отпечатков q0 по
каждой части означает, что план достаточен для воспроизводимости.

    прошёл  -> D': q0 становится артефактом K-14, цели q1* строятся от него,
               тренер идёт тем же планом, строгая побитовая сверка без допуска
    не прошёл -> C': неизменяемая маска расхождений, зарегистрированная до
               обучения

ПОЧЕМУ ПОРЯДОК ИСПОЛНЕНИЯ, А НЕ СОСТАВ. Сид тренера переставляет батчи; если
от этого меняется q0, то два прогона с разными сидами решают разные задачи, и
разность между ними нельзя приписать порядку данных.
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


def make_plan(parts, offs, batch):
    """План батчей: [(часть, смещение, строки)]. ДЕТЕРМИНИРОВАН.

    Состав батчей задаётся здесь раз и навсегда: строки сортируются, группы
    идут по возрастанию смещения, нарезка последовательная. Ни сид, ни порядок
    словаря на состав не влияют — иначе «тот же план» означал бы разное.
    """
    plan = []
    for name in sorted(parts):
        rows = np.sort(np.asarray(parts[name], np.int64))
        for po in sorted(set(int(x) for x in offs[rows])):
            sel = rows[offs[rows] == po]
            for i in range(0, len(sel), int(batch)):
                plan.append((str(name), int(po),
                             np.asarray(sel[i:i + int(batch)], np.int64)))
    return plan


def plan_sha(plan):
    """Отпечаток плана: части, смещения и сами строки в порядке плана."""
    h = hashlib.sha1()
    for name, po, rows in plan:
        h.update(name.encode())
        h.update(str(int(po)).encode())
        h.update(np.ascontiguousarray(rows).tobytes())
    return h.hexdigest()[:12]


def plan_stats(plan):
    out = {}
    for name, _po, rows in plan:
        d = out.setdefault(name, dict(batches=0, rows=0))
        d["batches"] += 1
        d["rows"] += int(len(rows))
    return out


def selftest():
    offs = np.array([3, 3, 4, 4, 4, 3, 4, 3], np.int64)
    parts = dict(train=np.array([0, 1, 2, 5]), val_sel=np.array([3, 4, 6, 7]))
    p1 = make_plan(parts, offs, 2)
    # СОСТАВ НЕ ЗАВИСИТ ОТ ПОРЯДКА ВХОДА
    parts_rev = dict(val_sel=np.array([7, 6, 4, 3]),
                     train=np.array([5, 2, 1, 0]))
    p2 = make_plan(parts_rev, offs, 2)
    assert plan_sha(p1) == plan_sha(p2), "план зависит от порядка входа"
    # СМЕЩЕНИЯ НЕ СМЕШИВАЮТСЯ
    for _n, po, rows in p1:
        assert len(set(int(x) for x in offs[rows])) == 1
        assert int(offs[rows][0]) == po
    st = plan_stats(p1)
    assert st["train"]["rows"] == 4 and st["val_sel"]["rows"] == 4
    assert sum(len(r) for _a, _b, r in p1) == 8
    # ДРУГОЙ РАЗМЕР БАТЧА — ДРУГОЙ ПЛАН
    assert plan_sha(make_plan(parts, offs, 4)) != plan_sha(p1)
    # ПЕРЕСТАНОВКА ПОРЯДКА ИСПОЛНЕНИЯ НЕ МЕНЯЕТ ОТПЕЧАТОК СОСТАВА
    idx = np.random.default_rng(0).permutation(len(p1))
    assert plan_sha([p1[i] for i in idx]) != plan_sha(p1), \
        "отпечаток обязан зависеть от порядка: он описывает план целиком"
    assert arr_sha(np.zeros(3, np.int64)) == arr_sha(np.zeros(3, np.int64))
    print("самопроверка k14d_build_q0 пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--q1-cache", default="data/k14b/q1_cache",
                    help="берутся только СОСТАВЫ ЧАСТЕЙ; цели не читаются")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--joint-ckpt", default="data/k9d_ep3.pt")
    ap.add_argument("--ckpt",
                    default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--exec-order-seed", type=int, default=0,
                    help="переставляет ПОРЯДОК исполнения батчей, не состав")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    out_p = a.out or f"data/k14d/q0_plan_b{a.batch}_e{a.exec_order_seed}"
    for suf in (".npz", ".manifest.json"):
        if os.path.exists(out_p + suf):
            raise SystemExit(f"{out_p}{suf} уже существует")

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(a.root)
    for p in (root, os.path.join(root, "src"), here,
              os.path.abspath("experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import inspect
    import k11a_build_hicora_cache as k11a
    import k12b_protocol as kb
    from joint12_vla import make_joint12_class
    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (STATE_Q01, STATE_Q99, VisionLanguageActionProcessor,
                       dict_apply, get_cfg, prompt_template)

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
    offs = np.asarray(d["pos_offset"])[:N].astype(np.int64)
    tsk = np.asarray(d["task"])[:N]
    q0hat_c = np.load(f"{a.cache}.q0hat.npy", mmap_mode="r")

    img_p = os.path.join(os.path.dirname(src), cmeta["images_file"])
    IMG = np.load(img_p, mmap_mode="r")
    st_p, stm_p = src + ".state.npy", src + ".state.json"
    sm = json.load(open(stm_p))
    if sm.get("keys_sha1") != keys_sha:
        raise SystemExit("состояния от других наблюдений")
    st_n = ((np.load(st_p)[:N] - STATE_Q01) / (STATE_Q99 - STATE_Q01)
            * 2.0 - 1.0)

    with np.load(a.q1_cache + ".npz", allow_pickle=True) as z:
        rows_all = np.asarray(z["rows"], np.int64)
        part_all = np.asarray(z["part"]).astype(str)
    parts = {nm: rows_all[part_all == nm]
             for nm in ("train", "val_sel", "val_confirm")}
    if a.limit:
        parts = {k: v[:a.limit] for k, v in parts.items()}
    plan = make_plan(parts, offs, a.batch)
    p_sha = plan_sha(plan)
    print(f"  план батчей: {len(plan)} батчей, размер {a.batch}, sha {p_sha}")
    for nm, st in sorted(plan_stats(plan).items()):
        print(f"    {nm}: {st['batches']} батчей, {st['rows']} строк")

    cfg = get_cfg(os.path.join(root, a.cfg_path))
    cfg.TRAINING.ckpt_dir = a.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = a.ckpt
    Base = make_joint12_class(SmolVLABlockwiseAR)
    model = Base.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    model.init_joint_fast(depth=int(meta["depth"]), head_dtype=torch.float32)
    j_sha = k11a.file_sha1(a.joint_ckpt)
    if j_sha != (meta.get("source") or {}).get("weights_sha1"):
        raise SystemExit(f"Joint12 {j_sha} не тот, которым собран кэш")
    obj = torch.load(a.joint_ckpt, map_location="cpu", weights_only=False)
    own = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in obj["state"].items():
            own[k].data = v.to(dev, torch.float32)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    print(f"  Joint12 загружен: {len(obj['state'])} тензоров, sha {j_sha}")

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

    # ПОРЯДОК ИСПОЛНЕНИЯ ПЕРЕСТАВЛЯЕТСЯ, СОСТАВ — НЕТ.
    order = np.arange(len(plan))
    if a.exec_order_seed:
        order = np.random.default_rng(a.exec_order_seed).permutation(order)
    q0_out = np.full((N, 16), -1, np.int32)
    ac16 = torch.autocast(device_type=dev.type, dtype=dt)
    with torch.no_grad():
        for k_, i in enumerate(order):
            name, po, sel = plan[int(i)]
            b = build(po, sel)
            with ac16:
                v_, p_ = model.build_inputs(position_offset=po, **b)
                out = model.forward_joint_fast(
                    vlm_inputs_embeds=v_,
                    attention_mask=b.get("attention_mask"), position_ids=p_)
            q0 = out["pred_codes"] if isinstance(out, dict) else out[1]
            q0_out[sel] = q0.cpu().numpy().astype(np.int32)
            if k_ % 200 == 0:
                print(f"    батч {k_}/{len(plan)}", flush=True)

    res, diff = {}, {}
    for nm, rows in sorted(parts.items()):
        rr = np.sort(np.asarray(rows, np.int64))
        q = q0_out[rr]
        if (q < 0).any():
            raise SystemExit(f"{nm}: не все строки посчитаны")
        res[nm] = dict(n_rows=int(len(rr)), rows_sha1=arr_sha(rr),
                       q0_sha1=arr_sha(q), dtype=str(q.dtype))
        cq = np.asarray(q0hat_c[rr]).astype(np.int32)
        nb = int((q != cq).sum())
        diff[nm] = dict(mismatch=nb, positions=int(q.size),
                        frac=float(nb) / max(q.size, 1))
        print(f"    {nm}: q0 sha {res[nm]['q0_sha1']}, расхождение с кэшем "
              f"K-11a {nb} из {q.size} ({100 * nb / max(q.size, 1):.4f}%)")

    os.makedirs(os.path.dirname(os.path.abspath(out_p)) or ".", exist_ok=True)
    npz = out_p + ".npz"
    tmp = npz + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, q0=q0_out,
                            plan_rows=np.concatenate([r for _a, _b, r in plan]),
                            plan_part=np.array([n for n, _b, r in plan
                                                for _ in range(len(r))]),
                            plan_offset=np.array([o for _n, o, r in plan
                                                  for _ in range(len(r))],
                                                 np.int64))
    os.replace(tmp, npz)
    man = dict(kind="k14_q0_by_plan", batch=int(a.batch),
               exec_order_seed=int(a.exec_order_seed), plan_sha1=p_sha,
               n_batches=len(plan), plan_stats=plan_stats(plan),
               parts=res, diff_vs_k11a=diff, device=str(dev), dtype=a.dtype,
               cache=a.cache, source_cache=src, source_cache_sha1=sha12(src),
               keys_sha1=keys_sha, ckpt=a.ckpt, joint_sha1=j_sha,
               q1_cache=a.q1_cache, limit=int(a.limit),
               npz_sha1=sha12(npz),
               bar_sha1=sha12(inspect.getfile(SmolVLABlockwiseAR)),
               torch_version=str(torch.__version__),
               cuda_version=str(getattr(torch.version, "cuda", None)),
               gpu=(torch.cuda.get_device_name(dev)
                    if dev.type == "cuda" else None),
               git_head=(os.popen("git rev-parse HEAD 2>/dev/null")
                         .read().strip() or None),
               code_version=kb.code_version([
                   os.path.abspath(__file__),
                   os.path.join(here, "joint12_vla.py")]),
               script_sha1=sha12(os.path.abspath(__file__)))
    mp = out_p + ".manifest.json"
    tmpm = mp + f".tmp.{os.getpid()}"
    json.dump(man, open(tmpm, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmpm, mp)
    print(f"\n  сохранено: {npz} и {mp}")
    print(f"  ДЛЯ GATE R: повторите с другим --exec-order-seed и сравните "
          f"q0_sha1 по каждой части")
    return 0


if __name__ == "__main__":
    sys.exit(main())
