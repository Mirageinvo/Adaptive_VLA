#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14d: канонический q0 по зафиксированному плану батчей и Gate R.

ЗАЧЕМ. Черновик q0 из кэша K-11a не воспроизводится сегодня побитово: три
пути вычисления совпадают между собой и одинаково расходятся с кэшем. Значит
различается не метод, а план батчей. Протокол D': q0 строится один раз своим
планом, становится артефактом K-14, и от него строятся цели.

GATE R. Тот же план исполняется дважды, в разных процессах, с разным ПОРЯДКОМ
исполнения батчей и неизменным их СОСТАВОМ. Совпадение отпечатков q0 по каждой
части означает, что план достаточен для воспроизводимости. Порядок важен
потому, что сид тренера переставляет батчи: если от этого меняется q0, два
прогона с разными сидами решают разные задачи.

ЧАСТИ БЕРУТСЯ ИЗ РАЗБИЕНИЯ, А НЕ ИЗ КЭША ЦЕЛЕЙ. Кэш целей будет перестроен от
этого самого q0; брать состав частей оттуда значило бы замкнуть зависимость в
круг. Источник — `split.npy` и та же разметка по эпизодам (сид 61, доля 0.4),
что во всех работах.

    python experiments/k14d_build_q0.py --device cuda:0
    python experiments/k14d_build_q0.py --gate-r A.manifest.json B.manifest.json
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


def gate_r(ma, mb):
    """Машинная проверка повторяемости. Расхождение любого поля — отказ.

    ОТЛИЧАТЬСЯ РАЗРЕШЕНО ТОЛЬКО ПОРЯДКУ ИСПОЛНЕНИЯ и тому, что от него
    зависит: имени файла, времени, номеру запуска. Всё остальное обязано
    совпасть, иначе сравниваются два разных вычисления, а не два исполнения
    одного.
    """
    bad = []
    for k in ("plan_sha1", "batch", "declared_parts", "n_batches",
              "cache", "source_cache_sha1", "keys_sha1", "ckpt",
              "joint_sha1", "bar_sha1", "code_version", "script_sha1",
              "q0_dtype", "compute_dtype", "device", "images_sha1",
              "state_npy_sha1", "state_json_sha1", "split_sha1",
              "git_head"):
        va, vb = ma.get(k), mb.get(k)
        if va is None or vb is None:
            bad.append(f"{k}: нет поля ({va} / {vb})")
        elif json.dumps(va, sort_keys=True) != json.dumps(vb, sort_keys=True):
            bad.append(f"{k}: {va} против {vb}")
    if ma.get("exec_order_seed") == mb.get("exec_order_seed"):
        bad.append(f"порядок исполнения одинаков "
                   f"({ma.get('exec_order_seed')}): это не два разных "
                   f"исполнения, а повтор одного")
    for k in ("run_id",):
        if not ma.get(k) or not mb.get(k) or ma[k] == mb[k]:
            bad.append(f"{k}: {ma.get(k)} / {mb.get(k)}")
    for m in (ma, mb):
        if m.get("git_dirty"):
            bad.append("построено при незакоммиченных изменениях")
        if m.get("limit"):
            bad.append(f"построено с ограничением --limit {m['limit']}: это "
                       f"не полный канонический план")
    pa, pb = ma.get("parts") or {}, mb.get("parts") or {}
    if sorted(pa) != sorted(pb):
        bad.append(f"части {sorted(pa)} против {sorted(pb)}")
    per = {}
    for nm in sorted(set(pa) & set(pb)):
        same = all(pa[nm].get(f) == pb[nm].get(f)
                   for f in ("n_rows", "rows_sha1", "q0_sha1", "dtype"))
        per[nm] = dict(a={f: pa[nm].get(f) for f in
                          ("n_rows", "rows_sha1", "q0_sha1")},
                       b={f: pb[nm].get(f) for f in
                          ("n_rows", "rows_sha1", "q0_sha1")}, same=bool(same))
        if not same:
            bad.append(f"часть {nm}: q0 или строки различаются")
    return dict(kind="k14_gate_r", passed=not bad, failures=bad, parts=per,
                batch=ma.get("batch") if ma.get("batch") == mb.get("batch")
                else None,
                declared_parts=ma.get("declared_parts"),
                plan_sha1=ma.get("plan_sha1"),
                exec_order_seeds=[ma.get("exec_order_seed"),
                                  mb.get("exec_order_seed")],
                run_ids=[ma.get("run_id"), mb.get("run_id")])


def selftest():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import k14_common as kc
    kc.selftest()

    base = dict(plan_sha1="P", batch=8, declared_parts=list(kc.CANONICAL_PARTS),
                n_batches=10, cache="c", source_cache_sha1="s",
                keys_sha1="k", ckpt="ck", joint_sha1="j", bar_sha1="b",
                code_version={"x": "1"}, script_sha1="sc", q0_dtype="int32",
                compute_dtype="float16", device="cuda:0", images_sha1="im",
                state_npy_sha1="sn", state_json_sha1="sj", split_sha1="sp",
                git_head="HEAD", git_dirty=False, limit=0,
                parts={p: dict(n_rows=1, rows_sha1="r" + p, q0_sha1="q" + p,
                               dtype="int32") for p in kc.CANONICAL_PARTS})
    A = dict(base, exec_order_seed=0, run_id="A")
    B = dict(base, exec_order_seed=7, run_id="B")
    r = gate_r(A, B)
    assert r["passed"], r["failures"]

    for patch, why in (
            ({"plan_sha1": "Q"}, "plan_sha1"),
            ({"batch": 16}, "batch"),
            ({"joint_sha1": "z"}, "joint_sha1"),
            ({"git_dirty": True}, "незакоммиченных"),
            ({"limit": 2048}, "--limit"),
            ({"images_sha1": None}, "нет поля")):
        r2 = gate_r(dict(A), dict(B, **patch))
        assert not r2["passed"] and any(why in x for x in r2["failures"]), \
            (why, r2["failures"])
    # одинаковый порядок исполнения — это повтор, а не проверка
    r3 = gate_r(dict(A), dict(B, exec_order_seed=0))
    assert not r3["passed"] and any("порядок исполнения одинаков" in x
                                    for x in r3["failures"])
    # различие q0 в одной части
    bad_parts = {p: dict(base["parts"][p]) for p in kc.CANONICAL_PARTS}
    bad_parts["val_sel"] = dict(bad_parts["val_sel"], q0_sha1="OTHER")
    r4 = gate_r(dict(A), dict(B, parts=bad_parts))
    assert not r4["passed"] and any("val_sel" in x for x in r4["failures"])
    assert r4["parts"]["train"]["same"] and not r4["parts"]["val_sel"]["same"]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        q = os.path.join(td, "gate_r.json")
        json.dump(gate_r(A, B), open(q, "w"))
        kc.check_gate_r(q, expect_plan_sha1="P")
    # тот же артефакт, но с расхождением — потребители обязаны отказать
    with tempfile.TemporaryDirectory() as td:
        q = os.path.join(td, "gate_r.json")
        json.dump(r4, open(q, "w"))
        try:
            kc.check_gate_r(q)
        except SystemExit:
            pass
        else:
            raise AssertionError("непройденный Gate R принят потребителем")
    print("самопроверка k14d_build_q0 пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gate-r", nargs=2, metavar=("A", "B"),
                    help="сравнить два манифеста и записать gate_r.json")
    ap.add_argument("--gate-r-out", default="reports/k14d/gate_r.json")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--joint-ckpt", default="data/k9d_ep3.pt")
    ap.add_argument("--ckpt",
                    default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--batch", type=int, default=8,
                    help="ЧАСТЬ ОПРЕДЕЛЕНИЯ ЗАДАЧИ, а не производительности")
    ap.add_argument("--exec-order-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="только для диагностики; канонический план идёт без "
                         "ограничения, и Gate R отвергает ограниченные")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import k14_common as kc

    if a.gate_r:
        ma, mb = (json.load(open(p)) for p in a.gate_r)
        r = gate_r(ma, mb)
        r["manifests"] = list(a.gate_r)
        r["manifest_sha1"] = [sha12(p) for p in a.gate_r]
        print(f"\n  GATE R по плану {r['plan_sha1']}, порядки "
              f"{r['exec_order_seeds']}:")
        for nm, d in sorted(r["parts"].items()):
            print(f"    {nm:12s} {d['a']['q0_sha1']} / {d['b']['q0_sha1']}  "
                  f"{'СОВПАЛИ' if d['same'] else 'РАЗОШЛИСЬ'} "
                  f"({d['a']['n_rows']} строк)")
        for x in r["failures"]:
            print(f"    ОТКАЗ: {x}")
        print(f"\n  GATE R: {'ПРОЙДЕН' if r['passed'] else 'НЕ ПРОЙДЕН'}")
        os.makedirs(os.path.dirname(os.path.abspath(a.gate_r_out)) or ".",
                    exist_ok=True)
        json.dump(r, open(a.gate_r_out, "w"), ensure_ascii=False, indent=1,
                  default=str)
        print(f"  сохранено: {a.gate_r_out}")
        return 0 if r["passed"] else 5

    out_p = a.out or f"data/k14d/q0_b{a.batch}_e{a.exec_order_seed}"
    for suf in (".npz", ".manifest.json"):
        if os.path.exists(out_p + suf):
            raise SystemExit(f"{out_p}{suf} уже существует")
    dirty = os.popen("git status --porcelain 2>/dev/null").read().strip()
    git_head = os.popen("git rev-parse HEAD 2>/dev/null").read().strip()
    if not git_head:
        raise SystemExit("не удалось определить git HEAD")
    if dirty and not a.allow_dirty:
        raise SystemExit(
            f"рабочее дерево не чисто ({len(dirty.splitlines())} файлов). "
            f"Канонический q0 обязан ссылаться на коммит, которым построен:\n"
            f"{dirty[:400]}")
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"

    root = os.path.abspath(a.root)
    for p in (root, os.path.join(root, "src"), os.path.abspath("experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import inspect
    import k11a_build_hicora_cache as k11a
    import k12b_protocol as kb
    import k13a_build_trajectory_basis as k13a
    import k14a_oracle_cache as k14a
    from k11c_train_d1 import split_episodes
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
    idx, _ = k13a.load_split(f"{a.cache}.split.npy", N)

    img_p = os.path.join(os.path.dirname(src), cmeta["images_file"])
    IMG = np.load(img_p, mmap_mode="r")
    if IMG.shape[0] < N or IMG.dtype != np.uint8:
        raise SystemExit(f"кадры {IMG.shape} {IMG.dtype}")
    st_n, sm, st_shas = kc.load_states(src, N, cmeta, keys_sha,
                                       STATE_Q01, STATE_Q99)

    # ЧАСТИ ИЗ РАЗБИЕНИЯ, А НЕ ИЗ КЭША ЦЕЛЕЙ: иначе q0 зависел бы от файла,
    # который сам будет перестроен от этого q0.
    parts, sample_meta = k14a.build_parts(
        idx, epi, kc.SEL_FRAC, kc.SPLIT_SEED, 0, np.random.default_rng(0),
        split_episodes)
    if a.limit:
        parts = {k: v[:a.limit] for k, v in parts.items()}
    plan = kc.make_plan(parts, offs, a.batch)
    p_sha = kc.plan_identity(plan, a.batch)
    print(f"  план: {len(plan)} батчей, размер {a.batch}, sha {p_sha}")
    for nm, st in sorted(kc.plan_stats(plan).items()):
        print(f"    {nm}: {st['batches']} батчей, {st['rows']} строк")

    cfg = get_cfg(os.path.join(root, a.cfg_path))
    cfg.TRAINING.ckpt_dir = a.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = a.ckpt
    Base = make_joint12_class(SmolVLABlockwiseAR)
    model = Base.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    model.init_joint_fast(depth=int(meta["depth"]), head_dtype=torch.float32)
    j_sha, n_t = kc.load_joint12_strict(
        model, a.joint_ckpt, int(meta["depth"]),
        (meta.get("source") or {}).get("weights_sha1"), torch, dev)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    print(f"  Joint12 загружен строго: {n_t} тензоров, sha {j_sha}")

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

    order = np.arange(len(plan))
    if a.exec_order_seed:
        order = np.random.default_rng(a.exec_order_seed).permutation(order)
    q0_out = np.full((N, 16), -1, np.int32)
    ac16 = torch.autocast(device_type=dev.type, dtype=dt)
    t0 = time.time()
    with torch.no_grad():
        for k_, i in enumerate(order):
            _nm, po, sel = plan[int(i)]
            b = build(po, sel)
            with ac16:
                v_, p_ = model.build_inputs(position_offset=po, **b)
                out = model.forward_joint_fast(
                    vlm_inputs_embeds=v_,
                    attention_mask=b.get("attention_mask"), position_ids=p_)
            q0 = out["pred_codes"] if isinstance(out, dict) else out[1]
            q0_out[sel] = q0.cpu().numpy().astype(np.int32)
            if k_ % 500 == 0:
                print(f"    батч {k_}/{len(plan)} "
                      f"({(time.time() - t0) / 60:.1f} мин)", flush=True)

    res, diff = {}, {}
    for nm in kc.CANONICAL_PARTS:
        rr = np.sort(np.asarray(parts[nm], np.int64))
        q = q0_out[rr]
        if (q < 0).any():
            raise SystemExit(f"{nm}: не все строки посчитаны")
        res[nm] = dict(n_rows=int(len(rr)), rows_sha1=kc.arr_sha(rr),
                       q0_sha1=kc.arr_sha(q), dtype=str(q.dtype))
        cq = np.asarray(q0hat_c[rr]).astype(np.int32)
        nb = int((q != cq).sum())
        diff[nm] = dict(mismatch=nb, positions=int(q.size),
                        frac=float(nb) / max(q.size, 1))
        print(f"    {nm}: q0 sha {res[nm]['q0_sha1']}, расхождение с K-11a "
              f"{nb} из {q.size} ({100 * nb / max(q.size, 1):.4f}%)")

    os.makedirs(os.path.dirname(os.path.abspath(out_p)) or ".", exist_ok=True)
    npz = out_p + ".npz"
    arrs = dict(q0=q0_out, **kc.plan_arrays(plan))
    tmp = npz + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrs)
    os.replace(tmp, npz)
    with np.load(npz, allow_pickle=True) as z:
        if sorted(z.files) != sorted(arrs):
            raise SystemExit(f"{npz}: массивы {sorted(z.files)}")
        for k_, v_ in arrs.items():
            if not np.array_equal(z[k_], v_):
                raise SystemExit(f"{npz}: {k_} прочитался иначе")

    dirty2 = os.popen("git status --porcelain 2>/dev/null").read().strip()
    man = dict(
        kind="k14_q0_by_plan", run_id=run_id, batch=int(a.batch),
        declared_parts=list(kc.CANONICAL_PARTS), schema=kc.SCHEMA_VERSION,
        exec_order_seed=int(a.exec_order_seed), plan_sha1=p_sha,
        n_batches=len(plan), plan_stats=kc.plan_stats(plan),
        parts=res, diff_vs_k11a=diff, sampling=sample_meta,
        split_seed=kc.SPLIT_SEED, sel_frac=kc.SEL_FRAC,
        device=str(dev), compute_dtype=a.dtype, q0_dtype="int32",
        cache=a.cache, cache_meta_sha1=k11a.file_sha1(f"{a.cache}.meta.json"),
        split_sha1=k11a.file_sha1(f"{a.cache}.split.npy"),
        q0hat_k11a_sha1=k11a.file_sha1(f"{a.cache}.q0hat.npy"),
        source_cache=src, source_cache_sha1=sha12(src), keys_sha1=keys_sha,
        images_file=img_p, images_sha1=sha12(img_p),
        state_npy_sha1=st_shas["state_npy"],
        state_json_sha1=st_shas["state_json"],
        dataset_repo=sm.get("dataset_repo"),
        dataset_revision=sm.get("dataset_revision"),
        ckpt=a.ckpt, joint_ckpt=a.joint_ckpt, joint_sha1=j_sha,
        limit=int(a.limit), npz_sha1=sha12(npz),
        bar_sha1=sha12(inspect.getfile(SmolVLABlockwiseAR)),
        cfg_sha1=sha12(os.path.join(root, a.cfg_path)),
        tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
        tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        torch_version=str(torch.__version__),
        cuda_version=str(getattr(torch.version, "cuda", None)),
        gpu=(torch.cuda.get_device_name(dev) if dev.type == "cuda" else None),
        git_head=git_head, git_dirty=bool(dirty or dirty2),
        git_dirty_files=len((dirty2 or dirty).splitlines()),
        minutes=float((time.time() - t0) / 60),
        code_version=kb.code_version([
            os.path.abspath(__file__),
            os.path.join(here, "k14_common.py"),
            os.path.join(here, "joint12_vla.py")]),
        script_sha1=sha12(os.path.abspath(__file__)))
    if dirty2 and not dirty and not a.allow_dirty:
        raise SystemExit("дерево стало грязным ВО ВРЕМЯ построения: отпечатки "
                         "кода в начале и в конце не совпадают")
    mp = out_p + ".manifest.json"
    tmpm = mp + f".tmp.{os.getpid()}"
    json.dump(man, open(tmpm, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmpm, mp)
    print(f"\n  сохранено: {npz} и {mp}")
    print(f"  ДЛЯ GATE R: повторите с другим --exec-order-seed, затем "
          f"--gate-r A.manifest.json B.manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
