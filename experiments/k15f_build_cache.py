#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15f: кэш h18 исходного замороженного backbone (только h18, без h24).

h18 здесь — состояние action-токенов после 18-го слоя в каноническом
проходе q0: префикс 0..11 как у forward_depth_aligned_rvq, затем исходные
слои 12..17 БЕЗ feedback K-14, без LoRA и φ K-15d (`plain_h18`).

ПОЧЕМУ НЕ КЭШ K-15c. Его h18 снят pre-hook'ом внутри
forward_depth_aligned_rvq, где после 12-го слоя применяется feedback K-14
(маска configure_depth_aligned_rvq = (True, True)). Построитель проверяет
маску на живой модели и пишет результат в манифест: при feedback[0] =
True переиспользование невозможно по построению.

ЧТО ХРАНИТСЯ ПО ЧАСТЯМ train и val_sel (val_confirm не открывается):
  rows      [n]           строки плана, порядок первого появления;
  h18       [n, 16, D]    fp16 (проход идёт в fp16 — хранение без потерь);
  q0        [n, 16]       канонические коды (сверены побитово);
  a0        [n, 8, 7]     декод q0 на исполняемых шагах;
  act       [n, 8, 7]     демонстрация на тех же шагах;
и norm.npz — вес и eps финальной нормы action expert: предобучение базиса
идёт по кэшу без загрузки модели.

Строка, встретившаяся в плане повторно, берётся из ПЕРВОГО батча: состав
батча влияет на числа, и смешивать два прохода одной строки нельзя.
Публикация атомарная: временный каталог -> manifest.json -> COMPLETE ->
переименование.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k15_context  # noqa: E402
import k15d_depth_refine as dr  # noqa: E402
import k15f_continuous_refine as kf  # noqa: E402

PARTS = ("train", "val_sel")
KIND = "k15f_h18_cache"


def sha_file(p):
    return k15_context.sha12(p)


def file_sha(p):
    return k15_context.sha12(p)


ENV_KEYS = ("joint_sha1", "plan_sha1", "q0_prov", "codec",
            "architecture_code_version", "k15a_gate")


def check_env(man, ctx, frozen):
    """Кэш снят в той же обстановке, что текущий контекст. Проблемы."""
    import k15_context as kc_
    cur = json.loads(json.dumps(dict(
        joint_sha1=ctx.joint_sha, plan_sha1=ctx.q0_prov["plan_sha1"],
        q0_prov=ctx.q0_prov, codec=ctx.codec_fp,
        architecture_code_version=ctx.code_version,
        k15a_gate=dict(ctx.gate_info)), default=kc_.k15t.json_scalar))
    p = [f"{k}: кэш {man.get(k)!r}, сейчас {cur[k]!r}"
         for k in ENV_KEYS if man.get(k) != cur[k]]
    if man.get("frozen_sha1") != frozen:
        p.append("замороженное не то, что при построении кэша")
    return p


def strided(n, k):
    if k >= n:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


def main():
    ap = argparse.ArgumentParser(description="K-15f: кэш h18")
    k15_context.add_common_arguments(ap)
    ap.add_argument("--out", default="data/k15f/h18_cache")
    ap.add_argument("--smoke-batches", type=int, default=0,
                    help="0 — полный кэш; иначе по N батчей на часть "
                         "(равномерно), в каталог <out>_smoke")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    smoke = int(a.smoke_batches) > 0
    final = a.out + ("_smoke" if smoke else "")
    if os.path.exists(os.path.join(final, "COMPLETE")) and not a.overwrite:
        raise SystemExit(f"{final} уже готов; --overwrite осознанно")
    t0 = time.time()
    ctx = k15_context.build(a)
    torch, model, dev = ctx.torch, ctx.model, ctx.dev
    k15t = k15_context.k15t
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    frozen0, n_frozen, _e = k15t.frozen_content_sha(model, torch, set())
    mask = getattr(model, "depth_aligned_feedback_mask", None)
    k15c_check = dict(
        k15c_hook="pre-hook depth_rvq_norms[0] внутри "
                  "forward_depth_aligned_rvq",
        feedback_mask=list(mask) if mask is not None else None,
        k15c_h18_includes_feedback=bool(mask is not None and mask[0]),
        # неизвестная маска — переиспользование НЕ доказано, значит невозможно
        reuse_possible=bool(mask is not None and not mask[0]))
    print(f"  кэш K-15c: маска feedback {mask} -> переиспользование "
          f"{'невозможно' if not k15c_check['reuse_possible'] else 'возможно'}")

    parts = {p: list(ctx.parts_full[p]) for p in PARTS}
    if smoke:
        parts = {p: [v[i] for i in strided(len(v), int(a.smoke_batches))]
                 for p, v in parts.items()}
    rows = {p: dr.plan_rows(parts[p]) for p in PARTS}
    slot = {p: {int(r): i for i, r in enumerate(rows[p])} for p in PARTS}
    d_model = int(model.action_expert.norm.weight.shape[0])
    n_pos = int(model.block_size)
    need = sum(len(rows[p]) for p in PARTS) * (n_pos * d_model * 2 + 600)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(final))
                             or ".").free
    print(f"  строк: train {len(rows['train'])}, val_sel "
          f"{len(rows['val_sel'])}; d_model {d_model}; нужно "
          f"{need / 2**30:.1f} ГиБ, свободно {free / 2**30:.1f} ГиБ")
    if need * 1.1 > free:
        raise SystemExit("мало места на диске")

    tmp = final + f".tmp.{os.getpid()}"
    os.makedirs(tmp)
    ok = False
    try:
        mm = {}
        for p in PARTS:
            n = len(rows[p])
            mm[(p, "h18")] = np.lib.format.open_memmap(
                os.path.join(tmp, f"{p}_h18.npy"), mode="w+",
                dtype=np.float16, shape=(n, n_pos, d_model))
            mm[(p, "q0")] = np.zeros((n, n_pos), np.int64)
            mm[(p, "a0")] = np.zeros((n, kf.H_EXEC, 7), np.float32)
            mm[(p, "act")] = np.zeros((n, kf.H_EXEC, 7), np.float32)
            mm[(p, "filled")] = np.zeros(n, bool)
        q0_dev = torch.as_tensor(np.asarray(ctx.q0_can), device=dev)
        ac16 = torch.autocast(device_type=dev.type, dtype=ctx.dt)
        total = sum(len(v) for v in parts.values())
        done, lossy, dups = 0, 0, 0
        for p in PARTS:
            for po, sel in parts[p]:
                b = ctx.build_batch(po, sel)
                with ac16:
                    v, p_ids = model.build_inputs(position_offset=po, **b)
                    q0, a0, h18 = kf.plain_h18(
                        model, vlm_inputs_embeds=v,
                        attention_mask=b.get("attention_mask"),
                        position_ids=p_ids, decode=ctx.decode_fp32)
                sel_t = torch.as_tensor(sel, device=dev)
                bad = int((q0 != q0_dev[sel_t]).sum())
                if bad:
                    raise SystemExit(f"q0 разошёлся с каноническим в {bad} "
                                     f"позициях (часть {p})")
                if not bool(torch.isfinite(h18).all()):
                    raise SystemExit("h18 не конечен")
                lossy += int((h18.half().to(h18.dtype) != h18).sum())
                act = np.asarray(ctx.ACT[sel], np.float32)[:, :kf.H_EXEC,
                                                           :7]
                h_np = h18.half().cpu().numpy()
                a0_np = a0[:, :kf.H_EXEC].cpu().numpy()
                q_np = q0.cpu().numpy()
                for k, r in enumerate(np.asarray(sel, np.int64).tolist()):
                    i = slot[p][int(r)]
                    if mm[(p, "filled")][i]:
                        dups += 1
                        continue
                    mm[(p, "h18")][i] = h_np[k]
                    mm[(p, "q0")][i] = q_np[k]
                    mm[(p, "a0")][i] = a0_np[k]
                    mm[(p, "act")][i] = act[k]
                    mm[(p, "filled")][i] = True
                done += 1
                if done % 500 == 0 or done == total:
                    el = time.time() - t0
                    print(f"  батч {done}/{total}, {el / 60:.0f} мин, "
                          f"осталось ~{el / done * (total - done) / 60:.0f} "
                          f"мин", flush=True)
        for p in PARTS:
            if not bool(mm[(p, "filled")].all()):
                raise SystemExit(f"часть {p}: не все строки заполнены")
            mm[(p, "h18")].flush()
            np.save(os.path.join(tmp, f"{p}_rows.npy"), rows[p])
            for k in ("q0", "a0", "act"):
                np.save(os.path.join(tmp, f"{p}_{k}.npy"), mm[(p, k)])
        w, eps = kf.norm_of(model)
        np.savez(os.path.join(tmp, "norm.npz"), weight=w.numpy(), eps=eps)
        # ОТПЕЧАТОК КАЖДОГО ФАЙЛА, включая h18: загрузчик пересчитывает их
        arrays = {}
        for fn in sorted(os.listdir(tmp)):
            if fn.endswith((".npy", ".npz")):
                arrays[fn] = file_sha(os.path.join(tmp, fn))
        frozen1, _n, _e = k15t.frozen_content_sha(model, torch, set())
        if frozen1 != frozen0:
            raise SystemExit("замороженное изменилось")
        man = dict(
            kind=KIND, smoke=smoke, canonical=not smoke,
            created=datetime.datetime.now().isoformat(timespec="seconds"),
            rows={p: int(len(rows[p])) for p in PARTS},
            rows_sha1={p: hashlib.sha1(np.ascontiguousarray(rows[p])
                                       .tobytes()).hexdigest()[:12]
                       for p in PARTS},
            d_model=d_model, n_pos=n_pos, h_exec=kf.H_EXEC,
            vocab=int(ctx.vocab), array_sha1=arrays,
            h18_depth=kf.H18_DEPTH, fp16_lossy_values=int(lossy),
            duplicate_rows_skipped=int(dups), k15c_reuse_check=k15c_check,
            device=str(dev), compute_dtype=a.dtype,
            code=dict(k15f_continuous_refine=sha_file(kf.__file__),
                      k15f_build_cache=sha_file(os.path.abspath(__file__))),
            architecture_code_version=ctx.code_version,
            joint_sha1=ctx.joint_sha, plan_sha1=ctx.q0_prov["plan_sha1"],
            q0_prov=ctx.q0_prov, codec=ctx.codec_fp,
            k15a_gate=dict(ctx.gate_info), frozen_sha1=frozen0,
            frozen_tensors=n_frozen, git_head=ctx.git_head,
            dirty=bool(ctx.dirty), val_confirm_opened=False,
            seconds=round(time.time() - t0, 1))
        with open(os.path.join(tmp, "manifest.json"), "w") as f:
            json.dump(man, f, indent=1, ensure_ascii=False,
                      default=k15t.json_scalar)
        open(os.path.join(tmp, "COMPLETE"), "w").write(man["created"])
        if os.path.exists(final):
            bak = f"{final}.bak.{os.getpid()}"
            os.replace(final, bak)
            print(f"  прежний кэш -> {bak}")
        os.replace(tmp, final)
        ok = True
        print(f"КЭШ h18 ГОТОВ: {final}; train {man['rows']['train']}, "
              f"val_sel {man['rows']['val_sel']}; потерь fp16 {lossy}; "
              f"{man['seconds'] / 60:.0f} мин")
    finally:
        if not ok and os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


def load_cache(path, *, allow_smoke=False, verify_h18=True):
    """Кэш со СТРОГОЙ проверкой; любое расхождение — отказ.

    Пересчитываются отпечатки всех файлов (h18 — если verify_h18),
    сверяются формы, dtype, конечность a0/act/h18 и диапазон кодов q0.
    """
    if not os.path.exists(os.path.join(path, "COMPLETE")):
        raise SystemExit(f"{path}: нет COMPLETE")
    man = json.load(open(os.path.join(path, "manifest.json")))
    if man.get("kind") != KIND:
        raise SystemExit(f"{path}: kind {man.get('kind')!r}")
    if man.get("smoke") and not allow_smoke:
        raise SystemExit(f"{path}: smoke-кэш")
    arrays = man.get("array_sha1") or {}
    want_files = {f"{p}_{k}.npy" for p in PARTS
                  for k in ("rows", "h18", "q0", "a0", "act")} | {"norm.npz"}
    if set(arrays) != want_files:
        raise SystemExit(f"{path}: в манифесте не все отпечатки файлов")
    for fn, sha in arrays.items():
        if fn.endswith("_h18.npy") and not verify_h18:
            continue
        if file_sha(os.path.join(path, fn)) != sha:
            raise SystemExit(f"{path}: {fn} изменён")
    out = dict(manifest=man)
    npos, dm, he = man["n_pos"], man["d_model"], man["h_exec"]
    for p in PARTS:
        n = man["rows"][p]
        rows = np.load(os.path.join(path, f"{p}_rows.npy"))
        h18 = np.load(os.path.join(path, f"{p}_h18.npy"), mmap_mode="r")
        arr = {k: np.load(os.path.join(path, f"{p}_{k}.npy"))
               for k in ("q0", "a0", "act")}
        spec = dict(rows=((n,), np.int64), h18=((n, npos, dm), np.float16),
                    q0=((n, npos), np.int64), a0=((n, he, 7), np.float32),
                    act=((n, he, 7), np.float32))
        for k, (shape, dt) in spec.items():
            x = rows if k == "rows" else h18 if k == "h18" else arr[k]
            if tuple(x.shape) != shape or x.dtype != dt:
                raise SystemExit(f"{path}: {p}_{k} формы {x.shape} "
                                 f"{x.dtype}, ожидалось {shape} {dt}")
        if hashlib.sha1(np.ascontiguousarray(rows).tobytes()
                        ).hexdigest()[:12] != man["rows_sha1"][p]:
            raise SystemExit(f"{path}: строки части {p} изменены")
        if len(np.unique(rows)) != n:
            raise SystemExit(f"{path}: повторяющиеся строки в {p}")
        for k in ("a0", "act"):
            if not np.isfinite(arr[k]).all():
                raise SystemExit(f"{path}: {p}_{k} не конечен")
        q = arr["q0"]
        if q.min() < 0 or q.max() >= int(man["vocab"]):
            raise SystemExit(f"{path}: коды q0 вне [0, {man['vocab']})")
        if verify_h18:
            for s0 in range(0, n, 4096):
                if not np.isfinite(np.asarray(h18[s0:s0 + 4096],
                                              np.float32)).all():
                    raise SystemExit(f"{path}: {p}_h18 не конечен")
        out[p] = dict(rows=rows, h18=h18, **arr)
    nz = np.load(os.path.join(path, "norm.npz"))
    out["norm"] = (nz["weight"], float(nz["eps"]))
    if out["norm"][0].shape != (dm,) or not np.isfinite(out["norm"][0]).all():
        raise SystemExit(f"{path}: норма неверной формы")
    return out


def verify_main(path):
    """Для раннера: 0 — кэш цел и пригоден, 1 — нет."""
    try:
        load_cache(path)
    except SystemExit as e:
        print(f"кэш {path} не принят: {e}")
        return 1
    print(f"кэш {path} цел")
    return 0

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--verify":
        sys.exit(verify_main(sys.argv[2]))
    sys.exit(main())
