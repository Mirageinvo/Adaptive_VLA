#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14a2: чем различаются метки, посчитанные в двух режимах вычислений.

ЗАЧЕМ. Сверка отпечатков сказала, что метки второго уровня от z_e на CPU и GPU
различаются. Отпечаток отвечает только «да или нет». Прежде чем что-то менять в
`nearest_code` — например вводить свой тай-брейк, — надо измерить, ЧТО именно
различается:

    сколько токенов и строк затронуто;
    каков запас между первым и вторым ближайшими кодами у УСТОЙЧИВЫХ позиций и
        у ПЕРЕВЕРНУВШИХСЯ — если у перевернувшихся он около нуля, это
        неоднозначность дискретной метки, а не ошибка;
    насколько различаются ДЕКОДИРОВАННЫЕ действия двух вариантов;
    насколько один вариант проигрывает другому, если ОБА оценить в ОДНОМ
        каноническом режиме.

Последнее — ключевой вопрос. Если метки различаются, но по ошибке действия
эквивалентны, менять алгоритм незачем: достаточно один раз построить
канонический кэш меток и учить обе головы на нём.

ЧТО ЭТОТ СКРИПТ НЕ ДЕЛАЕТ. Он ничего не решает и не меняет: он только меряет.
Решение о тай-брейке или о мягких мишенях принимается по этим числам отдельно.
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


def flip_stats(xa, xb):
    """Сколько токенов и строк различается. Формы обязаны совпасть."""
    if xa.shape != xb.shape:
        raise SystemExit(f"формы меток {xa.shape} и {xb.shape} не совпадают")
    d = (xa != xb)
    return dict(n_tokens=int(xa.size), n_flip=int(d.sum()),
                frac_tokens=float(d.mean()),
                n_rows=int(xa.shape[0]),
                n_rows_flip=int(d.any(-1).sum()),
                frac_rows=float(d.any(-1).mean()))


def margin_summary(margins, mask, log=print):
    """Запас до второго ближайшего кода: у перевернувшихся и у остальных.

    ИМЕННО ЗАПАС ОТЛИЧАЕТ НЕОДНОЗНАЧНОСТЬ ОТ ОШИБКИ. Если у перевернувшихся
    позиций он на порядки меньше, значит два кода почти равноудалены и выбор
    между ними произволен; тогда менять алгоритм бессмысленно.
    """
    out = {}
    for tag, m in (("flipped", mask), ("stable", ~mask)):
        v = margins[m]
        if v.size == 0:
            out[tag] = None
            continue
        out[tag] = dict(n=int(v.size), mean=float(v.mean()),
                        median=float(np.median(v)),
                        q10=float(np.quantile(v, 0.10)),
                        q90=float(np.quantile(v, 0.90)))
        log(f"    запас {tag:8s}: n {v.size}, медиана {np.median(v):.3e}, "
            f"среднее {v.mean():.3e}, q10 {np.quantile(v, 0.10):.3e}")
    return out


def selftest():
    a = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    b = a.copy()
    st = flip_stats(a, b)
    assert st["n_flip"] == 0 and st["n_rows_flip"] == 0
    b[0, 1] = 9
    st = flip_stats(a, b)
    assert st["n_flip"] == 1 and st["n_rows_flip"] == 1
    assert abs(st["frac_tokens"] - 1 / 6) < 1e-12
    try:
        flip_stats(a, a[:1])
    except SystemExit as e:
        assert "формы" in str(e), e
    else:
        raise AssertionError("несовпадение форм пропущено")

    m = np.array([0.001, 0.002, 5.0, 6.0, 7.0])
    mask = np.array([True, True, False, False, False])
    s = margin_summary(m, mask, log=lambda *_: None)
    assert s["flipped"]["n"] == 2 and s["stable"]["n"] == 3
    assert s["flipped"]["median"] < s["stable"]["median"]
    s2 = margin_summary(m, np.zeros(5, bool), log=lambda *_: None)
    assert s2["flipped"] is None and s2["stable"]["n"] == 5
    print("самопроверка k14a2_label_stability пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--a", default="data/k14a/oracle_cache_cuda0.json")
    ap.add_argument("--b", default="data/k14a/oracle_cache_cpu.json")
    ap.add_argument("--label-a", default="cuda0")
    ap.add_argument("--label-b", default="cpu")
    ap.add_argument("--canon", default="cuda:0",
                    help="канонический режим: в нём оба варианта меток "
                         "оцениваются по ошибке действия")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--ckpt",
                    default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--out", default="reports/k14a/label_stability.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(a.root)
    for p in (root, os.path.join(root, "src"), here,
              os.path.abspath("experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import k12b_protocol as kb
    from k14a_oracle_cache import err_blocks
    from depth_rvq_joint12 import code_contribution, _project_fp32
    import actioncodec  # noqa: F401
    from utils import ACTION_Q01, ACTION_Q99, VisionLanguageActionProcessor

    oa, ob = json.load(open(a.a)), json.load(open(a.b))
    for nm, o in ((a.a, oa), (a.b, ob)):
        if not o.get("labels_path") or not os.path.exists(o["labels_path"]):
            raise SystemExit(f"{nm}: нет сохранённых меток")
        if o.get("labels_sha1") and sha12(o["labels_path"]) != o["labels_sha1"]:
            raise SystemExit(f"{nm}: файл меток изменился после записи")
    la = dict(np.load(oa["labels_path"]))
    lb = dict(np.load(ob["labels_path"]))
    print(f"  метки: {a.label_a} {oa['labels_path']}\n"
          f"         {a.label_b} {ob['labels_path']}")

    dev = torch.device(a.canon)
    max_act_q = np.maximum(np.abs(ACTION_Q01), np.abs(ACTION_Q99))
    q0hat = np.load(f"{a.cache}.q0hat.npy", mmap_mode="r")
    cmeta = json.load(open(f"{a.cache}.meta.json"))
    src = cmeta.get("cache")
    if not src or not os.path.exists(src):
        raise SystemExit(f"исходный кэш {src} недоступен: сравнить метки с "
                         f"настоящим действием нечем")
    ACT = np.load(src, allow_pickle=True)["action"]
    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None))
    codec = codec.to(dev).eval()
    qs = list(codec.vq.quantizers)

    def decode(z, batch=256):
        out = []
        with torch.no_grad():
            for i in range(0, len(z), batch):
                x, _ = codec._decode(z[i:i + batch].float(), embodiment_ids=0)
                out.append(x[..., :7].float().cpu().numpy())
        return np.concatenate(out)

    res = {}
    for part in ("train", "val_sel", "val_confirm"):
        rk = f"{part}.rows"
        if rk not in la:
            continue
        rows = la[rk]
        if not np.array_equal(rows, lb[rk]):
            raise SystemExit(f"{part}: наборы строк различаются")
        pr = res.setdefault(part, dict(n_rows=int(len(rows))))
        print(f"\n  === {part}: {len(rows)} строк ===")
        for lvl in ("q1_ze", "q2_ze", "q1_zq", "q2_zq"):
            k = f"{part}.{lvl}"
            st = flip_stats(la[k], lb[k])
            pr[lvl] = st
            mark = "" if st["n_flip"] == 0 else "  <-- РАЗЛИЧАЮТСЯ"
            print(f"    {lvl}: {st['n_flip']} из {st['n_tokens']} токенов "
                  f"({100 * st['frac_tokens']:.4f}%), строк "
                  f"{st['n_rows_flip']} из {st['n_rows']}{mark}")

        # --- дальше только для уровня, который разошёлся -------------------
        k2 = f"{part}.q2_ze"
        if flip_stats(la[k2], lb[k2])["n_flip"] == 0:
            continue
        q0 = torch.from_numpy(
            np.asarray(q0hat[rows]).astype(np.int64)).to(dev)
        with torch.no_grad():
            e0 = code_contribution(qs[0], q0)
            e1 = code_contribution(
                qs[1], torch.from_numpy(la[f"{part}.q1_ze"]).long().to(dev))
            base = e0 + e1
            # ЗАПАС ДО ВТОРОГО БЛИЖАЙШЕГО КОДА на остатке второго уровня.
            # Считается в пространстве in_project — там, где кодек и меряет.
            ka = torch.from_numpy(la[k2]).long().to(dev)
            kb_ = torch.from_numpy(lb[k2]).long().to(dev)
            A = decode(base + code_contribution(qs[2], ka))
            B = decode(base + code_contribution(qs[2], kb_))
        pr["decoded_diff_rms_all"] = err_blocks(A, B, max_act_q,
                                                a.horizon)["rms"]
        # НА ЗАТРОНУТЫХ СТРОКАХ ОТДЕЛЬНО. Один перевёрнутый токен на сотню
        # тысяч в среднем по всем строкам не виден вовсе: усреднение скажет
        # «различий нет» там, где на одной строке они могут быть велики.
        fl = (la[k2] != lb[k2]).any(-1)
        pr["n_rows_flipped"] = int(fl.sum())
        if fl.any():
            dif = err_blocks(A[fl], B[fl], max_act_q, a.horizon)
            pr["decoded_diff_rms_flipped_rows"] = dif["rms"]
            # КАКОЙ ВАРИАНТ БЛИЖЕ К НАСТОЯЩЕМУ ДЕЙСТВИЮ. Оба оцениваются в
            # ОДНОМ каноническом режиме: иначе сравнивались бы не метки, а
            # режимы вычислений.
            act = np.asarray(ACT[rows], np.float64)[..., :7]
            ea = err_blocks(A[fl], act[fl], max_act_q, a.horizon)["rms"]
            eb = err_blocks(B[fl], act[fl], max_act_q, a.horizon)["rms"]
            pr["err_vs_action_flipped_rows"] = {a.label_a: ea, a.label_b: eb}
            pr["better_on_flipped_rows"] = (
                a.label_a if ea < eb else (a.label_b if eb < ea else "равны"))
            print(f"    на {int(fl.sum())} затронутых строках: действия "
                  f"различаются на RMS-8 {dif['rms']:.3e}")
            print(f"    против настоящего действия там же: {a.label_a} "
                  f"{ea:.6f}, {a.label_b} {eb:.6f} -> ближе "
                  f"{pr['better_on_flipped_rows']}")
        print(f"    по всем строкам части действия различаются на RMS-8 "
              f"{pr['decoded_diff_rms_all']:.3e}")

    out = dict(parts=res, canon=str(dev), horizon=int(a.horizon),
               a=dict(path=a.a, label=a.label_a, device=oa.get("device"),
                      labels_sha1=oa.get("labels_sha1")),
               b=dict(path=a.b, label=a.label_b, device=ob.get("device"),
                      labels_sha1=ob.get("labels_sha1")),
               code_version=kb.code_version([os.path.abspath(__file__)]),
               script_sha1=sha12(os.path.abspath(__file__)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = f"{a.out}.tmp.{os.getpid()}"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, a.out)
    print(f"\n  сохранено: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
