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


def top2_margin(residual, quantizer, torch, _project, chunk_rows=4096):
    """Разрыв между первым и вторым ближайшими кодами, по токенам.

    СЧИТАЕТСЯ ТАМ, ГДЕ КОДЕК И МЕРЯЕТ — после `in_project`. Разрыв в исходном
    пространстве относился бы к другой метрике, и вывод о близости кодов был бы
    о ней, а не о той, по которой код выбирается.

    Разбито по строкам: матрица (все токены, размер словаря) для целой части
    заняла бы больше гигабайта без всякой нужды.
    """
    enc = _project(quantizer.in_project, residual, "in_project")
    book = quantizer.codebook.float().to(enc.device)
    flat = enc.reshape(-1, enc.shape[-1])
    b2 = book.square().sum(-1).unsqueeze(0)
    out = []
    with torch.no_grad():
        for i in range(0, flat.shape[0], chunk_rows):
            x = flat[i:i + chunk_rows]
            d = x.square().sum(-1, keepdim=True) - 2.0 * x @ book.T + b2
            two = torch.topk(d, 2, dim=-1, largest=False).values
            out.append((two[:, 1] - two[:, 0]).float().cpu())
    return torch.cat(out).numpy().reshape(enc.shape[:-1])


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
    # LIBERO НУЖЕН ДАЖЕ ЗДЕСЬ. `utils` из actioncodec тянет `libero.libero` на
    # уровне импорта модуля, хотя этот скрипт симулятор не запускает. Обычно
    # путь задаётся снаружи через PYTHONPATH; здесь он подставляется сам, но с
    # явным сообщением — прятать требование окружения нельзя, о нём надо
    # знать. Editable-установка в этом окружении не работает, поэтому именно
    # путь, а не пакет.
    try:
        import libero  # noqa: F401
    except ModuleNotFoundError:
        lp = os.environ.get("LIBERO_PATH") or os.path.expanduser("~/LIBERO")
        if not os.path.isdir(lp):
            raise SystemExit(
                f"нет модуля libero и нет каталога {lp}: задайте PYTHONPATH "
                f"или LIBERO_PATH. Он нужен потому, что utils из actioncodec "
                f"импортирует libero.libero, а не потому, что здесь "
                f"запускается симулятор")
        sys.path.insert(0, lp)
        os.environ.setdefault("MUJOCO_GL", "egl")
        print(f"  LIBERO подставлен из {lp} (utils импортирует его на уровне "
              f"модуля)")
    import actioncodec  # noqa: F401
    from utils import ACTION_Q01, ACTION_Q99, VisionLanguageActionProcessor

    oa, ob = json.load(open(a.a)), json.load(open(a.b))

    def labels_of(json_path, o):
        """Путь к меткам, ПЕРЕНОСИМЫЙ между машинами.

        В артефакте лежит абсолютный путь вычислительного узла. После
        копирования JSON в reports/ он не откроется у того, кто проверяет
        результат. Поэтому рядом с JSON ищется файл с тем же именем, и в
        обоих случаях СВЕРЯЕТСЯ sha: подставить чужой файл нельзя.
        """
        lp = o.get("labels_path")
        if not lp:
            raise SystemExit(f"{json_path}: в артефакте нет labels_path")
        if not os.path.exists(lp):
            alt = os.path.join(os.path.dirname(os.path.abspath(json_path)),
                               os.path.basename(lp))
            if not os.path.exists(alt):
                raise SystemExit(f"{json_path}: нет файла меток ни по {lp}, "
                                 f"ни рядом с артефактом")
            print(f"  метки взяты рядом с артефактом: {alt}")
            lp = alt
        want = o.get("labels_sha1")
        if not want:
            raise SystemExit(f"{json_path}: в артефакте нет labels_sha1, "
                             f"подтвердить подлинность меток нечем")
        got = sha12(lp)
        if got != want:
            raise SystemExit(f"{lp}: sha {got}, в артефакте {want}")
        return lp

    # --- ДВА АРТЕФАКТА ОБЯЗАНЫ БЫТЬ ОТ ОДНОГО ЗАПУСКА И ОДНИХ ДАННЫХ ------
    # Иначе сравнивались бы не режимы вычислений, а разные эксперименты.
    for k in ("run_id", "source_cache_sha1", "codebooks_sha1",
              "codec_state_sha1", "ckpt", "horizon", "sample_seed",
              "n_rows", "split_seed", "sel_frac"):
        va, vb = oa.get(k), ob.get(k)
        if str(va) != str(vb):
            raise SystemExit(f"артефакты различаются по {k}: {va} против "
                             f"{vb} — это разные эксперименты, а не два "
                             f"режима одного")
    if not oa.get("run_id"):
        raise SystemExit("в артефактах нет run_id: подтвердить, что они от "
                         "одного запуска, нечем")
    lpa, lpb = labels_of(a.a, oa), labels_of(a.b, ob)
    la, lb = dict(np.load(lpa)), dict(np.load(lpb))
    print(f"  метки: {a.label_a} {lpa}\n         {a.label_b} {lpb}")
    print(f"  общий запуск {oa['run_id']}, кэш {oa.get('source_cache_sha1')}, "
          f"кодек {oa.get('codec_state_sha1')}")

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
    import k11a_build_hicora_cache as k11a
    st_now = k11a.state_sha1(codec)
    if oa.get("codec_state_sha1") and st_now != oa["codec_state_sha1"]:
        raise SystemExit(f"веса кодека сейчас {st_now}, в артефактах "
                         f"{oa['codec_state_sha1']}: декодируется не тем "
                         f"кодеком, которым считались метки")

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
        # ВАРИАНТ A БЕРЁТСЯ ОБЩЕЙ БАЗОЙ ДЛЯ ОБОИХ, поэтому q1 обязан совпасть.
        # Сейчас он совпадает, но если когда-нибудь разойдётся, сравнение q2
        # пойдёт от разных префиксов и потеряет смысл — молча.
        k1 = f"{part}.q1_ze"
        if flip_stats(la[k1], lb[k1])["n_flip"] != 0:
            raise SystemExit(
                f"{part}: метки q1_ze различаются между режимами. Сравнивать "
                f"q2 от общего q1 нельзя: префиксы разные, и разность q2 "
                f"означала бы не то")
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
            # ЗАПАС ДО ВТОРОГО БЛИЖАЙШЕГО КОДА на остатке второго уровня.
            # Без него утверждение «коды почти равноудалены» остаётся
            # гипотезой, а не измерением.
            act_t = torch.from_numpy(
                np.asarray(ACT[rows], np.float32)).to(dev)
            z_e = torch.cat([codec._encode(act_t[i:i + 256].float(),
                                           embodiment_ids=0).float()
                             for i in range(0, len(act_t), 256)])
            marg = top2_margin(z_e - base, qs[2], torch, _project_fp32)
        flip_mask = (la[k2] != lb[k2])
        print(f"    запас до второго ближайшего кода (в пространстве "
              f"in_project):")
        pr["top2_margin"] = margin_summary(marg, flip_mask,
                                           log=lambda m: print("  " + m))
        _f = pr["top2_margin"].get("flipped")
        _s = pr["top2_margin"].get("stable")
        if _f and _s and _s["median"] > 0:
            pr["margin_ratio_flipped_to_stable"] = (
                _f["median"] / _s["median"])
            print(f"      медиана у перевернувшихся меньше медианы "
                  f"устойчивых в "
                  f"{_s['median'] / max(_f['median'], 1e-30):.3g} раз")
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
               run_id=oa.get("run_id"),
               a=dict(path=a.a, label=a.label_a, device=oa.get("device"),
                      json_sha1=sha12(a.a), labels_path=lpa,
                      labels_sha1=oa.get("labels_sha1")),
               b=dict(path=a.b, label=a.label_b, device=ob.get("device"),
                      json_sha1=sha12(a.b), labels_path=lpb,
                      labels_sha1=ob.get("labels_sha1")),
               codec_state_sha1=st_now,
               source_cache_sha1=oa.get("source_cache_sha1"),
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
