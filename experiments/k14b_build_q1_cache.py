#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14b: канонический кэш условных меток q1*. Один режим, один файл, один sha.

ЗАЧЕМ КАНОНИЧЕСКИЙ КЭШ, А НЕ ПЕРЕСЧЁТ В ТРЕНЕРЕ. Метка q1* выбирается как
ближайший код к остатку z_e - E0[q0hat], и у части позиций два кода почти
равноудалены: измеренный разрыв там порядка 5e-08 при величинах 1e-02, то есть
несколько единиц последнего разряда fp32 (§42). Пересчитывай тренер метки сам,
два training seed на cuda:0 и cuda:1 получили бы РАЗНЫЕ задачи, и разность
между ними нельзя было бы приписать порядку данных. Поэтому метки строятся
ОДИН РАЗ в одном режиме, кладутся целыми числами, и обе головы читают один
файл с одним отпечатком.

ЧТО ЗДЕСЬ ЕСТЬ И ЧЕГО НЕТ. Только q1*. Оракульных q2* здесь НЕТ намеренно:
обучать q2 надо относительно ФАКТИЧЕСКИ предсказанного замороженной головой
q1, а не относительно оракульного. Кэш q2 строится позже и отдельно для каждой
обученной головы — иначе вернётся teacher-forcing mismatch, ради устранения
которого всё и делается.

ЧАСТИ. train, val_sel, val_confirm — теми же строками и тем же разбиением по
эпизодам (сид 61, доля 0.4), что в K-11c, K-13b и K-14a. Финальная выборка не
читается вовсе.
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


def check_codes(q1, vocab, q0hat, ktrue):
    """Метки обязаны быть целыми, в диапазоне и той же формы, что черновик."""
    if q1.dtype.kind != "i":
        raise SystemExit(f"метки типа {q1.dtype}: кэш обязан быть целым, "
                         f"иначе тренер получит числа с плавающей точкой и "
                         f"молча приведёт их сам")
    if q1.shape != q0hat.shape:
        raise SystemExit(f"метки формы {q1.shape}, черновик {q0hat.shape}")
    lo, hi = int(q1.min()), int(q1.max())
    if lo < 0 or hi >= vocab:
        raise SystemExit(f"метки в диапазоне [{lo}, {hi}] при словаре {vocab}")
    # ДИАГНОСТИКА, А НЕ ПРОВЕРКА: доля совпадений условной метки с истинной.
    # Совпадать они не обязаны — в том и смысл условной переразметки.
    return dict(vocab=int(vocab), lo=lo, hi=hi,
                frac_equal_static=float((q1 == ktrue).mean()))


def check_manifest(man, want):
    """Сверка манифеста кэша с ожидаемым составом. Отсутствие поля — отказ."""
    miss = [k for k in want if man.get(k) is None]
    if miss:
        raise SystemExit(f"в манифесте нет полей {miss}")
    bad = [(k, man[k], v) for k, v in want.items() if str(man[k]) != str(v)]
    if bad:
        raise SystemExit("манифест не совпал: "
                         + "; ".join(f"{k}: в кэше {a}, ожидалось {b}"
                                     for k, a, b in bad))
    return True


def selftest():
    q0 = np.zeros((5, 16), np.int64)
    kt = np.zeros((5, 16), np.int64)
    q1 = np.arange(80, dtype=np.int64).reshape(5, 16) % 10
    st = check_codes(q1, 2048, q0, kt)
    assert st["lo"] == 0 and st["hi"] == 9
    assert abs(st["frac_equal_static"] - (q1 == kt).mean()) < 1e-12
    for bad, why in ((q1.astype(np.float32), "целым"),
                     (q1[:, :8], "формы"),
                     (q1 + 5000, "диапазоне")):
        try:
            check_codes(bad, 2048, q0, kt)
        except SystemExit as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"пропущено: {why}")

    man = dict(a="1", b="2")
    check_manifest(man, dict(a="1"))
    try:
        check_manifest(man, dict(a="9"))
    except SystemExit as e:
        assert "не совпал" in str(e), e
    else:
        raise AssertionError("расхождение манифеста пропущено")
    try:
        check_manifest(man, dict(c="1"))
    except SystemExit as e:
        assert "нет полей" in str(e), e
    else:
        raise AssertionError("отсутствующее поле пропущено")
    print("самопроверка k14b_build_q1_cache пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--ckpt",
                    default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--device", default="cuda:0",
                    help="КАНОНИЧЕСКИЙ режим разметки; пишется в манифест")
    ap.add_argument("--sel-frac", type=float, default=0.4)
    ap.add_argument("--split-seed", type=int, default=61)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="data/k14b/q1_cache")
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
    import k11a_build_hicora_cache as k11a
    import k12b_protocol as kb
    import k13a_build_trajectory_basis as k13a
    import k14a_oracle_cache as k14a
    from k11c_train_d1 import split_episodes
    from depth_rvq_joint12 import code_contribution, nearest_code
    import actioncodec  # noqa: F401
    from utils import VisionLanguageActionProcessor

    dev = torch.device(a.device)

    # --- ПРОИСХОЖДЕНИЕ: ТОТ ЖЕ НАБОР ПРОВЕРОК, ЧТО В K-14a ------------------
    # Дублировать их нельзя, поэтому переиспользуются функции оттуда; кэш
    # меток обязан быть построен на тех же данных, на которых пройден гейт.
    meta = json.load(open(f"{a.cache}.meta.json"))
    if meta.get("ckpt") != a.ckpt:
        raise SystemExit(f"кэш собран чекпойнтом {meta.get('ckpt')}, а кодек "
                         f"берётся из {a.ckpt}")
    if meta.get("q0_source") != "joint12" or int(meta.get("depth", -1)) != 12:
        raise SystemExit(f"кэш собран источником {meta.get('q0_source')} на "
                         f"глубине {meta.get('depth')}")
    stamp_p = a.cache + ".artifacts.json"
    if not os.path.exists(stamp_p):
        raise SystemExit(f"нет {stamp_p}: кэш не заверен K-11b")
    stamp = json.load(open(stamp_p))
    if not stamp.get("identity_ok"):
        raise SystemExit("K-11b не подтвердила тождество для этого кэша")
    for nm in ("q0hat", "ktrue", "split", "codebooks"):
        got = k11a.file_sha1(f"{a.cache}.{nm}.npy")
        if got != (stamp.get("arrays") or {}).get(nm):
            raise SystemExit(f"{nm}.npy имеет sha {got}, K-11b заверила "
                             f"{(stamp.get('arrays') or {}).get(nm)}")

    q0hat = np.load(f"{a.cache}.q0hat.npy", mmap_mode="r")
    ktrue = np.load(f"{a.cache}.ktrue.npy", mmap_mode="r")
    E = np.load(f"{a.cache}.codebooks.npy")
    N = int(meta["n_obs"])
    idx, _ = k13a.load_split(f"{a.cache}.split.npy", N)

    src = meta.get("cache")
    if not src or not os.path.exists(src):
        raise SystemExit(f"исходный кэш {src} недоступен")
    src_npz = np.load(src, allow_pickle=True)
    keys_now = hashlib.sha1(np.ascontiguousarray(np.stack(
        [np.asarray(src_npz["episode"]),
         np.asarray(src_npz["step"])])).tobytes()).hexdigest()[:12]
    if not meta.get("keys_sha1") or keys_now != meta["keys_sha1"]:
        raise SystemExit(f"(episode, step) дают {keys_now}, в кэше "
                         f"{meta.get('keys_sha1')}")
    ACT = src_npz["action"]
    if ACT.shape[0] != N:
        raise SystemExit(f"в исходном кэше {ACT.shape[0]} действий при {N}")
    epi = np.asarray(src_npz["episode"]).astype(np.int64)[:N]

    # --- кодек --------------------------------------------------------------
    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None))
    codec = codec.to(dev).eval()
    qs = list(codec.vq.quantizers)
    with torch.no_grad():
        ii = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        Ecur = torch.stack([q.out_project(q.decode_code(ii))[0]
                            for q in qs]).float()
    if float((Ecur.cpu() - torch.from_numpy(E)).abs().max()) > 1e-5:
        raise SystemExit("книги разошлись с кэшем")
    got_w = dict(codebooks_sha1=arr_sha(E.astype(np.float32)),
                 codec_state_sha1=k11a.state_sha1(codec))
    bad_w = [k for k, v in got_w.items() if meta.get(k) != v]
    if bad_w:
        raise SystemExit(f"веса кодека не те, которыми собран кэш: {bad_w}")
    probe_now = k11a.decoder_probe(codec, Ecur.to(dev), dev)
    if meta.get("decoder_probe") != probe_now:
        raise SystemExit(
            f"поведение декодера отличается от записанного при сборке кэша "
            f"(в кэше {meta.get('decoder_probe')}, сейчас {probe_now}). Кэш "
            f"меток обязан строиться в ТОМ ЖЕ режиме, в каком пройден гейт "
            f"K-14a: устройство {dev}")
    print(f"  происхождение сверено: кэш {a.cache}, кодек {got_w}, "
          f"канонический режим {dev}")

    # --- части: те же, что в K-14a -----------------------------------------
    parts, sample_meta = k14a.build_parts(
        idx, epi, a.sel_frac, a.split_seed, 0, np.random.default_rng(0),
        split_episodes)
    print("  части: " + ", ".join(f"{k} {v['n_used']}"
                                  for k, v in sample_meta.items())
          + ". Подвыборки нет; финальная выборка не читается")

    # --- метки --------------------------------------------------------------
    out_rows, out_q1, out_part = [], [], []
    stats = {}
    for name in ("train", "val_sel", "val_confirm"):
        rows = parts[name]
        acc = []
        for i in range(0, len(rows), a.batch):
            r = rows[i:i + a.batch]
            with torch.no_grad():
                act = torch.from_numpy(
                    np.asarray(ACT[r], np.float32)).to(dev)
                z_e = codec._encode(act, embodiment_ids=0).float()
                e0 = code_contribution(
                    qs[0], torch.from_numpy(
                        np.asarray(q0hat[r]).astype(np.int64)).to(dev))
                acc.append(nearest_code(z_e - e0, qs[1]).cpu().numpy())
        q1 = np.concatenate(acc).astype(np.int32)
        stats[name] = check_codes(q1, int(codec.vocab_size),
                                  np.asarray(q0hat[rows]),
                                  np.asarray(ktrue[rows])[:, 1, :])
        stats[name].update(n_rows=int(len(rows)),
                           q1_sha1=arr_sha(q1),
                           rows_sha1=arr_sha(np.asarray(rows, np.int64)))
        out_rows.append(np.asarray(rows, np.int64))
        out_q1.append(q1)
        out_part.append(np.full(len(rows), name, dtype=object))
        print(f"    {name}: {len(rows)} строк, sha меток "
              f"{stats[name]['q1_sha1']}, совпадает с истинной q1 у "
              f"{100 * stats[name]['frac_equal_static']:.1f}% позиций")

    rows_all = np.concatenate(out_rows)
    q1_all = np.concatenate(out_q1)
    part_all = np.concatenate(out_part).astype(str)
    if len(np.unique(rows_all)) != len(rows_all):
        raise SystemExit("строки повторяются между частями")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    npz = a.out + ".npz"
    tmp = npz + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, rows=rows_all, q1=q1_all, part=part_all)
    os.replace(tmp, npz)
    with np.load(npz, allow_pickle=True) as z:
        if not (np.array_equal(z["q1"], q1_all)
                and np.array_equal(z["rows"], rows_all)):
            raise SystemExit(f"{npz}: прочиталось не то, что записано")

    man = dict(
        kind="canonical_q1_targets", target_latent="z_e = codec._encode(action)",
        device=str(dev), dtype="int32", n_rows=int(len(rows_all)),
        parts={k: dict(v) for k, v in stats.items()},
        sampling=sample_meta, split_seed=int(a.split_seed),
        sel_frac=float(a.sel_frac), cache=a.cache,
        cache_meta_sha1=k11a.file_sha1(f"{a.cache}.meta.json"),
        source_cache=src, source_cache_sha1=sha12(src), keys_sha1=keys_now,
        q0hat_sha1=k11a.file_sha1(f"{a.cache}.q0hat.npy"),
        ktrue_sha1=k11a.file_sha1(f"{a.cache}.ktrue.npy"),
        split_sha1=k11a.file_sha1(f"{a.cache}.split.npy"),
        ckpt=a.ckpt, vocab=int(codec.vocab_size),
        codebooks_sha1=got_w["codebooks_sha1"],
        codec_state_sha1=got_w["codec_state_sha1"], decoder_probe=probe_now,
        labels_npz=os.path.basename(npz), labels_sha1=sha12(npz),
        rows_sha1=arr_sha(rows_all), q1_sha1=arr_sha(q1_all),
        note="только q1. Цели q2 строятся позже и отдельно для каждой "
             "обученной головы, от её ФАКТИЧЕСКОГО q1, а не от оракульного",
        code_version=kb.code_version([
            os.path.abspath(__file__),
            os.path.join(here, "depth_rvq_joint12.py"),
            os.path.join(here, "k14a_oracle_cache.py")]),
        script_sha1=sha12(os.path.abspath(__file__)))
    mp = a.out + ".manifest.json"
    tmpm = mp + f".tmp.{os.getpid()}"
    json.dump(man, open(tmpm, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmpm, mp)
    print(f"\n  кэш: {npz} (sha {man['labels_sha1']}), манифест: {mp}")
    print(f"  всего {len(rows_all)} строк, метки int32, канонический режим "
          f"{dev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
