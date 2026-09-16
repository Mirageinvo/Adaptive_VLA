#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14a: условный оракул RVQ по кэшу K-11a. Без VLA, без раскатки, без GPU.

ВОПРОС. Умеют ли оставшиеся уровни RVQ исправить ОШИБОЧНЫЙ черновик q0? Это
главный вопрос K-14, и он решается до всякого обучения: если книг на это не
хватает, подбирать learning rate поздним головам бессмысленно.

ПОЧЕМУ ЭТО СЧИТАЕТСЯ ПО КЭШУ. В кэше K-11a уже лежат предсказанный q0hat,
истинные коды всех трёх уровней и разбиение. Ствол VLA в вопросе не участвует
вовсе: речь о латентном пространстве кодека. Значит ответ стоит десятки минут
на CPU и не мешает идущей лестнице K-13.

ЧТО СЧИТАЕТСЯ:

    z_q    = E0[k0] + E1[k1] + E2[k2]      что кодек вообще умеет представить
    A0     = decode(E0[q0hat])             нынешний ранний выход
    q1*    = Q1(z_q - E0[q0hat])           условная цель ОТ ПРЕДСКАЗАННОГО q0
    A01*   = decode(E0[q0hat] + E1[q1*])
    q2*    = Q2(z_q - E0[q0hat] - E1[q1*])
    A012*  = decode(... + E2[q2*])
    Acodec = decode(z_q)                   предел, достижимый тремя уровнями

ПОЧЕМУ z_q, А НЕ НЕПРЕРЫВНЫЙ z_e. План допускает оба. z_q чище для этого
вопроса: непрерывный латент содержит и ту часть, которую RVQ не представляет
НИ ПРИ КАКОМ префиксе, и она лишь размывала бы знаменатель `recovery`. Здесь
знаменатель — ровно разрыв от A0 до предела трёх уровней.

ЭТО ОРАКУЛ, А НЕ РЕЗУЛЬТАТ. q1* и q2* вычислены с доступом к истине. Мера
говорит, ЕСТЬ ЛИ что исправлять, и не говорит, сможет ли голова это
предсказать по h18. Между этими вопросами лежит разрыв, который проект уже
измерял: K-13b убрала 19.7% подтверждающей ошибки офлайн, а в раскатке это
дало статистически неопределённую разность.

ГОРИЗОНТ. Робот исполняет первые 8 позиций чанка из 16 (k12d_rollout,
k13c_cell). Улучшение, живущее в позициях 8..15, до него не доходит, поэтому
метрики считаются и по восьми, и по шестнадцати, а gate смотрит на восемь.
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


def err_blocks(a, b, max_act_q, horizon):
    """Ошибка в единицах робота: RMS, MAE и блоки каналов.

    ПОСЛЕДНИЙ КАНАЛ НЕ МАСШТАБИРУЕТСЯ — схват это команда +-1, а не
    физическая величина; в прогоне множитель применяется к `action[..., :-1]`.
    Блоки разделены, потому что одинаковый общий RMS может скрывать перенос
    ошибки из перемещения в поворот.
    """
    d = (np.asarray(a, np.float64) - np.asarray(b, np.float64))[:, :horizon]
    q = np.asarray(max_act_q, np.float64)[:d.shape[-1]].copy()
    q[-1] = 1.0
    d = d * q
    out = dict(rms=float(np.sqrt((d ** 2).mean())),
               mae=float(np.abs(d).mean()),
               rms_trans=float(np.sqrt((d[..., :3] ** 2).mean())),
               rms_rot=float(np.sqrt((d[..., 3:6] ** 2).mean())),
               rms_grip=float(np.sqrt((d[..., 6:7] ** 2).mean())),
               by_channel=[float(x) for x in np.sqrt((d ** 2).mean((0, 1)))],
               by_step=[float(x) for x in np.sqrt((d ** 2).mean((0, 2)))])
    # ПОСТРОЧНАЯ ошибка нужна для доли улучшившихся примеров: среднее по всем
    # строкам может падать за счёт немногих, при том что большинство хуже.
    out["per_row"] = np.sqrt((d ** 2).mean((1, 2)))
    return out


def recovery(e0, e1, e_lim, eps=1e-12):
    """Какую долю разрыва от A0 до предела трёх уровней закрывает поправка.

    ЗНАМЕНАТЕЛЬ ПРОВЕРЯЕТСЯ ЯВНО. При e0 <= e_lim разрыва нет, и отношение
    было бы делением на ноль или отрицательным числом с видом доли. Среднее
    построчных отношений тоже не годится: оно не равно отношению
    агрегированных ошибок и произвольно чувствительно к строкам с малым
    знаменателем.
    """
    gap = float(e0) - float(e_lim)
    if gap <= eps:
        return None
    return (float(e0) - float(e1)) / gap


def gate(res, log=print):
    """Заранее заданный Gate 2 плана K-14. Проверяется на подтверждающей части.

    Пороги взяты из плана и здесь не подбираются: смысл гейта в том, что он
    записан до того, как увидены числа.
    """
    c = res["val_confirm"]
    bad = []
    if not c["A01"]["rms"] < c["A0"]["rms"]:
        bad.append(f"A01* не улучшает RMS-8: {c['A01']['rms']:.5f} против "
                   f"{c['A0']['rms']:.5f}")
    if not res["val_sel"]["A01"]["rms"] < res["val_sel"]["A0"]["rms"]:
        bad.append("A01* не улучшает RMS-8 на отборочной половине")
    for blk in ("rms_trans", "rms_rot", "rms_grip"):
        if c["A012"][blk] > c["A01"][blk] * 1.005:
            bad.append(f"A012* хуже A01* по {blk} более чем на 0.5%: "
                       f"{c['A012'][blk]:.5f} против {c['A01'][blk]:.5f}")
    r = c["recovery_012"]
    if r is None or r < 0.25:
        bad.append(f"совокупное восстановление {r}: меньше 25% разрыва")
    # УЛУЧШЕНИЕ НЕ ДОЛЖНО ЖИТЬ ТОЛЬКО В НЕИСПОЛНЯЕМОМ ХВОСТЕ
    g8 = c["A0"]["rms"] - c["A01"]["rms"]
    g16 = c["A0_full16"]["rms"] - c["A01_full16"]["rms"]
    if g8 <= 0 < g16:
        bad.append("улучшение есть только на всех 16 позициях, а на "
                   "исполняемых восьми его нет")
    for b in bad:
        log(f"    ОТКАЗ: {b}")
    return (not bad), bad


def selftest():
    a = np.zeros((4, 16, 7))
    b = np.zeros((4, 16, 7))
    b[..., 0] = 1.0
    q = np.ones(7)
    e = err_blocks(a, b, q, 8)
    assert abs(e["rms"] - np.sqrt(1.0 / 7)) < 1e-12
    assert abs(e["rms_trans"] - np.sqrt(1.0 / 3)) < 1e-12
    assert e["rms_rot"] == 0.0 and e["rms_grip"] == 0.0
    assert len(e["by_step"]) == 8 and len(e["per_row"]) == 4

    # горизонт отсекает хвост
    c = np.zeros((4, 16, 7))
    c[:, 8:, 0] = 5.0
    assert err_blocks(a, c, q, 8)["rms"] == 0.0
    assert err_blocks(a, c, q, 16)["rms"] > 0.0

    # схват не масштабируется
    g = np.zeros((4, 16, 7))
    g[..., 6] = 1.0
    q2 = np.full(7, 10.0)
    assert abs(err_blocks(a, g, q2, 8)["rms"] - np.sqrt(1.0 / 7)) < 1e-12

    # восстановление: знаменатель проверяется
    assert abs(recovery(1.0, 0.5, 0.0) - 0.5) < 1e-12
    assert recovery(1.0, 0.5, 1.0) is None          # разрыва нет
    assert recovery(1.0, 0.5, 2.0) is None          # предел хуже опоры
    assert recovery(1.0, 1.5, 0.0) < 0              # стало хуже — видно знаком

    # гейт: отказ по каждому условию отдельно
    def mk(a0, a01, a012, rec, a0f=None, a01f=None):
        blk = lambda r: dict(rms=r, rms_trans=r, rms_rot=r, rms_grip=r)
        return dict(val_sel=dict(A0=blk(a0), A01=blk(a01)),
                    val_confirm=dict(A0=blk(a0), A01=blk(a01), A012=blk(a012),
                                     recovery_012=rec,
                                     A0_full16=blk(a0 if a0f is None else a0f),
                                     A01_full16=blk(a01 if a01f is None
                                                    else a01f)))
    ok, _ = gate(mk(1.0, 0.5, 0.4, 0.6), log=lambda *_: None)
    assert ok
    for args_, why in (((1.0, 1.2, 1.1, 0.6), "не улучшает"),
                       ((1.0, 0.5, 0.9, 0.6), "хуже A01"),
                       ((1.0, 0.5, 0.4, 0.1), "меньше 25%"),
                       ((1.0, 0.5, 0.4, None), "меньше 25%")):
        ok, bad = gate(mk(*args_), log=lambda *_: None)
        assert not ok and any(why in x for x in bad), (why, bad)
    # улучшение только в хвосте
    ok, bad = gate(mk(1.0, 1.0, 0.9, 0.6, a0f=1.0, a01f=0.5),
                   log=lambda *_: None)
    assert not ok and any("исполняемых восьми" in x for x in bad), bad
    print("самопроверка k14a_oracle_cache пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--ckpt",
                    default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--n-rows", type=int, default=4096,
                    help="сколько строк каждой части брать; 0 — все")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--sel-frac", type=float, default=0.4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/k14a/oracle_cache.json")
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
    from k11c_train_d1 import split_episodes
    import actioncodec  # noqa: F401
    from utils import ACTION_Q01, ACTION_Q99, VisionLanguageActionProcessor

    dev = torch.device(a.device)
    max_act_q = np.maximum(np.abs(ACTION_Q01), np.abs(ACTION_Q99))

    # --- кэш и происхождение ------------------------------------------------
    meta = json.load(open(f"{a.cache}.meta.json"))
    if meta.get("ckpt") != a.ckpt:
        raise SystemExit(f"кэш собран чекпойнтом {meta.get('ckpt')}, а кодек "
                         f"берётся из {a.ckpt}")
    ktrue = np.load(f"{a.cache}.ktrue.npy", mmap_mode="r")
    q0hat = np.load(f"{a.cache}.q0hat.npy", mmap_mode="r")
    E = np.load(f"{a.cache}.codebooks.npy")
    idx, _ = k13a.load_split(f"{a.cache}.split.npy", q0hat.shape[0])
    # ЭПИЗОДЫ БЕРУТСЯ ИЗ ИСХОДНОГО КЭША K-9a, как это делает K-13b: отдельного
    # .episode.npy у K-11a нет, а делить val по наблюдениям нельзя — кадры
    # одного эпизода сильно зависимы, и «подтверждающая» половина перестала бы
    # быть независимой.
    src = meta.get("cache")
    if not src or not os.path.exists(src):
        raise SystemExit(f"исходный кэш {src} недоступен: без эпизодов val "
                         f"нельзя разделить так же, как в K-11c и K-13b")
    epi = np.asarray(np.load(src, allow_pickle=True)["episode"]).astype(
        np.int64)[:q0hat.shape[0]]

    # --- кодек: только он и нужен -------------------------------------------
    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None))
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь")
    codec = codec.to(dev).eval()
    qs = list(codec.vq.quantizers)
    if len(qs) != 3:
        raise SystemExit(f"уровней {len(qs)}, ожидалось 3")

    with torch.no_grad():
        ii = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        Ecur = torch.stack([q.out_project(q.decode_code(ii))[0]
                            for q in qs]).float()
    dmax = float((Ecur.cpu() - torch.from_numpy(E)).abs().max())
    if dmax > 1e-5:
        raise SystemExit(f"книги разошлись с кэшем на {dmax:.3e}")
    k11a.check_fingerprints(meta, dict(
        codebooks_sha1=hashlib.sha1(np.ascontiguousarray(
            E.astype(np.float32)).tobytes()).hexdigest()[:12],
        decoder_probe=k11a.decoder_probe(codec, Ecur.to(dev), dev),
        codec_state_sha1=k11a.state_sha1(codec)))
    print(f"  кодек сверен с кэшем: книги max|Δ| = {dmax:.3e}, проба и веса "
          f"совпали; уровней {len(qs)}")

    # --- части: train и val, разделённый ПО ЭПИЗОДАМ ------------------------
    # ТА ЖЕ разметка, что в K-11c и K-13b (сид 61): подтверждающая половина
    # должна оставаться той же самой во всех работах, иначе «подтверждение»
    # каждый раз считается на новых данных.
    rng = np.random.default_rng(a.seed)
    dev_idx = idx["dev"]
    sel_eps, cnf_eps = split_episodes(np.asarray(epi[dev_idx]), a.sel_frac,
                                      seed=61)
    e_dev = np.asarray(epi[dev_idx])
    parts = {
        "train": idx["train"],
        "val_sel": dev_idx[np.isin(e_dev, list(sel_eps))],
        "val_confirm": dev_idx[np.isin(e_dev, list(cnf_eps))],
    }
    if a.n_rows:
        parts = {k: np.sort(rng.choice(v, size=min(a.n_rows, len(v)),
                                       replace=False))
                 for k, v in parts.items()}
    print("  части: " + ", ".join(f"{k} {len(v)}" for k, v in parts.items())
          + ". Финальная выборка не читается")

    from depth_rvq_joint12 import code_contribution, nearest_code

    def decode(z, batch=256):
        out = []
        with torch.no_grad():
            for i in range(0, len(z), batch):
                x, _ = codec._decode(z[i:i + batch].float(), embodiment_ids=0)
                out.append(x[..., :7].float().cpu().numpy())
        return np.concatenate(out)

    res, extra = {}, {}
    for name, rows in parts.items():
        k = torch.from_numpy(np.asarray(ktrue[rows]).astype(np.int64)).to(dev)
        q0 = torch.from_numpy(np.asarray(q0hat[rows]).astype(np.int64)).to(dev)
        with torch.no_grad():
            z_q = sum(code_contribution(qs[l], k[:, l, :]) for l in range(3))
            e0 = code_contribution(qs[0], q0)
            q1s = nearest_code(z_q - e0, qs[1])
            e1 = code_contribution(qs[1], q1s)
            q2s = nearest_code(z_q - e0 - e1, qs[2])
            e2 = code_contribution(qs[2], q2s)
        A = dict(A0=decode(e0), A01=decode(e0 + e1), A012=decode(e0 + e1 + e2),
                 Acodec=decode(z_q))
        # СТАТИЧЕСКАЯ ЦЕЛЬ K-8 ДЛЯ СРАВНЕНИЯ: q1 берётся истинный, без учёта
        # того, что q0 предсказан с ошибкой. Если разницы нет, вся идея
        # условной переразметки не нужна, и это надо знать до обучения.
        with torch.no_grad():
            e1_static = code_contribution(qs[1], k[:, 1, :])
        A["A01_static"] = decode(e0 + e1_static)
        atrue = A["Acodec"]
        r = {}
        for nm, arr in A.items():
            r[nm] = err_blocks(arr, atrue, max_act_q, a.horizon)
            r[nm + "_full16"] = err_blocks(arr, atrue, max_act_q, 16)
        r["recovery_01"] = recovery(r["A0"]["rms"], r["A01"]["rms"],
                                    r["Acodec"]["rms"])
        r["recovery_012"] = recovery(r["A0"]["rms"], r["A012"]["rms"],
                                     r["Acodec"]["rms"])
        r["frac_A01_better"] = float(
            (r["A01"]["per_row"] < r["A0"]["per_row"]).mean())
        r["frac_A012_better"] = float(
            (r["A012"]["per_row"] < r["A01"]["per_row"]).mean())
        wrong = (q0 != k[:, 0, :]).any(-1).cpu().numpy()
        r["frac_rows_with_wrong_q0"] = float(wrong.mean())
        for tag, m in (("q0_wrong", wrong), ("q0_right", ~wrong)):
            if m.sum() >= 8:
                r[f"rms_A0_{tag}"] = float(r["A0"]["per_row"][m].mean())
                r[f"rms_A01_{tag}"] = float(r["A01"]["per_row"][m].mean())
        r["dynamic_vs_static_q1_disagree"] = float(
            (q1s != k[:, 1, :]).float().mean())
        r["n_rows"] = int(len(rows))
        for nm in list(r):
            if isinstance(r[nm], dict) and "per_row" in r[nm]:
                extra[f"{name}.{nm}"] = r[nm].pop("per_row")
        res[name] = r
        print(f"\n  === {name}: {len(rows)} строк, строк с ошибочным q0 "
              f"{100 * r['frac_rows_with_wrong_q0']:.1f}% ===")
        for nm in ("A0", "A01", "A012", "A01_static", "Acodec"):
            print(f"    {nm:11s} RMS-8 {r[nm]['rms']:.5f}  "
                  f"(перемещение {r[nm]['rms_trans']:.5f}, поворот "
                  f"{r[nm]['rms_rot']:.5f}, схват {r[nm]['rms_grip']:.5f})")
        print(f"    восстановление: q0->q01 {r['recovery_01']}, "
              f"q0->q012 {r['recovery_012']}")
        print(f"    доля улучшившихся: A01 лучше A0 у "
              f"{100 * r['frac_A01_better']:.1f}%, A012 лучше A01 у "
              f"{100 * r['frac_A012_better']:.1f}%")
        print(f"    динамическая цель q1 отличается от истинной у "
              f"{100 * r['dynamic_vs_static_q1_disagree']:.1f}% позиций")

    print("\n  ГЕЙТ 2 (на подтверждающей половине):")
    ok, bad = gate(res)
    print(f"  РЕШЕНИЕ: {'continue — обучать головы' if ok else 'STOP'}")

    out = dict(parts=res, gate_passed=bool(ok), gate_failures=bad,
               horizon=int(a.horizon), cache=a.cache, ckpt=a.ckpt,
               split_seed=61, sel_frac=float(a.sel_frac),
               target_latent="z_q = sum E_l[k_l] (трёхуровневая опора кодека)",
               codebooks_sha1=meta.get("codebooks_sha1"),
               decoder_probe=meta.get("decoder_probe"),
               codec_state_sha1=meta.get("codec_state_sha1"),
               code_version=kb.code_version([
                   os.path.abspath(__file__),
                   os.path.join(here, "depth_rvq_joint12.py")]),
               script_sha1=sha12(os.path.abspath(__file__)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = f"{a.out}.tmp.{os.getpid()}"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, a.out)
    print(f"  сохранено: {a.out}")
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
