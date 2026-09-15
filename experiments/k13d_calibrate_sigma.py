#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-13d: калибровка sigma_T для гауссовой HiCoRA-T.

ЗАЧЕМ. sigma нельзя перенести из HiCoRA: там шум живёт в 16 * 32 = 512
координатах позиционного базиса, здесь — в 64 координатах траекторного. Одно и
то же число даёт разное возмущение действий, а исследование в RL определяется
именно возмущением действий, а не числом в конфигурации.

ЦЕЛЬ. Подобрать sigma_T так, чтобы RMS изменения ДЕКОДИРОВАННЫХ действий
совпал с тем, что даёт нынешняя HiCoRA при sigma = 0.10. Это переносит
рабочую точку, а не число.

ПОЧЕМУ В ПРОСТРАНСТВЕ ДЕЙСТВИЙ. Робот исполняет действия; одинаковое
возмущение латента при разных базисах даёт разное изменение действия, и
сравнивать латенты бессмысленно.

СОСТОЯНИЯ БЕРУТСЯ ИЗ TRAIN. Калибровка — это выбор, и делать его на тех же
наблюдениях, на которых потом измеряется результат, значит выбирать на
проверочной выборке. Dev и final не используются вовсе.

ОДИН НАБОР ШУМА НА ВСЕ sigma. Со свежим шумом на каждом измерении RMS гуляет
от выборки, и двоичный поиск гоняется за собственным шумом вместо величины.
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
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def rms_of(a, b, max_act_q=None):
    """RMS разности действий В ТЕХ ЖЕ ЕДИНИЦАХ, ЧТО ПОЛУЧАЕТ РОБОТ.

    Без масштаба сравнивались бы нормированные величины, а каналы имеют разные
    диапазоны: одно и то же нормированное возмущение означает разный сдвиг
    схвата и разный поворот.

    ПОСЛЕДНИЙ КАНАЛ НЕ МАСШТАБИРУЕТСЯ. В прогоне (k13c_cell, k12d_rollout)
    множитель применяется только к `action[..., :-1]`: схват — это команда
    +-1, а не физическая величина. Масштабировать его здесь значило бы мерить
    не то возмущение, которое исполняется.
    """
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    if max_act_q is not None:
        q = np.asarray(max_act_q, np.float64)[:d.shape[-1]].copy()
        q[-1] = 1.0
        d = d * q
    return float(np.sqrt((d ** 2).mean()))


def selftest():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hicora_t_g import calibrate_sigma, check_monotone, fixed_eps

    # --- RMS: масштаб каналов учитывается -------------------------------
    a = np.zeros((2, 3, 7))
    b = np.zeros((2, 3, 7))
    b[..., 0] = 1.0
    assert abs(rms_of(a, b) - np.sqrt(1.0 / 7)) < 1e-12
    q = np.ones(7)
    q[0] = 10.0
    assert abs(rms_of(a, b, q) - np.sqrt(100.0 / 7)) < 1e-12
    # последний канал (схват) остаётся без масштаба, как в прогоне
    b2 = np.zeros((2, 3, 7))
    b2[..., 6] = 1.0
    q2 = np.full(7, 10.0)
    assert abs(rms_of(a, b2, q2) - np.sqrt(1.0 / 7)) < 1e-12

    # --- один набор шума делает измерение детерминированным -------------
    e1, e2 = fixed_eps(16, 8, 3), fixed_eps(16, 8, 3)
    assert np.array_equal(e1, e2)

    # --- поиск на модели «RMS растёт линейно по sigma» ------------------
    got, r, hist = calibrate_sigma(0.05, lambda s: 0.5 * s,
                                   log=lambda *_: None)
    assert abs(got - 0.1) < 2e-3, got
    assert abs(r - 0.05) <= 0.02 * 0.05
    assert len(hist["grid"]) == 7
    # немонотонность внутри отрезка — отказ
    try:
        check_monotone(lambda s: 0.5 * s * (0.2 if 0.01 < s < 0.1 else 1.0),
                       log=lambda *_: None)
    except SystemExit as e:
        assert "не монотонен" in str(e), e
    else:
        raise AssertionError("немонотонность пропущена")
    print("самопроверка k13d_calibrate_sigma пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--basis-t", default="data/k13a_traj_basis")
    ap.add_argument("--head-t", default="data/k13b_hicora_t_s0.pt")
    ap.add_argument("--head-d1", default="data/k11d/d1_mlp_coef_0.001_wd0_s0.pt")
    ap.add_argument("--res-norm-cache", default="data/k11c_res_norm.pt")
    ap.add_argument("--ckpt", default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--sigma-ref", type=float, default=0.10)
    ap.add_argument("--n-rows", type=int, default=2048)
    ap.add_argument("--n-eps", type=int, default=8,
                    help="сколько реализаций шума усредняется на строку")
    ap.add_argument("--lo", type=float, default=1e-3)
    ap.add_argument("--hi", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="data/k13d_sigma_t.json")
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
    import k13a_build_trajectory_basis as k13a
    from hicora_t_g import (calibrate_sigma, fixed_eps, make_gaussian_trajectory_head)
    import hicora_g as hg
    import actioncodec  # noqa: F401
    from utils import ACTION_Q01, ACTION_Q99, VisionLanguageActionProcessor

    dev = torch.device(a.device)
    max_act_q = np.maximum(np.abs(ACTION_Q01), np.abs(ACTION_Q99))

    # --- данные: СОСТОЯНИЯ ТОЛЬКО ИЗ TRAIN ---------------------------------
    k_true = np.load(f"{a.cache}.ktrue.npy", mmap_mode="r")
    q0hat = np.load(f"{a.cache}.q0hat.npy", mmap_mode="r")
    E = np.load(f"{a.cache}.codebooks.npy")
    H24 = np.load(f"{a.cache}.h24.npy", mmap_mode="r")
    cmeta = json.load(open(f"{a.cache}.meta.json"))
    idx, _sp = k13a.load_split(f"{a.cache}.split.npy", H24.shape[0])
    rng = np.random.default_rng(a.seed)
    rows = np.sort(rng.choice(idx["train"],
                              size=min(a.n_rows, len(idx["train"])),
                              replace=False))
    print(f"  калибровка на {len(rows)} строках TRAIN (dev и final не "
          f"используются)")

    if not os.path.exists(a.res_norm_cache):
        raise SystemExit(f"нет {a.res_norm_cache}: в кэше лежат СЫРЫЕ отводы, "
                         f"и без res_norm голова получала бы вход, которого в "
                         f"прогоне не бывает")
    res_norm = torch.load(a.res_norm_cache, map_location=dev,
                          weights_only=False).eval()
    _rn = hashlib.sha1()
    for k_ in sorted(res_norm.state_dict()):
        _rn.update(k_.encode())
        _rn.update(np.ascontiguousarray(res_norm.state_dict()[k_]
                                        .detach().float().cpu().numpy()
                                        ).tobytes())
    rn_sha = _rn.hexdigest()[:12]
    rn_dtype = next(res_norm.parameters()).dtype
    Et = torch.from_numpy(E).to(dev)
    hb = torch.from_numpy(np.asarray(H24[rows])).to(dev, rn_dtype)
    with torch.no_grad():
        h24 = res_norm(hb).float()
        z0 = Et[0][torch.from_numpy(
            np.asarray(q0hat[rows]).astype(np.int64)).to(dev)].float()
    del hb

    # --- декодер ------------------------------------------------------------
    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = ac if hasattr(ac, "vq") else getattr(ac, "codec", None)
    codec = codec.to(dev).eval()

    def decode(z, batch=256):
        out = []
        with torch.no_grad():
            for i in range(0, len(z), batch):
                x, _ = codec._decode(z[i:i + batch].float(), embodiment_ids=0)
                out.append(x[..., :7].float().cpu().numpy())
        return np.concatenate(out)

    # --- ОПОРА: нынешняя HiCoRA при sigma_ref -------------------------------
    d1 = torch.load(a.head_d1, map_location="cpu", weights_only=False)
    # ОБЕ ГОЛОВЫ ОБЯЗАНЫ БЫТЬ ОБУЧЕНЫ НА ТОЙ ЖЕ res_norm. Иначе они видят
    # разный вход, и «одинаковое возмущение действий» сравнивало бы не то.
    if d1["res_norm_sha1"] != rn_sha:
        raise SystemExit(f"HiCoRA: res_norm sha {rn_sha}, голова обучена на "
                         f"{d1['res_norm_sha1']}")
    pref = d1["cache"]
    B_d1 = np.load(pref + ".basis.npy").astype(np.float32)
    rho_d1 = np.load(pref + ".rho.npy").astype(np.float32)
    d_h = int(h24.shape[-1])
    gd = hg.make_gaussian_residual_head()(
        d_h, int(E.shape[-1]), rank=int(d1["rank"]),
        hidden=int(d1.get("hidden", 512)), proj=int(d1.get("proj", 64))).to(dev)
    gd.set_basis(torch.as_tensor(B_d1).to(dev))
    gd.set_rho(torch.as_tensor(rho_d1).to(dev))
    st = {k[len("hicora_head."):]: v for k, v in d1["state"].items()}
    with torch.no_grad():
        for k, v in st.items():
            gd.state_dict()[k].copy_(v.to(dev, torch.float32))
    gd.eval()

    n_pos = int(h24.shape[1])
    eps_d1 = torch.from_numpy(
        np.random.default_rng(a.seed + 1).normal(
            size=(a.n_eps, len(rows), n_pos, int(d1["rank"]))
        ).astype(np.float32)).to(dev)
    with torch.no_grad():
        mu_d1 = gd.mean_coeffs(h24, z0)
        a_mean_d1 = decode(z0 + gd(h24, z0, u=mu_d1)["dz"])
    ref_vals = []
    for i in range(a.n_eps):
        with torch.no_grad():
            o = gd(h24, z0, u=mu_d1 + a.sigma_ref * eps_d1[i])
            ref_vals.append(rms_of(decode(z0 + o["dz"]), a_mean_d1, max_act_q))
    target = float(np.mean(ref_vals))
    print(f"  опора: HiCoRA (ранг {d1['rank']}, {n_pos} позиций) при sigma="
          f"{a.sigma_ref}: RMS изменения действий {target:.5f} "
          f"(разброс по {a.n_eps} реализациям {np.std(ref_vals):.5f})")

    # --- HiCoRA-T -----------------------------------------------------------
    t_obj = torch.load(a.head_t, map_location="cpu", weights_only=False)
    if t_obj["res_norm_sha1"] != rn_sha:
        raise SystemExit(f"HiCoRA-T: res_norm sha {rn_sha}, голова обучена на "
                         f"{t_obj['res_norm_sha1']}")
    bmeta = json.load(open(f"{a.basis_t}.meta.json"))
    B_t = np.load(f"{a.basis_t}.basis.npy")
    rho_t = np.load(f"{a.basis_t}.rho.npy")
    for nm, arr, want in (("базис", B_t, t_obj["basis_sha1"]),
                          ("rho", rho_t, t_obj["rho_sha1"])):
        if arr_sha(arr) != want:
            raise SystemExit(f"{nm} на диске {arr_sha(arr)}, голова обучена "
                             f"на {want}")
    gt = make_gaussian_trajectory_head()(
        d_h, int(t_obj["d_latent"]), n_pos=int(t_obj["n_pos"]),
        rank=int(t_obj["rank"]), proj=int(t_obj["proj"]),
        hidden=int(t_obj["hidden"])).to(dev)
    gt.set_basis(torch.as_tensor(B_t).to(dev))
    gt.set_rho(torch.as_tensor(rho_t).to(dev))
    with torch.no_grad():
        for k, v in t_obj["state"].items():
            if k in ("basis", "rho", "basis_set", "rho_set"):
                continue
            gt.state_dict()[k].copy_(v.to(dev, torch.float32))
    gt.freeze_log_std().eval()

    # СРЕДНЕЕ ОБЯЗАНО СОВПАСТЬ С D1-ВЕРСИЕЙ ГОЛОВЫ: гауссова голова при
    # deterministic=True — это ровно та же поправка, что исполнялась в K-13c
    import hicora_t_vla as ht
    det = ht.make_trajectory_head()(
        d_h, int(t_obj["d_latent"]), n_pos=int(t_obj["n_pos"]),
        rank=int(t_obj["rank"]), proj=int(t_obj["proj"]),
        hidden=int(t_obj["hidden"])).to(dev)
    det.set_basis(torch.as_tensor(B_t).to(dev))
    det.set_rho(torch.as_tensor(rho_t).to(dev))
    with torch.no_grad():
        for k, v in t_obj["state"].items():
            if k in ("basis", "rho", "basis_set", "rho_set"):
                continue
            det.state_dict()[k].copy_(v.to(dev, torch.float32))
        det.eval()
        dz_det, _c = det(h24, z0)
        mu_t = gt.mean_coeffs(h24, z0)
        dz_g = gt(h24, z0, u=mu_t)["dz"]
    d_head = float((dz_det - dz_g).abs().max())
    if d_head > 1e-5:
        raise SystemExit(f"среднее гауссовой головы расходится с "
                         f"детерминированной на {d_head:.2e}: RL стартовал бы "
                         f"не из проверенной точки")
    print(f"  среднее гауссовой головы совпало с детерминированной: "
          f"{d_head:.2e}")

    a_mean_t = decode(z0 + dz_g)
    eps_t = torch.from_numpy(
        fixed_eps(a.n_eps * len(rows), int(t_obj["rank"]), a.seed + 2)
    ).to(dev).reshape(a.n_eps, len(rows), int(t_obj["rank"]))

    def measure(sig):
        vals = []
        for i in range(a.n_eps):
            with torch.no_grad():
                o = gt(h24, z0, u=mu_t + float(sig) * eps_t[i])
                vals.append(rms_of(decode(z0 + o["dz"]), a_mean_t, max_act_q))
        return float(np.mean(vals))

    t0 = time.time()
    sigma_t, rms_t, hist = calibrate_sigma(target, measure, a.lo, a.hi, a.tol)
    print(f"\n  sigma_T = {sigma_t:.5f} даёт RMS {rms_t:.5f} против цели "
          f"{target:.5f} ({time.time() - t0:.0f} с)")
    print(f"    для сравнения: та же sigma, что у HiCoRA ({a.sigma_ref}), "
          f"дала бы RMS {measure(a.sigma_ref):.5f}")

    out = dict(sigma_t=sigma_t, rms_t=rms_t, target_rms=target,
               sigma_ref=a.sigma_ref, ref_spread=float(np.std(ref_vals)),
               rms_at_sigma_ref=measure(a.sigma_ref),
               n_rows=int(len(rows)), n_eps=int(a.n_eps), split="train",
               res_norm_sha1=rn_sha,
               rank_t=int(t_obj["rank"]), rank_d1=int(d1["rank"]),
               n_pos=n_pos, mean_vs_det=d_head,
               head_t=a.head_t, head_t_sha1=sha12(a.head_t),
               head_d1=a.head_d1, head_d1_sha1=sha12(a.head_d1),
               basis_t=a.basis_t, basis_t_sha1=bmeta["basis_sha1"],
               rho_t_sha1=bmeta["rho_sha1"], cache=a.cache,
               ckpt=a.ckpt, seed=int(a.seed), history=hist,
               code_version=kb.code_version([
                   os.path.abspath(__file__),
                   os.path.join(here, "hicora_t_g.py"),
                   os.path.join(here, "hicora_t_vla.py")]),
               script_sha1=sha12(os.path.abspath(__file__)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = f"{a.out}.tmp.{os.getpid()}"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, a.out)
    print(f"  сохранено: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
