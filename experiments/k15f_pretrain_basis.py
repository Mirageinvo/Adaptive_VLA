#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15f: предобучение rank-4 базиса поправок по кэшу h18 (без модели).

Это НЕ обучение по успеху и не результат: голова базиса учится строить
пространство, в котором проекция остатка демонстрации к q0 мала. Успех
впервые измеряет M1.

ЭТАПЫ:
  1. статистика касательного пространства по ВСЕМ строкам train: σ руки,
     σ логита схвата, якоря PCA U [4, 56], амплитуды ρ_j = p90 |<r*, U_j>|;
  2. обучение головы (ridge-реконструкция + ортогональность + нормы +
     якорь), отбор эпохи по реконструкции на val_sel; эпоха 0 — чистая
     PCA (выход головы нулевой) — всегда в истории;
  3. ОТЧЁТ ОБ АМПЛИТУДЕ на val_sel для c = ±e_j learned и контроля:
     RMS поправки (вся, по каналам руки), RMS приращения логита схвата,
     доля сменённых решений схвата, p50/p90/p99 |Δa|, отношение к RMS
     остатка demo − q0 и к средней поправке h18 K-15d, доля |a| > 1.5;
  4. АМПЛИТУДНЫЙ МНОЖИТЕЛЬ — правило, зафиксированное до роллаутов:
     берётся наибольший f из (1.0, 0.75, 0.5), при котором НИ ОДНО
     значение |a| на исполняемых шагах val_sel не превышает 1.5 ни у одного
     ±e_j learned и контроля. Решение — по действиям, без исходов.

Режимы: overfit (32 равномерно выбранные строки train smoke-кэша,
техническая обучаемость), smoke (одна эпоха по smoke-кэшу), full (полный
кэш). Коды: 0 — исправно; 3 — отказ.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k15f_build_cache as kb  # noqa: E402
import k15f_continuous_refine as kf  # noqa: E402

KIND = "k15f_basis"
CONTROL_SEED = 20261008            # зарегистрирован до роллаутов M1
AMP_FACTORS = (1.0, 0.75, 0.5)
ACTION_CLIP_BOUND = 1.5
# средняя |поправка| h18 K-15d на живой раскатке (сводный dev, по выданным
# чанкам): каналы руки и схват
H18_K15D_MEAN_ABS = [0.0048, 0.0022, 0.0082, 0.0033, 0.0057, 0.0018, 0.0405]
OVERFIT = dict(rows=32, steps=300, min_drop=0.5)


def amplitude_report(torch, head, h18, a0, act, R, factor, device,
                     batch=512):
    """Фактическая величина поправок ±e_j learned и контроля на val."""
    K = kf.K_BASIS
    sa, sg = head.sigma_arm, head.sigma_g
    res_rms = None
    acc = {}
    n = len(a0)
    for fam, rot in (("learned", None), ("control", R)):
        for j in range(K):
            for sgn in (1.0, -1.0):
                acc[(fam, j, sgn)] = dict(sq=np.zeros(7), absv=[],
                                          lg=0.0, flip=0, cnt=0, over=0,
                                          absmax=0.0)
    r_sq, r_cnt = np.zeros(7), 0
    with torch.no_grad():
        for s in range(0, n, batch):
            hb = torch.from_numpy(np.asarray(h18[s:s + batch])).to(device)
            ab = torch.from_numpy(a0[s:s + batch]).to(device)
            # a0 из кэша — только исполняемые шаги; хвост не нужен
            a0f = torch.cat([ab, ab[:, -1:].expand(-1, 12, -1)], dim=1)
            tb = torch.from_numpy(act[s:s + batch]).to(device)
            d_arm = (tb[..., :6] - ab[..., :6])
            r_sq[:6] += d_arm.pow(2).sum(dim=(0, 1)).cpu().numpy()
            r_cnt += d_arm.shape[0] * d_arm.shape[1]
            basis, _u = head(hb, a0f)
            basis = basis * factor
            for fam, rot in (("learned", None), ("control", R)):
                bb = basis if rot is None else basis @ rot.to(basis.dtype)
                for j in range(K):
                    for sgn in (1.0, -1.0):
                        c = torch.zeros(K, device=device)
                        c[j] = sgn
                        t = kf.compose(bb, c)
                        a = kf.combine(a0f, t, sa, sg)
                        d = (a[:, :kf.H_EXEC] - a0f[:, :kf.H_EXEC])
                        st = acc[(fam, j, sgn)]
                        st["sq"] += d.pow(2).sum(dim=(0, 1)).cpu().numpy()
                        st["absv"].append(d.abs().reshape(-1, 7).cpu()
                                          .numpy())
                        st["lg"] += float((sg * t[..., 6]).pow(2).sum())
                        g0 = a0f[:, :kf.H_EXEC, 6]
                        st["flip"] += int(((a[:, :kf.H_EXEC, 6] > 0)
                                           != (g0 > 0)).sum())
                        st["cnt"] += d.shape[0] * d.shape[1]
                        aa = a[:, :kf.H_EXEC].abs()
                        st["over"] += int((aa > ACTION_CLIP_BOUND).sum())
                        st["absmax"] = max(st["absmax"], float(aa.max()))
    res_rms = np.sqrt(r_sq[:6] / max(r_cnt, 1))
    out = {}
    for (fam, j, sgn), st in acc.items():
        av = np.concatenate(st["absv"])
        rms = np.sqrt(st["sq"] / max(st["cnt"], 1))
        mean_abs = av.mean(0)
        out[f"{fam}_e{j}{'+' if sgn > 0 else '-'}"] = dict(
            rms_all=float(np.sqrt(st["sq"][:6].sum() / (6 * st["cnt"]))),
            rms_arm=[float(x) for x in rms[:6]],
            rms_grip_logit=float(np.sqrt(st["lg"] / max(st["cnt"], 1))),
            grip_flip_share=st["flip"] / max(st["cnt"], 1),
            p50=[float(x) for x in np.percentile(av, 50, axis=0)],
            p90=[float(x) for x in np.percentile(av, 90, axis=0)],
            p99=[float(x) for x in np.percentile(av, 99, axis=0)],
            ratio_to_demo_residual_rms=float(
                np.sqrt((rms[:6] ** 2).mean())
                / np.sqrt((res_rms ** 2).mean())),
            ratio_to_h18_k15d_mean_abs=float(
                mean_abs[:6].mean() / np.mean(H18_K15D_MEAN_ABS[:6])),
            over_clip=int(st["over"]), absmax=st["absmax"])
    return out, [float(x) for x in res_rms]


def main():
    ap = argparse.ArgumentParser(description="K-15f: предобучение базиса")
    ap.add_argument("--cache", default="data/k15f/h18_cache")
    ap.add_argument("--mode", choices=("overfit", "smoke", "full"),
                    default="full")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import torch
    torch.manual_seed(int(a.seed))
    rng = np.random.default_rng(int(a.seed))
    t0 = time.time()
    # overfit и smoke идут ДО полного кэша — по smoke-кэшу; full — по полному
    cache_path = a.cache + ("" if a.mode == "full" else "_smoke")
    C = kb.load_cache(cache_path, allow_smoke=(a.mode != "full"))
    man = C["manifest"]
    dev = torch.device(a.device)
    tag = f"basis_s{a.seed}" + ("" if a.mode == "full" else f"_{a.mode}")
    out = a.out or f"data/k15f/{tag}.pt"
    report = a.report or f"reports/k15f/{tag}.json"
    tr, va = C["train"], C["val_sel"]
    # --- 1. статистика по ВСЕМ строкам train ------------------------------
    st = kf.tangent_stats(torch.from_numpy(tr["a0"]),
                          torch.from_numpy(tr["act"]))
    print("  касательное: σ руки " + ", ".join(
        f"{x:.4f}" for x in st["sigma_arm"]) + f"; σ логита схвата "
          f"{st['sigma_g']:.3f}; PCA-{kf.K_BASIS} объясняет "
          f"{100 * st['energy_k']:.1f}% энергии; ρ (p90) " + ", ".join(
              f"{x:.3f}" for x in st["rho"]))
    w, eps = C["norm"]
    head = kf.BasisHead(torch.from_numpy(w), eps, man["d_model"],
                        man["n_pos"], st).to(dev)

    def tangent_rows(part, idx):
        a0 = torch.from_numpy(part["a0"][idx]).to(dev)
        act = torch.from_numpy(part["act"][idx]).to(dev)
        r = kf.to_tangent(act, a0, head.sigma_arm, head.sigma_g)
        a0f = torch.cat([a0, a0[:, -1:].expand(-1, 12, -1)], dim=1)
        h = torch.from_numpy(np.asarray(part["h18"][np.sort(idx)])).to(dev)
        order = np.argsort(np.argsort(idx))
        return h[torch.as_tensor(order, device=dev)], a0f, r.reshape(
            len(idx), -1)

    def evaluate(part, n_max=None):
        n = len(part["rows"]) if n_max is None else min(n_max,
                                                         len(part["rows"]))
        tot, cnt, diag_rows = 0.0, 0, []
        with torch.no_grad():
            for s in range(0, n, 512):
                idx = np.arange(s, min(n, s + 512))
                h, a0f, r = tangent_rows(part, idx)
                b, u = head(h, a0f)
                loss, parts = kf.basis_loss(head, b, u, r)
                tot += float(parts["rec"]) * len(idx)
                cnt += len(idx)
                if len(diag_rows) < 8:
                    diag_rows.append((b, u, r))
            b = torch.cat([x[0] for x in diag_rows])
            u = torch.cat([x[1] for x in diag_rows])
            r = torch.cat([x[2] for x in diag_rows])
            d = kf.basis_diagnostics(head, b, u, r)
        d["rec"] = tot / max(cnt, 1)
        return d

    technical = {}
    history, states = [], {}
    if a.mode == "overfit":
        idx_all = np.array(sorted(kb.strided(len(tr["rows"]),
                                             OVERFIT["rows"])))
        opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
        losses = []
        for step in range(OVERFIT["steps"]):
            h, a0f, r = tangent_rows(tr, idx_all)
            b, u = head(h, a0f)
            loss, parts = kf.basis_loss(head, b, u, r)
            loss.backward()
            g = {n_: float(p_.grad.abs().sum()) if p_.grad is not None
                 else 0.0 for n_, p_ in head.named_parameters()}
            opt.step()
            opt.zero_grad()
            losses.append(float(parts["rec"]))
        with torch.no_grad():
            h, a0f, r = tangent_rows(tr, idx_all)
            b, u = head(h, a0f)
            d = kf.basis_diagnostics(head, b, u, r)
        first, last = np.mean(losses[:5]), np.mean(losses[-20:])
        technical.update(
            loss_drop=bool(last <= OVERFIT["min_drop"] * first),
            grads_finite_nonzero=bool(all(np.isfinite(v) and v > 0
                                          for v in g.values())),
            anchor_sign=bool(d["anchor_cos_nonpos"] == 0))
        print(f"  OVERFIT: реконструкция {first:.4f} -> {last:.4f}; "
              f"энергия {d['explained_energy']:.3f}; мин. косинус якоря "
              f"{min(d['anchor_cos_min']):.3f}; градиенты "
              f"{'все ненулевые' if technical['grads_finite_nonzero'] else g}")
        rep = dict(overfit=dict(first=float(first), last=float(last),
                                diag=d))
        sel_tag = None
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=float(a.lr),
                                weight_decay=0.0)
        n = len(tr["rows"])
        epochs = int(a.epochs) if a.mode == "full" else 1

        def snap(ep):
            d = evaluate(va)
            tagx = f"epoch{ep}"
            states[tagx] = {k: v.detach().cpu().clone()
                            for k, v in head.state_dict().items()}
            history.append(dict(tag=tagx, epoch=ep, val=d,
                                state_sha1=kf.state_sha(head)))
            print(f"  [{tagx}] val: реконструкция {d['rec']:.5f}, энергия "
                  f"{d['explained_energy']:.3f}, обусловленность "
                  f"{d['cond_median']:.2f}, мин. косинус якоря "
                  f"{min(d['anchor_cos_min']):.3f}, |ΔU| "
                  f"{d['delta_u_rms']:.4f}", flush=True)
        snap(0)
        for ep in range(1, epochs + 1):
            perm = rng.permutation(n)
            for s in range(0, n, int(a.batch)):
                idx = perm[s:s + int(a.batch)]
                h, a0f, r = tangent_rows(tr, idx)
                b, u = head(h, a0f)
                loss, parts = kf.basis_loss(head, b, u, r)
                if not bool(torch.isfinite(loss)):
                    raise SystemExit("потеря не конечна")
                loss.backward()
                opt.step()
                opt.zero_grad()
            snap(ep)
        # ОТБОР — только среди точек с допустимой геометрией базиса
        # (kf.GEOMETRY, зарегистрировано до данных)
        allowed = [h for h in history if kf.geometry_ok(h["val"])]
        technical["geometry_any_epoch"] = bool(allowed)
        best = min(allowed or history,
                   key=lambda x: (x["val"]["rec"], x["epoch"]))
        sel_tag = best["tag"]
        head.load_state_dict({k: v.to(dev) for k, v in
                              states[sel_tag].items()})
        print(f"  выбрана {sel_tag} по реконструкции val_sel среди точек с "
              f"допустимой геометрией ({len(allowed)} из {len(history)}): "
              f"обусловленность макс {best['val']['cond_max']:.2f}, "
              f"попарный косинус макс {best['val']['pair_cos_max']:.3f}")
        technical["anchor_sign"] = bool(
            best["val"]["anchor_cos_nonpos"] == 0)
        technical["geometry"] = kf.geometry_ok(best["val"])
        rep = dict(history=history, selected=sel_tag)

    # --- 3-4. амплитуда и множитель ----------------------------------------
    R = kf.rotation(CONTROL_SEED).to(dev)
    amp_choice, amp_reports = None, {}
    for f in AMP_FACTORS:
        ar, res_rms = amplitude_report(torch, head, va["h18"], va["a0"],
                                       va["act"], R, f, dev)
        amp_reports[str(f)] = ar
        worst = max(v["absmax"] for v in ar.values())
        over = sum(v["over_clip"] for v in ar.values())
        print(f"  амплитуда f={f}: max|a| {worst:.3f}, значений > "
              f"{ACTION_CLIP_BOUND}: {over}")
        if over == 0 and amp_choice is None:
            amp_choice = f
    technical["amplitude_within_range"] = amp_choice is not None
    if amp_choice is not None:
        ar = amp_reports[str(amp_choice)]
        print(f"  АМПЛИТУДА (f={amp_choice}, нормированные единицы; RMS "
              f"остатка demo − q0 по руке: " + ", ".join(
                  f"{x:.4f}" for x in res_rms) + ")")
        for k in sorted(ar):
            v = ar[k]
            print(f"    {k:12s} RMS {v['rms_all']:.4f} "
                  f"({100 * v['ratio_to_demo_residual_rms']:.0f}% остатка, "
                  f"x{v['ratio_to_h18_k15d_mean_abs']:.1f} к h18 K-15d), "
                  f"логит схвата {v['rms_grip_logit']:.3f}, смена схвата "
                  f"{100 * v['grip_flip_share']:.1f}%, p90 руки "
                  f"{np.mean(v['p90'][:6]):.4f}, max|a| {v['absmax']:.3f}")
    technical_ok = all(technical.values())
    code = 0 if technical_ok else 3
    max_flip = (max(v["grip_flip_share"] for v in
                    amp_reports[str(amp_choice)].values())
                if amp_choice is not None else None)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    obj = dict(
        kind=KIND, mode=a.mode, seed=int(a.seed),
        status="complete", technical_ok=technical_ok,
        state={k: v.detach().cpu().clone()
               for k, v in head.state_dict().items()},
        state_sha1=kf.state_sha(head), hp=dict(head.hp),
        stats={k: (v.cpu() if hasattr(v, "cpu") else v)
               for k, v in st.items()},
        stats_sha1=kf.stats_sha(st), amp_factor=amp_choice,
        max_grip_flip_share=max_flip, geometry_limits=dict(kf.GEOMETRY),
        amp_factors_considered=list(AMP_FACTORS),
        control_seed=CONTROL_SEED, selected=sel_tag,
        d_model=man["d_model"], n_pos=man["n_pos"],
        norm_eps=float(eps),
        cache=os.path.abspath(cache_path),
        cache_manifest_sha1=kb.sha_file(os.path.join(cache_path,
                                                      "manifest.json")),
        cache_frozen_sha1=man["frozen_sha1"], joint_sha1=man["joint_sha1"],
        plan_sha1=man["plan_sha1"],
        code=dict(k15f_continuous_refine=kb.sha_file(kf.__file__),
                  k15f_pretrain_basis=kb.sha_file(os.path.abspath(
                      __file__))),
        created=datetime.datetime.now().isoformat(timespec="seconds"))
    # ПОРЯДОК ПУБЛИКАЦИИ: чекпойнт во временный файл -> отчёт ->
    # канонический чекпойнт последним (его наличие — маркер завершения).
    tmp = out + ".tmp"
    torch.save(obj, tmp)
    rep.update(kind=KIND + "_report", mode=a.mode, technical=technical,
               technical_ok=technical_ok, code=code,
               stats=dict(sigma_arm=[float(x) for x in st["sigma_arm"]],
                          sigma_g=st["sigma_g"],
                          rho=[float(x) for x in st["rho"]],
                          eigvals=[float(x) for x in st["eigvals"]],
                          energy_k=st["energy_k"], rows=st["rows"]),
               stats_sha1=obj["stats_sha1"], amp_factor=amp_choice,
               max_grip_flip_share=max_flip,
               geometry_limits=dict(kf.GEOMETRY),
               amplitude=amp_reports, control_seed=CONTROL_SEED,
               out=out, state_sha1=obj["state_sha1"],
               seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(os.path.abspath(report)), exist_ok=True)
    with open(report + ".tmp", "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False, default=float)
    os.replace(report + ".tmp", report)
    os.replace(tmp, out)
    print(f"ИТОГ базис {a.mode}: {'исправно' if technical_ok else 'ОТКАЗ'} "
          f"{technical}; f={amp_choice}; {out}")
    return code


def selftest():
    """Сквозной прогон на синтетическом кэше: overfit и smoke."""
    import tempfile
    import torch
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "h18_cache_smoke")
        os.makedirs(path)
        rows = {"train": 300, "val_sel": 80}
        for p, n in rows.items():
            r = (np.arange(n) + (0 if p == "train" else 10_000)).astype(
                np.int64)
            np.save(os.path.join(path, f"{p}_rows.npy"), r)
            h = np.lib.format.open_memmap(os.path.join(path, f"{p}_h18.npy"),
                                          mode="w+", dtype=np.float16,
                                          shape=(n, 4, 16))
            h[:] = rng.standard_normal((n, 4, 16))
            h.flush()
            a0 = rng.uniform(-0.8, 0.8, (n, 8, 7)).astype(np.float32)
            act = a0.copy()
            act[..., :6] += 0.05 * rng.standard_normal((n, 8, 6))
            act[..., 6] = np.sign(a0[..., 6])
            np.save(os.path.join(path, f"{p}_q0.npy"),
                    np.zeros((n, 4), np.int64))
            np.save(os.path.join(path, f"{p}_a0.npy"), a0)
            np.save(os.path.join(path, f"{p}_act.npy"), act.astype(
                np.float32))
        np.savez(os.path.join(path, "norm.npz"), weight=np.ones(16), eps=1e-6)
        arrays = {fn: kb.file_sha(os.path.join(path, fn))
                  for fn in os.listdir(path) if fn.endswith((".npy", ".npz"))}
        man = dict(kind=kb.KIND, smoke=True, rows=rows, n_pos=4, d_model=16,
                   h_exec=8, vocab=2048, array_sha1=arrays,
                   rows_sha1={p: hashlib.sha1(np.ascontiguousarray(
                       np.load(os.path.join(path, f"{p}_rows.npy")))
                       .tobytes()).hexdigest()[:12] for p in rows},
                   frozen_sha1="F", joint_sha1="J", plan_sha1="P")
        json.dump(man, open(os.path.join(path, "manifest.json"), "w"))
        open(os.path.join(path, "COMPLETE"), "w").write("x")
        saved = sys.argv
        try:
            for mode in ("overfit", "smoke"):
                sys.argv = ["x", "--cache", os.path.join(td, "h18_cache"),
                            "--mode", mode, "--device", "cpu",
                            "--out", os.path.join(td, f"{mode}.pt"),
                            "--report", os.path.join(td, f"{mode}.json")]
                code = main()
                assert code == 0, (mode, code)
                ck = torch.load(os.path.join(td, f"{mode}.pt"),
                                weights_only=False)
                assert ck["kind"] == KIND and ck["amp_factor"] in AMP_FACTORS
        finally:
            sys.argv = saved
    print("самопроверка k15f_pretrain_basis пройдена: overfit и smoke на "
          "синтетическом кэше, отчёт об амплитуде, выбор множителя")
    return 0


if __name__ == "__main__":
    sys.exit(main())
