#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15g: анализ готовых раскаток K-15f M1 (без новых раскаток).

Принимает каталог артефактов M1, проверяет их целостность и строит отчёт.
При неполном наборе показывает покрытие и НЕ выносит вердикт; при полном —
вызывает официальный анализатор M1 (k15f_measure_subspace.m1) без
изменений и печатает его решение рядом с диагностикой.

ДИАГНОСТИКА:
  * спасения провалов q0 и вред на диагностических успехах — по задачам и
    по направлениям; learned-only, control-only и общие спасения;
  * фактически применённые поправки: RMS |a − a0| по каналам на
    ИСПОЛНЕННЫХ шагах (с учётом done_step: шаги после завершения среды и
    неисполненная часть последнего чанка не считаются);
  * переключения схвата относительно собственного a0 на исполненных шагах;
  * изменение базиса между соседними вызовами (относительный RMS);
  * геометрия (только learned: у контроля в журнале повёрнутый базис):
      — косинус направления с его PCA-якорем и доля «около границы»
        (cos <= 1/sqrt(1+κ²) + NEAR_BOUND_EPS, порог задан здесь заранее);
      — энергия направления вне ВСЕГО подпространства четырёх якорей
        (ортогональность своему якорю этого не гарантирует);
  * временная структура: DCT-II по 8 шагам — энергия частот направлений
    базиса (нормированные координаты) и фактических поправок (координаты
    действия), энергия временных разностей; проверка корректности
    контроля — на первом вызове (одинаковое состояние) суммарная по
    каналам руки энергия каждой частоты у l<j><s> и r<j><s> совпадает.

ОГРАНИЧЕНИЕ ИНТЕРПРЕТАЦИИ КОНТРОЛЯ. При одинаковых состоянии и
коэффициентах правило поправки схвата у learned и контроля одинаково. Но
после расхождения траекторий состояния различаются, и поздние команды
схвата тоже могут различаться. Разность learned − control нельзя называть
полностью изолированным причинным эффектом руки.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

H = 8
NEAR_BOUND_EPS = 0.02          # зарегистрировано до просмотра геометрии
CONFIG = dict(near_bound_eps=NEAR_BOUND_EPS, horizon=H,
              registered="10.10.2026, до анализа M1")


def dct_matrix(n=H):
    """Ортонормированная DCT-II [n, n]: строки — частоты."""
    k = np.arange(n)[:, None]
    t = np.arange(n)[None, :]
    m = np.cos(np.pi * (t + 0.5) * k / n)
    m[0] *= 1.0 / math.sqrt(n)
    m[1:] *= math.sqrt(2.0 / n)
    return m


def executed_mask(n_calls, done_step, total_steps):
    """[C, B, 8] — исполнен ли шаг t = 8c + k средой b.

    done_step = -1 — среда не завершилась: исполнено до total_steps.
    """
    t = (np.arange(n_calls)[:, None] * H + np.arange(H)[None, :])  # [C, 8]
    end = np.where(done_step >= 0, done_step, total_steps - 1)       # [B]
    return t[:, None, :] <= end[None, :, None]


def load_block(json_path):
    d = json.load(open(json_path))
    z = np.load(os.path.join(os.path.dirname(json_path), d["actions_npz"]),
                allow_pickle=True)
    return d, {k: z[k] for k in z.files}


def block_stats(d, z, label, U=None, rho_f=None):
    """Поправки, схват, изменение базиса, геометрия и спектры одного блока."""
    lvl = z.get(f"k15d_level_{label}")
    a0 = z.get("k15d_a0")
    basis = z.get("k15f_basis")
    if lvl is None or a0 is None:
        return None
    C, B = lvl.shape[:2]
    m = executed_mask(C, np.asarray(z["done_step"]), len(z["actions"]))
    delta = (lvl - a0)                                    # [C, B, 8, 7]
    me = m[..., None]
    n_exec = int(m.sum())
    out = dict(calls=int(C), envs=int(B), exec_steps=n_exec)
    out["sq_by_ch"] = (delta ** 2 * me).sum(axis=(0, 1, 2))
    out["grip_switch"] = int((((lvl[..., 6] > 0) != (a0[..., 6] > 0)) & m)
                             .sum())
    D = dct_matrix()
    # спектр фактических поправок: только полностью исполненные чанки
    full = m.all(axis=-1)                                  # [C, B]
    if full.any():
        dd = delta[full]                                   # [N, 8, 7]
        spec = np.einsum("ft,ntc->nfc", D, dd) ** 2
        out["act_spec_arm"] = spec[..., :6].sum(axis=(0, 2))
        out["act_spec_grip"] = spec[..., 6].sum(axis=0)
        out["act_tdiff"] = float(((dd[:, 1:] - dd[:, :-1]) ** 2).sum())
        out["act_full_chunks"] = int(full.sum())
    if basis is not None:
        called = m[..., 0]                                 # вызов исполнялся
        both = called[1:] & called[:-1]
        if both.any():
            db = basis[1:] - basis[:-1]
            num = (db ** 2).sum(axis=(-1, -2))[both]
            den = (basis[1:] ** 2).sum(axis=(-1, -2))[both]
            out["basis_rel_change"] = np.sqrt(num / np.maximum(den, 1e-12))
        bu = basis.reshape(C, B, 4, H, 7)
        sb = np.einsum("ft,cbjtq->cbjfq", D, bu) ** 2
        out["basis_spec_arm_first"] = sb[0, :, :, :, :6].sum(axis=-1)
        out["basis_spec_arm"] = sb[..., :6].sum(axis=-1)[called].mean(0)
        if U is not None and rho_f is not None:
            u = basis[called] / rho_f[None, :, None]       # [N, 4, 56]
            un = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True),
                                1e-12)
            out["anchor_cos"] = (un * U[None]).sum(-1)     # [N, 4]
            proj = un @ U.T                                # [N, 4, 4]
            out["outside_pca4"] = 1.0 - (proj ** 2).sum(-1)
    return out


def analyze(m1_dir, census, basis_ckpt=None, kappa=1.0):
    import k15f_measure_subspace as kms
    import k15f_policy as kp
    from k15d_behavior import check_actions_npz
    import k15e_measure_scale as ms
    labels = kp.all_labels()
    fails = [tuple(x) for x in census["fails"]]
    diag = [tuple(x) for x in census["diag"]]
    want_blocks = {(int(t), int(b)) for t, bl in census["blocks"].items()
                   for b in bl}
    q0_files = sorted(glob.glob(os.path.join(m1_dir, "q0_t*_i*.json")))
    by = {lab: sorted(glob.glob(os.path.join(m1_dir, f"{lab}_t*_i*.json")))
          for lab in labels}
    technical = check_actions_npz(q0_files) + ms.check_episodes(
        q0_files, seeds=[101])
    coverage = {}
    for lab, files in by.items():
        got = {(json.load(open(p))["task_id"], json.load(open(p))[
            "init_start"]) for p in files}
        coverage[lab] = dict(done=len(got & want_blocks),
                             total=len(want_blocks))
        technical += check_actions_npz(files) + ms.check_episodes(
            files, seeds=[101])
    complete = all(v["done"] == v["total"] for v in coverage.values())

    U = rho_f = None
    if basis_ckpt:
        import torch
        ck = torch.load(basis_ckpt, map_location="cpu", weights_only=False)
        U = ck["stats"]["U"].numpy().astype(np.float64)
        rho_f = (ck["stats"]["rho"].numpy().astype(np.float64)
                 * float(ck["amp_factor"]))
    bound = 1.0 / math.sqrt(1.0 + kappa ** 2)

    def episodes_of(files):
        ep = {}
        for p in files:
            d = json.load(open(p))
            for e in d["episodes"]:
                ep[(int(d["task_id"]), int(e["init_state_id"]))] = e
        return ep
    q0_ep = episodes_of(q0_files)
    res = dict(coverage=coverage, complete=complete, technical=technical,
               config=CONFIG, interpretation_note=(
                   "правило поправки схвата у learned и контроля одинаково "
                   "при одинаковых состоянии и коэффициентах, но после "
                   "расхождения траекторий поздние команды схвата могут "
                   "различаться: разность learned − control не является "
                   "изолированным эффектом руки"))
    per_arm, ep_arm = {}, {}
    agg = {}
    for lab in labels:
        ep_arm[lab] = episodes_of(by[lab])
        e = ep_arm[lab]
        per_arm[lab] = dict(
            rescued=sorted([list(k) for k in fails
                            if k in e and e[k]["success"]]),
            harmed=sorted([list(k) for k in diag
                           if k in e and not e[k]["success"]]),
            fails_covered=sum(k in e for k in fails),
            diag_covered=sum(k in e for k in diag))
        a = dict(sq=np.zeros(7), n=0, gs=0, spec_arm=np.zeros(H),
                 spec_grip=np.zeros(H), tdiff=0.0, chunks=0, brc=[],
                 bspec=np.zeros(H), cos=[], out=[], first={})
        for p in by[lab]:
            d, z = load_block(p)
            st = block_stats(d, z, lab, U if lab.startswith("l") else None,
                             rho_f)
            if st is None:
                technical.append(f"{os.path.basename(p)}: нет журнала "
                                 f"уровней")
                continue
            a["sq"] += st["sq_by_ch"]
            a["n"] += st["exec_steps"]
            a["gs"] += st["grip_switch"]
            if "act_spec_arm" in st:
                a["spec_arm"] += st["act_spec_arm"]
                a["spec_grip"] += st["act_spec_grip"]
                a["tdiff"] += st["act_tdiff"]
                a["chunks"] += st["act_full_chunks"]
            if "basis_rel_change" in st:
                a["brc"].append(st["basis_rel_change"])
                a["bspec"] += st["basis_spec_arm"].mean(0)
                a["first"][(d["task_id"], d["init_start"])] = \
                    st["basis_spec_arm_first"]
            if "anchor_cos" in st:
                a["cos"].append(st["anchor_cos"])
                a["out"].append(st["outside_pca4"])
        agg[lab] = a
        n = max(a["n"], 1)
        summ = dict(exec_steps=a["n"],
                    rms_by_channel=[float(x) for x in np.sqrt(a["sq"] / n)],
                    rms_arm=float(np.sqrt(a["sq"][:6].sum() / (6 * n))),
                    grip_switch_share=a["gs"] / n)
        if a["chunks"]:
            tot = a["spec_arm"].sum()
            summ["act_spec_arm_share"] = [float(x) for x in
                                          a["spec_arm"] / max(tot, 1e-18)]
            summ["act_tdiff_per_chunk"] = a["tdiff"] / a["chunks"]
        if a["brc"]:
            brc = np.concatenate(a["brc"])
            summ["basis_rel_change"] = dict(
                median=float(np.median(brc)), p90=float(np.percentile(
                    brc, 90)), max=float(brc.max()))
            summ["basis_spec_arm_share"] = [float(x) for x in
                                            a["bspec"] / max(a["bspec"]
                                                             .sum(), 1e-18)]
        if a["cos"]:
            cos = np.concatenate(a["cos"])
            out_e = np.concatenate(a["out"])
            summ["anchor_cos"] = dict(
                min=[float(x) for x in cos.min(0)],
                p05=[float(x) for x in np.percentile(cos, 5, axis=0)],
                median=[float(x) for x in np.median(cos, axis=0)],
                near_bound_share=[float(x) for x in
                                  (cos <= bound + NEAR_BOUND_EPS).mean(0)])
            summ["outside_pca4_energy"] = dict(
                median=[float(x) for x in np.median(out_e, axis=0)],
                p90=[float(x) for x in np.percentile(out_e, 90, axis=0)])
        per_arm[lab].update(summ)
    # learned-only / control-only / общие спасения и по направлениям
    learned = [x for x in labels if x.startswith("l")]
    control = [x for x in labels if x.startswith("r")]

    def saved(fam, k):
        return any(ep_arm[x].get(k, {}).get("success") for x in fam)
    L = {k: saved(learned, k) for k in fails}
    Cc = {k: saved(control, k) for k in fails}
    res["rescue"] = dict(
        F=len(fails), learned=sum(L.values()), control=sum(Cc.values()),
        learned_only=sum(L[k] and not Cc[k] for k in fails),
        control_only=sum(Cc[k] and not L[k] for k in fails),
        both=sum(L[k] and Cc[k] for k in fails),
        per_task={str(t): dict(
            F=sum(1 for k in fails if k[0] == t),
            learned=sum(L[k] for k in fails if k[0] == t),
            control=sum(Cc[k] for k in fails if k[0] == t))
            for t in sorted({k[0] for k in fails})})
    res["per_direction"] = {
        f"{j}{s}": dict(
            learned_rescues=len(per_arm[f"l{j}{s}"]["rescued"]),
            control_rescues=len(per_arm[f"r{j}{s}"]["rescued"]),
            learned_harms=len(per_arm[f"l{j}{s}"]["harmed"]),
            control_harms=len(per_arm[f"r{j}{s}"]["harmed"]))
        for j in range(4) for s in ("p", "m")}
    # корректность контроля: на первом вызове (одно состояние) суммарная
    # по руке энергия каждой частоты базиса у l и r совпадает
    gaps = []
    for j in range(4):
        for s in ("p", "m"):
            fl, fr = agg[f"l{j}{s}"]["first"], agg[f"r{j}{s}"]["first"]
            for key in set(fl) & set(fr):
                gaps.append(float(np.abs(fl[key] - fr[key]).max()
                                  / max(np.abs(fl[key]).max(), 1e-12)))
    res["control_spectrum_check"] = dict(
        blocks=len(gaps), max_rel_gap=(max(gaps) if gaps else None),
        passed=(bool(gaps) and max(gaps) < 1e-3) if gaps else None)
    res["q0_episodes"] = len(q0_ep)
    res["per_arm"] = per_arm
    if complete and not technical:
        q0 = [(p, json.load(open(p))) for p in q0_files]
        cand = {lab: [(p, json.load(open(p))) for p in by[lab]]
                for lab in labels}
        off = kms.m1(census, q0, cand)
        res["official_m1"] = dict(decision=off["decision"],
                                  code=off["code"], R_L=off["R_L"],
                                  R_C=off["R_C"],
                                  technical=off["technical"][:5])
    else:
        res["official_m1"] = None
    return res


def report(res):
    cov = res["coverage"]
    done = sum(v["done"] for v in cov.values())
    tot = sum(v["total"] for v in cov.values())
    print(f"  покрытие: {done}/{tot} блоков"
          + ("" if res["complete"] else " — НЕПОЛНО, вердикта нет"))
    r = res["rescue"]
    print(f"  спасения провалов q0 (F = {r['F']}): learned {r['learned']}, "
          f"контроль {r['control']}; только learned {r['learned_only']}, "
          f"только контроль {r['control_only']}, общие {r['both']}")
    print("  по задачам: " + "; ".join(
        f"{t}: F {v['F']}, L {v['learned']}, C {v['control']}"
        for t, v in r["per_task"].items()))
    print("  по направлениям (спасения L/C, вред L/C): " + ", ".join(
        f"{k} {v['learned_rescues']}/{v['control_rescues']},"
        f"{v['learned_harms']}/{v['control_harms']}"
        for k, v in res["per_direction"].items()))
    for lab, v in res["per_arm"].items():
        if not v.get("exec_steps"):
            continue
        s = (f"    {lab}: RMS руки {v['rms_arm']:.4f}, смена схвата "
             f"{100 * v['grip_switch_share']:.2f}%")
        if "basis_rel_change" in v:
            s += (f", изменение базиса медиана "
                  f"{v['basis_rel_change']['median']:.3f}")
        if "anchor_cos" in v:
            s += (f", около границы якоря "
                  f"{[round(x, 3) for x in v['anchor_cos']['near_bound_share']]}"
                  f", вне PCA4 (медиана) "
                  f"{[round(x, 3) for x in v['outside_pca4_energy']['median']]}")
        print(s)
    cc = res["control_spectrum_check"]
    print(f"  проверка контроля по спектру (первый вызов): блоков "
          f"{cc['blocks']}, макс. отн. расхождение {cc['max_rel_gap']}")
    if res["official_m1"]:
        print(f"  ОФИЦИАЛЬНОЕ РЕШЕНИЕ M1: {res['official_m1']['decision']}")
    if res["technical"]:
        print(f"  ТЕХНИЧЕСКИЕ ЗАМЕЧАНИЯ: {res['technical'][:6]}")


def main():
    ap = argparse.ArgumentParser(description="K-15g: анализ M1")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--m1-dir", default=None)
    ap.add_argument("--census", default="reports/k15f/m1_census.json")
    ap.add_argument("--basis", default="data/k15f/basis_s0.pt")
    ap.add_argument("--out", default="reports/k15g/m1_analysis.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    d = a.m1_dir or sorted(glob.glob("reports/k15f/m1/s101/*/"),
                           key=os.path.getmtime)[-1]
    res = analyze(d, json.load(open(a.census)),
                  a.basis if os.path.exists(a.basis) else None)
    res["m1_dir"] = d
    report(res)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1,
                  default=lambda x: x.tolist() if hasattr(x, "tolist")
                  else str(x))
    os.replace(a.out + ".tmp", a.out)
    print(f"  сводка: {a.out}")
    return 0


def selftest():
    D = dct_matrix()
    assert np.allclose(D @ D.T, np.eye(H), atol=1e-12)
    # маска исполнения: среда завершилась на шаге 10 -> вызов 0 целиком,
    # вызов 1 — шаги 8, 9, 10; незавершённая — до конца журнала
    m = executed_mask(3, np.array([10, -1]), 20)
    assert m[0, 0].all() and m[1, 0, :3].all() and not m[1, 0, 3:].any()
    assert not m[2, 0].any()
    assert m[1, 1].all() and m[2, 1, :4].all() and not m[2, 1, 4:].any()
    # спектральная проверка контроля: одна Q 6×6 на всех шагах сохраняет
    # суммарную по руке энергию каждой частоты
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    b = rng.standard_normal((H, 7))
    br = b.copy()
    br[:, :6] = b[:, :6] @ q
    e1 = ((D @ b[:, :6]) ** 2).sum(-1)
    e2 = ((D @ br[:, :6]) ** 2).sum(-1)
    assert np.allclose(e1, e2)
    # КОНТРОЛЬ: перемешивание по времени ломает совпадение спектров
    bt = b.copy()
    bt[:, :6] = b[rng.permutation(H), :6]
    assert not np.allclose(((D @ bt[:, :6]) ** 2).sum(-1), e1)
    # блок: поправка только на исполненных шагах, переключения схвата
    C, B = 3, 2
    a0 = np.full((C, B, H, 7), 0.5, np.float32)
    lvl = a0.copy()
    lvl[..., 0] += 0.1
    lvl[2, 0, :, 6] = -0.5        # не исполнено у среды 0 (done на шаге 10)
    lvl[1, 1, 0, 6] = -0.5        # исполнено у среды 1
    z = dict(k15d_a0=a0, k15d_level_l0p=lvl,
             done_step=np.array([10, -1]), actions=np.zeros((20, B, 7)))
    st = block_stats({}, z, "l0p")
    assert st["exec_steps"] == 11 + 20
    assert st["grip_switch"] == 1, st["grip_switch"]
    assert abs(st["sq_by_ch"][0] - 0.01 * 31) < 1e-5
    # СКВОЗНОЙ analyze() на синтетическом каталоге M1
    import tempfile
    import hashlib
    import k15f_policy as kp
    with tempfile.TemporaryDirectory() as td:
        census = dict(fails=[[8, 0], [8, 1], [9, 6]],
                      diag=[[0, 0], [0, 1], [8, 2]],
                      blocks={"0": [0], "8": [0], "9": [5]})
        Uq = np.linalg.qr(rng.standard_normal((56, 4)))[0].T

        def write(label, t, blk, succ):
            eps, Cn, Bn = [], 6, 5
            for k in range(5):
                s_ = blk + k
                h = hashlib.sha1(f"{label}{t}{s_}".encode()).hexdigest()[:16]
                eps.append(dict(success=bool(succ(label, t, s_)),
                                init_state_id=s_, env_index=k,
                                init_hash=f"i{t}{s_}",
                                init_hash_full=f"I{t}{s_}",
                                rollout_seed=101, action_sha1=h))
            base = os.path.join(td, f"{label}_t{t}_i{blk}")
            arrs = dict(actions=np.zeros((Cn * H, Bn, 7), np.float32),
                        done_step=np.full(Bn, 30),
                        action_sha1=np.asarray([e["action_sha1"]
                                                for e in eps]),
                        init_state_id=np.arange(blk, blk + 5))
            if label != "q0":
                a0_ = rng.uniform(-0.5, 0.5, (Cn, Bn, H, 7)).astype(
                    np.float32)
                arrs["k15d_a0"] = a0_
                arrs[f"k15d_level_{label}"] = a0_ + 0.05
                u_ = Uq[None, None] + 0.1 * rng.standard_normal(
                    (Cn, Bn, 4, 56))
                arrs["k15f_basis"] = (u_ * 3.0).astype(np.float32)
            np.savez(base + ".actions.npz", **arrs)
            d = dict(episodes=eps, arm_label=label, task_id=t,
                     init_start=blk, seed=101, actions_npz=os.path.basename(
                         base) + ".actions.npz")
            d["actions_npz_sha1"] = hashlib.sha1(open(
                base + ".actions.npz", "rb").read()).hexdigest()[:12]
            json.dump(d, open(base + ".json", "w"))

        def succ(label, t, s_):
            if (t, s_) in {(8, 0), (8, 1), (9, 6)}:
                return label == "l0p" and s_ == 0
            return not (label == "r1m" and (t, s_) == (8, 2))
        for lab in ["q0"] + kp.all_labels():
            for t, bl in census["blocks"].items():
                for blk in bl:
                    write(lab, int(t), blk, succ)
        import warnings
        with warnings.catch_warnings():
            # официальный пересчёт переписи K-15f усредняет по задачам, а в
            # синтетике q0 покрывает не все задачи — это не дефект анализа
            warnings.simplefilter("ignore", RuntimeWarning)
            res = analyze(td, census)
        assert res["complete"] and not res["technical"], res["technical"]
        r = res["rescue"]
        assert r["learned"] == 1 and r["control"] == 0
        assert r["learned_only"] == 1 and r["both"] == 0
        assert res["per_direction"]["1m"]["control_harms"] == 1
        assert abs(res["per_arm"]["l0p"]["rms_by_channel"][0] - 0.05) < 1e-4
        # частичный набор: покрытие видно, вердикта нет
        os.remove(os.path.join(td, "l3m_t9_i5.json"))
        res2 = analyze(td, census)
        assert not res2["complete"] and res2["official_m1"] is None
    print("самопроверка k15g_analyze_m1 пройдена: DCT, маска исполнения с "
          "done_step, спектральная проверка контроля, поправки и схват")
    return 0


if __name__ == "__main__":
    sys.exit(main())
