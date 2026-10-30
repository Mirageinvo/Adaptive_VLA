#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-28, пакет A: будущие реальные состояния для строк K-21.

Строки — те же train (A ∪ B) и val_sel подготовки K-21 (без нового
разбиения). Для строки (эпизод e, шаг t): state_seq8[j] = S[t + j], j =
0…20 (наблюдение S[t + j] записано ДО команды A[t + j]), state_valid[j] —
есть ли такое состояние в эпизоде; хвост не заполняется последним
состоянием и не читает соседний эпизод. Полные эпизоды — из того же
dataset repo/revision, что у K-16 (reports/k16/labels.json, SHA разметки
сверен с подготовкой K-21), штатной функцией k16_prepare.read_episode;
изображения не читаются.
Сверки: формы/конечность/frame_index/время (k16_prepare.episode_problems);
S[t] = state8 строки K-21 (1e−5); 20 нормированных экспертных команд
(k16_prepare.codec_norm, масштаб харнесса) = act K-21 на действительных
позициях (1e−5); train и val_sel не пересекаются по эпизодам.
Масштабы по train: σ_p — покоординатный RMS p(t+h) − p(t), h = 1…20, по всем
действительным парам (не меньше 1e−4 м); σ_R — RMS угла вращения
R(t+h) R(t)ᵀ (не меньше 1e−3 рад).
Выход: data/k28/prep.npz, reports/k28/prepare.json, k28_prep_id —
отпечаток контракта (k21 prep_id, SHA массивов и прочитанных parquet,
ревизия датасета, формы, маски, масштабы).
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

J = 21                                   # j = 0…20
H = 20
TOL = 1e-5
OUT = "data/k28/prep.npz"
REPORT = "reports/k28/prepare.json"
K16_JSON = "reports/k16/labels.json"
K16_NPZ = "reports/k16/labels.npz"
SIGMA_P_MIN, SIGMA_R_MIN = 1e-4, 1e-3
CONTRACT_KEYS = ("k21_prep_id", "dataset", "parquet_sha1", "npz_sha1",
                 "shapes", "coverage", "sigma", "rows", "tol")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def assemble(S_by_ep, A_by_ep, episode, step, rows, state8, act, valid,
             scale):
    """-> (state_seq8 [M, 21, 8], state_valid [M, 21], проблемы)."""
    import k16_prepare as k16
    M = len(rows)
    seq = np.zeros((M, J, 8), np.float32)
    ok = np.zeros((M, J), bool)
    prob, ds, da = [], 0.0, 0.0
    for i, r in enumerate(rows):
        e, t = int(episode[r]), int(step[r])
        S, A = S_by_ep[e], A_by_ep[e]
        L = len(S)
        if not 0 <= t < L:
            prob.append(f"строка {r}: шаг {t} вне эпизода {e} длины {L}")
            continue
        n = min(J, L - t)
        seq[i, :n] = S[t:t + n]
        ok[i, :n] = True
        d = float(np.abs(S[t] - np.asarray(state8[r], np.float64)).max())
        ds = max(ds, d)
        if d > TOL:
            prob.append(f"строка {r}: S[t] ≠ state8 K-21 ({d:.2e})")
        na = min(H, L - t)
        a = k16.codec_norm(A[t:t + na], scale)
        v = np.asarray(valid[r])[:na].astype(bool)
        if v.any():
            dd = float(np.abs(a[v] - np.asarray(act[r])[:na][v]).max())
            da = max(da, dd)
            if dd > TOL:
                prob.append(f"строка {r}: команды ≠ act K-21 ({dd:.2e})")
        if len(prob) > 20:
            break
    return seq, ok, prob, dict(state_max_abs=ds, act_max_abs=da)


def sigmas(torch, seq, ok):
    """По train: σ_p [3] и σ_R (см. описание модуля)."""
    import k28_geometry as kg
    G = kg.build(torch)
    s = torch.as_tensor(seq, dtype=torch.float64)
    m = torch.as_tensor(ok)
    p0, R0 = G["pose_of_state8"](s[:, 0])
    dp2, n, ang2 = torch.zeros(3, dtype=torch.float64), 0, 0.0
    for h in range(1, J):
        v = m[:, 0] & m[:, h]
        if not bool(v.any()):
            continue
        ph, Rh = G["pose_of_state8"](s[v, h])
        dp2 += ((ph - p0[v]) ** 2).sum(0)
        w = G["rot_err"](Rh, R0[v])
        ang2 += float((w * w).sum())
        n += int(v.sum())
    sp = np.maximum(np.sqrt((dp2 / max(n, 1)).numpy()), SIGMA_P_MIN)
    sr = max(float(np.sqrt(ang2 / max(n, 1))), SIGMA_R_MIN)
    return dict(sigma_p=sp.tolist(), sigma_R=sr, pairs=n,
                formula="σ_p_c = sqrt(mean_{train rows, h=1..20, valid} "
                        "(p_c(t+h) − p_c(t))²), не меньше 1e−4; σ_R = "
                        "sqrt(mean |Log(R(t+h) R(t)ᵀ)|²), не меньше 1e−3")


def state_at(torch, seq, ok, u):
    """Поза в момент u ∈ [0, 20] (дробный — линейно/геодезически между
    floor и ceil) -> (p, R, действительна). seq [B, 21, 8], ok [B, 21],
    u [B, K]."""
    import k28_geometry as kg
    G = kg.build(torch)
    s = torch.as_tensor(seq)
    dt = torch.promote_types(s.dtype, torch.float32)
    s = s.to(dt)
    u = torch.as_tensor(u, dtype=dt).clamp(0, J - 1)
    lo = torch.floor(u).long()
    hi = torch.ceil(u).long()
    a = (u - lo.to(dt))
    B, K = u.shape
    bi = torch.arange(B)[:, None].expand(B, K)
    sl, sh = s[bi, lo], s[bi, hi]
    okm = torch.as_tensor(ok)
    valid = okm[bi, lo] & okm[bi, hi]
    pL, RL = G["pose_of_state8"](sl)
    pH, RH = G["pose_of_state8"](sh)
    p, R = G["interp"](pL, RL, pH, RH, a)
    return p, R, valid


def contract_id(rep):
    body = {k: rep[k] for k in CONTRACT_KEYS}
    return hashlib.sha1(json.dumps(body, sort_keys=True, default=str)
                        .encode()).hexdigest()[:16]


def prepare():
    import torch
    import k15h_protocol as hpc
    import k16_prepare as k16
    import k21_data as d21
    from huggingface_hub import hf_hub_download
    D = d21.load_prepared(require_teacher=False)
    pr = D["prep_report"]
    if not os.path.exists(K16_NPZ) or sha(K16_NPZ) != \
            pr["inputs"]["k16_labels_sha1"]:
        raise SystemExit("разметка K-16 ≠ закреплённой в подготовке K-21")
    lj = json.load(open(K16_JSON, encoding="utf-8"))
    if lj["out"]["sha1"] != sha(K16_NPZ):
        raise SystemExit("labels.json K-16 не от закреплённой разметки")
    repo, rev = lj["dataset"]["repo"], lj["dataset"]["revision"]
    tr, va = np.asarray(D["train_rows"]), np.asarray(D["val_rows"])
    ep_tr = set(D["episode"][tr].tolist())
    ep_va = set(D["episode"][va].tolist())
    if ep_tr & ep_va:
        raise SystemExit("train и val_sel пересекаются по эпизодам")
    rows = np.r_[tr, va].astype(np.int64)
    split = np.r_[np.zeros(len(tr)), np.ones(len(va))].astype(np.int8)
    S_by, A_by, pq_sha, prob = {}, {}, {}, []
    eps = sorted(ep_tr | ep_va)
    for i, e in enumerate(eps):
        ep = k16.read_episode(repo, rev, e)
        need = D["step"][rows[D["episode"][rows] == e]]
        prob += k16.episode_problems(ep, need, e)
        f = hf_hub_download(repo, f"data/chunk-{int(e) // 1000:03d}/"
                            f"episode_{int(e):06d}.parquet",
                            repo_type="dataset", revision=rev)
        pq_sha[int(e)] = sha(f)
        S_by[int(e)], A_by[int(e)] = ep["S"], ep["A"]
        if (i + 1) % 200 == 0:
            print(f"  прочитано эпизодов {i + 1}/{len(eps)}", flush=True)
    if prob:
        raise SystemExit("эпизоды: " + "; ".join(prob[:5]))
    seq, ok, prob, mx = assemble(S_by, A_by, D["episode"], D["step"], rows,
                                 D["state8"], D["act"], D["valid"],
                                 hpc.harness_scale())
    if prob:
        raise SystemExit("сверка с K-21: " + "; ".join(prob[:5]))
    sig = sigmas(torch, seq[split == 0], ok[split == 0])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT + ".tmp.npz", rows=rows, split=split, state_seq8=seq,
             state_valid=ok)
    os.replace(OUT + ".tmp.npz", OUT)
    task = np.asarray(D["task"])[rows]
    cov = dict(valid_by_j={sp: ok[split == k].mean(0).round(4).tolist()
                           for k, sp in ((0, "train"), (1, "val"))},
               rows_full21={sp: int(ok[split == k].all(1).sum())
                            for k, sp in ((0, "train"), (1, "val"))},
               per_task_full21={str(t): float(ok[task == t].all(1).mean())
                                for t in sorted(set(task.tolist()))})
    rep = dict(k21_prep_id=pr["prep_id"],
               dataset=dict(repo=repo, revision=rev,
                             k16_labels_sha1=sha(K16_NPZ)),
               parquet_sha1=hashlib.sha1(json.dumps(
                   pq_sha, sort_keys=True).encode()).hexdigest(),
               parquet=dict(episodes=len(pq_sha), files=pq_sha),
               npz_sha1=sha(OUT), shapes=dict(state_seq8=list(seq.shape),
                                              state_valid=list(ok.shape)),
               coverage=cov, sigma=sig,
               rows=dict(train=int(len(tr)), val=int(len(va))),
               tol=TOL, checks=mx,
               conventions="S[t+j] — наблюдение ДО команды A[t+j]; хвост "
                           "маскируется, не копируется; позы — "
                           "k28_geometry (Log(R_ref R_nowᵀ))")
    rep["k28_prep_id"] = contract_id(rep)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    json.dump(rep, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(f"  строк train {len(tr)}, val {len(va)}; эпизодов {len(eps)}; "
          f"сверка K-21: |ΔS| ≤ {mx['state_max_abs']:.1e}, |Δact| ≤ "
          f"{mx['act_max_abs']:.1e}")
    print(f"  покрытие j = 0…20 (train): {cov['valid_by_j']['train'][::5]} "
          f"…; строк с полными 21 состояниями train "
          f"{cov['rows_full21']['train']}, val {cov['rows_full21']['val']}")
    print(f"  σ_p {np.round(sig['sigma_p'], 4).tolist()} м, σ_R "
          f"{sig['sigma_R']:.4f} рад ({sig['pairs']} пар); k28_prep_id "
          f"{rep['k28_prep_id']}; {OUT}, {REPORT}")
    return 0


def load(path=OUT, report=REPORT):
    import k21_data as d21
    if not os.path.exists(path) or not os.path.exists(report):
        raise SystemExit("prepare K-28 не выполнен")
    r = json.load(open(report, encoding="utf-8"))
    if sha(path) != r["npz_sha1"] or contract_id(r) != r["k28_prep_id"]:
        raise SystemExit("prep.npz K-28 изменён после подготовки")
    pr = json.load(open(d21.REPORT, encoding="utf-8"))
    if pr["prep_id"] != r["k21_prep_id"]:
        raise SystemExit("K-28 подготовлен для другой подготовки K-21")
    z = dict(np.load(path))
    pos = np.full(int(z["rows"].max()) + 1, -1, np.int64)
    pos[z["rows"]] = np.arange(len(z["rows"]))
    return dict(z, report=r, pos=pos,
                sigma_p=np.asarray(r["sigma"]["sigma_p"], np.float32),
                sigma_R=float(r["sigma"]["sigma_R"]))


def selftest():
    import torch
    import k16_prepare as k16
    rng = np.random.default_rng(0)
    scale = np.ones(7)
    S_by, A_by = {}, {}
    for e, L in ((5, 30), (6, 22)):
        S = np.zeros((L, 8))
        S[:, :3] = np.cumsum(rng.normal(0, 0.01, (L, 3)), 0)
        S[:, 3:6] = np.cumsum(rng.normal(0, 0.02, (L, 3)), 0) + [0.3, 0, 0]
        S_by[e] = S
        A_by[e] = rng.uniform(-0.5, 0.5, (L, 7))
    episode = np.array([5, 5, 6, 6])
    step = np.array([0, 10, 2, 1])            # t+20 == L (6, t=2) — хвост
    rows = np.arange(4)
    state8 = np.stack([S_by[e][t] for e, t in zip(episode, step)])
    act = np.zeros((4, 20, 7))
    valid = np.zeros((4, 20), bool)
    for i, (e, t) in enumerate(zip(episode, step)):
        n = min(20, len(S_by[e]) - t)
        act[i, :n] = k16.codec_norm(A_by[e][t:t + n], scale)
        valid[i, :n] = True
    seq, ok, prob, mx = assemble(S_by, A_by, episode, step, rows, state8,
                                 act, valid, scale)
    assert prob == [], prob
    assert np.allclose(seq[1, 0], S_by[5][10]) and np.allclose(
        seq[1, 19], S_by[5][29])                       # последний в эпизоде
    assert np.allclose(seq[0, 7], S_by[5][7]) and ok[0].all()
    assert ok[2, :20].all() and not ok[2, 20]          # t+20 == L
    assert float(np.abs(seq[2, 20]).sum()) == 0.0      # хвост не копируется
    assert ok[1, :20].all() and not ok[1, 20]          # 10 + 20 = 30 = L
    st2 = state8.copy()
    st2[3, 0] += 1e-3
    assert assemble(S_by, A_by, episode, step, rows, st2, act, valid,
                    scale)[2]
    ac2 = act.copy()
    ac2[0, 3, 2] += 1e-3
    assert assemble(S_by, A_by, episode, step, rows, state8, ac2, valid,
                    scale)[2]
    # σ: только по переданным строкам (train), val не влияет
    s1 = sigmas(torch, seq[:2], ok[:2])
    s2 = sigmas(torch, seq[:2], ok[:2])
    assert s1 == s2 and len(s1["sigma_p"]) == 3 and s1["sigma_R"] > 0
    assert sigmas(torch, seq[:3], ok[:3]) != s1
    # дробное время: целое u — само состояние; дробное — интерполяция;
    # отсутствие соседа — недействительно
    p, R, v = state_at(torch, seq[:1].astype(np.float64), ok[:1],
                       np.array([[0.0, 3.0, 3.5, 20.0]]))
    import k28_geometry as kg
    G = kg.build(torch)
    p3, R3 = G["pose_of_state8"](torch.as_tensor(seq[0, 3], dtype=torch.float64))
    p4, _R4 = G["pose_of_state8"](torch.as_tensor(seq[0, 4], dtype=torch.float64))
    assert torch.allclose(p[0, 1], p3) and torch.allclose(R[0, 1], R3)
    assert torch.allclose(p[0, 2], (p3 + p4) / 2) and bool(v[0].all())
    _p, _R, v2 = state_at(torch, seq[2:3], ok[2:3], np.array([[19.5, 20.0]]))
    assert not bool(v2[0, 0]) and not bool(v2[0, 1])
    # другое представление того же вращения -> та же поза
    w = np.array([0.0, 0.0, 2.0])
    w_alt = w * (1 - 2 * np.pi / np.linalg.norm(w))
    Ra = G["exp_so3"](torch.as_tensor(w))
    Rb = G["exp_so3"](torch.as_tensor(w_alt))
    assert torch.allclose(Ra, Rb, atol=1e-10)
    # k28_prep_id зависит от данных
    rep = {k: 1 for k in CONTRACT_KEYS}
    a = contract_id(rep)
    rep["npz_sha1"] = 2
    assert contract_id(rep) != a
    print("самопроверка k28_data пройдена: пары (S[t+j], A[t+j]) с j = 0 и "
          "последним действительным, хвост t+20 = L маскируется и не "
          "копируется, сверка с state8/act K-21 ловит расхождение 1e−3, σ "
          "только по train, дробное время и отсутствие соседа, "
          "представление вращения, k28_prep_id от данных")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    try:
        sys.exit(prepare())
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            sys.exit(3)
        raise
