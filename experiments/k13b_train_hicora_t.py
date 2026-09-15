#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-13b: supervised warm start траекторной головы HiCoRA-T.

РЕЦЕПТ НЕ НОВЫЙ. Это дословно принятый рецепт D1 из K-11c с мишенью `coef`:

    T = clamp(c*, -rho, rho) / rho,   loss = smooth_l1(tanh(net(x)), T),

Adam, lr 1e-3, wd 0. Заменены только коэффициенты: там их 16 * rank на чанк
(свои у каждой позиции), здесь rank на чанк (общие для всей траектории).
Ничего другого менять нельзя — иначе разница между HiCoRA и HiCoRA-T окажется
разницей рецептов обучения, а не архитектур.

ВХОД БЕРЁТСЯ ИЗ КЭША K-11a И ПРОХОДИТ ЧЕРЕЗ res_norm. В кэше лежат СЫРЫЕ
отводы fp16, а голова в прогоне видит `res_norm(h24).float()`. Пропустить
норму значило бы обучать голову на входе, которого в прогоне не бывает; эта
ошибка уже один раз обесценила числа K-11i, поэтому здесь она проверяется
прямо — сверкой, что норма действительно изменила вход.

ЧЕРНОВИК — ПРЕДСКАЗАННЫЙ. z0 = E0[q0hat], а не E0[k0_true]: поправка чинит и
ошибку квантования, и ошибку предсказания черновика, как в HiCoRA.
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


def targets_from_coef(coef, rho, eps=1e-8):
    """Мишень: ограниченные коэффициенты, приведённые к [-1, 1].

    Деление на rho обязательно: голова выдаёт tanh, то есть уже [-1, 1], и
    сравнивать его с коэффициентами в исходном масштабе значило бы требовать
    недостижимого. Ограничение тем же rho, что в forward, — иначе мишень
    лежала бы вне досягаемости при |c*| > rho.
    """
    r = np.maximum(np.asarray(rho, np.float32), eps)
    return np.clip(np.asarray(coef, np.float32), -r, r) / r


def split_val(idx, frac=0.5, seed=0):
    """Val делится надвое: отбор эпохи и подтверждение.

    Число, по которому выбрана эпоха, смещено вниз просто потому, что эпоха
    выбиралась по нему. Подтверждение на непересекающейся половине даёт
    честную величину — но и она описательная: решение принимается по rollout.
    """
    r = np.random.default_rng(seed).permutation(len(idx))
    cut = int(len(idx) * frac)
    return np.sort(idx[r[:cut]]), np.sort(idx[r[cut:]])


def batches(n, size):
    for i in range(0, n, size):
        yield i, min(i + size, n)


def build_head(d_hidden, d_latent, n_pos, rank, proj, hidden, B, rho, dev):
    import torch
    from hicora_t_vla import make_trajectory_head
    head = make_trajectory_head()(d_hidden, d_latent, n_pos=n_pos, rank=rank,
                                  proj=proj, hidden=hidden)
    head.set_basis(torch.as_tensor(B))
    head.set_rho(torch.as_tensor(rho))
    return head.to(dev)


def evaluate(head, feed, idx, tgt, batch, dev):
    """Средняя потеря на наборе. Без градиента и без autocast."""
    import torch
    tot, n = 0.0, 0
    head.eval()
    with torch.no_grad():
        for a, b in batches(len(idx), batch):
            s = idx[a:b]
            hn, z0 = feed(s)
            c = torch.tanh(head.mean_coeffs(hn, z0))
            t = torch.from_numpy(tgt[a:b]).to(dev)
            tot += float(torch.nn.functional.smooth_l1_loss(
                c.float(), t, reduction="sum"))
            n += t.numel()
    head.train()
    return tot / max(n, 1)


def selftest():
    import tempfile
    import torch

    # --- мишень: ограничение и приведение к [-1, 1] ------------------------
    rho = np.array([2.0, 0.5], np.float32)
    coef = np.array([[1.0, 0.25], [4.0, -3.0], [-2.0, 0.0]], np.float32)
    T = targets_from_coef(coef, rho)
    assert np.allclose(T[0], [0.5, 0.5]), T[0]
    assert np.allclose(T[1], [1.0, -1.0]), T[1]      # обрезано до предела
    assert np.abs(T).max() <= 1.0 + 1e-6
    # без деления на rho мишень была бы недостижима для tanh
    assert np.abs(np.clip(coef, -rho, rho)).max() > 1.0

    # --- деление val надвое ------------------------------------------------
    idx = np.arange(100)
    a, b = split_val(idx, 0.5, 0)
    assert len(a) == 50 and len(b) == 50
    assert len(np.intersect1d(a, b)) == 0, "половины val пересекаются"
    assert np.array_equal(np.sort(np.concatenate([a, b])), idx)

    # --- соответствие мишени и строк при перемешивании --------------------
    # Ошибка «мишень от другого батча» тихая: потеря считается, обучение идёт,
    # а голова учится сопоставлять случайные пары. Проверяется тождеством
    # порядка: для отсортированного tr сортировка позиций совпадает с
    # сортировкой индексов.
    tr_x = np.array([3, 7, 11, 15, 19])
    T_x = np.arange(5, dtype=np.float32)[:, None]
    ordr = np.array([4, 0, 2])
    pos_sorted = np.sort(ordr)
    assert np.array_equal(tr_x[pos_sorted], np.sort(tr_x[ordr]))
    assert np.array_equal(T_x[pos_sorted].ravel(), np.array([0., 2., 4.]))

    # --- обучение действительно уменьшает потерю --------------------------
    D_H, D_L, NP_, RK, N = 16, 24, 4, 6, 64
    rng = np.random.default_rng(0)
    q, _ = torch.linalg.qr(torch.randn(NP_ * D_L, RK, dtype=torch.float64))
    B = q.T.float().numpy()
    rho_v = (np.abs(rng.normal(size=RK)) + 0.5).astype(np.float32)
    dev = torch.device("cpu")
    head = build_head(D_H, D_L, NP_, RK, 8, 32, B, rho_v, dev)
    H = torch.randn(N, NP_, D_H)
    Z = torch.randn(N, NP_, D_L)
    with torch.no_grad():                # мишень достижима по построению
        true_c = torch.tanh(torch.randn(N, RK) * 0.5).numpy()
    tgt = true_c.astype(np.float32)

    def feed(s):
        return H[s].to(dev), Z[s].to(dev)

    opt = torch.optim.Adam([p for n_, p in head.named_parameters()
                            if n_.startswith(("proj_h.", "proj_z.", "net."))],
                           lr=1e-2)
    idx_all = np.arange(N)
    first = evaluate(head, feed, idx_all, tgt, 32, dev)
    for _ in range(60):
        for a_, b_ in batches(N, 16):
            s = idx_all[a_:b_]
            hn, z0 = feed(s)
            c = torch.tanh(head.mean_coeffs(hn, z0))
            loss = torch.nn.functional.smooth_l1_loss(
                c.float(), torch.from_numpy(tgt[a_:b_]).to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    last = evaluate(head, feed, idx_all, tgt, 32, dev)
    assert last < 0.5 * first, (first, last)

    # --- базис и rho НЕ изменились обучением ------------------------------
    assert torch.equal(head.basis, torch.as_tensor(B)), "базис поехал"
    assert torch.allclose(head.rho, torch.as_tensor(rho_v)), "rho поехала"

    # --- res_norm обязана менять вход: проверка, что её не забыли ---------
    ln = torch.nn.LayerNorm(D_H)
    raw = torch.randn(8, NP_, D_H) * 3.0 + 2.0
    with torch.no_grad():
        assert float((ln(raw) - raw).abs().max()) > 1e-3
    print("самопроверка k13b_train_hicora_t пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--basis", default="data/k13a_traj_basis")
    ap.add_argument("--res-norm-cache", default="data/k11c_res_norm.pt")
    ap.add_argument("--proj", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="data/k13b_hicora_t_s0.pt")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import torch
    import k12b_protocol as kb
    import k13a_build_trajectory_basis as k13a

    bmeta = json.load(open(f"{a.basis}.meta.json"))
    B = np.load(f"{a.basis}.basis.npy")
    rho = np.load(f"{a.basis}.rho.npy")
    if bmeta.get("centered") is not False:
        raise SystemExit("базис центрированный: тождество u=0 -> dZ=0 не "
                         "выполнялось бы, и нулевая инициализация не давала "
                         "бы совпадения с coarse24")
    rank, n_pos, d_lat = bmeta["rank"], bmeta["n_pos"], bmeta["d_latent"]

    k_true = np.load(f"{a.cache}.ktrue.npy", mmap_mode="r")
    q0hat = np.load(f"{a.cache}.q0hat.npy", mmap_mode="r")
    E = np.load(f"{a.cache}.codebooks.npy")
    H24 = np.load(f"{a.cache}.h24.npy", mmap_mode="r")
    cmeta = json.load(open(f"{a.cache}.meta.json"))
    if arr_sha(E) != bmeta["codebooks_sha1"]:
        raise SystemExit("кодовые книги не те, на которых построен базис")
    if arr_sha(np.asarray(q0hat)) != bmeta["q0hat_sha1"]:
        raise SystemExit("черновик q0hat не тот, на котором построен базис: "
                         "целевые коэффициенты относились бы к другому "
                         "остатку")
    n_obs, d_hidden = H24.shape[0], H24.shape[-1]
    if H24.shape[1] != n_pos:
        raise SystemExit(f"в кэше {H24.shape[1]} позиций, базис на {n_pos}")

    idx, _sp = k13a.load_split(f"{a.cache}.split.npy", n_obs)
    tr = idx["train"]
    vsel, vcnf = split_val(idx["val"], 0.5, a.seed)
    if a.limit:
        tr = tr[:a.limit]
        vsel, vcnf = vsel[:a.limit // 4], vcnf[:a.limit // 4]
    print(f"  кэш {a.cache}: {n_obs} наблюдений, d_hidden {d_hidden}, "
          f"позиций {n_pos}, латент {d_lat}")
    print(f"  базис {a.basis}: ранг {rank}, ||rho|| {bmeta['rho_norm']:.4f}, "
          f"объяснено {100 * bmeta['explained']:.2f}%")
    print(f"  train {len(tr)}, val отбор {len(vsel)}, val подтверждение "
          f"{len(vcnf)}")

    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    if not os.path.exists(a.res_norm_cache):
        raise SystemExit(
            f"нет {a.res_norm_cache}: в кэше лежат СЫРЫЕ отводы, и без "
            f"res_norm голова обучалась бы на входе, которого в прогоне не "
            f"бывает")
    res_norm = torch.load(a.res_norm_cache, map_location=dev,
                          weights_only=False).eval()
    for p_ in res_norm.parameters():
        p_.requires_grad_(False)
    rn_sha = hashlib.sha1()
    for k_ in sorted(res_norm.state_dict()):
        rn_sha.update(k_.encode())
        rn_sha.update(np.ascontiguousarray(
            res_norm.state_dict()[k_].detach().float().cpu().numpy()).tobytes())
    rn_sha = rn_sha.hexdigest()[:12]

    Et = torch.from_numpy(E).to(dev)

    def feed(s):
        hb = torch.from_numpy(np.asarray(H24[s])).to(dev)
        with torch.no_grad():
            hn = res_norm(hb.float()).float()
            z0 = Et[0][torch.from_numpy(
                np.asarray(q0hat[s]).astype(np.int64)).to(dev)]
        return hn, z0

    # НОРМА ДЕЙСТВИТЕЛЬНО МЕНЯЕТ ВХОД — проверяется, а не предполагается
    probe = np.asarray(tr[:64])
    hraw = torch.from_numpy(np.asarray(H24[probe])).to(dev).float()
    hn0, _z = feed(probe)
    d_norm = float((hn0 - hraw).abs().max())
    if d_norm < 1e-3:
        raise SystemExit(f"res_norm почти не изменила вход (max|d| = "
                         f"{d_norm:.2e}): вероятно, в кэше уже нормированные "
                         f"отводы, и норма применяется дважды")
    print(f"  res_norm применена: max|h_norm - h_raw| = {d_norm:.3f}, "
          f"sha {rn_sha}")

    # --- целевые коэффициенты ---------------------------------------------
    def coef_for(ix):
        return k13a.coeffs_stream(E, k_true, q0hat, ix, B, 4096, n_pos, d_lat)

    t0 = time.time()
    T_tr = targets_from_coef(coef_for(tr), rho)
    T_sel = targets_from_coef(coef_for(vsel), rho)
    T_cnf = targets_from_coef(coef_for(vcnf), rho)
    sat = float((np.abs(T_tr) >= 1.0 - 1e-6).mean())
    print(f"  мишени готовы за {time.time() - t0:.0f} с; доля координат на "
          f"пределе rho: {100 * sat:.2f}%")

    head = build_head(d_hidden, d_lat, n_pos, rank, a.proj, a.hidden, B, rho,
                      dev)
    train_p = [p for n_, p in head.named_parameters()
               if n_.startswith(("proj_h.", "proj_z.", "net."))]
    frozen = [n_ for n_, p in head.named_parameters() if p not in train_p]
    n_par = sum(p.numel() for p in train_p)
    print(f"  голова: обучаемых параметров {n_par}, заморожено {frozen}")
    opt = torch.optim.Adam(train_p, lr=a.lr, weight_decay=a.wd)

    base_sel = evaluate(head, feed, vsel, T_sel, a.batch, dev)
    print(f"\n  до обучения (поправка тождественно нулевая): потеря на val "
          f"отборе {base_sel:.5f}")

    best = dict(loss=float("inf"), epoch=-1, state=None)
    hist, bad = [], 0
    rng = np.random.default_rng(a.seed)
    for ep in range(a.epochs):
        order = rng.permutation(len(tr))
        run, nb = 0.0, 0
        for i, j in batches(len(tr), a.batch):
            # ПОЗИЦИИ, А НЕ ИНДЕКСЫ: `tr` отсортирован, поэтому сортировка
            # позиций даёт тот же порядок, что сортировка самих индексов, и
            # мишень берётся теми же позициями без поиска по словарю.
            # Словарь на 121 тысячу записей, построенный на каждом батче,
            # стоил бы 57 миллионов операций за эпоху впустую.
            sel = np.sort(order[i:j])
            s = tr[sel]
            t = torch.from_numpy(T_tr[sel]).to(dev)
            hn, z0 = feed(s)
            c = torch.tanh(head.mean_coeffs(hn, z0))
            loss = torch.nn.functional.smooth_l1_loss(c.float(), t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss)
            nb += 1
            if nb % 200 == 0:
                print(f"      эпоха {ep}, батч {nb}/{len(tr) // a.batch + 1}, "
                      f"потеря {run / nb:.5f}", flush=True)
        ev = evaluate(head, feed, vsel, T_sel, a.batch, dev)
        hist.append(dict(epoch=ep, train=run / max(nb, 1), val_sel=ev))
        print(f"    эпоха {ep}: train {run / max(nb, 1):.5f}, val отбор "
              f"{ev:.5f}" + ("  <- лучшая" if ev < best["loss"] else ""))
        if ev < best["loss"]:
            best = dict(loss=ev, epoch=ep,
                        state={k: v.detach().cpu().clone()
                               for k, v in head.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= a.patience:
                print(f"    остановка: {bad} эпох без улучшения")
                break

    if best["state"] is None:
        raise SystemExit("ни одна эпоха не улучшила потерю")
    head.load_state_dict(best["state"])
    cnf = evaluate(head, feed, vcnf, T_cnf, a.batch, dev)
    print(f"\n  выбрана эпоха {best['epoch']}: val отбор {best['loss']:.5f}, "
          f"val подтверждение {cnf:.5f}")
    print(f"    до обучения было {base_sel:.5f} — доля снятой потери "
          f"{100 * (1 - best['loss'] / base_sel):.1f}%")

    ck = dict(
        state={k: v.detach().cpu() for k, v in head.state_dict().items()},
        arch="trajectory_mlp", target="coef", rank=int(rank),
        n_pos=int(n_pos), d_latent=int(d_lat), d_hidden=int(d_hidden),
        proj=int(a.proj), hidden=int(a.hidden),
        basis=a.basis, basis_sha1=bmeta["basis_sha1"],
        rho_sha1=bmeta["rho_sha1"], rho_norm=bmeta["rho_norm"],
        res_norm_sha1=rn_sha, res_norm_cache=a.res_norm_cache,
        cache=a.cache, ckpt=cmeta.get("ckpt"),
        codebooks_sha1=bmeta["codebooks_sha1"],
        q0hat_sha1=bmeta["q0hat_sha1"],
        seed=int(a.seed), selected_epoch=int(best["epoch"]),
        lr=a.lr, wd=a.wd, batch=int(a.batch), epochs_run=len(hist),
        loss="smooth_l1_on_clamped_coef",
        val_sel=best["loss"], val_confirm=cnf, val_sel_before=base_sel,
        target_saturated_frac=sat, history=hist,
        code_version=kb.code_version([
            os.path.abspath(__file__), os.path.join(here, "hicora_t_vla.py"),
            os.path.join(here, "k13a_build_trajectory_basis.py")]),
        script_sha1=sha12(os.path.abspath(__file__)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = f"{a.out}.tmp.{os.getpid()}"
    torch.save(ck, tmp)
    os.replace(tmp, a.out)
    print(f"  сохранено: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
