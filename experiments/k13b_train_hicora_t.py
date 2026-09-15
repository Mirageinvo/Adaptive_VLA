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


SPLIT_SEED = 61          # тот же, что в K-11c: разбиение val одно на все головы


def split_val_by_episode(idx, epi, frac=0.4, seed=SPLIT_SEED):
    """Val делится ПО ЭПИЗОДАМ, а не по наблюдениям.

    Кадры одного эпизода сильно зависимы; разделив их по наблюдениям, мы
    получили бы одни и те же эпизоды в обеих половинах, и «подтверждающая»
    половина не была бы независимой.

    СИД ФИКСИРОВАН И НЕ РАВЕН СИДУ ГОЛОВЫ: разбиение обязано совпадать у s0 и
    s1, иначе их числа считаются на разных наборах и попарно не сравнимы.
    Меняется только инициализация головы.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from k11c_train_d1 import split_episodes
    sel_eps, hold_eps = split_episodes(epi[idx], frac, seed=seed)
    sel = idx[np.isin(epi[idx], list(sel_eps))]
    hold = idx[np.isin(epi[idx], list(hold_eps))]
    return np.sort(sel), np.sort(hold), len(sel_eps), len(hold_eps)


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

    # --- деление val ПО ЭПИЗОДАМ -------------------------------------------
    # По четыре кадра на эпизод: если делить по наблюдениям, эпизоды окажутся
    # в обеих половинах, и подтверждение перестанет быть независимым.
    idx = np.arange(160)
    epi = np.repeat(np.arange(40), 4)
    sel, hold, n_es, n_ec = split_val_by_episode(idx, epi, 0.4)
    assert len(np.intersect1d(sel, hold)) == 0, "половины пересекаются"
    assert len(sel) + len(hold) == len(idx)
    assert n_es + n_ec == 40 and min(n_es, n_ec) >= 8, (n_es, n_ec)
    e_sel, e_hold = set(epi[sel].tolist()), set(epi[hold].tolist())
    assert not (e_sel & e_hold), "эпизод попал в обе половины"
    # разбиение НЕ зависит от сида головы: у s0 и s1 оно обязано совпасть
    sel2, hold2, _, _ = split_val_by_episode(idx, epi, 0.4)
    assert np.array_equal(sel, sel2) and np.array_equal(hold, hold2)

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
    ap.add_argument("--seed", type=int, default=0,
                    help="сид ИНИЦИАЛИЗАЦИИ ГОЛОВЫ; разбиение val от него не "
                         "зависит и одинаково у всех голов")
    ap.add_argument("--sel-frac", type=float, default=0.4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-stamp", action="store_true",
                    help="пропустить заверение входа K-11b (только для "
                         "отладки: h24 останется непроверенным)")
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
    # СВЕРКА ФАКТИЧЕСКИ ЗАГРУЖЕННЫХ МАССИВОВ С ИХ ЖЕ ОТПЕЧАТКАМИ. Файл на
    # диске мог быть перезаписан другим прогоном после того, как meta была
    # записана, и тогда голова обучалась бы под один базис, а исполнялась под
    # другой.
    for nm, arr, want in (("базис", B, bmeta["basis_sha1"]),
                          ("rho", rho, bmeta["rho_sha1"])):
        if arr_sha(arr) != want:
            raise SystemExit(f"{nm} на диске имеет sha {arr_sha(arr)}, а в "
                             f"meta базиса {want}")
    if bmeta.get("centered") is not False:
        raise SystemExit(
            "базис центрированный. Тождество u=0 -> dZ=0 он сам по себе не "
            "нарушает; он задаёт ДРУГОЕ подпространство, то есть другой "
            "эксперимент, и смешивать его с уже построенным нельзя")
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
    for nm, path, want in (("ktrue", f"{a.cache}.ktrue.npy",
                            bmeta.get("ktrue_sha1")),
                           ("split", f"{a.cache}.split.npy",
                            bmeta.get("split_sha1"))):
        if want and sha12(path) != want:
            raise SystemExit(f"{nm} изменился после построения базиса: "
                             f"{sha12(path)} против {want}")
    n_obs, d_hidden = H24.shape[0], H24.shape[-1]
    if H24.shape[1] != n_pos:
        raise SystemExit(f"в кэше {H24.shape[1]} позиций, базис на {n_pos}")

    idx, _sp = k13a.load_split(f"{a.cache}.split.npy", n_obs)
    tr, va = idx["train"], idx["val"]
    # ЭПИЗОДЫ ИЗ ИСХОДНОГО КЭША: в производном их нет, а делить val надо по
    # ним, иначе кадры одного эпизода попадут в обе половины
    src = cmeta.get("cache")
    if not src or not os.path.exists(src):
        raise SystemExit(f"исходный кэш {src} недоступен: без эпизодов val "
                         f"нельзя разделить честно")
    epi = np.asarray(np.load(src, allow_pickle=True)["episode"]).astype(
        np.int64)[:n_obs]
    vsel, vcnf, n_es, n_ec = split_val_by_episode(va, epi, a.sel_frac)
    print(f"  val разделён ПО ЭПИЗОДАМ (сид {SPLIT_SEED}): {n_es} эпизодов "
          f"({len(vsel)} набл.) на выбор эпохи, {n_ec} эпизодов "
          f"({len(vcnf)} набл.) на подтверждение")
    if a.limit:
        tr = tr[:a.limit]
    print(f"  кэш {a.cache}: {n_obs} наблюдений, d_hidden {d_hidden}, "
          f"позиций {n_pos}, латент {d_lat}")
    print(f"  базис {a.basis}: ранг {rank}, ||rho|| {bmeta['rho_norm']:.4f}, "
          f"объяснено {100 * bmeta['explained']:.2f}%")
    print(f"  train {len(tr)}")

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

    # ТОТ ЖЕ dtype, ЧТО В ПРОГОНЕ: там norm получает h24 в fp16 и результат
    # приводится к float. Подача fp32 на вход дала бы другой результат нормы,
    # то есть обучение на входе, которого при исполнении не бывает.
    rn_dtype = next(res_norm.parameters()).dtype

    def feed(s):
        hb = torch.from_numpy(np.asarray(H24[s])).to(dev, rn_dtype)
        with torch.no_grad():
            hn = res_norm(hb).float()
            z0 = Et[0][torch.from_numpy(
                np.asarray(q0hat[s]).astype(np.int64)).to(dev)]
        return hn, z0

    # НОРМА ДЕЙСТВИТЕЛЬНО МЕНЯЕТ ВХОД — проверяется, а не предполагается
    probe = np.asarray(tr[:64])
    hraw = torch.from_numpy(np.asarray(H24[probe])).to(dev, rn_dtype).float()
    hn0, _z = feed(probe)
    d_norm = float((hn0 - hraw).abs().max())
    if d_norm < 1e-3:
        raise SystemExit(f"res_norm почти не изменила вход (max|d| = "
                         f"{d_norm:.2e}): вероятно, в кэше уже нормированные "
                         f"отводы, и норма применяется дважды")
    print(f"  res_norm применена: max|h_norm - h_raw| = {d_norm:.3f}, "
          f"sha {rn_sha}")

    # ЗАВЕРЕНИЕ ВХОДА K-11b. Проверка «норма что-то изменила» показывает лишь,
    # что преобразование не тождественно; что загружена ПРАВИЛЬНАЯ res_norm и
    # тот самый h24, подтверждает только отпечаток K-11b, снятый сверкой с
    # живым проходом. Базис и rho там относятся к ЛОКАЛЬНОМУ базису K-11a —
    # это другой артефакт, и наш траекторный сверяется своей meta выше.
    if not a.no_stamp:
        import k11a_build_hicora_cache as k11a
        import k11p_residual_probe as k11p
        stamp_p = f"{a.cache}.artifacts.json"
        if not os.path.exists(stamp_p):
            raise SystemExit(
                f"нет {stamp_p}: правильность позднего входа не подтверждена "
                f"(K-11b). Запустите его или, если это осознанно, укажите "
                f"--no-stamp — тогда обучение пойдёт на непроверенном h24")
        stamp = json.load(open(stamp_p))
        tap = max(cmeta["saved_taps"])
        ash = {f"h{tap}": k11a.file_sha1(f"{a.cache}.h{tap}.npy")}
        for nm in ("q0hat", "ktrue", "split", "codebooks"):
            p_ = f"{a.cache}.{nm}.npy"
            if os.path.exists(p_):
                ash[nm] = k11a.file_sha1(p_)
        k11p.check_stamp(stamp, ash, rn_sha, tap,
                         k11a.file_sha1(f"{a.cache}.meta.json"),
                         k11a.file_sha1(f"{a.cache}.basis.npy"),
                         k11a.file_sha1(f"{a.cache}.rho.npy"),
                         stamp.get("script_sha1"))
        print(f"  вход заверён K-11b ({stamp['script_sha1']}): отпечатки "
              f"{len(ash)} массивов и res_norm совпали")

    # --- целевые коэффициенты ---------------------------------------------
    # КОЭФФИЦИЕНТЫ БЕРУТСЯ ИЗ АРТЕФАКТА K-13a, а не считаются заново: это то
    # же произведение на 121 тысяче строк по 8192 измерения, и повторять его
    # для каждой головы незачем. Порядок строк тот же — оба скрипта берут
    # индексы одной и той же load_split, — и это проверяется по длине.
    t0 = time.time()
    cf_tr = np.load(f"{a.basis}.coef_train.npy")
    cf_va = np.load(f"{a.basis}.coef_val.npy")
    if len(cf_tr) != len(idx["train"]) or len(cf_va) != len(va):
        raise SystemExit(
            f"коэффициенты не соответствуют split: train {len(cf_tr)} против "
            f"{len(idx['train'])}, val {len(cf_va)} против {len(va)}")
    if cf_tr.shape[1] != rank:
        raise SystemExit(f"коэффициенты ранга {cf_tr.shape[1]}, базис {rank}")
    # ВЫБОРОЧНАЯ СВЕРКА С ПЕРЕСЧЁТОМ: равенство длин не доказывает, что
    # порядок строк тот же
    chk = np.asarray(idx["train"][:64])
    ref = k13a.coeffs_stream(E, k_true, q0hat, chk, B, 64, n_pos, d_lat)
    d_coef = float(np.abs(ref - cf_tr[:64]).max())
    if d_coef > 1e-3:
        raise SystemExit(f"сохранённые коэффициенты расходятся с пересчётом "
                         f"на {d_coef:.3e}: порядок строк не тот")
    pos_tr = {int(v): i for i, v in enumerate(idx["train"])}
    pos_va = {int(v): i for i, v in enumerate(va)}
    T_tr = targets_from_coef(cf_tr[[pos_tr[int(v)] for v in tr]], rho)
    T_sel = targets_from_coef(cf_va[[pos_va[int(v)] for v in vsel]], rho)
    T_cnf = targets_from_coef(cf_va[[pos_va[int(v)] for v in vcnf]], rho)
    print(f"  коэффициенты прочитаны из {a.basis}.coef_*.npy, сверка с "
          f"пересчётом: max|d| = {d_coef:.2e}")
    sat = float((np.abs(T_tr) >= 1.0 - 1e-6).mean())
    print(f"  мишени готовы за {time.time() - t0:.0f} с; доля координат на "
          f"пределе rho: {100 * sat:.2f}%")

    head = build_head(d_hidden, d_lat, n_pos, rank, a.proj, a.hidden, B, rho,
                      dev)
    # ПО ИМЕНАМ, А НЕ ПО ОБЪЕКТАМ: `p not in train_p` сравнивает тензоры
    # через ==, и bool() на многоэлементном результате падает
    pref = ("proj_h.", "proj_z.", "net.")
    named = list(head.named_parameters())
    train_p = [p for n_, p in named if n_.startswith(pref)]
    frozen = [n_ for n_, _p in named if not n_.startswith(pref)]
    n_par = sum(p.numel() for p in train_p)
    # БАЗИС И rho — БУФЕРЫ, их нет в named_parameters вовсе: они не заморожены
    # оптимизатором, а не могут обучаться по построению. Печатать пустой
    # список «заморожено» было бы обманчиво.
    bufs = [n_ for n_, _b in head.named_buffers()]
    print(f"  голова: обучаемых параметров {n_par} в {len(train_p)} тензорах; "
          f"не-обучаемых параметров {frozen or 'нет'}; буферы (не обучаются "
          f"по построению): {bufs}")
    opt = torch.optim.Adam(train_p, lr=a.lr, weight_decay=a.wd)

    base_sel = evaluate(head, feed, vsel, T_sel, a.batch, dev)
    base_cnf = evaluate(head, feed, vcnf, T_cnf, a.batch, dev)
    print(f"\n  эпоха 0 (поправка тождественно нулевая, это fast12): val "
          f"отбор {base_sel:.5f}, val подтверждение {base_cnf:.5f}")

    # НУЛЕВАЯ ГОЛОВА — ПОЛНОПРАВНЫЙ УЧАСТНИК ОТБОРА. С best=inf первая же
    # обученная эпоха сохранялась бы, даже будучи хуже fast12, и при этом
    # называлась бы «эпоха 0». Теперь эпоха 0 — это и есть необученная голова.
    best = dict(loss=base_sel, epoch=0,
                state={k: v.detach().cpu().clone()
                       for k, v in head.state_dict().items()})
    hist, bad = [], 0
    rng = np.random.default_rng(a.seed)
    for ep in range(1, a.epochs + 1):
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
            run += float(loss.detach())
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

    head.load_state_dict(best["state"])
    cnf = evaluate(head, feed, vcnf, T_cnf, a.batch, dev)
    print(f"\n  выбрана эпоха {best['epoch']}"
          + (" — НУЛЕВАЯ ГОЛОВА: обучение не улучшило ни одной эпохи"
             if best["epoch"] == 0 else ""))
    print(f"    val отбор {best['loss']:.5f} (было {base_sel:.5f}), снято "
          f"{100 * (1 - best['loss'] / base_sel):.1f}% — но по этому числу "
          f"эпоха и выбиралась")
    print(f"    val ПОДТВЕРЖДЕНИЕ {cnf:.5f} (было {base_cnf:.5f}), снято "
          f"{100 * (1 - cnf / base_cnf):.1f}% — честная величина")

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
        val_confirm_before=base_cnf,
        val_confirm_gain=float(1 - cnf / base_cnf) if base_cnf else None,
        split_seed=SPLIT_SEED, sel_frac=a.sel_frac,
        n_val_sel=int(len(vsel)), n_val_confirm=int(len(vcnf)),
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
