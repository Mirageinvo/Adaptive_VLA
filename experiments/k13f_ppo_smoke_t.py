#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-13f: PPO smoke для ТРАЕКТОРНОЙ головы. Сохранённый батч, без среды и VLM.

ЗАЧЕМ ОТДЕЛЬНО ОТ K-11i. Тот стенд написан под позиционную голову, где
действие имеет форму [batch, 16, 32] — 512 координат на чанк. Здесь действие
это [batch, 64]: ОДНО решение на чанк. Правдоподобие, KL и отношение
правдоподобий суммируются по другому числу координат, и переносить выводы
K-11i сюда нельзя — их надо получить заново.

ЧТО НАСТОЯЩЕЕ И ЧТО НЕТ. Настоящие: активации h24 и черновики z0 из кэша
K-11a, веса обеих голов, sigma_T из K-13d, вся механика обновления.
ПОДДЕЛЬНЫЕ: преимущества — случайные нормированные числа, награды здесь нет.
Стенд проверяет МЕХАНИКУ, а не обучение; утверждения «политика улучшается»
отсюда не следует.

ГЛАВНАЯ ПРОВЕРКА — ТОЖДЕСТВО ДО ПЕРВОГО ШАГА. При реплее сохранённого u
отношение правдоподобий обязано быть РОВНО единицей, KL нулём, доля обрезанных
нулём. Если нет, политика пересчитывает правдоподобие не того действия, и всё
дальнейшее опирается на неверное отношение.

ПОЧЕМУ log_std ЗАМОРОЖЕНА И ЧТО ИЗ ЭТОГО ПРОВЕРЯЕТСЯ. Рабочая точка sigma_T
подобрана в K-13d по RMS изменения исполняемых действий. Если log_std начнёт
учиться, эта точка уедет, и лестница пойдёт при неизвестной амплитуде
исследования. Поэтому здесь проверяется не проекция log_std к границе (её у
траекторной головы нет), а то, что log_std не сдвинулась НИ НА БИТ за все
эпохи обновления.

ОБЕ ГОЛОВЫ ОБЯЗАТЕЛЬНЫ: выбирать одну после того, как увидели результат
другой, — отбор по увиденному.
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


def check_frozen(before, after, name="log_std"):
    """Побитовое равенство, а не «изменилось незначительно».

    Замороженный параметр, сдвинувшийся на 1e-8, — это не «почти заморожен»:
    это работающий градиентный путь, который за сотню шагов уведёт sigma из
    откалиброванной точки. Допуск здесь означал бы, что проверки нет.
    """
    import torch
    if not torch.equal(before, after):
        d = float((after - before).abs().max())
        raise SystemExit(
            f"{name} сдвинулась на {d:.3e} при том, что заморожена: рабочая "
            f"точка sigma_T уехала бы, и лестница пошла бы при неизвестной "
            f"амплитуде исследования")
    return True


def check_grad_paths(head, u, z0, log=print):
    """Куда градиент идти обязан и куда не смеет.

    Сохранённое действие на обновлении — КОНСТАНТА. Если у него есть история,
    вместо score-function получается репараметризация, отношение
    правдоподобий теряет смысл, а обучение идёт по другому оценщику, не
    сообщив об этом.
    """
    import torch
    bad = []
    if u.grad_fn is not None:
        bad.append("у сохранённого u есть grad_fn: действие утекло в граф")
    if z0.grad is not None and float(z0.grad.abs().max()) > 0.0:
        bad.append("градиент прошёл в черновик z0: голова переучивает q0 "
                   "через себя")
    for nm in ("basis", "rho"):
        p = getattr(head, nm, None)
        if p is not None and getattr(p, "grad", None) is not None:
            if float(p.grad.abs().max()) > 0.0:
                bad.append(f"{nm} получил градиент: он обязан быть заморожен, "
                           f"иначе предел амплитуды перестаёт держаться")
    got = [n for n, p in head.named_parameters()
           if p.grad is not None and torch.isfinite(p.grad).all()
           and float(p.grad.abs().max()) > 0]
    if not got:
        bad.append("ни один параметр головы не получил градиента: "
                   "обновление ничего не меняет")
    if bad:
        raise SystemExit("ГРАДИЕНТНЫЕ ПУТИ НЕВЕРНЫ:\n    "
                         + "\n    ".join(bad))
    log(f"    градиент идёт в {len(got)} тензоров mu-ветви; u константа, "
        f"z0, базис и rho не учатся")
    return got


def selftest():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    import k11i_ppo_smoke as k11i
    from hicora_t_g import make_gaussian_trajectory_head

    # --- механика PPO переиспользуется, а не переписывается ---------------
    lp = torch.zeros(8)
    t = k11i.ppo_terms(lp, lp.clone(), torch.randn(8))
    assert abs(t["kl_k1"]) < 1e-12 and t["clip_frac"] == 0.0
    assert t["ratio_ok"]

    # --- заморозка проверяется ПОБИТОВО ----------------------------------
    a = torch.tensor([-2.3, -2.3])
    check_frozen(a, a.clone())
    # СДВИГ НА ОДИН БИТ, а не «на маленькое число»: 1e-8 к -2.3 во float32
    # просто округляется обратно, и такой тест ничего бы не проверил.
    nxt = torch.nextafter(a, torch.full_like(a, 0.0))
    assert not torch.equal(a, nxt)
    try:
        check_frozen(a, nxt)
    except SystemExit as e:
        assert "заморожена" in str(e), e
    else:
        raise AssertionError("сдвиг замороженного параметра пропущен")

    # --- тождество на настоящей траекторной голове ------------------------
    torch.manual_seed(0)
    D_H, D_L, NP_, RK = 24, 32, 4, 6
    q, _ = torch.linalg.qr(torch.randn(NP_ * D_L, RK, dtype=torch.float64))
    B = q.T.to(torch.float32).contiguous()          # [rank, n_pos*d_latent]
    head = make_gaussian_trajectory_head()(
        D_H, D_L, n_pos=NP_, rank=RK, proj=8, hidden=16)
    head.set_basis(B)
    head.set_rho(torch.full((RK,), 0.5))
    for p in head.net.parameters():
        torch.nn.init.normal_(p, 0.0, 0.2)
    head.freeze_log_std().eval()

    h = torch.randn(5, NP_, D_H)
    z = torch.randn(5, NP_, D_L)
    with torch.no_grad():
        o = head(h, z)
    u_buf = o["u"].detach().clone()
    assert tuple(u_buf.shape) == (5, RK), tuple(u_buf.shape)
    again = head(h, z, u=u_buf)
    tt = k11i.ppo_terms(again["log_prob_u"], o["log_prob_u"].detach(),
                        torch.randn(5))
    dev = float((tt["ratio"] - 1.0).abs().max())
    assert dev < 1e-5, f"отношение не единица до шага: {dev:.3e}"
    assert abs(tt["kl_k1"]) < 1e-9 and tt["clip_frac"] == 0.0

    # --- KL по 64 координатам суммируется, а не усредняется ---------------
    kl = k11i.analytic_kl(o["mu"], o["std"], o["mu"], o["std"])
    assert kl["n_dim"] == RK, (kl["n_dim"], RK)
    assert abs(kl["joint_mean"]) < 1e-9

    # --- градиентные пути --------------------------------------------------
    zg = z.clone().requires_grad_(True)
    out = head(h, zg, u=u_buf)
    (out["log_prob_u"].sum()).backward()
    check_grad_paths(head, u_buf, zg, log=lambda *_: None)
    assert head.log_std.grad is None, "замороженная log_std получила градиент"

    # --- утечка действия в граф обязана быть замечена ----------------------
    leaked = head(h, z)["u"]          # с историей
    try:
        check_grad_paths(head, leaked, z, log=lambda *_: None)
    except SystemExit as e:
        assert "grad_fn" in str(e), e
    else:
        raise AssertionError("утечка сохранённого действия пропущена")
    print("самопроверка k13f_ppo_smoke_t пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--basis-t", default="data/k13a_traj_basis")
    ap.add_argument("--head-s0", default="data/k13b_hicora_t_s0.pt")
    ap.add_argument("--head-s1", default="data/k13b_hicora_t_s1.pt")
    ap.add_argument("--sigma-json", action="append", default=[],
                    help="артефакты K-13d, по одному на голову")
    ap.add_argument("--res-norm-cache", default="data/k11c_res_norm.pt")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/k13f/ppo_smoke_t.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, os.path.abspath("experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import k11a_build_hicora_cache as k11a
    import k11i_ppo_smoke as k11i
    import k12b_protocol as kb
    import k13a_build_trajectory_basis as k13a
    from hicora_t_g import make_gaussian_trajectory_head

    dev = torch.device(a.device)
    torch.manual_seed(a.seed)

    # --- sigma_T БЕРЁТСЯ ИЗ АРТЕФАКТА, А НЕ ИЗ КОМАНДНОЙ СТРОКИ -----------
    # Число, введённое руками, невозможно связать с калибровкой, которая его
    # породила. Здесь читается тот самый файл, и его происхождение сверяется.
    if len(a.sigma_json) != 2:
        raise SystemExit("нужны два --sigma-json: по одному на голову")
    sig = {}
    for p in a.sigma_json:
        o = json.load(open(p))
        if o["split"] != "train":
            raise SystemExit(f"{p}: калибровка на {o['split']}, а не train")
        off = abs(o["rms_t"] - o["target_rms"]) / o["target_rms"]
        if off > 0.02:
            raise SystemExit(f"{p}: RMS отклоняется от цели на {100*off:.1f}%")
        sig[int(o["head_seed"])] = dict(sigma=float(o["sigma_t"]), path=p,
                                        sha=sha12(p), horizon=o["horizon"])
    if set(sig) != {0, 1}:
        raise SystemExit(f"артефакты калибровки для сидов {sorted(sig)}, "
                         f"нужны 0 и 1")

    # --- данные и res_norm -------------------------------------------------
    H24 = np.load(f"{a.cache}.h24.npy", mmap_mode="r")
    q0hat = np.load(f"{a.cache}.q0hat.npy", mmap_mode="r")
    E = np.load(f"{a.cache}.codebooks.npy")
    idx, _ = k13a.load_split(f"{a.cache}.split.npy", H24.shape[0])
    rng = np.random.default_rng(a.seed)
    rows = np.sort(rng.choice(idx["train"], size=min(a.batch,
                                                     len(idx["train"])),
                              replace=False))
    res_norm = torch.load(a.res_norm_cache, map_location=dev,
                          weights_only=False).to(dev).eval()
    for p_ in res_norm.parameters():
        p_.requires_grad_(False)
    rn_sha = k11a.state_sha1(res_norm)
    rn_dtype = next(res_norm.parameters()).dtype
    with torch.no_grad():
        h24 = res_norm(torch.from_numpy(np.asarray(H24[rows])).to(
            dev, rn_dtype)).float()
        z0 = torch.from_numpy(E[0]).to(dev)[torch.from_numpy(
            np.asarray(q0hat[rows]).astype(np.int64)).to(dev)].float()
    print(f"  батч: {len(rows)} строк TRAIN, h24 {tuple(h24.shape)}, "
          f"z0 {tuple(z0.shape)}")

    # --- головы и провенанс ------------------------------------------------
    B_t = np.load(f"{a.basis_t}.basis.npy")
    rho_t = np.load(f"{a.basis_t}.rho.npy")
    heads, objs = {}, {}
    for tag, path in (("s0", a.head_s0), ("s1", a.head_s1)):
        o = torch.load(path, map_location="cpu", weights_only=False)
        for nm, arr, want in (("базис", B_t, o["basis_sha1"]),
                              ("rho", rho_t, o["rho_sha1"])):
            if arr_sha(arr) != want:
                raise SystemExit(f"{tag}: {nm} на диске {arr_sha(arr)}, "
                                 f"голова обучена на {want}")
        if o["res_norm_sha1"] != rn_sha:
            raise SystemExit(f"{tag}: res_norm {rn_sha}, голова обучена на "
                             f"{o['res_norm_sha1']}")
        if os.path.realpath(o["cache"]) != os.path.realpath(a.cache):
            raise SystemExit(f"{tag}: голова обучена на кэше {o['cache']}")
        sd = int(o["seed"])
        if sd not in sig:
            raise SystemExit(f"{tag}: нет калибровки для сида {sd}")
        g = make_gaussian_trajectory_head()(
            int(o["d_hidden"]), int(o["d_latent"]), n_pos=int(o["n_pos"]),
            rank=int(o["rank"]), proj=int(o["proj"]),
            hidden=int(o["hidden"]),
            init_log_std=float(np.log(sig[sd]["sigma"]))).to(dev)
        g.set_basis(torch.as_tensor(B_t))
        g.set_rho(torch.as_tensor(rho_t))
        have = set(g.state_dict())
        miss = sorted(set(o["state"]) - have)
        if miss:
            raise SystemExit(f"{tag}: ключей нет в голове: {miss}")
        with torch.no_grad():
            for k, v in o["state"].items():
                if k in ("basis", "rho", "basis_set", "rho_set", "log_std"):
                    continue
                g.state_dict()[k].copy_(v.to(dev, torch.float32))
        g.freeze_log_std()
        s_got = float(g.std().max())
        if abs(s_got - sig[sd]["sigma"]) > 1e-6:
            raise SystemExit(f"{tag}: sigma головы {s_got:.6f}, "
                             f"калибровка дала {sig[sd]['sigma']:.6f}")
        heads[tag], objs[tag] = g, o
        print(f"  {tag}: сид {sd}, ранг {o['rank']}, sigma {s_got:.5f} из "
              f"{os.path.basename(sig[sd]['path'])}")
    if sha12(a.head_s0) == sha12(a.head_s1):
        raise SystemExit("обе головы — один файл: это не две реплики")
    if int(objs["s0"]["seed"]) == int(objs["s1"]["seed"]):
        raise SystemExit("у голов один сид: это не две реплики")

    # --- прогон -------------------------------------------------------------
    res = {}
    for tag, head in heads.items():
        print(f"\n  === {tag} ===")
        # ГЕНЕРАТОР НА УСТРОЙСТВЕ ГОЛОВЫ. `normal_` требует совпадения
        # устройств генератора и тензора; перекладывать батч на CPU ради
        # сэмплирования значило бы получить падение при --device cuda.
        gen = torch.Generator(device=dev).manual_seed(a.seed + 1)
        with torch.no_grad():
            o0 = head(h24, z0, generator=gen)
        u_buf = o0["u"].detach().clone()
        mu_old = o0["mu"].detach().clone()
        std_old = o0["std"].detach().clone()
        logp_old = o0["log_prob_u"].detach().clone()
        adv = torch.from_numpy(
            rng.normal(size=len(rows)).astype(np.float32)).to(dev)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ТОЖДЕСТВО ДО ПЕРВОГО ШАГА
        with torch.no_grad():
            rep = head(h24, z0, u=u_buf)
            t0 = k11i.ppo_terms(rep["log_prob_u"], logp_old, adv)
            t0["ratio_cpu"] = t0["ratio"].cpu().numpy()
        k11i.check_identity(t0)
        kl0 = k11i.analytic_kl(mu_old, std_old, rep["mu"], rep["std"])
        print(f"    тождество до шага: отношение 1 +- "
              f"{float(np.abs(t0['ratio_cpu'] - 1).max()):.1e}, KL "
              f"{kl0['joint_mean']:.2e} по {kl0['n_dim']} координатам")

        # ГРАДИЕНТНЫЕ ПУТИ
        head.zero_grad(set_to_none=True)
        zg = z0.clone().requires_grad_(True)
        out = head(h24, zg, u=u_buf)
        k11i.ppo_terms(out["log_prob_u"], logp_old, adv)["loss"].backward()
        trainable = check_grad_paths(head, u_buf, zg, log=print)
        if head.log_std.grad is not None:
            raise SystemExit("замороженная log_std получила градиент")

        ls_before = head.log_std.detach().clone()
        params = [p for n, p in head.named_parameters()
                  if n.startswith(head.trainable_prefixes())]
        opt = torch.optim.Adam(params, lr=a.lr)
        hist = []
        for ep in range(a.epochs):
            perm = torch.randperm(len(rows), device=dev)
            for i in range(0, len(rows), a.minibatch):
                s_ = perm[i:i + a.minibatch]
                opt.zero_grad(set_to_none=True)
                o_ = head(h24[s_], z0[s_], u=u_buf[s_])
                t_ = k11i.ppo_terms(o_["log_prob_u"], logp_old[s_], adv[s_])
                if not t_["ratio_ok"]:
                    raise SystemExit(
                        f"{tag}, эпоха {ep}: отношение правдоподобий "
                        f"непригодно (обнулилось {100*t_['ratio_zero_frac']:.1f}%"
                        f", конечно: {t_['ratio_finite']})")
                t_["loss"].backward()
                opt.step()
            with torch.no_grad():
                o_ = head(h24, z0, u=u_buf)
                t_ = k11i.ppo_terms(o_["log_prob_u"], logp_old, adv)
                kl = k11i.analytic_kl(mu_old, std_old, o_["mu"], o_["std"])
                nrm = float(torch.linalg.norm(
                    o_["dz"].flatten(1), dim=-1).max())
            hist.append(dict(epoch=ep + 1, kl_joint=kl["joint_mean"],
                             kl_per_dim=kl["per_dim_mean"],
                             clip_frac=t_["clip_frac"],
                             log_ratio_absmax=t_["log_ratio_absmax"],
                             dz_norm_max=nrm))
            print(f"    эпоха {ep + 1}: KL {kl['joint_mean']:.4f} "
                  f"(на координату {kl['per_dim_mean']:.2e}), обрезано "
                  f"{100 * t_['clip_frac']:.1f}%, max|log r| "
                  f"{t_['log_ratio_absmax']:.3f}, max||dz|| {nrm:.4f}")
        check_frozen(ls_before, head.log_std.detach())
        lim = head.bound()
        worst = max(h_["dz_norm_max"] for h_ in hist)
        if worst > lim + 1e-4:
            raise SystemExit(f"{tag}: ||dz|| дошла до {worst:.4f} при пределе "
                             f"{lim:.4f}: ограничение перестало держаться")
        print(f"    log_std не сдвинулась ни на бит; max||dz|| {worst:.4f} "
              f"при пределе {lim:.4f}")
        res[tag] = dict(seed=int(objs[tag]["seed"]),
                        sigma=float(head.std().max()),
                        sigma_json=sig[int(objs[tag]["seed"])]["path"],
                        sigma_json_sha1=sig[int(objs[tag]["seed"])]["sha"],
                        n_trainable=len(trainable), bound=lim,
                        identity_ratio_dev=float(
                            np.abs(t0["ratio_cpu"] - 1).max()),
                        kl_dims=kl0["n_dim"], history=hist)

    out = dict(arms=res, batch=int(len(rows)), split="train",
               epochs=int(a.epochs), lr=float(a.lr),
               minibatch=int(a.minibatch), cache=a.cache,
               res_norm_sha1=rn_sha, basis_sha1=arr_sha(B_t),
               rho_sha1=arr_sha(rho_t), advantages="fake_normalized_gaussian",
               note="механика обновления, не обучение: награды нет",
               code_version=kb.code_version([
                   os.path.abspath(__file__),
                   os.path.join(here, "hicora_t_g.py"),
                   os.path.join(here, "k11i_ppo_smoke.py")]),
               script_sha1=sha12(os.path.abspath(__file__)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = f"{a.out}.tmp.{os.getpid()}"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, a.out)
    print(f"\n  сохранено: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
