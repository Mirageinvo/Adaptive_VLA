#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-13f: PPO smoke для ТРАЕКТОРНОЙ головы. Сохранённый батч, без среды и VLM.

ЗАЧЕМ ОТДЕЛЬНО ОТ K-11i. Тот стенд написан под позиционную голову, где
действие имеет форму [batch, 16, 32] — 512 координат на чанк. Здесь действие
это [batch, 64]: ОДНО решение на чанк. Правдоподобие, KL и отношение
правдоподобий суммируются по другому числу координат, и переносить выводы
K-11i сюда нельзя — их надо получить заново.

ВОСПРОИЗВОДИТСЯ ПРОЦЕДУРА ЛЕСТНИЦЫ, А НЕ УЧЕБНЫЙ PPO. Обновление здесь — тот
самый `k12e.one_step`: один полнобатчевый шаг policy gradient, затем приёмка по
trust region (`q99 |log r|`, аварийный предел по абсолютному максимуму, предел
на KL), при отказе — дробление шага вдвое и побитовый откат параметров И
моментов Adam. Первая версия этого стенда гоняла clipped PPO с эпохами, то есть
проверяла процедуру, которой мы не пользуемся: её расхождение говорило о стенде,
а не о голове.

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


TRAIN_PREFIXES = ("proj_h.", "proj_z.", "net.")
BUFFERS = ("basis", "rho", "basis_set", "rho_set", "log_std")


def check_sigma_artifact(o, path, head_path, horizon=8):
    """Артефакт калибровки — ВХОДНЫЕ ДАННЫЕ, а не справка.

    Из него берётся рабочая точка исследования RL. Принять его, не проверив,
    значит стартовать лестницу при sigma, про которую неизвестно, чему она
    соответствует и от какой головы получена.
    """
    bad = []
    if o.get("split") != "train":
        bad.append(f"калибровка на {o.get('split')}, а не train")
    if int(o.get("horizon", -1)) != int(horizon):
        bad.append(f"горизонт {o.get('horizon')}, а исполняется {horizon}: "
                   f"sigma подобрана по возмущению, которого робот не видит")
    mv = o.get("mean_vs_det")
    if mv is None or float(mv) > 1e-5:
        bad.append(f"mean_vs_det {mv}: среднее гауссовой головы расходится с "
                   f"детерминированной, RL стартовал бы не из проверенной "
                   f"точки")
    grid = (o.get("history") or {}).get("grid")
    if not grid or len(grid) < 7:
        bad.append(f"в истории {0 if not grid else len(grid)} точек сетки "
                   f"вместо семи: монотонность не проверялась")
    else:
        vals = [v for _s, v in grid]
        if any(vals[i + 1] < vals[i] for i in range(len(vals) - 1)):
            bad.append("RMS на сетке не монотонен")
    off = abs(o["rms_t"] - o["target_rms"]) / o["target_rms"]
    if off > 0.02:
        bad.append(f"RMS отклоняется от цели на {100 * off:.1f}%")
    got = sha12(head_path)
    if str(o.get("head_t_sha1")) != got:
        bad.append(f"калибровка снята с головы {o.get('head_t_sha1')}, а "
                   f"подана {got}")
    if bad:
        raise SystemExit(f"{path}: " + "; ".join(bad))
    return True


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
    import k12e_pg_step as k12e
    from hicora_t_g import make_gaussian_trajectory_head

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

    # --- АРТЕФАКТ КАЛИБРОВКИ ПРОВЕРЯЕТСЯ, А НЕ ПРИНИМАЕТСЯ НА ВЕРУ -------
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        hp = os.path.join(td, "head.pt")
        open(hp, "wb").write(b"head-bytes")
        good = dict(split="train", horizon=8, mean_vs_det=0.0,
                    rms_t=1.0, target_rms=1.0, head_t_sha1=sha12(hp),
                    history=dict(grid=[[0.001 * (i + 1), 0.01 * (i + 1)]
                                       for i in range(7)]))
        check_sigma_artifact(good, "тест", hp)
        for bad, why in (
                (dict(good, split="dev"), "не train"),
                (dict(good, horizon=16), "горизонт"),
                (dict(good, mean_vs_det=1e-3), "mean_vs_det"),
                (dict(good, history=dict(grid=[[1, 1], [2, 2]])), "точек"),
                (dict(good, history=dict(grid=[[0.1, 5.0], [0.2, 1.0]]
                                         + [[0.3 + i, 9.0 + i]
                                            for i in range(5)])), "монотонен"),
                (dict(good, rms_t=1.5), "отклоняется"),
                (dict(good, head_t_sha1="deadbeef0000"), "снята с головы")):
            try:
                check_sigma_artifact(bad, "тест", hp)
            except SystemExit as e:
                assert why in str(e), (why, str(e))
            else:
                raise AssertionError(f"артефакт калибровки пропущен: {why}")

    # --- настоящая траекторная голова -------------------------------------
    torch.manual_seed(0)
    D_H, D_L, NP_, RK, N = 24, 32, 4, 6, 64
    q, _ = torch.linalg.qr(torch.randn(NP_ * D_L, RK, dtype=torch.float64))
    B = q.T.to(torch.float32).contiguous()          # [rank, n_pos*d_latent]
    head = make_gaussian_trajectory_head()(
        D_H, D_L, n_pos=NP_, rank=RK, proj=8, hidden=16, init_log_std=-2.3)
    head.set_basis(B)
    head.set_rho(torch.full((RK,), 0.5))
    for p_ in head.net.parameters():
        torch.nn.init.normal_(p_, 0.0, 0.2)
    head.freeze_log_std()

    h = torch.randn(N, NP_, D_H)
    cb0 = torch.randn(16, D_L)
    q0 = torch.randint(0, 16, (N, NP_))
    z0 = cb0[q0]
    std = head.std().detach()
    with torch.no_grad():
        o0 = head(h, z0)
    assert tuple(o0["u"].shape) == (N, RK), tuple(o0["u"].shape)
    buf = dict(n=N, h=h, q0=q0, cb0=cb0, u=o0["u"].clone(),
               mu=o0["mu"].clone(), logp=o0["log_prob_u"].clone(),
               order_sha1="test")

    # --- ЧЁТНОСТЬ И ТОЖДЕСТВО ДО ШАГА -------------------------------------
    par = k12e.parity_check(head, buf, std)
    assert par["ok"], par
    st = k12e.ratio_stats(buf["logp"].float(), buf["logp"].float(), 1.5, 3.0)
    assert st["ok"] and st["logdiff_q99"] == 0.0, st

    # --- KL СУММИРУЕТСЯ ПО 64 КООРДИНАТАМ, А НЕ ПО 512 --------------------
    # Ради этого стенд и отделён от K-11i: там действие имеет 16*32 координат,
    # здесь одно решение на чанк, и предел KL относится к другой величине.
    kl = k12e.kl_mu(buf["mu"], buf["mu"], std)
    assert tuple(kl.shape) == (N,) and float(kl.abs().max()) == 0.0
    kl2 = k12e.kl_mu(buf["mu"], buf["mu"] + float(std[0]), std)
    assert abs(float(kl2.mean()) - 0.5 * RK) < 1e-6, float(kl2.mean())

    # --- ГРАДИЕНТНЫЕ ПУТИ --------------------------------------------------
    head.zero_grad(set_to_none=True)
    zg = z0.clone().requires_grad_(True)
    mu_g = head.mean_coeffs(h, zg)
    head.log_prob_u(buf["u"], mu_g, std).sum().backward()
    check_grad_paths(head, buf["u"], zg, log=lambda *_: None)
    assert head.log_std.grad is None, "замороженная log_std получила градиент"
    leaked = head(h, z0)["u"]                       # с историей
    try:
        check_grad_paths(head, leaked, z0, log=lambda *_: None)
    except SystemExit as e:
        assert "grad_fn" in str(e), e
    else:
        raise AssertionError("утечка сохранённого действия пропущена")

    # --- ЧАСТИЧНАЯ ЗАГРУЗКА: ТОЖДЕСТВО log pi ЕЁ НЕ ЛОВИТ ----------------
    # Ключевой сценарий рецензента: один ключ головы не загружен, буфер
    # сэмплирован ЭТОЙ ЖЕ головой, и тождество сходится вокруг неправильного
    # среднего. Ловит его только сверка с независимо собранной головой.
    import k13g_traj_adapter as adp
    obj = dict(state={k_: v_.clone() for k_, v_ in head.state_dict().items()},
               arch="trajectory_mlp", target="coef", rank=RK, n_pos=NP_,
               d_latent=D_L, d_hidden=D_H, proj=8, hidden=16, cache="c",
               res_norm_sha1="a", basis_sha1=adp.array_sha(B.numpy()),
               rho_sha1=adp.array_sha(head.rho.detach().cpu().numpy()),
               selected_epoch=1, seed=0, script_sha1="b")
    ref_det = adp.build_heads(obj, B.numpy(),
                              head.rho.detach().cpu().numpy(), D_H, D_L,
                              torch.device("cpu"), torch)["det_cur"]
    with torch.no_grad():
        dz_ok = float((ref_det(h, z0)[0]
                       - head(h, z0, deterministic=True)["dz"]).abs().max())
    assert dz_ok <= 1e-5, dz_ok
    broken = make_gaussian_trajectory_head()(
        D_H, D_L, n_pos=NP_, rank=RK, proj=8, hidden=16, init_log_std=-2.3)
    broken.set_basis(B)
    broken.set_rho(head.rho.detach().cpu())
    with torch.no_grad():
        for k_, v_ in head.state_dict().items():
            if k_ in BUFFERS or k_.startswith("proj_z."):
                continue                       # «забытый» слой
            broken.state_dict()[k_].copy_(v_)
    broken.freeze_log_std()
    with torch.no_grad():
        ob = broken(h, z0)
        same = broken(h, z0, u=ob["u"])
        d_self = float((ob["log_prob_u"] - same["log_prob_u"]).abs().max())
        d_ind = float((ref_det(h, z0)[0]
                       - broken(h, z0, deterministic=True)["dz"]).abs().max())
    assert d_self <= 1e-6, "тождество должно сойтись даже у сломанной головы"
    assert d_ind > 1e-5, (
        "сверка с независимой головой обязана поймать неполную загрузку")

    # --- ШАГ С TRUST REGION: ПРИНЯТИЕ И ОТКАЗ -----------------------------
    adv = torch.randn(N)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    trust = dict(kl_max=0.02, ratio_max=1.5, ratio_hard=3.0)
    ls0 = head.log_std.detach().clone()
    params = [p_ for n_, p_ in head.named_parameters()
              if n_.startswith(head.trainable_prefixes())]
    opt = torch.optim.Adam(params, lr=1e-5)
    rec = k12e.one_step(head, opt, buf, adv, std, n_episodes=N, lr=1e-5,
                        trust=trust, max_halvings=8, micro=32,
                        log=lambda *_: None)
    assert rec["status"] == "stepped", rec["attempts"][-1]
    check_frozen(ls0, head.log_std.detach())

    # заведомо огромный шаг обязан быть отвергнут ЦЕЛИКОМ, с точным откатом
    snap = {k_: v_.detach().clone() for k_, v_ in head.state_dict().items()}
    opt2 = torch.optim.Adam(params, lr=1e9)
    rec2 = k12e.one_step(head, opt2, buf, adv, std, n_episodes=N, lr=1e9,
                         trust=trust, max_halvings=1, micro=32,
                         log=lambda *_: None)
    assert rec2["status"] == "no_step", rec2["status"]
    for k_, v_ in head.state_dict().items():
        assert torch.equal(v_.detach(), snap[k_]), f"откат неточен по {k_}"
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
    ap.add_argument("--micro", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    # ПРЕДЕЛЫ ТЕ ЖЕ, ЧТО В K-12e. Здесь они не подбираются: стенд обязан
    # принять или отвергнуть шаг по тому же правилу, по которому это сделает
    # лестница, иначе он проверяет другую процедуру.
    ap.add_argument("--kl-max", type=float, default=0.02)
    ap.add_argument("--ratio-max", type=float, default=1.5)
    ap.add_argument("--ratio-hard", type=float, default=3.0)
    ap.add_argument("--max-halvings", type=int, default=8)
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
    by_seed_ckpt = {}
    for path in (a.head_s0, a.head_s1):
        ob = torch.load(path, map_location="cpu", weights_only=False)
        by_seed_ckpt[int(ob["seed"])] = path
    sig = {}
    for p in a.sigma_json:
        o = json.load(open(p))
        sd = int(o["head_seed"])
        if sd not in by_seed_ckpt:
            raise SystemExit(f"{p}: калибровка для сида {sd}, а поданы головы "
                             f"сидов {sorted(by_seed_ckpt)}")
        check_sigma_artifact(o, p, by_seed_ckpt[sd], horizon=8)
        sig[sd] = dict(sigma=float(o["sigma_t"]), path=p, sha=sha12(p),
                       horizon=int(o["horizon"]))
        print(f"  калибровка {os.path.basename(p)}: sigma {o['sigma_t']:.5f}, "
              f"горизонт {o['horizon']}, mean_vs_det {o['mean_vs_det']:.1e}, "
              f"семь точек сетки монотонны, снята с головы {o['head_t_sha1']}")
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
        # РАВЕНСТВО МНОЖЕСТВ В ОБЕ СТОРОНЫ. Проверка только «ключи чекпойнта
        # есть в голове» пропускает главный случай: ключ головы, которого нет
        # в чекпойнте, остаётся в начальной инициализации. Дальше буфер
        # сэмплируется ЭТОЙ ЖЕ частично загруженной головой, и тождество
        # log pi сходится вокруг неправильного среднего — стенд говорит
        # «всё хорошо» о политике, которой не существует.
        want = {k for k in g.state_dict() if k.startswith(TRAIN_PREFIXES)}
        got = {k for k in o["state"] if k not in BUFFERS}
        if got != want:
            raise SystemExit(
                f"{tag}: обучаемые ключи не совпали — нет в чекпойнте "
                f"{sorted(want - got)[:5]}, нет в голове {sorted(got - want)[:5]}")
        stray = [k for k in o["state"]
                 if k not in BUFFERS and not k.startswith(TRAIN_PREFIXES)]
        if stray:
            raise SystemExit(f"{tag}: веса вне {TRAIN_PREFIXES}: {stray[:5]}")
        n_loaded = 0
        with torch.no_grad():
            for k, v in o["state"].items():
                if k in BUFFERS:
                    continue
                tgt = g.state_dict()[k]
                if tuple(tgt.shape) != tuple(v.shape):
                    raise SystemExit(f"{tag}: ключ {k} формы {tuple(v.shape)}, "
                                     f"в голове {tuple(tgt.shape)}")
                tgt.copy_(v.to(dev, torch.float32))
                n_loaded += 1
        if n_loaded != len(want):
            raise SystemExit(f"{tag}: загружено {n_loaded} из {len(want)}")
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

    # --- прогон: ТОТ ЖЕ ШАГ, ЧТО В ЛЕСТНИЦЕ ---------------------------------
    import k12e_pg_step as k12e
    cb0 = torch.from_numpy(E[0]).to(dev).float()
    q0_idx = torch.from_numpy(
        np.asarray(q0hat[rows]).astype(np.int64)).to(dev)
    trust = dict(kl_max=float(a.kl_max), ratio_max=float(a.ratio_max),
                 ratio_hard=float(a.ratio_hard))
    # СВЕРКА С ДЕТЕРМИНИРОВАННОЙ ГОЛОВОЙ НА НАСТОЯЩЕМ БАТЧЕ. Это последняя
    # защита от частичной загрузки: детерминированная голова строится
    # НЕЗАВИСИМО, тем же кодом, что исполнялся в K-13c, и её поправка обязана
    # совпасть со средним гауссовой. Тождество log pi этого не даёт — оно
    # сходится и вокруг неправильного среднего, потому что буфер сэмплирован
    # той же головой.
    import k13g_traj_adapter as adp
    mean_vs_det = {}
    for tag, head in heads.items():
        o = objs[tag]
        det = adp.build_heads(o, B_t, rho_t, int(o["d_hidden"]),
                              int(o["d_latent"]), dev, torch)["det_cur"]
        with torch.no_grad():
            dz_det, _c = det(h24, z0)
            dz_gau = head(h24, z0, deterministic=True)["dz"]
        d = float((dz_det - dz_gau).abs().max())
        if d > 1e-5:
            raise SystemExit(
                f"{tag}: среднее гауссовой головы расходится с независимо "
                f"собранной детерминированной на {d:.2e} при допуске 1e-5. "
                f"Это признак неполной загрузки весов: тождество log pi "
                f"сошлось бы и в этом случае")
        mean_vs_det[tag] = d
        print(f"  {tag}: среднее совпало с независимой детерминированной "
              f"головой, max|Δdz| = {d:.2e}")

    res = {}
    for tag, head in heads.items():
        print(f"\n  === {tag} ===")
        std = head.std().detach()
        gen = torch.Generator(device=dev).manual_seed(a.seed + 1)
        with torch.no_grad():
            o0 = head(h24, z0, generator=gen)
        buf = dict(n=len(rows), h=h24.detach(), q0=q0_idx, cb0=cb0,
                   u=o0["u"].detach().clone(),
                   mu=o0["mu"].detach().clone(),
                   logp=o0["log_prob_u"].detach().clone(),
                   order_sha1="smoke")
        adv = torch.from_numpy(
            rng.normal(size=len(rows)).astype(np.float32)).to(dev)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ЧЁТНОСТЬ: пересчитанное log pi обязано совпасть с записанным.
        par = k12e.parity_check(head, buf, std)
        if not par["ok"]:
            raise SystemExit(
                f"{tag}: пересчитанное log pi расходится с записанным на "
                f"{par['logp_max_abs_diff']:.3e} (mu на "
                f"{par['mu_max_abs_diff']:.3e}). Отношение правдоподобий "
                f"начиналось бы не с единицы, и шаг считался бы не по тем "
                f"сэмплам")
        t0 = k12e.ratio_stats(buf["logp"].float(), buf["logp"].float(),
                              float(a.ratio_max), float(a.ratio_hard))
        if not t0["ok"] or t0["logdiff_q99"] != 0.0:
            raise SystemExit(f"{tag}: тождество до шага нарушено: {t0}")
        print(f"    чётность: log pi совпало до {par['logp_max_abs_diff']:.1e}"
              f", mu до {par['mu_max_abs_diff']:.1e}; отношение до шага ровно "
              f"единица по {int(buf['u'].shape[-1])} координатам")

        # ГРАДИЕНТНЫЕ ПУТИ — до шага, на отдельном проходе
        head.zero_grad(set_to_none=True)
        zg = z0.clone().requires_grad_(True)
        mu_g = head.mean_coeffs(h24, zg)
        head.log_prob_u(buf["u"], mu_g, std).sum().backward()
        trainable = check_grad_paths(head, buf["u"], zg, log=print)
        if head.log_std.grad is not None:
            raise SystemExit("замороженная log_std получила градиент")

        ls_before = head.log_std.detach().clone()
        params = [p for n, p in head.named_parameters()
                  if n.startswith(head.trainable_prefixes())]
        opt = torch.optim.Adam(params, lr=a.lr)
        rec = k12e.one_step(head, opt, buf, adv, std,
                            n_episodes=len(rows), lr=a.lr, trust=trust,
                            max_halvings=a.max_halvings, micro=a.micro,
                            log=lambda m: print("    " + m))
        check_frozen(ls_before, head.log_std.detach())
        with torch.no_grad():
            nrm = float(torch.linalg.norm(
                head(h24, z0, u=buf["u"])["dz"].flatten(1), dim=-1).max())
        lim = head.bound()
        if nrm > lim + 1e-4:
            raise SystemExit(f"{tag}: ||dz|| дошла до {nrm:.4f} при пределе "
                             f"{lim:.4f}: ограничение перестало держаться")
        if rec["status"] != "stepped":
            raise SystemExit(
                f"{tag}: шаг не принят ни при одном из {a.max_halvings} "
                f"дроблений. Это не поломка стенда, а свойство рабочей точки: "
                f"при sigma_T={float(std.max()):.5f} и lr={a.lr} любой шаг "
                f"выходит за trust region. До лестницы нужно выбрать lr, "
                f"а не запускать её вслепую")
        print(f"    шаг принят после {rec['halvings']} дроблений при "
              f"lr={rec['lr_used']:.3g}: KL {rec['kl_mean']:.5f}, q99 |log r| "
              f"{rec['ratio']['logdiff_q99']:.4f}, сдвиг mu макс "
              f"{rec['mu_absdiff_max']:.4f}")
        print(f"    log_std не сдвинулась ни на бит; max||dz|| {nrm:.4f} "
              f"при пределе {lim:.4f}")
        res[tag] = dict(seed=int(objs[tag]["seed"]),
                        sigma=float(std.max()),
                        sigma_json=sig[int(objs[tag]["seed"])]["path"],
                        sigma_json_sha1=sig[int(objs[tag]["seed"])]["sha"],
                        n_trainable=len(trainable), bound=lim,
                        dz_norm_max=nrm, parity=par, step=rec,
                        mean_vs_independent_det=float(mean_vs_det[tag]),
                        kl_dims=int(buf["u"].shape[-1]))

    out = dict(arms=res, batch=int(len(rows)), split="train",
               lr=float(a.lr), trust=trust, micro=int(a.micro),
               max_halvings=int(a.max_halvings), cache=a.cache,
               res_norm_sha1=rn_sha, basis_sha1=arr_sha(B_t),
               rho_sha1=arr_sha(rho_t), advantages="fake_normalized_gaussian",
               note="механика обновления, не обучение: награды нет",
               code_version=kb.code_version([
                   os.path.abspath(__file__),
                   os.path.join(here, "hicora_t_g.py"),
                   os.path.join(here, "k12e_pg_step.py")]),
               script_sha1=sha12(os.path.abspath(__file__)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = f"{a.out}.tmp.{os.getpid()}"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, a.out)
    print(f"\n  сохранено: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
