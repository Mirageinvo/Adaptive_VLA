"""K-11i: PPO smoke на сохранённом батче. Без среды и без VLM.

ЗАЧЕМ. Обновление PPO ломается тихо: отношение правдоподобий может оказаться
не единицей до первого шага, сохранённое действие может утечь в граф,
log_ratio по 512 координатам может переполнить exp, а обрезанный log_std может
навсегда остановить обучение sigma. Всё это стоит часов раскатки, а
обнаруживается на батче за минуту.

ЧТО ЗДЕСЬ НАСТОЯЩЕЕ И ЧТО НЕТ. Настоящие: активации h24 и черновики z0 из кэша
K-11a, веса обеих голов D1, sigma = 0.10 (выбрана гейтом K-11g), вся механика
обновления. ПОДДЕЛЬНЫЕ: преимущества — случайные нормированные числа, потому
что награды здесь нет. Поэтому стенд проверяет МЕХАНИКУ обновления, а не
обучение: никакого утверждения о том, что политика улучшается, отсюда не
следует.

ОБЕ ГОЛОВЫ ОБЯЗАТЕЛЬНЫ. Выбирать s0 после того, как K-11g показал её большую
устойчивость, — отбор по увиденному.

ПРОВЕРКИ ДО ПЕРВОГО ШАГА (самое важное):
  отношение правдоподобий РОВНО единица при реплее сохранённого u;
  приближённая KL ноль, доля обрезанных ноль;
  сохранённое u — константа: grad_fn отсутствует, градиент идёт только через
    mu и log_std (путь score-function).
Если отношение не единица до обновления, значит политика пересчитывает
правдоподобие не того действия, и всё дальнейшее бессмысленно.

Запуск:
    python3 experiments/k11i_ppo_smoke.py --selftest
    python experiments/k11i_ppo_smoke.py --cache data/k11a_joint12 \
        --hicora-s0 ... --hicora-s1 ... --sigma 0.10
"""

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLIP_EPS = 0.2
RATIO_TOL = 1e-6        # допуск на «отношение ровно единица»
KL_TOL = 1e-9


def analytic_kl(mu_old, std_old, mu_new, std_new):
    """Точный KL(pi_old || pi_new) между диагональными гауссианами.

    ЗАЧЕМ СВЕРХ k1/k3. Те — ОЦЕНКИ по сэмплам, и при катастрофических
    отношениях правдоподобий они расходятся на порядки, так что величина
    «KL = 5e19» ничего не измеряет. Здесь политики гауссовы, и KL берётся
    аналитически: он конечен всегда и сравним между шагами.

    Возвращает joint KL (сумма по всем 512 координатам чанка) и KL на
    координату: первая величина — то, что ограничивают в PPO, вторая нужна,
    чтобы видеть, насколько сдвиг мал ПОКООРДИНАТНО.
    """
    import torch
    v_new = std_new * std_new
    per = (torch.log(std_new / std_old)
           + (std_old * std_old + (mu_old - mu_new) ** 2) / (2.0 * v_new)
           - 0.5)
    joint = per.flatten(1).sum(-1)
    n_dim = int(per.flatten(1).shape[-1])
    return dict(joint_mean=float(joint.mean()),
                joint_max=float(joint.max()),
                per_dim_mean=float(joint.mean()) / n_dim, n_dim=n_dim)


def ppo_terms(logp_new, logp_old, adv, clip_eps=CLIP_EPS):
    """Члены PPO и диагностика. Чистая функция, проверяется отдельно.

    `log_ratio` считается как разность логарифмов, а не делением
    правдоподобий: при 512 координатах сами правдоподобия уходят за пределы
    float задолго до того, как отношение станет интересным.
    """
    import torch
    log_ratio = logp_new - logp_old
    ratio = torch.exp(log_ratio)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    loss = -torch.min(unclipped, clipped).mean()
    with torch.no_grad():
        # Два оценщика KL: k1 смещён, k3 неотрицателен. Печатаются оба, потому
        # что при больших log_ratio они расходятся, и это само по себе сигнал.
        kl_k1 = (-log_ratio).mean()
        kl_k3 = (ratio - 1.0 - log_ratio).mean()
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()
    with torch.no_grad():
        # ОБНУЛЕНИЕ ОТНОШЕНИЯ — ТОЖЕ ОТКАЗ, А НЕ «конечное число». При
        # log_ratio около -90 exp в float32 даёт РОВНО нуль: обрезанный член
        # становится константой, градиент по таким примерам мёртв, а проверка
        # на конечность их пропускала, и вывод «переполнений нет» мог
        # печататься при массовом обнулении.
        zero_frac = float((ratio == 0).float().mean())
        finite = bool(torch.isfinite(ratio).all())
    return dict(loss=loss, ratio=ratio, log_ratio=log_ratio,
                kl_k1=float(kl_k1), kl_k3=float(kl_k3),
                clip_frac=float(clip_frac),
                log_ratio_absmax=float(log_ratio.abs().max()),
                log_ratio_min=float(log_ratio.min()),
                log_ratio_max=float(log_ratio.max()),
                log_ratio_std=float(log_ratio.std()),
                ratio_zero_frac=zero_frac,
                ratio_ok=bool(finite and zero_frac == 0.0),
                ratio_finite=finite,
                loss_finite=bool(torch.isfinite(loss).all()))


def step_and_project(opt, head):
    """Шаг оптимизатора И проекция log_std. ОТДЕЛЬНОЙ ФУНКЦИЕЙ НАМЕРЕННО.

    Удаление проекции из цикла обновления короткий прогон не замечает:
    log_std просто не успевает выйти за границу. Вынесенная функция позволяет
    проверить САМ ВЫЗОВ в самопроверке, подменив проекцию счётчиком. Это
    покрывает оркестрацию; естественное достижение границы остаётся
    непокрытым, и это сказано прямо.
    """
    opt.step()
    head.project_log_std_()


def check_identity(t, ratio_tol=RATIO_TOL, kl_tol=KL_TOL):
    """До первого шага обновление обязано быть тождественным."""
    bad = []
    dev = float(np.abs(np.asarray(t["ratio_cpu"]) - 1.0).max())
    if dev > ratio_tol:
        bad.append(f"отношение правдоподобий отклоняется от единицы на "
                   f"{dev:.3e} (допуск {ratio_tol:.0e})")
    if abs(t["kl_k1"]) > kl_tol or abs(t["kl_k3"]) > kl_tol:
        bad.append(f"KL не ноль: k1={t['kl_k1']:.3e}, k3={t['kl_k3']:.3e}")
    if t["clip_frac"] != 0.0:
        bad.append(f"доля обрезанных {t['clip_frac']:.3f} вместо нуля")
    if not t["ratio_finite"]:
        bad.append("нечисловые значения в отношении правдоподобий")
    if not t["loss_finite"]:
        bad.append("нечисловое значение потери")
    if t["ratio_zero_frac"]:
        bad.append(f"{100 * t['ratio_zero_frac']:.1f}% отношений обнулились: "
                   f"градиент по этим примерам мёртв")
    if bad:
        raise SystemExit(
            "ОБНОВЛЕНИЕ НЕ ТОЖДЕСТВЕННО ДО ПЕРВОГО ШАГА:\n    "
            + "\n    ".join(bad)
            + "\n  Значит политика пересчитывает правдоподобие НЕ ТОГО "
              "действия, и всё\n  дальнейшее обучение опирается на неверное "
              "отношение.")
    return True


def selftest():
    import torch

    # --- тождественность ----------------------------------------------------
    lp = torch.randn(64)
    adv = torch.randn(64)
    t = ppo_terms(lp, lp.clone(), adv)
    t["ratio_cpu"] = t["ratio"].detach().numpy()
    assert abs(t["kl_k1"]) < 1e-12 and abs(t["kl_k3"]) < 1e-12
    assert t["clip_frac"] == 0.0
    assert check_identity(t)
    # отклонение ловится
    for delta, why in ((1e-3, "сдвиг правдоподобия"), (0.5, "крупный сдвиг")):
        t2 = ppo_terms(lp + delta, lp, adv)
        t2["ratio_cpu"] = t2["ratio"].detach().numpy()
        try:
            check_identity(t2)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"принято: {why}")

    # --- обрезание живое ----------------------------------------------------
    # КОНТРОЛЬ: без него «доля обрезанных 0%» нельзя отличить от мёртвой ветви.
    t3 = ppo_terms(lp + 1.0, lp, adv)          # ratio = e ~ 2.72
    assert t3["clip_frac"] == 1.0, t3["clip_frac"]
    t4 = ppo_terms(lp + 0.05, lp, adv)         # ratio ~ 1.05, внутри допуска
    assert t4["clip_frac"] == 0.0
    # обрезание ОГРАНИЧИВАЕТ потерю сверху при положительном преимуществе
    a_pos = torch.ones(8)
    big = ppo_terms(torch.zeros(8) + 3.0, torch.zeros(8), a_pos)
    assert abs(float(big["loss"]) + (1.0 + CLIP_EPS)) < 1e-5, float(big["loss"])
    # и НЕ ограничивает при отрицательном, если отношение велико
    a_neg = -torch.ones(8)
    bigneg = ppo_terms(torch.zeros(8) + 3.0, torch.zeros(8), a_neg)
    assert float(bigneg["loss"]) > 10.0, float(bigneg["loss"])

    # --- аналитический KL против torch.distributions -----------------------
    mo = torch.randn(8, 4, 3)
    so = torch.rand(8, 4, 3) * 0.5 + 0.05
    mn = mo + torch.randn_like(mo) * 0.1
    sn = so * 1.3
    got = analytic_kl(mo, so, mn, sn)
    ref = torch.distributions.kl_divergence(
        torch.distributions.Normal(mo, so),
        torch.distributions.Normal(mn, sn)).flatten(1).sum(-1)
    assert abs(got["joint_mean"] - float(ref.mean())) < 1e-5, got
    assert got["n_dim"] == 12
    assert abs(got["per_dim_mean"] * 12 - got["joint_mean"]) < 1e-9
    # ТОЖДЕСТВЕННЫЕ ПОЛИТИКИ -> KL РОВНО НОЛЬ
    z = analytic_kl(mo, so, mo.clone(), so.clone())
    assert abs(z["joint_mean"]) < 1e-9 and abs(z["joint_max"]) < 1e-9
    # KL РАСТЁТ С РАСХОЖДЕНИЕМ
    far = analytic_kl(mo, so, mo + 1.0, so)
    assert far["joint_mean"] > got["joint_mean"]

    # --- ОБНУЛЕНИЕ ОТНОШЕНИЯ — ОТКАЗ --------------------------------------
    # ГДЕ ИМЕННО ОБНУЛЯЕТСЯ. exp(-100) в float32 это денормал 3.8e-44, ещё
    # не нуль; ровно нуль начинается около -105. Настоящий прогон давал
    # |log_ratio| до 133, то есть попадал в эту область. Прежде обнуление
    # считалось «конечным числом», и вывод «переполнений нет» печатался.
    assert float(torch.exp(torch.tensor(-100.0))) > 0.0
    assert float(torch.exp(torch.tensor(-130.0))) == 0.0
    zero = ppo_terms(torch.zeros(4) - 130.0, torch.zeros(4), torch.ones(4))
    assert zero["ratio_finite"], "должно быть конечным"
    assert zero["ratio_zero_frac"] == 1.0, zero["ratio_zero_frac"]
    assert not zero["ratio_ok"]
    zero["ratio_cpu"] = zero["ratio"].detach().numpy()
    try:
        check_identity(zero)
    except SystemExit as e:
        assert "обнулились" in str(e), str(e)
    else:
        raise AssertionError("обнулённое отношение принято")
    assert zero["log_ratio_min"] == -130.0 and zero["log_ratio_max"] == -130.0

    # --- ВЫЗОВ ПРОЕКЦИИ В ОРКЕСТРАЦИИ -------------------------------------
    class FakeOpt:
        def __init__(self):
            self.n = 0

        def step(self):
            self.n += 1

    class FakeHead:
        def __init__(self):
            self.n = 0

        def project_log_std_(self):
            self.n += 1

    fo, fh = FakeOpt(), FakeHead()
    step_and_project(fo, fh)
    assert fo.n == 1 and fh.n == 1, (fo.n, fh.n)

    # --- переполнение ловится ----------------------------------------------
    huge = ppo_terms(torch.zeros(4) + 800.0, torch.zeros(4), torch.ones(4))
    assert not huge["ratio_finite"], "переполнение exp не замечено"
    assert not huge["ratio_ok"]
    t5 = dict(huge); t5["ratio_cpu"] = np.array([np.inf] * 4)
    try:
        check_identity(t5)
    except SystemExit:
        pass
    else:
        raise AssertionError("нечисловое отношение принято")
    assert huge["log_ratio_absmax"] == 800.0

    print("самопроверка k11i пройдена: тождественность до первого шага "
          "требуется и её\n  нарушение ловится; аналитический KL сходится с "
          "torch.distributions и равен\n  нулю у тождественных политик; "
          "ОБНУЛЕНИЕ отношения — отказ наравне с\n  переполнением; обрезание "
          "ограничивает потерю сверху при плюсовом преимуществе\n  и не "
          "ограничивает при минусовом; вызов проекции в оркестрации проверен")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--hicora-s0",
                    default="data/k11d/d1_mlp_coef_0.001_wd0_s0.pt")
    ap.add_argument("--hicora-s1",
                    default="data/k11d/d1_mlp_coef_0.001_wd0_s1.pt")
    ap.add_argument("--res-norm-cache", default="data/k11c_res_norm.pt",
                    help="ФИНАЛЬНАЯ НОРМА, снятая K-11c. Кэш K-11a хранит "
                         "СЫРОЙ отвод, а голова обучена на res_norm(h24): "
                         "без неё вход головы из другого распределения")
    ap.add_argument("--expect-target", default="coef")
    ap.add_argument("--train-log-std", action="store_true",
                    help="обучать sigma. ПО УМОЛЧАНИЮ ЗАМОРОЖЕНА на --sigma: "
                         "гейт K-11g проверил именно фиксированную 0.10, и "
                         "обучение самой sigma — отдельная абляция, а не "
                         "часть проверенного режима")
    ap.add_argument("--sigma", type=float, default=0.10,
                    help="выбранная гейтом K-11g начальная sigma")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip-eps", type=float, default=CLIP_EPS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/k11i/ppo_smoke.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    import torch
    import hicora_vla as hv            # noqa: F401
    import hicora_g as hg
    import k9h_multiarm_gate as k9h
    import k11a_build_hicora_cache as k11a
    import k11e_protocol as kp

    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    rng = np.random.default_rng(a.seed)

    # --- настоящий батч из кэша K-11a --------------------------------------
    h_path = a.cache + ".h24.npy"
    q_path = a.cache + ".q0hat.npy"
    cb_path = a.cache + ".codebooks.npy"
    for p in (h_path, q_path, cb_path):
        if not os.path.exists(p):
            raise SystemExit(f"нет {p}: батч брать неоткуда")
    if not os.path.exists(a.res_norm_cache):
        raise SystemExit(
            f"нет {a.res_norm_cache}. Кэш K-11a хранит СЫРОЙ отвод h24, а "
            f"голова обучена на\n  res_norm(h24). Без нормы вход головы "
            f"относится к другому распределению, и\n  любые KL и отношения "
            f"правдоподобий отсюда были бы диагностически неверны.")
    res_norm = torch.load(a.res_norm_cache, map_location=dev,
                          weights_only=False).to(dev).eval()
    for p_ in res_norm.parameters():
        p_.requires_grad_(False)
    rn_sha = k11a.state_sha1(res_norm)

    # --- ПРОВЕНАНС ДО ЗАГРУЗКИ ДАННЫХ --------------------------------------
    objs, head_sha = {}, {}
    for tag, path in (("s0", a.hicora_s0), ("s1", a.hicora_s1)):
        o = torch.load(path, map_location="cpu", weights_only=False)
        k9h.check_hicora_ckpt(o, f"hicora_{tag}", a.expect_target)
        head_sha[tag] = k9h.file_sha12(path)
        objs[tag] = o
    if head_sha["s0"] == head_sha["s1"]:
        raise SystemExit("обе головы — один файл: это не две реплики")
    kp.check_replication(kp.head_config(objs["s0"]), kp.head_config(objs["s1"]))
    for tag, o in objs.items():
        if os.path.abspath(o["cache"]) != os.path.abspath(a.cache):
            raise SystemExit(f"голова {tag} обучена на кэше {o['cache']}, а "
                             f"подан {a.cache}")
        if rn_sha != o["res_norm_sha1"]:
            raise SystemExit(
                f"res_norm sha {rn_sha}, а голова {tag} обучена на "
                f"{o['res_norm_sha1']}: вход головы был бы другим")
    b_sha = k9h.file_sha12(a.cache + ".basis.npy")
    r_sha = k9h.file_sha12(a.cache + ".rho.npy")
    for tag, o in objs.items():
        if b_sha != o["basis_sha1"] or r_sha != o["rho_sha1"]:
            raise SystemExit(f"базис/предел {b_sha}/{r_sha}, а голова {tag} "
                             f"обучена на {o['basis_sha1']}/{o['rho_sha1']}")
    print(f"  провенанс: головы {head_sha['s0']} / {head_sha['s1']}, мишень "
          f"{a.expect_target}, ранг {objs['s0']['rank']}, сиды "
          f"{objs['s0'].get('seed')}/{objs['s1'].get('seed')}")
    print(f"  res_norm sha {rn_sha} совпала у обеих голов; базис {b_sha}, "
          f"предел {r_sha}")

    H = np.load(h_path, mmap_mode="r")
    Q = np.load(q_path)
    E = np.load(cb_path)
    n_rows = H.shape[0]
    if a.batch > n_rows:
        raise SystemExit(f"в кэше {n_rows} строк, запрошено {a.batch}")
    idx = np.sort(rng.choice(n_rows, size=a.batch, replace=False))
    # ТОТ ЖЕ ПУТЬ ПО ТИПАМ, ЧТО В K-11c: срез берётся в fp16, как он лежит в
    # кэше, норма применяется к нему, и только результат переводится в fp32.
    h_raw = torch.as_tensor(np.asarray(H[idx]), dtype=torch.float16,
                            device=dev)
    with torch.no_grad():
        h24 = res_norm(h_raw).float()
    q0 = torch.as_tensor(np.asarray(Q[idx]), dtype=torch.long, device=dev)
    Et = torch.as_tensor(E, dtype=torch.float32, device=dev)
    z0 = Et[0][q0]
    print(f"  батч из кэша: {a.batch} строк из {n_rows}, сырой h24 "
          f"{tuple(h_raw.shape)} {h_raw.dtype} -> res_norm -> "
          f"{tuple(h24.shape)}, z0 {tuple(z0.shape)}")
    # НОРМА ДОЛЖНА БЫТЬ ДЕЙСТВИТЕЛЬНО ПРИМЕНЕНА. Совпадение sha говорит лишь
    # о том, что взята ТА норма; если вызов убрать, sha всё равно сойдётся, а
    # голова получит сырой отвод — ровно та ошибка, из-за которой пришлось
    # снять числа первого прогона.
    if torch.equal(h24, h_raw.float()):
        raise SystemExit(
            "res_norm не изменила вход: либо вызов пропущен, либо норма "
            "тождественна.\n  Голова обучена на res_norm(h24), и сырой отвод "
            "относится к другому\n  распределению.")
    print(f"  после нормы: среднее {float(h24.mean()):+.4f}, стд "
          f"{float(h24.std()):.4f} (до нормы "
          f"{float(h_raw.float().mean()):+.4f} / "
          f"{float(h_raw.float().std()):.4f})")

    # --- обе головы --------------------------------------------------------
    heads = {}
    for tag, path in (("s0", a.hicora_s0), ("s1", a.hicora_s1)):
        o = objs[tag]
        B = np.load(a.cache + ".basis.npy").astype(np.float32)
        rho = np.load(a.cache + ".rho.npy").astype(np.float32)
        Gh = hg.make_gaussian_residual_head()
        rank_ = int(o["rank"])
        h_ = Gh(h24.shape[-1], int(Et.shape[-1]), rank=rank_,
                hidden=int(o.get("hidden", 512)),
                proj=int(o.get("proj", 64))).to(dev)
        h_.set_basis(torch.as_tensor(B))
        h_.set_rho(torch.as_tensor(rho))
        st = {k[len("hicora_head."):]: v for k, v in o["state"].items()}
        want = {k for k in h_.state_dict() if k.startswith(("proj.", "net."))}
        if set(st) != want:
            raise SystemExit(f"набор весов головы {tag} не совпал")
        with torch.no_grad():
            for k, v in st.items():
                h_.state_dict()[k].copy_(v.to(dev, torch.float32))
            h_.log_std.fill_(math.log(a.sigma))
        std = float(h_.std().max())
        if abs(std - a.sigma) > 1e-6:
            raise SystemExit(f"std {std} не равна sigma {a.sigma}")
        h_.train()
        heads[tag] = h_
        print(f"  голова {tag}: ранг {o['rank']}, сид {o['seed']}, "
              f"std {std:.6f}")

    out = dict(sigma=a.sigma, batch=a.batch, minibatch=a.minibatch,
               epochs=a.epochs, lr=a.lr, clip_eps=a.clip_eps,
               cache=a.cache, rows=int(n_rows), seed=a.seed,
               idx=[int(x) for x in idx],
               script_sha1=k9h.file_sha12(os.path.abspath(__file__)),
               hicora_g_sha1=k9h.file_sha12(hg.__file__),
               hicora_vla_sha1=k9h.file_sha12(hv.__file__),
               head_sha1=head_sha, res_norm_sha1=rn_sha, basis_sha1=b_sha,
               rho_sha1=r_sha, rank=int(objs["s0"]["rank"]),
               target=a.expect_target, heads={})

    # --- ОБЩИЕ БУФЕРЫ, НЕ ЗАВИСЯЩИЕ НИ ОТ ГОЛОВЫ, НИ ОТ ЧИСЛА ЭПОХ ----------
    # Прежде eps, преимущества и порядок минибатчей брались из общего потока,
    # и число эпох у s0 меняло состояние ГСЧ перед s1: сравнения голов и
    # режимов не были парными. Здесь всё выводится из ОТДЕЛЬНОГО генератора,
    # засеянного только --seed, поэтому при любом (epochs, lr) и для любой
    # головы буферы совпадают побитово.
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(a.seed) * 1000003 + 17)
    eps_shared = torch.empty(a.batch, h24.shape[1], rank_, device=dev,
                             dtype=torch.float32).normal_(generator=gen)
    adv = torch.empty(a.batch, device=dev,
                      dtype=torch.float32).normal_(generator=gen)
    # НОРМИРОВКА ОДИН РАЗ ПО ВСЕМУ БАТЧУ, а не по минибатчу: при редких
    # бинарных наградах минибатч легко однороден и теряет весь сигнал.
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    perms = [torch.randperm(a.batch, generator=gen, device=dev)
             for _ in range(a.epochs)]
    eps_sha = hashlib.sha1(np.ascontiguousarray(
        eps_shared.cpu().numpy()).tobytes()).hexdigest()[:16]
    adv_sha = hashlib.sha1(np.ascontiguousarray(
        adv.cpu().numpy()).tobytes()).hexdigest()[:16]
    perm_sha = hashlib.sha1(b"".join(
        np.ascontiguousarray(p.cpu().numpy()).tobytes()
        for p in perms)).hexdigest()[:16]
    print(f"  общие буферы: eps sha {eps_sha}, преимущества {adv_sha}, "
          f"перестановки {perm_sha}")
    out.update(eps_sha1=eps_sha, adv_sha1=adv_sha, perm_sha1=perm_sha,
               train_log_std=bool(a.train_log_std))

    def measure(head, mu_old, std_old, u_buf, logp_old):
        """Полный замер НА ВСЁМ буфере. Вызывается до первого шага и ПОСЛЕ
        КАЖДОГО.

        ЗАЧЕМ НА ВСЁМ БУФЕРЕ И ПОСЛЕ ШАГА. Прежде и ppo_terms, и analytic_kl
        считались ПЕРЕД step_and_project и только на текущем минибатче:
        последний шаг не измерялся вовсе, а «KL эпохи» был KL одного минибатча
        ДО его обновления. Из-за этого скачок отношения приписывался второй
        эпохе, хотя сдвиг мог быть сделан первой и просто обнаружен позже.
        """
        with torch.no_grad():
            o = head(h24, z0, u=u_buf)
            t = ppo_terms(o["log_prob_u"], logp_old, adv, a.clip_eps)
            std_new = o["std"].expand_as(o["mu"]).contiguous()
            akl = analytic_kl(mu_old, std_old, o["mu"], std_new)
            # ВЫРАВНИВАНИЕ СЧИТАЕТСЯ В СКРИПТЕ, а не одноразовой командой:
            # косинус между сдвигом среднего и отклонением сохранённого
            # действия — то, что объясняет хвост отношения.
            d = (o["mu"] - mu_old).flatten(1)
            e = (u_buf - mu_old).flatten(1)
            dn = d.norm(dim=-1)
            en = e.norm(dim=-1)
            cos = ((d * e).sum(-1) / (dn * en + 1e-30))
            lr_abs = t["log_ratio"].abs()
            q = torch.quantile(lr_abs.float(),
                               torch.tensor([0.5, 0.9, 0.99], device=dev))
            # ОТКУДА ХВОСТ: из выравнивания d с e или из НЕОДНОРОДНОСТИ ||d||
            # по состояниям? Средний косинус этого не различает. Здесь
            # считаются разброс ||d||, корреляция |log_ratio| с покоординатным
            # KL состояния, и выравнивание ИМЕННО у худшего процента примеров.
            dq = torch.quantile(dn.float(),
                                torch.tensor([0.5, 0.9, 0.99], device=dev))
            kl_s = (torch.log(std_new / std_old)
                    + (std_old ** 2 + (mu_old - o["mu"]) ** 2)
                    / (2 * std_new ** 2) - 0.5).flatten(1).sum(-1)
            def _corr(x, y):
                x = x.float() - x.float().mean()
                y = y.float() - y.float().mean()
                return float((x * y).sum()
                             / (x.norm() * y.norm() + 1e-30))
            k_top = max(1, int(0.01 * lr_abs.numel()))
            top = torch.topk(lr_abs.float(), k_top).indices
        return dict(kl_exact=akl["joint_mean"], kl_exact_max=akl["joint_max"],
                    kl_per_dim=akl["per_dim_mean"], n_dim=akl["n_dim"],
                    kl_k1=t["kl_k1"], kl_k3=t["kl_k3"],
                    clip_frac=t["clip_frac"],
                    log_ratio_min=t["log_ratio_min"],
                    log_ratio_max=t["log_ratio_max"],
                    log_ratio_absmax=t["log_ratio_absmax"],
                    log_ratio_q50=float(q[0]), log_ratio_q90=float(q[1]),
                    log_ratio_q99=float(q[2]),
                    ratio_zero_frac=t["ratio_zero_frac"],
                    ratio_ok=t["ratio_ok"],
                    align_mean=float(cos.mean()), align_max=float(cos.max()),
                    align_top1pct=float(cos[top].mean()),
                    d_norm_mean=float(dn.mean()),
                    d_norm_q50=float(dq[0]), d_norm_q90=float(dq[1]),
                    d_norm_q99=float(dq[2]),
                    d_norm_ratio_q99_q50=float(dq[2] / (dq[0] + 1e-30)),
                    kl_state_q99=float(torch.quantile(
                        kl_s.float(), torch.tensor(0.99, device=dev))),
                    corr_logratio_kl=_corr(lr_abs, kl_s),
                    corr_logratio_dnorm=_corr(lr_abs, dn),
                    std_mean=float(head.std().mean()))

    for tag, head in heads.items():
        print(f"\n=== голова {tag} ===")
        with torch.no_grad():
            head.log_std.fill_(math.log(a.sigma))
        head.log_std.requires_grad_(bool(a.train_log_std))
        # --- «раскатка» на ОБЩЕМ eps ---------------------------------------
        with torch.no_grad():
            o_roll = head(h24, z0, deterministic=True)
            mu_old = o_roll["mu"].detach().clone()
            std_old = head.std().detach().expand_as(mu_old).contiguous()
            u_buf = (mu_old + float(a.sigma) * eps_shared).detach().clone()
            logp_old = head.log_prob_u(u_buf, mu_old, std_old).detach().clone()
        assert not u_buf.requires_grad and not logp_old.requires_grad

        # --- ТОЖДЕСТВЕННОСТЬ ДО ПЕРВОГО ШАГА ------------------------------
        o_re = head(h24, z0, u=u_buf)
        assert o_re["u"].grad_fn is None, "сохранённое u попало в граф"
        t0 = ppo_terms(o_re["log_prob_u"], logp_old, adv, a.clip_eps)
        t0["ratio_cpu"] = t0["ratio"].detach().cpu().numpy()
        check_identity(t0)
        m0 = measure(head, mu_old, std_old, u_buf, logp_old)
        if abs(m0["kl_exact"]) > 1e-9 or m0["clip_frac"] != 0.0:
            raise SystemExit(f"замер до шага не тождественен: KL "
                             f"{m0['kl_exact']:.3e}, обрезано "
                             f"{m0['clip_frac']}")
        print(f"  до шага: отношение 1 +- "
              f"{np.abs(t0['ratio_cpu'] - 1).max():.2e}, точный KL "
              f"{m0['kl_exact']:.2e}, обрезано {100 * m0['clip_frac']:.1f}%, "
              f"выравнивание {m0['align_mean']:.3f}")
        with torch.no_grad():
            again = head(h24, z0, u=u_buf)["log_prob_u"]
        assert torch.allclose(again, o_re["log_prob_u"].detach(), atol=0), \
            "реплей не детерминирован"
        with torch.no_grad():
            from_cpu = head(h24, z0, u=u_buf.detach().cpu())["log_prob_u"]
        assert from_cpu.device == again.device and \
            torch.allclose(from_cpu, again, atol=0), \
            "реплей с CPU-буфера дал другое правдоподобие"
        head.zero_grad(set_to_none=True)
        t0["loss"].backward()
        assert torch.isfinite(head.net[-1].weight.grad).all()
        assert head.basis.grad is None and head.rho.grad is None
        if a.train_log_std:
            assert head.log_std.grad is not None

        # --- обновление с ЗАМЕРОМ ПОСЛЕ КАЖДОГО ШАГА ----------------------
        trainable = [p_ for n_, p_ in head.named_parameters()
                     if n_.startswith(("proj.", "net."))
                     or (n_ == "log_std" and a.train_log_std)]
        opt = torch.optim.Adam(trainable, lr=a.lr)
        net_before = head.net[-1].weight.detach().clone()
        post, step_i = [], 0
        for ep in range(a.epochs):
            for s_ in range(0, a.batch, a.minibatch):
                sl = perms[ep][s_:s_ + a.minibatch]
                o_mb = head(h24[sl], z0[sl], u=u_buf[sl])
                t = ppo_terms(o_mb["log_prob_u"], logp_old[sl], adv[sl],
                              a.clip_eps)
                if not t["ratio_ok"]:
                    why = ("переполнение" if not t["ratio_finite"]
                           else f"обнуление у "
                                f"{100 * t['ratio_zero_frac']:.1f}%")
                    raise SystemExit(
                        f"ОТНОШЕНИЕ НЕГОДНО на шаге {step_i} ({why}): "
                        f"log_ratio в\n  [{t['log_ratio_min']:.1f}, "
                        f"{t['log_ratio_max']:.1f}].")
                opt.zero_grad(set_to_none=True)
                t["loss"].backward()
                gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1e9)
                if not torch.isfinite(gnorm):
                    raise SystemExit(f"нечисловой градиент на шаге {step_i}")
                step_and_project(opt, head)
                mp = measure(head, mu_old, std_old, u_buf, logp_old)
                mp.update(step=step_i, epoch=ep, grad_norm=float(gnorm),
                          mb_clip_frac_pre=t["clip_frac"])
                post.append(mp)
                step_i += 1
                if not mp["ratio_ok"]:
                    raise SystemExit(
                        f"ПОСЛЕ шага {step_i - 1} отношение негодно: "
                        f"обнулилось у {100 * mp['ratio_zero_frac']:.1f}%, "
                        f"log_ratio\n  в [{mp['log_ratio_min']:.1f}, "
                        f"{mp['log_ratio_max']:.1f}]. Параметры уже изменены: "
                        f"для настоящего RL\n  здесь нужен откат состояния "
                        f"головы и Adam, а не только остановка.")
        fin = post[-1]
        print(f"  ПОСЛЕ {len(post)} шагов, замер на ВСЁМ буфере: точный KL "
              f"{fin['kl_exact']:.4g} (макс по состоянию "
              f"{fin['kl_exact_max']:.4g}),\n    обрезано "
              f"{100 * fin['clip_frac']:.1f}%, |log_ratio| медиана "
              f"{fin['log_ratio_q50']:.3f} q90 {fin['log_ratio_q90']:.3f} "
              f"q99 {fin['log_ratio_q99']:.3f} макс "
              f"{fin['log_ratio_absmax']:.3f},\n    выравнивание "
              f"{fin['align_mean']:.3f} (макс {fin['align_max']:.3f}), "
              f"k1 {fin['kl_k1']:.4g} k3 {fin['kl_k3']:.4g}")
        print(f"    ОТКУДА ХВОСТ: ||d|| q50 {fin['d_norm_q50']:.4f} q99 "
              f"{fin['d_norm_q99']:.4f} (отношение "
              f"{fin['d_norm_ratio_q99_q50']:.1f}x), выравнивание у худшего "
              f"процента {fin['align_top1pct']:+.3f},\n    корреляция "
              f"|log_ratio| с KL состояния {fin['corr_logratio_kl']:+.3f}, "
              f"с ||d|| {fin['corr_logratio_dnorm']:+.3f}")
        print(f"  по шагам обрезано: "
              + " ".join(f"{100 * m['clip_frac']:.0f}%" for m in post))
        d_net = float((head.net[-1].weight.detach() - net_before).abs().max())
        if d_net == 0.0:
            raise SystemExit("средняя ветвь не сдвинулась: путь "
                             "score-function мёртв")
        print(f"  сдвиг последнего слоя {d_net:.3e}; sigma "
              f"{'обучалась' if a.train_log_std else 'ЗАМОРОЖЕНА'} на "
              f"{fin['std_mean']:.6f}")
        out["heads"][tag] = dict(identity=m0, post=post, d_net_last=d_net,
                                 head_sha1=head_sha[tag])

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {a.out}")
    print("\n  PPO SMOKE ПРОЙДЕН на обеих головах: отношение тождественно до "
          "первого шага,\n  реплей детерминирован (включая буфер с CPU), "
          "градиент идёт через среднюю ветвь,\n  отношение годно ПОСЛЕ "
          "каждого шага, замер делается на ВСЁМ буфере.")
    print("  ЭТО МЕХАНИКА, НЕ ОБУЧЕНИЕ: преимущества поддельные, наград нет. "
          "Ни одного\n  утверждения об улучшении политики отсюда не следует, "
          "и выбирать предохранители\n  по одному сиду нельзя — нужен "
          "перебор по нескольким --seed.")


if __name__ == "__main__":
    main()
