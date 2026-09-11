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
        h_ = Gh(h24.shape[-1], int(Et.shape[-1]), rank=int(o["rank"]),
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

    for tag, head in heads.items():
        print(f"\n=== голова {tag} ===")
        # --- «раскатка»: сэмпл и сохранение в буфер -------------------------
        with torch.no_grad():
            o_roll = head(h24, z0)
        u_buf = o_roll["u"].detach().clone()
        logp_old = o_roll["log_prob_u"].detach().clone()
        # mu И std ПОЛИТИКИ СБОРА нужны для АНАЛИТИЧЕСКОГО KL: оценки k1/k3
        # при катастрофических отношениях расходятся на порядки и перестают
        # что-либо измерять.
        mu_old = o_roll["mu"].detach().clone()
        std_old = o_roll["std"].detach().clone().expand_as(mu_old).contiguous()
        assert not u_buf.requires_grad and not logp_old.requires_grad
        # ПОДДЕЛЬНЫЕ ПРЕИМУЩЕСТВА: награды здесь нет. Нормированы, как принято.
        adv = torch.as_tensor(rng.normal(0, 1, size=a.batch),
                              dtype=torch.float32, device=dev)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # --- ПРОВЕРКА ТОЖДЕСТВЕННОСТИ ДО ПЕРВОГО ШАГА ----------------------
        o_re = head(h24, z0, u=u_buf)
        assert o_re["u"].grad_fn is None, "сохранённое u попало в граф"
        t0 = ppo_terms(o_re["log_prob_u"], logp_old, adv, a.clip_eps)
        t0["ratio_cpu"] = t0["ratio"].detach().cpu().numpy()
        check_identity(t0)
        print(f"  до шага: отношение 1 +- "
              f"{np.abs(t0['ratio_cpu'] - 1).max():.2e}, KL k1 "
              f"{t0['kl_k1']:.2e} k3 {t0['kl_k3']:.2e}, обрезано "
              f"{100 * t0['clip_frac']:.1f}%")
        # РЕПЛЕЙ ДЕТЕРМИНИРОВАН
        with torch.no_grad():
            again = head(h24, z0, u=u_buf)["log_prob_u"]
        assert torch.allclose(again, o_re["log_prob_u"].detach(), atol=0), \
            "реплей не детерминирован"
        # И ПУТЬ «БУФЕР НА CPU -> КАРТА» ПРОВЕРЯЕТСЯ ЯВНО: именно так действие
        # приходит из буфера rollout, и заявлять этот путь, не исполнив его,
        # было бы утверждением без проверки.
        with torch.no_grad():
            from_cpu = head(h24, z0, u=u_buf.detach().cpu())["log_prob_u"]
        assert from_cpu.device == again.device, "результат ушёл не на ту карту"
        assert torch.allclose(from_cpu, again, atol=0), \
            "реплей с CPU-буфера дал другое правдоподобие"
        # ГРАДИЕНТ ТОЛЬКО ЧЕРЕЗ mu И log_std
        head.zero_grad(set_to_none=True)
        t0["loss"].backward()
        assert head.log_std.grad is not None
        assert torch.isfinite(head.log_std.grad).all()
        g_net = head.net[-1].weight.grad
        assert g_net is not None and torch.isfinite(g_net).all()
        assert head.basis.grad is None and head.rho.grad is None

        # --- собственно обновление -----------------------------------------
        log_std_before = head.log_std.detach().clone()
        net_before = head.net[-1].weight.detach().clone()
        opt = torch.optim.Adam(
            [p for n_, p in head.named_parameters()
             if n_.startswith(("proj.", "net.")) or n_ == "log_std"], lr=a.lr)
        steps = []
        for ep in range(a.epochs):
            order = torch.randperm(a.batch, device=dev)
            for s_ in range(0, a.batch, a.minibatch):
                sl = order[s_:s_ + a.minibatch]
                o_mb = head(h24[sl], z0[sl], u=u_buf[sl])
                t = ppo_terms(o_mb["log_prob_u"], logp_old[sl], adv[sl],
                              a.clip_eps)
                if not t["ratio_ok"]:
                    why = ("переполнение" if not t["ratio_finite"]
                           else f"обнуление у "
                                f"{100 * t['ratio_zero_frac']:.1f}% примеров")
                    raise SystemExit(
                        f"ОТНОШЕНИЕ ПРАВДОПОДОБИЙ НЕГОДНО на эпохе {ep} "
                        f"({why}): log_ratio в\n  "
                        f"[{t['log_ratio_min']:.1f}, {t['log_ratio_max']:.1f}]."
                        f" При 512 координатах оно расходится быстро — нужен\n"
                        f"  меньший lr, одна эпоха на батч и остановка по KL.")
                opt.zero_grad(set_to_none=True)
                t["loss"].backward()
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in head.parameters() if p.requires_grad], 1e9)
                if not torch.isfinite(gnorm):
                    raise SystemExit(f"нечисловой градиент на эпохе {ep}")
                with torch.no_grad():
                    akl = analytic_kl(mu_old[sl], std_old[sl],
                                      o_mb["mu"].detach(),
                                      o_mb["std"].expand_as(
                                          o_mb["mu"]).contiguous())
                step_and_project(opt, head)
                ls = head.log_std.detach()
                assert float(ls.max()) <= head.LOG_STD_MAX, "log_std выше предела"
                assert float(ls.min()) >= head.LOG_STD_MIN, "log_std ниже предела"
                steps.append(dict(epoch=ep, kl_k1=t["kl_k1"], kl_k3=t["kl_k3"],
                                  kl_exact=akl["joint_mean"],
                                  kl_exact_max=akl["joint_max"],
                                  kl_per_dim=akl["per_dim_mean"],
                                  n_dim=akl["n_dim"],
                                  clip_frac=t["clip_frac"],
                                  log_ratio_absmax=t["log_ratio_absmax"],
                                  log_ratio_min=t["log_ratio_min"],
                                  log_ratio_max=t["log_ratio_max"],
                                  ratio_zero_frac=t["ratio_zero_frac"],
                                  log_ratio_std=t["log_ratio_std"],
                                  grad_norm=float(gnorm),
                                  loss=float(t["loss"].detach()),
                                  std_mean=float(head.std().mean())))
        last = steps[-1]
        print(f"  шагов {len(steps)}; последний: ТОЧНЫЙ KL "
              f"{last['kl_exact']:.4g} по {last['n_dim']} координатам "
              f"({last['kl_per_dim']:.3g} на координату),\n    оценки k1 "
              f"{last['kl_k1']:.4g} k3 {last['kl_k3']:.4g}, обрезано "
              f"{100 * last['clip_frac']:.1f}%, log_ratio в "
              f"[{last['log_ratio_min']:.2f}, {last['log_ratio_max']:.2f}], "
              f"норма градиента {last['grad_norm']:.3e}")
        ever = max(s["clip_frac"] for s in steps)
        kl_max = max(s["kl_k3"] for s in steps)
        # ДИАГНОСТИКА, НЕ ОТКАЗ: преимущества здесь поддельные, поэтому
        # «слишком агрессивное обновление» о настоящем PPO не говорит. Но
        # цифру надо видеть: при 512 координатах доля обрезанных легко
        # уходит к единице, и тогда настоящему PPO понадобится меньший lr,
        # меньше эпох на батч или ранняя остановка по KL.
        if ever > 0.5 or kl_max > 0.05:
            print(f"  ВНИМАНИЕ: обрезано до {100 * ever:.0f}%, KL k3 до "
                  f"{kl_max:.4g}, |log_ratio| до "
                  f"{max(s_['log_ratio_absmax'] for s_ in steps):.1f}.")
            if kl_max > 1.0:
                print("    ЭТО НЕ ПРОСТО МНОГО: при 512 случайных координатах "
                      "изменение mu на\n    сотые доли множится по всем "
                      "координатам, и отношение правдоподобий\n    либо "
                      "обнуляется, либо взрывается. Обновление считается по "
                      "данным,\n    которых политика уже не порождает.")
                print("    Кандидатный режим для настоящего PPO: ОДНА эпоха "
                      "на батч, lr на\n    порядок меньше, ранняя остановка "
                      "по ТОЧНОМУ KL с порогом 0.01-0.02.")
                print("    Преимущества нормировать ПО ВСЕМУ батчу раскатки "
                      "(или по группе\n    эпизодов одного начального "
                      "состояния), а затем делить на минибатчи БЕЗ\n    "
                      "повторной нормировки: при редких бинарных наградах "
                      "минибатч легко\n    оказывается однородным и теряет "
                      "весь сигнал.")
                print("    Это КАНДИДАТЫ, а не подтверждённые настройки: их "
                      "надо прогнать отдельно\n    и потребовать точный KL в "
                      "пределах порога при умеренной доле обрезанных.")
            else:
                print("    Для настоящего PPO это сигнал уменьшить lr или "
                      "число эпох на батч\n    и добавить остановку по KL.")
            print("    Здесь преимущества ПОДДЕЛЬНЫЕ, поэтому это не отказ, "
                  "а величина к сведению.")
        # ПАРАМЕТРЫ ОБЯЗАНЫ СДВИНУТЬСЯ. Иначе «обучение прошло без ошибок»
        # неотличимо от обучения, которое ничего не меняло: при lr=1e-4 сдвиг
        # log_std порядка 1e-3 не виден в четырёх знаках, и без явной проверки
        # замерший параметр выглядел бы успехом.
        d_ls = float((head.log_std.detach() - log_std_before).abs().max())
        d_net = float((head.net[-1].weight.detach() - net_before).abs().max())
        if d_ls == 0.0:
            raise SystemExit(
                "log_std НЕ СДВИНУЛСЯ за весь прогон: либо он не в "
                "оптимизаторе, либо упёрся\n  в границу, где градиент clamp "
                "нулевой — ровно то, против чего нужна project_log_std_.")
        if d_net == 0.0:
            raise SystemExit("средняя ветвь не сдвинулась: путь "
                             "score-function мёртв")
        print(f"  sigma прошла {a.sigma:.6f} -> {last['std_mean']:.6f}; "
              f"сдвиг log_std {d_ls:.3e}, последнего слоя {d_net:.3e}")
        print(f"  log_std в пределах [{head.LOG_STD_MIN}, "
              f"{head.LOG_STD_MAX}], фактически "
              f"[{float(head.log_std.min()):.4f}, "
              f"{float(head.log_std.max()):.4f}]")
        # КОНТРОЛЬ НЕОБХОДИМОСТИ ПРОЕКЦИИ. В коротком прогоне log_std не
        # покидает диапазон, поэтому project_log_std_ оказывается пустой
        # операцией, и её удаление не проявилось бы ни в одном числе.
        #
        # ПОТЕРЯ ЗДЕСЬ ПРОСТАЯ, А ВЫНОС МАЛЫЙ, И ЭТО СУЩЕСТВЕННО. Первая
        # версия брала потерю PPO и выносила log_std на +5, то есть sigma с
        # 0.1 до ~2.7. Тогда logp_new - logp_old ~ -512*ln(27) ~ -1700,
        # отношение обнуляется в fp32 ТОЧНО, обрезанный член становится
        # константой, и градиент нулевой НЕ из-за clamp, а из-за исчезновения
        # отношения. Контроль путал две причины и на малых размерностях
        # проходил, а на настоящих 512 координатах ложно отказывал.
        with torch.no_grad():
            keep = head.log_std.detach().clone()
            head.log_std.fill_(head.LOG_STD_MAX + 0.5)
        head.zero_grad(set_to_none=True)
        (-head(h24, z0, u=u_buf)["log_prob_u"].mean()).backward()
        g_out = float(head.log_std.grad.abs().max())
        head.project_log_std_()
        inside = float(head.log_std.max())
        head.zero_grad(set_to_none=True)
        (-head(h24, z0, u=u_buf)["log_prob_u"].mean()).backward()
        g_in = float(head.log_std.grad.abs().max())
        with torch.no_grad():
            head.log_std.copy_(keep)
        if g_out != 0.0:
            raise SystemExit(
                f"за границей градиент log_std равен {g_out:.3e}, а не нулю: "
                f"значит clamp\n  пропускает градиент, и проекция не нужна — "
                f"либо контроль недействителен.")
        if not (inside < head.LOG_STD_MAX and g_in > 0.0):
            raise SystemExit(
                "ПРОЕКЦИЯ НЕ ВОЗВРАЩАЕТ ОБУЧЕНИЕ: после выноса log_std за "
                "границу\n  градиент не восстановился. Без этого sigma "
                "замерла бы навсегда.")
        print(f"  контроль проекции (потеря -log pi, вынос +0.5): за границей "
              f"градиент {g_out:.1e},\n    после проекции {g_in:.3e} при "
              f"log_std {inside:.4f}")

        # КОНТРОЛЬ: ветвь обрезания должна быть ЖИВОЙ. Если за весь прогон
        # ничего не обрезалось, «обрезано 0%» неотличимо от мёртвого кода.
        probe = None
        if ever == 0.0:
            with torch.no_grad():
                saved = head.log_std.detach().clone()
                head.log_std.add_(0.7)          # заметно иная политика
                o_p = head(h24, z0, u=u_buf)
            tp = ppo_terms(o_p["log_prob_u"], logp_old, adv, a.clip_eps)
            probe = dict(clip_frac=tp["clip_frac"],
                         log_ratio_absmax=tp["log_ratio_absmax"])
            with torch.no_grad():
                head.log_std.copy_(saved)
            if tp["clip_frac"] == 0.0:
                raise SystemExit(
                    "ВЕТВЬ ОБРЕЗАНИЯ МЁРТВА: даже при явно иной политике "
                    "ничего не обрезалось.")
            print(f"  контроль обрезания: при сдвиге log_std на +0.7 "
                  f"обрезано {100 * tp['clip_frac']:.1f}%")
        out["heads"][tag] = dict(
            identity=dict(ratio_dev=float(np.abs(t0["ratio_cpu"] - 1).max()),
                          kl_k1=t0["kl_k1"], kl_k3=t0["kl_k3"],
                          clip_frac=t0["clip_frac"]),
            steps=steps, clip_ever=ever, clip_probe=probe,
            kl_max=kl_max, d_log_std=d_ls, d_net_last=d_net,
            proj_grad_outside=g_out, proj_grad_inside=g_in,
            log_std_range=[float(head.log_std.min()),
                           float(head.log_std.max())],
            sigma_end=last["std_mean"])

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {a.out}")
    print("\n  PPO SMOKE ПРОЙДЕН на обеих головах: отношение тождественно до "
          "первого шага,\n  реплей детерминирован, градиент идёт через mu и "
          "log_std, переполнений нет,\n  log_std остаётся в пределах после "
          "проекции, ветвь обрезания жива.")
    print("  ЭТО МЕХАНИКА, НЕ ОБУЧЕНИЕ: преимущества здесь поддельные, "
          "наград нет.\n  Ни одного утверждения об улучшении политики отсюда "
          "не следует.")


if __name__ == "__main__":
    main()
