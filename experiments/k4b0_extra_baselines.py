"""K-4b0.1: причинные baseline равной стоимости и разложение разрыва.

ЗАЧЕМ. До обучения router надо зафиксировать, что именно он обязан обогнать, и
понять, из чего складывается разрыв 0.291 -> 0.701. Скрипт работает по
сохранённым features.npz и labels.npz, VLM не запускается.

ДВЕ ЗАДАЧИ.

1. BASELINE РАВНОЙ СТОИМОСТИ. Политика «только p» тратит одну позицию, а
   энтропийная — четыре, поэтому напрямую они несравнимы. Добавляются гибриды
   «p + три по признаку» и обучаемые по train таблицы. Итог — два числа:
       B_heur  — лучшая НЕОБУЧАЕМАЯ эвристика,
       B_prior — лучшая обученная по train таблица (тоже deployable).
   Обе выбираются ТОЛЬКО по validation, test печатается для уже выбранной.

2. РАЗЛОЖЕНИЕ РАЗРЫВА. Позиции вне changed-support дают ровно ноль, внутри
   support примерно треть даёт ОТРИЦАТЕЛЬНЫЙ выигрыш. Поэтому задача router
   распадается на «найти support» и «упорядочить внутри него», и надо знать,
   сколько стоит каждая половина. Считаются пять величин, см. decomposition().

АСИММЕТРИЯ, КОТОРУЮ ЛЕГКО ПОТЕРЯТЬ. Одиночный оракул O1 ранжирует все 16
позиций и предпочтёт БЕЗОПАСНЫЙ НОЛЬ вне support вредной позиции внутри него.
Поэтому «истинные gains, но принудительно четыре внутри support» — это НЕ O1, а
величина строго не выше. Разница и есть цена того, что идеальный классификатор
support сам по себе вредит.

Запуск:
    python3 experiments/k4b0_extra_baselines.py --dir data/k4b0_v2
    python3 experiments/k4b0_extra_baselines.py --selftest
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k4b0_verify import make_gmap, g_of, to_rms, cluster_ci, macro_by  # noqa: E402


def build_tables(lb, idx, P, kmax):
    """Таблицы G один раз на часть: их переиспользуют все политики.

    Без этого каждая из полутора десятков политик распаковывала бы рваный
    массив заново, а это и есть основная стоимость скрипта."""
    out = []
    for j, i in enumerate(idx):
        if j % 2000 == 0:
            print(f"    таблицы: {j}/{len(idx)}", flush=True)
        out.append(make_gmap(lb, int(i), P, kmax))
    return out


def eval_sets(tables, e0, idx, sets):
    """Выигрыш выбранных наборов на RMS-шкале: числитель и знаменатель.

    Возвращает пару массивов, а не готовое отношение, потому что интервал
    считается кластерным бутстрапом по эпизодам через частичные суммы."""
    num = np.empty(len(idx))
    for j, i in enumerate(idx):
        gmap, C = tables[j]
        num[j] = to_rms(e0[i], g_of(gmap, C, sets[j]))
    return num, np.sqrt(e0[idx])


def topk_sets(score, K, forced=None):
    """Top-K по убыванию score, детерминированно при ничьих.

    forced — позиция, которую политика обязана взять первой (для гибридов
    «p + три по признаку»). Остальные слоты добираются по score.
    """
    n, P = score.shape
    out = np.empty((n, K), np.int64)
    # argsort стабилен при kind="stable": ничьи разрешаются меньшим индексом
    # позиции, а не порядком, зависящим от реализации сортировки.
    order = np.argsort(-score, axis=1, kind="stable")
    if forced is None:
        return order[:, :K]
    for i in range(n):
        f = int(forced[i])
        rest = [q for q in order[i] if q != f]
        out[i] = [f] + rest[:K - 1]
    return out


def zrow(x):
    """Нормировка ВНУТРИ строки: комбинировать энтропию с индикатором можно
    только на общей шкале, а разброс энтропии сильно разный между строками."""
    mu = x.mean(1, keepdims=True)
    sd = x.std(1, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-8)


def fit_priors(lb, ft, tsk_idx, tr, P):
    """Таблицы среднего одиночного выигрыша, обученные ТОЛЬКО по train.

    Метка берётся нормированной (sing_gain_norm): иначе несколько строк с
    крупной исходной ошибкой определили бы всю таблицу.
    """
    y = lb["sing_gain_norm"][tr]
    p = lb["p"][tr].astype(np.int64)
    glob = y.mean(0)                                   # score[q]
    pq = np.zeros((P, P))
    cnt = np.zeros(P)
    for j in range(len(tr)):
        pq[p[j]] += y[j]
        cnt[p[j]] += 1
    pq /= np.maximum(cnt, 1)[:, None]

    off = np.zeros(2 * P - 1)
    ocnt = np.zeros(2 * P - 1)
    for j in range(len(tr)):
        for q in range(P):
            off[q - p[j] + P - 1] += y[j, q]
            ocnt[q - p[j] + P - 1] += 1
    off /= np.maximum(ocnt, 1)

    nt = int(tsk_idx.max()) + 1
    tpq = np.zeros((nt, P, P))
    tcnt = np.zeros((nt, P))
    tt = tsk_idx[tr]
    for j in range(len(tr)):
        tpq[tt[j], p[j]] += y[j]
        tcnt[tt[j], p[j]] += 1
    tpq /= np.maximum(tcnt, 1)[:, :, None]
    return dict(glob=glob, pq=pq, off=off, tpq=tpq, tcnt=tcnt)


def prior_scores(pri, kind, p, tsk_idx, P, alpha=0.0):
    """Развернуть таблицу prior в score[строка, позиция]."""
    n = len(p)
    if kind == "glob":
        return np.tile(pri["glob"], (n, 1))
    if kind == "pq":
        return pri["pq"][p]
    if kind == "off":
        q = np.arange(P)[None, :]
        return pri["off"][q - p[:, None] + P - 1]
    if kind == "task":
        # СГЛАЖИВАНИЕ. На задачу приходится порядка 17 строк с данным p, оценка
        # шумная; смесь с глобальной таблицей и вес alpha выбираются на val.
        return alpha * pri["tpq"][tsk_idx, p] + (1 - alpha) * pri["pq"][p]
    raise ValueError(kind)


def report(name, num, den, epi, tsk, out=None):
    pt, lo, hi = cluster_ci(num, den, epi)
    mac = macro_by(num, den, tsk)
    print(f"  {name:<34} {pt:.3f} [{lo:.3f}, {hi:.3f}]   macro {mac:.3f}")
    if out is not None:
        out[name] = dict(R=pt, lo=lo, hi=hi, macro=mac)
    return pt


def decomposition(tables, lb, e0, idx, ent, K, rng, n_rand=20):
    """Пять величин, разделяющих «найти support» и «упорядочить внутри него».

    1. random-16      — случайные K из всех 16;
    2. random-in-C    — случайные K внутри истинного support;
    3. entropy-in-C   — идеальный support + причинное ранжирование внутри;
    4. true-in-C      — истинные gains, но ПРИНУДИТЕЛЬНО K внутри support;
    5. O1             — истинные gains по всем 16 (это и есть 0.701).

    При |C| < K добираем позициями ВНЕ support: они дают ровно ноль, поэтому
    стоимость выровнена, а выигрыш не искажён.
    """
    P = ent.shape[1]
    sg = lb["sing_gain_rms"]
    res = {}

    def fill(base, C_sorted, all_pos):
        """Добить набор до K нейтральными позициями вне support."""
        need = K - len(base)
        if need <= 0:
            return base[:K]
        extra = [q for q in all_pos if q not in C_sorted][:need]
        return list(base) + extra

    for nm in ("random-16", "random-in-C", "entropy-in-C", "true-in-C", "O1"):
        acc = []
        reps = n_rand if nm.startswith("random") else 1
        for rep in range(reps):
            r = np.random.default_rng(1000 + rep)
            sets = []
            for j, i in enumerate(idx):
                _, C = tables[j]
                Cs = sorted(C)
                allp = list(range(P))
                if nm == "random-16":
                    sets.append(list(r.permutation(P)[:K]))
                elif nm == "random-in-C":
                    pick = list(r.permutation(Cs)[:K]) if Cs else []
                    sets.append(fill(pick, C, allp))
                elif nm == "entropy-in-C":
                    o = [q for q in np.argsort(-ent[i], kind="stable")
                         if q in C][:K]
                    sets.append(fill(o, C, allp))
                elif nm == "true-in-C":
                    o = [q for q in np.argsort(-sg[i], kind="stable")
                         if q in C][:K]
                    sets.append(fill(o, C, allp))
                else:
                    sets.append(list(np.argsort(-sg[i], kind="stable")[:K]))
            acc.append(eval_sets(tables, e0, idx, sets))
        res[nm] = (np.mean([a[0] for a in acc], 0), acc[0][1])
    return res


def selftest():
    """Синтетика с ИЗВЕСТНЫМ ОТВЕТОМ для машинки отбора и подсчёта.

    Строим одну строку, где support = {1, 3}, выигрыши заданы руками, и
    проверяем, что: top-K по score выбирает то, что должен; forced-политика
    действительно ставит p первым; добивка нейтральными не меняет G; величина
    true-in-C не выше O1 при вредной позиции внутри support.
    """
    P, K = 8, 4
    score = np.zeros((1, P))
    score[0, [5, 2, 7]] = [3.0, 2.0, 1.0]
    assert list(topk_sets(score, 3)[0]) == [5, 2, 7], "top-K сломан"
    assert list(topk_sets(score, 3, forced=np.array([4]))[0]) == [4, 5, 2], \
        "forced не ставит p первым"
    # ничьи: все нули -> должны идти подряд по индексу
    assert list(topk_sets(np.zeros((1, P)), 3)[0]) == [0, 1, 2], "ничьи неустойчивы"

    # support {1,3}: позиция 1 полезна, позиция 3 ВРЕДНА.
    gmap = {(): 0.0, (1,): 0.5, (3,): -0.2, (1, 3): 0.35}
    C = {1, 3}
    e0 = 1.0
    o1 = to_rms(e0, g_of(gmap, C, [1]))          # берёт 1, остальные нули
    tin = to_rms(e0, g_of(gmap, C, [1, 3]))      # принудительно оба из C
    assert tin < o1, "true-in-C обязан быть не выше O1 при вредной позиции"
    assert abs(to_rms(e0, g_of(gmap, C, [1, 0, 2])) - o1) < 1e-12, \
        "добивка позициями вне support изменила выигрыш"
    print("самопроверка пройдена: отбор, ничьи, forced, добивка, асимметрия O1")


def make_fake(d, P=8, kmax=4, n_obs=40, seed=0):
    """Крошечный датасет ТОЙ ЖЕ СХЕМЫ для дымового прогона.

    Нужен потому, что ошибки в именах ключей и формах иначе всплывают только на
    кластере после часа чтения. Числа бессмысленны, проверяется проходимость."""
    import itertools
    r = np.random.default_rng(seed)
    os.makedirs(d, exist_ok=True)
    n = n_obs * P
    obs = np.repeat(np.arange(n_obs), P)
    p = np.tile(np.arange(P), n_obs)
    ep = obs // 4
    split = np.where(ep % 5 == 3, 1, np.where(ep % 5 == 4, 2, 0)).astype(np.int8)
    e0 = r.uniform(0.5, 2.0, n)
    supp = np.array([int(r.integers(1, 1 << P)) for _ in range(n)], np.int64)
    g_flat, g_off, sg = [], [0], np.zeros((n, P), np.float32)
    for i in range(n):
        C = tuple(q for q in range(P) if supp[i] >> q & 1)
        w = r.normal(0.15, 0.2, len(C))
        for k in range(kmax + 1):
            for S in itertools.combinations(range(len(C)), k):
                g_flat.append(float(sum(w[j] for j in S)) * e0[i] * 0.4)
        g_off.append(len(g_flat))
        for j, q in enumerate(C):
            sg[i, q] = to_rms(e0[i], w[j] * e0[i] * 0.4)
    np.savez(os.path.join(d, "features.npz"),
             cand_entropy=r.uniform(0, 3, (n, P)).astype(np.float32),
             cand_margin=r.uniform(0, 1, (n, P)).astype(np.float32),
             obs_task_idx=(np.arange(n_obs) % 4).astype(np.int64),
             int_obs_idx=obs, int_p=p)
    np.savez(os.path.join(d, "labels.npz"), obs_idx=obs, p=p, episode=ep,
             split=split, e_empty=e0, support=supp,
             g_flat=np.array(g_flat, np.float32),
             g_off=np.array(g_off, np.int64), sing_gain_rms=sg,
             sing_gain_norm=sg / np.maximum(sg.max(1, keepdims=True), 1e-6),
             best_size_by_k=np.full((n, kmax + 1), kmax, np.int8))
    json.dump(dict(P=P, kmax=kmax, commit="fake",
                   tasks=[f"t{i}" for i in range(4)]),
              open(os.path.join(d, "metadata.json"), "w"))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="прогон на поддельном датасете той же схемы")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if args.smoke:
        import tempfile
        args.dir = make_fake(os.path.join(tempfile.mkdtemp(), "fake"))
        print(f"дымовой прогон на поддельном датасете {args.dir}\n")
    if not args.dir:
        raise SystemExit("нужен --dir, --smoke или --selftest")

    meta = json.load(open(os.path.join(args.dir, "metadata.json")))
    _ft = np.load(os.path.join(args.dir, "features.npz"), allow_pickle=True)
    _lb = np.load(os.path.join(args.dir, "labels.npz"), allow_pickle=True)
    ft = {k: _ft[k] for k in _ft.files}      # материализуем: NpzFile ленив
    lb = {k: _lb[k] for k in _lb.files}
    P, kmax = meta["P"], meta["kmax"]
    n = len(lb["obs_idx"])

    epi, sp = lb["episode"], lb["split"]
    e0 = lb["e_empty"].astype(np.float64)
    p_all = lb["p"].astype(np.int64)
    tsk_idx = ft["obs_task_idx"][lb["obs_idx"]].astype(np.int64)
    tsk = np.asarray(meta["tasks"])[tsk_idx]
    ent, mrg = ft["cand_entropy"], ft["cand_margin"]
    tr = np.where(sp == 0)[0]

    print("=" * 74)
    print(f"K-4b0.1 BASELINE РАВНОЙ СТОИМОСТИ, строк {n}, commit "
          f"{meta.get('commit', '?')}")
    print("=" * 74)

    pri = fit_priors(lb, ft, tsk_idx, tr, P)
    print(f"  таблицы prior обучены по train: {len(tr)} строк")

    zent, zmrg = zrow(ent.astype(np.float64)), zrow(-mrg.astype(np.float64))
    is_p = (np.arange(P)[None, :] == p_all[:, None]).astype(np.float64)

    def policies(lam, alpha):
        """Все причинные политики как score[строка, позиция] плюс forced."""
        return {
            "энтропия": (zent, None, "heur"),
            "малый запас top1-top2": (zmrg, None, "heur"),
            "окно вокруг p": (-np.abs(np.arange(P)[None, :] - p_all[:, None])
                              .astype(np.float64), None, "heur"),
            "p + энтропия": (zent, p_all, "heur"),
            "p + малый запас": (zmrg, p_all, "heur"),
            "p + окно": (-np.abs(np.arange(P)[None, :] - p_all[:, None])
                         .astype(np.float64), p_all, "heur"),
            f"энтропия + {lam}*[q=p]": (zent + lam * is_p, None, "heur"),
            f"запас + {lam}*[q=p]": (zmrg + lam * is_p, None, "heur"),
            "prior[q]": (prior_scores(pri, "glob", p_all, tsk_idx, P),
                         None, "prior"),
            "prior[p,q]": (prior_scores(pri, "pq", p_all, tsk_idx, P),
                           None, "prior"),
            "prior[q-p]": (prior_scores(pri, "off", p_all, tsk_idx, P),
                           None, "prior"),
            f"prior[task,p,q] a={alpha}": (
                prior_scores(pri, "task", p_all, tsk_idx, P, alpha),
                None, "prior"),
        }

    tables, idx_of = {}, {}
    for s_, nm in ((1, "val"), (2, "test")):
        idx_of[s_] = np.where(sp == s_)[0]
        print(f"\n  строю таблицы G для {nm} ({len(idx_of[s_])} строк)")
        tables[s_] = build_tables(lb, idx_of[s_], P, kmax)

    # ---- ВЫБОР ГИПЕРПАРАМЕТРОВ ТОЛЬКО НА VALIDATION -------------------------
    print("\n" + "=" * 74)
    print("ПОДБОР lambda И alpha НА VALIDATION (test не используется)")
    print("=" * 74)
    iv = idx_of[1]
    best_lam, best_lam_R = None, -1.0
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        sc = zent + lam * is_p
        sets = topk_sets(sc[iv], kmax)
        num, den = eval_sets(tables[1], e0, iv, sets)
        R = num.sum() / den.sum()
        print(f"    lambda={lam:<5} энтропия+индикатор  R={R:.4f}")
        if R > best_lam_R:
            best_lam, best_lam_R = lam, R
    best_a, best_a_R = None, -1.0
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        sc = prior_scores(pri, "task", p_all, tsk_idx, P, alpha)
        sets = topk_sets(sc[iv], kmax)
        num, den = eval_sets(tables[1], e0, iv, sets)
        R = num.sum() / den.sum()
        print(f"    alpha={alpha:<5} task-prior          R={R:.4f}")
        if R > best_a_R:
            best_a, best_a_R = alpha, R
    print(f"  выбрано: lambda={best_lam}, alpha={best_a}")

    pol = policies(best_lam, best_a)
    res = {}
    for s_, nm in ((1, "val"), (2, "test")):
        ii = idx_of[s_]
        print("\n" + "=" * 74)
        print(f"ПОЛИТИКИ РАВНОЙ СТОИМОСТИ, {nm}: строк {len(ii)}, эпизодов "
              f"{len(np.unique(epi[ii]))}")
        print("=" * 74)
        for K in (1, 2, kmax):
            print(f"\n  K = {K}")
            seen = set()
            for name, (sc, forced, kind) in pol.items():
                # ПРИ K=1 все гибриды «p + признак» вырождаются в одну и ту же
                # политику «только p». Печатать её трижды и трижды класть в res
                # под одним ключом нельзя: результаты молча затрут друг друга.
                if forced is not None and K == 1:
                    name_, sets = "только p", forced[ii][:, None]
                else:
                    name_ = name
                    sets = topk_sets(sc[ii], K,
                                     None if forced is None else forced[ii])
                if name_ in seen:
                    continue
                seen.add(name_)
                num, den = eval_sets(tables[s_], e0, ii, sets)
                report(name_, num, den, epi[ii], tsk[ii], res)
                res[f"{nm}/K{K}/{name_}"] = res.pop(name_)
                res[f"{nm}/K{K}/{name_}"]["kind"] = kind

    # ---- ФИКСАЦИЯ B_heur И B_prior ПО VALIDATION ---------------------------
    heur = {k: v for k, v in res.items()
            if k.startswith("val/K4/") and v.get("kind") == "heur"}
    prior = {k: v for k, v in res.items()
             if k.startswith("val/K4/") and v.get("kind") == "prior"}
    bh = max(heur, key=lambda k: heur[k]["R"])
    bp = max(prior, key=lambda k: prior[k]["R"])
    print("\n" + "=" * 74)
    print("ЗАФИКСИРОВАНО ПО VALIDATION")
    print("=" * 74)
    for lbl, k in (("B_heur ", bh), ("B_prior", bp)):
        nm_ = k.split("/")[-1]
        tk = f"test/K4/{nm_}"
        print(f"  {lbl} = {nm_:<28} val {res[k]['R']:.3f}   "
              f"test {res[tk]['R']:.3f} [{res[tk]['lo']:.3f}, "
              f"{res[tk]['hi']:.3f}]")
    print("  ворота B1: R >= 0.60 И R - B_prior >= 0.05 И нижняя граница")
    print("  парного ДИ выше нуля, воспроизводимо на трёх сидах обучения.")

    # ---- РАЗЛОЖЕНИЕ РАЗРЫВА -----------------------------------------------
    print("\n" + "=" * 74)
    print("РАЗЛОЖЕНИЕ РАЗРЫВА: «найти support» против «упорядочить внутри»")
    print("=" * 74)
    for s_, nm in ((2, "test"),):
        ii = idx_of[s_]
        dec = decomposition(tables[s_], lb, e0, ii, ent, kmax,
                            np.random.default_rng(0))
        print(f"\n  {nm}, K={kmax}")
        for k, (num, den) in dec.items():
            report(k, num, den, epi[ii], tsk[ii], res)
            res[f"{nm}/decomp/{k}"] = res.pop(k)
        a = res[f"{nm}/decomp/random-16"]["R"]
        b = res[f"{nm}/decomp/random-in-C"]["R"]
        c = res[f"{nm}/decomp/entropy-in-C"]["R"]
        d = res[f"{nm}/decomp/O1"]["R"]
        print(f"\n  знание support даёт          {b - a:+.3f}")
        print(f"  причинный порядок внутри     {c - b:+.3f}")
        print(f"  остаток до одиночного оракула{d - c:+.3f}")
        print("  ЧИТАТЬ ТАК: если первое слагаемое доминирует, router надо\n"
              "  строить и мерить прежде всего как классификатор support;\n"
              "  если второе — центр тяжести в знаке и величине выигрыша.")

    if args.out:
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
